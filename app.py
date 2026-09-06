import json
import html
import os
import re
import time
import urllib.error
import urllib.request

import gradio as gr


ROLE_TYPES = {
    "Product-facing AIE",
    "Internal-facing AIE",
    "FDE",
    "Research/ML role",
    "ML-heavy AIE",
    "platform-heavy AIE",
    "not fit",
}
RECOMMENDATIONS = {"apply", "maybe", "skip", "human_review"}
EVAL_CASES_PATH = os.getenv(
    "EVAL_CASES_PATH",
    os.path.join(os.path.dirname(__file__), "data", "eval_jds.example.jsonl"),
)

SYSTEM_PROMPT = """You are an AI job fit analysis assistant.
Return ONLY valid JSON. Do not include markdown.

User goal:
- Target: Product-facing Applied AI Engineer or FDE.
- Prefer AI application and product/customer-facing workflow.
- H-1B transfer support is required or must be explicitly marked unknown.
- Avoid pure infrastructure, backend maintenance, and research-only roles.

Decision rules:
1. Classify the role by its primary day-to-day responsibilities, not by the
   employer's product, industry, growth, or use of the word "AI".
   First identify the role's primary deliverable. Classify from the deliverable,
   not from domain knowledge listed under qualifications or preferred skills.
2. Set product_facing=true only when the role explicitly works with users,
   customers, domain experts, or product teams to discover needs, design a
   workflow, collect feedback, or iterate on user outcomes. Ownership,
   autonomy, remote work, building a platform, and merely building internal
   tools are not product-facing evidence by themselves.
3. Use platform-heavy AIE when the core work is GPU scheduling, workload
   orchestration, infrastructure, drivers, cluster networking, benchmarking,
   reliability, or on-call operations. For this role type,
   avoid_pure_infra=false and the recommendation cannot be apply.
   Do not use platform-heavy AIE merely because the domain is silicon,
   hardware, firmware, drivers, validation, or engineering infrastructure.
   When the primary deliverables are LLM-powered workflows, AI agents,
   intelligent automation, or production AI applications for internal teams,
   classify the role as Internal-facing AIE and set avoid_pure_infra=true.
   Platform-heavy AIE applies only when operating the underlying compute or
   service platform is itself the primary deliverable. Building an AI system
   that improves an internal silicon, validation, or engineering workflow is
   not platform-heavy work.
4. Use ML-heavy AIE for model training, fine-tuning, model architecture,
   inference optimization, or benchmark quality when platform operations are
   not the primary responsibility and the work is applied engineering. Use
   Research/ML role instead when the primary work is research, pretraining
   experiments, publishing papers, or developing model architecture without
   application delivery. For Research/ML role, set ai_application=false and
   product_facing=false.
5. Use Product-facing AIE only when both AI application delivery and direct
   user/product workflow collaboration are explicit.
6. Use FDE when the title is Forward Deployed Engineer, or when the primary
   work is deploying or prototyping solutions with named customers, adapting
   integrations to customer workflows, and feeding field learning back into
   the product. Do not relabel an explicit FDE as Product-facing AIE merely
   because both are customer-facing.
7. Use Internal-facing AIE when the role builds AI applications for employees
   or internal workflows but has no explicit direct user, customer,
   domain-expert, or product-team collaboration. Keep product_facing=false.
   Use not fit only when none of the defined AI role families describes the
   primary work.
8. For location and H-1B transfer, absence of information means "unknown",
   never "no". Use "no" only when the JD explicitly rules out the location,
   remote option, sponsorship, or transfer support.
9. Tie-breaker: if a role both requires infrastructure or hardware expertise
   and explicitly owns an LLM workflow, AI agent, or intelligent automation
   system from prototype through production, classify it as Internal-facing
   AIE unless its primary duties are operating clusters, serving systems,
   networking, storage, reliability, or on-call infrastructure.

Boundary example:
- A role that builds LLM-powered validation pipelines and AI agents for
  internal chip-engineering teams is Internal-facing AIE, with
  ai_application=true, product_facing=false, and avoid_pure_infra=true. Silicon,
  lab-debug, firmware, and driver experience are domain requirements, not proof
  that the role operates infrastructure.

Evidence and risk rules:
- Evidence must cover every hard-filter value that materially affects the
  recommendation, including explicit location/remote and H-1B support.
- Risks must be grounded in an explicit negative statement or a required fact
  that is absent from the JD. Do not invent possible infrastructure, platform,
  backend, location, or visa concerns when the JD provides no such signal.
- If required information is missing, use "unknown". Set needs_human_review to
  true only when the missing fact could change an otherwise viable role. A role
  already disqualified as platform-heavy, research-only, or not fit should be
  skipped without human review.
- fit_score measures how well the role fits the user's target. confidence
  measures how certain the classification and extracted facts are. A clearly
  disqualified role may have fit_score=0 and high confidence. Never copy
  fit_score into confidence or lower confidence merely because the fit is poor.
"""

USER_PROMPT_TEMPLATE = """Analyze this job description for fit.

Return this JSON shape:
{{
  "role_type": "Product-facing AIE | Internal-facing AIE | FDE | Research/ML role | ML-heavy AIE | platform-heavy AIE | not fit",
  "fit_score": 0,
  "hard_filter_result": {{
    "ai_application": true,
    "product_facing": true,
    "bay_area_or_remote": "yes | no | unknown",
    "h1b_transfer_signal": "yes | no | unknown",
    "avoid_pure_infra": true
  }},
  "evidence": ["JD evidence"],
  "risks": ["risk"],
  "confidence": 0.0,
  "needs_human_review": true,
  "recommendation": "apply | maybe | skip | human_review"
}}

JD:
{job_description}

/no_think
"""


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_model_json(data):
    errors = []
    if data.get("role_type") not in ROLE_TYPES:
        errors.append("role_type 不在允许范围内")
    if data.get("recommendation") not in RECOMMENDATIONS:
        errors.append("recommendation 不在允许范围内")
    if not isinstance(data.get("fit_score"), (int, float)):
        errors.append("fit_score 不是数字")
    if not isinstance(data.get("hard_filter_result"), dict):
        errors.append("缺少 hard_filter_result")
    if not data.get("evidence"):
        errors.append("缺少 JD 证据")
    if not isinstance(data.get("needs_human_review"), bool):
        errors.append("needs_human_review 不是布尔值")
    fit_score = data.get("fit_score")
    if isinstance(fit_score, (int, float)) and not 0 <= fit_score <= 100:
        errors.append("fit_score 必须在 0 到 100 之间")
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence <= 1:
        errors.append("confidence 必须大于 0 且不超过 1")

    hard_filters = data.get("hard_filter_result")
    if isinstance(hard_filters, dict):
        required_filters = {
            "ai_application": bool,
            "product_facing": bool,
            "bay_area_or_remote": str,
            "h1b_transfer_signal": str,
            "avoid_pure_infra": bool,
        }
        for field, expected_type in required_filters.items():
            if not isinstance(hard_filters.get(field), expected_type):
                errors.append(f"hard_filter_result.{field} 类型错误或缺失")
        if hard_filters.get("h1b_transfer_signal") not in {"yes", "no", "unknown"}:
            errors.append("h1b_transfer_signal 不在允许范围内")
        if hard_filters.get("bay_area_or_remote") not in {"yes", "no", "unknown"}:
            errors.append("bay_area_or_remote 不在允许范围内")
        if (
            data.get("role_type") == "platform-heavy AIE"
            and hard_filters.get("avoid_pure_infra") is not False
        ):
            errors.append("platform-heavy AIE 与 avoid_pure_infra 的判断矛盾")
        if (
            data.get("role_type") == "platform-heavy AIE"
            and hard_filters.get("product_facing") is True
        ):
            errors.append("platform-heavy AIE 与 product_facing 的判断矛盾")
        if data.get("role_type") == "Research/ML role":
            if hard_filters.get("ai_application") is not False:
                errors.append("Research/ML role 与 ai_application 的判断矛盾")
            if hard_filters.get("product_facing") is not False:
                errors.append("Research/ML role 与 product_facing 的判断矛盾")
    if data.get("recommendation") == "skip" and data.get("needs_human_review") is True:
        errors.append("skip 与 needs_human_review=true 的终态矛盾")
    return errors


def extract_h1b_signal(job_description):
    text = job_description.lower()
    negative_phrases = (
        "without current or future sponsorship",
        "without sponsorship",
        "no visa sponsorship",
        "does not sponsor",
        "do not sponsor",
        "cannot sponsor",
        "unable to sponsor",
        "will not sponsor",
        "not eligible for sponsorship",
        "h-1b transfer not supported",
        "h1b transfer not supported",
    )
    if any(phrase in text for phrase in negative_phrases):
        return "no"
    positive_phrases = (
        "visa sponsorship available",
        "visa sponsorship is available",
        "sponsorship and h-1b transfer support available",
        "h-1b transfer support available",
        "h1b transfer support available",
    )
    if any(phrase in text for phrase in positive_phrases):
        return "yes"
    return "unknown"


def apply_business_rules(data, job_description=None):
    hard_filters = data.get("hard_filter_result") or {}
    if job_description is not None:
        hard_filters["h1b_transfer_signal"] = extract_h1b_signal(job_description)
    if data.get("role_type") == "platform-heavy AIE":
        data["needs_human_review"] = False
        data["recommendation"] = "skip"
        return data
    if data.get("role_type") == "Research/ML role":
        data["needs_human_review"] = False
        data["recommendation"] = "skip"
        return data

    if (
        hard_filters.get("h1b_transfer_signal") == "unknown"
        and data.get("recommendation") != "skip"
    ):
        data["needs_human_review"] = True
        data["recommendation"] = "human_review"
    if data.get("recommendation") == "skip":
        data["needs_human_review"] = False
    return data


def call_llm(job_description, system_prompt=SYSTEM_PROMPT, temperature=0.1, max_tokens=500):
    base_url = os.getenv("RUNPOD_BASE_URL", "").rstrip("/")
    api_key = os.getenv("RUNPOD_API_KEY", "")
    model = os.getenv("RUNPOD_MODEL", "Qwen/Qwen3-8B")

    if not base_url:
        raise RuntimeError("缺少 RUNPOD_BASE_URL，请先配置或重新启动 RunPod endpoint。")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(job_description=job_description),
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 502 and "Waiting for service to respond" in detail:
            raise RuntimeError(
                "RunPod 已连接，但模型服务尚未开始响应。请等待 vLLM 完成加载后重试。"
            ) from exc
        if exc.code == 400 and "maximum context length" in detail:
            raise RuntimeError(
                "输入内容与输出预算超过模型上下文上限。请缩短 JD 或 system prompt，"
                "并把 Max output tokens 调低后重试。"
            ) from exc
        raise RuntimeError(f"模型服务返回 HTTP {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接模型服务: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("模型服务超过 90 秒仍未响应。") from exc

    try:
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if content:
            return content
        if reasoning:
            return reasoning
        raise RuntimeError("模型响应中 content 和 reasoning_content 都为空。")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型服务返回了非预期的响应结构。") from exc


def render_analysis(result):
    hard_filters = result["hard_filter_result"]
    recommendation = result["recommendation"]
    recommendation_labels = {
        "apply": ("Apply", "positive"),
        "maybe": ("Maybe", "warning"),
        "human_review": ("Review", "warning"),
        "skip": ("Skip", "negative"),
    }
    recommendation_label, recommendation_tone = recommendation_labels[recommendation]

    def status_badge(label, value, positive_value=True):
        display = {True: "Yes", False: "No", "yes": "Yes", "no": "No", "unknown": "Unknown"}.get(value, value)
        if value == "unknown":
            tone = "warning"
        elif value == positive_value:
            tone = "positive"
        else:
            tone = "negative"
        return (
            f'<div class="signal"><span>{html.escape(label)}</span>'
            f'<strong class="{tone}">{html.escape(str(display))}</strong></div>'
        )

    signals = "".join(
        [
            status_badge("AI application", hard_filters["ai_application"]),
            status_badge("Product-facing", hard_filters["product_facing"]),
            status_badge("Bay Area / remote", hard_filters["bay_area_or_remote"], "yes"),
            status_badge("H-1B transfer", hard_filters["h1b_transfer_signal"], "yes"),
            status_badge("Avoids pure infra", hard_filters["avoid_pure_infra"]),
        ]
    )
    summary = f"""
    <section class="result-summary">
      <div class="role-block">
        <span class="eyebrow">Role type</span>
        <h2>{html.escape(result['role_type'])}</h2>
      </div>
      <div class="score-block">
        <span class="eyebrow">Fit score</span>
        <strong>{result['fit_score']}<small>/100</small></strong>
      </div>
      <div class="recommendation {recommendation_tone}">
        <span class="eyebrow">Recommendation</span>
        <strong>{recommendation_label}</strong>
      </div>
    </section>
    <section class="signals">{signals}</section>
    <div class="confidence-line">Confidence: {result['confidence']:.0%}</div>
    """
    evidence_items = "\n".join(f"- {item}" for item in result.get("evidence", [])) or "- No evidence returned"
    risk_items = "\n".join(f"- {item}" for item in result.get("risks", [])) or "- No material risks found in this JD"
    evidence = f"### Evidence\n{evidence_items}"
    risks = f"### Risks\n{risk_items}"
    raw_json = json.dumps(result, ensure_ascii=False, indent=2)
    return summary, evidence, risks, raw_json, gr.update(visible=True)


def error_outputs(message):
    return (
        f'<div class="analysis-error">{html.escape(message)}</div>',
        "",
        "",
        "",
        gr.update(visible=False),
    )


def analyze_jd(jd_text, system_prompt=SYSTEM_PROMPT, temperature=0.1, max_tokens=500):
    if not jd_text or not jd_text.strip():
        return error_outputs("请输入一段 JD 后再分析。")

    try:
        model_text = call_llm(
            jd_text.strip(),
            system_prompt=system_prompt or SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = extract_json(model_text)
        result = apply_business_rules(result, jd_text.strip())
        validation_errors = validate_model_json(result)
        if validation_errors:
            return error_outputs("模型返回未通过校验：" + "；".join(validation_errors))
        return render_analysis(result)
    except json.JSONDecodeError:
        preview = model_text.strip()[:500] if "model_text" in locals() else ""
        message = "模型返回的内容不是有效 JSON，无法展示可信分析结果。"
        if preview:
            message += f"\n\n原始输出预览：\n{preview}"
        return error_outputs(message)
    except RuntimeError as exc:
        return error_outputs(f"调用失败：{exc}")
    except Exception as exc:
        return error_outputs(f"未预期错误：{type(exc).__name__}: {exc}")


EXPECTED_FIELDS = {
    "expected_recommendation": ("recommendation", None),
    "expected_role_type": ("role_type", None),
    "expected_ai_application": ("ai_application", "hard_filter_result"),
    "expected_product_facing": ("product_facing", "hard_filter_result"),
    "expected_bay_area_or_remote": ("bay_area_or_remote", "hard_filter_result"),
    "expected_h1b_transfer_signal": ("h1b_transfer_signal", "hard_filter_result"),
    "expected_avoid_pure_infra": ("avoid_pure_infra", "hard_filter_result"),
}


def compare_expected_fields(case, result):
    mismatches = []
    for expected_key, (result_key, parent_key) in EXPECTED_FIELDS.items():
        if expected_key not in case:
            continue
        actual_source = result.get(parent_key, {}) if parent_key else result
        actual = actual_source.get(result_key)
        expected = case[expected_key]
        if actual != expected:
            mismatches.append(f"{result_key}: expected {expected}, got {actual}")
    return mismatches


def evaluate_regression_cases(cases, analyze_case):
    rows = []
    passed = 0
    for case in cases:
        try:
            result = analyze_case(case)
            mismatches = compare_expected_fields(case, result)
            status = "PASS" if not mismatches else "FAIL"
            details = "; ".join(mismatches) if mismatches else "All expected fields matched"
        except Exception as exc:
            status = "ERROR"
            details = f"{type(exc).__name__}: {exc}"
        if status == "PASS":
            passed += 1
        rows.append({"jd_id": case["jd_id"], "status": status, "details": details})
    return {"passed": passed, "failed": len(cases) - passed, "rows": rows}


def load_eval_cases(path=EVAL_CASES_PATH):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_regression(system_prompt=SYSTEM_PROMPT, temperature=0.1, max_tokens=500, progress=gr.Progress()):
    cases = load_eval_cases()
    completed = 0
    started = time.monotonic()

    def analyze_case(case):
        nonlocal completed
        progress(completed / len(cases), desc=f"Running {case['jd_id']}")
        model_text = call_llm(
            case["job_description"],
            system_prompt=system_prompt or SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = apply_business_rules(extract_json(model_text), case["job_description"])
        errors = validate_model_json(result)
        if errors:
            raise ValueError("; ".join(errors))
        completed += 1
        return result

    report = evaluate_regression_cases(cases, analyze_case)
    progress(1, desc="Regression complete")
    elapsed = time.monotonic() - started
    lines = [
        f"### Regression: {report['passed']}/{len(cases)} passed",
        f"Completed in {elapsed:.1f}s. Failed: {report['failed']}.",
        "",
        "| Case | Status | Details |",
        "|---|---|---|",
    ]
    for row in report["rows"]:
        details = row["details"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['jd_id']} | **{row['status']}** | {details} |")
    return "\n".join(lines)


CSS = """
.gradio-container { max-width: 1600px !important; color: #17201d; padding-inline: 24px !important; }
.app-header { border-bottom: 1px solid #d8dedb; padding-bottom: 16px; margin-bottom: 22px; }
.app-header h1 { font-size: 28px !important; margin: 0 0 4px !important; letter-spacing: 0 !important; }
.app-header p { color: #5c6863; margin: 0; }
.workspace { gap: 28px; align-items: start; }
.input-panel, .result-panel { border: 1px solid #d8dedb; border-radius: 8px; padding: 24px; background: #fff; }
.result-summary { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(110px, .6fr) minmax(130px, .7fr); border-bottom: 1px solid #d8dedb; padding-bottom: 16px; gap: 16px; align-items: end; }
.eyebrow { display: block; color: #65716c; font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }
.role-block h2 { font-size: 24px; line-height: 1.2; margin: 0; letter-spacing: 0; }
.score-block > strong { font-size: 30px; line-height: 1; }
.score-block small { color: #65716c; font-size: 13px; margin-left: 2px; }
.recommendation { border-left: 4px solid currentColor; padding: 10px 12px; background: #f4f6f5; }
.recommendation strong { font-size: 20px; }
.positive { color: #177245 !important; }
.warning { color: #a45a00 !important; }
.negative { color: #b42318 !important; }
.signals { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 8px; margin-top: 16px; }
.signal { min-height: 64px; border: 1px solid #d8dedb; border-radius: 6px; padding: 10px; display: flex; flex-direction: column; justify-content: space-between; }
.signal span { color: #65716c; font-size: 12px; }
.signal strong { font-size: 15px; }
.confidence-line { color: #65716c; font-size: 12px; text-align: right; margin-top: 8px; }
.detail-row { margin-top: 24px; gap: 28px; }
.detail-panel { padding-top: 4px; }
.analysis-error { border-left: 4px solid #b42318; background: #fff1f0; color: #7a1b14; padding: 14px; }
#analyze-button { background: #17201d; border-color: #17201d; }
.regression-section { margin-top: 28px; padding: 24px 0 8px; }
.regression-section h2 { font-size: 20px !important; margin: 0 0 4px !important; }
.regression-section p { color: #65716c; margin: 0 0 14px; }
@media (max-width: 900px) {
  .workspace, .detail-row { flex-direction: column; }
  .result-summary { grid-template-columns: 1fr 1fr; }
  .role-block { grid-column: 1 / -1; }
  .signals { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

with gr.Blocks(title="AI Job Fit Analyzer") as demo:
    gr.HTML("""
    <header class="app-header">
      <h1>AI Job Fit Analyzer</h1>
      <p>Screen Applied AI roles by work content, constraints, and evidence.</p>
    </header>
    """)
    with gr.Row(elem_classes="workspace"):
        with gr.Column(scale=4, elem_classes="input-panel"):
            jd_input = gr.Textbox(lines=16, label="Job description", placeholder="Paste a JD here")
            with gr.Accordion("Model controls", open=False):
                system_prompt_input = gr.Textbox(
                    value=SYSTEM_PROMPT,
                    lines=12,
                    label="System prompt",
                )
                temperature_input = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.1,
                    step=0.05,
                    label="Temperature",
                )
                max_tokens_input = gr.Slider(
                    minimum=200,
                    maximum=700,
                    value=500,
                    step=100,
                    label="Max output tokens",
                )
            with gr.Row():
                clear_button = gr.ClearButton(value="Clear", components=[jd_input])
                analyze_button = gr.Button("Analyze role", variant="primary", elem_id="analyze-button")
        with gr.Column(scale=8, elem_classes="result-panel"):
            summary_output = gr.HTML('<div class="empty-state">Run an analysis to see the role fit.</div>')
            with gr.Row(visible=False, elem_classes="detail-row") as detail_row:
                evidence_output = gr.Markdown("", elem_classes="detail-panel")
                risks_output = gr.Markdown("", elem_classes="detail-panel")
            with gr.Accordion("Raw model output", open=False):
                raw_output = gr.Code(language="json", interactive=False, show_label=False)

    with gr.Column(elem_classes="regression-section"):
        gr.HTML("<h2>Regression suite</h2><p>Run every saved case against the current model controls.</p>")
        regression_button = gr.Button("Run regression suite")
        regression_output = gr.Markdown("Regression results will appear here.")

    analyze_button.click(
        fn=analyze_jd,
        inputs=[jd_input, system_prompt_input, temperature_input, max_tokens_input],
        outputs=[summary_output, evidence_output, risks_output, raw_output, detail_row],
        api_name="predict",
    )
    regression_button.click(
        fn=run_regression,
        inputs=[system_prompt_input, temperature_input, max_tokens_input],
        outputs=[regression_output],
        api_name="run_regression",
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
