import json
import html
import os
import time

import gradio as gr
from pydantic import ValidationError

from analysis_service import analyze_job as _analyze_job
from business_rules import finalize_assessment
from evaluation import (
    compare_expected_fields,
    evaluate_regression_cases,
    format_markdown_report,
    load_eval_cases,
)
from errors import (
    ModelOutputTruncatedError,
    ModelRequestError,
    ModelResponseError,
)
from llm_client import (
    DEFAULT_TEMPERATURE,
    USER_PROMPT_TEMPLATE,
    call_llm as _call_llm,
    parse_model_assessment_json,
)
from models import (
    EvidenceField,
    EvidenceItem,
    HardFilterResult,
    JobAnalysis,
    ModelAssessment,
    ModelAssessmentV3,
    Recommendation,
    RoleType,
    TernarySignal,
    WorkAuthorizationResult,
)
from prompts import SYSTEM_PROMPT

def parse_job_analysis_json(model_text, job_description=None):
    return parse_model_assessment_json(model_text, job_description)


EVAL_CASES_PATH = os.getenv(
    "EVAL_CASES_PATH",
    os.path.join(os.path.dirname(__file__), "data", "eval_jds.example.jsonl"),
)

def call_llm(
    job_description,
    system_prompt=SYSTEM_PROMPT,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=500,
):
    return _call_llm(
        job_description,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def analyze_job(job_description, system_prompt=SYSTEM_PROMPT, max_tokens=500):
    return _analyze_job(
        job_description,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        max_tokens=max_tokens,
        call_model=call_llm,
    )


def render_analysis(result):
    hard_filters = result["hard_filter_result"]
    recommendation = result["recommendation"]
    recommendation_labels = {
        "apply": ("Apply", "positive"),
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
            status_badge(
                "Work authorization",
                hard_filters["work_authorization"]["status"],
                "yes",
            ),
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
    evidence_items = "\n".join(
        f"- **{item['field']}**: “{item['quote']}”"
        for item in result.get("evidence", [])
    ) or "- No evidence returned"
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


def analyze_jd(jd_text, system_prompt=SYSTEM_PROMPT, max_tokens=500):
    if not jd_text or not jd_text.strip():
        return error_outputs("请输入一段 JD 后再分析。")

    try:
        result = analyze_job(
            jd_text.strip(),
            system_prompt=system_prompt or SYSTEM_PROMPT,
            max_tokens=max_tokens,
        ).model_dump(mode="json")
        return render_analysis(result)
    except ValidationError as exc:
        error_count = exc.error_count()
        return error_outputs(
            f"模型返回未通过数据契约校验，共 {error_count} 处错误。"
        )
    except RuntimeError as exc:
        return error_outputs(f"调用失败：{exc}")
    except Exception as exc:
        return error_outputs(f"未预期错误：{type(exc).__name__}: {exc}")


def run_regression(system_prompt=SYSTEM_PROMPT, max_tokens=500, progress=gr.Progress()):
    cases = load_eval_cases(EVAL_CASES_PATH)
    completed = 0
    started = time.monotonic()

    def analyze_case(case):
        nonlocal completed
        progress(completed / len(cases), desc=f"Running {case['jd_id']}")
        result = analyze_job(
            case["job_description"],
            system_prompt=system_prompt or SYSTEM_PROMPT,
            max_tokens=max_tokens,
        ).model_dump(mode="json")
        completed += 1
        return result

    report = evaluate_regression_cases(cases, analyze_case)
    progress(1, desc="Regression complete")
    elapsed = time.monotonic() - started
    return format_markdown_report(report, len(cases), elapsed)


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
        inputs=[jd_input, system_prompt_input, max_tokens_input],
        outputs=[summary_output, evidence_output, risks_output, raw_output, detail_row],
        api_name="predict",
    )
    regression_button.click(
        fn=run_regression,
        inputs=[system_prompt_input, max_tokens_input],
        outputs=[regression_output],
        api_name="run_regression",
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
