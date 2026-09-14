import html
import json

import gradio as gr
from pydantic import ValidationError

from analyzer import analyze_job
from llm_client import (
    ModelOutputTruncatedError,
    ModelRequestError,
    ModelResponseError,
)


def render_analysis(decision):
    result = decision.model_dump(mode="json")
    recommendation = result["recommendation"]
    tone = {
        "apply": "positive",
        "human_review": "warning",
        "skip": "negative",
    }[recommendation]
    label = {
        "apply": "Apply",
        "human_review": "Human review",
        "skip": "Skip",
    }[recommendation]

    tags = "".join(
        f'<span class="activity-tag">{html.escape(tag)}</span>'
        for tag in result["activity_tags"]
    )
    summary = f"""
    <section class="result-summary">
      <div>
        <span class="eyebrow">Recommendation</span>
        <h2 class="{tone}">{label}</h2>
      </div>
      <div>
        <span class="eyebrow">Location eligibility</span>
        <strong>{result["candidate_location_eligible"]}</strong>
      </div>
      <div>
        <span class="eyebrow">Work authorization</span>
        <strong>{result["work_authorization_eligible"]}</strong>
      </div>
    </section>
    <section class="activity-tags">{tags}</section>
    """

    responsibilities = "\n".join(
        f'- “{item["quote"]}”  \n  Activities: {", ".join(item["activities"])}'
        for item in result["facts"]["responsibilities"]
    )
    risks = "\n".join(
        f"- {risk}" for risk in result["facts"]["risks"]
    ) or "- No explicit risks returned."
    return (
        summary,
        "### Source-backed responsibilities\n" + responsibilities,
        "### Risks\n" + risks,
        json.dumps(result, ensure_ascii=False, indent=2),
    )


def error_outputs(message):
    escaped = html.escape(message)
    return f'<div class="analysis-error">{escaped}</div>', "", "", ""


def analyze_jd(jd_text, max_tokens):
    if not jd_text or not jd_text.strip():
        return error_outputs("请输入一段 JD 后再分析。")
    try:
        return render_analysis(
            analyze_job(jd_text.strip(), max_tokens=int(max_tokens))
        )
    except ValidationError as exc:
        return error_outputs(f"结构化结果未通过校验：{exc}")
    except ModelOutputTruncatedError as exc:
        return error_outputs(str(exc))
    except (ModelRequestError, ModelResponseError) as exc:
        return error_outputs(str(exc))
    except Exception as exc:
        return error_outputs(f"分析失败：{exc}")


CSS = """
.gradio-container { max-width: 1180px !important; }
.header { margin: 12px 0 22px; }
.header h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: 0; }
.header p { color: #65706b; margin: 0; }
.result-summary {
  display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 16px;
  padding: 18px; border: 1px solid #dfe4e1; border-radius: 8px;
}
.result-summary > div { min-width: 0; }
.result-summary h2 { font-size: 24px; margin: 4px 0 0; letter-spacing: 0; }
.eyebrow { display: block; color: #65706b; font-size: 12px; text-transform: uppercase; }
.positive { color: #18794e; }
.warning { color: #9a6700; }
.negative { color: #b42318; }
.activity-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.activity-tag {
  padding: 5px 8px; border: 1px solid #ccd5d0; border-radius: 5px;
  background: #f6f8f7; font-size: 13px;
}
.analysis-error {
  padding: 14px; border-left: 4px solid #b42318; background: #fff5f4;
  color: #8a1c13;
}
"""

with gr.Blocks(title="AI Job Fit Analyzer") as demo:
    gr.HTML(
        '<header class="header"><h1>AI Job Fit Analyzer</h1>'
        "<p>Source-backed responsibilities, candidate constraints, and a reproducible decision.</p></header>"
    )
    with gr.Row():
        with gr.Column(scale=5):
            jd_input = gr.Textbox(
                label="Job description",
                placeholder="Paste a complete JD here",
                lines=24,
            )
            max_tokens = gr.Slider(
                400, 2000, value=1200, step=100, label="Maximum output tokens"
            )
            with gr.Row():
                gr.ClearButton([jd_input], value="Clear", variant="secondary")
                analyze_button = gr.Button("Analyze role", variant="primary")
        with gr.Column(scale=7):
            summary = gr.HTML(
                '<div class="analysis-error">Run an analysis to inspect the evidence.</div>'
            )
            responsibilities = gr.Markdown()
            risks = gr.Markdown()
            raw_json = gr.Code(label="Validated result", language="json")

    analyze_button.click(
        analyze_jd,
        inputs=[jd_input, max_tokens],
        outputs=[summary, responsibilities, risks, raw_json],
    )


if __name__ == "__main__":
    demo.launch(css=CSS)
