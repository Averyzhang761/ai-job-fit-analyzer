import os
import re
import json
from copy import deepcopy
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI
from pydantic import BaseModel

from errors import ModelOutputTruncatedError, ModelRequestError, ModelResponseError
from models import ModelAssessment


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


DEFAULT_TEMPERATURE = 0.0
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_RUNPOD_MODEL = "Qwen/Qwen3-8B"
USER_PROMPT_TEMPLATE = """Analyze this job description for fit using the
classification, evidence, and risk rules in the system message.

Job description:
{job_description}
"""


def _status_error_detail(exc: APIStatusError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            metadata = error.get("metadata") or {}
            provider_detail = (
                metadata.get("raw") if isinstance(metadata, dict) else None
            )
            detail = provider_detail or error.get("message") or error.get("code")
        else:
            detail = error
    else:
        detail = body
    if not detail:
        return ""
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    return re.sub(r"\s+", " ", detail).strip()[:600]


def openai_compatible_json_schema(response_model: type[BaseModel]) -> dict:
    schema = response_model.model_json_schema()

    def resolve(ref: str):
        value = schema
        for part in ref.removeprefix("#/").split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[part]
        return deepcopy(value)

    def normalize(node):
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node and len(node) > 1:
            referenced = resolve(node["$ref"])
            referenced.update({key: value for key, value in node.items() if key != "$ref"})
            return normalize(referenced)
        return {key: normalize(value) for key, value in node.items()}

    return normalize(schema)


def build_evidence_chunks(job_description: str) -> tuple[str, dict[str, str]]:
    chunks: list[str] = []
    for raw_line in job_description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", line)
        chunks.extend(part.strip() for part in parts if part.strip())

    evidence_map = {
        f"E{index:03d}": chunk for index, chunk in enumerate(chunks, start=1)
    }
    annotated = "\n".join(
        f"[{evidence_id}] {quote}" for evidence_id, quote in evidence_map.items()
    )
    return annotated, evidence_map


def parse_model_assessment_json(
    model_text: str,
    job_description: str | None = None,
    evidence_map: dict[str, str] | None = None,
) -> ModelAssessment:
    context = {}
    if job_description:
        context["job_description"] = job_description
    if evidence_map is not None:
        context["evidence_map"] = evidence_map
    return ModelAssessment.model_validate_json(
        model_text,
        context=context or None,
    )


def call_structured_llm(
    job_description: str,
    system_prompt: str,
    response_model: type[StructuredModel],
    schema_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 500,
    task_context: str | None = None,
) -> StructuredModel:
    annotated_job_description, evidence_map = build_evidence_chunks(job_description)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_api_key:
        base_url = "https://openrouter.ai/api/v1"
        api_key = openrouter_api_key
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        provider_options = {"provider": {"require_parameters": True}}
    else:
        base_url = os.getenv("RUNPOD_BASE_URL", "").rstrip("/")
        api_key = os.getenv("RUNPOD_API_KEY", "")
        model = os.getenv("RUNPOD_MODEL", DEFAULT_RUNPOD_MODEL)
        provider_options = {"chat_template_kwargs": {"enable_thinking": False}}

    if not base_url:
        raise ModelRequestError(
            "缺少模型配置：请设置 OPENROUTER_API_KEY，或配置 RUNPOD_BASE_URL。"
        )

    api_base = base_url if base_url.endswith("/v1") else base_url + "/v1"
    user_content = USER_PROMPT_TEMPLATE.format(
        job_description=annotated_job_description
    )
    if task_context:
        user_content += "\nAdditional task context:\n" + task_context

    client = OpenAI(base_url=api_base, api_key=api_key or "EMPTY", timeout=90)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": openai_compatible_json_schema(response_model),
                    "strict": True,
                },
            },
            extra_body=provider_options,
        )
    except APITimeoutError as exc:
        raise ModelRequestError("模型服务超过 90 秒仍未响应。") from exc
    except APIConnectionError as exc:
        raise ModelRequestError("无法连接模型服务。") from exc
    except APIStatusError as exc:
        detail = _status_error_detail(exc)
        suffix = f"原因：{detail}" if detail else ""
        raise ModelRequestError(
            f"模型服务返回 HTTP {exc.status_code}。{suffix}"
        ) from exc

    if not response.choices:
        raise ModelResponseError("模型服务返回了空 choices。")
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ModelOutputTruncatedError(
            "模型输出因达到 max_tokens 被截断，未进入 JSON 解析。"
        )
    if choice.finish_reason != "stop":
        raise ModelResponseError(
            f"模型以未支持的原因结束：{choice.finish_reason or 'unknown'}。"
        )
    if not choice.message.content:
        raise ModelResponseError("模型响应中 content 为空。")

    return response_model.model_validate_json(
        choice.message.content,
        context={
            "job_description": job_description,
            "evidence_map": evidence_map,
        },
    )


def call_llm(
    job_description: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 500,
) -> ModelAssessment:
    return call_structured_llm(
        job_description,
        system_prompt,
        ModelAssessment,
        "job_analysis",
        temperature=temperature,
        max_tokens=max_tokens,
    )
