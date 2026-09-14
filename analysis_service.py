import hashlib
import os
import time

from pydantic import ValidationError

from business_rules import finalize_assessment
from errors import JobFitError
from llm_client import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_RUNPOD_MODEL,
    DEFAULT_TEMPERATURE,
    call_llm,
)
from run_log import record_run


def _prompt_version(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


def _error_type(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "contract_error"
    if isinstance(exc, JobFitError):
        return exc.error_type
    return "unexpected_error"


def analyze_job(
    job_description,
    system_prompt,
    max_tokens=500,
    *,
    call_model=call_llm,
    log_path=None,
):
    started = time.monotonic()
    config = {
        "model": (
            os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
            if os.getenv("OPENROUTER_API_KEY")
            else os.getenv("RUNPOD_MODEL", DEFAULT_RUNPOD_MODEL)
        ),
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": int(max_tokens),
        "prompt_version": _prompt_version(system_prompt),
        "schema_version": "model-assessment-v2",
    }
    try:
        assessment = call_model(
            job_description,
            system_prompt=system_prompt,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=max_tokens,
        )
        analysis = finalize_assessment(assessment)
        record_run(
            {
                "status": "success",
                "job_description": job_description,
                "model_facts": assessment.model_dump(mode="json"),
                "final_result": analysis.model_dump(mode="json"),
                "config": config,
                "latency_seconds": round(time.monotonic() - started, 3),
            },
            path=log_path,
        )
        return analysis
    except Exception as exc:
        record_run(
            {
                "status": "error",
                "job_description": job_description,
                "error_type": _error_type(exc),
                "error_message": str(exc),
                "config": config,
                "latency_seconds": round(time.monotonic() - started, 3),
            },
            path=log_path,
        )
        raise
