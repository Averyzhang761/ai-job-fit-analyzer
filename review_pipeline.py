"""Experimental extract-then-review pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json

from business_rules import finalize_assessment
from llm_client import call_structured_llm
from models import JobAnalysis, ModelAssessment
from prompts import SYSTEM_PROMPT


REVIEW_PROMPT = """Act as a factual reviewer, not a final decision maker.
Check every extracted fact against the source JD. Correct unsupported values,
missed explicit facts, and internally inconsistent classifications. Return the
complete corrected fact object. Use unknown when the source is insufficient.
Do not produce a score or recommendation. Evidence must use source chunk IDs
from the JD, never text from the candidate extraction."""


FACT_PATHS = (
    "role_type",
    "hard_filter_result.ai_application",
    "hard_filter_result.product_facing",
    "hard_filter_result.bay_area_or_remote",
    "hard_filter_result.work_authorization.sponsorship_available",
    "hard_filter_result.work_authorization.citizenship_or_green_card_required",
    "hard_filter_result.avoid_pure_infra",
)


@dataclass(frozen=True)
class ReviewOutcome:
    initial: ModelAssessment
    reviewed: ModelAssessment
    final: JobAnalysis
    changed_fields: list[str]


def _value_at(data, path):
    for part in path.split("."):
        data = data[part]
    return data


def changed_fact_fields(initial: ModelAssessment, reviewed: ModelAssessment) -> list[str]:
    before = initial.model_dump(mode="json", exclude_computed_fields=True)
    after = reviewed.model_dump(mode="json", exclude_computed_fields=True)
    return [path for path in FACT_PATHS if _value_at(before, path) != _value_at(after, path)]


def analyze_job_reviewed(
    job_description: str,
    *,
    max_tokens: int = 800,
    call_structured=call_structured_llm,
) -> ReviewOutcome:
    initial = call_structured(
        job_description,
        SYSTEM_PROMPT,
        ModelAssessment,
        "initial_job_facts",
        max_tokens=max_tokens,
    )
    candidate = json.dumps(
        initial.model_dump(mode="json", exclude_computed_fields=True),
        ensure_ascii=False,
        indent=2,
    )
    reviewed = call_structured(
        job_description,
        REVIEW_PROMPT,
        ModelAssessment,
        "reviewed_job_facts",
        max_tokens=max_tokens,
        task_context="Candidate extraction to audit:\n" + candidate,
    )
    return ReviewOutcome(
        initial=initial,
        reviewed=reviewed,
        final=finalize_assessment(reviewed),
        changed_fields=changed_fact_fields(initial, reviewed),
    )
