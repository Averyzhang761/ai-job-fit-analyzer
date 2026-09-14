"""Experimental multi-call pipeline for controlled comparison with the baseline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from business_rules import finalize_assessment
from llm_client import call_structured_llm
from models import (
    EvidenceItem,
    JobAnalysis,
    ModelAssessment,
    RoleType,
    TernarySignal,
    WorkAuthorizationResult,
)


ROLE_PROMPT = """Classify the role from its primary day-to-day deliverable.
Return facts only. Distinguish application delivery, customer/product workflow,
model engineering, research, and infrastructure. Do not produce a score or a
recommendation. Evidence must use source chunk IDs."""

LOCATION_PROMPT = """Determine whether the role is in the San Francisco Bay
Area or is remote for a candidate located in the United States. Interpret the
location and eligibility language in context. Missing information is unknown;
an explicit incompatible location is no. Evidence must use source chunk IDs."""

WORK_AUTH_PROMPT = """Extract only explicit work-authorization facts. Keep
employer sponsorship and a citizenship-or-green-card requirement separate.
Missing information is unknown. Do not infer sponsorship from the employer or
role. Evidence must use source chunk IDs."""

CORE_PROMPT = """Extract the role/content and work-authorization facts needed
for job-fit analysis. Classify from the primary day-to-day deliverable rather
than the employer or title. Keep sponsorship and citizenship-or-green-card
requirements separate, and use unknown when absent. Return facts only, without
a score, review state, or recommendation. Evidence must use source chunk IDs."""


class RoleFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role_type: RoleType
    ai_application: StrictBool
    product_facing: StrictBool
    avoid_pure_infra: StrictBool
    evidence: list[EvidenceItem] = Field(min_length=1)
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)


class LocationFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bay_area_or_remote: TernarySignal
    evidence: list[EvidenceItem] = Field(min_length=1)
    confidence: float = Field(gt=0, le=1, strict=True)


class WorkAuthorizationFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_authorization: WorkAuthorizationResult
    evidence: list[EvidenceItem] = Field(min_length=1)
    confidence: float = Field(gt=0, le=1, strict=True)


class CoreFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role_type: RoleType
    ai_application: StrictBool
    product_facing: StrictBool
    avoid_pure_infra: StrictBool
    work_authorization: WorkAuthorizationResult
    evidence: list[EvidenceItem] = Field(min_length=1)
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)


def analyze_job_focused(
    job_description: str,
    *,
    max_tokens: int = 500,
    call_structured=call_structured_llm,
) -> JobAnalysis:
    role = call_structured(
        job_description, ROLE_PROMPT, RoleFacts, "role_facts", max_tokens=max_tokens
    )
    location = call_structured(
        job_description,
        LOCATION_PROMPT,
        LocationFacts,
        "location_facts",
        max_tokens=max_tokens,
    )
    work_auth = call_structured(
        job_description,
        WORK_AUTH_PROMPT,
        WorkAuthorizationFacts,
        "work_authorization_facts",
        max_tokens=max_tokens,
    )
    assessment = ModelAssessment.model_validate(
        {
            "role_type": role.role_type,
            "hard_filter_result": {
                "ai_application": role.ai_application,
                "product_facing": role.product_facing,
                "bay_area_or_remote": location.bay_area_or_remote,
                "work_authorization": work_auth.work_authorization,
                "avoid_pure_infra": role.avoid_pure_infra,
            },
            "evidence": [
                *(item.model_dump() for item in role.evidence),
                *(item.model_dump() for item in location.evidence),
                *(item.model_dump() for item in work_auth.evidence),
            ],
            "risks": role.risks,
            "confidence": min(role.confidence, location.confidence, work_auth.confidence),
        },
        context={"job_description": job_description},
    )
    return finalize_assessment(assessment)


def analyze_job_hybrid(
    job_description: str,
    *,
    max_tokens: int = 800,
    call_structured=call_structured_llm,
) -> JobAnalysis:
    core = call_structured(
        job_description, CORE_PROMPT, CoreFacts, "core_facts", max_tokens=max_tokens
    )
    location = call_structured(
        job_description,
        LOCATION_PROMPT,
        LocationFacts,
        "location_facts",
        max_tokens=max_tokens,
    )
    assessment = ModelAssessment.model_validate(
        {
            "role_type": core.role_type,
            "hard_filter_result": {
                "ai_application": core.ai_application,
                "product_facing": core.product_facing,
                "bay_area_or_remote": location.bay_area_or_remote,
                "work_authorization": core.work_authorization,
                "avoid_pure_infra": core.avoid_pure_infra,
            },
            "evidence": [
                *(item.model_dump() for item in core.evidence),
                *(item.model_dump() for item in location.evidence),
            ],
            "risks": core.risks,
            "confidence": min(core.confidence, location.confidence),
        },
        context={"job_description": job_description},
    )
    return finalize_assessment(assessment)
