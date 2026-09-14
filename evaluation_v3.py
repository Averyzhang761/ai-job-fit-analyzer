from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from business_rules_v3 import decide_job, derive_role_type
from models import (
    CandidatePolicy,
    ConstraintFactsV3,
    ModelAssessmentV3,
    Recommendation,
    RoleFactsV3,
)


class LabelStatus(str, Enum):
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"


class GroundTruthV3(BaseModel):
    """Human-reviewed source facts; derived labels are never hand-entered."""

    model_config = ConfigDict(extra="forbid")

    role: RoleFactsV3
    constraints: ConstraintFactsV3
    evidence_by_fact: dict[str, list[str]] = Field(min_length=1)
    label_status: LabelStatus
    notes: str = ""


class EvalCaseV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jd_id: str
    source_company: str
    source_title: str
    source_url: str
    job_description: str
    expected: GroundTruthV3

    @model_validator(mode="after")
    def validate_ground_truth_evidence(self) -> Self:
        missing = []
        for fact, quotes in self.expected.evidence_by_fact.items():
            if not quotes:
                raise ValueError(f"Ground-truth fact 缺少证据：{fact}")
            for quote in quotes:
                if not quote.strip() or quote not in self.job_description:
                    missing.append(f"{fact}: {quote}")
        if missing:
            raise ValueError(
                "Ground-truth evidence 必须逐字存在于原始 JD：" + "; ".join(missing)
            )
        return self

    @computed_field
    @property
    def expected_role_type(self) -> str:
        return derive_role_type(_as_assessment(self.expected)).value

    def expected_recommendation(
        self, policy: CandidatePolicy | None = None
    ) -> Recommendation:
        return decide_job(_as_assessment(self.expected), policy).recommendation


def _as_assessment(expected: GroundTruthV3) -> ModelAssessmentV3:
    evidence = [
        {"field": "role_type", "quote": quote}
        for quotes in expected.evidence_by_fact.values()
        for quote in quotes
    ]
    return ModelAssessmentV3.model_validate(
        {
            "role": expected.role,
            "constraints": expected.constraints,
            "evidence": evidence,
            "risks": [],
            "confidence": 1.0,
        }
    )
