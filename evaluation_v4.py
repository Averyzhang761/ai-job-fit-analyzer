from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from business_rules_v4 import decide_job_v4, derive_role_tags
from evaluation_v3 import LabelStatus
from models import CandidatePolicyV4, ConstraintFactsV3, ModelAssessmentV4, RoleProfileV4


class GroundTruthV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: RoleProfileV4
    constraints: ConstraintFactsV3
    evidence_by_fact: dict[str, list[str]] = Field(min_length=1)
    label_status: LabelStatus
    notes: str = ""


class EvalCaseV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jd_id: str
    source_company: str
    source_title: str
    source_url: str
    job_description: str
    expected: GroundTruthV4

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
    def expected_role_tags(self) -> list[str]:
        return sorted(derive_role_tags(_as_assessment(self.expected)))

    def expected_recommendation(
        self, policy: CandidatePolicyV4 | None = None
    ):
        return decide_job_v4(_as_assessment(self.expected), policy).recommendation


def _as_assessment(expected: GroundTruthV4) -> ModelAssessmentV4:
    evidence = [
        {"field": "role_type", "quote": quote}
        for quotes in expected.evidence_by_fact.values()
        for quote in quotes
    ]
    return ModelAssessmentV4.model_validate(
        {
            "role": expected.role,
            "constraints": expected.constraints,
            "evidence": evidence,
            "risks": [],
            "confidence": 1.0,
        }
    )
