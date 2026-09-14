from enum import Enum
from typing import Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)


class Recommendation(str, Enum):
    APPLY = "apply"
    SKIP = "skip"
    HUMAN_REVIEW = "human_review"


class TernarySignal(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class WorkArrangement(str, Enum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RemoteScope(str, Enum):
    US = "us"
    GLOBAL = "global"
    RESTRICTED_REGIONS = "restricted_regions"
    SPECIFIC_TIMEZONES = "specific_timezones"
    NOT_REMOTE = "not_remote"
    UNKNOWN = "unknown"


class SponsorshipStatement(str, Enum):
    OFFERED = "offered"
    NOT_OFFERED = "not_offered"
    CONDITIONAL = "conditional"
    NOT_STATED = "not_stated"


class CitizenshipStatement(str, Enum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    NOT_STATED = "not_stated"


class WorkActivity(str, Enum):
    CUSTOMER_IMPLEMENTATION = "customer_implementation"
    CUSTOMER_ADVISORY = "customer_advisory"
    PRODUCT_DEVELOPMENT = "product_development"
    INTERNAL_TOOLS = "internal_tools"
    MODEL_ENGINEERING = "model_engineering"
    RESEARCH = "research"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    GENERAL_SOFTWARE = "general_software"


class SourceLocationFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_arrangement: WorkArrangement
    stated_locations: list[str]
    remote_scope: RemoteScope


class SourceWorkAuthorizationFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sponsorship: SponsorshipStatement
    citizenship_or_green_card: CitizenshipStatement


class Responsibility(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    quote: str = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_id", "quote"),
    )
    activities: list[WorkActivity] = Field(min_length=1)

    @field_validator("quote", mode="before")
    @classmethod
    def resolve_quote(cls, value: str, info: ValidationInfo) -> str:
        evidence_map = (info.context or {}).get("evidence_map")
        if evidence_map is not None:
            if value not in evidence_map:
                raise ValueError(f"evidence_id 不存在：{value}")
            return evidence_map[value]
        return value.strip()

    @field_validator("activities")
    @classmethod
    def deduplicate_activities(
        cls, activities: list[WorkActivity]
    ) -> list[WorkActivity]:
        return list(dict.fromkeys(activities))


class JobFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    responsibilities: list[Responsibility] = Field(min_length=1)
    location: SourceLocationFacts
    work_authorization: SourceWorkAuthorizationFacts
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)

    @field_validator("risks")
    @classmethod
    def clean_risks(cls, risks: list[str]) -> list[str]:
        return [risk.strip() for risk in risks if risk.strip()]

    @model_validator(mode="after")
    def validate_quotes(self, info: ValidationInfo) -> Self:
        job_description = (info.context or {}).get("job_description")
        if not job_description:
            return self
        missing = [
            item.quote
            for item in self.responsibilities
            if item.quote not in job_description
        ]
        if missing:
            raise ValueError(
                "Responsibility quote 必须逐字存在于原始 JD："
                + "; ".join(missing)
            )
        return self


class CandidateLocationJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_location_eligible: TernarySignal
    reason: str = Field(min_length=1)


class JobDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: JobFacts
    activity_tags: list[WorkActivity]
    candidate_location_eligible: TernarySignal
    work_authorization_eligible: TernarySignal
    recommendation: Recommendation
    needs_human_review: StrictBool
    decision_reasons: list[str] = Field(min_length=1)

    @field_validator("activity_tags")
    @classmethod
    def deduplicate_tags(cls, tags: list[WorkActivity]) -> list[WorkActivity]:
        return list(dict.fromkeys(tags))

    @model_validator(mode="after")
    def validate_terminal_state(self) -> Self:
        expected = self.recommendation is Recommendation.HUMAN_REVIEW
        if self.needs_human_review != expected:
            raise ValueError("human_review 与 needs_human_review 必须保持一致")
        return self
