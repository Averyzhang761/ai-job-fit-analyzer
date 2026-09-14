from enum import Enum
from typing import Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)


class RoleType(str, Enum):
    PRODUCT_FACING_AIE = "Product-facing AIE"
    INTERNAL_FACING_AIE = "Internal-facing AIE"
    FDE = "FDE"
    RESEARCH_ML = "Research/ML role"
    ML_HEAVY_AIE = "ML-heavy AIE"
    PLATFORM_HEAVY_AIE = "platform-heavy AIE"
    NOT_FIT = "not fit"


class Recommendation(str, Enum):
    APPLY = "apply"
    SKIP = "skip"
    HUMAN_REVIEW = "human_review"


class TernarySignal(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class PrimaryDeliverable(str, Enum):
    AI_APPLICATION = "ai_application"
    MODEL_ENGINEERING = "model_engineering"
    RESEARCH = "research"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    NON_AI_PLATFORM = "non_ai_platform"
    GENERAL_SOFTWARE = "general_software"
    UNKNOWN = "unknown"


class WorkArrangement(str, Enum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DerivedRoleType(str, Enum):
    PRODUCT_FACING_AIE = "Product-facing AIE"
    INTERNAL_FACING_AIE = "Internal-facing AIE"
    FDE = "FDE"
    RESEARCH_ML = "Research/ML role"
    ML_HEAVY_AIE = "ML-heavy AIE"
    PLATFORM_HEAVY_AIE = "platform-heavy AIE"
    NOT_FIT = "not fit"
    UNKNOWN = "unknown"


class EvidenceField(str, Enum):
    ROLE_TYPE = "role_type"
    AI_APPLICATION = "ai_application"
    PRODUCT_FACING = "product_facing"
    LOCATION = "location"
    WORK_AUTHORIZATION = "work_authorization"
    INFRASTRUCTURE = "infrastructure"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    field: EvidenceField
    quote: str = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_id", "quote"),
    )

    @field_validator("quote", mode="before")
    @classmethod
    def resolve_quote(cls, value: str, info: ValidationInfo) -> str:
        evidence_map = (info.context or {}).get("evidence_map")
        if evidence_map is not None:
            if value not in evidence_map:
                raise ValueError(f"evidence_id 不存在：{value}")
            return evidence_map[value]
        return value.strip()


class WorkAuthorizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sponsorship_available: TernarySignal
    citizenship_or_green_card_required: TernarySignal

    @computed_field
    @property
    def status(self) -> TernarySignal:
        if self.citizenship_or_green_card_required is TernarySignal.YES:
            return TernarySignal.NO
        if self.sponsorship_available is TernarySignal.YES:
            return TernarySignal.YES
        if self.sponsorship_available is TernarySignal.NO:
            return TernarySignal.NO
        return TernarySignal.UNKNOWN


class HardFilterResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ai_application: StrictBool
    product_facing: StrictBool
    bay_area_or_remote: TernarySignal
    work_authorization: WorkAuthorizationResult
    avoid_pure_infra: StrictBool


class ModelAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role_type: RoleType
    hard_filter_result: HardFilterResult
    evidence: list[EvidenceItem] = Field(min_length=1)
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)

    @field_validator("risks")
    @classmethod
    def clean_risks(cls, risks: list[str]) -> list[str]:
        return [risk.strip() for risk in risks if risk.strip()]

    @model_validator(mode="after")
    def validate_evidence_quotes(self, info: ValidationInfo) -> Self:
        job_description = (info.context or {}).get("job_description")
        if not job_description:
            return self
        missing = [
            item.quote for item in self.evidence if item.quote not in job_description
        ]
        if missing:
            raise ValueError(
                "Evidence quote 必须逐字存在于原始 JD：" + "; ".join(missing)
            )
        return self


class JobAnalysis(ModelAssessment):
    fit_score: int = Field(ge=0, le=100, strict=True)
    score_reasons: list[str] = Field(min_length=1)
    needs_human_review: StrictBool
    recommendation: Recommendation

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        hard_filters = self.hard_filter_result
        if self.recommendation is Recommendation.SKIP and self.needs_human_review:
            raise ValueError("skip 与 needs_human_review=true 的终态矛盾")
        if self.recommendation is Recommendation.HUMAN_REVIEW:
            if not self.needs_human_review:
                raise ValueError("human_review 必须对应 needs_human_review=true")
        if self.role_type is RoleType.PLATFORM_HEAVY_AIE:
            if hard_filters.product_facing:
                raise ValueError("platform-heavy AIE 不能同时是 product-facing")
            if hard_filters.avoid_pure_infra:
                raise ValueError("platform-heavy AIE 必须标记为未避开纯基础设施")
        if self.role_type is RoleType.RESEARCH_ML:
            if hard_filters.ai_application or hard_filters.product_facing:
                raise ValueError("Research/ML role 与应用或产品向信号矛盾")
        return self


class DeliveryFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_facing: TernarySignal
    product_collaboration: TernarySignal = Field(
        description=(
            "Whether the role directly collaborates with a product team or owns "
            "product development; collaboration with scientists or generic users "
            "alone is not product collaboration."
        )
    )
    embedded_customer_implementation: TernarySignal = Field(
        description=(
            "Yes only when the role owns hands-on implementation in or alongside "
            "a customer's environment; advising, workshops, architecture guidance, "
            "and support alone are not embedded implementation."
        )
    )
    internal_user_facing: TernarySignal


class RoleFactsV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_deliverable: PrimaryDeliverable = Field(
        description=(
            "The dominant artifact or outcome the role is accountable for, not every "
            "technology or secondary responsibility mentioned in the JD."
        )
    )
    delivers_ai_application: TernarySignal = Field(
        description=(
            "Whether the role primarily ships a user or business workflow powered by "
            "AI; model training, model research, and infrastructure alone are not an "
            "AI application."
        )
    )
    trains_or_optimizes_models: TernarySignal
    operates_ai_infrastructure: TernarySignal = Field(
        description=(
            "Whether the role operates infrastructure specifically for the AI model "
            "or AI-application lifecycle; ordinary software, media, cloud, or product "
            "platform infrastructure is not AI infrastructure."
        )
    )
    delivery: DeliveryFacts


class LocationFactsV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_arrangement: WorkArrangement
    candidate_location_eligible: TernarySignal = Field(
        description=(
            "Whether the stated geography permits this candidate to work from the San "
            "Francisco Bay Area or through US remote work without a conflicting region "
            "or time-zone restriction. Remote alone does not imply eligibility."
        )
    )


class ConstraintFactsV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: LocationFactsV3
    work_authorization: WorkAuthorizationResult


class ModelAssessmentV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: RoleFactsV3
    constraints: ConstraintFactsV3
    evidence: list[EvidenceItem] = Field(min_length=1)
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)

    @field_validator("risks")
    @classmethod
    def clean_v3_risks(cls, risks: list[str]) -> list[str]:
        return [risk.strip() for risk in risks if risk.strip()]

    @model_validator(mode="after")
    def validate_v3_evidence_quotes(self, info: ValidationInfo) -> Self:
        job_description = (info.context or {}).get("job_description")
        if not job_description:
            return self
        missing = [item.quote for item in self.evidence if item.quote not in job_description]
        if missing:
            raise ValueError(
                "Evidence quote 必须逐字存在于原始 JD：" + "; ".join(missing)
            )
        return self


class CandidatePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_roles: frozenset[DerivedRoleType] = frozenset(
        {DerivedRoleType.PRODUCT_FACING_AIE, DerivedRoleType.FDE}
    )
    review_roles: frozenset[DerivedRoleType] = frozenset(
        {DerivedRoleType.INTERNAL_FACING_AIE, DerivedRoleType.ML_HEAVY_AIE}
    )


class JobDecisionV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: ModelAssessmentV3
    derived_role_type: DerivedRoleType
    recommendation: Recommendation
    needs_human_review: StrictBool
    decision_reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_v3_terminal_state(self) -> Self:
        expects_review = self.recommendation is Recommendation.HUMAN_REVIEW
        if self.needs_human_review != expects_review:
            raise ValueError("human_review 与 needs_human_review 必须保持一致")
        return self


class WorkContentLevel(str, Enum):
    CORE = "core"
    PRESENT = "present"
    NOT_EVIDENCED = "not_evidenced"


class WorkContentV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_implementation: WorkContentLevel
    customer_advisory: WorkContentLevel
    product_development: WorkContentLevel
    internal_tools: WorkContentLevel
    model_engineering: WorkContentLevel
    research: WorkContentLevel
    ai_infrastructure: WorkContentLevel


class RoleProfileV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_deliverable: PrimaryDeliverable
    work_content: WorkContentV4


class ModelAssessmentV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: RoleProfileV4
    constraints: ConstraintFactsV3
    evidence: list[EvidenceItem] = Field(min_length=1)
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)


class CandidatePolicyV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_content: frozenset[str] = frozenset(
        {"customer_implementation", "customer_advisory", "product_development"}
    )


class JobDecisionV4(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: ModelAssessmentV4
    role_tags: frozenset[str]
    recommendation: Recommendation
    needs_human_review: StrictBool
    decision_reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_v4_terminal_state(self) -> Self:
        expects_review = self.recommendation is Recommendation.HUMAN_REVIEW
        if self.needs_human_review != expects_review:
            raise ValueError("human_review 与 needs_human_review 必须保持一致")
        return self


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


class SourceLocationFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_arrangement: WorkArrangement
    stated_locations: list[str]
    remote_scope: RemoteScope


class SourceWorkAuthorizationFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sponsorship: SponsorshipStatement
    citizenship_or_green_card: CitizenshipStatement


class SourceFactsV5(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: RoleProfileV4
    location: SourceLocationFacts
    work_authorization: SourceWorkAuthorizationFacts
    evidence: list[EvidenceItem] = Field(min_length=1)
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)


class CandidateConstraintJudgmentV5(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_location_eligible: TernarySignal
    work_authorization_eligible: TernarySignal
    reasons: list[str] = Field(min_length=1)


class CandidateLocationJudgmentV5(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_location_eligible: TernarySignal
    reason: str = Field(min_length=1)


class WorkActivity(str, Enum):
    CUSTOMER_IMPLEMENTATION = "customer_implementation"
    CUSTOMER_ADVISORY = "customer_advisory"
    PRODUCT_DEVELOPMENT = "product_development"
    INTERNAL_TOOLS = "internal_tools"
    MODEL_ENGINEERING = "model_engineering"
    RESEARCH = "research"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    GENERAL_SOFTWARE = "general_software"


class ResponsibilityItemV6(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    quote: str = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_id", "quote"),
    )
    activities: list[WorkActivity] = Field(min_length=1)

    @field_validator("quote", mode="before")
    @classmethod
    def resolve_responsibility_quote(cls, value: str, info: ValidationInfo) -> str:
        evidence_map = (info.context or {}).get("evidence_map")
        if evidence_map is not None:
            if value not in evidence_map:
                raise ValueError(f"evidence_id 不存在：{value}")
            return evidence_map[value]
        return value.strip()


class SourceFactsV6(BaseModel):
    model_config = ConfigDict(extra="forbid")

    responsibilities: list[ResponsibilityItemV6] = Field(min_length=1)
    location: SourceLocationFacts
    work_authorization: SourceWorkAuthorizationFacts
    risks: list[str]
    confidence: float = Field(gt=0, le=1, strict=True)

    @model_validator(mode="after")
    def validate_responsibility_quotes(self, info: ValidationInfo) -> Self:
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


class JobDecisionV6(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: SourceFactsV6
    activity_tags: frozenset[WorkActivity]
    candidate_location_eligible: TernarySignal
    work_authorization_eligible: TernarySignal
    recommendation: Recommendation
    needs_human_review: StrictBool
    decision_reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_v6_terminal_state(self) -> Self:
        if self.needs_human_review != (
            self.recommendation is Recommendation.HUMAN_REVIEW
        ):
            raise ValueError("human_review 与 needs_human_review 必须保持一致")
        return self
