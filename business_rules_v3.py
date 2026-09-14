from models import (
    CandidatePolicy,
    DerivedRoleType,
    JobDecisionV3,
    ModelAssessmentV3,
    PrimaryDeliverable,
    Recommendation,
    TernarySignal,
)


def derive_role_type(facts: ModelAssessmentV3) -> DerivedRoleType:
    role = facts.role
    delivery = role.delivery

    if role.delivers_ai_application is TernarySignal.YES:
        if delivery.embedded_customer_implementation is TernarySignal.YES:
            return DerivedRoleType.FDE
        if (
            delivery.customer_facing is TernarySignal.YES
            or delivery.product_collaboration is TernarySignal.YES
        ):
            return DerivedRoleType.PRODUCT_FACING_AIE
        if delivery.internal_user_facing is TernarySignal.YES:
            return DerivedRoleType.INTERNAL_FACING_AIE

    if role.primary_deliverable is PrimaryDeliverable.MODEL_ENGINEERING:
        return DerivedRoleType.ML_HEAVY_AIE
    if role.primary_deliverable is PrimaryDeliverable.RESEARCH:
        return DerivedRoleType.RESEARCH_ML
    if role.primary_deliverable is PrimaryDeliverable.AI_INFRASTRUCTURE:
        return DerivedRoleType.PLATFORM_HEAVY_AIE
    if role.primary_deliverable in {
        PrimaryDeliverable.NON_AI_PLATFORM,
        PrimaryDeliverable.GENERAL_SOFTWARE,
    }:
        return DerivedRoleType.NOT_FIT
    return DerivedRoleType.UNKNOWN


def decide_job(
    facts: ModelAssessmentV3,
    policy: CandidatePolicy | None = None,
) -> JobDecisionV3:
    policy = policy or CandidatePolicy()
    role_type = derive_role_type(facts)
    location = facts.constraints.location.candidate_location_eligible
    work_auth = facts.constraints.work_authorization.status

    if location is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit location blocker."]
    elif work_auth is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit work-authorization blocker."]
    elif role_type in policy.target_roles:
        if location is TernarySignal.UNKNOWN or work_auth is TernarySignal.UNKNOWN:
            recommendation = Recommendation.HUMAN_REVIEW
            reasons = ["Target role with unresolved location or work authorization."]
        else:
            recommendation = Recommendation.APPLY
            reasons = ["Target role with confirmed location and work authorization."]
    elif role_type in policy.review_roles or role_type is DerivedRoleType.UNKNOWN:
        recommendation = Recommendation.HUMAN_REVIEW
        reasons = ["Role fit requires human review under the candidate policy."]
    else:
        recommendation = Recommendation.SKIP
        reasons = ["Role family is outside the candidate policy."]

    return JobDecisionV3(
        facts=facts,
        derived_role_type=role_type,
        recommendation=recommendation,
        needs_human_review=recommendation is Recommendation.HUMAN_REVIEW,
        decision_reasons=reasons,
    )
