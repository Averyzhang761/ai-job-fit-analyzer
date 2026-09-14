from models import (
    CandidatePolicyV4,
    JobDecisionV4,
    ModelAssessmentV4,
    PrimaryDeliverable,
    Recommendation,
    TernarySignal,
    WorkContentLevel,
)


POSITIVE_LEVELS = {WorkContentLevel.CORE, WorkContentLevel.PRESENT}
STRONG_LEVELS = {WorkContentLevel.CORE}


def derive_role_tags(facts: ModelAssessmentV4) -> frozenset[str]:
    return frozenset(
        name
        for name, level in facts.role.work_content
        if level in POSITIVE_LEVELS
    )


def decide_job_v4(
    facts: ModelAssessmentV4,
    policy: CandidatePolicyV4 | None = None,
) -> JobDecisionV4:
    policy = policy or CandidatePolicyV4()
    tags = derive_role_tags(facts)
    location = facts.constraints.location.candidate_location_eligible
    work_auth = facts.constraints.work_authorization.status
    content = facts.role.work_content
    strong_target_content = {
        name
        for name in policy.target_content
        if getattr(content, name) in STRONG_LEVELS
    }
    if location is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit location blocker."]
    elif work_auth is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit work-authorization blocker."]
    elif location is TernarySignal.UNKNOWN or work_auth is TernarySignal.UNKNOWN:
        recommendation = Recommendation.HUMAN_REVIEW
        reasons = ["Location or work authorization remains unresolved."]
    elif (
        facts.role.primary_deliverable is PrimaryDeliverable.AI_APPLICATION
        and strong_target_content
    ):
        recommendation = Recommendation.APPLY
        reasons = ["AI application role contains substantial target work."]
    else:
        recommendation = Recommendation.SKIP
        reasons = ["The confirmed work mix is outside the candidate policy."]

    return JobDecisionV4(
        facts=facts,
        role_tags=tags,
        recommendation=recommendation,
        needs_human_review=recommendation is Recommendation.HUMAN_REVIEW,
        decision_reasons=reasons,
    )
