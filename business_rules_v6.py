from models import (
    JobDecisionV6,
    Recommendation,
    SourceFactsV6,
    TernarySignal,
    WorkActivity,
)


TARGET_ACTIVITIES = {
    WorkActivity.CUSTOMER_IMPLEMENTATION,
    WorkActivity.CUSTOMER_ADVISORY,
    WorkActivity.PRODUCT_DEVELOPMENT,
}


def derive_activity_tags(facts: SourceFactsV6) -> frozenset[WorkActivity]:
    return frozenset(
        activity
        for responsibility in facts.responsibilities
        for activity in responsibility.activities
    )


def decide_job_v6(
    facts: SourceFactsV6,
    location: TernarySignal,
    work_authorization: TernarySignal,
) -> JobDecisionV6:
    tags = derive_activity_tags(facts)
    if location is TernarySignal.NO or work_authorization is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit candidate constraint blocker."]
    elif location is TernarySignal.UNKNOWN or work_authorization is TernarySignal.UNKNOWN:
        recommendation = Recommendation.HUMAN_REVIEW
        reasons = ["Candidate constraints remain unresolved."]
    elif tags & TARGET_ACTIVITIES:
        recommendation = Recommendation.APPLY
        reasons = ["Source-backed target work is present and constraints are satisfied."]
    else:
        recommendation = Recommendation.SKIP
        reasons = ["No source-backed target work is present."]
    return JobDecisionV6(
        facts=facts,
        activity_tags=tags,
        candidate_location_eligible=location,
        work_authorization_eligible=work_authorization,
        recommendation=recommendation,
        needs_human_review=recommendation is Recommendation.HUMAN_REVIEW,
        decision_reasons=reasons,
    )
