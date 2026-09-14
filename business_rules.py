from models import JobAnalysis, ModelAssessment, Recommendation, RoleType, TernarySignal


ROLE_BASE_SCORES = {
    RoleType.PRODUCT_FACING_AIE: 65,
    RoleType.FDE: 65,
    RoleType.INTERNAL_FACING_AIE: 50,
    RoleType.ML_HEAVY_AIE: 35,
    RoleType.PLATFORM_HEAVY_AIE: 0,
    RoleType.RESEARCH_ML: 0,
    RoleType.NOT_FIT: 0,
}

DISQUALIFIED_ROLES = {
    RoleType.PLATFORM_HEAVY_AIE,
    RoleType.RESEARCH_ML,
    RoleType.NOT_FIT,
}


def calculate_fit_score(assessment: ModelAssessment) -> tuple[int, list[str]]:
    hard_filters = assessment.hard_filter_result
    work_auth = hard_filters.work_authorization
    score = ROLE_BASE_SCORES[assessment.role_type]
    reasons = [f"Role type base: {score}"]

    if assessment.role_type in DISQUALIFIED_ROLES:
        return 0, reasons + ["Role family is outside the target."]

    adjustments = (
        (hard_filters.ai_application, 10, "AI application work"),
        (hard_filters.product_facing, 10, "Product/customer-facing work"),
        (
            hard_filters.bay_area_or_remote is TernarySignal.YES,
            5,
            "Bay Area or remote eligible",
        ),
        (
            work_auth.status is TernarySignal.YES,
            5,
            "Required work authorization is explicitly supported",
        ),
        (hard_filters.avoid_pure_infra, 5, "Avoids pure infrastructure"),
    )
    for applies, points, reason in adjustments:
        if applies:
            score += points
            reasons.append(f"+{points}: {reason}")

    if (
        work_auth.status is TernarySignal.NO
    ):
        score = min(score, 20)
        reasons.append("Score capped at 20 by an explicit work-authorization blocker.")

    if hard_filters.bay_area_or_remote is TernarySignal.NO:
        score = min(score, 30)
        reasons.append("Score capped at 30 by an explicit location blocker.")

    return min(score, 100), reasons


def finalize_assessment(assessment: ModelAssessment) -> JobAnalysis:
    hard_filters = assessment.hard_filter_result
    work_auth = hard_filters.work_authorization
    score, score_reasons = calculate_fit_score(assessment)

    recommendation = Recommendation.HUMAN_REVIEW
    needs_human_review = True

    if assessment.role_type in DISQUALIFIED_ROLES:
        recommendation = Recommendation.SKIP
        needs_human_review = False
    elif (
        work_auth.status is TernarySignal.NO
        or hard_filters.bay_area_or_remote is TernarySignal.NO
    ):
        recommendation = Recommendation.SKIP
        needs_human_review = False
    elif (
        work_auth.status is TernarySignal.UNKNOWN
        or hard_filters.bay_area_or_remote is TernarySignal.UNKNOWN
    ):
        recommendation = Recommendation.HUMAN_REVIEW
        needs_human_review = True
    elif (
        hard_filters.ai_application
        and hard_filters.product_facing
        and hard_filters.avoid_pure_infra
    ):
        recommendation = Recommendation.APPLY
        needs_human_review = False

    return JobAnalysis.model_validate(
        {
            **assessment.model_dump(mode="json", exclude_computed_fields=True),
            "fit_score": score,
            "score_reasons": score_reasons,
            "needs_human_review": needs_human_review,
            "recommendation": recommendation,
        }
    )
