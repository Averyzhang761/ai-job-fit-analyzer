import json

from business_rules_v4 import STRONG_LEVELS, derive_role_tags
from llm_client import call_structured_llm
from models import (
    CandidateLocationJudgmentV5,
    CandidatePolicyV4,
    CitizenshipStatement,
    JobDecisionV4,
    ModelAssessmentV4,
    Recommendation,
    RemoteScope,
    SourceFactsV5,
    SponsorshipStatement,
    TernarySignal,
)
from two_stage_pipeline_v5 import EXTRACTION_PROMPT


LOCATION_REVIEW_PROMPT = """Judge only whether the candidate can perform this
role from the San Francisco Bay Area. Use the extracted location facts and the
original JD. Return yes only when permitted, no only when an explicit location
or time-zone restriction excludes it, and unknown when the source is ambiguous.
Do not judge work authorization, role content, or recommendation."""


def derive_work_authorization(facts):
    work_auth = facts.work_authorization
    if work_auth.citizenship_or_green_card is CitizenshipStatement.REQUIRED:
        return TernarySignal.NO
    if work_auth.sponsorship is SponsorshipStatement.NOT_OFFERED:
        return TernarySignal.NO
    if work_auth.sponsorship in {
        SponsorshipStatement.OFFERED,
        SponsorshipStatement.CONDITIONAL,
    }:
        return TernarySignal.YES
    return TernarySignal.UNKNOWN


def direct_location_eligibility(facts):
    if facts.location.remote_scope in {RemoteScope.US, RemoteScope.GLOBAL}:
        return TernarySignal.YES
    if facts.location.remote_scope is RemoteScope.UNKNOWN:
        return TernarySignal.UNKNOWN
    return None


def decide(facts, location, work_auth, policy=None):
    policy = policy or CandidatePolicyV4()
    content = facts.role.work_content
    has_core_target = any(
        getattr(content, name) in STRONG_LEVELS for name in policy.target_content
    )
    if location is TernarySignal.NO or work_auth is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit candidate constraint blocker."]
    elif location is TernarySignal.UNKNOWN or work_auth is TernarySignal.UNKNOWN:
        recommendation = Recommendation.HUMAN_REVIEW
        reasons = ["Candidate constraints remain unresolved."]
    elif has_core_target:
        recommendation = Recommendation.APPLY
        reasons = ["Core target work and candidate constraints are satisfied."]
    else:
        recommendation = Recommendation.SKIP
        reasons = ["No core target work is evidenced."]
    assessment = ModelAssessmentV4.model_validate({
        "role": facts.role,
        "constraints": {
            "location": {
                "work_arrangement": facts.location.work_arrangement,
                "candidate_location_eligible": location,
            },
            "work_authorization": {
                "sponsorship_available": work_auth,
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence": facts.evidence,
        "risks": facts.risks,
        "confidence": facts.confidence,
    })
    return JobDecisionV4(
        facts=assessment,
        role_tags=derive_role_tags(assessment),
        recommendation=recommendation,
        needs_human_review=recommendation is Recommendation.HUMAN_REVIEW,
        decision_reasons=reasons,
    )


def analyze_job_conditional(job_description, call_structured=call_structured_llm, max_tokens=900):
    facts = call_structured(
        job_description,
        EXTRACTION_PROMPT,
        SourceFactsV5,
        "job_source_facts_v5",
        max_tokens=max_tokens,
    )
    location = direct_location_eligibility(facts)
    review_used = location is None
    if review_used:
        review = call_structured(
            job_description,
            LOCATION_REVIEW_PROMPT,
            CandidateLocationJudgmentV5,
            "candidate_location_v5",
            max_tokens=250,
            task_context="Extracted location facts:\n" + json.dumps(
                facts.location.model_dump(mode="json"), ensure_ascii=False
            ),
        )
        location = review.candidate_location_eligible
    work_auth = derive_work_authorization(facts)
    return facts, location, work_auth, decide(facts, location, work_auth), review_used
