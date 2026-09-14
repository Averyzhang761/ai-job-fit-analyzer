import json

from business_rules_v6 import decide_job_v6
from llm_client import call_structured_llm
from models import (
    CandidateLocationJudgment,
    CitizenshipStatement,
    RemoteScope,
    SourceFactsV6,
    SponsorshipStatement,
    TernarySignal,
)


LOCATION_REVIEW_PROMPT = """Judge only whether the candidate can perform this
role from the San Francisco Bay Area. Use the extracted location facts and the
original JD. Return yes only when permitted, no only when an explicit location
or time-zone restriction excludes it, and unknown when the source is ambiguous.
Do not judge work authorization, role content, or recommendation."""


RESPONSIBILITY_PROMPT = """Extract explicit role responsibilities, not a role
category, primary deliverable, or importance score. Return each responsibility
as an exact source chunk ID and assign one or more applicable activity labels.
Ignore company descriptions, mission statements, benefits, and generic
boilerplate. Also extract stated location, work arrangement, remote scope, and
explicit work-authorization statements. Do not infer candidate eligibility or
produce a recommendation."""


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


def analyze_job_v6(job_description, call_structured=call_structured_llm, max_tokens=1200):
    facts = call_structured(
        job_description,
        RESPONSIBILITY_PROMPT,
        SourceFactsV6,
        "job_responsibilities_v6",
        max_tokens=max_tokens,
    )
    location = direct_location_eligibility(facts)
    review_used = location is None
    if review_used:
        review = call_structured(
            job_description,
            LOCATION_REVIEW_PROMPT,
            CandidateLocationJudgment,
            "candidate_location_v6",
            max_tokens=250,
            task_context="Extracted location facts:\n" + json.dumps(
                facts.location.model_dump(mode="json"), ensure_ascii=False
            ),
        )
        location = review.candidate_location_eligible
    work_auth = derive_work_authorization(facts)
    return decide_job_v6(facts, location, work_auth), review_used
