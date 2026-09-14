import json

from business_rules_v6 import decide_job_v6
from conditional_pipeline_v5 import (
    LOCATION_REVIEW_PROMPT,
    derive_work_authorization,
    direct_location_eligibility,
)
from llm_client import call_structured_llm
from models import CandidateLocationJudgmentV5, SourceFactsV6


RESPONSIBILITY_PROMPT = """Extract explicit role responsibilities, not a role
category, primary deliverable, or importance score. Return each responsibility
as an exact source chunk ID and assign one or more applicable activity labels.
Ignore company descriptions, mission statements, benefits, and generic
boilerplate. Also extract stated location, work arrangement, remote scope, and
explicit work-authorization statements. Do not infer candidate eligibility or
produce a recommendation."""


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
            CandidateLocationJudgmentV5,
            "candidate_location_v6",
            max_tokens=250,
            task_context="Extracted location facts:\n" + json.dumps(
                facts.location.model_dump(mode="json"), ensure_ascii=False
            ),
        )
        location = review.candidate_location_eligible
    work_auth = derive_work_authorization(facts)
    return decide_job_v6(facts, location, work_auth), review_used
