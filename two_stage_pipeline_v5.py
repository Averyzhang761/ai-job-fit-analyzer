import json

from business_rules_v4 import STRONG_LEVELS, derive_role_tags
from llm_client import call_structured_llm
from models import (
    CandidateConstraintJudgmentV5,
    CandidatePolicyV4,
    JobDecisionV4,
    ModelAssessmentV4,
    Recommendation,
    SourceFactsV5,
    TernarySignal,
)


EXTRACTION_PROMPT = """Extract source facts only. Describe the role's primary
deliverable and independent work-content dimensions as core, present, or
not_evidenced. Ignore company mission, company-wide research, benefits, and
generic boilerplate unless they explicitly describe this role's responsibilities.
Record stated locations, work arrangement, remote scope, sponsorship statement,
and citizenship or green-card statement. Do not infer candidate eligibility and
do not recommend apply, skip, or review. Evidence must use source chunk IDs."""


POLICY_PROMPT = """Audit the supplied extracted facts against the original JD,
then judge only candidate constraints. The candidate accepts San Francisco Bay
Area roles or US remote roles without conflicting region or time-zone limits.
Work authorization is eligible only when sponsorship is offered or conditional;
an explicit no-sponsorship or citizenship/green-card requirement is ineligible.
Use unknown when the source does not resolve the question. Do not classify the
role and do not produce a final recommendation."""


def decide_from_two_stages(facts, constraints, policy=None):
    policy = policy or CandidatePolicyV4()
    content = facts.role.work_content
    strong_target = any(
        getattr(content, name) in STRONG_LEVELS for name in policy.target_content
    )
    if constraints.candidate_location_eligible is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit location blocker."]
    elif constraints.work_authorization_eligible is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit work-authorization blocker."]
    elif (
        constraints.candidate_location_eligible is TernarySignal.UNKNOWN
        or constraints.work_authorization_eligible is TernarySignal.UNKNOWN
    ):
        recommendation = Recommendation.HUMAN_REVIEW
        reasons = ["Candidate constraints remain unresolved."]
    elif strong_target:
        recommendation = Recommendation.APPLY
        reasons = ["The role contains core target work and clears constraints."]
    else:
        recommendation = Recommendation.SKIP
        reasons = ["The work-content profile is outside the candidate policy."]

    compatibility_facts = ModelAssessmentV4.model_validate(
        {
            "role": facts.role,
            "constraints": {
                "location": {
                    "work_arrangement": facts.location.work_arrangement,
                    "candidate_location_eligible": constraints.candidate_location_eligible,
                },
                "work_authorization": {
                    "sponsorship_available": constraints.work_authorization_eligible,
                    "citizenship_or_green_card_required": "unknown",
                },
            },
            "evidence": facts.evidence,
            "risks": facts.risks,
            "confidence": facts.confidence,
        }
    )
    return JobDecisionV4(
        facts=compatibility_facts,
        role_tags=derive_role_tags(compatibility_facts),
        recommendation=recommendation,
        needs_human_review=recommendation is Recommendation.HUMAN_REVIEW,
        decision_reasons=reasons,
    )


def analyze_job_two_stage(job_description, call_structured=call_structured_llm, max_tokens=900):
    facts = call_structured(
        job_description,
        EXTRACTION_PROMPT,
        SourceFactsV5,
        "job_source_facts_v5",
        max_tokens=max_tokens,
    )
    constraints = call_structured(
        job_description,
        POLICY_PROMPT,
        CandidateConstraintJudgmentV5,
        "candidate_constraints_v5",
        max_tokens=400,
        task_context="Extracted source facts:\n" + json.dumps(
            facts.model_dump(mode="json", exclude_computed_fields=True),
            ensure_ascii=False,
        ),
    )
    return facts, constraints, decide_from_two_stages(facts, constraints)
