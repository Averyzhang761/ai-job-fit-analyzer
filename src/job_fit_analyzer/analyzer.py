import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .llm_client import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_RUNPOD_MODEL,
    JobAnalyzerError,
    call_structured_llm,
)
from .models import (
    CandidateLocationJudgment,
    CitizenshipStatement,
    JobDecision,
    JobFacts,
    Recommendation,
    RemoteScope,
    SponsorshipStatement,
    TernarySignal,
    WorkActivity,
)


EXTRACTION_PROMPT = """Extract only explicit job responsibilities and source
constraints. For each responsibility, return its exact source chunk ID and one
or more applicable activity labels. Do not classify the whole role, estimate
importance, infer candidate fit, or produce a recommendation. Ignore company
descriptions, benefits, and generic boilerplate. Extract stated locations, work
arrangement, remote scope, sponsorship, and citizenship or green-card
requirements without converting them into candidate eligibility."""

LOCATION_REVIEW_PROMPT = """Judge only whether the stated location constraints
allow this candidate to perform the role from the San Francisco Bay Area. Return
yes only when permitted, no only when an explicit location or time-zone
restriction excludes it, and unknown when the source remains ambiguous. Do not
judge work content, work authorization, or recommendation."""

TARGET_ACTIVITIES = {
    WorkActivity.CUSTOMER_IMPLEMENTATION,
    WorkActivity.CUSTOMER_ADVISORY,
    WorkActivity.PRODUCT_DEVELOPMENT,
}
DEFAULT_LOG_PATH = Path(__file__).parent / "output" / "run_history.jsonl"


def derive_activity_tags(facts: JobFacts) -> list[WorkActivity]:
    return list(
        dict.fromkeys(
            activity
            for responsibility in facts.responsibilities
            for activity in responsibility.activities
        )
    )


def derive_work_authorization(facts: JobFacts) -> TernarySignal:
    statement = facts.work_authorization
    if statement.citizenship_or_green_card is CitizenshipStatement.REQUIRED:
        return TernarySignal.NO
    if statement.sponsorship is SponsorshipStatement.NOT_OFFERED:
        return TernarySignal.NO
    if statement.sponsorship in {
        SponsorshipStatement.OFFERED,
        SponsorshipStatement.CONDITIONAL,
    }:
        return TernarySignal.YES
    return TernarySignal.UNKNOWN


def direct_location_eligibility(facts: JobFacts) -> TernarySignal | None:
    if facts.location.remote_scope in {RemoteScope.US, RemoteScope.GLOBAL}:
        return TernarySignal.YES
    if facts.location.remote_scope is RemoteScope.UNKNOWN:
        return TernarySignal.UNKNOWN
    return None


def decide_job(
    facts: JobFacts,
    location: TernarySignal,
    work_authorization: TernarySignal,
) -> JobDecision:
    tags = derive_activity_tags(facts)
    if location is TernarySignal.NO or work_authorization is TernarySignal.NO:
        recommendation = Recommendation.SKIP
        reasons = ["Explicit candidate constraint blocker."]
    elif location is TernarySignal.UNKNOWN or work_authorization is TernarySignal.UNKNOWN:
        recommendation = Recommendation.HUMAN_REVIEW
        reasons = ["Candidate constraints remain unresolved."]
    elif set(tags) & TARGET_ACTIVITIES:
        recommendation = Recommendation.APPLY
        reasons = ["Source-backed target work is present and constraints are satisfied."]
    else:
        recommendation = Recommendation.SKIP
        reasons = ["No source-backed target work is present."]

    return JobDecision(
        facts=facts,
        activity_tags=tags,
        candidate_location_eligible=location,
        work_authorization_eligible=work_authorization,
        recommendation=recommendation,
        needs_human_review=recommendation is Recommendation.HUMAN_REVIEW,
        decision_reasons=reasons,
    )


def _record_run(event: dict, path=None) -> None:
    log_path = Path(path or os.getenv("RUN_LOG_PATH") or DEFAULT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _error_type(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "contract_error"
    if isinstance(exc, JobAnalyzerError):
        return exc.error_type
    return "unexpected_error"


def analyze_job(
    job_description: str,
    max_tokens: int = 1200,
    *,
    call_model=call_structured_llm,
    log_path=None,
) -> JobDecision:
    started = time.monotonic()
    prompt_version = hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest()[:12]
    model = (
        os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        if os.getenv("OPENROUTER_API_KEY")
        else os.getenv("RUNPOD_MODEL", DEFAULT_RUNPOD_MODEL)
    )
    config = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "prompt_version": prompt_version,
        "schema_version": "responsibility-evidence-v6",
    }

    try:
        facts = call_model(
            job_description,
            EXTRACTION_PROMPT,
            JobFacts,
            "job_facts",
            max_tokens=max_tokens,
        )
        location = direct_location_eligibility(facts)
        reviewer_used = location is None
        if reviewer_used:
            review = call_model(
                job_description,
                LOCATION_REVIEW_PROMPT,
                CandidateLocationJudgment,
                "candidate_location",
                max_tokens=250,
                task_context="Extracted location facts:\n"
                + json.dumps(facts.location.model_dump(mode="json")),
            )
            location = review.candidate_location_eligible

        decision = decide_job(
            facts,
            location,
            derive_work_authorization(facts),
        )
        _record_run(
            {
                "status": "success",
                "job_description": job_description,
                "decision": decision.model_dump(mode="json"),
                "location_reviewer_used": reviewer_used,
                "config": config,
                "latency_seconds": round(time.monotonic() - started, 3),
            },
            path=log_path,
        )
        return decision
    except Exception as exc:
        _record_run(
            {
                "status": "error",
                "job_description": job_description,
                "error_type": _error_type(exc),
                "error_message": str(exc),
                "config": config,
                "latency_seconds": round(time.monotonic() - started, 3),
            },
            path=log_path,
        )
        raise
