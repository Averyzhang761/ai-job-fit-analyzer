import argparse
import json
import time
from pathlib import Path

from business_rules_v3 import decide_job
from evaluation import evaluate_regression_cases, load_eval_cases
from llm_client import call_structured_llm
from models import ModelAssessmentV3, PrimaryDeliverable, TernarySignal
from run_focused_baseline import select_cases


V3_FACT_PROMPT = """Extract independent job facts, not a fit label or final
decision. Identify the primary deliverable and separately record AI application
delivery, model engineering, AI infrastructure, customer contact, product
collaboration, embedded implementation, and internal users. These facts may
overlap. Use no only when the source directly supports a negative; otherwise use
unknown. Separately record the stated work arrangement and whether the candidate
is geographically eligible: eligible means the role permits work from the San
Francisco Bay Area, or permits US remote work without a conflicting region or
time-zone restriction. Extract work-authorization facts using unknown when the
source is insufficient. Evidence must use source chunk IDs. Do not produce a
role type, score, review state, or recommendation."""


def _yes(signal):
    return signal is TernarySignal.YES


def to_evaluation_result(decision):
    facts = decision.facts
    role = facts.role
    delivery = role.delivery
    product_facing = (
        delivery.customer_facing is TernarySignal.YES
        or delivery.product_collaboration is TernarySignal.YES
        or delivery.embedded_customer_implementation is TernarySignal.YES
    )
    avoid_pure_infra = role.primary_deliverable not in {
        PrimaryDeliverable.AI_INFRASTRUCTURE,
        PrimaryDeliverable.NON_AI_PLATFORM,
    }
    return {
        "role_type": decision.derived_role_type.value,
        "recommendation": decision.recommendation.value,
        "hard_filter_result": {
            "ai_application": _yes(role.delivers_ai_application),
            "product_facing": product_facing,
            "bay_area_or_remote": (
                facts.constraints.location.candidate_location_eligible.value
            ),
            "work_authorization": facts.constraints.work_authorization.model_dump(
                mode="json"
            ),
            "avoid_pure_infra": avoid_pure_infra,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run the V3 facts-first baseline.")
    parser.add_argument("--cases", default="data/eval_jds.real.jsonl")
    parser.add_argument("--output", default="output/v3_baseline.json")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", help="Comma-separated JD IDs.")
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    cases = select_cases(load_eval_cases(args.cases), ids=ids, limit=args.limit)
    started = time.monotonic()

    def analyze(case):
        facts = call_structured_llm(
            case["job_description"],
            V3_FACT_PROMPT,
            ModelAssessmentV3,
            "job_facts_v3",
            max_tokens=args.max_tokens,
        )
        return to_evaluation_result(decide_job(facts))

    report = evaluate_regression_cases(cases, analyze)
    report.update(
        total=len(cases),
        elapsed_seconds=round(time.monotonic() - started, 3),
        cases_path=args.cases,
        pipeline="v3-facts-first-one-call",
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
