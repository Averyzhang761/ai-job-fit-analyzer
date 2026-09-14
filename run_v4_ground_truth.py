import argparse
import json
import time
from collections import Counter
from pathlib import Path

from business_rules_v4 import decide_job_v4
from evaluation_v4 import EvalCaseV4
from llm_client import call_structured_llm
from models import ModelAssessmentV4


V4_FACT_PROMPT = """Extract the job's work-content profile without forcing it
into one role category. Identify the primary deliverable, then rate customer
implementation, customer advisory, product development, internal tools, model
engineering, research, and AI infrastructure independently as core, present, or
not_evidenced. Core means a central responsibility; present means explicitly
part of the role; not_evidenced means the role responsibilities do not support
the claim. Company descriptions, mission statements, and generic boilerplate are
not role-responsibility evidence. Extract work arrangement, candidate location
eligibility, and work authorization using the schema definitions. Evidence must
use source chunk IDs. Do not produce a role type, score, or recommendation."""


def compare_case(case, actual):
    mismatches = []
    expected_role = case.expected.role.model_dump(mode="json")
    actual_role = actual.role.model_dump(mode="json")
    if expected_role["primary_deliverable"] != actual_role["primary_deliverable"]:
        mismatches.append({
            "field": "primary_deliverable",
            "expected": expected_role["primary_deliverable"],
            "actual": actual_role["primary_deliverable"],
        })
    for field, expected in expected_role["work_content"].items():
        actual_value = actual_role["work_content"][field]
        if expected != actual_value:
            mismatches.append({"field": field, "expected": expected, "actual": actual_value})

    expected_constraints = case.expected.constraints.model_dump(
        mode="json", exclude_computed_fields=True
    )
    actual_constraints = actual.constraints.model_dump(
        mode="json", exclude_computed_fields=True
    )
    constraint_fields = {
        "work_arrangement": ("location", "work_arrangement"),
        "candidate_location_eligible": ("location", "candidate_location_eligible"),
        "sponsorship_available": ("work_authorization", "sponsorship_available"),
        "citizenship_or_green_card_required": (
            "work_authorization",
            "citizenship_or_green_card_required",
        ),
    }
    for field, path in constraint_fields.items():
        expected = expected_constraints[path[0]][path[1]]
        actual_value = actual_constraints[path[0]][path[1]]
        if expected != actual_value:
            mismatches.append({"field": field, "expected": expected, "actual": actual_value})

    expected_decision = case.expected_recommendation().value
    actual_decision = decide_job_v4(actual)
    if expected_decision != actual_decision.recommendation.value:
        mismatches.append({
            "field": "recommendation",
            "expected": expected_decision,
            "actual": actual_decision.recommendation.value,
        })
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Run the V4 multi-dimensional evaluation.")
    parser.add_argument("--cases", default="data/eval_jds.v4.diagnostic.jsonl")
    parser.add_argument("--output", default="output/v4_ground_truth_report.json")
    parser.add_argument("--max-tokens", type=int, default=900)
    args = parser.parse_args()

    cases = [
        EvalCaseV4.model_validate_json(line)
        for line in Path(args.cases).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = time.monotonic()
    rows = []
    failures = Counter()
    for case in cases:
        try:
            actual = call_structured_llm(
                case.job_description,
                V4_FACT_PROMPT,
                ModelAssessmentV4,
                "job_facts_v4",
                max_tokens=args.max_tokens,
            )
            mismatches = compare_case(case, actual)
            for mismatch in mismatches:
                failures[mismatch["field"]] += 1
            rows.append({
                "jd_id": case.jd_id,
                "label_status": case.expected.label_status.value,
                "status": "PASS" if not mismatches else "FAIL",
                "mismatches": mismatches,
            })
        except Exception as exc:
            failures["runtime_error"] += 1
            rows.append({
                "jd_id": case.jd_id,
                "label_status": case.expected.label_status.value,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })
    passed = sum(row["status"] == "PASS" for row in rows)
    report = {
        "passed": passed,
        "failed": len(rows) - passed,
        "confirmed_labels": sum(case.expected.label_status.value == "confirmed" for case in cases),
        "provisional_labels": sum(case.expected.label_status.value == "provisional" for case in cases),
        "rows": rows,
        "field_failures": dict(failures),
        "total": len(rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pipeline": "v4-multidimensional-one-call",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
