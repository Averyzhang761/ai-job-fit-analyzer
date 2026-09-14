import argparse
import json
import time
from collections import Counter
from pathlib import Path

from evaluation_v4 import EvalCaseV4
from two_stage_pipeline_v5 import analyze_job_two_stage


def compare(case, facts, constraints, decision):
    mismatches = []
    expected_role = case.expected.role.model_dump(mode="json")
    actual_role = facts.role.model_dump(mode="json")
    if expected_role["primary_deliverable"] != actual_role["primary_deliverable"]:
        mismatches.append(("primary_deliverable", expected_role["primary_deliverable"], actual_role["primary_deliverable"]))
    for field, expected in expected_role["work_content"].items():
        actual = actual_role["work_content"][field]
        if expected != actual:
            mismatches.append((field, expected, actual))
    expected_location = case.expected.constraints.location.candidate_location_eligible.value
    if expected_location != constraints.candidate_location_eligible.value:
        mismatches.append(("candidate_location_eligible", expected_location, constraints.candidate_location_eligible.value))
    expected_auth = case.expected.constraints.work_authorization.status.value
    if expected_auth != constraints.work_authorization_eligible.value:
        mismatches.append(("work_authorization_eligible", expected_auth, constraints.work_authorization_eligible.value))
    expected_recommendation = case.expected_recommendation().value
    if expected_recommendation != decision.recommendation.value:
        mismatches.append(("recommendation", expected_recommendation, decision.recommendation.value))
    return [
        {"field": field, "expected": expected, "actual": actual}
        for field, expected, actual in mismatches
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the V5 two-stage evaluation.")
    parser.add_argument("--cases", default="data/eval_jds.v4.diagnostic.jsonl")
    parser.add_argument("--output", default="output/v5_two_stage_report.json")
    parser.add_argument("--max-tokens", type=int, default=900)
    args = parser.parse_args()
    cases = [
        EvalCaseV4.model_validate_json(line)
        for line in Path(args.cases).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = time.monotonic()
    rows, failures = [], Counter()
    for case in cases:
        try:
            facts, constraints, decision = analyze_job_two_stage(
                case.job_description, max_tokens=args.max_tokens
            )
            mismatches = compare(case, facts, constraints, decision)
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
            rows.append({"jd_id": case.jd_id, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    passed = sum(row["status"] == "PASS" for row in rows)
    report = {
        "passed": passed,
        "failed": len(rows) - passed,
        "rows": rows,
        "field_failures": dict(failures),
        "total": len(rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pipeline": "v5-two-stage-facts-then-policy",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
