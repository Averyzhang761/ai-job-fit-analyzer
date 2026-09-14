import argparse
import json
import time
from collections import Counter
from pathlib import Path

from business_rules_v3 import decide_job
from evaluation_v3 import EvalCaseV3
from llm_client import call_structured_llm
from models import ModelAssessmentV3
from run_v3_baseline import V3_FACT_PROMPT


def _flatten(data, prefix=""):
    values = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten(value, path))
        else:
            values[path] = value
    return values


def compare_v3_facts(case: EvalCaseV3, actual: ModelAssessmentV3):
    expected_facts = {
        "role": case.expected.role.model_dump(mode="json"),
        "constraints": case.expected.constraints.model_dump(
            mode="json", exclude_computed_fields=True
        ),
    }
    actual_facts = {
        "role": actual.role.model_dump(mode="json"),
        "constraints": actual.constraints.model_dump(
            mode="json", exclude_computed_fields=True
        ),
    }
    expected_flat = _flatten(expected_facts)
    actual_flat = _flatten(actual_facts)
    mismatches = []
    for path, expected in expected_flat.items():
        actual_value = actual_flat[path]
        if actual_value != expected:
            mismatches.append(
                {"field": path, "expected": expected, "actual": actual_value}
            )

    actual_decision = decide_job(actual)
    if actual_decision.derived_role_type.value != case.expected_role_type:
        mismatches.append(
            {
                "field": "derived_role_type",
                "expected": case.expected_role_type,
                "actual": actual_decision.derived_role_type.value,
            }
        )
    expected_recommendation = case.expected_recommendation().value
    if actual_decision.recommendation.value != expected_recommendation:
        mismatches.append(
            {
                "field": "recommendation",
                "expected": expected_recommendation,
                "actual": actual_decision.recommendation.value,
            }
        )
    return mismatches


def evaluate_cases(cases, analyze_case):
    rows = []
    failures = Counter()
    for case in cases:
        try:
            actual = analyze_case(case)
            mismatches = compare_v3_facts(case, actual)
            status = "PASS" if not mismatches else "FAIL"
            mismatch_fields = {mismatch["field"] for mismatch in mismatches}
            role_match = "derived_role_type" not in mismatch_fields
            recommendation_match = "recommendation" not in mismatch_fields
            for mismatch in mismatches:
                failures[mismatch["field"]] += 1
            rows.append(
                {
                    "jd_id": case.jd_id,
                    "status": status,
                    "role_match": role_match,
                    "recommendation_match": recommendation_match,
                    "decision_match": role_match and recommendation_match,
                    "mismatches": mismatches,
                }
            )
        except Exception as exc:
            failures["runtime_error"] += 1
            rows.append(
                {
                    "jd_id": case.jd_id,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    passed = sum(row["status"] == "PASS" for row in rows)
    return {
        "passed": passed,
        "failed": len(rows) - passed,
        "metrics": {
            "exact_fact_rows": passed,
            "derived_role_rows": sum(row.get("role_match", False) for row in rows),
            "recommendation_rows": sum(
                row.get("recommendation_match", False) for row in rows
            ),
            "decision_rows": sum(row.get("decision_match", False) for row in rows),
        },
        "rows": rows,
        "field_failures": dict(failures),
    }


def load_cases(path):
    return [
        EvalCaseV3.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_cases(cases, ids=None, limit=None):
    if ids:
        wanted = set(ids)
        cases = [case for case in cases if case.jd_id in wanted]
    if limit is not None:
        cases = cases[:limit]
    return cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3 facts against confirmed labels.")
    parser.add_argument("--cases", default="data/eval_jds.v3.diagnostic.jsonl")
    parser.add_argument("--output", default="output/v3_ground_truth_report.json")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="Comma-separated JD IDs.")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    cases = select_cases(cases, ids=ids, limit=args.limit)
    started = time.monotonic()

    def analyze(case):
        return call_structured_llm(
            case.job_description,
            V3_FACT_PROMPT,
            ModelAssessmentV3,
            "job_facts_v3",
            max_tokens=args.max_tokens,
        )

    report = evaluate_cases(cases, analyze)
    report.update(
        total=len(cases),
        elapsed_seconds=round(time.monotonic() - started, 3),
        cases_path=args.cases,
        pipeline="v3-confirmed-facts-one-call",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
