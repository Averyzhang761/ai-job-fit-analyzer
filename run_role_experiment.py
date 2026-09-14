import argparse
import json
import time
from collections import Counter
from pathlib import Path

from evaluation import load_eval_cases
from focused_pipeline import ROLE_PROMPT, RoleFacts
from llm_client import call_structured_llm
from run_focused_baseline import select_cases


ROLE_FIELDS = {
    "role_type": "expected_role_type",
    "ai_application": "expected_ai_application",
    "product_facing": "expected_product_facing",
    "avoid_pure_infra": "expected_avoid_pure_infra",
}


def compare_role_facts(case, facts):
    actual = facts.model_dump(mode="json")
    return [
        f"{field}: expected {case[expected_key]}, got {actual[field]}"
        for field, expected_key in ROLE_FIELDS.items()
        if case[expected_key] != actual[field]
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the focused role experiment.")
    parser.add_argument("--cases", default="data/eval_jds.real.jsonl")
    parser.add_argument("--output", default="output/focused_role_experiment.json")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--ids", required=True, help="Comma-separated JD IDs.")
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",")]
    cases = select_cases(load_eval_cases(args.cases), ids=ids)
    rows = []
    failures = Counter()
    started = time.monotonic()
    for case in cases:
        try:
            facts = call_structured_llm(
                case["job_description"],
                ROLE_PROMPT,
                RoleFacts,
                "role_facts",
                max_tokens=args.max_tokens,
            )
            mismatches = compare_role_facts(case, facts)
            for mismatch in mismatches:
                failures[mismatch.split(":", 1)[0]] += 1
            rows.append(
                {
                    "jd_id": case["jd_id"],
                    "status": "PASS" if not mismatches else "FAIL",
                    "details": "; ".join(mismatches) or "All expected role fields matched",
                }
            )
        except Exception as exc:
            failures["runtime_error"] += 1
            rows.append(
                {
                    "jd_id": case["jd_id"],
                    "status": "ERROR",
                    "details": f"{type(exc).__name__}: {exc}",
                }
            )

    passed = sum(row["status"] == "PASS" for row in rows)
    report = {
        "passed": passed,
        "failed": len(rows) - passed,
        "rows": rows,
        "field_failures": dict(failures),
        "total": len(rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pipeline": "focused-role-one-call",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
