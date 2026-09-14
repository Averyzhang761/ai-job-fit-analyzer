import argparse
from collections import Counter
import json
import time
from pathlib import Path

from evaluation import compare_expected_fields, load_eval_cases
from review_pipeline import analyze_job_reviewed
from run_focused_baseline import select_cases


def main():
    parser = argparse.ArgumentParser(description="Run extract-then-review experiment.")
    parser.add_argument("--cases", default="data/eval_jds.real.jsonl")
    parser.add_argument("--output", default="output/review_baseline.json")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--ids", required=True, help="Comma-separated JD IDs.")
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",")]
    cases = select_cases(load_eval_cases(args.cases), ids=ids)
    rows = []
    failures = Counter()
    started = time.monotonic()
    for case in cases:
        try:
            outcome = analyze_job_reviewed(
                case["job_description"], max_tokens=args.max_tokens
            )
            result = outcome.final.model_dump(mode="json")
            mismatches = compare_expected_fields(case, result)
            for mismatch in mismatches:
                failures[mismatch.split(":", 1)[0]] += 1
            rows.append(
                {
                    "jd_id": case["jd_id"],
                    "status": "PASS" if not mismatches else "FAIL",
                    "details": "; ".join(mismatches) or "All expected fields matched",
                    "review_changed_fields": outcome.changed_fields,
                }
            )
        except Exception as exc:
            failures["runtime_error"] += 1
            rows.append(
                {
                    "jd_id": case["jd_id"],
                    "status": "ERROR",
                    "details": f"{type(exc).__name__}: {exc}",
                    "review_changed_fields": [],
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
        "pipeline": "extract-then-review-two-call",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
