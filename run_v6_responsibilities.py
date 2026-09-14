import argparse
import json
import time
from pathlib import Path

from evaluation import load_eval_cases
from run_focused_baseline import select_cases
from v6_pipeline import analyze_job_v6


def main():
    parser = argparse.ArgumentParser(description="Inspect V6 responsibility extraction.")
    parser.add_argument("--cases", default="data/eval_jds.real.jsonl")
    parser.add_argument("--output", default="output/v6_responsibilities.json")
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    cases = select_cases(load_eval_cases(args.cases), ids=ids, limit=args.limit)
    started = time.monotonic()
    rows = []
    for case in cases:
        try:
            decision, review_used = analyze_job_v6(
                case["job_description"], max_tokens=args.max_tokens
            )
            rows.append({
                "jd_id": case["jd_id"],
                "status": "OK",
                "location_reviewer_used": review_used,
                "decision": decision.model_dump(mode="json"),
            })
        except Exception as exc:
            rows.append({"jd_id": case["jd_id"], "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "successful": sum(row["status"] == "OK" for row in rows),
        "failed": sum(row["status"] == "ERROR" for row in rows),
        "rows": rows,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pipeline": "v6-source-backed-responsibilities",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
