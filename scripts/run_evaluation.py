import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyzer import analyze_job


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_cases(cases, ids=None, limit=None):
    selected = cases
    if ids:
        requested = set(ids)
        selected = [case for case in selected if case["jd_id"] in requested]
    return selected[:limit] if limit is not None else selected


def main():
    parser = argparse.ArgumentParser(
        description="Run the responsibility-evidence pipeline on saved JDs."
    )
    parser.add_argument("--cases", default="data/eval_jds.real.jsonl")
    parser.add_argument("--output", default="output/evaluation.json")
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    cases = select_cases(load_cases(args.cases), ids=ids, limit=args.limit)
    started = time.monotonic()
    rows = []
    for case in cases:
        try:
            decision = analyze_job(
                case["job_description"], max_tokens=args.max_tokens
            )
            rows.append(
                {
                    "jd_id": case["jd_id"],
                    "status": "OK",
                    "decision": decision.model_dump(mode="json"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "jd_id": case["jd_id"],
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "successful": sum(row["status"] == "OK" for row in rows),
        "failed": sum(row["status"] == "ERROR" for row in rows),
        "rows": rows,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pipeline": "responsibility-evidence-v6",
        "quality_boundary": (
            "Request success is not accuracy. Activity labels require "
            "responsibility-level human ground truth."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
