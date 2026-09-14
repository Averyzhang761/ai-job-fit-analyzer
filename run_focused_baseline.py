import argparse
import json
import time
from pathlib import Path

from evaluation import evaluate_regression_cases, load_eval_cases
from focused_pipeline import analyze_job_focused


def select_cases(cases, ids=None, limit=None):
    if ids:
        requested = set(ids)
        cases = [case for case in cases if case["jd_id"] in requested]
    if limit is not None:
        cases = cases[:limit]
    return cases


def main():
    parser = argparse.ArgumentParser(description="Run the focused-call baseline.")
    parser.add_argument("--cases", default="data/eval_jds.real.jsonl")
    parser.add_argument("--output", default="output/focused_baseline.json")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ids",
        help="Comma-separated JD IDs for a low-cost focused experiment.",
    )
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    cases = select_cases(load_eval_cases(args.cases), ids=ids, limit=args.limit)
    started = time.monotonic()
    report = evaluate_regression_cases(
        cases,
        lambda case: analyze_job_focused(
            case["job_description"], max_tokens=args.max_tokens
        ).model_dump(mode="json"),
    )
    report.update(
        total=len(cases),
        elapsed_seconds=round(time.monotonic() - started, 3),
        cases_path=args.cases,
        pipeline="focused-three-call",
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
