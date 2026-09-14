import argparse
import json
import time
from pathlib import Path

from analysis_service import analyze_job
from evaluation import evaluate_regression_cases, load_eval_cases
from prompts import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(
        description="Run a source-backed JD baseline against the configured model."
    )
    parser.add_argument(
        "--cases",
        default="data/eval_jds.real.jsonl",
        help="Path to a JSONL evaluation set.",
    )
    parser.add_argument(
        "--output",
        default="output/real_baseline_report.json",
        help="Path for the machine-readable report.",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases (useful for a low-cost smoke test).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    cases = load_eval_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    started = time.monotonic()

    def analyze_case(case):
        return analyze_job(
            case["job_description"],
            SYSTEM_PROMPT,
            max_tokens=args.max_tokens,
        ).model_dump(mode="json")

    report = evaluate_regression_cases(cases, analyze_case)
    report["total"] = len(cases)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["cases_path"] = args.cases

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
