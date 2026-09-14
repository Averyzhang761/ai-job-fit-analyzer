import json


EXPECTED_FIELDS = {
    "expected_recommendation": ("recommendation",),
    "expected_role_type": ("role_type",),
    "expected_ai_application": ("hard_filter_result", "ai_application"),
    "expected_product_facing": ("hard_filter_result", "product_facing"),
    "expected_bay_area_or_remote": ("hard_filter_result", "bay_area_or_remote"),
    "expected_work_authorization": (
        "hard_filter_result",
        "work_authorization",
        "status",
    ),
    "expected_sponsorship_available": (
        "hard_filter_result",
        "work_authorization",
        "sponsorship_available",
    ),
    "expected_citizenship_or_green_card_required": (
        "hard_filter_result",
        "work_authorization",
        "citizenship_or_green_card_required",
    ),
    "expected_avoid_pure_infra": ("hard_filter_result", "avoid_pure_infra"),
}


def _get_nested(data, path):
    value = data
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def compare_expected_fields(case, result):
    mismatches = []
    for expected_key, path in EXPECTED_FIELDS.items():
        if expected_key not in case:
            continue
        actual = _get_nested(result, path)
        expected = case[expected_key]
        if actual != expected:
            mismatches.append(
                f"{path[-1]}: expected {expected}, got {actual}"
            )
    return mismatches


def evaluate_regression_cases(cases, analyze_case):
    rows = []
    passed = 0
    field_failures = {}
    for case in cases:
        try:
            result = analyze_case(case)
            mismatches = compare_expected_fields(case, result)
            status = "PASS" if not mismatches else "FAIL"
            details = "; ".join(mismatches) if mismatches else "All expected fields matched"
            for mismatch in mismatches:
                field = mismatch.split(":", 1)[0]
                field_failures[field] = field_failures.get(field, 0) + 1
        except Exception as exc:
            status = "ERROR"
            details = f"{type(exc).__name__}: {exc}"
            field_failures["runtime_error"] = field_failures.get("runtime_error", 0) + 1
        if status == "PASS":
            passed += 1
        rows.append({"jd_id": case["jd_id"], "status": status, "details": details})
    return {
        "passed": passed,
        "failed": len(cases) - passed,
        "rows": rows,
        "field_failures": field_failures,
    }


def load_eval_cases(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_markdown_report(report, total, elapsed_seconds):
    lines = [
        f"### Regression: {report['passed']}/{total} passed",
        f"Completed in {elapsed_seconds:.1f}s. Failed: {report['failed']}.",
    ]
    if report["field_failures"]:
        failure_summary = ", ".join(
            f"{field}={count}"
            for field, count in sorted(report["field_failures"].items())
        )
        lines.append(f"Failure breakdown: {failure_summary}.")
    lines.extend(
        [
            "",
            "| Case | Status | Details |",
            "|---|---|---|",
        ]
    )
    for row in report["rows"]:
        details = row["details"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['jd_id']} | **{row['status']}** | {details} |")
    return "\n".join(lines)
