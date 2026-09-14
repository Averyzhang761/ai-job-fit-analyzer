import json
from pathlib import Path

from evaluation_v4 import EvalCaseV4


SOURCE_PATH = Path("data/eval_jds.v3.diagnostic.jsonl")
OUTPUT_PATH = Path("data/eval_jds.v4.diagnostic.jsonl")
CONFIRMED_IDS = {"REAL-006"}


WORK_CONTENT = {
    "REAL-006": ["ai_application", "core", "core", "present", "not_evidenced", "not_evidenced", "not_evidenced", "core"],
    "REAL-010": ["ai_application", "core", "core", "present", "not_evidenced", "present", "not_evidenced", "present"],
    "REAL-012": ["ai_application", "present", "present", "core", "core", "present", "not_evidenced", "present"],
    "REAL-013": ["ai_application", "present", "core", "core", "not_evidenced", "not_evidenced", "not_evidenced", "not_evidenced"],
    "REAL-018": ["ai_application", "core", "present", "core", "core", "not_evidenced", "not_evidenced", "present"],
    "REAL-027": ["non_ai_platform", "not_evidenced", "not_evidenced", "core", "not_evidenced", "not_evidenced", "not_evidenced", "not_evidenced"],
    "REAL-029": ["model_engineering", "not_evidenced", "not_evidenced", "not_evidenced", "present", "core", "core", "core"],
    "REAL-035": ["ai_application", "present", "core", "core", "not_evidenced", "not_evidenced", "not_evidenced", "not_evidenced"],
}


FIELDS = (
    "customer_implementation",
    "customer_advisory",
    "product_development",
    "internal_tools",
    "model_engineering",
    "research",
    "ai_infrastructure",
)


def build_cases(source_path=SOURCE_PATH):
    sources = {
        row["jd_id"]: row
        for row in (
            json.loads(line)
            for line in Path(source_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    cases = []
    for jd_id, values in WORK_CONTENT.items():
        source = sources[jd_id]
        primary, *levels = values
        old_expected = source["expected"]
        quotes = [
            quote
            for items in old_expected["evidence_by_fact"].values()
            for quote in items
        ]
        cases.append(
            EvalCaseV4.model_validate(
                {
                    **{key: source[key] for key in ("jd_id", "source_company", "source_title", "source_url", "job_description")},
                    "expected": {
                        "role": {
                            "primary_deliverable": primary,
                            "work_content": dict(zip(FIELDS, levels)),
                        },
                        "constraints": old_expected["constraints"],
                        "evidence_by_fact": {"v4_work_content": quotes},
                        "label_status": (
                            "confirmed" if jd_id in CONFIRMED_IDS else "provisional"
                        ),
                        "notes": (
                            "V4 work-content intensities reviewed and confirmed."
                            if jd_id in CONFIRMED_IDS
                            else "V4 work-content intensity draft; requires human confirmation."
                        ),
                    },
                }
            )
        )
    return cases


def main():
    cases = build_cases()
    OUTPUT_PATH.write_text(
        "".join(
            case.model_dump_json(exclude_computed_fields=True) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} provisional V4 cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
