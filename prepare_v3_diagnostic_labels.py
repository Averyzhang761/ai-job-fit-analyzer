import json
from pathlib import Path

from evaluation_v3 import EvalCaseV3


SOURCE_PATH = Path("data/eval_jds.real.jsonl")
OUTPUT_PATH = Path("data/eval_jds.v3.diagnostic.jsonl")
CONFIRMED_IDS = {
    "REAL-006",
    "REAL-010",
    "REAL-012",
    "REAL-013",
    "REAL-018",
    "REAL-027",
    "REAL-029",
    "REAL-035",
}


ANNOTATIONS = {
    "REAL-006": {
        "role": {
            "primary_deliverable": "ai_application",
            "delivers_ai_application": "yes",
            "trains_or_optimizes_models": "unknown",
            "operates_ai_infrastructure": "yes",
            "delivery": {
                "customer_facing": "yes",
                "product_collaboration": "yes",
                "embedded_customer_implementation": "yes",
                "internal_user_facing": "unknown",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "unknown",
                "candidate_location_eligible": "unknown",
            },
            "work_authorization": {
                "sponsorship_available": "unknown",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.primary_deliverable": [
                "Build production-grade solutions alongside customers."
            ],
            "role.delivery.embedded_customer_implementation": [
                "you'll build alongside customers, not just advise them."
            ],
            "role.operates_ai_infrastructure": [
                "Guide self-hosted deployments."
            ],
        },
    },
    "REAL-010": {
        "role": {
            "primary_deliverable": "ai_application",
            "delivers_ai_application": "yes",
            "trains_or_optimizes_models": "unknown",
            "operates_ai_infrastructure": "unknown",
            "delivery": {
                "customer_facing": "yes",
                "product_collaboration": "yes",
                "embedded_customer_implementation": "yes",
                "internal_user_facing": "unknown",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "remote",
                "candidate_location_eligible": "yes",
            },
            "work_authorization": {
                "sponsorship_available": "unknown",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.delivery.embedded_customer_implementation": [
                "customer-embedded delivery engineer responsible for turning qualified AI opportunities into live, production-ready agent deployments"
            ],
            "constraints.location": ["Location: Anywhere, US"],
        },
    },
    "REAL-012": {
        "role": {
            "primary_deliverable": "ai_application",
            "delivers_ai_application": "yes",
            "trains_or_optimizes_models": "unknown",
            "operates_ai_infrastructure": "unknown",
            "delivery": {
                "customer_facing": "yes",
                "product_collaboration": "yes",
                "embedded_customer_implementation": "unknown",
                "internal_user_facing": "yes",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "unknown",
                "candidate_location_eligible": "no",
            },
            "work_authorization": {
                "sponsorship_available": "unknown",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.primary_deliverable": [
                "you are building the agentic layer of ai&"
            ],
            "role.delivery.customer_facing": [
                "you will be close to customers, understanding how they want to deploy agents"
            ],
            "constraints.location": ["Location: Yokohama; Tokyo"],
        },
    },
    "REAL-013": {
        "role": {
            "primary_deliverable": "ai_application",
            "delivers_ai_application": "yes",
            "trains_or_optimizes_models": "unknown",
            "operates_ai_infrastructure": "unknown",
            "delivery": {
                "customer_facing": "yes",
                "product_collaboration": "yes",
                "embedded_customer_implementation": "unknown",
                "internal_user_facing": "unknown",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "hybrid",
                "candidate_location_eligible": "yes",
            },
            "work_authorization": {
                "sponsorship_available": "yes",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.delivery.customer_facing": [
                "trusted technical advisor to Enterprise Technology companies adopting the Claude API into their core products"
            ],
            "constraints.location": [
                "Location: San Francisco, CA | New York City, NY | Seattle, WA"
            ],
            "constraints.work_authorization.sponsorship_available": [
                "Visa sponsorship: We do sponsor visas!"
            ],
        },
    },
    "REAL-018": {
        "role": {
            "primary_deliverable": "ai_application",
            "delivers_ai_application": "yes",
            "trains_or_optimizes_models": "unknown",
            "operates_ai_infrastructure": "unknown",
            "delivery": {
                "customer_facing": "no",
                "product_collaboration": "unknown",
                "embedded_customer_implementation": "no",
                "internal_user_facing": "yes",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "remote",
                "candidate_location_eligible": "no",
            },
            "work_authorization": {
                "sponsorship_available": "unknown",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.delivery.internal_user_facing": [
                "developing an internal platform that will centralize employee resources"
            ],
            "constraints.location": [
                "This role is open to candidates based in LATAM, Africa, and Eastern Europe."
            ],
        },
    },
    "REAL-027": {
        "role": {
            "primary_deliverable": "non_ai_platform",
            "delivers_ai_application": "no",
            "trains_or_optimizes_models": "no",
            "operates_ai_infrastructure": "no",
            "delivery": {
                "customer_facing": "unknown",
                "product_collaboration": "yes",
                "embedded_customer_implementation": "no",
                "internal_user_facing": "unknown",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "remote",
                "candidate_location_eligible": "no",
            },
            "work_authorization": {
                "sponsorship_available": "no",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.primary_deliverable": [
                "We're building the video and rendering platform that powers Slate's collaborative video editor."
            ],
            "constraints.location": [
                "Location: Remote, anywhere from EST (UTC-5) to UTC+2"
            ],
            "constraints.work_authorization.sponsorship_available": [
                "Note: We do not sponsor work visas."
            ],
        },
    },
    "REAL-029": {
        "role": {
            "primary_deliverable": "model_engineering",
            "delivers_ai_application": "no",
            "trains_or_optimizes_models": "yes",
            "operates_ai_infrastructure": "yes",
            "delivery": {
                "customer_facing": "unknown",
                "product_collaboration": "no",
                "embedded_customer_implementation": "no",
                "internal_user_facing": "yes",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "hybrid",
                "candidate_location_eligible": "yes",
            },
            "work_authorization": {
                "sponsorship_available": "unknown",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.primary_deliverable": [
                "build and improve the models and ML systems that drive our protein design efforts"
            ],
            "role.trains_or_optimizes_models": [
                "Optimize model training and inference code"
            ],
            "constraints.location": [
                "Location: Emeryville, California, United States; Hybrid (2-3 days on-site)"
            ],
        },
    },
    "REAL-035": {
        "role": {
            "primary_deliverable": "ai_application",
            "delivers_ai_application": "yes",
            "trains_or_optimizes_models": "unknown",
            "operates_ai_infrastructure": "unknown",
            "delivery": {
                "customer_facing": "yes",
                "product_collaboration": "yes",
                "embedded_customer_implementation": "no",
                "internal_user_facing": "unknown",
            },
        },
        "constraints": {
            "location": {
                "work_arrangement": "hybrid",
                "candidate_location_eligible": "yes",
            },
            "work_authorization": {
                "sponsorship_available": "yes",
                "citizenship_or_green_card_required": "unknown",
            },
        },
        "evidence_by_fact": {
            "role.primary_deliverable": [
                "Collaborate with partners to identify high value industry-specific GenAI applications"
            ],
            "role.delivery.customer_facing": [
                "Customer Deal Support: Intervene directly to unblock strategic customer deals"
            ],
            "constraints.work_authorization.sponsorship_available": [
                "Visa sponsorship: We do sponsor visas!"
            ],
        },
    },
}


def build_cases(source_path=SOURCE_PATH):
    source_cases = {
        row["jd_id"]: row
        for row in (
            json.loads(line)
            for line in Path(source_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    cases = []
    for jd_id, annotation in ANNOTATIONS.items():
        source = source_cases[jd_id]
        cases.append(
            EvalCaseV3.model_validate(
                {
                    "jd_id": jd_id,
                    "source_company": source["source_company"],
                    "source_title": source["source_title"],
                    "source_url": source["source_url"],
                    "job_description": source["job_description"],
                    "expected": {
                        **annotation,
                        "label_status": (
                            "confirmed" if jd_id in CONFIRMED_IDS else "provisional"
                        ),
                        "notes": (
                            "V3 facts reviewed and confirmed by the candidate."
                            if jd_id in CONFIRMED_IDS
                            else "V3 migration draft; requires human confirmation."
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
            json.dumps(
                case.model_dump(mode="json", exclude_computed_fields=True),
                ensure_ascii=False,
            )
            + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    confirmed = sum(
        case.expected.label_status.value == "confirmed" for case in cases
    )
    print(f"Wrote {len(cases)} V3 cases ({confirmed} confirmed) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
