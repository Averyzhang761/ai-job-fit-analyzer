"""Refresh the real-JD evaluation set from official job-board APIs."""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen


DATA_PATH = Path("data/eval_jds.real.jsonl")
USER_AGENT = "Mozilla/5.0 (compatible; AIJobFitEval/1.0)"


# These postings disappeared from their public boards after the initial dataset
# was assembled. Replace them with current official postings while retaining the
# old URL in replaced_source_url for auditability.
REPLACEMENTS: dict[str, dict[str, Any]] = {
    "REAL-012": {
        "source_company": "ai&",
        "source_title": "Member of Technical Staff - Applied & Agentic AI",
        "source_url": "https://jobs.ashbyhq.com/aiand/b2401480-d371-466a-bcf4-2f3cda58a186",
        "expected_recommendation": "skip",
        "expected_role_type": "Product-facing AIE",
        "expected_ai_application": True,
        "expected_product_facing": True,
        "expected_bay_area_or_remote": "no",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": True,
    },
    "REAL-014": {
        "source_company": "OpenAI",
        "source_title": "Software Engineer, Full Stack (People Innovation)",
        "source_url": "https://jobs.ashbyhq.com/openai/d4780eac-03ad-4dae-861f-99af22b4287e",
        "expected_recommendation": "human_review",
        "expected_role_type": "Internal-facing AIE",
        "expected_ai_application": True,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "yes",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": True,
    },
    "REAL-015": {
        "source_company": "OpenAI",
        "source_title": "Software Engineer, Developer Productivity",
        "source_url": "https://jobs.ashbyhq.com/openai/2cba0d45-7a4f-4f38-ac73-3f8633bf0349",
        "expected_recommendation": "skip",
        "expected_role_type": "platform-heavy AIE",
        "expected_ai_application": True,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "yes",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": False,
    },
    "REAL-018": {
        "source_company": "Scale Army Careers",
        "source_title": "AI Internal Tools & Automation Engineer",
        "source_url": "https://jobs.ashbyhq.com/Scale%20Army%20Careers/8f695c4a-f5c9-4304-8681-ed3d431a7e4f",
        "expected_recommendation": "skip",
        "expected_role_type": "Internal-facing AIE",
        "expected_ai_application": True,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "no",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": True,
    },
    "REAL-019": {
        "source_company": "OpenAI",
        "source_title": "Software Engineer, Financial Engineering",
        "source_url": "https://jobs.ashbyhq.com/openai/4ef5bf23-cf0e-4b97-a639-11f963c99b88",
        "expected_recommendation": "skip",
        "expected_role_type": "platform-heavy AIE",
        "expected_ai_application": False,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "yes",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": False,
    },
    "REAL-020": {
        "source_company": "Build",
        "source_title": "AI Engineer - Assistant",
        "source_url": "https://jobs.ashbyhq.com/build/1eac54d0-ec02-401c-a6fc-42b41dfb74c4",
        "expected_recommendation": "skip",
        "expected_role_type": "Product-facing AIE",
        "expected_ai_application": True,
        "expected_product_facing": True,
        "expected_bay_area_or_remote": "no",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": True,
    },
    "REAL-026": {
        "source_company": "Kernel",
        "source_title": "Infrastructure Engineer",
        "source_url": "https://jobs.ashbyhq.com/usekernel/a4b11f7d-c748-4af7-9682-77f7ff297a9b",
        "expected_recommendation": "skip",
        "expected_role_type": "platform-heavy AIE",
        "expected_ai_application": False,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "yes",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": False,
    },
    "REAL-027": {
        "source_company": "Slate",
        "source_title": "Senior Software Engineer - Video Platform",
        "source_url": "https://jobs.ashbyhq.com/slate/4b5c45bc-9084-4bca-8293-53f424978714",
        "expected_recommendation": "skip",
        "expected_role_type": "not fit",
        "expected_ai_application": False,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "yes",
        "expected_sponsorship_available": "no",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": False,
    },
    "REAL-030": {
        "source_company": "OpenAI",
        "source_title": "Research Scientist",
        "source_url": "https://jobs.ashbyhq.com/openai/5f0c6579-0bfb-4a06-8a43-1dd371499e10",
        "expected_recommendation": "skip",
        "expected_role_type": "Research/ML role",
        "expected_ai_application": False,
        "expected_product_facing": False,
        "expected_bay_area_or_remote": "yes",
        "expected_work_authorization": "unknown",
        "expected_avoid_pure_infra": True,
    },
}


KEEP_SKIP_IDS = {
    "REAL-002", "REAL-005", "REAL-011", "REAL-012", "REAL-015", "REAL-018",
    "REAL-021", "REAL-024", "REAL-027", "REAL-028", "REAL-030",
}


# Human-audited corrections that must survive future source refreshes.
AUDITED_LABEL_CORRECTIONS: dict[str, dict[str, Any]] = {
    "REAL-006": {"expected_bay_area_or_remote": "unknown"},
    "REAL-007": {
        "expected_bay_area_or_remote": "no",
        "expected_recommendation": "skip",
    },
    "REAL-015": {"expected_ai_application": False},
}


def _new_case(
    jd_id: str,
    company: str,
    title: str,
    url: str,
    recommendation: str,
    sponsorship: str,
) -> dict[str, Any]:
    return {
        "jd_id": jd_id,
        "source_company": company,
        "source_title": title,
        "source_url": url,
        "retrieved_at": "2026-09-13",
        "label_status": "provisional",
        "expected_recommendation": recommendation,
        "expected_role_type": "Product-facing AIE",
        "expected_ai_application": True,
        "expected_product_facing": True,
        "expected_bay_area_or_remote": "yes",
        "expected_sponsorship_available": sponsorship,
        "expected_work_authorization": (
            "yes" if sponsorship == "yes" else "no" if sponsorship == "no" else "unknown"
        ),
        "expected_avoid_pure_infra": True,
        "job_description": "pending refresh",
    }


ADDITIONS = [
    _new_case("REAL-031", "Anthropic", "Applied AI Architect, Commercial", "https://job-boards.greenhouse.io/anthropic/jobs/5192805008", "apply", "yes"),
    _new_case("REAL-032", "Anthropic", "Applied AI Architect, Cyber", "https://job-boards.greenhouse.io/anthropic/jobs/5387733008", "apply", "yes"),
    _new_case("REAL-033", "Anthropic", "Applied AI Architect, Enterprise Tech", "https://job-boards.greenhouse.io/anthropic/jobs/5383335008", "apply", "yes"),
    _new_case("REAL-034", "Anthropic", "Applied AI Architect, Industries", "https://job-boards.greenhouse.io/anthropic/jobs/4461444008", "apply", "yes"),
    _new_case("REAL-035", "Anthropic", "Applied AI Architect, Partnerships", "https://job-boards.greenhouse.io/anthropic/jobs/5300430008", "apply", "yes"),
    _new_case("REAL-036", "Anthropic", "Applied AI Architect, Startups", "https://job-boards.greenhouse.io/anthropic/jobs/5406982008", "apply", "yes"),
    _new_case("REAL-037", "Anthropic", "Applied AI Architect, Strategic Enterprise Tech", "https://job-boards.greenhouse.io/anthropic/jobs/5409008008", "apply", "yes"),
    _new_case("REAL-038", "Anthropic", "Applied AI Engineer, Beneficial Deployments (Life Sciences)", "https://job-boards.greenhouse.io/anthropic/jobs/5021015008", "apply", "yes"),
    _new_case("REAL-039", "Anthropic", "Applied AI, Research Engineer", "https://job-boards.greenhouse.io/anthropic/jobs/5390811008", "apply", "yes"),
    _new_case("REAL-040", "OpenAI", "Applied AI Engineer, Startups", "https://jobs.ashbyhq.com/openai/71e7252f-abb1-4b74-8e69-318413042357", "human_review", "unknown"),
    _new_case("REAL-041", "OpenAI", "Product Engineer, Full Stack - Agents", "https://jobs.ashbyhq.com/openai/5ed99d32-eed1-4679-b7b4-037de073e57c", "human_review", "unknown"),
]


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "br", "div", "h1", "h2", "h3", "h4", "li", "p", "section", "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts)).replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(line for line in lines if line)


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _greenhouse_description(url: str) -> str:
    match = re.match(r"https://job-boards\.greenhouse\.io/([^/]+)/jobs/(\d+)", url)
    if not match:
        raise ValueError(f"Unsupported Greenhouse URL: {url}")
    board, job_id = match.groups()
    payload = _request_json(
        f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
    )
    parser = _TextExtractor()
    parser.feed(html.unescape(payload["content"]))
    location = payload.get("location", {}).get("name", "").strip()
    metadata = [f"Job title: {payload.get('title', '').strip()}"]
    if location:
        metadata.append(f"Location: {location}")
    return "\n".join(metadata) + "\n\n" + parser.text()


def _ashby_description(url: str, cache: dict[str, list[dict[str, Any]]]) -> str:
    match = re.match(r"https://jobs\.ashbyhq\.com/([^/]+)/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Unsupported Ashby URL: {url}")
    encoded_org, job_id = match.groups()
    org = unquote(encoded_org)
    if org not in cache:
        payload = _request_json(
            f"https://api.ashbyhq.com/posting-api/job-board/{quote(org, safe='')}"
        )
        cache[org] = payload.get("jobs", [])
    job = next((item for item in cache[org] if item.get("id") == job_id), None)
    if not job:
        raise ValueError(f"Ashby posting is not active: {url}")
    locations = [job.get("location", "").strip()]
    locations.extend(
        item.get("location", "").strip()
        for item in job.get("secondaryLocations", [])
    )
    locations = list(dict.fromkeys(location for location in locations if location))
    metadata = [f"Job title: {job.get('title', '').strip()}"]
    if locations:
        metadata.append(f"Location: {'; '.join(locations)}")
    return "\n".join(metadata) + "\n\n" + job["descriptionPlain"].strip()


def refresh(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["jd_id"] in {"REAL-001", "REAL-013"}:
            row["expected_recommendation"] = "apply"
    rows = [
        row
        for row in rows
        if row["expected_recommendation"] != "skip" or row["jd_id"] in KEEP_SKIP_IDS
    ]
    existing_ids = {row["jd_id"] for row in rows}
    rows.extend(case for case in ADDITIONS if case["jd_id"] not in existing_ids)
    rows.sort(key=lambda row: row["jd_id"])
    ashby_cache: dict[str, list[dict[str, Any]]] = {}
    updated: list[dict[str, Any]] = []

    for row in rows:
        replacement = REPLACEMENTS.get(row["jd_id"])
        if replacement:
            old_url = row["source_url"]
            row.update(replacement)
            row["replaced_source_url"] = old_url

        row.update(AUDITED_LABEL_CORRECTIONS.get(row["jd_id"], {}))

        sponsorship = row.get("expected_sponsorship_available", "unknown")
        citizenship_required = row.get(
            "expected_citizenship_or_green_card_required", "unknown"
        )
        if citizenship_required == "yes" or sponsorship == "no":
            row["expected_work_authorization"] = "no"
        elif sponsorship == "yes":
            row["expected_work_authorization"] = "yes"
        else:
            row["expected_work_authorization"] = "unknown"
        row.pop("expected_unrestricted_work_authorization_required", None)
        row.pop("expected_h1b_transfer_signal", None)

        url = row["source_url"]
        if "greenhouse.io" in url:
            body = _greenhouse_description(url)
            source = "greenhouse_official_api"
        elif "ashbyhq.com" in url:
            body = _ashby_description(url, ashby_cache)
            source = "ashby_official_api"
        else:
            raise ValueError(f"Unsupported source: {url}")

        if len(body) < 500:
            raise ValueError(f"Body is unexpectedly short for {row['jd_id']}: {len(body)}")
        row["job_description"] = body
        row["content_status"] = "full_job_posting_body"
        row["content_source"] = source
        row["retrieved_at"] = "2026-09-13"
        updated.append(row)

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in updated)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    refresh(args.path)


if __name__ == "__main__":
    main()
