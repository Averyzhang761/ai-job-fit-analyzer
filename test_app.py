import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import analyzer
import app
import llm_client
from models import (
    CandidateLocationJudgment,
    JobFacts,
    Recommendation,
    TernarySignal,
)


JD = "Build AI workflows with customers. Remote in the US. Visa sponsorship is available."


def facts_data(
    *,
    remote_scope="us",
    sponsorship="offered",
    citizenship="not_stated",
    activities=None,
):
    return {
        "responsibilities": [
            {
                "quote": "Build AI workflows with customers.",
                "activities": activities or [
                    "customer_implementation",
                    "customer_advisory",
                ],
            }
        ],
        "location": {
            "work_arrangement": "remote",
            "stated_locations": ["United States"],
            "remote_scope": remote_scope,
        },
        "work_authorization": {
            "sponsorship": sponsorship,
            "citizenship_or_green_card": citizenship,
        },
        "risks": [],
        "confidence": 0.9,
    }


class JobAnalyzerTests(unittest.TestCase):
    def facts(self, **overrides):
        return JobFacts.model_validate(
            facts_data(**overrides),
            context={"job_description": JD},
        )

    def test_job_facts_resolve_source_id_and_validate_exact_quote(self):
        data = facts_data()
        data["responsibilities"][0]["quote"] = "E001"
        facts = JobFacts.model_validate(
            data,
            context={
                "job_description": JD,
                "evidence_map": {"E001": "Build AI workflows with customers."},
            },
        )
        self.assertEqual(
            facts.responsibilities[0].quote,
            "Build AI workflows with customers.",
        )

    def test_job_facts_reject_missing_source_quote(self):
        with self.assertRaisesRegex(ValueError, "Responsibility quote"):
            JobFacts.model_validate(
                facts_data(),
                context={"job_description": "Different JD text."},
            )

    def test_responsibility_activities_are_deduplicated(self):
        facts = self.facts(
            activities=["customer_advisory", "customer_advisory"]
        )
        self.assertEqual(
            [item.value for item in facts.responsibilities[0].activities],
            ["customer_advisory"],
        )

    def test_activity_tags_preserve_multiple_job_contents(self):
        decision = analyzer.decide_job(
            self.facts(),
            TernarySignal.YES,
            TernarySignal.YES,
        )
        self.assertEqual(
            [tag.value for tag in decision.activity_tags],
            ["customer_implementation", "customer_advisory"],
        )
        self.assertEqual(decision.recommendation, Recommendation.APPLY)

    def test_explicit_constraint_blocker_skips(self):
        decision = analyzer.decide_job(
            self.facts(sponsorship="not_offered"),
            TernarySignal.YES,
            TernarySignal.NO,
        )
        self.assertEqual(decision.recommendation, Recommendation.SKIP)

    def test_unknown_constraint_requires_human_review(self):
        decision = analyzer.decide_job(
            self.facts(sponsorship="not_stated"),
            TernarySignal.YES,
            TernarySignal.UNKNOWN,
        )
        self.assertEqual(decision.recommendation, Recommendation.HUMAN_REVIEW)
        self.assertTrue(decision.needs_human_review)

    def test_non_target_work_skips_after_constraints_pass(self):
        decision = analyzer.decide_job(
            self.facts(activities=["general_software"]),
            TernarySignal.YES,
            TernarySignal.YES,
        )
        self.assertEqual(decision.recommendation, Recommendation.SKIP)

    def test_work_authorization_uses_only_explicit_source_statements(self):
        self.assertEqual(
            analyzer.derive_work_authorization(self.facts()),
            TernarySignal.YES,
        )
        self.assertEqual(
            analyzer.derive_work_authorization(
                self.facts(citizenship="required")
            ),
            TernarySignal.NO,
        )
        self.assertEqual(
            analyzer.derive_work_authorization(
                self.facts(sponsorship="not_stated")
            ),
            TernarySignal.UNKNOWN,
        )

    def test_us_remote_is_decided_without_location_reviewer(self):
        self.assertEqual(
            analyzer.direct_location_eligibility(self.facts()),
            TernarySignal.YES,
        )

    def test_restricted_location_triggers_focused_reviewer(self):
        calls = []

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            calls.append(response_model)
            if response_model is JobFacts:
                return self.facts(remote_scope="restricted_regions")
            return CandidateLocationJudgment(
                candidate_location_eligible="yes",
                reason="San Francisco is listed.",
            )

        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_job(
                JD,
                call_model=fake_call,
                log_path=Path(directory) / "runs.jsonl",
            )

        self.assertEqual(calls, [JobFacts, CandidateLocationJudgment])
        self.assertEqual(result.recommendation, Recommendation.APPLY)

    def test_direct_location_uses_one_model_call_and_records_run(self):
        calls = []

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            calls.append(response_model)
            return self.facts()

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "runs.jsonl"
            result = analyzer.analyze_job(
                JD, call_model=fake_call, log_path=log_path
            )
            event = json.loads(log_path.read_text())

        self.assertEqual(calls, [JobFacts])
        self.assertEqual(result.recommendation, Recommendation.APPLY)
        self.assertFalse(event["location_reviewer_used"])
        self.assertEqual(event["config"]["schema_version"], "responsibility-evidence-v6")

    def test_openai_schema_avoids_provider_unsupported_unique_items(self):
        schema = llm_client.openai_compatible_json_schema(JobFacts)
        self.assertNotIn("uniqueItems", json.dumps(schema))

    def test_prompt_does_not_duplicate_schema_or_request_role_classification(self):
        self.assertNotIn("Return this JSON", llm_client.USER_PROMPT_TEMPLATE)
        self.assertIn("Do not classify the whole role", analyzer.EXTRACTION_PROMPT)
        self.assertNotIn("primary_deliverable", analyzer.EXTRACTION_PROMPT)

    @patch("llm_client.OpenAI")
    def test_client_rejects_truncated_output_before_pydantic(self, openai_class):
        choice = SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content='{"incomplete":'),
        )
        openai_class.return_value.chat.completions.create.return_value = (
            SimpleNamespace(choices=[choice])
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}, clear=True):
            with self.assertRaises(llm_client.ModelOutputTruncatedError):
                llm_client.call_structured_llm(
                    JD,
                    analyzer.EXTRACTION_PROMPT,
                    JobFacts,
                    "job_facts",
                    max_tokens=1200,
                )

    @patch("llm_client.OpenAI")
    def test_client_sends_generated_json_schema(self, openai_class):
        content = json.dumps(
            {
                **facts_data(),
                "responsibilities": [
                    {
                        "evidence_id": "E001",
                        "activities": ["customer_advisory"],
                    }
                ],
            }
        )
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=content),
        )
        create = openai_class.return_value.chat.completions.create
        create.return_value = SimpleNamespace(choices=[choice])

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}, clear=True):
            result = llm_client.call_structured_llm(
                JD,
                analyzer.EXTRACTION_PROMPT,
                JobFacts,
                "job_facts",
                max_tokens=1200,
            )

        request = create.call_args.kwargs
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            result.responsibilities[0].quote,
            "Build AI workflows with customers.",
        )

    def test_app_renders_responsibilities_and_validated_json(self):
        decision = analyzer.decide_job(
            self.facts(),
            TernarySignal.YES,
            TernarySignal.YES,
        )
        summary, responsibilities, risks, raw_json = app.render_analysis(decision)
        self.assertIn("Apply", summary)
        self.assertIn("Build AI workflows", responsibilities)
        self.assertEqual(json.loads(raw_json)["recommendation"], "apply")

    def test_real_dataset_keeps_33_complete_source_jobs(self):
        path = Path("data/eval_jds.real.jsonl")
        cases = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(cases), 33)
        self.assertEqual(len({case["source_url"] for case in cases}), 33)
        self.assertTrue(all(len(case["job_description"]) > 1000 for case in cases))


if __name__ == "__main__":
    unittest.main()
