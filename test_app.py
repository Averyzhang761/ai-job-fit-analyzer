import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import app
import analysis_service
import business_rules_v6
import llm_client
import v6_pipeline
from pydantic import ValidationError
from models import SourceFactsV6, TernarySignal


class RegressionEvaluationTests(unittest.TestCase):
    def valid_analysis_data(self):
        return {
            "role_type": "Product-facing AIE",
            "fit_score": 90,
            "score_reasons": ["Role type base: 65", "+10: AI application work"],
            "hard_filter_result": {
                "ai_application": True,
                "product_facing": True,
                "bay_area_or_remote": "yes",
                "work_authorization": {
                    "sponsorship_available": "yes",
                    "citizenship_or_green_card_required": "no",
                },
                "avoid_pure_infra": True,
            },
            "evidence": [
                {
                    "field": "role_type",
                    "quote": "Works directly with users to build AI workflows.",
                }
            ],
            "risks": [],
            "confidence": 0.95,
            "needs_human_review": False,
            "recommendation": "apply",
        }

    def valid_model_data(self):
        data = self.valid_analysis_data()
        for field in (
            "fit_score",
            "score_reasons",
            "needs_human_review",
            "recommendation",
        ):
            del data[field]
        return data

    def test_v6_derives_multiple_activities_from_source_backed_responsibilities(self):
        facts = SourceFactsV6.model_validate(
            {
                "responsibilities": [
                    {
                        "quote": "Build AI workflows with customer product teams.",
                        "activities": [
                            "customer_implementation",
                            "customer_advisory",
                            "product_development",
                        ],
                    }
                ],
                "location": {
                    "work_arrangement": "remote",
                    "stated_locations": ["Anywhere, US"],
                    "remote_scope": "us",
                },
                "work_authorization": {
                    "sponsorship": "offered",
                    "citizenship_or_green_card": "not_stated",
                },
                "risks": [],
                "confidence": 0.9,
            },
            context={
                "job_description": "Build AI workflows with customer product teams."
            },
        )

        decision = business_rules_v6.decide_job_v6(
            facts, TernarySignal.YES, TernarySignal.YES
        )

        self.assertEqual(
            {tag.value for tag in decision.activity_tags},
            {
                "customer_implementation",
                "customer_advisory",
                "product_development",
            },
        )
        self.assertEqual(decision.recommendation.value, "apply")
        self.assertFalse(hasattr(decision, "primary_deliverable"))

    def test_v6_rejects_responsibility_quote_missing_from_jd(self):
        with self.assertRaisesRegex(ValidationError, "Responsibility quote"):
            SourceFactsV6.model_validate(
                {
                    "responsibilities": [
                        {"quote": "Invented duty.", "activities": ["research"]}
                    ],
                    "location": {
                        "work_arrangement": "unknown",
                        "stated_locations": [],
                        "remote_scope": "unknown",
                    },
                    "work_authorization": {
                        "sponsorship": "not_stated",
                        "citizenship_or_green_card": "not_stated",
                    },
                    "risks": [],
                    "confidence": 0.8,
                },
                context={"job_description": "A different responsibility."},
            )

    def test_v6_prompt_has_no_role_or_importance_classification(self):
        self.assertIn("exact source chunk ID", v6_pipeline.RESPONSIBILITY_PROMPT)
        self.assertNotIn(" as core", v6_pipeline.RESPONSIBILITY_PROMPT)
        self.assertNotIn(" as present", v6_pipeline.RESPONSIBILITY_PROMPT)
        self.assertNotIn("primary_deliverable", v6_pipeline.RESPONSIBILITY_PROMPT)

    def test_v6_openai_schema_does_not_require_unique_items(self):
        schema = llm_client.openai_compatible_json_schema(SourceFactsV6)

        self.assertNotIn("uniqueItems", json.dumps(schema))

    def test_v6_derives_work_authorization_from_source_statements(self):
        facts = SourceFactsV6.model_validate(
            {
                "responsibilities": [
                    {"quote": "Build AI products.", "activities": ["product_development"]}
                ],
                "location": {
                    "work_arrangement": "remote",
                    "stated_locations": ["United States"],
                    "remote_scope": "us",
                },
                "work_authorization": {
                    "sponsorship": "offered",
                    "citizenship_or_green_card": "not_stated",
                },
                "risks": [],
                "confidence": 0.9,
            }
        )

        self.assertEqual(
            v6_pipeline.derive_work_authorization(facts), TernarySignal.YES
        )
        self.assertEqual(
            v6_pipeline.direct_location_eligibility(facts), TernarySignal.YES
        )

    def test_job_analysis_accepts_a_valid_result(self):
        result = app.JobAnalysis.model_validate(self.valid_analysis_data())

        self.assertEqual(result.fit_score, 90)
        self.assertEqual(result.recommendation.value, "apply")

    def test_status_error_detail_extracts_provider_message(self):
        error = SimpleNamespace(
            body={"error": {"message": "Invalid response schema"}}
        )

        self.assertEqual(
            llm_client._status_error_detail(error), "Invalid response schema"
        )

    def test_status_error_detail_prefers_nested_provider_reason(self):
        error = SimpleNamespace(
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"raw": "Schema rejected: unsupported keyword"},
                }
            }
        )

        self.assertEqual(
            llm_client._status_error_detail(error),
            "Schema rejected: unsupported keyword",
        )

    def test_job_analysis_rejects_a_missing_required_field(self):
        data = self.valid_analysis_data()
        del data["role_type"]

        with self.assertRaises(ValidationError):
            app.JobAnalysis.model_validate(data)

    def test_job_analysis_rejects_an_unknown_recommendation(self):
        data = self.valid_analysis_data()
        data["recommendation"] = "consider"

        with self.assertRaises(ValidationError):
            app.JobAnalysis.model_validate(data)

    def test_job_analysis_rejects_an_out_of_range_score(self):
        data = self.valid_analysis_data()
        data["fit_score"] = 120

        with self.assertRaises(ValidationError):
            app.JobAnalysis.model_validate(data)

    def test_job_analysis_rejects_extra_fields(self):
        data = self.valid_analysis_data()
        data["candidate_age"] = 35

        with self.assertRaises(ValidationError):
            app.JobAnalysis.model_validate(data)

    def test_job_analysis_cleans_evidence_and_rejects_empty_items(self):
        data = self.valid_analysis_data()
        data["evidence"] = [
            {"field": "product_facing", "quote": "  Direct user collaboration.  "}
        ]

        result = app.JobAnalysis.model_validate(data)

        self.assertEqual(result.evidence[0].quote, "Direct user collaboration.")

    def test_job_analysis_rejects_conflicting_review_state(self):
        data = self.valid_analysis_data()
        data["needs_human_review"] = True
        data["recommendation"] = "skip"

        with self.assertRaises(ValidationError):
            app.JobAnalysis.model_validate(data)

    def test_job_analysis_rejects_product_facing_platform_role(self):
        data = self.valid_analysis_data()
        data["role_type"] = "platform-heavy AIE"
        data["hard_filter_result"]["product_facing"] = True
        data["hard_filter_result"]["avoid_pure_infra"] = False

        with self.assertRaises(ValidationError):
            app.JobAnalysis.model_validate(data)

    def test_parse_job_analysis_json_returns_validated_plain_data(self):
        model_data = self.valid_model_data()
        model_text = json.dumps(model_data)

        result = app.parse_job_analysis_json(model_text)

        self.assertEqual(result.role_type.value, "Product-facing AIE")
        self.assertFalse(hasattr(result, "recommendation"))

    def test_parse_job_analysis_json_rejects_markdown_wrapping(self):
        model_text = "```json\n" + json.dumps(self.valid_analysis_data()) + "\n```"

        with self.assertRaises(ValidationError):
            app.parse_job_analysis_json(model_text)

    def test_job_analysis_schema_contains_contract_constraints(self):
        schema = app.ModelAssessment.model_json_schema()

        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("fit_score", schema["properties"])
        self.assertIn(
            "work_authorization",
            schema["$defs"]["HardFilterResult"]["properties"],
        )

    def test_model_assessment_rejects_evidence_not_found_in_the_jd(self):
        model_text = json.dumps(self.valid_model_data())

        with self.assertRaisesRegex(ValidationError, "逐字存在"):
            app.parse_job_analysis_json(model_text, "A different job description.")

    def test_evidence_chunks_preserve_exact_source_text(self):
        annotated, evidence_map = llm_client.build_evidence_chunks(
            "Location: San Francisco, CA\n\nBuild AI tools. Work with customers."
        )

        self.assertIn("[E001] Location: San Francisco, CA", annotated)
        self.assertEqual(evidence_map["E002"], "Build AI tools.")
        self.assertEqual(evidence_map["E003"], "Work with customers.")

    def test_model_assessment_resolves_valid_evidence_ids_and_rejects_unknown_ids(self):
        model_data = self.valid_model_data()
        model_data["evidence"] = [{"field": "role_type", "evidence_id": "E001"}]
        model_text = json.dumps(model_data)

        result = llm_client.parse_model_assessment_json(
            model_text,
            evidence_map={"E001": "Exact source sentence."},
        )
        self.assertEqual(result.evidence[0].quote, "Exact source sentence.")

        with self.assertRaisesRegex(ValidationError, "evidence_id"):
            llm_client.parse_model_assessment_json(
                model_text,
                evidence_map={"E002": "A different sentence."},
            )

    def test_business_rules_calculate_score_from_validated_facts(self):
        assessment = app.ModelAssessment.model_validate(self.valid_model_data())

        result = app.finalize_assessment(assessment)

        self.assertEqual(result.fit_score, 100)
        self.assertEqual(result.recommendation.value, "apply")
        self.assertIn("Role type base: 65", result.score_reasons)

    def test_explicit_visa_sponsorship_can_produce_apply(self):
        assessment = app.ModelAssessment.model_validate(self.valid_model_data())

        result = app.finalize_assessment(assessment)

        self.assertFalse(result.needs_human_review)
        self.assertEqual(result.recommendation.value, "apply")

    def test_unknown_work_authorization_requires_human_review(self):
        model_data = self.valid_model_data()
        work_auth = model_data["hard_filter_result"]["work_authorization"]
        work_auth["sponsorship_available"] = "unknown"
        assessment = app.ModelAssessment.model_validate(model_data)

        result = app.finalize_assessment(assessment)

        self.assertTrue(result.needs_human_review)
        self.assertEqual(result.recommendation.value, "human_review")

    def test_recommendation_has_only_three_terminal_states(self):
        self.assertEqual(
            {item.value for item in app.Recommendation},
            {"apply", "skip", "human_review"},
        )

    def test_unknown_location_requires_human_review(self):
        model_data = self.valid_model_data()
        model_data["role_type"] = "Internal-facing AIE"
        model_data["hard_filter_result"]["product_facing"] = False
        model_data["hard_filter_result"]["bay_area_or_remote"] = "unknown"
        assessment = app.ModelAssessment.model_validate(model_data)

        result = app.finalize_assessment(assessment)

        self.assertTrue(result.needs_human_review)
        self.assertEqual(result.recommendation.value, "human_review")

    def test_analyze_jd_renders_a_pydantic_validated_result(self):
        assessment = app.ModelAssessment.model_validate(self.valid_model_data())
        with patch("app.call_llm", return_value=assessment):
            outputs = app.analyze_jd(
                "A product-facing AI role. Visa sponsorship available."
            )

        self.assertIn("Product-facing AIE", outputs[0])
        self.assertIn('"recommendation": "apply"', outputs[3])

    def test_analyze_jd_rejects_markdown_wrapped_json(self):
        try:
            app.ModelAssessment.model_validate({})
        except ValidationError as invalid_output:
            contract_error = invalid_output

        with patch("app.call_llm", side_effect=contract_error):
            outputs = app.analyze_jd("A product-facing AI role")

        self.assertIn("数据契约", outputs[0])
        self.assertEqual(outputs[3], "")

    @patch.dict(
        os.environ,
        {
            "RUNPOD_BASE_URL": "https://example.test",
            "RUNPOD_API_KEY": "secret",
            "RUNPOD_MODEL": "test-model",
        },
    )
    @patch("llm_client.OpenAI")
    def test_call_llm_uses_openai_sdk_with_generated_schema(self, openai_class):
        client = openai_class.return_value
        model_data = self.valid_model_data()
        model_data["evidence"] = [{"field": "role_type", "evidence_id": "E001"}]
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(model_data)),
                )
            ]
        )

        result = app.call_llm("A job description")

        self.assertIsInstance(result, app.ModelAssessment)
        self.assertEqual(result.role_type.value, "Product-facing AIE")
        openai_class.assert_called_once_with(
            base_url="https://example.test/v1",
            api_key="secret",
            timeout=90,
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(
            request["response_format"]["json_schema"]["schema"],
            app.ModelAssessment.model_json_schema(),
        )
        self.assertIn("[E001] A job description", request["messages"][1]["content"])
        evidence_schema = request["response_format"]["json_schema"]["schema"]["$defs"]["EvidenceItem"]
        self.assertIn("evidence_id", evidence_schema["properties"])
        self.assertNotIn("quote", evidence_schema["properties"])

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "openrouter-secret",
            "OPENROUTER_MODEL": "openai/gpt-4.1-mini",
        },
        clear=True,
    )
    @patch("llm_client.OpenAI")
    def test_call_llm_uses_openrouter_and_requires_schema_support(self, openai_class):
        client = openai_class.return_value
        model_data = self.valid_model_data()
        model_data["evidence"] = [{"field": "role_type", "evidence_id": "E001"}]
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(model_data)),
                )
            ]
        )

        app.call_llm("A job description")

        openai_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="openrouter-secret",
            timeout=90,
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-4.1-mini")
        self.assertEqual(
            request["extra_body"], {"provider": {"require_parameters": True}}
        )

    @patch.dict(
        os.environ,
        {"RUNPOD_BASE_URL": "https://example.test", "RUNPOD_API_KEY": "secret"},
    )
    @patch("llm_client.OpenAI")
    def test_call_llm_rejects_truncated_output_before_parsing(self, openai_class):
        client = openai_class.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content='{"role_type":'),
                )
            ]
        )

        with self.assertRaisesRegex(app.ModelOutputTruncatedError, "截断"):
            app.call_llm("A job description")

    @patch.dict(
        os.environ,
        {"RUNPOD_BASE_URL": "https://example.test", "RUNPOD_API_KEY": "secret"},
    )
    @patch("llm_client.OpenAI")
    def test_call_llm_rejects_empty_content_as_response_error(self, openai_class):
        client = openai_class.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=""),
                )
            ]
        )

        with self.assertRaises(app.ModelResponseError):
            app.call_llm("A job description")

    def test_user_prompt_does_not_duplicate_the_json_schema(self):
        self.assertNotIn("Return this JSON shape", app.USER_PROMPT_TEMPLATE)
        self.assertNotIn('"role_type"', app.USER_PROMPT_TEMPLATE)

    def test_system_prompt_defines_san_francisco_as_bay_area(self):
        self.assertIn("Francisco Bay Area", app.SYSTEM_PROMPT)
        self.assertIn('bay_area_or_remote="yes"', app.SYSTEM_PROMPT)

    def test_system_prompt_defines_sponsorship_support(self):
        self.assertIn('sponsorship_available="yes"', app.SYSTEM_PROMPT)

    def test_work_authorization_status_is_derived_from_source_facts(self):
        supported = app.WorkAuthorizationResult.model_validate(
            {
                "sponsorship_available": "yes",
                "citizenship_or_green_card_required": "no",
            }
        )
        blocked = app.WorkAuthorizationResult.model_validate(
            {
                "sponsorship_available": "unknown",
                "citizenship_or_green_card_required": "yes",
            }
        )

        self.assertEqual(supported.status.value, "yes")
        self.assertEqual(blocked.status.value, "no")
        schema_text = json.dumps(app.ModelAssessment.model_json_schema())
        self.assertNotIn("h1b_transfer_signal", schema_text)
        self.assertNotIn("unrestricted_work_authorization_required", schema_text)
        self.assertNotIn("/no_think", app.USER_PROMPT_TEMPLATE)

    def test_temperature_is_fixed_configuration_not_a_user_control(self):
        component_labels = {
            component.get("props", {}).get("label")
            for component in app.demo.config["components"]
        }

        self.assertNotIn("Temperature", component_labels)

    def test_business_rules_derive_human_review_instead_of_using_model_terminal_state(self):
        data = self.valid_model_data()
        data["role_type"] = "Internal-facing AIE"
        data["hard_filter_result"]["product_facing"] = False
        data["hard_filter_result"]["bay_area_or_remote"] = "unknown"
        data["hard_filter_result"]["work_authorization"]["sponsorship_available"] = "unknown"

        normalized = app.finalize_assessment(app.ModelAssessment.model_validate(data))

        self.assertEqual(normalized.recommendation.value, "human_review")
        self.assertTrue(normalized.needs_human_review)

    def test_regression_report_is_outside_input_panel(self):
        config = app.demo.config
        components = {component["id"]: component for component in config["components"]}
        parents = {}

        def walk(node):
            for child in node.get("children", []):
                parents[child["id"]] = node["id"]
                walk(child)

        walk(config["layout"])
        input_panel_id = next(
            component_id
            for component_id, component in components.items()
            if component.get("props", {}).get("elem_classes") == ["input-panel"]
        )
        regression_button_id = next(
            component_id
            for component_id, component in components.items()
            if component.get("props", {}).get("value") == "Run regression suite"
        )
        ancestors = []
        current = regression_button_id
        while current in parents:
            current = parents[current]
            ancestors.append(current)

        self.assertNotIn(input_panel_id, ancestors)

    def test_detail_row_is_hidden_before_analysis(self):
        detail_row = next(
            component
            for component in app.demo.config["components"]
            if component.get("props", {}).get("elem_classes") == ["detail-row"]
        )

        self.assertFalse(detail_row["props"]["visible"])

    def test_explicit_no_sponsorship_is_a_deterministic_blocker(self):
        data = self.valid_model_data()
        data["role_type"] = "Internal-facing AIE"
        data["hard_filter_result"]["product_facing"] = False
        data["hard_filter_result"]["work_authorization"]["sponsorship_available"] = "no"

        result = app.finalize_assessment(app.ModelAssessment.model_validate(data))

        self.assertEqual(result.recommendation.value, "skip")
        self.assertLessEqual(result.fit_score, 20)

    def test_compare_expected_fields_reports_nested_mismatches(self):
        case = {
            "expected_recommendation": "human_review",
            "expected_role_type": "Internal-facing AIE",
            "expected_product_facing": False,
            "expected_work_authorization": "yes",
            "expected_sponsorship_available": "yes",
        }
        result = {
            "recommendation": "skip",
            "role_type": "platform-heavy AIE",
            "hard_filter_result": {
                "product_facing": False,
                "work_authorization": {
                    "status": "no",
                    "sponsorship_available": "no",
                },
            },
        }

        mismatches = app.compare_expected_fields(case, result)

        self.assertEqual(
            mismatches,
            [
                "recommendation: expected human_review, got skip",
                "role_type: expected Internal-facing AIE, got platform-heavy AIE",
                "status: expected yes, got no",
                "sponsorship_available: expected yes, got no",
            ],
        )

    def test_run_regression_continues_after_case_error(self):
        cases = [
            {"jd_id": "JD-1", "expected_recommendation": "apply", "job_description": "good"},
            {"jd_id": "JD-2", "expected_recommendation": "skip", "job_description": "bad"},
        ]

        def analyze(case):
            if case["jd_id"] == "JD-2":
                raise RuntimeError("service unavailable")
            return {"recommendation": "apply", "hard_filter_result": {}}

        report = app.evaluate_regression_cases(cases, analyze)

        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["rows"][1]["status"], "ERROR")
        self.assertIn("service unavailable", report["rows"][1]["details"])
        self.assertEqual(report["field_failures"]["runtime_error"], 1)

    def test_analysis_service_records_reproducible_success(self):
        assessment = app.ModelAssessment.model_validate(self.valid_model_data())
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "runs.jsonl")

            result = analysis_service.analyze_job(
                "Works directly with users to build AI workflows.",
                app.SYSTEM_PROMPT,
                call_model=lambda *args, **kwargs: assessment,
                log_path=log_path,
            )

            with open(log_path, encoding="utf-8") as handle:
                event = json.loads(handle.readline())

        self.assertEqual(result.recommendation.value, "apply")
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["final_result"]["fit_score"], 100)
        self.assertEqual(event["config"]["temperature"], 0.0)
        self.assertNotIn("api_key", event["config"])

    def test_analysis_service_records_typed_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "runs.jsonl")

            with self.assertRaises(app.ModelOutputTruncatedError):
                analysis_service.analyze_job(
                    "A job description",
                    app.SYSTEM_PROMPT,
                    call_model=lambda *args, **kwargs: (_ for _ in ()).throw(
                        app.ModelOutputTruncatedError("truncated")
                    ),
                    log_path=log_path,
                )

            with open(log_path, encoding="utf-8") as handle:
                event = json.loads(handle.readline())

        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error_type"], "output_truncated")

    def test_real_eval_set_contains_full_official_job_bodies(self):
        with open("data/eval_jds.real.jsonl", encoding="utf-8") as handle:
            cases = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(len(cases), 33)
        self.assertEqual(len({case["source_url"] for case in cases}), 33)
        self.assertEqual(
            {
                recommendation: sum(
                    case["expected_recommendation"] == recommendation
                    for case in cases
                )
                for recommendation in ("apply", "skip", "human_review")
            },
            {"apply": 11, "skip": 12, "human_review": 10},
        )
        for case in cases:
            self.assertEqual(case["content_status"], "full_job_posting_body")
            self.assertIn(
                case["content_source"],
                {"greenhouse_official_api", "ashby_official_api"},
            )
            self.assertGreaterEqual(len(case["job_description"]), 500)

        first_case = next(case for case in cases if case["jd_id"] == "REAL-001")
        self.assertIn(
            "Location: New York City, NY; San Francisco, CA; Seattle, WA",
            first_case["job_description"],
        )

    def test_audited_real_eval_labels_are_corrected(self):
        with open("data/eval_jds.real.jsonl", encoding="utf-8") as handle:
            cases = {row["jd_id"]: row for row in map(json.loads, handle)}

        self.assertEqual(cases["REAL-006"]["expected_bay_area_or_remote"], "unknown")
        self.assertEqual(cases["REAL-007"]["expected_bay_area_or_remote"], "no")
        self.assertEqual(cases["REAL-007"]["expected_recommendation"], "skip")
        self.assertFalse(cases["REAL-015"]["expected_ai_application"])

    def test_system_prompt_keeps_applied_ai_titles_out_of_fde_by_default(self):
        self.assertIn("explicitly titled Forward Deployed Engineer", app.SYSTEM_PROMPT)
        self.assertIn("Applied AI Engineer or Applied AI Architect", app.SYSTEM_PROMPT)

    def test_system_prompt_distinguishes_platform_and_research_from_ai_apps(self):
        self.assertIn("platform-heavy AIE, set ai_application=false", app.SYSTEM_PROMPT)
        self.assertIn("Research-only work is not pure infrastructure", app.SYSTEM_PROMPT)

if __name__ == "__main__":
    unittest.main()
