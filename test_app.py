import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import app
import analysis_service
import business_rules_v3
import business_rules_v4
import business_rules_v6
import conditional_pipeline_v5
import evaluation_v3
import evaluation_v4
import focused_pipeline
import llm_client
import prepare_v3_diagnostic_labels
import prepare_v4_diagnostic_labels
import run_focused_baseline
import run_role_experiment
import review_pipeline
import run_v3_baseline
import run_v3_ground_truth
import run_v4_ground_truth
import two_stage_pipeline_v5
import v6_pipeline
from pydantic import ValidationError
from models import (
    CandidateConstraintJudgmentV5,
    CandidateLocationJudgmentV5,
    ModelAssessmentV4,
    SourceFactsV5,
    SourceFactsV6,
    TernarySignal,
    WorkContentLevel,
)


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

    def valid_v3_data(self):
        return {
            "role": {
                "primary_deliverable": "ai_application",
                "delivers_ai_application": "yes",
                "trains_or_optimizes_models": "yes",
                "operates_ai_infrastructure": "no",
                "delivery": {
                    "customer_facing": "yes",
                    "product_collaboration": "yes",
                    "embedded_customer_implementation": "no",
                    "internal_user_facing": "no",
                },
            },
            "constraints": {
                "location": {
                    "work_arrangement": "remote",
                    "candidate_location_eligible": "yes",
                },
                "work_authorization": {
                    "sponsorship_available": "yes",
                    "citizenship_or_green_card_required": "no",
                },
            },
            "evidence": [{"field": "role_type", "quote": "Build AI products."}],
            "risks": [],
            "confidence": 0.9,
        }

    def valid_v4_data(self):
        return {
            "role": {
                "primary_deliverable": "ai_application",
                "work_content": {
                    "customer_implementation": "core",
                    "customer_advisory": "core",
                    "product_development": "present",
                    "internal_tools": "not_evidenced",
                    "model_engineering": "present",
                    "research": "not_evidenced",
                    "ai_infrastructure": "present",
                },
            },
            "constraints": {
                "location": {
                    "work_arrangement": "hybrid",
                    "candidate_location_eligible": "yes",
                },
                "work_authorization": {
                    "sponsorship_available": "yes",
                    "citizenship_or_green_card_required": "no",
                },
            },
            "evidence": [{"field": "role_type", "quote": "Build AI products."}],
            "risks": [],
            "confidence": 0.9,
        }

    def test_v4_keeps_multiple_work_contents_without_unique_role_type(self):
        facts = ModelAssessmentV4.model_validate(self.valid_v4_data())

        decision = business_rules_v4.decide_job_v4(facts)

        self.assertEqual(
            decision.role_tags,
            {
                "customer_implementation",
                "customer_advisory",
                "product_development",
                "model_engineering",
                "ai_infrastructure",
            },
        )
        self.assertEqual(decision.recommendation.value, "apply")
        self.assertFalse(hasattr(decision, "derived_role_type"))

    def test_v4_recommendation_uses_work_mix_not_a_role_label(self):
        data = self.valid_v4_data()
        for field in (
            "customer_implementation",
            "customer_advisory",
            "product_development",
        ):
            data["role"]["work_content"][field] = "present"

        decision = business_rules_v4.decide_job_v4(
            ModelAssessmentV4.model_validate(data)
        )

        self.assertEqual(decision.recommendation.value, "skip")

    def test_v4_diagnostic_drafts_are_provisional_and_round_trip(self):
        cases = prepare_v4_diagnostic_labels.build_cases()

        self.assertEqual(len(cases), 8)
        for case in cases:
            expected_status = (
                "confirmed"
                if case.jd_id in prepare_v4_diagnostic_labels.CONFIRMED_IDS
                else "provisional"
            )
            self.assertEqual(case.expected.label_status.value, expected_status)
            self.assertFalse(hasattr(case.expected, "role_type"))
            serialized = case.model_dump_json(exclude_computed_fields=True)
            reloaded = evaluation_v4.EvalCaseV4.model_validate_json(serialized)
            self.assertEqual(reloaded.expected_role_tags, case.expected_role_tags)

    def test_v4_runner_reports_work_content_mismatches_without_role_type(self):
        case = prepare_v4_diagnostic_labels.build_cases()[0]
        actual = ModelAssessmentV4.model_validate({
            "role": case.expected.role.model_dump(mode="json"),
            "constraints": case.expected.constraints.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "evidence": [{"field": "role_type", "quote": "evidence"}],
            "risks": [],
            "confidence": 0.9,
        })
        actual.role.work_content.customer_advisory = WorkContentLevel.PRESENT

        mismatches = run_v4_ground_truth.compare_case(case, actual)

        self.assertEqual(mismatches[0]["field"], "customer_advisory")
        self.assertNotIn("role_type", {item["field"] for item in mismatches})

    def test_v5_two_stage_keeps_extraction_and_policy_judgment_separate(self):
        calls = []
        v4 = self.valid_v4_data()

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            calls.append((response_model, kwargs.get("task_context")))
            if response_model is SourceFactsV5:
                return SourceFactsV5.model_validate({
                    "role": v4["role"],
                    "location": {
                        "work_arrangement": "remote",
                        "stated_locations": ["Anywhere, US"],
                        "remote_scope": "us",
                    },
                    "work_authorization": {
                        "sponsorship": "offered",
                        "citizenship_or_green_card": "not_stated",
                    },
                    "evidence": v4["evidence"],
                    "risks": [],
                    "confidence": 0.9,
                })
            return CandidateConstraintJudgmentV5.model_validate({
                "candidate_location_eligible": "yes",
                "work_authorization_eligible": "yes",
                "reasons": ["US remote and sponsorship offered."],
            })

        _, _, decision = two_stage_pipeline_v5.analyze_job_two_stage(
            "JD", call_structured=fake_call
        )

        self.assertEqual([call[0] for call in calls], [SourceFactsV5, CandidateConstraintJudgmentV5])
        self.assertIsNone(calls[0][1])
        self.assertIn("Extracted source facts", calls[1][1])
        self.assertEqual(decision.recommendation.value, "apply")

    def test_v5_conditional_pipeline_skips_reviewer_for_us_remote(self):
        v4 = self.valid_v4_data()
        calls = []

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            calls.append(response_model)
            return SourceFactsV5.model_validate({
                "role": v4["role"],
                "location": {
                    "work_arrangement": "remote",
                    "stated_locations": ["Anywhere, US"],
                    "remote_scope": "us",
                },
                "work_authorization": {
                    "sponsorship": "offered",
                    "citizenship_or_green_card": "not_stated",
                },
                "evidence": v4["evidence"],
                "risks": [],
                "confidence": 0.9,
            })

        _, location, work_auth, decision, review_used = (
            conditional_pipeline_v5.analyze_job_conditional(
                "JD", call_structured=fake_call
            )
        )

        self.assertEqual(calls, [SourceFactsV5])
        self.assertFalse(review_used)
        self.assertEqual(location.value, "yes")
        self.assertEqual(work_auth.value, "yes")
        self.assertEqual(decision.recommendation.value, "apply")

    def test_v5_conditional_pipeline_reviews_restricted_remote_location(self):
        v4 = self.valid_v4_data()

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            if response_model is SourceFactsV5:
                return SourceFactsV5.model_validate({
                    "role": v4["role"],
                    "location": {
                        "work_arrangement": "remote",
                        "stated_locations": ["Europe only"],
                        "remote_scope": "restricted_regions",
                    },
                    "work_authorization": {
                        "sponsorship": "not_stated",
                        "citizenship_or_green_card": "not_stated",
                    },
                    "evidence": v4["evidence"],
                    "risks": [],
                    "confidence": 0.9,
                })
            return CandidateLocationJudgmentV5.model_validate({
                "candidate_location_eligible": "no",
                "reason": "Europe-only restriction.",
            })

        _, location, work_auth, decision, review_used = (
            conditional_pipeline_v5.analyze_job_conditional(
                "JD", call_structured=fake_call
            )
        )

        self.assertTrue(review_used)
        self.assertEqual(location.value, "no")
        self.assertEqual(work_auth.value, "unknown")
        self.assertEqual(decision.recommendation.value, "skip")

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

    def test_job_analysis_accepts_a_valid_result(self):
        result = app.JobAnalysis.model_validate(self.valid_analysis_data())

        self.assertEqual(result.fit_score, 90)
        self.assertEqual(result.recommendation.value, "apply")

    def test_v3_accepts_overlapping_role_facts(self):
        data = self.valid_v3_data()
        data["role"]["primary_deliverable"] = "research"
        data["role"]["operates_ai_infrastructure"] = "yes"

        result = app.ModelAssessmentV3.model_validate(data)

        self.assertEqual(result.role.primary_deliverable.value, "research")
        self.assertEqual(result.role.delivers_ai_application.value, "yes")
        self.assertEqual(result.role.operates_ai_infrastructure.value, "yes")

    def test_v3_derives_one_primary_label_from_multiple_facts(self):
        facts = app.ModelAssessmentV3.model_validate(self.valid_v3_data())

        role_type = business_rules_v3.derive_role_type(facts)

        self.assertEqual(role_type.value, "Product-facing AIE")

        embedded_data = self.valid_v3_data()
        embedded_data["role"]["delivery"]["embedded_customer_implementation"] = "yes"
        embedded = app.ModelAssessmentV3.model_validate(embedded_data)
        self.assertEqual(
            business_rules_v3.derive_role_type(embedded).value,
            "FDE",
        )

    def test_v3_candidate_policy_decides_apply_review_and_skip(self):
        apply_facts = app.ModelAssessmentV3.model_validate(self.valid_v3_data())
        self.assertEqual(
            business_rules_v3.decide_job(apply_facts).recommendation.value,
            "apply",
        )

        review_data = self.valid_v3_data()
        review_data["constraints"]["work_authorization"]["sponsorship_available"] = "unknown"
        review_facts = app.ModelAssessmentV3.model_validate(review_data)
        self.assertEqual(
            business_rules_v3.decide_job(review_facts).recommendation.value,
            "human_review",
        )

        skip_data = self.valid_v3_data()
        skip_data["constraints"]["location"]["candidate_location_eligible"] = "no"
        skip_facts = app.ModelAssessmentV3.model_validate(skip_data)
        self.assertEqual(
            business_rules_v3.decide_job(skip_facts).recommendation.value,
            "skip",
        )

    def test_v3_separates_remote_arrangement_from_candidate_eligibility(self):
        data = self.valid_v3_data()
        data["constraints"]["location"] = {
            "work_arrangement": "remote",
            "candidate_location_eligible": "no",
        }

        facts = app.ModelAssessmentV3.model_validate(data)
        decision = business_rules_v3.decide_job(facts)

        self.assertEqual(facts.constraints.location.work_arrangement.value, "remote")
        self.assertEqual(decision.recommendation.value, "skip")

    def test_v3_prompt_reserves_no_for_source_supported_negatives(self):
        self.assertIn(
            "Use no only when the source directly supports",
            run_v3_baseline.V3_FACT_PROMPT,
        )
        self.assertIn(
            "Separately record the stated work arrangement",
            run_v3_baseline.V3_FACT_PROMPT,
        )

    def test_v3_schema_describes_decision_impacting_boundaries(self):
        schema = app.ModelAssessmentV3.model_json_schema()
        role = schema["$defs"]["RoleFactsV3"]["properties"]
        delivery = schema["$defs"]["DeliveryFacts"]["properties"]
        location = schema["$defs"]["LocationFactsV3"]["properties"]

        self.assertIn("dominant artifact", role["primary_deliverable"]["description"])
        self.assertIn("ordinary software", role["operates_ai_infrastructure"]["description"])
        self.assertIn("advising", delivery["embedded_customer_implementation"]["description"])
        self.assertIn("Remote alone", location["candidate_location_eligible"]["description"])

    def test_v3_projects_new_facts_into_the_existing_evaluation_contract(self):
        facts = app.ModelAssessmentV3.model_validate(self.valid_v3_data())

        result = run_v3_baseline.to_evaluation_result(
            business_rules_v3.decide_job(facts)
        )

        self.assertEqual(result["role_type"], "Product-facing AIE")
        self.assertTrue(result["hard_filter_result"]["ai_application"])
        self.assertTrue(result["hard_filter_result"]["product_facing"])
        self.assertTrue(result["hard_filter_result"]["avoid_pure_infra"])
        self.assertEqual(result["recommendation"], "apply")

    def test_v3_eval_derives_labels_from_human_reviewed_facts(self):
        data = self.valid_v3_data()
        case = evaluation_v3.EvalCaseV3.model_validate(
            {
                "jd_id": "REAL-TEST",
                "source_company": "Example",
                "source_title": "Applied AI Engineer",
                "source_url": "https://example.com/job",
                "job_description": "Build AI products.",
                "expected": {
                    "role": data["role"],
                    "constraints": data["constraints"],
                    "evidence_by_fact": {
                        "role.delivers_ai_application": ["Build AI products."]
                    },
                    "label_status": "confirmed",
                },
            }
        )

        self.assertEqual(case.expected_role_type, "Product-facing AIE")
        self.assertEqual(case.expected_recommendation().value, "apply")
        expected = case.model_dump(mode="json")["expected"]
        self.assertNotIn("expected_role_type", expected)
        self.assertNotIn("expected_recommendation", expected)

    def test_v3_eval_rejects_ground_truth_evidence_not_found_in_jd(self):
        data = self.valid_v3_data()

        with self.assertRaisesRegex(
            ValidationError, "Ground-truth evidence.*原始 JD"
        ):
            evaluation_v3.EvalCaseV3.model_validate(
                {
                    "jd_id": "REAL-TEST",
                    "source_company": "Example",
                    "source_title": "Applied AI Engineer",
                    "source_url": "https://example.com/job",
                    "job_description": "Build AI products.",
                    "expected": {
                        "role": data["role"],
                        "constraints": data["constraints"],
                        "evidence_by_fact": {
                            "role.delivers_ai_application": [
                                "This sentence is not in the JD."
                            ]
                        },
                        "label_status": "confirmed",
                    },
                }
            )

    def test_v3_diagnostic_migration_keeps_labels_provisional_and_derived(self):
        cases = prepare_v3_diagnostic_labels.build_cases()

        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case.jd_id for case in cases}), 8)
        for case in cases:
            expected_status = (
                "confirmed"
                if case.jd_id in prepare_v3_diagnostic_labels.CONFIRMED_IDS
                else "provisional"
            )
            self.assertEqual(case.expected.label_status.value, expected_status)
            self.assertNotIn("expected_role_type", case.expected.model_fields_set)
            self.assertNotIn("expected_recommendation", case.expected.model_fields_set)

            serialized = case.model_dump_json(exclude_computed_fields=True)
            reloaded = evaluation_v3.EvalCaseV3.model_validate_json(serialized)
            self.assertEqual(reloaded.expected_role_type, case.expected_role_type)

    def test_v3_ground_truth_runner_compares_facts_and_derived_outputs(self):
        case = prepare_v3_diagnostic_labels.build_cases()[0]
        matching = app.ModelAssessmentV3.model_validate(
            {
                "role": case.expected.role,
                "constraints": case.expected.constraints,
                "evidence": [{"field": "role_type", "quote": "evidence"}],
                "risks": [],
                "confidence": 0.9,
            }
        )
        self.assertEqual(
            run_v3_ground_truth.compare_v3_facts(case, matching), []
        )

        changed = matching.model_copy(deep=True)
        changed.role.delivery.embedded_customer_implementation = TernarySignal.NO
        mismatches = run_v3_ground_truth.compare_v3_facts(case, changed)
        fields = {mismatch["field"] for mismatch in mismatches}
        self.assertIn(
            "role.delivery.embedded_customer_implementation", fields
        )
        self.assertIn("derived_role_type", fields)

    def test_v3_ground_truth_report_separates_fact_and_decision_quality(self):
        cases = prepare_v3_diagnostic_labels.build_cases()[:2]
        matching = [
            app.ModelAssessmentV3.model_validate(
                {
                    "role": case.expected.role.model_dump(mode="json"),
                    "constraints": case.expected.constraints.model_dump(
                        mode="json", exclude_computed_fields=True
                    ),
                    "evidence": [{"field": "role_type", "quote": "evidence"}],
                    "risks": [],
                    "confidence": 0.9,
                }
            )
            for case in cases
        ]
        matching[1].role.operates_ai_infrastructure = TernarySignal.YES
        results = iter(matching)

        report = run_v3_ground_truth.evaluate_cases(cases, lambda _: next(results))

        self.assertEqual(report["metrics"]["exact_fact_rows"], 1)
        self.assertEqual(report["metrics"]["derived_role_rows"], 2)
        self.assertEqual(report["metrics"]["recommendation_rows"], 2)
        self.assertEqual(report["metrics"]["decision_rows"], 2)

    def test_v3_ground_truth_runner_selects_requested_ids(self):
        cases = prepare_v3_diagnostic_labels.build_cases()

        selected = run_v3_ground_truth.select_cases(
            cases, ids=["REAL-013", "REAL-029"]
        )

        self.assertEqual([case.jd_id for case in selected], ["REAL-013", "REAL-029"])

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

    def test_openai_schema_inlines_refs_that_have_field_descriptions(self):
        schema = llm_client.openai_compatible_json_schema(app.ModelAssessmentV3)
        product_collaboration = schema["$defs"]["DeliveryFacts"]["properties"][
            "product_collaboration"
        ]

        self.assertNotIn("$ref", product_collaboration)
        self.assertEqual(product_collaboration["type"], "string")
        self.assertIn("product team", product_collaboration["description"])

        bare_ref = schema["properties"]["role"]
        self.assertEqual(bare_ref, {"$ref": "#/$defs/RoleFactsV3"})

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

    def test_focused_pipeline_uses_three_schemas_and_existing_business_rules(self):
        jd = "Build AI workflows with customers. Remote in the US. Visa sponsorship."
        requested_models = []

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            self.assertEqual(job_description, jd)
            requested_models.append(response_model)
            common_evidence = [
                {"field": "role_type", "quote": "Build AI workflows with customers."}
            ]
            if response_model is focused_pipeline.RoleFacts:
                return response_model.model_validate(
                    {
                        "role_type": "Product-facing AIE",
                        "ai_application": True,
                        "product_facing": True,
                        "avoid_pure_infra": True,
                        "evidence": common_evidence,
                        "risks": [],
                        "confidence": 0.9,
                    }
                )
            if response_model is focused_pipeline.LocationFacts:
                return response_model.model_validate(
                    {
                        "bay_area_or_remote": "yes",
                        "evidence": [
                            {"field": "location", "quote": "Remote in the US."}
                        ],
                        "confidence": 0.95,
                    }
                )
            return response_model.model_validate(
                {
                    "work_authorization": {
                        "sponsorship_available": "yes",
                        "citizenship_or_green_card_required": "unknown",
                    },
                    "evidence": [
                        {
                            "field": "work_authorization",
                            "quote": "Visa sponsorship.",
                        }
                    ],
                    "confidence": 0.85,
                }
            )

        result = focused_pipeline.analyze_job_focused(
            jd, call_structured=fake_call
        )

        self.assertEqual(
            requested_models,
            [
                focused_pipeline.RoleFacts,
                focused_pipeline.LocationFacts,
                focused_pipeline.WorkAuthorizationFacts,
            ],
        )
        self.assertEqual(result.recommendation.value, "apply")
        self.assertEqual(result.confidence, 0.85)

    def test_hybrid_pipeline_uses_two_calls_and_existing_business_rules(self):
        jd = "Build AI workflows. Remote in the US. Visa sponsorship."
        requested_models = []

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            requested_models.append(response_model)
            if response_model is focused_pipeline.CoreFacts:
                return response_model.model_validate(
                    {
                        "role_type": "Product-facing AIE",
                        "ai_application": True,
                        "product_facing": True,
                        "avoid_pure_infra": True,
                        "work_authorization": {
                            "sponsorship_available": "yes",
                            "citizenship_or_green_card_required": "unknown",
                        },
                        "evidence": [
                            {"field": "role_type", "quote": "Build AI workflows."},
                            {
                                "field": "work_authorization",
                                "quote": "Visa sponsorship.",
                            },
                        ],
                        "risks": [],
                        "confidence": 0.9,
                    }
                )
            return response_model.model_validate(
                {
                    "bay_area_or_remote": "yes",
                    "evidence": [
                        {"field": "location", "quote": "Remote in the US."}
                    ],
                    "confidence": 0.95,
                }
            )

        result = focused_pipeline.analyze_job_hybrid(
            jd, call_structured=fake_call
        )

        self.assertEqual(
            requested_models,
            [focused_pipeline.CoreFacts, focused_pipeline.LocationFacts],
        )
        self.assertEqual(result.recommendation.value, "apply")
        self.assertEqual(result.confidence, 0.9)

    def test_review_pipeline_records_corrections_before_business_rules(self):
        jd = "Build AI workflows with customers. Remote in the US. Visa sponsorship."
        initial_data = self.valid_model_data()
        initial_data["role_type"] = "Internal-facing AIE"
        initial_data["hard_filter_result"]["product_facing"] = False
        reviewed_data = self.valid_model_data()
        calls = []

        def fake_call(job_description, prompt, response_model, schema_name, **kwargs):
            calls.append((schema_name, kwargs.get("task_context")))
            data = initial_data if len(calls) == 1 else reviewed_data
            return response_model.model_validate(data)

        outcome = review_pipeline.analyze_job_reviewed(
            jd, call_structured=fake_call
        )

        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0][1])
        self.assertIn("Candidate extraction to audit", calls[1][1])
        self.assertEqual(
            outcome.changed_fields,
            ["role_type", "hard_filter_result.product_facing"],
        )
        self.assertEqual(outcome.final.recommendation.value, "apply")

    def test_focused_runner_can_select_specific_jd_ids(self):
        cases = [{"jd_id": "REAL-006"}, {"jd_id": "REAL-012"}, {"jd_id": "REAL-018"}]

        selected = run_focused_baseline.select_cases(
            cases, ids=["REAL-006", "REAL-018"]
        )

        self.assertEqual([case["jd_id"] for case in selected], ["REAL-006", "REAL-018"])

    def test_focused_location_prompt_does_not_use_a_city_allowlist(self):
        self.assertNotIn("Emeryville", focused_pipeline.LOCATION_PROMPT)

    def test_role_experiment_compares_only_role_facts(self):
        case = {
            "expected_role_type": "Product-facing AIE",
            "expected_ai_application": True,
            "expected_product_facing": True,
            "expected_avoid_pure_infra": True,
        }
        facts = focused_pipeline.RoleFacts.model_validate(
            {
                "role_type": "FDE",
                "ai_application": True,
                "product_facing": True,
                "avoid_pure_infra": True,
                "evidence": [{"field": "role_type", "quote": "Customer work."}],
                "risks": [],
                "confidence": 0.9,
            }
        )

        self.assertEqual(
            run_role_experiment.compare_role_facts(case, facts),
            ["role_type: expected Product-facing AIE, got FDE"],
        )

    def test_focused_role_prompt_does_not_name_evaluation_cases(self):
        self.assertNotIn("REAL-", focused_pipeline.ROLE_PROMPT)
        self.assertNotIn("Anthropic", focused_pipeline.ROLE_PROMPT)

if __name__ == "__main__":
    unittest.main()
