import unittest

import app


class RegressionEvaluationTests(unittest.TestCase):
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

    def test_missing_h1b_information_is_normalized_to_unknown(self):
        result = {
            "role_type": "Internal-facing AIE",
            "recommendation": "maybe",
            "needs_human_review": False,
            "hard_filter_result": {"h1b_transfer_signal": "no"},
        }

        normalized = app.apply_business_rules(
            result,
            "The posting does not specify visa sponsorship or H-1B transfer support.",
        )

        self.assertEqual(normalized["hard_filter_result"]["h1b_transfer_signal"], "unknown")
        self.assertEqual(normalized["recommendation"], "human_review")
        self.assertTrue(normalized["needs_human_review"])

    def test_explicit_no_sponsorship_remains_no(self):
        result = {
            "role_type": "Internal-facing AIE",
            "recommendation": "skip",
            "needs_human_review": False,
            "hard_filter_result": {"h1b_transfer_signal": "no"},
        }

        normalized = app.apply_business_rules(
            result,
            "Candidates must be authorized to work without current or future sponsorship.",
        )

        self.assertEqual(normalized["hard_filter_result"]["h1b_transfer_signal"], "no")

    def test_compare_expected_fields_reports_nested_mismatches(self):
        case = {
            "expected_recommendation": "human_review",
            "expected_role_type": "Internal-facing AIE",
            "expected_product_facing": False,
            "expected_h1b_transfer_signal": "unknown",
        }
        result = {
            "recommendation": "skip",
            "role_type": "platform-heavy AIE",
            "hard_filter_result": {
                "product_facing": False,
                "h1b_transfer_signal": "no",
            },
        }

        mismatches = app.compare_expected_fields(case, result)

        self.assertEqual(
            mismatches,
            [
                "recommendation: expected human_review, got skip",
                "role_type: expected Internal-facing AIE, got platform-heavy AIE",
                "h1b_transfer_signal: expected unknown, got no",
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


if __name__ == "__main__":
    unittest.main()
