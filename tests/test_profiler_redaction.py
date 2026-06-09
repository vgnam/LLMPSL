from __future__ import annotations

import unittest

from llm4ad.tools.profiler import ProfilerBase


class ProfilerRedactionTests(unittest.TestCase):
    def test_redacts_direct_and_nested_credentials(self):
        self.assertEqual(
            ProfilerBase._safe_parameter_value("_api_key", "sk-secret"),
            "<redacted>",
        )
        self.assertEqual(
            ProfilerBase._safe_parameter_value(
                "_kwargs",
                {"timeout": 30, "access_token": "secret-token"},
            ),
            {"timeout": 30, "access_token": "<redacted>"},
        )

    def test_keeps_non_sensitive_parameters(self):
        self.assertEqual(
            ProfilerBase._safe_parameter_value("_model", "openai/gpt-4o-mini"),
            "openai/gpt-4o-mini",
        )


if __name__ == "__main__":
    unittest.main()
