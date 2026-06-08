from __future__ import annotations

import unittest

from llm4ad.tools.evaluate_code_diversity import canonicalize_code, evaluate_code_diversity


class CodeDiversityTests(unittest.TestCase):
    def test_canonicalize_removes_docstrings(self):
        code = 'def f(x):\n    """doc"""\n    # comment\n    return x + 1\n'
        canonical = canonicalize_code(code)
        self.assertNotIn("doc", canonical)
        self.assertNotIn("comment", canonical)
        self.assertIn("return x + 1", canonical)

    def test_evaluate_code_diversity_reports_swdi_and_cdi(self):
        records = [
            {"function": "def f(x):\n    return x + 1\n"},
            {"function": "def g(x):\n    return x + 2\n"},
            {"function": "def h(x):\n    return x * 2\n"},
        ]
        report = evaluate_code_diversity(records, alpha=0.95, encoding="tfidf")
        self.assertEqual(report["num_encoded"], 3)
        self.assertGreaterEqual(report["swdi"], 0.0)
        self.assertGreaterEqual(report["cdi"], 0.0)


if __name__ == "__main__":
    unittest.main()
