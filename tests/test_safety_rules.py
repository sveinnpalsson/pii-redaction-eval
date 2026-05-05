from __future__ import annotations

import unittest

from redaction_research.metrics import RedactionSpan
from redaction_research.safety_rules import add_safety_rule_spans, detect_safety_rule_matches


class SafetyRulesTests(unittest.TestCase):
    def test_detects_high_risk_tokens_and_contact_values(self) -> None:
        text = (
            "Contact amy@example.test with token sk-abc1234567890abcdef and "
            "path /home/amy/vault/private.md via @amy_ops."
        )
        matches = detect_safety_rule_matches(text)
        seen = {(match.name, match.span.label, match.text) for match in matches}

        self.assertIn(("openai-api-key", "CUSTOM", "sk-abc1234567890abcdef"), seen)
        self.assertIn(("home-path", "CUSTOM", "/home/amy/vault/private.md"), seen)
        self.assertIn(("email", "EMAIL", "amy@example.test"), seen)
        self.assertIn(("handle", "CUSTOM", "@amy_ops"), seen)

    def test_safety_pass_skips_existing_opf_span_overlap(self) -> None:
        text = "Contact amy@example.test and ACCT-1234-5678."
        existing = [
            RedactionSpan(
                start=text.index("amy@example.test"),
                end=text.index("amy@example.test") + len("amy@example.test"),
                label="EMAIL",
                placeholder="<REDACTED_EMAIL_A>",
            )
        ]

        combined, matches = add_safety_rule_spans(text, existing)

        self.assertEqual([match.name for match in matches], ["acct-token"])
        self.assertEqual([span.label for span in combined], ["EMAIL", "ACCOUNT"])
        self.assertEqual(combined[1].placeholder, "<REDACTED_ACCOUNT_A>")


if __name__ == "__main__":
    unittest.main()
