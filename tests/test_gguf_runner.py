from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_gguf_fixture.py"
    spec = importlib.util.spec_from_file_location("run_gguf_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_gguf_fixture.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GgufRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner_module()

    def test_extract_assistant_text_from_chat_completion(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Contact <REDACTED_EMAIL_A>.",
                    }
                }
            ]
        }
        self.assertEqual(self.runner.extract_assistant_text(response), "Contact <REDACTED_EMAIL_A>.")

    def test_extract_assistant_text_from_text_completion(self) -> None:
        response = {"choices": [{"text": "ID <REDACTED_ACCOUNT_A>"}]}
        self.assertEqual(self.runner.extract_assistant_text(response), "ID <REDACTED_ACCOUNT_A>")

    def test_build_messages_keeps_prompt_and_source_separate(self) -> None:
        messages = self.runner.build_messages("Return only redacted text.", "email alice@example.test")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Return only redacted text.", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "email alice@example.test")


if __name__ == "__main__":
    unittest.main()
