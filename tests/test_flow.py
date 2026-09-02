"""Behavior tests for flow.py's VLM JSON-call boundary.

Reproduces the production crash (streamlit_app.py -> generate_model_candidate ->
generate_model_prompt_via_llm -> _vlm_json_call): json.decoder.JSONDecodeError
"Expecting value: line 2 column 12 (char 13)" with no indication of what the model
actually returned. fal_client is faked at the module boundary so no network/API key
is needed.
"""

import sys
import unittest
from unittest.mock import Mock

import flow


class VlmJsonCallTests(unittest.TestCase):
    def setUp(self):
        self._real_subscribe = flow.fal_client.subscribe
        self.addCleanup(setattr, flow.fal_client, "subscribe", self._real_subscribe)

    def test_non_json_output_raises_diagnosable_error(self):
        """Real failure mode: a reasoning-mandatory model (e.g. gemini-3.1-pro-preview)
        burns its token budget on hidden reasoning and returns near-empty/non-JSON
        content. This must not surface as a bare, contextless JSONDecodeError — the
        raw model output has to be visible in the raised error so it's diagnosable."""
        flow.fal_client.subscribe = Mock(
            return_value={"output": "\n           Sorry, I cannot"}
        )

        with self.assertRaises(Exception) as ctx:
            flow._vlm_json_call(model="m", system="s", prompt="p")

        message = str(ctx.exception)
        self.assertNotIsInstance(
            ctx.exception, __import__("json").decoder.JSONDecodeError,
            "must not leak a bare JSONDecodeError with no context about the raw output",
        )
        self.assertIn("Sorry, I cannot", message)

    def test_empty_output_raises_diagnosable_error(self):
        flow.fal_client.subscribe = Mock(return_value={"output": ""})

        with self.assertRaises(Exception) as ctx:
            flow._vlm_json_call(model="m", system="s", prompt="p")

        self.assertNotIsInstance(ctx.exception, __import__("json").decoder.JSONDecodeError)


class GenerateModelPromptTokenBudgetTests(unittest.TestCase):
    """Guards against regressing to the under-provisioned token budget that caused
    this exact crash: PROMPT_WRITER_MODEL is gemini-3.1-pro-preview, a reasoning-
    mandatory model (flow.py sends reasoning=True unconditionally); with max_tokens=300
    its hidden reasoning trace can exhaust the budget before any JSON answer is
    emitted."""

    def test_prompt_writer_call_has_headroom_for_reasoning(self):
        captured = {}

        def fake_vlm_json_call(model, system, prompt, image_urls=None, max_tokens=1000):
            captured["max_tokens"] = max_tokens
            return {"prompt": "a description"}

        original = flow._vlm_json_call
        flow._vlm_json_call = fake_vlm_json_call
        try:
            flow.generate_model_prompt_via_llm("Female", "Adult (30s-40s)", "Fair", "Slim")
        finally:
            flow._vlm_json_call = original

        self.assertGreaterEqual(
            captured["max_tokens"], 600,
            "max_tokens too low for a reasoning-mandatory model to both reason and answer",
        )


if __name__ == "__main__":
    unittest.main()
