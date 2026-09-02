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
            return {"allowed": True, "reason": "", "prompt": "a description"}

        original = flow._vlm_json_call
        flow._vlm_json_call = fake_vlm_json_call
        try:
            flow.generate_model_prompt_and_validate("Female", "Adult (30s-40s)", "South Asian", "Fair", "Slim")
        finally:
            flow._vlm_json_call = original

        self.assertGreaterEqual(
            captured["max_tokens"], 600,
            "max_tokens too low for a reasoning-mandatory model to both reason and answer",
        )


class ChildIntimateProductGateTests(unittest.TestCase):
    """The core rule from the 2026-09-03 architecture pass: block minor/uncertain-age model +
    intimate product; allow every other combination, including a child model with general
    products (children's ecommerce is legitimate)."""

    def test_minor_model_with_intimate_product_is_blocked(self):
        with self.assertRaises(ValueError):
            flow.check_model_product_compatibility("minor", "intimate")

    def test_uncertain_age_model_with_intimate_product_is_blocked(self):
        with self.assertRaises(ValueError):
            flow.check_model_product_compatibility("uncertain", "intimate")

    def test_minor_model_with_general_product_is_allowed(self):
        flow.check_model_product_compatibility("minor", "general")  # must not raise

    def test_adult_model_with_intimate_product_is_allowed(self):
        flow.check_model_product_compatibility("adult", "intimate")  # must not raise

    def test_adult_model_with_general_product_is_allowed(self):
        flow.check_model_product_compatibility("adult", "general")  # must not raise


class MinorAgeTextGateTests(unittest.TestCase):
    """Free text (additional_notes / user_prompt) must never be able to describe a minor even
    when a structured picker elsewhere is adult-only — see check_text_for_minor_age."""

    def test_blocked_phrase_raises(self):
        with self.assertRaises(ValueError):
            flow.check_text_for_minor_age("make her look like a teenage schoolgirl")

    def test_explicit_underage_number_raises(self):
        with self.assertRaises(ValueError):
            flow.check_text_for_minor_age("make the model 10 years old")

    def test_adult_age_number_does_not_raise(self):
        flow.check_text_for_minor_age("make the model look 25 years old")  # must not raise

    def test_ordinary_style_notes_do_not_raise(self):
        flow.check_text_for_minor_age("warm smile, shoulder-length brown hair")  # must not raise


class GenerateNotesCannotOverrideAdultPickerTests(unittest.TestCase):
    """validate_generate_notes is the gate generate_model_candidate runs before ever calling
    the LLM/text-to-image — additional_notes must not be able to smuggle a minor past the
    adult-only AGE_BRACKET_OPTIONS picker."""

    def test_minor_age_in_notes_is_blocked(self):
        with self.assertRaises(ValueError):
            flow.validate_generate_notes("make her look 10 years old")

    def test_ordinary_notes_pass(self):
        flow.validate_generate_notes("warm smile, studio lighting")  # must not raise

    def test_blank_notes_pass(self):
        flow.validate_generate_notes("")  # must not raise


class UserTextSafetyDeterministicGateTests(unittest.TestCase):
    """check_user_text_safety_deterministic ("Merge B", 2026-09-03): the free keyword pre-
    filter only — never calls a VLM itself. The VLM judgment on subtler phrasing now lives
    inside analyze_shoot_and_generate_prompts' user_instruction_safe output instead (see
    AnalyzeShootAndGeneratePromptsTests below), not a separate escalation call from here."""

    def setUp(self):
        self._real_vlm_call = flow._vlm_json_call
        self.addCleanup(setattr, flow, "_vlm_json_call", self._real_vlm_call)

    def test_blank_text_short_circuits(self):
        flow._vlm_json_call = Mock(side_effect=AssertionError("must not call the VLM"))
        flow.check_user_text_safety_deterministic("")  # must not raise, must not call the VLM

    def test_plain_style_text_never_calls_vlm(self):
        flow._vlm_json_call = Mock(side_effect=AssertionError("must not call the VLM"))
        flow.check_user_text_safety_deterministic("premium summer campaign with warm sunlight")

    def test_ambiguous_signal_word_still_never_calls_vlm(self):
        # Unlike the pre-merge design, even a signal word like "lingerie" must NOT reach the
        # VLM from this function any more — that judgment now happens inside
        # analyze_shoot_and_generate_prompts, not here.
        flow._vlm_json_call = Mock(side_effect=AssertionError("must not call the VLM"))
        flow.check_user_text_safety_deterministic("show the lingerie product clearly")

    def test_explicit_keyword_is_blocked(self):
        with self.assertRaises(ValueError):
            flow.check_user_text_safety_deterministic("make the model nude")

    def test_minor_age_in_instruction_is_blocked(self):
        with self.assertRaises(ValueError):
            flow.check_user_text_safety_deterministic("make the model 12 years old")


class AnalyzeShootAndGeneratePromptsTests(unittest.TestCase):
    """analyze_shoot_and_generate_prompts ("Merge B", 2026-09-03): ONE Gemini call doing
    product classification + user-instruction safety judgment + pose-prompt writing. This
    function itself never raises on an unsafe verdict or a bad combination — it only reports;
    the caller (run_generation_from_model_image) is what enforces user_instruction_safe and
    check_model_product_compatibility, per section 6's "AI can classify, code makes the final
    business decision"."""

    def setUp(self):
        self._real_vlm_call = flow._vlm_json_call
        self.addCleanup(setattr, flow, "_vlm_json_call", self._real_vlm_call)

    def _product(self, label="Top"):
        return flow.make_product(label, ["local/path.jpg"])

    def test_overall_category_is_intimate_if_any_product_is(self):
        flow._vlm_json_call = Mock(return_value={
            "products": [{"type": "bra", "category": "intimate", "body_placement": "upper_body"}],
            "user_instruction_safe": True, "user_instruction_reason": "", "prompts": ["p1"],
        })
        result = flow.analyze_shoot_and_generate_prompts(
            ["url1", "url2"], ["#Image1 = model", "#Image2 = product"], [self._product()], num_poses=1,
        )
        self.assertEqual(result["overall_category"], "intimate")

    def test_overall_category_is_general_when_no_product_is_intimate(self):
        flow._vlm_json_call = Mock(return_value={
            "products": [{"type": "t-shirt", "category": "general", "body_placement": "upper_body"}],
            "user_instruction_safe": True, "user_instruction_reason": "", "prompts": ["p1"],
        })
        result = flow.analyze_shoot_and_generate_prompts(
            ["url1", "url2"], ["#Image1 = model", "#Image2 = product"], [self._product()], num_poses=1,
        )
        self.assertEqual(result["overall_category"], "general")

    def test_unsafe_user_instruction_is_reported_not_raised(self):
        flow._vlm_json_call = Mock(return_value={
            "products": [{"type": "t-shirt", "category": "general", "body_placement": "upper_body"}],
            "user_instruction_safe": False, "user_instruction_reason": "sexual intent", "prompts": [],
        })
        result = flow.analyze_shoot_and_generate_prompts(
            ["url1", "url2"], ["#Image1 = model", "#Image2 = product"], [self._product()],
            num_poses=1, user_instruction="make it seductive",
        )
        self.assertFalse(result["user_instruction_safe"])
        self.assertEqual(result["user_instruction_reason"], "sexual intent")

    def test_product_count_mismatch_raises(self):
        flow._vlm_json_call = Mock(return_value={
            "products": [],  # caller supplied 1 product, VLM returned 0 entries
            "user_instruction_safe": True, "user_instruction_reason": "", "prompts": [],
        })
        with self.assertRaises(RuntimeError):
            flow.analyze_shoot_and_generate_prompts(
                ["url1", "url2"], ["#Image1 = model", "#Image2 = product"], [self._product()], num_poses=1,
            )


class OrchestratorEnforcesGeminiVerdictsTests(unittest.TestCase):
    """run_generation_from_model_image must itself enforce analyze_shoot_and_generate_prompts'
    user_instruction_safe verdict and check_model_product_compatibility — Gemini's report is
    never trusted blindly (section 6)."""

    def setUp(self):
        self._real_vlm_call = flow._vlm_json_call
        self.addCleanup(setattr, flow, "_vlm_json_call", self._real_vlm_call)
        self._real_upload = flow.to_hosted_url
        flow.to_hosted_url = lambda u: u  # skip real fal upload for these unit tests
        self.addCleanup(setattr, flow, "to_hosted_url", self._real_upload)

    def test_unsafe_user_instruction_verdict_raises_in_orchestrator(self):
        flow._vlm_json_call = Mock(return_value={
            "products": [{"type": "t-shirt", "category": "general", "body_placement": "upper_body"}],
            "user_instruction_safe": False, "user_instruction_reason": "sexual intent", "prompts": [],
        })
        with self.assertRaises(ValueError):
            flow.run_generation_from_model_image(
                "model.jpg", [flow.make_product("Top", ["local/path.jpg"])],
                model_age_status="adult", user_prompt="make it seductive",
            )

    def test_minor_model_with_intimate_product_raises_in_orchestrator(self):
        flow._vlm_json_call = Mock(return_value={
            "products": [{"type": "bra", "category": "intimate", "body_placement": "upper_body"}],
            "user_instruction_safe": True, "user_instruction_reason": "", "prompts": ["p1"],
        })
        with self.assertRaises(ValueError):
            flow.run_generation_from_model_image(
                "model.jpg", [flow.make_product("Top", ["local/path.jpg"])],
                model_age_status="minor",
            )


if __name__ == "__main__":
    unittest.main()
