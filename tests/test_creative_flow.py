"""Behavior tests for the Creative Photoshoot orchestration contract."""

import importlib
import sys
import types
import unittest
from unittest.mock import Mock


class CreativePhotoshootFlowTests(unittest.TestCase):
    """Network boundaries are faked; the three-case routing stays real."""

    def setUp(self):
        self.catalog = types.ModuleType("catalog_flow")
        self.catalog.assemble_product_images = Mock(
            return_value=(
                ["https://example.test/product-front.png"],
                ["Image 1 = product reference (exact product, preserve fidelity)"],
            )
        )
        self.catalog.run_safety_check = Mock(return_value=(True, ""))

        self.on_model = types.ModuleType("flow")
        self.on_model.MODEL_CONFIG = {"prompt_writer": "vision-model"}
        self.on_model._vlm_json_call = Mock(return_value={"prompt": "vision-written scene"})
        self.on_model.run_final_generation = Mock(return_value=["https://example.test/output.png"])

        sys.modules["catalog_flow"] = self.catalog
        sys.modules["flow"] = self.on_model
        sys.modules.pop("creative_flow", None)
        self.creative = importlib.import_module("creative_flow")

    def tearDown(self):
        sys.modules.pop("creative_flow", None)
        sys.modules.pop("catalog_flow", None)
        sys.modules.pop("flow", None)

    def test_prompt_only_bypasses_vision_and_preserves_prompt_verbatim(self):
        """A regression in the prompt-only branch must not spend an LLM call or rewrite text."""
        prompt = "  crisp studio shot with warm shadows  "

        result = self.creative.run_creative_photoshoot(["product.jpg"], user_prompt=prompt)

        self.assertEqual(prompt, result["final_prompt"])
        self.on_model._vlm_json_call.assert_not_called()
        self.assertEqual("https://example.test/output.png", result["output"])

    def test_idea_only_writes_one_scene_prompt(self):
        """A regression that skips the idea-to-scene writer must fail this case."""
        result = self.creative.run_creative_photoshoot(["product.jpg"], idea_tags=["Editorial"])

        self.assertEqual("vision-written scene", result["final_prompt"])
        self.on_model._vlm_json_call.assert_called_once()
        self.assertEqual("https://example.test/output.png", result["output"])

    def test_idea_and_prompt_send_one_combined_vision_request(self):
        """A regression that splits creative inputs into two prompts must fail this case."""
        self.creative.run_creative_photoshoot(
            ["product.jpg"], idea_tags=["Beach", "Editorial"], user_prompt="at sunrise"
        )

        _, kwargs = self.on_model._vlm_json_call.call_args
        self.assertIn("Beach, Editorial", kwargs["prompt"])
        self.assertIn("at sunrise", kwargs["prompt"])
        self.on_model._vlm_json_call.assert_called_once()

    def test_failed_safety_stops_before_prompt_or_generation(self):
        """A regression that calls paid services after a safety rejection must fail."""
        self.catalog.run_safety_check.return_value = (False, "unsafe content")

        with self.assertRaisesRegex(ValueError, "Blocked at safety check: unsafe content"):
            self.creative.run_creative_photoshoot(["product.jpg"], idea_tags=["Editorial"])

        self.on_model._vlm_json_call.assert_not_called()
        self.on_model.run_final_generation.assert_not_called()

    def test_generation_is_fixed_to_exactly_one_image(self):
        """A regression that restores batch generation must fail this single-output contract."""
        self.creative.run_creative_photoshoot(["product.jpg"], user_prompt="simple scene")

        _, kwargs = self.on_model.run_final_generation.call_args
        self.assertEqual(1, kwargs["num_images"])

    def test_missing_idea_and_prompt_is_rejected_before_safety(self):
        """A regression that allows a directionless request must fail this validation."""
        with self.assertRaisesRegex(ValueError, "Select at least one idea or enter a prompt"):
            self.creative.run_creative_photoshoot(["product.jpg"], user_prompt="   ")

        self.catalog.run_safety_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
