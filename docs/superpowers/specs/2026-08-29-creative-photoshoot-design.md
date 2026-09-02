# Creative Photoshoot Tester Design

## Goal

Provide a standalone Streamlit tester for generating one creative product image per click from one to ten product references and either selected idea tags, a user prompt, or both.

## Scope

The tester consists of `creative_flow.py` for business logic and `creative_streamlit_app.py` for the user interface. It does not create database records or modify a production application because this workspace contains only standalone test modules and no `GenerationJob` implementation or production template files.

## Shared dependencies

`creative_flow.py` imports `catalog_flow` to reuse `assemble_product_images`, `run_safety_check`, and the Seedream image-size mapping. It imports `flow` to reuse the existing Seedream 4.5 generation endpoint and download helper. The existing workspace has no reusable category-classification, intimate-apparel redirect, or adult-floor helpers; those production-only requirements cannot be exercised by this tester without inventing a second implementation.

## Data flow

1. The UI writes uploaded product files to temporary files and sends their paths to `run_creative_photoshoot`.
2. The flow calls `catalog_flow.assemble_product_images`, retaining its explicit one-to-ten image cap and ordered labels.
3. The flow calls `catalog_flow.run_safety_check` once, with uploaded image URLs and the available prompt text. A rejected request stops before prompt generation or image generation.
4. Prompt resolution has three mutually exclusive paths: prompt-only returns the supplied user prompt verbatim (whitespace is used only to decide whether it is empty); idea-only calls the vision model once to create one product-faithful scene prompt; idea-plus-prompt calls it once to create one refined scene prompt.
5. The flow performs one call to `flow.run_final_generation` with `num_images=1` and returns only the first output URL.

## Public interface

```python
def run_creative_photoshoot(
    product_paths: list[str],
    idea_tags: list[str] | None = None,
    user_prompt: str | None = None,
    resolution_mode: str = "standard",
    aspect_ratio: str = "1:1",
    custom_width: int | None = None,
    custom_height: int | None = None,
) -> dict:
```

The return value contains `image_urls`, `labels`, `final_prompt`, and `output`. It never contains a batch of generated outputs.

## UI

The Streamlit page has product upload, resolution/aspect ratio controls, multi-select idea tags, an optional free-text prompt, one Generate button, and a single saved result. Generate is enabled only when at least one product image and at least one non-empty creative input are present.

## Verification

Standard-library `unittest` mocks the shared network boundaries and verifies the prompt-only bypass, both vision-prompt paths, safety blocking, image input ordering, one-image generation argument, and missing-input validation. A Streamlit syntax check verifies both new files compile.
