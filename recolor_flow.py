"""ShootPX — Recolor: shared flow logic.

Fourth tool. flux-2/edit's `prompt` field is a single string with no separate system_prompt
input (unlike Seedream and the openrouter VLM route used elsewhere), so the fixed
preservation instructions are composed directly into one prompt string via
build_recolor_prompt(). There IS one narrow LLM step: if the user leaves "which part" blank,
suggest_recolor_target() asks Gemini to look at the photo and name the subject — that answer
still flows through the exact same build_recolor_prompt() template a manual description
would, so there's only one prompt-composition code path either way.

Reuses (not duplicates):
  - to_hosted_url, fetch_image_bytes, download_image: catalog_flow (model-agnostic)
  - nsfw/minors safety check: catalog_flow.run_safety_check (same calibration as every
    other tool — a recolor edit still touches a real input photo that could show a person)
  - the VLM call plumbing + prompt_writer model: catalog_flow._vlm_json_call /
    catalog_flow.MODEL_CONFIG['prompt_writer'] — same reasoning-model fix, same JSON
    extraction, not re-derived here.

Does NOT reuse catalog_flow's image assembly or resolution mapping — flux-2/edit has
genuinely different constraints, confirmed live from its fal.ai schema, 2026-08-29:
  - max 4 input images (every other tool here caps at 10)
  - image_size enum has no auto_2K/3K/4K options, and a custom size is a hard 512-2048px
    per-axis requirement (no auto-scaling like Catalog's v5-lite, no 1920-4096 range like
    On-Model Shots' v4.5) — reusing either sibling's mapping here would be wrong.
"""

import os

import fal_client
from dotenv import load_dotenv
from PIL import Image

import catalog_flow

load_dotenv()

MODEL_CONFIG = {
    "recolor_generation": os.environ["RECOLOR_GENERATION_MODEL"],   # fal-ai/flux-2/edit
    "prompt_writer": catalog_flow.MODEL_CONFIG["prompt_writer"],    # only used when description is blank
}

# Re-exported so streamlit only needs to import this one module.
fetch_image_bytes = catalog_flow.fetch_image_bytes
download_image = catalog_flow.download_image


# =============================================================================
# 1. Reference images — up to 4, first one is always "the photo being edited"
# =============================================================================

RECOLOR_MAX_IMAGES = 4  # confirmed: flux-2/edit silently uses only the first 4 if more are sent


def assemble_recolor_images(paths: list[str]) -> tuple[list[str], list[str]]:
    """Enforces the real 4-image limit ourselves rather than relying on fal silently
    dropping the rest — same principle as every other tool's 10-image reject, different
    number because this is a different model. Image 1 is always the photo being recolored;
    any additional images are optional references (e.g. a color-swatch image) per flux-2's
    own multi-reference support."""
    if len(paths) > RECOLOR_MAX_IMAGES:
        raise ValueError(
            f"{len(paths)} images exceeds flux-2/edit's {RECOLOR_MAX_IMAGES}-image limit — "
            f"trim {len(paths) - RECOLOR_MAX_IMAGES} image(s) and try again."
        )
    image_urls = [catalog_flow.to_hosted_url(p) for p in paths]
    labels = ["Image 1 = the photo being edited (preserve everything except the requested change)"]
    labels += [f"Image {i+2} = additional reference (e.g. color/style reference only)" for i in range(len(image_urls) - 1)]
    return image_urls, labels


# =============================================================================
# 2. Output settings — aspect ratio / resolution
# =============================================================================
# Verified against fal-ai/flux-2/edit's schema, 2026-08-29 — same preset NAMES as Seedream
# happen to appear, but do not assume the same behavior: this model's custom-size rule is a
# hard 512-2048px per axis (reject outside that range), not Seedream 4.5's 1920-4096 or
# v5-lite's total-pixel-with-auto-scale.

ASPECT_RATIO_MAP = {
    "1:1": "square_hd",
    "3:4": "portrait_4_3",
    "9:16": "portrait_16_9",
    "4:3": "landscape_4_3",
    "16:9": "landscape_16_9",
}
RECOLOR_MIN_DIM = 512
RECOLOR_MAX_DIM = 2048


def build_image_size(resolution_mode: str = "standard", aspect_ratio: str = "1:1", custom_width: int | None = None, custom_height: int | None = None):
    """resolution_mode: 'standard' (named preset) | 'custom' (custom_width/custom_height,
    hard-validated — this model rejects, not auto-scales, an out-of-range size). Does not
    handle 'match_input' — that's resolved to a concrete 'custom' size by
    compute_matching_size() before this is called; see run_recolor()."""
    if resolution_mode == "standard":
        if aspect_ratio not in ASPECT_RATIO_MAP:
            raise ValueError(f"Unknown aspect_ratio {aspect_ratio!r} — one of {list(ASPECT_RATIO_MAP)}")
        return ASPECT_RATIO_MAP[aspect_ratio]
    if resolution_mode == "custom":
        if custom_width is None or custom_height is None:
            raise ValueError("resolution_mode='custom' requires custom_width and custom_height")
        if not (RECOLOR_MIN_DIM <= custom_width <= RECOLOR_MAX_DIM and RECOLOR_MIN_DIM <= custom_height <= RECOLOR_MAX_DIM):
            raise ValueError(
                f"{custom_width}x{custom_height} invalid — each side must be "
                f"{RECOLOR_MIN_DIM}-{RECOLOR_MAX_DIM}px for this model."
            )
        return {"width": custom_width, "height": custom_height}
    raise ValueError(f"Unknown resolution_mode: {resolution_mode!r} — one of 'standard', 'custom'")


def compute_matching_size(image_path: str) -> dict:
    """Default behavior: no explicit aspect-ratio/quality choice means "don't change the
    photo's own proportions" — reads the uploaded image's real width/height and scales them
    (preserving aspect ratio) to fit this model's 512-2048 per-axis range, rather than
    silently forcing a fixed ratio like 1:1 onto every input."""
    with Image.open(image_path) as img:
        width, height = img.size

    scale = 1.0
    if max(width, height) > RECOLOR_MAX_DIM:
        scale = RECOLOR_MAX_DIM / max(width, height)
    elif min(width, height) < RECOLOR_MIN_DIM:
        scale = RECOLOR_MIN_DIM / min(width, height)

    new_width = max(RECOLOR_MIN_DIM, min(RECOLOR_MAX_DIM, round(width * scale)))
    new_height = max(RECOLOR_MIN_DIM, min(RECOLOR_MAX_DIM, round(height * scale)))
    return {"width": new_width, "height": new_height}


# =============================================================================
# 3. Prompt composition — no LLM call, just a fixed template
# =============================================================================
# From the product review, 2026-08-29: keep the USER's input dead simple (one part, one
# color); the app — not the user — carries the preservation/fidelity instructions, prepended
# onto the single prompt string flux-2/edit actually receives (it has no separate
# system_prompt field like the openrouter VLM route the other tools use).

RECOLOR_PRESERVATION_INSTRUCTIONS = """Edit only the elements requested by the user.

Preserve everything else in the input image exactly as much as possible, including the person, pose, body proportions, garment shape, garment texture, patterns, details, lighting, shadows, background, camera framing and composition.

For color changes, modify only the requested garment or object while preserving its original shape, material, texture, folds, stitching and construction.

Do not add, remove, replace or redesign other elements unless explicitly requested.

Follow the user's requested color precisely. When a HEX color is provided, match that HEX color as closely as possible.

If multiple objects are mentioned, apply each requested change only to the corresponding object.

Produce a clean, realistic commercial-quality image."""


RECOLOR_TARGET_WRITER_SYSTEM = """You look at a product/photo image and identify the single most prominent recolorable subject in it — e.g. the main garment, product, or object a customer would most likely want to change the color of.

Respond with a short noun phrase naming that subject only (e.g. "the shirt", "the sofa", "the jacket"). Do not add commentary or explanation.

Return only the requested JSON structure."""


def suggest_recolor_target(image_urls: list[str]) -> str:
    """Node: user gave no description -> one LLM+Vision call looks at Image 1 and picks a
    short target phrase. That phrase then goes through the exact same build_recolor_prompt()
    template a manually-typed description would — this function only ever decides WHAT the
    description is, never composes the final prompt itself."""
    result = catalog_flow._vlm_json_call(
        model=MODEL_CONFIG["prompt_writer"], system=RECOLOR_TARGET_WRITER_SYSTEM,
        prompt='Identify the recolorable subject in Image 1. Return ONLY a JSON object: {"target": "<short noun phrase>"}.',
        image_urls=image_urls[:1], max_tokens=50,
    )
    return result["target"]


def build_recolor_prompt(description: str, color: str) -> str:
    """description: one free-text field — "which part should be recolored", the user's whole
    request in their own words (e.g. "the shirt", or "just the sleeves of the jacket") — no
    separate target/extra-instruction split; that was the actual complication. Always
    non-empty by the time this is called: run_recolor() fills a blank one in via
    suggest_recolor_target() before this runs. UI never asks the user to type "#Image_1" —
    that's always resolved internally to the primary uploaded photo. color: a hex code (e.g.
    "#FF0000") or a plain color name — either works per fal's own docs."""
    return (
        f"{RECOLOR_PRESERVATION_INSTRUCTIONS}\n\nUser request:\n"
        f"Change {description} in #Image_1 to {color}. Keep everything else unchanged."
    )


# =============================================================================
# 4. Generation — fal-ai/flux-2/edit
# =============================================================================

def run_recolor_generation(
    prompt: str, image_urls: list[str],
    resolution_mode: str = "standard", aspect_ratio: str = "1:1",
    custom_width: int | None = None, custom_height: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Always num_images=1 — recolor is a single-image-in, single-image-out tool, no batch
    setting anywhere. Note: per fal's own docs, an unsafe result comes back as a black image
    rather than an error when enable_safety_checker can't be disabled — this is a second,
    independent backstop behind run_safety_check() below, not a replacement for it."""
    args = {
        "prompt": prompt,
        "image_urls": image_urls,
        "image_size": build_image_size(resolution_mode, aspect_ratio, custom_width, custom_height),
        "num_images": 1,
        "enable_safety_checker": True,  # do not disable
    }
    if seed is not None:
        args["seed"] = seed

    result = fal_client.subscribe(MODEL_CONFIG["recolor_generation"], arguments=args, with_logs=True)
    return [img["url"] for img in result["images"]]


# =============================================================================
# 5. Orchestrator
# =============================================================================

def run_recolor(
    image_paths: list[str],
    color: str,
    description: str = "",       # optional — blank means "let the LLM name the subject"
    resolution_mode: str = "match_input",   # default: keep the uploaded photo's own proportions
    aspect_ratio: str = "1:1",
    custom_width: int | None = None,
    custom_height: int | None = None,
) -> dict:
    """Upload -> safety check (images + description) -> fill in a blank description via one
    LLM call -> compose the templated prompt -> one generation call -> single output image.

    resolution_mode='match_input' (the default — no explicit choice needed) reads
    image_paths[0]'s real dimensions and preserves that aspect ratio; pass 'standard' with an
    aspect_ratio, or 'custom' with custom_width/custom_height, to override it explicitly —
    quality/size is always optional to change, never required."""
    image_urls, labels = assemble_recolor_images(image_paths)

    passed, reason = catalog_flow.run_safety_check(image_urls, description or None)
    if not passed:
        raise ValueError(f"Blocked at safety check: {reason}")

    if not description:
        description = suggest_recolor_target(image_urls)

    prompt = build_recolor_prompt(description, color)

    if resolution_mode == "match_input":
        size = compute_matching_size(image_paths[0])
        urls = run_recolor_generation(prompt, image_urls, resolution_mode="custom", custom_width=size["width"], custom_height=size["height"])
    else:
        urls = run_recolor_generation(
            prompt, image_urls,
            resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
            custom_width=custom_width, custom_height=custom_height,
        )
    return {"prompt": prompt, "description": description, "image_urls": image_urls, "labels": labels, "output": urls[0]}
