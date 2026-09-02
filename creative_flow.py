"""ShootPX — Creative Photoshoot: shared flow logic.

Third tool alongside flow.py (On-Model Shots) and catalog_flow.py (Catalog Photoshoot).
Always produces exactly ONE image per Generate click — no num_images/batch setting anywhere
in this tool, by design. A second variation is a second click of Generate (full flow re-run,
safety check included), not a batch fan-out.

Deliberately thin — reuses rather than duplicates:
  - image assembly + 10-image cap: catalog_flow.assemble_product_images
  - safety check (nsfw + minors, one VLM call): catalog_flow.run_safety_check
  - VLM call plumbing (reasoning=True fix, JSON-fence/reasoning-trace extraction):
    catalog_flow._vlm_json_call — re-deriving that here would reintroduce the exact bug
    classes already fixed once this session; not worth a third copy.
  - resolution mapping + the actual generation call: catalog_flow.build_image_size /
    catalog_flow.run_catalog_generation / catalog_flow.ASPECT_RATIO_MAP. Generation here
    targets bytedance/seedream/v5/lite/edit — the SAME model Catalog Photoshoot uses
    (confirmed, 2026-08-29) — so catalog_flow's implementation is directly correct here, not
    flow.py's (which is calibrated for the different Seedream 4.5 edit schema: per-axis size
    constraint + a seed param, neither of which v5-lite has). No dependency on flow.py at all
    now — this tool only ever imports catalog_flow.

No category classifier / intimate-apparel redirect gate here — catalog_flow.py doesn't have
one either (that machinery was explicitly dropped from an earlier draft, never built), so
there's nothing real to import. Only the shared nsfw+minors safety check applies.
"""

from dotenv import load_dotenv

import catalog_flow

load_dotenv()

# Only what THIS tool configures directly. Generation and safety_check are both delegated
# to catalog_flow (catalog_flow.MODEL_CONFIG['catalog_generation'] / ['safety_check']) —
# not duplicated here, so there's one place those stay correct for v5-lite's schema.
MODEL_CONFIG = {
    "prompt_writer": catalog_flow.MODEL_CONFIG["prompt_writer"],
}

# Re-exported so streamlit only needs to import this one module.
fetch_image_bytes = catalog_flow.fetch_image_bytes
download_image = catalog_flow.download_image
ASPECT_RATIO_MAP = catalog_flow.ASPECT_RATIO_MAP


IDEA_OPTIONS = [
    "Lifestyle / in-use scene",
    "Studio hero shot",
    "Outdoor / nature setting",
    "Minimalist flat lay",
    "Editorial / fashion",
    "Seasonal / festive",
    "Social media story (vertical)",
    "Moody / dramatic lighting",
    "Bright & airy",
    "Product-in-hand close-up",
]


# =============================================================================
# Adult-floor guardrail — applied to EVERY final prompt before generation, regardless of
# which case produced it (including Case 1's verbatim user prompt, which never touches an
# LLM). Cheap, unconditional, harmless if no person ends up in the scene.
# =============================================================================

ADULT_FLOOR_GUARDRAIL = (
    "If this scene includes any person, they must be a clearly adult, professional "
    "photography subject — mature adult facial structure and proportions, never a minor. "
)


def apply_adult_floor(prompt: str) -> str:
    return ADULT_FLOOR_GUARDRAIL + prompt


# =============================================================================
# Case branching — exactly one of three shapes, one final_prompt out
# =============================================================================

# Upgraded, 2026-08-29: the previous version told the VLM to treat an idea tag like
# "Outdoor / nature setting" as literal text to work from. This version instead tells it to
# interpret what the idea MEANS visually (worked examples for each IDEA_OPTIONS entry below)
# and translate that into a real photography concept, rather than echoing the tag.
CREATIVE_PROMPT_WRITER_SYSTEM = """You are the Creative Director for ShootPX Creative Photoshoot.

Your job is to create ONE production-ready image-generation prompt from:
1. the uploaded product reference images,
2. the selected creative idea(s), and/or
3. the user's additional prompt.

This tool creates exactly ONE image per Generate click.

PRODUCT IS THE SOURCE OF TRUTH

Analyze ALL supplied reference images together to understand the exact physical product.

Preserve:
- exact geometry
- proportions
- shape
- construction
- materials
- colors
- textures
- patterns
- graphics
- logos
- branding
- labels
- stitching
- hardware
- distinctive physical details

Never redesign, simplify, beautify, replace, or invent product features.

If multiple reference images show different views of the same product, combine them to understand the complete product.

REFERENCE IMAGE HANDLING

Reference images may contain:
- boxes
- packaging
- people
- furniture
- tables
- backgrounds
- screenshots
- marketplace UI
- watermarks
- unrelated objects

These are NOT automatically part of the product.

Do not reproduce them unless the user explicitly asks for them.

CREATIVE IDEA INTERPRETATION

Do not treat an idea tag as literal text that must simply be repeated.

Understand what the idea means visually and translate it into a professional commercial photography concept.

Examples:

"Sci-fi / futuristic":
Create a believable futuristic environment, cinematic composition, advanced architecture or technology-inspired surroundings, controlled dramatic lighting and atmospheric depth while keeping the exact product unchanged and clearly visible.

"Lifestyle / in-use":
Place the exact product naturally in a believable real-world situation where a customer would actually use it.

"Product-in-hand close-up":
Show the exact product naturally held or used by a clearly adult person, with the product remaining the visual focus.

"Editorial / fashion":
Create a sophisticated fashion/editorial campaign composition with professional adult styling, intentional composition and premium lighting.

"Outdoor / nature":
Place the exact product naturally in an appropriate outdoor environment with realistic environmental lighting and professional commercial photography.

"Studio hero":
Create a premium controlled studio product photograph with professional lighting, composition and product emphasis.

"Minimalist flat lay":
Arrange the exact product in a clean overhead composition with controlled spacing and minimal visual distractions.

"Moody / dramatic lighting":
Use cinematic contrast, controlled shadows and atmospheric lighting while maintaining accurate product appearance.

"Bright & airy":
Use bright natural-looking illumination, soft shadows, clean composition and an uplifting commercial aesthetic.

These are examples of interpretation, not mandatory templates. Select the visual treatment that best fits the actual product and user's intent.

USER PROMPT

If the user provides a free-text prompt:
- understand what they actually want
- preserve their important instructions
- improve clarity and specificity
- integrate it naturally with the selected idea
- do not contradict the user's explicit requirements

If the user provides both an idea and a prompt:
the idea establishes the overall creative direction and the user's prompt provides additional or more specific instructions.

Do NOT simply concatenate them.
Synthesize them into ONE coherent generation prompt.

If the user provides only an idea:
invent the missing visual details needed to make a strong professional image.

If the user provides only a prompt:
the prompt is the primary creative direction.

PEOPLE

If the concept requires a person, use a clearly adult professional subject.

Never depict minors or ambiguous-age people.

The person must support the product rather than obscure it.

PRODUCT PRIORITY

The creative environment may change dramatically.

The product may NOT.

The final scene must make the product recognizable and commercially useful.

Do not allow the environment, props, people, lighting or effects to obscure important product details.

PROMPT QUALITY

Write ONE concise but detailed production-ready prompt.

Include the appropriate:
- subject/product presentation
- environment
- composition
- camera perspective
- lighting
- atmosphere
- photography style
- product placement

Do not provide multiple concepts.
Do not provide alternatives.
Do not explain your reasoning.

Return only the requested JSON structure."""


def build_creative_prompt(
    image_urls: list[str], labels: list[str],
    idea_tags: list[str] | None = None, user_prompt: str | None = None,
) -> str:
    """Node: 'Which case?' -> Case 1/2/3 -> final_prompt. Exactly one LLM call for Case 2 or
    3, zero for Case 1 (prompt used verbatim — the whole point of that case is skipping the
    LLM entirely)."""
    idea_tags = idea_tags or []
    if not idea_tags and not user_prompt:
        raise ValueError("Provide an idea, a prompt, or both — Generate needs at least one.")

    # Case 1: prompt only, no idea.
    if user_prompt and not idea_tags:
        return user_prompt

    # Case 2 (idea only) / Case 3 (idea + prompt): one LLM+Vision call either way — the
    # system prompt above handles both shapes, so no separate code path is needed.
    prompt_text = (
        "Product reference images, in order:\n" + "\n".join(labels)
        + f"\n\nCreative idea tag(s): {', '.join(idea_tags)}"
    )
    if user_prompt:
        prompt_text += f"\nUser's additional instruction (merge with the idea into one refined prompt): {user_prompt}"
    prompt_text += '\n\nReturn ONLY a JSON object: {"prompt": "<the single final prompt>"}.'

    result = catalog_flow._vlm_json_call(
        model=MODEL_CONFIG["prompt_writer"], system=CREATIVE_PROMPT_WRITER_SYSTEM,
        prompt=prompt_text, image_urls=image_urls, max_tokens=500,
    )
    return result["prompt"]


# =============================================================================
# Orchestrator — upload -> safety check -> case branch -> adult floor -> single generation
# =============================================================================

def run_creative_photoshoot(
    product_paths: list[str],
    idea_tags: list[str] | None = None,
    user_prompt: str | None = None,
    resolution_mode: str = "standard",
    aspect_ratio: str = "3:4",
    custom_width: int | None = None,
    custom_height: int | None = None,
) -> dict:
    """Full flow, node for node. Always exactly one generation call — no batch path exists
    anywhere in this function; a second variation means calling this again."""
    image_urls, labels = catalog_flow.assemble_product_images(product_paths)

    # Safety check runs on the RAW user prompt + images, before any LLM refinement —
    # matches the diagram: safety gates the case branch, not the other way around.
    passed, reason = catalog_flow.run_safety_check(image_urls, user_prompt)
    if not passed:
        raise ValueError(f"Blocked at safety check: {reason}")

    final_prompt = build_creative_prompt(image_urls, labels, idea_tags, user_prompt)
    final_prompt = apply_adult_floor(final_prompt)

    urls = catalog_flow.run_catalog_generation(
        final_prompt, image_urls,
        resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
        custom_width=custom_width, custom_height=custom_height,
        num_images=1,
    )
    return {"final_prompt": final_prompt, "image_urls": image_urls, "output": urls[0]}
