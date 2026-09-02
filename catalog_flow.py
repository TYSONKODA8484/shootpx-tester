"""ShootPX — Catalog Photoshoot: shared flow logic.

Different tool from On-Model Shots (flow.py): pure product photography, N shots per product,
no model-selection UI. Any person that appears in an output is incidental to a shot the VLM
chose to write (e.g. a worn/lifestyle angle) — not a chosen/reused identity — so this module
has none of flow.py's generate/upload/default model-source machinery.

Kept deliberately simple per spec: no category classification, no per-category shot
templates, no intimate-apparel redirect policy. The VLM decides the N shots for whatever
product it's looking at, in one call. The one thing NOT simplified away: "no minors" is
folded into the same single safety check as nsfw — a worn/lifestyle shot showing a person is
still something the VLM can choose to write here, so that check doesn't get to disappear
just because there's no model-picker UI in front of it.

Standalone on purpose (does not import flow.py) — same .env-driven MODEL_CONFIG convention,
same safety-check calibration, duplicated rather than shared so this tool works even if
flow.py isn't present.
"""

import io
import json
import os

import fal_client
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

assert os.environ.get("FAL_KEY"), "Set FAL_KEY in .env before running."

MODEL_CONFIG = {
    "catalog_generation": os.environ["CATALOG_GENERATION_MODEL"],   # bytedance/seedream/v5/lite/edit
    "vlm_endpoint": "openrouter/router/vision",                     # fal slug every VLM call goes through
    "prompt_writer": os.environ["PROMPT_WRITER_MODEL"],
    "safety_check": os.environ["SAFETY_CHECK_MODEL"],
}


# =============================================================================
# Shared helpers
# =============================================================================

def to_hosted_url(path_or_url: str) -> str:
    """Return a fal-hosted URL for a local file, or pass a URL through unchanged."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return fal_client.upload_file(path_or_url)


def show(url_or_path: str, caption: str = ""):
    """Default `on_image` renderer — notebook/IPython display. The Streamlit app passes a
    no-op instead and renders results itself from the returned dict."""
    from IPython.display import display

    if url_or_path.startswith("http"):
        img = Image.open(io.BytesIO(requests.get(url_or_path, timeout=30).content))
    else:
        img = Image.open(url_or_path)
    print(caption)
    display(img)


def fetch_image_bytes(url: str) -> bytes:
    return requests.get(url, timeout=30).content


def download_image(url: str, out_path: str) -> str:
    """fal output URLs are not permanent (subject to retention/expiry controls) — download
    what you want to keep before the session ends, don't just store the URL."""
    with open(out_path, "wb") as f:
        f.write(fetch_image_bytes(url))
    return out_path


def _strip_json_fence(text: str) -> str:
    """Gemini (and most chat models) often wrap JSON in ```json ... ``` even when told not
    to. Strip that before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _extract_json_text(text: str) -> str:
    """Fence-strip, then (real failure, 2026-08-29, reasoning-mandatory models like
    gemini-3.1-pro-preview) fall back to the outermost {...}/[...] block if the text still
    isn't valid JSON on its own — a reasoning model's thinking trace can land in the same
    `output` string ahead of its actual JSON answer."""
    text = _strip_json_fence(text)
    try:
        json.loads(text, strict=False)  # strict=False: tolerate literal control chars (raw
        return text                     # newlines etc.) inside string values — real failure,
    except json.JSONDecodeError:        # 2026-08-29: "Unterminated string" from a multi-line
        pass                            # prompt Gemini didn't escape as \n
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate, strict=False)
                return candidate
            except json.JSONDecodeError:
                continue
    return text  # give up — let json.loads raise with the real parse error


def _vlm_json_call(model: str, system: str, prompt: str, image_urls: list[str] | None = None, max_tokens: int = 1000) -> dict:
    """One call through fal's `openrouter/router/vision` (MODEL_CONFIG['vlm_endpoint']) —
    `model` picks the underlying model. `reasoning: True` is mandatory for some models on
    this endpoint (confirmed, 2026-08-29: gemini-3.1-pro-preview 400s without it —
    "Reasoning is mandatory for this endpoint and cannot be disabled") — sent unconditionally,
    non-reasoning models just ignore it. Must return a bare JSON object/array as text."""
    args = {
        "model": model,
        "system_prompt": system,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning": True,
    }
    if image_urls:
        args["image_urls"] = image_urls
    result = fal_client.subscribe(MODEL_CONFIG["vlm_endpoint"], arguments=args, with_logs=True)
    return json.loads(_extract_json_text(result["output"]), strict=False)


# =============================================================================
# 1. Product images
# =============================================================================

def assemble_product_images(product_paths: list[str]) -> tuple[list[str], list[str]]:
    """Same 10-image fal input ceiling and explicit-reject (not silent-trim) behavior as
    On-Model Shots."""
    if len(product_paths) > 10:
        raise ValueError(
            f"{len(product_paths)} product images exceeds the 10-image input limit — "
            f"trim {len(product_paths) - 10} image(s) and try again."
        )
    image_urls = [to_hosted_url(p) for p in product_paths]
    labels = [f"Image {i+1} = product reference (exact product, preserve fidelity)" for i in range(len(image_urls))]
    return image_urls, labels


# =============================================================================
# 2. Output settings — aspect ratio / resolution
# =============================================================================
# Verified against bytedance/seedream/v5/lite/edit's schema, 2026-08-29 — do NOT assume this
# matches Seedream 4.5 edit's constraints (flow.py):
#   - adds "auto_3K" to the enum (4.5 doesn't have it)
#   - custom size constraint is total-pixels-only (2560x1440-4096x4096), no per-axis
#     1920-4096 alternative like 4.5 has
#   - a custom size outside that range is auto-scaled to fit, not rejected — so validation
#     here is informational, not a hard gate
#   - no `seed` input field on this endpoint (4.5 edit has one) — omitted in
#     run_catalog_generation below rather than assumed

ASPECT_RATIO_MAP = {
    "1:1": "square_hd",
    "3:4": "portrait_4_3",
    "9:16": "portrait_16_9",
    "4:3": "landscape_4_3",
    "16:9": "landscape_16_9",
}
RESOLUTION_MODES = ["standard", "auto_2K", "auto_3K", "auto_4K", "custom"]
CATALOG_MIN_TOTAL_PX = 2560 * 1440   # 3,686,400
CATALOG_MAX_TOTAL_PX = 4096 * 4096   # 16,777,216


def build_image_size(resolution_mode: str = "standard", aspect_ratio: str = "1:1", custom_width: int | None = None, custom_height: int | None = None):
    """resolution_mode: 'standard' (named preset from aspect_ratio) | 'auto_2K'/'auto_3K'/
    'auto_4K' (model decides aspect ratio) | 'custom' (custom_width/custom_height — fal
    auto-scales this to fit its supported range, so out-of-range values aren't rejected)."""
    if resolution_mode == "standard":
        if aspect_ratio not in ASPECT_RATIO_MAP:
            raise ValueError(f"Unknown aspect_ratio {aspect_ratio!r} — one of {list(ASPECT_RATIO_MAP)}")
        return ASPECT_RATIO_MAP[aspect_ratio]
    if resolution_mode in ("auto_2K", "auto_3K", "auto_4K"):
        return resolution_mode
    if resolution_mode == "custom":
        if custom_width is None or custom_height is None:
            raise ValueError("resolution_mode='custom' requires custom_width and custom_height")
        total_px = custom_width * custom_height
        if not (CATALOG_MIN_TOTAL_PX <= total_px <= CATALOG_MAX_TOTAL_PX):
            print(
                f"NOTE: {custom_width}x{custom_height} ({total_px:,}px) is outside "
                f"{CATALOG_MIN_TOTAL_PX:,}-{CATALOG_MAX_TOTAL_PX:,} — fal will auto-scale it, "
                f"the exact size you set won't be honored."
            )
        return {"width": custom_width, "height": custom_height}
    raise ValueError(f"Unknown resolution_mode: {resolution_mode!r} — one of {RESOLUTION_MODES}")


# =============================================================================
# 3. Safety check — one simple check before Generate proceeds: nsfw/explicit content, with
# "no minors" folded into the same single call rather than a separate step.
# =============================================================================

BLOCKED_TERMS = ["child", "minor", "teen", "kid", "underage"]


def run_safety_check(image_urls: list[str], user_prompt: str | None = None) -> tuple[bool, str]:
    """Node: 'content and the if user prompt give check for any nsfw content asked to
    create' -> 'Unsafe?'. Returns (passed, reason). Cheap keyword check on the prompt first,
    then one VLM call on the images — same calibration as On-Model Shots (flag only on a
    real, specific reason to suspect a minor; ordinary product photography, including worn
    apparel, is not explicit on its own)."""
    if user_prompt:
        text = user_prompt.lower()
        for term in BLOCKED_TERMS:
            if term in text:
                return False, f"blocked term '{term}' in prompt"

    result = _vlm_json_call(
        model=MODEL_CONFIG["safety_check"],
        system=(
            "You are a content-safety classifier for an ecommerce product photo pipeline. "
            "Check two things only:\n"
            "1. Age: if any person appears in these images, could they plausibly be under "
            "18 — a genuine, specific reason to suspect a minor, not just youthful-looking? "
            "Do not flag on youthfulness alone.\n"
            "2. Explicit content: sexually explicit (visible nudity, exposed genitalia/"
            "nipples, sexual poses)? Ordinary product photography — including apparel or "
            "underwear being worn, which is the product itself — is NOT explicit on its own.\n"
            'Respond with ONLY a JSON object: {"pass": true|false, "reason": "short reason"}.'
        ),
        prompt="Classify these images per the rules in the system prompt.",
        image_urls=image_urls,
        max_tokens=200,
    )
    return bool(result["pass"]), result.get("reason", "")


# =============================================================================
# 4. LLM+Vision — build N shot prompts from the product images (+ optional user prompt)
# =============================================================================

# Real-world failure, 2026-08-29: reference sets from actual product listings include boxes,
# marketplace backgrounds, watermarks — the previous prompt didn't tell the VLM to separate
# the product from its surroundings, so those leaked into generated shots. Also dropped
# "state fidelity once, not per shot" — wrong, since each generation call is independent
# (prompt 1 + images -> Seedream, prompt 2 + images -> Seedream, ...), so every prompt needs
# its own fidelity instruction, not a shared one that only the first call would see.
SHOT_PROMPT_WRITER_SYSTEM = """You are the AI catalog photography director for an e-commerce product photography system.

You receive multiple reference images of the SAME physical product. The reference images may show different angles, backgrounds, packaging, boxes, people, screenshots, watermarks, or other surrounding objects.

Your job is to understand the EXACT PRODUCT and create exactly N distinct, professional e-commerce catalog photography prompts.

PRODUCT IS THE SOURCE OF TRUTH:
- Preserve the exact product identity.
- Preserve geometry, proportions, shape, construction, materials, textures, colors, patterns, logos, branding, labels and visible product details.
- Use all reference images together to understand the same physical product.
- Different reference images are different views of the same product, not different products.

PER-IMAGE ANALYSIS:
Analyze every reference image individually and then together.
For each image, determine: which parts of the product are visible, which product angle/view it shows, whether it contains packaging or unrelated objects, and which product details are reliable in that image.
Then combine all references to understand ONE physical product.
Never treat packaging, boxes, backgrounds, furniture, watermarks, marketplace UI or unrelated objects as part of the product.

REFERENCE CLEANLINESS:
- Separate the product from everything surrounding it.
- Do NOT reproduce boxes, packaging, stands, tables, backgrounds, watermarks, screenshots, marketplace UI, prices, text overlays, or unrelated objects from the reference images unless the user explicitly asks for them.
- Do not treat the background of a reference image as part of the product.

CATALOG PHOTOGRAPHY:
Create commercially useful e-commerce product images.
Choose different camera angles and compositions that help a customer understand and evaluate the product.
Shots must be meaningfully different and should not unnecessarily repeat the same view.

Before writing the prompts, mentally create a complete shot plan. Each shot must reveal different useful information about the product. Avoid duplicate shots that differ only slightly in camera position, lighting or crop. The complete set should maximize product coverage rather than maximize visual variation — e.g. for a shoe, prefer hero/side/front/heel/top/sole over six near-identical 3/4 shots.

For products such as shoes, useful shots may include:
hero 3/4 view, side profile, front, rear/heel, top, sole, and detail shots.

For apparel, useful shots may include:
front, back, side, detail, fabric/texture and appropriate product presentation.

For other products, choose the most useful views based on the actual product.

Do not force a specific shot type if it does not make sense for the product.

PEOPLE:
Only include a person when a person genuinely improves the catalog presentation of the product, such as footwear or apparel being worn.
Any person must clearly be an adult.
Do not create minors.

PROMPT REQUIREMENTS:
- Write exactly N prompts.
- Each prompt must be standalone because it will be sent to the image-generation model independently.
- Every prompt must contain the essential product-fidelity instruction.
- Clearly describe the desired camera angle, composition, environment, lighting and photography style.
- Keep prompts concise and directly actionable.
- Never ask the image model to redesign or reinterpret the product.

Return ONLY a JSON array containing exactly N strings."""


def _validate_shot_prompts(prompts, num_outputs: int) -> tuple[bool, str]:
    """Production-code check on the VLM's response, not just a JSON-parses check: exactly N
    entries, every entry a non-empty string, no exact-duplicate prompts (a sign the VLM
    degenerated into repeating one shot instead of planning N distinct ones)."""
    if not isinstance(prompts, list):
        return False, f"expected a JSON array, got {type(prompts).__name__}"
    if len(prompts) != num_outputs:
        return False, f"expected exactly {num_outputs} prompts, got {len(prompts)}"
    if not all(isinstance(p, str) and p.strip() for p in prompts):
        return False, "one or more prompts is empty or not a string"
    if len(set(p.strip() for p in prompts)) != len(prompts):
        return False, "duplicate prompts — the VLM repeated one shot instead of planning N distinct ones"
    return True, ""


def build_shot_prompts(image_urls: list[str], labels: list[str], num_outputs: int, user_prompt: str | None = None) -> list[str]:
    """Node: 'LLM+Vision: understand product images (+ user_prompt if present) -> N prompts'.
    Always routed through the VLM, with or without a user prompt — an advanced prompt is
    creative direction layered onto this step (same principle as On-Model Shots), not a
    bypass that would just repeat one literal prompt N times.

    Validates the response (exact-N, non-empty, no duplicates) and retries the VLM call once
    on failure before giving up — a malformed or degenerate shot list would otherwise burn a
    real Seedream generation call per bad prompt, which is the expensive way to find out."""
    prompt_text = (
        "Product reference images, in order:\n" + "\n".join(labels)
        + f"\n\nWrite exactly {num_outputs} distinct catalog-shot prompts for this product."
    )
    if user_prompt:
        prompt_text += (
            f"\n\nUser's additional instructions (creative direction — still preserve exact "
            f"product fidelity and the adult-only rule above): {user_prompt}"
        )
    prompt_text += f" Return ONLY a JSON array of {num_outputs} strings, nothing else."

    last_error = None
    for attempt in range(2):  # one retry
        prompts = _vlm_json_call(
            model=MODEL_CONFIG["prompt_writer"], system=SHOT_PROMPT_WRITER_SYSTEM,
            prompt=prompt_text, image_urls=image_urls, max_tokens=1500,
        )
        ok, reason = _validate_shot_prompts(prompts, num_outputs)
        if ok:
            return prompts
        last_error = reason
        print(f"[attempt {attempt + 1}] invalid shot list ({reason}) — retrying")

    raise ValueError(f"Shot-list planning failed twice: {last_error}")


# =============================================================================
# 5. Generation — bytedance/seedream/v5/lite/edit
# =============================================================================

def run_catalog_generation(
    prompt: str, image_urls: list[str],
    resolution_mode: str = "standard", aspect_ratio: str = "1:1",
    custom_width: int | None = None, custom_height: int | None = None,
    num_images: int = 1,
) -> list[str]:
    args = {
        "prompt": prompt,
        "image_urls": image_urls,
        "image_size": build_image_size(resolution_mode, aspect_ratio, custom_width, custom_height),
        "num_images": num_images,
        "max_images": 1,
        "enable_safety_checker": True,  # do not disable — second, independent safety layer
    }
    result = fal_client.subscribe(MODEL_CONFIG["catalog_generation"], arguments=args, with_logs=True)
    return [img["url"] for img in result["images"]]


# =============================================================================
# 6. Orchestrator
# =============================================================================

def run_catalog_photoshoot(
    product_paths: list[str],
    num_outputs: int = 6,
    resolution_mode: str = "standard",
    aspect_ratio: str = "1:1",
    custom_width: int | None = None,
    custom_height: int | None = None,
    user_prompt: str | None = None,
    on_image=None,
) -> dict:
    """Full flow, node for node: upload products -> safety check (images + prompt) ->
    LLM+Vision builds N shot prompts -> one generation call per shot, same input images
    reused every call, run in sequence ('in queue') -> collect outputs."""
    on_image = on_image or show

    image_urls, labels = assemble_product_images(product_paths)

    passed, reason = run_safety_check(image_urls, user_prompt)
    if not passed:
        raise ValueError(f"Blocked at safety check: {reason}")

    prompts = build_shot_prompts(image_urls, labels, num_outputs, user_prompt)

    outputs = []
    for i, p in enumerate(prompts):
        print(f"--- Shot {i+1}/{len(prompts)} ---\nPrompt: {p}\n")
        urls = run_catalog_generation(
            p, image_urls,
            resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
            custom_width=custom_width, custom_height=custom_height,
            num_images=1,
        )
        outputs.extend(urls)
        for u in urls:
            on_image(u, f"Shot {i+1}")

    return {"prompts": prompts, "image_urls": image_urls, "outputs": outputs}
