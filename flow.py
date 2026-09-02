"""ShootPX — On-Model Shots: shared flow logic.

One module, two frontends: shootpx_on_model_flow.ipynb (unattended, auto-approve) and
streamlit_app.py (interactive, human approve/regenerate). Neither frontend duplicates this
logic — they only call into it and render results their own way (the `on_image` callback
below is the seam: notebook defaults to `show()`, Streamlit passes `st.image`).

Every model is confirmed and read from `.env` via MODEL_CONFIG — nothing hardcoded. See the
mermaid flow diagram in shootpx_on_model_flow.ipynb's first cell for the full picture; every
function below is commented with which node(s) of that diagram it implements.
"""

import io
import json
import os
import re
from typing import TypedDict

import fal_client
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

assert os.environ.get("FAL_KEY"), "Set FAL_KEY in .env before running."

# --- MODEL_CONFIG: the ONE place to see/swap every model this flow calls. ---
# All read straight from .env, no code-side fallback drift risk.
MODEL_CONFIG = {
    "final_generation": os.environ["FINAL_GENERATION_MODEL"],   # Seedream 4.5 edit
    "text_to_image": os.environ["TEXT_TO_IMAGE_MODEL"],          # Seedream 4.5 text-to-image
    "vlm_endpoint": "openrouter/router/vision",                  # fal slug every VLM/LLM call below goes through
    "prompt_writer": os.environ["PROMPT_WRITER_MODEL"],          # Gemini 2.5 Flash, via vlm_endpoint
    "safety_check": os.environ["SAFETY_CHECK_MODEL"],            # Gemini 2.5 Flash, via vlm_endpoint
}


# =============================================================================
# 1. Shared helpers
# =============================================================================

def to_hosted_url(path_or_url: str) -> str:
    """Return a fal-hosted URL for a local file, or pass a URL through unchanged."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return fal_client.upload_file(path_or_url)


def show(url_or_path: str, caption: str = ""):
    """Default `on_image` renderer — notebook/IPython display. Streamlit passes its own
    (st.image) instead; nothing in this module calls show() directly except the defaults
    below, so swapping frontends never touches this function."""
    from IPython.display import display  # imported here so this module has no hard IPython dep

    if url_or_path.startswith("http"):
        img = Image.open(io.BytesIO(requests.get(url_or_path, timeout=30).content))
    else:
        img = Image.open(url_or_path)
    print(caption)
    display(img)


def fetch_image_bytes(url: str) -> bytes:
    """Raw GET — used both by download_image() below and by streamlit_app.py's download
    buttons (which need bytes in memory, not a file on disk)."""
    return requests.get(url, timeout=30).content


def download_image(url: str, out_path: str) -> str:
    """Save a generated image to local disk. fal output URLs are NOT permanent (confirmed:
    subject to the same X-Fal-Object-Lifecycle-Preference retention controls as uploads, and
    are permanently deleted once they expire) — call this before you need the file long-term,
    not just store the URL."""
    with open(out_path, "wb") as f:
        f.write(fetch_image_bytes(url))
    return out_path


def _strip_json_fence(text: str) -> str:
    """Gemini (and most chat models) often wrap JSON in ```json ... ``` even when told not
    to. Strip that before parsing so we don't hard-fail on the fence, not the content."""
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
    gemini-3.1-pro-preview) fall back to pulling out the outermost {...}/[...] block if the
    text still isn't valid JSON on its own — a reasoning model's thinking trace can land in
    the same `output` string ahead of its actual JSON answer."""
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
    `model` picks the underlying model. Works with or without images (image_urls=None is a
    plain text LLM call, used by generate_model_prompt_and_validate below). `reasoning: True` is
    mandatory for some models on this endpoint (confirmed, 2026-08-29: gemini-3.1-pro-preview
    400s without it — "Reasoning is mandatory for this endpoint and cannot be disabled") —
    sent unconditionally, since non-reasoning models just ignore it. Must return a bare JSON
    object/array as text; parsed via _extract_json_text (handles reasoning trace + fences)."""
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
    output = result.get("output", "")
    try:
        return json.loads(_extract_json_text(output), strict=False)
    except json.JSONDecodeError as e:
        # Real failure, 2026-08-31: a reasoning-mandatory model (reasoning=True above) can
        # burn its whole max_tokens budget on the hidden reasoning trace and return
        # near-empty/non-JSON content — every extraction strategy in _extract_json_text then
        # fails too. A bare JSONDecodeError gives no way to tell that apart from a real
        # parsing bug, so surface what the model actually returned.
        snippet = output[:500] + ("..." if len(output) > 500 else "")
        raise RuntimeError(
            f"{model} did not return valid JSON ({e}). Raw output: {snippet!r}"
        ) from e


# =============================================================================
# 2. Model Source — generate / upload / default
# =============================================================================
# Matches the left/middle/right branches of the diagram: both NSFW gates (on the LLM-written
# description, before spending a generation call; and on the generated image, before showing
# a preview) and the approve/regenerate loop for the `generate` path.

MODEL_LIBRARY = {
    # "preset_id": "path/or/url/to/model.jpg" — pre-vetted, skips the NSFW check below.
    "preset_1": "assets/models/preset_1.jpg",
    "preset_2": "assets/models/preset_2.jpg",
}

# Deterministic minor-age blocklist — used on: (a) the "generate" path's additional_notes, so
# free text can never override the adult-only structured picker (e.g. age="25-30" + notes="make
# her look 10" must still block); (b) the user's free-text generation instructions, alongside
# NSFW_BLOCKED_TERMS below (see section 6). Phrasing, not just single words, since "10 year old"
# and "under 18" are the actual real-world attack shapes, not just the word "child" in isolation.
MINOR_AGE_BLOCKED_PATTERNS = [
    "child", "minor", "teen", "kid", "toddler", "baby", "underage", "under 18", "under-18",
    "under age", "schoolgirl", "school girl", "schoolboy", "school boy",
]
MINOR_AGE_NUMBER_RE = re.compile(r"\b(\d{1,2})\s*[-\s]?year", re.IGNORECASE)


def check_text_for_minor_age(text: str) -> None:
    """Deterministic gate — no VLM. Raises ValueError if `text` implies a minor/underage
    subject, by keyword/phrase match or by an explicit age number under 18 (e.g. "10 year old",
    "16-year-old"). Used to stop free text (additional_notes, user_prompt) from describing a
    minor even when a structured picker elsewhere is adult-only — the picker being adult-only
    does not stop someone typing an age into a text field."""
    lowered = text.lower()
    for phrase in MINOR_AGE_BLOCKED_PATTERNS:
        if phrase in lowered:
            raise ValueError(f"Blocked: text implies a minor/underage subject ('{phrase}').")
    for match in MINOR_AGE_NUMBER_RE.finditer(lowered):
        if int(match.group(1)) < 18:
            raise ValueError(f"Blocked: text specifies an age under 18 ('{match.group(0)}').")

# Structured-picker options for the "generate" path. Deliberately adult-only — no "children"/
# "youth" bracket, unlike a typical reference UI — this has to stay consistent with the nsfw/
# minor safety gates already built into this flow, not just mirror a competitor's screenshot.
GENDER_OPTIONS = ["Female", "Male"]
# Bracket bumped from "(20s)" to "(25-30)" — real-world failure, 2026-08-29: "20s" alone
# doesn't anchor the text-to-image model away from youthful/ambiguous faces, and the
# safety-check VLM correctly flagged the result as looking underage. Starting the range at
# 25 gives real headroom before the ambiguous-age zone.
AGE_BRACKET_OPTIONS = ["Young adult (25-30)", "Adult (30s-40s)", "Mature adult (50+)"]
ETHNICITY_OPTIONS = ["South Asian", "East Asian", "Southeast Asian", "Middle Eastern", "Black", "Hispanic/Latino", "White/Caucasian", "Mixed/Other"]
SKIN_TONE_OPTIONS = ["Fair", "Light", "Tan", "Deep"]
BODY_TYPE_OPTIONS = ["Slim", "Athletic", "Average", "Curvy", "Plus size"]

# Real-world failure, 2026-08-29: a prompt saying "a young adult female in her 20s" produced
# an image the safety-check VLM flagged as looking underage — passing the age bracket through
# as a bare label doesn't push the image model toward unambiguous adult features. Now
# explicitly instructed to describe visual maturity signals, not just state a number.
#
# Architecture, 2026-09-03: "one Gemini request should perform every logically related Gemini
# job that can be safely combined" — this used to be two calls (write the prompt, then a
# separate VLM call to check additional_notes for unsafe intent). Now it's one call that does
# both: judges additional_notes for unsafe/inappropriate intent AND writes the model prompt in
# the same response — never both prompt-writing AND safety-judgment as separate round trips for
# the same input. The deterministic check_text_for_minor_age keyword/regex gate still runs
# BEFORE this call (see generate_model_prompt_and_validate below) as a free pre-filter for the
# obvious cases; this system prompt's "allowed" judgment is a second, independent layer for
# subtler phrasing that pre-filter can't catch — and per section 6's rule ("AI can classify,
# code makes the final business decision"), the caller still enforces "allowed" in Python, it is
# never trusted blindly.
MODEL_PROMPT_WRITER_SYSTEM = """You have two jobs in one response for an e-commerce fashion photography studio: (1) judge whether the user's additional notes are safe to act on, and (2) if safe, write the text-to-image prompt.

JOB 1 — SAFETY JUDGMENT:
Given the structured attributes and the user's additional notes (if any), decide "allowed": false if the additional notes imply, even subtly or indirectly, an underage/age-ambiguous subject, or sexually explicit/inappropriate content. Ordinary style/appearance notes (hair, expression, lighting, setting) are always "allowed": true.

JOB 2 — PROMPT WRITING (only if allowed is true):
Write a single, vivid, photorealistic text-to-image prompt describing a human model for a professional ecommerce fashion photography studio portrait.

Given structured attributes (including gender, age bracket, ethnicity/heritage, skin tone and body type) and optional additional notes, write ONE descriptive paragraph (not a list) that a text-to-image model can use directly: appearance, studio setting, neutral standing pose, lighting, photographic quality. Reflect the requested ethnicity/heritage naturally and respectfully in the described facial features and styling, consistent with the requested skin tone.

The model should be attractive, photogenic and camera-ready the way a professional fashion catalog model is: polished, well-groomed, confident, flattering studio lighting, a genuine and warm expression. That is the actual point of this step — produce a good-looking model a fashion brand would want to use, not just a technically compliant photo.

The model must also read as unmistakably an adult — mature, fully developed adult facial structure and body proportions, adult styling and grooming, confident adult posture and expression. Never describe or imply a youthful, adolescent, or age-ambiguous appearance, even for the youngest requested bracket. If there is any doubt, describe the model as visibly closer to 30 than to 18 — err toward more visibly mature, never less.

Being attractive and being unambiguously adult are not in tension: professional fashion models are typically confident adults in their mid-20s to 30s — describe exactly that standard, not a compromise between "attractive" and "adult."

Return ONLY a JSON object: {"allowed": true|false, "reason": "<short reason, empty string if allowed>", "prompt": "<the single descriptive paragraph, empty string if not allowed>"}."""


def generate_model_prompt_and_validate(gender: str, age_bracket: str, ethnicity: str, skin_tone: str, body_type: str, additional_notes: str = "") -> dict:
    """Node: feeds 'Structured picker' + 'Optional free-text refinement' into ONE Gemini call
    that both judges additional_notes for unsafe intent AND writes the text-to-image prompt —
    replaces the old generate_model_prompt_via_llm() + a separate notes-safety VLM check.
    Returns {"allowed": bool, "reason": str, "prompt": str}. Caller (generate_model_candidate)
    still enforces "allowed" itself — this function never raises on an unsafe judgment, it only
    reports it, per section 6's "AI can classify, code makes the final business decision"."""
    attrs = {"gender": gender, "age_bracket": age_bracket, "ethnicity": ethnicity, "skin_tone": skin_tone, "body_type": body_type}
    prompt_text = (
        "Structured attributes: " + json.dumps(attrs)
        + (f"\nAdditional notes from user: {additional_notes}" if additional_notes else "")
        + '\n\nReturn ONLY the requested JSON object.'
    )
    result = _vlm_json_call(
        # max_tokens=800, not 300: PROMPT_WRITER_MODEL is a reasoning-mandatory model (see
        # _vlm_json_call's `reasoning: True` note) — 300 left too little headroom for both
        # the hidden reasoning trace and the actual JSON answer, producing truncated/
        # non-JSON output (real failure, 2026-08-31: JSONDecodeError with no diagnosable
        # cause until _vlm_json_call started surfacing the raw output above).
        model=MODEL_CONFIG["prompt_writer"], system=MODEL_PROMPT_WRITER_SYSTEM,
        prompt=prompt_text, image_urls=None, max_tokens=800,
    )
    return {
        "allowed": bool(result.get("allowed", True)),
        "reason": result.get("reason", ""),
        "prompt": result.get("prompt", ""),
    }


# Architecture, 2026-09-03 (see the "child + intimate garment" design discussion): safety now
# happens at the point where the relevant information becomes known, not "VLM at every stage".
# The generate path's age is a deterministic, adult-only structured picker (AGE_BRACKET_OPTIONS)
# — free text (additional_notes) must never be able to override that into a minor. That's a
# deterministic keyword/phrase check (check_text_for_minor_age above), not a VLM call: it's not
# ambiguous natural language needing interpretation, it's specific blocked phrasing/ages. No
# separate image-NSFW check runs on the generated model image any more — a generated model is
# always age_status="adult" once its notes pass this gate, because the picker never offers a
# minor option and the notes can't smuggle one in.
def validate_generate_notes(additional_notes: str) -> None:
    """Node: 'generate' path's only safety gate, pass-or-raise. Runs before the paid
    text-to-image call. Raises ValueError if additional_notes implies a minor."""
    if additional_notes:
        check_text_for_minor_age(additional_notes)


# Calibration fix, 2026-08-29: an actually-fine adult model (loungewear, ordinary studio
# shot) got a false-positive "appears underage" flag. Root cause — the previous wording
# ("clearly 18+, no ambiguity") gave the VLM no way to distinguish "youthful bone structure
# on an obvious adult" from "genuinely could be a minor", so it flagged on ANY youthfulness.
# Recalibrated to require a specific, real reason to suspect a minor, and to explicitly say
# normal underwear/lingerie e-commerce photography isn't inherently "explicit" — this is a
# bra/underwear business, that content is the product, not a violation.
#
# This is now ONLY used on the "upload" path (see validate_uploaded_model below) — an arbitrary
# user-supplied photo is the one case the app doesn't control the contents of, so it's the one
# case that actually needs a vision check. Presets are pre-vetted; generated models are age-
# gated deterministically above. Returns age_status so the caller can feed the later
# child+intimate-product gate, not just a clean/not-clean bool.
def check_image_nsfw(image_url: str) -> dict:
    """Node: 'Check for nsfw image' (upload path only). VLM-based — model is
    MODEL_CONFIG['safety_check'], swap it there. Returns {"clean": bool, "age_status":
    "adult"|"minor"|"uncertain", "reason": str}."""
    result = _vlm_json_call(
        model=MODEL_CONFIG["safety_check"],
        system=(
            "You are a content-safety classifier for an e-commerce fashion photo pipeline. "
            "Given one image, check two things only:\n"
            "1. Age: classify the person as \"adult\" (unmistakably 18+), \"minor\" (a "
            "genuine, specific reason to suspect under 18 — not just youthful-looking), or "
            "\"uncertain\" (genuinely cannot tell). Professional adult models often have soft "
            "studio lighting, natural/minimal makeup, or delicate features and are still "
            "obviously adult — do NOT classify as minor or uncertain on youthfulness alone, "
            "only with a real, articulable reason.\n"
            "2. Explicit content: is the image sexually explicit (visible nudity, exposed "
            "genitalia/nipples, sexual acts or poses)? Ordinary e-commerce underwear/lingerie/"
            "swimwear photography — the actual product being sold — is NOT explicit on its own.\n"
            'Respond with ONLY a JSON object: {"clean": true|false, "age_status": '
            '"adult"|"minor"|"uncertain", "reason": "short reason"}.'
        ),
        prompt="Classify this image per the rules in the system prompt.",
        image_urls=[image_url],
        max_tokens=200,
    )
    return {
        "clean": bool(result["clean"]),
        "age_status": result.get("age_status", "uncertain"),
        "reason": result.get("reason", ""),
    }


def generate_model_via_text2image(description: str, image_size: str = "portrait_4_3") -> str:
    """Node: 'Text-to-Image call'."""
    result = fal_client.subscribe(
        MODEL_CONFIG["text_to_image"],
        arguments={"prompt": description, "image_size": image_size, "num_images": 1},
        with_logs=True,
    )
    return result["images"][0]["url"]


def generate_model_candidate(gender: str, age_bracket: str, ethnicity: str, skin_tone: str, body_type: str, additional_notes: str = "") -> dict:
    """One full attempt: deterministic minor-age pre-filter on additional_notes (free, catches
    the obvious cases before spending a Gemini call) -> ONE combined Gemini call that judges
    the notes AND writes the prompt (generate_model_prompt_and_validate) -> text-to-image.
    "allowed" from Gemini is enforced here in Python, not trusted blindly (section 6's "AI can
    classify, code makes the final business decision") — a False raises just like the
    deterministic pre-filter does, so both layers actually block, neither just advises.
    No image-NSFW/age vision check on the result — the generate path is deterministically adult
    by construction (AGE_BRACKET_OPTIONS is adult-only and both safety layers above already stop
    additional_notes from smuggling in a minor), so there is nothing left for a vision check to
    catch that these gates didn't already catch. age_status is always "adult" here. This is the
    single-attempt building block; resolve_model_via_generate below loops it for unattended use,
    and streamlit_app.py calls it directly per button click for interactive approve/regenerate.
    """
    validate_generate_notes(additional_notes)
    result = generate_model_prompt_and_validate(gender, age_bracket, ethnicity, skin_tone, body_type, additional_notes)
    if not result["allowed"]:
        raise ValueError(f"Blocked at prompt-level safety check: {result['reason']}")
    url = generate_model_via_text2image(result["prompt"])
    return {"url": url, "age_status": "adult", "description": result["prompt"]}


def resolve_model_via_generate(
    gender: str, age_bracket: str, ethnicity: str, skin_tone: str, body_type: str,
    additional_notes: str = "", approve_fn=lambda url: True, on_image=None, max_attempts: int = 3,
) -> dict:
    """Nodes: structured picker -> free-text refinement -> minor-age gate -> Text-to-Image
    call -> preview -> Approve?. Loops on a 'No, regenerate' from approve_fn, up to
    max_attempts (no nsfw-retry loop any more — see generate_model_candidate). Unattended/
    notebook use — for an interactive UI, call generate_model_candidate() per button click
    instead (see streamlit_app.py) since a real approve/regenerate loop needs UI state between
    attempts, not a blocking Python loop.

    approve_fn(url) -> bool is the human-approval hook; defaults to auto-approve so this runs
    unattended in a notebook. on_image(url, caption) renders each preview; defaults to show().
    Returns {"url": ..., "age_status": "adult"}.
    """
    on_image = on_image or show
    for attempt in range(1, max_attempts + 1):
        candidate = generate_model_candidate(gender, age_bracket, ethnicity, skin_tone, body_type, additional_notes)
        on_image(candidate["url"], f"Preview — attempt {attempt}")
        if approve_fn(candidate["url"]):
            return {"url": candidate["url"], "age_status": candidate["age_status"]}
        print(f"[attempt {attempt}] not approved — regenerating")

    raise RuntimeError(f"Could not produce an approved model image in {max_attempts} attempts.")


def validate_uploaded_model(path_or_url: str) -> dict:
    """Nodes: Upload Model -> Check for nsfw image -> Upload to hosted storage. The one model
    source the app doesn't control the contents of, so it's the one that needs a real vision
    check — for age status (feeds the later child+intimate-product gate) and NSFW. An NSFW
    upload can't be regenerated — raise and let the caller ask for a different file ('If NSFW'
    loops back to Model Source? in the diagram). Returns {"url": ..., "age_status": ...}."""
    hosted_url = to_hosted_url(path_or_url)
    result = check_image_nsfw(hosted_url)
    if not result["clean"]:
        raise ValueError(f"Uploaded model image flagged nsfw ({result['reason']}) — please upload a different photo.")
    return {"url": hosted_url, "age_status": result["age_status"]}


def resolve_model_via_default(preset_id: str) -> dict:
    """Node: Pick from Model Library — presets are pre-vetted trusted assets: no nsfw/age check,
    always age_status="adult"."""
    return {"url": to_hosted_url(MODEL_LIBRARY[preset_id]), "age_status": "adult"}


def resolve_model_image(
    mode: str,
    gender: str = "", age_bracket: str = "", ethnicity: str = "", skin_tone: str = "", body_type: str = "",
    additional_notes: str = "",
    upload_path: str | None = None,
    preset_id: str | None = None,
    approve_fn=lambda url: True,
    on_image=None,
) -> dict:
    """mode: 'generate' | 'upload' | 'default' — dispatches to the three branches above and
    returns {"url": model_image_url, "age_status": "adult"|"minor"|"uncertain"}."""
    if mode == "generate":
        return resolve_model_via_generate(gender, age_bracket, ethnicity, skin_tone, body_type, additional_notes, approve_fn, on_image)
    if mode == "upload":
        return validate_uploaded_model(upload_path)
    if mode == "default":
        return resolve_model_via_default(preset_id)
    raise ValueError(f"Unknown model mode: {mode}")


# =============================================================================
# 3. Products + reference images
# =============================================================================
# A "product" is one physical item the shot needs to show worn/carried — a top, a pair of
# shoes, a watch, a bag, anything. Multiple photos of the SAME item are multiple images on ONE
# product; a second item (e.g. Bottom vs Top) is a second product. This replaced a flat
# garment_paths list + a user-typed garment_type string (see section 5 below for why) — the
# user no longer says what a product IS, only uploads it and optionally gives it a label.


class Product(TypedDict):
    id: str               # slug, deduped — used internally, never shown to the user
    label: str             # display label: "Top", "Bottom", "Watch", "Garment 3", ...
    image_paths: list[str]  # 1+ local paths or URLs, all of the same physical product


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_") or "product"


def make_product(label: str, image_paths: list[str], existing_ids: set[str] | None = None) -> Product:
    """Builds one Product from a UI slot. A blank label defaults to 'Garment' — the user is
    never required to type anything (Single Garment mode never calls this with a label at
    all; see streamlit_app.py). existing_ids lets the caller dedup labels typed twice (two
    custom slots both called "Watch") into distinct ids, since classify_products_via_vlm and
    assemble_inputs below both key off id."""
    label = label.strip() or "Garment"
    existing_ids = existing_ids if existing_ids is not None else set()
    base_id = _slugify(label)
    product_id, n = base_id, 2
    while product_id in existing_ids:
        product_id = f"{base_id}_{n}"
        n += 1
    return {"id": product_id, "label": label, "image_paths": list(image_paths)}


# Node: 'Assemble ordered image list + labels' -> 'Total images <= 10?'. Raising here (rather
# than silently trimming) matches the diagram's explicit reject-and-suggest-trim step.

def assemble_inputs(model_image: str, products: list[Product], reference_images: list[str] | None = None):
    """Returns (image_urls, labels) in order. Raises if the fal input-count ceiling (10) is
    exceeded, naming exactly how many images need to be trimmed."""
    reference_images = reference_images or []
    total_product_images = sum(len(p["image_paths"]) for p in products)
    total = 1 + total_product_images + len(reference_images)
    if total > 10:
        raise ValueError(
            f"{total} images (1 model + {total_product_images} product + "
            f"{len(reference_images)} reference) exceeds the 10-image limit — "
            f"trim {total - 10} image(s) and try again."
        )

    urls = [model_image]
    labels = ["#Image1 = model reference"]
    for product in products:
        n = len(product["image_paths"])
        for i, path in enumerate(product["image_paths"], start=1):
            urls.append(path)
            angle = f" ({i}/{n})" if n > 1 else ""
            labels.append(f"#Image{len(labels)+1} = product '{product['label']}'{angle}, exact product, preserve fidelity")
    for r in reference_images:
        urls.append(r)
        labels.append(f"#Image{len(labels)+1} = style/pose reference only")

    image_urls = [to_hosted_url(u) for u in urls]
    return image_urls, labels


# =============================================================================
# 4. Output settings — aspect ratio / resolution
# =============================================================================
# Only final_generation (Seedream 4.5 edit) is wired, so this maps straight to its
# image_size preset vocabulary.

ASPECT_RATIO_MAP = {
    "1:1": "square_hd",
    "3:4": "portrait_4_3",
    "9:16": "portrait_16_9",
    "4:3": "landscape_4_3",
    "16:9": "landscape_16_9",
}

# Confirmed live against fal's Seedream 4.5 edit schema, 2026-08-29: image_size is either one
# of the named presets above (aspect ratio + resolution baked into one value — no separate
# resolution control for those), the two resolution-priority values below (model decides
# aspect ratio from the inputs), or a custom {width, height} object bound by these limits.
RESOLUTION_MODES = ["standard", "auto_2K", "auto_4K", "custom"]
SEEDREAM_MIN_DIM = 1920
SEEDREAM_MAX_DIM = 4096
SEEDREAM_MIN_TOTAL_PX = 2560 * 1440   # 3,686,400
SEEDREAM_MAX_TOTAL_PX = 4096 * 4096   # 16,777,216


def validate_custom_size(width: int, height: int) -> None:
    """Raises ValueError if width/height don't satisfy Seedream's documented constraint:
    each side 1920-4096px, OR total pixel count 2560x1440-4096x4096."""
    per_axis_ok = SEEDREAM_MIN_DIM <= width <= SEEDREAM_MAX_DIM and SEEDREAM_MIN_DIM <= height <= SEEDREAM_MAX_DIM
    total_px = width * height
    total_ok = SEEDREAM_MIN_TOTAL_PX <= total_px <= SEEDREAM_MAX_TOTAL_PX
    if not (per_axis_ok or total_ok):
        raise ValueError(
            f"{width}x{height} isn't a valid size: each side must be "
            f"{SEEDREAM_MIN_DIM}-{SEEDREAM_MAX_DIM}px, or total pixels must be "
            f"{SEEDREAM_MIN_TOTAL_PX:,}-{SEEDREAM_MAX_TOTAL_PX:,} (you gave {total_px:,})."
        )


def build_image_size(resolution_mode: str = "standard", aspect_ratio: str = "3:4", custom_width: int | None = None, custom_height: int | None = None):
    """resolution_mode: 'standard' (named preset from aspect_ratio, via ASPECT_RATIO_MAP) |
    'auto_2K' | 'auto_4K' (model decides aspect ratio) | 'custom' (custom_width/custom_height,
    validated against Seedream's limits above)."""
    if resolution_mode == "standard":
        if aspect_ratio not in ASPECT_RATIO_MAP:
            raise ValueError(f"Unknown aspect_ratio {aspect_ratio!r} — one of {list(ASPECT_RATIO_MAP)}")
        return ASPECT_RATIO_MAP[aspect_ratio]
    if resolution_mode in ("auto_2K", "auto_4K"):
        return resolution_mode
    if resolution_mode == "custom":
        if custom_width is None or custom_height is None:
            raise ValueError("resolution_mode='custom' requires custom_width and custom_height")
        validate_custom_size(custom_width, custom_height)
        return {"width": custom_width, "height": custom_height}
    raise ValueError(f"Unknown resolution_mode: {resolution_mode!r} — one of {RESOLUTION_MODES}")


# =============================================================================
# 5. Product understanding + user-instruction safety + pose-prompt writing — ONE Gemini call
# =============================================================================
# Architecture, 2026-09-03: "one Gemini request should perform every logically related Gemini
# job that can be safely combined." This used to be three separate calls: classify_products_via_vlm
# (product type/category/body_placement) -> a user-instruction-safety VLM escalation -> then
# generate_pose_prompts_via_vlm (N pose prompts, using whichever system prompt the classification
# picked). That ordering is now impossible to keep as separate calls without ALSO merging them,
# because a merged call can't know the category before it runs (chicken-and-egg: category picks
# the system prompt, but classification is now part of what that one call does) — so instead of
# two system prompts (GENERAL/INTIMATE) selected AFTER classification, there is now ONE unified
# system prompt that includes classification instructions, both rule sets, AND pose-writing, and
# Gemini self-applies the intimate-specific fidelity/safety rules only to the products it
# classifies as intimate. See the "Merge B" design discussion, 2026-09-03, for why this shape was
# chosen over keeping classification as a separate first call.
#
# The deterministic Python gates around this call are UNCHANGED and still authoritative:
# check_model_product_compatibility (child+intimate block) still runs on this call's own
# overall_category output, in Python, after the call returns — Gemini reports the category,
# Python still makes the final business decision (section 6). Likewise check_user_text_safety's
# deterministic keyword pass still runs on the raw user_prompt BEFORE this call, for the same
# "free pre-filter first" reason as generate_model_candidate above; this call's
# user_instruction_safe is a second, independent layer for subtler phrasing, not a replacement.

# Product review pass, 2026-08-29: shifted from a short imperative-checklist prompt to a full
# creative-director brief — the goal is attractive, commercially usable output across every
# product category (bra, t-shirt, pants, shoes, hats, watches, bags, ...), not sterile catalog
# shots. PRODUCT-CATEGORY FRAMING tells the VLM to adapt composition per category itself
# (e.g. show a hand for a watch, a leg for a shoe, half/full body when the product needs it)
# rather than us hardcoding a framing rule per category. Kept verbatim from the pre-merge
# PROMPT_WRITER_SYSTEM_GENERAL/INTIMATE prompts below, combined into one system prompt instead
# of two selected by a prior classification call (see this section's header comment).
SHOOT_ANALYST_SYSTEM = """You are the creative image director for ShootPX, an AI fashion ecommerce photography system, and you have THREE jobs in one response:

JOB 1 — PRODUCT UNDERSTANDING:
You are given a labeled sequence of images: one model reference (#Image1), one or more products (a product may have multiple angle images sharing the same label), and optional style/pose references. For each product listed, determine:
- type: a short specific noun phrase, e.g. "bra", "t-shirt", "sneakers", "watch", "baseball cap", "tote bag"
- category: "intimate" if it is intimate apparel, underwear, lingerie, or swimwear worn as underwear — otherwise "general"
- body_placement: one of "upper_body", "lower_body", "full_body", "feet", "head", "wrist", "hand", "neck", "shoulder", "waist", "carried" — or another short specific value if none of these fit

JOB 2 — USER INSTRUCTION SAFETY:
If a user creative instruction is supplied, judge "user_instruction_safe": false if it implies, even subtly or indirectly, sexually explicit/inappropriate content, nudity, or an underage subject. Ordinary style/scene direction — including mentioning an intimate product by name, since that may be the actual product being sold — is NOT a violation on its own. If no user instruction is supplied, user_instruction_safe is true.

JOB 3 — POSE PROMPT WRITING (skip if user_instruction_safe is false — return empty prompts):
Your prompts are sent directly to a professional image-editing model (Seedream) that takes the same numbered reference images you are shown — #Image1, #Image2, #Image3, etc. — and edits them together. That editing model can already see every pixel of every reference image. It does not need you to describe what a reference image looks like — it needs you to tell it, briefly, which image plays which role and what to do with them.

The supplied product is the hero of the image. The model, pose, lighting and environment exist to present the product beautifully.

IMAGE REFERENCE SYNTAX (read this carefully — this is the most important rule):
Every prompt you write MUST refer to the supplied images by their number tags exactly as given to you (e.g. #Image1, #Image2), never by a description of what they show. Use the tags to assign roles, e.g.: "Dress the model in #Image1 with the product from #Image2." Do NOT restate, re-describe or re-specify any visual attribute that is already visible in a reference image — not its color, pattern, print, texture, silhouette, stitching, trim, strap, closure, hardware, or any other detail. The editing model copies those details directly from the pixels of the reference image; writing them out in words does not improve fidelity, it invites the model to "correct" or reinterpret details that were already correct, which is a common cause of hallucinated colors, warped patterns and broken stitching. If a detail is visible in a reference image, your job is to point at that image with its tag, not to describe the detail. Only write words for what is NOT already visible in a reference image: the pose, camera angle, framing, composition, lighting, environment, and how the product and model should interact.

Prioritize, in this order:
1. Accurate preservation of the supplied product (via correct #ImageN referencing, not description)
2. Clear and attractive product presentation
3. Natural fit and believable interaction with the model
4. Strong ecommerce composition
5. Professional fashion photography
6. Attractive but restrained styling
7. Natural model expression and pose
8. Clean, polished background and lighting

PRODUCT:
Treat the tagged product image(s) as the source of truth. Reference them by tag only. Never re-describe the product's design, silhouette, proportions, construction, materials, colors, patterns, textures, seams, stitching, trims or hardware — the editing model already sees them in the tagged image. Do not unnecessarily redesign, simplify or invent product details. Do not invent a different product or turn a reference garment into a generic version of the product.

COMPOSITION:
Choose framing appropriate to the product category. The product should occupy a meaningful portion of the image and remain easy to see. Avoid unnecessary empty space. Avoid poses, hair, hands, arms, props or camera angles that hide important product details. Do not force every product into the same crop.

PRODUCT-CATEGORY FRAMING:
Choose framing based on what is being sold.
Upper-body garments: use upper-body or half-body framing when appropriate.
Full-body garments: show enough of the body to clearly present the complete garment without excessive empty space.
Footwear: use a lower-body or full-body composition that makes the footwear clearly visible.
Watches, jewelry and small accessories: use a closer composition where the product is large and easy to evaluate.
Hats and headwear: keep the head and product clearly visible.
Bags and carried accessories: show the complete product and, where useful, how it is naturally carried or worn.
Do not force a fixed camera crop when it would reduce product visibility.

FOR ANY PRODUCT YOU CLASSIFIED AS "intimate" IN JOB 1, ALSO APPLY THESE ADDITIONAL RULES (skip this block entirely for general-category products):
Create premium, attractive, realistic ecommerce fashion photography for brands selling on marketplaces such as Myntra, Amazon, Flipkart and Shopify — high-quality mainstream fashion retail photography: polished, confident, modern, attractive and commercially usable. The intimate-apparel product is the commercial hero; the adult model exists to present it naturally and professionally. For intimate upper-body garments, generally favor close or medium upper-body compositions where the garment is large enough to clearly evaluate while still presenting the model naturally. The garment should receive more visual attention than the model's face, background or environment. INTIMATE APPAREL SAFETY: the model must be clearly an adult; keep the photography suitable for mainstream fashion ecommerce; use tasteful fashion poses and professional commercial presentation; avoid sexualized, explicit or boudoir-style direction.

PHOTOGRAPHY:
Create the visual quality of a premium commercial fashion shoot rather than a generic AI-generated image. Use believable professional lighting, realistic shadows, accurate color, natural skin and material texture, realistic perspective and tasteful retouching, refined but realistic. Do not make every image look identical — allow tasteful variation in pose, camera angle, expression, lighting and composition while maintaining product visibility.

STYLE:
Aim for aspirational, polished and contemporary ecommerce fashion photography. The image should feel attractive enough for a brand's storefront while remaining commercially practical.

REFERENCES:
Use the tagged product and model images as the primary source of truth, referenced only by tag. Additional reference images may guide composition, pose, lighting, framing or photographic style — reference those by tag too, and only for the non-visual direction they add (e.g. "match the camera angle of #Image4"), never by describing their contents. Do not copy unrelated products, branding, logos, text or identities from any reference image.

POSE VARIETY:
When generating multiple images, make each pose and camera angle meaningfully different while maintaining product visibility and commercial usefulness.

PROMPT STYLE:
Write short, direct instructions for the image-editing model, built around #ImageN tags. Do not write a long scene description. Do not describe any attribute already visible in a reference image — point at its tag instead. Do not overuse negative instructions. Every prompt must be self-contained and directly usable by the image-editing model.

Example of the shape to write (adapt the tags/roles/pose to the actual images and pose direction you were given — never copy this example's wording):
"#Image2 is the product. #Image1 is the model. Dress the model in #Image1 with the product from #Image2. Front-facing hero composition, confident natural pose, clean studio lighting."

Return ONLY a JSON object of this exact shape:
{"products": [{"type": "...", "category": "intimate"|"general", "body_placement": "..."}, ...one entry per product in the order listed...], "user_instruction_safe": true|false, "user_instruction_reason": "<short reason, empty string if safe or no instruction>", "prompts": ["...", ...N strings, empty array if user_instruction_safe is false...]}."""


# Product review pass, 2026-08-29: category-neutral composition *directions*, not hardcoded
# upper-body framing — the system prompt's PRODUCT-CATEGORY FRAMING section is what actually
# adapts these to bra vs. t-shirt vs. shoe vs. watch vs. hat vs. bag, per product.
DEFAULT_POSES = [
    "front-facing product hero composition",
    "front three-quarter product-focused fashion composition",
    "subtle three-quarter side product presentation",
    "close product-focused commercial composition emphasizing product details and material quality",
]


def analyze_shoot_and_generate_prompts(
    image_urls: list[str], labels: list[str], products: list[Product], num_poses: int = 4, user_instruction: str | None = None,
) -> dict:
    """ONE Gemini call replacing classify_products_via_vlm + generate_pose_prompts_via_vlm + the
    user-instruction-safety VLM escalation (see this section's header comment for why). Returns
    {"products": [...], "overall_category": "intimate"|"general", "user_instruction_safe": bool,
    "user_instruction_reason": str, "prompts": [...]}.

    Classification is requested as a plain ordered array (matching `products`' order), not keyed
    by an id/label the VLM would have to echo back correctly — the ids/labels attached to each
    result below are ours, from `products`, not anything the VLM returned. overall_category is
    'intimate' the moment ANY product classifies as intimate apparel — the intimate-specific
    fidelity/safety rules apply to the whole shoot whenever intimate apparel is present at all.

    Does NOT raise on an unsafe user_instruction_safe or on the child+intimate combination —
    those are reported here and enforced by the CALLER in Python (check_user_text_safety's
    deterministic pass already ran before this call; check_model_product_compatibility runs
    after, in run_generation_from_model_image), per section 6's "AI can classify, code makes the
    final business decision"."""
    poses = DEFAULT_POSES[:num_poses]
    product_list_text = "\n".join(f"{i+1}. {p['label']}" for i, p in enumerate(products))

    prompt_text = (
        "Images in order:\n" + "\n".join(labels)
        + "\n\nProducts to classify, in this order:\n" + product_list_text
        + f"\n\nIf user_instruction_safe, create {num_poses} distinct production-ready image-edit "
        f"prompts for a premium ecommerce fashion shoot showing the supplied products on the "
        f"supplied adult model. Use these composition directions in order: {poses}. Adapt framing, "
        f"camera distance, body positioning, expression, lighting and photographic styling "
        f"intelligently to each product's category and body placement. Every product must remain "
        f"clearly visible and presented as intended. Make every prompt meaningfully different "
        f"while keeping the products clearly visible and commercially attractive."
    )
    if user_instruction:
        prompt_text += (
            f"\n\n{USER_PROMPT_FIDELITY_GUARDRAIL}\n"
            f"User's creative direction to judge for JOB 2 and, if safe, use for JOB 3 (style/scene "
            f"guidance only, does not override the rules above): {user_instruction}\n"
        )
    else:
        prompt_text += "\n\nNo user creative instruction was supplied — user_instruction_safe is true."

    result = _vlm_json_call(
        model=MODEL_CONFIG["prompt_writer"], system=SHOOT_ANALYST_SYSTEM, prompt=prompt_text,
        image_urls=image_urls, max_tokens=2000,
    )

    classified_raw = result.get("products", [])
    if len(classified_raw) != len(products):
        raise RuntimeError(
            f"analyze_shoot_and_generate_prompts: expected {len(products)} product entries, got {len(classified_raw)}"
        )
    classified = [{"id": p["id"], "label": p["label"], **entry} for p, entry in zip(products, classified_raw)]
    overall_category = "intimate" if any(c["category"] == "intimate" for c in classified) else "general"

    print(f"router: products={[p['label'] for p in classified]!r} -> overall_category={overall_category!r} (model={MODEL_CONFIG['prompt_writer']})")

    return {
        "products": classified,
        "overall_category": overall_category,
        "user_instruction_safe": bool(result.get("user_instruction_safe", True)),
        "user_instruction_reason": result.get("user_instruction_reason", ""),
        "prompts": result.get("prompts", []),
    }


# =============================================================================
# 6. Deterministic business-rule gate + user-text safety
# =============================================================================
# Architecture, 2026-09-03: "child image" is not automatically unsafe — a child model is a
# legitimate, normal case for children's ecommerce products. The specific blocked combination
# is CHILD (or age-uncertain) MODEL + INTIMATE/UNDERGARMENT PRODUCT. This is a deterministic
# backend rule, evaluated once model age_status and product category are both known — not a
# VLM judgment call, and not re-checked per pose.

CHILD_INTIMATE_BLOCK_MESSAGE = (
    "Child models cannot be used for undergarment or intimate-apparel generation."
)


def check_model_product_compatibility(age_status: str, overall_category: str) -> None:
    """Node: the child+intimate-product gate. Raises ValueError if age_status is "minor" or
    "uncertain" AND overall_category is "intimate". Any other combination (including a child
    model + general products, e.g. t-shirts/pants/shoes for children's ecommerce) is allowed."""
    if overall_category == "intimate" and age_status in ("minor", "uncertain"):
        raise ValueError(CHILD_INTIMATE_BLOCK_MESSAGE)


# Deterministic keyword pass for explicit/sexual user free-text intent — cheap, runs first,
# catches the obvious cases for free (same shape as MINOR_AGE_BLOCKED_PATTERNS above).
NSFW_BLOCKED_TERMS = [
    "nude", "naked", "topless", "sex", "sexual", "erotic", "porn", "explicit",
    "nsfw", "orgasm", "masturbat", "genitalia", "penis", "vagina", "fetish",
]


# Architecture, 2026-09-03 ("Merge B"): this used to also escalate to its OWN Gemini call for
# ambiguous phrasing. That escalation call is now folded into analyze_shoot_and_generate_prompts
# (section 5) — its "user_instruction_safe"/"user_instruction_reason" output IS that second,
# independent judgment layer, made in the same request that also classifies products and writes
# pose prompts, instead of a separate round trip for the same input. So this function is now
# ONLY the deterministic pre-filter: free, always runs first (catches the obvious cases and any
# minor-age phrasing before spending the merged call), never itself calls a VLM. The caller
# (run_generation_from_model_image) still enforces the merged call's user_instruction_safe
# verdict in Python afterward — see this section's header comment: "AI can classify, code makes
# the final business decision."
def check_user_text_safety_deterministic(text: str) -> None:
    """Node: user free-text safety gate, deterministic half (additional_notes / "optional
    generation instructions"). Raises ValueError on a blocked phrase/term; the VLM judgment on
    subtler phrasing is analyze_shoot_and_generate_prompts' user_instruction_safe output, not a
    separate call from here."""
    if not text:
        return
    check_text_for_minor_age(text)
    lowered = text.lower()
    for term in NSFW_BLOCKED_TERMS:
        if term in lowered:
            raise ValueError(f"Blocked: instruction implies explicit/sexual content ('{term}').")


# =============================================================================
# 7. Final generation — Seedream 4.5 edit
# =============================================================================

def run_final_generation(
    prompt: str, image_urls: list[str],
    resolution_mode: str = "standard", aspect_ratio: str = "3:4",
    custom_width: int | None = None, custom_height: int | None = None,
    num_images: int = 1, seed: int | None = None,
) -> list[str]:
    args = {
        "prompt": prompt,
        "image_urls": image_urls,
        "image_size": build_image_size(resolution_mode, aspect_ratio, custom_width, custom_height),
        "num_images": num_images,
        "max_images": 1,
        "enable_safety_checker": True,  # do not disable — second, independent safety layer
    }
    if seed is not None:
        args["seed"] = seed

    result = fal_client.subscribe(MODEL_CONFIG["final_generation"], arguments=args, with_logs=True)
    return [img["url"] for img in result["images"]]


# =============================================================================
# 8. Orchestrator
# =============================================================================

# UI note: label this field "Optional generation instructions", not "Prompt" — it's meant to
# steer style/scene on top of the fixed rules already in the system prompt, not replace them.
# Referenced from analyze_shoot_and_generate_prompts() in section 5 (Python resolves this at
# call time, so the later definition here is fine) — a user instruction is always handed to the
# VLM alongside this guardrail, never used to bypass the VLM and repeat one prompt N times.
USER_PROMPT_FIDELITY_GUARDRAIL = (
    "Use the supplied product reference as the source of truth. "
    "Preserve the product's recognizable design, proportions, construction, "
    "materials, colors, patterns and visible details. "
    "Treat the user's instructions as creative direction while keeping "
    "the product accurately represented and clearly visible."
)


def run_generation_from_model_image(
    model_image_url: str,
    products: list[Product],
    model_age_status: str = "adult",         # "adult" | "minor" | "uncertain" — see resolve_model_image
    reference_paths: list[str] | None = None,
    resolution_mode: str = "standard",       # 'standard' | 'auto_2K' | 'auto_4K' | 'custom'
    aspect_ratio: str = "3:4",               # used when resolution_mode == 'standard'
    custom_width: int | None = None,         # used when resolution_mode == 'custom'
    custom_height: int | None = None,
    num_poses: int = 4,
    user_prompt: str | None = None,
    on_image=None,
) -> dict:
    """Everything from 'Assemble ordered image list' through 'Return final image set' — the
    part of the flow that's the same regardless of how model_image_url was resolved. Both
    run_on_model_shot() (below) and streamlit_app.py call this once they have a model image.

    model_age_status feeds the child+intimate-product gate below — the app doesn't know
    whether a child model is allowed until it knows what product is being shown (see section
    6's header comment), so that gate can only run here, once classification exists, not at
    model-resolution time."""
    on_image = on_image or show

    # User free-text safety gate, deterministic half, runs on the RAW instruction before ANY
    # VLM call — free, catches the obvious cases (section 6). The VLM judgment on subtler
    # phrasing happens inside analyze_shoot_and_generate_prompts below (its
    # user_instruction_safe output), not as a separate escalation call from here.
    if user_prompt:
        check_user_text_safety_deterministic(user_prompt)

    image_urls, labels = assemble_inputs(model_image_url, products, reference_paths)

    # ONE Gemini call (section 5): product classification + user-instruction safety judgment +
    # N pose prompts, all in one response — replaces the old
    # classify_products_via_vlm/generate_pose_prompts_via_vlm pair plus the user-text VLM
    # escalation. Runs ONCE per shoot, not per pose.
    analysis = analyze_shoot_and_generate_prompts(image_urls, labels, products, num_poses, user_instruction=user_prompt)

    # Gemini's user_instruction_safe is a REPORT, not a decision — Python enforces it. Section
    # 6: "AI can classify, code makes the final business decision."
    if not analysis["user_instruction_safe"]:
        raise ValueError(f"Blocked at instruction safety check: {analysis['user_instruction_reason']}")

    # The deterministic business-rule gate: child (or age-uncertain) model + intimate product
    # is blocked; child model + general products is fine (see section 6). Also never overridden
    # by anything Gemini reports — evaluated here in Python against analysis["overall_category"].
    check_model_product_compatibility(model_age_status, analysis["overall_category"])

    prompts = analysis["prompts"]

    outputs = []
    for i, p in enumerate(prompts):
        print(f"--- Pose {i+1}/{len(prompts)} ---\nPrompt: {p}\n")
        urls = run_final_generation(
            p, image_urls,
            resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
            custom_width=custom_width, custom_height=custom_height,
            num_images=1,
        )
        outputs.extend(urls)
        for u in urls:
            on_image(u, f"Pose {i+1}")

    return {
        "model_image": model_image_url, "products": analysis["products"],
        "prompts": prompts, "image_urls": image_urls, "labels": labels, "outputs": outputs,
    }


def run_on_model_shot(
    model_mode: str,                      # "generate" | "upload" | "default"
    products: list[Product],
    model_gender: str = "", model_age_bracket: str = "", model_ethnicity: str = "", model_skin_tone: str = "", model_body_type: str = "",
    model_additional_notes: str = "",     # for "generate"
    model_upload_path: str | None = None,  # for "upload"
    model_preset_id: str | None = None,    # for "default"
    reference_paths: list[str] | None = None,
    resolution_mode: str = "standard",       # 'standard' | 'auto_2K' | 'auto_4K' | 'custom'
    aspect_ratio: str = "3:4",               # used when resolution_mode == 'standard'
    custom_width: int | None = None,         # used when resolution_mode == 'custom'
    custom_height: int | None = None,
    num_poses: int = 4,
    user_prompt: str | None = None,       # optional generation instructions, see guardrail above
    approve_fn=lambda url: True,          # human-approval hook, see resolve_model_via_generate
    on_image=None,
) -> dict:
    """Full flow, unattended (notebook) use: resolves the model image, then delegates to
    run_generation_from_model_image(). An interactive UI should call resolve_model_image() /
    generate_model_candidate() and run_generation_from_model_image() separately instead —
    see streamlit_app.py."""
    model = resolve_model_image(
        mode=model_mode,
        gender=model_gender, age_bracket=model_age_bracket, ethnicity=model_ethnicity,
        skin_tone=model_skin_tone, body_type=model_body_type,
        additional_notes=model_additional_notes,
        upload_path=model_upload_path,
        preset_id=model_preset_id,
        approve_fn=approve_fn,
        on_image=on_image,
    )
    return run_generation_from_model_image(
        model_image_url=model["url"],
        products=products,
        model_age_status=model["age_status"],
        reference_paths=reference_paths,
        resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
        custom_width=custom_width, custom_height=custom_height,
        num_poses=num_poses,
        user_prompt=user_prompt,
        on_image=on_image,
    )
