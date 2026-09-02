"""ShootPX — On-Model Shots: interactive test app.

Run with: streamlit run streamlit_app.py

Exercises the exact flow in flow.py against real fal.ai calls. No app-repo code touched —
this is a standalone tester in the Tool Testing sandbox.
"""

import tempfile
from pathlib import Path

import streamlit as st

import flow

st.set_page_config(page_title="ShootPX — On-Model Shot Tester", layout="wide")
st.title("ShootPX — On-Model Shot Tester")
st.caption(
    "Every model comes from `.env` via `flow.MODEL_CONFIG` — nothing hardcoded here. "
    "This talks to real fal.ai APIs and spends real credits."
)


def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# =============================================================================
# 1. Model source
# =============================================================================
st.header("1. Model source")
mode = st.radio("Source", ["generate", "upload", "default"], horizontal=True, key="mode")

if mode == "generate":
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", flow.GENDER_OPTIONS, horizontal=True)
        age_bracket = st.radio("Age", flow.AGE_BRACKET_OPTIONS)
    with col2:
        skin_tone = st.selectbox("Skin tone", flow.SKIN_TONE_OPTIONS)
        body_type = st.selectbox("Body type", flow.BODY_TYPE_OPTIONS)

    additional_notes = st.text_area(
        "Additional notes (optional)",
        placeholder="e.g. warm smile, shoulder-length brown hair — the LLM writes the actual "
        "text-to-image prompt from your selections + this, you don't write the prompt.",
    )

    if st.button("Generate model preview"):
        with st.spinner("Writing prompt + generating..."):
            st.session_state["candidate"] = flow.generate_model_candidate(
                gender, age_bracket, skin_tone, body_type, additional_notes
            )

    candidate = st.session_state.get("candidate")
    if candidate:
        st.caption(f"LLM-written prompt: _{candidate['description']}_")
        # Always show the image, flagged or not — an auto-check that hides what it flagged
        # gives you no way to catch its own false positives. Look at it, then decide.
        st.image(candidate["url"], width=320)
        st.download_button(
            "Download preview", data=flow.fetch_image_bytes(candidate["url"]),
            file_name="model_preview.png", mime="image/png", key="dl_model_preview",
        )
        if not candidate["clean"]:
            st.warning(
                f"Safety-check VLM flagged this: {candidate['reason']}. If that's a clear "
                f"false positive on the image above, Approve anyway — otherwise Regenerate."
            )
        c1, c2 = st.columns(2)
        if c1.button("Approve"):
            st.session_state["model_image_url"] = candidate["url"]
            st.success("Model approved — proceed to garment below.")
        if c2.button("Regenerate"):
            with st.spinner("Regenerating..."):
                st.session_state["candidate"] = flow.generate_model_candidate(
                    gender, age_bracket, skin_tone, body_type, additional_notes
                )
            st.rerun()

elif mode == "upload":
    up = st.file_uploader("Upload model photo", type=["jpg", "jpeg", "png"])
    if up is not None and st.button("Use this photo"):
        try:
            with st.spinner("Uploading + safety-checking..."):
                st.session_state["model_image_url"] = flow.resolve_model_via_upload(save_upload(up))
            st.success("Uploaded and passed the safety check.")
        except ValueError as e:
            st.error(str(e))

else:  # default
    if flow.MODEL_LIBRARY:
        preset_id = st.selectbox("Preset model", list(flow.MODEL_LIBRARY.keys()))
        if st.button("Use preset"):
            st.session_state["model_image_url"] = flow.resolve_model_via_default(preset_id)
            st.success("Preset selected.")
    else:
        st.info("MODEL_LIBRARY in flow.py is empty — add preset id -> path/URL entries to use this mode.")

model_image_url = st.session_state.get("model_image_url")
if model_image_url:
    st.success(f"Current model image: {model_image_url}")

st.divider()

# =============================================================================
# 2. Products
# =============================================================================
# No "what is this" text field — flow.classify_products_via_vlm looks at the actual images
# and works out type/category/body-placement itself. The user only uploads and (in Multiple
# Garments mode, for anything past Top/Bottom) optionally labels for their own bookkeeping.
st.header("2. Products")

product_mode = st.radio("Mode", ["Single Garment", "Multiple Garments"], horizontal=True, key="product_mode")

products: list[flow.Product] = []
product_image_count = 0
existing_ids: set[str] = set()


def _add_product(label: str, files) -> None:
    if not files:
        return
    paths = [save_upload(f) for f in files]
    product = flow.make_product(label, paths, existing_ids)
    existing_ids.add(product["id"])
    products.append(product)
    global product_image_count
    product_image_count += len(paths)


if product_mode == "Single Garment":
    single_files = st.file_uploader(
        "Upload garment images — front, back, side, detail, as many angles as you have",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="single_product_files",
    )
    _add_product("Garment", single_files)

else:  # Multiple Garments
    col1, col2 = st.columns(2)
    with col1:
        top_files = st.file_uploader("Top", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="top_files")
    with col2:
        bottom_files = st.file_uploader("Bottom", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bottom_files")
    _add_product("Top", top_files)
    _add_product("Bottom", bottom_files)

    # Unlimited extra product slots (watch, hat, shoes, bag, ...) — each tracked by a stable
    # id in session_state so adding/removing rows doesn't collide with Streamlit's widget-key
    # reuse across reruns.
    st.session_state.setdefault("extra_slot_ids", [])
    st.session_state.setdefault("next_slot_id", 1)

    for slot_id in list(st.session_state["extra_slot_ids"]):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            label = c1.text_input(
                "Garment / product name", key=f"slot_{slot_id}_label",
                placeholder="e.g. Watch, Hat, Shoes, Bag",
            )
            c2.write("")  # vertical spacer so the button lines up with the text input
            if c2.button("Remove", key=f"slot_{slot_id}_remove"):
                st.session_state["extra_slot_ids"].remove(slot_id)
                st.rerun()
            slot_files = st.file_uploader(
                "Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"slot_{slot_id}_files",
            )
            _add_product(label, slot_files)

    images_so_far = 1 + product_image_count  # +1 for the model image
    if st.button("+ Add Garment", disabled=images_so_far >= 10):
        st.session_state["extra_slot_ids"].append(st.session_state["next_slot_id"])
        st.session_state["next_slot_id"] += 1
        st.rerun()

reference_files = st.file_uploader(
    "Reference images (optional — style/pose only, not product identity)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)

total_images = 1 + product_image_count + len(reference_files or [])  # +1 for the model image
st.caption(f"{total_images} / 10 images" + (" — over the limit, remove some before generating" if total_images > 10 else ""))

st.divider()

# =============================================================================
# 3. Output settings
# =============================================================================
st.header("3. Output settings")

RESOLUTION_LABELS = {
    "Aspect ratio preset": "standard",
    "Auto 2K (model picks aspect ratio)": "auto_2K",
    "Auto 4K (model picks aspect ratio)": "auto_4K",
    "Custom size": "custom",
}
resolution_label = st.radio("Resolution", list(RESOLUTION_LABELS.keys()), horizontal=True)
resolution_mode = RESOLUTION_LABELS[resolution_label]

aspect_ratio, custom_width, custom_height = "3:4", None, None
if resolution_mode == "standard":
    aspect_ratio = st.selectbox("Aspect ratio", list(flow.ASPECT_RATIO_MAP.keys()), index=1)
elif resolution_mode == "custom":
    c1, c2 = st.columns(2)
    custom_width = c1.number_input("Width (px)", min_value=512, max_value=4096, value=2048, step=64)
    custom_height = c2.number_input("Height (px)", min_value=512, max_value=4096, value=2048, step=64)
    try:
        flow.validate_custom_size(custom_width, custom_height)
    except ValueError as e:
        st.warning(str(e))
else:
    st.caption("Model decides the aspect ratio from the inputs — no further size input needed.")

num_poses = st.slider("Number of poses", 1, 4, 4)

user_prompt = st.text_area(
    "Optional generation instructions",
    placeholder="Style/scene guidance only — always routed through the VLM alongside the "
    "fixed fidelity/framing rules (see USER_PROMPT_FIDELITY_GUARDRAIL in flow.py), so you "
    "still get real pose variety instead of one prompt repeated. Leave blank to let the "
    "router + VLM decide everything.",
)

st.divider()

# =============================================================================
# 4. Run
# =============================================================================
can_run = bool(model_image_url) and bool(products) and total_images <= 10
if not model_image_url:
    st.warning("Resolve a model image in step 1 first.")
if not products:
    st.warning("Upload at least one product photo in step 2.")

if st.button("Generate on-model shots", type="primary", disabled=not can_run):
    reference_paths = [save_upload(f) for f in (reference_files or [])]
    try:
        with st.spinner("Running the full flow — this makes real fal.ai calls..."):
            result = flow.run_generation_from_model_image(
                model_image_url=model_image_url,
                products=products,
                reference_paths=reference_paths,
                resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
                custom_width=custom_width, custom_height=custom_height,
                num_poses=num_poses,
                user_prompt=user_prompt or None,
            )
        st.session_state["last_result"] = result
        st.success(f"Done — {len(result['outputs'])} image(s) generated.")
    except ValueError as e:
        st.error(str(e))

# Rendered from session_state (not inline in the button block above) so download-button
# clicks — which trigger a Streamlit rerun — don't lose the results and force a re-run of
# the whole (paid) generation call.
last_result = st.session_state.get("last_result")
if last_result:
    st.divider()
    st.header("4. Results")
    # fal output URLs are not permanent (see download_image()'s docstring in flow.py) —
    # download whatever you want to keep before closing this session.
    cols = st.columns(len(last_result["outputs"]) or 1)
    for i, (p, url) in enumerate(zip(last_result["prompts"], last_result["outputs"])):
        with cols[i % len(cols)]:
            st.image(url, caption=f"Pose {i+1}")
            st.caption(p)
            st.download_button(
                "Download image",
                data=flow.fetch_image_bytes(url),
                file_name=f"pose_{i+1}.png",
                mime="image/png",
                key=f"dl_pose_{i}",
            )
