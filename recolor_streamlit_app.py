"""ShootPX — Recolor: interactive test app.

Run with: streamlit run recolor_streamlit_app.py

Fourth tool, alongside streamlit_app.py, catalog_streamlit_app.py, creative_streamlit_app.py.
Simplest of the four — no idea/prompt case-branching, no LLM step: pick a part, pick a
color, done. Every path is exactly one image in, one image out.
"""

import tempfile
from pathlib import Path

import streamlit as st

import recolor_flow as flow

st.set_page_config(page_title="ShootPX — Recolor Tester", layout="wide")
st.title("ShootPX — Recolor Tester")
st.caption(
    "Every model comes from `.env` via `recolor_flow.MODEL_CONFIG`. No LLM/VLM step in this "
    "tool — just a templated prompt. Real fal.ai calls, real credits."
)


def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# =============================================================================
# 1. Image(s)
# =============================================================================
st.header("1. Image")
st.caption("First image is the photo being edited. Up to 3 more are optional references (e.g. a color swatch).")
image_files = st.file_uploader(
    f"Photo to recolor, plus up to {flow.RECOLOR_MAX_IMAGES - 1} optional reference images — max {flow.RECOLOR_MAX_IMAGES} total",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)
if image_files and len(image_files) > flow.RECOLOR_MAX_IMAGES:
    st.error(f"{len(image_files)} images selected — max {flow.RECOLOR_MAX_IMAGES}. Remove {len(image_files) - flow.RECOLOR_MAX_IMAGES} before continuing.")
    image_files = image_files[:0]
if image_files:
    st.image(image_files[0], caption="Image 1 — the photo being edited", width=240)

st.divider()

# =============================================================================
# 2. What to recolor
# =============================================================================
st.header("2. What to recolor")
color = st.color_picker("Color", value="#FF0000")
description = st.text_area(
    "Which part should be recolored? (optional)",
    placeholder="e.g. 'the shirt', or 'just the sleeves of the jacket'. Leave blank and the "
    "model finds the main recolorable subject itself.",
)

st.divider()

# =============================================================================
# 3. Output settings
# =============================================================================
st.header("3. Output settings")
st.caption("Optional — leave on 'Match input image' and this is nothing you need to touch.")
RESOLUTION_LABELS = {
    "Match input image": "match_input",
    "Aspect ratio preset": "standard",
    "Custom size": "custom",
}
resolution_label = st.radio("Resolution", list(RESOLUTION_LABELS.keys()), horizontal=True)
resolution_mode = RESOLUTION_LABELS[resolution_label]

aspect_ratio, custom_width, custom_height = "1:1", None, None
if resolution_mode == "match_input":
    st.caption("Uses the uploaded photo's own proportions — nothing else to set.")
elif resolution_mode == "standard":
    aspect_ratio = st.selectbox("Aspect ratio", list(flow.ASPECT_RATIO_MAP.keys()), index=0)
else:
    c1, c2 = st.columns(2)
    custom_width = c1.number_input("Width (px)", min_value=flow.RECOLOR_MIN_DIM, max_value=flow.RECOLOR_MAX_DIM, value=1024, step=64)
    custom_height = c2.number_input("Height (px)", min_value=flow.RECOLOR_MIN_DIM, max_value=flow.RECOLOR_MAX_DIM, value=1024, step=64)
    st.caption(f"Must be {flow.RECOLOR_MIN_DIM}-{flow.RECOLOR_MAX_DIM}px on each side for this model — hard limit, not auto-scaled.")

st.divider()

# =============================================================================
# 4. Run
# =============================================================================
can_run = bool(image_files)
if not image_files:
    st.warning("Upload the photo to recolor.")

if st.button("Generate 1 image", type="primary", disabled=not can_run):
    image_paths = [save_upload(f) for f in image_files]
    try:
        with st.spinner("Running the full flow..."):
            result = flow.run_recolor(
                image_paths=image_paths,
                description=description.strip(),
                color=color,
                resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
                custom_width=custom_width, custom_height=custom_height,
            )
        st.session_state["recolor_result"] = result
        st.success("Done.")
    except ValueError as e:
        st.error(str(e))

result = st.session_state.get("recolor_result")
if result:
    st.divider()
    st.header("5. Result")
    st.image(result["output"], width=480)
    if not description.strip():
        st.caption(f"You left the target blank — the model picked: **{result['description']}**")
    with st.expander("Full prompt sent to the model"):
        st.code(result["prompt"])
    st.download_button(
        "Download image", data=flow.fetch_image_bytes(result["output"]),
        file_name="recolor_output.png", mime="image/png", key="dl_recolor",
    )
