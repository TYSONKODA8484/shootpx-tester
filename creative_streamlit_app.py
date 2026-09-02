"""ShootPX — Creative Photoshoot: interactive test app.

Run with: streamlit run creative_streamlit_app.py

Third tool, alongside streamlit_app.py (On-Model Shots) and catalog_streamlit_app.py
(Catalog Photoshoot). Always exactly one output image — no "number of outputs" control
anywhere in this UI, by design. Want another variation? Click Generate again.
"""

import tempfile
from pathlib import Path

import streamlit as st

import creative_flow as flow

st.set_page_config(page_title="ShootPX — Creative Photoshoot Tester", layout="wide")
st.title("ShootPX — Creative Photoshoot Tester")
st.caption(
    "One Generate click = one image, always. A second variation is a second click "
    "(full flow re-run, safety check included), not a batch setting. Real fal.ai calls, real credits."
)


def save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# =============================================================================
# 1. Product images
# =============================================================================
st.header("1. Product images")
product_files = st.file_uploader(
    "Product photos (multiple angles/details of the same product allowed) — max 10",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)
if product_files and len(product_files) > 10:
    st.error(f"{len(product_files)} images selected — max 10. Remove {len(product_files) - 10} before continuing.")
    product_files = product_files[:0]

st.divider()

# =============================================================================
# 2. Settings — aspect ratio + resolution only. No output-count control on purpose.
# =============================================================================
st.header("2. Settings")
RESOLUTION_LABELS = {
    "Aspect ratio preset": "standard",
    "Auto 2K": "auto_2K",
    "Auto 3K": "auto_3K",
    "Auto 4K": "auto_4K",
}
resolution_label = st.radio("Resolution", list(RESOLUTION_LABELS.keys()), horizontal=True)
resolution_mode = RESOLUTION_LABELS[resolution_label]

aspect_ratio, custom_width, custom_height = "3:4", None, None
if resolution_mode == "standard":
    aspect_ratio = st.selectbox("Aspect ratio", list(flow.ASPECT_RATIO_MAP.keys()), index=1)
else:
    st.caption("Model decides the aspect ratio from the inputs — no further size input needed.")

with st.expander("Advanced: custom pixel size (bytedance/seedream/v5/lite/edit — total pixels 2560x1440 to 4096x4096; outside that range gets auto-scaled to fit, not rejected)"):
    use_custom_size = st.checkbox("Override with a custom width/height instead of the resolution above")
    if use_custom_size:
        resolution_mode = "custom"
        c1, c2 = st.columns(2)
        custom_width = c1.number_input("Width (px)", min_value=512, max_value=4096, value=2048, step=64)
        custom_height = c2.number_input("Height (px)", min_value=512, max_value=4096, value=2048, step=64)

st.divider()

# =============================================================================
# 3. Idea and/or Prompt — at least one required
# =============================================================================
st.header("3. Idea and/or Prompt")
idea_tags = st.multiselect("Idea", flow.IDEA_OPTIONS)
user_prompt = st.text_input(
    "Prompt (optional if an Idea is selected, required if not)",
    placeholder="e.g. 'on a marble kitchen counter at golden hour' — merged with the Idea if both are set.",
)

if idea_tags and user_prompt:
    st.caption("Case 3: idea + prompt — one LLM call merges both into a single refined prompt.")
elif idea_tags:
    st.caption("Case 2: idea only — one LLM call writes the scene from scratch.")
elif user_prompt:
    st.caption("Case 1: prompt only — used verbatim, no LLM call.")

st.divider()

# =============================================================================
# 4. Run
# =============================================================================
can_run = bool(product_files) and (bool(idea_tags) or bool(user_prompt))
if not product_files:
    st.warning("Upload at least one product photo.")
elif not idea_tags and not user_prompt:
    st.warning("Select an Idea, enter a Prompt, or both.")

if st.button("Generate", type="primary", disabled=not can_run):
    product_paths = [save_upload(f) for f in product_files]
    try:
        with st.spinner("Running the full flow — safety check, prompt, generation..."):
            result = flow.run_creative_photoshoot(
                product_paths=product_paths,
                idea_tags=idea_tags or None,
                user_prompt=user_prompt or None,
                resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
                custom_width=custom_width, custom_height=custom_height,
            )
        st.session_state["creative_result"] = result
        st.success("Done.")
    except ValueError as e:
        st.error(str(e))

result = st.session_state.get("creative_result")
if result:
    st.divider()
    st.header("5. Result")
    st.image(result["output"], width=480)
    st.caption(f"Final prompt: _{result['final_prompt']}_")
    st.download_button(
        "Download image", data=flow.fetch_image_bytes(result["output"]),
        file_name="creative_output.png", mime="image/png", key="dl_creative",
    )
