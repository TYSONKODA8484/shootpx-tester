"""ShootPX — Catalog Photoshoot: interactive test app.

Run with: streamlit run catalog_streamlit_app.py

Different tool from streamlit_app.py (On-Model Shots) — no model-selection step, just
product photos in, N catalog shots out. Standalone (imports catalog_flow.py, not flow.py).
"""

import tempfile
from pathlib import Path

import streamlit as st

import catalog_flow as flow

st.set_page_config(page_title="ShootPX — Catalog Photoshoot Tester", layout="wide")
st.title("ShootPX — Catalog Photoshoot Tester")
st.caption(
    "Every model comes from `.env` via `catalog_flow.MODEL_CONFIG` — nothing hardcoded here. "
    "This talks to real fal.ai APIs and spends real credits."
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
# Enforce the 10-image cap at the UI, not just via catalog_flow's backend ValueError — don't
# rely on the user finding out only after clicking Generate.
if product_files and len(product_files) > 10:
    st.error(f"{len(product_files)} images selected — max 10. Remove {len(product_files) - 10} before continuing.")
    product_files = product_files[:0]  # block Generate below rather than silently using the first 10

st.divider()

# =============================================================================
# 2. Output settings
# =============================================================================
st.header("2. Output settings")
num_outputs = st.slider("Number of outputs", 1, 10, 6)

# Primary choices are the simple, exact ones (a named preset or "let the model decide") —
# "Custom size" is demoted behind a checkbox since fal auto-scales an out-of-range custom
# size rather than honoring it exactly; MVP shouldn't promise a precision it can't guarantee.
RESOLUTION_LABELS = {
    "Aspect ratio preset": "standard",
    "Auto 2K": "auto_2K",
    "Auto 3K": "auto_3K",
    "Auto 4K": "auto_4K",
}
resolution_label = st.radio("Resolution", list(RESOLUTION_LABELS.keys()), horizontal=True)
resolution_mode = RESOLUTION_LABELS[resolution_label]

aspect_ratio, custom_width, custom_height = "1:1", None, None
if resolution_mode == "standard":
    aspect_ratio = st.selectbox("Aspect ratio", list(flow.ASPECT_RATIO_MAP.keys()), index=0)
else:
    st.caption("Model decides the aspect ratio from the inputs — no further size input needed.")

with st.expander("Advanced: custom pixel size (not guaranteed exact)"):
    use_custom_size = st.checkbox("Override with a custom width/height instead of the resolution above")
    if use_custom_size:
        resolution_mode = "custom"
        c1, c2 = st.columns(2)
        custom_width = c1.number_input("Width (px)", min_value=512, max_value=4096, value=2048, step=64)
        custom_height = c2.number_input("Height (px)", min_value=512, max_value=4096, value=2048, step=64)
        total_px = custom_width * custom_height
        if not (flow.CATALOG_MIN_TOTAL_PX <= total_px <= flow.CATALOG_MAX_TOTAL_PX):
            st.info(
                f"{custom_width}x{custom_height} ({total_px:,}px) is outside "
                f"{flow.CATALOG_MIN_TOTAL_PX:,}-{flow.CATALOG_MAX_TOTAL_PX:,} — fal will "
                f"auto-scale it to fit rather than reject it, so the exact size won't be honored."
            )

st.divider()

# =============================================================================
# 3. Advanced (optional)
# =============================================================================
st.header("3. Advanced (optional)")
use_advanced = st.checkbox("Use advanced prompt")
user_prompt = None
if use_advanced:
    user_prompt = st.text_area(
        "Additional instructions",
        placeholder="Style/scene guidance for the shot list — still routed through the "
        "LLM+Vision step alongside product fidelity and the adult-only rule, so this steers "
        "shot planning rather than replacing it.",
    )

st.divider()

# =============================================================================
# 4. Run
# =============================================================================
can_run = bool(product_files)
if not product_files:
    st.warning("Upload at least one product photo.")

if "catalog_results" not in st.session_state:
    st.session_state["catalog_results"] = []  # list of {"prompt", "url"}, one per completed shot


def render_saved_results():
    """Redraws the results grid from session_state — used when THIS run didn't just generate
    anything (e.g. a download-button click reran the whole script). Every element here is
    created exactly once per run, so keys never collide. Not used during an active generation
    run — that path (below) writes each shot into its own placeholder instead, since
    redrawing the full history on every new shot would re-register the same keys twice in
    one run, which is the bug that crashed this earlier."""
    results = st.session_state["catalog_results"]
    if results:
        st.divider()
        st.header("5. Results")
        cols = st.columns(min(len(results), 3))
        for i, item in enumerate(results):
            with cols[i % len(cols)]:
                st.image(item["url"], caption=f"Shot {i+1}")
                st.caption(item["prompt"])
                st.download_button(
                    "Download image", data=flow.fetch_image_bytes(item["url"]),
                    file_name=f"shot_{i+1}.png", mime="image/png", key=f"dl_shot_{i}",
                )


if st.button("Generate", type="primary", disabled=not can_run):
    product_paths = [save_upload(f) for f in product_files]
    st.session_state["catalog_results"] = []
    try:
        with st.spinner("Running the safety check..."):
            image_urls, labels = flow.assemble_product_images(product_paths)
            passed, reason = flow.run_safety_check(image_urls, user_prompt)
        if not passed:
            st.error(f"Blocked at safety check: {reason}")
        else:
            # One LLM+VLM call plans all N prompts up front (cheap, one call regardless of
            # N) — generation itself is what's sequential from here.
            with st.spinner("LLM+Vision planning the shot list..."):
                prompts = flow.build_shot_prompts(image_urls, labels, num_outputs, user_prompt)

            st.divider()
            st.header("5. Results")
            cols = st.columns(min(len(prompts), 3))
            # One placeholder per shot, created up front — each is written into exactly once
            # (when that shot's generation call returns), never re-touched afterward, so no
            # key is ever registered twice in this run.
            slots = [cols[i % len(cols)].empty() for i in range(len(prompts))]
            status = st.empty()

            for i, p in enumerate(prompts):
                status.info(f"Generating shot {i+1}/{len(prompts)} — not waiting for the rest to finish...")
                urls = flow.run_catalog_generation(
                    p, image_urls,
                    resolution_mode=resolution_mode, aspect_ratio=aspect_ratio,
                    custom_width=custom_width, custom_height=custom_height,
                    num_images=1,
                )
                url = urls[0]  # num_images=1 per shot, always exactly one
                st.session_state["catalog_results"].append({"prompt": p, "url": url})
                with slots[i].container():
                    st.image(url, caption=f"Shot {i+1}")
                    st.caption(p)
                    st.download_button(
                        "Download image", data=flow.fetch_image_bytes(url),
                        file_name=f"shot_{i+1}.png", mime="image/png", key=f"dl_shot_{i}",
                    )

            status.empty()
            st.success(f"Done — {len(st.session_state['catalog_results'])} image(s) generated.")
    except ValueError as e:
        st.error(str(e))
else:
    render_saved_results()  # no button click this run — redraw whatever's already in state
