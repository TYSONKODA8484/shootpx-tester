# Creative Photoshoot Tester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Streamlit tester that creates exactly one creative product image per Generate click.

**Architecture:** `creative_flow.py` coordinates existing Catalog helpers for product inputs and safety, uses one vision prompt-writer call only when idea tags exist, and delegates a one-image call to the existing On-Model Seedream 4.5 helper. `creative_streamlit_app.py` collects inputs, invokes the flow, and persists a single result through reruns.

**Tech Stack:** Python 3, Streamlit, unittest, fal-client, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-08-29-creative-photoshoot-design.md`

## Global Constraints

- Accept one through ten product images using `catalog_flow.assemble_product_images`.
- Run `catalog_flow.run_safety_check` before every vision or generation call.
- Prompt-only input bypasses the vision model and stays verbatim after whitespace validation.
- Every generation call uses the existing Seedream 4.5 helper with `num_images=1`.
- The UI contains no output-count control and displays one output only.

---

### Task 1: Test and implement the creative flow

**Files:**
- Create: `tests/test_creative_flow.py`
- Create: `creative_flow.py`

**Interfaces:**
- Consumes: `catalog_flow.assemble_product_images(paths) -> tuple[list[str], list[str]]`, `catalog_flow.run_safety_check(urls, prompt) -> tuple[bool, str]`, `flow.run_final_generation(prompt, urls, ..., num_images=1) -> list[str]`.
- Produces: `run_creative_photoshoot(...) -> dict` and `build_creative_prompt(...) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prompt_only_is_used_verbatim_without_a_vision_call(monkeypatch):
    monkeypatch.setattr(creative.catalog, "assemble_product_images", lambda _: (["product"], ["Image 1"]))
    monkeypatch.setattr(creative.catalog, "run_safety_check", lambda *_: (True, ""))
    monkeypatch.setattr(creative.on_model, "run_final_generation", lambda prompt, *_args, **kwargs: ["output"])
    result = creative.run_creative_photoshoot(["p.jpg"], user_prompt="  studio scene  ")
    assert result["final_prompt"] == "studio scene"
    assert result["output"] == "output"

def test_idea_only_calls_vision_once(monkeypatch):
    monkeypatch.setattr(creative.catalog, "assemble_product_images", lambda _: (["product"], ["Image 1"]))
    monkeypatch.setattr(creative.catalog, "run_safety_check", lambda *_: (True, ""))
    monkeypatch.setattr(creative, "_write_scene_prompt", lambda **_: "vision scene")
    monkeypatch.setattr(creative.on_model, "run_final_generation", lambda *_args, **_kwargs: ["output"])
    assert creative.run_creative_photoshoot(["p.jpg"], idea_tags=["Editorial"])["final_prompt"] == "vision scene"
```

- [ ] **Step 2: Run the tests and verify they fail because `creative_flow` does not exist**

Run: `& 'C:\Program Files\Comfy Desktop\resources\bootstrap-python\python.exe' -m unittest tests.test_creative_flow -v`

- [ ] **Step 3: Implement the minimum flow**

```python
def run_creative_photoshoot(...):
    image_urls, labels = catalog.assemble_product_images(product_paths)
    passed, reason = catalog.run_safety_check(image_urls, user_prompt or " ".join(idea_tags or []))
    if not passed:
        raise ValueError(f"Blocked at safety check: {reason}")
    final_prompt = build_creative_prompt(image_urls, labels, idea_tags, user_prompt)
    outputs = on_model.run_final_generation(final_prompt, image_urls, num_images=1, ...)
    if len(outputs) != 1:
        raise ValueError("Creative Photoshoot expected exactly one output image")
    return {"image_urls": image_urls, "labels": labels, "final_prompt": final_prompt, "output": outputs[0]}
```

- [ ] **Step 4: Run all flow tests**

Run: `& 'C:\Program Files\Comfy Desktop\resources\bootstrap-python\python.exe' -m unittest tests.test_creative_flow -v`

### Task 2: Create the Streamlit tester

**Files:**
- Create: `creative_streamlit_app.py`

**Interfaces:**
- Consumes: `creative_flow.run_creative_photoshoot(...) -> dict`.
- Produces: A Streamlit test application runnable with `py -3 -m streamlit run creative_streamlit_app.py`.

- [ ] **Step 1: Add a compile check before the file exists**

Run: `& 'C:\Program Files\Comfy Desktop\resources\bootstrap-python\python.exe' -m py_compile creative_streamlit_app.py`

- [ ] **Step 2: Implement the page**

```python
can_run = bool(product_files) and bool(idea_tags or user_prompt.strip())
if st.button("Generate", type="primary", disabled=not can_run):
    result = creative.run_creative_photoshoot(...)
    st.session_state["creative_result"] = result
```

- [ ] **Step 3: Verify both modules compile**

Run: `& 'C:\Program Files\Comfy Desktop\resources\bootstrap-python\python.exe' -m py_compile creative_flow.py creative_streamlit_app.py`

### Task 3: Verify the feature contract

**Files:**
- Verify: `tests/test_creative_flow.py`
- Verify: `creative_flow.py`
- Verify: `creative_streamlit_app.py`

- [ ] **Step 1: Run the feature tests**

Run: `& 'C:\Program Files\Comfy Desktop\resources\bootstrap-python\python.exe' -m unittest tests.test_creative_flow -v`

- [ ] **Step 2: Check the UI has no output-count control and the generation call is fixed to one image**

Run: `rg -n "num_outputs|Number of outputs|num_images=1" creative_flow.py creative_streamlit_app.py`

- [ ] **Step 3: Compile both modules**

Run: `py -3 -m py_compile creative_flow.py creative_streamlit_app.py`
