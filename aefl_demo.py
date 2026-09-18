"""
aefl_demo.py — Gradio live demo for judges.

Run with:  python aefl_demo.py
Launches with share=True so you get a public URL to show the panel.

Loads whatever has been trained so far (prefers the AEFL model, falls back
to any other saved model) and whatever plots/reports exist in results/, so
it works even mid-training after a partial run.
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import evaluation, explainability

# --------------------------------------------------------------------------
# Fill these in before presenting to the judges
# --------------------------------------------------------------------------
COLLEGE_NAME = "Your College Name"
DEPARTMENT_NAME = "Department of Computer Science & Engineering"
GUIDE_NAME = "Guide: Dr. [Guide Name]"
TEAM_NAME = "Team: [Your Name(s)]"

RESULTS_DIR = "results"
MODELS_DIR = "saved_models"
IMAGE_SIZE = 128

MODEL_PREFERENCE = [
    "AEFL", "AEFL-AccFocus", "AEFL-CommFocus", "AEFL-Top2",
    "FedProx", "FedAvg", "Centralized",
]
CLASS_NAMES = ("Normal", "Pneumonia")


# --------------------------------------------------------------------------
# Load whatever is available
# --------------------------------------------------------------------------

def _find_available_model():
    import tensorflow as tf
    for name in MODEL_PREFERENCE:
        path = os.path.join(MODELS_DIR, f"{name}_model.keras")
        if os.path.exists(path):
            print(f"Loading model: {name}")
            return tf.keras.models.load_model(path), name
    print("⚠️ No saved model found in saved_models/. Live prediction tab will be disabled "
          "until aefl_main.py has been run.")
    return None, None


def _res(path):
    p = os.path.join(RESULTS_DIR, path)
    return p if os.path.exists(p) else None


def _load_csv(path):
    p = os.path.join(RESULTS_DIR, path)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def _load_text(path):
    p = os.path.join(RESULTS_DIR, path)
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    return "Not generated yet — run `python aefl_main.py` first."


def _find_sample_images(n=3):
    candidates = []
    for pattern in ("data/chest_xray/test/*/*.jpeg", "data/chest_xray/test/*/*.png",
                     "data/**/test/*/*.jpeg"):
        candidates.extend(glob.glob(pattern, recursive=True))
    return candidates[:n]


MODEL, MODEL_NAME = _find_available_model()
SAMPLE_IMAGES = _find_sample_images()


# --------------------------------------------------------------------------
# Tab 1 — Live prediction
# --------------------------------------------------------------------------

def predict_xray(image):
    if image is None:
        return "Please upload a chest X-ray image.", None, None, ""
    if MODEL is None:
        return "No trained model found. Run `python aefl_main.py` first.", None, None, ""

    from PIL import Image
    img = Image.fromarray(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype="float32") / 255.0

    prob = float(MODEL.predict(arr[np.newaxis, ...], verbose=0)[0, 0])
    pred_class = int(prob >= 0.5)
    confidence = prob if pred_class == 1 else 1 - prob
    label = CLASS_NAMES[pred_class]

    saliency = explainability.generate_saliency(MODEL, arr)
    region = explainability.localize_region(saliency)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr)
    ax.imshow(saliency, cmap="jet", alpha=0.5)
    ax.set_title(f"Saliency overlay — focus: {region}")
    ax.axis("off")
    fig.tight_layout()
    overlay_path = os.path.join(RESULTS_DIR, "_live_overlay.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(overlay_path, dpi=120)
    plt.close(fig)

    result_md = (
        f"### Prediction: **{label}**\n"
        f"**Confidence: {confidence * 100:.1f}%**\n\n"
        f"Model used: `{MODEL_NAME}` (Proposed AEFL framework)\n\n"
        f"Highlighted focus region: **{region}**\n\n"
        f"> ⚕️ *Clinical note: AI suggests **{label}**. "
        f"Please consult a radiologist for a confirmed diagnosis. "
        f"This tool is for research/educational demonstration only.*"
    )
    prob_df = pd.DataFrame({
        "Class": list(CLASS_NAMES),
        "Probability": [1 - prob, prob],
    })
    return result_md, overlay_path, prob_df, label


# --------------------------------------------------------------------------
# Tab 2 — Performance table (AEFL row highlighted)
# --------------------------------------------------------------------------

def _performance_table_html():
    df = _load_csv("performance_comparison.csv")
    if df is None:
        return "<p>Not generated yet — run <code>python aefl_main.py</code> first.</p>"

    def highlight(row):
        if "AEFL" in row["Method"] and "Acc" not in row["Method"] and "Comm" not in row["Method"] \
                and "Top" not in row["Method"]:
            return ["background-color:#dcfce7;font-weight:600"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(highlight, axis=1).hide(axis="index")
    return styled.to_html()


# --------------------------------------------------------------------------
# Tab 3 — Communication summary text
# --------------------------------------------------------------------------

def _communication_summary_text():
    df = _load_csv("communication_summary.csv")
    if df is None:
        return "Not generated yet — run `python aefl_main.py` first."
    try:
        row = df.set_index("Method")
        fedavg_mb = float(row.loc["FedAvg", "Total_MB"])
        aefl_mb = float(row.loc["Proposed AEFL", "Total_MB"])
        savings = 100 * (1 - aefl_mb / fedavg_mb) if fedavg_mb else 0
        return (f"**AEFL saved {savings:.1f}% communication vs FedAvg** "
                f"({aefl_mb:.1f} MB vs {fedavg_mb:.1f} MB) by adaptively selecting only "
                f"the top-performing hospitals each round instead of contacting all of them.")
    except Exception:
        return "Communication summary available in the CSV, but AEFL/FedAvg rows weren't both found."


# --------------------------------------------------------------------------
# Tab 5 — SHAP gallery
# --------------------------------------------------------------------------

def _shap_gallery():
    shap_dir = os.path.join(RESULTS_DIR, "shap_explanations")
    files = sorted(glob.glob(os.path.join(shap_dir, "*.png")))
    items = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        caption = base.replace("shap_", "").replace("_", " ").title()
        items.append((f, caption))
    return items


# --------------------------------------------------------------------------
# Tab 7 — Architecture text
# --------------------------------------------------------------------------

ARCHITECTURE_MD = """
## AEFL — System Architecture

**Goal:** collaboratively train a chest-disease detector across multiple
hospitals **without ever moving patient images off-site**, while remaining
adaptive (choosing which hospitals to trust each round), explainable
(showing *why* a prediction was made), and communication-efficient.

### Pipeline
```
[Hospital 1]  [Hospital 2]  [Hospital 3]  [Hospital 4]  [Hospital 5]
     |             |             |             |             |
     +----- local training only — raw data never leaves site -----+
                              |
                 participation scoring + top-k selection
                              |
                    adaptive weighted aggregation
                              |
                        global model update
                              |
              SHAP / LIME / Saliency explainability layer
```

### CNN architecture (trained from scratch, 128×128×3 input)
```
Conv(32) → BN → MaxPool → Dropout(0.25)
Conv(64) → BN → MaxPool → Dropout(0.25)
Conv(128) → BN → MaxPool → Dropout(0.25)
Conv(256) → BN → GlobalAvgPool
Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)
```

### AEFL algorithm (pseudocode)
```
for round in 1..R:
    if round == 1:
        selected = all hospitals
    else:
        for each hospital i:
            S_i = alpha * A_i + beta * D_i + gamma * C_i
        selected = top_k(S, k=3)

    for hospital i in selected:
        w_i = local_train(global_weights, hospital_i.data, epochs=2)
        A_i = evaluate(w_i, hospital_i.local_val_split)     # stays local

    omega_i = S_i / sum(S_j for j in selected)
    global_weights = sum(omega_i * w_i for i in selected)
```

### Key formulas
- **Participation score:** `S_i = α·A_i + β·D_i + γ·C_i`
- **Data quality:** `D_i = 2 · min(n_pos/n_total, n_neg/n_total)`
- **Adaptive aggregation weight:** `ω_i = S_i / Σ_j S_j`  (over selected clients)
- **FedProx local objective:** `loss + (μ/2)·‖w − w_global‖²`, μ = 0.01

### Why this is privacy-preserving
Only model weight updates and a single locally-computed accuracy scalar
ever leave a hospital — never the underlying patient images — matching
HIPAA/GDPR-style data-locality requirements.
"""

TEST_TAB_HELP = (
    "Automated checks covering model output validity, adaptive-weight "
    "correctness, Non-IID split constraints, FedAvg aggregation math, "
    "explanation output shapes, communication-cost accounting, and "
    "model save/load consistency."
)


# --------------------------------------------------------------------------
# Build the Gradio app
# --------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(theme=gr.themes.Soft(), title="AEFL — Federated Healthcare AI") as demo:
        gr.Markdown(
            "# 🏥 AEFL — Adaptive Explainable Federated Learning Framework\n"
            "### Privacy-Preserving Multi-Hospital Chest X-ray Diagnosis\n"
            f"Currently serving model: **{MODEL_NAME or 'none — train first'}**"
        )

        with gr.Tabs():
            with gr.Tab("🔬 Live X-ray Analysis"):
                gr.Markdown("Upload a chest X-ray, or click one of the examples below.")
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(type="numpy", label="Chest X-ray")
                        btn = gr.Button("Analyze", variant="primary")
                        if SAMPLE_IMAGES:
                            gr.Examples(examples=SAMPLE_IMAGES, inputs=img_in)
                    with gr.Column():
                        result_md = gr.Markdown()
                        overlay_img = gr.Image(label="Saliency Explanation Overlay")
                        prob_table = gr.Dataframe(label="Class probabilities", interactive=False)
                        pred_state = gr.Textbox(visible=False)
                btn.click(predict_xray, inputs=img_in,
                          outputs=[result_md, overlay_img, prob_table, pred_state])

            with gr.Tab("📊 Performance Results"):
                gr.Markdown("## Test-set performance across all trained methods")
                perf_img = _res("performance_comparison.png")
                if perf_img:
                    gr.Image(perf_img, label="Accuracy / F1 / AUC comparison")
                gr.HTML(_performance_table_html())

            with gr.Tab("🔄 Convergence & Communication"):
                gr.Markdown("## Convergence speed and communication cost")
                conv_img = _res("convergence_plot.png")
                if conv_img:
                    gr.Image(conv_img, label="Validation accuracy vs. round")
                comm_img = _res("communication_cost.png")
                if comm_img:
                    gr.Image(comm_img, label="Communication cost / training time / clients per round")
                gr.Markdown(_communication_summary_text())

            with gr.Tab("🏥 Hospital Data Simulation"):
                gr.Markdown(
                    "## Non-IID multi-hospital simulation\n"
                    "Each simulated hospital holds a different amount of data with a "
                    "different Normal:Pneumonia ratio and a different communication "
                    "reliability score — mirroring how real hospitals never have "
                    "identically distributed patient populations."
                )
                hosp_img = _res("hospital_distribution.png")
                if hosp_img:
                    gr.Image(hosp_img, label="Class distribution per hospital")
                part_img = _res("participation_scores.png")
                if part_img:
                    gr.Image(part_img, label="AEFL participation scores over rounds")

            with gr.Tab("🔍 SHAP Explanations Gallery"):
                gr.Markdown("## Explainability gallery — Original | SHAP | LIME | Saliency | Overlay")
                gallery_items = _shap_gallery()
                if gallery_items:
                    gr.Gallery(value=gallery_items, columns=2, height="auto")
                else:
                    gr.Markdown("Not generated yet — run `python aefl_main.py` first.")

            with gr.Tab("✅ Test Results"):
                gr.Markdown("## Automated Testing & Validation\n" + TEST_TAB_HELP)
                gr.Code(_load_text("test_report.txt"), language=None)

            with gr.Tab("🏗️ Architecture"):
                gr.Markdown(ARCHITECTURE_MD)

        gr.Markdown(
            f"---\n**{COLLEGE_NAME}** · {DEPARTMENT_NAME}  \n{GUIDE_NAME} · {TEAM_NAME}"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
