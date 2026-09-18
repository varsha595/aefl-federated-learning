# AEFL — Adaptive Explainable Federated Learning Framework for Privacy-Preserving Multi-Hospital Healthcare AI

AEFL simulates five hospitals collaboratively training a chest X-ray disease
classifier **without sharing patient data**. It is:

- **Federated** — raw images never leave a hospital; only model weights (and a
  single locally-computed accuracy scalar) are exchanged.
- **Adaptive** — each round, hospitals are scored on local accuracy, data
  quality, and communication reliability, and only the top-k are selected and
  weighted accordingly, instead of blindly averaging everyone every round.
- **Explainable** — every prediction ships with SHAP, LIME, and gradient
  saliency explanations, localized to a lung region.
- **Communication-efficient** — by training fewer, better-chosen clients per
  round, AEFL transmits meaningfully less data than standard FedAvg for
  comparable or better accuracy.

This solves a real deployment blocker: hospitals cannot pool patient data
under HIPAA/GDPR, but still need to collaborate to train AI that generalizes
across populations.

## Rubric mapping

| Rubric criterion | Where it's satisfied |
|---|---|
| Algorithm Design & Architecture | `modules/federated.py` (7 FL algorithms), `modules/model.py` (CNN + FedProx), participation-score formulas in `aefl_demo.py`'s Architecture tab |
| Working Prototype & Simulation | `aefl_main.py` end-to-end pipeline; `modules/data_loader.py` Non-IID hospital simulation |
| Result Analysis & Comparison | `modules/evaluation.py` — 9 plots, 2 CSVs, `summary.md`; ablation study (Algorithms 5–7) |
| Live Demo with real output | `aefl_demo.py` — 7-tab Gradio app with live inference, explanations, and every generated result |
| Testing & Validation | `modules/utils.py::run_tests()` — 7 automated checks, run before *and* after training, saved to `results/test_report.txt` |

## File structure

```
aefl-federated-learning/
├── aefl_main.py           # Master training script (runs everything)
├── aefl_demo.py            # Gradio web demo for judges
├── modules/
│   ├── data_loader.py      # Dataset loading + Non-IID hospital splitting
│   ├── model.py             # CNN architecture + FedProx proximal term
│   ├── federated.py         # Centralized / FedAvg / FedProx / AEFL + ablations
│   ├── explainability.py    # SHAP + LIME + saliency
│   ├── evaluation.py        # Metrics + every plot/table
│   └── utils.py             # Seeding, logging, checkpointing, test suite
├── results/                # Auto-created — all plots, CSVs, reports
├── saved_models/           # Auto-created — one .keras file per algorithm
└── data/                   # Kaggle dataset cache (or manual download target)
```

## Setup on Google Colab (T4 GPU)

1. **Runtime → Change runtime type → T4 GPU.**
2. Install dependencies:
   ```bash
   !pip install -q kagglehub shap lime gradio scikit-learn
   ```
   (TensorFlow, NumPy, pandas, matplotlib, Pillow are pre-installed on Colab.)
3. **Kaggle credentials** — upload your `kaggle.json` (from
   kaggle.com/settings) or set:
   ```python
   import os
   os.environ["KAGGLE_USERNAME"] = "..."
   os.environ["KAGGLE_KEY"] = "..."
   ```
   `kagglehub` will use these automatically. If a dataset can't be
   downloaded (no credentials, offline, quota), the pipeline logs a
   warning and continues with whatever loaded — the primary dataset
   (Chest X-ray Pneumonia) is required; the two secondary datasets are
   optional and only used for cross-dataset generalization checks.
4. **Get the project into Colab:**
   ```bash
   !git clone https://github.com/varsha595/aefl-federated-learning.git
   %cd aefl-federated-learning
   !python aefl_main.py
   ```
   This trains all 7 algorithms, checkpointing after each one — if Colab
   disconnects, just re-run `aefl_main.py` and already-finished algorithms
   are skipped automatically.
5. **Launch the demo:**
   ```bash
   !python aefl_demo.py
   ```
   This prints a public `*.gradio.live` URL — open it and present from there.

Everything runs at `image_size=128` and modest round/epoch counts
specifically so the full pipeline (train + explain + plot) finishes on a
free-tier T4 in well under three hours.

## File descriptions

- **`aefl_main.py`** — the `CONFIG` dict, dataset loading, hospital
  simulation, training loop over all 7 algorithms, explainability
  generation, plotting, and the final console report.
- **`aefl_demo.py`** — standalone Gradio app; loads whatever models/results
  already exist in `saved_models/` and `results/`, so it also works mid-way
  through a training run (it just shows less).
- **`modules/data_loader.py`** — `kagglehub`-based downloads with
  local-cache fallback; `create_hospital_splits()` implements the Non-IID
  simulation with a guaranteed minimum-10%-per-class constraint per
  hospital.
- **`modules/model.py`** — `create_model()` builds the 4-block CNN;
  `proximal_term()` implements the FedProx regularizer.
- **`modules/federated.py`** — `local_train()` is the shared manual
  training loop (GradientTape-based, so the proximal term can be injected);
  `run_centralized`, `run_fedavg`, `run_fedprox`, `run_aefl` implement each
  algorithm; `run_aefl` is reused for all three ablations via different
  `alpha/beta/gamma/top_k` arguments.
- **`modules/explainability.py`** — SHAP `GradientExplainer`, LIME
  `LimeImageExplainer`, and a GradientTape saliency map, each falling back
  to saliency on failure; `generate_all_explanations()` builds the
  6-image SHAP gallery.
- **`modules/evaluation.py`** — `compute_metrics()` plus all 9 plotting
  functions and CSV/Markdown writers.
- **`modules/utils.py`** — `run_tests()` implements the 7-test validation
  suite described below; checkpoint save/load so training survives Colab
  disconnects.

## How to present to judges

1. Open with the **Architecture tab** in the demo — explain the privacy
   problem, then the adaptive-selection formula.
2. Switch to **Hospital Data Simulation** — show the 5 hospitals' skewed
   class distributions; this is the realism/heterogeneity story.
3. Go to **Live X-ray Analysis** — click an example image, show the
   prediction, confidence, and saliency overlay updating live. This is the
   most convincing moment of the demo — do it twice (one Normal, one
   Pneumonia example).
4. Show **Convergence & Communication** — point at the AEFL line converging
   as fast or faster than FedAvg while transmitting less data; read out the
   "AEFL saved X% communication" line.
5. Show **Performance Results** — the AEFL row is highlighted green in the
   metrics table.
6. Show **SHAP Explanations Gallery** — a few pre-generated explanation
   grids, to demonstrate the explainability story isn't just the live tab.
7. Close on **Test Results** — all automated tests passing is a concrete,
   defensible answer to "how do you know this works?"

## Testing & Validation

`modules/utils.py::run_tests()` runs automatically before and after
training (`results/test_report.txt` holds the post-training report) and
checks:

1. Model output is bounded in `[0, 1]` (valid probability).
2. Adaptive aggregation weights sum to 1.
3. Every simulated hospital keeps ≥10% of each class (no degenerate splits).
4. Averaging two identical models via FedAvg reproduces the same weights.
5. Explanation maps (SHAP/saliency) match the input's spatial shape.
6. Communication-cost bookkeeping matches `bytes = model_size × 2 × clients × rounds`.
7. A saved-then-reloaded model reproduces identical predictions.

## Key results

*(Filled in automatically after `python aefl_main.py` finishes — see
`results/performance_comparison.csv`, `results/communication_summary.csv`,
and `results/summary.md` for the live numbers.)*

| Method | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Centralized CNN | XX.X | XX.X | XX.X | XX.X | 0.XX |
| FedAvg | XX.X | XX.X | XX.X | XX.X | 0.XX |
| FedProx | XX.X | XX.X | XX.X | XX.X | 0.XX |
| **Proposed AEFL** | **XX.X** | **XX.X** | **XX.X** | **XX.X** | **0.XX** |
| AEFL (Acc-focused) | XX.X | XX.X | XX.X | XX.X | 0.XX |
| AEFL (Comm-focused) | XX.X | XX.X | XX.X | XX.X | 0.XX |
| AEFL (Top-2) | XX.X | XX.X | XX.X | XX.X | 0.XX |

**Communication savings (AEFL vs. FedAvg): XX.X%** · **Time savings: XX.X%**

## References

1. McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep
   Networks from Decentralized Data.* AISTATS — the FedAvg baseline.
2. Li, T. et al. (2020). *Federated Optimization in Heterogeneous
   Networks.* MLSys — FedProx and the proximal-term formulation.
3. Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model
   Predictions.* NeurIPS — SHAP.
4. Ribeiro, M. et al. (2016). *"Why Should I Trust You?": Explaining the
   Predictions of Any Classifier.* KDD — LIME.
5. Kermany, D. et al. (2018). *Identifying Medical Diagnoses and Treatable
   Diseases by Image-Based Deep Learning.* Cell — source of the Chest X-ray
   Pneumonia dataset.

## Datasets

- **Primary:** Chest X-ray Pneumonia (Kaggle: `paultimothymooney/chest-xray-pneumonia`)
- **Secondary (generalization testing):** COVID-19 Radiography Database
  (`tawsifurrahman/covid19-radiography-database`), RSNA Pneumonia Detection
  Challenge (`c/rsna-pneumonia-detection-challenge`, stage 2 labels)
- **Future work (too large for this Colab-scale prototype):** NIH
  ChestX-ray14 (112,120 images, 14 diseases), CheXpert Stanford (224,000+
  images)

All training uses `image_size=128` and `seed=42` throughout for
reproducibility and Colab-feasibility.
