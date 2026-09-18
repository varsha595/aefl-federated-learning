"""
evaluation.py — metrics computation and every plot/table in Part 7 of the spec.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 150

ALGO_COLORS = {
    "Centralized": "#6b7280",
    "FedAvg": "#3b82f6",
    "FedProx": "#8b5cf6",
    "AEFL": "#16a34a",
    "AEFL-AccFocus": "#f59e0b",
    "AEFL-CommFocus": "#ef4444",
    "AEFL-Top2": "#0ea5e9",
}

DISPLAY_NAMES = {
    "Centralized": "Centralized CNN",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "AEFL": "Proposed AEFL",
    "AEFL-AccFocus": "AEFL (Acc-focused)",
    "AEFL-CommFocus": "AEFL (Comm-focused)",
    "AEFL-Top2": "AEFL (Top-2)",
}


def _color(name):
    return ALGO_COLORS.get(name, "#111827")


def _label(name):
    return DISPLAY_NAMES.get(name, name)


# public aliases
color_for = _color
label_for = _label


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix, roc_curve)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        auc, fpr, tpr = 0.5, np.array([0, 1]), np.array([0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
    }


def rounds_to_target_acc(history, target=0.8):
    for r, acc in zip(history.get("round", []), history.get("val_acc", [])):
        if acc >= target:
            return r
    return None


# --------------------------------------------------------------------------
# Plot 1 — Hospital data distribution
# --------------------------------------------------------------------------

def plot_hospital_distribution(hospitals, save_path):
    names = [h["name"] for h in hospitals]
    normal_pct = [h["actual_normal_pct"] * 100 for h in hospitals]
    pneumonia_pct = [100 - p for p in normal_pct]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, normal_pct, width, label="Normal", color="#3b82f6")
    ax.bar(x + width / 2, pneumonia_pct, width, label="Pneumonia", color="#ef4444")
    ax.set_ylabel("Percentage of hospital's data")
    ax.set_title("Non-IID Class Distribution Across Simulated Hospitals")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    for i, h in enumerate(hospitals):
        ax.text(i, 95, f"n={h['n_total']}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 2 — Convergence
# --------------------------------------------------------------------------

def plot_convergence(histories, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, hist in histories.items():
        rounds = hist.get("round") or hist.get("epoch")
        if rounds is None:
            continue
        ax.plot(rounds, hist["val_acc"], label=_label(name), color=_color(name),
                marker="o", markersize=3, linewidth=2)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="80% target")
    ax.set_xlabel("Round / Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Convergence Comparison Across FL Algorithms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 3 — Performance comparison
# --------------------------------------------------------------------------

def plot_performance_comparison(metrics_dict, save_path):
    algos = list(metrics_dict.keys())
    metric_keys = ["accuracy", "f1", "auc"]
    metric_labels = ["Accuracy", "F1-Score", "AUC-ROC"]

    x = np.arange(len(algos))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        values = [metrics_dict[a][mk] for a in algos]
        ax.bar(x + (i - 1) * width, values, width, label=ml)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a) for a in algos], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison — All Methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 4 — Communication cost / time / avg clients
# --------------------------------------------------------------------------

def plot_communication_cost(results_dict, save_path):
    algos = [a for a in results_dict if results_dict[a].get("rounds", 0) > 0
             and "history" in results_dict[a] and "round" in results_dict[a]["history"]]
    total_mb = [results_dict[a]["total_bytes"] / 1e6 for a in algos]
    total_time = [results_dict[a]["total_time"] for a in algos]
    avg_clients = []
    for a in algos:
        sel = results_dict[a]["history"].get("clients_selected", [])
        avg_clients.append(np.mean([len(c) for c in sel]) if sel else 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar([_label(a) for a in algos], total_mb, color=[_color(a) for a in algos])
    axes[0].set_title("Total Communication (MB)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar([_label(a) for a in algos], total_time, color=[_color(a) for a in algos])
    axes[1].set_title("Total Training Time (s)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar([_label(a) for a in algos], avg_clients, color=[_color(a) for a in algos])
    axes[2].set_title("Avg. Clients Participating / Round")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 5 — Participation scores over rounds (AEFL)
# --------------------------------------------------------------------------

def plot_participation_scores(aefl_history, hospital_names, save_path):
    scores = np.array(aefl_history["participation_scores"])  # (rounds, n_hospitals)
    rounds = aefl_history["round"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, name in enumerate(hospital_names):
        ax.plot(rounds, scores[:, i], marker="o", markersize=3, label=name, linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Participation Score $S_i$")
    ax.set_title("AEFL — Per-Hospital Participation Score Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 6 — Confusion matrices
# --------------------------------------------------------------------------

def plot_confusion_matrices(metrics_dict, save_path, class_names=("Normal", "Pneumonia")):
    algos = list(metrics_dict.keys())
    n = len(algos)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, a in enumerate(algos):
        cm = np.array(metrics_dict[a]["confusion_matrix"])
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(_label(a), fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names, fontsize=8)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
    for j in range(len(algos), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 8 — Ablation study
# --------------------------------------------------------------------------

def plot_ablation_study(metrics_dict, save_path):
    variants = [a for a in metrics_dict if a.startswith("AEFL")]
    x = np.arange(len(variants))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (mk, ml) in enumerate(zip(["accuracy", "f1", "auc"], ["Accuracy", "F1", "AUC"])):
        values = [metrics_dict[a][mk] for a in variants]
        ax.bar(x + (i - 1) * width, values, width, label=ml)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a) for a in variants], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation Study — AEFL Variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 9 — ROC curves
# --------------------------------------------------------------------------

def plot_roc_curves(metrics_dict, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    for a, m in metrics_dict.items():
        ax.plot(m["roc_fpr"], m["roc_tpr"], label=f"{_label(a)} (AUC={m['auc']:.3f})",
                color=_color(a), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Methods")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# CSV / Markdown summaries
# --------------------------------------------------------------------------

def save_performance_csv(metrics_dict, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"])
        for a, m in metrics_dict.items():
            writer.writerow([_label(a), f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                              f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['auc']:.4f}"])


def save_communication_csv(results_dict, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Total_MB", "Total_Time_s", "Rounds", "Avg_Clients_Per_Round"])
        for a, r in results_dict.items():
            sel = r.get("history", {}).get("clients_selected", [])
            avg_clients = np.mean([len(c) for c in sel]) if sel else "-"
            writer.writerow([_label(a), f"{r['total_bytes'] / 1e6:.2f}",
                              f"{r['total_time']:.1f}", r.get("rounds", "-"), avg_clients])


def write_summary_markdown(metrics_dict, results_dict, hospitals, save_path):
    lines = ["# AEFL — Key Findings\n"]
    lines.append("## Performance comparison\n")
    lines.append("| Method | Accuracy | Precision | Recall | F1 | AUC-ROC |")
    lines.append("|---|---|---|---|---|---|")
    for a, m in metrics_dict.items():
        lines.append(f"| {_label(a)} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                      f"{m['recall']:.3f} | {m['f1']:.3f} | {m['auc']:.3f} |")

    lines.append("\n## Communication efficiency\n")
    if "FedAvg" in results_dict and "AEFL" in results_dict:
        fedavg_mb = results_dict["FedAvg"]["total_bytes"] / 1e6
        aefl_mb = results_dict["AEFL"]["total_bytes"] / 1e6
        savings = 100 * (1 - aefl_mb / fedavg_mb) if fedavg_mb else 0
        fedavg_t = results_dict["FedAvg"]["total_time"]
        aefl_t = results_dict["AEFL"]["total_time"]
        time_savings = 100 * (1 - aefl_t / fedavg_t) if fedavg_t else 0
        lines.append(f"- AEFL transmitted **{aefl_mb:.1f} MB** vs FedAvg's **{fedavg_mb:.1f} MB** "
                      f"— a **{savings:.1f}%** communication reduction.")
        lines.append(f"- AEFL took **{aefl_t:.1f}s** vs FedAvg's **{fedavg_t:.1f}s** "
                      f"— a **{time_savings:.1f}%** time reduction.")

    lines.append("\n## Convergence speed (rounds to reach 80% val accuracy)\n")
    for a, r in results_dict.items():
        if "history" in r and "round" in r.get("history", {}):
            rt = rounds_to_target_acc(r["history"])
            lines.append(f"- {_label(a)}: {rt if rt else 'not reached'}")

    lines.append("\n## Hospital simulation (Non-IID)\n")
    for h in hospitals:
        lines.append(f"- **{h['name']}**: {h['n_total']} images, "
                      f"{h['actual_normal_pct']*100:.0f}% Normal / "
                      f"{100 - h['actual_normal_pct']*100:.0f}% Pneumonia, "
                      f"comm_score={h['comm_score']}")

    best_fl = max(
        (a for a in metrics_dict if a != "Centralized"),
        key=lambda a: metrics_dict[a]["f1"], default=None,
    )
    if best_fl:
        lines.append(f"\n**Best federated method: {_label(best_fl)}** "
                      f"(F1={metrics_dict[best_fl]['f1']:.3f})")

    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
