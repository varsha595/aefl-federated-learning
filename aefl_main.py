"""
aefl_main.py — AEFL: Adaptive Explainable Federated Learning Framework for
Privacy-Preserving Multi-Hospital Healthcare AI.

Master training script. Run with:  python aefl_main.py

Trains and compares 7 algorithms (Centralized, FedAvg, FedProx, AEFL, and 3
AEFL ablations) on a simulated 5-hospital Non-IID federation for chest X-ray
pneumonia detection, generates SHAP/LIME/saliency explanations, and writes
every plot/table the results/ directory needs for the demo and the report.

Safe to interrupt and re-run: every algorithm checkpoints itself and is
skipped on the next run if already trained (see modules/utils.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import utils, data_loader, evaluation, federated, explainability

# --------------------------------------------------------------------------
# CONFIG — every tunable parameter lives here (Part 11 of the spec)
# --------------------------------------------------------------------------

CONFIG = {
    "seed": 42,
    "image_size": 128,
    "batch_size": 32,
    "num_clients": 5,
    "num_rounds": 20,
    "local_epochs": 2,
    "learning_rate": 0.001,
    "centralized_epochs": 15,
    "fedprox_mu": 0.01,
    "alpha": 1 / 3,
    "beta": 1 / 3,
    "gamma": 1 / 3,
    "top_k_clients": 3,
    "results_dir": "results",
    "models_dir": "saved_models",
    "min_class_pct": 0.10,
}


def main():
    utils.banner("🚀 AEFL — Adaptive Explainable Federated Learning Framework")
    utils.set_seed(CONFIG["seed"])
    utils.ensure_dirs(CONFIG)

    utils.section("📋 Running pre-training tests...")
    pre_ok = utils.run_tests(CONFIG, save_report=False)
    print(f"  {'✅ All tests passed' if pre_ok else '⚠️ Some tests failed — continuing anyway'}")

    # ----------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------
    utils.section("📂 Loading datasets...")
    primary = data_loader.load_chest_xray_pneumonia(
        image_size=CONFIG["image_size"], seed=CONFIG["seed"])
    if primary is None:
        print("❌ Primary dataset (Chest X-ray Pneumonia) could not be loaded. Aborting.")
        sys.exit(1)
    X_train, y_train, X_val, y_val, X_test, y_test = primary

    generalization_sets = {}
    covid = data_loader.load_covid_radiography(image_size=CONFIG["image_size"],
                                                seed=CONFIG["seed"])
    if covid is not None:
        generalization_sets["COVID-19 Radiography"] = covid
    rsna = data_loader.load_rsna_pneumonia(image_size=CONFIG["image_size"],
                                            seed=CONFIG["seed"])
    if rsna is not None:
        generalization_sets["RSNA Pneumonia"] = rsna

    # ----------------------------------------------------------------
    # Non-IID hospital simulation
    # ----------------------------------------------------------------
    utils.section("🏥 Creating Non-IID hospital simulation...")
    hospitals = data_loader.create_hospital_splits(
        X_train, y_train, data_loader.HOSPITAL_CONFIG,
        seed=CONFIG["seed"], min_class_pct=CONFIG["min_class_pct"])

    evaluation.plot_hospital_distribution(
        hospitals, os.path.join(CONFIG["results_dir"], "hospital_distribution.png"))
    print("  ✅ hospital_distribution.png")

    # ----------------------------------------------------------------
    # Train every algorithm (each is checkpoint-aware and skips if done)
    # ----------------------------------------------------------------
    all_results = {}

    utils.section("📊 Training Centralized CNN (upper bound)...")
    _, res = federated.run_centralized(X_train, y_train, X_val, y_val, X_test, y_test, CONFIG)
    all_results["Centralized"] = res

    utils.section("📊 Training FedAvg (baseline)...")
    _, res = federated.run_fedavg(hospitals, X_val, y_val, X_test, y_test, CONFIG)
    all_results["FedAvg"] = res

    utils.section("📊 Training FedProx (baseline)...")
    _, res = federated.run_fedprox(hospitals, X_val, y_val, X_test, y_test, CONFIG)
    all_results["FedProx"] = res

    utils.section("📊 Training Proposed AEFL...")
    aefl_model, res = federated.run_aefl(
        hospitals, X_val, y_val, X_test, y_test, CONFIG, algorithm_name="AEFL")
    all_results["AEFL"] = res

    utils.section("📊 Training AEFL Ablation A (accuracy-focused, alpha=0.6)...")
    _, res = federated.run_aefl(
        hospitals, X_val, y_val, X_test, y_test, CONFIG, algorithm_name="AEFL-AccFocus",
        alpha=0.6, beta=0.2, gamma=0.2)
    all_results["AEFL-AccFocus"] = res

    utils.section("📊 Training AEFL Ablation B (communication-focused, gamma=0.6)...")
    _, res = federated.run_aefl(
        hospitals, X_val, y_val, X_test, y_test, CONFIG, algorithm_name="AEFL-CommFocus",
        alpha=0.2, beta=0.2, gamma=0.6)
    all_results["AEFL-CommFocus"] = res

    utils.section("📊 Training AEFL Ablation C (top-2 clients)...")
    _, res = federated.run_aefl(
        hospitals, X_val, y_val, X_test, y_test, CONFIG, algorithm_name="AEFL-Top2", top_k=2)
    all_results["AEFL-Top2"] = res

    # ----------------------------------------------------------------
    # Explainability — SHAP / LIME / Saliency on the AEFL model
    # ----------------------------------------------------------------
    utils.section("🔍 Generating SHAP/LIME/Saliency explanations...")
    explainability.generate_all_explanations(aefl_model, X_test, y_test, CONFIG["results_dir"],
                                              seed=CONFIG["seed"])

    # ----------------------------------------------------------------
    # Cross-dataset generalization (secondary datasets, if loaded)
    # ----------------------------------------------------------------
    if generalization_sets:
        utils.section("🌍 Evaluating cross-dataset generalization of AEFL...")
        for name, (Xg, yg) in generalization_sets.items():
            m = evaluation.compute_metrics(yg, aefl_model.predict(Xg, verbose=0).ravel())
            print(f"  {name}: accuracy={m['accuracy']:.3f} | f1={m['f1']:.3f} | auc={m['auc']:.3f}")

    # ----------------------------------------------------------------
    # Plots, CSVs, summary
    # ----------------------------------------------------------------
    utils.section("📊 Generating comparison plots...")
    metrics_dict = {a: r["test_metrics"] for a, r in all_results.items()}
    results_dir = CONFIG["results_dir"]

    histories = {a: r["history"] for a, r in all_results.items()}
    evaluation.plot_convergence(histories, os.path.join(results_dir, "convergence_plot.png"))
    print("  ✅ convergence_plot.png")

    evaluation.plot_performance_comparison(
        metrics_dict, os.path.join(results_dir, "performance_comparison.png"))
    print("  ✅ performance_comparison.png")

    evaluation.plot_communication_cost(
        all_results, os.path.join(results_dir, "communication_cost.png"))
    print("  ✅ communication_cost.png")

    evaluation.plot_participation_scores(
        all_results["AEFL"]["history"], [h["name"] for h in hospitals],
        os.path.join(results_dir, "participation_scores.png"))
    print("  ✅ participation_scores.png")

    evaluation.plot_confusion_matrices(
        metrics_dict, os.path.join(results_dir, "confusion_matrices.png"))
    print("  ✅ confusion_matrices.png")

    evaluation.plot_ablation_study(
        metrics_dict, os.path.join(results_dir, "ablation_study.png"))
    print("  ✅ ablation_study.png")

    evaluation.plot_roc_curves(metrics_dict, os.path.join(results_dir, "roc_curves.png"))
    print("  ✅ roc_curves.png")

    evaluation.save_performance_csv(
        metrics_dict, os.path.join(results_dir, "performance_comparison.csv"))
    evaluation.save_communication_csv(
        all_results, os.path.join(results_dir, "communication_summary.csv"))
    evaluation.write_summary_markdown(
        metrics_dict, all_results, hospitals, os.path.join(results_dir, "summary.md"))
    print("  ✅ performance_comparison.csv, communication_summary.csv, summary.md")

    # ----------------------------------------------------------------
    # Post-training tests
    # ----------------------------------------------------------------
    utils.section("📋 Running post-training tests...")
    utils.run_tests(CONFIG, hospitals=hospitals, save_report=True)

    # ----------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------
    utils.banner("✅ TRAINING COMPLETE")
    print(f"{'Method':<20}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}")
    best_fl = max((a for a in metrics_dict if a != "Centralized"),
                  key=lambda a: metrics_dict[a]["f1"], default=None)
    for a, m in metrics_dict.items():
        marker = "  ← BEST FL METHOD" if a == best_fl else ""
        print(f"{evaluation.label_for(a):<20}{m['accuracy']*100:>6.1f} {m['precision']*100:>6.1f} "
              f"{m['recall']*100:>6.1f} {m['f1']*100:>6.1f} {m['auc']:>6.2f}{marker}")

    if "FedAvg" in all_results and "AEFL" in all_results:
        fedavg_mb = all_results["FedAvg"]["total_bytes"] / 1e6
        aefl_mb = all_results["AEFL"]["total_bytes"] / 1e6
        comm_savings = 100 * (1 - aefl_mb / fedavg_mb) if fedavg_mb else 0
        fedavg_t = all_results["FedAvg"]["total_time"]
        aefl_t = all_results["AEFL"]["total_time"]
        time_savings = 100 * (1 - aefl_t / fedavg_t) if fedavg_t else 0
        print(f"\nCommunication savings (AEFL vs FedAvg): {comm_savings:.1f}%")
        print(f"Time savings: {time_savings:.1f}%")

    print("\n🚀 Run: python aefl_demo.py  →  get public URL for judges")


if __name__ == "__main__":
    main()
