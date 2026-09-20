"""
federated.py — every FL algorithm compared in the AEFL study:
Centralized (upper bound), FedAvg, FedProx, and the proposed AEFL together
with its three ablations. All share `local_train` / `fedavg_aggregate` /
`weighted_aggregate` primitives so the comparison is apples-to-apples.
"""

import time
import numpy as np
import tensorflow as tf

from modules.model import create_model, model_size_bytes, proximal_term
from modules.utils import checkpoint_exists, save_checkpoint, load_checkpoint, fmt_time


# --------------------------------------------------------------------------
# Core FL primitives
# --------------------------------------------------------------------------

def local_train(model, X, y, epochs, batch_size, learning_rate,
                 global_trainable_weights=None, mu=0.0):
    """Manual training loop (so we can add the FedProx proximal term).

    `global_trainable_weights`, when given, must be a numpy snapshot of
    `model.trainable_variables` taken right after the model was set to the
    current global weights (i.e. before this call mutates them).

    Returns the model's new weights and a small per-epoch history.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bce = tf.keras.losses.BinaryCrossentropy()
    n = len(X)
    history = {"loss": [], "accuracy": []}
    use_proximal = mu > 0 and global_trainable_weights is not None

    for _ in range(epochs):
        perm = np.random.permutation(n)
        Xs, ys = X[perm], y[perm]
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size].reshape(-1, 1).astype("float32")
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                loss = bce(yb, preds)
                if use_proximal:
                    loss = loss + proximal_term(model.trainable_variables,
                                                 global_trainable_weights, mu)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(yb, preds))
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            n_batches += 1
        history["loss"].append(epoch_loss / max(n_batches, 1))
        history["accuracy"].append(epoch_acc / max(n_batches, 1))

    return model.get_weights(), history


def evaluate_model(model, X, y, batch_size=64):
    loss, acc = model.evaluate(X, y.reshape(-1, 1).astype("float32"),
                                batch_size=batch_size, verbose=0)
    return float(loss), float(acc)


def evaluate_balanced_accuracy(model, X, y, batch_size=64):
    """Average of per-class recall (sensitivity + specificity) / 2.

    Unlike raw accuracy, this can't be gamed by a model that collapses to
    always predicting the majority class under severe class imbalance —
    exactly the failure mode data_quality (D_i) alone doesn't always catch
    when a hospital's local A_i term rewards that shortcut.
    """
    y_prob = model.predict(X, batch_size=batch_size, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(y).ravel()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sensitivity + specificity) / 2.0


def fedavg_aggregate(weights_list, sample_counts):
    total = sum(sample_counts)
    avg = [np.zeros_like(w) for w in weights_list[0]]
    for weights, n in zip(weights_list, sample_counts):
        frac = n / total
        for i, w in enumerate(weights):
            avg[i] += frac * w
    return avg


def weighted_aggregate(weights_list, omega):
    avg = [np.zeros_like(w) for w in weights_list[0]]
    for weights, w_i in zip(weights_list, omega):
        for i, w in enumerate(weights):
            avg[i] += w_i * w
    return avg


def compute_data_quality_score(y):
    n = len(y)
    n_pos = float(np.sum(y == 1))
    n_neg = n - n_pos
    return 2.0 * min(n_pos / n, n_neg / n)


def compute_participation_scores(accuracies, data_quality, comm_scores, alpha, beta, gamma):
    return [alpha * a + beta * d + gamma * c
            for a, d, c in zip(accuracies, data_quality, comm_scores)]


def select_top_k(scores, k):
    return list(np.argsort(scores)[::-1][:k])


# --------------------------------------------------------------------------
# Algorithm 1 — Centralized upper bound
# --------------------------------------------------------------------------

def run_centralized(X_train, y_train, X_val, y_val, X_test, y_test, config,
                     algorithm_name="Centralized"):
    if checkpoint_exists(algorithm_name, config):
        model, results = load_checkpoint(algorithm_name, config)
        return model, results

    model = create_model(config["image_size"], config["learning_rate"])
    history = {"epoch": [], "train_acc": [], "val_acc": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(1, config["centralized_epochs"] + 1):
        _, epoch_hist = local_train(model, X_train, y_train, epochs=1,
                                     batch_size=config["batch_size"],
                                     learning_rate=config["learning_rate"])
        val_loss, val_acc = evaluate_model(model, X_val, y_val)
        history["epoch"].append(epoch)
        history["train_acc"].append(epoch_hist["accuracy"][0])
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        print(f"  Epoch {epoch}/{config['centralized_epochs']}: "
              f"acc={epoch_hist['accuracy'][0]:.3f} | val_acc={val_acc:.3f}")
    total_time = time.time() - t0

    test_metrics = _test_metrics(model, X_test, y_test)
    results = {
        "algorithm": algorithm_name,
        "history": history,
        "test_metrics": test_metrics,
        "total_time": total_time,
        "total_bytes": 0,
        "rounds": config["centralized_epochs"],
    }
    save_checkpoint(algorithm_name, model, results, config)
    return model, results


# --------------------------------------------------------------------------
# Algorithm 2 / 3 — FedAvg / FedProx (shared implementation, mu=0 for FedAvg)
# --------------------------------------------------------------------------

def run_fedavg_family(hospitals, X_val, y_val, X_test, y_test, config,
                       algorithm_name, mu=0.0):
    if checkpoint_exists(algorithm_name, config):
        model, results = load_checkpoint(algorithm_name, config)
        return model, results

    model = create_model(config["image_size"], config["learning_rate"])
    global_weights = model.get_weights()
    model_bytes = model_size_bytes(model)

    history = {"round": [], "val_acc": [], "val_loss": [], "clients_selected": [],
               "bytes_transmitted": [], "time_taken": []}
    total_bytes = 0
    t_start = time.time()

    for rnd in range(1, config["num_rounds"] + 1):
        t0 = time.time()
        global_trainable_snapshot = None
        if mu > 0:
            model.set_weights(global_weights)
            global_trainable_snapshot = [v.numpy() for v in model.trainable_variables]

        local_weights, sample_counts = [], []
        for h in hospitals:
            model.set_weights(global_weights)
            w, _ = local_train(model, h["X"], h["y"], epochs=config["local_epochs"],
                                batch_size=config["batch_size"],
                                learning_rate=config["learning_rate"],
                                global_trainable_weights=global_trainable_snapshot, mu=mu)
            local_weights.append(w)
            sample_counts.append(len(h["X"]))

        global_weights = fedavg_aggregate(local_weights, sample_counts)
        model.set_weights(global_weights)
        val_loss, val_acc = evaluate_model(model, X_val, y_val)
        elapsed = time.time() - t0
        round_bytes = model_bytes * 2 * len(hospitals)
        total_bytes += round_bytes

        history["round"].append(rnd)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["clients_selected"].append([h["name"] for h in hospitals])
        history["bytes_transmitted"].append(round_bytes)
        history["time_taken"].append(elapsed)

        print(f"  Round {rnd}/{config['num_rounds']}: val_acc={val_acc:.3f} | "
              f"clients={len(hospitals)}/{len(hospitals)} | "
              f"{round_bytes / 1e6:.1f} MB transmitted")

    total_time = time.time() - t_start
    test_metrics = _test_metrics(model, X_test, y_test)
    results = {
        "algorithm": algorithm_name,
        "history": history,
        "test_metrics": test_metrics,
        "total_time": total_time,
        "total_bytes": total_bytes,
        "rounds": config["num_rounds"],
    }
    save_checkpoint(algorithm_name, model, results, config)
    return model, results


def run_fedavg(hospitals, X_val, y_val, X_test, y_test, config):
    return run_fedavg_family(hospitals, X_val, y_val, X_test, y_test, config,
                              algorithm_name="FedAvg", mu=0.0)


def run_fedprox(hospitals, X_val, y_val, X_test, y_test, config):
    return run_fedavg_family(hospitals, X_val, y_val, X_test, y_test, config,
                              algorithm_name="FedProx", mu=config["fedprox_mu"])


# --------------------------------------------------------------------------
# Algorithm 4-7 — Proposed AEFL and its ablations
# --------------------------------------------------------------------------

def run_aefl(hospitals, X_val, y_val, X_test, y_test, config, algorithm_name="AEFL",
             alpha=None, beta=None, gamma=None, top_k=None, verbose_label="all"):
    if checkpoint_exists(algorithm_name, config):
        model, results = load_checkpoint(algorithm_name, config)
        return model, results

    alpha = config["alpha"] if alpha is None else alpha
    beta = config["beta"] if beta is None else beta
    gamma = config["gamma"] if gamma is None else gamma
    top_k = config["top_k_clients"] if top_k is None else top_k

    num_hospitals = len(hospitals)
    model = create_model(config["image_size"], config["learning_rate"])
    global_weights = model.get_weights()
    model_bytes = model_size_bytes(model)

    data_quality = [compute_data_quality_score(h["y"]) for h in hospitals]
    comm_scores = [h["comm_score"] for h in hospitals]
    local_val_accs = [0.5] * num_hospitals

    history = {"round": [], "val_acc": [], "val_loss": [], "clients_selected": [],
               "bytes_transmitted": [], "time_taken": [], "participation_scores": []}
    total_bytes = 0
    t_start = time.time()

    for rnd in range(1, config["num_rounds"] + 1):
        t0 = time.time()
        scores = compute_participation_scores(local_val_accs, data_quality, comm_scores,
                                               alpha, beta, gamma)

        if rnd == 1:
            selected = list(range(num_hospitals))
            selected_label = "first round all"
        else:
            selected = select_top_k(scores, top_k)
            names = [hospitals[i]["name"] for i in selected]
            selected_label = ",".join(names)

        local_weights, sample_counts = [], []
        for idx in selected:
            h = hospitals[idx]
            model.set_weights(global_weights)
            w, _ = local_train(model, h["X"], h["y"], epochs=config["local_epochs"],
                                batch_size=config["batch_size"],
                                learning_rate=config["learning_rate"])
            local_weights.append(w)
            sample_counts.append(len(h["X"]))

            # A_i for next round: balanced accuracy on this hospital's own
            # local val split (not raw accuracy — see evaluate_balanced_accuracy).
            model.set_weights(w)
            local_acc = evaluate_balanced_accuracy(model, h["X_val"], h["y_val"])
            local_val_accs[idx] = local_acc

        sel_scores = [scores[i] for i in selected]
        score_sum = sum(sel_scores) if sum(sel_scores) > 0 else 1.0
        omega = [s / score_sum for s in sel_scores]
        global_weights = weighted_aggregate(local_weights, omega)
        model.set_weights(global_weights)

        val_loss, val_acc = evaluate_model(model, X_val, y_val)
        elapsed = time.time() - t0
        round_bytes = model_bytes * 2 * len(selected)
        total_bytes += round_bytes

        history["round"].append(rnd)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["clients_selected"].append([hospitals[i]["name"] for i in selected])
        history["bytes_transmitted"].append(round_bytes)
        history["time_taken"].append(elapsed)
        history["participation_scores"].append(list(scores))

        savings_note = ""
        if rnd > 1:
            fedavg_bytes = model_bytes * 2 * num_hospitals
            saved_pct = 100.0 * (1 - round_bytes / fedavg_bytes)
            savings_note = f" | SAVED {saved_pct:.0f}%"

        print(f"  Round {rnd}/{config['num_rounds']}: val_acc={val_acc:.3f} | "
              f"clients={len(selected)}/{num_hospitals} "
              f"({'first round all' if rnd == 1 else '[' + selected_label + ']'}) | "
              f"{round_bytes / 1e6:.1f} MB{savings_note}"
              + (f" | scores={[round(s, 2) for s in scores]}" if rnd == 1 else ""))

    total_time = time.time() - t_start
    test_metrics = _test_metrics(model, X_test, y_test)
    results = {
        "algorithm": algorithm_name,
        "history": history,
        "test_metrics": test_metrics,
        "total_time": total_time,
        "total_bytes": total_bytes,
        "rounds": config["num_rounds"],
        "hyperparams": {"alpha": alpha, "beta": beta, "gamma": gamma, "top_k": top_k},
    }
    save_checkpoint(algorithm_name, model, results, config)
    return model, results


# --------------------------------------------------------------------------
# Shared test-set evaluation
# --------------------------------------------------------------------------

def _test_metrics(model, X_test, y_test):
    from modules.evaluation import compute_metrics
    y_prob = model.predict(X_test, batch_size=64, verbose=0).ravel()
    return compute_metrics(y_test, y_prob)
