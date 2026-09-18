"""
explainability.py — SHAP GradientExplainer, LIME, and gradient saliency, with
graceful fallback (SHAP/LIME failures degrade to saliency, which always
works) plus the side-by-side explanation grid used for the SHAP gallery.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Method 3 — Gradient saliency (always works, fast; also the SHAP/LIME fallback)
# --------------------------------------------------------------------------

def generate_saliency(model, image):
    """image: (H, W, 3) float32 in [0,1]. Returns an (H, W) saliency map in [0,1]."""
    img_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        pred = model(img_tensor, training=False)
    grads = tape.gradient(pred, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    smax = saliency.max()
    if smax > 0:
        saliency = saliency / smax
    return saliency


# --------------------------------------------------------------------------
# Method 1 — SHAP GradientExplainer (falls back to saliency on failure)
# --------------------------------------------------------------------------

def generate_shap(model, background_samples, image):
    try:
        import shap
        explainer = shap.GradientExplainer(model, background_samples)
        shap_values = explainer.shap_values(image[np.newaxis, ...])
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        sv = np.array(sv)
        sv = sv.reshape(sv.shape[-3], sv.shape[-2], -1) if sv.ndim > 3 else sv[0]
        heatmap = np.abs(sv).sum(axis=-1) if sv.ndim == 3 else np.abs(sv)
        hmax = heatmap.max()
        if hmax > 0:
            heatmap = heatmap / hmax
        return heatmap, "SHAP"
    except Exception as e:
        print(f"  ⚠️  SHAP failed ({e}) — falling back to saliency")
        return generate_saliency(model, image), "Saliency (SHAP fallback)"


# --------------------------------------------------------------------------
# Method 2 — LIME (falls back to saliency on failure)
# --------------------------------------------------------------------------

def generate_lime(model, image, num_samples=100):
    try:
        from lime import lime_image

        def predict_fn(images):
            images = np.array(images, dtype="float32")
            if images.max() > 1.0:
                images = images / 255.0
            probs = model.predict(images, verbose=0).ravel()
            return np.stack([1 - probs, probs], axis=1)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            (image * 255).astype("uint8"), predict_fn,
            top_labels=1, hide_color=0, num_samples=num_samples,
        )
        label = explanation.top_labels[0]
        _, mask = explanation.get_image_and_mask(
            label, positive_only=True, num_features=8, hide_rest=False)
        return mask.astype("float32"), "LIME"
    except Exception as e:
        print(f"  ⚠️  LIME failed ({e}) — falling back to saliency")
        return generate_saliency(model, image), "Saliency (LIME fallback)"


# --------------------------------------------------------------------------
# Region localization — turns a heatmap into a human-readable lung region
# --------------------------------------------------------------------------

def localize_region(heatmap):
    h, w = heatmap.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    total = heatmap.sum()
    if total <= 0:
        return "diffuse / no strong focus"
    cy = float((ys * heatmap).sum() / total)
    cx = float((xs * heatmap).sum() / total)
    vertical = "upper" if cy < h * 0.5 else "lower"
    # radiograph convention: patient's right lung appears on the image's left half
    horizontal = "right lung" if cx < w * 0.5 else "left lung"
    return f"{vertical} {horizontal}"


# --------------------------------------------------------------------------
# Composite [Original | SHAP | LIME | Saliency | Overlay] figure
# --------------------------------------------------------------------------

def generate_explanation_grid(model, image, true_label, save_path, background_samples,
                               class_names=("Normal", "Pneumonia")):
    pred_prob = float(model.predict(image[np.newaxis, ...], verbose=0)[0, 0])
    pred_class = int(pred_prob >= 0.5)
    confidence = pred_prob if pred_class == 1 else 1 - pred_prob

    saliency = generate_saliency(model, image)
    shap_map, shap_label = generate_shap(model, background_samples, image)
    lime_map, lime_label = generate_lime(model, image)
    region = localize_region(saliency)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Original")

    axes[1].imshow(image)
    axes[1].imshow(_resize_map(shap_map, image.shape[:2]), cmap="jet", alpha=0.5)
    axes[1].set_title(shap_label)

    axes[2].imshow(image)
    axes[2].imshow(_resize_map(lime_map, image.shape[:2]), cmap="jet", alpha=0.5)
    axes[2].set_title(lime_label)

    axes[3].imshow(saliency, cmap="hot")
    axes[3].set_title("Saliency")

    axes[4].imshow(image)
    axes[4].imshow(saliency, cmap="jet", alpha=0.5)
    axes[4].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"True: {class_names[true_label]}  |  Predicted: {class_names[pred_class]} "
        f"({confidence * 100:.1f}% confidence)  |  Focus region: {region}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return {
        "pred_class": class_names[pred_class],
        "true_class": class_names[true_label],
        "confidence": confidence,
        "region": region,
        "path": save_path,
    }


def _resize_map(m, target_shape):
    if m.shape[:2] == target_shape:
        return m
    from PIL import Image
    img = Image.fromarray((np.clip(m, 0, 1) * 255).astype("uint8"))
    img = img.resize((target_shape[1], target_shape[0]))
    return np.asarray(img, dtype="float32") / 255.0


def generate_all_explanations(model, X_test, y_test, results_dir, seed=42, n_per_class=3):
    """Deterministically picks n_per_class Normal + n_per_class Pneumonia test
    samples and generates the full explanation grid + gallery metadata for each.
    """
    rng = np.random.RandomState(seed)
    save_dir = os.path.join(results_dir, "shap_explanations")
    os.makedirs(save_dir, exist_ok=True)

    background_idx = rng.choice(len(X_test), size=min(20, len(X_test)), replace=False)
    background_samples = X_test[background_idx]

    gallery = []
    for label, name in ((0, "normal"), (1, "pneumonia")):
        idx = np.where(y_test == label)[0]
        rng.shuffle(idx)
        chosen = idx[:n_per_class]
        for k, i in enumerate(chosen, start=1):
            save_path = os.path.join(save_dir, f"shap_{name}_{k}.png")
            info = generate_explanation_grid(model, X_test[i], int(y_test[i]),
                                              save_path, background_samples)
            gallery.append(info)
            print(f"  ✅ {os.path.basename(save_path)}")
    return gallery
