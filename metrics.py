"""Metrics and utilities for CAM evaluation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def _as_logits(output):
    """Handle common model outputs. The normal case is a tensor."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
    if isinstance(output, dict):
        for key in ("logits", "pred", "output"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Could not extract logits from model output type: {type(output)}")


def _as_float(value):
    """Convert tensor/list/numpy scalar-like values to Python float."""
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


def predict_softmax(model, input_tensor):
    """Return softmax probabilities, top-1 class id, and top-1 confidence."""
    with torch.no_grad():
        logits = _as_logits(model(input_tensor))
        probs = torch.softmax(logits, dim=1)
        confidence, pred_id = probs.max(dim=1)
    return probs, int(pred_id.item()), float(confidence.item())


def normalize_cam(cam):
    """Normalize CAM to [0, 1], avoiding division by zero."""
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 1e-8:
        cam = cam / denom
    return cam


def maybe_resize_cam(cam, size_hw):
    """Resize CAM to (H, W) if needed."""
    cam = np.asarray(cam, dtype=np.float32)
    h, w = size_hw
    if cam.shape[0] == h and cam.shape[1] == w:
        return cam
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)


def trapezoid_auc(y, x):
    """Compatible trapezoidal integration for different NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def deletion_insertion_auc(
    model,
    input_tensor,
    cam,
    target_category,
    steps=20,
    mode="deletion",
):
    """
    Compute deletion or insertion AUC for a target class.

    Deletion:
        progressively replace high-CAM pixels with baseline.

    Insertion:
        progressively restore high-CAM pixels from baseline.

    Baseline:
        zeros in normalized tensor space, equivalent to the mean image if the
        input was normalized by mean/std.
    """
    if mode not in ("deletion", "insertion"):
        raise ValueError(f"mode must be 'deletion' or 'insertion', got {mode!r}")

    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(f"Expected input_tensor shape [1, 3, H, W], got {tuple(input_tensor.shape)}")

    target_category = int(target_category)

    _, _, h, w = input_tensor.shape
    cam = maybe_resize_cam(normalize_cam(cam), (h, w))

    flat_cam = cam.reshape(-1)
    order_np = np.argsort(-flat_cam)
    order = torch.as_tensor(order_np, device=input_tensor.device, dtype=torch.long)

    total_pixels = h * w
    baseline = torch.zeros_like(input_tensor)
    original = input_tensor.detach().clone()
    current = original.clone() if mode == "deletion" else baseline.clone()

    curve = []
    x_axis = np.linspace(0.0, 1.0, steps + 1)

    pixels_per_step = max(1, total_pixels // steps)
    flat_baseline = baseline.reshape(1, 3, -1)
    flat_original = original.reshape(1, 3, -1)

    with torch.no_grad():
        logits = _as_logits(model(current))
        prob = torch.softmax(logits, dim=1)[0, target_category].item()
        curve.append(prob)

        for step in range(1, steps + 1):
            start = (step - 1) * pixels_per_step
            end = step * pixels_per_step if step < steps else total_pixels
            idx = order[start:end]

            flat_current = current.reshape(1, 3, -1)

            if mode == "deletion":
                flat_current[0, :, idx] = flat_baseline[0, :, idx]
            else:
                flat_current[0, :, idx] = flat_original[0, :, idx]

            logits = _as_logits(model(current))
            prob = torch.softmax(logits, dim=1)[0, target_category].item()
            curve.append(prob)

    auc = trapezoid_auc(curve, x_axis)
    return auc, np.array(curve, dtype=np.float32)


def compute_gradcam_builtin_metrics(model, input_tensor, cam_batch, target_category):
    """Use pytorch-grad-cam built-in confidence and ROAD metrics."""
    from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
    from pytorch_grad_cam.metrics.road import ROADCombined
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

    targets = [ClassifierOutputSoftmaxTarget(int(target_category))]
    result = {
        "confidence_drop": float("nan"),
        "confidence_increase": float("nan"),
        "road_combined": float("nan"),
    }

    if cam_batch.ndim == 2:
        cam_batch = cam_batch[np.newaxis, ...]

    try:
        scores = DropInConfidence()(input_tensor, cam_batch, targets, model)
        result["confidence_drop"] = _as_float(scores)
    except Exception:
        pass

    try:
        scores = IncreaseInConfidence()(input_tensor, cam_batch, targets, model)
        result["confidence_increase"] = _as_float(scores)
    except Exception:
        pass

    try:
        road = ROADCombined(percentiles=[20, 40, 60, 80])
        scores = road(input_tensor, cam_batch, targets, model)
        result["road_combined"] = _as_float(scores)
    except Exception:
        pass

    return result


def plot_curves(deletion_curve, insertion_curve, save_path):
    """Save deletion and insertion confidence curves."""
    x = np.linspace(0.0, 1.0, len(deletion_curve))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, deletion_curve, label="deletion")
    ax.plot(x, insertion_curve, label="insertion")
    ax.set_xlabel("Fraction of pixels modified")
    ax.set_ylabel("Target class confidence")
    ax.set_title("Deletion / Insertion curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)