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


def aopc_from_deletion_curve(deletion_curve):
    """
    AOPC from deletion curve.

    Larger AOPC means the target confidence drops more after deleting
    high-attribution pixels.
    """
    curve = np.asarray(deletion_curve, dtype=np.float32)
    if curve.size <= 1:
        return float("nan")
    return float(np.mean(curve[0] - curve[1:]))


def _empty_bbox_metrics():
    return {
        "pointing_game": float("nan"),
        "bbox_iou_top20pct": float("nan"),
        "bbox_energy_ratio": float("nan"),
        "fp_error_top20pct": float("nan"),
        "fn_error_top20pct": float("nan"),
    }


def compute_bbox_localization_metrics(
    cam,
    bbox_xywh,
    orig_size_wh,
    top_ratio=0.2,
):
    """
    Compute weak localization metrics using a CUB bounding box.

    Args:
        cam:
            CAM heatmap, shape [H, W].
        bbox_xywh:
            Bounding box in the coordinate frame of orig_size_wh:
            [x, y, width, height].
        orig_size_wh:
            Original image size as [width, height].
            If bbox has already been transformed into CAM/input coordinates,
            pass orig_size_wh as [cam_width, cam_height].
        top_ratio:
            Top attribution ratio used to binarize CAM. Default 0.2 means
            selecting the top 20% highest-CAM pixels.

    Returns:
        Dict with:
            pointing_game:
                1 if max-CAM point falls inside bbox, else 0.
            bbox_iou_top20pct:
                IoU between bbox mask and CAM top-20% mask.
            bbox_energy_ratio:
                sum(CAM inside bbox) / sum(CAM over the whole image).
            fp_error_top20pct:
                fraction of CAM top-20% pixels outside bbox.
            fn_error_top20pct:
                fraction of bbox pixels not covered by CAM top-20% mask.
    """
    result = _empty_bbox_metrics()

    if bbox_xywh is None or orig_size_wh is None:
        return result

    cam = normalize_cam(cam)
    if cam.ndim != 2:
        return result

    h, w = cam.shape
    if h <= 0 or w <= 0:
        return result

    try:
        orig_w, orig_h = float(orig_size_wh[0]), float(orig_size_wh[1])
        x, y, bw, bh = [float(v) for v in bbox_xywh]
    except Exception:
        return result

    if orig_w <= 0 or orig_h <= 0 or bw <= 0 or bh <= 0:
        return result

    # Scale bbox from orig_size_wh coordinate frame to CAM coordinate frame.
    x1 = x / orig_w * w
    y1 = y / orig_h * h
    x2 = (x + bw) / orig_w * w
    y2 = (y + bh) / orig_h * h

    xi1 = int(np.floor(max(0.0, min(float(w), x1))))
    yi1 = int(np.floor(max(0.0, min(float(h), y1))))
    xi2 = int(np.ceil(max(0.0, min(float(w), x2))))
    yi2 = int(np.ceil(max(0.0, min(float(h), y2))))

    if xi2 <= xi1 or yi2 <= yi1:
        return result

    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[yi1:yi2, xi1:xi2] = True
    bbox_area = int(bbox_mask.sum())
    if bbox_area <= 0:
        return result

    # Pointing Game: whether max attribution point lies inside bbox.
    max_y, max_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    result["pointing_game"] = float(bbox_mask[max_y, max_x])

    # Energy ratio: continuous metric, no threshold.
    cam_sum = float(cam.sum())
    if cam_sum > 1e-8:
        result["bbox_energy_ratio"] = float(cam[bbox_mask].sum() / cam_sum)

    # CAM top-k mask.
    flat_cam = cam.reshape(-1)
    total_pixels = flat_cam.size
    top_ratio = float(np.clip(top_ratio, 1e-6, 1.0))
    k = max(1, int(np.ceil(total_pixels * top_ratio)))

    top_indices = np.argpartition(flat_cam, -k)[-k:]
    top_mask_flat = np.zeros(total_pixels, dtype=bool)
    top_mask_flat[top_indices] = True
    top_mask = top_mask_flat.reshape(h, w)

    top_area = int(top_mask.sum())
    if top_area <= 0:
        return result

    intersection = int(np.logical_and(top_mask, bbox_mask).sum())
    union = int(np.logical_or(top_mask, bbox_mask).sum())

    if union > 0:
        result["bbox_iou_top20pct"] = float(intersection / union)

    result["fp_error_top20pct"] = float((top_area - intersection) / top_area)
    result["fn_error_top20pct"] = float((bbox_area - intersection) / bbox_area)

    return result


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
