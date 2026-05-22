#!/usr/bin/env python3
"""Minimal CAM method comparison on CUB-200-2011."""

from __future__ import annotations

import argparse
import random
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
from PIL import Image
from tqdm import tqdm

from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from metrics import (
    compute_gradcam_builtin_metrics,
    deletion_insertion_auc,
    maybe_resize_cam,
    normalize_cam,
    plot_curves,
    predict_softmax,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_METHODS = "gradcam,gradcampp,xgradcam,eigencam,layercam"

CAM_METHOD_MAP = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "ablationcam": AblationCAM,
    "layercam": LayerCAM,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CAM methods on CUB-200-2011")
    parser.add_argument("--data-root", type=Path, required=True, help="CUB_200_2011 root directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--model-mode", type=str, default="open_clip", choices=["cub_classifier", "open_clip"])
    parser.add_argument("--cub-arch", type=str, default="resnet50")
    parser.add_argument("--cub-checkpoint", type=Path, default=None)
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-prompt-template", type=str, default="a photo of a {}, a type of bird.")
    parser.add_argument("--target", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_class_name(raw_name: str) -> str:
    name = re.sub(r"^\d+\.", "", raw_name.strip())
    name = name.replace("_", " ")
    return name.lower()


def load_cub_metadata(data_root: Path):
    images_path = data_root / "images.txt"
    labels_path = data_root / "image_class_labels.txt"
    classes_path = data_root / "classes.txt"
    split_path = data_root / "train_test_split.txt"

    id_to_path = {}
    with open(images_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_path[int(parts[0])] = parts[1]

    id_to_class = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_to_class[int(parts[0])] = int(parts[1]) - 1

    class_id_to_name = {}
    class_id_to_raw = {}
    with open(classes_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                cid = int(parts[0]) - 1
                class_id_to_raw[cid] = parts[1]
                class_id_to_name[cid] = format_class_name(parts[1])

    test_ids = None
    if split_path.exists():
        test_ids = set()
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[1]) == 0:
                    test_ids.add(int(parts[0]))

    return id_to_path, id_to_class, class_id_to_name, class_id_to_raw, test_ids


def sample_cub_images(data_root: Path, num_images: int, seed: int):
    id_to_path, id_to_class, class_id_to_name, _, test_ids = load_cub_metadata(data_root)

    if test_ids is not None:
        candidate_ids = sorted(test_ids)
    else:
        candidate_ids = sorted(id_to_path.keys())

    rng = random.Random(seed)
    rng.shuffle(candidate_ids)
    selected_ids = candidate_ids[:num_images]

    samples = []
    for image_id in selected_ids:
        rel_path = id_to_path[image_id]
        gt_class_id = id_to_class.get(image_id)
        gt_class_name = class_id_to_name.get(gt_class_id, "") if gt_class_id is not None else ""
        samples.append(
            {
                "image_id": image_id,
                "image_path": data_root / "images" / rel_path,
                "gt_class_id": gt_class_id,
                "gt_class_name": gt_class_name,
            }
        )
    return samples, class_id_to_name


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def load_checkpoint_state_dict(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return strip_module_prefix(ckpt)


def _looks_like_vit_tokens_with_cls(n: int) -> bool:
    if n <= 1:
        return False
    side = int(round((n - 1) ** 0.5))
    return side * side == n - 1


def vit_reshape_transform(tensor, height=None, width=None):
    """Reshape ViT tokens to [B, C, H, W]. Supports [B, N, C] and [N, B, C]."""
    if tensor.ndim == 4:
        return tensor

    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D or 4D activation, got shape={tuple(tensor.shape)}")

    # Some OpenCLIP visual transformers use [N, B, C] instead of [B, N, C].
    if _looks_like_vit_tokens_with_cls(tensor.shape[0]) and not _looks_like_vit_tokens_with_cls(tensor.shape[1]):
        tensor = tensor.permute(1, 0, 2)

    b, n, c = tensor.shape
    if not _looks_like_vit_tokens_with_cls(n):
        raise ValueError(f"Token grid is not square after removing cls token: shape={tuple(tensor.shape)}")

    tokens = tensor[:, 1:, :]
    side = int(round((n - 1) ** 0.5))
    return tokens.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()


def get_target_layer_cub(model: nn.Module):
    if hasattr(model, "layer4"):
        return [model.layer4[-1]], None

    if hasattr(model, "blocks"):
        block = model.blocks[-1]
        if hasattr(block, "norm1"):
            return [block.norm1], vit_reshape_transform
        return [block], vit_reshape_transform

    raise RuntimeError(
        "Could not auto-select target layer for CUB classifier. "
        "Edit get_target_layer_cub() in run_compare.py."
    )


def get_target_layer_clip(visual: nn.Module):
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        block = visual.transformer.resblocks[-1]
        if hasattr(block, "ln_1"):
            return [block.ln_1], vit_reshape_transform
        return [block], vit_reshape_transform

    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        block = visual.trunk.blocks[-1]
        if hasattr(block, "norm1"):
            return [block.norm1], vit_reshape_transform
        return [block], vit_reshape_transform

    if hasattr(visual, "layer4"):
        return [visual.layer4[-1]], None

    if hasattr(visual, "trunk") and hasattr(visual.trunk, "stages"):
        stages = visual.trunk.stages
        if len(stages) > 0 and hasattr(stages[-1], "blocks") and len(stages[-1].blocks) > 0:
            return [stages[-1].blocks[-1]], None

    if hasattr(visual, "stages") and len(visual.stages) > 0:
        stage = visual.stages[-1]
        if hasattr(stage, "blocks") and len(stage.blocks) > 0:
            return [stage.blocks[-1]], None

    raise RuntimeError(
        "Could not auto-select target layer for OpenCLIP visual encoder. "
        "Edit get_target_layer_clip() in run_compare.py."
    )


class CLIPZeroShotWrapper(nn.Module):
    """Wrap OpenCLIP for 200-way CUB zero-shot logits compatible with grad-cam."""

    def __init__(self, clip_model, text_features: torch.Tensor):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_features", text_features.detach())

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)
        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.T


def build_cub_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_cub_classifier(arch: str, checkpoint: Path | None, device: torch.device, image_size: int):
    """
    Build a CUB classifier.

    If arch starts with 'hf_hub:' and checkpoint is None, timm will load the
    pretrained weights from Hugging Face Hub, e.g.
      --cub-arch hf_hub:anonauthors/cub200-resnet50
    Otherwise, a local checkpoint is required.
    """
    if arch.startswith("hf_hub:") and checkpoint is None:
        model = timm.create_model(arch, pretrained=True)
    else:
        if checkpoint is None:
            raise ValueError("--cub-checkpoint is required unless --cub-arch starts with 'hf_hub:'")
        model = timm.create_model(arch, pretrained=False, num_classes=200)
        state_dict = load_checkpoint_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")

    model = model.to(device).eval()
    target_layers, reshape_transform = get_target_layer_cub(model)
    transform = build_cub_transform(image_size)
    return model, target_layers, reshape_transform, transform


def build_open_clip_model(
    clip_model: str,
    clip_pretrained: str,
    prompt_template: str,
    class_id_to_name: dict,
    device: torch.device,
    image_size: int,
):
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
            force_image_size=image_size,
        )
    except TypeError:
        warnings.warn(
            "This open_clip version does not support force_image_size; "
            "using the model default preprocess size."
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
        )

    tokenizer = open_clip.get_tokenizer(clip_model)

    prompts = [prompt_template.format(class_id_to_name[i]) for i in range(200)]
    model = model.to(device).eval()

    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    wrapper = CLIPZeroShotWrapper(model, text_features).to(device).eval()
    target_layers, reshape_transform = get_target_layer_clip(model.visual)

    return wrapper, target_layers, reshape_transform, preprocess


def preprocess_clip_image_with_rgb(pil_image: Image.Image, preprocess, device: torch.device):
    """
    Apply OpenCLIP preprocess and also recover the RGB image after the same
    geometric transforms but before Normalize, so CAM overlay matches model input.
    """
    if hasattr(preprocess, "transforms"):
        x = pil_image
        rgb_tensor = None

        for transform in preprocess.transforms:
            if transform.__class__.__name__ == "Normalize":
                if not torch.is_tensor(x):
                    raise TypeError("Expected tensor before Normalize in OpenCLIP preprocess")
                rgb_tensor = x.detach().clone()
            x = transform(x)

        if not torch.is_tensor(x):
            raise TypeError("OpenCLIP preprocess did not return a tensor")

        if rgb_tensor is None:
            rgb_tensor = x.detach().clone()

        rgb_img = rgb_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_img = np.float32(np.clip(rgb_img, 0.0, 1.0))
        return x.unsqueeze(0).to(device), rgb_img

    # Fallback for unusual preprocess objects.
    input_tensor = preprocess(pil_image)
    h, w = input_tensor.shape[-2:]
    rgb_img = np.float32(pil_image.resize((w, h), Image.BILINEAR)) / 255.0
    return input_tensor.unsqueeze(0).to(device), rgb_img


def resolve_target(sample, pred_class_id, target_mode: str):
    gt_available = sample["gt_class_id"] is not None
    effective_mode = target_mode

    if target_mode == "gt" and not gt_available:
        warnings.warn(
            f"Ground truth unavailable for image_id={sample['image_id']}; falling back to pred."
        )
        effective_mode = "pred"

    if effective_mode == "gt":
        target_class_id = int(sample["gt_class_id"])
        target_class_name = sample["gt_class_name"]
    else:
        target_class_id = int(pred_class_id)
        target_class_name = ""

    return target_class_id, target_class_name, effective_mode


def save_grid(original_rgb, overlay_map, save_path: Path):
    panels = [np.float32(np.clip(original_rgb, 0.0, 1.0))]
    for method in sorted(overlay_map.keys()):
        panels.append(np.float32(np.clip(overlay_map[method], 0.0, 1.0)))

    grid = np.concatenate(panels, axis=1)
    grid_uint8 = np.uint8(grid * 255)
    grid_bgr = cv2.cvtColor(grid_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), grid_bgr)


def run_method(
    method_name,
    cam_cls,
    model,
    input_tensor,
    target_layers,
    reshape_transform,
    target_class_id,
    rgb_img,
    overlay_path,
    curve_path,
    steps,
):
    cam_kwargs = {"model": model, "target_layers": target_layers}
    if reshape_transform is not None:
        cam_kwargs["reshape_transform"] = reshape_transform

    if method_name == "ablationcam" and reshape_transform is not None:
        cam_kwargs["ablation_layer"] = AblationLayerVit()

    targets = [ClassifierOutputTarget(int(target_class_id))]

    with cam_cls(**cam_kwargs) as cam:
        if method_name in ("scorecam", "ablationcam"):
            cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

    cam_np = normalize_cam(grayscale_cam)
    cam_np = maybe_resize_cam(cam_np, rgb_img.shape[:2])

    overlay_uint8 = show_cam_on_image(rgb_img, cam_np, use_rgb=True)
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    deletion_auc, deletion_curve = deletion_insertion_auc(
        model, input_tensor, cam_np, target_class_id, steps=steps, mode="deletion"
    )
    insertion_auc, insertion_curve = deletion_insertion_auc(
        model, input_tensor, cam_np, target_class_id, steps=steps, mode="insertion"
    )
    plot_curves(deletion_curve, insertion_curve, curve_path)

    builtin = compute_gradcam_builtin_metrics(
        model, input_tensor, cam_np[np.newaxis, ...], target_class_id
    )

    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        **builtin,
        "overlay_rgb": overlay_uint8.astype(np.float32) / 255.0,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in CAM_METHOD_MAP]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {sorted(CAM_METHOD_MAP)}")

    output_dir = args.output_dir
    overlay_dir = output_dir / "overlays"
    grid_dir = output_dir / "grids"
    curve_dir = output_dir / "curves"
    for d in (overlay_dir, grid_dir, curve_dir):
        d.mkdir(parents=True, exist_ok=True)

    samples, class_id_to_name = sample_cub_images(args.data_root, args.num_images, args.seed)

    if args.model_mode == "cub_classifier":
        model, target_layers, reshape_transform, transform_fn = build_cub_classifier(
            args.cub_arch, args.cub_checkpoint, device, args.image_size
        )
    else:
        model, target_layers, reshape_transform, transform_fn = build_open_clip_model(
            args.clip_model,
            args.clip_pretrained,
            args.clip_prompt_template,
            class_id_to_name,
            device,
            args.image_size,
        )

    rows = []

    for idx, sample in enumerate(tqdm(samples, desc="Images")):
        tag = f"image_{idx:03d}"
        pil_image = Image.open(sample["image_path"]).convert("RGB")

        if args.model_mode == "cub_classifier":
            pil_resized = pil_image.resize((args.image_size, args.image_size), Image.BILINEAR)
            input_tensor = transform_fn(pil_resized).unsqueeze(0).to(device)
            rgb_img = np.float32(pil_resized) / 255.0
        else:
            input_tensor, rgb_img = preprocess_clip_image_with_rgb(pil_image, transform_fn, device)

        _, pred_class_id, pred_confidence = predict_softmax(model, input_tensor)
        pred_class_name = class_id_to_name.get(pred_class_id, "")

        target_class_id, target_class_name, effective_target_mode = resolve_target(
            sample, pred_class_id, args.target
        )
        if effective_target_mode == "pred" and not target_class_name:
            target_class_name = pred_class_name

        overlay_map = {}

        for method_name in methods:
            row = {
                "image_id": sample["image_id"],
                "image_path": str(sample["image_path"]),
                "model_mode": args.model_mode,
                "method": method_name,
                "gt_class_id": sample["gt_class_id"],
                "gt_class_name": sample["gt_class_name"],
                "pred_class_id": pred_class_id,
                "pred_class_name": pred_class_name,
                "pred_confidence": pred_confidence,
                "target_class_id": target_class_id,
                "target_class_name": target_class_name,
                "target_mode": effective_target_mode,
                "deletion_auc": float("nan"),
                "insertion_auc": float("nan"),
                "confidence_drop": float("nan"),
                "confidence_increase": float("nan"),
                "road_combined": float("nan"),
                "runtime_sec": float("nan"),
                "error": "",
            }

            overlay_path = overlay_dir / f"{tag}_{method_name}.jpg"
            curve_path = curve_dir / f"{tag}_{method_name}_curves.png"

            t0 = time.perf_counter()
            try:
                metrics = run_method(
                    method_name,
                    CAM_METHOD_MAP[method_name],
                    model,
                    input_tensor,
                    target_layers,
                    reshape_transform,
                    target_class_id,
                    rgb_img,
                    overlay_path,
                    curve_path,
                    args.steps,
                )
                row.update(
                    {
                        "deletion_auc": metrics["deletion_auc"],
                        "insertion_auc": metrics["insertion_auc"],
                        "confidence_drop": metrics["confidence_drop"],
                        "confidence_increase": metrics["confidence_increase"],
                        "road_combined": metrics["road_combined"],
                    }
                )
                overlay_map[method_name] = metrics["overlay_rgb"]
            except Exception as exc:
                row["error"] = repr(exc)

            row["runtime_sec"] = time.perf_counter() - t0
            rows.append(row)

        save_grid(rgb_img, overlay_map, grid_dir / f"{tag}_grid.jpg")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"Done. Saved results to {output_dir}")


if __name__ == "__main__":
    main()
