# XAI Grad-CAM Comparison

A toolkit for comparing Class Activation Mapping (CAM) methods on **CUB-200-2011**. Supports both supervised CUB classifiers and OpenCLIP zero-shot models. Generates CAM overlays, deletion/insertion curves, bounding-box localization metrics, and summary tables suitable for paper figures.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch, torchvision
- timm
- grad-cam (pytorch-grad-cam)
- open-clip-torch
- huggingface_hub
- numpy, Pillow, opencv-python
- pandas, matplotlib, tqdm

## Project Structure

```text
.
├── run_compare.py          # Run CAM methods, save overlays / curves / metrics.csv
├── metrics.py              # Deletion, insertion, bbox localization, pytorch-grad-cam metric utilities
├── summarize_metrics.py    # Print summary statistics from an output directory
├── requirements.txt
└── README.md
```

## Experiment Settings

| Mode | Model | Setting |
|------|-------|---------|
| `cub_classifier` | CUB-200-2011 classifier, local checkpoint or Hugging Face Hub timm model | Supervised closed-set CUB classifier |
| `open_clip` | OpenCLIP ViT / ResNet | Open-vocabulary zero-shot via CUB class-name text prompts |

**Note:** OpenCLIP zero-shot is not a CUB supervised classifier. These two settings should be reported separately in publications.

## Usage

### CUB Supervised Classifier

#### Option A: Hugging Face Hub timm model

The script supports timm models loaded from Hugging Face Hub via `hf_hub:`.

Example:

```bash
python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode cub_classifier \
  --cub-arch hf_hub:anonauthors/cub200-resnet50 \
  --target gt \
  --num-images 20 \
  --output-dir outputs_cub_resnet50
```

In this mode, `--cub-checkpoint` is not needed.

#### Option B: Local CUB checkpoint

Provide your own checkpoint trained on CUB-200-2011 with 200 output classes.

```bash
python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode cub_classifier \
  --cub-arch resnet50 \
  --cub-checkpoint /path/to/cub_resnet50_checkpoint.pth \
  --target gt \
  --num-images 20 \
  --output-dir outputs_cub_classifier
```

Supported checkpoint formats:

- raw `state_dict`
- dict containing `state_dict`
- dict containing `model`
- dict containing `model_state_dict`

Keys prefixed with `module.` are stripped automatically.

The default preprocessing for `cub_classifier` mode uses ImageNet mean/std:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

If your checkpoint was trained with different normalization, resize, or crop settings, edit `build_cub_transform()` in `run_compare.py`.

### OpenCLIP Zero-Shot

OpenCLIP mode uses CUB class names as text prompts.  
The default prompt template is:

```text
a photo of a {}, a type of bird.
```

Example:

```bash
python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode open_clip \
  --clip-model ViT-B-32 \
  --clip-pretrained openai \
  --target gt \
  --num-images 20 \
  --output-dir outputs_openclip
```

You can change the prompt template:

```bash
python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode open_clip \
  --clip-model ViT-B-32 \
  --clip-pretrained openai \
  --clip-prompt-template "a photo of a bird called {}." \
  --target gt \
  --num-images 20 \
  --output-dir outputs_openclip_prompt2
```

In OpenCLIP mode:

- `--target gt` explains the ground-truth CUB class text.
- `--target pred` explains the zero-shot predicted class.

## Model Weights and Caching

The CUB dataset is **not** downloaded automatically (see [CUB Directory Layout](#cub-directory-layout)).

Model weights may be downloaded automatically in these cases:

- `--cub-arch hf_hub:...` downloads a Hugging Face Hub model through timm.
- `--model-mode open_clip` may download OpenCLIP weights.

By default, Hugging Face models are cached under:

```text
~/.cache/huggingface/hub
```

To keep downloaded model files inside this project, use a local `checkpoints/` directory.

### Bash / Linux / macOS

```bash
mkdir -p checkpoints/hf checkpoints/torch

export HF_HOME=$PWD/checkpoints/hf
export HF_HUB_CACHE=$PWD/checkpoints/hf/hub
export TORCH_HOME=$PWD/checkpoints/torch

python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode cub_classifier \
  --cub-arch hf_hub:anonauthors/cub200-resnet50 \
  --target gt \
  --num-images 20 \
  --output-dir outputs_cub_resnet50
```

### PowerShell / Windows

```powershell
New-Item -ItemType Directory -Force -Path checkpoints\hf, checkpoints\torch | Out-Null

$env:HF_HOME = "$PWD\checkpoints\hf"
$env:HF_HUB_CACHE = "$PWD\checkpoints\hf\hub"
$env:TORCH_HOME = "$PWD\checkpoints\torch"

python run_compare.py `
  --data-root "D:\datasets\CUB_200_2011" `
  --model-mode cub_classifier `
  --cub-arch "hf_hub:anonauthors/cub200-resnet50" `
  --target gt `
  --num-images 20 `
  --output-dir outputs_cub_resnet50
```

For OpenCLIP on PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path checkpoints\hf, checkpoints\torch | Out-Null

$env:HF_HOME = "$PWD\checkpoints\hf"
$env:HF_HUB_CACHE = "$PWD\checkpoints\hf\hub"
$env:TORCH_HOME = "$PWD\checkpoints\torch"

python run_compare.py `
  --data-root "D:\datasets\CUB_200_2011" `
  --model-mode open_clip `
  --clip-model "ViT-B-32" `
  --clip-pretrained "openai" `
  --target gt `
  --num-images 20 `
  --output-dir outputs_openclip
```

These environment variables are valid only in the current terminal session.

## Supported CAM Methods

All methods are selected via `--methods`.

```bash
python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode open_clip \
  --methods gradcam,ablationcam,eigencam,layercam,finercam
```

| CLI name | Class |
|----------|-------|
| `gradcam` | GradCAM |
| `gradcampp` | GradCAMPlusPlus |
| `scorecam` | ScoreCAM |
| `xgradcam` | XGradCAM |
| `eigencam` | EigenCAM |
| `ablationcam` | AblationCAM |
| `layercam` | LayerCAM |
| `finercam` | FinerCAM |

`scorecam` and `ablationcam` are significantly slower than gradient-based methods. For a quick test, use fewer methods and images:

```bash
python run_compare.py \
  --data-root /path/to/CUB_200_2011 \
  --model-mode cub_classifier \
  --cub-arch hf_hub:anonauthors/cub200-resnet50 \
  --methods gradcam,eigencam \
  --num-images 3 \
  --steps 10 \
  --output-dir outputs_smoke_test
```

## CUB Directory Layout

Expected CUB-200-2011 directory structure:

```text
CUB_200_2011/
  images/
  images.txt
  classes.txt
  image_class_labels.txt
  train_test_split.txt
  bounding_boxes.txt
```

If `train_test_split.txt` exists, the script preferentially samples test images (`is_training_image = 0`).

If `bounding_boxes.txt` exists, per-image bounding boxes are loaded and used to compute localization metrics (pointing game, IoU, energy ratio, FP/FN error). When missing, these metrics are reported as `NaN`.

The CUB dataset must be downloaded and extracted manually before running experiments.

## Outputs

```text
outputs/
├── overlays/          # per-image, per-method CAM overlays
├── grids/             # horizontal comparison grids
├── curves/            # deletion / insertion curves
└── metrics.csv        # per-image, per-method metrics
```

Example overlay files:

```text
outputs/overlays/image_000_gradcam.jpg
outputs/overlays/image_000_scorecam.jpg
outputs/overlays/image_001_layercam.jpg
```

Example grid file:

```text
outputs/grids/image_000_grid.jpg
```

Example curve file:

```text
outputs/curves/image_000_gradcam_curves.png
```

## Metrics

### `metrics.csv` Columns

#### Image and model

- `image_id`
- `image_path`
- `model_mode`
- `method`

#### Labels and predictions

- `gt_class_id`
- `gt_class_name`
- `pred_class_id`
- `pred_class_name`
- `pred_confidence`

#### Explanation target

- `target_class_id`
- `target_class_name`
- `target_mode`

`target_mode` is either:

- `gt`: explain the CUB ground-truth class
- `pred`: explain the model predicted class

If `--target gt` is requested but ground-truth labels are missing, the script falls back to `pred` with a warning.

#### Bounding box

- `bbox_xywh` — bounding box in original image coordinates `[x, y, w, h]`
- `has_bbox` — whether a bounding box was available for this image

#### Metric values

- `deletion_auc`
- `insertion_auc`
- `aopc`
- `confidence_drop`
- `confidence_increase`
- `road_combined`
- `pointing_game`
- `bbox_iou_top20pct`
- `bbox_energy_ratio`
- `fp_error_top20pct`
- `fn_error_top20pct`
- `runtime_sec`
- `error`

If a CAM method fails on one image, the script records the error in `metrics.csv` and continues with the remaining methods/images.

### Summarizing Results

After running `run_compare.py`, use `summarize_metrics.py` to generate summary statistics:

```bash
python summarize_metrics.py outputs_cub_resnet50
```

The script reads `metrics.csv` from the specified output directory and prints:

- Basic information and row counts (valid / failed)
- Failure summary
- Overall metric statistics
- Grouped summary table (trimmed mean: drops one min and one max before aggregating)
- Ranking by metric mean

#### Group by method

Default grouping is by `method`:

```bash
python summarize_metrics.py outputs_cub_resnet50 --group-by method
```

#### Group by multiple columns

For example, group by method and target mode:

```bash
python summarize_metrics.py outputs_cub_resnet50 --group-by method,target_mode
```

For mixed experiments, group by model mode and method:

```bash
python summarize_metrics.py outputs_mixed --group-by model_mode,method
```

#### Sort by a metric

Sort by `insertion_auc`:

```bash
python summarize_metrics.py outputs_cub_resnet50 --sort-by insertion_auc
```

Sort by `deletion_auc`:

```bash
python summarize_metrics.py outputs_cub_resnet50 --sort-by deletion_auc
```

Sorting direction is chosen automatically for common metrics:

| Metric | Sorting direction |
|--------|-------------------|
| `deletion_auc` | ascending |
| `fp_error_top20pct` | ascending |
| `fn_error_top20pct` | ascending |
| `runtime_sec` | ascending |
| `insertion_auc` | descending |
| `aopc` | descending |
| `confidence_drop` | descending |
| `confidence_increase` | descending |
| `road_combined` | descending |
| `pointing_game` | descending |
| `bbox_iou_top20pct` | descending |
| `bbox_energy_ratio` | descending |

Other metrics should be interpreted with care.

#### Save summary as CSV

```bash
python summarize_metrics.py outputs_cub_resnet50 \
  --save-csv outputs_cub_resnet50/summary.csv
```

This saves the full grouped summary, including mean, standard deviation, median, min, max, and valid count for each metric.

#### Include failed rows

By default, rows with non-empty `error` are excluded from metric summaries.

To include them:

```bash
python summarize_metrics.py outputs_cub_resnet50 --include-failed
```

Usually, failed rows have `NaN` metric values, so they may still be ignored by pandas aggregation.  
Use this option mainly when debugging the output table.

### Metric Interpretation

#### Perturbation-based metrics

| Metric | Direction | Notes |
|--------|-----------|-------|
| `deletion_auc` | lower is often better | Removing high-attribution pixels should drop target confidence quickly |
| `insertion_auc` | higher is often better | Restoring high-attribution pixels should recover target confidence quickly |
| `aopc` | higher is often better | Average confidence drop after progressively deleting high-attribution pixels |
| `road_combined` | higher is often better | ROAD combined score from pytorch-grad-cam |
| `confidence_drop` | higher means more faithful | Drop in target confidence after CAM-based masking |
| `confidence_increase` | see pytorch-grad-cam definition | Whether target confidence increases after CAM-based masking |
| `runtime_sec` | lower is faster | Runtime per image-method pair |

The implemented deletion/insertion baseline is zero in normalized tensor space.  
For ImageNet-style normalization, this is equivalent to a mean-color image.  
Report this choice if using the numbers in a paper.

#### Bounding-box localization metrics

These metrics require `bounding_boxes.txt` in the CUB dataset directory. For OpenCLIP mode, the bounding box is automatically transformed through the preprocess pipeline (Resize + CenterCrop) before evaluation.

| Metric | Direction | Notes |
|--------|-----------|-------|
| `pointing_game` | higher is better | 1 if the max-CAM point falls inside the ground-truth bbox, else 0 |
| `bbox_iou_top20pct` | higher is better | IoU between the ground-truth bbox mask and the CAM top-20% mask |
| `bbox_energy_ratio` | higher is better | Fraction of total CAM energy that falls inside the ground-truth bbox |
| `fp_error_top20pct` | lower is better | Fraction of CAM top-20% pixels that fall outside the bbox |
| `fn_error_top20pct` | lower is better | Fraction of bbox pixels not covered by the CAM top-20% mask |

## CLI Reference

### `run_compare.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | required | CUB_200_2011 root directory |
| `--output-dir` | `outputs` | Output directory |
| `--num-images` | `20` | Number of images to process |
| `--device` | `auto` | `auto`, `cuda`, `cpu`, or `mps` |
| `--model-mode` | `open_clip` | `cub_classifier` or `open_clip` |
| `--cub-arch` | `resnet50` | timm model name, or `hf_hub:...` |
| `--cub-checkpoint` | `None` | Local CUB checkpoint path; not needed for `hf_hub:` models |
| `--clip-model` | `ViT-B-32` | OpenCLIP model name |
| `--clip-pretrained` | `openai` | OpenCLIP pretrained weights name |
| `--clip-prompt-template` | `a photo of a {}, a type of bird.` | Prompt template for CUB class names |
| `--target` | `gt` | `gt` or `pred` |
| `--methods` | `gradcam,ablationcam,eigencam,layercam,finercam` | CAM methods to run |
| `--steps` | `20` | Deletion/insertion curve steps |
| `--seed` | `42` | Random seed |
| `--image-size` | `224` | Input size; used by CUB classifier and requested for OpenCLIP if supported |

### `summarize_metrics.py`

| Flag | Default | Description |
|------|---------|-------------|
| `output_dir` | required | Output directory containing `metrics.csv` |
| `--group-by` | `method` | Comma-separated grouping columns |
| `--sort-by` | empty | Optional metric used to sort the summary table |
| `--include-failed` | disabled | Include rows with non-empty `error` field |
| `--save-csv` | `None` | Optional path to save summary table as CSV |
