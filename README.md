# NNCV Final Assignment

This repository contains the final assignment for semantic segmentation on the **Cityscapes** dataset, with experiments on both:

- **Peak Performance**
- **Out-of-Distribution (OOD) Detection**

The project starts from the provided **U-Net baseline** and improves the segmentation pipeline with **DeepLabV3+** using pretrained **ResNet50** and **ResNet101** encoders. In addition, a lightweight **post-hoc OOD rejection** mechanism is implemented using uncertainty-based scores such as **maximum softmax probability (MSP)** and **percentile entropy**.

## Repository structure

```text
final_assignment/
├── configs/                  # YAML experiment configurations
├── scripts/                  # Slurm / utility scripts
│   └── plots/                # Figure generation scripts
├── src/
│   ├── models/               # Segmentation models
│   ├── ood/                  # OOD score utilities
│   └── ...                   # Data loading, metrics, engine, config utilities
├── checkpoints/              # Saved checkpoints (not all should be versioned)
├── evaluation_outputs/       # Saved evaluation masks / metrics
├── ood_outputs/              # OOD scoring outputs
├── figures/                  # Final report figures
├── train.py                  # Main training entry point
├── evaluate.py               # Segmentation evaluation entry point
├── ood_evaluate.py           # OOD scoring on Cityscapes validation
├── ood_score_folder.py       # OOD scoring on a plain image folder (e.g. COCO)
├── analyze_ood_thresholds.py # Threshold analysis on ID validation scores
├── compare_ood_rejection.py  # Proxy OOD rejection analysis
├── predict.py                # Peak-performance submission inference script
├── predict_ood.py            # OOD submission inference script
└── Dockerfile                # Submission container
```

## Environment setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install torch torchvision
pip install pillow numpy pandas matplotlib scikit-learn segmentation-models-pytorch
```

For experiment tracking:

```bash
pip install wandb
```

## Data

### Cityscapes

```text
data/cityscapes/
├── leftImg8bit/
│ ├── train/
│ └── val/
└── gtFine/
├── train/
└── val/
```

### COCO proxy OOD

```text
data/coco_ood/
└── val2017/
```

## Training

Training is controlled through YAML configuration files in `configs/`

Example local command:

```bash
python train.py --config configs/peak_deeplabv3plus_r101_tuned.yaml
```

Snellius training was done though the scripts in `scripts/`

## Evaluation

### Segmentation evaluation

To evaluate a trained checkpoint on the Cityscapes validation set:

```bash
python evaluate.py \
  --config configs/peak_deeplabv3plus_r101_tuned.yaml \
  --checkpoint checkpoints/peak-deeplabv3plus-r101-tuned/best_model_dice.pt \
  --save-dir evaluation_outputs/r101_tuned \
  --num-samples 8
```

This saves summary metrics, selected prediction and ground-truth masks.

### OOD evaluation on Cityscapes validation

```bash
python ood_evaluate.py \
  --config configs/peak_deeplabv3plus_r101_tuned.yaml \
  --checkpoint checkpoints/peak-deeplabv3plus-r101-tuned/best_model_dice.pt \
  --save-dir ood_outputs/r101_tuned \
  --num-samples 8
```

This generates a `ood_scores.csv`, summary file and optional qualitative masks.

### OOD scoring on a folder of images

```bash
python ood_score_folder.py \
  --checkpoint checkpoints/peak-deeplabv3plus-r101-tuned/best_model_dice.pt \
  --encoder-name resnet101 \
  --image-dir data/coco_ood/val2017 \
  --save-dir ood_outputs/coco_r101 \
  --num-samples 0
```

## Threshold analysis

Candidat thresholds were selected from score quantiles on the Cityscapes validation set:

```bash
python analyze_ood_thresholds.py --csv ood_outputs/r101_tuned/ood_scores.csv
```

This prints candidate thresholds and corresponding in-distribution rejection rates.
To compare rejection rates on proxy OOD data:

```bash
python compare_ood_rejection.py --csv ood_outputs/coco_r101/ood_scores.csv
```

## Plot generation

The figures used in the report can be reproduced with scripts in `scripts/plots/`

1. Segmentation qualitative figure:

```bash
python scripts/plots/make_segmentation_comparison_figure.py \
  --gt evaluation_outputs/run_r101_tuned_dice/sample_00_label.png \
  --pred evaluation_outputs/run_r101_tuned_dice/sample_00_pred.png \
  --output figures/segmentation_r101_vs_gt.png
```

2. OOD distribution figure

```bash
python scripts/plots/make_ood_distribution_figure.py \
  --id-csv ood_outputs/r101_tuned/ood_scores.csv \
  --ood-csv ood_outputs/coco_r101/ood_scores.csv \
  --score mean_msp \
  --threshold 0.888828 \
  --output figures/ood_msp_distribution.png \
  --title "Mean MSP distributions for ID and proxy OOD"
```

3. OOD qualitative example figure

```bash
python scripts/plots/make_ood_example_figure.py \
  --id-image local_test/data/zurich_000000_000019_leftImg8bit.png \
  --ood-image figures/coco_example.jpg \
  --id-score 0.9180 \
  --ood-score 0.7420 \
  --score-name mean_msp \
  --output figures/ood_example.png \
  --title "Example include/exclude decisions for the MSP-based OOD detector"
```

### Submission preparation

The peak benchmark uses `predict.py`, `model.py`, and `Dockerfile`.
Build the container:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile \
  -t nncv-submission:peak \
  --load .
```

Run locally:

```bash
docker run --rm \
  --network none \
  --platform linux/amd64 \
  -v "$(pwd)/local_test/data:/data" \
  -v "$(pwd)/local_test/output:/output" \
  nncv-submission:peak
```

Export to tar:
docker save nncv-submission:peak | gzip > submission_peak.tar.gz

OOD submission utilizes similar approach but uses `predict_ood.py` instead of `predict.py`.

## Reproducibility notes

- Final peak and OOD benchmark numbers come from the **challenge submission server**
- Validation metrics were used only for model selection and threshold calibration
- Training and large-scale evaluation were run on **Snellius**
- Local scripts in this repository reproduce:
  - evaluation
  - OOD threshold analysis
  - plotting
  - Docker-based submission tests

Note: final training checkpoints are not stored directly in this forked repository due to storage and Git LFS limitations on public forks. The repository contains the code, configuration files, evaluation scripts, and submission pipeline used to reproduce the reported results.

## Acknowledgment

This work was carried out as part of the NNCV final assignment. The author would like to thank the course team for providing the challenge framework, baseline code, and evaluation infrastructure. The author is also grateful for access to the Snellius high-performance computing resources used for model training and evaluation.
