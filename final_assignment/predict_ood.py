from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from src.models.deeplabv3plus import Model
from src.ood.scores import image_mean_msp, image_percentile_entropy
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
IMAGE_SIZE = (512, 1024)

# ----------------------------
# OOD configuration
# ----------------------------
# Choose one:
#   "mean_msp"
#   "p95_entropy"
OOD_SCORE_NAME = "p95_entropy"

# Conservative thresholds from your calibration:
# mean_msp: exclude if < 0.888828
# p95_entropy: exclude if > 1.515980
OOD_THRESHOLD = 1.515980


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose(
        [
            ToImage(),
            Resize(size=IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    img = transform(img)
    return img.unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    pred_max = pred.argmax(dim=1, keepdim=True)
    pred_resized = Resize(
        size=original_shape,
        interpolation=InterpolationMode.NEAREST,
    )(pred_max)
    pred_numpy = pred_resized.cpu().numpy().squeeze()
    return pred_numpy.astype(np.uint8)


def compute_ood_score(logits: torch.Tensor) -> float:
    if OOD_SCORE_NAME == "mean_msp":
        return float(image_mean_msp(logits)[0].item())
    if OOD_SCORE_NAME == "p95_entropy":
        return float(image_percentile_entropy(logits, percentile=95.0)[0].item())
    raise ValueError(f"Unsupported OOD score: {OOD_SCORE_NAME}")


def should_include(score: float) -> bool:
    if OOD_SCORE_NAME == "mean_msp":
        return score >= OOD_THRESHOLD
    if OOD_SCORE_NAME == "p95_entropy":
        return score <= OOD_THRESHOLD
    raise ValueError(f"Unsupported OOD score: {OOD_SCORE_NAME}")


def find_image_files(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    image_files: list[Path] = []

    for pattern in ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp"):
        image_files.extend(root.glob(pattern))

    return sorted(set(image_files))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=19,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_root = Path(IMAGE_DIR)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(IMAGE_DIR)
    print(f"Found {len(image_files)} images to process.")

    csv_path = output_root / "predictions.csv"
    predictions: list[dict[str, object]] = []

    num_include = 0
    num_exclude = 0

    with torch.no_grad():
        for idx, img_path in enumerate(image_files):
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            logits = model(img_tensor)

            score = compute_ood_score(logits)
            include = should_include(score)

            if idx < 20:
                print(f"SCORE {img_path} | {OOD_SCORE_NAME}={score:.6f} | include={include}")

            seg_pred = postprocess(logits, original_shape)

            relative_path = img_path.relative_to(image_root)
            out_path = output_root / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred).save(out_path)

            predictions.append(
                {
                    "image_name": str(relative_path).replace("\\", "/"),
                    "include": bool(include),
                }
            )

            if include:
                num_include += 1
            else:
                num_exclude += 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "include"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")
    print(f"Finished. include={num_include}, exclude={num_exclude}")


if __name__ == "__main__":
    main()
