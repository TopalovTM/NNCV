from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)

from src.models.deeplabv3plus import Model
from src.ood.scores import (
    image_mean_energy,
    image_mean_entropy,
    image_mean_msp,
    image_percentile_entropy,
    image_percentile_low_msp,
)

IMAGE_SIZE = (512, 1024)


def parse_args():
    parser = ArgumentParser("Score a folder of images for OOD")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder-name", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    return parser.parse_args()


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
    return transform(img).unsqueeze(0)


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        encoder_name=args.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=19,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image_dir = Path(args.image_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(sorted(image_dir.glob(ext)))

    csv_path = save_dir / "ood_scores.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_name",
                "mean_msp",
                "mean_entropy",
                "mean_energy",
                "p95_entropy",
                "p05_msp",
            ]
        )

        for idx, img_path in enumerate(image_files):
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(device)

            logits = model(x)

            mean_msp = float(image_mean_msp(logits)[0].item())
            mean_entropy = float(image_mean_entropy(logits)[0].item())
            mean_energy = float(
                image_mean_energy(
                    logits, temperature=args.energy_temperature
                )[0].item()
            )
            p95_entropy = float(image_percentile_entropy(logits, 95.0)[0].item())
            p05_msp = float(image_percentile_low_msp(logits, 5.0)[0].item())

            writer.writerow(
                [
                    img_path.name,
                    mean_msp,
                    mean_entropy,
                    mean_energy,
                    p95_entropy,
                    p05_msp,
                ]
            )

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_files)} images", flush=True)

    print(f"Saved scores to {csv_path}")


if __name__ == "__main__":
    main()