from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from src.config import load_config
from src.data import (
    build_dataloaders,
    convert_to_train_id,
    convert_train_id_to_color,
)
from src.metrics import mean_dice, mean_iou
from src.models.deeplabv3plus import Model
from src.ood.scores import (
    image_mean_energy,
    image_mean_entropy,
    image_mean_msp,
    image_percentile_entropy,
    image_percentile_low_msp,
)


def parse_args():
    parser = ArgumentParser("OOD scoring for semantic segmentation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="ood_outputs")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(cfg["data"]["image_size"])
    num_classes = int(cfg["data"]["num_classes"])

    _, valid_loader = build_dataloaders(
        data_dir=cfg["data"]["root_dir"],
        image_size=image_size,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = Model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "ood_scores.csv"

    sample_count = 0
    total_dice = []
    total_miou = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "global_index",
                "mean_msp",
                "mean_entropy",
                "mean_energy",
                "p95_entropy",
                "p05_msp",
                "dice",
                "miou",
            ]
        )

        global_index = 0

        for images, labels in valid_loader:
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().squeeze(1)

            logits = model(images)

            batch_mean_msp = image_mean_msp(logits)
            batch_mean_entropy = image_mean_entropy(logits)
            batch_mean_energy = image_mean_energy(
                logits,
                temperature=args.energy_temperature,
            )
            batch_p95_entropy = image_percentile_entropy(logits, percentile=95.0)
            batch_p05_msp = image_percentile_low_msp(logits, percentile=5.0)

            preds = logits.argmax(dim=1, keepdim=True)
            pred_color = convert_train_id_to_color(preds.cpu())
            label_color = convert_train_id_to_color(labels.unsqueeze(1).cpu())

            for i in range(images.shape[0]):
                dice_i = mean_dice(
                    logits[i : i + 1],
                    labels[i : i + 1],
                    num_classes=num_classes,
                )
                miou_i = mean_iou(
                    logits[i : i + 1],
                    labels[i : i + 1],
                    num_classes=num_classes,
                )

                total_dice.append(dice_i)
                total_miou.append(miou_i)

                writer.writerow(
                    [
                        global_index,
                        float(batch_mean_msp[i].item()),
                        float(batch_mean_entropy[i].item()),
                        float(batch_mean_energy[i].item()),
                        float(batch_p95_entropy[i].item()),
                        float(batch_p05_msp[i].item()),
                        float(dice_i),
                        float(miou_i),
                    ]
                )

                if sample_count < args.num_samples:
                    save_image(
                        pred_color[i].float() / 255.0,
                        save_dir / f"sample_{sample_count:02d}_pred.png",
                    )
                    save_image(
                        label_color[i].float() / 255.0,
                        save_dir / f"sample_{sample_count:02d}_label.png",
                    )
                    sample_count += 1

                global_index += 1

    summary_path = save_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"mean_dice: {sum(total_dice)/len(total_dice):.6f}\n")
        f.write(f"mean_miou: {sum(total_miou)/len(total_miou):.6f}\n")

    print(f"Saved OOD scores to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()