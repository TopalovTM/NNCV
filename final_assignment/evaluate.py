from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from src.config import load_config
from src.data import (
    build_dataloaders,
    convert_to_train_id,
    convert_train_id_to_color,
)
from src.metrics import mean_dice, mean_iou
from src.models.deeplabv3plus import Model
from torchvision.utils import save_image


def parse_args():
    parser = ArgumentParser("Cityscapes checkpoint evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to evaluate",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_outputs",
        help="Directory where metrics and sample predictions are saved",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of qualitative prediction samples to save",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    save_dir: Path,
    num_samples: int,
) -> dict[str, float]:
    model.eval()

    losses = []
    dice_scores = []
    iou_scores = []

    sample_count = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = convert_to_train_id(labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().squeeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        losses.append(loss.item())
        dice_scores.append(mean_dice(outputs, labels, num_classes=num_classes))
        iou_scores.append(mean_iou(outputs, labels, num_classes=num_classes))

        if sample_count < num_samples:
            preds = outputs.softmax(1).argmax(1).unsqueeze(1)

            pred_color = convert_train_id_to_color(preds.cpu())
            label_color = convert_train_id_to_color(labels.unsqueeze(1).cpu())

            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break

                # Save prediction and label masks as PNGs
                pred_img = pred_color[i]
                label_img = label_color[i]

                save_image(
                    pred_img.float() / 255.0,
                    save_dir / f"sample_{sample_count:02d}_pred.png",
                )
                save_image(
                    label_img.float() / 255.0,
                    save_dir / f"sample_{sample_count:02d}_label.png",
                )

                sample_count += 1

    metrics = {
        "val_loss": sum(losses) / len(losses),
        "val_dice": sum(dice_scores) / len(dice_scores),
        "val_miou": sum(iou_scores) / len(iou_scores),
    }

    return metrics


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

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    save_dir = Path(args.save_dir) / checkpoint_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        save_dir=save_dir,
        num_samples=args.num_samples,
    )

    metrics_path = save_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    print(f"Checkpoint: {checkpoint_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
