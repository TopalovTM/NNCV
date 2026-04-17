"""
This script implements a training loop for the model. It is designed to be flexible,
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the
 `main.sh` script or directly
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress
 is not visible on the console.
   To address this, we use the `wandb` library for logging and tracking progress and
     results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be
 called from the main block.
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""

from __future__ import annotations

import random
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from src.config import load_config
from src.data import build_dataloaders
from src.engine import train_one_epoch, validate
from src.models.deeplabv3plus import Model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_args():
    parser = ArgumentParser("Cityscapes training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peak_deeplabv3plus_r50.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"]["amp"])

    experiment_name = cfg["experiment_name"]
    output_dir = Path("checkpoints") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(cfg["logging"]["use_wandb"])
    if use_wandb:
        wandb.init(
            project=cfg["logging"]["project"],
            name=experiment_name,
            config=cfg,
        )

    image_size = tuple(cfg["data"]["image_size"])
    num_classes = int(cfg["data"]["num_classes"])

    train_loader, valid_loader = build_dataloaders(
        data_dir=cfg["data"]["root_dir"],
        image_size=image_size,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = Model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=255,
        label_smoothing=cfg["train"].get("label_smoothing", 0.0),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["train"].get("lr_scheduler_factor", 0.5),
        patience=cfg["train"].get("lr_scheduler_patience", 3),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid_loss = float("inf")
    best_valid_dice = float("-inf")

    best_loss_model_path = output_dir / "best_model_loss.pt"
    best_dice_model_path = output_dir / "best_model_dice.pt"
    last_model_path = output_dir / "last_model.pt"

    # Early stopping
    early_stopping_patience = cfg["train"].get("early_stopping_patience", 6)
    epochs_without_improvement = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            use_amp=use_amp,
            use_wandb=use_wandb,
        )

        val_metrics = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            use_wandb=use_wandb,
            log_images=bool(cfg["logging"]["log_images"]),
            num_classes=num_classes,
        )

        val_loss = val_metrics["val_loss"]
        val_dice = val_metrics["val_dice"]
        val_miou = val_metrics["val_miou"]

        # Add scheduler step
        scheduler.step(val_dice)

        print(
            f"Epoch {epoch:03d}/{cfg['train']['epochs']:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_miou={val_miou:.4f}",
            flush=True,
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "val/loss_epoch": val_loss,
                    "val/dice_epoch": val_dice,
                    "val/miou_epoch": val_miou,
                    "train/lr_epoch": optimizer.param_groups[0]["lr"],
                }
            )

        torch.save(model.state_dict(), last_model_path)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), best_loss_model_path)

        if val_dice > best_valid_dice:
            best_valid_dice = val_dice
            torch.save(model.state_dict(), best_dice_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Current LR: {current_lr:.6f} | "
            f"Epochs without improvement: {epochs_without_improvement}",
            flush=True,
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs. "
                f"Best val_dice: {best_valid_dice:.4f}",
                flush=True,
            )
            break

    if use_wandb:
        wandb.finish()

    print(f"Training complete. Best loss model: {best_loss_model_path}")
    print(f"Training complete. Best dice model: {best_dice_model_path}")


if __name__ == "__main__":
    main()
