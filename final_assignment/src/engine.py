from __future__ import annotations

import torch
from torchvision.utils import make_grid
import wandb

from src.data import convert_to_train_id, convert_train_id_to_color
from src.metrics import mean_dice, mean_iou


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    epoch: int,
    use_amp: bool,
    use_wandb: bool,
) -> float:
    model.train()
    running_loss = 0.0

    for step, (images, labels) in enumerate(loader):
        labels = convert_to_train_id(labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if use_wandb:
            wandb.log(
                {
                    "train/loss_step": loss.item(),
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    epoch: int,
    use_amp: bool,
    use_wandb: bool,
    log_images: bool,
    num_classes: int,
) -> dict[str, float]:
    model.eval()

    losses = []
    dice_scores = []
    iou_scores = []

    for step, (images, labels) in enumerate(loader):
        labels = convert_to_train_id(labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().squeeze(1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        losses.append(loss.item())
        dice_scores.append(mean_dice(outputs, labels, num_classes=num_classes))
        iou_scores.append(mean_iou(outputs, labels, num_classes=num_classes))

        if step == 0 and use_wandb and log_images:
            predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
            labels_vis = labels.unsqueeze(1)

            predictions = convert_train_id_to_color(predictions.cpu())
            labels_vis = convert_train_id_to_color(labels_vis.cpu())

            predictions_img = make_grid(predictions, nrow=4).permute(1, 2, 0).numpy()
            labels_img = make_grid(labels_vis, nrow=4).permute(1, 2, 0).numpy()

            wandb.log(
                {
                    "val/predictions": [wandb.Image(predictions_img)],
                    "val/labels": [wandb.Image(labels_img)],
                    "val/epoch": epoch,
                }
            )

    metrics = {
        "val_loss": sum(losses) / len(losses),
        "val_dice": sum(dice_scores) / len(dice_scores),
        "val_miou": sum(iou_scores) / len(iou_scores),
    }

    if use_wandb:
        wandb.log(
            {
                "val/loss": metrics["val_loss"],
                "val/dice": metrics["val_dice"],
                "val/miou": metrics["val_miou"],
                "val/epoch": epoch,
            }
        )

    return metrics