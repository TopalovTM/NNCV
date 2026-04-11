from __future__ import annotations

import torch
from torchvision.utils import make_grid
import wandb

from src.data import convert_to_train_id, convert_train_id_to_color


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
) -> float:
    model.eval()
    losses = []

    for step, (images, labels) in enumerate(loader):
        labels = convert_to_train_id(labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().squeeze(1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        losses.append(loss.item())

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

    avg_loss = sum(losses) / len(losses)

    if use_wandb:
        wandb.log({"val/loss": avg_loss, "val/epoch": epoch})

    return avg_loss