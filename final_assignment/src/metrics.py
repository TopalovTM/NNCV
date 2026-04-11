from __future__ import annotations

import torch


def _flatten_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = logits.argmax(dim=1)
    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    return preds, targets


@torch.no_grad()
def mean_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    preds, targets = _flatten_predictions(logits, targets, ignore_index)

    if preds.numel() == 0:
        return 0.0

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union > 0:
            ious.append(intersection / union)

    if not ious:
        return 0.0

    return float(sum(ious) / len(ious))


@torch.no_grad()
def mean_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    preds, targets = _flatten_predictions(logits, targets, ignore_index)

    if preds.numel() == 0:
        return 0.0

    dices = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().item()
        denom = pred_mask.sum().item() + target_mask.sum().item()

        if denom > 0:
            dices.append((2.0 * intersection) / denom)

    if not dices:
        return 0.0

    return float(sum(dices) / len(dices))