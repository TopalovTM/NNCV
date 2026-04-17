from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)

ID_TO_TRAIN_ID = {cls.id: cls.train_id for cls in Cityscapes.classes}
TRAIN_ID_TO_COLOR = {
    cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255
}
TRAIN_ID_TO_COLOR[255] = (0, 0, 0)


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: ID_TO_TRAIN_ID[x])


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in TRAIN_ID_TO_COLOR.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def build_transforms(image_size: tuple[int, int]) -> tuple[Compose, Compose]:
    img_transform = Compose(
        [
            ToImage(),
            Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    target_transform = Compose(
        [
            ToImage(),
            Resize(image_size, interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64, scale=False),
        ]
    )

    return img_transform, target_transform


def build_dataloaders(
    data_dir: str,
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    img_transform, target_transform = build_transforms(image_size)

    train_dataset = Cityscapes(
        data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader
