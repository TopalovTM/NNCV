from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from src.models.deeplabv3plus import Model
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=19,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = model(img_tensor)
            seg_pred = postprocess(pred, original_shape)

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred).save(out_path)


if __name__ == "__main__":
    main()
