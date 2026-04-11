"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
IMAGE_SIZE = (512, 1024)


def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
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
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
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
    model.load_state_dict(state_dict, strict=True) # Ensure the state dict matches the model architecture
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png")) # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
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