from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="Path to ground-truth visualization")
    parser.add_argument("--r101l", type=str, required=True, help="Path to R101 label visualization")
    parser.add_argument("--r101p", type=str, required=True, help="Path to R101 prediction visualization")
    parser.add_argument("--output", type=str, required=True, help="Output figure path")
    parser.add_argument("--title", type=str, default="", help="Optional figure title")
    args = parser.parse_args()

    gt_img = load_image(args.gt)
    r101l_img = load_image(args.r101l)
    r101p_img = load_image(args.r101p)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    panels = [
        ("Input Image", gt_img),
        ("Ground Truth Label", r101l_img),
        ("DeepLabV3+ R101 Prediction", r101p_img),
    ]

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=11)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
