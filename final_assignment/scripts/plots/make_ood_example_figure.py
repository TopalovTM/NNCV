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
    parser.add_argument("--id-image", type=str, required=True, help="Path to ID example image")
    parser.add_argument("--ood-image", type=str, required=True, help="Path to OOD example image")
    parser.add_argument("--id-score", type=float, required=True, help="Score for ID image")
    parser.add_argument("--ood-score", type=float, required=True, help="Score for OOD image")
    parser.add_argument("--score-name", type=str, default="mean_msp", help="Score name")
    parser.add_argument("--output", type=str, required=True, help="Output figure path")
    parser.add_argument("--title", type=str, default="", help="Optional figure title")
    args = parser.parse_args()

    id_img = load_image(args.id_image)
    ood_img = load_image(args.ood_image)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))

    axes[0].imshow(id_img)
    axes[0].set_title(
        f"ID example\n{args.score_name}={args.id_score:.4f}\ninclude=True",
        fontsize=7,
    )
    axes[0].axis("off")

    axes[1].imshow(ood_img)
    axes[1].set_title(
        f"OOD example\n{args.score_name}={args.ood_score:.4f}\ninclude=False",
        fontsize=7,
    )
    axes[1].axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=9)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
