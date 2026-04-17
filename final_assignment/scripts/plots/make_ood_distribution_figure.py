from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-csv", type=str, required=True, help="CSV with ID scores")
    parser.add_argument("--ood-csv", type=str, required=True, help="CSV with OOD scores")
    parser.add_argument("--score", type=str, required=True, help="Column name to plot")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold to draw")
    parser.add_argument("--output", type=str, required=True, help="Output figure path")
    parser.add_argument("--bins", type=int, default=40, help="Number of bins")
    parser.add_argument("--id-label", type=str, default="Cityscapes (ID)")
    parser.add_argument("--ood-label", type=str, default="COCO (OOD proxy)")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    id_df = pd.read_csv(args.id_csv)
    ood_df = pd.read_csv(args.ood_csv)

    if args.score not in id_df.columns:
        raise ValueError(f"{args.score} not found in ID CSV")
    if args.score not in ood_df.columns:
        raise ValueError(f"{args.score} not found in OOD CSV")

    id_scores = id_df[args.score].dropna()
    ood_scores = ood_df[args.score].dropna()

    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    ax.hist(id_scores, bins=args.bins, alpha=0.65, density=True, label=args.id_label)
    ax.hist(ood_scores, bins=args.bins, alpha=0.65, density=True, label=args.ood_label)

    ax.axvline(
        args.threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {args.threshold:.4f}",
    )

    ax.set_xlabel(args.score.replace("_", " "))
    ax.set_ylabel("Density")
    if args.title:
        ax.set_title(args.title)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
