from __future__ import annotations

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    high_bad = ["mean_entropy", "p95_entropy", "mean_energy"]
    low_bad = ["mean_msp", "p05_msp"]

    print("\n=== High-is-bad scores (reject if score > threshold) ===")
    for col in high_bad:
        print(f"\n{col}")
        for q in [0.90, 0.95, 0.97, 0.99]:
            thr = df[col].quantile(q)
            rejected = (df[col] > thr).mean()
            kept_mean_dice = df.loc[df[col] <= thr, "dice"].mean()
            print(
                f"  q={q:.2f}  threshold={thr:.6f}  "
                f"rejected={rejected:.3%}  kept_mean_dice={kept_mean_dice:.6f}"
            )

    print("\n=== Low-is-bad scores (reject if score < threshold) ===")
    for col in low_bad:
        print(f"\n{col}")
        for q in [0.10, 0.05, 0.03, 0.01]:
            thr = df[col].quantile(q)
            rejected = (df[col] < thr).mean()
            kept_mean_dice = df.loc[df[col] >= thr, "dice"].mean()
            print(
                f"  q={q:.2f}  threshold={thr:.6f}  "
                f"rejected={rejected:.3%}  kept_mean_dice={kept_mean_dice:.6f}"
            )

    print("\n=== Correlation with Dice ===")
    cols = ["mean_msp", "mean_entropy", "mean_energy", "p95_entropy", "p05_msp", "dice"]
    print(df[cols].corr()["dice"].sort_values(ascending=False))


if __name__ == "__main__":
    main()
