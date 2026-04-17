from __future__ import annotations

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    p95_entropy_thresholds = [1.334278, 1.515980]
    mean_msp_thresholds = [0.900499, 0.888828]

    print("p95_entropy rejection rates")
    for thr in p95_entropy_thresholds:
        rej = (df["p95_entropy"] > thr).mean()
        print(f"  threshold={thr:.6f} -> rejected={rej:.3%}")

    print("\nmean_msp rejection rates")
    for thr in mean_msp_thresholds:
        rej = (df["mean_msp"] < thr).mean()
        print(f"  threshold={thr:.6f} -> rejected={rej:.3%}")


if __name__ == "__main__":
    main()
