#!/usr/bin/env python3
"""
Plot loss curves from loss_curve_all_nodes_3.tsv (e.g. from /tmp/RTE/ or /tmp/Rte).
Usage: python plot_loss_curve.py [path_to_tsv]
  Default path: /tmp/Rte/loss_curve_all_nodes_3.tsv
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# Save figures under project root /figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")


def main():
    default_path = "/tmp/RTE/loss_curve_all_nodes_3.tsv"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.isfile(path):
        print("File not found:", path)
        sys.exit(1)

    with open(path) as f:
        header = f.readline().strip().split("\t")
    data = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            row = line.strip().split("\t")
            if row:
                data.append([float(x) for x in row])

    if not data:
        print("No data in", path)
        sys.exit(1)

    batch = [r[0] for r in data]
    loss_cols = header[1:]  # loss_rank0, loss_rank1, ... or just "loss"

    plt.figure(figsize=(8, 5))
    for i, col in enumerate(loss_cols):
        losses = [r[i + 1] for r in data]
        label = col if col != "loss" else "loss (single node)"
        plt.plot(batch, losses, label=label, alpha=0.9)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss curve (all nodes)" if len(loss_cols) > 1 else "Loss curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "loss_curve_all_nodes.png")
    plt.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
