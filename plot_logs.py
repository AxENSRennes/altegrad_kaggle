#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot training & validation metrics from CSV logs
and save figures to /figures, with version_gnn in filenames
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG (DOIT MATCHER LE TRAIN SCRIPT)
# =========================================================
version_gnn = "v4"

LOG_CSV = f"logs/log_{version_gnn}.csv"
FIG_DIR = "figures"

# =========================================================
# UTILS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.exists(LOG_CSV):
        raise FileNotFoundError(f"Missing log file: {LOG_CSV}")

    ensure_dir(FIG_DIR)

    df = pd.read_csv(LOG_CSV)
    epochs = df["epoch"]

    prefix = f"{version_gnn}_"

    # ===============================
    # 1) Training loss
    # ===============================
    plt.figure()
    plt.plot(epochs, df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(f"Training loss ({version_gnn})")
    save_fig(prefix + "train_loss.png")

    # ===============================
    # 2) Logit scale
    # ===============================
    plt.figure()
    plt.plot(epochs, df["logit_scale"])
    plt.xlabel("Epoch")
    plt.ylabel("Scale")
    plt.title(f"Logit scale ({version_gnn})")
    save_fig(prefix + "logit_scale.png")

    # ===============================
    # 3) Validation MRR
    # ===============================
    plt.figure()
    plt.plot(epochs, df["val_MRR"])
    plt.xlabel("Epoch")
    plt.ylabel("MRR")
    plt.title(f"Validation MRR ({version_gnn})")
    save_fig(prefix + "val_mrr.png")

    # ===============================
    # 4) Validation Recall@k
    # ===============================
    plt.figure()
    plt.plot(epochs, df["val_R@1"], label="R@1")
    plt.plot(epochs, df["val_R@5"], label="R@5")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title(f"Validation Recall@k ({version_gnn})")
    plt.legend()
    save_fig(prefix + "val_recall.png")

    # ===============================
    # 5) Overview (all validation metrics)
    # ===============================
    plt.figure()
    plt.plot(epochs, df["val_MRR"], label="MRR")
    plt.plot(epochs, df["val_R@1"], label="R@1")
    plt.plot(epochs, df["val_R@5"], label="R@5")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Validation metrics overview ({version_gnn})")
    plt.legend()
    save_fig(prefix + "val_metrics_overview.png")

    print("\nAll figures generated successfully.")

# =========================================================
if __name__ == "__main__":
    main()
