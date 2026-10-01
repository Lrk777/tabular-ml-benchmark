"""Evaluate a completed run: print table + write comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def print_table(metrics: dict) -> None:
    print(f"seed={metrics.get('seed')}  best={metrics.get('best_model')}")
    print(f"{'model':24s} {'acc':>7} {'f1':>7} {'roc_auc':>8} {'ap':>7}")
    for name, block in metrics["models"].items():
        t = block["test"]
        marker = " *" if name == metrics.get("best_model") else ""
        print(
            f"{name:24s} {t['accuracy']:7.3f} {t['f1']:7.3f} "
            f"{t['roc_auc']:8.3f} {t['average_precision']:7.3f}{marker}"
        )


def plot_roc_pr(run_dir: Path, figures_dir: Path) -> list[Path]:
    pred_path = run_dir / "predictions.npz"
    if not pred_path.exists():
        return []
    data = np.load(pred_path)
    y_true = data["y_test"]
    score_keys = [k for k in data.files if k.endswith("_score")]
    written: list[Path] = []

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in score_keys:
        name = key.replace("_score", "")
        RocCurveDisplay.from_predictions(y_true, data[key], name=name, ax=ax)
    ax.set_title("ROC — Adult income (test)")
    fig.tight_layout()
    out = figures_dir / "roc_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    written.append(out)

    fig, ax = plt.subplots(figsize=(8, 6))
    for key in score_keys:
        name = key.replace("_score", "")
        PrecisionRecallDisplay.from_predictions(y_true, data[key], name=name, ax=ax)
    ax.set_title("Precision–Recall — Adult income (test)")
    fig.tight_layout()
    out = figures_dir / "pr_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    written.append(out)

    # Confusion matrix for best model if present
    metrics = load_metrics(run_dir)
    best = metrics.get("best_model")
    if best and f"{best}_pred" in data.files:
        cm = confusion_matrix(y_true, data[f"{best}_pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix — {best}")
        fig.tight_layout()
        out = figures_dir / "confusion_best.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)

    # Bar chart of test ROC-AUC
    names = list(metrics["models"].keys())
    aucs = [metrics["models"][n]["test"]["roc_auc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2a6f97" if n != best else "#1b4332" for n in names]
    ax.barh(names, aucs, color=colors)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Model comparison (test)")
    fig.tight_layout()
    out = figures_dir / "roc_auc_bars.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    written.append(out)

    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Adult income run")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("reports/latest"),
        help="Directory with metrics.json from train",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Where to write PNG figures",
    )
    args = parser.parse_args(argv)

    metrics = load_metrics(args.run_dir)
    print_table(metrics)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_roc_pr(args.run_dir, args.figures_dir)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
