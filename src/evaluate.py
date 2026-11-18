"""Metrics, result files and confusion-matrix figures shared by all models."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binom
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from src.data import CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, Cohen's kappa and confusion matrix (rows = true class)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))),
    }


def chance_upper_bound(n_trials: int, n_classes: int = 4, alpha: float = 0.05) -> float:
    """Accuracy a random classifier exceeds with probability < alpha (binomial).

    With 288 balanced test trials and 4 classes, accuracies below this value
    are not distinguishable from chance (Mueller-Putz et al., 2008).
    """
    return float(binom.ppf(1 - alpha, n_trials, 1 / n_classes) / n_trials)


def summarize(per_subject: pd.DataFrame) -> dict:
    """Mean and sample std (ddof=1) across subjects."""
    out = {}
    for col in ("accuracy", "kappa"):
        out[f"{col}_mean"] = float(per_subject[col].mean())
        out[f"{col}_std"] = float(per_subject[col].std(ddof=1))
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def environment_info() -> dict:
    import mne
    import moabb
    import sklearn

    info = {"python": platform.python_version(), "mne": mne.__version__,
            "moabb": moabb.__version__, "scikit-learn": sklearn.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def save_run(out_dir: Path, cfg: dict, per_subject: pd.DataFrame, predictions: pd.DataFrame,
             confusion: np.ndarray) -> dict:
    """Write config, per-subject metrics, predictions, confusion matrix and summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    per_subject.to_csv(out_dir / "per_subject.csv", index=False, float_format="%.4f")
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(out_dir / "confusion_matrix.csv")
    n_test = int(predictions.groupby("subject").size().iloc[0])
    summary = {
        "experiment": cfg["name"],
        **summarize(per_subject),
        "n_subjects": int(per_subject["subject"].nunique()),
        "chance_upper_bound_accuracy_p05": chance_upper_bound(n_test),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "environment": environment_info(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_confusion(confusion: np.ndarray, title: str, path: Path) -> None:
    """Row-normalized confusion matrix (percent of each true class), summed over subjects."""
    import matplotlib.pyplot as plt

    pct = 100 * confusion / confusion.sum(axis=1, keepdims=True)
    labels = ["Left hand", "Right hand", "Feet", "Tongue"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                    color="white" if pct[i, j] > 55 else "#1f1f1e", fontsize=10)
    ax.set_xticks(range(4), labels, rotation=30, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("% of true-class trials")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
