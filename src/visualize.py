"""Figures that need a fitted model.

    python -m src.visualize csp-patterns --config configs/csp_lda.yaml

CSP *filters* (rows of ``filters_``) are what the model multiplies the data
with; they are hard to interpret, because they also cancel noise. CSP
*patterns* (rows of ``patterns_``, the columns of the inverse of the filter
matrix) show how each source projects onto the scalp, and are the ones that
should be plotted (Haufe et al., 2014). The sign of a pattern is arbitrary; we
flip each one so that its largest-magnitude channel is positive.

Pattern figures are fitted on session 1 only, with the hyper-parameters chosen
in ``python -m src.train`` (read from ``results/<name>/per_subject.csv``). The
subject shown in detail is the one with the best session-1 cross-validation
accuracy, so it is not picked by looking at test results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.data import CLASSES, SUBJECTS, EpochConfig, load_epochs, load_subject
from src.models.csp import OneVsRestCSP
from src.train import load_config

LABELS = {"left_hand": "Left hand", "right_hand": "Right hand", "feet": "Feet", "tongue": "Tongue"}


def _eeg_info() -> mne.Info:
    epochs, _, _ = load_epochs(1, EpochConfig(fmin=8.0, fmax=30.0))
    return epochs.info


def _oriented(pattern: np.ndarray) -> np.ndarray:
    return pattern * np.sign(pattern[np.argmax(np.abs(pattern))])


def class_patterns(ovr: OneVsRestCSP, X: np.ndarray, y: np.ndarray) -> list[dict]:
    """For each class-vs-rest CSP, describe each kept component.

    ``effect`` is the mean log-variance of the component for the class minus
    for the rest, on the training data: negative = power *decreases* for that
    class relative to the others (an ERD-like component).
    """
    out = []
    for k, csp in enumerate(ovr.csps_):
        W = csp.filters_[: csp.n_components]
        logvar = np.log(np.var(np.einsum("cn,tns->tcs", W, X), axis=-1))
        effect = logvar[y == k].mean(axis=0) - logvar[y != k].mean(axis=0)
        for i in range(csp.n_components):
            out.append({"class": CLASSES[k], "component": i, "effect": float(effect[i]),
                        "pattern": _oriented(csp.patterns_[i])})
    return out


def fit_subject(cfg: dict, subject: int, n_components: int):
    d = load_subject(subject, EpochConfig.from_dict(cfg["data"]))
    X = d.X_train.astype(np.float64)
    ovr = OneVsRestCSP(n_components=n_components, reg=cfg["model"].get("reg", "ledoit_wolf"))
    return class_patterns(ovr.fit(X, d.y_train), X, d.y_train)


def _topo(ax, values, info, title):
    vmax = np.abs(values).max()
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vmax, vmax), contours=0, sensors=True)
    ax.set_title(title, fontsize=9)
    return im


def plot_subject_patterns(pats: list[dict], info, subject: int, path: Path) -> None:
    """Rows: class-vs-rest CSP. Columns: strongest decrease / strongest increase."""
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(5.2, 10))
    for r, cls in enumerate(CLASSES):
        comps = [p for p in pats if p["class"] == cls]
        down = min(comps, key=lambda p: p["effect"])
        up = max(comps, key=lambda p: p["effect"])
        _topo(axes[r, 0], down["pattern"], info,
              f"{LABELS[cls]} vs rest\npower lower for {LABELS[cls].lower()} ({down['effect']:+.2f})")
        _topo(axes[r, 1], up["pattern"], info,
              f"{LABELS[cls]} vs rest\npower higher for {LABELS[cls].lower()} ({up['effect']:+.2f})")
    fig.suptitle(f"CSP patterns, subject {subject} (session 1, 8-30 Hz)\n"
                 "value in brackets: log-variance difference, class minus rest", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_subjects(all_pats: dict[int, list[dict]], info, path: Path) -> None:
    """For left and right hand: the component whose power drops most for that class."""
    fig, axes = plt.subplots(2, len(all_pats), figsize=(1.6 * len(all_pats), 3.8))
    for c, (s, pats) in enumerate(all_pats.items()):
        for r, cls in enumerate(["left_hand", "right_hand"]):
            down = min((p for p in pats if p["class"] == cls), key=lambda p: p["effect"])
            _topo(axes[r, c], down["pattern"], info, f"S{s}" if r == 0 else "")
    fig.text(0.0, 0.73, "Left hand\nvs rest", ha="right", va="center", fontsize=9)
    fig.text(0.0, 0.29, "Right hand\nvs rest", ha="right", va="center", fontsize=9)
    fig.suptitle("CSP pattern with the strongest power decrease for each hand, per subject "
                 "(session 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def csp_patterns(config_path: str, results_dir: str, figures_dir: str) -> None:
    cfg = load_config(config_path)
    if cfg["model"]["type"] != "csp_lda" or cfg["model"].get("multiclass", "ovr") != "ovr":
        raise ValueError("pattern plots are implemented for one-vs-rest CSP + LDA")
    per_subject = pd.read_csv(Path(results_dir) / cfg["name"] / "per_subject.csv")
    n_comp = {int(r.subject): json.loads(r.best_params)["csp__n_components"]
              for r in per_subject.itertuples()}
    info = _eeg_info()
    all_pats = {s: fit_subject(cfg, s, n_comp[s]) for s in SUBJECTS}

    best = int(per_subject.sort_values("cv_accuracy_session1", ascending=False).iloc[0]["subject"])
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_subject_patterns(all_pats[best], info, best, fig_dir / f"{cfg['name']}_patterns_S{best}.png")
    plot_all_subjects(all_pats, info, fig_dir / f"{cfg['name']}_patterns_hands_all_subjects.png")
    print(f"detailed patterns: subject {best} (best session-1 CV accuracy); figures in {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["csp-patterns"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    if args.figure == "csp-patterns":
        csp_patterns(args.config, args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
