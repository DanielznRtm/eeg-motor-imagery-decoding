"""Single entry point for every experiment.

    python -m src.train --config configs/csp_lda.yaml [--subjects 1 3]

Protocol (all models): fit on session 1 (``0train``), evaluate once on
session 2 (``1test``). Anything that is *chosen* (hyper-parameters, early
stopping) is chosen with data from session 1 only.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data import SUBJECTS, EpochConfig, load_subject
from src.evaluate import compute_metrics, plot_confusion, save_run

CLASSICAL = {"csp_lda", "fbcsp_lda"}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_classical(cfg: dict, subjects: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Grid-search hyper-parameters with stratified K-fold CV on session 1,
    refit the best pipeline on all of session 1, then predict session 2."""
    from src.models.csp import build_pipeline

    seed = cfg["seed"]
    data_cfg = EpochConfig.from_dict(cfg["data"])
    search_cfg = cfg.get("search", {})
    rows, preds, confusion = [], [], 0
    cache_dir = tempfile.mkdtemp(prefix="sk_cache_")  # pipeline transformer cache
    for s in subjects:
        d = load_subject(s, data_cfg)
        cv = StratifiedKFold(search_cfg.get("cv_folds", 5), shuffle=True, random_state=seed)
        search = GridSearchCV(
            build_pipeline(cfg["model"], seed, memory=cache_dir),
            param_grid=search_cfg.get("param_grid", {}),
            cv=cv,
            scoring="accuracy",
            refit=True,
            n_jobs=search_cfg.get("n_jobs", 1),
        )
        t0 = time.perf_counter()
        search.fit(d.X_train, d.y_train)  # session 1 only
        fit_time = time.perf_counter() - t0
        y_pred = search.predict(d.X_test)  # session 2, used once
        m = compute_metrics(d.y_test, y_pred)
        confusion = confusion + m["confusion"]
        rows.append({
            "subject": s,
            "accuracy": m["accuracy"],
            "kappa": m["kappa"],
            "cv_accuracy_session1": float(search.best_score_),
            "best_params": json.dumps(search.best_params_),
            "fit_time_s": fit_time,
            "n_train": len(d.y_train),
            "n_test": len(d.y_test),
        })
        preds.append(pd.DataFrame({"subject": s, "trial": np.arange(len(y_pred)),
                                   "y_true": d.y_test, "y_pred": y_pred}))
        print(f"S{s}: acc={m['accuracy']:.3f} kappa={m['kappa']:.3f} "
              f"(session-1 CV acc={search.best_score_:.3f}, {search.best_params_}, {fit_time:.1f}s)")
    shutil.rmtree(cache_dir, ignore_errors=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subjects", type=int, nargs="*", help="default: all 9 subjects")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    subjects = args.subjects or SUBJECTS
    model_type = cfg["model"]["type"]
    if model_type in CLASSICAL:
        per_subject, predictions, confusion = run_classical(cfg, subjects)
    else:
        raise NotImplementedError(f"model type {model_type!r} is not implemented yet")

    out_dir = Path(args.results_dir) / cfg["name"]
    summary = save_run(out_dir, cfg, per_subject, predictions, confusion)
    plot_confusion(confusion, f"{cfg['name']}: session 2, all subjects",
                   Path(args.figures_dir) / f"{cfg['name']}_confusion.png")
    print(f"\n{cfg['name']}: accuracy {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}, "
          f"kappa {summary['kappa_mean']:.3f} ± {summary['kappa_std']:.3f} "
          f"(chance upper bound p<.05: {summary['chance_upper_bound_accuracy_p05']:.3f})")
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
