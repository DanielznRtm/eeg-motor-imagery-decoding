"""Tests for the classical pipelines and metrics, on synthetic data.

The synthetic data mimics motor imagery: each class has more variance in a
different "source" channel, which is exactly what CSP is designed to find.

The variance ratio is 2, comparable to a strong ERD/ERS. Much larger ratios
break the mutual-information approximation used to rank components in MNE's
joint multiclass CSP (Grosse-Wentrup & Buss, 2008), which is only accurate
when class variances are close: with a ratio of 9, informative components get
a *lower* score than pure-noise ones.
"""

import numpy as np
import pytest
from sklearn.base import clone

from src.evaluate import chance_upper_bound, compute_metrics
from src.models.csp import FilterBankCSP, OneVsRestCSP, build_pipeline, make_csp


def _synthetic(n_per_class=30, n_ch=8, n_t=200, n_bands=None, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(4):
        trials = rng.normal(size=(n_per_class, n_ch, n_t))
        trials[:, k] *= np.sqrt(2.0)  # class k: twice the variance on channel k
        X.append(trials)
        y += [k] * n_per_class
    X = np.concatenate(X)
    if n_bands:
        X = np.repeat(X[:, None], n_bands, axis=1) + 0.1 * rng.normal(size=(len(X), n_bands, n_ch, n_t))
    return X, np.array(y)


@pytest.mark.parametrize("multiclass,n_components,n_features", [("ovr", 4, 16), ("joint", 6, 6)])
def test_csp_feature_shapes(multiclass, n_components, n_features):
    X, y = _synthetic()
    feats = make_csp(multiclass, n_components, "ledoit_wolf").fit(X, y).transform(X)
    assert feats.shape == (len(y), n_features)
    assert np.isfinite(feats).all()


def test_filter_bank_csp_concatenates_bands():
    X, y = _synthetic(n_bands=3)
    fb = FilterBankCSP(OneVsRestCSP(n_components=2)).fit(X, y)
    assert len(fb.csps_) == 3
    assert fb.transform(X).shape == (len(y), 3 * 4 * 2)


@pytest.mark.parametrize("cfg", [
    {"type": "csp_lda", "multiclass": "ovr", "n_components": 2},
    {"type": "csp_lda", "multiclass": "joint", "n_components": 4},
    {"type": "fbcsp_lda", "multiclass": "ovr", "n_components": 2, "k_features": 8},
])
def test_pipelines_learn_separable_synthetic_data(cfg):
    n_bands = 2 if cfg["type"] == "fbcsp_lda" else None
    X_tr, y_tr = _synthetic(n_bands=n_bands, seed=0)
    X_te, y_te = _synthetic(n_bands=n_bands, seed=1)
    pipe = build_pipeline(cfg, seed=0).fit(X_tr, y_tr)
    assert (pipe.predict(X_te) == y_te).mean() > 0.9


def test_pipeline_is_clonable_for_grid_search():
    pipe = build_pipeline({"type": "fbcsp_lda", "n_components": 2, "k_features": 8}, seed=0)
    params = clone(pipe).get_params()
    assert "fbcsp__csp__n_components" in params and "select__k" in params


def test_metrics_perfect_and_chance():
    y = np.repeat(np.arange(4), 10)
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["kappa"] == 1.0
    assert m["confusion"].trace() == len(y)
    # balanced 4 classes: kappa = (acc - 0.25) / 0.75
    y_pred = y.copy()
    y_pred[:20] = (y_pred[:20] + 1) % 4
    m = compute_metrics(y, y_pred)
    assert m["kappa"] == pytest.approx((m["accuracy"] - 0.25) / 0.75)


def test_chance_upper_bound_288_trials():
    # 95th percentile of a Binomial(288, 0.25), as a fraction of 288
    assert 0.28 < chance_upper_bound(288) < 0.30
