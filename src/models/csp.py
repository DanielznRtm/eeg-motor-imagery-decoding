"""Classical baselines: CSP + LDA and Filter Bank CSP + LDA.

Common Spatial Patterns (CSP) learns spatial filters w (linear combinations of
the 22 channels) that maximize the variance of the filtered signal for one
class while minimizing it for the other. Because motor imagery shows up as a
change in band power (ERD/ERS), the log-variance of a few CSP-filtered signals
is a compact, highly discriminative feature vector. LDA then separates the
classes with a linear boundary.

CSP is defined for two classes. Two multiclass extensions are supported:

* ``"ovr"`` (one-vs-rest): one binary CSP per class (class k vs. all others),
  features concatenated. This is what the FBCSP winner of the BCI Competition
  IV 2a used (Ang et al., 2012).
* ``"joint"``: MNE's native multiclass CSP, based on approximate joint
  diagonalization of the four class covariance matrices, with components
  ranked by mutual information (Grosse-Wentrup & Buss, 2008).
"""

from __future__ import annotations

from functools import partial

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

# rank="full" skips MNE's per-fit rank estimation (60% of the fit time). It is
# safe here: the 22 EEG channels have full rank (no average reference, no
# projectors), which MNE's own estimate confirms.


class OneVsRestCSP(BaseEstimator, TransformerMixin):
    """One binary CSP per class (class k vs. rest); log-variance features concatenated.

    With ``n_components = 2m`` each binary CSP keeps the m filters with the
    largest and the m with the smallest eigenvalues (``component_order=
    "alternate"``): the first maximize variance for class k relative to the
    rest, the second minimize it (e.g. a mu ERD specific to class k).
    """

    def __init__(self, n_components: int = 4, reg: str | float | None = "ledoit_wolf"):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.csps_ = [
            CSP(n_components=self.n_components, reg=self.reg, log=True, rank="full",
                component_order="alternate").fit(X, (y == c).astype(int))
            for c in self.classes_
        ]
        return self

    def transform(self, X):
        return np.concatenate([csp.transform(X) for csp in self.csps_], axis=1)


def make_csp(multiclass: str, n_components: int, reg) -> BaseEstimator:
    if multiclass == "ovr":
        return OneVsRestCSP(n_components=n_components, reg=reg)
    if multiclass == "joint":
        return CSP(n_components=n_components, reg=reg, log=True, rank="full",
                   component_order="mutual_info")
    raise ValueError(f"unknown multiclass strategy: {multiclass!r}")


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Fit an independent CSP on each frequency band and concatenate the features.

    Input: (n_trials, n_bands, n_channels, n_times).
    """

    def __init__(self, csp: BaseEstimator):
        self.csp = csp

    def fit(self, X, y):
        self.csps_ = [clone(self.csp).fit(X[:, b], y) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.concatenate([csp.transform(X[:, b]) for b, csp in enumerate(self.csps_)], axis=1)


def _lda() -> LinearDiscriminantAnalysis:
    # Shrinkage LDA (Ledoit-Wolf, chosen analytically): robust when the number of
    # features is not small compared with the ~230 training trials.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def make_csp_lda(multiclass: str = "ovr", n_components: int = 4, reg="ledoit_wolf") -> Pipeline:
    return Pipeline([("csp", make_csp(multiclass, n_components, reg)), ("lda", _lda())])


def make_fbcsp_lda(
    multiclass: str = "ovr",
    n_components: int = 4,
    reg="ledoit_wolf",
    k_features: int = 16,
    seed: int = 0,
    memory=None,
) -> Pipeline:
    """Filter Bank CSP: per-band CSP -> mutual-information feature selection -> LDA.

    Feature selection sits inside the pipeline, so during cross-validation it
    is re-fitted on each training fold and never sees the held-out fold.
    ``memory`` caches fitted transformers, so grid points that only change
    ``k`` reuse the per-band CSPs fitted on the same fold.
    """
    score = partial(mutual_info_classif, random_state=seed)
    return Pipeline([
        ("fbcsp", FilterBankCSP(make_csp(multiclass, n_components, reg))),
        ("select", SelectKBest(score_func=score, k=k_features)),
        ("lda", _lda()),
    ], memory=memory)


def build_pipeline(model_cfg: dict, seed: int, memory=None) -> Pipeline:
    """Build a pipeline from the ``model`` section of a config file."""
    kind = model_cfg["type"]
    common = dict(
        multiclass=model_cfg.get("multiclass", "ovr"),
        n_components=model_cfg.get("n_components", 4),
        reg=model_cfg.get("reg", "ledoit_wolf"),
    )
    if kind == "csp_lda":
        return make_csp_lda(**common)
    if kind == "fbcsp_lda":
        return make_fbcsp_lda(**common, k_features=model_cfg.get("k_features", 16), seed=seed,
                              memory=memory)
    raise ValueError(f"unknown classical model type: {kind!r}")
