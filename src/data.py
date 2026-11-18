"""Data loading for BCI Competition IV 2a (MOABB: ``BNCI2014_001``).

Timeline of one trial (seconds, trial time):

    0.0  fixation cross + warning tone
    2.0  cue (arrow: left / right / down / up), shown for 1.25 s
    2.0-6.0  motor imagery
    6.0  fixation cross disappears, short break

MOABB defines the task interval as [2, 6] s, so the ``tmin``/``tmax`` passed to
the paradigm are *relative to the cue*. ``tmin=0.5, tmax=2.5`` therefore selects
0.5-2.5 s after the cue (2.5-4.5 s in trial time). The ``mne.Epochs`` returned by
MOABB keep trial time on their time axis; :func:`load_epochs` shifts it so that
the cue is at t = 0, which is what every figure in this repo uses.

Run ``python -m src.data`` to download all subjects and write a trial-count
summary to ``results/data_summary.csv``.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# MOABB 1.7 calls a montage name that MNE 1.13 deprecates; harmless for us.
warnings.filterwarnings("ignore", message="Montage name 'standard_1005'", category=FutureWarning)
os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")  # also inherited by worker processes
mne.set_log_level("WARNING")
logging.getLogger("moabb").setLevel(logging.WARNING)

from moabb.datasets import BNCI2014_001  # noqa: E402
from moabb.paradigms import FilterBankMotorImagery, MotorImagery  # noqa: E402

SUBJECTS = list(range(1, 10))
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]  # label index = position
EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]  # order returned by MOABB; EOG channels are dropped by the paradigm
SESSION_TRAIN = "0train"
SESSION_TEST = "1test"
CUE_ONSET = 2.0  # seconds after trial start
RAW_SFREQ = 250.0


@dataclass(frozen=True)
class EpochConfig:
    """Preprocessing applied by the MOABB paradigm.

    Band-pass filtering is applied by MOABB to the *continuous* recording
    (before epoching), which avoids filter edge artefacts inside the epochs.
    Filtering and resampling are per-sample operations that do not use any
    statistics of other trials, so applying them to both sessions is not a
    source of leakage.
    """

    fmin: float = 4.0
    fmax: float = 40.0
    tmin: float = 0.5  # relative to cue
    tmax: float = 2.5  # relative to cue
    resample: float | None = None  # Hz; None keeps 250 Hz
    # Filter bank: if given, fmin/fmax are ignored and each (lo, hi) band is
    # applied separately; arrays then get a band axis (see SubjectData).
    bands: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "EpochConfig":
        d = dict(d)
        if d.get("bands") is not None:
            d["bands"] = tuple(tuple(float(f) for f in b) for b in d["bands"])
        return cls(**d)


@dataclass
class SubjectData:
    """Arrays for one subject, split by the competition protocol.

    X arrays are (n_trials, n_channels, n_times) in microvolts, or
    (n_trials, n_bands, n_channels, n_times) when a filter bank is used.
    """

    X_train: np.ndarray
    y_train: np.ndarray  # (n_trials,), int in [0, 3]
    X_test: np.ndarray
    y_test: np.ndarray
    run_train: np.ndarray  # run index of each training trial
    ch_names: list[str]
    sfreq: float
    times: np.ndarray  # seconds relative to cue


def _paradigm(cfg: EpochConfig) -> MotorImagery:
    return MotorImagery(
        n_classes=4,
        events=CLASSES,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        resample=cfg.resample,
    )


def load_epochs(subject: int, cfg: EpochConfig) -> tuple[mne.Epochs, np.ndarray, pd.DataFrame]:
    """Return MNE epochs (time axis shifted so the cue is at 0), labels and metadata."""
    epochs, labels, meta = _paradigm(cfg).get_data(
        BNCI2014_001(), subjects=[subject], return_epochs=True
    )
    epochs.shift_time(-CUE_ONSET, relative=True)
    y = encode_labels(labels)
    return epochs, y, meta.reset_index(drop=True)


def encode_labels(labels) -> np.ndarray:
    """Map class names to fixed integers (order of ``CLASSES``, not alphabetical)."""
    mapping = {name: i for i, name in enumerate(CLASSES)}
    return np.array([mapping[str(lab)] for lab in labels], dtype=np.int64)


def _load_filterbank(subject: int, cfg: EpochConfig):
    """Filter-bank arrays: (n_trials, n_bands, n_channels, n_times) in microvolts.

    Unlike ``mne.Epochs`` (volts), MOABB's array output is already in microvolts.
    """
    paradigm = FilterBankMotorImagery(
        n_classes=4,
        events=CLASSES,
        filters=[list(b) for b in cfg.bands],
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        resample=cfg.resample,
    )
    X, labels, meta = paradigm.get_data(BNCI2014_001(), subjects=[subject])
    if X.ndim == 3:  # MOABB drops the band axis when there is a single band
        X = X[..., np.newaxis]
    X = np.moveaxis(X, -1, 1).astype(np.float32)
    sfreq = cfg.resample or RAW_SFREQ
    times = cfg.tmin + np.arange(X.shape[-1]) / sfreq
    return X, encode_labels(labels), meta.reset_index(drop=True), list(EEG_CHANNELS), sfreq, times


def load_subject(subject: int, cfg: EpochConfig) -> SubjectData:
    """Load one subject and split it into session 1 (train) and session 2 (test)."""
    if cfg.bands:
        X, y, meta, ch_names, sfreq, times = _load_filterbank(subject, cfg)
    else:
        epochs, y, meta = load_epochs(subject, cfg)
        X = epochs.get_data(copy=True).astype(np.float32) * 1e6  # V -> uV
        ch_names, sfreq, times = list(epochs.ch_names), float(epochs.info["sfreq"]), epochs.times.copy()
        assert ch_names == EEG_CHANNELS, ch_names
    is_train = (meta["session"] == SESSION_TRAIN).to_numpy()
    is_test = (meta["session"] == SESSION_TEST).to_numpy()
    assert is_train.sum() + is_test.sum() == len(y), "unexpected session names"
    return SubjectData(
        X_train=X[is_train],
        y_train=y[is_train],
        X_test=X[is_test],
        y_test=y[is_test],
        run_train=meta.loc[is_train, "run"].astype(int).to_numpy(),
        ch_names=ch_names,
        sfreq=sfreq,
        times=times,
    )


def train_val_split(
    y: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split of *training-session* indices into train / validation.

    The validation set is used for early stopping and hyper-parameter choice;
    the test session is never touched until the final evaluation.
    """
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=val_fraction, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(va)


class ChannelStandardizer:
    """Per-channel z-score whose statistics come only from the data passed to ``fit``.

    Mean and std are computed per channel over all trials and time samples of
    the training set, then reused unchanged for validation and test data.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("ChannelStandardizer must be fitted before transform")
        return ((X - self.mean_) / (self.std_ + self.eps)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def summarize(subjects=SUBJECTS, cfg: EpochConfig | None = None) -> pd.DataFrame:
    """Download (if needed) every subject and count trials per session and class."""
    cfg = cfg or EpochConfig()
    rows = []
    for s in subjects:
        _, y, meta = load_epochs(s, cfg)
        for session in (SESSION_TRAIN, SESSION_TEST):
            mask = (meta["session"] == session).to_numpy()
            counts = np.bincount(y[mask], minlength=len(CLASSES))
            rows.append(
                {"subject": s, "session": session, "n_runs": meta.loc[mask, "run"].nunique(),
                 "n_trials": int(mask.sum()), **dict(zip(CLASSES, counts.tolist()))}
            )
        print(f"subject {s}: ok")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BNCI2014_001 and summarize trials.")
    parser.add_argument("--out", default="results/data_summary.csv")
    args = parser.parse_args()
    df = summarize()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nMNE data directory: {mne.get_config('MNE_DATA') or Path.home() / 'mne_data'}")


if __name__ == "__main__":
    main()
