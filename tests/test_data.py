"""Tests for the data pipeline.

Fast tests use synthetic arrays. Tests marked ``data`` load subject 1 through
MOABB (downloading ~80 MB on first use); skip them with ``pytest -m "not data"``.
"""

import numpy as np
import pytest

from src.data import (
    CLASSES,
    ChannelStandardizer,
    EpochConfig,
    encode_labels,
    load_subject,
    train_val_split,
)


def test_encode_labels_uses_fixed_class_order():
    y = encode_labels(["tongue", "left_hand", "feet", "right_hand"])
    assert y.tolist() == [3, 0, 2, 1]
    assert CLASSES == ["left_hand", "right_hand", "feet", "tongue"]


def test_standardizer_uses_only_training_statistics():
    rng = np.random.default_rng(0)
    X_train = rng.normal(5.0, 2.0, size=(40, 3, 100))
    X_test = rng.normal(-3.0, 10.0, size=(10, 3, 100))  # very different distribution
    sc = ChannelStandardizer().fit(X_train)
    mean_before, std_before = sc.mean_.copy(), sc.std_.copy()

    Z_train = sc.transform(X_train)
    Z_test = sc.transform(X_test)

    np.testing.assert_allclose(Z_train.mean(axis=(0, 2)), 0.0, atol=1e-5)
    np.testing.assert_allclose(Z_train.std(axis=(0, 2)), 1.0, atol=1e-4)
    # transforming test data must not change the fitted statistics ...
    np.testing.assert_array_equal(sc.mean_, mean_before)
    np.testing.assert_array_equal(sc.std_, std_before)
    # ... and test data is NOT re-centred (it keeps its own shift)
    assert Z_test.mean() < -3.0


def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        ChannelStandardizer().transform(np.zeros((1, 1, 1)))


def test_train_val_split_is_stratified_disjoint_and_deterministic():
    y = np.repeat(np.arange(4), 72)  # 288 trials, like one session
    tr, va = train_val_split(y, val_fraction=0.2, seed=0)
    assert len(np.intersect1d(tr, va)) == 0
    assert len(tr) + len(va) == len(y)
    assert np.ptp(np.bincount(y[va])) <= 1  # classes balanced in the validation set
    tr2, va2 = train_val_split(y, val_fraction=0.2, seed=0)
    np.testing.assert_array_equal(va, va2)
    _, va3 = train_val_split(y, val_fraction=0.2, seed=1)
    assert not np.array_equal(va, va3)


@pytest.fixture(scope="module")
def subject1():
    return load_subject(1, EpochConfig(fmin=4.0, fmax=40.0, tmin=0.5, tmax=2.5))


@pytest.mark.data
def test_subject_shapes_and_sessions(subject1):
    d = subject1
    # 288 trials per session, 22 EEG channels (EOG excluded), 2 s at 250 Hz (inclusive end)
    assert d.X_train.shape == (288, 22, 501)
    assert d.X_test.shape == (288, 22, 501)
    assert len(d.ch_names) == 22 and not any("EOG" in c for c in d.ch_names)
    assert {"C3", "Cz", "C4"} <= set(d.ch_names)
    assert np.bincount(d.y_train).tolist() == [72, 72, 72, 72]
    assert np.bincount(d.y_test).tolist() == [72, 72, 72, 72]
    assert sorted(set(d.run_train.tolist())) == [0, 1, 2, 3, 4, 5]


@pytest.mark.data
def test_subject_time_axis_is_relative_to_cue(subject1):
    assert subject1.sfreq == 250.0
    assert subject1.times[0] == pytest.approx(0.5)
    assert subject1.times[-1] == pytest.approx(2.5)


@pytest.mark.data
def test_subject_units_are_microvolts(subject1):
    # Band-passed scalp EEG has a standard deviation of a few to tens of uV.
    std = subject1.X_train.std()
    assert 1.0 < std < 100.0
    assert subject1.X_train.dtype == np.float32


@pytest.mark.data
def test_filter_bank_matches_single_band_path(subject1):
    """MOABB arrays are in uV but mne.Epochs are in V: both load paths must agree."""
    single = load_subject(1, EpochConfig(fmin=8.0, fmax=12.0, tmin=0.5, tmax=2.5))
    bank = load_subject(1, EpochConfig(tmin=0.5, tmax=2.5, bands=((8.0, 12.0), (12.0, 16.0))))
    assert bank.X_train.shape == (288, 2, 22, 501)
    assert bank.ch_names == single.ch_names
    np.testing.assert_allclose(bank.X_train[:, 0], single.X_train, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(bank.y_test, single.y_test)


@pytest.mark.data
def test_resampling_to_128_hz():
    d = load_subject(1, EpochConfig(fmin=4.0, fmax=40.0, tmin=0.5, tmax=2.5, resample=128.0))
    assert d.sfreq == 128.0
    assert d.X_train.shape == (288, 22, 257)
