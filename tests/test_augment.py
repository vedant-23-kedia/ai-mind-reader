"""Tests for the player-noise simulation.

The whole of Phase 3 rests on this being correct: if the noise model is
broken, the accuracy figures are meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader.augment import NOISE_LEVELS, make_noisy, top_k_accuracy  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402


@pytest.fixture(scope="module")
def protos():
    return load_prototypes()


def test_prototypes_are_binary(protos):
    assert np.isin(protos.X, (0, 1)).all()
    assert protos.X.shape == (protos.n_objects, protos.n_features)


def test_shape_and_labels(protos):
    ds = make_noisy(protos.X, 10, NOISE_LEVELS, seed=0)
    assert ds.X.shape == (protos.n_objects * 10, protos.n_features)
    assert ds.y.shape == (protos.n_objects * 10,)
    # every object equally represented
    counts = np.bincount(ds.y, minlength=protos.n_objects)
    assert set(counts.tolist()) == {10}


def test_output_stays_binary(protos):
    ds = make_noisy(protos.X, 10, (0.2,), seed=0)
    assert np.isin(ds.X, (0, 1)).all(), "flipping must not produce 0.5 or other values"


def test_zero_noise_reproduces_prototypes_exactly(protos):
    ds = make_noisy(protos.X, 4, (0.0,), seed=3)
    for xi, yi in zip(ds.X, ds.y):
        assert np.array_equal(xi, protos.X[yi]), "0%% noise altered an answer"


def test_deterministic_for_a_given_seed(protos):
    a = make_noisy(protos.X, 6, NOISE_LEVELS, seed=11)
    b = make_noisy(protos.X, 6, NOISE_LEVELS, seed=11)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)


def test_different_seeds_differ(protos):
    a = make_noisy(protos.X, 6, (0.2,), seed=1)
    b = make_noisy(protos.X, 6, (0.2,), seed=2)
    assert not np.array_equal(a.X, b.X)


def test_observed_flip_rate_matches_requested(protos):
    """The realised noise must be close to the requested probability."""
    for level in (0.05, 0.2):
        ds = make_noisy(protos.X, 60, (level,), seed=5)
        differing = (ds.X != protos.X[ds.y]).mean()
        assert abs(differing - level) < 0.01, (
            "requested %.2f flips, observed %.4f" % (level, differing))


def test_noise_column_records_the_level_used(protos):
    ds = make_noisy(protos.X, 10, NOISE_LEVELS, seed=0)
    assert set(np.unique(ds.noise).tolist()) == set(NOISE_LEVELS)
    for level in NOISE_LEVELS:
        Xl, yl = ds.at_noise(level)
        assert len(Xl) == len(yl) > 0


def test_rejects_too_few_samples_for_the_levels(protos):
    with pytest.raises(ValueError, match="at least the number of noise"):
        make_noisy(protos.X, 2, NOISE_LEVELS, seed=0)


def test_top_k_accuracy():
    # 3 samples, 4 classes; true classes 0, 1, 3
    proba = np.array([
        [0.7, 0.1, 0.1, 0.1],   # rank 1 correct
        [0.5, 0.3, 0.1, 0.1],   # true class 1 is 2nd
        [0.4, 0.3, 0.2, 0.1],   # true class 3 is 4th
    ])
    y = np.array([0, 1, 3])
    assert top_k_accuracy(proba, y, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(proba, y, 2) == pytest.approx(2 / 3)
    assert top_k_accuracy(proba, y, 4) == 1.0


def test_top_k_is_monotonic_in_k():
    rng = np.random.default_rng(0)
    proba = rng.random((50, 10))
    proba /= proba.sum(axis=1, keepdims=True)
    y = rng.integers(0, 10, size=50)
    scores = [top_k_accuracy(proba, y, k) for k in (1, 2, 3, 5)]
    assert scores == sorted(scores), "top-k accuracy must not decrease with k"
