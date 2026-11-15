"""Tests for the trained object-level models.

Skipped cleanly if training has not been run yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.augment import make_noisy  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402

joblib = pytest.importorskip("joblib")

INFO = cfg.MODELS_DIR / "object_model_info.json"
BEST = cfg.MODELS_DIR / "object_best_model.pkl"

pytestmark = pytest.mark.skipif(
    not (INFO.exists() and BEST.exists()),
    reason="object models not trained yet - run scripts/train_models.py",
)


@pytest.fixture(scope="module")
def protos():
    return load_prototypes()


@pytest.fixture(scope="module")
def info():
    return json.loads(INFO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return joblib.load(BEST)


def test_model_classes_match_dataset_objects(info, protos):
    assert list(info["classes"]) == protos.names


def test_model_features_match_dataset_attributes(info, protos):
    assert list(info["features"]) == protos.features


def test_model_predicts_objects_not_categories(model, protos):
    """The original models predicted 9 categories; these must predict objects."""
    assert len(model.classes_) == protos.n_objects, (
        "model has %d classes, expected %d objects"
        % (len(model.classes_), protos.n_objects))


def test_identifies_every_object_from_clean_prototype(model, protos):
    pred = model.predict(protos.X.astype(np.int8))
    wrong = [(protos.names[i], protos.names[int(p)])
             for i, p in enumerate(pred) if int(p) != i]
    assert not wrong, "misidentified from a perfect answer sheet: %s" % wrong[:5]


def test_accuracy_degrades_but_does_not_collapse(model, protos):
    """A 5%-wrong player should still be identified most of the time."""
    ds = make_noisy(protos.X, 20, (0.05,), seed=99)
    acc = float(np.mean(model.predict(ds.X) == ds.y))
    assert acc > 0.85, "only %.1f%% accurate at a 5%% flip rate" % (100 * acc)


def test_accuracy_is_monotonic_in_noise(model, protos):
    accs = []
    for lvl in (0.0, 0.05, 0.10, 0.15, 0.20):
        ds = make_noisy(protos.X, 20, (lvl,), seed=4)
        accs.append(float(np.mean(model.predict(ds.X) == ds.y)))
    assert accs == sorted(accs, reverse=True), (
        "accuracy should fall as noise rises, got %s" % accs)


def test_bernoulli_beats_gaussian_nb(info):
    """Binary attributes need a Bernoulli likelihood, not a Gaussian one.

    This was the point of switching variants, so it is worth asserting
    rather than trusting.
    """
    r = info["results"]
    if "bernoulli_nb" not in r or "gaussian_nb" not in r:
        pytest.skip("both Naive Bayes variants not present in results")
    assert r["bernoulli_nb"]["test_top1"] > r["gaussian_nb"]["test_top1"]


def test_reported_results_are_plausible(info):
    for name, r in info["results"].items():
        assert 0 <= r["test_top1"] <= 100
        assert r["test_top3"] >= r["test_top1"] - 1e-6, name
        assert r["test_top5"] >= r["test_top3"] - 1e-6, name
        # every model should nail a perfect answer sheet: the dataset
        # guarantees all objects are distinguishable
        assert r["clean_top1"] == 100.0, "%s only %.1f%% on clean prototypes" % (
            name, r["clean_top1"])
