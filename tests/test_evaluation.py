"""Tests for the evaluation maths.

The calibration numbers are a headline claim, so the code that computes them
needs checking against cases with known answers.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.evaluation import (  # noqa: E402
    all_steps,
    calibration_curve,
    expected_calibration_error,
    overconfidence,
    play_game,
    run_games,
    summarise,
)


@pytest.fixture(scope="module")
def protos():
    return load_prototypes()


# ---- calibration maths on hand-built cases ------------------------------

def test_perfect_calibration_scores_zero_ece():
    """Claim 100% and always be right; claim 0% and never be right."""
    pairs = [(1.0, True)] * 50 + [(0.0, False)] * 50
    assert expected_calibration_error(pairs) == pytest.approx(0.0, abs=1e-9)
    assert overconfidence(pairs) == pytest.approx(0.0, abs=1e-9)


def test_total_overconfidence_scores_one():
    """Claim certainty and always be wrong."""
    pairs = [(1.0, False)] * 20
    assert expected_calibration_error(pairs) == pytest.approx(1.0)
    assert overconfidence(pairs) == pytest.approx(1.0)


def test_underconfidence_has_a_negative_sign():
    pairs = [(0.2, True)] * 20
    assert overconfidence(pairs) == pytest.approx(-0.8)
    # ECE is unsigned, so it cannot tell the two apart
    assert expected_calibration_error(pairs) == pytest.approx(0.8)


def test_half_confidence_half_right_is_calibrated():
    pairs = [(0.5, True)] * 50 + [(0.5, False)] * 50
    assert expected_calibration_error(pairs) == pytest.approx(0.0, abs=1e-9)


def test_calibration_curve_bins_and_counts():
    pairs = [(0.05, False), (0.15, False), (0.95, True), (0.96, True)]
    rows = calibration_curve(pairs, n_bins=10)
    assert sum(r["count"] for r in rows) == len(pairs), "a prediction was dropped"
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
    assert all(r["bin_low"] <= r["mean_confidence"] <= r["bin_high"] for r in rows)


def test_confidence_of_exactly_one_is_not_dropped():
    """The top bin must be closed, or perfect certainty vanishes."""
    rows = calibration_curve([(1.0, True)], n_bins=10)
    assert sum(r["count"] for r in rows) == 1


def test_empty_input_is_handled():
    assert calibration_curve([]) == []
    import math
    assert math.isnan(expected_calibration_error([]))


# ---- game simulation ----------------------------------------------------

def test_truthful_game_finds_the_object(protos):
    rng = random.Random(0)
    r = play_game(protos, protos.index_of("Dog"), noise=0.0, rng=rng,
                  maybe_rate=0.0)
    assert r.correct
    assert r.guess == "Dog"
    assert r.questions > 0
    assert r.asked_features, "no attributes recorded as asked"
    assert len(set(r.asked_features)) == len(r.asked_features), "asked twice"


def test_steps_are_recorded_per_question(protos):
    rng = random.Random(0)
    r = play_game(protos, 0, noise=0.1, rng=rng)
    assert len(r.steps) == r.questions
    assert all(0.0 <= c <= 1.0 for c, _ in r.steps)


def test_zero_noise_across_all_objects_is_perfect(protos):
    res = run_games(protos, noise=0.0, games_per_object=1, maybe_rate=0.0)
    wrong = [(r.target, r.guess) for r in res if not r.correct]
    assert not wrong, "failed on truthful play: %s" % wrong[:5]


def test_accuracy_falls_as_the_player_gets_worse(protos):
    accs = []
    for noise in (0.0, 0.1, 0.2):
        res = run_games(protos, noise=noise, games_per_object=2)
        accs.append(summarise(res)["top1"])
    assert accs == sorted(accs, reverse=True), accs


def test_top3_is_never_worse_than_top1(protos):
    res = run_games(protos, noise=0.15, games_per_object=2)
    s = summarise(res)
    assert s["top3"] >= s["top1"]


def test_engine_noise_can_differ_from_player_noise(protos):
    """Needed for the sensitivity analysis: the engine's assumption is a knob."""
    rng = random.Random(3)
    r = play_game(protos, 0, noise=0.2, rng=rng, engine_noise=0.05)
    assert r.questions > 0


def test_all_steps_spans_a_wide_confidence_range(protos):
    """Final-guess confidences cluster near the threshold; steps should not."""
    res = run_games(protos, noise=0.1, games_per_object=2)
    pairs = all_steps(res)
    confs = [c for c, _ in pairs]
    assert min(confs) < 0.2, "no low-confidence predictions to calibrate against"
    assert max(confs) > 0.8
