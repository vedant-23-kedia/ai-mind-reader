"""Tests for the Bayesian belief engine.

The claims Phase 4 rests on are all here: nothing is ever eliminated, a wrong
answer is recoverable, "maybe" is a no-op, undo is exact, and the confidence
figure is a real probability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import (  # noqa: E402
    MAYBE,
    NO,
    YES,
    BeliefEngine,
)


@pytest.fixture(scope="module")
def protos():
    return load_prototypes()


@pytest.fixture
def engine(protos):
    return BeliefEngine(protos)


def truthful_answer(protos, target: str, feature: str) -> int:
    i = protos.index_of(target)
    k = protos.features.index(feature)
    return int(protos.X[i, k])


# ---- basic properties ---------------------------------------------------

def test_posterior_is_a_distribution(engine):
    p = engine.posterior()
    assert p.shape == (engine.protos.n_objects,)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


def test_prior_is_uniform_before_any_answer(engine):
    p = engine.posterior()
    assert np.allclose(p, 1.0 / engine.protos.n_objects)


def test_entropy_starts_at_log2_n(engine):
    assert engine.entropy() == pytest.approx(
        np.log2(engine.protos.n_objects), abs=1e-9)


def test_nothing_is_ever_eliminated(engine, protos):
    """The core difference from hard filtering: no object hits zero."""
    for feat in protos.features[:40]:
        engine.answer(feat, YES)
    p = engine.posterior()
    assert (p > 0).all(), "an object was eliminated outright"


def test_rejects_invalid_noise(protos):
    for bad in (0.0, 0.5, 0.9, -0.1):
        with pytest.raises(ValueError, match="noise must be"):
            BeliefEngine(protos, noise=bad)


def test_rejects_unknown_attribute_and_repeat_answers(engine, protos):
    with pytest.raises(KeyError):
        engine.answer("not_a_real_attribute", YES)
    f = protos.features[0]
    engine.answer(f, YES)
    with pytest.raises(ValueError, match="already been answered"):
        engine.answer(f, NO)


# ---- maybe --------------------------------------------------------------

def test_maybe_leaves_the_belief_state_untouched(engine, protos):
    before = engine.posterior().copy()
    engine.answer(protos.features[0], MAYBE)
    engine.answer(protos.features[1], MAYBE)
    assert np.allclose(engine.posterior(), before), (
        "MAYBE must carry no evidence")


def test_maybe_still_counts_as_asked(engine, protos):
    f = protos.features[0]
    engine.answer(f, MAYBE)
    assert f in engine.asked()
    assert f not in engine.information_gain(), (
        "an answered question must not be offered again")


# ---- undo ---------------------------------------------------------------

def test_undo_restores_the_previous_state_exactly(engine, protos):
    engine.answer(protos.features[0], YES)
    snapshot = engine.posterior().copy()
    engine.answer(protos.features[1], NO)
    assert not np.allclose(engine.posterior(), snapshot)
    engine.undo()
    assert np.allclose(engine.posterior(), snapshot)


def test_undo_on_empty_history_returns_none(engine):
    assert engine.undo() is None


# ---- convergence --------------------------------------------------------

@pytest.mark.parametrize(
    "target", ["Dog", "Laptop", "Spider_Man", "Socks", "Moon", "Broom"])
def test_truthful_play_identifies_the_object(protos, target):
    eng = BeliefEngine(protos)
    for _ in range(25):
        f = eng.best_question()
        if f is None or eng.is_confident():
            break
        eng.answer(f, truthful_answer(protos, target, f))
    guess, conf = eng.best_guess()
    assert guess == target, "guessed %s (%.2f) instead of %s" % (
        guess, conf, target)


def test_every_object_is_identified_by_truthful_play(protos):
    """All 117, not just a sample."""
    wrong = []
    questions = []
    for target in protos.names:
        eng = BeliefEngine(protos)
        for _ in range(30):
            f = eng.best_question()
            if f is None or eng.is_confident():
                break
            eng.answer(f, truthful_answer(protos, target, f))
        guess, _ = eng.best_guess()
        questions.append(len(eng.answers))
        if guess != target:
            wrong.append((target, guess))
    assert not wrong, "misidentified under truthful play: %s" % wrong[:5]
    # log2(117) ~= 6.9, so a good questioner should be close to that
    assert np.mean(questions) < 12, (
        "averaged %.1f questions, expected well under 12" % np.mean(questions))


def test_a_single_wrong_answer_is_recoverable(protos):
    """The whole point. Hard filtering could never do this."""
    target = "Dog"
    eng = BeliefEngine(protos)

    # Lie about the very first question, which is the most damaging moment.
    first = eng.best_question()
    eng.answer(first, 1 - truthful_answer(protos, target, first))

    # Then answer everything else honestly.
    for _ in range(40):
        f = eng.best_question()
        if f is None:
            break
        eng.answer(f, truthful_answer(protos, target, f))
        if eng.is_confident() and eng.best_guess()[0] == target:
            break

    guess, conf = eng.best_guess()
    assert guess == target, (
        "one wrong answer was unrecoverable: ended on %s (%.2f)" % (guess, conf))


def test_target_stays_reachable_after_several_lies(protos):
    target = "Laptop"
    eng = BeliefEngine(protos)
    for _ in range(3):
        f = eng.best_question()
        eng.answer(f, 1 - truthful_answer(protos, target, f))
    p = eng.posterior()
    assert p[protos.index_of(target)] > 0, "lies must not zero out the truth"


# ---- information gain ---------------------------------------------------

def test_information_gain_is_non_negative(engine):
    gains = engine.information_gain()
    assert gains
    worst = min(gains.values())
    assert worst > -1e-9, "negative expected information gain: %r" % worst


def test_information_gain_excludes_answered_questions(engine, protos):
    f = protos.features[3]
    engine.answer(f, YES)
    assert f not in engine.information_gain()


def test_best_question_beats_a_near_constant_attribute(engine):
    """is_reusable is true for 99 of 117 objects, so it is a poor opener."""
    gains = engine.information_gain()
    assert gains["is_reusable"] < gains[engine.best_question()]


def _binary_entropy(q: float) -> float:
    return float(-(q * np.log2(q) + (1 - q) * np.log2(1 - q)))


def test_no_question_can_exceed_the_noisy_channel_capacity(engine):
    """A yes/no answer that is wrong 10% of the time cannot carry a full bit.

    The ceiling is 1 - H(noise) bits, and it is the noise - not the dataset -
    that sets it. At noise=0.1 that is 0.531 bits, so identifying one of 117
    objects needs at least log2(117)/0.531 ~= 13 questions in theory, rather
    than the 6.9 a noiseless questioner would need.
    """
    capacity = 1.0 - _binary_entropy(engine.noise)
    gains = engine.information_gain()
    worst_violation = max(gains.values()) - capacity
    assert worst_violation < 1e-9, (
        "a question claimed %.4f bits, above the %.4f-bit ceiling"
        % (max(gains.values()), capacity))


def test_best_opening_question_is_near_optimal(engine):
    """The chosen opener should be close to the ceiling, not merely positive."""
    capacity = 1.0 - _binary_entropy(engine.noise)
    best = max(engine.information_gain().values())
    assert best > 0.95 * capacity, (
        "best opener worth %.4f bits of a possible %.4f" % (best, capacity))


def test_answers_reduce_entropy_on_average(protos):
    eng = BeliefEngine(protos)
    before = eng.entropy()
    for _ in range(6):
        f = eng.best_question()
        eng.answer(f, truthful_answer(protos, "Dog", f))
    assert eng.entropy() < before


# ---- confidence --------------------------------------------------------

def test_confidence_is_a_probability_not_a_score(protos):
    eng = BeliefEngine(protos)
    for _ in range(8):
        f = eng.best_question()
        eng.answer(f, truthful_answer(protos, "Moon", f))
    _, conf = eng.best_guess()
    assert 0.0 <= conf <= 1.0
    assert np.isclose(sum(p for _, p in eng.top(eng.protos.n_objects)), 1.0)


def test_top_k_is_ordered_and_sized(protos):
    eng = BeliefEngine(protos)
    eng.answer("is_animal", YES)
    top = eng.top(5)
    assert len(top) == 5
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)


def test_contradictions_explain_a_low_ranking(protos):
    eng = BeliefEngine(protos)
    eng.answer("is_animal", YES)
    # Laptop is not an animal, so that answer should be listed against it.
    assert "is_animal" in eng.contradictions("Laptop")
    assert eng.contradictions("Dog") == []


# ---- likelihood weight (the calibration knob) ---------------------------

def test_weight_of_one_is_plain_bayes(protos):
    """The default must be mathematically untouched by the knob."""
    a = BeliefEngine(protos, likelihood_weight=1.0)
    b = BeliefEngine(protos)
    for f, v in [("is_animal", YES), ("is_big", NO), ("can_fly", YES)]:
        a.answer(f, v)
        b.answer(f, v)
    assert np.allclose(a.posterior(), b.posterior())


def test_weight_rejects_out_of_range_values(protos):
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="likelihood_weight"):
            BeliefEngine(protos, likelihood_weight=bad)


@pytest.mark.parametrize("w", [1.0, 0.8, 0.5, 0.3])
def test_information_gain_stays_valid_at_any_weight(protos, w):
    """Gain must stay non-negative and under the channel ceiling.

    A lower weight raises both answer likelihoods to a power, which breaks
    their sum to 1; if that is not renormalised the 'expected' entropy is not
    an expectation and the gain can exceed what a single answer can carry.
    """
    eng = BeliefEngine(protos, likelihood_weight=w)
    eng.answer("is_animal", YES)
    gains = eng.information_gain()
    capacity = 1.0 - _binary_entropy(eng.noise)
    assert min(gains.values()) > -1e-9, "negative gain at weight %s" % w
    assert max(gains.values()) <= capacity + 1e-6, (
        "gain %.4f exceeds the %.4f-bit ceiling at weight %s"
        % (max(gains.values()), capacity, w))


def test_lower_weight_makes_the_posterior_less_peaked(protos):
    """That is the knob's whole purpose: discount each answer."""
    sharp = BeliefEngine(protos, likelihood_weight=1.0)
    soft = BeliefEngine(protos, likelihood_weight=0.5)
    for eng in (sharp, soft):
        for f, v in [("is_animal", YES), ("is_mammal", YES), ("is_big", YES)]:
            eng.answer(f, v)
    assert soft.posterior().max() < sharp.posterior().max()
    assert soft.entropy() > sharp.entropy()
