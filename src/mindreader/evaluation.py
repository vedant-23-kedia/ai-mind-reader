"""Measuring whether the engine is any good, and whether it knows it.

Two different questions, often confused:

  * **Accuracy** - how often the final guess is right. Phase 4 established
    this against the old engine.
  * **Calibration** - whether the confidence figure means anything. An engine
    that says 92% should be right 92% of the time. One that says 92% and is
    right 70% of the time is overconfident, and its confidence number is
    decoration.

Calibration is the more interesting of the two here, because the engine's
independence assumption is knowingly false: the dataset *enforces* rules like
is_mammal -> is_animal, so answering both counts one underlying fact twice.
Naive Bayes is expected to be overconfident under exactly that condition.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .dataset import Prototypes
from .engine import MAYBE, BeliefEngine


@dataclass
class GameResult:
    target: str
    guess: str
    confidence: float
    questions: int
    correct: bool
    in_top3: bool
    # (claimed confidence, was the top guess correct) at every step
    steps: list[tuple[float, bool]]
    # which attributes were actually asked, for redundancy analysis
    asked_features: list[str]


def play_game(
    protos: Prototypes,
    target_idx: int,
    noise: float,
    rng: random.Random,
    maybe_rate: float = 0.05,
    max_questions: int = 20,
    engine_noise: float | None = None,
    likelihood_weight: float = 1.0,
) -> GameResult:
    """One simulated player answering until the engine commits.

    `noise` is how often the player actually answers wrongly; `engine_noise`
    is what the engine *assumes*. Letting them differ is how the sensitivity
    analysis works - in reality you never know a player's error rate.
    """
    eng = BeliefEngine(
        protos,
        noise=engine_noise if engine_noise is not None else noise or 0.01,
        likelihood_weight=likelihood_weight,
    )
    fidx = {f: i for i, f in enumerate(protos.features)}
    target = protos.names[target_idx]
    steps: list[tuple[float, bool]] = []

    asked = 0
    while asked < max_questions and not eng.is_confident():
        feat = eng.best_question()
        if feat is None:
            break
        truth = int(protos.X[target_idx, fidx[feat]])
        if rng.random() < maybe_rate:
            eng.answer(feat, MAYBE)
        else:
            eng.answer(feat, 1 - truth if rng.random() < noise else truth)
        asked += 1

        guess, conf = eng.best_guess()
        steps.append((conf, guess == target))

    top = eng.top(3)
    guess, conf = top[0]
    return GameResult(
        target=target,
        guess=guess,
        confidence=float(conf),
        questions=asked,
        correct=guess == target,
        in_top3=target in [n for n, _ in top],
        steps=steps,
        asked_features=[a.feature for a in eng.answers],
    )


def run_games(
    protos: Prototypes,
    noise: float,
    games_per_object: int,
    seed: int = 1234,
    **kwargs,
) -> list[GameResult]:
    rng = random.Random(seed)
    out = []
    for ti in range(protos.n_objects):
        for _ in range(games_per_object):
            out.append(play_game(protos, ti, noise, rng, **kwargs))
    return out


def calibration_curve(
    pairs: list[tuple[float, bool]], n_bins: int = 10
) -> list[dict]:
    """Group predictions by claimed confidence and measure real accuracy.

    Returns one row per non-empty bin with the mean claimed confidence, the
    observed accuracy, and the count. A perfectly calibrated engine puts
    every row on the diagonal.
    """
    if not pairs:
        return []
    conf = np.array([c for c, _ in pairs], dtype=float)
    hit = np.array([bool(h) for _, h in pairs], dtype=bool)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # include the right edge in the final bin so 1.0 is not dropped
        m = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        rows.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "mean_confidence": float(conf[m].mean()),
            "accuracy": float(hit[m].mean()),
            "count": int(m.sum()),
        })
    return rows


def expected_calibration_error(
    pairs: list[tuple[float, bool]], n_bins: int = 10
) -> float:
    """Average gap between claimed confidence and observed accuracy.

    Weighted by how many predictions fall in each bin. 0 is perfect; 0.1
    means the confidence figure is off by 10 points on average.
    """
    rows = calibration_curve(pairs, n_bins)
    if not rows:
        return float("nan")
    total = sum(r["count"] for r in rows)
    return float(sum(
        r["count"] / total * abs(r["accuracy"] - r["mean_confidence"])
        for r in rows
    ))


def overconfidence(pairs: list[tuple[float, bool]]) -> float:
    """Mean claimed confidence minus actual accuracy.

    Positive means the engine overstates how sure it is; negative means it
    understates. Unlike ECE this keeps the sign, so it says which way the
    engine is wrong.
    """
    if not pairs:
        return float("nan")
    conf = np.array([c for c, _ in pairs], dtype=float)
    hit = np.array([bool(h) for _, h in pairs], dtype=bool)
    return float(conf.mean() - hit.mean())


def summarise(results: list[GameResult]) -> dict:
    n = len(results)
    return {
        "games": n,
        "top1": 100 * sum(r.correct for r in results) / n,
        "top3": 100 * sum(r.in_top3 for r in results) / n,
        "mean_questions": float(np.mean([r.questions for r in results])),
        "max_questions": int(np.max([r.questions for r in results])),
        "mean_confidence": 100 * float(np.mean([r.confidence for r in results])),
    }


def all_steps(results: list[GameResult]) -> list[tuple[float, bool]]:
    """Every (confidence, correct) pair from every step of every game.

    Final-guess confidences cluster near the stopping threshold, which gives
    no coverage of the low end. Using every step spans the whole range from
    1/n up to near-certainty, which is what actually tests the probability
    model rather than the stopping rule.
    """
    return [s for r in results for s in r.steps]
