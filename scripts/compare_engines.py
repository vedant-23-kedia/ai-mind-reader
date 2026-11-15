"""Old engine vs new engine, playing full games against simulated players.

The old pipeline is reimplemented here faithfully - hard candidate filtering,
the match-minus-contradiction score, and the tiered/shuffled question order
with an information-gain switch below 10 candidates - so the comparison is
against what the project actually did, not a strawman.

A game is one simulated player who answers every question with a fixed
probability of being wrong. Reported: how often the final guess is right,
how often the right answer is in the top 3, and how many questions it took.

Run:  .venv/Scripts/python scripts/compare_engines.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import MAYBE, BeliefEngine  # noqa: E402

MAX_QUESTIONS = 20
GAMES_PER_OBJECT = 8
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
MAYBE_RATE = 0.05          # how often a simulated player says "maybe"

# Tier 1 from the original app: the broad questions it always asked first.
OLD_TIER1 = ["is_living", "is_animal", "is_character", "made_by_humans",
             "is_fictional", "is_natural"]
OLD_TIER2 = ["is_electronic", "is_wearable", "is_vehicle", "is_food",
             "is_edible", "is_marvel", "is_cartoons", "can_fly", "can_swim",
             "is_big", "used_daily"]


def old_question_order(features, rng):
    t1 = [f for f in OLD_TIER1 if f in features]
    t2 = [f for f in OLD_TIER2 if f in features]
    t3 = [f for f in features if f not in t1 and f not in t2]
    rng.shuffle(t2)
    rng.shuffle(t3)
    return t1 + t2 + t3


def old_engine_game(protos, target_idx, noise, rng):
    """The original algorithm: irreversible filtering plus a heuristic score."""
    X, names, features = protos.X, protos.names, protos.features
    order = old_question_order(features, rng)
    fidx = {f: i for i, f in enumerate(features)}

    answers: dict[str, float] = {}
    candidates = list(range(len(names)))
    ptr = 0
    asked = 0

    while asked < MAX_QUESTIONS:
        # guess once a single candidate survives
        if len(candidates) == 1 and asked >= 3:
            break

        # pick the next question
        feat = None
        if 1 < len(candidates) <= 10:
            best, best_split = None, 10 ** 9
            for f in order:
                if f in answers:
                    continue
                k = fidx[f]
                yes = sum(1 for c in candidates if X[c, k] == 1)
                split = abs(yes - (len(candidates) - yes))
                if split < best_split:
                    best, best_split = f, split
            feat = best
        if feat is None:
            while ptr < len(order) and order[ptr] in answers:
                ptr += 1
            if ptr >= len(order):
                break
            feat = order[ptr]

        # simulated player answers
        k = fidx[feat]
        truth = int(X[target_idx, k])
        if rng.random() < MAYBE_RATE:
            ans = 0.5
        else:
            ans = float(1 - truth if rng.random() < noise else truth)
        answers[feat] = ans
        asked += 1

        # hard filtering: anything contradicting the answer is struck off
        if ans in (0.0, 1.0):
            candidates = [c for c in candidates if X[c, k] == int(ans)]

    # final guess: match minus contradiction score over whatever survived
    pool = candidates if candidates else list(range(len(names)))
    scored = []
    for c in pool:
        match = contra = total = 0
        for f, a in answers.items():
            if a == 0.5:
                continue
            total += 1
            if int(X[c, fidx[f]]) == int(a):
                match += 1
            else:
                contra += 1
        score = ((match - contra * 0.5) / total * 100) if total else 50.0
        scored.append((max(score, 0.0), c))
    scored.sort(reverse=True)
    ranked = [c for _, c in scored]
    return ranked[:3], asked


def new_engine_game(protos, target_idx, noise, rng):
    """The Bayesian engine: belief updating plus expected information gain."""
    eng = BeliefEngine(protos)
    X = protos.X
    fidx = {f: i for i, f in enumerate(protos.features)}
    asked = 0

    while asked < MAX_QUESTIONS and not eng.is_confident():
        feat = eng.best_question()
        if feat is None:
            break
        truth = int(X[target_idx, fidx[feat]])
        if rng.random() < MAYBE_RATE:
            eng.answer(feat, MAYBE)
        else:
            eng.answer(feat, 1 - truth if rng.random() < noise else truth)
        asked += 1

    top3 = [protos.index_of(n) for n, _ in eng.top(3)]
    return top3, asked


def main() -> int:
    protos = load_prototypes()
    print("=" * 78)
    print("OLD ENGINE vs NEW ENGINE - %d games per object, %d objects, "
          "max %d questions" % (GAMES_PER_OBJECT, protos.n_objects, MAX_QUESTIONS))
    print("simulated players also say 'maybe' %.0f%% of the time"
          % (MAYBE_RATE * 100))
    print("=" * 78)
    print()
    print("  %-8s | %-24s | %-24s" % ("wrong", "OLD (hard filtering)",
                                      "NEW (Bayesian)"))
    print("  %-8s | %7s %7s %7s | %7s %7s %7s"
          % ("answers", "top1", "top3", "quest", "top1", "top3", "quest"))
    print("  " + "-" * 74)

    summary = {}
    for noise in NOISE_LEVELS:
        stats = {}
        for label, fn in (("old", old_engine_game), ("new", new_engine_game)):
            hit1 = hit3 = 0
            qs = []
            rng = random.Random(1234)
            for ti in range(protos.n_objects):
                for _ in range(GAMES_PER_OBJECT):
                    top3, asked = fn(protos, ti, noise, rng)
                    if top3 and top3[0] == ti:
                        hit1 += 1
                    if ti in top3:
                        hit3 += 1
                    qs.append(asked)
            n = protos.n_objects * GAMES_PER_OBJECT
            stats[label] = (100 * hit1 / n, 100 * hit3 / n, float(np.mean(qs)))
        o, nw = stats["old"], stats["new"]
        summary[noise] = stats
        print("  %7.0f%% | %6.1f%% %6.1f%% %7.1f | %6.1f%% %6.1f%% %7.1f"
              % (noise * 100, o[0], o[1], o[2], nw[0], nw[1], nw[2]))

    print()
    print("=" * 78)
    print("  change in top-1 accuracy, new minus old:")
    for noise, s in summary.items():
        d = s["new"][0] - s["old"][0]
        print("    %3.0f%% wrong answers: %+6.1f points   (%.1f%% -> %.1f%%)"
              % (noise * 100, d, s["old"][0], s["new"][0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
