"""Generate a SYNTHETIC feedback log, so the learning step is demonstrable.

No real players have used this yet, so there is no real log. This writes a
fake one with a known structure: some attributes are made deliberately
unreliable and others deliberately solid, so running
scripts/learn_from_feedback.py against it should recover roughly those rates.

That is the point - it tests whether the estimator can find a signal that was
planted on purpose. It is not evidence about real players, and the output
file says so.

Run:  .venv/Scripts/python scripts/make_sample_feedback.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import MAYBE, BeliefEngine  # noqa: E402

OUT = cfg.DATA_DIR / "sample_feedback_log.csv"
N_GAMES = 400
SEED = 7

# Attributes people genuinely argue about. These get a high planted error
# rate; everything else gets a low one.
VAGUE = {
    "is_big": 0.35,
    "is_necessary": 0.30,
    "used_daily": 0.28,
    "is_reusable": 0.25,
    "is_purchased": 0.22,
    "can_make_sound": 0.20,
    "is_beauty": 0.20,
}
SOLID_RATE = 0.04        # unambiguous attributes like is_animal
MAYBE_RATE = 0.06

# Objects a player might think of that the dataset has never heard of.
UNKNOWN_OBJECTS = ["Camel", "Guitar_Pick", "Stapler", "Camel", "Stapler",
                   "Camel"]


def main() -> int:
    protos = load_prototypes()
    rng = random.Random(SEED)
    fidx = {f: i for i, f in enumerate(protos.features)}
    t0 = datetime.now(timezone.utc) - timedelta(days=30)

    rows = []
    for g in range(N_GAMES):
        use_unknown = g % 40 == 39 and UNKNOWN_OBJECTS
        if use_unknown:
            actual = UNKNOWN_OBJECTS[(g // 40) % len(UNKNOWN_OBJECTS)]
            # a player thinking of something unlisted: answers come from a
            # random real object, which is the best proxy available here
            proxy = rng.randrange(protos.n_objects)
        else:
            actual = protos.names[rng.randrange(protos.n_objects)]
            proxy = protos.index_of(actual)

        eng = BeliefEngine(protos)
        answers: dict[str, int | None] = {}
        for _ in range(rng.randint(8, 16)):
            feat = eng.best_question()
            if feat is None:
                break
            truth = int(protos.X[proxy, fidx[feat]])
            if rng.random() < MAYBE_RATE:
                eng.answer(feat, MAYBE)
                answers[feat] = None
                continue
            rate = VAGUE.get(feat, SOLID_RATE)
            given = 1 - truth if rng.random() < rate else truth
            eng.answer(feat, given)
            answers[feat] = given

        guess, _ = eng.best_guess()
        # Both outcomes are recorded. Logging only losses would bias the
        # learned error rates upward, because games are lost precisely when
        # answers were wrong.
        rows.append([
            (t0 + timedelta(minutes=7 * g)).isoformat(timespec="seconds"),
            "correct" if guess == actual else "wrong",
            guess, actual, len(answers), json.dumps(answers),
        ])

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "outcome", "guessed", "actual",
                    "n_answers", "answers_json"])
        w.writerows(rows)

    print("wrote %s" % OUT.relative_to(cfg.ROOT))
    won = sum(1 for r in rows if r[1] == "correct")
    print("  %d rows (synthetic - generated, not collected from players)" % len(rows))
    print("  %d games the engine won, %d it lost - both are logged, so the"
          % (won, len(rows) - won))
    print("  estimated rates are not biased by only seeing failures")
    print("  planted error rates: %s"
          % ", ".join("%s %.0f%%" % (k, v * 100) for k, v in VAGUE.items()))
    print("  every other attribute: %.0f%%" % (SOLID_RATE * 100))
    print("  unlisted objects seeded: %s" % sorted(set(UNKNOWN_OBJECTS)))
    print()
    print("Now run: python scripts/learn_from_feedback.py --evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
