"""Play the real app many times and assert nothing ever breaks.

The unit tests check pieces. This drives the actual Streamlit app the way a
person would - clicking through whole games with different answering styles -
and checks the invariants that must hold no matter how someone plays:

  * the game always terminates
  * it never raises
  * the confidence shown is always a probability
  * it always names a real object, with three ranked candidates
  * scores accumulate correctly across games
  * undo always rewinds exactly one answer

Then it plays a far larger number of games at the engine level, where each
game is milliseconds rather than seconds, to cover many more objects.

Run:  .venv/Scripts/python scripts/playtest.py
      .venv/Scripts/python scripts/playtest.py --engine-games 2000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import MAYBE, NO, YES, BeliefEngine  # noqa: E402
from mindreader.feedback import load_noise_profile  # noqa: E402

APP = cfg.ROOT / "app" / "app_final.py"
FAILURES: list[str] = []
CHECKS = [0]


def check(label: str, ok: bool, detail: str = "") -> None:
    CHECKS[0] += 1
    if not ok:
        print("  [FAIL] %s  -> %s" % (label, detail))
        FAILURES.append(label)


def report(section: str) -> None:
    print("  [ok]   %s" % section)


# ---------------------------------------------------------------------------
# Part 1 - drive the real app
# ---------------------------------------------------------------------------

def app_games(n_games: int) -> None:
    from streamlit.testing.v1 import AppTest

    protos = load_prototypes()
    valid = set(protos.names)
    fidx = {f: i for i, f in enumerate(protos.features)}
    rng = random.Random(4242)

    styles = ["truthful", "truthful", "noisy", "all_yes", "all_no",
              "all_maybe", "undo_heavy"]

    def new_session():
        a = AppTest.from_file(str(APP), default_timeout=120)
        a.run()
        return a

    at = new_session()
    check("app starts", not at.exception, str([str(e) for e in at.exception]))
    if at.exception:
        return

    # Games that end "wrong" use the text box, whose key carries the game
    # number so each game gets a fresh, empty one. AppTest re-reads the
    # previous run's element tree, so once that key changes it looks up a
    # widget id that no longer exists and raises inside its own
    # get_widget_states(). That is a harness limitation, not an app fault -
    # the app raises nothing on this path - so those games get their own
    # session, while a single long-lived session is kept for the rest to
    # prove the score accumulates across games.
    used_text_box = False

    for g in range(n_games):
        style = styles[g % len(styles)]
        if used_text_box:
            at = new_session()
            used_text_box = False
        target = protos.names[rng.randrange(protos.n_objects)]
        ti = protos.index_of(target)

        at.button(key="start_btn").click().run()
        check("[%s] game starts" % style, not at.exception,
              str([str(e) for e in at.exception]))

        undos = 0
        for step in range(40):
            if at.session_state["over"]:
                break
            asked = at.session_state["asked"]

            # occasionally undo, then verify it rewound exactly one answer
            if style == "undo_heavy" and asked > 0 and undos < 4 and step % 3 == 0:
                before = len(at.session_state["answers"])
                btn = [b for b in at.button if b.key == "undo_q%d" % asked]
                if btn and not btn[0].disabled:
                    btn[0].click().run()
                    after = len(at.session_state["answers"])
                    check("[%s] undo removes exactly one answer" % style,
                          after == before - 1,
                          "%d -> %d" % (before, after))
                    check("[%s] undo keeps asked in step" % style,
                          at.session_state["asked"] == after,
                          "asked=%s answers=%d"
                          % (at.session_state["asked"], after))
                    undos += 1
                    continue

            # work out which attribute is on screen, to answer truthfully
            answered = {f for f, _ in at.session_state["answers"]}
            eng = BeliefEngine(
                protos,
                noise=load_noise_profile(cfg.NOISE_PROFILE, protos.features)
                if cfg.NOISE_PROFILE.exists() else 0.10,
            )
            for f, v in at.session_state["answers"]:
                eng.answer(f, v)
            feat = eng.best_question()

            if style == "all_yes":
                key = "yes_q%d" % asked
            elif style == "all_no":
                key = "no_q%d" % asked
            elif style == "all_maybe":
                key = "maybe_q%d" % asked
            else:
                truth = int(protos.X[ti, fidx[feat]]) if feat else 1
                if style == "noisy" and rng.random() < 0.15:
                    truth = 1 - truth
                key = ("yes_q%d" if truth == 1 else "no_q%d") % asked

            btn = [b for b in at.button if b.key == key]
            if not btn:
                break
            btn[0].click().run()
            check("[%s] answering never raises" % style, not at.exception,
                  str([str(e) for e in at.exception]))
            if at.exception:
                return

        # ---- invariants at the end of the game ----
        check("[%s] game terminates" % style, at.session_state["over"],
              "still running after 40 steps")
        guess = at.session_state["guess"]
        conf = at.session_state["confidence"]
        top3 = at.session_state["top3"]
        check("[%s] names a real object" % style, guess in valid,
              "guessed %r" % guess)
        check("[%s] confidence is a probability" % style, 0.0 <= conf <= 1.0,
              "conf=%r" % conf)
        check("[%s] three ranked candidates" % style, len(top3) == 3,
              "got %d" % len(top3))
        if len(top3) == 3:
            probs = [p for _, p in top3]
            check("[%s] candidates are ranked" % style,
                  probs == sorted(probs, reverse=True), str(probs))
            check("[%s] top candidate is the guess" % style,
                  top3[0][0] == guess, "%s vs %s" % (top3[0][0], guess))

        # ---- result screen ----
        wins_before = at.session_state["wins"]
        games_before = at.session_state["games"]
        if g % 2 == 0:
            at.button(key="correct_btn").click().run()
            check("[%s] marking correct works" % style, not at.exception,
                  str([str(e) for e in at.exception]))
            check("[%s] win counted" % style,
                  at.session_state["wins"] == wins_before + 1,
                  "%d -> %d" % (wins_before, at.session_state["wins"]))
        else:
            at.button(key="wrong_btn").click().run()
            check("[%s] marking wrong works" % style, not at.exception,
                  str([str(e) for e in at.exception]))
            check("[%s] win not counted" % style,
                  at.session_state["wins"] == wins_before, "")
            ti_box = list(at.text_input)
            if ti_box:
                ti_box[0].input(target).run()
                check("[%s] naming the real object works" % style,
                      not at.exception, str([str(e) for e in at.exception]))
                used_text_box = True

        check("[%s] game counted exactly once" % style,
              at.session_state["games"] == games_before, "")

        if used_text_box:
            continue      # this session is retired; a fresh one starts above

        at.button(key="play_again_btn").click().run()
        check("[%s] play again resets the board" % style,
              at.session_state["asked"] == 0
              and at.session_state["answers"] == []
              and not at.session_state["over"],
              "asked=%s" % at.session_state["asked"])
        check("[%s] play again keeps the score" % style,
              at.session_state["games"] == games_before, "")

    report("drove %d real app games across %d answering styles"
           % (n_games, len(styles)))


# ---------------------------------------------------------------------------
# Part 2 - many more games at engine level
# ---------------------------------------------------------------------------

def engine_games(n_games: int) -> None:
    protos = load_prototypes()
    fidx = {f: i for i, f in enumerate(protos.features)}
    rng = random.Random(99)
    valid = set(protos.names)

    wins = 0
    lengths = []
    for g in range(n_games):
        ti = rng.randrange(protos.n_objects)
        target = protos.names[ti]
        noise = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20])
        eng = BeliefEngine(protos)

        for _ in range(20):
            if eng.is_confident():
                break
            feat = eng.best_question()
            if feat is None:
                break
            truth = int(protos.X[ti, fidx[feat]])
            r = rng.random()
            if r < 0.05:
                eng.answer(feat, MAYBE)
            else:
                eng.answer(feat, (1 - truth) if rng.random() < noise else truth)

            p = eng.posterior()
            if not (np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-9
                    and (p > 0).all()):
                check("posterior stays a valid distribution", False,
                      "game %d: sum=%.6f min=%.3g" % (g, p.sum(), p.min()))
                return

        guess, conf = eng.best_guess()
        if guess not in valid or not 0.0 <= conf <= 1.0:
            check("engine returns a valid guess", False,
                  "game %d: %r conf=%r" % (g, guess, conf))
            return
        lengths.append(len(eng.answers))
        if guess == target:
            wins += 1

    check("engine games stay within the question cap", max(lengths) <= 20,
          "worst was %d" % max(lengths))
    report("played %d engine games: %.1f%% correct, %.1f questions mean, "
           "%d worst" % (n_games, 100 * wins / n_games,
                         float(np.mean(lengths)), max(lengths)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-games", type=int, default=14)
    ap.add_argument("--engine-games", type=int, default=1500)
    args = ap.parse_args()

    print("=" * 76)
    print("PLAYTEST")
    print("=" * 76)
    if cfg.NOISE_PROFILE.exists():
        print("  a learned noise profile is present and will be exercised")
    else:
        print("  no learned noise profile; the global default is in use")

    app_games(args.app_games)
    engine_games(args.engine_games)

    print()
    print("=" * 76)
    if FAILURES:
        print("PLAYTEST FAILED: %d of %d checks" % (len(FAILURES), CHECKS[0]))
        for f in sorted(set(FAILURES)):
            print("  - %s" % f)
        return 1
    print("PLAYTEST PASSED - %d checks" % CHECKS[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
