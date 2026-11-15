"""End-to-end check that the pipeline holds together.

Verifies:
  1. the trained object model's attributes match the compiled dataset
  2. the two originally-broken questions still leave candidates
  3. answering truthfully identifies every object uniquely (hard filtering)
  4. the trained model identifies every object from its clean prototype
  5. the model degrades sensibly rather than collapsing as answers get noisier

Run:  .venv/Scripts/python scripts/smoke_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.augment import make_noisy  # noqa: E402
from mindreader.engine import BeliefEngine  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                           "" if ok else "  -> " + detail))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    protos = load_prototypes()
    df = pd.read_csv(cfg.DATASET)
    features = [c for c in df.columns if c != "object"]

    print("=" * 72)
    print("1. TRAINED MODEL / DATASET ALIGNMENT")
    print("=" * 72)
    info_path = cfg.MODELS_DIR / "object_model_info.json"
    model_path = cfg.MODELS_DIR / "object_best_model.pkl"
    if not info_path.exists() or not model_path.exists():
        check("object model present", False,
              "run: python scripts/train_models.py")
        print("\nSMOKE CHECK FAILED")
        return 1

    info = json.loads(info_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)

    check("model attribute list matches the dataset",
          list(info["features"]) == features,
          "model trained on %d attributes, dataset has %d"
          % (len(info["features"]), len(features)))
    check("model class list matches the dataset's objects",
          list(info["classes"]) == protos.names,
          "model knows %d objects, dataset has %d"
          % (len(info["classes"]), protos.n_objects))
    print("         best model: %s, trained on %d samples"
          % (info["best_model"], info["trained_on"]["train_samples"]))

    print()
    print("=" * 72)
    print("2. ORIGINALLY BROKEN QUESTIONS")
    print("=" * 72)
    # These columns used to load as all-zero because the dataset headers were
    # misspelled, so answering YES wiped out every candidate.
    for feat in cfg.LEGACY_COLUMN_RENAMES.values():
        if feat not in df.columns:
            check("%s exists" % feat, False, "column absent")
            continue
        yes = df[df[feat] == 1]["object"].tolist()
        check("answering YES to %r leaves candidates" % feat, len(yes) > 0,
              "no object has %s=1" % feat)
        shown = ", ".join(yes[:6]) + (" ..." if len(yes) > 6 else "")
        print("         %s=1 -> %d object(s): %s" % (feat, len(yes), shown))

    print()
    print("=" * 72)
    print("3. THE BELIEF ENGINE PLAYS EVERY OBJECT (truthful answers)")
    print("=" * 72)
    fidx = {f: i for i, f in enumerate(protos.features)}
    wrong, lengths = [], []
    for ti, target in enumerate(protos.names):
        eng = BeliefEngine(protos)
        for _ in range(30):
            feat = eng.best_question()
            if feat is None or eng.is_confident():
                break
            eng.answer(feat, int(protos.X[ti, fidx[feat]]))
        guess, _ = eng.best_guess()
        lengths.append(len(eng.answers))
        if guess != target:
            wrong.append((target, guess))
    check("identifies all %d objects from truthful answers" % protos.n_objects,
          not wrong, str(wrong[:5]))
    print("         questions needed: mean %.1f, worst %d (log2(%d) = %.1f is "
          "the noiseless floor)"
          % (float(np.mean(lengths)), max(lengths), protos.n_objects,
             float(np.log2(protos.n_objects))))

    print()
    print("=" * 72)
    print("4. A WRONG ANSWER IS RECOVERABLE")
    print("=" * 72)
    # The old engine deleted candidates, so lying once about a broad question
    # made the true object unreachable for the rest of the game.
    recovered = 0
    sample = protos.names[::9]
    for target in sample:
        ti = protos.index_of(target)
        eng = BeliefEngine(protos)
        first = eng.best_question()
        eng.answer(first, 1 - int(protos.X[ti, fidx[first]]))   # lie
        for _ in range(30):
            feat = eng.best_question()
            if feat is None:
                break
            eng.answer(feat, int(protos.X[ti, fidx[feat]]))
            if eng.is_confident() and eng.best_guess()[0] == target:
                break
        if eng.best_guess()[0] == target:
            recovered += 1
    check("recovers from one lie on the opening question (%d/%d sampled)"
          % (recovered, len(sample)),
          recovered == len(sample),
          "%d of %d failed to recover" % (len(sample) - recovered, len(sample)))

    print()
    print("=" * 72)
    print("5. TRAINED MODEL ON CLEAN PROTOTYPES")
    print("=" * 72)
    pred = model.predict(protos.X.astype(np.int8))
    correct = int(np.sum(pred == np.arange(protos.n_objects)))
    check("model identifies every object from a perfect answer sheet",
          correct == protos.n_objects,
          "%d of %d correct" % (correct, protos.n_objects))

    print()
    print("=" * 72)
    print("6. DEGRADATION UNDER NOISY ANSWERS")
    print("=" * 72)
    prev = 101.0
    monotonic = True
    for lvl in (0.0, 0.05, 0.10, 0.15, 0.20):
        ds = make_noisy(protos.X, 20, (lvl,), seed=7)
        acc = 100 * float(np.mean(model.predict(ds.X) == ds.y))
        print("         %3.0f%% of answers wrong -> %5.1f%% top-1"
              % (lvl * 100, acc))
        if acc > prev + 1e-9:
            monotonic = False
        prev = acc
    check("accuracy falls as answers get noisier (no anomaly)", monotonic,
          "accuracy increased at a higher noise level")

    print()
    print("=" * 72)
    if FAILURES:
        print("SMOKE CHECK FAILED: %d problem(s)" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("SMOKE CHECK PASSED - %d objects, %d attributes"
          % (protos.n_objects, len(features)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
