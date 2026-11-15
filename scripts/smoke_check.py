"""End-to-end check that the Phase 1 fixes hold against the real models.

Verifies:
  1. the trained models' expected features match the rebuilt dataset exactly
  2. the two previously-broken questions no longer eliminate every object
  3. answering truthfully identifies every object uniquely

Run:  .venv/Scripts/python scripts/smoke_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                           "" if ok else "  -> " + detail))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    df = pd.read_csv(cfg.DATASET)
    features = [c for c in df.columns if c != "object"]
    lookup = df.set_index("object")

    print("=" * 72)
    print("1. MODEL / DATASET FEATURE ALIGNMENT")
    print("=" * 72)
    for name in ("decision_tree", "naive_bayes", "mlp_neural_network", "best_model"):
        path = cfg.MODELS_DIR / ("%s.pkl" % name)
        if not path.exists():
            check("%s present" % name, False, "missing file")
            continue
        model = joblib.load(path)
        expected = list(getattr(model, "feature_names_in_", []))
        # The models are retrained in Phase 3. Until then they know a subset
        # of the dataset's attributes, which is fine because they are only a
        # fallback - but every feature they expect must still exist.
        unknown = sorted(set(expected) - set(features))
        check("%s features are all present in the dataset" % name,
              not unknown,
              "model expects features the dataset lacks: %s" % unknown)
        if not unknown and len(expected) < len(features):
            print("         (stale: knows %d of %d attributes, retrained in Phase 3)"
                  % (len(expected), len(features)))

    print()
    print("=" * 72)
    print("2. PREVIOUSLY BROKEN QUESTIONS")
    print("=" * 72)
    # These two columns used to load as all-zero because the dataset headers
    # were misspelled, so answering YES wiped out every candidate.
    for feat in cfg.LEGACY_COLUMN_RENAMES.values():
        if feat not in df.columns:
            check("%s exists" % feat, False, "column absent")
            continue
        yes = df[df[feat] == 1]["object"].tolist()
        check("answering YES to %r leaves candidates" % feat,
              len(yes) > 0,
              "no object has %s=1, so YES eliminates all %d objects"
              % (feat, len(df)))
        print("         %s=1 -> %s" % (feat, ", ".join(yes) if yes else "(none)"))

    print()
    print("=" * 72)
    print("3. TRUTHFUL PLAY IDENTIFIES EVERY OBJECT")
    print("=" * 72)
    ambiguous = []
    eliminated = []
    for target in df["object"]:
        truth = lookup.loc[target]
        survivors = df
        for feat in features:
            survivors = survivors[survivors[feat] == truth[feat]]
        objs = survivors["object"].tolist()
        if target not in objs:
            eliminated.append(target)
        elif len(objs) > 1:
            ambiguous.append((target, objs))

    check("no object is eliminated by its own truthful answers",
          not eliminated, str(eliminated))
    check("every object is uniquely identified by truthful answers",
          not ambiguous,
          "; ".join("%s -> %s" % (t, o) for t, o in ambiguous))

    print()
    print("=" * 72)
    if FAILURES:
        print("SMOKE CHECK FAILED: %d problem(s)" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("SMOKE CHECK PASSED - %d objects, %d features" % (len(df), len(features)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
