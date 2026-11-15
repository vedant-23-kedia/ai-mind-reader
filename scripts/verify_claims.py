"""Cross-check every number the README claims against the generated data.

A README that drifts from its own artifacts is worse than one with no
numbers, because the numbers look authoritative. This re-derives each claim
from the data files and fails if any disagree.

Run:  .venv/Scripts/python scripts/verify_claims.py
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import BeliefEngine  # noqa: E402

README = cfg.ROOT / "README.md"
PROBLEMS: list[str] = []
CHECKS = [0]


def claim(label: str, ok: bool, detail: str = "") -> None:
    CHECKS[0] += 1
    if ok:
        print("  [ok]   %s" % label)
    else:
        print("  [FAIL] %s  -> %s" % (label, detail))
        PROBLEMS.append("%s (%s)" % (label, detail))


def readme_has(text: str) -> bool:
    return text in README.read_text(encoding="utf-8")


def main() -> int:
    md = README.read_text(encoding="utf-8")
    protos = load_prototypes()

    print("=" * 72)
    print("1. DATASET CLAIMS")
    print("=" * 72)
    claim("object count (%d) appears in the README" % protos.n_objects,
          "117 objects" in md and protos.n_objects == 117,
          "dataset has %d" % protos.n_objects)
    claim("attribute count (%d) appears in the README" % protos.n_features,
          "141 binary attributes" in md and protos.n_features == 141,
          "dataset has %d" % protos.n_features)

    d = np.array([(protos.X[i] != protos.X[j]).sum()
                  for i, j in itertools.combinations(range(protos.n_objects), 2)])
    claim("minimum separation is 3", d.min() == 3, "it is %d" % d.min())
    claim("no pairs 1 or 2 answers apart",
          (d == 1).sum() == 0 and (d == 2).sum() == 0,
          "%d at 1, %d at 2" % ((d == 1).sum(), (d == 2).sum()))
    claim("median separation 13 as claimed", int(np.median(d)) == 13,
          "median is %d" % np.median(d))

    rules = json.loads(
        (cfg.DATA_DIR / "source" / "implications.json").read_text(encoding="utf-8"))
    claim("65 implication rules", len(rules) == 65, "there are %d" % len(rules))

    # every rule must actually hold
    idx = {f: i for i, f in enumerate(protos.features)}
    broken = []
    for a, b in rules:
        bad = [protos.names[i] for i in range(protos.n_objects)
               if protos.X[i, idx[a]] == 1 and protos.X[i, idx[b]] == 0]
        if bad:
            broken.append("%s->%s: %s" % (a, b, bad[:3]))
    claim("all implication rules hold in the data", not broken, "; ".join(broken))

    print()
    print("=" * 72)
    print("2. MODEL CLAIMS (data/model_comparison.csv)")
    print("=" * 72)
    with (cfg.DATA_DIR / "model_comparison.csv").open(encoding="utf-8") as fh:
        rows = {r["Model"]: r for r in csv.DictReader(fh)}
    for name, top1, top3, top5 in [
        ("bernoulli_nb", 86.2, 95.3, 97.3),
        ("random_forest", 83.5, 93.2, 95.8),
        ("logistic_regression", 81.5, 92.6, 95.3),
        ("mlp_neural_network", 79.7, 91.8, 95.0),
        ("gaussian_nb", 78.9, 91.8, 95.2),
        ("decision_tree", 59.7, 60.3, 61.0),
    ]:
        r = rows.get(name)
        if r is None:
            claim("%s present in comparison" % name, False, "missing row")
            continue
        got = (float(r["Noisy Top-1 (%)"]), float(r["Noisy Top-3 (%)"]),
               float(r["Noisy Top-5 (%)"]))
        ok = all(abs(g - c) < 0.1 for g, c in zip(got, (top1, top3, top5)))
        claim("%s = %.1f/%.1f/%.1f" % (name, top1, top3, top5), ok,
              "actually %.1f/%.1f/%.1f" % got)
        claim("%s README row matches" % name,
              "| %.1f%% | %.1f%% | %.1f%% |" % got in md
              or "**%.1f%%** | **%.1f%%** | **%.1f%%**" % got in md,
              "README does not contain %.1f/%.1f/%.1f" % got)

    bnb = float(rows["bernoulli_nb"]["Noisy Top-1 (%)"])
    gnb = float(rows["gaussian_nb"]["Noisy Top-1 (%)"])
    mlp = float(rows["mlp_neural_network"]["Noisy Top-1 (%)"])
    claim("BernoulliNB beats GaussianNB by 7.2 points",
          abs((bnb - gnb) - 7.2) < 0.1, "gap is %+.1f" % (bnb - gnb))
    claim("BernoulliNB beats the MLP by 6.5 points",
          abs((bnb - mlp) - 6.5) < 0.1, "gap is %+.1f" % (bnb - mlp))
    claim("every model is perfect on clean prototypes",
          all(float(r["Clean Top-1 (%)"]) == 100.0 for r in rows.values()),
          "some model is not 100%")

    info = json.loads((cfg.MODELS_DIR / "object_model_info.json")
                      .read_text(encoding="utf-8"))
    t = info["trained_on"]
    claim("23,400 training sheets", t["train_samples"] == 23400,
          "actually %d" % t["train_samples"])
    claim("7,020 test sheets", t["test_samples"] == 7020,
          "actually %d" % t["test_samples"])
    claim("model classes match dataset objects",
          list(info["classes"]) == protos.names, "mismatch")
    claim("model attributes match dataset attributes",
          list(info["features"]) == protos.features, "mismatch")

    print()
    print("=" * 72)
    print("3. EVALUATION CLAIMS (data/evaluation*.csv/json)")
    print("=" * 72)
    summary = json.loads((cfg.DATA_DIR / "evaluation_summary.json")
                         .read_text(encoding="utf-8"))
    claim("expected calibration error 0.010",
          abs(summary["ece_before"] - 0.010) < 0.0015,
          "actually %.4f" % summary["ece_before"])
    claim("overconfidence -0.004",
          abs(summary["overconfidence_before"] - (-0.004)) < 0.0015,
          "actually %+.4f" % summary["overconfidence_before"])
    claim("fitted likelihood weight is 1.0",
          summary["fitted_likelihood_weight"] == 1.0,
          "fitted %s" % summary["fitted_likelihood_weight"])

    with (cfg.DATA_DIR / "evaluation.csv").open(encoding="utf-8") as fh:
        ev = {float(r["player_error_rate"]): r for r in csv.DictReader(fh)}
    for noise, top1, conf in [(0.0, 100.0, 97.0), (0.05, 93.5, 93.7),
                              (0.1, 85.8, 84.4), (0.15, 70.2, 68.3),
                              (0.2, 49.9, 49.3)]:
        r = ev[noise]
        claim("game top-1 at %.0f%% noise = %.1f%%" % (noise * 100, top1),
              abs(float(r["top1_pct"]) - top1) < 0.15,
              "actually %.2f" % float(r["top1_pct"]))
        claim("mean confidence at %.0f%% noise = %.1f%%" % (noise * 100, conf),
              abs(float(r["mean_confidence_pct"]) - conf) < 0.15,
              "actually %.2f" % float(r["mean_confidence_pct"]))

    with (cfg.DATA_DIR / "per_object.csv").open(encoding="utf-8") as fh:
        per = {r["object"]: float(r["accuracy_pct"]) for r in csv.DictReader(fh)}
    claim("Scissors is the hardest object at 38%",
          abs(per.get("Scissors", -1) - 37.5) < 1.0,
          "Scissors scores %.1f" % per.get("Scissors", -1))
    claim("Scissors really is the worst",
          min(per, key=per.get) == "Scissors",
          "worst is %s" % min(per, key=per.get))

    print()
    print("=" * 72)
    print("4. ENGINE THEORY CLAIMS")
    print("=" * 72)
    eng = BeliefEngine(protos)
    cap = 1.0 - (-(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9)))
    claim("noisy-channel capacity is 0.531 bits at 10%% noise",
          abs(cap - 0.531) < 0.001, "it is %.4f" % cap)
    best_gain = max(eng.information_gain().values())
    claim("best opening question reaches that capacity",
          abs(best_gain - cap) < 0.01,
          "best gain %.4f vs capacity %.4f" % (best_gain, cap))
    claim("no question exceeds the capacity",
          max(eng.information_gain().values()) <= cap + 1e-9, "one does")
    claim("log2(117) = 6.9", abs(math.log2(117) - 6.9) < 0.05,
          "%.3f" % math.log2(117))

    # mean questions on truthful play
    fidx = {f: i for i, f in enumerate(protos.features)}
    lengths = []
    for ti in range(protos.n_objects):
        e = BeliefEngine(protos)
        for _ in range(30):
            f = e.best_question()
            if f is None or e.is_confident():
                break
            e.answer(f, int(protos.X[ti, fidx[f]]))
        lengths.append(len(e.answers))
    mean_q = float(np.mean(lengths))
    claim("truthful play averages 9.3 questions", abs(mean_q - 9.3) < 0.3,
          "averages %.2f" % mean_q)

    print()
    print("=" * 72)
    print("5. README INTERNAL CONSISTENCY")
    print("=" * 72)
    m = re.search(r"# (\d+) tests", md)
    claim("README test count matches the suite",
          m is not None, "no test count found in README")
    if m:
        import subprocess
        out = subprocess.run(
            [str(cfg.ROOT / ".venv/Scripts/python.exe"), "-m", "pytest",
             str(cfg.ROOT / "tests"), "-q", "--collect-only"],
            capture_output=True, text=True, cwd=str(cfg.ROOT))
        found = re.search(r"(\d+) tests? collected", out.stdout)
        actual = int(found.group(1)) if found else -1
        claim("README says %s tests, suite has %d" % (m.group(1), actual),
              int(m.group(1)) == actual,
              "README %s vs actual %d" % (m.group(1), actual))

    claim("no 'Phase N' marked done beyond 6",
          "Phase 7" not in md, "stray phase reference")
    claim("the retired 92.86% figure is only ever discussed as meaningless",
          md.count("92.86%") == 0 or "never meaningful" in md or
          "not meaningful" in md, "92.86% quoted without caveat")

    print()
    print("=" * 72)
    if PROBLEMS:
        print("%d of %d CLAIMS FAILED" % (len(PROBLEMS), CHECKS[0]))
        for p in PROBLEMS:
            print("  - %s" % p)
        return 1
    print("ALL %d CLAIMS VERIFIED" % CHECKS[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
