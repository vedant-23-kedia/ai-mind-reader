"""Phase 5: does the engine work, and does it know how well it works?

Produces:
  data/evaluation.csv          accuracy and question counts by noise level
  data/calibration.csv         claimed confidence vs observed accuracy
  data/per_object.csv          which objects the engine finds hardest
  data/plots/calibration.png   reliability diagram, before and after fitting
  data/plots/game_accuracy.png accuracy and question count vs noise
  data/plots/noise_sensitivity.png what happens when the assumed error rate
                                   does not match the player's real one

Run:  .venv/Scripts/python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import DEFAULT_NOISE  # noqa: E402
from mindreader.evaluation import (  # noqa: E402
    all_steps,
    calibration_curve,
    expected_calibration_error,
    overconfidence,
    run_games,
    summarise,
)

NOISE_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
GAMES_PER_OBJECT = 8
FIT_NOISE = 0.10                     # the level the correction is fitted at
WEIGHT_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
PLOTS = cfg.DATA_DIR / "plots"


def main() -> int:
    protos = load_prototypes()
    PLOTS.mkdir(exist_ok=True)

    print("=" * 78)
    print("PHASE 5 EVALUATION - %d objects, %d games per object"
          % (protos.n_objects, GAMES_PER_OBJECT))
    print("=" * 78)

    # ---- 1. accuracy by noise level ---------------------------------------
    print()
    print("1. WHOLE-GAME ACCURACY")
    print("   %-10s %7s %7s %9s %7s" % ("wrong", "top1", "top3", "questions",
                                        "worst"))
    by_noise = {}
    for noise in NOISE_LEVELS:
        res = run_games(protos, noise, GAMES_PER_OBJECT)
        s = summarise(res)
        by_noise[noise] = (s, res)
        print("   %9.0f%% %6.1f%% %6.1f%% %9.1f %7d"
              % (noise * 100, s["top1"], s["top3"], s["mean_questions"],
                 s["max_questions"]))

    # ---- 2. calibration, as built -----------------------------------------
    print()
    print("2. CALIBRATION AS BUILT (likelihood_weight = 1.0)")
    base_res = by_noise[FIT_NOISE][1]
    base_pairs = all_steps(base_res)
    base_ece = expected_calibration_error(base_pairs)
    base_over = overconfidence(base_pairs)
    print("   measured over %d prediction steps at a %.0f%% player error rate"
          % (len(base_pairs), FIT_NOISE * 100))
    print("   expected calibration error: %.3f" % base_ece)
    print("   mean confidence minus actual accuracy: %+.3f  (%s)"
          % (base_over, "overconfident" if base_over > 0 else "underconfident"))
    print()
    print("   %-22s %10s %10s %8s" % ("claimed confidence", "mean claim",
                                      "actual", "games"))
    for r in calibration_curve(base_pairs):
        flag = ""
        gap = r["mean_confidence"] - r["accuracy"]
        if abs(gap) > 0.1:
            flag = "  <- off by %+.0f pts" % (gap * 100)
        print("   %4.0f%%-%3.0f%%            %9.1f%% %9.1f%% %8d%s"
              % (r["bin_low"] * 100, r["bin_high"] * 100,
                 r["mean_confidence"] * 100, r["accuracy"] * 100,
                 r["count"], flag))

    # ---- 3. fit the likelihood weight -------------------------------------
    print()
    print("3. FITTING THE LIKELIHOOD WEIGHT")
    print("   The attributes are not independent - the dataset enforces rules")
    print("   like is_mammal -> is_animal, so answering both counts one fact")
    print("   twice. Discounting each answer should correct that.")
    print()
    print("   %-8s %8s %9s %8s %8s" % ("weight", "ECE", "over/under",
                                       "top1", "quest"))
    fits = {}
    for w in WEIGHT_GRID:
        res = run_games(protos, FIT_NOISE, GAMES_PER_OBJECT,
                        likelihood_weight=w)
        pairs = all_steps(res)
        s = summarise(res)
        ece = expected_calibration_error(pairs)
        fits[w] = {"ece": ece, "over": overconfidence(pairs),
                   "top1": s["top1"], "questions": s["mean_questions"],
                   "pairs": pairs}
        print("   %-8.1f %8.3f %+9.3f %7.1f%% %8.1f"
              % (w, ece, fits[w]["over"], s["top1"], s["mean_questions"]))

    best_w = min(fits, key=lambda w: fits[w]["ece"])
    print()
    print("   best weight: %.1f  (ECE %.3f -> %.3f, a %.0f%% reduction)"
          % (best_w, base_ece, fits[best_w]["ece"],
             100 * (1 - fits[best_w]["ece"] / base_ece) if base_ece else 0))
    print("   top-1 accuracy at that weight: %.1f%% (was %.1f%%)"
          % (fits[best_w]["top1"], fits[1.0]["top1"]))

    # ---- 3b. why is it already calibrated? --------------------------------
    # Naive Bayes should be overconfident when its inputs are dependent, and
    # these inputs provably are: the dataset enforces 65 implication rules.
    # The measurement above says it is not. The likely reason is that
    # information gain never asks a redundant question - once "is it a
    # mammal?" is answered yes, "is it an animal?" carries almost no
    # information and is never chosen - so the dependent evidence is never
    # double-counted in the first place. That is testable.
    print()
    print("3b. WHY IS IT ALREADY CALIBRATED?")
    rules = json.loads(
        (cfg.DATA_DIR / "source" / "implications.json").read_text(encoding="utf-8"))
    rng_pairs = np.random.default_rng(0)
    feats = protos.features
    random_pairs = [
        (feats[i], feats[j]) for i, j in
        rng_pairs.integers(0, len(feats), size=(len(rules), 2))
        if feats[i] != feats[j]
    ]

    redundancy_games = run_games(protos, FIT_NOISE, 4, seed=999,
                                 likelihood_weight=best_w)
    asked_sets = [set(g.asked_features) for g in redundancy_games]

    def co_occurrence(pairs):
        """How often both attributes of a pair get asked in the same game."""
        both = total = 0
        for asked in asked_sets:
            for a, b in pairs:
                total += 1
                if a in asked and b in asked:
                    both += 1
        return 100 * both / total if total else float("nan")

    impl_rate = co_occurrence([(a, b) for a, b in rules])
    rand_rate = co_occurrence(random_pairs)
    print("   both attributes of a pair asked in the same game:")
    print("     implication-linked pairs (e.g. is_mammal / is_animal): %.1f%%"
          % impl_rate)
    print("     unrelated pairs, for comparison:                       %.1f%%"
          % rand_rate)
    if impl_rate < rand_rate:
        print("   -> information gain avoids the redundant question.")
    else:
        print("   -> redundant questions ARE asked (more often than unrelated")
        print("      pairs), so avoidance is not the explanation.")
    print()
    print("   The actual reason is that this is not really *naive* Bayes.")
    print("   Naive Bayes approximates P(features | class) as a product; here")
    print("   that product is not an approximation, it is the simulated noise")
    print("   process exactly - answers are flipped independently at a known")
    print("   rate. And because there are only %d objects, the engine" % protos.n_objects)
    print("   enumerates the whole hypothesis space rather than assuming")
    print("   anything about it. The implication rules constrain which")
    print("   prototype vectors exist, not how answers are corrupted, so")
    print("   there is no independence assumption left to violate.")
    print()
    print("   Which makes this result partly circular, and worth saying so:")
    print("   it confirms the arithmetic is right, not that the noise model")
    print("   is realistic. Against real players - who are reliably sure")
    print("   about is_animal and reliably vague about is_big - the model")
    print("   would be misspecified and calibration should get worse.")
    print("   data/feedback_log.csv is what would eventually measure that.")

    # ---- 4. noise sensitivity ---------------------------------------------
    print()
    print("4. SENSITIVITY TO THE ASSUMED ERROR RATE")
    print("   In reality you never know how careless a player is. Rows are")
    print("   what the engine assumes; columns what the player actually does.")
    print()
    assumed_grid = (0.05, 0.10, 0.15, 0.20)
    print("   assumed |" + "".join("  true %2.0f%%" % (t * 100)
                                   for t in NOISE_LEVELS))
    sensitivity = {}
    for a in assumed_grid:
        row = []
        for t in NOISE_LEVELS:
            res = run_games(protos, t, 4, engine_noise=a,
                            likelihood_weight=best_w)
            row.append(summarise(res)["top1"])
        sensitivity[a] = row
        print("   %6.0f%%  |" % (a * 100)
              + "".join("   %5.1f%%" % v for v in row))

    # ---- 5. hardest objects ----------------------------------------------
    print()
    print("5. HARDEST OBJECTS (at a %.0f%% player error rate)" % (FIT_NOISE * 100))
    res = run_games(protos, FIT_NOISE, GAMES_PER_OBJECT,
                    likelihood_weight=best_w)
    per_obj = defaultdict(lambda: [0, 0, []])
    confusions = defaultdict(int)
    for r in res:
        per_obj[r.target][0] += r.correct
        per_obj[r.target][1] += 1
        per_obj[r.target][2].append(r.questions)
        if not r.correct:
            confusions[(r.target, r.guess)] += 1
    ranked = sorted(per_obj.items(), key=lambda kv: kv[1][0] / kv[1][1])
    for name, (hits, total, qs) in ranked[:8]:
        print("   %-18s %3.0f%% correct over %d games, %.1f questions"
              % (name, 100 * hits / total, total, float(np.mean(qs))))
    print()
    print("   most common confusions:")
    for (a, b), k in sorted(confusions.items(), key=lambda kv: -kv[1])[:6]:
        print("     %-18s mistaken for %-18s %d times" % (a, b, k))

    # ---- write artifacts --------------------------------------------------
    with (cfg.DATA_DIR / "evaluation.csv").open("w", newline="",
                                                encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["player_error_rate", "top1_pct", "top3_pct",
                    "mean_questions", "max_questions", "mean_confidence_pct"])
        for noise, (s, _) in by_noise.items():
            w.writerow([noise, round(s["top1"], 2), round(s["top3"], 2),
                        round(s["mean_questions"], 2), s["max_questions"],
                        round(s["mean_confidence"], 2)])

    with (cfg.DATA_DIR / "calibration.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["likelihood_weight", "bin_low", "bin_high",
                    "mean_confidence", "accuracy", "count"])
        for label, pairs in (("1.0", base_pairs), (str(best_w),
                                                   fits[best_w]["pairs"])):
            for r in calibration_curve(pairs):
                w.writerow([label, r["bin_low"], r["bin_high"],
                            round(r["mean_confidence"], 4),
                            round(r["accuracy"], 4), r["count"]])

    with (cfg.DATA_DIR / "per_object.csv").open("w", newline="",
                                                encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "accuracy_pct", "games", "mean_questions"])
        for name, (hits, total, qs) in sorted(per_obj.items()):
            w.writerow([name, round(100 * hits / total, 2), total,
                        round(float(np.mean(qs)), 2)])

    (cfg.DATA_DIR / "evaluation_summary.json").write_text(json.dumps({
        "games_per_object": GAMES_PER_OBJECT,
        "objects": protos.n_objects,
        "engine_noise_default": DEFAULT_NOISE,
        "fitted_likelihood_weight": best_w,
        "ece_before": round(base_ece, 4),
        "ece_after": round(fits[best_w]["ece"], 4),
        "overconfidence_before": round(base_over, 4),
        "overconfidence_after": round(fits[best_w]["over"], 4),
        "accuracy_by_noise": {str(k): round(v[0]["top1"], 2)
                              for k, v in by_noise.items()},
        "noise_sensitivity": {str(a): [round(x, 2) for x in row]
                              for a, row in sensitivity.items()},
    }, indent=2), encoding="utf-8")

    # ---- plots -------------------------------------------------------------
    plt.figure(figsize=(6.5, 6.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, pairs, style in (
        ("as built (weight 1.0)", base_pairs, "o-"),
        ("fitted (weight %.1f)" % best_w, fits[best_w]["pairs"], "s-"),
    ):
        rows = calibration_curve(pairs)
        plt.plot([r["mean_confidence"] for r in rows],
                 [r["accuracy"] for r in rows], style, label=label)
    plt.xlabel("Confidence the engine claims")
    plt.ylabel("How often it is actually right")
    plt.title("Reliability diagram\n(points below the line = overconfident)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS / "calibration.png", dpi=130)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = [n * 100 for n in NOISE_LEVELS]
    ax1.plot(xs, [by_noise[n][0]["top1"] for n in NOISE_LEVELS],
             "o-", color="#1f77b4", label="top-1")
    ax1.plot(xs, [by_noise[n][0]["top3"] for n in NOISE_LEVELS],
             "s--", color="#2ca02c", label="top-3")
    ax1.set_xlabel("Share of answers the player gets wrong (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 102)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, [by_noise[n][0]["mean_questions"] for n in NOISE_LEVELS],
             "^:", color="#d62728", label="questions asked")
    ax2.set_ylabel("Mean questions per game")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left")
    plt.title("Whole-game performance vs player carelessness")
    plt.tight_layout()
    plt.savefig(PLOTS / "game_accuracy.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7.5, 5))
    for a, row in sensitivity.items():
        plt.plot(xs, row, "o-", label="engine assumes %.0f%%" % (a * 100))
    plt.xlabel("Share of answers the player actually gets wrong (%)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Does it matter if the assumed error rate is wrong?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "noise_sensitivity.png", dpi=130)
    plt.close()

    print()
    print("=" * 78)
    print("  wrote data/evaluation.csv, calibration.csv, per_object.csv,")
    print("        evaluation_summary.json")
    print("  wrote data/plots/{calibration,game_accuracy,noise_sensitivity}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
