"""Phase 3: train object-level models on noise-augmented data.

The original models predicted one of 9 categories and were reported at
92.86% accuracy - which was 13 of 14 rows on a single split of a 56-row
dataset. They never identified the object, which is what the game is for.

These models predict the object itself: 117 classes from 141 binary
attributes, trained on simulated players who get some answers wrong.

Evaluation is on an independently generated test set (different seed), with
accuracy broken down by how noisy the simulated player was. That breakdown
is the honest headline: a single number hides the fact that performance
depends entirely on how carefully the human answers.

Run:  .venv/Scripts/python scripts/train_models.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import BernoulliNB, GaussianNB  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.augment import NOISE_LEVELS, make_noisy, top_k_accuracy  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402

SEED = 42
TRAIN_PER_OBJECT = 200
TEST_PER_OBJECT = 60

PLOTS_DIR = cfg.DATA_DIR / "plots"


def build_models():
    """The three models from the original project, plus two baselines.

    BernoulliNB replaces GaussianNB: the attributes are binary, and a
    Gaussian likelihood models them as continuous normal variables, which is
    the wrong distribution. Both are trained below so the difference is
    measured rather than asserted.
    """
    return {
        "decision_tree": DecisionTreeClassifier(random_state=SEED),
        "bernoulli_nb": BernoulliNB(),
        "gaussian_nb": GaussianNB(),
        "mlp_neural_network": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=SEED,
        ),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        # min_samples_leaf is not tuning for accuracy, it is tuning for size.
        # Every node in every tree stores a 117-class distribution (936
        # bytes), so 300 unpruned trees pickle to 4.2 GB - more than GitHub
        # will accept in a single file, for a toy problem. Limiting leaf size
        # costs 1.3 points of accuracy (84.8% -> 83.5%) and cuts the artifact
        # 39x to roughly 107 MB.
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=10, random_state=SEED, n_jobs=-1
        ),
    }


def main() -> int:
    protos = load_prototypes()
    print("=" * 76)
    print("PHASE 3: OBJECT-LEVEL TRAINING")
    print("=" * 76)
    print("  %d objects, %d attributes -> %d-class problem"
          % (protos.n_objects, protos.n_features, protos.n_objects))

    train = make_noisy(protos.X, TRAIN_PER_OBJECT, NOISE_LEVELS, seed=SEED)
    test = make_noisy(protos.X, TEST_PER_OBJECT, NOISE_LEVELS, seed=SEED + 1000)
    print("  train: %d samples   test: %d samples (independent seed)"
          % (len(train), len(test)))
    print("  flip rates: %s" % ", ".join("%.0f%%" % (p * 100) for p in NOISE_LEVELS))

    # Labels stay as integer object indices. sklearn 1.7.2's MLPClassifier
    # crashes on string labels when early_stopping is on (it runs np.isnan
    # over the predicted labels), and early stopping is worth keeping. The
    # index -> name mapping is written to object_model_info.json so nothing
    # downstream has to infer it from row order.
    names_arr = np.array(protos.names)
    y_train = train.y
    y_test = test.y

    # The clean prototypes are the easiest possible case: a perfect player.
    X_clean = protos.X.astype(np.int8)
    y_clean = np.arange(protos.n_objects)

    results = {}
    models = build_models()
    for name, model in models.items():
        print()
        print("-" * 76)
        print("  %s" % name)
        t0 = time.time()
        model.fit(train.X, y_train)
        fit_s = time.time() - t0

        # proba columns follow model.classes_, so map each true name to its
        # column before computing top-k.
        col_of = {c: i for i, c in enumerate(model.classes_)}
        proba = model.predict_proba(test.X)
        y_test_cols = np.array([col_of[n] for n in y_test])

        row = {
            "fit_seconds": round(fit_s, 1),
            "test_top1": round(100 * float(np.mean(model.predict(test.X) == y_test)), 2),
            "test_top3": round(100 * top_k_accuracy(proba, y_test_cols, 3), 2),
            "test_top5": round(100 * top_k_accuracy(proba, y_test_cols, 5), 2),
            "clean_top1": round(
                100 * float(np.mean(model.predict(X_clean) == y_clean)), 2),
            "by_noise": {},
        }
        for lvl in NOISE_LEVELS:
            Xl, yl = test.at_noise(lvl)
            row["by_noise"][lvl] = round(
                100 * float(np.mean(model.predict(Xl) == yl)), 2)

        results[name] = row
        print("    fit %.1fs | clean %.1f%% | test top1 %.1f%% top3 %.1f%% top5 %.1f%%"
              % (fit_s, row["clean_top1"], row["test_top1"],
                 row["test_top3"], row["test_top5"]))
        print("    by flip rate: %s" % "  ".join(
            "%.0f%%:%.1f" % (k * 100, v) for k, v in row["by_noise"].items()))

        cfg.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, cfg.MODELS_DIR / ("object_%s.pkl" % name))

    # ---- pick the best on the noisy test set ------------------------------
    best_name = max(results, key=lambda n: results[n]["test_top1"])
    joblib.dump(models[best_name], cfg.MODELS_DIR / "object_best_model.pkl")
    (cfg.MODELS_DIR / "object_model_info.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "classes": protos.names,
                "features": protos.features,
                "trained_on": {
                    "objects": protos.n_objects,
                    "attributes": protos.n_features,
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "noise_levels": list(NOISE_LEVELS),
                    "seed": SEED,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- comparison table --------------------------------------------------
    with (cfg.DATA_DIR / "model_comparison.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Model", "Clean Top-1 (%)", "Noisy Top-1 (%)",
                    "Noisy Top-3 (%)", "Noisy Top-5 (%)"]
                   + ["Top-1 @ %.0f%% flips" % (p * 100) for p in NOISE_LEVELS]
                   + ["Fit (s)"])
        for name, r in sorted(results.items(), key=lambda x: -x[1]["test_top1"]):
            w.writerow([name, r["clean_top1"], r["test_top1"], r["test_top3"],
                        r["test_top5"]]
                       + [r["by_noise"][p] for p in NOISE_LEVELS]
                       + [r["fit_seconds"]])

    # ---- plots -------------------------------------------------------------
    PLOTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 5.5))
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_top1"]):
        plt.plot([p * 100 for p in NOISE_LEVELS],
                 [r["by_noise"][p] for p in NOISE_LEVELS],
                 marker="o", label=name)
    plt.xlabel("Share of answers the simulated player got wrong (%)")
    plt.ylabel("Top-1 object accuracy (%)")
    plt.title("Object identification accuracy vs answer noise (%d objects)"
              % protos.n_objects)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_vs_noise.png", dpi=130)
    plt.close()

    dt = models["decision_tree"]
    imp = np.argsort(-dt.feature_importances_)[:25]
    plt.figure(figsize=(8, 7))
    plt.barh([protos.features[i] for i in imp][::-1],
             dt.feature_importances_[imp][::-1])
    plt.xlabel("Gini importance")
    plt.title("Decision tree - 25 most informative attributes")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dt_feature_importance.png", dpi=130)
    plt.close()

    mlp = models["mlp_neural_network"]
    if hasattr(mlp, "loss_curve_"):
        plt.figure(figsize=(7, 4.5))
        plt.plot(mlp.loss_curve_)
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("MLP training loss (%d epochs, early stopping)"
                  % len(mlp.loss_curve_))
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mlp_loss_curve.png", dpi=130)
        plt.close()

    # ---- summary -----------------------------------------------------------
    print()
    print("=" * 76)
    print("RESULTS (test set = independently generated noisy players)")
    print("=" * 76)
    print("  %-22s %7s %7s %7s %7s" % ("model", "clean", "top1", "top3", "top5"))
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_top1"]):
        print("  %-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
              % (name, r["clean_top1"], r["test_top1"], r["test_top3"],
                 r["test_top5"]))

    bnb = results["bernoulli_nb"]["test_top1"]
    gnb = results["gaussian_nb"]["test_top1"]
    print()
    print("  Naive Bayes variant: BernoulliNB %.1f%% vs GaussianNB %.1f%%  (%+.1f pts)"
          % (bnb, gnb, bnb - gnb))
    print("  best model: %s -> models/object_best_model.pkl" % best_name)
    print()
    print("  wrote data/model_comparison.csv, models/object_*.pkl,")
    print("        data/plots/{accuracy_vs_noise,dt_feature_importance,"
          "mlp_loss_curve}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
