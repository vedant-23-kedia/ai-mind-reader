"""Compile the authoring spec into the dataset the app and models consume.

    data/source/features.json      feature -> question text + group
    data/source/objects.json       object  -> category + TRUE attributes
    data/source/implications.json  rules like is_mammal -> is_animal
        |
        v
    data/dataset_clean.csv, category_map.csv,
    feature_columns.json, question_map.json, question_order.json

Validation refuses to emit a dataset that is internally inconsistent:
  * an object referencing an unknown attribute
  * a feature that is constant (true for all or no objects)
  * two objects with identical attributes (no question could separate them)
  * a violated implication rule
  * objects separated by fewer than MIN_HAMMING_TARGET answers

The last check is the important one: if two objects differ by a single
answer, one mistake by the player makes them impossible to tell apart.

Run:  python scripts/build_dataset.py [--allow-fragile]
Only pass --allow-fragile while deliberately mid-authoring.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

SOURCE_DIR = cfg.DATA_DIR / "source"
FEATURES_SRC = SOURCE_DIR / "features.json"
OBJECTS_SRC = SOURCE_DIR / "objects.json"
IMPLICATIONS_SRC = SOURCE_DIR / "implications.json"


class BuildError(RuntimeError):
    pass


def load_spec():
    features = json.loads(FEATURES_SRC.read_text(encoding="utf-8"))
    objects = json.loads(OBJECTS_SRC.read_text(encoding="utf-8"))
    implications = json.loads(IMPLICATIONS_SRC.read_text(encoding="utf-8"))
    return features, objects, implications


def compile_matrix(features, objects):
    feature_names = list(features)
    names = list(objects)
    vectors = []
    for name in names:
        attrs = set(objects[name]["attributes"])
        unknown = sorted(attrs - set(feature_names))
        if unknown:
            raise BuildError("%s lists unknown attributes: %s" % (name, unknown))
        vectors.append([1 if f in attrs else 0 for f in feature_names])
    return feature_names, names, vectors


def validate(feature_names, names, vectors, objects, implications):
    problems = []
    idx = {f: i for i, f in enumerate(feature_names)}

    for name in names:
        if not objects[name].get("category"):
            problems.append("%s has no category" % name)

    for k, feat in enumerate(feature_names):
        col = [v[k] for v in vectors]
        if not any(col):
            problems.append("feature %r is true for no object" % feat)
        elif all(col):
            problems.append("feature %r is true for every object" % feat)

    seen = {}
    for name, vec in zip(names, vectors):
        key = tuple(vec)
        if key in seen:
            problems.append("%s and %s have identical attributes"
                            % (seen[key], name))
        seen[key] = name

    for a, b in implications:
        if a not in idx or b not in idx:
            problems.append("implication references unknown feature: %s -> %s" % (a, b))
            continue
        bad = [names[i] for i, v in enumerate(vectors)
               if v[idx[a]] == 1 and v[idx[b]] == 0]
        if bad:
            problems.append("implication %s -> %s broken by: %s"
                            % (a, b, ", ".join(bad)))
    return problems


def margins(names, vectors, feature_names):
    fragile = []
    best = len(feature_names)
    for i, j in itertools.combinations(range(len(names)), 2):
        d = sum(a != b for a, b in zip(vectors[i], vectors[j]))
        best = min(best, d)
        if d < cfg.MIN_HAMMING_TARGET:
            diff = [feature_names[k] for k in range(len(feature_names))
                    if vectors[i][k] != vectors[j][k]]
            fragile.append((d, names[i], names[j], diff))
    fragile.sort()
    return best, fragile


def write_artifacts(features, objects, feature_names, names, vectors):
    with cfg.DATASET.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["object"] + feature_names)
        for name, vec in zip(names, vectors):
            w.writerow([name] + vec)

    with cfg.CATEGORY_MAP.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "category"])
        for name in names:
            w.writerow([name, objects[name]["category"]])

    cfg.FEATURE_COLUMNS.write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8")

    qmap = {f: features[f]["question"] for f in feature_names}
    cfg.QUESTION_MAP.write_text(json.dumps(qmap, indent=2), encoding="utf-8")

    # Preserve the existing asked-order and append anything new, so adding a
    # feature never reshuffles the early questions unexpectedly.
    previous = []
    if cfg.QUESTION_ORDER.exists():
        previous = [f for f in json.loads(
            cfg.QUESTION_ORDER.read_text(encoding="utf-8")) if f in feature_names]
    order = previous + [f for f in feature_names if f not in previous]
    cfg.QUESTION_ORDER.write_text(json.dumps(order, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-fragile", action="store_true",
                    help="emit even if the minimum-distance gate fails")
    args = ap.parse_args()

    print("=" * 72)
    print("COMPILE DATASET FROM SOURCE SPEC")
    print("=" * 72)

    features, objects, implications = load_spec()
    feature_names, names, vectors = compile_matrix(features, objects)
    print("  %d objects x %d features, %d implication rules"
          % (len(names), len(feature_names), len(implications)))

    problems = validate(feature_names, names, vectors, objects, implications)
    if problems:
        print()
        print("  VALIDATION FAILED (%d):" % len(problems))
        for p in problems:
            print("    - %s" % p)
        return 1
    print("  validation passed")

    best, fragile = margins(names, vectors, feature_names)
    gate_ok = best >= cfg.MIN_HAMMING_TARGET

    if not gate_ok and not args.allow_fragile:
        print()
        print("  SEPARABILITY GATE FAILED: minimum distance %d, target %d"
              % (best, cfg.MIN_HAMMING_TARGET))
        print("  %d pair(s) too close:" % len(fragile))
        for d, a, b, diff in fragile:
            print("    [%d] %-18s vs %-18s  %s" % (d, a, b, ", ".join(diff)))
        print()
        print("  Nothing was written. Fix the spec, or pass --allow-fragile")
        print("  to emit anyway while mid-authoring.")
        return 1

    write_artifacts(features, objects, feature_names, names, vectors)

    cat_counts = {}
    for name in names:
        c = objects[name]["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print()
    print("  wrote dataset_clean.csv, category_map.csv, feature_columns.json,")
    print("        question_map.json, question_order.json")
    print()
    print("=" * 72)
    print("SEPARABILITY")
    print("=" * 72)
    print("  minimum pairwise distance: %d   (target %d: %s)"
          % (best, cfg.MIN_HAMMING_TARGET, "MET" if gate_ok else "NOT MET"))
    if fragile:
        print("  %d pair(s) below target:" % len(fragile))
        for d, a, b, diff in fragile:
            print("    [%d] %-18s vs %-18s  %s" % (d, a, b, ", ".join(diff)))
    print()
    print("  category balance:")
    for c, k in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print("    %-16s %d" % (c, k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
