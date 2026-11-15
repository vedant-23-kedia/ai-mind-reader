"""Rebuild the clean dataset from the immutable raw source.

Pipeline:  raw  ->  rename legacy columns  ->  apply audited corrections
                ->  validate  ->  data/dataset_clean.csv + data/category_map.csv

Uses only the standard library so the data layer never depends on the ML stack.
Run:  python scripts/build_dataset.py
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402


class BuildError(RuntimeError):
    pass


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise BuildError("%s is empty" % path)
    return rows[0], rows[1:]


def read_dicts(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def margin_report(names, vectors, features):
    """Return (min_distance, fragile_pairs) where fragile pairs are distance <= 2."""
    fragile = []
    best = len(features)
    for i, j in itertools.combinations(range(len(names)), 2):
        d = hamming(vectors[i], vectors[j])
        best = min(best, d)
        if d <= 2:
            diff = [features[k] for k in range(len(features))
                    if vectors[i][k] != vectors[j][k]]
            fragile.append((d, names[i], names[j], diff))
    fragile.sort()
    return best, fragile


def main():
    print("=" * 72)
    print("BUILD DATASET")
    print("=" * 72)

    header, rows = read_csv(cfg.RAW_DATASET)
    features_raw = header[1:]

    # ---- 1. rename legacy misspelled columns -------------------------------
    features = [cfg.LEGACY_COLUMN_RENAMES.get(f, f) for f in features_raw]
    renamed = [(a, b) for a, b in zip(features_raw, features) if a != b]
    for old, new in renamed:
        print("  renamed column  %r -> %r" % (old, new))
    if not renamed:
        print("  (no legacy columns to rename)")

    if len(set(features)) != len(features):
        dupes = sorted({f for f in features if features.count(f) > 1})
        raise BuildError("duplicate feature columns after rename: %s" % dupes)

    names = [r[0] for r in rows]
    if len(set(names)) != len(names):
        raise BuildError("duplicate object names in raw dataset")

    vectors = []
    for r in rows:
        vals = r[1:]
        if len(vals) != len(features):
            raise BuildError("%s: expected %d values, got %d"
                             % (r[0], len(features), len(vals)))
        for v in vals:
            if v not in ("0", "1"):
                raise BuildError("%s: non-binary value %r" % (r[0], v))
        vectors.append([int(v) for v in vals])

    idx_of_obj = {n: i for i, n in enumerate(names)}
    idx_of_feat = {f: i for i, f in enumerate(features)}

    before_min, before_fragile = margin_report(names, vectors, features)

    # ---- 2. apply audited feature corrections ------------------------------
    print()
    applied = 0
    for row in read_dicts(cfg.CORRECTIONS):
        obj, feat = row["object"], row["feature"]
        want_from, want_to = int(row["from"]), int(row["to"])
        if obj not in idx_of_obj:
            raise BuildError("correction references unknown object %r" % obj)
        if feat not in idx_of_feat:
            raise BuildError("correction references unknown feature %r" % feat)
        i, k = idx_of_obj[obj], idx_of_feat[feat]
        actual = vectors[i][k]
        if actual != want_from:
            raise BuildError(
                "correction drift: %s.%s is %d, correction expected from=%d. "
                "Raw data changed?" % (obj, feat, actual, want_from))
        vectors[i][k] = want_to
        applied += 1
    print("  applied %d audited feature corrections" % applied)

    deferred = read_dicts(cfg.DEFERRED_CORRECTIONS)
    print("  %d corrections deliberately deferred to Phase 2 (see %s)"
          % (len(deferred), cfg.DEFERRED_CORRECTIONS.name))

    # ---- 3. category map ---------------------------------------------------
    cat_header, cat_rows = read_csv(cfg.RAW_CATEGORY_MAP)
    cat_of = {r[0]: r[1] for r in cat_rows}
    cat_fixes = read_dicts(cfg.CATEGORY_CORRECTIONS)
    for row in cat_fixes:
        obj = row["object"]
        if obj not in cat_of:
            raise BuildError("category correction references unknown object %r" % obj)
        if cat_of[obj] != row["from"]:
            raise BuildError("category drift: %s is %r, expected %r"
                             % (obj, cat_of[obj], row["from"]))
        cat_of[obj] = row["to"]
    print("  applied %d category corrections" % len(cat_fixes))

    # ---- 4. validate -------------------------------------------------------
    print()
    problems = []

    missing_cat = [n for n in names if n not in cat_of]
    if missing_cat:
        problems.append("objects absent from category map: %s" % missing_cat)
    orphan_cat = [n for n in cat_of if n not in idx_of_obj]
    if orphan_cat:
        problems.append("category map lists unknown objects: %s" % orphan_cat)

    qmap = json.loads(cfg.QUESTION_MAP.read_text(encoding="utf-8"))
    no_question = [f for f in features if f not in qmap]
    if no_question:
        problems.append("features with no question text: %s" % no_question)
    orphan_q = [f for f in qmap if f not in idx_of_feat]
    if orphan_q:
        problems.append("question_map has entries for unknown features: %s" % orphan_q)

    seen = {}
    for n, v in zip(names, vectors):
        key = tuple(v)
        if key in seen:
            problems.append("identical feature vectors: %s and %s" % (seen[key], n))
        seen[key] = n

    if problems:
        print("  VALIDATION FAILED")
        for p in problems:
            print("    - %s" % p)
        return 1
    print("  validation passed: binary values, unique vectors, "
          "complete category map, complete question map")

    after_min, after_fragile = margin_report(names, vectors, features)

    # ---- 5. write artifacts ------------------------------------------------
    with cfg.DATASET.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["object"] + features)
        for n, v in zip(names, vectors):
            w.writerow([n] + v)

    with cfg.CATEGORY_MAP.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cat_header)
        for n in names:
            w.writerow([n, cat_of[n]])

    cfg.FEATURE_COLUMNS.write_text(json.dumps(features, indent=2), encoding="utf-8")

    order = json.loads(cfg.QUESTION_ORDER.read_text(encoding="utf-8"))
    order = [f for f in order if f in idx_of_feat]
    order += [f for f in features if f not in order]
    cfg.QUESTION_ORDER.write_text(json.dumps(order, indent=2), encoding="utf-8")

    print()
    print("  wrote %s  (%d objects x %d features)"
          % (cfg.DATASET.relative_to(cfg.ROOT), len(names), len(features)))
    print("  wrote %s" % cfg.CATEGORY_MAP.relative_to(cfg.ROOT))
    print("  wrote %s" % cfg.FEATURE_COLUMNS.relative_to(cfg.ROOT))
    print("  wrote %s" % cfg.QUESTION_ORDER.relative_to(cfg.ROOT))

    # ---- 6. separability report -------------------------------------------
    print()
    print("=" * 72)
    print("SEPARABILITY (how many answers must be wrong to confuse two objects)")
    print("=" * 72)
    b1 = sum(1 for f in before_fragile if f[0] == 1)
    a1 = sum(1 for f in after_fragile if f[0] == 1)
    b2 = sum(1 for f in before_fragile if f[0] == 2)
    a2 = sum(1 for f in after_fragile if f[0] == 2)
    print("  min pairwise distance   before: %d      after: %d" % (before_min, after_min))
    print("  pairs at distance 1     before: %3d      after: %3d" % (b1, a1))
    print("  pairs at distance 2     before: %3d      after: %3d" % (b2, a2))
    met = "MET" if after_min >= cfg.MIN_HAMMING_TARGET else "NOT MET - Phase 2 work"
    print("  target min distance: %d (%s)" % (cfg.MIN_HAMMING_TARGET, met))
    print()
    print("  Phase 2 work queue - pairs still one answer apart:")
    for d, a, b, diff in after_fragile:
        if d == 1:
            print("    %-18s vs %-18s only: %s" % (a, b, ", ".join(diff)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
