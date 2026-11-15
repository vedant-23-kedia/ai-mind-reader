"""One-off migration: turn the wide CSV into a readable authoring spec.

Generates:
  data/source/features.json   feature -> question text + group
  data/source/objects.json    object  -> category + list of TRUE attributes

Authoring a 56x52 matrix by hand is unmaintainable and that is how the
original data errors crept in. Listing only the attributes that are true for
each object is far easier to read and review, and the matrix is compiled from
it. This migration is lossless by construction - verified by recompiling and
diffing against the current dataset.

Also reports which candidate implication rules (e.g. is_mammal -> is_animal)
actually hold in the data, so the real ones can be enforced from then on.

Run once:  python scripts/migrate_to_source_spec.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

SOURCE_DIR = cfg.DATA_DIR / "source"

# Which broad area each existing feature belongs to. Used only to keep the
# spec file readable and grouped; it has no effect on the compiled matrix.
GROUPS = {
    "nature": ["is_living", "is_animal", "is_natural", "is_mammal", "is_wild",
               "is_domestic_pet", "lives_in_water", "can_fly", "can_swim",
               "has_4_legs", "is_big"],
    "fiction": ["is_character", "is_cartoons", "is_marvel", "is_fictional",
                "is_superpower", "is_indian"],
    "food": ["is_food", "is_edible", "is_liquid", "holds_food",
             "used_for_cooking"],
    "made": ["made_by_humans", "is_purchased", "is_necessary", "used_daily",
             "is_reusable", "is_electronic", "is_vehicle"],
    "shape": ["is_round", "has_sharp_edge", "is_paper_like", "is_elastic"],
    "parts": ["has_seat", "has_screen", "has_blades", "has_cable",
              "shows_time", "holds_money", "produces_fire"],
    "wear": ["is_wearable", "is_worn_on_body", "worn_on_feet", "worn_on_ears",
             "worn_on_waist", "is_hat", "is_beauty"],
    "use": ["is_for_reading", "is_carried", "is_held_in_hand",
            "can_make_sound", "is_adult_only"],
}

# Candidate rules. The migration reports which hold so only true ones get
# enforced; a rule that fails points at either a bad rule or bad data.
CANDIDATE_IMPLICATIONS = [
    ("is_mammal", "is_animal"),
    ("is_animal", "is_living"),
    ("is_domestic_pet", "is_animal"),
    ("is_wild", "is_animal"),
    ("is_marvel", "is_character"),
    ("is_character", "is_fictional"),
    ("is_cartoons", "is_fictional"),
    ("is_food", "is_edible"),
    ("has_screen", "is_electronic"),
    ("is_electronic", "made_by_humans"),
    ("is_vehicle", "made_by_humans"),
    ("worn_on_feet", "is_worn_on_body"),
    ("worn_on_ears", "is_worn_on_body"),
    ("worn_on_waist", "is_worn_on_body"),
    ("is_hat", "is_worn_on_body"),
    ("is_hat", "is_wearable"),
    ("is_worn_on_body", "is_wearable"),
    ("holds_food", "made_by_humans"),
    ("used_for_cooking", "made_by_humans"),
    ("shows_time", "made_by_humans"),
    ("is_for_reading", "made_by_humans"),
    ("lives_in_water", "can_swim"),
    ("is_beauty", "is_purchased"),
]


def main() -> int:
    with cfg.DATASET.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    features = header[1:]

    with cfg.CATEGORY_MAP.open(newline="", encoding="utf-8") as fh:
        cat_of = {r["object"]: r["category"] for r in csv.DictReader(fh)}

    qmap = json.loads(cfg.QUESTION_MAP.read_text(encoding="utf-8"))

    group_of = {}
    for group, feats in GROUPS.items():
        for f in feats:
            group_of[f] = group
    ungrouped = [f for f in features if f not in group_of]
    if ungrouped:
        print("  note: features with no group (defaulting to 'other'): %s" % ungrouped)

    # ---- features.json -----------------------------------------------------
    feature_spec = {}
    for f in features:
        feature_spec[f] = {
            "question": qmap[f],
            "group": group_of.get(f, "other"),
        }

    # ---- objects.json ------------------------------------------------------
    object_spec = {}
    vectors = {}
    for row in body:
        name = row[0]
        vals = [int(v) for v in row[1:]]
        vectors[name] = vals
        object_spec[name] = {
            "category": cat_of[name],
            "attributes": [f for f, v in zip(features, vals) if v == 1],
        }

    # ---- which implication rules actually hold ----------------------------
    print()
    print("=" * 72)
    print("IMPLICATION RULES")
    print("=" * 72)
    idx = {f: i for i, f in enumerate(features)}
    holds, violated = [], []
    for a, b in CANDIDATE_IMPLICATIONS:
        if a not in idx or b not in idx:
            continue
        bad = [n for n, v in vectors.items() if v[idx[a]] == 1 and v[idx[b]] == 0]
        if bad:
            violated.append((a, b, bad))
        else:
            holds.append([a, b])
    print("  rules that hold (%d):" % len(holds))
    for a, b in holds:
        print("    %-18s -> %s" % (a, b))
    print()
    print("  rules violated by current data (%d):" % len(violated))
    for a, b, bad in violated:
        print("    %-18s -> %-18s broken by: %s" % (a, b, ", ".join(bad)))

    # ---- write -------------------------------------------------------------
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR / "features.json").write_text(
        json.dumps(feature_spec, indent=2), encoding="utf-8")
    (SOURCE_DIR / "objects.json").write_text(
        json.dumps(object_spec, indent=2), encoding="utf-8")
    (SOURCE_DIR / "implications.json").write_text(
        json.dumps(holds, indent=2), encoding="utf-8")

    print()
    print("  wrote data/source/features.json      (%d features)" % len(feature_spec))
    print("  wrote data/source/objects.json       (%d objects)" % len(object_spec))
    print("  wrote data/source/implications.json  (%d verified rules)" % len(holds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
