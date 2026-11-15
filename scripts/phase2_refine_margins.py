"""Phase 2d: close the remaining margins the separability gate reported.

Growing the object set created new near-collisions - Shoes beside Sandals,
Trousers beside Shirt, Spider-Man beside Black Panther. The gate named every
one of them, and each is closed here with an attribute that reflects a real
difference rather than a flag invented to fit a single object.

Idempotent: safe to re-run.
Run:  python scripts/phase2_refine_margins.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

SOURCE_DIR = cfg.DATA_DIR / "source"

# feature -> (question, group, objects it is true for)
REFINEMENT_FEATURES = {
    # Belt / Trousers / Shirt were within one answer of each other
    "has_pockets": ("Does it have pockets?", "parts",
                    ["Trousers", "Shirt", "Handbag"]),
    "covers_legs": ("Is it worn over the legs?", "wear", ["Trousers"]),
    "has_sleeves": ("Does it have sleeves?", "wear", ["Shirt"]),

    # Comb / Toothbrush / Soap / Broom
    "has_bristles": ("Does it have bristles?", "parts",
                     ["Toothbrush", "Broom"]),
    "used_on_hair": ("Is it used on hair?", "wear", ["Comb", "Hairband"]),
    "dissolves_in_water": ("Does it dissolve in water?", "shape", ["Soap"]),
    "used_on_floor": ("Is it used on the floor?", "use", ["Broom"]),

    # Screwdriver / Scissors / Knife
    "cuts_things": ("Is it used to cut things?", "use", ["Scissors", "Knife"]),
    "used_for_repair": ("Is it used to repair or build things?", "use",
                        ["Screwdriver"]),

    # Shoes / Sandals / Socks
    "has_laces": ("Does it have laces?", "parts", ["Shoes"]),
    "is_open_toed": ("Does it leave the toes open?", "wear", ["Sandals"]),

    # Spider-Man / Black Panther / Thor
    "is_king": ("Is it a king or prince?", "fiction",
                ["Thor", "Black_Panther"]),
    "shoots_web": ("Does it shoot webs?", "fiction", ["Spider_Man"]),

    # Chota Bheem / Motu Patlu
    "is_duo": ("Is it a pair of characters rather than one?", "fiction",
               ["Motu_Patlu"]),

    # Cow / Goat
    "has_beard": ("Does it have a beard?", "nature", ["Goat"]),

    # Pen / Key
    "opens_locks": ("Is it used to open locks?", "use", ["Key"]),

    # Book / Newspaper separated only via the weak is_reusable attribute
    "has_cover": ("Does it have a cover or binding?", "parts", ["Book"]),

    # Stars / Moon
    "orbits_earth": ("Does it orbit the Earth?", "nature", ["Moon"]),

    # Table / Bed
    "used_for_sleeping": ("Is it used for sleeping?", "use", ["Bed"]),

    # car / Bus / Train
    "runs_on_road": ("Does it travel on roads?", "made",
                     ["car", "Bus", "Motorcycle", "Bicycle"]),
    "is_privately_owned": ("Is it usually owned by one person or family?",
                           "made", ["car", "Motorcycle", "Bicycle"]),
    "carries_many_people": ("Can it carry dozens of people at once?", "use",
                            ["Bus", "Train", "Aeroplane", "Ship"]),
}

# (object, attribute, reason)
ADD_ATTRS = [
    ("Goat", "has_tail", "goats have tails, like the other animals labelled so"),
    ("Goat", "can_climb_walls", "goats are well known for climbing"),
    ("car", "has_seat", "cars have built-in seats"),
]

NEW_IMPLICATIONS = [
    ["has_pockets", "made_by_humans"],
    ["covers_legs", "is_worn_on_body"],
    ["has_sleeves", "is_worn_on_body"],
    ["is_open_toed", "worn_on_feet"],
    ["has_laces", "made_by_humans"],
    ["used_on_floor", "made_by_humans"],
    ["cuts_things", "has_sharp_edge"],
    ["used_for_repair", "made_by_humans"],
    ["is_king", "is_character"],
    ["shoots_web", "is_character"],
    ["is_duo", "is_character"],
    ["opens_locks", "made_by_humans"],
    ["used_for_sleeping", "made_by_humans"],
    ["runs_on_road", "is_vehicle"],
    ["is_privately_owned", "is_vehicle"],
    ["is_public_transport", "is_vehicle"],
]


def main() -> int:
    features = json.loads((SOURCE_DIR / "features.json").read_text(encoding="utf-8"))
    objects = json.loads((SOURCE_DIR / "objects.json").read_text(encoding="utf-8"))
    implications = json.loads(
        (SOURCE_DIR / "implications.json").read_text(encoding="utf-8"))

    print("=" * 72)
    print("PHASE 2d: CLOSE REMAINING MARGINS")
    print("=" * 72)
    print("  starting from %d objects x %d features"
          % (len(objects), len(features)))

    added_feats = 0
    for feat, (question, group, holders) in REFINEMENT_FEATURES.items():
        unknown = [o for o in holders if o not in objects]
        if unknown:
            raise SystemExit("attribute %r lists unknown objects: %s" % (feat, unknown))
        if feat not in features:
            features[feat] = {"question": question, "group": group}
            added_feats += 1
        for obj in holders:
            if feat not in objects[obj]["attributes"]:
                objects[obj]["attributes"].append(feat)

    added_attrs = 0
    for obj, feat, _reason in ADD_ATTRS:
        if obj not in objects:
            raise SystemExit("addition references unknown object %r" % obj)
        if feat not in features:
            raise SystemExit("addition references unknown attribute %r" % feat)
        if feat not in objects[obj]["attributes"]:
            objects[obj]["attributes"].append(feat)
            added_attrs += 1

    have = {tuple(r) for r in implications}
    added_rules = 0
    for rule in NEW_IMPLICATIONS:
        if tuple(rule) not in have:
            implications.append(rule)
            have.add(tuple(rule))
            added_rules += 1

    print("  added %d attributes, %d attribute values, %d implication rules"
          % (added_feats, added_attrs, added_rules))

    rank = {f: i for i, f in enumerate(features)}
    for spec in objects.values():
        spec["attributes"] = sorted(
            {a for a in spec["attributes"] if a in rank}, key=lambda f: rank[f])

    (SOURCE_DIR / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8")
    (SOURCE_DIR / "objects.json").write_text(
        json.dumps(objects, indent=2), encoding="utf-8")
    (SOURCE_DIR / "implications.json").write_text(
        json.dumps(implications, indent=2), encoding="utf-8")

    print()
    print("  now %d objects x %d features, %d implication rules"
          % (len(objects), len(features), len(implications)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
