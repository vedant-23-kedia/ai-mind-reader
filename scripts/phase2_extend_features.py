"""Phase 2: add discriminating attributes and release the deferred corrections.

The dataset's weakness was never missing objects - it was that whole clusters
of objects (the superheroes, the furniture, the sky objects) were separated by
a single attribute. One wrong answer from the player made them unrecoverable,
and five factually wrong values could not be corrected at all because fixing
them would have made two objects identical.

This script adds attributes that genuinely distinguish those clusters, then
applies the corrections that were blocked without them.

Idempotent: safe to re-run.
Run:  python scripts/phase2_extend_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

SOURCE_DIR = cfg.DATA_DIR / "source"

# ---------------------------------------------------------------------------
# New attributes: feature -> (question, group, objects it is true for)
# ---------------------------------------------------------------------------
NEW_FEATURES = {
    # --- the superhero / cartoon cluster (8 objects, all within 2 answers) --
    "is_superhero": ("Is it a superhero?", "fiction",
                     ["Thor", "Captian_America", "Hulk", "Spider_Man", "Batman"]),
    "is_human": ("Is it a human being?", "fiction",
                 ["Captian_America", "Spider_Man", "Batman", "sinchan",
                  "Chota_Bheem"]),
    "is_robot": ("Is it a robot?", "fiction", ["Doreamon"]),
    "is_child": ("Is it a child?", "fiction", ["sinchan", "Chota_Bheem"]),
    "is_teenager": ("Is it a teenager?", "fiction", ["Spider_Man"]),
    "has_cape": ("Does it wear a cape?", "fiction", ["Thor", "Batman"]),
    "wears_mask": ("Does it wear a mask?", "fiction",
                   ["Captian_America", "Spider_Man", "Batman"]),
    "has_weapon": ("Does it carry a weapon?", "fiction",
                   ["Thor", "Captian_America", "Batman"]),
    "is_green": ("Is it green?", "fiction", ["Hulk"]),
    "is_japanese": ("Is it a Japanese character?", "fiction",
                    ["Doreamon", "sinchan"]),
    "is_soldier": ("Is it a soldier?", "fiction", ["Captian_America"]),
    "can_climb_walls": ("Can it climb walls?", "fiction", ["Spider_Man"]),

    # --- the animal cluster -------------------------------------------------
    "is_bird": ("Is it a bird?", "nature", ["Eagle", "Penguin"]),
    "is_reptile": ("Is it a reptile?", "nature", ["Snake"]),
    "has_feathers": ("Does it have feathers?", "nature", ["Eagle", "Penguin"]),
    "has_fins": ("Does it have fins?", "nature", ["Dolphin", "Fish"]),
    "lays_eggs": ("Does it lay eggs?", "nature",
                  ["Eagle", "Penguin", "Snake", "Fish"]),

    # --- the sky / plant cluster -------------------------------------------
    "is_in_sky": ("Is it found in the sky?", "nature",
                  ["Clouds", "Stars", "Sun"]),
    "gives_light": ("Does it give off light?", "nature", ["Stars", "Sun"]),
    "visible_at_night": ("Is it visible at night?", "nature", ["Stars"]),
    "produces_rain": ("Does it produce rain?", "nature", ["Clouds"]),
    "is_plant": ("Is it a plant?", "nature", ["Tree", "Rose"]),

    # --- the vehicle / furniture cluster -----------------------------------
    "has_wheels": ("Does it have wheels?", "parts", ["car", "Bicycle"]),
    "has_engine": ("Does it have an engine?", "parts", ["car"]),
    "is_furniture": ("Is it a piece of furniture?", "made", ["Chair", "Table"]),
    "has_flat_surface": ("Does it have a flat surface on top?", "shape",
                         ["Table"]),
    "holds_objects": ("Is it used to hold or carry other things?", "use",
                      ["Table", "Plates", "Flowerpot", "Handbag", "Wallet"]),
    "is_container": ("Is it a container?", "use",
                     ["Matchbox", "Gas", "Paint", "Flowerpot", "Handbag",
                      "Wallet"]),
    "fits_in_pocket": ("Can it fit in your pocket?", "shape",
                       ["phone", "Matchbox", "Wallet", "Lipstick", "Comb",
                        "Ring", "Condom", "Hairband", "Earbuds",
                        "Tissue_Paper"]),

    # --- the paper cluster --------------------------------------------------
    "is_disposable": ("Is it thrown away after use?", "use",
                      ["Newspaper", "Tissue_Paper", "Condom", "Matchbox"]),
    "has_pages": ("Does it have pages?", "parts", ["Book", "Newspaper"]),
    "published_daily": ("Is a new one published every day?", "use",
                        ["Newspaper"]),

    # --- the wearable cluster ----------------------------------------------
    "is_applied_to_skin": ("Is it applied onto the skin?", "wear",
                           ["Lipstick", "Perfume"]),
    "is_made_of_cloth": ("Is it made of cloth?", "shape",
                         ["Cap", "Shoes", "Handbag"]),
    "worn_in_pairs": ("Is it worn as a pair?", "wear", ["Shoes", "Earbuds"]),
    "worn_on_head": ("Is it worn on the head?", "wear", ["Cap", "Hairband"]),

    # --- the electronics cluster -------------------------------------------
    "has_keyboard": ("Does it have a keyboard?", "parts", ["Laptop"]),
    "can_make_calls": ("Can you make phone calls with it?", "use", ["phone"]),
    "is_wall_mounted": ("Is it mounted on a wall?", "made", ["Ac"]),
}

# ---------------------------------------------------------------------------
# Corrections that the new attributes finally make safe to apply.
# (object, attribute, add_or_remove, reason)
# ---------------------------------------------------------------------------
RELEASED_CORRECTIONS = [
    ("Captian_America", "can_fly", "remove",
     "Captain America cannot fly; was blocked because he and Hulk differed only here"),
    ("Spider_Man", "can_fly", "remove",
     "Spider-Man swings rather than flies"),
    ("Batman", "can_fly", "remove",
     "Batman glides rather than flies"),
    ("Batman", "is_superpower", "remove",
     "Batman has no superpowers; he is a non-powered human"),
    ("Spider_Man", "is_worn_on_body", "remove",
     "A costume is not the character; this also broke is_worn_on_body -> is_wearable"),
    ("Clouds", "is_round", "remove",
     "Clouds are not round; was blocked because Clouds and Stars differed only here"),
    ("Laptop", "can_make_sound", "add",
     "Laptops produce sound; was blocked because phone and Laptop differed only here"),
]

# ---------------------------------------------------------------------------
# Additional implication rules the new attributes make meaningful.
# Deliberately omitted: gives_light -> is_in_sky and visible_at_night ->
# is_in_sky, which hold today but are not real implications (a lamp gives
# light and is not in the sky) and would block sensible future objects.
# ---------------------------------------------------------------------------
NEW_IMPLICATIONS = [
    ["is_superhero", "is_character"],
    ["is_robot", "is_character"],
    ["is_japanese", "is_character"],
    ["is_teenager", "is_human"],
    ["is_child", "is_human"],
    ["is_soldier", "is_human"],
    ["is_bird", "is_animal"],
    ["is_reptile", "is_animal"],
    ["has_feathers", "is_bird"],
    ["has_fins", "is_animal"],
    ["is_plant", "is_natural"],
    ["produces_rain", "is_in_sky"],
    ["is_in_sky", "is_natural"],
    ["has_wheels", "made_by_humans"],
    ["is_furniture", "made_by_humans"],
    ["is_container", "made_by_humans"],
    ["is_disposable", "made_by_humans"],
    ["is_made_of_cloth", "made_by_humans"],
    ["holds_objects", "made_by_humans"],
    ["is_wall_mounted", "made_by_humans"],
    ["has_keyboard", "is_electronic"],
    ["can_make_calls", "is_electronic"],
    ["has_pages", "is_paper_like"],
    ["published_daily", "has_pages"],
    ["is_applied_to_skin", "is_beauty"],
    ["worn_in_pairs", "is_worn_on_body"],
    ["worn_on_head", "is_worn_on_body"],
]


def main() -> int:
    features = json.loads((SOURCE_DIR / "features.json").read_text(encoding="utf-8"))
    objects = json.loads((SOURCE_DIR / "objects.json").read_text(encoding="utf-8"))
    implications = json.loads(
        (SOURCE_DIR / "implications.json").read_text(encoding="utf-8"))

    print("=" * 72)
    print("PHASE 2: EXTEND ATTRIBUTES")
    print("=" * 72)
    print("  starting from %d objects x %d features"
          % (len(objects), len(features)))

    # ---- 1. new feature definitions ---------------------------------------
    added_feats = 0
    for feat, (question, group, holders) in NEW_FEATURES.items():
        unknown = [o for o in holders if o not in objects]
        if unknown:
            raise SystemExit("feature %r lists unknown objects: %s" % (feat, unknown))
        if feat not in features:
            features[feat] = {"question": question, "group": group}
            added_feats += 1
        for obj in holders:
            attrs = objects[obj]["attributes"]
            if feat not in attrs:
                attrs.append(feat)
    print("  added %d new attributes" % added_feats)

    # ---- 2. released corrections ------------------------------------------
    applied = 0
    for obj, feat, action, _reason in RELEASED_CORRECTIONS:
        if obj not in objects:
            raise SystemExit("correction references unknown object %r" % obj)
        if feat not in features:
            raise SystemExit("correction references unknown feature %r" % feat)
        attrs = objects[obj]["attributes"]
        if action == "remove" and feat in attrs:
            attrs.remove(feat)
            applied += 1
        elif action == "add" and feat not in attrs:
            attrs.append(feat)
            applied += 1
    print("  applied %d previously blocked corrections" % applied)

    # ---- 3. new implication rules -----------------------------------------
    have = {tuple(r) for r in implications}
    added_rules = 0
    for rule in NEW_IMPLICATIONS:
        if tuple(rule) not in have:
            implications.append(rule)
            have.add(tuple(rule))
            added_rules += 1
    print("  added %d implication rules" % added_rules)

    # ---- 4. keep attribute lists in feature order -------------------------
    order = list(features)
    rank = {f: i for i, f in enumerate(order)}
    for obj in objects.values():
        obj["attributes"] = sorted(set(obj["attributes"]), key=lambda f: rank[f])

    (SOURCE_DIR / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8")
    (SOURCE_DIR / "objects.json").write_text(
        json.dumps(objects, indent=2), encoding="utf-8")
    (SOURCE_DIR / "implications.json").write_text(
        json.dumps(implications, indent=2), encoding="utf-8")

    print()
    print("  now %d objects x %d features, %d implication rules"
          % (len(objects), len(features), len(implications)))
    print("  next: python scripts/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
