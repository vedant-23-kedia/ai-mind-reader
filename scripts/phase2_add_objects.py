"""Phase 2c: grow the object set and make singleton attributes generalise.

After 2b, 26 of 91 attributes were true for exactly one object. A feature
true for a single object is a giveaway flag: it identifies that object
outright and teaches a model nothing. The fix is not to delete good features
like worn_on_feet - it is that the object set was too small for them to
describe more than one thing. Adding Socks and Sandals turns worn_on_feet
into a real attribute.

Objects were chosen to be mutually distinct. A Cat next to a Dog would share
every attribute in the schema and need new ones invented purely to separate
them, which is how giveaway flags are born.

Idempotent: safe to re-run.
Run:  python scripts/phase2_add_objects.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

SOURCE_DIR = cfg.DATA_DIR / "source"

# ---------------------------------------------------------------------------
# Attributes the expanded object set needs. Several also apply to objects
# that already existed, listed here so the labelling stays consistent.
# feature -> (question, group, objects it is true for)
# ---------------------------------------------------------------------------
EXTRA_FEATURES = {
    "has_tail": ("Does it have a tail?", "nature",
                 ["Dog", "Lion", "Dolphin", "Fish", "Snake", "Elephant",
                  "Monkey", "Cow", "Crocodile", "Lizard", "Parrot", "Horse"]),
    "has_hands": ("Does it have hands?", "nature",
                  ["Thor", "Captian_America", "Hulk", "Spider_Man", "Batman",
                   "sinchan", "Chota_Bheem", "Monkey"]),
    "has_scales": ("Does it have scales?", "nature",
                   ["Fish", "Snake", "Crocodile", "Lizard"]),
    "has_horns": ("Does it have horns?", "nature", ["Cow", "Goat"]),
    "gives_milk": ("Does it give milk?", "nature", ["Cow", "Goat"]),
    "is_farm_animal": ("Is it a farm animal?", "nature",
                       ["Cow", "Horse", "Goat"]),
    "can_talk": ("Can it imitate human speech?", "nature", ["Parrot"]),
    "is_insect": ("Is it an insect?", "nature", ["Butterfly", "Mosquito"]),
    "is_sweet": ("Does it taste sweet?", "food",
                 ["Chocolate", "Ice_Cream", "Banana", "Honey"]),
    "is_grain": ("Is it a grain or made from grain?", "food",
                 ["Rice", "Bread"]),
    "is_hot_drink": ("Is it usually drunk hot?", "food", ["Tea"]),
    "is_frozen": ("Is it served frozen?", "food", ["Ice_Cream"]),
    "is_two_wheeler": ("Does it have exactly two wheels?", "parts",
                       ["Bicycle", "Motorcycle"]),
    "is_public_transport": ("Is it public transport?", "made",
                            ["Bus", "Train"]),
    "floats_on_water": ("Does it float on water?", "nature", ["Ship", "Boat"]),
    "runs_on_rails": ("Does it run on rails?", "parts", ["Train"]),
    "cleans_things": ("Is it used for cleaning?", "use",
                      ["Toothbrush", "Soap", "Washing_Machine", "Broom"]),
    "heats_food": ("Is it used to heat food?", "use", ["Microwave", "Gas"]),
    "keeps_things_cold": ("Does it keep things cold?", "use",
                          ["Refrigerator"]),
    "is_cutlery": ("Is it cutlery?", "use", ["Spoon", "Knife"]),
    "used_for_writing": ("Is it used for writing?", "use", ["Pen", "Pencil"]),
    "is_footwear": ("Is it footwear?", "wear",
                    ["Shoes", "Socks", "Sandals"]),
    "is_jewellery": ("Is it jewellery?", "wear", ["Ring", "Earrings"]),
    "protects_head": ("Does it protect the head?", "wear", ["Cap", "Helmet"]),
    "has_mirror": ("Does it have a reflective surface?", "parts", ["Mirror"]),
    "is_medicine": ("Is it medicine?", "use", ["Tablet_Medicine"]),
    # Lion and Elephant were otherwise identical: both large wild mammals
    # with four legs and a tail. These separate them on real distinctions
    # rather than inventing a flag that fits only one of them.
    "is_carnivore": ("Does it eat meat?", "nature",
                     ["Lion", "Crocodile", "Eagle", "Snake", "Dolphin"]),
    "used_for_riding": ("Can you ride on it?", "use", ["Horse", "Elephant"]),
}

# Attributes to take away: (object, attribute, reason)
REMOVE_ATTRS = [
    ("Bus", "has_screen",
     "a bus is not an electronic device; this broke has_screen -> is_electronic"),
    ("Horse", "has_seat",
     "a saddle is not a built-in seat; used_for_riding says this properly"),
]

# Corrections to existing objects that the expanded set made necessary.
ADD_ATTRS = [
    ("Lion", "has_sharp_edge", "lions have claws and fangs, like the other predators"),
]

# ---------------------------------------------------------------------------
# New objects: name -> (category, attributes)
# ---------------------------------------------------------------------------
NEW_OBJECTS = {
    # ---- Animals ----------------------------------------------------------
    "Elephant": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "is_big", "can_make_sound",
        "is_natural", "is_reusable", "is_wild", "is_mammal"]),
    "Monkey": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "can_make_sound", "is_natural",
        "is_reusable", "is_wild", "is_mammal", "can_climb_walls"]),
    "Cow": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "is_big", "can_make_sound",
        "is_natural", "is_reusable", "is_domestic_pet", "is_mammal"]),
    "Horse": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "is_big", "can_make_sound",
        "is_natural", "is_reusable", "is_domestic_pet", "is_mammal",
        "has_seat"]),
    "Goat": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "can_make_sound", "is_natural",
        "is_reusable", "is_domestic_pet", "is_mammal"]),
    "Crocodile": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "can_swim", "is_big",
        "can_make_sound", "is_natural", "lives_in_water", "is_reusable",
        "is_wild", "is_reptile", "lays_eggs", "has_sharp_edge"]),
    "Lizard": ("Animal", [
        "is_living", "is_animal", "has_4_legs", "is_natural", "is_reusable",
        "is_wild", "is_reptile", "lays_eggs", "can_climb_walls"]),
    "Frog": ("Animal", [
        "is_living", "is_animal", "can_swim", "can_make_sound", "is_natural",
        "lives_in_water", "is_reusable", "is_wild", "is_green", "lays_eggs"]),
    "Parrot": ("Animal", [
        "is_living", "is_animal", "can_fly", "can_make_sound", "is_natural",
        "is_reusable", "is_domestic_pet", "is_bird", "has_feathers",
        "lays_eggs", "is_green"]),
    "Butterfly": ("Animal", [
        "is_living", "is_animal", "can_fly", "is_natural", "is_reusable",
        "is_wild", "lays_eggs", "is_beauty", "is_purchased"]),
    "Mosquito": ("Animal", [
        "is_living", "is_animal", "can_fly", "can_make_sound", "is_natural",
        "is_wild", "lays_eggs"]),

    # ---- Food -------------------------------------------------------------
    "Banana": ("Food", [
        "is_food", "is_purchased", "is_natural", "is_edible", "used_daily",
        "is_held_in_hand", "fits_in_pocket"]),
    "Rice": ("Food", [
        "is_food", "is_purchased", "is_necessary", "is_natural", "is_edible",
        "used_daily", "used_for_cooking", "made_by_humans"]),
    "Bread": ("Food", [
        "is_food", "made_by_humans", "is_purchased", "is_necessary",
        "is_edible", "used_daily", "is_held_in_hand"]),
    "Milk": ("Food", [
        "is_food", "is_liquid", "is_purchased", "is_necessary", "is_natural",
        "is_edible", "used_daily"]),
    "Tea": ("Food", [
        "is_food", "is_liquid", "made_by_humans", "is_purchased", "is_edible",
        "used_daily", "is_held_in_hand"]),
    "Chocolate": ("Food", [
        "is_food", "made_by_humans", "is_purchased", "is_edible",
        "is_held_in_hand", "fits_in_pocket"]),
    "Ice_Cream": ("Food", [
        "is_food", "made_by_humans", "is_purchased", "is_edible",
        "is_held_in_hand", "is_round"]),
    "Honey": ("Food", [
        "is_food", "is_liquid", "is_purchased", "is_natural", "is_edible",
        "is_container", "made_by_humans"]),

    # ---- Vehicles ---------------------------------------------------------
    "Bus": ("Vehicle", [
        "is_big", "made_by_humans", "is_purchased", "is_necessary",
        "can_make_sound", "is_vehicle", "is_reusable", "used_daily",
        "has_wheels", "has_engine", "has_seat", "has_screen"]),
    "Train": ("Vehicle", [
        "is_big", "made_by_humans", "is_purchased", "is_necessary",
        "can_make_sound", "is_vehicle", "is_reusable", "has_wheels",
        "has_engine", "has_seat"]),
    "Motorcycle": ("Vehicle", [
        "made_by_humans", "is_purchased", "can_make_sound", "is_vehicle",
        "is_reusable", "used_daily", "has_wheels", "has_engine", "has_seat"]),
    "Ship": ("Vehicle", [
        "is_big", "made_by_humans", "is_purchased", "can_make_sound",
        "is_vehicle", "is_reusable", "has_engine", "has_seat"]),
    "Boat": ("Vehicle", [
        "made_by_humans", "is_purchased", "is_vehicle", "is_reusable",
        "has_seat"]),
    "Helicopter": ("Vehicle", [
        "can_fly", "is_big", "made_by_humans", "is_purchased",
        "can_make_sound", "is_vehicle", "is_reusable", "has_engine",
        "has_blades", "has_seat"]),

    # ---- Electronics ------------------------------------------------------
    "Television": ("Electronic", [
        "is_big", "made_by_humans", "is_electronic", "used_daily",
        "is_purchased", "can_make_sound", "has_screen", "is_reusable",
        "has_cable", "is_wall_mounted"]),
    "Refrigerator": ("Electronic", [
        "is_big", "made_by_humans", "is_electronic", "used_daily",
        "is_purchased", "is_necessary", "is_reusable", "has_cable",
        "is_container", "holds_food"]),
    "Washing_Machine": ("Electronic", [
        "is_big", "made_by_humans", "is_electronic", "used_daily",
        "is_purchased", "is_necessary", "can_make_sound", "is_reusable",
        "has_cable", "is_container"]),
    "Microwave": ("Electronic", [
        "made_by_humans", "is_electronic", "used_daily", "is_purchased",
        "is_reusable", "has_cable", "is_container", "used_for_cooking"]),
    "Smartwatch": ("Electronic", [
        "made_by_humans", "is_electronic", "is_wearable", "is_worn_on_body",
        "is_purchased", "can_make_sound", "has_screen", "is_reusable",
        "shows_time", "fits_in_pocket", "can_make_calls", "is_held_in_hand"]),
    "Mixer_Grinder": ("Electronic", [
        "made_by_humans", "is_electronic", "is_purchased", "can_make_sound",
        "is_reusable", "has_cable", "has_blades", "is_container",
        "used_for_cooking"]),

    # ---- Wearables --------------------------------------------------------
    "Socks": ("Wearable", [
        "made_by_humans", "is_wearable", "is_worn_on_body", "worn_on_feet",
        "is_purchased", "used_daily", "is_reusable", "worn_in_pairs",
        "is_made_of_cloth", "is_elastic"]),
    "Sandals": ("Wearable", [
        "made_by_humans", "is_wearable", "is_worn_on_body", "worn_on_feet",
        "is_purchased", "used_daily", "is_necessary", "is_reusable",
        "worn_in_pairs"]),
    "Trousers": ("Wearable", [
        "made_by_humans", "is_wearable", "is_worn_on_body", "worn_on_waist",
        "is_purchased", "used_daily", "is_necessary", "is_reusable",
        "is_made_of_cloth"]),
    "Shirt": ("Wearable", [
        "made_by_humans", "is_wearable", "is_worn_on_body", "is_purchased",
        "used_daily", "is_necessary", "is_reusable", "is_made_of_cloth"]),
    "Helmet": ("Wearable", [
        "made_by_humans", "is_wearable", "is_worn_on_body", "worn_on_head",
        "is_purchased", "is_necessary", "is_reusable", "is_round",
        "is_held_in_hand"]),
    "Earrings": ("Wearable", [
        "made_by_humans", "is_wearable", "is_worn_on_body", "worn_on_ears",
        "is_purchased", "is_beauty", "is_reusable", "worn_in_pairs",
        "fits_in_pocket", "is_held_in_hand"]),

    # ---- Natural ----------------------------------------------------------
    "Moon": ("Natural", [
        "is_big", "is_natural", "is_reusable", "is_round", "is_in_sky",
        "gives_light", "visible_at_night"]),
    "River": ("Natural", [
        "is_liquid", "is_big", "is_natural", "is_necessary", "is_reusable",
        "can_make_sound", "can_swim"]),
    "Mountain": ("Natural", [
        "is_big", "is_natural", "is_reusable"]),
    "Leaf": ("Natural", [
        "is_living", "is_natural", "is_green", "is_plant", "is_reusable",
        "is_held_in_hand", "fits_in_pocket"]),
    "Grass": ("Natural", [
        "is_living", "is_natural", "is_green", "is_plant", "is_reusable",
        "used_daily"]),

    # ---- Daily objects ----------------------------------------------------
    "Spoon": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand", "holds_food"]),
    "Knife": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand", "has_sharp_edge",
        "used_for_cooking"]),
    "Scissors": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_reusable", "is_held_in_hand",
        "has_sharp_edge", "has_blades"]),
    "Mirror": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_beauty", "is_wall_mounted"]),
    "Bed": ("Daily Object", [
        "is_big", "made_by_humans", "is_purchased", "is_necessary",
        "used_daily", "is_reusable", "is_furniture", "has_flat_surface",
        "has_4_legs"]),
    "Pen": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand", "fits_in_pocket", "is_carried"]),
    "Pencil": ("Daily Object", [
        "made_by_humans", "is_purchased", "used_daily", "is_held_in_hand",
        "fits_in_pocket", "is_carried", "has_sharp_edge"]),
    "Toothbrush": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand", "fits_in_pocket", "is_beauty"]),
    "Soap": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_held_in_hand", "is_beauty", "fits_in_pocket"]),
    "Umbrella": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_reusable", "is_held_in_hand",
        "is_carried", "is_made_of_cloth"]),
    "Broom": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand"]),
    "Key": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand", "fits_in_pocket", "is_carried",
        "has_sharp_edge"]),
    "Bottle": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "used_daily",
        "is_reusable", "is_held_in_hand", "is_carried", "is_container"]),
    "Tablet_Medicine": ("Daily Object", [
        "made_by_humans", "is_purchased", "is_necessary", "is_edible",
        "is_round", "fits_in_pocket", "is_held_in_hand", "is_disposable"]),

    # ---- Fictional characters --------------------------------------------
    "Iron_Man": ("Marvel", [
        "is_living", "is_character", "is_fictional", "is_marvel",
        "is_superhero", "is_human", "can_fly", "is_big", "can_make_sound",
        "has_weapon", "wears_mask", "is_reusable", "has_hands"]),
    "Black_Panther": ("Marvel", [
        "is_living", "is_character", "is_fictional", "is_marvel",
        "is_superhero", "is_human", "is_big", "can_make_sound", "wears_mask",
        "is_superpower", "is_reusable", "has_hands", "can_climb_walls"]),
    "Wall_E": ("Cartoon", [
        "is_character", "is_cartoons", "is_fictional", "is_robot",
        "can_make_sound", "is_reusable", "has_wheels", "made_by_humans",
        "has_hands"]),
    "Ben_10": ("Cartoon", [
        "is_living", "is_character", "is_cartoons", "is_fictional",
        "is_teenager", "is_human", "is_superpower", "can_make_sound",
        "is_reusable", "has_hands", "is_green"]),
    "Motu_Patlu": ("Cartoon", [
        "is_living", "is_character", "is_cartoons", "is_fictional",
        "is_human", "can_make_sound", "is_reusable", "is_indian",
        "has_hands"]),
}


def main() -> int:
    features = json.loads((SOURCE_DIR / "features.json").read_text(encoding="utf-8"))
    objects = json.loads((SOURCE_DIR / "objects.json").read_text(encoding="utf-8"))

    print("=" * 72)
    print("PHASE 2c: GROW THE OBJECT SET")
    print("=" * 72)
    print("  starting from %d objects x %d features"
          % (len(objects), len(features)))

    # ---- 1. new objects ---------------------------------------------------
    added_objs = 0
    for name, (category, attrs) in NEW_OBJECTS.items():
        if name not in objects:
            objects[name] = {"category": category, "attributes": list(attrs)}
            added_objs += 1
        else:
            objects[name]["category"] = category
            for a in attrs:
                if a not in objects[name]["attributes"]:
                    objects[name]["attributes"].append(a)
    print("  added %d objects" % added_objs)

    # ---- 2. extra attributes ----------------------------------------------
    added_feats = 0
    for feat, (question, group, holders) in EXTRA_FEATURES.items():
        unknown = [o for o in holders if o not in objects]
        if unknown:
            raise SystemExit(
                "attribute %r lists objects that do not exist: %s" % (feat, unknown))
        if feat not in features:
            features[feat] = {"question": question, "group": group}
            added_feats += 1
        for obj in holders:
            if feat not in objects[obj]["attributes"]:
                objects[obj]["attributes"].append(feat)
    print("  added %d attributes" % added_feats)

    # ---- 2b. targeted removals and additions ------------------------------
    removed = 0
    for obj, feat, _reason in REMOVE_ATTRS:
        if obj not in objects:
            raise SystemExit("removal references unknown object %r" % obj)
        if feat in objects[obj]["attributes"]:
            objects[obj]["attributes"].remove(feat)
            removed += 1
    added_attrs = 0
    for obj, feat, _reason in ADD_ATTRS:
        if obj not in objects:
            raise SystemExit("addition references unknown object %r" % obj)
        if feat not in features:
            raise SystemExit("addition references unknown attribute %r" % feat)
        if feat not in objects[obj]["attributes"]:
            objects[obj]["attributes"].append(feat)
            added_attrs += 1
    print("  removed %d attribute values, added %d" % (removed, added_attrs))

    # ---- 3. keep attribute lists in feature order -------------------------
    rank = {f: i for i, f in enumerate(features)}
    unknown_attrs = {}
    for name, spec in objects.items():
        bad = [a for a in spec["attributes"] if a not in rank]
        if bad:
            unknown_attrs[name] = bad
        spec["attributes"] = sorted(
            {a for a in spec["attributes"] if a in rank}, key=lambda f: rank[f])
    if unknown_attrs:
        raise SystemExit("objects reference unknown attributes: %s" % unknown_attrs)

    (SOURCE_DIR / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8")
    (SOURCE_DIR / "objects.json").write_text(
        json.dumps(objects, indent=2), encoding="utf-8")

    print()
    print("  now %d objects x %d features" % (len(objects), len(features)))
    print("  next: python scripts/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
