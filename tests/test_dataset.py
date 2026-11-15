"""Dataset invariants.

These lock in the Phase 1 fixes so the bugs cannot silently come back.
Run:  python -m pytest -q
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    with cfg.DATASET.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    features = header[1:]
    names = [r[0] for r in body]
    vectors = [[int(v) for v in r[1:]] for r in body]
    return names, features, vectors


@pytest.fixture(scope="module")
def category_map():
    with cfg.CATEGORY_MAP.open(newline="", encoding="utf-8") as fh:
        return {r["object"]: r["category"] for r in csv.DictReader(fh)}


def test_dataset_exists():
    assert cfg.DATASET.exists(), "run python scripts/build_dataset.py"


def test_all_values_binary(dataset):
    _, features, vectors = dataset
    for vec in vectors:
        assert len(vec) == len(features)
        assert all(v in (0, 1) for v in vec)


def test_object_names_unique(dataset):
    names, _, _ = dataset
    assert len(set(names)) == len(names)


def test_no_identical_feature_vectors(dataset):
    """Two objects sharing a vector could never be told apart by any question."""
    names, _, vectors = dataset
    seen = {}
    collisions = []
    for name, vec in zip(names, vectors):
        key = tuple(vec)
        if key in seen:
            collisions.append((seen[key], name))
        seen[key] = name
    assert not collisions, "identical feature vectors: %s" % collisions


def test_legacy_misspelled_columns_are_gone(dataset):
    """The misspellings that silently produced all-zero feature columns."""
    _, features, _ = dataset
    for bad in cfg.LEGACY_COLUMN_RENAMES:
        assert bad not in features, "misspelled column %r is back" % bad
    for good in cfg.LEGACY_COLUMN_RENAMES.values():
        assert good in features, "corrected column %r is missing" % good


def test_no_all_zero_or_all_one_columns(dataset):
    """A constant column carries no information and cannot narrow anything."""
    names, features, vectors = dataset
    for k, feat in enumerate(features):
        col = [v[k] for v in vectors]
        assert any(col), "feature %r is 0 for every object" % feat
        assert not all(col), "feature %r is 1 for every object" % feat


def test_every_object_has_a_category(dataset, category_map):
    names, _, _ = dataset
    missing = [n for n in names if n not in category_map]
    assert not missing, "objects with no category: %s" % missing


def test_category_map_has_no_unknown_objects(dataset, category_map):
    names, _, _ = dataset
    orphans = [n for n in category_map if n not in set(names)]
    assert not orphans, "category map lists unknown objects: %s" % orphans


def test_every_feature_has_question_text(dataset):
    _, features, _ = dataset
    qmap = json.loads(cfg.QUESTION_MAP.read_text(encoding="utf-8"))
    missing = [f for f in features if f not in qmap]
    assert not missing, "features with no question: %s" % missing


def test_question_order_covers_every_feature(dataset):
    _, features, _ = dataset
    order = json.loads(cfg.QUESTION_ORDER.read_text(encoding="utf-8"))
    assert sorted(order) == sorted(features)
    assert len(order) == len(set(order)), "question_order has duplicates"


def test_feature_columns_json_matches_dataset(dataset):
    _, features, _ = dataset
    declared = json.loads(cfg.FEATURE_COLUMNS.read_text(encoding="utf-8"))
    assert declared == features


def test_corrections_are_traceable(dataset):
    """Every applied correction must name a real object and feature."""
    names, features, _ = dataset
    with cfg.CORRECTIONS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assert row["object"] in names
            assert row["feature"] in features
            assert row["from"] in ("0", "1")
            assert row["to"] in ("0", "1")
            assert row["from"] != row["to"], "no-op correction for %s" % row["object"]
            assert row["reason"].strip(), "correction without a reason"


def test_min_hamming_distance_target(dataset):
    """Any single wrong answer should still be recoverable."""
    names, _, vectors = dataset
    worst = min(
        (sum(a != b for a, b in zip(vectors[i], vectors[j])), names[i], names[j])
        for i, j in itertools.combinations(range(len(names)), 2)
    )
    assert worst[0] >= cfg.MIN_HAMMING_TARGET, (
        "%s vs %s differ by only %d answer(s)" % (worst[1], worst[2], worst[0])
    )


def test_implication_rules_hold(dataset):
    """Semantic rules like is_mammal -> is_animal must not be violated.

    These catch authoring mistakes automatically: labelling a bus with
    has_screen but not is_electronic, for example.
    """
    names, features, vectors = dataset
    rules = json.loads(
        (cfg.DATA_DIR / "source" / "implications.json").read_text(encoding="utf-8"))
    assert rules, "no implication rules defined"
    idx = {f: i for i, f in enumerate(features)}
    violations = []
    for a, b in rules:
        assert a in idx, "rule references unknown feature %r" % a
        assert b in idx, "rule references unknown feature %r" % b
        for name, vec in zip(names, vectors):
            if vec[idx[a]] == 1 and vec[idx[b]] == 0:
                violations.append("%s: %s but not %s" % (name, a, b))
    assert not violations, violations


def test_source_spec_matches_compiled_dataset(dataset):
    """The CSV must be exactly what the authoring spec compiles to."""
    names, features, vectors = dataset
    src = cfg.DATA_DIR / "source"
    spec_features = json.loads((src / "features.json").read_text(encoding="utf-8"))
    spec_objects = json.loads((src / "objects.json").read_text(encoding="utf-8"))
    assert list(spec_features) == features
    assert list(spec_objects) == names
    for name, vec in zip(names, vectors):
        attrs = set(spec_objects[name]["attributes"])
        expected = [1 if f in attrs else 0 for f in features]
        assert vec == expected, "%s does not match its spec" % name


def test_no_pairs_within_two_answers(dataset):
    """Stronger than the minimum-distance gate: nothing may be at 1 or 2."""
    names, _, vectors = dataset
    close = []
    for i, j in itertools.combinations(range(len(names)), 2):
        d = sum(a != b for a, b in zip(vectors[i], vectors[j]))
        if d < 3:
            close.append("%s vs %s (%d)" % (names[i], names[j], d))
    assert not close, close
