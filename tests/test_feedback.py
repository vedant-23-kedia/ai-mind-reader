"""Tests for learning from recorded games.

The estimator's job is to recover a per-attribute error rate from answer
sheets. These tests plant a known rate and check it comes back, because an
estimator that silently returns the prior would look fine otherwise.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import BeliefEngine  # noqa: E402
from mindreader.feedback import (  # noqa: E402
    MAX_RATE,
    MIN_RATE,
    estimate_attribute_noise,
    load_feedback,
    load_noise_profile,
    new_object_candidates,
    write_noise_profile,
)


@pytest.fixture(scope="module")
def protos():
    return load_prototypes()


def write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "outcome", "guessed", "actual", "n_answers",
                    "answers_json"])
        for r in rows:
            w.writerow(["2026-01-01T00:00:00+00:00", r.get("outcome", "wrong"),
                        r.get("guessed", "Dog"), r["actual"],
                        len(r["answers"]), json.dumps(r["answers"])])


def truth(protos, obj: str, feat: str) -> int:
    return int(protos.X[protos.index_of(obj), protos.features.index(feat)])


# ---- reading the log ----------------------------------------------------

def test_missing_log_is_not_an_error(tmp_path):
    assert load_feedback(tmp_path / "nope.csv") == []


def test_reads_rows(tmp_path, protos):
    p = tmp_path / "log.csv"
    write_log(p, [{"actual": "Dog", "answers": {"is_animal": 1}}])
    rows = load_feedback(p)
    assert len(rows) == 1
    assert rows[0].actual == "Dog"
    assert rows[0].answers == {"is_animal": 1}


def test_malformed_rows_are_skipped_not_fatal(tmp_path):
    """A half-written final row must not break the learning step."""
    p = tmp_path / "log.csv"
    p.write_text(
        "timestamp,outcome,guessed,actual,n_answers,answers_json\n"
        "t,wrong,Dog,Lion,1,\"{\"\"is_animal\"\": 1}\"\n"
        "t,wrong,Dog,,1,{}\n"                      # no actual
        "t,wrong,Dog,Cow,1,not-json\n"             # bad json
        "t,wrong,Dog,Cat,1,\"[1,2,3]\"\n",         # json but not a dict
        encoding="utf-8")
    rows = load_feedback(p)
    assert len(rows) == 1 and rows[0].actual == "Lion"


def test_old_logs_without_an_outcome_column_still_load(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text(
        "timestamp,guessed,actual,n_answers,answers_json\n"
        "t,Dog,Lion,1,\"{\"\"is_animal\"\": 1}\"\n", encoding="utf-8")
    rows = load_feedback(p)
    assert len(rows) == 1
    assert rows[0].outcome == "wrong"


# ---- estimating the rates ----------------------------------------------

def test_recovers_a_planted_error_rate(tmp_path, protos):
    """Half the answers to is_big are wrong; the estimate should say so."""
    feat = "is_big"
    rows = []
    for i in range(400):
        t = truth(protos, "Dog", feat)
        rows.append({"actual": "Dog",
                     "answers": {feat: (1 - t) if i % 2 == 0 else t}})
    p = tmp_path / "log.csv"
    write_log(p, rows)
    est = estimate_attribute_noise(load_feedback(p), protos)
    # 50% planted, clamped to MAX_RATE because an answer wrong half the time
    # carries no information at all
    assert est.per_attribute[feat] == pytest.approx(MAX_RATE, abs=0.02)


def test_distinguishes_a_vague_attribute_from_a_solid_one(tmp_path, protos):
    rows = []
    for i in range(300):
        tb = truth(protos, "Laptop", "is_big")
        ta = truth(protos, "Laptop", "is_animal")
        rows.append({"actual": "Laptop", "answers": {
            "is_big": (1 - tb) if i % 3 == 0 else tb,     # ~33% wrong
            "is_animal": ta,                               # never wrong
        }})
    p = tmp_path / "log.csv"
    write_log(p, rows)
    est = estimate_attribute_noise(load_feedback(p), protos)
    assert est.per_attribute["is_big"] > 0.25
    assert est.per_attribute["is_animal"] < 0.08
    assert est.per_attribute["is_big"] > est.per_attribute["is_animal"] * 3


def test_smoothing_protects_against_a_single_observation(tmp_path, protos):
    """One wrong answer must not declare an attribute 100% unreliable."""
    feat = "is_round"
    t = truth(protos, "Dog", feat)
    p = tmp_path / "log.csv"
    write_log(p, [{"actual": "Dog", "answers": {feat: 1 - t}}])
    est = estimate_attribute_noise(load_feedback(p), protos)
    assert est.per_attribute[feat] < 0.3, (
        "a single observation swung the estimate to %.2f"
        % est.per_attribute[feat])


def test_maybe_answers_are_not_counted_as_errors(tmp_path, protos):
    p = tmp_path / "log.csv"
    write_log(p, [{"actual": "Dog", "answers": {"is_animal": None}}] * 50)
    est = estimate_attribute_noise(load_feedback(p), protos)
    assert est.observations["is_animal"] == 0, "MAYBE was counted as an answer"


def test_unknown_objects_are_skipped_for_rate_estimation(tmp_path, protos):
    p = tmp_path / "log.csv"
    write_log(p, [{"actual": "Hovercraft", "answers": {"is_animal": 1}}])
    est = estimate_attribute_noise(load_feedback(p), protos)
    assert est.rows_used == 0 and est.rows_skipped == 1


def test_every_rate_is_inside_the_engine_accepted_range(tmp_path, protos):
    p = tmp_path / "log.csv"
    rows = [{"actual": "Dog", "answers": {f: 1 for f in protos.features[:20]}}
            for _ in range(60)]
    write_log(p, rows)
    est = estimate_attribute_noise(load_feedback(p), protos)
    v = est.as_vector(protos.features)
    assert (v > 0).all() and (v < 0.5).all()
    assert v.min() >= MIN_RATE and v.max() <= MAX_RATE
    # and the engine must actually accept it
    BeliefEngine(protos, noise=v)


def test_vector_length_matches_the_attribute_count(tmp_path, protos):
    est = estimate_attribute_noise([], protos)
    assert est.as_vector(protos.features).shape == (protos.n_features,)


# ---- new object candidates ---------------------------------------------

def test_collects_unknown_objects_with_their_answers(tmp_path, protos):
    p = tmp_path / "log.csv"
    write_log(p, [
        {"actual": "Camel", "answers": {"is_animal": 1, "is_big": 1}},
        {"actual": "Camel", "answers": {"is_animal": 1, "is_big": 0}},
        {"actual": "Dog", "answers": {"is_animal": 1}},
    ])
    cands = new_object_candidates(load_feedback(p), protos)
    assert [c.name for c in cands] == ["Camel"], "known objects leaked in"
    c = cands[0]
    assert c.reports == 2
    assert "is_animal" in c.draft_attributes()
    assert "is_big" in c.disputed(), "disagreement should be flagged"


def test_candidate_drafts_exclude_unknown_attributes(tmp_path, protos):
    p = tmp_path / "log.csv"
    write_log(p, [{"actual": "Camel",
                   "answers": {"is_animal": 1, "not_a_real_attr": 1}}])
    c = new_object_candidates(load_feedback(p), protos)[0]
    assert "not_a_real_attr" not in c.votes


# ---- the saved profile --------------------------------------------------

def test_profile_round_trips(tmp_path, protos):
    log = tmp_path / "log.csv"
    write_log(log, [{"actual": "Dog", "answers": {"is_animal": 1}}] * 30)
    est = estimate_attribute_noise(load_feedback(log), protos)
    out = tmp_path / "profile.json"
    write_noise_profile(est, out)

    vec = load_noise_profile(out, protos.features)
    assert vec is not None
    assert vec.shape == (protos.n_features,)
    assert np.allclose(vec, est.as_vector(protos.features), atol=1e-3)


def test_absent_or_broken_profile_returns_none(tmp_path, protos):
    assert load_noise_profile(tmp_path / "nope.json", protos.features) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_noise_profile(bad, protos.features) is None
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert load_noise_profile(empty, protos.features) is None


def test_profile_missing_an_attribute_falls_back_to_the_global_rate(
        tmp_path, protos):
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({
        "global_rate": 0.2,
        "per_attribute": {protos.features[0]: 0.3},
    }), encoding="utf-8")
    vec = load_noise_profile(p, protos.features)
    assert vec[0] == pytest.approx(0.3)
    assert vec[1] == pytest.approx(0.2), "missing attribute did not fall back"


def test_committed_sample_log_is_usable(protos):
    """The sample shipped for demonstration must actually parse and teach."""
    if not cfg.SAMPLE_FEEDBACK_LOG.exists():
        pytest.skip("sample log not generated")
    rows = load_feedback(cfg.SAMPLE_FEEDBACK_LOG)
    assert len(rows) > 50, "sample log is too small to demonstrate anything"
    assert any(r.outcome == "correct" for r in rows), (
        "sample has no winning games, which would bias the estimates")
    assert any(r.outcome == "wrong" for r in rows)
    est = estimate_attribute_noise(rows, protos)
    assert est.rows_used > 0
    vec = est.as_vector(protos.features)
    assert vec.std() > 0.01, "estimates are flat - nothing was learned"
    BeliefEngine(protos, noise=vec)


# ---- the engine accepting per-attribute noise --------------------------

def test_engine_accepts_a_noise_vector(protos):
    vec = np.full(protos.n_features, 0.12)
    eng = BeliefEngine(protos, noise=vec)
    assert eng.mean_noise == pytest.approx(0.12)
    assert eng.noise_vector.shape == (protos.n_features,)


def test_engine_rejects_a_wrongly_sized_vector(protos):
    with pytest.raises(ValueError, match="one value per attribute"):
        BeliefEngine(protos, noise=np.full(5, 0.1))


def test_engine_rejects_out_of_range_entries(protos):
    vec = np.full(protos.n_features, 0.1)
    vec[3] = 0.9
    with pytest.raises(ValueError, match="noise must be in"):
        BeliefEngine(protos, noise=vec)


def test_per_attribute_noise_changes_question_choice(protos):
    """An attribute known to be answered unreliably should be asked less eagerly."""
    plain = BeliefEngine(protos)
    first = plain.best_question()

    vec = np.full(protos.n_features, 0.05)
    vec[protos.features.index(first)] = 0.45   # declare it near-useless
    tuned = BeliefEngine(protos, noise=vec)
    assert tuned.best_question() != first, (
        "a near-useless question was still chosen first")
