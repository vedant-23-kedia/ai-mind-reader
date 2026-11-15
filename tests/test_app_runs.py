"""The Streamlit app must execute end to end without raising.

Uses Streamlit's AppTest harness, which runs the real script headlessly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app_final.py"

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def test_home_screen_renders():
    at = run_app()
    assert not at.exception, "app raised: %s" % [str(e) for e in at.exception]


def test_object_count_is_not_hardcoded():
    """The metric must come from the data, not the literal 56."""
    at = run_app()
    labels = {m.label: m.value for m in at.metric}
    assert any("Objects" in k for k in labels), labels
    import csv
    with (ROOT / "data" / "dataset_clean.csv").open(newline="", encoding="utf-8") as fh:
        n = sum(1 for _ in csv.reader(fh)) - 1
    shown = next(v for k, v in labels.items() if "Objects" in k)
    assert str(shown) == str(n), "shows %r but dataset has %d objects" % (shown, n)


def test_start_game_then_answer_first_question():
    """Starting a game and answering must not raise and must consume a question."""
    at = run_app()
    at.button(key="start_btn").click().run()
    assert not at.exception, "starting raised: %s" % [str(e) for e in at.exception]

    # A question card and the three answer buttons should be present.
    assert at.button(key="yes_q0"), "no YES button on the first question"

    at.button(key="yes_q0").click().run()
    assert not at.exception, "answering raised: %s" % [str(e) for e in at.exception]
    assert at.session_state["asked"] == 1
    assert len(at.session_state["answers"]) == 1


def test_answering_every_question_never_raises():
    """Play a whole game by always answering NO; must reach a guess cleanly."""
    at = run_app()
    at.button(key="start_btn").click().run()
    for i in range(25):
        if at.session_state["over"]:
            break
        key = "no_q%d" % at.session_state["asked"]
        buttons = [b for b in at.button if b.key == key]
        if not buttons:
            break
        buttons[0].click().run()
        assert not at.exception, (
            "raised at question %d: %s" % (i, [str(e) for e in at.exception])
        )
    assert at.session_state["over"], "game never reached a guess"
    assert at.session_state["guess"], "game ended without naming a guess"


def test_maybe_button_records_an_unknown_answer():
    """MAYBE must be recorded as a skipped question, not as a 'no'."""
    at = run_app()
    at.button(key="start_btn").click().run()
    at.button(key="maybe_q0").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    answers = at.session_state["answers"]
    assert len(answers) == 1
    assert answers[0][1] is None, "MAYBE should store None, got %r" % (answers[0][1],)


def test_undo_removes_the_last_answer():
    at = run_app()
    at.button(key="start_btn").click().run()
    at.button(key="yes_q0").click().run()
    assert at.session_state["asked"] == 1
    at.button(key="undo_q1").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["asked"] == 0
    assert at.session_state["answers"] == []


def test_undo_is_disabled_before_any_answer():
    at = run_app()
    at.button(key="start_btn").click().run()
    undo = [b for b in at.button if b.key == "undo_q0"]
    assert undo, "undo button missing"
    assert undo[0].disabled, "undo should be disabled with no answers yet"


def test_game_reaches_a_guess_and_reports_a_probability():
    """Truthful-ish play should converge and the confidence must be a probability."""
    at = run_app()
    at.button(key="start_btn").click().run()
    for _ in range(25):
        if at.session_state["over"]:
            break
        key = "yes_q%d" % at.session_state["asked"]
        btns = [b for b in at.button if b.key == key]
        if not btns:
            break
        btns[0].click().run()
        assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["over"], "game never reached a guess"
    conf = at.session_state["confidence"]
    assert 0.0 <= conf <= 1.0, "confidence %r is not a probability" % conf
    assert at.session_state["guess"]
    assert len(at.session_state["top3"]) == 3
