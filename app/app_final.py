# ============================================================
# AI MIND READER - Bayesian belief engine (Phase 4)
# Run: streamlit run app/app_final.py
# ============================================================

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st  # type: ignore

# Make the project's own package importable no matter where streamlit is
# launched from, so no absolute drive path is ever hardcoded.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402
from mindreader.dataset import load_prototypes  # noqa: E402
from mindreader.engine import (  # noqa: E402
    MAYBE,
    NO,
    YES,
    BeliefEngine,
)

MAX_QUESTIONS = 20

st.set_page_config(
    page_title="AI Mind Reader",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C6FF, #7C4DFF, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #aaa;
        margin-bottom: 1rem;
    }
    .question-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #00C6FF;
        border-left: 6px solid #7C4DFF;
        padding: 1.5rem 1.8rem;
        border-radius: 15px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 198, 255, 0.15);
    }
    .guess-card {
        background: linear-gradient(135deg, #0F3460 0%, #533483 50%, #E94560 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .guess-label {
        font-size: 0.95rem;
        opacity: 0.8;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .guess-object {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 3px;
    }
    .guess-meta {
        font-size: 0.9rem;
        opacity: 0.75;
        margin-top: 0.8rem;
    }
    .correct-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.8rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .wrong-card {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 1.8rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .hint-box {
        background: rgba(0, 198, 255, 0.1);
        border: 1px solid rgba(0, 198, 255, 0.3);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #00C6FF;
        margin: 0.5rem 0;
    }
    .belief-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #ddd;
        padding: 0.15rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_all():
    protos = load_prototypes()
    question_map = json.loads(cfg.QUESTION_MAP.read_text(encoding="utf-8"))

    unmapped = [f for f in protos.features if f not in question_map]
    if unmapped:
        raise RuntimeError("attributes with no question text: %s" % unmapped)

    cats: dict[str, str] = dict(zip(protos.names, protos.categories))

    # Phase 3 benchmark figures, shown in the sidebar. Rendered from the file
    # the training script writes, so they cannot go stale the way the
    # hardcoded 85.7% / 92.9% figures did.
    info_path = cfg.MODELS_DIR / "object_model_info.json"
    model_info = (json.loads(info_path.read_text(encoding="utf-8"))
                  if info_path.exists() else None)
    return protos, question_map, cats, model_info


protos, question_map, cat_map_dict, model_info = load_all()

N_OBJECTS = protos.n_objects
N_FEATURES = protos.n_features


def build_engine() -> BeliefEngine:
    """Rebuild the belief state from the answers recorded this game.

    The posterior is a pure function of the answer list, so there is no
    hidden state to keep in sync - which is exactly what makes undo possible.
    """
    eng = BeliefEngine(protos)
    for feat, val in st.session_state.answers:
        eng.answer(feat, val)
    return eng


def log_feedback(guess: str, actual: str, answers) -> bool:
    """Record a wrong guess so it can be learned from later.

    The original app asked what the player was thinking of and then threw the
    answer away, while telling them it would learn from it.

    The row keeps the player's full answer sheet alongside what they were
    actually thinking of, which is what scripts/learn_from_feedback.py needs
    to estimate how reliably each attribute gets answered.

    Returns True if the row was written.
    """
    try:
        new = not cfg.FEEDBACK_LOG.exists()
        with cfg.FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["timestamp", "guessed", "actual", "n_answers",
                            "answers_json"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                guess, actual.strip(), len(answers),
                json.dumps({f: v for f, v in answers}),
            ])
        return True
    except OSError:
        return False


# ============================================================
# SESSION STATE
# ============================================================

def reset_game():
    wins = st.session_state.get("wins", 0)
    games = st.session_state.get("games", 0)
    st.session_state.clear()
    st.session_state.wins = wins
    st.session_state.games = games
    st.session_state.started = False
    st.session_state.over = False
    st.session_state.answers = []      # list of (feature, YES/NO/MAYBE)
    st.session_state.asked = 0
    st.session_state.guess = ""
    st.session_state.confidence = 0.0
    st.session_state.top3 = []
    st.session_state.correct = None
    st.session_state.logged = False


if "started" not in st.session_state:
    reset_game()

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🧠 AI Mind Reader</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">✨ Think of anything — I will read your mind! ✨</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Objects", str(N_OBJECTS))
c2.metric("❓ Attributes", str(N_FEATURES))
c3.metric("🎮 Games", st.session_state.get("games", 0))
g = st.session_state.get("games", 0)
w = st.session_state.get("wins", 0)
c4.metric("🏆 AI Wins", str(round(w / g * 100)) + "%" if g > 0 else "—")

st.markdown("---")

# ============================================================
# SCREEN 1 - HOME
# ============================================================

if not st.session_state.started and not st.session_state.over:

    st.markdown("### 🎮 How to Play")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**1.** 🤔 Think of any object")
        st.markdown("**2.** 🚀 Click Start Game")
    with col_r:
        st.markdown("**3.** ✅ Answer Yes / No / Maybe")
        st.markdown("**4.** 🎯 The AI will guess it!")

    st.markdown("")
    st.markdown(
        '<div class="hint-box">🛟 Get one wrong? It no longer matters. '
        'The AI keeps a probability for every object instead of crossing them '
        'off, so a mistake can be outvoted by your later answers — and you can '
        'undo mid-game.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    with st.expander(f"📋 See all {N_OBJECTS} objects you can think of"):
        grouped: dict[str, list[str]] = {}
        for obj, cat in cat_map_dict.items():
            grouped.setdefault(cat, []).append(obj)
        for cat, objs in sorted(grouped.items()):
            st.markdown(f"**{cat}:** {', '.join(sorted(objs))}")

    st.markdown("")
    if st.button("🚀  Start Game — Read My Mind!", type="primary",
                 use_container_width=True, key="start_btn"):
        st.session_state.started = True
        st.rerun()

# ============================================================
# SCREEN 2 - GAME IN PROGRESS
# ============================================================

elif st.session_state.started and not st.session_state.over:

    engine = build_engine()
    asked = st.session_state.asked
    top3 = engine.top(3)
    best_obj, best_p = top3[0]

    st.progress(min(asked / MAX_QUESTIONS, 1.0))
    pa, pb = st.columns(2)
    pa.markdown(f"**🔢 Question {asked + 1} / {MAX_QUESTIONS}**")
    pb.markdown(f"**🎲 Uncertainty: {engine.entropy():.1f} bits**")

    if asked >= 2:
        emoji = "🟢" if best_p >= 0.8 else ("🟡" if best_p >= 0.4 else "🔴")
        st.markdown(f"**{emoji} Confidence in top guess: {best_p * 100:.1f}%**")
        st.progress(float(min(best_p, 1.0)))

        with st.expander("🧠 What the AI currently believes"):
            for name, p in engine.top(5):
                st.markdown(
                    f'<div class="belief-row"><span>{name}</span>'
                    f'<span>{p * 100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                "These are posterior probabilities over all "
                f"{N_OBJECTS} objects and always sum to 100%. Nothing is ever "
                "eliminated, so every object keeps a non-zero chance."
            )

    st.markdown("")

    should_guess = engine.is_confident() or asked >= MAX_QUESTIONS

    if should_guess:
        st.session_state.guess = best_obj
        st.session_state.confidence = float(best_p)
        st.session_state.top3 = [(n, float(p)) for n, p in top3]
        st.session_state.over = True
        st.session_state.games += 1
        st.rerun()

    else:
        current_feat = engine.best_question()

        if current_feat is None:
            # Out of questions to ask: guess with what we have.
            st.session_state.guess = best_obj
            st.session_state.confidence = float(best_p)
            st.session_state.top3 = [(n, float(p)) for n, p in top3]
            st.session_state.over = True
            st.session_state.games += 1
            st.rerun()
        else:
            st.markdown(
                f'<div class="question-card">❓ {question_map[current_feat]}</div>',
                unsafe_allow_html=True
            )

            answer_given = "none"
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✅  YES", use_container_width=True,
                             type="primary", key=f"yes_q{asked}"):
                    answer_given = YES
            with b2:
                if st.button("❌  NO", use_container_width=True,
                             key=f"no_q{asked}"):
                    answer_given = NO
            with b3:
                if st.button("🤔  MAYBE", use_container_width=True,
                             key=f"maybe_q{asked}"):
                    answer_given = MAYBE

            if answer_given != "none":
                st.session_state.answers.append((current_feat, answer_given))
                st.session_state.asked += 1
                st.rerun()

            st.caption(
                "“Maybe” genuinely skips the question — the attribute is "
                "left out of the calculation rather than counted as a no."
            )

            st.markdown("")
            u1, u2 = st.columns(2)
            with u1:
                if st.button("↩️  Undo last answer", use_container_width=True,
                             disabled=not st.session_state.answers,
                             key=f"undo_q{asked}"):
                    st.session_state.answers.pop()
                    st.session_state.asked = max(0, st.session_state.asked - 1)
                    st.rerun()
            with u2:
                if st.button("🏳️  Give up & see my guess",
                             use_container_width=True, key=f"giveup_q{asked}"):
                    st.session_state.guess = best_obj
                    st.session_state.confidence = float(best_p)
                    st.session_state.top3 = [(n, float(p)) for n, p in top3]
                    st.session_state.over = True
                    st.session_state.games += 1
                    st.rerun()

# ============================================================
# SCREEN 3 - RESULT
# ============================================================

elif st.session_state.over:

    final_obj = st.session_state.guess
    conf = st.session_state.confidence
    asked = st.session_state.asked
    category = cat_map_dict.get(final_obj, "Unknown")
    top3 = st.session_state.get("top3", [])

    st.markdown(f"### 🎯 After **{asked}** questions...")
    st.markdown("")

    st.markdown(f"""
    <div class="guess-card">
        <div class="guess-label">🔮 My Answer is...</div>
        <div class="guess-object">{final_obj.upper()}</div>
        <div class="guess-meta">
            📂 Category: {category} &nbsp;&nbsp;|&nbsp;&nbsp;
            📊 Probability: {conf * 100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(top3) > 1:
        runners = " · ".join(f"{n} ({p * 100:.1f}%)" for n, p in top3[1:])
        st.markdown(
            f'<div class="hint-box">🥈 If that is wrong, my next guesses are: '
            f'{runners}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("### Was I correct?")

    r1, r2 = st.columns(2)
    with r1:
        if st.button("✅  Yes! You got it!", type="primary",
                     use_container_width=True, key="correct_btn"):
            st.session_state.correct = True
            st.session_state.wins += 1
            st.rerun()
    with r2:
        if st.button("❌  Nope, wrong!", use_container_width=True,
                     key="wrong_btn"):
            st.session_state.correct = False
            st.rerun()

    if st.session_state.correct is True:
        st.markdown(
            '<div class="correct-card">🎉 YESSS! I read your mind!<br>'
            '🤖 The AI wins this round!</div>',
            unsafe_allow_html=True
        )
        st.balloons()

    elif st.session_state.correct is False:
        st.markdown(
            '<div class="wrong-card">😅 Oops! You outsmarted me!<br>'
            '🧠 Human wins this round!</div>',
            unsafe_allow_html=True
        )
        correct_input = st.text_input(
            "What were you thinking of?",
            placeholder="Type it here so I can learn...",
            key="actual_input",
        )
        if correct_input and not st.session_state.get("logged"):
            ok = log_feedback(final_obj, correct_input,
                              st.session_state.answers)
            st.session_state.logged = True
            if ok:
                st.success(
                    f"Saved — you were thinking of **{correct_input}**. "
                    f"Logged to `{cfg.FEEDBACK_LOG.name}` with all "
                    f"{len(st.session_state.answers)} of your answers."
                )
            else:
                st.warning("Could not write the feedback log.")

        if correct_input and correct_input.strip() in protos.names:
            eng = build_engine()
            missed = eng.contradictions(correct_input.strip())
            if missed:
                st.markdown(
                    '<div class="hint-box">🔍 Your answers disagreed with '
                    f'<b>{correct_input.strip()}</b> on: '
                    f'{"; ".join(question_map.get(f, f) for f in missed[:4])}'
                    '</div>',
                    unsafe_allow_html=True,
                )

    if st.session_state.correct is not None:
        st.markdown("")
        st.markdown("### 📊 Game Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("❓ Questions", asked)
        m2.metric("📊 Probability", f"{conf * 100:.1f}%")
        m3.metric("🎮 Total Games", st.session_state.games)
        g = st.session_state.games
        w = st.session_state.wins
        m4.metric("🏆 AI Win Rate",
                  str(round(w / g * 100)) + "%" if g > 0 else "0%")

        with st.expander("📝 Review all your answers"):
            for feat, val in st.session_state.answers:
                q = question_map.get(feat, feat)
                ans = ("Yes ✅" if val == YES
                       else ("No ❌" if val == NO else "Maybe 🤔"))
                st.markdown(f"**{q}** → {ans}")

        st.markdown("")
        if st.button("🔄  Play Again!", type="primary",
                     use_container_width=True, key="play_again_btn"):
            reset_game()
            st.rerun()

# ============================================================
# SIDEBAR
# ============================================================

if model_info:
    _res = sorted(model_info["results"].items(), key=lambda kv: -kv[1]["test_top1"])
    bench_lines = "\n".join(
        "- %s — %.1f%%" % (name.replace("_", " "), r["test_top1"])
        for name, r in _res[:3]
    )
else:
    bench_lines = "- (run scripts/train_models.py)"

with st.sidebar:
    st.markdown("## 🧠 About This Game")
    st.markdown(f"""
Think of any object from the list and
the AI will guess it by asking smart
yes/no questions.

---
**⚙️ How it works**

It keeps a probability for all
{N_OBJECTS} objects and updates them
after each answer (Bayes' rule), then
asks whichever question should tell it
the most (expected information gain).

A wrong answer costs an object some
probability instead of eliminating it,
so mistakes are survivable.

---
**📈 Model benchmarks**
*(top-1 on noisy simulated players)*
{bench_lines}

**📦 Dataset**
{N_OBJECTS} objects, {N_FEATURES} attributes

**🛠️ Stack**
Python • scikit-learn • Streamlit

---
    """)
    st.markdown("**👨‍💻 Built by:** Vedant")
    st.markdown("**🎓 Type:** ML + DL Portfolio Project")
