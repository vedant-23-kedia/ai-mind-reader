# ============================================================
# AI MIND READER — FINAL VERSION with shuffled questions
# Save as: D:\ai-mind-reader\app\app_final.py
# Run: streamlit run app_final.py
# ============================================================

import json
import random
import sys
from pathlib import Path

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore

# Make the project's own package importable no matter where streamlit is
# launched from, so no absolute drive path is ever hardcoded.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mindreader import config as cfg  # noqa: E402

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_all():
    if not cfg.DATASET.exists():
        raise RuntimeError(
            "%s not found. Build it first with:\n"
            "    python scripts/build_dataset.py" % cfg.DATASET
        )

    df = pd.read_csv(cfg.DATASET)

    for col in ["feature_sum", "category"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    dt_model   = joblib.load(cfg.MODELS_DIR / "decision_tree.pkl")
    nb_model   = joblib.load(cfg.MODELS_DIR / "naive_bayes.pkl")
    mlp_model  = joblib.load(cfg.MODELS_DIR / "mlp_neural_network.pkl")
    best_path  = cfg.MODELS_DIR / "best_model.pkl"
    best_model = joblib.load(best_path) if best_path.exists() else dt_model

    category_map = pd.read_csv(cfg.CATEGORY_MAP)
    cat_map_dict = dict(zip(category_map["object"], category_map["category"]))

    # The dataset must already carry every feature the models were trained on.
    # The original code silently created missing columns filled with zeros,
    # which is how the misspelled headers turned two questions into answers
    # that eliminated every object. Failing loudly is the fix.
    expected_features = list(dt_model.feature_names_in_)
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise RuntimeError(
            "dataset is missing features the models were trained on: %s\n"
            "Re-run: python scripts/build_dataset.py" % missing
        )
    df = df[["object"] + expected_features]

    question_order = json.loads(cfg.QUESTION_ORDER.read_text(encoding="utf-8"))
    question_map   = json.loads(cfg.QUESTION_MAP.read_text(encoding="utf-8"))

    unmapped = [f for f in expected_features if f not in question_map]
    if unmapped:
        raise RuntimeError("features with no question text: %s" % unmapped)

    return (df, dt_model, nb_model, mlp_model, best_model,
            cat_map_dict, question_order, question_map, expected_features)

(df, dt_model, nb_model, mlp_model, best_model,
 cat_map_dict, question_order, question_map, feature_columns) = load_all()

# Built once. The original code called df[df["object"] == obj] inside nested
# loops, rescanning every row for every candidate on every question.
feature_lookup = df.set_index("object")

N_OBJECTS = len(df)

# ============================================================
# SMART QUESTION SHUFFLER
# Splits questions into tiers — broad questions first,
# specific questions shuffled within each tier
# This ensures different game experience each time
# ============================================================

def get_shuffled_question_order(question_order, feature_columns):
    # Tier 1 — ALWAYS ask these first (broad category questions)
    tier1 = [
        "is_living", "is_animal", "is_character", "made_by_humans",
        "is_fictional", "is_natural"
    ]

    # Tier 2 — Ask these second (mid-level questions)
    tier2 = [
        "is_electronic", "is_wearable", "is_vehicle", "is_food",
        "is_edible", "is_marvel", "is_cartoons", "is_musical",
        "can_fly", "can_swim", "is_big", "used_daily"
    ]

    # Tier 3 — Ask these last (specific distinguishing questions)
    # Everything not in tier1 or tier2
    tier3 = [q for q in question_order
             if q not in tier1 and q not in tier2]

    # Filter to only include features that exist in model
    t1 = [q for q in tier1 if q in feature_columns]
    t2 = [q for q in tier2 if q in feature_columns]
    t3 = [q for q in tier3 if q in feature_columns]

    # Shuffle tier2 and tier3 so each game is different
    random.shuffle(t2)
    random.shuffle(t3)

    # Combine: fixed broad → shuffled mid → shuffled specific
    final_order = t1 + t2 + t3

    # Add any remaining features not covered
    for q in question_order:
        if q not in final_order and q in feature_columns:
            final_order.append(q)

    return final_order

# ============================================================
# GAME FUNCTIONS
# ============================================================

def filter_candidates(answers_so_far):
    candidates = df.copy()
    for feature, value in answers_so_far.items():
        if feature not in candidates.columns:
            continue
        if value == 1:
            candidates = candidates[candidates[feature] == 1]
        elif value == 0:
            candidates = candidates[candidates[feature] == 0]
    return candidates["object"].tolist()

def get_object_scores(answers_so_far, remaining_candidates):
    # Score based on how many answers MATCH each object
    # Also penalize objects that CONTRADICT answers
    scores = {}
    for _, row in df.iterrows():
        obj = row["object"]
        if obj not in remaining_candidates:
            continue
        match   = 0
        contra  = 0
        total   = 0
        for feat, answer in answers_so_far.items():
            if feat not in df.columns or answer == 0.5:
                continue
            total += 1
            if row[feat] == answer:
                match += 1
            else:
                contra += 1

        if total > 0:
            # Match score minus penalty for contradictions
            score = ((match - (contra * 0.5)) / total) * 100
            scores[obj] = round(max(score, 0), 2)
        else:
            scores[obj] = 50.0

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def get_best_next_question(remaining_candidates, asked_features, all_question_order):
    # Find the question that splits remaining candidates most evenly
    # This is Information Gain — best question = closest to 50/50 split
    if len(remaining_candidates) <= 1:
        return None

    best_feat  = None
    best_split = float("inf")

    for feat in all_question_order:
        if feat not in feature_columns:
            continue
        if feat in asked_features:
            continue
        if feat not in df.columns:
            continue

        # The original version indexed .values[0] BEFORE checking that the
        # object existed, so an unknown candidate raised IndexError, and it
        # re-scanned the whole dataframe once per candidate per question.
        known = [o for o in remaining_candidates if o in feature_lookup.index]
        if not known:
            continue
        yes_count = int(feature_lookup.loc[known, feat].sum())
        no_count  = len(known) - yes_count

        # Best question = closest to 50/50 split
        split_diff = abs(yes_count - no_count)
        if split_diff < best_split:
            best_split = split_diff
            best_feat  = feat

    return best_feat

def make_final_guess(answers_so_far, remaining_candidates):
    if len(remaining_candidates) == 1:
        scores = get_object_scores(answers_so_far, remaining_candidates)
        conf   = scores[0][1] if scores else 90.0
        return remaining_candidates[0], conf

    # Use all objects if no candidates remain
    search_pool = remaining_candidates if remaining_candidates else df["object"].tolist()

    scores = get_object_scores(answers_so_far, search_pool)
    if scores and scores[0][1] > 0:
        return scores[0][0], min(scores[0][1], 99.0)

    # Fallback to ensemble
    input_data = {feat: answers_so_far.get(feat, 0.5) for feat in feature_columns}
    input_df   = pd.DataFrame([input_data])[feature_columns]

    dt_probs  = dt_model.predict_proba(input_df)[0]
    nb_probs  = nb_model.predict_proba(input_df)[0]
    mlp_probs = mlp_model.predict_proba(input_df)[0]
    ensemble  = (dt_probs + nb_probs + mlp_probs) / 3

    class_names = best_model.classes_
    top_cat     = class_names[np.argmax(ensemble)]
    top_conf    = round(float(np.max(ensemble)) * 100, 2)

    cat_cands   = [o for o in search_pool if cat_map_dict.get(o, "") == top_cat]
    final_obj   = cat_cands[0] if cat_cands else search_pool[0]
    return final_obj, top_conf

# ============================================================
# SESSION STATE
# ============================================================

def reset_game():
    wins  = st.session_state.get("wins", 0)
    games = st.session_state.get("games", 0)
    st.session_state.clear()
    st.session_state.wins            = wins
    st.session_state.games           = games
    st.session_state.started         = False
    st.session_state.over            = False
    st.session_state.answers         = {}
    st.session_state.asked           = 0
    st.session_state.candidates      = df["object"].tolist()
    st.session_state.confidence      = 0.0
    st.session_state.guess           = ""
    st.session_state.correct         = None
    # Generate NEW shuffled question order for each game
    st.session_state.game_q_order    = get_shuffled_question_order(
        question_order, feature_columns
    )
    st.session_state.q_pointer       = 0

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
c2.metric("🤖 Models", "3")
c3.metric("🎮 Games", st.session_state.get("games", 0))
g = st.session_state.get("games", 0)
w = st.session_state.get("wins", 0)
c4.metric("🏆 AI Wins", str(round(w/g*100)) + "%" if g > 0 else "—")

st.markdown("---")

# ============================================================
# SCREEN 1 — HOME
# ============================================================

if not st.session_state.started and not st.session_state.over:

    st.markdown("### 🎮 How to Play")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**1.** 🤔 Think of any object")
        st.markdown("**2.** 🚀 Click Start Game")
    with col_r:
        st.markdown("**3.** ✅ Answer Yes / No / Maybe")
        st.markdown("**4.** 🎯 AI will guess it!")

    st.markdown("")

    with st.expander(f"📋 See all {N_OBJECTS} objects you can think of"):
        grouped = {}
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
# SCREEN 2 — GAME IN PROGRESS
# ============================================================

elif st.session_state.started and not st.session_state.over:

    asked      = st.session_state.asked
    max_q      = 20
    candidates = st.session_state.candidates
    game_order = st.session_state.game_q_order

    # Progress
    st.progress(asked / max_q)
    pa, pb = st.columns(2)
    pa.markdown(f"**🔢 Question {asked + 1} / {max_q}**")
    pb.markdown(f"**🎯 Candidates: {len(candidates)}**")

    # Confidence
    if asked >= 3 and st.session_state.confidence > 0:
        conf_val = st.session_state.confidence
        emoji    = "🟢" if conf_val >= 80 else ("🟡" if conf_val >= 50 else "🔴")
        st.markdown(f"**{emoji} AI Confidence: {round(conf_val, 1)}%**")
        st.progress(min(conf_val / 100, 1.0))

    st.markdown("")

    # Guess condition
    should_guess = (
        (len(candidates) == 1 and asked >= 3) or
        asked >= max_q or
        (st.session_state.confidence >= 96.0 and asked >= 6)
    )

    if should_guess:
        final_obj, final_conf         = make_final_guess(st.session_state.answers, candidates)
        st.session_state.guess        = final_obj
        st.session_state.confidence   = final_conf
        st.session_state.over         = True
        st.session_state.games       += 1
        st.rerun()

    else:
        # SMART QUESTION SELECTION
        # When candidates > 10: use shuffled pre-generated order
        # When candidates <= 10: use information gain to pick best next question
        asked_feats = set(st.session_state.answers.keys())

        if len(candidates) <= 10 and len(candidates) > 1:
            # Use information gain for smart selection
            current_feat = get_best_next_question(candidates, asked_feats, game_order)
            if current_feat is None:
                # Fallback to next in order
                ptr = st.session_state.q_pointer
                while ptr < len(game_order) and game_order[ptr] in asked_feats:
                    ptr += 1
                st.session_state.q_pointer = ptr
                current_feat = game_order[ptr] if ptr < len(game_order) else None
        else:
            # Use shuffled order
            ptr = st.session_state.q_pointer
            while ptr < len(game_order) and game_order[ptr] in asked_feats:
                ptr += 1
            st.session_state.q_pointer = ptr
            current_feat = game_order[ptr] if ptr < len(game_order) else None

        if current_feat and current_feat in question_map:
            question_txt = question_map[current_feat]

            st.markdown(
                f'<div class="question-card">❓ {question_txt}</div>',
                unsafe_allow_html=True
            )

            # Hint when narrowed down
            if 1 < len(candidates) <= 5:
                st.markdown(
                    f'<div class="hint-box">🔍 Narrowed to: <b>{", ".join(candidates)}</b></div>',
                    unsafe_allow_html=True
                )

            st.markdown("")

            b1, b2, b3 = st.columns(3)
            answer_given = None

            with b1:
                if st.button("✅  YES", use_container_width=True,
                             type="primary", key=f"yes_q{asked}"):
                    answer_given = 1
            with b2:
                if st.button("❌  NO", use_container_width=True,
                             key=f"no_q{asked}"):
                    answer_given = 0
            with b3:
                if st.button("🤔  MAYBE", use_container_width=True,
                             key=f"maybe_q{asked}"):
                    answer_given = 0.5

            if answer_given is not None:
                st.session_state.answers[current_feat] = answer_given
                st.session_state.asked                += 1
                st.session_state.q_pointer            += 1
                st.session_state.candidates            = filter_candidates(
                    st.session_state.answers
                )
                remaining = st.session_state.candidates
                if st.session_state.asked >= 3:
                    pool   = remaining if remaining else df["object"].tolist()
                    scores = get_object_scores(st.session_state.answers, pool)
                    st.session_state.confidence = scores[0][1] if scores else 50.0
                st.rerun()

        st.markdown("")
        if st.button("🏳️  Give Up & See AI Guess", key=f"giveup_q{asked}"):
            final_obj, final_conf         = make_final_guess(
                st.session_state.answers, candidates
            )
            st.session_state.guess        = final_obj
            st.session_state.confidence   = final_conf
            st.session_state.over         = True
            st.session_state.games       += 1
            st.rerun()

# ============================================================
# SCREEN 3 — RESULT
# ============================================================

elif st.session_state.over:

    final_obj = st.session_state.guess
    conf      = round(st.session_state.confidence, 1)
    asked     = st.session_state.asked
    category  = cat_map_dict.get(final_obj, "Unknown")

    st.markdown(f"### 🎯 After **{asked}** questions...")
    st.markdown("")

    st.markdown(f"""
    <div class="guess-card">
        <div class="guess-label">🔮 My Answer is...</div>
        <div class="guess-object">{final_obj.upper()}</div>
        <div class="guess-meta">
            📂 Category: {category} &nbsp;&nbsp;|&nbsp;&nbsp;
            📊 Confidence: {conf}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### Was I correct?")

    r1, r2 = st.columns(2)
    with r1:
        if st.button("✅  Yes! You got it!", type="primary",
                     use_container_width=True, key="correct_btn"):
            st.session_state.correct  = True
            st.session_state.wins    += 1
            st.rerun()
    with r2:
        if st.button("❌  Nope, Wrong!", use_container_width=True,
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
            placeholder="Type it here so I can learn..."
        )
        if correct_input:
            st.success("Thanks! You were thinking of **" + correct_input + "**! 🧠")

    if st.session_state.correct is not None:
        st.markdown("")
        st.markdown("### 📊 Game Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("❓ Questions", asked)
        m2.metric("📊 Confidence", str(conf) + "%")
        m3.metric("🎮 Total Games", st.session_state.games)
        g = st.session_state.games
        w = st.session_state.wins
        m4.metric("🏆 AI Win Rate",
                  str(round(w/g*100)) + "%" if g > 0 else "0%")

        with st.expander("📝 Review all your answers"):
            for feat, val in st.session_state.answers.items():
                q   = question_map.get(feat, feat)
                ans = "Yes ✅" if val == 1 else ("No ❌" if val == 0 else "Maybe 🤔")
                st.markdown(f"**{q}** → {ans}")

        st.markdown("")
        if st.button("🔄  Play Again!", type="primary",
                     use_container_width=True, key="play_again_btn"):
            reset_game()
            st.rerun()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 About This Game")
    st.markdown(f"""
Think of any object from the list and
the AI will guess it by asking smart
yes/no questions!

---
**🤖 ML Models** *(category accuracy,
pending retraining in Phase 3)*:
- 🌳 Decision Tree — 85.7%
- 📊 Naive Bayes — 92.9%
- 🧠 MLP Neural Net — 92.9%

**⚙️ Method:**
Ensemble + Object Scoring + Info Gain

**📦 Dataset:**
{N_OBJECTS} objects, {len(feature_columns)} features

**🛠️ Stack:**
Python • scikit-learn • Streamlit

---
    """)
    st.markdown("**👨‍💻 Built by:** Vedant")
    st.markdown("**🎓 Type:** ML + DL Portfolio Project")