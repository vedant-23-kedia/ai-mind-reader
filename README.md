# 🧠 AI Mind Reader

An Akinator-style guessing game. You think of an object; the app asks yes/no
questions and works out what it is. Built on a hand-authored dataset of 117
objects described by 141 binary attributes.

> **Project status: Phases 1–2 of 6 complete.** The data layer is reproducible,
> validated and separable, and the app runs from a clean checkout. The models
> and the inference engine are still the original versions — the models are
> stale by design and are retrained in Phase 3. See [Roadmap](#roadmap).

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

python scripts/build_dataset.py     # compile the dataset from its spec
streamlit run app/app_final.py
```

Verify an install:

```bash
python -m pytest -q                 # 20 dataset and app tests
python scripts/smoke_check.py       # model/dataset alignment + full-game replay
```

Requires Python 3.12. No paths are hardcoded; everything resolves from the
repository root via `src/mindreader/config.py`.

---

## How the guessing works

Three mechanisms, in order of how much they actually contribute:

1. **Candidate filtering** — after each answer, every object whose attribute
   contradicts it is dropped. This is what narrows the set down to one.
2. **Object scoring** — surviving objects are ranked by
   `(matches − 0.5 × contradictions) / answered`, and the top one becomes the
   guess. Its score is displayed as the confidence percentage.
3. **Question selection** — a tiered, partly shuffled order while many
   candidates remain, switching to a greedy information-gain pick at 10 or
   fewer.

**The trained models do not identify the object.** All three predict one of 9
*categories*, and are consulted only in a fallback branch that normal play
never reaches. Making the ML central to the guess is Phase 3–4 work.

With 117 objects, an optimal questioner needs about **log₂(117) ≈ 6.9
questions**.

---

## Repository layout

```
app/app_final.py                Streamlit UI and the current game engine
src/mindreader/config.py        all paths and schema constants
data/source/                    the authoring spec - edit this, not the CSV
  features.json                 attribute -> question text + group
  objects.json                  object -> category + attributes that are TRUE
  implications.json             semantic rules, e.g. is_mammal -> is_animal
scripts/build_dataset.py        compiles the spec into the dataset
scripts/smoke_check.py          end-to-end verification
tests/                          20 tests
data/raw/                       immutable original source, never written to
models/*.pkl                    trained models (stale until Phase 3)
```

## The data pipeline

The dataset is **compiled, never hand-edited**:

```
data/source/{features,objects,implications}.json
        -> scripts/build_dataset.py
        -> data/dataset_clean.csv, category_map.csv,
           feature_columns.json, question_map.json, question_order.json
```

Objects are authored as a list of the attributes that are *true* for them,
rather than as a row of 141 ones and zeros. A 117 × 141 matrix is unreadable
by hand, and that unreadability is how the original data errors survived.

The build **refuses to emit** a dataset where any attribute is constant, any
two objects are identical, any implication rule is broken, or any two objects
are separated by fewer than three answers.

That last gate is the important one. If two objects differ by a single
answer, one mistake by the player makes them impossible to tell apart — and
because filtering is irreversible, the true object is gone for good.

---

## Results so far

| Measure | Original | After Phase 1 | After Phase 2 |
|---|---|---|---|
| Objects | 56 | 56 | **117** |
| Attributes | 52 | 52 | **141** |
| Minimum answers between two objects | 1 | 1 | **3** |
| Pairs separated by a single answer | 12 | 9 | **0** |
| Pairs separated by two answers | 28 | 26 | **0** |
| Median distance between objects | 9 | 9 | **13** |
| Attributes true for exactly one object | 13 (25%) | 12 (23%) | 27 (19%) |
| Semantic rules enforced | 0 | 0 | **65** |
| Tests | 0 | 17 | **20** |

### Phase 1 — bug fixes

| Bug | Effect | Fix |
|---|---|---|
| Columns `ia_adult_only` / `is_vechicle` misspelled, while the models expected `is_adult_only` / `is_vehicle` | Both loaded as **all-zero**, so answering YES to "Is it a vehicle?" eliminated all 56 objects | Renamed in the pipeline; the loader now raises instead of fabricating zero-filled columns |
| 23 factually wrong attribute values | e.g. `Sun` flagged as a vehicle, `produces_fire` set on `Plates` instead of `Matchbox`, `fan` flagged `is_for_reading` | Corrected through an audited log |
| 4 wrong categories | `Sun`→Vehicle, `Condom`→Vehicle, `Aeroplane`→Daily Object, `Batman`→Daily Object | Corrected |
| `.values[0]` indexed before the emptiness guard in `get_best_next_question` | Latent `IndexError`; also rescanned the dataframe per candidate per question | Reordered and replaced with an indexed lookup |
| `base = r"D:\ai-mind-reader"` hardcoded | Ran only from one directory on one machine | Paths resolve from the repo root |
| `dataset_clean.csv` was a byte-identical copy of `dataset.csv` | No cleaning actually happened | Now a validated build artifact |
| Dead `extra_questions` block | Referenced features that did not exist | Removed |
| Object count hardcoded as `"56"` in three places | Would silently lie once the dataset grew | Derived from the data |
| `readme.md` existed as an empty **directory** | No project documentation | This file |

### Phase 2 — separability

- **Released all 5 blocked corrections.** Captain America cannot fly, but he
  and Hulk differed *only* in `can_fly`, so correcting it would have made them
  identical. The same applied to `phone`/`Laptop` (`can_make_sound`) and
  `Clouds`/`Stars` (`is_round`). New attributes made all five safe to apply.
- **Added 89 attributes** targeting the clusters the gate flagged: superheroes
  (`has_cape`, `wears_mask`, `is_king`, `shoots_web`), furniture
  (`is_furniture`, `has_flat_surface`), the sky (`is_in_sky`, `orbits_earth`),
  paper (`has_pages`, `published_daily`, `has_cover`).
- **Added 61 objects**, chosen to be mutually distinct and to make existing
  attributes generalise — `worn_on_feet` described only `Shoes`; it now also
  covers `Socks` and `Sandals`.
- **Fixed the app's coupling to the models.** The question set came from
  `dt_model.feature_names_in_`, so every attribute added after the models were
  last trained was silently discarded. The game now asks about the dataset's
  attributes; the models get their own narrower list for the fallback.
- **65 implication rules** now run on every build. They immediately caught a
  bus labelled `has_screen` without `is_electronic`, and an invalid rule of my
  own (`has_engine → has_wheels`, which only held while `car` was the sole
  object with an engine — ships and helicopters break it).

---

## Known limitations

These are real and deliberately not yet addressed:

- **The models are stale.** They know 52 of the 141 attributes and predict a
  *category* rather than an object. They are fallback-only, so the game works,
  but they contribute nothing. Phase 3 retrains them on object identity.
- **The reported accuracies are not meaningful.** 92.86% is 13 of 14 rows on a
  single test split, measured on category prediction.
- **Filtering is still irreversible.** A minimum distance of 3 means one wrong
  answer is survivable in principle, but the current engine deletes candidates
  outright, so it cannot recover. Phase 4 replaces this with Bayesian belief
  updating.
- **"Maybe" is discarded** rather than treated as uncertainty.
- **27 attributes are still true for exactly one object.** These are kept
  deliberately: in a 117-object set a single-object attribute is *specific*,
  not dead, and they are what makes the separability gate achievable. The real
  risk is that they inflate apparent model accuracy, which Phase 5 addresses by
  evaluating under answer noise rather than by deleting them.
- **`is_reusable` is true for 99 of 117 objects**, so it carries little
  information. It is kept because information-gain selection already
  deprioritises weak questions, making removal low-value.
- **Category balance is still uneven** — `Daily Object` holds 31 of 117.
- The wrong-guess text box says it will learn from the answer, but nothing is
  persisted.

---

## Roadmap

| Phase | Goal |
|---|---|
| **1 — done** | Reproducible, validated, bug-free data layer; app runs anywhere |
| **2 — done** | Minimum separation of 3 answers; 117 objects, 141 attributes, 65 semantic rules |
| 3 | Train real *object-level* models on noise-augmented data; `BernoulliNB` instead of `GaussianNB` for binary features; add baselines |
| 4 | Replace irreversible filtering with Bayesian belief updating + expected information gain; calibrated confidence, recoverable wrong answers, working "maybe", undo |
| 5 | Simulated-game evaluation across noise levels; before/after comparison and a calibration curve |
| 6 | Persist player feedback into a retraining loop; deploy |

---

**Built by:** Vedant · ML + DL portfolio project
**Stack:** Python · scikit-learn · pandas · Streamlit
