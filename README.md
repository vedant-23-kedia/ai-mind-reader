# 🧠 AI Mind Reader

An Akinator-style guessing game. You think of an object; the app asks yes/no
questions and works out what it is. Built on a hand-authored dataset of 56
objects described by 52 binary attributes.

> **Project status: Phase 1 of 6 complete.** The data layer is now reproducible,
> validated and bug-free, and the app runs from a clean checkout. The inference
> engine and the models are still the original versions and are being rebuilt in
> Phases 2–4. See [Roadmap](#roadmap) for what is deliberately not fixed yet.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

python scripts/build_dataset.py     # regenerate the clean dataset
streamlit run app/app_final.py
```

Verify an install:

```bash
python -m pytest -q                 # dataset invariants
python scripts/smoke_check.py       # model/dataset alignment + full-game replay
```

Requires Python 3.12. No paths are hardcoded; everything resolves from the
repository root via `src/mindreader/config.py`.

---

## How the guessing works

Three mechanisms, in order of how much they actually contribute:

1. **Candidate filtering** — after each answer, every object whose attribute
   contradicts it is dropped. This is what narrows 56 objects down to one.
2. **Object scoring** — surviving objects are ranked by
   `(matches − 0.5 × contradictions) / answered`, and the top one becomes the
   guess. Its score is displayed as the confidence percentage.
3. **Question selection** — a tiered, partly shuffled order while many
   candidates remain, switching to a greedy information-gain pick (the question
   splitting the remaining candidates closest to 50/50) at 10 or fewer.

**The trained models do not identify the object.** All three predict one of 9
*categories*, and are only consulted in a fallback branch that normal play never
reaches. Making the ML central to the guess is Phase 3–4 work.

---

## Repository layout

```
app/app_final.py            Streamlit UI and the current game engine
src/mindreader/config.py    all paths and schema constants
scripts/build_dataset.py    raw + corrections -> validated clean dataset
scripts/smoke_check.py      end-to-end verification
tests/test_dataset.py       dataset invariants
data/raw/                   immutable hand-authored source, never written to
data/*.csv, *.json          generated artifacts
models/*.pkl                trained scikit-learn models
notebooks/                  (empty - training notebooks to be reconstructed)
```

## The data pipeline

The dataset is generated, not hand-edited:

```
data/raw/dataset_original.csv   (immutable source)
        + data/corrections.csv            (23 audited fixes, each with a reason)
        + data/category_corrections.csv   (4 category fixes)
        -> data/dataset_clean.csv         (validated output)
```

Every correction records `object, feature, from, to, reason`. The build fails
loudly if a correction's `from` value no longer matches the raw data, so edits
to the source can never silently invalidate the log.
`data/deferred_corrections.csv` holds 5 known-wrong values that cannot be fixed
yet — see [Known limitations](#known-limitations).

`build_dataset.py` refuses to emit a dataset unless all values are binary, all
feature vectors are unique, every object has a category, and every feature has
question text.

---

## What Phase 1 fixed

| Bug | Effect | Fix |
|---|---|---|
| Columns `ia_adult_only` / `is_vechicle` misspelled, while the models expected `is_adult_only` / `is_vehicle` | Both loaded as **all-zero**, so answering YES to "Is it a vehicle?" eliminated all 56 objects | Renamed in the pipeline; the loader now raises instead of fabricating zero-filled columns |
| 23 factually wrong attribute values | e.g. `Sun` flagged as a vehicle, `produces_fire` set on `Plates` instead of `Matchbox`, `fan` flagged `is_for_reading` | Corrected through the audited log |
| 4 wrong categories | `Sun`→Vehicle, `Condom`→Vehicle, `Aeroplane`→Daily Object, `Batman`→Daily Object | Corrected |
| `.values[0]` indexed before the emptiness guard in `get_best_next_question` | Latent `IndexError`; also rescanned the dataframe once per candidate per question | Reordered and replaced with an indexed lookup |
| `base = r"D:\ai-mind-reader"` hardcoded | Ran only from one directory on one machine | Paths resolve from the repo root |
| `dataset_clean.csv` was a byte-identical copy of `dataset.csv` | No cleaning actually happened | Now a validated build artifact |
| Dead `extra_questions` block referencing non-existent features | Never executed | Removed |
| Object count hardcoded as `"56"` in three places | Would silently lie once the dataset grows | Derived from the data |
| `readme.md` existed as an empty **directory** | No project documentation | This file |

Fragile object pairs (separated by a single answer) dropped from **12 to 9**,
and pairs separated by two answers from 28 to 26.

---

## Known limitations

These are real and deliberately not yet addressed:

- **9 object pairs still differ by exactly one answer** (e.g. `Chair`/`Table`
  only in `has_seat`, `phone`/`Laptop` only in `can_make_sound`). One wrong
  answer makes them unrecoverable, because filtering is irreversible.
- **5 corrections are blocked** — fixing them with the current attributes would
  make two objects *identical*. `Captian_America` cannot fly, but he and `Hulk`
  differ only in `can_fly`; correcting it collapses them. These need new
  distinguishing attributes first.
- **13 attributes are true for exactly one object** (`has_seat`, `is_hat`,
  `worn_on_ears`, …) — giveaway flags that inflate apparent accuracy, while
  `is_reusable` is true for 49 of 56 objects and carries almost no information.
- **The reported model accuracies are not meaningful.** 92.86% is 13 of 14 rows
  on a single test split of a 56-row dataset, measured on category prediction
  rather than on identifying the object.
- **"Maybe" is discarded** rather than treated as uncertainty.
- The wrong-guess text box says it will learn from the answer, but nothing is
  persisted.

---

## Roadmap

| Phase | Goal |
|---|---|
| **1 — done** | Reproducible, validated, bug-free data layer; app runs anywhere |
| 2 | Raise the data to minimum Hamming distance ≥ 3; add discriminating attributes, drop dead ones, grow the object set |
| 3 | Train real *object-level* models on noise-augmented data; `BernoulliNB` instead of `GaussianNB` for binary features; add baselines |
| 4 | Replace irreversible filtering with Bayesian belief updating + expected information gain; calibrated confidence, recoverable wrong answers, working "maybe", undo |
| 5 | Simulated-game evaluation across noise levels; before/after comparison and a calibration curve |
| 6 | Persist player feedback into a retraining loop; deploy |

---

**Built by:** Vedant · ML + DL portfolio project
**Stack:** Python · scikit-learn · pandas · Streamlit
