"""Central path and schema configuration.

Every path is derived from the repository root, which is located relative to
this file. Nothing here depends on the project living at a particular drive
or directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"

# Immutable hand-authored source, never written to by any script.
RAW_DATASET = RAW_DIR / "dataset_original.csv"
RAW_CATEGORY_MAP = RAW_DIR / "category_map_original.csv"

# Auditable edit logs applied by scripts/build_dataset.py.
CORRECTIONS = DATA_DIR / "corrections.csv"
CATEGORY_CORRECTIONS = DATA_DIR / "category_corrections.csv"
DEFERRED_CORRECTIONS = DATA_DIR / "deferred_corrections.csv"

# Generated artifacts - the canonical inputs the app and training read.
DATASET = DATA_DIR / "dataset_clean.csv"
CATEGORY_MAP = DATA_DIR / "category_map.csv"
FEATURE_COLUMNS = DATA_DIR / "feature_columns.json"
QUESTION_MAP = DATA_DIR / "question_map.json"
QUESTION_ORDER = DATA_DIR / "question_order.json"

# Feedback captured from players when the AI guesses wrong.
FEEDBACK_LOG = DATA_DIR / "feedback_log.csv"

# The original dataset shipped two misspelled column headers while the trained
# models expected the corrected names. Loading it without this rename silently
# produced all-zero columns, which made two questions eliminate every object.
LEGACY_COLUMN_RENAMES = {
    "ia_adult_only": "is_adult_only",
    "is_vechicle": "is_vehicle",
}

# Minimum pairwise Hamming distance the dataset should satisfy so that any one
# wrong answer is still recoverable. Phase 1 records the real value; Phase 2
# raises the data to meet it.
MIN_HAMMING_TARGET = 3
