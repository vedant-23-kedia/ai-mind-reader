"""Loading the compiled dataset.

One place that reads data/dataset_clean.csv, so the app, the training scripts
and the tests cannot drift apart on how the data is interpreted.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from . import config as cfg


@dataclass(frozen=True)
class Prototypes:
    """The canonical attribute vector for every object.

    One row per object: the answers a player would give if they described the
    object perfectly and never made a mistake.
    """

    names: list[str]
    features: list[str]
    X: np.ndarray            # shape (n_objects, n_features), dtype int8, {0,1}
    categories: list[str]

    @property
    def n_objects(self) -> int:
        return len(self.names)

    @property
    def n_features(self) -> int:
        return len(self.features)

    def index_of(self, name: str) -> int:
        return self.names.index(name)


def load_prototypes() -> Prototypes:
    if not cfg.DATASET.exists():
        raise FileNotFoundError(
            "%s not found. Build it with: python scripts/build_dataset.py"
            % cfg.DATASET
        )

    with cfg.DATASET.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    features = header[1:]
    names = [r[0] for r in body]

    X = np.array([[int(v) for v in r[1:]] for r in body], dtype=np.int8)
    if not np.isin(X, (0, 1)).all():
        raise ValueError("dataset contains non-binary values")

    with cfg.CATEGORY_MAP.open(newline="", encoding="utf-8") as fh:
        cat_of = {r["object"]: r["category"] for r in csv.DictReader(fh)}
    missing = [n for n in names if n not in cat_of]
    if missing:
        raise ValueError("objects missing from category map: %s" % missing)

    return Prototypes(
        names=names,
        features=features,
        X=X,
        categories=[cat_of[n] for n in names],
    )
