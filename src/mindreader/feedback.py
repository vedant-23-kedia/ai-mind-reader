"""Learning from games the engine got wrong.

Every phase up to now assumed players make mistakes at one uniform rate. They
do not. People are reliably certain whether something is an animal and
reliably vague about whether it is "big", and no amount of simulation can
reveal which attributes those are - only real answers can.

When the engine guesses wrong, the app records the player's full answer sheet
alongside what they were actually thinking of. That gives two things:

  * **Per-attribute error rates.** If the player was thinking of a dog and
    said "no" to "is it an animal?", that answer was wrong. Counting those
    per attribute, over many games, gives a measured error rate per question
    rather than one assumed constant.

  * **Candidate new objects.** If they name something the dataset has never
    heard of, their answers are already a partial description of it. Several
    reports can be merged into a draft entry for `data/source/objects.json`.

Nothing here edits the dataset automatically. A draft is proposed for review,
because an attribute vector assembled from a handful of strangers' answers is
a suggestion, not ground truth.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import Prototypes

# Pseudo-count pulling each attribute's estimate toward the global rate.
# With only a handful of observations a raw rate is useless - one wrong
# answer out of one would say "this attribute is answered wrongly 100% of the
# time". A Beta prior worth this many observations keeps early estimates
# sane and lets real data win once there is enough of it.
SMOOTHING_STRENGTH = 20.0

# Keep estimates inside what the engine accepts, and away from the extremes:
# an attribute that is never answered wrongly in 50 games is not therefore
# answered wrongly 0% of the time for ever.
MIN_RATE = 0.01
MAX_RATE = 0.45


@dataclass
class FeedbackRow:
    timestamp: str
    guessed: str
    actual: str
    answers: dict[str, int | None]
    # "correct" or "wrong". Older logs predate this column and are treated
    # as "wrong", which is what they recorded.
    outcome: str = "wrong"


@dataclass
class NoiseEstimate:
    global_rate: float
    per_attribute: dict[str, float]
    observations: dict[str, int]
    errors: dict[str, int]
    rows_used: int
    rows_skipped: int

    def as_vector(self, features: list[str]) -> np.ndarray:
        return np.array(
            [self.per_attribute.get(f, self.global_rate) for f in features],
            dtype=np.float64,
        )


@dataclass
class ObjectCandidate:
    name: str
    reports: int = 0
    # attribute -> [yes_count, no_count]
    votes: dict[str, list[int]] = field(default_factory=dict)

    def draft_attributes(self, min_votes: int = 1) -> list[str]:
        """Attributes a majority of reporters said YES to."""
        out = []
        for feat, (yes, no) in sorted(self.votes.items()):
            if yes + no >= min_votes and yes > no:
                out.append(feat)
        return out

    def disputed(self) -> list[str]:
        """Attributes reporters disagreed about - these need a human."""
        return sorted(f for f, (y, n) in self.votes.items() if y > 0 and n > 0)


def load_feedback(path: Path) -> list[FeedbackRow]:
    """Read the log, skipping rows too malformed to use.

    The log is written by players mid-game, so it has to be read defensively:
    a half-written final row or a hand-edited file must not crash the
    learning step.
    """
    if not path.exists():
        return []
    rows: list[FeedbackRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                answers = json.loads(raw.get("answers_json") or "{}")
                if not isinstance(answers, dict):
                    continue
                actual = (raw.get("actual") or "").strip()
                if not actual:
                    continue
                rows.append(FeedbackRow(
                    timestamp=raw.get("timestamp", ""),
                    guessed=(raw.get("guessed") or "").strip(),
                    actual=actual,
                    answers={k: v for k, v in answers.items()},
                    outcome=(raw.get("outcome") or "wrong").strip(),
                ))
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
    return rows


def estimate_attribute_noise(
    rows: list[FeedbackRow],
    protos: Prototypes,
    smoothing: float = SMOOTHING_STRENGTH,
    default_rate: float = 0.10,
) -> NoiseEstimate:
    """Measure how often each attribute is answered wrongly.

    Only rows naming a known object can be used, since otherwise there is
    nothing to compare the answers against.
    """
    fidx = {f: i for i, f in enumerate(protos.features)}
    name_set = set(protos.names)

    observations = {f: 0 for f in protos.features}
    errors = {f: 0 for f in protos.features}
    used = skipped = 0

    for row in rows:
        if row.actual not in name_set:
            skipped += 1
            continue
        used += 1
        oi = protos.index_of(row.actual)
        for feat, value in row.answers.items():
            if feat not in fidx or value is None:
                continue          # unknown attribute, or a "maybe"
            if value not in (0, 1):
                continue
            observations[feat] += 1
            if int(protos.X[oi, fidx[feat]]) != int(value):
                errors[feat] += 1

    total_obs = sum(observations.values())
    total_err = sum(errors.values())

    # The global rate is itself smoothed, toward the assumed default. Taking
    # it raw would make the whole hierarchy useless on a small log: one wrong
    # answer out of one observation gives a global rate of 100%, and every
    # per-attribute estimate would then shrink toward *that* rather than
    # toward something sensible.
    global_rate = (total_err + smoothing * default_rate) / (total_obs + smoothing)
    global_rate = min(max(global_rate, MIN_RATE), MAX_RATE)

    per_attribute = {}
    for feat in protos.features:
        n, e = observations[feat], errors[feat]
        # Beta-smoothed toward the global rate
        rate = (e + smoothing * global_rate) / (n + smoothing)
        per_attribute[feat] = float(min(max(rate, MIN_RATE), MAX_RATE))

    return NoiseEstimate(
        global_rate=float(global_rate),
        per_attribute=per_attribute,
        observations=observations,
        errors=errors,
        rows_used=used,
        rows_skipped=skipped,
    )


def new_object_candidates(
    rows: list[FeedbackRow], protos: Prototypes
) -> list[ObjectCandidate]:
    """Objects players named that the dataset does not contain."""
    known = set(protos.names)
    valid_feats = set(protos.features)
    found: dict[str, ObjectCandidate] = {}

    for row in rows:
        if row.actual in known:
            continue
        cand = found.setdefault(row.actual, ObjectCandidate(name=row.actual))
        cand.reports += 1
        for feat, value in row.answers.items():
            if feat not in valid_feats or value not in (0, 1):
                continue
            tally = cand.votes.setdefault(feat, [0, 0])
            tally[0 if value == 1 else 1] += 1

    return sorted(found.values(), key=lambda c: -c.reports)


def write_noise_profile(est: NoiseEstimate, path: Path) -> None:
    path.write_text(json.dumps({
        "global_rate": round(est.global_rate, 4),
        "rows_used": est.rows_used,
        "rows_skipped": est.rows_skipped,
        "smoothing_strength": SMOOTHING_STRENGTH,
        "note": (
            "Per-attribute error rates measured from real games. Each is "
            "smoothed toward the global rate by a Beta prior worth "
            "%g observations, so attributes with little data stay near the "
            "global default instead of swinging to 0 or 1."
            % SMOOTHING_STRENGTH
        ),
        "per_attribute": {k: round(v, 4) for k, v in
                          sorted(est.per_attribute.items())},
        "observations": {k: v for k, v in sorted(est.observations.items())
                         if v > 0},
        "errors": {k: v for k, v in sorted(est.errors.items()) if v > 0},
    }, indent=2), encoding="utf-8")


def load_noise_profile(path: Path, features: list[str],
                       default_rate: float = 0.10) -> np.ndarray | None:
    """Load a saved profile as a per-attribute vector, or None if absent."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    per = data.get("per_attribute") or {}
    if not per:
        return None
    fallback = float(data.get("global_rate", default_rate))
    vec = np.array([float(per.get(f, fallback)) for f in features],
                   dtype=np.float64)
    return np.clip(vec, MIN_RATE, MAX_RATE)
