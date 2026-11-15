"""Simulating imperfect players.

The dataset holds one perfect description per object - 117 rows. You cannot
train an object-level classifier on 117 examples with a single example per
class, which is exactly why the original project fell back to predicting a
9-way category instead of the object itself.

The fix is to model how a real player answers. People disagree about whether
a laptop is "big", misremember whether a penguin can swim, and occasionally
just misclick. So each training example is a prototype with some answers
flipped.

What this does NOT do is create information. Flipping bits adds no knowledge
about the objects; it teaches the model the *noise process*, which is the
condition it actually runs under. A model trained only on the 117 clean rows
assumes the player is infallible, and collapses the moment one answer is
wrong. That is the whole point.

Only flips are simulated here, not "maybe" answers. Flips keep the data
binary, which matters because BernoulliNB - the correct Naive Bayes variant
for binary attributes - thresholds its input. Unknown answers are better
handled by marginalising them at inference time, which is Phase 4's
Bayesian engine rather than something to bake into the training data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Flip probabilities to train and evaluate across. 0.0 keeps the perfect
# prototype in the mix; 0.20 is a player getting roughly one answer in five
# wrong, which is already a fairly careless human.
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class NoisyDataset:
    X: np.ndarray        # (n_samples, n_features) int8, {0,1}
    y: np.ndarray        # (n_samples,) int, index into prototype names
    noise: np.ndarray    # (n_samples,) float, the flip rate used per sample

    def __len__(self) -> int:
        return self.X.shape[0]

    def at_noise(self, level: float):
        """The subset generated at one flip rate, for per-level reporting."""
        m = self.noise == level
        return self.X[m], self.y[m]


def make_noisy(
    X_proto: np.ndarray,
    samples_per_object: int,
    noise_levels: tuple[float, ...] = NOISE_LEVELS,
    seed: int = 0,
) -> NoisyDataset:
    """Expand prototypes into simulated answer sheets.

    Every object gets the same number of samples, and the flip rates are
    spread evenly across `noise_levels`, so no object or noise level is
    over-represented and accuracy can be reported per level.
    """
    if samples_per_object < len(noise_levels):
        raise ValueError(
            "samples_per_object (%d) must be at least the number of noise "
            "levels (%d) so every level is represented"
            % (samples_per_object, len(noise_levels))
        )

    rng = np.random.default_rng(seed)
    n_objects, n_features = X_proto.shape

    # Assign noise levels round-robin, so counts differ by at most one.
    level_for_sample = np.array(
        [noise_levels[i % len(noise_levels)] for i in range(samples_per_object)],
        dtype=float,
    )

    Xs, ys, ns = [], [], []
    for obj in range(n_objects):
        block = np.repeat(X_proto[obj][None, :], samples_per_object, axis=0)
        flip_prob = level_for_sample[:, None]
        flips = rng.random((samples_per_object, n_features)) < flip_prob
        block = np.where(flips, 1 - block, block).astype(np.int8)

        Xs.append(block)
        ys.append(np.full(samples_per_object, obj, dtype=int))
        ns.append(level_for_sample.copy())

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    noise = np.concatenate(ns, axis=0)

    perm = rng.permutation(X.shape[0])
    return NoisyDataset(X=X[perm], y=y[perm], noise=noise[perm])


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction of samples whose true object is in the k highest-scored.

    Reported alongside top-1 because the game can offer a second and third
    guess: being in the top 3 is a usable outcome, not a failure.
    """
    if k >= proba.shape[1]:
        return 1.0
    topk = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))
