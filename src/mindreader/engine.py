"""Bayesian belief engine.

The original engine deleted candidates. Answer "no" to "Is it an animal?" and
every animal was struck off permanently - so one wrong answer did not just
mislead the guess, it made the right answer unreachable for the rest of the
game. There was no way back, "maybe" was thrown away, and the confidence
percentage was an ad-hoc score that could read 100% while being wrong.

This keeps a probability distribution over every object instead, and
multiplies it by a likelihood after each answer:

    P(answer | object) = 1 - noise   if the answer matches the object
                       =     noise   if it contradicts it

Because a contradicting answer costs a factor of `noise` rather than zeroing
the object out, a mistake is survivable: later answers can outvote it. Three
things follow from that single change.

  * Nothing is ever eliminated, so a wrong answer is recoverable.
  * The reported confidence is a real posterior probability, so it can be
    checked against how often the guess is actually right.
  * An unknown answer is handled by simply not updating - the attribute is
    marginalised out - which is what "maybe" should always have meant.

This is Naive Bayes, applied one question at a time. The Phase 3 comparison
is what justifies the Bernoulli form: on binary attributes it beat a Gaussian
likelihood by 7.2 points and beat a 256x128 neural network by 6.5.

Questions are chosen by expected information gain: the question whose answer
is expected to reduce the entropy of the belief state the most.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import Prototypes

# Probability that a player gives the wrong answer to a question they did
# answer. 0.1 matches the middle of the range the models were trained across
# and leaves enough slack that two or three mistakes are still recoverable.
DEFAULT_NOISE = 0.10

# Stop once one object holds this much of the belief mass.
DEFAULT_CONFIDENCE_THRESHOLD = 0.92

# Answer encoding. MAYBE is deliberately not a number: it is the absence of
# evidence, not a value halfway between yes and no.
YES = 1
NO = 0
MAYBE = None


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits, ignoring zero-probability entries."""
    nz = p[p > 0]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log2(nz)))


@dataclass
class Answer:
    feature: str
    value: int | None       # YES, NO, or MAYBE


# Weight applied to each answer's log-likelihood. 1.0 is textbook Naive
# Bayes, which treats every attribute as independent evidence. Ours are not
# independent - the dataset deliberately enforces rules like
# is_mammal -> is_animal, so answering both effectively counts the same fact
# twice and the posterior ends up overconfident. A weight below 1 discounts
# each answer to compensate. scripts/evaluate.py fits the value empirically.
DEFAULT_LIKELIHOOD_WEIGHT = 1.0


@dataclass
class BeliefEngine:
    """A belief state over objects, updated by answers."""

    protos: Prototypes
    noise: float = DEFAULT_NOISE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    likelihood_weight: float = DEFAULT_LIKELIHOOD_WEIGHT
    answers: list[Answer] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.noise < 0.5:
            raise ValueError(
                "noise must be in (0, 0.5); got %r. At 0 a single wrong "
                "answer is unrecoverable, and at >=0.5 an answer is no better "
                "than a coin flip." % self.noise
            )
        if not 0.0 < self.likelihood_weight <= 1.0:
            raise ValueError(
                "likelihood_weight must be in (0, 1]; got %r"
                % self.likelihood_weight
            )
        X = self.protos.X.astype(np.float64)
        # like1[o, f] = P(player answers YES | object o)
        self._like1 = np.where(X == 1, 1.0 - self.noise, self.noise)
        self._like0 = 1.0 - self._like1
        self._feat_index = {f: i for i, f in enumerate(self.protos.features)}
        self._log_prior = np.full(
            self.protos.n_objects, -np.log(self.protos.n_objects)
        )

    # ---- belief state ----------------------------------------------------
    def log_posterior(self) -> np.ndarray:
        logp = self._log_prior.copy()
        for ans in self.answers:
            if ans.value is MAYBE:
                continue  # no evidence: marginalise this attribute out
            k = self._feat_index[ans.feature]
            like = self._like1[:, k] if ans.value == YES else self._like0[:, k]
            logp = logp + self.likelihood_weight * np.log(like)
        return logp

    def posterior(self) -> np.ndarray:
        logp = self.log_posterior()
        logp = logp - logp.max()           # stabilise before exponentiating
        p = np.exp(logp)
        return p / p.sum()

    def entropy(self) -> float:
        return _entropy(self.posterior())

    # ---- answering -------------------------------------------------------
    def asked(self) -> set[str]:
        return {a.feature for a in self.answers}

    def answer(self, feature: str, value: int | None) -> None:
        if feature not in self._feat_index:
            raise KeyError("unknown attribute %r" % feature)
        if feature in self.asked():
            raise ValueError("%r has already been answered" % feature)
        if value not in (YES, NO, MAYBE):
            raise ValueError("value must be YES, NO or MAYBE; got %r" % value)
        self.answers.append(Answer(feature, value))

    def undo(self) -> Answer | None:
        """Take back the last answer.

        Possible only because the belief state is a pure function of the
        answer list - the old engine had destroyed the information needed to
        reconstruct it.
        """
        return self.answers.pop() if self.answers else None

    def reset(self) -> None:
        self.answers.clear()

    # ---- guessing --------------------------------------------------------
    def top(self, k: int = 3) -> list[tuple[str, float]]:
        p = self.posterior()
        idx = np.argsort(-p)[:k]
        return [(self.protos.names[i], float(p[i])) for i in idx]

    def best_guess(self) -> tuple[str, float]:
        return self.top(1)[0]

    def is_confident(self) -> bool:
        return self.posterior().max() >= self.confidence_threshold

    # ---- question selection ---------------------------------------------
    def information_gain(self) -> dict[str, float]:
        """Expected reduction in entropy, in bits, for each unasked question.

        The old engine looked for the question splitting candidates closest
        to 50/50, and only bothered below 10 candidates. This is the measure
        that heuristic was approximating, and it applies from the first
        question.
        """
        p = self.posterior()
        h_now = _entropy(p)

        # Probability the player answers YES to each attribute, and the
        # posterior that would result - computed for all attributes at once.
        w = self.likelihood_weight

        # How likely each answer is must come from the TRUE likelihood, since
        # that is what actually generates the player's answer. These sum to 1
        # because like1 + like0 = 1.
        p_yes = (p[:, None] * self._like1).sum(axis=0)
        p_no = (p[:, None] * self._like0).sum(axis=0)

        # The belief that would RESULT uses the weighted likelihood, because
        # that is how this engine updates. Mixing the two the other way round
        # - drawing the answer's probability from the weighted likelihood -
        # breaks the guarantee that information never hurts on average, and
        # produced negative expected gain at weights below 1.
        num1 = p[:, None] * self._like1 ** w     # (n_objects, n_features)
        num0 = p[:, None] * self._like0 ** w
        den1 = num1.sum(axis=0)
        den0 = num0.sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            post1 = np.where(den1 > 0, num1 / den1, 0.0)
            post0 = np.where(den0 > 0, num0 / den0, 0.0)
            h1 = -np.nansum(np.where(post1 > 0, post1 * np.log2(post1), 0.0), axis=0)
            h0 = -np.nansum(np.where(post0 > 0, post0 * np.log2(post0), 0.0), axis=0)

        expected_h = p_yes * h1 + p_no * h0
        gain = h_now - expected_h

        asked = self.asked()
        return {
            f: float(gain[k])
            for f, k in self._feat_index.items()
            if f not in asked
        }

    def best_question(self) -> str | None:
        """The unasked question expected to be most informative."""
        gains = self.information_gain()
        if not gains:
            return None
        return max(gains, key=gains.get)

    # ---- diagnostics -----------------------------------------------------
    def contradictions(self, name: str) -> list[str]:
        """Attributes where the given object disagrees with the answers.

        Lets the result screen explain *why* an object was ranked low,
        instead of it having silently vanished.
        """
        i = self.protos.index_of(name)
        out = []
        for ans in self.answers:
            if ans.value is MAYBE:
                continue
            k = self._feat_index[ans.feature]
            if int(self.protos.X[i, k]) != ans.value:
                out.append(ans.feature)
        return out
