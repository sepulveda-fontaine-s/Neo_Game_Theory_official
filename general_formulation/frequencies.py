'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Cumulative execution, regime, contextual, and state frequencies.

Scientific basis
----------------

- Part I Eqs. (23)–(26): agreement, contextual, and disagreement frequencies
  and their partition identity.
- Section 5.3, Plot 3: cumulative Human and AI execution frequencies
  ``f_H(T)`` and ``f_AI(T)`` and the identity ``f_H(T) + f_AI(T) = 1``.
- Table 1: ``f_H(0) = f_AI(0) = 0``.
- Appendix B pseudocode: execution, regime, and contextual-owner counters are
  updated after each realized decision. Contextual ownership is undefined
  when the contextual regime has not occurred.


This module receives ``owner_T`` and ``regime_T`` after delegation. It neither
classifies ``D_JS_T`` nor selects an executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .numerics import DEFAULT_ATOL, as_probability


OWNER_VALUES = ("H", "AI")
REGIME_VALUES = ("agreement", "contextual", "disagreement")


@dataclass(frozen=True, slots=True)
class FrequencyState:
    """Cumulative counters after the latest realized decision.

    Frequencies are derived properties rather than independently stored data.
    This prevents count-frequency inconsistencies by construction. All
    ``*_T`` values include the current processed decision.
    """

    decision_count_T: int
    n_H_exec_T: int
    n_AI_exec_T: int
    n_agree_T: int
    n_ctx_T: int
    n_disagree_T: int
    n_H_ctx_T: int
    n_AI_ctx_T: int
    state_counts_T: Mapping[str, int]

    @property
    def f_H_T(self) -> float:
        """Cumulative Human execution frequency."""
        return _ratio(self.n_H_exec_T, self.decision_count_T, zero_value=0.0)

    @property
    def f_AI_T(self) -> float:
        """Cumulative AI execution frequency."""
        return _ratio(self.n_AI_exec_T, self.decision_count_T, zero_value=0.0)

    @property
    def f_agree_T(self) -> float:
        """Cumulative agreement-regime frequency."""
        return _ratio(self.n_agree_T, self.decision_count_T, zero_value=0.0)

    @property
    def f_ctx_T(self) -> float:
        """Cumulative contextual-regime frequency."""
        return _ratio(self.n_ctx_T, self.decision_count_T, zero_value=0.0)

    @property
    def f_disagree_T(self) -> float:
        """Cumulative disagreement-regime frequency."""
        return _ratio(self.n_disagree_T, self.decision_count_T, zero_value=0.0)

    @property
    def f_H_ctx_T(self) -> float:
        """Human ownership frequency conditional on contextual entry."""
        return _ratio(self.n_H_ctx_T, self.n_ctx_T, zero_value=float("nan"))

    @property
    def f_AI_ctx_T(self) -> float:
        """AI ownership frequency conditional on contextual entry."""
        return _ratio(self.n_AI_ctx_T, self.n_ctx_T, zero_value=float("nan"))

    @property
    def state_frequencies_T(self) -> dict[str, float]:
        """Return cumulative realized-state frequencies on the declared support."""
        return {
            state: _ratio(count, self.decision_count_T, zero_value=0.0)
            for state, count in self.state_counts_T.items()
        }


def _ratio(numerator: int, denominator: int, *, zero_value: float) -> float:
    """Return one exact count ratio or the declared zero-denominator value."""
    if denominator == 0:
        return zero_value
    return float(numerator / denominator)


def _validated_atol(atol: float) -> float:
    """Return a finite, nonnegative absolute tolerance."""
    if isinstance(atol, (bool, np.bool_)):
        raise TypeError("atol must be a real number, not Boolean.")
    value = float(atol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("atol must be a finite, nonnegative number.")
    return value


def _as_nonnegative_integer(value: int, *, name: str) -> int:
    """Return a nonnegative integer and reject Boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return integer


def _as_state_support(states: Sequence[str]) -> tuple[str, ...]:
    """Return a nonempty ordered support of unique state labels."""
    if isinstance(states, (str, bytes)):
        raise TypeError("states must be a sequence of labels, not a string.")

    labels = tuple(states)
    if not labels:
        raise ValueError("states cannot be empty.")

    seen: set[str] = set()
    for index, state in enumerate(labels):
        if not isinstance(state, str):
            raise TypeError(f"states[{index}] must be a string.")
        if not state:
            raise ValueError(f"states[{index}] cannot be empty.")
        if state in seen:
            raise ValueError(f"Duplicate state label: {state!r}.")
        seen.add(state)
    return labels


def _as_owner(owner_T: str) -> str:
    """Return the realized owner/executor label."""
    if not isinstance(owner_T, str):
        raise TypeError("owner_T must be a string.")
    if owner_T not in OWNER_VALUES:
        raise ValueError(
            f"owner_T must be one of {OWNER_VALUES}; received {owner_T!r}."
        )
    return owner_T


def _as_regime(regime_T: str) -> str:
    """Return the realized delegation-regime label."""
    if not isinstance(regime_T, str):
        raise TypeError("regime_T must be a string.")
    if regime_T not in REGIME_VALUES:
        raise ValueError(
            f"regime_T must be one of {REGIME_VALUES}; received {regime_T!r}."
        )
    return regime_T


def _as_state_label(s_T: str, *, states: tuple[str, ...]) -> str:
    """Return the realized state on the declared support."""
    if not isinstance(s_T, str):
        raise TypeError("s_T must be a string state label.")
    if s_T not in states:
        raise KeyError(f"s_T={s_T!r} is outside the state support {states}.")
    return s_T


def initialize_frequency_state(states: Sequence[str]) -> FrequencyState:
    """Create the zero-count state used before the first decision."""
    labels = _as_state_support(states)
    return FrequencyState(
        decision_count_T=0,
        n_H_exec_T=0,
        n_AI_exec_T=0,
        n_agree_T=0,
        n_ctx_T=0,
        n_disagree_T=0,
        n_H_ctx_T=0,
        n_AI_ctx_T=0,
        state_counts_T=MappingProxyType({state: 0 for state in labels}),
    )


def validate_frequency_state(
    frequency_state: FrequencyState,
    *,
    atol: float = DEFAULT_ATOL,
) -> FrequencyState:
    """Validate all count partitions and their derived frequencies.

    The returned object contains a defensive, read-only copy of the state
    counts. ``atol`` is used only to verify the theoretical frequency-sum
    identities against floating-point roundoff.
    """
    if not isinstance(frequency_state, FrequencyState):
        raise TypeError("frequency_state must be a FrequencyState instance.")
    tolerance = _validated_atol(atol)

    count_names = (
        "decision_count_T",
        "n_H_exec_T",
        "n_AI_exec_T",
        "n_agree_T",
        "n_ctx_T",
        "n_disagree_T",
        "n_H_ctx_T",
        "n_AI_ctx_T",
    )
    counts = {
        name: _as_nonnegative_integer(getattr(frequency_state, name), name=name)
        for name in count_names
    }
    total = counts["decision_count_T"]

    if counts["n_H_exec_T"] + counts["n_AI_exec_T"] != total:
        raise ValueError("n_H_exec_T + n_AI_exec_T must equal decision_count_T.")
    if (
        counts["n_agree_T"]
        + counts["n_ctx_T"]
        + counts["n_disagree_T"]
        != total
    ):
        raise ValueError(
            "n_agree_T + n_ctx_T + n_disagree_T must equal decision_count_T."
        )
    if counts["n_H_ctx_T"] + counts["n_AI_ctx_T"] != counts["n_ctx_T"]:
        raise ValueError("n_H_ctx_T + n_AI_ctx_T must equal n_ctx_T.")

    if not isinstance(frequency_state.state_counts_T, Mapping):
        raise TypeError("state_counts_T must be a mapping.")
    if not frequency_state.state_counts_T:
        raise ValueError("state_counts_T cannot be empty.")

    state_counts: dict[str, int] = {}
    for state, count in frequency_state.state_counts_T.items():
        if not isinstance(state, str) or not state:
            raise ValueError("state_counts_T keys must be nonempty strings.")
        state_counts[state] = _as_nonnegative_integer(
            count,
            name=f"state_counts_T[{state!r}]",
        )
    if sum(state_counts.values()) != total:
        raise ValueError("State-visit counts must sum to decision_count_T.")

    normalized = FrequencyState(
        **{name: counts[name] for name in count_names},
        state_counts_T=MappingProxyType(state_counts),
    )

    if total > 0:
        execution_sum = normalized.f_H_T + normalized.f_AI_T
        regime_sum = (
            normalized.f_agree_T
            + normalized.f_ctx_T
            + normalized.f_disagree_T
        )
        state_sum = sum(normalized.state_frequencies_T.values())
        for value, name in (
            (execution_sum, "f_H_T + f_AI_T"),
            (regime_sum, "f_agree_T + f_ctx_T + f_disagree_T"),
            (state_sum, "sum(state_frequencies_T)"),
        ):
            as_probability(value, name=name, atol=tolerance)
            if not np.isclose(value, 1.0, atol=tolerance, rtol=0.0):
                raise ArithmeticError(f"{name} must equal one.")

    if normalized.n_ctx_T > 0:
        contextual_sum = normalized.f_H_ctx_T + normalized.f_AI_ctx_T
        as_probability(contextual_sum, name="f_H_ctx_T + f_AI_ctx_T", atol=tolerance)
        if not np.isclose(contextual_sum, 1.0, atol=tolerance, rtol=0.0):
            raise ArithmeticError("Contextual ownership frequencies must sum to one.")
    elif not (np.isnan(normalized.f_H_ctx_T) and np.isnan(normalized.f_AI_ctx_T)):
        raise ArithmeticError(
            "Contextual ownership frequencies must be NaN when n_ctx_T is zero."
        )

    return normalized


def update_frequency_state(
    frequency_state_before: FrequencyState,
    *,
    s_T: str,
    owner_T: str,
    regime_T: str,
    atol: float = DEFAULT_ATOL,
) -> FrequencyState:
    """Include decision ``T`` in every cumulative counter and frequency.

    In the contextual regime, ``owner_T`` also increments the corresponding
    conditional ownership count. The previous state is never mutated.
    """
    before = validate_frequency_state(frequency_state_before, atol=atol)
    states = tuple(before.state_counts_T)
    state = _as_state_label(s_T, states=states)
    owner = _as_owner(owner_T)
    regime = _as_regime(regime_T)

    state_counts = dict(before.state_counts_T)
    state_counts[state] += 1

    updated = FrequencyState(
        decision_count_T=before.decision_count_T + 1,
        n_H_exec_T=before.n_H_exec_T + int(owner == "H"),
        n_AI_exec_T=before.n_AI_exec_T + int(owner == "AI"),
        n_agree_T=before.n_agree_T + int(regime == "agreement"),
        n_ctx_T=before.n_ctx_T + int(regime == "contextual"),
        n_disagree_T=before.n_disagree_T + int(regime == "disagreement"),
        n_H_ctx_T=before.n_H_ctx_T
        + int(regime == "contextual" and owner == "H"),
        n_AI_ctx_T=before.n_AI_ctx_T
        + int(regime == "contextual" and owner == "AI"),
        state_counts_T=MappingProxyType(state_counts),
    )
    return validate_frequency_state(updated, atol=atol)


__all__ = [
    "FrequencyState",
    "OWNER_VALUES",
    "REGIME_VALUES",
    "initialize_frequency_state",
    "update_frequency_state",
    "validate_frequency_state",
]
