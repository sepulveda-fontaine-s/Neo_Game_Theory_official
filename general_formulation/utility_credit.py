'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''

"""Owner-specific EWMA utility-credit traces.

Scientific basis
----------------

- Definition 6: ``owner_T`` is the origin of the proposal selected for
  execution. It is a credit label and must be supplied explicitly rather than
  inferred from the numerical action when the two proposals coincide.
- Part I Eq. (22) [estimated_utility]; Part II Eq. (14) [estimated_utility]: 
  every Human and AI state-action trace entry is multiplied by
  ``1 - beta`` at each decision instant, and only the owner of the executed
  proposal receives the positive ``beta`` reinforcement at
  ``(s_T, a_star_T)``.
- Section 5.3(c): ``Uhat_coal_T`` is recorded after the EWMA update as
  ``Uhat_{owner_T}^{T+1}(s_T, a_star_T)``. It is a derived diagnostic and not a
  third independently updated trace.
- Table 1: both utility-credit tables are initialized at zero.

The structural utilities ``U_H`` and ``U_AI`` are deliberately absent from
this module. Utility-credit traces do not define ``reward_structural_T`` and do
not enter the theoretical or computational Bellman recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .numerics import DEFAULT_ATOL, as_probability


UtilityCreditVector: TypeAlias = NDArray[np.float64]
UtilityCreditTable: TypeAlias = dict[str, UtilityCreditVector]

OWNER_VALUES = ("H", "AI")


@dataclass(frozen=True, slots=True)
class UtilityCreditUpdate:
    """Post-EWMA tables and realized utility-credit values for decision ``T``.

    The scalar field names are identical to the longitudinal CSV columns used
    by Plot 2. They are recorded after processing decision instant ``T``.
    """

    Uhat_H_T_plus_1: UtilityCreditTable
    Uhat_AI_T_plus_1: UtilityCreditTable
    Uhat_H_realized_T: float
    Uhat_AI_realized_T: float
    Uhat_coal_T: float


def _validated_atol(atol: float) -> float:
    """Return a finite, nonnegative absolute tolerance."""
    value = float(atol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("atol must be a finite, nonnegative number.")
    return value


def _as_positive_integer(value: int, *, name: str) -> int:
    """Return a strictly positive integer and reject booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return integer


def _as_action_index(a_star_T: int, *, n_actions: int) -> int:
    """Return a validated executed-action index."""
    if isinstance(a_star_T, bool) or not isinstance(a_star_T, Integral):
        raise TypeError("a_star_T must be an integer action index.")

    action_index = int(a_star_T)
    if action_index < 0 or action_index >= n_actions:
        raise IndexError(
            f"a_star_T={action_index} is outside the action support "
            f"[0, {n_actions - 1}]."
        )
    return action_index


def _as_owner(owner_T: str) -> str:
    """Return the validated owner-of-origin label from Definition 6."""
    if not isinstance(owner_T, str):
        raise TypeError("owner_T must be a string.")
    if owner_T not in OWNER_VALUES:
        raise ValueError(
            f"owner_T must be one of {OWNER_VALUES}; received {owner_T!r}."
        )
    return owner_T


def _as_state_label(s_T: str) -> str:
    """Return a nonempty state label."""
    if not isinstance(s_T, str):
        raise TypeError("s_T must be a string state label.")
    if not s_T:
        raise ValueError("s_T cannot be empty.")
    return s_T


def initialize_utility_credit_table(
    states: Sequence[str],
    *,
    n_actions: int,
) -> UtilityCreditTable:
    """Create the zero-initialized utility-credit table specified in Part I Table 1."""
    action_count = _as_positive_integer(n_actions, name="n_actions")

    if isinstance(states, (str, bytes)):
        raise TypeError("states must be a sequence of state labels, not a string.")

    state_labels = list(states)
    if not state_labels:
        raise ValueError("states cannot be empty.")

    validated_states: list[str] = []
    seen: set[str] = set()
    for index, state in enumerate(state_labels):
        if not isinstance(state, str):
            raise TypeError(f"states[{index}] must be a string.")
        if not state:
            raise ValueError(f"states[{index}] cannot be empty.")
        if state in seen:
            raise ValueError(f"Duplicate state label: {state!r}.")
        seen.add(state)
        validated_states.append(state)

    return {
        state: np.zeros(action_count, dtype=np.float64)
        for state in validated_states
    }


def as_utility_credit_table(
    values: Mapping[str, ArrayLike],
    *,
    name: str = "Uhat_i_T",
    atol: float = DEFAULT_ATOL,
) -> UtilityCreditTable:
    """Return a strict finite state-action utility-credit table.

    Each entry must already lie in ``[0, 1]`` within ``atol``. Unlike a policy,
    a utility-credit row is not required to sum to one. The input mapping and
    its arrays are never modified.
    """
    tolerance = _validated_atol(atol)

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping from states to vectors.")
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    result: UtilityCreditTable = {}
    expected_actions: int | None = None

    for state, raw_vector in values.items():
        if not isinstance(state, str):
            raise TypeError(f"Every state key in {name} must be a string.")
        if not state:
            raise ValueError(f"State keys in {name} cannot be empty.")

        vector = np.asarray(raw_vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"{name}[{state!r}] must be one-dimensional.")
        if vector.size == 0:
            raise ValueError(f"{name}[{state!r}] cannot be empty.")
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name}[{state!r}] must contain only finite values."
            )

        if expected_actions is None:
            expected_actions = int(vector.size)
        elif vector.size != expected_actions:
            raise ValueError(
                f"Every state in {name} must use the same action support; "
                f"expected {expected_actions}, received {vector.size} for "
                f"state {state!r}."
            )

        if np.any(vector < -tolerance) or np.any(vector > 1.0 + tolerance):
            raise ValueError(
                f"Every entry of {name}[{state!r}] must lie in [0, 1] "
                f"within atol={tolerance:g}."
            )

        validated = vector.copy()
        validated[validated < 0.0] = 0.0
        validated[validated > 1.0] = 1.0
        result[state] = validated

    return result


def _validate_matching_tables(
    Uhat_H_T: Mapping[str, ArrayLike],
    Uhat_AI_T: Mapping[str, ArrayLike],
    *,
    atol: float,
) -> tuple[UtilityCreditTable, UtilityCreditTable]:
    """Return validated Human and AI tables with identical supports."""
    H_table = as_utility_credit_table(
        Uhat_H_T,
        name="Uhat_H_T",
        atol=atol,
    )
    AI_table = as_utility_credit_table(
        Uhat_AI_T,
        name="Uhat_AI_T",
        atol=atol,
    )

    if tuple(H_table.keys()) != tuple(AI_table.keys()):
        if set(H_table) != set(AI_table):
            raise ValueError(
                "Uhat_H_T and Uhat_AI_T must contain the same state labels."
            )
        # Preserve the Human table's deterministic state order.
        AI_table = {state: AI_table[state] for state in H_table}

    for state in H_table:
        if H_table[state].shape != AI_table[state].shape:
            raise ValueError(
                "Uhat_H_T and Uhat_AI_T must use the same action support "
                f"at state {state!r}."
            )

    return H_table, AI_table


def update_utility_credit(
    Uhat_H_T: Mapping[str, ArrayLike],
    Uhat_AI_T: Mapping[str, ArrayLike],
    *,
    s_T: str,
    a_star_T: int,
    owner_T: str,
    beta: float,
    atol: float = DEFAULT_ATOL,
) -> UtilityCreditUpdate:
    """Apply Part I Eq. (22) [estimated_utility] and Part II Eq. (14)
    [estimated_utility], and derive the post-update Plot-2 scalars.

    Every state-action entry in both tables decays by ``1 - beta``. Exactly one
    positive reinforcement is then added: ``beta`` at ``(s_T, a_star_T)`` in
    the table identified by ``owner_T``.

    ``owner_T`` must be supplied by the execution module. It must not be
    inferred by comparing ``a_star_T`` with the two proposal values, because
    coincident proposals do not identify which proposal was selected.
    """
    H_table, AI_table = _validate_matching_tables(
        Uhat_H_T,
        Uhat_AI_T,
        atol=atol,
    )
    state = _as_state_label(s_T)
    owner = _as_owner(owner_T)

    if state not in H_table:
        raise KeyError(f"Unknown state s_T={state!r}.")

    action_index = _as_action_index(
        a_star_T,
        n_actions=H_table[state].size,
    )

    beta_value = as_probability(beta, name="beta", atol=atol)
    if beta_value <= 0.0:
        raise ValueError("beta must lie in (0, 1].")

    decay = 1.0 - beta_value
    Uhat_H_T_plus_1 = {
        state_label: decay * vector
        for state_label, vector in H_table.items()
    }
    Uhat_AI_T_plus_1 = {
        state_label: decay * vector
        for state_label, vector in AI_table.items()
    }

    # Part I Eq. (22) [estimated_utility]; Part II Eq. (14).
    # Only the owner receives the positive reinforcement term: 
    if owner == "H":
        Uhat_H_T_plus_1[state][action_index] += beta_value
    else:
        Uhat_AI_T_plus_1[state][action_index] += beta_value

    Uhat_H_T_plus_1 = as_utility_credit_table(
        Uhat_H_T_plus_1,
        name="Uhat_H_T_plus_1",
        atol=atol,
    )
    Uhat_AI_T_plus_1 = as_utility_credit_table(
        Uhat_AI_T_plus_1,
        name="Uhat_AI_T_plus_1",
        atol=atol,
    )

    # Section 5.3(c): all three diagnostics are post-EWMA values. The
    # coalition series is selected from the owner's table and is not updated
    # independently.
    Uhat_H_realized_T = float(Uhat_H_T_plus_1[state][action_index])
    Uhat_AI_realized_T = float(Uhat_AI_T_plus_1[state][action_index])
    Uhat_coal_T = (
        Uhat_H_realized_T if owner == "H" else Uhat_AI_realized_T
    )

    return UtilityCreditUpdate(
        Uhat_H_T_plus_1=Uhat_H_T_plus_1,
        Uhat_AI_T_plus_1=Uhat_AI_T_plus_1,
        Uhat_H_realized_T=Uhat_H_realized_T,
        Uhat_AI_realized_T=Uhat_AI_realized_T,
        Uhat_coal_T=Uhat_coal_T,
    )


__all__ = [
    "OWNER_VALUES",
    "UtilityCreditTable",
    "UtilityCreditUpdate",
    "UtilityCreditVector",
    "as_utility_credit_table",
    "initialize_utility_credit_table",
    "update_utility_credit",
]
