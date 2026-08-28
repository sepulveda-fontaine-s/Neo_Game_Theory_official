'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Structural reward and finite-time computational Bellman backup.

Scientific basis
----------------


- Definition 6: ``owner_T`` identifies the origin of the proposal selected for
  execution. Supplying that label explicitly removes the ambiguity created
  when Human and AI propose the same numerical action.
- Part I Eqs. (38)–(39): the realized coalition reward is structural. It is
  taken from ``U_H`` when the Human proposal is selected and from ``U_AI``
  when the AI proposal is selected.
- Proposition 2 and Eq. (40): conditional on the realized execution
  selector, the Bellman maximization is over the feasible executed-action set
  induced by the two proposals at decision instant ``T``.
- Part I Eq. (41), Part II specified on Eqs. (15)–(16): the simulation maintains one state-indexed finite-time value
  table ``Vhat`` and applies an asynchronous Bellman backup only at ``s_T``,
  using the complete pre-backup value vector on the right-hand side.
- Section 5.3(c), Implementation: ``Vhat_coal_T`` is recorded after the backup
  as ``Vhat^{T+1}(s_T)``. It is derived from the single value table; separate
  ``Vhat_H`` and ``Vhat_AI`` tables do not exist.
- Table 1: the computational value table is initialized at zero.

The EWMA utility-credit traces ``Uhat_H`` and ``Uhat_AI`` are deliberately
absent from this module. They neither define ``reward_structural_T`` nor enter
the theoretical or computational Bellman recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .numerics import DEFAULT_ATOL
from .state_generation import TransitionKernel, transition_distribution


StructuralUtilityVector: TypeAlias = NDArray[np.float64]
StructuralUtilityTable: TypeAlias = dict[str, StructuralUtilityVector]
ValueTable: TypeAlias = dict[str, float]

OWNER_VALUES = ("H", "AI")


@dataclass(frozen=True, slots=True)
class BellmanUpdate:
    """Structural reward and post-backup computational values for decision T.

    ``Vhat_T_plus_1`` is the complete state-indexed table after the
    asynchronous backup. ``Vhat_coal_T`` is the post-backup value at the
    realized state and is therefore a derived reporting series.
    """

    reward_structural_T: float
    Vhat_T_plus_1: ValueTable
    Vhat_coal_T: float


def _validated_atol(atol: float) -> float:
    """Return a finite, nonnegative absolute tolerance."""
    if isinstance(atol, (bool, np.bool_)):
        raise TypeError("atol must be a real number, not Boolean.")

    value = float(atol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("atol must be a finite, nonnegative number.")
    return value


def _as_state_label(value: str, *, states: tuple[str, ...], name: str) -> str:
    """Return one state label on the declared support."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string state label.")
    if value not in states:
        raise KeyError(f"{name}={value!r} is outside the state support {states}.")
    return value


def _as_action_index(value: int, *, n_actions: int, name: str) -> int:
    """Return one action index on ``[0, n_actions - 1]``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer action index.")

    action_index = int(value)
    if action_index < 0 or action_index >= n_actions:
        raise IndexError(
            f"{name}={action_index} is outside the action support "
            f"[0, {n_actions - 1}]."
        )
    return action_index


def _as_owner(owner_T: str) -> str:
    """Return the origin label of the proposal selected for execution."""
    if not isinstance(owner_T, str):
        raise TypeError("owner_T must be a string.")
    if owner_T not in OWNER_VALUES:
        raise ValueError(
            f"owner_T must be one of {OWNER_VALUES}; received {owner_T!r}."
        )
    return owner_T


def _as_gamma(gamma: float) -> float:
    """Return the discount factor required by Part I, ``0 < gamma < 1``."""
    if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real):
        raise TypeError("gamma must be a real number.")

    value = float(gamma)
    if not np.isfinite(value):
        raise ValueError("gamma must be finite.")
    if not 0.0 < value < 1.0:
        raise ValueError("gamma must satisfy 0 < gamma < 1.")
    return value


def _as_state_labels_from_mapping(
    values: Mapping[str, object],
    *,
    name: str,
) -> tuple[str, ...]:
    """Return the nonempty ordered state support of a mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping indexed by state.")
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    states = tuple(values.keys())
    seen: set[str] = set()
    for state in states:
        if not isinstance(state, str):
            raise TypeError(f"Every state key in {name} must be a string.")
        if not state:
            raise ValueError(f"State keys in {name} cannot be empty.")
        if state in seen:
            raise ValueError(f"Duplicate state label in {name}: {state!r}.")
        seen.add(state)
    return states


def as_structural_utility_table(
    values: Mapping[str, ArrayLike],
    *,
    name: str = "U_i",
) -> StructuralUtilityTable:
    """Return a strict finite state-action structural-utility table.

    Structural utilities are real-valued and are therefore not clipped to
    ``[0, 1]``. Every state must use the same nonempty action support. The
    input mapping and its arrays are never modified.
    """
    states = _as_state_labels_from_mapping(values, name=name)

    result: StructuralUtilityTable = {}
    expected_actions: int | None = None

    for state in states:
        vector = np.asarray(values[state], dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"{name}[{state!r}] must be one-dimensional.")
        if vector.size == 0:
            raise ValueError(f"{name}[{state!r}] cannot be empty.")
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name}[{state!r}] must contain only finite utilities."
            )

        if expected_actions is None:
            expected_actions = int(vector.size)
        elif vector.size != expected_actions:
            raise ValueError(
                f"Every state in {name} must use the same action support; "
                f"expected {expected_actions}, received {vector.size} for "
                f"state {state!r}."
            )

        result[state] = vector.copy()

    return result


def _validate_matching_utility_tables(
    U_H: Mapping[str, ArrayLike],
    U_AI: Mapping[str, ArrayLike],
) -> tuple[StructuralUtilityTable, StructuralUtilityTable]:
    """Return Human and AI structural utilities on identical supports."""
    H_table = as_structural_utility_table(U_H, name="U_H")
    AI_table = as_structural_utility_table(U_AI, name="U_AI")

    if set(H_table) != set(AI_table):
        raise ValueError("U_H and U_AI must contain the same state labels.")

    # Preserve the deterministic state order supplied by U_H.
    AI_table = {state: AI_table[state] for state in H_table}

    for state in H_table:
        if H_table[state].shape != AI_table[state].shape:
            raise ValueError(
                "U_H and U_AI must use the same action support at "
                f"state {state!r}."
            )

    return H_table, AI_table


def initialize_value_table(states: Sequence[str]) -> ValueTable:
    """Create the zero-initialized computational value table from Table 1."""
    if isinstance(states, (str, bytes)):
        raise TypeError("states must be a sequence of labels, not a string.")

    labels = tuple(states)
    if not labels:
        raise ValueError("states cannot be empty.")

    result: ValueTable = {}
    for index, state in enumerate(labels):
        if not isinstance(state, str):
            raise TypeError(f"states[{index}] must be a string.")
        if not state:
            raise ValueError(f"states[{index}] cannot be empty.")
        if state in result:
            raise ValueError(f"Duplicate state label: {state!r}.")
        result[state] = 0.0

    return result


def as_value_table(
    values: Mapping[str, float],
    *,
    states: Sequence[str] | None = None,
    name: str = "Vhat_T",
) -> ValueTable:
    """Return a finite state-indexed computational value table.

    Values may be negative because Part I defines structural utilities on the
    real line. The returned table is always a new dictionary.
    """
    actual_states = _as_state_labels_from_mapping(values, name=name)

    if states is None:
        state_order = actual_states
    else:
        if isinstance(states, (str, bytes)):
            raise TypeError("states must be a sequence of labels, not a string.")
        state_order = tuple(states)
        if not state_order:
            raise ValueError("states cannot be empty.")
        if len(set(state_order)) != len(state_order):
            raise ValueError("states cannot contain duplicate labels.")
        for index, state in enumerate(state_order):
            if not isinstance(state, str):
                raise TypeError(f"states[{index}] must be a string.")
            if not state:
                raise ValueError(f"states[{index}] cannot be empty.")
        if set(actual_states) != set(state_order):
            raise ValueError(
                f"{name} must contain exactly the states {state_order}."
            )

    result: ValueTable = {}
    for state in state_order:
        raw_value = values[state]
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, Real
        ):
            raise TypeError(f"{name}[{state!r}] must be a real number.")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{name}[{state!r}] must be finite.")
        result[state] = value

    return result


def feasible_executed_actions(
    *,
    a_H_T: int,
    a_AI_T: int,
    n_actions: int,
) -> tuple[int, ...]:
    """Return ``A_exec(s_T)`` from the two proposals at decision instant T.

    The mathematical object is a set. The implementation preserves proposal
    order while removing a duplicate when both agents propose the same action.
    """
    if isinstance(n_actions, (bool, np.bool_)) or not isinstance(
        n_actions, Integral
    ):
        raise TypeError("n_actions must be an integer.")
    action_count = int(n_actions)
    if action_count <= 0:
        raise ValueError("n_actions must be strictly positive.")

    human_action = _as_action_index(
        a_H_T,
        n_actions=action_count,
        name="a_H_T",
    )
    ai_action = _as_action_index(
        a_AI_T,
        n_actions=action_count,
        name="a_AI_T",
    )

    if human_action == ai_action:
        return (human_action,)
    return (human_action, ai_action)


def _validate_execution_origin(
    *,
    a_star_T: int,
    a_H_T: int,
    a_AI_T: int,
    owner_T: str,
) -> None:
    """Require the executed action to equal the selected owner's proposal."""
    selected_proposal = a_H_T if owner_T == "H" else a_AI_T
    if a_star_T != selected_proposal:
        raise ValueError(
            "a_star_T must equal the proposal identified by owner_T; "
            f"owner_T={owner_T!r}, selected proposal={selected_proposal}, "
            f"a_star_T={a_star_T}."
        )


def structural_reward(
    U_H: Mapping[str, ArrayLike],
    U_AI: Mapping[str, ArrayLike],
    *,
    s_T: str,
    a_star_T: int,
    owner_T: str,
) -> float:
    """Return the realized structural reward from Part I Eqs. (38)--(39) [eq:reward_realized, 
    rew(s)]; Part II Eq. (15) [eq:structural_reward].

    ``owner_T`` is required even when the two proposals happen to have the
    same numerical action. This function intentionally has no ``Uhat`` input.
    """
    H_table, AI_table = _validate_matching_utility_tables(U_H, U_AI)
    states = tuple(H_table)
    state = _as_state_label(s_T, states=states, name="s_T")
    owner = _as_owner(owner_T)
    action_index = _as_action_index(
        a_star_T,
        n_actions=H_table[state].size,
        name="a_star_T",
    )

    selected_table = H_table if owner == "H" else AI_table
    return float(selected_table[state][action_index])


def computational_bellman_backup(
    Vhat_T: Mapping[str, float],
    U_H: Mapping[str, ArrayLike],
    U_AI: Mapping[str, ArrayLike],
    P_V: TransitionKernel,
    *,
    s_T: str,
    a_H_T: int,
    a_AI_T: int,
    a_star_T: int,
    owner_T: str,
    gamma: float,
    atol: float = DEFAULT_ATOL,
) -> BellmanUpdate:
    """Apply the structural-reward backup in Part I Eq. (41) [eq:computational_bellman]; 
    Part II Eq. (16) [eq:computational_bellman].

    The execution outcome ``owner_T`` is held fixed while the Bellman operator
    evaluates the feasible actions induced by ``a_H_T`` and ``a_AI_T``. For
    each candidate action, the stage reward is therefore read from the
    structural utility table of that selected owner. The expectation uses the
    corresponding row ``P_V(· | s_T, candidate_action)`` and the complete
    pre-backup table ``Vhat_T``.

    Only the component at ``s_T`` is replaced. All other states are carried
    forward unchanged. The inputs are never modified.
    """
    tolerance = _validated_atol(atol)
    gamma_value = _as_gamma(gamma)
    H_table, AI_table = _validate_matching_utility_tables(U_H, U_AI)

    states = tuple(H_table)
    Vhat_before = as_value_table(Vhat_T, states=states, name="Vhat_T")
    state = _as_state_label(s_T, states=states, name="s_T")
    owner = _as_owner(owner_T)
    n_actions = int(H_table[state].size)

    human_action = _as_action_index(
        a_H_T,
        n_actions=n_actions,
        name="a_H_T",
    )
    ai_action = _as_action_index(
        a_AI_T,
        n_actions=n_actions,
        name="a_AI_T",
    )
    executed_action = _as_action_index(
        a_star_T,
        n_actions=n_actions,
        name="a_star_T",
    )
    _validate_execution_origin(
        a_star_T=executed_action,
        a_H_T=human_action,
        a_AI_T=ai_action,
        owner_T=owner,
    )

    candidate_actions = feasible_executed_actions(
        a_H_T=human_action,
        a_AI_T=ai_action,
        n_actions=n_actions,
    )
    selected_utilities = H_table if owner == "H" else AI_table

    candidate_values: list[float] = []
    for candidate_action in candidate_actions:
        state_probabilities = transition_distribution(
            P_V,
            s_T=state,
            a_star_T=candidate_action,
            states=states,
            n_actions=n_actions,
            atol=tolerance,
        )
        expected_next_value = float(
            sum(
                state_probabilities[next_state] * Vhat_before[next_state]
                for next_state in states
            )
        )
        candidate_value = (
            float(selected_utilities[state][candidate_action])
            + gamma_value * expected_next_value
        )
        if not np.isfinite(candidate_value):
            raise ArithmeticError(
                "The computational Bellman candidate value is not finite."
            )
        candidate_values.append(candidate_value)

    # ``candidate_actions`` is nonempty because it is constructed from the two
    # validated proposals. Built-in max preserves the exact scalar semantics
    # of Equation (41) without introducing an unreported tie-breaking action.
    backed_up_value = float(max(candidate_values))

    Vhat_T_plus_1 = Vhat_before.copy()
    Vhat_T_plus_1[state] = backed_up_value

    reward_structural_T = float(selected_utilities[state][executed_action])
    Vhat_coal_T = float(Vhat_T_plus_1[state])

    return BellmanUpdate(
        reward_structural_T=reward_structural_T,
        Vhat_T_plus_1=Vhat_T_plus_1,
        Vhat_coal_T=Vhat_coal_T,
    )


__all__ = [
    "BellmanUpdate",
    "OWNER_VALUES",
    "StructuralUtilityTable",
    "StructuralUtilityVector",
    "ValueTable",
    "as_structural_utility_table",
    "as_value_table",
    "computational_bellman_backup",
    "feasible_executed_actions",
    "initialize_value_table",
    "structural_reward",
]
