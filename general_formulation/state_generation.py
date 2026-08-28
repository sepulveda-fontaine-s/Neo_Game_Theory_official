'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''

"""Finite-state Virtual Nature kernels and state sampling.

Scientific basis
----------------

- Definition 2: Virtual Nature is represented by a transition function from
  state--action pairs to probability distributions over successor states.
- Definition 3: the transition kernel is denoted ``P_V(· | s, a)``.
- Section 5.3(a), Implementation: when ``random_states=False``, the successor
  state follows the action-dependent kernel ``P(· | s_T, a_star_T)`` and
  ``pA``/``pB`` are not applicable.
- Section 5.3(a), Implementation: when ``random_states=True``, states are
  generated exogenously from ``(pA, pB)``, with ``pB = 1 - pA``, independently
  of the current state and executed action.
- Table 1: the initial state is sampled from ``(pA, pB)`` in the exogenous
  specification or is initialized explicitly (``A`` in the base case) in the
  endogenous specification.
- Part I Eq. (41); Part II Eq. (16): the same validated Virtual Nature kernel is later used by the
  computational Bellman backup.

Part I does not prescribe a numbered numerical action-dependent kernel. The
``build_binary_stay_switch_kernel`` helper is therefore an explicit
implementation constructor: its switching probabilities must be supplied by
configuration rather than being hidden as unlabelled defaults.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Mapping, Sequence, TypeAlias

import numpy as np

from .numerics import DEFAULT_ATOL, as_probability


StateDistribution: TypeAlias = dict[str, float]
TransitionKernel: TypeAlias = dict[str, dict[int, StateDistribution]]
BinaryKernelKey: TypeAlias = tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]


class StateGenerationError(ValueError):
    """Raised when a state-generation specification is internally invalid."""


def _validated_atol(atol: float) -> float:
    """Return a finite, nonnegative absolute tolerance."""
    value = float(atol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("atol must be a finite, nonnegative number.")
    return value


def _as_positive_integer(value: int, *, name: str) -> int:
    """Return a strictly positive integer and reject booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return integer


def _as_boolean(value: bool, *, name: str) -> bool:
    """Return a strict Boolean flag."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be Boolean.")
    return bool(value)


def _as_state_labels(states: Sequence[str]) -> tuple[str, ...]:
    """Return a nonempty ordered tuple of unique state labels."""
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


def _as_state_label(value: str, *, states: tuple[str, ...], name: str) -> str:
    """Return one state label on the declared support."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string state label.")
    if value not in states:
        raise StateGenerationError(
            f"{name}={value!r} is outside the state support {states}."
        )
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


def _as_rng(rng: np.random.Generator) -> np.random.Generator:
    """Return a NumPy Generator used for reproducible state sampling."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    return rng


def as_state_distribution(
    probabilities: Mapping[str, float],
    *,
    states: Sequence[str],
    name: str = "state_distribution",
    atol: float = DEFAULT_ATOL,
) -> StateDistribution:
    """Return a strict probability distribution on a declared state support.

    The mapping must contain every declared state exactly once. Arbitrary
    positive weights are not silently normalized. Only unit-sum drift within
    ``atol`` is corrected, and the input mapping is never modified.
    """
    tolerance = _validated_atol(atol)
    state_labels = _as_state_labels(states)

    if not isinstance(probabilities, Mapping):
        raise TypeError(f"{name} must be a mapping from states to probabilities.")

    actual_states = set(probabilities.keys())
    expected_states = set(state_labels)
    if actual_states != expected_states:
        missing = tuple(state for state in state_labels if state not in actual_states)
        extra = tuple(state for state in probabilities if state not in expected_states)
        raise StateGenerationError(
            f"{name} must contain exactly the states {state_labels}; "
            f"missing={missing}, extra={extra}."
        )

    result = {
        state: as_probability(
            probabilities[state],
            name=f"{name}[{state!r}]",
            atol=tolerance,
        )
        for state in state_labels
    }

    total = float(sum(result.values()))
    if not np.isclose(total, 1.0, rtol=0.0, atol=tolerance):
        raise StateGenerationError(
            f"{name} must sum to 1 within atol={tolerance:g}; "
            f"received {total!r}."
        )

    if total != 1.0:
        result = {state: probability / total for state, probability in result.items()}

    return result


def exogenous_binary_state_distribution(
    *,
    pA: float,
    pB: float | None = None,
    atol: float = DEFAULT_ATOL,
) -> StateDistribution:
    """Return the Scenario-1 exogenous distribution ``(pA, pB)``.

    When ``pB`` is omitted it is computed as ``1 - pA``. When supplied, it
    must satisfy the Part-I identity ``pB = 1 - pA`` within ``atol``.
    """
    tolerance = _validated_atol(atol)
    pA_value = as_probability(pA, name="pA", atol=tolerance)

    if pB is None:
        pB_value = 1.0 - pA_value
    else:
        pB_value = as_probability(pB, name="pB", atol=tolerance)
        if not np.isclose(
            pB_value,
            1.0 - pA_value,
            rtol=0.0,
            atol=tolerance,
        ):
            raise StateGenerationError(
                "The exogenous specification requires pB = 1 - pA."
            )

    return as_state_distribution(
        {"A": pA_value, "B": pB_value},
        states=("A", "B"),
        name="exogenous_state_distribution",
        atol=tolerance,
    )


def as_transition_kernel(
    kernel: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    states: Sequence[str],
    n_actions: int,
    name: str = "P_V",
    atol: float = DEFAULT_ATOL,
) -> TransitionKernel:
    """Return a complete finite transition kernel ``P_V(· | s, a)``.

    Every current state must define one successor-state distribution for every
    action index ``0, ..., n_actions - 1``. Each row is copied and validated on
    the same successor-state support.
    """
    state_labels = _as_state_labels(states)
    action_count = _as_positive_integer(n_actions, name="n_actions")
    tolerance = _validated_atol(atol)

    if not isinstance(kernel, Mapping):
        raise TypeError(f"{name} must be a mapping indexed by current state.")

    actual_states = set(kernel.keys())
    expected_states = set(state_labels)
    if actual_states != expected_states:
        missing = tuple(state for state in state_labels if state not in actual_states)
        extra = tuple(state for state in kernel if state not in expected_states)
        raise StateGenerationError(
            f"{name} must contain exactly the current states {state_labels}; "
            f"missing={missing}, extra={extra}."
        )

    expected_actions = set(range(action_count))
    validated: TransitionKernel = {}

    for state in state_labels:
        action_rows = kernel[state]
        if not isinstance(action_rows, Mapping):
            raise TypeError(f"{name}[{state!r}] must map actions to distributions.")

        raw_action_keys = tuple(action_rows.keys())
        for action_key in raw_action_keys:
            if isinstance(action_key, (bool, np.bool_)) or not isinstance(
                action_key, Integral
            ):
                raise TypeError(
                    f"Every action key in {name}[{state!r}] must be an integer."
                )

        actual_actions = {int(action_key) for action_key in raw_action_keys}
        if actual_actions != expected_actions or len(raw_action_keys) != action_count:
            missing_actions = tuple(
                action for action in range(action_count) if action not in actual_actions
            )
            extra_actions = tuple(
                action for action in actual_actions if action not in expected_actions
            )
            raise StateGenerationError(
                f"{name}[{state!r}] must contain exactly action indices "
                f"0..{action_count - 1}; missing={missing_actions}, "
                f"extra={extra_actions}."
            )

        validated[state] = {
            action: as_state_distribution(
                action_rows[action],
                states=state_labels,
                name=f"{name}[{state!r}][{action}]",
                atol=tolerance,
            )
            for action in range(action_count)
        }

    return validated


def build_exogenous_transition_kernel(
    *,
    pA: float,
    pB: float | None = None,
    n_actions: int = 2,
    atol: float = DEFAULT_ATOL,
) -> TransitionKernel:
    """Build the action-independent kernel used when ``random_states=True``.

    Every row equals ``(pA, pB)``. Consequently, the successor distribution is
    independent of both the current state and the executed action, exactly as
    required by Section 5.3(a) and Algorithm 1.
    """
    action_count = _as_positive_integer(n_actions, name="n_actions")
    distribution = exogenous_binary_state_distribution(
        pA=pA,
        pB=pB,
        atol=atol,
    )

    kernel: TransitionKernel = {
        state: {
            action: distribution.copy()
            for action in range(action_count)
        }
        for state in ("A", "B")
    }

    return as_transition_kernel(
        kernel,
        states=("A", "B"),
        n_actions=action_count,
        name="P_V_exogenous",
        atol=atol,
    )


def build_binary_stay_switch_kernel(
    *,
    p_switch_a0: float,
    p_switch_a1: float,
    atol: float = DEFAULT_ATOL,
) -> TransitionKernel:
    """Build a transparent two-state, two-action stay/switch kernel.

    ``p_switch_a0`` and ``p_switch_a1`` are the probabilities of moving to the
    other state under actions ``a0`` and ``a1``. No numerical defaults are
    supplied because Part I does not assign a numbered base-case kernel.
    """
    tolerance = _validated_atol(atol)
    switch_a0 = as_probability(
        p_switch_a0,
        name="p_switch_a0",
        atol=tolerance,
    )
    switch_a1 = as_probability(
        p_switch_a1,
        name="p_switch_a1",
        atol=tolerance,
    )

    kernel: TransitionKernel = {
        "A": {
            0: {"A": 1.0 - switch_a0, "B": switch_a0},
            1: {"A": 1.0 - switch_a1, "B": switch_a1},
        },
        "B": {
            0: {"A": switch_a0, "B": 1.0 - switch_a0},
            1: {"A": switch_a1, "B": 1.0 - switch_a1},
        },
    }

    return as_transition_kernel(
        kernel,
        states=("A", "B"),
        n_actions=2,
        name="P_V_action_dependent",
        atol=tolerance,
    )


def resolve_virtual_nature_kernel(
    *,
    random_states: bool,
    pA: float | None,
    pB: float | None,
    action_dependent_kernel: (
        Mapping[str, Mapping[int, Mapping[str, float]]] | None
    ),
    states: Sequence[str] = ("A", "B"),
    n_actions: int = 2,
    atol: float = DEFAULT_ATOL,
) -> TransitionKernel:
    """Resolve exactly one of the two Part-I state-generation mechanisms.

    Exogenous mode requires ``pA`` and ``pB`` and rejects an action-dependent
    kernel. Endogenous mode requires the kernel and rejects ``pA``/``pB`` so
    non-applicable parameters cannot silently enter configuration IDs or CSVs.
    """
    exogenous = _as_boolean(random_states, name="random_states")
    state_labels = _as_state_labels(states)
    action_count = _as_positive_integer(n_actions, name="n_actions")

    if exogenous:
        if state_labels != ("A", "B"):
            raise StateGenerationError(
                "The pA/pB exogenous specification requires states ('A', 'B')."
            )
        if pA is None or pB is None:
            raise StateGenerationError(
                "pA and pB are required when random_states=True."
            )
        if action_dependent_kernel is not None:
            raise StateGenerationError(
                "action_dependent_kernel must be None when random_states=True."
            )
        return build_exogenous_transition_kernel(
            pA=pA,
            pB=pB,
            n_actions=action_count,
            atol=atol,
        )

    if pA is not None or pB is not None:
        raise StateGenerationError(
            "pA and pB must be None when random_states=False."
        )
    if action_dependent_kernel is None:
        raise StateGenerationError(
            "action_dependent_kernel is required when random_states=False."
        )

    return as_transition_kernel(
        action_dependent_kernel,
        states=state_labels,
        n_actions=action_count,
        name="P_V_action_dependent",
        atol=atol,
    )


def _binary_kernel_key_validated(
    P_V: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> BinaryKernelKey:
    """Freeze an already validated ``A/B x a0/a1`` kernel as a hashable key."""
    return (
        (
            (float(P_V["A"][0]["A"]), float(P_V["A"][0]["B"])),
            (float(P_V["A"][1]["A"]), float(P_V["A"][1]["B"])),
        ),
        (
            (float(P_V["B"][0]["A"]), float(P_V["B"][0]["B"])),
            (float(P_V["B"][1]["A"]), float(P_V["B"][1]["B"])),
        ),
    )


@lru_cache(maxsize=32)
def _binary_kernel_array_validated(
    kernel_key: BinaryKernelKey,
) -> np.ndarray:
    """Return a cached read-only binary Virtual-Nature kernel array.

    Only the distinct validated kernels are materialized: six exogenous rows
    in the active grid and one common endogenous kernel.
    """
    array = np.asarray(kernel_key, dtype=np.float64)
    if array.shape != (2, 2, 2):
        raise RuntimeError("The validated binary kernel must have shape (2, 2, 2).")
    array.setflags(write=False)
    return array


def transition_distribution(
    P_V: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    s_T: str,
    a_star_T: int,
    states: Sequence[str] = ("A", "B"),
    n_actions: int = 2,
    atol: float = DEFAULT_ATOL,
) -> StateDistribution:
    """Return ``P_V(· | s_T, a_star_T)`` as a defensive copy."""
    state_labels = _as_state_labels(states)
    action_count = _as_positive_integer(n_actions, name="n_actions")
    current_state = _as_state_label(s_T, states=state_labels, name="s_T")
    action_index = _as_action_index(
        a_star_T,
        n_actions=action_count,
        name="a_star_T",
    )

    if current_state not in P_V:
        raise StateGenerationError(f"P_V has no row for s_T={current_state!r}.")
    if action_index not in P_V[current_state]:
        raise StateGenerationError(
            f"P_V[{current_state!r}] has no row for action {action_index}."
        )

    return as_state_distribution(
        P_V[current_state][action_index],
        states=state_labels,
        name=f"P_V[{current_state!r}][{action_index}]",
        atol=atol,
    )


def sample_state(
    probabilities: Mapping[str, float],
    *,
    states: Sequence[str],
    rng: np.random.Generator,
    name: str = "state_distribution",
    atol: float = DEFAULT_ATOL,
) -> str:
    """Sample one state from a validated finite distribution."""
    state_labels = _as_state_labels(states)
    distribution = as_state_distribution(
        probabilities,
        states=state_labels,
        name=name,
        atol=atol,
    )
    generator = _as_rng(rng)

    draw = float(generator.random())
    cumulative = 0.0
    for state in state_labels[:-1]:
        cumulative += distribution[state]
        if draw < cumulative:
            return state

    # The final state absorbs any harmless cumulative floating-point drift.
    return state_labels[-1]



def _sample_binary_next_state_validated(
    probability_A: float,
    *,
    rng: np.random.Generator,
    atol: float = DEFAULT_ATOL,
) -> int:
    """Sample ``A=0`` or ``B=1`` from an already validated binary row.

    Only a cheap scalar range guard remains in the production path.
    """
    pA = float(probability_A)
    if pA < -atol or pA > 1.0 + atol:
        raise ArithmeticError(f"Validated transition probability left [0,1]: {pA!r}.")
    if pA < 0.0:
        pA = 0.0
    elif pA > 1.0:
        pA = 1.0
    return 0 if float(rng.random()) < pA else 1

def sample_next_state(
    P_V: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    s_T: str,
    a_star_T: int,
    rng: np.random.Generator,
    states: Sequence[str] = ("A", "B"),
    n_actions: int = 2,
    atol: float = DEFAULT_ATOL,
) -> str:
    """Sample ``s_next`` from ``P_V(· | s_T, a_star_T)``.

    The same function is valid for both mechanisms. In exogenous mode every
    kernel row is identical, so the sampled state is independent of
    ``s_T`` and ``a_star_T``. In endogenous mode the selected row depends on
    both, as required by Part I.

    ``P_V`` is a hot-loop object and must be validated once at the setup
    boundary with ``resolve_virtual_nature_kernel`` or ``as_transition_kernel``.
    This sampler deliberately avoids revalidating the complete probability row
    at every decision instant.
    """
    del atol  # Validation belongs to the setup boundary, not the hot loop.

    state_labels = _as_state_labels(states)
    action_count = _as_positive_integer(n_actions, name="n_actions")
    current_state = _as_state_label(s_T, states=state_labels, name="s_T")
    action_index = _as_action_index(
        a_star_T,
        n_actions=action_count,
        name="a_star_T",
    )
    generator = _as_rng(rng)

    if current_state not in P_V:
        raise StateGenerationError(f"P_V has no row for s_T={current_state!r}.")
    if action_index not in P_V[current_state]:
        raise StateGenerationError(
            f"P_V[{current_state!r}] has no row for action {action_index}."
        )

    distribution = P_V[current_state][action_index]
    draw = float(generator.random())
    cumulative = 0.0

    try:
        for state in state_labels[:-1]:
            cumulative += float(distribution[state])
            if draw < cumulative:
                return state
        # The last state absorbs harmless cumulative floating-point drift in a
        # kernel that was validated once before simulation.
        return state_labels[-1]
    except KeyError as error:
        raise StateGenerationError(
            "P_V contains an incomplete successor-state row; validate the "
            "kernel before simulation."
        ) from error


def initialize_state(
    *,
    random_states: bool,
    P_V: Mapping[str, Mapping[int, Mapping[str, float]]],
    rng: np.random.Generator,
    initial_state: str | None,
    states: Sequence[str] = ("A", "B"),
    n_actions: int = 2,
    atol: float = DEFAULT_ATOL,
) -> str:
    """Initialize ``s_0`` according to Table 1 in Part I.

    In exogenous mode, ``initial_state`` must be ``None`` and ``s_0`` is
    sampled from the common kernel row. In endogenous mode, an explicit state
    such as the base-case value ``'A'`` is required and returned unchanged.
    """
    exogenous = _as_boolean(random_states, name="random_states")
    state_labels = _as_state_labels(states)
    action_count = _as_positive_integer(n_actions, name="n_actions")
    generator = _as_rng(rng)

    if exogenous:
        if initial_state is not None:
            raise StateGenerationError(
                "initial_state must be None when random_states=True; "
                "s_0 is sampled from (pA, pB)."
            )

        # Every row is identical in a valid exogenous kernel; use the first
        # declared state and action only to retrieve that common distribution.
        return sample_next_state(
            P_V,
            s_T=state_labels[0],
            a_star_T=0,
            rng=generator,
            states=state_labels,
            n_actions=action_count,
            atol=atol,
        )

    if initial_state is None:
        raise StateGenerationError(
            "initial_state is required when random_states=False."
        )

    # Validate the kernel once here as part of the setup boundary. Sampling
    # later can use the same validated object returned by
    # resolve_virtual_nature_kernel.
    as_transition_kernel(
        P_V,
        states=state_labels,
        n_actions=action_count,
        name="P_V",
        atol=atol,
    )
    return _as_state_label(
        initial_state,
        states=state_labels,
        name="initial_state",
    )


__all__ = [
    "StateDistribution",
    "StateGenerationError",
    "TransitionKernel",
    "as_state_distribution",
    "as_transition_kernel",
    "build_binary_stay_switch_kernel",
    "build_exogenous_transition_kernel",
    "exogenous_binary_state_distribution",
    "initialize_state",
    "resolve_virtual_nature_kernel",
    "sample_next_state",
    "sample_state",
    "transition_distribution",
]
