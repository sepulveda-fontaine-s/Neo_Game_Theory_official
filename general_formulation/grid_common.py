'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario-independent grid construction for binary Neo-Game experiments.

This module contains only mechanisms shared by Scenarios 1, 2 and 3:
validation and freezing of common model objects, canonical learning-rate grid
construction, threshold/beta/gamma validation, reusable state specifications,
and fast deterministic identifier assembly.

Scenario-specific modules remain responsible for their own run dataclasses,
execution branches, delegation semantics, provenance fields, and concrete
configuration identifiers.  In particular, this module contains no lambda
convention and no scenario-specific delegation rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
from numpy.typing import ArrayLike

from general_formulation.bellman import as_structural_utility_table
from general_formulation.identifiers import (
    _build_config_id_from_canonical_fragments,
    _canonical_json_value,
    build_winner_group_id,
    safe_token,
)
from general_formulation.learning_rates import (
    ETA_KINDS,
    build_eta_label,
    validate_eta_spec,
)
from general_formulation.numerics import (
    DEFAULT_ATOL,
    as_probability,
    as_probability_vector,
    uniform_probability_vector,
)
from general_formulation.state_generation import (
    BinaryKernelKey,
    _binary_kernel_key_validated,
    as_transition_kernel,
    resolve_virtual_nature_kernel,
)
from general_formulation.validation import (
    validate_probability_complement,
    validate_thresholds,
)

BINARY_STATES = ("A", "B")
BINARY_N_ACTIONS = 2

DEFAULT_HORIZON_GRID = (200, 1000, 5000, 15000, 30000, 50000)
DEFAULT_ALPHA_AGREE_GRID = (0.20, 0.35, 0.49, 0.70)
DEFAULT_ALPHA_DISAGREE_GRID = (0.35, 0.51, 0.85)
DEFAULT_BETA_GRID = (0.02, 0.05, 0.10)
DEFAULT_GAMMA_GRID = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_ETA0_GRID = (0.025, 0.05, 0.08, 0.10, 0.15)
DEFAULT_DECAY_C_GRID = (0.0005, 0.001, 0.002)
DEFAULT_STATE_PROBABILITY_GRID = (
    (0.70, 0.30),
    (0.90, 0.10),
    (0.01, 0.99),
    (0.30, 0.70),
    (0.10, 0.90),
    (0.99, 0.01),
)
DEFAULT_SEEDS = (42,)


@dataclass(frozen=True, slots=True)
class EtaGridSpec:
    """One canonical learning-rate specification in a parameter grid."""

    eta_kind: str
    eta_label: str
    eta0: float | None
    c: float | None


@dataclass(frozen=True, slots=True)
class BinaryStateSpec:
    """One prevalidated state-generation specification for a binary run."""

    random_states: bool
    pA: float | None
    pB: float | None
    initial_state: str | None
    action_dependent_kernel: (
        Mapping[str, Mapping[int, Mapping[str, float]]] | None
    )
    P_V_key: BinaryKernelKey
    transition_spec_id: str


@dataclass(frozen=True, slots=True)
class PreparedBinaryGrid:
    """Common immutable objects validated once before exhaustive iteration."""

    H: int
    seeds: tuple[int, ...]
    utility_spec_id: str
    transition_spec_id: str
    endogenous_initial_state: str
    p_H_init: Mapping[str, tuple[float, ...]]
    p_AI_init: Mapping[str, tuple[float, ...]]
    U_H: Mapping[str, tuple[float, ...]]
    U_AI: Mapping[str, tuple[float, ...]]
    eta_specs: tuple[EtaGridSpec, ...]
    threshold_pairs: tuple[tuple[float, float], ...]
    beta_values: tuple[float, ...]
    gamma_values: tuple[float, ...]
    exogenous_state_specs: tuple[BinaryStateSpec, ...]
    endogenous_state_spec: BinaryStateSpec


def as_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def as_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def as_nonempty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty.")
    return stripped


def as_optional_string(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    return as_nonempty_string(value, name=name)


def as_gamma(gamma: float) -> float:
    if isinstance(gamma, (bool, np.bool_)):
        raise TypeError("gamma must be a real number, not Boolean.")
    value = float(gamma)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("gamma must satisfy 0 < gamma < 1.")
    return value


def validate_seed_sequence(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if not values:
        raise ValueError("seeds must contain at least one seed.")
    return tuple(
        as_nonnegative_integer(seed, name=f"seeds[{index}]")
        for index, seed in enumerate(values)
    )


def freeze_probability_table(
    values: Mapping[str, ArrayLike] | None,
    *,
    name: str,
    states: Sequence[str] = BINARY_STATES,
    n_actions: int = BINARY_N_ACTIONS,
    uniform_if_none: bool,
    atol: float = DEFAULT_ATOL,
) -> Mapping[str, tuple[float, ...]]:
    state_labels = tuple(states)
    action_count = as_positive_integer(n_actions, name="n_actions")
    if values is None:
        if not uniform_if_none:
            raise ValueError(f"{name} must be supplied explicitly.")
        table = {
            state: uniform_probability_vector(action_count)
            for state in state_labels
        }
    else:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a state-indexed mapping.")
        if set(values) != set(state_labels):
            raise ValueError(
                f"{name} must contain exactly the states {state_labels}."
            )
        table = {
            state: as_probability_vector(
                values[state],
                name=f"{name}[{state!r}]",
                atol=atol,
            )
            for state in state_labels
        }

    frozen = {
        state: tuple(float(value) for value in table[state])
        for state in state_labels
    }
    if any(len(vector) != action_count for vector in frozen.values()):
        raise ValueError(f"{name} must use exactly {action_count} actions.")
    return MappingProxyType(frozen)


def freeze_structural_utility_table(
    values: Mapping[str, ArrayLike],
    *,
    name: str,
    states: Sequence[str] = BINARY_STATES,
    n_actions: int = BINARY_N_ACTIONS,
) -> Mapping[str, tuple[float, ...]]:
    state_labels = tuple(states)
    action_count = as_positive_integer(n_actions, name="n_actions")
    table = as_structural_utility_table(values, name=name)
    if tuple(table) != state_labels:
        if set(table) != set(state_labels):
            raise ValueError(
                f"{name} must contain exactly the states {state_labels}."
            )
        table = {state: table[state] for state in state_labels}
    frozen = {
        state: tuple(float(value) for value in table[state])
        for state in state_labels
    }
    if any(len(vector) != action_count for vector in frozen.values()):
        raise ValueError(f"{name} must use exactly {action_count} actions.")
    return MappingProxyType(frozen)


def freeze_transition_kernel(
    kernel: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    states: Sequence[str] = BINARY_STATES,
    n_actions: int = BINARY_N_ACTIONS,
    atol: float = DEFAULT_ATOL,
) -> Mapping[str, Mapping[int, Mapping[str, float]]]:
    state_labels = tuple(states)
    action_count = as_positive_integer(n_actions, name="n_actions")
    validated = as_transition_kernel(
        kernel,
        states=state_labels,
        n_actions=action_count,
        name="action_dependent_kernel",
        atol=atol,
    )
    outer: dict[str, Mapping[int, Mapping[str, float]]] = {}
    for state in state_labels:
        action_rows: dict[int, Mapping[str, float]] = {}
        for action in range(action_count):
            action_rows[action] = MappingProxyType(
                {
                    successor: float(validated[state][action][successor])
                    for successor in state_labels
                }
            )
        outer[state] = MappingProxyType(action_rows)
    return MappingProxyType(outer)


def iter_eta_grid_specs(
    *,
    eta0_grid: Sequence[float] = DEFAULT_ETA0_GRID,
    decay_c_grid: Sequence[float] = DEFAULT_DECAY_C_GRID,
) -> Iterator[EtaGridSpec]:
    """Yield constant, global-decay, and exact-empirical specifications."""
    for eta0 in eta0_grid:
        kind, eta0_value, c_value = validate_eta_spec(
            eta_kind="constant", eta0=eta0, c=None
        )
        yield EtaGridSpec(
            eta_kind=kind,
            eta_label=build_eta_label(
                eta_kind=kind, eta0=eta0_value, c=c_value
            ),
            eta0=eta0_value,
            c=c_value,
        )
    for eta0, c in product(eta0_grid, decay_c_grid):
        kind, eta0_value, c_value = validate_eta_spec(
            eta_kind="global_decay", eta0=eta0, c=c
        )
        yield EtaGridSpec(
            eta_kind=kind,
            eta_label=build_eta_label(
                eta_kind=kind, eta0=eta0_value, c=c_value
            ),
            eta0=eta0_value,
            c=c_value,
        )
    kind, eta0_value, c_value = validate_eta_spec(
        eta_kind="exact_empirical", eta0=None, c=None
    )
    yield EtaGridSpec(
        eta_kind=kind,
        eta_label=build_eta_label(
            eta_kind=kind, eta0=eta0_value, c=c_value
        ),
        eta0=eta0_value,
        c=c_value,
    )


def validated_eta_grid_specs(
    *,
    eta0_grid: Sequence[float] = DEFAULT_ETA0_GRID,
    decay_c_grid: Sequence[float] = DEFAULT_DECAY_C_GRID,
) -> tuple[EtaGridSpec, ...]:
    specs = tuple(
        iter_eta_grid_specs(eta0_grid=eta0_grid, decay_c_grid=decay_c_grid)
    )
    if {spec.eta_kind for spec in specs} != set(ETA_KINDS):
        raise RuntimeError("The eta grid does not cover all canonical schedules.")
    return specs


def validated_threshold_pairs(
    *,
    alpha_agree_grid: Sequence[float] = DEFAULT_ALPHA_AGREE_GRID,
    alpha_disagree_grid: Sequence[float] = DEFAULT_ALPHA_DISAGREE_GRID,
    atol: float = DEFAULT_ATOL,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        validate_thresholds(a, d, atol=atol)
        for a in alpha_agree_grid
        for d in alpha_disagree_grid
        if a < d
    )


def validated_beta_values(
    beta_grid: Sequence[float] = DEFAULT_BETA_GRID,
    *,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, ...]:
    values = tuple(
        as_probability(beta, name="beta", atol=atol) for beta in beta_grid
    )
    if any(beta <= 0.0 for beta in values):
        raise ValueError("Every beta grid value must lie in (0,1].")
    return values


def validated_gamma_values(
    gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
) -> tuple[float, ...]:
    return tuple(as_gamma(gamma) for gamma in gamma_grid)


def build_winner_group_id_with_branch(
    *,
    scenario: str,
    H: int,
    eta_kind: str,
    random_states: bool,
    execution_branch: str | None = None,
) -> str:
    base = build_winner_group_id(
        scenario=scenario,
        H=H,
        eta_kind=eta_kind,
        random_states=random_states,
    )
    if execution_branch is None:
        return base
    branch = safe_token(as_nonempty_string(execution_branch, name="execution_branch"))
    return f"{base}__branch-{branch}"


def canonical_identifier_fragments(
    fields: Mapping[str, Any],
) -> Mapping[str, str]:
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a non-empty mapping.")
    return MappingProxyType(
        {str(key): _canonical_json_value(value) for key, value in fields.items()}
    )


def build_config_id_from_fragments(
    base_fragments: Mapping[str, str],
    varying_fields: Mapping[str, Any],
) -> str:
    fragments = dict(base_fragments)
    fragments.update(
        {str(key): _canonical_json_value(value) for key, value in varying_fields.items()}
    )
    return _build_config_id_from_canonical_fragments(fragments)


def prepare_binary_grid(
    *,
    H: int,
    U_H: Mapping[str, ArrayLike],
    U_AI: Mapping[str, ArrayLike],
    utility_spec_id: str,
    action_dependent_kernel: Mapping[str, Mapping[int, Mapping[str, float]]],
    transition_spec_id: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    p_H_init: Mapping[str, ArrayLike] | None = None,
    p_AI_init: Mapping[str, ArrayLike] | None = None,
    endogenous_initial_state: str = "A",
    horizon_grid: Sequence[int] = DEFAULT_HORIZON_GRID,
    alpha_agree_grid: Sequence[float] = DEFAULT_ALPHA_AGREE_GRID,
    alpha_disagree_grid: Sequence[float] = DEFAULT_ALPHA_DISAGREE_GRID,
    beta_grid: Sequence[float] = DEFAULT_BETA_GRID,
    gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
    eta0_grid: Sequence[float] = DEFAULT_ETA0_GRID,
    decay_c_grid: Sequence[float] = DEFAULT_DECAY_C_GRID,
    state_probability_grid: Sequence[tuple[float, float]] = DEFAULT_STATE_PROBABILITY_GRID,
    states: Sequence[str] = BINARY_STATES,
    n_actions: int = BINARY_N_ACTIONS,
    atol: float = DEFAULT_ATOL,
) -> PreparedBinaryGrid:
    """Validate every common invariant once for one exhaustive grid."""
    state_labels = tuple(states)
    action_count = as_positive_integer(n_actions, name="n_actions")
    if state_labels != BINARY_STATES or action_count != BINARY_N_ACTIONS:
        raise ValueError(
            "prepare_binary_grid currently supports exactly states ('A','B') "
            "and two actions."
        )

    horizon = as_positive_integer(H, name="H")
    allowed_horizons = tuple(
        as_positive_integer(value, name=f"horizon_grid[{index}]")
        for index, value in enumerate(horizon_grid)
    )
    # Development horizons are valid too; the grid list is descriptive rather
    # than a hard execution restriction.
    del allowed_horizons

    seed_values = validate_seed_sequence(seeds)
    utility_id = as_nonempty_string(utility_spec_id, name="utility_spec_id")
    transition_id = as_nonempty_string(
        transition_spec_id, name="transition_spec_id"
    )
    if endogenous_initial_state not in state_labels:
        raise ValueError(
            f"endogenous_initial_state must be one of {state_labels}."
        )

    p_H_frozen = freeze_probability_table(
        p_H_init,
        name="p_H_init",
        states=state_labels,
        n_actions=action_count,
        uniform_if_none=True,
        atol=atol,
    )
    p_AI_frozen = freeze_probability_table(
        p_AI_init,
        name="p_AI_init",
        states=state_labels,
        n_actions=action_count,
        uniform_if_none=True,
        atol=atol,
    )
    U_H_frozen = freeze_structural_utility_table(
        U_H, name="U_H", states=state_labels, n_actions=action_count
    )
    U_AI_frozen = freeze_structural_utility_table(
        U_AI, name="U_AI", states=state_labels, n_actions=action_count
    )
    for state in state_labels:
        if len(U_H_frozen[state]) != len(U_AI_frozen[state]):
            raise ValueError("U_H and U_AI must use the same action support.")

    endogenous_kernel = freeze_transition_kernel(
        action_dependent_kernel,
        states=state_labels,
        n_actions=action_count,
        atol=atol,
    )
    endogenous_kernel_key = _binary_kernel_key_validated(endogenous_kernel)
    endogenous_spec = BinaryStateSpec(
        random_states=False,
        pA=None,
        pB=None,
        initial_state=endogenous_initial_state,
        action_dependent_kernel=endogenous_kernel,
        P_V_key=endogenous_kernel_key,
        transition_spec_id=transition_id,
    )

    exogenous_specs: list[BinaryStateSpec] = []
    for raw_pA, raw_pB in state_probability_grid:
        pA_value, pB_value = validate_probability_complement(
            raw_pA,
            raw_pB,
            probability_name="pA",
            complement_name="pB",
            atol=atol,
        )
        resolved = resolve_virtual_nature_kernel(
            random_states=True,
            pA=pA_value,
            pB=pB_value,
            action_dependent_kernel=None,
            states=state_labels,
            n_actions=action_count,
            atol=atol,
        )
        exogenous_specs.append(
            BinaryStateSpec(
                random_states=True,
                pA=pA_value,
                pB=pB_value,
                initial_state=None,
                action_dependent_kernel=None,
                P_V_key=_binary_kernel_key_validated(resolved),
                transition_spec_id=f"{transition_id}__exogenous",
            )
        )

    return PreparedBinaryGrid(
        H=horizon,
        seeds=seed_values,
        utility_spec_id=utility_id,
        transition_spec_id=transition_id,
        endogenous_initial_state=endogenous_initial_state,
        p_H_init=p_H_frozen,
        p_AI_init=p_AI_frozen,
        U_H=U_H_frozen,
        U_AI=U_AI_frozen,
        eta_specs=validated_eta_grid_specs(
            eta0_grid=eta0_grid, decay_c_grid=decay_c_grid
        ),
        threshold_pairs=validated_threshold_pairs(
            alpha_agree_grid=alpha_agree_grid,
            alpha_disagree_grid=alpha_disagree_grid,
            atol=atol,
        ),
        beta_values=validated_beta_values(beta_grid, atol=atol),
        gamma_values=validated_gamma_values(gamma_grid),
        exogenous_state_specs=tuple(exogenous_specs),
        endogenous_state_spec=endogenous_spec,
    )


__all__ = [
    "BINARY_N_ACTIONS",
    "BINARY_STATES",
    "BinaryStateSpec",
    "DEFAULT_ALPHA_AGREE_GRID",
    "DEFAULT_ALPHA_DISAGREE_GRID",
    "DEFAULT_BETA_GRID",
    "DEFAULT_DECAY_C_GRID",
    "DEFAULT_ETA0_GRID",
    "DEFAULT_GAMMA_GRID",
    "DEFAULT_HORIZON_GRID",
    "DEFAULT_SEEDS",
    "DEFAULT_STATE_PROBABILITY_GRID",
    "EtaGridSpec",
    "PreparedBinaryGrid",
    "as_gamma",
    "as_nonempty_string",
    "as_nonnegative_integer",
    "as_optional_string",
    "as_positive_integer",
    "build_config_id_from_fragments",
    "build_winner_group_id_with_branch",
    "canonical_identifier_fragments",
    "freeze_probability_table",
    "freeze_structural_utility_table",
    "freeze_transition_kernel",
    "iter_eta_grid_specs",
    "prepare_binary_grid",
    "validate_seed_sequence",
    "validated_beta_values",
    "validated_eta_grid_specs",
    "validated_gamma_values",
    "validated_threshold_pairs",
]
