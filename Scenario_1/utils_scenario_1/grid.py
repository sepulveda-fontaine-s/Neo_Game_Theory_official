'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario 1 grid built on scenario-independent common mechanisms.

Only Scenario 1 contracts remain here: the run dataclass, Scenario 1 identifier
payload, winner groups, and the concrete exhaustive iterator. 
Common validation and grid materialization are handled in
``general_formulation.grid_common``, while this module retains only the
Scenario 1-specific run configuration and iterator.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator

from numpy.typing import ArrayLike

from general_formulation.grid_common import (
    BINARY_N_ACTIONS,
    BINARY_STATES,
    DEFAULT_ALPHA_AGREE_GRID,
    DEFAULT_ALPHA_DISAGREE_GRID,
    DEFAULT_BETA_GRID,
    DEFAULT_DECAY_C_GRID,
    DEFAULT_ETA0_GRID,
    DEFAULT_GAMMA_GRID,
    DEFAULT_HORIZON_GRID,
    DEFAULT_SEEDS as COMMON_DEFAULT_SEEDS,
    DEFAULT_STATE_PROBABILITY_GRID,
    EtaGridSpec,
    as_gamma,
    as_nonempty_string,
    as_nonnegative_integer,
    as_positive_integer,
    build_config_id_from_fragments,
    build_winner_group_id_with_branch,
    canonical_identifier_fragments,
    freeze_probability_table,
    freeze_structural_utility_table,
    freeze_transition_kernel,
    iter_eta_grid_specs as iter_common_eta_grid_specs,
    prepare_binary_grid,
)
from general_formulation.identifiers import build_config_id, build_run_id
from general_formulation.learning_rates import build_eta_label, validate_eta_spec
from general_formulation.numerics import DEFAULT_ATOL, as_probability
from general_formulation.state_generation import (
    BinaryKernelKey,
    _binary_kernel_key_validated,
    resolve_virtual_nature_kernel,
    build_binary_stay_switch_kernel,
)
from general_formulation.validation import (
    validate_probability_complement,
    validate_thresholds,
)

SCENARIO_NAME = "scenario1"
STATES = BINARY_STATES
N_ACTIONS = BINARY_N_ACTIONS

HORIZON_GRID = DEFAULT_HORIZON_GRID
ALPHA_AGREE_GRID = DEFAULT_ALPHA_AGREE_GRID
ALPHA_DISAGREE_GRID = DEFAULT_ALPHA_DISAGREE_GRID
BETA_GRID = DEFAULT_BETA_GRID
GAMMA_GRID = DEFAULT_GAMMA_GRID
ETA0_GRID = DEFAULT_ETA0_GRID
DECAY_C_GRID = DEFAULT_DECAY_C_GRID
STATE_PROBABILITY_GRID = DEFAULT_STATE_PROBABILITY_GRID
DEFAULT_SEEDS = COMMON_DEFAULT_SEEDS

# Scenario-specific independent-grid model specification.
# These values intentionally live with the scenario grid, not in config.py.
MODEL_P_H_INIT = {
    "A": (0.99, 0.01),
    "B": (0.01, 0.99),
}
MODEL_P_AI_INIT = {
    "A": (0.01, 0.99),
    "B": (0.99, 0.01),
}
MODEL_UTILITY_SPEC_ID = "opposed_binary_state_preferences_v1"
MODEL_U_H = {
    "A": (1.0, 0.0),
    "B": (0.0, 1.0),
}
MODEL_U_AI = {
    "A": (0.0, 1.0),
    "B": (1.0, 0.0),
}
MODEL_TRANSITION_SPEC_ID = "binary_stay_switch_a0_0p1_a1_0p9_v1"
MODEL_ACTION_DEPENDENT_KERNEL = build_binary_stay_switch_kernel(
    p_switch_a0=0.10,
    p_switch_a1=0.90,
    atol=DEFAULT_ATOL,
)
MODEL_ENDOGENOUS_INITIAL_STATE = "A"
MODEL_SEEDS = DEFAULT_SEEDS


@dataclass(frozen=True, slots=True)
class Scenario1RunConfig:
    """Validated immutable specification of one concrete Scenario 1 run."""

    scenario: str
    H: int
    winner_group_id: str
    config_id: str
    run_id: str
    replicate: int
    seed: int

    alpha_agree: float
    alpha_disagree: float
    eta_kind: str
    eta_label: str
    eta0: float | None
    c: float | None
    beta: float
    gamma: float
    random_states: bool
    pA: float | None
    pB: float | None
    initial_state: str | None
    utility_spec_id: str
    transition_spec_id: str

    p_H_init: Mapping[str, tuple[float, ...]]
    p_AI_init: Mapping[str, tuple[float, ...]]
    U_H: Mapping[str, tuple[float, ...]]
    U_AI: Mapping[str, tuple[float, ...]]
    action_dependent_kernel: (
        Mapping[str, Mapping[int, Mapping[str, float]]] | None
    )
    P_V_key: BinaryKernelKey


def iter_eta_grid_specs() -> Iterator[EtaGridSpec]:
    """Yield the canonical Scenario 1 learning-rate specifications."""
    yield from iter_common_eta_grid_specs(
        eta0_grid=ETA0_GRID,
        decay_c_grid=DECAY_C_GRID,
    )


def build_scenario1_run_config(
    *,
    H: int,
    alpha_agree: float,
    alpha_disagree: float,
    eta_kind: str,
    eta0: float | None,
    c: float | None,
    beta: float,
    gamma: float,
    random_states: bool,
    pA: float | None,
    pB: float | None,
    initial_state: str | None,
    utility_spec_id: str,
    transition_spec_id: str,
    U_H: Mapping[str, ArrayLike],
    U_AI: Mapping[str, ArrayLike],
    action_dependent_kernel: (
        Mapping[str, Mapping[int, Mapping[str, float]]] | None
    ),
    replicate: int,
    seed: int,
    p_H_init: Mapping[str, ArrayLike] | None = None,
    p_AI_init: Mapping[str, ArrayLike] | None = None,
    atol: float = DEFAULT_ATOL,
) -> Scenario1RunConfig:
    """Validate one ad-hoc run and construct stable identifiers."""
    horizon = as_positive_integer(H, name="H")
    replication = as_positive_integer(replicate, name="replicate")
    seed_value = as_nonnegative_integer(seed, name="seed")
    agreement, disagreement = validate_thresholds(
        alpha_agree, alpha_disagree, atol=atol
    )
    eta_kind_value, eta0_value, c_value = validate_eta_spec(
        eta_kind=eta_kind, eta0=eta0, c=c
    )
    eta_label = build_eta_label(
        eta_kind=eta_kind_value, eta0=eta0_value, c=c_value
    )
    beta_value = as_probability(beta, name="beta", atol=atol)
    if beta_value <= 0.0:
        raise ValueError("beta must lie in (0, 1].")
    gamma_value = as_gamma(gamma)

    if not isinstance(random_states, bool):
        raise TypeError("random_states must be boolean.")
    if random_states:
        if pA is None or pB is None:
            raise ValueError("pA and pB are required when random_states=True.")
        pA_value, pB_value = validate_probability_complement(
            pA,
            pB,
            probability_name="pA",
            complement_name="pB",
            atol=atol,
        )
        if initial_state is not None:
            raise ValueError(
                "initial_state must be None when random_states=True; s_0 is sampled."
            )
        if action_dependent_kernel is not None:
            raise ValueError(
                "action_dependent_kernel must be None when random_states=True."
            )
        frozen_kernel = None
    else:
        if pA is not None or pB is not None:
            raise ValueError("pA and pB must be None when random_states=False.")
        pA_value = None
        pB_value = None
        if initial_state not in STATES:
            raise ValueError(
                f"initial_state must be one of {STATES} when random_states=False."
            )
        if action_dependent_kernel is None:
            raise ValueError(
                "action_dependent_kernel is required when random_states=False."
            )
        frozen_kernel = freeze_transition_kernel(
            action_dependent_kernel,
            states=STATES,
            n_actions=N_ACTIONS,
            atol=atol,
        )

    resolved_kernel = resolve_virtual_nature_kernel(
        random_states=random_states,
        pA=pA_value,
        pB=pB_value,
        action_dependent_kernel=frozen_kernel,
        states=STATES,
        n_actions=N_ACTIONS,
        atol=atol,
    )
    P_V_key = _binary_kernel_key_validated(resolved_kernel)

    p_H_frozen = freeze_probability_table(
        p_H_init,
        name="p_H_init",
        states=STATES,
        n_actions=N_ACTIONS,
        uniform_if_none=True,
        atol=atol,
    )
    p_AI_frozen = freeze_probability_table(
        p_AI_init,
        name="p_AI_init",
        states=STATES,
        n_actions=N_ACTIONS,
        uniform_if_none=True,
        atol=atol,
    )
    U_H_frozen = freeze_structural_utility_table(
        U_H, name="U_H", states=STATES, n_actions=N_ACTIONS
    )
    U_AI_frozen = freeze_structural_utility_table(
        U_AI, name="U_AI", states=STATES, n_actions=N_ACTIONS
    )
    utility_id = as_nonempty_string(utility_spec_id, name="utility_spec_id")
    transition_id = as_nonempty_string(
        transition_spec_id, name="transition_spec_id"
    )

    winner_group_id = build_winner_group_id_with_branch(
        scenario=SCENARIO_NAME,
        H=horizon,
        eta_kind=eta_kind_value,
        random_states=random_states,
    )
    config_fields: dict[str, Any] = {
        "H": horizon,
        "alpha_agree": agreement,
        "alpha_disagree": disagreement,
        "eta_kind": eta_kind_value,
        "eta0": eta0_value,
        "c": c_value,
        "beta": beta_value,
        "gamma": gamma_value,
        "random_states": random_states,
        "pA": pA_value,
        "pB": pB_value,
        "initial_state": initial_state,
        "utility_spec_id": utility_id,
        "transition_spec_id": transition_id,
        "p_H_init": p_H_frozen,
        "p_AI_init": p_AI_frozen,
    }
    config_id = build_config_id(config_fields)
    run_id = build_run_id(
        config_id=config_id, replicate=replication, seed=seed_value
    )

    return Scenario1RunConfig(
        scenario=SCENARIO_NAME,
        H=horizon,
        winner_group_id=winner_group_id,
        config_id=config_id,
        run_id=run_id,
        replicate=replication,
        seed=seed_value,
        alpha_agree=agreement,
        alpha_disagree=disagreement,
        eta_kind=eta_kind_value,
        eta_label=eta_label,
        eta0=eta0_value,
        c=c_value,
        beta=beta_value,
        gamma=gamma_value,
        random_states=random_states,
        pA=pA_value,
        pB=pB_value,
        initial_state=initial_state,
        utility_spec_id=utility_id,
        transition_spec_id=transition_id,
        p_H_init=p_H_frozen,
        p_AI_init=p_AI_frozen,
        U_H=U_H_frozen,
        U_AI=U_AI_frozen,
        action_dependent_kernel=frozen_kernel,
        P_V_key=P_V_key,
    )


def _build_prevalidated_grid_config(
    *,
    H: int,
    alpha_agree: float,
    alpha_disagree: float,
    eta_spec: EtaGridSpec,
    beta: float,
    gamma: float,
    random_states: bool,
    pA: float | None,
    pB: float | None,
    initial_state: str | None,
    utility_spec_id: str,
    transition_spec_id: str,
    replicate: int,
    seed: int,
    p_H_init: Mapping[str, tuple[float, ...]],
    p_AI_init: Mapping[str, tuple[float, ...]],
    U_H: Mapping[str, tuple[float, ...]],
    U_AI: Mapping[str, tuple[float, ...]],
    action_dependent_kernel: (
        Mapping[str, Mapping[int, Mapping[str, float]]] | None
    ),
    P_V_key: BinaryKernelKey,
    winner_group_id: str,
    identifier_base_fragments: Mapping[str, str],
) -> Scenario1RunConfig:
    config_id = build_config_id_from_fragments(
        identifier_base_fragments,
        {
            "alpha_agree": alpha_agree,
            "alpha_disagree": alpha_disagree,
            "eta_kind": eta_spec.eta_kind,
            "eta0": eta_spec.eta0,
            "c": eta_spec.c,
            "beta": beta,
            "gamma": gamma,
            "random_states": random_states,
            "pA": pA,
            "pB": pB,
            "initial_state": initial_state,
            "transition_spec_id": transition_spec_id,
        },
    )
    run_id = build_run_id(config_id=config_id, replicate=replicate, seed=seed)
    return Scenario1RunConfig(
        scenario=SCENARIO_NAME,
        H=H,
        winner_group_id=winner_group_id,
        config_id=config_id,
        run_id=run_id,
        replicate=replicate,
        seed=seed,
        alpha_agree=alpha_agree,
        alpha_disagree=alpha_disagree,
        eta_kind=eta_spec.eta_kind,
        eta_label=eta_spec.eta_label,
        eta0=eta_spec.eta0,
        c=eta_spec.c,
        beta=beta,
        gamma=gamma,
        random_states=random_states,
        pA=pA,
        pB=pB,
        initial_state=initial_state,
        utility_spec_id=utility_spec_id,
        transition_spec_id=transition_spec_id,
        p_H_init=p_H_init,
        p_AI_init=p_AI_init,
        U_H=U_H,
        U_AI=U_AI,
        action_dependent_kernel=action_dependent_kernel,
        P_V_key=P_V_key,
    )


def iter_scenario1_grid(
    *,
    H: int,
    U_H: Mapping[str, ArrayLike] = MODEL_U_H,
    U_AI: Mapping[str, ArrayLike] = MODEL_U_AI,
    utility_spec_id: str = MODEL_UTILITY_SPEC_ID,
    action_dependent_kernel: Mapping[str, Mapping[int, Mapping[str, float]]] = MODEL_ACTION_DEPENDENT_KERNEL,
    transition_spec_id: str = MODEL_TRANSITION_SPEC_ID,
    seeds: Sequence[int] = MODEL_SEEDS,
    p_H_init: Mapping[str, ArrayLike] | None = MODEL_P_H_INIT,
    p_AI_init: Mapping[str, ArrayLike] | None = MODEL_P_AI_INIT,
    endogenous_initial_state: str = MODEL_ENDOGENOUS_INITIAL_STATE,
    atol: float = DEFAULT_ATOL,
) -> Iterator[Scenario1RunConfig]:
    """Lazily yield Scenario 1 configurations from common prepared assets."""
    prepared = prepare_binary_grid(
        H=H,
        U_H=U_H,
        U_AI=U_AI,
        utility_spec_id=utility_spec_id,
        action_dependent_kernel=action_dependent_kernel,
        transition_spec_id=transition_spec_id,
        seeds=seeds,
        p_H_init=p_H_init,
        p_AI_init=p_AI_init,
        endogenous_initial_state=endogenous_initial_state,
        horizon_grid=HORIZON_GRID,
        alpha_agree_grid=ALPHA_AGREE_GRID,
        alpha_disagree_grid=ALPHA_DISAGREE_GRID,
        beta_grid=BETA_GRID,
        gamma_grid=GAMMA_GRID,
        eta0_grid=ETA0_GRID,
        decay_c_grid=DECAY_C_GRID,
        state_probability_grid=STATE_PROBABILITY_GRID,
        states=STATES,
        n_actions=N_ACTIONS,
        atol=atol,
    )
    identifier_base_fragments = canonical_identifier_fragments(
        {
            "H": prepared.H,
            "utility_spec_id": prepared.utility_spec_id,
            "p_H_init": prepared.p_H_init,
            "p_AI_init": prepared.p_AI_init,
        }
    )
    winner_ids = {
        (eta_spec.eta_kind, random_states): build_winner_group_id_with_branch(
            scenario=SCENARIO_NAME,
            H=prepared.H,
            eta_kind=eta_spec.eta_kind,
            random_states=random_states,
        )
        for eta_spec in prepared.eta_specs
        for random_states in (True, False)
    }

    for (alpha_agree, alpha_disagree), beta, gamma, eta_spec, random_states in product(
        prepared.threshold_pairs,
        prepared.beta_values,
        prepared.gamma_values,
        prepared.eta_specs,
        (True, False),
    ):
        state_specs = (
            prepared.exogenous_state_specs
            if random_states
            else (prepared.endogenous_state_spec,)
        )
        winner_group_id = winner_ids[(eta_spec.eta_kind, random_states)]
        for state_spec in state_specs:
            for replicate, seed in enumerate(prepared.seeds, start=1):
                yield _build_prevalidated_grid_config(
                    H=prepared.H,
                    alpha_agree=alpha_agree,
                    alpha_disagree=alpha_disagree,
                    eta_spec=eta_spec,
                    beta=beta,
                    gamma=gamma,
                    random_states=random_states,
                    pA=state_spec.pA,
                    pB=state_spec.pB,
                    initial_state=state_spec.initial_state,
                    utility_spec_id=prepared.utility_spec_id,
                    transition_spec_id=state_spec.transition_spec_id,
                    replicate=replicate,
                    seed=seed,
                    p_H_init=prepared.p_H_init,
                    p_AI_init=prepared.p_AI_init,
                    U_H=prepared.U_H,
                    U_AI=prepared.U_AI,
                    action_dependent_kernel=state_spec.action_dependent_kernel,
                    P_V_key=state_spec.P_V_key,
                    winner_group_id=winner_group_id,
                    identifier_base_fragments=identifier_base_fragments,
                )


__all__ = [
    "ALPHA_AGREE_GRID",
    "ALPHA_DISAGREE_GRID",
    "BETA_GRID",
    "DECAY_C_GRID",
    "DEFAULT_SEEDS",
    "ETA0_GRID",
    "EtaGridSpec",
    "GAMMA_GRID",
    "HORIZON_GRID",
    "N_ACTIONS",
    "SCENARIO_NAME",
    "STATE_PROBABILITY_GRID",
    "STATES",
    "Scenario1RunConfig",
    "build_scenario1_run_config",
    "iter_eta_grid_specs",
    "iter_scenario1_grid",
]
