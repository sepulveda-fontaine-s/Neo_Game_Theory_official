'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Fast Scenario 1 decision loop with boundary-only validation.

The public entry point validates the immutable ``Scenario1RunConfig`` once and
then executes a compact numerical kernel.  The strict reference
implementation remains available in ``simulation_reference.py`` for tests and
audits; it is intentionally not used by the exhaustive production grid.

Decision timing and every scientific update are unchanged:
predecision policies -> entropy/D_JS -> proposals -> delegation -> execution ->
structural reward -> EWMA credit -> Bellman backup -> policy updates ->
frequencies -> next state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from general_formulation.contextual import (
    _contextual_selector_probabilities_validated,
    contextual_max_probability_selector,
)
from general_formulation.entropy import (
    _binary_entropy_validated,
    _binary_js_validated,
)
from general_formulation.inheritance import chain_payload_fields
from general_formulation.numerics import (
    DEFAULT_ATOL,
    _binary_table_array_validated,
)
from general_formulation.plots import (
    DIVERGENCE_FREQUENCY_COLUMNS,
    POLICY_A0_COLUMNS,
    UTILITY_VALUE_COLUMNS,
)
from general_formulation.policy_updates import _update_binary_policy_inplace
from general_formulation.state_generation import (
    _binary_kernel_array_validated,
    _sample_binary_next_state_validated,
)

from .grid import N_ACTIONS, STATES, Scenario1RunConfig


PolicyTable = dict[str, np.ndarray]
CountTable = dict[str, np.ndarray]
TrajectorySink = Callable[[Mapping[str, Any]], None]

_PLOT_DATA_COLUMNS = (
    "T",
    *POLICY_A0_COLUMNS,
    *UTILITY_VALUE_COLUMNS,
    *DIVERGENCE_FREQUENCY_COLUMNS,
)

_STATE_TO_INDEX = {"A": 0, "B": 1}
_STATE_LABELS = ("A", "B")
_ACTION_LABELS = ("a0", "a1")
_OWNER_H = "H"
_OWNER_AI = "AI"
_REGIME_AGREEMENT = "agreement"
_REGIME_CONTEXTUAL = "contextual"
_REGIME_DISAGREEMENT = "disagreement"


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Run summary plus optional rows or compact representative plot data."""

    summary_row: dict[str, Any]
    trajectory_rows: tuple[dict[str, Any], ...]
    plot_data: dict[str, tuple[Any, ...]] | None = None


def _kernel_array(config: Scenario1RunConfig) -> np.ndarray:
    """Resolve one of the seven cached read-only Virtual-Nature arrays."""
    return _binary_kernel_array_validated(config.P_V_key)


def _table_array(values: Any, *, mutable: bool) -> np.ndarray:
    """Resolve a cached validated 2x2 table, copying only mutable policies."""
    key = (
        (float(values["A"][0]), float(values["A"][1])),
        (float(values["B"][0]), float(values["B"][1])),
    )
    cached = _binary_table_array_validated(key)
    return cached.copy() if mutable else cached


def _repeated_run_fields(
    config: Scenario1RunConfig,
    *,
    initial_state_realized: str,
) -> dict[str, Any]:
    return {
        "scenario": config.scenario,
        "H": config.H,
        "winner_group_id": config.winner_group_id,
        "config_id": config.config_id,
        "run_id": config.run_id,
        "replicate": config.replicate,
        "seed": config.seed,
        "eta_kind": config.eta_kind,
        "eta_label": config.eta_label,
        "random_states": config.random_states,
        "alpha_agree": config.alpha_agree,
        "alpha_disagree": config.alpha_disagree,
        "eta0": config.eta0,
        "c": config.c,
        "beta": config.beta,
        "gamma": config.gamma,
        "pA": config.pA,
        "pB": config.pB,
        "initial_state": initial_state_realized,
        "utility_spec_id": config.utility_spec_id,
        "transition_spec_id": config.transition_spec_id,
    }


def simulate_scenario1(
    config: Scenario1RunConfig,
    *,
    return_trajectory: bool = False,
    trajectory_sink: TrajectorySink | None = None,
    collect_plot_data: bool = False,
    atol: float = DEFAULT_ATOL,
) -> SimulationResult:
    """Execute one complete Scenario 1 run through the production fast path.

    ``atol`` is accepted for API compatibility.  Numerical inputs were already
    validated when ``Scenario1RunConfig`` was built, so it is not revalidated
    at every decision instant.
    """
    if not isinstance(config, Scenario1RunConfig):
        raise TypeError("config must be a Scenario1RunConfig instance.")
    if not isinstance(return_trajectory, bool):
        raise TypeError("return_trajectory must be boolean.")
    if trajectory_sink is not None and not callable(trajectory_sink):
        raise TypeError("trajectory_sink must be callable or None.")
    if not isinstance(collect_plot_data, bool):
        raise TypeError("collect_plot_data must be boolean.")
    if not np.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError("atol must be finite and nonnegative.")
    if N_ACTIONS != 2 or tuple(STATES) != _STATE_LABELS:
        raise RuntimeError("The fast kernel requires states (A, B) and two actions.")

    rng = np.random.default_rng(config.seed)
    P_V = _kernel_array(config)

    if config.random_states:
        current_state = _sample_binary_next_state_validated(float(P_V[0, 0, 0]), rng=rng, atol=atol)
    else:
        current_state = _STATE_TO_INDEX[str(config.initial_state)]
    initial_state_realized = _STATE_LABELS[current_state]

    p_H = _table_array(config.p_H_init, mutable=True)
    p_AI = _table_array(config.p_AI_init, mutable=True)
    U_H = _table_array(config.U_H, mutable=False)
    U_AI = _table_array(config.U_AI, mutable=False)
    Uhat_H = np.zeros((2, 2), dtype=np.float64)
    Uhat_AI = np.zeros((2, 2), dtype=np.float64)
    Vhat = np.zeros(2, dtype=np.float64)
    N_H = np.zeros((2, 2), dtype=np.int64)
    N_AI = np.zeros((2, 2), dtype=np.int64)

    # Running counters and sums.  Phase 1 never allocates longitudinal rows.
    # Phase 2 may stream rows directly to CSV while retaining only compact plot
    # series for the representative run.
    trajectory_rows: list[dict[str, Any]] = []
    needs_trajectory_row = (
        return_trajectory or trajectory_sink is not None or collect_plot_data
    )
    plot_lists: dict[str, list[Any]] | None = (
        {column: [] for column in _PLOT_DATA_COLUMNS}
        if collect_plot_data
        else None
    )
    n_H_exec = n_AI_exec = 0
    n_agree = n_ctx = n_disagree = 0
    n_H_ctx = n_AI_ctx = 0
    n_state_A = n_state_B = 0
    D_JS_sum = 0.0
    Sh_entr_H_sum = 0.0
    Sh_entr_AI_sum = 0.0
    reward_sum = 0.0
    reward_cumulative = 0.0
    contextual_prob_H_sum = 0.0
    contextual_prob_AI_sum = 0.0
    contextual_prob_count = 0
    ctx_prob_H_final: float | None = None
    ctx_prob_AI_final: float | None = None

    repeated_fields = _repeated_run_fields(
        config,
        initial_state_realized=initial_state_realized,
    )

    # Last-decision values are overwritten on every positive-horizon step.
    D_JS_T = Sh_entr_H_T = Sh_entr_AI_T = 0.0
    reward_structural_T = 0.0
    Uhat_H_realized_T = Uhat_AI_realized_T = Uhat_coal_T = 0.0
    Vhat_coal_T = 0.0
    s_T = a_H_T = a_AI_T = a_star_T = lambda_T = 0
    owner_T = _OWNER_AI
    regime_T = _REGIME_AGREEMENT

    alpha_agree = float(config.alpha_agree)
    alpha_disagree = float(config.alpha_disagree)
    beta = float(config.beta)
    decay = 1.0 - beta
    gamma = float(config.gamma)
    eta_kind = config.eta_kind
    eta0 = config.eta0
    c = config.c

    for T in range(1, config.H + 1):
        s_T = current_state

        # Predecision policy values and diagnostics.
        p_H_A_T = float(p_H[0, 0])
        p_AI_A_T = float(p_AI[0, 0])
        p_H_B_T = float(p_H[1, 0])
        p_AI_B_T = float(p_AI[1, 0])

        Sh_entr_H_T = _binary_entropy_validated(float(p_H[s_T, 0]), float(p_H[s_T, 1]), atol=atol)
        Sh_entr_AI_T = _binary_entropy_validated(float(p_AI[s_T, 0]), float(p_AI[s_T, 1]), atol=atol)
        D_JS_T = _binary_js_validated(
            float(p_H[s_T, 0]),
            float(p_H[s_T, 1]),
            float(p_AI[s_T, 0]),
            float(p_AI[s_T, 1]),
            Sh_entr_H_T,
            Sh_entr_AI_T,
            atol=atol,
        )

        # Regime, proposals, and scenario-specific selector.
        if D_JS_T <= alpha_agree:
            regime_T = _REGIME_AGREEMENT
        elif D_JS_T >= alpha_disagree:
            regime_T = _REGIME_DISAGREEMENT
        else:
            regime_T = _REGIME_CONTEXTUAL

        # Preserve the original RNG call pattern for reproducibility.
        a_H_T = int(rng.choice(2, p=p_H[s_T]))
        a_AI_T = int(rng.choice(2, p=p_AI[s_T]))

        ctx_prob_H_T: float | None
        ctx_prob_AI_T: float | None
        if regime_T == _REGIME_AGREEMENT:
            lambda_T = 0
            ctx_prob_H_T = None
            ctx_prob_AI_T = None
        elif regime_T == _REGIME_DISAGREEMENT:
            lambda_T = 1
            ctx_prob_H_T = None
            ctx_prob_AI_T = None
        else:

            #Part I Eqs. (43)--(44): contextual support and deterministic selector; exact tie -> Human.
            q_one, q_zero = _contextual_selector_probabilities_validated(
                D_JS_T, alpha_agree, alpha_disagree
            )
            ctx_prob_H_T, ctx_prob_AI_T = q_one, q_zero
            lambda_T = contextual_max_probability_selector(
                ctx_prob_H_T,
                ctx_prob_AI_T,
                tie_selector=1,
                atol=atol,
            )

        if lambda_T == 1:
            owner_T = _OWNER_H
            a_star_T = a_H_T
            selected_utilities = U_H
            n_H_exec += 1
        else:
            owner_T = _OWNER_AI
            a_star_T = a_AI_T
            selected_utilities = U_AI
            n_AI_exec += 1

        # Theory: Part I Eq. (38) [eq:reward_realized]: structural reward follows proposal owner.
        reward_structural_T = float(selected_utilities[s_T, a_star_T])

        # EWMA utility-credit update, Part I Eq. (22) [estimated_utility].
        Uhat_H *= decay
        Uhat_AI *= decay
        if owner_T == _OWNER_H:
            Uhat_H[s_T, a_star_T] += beta
        else:
            Uhat_AI[s_T, a_star_T] += beta
        Uhat_H_realized_T = float(Uhat_H[s_T, a_star_T])
        Uhat_AI_realized_T = float(Uhat_AI[s_T, a_star_T])
        Uhat_coal_T = (
            Uhat_H_realized_T
            if owner_T == _OWNER_H
            else Uhat_AI_realized_T
        )

        # Structural-reward Bellman backup, Part I Eq. (41) [eq:computational_bellman].
        value0 = float(
            selected_utilities[s_T, a_H_T]
            + gamma * np.dot(P_V[s_T, a_H_T], Vhat)
        )
        if a_AI_T == a_H_T:
            backed_up_value = value0
        else:
            value1 = float(
                selected_utilities[s_T, a_AI_T]
                + gamma * np.dot(P_V[s_T, a_AI_T], Vhat)
            )
            backed_up_value = value0 if value0 >= value1 else value1
        Vhat[s_T] = backed_up_value
        Vhat_coal_T = backed_up_value

        # Scenario 1 update mask: AI always; Human only in agreement, Part I Eq. (35) and Proposition 3.
        eta_AI_T = _update_binary_policy_inplace(
            p_AI[s_T],
            N_AI[s_T],
            action=a_star_T,
            eta_kind=eta_kind,
            eta0=eta0,
            c=c,
            T=T,
            atol=atol,
        )
        if regime_T == _REGIME_AGREEMENT:
            eta_H_T = _update_binary_policy_inplace(
                p_H[s_T],
                N_H[s_T],
                action=a_star_T,
                eta_kind=eta_kind,
                eta0=eta0,
                c=c,
                T=T,
                atol=atol,
            )
            human_updated_T = True
            n_agree += 1
        else:
            eta_H_T = 0.0
            human_updated_T = False
            if regime_T == _REGIME_CONTEXTUAL:
                n_ctx += 1
            else:
                n_disagree += 1

        if regime_T == _REGIME_CONTEXTUAL:
            assert ctx_prob_H_T is not None and ctx_prob_AI_T is not None
            contextual_prob_H_sum += ctx_prob_H_T
            contextual_prob_AI_sum += ctx_prob_AI_T
            contextual_prob_count += 1
            ctx_prob_H_final = ctx_prob_H_T
            ctx_prob_AI_final = ctx_prob_AI_T
            if owner_T == _OWNER_H:
                n_H_ctx += 1
            else:
                n_AI_ctx += 1

        if s_T == 0:
            n_state_A += 1
        else:
            n_state_B += 1

        # State transition uses one uniform draw, as in sample_next_state().
        s_next = _sample_binary_next_state_validated(float(P_V[s_T, a_star_T, 0]), rng=rng, atol=atol)

        D_JS_sum += D_JS_T
        Sh_entr_H_sum += Sh_entr_H_T
        Sh_entr_AI_sum += Sh_entr_AI_T
        reward_sum += reward_structural_T
        reward_cumulative += reward_structural_T

        # Part I Eqs. (23)--(26): empirical regime frequencies and partition.
        if needs_trajectory_row:
            f_H_T = n_H_exec / T
            f_AI_T = n_AI_exec / T
            f_agree_T = n_agree / T
            f_ctx_T = n_ctx / T
            f_disagree_T = n_disagree / T
            row: dict[str, Any] = {
                **repeated_fields,
                "T": T,
                "s_T": _STATE_LABELS[s_T],
                "s_next": _STATE_LABELS[s_next],
                "p_H_A_T": p_H_A_T,
                "p_AI_A_T": p_AI_A_T,
                "p_H_B_T": p_H_B_T,
                "p_AI_B_T": p_AI_B_T,
                "Sh_entr_H_T": Sh_entr_H_T,
                "Sh_entr_AI_T": Sh_entr_AI_T,
                "D_JS_T": D_JS_T,
                "regime_T": regime_T,
                "ctx_prob_H_T": ctx_prob_H_T,
                "ctx_prob_AI_T": ctx_prob_AI_T,
                "a_H_T": _ACTION_LABELS[a_H_T],
                "a_AI_T": _ACTION_LABELS[a_AI_T],
                "lambda_T": lambda_T,
                "executor_T": owner_T,
                "a_star_T": _ACTION_LABELS[a_star_T],
                "owner_T": owner_T,
                "eta_H_T": eta_H_T,
                "eta_AI_T": eta_AI_T,
                "human_updated_T": human_updated_T,
                "ai_updated_T": True,
                "reward_structural_T": reward_structural_T,
                "Uhat_H_realized_T": Uhat_H_realized_T,
                "Uhat_AI_realized_T": Uhat_AI_realized_T,
                "Uhat_coal_T": Uhat_coal_T,
                "Vhat_A_T": float(Vhat[0]),
                "Vhat_B_T": float(Vhat[1]),
                "Vhat_coal_T": Vhat_coal_T,
                "n_H_exec_T": n_H_exec,
                "n_AI_exec_T": n_AI_exec,
                "f_H_T": f_H_T,
                "f_AI_T": f_AI_T,
                "n_agree_T": n_agree,
                "n_ctx_T": n_ctx,
                "n_disagree_T": n_disagree,
                "f_agree_T": f_agree_T,
                "f_ctx_T": f_ctx_T,
                "f_disagree_T": f_disagree_T,
                "n_H_ctx_T": n_H_ctx,
                "n_AI_ctx_T": n_AI_ctx,
                "f_H_ctx_T": None if n_ctx == 0 else n_H_ctx / n_ctx,
                "f_AI_ctx_T": None if n_ctx == 0 else n_AI_ctx / n_ctx,
                "n_state_A_T": n_state_A,
                "n_state_B_T": n_state_B,
                "state_A_freq_T": n_state_A / T,
                "state_B_freq_T": n_state_B / T,
                "N_H_A_a0_T": int(N_H[0, 0]),
                "N_H_A_a1_T": int(N_H[0, 1]),
                "N_H_B_a0_T": int(N_H[1, 0]),
                "N_H_B_a1_T": int(N_H[1, 1]),
                "N_AI_A_a0_T": int(N_AI[0, 0]),
                "N_AI_A_a1_T": int(N_AI[0, 1]),
                "N_AI_B_a0_T": int(N_AI[1, 0]),
                "N_AI_B_a1_T": int(N_AI[1, 1]),
            }
            if return_trajectory:
                trajectory_rows.append(row)
            if plot_lists is not None:
                for column in _PLOT_DATA_COLUMNS:
                    plot_lists[column].append(row[column])
            if trajectory_sink is not None:
                trajectory_sink(row)

        current_state = s_next

    # Final postdecision diagnostics.
    p_H_A_final = float(p_H[0, 0])
    p_AI_A_final = float(p_AI[0, 0])
    p_H_B_final = float(p_H[1, 0])
    p_AI_B_final = float(p_AI[1, 0])
    D_JS_A_final = _binary_js_validated(float(p_H[0, 0]), float(p_H[0, 1]), float(p_AI[0, 0]), float(p_AI[0, 1]), atol=atol)
    D_JS_B_final = _binary_js_validated(float(p_H[1, 0]), float(p_H[1, 1]), float(p_AI[1, 0]), float(p_AI[1, 1]), atol=atol)
    absdiff_A_final = abs(p_H_A_final - p_AI_A_final)
    absdiff_B_final = abs(p_H_B_final - p_AI_B_final)

    if int(N_AI.sum()) != config.H:
        raise ArithmeticError("AI effective counts must sum to H.")
    if int(N_H.sum()) != n_agree:
        raise ArithmeticError("Human effective counts must sum to n_agree.")
    if n_H_exec + n_AI_exec != config.H:
        raise ArithmeticError("Execution counts must partition H.")
    if n_agree + n_ctx + n_disagree != config.H:
        raise ArithmeticError("Regime counts must partition H.")
    if n_state_A + n_state_B != config.H:
        raise ArithmeticError("State counts must partition H.")

    if contextual_prob_count:
        ctx_prob_H_mean: float | None = (
            contextual_prob_H_sum / contextual_prob_count
        )
        ctx_prob_AI_mean: float | None = (
            contextual_prob_AI_sum / contextual_prob_count
        )
        f_H_ctx: float | None = n_H_ctx / n_ctx
        f_AI_ctx: float | None = n_AI_ctx / n_ctx
    else:
        ctx_prob_H_mean = None
        ctx_prob_AI_mean = None
        f_H_ctx = None
        f_AI_ctx = None

    H_float = float(config.H)
    summary_row: dict[str, Any] = {
        **repeated_fields,
        "selection_rank": 0,
        "is_selected": False,
        "p_H_A_init": float(config.p_H_init["A"][0]),
        "p_AI_A_init": float(config.p_AI_init["A"][0]),
        "p_H_B_init": float(config.p_H_init["B"][0]),
        "p_AI_B_init": float(config.p_AI_init["B"][0]),
        "D_JS_mean": D_JS_sum / H_float,
        "D_JS_final": D_JS_T,
        "D_JS_A_final": D_JS_A_final,
        "D_JS_B_final": D_JS_B_final,
        "Sh_entr_H_mean": Sh_entr_H_sum / H_float,
        "Sh_entr_AI_mean": Sh_entr_AI_sum / H_float,
        "Sh_entr_H_final": Sh_entr_H_T,
        "Sh_entr_AI_final": Sh_entr_AI_T,
        "p_H_A_final": p_H_A_final,
        "p_AI_A_final": p_AI_A_final,
        "p_H_B_final": p_H_B_final,
        "p_AI_B_final": p_AI_B_final,
        "absdiff_A_final": absdiff_A_final,
        "absdiff_B_final": absdiff_B_final,
        "reward_structural_final": reward_structural_T,
        "reward_structural_mean": reward_sum / H_float,
        "reward_structural_cumulative": float(reward_cumulative),
        "Uhat_H_realized_final": Uhat_H_realized_T,
        "Uhat_AI_realized_final": Uhat_AI_realized_T,
        "Uhat_coal_final": Uhat_coal_T,
        "Vhat_A_final": float(Vhat[0]),
        "Vhat_B_final": float(Vhat[1]),
        "Vhat_coal_final": Vhat_coal_T,
        "n_H_exec": n_H_exec,
        "n_AI_exec": n_AI_exec,
        "f_H_final": n_H_exec / H_float,
        "f_AI_final": n_AI_exec / H_float,
        "n_agree": n_agree,
        "n_ctx": n_ctx,
        "n_disagree": n_disagree,
        "f_agree": n_agree / H_float,
        "f_ctx": n_ctx / H_float,
        "f_disagree": n_disagree / H_float,
        "ctx_prob_H_mean": ctx_prob_H_mean,
        "ctx_prob_AI_mean": ctx_prob_AI_mean,
        "ctx_prob_H_final": ctx_prob_H_final,
        "ctx_prob_AI_final": ctx_prob_AI_final,
        "n_H_ctx": n_H_ctx,
        "n_AI_ctx": n_AI_ctx,
        "f_H_ctx": f_H_ctx,
        "f_AI_ctx": f_AI_ctx,
        "n_state_A": n_state_A,
        "n_state_B": n_state_B,
        "state_A_freq": n_state_A / H_float,
        "state_B_freq": n_state_B / H_float,
        "N_H_A_a0_final": int(N_H[0, 0]),
        "N_H_A_a1_final": int(N_H[0, 1]),
        "N_H_B_a0_final": int(N_H[1, 0]),
        "N_H_B_a1_final": int(N_H[1, 1]),
        "N_AI_A_a0_final": int(N_AI[0, 0]),
        "N_AI_A_a1_final": int(N_AI[0, 1]),
        "N_AI_B_a0_final": int(N_AI[1, 0]),
        "N_AI_B_a1_final": int(N_AI[1, 1]),
        "state_final": _STATE_LABELS[s_T],
        "a_H_final": _ACTION_LABELS[a_H_T],
        "a_AI_final": _ACTION_LABELS[a_AI_T],
        "a_star_final": _ACTION_LABELS[a_star_T],
        "owner_final": owner_T,
        "lambda_final": lambda_T,
        "regime_final": regime_T,
        **chain_payload_fields(config, p_H_final=p_H, p_AI_final=p_AI),
    }

    plot_data = (
        None
        if plot_lists is None
        else {column: tuple(values) for column, values in plot_lists.items()}
    )
    return SimulationResult(
        summary_row=summary_row,
        trajectory_rows=tuple(trajectory_rows),
        plot_data=plot_data,
    )


__all__ = [
    "CountTable",
    "PolicyTable",
    "SimulationResult",
    "TrajectorySink",
    "simulate_scenario1",
]
