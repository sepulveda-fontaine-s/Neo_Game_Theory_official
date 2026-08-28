'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario 1 decision loop implementing the temporal contract.

Decision indices are ``T = 1, ..., H``. At each decision:

1. observe ``s_T`` and read the policies available before the decision;
2. record all four predecision ``a0`` policy components;
3. compute ``Sh_entr`` and ``D_JS_T`` before proposals;
4. classify the regime, sample proposals, and select ``lambda_T``;
5. resolve executor, owner, and ``a_star_T``;
6. compute the structural reward;
7. update the separate EWMA utility-credit traces;
8. apply the structural-reward Bellman backup;
9. apply the asymmetric policy updates and increment effective counts;
10. update cumulative frequencies including decision ``T``;
11. generate ``s_next``.

Consequently, ``D_JS_final`` is the last predecision divergence, whereas
``p_*_final``, ``D_JS_A_final``, and ``D_JS_B_final`` use the postdecision
policies after decision ``H``. The longitudinal ``Uhat`` values are post-EWMA,
``Vhat`` values are post-backup, and effective counts are postdecision.

Theory map
----------
- Steps 1--4: Part I Eqs. (5)--(10) for predecision entropy/D_JS and
  Eqs. (42)--(46) for Scenario 1 delegation.
- Step 6: Part I Eq. (38) [eq:reward_realized] for structural reward.
- Step 7: Part I Eq. (22) [estimated_utility] for EWMA utility credit.
- Step 8: Part I Eq. (41) [eq:computational_bellman] for the numerical
  Bellman backup.
- Step 9: Part I Eqs. (27)--(35) and Proposition 3 for learning rates,
  effective counts, and asymmetric policy updating.
- Step 10: Part I Eqs. (23)--(26) for cumulative regime frequencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from general_formulation.bellman import (
    computational_bellman_backup,
    initialize_value_table,
    structural_reward,
)
from general_formulation.inheritance import chain_payload_fields
from general_formulation.entropy import (
    jensen_shannon_divergence,
    shannon_entropy,
)
from general_formulation.frequencies import (
    FrequencyState,
    initialize_frequency_state,
    update_frequency_state,
)
from general_formulation.numerics import DEFAULT_ATOL, as_probability_vector
from general_formulation.policy_updates import (
    carry_policy_forward,
)
from general_formulation.state_generation import (
    initialize_state,
    resolve_virtual_nature_kernel,
    sample_next_state,
)
from general_formulation.utility_credit import (
    initialize_utility_credit_table,
    update_utility_credit,
)
from general_formulation.validation import assert_close

from .delegation import (
    REGIME_AGREEMENT,
    classify_scenario1_regime,
    select_scenario1_lambda,
)
from .execution import (
    action_label,
    resolve_scenario1_execution,
    sample_scenario1_proposals,
)
from .grid import N_ACTIONS, STATES, Scenario1RunConfig
from .update_rules import apply_scenario1_policy_updates


PolicyTable = dict[str, np.ndarray]
CountTable = dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """One complete run summary and, optionally, its longitudinal rows."""

    summary_row: dict[str, Any]
    trajectory_rows: tuple[dict[str, Any], ...]

class _RunningSeries:
    """Running mean and final value for one simulation series."""

    __slots__ = ("_sum", "_count", "last")

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0
        self.last = float("nan")

    def add(self, value: float) -> None:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ArithmeticError(
                "A recorded simulation series became non-finite."
            )

        self._sum += numeric
        self._count += 1
        self.last = numeric

    @property
    def mean(self) -> float:
        if self._count == 0:
            raise RuntimeError("Cannot summarize an empty series.")

        return float(self._sum / self._count)


def _copy_policy_table(values: Mapping[str, tuple[float, ...]]) -> PolicyTable:
    table = {
        state: as_probability_vector(
            values[state],
            name=f"policy[{state!r}]",
            atol=DEFAULT_ATOL,
        ).copy()
        for state in STATES
    }
    return table


def _copy_count_table(values: CountTable) -> CountTable:
    return {state: values[state].copy() for state in STATES}


def _count_columns(
    N_H: CountTable,
    N_AI: CountTable,
    *,
    suffix: str,
) -> dict[str, int]:
    return {
        f"N_H_A_a0_{suffix}": int(N_H["A"][0]),
        f"N_H_A_a1_{suffix}": int(N_H["A"][1]),
        f"N_H_B_a0_{suffix}": int(N_H["B"][0]),
        f"N_H_B_a1_{suffix}": int(N_H["B"][1]),
        f"N_AI_A_a0_{suffix}": int(N_AI["A"][0]),
        f"N_AI_A_a1_{suffix}": int(N_AI["A"][1]),
        f"N_AI_B_a0_{suffix}": int(N_AI["B"][0]),
        f"N_AI_B_a1_{suffix}": int(N_AI["B"][1]),
    }


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


def _frequency_columns(frequency_state: FrequencyState) -> dict[str, Any]:
    state_frequencies = frequency_state.state_frequencies_T
    return {
        "n_H_exec_T": frequency_state.n_H_exec_T,
        "n_AI_exec_T": frequency_state.n_AI_exec_T,
        "f_H_T": frequency_state.f_H_T,
        "f_AI_T": frequency_state.f_AI_T,
        "n_agree_T": frequency_state.n_agree_T,
        "n_ctx_T": frequency_state.n_ctx_T,
        "n_disagree_T": frequency_state.n_disagree_T,
        "f_agree_T": frequency_state.f_agree_T,
        "f_ctx_T": frequency_state.f_ctx_T,
        "f_disagree_T": frequency_state.f_disagree_T,
        "n_H_ctx_T": frequency_state.n_H_ctx_T,
        "n_AI_ctx_T": frequency_state.n_AI_ctx_T,
        "f_H_ctx_T": (
            None if frequency_state.n_ctx_T == 0 else frequency_state.f_H_ctx_T
        ),
        "f_AI_ctx_T": (
            None if frequency_state.n_ctx_T == 0 else frequency_state.f_AI_ctx_T
        ),
        "n_state_A_T": int(frequency_state.state_counts_T["A"]),
        "n_state_B_T": int(frequency_state.state_counts_T["B"]),
        "state_A_freq_T": float(state_frequencies["A"]),
        "state_B_freq_T": float(state_frequencies["B"]),
    }


def simulate_scenario1_reference(
    config: Scenario1RunConfig,
    *,
    return_trajectory: bool = False,
    atol: float = DEFAULT_ATOL,
) -> SimulationResult:
    """Execute one complete Scenario 1 run.

    Phase 1 should call this with ``return_trajectory=False``. Phase 2 reruns
    only the selected configurations with ``return_trajectory=True``.
    """
    if not isinstance(config, Scenario1RunConfig):
        raise TypeError("config must be a Scenario1RunConfig instance.")
    if not isinstance(return_trajectory, bool):
        raise TypeError("return_trajectory must be boolean.")

    rng = np.random.default_rng(config.seed)
    P_V = resolve_virtual_nature_kernel(
        random_states=config.random_states,
        pA=config.pA,
        pB=config.pB,
        action_dependent_kernel=config.action_dependent_kernel,
        states=STATES,
        n_actions=N_ACTIONS,
        atol=atol,
    )

    current_state = initialize_state(
        random_states=config.random_states,
        P_V=P_V,
        rng=rng,
        initial_state=config.initial_state,
        states=STATES,
        n_actions=N_ACTIONS,
        atol=atol,
    )
    initial_state_realized = current_state

    p_H = _copy_policy_table(config.p_H_init)
    p_AI = _copy_policy_table(config.p_AI_init)
    Uhat_H = initialize_utility_credit_table(STATES, n_actions=N_ACTIONS)
    Uhat_AI = initialize_utility_credit_table(STATES, n_actions=N_ACTIONS)
    Vhat = initialize_value_table(STATES)
    N_H: CountTable = {
        state: np.zeros(N_ACTIONS, dtype=np.int64) for state in STATES
    }
    N_AI: CountTable = {
        state: np.zeros(N_ACTIONS, dtype=np.int64) for state in STATES
    }
    frequencies = initialize_frequency_state(STATES)

    series_names = (
        "D_JS",
        "Sh_entr_H",
        "Sh_entr_AI",
        "p_H_A",
        "p_AI_A",
        "p_H_B",
        "p_AI_B",
        "absdiff_A",
        "absdiff_B",
        "reward_structural",
        "Uhat_H",
        "Uhat_AI",
        "Uhat_coal",
        "Vhat_A",
        "Vhat_B",
        "Vhat_coal",
    )
    series = {     name: _RunningSeries() for name in series_names }

    trajectory_rows: list[dict[str, Any]] = []
    contextual_prob_H_sum = 0.0
    contextual_prob_AI_sum = 0.0
    contextual_prob_count = 0
    ctx_prob_H_final: float | None = None
    ctx_prob_AI_final: float | None = None
    reward_cumulative = 0.0
    last_decision: dict[str, Any] | None = None

    repeated_fields = _repeated_run_fields(
        config,
        initial_state_realized=initial_state_realized,
    )

    for T in range(1, config.H + 1):
        s_T = current_state

        # Policies and all Plot-1 values are recorded before the decision.
        p_H_A_T = float(p_H["A"][0])
        p_AI_A_T = float(p_AI["A"][0])
        p_H_B_T = float(p_H["B"][0])
        p_AI_B_T = float(p_AI["B"][0])
        absdiff_A_T = abs(p_H_A_T - p_AI_A_T)
        absdiff_B_T = abs(p_H_B_T - p_AI_B_T)

        Sh_entr_H_T = shannon_entropy(
            p_H[s_T],
            name="Sh_entr_H_T_policy",
            atol=atol,
        )
        Sh_entr_AI_T = shannon_entropy(
            p_AI[s_T],
            name="Sh_entr_AI_T_policy",
            atol=atol,
        )
        D_JS_T = jensen_shannon_divergence(
            p_H[s_T],
            p_AI[s_T],
            atol=atol,
        )

        # Regime classification occurs before proposal sampling. Contextual
        # Contextual selection is deterministic after both proposals are available.
        regime_T = classify_scenario1_regime(
            D_JS_T,
            alpha_agree=config.alpha_agree,
            alpha_disagree=config.alpha_disagree,
            atol=atol,
        )
        proposals = sample_scenario1_proposals(
            p_H[s_T],
            p_AI[s_T],
            rng=rng,
            atol=atol,
        )
        delegation = select_scenario1_lambda(
            D_JS_T,
            alpha_agree=config.alpha_agree,
            alpha_disagree=config.alpha_disagree,
            rng=rng,
            regime_T=regime_T,
            atol=atol,
        )
        execution = resolve_scenario1_execution(
            lambda_T=delegation.lambda_T,
            a_H_T=proposals.a_H_T,
            a_AI_T=proposals.a_AI_T,
            n_actions=N_ACTIONS,
        )

        reward_structural_T = structural_reward(
            config.U_H,
            config.U_AI,
            s_T=s_T,
            a_star_T=execution.a_star_T,
            owner_T=execution.owner_T,
        )

        utility_update = update_utility_credit(
            Uhat_H,
            Uhat_AI,
            s_T=s_T,
            a_star_T=execution.a_star_T,
            owner_T=execution.owner_T,
            beta=config.beta,
            atol=atol,
        )
        Uhat_H = utility_update.Uhat_H_T_plus_1
        Uhat_AI = utility_update.Uhat_AI_T_plus_1

        bellman_update = computational_bellman_backup(
            Vhat,
            config.U_H,
            config.U_AI,
            P_V,
            s_T=s_T,
            a_H_T=proposals.a_H_T,
            a_AI_T=proposals.a_AI_T,
            a_star_T=execution.a_star_T,
            owner_T=execution.owner_T,
            gamma=config.gamma,
            atol=atol,
        )
        assert_close(
            bellman_update.reward_structural_T,
            reward_structural_T,
            name="reward_structural_T",
            atol=atol,
        )
        Vhat = bellman_update.Vhat_T_plus_1

        # Carry every state component forward, then replace only the current
        # state with the effective Scenario 1 updates.
        p_H_next = {
            state: carry_policy_forward(
                p_H[state],
                name=f"p_H[{state}]_T_plus_1",
                atol=atol,
            )
            for state in STATES
        }
        p_AI_next = {
            state: carry_policy_forward(
                p_AI[state],
                name=f"p_AI[{state}]_T_plus_1",
                atol=atol,
            )
            for state in STATES
        }
        N_H_next = _copy_count_table(N_H)
        N_AI_next = _copy_count_table(N_AI)

        policy_update = apply_scenario1_policy_updates(
            p_H[s_T],
            p_AI[s_T],
            N_H[s_T],
            N_AI[s_T],
            a_star_T=execution.a_star_T,
            regime_T=regime_T,
            eta_kind=config.eta_kind,
            eta0=config.eta0,
            c=config.c,
            T=T,
            atol=atol,
        )
        p_H_next[s_T] = policy_update.p_H_T_plus_1
        p_AI_next[s_T] = policy_update.p_AI_T_plus_1
        N_H_next[s_T] = policy_update.N_H_s_a_T
        N_AI_next[s_T] = policy_update.N_AI_s_a_T
        p_H, p_AI = p_H_next, p_AI_next
        N_H, N_AI = N_H_next, N_AI_next

        frequencies = update_frequency_state(
            frequencies,
            s_T=s_T,
            owner_T=execution.owner_T,
            regime_T=regime_T,
            atol=atol,
        )
        if frequencies.decision_count_T != T:
            raise ArithmeticError("Frequency counters are temporally misaligned.")

        s_next = sample_next_state(
            P_V,
            s_T=s_T,
            a_star_T=execution.a_star_T,
            rng=rng,
            states=STATES,
            n_actions=N_ACTIONS,
            atol=atol,
        )

        # Update running statistics in the same timing convention as the longitudinal columns and plots.
        for name, value in (
            ("D_JS", D_JS_T),
            ("Sh_entr_H", Sh_entr_H_T),
            ("Sh_entr_AI", Sh_entr_AI_T),
            ("p_H_A", p_H_A_T),
            ("p_AI_A", p_AI_A_T),
            ("p_H_B", p_H_B_T),
            ("p_AI_B", p_AI_B_T),
            ("absdiff_A", absdiff_A_T),
            ("absdiff_B", absdiff_B_T),
            ("reward_structural", reward_structural_T),
            ("Uhat_H", utility_update.Uhat_H_realized_T),
            ("Uhat_AI", utility_update.Uhat_AI_realized_T),
            ("Uhat_coal", utility_update.Uhat_coal_T),
            ("Vhat_A", Vhat["A"]),
            ("Vhat_B", Vhat["B"]),
            ("Vhat_coal", bellman_update.Vhat_coal_T),
        ):
            series[name].add(value)

        reward_cumulative += reward_structural_T

        if regime_T == "contextual":
            assert delegation.ctx_prob_H_T is not None
            assert delegation.ctx_prob_AI_T is not None
            contextual_prob_H_sum += delegation.ctx_prob_H_T
            contextual_prob_AI_sum += delegation.ctx_prob_AI_T
            contextual_prob_count += 1
            ctx_prob_H_final = delegation.ctx_prob_H_T
            ctx_prob_AI_final = delegation.ctx_prob_AI_T
        elif delegation.ctx_prob_H_T is not None or delegation.ctx_prob_AI_T is not None:
            raise ArithmeticError(
                "Contextual probabilities must be undefined outside contextual entry."
            )

        if return_trajectory:
            row: dict[str, Any] = {
                **repeated_fields,
                "T": T,
                "s_T": s_T,
                "s_next": s_next,
                "p_H_A_T": p_H_A_T,
                "p_AI_A_T": p_AI_A_T,
                "p_H_B_T": p_H_B_T,
                "p_AI_B_T": p_AI_B_T,
                "Sh_entr_H_T": Sh_entr_H_T,
                "Sh_entr_AI_T": Sh_entr_AI_T,
                "D_JS_T": D_JS_T,
                "regime_T": regime_T,
                "ctx_prob_H_T": delegation.ctx_prob_H_T,
                "ctx_prob_AI_T": delegation.ctx_prob_AI_T,
                "a_H_T": action_label(proposals.a_H_T),
                "a_AI_T": action_label(proposals.a_AI_T),
                "lambda_T": execution.lambda_T,
                "executor_T": execution.executor_T,
                "a_star_T": action_label(execution.a_star_T),
                "owner_T": execution.owner_T,
                "eta_H_T": policy_update.eta_H_T,
                "eta_AI_T": policy_update.eta_AI_T,
                "human_updated_T": policy_update.human_updated_T,
                "ai_updated_T": policy_update.ai_updated_T,
                "reward_structural_T": reward_structural_T,
                "Uhat_H_realized_T": utility_update.Uhat_H_realized_T,
                "Uhat_AI_realized_T": utility_update.Uhat_AI_realized_T,
                "Uhat_coal_T": utility_update.Uhat_coal_T,
                "Vhat_A_T": float(Vhat["A"]),
                "Vhat_B_T": float(Vhat["B"]),
                "Vhat_coal_T": bellman_update.Vhat_coal_T,
                **_frequency_columns(frequencies),
                **_count_columns(N_H, N_AI, suffix="T"),
            }
            trajectory_rows.append(row)

        last_decision = {
            "state_final": s_T,
            "a_H_final": action_label(proposals.a_H_T),
            "a_AI_final": action_label(proposals.a_AI_T),
            "a_star_final": action_label(execution.a_star_T),
            "owner_final": execution.owner_T,
            "lambda_final": execution.lambda_T,
            "regime_final": regime_T,
            "Uhat_H_realized_final": utility_update.Uhat_H_realized_T,
            "Uhat_AI_realized_final": utility_update.Uhat_AI_realized_T,
            "Uhat_coal_final": utility_update.Uhat_coal_T,
            "Vhat_coal_final": bellman_update.Vhat_coal_T,
            "reward_structural_final": reward_structural_T,
        }
        current_state = s_next

    if last_decision is None:  # guarded by validated positive H
        raise RuntimeError("The simulation produced no decisions.")

    # Final policy diagnostics are evaluated after processing decision H.
    p_H_A_final = float(p_H["A"][0])
    p_AI_A_final = float(p_AI["A"][0])
    p_H_B_final = float(p_H["B"][0])
    p_AI_B_final = float(p_AI["B"][0])
    D_JS_A_final = jensen_shannon_divergence(p_H["A"], p_AI["A"], atol=atol)
    D_JS_B_final = jensen_shannon_divergence(p_H["B"], p_AI["B"], atol=atol)
    absdiff_A_final = abs(p_H_A_final - p_AI_A_final)
    absdiff_B_final = abs(p_H_B_final - p_AI_B_final)

    state_frequencies = frequencies.state_frequencies_T
    if contextual_prob_count > 0:
        ctx_prob_H_mean: float | None = (
            contextual_prob_H_sum / contextual_prob_count
        )
        ctx_prob_AI_mean: float | None = (
            contextual_prob_AI_sum / contextual_prob_count
        )
        f_H_ctx: float | None = frequencies.f_H_ctx_T
        f_AI_ctx: float | None = frequencies.f_AI_ctx_T
    else:
        ctx_prob_H_mean = None
        ctx_prob_AI_mean = None
        f_H_ctx = None
        f_AI_ctx = None

    # AI incorporates one executed action at every decision; Human does so
    # exactly in agreement. These identities are independent of eta_kind.
    if sum(int(value) for state in STATES for value in N_AI[state]) != config.H:
        raise ArithmeticError("AI effective counts must sum to H.")
    if (
        sum(int(value) for state in STATES for value in N_H[state])
        != frequencies.n_agree_T
    ):
        raise ArithmeticError("Human effective counts must sum to n_agree.")

    summary_row: dict[str, Any] = {
        **repeated_fields,
        "selection_rank": 0,
        "is_selected": False,
        "p_H_A_init": float(config.p_H_init["A"][0]),
        "p_AI_A_init": float(config.p_AI_init["A"][0]),
        "p_H_B_init": float(config.p_H_init["B"][0]),
        "p_AI_B_init": float(config.p_AI_init["B"][0]),
        "D_JS_mean": series["D_JS"].mean,
        "D_JS_final": series["D_JS"].last,
        "D_JS_A_final": D_JS_A_final,
        "D_JS_B_final": D_JS_B_final,
        "Sh_entr_H_mean": series["Sh_entr_H"].mean,
        "Sh_entr_AI_mean": series["Sh_entr_AI"].mean,
        "Sh_entr_H_final": series["Sh_entr_H"].last,
        "Sh_entr_AI_final": series["Sh_entr_AI"].last,
        "p_H_A_final": p_H_A_final,
        "p_AI_A_final": p_AI_A_final,
        "p_H_B_final": p_H_B_final,
        "p_AI_B_final": p_AI_B_final,
        "absdiff_A_final": absdiff_A_final,
        "absdiff_B_final": absdiff_B_final,
        "reward_structural_final": last_decision["reward_structural_final"],
        "reward_structural_mean": series["reward_structural"].mean,
        "reward_structural_cumulative": float(reward_cumulative),
        "Uhat_H_realized_final": last_decision["Uhat_H_realized_final"],
        "Uhat_AI_realized_final": last_decision["Uhat_AI_realized_final"],
        "Uhat_coal_final": last_decision["Uhat_coal_final"],
        "Vhat_A_final": float(Vhat["A"]),
        "Vhat_B_final": float(Vhat["B"]),
        "Vhat_coal_final": last_decision["Vhat_coal_final"],
        "n_H_exec": frequencies.n_H_exec_T,
        "n_AI_exec": frequencies.n_AI_exec_T,
        "f_H_final": frequencies.f_H_T,
        "f_AI_final": frequencies.f_AI_T,
        "n_agree": frequencies.n_agree_T,
        "n_ctx": frequencies.n_ctx_T,
        "n_disagree": frequencies.n_disagree_T,
        "f_agree": frequencies.f_agree_T,
        "f_ctx": frequencies.f_ctx_T,
        "f_disagree": frequencies.f_disagree_T,
        "ctx_prob_H_mean": ctx_prob_H_mean,
        "ctx_prob_AI_mean": ctx_prob_AI_mean,
        "ctx_prob_H_final": ctx_prob_H_final,
        "ctx_prob_AI_final": ctx_prob_AI_final,
        "n_H_ctx": frequencies.n_H_ctx_T,
        "n_AI_ctx": frequencies.n_AI_ctx_T,
        "f_H_ctx": f_H_ctx,
        "f_AI_ctx": f_AI_ctx,
        "n_state_A": int(frequencies.state_counts_T["A"]),
        "n_state_B": int(frequencies.state_counts_T["B"]),
        "state_A_freq": float(state_frequencies["A"]),
        "state_B_freq": float(state_frequencies["B"]),
        **_count_columns(N_H, N_AI, suffix="final"),
        "state_final": last_decision["state_final"],
        "a_H_final": last_decision["a_H_final"],
        "a_AI_final": last_decision["a_AI_final"],
        "a_star_final": last_decision["a_star_final"],
        "owner_final": last_decision["owner_final"],
        "lambda_final": last_decision["lambda_final"],
        "regime_final": last_decision["regime_final"],
        **chain_payload_fields(config, p_H_final=p_H, p_AI_final=p_AI),
    }

    return SimulationResult(
        summary_row=summary_row,
        trajectory_rows=tuple(trajectory_rows),
    )


__all__ = [
    "CountTable",
    "PolicyTable",
    "SimulationResult",
    "simulate_scenario1_reference",
]
