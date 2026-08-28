'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Ordered Scenario 1 contracts for the three CSV outputs.

The schemas implement the approved two-phase output design:

- ``scenario1_all_runs.csv``: one summary row per grid run;
- ``scenario1_best_joint_allruns.csv``: one row per ``(run_id, T)`` for
  runs belonging to selected configurations;
- ``scenario1_best_joint_final.csv``: one compact winner row per
  ``(H, eta_kind, random_states)`` group, with four explicit plot filenames.

Action ``a1`` policy columns are intentionally absent because the binary-action
probabilities are reconstructed exactly as ``1 - p(a0)``. ``kappa``, ``q_ctx``,
``joint_score_final``, ``Vhat_H``, and ``Vhat_AI`` are forbidden.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from general_formulation.csv_outputs import CsvSchema
from general_formulation.inheritance import CHAIN_COLUMNS
from general_formulation.plots import (
    DIVERGENCE_FREQUENCY_COLUMNS,
    POLICY_A0_COLUMNS,
    UTILITY_VALUE_COLUMNS,
)
from general_formulation.validation import validate_unique_records


ALL_RUNS_COLUMNS: Final[tuple[str, ...]] = (
    # Identifiers
    "scenario",
    "H",
    "winner_group_id",
    "config_id",
    "run_id",
    "replicate",
    "seed",
    "selection_rank",
    "is_selected",
    # Model parameters
    "alpha_agree",
    "alpha_disagree",
    "eta_kind",
    "eta_label",
    "eta0",
    "c",
    "beta",
    "gamma",
    "random_states",
    "pA",
    "pB",
    "initial_state",
    "utility_spec_id",
    "transition_spec_id",
    # Initial policies, a0 only
    "p_H_A_init",
    "p_AI_A_init",
    "p_H_B_init",
    "p_AI_B_init",
    # Divergence and entropy
    "D_JS_mean",
    "D_JS_final",
    "D_JS_A_final",
    "D_JS_B_final",
    "Sh_entr_H_mean",
    "Sh_entr_AI_mean",
    "Sh_entr_H_final",
    "Sh_entr_AI_final",
    # Final policies and absolute policy differences
    "p_H_A_final",
    "p_AI_A_final",
    "p_H_B_final",
    "p_AI_B_final",
    "absdiff_A_final",
    "absdiff_B_final",
    # Structural reward
    "reward_structural_final",
    "reward_structural_mean",
    "reward_structural_cumulative",
    # Utility-credit traces
    "Uhat_H_realized_final",
    "Uhat_AI_realized_final",
    "Uhat_coal_final",
    # Computational value
    "Vhat_A_final",
    "Vhat_B_final",
    "Vhat_coal_final",
    # Execution and regime frequencies
    "n_H_exec",
    "n_AI_exec",
    "f_H_final",
    "f_AI_final",
    "n_agree",
    "n_ctx",
    "n_disagree",
    "f_agree",
    "f_ctx",
    "f_disagree",
    # Contextual diagnostics
    "ctx_prob_H_mean",
    "ctx_prob_AI_mean",
    "ctx_prob_H_final",
    "ctx_prob_AI_final",
    "n_H_ctx",
    "n_AI_ctx",
    "f_H_ctx",
    "f_AI_ctx",
    # State visits
    "n_state_A",
    "n_state_B",
    "state_A_freq",
    "state_B_freq",
    # Effective incorporation counts
    "N_H_A_a0_final",
    "N_H_A_a1_final",
    "N_H_B_a0_final",
    "N_H_B_a1_final",
    "N_AI_A_a0_final",
    "N_AI_A_a1_final",
    "N_AI_B_a0_final",
    "N_AI_B_a1_final",
    # Self-contained payload for the next scenario
    *CHAIN_COLUMNS,
    # Last decision
    "state_final",
    "a_H_final",
    "a_AI_final",
    "a_star_final",
    "owner_final",
    "lambda_final",
    "regime_final",
)

ALL_RUNS_NULLABLE: Final[frozenset[str]] = frozenset(
    {
        "eta0",
        "c",
        "pA",
        "pB",
        "ctx_prob_H_mean",
        "ctx_prob_AI_mean",
        "ctx_prob_H_final",
        "ctx_prob_AI_final",
        "f_H_ctx",
        "f_AI_ctx",
    }
)


BEST_JOINT_ALLRUNS_COLUMNS: Final[tuple[str, ...]] = (
    # Repeated identifiers and parameters
    "scenario",
    "H",
    "winner_group_id",
    "config_id",
    "run_id",
    "replicate",
    "seed",
    "eta_kind",
    "eta_label",
    "random_states",
    "alpha_agree",
    "alpha_disagree",
    "eta0",
    "c",
    "beta",
    "gamma",
    "pA",
    "pB",
    "initial_state",
    "utility_spec_id",
    "transition_spec_id",
    # Decision and state
    "T",
    "s_T",
    "s_next",
    # Predecision policies and entropy
    "p_H_A_T",
    "p_AI_A_T",
    "p_H_B_T",
    "p_AI_B_T",
    "Sh_entr_H_T",
    "Sh_entr_AI_T",
    # Delegation
    "D_JS_T",
    "regime_T",
    "ctx_prob_H_T",
    "ctx_prob_AI_T",
    # Proposals and execution
    "a_H_T",
    "a_AI_T",
    "lambda_T",
    "executor_T",
    "a_star_T",
    "owner_T",
    # Learning rates and update mask
    "eta_H_T",
    "eta_AI_T",
    "human_updated_T",
    "ai_updated_T",
    # Structural reward
    "reward_structural_T",
    # Post-EWMA utility credit
    "Uhat_H_realized_T",
    "Uhat_AI_realized_T",
    "Uhat_coal_T",
    # Post-backup value
    "Vhat_A_T",
    "Vhat_B_T",
    "Vhat_coal_T",
    # Cumulative execution, regime, contextual, and state statistics
    "n_H_exec_T",
    "n_AI_exec_T",
    "f_H_T",
    "f_AI_T",
    "n_agree_T",
    "n_ctx_T",
    "n_disagree_T",
    "f_agree_T",
    "f_ctx_T",
    "f_disagree_T",
    "n_H_ctx_T",
    "n_AI_ctx_T",
    "f_H_ctx_T",
    "f_AI_ctx_T",
    "n_state_A_T",
    "n_state_B_T",
    "state_A_freq_T",
    "state_B_freq_T",
    # Postdecision effective counts
    "N_H_A_a0_T",
    "N_H_A_a1_T",
    "N_H_B_a0_T",
    "N_H_B_a1_T",
    "N_AI_A_a0_T",
    "N_AI_A_a1_T",
    "N_AI_B_a0_T",
    "N_AI_B_a1_T",
)

BEST_JOINT_ALLRUNS_NULLABLE: Final[frozenset[str]] = frozenset(
    {
        "eta0",
        "c",
        "pA",
        "pB",
        "ctx_prob_H_T",
        "ctx_prob_AI_T",
        "f_H_ctx_T",
        "f_AI_ctx_T",
    }
)


BEST_JOINT_FINAL_COLUMNS: Final[tuple[str, ...]] = (
    # Winner configuration
    "scenario",
    "H",
    "winner_group_id",
    "config_id",
    "eta_kind",
    "eta_label",
    "random_states",
    "alpha_agree",
    "alpha_disagree",
    "eta0",
    "c",
    "beta",
    "gamma",
    "pA",
    "pB",
    "initial_state",
    "utility_spec_id",
    "transition_spec_id",
    # Representative run
    "representative_run_id",
    "representative_replicate",
    "representative_seed",
    "D_JS_final",
    "D_JS_A_final",
    "D_JS_B_final",
    "Sh_entr_H_final",
    "Sh_entr_AI_final",
    "absdiff_A_final",
    "absdiff_B_final",
    "p_H_A_final",
    "p_AI_A_final",
    "p_H_B_final",
    "p_AI_B_final",
    *CHAIN_COLUMNS,
    "Uhat_H_realized_final",
    "Uhat_AI_realized_final",
    "Uhat_coal_final",
    "Vhat_A_final",
    "Vhat_B_final",
    "Vhat_coal_final",
    "reward_structural_final",
    "reward_structural_mean",
    "f_H_final",
    "f_AI_final",
    "f_agree",
    "f_ctx",
    "f_disagree",
    "state_A_freq",
    "state_B_freq",
    # Robustness across all runs of the selected configuration
    "n_replicates",
    "D_JS_final_mean",
    "D_JS_final_sd",
    "absdiff_A_final_mean",
    "absdiff_A_final_sd",
    "absdiff_B_final_mean",
    "absdiff_B_final_sd",
    "f_H_final_mean",
    "f_H_final_sd",
    "f_AI_final_mean",
    "f_AI_final_sd",
    "f_agree_mean",
    "f_agree_sd",
    "f_ctx_mean",
    "f_ctx_sd",
    "f_disagree_mean",
    "f_disagree_sd",
    "reward_structural_mean_across_runs",
    "reward_structural_sd_across_runs",
    # Four explicit plot links
    "plot_prefix",
    "plot1_a0_file",
    "plot1_a1_file",
    "plot2_file",
    "plot3_file",
)

BEST_JOINT_FINAL_NULLABLE: Final[frozenset[str]] = frozenset(
    {"eta0", "c", "pA", "pB"}
)


ALL_RUNS_SCHEMA = CsvSchema(
    name="scenario1_all_runs",
    columns=ALL_RUNS_COLUMNS,
    nullable=ALL_RUNS_NULLABLE,
    primary_key=("run_id",),
)

BEST_JOINT_ALLRUNS_SCHEMA = CsvSchema(
    name="scenario1_best_joint_allruns",
    columns=BEST_JOINT_ALLRUNS_COLUMNS,
    nullable=BEST_JOINT_ALLRUNS_NULLABLE,
    primary_key=("run_id", "T"),
)

BEST_JOINT_FINAL_SCHEMA = CsvSchema(
    name="scenario1_best_joint_final",
    columns=BEST_JOINT_FINAL_COLUMNS,
    nullable=BEST_JOINT_FINAL_NULLABLE,
    primary_key=("winner_group_id",),
)


FORBIDDEN_OUTPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "kappa",
        "q_ctx",
        "joint_score_final",
        "Vhat_H",
        "Vhat_AI",
        "winner_eta_kind",
        "winner_random_states",
    }
)


def validate_no_forbidden_fields(records: Iterable[Mapping[str, Any]]) -> None:
    """Reject obsolete or scientifically invalid output fields."""
    for row_number, row in enumerate(records, start=1):
        present = FORBIDDEN_OUTPUT_FIELDS.intersection(row)
        if present:
            raise ValueError(
                f"Output row {row_number} contains forbidden fields: "
                f"{sorted(present)}."
            )


def trajectory_rows_to_plot_data(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Any]]:
    """Convert one run's longitudinal rows into the four-plot data mapping."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("At least one longitudinal row is required.")

    validate_unique_records(
        materialized,
        key_fields=("run_id", "T"),
        record_name="trajectory",
    )
    run_ids = {row.get("run_id") for row in materialized}
    if len(run_ids) != 1:
        raise ValueError("Plot data must contain exactly one run_id.")

    ordered = sorted(materialized, key=lambda row: int(row["T"]))
    expected_T = list(range(1, len(ordered) + 1))
    observed_T = [int(row["T"]) for row in ordered]
    if observed_T != expected_T:
        raise ValueError(
            f"Trajectory T values must be consecutive {expected_T}; "
            f"received {observed_T}."
        )

    required = (
        "T",
        *POLICY_A0_COLUMNS,
        *UTILITY_VALUE_COLUMNS,
        *DIVERGENCE_FREQUENCY_COLUMNS,
    )
    missing = [
        column
        for column in required
        if any(column not in row for row in ordered)
    ]
    if missing:
        raise ValueError(f"Trajectory rows are missing plot columns: {missing}.")

    return {
        column: [row[column] for row in ordered]
        for column in required
    }


__all__ = [
    "ALL_RUNS_COLUMNS",
    "ALL_RUNS_NULLABLE",
    "ALL_RUNS_SCHEMA",
    "BEST_JOINT_ALLRUNS_COLUMNS",
    "BEST_JOINT_ALLRUNS_NULLABLE",
    "BEST_JOINT_ALLRUNS_SCHEMA",
    "BEST_JOINT_FINAL_COLUMNS",
    "BEST_JOINT_FINAL_NULLABLE",
    "BEST_JOINT_FINAL_SCHEMA",
    "FORBIDDEN_OUTPUT_FIELDS",
    "trajectory_rows_to_plot_data",
    "validate_no_forbidden_fields",
]
