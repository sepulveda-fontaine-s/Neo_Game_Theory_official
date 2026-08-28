'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario 1 run ranking and representative-run selection.

The approved selection rule is strictly lexicographic:

1. minimum ``D_JS_final``;
2. minimum ``absdiff_A_final``;
3. minimum ``absdiff_B_final``;
4. deterministic identifier tie-break.

No weighted sum or ``joint_score_final`` is constructed. Winner groups are
exactly ``H x eta_kind x random_states`` and therefore contain six selected
rows for each horizon. With the manuscript's single seed, selecting the best
run and selecting the best configuration are equivalent. If additional seeds
are supplied for robustness analysis, the selected run identifies the winning
``config_id`` and all runs of that configuration are retained for across-run
statistics and optional trajectory re-execution.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np

from general_formulation.inheritance import CHAIN_COLUMNS
from general_formulation.identifiers import (
    build_plot_filenames,
    build_plot_prefix,
)
from general_formulation.validation import (
    validate_one_selected_per_group,
    validate_selection_ranks,
    validate_unique_records,
)


SELECTION_FIELDS = (
    "D_JS_final",
    "absdiff_A_final",
    "absdiff_B_final",
)


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _require_fields(row: Mapping[str, Any], fields: Sequence[str], *, name: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}.")

def lexicographic_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the experimental alignment-ranking key used for Scenario 1.
    Part I describes this ranking in the simulation design, and Part II
    Eq. (53) [eq:lexicographic_selection] states it explicitly:
    D_JS_final, then absdiff_A_final, then absdiff_B_final. Deterministic
    identifiers are used only as final tie-breaks:   """

    _require_fields(
        row,
        (*SELECTION_FIELDS, "config_id", "run_id"),
        name="selection row",
    )
    return (
        _as_finite_float(row["D_JS_final"], name="D_JS_final"),
        _as_finite_float(row["absdiff_A_final"], name="absdiff_A_final"),
        _as_finite_float(row["absdiff_B_final"], name="absdiff_B_final"),
        str(row["config_id"]),
        str(row["run_id"]),
    )


def representative_run_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the representative-run key within one winning configuration."""
    _require_fields(row, (*SELECTION_FIELDS, "run_id"), name="run row")
    return (
        _as_finite_float(row["D_JS_final"], name="D_JS_final"),
        _as_finite_float(row["absdiff_A_final"], name="absdiff_A_final"),
        _as_finite_float(row["absdiff_B_final"], name="absdiff_B_final"),
        str(row["run_id"]),
    )


def rank_all_runs(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank every run within its winner group and mark exactly one selected row."""
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("records must contain at least one run.")

    validate_unique_records(
        rows,
        key_fields=("run_id",),
        record_name="all-runs",
    )
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _require_fields(row, ("winner_group_id",), name="all-runs row")
        groups[row["winner_group_id"]].append(row)

    ranked: list[dict[str, Any]] = []
    for group_id in sorted(groups, key=str):
        ordered = sorted(groups[group_id], key=lexicographic_selection_key)
        for rank, row in enumerate(ordered, start=1):
            annotated = dict(row)
            annotated["selection_rank"] = rank
            annotated["is_selected"] = rank == 1
            ranked.append(annotated)

    validate_selection_ranks(ranked)
    validate_one_selected_per_group(ranked)
    return ranked


def selected_run_rows(
    ranked_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the unique rank-1 row from every winner group."""
    rows = [dict(record) for record in ranked_records]
    validate_one_selected_per_group(rows)
    selected = [row for row in rows if row["is_selected"] is True]
    return sorted(selected, key=lambda row: str(row["winner_group_id"]))


def winning_configuration_rows(
    ranked_records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return all runs belonging to each selected ``config_id`` by group."""
    rows = [dict(record) for record in ranked_records]
    selected = selected_run_rows(rows)
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config_id"])].append(row)

    result: dict[str, list[dict[str, Any]]] = {}
    for winner in selected:
        group_id = str(winner["winner_group_id"])
        config_id = str(winner["config_id"])
        config_rows = by_config.get(config_id, [])
        if not config_rows:
            raise RuntimeError(f"No rows found for selected config_id={config_id!r}.")
        if any(str(row["winner_group_id"]) != group_id for row in config_rows):
            raise ValueError("A config_id cannot belong to multiple winner groups.")
        result[group_id] = sorted(config_rows, key=lambda row: str(row["run_id"]))
    return result


def select_representative_run(
    configuration_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the best run of one configuration using the approved tie-break."""
    rows = [dict(row) for row in configuration_rows]
    if not rows:
        raise ValueError("configuration_rows must not be empty.")
    config_ids = {str(row.get("config_id")) for row in rows}
    if len(config_ids) != 1:
        raise ValueError("configuration_rows must contain exactly one config_id.")
    return min(rows, key=representative_run_key)


def _population_mean_sd(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[float, float]:
    values = np.asarray(
        [_as_finite_float(row[field], name=field) for row in rows],
        dtype=np.float64,
    )
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _plot_fields(representative: Mapping[str, Any]) -> dict[str, str]:
    required = (
        "H",
        "eta_kind",
        "eta0",
        "c",
        "random_states",
        "pA",
        "alpha_agree",
        "alpha_disagree",
        "beta",
        "gamma",
        "seed",
    )
    _require_fields(representative, required, name="representative row")
    prefix = build_plot_prefix(
        {
            "H": representative["H"],
            "eta": representative["eta_kind"],
            "eta0": representative["eta0"],
            "c": representative["c"],
            "random_states": representative["random_states"],
            "pA": representative["pA"],
            "alpha_agree": representative["alpha_agree"],
            "alpha_disagree": representative["alpha_disagree"],
            "beta": representative["beta"],
            "gamma": representative["gamma"],
            "seed": representative["seed"],
        }
    )
    return {"plot_prefix": prefix, **build_plot_filenames(prefix)}


def build_best_joint_final_rows(
    ranked_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one compact winner/robustness/plot row per winner group."""
    rows = [dict(record) for record in ranked_records]
    selected_configs = winning_configuration_rows(rows)
    final_rows: list[dict[str, Any]] = []

    representative_fields = (
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
    )
    parameter_fields = (
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
    )

    for group_id in sorted(selected_configs):
        config_rows = selected_configs[group_id]
        representative = select_representative_run(config_rows)
        _require_fields(
            representative,
            (*parameter_fields, *representative_fields, "replicate", "seed", "run_id"),
            name="representative row",
        )

        final_row: dict[str, Any] = {
            field: representative[field] for field in parameter_fields
        }
        final_row.update(
            {
                "representative_run_id": representative["run_id"],
                "representative_replicate": representative["replicate"],
                "representative_seed": representative["seed"],
                **{
                    field: representative[field]
                    for field in representative_fields
                },
                "n_replicates": len(config_rows),
            }
        )

        robustness_map = {
            "D_JS_final": ("D_JS_final_mean", "D_JS_final_sd"),
            "absdiff_A_final": (
                "absdiff_A_final_mean",
                "absdiff_A_final_sd",
            ),
            "absdiff_B_final": (
                "absdiff_B_final_mean",
                "absdiff_B_final_sd",
            ),
            "f_H_final": ("f_H_final_mean", "f_H_final_sd"),
            "f_AI_final": ("f_AI_final_mean", "f_AI_final_sd"),
            "f_agree": ("f_agree_mean", "f_agree_sd"),
            "f_ctx": ("f_ctx_mean", "f_ctx_sd"),
            "f_disagree": ("f_disagree_mean", "f_disagree_sd"),
            "reward_structural_mean": (
                "reward_structural_mean_across_runs",
                "reward_structural_sd_across_runs",
            ),
        }
        for source_field, (mean_field, sd_field) in robustness_map.items():
            mean_value, sd_value = _population_mean_sd(config_rows, source_field)
            final_row[mean_field] = mean_value
            final_row[sd_field] = sd_value

        final_row.update(_plot_fields(representative))
        final_rows.append(final_row)

    validate_unique_records(
        final_rows,
        key_fields=("winner_group_id",),
        record_name="best-final",
    )
    return final_rows


def selected_configuration_run_ids(
    ranked_records: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return every run_id belonging to the six selected configurations."""
    selected_configs = winning_configuration_rows(ranked_records)
    return tuple(
        str(row["run_id"])
        for group_id in sorted(selected_configs)
        for row in selected_configs[group_id]
    )


__all__ = [
    "SELECTION_FIELDS",
    "build_best_joint_final_rows",
    "lexicographic_selection_key",
    "rank_all_runs",
    "representative_run_key",
    "select_representative_run",
    "selected_configuration_run_ids",
    "selected_run_rows",
    "winning_configuration_rows",
]
