'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""End-to-end orchestration for Scenario 1: Human arbitration.

Run from the repository root with::

    python -m Scenario_1.main

The workflow is deliberately split into two phases:

1. evaluate the complete active grid and retain one summary row per run;
2. rank runs lexicographically, rerun only the selected configurations with
   longitudinal recording, generate four plots per representative run, and
   write the contracted CSV outputs.

The second phase is a deterministic re-execution using the same validated
configuration and seed. A reproducibility check compares the rerun summary
against the Phase-1 summary before any trajectory is accepted.
"""

from __future__ import annotations
import argparse
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from general_formulation.csv_outputs import CsvRowWriter, write_csv_rows
from general_formulation.plots import (
    plot_divergence_and_execution_frequencies,
    plot_policy_a0,
    plot_policy_a1,
    plot_utility_credit_and_value,
)
from general_formulation.validation import (
    validate_one_selected_per_group,
    validate_plot_references,
    validate_unique_records,
)

from Scenario_1.config import (
    ACTIVE_H,
    ALL_RUNS_PATH,
    ATOL,
    BEST_JOINT_ALLRUNS_PATH,
    BEST_JOINT_FINAL_PATH,
    EXPECTED_WINNER_GROUPS,
    OUTPUT_DIR,
    PLOT_DPI,
    PROGRESS_INTERVAL,
)
from Scenario_1.utils_scenario_1.grid import (
    Scenario1RunConfig,
    iter_scenario1_grid,
)
from Scenario_1.utils_scenario_1.output_schema import (
    ALL_RUNS_SCHEMA,
    BEST_JOINT_ALLRUNS_SCHEMA,
    BEST_JOINT_FINAL_SCHEMA,
    validate_no_forbidden_fields,
)
from Scenario_1.utils_scenario_1.selection import (
    build_best_joint_final_rows,
    rank_all_runs,
    selected_configuration_run_ids,
)
from Scenario_1.utils_scenario_1.simulation import (
    SimulationResult,
    simulate_scenario1,
)


ConfigurationFactory = Callable[[], Iterable[Scenario1RunConfig]]


@dataclass(frozen=True, slots=True)
class Scenario1Artifacts:
    """Paths and cardinalities produced by one complete pipeline run."""

    all_runs_csv: Path
    best_joint_allruns_csv: Path
    best_joint_final_csv: Path
    best_joint_final_named_csv: Path
    output_dir: Path
    run_count: int
    selected_run_count: int
    winner_count: int
    plot_count: int
    elapsed_seconds: float


# Summary fields that must be identical after deterministic re-execution.
_REEXECUTION_FIELDS = (
    "D_JS_final",
    "D_JS_A_final",
    "D_JS_B_final",
    "absdiff_A_final",
    "absdiff_B_final",
    "p_H_A_final",
    "p_AI_A_final",
    "p_H_B_final",
    "p_AI_B_final",
    "reward_structural_final",
    "reward_structural_mean",
    "reward_structural_cumulative",
    "Uhat_H_realized_final",
    "Uhat_AI_realized_final",
    "Uhat_coal_final",
    "Vhat_A_final",
    "Vhat_B_final",
    "Vhat_coal_final",
    "n_H_exec",
    "n_AI_exec",
    "n_agree",
    "n_ctx",
    "n_disagree",
    "state_final",
    "a_H_final",
    "a_AI_final",
    "a_star_final",
    "owner_final",
    "lambda_final",
    "regime_final",
)


def iter_active_configurations(
    *,
    H: int = ACTIVE_H,
) -> Iterator[Scenario1RunConfig]:
    """Yield the complete validated grid for one horizon."""
    if isinstance(H, bool) or not isinstance(H, int) or H <= 0:
        raise ValueError("H must be a strictly positive integer.")

    yield from iter_scenario1_grid(H=H, atol=ATOL)


def _phase1_summary_rows(
    configuration_factory: ConfigurationFactory,
    *,
    progress_interval: int,
) -> list[dict[str, Any]]:
    """Run the exhaustive phase without retaining longitudinal trajectories."""
    if progress_interval < 0:
        raise ValueError("progress_interval must be nonnegative.")

    rows: list[dict[str, Any]] = []
    for run_number, config in enumerate(configuration_factory(), start=1):
        result = simulate_scenario1(
            config,
            return_trajectory=False,
            atol=ATOL,
        )
        if result.trajectory_rows:
            raise RuntimeError(
                "Phase 1 must not retain longitudinal trajectory rows."
            )
        rows.append(result.summary_row)

        if progress_interval and run_number % progress_interval == 0:
            print(f"Phase 1: completed {run_number:,}/17,640 runs.")

    if not rows:
        raise RuntimeError("The active Scenario 1 grid produced no runs.")

    validate_unique_records(
        rows,
        key_fields=("run_id",),
        record_name="phase-1 summary",
    )
    validate_no_forbidden_fields(rows)
    return rows


def _equal_reexecution_value(
    phase1_value: Any,
    phase2_value: Any,
    *,
    atol: float,
) -> bool:
    """Compare one deterministic rerun field without hiding type mismatches."""
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(phase1_value, bool) or isinstance(phase2_value, bool):
        return phase1_value is phase2_value
    if isinstance(phase1_value, numeric_types) and isinstance(
        phase2_value,
        numeric_types,
    ):
        return bool(
            np.isclose(
                float(phase1_value),
                float(phase2_value),
                rtol=0.0,
                atol=atol,
            )
        )
    return phase1_value == phase2_value


def _validate_reexecution(
    phase1_row: Mapping[str, Any],
    phase2_result: SimulationResult,
    *,
    atol: float,
) -> None:
    """Require the selected rerun to reproduce its Phase-1 summary."""
    phase2_row = phase2_result.summary_row
    if phase1_row.get("run_id") != phase2_row.get("run_id"):
        raise RuntimeError("Phase-1 and Phase-2 run_id values do not match.")

    for field in _REEXECUTION_FIELDS:
        if field not in phase1_row or field not in phase2_row:
            raise RuntimeError(
                f"Reexecution comparison is missing field {field!r}."
            )
        if not _equal_reexecution_value(
            phase1_row[field],
            phase2_row[field],
            atol=atol,
        ):
            raise RuntimeError(
                "Deterministic reexecution mismatch for "
                f"run_id={phase1_row['run_id']!r}, field={field!r}: "
                f"phase1={phase1_row[field]!r}, "
                f"phase2={phase2_row[field]!r}."
            )


def _generate_representative_plots(
    final_row: Mapping[str, Any],
    plot_data: Mapping[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Generate the four plots for one representative run."""
    return (
        plot_policy_a0(
            plot_data,
            output_dir / str(final_row["plot1_a0_file"]),
            atol=ATOL,
            dpi=PLOT_DPI,
        ),
        plot_policy_a1(
            plot_data,
            output_dir / str(final_row["plot1_a1_file"]),
            atol=ATOL,
            dpi=PLOT_DPI,
        ),
        plot_utility_credit_and_value(
            plot_data,
            output_dir / str(final_row["plot2_file"]),
            atol=ATOL,
            dpi=PLOT_DPI,
        ),
        plot_divergence_and_execution_frequencies(
            plot_data,
            output_dir / str(final_row["plot3_file"]),
            alpha_agree=float(final_row["alpha_agree"]),
            alpha_disagree=float(final_row["alpha_disagree"]),
            atol=ATOL,
            dpi=PLOT_DPI,
        ),
    )


def _phase2_stream_selected_runs(
    configuration_factory: ConfigurationFactory,
    *,
    ranked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    trajectory_csv_path: Path,
    plots_directory: Path,
) -> tuple[int, tuple[Path, ...]]:
    """Rerun selected configurations and stream longitudinal rows to CSV.

    Only the compact plot columns of the current representative run are held in
    memory.  Full trajectory dictionaries are validated and written one at a
    time to an atomic temporary CSV file.
    """
    selected_run_ids = set(selected_configuration_run_ids(ranked_rows))
    if not selected_run_ids:
        raise RuntimeError("Winner selection produced no selected run IDs.")

    phase1_by_run_id = {
        str(row["run_id"]): row
        for row in ranked_rows
        if str(row["run_id"]) in selected_run_ids
    }
    if set(phase1_by_run_id) != selected_run_ids:
        raise RuntimeError("Selected run IDs are incomplete in Phase-1 results.")

    representative_rows = {
        str(row["representative_run_id"]): row
        for row in final_rows
    }
    if not set(representative_rows).issubset(selected_run_ids):
        raise RuntimeError(
            "Every representative run must belong to a selected configuration."
        )

    found_run_ids: set[str] = set()
    plotted_run_ids: set[str] = set()
    generated_plots: list[Path] = []

    try:
        with CsvRowWriter(
            trajectory_csv_path,
            BEST_JOINT_ALLRUNS_SCHEMA,
            atomic=True,
        ) as trajectory_writer:
            for config in configuration_factory():
                if config.run_id not in selected_run_ids:
                    continue

                is_representative = config.run_id in representative_rows
                row_count_before = trajectory_writer.row_count
                result = simulate_scenario1(
                    config,
                    return_trajectory=False,
                    trajectory_sink=trajectory_writer.write_row,
                    collect_plot_data=is_representative,
                    atol=ATOL,
                )
                _validate_reexecution(
                    phase1_by_run_id[config.run_id],
                    result,
                    atol=ATOL,
                )

                if result.trajectory_rows:
                    raise RuntimeError(
                        "Streaming Phase 2 must not retain trajectory dictionaries."
                    )
                produced_rows = trajectory_writer.row_count - row_count_before
                if produced_rows != config.H:
                    raise RuntimeError(
                        f"run_id={config.run_id!r} streamed {produced_rows} "
                        f"trajectory rows; expected H={config.H}."
                    )

                if is_representative:
                    if result.plot_data is None:
                        raise RuntimeError(
                            f"Representative run {config.run_id!r} produced no plot data."
                        )
                    if len(result.plot_data.get("T", ())) != config.H:
                        raise RuntimeError(
                            f"Representative run {config.run_id!r} has an invalid "
                            "plot-data length."
                        )
                    paths = _generate_representative_plots(
                        representative_rows[config.run_id],
                        result.plot_data,
                        output_dir=plots_directory,
                    )
                    generated_plots.extend(paths)
                    plotted_run_ids.add(config.run_id)

                found_run_ids.add(config.run_id)

            missing = selected_run_ids - found_run_ids
            if missing:
                raise RuntimeError(
                    "The second grid pass did not reconstruct selected run IDs: "
                    f"{sorted(missing)}."
                )
            missing_plots = set(representative_rows) - plotted_run_ids
            if missing_plots:
                raise RuntimeError(
                    "Representative runs were not reconstructed for plotting: "
                    f"{sorted(missing_plots)}."
                )

        if len(set(generated_plots)) != len(generated_plots):
            raise RuntimeError("Plot generation produced duplicate output paths.")
    except Exception:
        for plot_path in generated_plots:
            try:
                plot_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return len(found_run_ids), tuple(generated_plots)


def _best_joint_final_named_path(
    output_directory: Path,
    final_rows: list[dict[str, Any]],
    *,
    scenario_token: str,
) -> Path:
    """Return metadata-rich best-joint-final CSV path.

    The legacy filename is still written for inherited-chain compatibility.
    """
    if not final_rows:
        raise RuntimeError("final_rows must not be empty.")
    H_values = {int(row["H"]) for row in final_rows}
    if len(H_values) != 1:
        raise RuntimeError("best_joint_final rows must have one unique H.")
    H = H_values.pop()
    return output_directory / f"scenario1_best_joint_final_H_{H}_{scenario_token}.csv"


def run_scenario1_pipeline(
    *,
    configuration_factory: ConfigurationFactory = iter_active_configurations,
    output_dir: Path = OUTPUT_DIR,
    progress_interval: int = PROGRESS_INTERVAL,
    expected_winner_groups: int = EXPECTED_WINNER_GROUPS,
) -> Scenario1Artifacts:
    """Execute, validate, and persist the complete two-phase Scenario 1 flow."""
    start = perf_counter()
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    plots_directory = output_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    phase1_rows = _phase1_summary_rows(
        configuration_factory,
        progress_interval=progress_interval,
    )
    ranked_rows = rank_all_runs(phase1_rows)
    validate_one_selected_per_group(
        ranked_rows,
        expected_group_count=expected_winner_groups,
    )
    validate_no_forbidden_fields(ranked_rows)

    final_rows = build_best_joint_final_rows(ranked_rows)
    if len(final_rows) != expected_winner_groups:
        raise RuntimeError(
            f"Expected {expected_winner_groups} final winner rows, "
            f"received {len(final_rows)}."
        )
    validate_no_forbidden_fields(final_rows)

    all_runs_path = output_directory / ALL_RUNS_PATH.name
    best_allruns_path = output_directory / BEST_JOINT_ALLRUNS_PATH.name
    best_final_path = output_directory / BEST_JOINT_FINAL_PATH.name

    selected_run_count, generated_plots = _phase2_stream_selected_runs(
        configuration_factory,
        ranked_rows=ranked_rows,
        final_rows=final_rows,
        trajectory_csv_path=best_allruns_path,
        plots_directory=plots_directory,
    )

    expected_plot_count = expected_winner_groups * 4
    validate_plot_references(
        final_rows,
        output_dir=plots_directory,
        expected_file_count=expected_plot_count,
    )
    if len(generated_plots) != expected_plot_count:
        raise RuntimeError(
            f"Expected {expected_plot_count} generated plots, "
            f"received {len(generated_plots)}."
        )

    best_final_named_path = _best_joint_final_named_path(
        output_directory, final_rows, scenario_token="Sc1"
    )

    write_csv_rows(all_runs_path, ranked_rows, ALL_RUNS_SCHEMA)
    write_csv_rows(best_final_path, final_rows, BEST_JOINT_FINAL_SCHEMA)
    write_csv_rows(best_final_named_path, final_rows, BEST_JOINT_FINAL_SCHEMA)

    elapsed = perf_counter() - start
    return Scenario1Artifacts(
        all_runs_csv=all_runs_path,
        best_joint_allruns_csv=best_allruns_path,
        best_joint_final_csv=best_final_path,
        best_joint_final_named_csv=best_final_named_path,
        output_dir=output_directory,
        run_count=len(ranked_rows),
        selected_run_count=selected_run_count,
        winner_count=len(final_rows),
        plot_count=len(generated_plots),
        elapsed_seconds=elapsed,
    )

def _positive_horizon(value: str) -> int:
    """Parse a strictly positive command-line horizon."""
    try:
        H = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "H must be an integer."
        ) from exc

    if H <= 0:
        raise argparse.ArgumentTypeError(
            "H must be strictly positive."
        )

    return H

def main(argv: list[str] | None = None) -> int:
    """Run Scenario 1 for the requested horizon."""
    parser = argparse.ArgumentParser(
        description="Run the complete Scenario 1 experiment."
    )
    parser.add_argument(
        "--H",
        "--horizon",
        dest="H",
        type=_positive_horizon,
        default=ACTIVE_H,
        help=(
            "Simulation horizon. "
            f"Default from config.py: {ACTIVE_H}."
        ),
    )
    args = parser.parse_args(argv)
    H = args.H

    output_directory = (
        Path(__file__).resolve().parent
        / f"Scenario1_outputs_H_{H}"
    )

    artifacts = run_scenario1_pipeline(
        configuration_factory=lambda: iter_active_configurations(H=H),
        output_dir=output_directory,
    )

    print("Scenario 1 completed successfully.")
    print(f"  H: {H}")
    print(f"  exhaustive runs: {artifacts.run_count:,}")
    print(f"  selected reruns: {artifacts.selected_run_count:,}")
    print(f"  winner rows: {artifacts.winner_count:,}")
    print(f"  plots: {artifacts.plot_count:,}")
    print(f"  output directory: {artifacts.output_dir}")
    print(f"  best final CSV: {artifacts.best_joint_final_csv}")
    print(f"  best final CSV with H/Sc: {artifacts.best_joint_final_named_csv}")
    print(f"  elapsed seconds: {artifacts.elapsed_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfigurationFactory",
    "Scenario1Artifacts",
    "iter_active_configurations",
    "main",
    "run_scenario1_pipeline",
]
