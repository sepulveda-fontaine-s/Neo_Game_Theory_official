'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Shared plotting primitives for the four PNG artifacts produced by all scenarios.

The plotting contract follows the common simulation outputs:

- Plot 1a: four predecision ``a0`` policy probabilities;
- Plot 1b: the four complementary ``a1`` probabilities, derived as ``1-a0``;
- Plot 2: three post-EWMA utility-credit series and three post-backup values;
- Plot 3: predecision ``D_JS_T`` and cumulative Human/AI execution frequencies.

These plotting primitives are shared by Scenarios 1, 2, and 3.
Scenario-specific differences are reflected in the trajectories supplied to
the plotting functions, not in the plotting contract itself.

This module validates plotted values before drawing them. It never clips a
materially invalid probability merely to make a figure render.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .numerics import DEFAULT_ATOL


POLICY_A0_COLUMNS: Final[tuple[str, ...]] = (
    "p_H_A_T",
    "p_AI_A_T",
    "p_H_B_T",
    "p_AI_B_T",
)
UTILITY_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "Uhat_H_realized_T",
    "Uhat_AI_realized_T",
    "Uhat_coal_T",
    "Vhat_A_T",
    "Vhat_B_T",
    "Vhat_coal_T",
)
DIVERGENCE_FREQUENCY_COLUMNS: Final[tuple[str, ...]] = (
    "D_JS_T",
    "f_H_T",
    "f_AI_T",
)


class PlotDataError(ValueError):
    """Raised when longitudinal data cannot support a requested plot."""


def _validated_atol(atol: float) -> float:
    try:
        tolerance = float(atol)
    except (TypeError, ValueError) as exc:
        raise TypeError("atol must be a finite nonnegative float.") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be a finite nonnegative float.")
    return tolerance


def _get_column(data: Mapping[str, Any], column: str) -> Any:
    try:
        return data[column]
    except (KeyError, TypeError) as exc:
        raise PlotDataError(f"Missing required plot column {column!r}.") from exc


def _numeric_series(
    data: Mapping[str, Any],
    column: str,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    raw = _get_column(data, column)
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PlotDataError(f"Column {column!r} must be numeric.") from exc

    if values.ndim != 1:
        raise PlotDataError(f"Column {column!r} must be one-dimensional.")
    if values.size == 0:
        raise PlotDataError(f"Column {column!r} must not be empty.")
    if expected_length is not None and values.size != expected_length:
        raise PlotDataError(
            f"Column {column!r} has length {values.size}; expected {expected_length}."
        )
    if not np.all(np.isfinite(values)):
        raise PlotDataError(f"Column {column!r} contains non-finite values.")
    return values.copy()


def _time_series(data: Mapping[str, Any], *, column: str = "T") -> np.ndarray:
    T = _numeric_series(data, column)
    if np.any(np.diff(T) <= 0.0):
        raise PlotDataError(f"Column {column!r} must be strictly increasing.")
    if np.any(T < 0.0):
        raise PlotDataError(f"Column {column!r} must be nonnegative.")
    return T


def _probability_series(
    data: Mapping[str, Any],
    column: str,
    *,
    expected_length: int,
    atol: float,
) -> np.ndarray:
    values = _numeric_series(data, column, expected_length=expected_length)
    tolerance = _validated_atol(atol)
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        raise PlotDataError(
            f"Column {column!r} must lie in [0, 1]; observed range "
            f"[{minimum}, {maximum}]."
        )

    # Correct only boundary-level floating-point drift after validation.
    values[values < 0.0] = 0.0
    values[values > 1.0] = 1.0
    return values


def _output_path(out_png: str | Path) -> Path:
    path = Path(out_png)
    if path.suffix.lower() != ".png":
        raise ValueError("Plot output filename must use the .png extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def marker_layout(length: int, series_count: int) -> tuple[int, tuple[int, ...]]:
    """Return a marker step and distinct offsets for overlapping series."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer.")
    if (
        isinstance(series_count, bool)
        or not isinstance(series_count, int)
        or series_count <= 0
    ):
        raise ValueError("series_count must be a positive integer.")

    marker_step = max(series_count + 1, length // 40)
    offsets = tuple(
        min(marker_step - 1, int(round(index * marker_step / series_count)))
        for index in range(series_count)
    )
    return marker_step, offsets


def _draw_lines(
    ax: Any,
    T: np.ndarray,
    series: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    linestyles: Sequence[Any],
    markers: Sequence[str],
    linewidths: Sequence[float],
) -> None:
    count = len(series)
    if not (
        len(labels) == len(linestyles) == len(markers) == len(linewidths) == count
    ):
        raise RuntimeError("Plot style specifications must match the series count.")

    marker_step, offsets = marker_layout(len(T), count)
    for index, values in enumerate(series):
        marker_kwargs: dict[str, Any] = {
            "marker": markers[index],
            "markevery": (offsets[index], marker_step),
            "markersize": 6,
            "markeredgewidth": 1.3,
        }
        if markers[index] not in {"x", "+"}:
            marker_kwargs["markerfacecolor"] = "none"

        ax.plot(
            T,
            values,
            label=labels[index],
            linewidth=linewidths[index],
            linestyle=linestyles[index],
            alpha=0.92,
            zorder=index + 1,
            **marker_kwargs,
        )


def _finish_and_save(
    fig: Any,
    ax: Any,
    out_png: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    y_limits: tuple[float, float] | None,
    legend_columns: int = 1,
    dpi: int = 200,
) -> Path:
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    output = _output_path(out_png)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=13, ncol=legend_columns, markerscale=1.4)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_policy_a0(
    data: Mapping[str, Any],
    out_png: str | Path,
    *,
    atol: float = DEFAULT_ATOL,
    dpi: int = 200,
) -> Path:
    """Create Plot 1a from the four predecision ``a0`` policy series."""
    T = _time_series(data)
    series = [
        _probability_series(data, column, expected_length=len(T), atol=atol)
        for column in POLICY_A0_COLUMNS
    ]
    labels = [
        r"$p_H(a_0\mid A,T)$",
        r"$p_{AI}(a_0\mid A,T)$",
        r"$p_H(a_0\mid B,T)$",
        r"$p_{AI}(a_0\mid B,T)$",
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    _draw_lines(
        ax,
        T,
        series,
        labels,
        linestyles=("-", (0, (9, 4)), "-.", (0, (2, 3))),
        markers=("o", "s", "^", "D"),
        linewidths=(3.6, 3.0, 2.4, 1.9),
    )
    return _finish_and_save(
        fig,
        ax,
        out_png,
        xlabel=r"Decision instant $T$",
        ylabel=r"Predecision probability of $a_0$",
        y_limits=(-0.02, 1.02),
        dpi=dpi,
    )


def plot_policy_a1(
    data: Mapping[str, Any],
    out_png: str | Path,
    *,
    atol: float = DEFAULT_ATOL,
    dpi: int = 200,
) -> Path:
    """Create Plot 1b by deriving every ``a1`` series as ``1 - p(a0)``."""
    T = _time_series(data)
    a0_series = [
        _probability_series(data, column, expected_length=len(T), atol=atol)
        for column in POLICY_A0_COLUMNS
    ]
    series = [1.0 - values for values in a0_series]
    labels = [
        r"$p_H(a_1\mid A,T)$",
        r"$p_{AI}(a_1\mid A,T)$",
        r"$p_H(a_1\mid B,T)$",
        r"$p_{AI}(a_1\mid B,T)$",
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    _draw_lines(
        ax,
        T,
        series,
        labels,
        linestyles=("-", (0, (9, 4)), "-.", (0, (2, 3))),
        markers=("o", "s", "^", "D"),
        linewidths=(3.6, 3.0, 2.4, 1.9),
    )
    return _finish_and_save(
        fig,
        ax,
        out_png,
        xlabel=r"Decision instant $T$",
        ylabel=r"Derived predecision probability of $a_1$",
        y_limits=(-0.02, 1.02),
        dpi=dpi,
    )


def plot_utility_credit_and_value(
    data: Mapping[str, Any],
    out_png: str | Path,
    *,
    atol: float = DEFAULT_ATOL,
    dpi: int = 200,
) -> Path:
    """Create Plot with the shared utility-credit/value symbology.

    Utility-credit traces use the left axis: Human and AI are solid lines and
    the realized coalition trace is shown with red ``x`` markers. Computational
    values use the right axis and are shown as circular markers for states
    ``A``, ``B``, and the realized coalition value.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    T = _time_series(data)
    Uhat_H, Uhat_AI, Uhat_coal = [
        _probability_series(data, column, expected_length=len(T), atol=atol)
        for column in UTILITY_VALUE_COLUMNS[:3]
    ]
    Vhat_A, Vhat_B, Vhat_coal = [
        _numeric_series(data, column, expected_length=len(T))
        for column in UTILITY_VALUE_COLUMNS[3:]
    ]

    def axis_limits(series: Sequence[np.ndarray]) -> tuple[float, float]:
        values = np.concatenate(series)
        lower = min(0.0, float(np.min(values)))
        upper = float(np.max(values))
        span = upper - lower
        padding = 0.05 * span if span > 0.0 else 0.05
        return lower - padding, upper + padding

    fig, ax_utility = plt.subplots(figsize=(14, 7))
    ax_value = ax_utility.twinx()
    marker_step, marker_offsets = marker_layout(len(T), 4)

    utility_handles = [
        ax_utility.plot(
            T,
            Uhat_H,
            color="tab:blue",
            linewidth=1.2,
            label=r"$\widehat U_H(s_T,a_T^*)$",
        )[0],
        ax_utility.plot(
            T,
            Uhat_AI,
            color="tab:orange",
            linewidth=1.2,
            label=r"$\widehat U_{AI}(s_T,a_T^*)$",
        )[0],
        ax_utility.plot(
            T,
            Uhat_coal,
            linestyle="None",
            marker="x",
            color="red",
            markersize=6.5,
            markeredgewidth=1.2,
            markevery=(marker_offsets[0], marker_step),
            label=r"$\widehat U_{coal}(T)$",
        )[0],
    ]

    value_handles = [
        ax_value.plot(
            T,
            Vhat_A,
            linestyle="None",
            marker="o",
            color="yellow",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=4.0,
            markevery=(marker_offsets[1], marker_step),
            label=r"$\widehat V(A)$",
        )[0],
        ax_value.plot(
            T,
            Vhat_B,
            linestyle="None",
            marker="o",
            color="purple",
            markersize=4.0,
            markevery=(marker_offsets[2], marker_step),
            label=r"$\widehat V(B)$",
        )[0],
        ax_value.plot(
            T,
            Vhat_coal,
            linestyle="None",
            marker="o",
            color="lightgreen",
            markersize=4.0,
            markevery=(marker_offsets[3], marker_step),
            label=r"$\widehat V_{coal}(T)$",
        )[0],
    ]

    ax_utility.set_xlabel("Iteration", fontsize=18)
    ax_utility.set_ylabel("Utility estimates", fontsize=18)
    ax_value.set_ylabel("Value estimates", fontsize=18)
    ax_utility.set_ylim(*axis_limits((Uhat_H, Uhat_AI, Uhat_coal)))
    ax_value.set_ylim(*axis_limits((Vhat_A, Vhat_B, Vhat_coal)))
    ax_utility.tick_params(axis="both", labelsize=13)
    ax_value.tick_params(axis="y", labelsize=13)
    ax_utility.grid(True, alpha=0.25)

    handles = utility_handles + value_handles
    labels = [handle.get_label() for handle in handles]
    ax_utility.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=13,
        markerscale=1.4,
    )

    output = _output_path(out_png)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_divergence_and_execution_frequencies(
    data: Mapping[str, Any],
    out_png: str | Path,
    *,
    alpha_agree: float | None = None,
    alpha_disagree: float | None = None,
    atol: float = DEFAULT_ATOL,
    dpi: int = 200,
) -> Path:
    """Create Plot 3 from ``D_JS_T``, ``f_H_T``, and ``f_AI_T``.

    The y-axis remains linear on ``[0, 1]``. This preserves legitimate zeros
    and avoids the artificial epsilon replacement required by a logarithmic
    axis.
    """
    T = _time_series(data)
    D_JS_T, f_H_T, f_AI_T = [
        _probability_series(data, column, expected_length=len(T), atol=atol)
        for column in DIVERGENCE_FREQUENCY_COLUMNS
    ]
    tolerance = _validated_atol(atol)
    if not np.allclose(f_H_T + f_AI_T, 1.0, rtol=0.0, atol=tolerance):
        maximum_error = float(np.max(np.abs(f_H_T + f_AI_T - 1.0)))
        raise PlotDataError(
            "Execution frequencies must satisfy f_H_T + f_AI_T = 1; "
            f"maximum error is {maximum_error}."
        )

    fig, ax = plt.subplots(figsize=(14, 7))
    _draw_lines(
        ax,
        T,
        (D_JS_T, f_H_T, f_AI_T),
        (r"$D_{JS}^{T}$", r"$f_H(T)$", r"$f_{AI}(T)$"),
        linestyles=("-.", "-", (0, (8, 4))),
        markers=("^", "o", "s"),
        linewidths=(3.6, 2.8, 2.1),
    )

    if alpha_agree is not None:
        alpha_agree_value = float(alpha_agree)
        if not 0.0 <= alpha_agree_value <= 1.0:
            raise ValueError("alpha_agree must lie in [0, 1].")
        ax.axhline(
            alpha_agree_value,
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            label=r"$\alpha_{agree}$",
        )

    if alpha_disagree is not None:
        alpha_disagree_value = float(alpha_disagree)
        if not 0.0 <= alpha_disagree_value <= 1.0:
            raise ValueError("alpha_disagree must lie in [0, 1].")
        ax.axhline(
            alpha_disagree_value,
            linestyle=(0, (1, 4)),
            linewidth=1.2,
            label=r"$\alpha_{disagree}$",
        )

    return _finish_and_save(
        fig,
        ax,
        out_png,
        xlabel=r"Decision instant $T$",
        ylabel="Divergence and cumulative execution frequency",
        y_limits=(-0.02, 1.02),
        dpi=dpi,
    )


__all__ = [
    "DIVERGENCE_FREQUENCY_COLUMNS",
    "POLICY_A0_COLUMNS",
    "UTILITY_VALUE_COLUMNS",
    "PlotDataError",
    "marker_layout",
    "plot_divergence_and_execution_frequencies",
    "plot_policy_a0",
    "plot_policy_a1",
    "plot_utility_credit_and_value",
]
