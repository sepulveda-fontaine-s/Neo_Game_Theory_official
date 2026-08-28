'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''



"""General mathematical, temporal, probability, and output validations.

The functions in this module enforce identities shared by the simulation and
its CSV/plot outputs. Scenario-specific delegation and winner-selection rules
remain in the scenario-specific modules.

Part I basis
------------
- Part I Eqs. (11)–(15): threshold order and contextual complements.
- Part I Equation (26): regime frequencies form a partition.
- Part I Equation (35) and Proposition 3: policy vectors remain in the simplex.
- Plot 3 contract: Human and AI execution frequencies sum to one.

Output-specific checks such as exact selected-row counts and plot references
are implementation safeguards and therefore have no independent equation
number in Part I.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .numerics import DEFAULT_ATOL, as_probability, as_probability_vector


class ModelValidationError(ValueError):
    """Raised when a mathematical or output invariant is violated."""


def _validated_atol(atol: float) -> float:
    try:
        tolerance = float(atol)
    except (TypeError, ValueError) as exc:
        raise TypeError("atol must be a finite nonnegative float.") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be a finite nonnegative float.")
    return tolerance


def _as_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ModelValidationError(
            f"{name} must be nonnegative; received {value!r}."
        )
    return value


def _as_positive_integer(value: int, *, name: str) -> int:
    validated = _as_nonnegative_integer(value, name=name)
    if validated == 0:
        raise ModelValidationError(f"{name} must be positive.")
    return validated


def assert_close(
    actual: float,
    expected: float,
    *,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Raise when two finite scalars differ beyond absolute tolerance."""
    tolerance = _validated_atol(atol)
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} values must be numeric.") from exc

    if not math.isfinite(actual_value) or not math.isfinite(expected_value):
        raise ModelValidationError(f"{name} values must be finite.")
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ModelValidationError(
            f"{name} mismatch: observed {actual_value!r}, expected "
            f"{expected_value!r}, tolerance {tolerance!r}."
        )


def validate_thresholds(
    alpha_agree: float,
    alpha_disagree: float,
    *,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, float]:
    """Validate ``0 <= alpha_agree < alpha_disagree <= 1``."""
    agreement = as_probability(alpha_agree, name="alpha_agree", atol=atol)
    disagreement = as_probability(
        alpha_disagree,
        name="alpha_disagree",
        atol=atol,
    )
    if not agreement < disagreement:
        raise ModelValidationError(
            "Delegation thresholds must satisfy alpha_agree < "
            f"alpha_disagree; received {agreement!r} and {disagreement!r}."
        )
    return agreement, disagreement


def validate_probability_complement(
    probability: float,
    complement: float,
    *,
    probability_name: str = "probability",
    complement_name: str = "complement",
    atol: float = DEFAULT_ATOL,
) -> tuple[float, float]:
    """Validate two probabilities and their sum-to-one identity."""
    first = as_probability(probability, name=probability_name, atol=atol)
    second = as_probability(complement, name=complement_name, atol=atol)
    assert_close(
        first + second,
        1.0,
        name=f"{probability_name} + {complement_name}",
        atol=atol,
    )
    return first, second


def validate_policy_pair(
    p_H: ArrayLike,
    p_AI: ArrayLike,
    *,
    atol: float = DEFAULT_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate Human and AI policies on one common finite action support."""
    p_H_vector = as_probability_vector(p_H, name="p_H", atol=atol)
    p_AI_vector = as_probability_vector(p_AI, name="p_AI", atol=atol)
    if p_H_vector.shape != p_AI_vector.shape:
        raise ModelValidationError(
            "p_H and p_AI must use the same action support; received shapes "
            f"{p_H_vector.shape} and {p_AI_vector.shape}."
        )
    return p_H_vector, p_AI_vector


def validate_frequency_partition(
    frequencies: Mapping[str, float] | Sequence[float],
    *,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, ...]:
    """Validate that finite frequencies are individually valid and sum to one."""
    if isinstance(frequencies, Mapping):
        items = list(frequencies.items())
    else:
        items = [(str(index), value) for index, value in enumerate(frequencies)]

    if not items:
        raise ModelValidationError(f"{name} must contain at least one frequency.")

    validated = tuple(
        as_probability(value, name=f"{name}[{label}]", atol=atol)
        for label, value in items
    )
    assert_close(sum(validated), 1.0, name=f"sum({name})", atol=atol)
    return validated


def validate_execution_frequencies(
    f_H_T: float,
    f_AI_T: float,
    *,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, float]:
    """Validate the Plot 3 identity ``f_H_T + f_AI_T = 1``."""
    return validate_probability_complement(
        f_H_T,
        f_AI_T,
        probability_name="f_H_T",
        complement_name="f_AI_T",
        atol=atol,
    )


def validate_regime_frequencies(
    f_agree_T: float,
    f_ctx_T: float,
    f_disagree_T: float,
    *,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, float, float]:
    """Validate Part I Eq. (26) [freq_summa]."""
    validated = validate_frequency_partition(
        {
            "agreement": f_agree_T,
            "contextual": f_ctx_T,
            "disagreement": f_disagree_T,
        },
        name="regime_frequencies",
        atol=atol,
    )
    return validated[0], validated[1], validated[2]


def validate_count_partition(
    counts: Mapping[str, int] | Sequence[int],
    *,
    total: int,
    name: str,
) -> tuple[int, ...]:
    """Validate nonnegative integer counts and their exact total."""
    total_value = _as_nonnegative_integer(total, name="total")
    if isinstance(counts, Mapping):
        items = list(counts.items())
    else:
        items = [(str(index), value) for index, value in enumerate(counts)]
    if not items:
        raise ModelValidationError(f"{name} must contain at least one count.")

    validated = tuple(
        _as_nonnegative_integer(value, name=f"{name}[{label}]")
        for label, value in items
    )
    observed_total = sum(validated)
    if observed_total != total_value:
        raise ModelValidationError(
            f"{name} sums to {observed_total}, expected total {total_value}."
        )
    return validated


def validate_absdiff(
    left: float,
    right: float,
    reported_absdiff: float,
    *,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Validate a reported absolute policy difference."""
    left_value = as_probability(left, name=f"{name}_left", atol=atol)
    right_value = as_probability(right, name=f"{name}_right", atol=atol)
    reported = as_probability(reported_absdiff, name=name, atol=atol)
    assert_close(
        reported,
        abs(left_value - right_value),
        name=name,
        atol=atol,
    )
    return reported


def validate_unique_records(
    records: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    record_name: str,
) -> None:
    """Reject duplicate composite keys in an iterable of mappings."""
    if not key_fields:
        raise ValueError("key_fields must not be empty.")
    if any(not isinstance(field, str) or not field for field in key_fields):
        raise ValueError("Every key field must be a non-empty string.")

    seen: dict[tuple[Any, ...], int] = {}
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"{record_name} record {row_number} must be a mapping.")
        missing = [field for field in key_fields if field not in record]
        if missing:
            raise ModelValidationError(
                f"{record_name} record {row_number} is missing key fields {missing}."
            )
        key = tuple(record[field] for field in key_fields)
        if key in seen:
            raise ModelValidationError(
                f"Duplicate {record_name} key {key!r} at records "
                f"{seen[key]} and {row_number}."
            )
        seen[key] = row_number


def validate_one_selected_per_group(
    records: Iterable[Mapping[str, Any]],
    *,
    group_field: str = "winner_group_id",
    selected_field: str = "is_selected",
    expected_group_count: int | None = None,
) -> None:
    """Validate exactly one selected record in every winner group."""
    rows = list(records)
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row_number, record in enumerate(rows, start=1):
        if group_field not in record or selected_field not in record:
            raise ModelValidationError(
                f"Record {row_number} must contain {group_field!r} and "
                f"{selected_field!r}."
            )
        groups.setdefault(record[group_field], []).append(record)

    if expected_group_count is not None:
        expected = _as_positive_integer(
            expected_group_count,
            name="expected_group_count",
        )
        if len(groups) != expected:
            raise ModelValidationError(
                f"Expected {expected} winner groups, found {len(groups)}."
            )

    for group, group_rows in groups.items():
        selected_count = sum(record[selected_field] is True for record in group_rows)
        non_boolean = [
            record[selected_field]
            for record in group_rows
            if not isinstance(record[selected_field], bool)
        ]
        if non_boolean:
            raise ModelValidationError(
                f"{selected_field} must be boolean in group {group!r}."
            )
        if selected_count != 1:
            raise ModelValidationError(
                f"Group {group!r} must contain exactly one selected row; "
                f"found {selected_count}."
            )


def validate_selection_ranks(
    records: Iterable[Mapping[str, Any]],
    *,
    group_field: str = "winner_group_id",
    rank_field: str = "selection_rank",
) -> None:
    """Validate that each group's ranks are exactly ``1..n`` without ties."""
    groups: dict[Any, list[int]] = {}
    for row_number, record in enumerate(records, start=1):
        if group_field not in record or rank_field not in record:
            raise ModelValidationError(
                f"Record {row_number} must contain {group_field!r} and "
                f"{rank_field!r}."
            )
        rank = _as_positive_integer(record[rank_field], name=rank_field)
        groups.setdefault(record[group_field], []).append(rank)

    for group, ranks in groups.items():
        expected = list(range(1, len(ranks) + 1))
        if sorted(ranks) != expected:
            raise ModelValidationError(
                f"Selection ranks in group {group!r} must be {expected}; "
                f"found {sorted(ranks)}."
            )


def validate_plot_references(
    records: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    filename_fields: Sequence[str] = (
        "plot1_a0_file",
        "plot1_a1_file",
        "plot2_file",
        "plot3_file",
    ),
    expected_file_count: int | None = None,
) -> tuple[Path, ...]:
    """Validate nonempty, unique, existing PNG references from output rows."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ModelValidationError(f"Output directory does not exist: {directory}.")
    if not filename_fields:
        raise ValueError("filename_fields must not be empty.")

    referenced: list[Path] = []
    for row_number, record in enumerate(records, start=1):
        for field in filename_fields:
            if field not in record:
                raise ModelValidationError(
                    f"Record {row_number} is missing plot field {field!r}."
                )
            filename = record[field]
            if not isinstance(filename, str) or not filename.strip():
                raise ModelValidationError(
                    f"Record {row_number} has an invalid filename in {field!r}."
                )
            if Path(filename).name != filename:
                raise ModelValidationError(
                    f"Plot field {field!r} must contain a filename, not a path: "
                    f"{filename!r}."
                )
            if Path(filename).suffix.lower() != ".png":
                raise ModelValidationError(
                    f"Plot field {field!r} must reference a PNG: {filename!r}."
                )
            path = directory / filename
            if not path.is_file():
                raise ModelValidationError(f"Referenced plot does not exist: {path}.")
            referenced.append(path)

    duplicates = [
        str(path)
        for path, count in Counter(referenced).items()
        if count > 1
    ]
    if duplicates:
        raise ModelValidationError(
            f"Plot filenames must be unique across selected rows: {duplicates}."
        )

    if expected_file_count is not None:
        expected = _as_nonnegative_integer(
            expected_file_count,
            name="expected_file_count",
        )
        if len(referenced) != expected:
            raise ModelValidationError(
                f"Expected {expected} plot references, found {len(referenced)}."
            )

    return tuple(referenced)


__all__ = [
    "ModelValidationError",
    "assert_close",
    "validate_absdiff",
    "validate_count_partition",
    "validate_execution_frequencies",
    "validate_frequency_partition",
    "validate_one_selected_per_group",
    "validate_plot_references",
    "validate_policy_pair",
    "validate_probability_complement",
    "validate_regime_frequencies",
    "validate_selection_ranks",
    "validate_thresholds",
    "validate_unique_records",
]
