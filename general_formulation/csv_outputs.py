'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Ordered CSV schema enforcement and deterministic writing utilities.

The scientific CSV contracts are declared later in the scenario-specific
``output_schema.py`` module. This module provides only scenario-independent
infrastructure:

- exact column-order enforcement;
- nullable/non-null validation;
- primary-key uniqueness checks;
- deterministic scalar serialization;
- atomic replacement for complete CSV files;
- guarded append operations with header compatibility checks;
- incremental atomic writing for long trajectories.

These helpers do not derive simulation metrics and do not alter values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final, TextIO


CSV_ENCODING: Final[str] = "utf-8"
CSV_NEWLINE: Final[str] = ""


class CsvOutputError(ValueError):
    """Raised when a row or file violates its declared CSV contract."""


def _as_column_name(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty.")
    if stripped != value:
        raise ValueError(f"{name} must not contain leading or trailing whitespace.")
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _python_scalar(value: Any) -> Any:
    """Convert scalar-like objects without silently flattening arrays."""
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item") and callable(value.item):
        try:
            scalar = value.item()
        except (ValueError, TypeError):
            return value
        if scalar is not value:
            return scalar

    return value


def serialize_csv_value(value: Any) -> Any:
    """Return one deterministic CSV field representation.

    Missing values are written as empty fields. Booleans use ``True`` and
    ``False`` explicitly. Finite numbers remain numeric values so Python's CSV
    writer serializes them without locale-dependent formatting.
    """
    value = _python_scalar(value)
    if _is_missing(value):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CsvOutputError(
                f"CSV values must be finite or missing; received {value!r}."
            )
        return repr(value)
    if isinstance(value, (str, int)):
        return value
    raise TypeError(
        "CSV cells must be scalar strings, integers, floats, booleans, paths, "
        f"or missing values; received {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class CsvSchema:
    """Exact ordered schema for one CSV output."""

    name: str
    columns: tuple[str, ...]
    nullable: frozenset[str] = field(default_factory=frozenset)
    primary_key: tuple[str, ...] = ()
    allow_extra_columns: bool = False
    _column_set: frozenset[str] = field(init=False, repr=False, compare=False)
    _required_non_null: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _as_column_name(self.name, name="schema name")
        if not isinstance(self.columns, tuple) or not self.columns:
            raise ValueError("columns must be a non-empty tuple.")

        validated_columns = tuple(
            _as_column_name(column, name="column") for column in self.columns
        )
        if len(set(validated_columns)) != len(validated_columns):
            duplicates = sorted(
                column
                for column in set(validated_columns)
                if validated_columns.count(column) > 1
            )
            raise ValueError(f"Duplicate schema columns: {duplicates}.")

        unknown_nullable = set(self.nullable) - set(validated_columns)
        if unknown_nullable:
            raise ValueError(
                f"nullable contains unknown columns: {sorted(unknown_nullable)}."
            )

        unknown_key = set(self.primary_key) - set(validated_columns)
        if unknown_key:
            raise ValueError(
                f"primary_key contains unknown columns: {sorted(unknown_key)}."
            )
        if len(set(self.primary_key)) != len(self.primary_key):
            raise ValueError("primary_key must not contain duplicate columns.")

        column_set = frozenset(validated_columns)
        object.__setattr__(self, "_column_set", column_set)
        object.__setattr__(
            self,
            "_required_non_null",
            column_set - self.nullable,
        )

    @property
    def required_non_null(self) -> frozenset[str]:
        """Columns that must contain a non-missing scalar value."""
        return self._required_non_null


class CsvRowWriter:
    """Incrementally validate and atomically write one complete CSV file.

    Rows are never materialized as a list.  The writer keeps only declared
    primary keys needed for duplicate detection and writes each validated row
    immediately to a temporary file.  The destination is replaced only after
    the context exits successfully.
    """

    def __init__(
        self,
        path: str | Path,
        schema: CsvSchema,
        *,
        atomic: bool = True,
    ) -> None:
        if not isinstance(schema, CsvSchema):
            raise TypeError("schema must be a CsvSchema.")
        if not isinstance(atomic, bool):
            raise TypeError("atomic must be boolean.")
        self.path = Path(path)
        self.schema = schema
        self.atomic = atomic
        self.row_count = 0
        self._seen_keys: set[tuple[Any, ...]] = set()
        self._handle: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._temporary_path: Path | None = None

    def __enter__(self) -> "CsvRowWriter":
        if self._handle is not None:
            raise RuntimeError("CsvRowWriter cannot be entered more than once.")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.atomic:
            temporary_handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding=CSV_ENCODING,
                newline=CSV_NEWLINE,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            )
            self._temporary_path = Path(temporary_handle.name)
            self._handle = temporary_handle
        else:
            self._temporary_path = self.path
            self._handle = self.path.open(
                "w",
                encoding=CSV_ENCODING,
                newline=CSV_NEWLINE,
            )

        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=list(self.schema.columns),
            extrasaction="raise",
            lineterminator="\n",
        )
        self._writer.writeheader()
        return self

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Validate and write one row without retaining it."""
        if self._writer is None or self._handle is None:
            raise RuntimeError("CsvRowWriter must be used inside a with block.")

        row_number = self.row_count + 1
        ordered = validate_csv_row(
            row,
            self.schema,
            row_number=row_number,
        )
        if self.schema.primary_key:
            key = _primary_key_tuple(ordered, self.schema)
            if key in self._seen_keys:
                raise CsvOutputError(
                    f"Duplicate primary key {key!r} in {self.schema.name}."
                )
            self._seen_keys.add(key)

        self._writer.writerow(_serialized_row(ordered, self.schema))
        self.row_count = row_number

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write an iterable row by row."""
        for row in rows:
            self.write_row(row)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        handle = self._handle
        temporary_path = self._temporary_path
        self._writer = None
        self._handle = None
        self._temporary_path = None

        try:
            if handle is not None:
                handle.close()

            if temporary_path is None:
                return False

            if exc_type is None and self.atomic:
                os.replace(temporary_path, self.path)
            elif exc_type is not None and self.atomic and temporary_path.exists():
                temporary_path.unlink()
        except Exception:
            if self.atomic and temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise

        return False


def validate_csv_row(
    row: Mapping[str, Any],
    schema: CsvSchema,
    *,
    row_number: int | None = None,
) -> dict[str, Any]:
    """Validate and reorder one row according to ``schema.columns``."""
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping.")
    if not isinstance(schema, CsvSchema):
        raise TypeError("schema must be a CsvSchema.")

    location = "" if row_number is None else f" at row {row_number}"
    row_columns = set(row)
    schema_columns = schema._column_set

    missing_columns = schema_columns - row_columns
    if missing_columns:
        raise CsvOutputError(
            f"{schema.name}{location} is missing columns: "
            f"{sorted(missing_columns)}."
        )

    extra_columns = row_columns - schema_columns
    if extra_columns and not schema.allow_extra_columns:
        raise CsvOutputError(
            f"{schema.name}{location} contains undeclared columns: "
            f"{sorted(extra_columns)}."
        )

    ordered: dict[str, Any] = {}
    for column in schema.columns:
        value = _python_scalar(row[column])
        if column in schema.required_non_null and _is_missing(value):
            raise CsvOutputError(
                f"{schema.name}{location} has a missing value in required "
                f"column {column!r}."
            )
        ordered[column] = value

    return ordered


def _primary_key_tuple(row: Mapping[str, Any], schema: CsvSchema) -> tuple[Any, ...]:
    values = tuple(_python_scalar(row[column]) for column in schema.primary_key)
    if any(_is_missing(value) for value in values):
        raise CsvOutputError(
            f"Primary key {schema.primary_key} in {schema.name} contains a missing value."
        )
    return values


def validate_csv_rows(
    rows: Iterable[Mapping[str, Any]],
    schema: CsvSchema,
) -> list[dict[str, Any]]:
    """Validate all rows and reject duplicate declared primary keys."""
    validated: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    for row_number, row in enumerate(rows, start=1):
        ordered = validate_csv_row(row, schema, row_number=row_number)
        if schema.primary_key:
            key = _primary_key_tuple(ordered, schema)
            if key in seen_keys:
                raise CsvOutputError(
                    f"Duplicate primary key {key!r} in {schema.name}."
                )
            seen_keys.add(key)
        validated.append(ordered)

    return validated


def _serialized_row(row: Mapping[str, Any], schema: CsvSchema) -> dict[str, Any]:
    return {
        column: serialize_csv_value(row[column])
        for column in schema.columns
    }


def read_csv_header(path: str | Path) -> tuple[str, ...]:
    """Read and return the exact header of an existing CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", encoding=CSV_ENCODING, newline=CSV_NEWLINE) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise CsvOutputError(f"CSV file is empty: {csv_path}.") from exc

    return tuple(header)


def _existing_primary_keys(path: Path, schema: CsvSchema) -> set[tuple[str, ...]]:
    if not schema.primary_key or not path.exists():
        return set()

    with path.open("r", encoding=CSV_ENCODING, newline=CSV_NEWLINE) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != schema.columns:
            raise CsvOutputError(
                f"Existing header in {path} does not match schema {schema.name}."
            )
        return {
            tuple(row[column] for column in schema.primary_key)
            for row in reader
        }


def write_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    schema: CsvSchema,
    *,
    atomic: bool = True,
) -> Path:
    """Write a complete CSV file incrementally in exact schema order."""
    with CsvRowWriter(path, schema, atomic=atomic) as writer:
        writer.write_rows(rows)
    return Path(path)


def append_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    schema: CsvSchema,
) -> Path:
    """Append validated rows while preserving header and key integrity."""
    csv_path = Path(path)
    validated = validate_csv_rows(rows, schema)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        return write_csv_rows(csv_path, validated, schema)

    existing_header = read_csv_header(csv_path)
    if existing_header != schema.columns:
        raise CsvOutputError(
            f"Header mismatch for {csv_path}: expected {schema.columns}, "
            f"found {existing_header}."
        )

    if schema.primary_key:
        existing_keys = _existing_primary_keys(csv_path, schema)
        new_keys: set[tuple[str, ...]] = set()
        for row in validated:
            serialized = _serialized_row(row, schema)
            key = tuple(str(serialized[column]) for column in schema.primary_key)
            if key in existing_keys or key in new_keys:
                raise CsvOutputError(
                    f"Appending would duplicate primary key {key!r} in {schema.name}."
                )
            new_keys.add(key)

    with csv_path.open("a", encoding=CSV_ENCODING, newline=CSV_NEWLINE) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(schema.columns),
            extrasaction="raise",
            lineterminator="\n",
        )
        for row in validated:
            writer.writerow(_serialized_row(row, schema))

    return csv_path


__all__ = [
    "CSV_ENCODING",
    "CsvOutputError",
    "CsvRowWriter",
    "CsvSchema",
    "append_csv_rows",
    "read_csv_header",
    "serialize_csv_value",
    "validate_csv_row",
    "validate_csv_rows",
    "write_csv_rows",
]
