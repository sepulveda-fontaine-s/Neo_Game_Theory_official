'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Stable identifiers and artifact filenames for simulation outputs.

This module contains implementation infrastructure rather than a mathematical
object from Part I. It guarantees that configurations, runs, winner groups,
and plot files can be joined reproducibly across CSV outputs without relying
on row order.

The public helpers deliberately separate:

- ``winner_group_id``: the comparison group used for winner selection;
- ``config_id``: the deterministic identity of a parameter configuration;
- ``run_id``: the identity of one replicate and seed of that configuration;
- ``plot_prefix``: a readable, filesystem-safe prefix for the four plots.

No identifier produced here changes the simulation or its scientific results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Any, Final


_IDENTIFIER_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_UNSAFE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")
_MULTIPLE_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(r"[-_.]{2,}")


class IdentifierError(ValueError):
    """Raised when an identifier cannot be constructed deterministically."""


def _as_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; received {value!r}.")
    return value


def _as_positive_integer(value: int, *, name: str) -> int:
    validated = _as_nonnegative_integer(value, name=name)
    if validated == 0:
        raise ValueError(f"{name} must be positive.")
    return validated


def _as_nonempty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty.")
    return stripped


@lru_cache(maxsize=256)
def _canonical_float(value: float) -> str:
    """Return a platform-stable decimal representation of a finite float."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IdentifierError(
            f"Identifier values must be finite; received {value!r}."
        )
    if numeric == 0.0:
        return "0"

    try:
        decimal_value = Decimal(str(numeric)).normalize()
    except InvalidOperation as exc:  # pragma: no cover - defensive guard
        raise IdentifierError(f"Could not canonicalize float {value!r}.") from exc

    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def canonicalize_identifier_value(value: Any) -> Any:
    """Convert a supported value into a deterministic JSON-compatible form.

    Mappings are sorted by key. Sequences preserve order. Sets are sorted by
    the canonical JSON representation of their members. NumPy scalar-like
    values are accepted when they expose an ``item()`` method.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return {"__float__": _canonical_float(value)}

    if hasattr(value, "item") and callable(value.item):
        scalar = value.item()
        if scalar is not value:
            return canonicalize_identifier_value(scalar)

    if isinstance(value, Mapping):
        canonical_mapping: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            canonical_key = _as_nonempty_string(str(key), name="mapping key")
            canonical_mapping[canonical_key] = canonicalize_identifier_value(value[key])
        return canonical_mapping

    if isinstance(value, (set, frozenset)):
        members = [canonicalize_identifier_value(member) for member in value]
        return sorted(
            members,
            key=lambda member: json.dumps(
                member,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize_identifier_value(member) for member in value]

    raise TypeError(
        "Unsupported identifier value type: "
        f"{type(value).__name__}. Convert it to a scalar, sequence, or mapping."
    )


def canonical_json(fields: Mapping[str, Any]) -> str:
    """Serialize identifier fields with deterministic ordering and floats."""
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping.")
    canonical = canonicalize_identifier_value(fields)
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _stable_digest_payload(payload: str, *, length: int = 16) -> str:
    """Hash an already canonical JSON payload without rebuilding it."""
    _as_positive_integer(length, name="length")
    if length > 64:
        raise ValueError("length must not exceed 64 hexadecimal characters.")
    if not isinstance(payload, str):
        raise TypeError("payload must be a canonical JSON string.")

    digest_size = max(1, math.ceil(length / 2))
    return hashlib.blake2b(
        payload.encode("utf-8"),
        digest_size=digest_size,
    ).hexdigest()[:length]


def _canonical_json_value(value: Any) -> str:
    """Serialize one value using the exact canonical identifier convention."""
    return json.dumps(
        canonicalize_identifier_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_object_from_fragments(
    fragments: Mapping[str, str],
) -> str:
    """Join pre-canonicalized JSON value fragments into one sorted object.

    This is an internal production primitive.  It preserves byte-for-byte the
    payload emitted by :func:`canonical_json`, while allowing invariant nested
    values (for example initial policy tables) to be canonicalized once and
    reused across the exhaustive grid.
    """
    if not isinstance(fragments, Mapping) or not fragments:
        raise ValueError("fragments must be a non-empty mapping.")

    encoded: list[str] = []
    for key in sorted(fragments):
        canonical_key = _as_nonempty_string(str(key), name="fragment key")
        value_fragment = fragments[key]
        if not isinstance(value_fragment, str) or not value_fragment:
            raise TypeError("Every canonical value fragment must be a non-empty string.")
        encoded.append(
            json.dumps(canonical_key, ensure_ascii=True) + ":" + value_fragment
        )
    return "{" + ",".join(encoded) + "}"


def _build_config_id_from_canonical_fragments(
    fragments: Mapping[str, str],
    *,
    digest_length: int = 20,
) -> str:
    """Build a config ID from reusable canonical value fragments."""
    payload = _canonical_object_from_fragments(fragments)
    return f"cfg_{_stable_digest_payload(payload, length=digest_length)}"


def stable_digest(fields: Mapping[str, Any], *, length: int = 16) -> str:
    """Return a lowercase BLAKE2 digest of the canonical field mapping."""
    return _stable_digest_payload(canonical_json(fields), length=length)


def safe_token(value: Any, *, max_length: int = 48) -> str:
    """Return one readable filesystem-safe token.

    Decimal points are encoded as ``p`` and negative signs as ``m`` for
    compact parameter labels. ``None`` is encoded as ``NA``.
    """
    _as_positive_integer(max_length, name="max_length")

    if value is None:
        raw = "NA"
    elif isinstance(value, bool):
        raw = "true" if value else "false"
    elif isinstance(value, float):
        raw = _canonical_float(value).replace("-", "m").replace(".", "p")
    elif isinstance(value, int) and not isinstance(value, bool):
        raw = str(value)
    else:
        raw = _as_nonempty_string(str(value), name="value")

    token = _UNSAFE_TOKEN_RE.sub("-", raw.strip())
    token = _MULTIPLE_SEPARATOR_RE.sub("-", token).strip("-_.")
    if not token:
        raise IdentifierError(f"Value {value!r} produced an empty safe token.")
    return token[:max_length]


def build_stable_identifier(
    prefix: str,
    fields: Mapping[str, Any],
    *,
    digest_length: int = 16,
) -> str:
    """Build ``<prefix>_<digest>`` from a canonical field mapping."""
    prefix_value = _as_nonempty_string(prefix, name="prefix")
    if not _IDENTIFIER_PREFIX_RE.fullmatch(prefix_value):
        raise IdentifierError(
            "prefix must start with a letter and contain only letters, digits, "
            "underscores, or hyphens."
        )
    return f"{prefix_value}_{stable_digest(fields, length=digest_length)}"


def build_winner_group_id(
    *,
    scenario: str,
    H: int,
    eta_kind: str,
    random_states: bool,
) -> str:
    """Build the readable winner group ``H x eta_kind x random_states``."""
    scenario_value = safe_token(_as_nonempty_string(scenario, name="scenario"))
    H_value = _as_positive_integer(H, name="H")
    eta_value = safe_token(_as_nonempty_string(eta_kind, name="eta_kind"))
    if not isinstance(random_states, bool):
        raise TypeError("random_states must be boolean.")
    state_mode = "exogenous" if random_states else "endogenous"
    return f"{scenario_value}__H{H_value}__eta-{eta_value}__states-{state_mode}"


def build_config_id(
    fields: Mapping[str, Any],
    *,
    digest_length: int = 20,
) -> str:
    """Build a deterministic configuration identifier from model parameters."""
    if not fields:
        raise ValueError("fields must contain the configuration parameters.")
    return build_stable_identifier("cfg", fields, digest_length=digest_length)


def build_run_id(
    *,
    config_id: str,
    replicate: int,
    seed: int,
    digest_length: int = 16,
) -> str:
    """Build the globally unique identity of one concrete simulation run."""
    config_value = _as_nonempty_string(config_id, name="config_id")
    replicate_value = _as_nonnegative_integer(replicate, name="replicate")
    seed_value = _as_nonnegative_integer(seed, name="seed")
    # The key order below is the exact lexicographic order used by
    # ``canonical_json(..., sort_keys=True)``.  Building the tiny payload
    # directly avoids the generic recursive canonicalizer for every run while
    # preserving existing run IDs byte-for-byte.
    payload = (
        "{\"config_id\":"
        + json.dumps(config_value, ensure_ascii=True)
        + ",\"replicate\":"
        + str(replicate_value)
        + ",\"seed\":"
        + str(seed_value)
        + "}"
    )
    return f"run_{_stable_digest_payload(payload, length=digest_length)}"


def build_plot_prefix(
    fields: Mapping[str, Any],
    *,
    leading_token: str = "scenario1",
    digest_length: int = 10,
    max_token_length: int = 32,
) -> str:
    """Build a readable and collision-resistant prefix for plot filenames.

    Field order follows mapping insertion order for readability. A stable hash
    of the complete mapping is appended so that truncated display tokens cannot
    create filename collisions.
    """
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a non-empty mapping.")

    parts = [safe_token(leading_token, max_length=max_token_length)]
    for key, value in fields.items():
        key_token = safe_token(key, max_length=max_token_length)
        value_token = safe_token(value, max_length=max_token_length)
        parts.append(f"{key_token}-{value_token}")

    parts.append(f"id-{stable_digest(fields, length=digest_length)}")
    return "__".join(parts)


def build_plot_filenames(plot_prefix: str) -> dict[str, str]:
    """Return the four canonical Scenario 1 plot filenames."""
    prefix = _as_nonempty_string(plot_prefix, name="plot_prefix")
    if "/" in prefix or "\\" in prefix:
        raise IdentifierError("plot_prefix must not contain path separators.")

    return {
        "plot1_a0_file": f"{prefix}__plot1_a0.png",
        "plot1_a1_file": f"{prefix}__plot1_a1.png",
        "plot2_file": f"{prefix}__plot2.png",
        "plot3_file": f"{prefix}__plot3.png",
    }


__all__ = [
    "IdentifierError",
    "build_config_id",
    "build_plot_filenames",
    "build_plot_prefix",
    "build_run_id",
    "build_stable_identifier",
    "build_winner_group_id",
    "canonical_json",
    "canonicalize_identifier_value",
    "safe_token",
    "stable_digest",
]
