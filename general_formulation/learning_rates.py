'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Learning-rate specifications for finite-time policy updates.

Scientific basis
----------------

- Equation (27): constant-gain learning rate ``eta_i,T(s) = eta0``.
- Equation (28): global diminishing learning rate
  ``eta_i,T(s) = eta0 / (1 + c*T)``.
- Assumption 2 and Equation (29): the global diminishing family satisfies the
  Robbins--Monro summability conditions for ``eta0 > 0`` and ``c > 0``.
- Equations (30)--(31): exact-empirical learning rate
  ``eta_i,T(s) = 1 / (N_i(s,T-1) + 1)``, based on the effective update count
  available before decision instant ``T``.
- Lemma 1 and Equation (32): the first exact-empirical update and subsequent updates 
reproduce the exact empirical action frequencies.

This module determines the magnitude of an effective policy update. It does
not decide whether an update occurs. Scenario-specific update rules must set
``eta_H_T = 0`` when the Human policy is not updated and must compute
``eta_AI_T`` in every scenario.
"""

from __future__ import annotations


from numbers import Integral
from typing import Final

import numpy as np

from .numerics import DEFAULT_ATOL, as_probability


ETA_KINDS: Final[tuple[str, str, str]] = (
    "constant",
    "global_decay",
    "exact_empirical",
)


def _as_nonnegative_integer(value: int, *, name: str) -> int:
    """Return a validated nonnegative integer and reject booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return integer


def _as_eta0(value: float | None) -> float:
    """Return a finite ``eta0`` in ``(0, 1]``."""
    if value is None:
        raise ValueError("eta0 is required for this eta_kind.")

    eta0 = float(value)
    if not np.isfinite(eta0) or not 0.0 < eta0 <= 1.0:
        raise ValueError("eta0 must be finite and lie in (0, 1].")
    return eta0


def _as_positive_c(value: float | None) -> float:
    """Return a finite global-decay parameter ``c > 0``."""
    if value is None:
        raise ValueError("c is required when eta_kind='global_decay'.")

    c = float(value)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c must be finite and strictly positive.")
    return c


def validate_eta_spec(
    *,
    eta_kind: str,
    eta0: float | None,
    c: float | None,
) -> tuple[str, float | None, float | None]:
    """Validate and return one canonical learning-rate specification.

    The names match the configuration and CSV columns used by the grid:
    ``eta_kind``, ``eta0``, and ``c``. Parameters that do not apply to a
    schedule must be ``None`` so configuration errors are detected early.
    """
    if not isinstance(eta_kind, str):
        raise TypeError("eta_kind must be a string.")
    if eta_kind not in ETA_KINDS:
        raise ValueError(
            f"eta_kind must be one of {ETA_KINDS}; received {eta_kind!r}."
        )

    if eta_kind == "constant":
        validated_eta0 = _as_eta0(eta0)
        if c is not None:
            raise ValueError("c must be None when eta_kind='constant'.")
        return eta_kind, validated_eta0, None

    if eta_kind == "global_decay":
        return eta_kind, _as_eta0(eta0), _as_positive_c(c)

    # Part I Eq. (30) [eq:eta_exact_empirical]; Part II Eq. (11)
    if eta0 is not None or c is not None:
        raise ValueError(
            "eta0 and c must both be None when eta_kind='exact_empirical'."
        )
    return eta_kind, None, None


def _decimal_token(value: float) -> str:
    """Encode a finite nonnegative decimal for a stable ``eta_label``."""
    decimal = np.format_float_positional(float(value), trim="-")
    return decimal.replace(".", "p")


def build_eta_label(
    *,
    eta_kind: str,
    eta0: float | None,
    c: float | None,
) -> str:
    """Return the unique human-readable schedule label used by the grid.

    Examples
    --------
    ``constant_eta0_0p05``
    ``global_decay_eta0_0p05_c_0p001``
    ``exact_empirical``
    """
    eta_kind, eta0, c = validate_eta_spec(
        eta_kind=eta_kind,
        eta0=eta0,
        c=c,
    )

    if eta_kind == "constant":
        assert eta0 is not None
        return f"constant_eta0_{_decimal_token(eta0)}"

    if eta_kind == "global_decay":
        assert eta0 is not None and c is not None
        return (
            f"global_decay_eta0_{_decimal_token(eta0)}"
            f"_c_{_decimal_token(c)}"
        )

    return "exact_empirical"


def constant_eta(*, eta0: float, name: str = "eta_T") -> float:
    """Return the constant-gain step size from Part I Eq. (27); Part II Eq. (9)."""
    eta_T = _as_eta0(eta0)
    return as_probability(eta_T, name=name, atol=DEFAULT_ATOL)


def global_decay_eta(
    *,
    eta0: float,
    c: float,
    T: int,
    name: str = "eta_T",
) -> float:
    """Return the global diminishing step size from Part I Eq. (28); Part II Eq. (10)."""
    eta0_value = _as_eta0(eta0)
    c_value = _as_positive_c(c)
    decision_index = _as_nonnegative_integer(T, name="T")

    eta_T = eta0_value / (1.0 + c_value * float(decision_index))
    eta_T = as_probability(eta_T, name=name, atol=DEFAULT_ATOL)
    if eta_T <= 0.0:
        raise ArithmeticError("The global diminishing step size must be positive.")
    return eta_T


def exact_empirical_eta(
    *,
    N_i_before: int,
    name: str = "eta_T",
) -> float:
    """Return the exact-empirical step size from Part I Eqs. (30)–(31); Part II Eqs. (11)–(12).

    ``N_i_before`` is the total effective incorporation count for one
    agent--state policy component before decision instant ``T``. It represents
    ``N_i(s,T-1)`` in Equations (23)--(24), not a global decision count.
    """
    effective_count = _as_nonnegative_integer(
        N_i_before,
        name="N_i_before",
    )
    eta_T = 1.0 / (float(effective_count) + 1.0)
    eta_T = as_probability(eta_T, name=name, atol=DEFAULT_ATOL)
    if eta_T <= 0.0:
        raise ArithmeticError("The exact-empirical step size must be positive.")
    return eta_T



def _compute_eta_validated(
    eta_kind: str,
    *,
    eta0: float | None,
    c: float | None,
    T: int,
    effective_count_before: int,
) -> float:
    """Evaluate an already validated learning-rate specification quickly.

    The formulas are exactly the constant, global-decay, and exact-empirical
    rules in Part I. Only cheap post-computation range checks remain.
    """
    if eta_kind == "constant":
        eta = float(eta0)  # type: ignore[arg-type]
    elif eta_kind == "global_decay":
        eta = float(eta0) / (1.0 + float(c) * float(T))  # type: ignore[arg-type]
    elif eta_kind == "exact_empirical":
        eta = 1.0 / (float(effective_count_before) + 1.0)
    else:
        raise RuntimeError(f"Unknown validated eta_kind: {eta_kind!r}.")

    if not 0.0 < eta <= 1.0:
        raise ArithmeticError(f"Computed learning rate left (0,1]: {eta!r}.")
    return float(eta)

def compute_eta_T(
    *,
    eta_kind: str,
    T: int,
    eta0: float | None,
    c: float | None,
    N_i_before: int | None = None,
    name: str = "eta_T",
) -> float:
    """Compute the effective step size for one agent--state policy update.

    The flat parameter names deliberately match ``config.py``, the grid, and
    the CSV schemas. ``N_i_before`` is required only for ``exact_empirical``.
    The function returns a strictly positive value in ``(0, 1]``; a zero used
    to denote "no policy update" must be assigned by the scenario-specific
    update mask rather than by this module.
    """
    eta_kind, eta0, c = validate_eta_spec(
        eta_kind=eta_kind,
        eta0=eta0,
        c=c,
    )
    _as_nonnegative_integer(T, name="T")

    if eta_kind == "constant":
        if N_i_before is not None:
            raise ValueError(
                "N_i_before must be None when eta_kind='constant'."
            )
        assert eta0 is not None
        return constant_eta(eta0=eta0, name=name)

    if eta_kind == "global_decay":
        if N_i_before is not None:
            raise ValueError(
                "N_i_before must be None when eta_kind='global_decay'."
            )
        assert eta0 is not None and c is not None
        return global_decay_eta(
            eta0=eta0,
            c=c,
            T=T,
            name=name,
        )

    if N_i_before is None:
        raise ValueError(
            "N_i_before is required when eta_kind='exact_empirical'."
        )
    return exact_empirical_eta(
        N_i_before=N_i_before,
        name=name,
    )


__all__ = [
    "ETA_KINDS",
    "build_eta_label",
    "compute_eta_T",
    "constant_eta",
    "exact_empirical_eta",
    "global_decay_eta",
    "validate_eta_spec",
]