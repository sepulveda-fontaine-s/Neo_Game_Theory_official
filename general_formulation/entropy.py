'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Entropy and policy-divergence primitives for Neo-Game Theory.

Scientific basis
----------------
The operational order is taken from ``Part_I``. 
- Equations (5)--(6): Shannon entropy of the Human and AI policies.
- Part I Eq. (8) [JSdivergence]: symmetric Jensen--Shannon divergence.
- Equation Part I Eq. (9) [eq:DJS]: Jensen--Shannon divergence applied to the state-conditional
  Human and AI policies.
- Part I Eq. (10) [D_JS_T]: ``D_JS^T`` is computed from the policies available at the
  current state before proposals, delegation, execution, and policy updates.

Base-2 logarithms are mandatory in the implementation because Part I uses
that convention to normalize ``D_JS`` to ``[0, 1]``. This module does not
select a delegation regime and does not update either policy.
"""

from __future__ import annotations

from math import log2

import numpy as np
from numpy.typing import ArrayLike

from .numerics import DEFAULT_ATOL, as_probability, as_probability_vector



def _binary_entropy_validated(
    p0: float,
    p1: float | None = None,
    *,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Fast base-2 entropy for an already validated binary policy.

    Passing both components preserves the floating-point operation order used
    by the strict vector implementation. Expensive type, shape, and simplex
    validation is omitted; a cheap scalar range check remains.
    """
    x0 = float(p0)
    x1 = 1.0 - x0 if p1 is None else float(p1)
    if (
        x0 < -atol
        or x1 < -atol
        or x0 > 1.0 + atol
        or x1 > 1.0 + atol
        or abs((x0 + x1) - 1.0) > atol
    ):
        raise ArithmeticError(
            f"Validated binary policy left the simplex: ({x0!r}, {x1!r})."
        )
    if x0 < 0.0:
        x0 = 0.0
    elif x0 > 1.0:
        x0 = 1.0
    if x1 < 0.0:
        x1 = 0.0
    elif x1 > 1.0:
        x1 = 1.0
    total = x0 + x1
    if total != 1.0:
        x0 /= total
        x1 /= total

    result = 0.0
    if x0 > 0.0:
        result -= x0 * log2(x0)
    if x1 > 0.0:
        result -= x1 * log2(x1)
    if result < -atol or result > 1.0 + atol:
        raise ArithmeticError(f"Binary entropy left [0,1]: {result!r}.")
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


def _binary_js_validated(
    p0_H: float,
    p1_H: float,
    p0_AI: float,
    p1_AI: float,
    entropy_H: float | None = None,
    entropy_AI: float | None = None,
    *,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Fast normalized Jensen--Shannon divergence for binary policies.

    Both policy components are supplied so the calculation follows the strict
    vector formula exactly, while omitting repeated vector validation.
    """
    h0, h1 = float(p0_H), float(p1_H)
    a0, a1 = float(p0_AI), float(p1_AI)
    h_total = h0 + h1
    a_total = a0 + a1
    if h_total != 1.0:
        h0 /= h_total
        h1 /= h_total
    if a_total != 1.0:
        a0 /= a_total
        a1 /= a_total

    h_H = (
        _binary_entropy_validated(h0, h1, atol=atol)
        if entropy_H is None
        else float(entropy_H)
    )
    h_AI = (
        _binary_entropy_validated(a0, a1, atol=atol)
        if entropy_AI is None
        else float(entropy_AI)
    )
    m0 = 0.5 * (h0 + a0)
    m1 = 0.5 * (h1 + a1)
    h_m = _binary_entropy_validated(m0, m1, atol=atol)
    result = h_m - 0.5 * h_H - 0.5 * h_AI
    if result < -atol or result > 1.0 + atol:
        raise ArithmeticError(f"Normalized binary D_JS left [0,1]: {result!r}.")
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


def shannon_entropy(
    policy: ArrayLike,
    *,
    name: str = "policy",
    atol: float = DEFAULT_ATOL,
) -> float:
    """Return the base-2 Shannon entropy of a finite policy.

    The input must already be a valid probability vector. Components equal to
    zero contribute zero to the entropy, following the standard convention
    ``0 * log2(0) = 0`` used in Equations (5)--(6).

    Parameters
    ----------
    policy:
        Finite probability vector on the common executed-action space.
    name:
        Variable name used in validation errors.
    atol:
        Absolute tolerance used only for floating-point validation.

    Returns
    -------
    float
        Shannon entropy in bits.
    """
    probability_vector = as_probability_vector(policy, name=name, atol=atol)

    # Zero-probability actions have zero entropy contribution. Masking avoids
    # evaluating log2(0) without introducing smoothing or pseudocount mass.
    positive = probability_vector > 0.0
    Sh_entr = -np.sum(
        probability_vector[positive] * np.log2(probability_vector[positive]),
        dtype=np.float64,
    )

    # Shannon entropy is nonnegative and is bounded by log2(|A|). For the
    # binary simulations this upper bound equals one, while the generic public
    # function remains correct for larger finite action supports.
    entropy_value = float(Sh_entr)
    maximum = float(np.log2(probability_vector.size))
    if entropy_value < -atol or entropy_value > maximum + atol:
        raise ArithmeticError(
            "Shannon entropy left its theoretical range "
            f"[0, log2(|A|)]: {entropy_value!r}."
        )
    if entropy_value < 0.0:
        return 0.0
    if entropy_value > maximum:
        return maximum
    return entropy_value


def jensen_shannon_divergence(
    p_H: ArrayLike,
    p_AI: ArrayLike,
    *,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Return the base-2 Jensen--Shannon divergence between two policies.

    The variable names are intended to remain stable across
    the simulation, longitudinal CSV writer, winner selection, plots, and
    ``main.py``. At decision instant ``T``, callers must pass the Human and AI
    policies available at the current realized state before either policy is
    updated. The returned scalar is therefore the computational counterpart of
    Equation (10) [D_JS_T].

    Parameters
    ----------
    p_H:
        Human policy ``p_H(· | s_T, T)`` on the common action support.
    p_AI:
        AI policy ``p_AI(· | s_T, T)`` on the same action support.
    atol:
        Absolute tolerance used only for probability and range validation.

    Returns
    -------
    float
        Symmetric Jensen--Shannon divergence in ``[0, 1]``.
    """
    p_H_vector = as_probability_vector(p_H, name="p_H", atol=atol)
    p_AI_vector = as_probability_vector(p_AI, name="p_AI", atol=atol)

    if p_H_vector.shape != p_AI_vector.shape:
        raise ValueError(
            "p_H and p_AI must use the same finite action support; "
            f"received shapes {p_H_vector.shape} and {p_AI_vector.shape}."
        )

    mixture_policy = 0.5 * (p_H_vector + p_AI_vector)

    # Part I Eq. (8) [JSdivergence]: D_JS(P || Q) = S((P + Q)/2) - S(P)/2 - S(Q)/2.
    D_JS = (
        shannon_entropy(mixture_policy, name="mixture_policy", atol=atol)
        - 0.5 * shannon_entropy(p_H_vector, name="p_H", atol=atol)
        - 0.5 * shannon_entropy(p_AI_vector, name="p_AI", atol=atol)
    )

    # With base-2 logarithms, Part I bounds D_JS to [0, 1]. This helper clips
    # only boundary-level roundoff and rejects a materially invalid result.
    return as_probability(D_JS, name="D_JS", atol=atol)


__all__ = [
    "jensen_shannon_divergence",
    "shannon_entropy",
]
