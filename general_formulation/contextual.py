'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Common contextual selector probabilities.

The general formulation computes probabilities for the *binary selector* only 
(Part I Eqs. (13)--(14); Part II Eqs. (3)--(4).):

    q_one  = (D_JS - alpha_agree) / (alpha_disagree - alpha_agree)
    q_zero = 1 - q_one

The numerical meaning of selector values 1 and 0 is scenario-specific.  Public
functions validate boundary inputs once.  Production kernels use the internal
validated primitive after the contextual region has already been classified.
The contextual selector itself is deterministic: the value with maximum
conditional support is selected, with a scenario-specific exact-tie convention.
"""

from __future__ import annotations

from general_formulation.numerics import DEFAULT_ATOL, as_probability
from general_formulation.validation import validate_thresholds


def _contextual_selector_probabilities_validated(
    D_JS_T: float,
    alpha_agree: float,
    alpha_disagree: float,
) -> tuple[float, float]:
    """Return ``(q_one, q_zero)`` for already validated contextual inputs."""
    q_one = (D_JS_T - alpha_agree) / (alpha_disagree - alpha_agree)
    return q_one, 1.0 - q_one


def contextual_selector_probabilities(
    D_JS_T: float,
    *,
    alpha_agree: float,
    alpha_disagree: float,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, float]:
    """Validate one external call and return neutral selector probabilities."""
    agreement, disagreement = validate_thresholds(
        alpha_agree, alpha_disagree, atol=atol
    )
    divergence = as_probability(D_JS_T, name="D_JS_T", atol=atol)
    if not agreement < divergence < disagreement:
        raise ValueError(
            "Contextual probabilities require "
            "alpha_agree < D_JS_T < alpha_disagree."
        )
    q_one, q_zero = _contextual_selector_probabilities_validated(
        divergence, agreement, disagreement
    )
    # Guard only the public boundary.  The production loop does not repeat it.
    if q_one < -atol or q_one > 1.0 + atol or q_zero < -atol or q_zero > 1.0 + atol:
        raise ArithmeticError("Contextual selector probabilities left [0,1].")
    if abs((q_one + q_zero) - 1.0) > atol:
        raise ArithmeticError("Contextual selector probabilities do not sum to one.")
    return min(max(q_one, 0.0), 1.0), min(max(q_zero, 0.0), 1.0)


def contextual_max_probability_selector(
    q_one: float,
    q_zero: float,
    *,
    tie_selector: int,
    atol: float = DEFAULT_ATOL,
) -> int:

    """
    Part I Eqs. (16)--(17); Part II Eqs. (6)--(7). Exact ties use the scenario-specific convention.
    Return the deterministic max-probability contextual selector.

    ``tie_selector`` is the scenario-specific convention used only when the
    two contextual probabilities are equal.  
    """
    if tie_selector not in (0, 1):
        raise ValueError("tie_selector must be either 0 or 1.")
    q1 = as_probability(q_one, name="q_one", atol=atol)
    q0 = as_probability(q_zero, name="q_zero", atol=atol)

    if abs((q1 + q0) - 1.0) > atol:
        raise ArithmeticError("Contextual selector probabilities do not sum to one.")

    # Direct floating-point comparison is intentional: the scenario-specific tie rule applies only to exact numerical equality.
    if q1 > q0:
        return 1
    if q0 > q1:
        return 0
    return int(tie_selector)


__all__ = [
    "_contextual_selector_probabilities_validated",
    "contextual_max_probability_selector",
    "contextual_selector_probabilities",
]
