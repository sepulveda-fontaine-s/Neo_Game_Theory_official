'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario 1 delegation under Human arbitration.


Scientific basis
----------------

- Part I Definition 7 and Eq. (42) [eq:lambda_case1] define the Scenario 1
  regional selector: ``lambda_T=0`` in agreement selects the AI proposal,
  while ``lambda_T=1`` in disagreement selects the Human proposal.
- Eqs. (43)--(46) specialize the contextual rule for Scenario 1: they define
  the contextual Human/AI support, deterministic maximum-support selection,
  the equivalent midpoint rule, and the resulting execution/owner mapping.
- The common contextual mechanism is evaluated only after the decision has
  entered the contextual region. Exact contextual ties are resolved in favour
  of the Human in Scenario 1.

The implementation contains no ``kappa``, ``q_ctx``, or contextual random draw.
``D_JS_T`` first classifies the regime; only after contextual entry is its
normalized position used by the deterministic maximum-probability selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from general_formulation.contextual import (
    contextual_max_probability_selector,
    contextual_selector_probabilities,
)
from general_formulation.numerics import DEFAULT_ATOL, as_probability
from general_formulation.validation import validate_thresholds


REGIME_AGREEMENT = "agreement"
REGIME_CONTEXTUAL = "contextual"
REGIME_DISAGREEMENT = "disagreement"
REGIME_VALUES = (
    REGIME_AGREEMENT,
    REGIME_CONTEXTUAL,
    REGIME_DISAGREEMENT,
)

LAMBDA_HUMAN = 1
LAMBDA_AI = 0


@dataclass(frozen=True, slots=True)
class DelegationDecision:
    """Scenario 1 selector result at one decision instant.

    ``ctx_prob_H_T`` and ``ctx_prob_AI_T`` are ``None`` outside the contextual
    regime because the conditional probabilities are then not applicable.
    """

    D_JS_T: float
    regime_T: str
    lambda_T: int
    ctx_prob_H_T: float | None
    ctx_prob_AI_T: float | None


def _as_rng(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    return rng


def _as_regime(regime_T: str) -> str:
    if not isinstance(regime_T, str):
        raise TypeError("regime_T must be a string.")
    if regime_T not in REGIME_VALUES:
        raise ValueError(
            f"regime_T must be one of {REGIME_VALUES}; received {regime_T!r}."
        )
    return regime_T


def classify_scenario1_regime(
    D_JS_T: float,
    *,
    alpha_agree: float,
    alpha_disagree: float,
    atol: float = DEFAULT_ATOL,
) -> str:
    """Classify the predecision divergence using Part I Eq. (42) [eq:lambda_case1].

    Threshold membership is exact after the shared probability validators have
    corrected only boundary-level floating-point drift. Agreement includes the
    lower boundary, contextual uses strict inequalities, and disagreement
    includes the upper boundary.
    """
    agreement, disagreement = validate_thresholds(
        alpha_agree,
        alpha_disagree,
        atol=atol,
    )
    divergence = as_probability(D_JS_T, name="D_JS_T", atol=atol)

    if divergence <= agreement:
        return REGIME_AGREEMENT
    if divergence >= disagreement:
        return REGIME_DISAGREEMENT
    return REGIME_CONTEXTUAL


def contextual_execution_probabilities(
    D_JS_T: float,
    *,
    alpha_agree: float,
    alpha_disagree: float,
    atol: float = DEFAULT_ATOL,
) -> tuple[float, float]:
    
    # Return Scenario 1 contextual support from Part I Eq. (43) [eq:s1_contextual_probabilities]:

    agreement, disagreement = validate_thresholds(
        alpha_agree,
        alpha_disagree,
        atol=atol,
    )
    divergence = as_probability(D_JS_T, name="D_JS_T", atol=atol)

    if not agreement < divergence < disagreement:
        raise ValueError(
            "Contextual execution probabilities require "
            "alpha_agree < D_JS_T < alpha_disagree."
        )

    q_one, q_zero = contextual_selector_probabilities(
        divergence,
        alpha_agree=agreement,
        alpha_disagree=disagreement,
        atol=atol,
    )
    # Scenario 1 convention: selector 1 = Human, selector 0 = AI.
    return q_one, q_zero


def select_scenario1_lambda(
    D_JS_T: float,
    *,
    alpha_agree: float,
    alpha_disagree: float,
    rng: np.random.Generator,
    regime_T: str | None = None,
    atol: float = DEFAULT_ATOL,
) -> DelegationDecision:
    """Select the Scenario 1 delegation variable at decision instant ``T``.

    ``regime_T`` may be supplied when classification was deliberately performed
    before proposal sampling. The function rechecks that the supplied regime is
    consistent with ``D_JS_T`` and the thresholds, preventing temporal or
    boundary mismatches between classification and selector sampling.
    """
    generator = _as_rng(rng)
    divergence = as_probability(D_JS_T, name="D_JS_T", atol=atol)
    classified = classify_scenario1_regime(
        divergence,
        alpha_agree=alpha_agree,
        alpha_disagree=alpha_disagree,
        atol=atol,
    )

    if regime_T is not None:
        supplied = _as_regime(regime_T)
        if supplied != classified:
            raise ValueError(
                "regime_T is inconsistent with D_JS_T and the delegation "
                f"thresholds: supplied {supplied!r}, expected {classified!r}."
            )

    if classified == REGIME_AGREEMENT:
        return DelegationDecision(
            D_JS_T=divergence,
            regime_T=classified,
            lambda_T=LAMBDA_AI,
            ctx_prob_H_T=None,
            ctx_prob_AI_T=None,
        )

    if classified == REGIME_DISAGREEMENT:
        return DelegationDecision(
            D_JS_T=divergence,
            regime_T=classified,
            lambda_T=LAMBDA_HUMAN,
            ctx_prob_H_T=None,
            ctx_prob_AI_T=None,
        )

    ctx_prob_H_T, ctx_prob_AI_T = contextual_execution_probabilities(
        divergence,
        alpha_agree=alpha_agree,
        alpha_disagree=alpha_disagree,
        atol=atol,
    )
    del generator

    #Part I Eq. (44) [lambda_context]; exact contextual ties select Human.
    lambda_T = contextual_max_probability_selector(
        ctx_prob_H_T,
        ctx_prob_AI_T,
        tie_selector=LAMBDA_HUMAN,
        atol=atol,
    )

    return DelegationDecision(
        D_JS_T=divergence,
        regime_T=classified,
        lambda_T=lambda_T,
        ctx_prob_H_T=ctx_prob_H_T,
        ctx_prob_AI_T=ctx_prob_AI_T,
    )


__all__ = [
    "DelegationDecision",
    "LAMBDA_AI",
    "LAMBDA_HUMAN",
    "REGIME_AGREEMENT",
    "REGIME_CONTEXTUAL",
    "REGIME_DISAGREEMENT",
    "REGIME_VALUES",
    "classify_scenario1_regime",
    "contextual_execution_probabilities",
    "select_scenario1_lambda",
]
