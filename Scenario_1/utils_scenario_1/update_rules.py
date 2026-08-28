'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario 1 asymmetric policy-update eligibility and execution.

Scientific basis
----------------
The learning-rate families are defined in Part I Eqs. (27)--(32).
The generic executed-action policy recursion is Part I Eq. (35)
[monroeq]. Proposition 3 establishes the Scenario 1 update structure:
the AI updates after every decision, whereas the Human updates only
in the agreement region. Scenario 1 applies the following asymmetric update mask:

- the AI policy at ``s_T`` receives an effective update at every decision;
- the Human policy at ``s_T`` receives an effective update only in agreement;
- every disabled or unvisited policy component is carried forward unchanged.

Effective incorporation counts are updated only when the corresponding policy
is effectively updated. ``eta_H_T`` is reported as zero outside agreement;
that zero is a CSV sentinel and is never passed to the generic update formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from general_formulation.learning_rates import compute_eta_T
from general_formulation.numerics import DEFAULT_ATOL, ProbabilityVector
from general_formulation.policy_updates import (
    EffectiveCountVector,
    as_effective_count_vector,
    carry_policy_forward,
    effective_update_count,
    increment_effective_action_counts,
    policy_from_effective_action_counts,
    update_policy_toward_executed_action,
)

from .delegation import REGIME_AGREEMENT, REGIME_VALUES


@dataclass(frozen=True, slots=True)
class Scenario1UpdateEligibility:
    """Scenario 1 update mask for one realized regime."""

    human_updated_T: bool
    ai_updated_T: bool


@dataclass(frozen=True, slots=True)
class Scenario1PolicyUpdate:
    """Postdecision policies, counts, and applied learning rates at ``s_T``."""

    p_H_T_plus_1: ProbabilityVector
    p_AI_T_plus_1: ProbabilityVector
    N_H_s_a_T: EffectiveCountVector
    N_AI_s_a_T: EffectiveCountVector
    eta_H_T: float
    eta_AI_T: float
    human_updated_T: bool
    ai_updated_T: bool


def scenario1_update_eligibility(regime_T: str) -> Scenario1UpdateEligibility:
    """Return the Scenario 1 update mask from Part I Proposition 3:
    AI updates always; Human updates only in agreement"""
      
    if not isinstance(regime_T, str):
        raise TypeError("regime_T must be a string.")
    if regime_T not in REGIME_VALUES:
        raise ValueError(
            f"regime_T must be one of {REGIME_VALUES}; received {regime_T!r}."
        )
    return Scenario1UpdateEligibility(
        human_updated_T=(regime_T == REGIME_AGREEMENT),
        ai_updated_T=True,
    )


def _eta_for_agent(
    *,
    eta_kind: str,
    eta0: float | None,
    c: float | None,
    T: int,
    N_i_s_a_before: EffectiveCountVector,
    name: str,
) -> float:
    N_i_before = (
        effective_update_count(N_i_s_a_before, name=f"{name}_counts")
        if eta_kind == "exact_empirical"
        else None
    )
    return compute_eta_T(
        eta_kind=eta_kind,
        T=T,
        eta0=eta0,
        c=c,
        N_i_before=N_i_before,
        name=name,
    )


def _validate_exact_empirical_predecision_identity(
    p_i_T: ArrayLike,
    N_i_s_a_before: EffectiveCountVector,
    *,
    name: str,
    atol: float,
) -> None:
    """Check the Lemma 1 identity whenever at least one count exists."""
    if effective_update_count(N_i_s_a_before, name=f"{name}_counts") == 0:
        return
    expected = policy_from_effective_action_counts(
        N_i_s_a_before,
        initial_policy=p_i_T,
        name=f"{name}_expected",
        atol=atol,
    )
    observed = carry_policy_forward(p_i_T, name=name, atol=atol)
    if not np.allclose(observed, expected, rtol=0.0, atol=atol):
        raise ValueError(
            f"{name} is inconsistent with its exact-empirical effective counts."
        )


def apply_scenario1_policy_updates(
    p_H_T: ArrayLike,
    p_AI_T: ArrayLike,
    N_H_s_a_before: ArrayLike,
    N_AI_s_a_before: ArrayLike,
    *,
    a_star_T: int,
    regime_T: str,
    eta_kind: str,
    eta0: float | None,
    c: float | None,
    T: int,
    atol: float = DEFAULT_ATOL,
) -> Scenario1PolicyUpdate:
    """Apply the Scenario 1 update mask to the current state component.

    Counts are maintained under all three schedules because they are useful
    audit variables. Only ``exact_empirical`` reads them when computing eta.
    The inputs are not modified.
    """
    eligibility = scenario1_update_eligibility(regime_T)
    H_counts_before = as_effective_count_vector(
        N_H_s_a_before,
        name="N_H_s_a_before",
    )
    AI_counts_before = as_effective_count_vector(
        N_AI_s_a_before,
        name="N_AI_s_a_before",
    )
    if H_counts_before.shape != AI_counts_before.shape:
        raise ValueError(
            "Human and AI effective-count vectors must use the same action support."
        )

    p_H_before = carry_policy_forward(p_H_T, name="p_H_T", atol=atol)
    p_AI_before = carry_policy_forward(p_AI_T, name="p_AI_T", atol=atol)
    if p_H_before.shape != p_AI_before.shape:
        raise ValueError("Human and AI policies must use the same action support.")
    if p_H_before.size != H_counts_before.size:
        raise ValueError(
            "Policies and effective-count vectors must use the same action support."
        )

    if eta_kind == "exact_empirical":
        _validate_exact_empirical_predecision_identity(
            p_H_before,
            H_counts_before,
            name="p_H_T",
            atol=atol,
        )
        _validate_exact_empirical_predecision_identity(
            p_AI_before,
            AI_counts_before,
            name="p_AI_T",
            atol=atol,
        )

    eta_AI_T = _eta_for_agent(
        eta_kind=eta_kind,
        eta0=eta0,
        c=c,
        T=T,
        N_i_s_a_before=AI_counts_before,
        name="eta_AI_T",
    )
    # Theory: Part I Eq. (35) [monroeq]:
    p_AI_T_plus_1 = update_policy_toward_executed_action(
        p_AI_before,
        a_star_T=a_star_T,
        eta_i_T=eta_AI_T,
        name="p_AI_T_plus_1",
        atol=atol,
    )
    N_AI_s_a_T = increment_effective_action_counts(
        AI_counts_before,
        a_star_T=a_star_T,
        name="N_AI_s_a_before",
    )

    if eligibility.human_updated_T:
        eta_H_T = _eta_for_agent(
            eta_kind=eta_kind,
            eta0=eta0,
            c=c,
            T=T,
            N_i_s_a_before=H_counts_before,
            name="eta_H_T",
        )
        # Theory: Part I Eq. (35) [monroeq]: 
        p_H_T_plus_1 = update_policy_toward_executed_action(
            p_H_before,
            a_star_T=a_star_T,
            eta_i_T=eta_H_T,
            name="p_H_T_plus_1",
            atol=atol,
        )
        N_H_s_a_T = increment_effective_action_counts(
            H_counts_before,
            a_star_T=a_star_T,
            name="N_H_s_a_before",
        )
    else:
        eta_H_T = 0.0
        p_H_T_plus_1 = carry_policy_forward(
            p_H_before,
            name="p_H_T_plus_1",
            atol=atol,
        )
        N_H_s_a_T = H_counts_before.copy()

    if eta_kind == "exact_empirical":
        # Reconstruct the exact sample-frequency policies and compare them with
        # the recursive updates. Returning the reconstructed vectors removes
        # accumulated roundoff without changing the mathematical result.
        p_AI_expected = policy_from_effective_action_counts(
            N_AI_s_a_T,
            initial_policy=p_AI_before,
            name="p_AI_T_plus_1_exact",
            atol=atol,
        )
        if not np.allclose(
            p_AI_T_plus_1,
            p_AI_expected,
            rtol=0.0,
            atol=atol,
        ):
            raise ArithmeticError(
                "The AI exact-empirical recursion violated Lemma 1."
            )
        p_AI_T_plus_1 = p_AI_expected

        if eligibility.human_updated_T:
            p_H_expected = policy_from_effective_action_counts(
                N_H_s_a_T,
                initial_policy=p_H_before,
                name="p_H_T_plus_1_exact",
                atol=atol,
            )
            if not np.allclose(
                p_H_T_plus_1,
                p_H_expected,
                rtol=0.0,
                atol=atol,
            ):
                raise ArithmeticError(
                    "The Human exact-empirical recursion violated Lemma 1."
                )
            p_H_T_plus_1 = p_H_expected

    return Scenario1PolicyUpdate(
        p_H_T_plus_1=p_H_T_plus_1,
        p_AI_T_plus_1=p_AI_T_plus_1,
        N_H_s_a_T=N_H_s_a_T,
        N_AI_s_a_T=N_AI_s_a_T,
        eta_H_T=float(eta_H_T),
        eta_AI_T=float(eta_AI_T),
        human_updated_T=eligibility.human_updated_T,
        ai_updated_T=eligibility.ai_updated_T,
    )


__all__ = [
    "Scenario1PolicyUpdate",
    "Scenario1UpdateEligibility",
    "apply_scenario1_policy_updates",
    "scenario1_update_eligibility",
]
