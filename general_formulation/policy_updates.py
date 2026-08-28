'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Executed-action policy recursion and effective incorporation counts.

Scientific basis
----------------

- Part I Eqs. (30)--(31): the exact-empirical learning rate uses the total
  number of effective executed-action incorporations available before
  decision instant ``T``.
- Lemma 1 and Part I Eq. (32) [eq:exact_empirical_identity]: after at least
  one effective update, the exact-empirical policy equals the relative
  frequency of the incorporated executed actions.
- Remark 1(ii): before the first effective update, the supplied initial policy
  is used and carries no pseudocount mass.
- Part I Eq. (35) [monroeq]: an effective policy update moves the current
  policy toward the Dirac/one-hot vector of the executed action.
- Proposition 3: the convex-combination update leaves the finite policy
  simplex invariant.

This module implements only generic finite-action mechanics. It does not decide
whether the Human or AI policy is updated. Scenario-specific code must apply
the update mask, call ``update_policy_toward_executed_action`` only for an
effective update, and carry every other policy component forward unchanged.
"""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .learning_rates import _compute_eta_validated
from .numerics import (
    DEFAULT_ATOL,
    ProbabilityVector,
    as_probability,
    as_probability_vector,
    one_hot_probability_vector,
)


EffectiveCountVector: TypeAlias = NDArray[np.int64]


def _as_action_index(a_star_T: int, *, n_actions: int) -> int:
    """Return a validated executed-action index on the common support."""
    if isinstance(a_star_T, bool) or not isinstance(a_star_T, Integral):
        raise TypeError("a_star_T must be an integer action index.")

    action_index = int(a_star_T)
    if action_index < 0 or action_index >= n_actions:
        raise IndexError(
            f"a_star_T={action_index} is outside the action support "
            f"[0, {n_actions - 1}]."
        )
    return action_index


def as_effective_count_vector(
    values: ArrayLike,
    *,
    name: str = "N_i_s_a",
) -> EffectiveCountVector:
    """Return a strict one-dimensional vector of nonnegative integer counts.

    Floating-point values such as ``1.0`` are rejected rather than silently
    cast to integers. Effective incorporation counts are discrete state
    variables and must not contain pseudocounts, negative entries, or booleans.
    """
    raw = np.asarray(values, dtype=object)

    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if raw.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    int64_max = np.iinfo(np.int64).max
    validated: list[int] = []

    for index, item in enumerate(raw.tolist()):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
            raise TypeError(
                f"{name}[{index}] must be a nonnegative integer; "
                f"received {item!r}."
            )

        count = int(item)
        if count < 0:
            raise ValueError(f"{name}[{index}] must be nonnegative.")
        if count > int64_max:
            raise OverflowError(
                f"{name}[{index}] exceeds the maximum supported int64 count."
            )
        validated.append(count)

    return np.asarray(validated, dtype=np.int64)


def effective_update_count(
    N_i_s_a: ArrayLike,
    *,
    name: str = "N_i_s_a",
) -> int:
    """Return ``N_i(s,T-1)`` as the sum over action-specific counts.

    The returned value is a Python integer, so summing several valid int64
    action counts cannot overflow silently.
    """
    count_vector = as_effective_count_vector(N_i_s_a, name=name)
    return sum(int(count) for count in count_vector)


def increment_effective_action_counts(
    N_i_s_a_before: ArrayLike,
    *,
    a_star_T: int,
    name: str = "N_i_s_a_before",
) -> EffectiveCountVector:
    """Record one effective incorporation of the executed action.

    Parameters
    ----------
    N_i_s_a_before:
        Action-specific effective incorporation counts for one agent--state
        policy component before decision instant ``T``.
    a_star_T:
        Integer index of the executed action at decision instant ``T``.
    name:
        Variable name used in validation errors.

    Returns
    -------
    numpy.ndarray
        A new post-decision count vector. The input is never modified.
    """
    counts_after = as_effective_count_vector(
        N_i_s_a_before,
        name=name,
    ).copy()
    action_index = _as_action_index(a_star_T, n_actions=counts_after.size)

    if counts_after[action_index] == np.iinfo(np.int64).max:
        raise OverflowError(
            "The selected effective incorporation count cannot be incremented "
            "without int64 overflow."
        )

    counts_after[action_index] += 1
    return counts_after



def policy_from_effective_action_counts(
    N_i_s_a_before: ArrayLike,
    *,
    initial_policy: ArrayLike,
    name: str = "p_i_T",
    atol: float = DEFAULT_ATOL,
) -> ProbabilityVector:
    """Construct the exact-empirical policy available before decision ``T``.
    (Part I Lemma 1, Eq. (32) [eq:exact_empirical_identity].)
    If the total effective count is zero, the supplied initial policy is
    returned, consistently with Remark 1(ii). Once the total is positive, the
    result is the relative-frequency policy in Lemma 1 and Part I Eq. (32) 
    [eq:exact_empirical_identity]. The initial policy contributes no pseudocount mass """

    counts = as_effective_count_vector(
        N_i_s_a_before,
        name="N_i_s_a_before",
    )
    initial = as_probability_vector(
        initial_policy,
        name="initial_policy",
        atol=atol,
    )

    if counts.size != initial.size:
        raise ValueError(
            "N_i_s_a_before and initial_policy must use the same finite "
            f"action support; received sizes {counts.size} and {initial.size}."
        )

    N_i_before = effective_update_count(
        counts,
        name="N_i_s_a_before",
    )
    if N_i_before == 0:
        return initial.copy()

    p_i_T = counts.astype(np.float64) / float(N_i_before)
    return as_probability_vector(p_i_T, name=name, atol=atol)



def _update_binary_policy_inplace(
    policy: np.ndarray,
    counts: np.ndarray,
    *,
    action: int,
    eta_kind: str,
    eta0: float | None,
    c: float | None,
    T: int,
    atol: float = DEFAULT_ATOL,
) -> float:
    """Update one validated binary policy/count pair in place.

    The arithmetic mirrors the strict public recursion, including correction
    of unit-sum roundoff. Repeated type, shape, and allocation-heavy simplex
    validation is omitted.
    """
    count_before = int(counts[0]) + int(counts[1])
    eta = _compute_eta_validated(
        eta_kind,
        eta0=eta0,
        c=c,
        T=T,
        effective_count_before=count_before,
    )
    counts[action] += 1

    if eta_kind == "exact_empirical":
        total_count = float(count_before + 1)
        policy[0] = float(counts[0]) / total_count
        policy[1] = float(counts[1]) / total_count
    else:
        keep = 1.0 - eta
        policy *= keep
        policy[action] += eta

    total_probability = float(policy[0]) + float(policy[1])
    if abs(total_probability - 1.0) > atol:
        raise ArithmeticError(
            "Binary policy lost unit mass after a validated update: "
            f"{total_probability!r}."
        )
    if total_probability != 1.0:
        policy /= total_probability

    p0 = float(policy[0])
    p1 = float(policy[1])
    if p0 < -atol or p1 < -atol or p0 > 1.0 + atol or p1 > 1.0 + atol:
        raise ArithmeticError(
            "Binary policy left [0,1] after a validated update: "
            f"({p0!r}, {p1!r})."
        )
    return eta


def update_policy_toward_executed_action(
    p_i_T: ArrayLike,
    *,
    a_star_T: int,
    eta_i_T: float,
    name: str = "p_i_T_plus_1",
    atol: float = DEFAULT_ATOL,
) -> ProbabilityVector:
    """Apply the effective executed-action update from Part I Eq. (35) [monroeq]; 
    Part II Eq. (13) [eq:common_policy_update].

    This function is called only when an effective update occurs. Therefore
    ``eta_i_T`` must lie in ``(0, 1]``. A zero stored in the longitudinal CSV
    to indicate that an agent was not updated must never be passed here.

    The input policy is treated as the pre-decision policy
    ``p_i(· | s_T,T)``. The returned vector is the post-update policy
    ``p_i(· | s_T,T+1)`` for the current state. The input is never modified.
    """
    p_i_vector = as_probability_vector(p_i_T, name="p_i_T", atol=atol)
    action_index = _as_action_index(a_star_T, n_actions=p_i_vector.size)

    eta_value = as_probability(eta_i_T, name="eta_i_T", atol=atol)
    if eta_value <= 0.0:
        raise ValueError("An effective policy update requires eta_i_T in (0, 1].")

    delta_a_star_T = one_hot_probability_vector(
        p_i_vector.size,
        action_index,
    )

    # Part I Eq. (35) [monroeq]; Part II Eq. (13) [eq:common_policy_update]:
    p_i_T_plus_1 = (
        (1.0 - eta_value) * p_i_vector
        + eta_value * delta_a_star_T
    )

    # Proposition 3 guarantees simplex invariance theoretically. 
    # Validation here detects implementation errors while correcting only boundary-level  floating-point drift.
    return as_probability_vector(p_i_T_plus_1, name=name, atol=atol)


def carry_policy_forward(
    p_i_T: ArrayLike,
    *,
    name: str = "p_i_T_plus_1",
    atol: float = DEFAULT_ATOL,
) -> ProbabilityVector:
    """Carry an unmodified policy component from ``T`` to ``T+1``.

    This is used when the scenario-specific update mask disables an agent's
    update or when a policy belongs to a state different from ``s_T``.
    A new array is returned so callers cannot mutate the pre-decision policy
    through aliasing.
    """
    return as_probability_vector(p_i_T, name=name, atol=atol).copy()


__all__ = [
    "EffectiveCountVector",
    "as_effective_count_vector",
    "carry_policy_forward",
    "effective_update_count",
    "increment_effective_action_counts",
    "policy_from_effective_action_counts",
    "update_policy_toward_executed_action",
]
