'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Validation of scalar probabilities; validation of probability vectors;
construction of uniform distributions; construction of one-hot vectors;
simplex invariance checks and numerical-tolerance handling.

Scientific basis
----------------
The operational rules are taken from ``Part_I``:

- The Policy Learning subsection, Part I Eq. (35) [monroeq], and
  Proposition 3 establish that every effective update preserves the
  policy in the finite-action probability simplex ``Delta(A)``, namely,
  the set of all admissible policies over the action set ``A``. In other
  words, the updated policy remains a valid probability vector: all
  components are nonnegative and their sum equals one.
- Remark 1(ii): a missing initial policy is initialized uniformly and this
  numerical initialization carries no pseudocount mass.
- Lemma 1 and Equations (32) [eq:exact_empirical_identity]: the exact-empirical schedule can produce
  pure one-hot policies, so exact zeros and ones must remain admissible.
- Part I Eq. (35) [monroeq], Part II Eq. (13).: an effective policy update uses 
  the Dirac/one-hot vector of the executed action.
- Proposition 3: the convex-combination update leaves the policy simplex
  invariant.

This module does not implement learning, delegation, or entropy. It only
validates and constructs the probability vectors used by those modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


ProbabilityVector: TypeAlias = NDArray[np.float64]

# Explicit tolerance for floating-point validation. This value is a numerical
# implementation safeguard; Part I does not assign a separate equation or
# named result to a machine-precision tolerance.
DEFAULT_ATOL = 1e-12


class ProbabilityValidationError(ValueError):
    """Raised when a scalar or vector cannot represent a probability."""


def _validated_atol(atol: float) -> float:
    """Return a finite, nonnegative absolute tolerance."""
    value = float(atol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("atol must be a finite, nonnegative number.")
    return value


def as_probability(value: float, *, name: str = "probability", atol: float = DEFAULT_ATOL) -> float:
    """Validate one probability and correct only boundary-level roundoff.

    Values outside ``[0, 1]`` by more than ``atol`` are rejected. A value that
    differs from 0 or 1 only because of floating-point roundoff is clipped to
    the corresponding boundary.
    """
    tolerance = _validated_atol(atol)
    probability = float(value)

    if not np.isfinite(probability):
        raise ProbabilityValidationError(f"{name} must be finite.")
    if probability < -tolerance or probability > 1.0 + tolerance:
        raise ProbabilityValidationError(
            f"{name} must lie in [0, 1]; received {probability!r}."
        )

    return float(np.clip(probability, 0.0, 1.0))


def as_probability_vector(
    values: ArrayLike,
    *,
    name: str = "probability vector",
    atol: float = DEFAULT_ATOL,
) -> ProbabilityVector:
    """Return a validated one-dimensional probability vector.

    The function is deliberately strict: arbitrary positive weights are not
    silently converted into a policy. The supplied components must already
    sum to one within ``atol``. Only floating-point drift at the boundaries or
    in the total mass is corrected.

    No pseudocount or smoothing mass is introduced, and the input object is
    never modified.
    """
    tolerance = _validated_atol(atol)
    vector = np.asarray(values, dtype=np.float64)

    if vector.ndim != 1:
        raise ProbabilityValidationError(f"{name} must be one-dimensional.")
    if vector.size == 0:
        raise ProbabilityValidationError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(vector)):
        raise ProbabilityValidationError(f"{name} must contain only finite values.")

    lower_violation = vector < -tolerance
    upper_violation = vector > 1.0 + tolerance
    if np.any(lower_violation) or np.any(upper_violation):
        raise ProbabilityValidationError(
            f"Every component of {name} must lie in [0, 1] within atol={tolerance:g}."
        )

    # Work on a copy and correct only roundoff that crossed a simplex boundary.
    result = vector.copy()
    result[result < 0.0] = 0.0
    result[result > 1.0] = 1.0

    total = float(np.sum(result, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=tolerance):
        raise ProbabilityValidationError(
            f"{name} must sum to 1 within atol={tolerance:g}; received {total!r}."
        )

    # Remove only accumulated floating-point drift. This is not a conversion
    # from arbitrary weights because the unit-sum condition was checked first.
    if total != 1.0:
        result /= total

    return result


@lru_cache(maxsize=64)
def _binary_table_array_validated(
    table_key: tuple[tuple[float, float], tuple[float, float]],
) -> NDArray[np.float64]:
    """Return one cached read-only 2x2 table from validated scalar values.

    The exhaustive simulation uses this internal primitive only after the
    public grid boundary has validated the table.  Mutable policies take a
    per-run ``copy()``; immutable structural utility tables are shared.
    """
    array = np.asarray(table_key, dtype=np.float64)
    if array.shape != (2, 2):
        raise RuntimeError("The validated binary table must have shape (2, 2).")
    array.setflags(write=False)
    return array


def uniform_probability_vector(n_actions: int) -> ProbabilityVector:
    """Construct the uniform finite-action policy from Remark 1(ii)."""
    if isinstance(n_actions, bool) or not isinstance(n_actions, (int, np.integer)):
        raise TypeError("n_actions must be an integer.")
    if int(n_actions) <= 0:
        raise ValueError("n_actions must be strictly positive.")

    size = int(n_actions)
    return np.full(size, 1.0 / float(size), dtype=np.float64)


def one_hot_probability_vector(n_actions: int, action_index: int) -> ProbabilityVector:
    """Construct the Dirac/one-hot vector used in Part I Eq. (35) [monroeq]
    and Part II Eq. (13) [eq:common_policy_update]: """

    if isinstance(n_actions, bool) or not isinstance(n_actions, (int, np.integer)):
        raise TypeError("n_actions must be an integer.")
    if isinstance(action_index, bool) or not isinstance(action_index, (int, np.integer)):
        raise TypeError("action_index must be an integer.")

    size = int(n_actions)
    index = int(action_index)

    if size <= 0:
        raise ValueError("n_actions must be strictly positive.")
    if index < 0 or index >= size:
        raise IndexError(
            f"action_index={index} is outside the action support [0, {size - 1}]."
        )

    vector = np.zeros(size, dtype=np.float64)
    vector[index] = 1.0
    return vector


__all__ = [
    "DEFAULT_ATOL",
    "ProbabilityValidationError",
    "ProbabilityVector",
    "as_probability",
    "as_probability_vector",
    "one_hot_probability_vector",
    "uniform_probability_vector",
]
