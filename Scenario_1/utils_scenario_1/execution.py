'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario 1 proposal sampling and executed-action resolution.

Under Human arbitration, Part I Eqs. (42) and (46) imply that
``lambda_T=1`` selects the Human proposal and ``lambda_T=0`` selects
the AI proposal. The proposal owner, defined separately in Part I
Definition 6, is recorded explicitly so that coincident Human and AI
action values do not erase proposal origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from general_formulation.numerics import DEFAULT_ATOL, as_probability_vector

from .delegation import LAMBDA_AI, LAMBDA_HUMAN


OWNER_HUMAN = "H"
OWNER_AI = "AI"
OWNER_VALUES = (OWNER_HUMAN, OWNER_AI)


@dataclass(frozen=True, slots=True)
class ProposalPair:
    """Human and AI actions sampled from the predecision policies."""

    a_H_T: int
    a_AI_T: int


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """Executor, owner, and action selected by ``lambda_T``."""

    lambda_T: int
    executor_T: str
    owner_T: str
    a_star_T: int


def _as_rng(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    return rng


def _as_lambda(lambda_T: int) -> int:
    if isinstance(lambda_T, (bool, np.bool_)) or not isinstance(lambda_T, Integral):
        raise TypeError("lambda_T must be the integer 0 or 1.")
    value = int(lambda_T)
    if value not in (LAMBDA_AI, LAMBDA_HUMAN):
        raise ValueError("lambda_T must equal 0 (AI) or 1 (Human).")
    return value


def _as_action_index(value: int, *, n_actions: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer action index.")
    action = int(value)
    if action < 0 or action >= n_actions:
        raise IndexError(
            f"{name}={action} is outside the action support [0, {n_actions - 1}]."
        )
    return action


def sample_policy_action(
    policy: ArrayLike,
    *,
    rng: np.random.Generator,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> int:
    """Sample one action index from a validated finite policy."""
    generator = _as_rng(rng)
    probabilities = as_probability_vector(policy, name=name, atol=atol)
    return int(generator.choice(probabilities.size, p=probabilities))


def sample_scenario1_proposals(
    p_H_s_T: ArrayLike,
    p_AI_s_T: ArrayLike,
    *,
    rng: np.random.Generator,
    atol: float = DEFAULT_ATOL,
) -> ProposalPair:
    """Sample Human and AI proposals from the policies available at ``T``."""
    generator = _as_rng(rng)
    p_H = as_probability_vector(p_H_s_T, name="p_H_s_T", atol=atol)
    p_AI = as_probability_vector(p_AI_s_T, name="p_AI_s_T", atol=atol)
    if p_H.shape != p_AI.shape:
        raise ValueError(
            "p_H_s_T and p_AI_s_T must use the same action support."
        )

    return ProposalPair(
        a_H_T=int(generator.choice(p_H.size, p=p_H)),
        a_AI_T=int(generator.choice(p_AI.size, p=p_AI)),
    )


def resolve_scenario1_execution(
    *,
    lambda_T: int,
    a_H_T: int,
    a_AI_T: int,
    n_actions: int,
) -> ExecutionDecision:
    """Resolve executor, owner, and executed action without origin ambiguity."""
    if isinstance(n_actions, (bool, np.bool_)) or not isinstance(n_actions, Integral):
        raise TypeError("n_actions must be a positive integer.")
    action_count = int(n_actions)
    if action_count <= 0:
        raise ValueError("n_actions must be a positive integer.")

    selector = _as_lambda(lambda_T)
    human_action = _as_action_index(a_H_T, n_actions=action_count, name="a_H_T")
    ai_action = _as_action_index(a_AI_T, n_actions=action_count, name="a_AI_T")

    if selector == LAMBDA_HUMAN:
        owner = OWNER_HUMAN
        executed_action = human_action
    else:
        owner = OWNER_AI
        executed_action = ai_action

    return ExecutionDecision(
        lambda_T=selector,
        executor_T=owner,
        owner_T=owner,
        a_star_T=executed_action,
    )


def action_label(action_index: int, *, n_actions: int = 2) -> str:
    """Return the canonical CSV label ``a0``, ``a1``, ... for an action index."""
    action = _as_action_index(
        action_index,
        n_actions=n_actions,
        name="action_index",
    )
    return f"a{action}"


__all__ = [
    "ExecutionDecision",
    "OWNER_AI",
    "OWNER_HUMAN",
    "OWNER_VALUES",
    "ProposalPair",
    "action_label",
    "resolve_scenario1_execution",
    "sample_policy_action",
    "sample_scenario1_proposals",
]
