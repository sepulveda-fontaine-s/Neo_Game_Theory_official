'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Self-contained CSV payload used by the sequential scenario chain.

Winner rows carry every numerical object required to initialize the next
scenario: terminal Human/AI policies, structural utilities, and the resolved
binary Virtual-Nature kernel.  Inherited grids therefore depend only on the
winner CSV and their own scenario-specific extension parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from general_formulation.grid_common import (
    BINARY_N_ACTIONS,
    BINARY_STATES,
    freeze_probability_table,
    freeze_structural_utility_table,
    freeze_transition_kernel,
)
from general_formulation.numerics import DEFAULT_ATOL
from general_formulation.state_generation import _binary_kernel_array_validated

CHAIN_POLICY_COLUMNS = (
    "chain_p_H_A_a0", "chain_p_H_A_a1",
    "chain_p_H_B_a0", "chain_p_H_B_a1",
    "chain_p_AI_A_a0", "chain_p_AI_A_a1",
    "chain_p_AI_B_a0", "chain_p_AI_B_a1",
)
CHAIN_UTILITY_COLUMNS = (
    "chain_U_H_A_a0", "chain_U_H_A_a1",
    "chain_U_H_B_a0", "chain_U_H_B_a1",
    "chain_U_AI_A_a0", "chain_U_AI_A_a1",
    "chain_U_AI_B_a0", "chain_U_AI_B_a1",
)
CHAIN_KERNEL_COLUMNS = (
    "chain_P_A_a0_A", "chain_P_A_a0_B",
    "chain_P_A_a1_A", "chain_P_A_a1_B",
    "chain_P_B_a0_A", "chain_P_B_a0_B",
    "chain_P_B_a1_A", "chain_P_B_a1_B",
)
CHAIN_COLUMNS = CHAIN_POLICY_COLUMNS + CHAIN_UTILITY_COLUMNS + CHAIN_KERNEL_COLUMNS


@dataclass(frozen=True, slots=True)
class ChainPayload:
    p_H_init: Mapping[str, tuple[float, ...]]
    p_AI_init: Mapping[str, tuple[float, ...]]
    U_H: Mapping[str, tuple[float, ...]]
    U_AI: Mapping[str, tuple[float, ...]]
    action_dependent_kernel: (
        Mapping[str, Mapping[int, Mapping[str, float]]] | None
    )


def chain_payload_fields(
    config: Any,
    *,
    p_H_final: Any,
    p_AI_final: Any,
) -> dict[str, float]:
    """Flatten one validated binary run into its next-scenario payload."""
    def as_array(values: Any) -> np.ndarray:
        if isinstance(values, Mapping):
            array = np.asarray([values["A"], values["B"]], dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.shape != (2, 2):
            raise ValueError("Chain policies must be 2x2 tables.")
        return array

    p_H_final = as_array(p_H_final)
    p_AI_final = as_array(p_AI_final)
    U_H = config.U_H
    U_AI = config.U_AI
    P_V = _binary_kernel_array_validated(config.P_V_key)
    return {
        "chain_p_H_A_a0": float(p_H_final[0, 0]),
        "chain_p_H_A_a1": float(p_H_final[0, 1]),
        "chain_p_H_B_a0": float(p_H_final[1, 0]),
        "chain_p_H_B_a1": float(p_H_final[1, 1]),
        "chain_p_AI_A_a0": float(p_AI_final[0, 0]),
        "chain_p_AI_A_a1": float(p_AI_final[0, 1]),
        "chain_p_AI_B_a0": float(p_AI_final[1, 0]),
        "chain_p_AI_B_a1": float(p_AI_final[1, 1]),
        "chain_U_H_A_a0": float(U_H["A"][0]),
        "chain_U_H_A_a1": float(U_H["A"][1]),
        "chain_U_H_B_a0": float(U_H["B"][0]),
        "chain_U_H_B_a1": float(U_H["B"][1]),
        "chain_U_AI_A_a0": float(U_AI["A"][0]),
        "chain_U_AI_A_a1": float(U_AI["A"][1]),
        "chain_U_AI_B_a0": float(U_AI["B"][0]),
        "chain_U_AI_B_a1": float(U_AI["B"][1]),
        "chain_P_A_a0_A": float(P_V[0, 0, 0]),
        "chain_P_A_a0_B": float(P_V[0, 0, 1]),
        "chain_P_A_a1_A": float(P_V[0, 1, 0]),
        "chain_P_A_a1_B": float(P_V[0, 1, 1]),
        "chain_P_B_a0_A": float(P_V[1, 0, 0]),
        "chain_P_B_a0_B": float(P_V[1, 0, 1]),
        "chain_P_B_a1_A": float(P_V[1, 1, 0]),
        "chain_P_B_a1_B": float(P_V[1, 1, 1]),
    }


def _required_float(row: Mapping[str, Any], name: str) -> float:
    if name not in row:
        raise ValueError(f"Inherited winner CSV is missing column {name!r}.")
    raw = row[name]
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"Inherited winner CSV column {name!r} is empty.")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"Inherited winner CSV column {name!r} is non-finite.")
    return value


def parse_chain_payload(
    row: Mapping[str, Any],
    *,
    random_states: bool,
    atol: float = DEFAULT_ATOL,
) -> ChainPayload:
    """Validate and reconstruct all numerical inheritance objects from one row."""
    p_H = freeze_probability_table(
        {
            "A": (_required_float(row, "chain_p_H_A_a0"), _required_float(row, "chain_p_H_A_a1")),
            "B": (_required_float(row, "chain_p_H_B_a0"), _required_float(row, "chain_p_H_B_a1")),
        },
        name="inherited p_H_init",
        states=BINARY_STATES,
        n_actions=BINARY_N_ACTIONS,
        uniform_if_none=False,
        atol=atol,
    )
    p_AI = freeze_probability_table(
        {
            "A": (_required_float(row, "chain_p_AI_A_a0"), _required_float(row, "chain_p_AI_A_a1")),
            "B": (_required_float(row, "chain_p_AI_B_a0"), _required_float(row, "chain_p_AI_B_a1")),
        },
        name="inherited p_AI_init",
        states=BINARY_STATES,
        n_actions=BINARY_N_ACTIONS,
        uniform_if_none=False,
        atol=atol,
    )
    U_H = freeze_structural_utility_table(
        {
            "A": (_required_float(row, "chain_U_H_A_a0"), _required_float(row, "chain_U_H_A_a1")),
            "B": (_required_float(row, "chain_U_H_B_a0"), _required_float(row, "chain_U_H_B_a1")),
        },
        name="inherited U_H",
        states=BINARY_STATES,
        n_actions=BINARY_N_ACTIONS,
    )
    U_AI = freeze_structural_utility_table(
        {
            "A": (_required_float(row, "chain_U_AI_A_a0"), _required_float(row, "chain_U_AI_A_a1")),
            "B": (_required_float(row, "chain_U_AI_B_a0"), _required_float(row, "chain_U_AI_B_a1")),
        },
        name="inherited U_AI",
        states=BINARY_STATES,
        n_actions=BINARY_N_ACTIONS,
    )
    if random_states:
        kernel = None
    else:
        kernel = freeze_transition_kernel(
            {
                "A": {
                    0: {"A": _required_float(row, "chain_P_A_a0_A"), "B": _required_float(row, "chain_P_A_a0_B")},
                    1: {"A": _required_float(row, "chain_P_A_a1_A"), "B": _required_float(row, "chain_P_A_a1_B")},
                },
                "B": {
                    0: {"A": _required_float(row, "chain_P_B_a0_A"), "B": _required_float(row, "chain_P_B_a0_B")},
                    1: {"A": _required_float(row, "chain_P_B_a1_A"), "B": _required_float(row, "chain_P_B_a1_B")},
                },
            },
            states=BINARY_STATES,
            n_actions=BINARY_N_ACTIONS,
            atol=atol,
        )
    return ChainPayload(
        p_H_init=p_H,
        p_AI_init=p_AI,
        U_H=U_H,
        U_AI=U_AI,
        action_dependent_kernel=kernel,
    )


__all__ = [
    "CHAIN_COLUMNS",
    "CHAIN_KERNEL_COLUMNS",
    "CHAIN_POLICY_COLUMNS",
    "CHAIN_UTILITY_COLUMNS",
    "ChainPayload",
    "chain_payload_fields",
    "parse_chain_payload",
]
