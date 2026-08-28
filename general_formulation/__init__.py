'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Scenario-independent computational primitives for Neo-Game Theory.

The package separates mathematical operations from the scenario-specific implementations of Scenarios 1, 2, and 3,
grid construction, simulation orchestration, selection, and output schemas.
Import functions from their defining modules so scientific dependencies remain
explicit and circular imports are avoided.
"""

__all__ = [
    "bellman",
    "csv_outputs",
    "entropy",
    "frequencies",
    "grid_common",
    "identifiers",
    "learning_rates",
    "numerics",
    "plots",
    "policy_updates",
    "state_generation",
    "utility_credit",
    "validation",
]
