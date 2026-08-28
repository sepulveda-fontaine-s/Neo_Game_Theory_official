'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


"""Operational settings for Scenario 1.

Scientific model objects and the complete parameter grid live in
``Scenario_1.utils_scenario_1.grid``.  This module contains only execution and
output settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from general_formulation.numerics import DEFAULT_ATOL

ACTIVE_H: Final[int] = 50
ATOL: Final[float] = DEFAULT_ATOL
PLOT_DPI: Final[int] = 200
PROGRESS_INTERVAL: Final[int] = 500
EXPECTED_WINNER_GROUPS: Final[int] = 6
PLOTS_PER_WINNER: Final[int] = 4
EXPECTED_PLOT_COUNT: Final[int] = EXPECTED_WINNER_GROUPS * PLOTS_PER_WINNER

SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
OUTPUT_DIR: Final[Path] = SCRIPT_DIR / f"Scenario1_outputs_H_{ACTIVE_H}"
ALL_RUNS_FILENAME: Final[str] = "scenario1_all_runs.csv"
BEST_JOINT_ALLRUNS_FILENAME: Final[str] = "scenario1_best_joint_allruns.csv"
BEST_JOINT_FINAL_FILENAME: Final[str] = "scenario1_best_joint_final.csv"
ALL_RUNS_PATH: Final[Path] = OUTPUT_DIR / ALL_RUNS_FILENAME
BEST_JOINT_ALLRUNS_PATH: Final[Path] = OUTPUT_DIR / BEST_JOINT_ALLRUNS_FILENAME
BEST_JOINT_FINAL_PATH: Final[Path] = OUTPUT_DIR / BEST_JOINT_FINAL_FILENAME

__all__ = [
    "ACTIVE_H", "ATOL", "PLOT_DPI", "PROGRESS_INTERVAL",
    "EXPECTED_WINNER_GROUPS", "PLOTS_PER_WINNER", "EXPECTED_PLOT_COUNT",
    "SCRIPT_DIR", "OUTPUT_DIR", "ALL_RUNS_FILENAME", "BEST_JOINT_ALLRUNS_FILENAME",
    "BEST_JOINT_FINAL_FILENAME", "ALL_RUNS_PATH", "BEST_JOINT_ALLRUNS_PATH",
    "BEST_JOINT_FINAL_PATH",
]
