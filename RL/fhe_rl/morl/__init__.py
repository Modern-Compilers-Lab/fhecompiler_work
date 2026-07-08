"""
fhe_rl/interactive/
-------------------
Sub-package for CHEHAB-MORL interactive inference modes.

Modules
-------
display.py          — terminal colours, Pareto table, scatter plot
pareto_frontier.py  — dominance filtering, get_pareto_frontier
benchmark_runner.py — run_one_point, execute_solution, benchmark discovery
bisection.py        — coarse-grid builder, recursive bisection search
interactive.py      — prompt helpers, Direct Mode, Menu Mode, CLI entry point
"""

from .morl import run_interactive, add_subparser

__all__ = ["run_interactive", "add_subparser"]