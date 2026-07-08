"""
interactive.py — CHEHAB-MORL interactive interface (prompts + orchestration only).

Two modes (Section 5.2.6 of the thesis):
  direct  — single preference -> one optimised circuit
  menu    — preference range  -> Pareto frontier + user selection + execution

Usage:
    python -m fhe_rl interactive
    python -m fhe_rl interactive --mode direct
    python -m fhe_rl interactive --mode menu
"""

from __future__ import annotations
from typing import Optional

from .benchmark_runner import (
    run_one_point, save_solution_to_he, cleanup_solutions_cache,
    discover_benchmarks, is_poly_benchmark,
    POLY_BENCHMARK,
)
from .bisection       import build_coarse_grid, bisect_interval, DEFAULT_MAX_BISECT_DEPTH
from .pareto_frontier import get_pareto_frontier
from .display         import (
    _h, _ok, _warn, _err,
    display_pareto_table, display_pareto_plot,
    BOLD, CYAN, MAGENTA, RESET,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{BOLD}{MAGENTA}{prompt}{suffix}: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    return val if val else default


def _pick_benchmark() -> str:
    benchmarks = discover_benchmarks()
    if not benchmarks:
        print(_warn("  No benchmarks found under build/benchmarks/."))
        return _ask("Enter benchmark name manually")

    print()
    print(_h("Available benchmarks:"))
    for i, b in enumerate(benchmarks, 1):
        print(f"  {i:>2}. {b}")

    while True:
        raw = _ask("Pick a number or type a name", benchmarks[0])
        if raw.isdigit() and 0 <= int(raw) - 1 < len(benchmarks):
            return benchmarks[int(raw) - 1]
        if raw in benchmarks:
            return raw
        print(_err("  Invalid choice, try again."))


def _pick_slot_count() -> int:
    """Ask the user to enter a slot count directly (no discovery)."""
    print()
    while True:
        raw = _ask("Slot count (e.g. 4, 8, 16, 32)", "16")
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print(_err("  Please enter a positive integer."))


def _pick_poly_args() -> dict:
    """Prompt for the extra arguments required by polynomials_coyote."""
    print()
    print(_h("polynomials_coyote parameters:"))
    tree_depth = int(_ask("  tree_depth", "5"))
    instance   = int(_ask("  instance",   "1"))
    regime     = _ask("  regime (e.g. 100-50)", "100-50")
    return {"tree_depth": tree_depth, "instance": instance, "regime": regime}


def _pick_w_ops(label: str = "w_exec (0-1)") -> float:
    while True:
        raw = _ask(label, "0.5")
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return round(val, 10)
        except ValueError:
            pass
        print(_err("  Must be a float in [0, 1]."))


def _pick_w_range() -> tuple[float, float]:
    print()
    print(_h("Define the w_exec search range  (w_keys = 1 - w_exec)"))
    while True:
        lo = _pick_w_ops("  lower bound w_exec_lo")
        hi = _pick_w_ops("  upper bound w_exec_hi")
        if lo < hi:
            return lo, hi
        print(_err("  Lower bound must be strictly less than upper bound."))


def _pick_bisect_depth() -> int:
    while True:
        raw = _ask("Maximum bisection depth", str(DEFAULT_MAX_BISECT_DEPTH))
        try:
            val = int(raw)
            if val >= 0:
                return val
        except ValueError:
            pass
        print(_err("  Must be a non-negative integer."))


# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECT MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_direct_mode() -> None:
    print()
    print(_h("---  DIRECT MODE  ---"))
    print("Generate a single optimised FHE circuit for a specific preference.\n")

    benchmark  = _pick_benchmark()
    extra_args = _pick_poly_args() if is_poly_benchmark(benchmark) else None
    slot_count = 0 if is_poly_benchmark(benchmark) else _pick_slot_count()
    w_ops      = _pick_w_ops("w_exec (execution weight)")

    print()
    print(_warn(f"  Running {benchmark}  slot={slot_count}  "
                f"w_exec={w_ops:.4f}  w_keys={round(1-w_ops,4):.4f} ..."))

    result = run_one_point(benchmark, slot_count, w_ops, extra_args)
    if result is None:
        print(_err("  Benchmark timed out or failed."))
        return

    print()
    print(_ok("  --- Result ---"))
    print(f"  Execution cost : {result['final_exec_cost']}")
    print(f"  Key size cost  : {result['final_keys_cost']}")
    print(_ok("  ---------------"))

    save_solution_to_he(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  MENU MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_menu_mode() -> None:
    print()
    print(_h("---  MENU MODE  (Pareto Frontier)  ---"))
    print("Scan a preference range, find hidden trade-offs, pick your circuit.\n")

    benchmark  = _pick_benchmark()
    extra_args = _pick_poly_args() if is_poly_benchmark(benchmark) else None
    slot_count = 0 if is_poly_benchmark(benchmark) else _pick_slot_count()
    w_lo, w_hi = _pick_w_range()
    max_depth  = _pick_bisect_depth()

    # ── Phase 1: coarse grid ──────────────────────────────────────────────────
    grid = build_coarse_grid(w_lo, w_hi)
    print()
    print(_h(f"Phase 1 - {len(grid)} coarse-grid points: "
             + "  ".join(f"{w:.4f}" for w in grid)))

    all_results: list[dict] = []
    for w in grid:
        print(_warn(f"\n  w_exec={w:.4f}  w_keys={round(1-w,4):.4f} ..."))
        r = run_one_point(benchmark, slot_count, w, extra_args)
        if r is not None:
            all_results.append(r)
            print(f"    ops={r['final_exec_cost']}  keys={r['final_keys_cost']}")
        else:
            print(_err("    timed out - skipped"))

    # ── Phase 2: bisection ────────────────────────────────────────────────────
    if max_depth > 0 and len(all_results) >= 2:
        print()
        print(_h("Phase 2 - adaptive bisection on key-count gaps ..."))
        ordered = sorted(all_results, key=lambda r: r["w_ops"])
        for i in range(len(ordered) - 1):
            lo_pt, hi_pt = ordered[i], ordered[i + 1]
            ks_lo = lo_pt.get("final_keys_cost")
            ks_hi = hi_pt.get("final_keys_cost")
            if ks_lo is None or ks_hi is None or int(ks_lo) == int(ks_hi):
                continue
            bisect_interval(
                benchmark, slot_count,
                hi_pt["w_ops"], int(ks_hi),
                lo_pt["w_ops"], int(ks_lo),
                all_results, max_depth,
                extra_args=extra_args,
            )

    # ── Pareto frontier ───────────────────────────────────────────────────────
    front = get_pareto_frontier(all_results)
    if not front:
        print(_err("  No valid solutions found."))
        return

    print()
    print(_h(f"Pareto Frontier  ({len(front)} non-dominated solutions):"))
    display_pareto_table(front)
    display_pareto_plot(front, benchmark, slot_count)

    # ── User selection ────────────────────────────────────────────────────────
    print(_h("Select a solution:"))
    print("  1. Pick by solution number")
    print("  2. Pick by custom w_exec weight")
    print("  3. Skip")

    chosen: Optional[dict] = None
    mode_raw = _ask("\nSelection method", "1")

    if mode_raw == "3" or not mode_raw:
        print("  No selection - exiting.")
        return
    elif mode_raw == "1":
        raw = _ask(f"Solution number (1-{len(front)})", "1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(front):
                chosen = front[idx]
            else:
                print(_err(f"  Index out of range (1-{len(front)})."))
                return
        except ValueError:
            print(_err("  Invalid number."))
            return
    elif mode_raw == "2":
        raw = _ask("Custom w_exec (0-1)", "0.5")
        try:
            cw = float(raw)
            if not 0.0 <= cw <= 1.0:
                raise ValueError
            cwk = round(1.0 - cw, 10)
            chosen = min(front, key=lambda p: abs(p["w_ops"] - cw))
            print(_ok(f"  w_exec={cw:.4f} -> nearest solution is "
                      f"#{front.index(chosen)+1}  "
                      f"(w_exec={chosen['w_ops']:.4f})"))
        except ValueError:
            print(_err("  Must be a float in [0, 1]."))
            return
    else:
        print(_err("  Invalid choice."))
        return

    if chosen is None:
        return

    print()
    print(_ok(f"  Selected solution #{front.index(chosen)+1}:"))
    print(f"    w_exec={chosen['w_ops']:.4f}  "
          f"ops={chosen['final_exec_cost']}  "
          f"keys={chosen['final_keys_cost']}")

    
    save_solution_to_he(chosen)
    cleanup_solutions_cache()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_interactive(mode: Optional[str] = None) -> None:
    print()
    print(BOLD + CYAN + "CHEHAB-MORL  Interactive Search" + RESET)
    print(BOLD + CYAN + "================================" + RESET)

    if mode is None:
        print()
        print("  1. Direct Mode  - single preference -> one optimised circuit")
        print("  2. Menu Mode    - preference range  -> Pareto frontier + selection")
        while True:
            raw = _ask("\nChoose mode", "1")
            if raw == "1":  mode = "direct"; break
            if raw == "2":  mode = "menu";   break
            print(_err("  Enter 1 or 2."))

    if mode == "direct":
        run_direct_mode()
    elif mode == "menu":
        run_menu_mode()
    else:
        print(_err(f"Unknown mode '{mode}'."))


def add_subparser(subparsers) -> None:
    """Register the 'interactive' sub-command on the existing argparse object."""
    p = subparsers.add_parser(
        "interactive",
        help="Interactive Direct / Menu mode for Pareto exploration",
    )
    p.add_argument(
        "--mode",
        dest="interactive_mode",
        choices=["direct", "menu"],
        default=None,
        help="Skip the mode prompt",
    )