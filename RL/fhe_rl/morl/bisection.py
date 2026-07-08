"""
bisection.py — coarse-grid generation and recursive bisection search for
hidden Pareto trade-offs (Algorithm 3 in the thesis).
"""

from __future__ import annotations
from typing import Optional

from .benchmark_runner import run_one_point
from .display import _warn, _err

DEFAULT_MAX_BISECT_DEPTH = 3
_COARSE_STEP = 0.1


# ── coarse grid ────────────────────────────────────────────────────────────────

def build_coarse_grid(w_lo: float, w_hi: float) -> list[float]:
    """
    Build the evaluation grid within [w_lo, w_hi]:
      - Always includes w_lo and w_hi.
      - If the range is wider than 0.1, interior multiples of 0.1 are added
        (e.g. 0.72-0.81 -> [0.72, 0.80, 0.81]).
      - Narrow ranges (<= 0.1) evaluate only the two endpoints
        (e.g. 0.72-0.79 -> [0.72, 0.79]).
    """
    points = {round(w_lo, 10), round(w_hi, 10)}
    if w_hi - w_lo > _COARSE_STEP + 1e-9:
        cursor = round((w_lo // _COARSE_STEP + 1) * _COARSE_STEP, 2)
        while cursor < w_hi - 1e-9:
            points.add(round(cursor, 10))
            cursor = round(cursor + _COARSE_STEP, 2)
    return sorted(points)


# ── recursive bisection ────────────────────────────────────────────────────────

def _bisect(
    benchmark:  str,
    slot_count: int,
    w_hi:       float,
    ks_hi:      int,
    w_lo:       float,
    ks_lo:      int,
    target:     int,
    found:      set,
    all_results: list[dict],
    max_depth:  int,
    extra_args: Optional[dict] = None,
    depth:      int = 0,
) -> None:
    if depth >= max_depth or target in found:
        return

    w_mid = round((w_hi + w_lo) / 2, 10)
    print(f"  {'  ' * depth}[bisect d={depth}] w_mid={w_mid:.6f}  target_keys={target}")

    result = run_one_point(benchmark, slot_count, w_mid, extra_args)
    if result is None:
        return

    all_results.append(result)
    ks_mid = (int(result["final_keys_cost"])
              if result["final_keys_cost"] is not None else None)
    if ks_mid is not None:
        found.add(ks_mid)

    print(f"  {'  ' * depth}           -> ops={result['final_exec_cost']:.1f}  "
          f"keys={result['final_keys_cost']}")

    if ks_mid is None or ks_mid == target:
        return

    if ks_mid < target:
        _bisect(benchmark, slot_count, w_hi, ks_hi, w_mid, ks_mid,
                target, found, all_results, max_depth, extra_args, depth + 1)
    else:
        _bisect(benchmark, slot_count, w_mid, ks_mid, w_lo, ks_lo,
                target, found, all_results, max_depth, extra_args, depth + 1)


def bisect_interval(
    benchmark:   str,
    slot_count:  int,
    w_hi:        float,
    ks_hi:       int,
    w_lo:        float,
    ks_lo:       int,
    all_results: list[dict],
    max_depth:   int,
    extra_args:  Optional[dict] = None,
) -> None:
    """
    Search independently for every missing integer key-size level between
    ks_lo and ks_hi. Each target gets its own max_depth budget.
    """
    lo, hi  = min(ks_lo, ks_hi), max(ks_lo, ks_hi)
    targets = list(range(lo + 1, hi))
    if not targets:
        return

    found: set = set()
    for t in targets:
        if t in found:
            print(f"  [skip] target={t} already found by a previous probe")
            continue
        print(_warn(f"\n  searching for keys={t} in w_ops in [{w_lo:.4f}, {w_hi:.4f}]"))
        _bisect(benchmark, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                t, found, all_results, max_depth, extra_args)