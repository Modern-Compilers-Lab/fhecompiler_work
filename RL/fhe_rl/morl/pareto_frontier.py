"""
pareto_frontier.py — Pareto dominance filtering for interactive search results.

Named pareto_frontier.py to avoid collision with the existing pareto.py
(which contains generate_pref_list used during training).
"""

from __future__ import annotations


def is_dominated(point: dict, others: list[dict]) -> bool:
    """Return True if `point` is strictly dominated by at least one entry in `others`."""
    ops  = point.get("final_exec_cost")
    keys = point.get("final_keys_cost")
    if ops is None or keys is None:
        return True
    for o in others:
        if o is point:
            continue
        o_ops  = o.get("final_exec_cost")
        o_keys = o.get("final_keys_cost")
        if o_ops is None or o_keys is None:
            continue
        if o_ops <= ops and o_keys <= keys and (o_ops < ops or o_keys < keys):
            return True
    return False


def get_pareto_frontier(points: list[dict]) -> list[dict]:
    """Return non-dominated points sorted by execution cost (ascending)."""
    valid = [
        p for p in points
        if p.get("final_exec_cost")  is not None
        and p.get("final_keys_cost") is not None
    ]
    front = [p for p in valid if not is_dominated(p, valid)]
    seen = set()
    deduped = []
    for p in front:
        key = (p["final_exec_cost"], p["final_keys_cost"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    front = deduped    
    return sorted(front, key=lambda p: p["final_exec_cost"])