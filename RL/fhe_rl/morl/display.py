"""
display.py — terminal colour helpers, Pareto table, and scatter plot.
"""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ── ANSI colours ───────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"


def _h(text: str)    -> str: return f"{BOLD}{CYAN}{text}{RESET}"
def _ok(text: str)   -> str: return f"{GREEN}{text}{RESET}"
def _warn(text: str) -> str: return f"{YELLOW}{text}{RESET}"
def _err(text: str)  -> str: return f"{RED}{text}{RESET}"


# ── table ──────────────────────────────────────────────────────────────────────

def display_pareto_table(front: list[dict]) -> None:
    """Pretty-print the Pareto frontier as a numbered table."""
    print()
    print(_h(
        f"{'#':>3}  {'w_exec':>7}  {'w_keys':>7}  "
        f"{'Exec cost':>10}  {'Keys cost':>10}"
    ))
    print(CYAN + "─" * 45 + RESET)
    for i, p in enumerate(front, 1):
        print(
            f"{BOLD}{i:>3}{RESET}  "
            f"{p['w_ops']:>7.4f}  "
            f"{p['w_keys']:>7.4f}  "
            f"{p['final_exec_cost']:>10.1f}  "
            f"{p['final_keys_cost']:>10.1f}"
        )
    print()


# ── plot ───────────────────────────────────────────────────────────────────────

def display_pareto_plot(front: list[dict], benchmark: str, slot_count: int) -> None:
    """Normalised scatter plot of the Pareto frontier."""
    if not _HAS_MPL:
        print(_warn("  matplotlib not installed — skipping plot"))
        return
    if not front:
        return

    ops_vals  = [p["final_exec_cost"]  for p in front]
    keys_vals = [p["final_keys_cost"] for p in front]

    ops_n  = ops_vals
    keys_n = keys_vals

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ops_n, keys_n, color="steelblue", zorder=5, s=80)

    for i, (xv, yv) in enumerate(zip(ops_n, keys_n), 1):
        ax.annotate(str(i), (xv, yv),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color="navy")

    ax.set_xlabel("Normalised Execution Cost", fontsize=11)
    ax.set_ylabel("Normalised Key Size",        fontsize=11)
    ax.set_title(
        f"Pareto Frontier — {benchmark} (slot={slot_count})",
        fontsize=12, fontweight="bold",
    )
    # ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    # ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Execution Cost", fontsize=11)
    ax.set_ylabel("Key Size",       fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()