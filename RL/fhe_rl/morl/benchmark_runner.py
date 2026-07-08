"""
benchmark_runner.py — run compiled FHE benchmark binaries and manage solutions.

Key design
----------
- run_one_point()        : runs the binary once, streams output live, parses the
                           two costs, then snapshots he/_gen_he_fhe.{cpp,hpp}
                           into a solutions/ cache keyed by (benchmark, slot, w_ops).
- save_solution_to_he()  : copies the chosen cached .cpp/.hpp back into he/ —
                           no rebuild, no execution.
- cleanup_solutions_cache(): removes the entire solutions/ cache directory.

Polynomials-coyote special case
---------------------------------
The binary has a different argument order:
    ./<name> <tree_depth> <instance> <regime> <vectorize> <opt_method>
             <window> 1 <cse> 1 <w_ops> <w_keys>
Pass extra_args={"tree_depth":…, "instance":…, "regime":…} to both functions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional

from .display import _ok, _warn, _err

# ── config ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_FOLDER = os.path.join(_HERE, "..", "..", "..", "build", "benchmarks")
SOLUTIONS_FOLDER        = "solutions"          # root cache dir for saved circuits
COMPILE_TIMEOUT_SECONDS = 7200
VECTORIZE_CODE          = 1
OPTIMIZATION_METHOD     = 1                    # 1 = RL
WINDOW_SIZE             = 0
CSE_ENABLED             = 1

POLY_BENCHMARK          = "polynomials_coyote"


# ── discovery ──────────────────────────────────────────────────────────────────

def discover_benchmarks() -> list[str]:
    """Return all sub-folder names found under BUILD_FOLDER."""
    if not os.path.isdir(BUILD_FOLDER):
        return []
    return sorted(
        d for d in os.listdir(BUILD_FOLDER)
        if os.path.isdir(os.path.join(BUILD_FOLDER, d))
    )


def is_poly_benchmark(benchmark: str) -> bool:
    return benchmark == POLY_BENCHMARK


# ── command builders ───────────────────────────────────────────────────────────

def _build_cmd(benchmark: str, slot_count: int, w_ops: float,
               extra_args: Optional[dict] = None) -> str:
    """
    Build the shell command string for the benchmark binary.

    Regular benchmarks:
        ./<name> <vectorize> <slot_count> <opt_method> <window> 1 <cse> 1 <w_ops> <w_keys>

    polynomials_coyote:
        ./<name> <tree_depth> <instance> <regime> <vectorize> <opt_method>
                 <window> 1 <cse> 1 <w_ops> <w_keys>
        extra_args must contain: tree_depth, instance, regime
    """
    w_keys = round(1.0 - w_ops, 10)
    if is_poly_benchmark(benchmark):
        ea = extra_args or {}
        tree_depth = ea.get("tree_depth", 5)
        instance   = ea.get("instance",   1)
        regime     = ea.get("regime",     "100-50")
        return (
            f"./{benchmark} {tree_depth} {instance} {regime} "
            f"{VECTORIZE_CODE} {OPTIMIZATION_METHOD} {WINDOW_SIZE} "
            f"1 {CSE_ENABLED} 1 {w_ops} {w_keys}"
        )
    else:
        return (
            f"./{benchmark} {VECTORIZE_CODE} {slot_count} "
            f"{OPTIMIZATION_METHOD} {WINDOW_SIZE} 1 {CSE_ENABLED} 1 "
            f"{w_ops} {w_keys}"
        )


# ── solution cache helpers ─────────────────────────────────────────────────────

def _solution_dir(benchmark: str, slot_count: int, w_ops: float,
                  extra_args: Optional[dict] = None) -> str:
    """
    Return the directory where the generated HE files for one point are stored.

    Regular  : solutions/<benchmark>/<slot_count>/w_<w_ops>/
    Poly     : solutions/<benchmark>/<regime>/<tree_depth>/<instance>/w_<w_ops>/
    """
    w_tag = f"w_{w_ops:.6f}"
    if is_poly_benchmark(benchmark) and extra_args:
        regime     = extra_args.get("regime",     "100-50")
        tree_depth = extra_args.get("tree_depth", 5)
        instance   = extra_args.get("instance",   1)
        return os.path.join(
            SOLUTIONS_FOLDER, benchmark,
            str(regime), str(tree_depth), str(instance), w_tag,
        )
    return os.path.join(SOLUTIONS_FOLDER, benchmark, str(slot_count), w_tag)


def _save_solution(build_path_he: str, sol_dir: str) -> bool:
    """
    Copy _gen_he_fhe.cpp and _gen_he_fhe.hpp from he/ into sol_dir.
    Returns True on success.
    """
    os.makedirs(sol_dir, exist_ok=True)
    saved = False
    for ext in ("cpp", "hpp"):
        src = os.path.join(build_path_he, f"_gen_he_fhe.{ext}")
        dst = os.path.join(sol_dir,       f"_gen_he_fhe.{ext}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            saved = True
        else:
            print(_warn(f"  [save] {src} not found — skipping"))
    return saved


def _restore_solution(sol_dir: str, build_path_he: str) -> bool:
    """
    Copy the cached .cpp/.hpp back into he/ so the HE binary can be rebuilt.
    Returns True if both files were restored.
    """
    ok = True
    for ext in ("cpp", "hpp"):
        src = os.path.join(sol_dir,       f"_gen_he_fhe.{ext}")
        dst = os.path.join(build_path_he, f"_gen_he_fhe.{ext}")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            print(_err(f"  [restore] {src} not found"))
            ok = False
    return ok


# ── core runner ────────────────────────────────────────────────────────────────

def run_one_point(
    benchmark:   str,
    slot_count:  int,
    w_ops:       float,
    extra_args:  Optional[dict] = None,
) -> Optional[dict]:
    """
    Run the compiled benchmark binary once.
    Streams every output line to the terminal and parses final ops/keys costs.
    After a successful run the generated he/_gen_he_fhe.{cpp,hpp} are copied
    into the solutions cache so execute_solution() can reuse them without
    re-optimising.
    """
    w_keys     = round(1.0 - w_ops, 10)
    build_path = os.path.join(BUILD_FOLDER, benchmark)
    build_path_he = os.path.join(build_path, "he")

    # optional source generation (regular benchmarks only)
    if not is_poly_benchmark(benchmark):
        gen_script = os.path.join(build_path, f"generate_{benchmark}.py")
        if os.path.isfile(gen_script):
            proc = subprocess.Popen(
                ["python3", gen_script, "--slot_count", str(slot_count)],
                cwd=build_path,
            )
            proc.wait()

    cmd = _build_cmd(benchmark, slot_count, w_ops, extra_args)

    final_ops_cost:  Optional[float] = None
    final_keys_cost: Optional[float] = None

    try:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, cwd=build_path,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            clean_lower = clean.lower()
            if 'final exec cost' in clean_lower:
                try:
                    final_ops_cost = float(clean.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass
            if 'final keys cost' in clean_lower:
                try:
                    final_keys_cost = float(clean.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass

        proc.wait(timeout=COMPILE_TIMEOUT_SECONDS)

    except subprocess.TimeoutExpired:
        proc.kill()
        print(_err(f"\n  Timed out after {COMPILE_TIMEOUT_SECONDS}s"))
        return None
    except Exception as e:
        print(_err(f"\n  Error: {e}"))
        return None

    if final_ops_cost is None or final_keys_cost is None:
        print(_err("  Could not parse costs from output."))
        return None

    # ── snapshot the generated HE files ───────────────────────────────────────
    sol_dir = _solution_dir(benchmark, slot_count, w_ops, extra_args)
    if _save_solution(build_path_he, sol_dir):
        print(_ok(f"  [saved] circuit → {sol_dir}"))
    else:
        print(_warn("  [save] no generated files found to cache"))

    return {
        "benchmark":       benchmark,
        "slot_count":      slot_count,
        "w_ops":           w_ops,
        "w_keys":          w_keys,
        "final_exec_cost":  final_ops_cost,
        "final_keys_cost": final_keys_cost,
        "sol_dir":         sol_dir,        # carry the path for execute_solution
        "extra_args":      extra_args,
    }


# ── save chosen solution back into he/ ────────────────────────────────────────

def save_solution_to_he(point: dict) -> None:
    """
    Copy the cached _gen_he_fhe.{cpp,hpp} for the chosen solution back into
    build/benchmarks/<benchmark>/he/ — no rebuild, no execution.
    """
    benchmark  = point["benchmark"]
    slot_count = point["slot_count"]
    w_ops      = point["w_ops"]
    w_keys     = point["w_keys"]
    extra_args = point.get("extra_args")
    sol_dir    = point.get("sol_dir") or _solution_dir(
        benchmark, slot_count, w_ops, extra_args
    )

    build_path_he = os.path.join(BUILD_FOLDER, benchmark, "he")

    print()
    print(f"  Saving circuit from  {sol_dir}")
    print(f"               into   {build_path_he}")

    if not _restore_solution(sol_dir, build_path_he):
        print(_err("  Cached solution files missing — cannot save."))
        return

    print(_ok(f"  ✓ _gen_he_fhe.cpp/.hpp written to {build_path_he}"))


# ── cache cleanup ──────────────────────────────────────────────────────────────

def cleanup_solutions_cache() -> None:
    """Remove the entire solutions/ cache directory."""
    if os.path.isdir(SOLUTIONS_FOLDER):
        shutil.rmtree(SOLUTIONS_FOLDER)
        print(_ok(f"  [cache] {SOLUTIONS_FOLDER}/ removed."))