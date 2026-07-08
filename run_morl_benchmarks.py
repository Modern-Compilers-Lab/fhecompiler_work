import os
import subprocess 
import csv 
import re
import statistics
from RL.fhe_rl.pareto import generate_pref_list

benchmarks_folder = "benchmarks"  
build_folder = os.path.join("build", "benchmarks")
operations = ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply"]
infos = ["benchmark", "w_ops", "w_keys"]
additional_infos = ["Depth", "Multiplicative Depth", "compile_time (s)", "circuit_execution_time (s)",
                    'galois_keys_generation_time (s)', 'total_execution_time (s)', "Remaining_noise_budget",
                    'rotation_keys_size (MB)', 'final_ops_cost', 'final_keys_cost']
infos.extend(operations)
infos.extend(additional_infos)

try:
    print("run=> cmake', '-S', '.', '-B', 'build' ")
    result = subprocess.run(
        ['cmake', '-S', '.', '-B', 'build'], 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )
    print("run=> 'cmake', '--build', 'build'")
    result = subprocess.run(
        ['cmake', '--build', 'build'], 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )  
except subprocess.CalledProcessError as e:
    print(f"Command failed with error:\n{e.stderr}")   

benchmark_folders = ["lin_reg"]
exceptions = ["max", "sort", "discrete_cosin_transform", "poly_derivative"]
benchmarks_slot_counts = {
    "max": [3, 4, 5],
    "sort": [3],
    "discrete_cosin_transform": [1],
    "poly_derivative": [1]
}

optimization_method = 1
cse_enabled = 1
vectorize_code = 1
slot_counts = [4, 8, 16, 32]
pref_list = [[0.0, 1.0]]
iterations = 1
window_size = 0
depths = [5]
regimes = ["100-50"]
number_instances_each_polynomial_configuration = 1
compile_time_timeout_seconds = 7200
output_csv = f"results_{'RL' if optimization_method == 1 else 'EGraph'}.csv"

# ── adaptive bisection config ──────────────────────────────────────────────────
BASE_UNIT = 18          # known spacing between key-size levels
MAX_BISECT_DEPTH = 3   # safety cap on recursion

written_w_ops = set()

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(infos)


# ── helpers ────────────────────────────────────────────────────────────────────
def get_key_size_level(key_size):
    return int(key_size)


def run_benchmark(subfolder_name, slot_count, w_ops, w_keys, build_path, build_path_he, build_path_he_build):
    """
    Run one (subfolder_name, slot_count, w_ops) point exactly like the original loop body.
    Returns (row, key_size_level) or (None, None) on timeout.
    """
    benchmark_compilation_timed_out = False
    operation_stats = {
        "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
        "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
        "compile_time (s)": [], "circuit_execution_time (s)": [], "galois_keys_generation_time (s)": [],
        "total_execution_time (s)": [], "Remaining_noise_budget": [], "rotation_keys_size (MB)": [],
        "final_ops_cost": [], "final_keys_cost": []
    }

    if not subfolder_name in exceptions:
        pro = subprocess.Popen(['python3', 'generate_{}.py'.format(subfolder_name), '--slot_count', str(slot_count)], cwd=build_path)
        pro.wait()

    for iteration in range(iterations):
        print(f"===> Running iteration : {iteration + 1}")
        benchmark_run_command = f"./{subfolder_name} {vectorize_code} {slot_count} {optimization_method} {window_size} 1 {cse_enabled} 1 {w_ops} {w_keys}"
        try:
            result = subprocess.run(
                benchmark_run_command, shell=True, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, cwd=build_path,
                timeout=compile_time_timeout_seconds
            )
            lines = result.stdout.splitlines()
            compile_time_found = False
            poly_mod_found = True
            for line in lines:
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                if 'final ops cost:' in clean_line:
                    try:
                        operation_stats["final_ops_cost"].append(float(clean_line.split(':')[1].strip()))
                    except (IndexError, ValueError):
                        pass
                if 'final keys cost:' in clean_line:
                    try:
                        operation_stats["final_keys_cost"].append(float(clean_line.split(':')[1].strip()))
                    except (IndexError, ValueError):
                        pass
                if ' ms' in line:
                    optimization_time = float(line.split()[0])
                    operation_stats["compile_time (s)"].append(optimization_time)
                    compile_time_found = True
                if 'poly_mod:' in line:
                    print(f"======> poly_mod : {line}")
                    poly_mod = float(line.split()[1])
                    poly_mod_found = True
                if compile_time_found and poly_mod_found:
                    break
            depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
            depth = int(depth_match.group(1)) if depth_match else None
            multiplicative_depth = int(depth_match.group(2)) if depth_match else None
            print(f"Depth=>{depth}, multiplicative_depth=>{multiplicative_depth}")
            operation_stats["Depth"].append(depth)
            operation_stats["Multiplicative Depth"].append(multiplicative_depth)
        except subprocess.TimeoutExpired:
            print(f"Command `{benchmark_run_command}` timed out after {compile_time_timeout_seconds} seconds.")
            benchmark_compilation_timed_out = True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr if e.stderr else "No error message available."
            print("Command for {} failed with error:\n{}".format(subfolder_name, error_message))
            continue

        if benchmark_compilation_timed_out:
            break

        he_build_ok = (result.returncode == 0)
        if he_build_ok:
            result = subprocess.run(['cmake', '-S', '.', '-B', 'build'],
                                    cwd=build_path_he, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            result = subprocess.run(['cmake', '--build', 'build'],
                                    cwd=build_path_he, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)

            if iteration == iterations - 1:
                try:
                    for counter in range(iterations):
                        command = f"./main"
                        result = subprocess.run(
                            command, shell=True, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, cwd=build_path_he_build
                        )
                        print("**fhe run done**")
                        if counter > 0:
                            lines = result.stdout.splitlines()
                            comp = 0
                            print(f"returned lines : \n {lines} \n\n")
                            for line in lines:
                                if 'circuit_execution_time_(ms):' in line:
                                    operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                                if 'galois_keys_generation_time_(ms):' in line:
                                    operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                                if 'total_execution_time_(ms):' in line:
                                    operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                                if 'rotation_keys_size_(MB):' in line:
                                    operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                                if 'Remaining_noise_budget:' in line:
                                    operation_stats["Remaining_noise_budget"].append(int(line.split()[1]))
                                if comp == 2:
                                    break
                except subprocess.CalledProcessError as e:
                    print(f"Failed in building fhe_code for benchmark: {subfolder_name}")

        file_name = os.path.join(build_path_he, "_gen_he_fhe.cpp")
        with open(file_name, "r") as file:
            file_content = file.read()
            for op in operations:
                nb_occurrences = len(re.findall(rf'\b{op}', file_content))
                operation_stats[op].append(int(nb_occurrences))

    # ── build row ──────────────────────────────────────────────────────────────
    bench_name = subfolder_name + "_" + str(slot_count)
    row = [bench_name, w_ops, w_keys]
    raw_key_size = None

    if not benchmark_compilation_timed_out:
        for key, values in operation_stats.items():
            if values == []:
                print(f"Warning: No values found for {key} in {subfolder_name} with slot_count {slot_count}.")
                result = "N/A"
            else:
                result = statistics.median(values)
                if key in ["compile_time (s)", "circuit_execution_time (s)",
                           "galois_keys_generation_time (s)", "total_execution_time (s)"]:
                    result = result / 1000
                    result = format(result, ".3f")
                if key == "final_keys_cost":
                    raw_key_size = statistics.median(values)   # keep raw float for level comparison
                row.append(result)
            print(f"{key} {values} {result}")

    # ── write immediately (open → write → close, same as original) ────────────
    if w_ops not in written_w_ops:
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        written_w_ops.add(w_ops)
    else:
        print(f"  [skip] w_ops={w_ops} already written, skipping duplicate row")

    if benchmark_compilation_timed_out:
        return None, None

    key_size_level = get_key_size_level(raw_key_size) if raw_key_size is not None else None
    return row, key_size_level


def _bisect_one_target(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                       build_path, build_path_he, build_path_he_build,
                       target, found, depth=0):
    """
    Search for a single integer key-size level `target` in (w_lo, w_hi).
    `found` is a shared set — if the target (or any level) gets discovered
    by a mid-point probe it is added there so sibling searches can skip it.
    """
    if depth >= MAX_BISECT_DEPTH:
        print(f"    [max depth] target={target} stopping at [{w_lo:.6f}, {w_hi:.6f}]")
        return

    print(f"  {'  ' * depth}[bisect d={depth}] [{w_lo:.6f}, {w_hi:.6f}] "
          f"endpoints=[{ks_lo}, {ks_hi}]  target={target}")

    w_mid = (w_hi + w_lo) / 2
    w_mid = round(w_mid, 10)  # avoid floating point precision issues
    w_mid_keys = round(1.0 - w_mid, 10)

    row, ks_mid = run_benchmark(subfolder_name, slot_count, w_mid, w_mid_keys,
                                build_path, build_path_he, build_path_he_build)
    if row is None:
        return

    print(f"  {'  ' * depth}[result  d={depth}] w_ops={w_mid:.6f} → keys={ks_mid}")
    found.add(ks_mid)

    if ks_mid == target:
        return  # found — done for this target

    elif ks_mid < target:
        # target is in the upper half [w_mid, w_hi]
        _bisect_one_target(subfolder_name, slot_count, w_hi, ks_hi, w_mid, ks_mid,
                           build_path, build_path_he, build_path_he_build,
                           target, found, depth=depth + 1)

    else:  # ks_mid > target
        # target is in the lower half [w_lo, w_mid]
        _bisect_one_target(subfolder_name, slot_count, w_mid, ks_mid, w_lo, ks_lo,
                           build_path, build_path_he, build_path_he_build,
                           target, found, depth=depth + 1)


def bisect_search(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                  build_path, build_path_he, build_path_he_build):
    """
    Entry point: search independently for every integer level between ks_lo
    and ks_hi. Each target gets its own MAX_BISECT_DEPTH budget.
    Targets already discovered by a previous probe are skipped.
    """
    lo, hi = min(ks_lo, ks_hi), max(ks_lo, ks_hi)
    missing_targets = list(range(lo + 1, hi))   # e.g. ks=1,ks=4 → [2, 3]

    if not missing_targets:
        return

    found = set()   # levels discovered by any probe, shared across targets

    for target in missing_targets:
        if target in found:
            print(f"  [skip] target={target} already found by a previous probe")
            continue
        print(f"\n  === independent search for target={target} "
              f"in [{w_lo:.6f}, {w_hi:.6f}] ===")
        _bisect_one_target(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                           build_path, build_path_he, build_path_he_build,
                           target, found)
        
# ── main loop (your original structure) ───────────────────────────────────────
for subfolder_name in benchmark_folders:
    benchmark_path = os.path.join(benchmarks_folder, subfolder_name)
    build_path = os.path.join(build_folder, subfolder_name)
    if os.path.isdir(build_path):
        updated_slot_counts = slot_counts
        if subfolder_name in exceptions:
            updated_slot_counts = benchmarks_slot_counts[subfolder_name]

        for slot_count in updated_slot_counts:
            build_path_he = os.path.join(build_path, "he")
            build_path_he_build = os.path.join(build_path_he, "build")

            # ── Phase 1: coarse grid (your original pref_list loop) ───────────
            written_w_ops = set()
            known_points = []   # list of (w_ops, ks_level)
            for w in pref_list:
                w_ops, w_keys = w
                print("****************************************************************")
                print(f"*****run {subfolder_name} , for slot_count : {slot_count}******")
                try:
                    row, ks_level = run_benchmark(subfolder_name, slot_count, w_ops, w_keys,
                                                  build_path, build_path_he, build_path_he_build)
                    if ks_level is not None:
                        known_points.append((w_ops, ks_level))
                except Exception as e:
                    print(f"Command for {subfolder_name} failed with error:\n{e}")
                    continue

            # ── Phase 2: bisect every interval where levels differ ─────────────
            print(f"\n--- Adaptive bisection for {subfolder_name} slot_count={slot_count} ---")
            known_points.sort(key=lambda x: x[0])  # ascending w_ops
            for i in range(len(known_points) - 1):
                w_lo, ks_lo = known_points[i]
                w_hi, ks_hi = known_points[i + 1]
                if ks_lo != ks_hi:
                    try:
                        bisect_search(subfolder_name, slot_count, w_hi, ks_hi, w_lo, ks_lo,
                                      build_path, build_path_he, build_path_he_build)
                    except Exception as e:
                        print(f"Bisect failed for {subfolder_name} [{w_lo}, {w_hi}]: {e}")
                        continue
#################################################################################################
# ── poly-tree helpers ──────────────────────────────────────────────────────────

def run_poly_benchmark(subfolder_name, build_path, build_path_he, build_path_he_build,
                       benchmark_name, tree_depth, instance, regime,
                       w_ops, w_keys):
    """
    Run one (benchmark_name, w_ops, w_keys) point for polynomial trees.
    Returns (row, ks_ops_level, ks_keys_level) or (None, None, None) on timeout.
    """
    benchmark_compilation_timed_out = False
    operation_stats = {
        "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
        "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
        "compile_time (s)": [], "circuit_execution_time (s)": [],
        "galois_keys_generation_time (s)": [], "total_execution_time (s)": [],
        "Remaining_noise_budget": [], "rotation_keys_size (MB)": [],
        "final_ops_cost": [], "final_keys_cost": []
    }

    for iteration in range(iterations):
        optimization_time = ""
        execution_time = ""
        depth = ""
        multiplicative_depth = ""

        if not os.path.isdir(build_path):
            continue

        print(f"=========> Iteration : {iteration + 1}")
        command = (f"./{subfolder_name} {tree_depth} {instance} {regime} "
                   f"{vectorize_code} {optimization_method} {window_size} "
                   f"1 {cse_enabled} 1 {w_ops} {w_keys}")
        try:
            result = subprocess.run(
                command, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, cwd=build_path,
                timeout=compile_time_timeout_seconds
            )
            lines = result.stdout.splitlines()
            compile_time_found = False
            poly_mod_found = True
            for line in lines:
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                if 'final ops cost:' in clean_line:
                    try:
                        operation_stats["final_ops_cost"].append(float(clean_line.split(':')[1].strip()))
                    except (IndexError, ValueError):
                        pass
                if 'final keys cost:' in clean_line:
                    try:
                        operation_stats["final_keys_cost"].append(float(clean_line.split(':')[1].strip()))
                    except (IndexError, ValueError):
                        pass
                if ' ms' in line:
                    optimization_time = float(line.split()[0])
                    operation_stats["compile_time (s)"].append(optimization_time)
                    compile_time_found = True
                if 'poly_mod:' in line:
                    print(f"======> poly_mod : {line}")
                    poly_mod_found = True
                if compile_time_found and poly_mod_found:
                    break

            depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
            depth = int(depth_match.group(1)) if depth_match else None
            multiplicative_depth = int(depth_match.group(2)) if depth_match else None
            print(f"Depth: {depth} -- MultiplicativeDepth: {multiplicative_depth}")
            operation_stats["Depth"].append(depth)
            operation_stats["Multiplicative Depth"].append(multiplicative_depth)

        except subprocess.TimeoutExpired:
            print(f"Command `{command}` timed out after {compile_time_timeout_seconds} seconds.")
            benchmark_compilation_timed_out = True
        except subprocess.CalledProcessError as e:
            print(f"Command for {subfolder_name} failed with error:\n{e.stderr}")

        if benchmark_compilation_timed_out:
            break

        # build HE code
        try:
            subprocess.run(['cmake', '-S', '.', '-B', 'build'],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, cwd=build_path_he)
            subprocess.run(['cmake', '--build', 'build'],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, cwd=build_path_he)
        except Exception:
            print(f"Failed in building fhe_code for benchmark: {subfolder_name}")

        if iteration == iterations - 1:
            try:
                for counter in range(iterations):
                    result = subprocess.run(
                        "./main", shell=True, check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        universal_newlines=True, cwd=build_path_he_build
                    )
                    print("**fhe run done**")
                    if counter > 0:
                        lines = result.stdout.splitlines()
                        comp = 0
                        print(f"returned lines : \n {lines} \n\n")
                        for line in lines:
                            if 'circuit_execution_time_(ms):' in line:
                                operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                            if 'galois_keys_generation_time_(ms):' in line:
                                operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                            if 'total_execution_time_(ms):' in line:
                                operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                            if 'rotation_keys_size_(MB):' in line:
                                operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                            if 'Remaining_noise_budget:' in line:
                                operation_stats["Remaining_noise_budget"].append(int(line.split()[1]))
                            if comp == 2:
                                break
            except subprocess.CalledProcessError:
                print(f"Failed in running fhe_code for benchmark: {subfolder_name}")

        # parse operation counts from generated C++ file
        file_name = os.path.join(build_path_he, "_gen_he_fhe.cpp")
        with open(file_name, "r") as f:
            file_content = f.read()
            for op in operations:
                nb_occurrences = len(re.findall(rf'\b{op}', file_content))
                operation_stats[op].append(int(nb_occurrences))

    # ── build row ──────────────────────────────────────────────────────────────
    row = [benchmark_name, w_ops, w_keys]
    raw_ops_cost = None
    raw_keys_cost = None

    if not benchmark_compilation_timed_out:
        for key, values in operation_stats.items():
            if not values:
                print(f"Warning: No values found for {key} in {benchmark_name}.")
                result_val = "N/A"
            else:
                result_val = statistics.median(values)
                if key in ["compile_time (s)", "circuit_execution_time (s)",
                           "galois_keys_generation_time (s)", "total_execution_time (s)"]:
                    result_val = result_val / 1000
                    result_val = format(result_val, ".3f")
                if key == "final_ops_cost":
                    raw_ops_cost = statistics.median(values)
                if key == "final_keys_cost":
                    raw_keys_cost = statistics.median(values)
            row.append(result_val)
            print(f"{key} {values} {result_val}")

    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    if benchmark_compilation_timed_out:
        return None, None, None

    ops_level  = int(raw_ops_cost)  if raw_ops_cost  is not None else None
    keys_level = int(raw_keys_cost) if raw_keys_cost is not None else None
    return row, ops_level, keys_level


def _poly_bisect_one_target(subfolder_name, build_path, build_path_he, build_path_he_build,
                             benchmark_name, tree_depth, instance, regime,
                             w_hi, ks_ops_hi, ks_keys_hi,
                             w_lo, ks_ops_lo, ks_keys_lo,
                             target_ops, target_keys, found, depth=0):
    """
    Search for a single (target_ops, target_keys) level pair.
    Bisects on w_ops; w_keys = 1 - w_ops.
    """
    if depth >= MAX_BISECT_DEPTH:
        print(f"    [max depth] targets=({target_ops},{target_keys}) "
              f"stopping at [{w_lo:.6f}, {w_hi:.6f}]")
        return

    print(f"  {'  ' * depth}[poly bisect d={depth}] [{w_lo:.6f}, {w_hi:.6f}] "
          f"endpoints=ops[{ks_ops_lo},{ks_ops_hi}] keys[{ks_keys_lo},{ks_keys_hi}] "
          f"target=({target_ops},{target_keys})")

    w_mid = (w_hi + w_lo) / 2
    w_mid = round(w_mid, 10)  # avoid floating point precision issues
    w_mid_keys = round(1.0 - w_mid, 10)

    row, ops_mid, keys_mid = run_poly_benchmark(
        subfolder_name, build_path, build_path_he, build_path_he_build,
        benchmark_name, tree_depth, instance, regime, w_mid, w_mid_keys
    )
    if row is None:
        return

    print(f"  {'  ' * depth}[result d={depth}] w_ops={w_mid:.6f} → "
          f"ops={ops_mid} keys={keys_mid}")
    found.add((ops_mid, keys_mid))

    if ops_mid == target_ops and keys_mid == target_keys:
        return  # found — done for this target

    # use ops cost as the primary bisection axis (same convention as first script)
    if ops_mid < target_ops:
        _poly_bisect_one_target(
            subfolder_name, build_path, build_path_he, build_path_he_build,
            benchmark_name, tree_depth, instance, regime,
            w_hi, ks_ops_hi, ks_keys_hi, w_mid, ops_mid, keys_mid,
            target_ops, target_keys, found, depth=depth + 1
        )
    else:
        _poly_bisect_one_target(
            subfolder_name, build_path, build_path_he, build_path_he_build,
            benchmark_name, tree_depth, instance, regime,
            w_mid, ops_mid, keys_mid, w_lo, ks_ops_lo, ks_keys_lo,
            target_ops, target_keys, found, depth=depth + 1
        )


def poly_bisect_search(subfolder_name, build_path, build_path_he, build_path_he_build,
                       benchmark_name, tree_depth, instance, regime,
                       w_hi, ks_ops_hi, ks_keys_hi,
                       w_lo, ks_ops_lo, ks_keys_lo):
    """
    Entry point for poly-tree bisection between two adjacent coarse-grid points.
    Searches independently for every missing (ops, keys) level pair.
    Each target gets its own MAX_BISECT_DEPTH budget.
    """
    ops_lo,  ops_hi  = min(ks_ops_lo,  ks_ops_hi),  max(ks_ops_lo,  ks_ops_hi)
    keys_lo, keys_hi = min(ks_keys_lo, ks_keys_hi), max(ks_keys_lo, ks_keys_hi)

    # build all missing (ops, keys) integer pairs between the two endpoints
    missing_targets = [
        (o, k)
        for o in range(ops_lo + 1, ops_hi)
        for k in range(keys_lo + 1, keys_hi)
    ]

    if not missing_targets:
        return

    found = set()

    for target_ops, target_keys in missing_targets:
        if (target_ops, target_keys) in found:
            print(f"  [skip] target=({target_ops},{target_keys}) already found")
            continue
        print(f"\n  === poly independent search for target=({target_ops},{target_keys}) "
              f"in [{w_lo:.6f}, {w_hi:.6f}] ===")
        _poly_bisect_one_target(
            subfolder_name, build_path, build_path_he, build_path_he_build,
            benchmark_name, tree_depth, instance, regime,
            w_hi, ks_ops_hi, ks_keys_hi, w_lo, ks_ops_lo, ks_keys_lo,
            target_ops, target_keys, found
        )


# ── poly-tree main loop ────────────────────────────────────────────────────────
print("Run polynomial benchmarks !!!!!!")
polynomial_folders = ["polynomials_coyote"]

for subfolder_name in polynomial_folders:
    build_path = os.path.join(build_folder, subfolder_name)
    build_path_he = os.path.join(build_path, "he")
    build_path_he_build = os.path.join(build_path_he, "build")

    for regime in regimes:
        for tree_depth in depths:
            for instance in range(1, number_instances_each_polynomial_configuration + 1):
                benchmark_name = f'tree_{regime}_{tree_depth}_{instance}'
                print(f"\nBenchmark '{benchmark_name}' will be run...")

                # ── Phase 1: coarse grid ───────────────────────────────────────
                known_points = []   # list of (w_ops, ops_level, keys_level)
                written_w_ops_poly = set()

                for w in pref_list:
                    w_ops, w_keys = w
                    if w_ops in written_w_ops_poly:
                        continue
                    print("*" * 64)
                    try:
                        row, ops_level, keys_level = run_poly_benchmark(
                            subfolder_name, build_path, build_path_he, build_path_he_build,
                            benchmark_name, tree_depth, instance, regime, w_ops, w_keys
                        )
                        written_w_ops_poly.add(w_ops)
                        if ops_level is not None and keys_level is not None:
                            known_points.append((w_ops, ops_level, keys_level))
                    except Exception as e:
                        print(f"Command for {benchmark_name} failed with error:\n{e}")
                        continue

                # ── Phase 2: adaptive bisection ────────────────────────────────
                print(f"\n--- Adaptive bisection for {benchmark_name} ---")
                known_points.sort(key=lambda x: x[0])   # ascending w_ops

                for i in range(len(known_points) - 1):
                    w_lo, ops_lo, keys_lo = known_points[i]
                    w_hi, ops_hi, keys_hi = known_points[i + 1]
                    if ops_lo != ops_hi or keys_lo != keys_hi:
                        try:
                            poly_bisect_search(
                                subfolder_name, build_path, build_path_he, build_path_he_build,
                                benchmark_name, tree_depth, instance, regime,
                                w_hi, ops_hi, keys_hi, w_lo, ops_lo, keys_lo
                            )
                        except Exception as e:
                            print(f"Poly bisect failed for {benchmark_name} "
                                  f"[{w_lo}, {w_hi}]: {e}")
                            continue                    