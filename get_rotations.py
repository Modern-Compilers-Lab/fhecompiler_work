#!/usr/bin/env python3
import os
import subprocess
import csv
import re
from pathlib import Path

def get_benchmarks(benchmarks_dir):
    """Get all benchmark directories."""
    try:
        benchmarks_path = Path(benchmarks_dir)
        if not benchmarks_path.exists():
            print(f"Error: Directory {benchmarks_dir} does not exist")
            return []
        
        # Get all subdirectories in the benchmarks folder
        benchmarks = [d.name for d in benchmarks_path.iterdir() if d.is_dir()]
        return benchmarks
    except Exception as e:
        print(f"Error getting benchmarks: {e}")
        return []

def run_benchmark(benchmark_name, slot_count):
    """Run a single benchmark with given slot_count and extract rotation count."""
    try:
        benchmark_dir = f"build/benchmarks/{benchmark_name}"
        
        # Change to benchmark directory
        os.chdir(benchmark_dir)
        
        # Run the generate script
        generate_cmd = f"python3 generate_{benchmark_name}.py --slot_count {slot_count}"
        print(f"  Running: {generate_cmd}")
        result = subprocess.run(generate_cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"  Generate script failed for {benchmark_name} with slot_count {slot_count}")
            os.chdir("../../..")
            return None
        
        # Run the benchmark
        benchmark_cmd = f"./{benchmark_name} 1 {slot_count} 1 0 1 1 1"
        print(f"  Running: {benchmark_cmd}")
        result = subprocess.run(benchmark_cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        # Go back to original directory
        os.chdir("../../..")
        
        if result.returncode != 0:
            print(f"  Benchmark execution failed for {benchmark_name} with slot_count {slot_count}")
            return None
        
        # Extract rotation count from output
        output = result.stdout
        rotation_count = extract_rotation_count(output)
        
        if rotation_count is not None:
            print(f"  ✓ Success: {benchmark_name}, slot_count={slot_count}, rotations={rotation_count}")
            return rotation_count
        else:
            print(f"  Could not extract rotation count for {benchmark_name} with slot_count {slot_count}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout for {benchmark_name} with slot_count {slot_count}")
        try:
            os.chdir("../../..")
        except:
            pass
        return None
    except Exception as e:
        print(f"  Error running {benchmark_name} with slot_count {slot_count}: {e}")
        try:
            os.chdir("../../..")
        except:
            pass
        return None

def extract_rotation_count(output):
    """Extract the total rotation count from benchmark output."""
    try:
        # Look for the |rotate| section
        lines = output.split('\n')
        in_rotate_section = False
        
        for line in lines:
            # Check if we're entering the rotate section
            if '|rotate|' in line:
                in_rotate_section = True
                continue
            
            # If we're in the rotate section, look for the total line
            if in_rotate_section:
                # Look for "total: <number>"
                match = re.search(r'total:\s*(\d+)', line)
                if match:
                    return int(match.group(1))
                
                # If we hit another operation section, we've left rotate
                if line.strip().startswith('|') and '|rotate|' not in line:
                    in_rotate_section = False
        
        return None
    except Exception as e:
        print(f"  Error extracting rotation count: {e}")
        return None

def main():
    benchmarks_dir = "build/benchmarks"
    output_csv = "benchmark_results.csv"
    slot_counts = range(2, 33)  # 2 to 32 inclusive
    
    # Get all benchmarks
    benchmarks = get_benchmarks(benchmarks_dir)
    
    if not benchmarks:
        print("No benchmarks found!")
        return
    
    print(f"Found {len(benchmarks)} benchmarks: {', '.join(benchmarks)}\n")
    
    # Create CSV file and write header
    csv_exists = False
    total_successful = 0
    
    # Loop through each benchmark
    for benchmark in benchmarks:
        print(f"Processing benchmark: {benchmark}")
        
        # Loop through slot counts
        for slot_count in slot_counts:
            rotation_count = run_benchmark(benchmark, slot_count)
            
            if rotation_count is not None:
                # Write to CSV immediately
                mode = 'a' if csv_exists else 'w'
                with open(output_csv, mode, newline='') as csvfile:
                    fieldnames = ['benchmark', 'slot_count', 'number_of_rotations']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header only if this is the first write
                    if not csv_exists:
                        writer.writeheader()
                        csv_exists = True
                    
                    writer.writerow({
                        'benchmark': benchmark,
                        'slot_count': slot_count,
                        'number_of_rotations': rotation_count
                    })
                
                total_successful += 1
                print(f"  → Saved to CSV (total entries: {total_successful})")
    
    if total_successful > 0:
        print(f"\n✓ Results saved to {output_csv}")
        print(f"Total successful runs: {total_successful}")
    else:
        print("\nNo successful runs to save!")

if __name__ == "__main__":
    main()