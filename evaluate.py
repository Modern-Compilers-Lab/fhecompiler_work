import re

def parse_and_save(input_file, output_file):
    rewards = []
    final_cost = None
    
    # 1. Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 2. Parse line by line
    for line in lines:
        line = line.strip()
        
        # Capture Rewards
        if line.startswith("Reward"):
            # Split by ':' and take the second part
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    reward_value = float(parts[1].strip())
                    rewards.append(reward_value)
                except ValueError:
                    continue

        # Capture New Cost
        # We update this every time we see it. 
        # By the end of the loop, it will hold the cost of the final expression.
        if line.startswith("New cost"):
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    final_cost = float(parts[1].strip())
                except ValueError:
                    continue

    # 3. Calculate the Return (Sum of rewards)
    total_return = sum(rewards)

    # 4. Save to file
    with open(output_file, 'w') as f:
        f.write("--- Trajectory Statistics ---\n")
        f.write(f"Final Expression Cost: {final_cost}\n")
        f.write(f"Total Return (Sum):    {total_return}\n")
        f.write(f"Reward History:        {rewards}\n")
        
        # Optional: Save in a format easy to parse later (like CSV style)
        f.write("\n--- Raw Data ---\n")
        f.write(f"cost={final_cost}\n")
        f.write(f"return={total_return}\n")
        f.write(f"rewards={','.join(map(str, rewards))}\n")

    print(f"Successfully saved stats to {output_file}")
    print(f"Return: {total_return}")
    print(f"Final Cost: {final_cost}")

# Run the parser
parse_and_save("output.txt", "trajectory_stats.txt")