#!/usr/bin/env python3

def generate_rotation_rules():
    """
    Generate rewrite rules for decomposing rotations.
    Valid rotation amounts: 1, 2, 4, 8, 16, 32
    Special case: 9 = 3 + 3 + 3, and 3 = 1 + 2
    """
    rules = []
    
    # Standard power-of-2 decompositions
    # Each rotation can be split into two rotations of half the amount
    rotations = [32, 16, 8, 4, 2]
    
    for rot in rotations:
        half = rot // 2
        rule = f'Rewrite {{ name: "rotate_{rot}", searcher: (<< ?a {rot}), applier: (<< (<< ?a {half}) {half}) }}'
        rules.append(rule)
    
    # Special case: rotation of 9
    # 9 = 3 + 3 + 3
    rule_9 = 'Rewrite { name: "rotate_9", searcher: (<< ?a 9), applier: (<< (<< (<< ?a 3) 3) 3) }'
    rules.append(rule_9)
    
    # Special case: rotation of 3
    # 3 = 1 + 2
    rule_3 = 'Rewrite { name: "rotate_3", searcher: (<< ?a 3), applier: (<< (<< ?a 1) 2) }'
    rules.append(rule_3)
    
    return rules

def main():
    rules = generate_rotation_rules()
    
    # Write rules to file
    with open("../rotations_rules.txt", "w") as f:
        for rule in rules:
            f.write(rule + "\n")
    
    print("Rules saved to rotations_rules.txt")

if __name__ == "__main__":
    main()