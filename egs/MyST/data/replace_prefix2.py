#!/usr/bin/env python3

# Define the input and output file paths
input_file = './test_myst/wav.scp'
output_file = './test_myst/wav_new.scp'

# Define the old and new prefixes
old_prefix = "/home/klp65/rds/hpc-work/"
new_prefix = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/"

# Process the file - preserve exact spacing and formatting
with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        # Preserve completely empty lines
        if not line.strip():
            fout.write(line)
            continue

        # Replace prefix but maintain exact format otherwise
        new_line = line.replace(old_prefix, new_prefix)
        fout.write(new_line)

# Print a few examples to verify the changes look good
print("\nSample of changes (first 3 lines):")
with open(input_file, 'r') as fin, open(output_file, 'r') as fout:
    for i, (old_line, new_line) in enumerate(zip(fin, fout)):
        if i >= 3:  # Show only first 3 lines
            break
        print(f"OLD: {old_line.strip()}")
        print(f"NEW: {new_line.strip()}")
        print()

print(f"Processed {input_file} and saved to {output_file}")
print(f"To review changes: diff {input_file} {output_file}")
print(f"To replace the original file: mv {output_file} {input_file}")
