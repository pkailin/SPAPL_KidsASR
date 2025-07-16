#!/usr/bin/env python3

# Define the input and output file paths
input_file = './train_myst/wav.scp'
output_file = './train_myst/wav_new.scp'

# Define the old and new prefixes
old_prefix = "/home/klp65/rds/hpc-work/"
new_prefix = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/"

# Process the file
with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        if not line.strip():  # Skip empty lines
            fout.write(line)
            continue
            
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utterance_id, path = parts
            # Replace the prefix
            if path.startswith(old_prefix):
                new_path = path.replace(old_prefix, new_prefix, 1)
                fout.write(f"{utterance_id} {new_path}\n")
            else:
                # If the path doesn't have the prefix, keep it as is
                fout.write(line)
        else:
            # If the line doesn't have the expected format, keep it as is
            fout.write(line)

print(f"Processed {input_file} and saved to {output_file}")
print(f"To review changes: diff {input_file} {output_file}")
print(f"To replace the original file: mv {output_file} {input_file}")
