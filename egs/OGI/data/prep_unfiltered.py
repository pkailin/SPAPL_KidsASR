#!/usr/bin/env python3
import os
import re

def convert_path(old_path):
    """
    Convert a path from the old format to the new format.
    
    Old format: /home/dawna/sld/imports/cslu_kids/...
    New format: /home/klp65/rds/hpc-work/cslu_kids/...
    """
    # Replace the base directory
    new_path = old_path.replace('/home/dawna/sld/imports/cslu_kids', 
                                '/home/klp65/rds/hpc-work/cslu_kids')
    
    return new_path

def process_tsv_file(input_file, output_file):
    """
    Process the TSV file and create a new wav.scp file with the updated paths.
    """
    print(f"Reading from {input_file}")
    print(f"Writing to {output_file}")
    
    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the line into utterance ID and path
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue
                
            utterance_id, old_path = parts
            
            # Convert the path
            new_path = convert_path(old_path)
            
            # Write the new line to the output file
            fout.write(f"{utterance_id} {new_path}\n")
            
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} lines...")
    
    print(f"Conversion complete. Processed {count} lines.")

def main():
    # Define input and output files
    input_file = "/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/OGIKItrn01.tsv"  # Change this to your input file name
    output_file = "wav_unfiltered.scp"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please place your TSV file in the same directory as this script and update the input_file variable.")
        return
    
    # Process the file
    process_tsv_file(input_file, output_file)

if __name__ == "__main__":
    main()
