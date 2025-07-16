#!/usr/bin/env python3
import os
import time
from pathlib import Path

def find_wav_files_with_transcriptions(root_dir):
    """
    Find all .wav files that have corresponding .trn files.
    Returns a list of tuples (wav_path, utterance_id).
    """
    valid_files = []
    total_files_checked = 0
    total_wav_files = 0
    total_trn_files = 0
    total_matches = 0
    
    print(f"Starting to search in {root_dir}...")
    start_time = time.time()
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if total_files_checked % 1000 == 0 and total_files_checked > 0:
            elapsed = time.time() - start_time
            print(f"Checked {total_files_checked} files so far... (elapsed time: {elapsed:.2f}s)")
            print(f"Found {total_wav_files} .wav files, {total_trn_files} .trn files, and {total_matches} matches")
            print(f"Current directory: {dirpath}")
        
        # Update counts
        total_files_checked += len(filenames)
        
        # Find all .wav files in the current directory
        wav_files = [f for f in filenames if f.endswith('.wav')]
        total_wav_files += len(wav_files)
        
        # Count .trn files for reporting
        trn_files = [f for f in filenames if f.endswith('.trn')]
        total_trn_files += len(trn_files)
        
        for wav_file in wav_files:
            # Check if a corresponding .trn file exists
            trn_file = wav_file.replace('.wav', '.trn')
            wav_path = os.path.join(dirpath, wav_file)
            trn_path = os.path.join(dirpath, trn_file)
            
            if os.path.exists(trn_path):
                # Extract utterance_id from filename
                utterance_id = wav_file.replace('.wav', '')
                valid_files.append((wav_path, utterance_id))
                total_matches += 1
                
                if total_matches % 100 == 0:
                    print(f"Found {total_matches} valid matches so far...")
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\nSearch completed in {elapsed:.2f} seconds")
    print(f"Total files checked: {total_files_checked}")
    print(f"Total .wav files found: {total_wav_files}")
    print(f"Total .trn files found: {total_trn_files}")
    print(f"Total matches (.wav with corresponding .trn): {total_matches}")
    
    return valid_files

def main():
    # Set your root directory here
    root_dir = "/home/klp65/rds/hpc-work/myst_child_conv_speech/data/test"
    
    # Output file
    output_file = "test_wav_unfiltered.scp"
    
    print(f"Starting script at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Searching for valid .wav files in {root_dir}...")
    
    valid_files = find_wav_files_with_transcriptions(root_dir)
    print(f"Found {len(valid_files)} valid .wav files with corresponding .trn files.")
    
    # Create output directory for wav.scp if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Process files and write to wav.scp
    print(f"Writing results to {output_file}...")
    write_start_time = time.time()
    
    with open(output_file, 'w') as f:
        for i, (wav_path, utterance_id) in enumerate(valid_files):
            if i % 100 == 0:
                print(f"Writing file {i+1}/{len(valid_files)} to wav.scp...")
            
            # Write to wav.scp
            f.write(f"{utterance_id} {wav_path}\n")
    
    write_elapsed = time.time() - write_start_time
    print(f"Writing completed in {write_elapsed:.2f} seconds")
    print(f"Done! Created {output_file} with {len(valid_files)} entries.")
    print(f"Script finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
