#!/usr/bin/env python3
"""
Script to correct transcriptions by searching through directory structure.
Finds transcription files matching utterance IDs and creates corrected text file.
Removes unwanted markers: <>, (), **, ++
"""

import os
import sys
import re
from pathlib import Path

def clean_transcription(text):
    """
    Clean transcription text by removing unwanted markers and their contents.
    
    Args:
        text (str): Raw transcription text
    
    Returns:
        str: Cleaned transcription text
    """
    # Remove content within <> and the brackets themselves
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove content within () and the parentheses themselves
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove content within ** and the asterisks themselves
    text = re.sub(r'\*[^*]*\*', '', text)
    
    # Remove content within ++ and the plus signs themselves
    text = re.sub(r'\+[^+]*\+', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def find_transcription_file(utterance_id, base_dir):
    """
    Search for transcription file matching the utterance ID in the directory structure.
    
    Args:
        utterance_id (str): The utterance ID to search for
        base_dir (str): Base directory to search in
    
    Returns:
        str or None: Path to the transcription file if found, None otherwise
    """
    target_filename = f"{utterance_id}.txt"
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    
    return None

def read_transcription_file(file_path):
    """
    Read and clean transcription from a file.
    
    Args:
        file_path (str): Path to the transcription file
    
    Returns:
        str: Cleaned transcription text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Clean the transcription
        cleaned_content = clean_transcription(content)
        return cleaned_content
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def build_transcription_cache(base_dir):
    """
    Build a cache of all transcription files for faster lookup.
    
    Args:
        base_dir (str): Base directory to search in
    
    Returns:
        dict: Dictionary mapping utterance IDs to file paths
    """
    print("Building transcription file cache...")
    cache = {}
    file_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                utterance_id = file[:-4]  # Remove .txt extension
                file_path = os.path.join(root, file)
                cache[utterance_id] = file_path
                file_count += 1
                
                if file_count % 1000 == 0:
                    print(f"  Cached {file_count} files...")
    
    print(f"Cache built: {len(cache)} transcription files found")
    return cache

def correct_transcriptions_from_directory(input_text_path, transcription_dir, output_path):
    """
    Read input text file and create corrected version using transcriptions from directory.
    
    Args:
        input_text_path (str): Path to input text file with incorrect transcriptions
        transcription_dir (str): Directory containing transcription files
        output_path (str): Path for output corrected text file
    """
    # Build cache of transcription files
    transcription_cache = build_transcription_cache(transcription_dir)
    
    corrected_lines = []
    corrections_made = 0
    lines_processed = 0
    not_found_count = 0
    
    try:
        with open(input_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                lines_processed += 1
                
                # Split line into utterance ID and transcription
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                
                utterance_id = parts[0]
                old_transcription = parts[1]
                
                # Look for transcription file
                if utterance_id in transcription_cache:
                    file_path = transcription_cache[utterance_id]
                    new_transcription = read_transcription_file(file_path)
                    
                    if new_transcription is not None:
                        corrected_lines.append(f"{utterance_id} {new_transcription}")
                        corrections_made += 1
                        
                        if old_transcription != new_transcription:
                            print(f"Corrected {utterance_id}:")
                            print(f"  File: {file_path}")
                            print(f"  Old: {old_transcription}")
                            print(f"  New: {new_transcription}")
                            print()
                    else:
                        # Keep original if file couldn't be read
                        corrected_lines.append(line)
                        print(f"Warning: Could not read file for {utterance_id}")
                else:
                    # Keep original if no file found
                    corrected_lines.append(line)
                    not_found_count += 1
                    print(f"Warning: No transcription file found for {utterance_id}")
                
                # Progress indicator
                if lines_processed % 100 == 0:
                    print(f"Processed {lines_processed} lines...")
        
        # Write corrected transcriptions to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in corrected_lines:
                f.write(line + '\n')
        
        print(f"\nProcessing complete:")
        print(f"  Lines processed: {lines_processed}")
        print(f"  Corrections made: {corrections_made}")
        print(f"  Files not found: {not_found_count}")
        print(f"  Output written to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input text file not found at {input_text_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)

def test_cleaning():
    """Test the text cleaning function with examples."""
    test_cases = [
        "hello <noise> world (laughter) test",
        "this *cough* is **unclear** text",
        "remove ++background++ and <music> please",
        "normal text without markers",
        "<start> hello (world) *test* ++end++",
    ]
    
    print("Testing text cleaning:")
    for test in test_cases:
        cleaned = clean_transcription(test)
        print(f"  Original: {test}")
        print(f"  Cleaned:  {cleaned}")
        print()

def main():
    """Main function to run the transcription correction."""
    
    # Directory containing transcription files
    transcription_dir = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/cslu_kids/trans/spontaneous/"
    
    # Get input and output file paths from user
    if len(sys.argv) < 2:
        print("Usage: python transcription_corrector.py <input_text_file> [output_text_file] [--test-cleaning]")
        print("Example: python transcription_corrector.py transcriptions.txt corrected_transcriptions.txt")
        print("Add --test-cleaning to test the text cleaning function")
        sys.exit(1)
    
    # Check if user wants to test cleaning
    if "--test-cleaning" in sys.argv:
        test_cleaning()
        return
    
    input_text_path = sys.argv[1]
    
    # Default output path if not provided
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        output_path = sys.argv[2]
    else:
        # Create output filename by adding '_corrected' before extension
        input_path = Path(input_text_path)
        output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
    
    print(f"Transcription directory: {transcription_dir}")
    print(f"Input text file: {input_text_path}")
    print(f"Output text file: {output_path}")
    print()
    
    # Check if transcription directory exists
    if not os.path.exists(transcription_dir):
        print(f"Error: Transcription directory not found: {transcription_dir}")
        sys.exit(1)
    
    # Correct transcriptions
    print("Starting transcription correction...")
    correct_transcriptions_from_directory(input_text_path, transcription_dir, output_path)

if __name__ == "__main__":
    main()
