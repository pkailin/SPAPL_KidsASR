#!/usr/bin/env python3
import os
import re

# Hardcoded path to wav.scp file
WAV_SCP_PATH = "./test_myst/wav.scp"  # Change this to your actual path
OUTPUT_FILE = "text.txt"

def clean_transcription(text):
    """
    Clean transcription by:
    - Converting to lowercase
    - Removing text within <>, **, (), ++ and the symbols themselves
    """
    # Convert to lowercase
    text = text.lower()

    # Remove text within brackets/symbols and the symbols themselves
    text = re.sub(r'<[^>]*>', '', text)  # Remove <...>
    text = re.sub(r'\*[^*]*\*', '', text)  # Remove *...*
    text = re.sub(r'\([^)]*\)', '', text)  # Remove (...)
    text = re.sub(r'\+[^+]*\+', '', text)  # Remove +...+

    # Clean up extra whitespace
    text = ' '.join(text.split())

    return text

def create_transcription_file():
    """
    Read wav.scp file and create transcription file by reading corresponding .trn files
    """
    transcriptions = []
    
    try:
        with open(WAV_SCP_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split line into utterance_id and wav_path
                parts = line.split(None, 1)  # Split on whitespace, max 1 split
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                
                utterance_id, wav_path = parts
                
                # Replace .wav extension with .trn
                trn_path = wav_path.rsplit('.wav', 1)[0] + '.trn'
                
                # Read transcription from .trn file
                try:
                    with open(trn_path, 'r') as trn_file:
                        transcription = clean_transcription(trn_file.read().strip())
                        transcriptions.append(f"{utterance_id} {transcription}")
                        print(f"Processed: {utterance_id}")
                except FileNotFoundError:
                    print(f"Warning: Transcription file not found: {trn_path}")
                except Exception as e:
                    print(f"Error reading {trn_path}: {e}")
    
    except FileNotFoundError:
        print(f"Error: wav.scp file not found at {WAV_SCP_PATH}")
        return
    except Exception as e:
        print(f"Error reading wav.scp file: {e}")
        return
    
    # Write transcriptions to output file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            for transcription in transcriptions:
                f.write(transcription + '\n')
        print(f"\nSuccessfully created {OUTPUT_FILE} with {len(transcriptions)} transcriptions")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    create_transcription_file()
