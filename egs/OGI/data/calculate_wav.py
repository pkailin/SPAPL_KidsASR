#!/usr/bin/env python3

import os
import wave
from concurrent.futures import ThreadPoolExecutor

# Hardcoded path to wav.scp file
WAV_SCP_PATH = "spont_wav.scp"  # Change this to your actual path

def get_wav_duration(wav_path):
    """Get the duration of a WAV file in seconds."""
    try:
        with wave.open(wav_path, 'rb') as wf:
            # Calculate duration in seconds
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return 0

def process_scp_file(scp_file):
    """Process wav.scp file and return total duration in seconds."""
    wav_paths = []
    
    # Read the scp file
    with open(scp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Each line has format: <utterance-id> <path to wav file>
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                _, wav_path = parts
                wav_paths.append(wav_path)
    
    total_duration = 0
    
    # Process files in parallel for better performance with large datasets
    with ThreadPoolExecutor() as executor:
        durations = list(executor.map(get_wav_duration, wav_paths))
        total_duration = sum(durations)
    
    return total_duration, len(wav_paths)

def main():
    if not os.path.exists(WAV_SCP_PATH):
        print(f"Error: File {WAV_SCP_PATH} not found")
        return
        
    print(f"Processing {WAV_SCP_PATH}...")
    total_seconds, file_count = process_scp_file(WAV_SCP_PATH)
    
    # Convert to hours, minutes, seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    print(f"Processed {file_count} WAV files")
    print(f"Total duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
    print(f"Total hours: {total_seconds/3600:.2f}")

if __name__ == "__main__":
    main()
