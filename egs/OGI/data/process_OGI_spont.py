#!/usr/bin/env python3
import os
import glob
import wave
import re
from pathlib import Path
from whisper_normalizer.english import EnglishTextNormalizer

# Initialize the WhisperNormalizer
normalizer = EnglishTextNormalizer()

# Define base directories
TRANS_DIR = "/home/klp65/rds/hpc-work/cslu_kids/trans/spontaneous/"
SPEECH_DIR = "/home/klp65/rds/hpc-work/cslu_kids/speech/spontaneous/"

# Output files
WAV_SCP = "spont_wav.scp"
TEXT_FILE = "spont_text"

def get_duration(wav_file):
    """Get duration of a wav file in seconds"""
    try:
        with wave.open(wav_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return 0

def normalize_text(text):
    """Normalize transcription text"""
    # Convert to lowercase
    text = text.lower()
    
    # Apply WhisperNormalizer
    text = normalizer(text)
    
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def read_transcription(trans_file):
    """Read and normalize transcription from file"""
    try:
        with open(trans_file, 'r', encoding='utf-8') as f:
            text = f.read()
        return normalize_text(text)
    except Exception as e:
        print(f"Error reading transcription {trans_file}: {e}")
        return ""

def main():
    # Find all transcription files
    trans_files = glob.glob(os.path.join(TRANS_DIR, "**/*.txt"), recursive=True)
    
    wav_scp_entries = []
    text_entries = []
    total_duration = 0
    
    # Process each transcription file
    for trans_file in trans_files:
        # Extract utterance_id from the transcription file path
        rel_path = os.path.relpath(trans_file, TRANS_DIR)
        dir_path = os.path.dirname(rel_path)
        file_name = os.path.basename(trans_file)
        utterance_id = os.path.splitext(file_name)[0]
        
        # Construct corresponding wav file path
        wav_file_path = os.path.join(SPEECH_DIR, dir_path, f"{utterance_id}.wav")
        
        # Check if wav file exists
        if os.path.exists(wav_file_path):
            # Get audio duration
            duration = get_duration(wav_file_path)
            total_duration += duration
            print(str(total_duration) + ' processed!')
            
            # Read and normalize transcription
            transcription = read_transcription(trans_file)
            
            # Add entries if transcription is not empty
            if transcription:
                wav_scp_entries.append(f"{utterance_id} {wav_file_path}")
                text_entries.append(f"{utterance_id} {transcription}")
        else:
            print(f"Warning: WAV file not found for {trans_file}")
    
    # Write wav.scp file
    with open(WAV_SCP, 'w') as f:
        f.write('\n'.join(wav_scp_entries))
    
    # Write text file
    with open(TEXT_FILE, 'w') as f:
        f.write('\n'.join(text_entries))
    
    print(f"Processed {len(wav_scp_entries)} files")
    print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")

if __name__ == "__main__":
    main()
