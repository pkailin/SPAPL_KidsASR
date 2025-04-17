import os
import re
import random
import librosa
import numpy as np
import torch
import jiwer
import soundfile as sf
from collections import defaultdict
from whisper_normalizer.english import EnglishTextNormalizer
import whisper

# For debugging
DEBUG = True

# Initialize the WhisperNormalizer
normalizer = EnglishTextNormalizer()

# Load Whisper small model
print("Loading Whisper small model...")
whisper_model = whisper.load_model("small")
print("Whisper model loaded")

# Function to parse the .tsv file with better error handling
def parse_tsv(tsv_file):
    utterance_paths = {}

    try:
        with open(tsv_file, 'r') as f:
            content = f.read()

        # Split on newlines and process each line
        lines = content.strip().split('\n')
        print(len(lines))

        for line in lines:
            # Split by tab (or whitespace if tabs aren't used)
            parts = line.strip().split('\t')

            print(parts)

            utterance_id, path = parts
            utterance_paths[utterance_id] = path
            
    except Exception as e:
        print(f"Error parsing TSV file: {e}")
        
    print(f"Total utterances found in TSV: {len(utterance_paths)}")

    return utterance_paths


def parse_mlf(mlf_file):
    """Parse the MLF file to get utterance transcriptions."""
    utterance_to_words = {}

    with open(mlf_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if len(lines) == 0 or lines[0] != "#!mlf!#":
        print(f"Warning: MLF file does not start with #!mlf!# header")
        return utterance_to_words

    i = 1  # Start after the header
    while i < len(lines):
        # Look for utterance ID lines (enclosed in quotes and ending with .lab")
        if lines[i].startswith('"') and lines[i].endswith('.lab"'):
            # Extract utterance ID without the .lab extension and quotes
            utterance_id = lines[i].strip('"').replace('.lab', '')

            # Collect words until we hit a period on its own line
            words = []
            i += 1
            while i < len(lines) and lines[i] != '.':
                # Skip empty lines
                if lines[i] and not lines[i].startswith('#'):
                    words.append(lines[i])
                i += 1

            # Store the words for this utterance
            full_transcript = ' '.join(words)
            
            # Clean and normalize the transcript
            clean_transcript = ' '.join(re.findall(r'[a-zA-Z]+', full_transcript)).lower()

            # Apply WhisperNormalizer
            try:
                normalized_transcript = normalizer(clean_transcript)
            except Exception as e:
                print(f"Error normalizing transcript '{clean_transcript}': {e}")
                normalized_transcript = clean_transcript
            
            utterance_to_words[utterance_id] = normalized_transcript

            # Move past the period
            if i < len(lines) and lines[i] == '.':
                i += 1
        else:
            # Skip any unexpected lines
            i += 1

    return utterance_to_words

# Function to extract speaker ID from path
def extract_speaker(path):
    # Extract the speaker ID from the path
    # Example: /home/dawna/sld/imports/cslu_kids/speech/scripted/01/2/ksc00/ksc00020.wav
    # Speaker ID: ksc00
    try:
        match = re.search(r'/([^/]+)/[^/]+\.wav$', path)
        if match:
            return match.group(1)
        # Try another pattern if the first one doesn't match
        match = re.search(r'/([\w\d]+)[\d\w]*\.wav$', path)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error extracting speaker from {path}: {e}")
    return None

# Function to replace path prefix
def replace_path_prefix(path):
    return path.replace('/home/dawna/sld/imports/cslu_kids/speech/scripted/', 
                      '/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/')

# Function to get audio duration
def get_audio_duration(file_path):
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0

def load_audio(file_path):
    """Load audio file using soundfile instead of relying on ffmpeg"""
    try:
        # Load the audio data
        audio_data, sample_rate = sf.read(file_path)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Return the audio data and sample rate
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

# Function to count words in a transcript
def count_words(text):
    return len(text.split())

# Function to calculate WER using Whisper
def calculate_wer(audio_file, reference_text):
    try:
        # Load audio directly for Whisper
        audio_array, sample_rate = load_audio(audio_file)

        result = whisper_model.transcribe(audio_array, language="en")
        whisper_text = result["text"].strip()

        # Calculate WER before normalization
        wer = jiwer.wer(
            reference_text.lower().strip(),
            whisper_text.lower().strip()
        )
        
        return wer*100
    
    except Exception as e:
        print(f"Error calculating WER for {audio_file}: {e}")
        return 100.0  # Return max WER on error

# Function to filter utterances
def filter_utterances(utterance_data):
    filtered_data = {}
    duration_filtered = 0
    wer_filtered = 0
    
    print("Filtering utterances...")
    total = len(utterance_data)
    
    for idx, (utterance_id, (path, transcript)) in enumerate(utterance_data.items()):
        if idx % 10 == 0:
            print(f"Filtering progress: {idx}/{total}")
        
        # Check duration
        duration = get_audio_duration(path)
        if duration > 30.0:
            duration_filtered += 1
            continue
        
        # Check WER
        wer = calculate_wer(path, transcript)
        if wer > 70.0:
            wer_filtered += 1
            continue
        
        # If passed both filters
        filtered_data[utterance_id] = (path, transcript)
    
    print(f"Filtered out {duration_filtered} utterances due to duration > 30s")
    print(f"Filtered out {wer_filtered} utterances due to WER > 70%")
    print(f"Remaining utterances: {len(filtered_data)}")
    
    return filtered_data

# Function to create data splits
def split_data(utterance_data):
    # Group utterances by speaker
    speakers = defaultdict(list)
    for utterance_id, (path, transcript) in utterance_data.items():
        speaker = extract_speaker(path)
        if speaker:
            speakers[speaker].append((utterance_id, path, transcript))
        else:
            print(f"Warning: Could not extract speaker from {path}")
    
    if DEBUG:
        print(f"Found {len(speakers)} unique speakers")
        for speaker, utterances in list(speakers.items())[:3]:  # Print first 3 speakers for debugging
            print(f"Speaker {speaker}: {len(utterances)} utterances")
    
    # Convert to list and shuffle
    speaker_list = list(speakers.items())
    random.shuffle(speaker_list)
    
    # Calculate splits
    total_speakers = len(speaker_list)
    train_count = int(total_speakers * 0.8)
    dev_count = int(total_speakers * 0.1)
    
    # Split speakers
    train_speakers = speaker_list[:train_count]
    dev_speakers = speaker_list[train_count:train_count+dev_count]
    test_speakers = speaker_list[train_count+dev_count:]
    
    # Create data splits
    train_data = []
    for speaker, utterances in train_speakers:
        train_data.extend(utterances)
    
    dev_data = []
    for speaker, utterances in dev_speakers:
        dev_data.extend(utterances)
    
    test_data = []
    for speaker, utterances in test_speakers:
        test_data.extend(utterances)
    
    return train_data, dev_data, test_data

# Function to count utterances with fewer than 3 words
def count_short_utterances(data):
    count = 0
    for _, _, transcript in data:
        if count_words(transcript) < 3:
            count += 1
    return count

# Function to write output files
def write_output_files(data, prefix):
    wav_scp_file = f"{prefix}_wav.scp"
    text_file = f"{prefix}_text"
    
    with open(wav_scp_file, 'w') as wav_f, open(text_file, 'w') as text_f:
        for utterance_id, path, transcript in data:
            wav_f.write(f"{utterance_id} {path}\n")
            text_f.write(f"{utterance_id} {transcript}\n")

# Function to calculate total duration
def calculate_total_duration(data):
    total_duration = 0
    print(f"Calculating duration for {len(data)} files...")
    for idx, (utterance_id, path, transcript) in enumerate(data):
        if idx % 50 == 0:  # Print progress every 50 files
            print(f"Processing file {idx}/{len(data)}")
        duration = get_audio_duration(path)
        total_duration += duration
    return total_duration

# Format duration in hours, minutes, and seconds
def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

# Main function
def main():
    # Set file paths
    tsv_file = "/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/OGIKItrn01.tsv"  # Replace with your actual file path
    mlf_file = "/home/klp65/rds/hpc-work/cslu_kids/trans/scripted/OGIKItrn01.mlf"  # Replace with your actual file path
    
    if DEBUG:
        print(f"Processing TSV file: {tsv_file}")
        print(f"Processing MLF file: {mlf_file}")
    
    # Check if files exist
    if not os.path.exists(tsv_file):
        print(f"Error: TSV file {tsv_file} does not exist")
        return
    if not os.path.exists(mlf_file):
        print(f"Error: MLF file {mlf_file} does not exist")
        return
    
    # Parse input files
    utterance_paths = parse_tsv(tsv_file)
    utterance_transcripts = parse_mlf(mlf_file)
    
    print(len(utterance_paths))
    print(len(utterance_transcripts))

    if not utterance_paths:
        print("Error: No utterance paths found in TSV file")
        return
    
    if not utterance_transcripts:
        print("Error: No transcripts found in MLF file")
        return
    
    # Display some examples for debugging
    if DEBUG:
        print("\nSample utterance paths:")
        for i, (k, v) in enumerate(list(utterance_paths.items())[:3]):
            print(f"{i+1}. {k}: {v}")
        
        print("\nSample transcripts:")
        for i, (k, v) in enumerate(list(utterance_transcripts.items())[:3]):
            print(f"{i+1}. {k}: {v}")
    
    # Combine data and replace path prefix
    utterance_data = {}
    for utterance_id, path in utterance_paths.items():
        if utterance_id in utterance_transcripts:
            new_path = replace_path_prefix(path)
            utterance_data[utterance_id] = (new_path, utterance_transcripts[utterance_id])
    
    if DEBUG:
        print(f"\nMatched {len(utterance_data)} utterances between TSV and MLF files")
        if len(utterance_data) == 0:
            print("No matches found! Checking for potential format issues...")
            # Check if IDs might have different formats in the two files
            tsv_ids = list(utterance_paths.keys())[:5]
            mlf_ids = list(utterance_transcripts.keys())[:5]
            print(f"Sample TSV IDs: {tsv_ids}")
            print(f"Sample MLF IDs: {mlf_ids}")
    
    if not utterance_data:
        print("Error: No matching utterances found between TSV and MLF files")
        return
    
    # Filter utterances
    filtered_data = filter_utterances(utterance_data)
    
    if not filtered_data:
        print("Error: No utterances left after filtering")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split data
    train_data, dev_data, test_data = split_data(filtered_data)
    
    # Count short utterances
    train_short = count_short_utterances(train_data)
    dev_short = count_short_utterances(dev_data)
    test_short = count_short_utterances(test_data)
    
    # Write output files
    write_output_files(train_data, "train")
    write_output_files(dev_data, "dev")
    write_output_files(test_data, "test")
    
    # Calculate durations
    print("Calculating train set duration...")
    train_duration = calculate_total_duration(train_data)
    print("Calculating dev set duration...")
    dev_duration = calculate_total_duration(dev_data)
    print("Calculating test set duration...")
    test_duration = calculate_total_duration(test_data)
    total_duration = train_duration + dev_duration + test_duration
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total utterances before filtering: {len(utterance_data)}")
    print(f"Total utterances after filtering: {len(filtered_data)}")
    print(f"Train utterances: {len(train_data)} ({format_duration(train_duration)})")
    print(f"Dev utterances: {len(dev_data)} ({format_duration(dev_duration)})")
    print(f"Test utterances: {len(test_data)} ({format_duration(test_duration)})")
    print(f"Total audio duration: {format_duration(total_duration)}")
    print(f"Short utterances (< 3 words) in train: {train_short}")
    print(f"Short utterances (< 3 words) in dev: {dev_short}")
    print(f"Short utterances (< 3 words) in test: {test_short}")
    print(f"Total short utterances: {train_short + dev_short + test_short}")

if __name__ == "__main__":
    main()
