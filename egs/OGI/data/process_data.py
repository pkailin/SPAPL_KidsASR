import os
import re
import random
import librosa
from collections import defaultdict
from whisper_normalizer.english import EnglishTextNormalizer
import inflect

# Create an inflect engine
p = inflect.engine()

# Initialize the WhisperNormalizer
normalizer = EnglishTextNormalizer()

# Function to parse the .tsv file
def parse_tsv(tsv_file):
    utterance_paths = {}
    with open(tsv_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                i = 0
                while i < len(parts):
                    if parts[i].startswith('OGKF'):  # This is an utterance ID
                        if i + 1 < len(parts) and parts[i+1].startswith('/home'):  # Next part is a path
                            utterance_paths[parts[i]] = parts[i+1]
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
    return utterance_paths

# Function to parse the .mlf file with multi-line transcriptions
def parse_mlf(mlf_file):
    utterance_transcripts = {}
    try:
        with open(mlf_file, 'r') as f:
            lines = f.readlines()
            
            # Skip the first line if it contains #!mlf!#
            start_idx = 1 if lines and lines[0].strip() == "#!mlf!#" else 0
            
            i = start_idx
            while i < len(lines):
                line = lines[i].strip()
                
                # Check if this line contains an utterance ID
                if line.startswith('"') and line.endswith('.lab"'):
                    utterance_id = line.strip('"').replace('.lab', '')
                    
                    # Collect all transcription lines until we encounter a period on a line by itself
                    transcript_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() != ".":
                        transcript_lines.append(lines[j].strip())
                        j += 1
                    
                    # Combine all transcription lines
                    full_transcript = ' '.join(transcript_lines)
                    
                    # Clean and normalize the transcript
                    clean_transcript = ' '.join(re.findall(r'[a-zA-Z]+', full_transcript)).lower()
                    
                    # Apply WhisperNormalizer
                    try:
                        normalized_transcript = normalizer(clean_transcript)
                    except Exception as e:
                        print(f"Error normalizing transcript '{clean_transcript}': {e}")
                        normalized_transcript = clean_transcript
                    
                    utterance_transcripts[utterance_id] = normalized_transcript
                    # Skip to after the period
                    i = j + 1
                else:
                    i += 1

    except Exception as e:
        print(f"Error parsing MLF file")
    
    return utterance_transcripts

# Function to extract speaker ID from path
def extract_speaker(path):
    # Extract the speaker ID from the path
    # Example: /home/dawna/sld/imports/cslu_kids/speech/scripted/01/2/ksc00/ksc00020.wav
    # Speaker ID: ksc00
    match = re.search(r'/([^/]+)/[^/]+\.wav$', path)
    if match:
        return match.group(1)
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

# Function to create data splits
def split_data(utterance_data):
    # Group utterances by speaker
    speakers = defaultdict(list)
    for utterance_id, (path, transcript) in utterance_data.items():
        speaker = extract_speaker(path)
        if speaker:
            speakers[speaker].append((utterance_id, path, transcript))
    
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
    for utterance_id, path, transcript in data:
        duration = get_audio_duration(path)
        total_duration += duration
        print(total_duration)
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
    tsv_file = "/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/OGIKItrn01.tsv"
    mlf_file = "/home/klp65/rds/hpc-work/cslu_kids/trans/scripted/OGIKItrn01.mlf"  # Replace with your actual file path
   
    # Parse input files
    utterance_paths = parse_tsv(tsv_file)
    utterance_transcripts = parse_mlf(mlf_file)
  
    print(len(utterance_paths))
    print(len(utterance_transcripts))

    # Combine data and replace path prefix
    utterance_data = {}
    for utterance_id, path in utterance_paths.items():
        if utterance_id in utterance_transcripts:
            new_path = replace_path_prefix(path)
            utterance_data[utterance_id] = (new_path, utterance_transcripts[utterance_id])
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split data
    train_data, dev_data, test_data = split_data(utterance_data)
    
    # Write output files
    write_output_files(train_data, "train")
    write_output_files(dev_data, "dev")
    write_output_files(test_data, "test")
    
    # Calculate durations
    train_duration = calculate_total_duration(train_data)
    dev_duration = calculate_total_duration(dev_data)
    test_duration = calculate_total_duration(test_data)
    total_duration = train_duration + dev_duration + test_duration
    
    # Print summary
    print(f"Total utterances: {len(utterance_data)}")
    print(f"Train utterances: {len(train_data)} ({format_duration(train_duration)})")
    print(f"Dev utterances: {len(dev_data)} ({format_duration(dev_duration)})")
    print(f"Test utterances: {len(test_data)} ({format_duration(test_duration)})")
    print(f"Total audio duration: {format_duration(total_duration)}")

if __name__ == "__main__":
    main()
