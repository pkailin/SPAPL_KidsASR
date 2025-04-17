import os
import wave
import numpy as np
from collections import defaultdict

# Hardcoded file paths - replace these with your actual paths
WAV_SCP = "train_wav.scp"  # Path to your wav.scp file
TEXT_FILE = "train_text_edited"   # Path to your text file
WAV_OUTPUT_DIR = "/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/train"  # Directory for combined wav files
META_OUTPUT_DIR = "./"  # Directory for wav_comb.scp and text_comb files
MAX_DURATION = 30.0  # Maximum duration in seconds for combined files

def get_wav_duration(wav_path):
    """Get the duration of a wav file in seconds."""
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration

def combine_wav_files(file_paths, output_path):
    """Combine multiple wav files into one."""
    data = []
    first_file = None
    
    for file_path in file_paths:
        with wave.open(file_path, 'rb') as w:
            if not first_file:
                first_file = w
                params = w.getparams()
            
            frames = w.readframes(w.getnframes())
            data.append(np.frombuffer(frames, dtype=np.int16))
    
    combined_data = np.concatenate(data)
    
    with wave.open(output_path, 'wb') as w:
        w.setparams(params)
        w.writeframes(combined_data.tobytes())

def main():
    # Create output directories if they don't exist
    os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)
    os.makedirs(META_OUTPUT_DIR, exist_ok=True)
    
    # Read wav.scp
    wav_dict = {}
    total_duration = 0.0
    with open(WAV_SCP, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, wav_path = parts
                wav_dict[utt_id] = wav_path
                total_duration += get_wav_duration(wav_path)

                print(str(total_duration) + ' processed!')
    
    
    # Read text file
    text_dict = {}
    with open(TEXT_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, transcript = parts
                text_dict[utt_id] = transcript


    print('Read wav.scp and text files into dict!')
    
    # Group files to combine
    groups = []
    current_group = []
    current_duration = 0.0
    
    for utt_id, wav_path in wav_dict.items():

        print('1 file parsed!')

        if utt_id not in text_dict:
            print(f"Warning: No transcript found for {utt_id}, skipping...")
            continue
        
        file_duration = get_wav_duration(wav_path)
        
        if current_duration + file_duration > MAX_DURATION and current_group:
            groups.append(current_group)
            current_group = []
            current_duration = 0.0
        
        current_group.append(utt_id)
        current_duration += file_duration
    
    # Add the last group if not empty
    if current_group:
        groups.append(current_group)
    
    # Combine files and generate new wav.scp and text files
    combined_wav_scp = []
    combined_text = []
    
    for i, group in enumerate(groups):
        # Use first utterance ID + _comb for the combined file name
        # We'll still store all utterance IDs in the metadata for reference
        combined_utt_id = f"{group[0]}_comb"
        combined_wav_path = os.path.join(WAV_OUTPUT_DIR, f"{combined_utt_id}.wav")
        
        # For the metadata entry, we'll keep track of all the combined utterance IDs
        full_combined_utt_id = "-".join(group)
        combined_transcription = " ".join([text_dict[utt_id] for utt_id in group])
        
        # Combine wav files
        wav_paths = [wav_dict[utt_id] for utt_id in group]
        combine_wav_files(wav_paths, combined_wav_path)
        
        # Store the full path in wav_comb.scp, but use the shorter filename
        combined_wav_scp.append(f"{combined_utt_id} {os.path.abspath(combined_wav_path)}")
        combined_text.append(f"{combined_utt_id} {combined_transcription}")
    
    # Write output files to META_OUTPUT_DIR
    with open(os.path.join(META_OUTPUT_DIR, "train_wav_comb.scp"), 'w') as f:
        f.write("\n".join(combined_wav_scp))
    
    with open(os.path.join(META_OUTPUT_DIR, "train_text_edited_comb"), 'w') as f:
        f.write("\n".join(combined_text))
    
    # Create a mapping file to keep track of which utterances were combined
    with open(os.path.join(META_OUTPUT_DIR, "utt_mapping"), 'w') as f:
        for i, group in enumerate(groups):
            combined_utt_id = f"{group[0]}_comb"
            f.write(f"{combined_utt_id} {' '.join(group)}\n")
    
    print(f"Total duration of all input files: {total_duration:.2f} seconds")
    print(f"Generated {len(groups)} combined files")
    print(f"WAV files written to {WAV_OUTPUT_DIR}")
    print(f"Metadata files written to {META_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
