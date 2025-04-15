import os
import re
from pathlib import Path
import contextlib
import numpy as np
import whisper
import jiwer
import soundfile as sf
from whisper_normalizer.english import EnglishTextNormalizer

# Define the base directory
BASE_DIR = "/home/klp65/rds/hpc-work/myst_child_conv_speech/data/development"
# Output files
OUTPUT_WAV_SCP = "dev_wav.scp"
OUTPUT_TEXT = "dev_text"

# Initialize the Whisper text normalizer
normalizer = EnglishTextNormalizer()

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

def convert_flac_to_wav(flac_path, wav_path):
    """Convert FLAC to WAV using soundfile"""
    try:
        # Read the FLAC file
        data, samplerate = sf.read(flac_path)
        
        # Write to WAV file
        sf.write(wav_path, data, samplerate, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Failed to convert {flac_path}: {e}")
        return False

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using soundfile"""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0

def read_trn_file(trn_path):
    """Read the transcription from a .trn file"""
    try:
        with open(trn_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content
    except Exception as e:
        print(f"Error reading {trn_path}: {e}")
        return ""

def normalize_text(text):
    """
    Normalize text with the following steps:
    1. Convert to lowercase
    2. Apply Whisper's normalizer
    3. Remove all characters except letters, numbers, and spaces
    """
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Apply Whisper's normalizer
    text = normalizer(text)
    
    # Step 3: Remove all characters except letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate between reference and hypothesis"""
    try:
        wer = jiwer.wer(
            reference.lower().strip(),
            hypothesis.lower().strip()
        )
        return wer
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 1.0  # Return high WER on error

def count_words(text):
    """Count the number of words in a text"""
    return len(text.split())

def main():
    # Load the Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("small")
    
    wav_scp_entries = []
    text_entries = []
    
    # Track processed files for status reporting
    total_files = 0
    processed_files = 0
    skipped_no_transcription = 0
    skipped_long_duration = 0
    skipped_high_wer = 0
    skipped_few_words = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(BASE_DIR):
        # Get all FLAC files
        flac_files = [f for f in files if f.endswith('.flac')]
        total_files += len(flac_files)
        
        for flac_file in flac_files:
            # Get the corresponding TRN file
            trn_file = flac_file.replace('.flac', '.trn')
            trn_path = os.path.join(root, trn_file)
            
            # Skip if TRN file doesn't exist
            if not os.path.exists(trn_path):
                skipped_no_transcription += 1
                continue
            
            # Extract utterance ID from filename
            utterance_id = flac_file.replace('.flac', '')
            
            # Convert FLAC to WAV
            flac_path = os.path.join(root, flac_file)
            wav_path = os.path.join(root, flac_file.replace('.flac', '.wav'))
            
            if not convert_flac_to_wav(flac_path, wav_path):
                continue
            
            # Check audio duration
            duration = get_audio_duration(wav_path)
            if duration > 30:
                skipped_long_duration += 1
                # Remove the WAV file if we're not using it
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                continue
            
            # Read reference transcription
            reference_text = read_trn_file(trn_path)
            if not reference_text:
                continue
            
            # Use Whisper to transcribe the audio
            try:
                # Load audio directly for Whisper
                audio_array, sample_rate = load_audio(wav_path)

                result = model.transcribe(audio_array, language="en")
                whisper_text = result["text"].strip()
                
                # Calculate WER before normalization
                wer = calculate_wer(reference_text, whisper_text)
                word_count = count_words(reference_text)
                
                # Skip if WER > 50% or word count < 3
                if wer > 0.5:
                    skipped_high_wer += 1
                    continue
                
                if word_count < 3:
                    skipped_few_words += 1
                    continue
                
                # Normalize the reference text for output
                normalized_text = normalize_text(reference_text)
                
                # Add to our output lists
                wav_scp_entries.append(f"{utterance_id} {wav_path}")
                text_entries.append(f"{utterance_id} {normalized_text}")
                
                processed_files += 1
                print(processed_files)

                # Print progress every 100 files
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files} files...")
                
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
    
    # Write to output files
    with open(OUTPUT_WAV_SCP, 'w', encoding='utf-8') as f:
        f.write('\n'.join(wav_scp_entries))
    
    with open(OUTPUT_TEXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_entries))
    
    # Print summary
    print("\nProcessing completed!")
    print(f"Total files found: {total_files}")
    print(f"Files processed and included: {processed_files}")
    print(f"Skipped (no transcription): {skipped_no_transcription}")
    print(f"Skipped (duration > 30s): {skipped_long_duration}")
    print(f"Skipped (WER > 50%): {skipped_high_wer}")
    print(f"Skipped (word count < 3): {skipped_few_words}")
    print(f"wav.scp entries: {len(wav_scp_entries)}")
    print(f"text entries: {len(text_entries)}")

if __name__ == "__main__":
    main()
