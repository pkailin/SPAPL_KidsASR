import os
import wave

# Hardcoded file paths
TSV_FILE = "/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/OGIKItrn01.tsv"  # Replace with your actual TSV file path
MLF_FILE = "/home/klp65/rds/hpc-work/cslu_kids/trans/scripted/OGIKItrn01.mlf"  # Replace with your actual MLF file path

def replace_path_prefix(path):
    return path.replace('/home/dawna/sld/imports/cslu_kids/speech/scripted/',
                        '/home/klp65/rds/hpc-work/cslu_kids/speech/scripted/')

def parse_tsv_file(tsv_file):
    """Parse the TSV file to get utterance IDs and their WAV file paths."""
    utterance_to_wav = {}
    
    with open(tsv_file, 'r') as f:
        content = f.read().strip()
        entries = content.split()
        
        i = 0
        while i < len(entries):
            # Check if we have both an utterance ID and a path
            if i + 1 < len(entries) and entries[i+1].startswith('/'):
                utterance_id = entries[i]
                wav_path = entries[i+1]
                wav_path = replace_path_prefix(wav_path)
                utterance_to_wav[utterance_id] = wav_path
                i += 2
            else:
                # Handle incomplete entries at the end
                i += 1
    
    return utterance_to_wav

def parse_mlf_file(mlf_file):
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
            utterance_to_words[utterance_id] = words
            
            # Move past the period
            if i < len(lines) and lines[i] == '.':
                i += 1
        else:
            # Skip any unexpected lines
            i += 1
    
    return utterance_to_words

def get_wav_duration(wav_file):
    """Get the duration of a WAV file in seconds."""
    try:
        with wave.open(wav_file, 'rb') as wf:
            # Duration = (number of frames) / (frame rate)
            duration = wf.getnframes() / wf.getframerate()
            return duration
    except (wave.Error, FileNotFoundError):
        print(f"Warning: Could not open WAV file {wav_file}")
        return 0

def analyze_utterances():
    """Analyze utterances based on transcription length and audio duration."""
    
    # Parse input files
    utterance_to_wav = parse_tsv_file(TSV_FILE)
    utterance_to_words = parse_mlf_file(MLF_FILE)

    print(utterance_to_wav)
    print(utterance_to_words)
    
    # Counters
    short_transcriptions = 0
    long_durations = 0
    
    # Count utterances with fewer than 3 words
    print(len(utterance_to_words))
    print(len(utterance_to_wav))
    for utterance_id, words in utterance_to_words.items():
        if len(words) < 3:
            short_transcriptions += 1
    
    # Count utterances longer than 30 seconds
    for utterance_id, wav_path in utterance_to_wav.items():
        if not os.path.exists(wav_path):
            print(f"Warning: WAV file not found: {wav_path}")
            continue
        
        #duration = get_wav_duration(wav_path)
        
        print(duration)
        if duration > 30:
            long_durations += 1
    
    return short_transcriptions, long_durations

def main():
    short_transcriptions, long_durations = analyze_utterances()
    
    print(f"Number of utterances with fewer than 3 words: {short_transcriptions}")
    print(f"Number of utterances longer than 30 seconds: {long_durations}")

if __name__ == "__main__":
    main()
