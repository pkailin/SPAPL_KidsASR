import os
import sys
from pathlib import Path

def parse_mlf_file(mlf_path):
    """
    Parse MLF file and extract transcriptions for each utterance.
    
    Args:
        mlf_path (str): Path to the MLF file
    
    Returns:
        dict: Dictionary mapping utterance IDs to their transcriptions
    """
    transcriptions = {}
    current_utterance = None
    current_words = []
    
    try:
        with open(mlf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if this is a file header (starts and ends with quotes)
                if line.startswith('"') and line.endswith('"'):
                    # Save previous utterance if exists
                    if current_utterance and current_words:
                        # Remove .lab extension and extract utterance ID
                        utterance_id = current_utterance.replace('.lab', '')
                        transcription = ' '.join(current_words)
                        transcriptions[utterance_id] = transcription
                    
                    # Start new utterance
                    current_utterance = line.strip('"')
                    current_words = []
                
                # Check if this is the end marker (single period)
                elif line == '.':
                    # Save current utterance
                    if current_utterance and current_words:
                        utterance_id = current_utterance.replace('.lab', '')
                        transcription = ' '.join(current_words)
                        transcriptions[utterance_id] = transcription
                    
                    # Reset for next utterance
                    current_utterance = None
                    current_words = []
                
                # Otherwise, it's a word
                else:
                    if current_utterance:  # Only add words if we're in an utterance
                        current_words.append(line)
    
    except FileNotFoundError:
        print(f"Error: MLF file not found at {mlf_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MLF file: {e}")
        sys.exit(1)
    
    return transcriptions

def correct_transcriptions(input_text_path, mlf_transcriptions, output_path):
    """
    Read the input text file and create a corrected version using MLF transcriptions.
    
    Args:
        input_text_path (str): Path to input text file with incorrect transcriptions
        mlf_transcriptions (dict): Dictionary of correct transcriptions from MLF
        output_path (str): Path for output corrected text file
    """
    corrected_lines = []
    corrections_made = 0
    lines_processed = 0
    
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
                
                # Check if we have a correction for this utterance
                if utterance_id in mlf_transcriptions:
                    new_transcription = mlf_transcriptions[utterance_id]
                    corrected_lines.append(f"{utterance_id} {new_transcription}")
                    corrections_made += 1
                    
                    if old_transcription != new_transcription:
                        print(f"Corrected {utterance_id}:")
                        print(f"  Old: {old_transcription}")
                        print(f"  New: {new_transcription}")
                else:
                    # Keep original if no correction available
                    corrected_lines.append(line)
                    print(f"Warning: No correction found for {utterance_id}")
        
        # Write corrected transcriptions to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in corrected_lines:
                f.write(line + '\n')
        
        print(f"\nProcessing complete:")
        print(f"  Lines processed: {lines_processed}")
        print(f"  Corrections made: {corrections_made}")
        print(f"  Output written to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input text file not found at {input_text_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)

def main():
    """Main function to run the transcription correction."""
    
    # File paths
    mlf_path = "/home/klp65/rds/rds-altaslp-8YSp2LXTlkY/data/cslu_kids/trans/scripted/OGIKItrn01.mlf"
    
    # Get input and output file paths from user
    if len(sys.argv) < 2:
        print("Usage: python mlf_corrector.py <input_text_file> [output_text_file]")
        print("Example: python mlf_corrector.py transcriptions.txt corrected_transcriptions.txt")
        sys.exit(1)
    
    input_text_path = sys.argv[1]
    
    # Default output path if not provided
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Create output filename by adding '_corrected' before extension
        input_path = Path(input_text_path)
        output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
    
    print(f"MLF file: {mlf_path}")
    print(f"Input text file: {input_text_path}")
    print(f"Output text file: {output_path}")
    print()
    
    # Parse MLF file
    print("Parsing MLF file...")
    mlf_transcriptions = parse_mlf_file(mlf_path)
    print(f"Found {len(mlf_transcriptions)} transcriptions in MLF file")
    
    # Show a few examples
    print("\nSample transcriptions from MLF:")
    for i, (utterance_id, transcription) in enumerate(mlf_transcriptions.items()):
        if i >= 3:  # Show only first 3
            break
        print(f"  {utterance_id}: {transcription}")
    
    if len(mlf_transcriptions) > 3:
        print(f"  ... and {len(mlf_transcriptions) - 3} more")
    
    print()
    
    # Correct transcriptions
    print("Correcting transcriptions...")
    correct_transcriptions(input_text_path, mlf_transcriptions, output_path)

if __name__ == "__main__":
    main()
