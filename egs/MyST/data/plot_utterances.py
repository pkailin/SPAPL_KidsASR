import matplotlib.pyplot as plt
import argparse
import sys
from collections import Counter

def read_utterances(file_path):
    """Read utterances from text file and return list of transcription lengths."""
    utterance_lengths = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split line into utterance_id and transcription
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    print(f"Warning: Line {line_num} doesn't have proper format (id + transcription)")
                    continue
                
                utterance_id, transcription = parts
                # Count words in transcription
                word_count = len(transcription.split())
                utterance_lengths.append(word_count)
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return utterance_lengths

def create_histogram(utterance_lengths, output_prefix='word_length_histogram'):
    """Create and save histogram of utterance lengths."""
    if not utterance_lengths:
        print("No valid utterances found.")
        return
    
    # Count occurrences of each length
    length_counts = Counter(utterance_lengths)
    total_utterances = len(utterance_lengths)
    
    # Prepare data for plotting
    lengths = sorted(length_counts.keys())
    counts = [length_counts[length] for length in lengths]
    percentages = [(count / total_utterances) * 100 for count in counts]
    
    # Print some statistics
    print(f"Total utterances: {total_utterances}")
    print(f"Average length: {sum(utterance_lengths) / len(utterance_lengths):.1f} words")
    print(f"Length range: {min(lengths)} - {max(lengths)} words")
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    bars = plt.bar(lengths, percentages, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title('Distribution of Utterance Lengths in MyST Test Set', fontsize=16, pad=20)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Percentage of Utterances (%)', fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars for significant percentages
    for bar, length, percentage in zip(bars, lengths, percentages):
        if percentage >= 1.0:  # Only label bars with >= 1% to avoid clutter
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Set x-axis to show reasonable range
    max_length = max(lengths)
    if max_length > 50:
        plt.xlim(0, min(50, max_length + 2))  # Limit x-axis if too many long utterances
        plt.text(0.98, 0.98, f'Note: Some utterances extend to {max_length} words',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        plt.xlim(0, max_length + 1)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the histogram
    output_filename = f'{output_prefix}_filtered.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved as '{output_filename}'")
    
    # Also save as PDF for better quality
    pdf_filename = f'{output_prefix}.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Histogram also saved as '{pdf_filename}'")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot distribution of utterance lengths from transcription file')
    parser.add_argument('input_file', help='Path to input text file with utterance_id and transcriptions')
    parser.add_argument('--output', '-o', default='myst_histogram',
                       help='Output filename prefix (default: myst_histogram)')
    
    args = parser.parse_args()
    
    # Read utterances and get lengths
    utterance_lengths = read_utterances(args.input_file)
    
    # Create and save histogram
    create_histogram(utterance_lengths, args.output)

if __name__ == "__main__":
    main()
