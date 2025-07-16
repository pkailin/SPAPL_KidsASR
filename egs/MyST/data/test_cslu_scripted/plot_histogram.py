import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter

def extract_grade_from_utterance_id(utterance_id):
    """
    Extract grade from utterance ID format: OGKF05-00041-SC006E
    The grade is in positions 4-5 (0-indexed), e.g., '05' for grade 5
    Special case: '0K' represents kindergarten
    """
    if len(utterance_id) < 6:
        return None
    
    grade_str = utterance_id[4:6]
    
    if grade_str == '0K':
        return 'K'  # Kindergarten
    elif grade_str.isdigit():
        grade_num = int(grade_str)
        if grade_num == 0:
            return 'K'  # Alternative kindergarten representation
        else:
            return str(grade_num)
    else:
        return None

def parse_file_content(file_content):
    """
    Parse the file content and extract unique speakers and their grades
    """
    lines = file_content.strip().split('\n')
    speaker_grades = {}  # speaker_id -> grade
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Split by whitespace, first part should be the utterance ID
        parts = line.split()
        if len(parts) < 2:
            continue
            
        utterance_id = parts[0]
        
        # Extract speaker ID (first two parts separated by dash)
        speaker_parts = utterance_id.split('-')
        if len(speaker_parts) < 2:
            continue
            
        speaker_id = f"{speaker_parts[0]}-{speaker_parts[1]}"
        grade = extract_grade_from_utterance_id(utterance_id)
        
        if grade is not None:
            speaker_grades[speaker_id] = grade
    
    return speaker_grades

def create_grade_histogram(speaker_grades):
    """
    Create a histogram showing the proportion of children in each grade
    """
    # Count grades
    grade_counts = Counter(speaker_grades.values())
    total_speakers = len(speaker_grades)
    
    # Define grade order for plotting
    all_grades = ['K'] + [str(i) for i in range(1, 11)]
    
    # Calculate proportions
    grades = []
    proportions = []
    
    for grade in all_grades:
        if grade in grade_counts:
            grades.append(grade)
            proportions.append(grade_counts[grade] / total_speakers)
    
    # Create the histogram
    plt.figure(figsize=(12, 6))
    bars = plt.bar(grades, proportions, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Grade Level', fontsize=12)
    plt.ylabel('Proportion of Children', fontsize=12)
    plt.title('Distribution of Children Across Grade Levels', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar, prop in zip(bars, proportions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{prop:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add count information
    for i, grade in enumerate(grades):
        count = grade_counts[grade]
        plt.text(i, proportions[i]/2, f'n={count}', ha='center', va='center', 
                fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    
    # Save the histogram as PNG
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
    print("Histogram saved as 'histogram.png'")
    
    # Print summary statistics
    print(f"Total unique speakers: {total_speakers}")
    print(f"Grade distribution:")
    for grade in grades:
        count = grade_counts[grade]
        prop = count / total_speakers
        print(f"  Grade {grade}: {count} children ({prop:.1%})")
    
    plt.show()

# Main execution - reads from file
def main():
    # Ask user for file path
    file_path = input("Enter the path to your text file: ")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        print(f"Successfully read file: {file_path}")
        print(f"File contains {len(file_content.strip().split())} lines")
        
        # Parse the data
        speaker_grades = parse_file_content(file_content)
        print("Parsed speaker grades:", speaker_grades)
        
        # Create the histogram
        create_grade_histogram(speaker_grades)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
