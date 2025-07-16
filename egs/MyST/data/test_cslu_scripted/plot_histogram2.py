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

def parse_file_for_utterances(file_content):
    """
    Parse the file content and count utterances per grade
    Each line represents one utterance
    """
    lines = file_content.strip().split('\n')
    utterance_grades = []  # List of grades for each utterance
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Split by whitespace, first part should be the utterance ID
        parts = line.split()
        if len(parts) < 2:
            continue
            
        utterance_id = parts[0]
        grade = extract_grade_from_utterance_id(utterance_id)
        
        if grade is not None:
            utterance_grades.append(grade)
    
    return utterance_grades

def create_utterance_histogram(utterance_grades):
    """
    Create a histogram showing the number of utterances by grade level
    """
    # Count utterances per grade
    grade_counts = Counter(utterance_grades)
    total_utterances = len(utterance_grades)
    
    # Define grade order for plotting
    all_grades = ['K'] + [str(i) for i in range(1, 11)]
    
    # Prepare data for plotting (only grades that have utterances)
    grades = []
    counts = []
    
    for grade in all_grades:
        if grade in grade_counts:
            grades.append(grade)
            counts.append(grade_counts[grade])
    
    # Create the histogram
    plt.figure(figsize=(12, 6))
    bars = plt.bar(grades, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Grade Level', fontsize=12)
    plt.ylabel('Number of Utterances', fontsize=12)
    plt.title('Number of Utterances by Grade Level', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage labels inside bars
    for i, grade in enumerate(grades):
        count = grade_counts[grade]
        percentage = (count / total_utterances) * 100
        plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
    
    plt.tight_layout()
    
    # Save the histogram as PNG
    plt.savefig('utterance_histogram.png', dpi=300, bbox_inches='tight')
    print("Utterance histogram saved as 'utterance_histogram.png'")
    
    # Print summary statistics
    print(f"Total utterances: {total_utterances}")
    print(f"Utterance distribution by grade:")
    for grade in grades:
        count = grade_counts[grade]
        percentage = (count / total_utterances) * 100
        print(f"  Grade {grade}: {count} utterances ({percentage:.1f}%)")
    
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
        
        # Parse the data for utterance counting
        utterance_grades = parse_file_for_utterances(file_content)
        print(f"Found {len(utterance_grades)} valid utterances")
        
        if not utterance_grades:
            print("No valid utterances found in the file.")
            return
        
        # Create the histogram
        create_utterance_histogram(utterance_grades)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
