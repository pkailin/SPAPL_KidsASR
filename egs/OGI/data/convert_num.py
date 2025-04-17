import re
from num2words import num2words

def convert_numbers_to_words(text):
    """
    Convert all numeric digits and ordinals in text to their word equivalents.
    """
    # First, handle ordinal numbers like 1st, 2nd, 3rd, 4th
    def replace_ordinal(match):
        num_str = match.group(1)
        try:
            return num2words(int(num_str), ordinal=True)
        except ValueError:
            return match.group(0)
    
    # Handle regular numbers
    def replace_number(match):
        num = match.group(0)
        
        # Handle special cases like decimal numbers
        if '.' in num:  # Handle decimal numbers
            try:
                return num2words(float(num))
            except ValueError:
                return num
        else:
            try:
                return num2words(int(num))
            except ValueError:
                return num
    
    # First replace ordinals (must be done before regular numbers)
    # Pattern for ordinals like 1st, 2nd, 3rd, 4th, etc.
    ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
    text = re.sub(ordinal_pattern, replace_ordinal, text)
    
    # Then replace regular numbers
    # Pattern for regular numbers
    number_pattern = r'\b\d+\b|\b\d+\.\d+\b'
    text = re.sub(number_pattern, replace_number, text)
    
    return text

def process_file(input_file, output_file):
    """
    Process the input file, convert numbers to words, and write to output file.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            
            # Check if the line is empty
            if not line:
                f_out.write('\n')
                continue
                
            # Split the line into utterance ID and transcription
            parts = line.split(' ', 1)
            
            if len(parts) == 2:
                utterance_id, transcription = parts
                converted_transcription = convert_numbers_to_words(transcription)
                f_out.write(f"{utterance_id} {converted_transcription}\n")
            else:
                # If there's no space to split, just write the line as is
                f_out.write(f"{line}\n")

if __name__ == "__main__":
    input_file = "train_text"
    output_file = "train_text_edited"
    process_file(input_file, output_file)
    print(f"Conversion complete! Numbers and ordinals have been converted to words in {output_file}")

    input_file = "dev_text"
    output_file = "dev_text_edited"
    process_file(input_file, output_file)
    print(f"Conversion complete! Numbers and ordinals have been converted to words in {output_file}")

    input_file = "test_text"
    output_file = "test_text_edited"
    process_file(input_file, output_file)
    print(f"Conversion complete! Numbers and ordinals have been converted to words in {output_file}")
