import json

# Load JSON
with open('test_clean.json', 'r') as f:
    data = json.load(f)

    # Open output files
    with open('wav.scp', 'w') as wav_f, open('text', 'w') as text_f:
        for utt_id, info in data.items():
            wav_path = info["wav"]
            transcript = info["word"]

            # Write to wav.scp
            wav_f.write(f"{utt_id} {wav_path}\n")

            # Write to text
            text_f.write(f"{utt_id} {transcript}\n")
