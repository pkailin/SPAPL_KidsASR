import re

def load_transcriptions(file_path):
    """
    Load transcription file into a dictionary {utterance_id: words}.
    Handles empty hypotheses gracefully.
    """
    transcriptions = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(.*)\s*\((.+?)\)", line)
            if match:
                words, utt_id = match.groups()
                words = words.strip()  # remove extra spaces
                transcriptions[utt_id] = words
            else:
                raise ValueError(f"Line format error: {line}")
    return transcriptions


def wer_score(ref, hyp):
    """
    Calculate WER given two dictionaries {utt_id: words}.
    """
    total_words = 0
    total_errors = 0
    
    for utt_id in ref:
        ref_words = ref[utt_id].split()
        hyp_words = hyp.get(utt_id, "").split()
        
        # Compute edit distance
        errors = edit_distance(ref_words, hyp_words)
        total_errors += errors
        total_words += len(ref_words)
    
    if total_words == 0:
        return 0.0
    return total_errors / total_words

def edit_distance(ref_words, hyp_words):
    """
    Standard Levenshtein edit distance between two word lists.
    """
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # deletion
                    dp[i][j - 1],    # insertion
                    dp[i - 1][j - 1] # substitution
                )
    return dp[-1][-1]

# Usage Example:
ref = load_transcriptions("/home/klp65/SPAPL_KidsASR/egs/MyST/exp/whisper_zero_shot/small.en/test_cslu_scripted/ref.txt")
hyp = load_transcriptions("/home/klp65/SPAPL_KidsASR/egs/MyST/exp/whisper_zero_shot/small.en/test_cslu_scripted/hyp.txt")
wer = wer_score(ref, hyp)
print(f"WER: {wer:.2%}")

