---
language: en
license: apache-2.0
tags:
- whisper
- speech-recognition
- audio
---

# child_asr

This is a fine-tuned Whisper model for speech recognition.

## Model details

- Base model: ./exp/PP_noCSLU_adapterFT_lr1e-4_8ksteps/checkpoint-5000
- Processor: ./exp/PP_noCSLU_adapterFT_lr1e-4_8ksteps

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("child_asr")
model = WhisperForConditionalGeneration.from_pretrained("child_asr")

# Now you can use the model for inference
```
