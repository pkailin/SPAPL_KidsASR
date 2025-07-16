import argparse
import os
from transformers import (
    AutoConfig,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from huggingface_hub import login
import yaml
import torch

import sys
sys.path.append(os.path.abspath("/home/klp65/rds/hpc-work/SPAPL_KidsASR/src"))


# Try to import adapter and PEFT modules
try:
    import transformers.adapters
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

try:
    import peft
    from peft import get_peft_config, PeftConfig, PeftModel
    PEFT_AVAILABLE = True
    print("peft import successful!")
except ImportError:
    PEFT_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Load model and push to Hugging Face Hub")
    parser.add_argument("--wav_scp", default="./data/train_myst/wav.scp", type=str, help="Path to wav scp file")
    parser.add_argument("--trn_scp", default="./data/train_myst/text", type=str, help="Path to transcription scp file")
    parser.add_argument("--model", default="./exp/PP_noCSLU_adapterFT_lr1e-4_8ksteps/checkpoint-5000", type=str, help="Path to model or model identifier")
    parser.add_argument("--processor", default="./exp/PP_noCSLU_adapterFT_lr1e-4_8ksteps", type=str, help="Path to processor or processor identifier")
    parser.add_argument("--compute_wer", default=True, type=bool, help="Whether to compute WER")
    parser.add_argument("--chunk_length", default=30, type=int, help="Chunk length in seconds")
    
    # Arguments for Hugging Face Hub
    parser.add_argument("--repo_id", default="child_asr", type=str, help="Repository ID on Hugging Face Hub (username/repo_name)")
    parser.add_argument("--token", default="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", type=str, help="Hugging Face access token")
    parser.add_argument("--commit_message", default="Upload model", type=str, help="Commit message")
    parser.add_argument("--model_card", type=str, help="Path to model card markdown file (optional)")
    
    parser.add_argument("--use_adapters", action="store_true", help="Explicitly enable adapter support")
    parser.add_argument("--config_yaml", default="./conf/whisper_small_train_PP.yaml", help="Path to YAML config file with PEFT parameters")

    args = parser.parse_args()

    # Log in to Hugging Face Hub
    print("Logging in to Hugging Face Hub...")
    login(token=args.token)

    # Load YAML config if provided
    peft_config = None
    if args.config_yaml:
        print(f"Loading config from {args.config_yaml}...")
        with open(args.config_yaml, 'r') as file:
            yaml_config = yaml.safe_load(file)
            
        # Extract PEFT-related parameters
        peft_type = yaml_config.get('peft_type', 'False')
        if peft_type != 'False':
            print(f"Found PEFT type: {peft_type}")
            
            # For adapter-specific parameters
            if peft_type == 'adapter':
                adapter_config = {
                    'bottleneck_dim': yaml_config.get('bottleneck_dim', 32),
                    'dropout': yaml_config.get('dropout', 0.1),
                    'to_encoder': yaml_config.get('to_encoder', True),
                    'peft_encoder_layers': yaml_config.get('peft_encoder_layers', []),
                    'to_decoder': yaml_config.get('to_decoder', True),
                    'peft_decoder_layers': yaml_config.get('peft_decoder_layers', [])
                }
                print(f"Adapter config: {adapter_config}")
    else:
        yaml_config = None

    # Load model and processor
    print("Loading Model...")
    cache_dir_processor = "cached_whisper_models/" if not os.path.exists(args.processor) else None
    cache_dir_model = "cached_whisper_models/" if not os.path.exists(args.model) else None 
    
    config = AutoConfig.from_pretrained(args.model, cache_dir=cache_dir_model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.processor, cache_dir=cache_dir_processor) 
    processor = WhisperProcessor.from_pretrained(args.processor, cache_dir=cache_dir_processor)
    processor.current_processor = feature_extractor
    processor.feature_extractor = feature_extractor
    

    from arguments import PEFTArguments

    # Handle PEFT config if present (remove if not using PEFT)
    try:
        from arguments import PEFTArguments

        # Load PEFT
        with open(args.config_yaml, "r") as f:
            # Convert to PEFTArguments
            peft_args = PEFTArguments(**config.peft_config)

            # Assign to your config (if needed)
            config.peft_config = peft_args

            print(peft_args)

        """
        if hasattr(config, "peft_config"):
            from peft.peft_model import PEFTArguments
            config.peft_config = PEFTArguments(**config.peft_config)
        """
    except (ImportError, AttributeError):
        print("PEFT not available or not used in this model. Continuing without PEFT.")
    
    # Load model with appropriate adapter settings
    print(f"Loading model from {args.model}...")
    
    # Determine if we need adapter/PEFT support
    use_adapters = args.use_adapters
    if yaml_config and yaml_config.get('peft_type') != 'False':
        use_adapters = True
    
    # Load model with adapter support if needed
    if use_adapters:
        if PEFT_AVAILABLE:
            # Set up loading with PEFT framework
            if args.config_yaml and yaml_config.get('peft_type') == 'adapter':
                print("Loading model with PEFT adapter support...")
                """
                # First load the base model
                base_model = WhisperForConditionalGeneration.from_pretrained(
                    args.model,
                    config=config,
                    cache_dir=cache_dir_model
                )
                
                # It's a saved PEFT model but needs to be loaded as such
                model = PeftModel.from_pretrained(
                    base_model,
                    args.model
                )
                print("Successfully loaded PEFT Model with Adapters!")
    
                """

                # Load the config first
                config_path = "exp/PP_noCSLU_adapterFT_lr1e-4_8ksteps/config.json"
                config = WhisperConfig.from_json_file(config_path)

                # Create the model with this config (which includes the adapter configuration)
                model = WhisperForConditionalGeneration(config)

                # Load the finetuned weights
                checkpoint_path = "exp/PP_noCSLU_adapterFT_lr1e-4_8ksteps/checkpoint-5000/pytorch_model.bin"
                model.load_state_dict(torch.load(checkpoint_path), strict=False)

    # Create model card if not provided
    if not args.model_card:
        model_card_content = f"""---
language: en
license: apache-2.0
tags:
- whisper
- speech-recognition
- audio
---

# {args.repo_id.split('/')[-1]}

This is a fine-tuned Whisper model for speech recognition.

## Model details

- Base model: {args.model if '/' in args.model else 'custom Whisper model'}
- Processor: {args.processor if '/' in args.processor else 'custom Whisper processor'}

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("{args.repo_id}")
model = WhisperForConditionalGeneration.from_pretrained("{args.repo_id}")

# Now you can use the model for inference
```
"""
        with open("README.md", "w") as f:
            f.write(model_card_content)
        model_card_path = "README.md"
    else:
        model_card_path = args.model_card

    # Push model and processor to the Hub
    print(f"Pushing model to {args.repo_id}...")
    model.push_to_hub(
        args.repo_id, 
        commit_message=args.commit_message,
        token=args.token,
    )
    
    print(f"Pushing processor to {args.repo_id}...")
    processor.push_to_hub(
        args.repo_id,
        commit_message=args.commit_message,
        token=args.token,
    )
    
    print(f"Successfully uploaded model and processor to {args.repo_id}")
    print(f"Model available at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
