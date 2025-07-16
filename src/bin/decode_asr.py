#!/usr/bin/env python3
# 2023-2024 Ruchao Fan  UCLA SPAPL

import os

os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import argparse
import torch
import evaluate
from transformers import WhisperProcessor, AutoConfig

sys.path.append(os.environ['rootdir']+'/src')
from data.whisper_loader import WhisperDataset
from models.feature_extractor import WhisperFeatureExtractor
from models.modeling_whisper import WhisperForConditionalGeneration
from arguments import PEFTArguments

import jiwer

def main():
    parser = argparse.ArgumentParser(description="Decoding the evaluation data in the wav_scp file")
    parser.add_argument("--wav_scp", required=True, type=str)
    parser.add_argument("--trn_scp", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--processor", required=True, type=str)
    parser.add_argument("--result_ref_file", required=True, type=str)
    parser.add_argument("--result_hyp_file", required=True, type=str)
    parser.add_argument("--compute_wer", required=True, default=True, type=bool)
    parser.add_argument("--chunk_length", required=True, default=30, type=int)
    parser.add_argument("--results_file", required=True, default='data.txt', type=str)

    parser.add_argument("--detailed_wer", default=True, type=bool, help="Whether to compute detailed WER statistics")

    args = parser.parse_args()

    data_path = {"data": {"scp_path": args.wav_scp, "text_label": args.trn_scp}}
    dataset = WhisperDataset(data_path).data

    print("Loading Model....")
    cache_dir_processor = "cached_whisper_models/" if not os.path.exists(args.processor) else None
    cache_dir_model = "cached_whisper_models/" if not os.path.exists(args.model) else None 
    
    config = AutoConfig.from_pretrained(args.model, cache_dir=cache_dir_model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.processor, cache_dir=cache_dir_processor) 
    processor = WhisperProcessor.from_pretrained(args.processor, cache_dir=cache_dir_processor)
    processor.current_processor = feature_extractor
    processor.feature_extractor = feature_extractor
    
    if hasattr(config, "peft_config"): 
        config.peft_config = PEFTArguments(**config.peft_config)
    
    model = WhisperForConditionalGeneration.from_pretrained(args.model, config=config, cache_dir=cache_dir_model).to("cuda")

    if hasattr(config, "peft_config"):
        print(config.peft_config)
        
        #config.peft_config = PEFTArguments(**config.peft_config)
        #print(config.peft_config.peft_type)

        if isinstance(config.peft_config, dict):
            config.peft_config = PEFTArguments(**config.peft_config)
        elif not isinstance(config.peft_config, PEFTArguments):
            config.peft_config = PEFTArguments(config.peft_config)


        if config.peft_config.peft_type == "prefix_tuning":
            model.generation_config.max_length -= config.peft_config.prefix_seq_len[1]
        
    num_utt = 0
    if args.compute_wer:
        metric_wer = evaluate.load("wer")
        references = []
        transcriptions = []

    ref_writer = open(args.result_ref_file, 'w')
    hyp_writer = open(args.result_hyp_file, 'w')

    with open(args.results_file, "w", encoding="utf-8") as results_file:

        for testdata in dataset:
            num_utt += 1

            audio = testdata["audio"]
            audio_duration = len(audio["array"]) / audio["sampling_rate"]
        
            if audio_duration > args.chunk_length:
                #inputs = processor(audio["array"], return_tensors="pt", truncation=False, padding=True, return_attention_mask=True,                                                                                          sampling_rate=audio["sampling_rate"],) #use_vtlp=True,)
                inputs = processor(audio["array"], return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, 
                                sampling_rate=audio["sampling_rate"],) #use_vtlp=True,)
                inputs = inputs.to("cuda")
                with torch.no_grad():
                    
                    segments = model.generate(**inputs, return_segments=True)["segments"][0]

                sequence = []
                for i in range(len(segments)):
                    sequence.append(processor.decode(segments[i]['tokens'], skip_special_tokens=True))
                prediction = " ".join(sequence)
                
            else:
                #inputs = processor(audio["array"], return_tensors="pt", padding=True, sampling_rate=audio["sampling_rate"],) #use_vtlp=True,)
                inputs = processor(audio["array"], return_tensors="pt", sampling_rate=audio["sampling_rate"],) #use_vtlp=True,)
                input_features = inputs.input_features.to("cuda")
                with torch.no_grad():
                    generated_ids = model.generate(inputs=input_features)

                prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            reference = processor.tokenizer._normalize(testdata['sentence'])
            prediction = processor.tokenizer._normalize(prediction)
    
            if args.compute_wer and len(reference) > 0:
                references.append(reference)
                transcriptions.append(prediction)

            results_file.write(f"{testdata['utt_id']}<DIV>{prediction}<DIV>{reference}\n")
            print(f"{testdata['utt_id']}<DIV>{prediction}<DIV>{reference}\n")

            utt_id = testdata["utt_id"].replace('-', '_')
            ref_writer.write(reference.upper() + ' (' + utt_id + ')\n')
            hyp_writer.write(prediction.upper() + ' (' + utt_id + ')\n')

            if num_utt % 200 == 0:
                print("Processed {} utterances out of {}".format(num_utt, len(dataset)), flush=True)
    
    if args.compute_wer:
        wer = metric_wer.compute(references=references, predictions=transcriptions)
        print("Word Error Rate: {}".format(wer), flush=True)

         # Detailed WER statistics using jiwer if requested
        if args.detailed_wer:
            # Compute detailed WER statistics
            measures = jiwer.compute_measures(
                references,
                transcriptions
            )

            # Extract and print the detailed metrics
            print("\nDetailed WER Statistics:")
            print(f"WER: {measures['wer']:.4f}")
            print(f"Hits: {measures['hits']}")
            print(f"Substitutions: {measures['substitutions']}")
            print(f"Deletions: {measures['deletions']}")
            print(f"Insertions: {measures['insertions']}")
            print(f"Total Words (Reference): {measures['substitutions'] + measures['deletions'] + measures['hits']}")

            # Calculate percentages
            total_ref_words = measures['substitutions'] + measures['deletions'] + measures['hits']
            print(f"\nSubstitution Rate: {measures['substitutions'] / total_ref_words:.4f}")
            print(f"Deletion Rate: {measures['deletions'] / total_ref_words:.4f}")
            print(f"Insertion Rate: {measures['insertions'] / total_ref_words:.4f}")


    ref_writer.close()
    hyp_writer.close()
        
if __name__ == "__main__":
    main()
