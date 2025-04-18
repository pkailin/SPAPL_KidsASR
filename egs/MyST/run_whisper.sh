#!/usr/bin/env bash

# 2023-2024 (Ruchao Fan)
# experiments for whisper model

export rootdir=/home/klp65/SPAPL_KidsASR/
export PATH=$PATH:/home/klp65/kaldi/tools/sctk-20159b5/bin/:$rootdir/src/bin:

# control which stages to run 
stage=3
end_stage=3

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # decode myst development and test sets with openai whisper models
  # models: 
  # openai/whisper-tiny.en     39M
  # openai/whisper-base.en     74M
  # openai/whisper-small.en    244M
  # openai/whisper-medium.en   769M
  # openai/whisper-large       1550M
  # openai/whisper-large-v2    1550M

  #models="tiny.en base.en small.en medium.en large large-v2"

  models="small.en" # KL: only training and evaluating whisper small 
  comupte_wer=true     # in python code
  using_sclite=false    # post python code
  chunk_length=30
  expdir=exp/whisper_zero_shot

  # stage 1: evaluation of whisper model without any finetuning
  for model in $models; do
    model_name=openai/whisper-$model
    echo "Evaluating Model: $model_name"

    #for x in test_filter_lt30 test_filter_gt30 development_filter; do # test_raw development_raw; do # test_filter_lt30
    for x in test_cslu_scripted; do

      resultdir=$expdir/$model/${x}/
      [ ! -d $resultdir ] && mkdir -p $resultdir

      CUDA_VISIBLE_DEVICES="0" decode_asr.py \
        --wav_scp data/$x/wav.scp \
        --trn_scp data/$x/text \
        --model $model_name \
        --processor $model_name \
        --compute_wer $comupte_wer \
        --result_ref_file $resultdir/ref.txt \
        --result_hyp_file $resultdir/hyp.txt \
        --chunk_length $chunk_length > $resultdir/decode.log 2>&1
        
      if [ $using_sclite ]; then
        echo "compute WER using sclite for $x"
        sclite -r $resultdir/ref.txt -h $resultdir/hyp.txt -i rm -o all stdout > $resultdir/result.wrd.txt
      fi
    done
  done
  
  echo "[Stage 1] Evaluation of Whisper Models Finished."
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  # stage 2: finetuning whisper model

  exp_dir="exp/noDA_noCSLU_promptFT_lr1e-4_4ksteps/"
  #exp_dir="exp/whisper_small_en_trans_adapter_encdec_lr1e-4_bn32_zeroinit_2gpus_4ksteps/"

  [ ! -d $exp_dir ] && mkdir -p $exp_dir

  train_config=conf/whisper_small_train.yaml # change this configuration file for finetuning method

  #CUDA_VISIBLE_DEVICES="0,1" torchrun --rdzv-endpoint=localhost:21221 \
 	  #--nproc_per_node 2 $rootdir/src/bin/train_asr.py $train_config  > $exp_dir/train.log 2>&1 &
  
  # 1 GPU Hardcode  
  $rootdir/src/bin/train_asr.py $train_config  #> $exp_dir/train.log 2>&1 &
  
  echo "[Stage 2] Finetuning Whisper Models Finished."
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  # stage 3: evaluation of the finetuned whisper model

  exp_dir="exp/noDA_noCSLU_promptFT_lr1e-4_4ksteps/"

  comupte_wer=true     # in python code
  using_sclite=false    # post python code
  chunk_length=30

  for x in test_myst test_cslu_scripted test_cslu_spont; do

    checkpoints="checkpoint-4000"
    for checkpoint in $checkpoints; do
      resultdir=$exp_dir/$checkpoint/${x}/
      [ ! -d $resultdir ] && mkdir -p $resultdir

      CUDA_VISIBLE_DEVICES="0" decode_asr.py \
        --wav_scp data/$x/wav.scp \
        --trn_scp data/$x/text \
        --model $exp_dir/$checkpoint \
        --processor $exp_dir \
        --compute_wer $comupte_wer \
        --result_ref_file $resultdir/ref.txt \
        --result_hyp_file $resultdir/hyp.txt \
        --chunk_length $chunk_length > $resultdir/decode.log 2>&1
        
      if [ $using_sclite ]; then
        echo "compute WER using sclite for $x"
        sclite -r $resultdir/ref.txt -h $resultdir/hyp.txt -i rm -o all stdout > $resultdir/result.wrd.txt
      fi
    done
  done
fi
