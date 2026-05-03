#!/bin/bash
START=${1:-0}
END=${2:-255}
[[ $# -eq 1 ]] && START=0 && END=$(($1 - 1))

mkdir dataset/qwen3_processed

for i in $(seq $START $END); do
  idx=$(printf "%05d" $i)
  python Megatron-LM/tools/preprocess_data.py \
    --input dataset/c4_data/en/c4-train.${idx}-of-01024.json.gz \
    --output-prefix dataset/qwen3_processed/qwen3-processed-c4-${idx} \
    --tokenizer-model model_home/qwen3-30b-a3b \
    --tokenizer-type HuggingFaceTokenizer \
    --partitions 1 \
    --append-eod \
    --workers 97
done