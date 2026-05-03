#!/bin/bash
export PYTHONPATH="Megatron-LM:utils_grouter:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

GROUTER_CONFIG_PATH=${1}
GROUTER_CHECKPOINT_PATH=${2}

DATA_BLEND="1.0 dataset/qwen3_processed/qwen3-processed-c4-00000_text_document"

torchrun --nproc-per-node 4 utils_grouter/tools/finetune_grouter.py \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --max-length 4096 \
    --bf16 \
    --random-seed 1423 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model model_home/qwen3-30b-a3b \
    --data-prefix $DATA_BLEND \
    --grouter-config-path ${GROUTER_CONFIG_PATH} \
    --grouter-checkpoint-path ${GROUTER_CHECKPOINT_PATH} \
    --output-dir utils_grouter/grouter/grouter_model \
    --learning-rate 1e-3 \
    --max-steps 400 \
    --loss-type aux_loss \
    --finetune-optim gradient \
    --warmup-steps 0 \
    --log-interval 1 \
    --finetune-mode last_layer \
    --verbose 
