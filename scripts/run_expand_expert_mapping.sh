export PYTHONPATH="Megatron-LM:utils_grouter:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Usage:
#   bash scripts/run_expand_expert_mapping.sh [TARGET_NUM_EXPERTS] [NPROC_PER_NODE]
# Example:
#   bash scripts/run_expand_expert_mapping.sh 256 1

TARGET_NUM_EXPERTS=${1:-256}

GROUTER_CONFIG_PATH="utils_grouter/grouter/grouter_model/config.json"
GROUTER_CHECKPOINT_PATH="utils_grouter/grouter/grouter_model/grouter_base.pth"
OUTPUT_CHECKPOINT_PATH="utils_grouter/grouter/grouter_model/grouter_expanded_${TARGET_NUM_EXPERTS}.pth"

DATA_HOME="dataset/qwen3_processed"
DATA_BLEND=""
for i in {00000..00003}; do # 1/25
    DATA_BLEND="${DATA_BLEND} 0.04 ${DATA_HOME}/qwen3-processed-c4-${i}_text_document"
done

torchrun --nproc-per-node 8 utils_grouter/tools/expand_expert_mapping.py \
    --batch-size 8 \
    --max-length 4096 \
    --data-prefix $DATA_BLEND \
    --random-seed 1423 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model model_home/qwen3-30b-a3b \
    --total-steps 400 \
    --bf16 \
    --grouter-config-path ${GROUTER_CONFIG_PATH} \
    --grouter-checkpoint-path ${GROUTER_CHECKPOINT_PATH} \
    --output-checkpoint-path ${OUTPUT_CHECKPOINT_PATH} \
    --target-num-experts ${TARGET_NUM_EXPERTS} \
    --bad-token-quantile 0.2 \
    --kmeans-iters 100 \
    --verbose
