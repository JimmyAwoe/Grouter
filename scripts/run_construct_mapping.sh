export PYTHONPATH="Megatron-LM:utils_grouter:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7,8

TARGET_NUM_EXPERTS=${1:-64}
OUTPUT_MAPPING="utils_grouter/grouter/grouter_model/cvt${TARGET_NUM_EXPERTS}_map_affinity.json"

DATA_HOME="dataset/qwen3_processed"
DATA_BLEND=""
for i in {00000..00003}; do # 1/25
    DATA_BLEND="${DATA_BLEND} 0.04 ${DATA_HOME}/qwen3-processed-c4-${i}_text_document"
done

torchrun --nproc-per-node 8 utils_grouter/tools/construct_mapping.py \
    --batch-size 16 \
    --max-length 4096 \
    --data-prefix $DATA_BLEND \
    --random-seed 1423 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model model_home/qwen3-30b-a3b \
    --total-steps 100 \
    --bf16 \
    --grouter-config-path utils_grouter/grouter/grouter_model/config.json \
    --grouter-checkpoint-path utils_grouter/grouter/grouter_model/grouter.pth \
    --target-num-experts ${TARGET_NUM_EXPERTS} \
    --output-mapping ${OUTPUT_MAPPING} \
    --mapping-strategy affinity \
    --verbose 