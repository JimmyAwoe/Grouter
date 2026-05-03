#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:utils_grouter

CHECKPOINT_PATH="model_home/qwen3-30b-a3b-converted"
SAVE_PATH="checkpoints/distilled_grouter/48layer"
TOKENIZER_MODEL="model_home/qwen3-30b-a3b"
TOKENIZER_TYPE="HuggingFaceTokenizer"
DATA_HOME="dataset/qwen3_processed"
DATA_PATH=""
for i in {00000..00039}; do 
    DATA_PATH="${DATA_PATH} 0.01 ${DATA_HOME}/qwen3-processed-c4-${i}_text_document"
done


# distributed training setting
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6003"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}


# torchrun parameter
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)


MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 40960
    #--num-layers 48
    # only load 1 layer 
    # please make sure to use grouter-allow-partial-load
    --num-layers 1
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --use-flash-attn
    --vocab-size 151936
    --qk-layernorm
)

DISTILLATION_ARGS=(
    --grouter-enable-distillation
    --grouter-allow-partial-load
    --grouter-distillation-temperature 1.0
    --grouter-moe-layer-start 47
    --grouter-moe-layer-end 48
    --grouter-checkpoint-dir checkpoints/qwen3_distillation_48layer
    --grouter-checkpoint-interval 1000 
    --grouter-config-path utils_grouter/grouter/grouter_model/config.json
    --grouter-init-seed 1234
    #--grouter-resume-from checkpoints/qwen3_distillation/grouter_checkpoint_step_5000.pt
    --seed 42
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-grouped-gemm
    --moe-permute-fusion
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-token-dispatcher-type alltoall
    --moe-ffn-hidden-size 768
    --moe-router-topk-scaling-factor 1.0
    --moe-layer-freq 1
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function softmax
    #--moe-router-enable-expert-bias
    #--moe-router-bias-update-rate 1e-3
)

DATA_ARGS=(
    --tokenizer-model ${TOKENIZER_MODEL}
    --tokenizer-type ${TOKENIZER_TYPE}
    --data-path "$DATA_PATH"
    --split 1,0,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    #--recompute-granularity full
    #--recompute-method uniform
    #--recompute-num-layers 1
    --lr 0.0003
    --train-iters 10010
    --lr-decay-iters 8000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --lr-warmup-iters 3000
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --sequence-parallel 
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 100000
    --eval-interval 10000000
    --eval-iters 10
    --save $SAVE_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${SAVE_PATH}/tensorboard"
    --no-load-optim
    --no-load-rng
)

torchrun ${DISTRIBUTED_ARGS[@]} Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DISTILLATION_ARGS[@]} \
    ${LOGGING_ARGS[@]} > test.log


