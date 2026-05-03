#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:Megatron-LM:grouter_ep_optimizer

CHECKPOINT_PATH="checkpoints/tiny_qwen3"
TOKENIZER_MODEL="model_home/qwen3-30b-a3b"
TOKENIZER_TYPE="HuggingFaceTokenizer"
DATA_HOME="dataset/qwen3_processed"
WANDB_NAME="MoE_Router_Change"

ROUTER_ARGS="\
    --use-grouter-weight \
    --moe-router-load-balancing-type none \
    --moe-use-grouter \
    --use-single-grouter \
    --grouter-checkpoint-path utils_grouter/grouter/grouter_model/grouter.pth \
    --grouter-config-path utils_grouter/grouter/grouter_model/config.json \
"
DATA_PATH=""
for i in {00000..00249}; do # 1/25
    DATA_PATH="${DATA_PATH} 0.01 ${DATA_HOME}/qwen3-processed-c4-${i}_text_document"
done

# distrubuted training setting
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6002"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

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
    --max-position-embeddings 32768
    --num-layers 8
    --hidden-size 384
    --ffn-hidden-size 688
    --num-attention-heads 8
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000 
    --use-flash-attn
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-grouped-gemm
    --moe-permute-fusion
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-token-dispatcher-type alltoall
    --moe-ffn-hidden-size 344
    --moe-router-topk-scaling-factor 2.5
    --moe-shared-expert-overlap
    --mscale 1.0
    --mscale-all-dim 1.0 
    --moe-layer-freq 1
    --moe-shared-expert-intermediate-size 344
    ${ROUTER_ARGS}
)

DATA_ARGS=(
    # DeepSeek 使用其自定义的分词器，通常在 Megatron-LM 中作为 GPT2BPETokenizer 实现
    #--tokenizer-type Llama2Tokenizer
    # hunyuan tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --tokenizer-type ${TOKENIZER_TYPE}
    --data-path "$DATA_PATH"
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 8
    --global-batch-size 256
    #--recompute-granularity full
    #--recompute-method uniform
    #--recompute-num-layers 1
    --lr 1e-4
    --train-iters 30000
    --lr-decay-iters 25000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 5000
    --clip-grad 1.0
    --bf16
    # --recompute-granularity selective 
    # --recompute-modules moe 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --sequence-parallel # 使用 MoE 和 TP 时，必须启用序列并行
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
)

# --- WandB 配置 ---
if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"DeepSeekV3-Training"}
        --wandb-exp-name ${WANDB_NAME:-"Hunyuan"}
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} 
