export PYTHONPATH=$PYTHONPATH:Megatron-Bridge/src:Megatron-Bridge/Megatron-LM

python Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
    --hf-model model_home/qwen3-30b-a3b \
    --megatron-path model_home/qwen3-30b-a3b-converted \
    --torch-dtype bfloat16 \
    --device-map auto \
    --trust-remote-code