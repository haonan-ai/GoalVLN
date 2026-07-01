export NCCL_P2P_LEVEL=NVL

MODEL_PATH="/data2/haonan/NaVIDA/checkpoint"

# TP size must divide num attention heads (Qwen2.5-VL: 16 heads -> valid TP: 1,2,4,8,16)
VLLM_GPUS="0,1"
TENSOR_PARALLEL_SIZE=2
VLLM_PORT=8201

CUDA_VISIBLE_DEVICES=$VLLM_GPUS vllm serve $MODEL_PATH --task generate \
    --trust-remote-code  --limit-mm-per-prompt image=99999 \
    --mm_processor_kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 --max-num-batched-tokens 65536 \
    --port $VLLM_PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.9
