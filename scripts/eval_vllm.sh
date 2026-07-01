#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="eval_log/r2r"
SAVE_GIF=true

# Habitat simulation workers run on a separate GPU from vLLM
SIM_GPUS="3"
VLLM_PORT=8201

CHUNKS=32
# MAX_EPISODES=1

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"

EXTRA_ARGS=()
if [ -n "$MAX_EPISODES" ]; then
    EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
fi
EXTRA_ARGS+=(--save-gif "$SAVE_GIF")

CUDA_VISIBLE_DEVICES=$SIM_GPUS python src/eval/eval_vllm.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --forward-distance 25 \
    --turn-angle 15 \
    --max-action-history 200 \
    --num-generations 1 \
    "${EXTRA_ARGS[@]}" \
    --result-path $SAVE_PATH

python src/eval/analyze_results.py \
    --path $SAVE_PATH
