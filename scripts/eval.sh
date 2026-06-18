#!/bin/bash
EPISODE_IDS_FILE="eval_log/navida_r2r/failed_episodes.txt"
MODEL_PATH="/data2/haonan/NaVIDA/checkpoint"
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="eval_log/navida_r2r_failed_true"

CHUNKS=6
gpus=(0 0 1 1 2 2)

EXTRA_ARGS=()
if [ -n "$EPISODE_IDS_FILE" ]; then
    EXTRA_ARGS+=(--episode-ids-file "$EPISODE_IDS_FILE")
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo ${gpus[$IDX]}
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} python src/eval/eval.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --forward-distance 25 \
    --turn-angle 15 \
    --resolution-ratio 0.5 \
    --max-action-history 200 \
    --model-path $MODEL_PATH \
    --num-generations 1 \
    --result-path $SAVE_PATH \
    "${EXTRA_ARGS[@]}" &
    
done

wait

python src/eval/analyze_results.py \
    --path $SAVE_PATH

