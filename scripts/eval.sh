#!/bin/bash
# echo "Waiting 9 hours before training..."
# sleep 32400
SELECTED_EPISODES_FILE="data/selected_episodes_val_unseen_200.txt"
# DATASET_SPLIT=val_unseen
# MAX_EPISODES=200
USE_GOAL_HINT=false
SAVE_GIF=true

BASE_MODEL_PATH="checkpoint/navida"
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r.yaml"
EVAL_LOG_ROOT="eval_log"

CHECKPOINTS=(
    result/distill_opd0/checkpoint-50
    result/distill_opd0/checkpoint-100
    result/distill_opd0/checkpoint-150
    result/distill_opd0/checkpoint-200
    result/distill_opd0/checkpoint-250
)

CHUNKS=8
gpus=(0 0 1 1 2 2 3 3)

EXTRA_ARGS=()
if [ -n "$SELECTED_EPISODES_FILE" ]; then
    EXTRA_ARGS+=(--episode-ids-file "$SELECTED_EPISODES_FILE")
fi
if [ -n "$MAX_EPISODES" ]; then
    EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
fi
if [ -n "$DATASET_SPLIT" ]; then
    EXTRA_ARGS+=(--dataset-split "$DATASET_SPLIT")
fi
if [ "$USE_GOAL_HINT" = true ]; then
    EXTRA_ARGS+=(--use-goal-hint true)
fi
EXTRA_ARGS+=(--save-gif "$SAVE_GIF")

sync_missing_model_files() {
    local target="$1"
    local source="$2"
    if [ ! -d "$source" ] || [ "$(realpath "$target")" = "$(realpath "$source")" ]; then
        return
    fi
    mkdir -p "$target"
    for src in "$source"/*; do
        [ -e "$src" ] || continue
        name=$(basename "$src")
        if [ ! -e "$target/$name" ]; then
            echo "Copying missing model file: $name"
            cp -a "$src" "$target/$name"
        fi
    done
}

for MODEL_PATH in "${CHECKPOINTS[@]}"; do
    EXPERIMENT_NAME=$(basename "$(dirname "$MODEL_PATH")")
    CHECKPOINT_NAME=$(basename "$MODEL_PATH")
    SAVE_PATH="${EVAL_LOG_ROOT}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

    echo "========================================"
    echo "Evaluating checkpoint: $MODEL_PATH"
    echo "Saving results to: $SAVE_PATH"
    echo "========================================"

    sync_missing_model_files "$MODEL_PATH" "$BASE_MODEL_PATH"

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
        --base-model-path $BASE_MODEL_PATH \
        --num-generations 1 \
        --result-path $SAVE_PATH \
        "${EXTRA_ARGS[@]}" &

    done

    wait

    python src/eval/analyze_results.py \
        --path $SAVE_PATH
done
