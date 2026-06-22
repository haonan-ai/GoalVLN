#!/bin/bash
SELECTED_EPISODES_FILE="data/selected_episodes_val_unseen_200.txt"
DATASET_SPLIT=val_unseen
BASE_MODEL_PATH="/data1/haonan/NaVIDA/checkpoint"
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r.yaml"
EVAL_LOG_ROOT="eval_log"

CHECKPOINTS=(
    /data1/haonan/NaVIDA/checkpoint
    result/r2r_vln/checkpoint-500
    result/r2r_vln/checkpoint-1000
    result/r2r_vln/checkpoint-1500
    result/r2r_vln/checkpoint-2000
)

CHUNKS=8
gpus=(0 0 1 1 2 2 3 3)
MAX_EPISODES=200

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
    CHECKPOINT_NAME=$(basename "$MODEL_PATH")
    SAVE_PATH="${EVAL_LOG_ROOT}/${CHECKPOINT_NAME}"

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
