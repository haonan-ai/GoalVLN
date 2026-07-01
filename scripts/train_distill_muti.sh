#!/usr/bin/env bash
set -euo pipefail

NUM_ROUNDS=10
START_ROUND=0

CUDA_VISIBLE_DEVICES="0,1,2,3"
MASTER_PORT_BASE=25421

TEACHER_MODEL="checkpoint/opd0"
BASE_MODEL_PATH="checkpoint/navida"
DATA_PATH="data/navida_train_data/r2r_goal.jsonl"
NUM_SAMPLES=5000
SEED=42
TOP_K=5

# distill training
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1.0e-5
DISTILL_TEMPERATURE=1.0
LOGGING_STEPS=5
EVAL_STEPS=25
WARMUP_RATIO=0.05

# eval (runs only after all training rounds finish)
RUN_EVAL_AFTER_TRAIN=true
EVAL_LOG_ROOT="eval_log"
SELECTED_EPISODES_FILE="data/selected_episodes_val_unseen_200.txt"
CONFIG_PATH="config/vln_r2r.yaml"
EVAL_CHUNKS=8
EVAL_GPUS=(0 0 1 1 2 2 3 3)
USE_GOAL_HINT=false
SAVE_GIF=true
# ============================================

export PYTHONPATH="./:${PYTHONPATH:-}"
export NCCL_P2P_LEVEL=NVL

mkdir -p data/distill result checkpoint eval_log

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

END_ROUND=$((START_ROUND + NUM_ROUNDS - 1))
TOTAL_NUM_ROUNDS=$((START_ROUND + NUM_ROUNDS))

for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
    if [ "$ROUND" -eq "$START_ROUND" ]; then
        STUDENT_MODEL="checkpoint/opd${ROUND}"
    else
        STUDENT_MODEL="result/distill_opd$((ROUND - 1))"
    fi
    STUDENT_PREFIX="data/distill/student_prefix_${ROUND}.jsonl"
    TEACHER_TOPK="data/distill/teacher_topk_${ROUND}.jsonl"
    OUTPUT_DIR="result/distill_opd${ROUND}"
    MASTER_PORT=$((MASTER_PORT_BASE + ROUND))

    echo "========================================"
    echo "Distill round ${ROUND}"
    echo "  student:        ${STUDENT_MODEL}"
    echo "  teacher:        ${TEACHER_MODEL}"
    echo "  student_prefix: ${STUDENT_PREFIX}"
    echo "  teacher_topk:   ${TEACHER_TOPK}"
    echo "  train_output:   ${OUTPUT_DIR}"
    echo "========================================"

    echo "[Round ${ROUND}] Step 1/3: generate student prefix"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 src/distill/generate_student_prefix.py \
        --model_path "$STUDENT_MODEL" \
        --base_model_path "$BASE_MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_path "$STUDENT_PREFIX" \
        --num_samples "$NUM_SAMPLES" \
        --seed $((SEED + ROUND))

    echo "[Round ${ROUND}] Step 2/3: compute teacher top-k"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 src/distill/compute_teacher_topk.py \
        --input "$STUDENT_PREFIX" \
        --output "$TEACHER_TOPK" \
        --teacher_model "$TEACHER_MODEL" \
        --base_model_path "$BASE_MODEL_PATH" \
        --top_k "$TOP_K"

    echo "[Round ${ROUND}] Step 3/3: distill train"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" deepspeed --master_port "$MASTER_PORT" src/train/train_distill.py \
        --deepspeed scripts/zero2.json \
        --dataset_name "$TEACHER_TOPK" \
        --model_name_or_path "$STUDENT_MODEL" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --bf16 \
        --torch_dtype bfloat16 \
        --attn_implementation flash_attention_2 \
        --lr_scheduler_type cosine \
        --gradient_checkpointing True \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --dataloader_num_workers 4 \
        --dataloader_pin_memory \
        --learning_rate "$LEARNING_RATE" \
        --distill_temperature "$DISTILL_TEMPERATURE" \
        --global_num_rounds "$TOTAL_NUM_ROUNDS" \
        --global_round_index "$ROUND" \
        --logging_steps "$LOGGING_STEPS" \
        --eval_strategy steps \
        --eval_steps "$EVAL_STEPS" \
        --save_strategy no \
        --output_dir "$OUTPUT_DIR" \
        --report_to tensorboard \
        --warmup_ratio "$WARMUP_RATIO"

    echo "[Round ${ROUND}] Done. Training output saved to ${OUTPUT_DIR}"
done

echo "All ${NUM_ROUNDS} distill round(s) finished."

if [ "$RUN_EVAL_AFTER_TRAIN" = true ]; then
    EVAL_EXTRA_ARGS=()
    if [ -n "$SELECTED_EPISODES_FILE" ]; then
        EVAL_EXTRA_ARGS+=(--episode-ids-file "$SELECTED_EPISODES_FILE")
    fi
    if [ -n "${MAX_EPISODES:-}" ]; then
        EVAL_EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
    fi
    if [ -n "${DATASET_SPLIT:-}" ]; then
        EVAL_EXTRA_ARGS+=(--dataset-split "$DATASET_SPLIT")
    fi
    if [ "$USE_GOAL_HINT" = true ]; then
        EVAL_EXTRA_ARGS+=(--use-goal-hint true)
    fi
    EVAL_EXTRA_ARGS+=(--save-gif "$SAVE_GIF")

    echo "========================================"
    echo "Evaluating all round checkpoints"
    echo "========================================"

    for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
        MODEL_PATH="result/distill_opd${ROUND}"
        SAVE_PATH="${EVAL_LOG_ROOT}/distill_opd${ROUND}"

        echo "========================================"
        echo "Evaluating round ${ROUND}: ${MODEL_PATH}"
        echo "Saving results to: ${SAVE_PATH}"
        echo "========================================"

        sync_missing_model_files "$MODEL_PATH" "$BASE_MODEL_PATH"

        for IDX in $(seq 0 $((EVAL_CHUNKS - 1))); do
            echo "${EVAL_GPUS[$IDX]}"
            CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$IDX]}" python3 src/eval/eval.py \
                --exp-config "$CONFIG_PATH" \
                --split-num "$EVAL_CHUNKS" \
                --split-id "$IDX" \
                --forward-distance 25 \
                --turn-angle 15 \
                --resolution-ratio 0.5 \
                --max-action-history 200 \
                --model-path "$MODEL_PATH" \
                --base-model-path "$BASE_MODEL_PATH" \
                --num-generations 1 \
                --result-path "$SAVE_PATH" \
                "${EVAL_EXTRA_ARGS[@]}" &
        done

        wait

        python3 src/eval/analyze_results.py \
            --path "$SAVE_PATH"
    done

    echo "All round evaluations finished."
fi
