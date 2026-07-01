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

# eval (runs only after all training rounds finish, vLLM multi-GPU)
RUN_EVAL_AFTER_TRAIN=true
EVAL_LOG_ROOT="eval_log"
SELECTED_EPISODES_FILE="data/selected_episodes_val_unseen_200.txt"
CONFIG_PATH="config/vln_r2r.yaml"
USE_GOAL_HINT=false
SAVE_GIF=false
# One GPU list: each card runs one vLLM (TP=1) + round-robin sim workers on the same card
EVAL_GPUS="0,1,2,3"
EVAL_CHUNKS=16
VLLM_BASE_PORT=8201
GPU_MEMORY_UTILIZATION=0.95
VLLM_STARTUP_TIMEOUT=600
ENFORCE_EAGER=false
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
    export OPENAI_API_KEY="EMPTY"

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

    IFS=',' read -ra GPU_ARR <<< "$EVAL_GPUS"
    NUM_GPUS=${#GPU_ARR[@]}
    VLLM_PIDS=()
    VLLM_PORTS=()
    CLEANUP_DONE=false

    start_vllm_servers() {
        local model_path="$1"
        local model_name
        model_name=$(basename "$model_path")
        export NCCL_P2P_LEVEL=NVL
        VLLM_PIDS=()
        VLLM_PORTS=()

        local vllm_extra_args=()
        if [ "$ENFORCE_EAGER" = true ]; then
            vllm_extra_args+=(--enforce-eager)
        fi

        for i in "${!GPU_ARR[@]}"; do
            local gpu="${GPU_ARR[$i]}"
            local port=$((VLLM_BASE_PORT + i))
            local cache_root="$(pwd)/${EVAL_LOG_ROOT}/.vllm_cache/${model_name}/gpu${gpu}_port${port}"
            rm -rf "${cache_root}/torch_compile_cache"
            mkdir -p "$cache_root"
            VLLM_PORTS+=("$port")
            echo "Starting vLLM on GPU ${gpu} port ${port} (cache: ${cache_root})..."

            setsid env \
                VLLM_CACHE_ROOT="$cache_root" \
                TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor" \
                CUDA_VISIBLE_DEVICES=$gpu \
                vllm serve "$model_path" --task generate \
                --trust-remote-code --limit-mm-per-prompt image=99999 \
                --mm_processor_kwargs '{"max_pixels": 501760}' \
                --max-model-len 32768 --max-num-batched-tokens 65536 \
                --port "$port" \
                --tensor-parallel-size 1 \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                "${vllm_extra_args[@]}" &
            VLLM_PIDS+=("$!")
        done
    }

    wait_for_vllm_servers() {
        local elapsed=0
        echo "Waiting for ${NUM_GPUS} vLLM server(s)..."
        while [ "$elapsed" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
            local all_ready=true
            for i in "${!VLLM_PORTS[@]}"; do
                local port="${VLLM_PORTS[$i]}"
                if ! curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null; then
                    all_ready=false
                    break
                fi
            done
            if $all_ready; then
                echo "All vLLM servers are ready"
                return 0
            fi
            for pid in "${VLLM_PIDS[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "vLLM server process exited unexpectedly (pid=$pid)"
                    return 1
                fi
            done
            sleep 5
            elapsed=$((elapsed + 5))
        done
        echo "vLLM servers failed to start within ${VLLM_STARTUP_TIMEOUT}s"
        return 1
    }

    stop_vllm_servers() {
        local pids=("${VLLM_PIDS[@]}")
        local ports=("${VLLM_PORTS[@]}")
        VLLM_PIDS=()
        VLLM_PORTS=()

        if [ ${#pids[@]} -eq 0 ] && [ ${#ports[@]} -eq 0 ]; then
            return
        fi

        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping vLLM server (pid=$pid)..."
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done

        local waited=0
        while [ "$waited" -lt 30 ]; do
            local any_alive=false
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    any_alive=true
                    break
                fi
            done
            if ! $any_alive; then
                break
            fi
            sleep 1
            waited=$((waited + 1))
        done

        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing vLLM server (pid=$pid)..."
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done

        for port in "${ports[@]}"; do
            pkill -TERM -f "vllm serve.*--port ${port}" 2>/dev/null || true
        done
        sleep 1
        for port in "${ports[@]}"; do
            pkill -KILL -f "vllm serve.*--port ${port}" 2>/dev/null || true
        done
        sleep 1
    }

    cleanup_vllm() {
        if [ "$CLEANUP_DONE" = true ]; then
            return
        fi
        CLEANUP_DONE=true
        trap - EXIT INT TERM
        stop_vllm_servers
    }
    trap cleanup_vllm EXIT INT TERM

    CHECKPOINTS=()
    for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
        CHECKPOINTS+=("result/distill_opd${ROUND}")
    done

    echo "========================================"
    echo "Evaluating all round checkpoints (vLLM)"
    echo "Checkpoints: ${CHECKPOINTS[*]}"
    echo "GPUs: ${EVAL_GPUS} (${NUM_GPUS} vLLM + sim pairs, ${EVAL_CHUNKS} workers)"
    echo "========================================"

    for MODEL_PATH in "${CHECKPOINTS[@]}"; do
        EXPERIMENT_NAME=$(basename "$(dirname "$MODEL_PATH")")
        CHECKPOINT_NAME=$(basename "$MODEL_PATH")
        SAVE_PATH="${EVAL_LOG_ROOT}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

        echo "========================================"
        echo "Evaluating checkpoint: ${MODEL_PATH}"
        echo "Saving results to: ${SAVE_PATH}"
        echo "========================================"

        sync_missing_model_files "$MODEL_PATH" "$BASE_MODEL_PATH"

        stop_vllm_servers
        start_vllm_servers "$MODEL_PATH"

        if ! wait_for_vllm_servers; then
            echo "Skipping checkpoint due to vLLM startup failure: ${MODEL_PATH}"
            continue
        fi

        WORKER_PIDS=()
        for IDX in $(seq 0 $((EVAL_CHUNKS - 1))); do
            GPU_IDX=$((IDX % NUM_GPUS))
            GPU="${GPU_ARR[$GPU_IDX]}"
            PORT="${VLLM_PORTS[$GPU_IDX]}"
            echo "Worker ${IDX}: GPU ${GPU} -> vLLM port ${PORT}"
            CUDA_VISIBLE_DEVICES=$GPU OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1" \
                python3 src/eval/eval_vllm.py \
                --exp-config "$CONFIG_PATH" \
                --split-num "$EVAL_CHUNKS" \
                --split-id "$IDX" \
                --forward-distance 25 \
                --turn-angle 15 \
                --resolution-ratio 0.5 \
                --max-action-history 200 \
                --num-generations 1 \
                --result-path "$SAVE_PATH" \
                "${EVAL_EXTRA_ARGS[@]}" &
            WORKER_PIDS+=("$!")
        done

        for pid in "${WORKER_PIDS[@]}"; do
            wait "$pid"
        done

        stop_vllm_servers

        python3 src/eval/analyze_results.py \
            --path "$SAVE_PATH"
    done

    cleanup_vllm
    echo "All round evaluations finished."
fi
