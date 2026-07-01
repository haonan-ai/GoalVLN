#!/bin/bash
SELECTED_EPISODES_FILE="data/selected_episodes_val_unseen_200.txt"
# DATASET_SPLIT=val_unseen
# MAX_EPISODES=200
USE_GOAL_HINT=false
SAVE_GIF=false

BASE_MODEL_PATH="checkpoint/navida"
export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG_PATH="config/vln_r2r.yaml"
EVAL_LOG_ROOT="eval_log"

CHECKPOINTS=(
    result/distill_opd0
    result/distill_opd1
    result/distill_opd2
    result/distill_opd3
    result/distill_opd4
    result/distill_opd5
    result/distill_opd6
    result/distill_opd7
    result/distill_opd8
    result/distill_opd9
)

# One GPU list: each card runs one vLLM (TP=1) + round-robin sim workers on the same card
GPUS="0,1,2,3"
CHUNKS=16
VLLM_BASE_PORT=8201
# Leave headroom on each GPU for Habitat (vLLM + sim share the same card)
GPU_MEMORY_UTILIZATION=0.95
VLLM_STARTUP_TIMEOUT=600
# Set true only if torch.compile cache issues persist after cache isolation below
ENFORCE_EAGER=false

export OPENAI_API_KEY="EMPTY"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}
VLLM_PIDS=()
VLLM_PORTS=()
CLEANUP_DONE=false

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
        # Per-checkpoint + per-GPU cache; stale artifacts from force-kill break the next start
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

    # Catch EngineCore / worker children that outlive the API server parent
    for port in "${ports[@]}"; do
        pkill -TERM -f "vllm serve.*--port ${port}" 2>/dev/null || true
    done
    sleep 1
    for port in "${ports[@]}"; do
        pkill -KILL -f "vllm serve.*--port ${port}" 2>/dev/null || true
    done
    sleep 1
}

cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    trap - EXIT INT TERM
    stop_vllm_servers
}
trap cleanup EXIT INT TERM

for MODEL_PATH in "${CHECKPOINTS[@]}"; do
    EXPERIMENT_NAME=$(basename "$(dirname "$MODEL_PATH")")
    CHECKPOINT_NAME=$(basename "$MODEL_PATH")
    SAVE_PATH="${EVAL_LOG_ROOT}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"

    echo "========================================"
    echo "Evaluating checkpoint: $MODEL_PATH"
    echo "Saving results to: $SAVE_PATH"
    echo "GPUs: $GPUS (${NUM_GPUS} vLLM + sim pairs, $CHUNKS workers)"
    echo "========================================"

    sync_missing_model_files "$MODEL_PATH" "$BASE_MODEL_PATH"

    stop_vllm_servers
    start_vllm_servers "$MODEL_PATH"

    if ! wait_for_vllm_servers; then
        echo "Skipping checkpoint due to vLLM startup failure: $MODEL_PATH"
        continue
    fi

    WORKER_PIDS=()
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        GPU_IDX=$((IDX % NUM_GPUS))
        GPU="${GPU_ARR[$GPU_IDX]}"
        PORT="${VLLM_PORTS[$GPU_IDX]}"
        echo "Worker ${IDX}: GPU ${GPU} -> vLLM port ${PORT}"
        CUDA_VISIBLE_DEVICES=$GPU OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1" \
            python src/eval/eval_vllm.py \
            --exp-config $CONFIG_PATH \
            --split-num $CHUNKS \
            --split-id $IDX \
            --forward-distance 25 \
            --turn-angle 15 \
            --resolution-ratio 0.5 \
            --max-action-history 200 \
            --num-generations 1 \
            --result-path $SAVE_PATH \
            "${EXTRA_ARGS[@]}" &
        WORKER_PIDS+=("$!")
    done

    for pid in "${WORKER_PIDS[@]}"; do
        wait "$pid"
    done

    stop_vllm_servers

    python src/eval/analyze_results.py \
        --path $SAVE_PATH
done
