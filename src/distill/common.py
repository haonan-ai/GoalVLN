import json
import math
import multiprocessing as mp
import os
import re
import shutil
import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYSTEM_PROMPT = "You are a helpful assistant."
GOAL_HINT_PATTERN = re.compile(
    r" Goal hint: Relative to the robot, the goal is "
    r"[\d.]+ m (?:front|back) and [\d.]+ m to the (?:left|right)\."
)
IMAGE_SIZE = (308, 252)
MAX_HISTORY_IMAGES = 8
VLN_HISTORY_PROMPT = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations"
)


def uniform_sample_with_ends(data, n):
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def normalize_images(images):
    if not isinstance(images, list):
        images = [images]
    return images


def select_inference_images(images):
    images = normalize_images(images)
    if len(images) > 1:
        return uniform_sample_with_ends(images[:-1], MAX_HISTORY_IMAGES) + [images[-1]]
    return [images[-1]]


def strip_goal_hint(user_value):
    match = GOAL_HINT_PATTERN.search(user_value)
    goal_hint = match.group(0).strip() if match else None
    vln_value = GOAL_HINT_PATTERN.sub("", user_value)
    return vln_value, goal_hint


def build_vln_user_messages(
    images,
    suffix_text,
    system_prompt=SYSTEM_PROMPT,
    assistant_prefix=None,
):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
    ]

    content = [{"type": "text", "text": VLN_HISTORY_PROMPT}]
    if len(images) > 1:
        content.extend([{"type": "image", "image": item} for item in images[:-1]])
    else:
        content.append({"type": "image", "image": images[-1]})
    content.append({"type": "text", "text": "and an image of the current observation"})
    content.append({"type": "image", "image": images[-1]})
    content.append({"type": "text", "text": suffix_text})
    messages.append({"role": "user", "content": content})

    if assistant_prefix is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_prefix}],
            }
        )
    return messages


def sync_missing_model_files(model_path, base_model_path):
    if not base_model_path or not os.path.isdir(base_model_path):
        return
    if os.path.abspath(model_path) == os.path.abspath(base_model_path):
        return

    os.makedirs(model_path, exist_ok=True)
    for name in os.listdir(base_model_path):
        src = os.path.join(base_model_path, name)
        dst = os.path.join(model_path, name)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"Copied missing model file: {name}")


def resolve_processor_path(model_path, base_model_path):
    if os.path.exists(os.path.join(model_path, "preprocessor_config.json")):
        return model_path
    return base_model_path


def load_vl_model(
    model_path,
    base_model_path,
    device,
    attn_implementation,
    torch_dtype=torch.bfloat16,
    lora_path=None,
    merge_lora=True,
    use_cache=True,
):
    sync_missing_model_files(model_path, base_model_path)

    model_init_kwargs = {
        "attn_implementation": attn_implementation,
        "use_cache": use_cache,
        "torch_dtype": torch_dtype,
    }
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, **model_init_kwargs
    )

    if lora_path:
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        if merge_lora:
            model = model.merge_and_unload()

    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        resolve_processor_path(model_path, base_model_path),
        use_fast=False,
    )
    processor.image_processor.max_pixels = 501760
    return model, processor


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_images_from_messages(messages):
    imgs, _ = process_vision_info(messages)
    processed = []
    for img in imgs:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        processed.append(img.resize(IMAGE_SIZE))
    return processed


def strip_trailing_special_token_ids(token_ids, tokenizer):
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for name in tokenizer.additional_special_tokens or []:
        if "im_end" in name or "endoftext" in name:
            special_ids.add(tokenizer.convert_tokens_to_ids(name))
    if tokenizer.eos_token:
        special_ids.add(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    tokens = list(token_ids)
    while tokens:
        if tokens[-1] in special_ids:
            tokens.pop()
            continue
        tail_text = tokenizer.decode([tokens[-1]], skip_special_tokens=False)
        if tail_text.strip() == "":
            tokens.pop()
            continue
        break
    return tokens


def parse_torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def resolve_device(rank):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return device


def resolve_num_gpus(num_gpus=None):
    if num_gpus is not None:
        return num_gpus
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def parse_distributed_setup(num_gpus):
    if "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, world_size, "worker"
    if num_gpus > 1:
        return -1, num_gpus, "spawn_parent"
    return 0, 1, "single"


def shard_by_chunks(items: Sequence[Any], rank: int, world_size: int):
    if world_size <= 1:
        return list(items)
    chunk_size = math.ceil(len(items) / world_size)
    start = rank * chunk_size
    end = min(start + chunk_size, len(items))
    return list(items[start:end])


def part_output_path(output_path, rank):
    return f"{output_path}.part{rank}"


def ensure_output_dir(output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)


def count_part_lines(output_path, world_size):
    total = 0
    for rank in range(world_size):
        part_path = part_output_path(output_path, rank)
        if not os.path.exists(part_path):
            continue
        with open(part_path, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def iter_with_progress(
    iterable: Iterable,
    rank: int,
    world_size: int,
    total: Optional[int] = None,
    desc: str = "Processing",
):
    if world_size <= 1:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def merge_part_files(output_path, num_parts):
    ensure_output_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rank in range(num_parts):
            part_path = part_output_path(output_path, rank)
            if not os.path.exists(part_path):
                raise FileNotFoundError(f"Missing partial output: {part_path}")
            with open(part_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
            os.remove(part_path)


def run_multigpu(
    num_gpus: int,
    output_path: str,
    worker_fn: Callable[..., None],
    worker_args: tuple = (),
    total_items: Optional[int] = None,
    progress_desc: str = "Processing",
):
    """Run workers and merge partial outputs; supports --num_gpus and torchrun."""
    ensure_output_dir(output_path)
    rank, world_size, mode = parse_distributed_setup(num_gpus)

    if mode == "spawn_parent":
        ctx = mp.get_context("spawn")
        processes = []
        for worker_rank in range(world_size):
            proc = ctx.Process(
                target=worker_fn,
                args=(worker_rank, world_size, *worker_args),
                daemon=False,
            )
            proc.start()
            processes.append(proc)

        stop_event = threading.Event()
        monitor = None
        pbar = None
        if total_items is not None:
            pbar = tqdm(total=total_items, desc=progress_desc)

            def _monitor_progress():
                last = 0
                while not stop_event.is_set():
                    done = count_part_lines(output_path, world_size)
                    if done > last:
                        pbar.update(done - last)
                        last = done
                    if done >= total_items:
                        break
                    time.sleep(0.5)
                done = count_part_lines(output_path, world_size)
                if done > last:
                    pbar.update(done - last)

            monitor = threading.Thread(target=_monitor_progress, daemon=True)
            monitor.start()

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                stop_event.set()
                if monitor is not None:
                    monitor.join()
                if pbar is not None:
                    pbar.close()
                raise RuntimeError(f"Worker failed with exit code {proc.exitcode}")

        stop_event.set()
        if monitor is not None:
            monitor.join()
        if pbar is not None:
            if total_items is not None:
                remaining = total_items - pbar.n
                if remaining > 0:
                    pbar.update(remaining)
            pbar.close()

        merge_part_files(output_path, world_size)
        return

    worker_fn(rank, world_size, *worker_args)

    if world_size > 1 and rank == 0:
        _wait_for_part_files(output_path, world_size)
        merge_part_files(output_path, world_size)


def _wait_for_part_files(output_path, world_size, timeout_sec=3600):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if all(
            os.path.exists(part_output_path(output_path, r)) for r in range(world_size)
        ):
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for partial outputs of {output_path}")
