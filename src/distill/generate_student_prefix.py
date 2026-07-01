import argparse
import json
import random

import torch
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig

from src.distill.common import (
    IMAGE_SIZE,
    SYSTEM_PROMPT,
    build_vln_user_messages,
    iter_with_progress,
    load_jsonl,
    load_vl_model,
    part_output_path,
    resolve_device,
    resolve_num_gpus,
    run_multigpu,
    select_inference_images,
    shard_by_chunks,
    strip_goal_hint,
)


def generate_prefix(model, processor, messages, device, generation_config):
    texts = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ]
    imgs, _ = process_vision_info(messages)
    imgs = [item.resize(IMAGE_SIZE) for item in imgs]

    prompt_inputs = processor(
        text=texts,
        images=[imgs],
        return_tensors="pt",
        padding=True,
    )
    prompt_inputs = prompt_inputs.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **prompt_inputs,
            generation_config=generation_config,
            use_model_defaults=True,
        )

    input_token_len = prompt_inputs["input_ids"].shape[1]
    generated_text = processor.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    return generated_text.strip()


def process_one_example(
    sample_id,
    source_index,
    example,
    model,
    processor,
    device,
    generation_config,
):
    raw_user_prompt = example["conversations"][0]["value"]
    user_prompt, goal_hint = strip_goal_hint(raw_user_prompt)
    if goal_hint is None:
        raise ValueError(
            f"Sample at index {source_index} does not contain a goal hint."
        )

    sampled_images = select_inference_images(example["conversations"][0]["image"])
    suffix = user_prompt.split("current observation", 1)[1]
    messages = build_vln_user_messages(
        sampled_images,
        suffix,
        system_prompt=example.get("system", SYSTEM_PROMPT),
    )
    student_prefix = generate_prefix(
        model,
        processor,
        messages,
        device,
        generation_config,
    )

    return {
        "sample_id": sample_id,
        "source_index": source_index,
        "episode_id": example.get("episode_id"),
        "images": sampled_images,
        "user_prompt": user_prompt,
        "goal_hint": goal_hint,
        "ground_truth": example["conversations"][1]["value"],
        "student_prefix": student_prefix,
    }


def generate_worker(rank, world_size, args, tasks):
    device = resolve_device(rank)
    shard = shard_by_chunks(tasks, rank, world_size)
    output_path = (
        part_output_path(args.output_path, rank)
        if world_size > 1
        else args.output_path
    )
    if not shard:
        open(output_path, "w", encoding="utf-8").close()
        return

    model, processor = load_vl_model(
        args.model_path,
        args.base_model_path,
        device,
        args.attn_implementation,
        lora_path=args.lora_path,
    )

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id, source_index, example in iter_with_progress(
            shard,
            rank,
            world_size,
            total=len(shard),
            desc="Generating student prefixes",
        ):
            record = process_one_example(
                sample_id,
                source_index,
                example,
                model,
                processor,
                device,
                generation_config,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args, tasks):
    run_multigpu(
        args.num_gpus,
        args.output_path,
        generate_worker,
        (args, tasks),
        total_items=len(tasks),
        progress_desc="Generating student prefixes",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate student action prefixes on goal data with goal hints removed."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoint/opd0",
        help="Path to student model weights.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="checkpoint/navida",
        help="Base model path used to补齐 processor/config files.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional LoRA weights to merge before generation.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/navida_train_data/r2r_goal.jsonl",
        help="Goal training data used as sampling pool.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/distill/student_prefix_256.jsonl",
        help="Output jsonl with student prefixes.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of random samples to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="GPU worker count. Default: all visible GPUs (see CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    random.seed(args.seed)

    all_examples = load_jsonl(args.data_path)
    if args.num_samples > len(all_examples):
        raise ValueError(
            f"Requested {args.num_samples} samples but only found {len(all_examples)} examples."
        )

    selected_indices = random.sample(range(len(all_examples)), args.num_samples)
    tasks = [
        (sample_id, source_index, all_examples[source_index])
        for sample_id, source_index in enumerate(selected_indices)
    ]

    args.num_gpus = resolve_num_gpus(args.num_gpus)
    print(f"Using {args.num_gpus} GPU worker(s)")

    run(args, tasks)
    print(f"Saved {len(tasks)} student prefixes to {args.output_path}")


if __name__ == "__main__":
    main()
