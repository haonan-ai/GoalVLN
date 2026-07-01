import argparse
import json

import torch

from src.distill.common import (
    SYSTEM_PROMPT,
    build_vln_user_messages,
    iter_with_progress,
    load_images_from_messages,
    load_jsonl,
    load_vl_model,
    parse_torch_dtype,
    part_output_path,
    resolve_device,
    resolve_num_gpus,
    run_multigpu,
    shard_by_chunks,
    strip_trailing_special_token_ids,
)


def build_teacher_messages(example, prefix=None):
    images = example["images"]
    user_prompt = example["user_prompt"]
    goal_hint = example["goal_hint"]

    suffix = user_prompt.split("current observation", 1)[1]
    suffix = f"{suffix.rstrip()} {goal_hint.strip()}"
    return build_vln_user_messages(
        images,
        suffix,
        system_prompt=SYSTEM_PROMPT,
        assistant_prefix=prefix,
    )


def compute_one_example(
    example,
    model,
    processor,
    device,
    top_k=5,
    temperature=1.0,
):
    prefix = example.get("student_prefix", "").strip()
    if not prefix:
        raise ValueError("Missing non-empty prefix in field `student_prefix`")

    prompt_messages = build_teacher_messages(example, prefix=None)
    full_messages = build_teacher_messages(example, prefix=prefix)

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs = load_images_from_messages(prompt_messages)

    prompt_inputs = processor(
        text=[prompt_text],
        images=[image_inputs],
        return_tensors="pt",
        padding=False,
    )
    full_inputs = processor(
        text=[full_text],
        images=[image_inputs],
        return_tensors="pt",
        padding=False,
    )

    prompt_len = prompt_inputs["input_ids"].shape[1]
    prefix_token_ids = strip_trailing_special_token_ids(
        full_inputs["input_ids"][0, prompt_len:].tolist(),
        processor.tokenizer,
    )
    if not prefix_token_ids:
        raise ValueError(f"Empty prefix token span. prefix={prefix}")

    decoded_prefix = processor.tokenizer.decode(
        prefix_token_ids,
        skip_special_tokens=True,
    ).strip()
    if decoded_prefix != prefix:
        raise ValueError(
            f"Decoded prefix mismatch.\n"
            f"  expected: {prefix!r}\n"
            f"  decoded:  {decoded_prefix!r}"
        )

    full_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in full_inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**full_inputs)
        logits = outputs.logits

    prefix_start = prompt_len
    prefix_end = prompt_len + len(prefix_token_ids)
    prefix_logits = logits[0, prefix_start - 1 : prefix_end - 1, :]

    if prefix_logits.shape[0] != len(prefix_token_ids):
        raise RuntimeError(
            f"Logit/token length mismatch: "
            f"logits={prefix_logits.shape[0]}, tokens={len(prefix_token_ids)}"
        )

    teacher_logprobs = torch.log_softmax(
        prefix_logits.float() / temperature,
        dim=-1,
    )
    topk_logprobs, topk_ids = torch.topk(teacher_logprobs, k=top_k, dim=-1)

    out = dict(example)
    out["prefix_token_ids"] = prefix_token_ids
    out["prefix_topk_token_ids"] = topk_ids.cpu().tolist()
    out["prefix_topk_logprobs"] = topk_logprobs.cpu().tolist()
    out["prefix_top_k"] = top_k
    return out


def teacher_topk_worker(rank, world_size, args, examples):
    device = resolve_device(rank)
    shard = shard_by_chunks(examples, rank, world_size)
    output_path = (
        part_output_path(args.output, rank)
        if world_size > 1
        else args.output
    )
    if not shard:
        open(output_path, "w", encoding="utf-8").close()
        return

    model, processor = load_vl_model(
        args.teacher_model,
        args.base_model_path,
        device,
        args.attn_implementation,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        use_cache=False,
    )
    for param in model.parameters():
        param.requires_grad_(False)

    with open(output_path, "w", encoding="utf-8") as fout:
        for example in iter_with_progress(
            shard,
            rank,
            world_size,
            total=len(shard),
            desc="teacher top-k scoring",
        ):
            try:
                out = compute_one_example(
                    example=example,
                    model=model,
                    processor=processor,
                    device=device,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )
            except Exception as exc:
                out = dict(example)
                out["topk_error"] = str(exc)

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def run(args, examples):
    run_multigpu(
        args.num_gpus,
        args.output,
        teacher_topk_worker,
        (args, examples),
        total_items=len(examples),
        progress_desc="teacher top-k scoring",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute teacher top-k logprobs on student prefixes with goal hints."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/distill/student_prefix_0.jsonl",
        help="Student prefix jsonl from step 1.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/distill/teacher_topk_0.jsonl",
        help="Output jsonl with teacher top-k distributions.",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="checkpoint/opd0",
        help="Teacher model weights.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="checkpoint/navida",
        help="Base model path used to补齐 processor/config files.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
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
    args = parser.parse_args()

    examples = load_jsonl(args.input)
    if args.limit > 0:
        examples = examples[: args.limit]

    args.num_gpus = resolve_num_gpus(args.num_gpus)
    print(f"Using {args.num_gpus} GPU worker(s)")

    run(args, examples)
    print(f"Saved teacher top-k results to {args.output}")


if __name__ == "__main__":
    main()
