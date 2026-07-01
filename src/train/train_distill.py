import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR

import datasets
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint

from src.distill.common import (
    SYSTEM_PROMPT,
    build_vln_user_messages,
    load_images_from_messages,
    strip_trailing_special_token_ids,
)

os.environ["HF_DATASETS_CACHE"] = "./.cache/huggingface_cache"

logger = logging.getLogger(__name__)
processor = None


@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "Teacher top-k jsonl produced by compute_teacher_topk.py."}
    )
    eval_split_ratio: float = field(
        default=0.05,
        metadata={"help": "Fraction of data reserved for validation."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = None
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation, e.g. flash_attention_2."},
    )


@dataclass
class DistillArguments:
    distill_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature applied to student logits during distillation."},
    )
    global_num_rounds: Optional[int] = field(
        default=None,
        metadata={
            "help": "Total distill rounds for a global warmup+cosine LR schedule. "
            "When set (>1), LR is computed over all rounds instead of resetting each round."
        },
    )
    global_round_index: int = field(
        default=0,
        metadata={"help": "Current round index (0-based) within global_num_rounds."},
    )


class LogCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(logs)


def _student_suffix(user_prompt):
    return user_prompt.split("current observation", 1)[1]


def build_student_messages(example):
    """Student prompt without goal_hint; optional assistant prefix for training."""
    prefix = example.get("student_prefix", "").strip()
    return build_vln_user_messages(
        example["images"],
        _student_suffix(example["user_prompt"]),
        system_prompt=SYSTEM_PROMPT,
        assistant_prefix=prefix or None,
    )


def find_subsequence_start(sequence, subsequence):
    sequence = sequence.tolist() if torch.is_tensor(sequence) else list(sequence)
    subsequence = (
        subsequence.tolist() if torch.is_tensor(subsequence) else list(subsequence)
    )
    for idx in range(len(sequence) - len(subsequence) + 1):
        if sequence[idx : idx + len(subsequence)] == subsequence:
            return idx
    raise ValueError("Failed to align tokenized sample within padded batch.")


def collate_distill_fn(examples):
    prefix_starts = []
    prefix_lengths = []
    topk_token_ids = []
    topk_logprobs = []
    unbatched_input_ids = []

    texts = []
    image_inputs = []
    for example in examples:
        prompt_messages = build_vln_user_messages(
            example["images"],
            _student_suffix(example["user_prompt"]),
            system_prompt=SYSTEM_PROMPT,
        )
        full_messages = build_student_messages(example)
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
        image_input = load_images_from_messages(prompt_messages)

        prompt_inputs = processor(
            text=[prompt_text],
            images=[image_input],
            return_tensors="pt",
            padding=False,
        )
        full_inputs = processor(
            text=[full_text],
            images=[image_input],
            return_tensors="pt",
            padding=False,
        )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        prefix_token_ids = strip_trailing_special_token_ids(
            full_inputs["input_ids"][0, prompt_len:].tolist(),
            processor.tokenizer,
        )
        if not prefix_token_ids:
            raise ValueError(f"Empty prefix token span for sample {example.get('sample_id')}")

        teacher_topk_ids = example["prefix_topk_token_ids"]
        teacher_topk_logprobs = example["prefix_topk_logprobs"]
        if len(teacher_topk_ids) != len(prefix_token_ids):
            raise ValueError(
                f"Teacher top-k length ({len(teacher_topk_ids)}) != "
                f"student prefix length ({len(prefix_token_ids)}) "
                f"for sample {example.get('sample_id')}."
            )

        unbatched_input_ids.append(full_inputs["input_ids"][0])
        prefix_starts.append(prompt_len)
        prefix_lengths.append(len(prefix_token_ids))
        topk_token_ids.append(teacher_topk_ids)
        topk_logprobs.append(teacher_topk_logprobs)
        texts.append(full_text)
        image_inputs.append(image_input)

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    aligned_prefix_starts = []
    for input_id, raw_ids, prompt_len in zip(
        batch["input_ids"], unbatched_input_ids, prefix_starts
    ):
        seq_start = find_subsequence_start(input_id, raw_ids)
        aligned_prefix_starts.append(seq_start + prompt_len)

    batch["prefix_starts"] = torch.tensor(aligned_prefix_starts, dtype=torch.long)
    batch["prefix_lengths"] = torch.tensor(prefix_lengths, dtype=torch.long)
    batch["topk_token_ids"] = topk_token_ids
    batch["topk_logprobs"] = topk_logprobs
    return batch


def sparse_topk_kl_loss(
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL(teacher || student) on a sparse top-k support.
    Teacher logprobs are renormalized over the provided top-k candidates.
    """
    student_logprobs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_topk_logprobs = student_logprobs.gather(-1, topk_ids)

    teacher_probs = F.softmax(teacher_topk_logprobs.float(), dim=-1)
    loss = -(teacher_probs * student_topk_logprobs).sum(dim=-1)
    return loss.mean()


def _cosine_warmup_lr_multiplier(
    step: int,
    num_warmup_steps: int,
    num_training_steps: int,
) -> float:
    if step < num_warmup_steps:
        return float(step) / float(max(1, num_warmup_steps))
    progress = float(step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class DistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        distill_temperature=1.0,
        global_num_rounds=None,
        global_round_index=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distill_temperature = distill_temperature
        self.global_num_rounds = global_num_rounds
        self.global_round_index = global_round_index

    def create_scheduler(self, num_training_steps: int, optimizer):
        if self.global_num_rounds is None or self.global_num_rounds <= 1:
            return super().create_scheduler(num_training_steps, optimizer)

        global_total_steps = num_training_steps * self.global_num_rounds
        global_warmup_steps = self.args.get_warmup_steps(global_total_steps)
        global_step_offset = self.global_round_index * num_training_steps

        logger.info(
            "Global LR schedule: round=%s/%s, local_steps=%s, "
            "global_total_steps=%s, global_warmup_steps=%s, global_step_offset=%s",
            self.global_round_index + 1,
            self.global_num_rounds,
            num_training_steps,
            global_total_steps,
            global_warmup_steps,
            global_step_offset,
        )

        def lr_lambda(local_step: int):
            global_step = global_step_offset + local_step
            return _cosine_warmup_lr_multiplier(
                global_step,
                global_warmup_steps,
                global_total_steps,
            )

        return LambdaLR(optimizer, lr_lambda)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prefix_starts = inputs.pop("prefix_starts")
        prefix_lengths = inputs.pop("prefix_lengths")
        topk_token_ids = inputs.pop("topk_token_ids")
        topk_logprobs = inputs.pop("topk_logprobs")

        outputs = model(**inputs)
        logits = outputs.logits

        losses = []
        batch_size = logits.shape[0]
        for i in range(batch_size):
            start = int(prefix_starts[i].item())
            length = int(prefix_lengths[i].item())
            if length == 0:
                continue

            token_logits = logits[i, start - 1 : start - 1 + length, :]
            token_topk_ids = torch.tensor(
                topk_token_ids[i],
                device=logits.device,
                dtype=torch.long,
            )
            token_teacher_logprobs = torch.tensor(
                topk_logprobs[i],
                device=logits.device,
                dtype=logits.dtype,
            )
            losses.append(
                sparse_topk_kl_loss(
                    token_logits,
                    token_topk_ids,
                    token_teacher_logprobs,
                    temperature=self.distill_temperature,
                )
            )

        if not losses:
            loss = logits.sum() * 0.0
        else:
            loss = torch.stack(losses).mean()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, dict(inputs))
        return (loss.detach(), None, None)


def filter_valid_examples(dataset):
    def _is_valid(example):
        if example.get("topk_error"):
            return False
        if not example.get("student_prefix", "").strip():
            return False
        if "prefix_topk_token_ids" not in example:
            return False
        return True

    return dataset.filter(_is_valid)


def main(model_args, data_args, training_args, distill_args):
    set_seed(training_args.seed)

    handlers = [logging.StreamHandler(sys.stdout)]
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        os.makedirs(training_args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(training_args.output_dir, "train_distill.log")
        )
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)

    log_level = training_args.get_process_log_level()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: "
        f"{bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Distill parameters {distill_args}")
    logger.info(f"Training parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    dataset = load_dataset("json", data_files=data_args.dataset_name)
    raw_dataset = dataset["train"].shuffle(seed=42)
    raw_dataset = filter_valid_examples(raw_dataset)
    logger.info(f"Valid distill samples: {len(raw_dataset)}")

    eval_dataset = None
    if training_args.eval_strategy != "no":
        split = raw_dataset.train_test_split(
            test_size=data_args.eval_split_ratio,
            seed=42,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(
            f"Train samples: {len(train_dataset)}, eval samples: {len(eval_dataset)}"
        )
    else:
        train_dataset = raw_dataset

    global processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )
    processor.image_processor.max_pixels = 501760
    logger.info("Loaded processor for distill training.")

    torch_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
    )
    model = model.to(torch_dtype)
    model.accepts_loss_kwargs = False

    for p in model.visual.parameters():
        p.requires_grad = False
    for p in model.visual.merger.parameters():
        p.requires_grad = True

    logger.info(f"Student model initialized in {torch_dtype}")

    training_args.remove_unused_columns = False
    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_distill_fn,
        distill_temperature=distill_args.distill_temperature,
        global_num_rounds=distill_args.global_num_rounds,
        global_round_index=distill_args.global_round_index,
        callbacks=[LogCallback(logger)],
    )

    logger.info("*** Distill Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save distilled student model ***")
    trainer.accelerator.wait_for_everyone()
    if trainer.is_deepspeed_enabled:
        state_dict = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.accelerator.is_main_process:
            trainer.model.save_pretrained(
                training_args.output_dir,
                state_dict=state_dict,
            )
            processor.save_pretrained(training_args.output_dir)
            trainer.model.config.save_pretrained(training_args.output_dir)
            logger.info(f"Model saved to {training_args.output_dir}")
    else:
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            dataset=data_args.dataset_name,
            tags=["NaVIDA Distill"],
        )


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DistillArguments)
    )
    model_args, data_args, training_args, distill_args = (
        parser.parse_args_into_dataclasses()
    )
    main(model_args, data_args, training_args, distill_args)
