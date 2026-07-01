#!/usr/bin/env bash
export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 25421 src/train/train_distill.py \
    --deepspeed scripts/zero2.json \
    --dataset_name data/distill/teacher_topk_0.jsonl \
    --model_name_or_path checkpoint/opd0 \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --learning_rate 5.0e-6 \
    --distill_temperature 1.0 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 25 \
    --save_strategy no \
    --output_dir result/distill_opd0 \
    --report_to tensorboard \
    --warmup_ratio 0.05
