# echo "Waiting 1 hours before training..."
# sleep 3600
export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 25420 src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name data/navida_train_data/r2r_vln_goal.jsonl \
    --model_name_or_path checkpoint/navida \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --learning_rate 1.0e-5 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --output_dir result/r2r_goal \
    --report_to tensorboard \
    --warmup_ratio 0.05