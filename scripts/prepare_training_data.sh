export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python3 src/data/prepare_training_data.py \
    --dataset_name r2r \
    --task_type vln goal \
    --output_path data/navida_train_data/r2r_vln_goal.jsonl

CUDA_VISIBLE_DEVICES=0 python3 src/data/prepare_training_data.py \
    --dataset_name r2r \
    --task_type idm \
    --output_path data/navida_train_data/r2r_idm.jsonl