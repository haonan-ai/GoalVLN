# Conda Environment
## 1. Create Conda Environment
```bash
conda create -n navida python=3.10
conda activate navida
```

## 2. Install habitat-sim v0.2.4
```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
cd ..
```

## 3. Install habitat-lab v0.2.4
```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
pip install dtw fastdtw gym
cd ..
```

## 4. Install NaVIDA Dependencies
```bash
pip install peft trl==0.16.0 transformers==4.50.3 tensorboardx qwen_vl_utils deepspeed distilabel wandb==0.18.3
pip install numpy==1.24.0 numba==0.60.0 tqdm opencv-python
pip install vllm==0.9.1 torch torchvision protobuf==3.20.1
pip install flash-attn --no-build-isolation --no-cache-dir
pip install "transformers<4.54.0"
pip install numpy==1.24.0
pip install "fastapi<0.137"
```

# Evaluation
```bash
ln -s /data1/haonan/NaVIDA/eval_log eval_log
```
## Option 1
```bash
./scripts/eval.sh

python src/eval/analyze_results.py \
    --path 

python src/eval/list_failed_episodes.py --path eval_log/navida_r2r

python src/eval/sample_episodes.py \
    --output data/selected_episodes_val_unseen_200.txt

python src/eval/summarize_results.py --path eval_log
```

## Option 2
Step 1: Launch vLLM Server
```bash
./scripts/start_vllm_server.sh
```

Step 2: Run Evaluation
```bash
./scripts/eval_vllm.sh
```

# Data
```bash
mkdir -p /data1/haonan/NaVIDA/data/sub_dataset
ln -s /data1/haonan/NaVIDA/data/sub_dataset data/sub_dataset
./scripts/preprocess.sh

mkdir -p /data1/haonan/NaVIDA/data/images
ln -s /data1/haonan/NaVIDA/data/images data/images
mkdir -p /data2/haonan/NaVIDA/data/poses
ln -s /data2/haonan/NaVIDA/data/poses data/poses
./scripts/extract_frame.sh

mkdir -p /data1/haonan/NaVIDA/data/navida_train_data
ln -s /data1/haonan/NaVIDA/data/navida_train_data data/navida_train_data
./scripts/prepare_training_data.sh
```

# Train
```bash
mkdir -p /data1/haonan/NaVIDA/result
ln -s /data1/haonan/NaVIDA/result result
./scripts/train.sh
```