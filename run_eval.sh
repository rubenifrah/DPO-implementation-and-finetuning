#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --account=m25146
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --job-name=dpo_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export PYTHONUNBUFFERED=1

# Activate venv
source venv/bin/activate

# Install requirements if needed (ensure acceleration/transformers are up to date)
# pip install -r requirements.txt

# 1. Generate Baseline (SFT)
echo "Generating Baseline (SFT)..."
python generate.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
    --output_file generations_sft.jsonl \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --split test_prefs \
    --limit 200 \
    --batch_size 4

# 2. Generate DPO (Beta 0.1)
echo "Generating DPO (Beta 0.1)..."
python generate.py \
    --model_name_or_path output_mistral_beta_0.1 \
    --output_file generations_dpo_0.1.jsonl \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --split test_prefs \
    --limit 200 \
    --batch_size 4

# 3. Judge Results
echo "Running Judge..."
python judge.py \
    --baseline_file generations_sft.jsonl \
    --candidate_file generations_dpo_0.1.jsonl \
    --judge_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_file eval_results_beta_0.1.json

echo "Evaluation Complete! Check eval_results_beta_0.1.json"
