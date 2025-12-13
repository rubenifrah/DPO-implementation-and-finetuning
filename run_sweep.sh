#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 4              # 4 CPUs per job (much safer)
#SBATCH --gres=gpu:1      # 1 GPU per job
#SBATCH --time=12:00:00
#SBATCH --mem=32G         # 32GB RAM per job (safer for Llama 3)
#SBATCH --account=m25146
#SBATCH --job-name=dpo_sweep
#SBATCH --output=logs/%x_%A_%a.out  # %A=JobArrayID, %a=TaskID
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-15      # Run 16 jobs, indexed 0 to 15

export PYTHONUNBUFFERED=1

# Activate venv (create if not exists)
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    source venv/bin/activate
fi

# Always check/install dependencies
pip install -r requirements.txt

# Array of Beta values
BETAS=(0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Get Beta for this specific job task
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

MODEL="mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_BASE="./output_sweep"
OUTPUT_DIR="${OUTPUT_BASE}/beta_${BETA}"

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Beta: $BETA"
echo "Output: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR

# Run Training
# Note: No need for CUDA_VISIBLE_DEVICES loop or background & 
# Slurm assigns a unique GPU to this job automatically.

# Use offline mode for W&B to avoid login errors
export WANDB_MODE=offline

# Force usage of the venv's accelerate to avoid system path issues
./venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 train_dpo.py \
    --model_name $MODEL \
    --beta $BETA \
    --output_dir $OUTPUT_DIR \
    --batch_size 2 \
    --grad_accum 8 \
    --epochs 1 \
    --wandb_project "dpo-sweep-llama3-8b" \
    > "${OUTPUT_DIR}/train.log" 2>&1

echo "Task $SLURM_ARRAY_TASK_ID (Beta=$BETA) Completed!"
