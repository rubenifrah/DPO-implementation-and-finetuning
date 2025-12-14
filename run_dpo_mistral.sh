#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 4              # 4 CPUs per job
#SBATCH --gres=gpu:1      # 1 GPU per job
#SBATCH --time=24:00:00   # Increased time for larger model
#SBATCH --mem=64G         # Increased RAM for 7B model
#SBATCH --account=m25146
#SBATCH --job-name=dpo_mistral
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

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

# Configuration
BETAS=(0.01 0.03 0.05 0.1 0.2)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR="./output_mistral_beta_${BETA}"

echo "Starting DPO Training"
echo "Model: $MODEL"
echo "Beta: $BETA"
echo "Output: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR    

# Run Training
# Using gradient accumulation to handle 7B model on single GPU if needed
# Batch size 1 or 2 depending on GPU VRAM (A100 40GB/80GB should handle 2-4)

./venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 train_dpo.py \
    --model_name $MODEL \
    --beta $BETA \
    --output_dir $OUTPUT_DIR \
    --batch_size 2 \
    --grad_accum 16 \
    --epochs 1 \
    > "${OUTPUT_DIR}/train.log" 2>&1

echo "Training Completed!"
