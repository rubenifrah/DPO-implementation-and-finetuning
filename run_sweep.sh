#!/bin/bash

# Array of Beta values to sweep
BETAS=(0.01 0.05 0.1 0.2 0.3 0.5 0.7 1.0)

# Base model
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Output directory base
OUTPUT_BASE="./output_sweep"

echo "Starting DPO Sweep on 8 GPUs..."

for i in {0..7}; do
    BETA=${BETAS[$i]}
    GPU_ID=$i
    
    echo "Launching run on GPU $GPU_ID with Beta=$BETA"
    
    # Create specific output directory
    OUTPUT_DIR="${OUTPUT_BASE}/beta_${BETA}"
    mkdir -p $OUTPUT_DIR
    
    # Launch in background
    # We use CUDA_VISIBLE_DEVICES to isolate the GPU for each process
    # and accelerate launch with num_processes=1 since we are running independent jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch --num_processes 1 --mixed_precision bf16 train_dpo.py \
        --model_name_or_path $MODEL \
        --beta $BETA \
        --output_dir $OUTPUT_DIR \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --num_epochs 1 \
        --wandb_project "dpo-sweep-llama3-8b" \
        > "${OUTPUT_DIR}/train.log" 2>&1 &
        
    # Small sleep to avoid race conditions in file creation/wandb init
    sleep 10
done

echo "All 8 runs launched! Check logs in ${OUTPUT_BASE}"
