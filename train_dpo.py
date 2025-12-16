import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import json
import time

# Custom imports - our refactored modules
from dpo_loss import DPOLoss, get_batch_logps
from data_loader import get_dpo_dataset, DPODataCollator

def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) Training Script")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="The pretrained model to align (e.g. Llama-3, Mistral, GPT-2).")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="HuggingFace dataset containing preferences.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to save the aligned model and logs.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta (lower = closer to reference).")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (usually very small for DPO).")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps for effective batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for the tokenizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set seeds for reproducibility across runs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 2. Setup Accelerator (Multi-GPU handling)
    # -------------------------------------------------------------------------
    # Accelerator handles device placement (CPU/GPU) and distributed training (DDP) automatically.
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)

    if accelerator.is_main_process:
        print(f"Starting DPO training for: {args.model_name}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup local JSONL logging - preferred over WandB for simple cluster jobs
        log_file_path = os.path.join(args.output_dir, "training_log.jsonl")
        print(f"Metrics will be logged to: {log_file_path}")
        
        # Write the run configuration first
        with open(log_file_path, "a") as f:
            f.write(json.dumps({"type": "config", "data": vars(args)}) + "\n")

    # -------------------------------------------------------------------------
    # 3. Load Tokenizer and Models
    # -------------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure we have a pad token (EOS acts as pad for Llama-family models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Policy Model (Optimized)...")
    # This is the model we will actually train.
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for stability on Ampere GPUs (A100)
        use_cache=False,            # Disable KV cache during training (saves memory)
        attn_implementation="eager" # Reliable attention implementation (avoids some FlashAttn quirks)
    )
    # Enable gradient checkpointing to save VRAM (trades compute for memory)
    policy_model.gradient_checkpointing_enable()

    print("Loading Reference Model (Frozen)...")
    # This acts as the "baseline". We want to improve upon this model without deviating too far.
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="eager"
    )
    ref_model.eval()               # Set to evaluation mode
    ref_model.requires_grad_(False) # Freeze weights completely

    # -------------------------------------------------------------------------
    # 4. Prepare Dataset
    # -------------------------------------------------------------------------
    print(f"Loading and processing dataset: {args.dataset_name}")
    dataset = get_dpo_dataset(tokenizer, dataset_name=args.dataset_name, max_length=args.max_length)
    collator = DPODataCollator(tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=4
    )

    # -------------------------------------------------------------------------
    # 5. Optimizer & Scheduling
    # -------------------------------------------------------------------------
    # RMSprop is often preferred for DPO/RLHF over AdamW, but AdamW works too.
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=args.lr)

    # Hand over everything to Accelerator to prepare for distributed training
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
    
    # Move ref model to the correct device manually (accelerator doesn't prepare frozen models automatically)
    ref_model = ref_model.to(accelerator.device)

    # Initialize our custom DPO Loss module
    dpo_loss_func = DPOLoss(beta=args.beta)

    # Scheduler setup
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps), # 10% warmup
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    print(f"Starting training loop for {args.epochs} epochs ({total_steps} steps)...")
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(args.epochs):
        policy_model.train()
        
        for batch in dataloader:
            # Gradient Accumulation Context
            with accelerator.accumulate(policy_model):
                
                # --- Efficiency Trick: Concatenation ---
                # We have 'chosen' samples and 'rejected' samples.
                # Instead of doing two separate forward passes (one for chosen, one for rejected),
                # we concatenate them into a single batch of size (2 * batch_size).
                # This is faster on GPUs due to better parallelism.
                len_chosen = batch["policy_chosen_input_ids"].shape[0]
                
                all_input_ids = torch.cat([batch["policy_chosen_input_ids"], batch["policy_rejected_input_ids"]], dim=0)
                all_mask = torch.cat([batch["policy_chosen_attention_mask"], batch["policy_rejected_attention_mask"]], dim=0)
                all_labels = torch.cat([batch["policy_chosen_labels"], batch["policy_rejected_labels"]], dim=0)

                # --- Policy Model Forward Pass ---
                policy_out = policy_model(input_ids=all_input_ids, attention_mask=all_mask)
                
                # Calculate log probabilities for every token
                policy_logps_all = get_batch_logps(policy_out.logits, all_labels)
                
                # Split back into chosen/rejected
                policy_chosen_logps = policy_logps_all[:len_chosen]
                policy_rejected_logps = policy_logps_all[len_chosen:]

                # --- Reference Model Forward Pass ---
                # We don't need gradients for the reference model, so we iterate inside no_grad()
                with torch.no_grad():
                    ref_out = ref_model(input_ids=all_input_ids, attention_mask=all_mask)
                    ref_logps_all = get_batch_logps(ref_out.logits, all_labels)
                    
                    ref_chosen_logps = ref_logps_all[:len_chosen]
                    ref_rejected_logps = ref_logps_all[len_chosen:]

                # --- Compute DPO Loss ---
                loss_elements, chosen_rewards, rejected_rewards = dpo_loss_func(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                
                loss = loss_elements.mean()
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient Clipping (prevents exploding gradients)
                accelerator.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                # Optimizer Step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # --- Logging (only on main process) ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    # Calculate metrics for logging
                    current_margin = (chosen_rewards - rejected_rewards).mean().item()
                    current_loss = loss.item()
                    
                    log_data = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": current_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "reward_chosen": chosen_rewards.mean().item(),
                        "reward_rejected": rejected_rewards.mean().item(),
                        "margin": current_margin,
                        "timestamp": time.time()
                    }
                    
                    # Append to local log file
                    with open(log_file_path, "a") as f:
                        f.write(json.dumps({"type": "step", "data": log_data}) + "\n")
                        
                    # Console print for liveness
                    if global_step % 10 == 0:
                        print(f"[Step {global_step}] Loss: {current_loss:.4f} | Margin: {current_margin:.4f}")

            if global_step >= total_steps:
                break

    # -------------------------------------------------------------------------
    # 7. Saving the Model
    # -------------------------------------------------------------------------
    print("Training complete. Saving model...")
    accelerator.wait_for_everyone()
    
    # Unwrap model from Accelerator/DDP wrapper to save it cleanly
    unwrapped_model = accelerator.unwrap_model(policy_model)
    unwrapped_model.save_pretrained(
        args.output_dir, 
        is_main_process=accelerator.is_main_process, 
        save_function=accelerator.save
    )
    
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
