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

# Custom imports
from dpo_loss import DPOLoss, get_batch_logps
from data_loader import get_dpo_dataset, DPODataCollator

def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name (meta-llama/Meta-Llama-3-8B-Instruct, gpt2)")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)

    # Setup Logging
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file_path = os.path.join(args.output_dir, "training_log.jsonl")
        print(f"Training DPO on {args.model_name}")
        print(f"Logging to {log_file_path}")
        
        # Log configuration
        config = vars(args)
        with open(log_file_path, "a") as f:
            f.write(json.dumps({"type": "config", "data": config}) + "\n")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Models
    print("Loading models...")
    # Policy model (we optimize this one)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="eager"
    )
    policy_model.gradient_checkpointing_enable()

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="eager"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    # Dataset & Dataloader
    print("Preparing data...")
    dataset = get_dpo_dataset(tokenizer, dataset_name=args.dataset_name, max_length=args.max_length)
    collator = DPODataCollator(tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=4
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=args.lr)

    # Prepare with accelerator
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
    ref_model = ref_model.to(accelerator.device)

    # Loss function
    dpo_loss = DPOLoss(beta=args.beta)

    # Training setup
    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    # Loop
    print("Starting training loop...")
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(args.epochs):
        policy_model.train()
        
        for batch in dataloader:
            with accelerator.accumulate(policy_model):
                # Concatenate chosen and rejected for efficiency
                len_chosen = batch["policy_chosen_input_ids"].shape[0]
                
                input_ids = torch.cat([batch["policy_chosen_input_ids"], batch["policy_rejected_input_ids"]], dim=0)
                mask = torch.cat([batch["policy_chosen_attention_mask"], batch["policy_rejected_attention_mask"]], dim=0)
                labels = torch.cat([batch["policy_chosen_labels"], batch["policy_rejected_labels"]], dim=0)

                # Forward Policy
                policy_out = policy_model(input_ids=input_ids, attention_mask=mask)
                policy_logps = get_batch_logps(policy_out.logits, labels)
                
                policy_chosen_logps = policy_logps[:len_chosen]
                policy_rejected_logps = policy_logps[len_chosen:]

                # Forward Reference (no grad)
                with torch.no_grad():
                    ref_out = ref_model(input_ids=input_ids, attention_mask=mask)
                    ref_logps = get_batch_logps(ref_out.logits, labels)
                    
                    ref_chosen_logps = ref_logps[:len_chosen]
                    ref_rejected_logps = ref_logps[len_chosen:]

                # Calculate Loss
                loss, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                
                loss = loss.mean()
                accelerator.backward(loss)
                
                # Clip grads
                accelerator.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    # Local logging
                    log_data = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "reward_chosen": chosen_rewards.mean().item(),
                        "reward_rejected": rejected_rewards.mean().item(),
                        "margin": (chosen_rewards - rejected_rewards).mean().item(),
                        "timestamp": time.time()
                    }
                    
                    with open(log_file_path, "a") as f:
                        f.write(json.dumps({"type": "step", "data": log_data}) + "\n")
                        
                    # Optional: Print to console every N steps
                    if global_step % 10 == 0:
                        print(f"Step {global_step}: Loss={log_data['loss']:.4f}, Margin={log_data['margin']:.4f}")

            if global_step >= total_steps:
                break

    # Save
    print("Saving model...")
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(policy_model)
    unwrapped.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        print("Training finished!")

if __name__ == "__main__":
    main()
