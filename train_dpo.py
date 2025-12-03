import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import numpy as np

# Import our custom modules
from dpo_loss import DPOLoss, get_batch_logps
from data_loader import get_dpo_dataset, DPODataCollator

def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="dpo-scratch", help="Wandb project name")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize WandB
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, config=args)

    accelerator.print(f"Loading model: {args.model_name_or_path}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Policy Model
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False, # Gradient checkpointing usually requires this off
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    
    # Enable gradient checkpointing for memory efficiency
    policy_model.gradient_checkpointing_enable()

    # Load Reference Model
    # For DPO, we need a reference model that is frozen.
    # To save memory, we can load it in 8-bit or 4-bit if needed, but for A100s bf16 is fine.
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load Dataset
    accelerator.print("Loading dataset...")
    train_dataset = get_dpo_dataset(tokenizer, dataset_name=args.dataset_name, max_length=args.max_length)
    
    # Data Collator
    collator = DPODataCollator(tokenizer, max_length=args.max_length)

    # Data Loader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=4
    )

    # Optimizer
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=args.learning_rate)

    # Prepare with Accelerator
    policy_model, optimizer, train_dataloader = accelerator.prepare(
        policy_model, optimizer, train_dataloader
    )
    
    # Reference model doesn't need to be prepared with optimizer, but should be on correct device
    ref_model = ref_model.to(accelerator.device)

    # Loss function
    dpo_loss_fn = DPOLoss(beta=args.beta)

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Training Loop
    accelerator.print("Starting training...")
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    global_step = 0
    for epoch in range(args.num_epochs):
        policy_model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(policy_model):
                # 1. Run Policy Model
                # We need logprobs for chosen and rejected
                # Concatenate chosen and rejected to run in one forward pass (efficiency)
                # batch keys: policy_chosen_input_ids, policy_rejected_input_ids, etc.
                
                # Concatenate inputs: [chosen, rejected]
                len_chosen = batch["policy_chosen_input_ids"].shape[0]
                
                all_input_ids = torch.cat([batch["policy_chosen_input_ids"], batch["policy_rejected_input_ids"]], dim=0)
                all_attention_mask = torch.cat([batch["policy_chosen_attention_mask"], batch["policy_rejected_attention_mask"]], dim=0)
                all_labels = torch.cat([batch["policy_chosen_labels"], batch["policy_rejected_labels"]], dim=0)

                # Policy Forward
                policy_outputs = policy_model(input_ids=all_input_ids, attention_mask=all_attention_mask)
                policy_logits = policy_outputs.logits
                
                # Compute Policy Logprobs
                policy_logps = get_batch_logps(policy_logits, all_labels, average_log_prob=False)
                policy_chosen_logps = policy_logps[:len_chosen]
                policy_rejected_logps = policy_logps[len_chosen:]

                # 2. Run Reference Model (No Grad)
                with torch.no_grad():
                    ref_outputs = ref_model(input_ids=all_input_ids, attention_mask=all_attention_mask)
                    ref_logits = ref_outputs.logits
                    
                    # Compute Reference Logprobs
                    ref_logps = get_batch_logps(ref_logits, all_labels, average_log_prob=False)
                    ref_chosen_logps = ref_logps[:len_chosen]
                    ref_rejected_logps = ref_logps[len_chosen:]

                # 3. Compute Loss
                loss, chosen_rewards, rejected_rewards = dpo_loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps
                )
                
                # Average loss over batch
                loss = loss.mean()

                # Backward
                accelerator.backward(loss)
                
                # Clip gradients
                accelerator.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if accelerator.is_main_process:
                    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
                    wandb.log({
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "reward_chosen": chosen_rewards.mean().item(),
                        "reward_rejected": rejected_rewards.mean().item(),
                        "reward_margin": reward_margin,
                        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item()
                    })

            if global_step >= max_train_steps:
                break

    # Save Model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(policy_model)
    unwrapped_model.save_pretrained(
        args.output_dir, 
        is_main_process=accelerator.is_main_process, 
        save_function=accelerator.save
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        wandb.finish()

if __name__ == "__main__":
    main()
