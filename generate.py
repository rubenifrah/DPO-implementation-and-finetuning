import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import os

def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Response Generation Script for Evaluation")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model (e.g., ./output_dpo) or HuggingFace ID")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="Dataset containing the prompts")
    parser.add_argument("--split", type=str, default="test_prefs", help="Dataset split to evaluate on (e.g., test_prefs)")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the JSONL results")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum length of the generated response")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (useful for debugging)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 2. Model Loading
    # -------------------------------------------------------------------------
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Ensure pad token exists (Llama/Mistral usually rely on EOS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" # Automatically disperses model across available GPUs
    )
    model.eval() # Set to evaluation mode (disables dropout, etc.)

    # -------------------------------------------------------------------------
    # 3. Dataset Preparation
    # -------------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # Optional: Debug with a smaller subset
    if args.limit:
        print(f"LIMITING to {args.limit} samples.")
        dataset = dataset.select(range(args.limit))

    print(f"Generating responses for {len(dataset)} prompts...")
    
    results = []
    prompts = dataset["prompt"]
    
    # -------------------------------------------------------------------------
    # 4. Generation Loop
    # -------------------------------------------------------------------------
    # We iterate in batches for efficiency
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        
        # --- Prompt Formatting ---
        # Models expect a specific chat structure (e.g., [INST] Prompt [/INST]).
        # We use the tokenizer's chat template to apply this correct formatting.
        formatted_prompts = []
        for p in batch_prompts:
            # If the prompt is a simple string, we wrap it in a "user" message object
            if isinstance(p, str):
                messages = [{"role": "user", "content": p}]
            else:
                # If it's already a list of messages
                messages = p
            
            try:
                # Apply template (tokenize=False gives us the raw string with special tokens)
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                # Fallback purely for robustness
                if isinstance(messages, list):
                     text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
                else:
                    text = messages
            formatted_prompts.append(text)

        # --- Tokenization ---
        # Convert formatted strings to input tensors
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # --- Inference ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,   # We typically want some variety, or False for greedy decoding
                temperature=0.7,  # Controls randomness (higher = more creative/random)
                pad_token_id=tokenizer.pad_token_id
            )
        
        # --- Decoding ---
        # Convert output tokens back to text
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Store results
        for prompt, full_text in zip(batch_prompts, generated_texts):
            # In a real pipeline, we might strip the input prompt from the output here.
            # For now, we save the full text to be safe.
            results.append({
                "prompt": prompt,
                "generated_text": full_text
            })

    # -------------------------------------------------------------------------
    # 5. Saving Results
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"Success! Saved {len(results)} generations to {args.output_file}")

if __name__ == "__main__":
    main()
