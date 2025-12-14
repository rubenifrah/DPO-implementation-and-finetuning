import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Generate responses for evaluation")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model or HF model name")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="Dataset name")
    parser.add_argument("--split", type=str, default="test_prefs", help="Dataset split to use")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    if args.limit:
        dataset = dataset.select(range(args.limit))

    print(f"Generating responses for {len(dataset)} prompts...")
    
    results = []
    
    # Simple batching
    prompts = dataset["prompt"]
    
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        
        # Apply chat template if possible, otherwise raw prompt
        formatted_prompts = []
        for p in batch_prompts:
            # Check if prompt is a list of dicts (chat) or string
            if isinstance(p, str):
                # Wrap in user message
                messages = [{"role": "user", "content": p}]
            else:
                messages = p
            
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                # Fallback for models without chat template
                if isinstance(messages, list):
                     text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
                else:
                    text = messages
            formatted_prompts.append(text)

        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the new part (simple heuristic)
        for prompt, full_text in zip(batch_prompts, generated_texts):
            # Try to remove the prompt part if possible, but for now saving full text is safer
            # or we can try to find the prompt length
            
            results.append({
                "prompt": prompt,
                "generated_text": full_text
            })

    # Save
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(results)} generations to {args.output_file}")

if __name__ == "__main__":
    main()
