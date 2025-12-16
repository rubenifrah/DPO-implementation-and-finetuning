import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

def get_judge_messages(prompt, response_a, response_b):
    """
    Constructs the prompt for the Judge LLM.
    
    We ask the judge to act as an impartial evaluator.
    The prompt is structured to present the User's query followed by two candidate responses.
    """
    system_prompt = "You are a helpful assistant. You will be given a user prompt and two responses: Response A and Response B. Your task is to evaluate which response is better based on helpfulness, clarity, and accuracy. Output 'A' if Response A is better, 'B' if Response B is better, or 'Tie' if they are of equal quality."
    
    user_content = f"""User Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Output only the single letter without any comment: "A", "B", or "Tie"."""

    # Return standard chat format (list of dicts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

def main():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation Script")
    parser.add_argument("--baseline_file", type=str, required=True, help="JSONL file containing responses from the Baseline model (e.g., SFT).")
    parser.add_argument("--candidate_file", type=str, required=True, help="JSONL file containing responses from the Candidate model (e.g., DPO).")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="The impartial judge model (should be strong, e.g., Llama-3 or GPT-4).")
    parser.add_argument("--output_file", type=str, default="judge_results.json", help="Where to save the win-rate analysis.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    print("Loading response files...")
    with open(args.baseline_file, "r") as f:
        baseline_data = [json.loads(line) for line in f]
    
    with open(args.candidate_file, "r") as f:
        candidate_data = [json.loads(line) for line in f]
    
    # Sanity check: comparisons must be apples-to-apples
    assert len(baseline_data) == len(candidate_data), "Error: Files must have the same number of samples to compare!"
    
    # -------------------------------------------------------------------------
    # 3. Load Judge Model
    # -------------------------------------------------------------------------
    print(f"Loading Judge Model: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    # Metrics
    wins = 0      # Candidate wins
    losses = 0    # Candidate loses (Baseline wins)
    ties = 0
    results = []

    print("Starting evaluation...")
    
    # -------------------------------------------------------------------------
    # 4. Evaluation Loop
    # -------------------------------------------------------------------------
    for i in tqdm(range(len(baseline_data))):
        prompt = baseline_data[i]["prompt"]
        resp_a = baseline_data[i]["generated_text"] # Originally Baseline
        resp_b = candidate_data[i]["generated_text"] # Originally Candidate
        
        # --- Mitigate Position Bias ---
        # LLMs often have a "bias towards the first option" (or sometimes the second).
        # We flip a coin. 
        # If swapped=True: Response A presented to judge is Candidate (resp_b), Response B is Baseline (resp_a).
        if random.random() > 0.5:
            swapped = True
            messages = get_judge_messages(prompt, resp_b, resp_a)
        else:
            swapped = False
            messages = get_judge_messages(prompt, resp_a, resp_b)
            
        # Apply Chat Template
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Validation generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # DETERMINISTIC: We want the judge to be consistent (Temperature=0)
                temperature=0.0
            )
            
        # Parse output
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        judgment = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Heuristic to find the verdict
        if "A" in judgment and "B" not in judgment:
            winner = "A"
        elif "B" in judgment and "A" not in judgment:
            winner = "B"
        else:
            winner = "Tie" # If judge is confused or outputs both
            
        # --- Map back to real identity ---
        if swapped:
            # If swapped, "A" was actually Candidate, "B" was Baseline
            if winner == "A": final_winner = "Candidate" 
            elif winner == "B": final_winner = "Baseline"
            else: final_winner = "Tie"
        else:
            # Standard order
            if winner == "A": final_winner = "Baseline"
            elif winner == "B": final_winner = "Candidate"
            else: final_winner = "Tie"
            
        # Update Stats
        if final_winner == "Candidate":
            wins += 1
        elif final_winner == "Baseline":
            losses += 1
        else:
            ties += 1
            
        results.append({
            "prompt": prompt,
            "baseline_response": resp_a,
            "candidate_response": resp_b,
            "winner": final_winner,
            "raw_judge_output": judgment
        })

    # -------------------------------------------------------------------------
    # 5. Final Report
    # -------------------------------------------------------------------------
    total = wins + losses + ties
    win_rate = wins / total if total > 0 else 0
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Comparisons: {total}")
    print(f"Candidate Wins:    {wins}")
    print(f"Baseline Wins:     {losses}")
    print(f"Ties:              {ties}")
    print("-" * 40)
    print(f"Win Rate:          {win_rate:.2%}")
    print("="*40 + "\n")
    
    # Save detailed logs
    with open(args.output_file, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": win_rate
            },
            "details": results
        }, f, indent=2)
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
