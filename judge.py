import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

def get_judge_prompt(prompt, response_a, response_b):
    return f"""[INST] You are a helpful assistant. You will be given a user prompt and two responses: Response A and Response B.
Your task is to evaluate which response is better based on helpfulness, clarity, and accuracy.
Output "A" if Response A is better, "B" if Response B is better, or "Tie" if they are of equal quality.

User Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Output only the single letter "A", "B", or "Tie". [/INST]
"""

def main():
    parser = argparse.ArgumentParser(description="Judge two sets of responses")
    parser.add_argument("--baseline_file", type=str, required=True, help="JSONL file for baseline model (e.g., SFT)")
    parser.add_argument("--candidate_file", type=str, required=True, help="JSONL file for candidate model (e.g., DPO)")
    parser.add_argument("--judge_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model to use as judge")
    parser.add_argument("--output_file", type=str, default="judge_results.json", help="Output results file")
    args = parser.parse_args()

    # Load data
    with open(args.baseline_file, "r") as f:
        baseline_data = [json.loads(line) for line in f]
    
    with open(args.candidate_file, "r") as f:
        candidate_data = [json.loads(line) for line in f]
    
    # Ensure alignment
    assert len(baseline_data) == len(candidate_data), "Files must have same number of samples"
    
    # Load Judge
    print(f"Loading Judge Model: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    wins = 0
    losses = 0
    ties = 0
    
    results = []

    print("Judging responses...")
    for i in tqdm(range(len(baseline_data))):
        prompt = baseline_data[i]["prompt"]
        resp_a = baseline_data[i]["generated_text"]
        resp_b = candidate_data[i]["generated_text"]
        
        # Randomize order to avoid position bias
        if random.random() > 0.5:
            # Swap
            swapped = True
            input_text = get_judge_prompt(prompt, resp_b, resp_a)
        else:
            swapped = False
            input_text = get_judge_prompt(prompt, resp_a, resp_b)
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            
        judgment = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract last part
        judgment = judgment.split("[/INST]")[-1].strip()
        
        # Simple parsing
        if "A" in judgment and "B" not in judgment:
            winner = "A"
        elif "B" in judgment and "A" not in judgment:
            winner = "B"
        else:
            winner = "Tie"
            
        # Map back if swapped
        if swapped:
            if winner == "A": final_winner = "Candidate" # B was A
            elif winner == "B": final_winner = "Baseline" # A was B
            else: final_winner = "Tie"
        else:
            if winner == "A": final_winner = "Baseline"
            elif winner == "B": final_winner = "Candidate"
            else: final_winner = "Tie"
            
        if final_winner == "Candidate":
            wins += 1
        elif final_winner == "Baseline":
            losses += 1
        else:
            ties += 1
            
        results.append({
            "prompt": prompt,
            "baseline": resp_a,
            "candidate": resp_b,
            "winner": final_winner,
            "raw_judgment": judgment
        })

    # Stats
    total = wins + losses + ties
    win_rate = wins / total if total > 0 else 0
    
    print(f"Total: {total}")
    print(f"Wins (Candidate): {wins}")
    print(f"Losses (Baseline): {losses}")
    print(f"Ties: {ties}")
    print(f"Win Rate: {win_rate:.2%}")
    
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

if __name__ == "__main__":
    main()
