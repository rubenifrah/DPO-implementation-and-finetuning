import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_mistral(base_dir="mistral_DPO"):
    """
    Analyzes the training logs from the Mistral DPO experiments.
    
    What this script does:
    1. Finds all log files (one per Beta value).
    2. Extracts key metrics: Margin, Loss, Accuracy, and Rewards.
    3. Plots them all in a nice 4-subplot figure.
    """
    
    # log loading
    log_files = glob.glob(os.path.join(base_dir, "beta_*.jsonl"))
    
    results = {}
    print(f"Found {len(log_files)} log files in {base_dir}")
    
    for log_file in log_files:
        # parse beta from filename (e.g. "beta_0.01.jsonl" -> 0.01)
        filename = os.path.basename(log_file)
        beta_str = filename.replace("beta_", "").replace(".jsonl", "")
        try:
            beta = float(beta_str)
        except ValueError:
            print(f"Skipping {filename}, cannot parse beta.")
            continue
            
        # Lists to store time-series data
        steps = []
        margins = []
        losses = []
        rewards_chosen = []
        rewards_rejected = []
        accuracies = []
        
        # Read the JSONL file line by line
        with open(log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # We only care about "step" entries, not "config"
                    if entry["type"] == "step":
                        data = entry["data"]
                        steps.append(data["step"])
                        margins.append(data["margin"])
                        losses.append(data["loss"])
                        rewards_chosen.append(data["reward_chosen"])
                        rewards_rejected.append(data["reward_rejected"])
                        
                        # Infer accuracy: DPO works if Reward(Chosen) > Reward(Rejected)
                        acc = 1.0 if data["reward_chosen"] > data["reward_rejected"] else 0.0
                        accuracies.append(acc)
                except:
                    continue
        
        if not steps:
            print(f"Warning: No steps found in {log_file}")
            continue
            
        # Store everything for this beta
        results[beta] = {
            "steps": steps,
            "margins": margins,
            "losses": losses,
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
            "accuracies": accuracies,
        }

    # sort betas so the legend is clean
    sorted_betas = sorted(results.keys())
    
    # helper for smoothing
    def smooth(scalars, weight=0.9):
        """
        Exponential Moving Average (EMA) to make plots less noisy.
        Input: [1, 2, 5, 4...]
        Output: Smoothed version
        """
        if not scalars: return []
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    #  summary table
    print("\n" + "="*80)
    print(f"{'Beta':<10} | {'Final Margin':<15} | {'Final Loss':<15} | {'Final Acc':<15}")
    print("-" * 80)
    for beta in sorted_betas:
        data = results[beta]
        final_margin = data["margins"][-1]
        final_loss = data["losses"][-1]
        # Average the last 50 steps for a stable accuracy reading
        final_acc = np.mean(data["accuracies"][-50:]) if len(data["accuracies"]) >= 50 else np.mean(data["accuracies"])
        print(f"{beta:<10.2f} | {final_margin:<15.4f} | {final_loss:<15.4f} | {final_acc:<15.4%}")
    print("="*80 + "\n")

    # visualization and figure creation
    plt.figure(figsize=(15, 16)) # Taller figure
    
    # margin plot
    plt.subplot(4, 1, 1)
    for beta in sorted_betas:
        smoothed_margins = smooth(results[beta]["margins"], weight=0.95)
        plt.plot(results[beta]["steps"], smoothed_margins, label=f"Beta={beta}")
    plt.title("Margin (Reward Chosen - Reward Rejected)")
    plt.ylabel("Margin")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # loss plot
    plt.subplot(4, 1, 2)
    for beta in sorted_betas:
        smoothed_losses = smooth(results[beta]["losses"], weight=0.95)
        plt.plot(results[beta]["steps"], smoothed_losses, label=f"Beta={beta}")
    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # accuracy plot
    plt.subplot(4, 1, 3)
    for beta in sorted_betas:
        # smoothing (0.99) for accuracy (may be noisy)
        smoothed_acc = smooth(results[beta]["accuracies"], weight=0.99) 
        plt.plot(results[beta]["steps"], smoothed_acc, label=f"Beta={beta}")
    plt.title("Training Accuracy (P(chosen > rejected))")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # kl divergence plot
    plt.subplot(4, 1, 4)
    for beta in sorted_betas:
        # In DPO, the 'Reward' is actually defines as: beta * (log(pi) - log(ref))
        # we derive the KL divergence from the reward by dividing by beta
        kl_divs = [r / beta for r in results[beta]["rewards_chosen"]]
        smoothed_kl = smooth(kl_divs, weight=0.95)
        plt.plot(results[beta]["steps"], smoothed_kl, label=f"Beta={beta}")
    plt.title("Approximate KL Divergence (Log Ratio of Chosen)")
    plt.xlabel("Steps")
    plt.ylabel("KL (nats)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mistral_analysis.png") # Main file
    print("Saved plot to mistral_analysis.png")

if __name__ == "__main__":
    analyze_mistral()
