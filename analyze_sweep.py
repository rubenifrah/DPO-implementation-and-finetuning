import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_sweep(base_dir="."):
    """
    Analyzes the results of the Hyperparameter Sweep (Beta optimization) for GPT-2.
    
    """
    
    # Locate all training logs from the sweep
    log_files = glob.glob(os.path.join(base_dir, "GPT2/beta_*/training_log.jsonl"))
    
    results = {}
    print(f"Found {len(log_files)} log files from the sweep.")

    #only keep one file every 2 betas
    log_files = log_files[::2]
    print(f"Only keeping {len(log_files)} log files.")
    
    for log_file in log_files:
        # Infer Beta value from the directory name (e.g. "./beta_0.1/training_log.jsonl")
        dirname = os.path.dirname(log_file)
        beta_str = os.path.basename(dirname).replace("beta_", "")
        try:
            beta = float(beta_str)
        except ValueError:
            print(f"Skipping {dirname}, could not parse beta.")
            continue
            
        steps = []
        margins = []
        losses = []
        rewards_chosen = []
        rewards_rejected = []
        
        # Read the log file
        with open(log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry["type"] == "step":
                        data = entry["data"]
                        steps.append(data["step"])
                        margins.append(data["margin"])
                        losses.append(data["loss"])
                        rewards_chosen.append(data["reward_chosen"])
                        rewards_rejected.append(data["reward_rejected"])
                except:
                    continue
        
        if not steps:
            print(f"Warning: No steps found in {log_file}")
            continue
            
        # Store results
        results[beta] = {
            "steps": steps,
            "margins": margins,
            "losses": losses,
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
            # Calculate final margin (average of last 10 steps for stability)
            "final_margin": np.mean(margins[-10:]) if len(margins) >= 10 else margins[-1]
        }

    # Sort betas for cleaner plotting
    sorted_betas = sorted(results.keys())
    
    # summary table
    print("\n" + "="*40)
    print(f"{'Beta':<10} | {'Final Margin':<15} | {'Final Loss':<15}")
    print("-" * 40)
    for beta in sorted_betas:
        data = results[beta]
        final_margin = data["margins"][-1]
        final_loss = data["losses"][-1]
        print(f"{beta:<10.2f} | {final_margin:<15.4f} | {final_loss:<15.4f}")
    print("="*40 + "\n")

    # plotting
    
    def smooth(scalars, weight=0.9):
        """
        Simple Exponential Moving Average (EMA) for smoother plots.
        """
        if not scalars: return []
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.figure(figsize=(15, 10))
    
    # margin plot
    plt.subplot(2, 1, 1)
    for beta in sorted_betas:
        smoothed_margins = smooth(results[beta]["margins"], weight=0.95)
        plt.plot(results[beta]["steps"], smoothed_margins, label=f"Beta={beta}")
    plt.title("DPO Margin (Smoothed)")
    plt.xlabel("Steps")
    plt.ylabel("Margin")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # loss plot
    plt.subplot(2, 1, 2)
    for beta in sorted_betas:
        smoothed_losses = smooth(results[beta]["losses"], weight=0.95)
        plt.plot(results[beta]["steps"], smoothed_losses, label=f"Beta={beta}")
    plt.title("DPO Training Loss (Smoothed)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sweep_analysis_smoothed.png")
    print("Saved plot to sweep_analysis_smoothed.png")

if __name__ == "__main__":
    analyze_sweep()
