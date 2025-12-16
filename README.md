# Direct Preference Optimization (DPO) - Implementation and Finetuning 

This repository contains an implementation of Direct Preference Optimization (DPO) based on the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" by Rafael Rafailov and al., a method for aligning Large Language Models (LLMs) with human preferences without the need for a complex reward model and PPO pipeline.

The project explores the stability and performance of DPO across different model scales (GPT-2 and Mistral-7B) and investigates the crucial role of the $\beta$ hyperparameter.

The full report can be found in the `report.pdf` file, along with a detailed presentation in the `presentation_DPO.pdf` file.

---

## Goal
The primary objective of this project is to demystify DPO by building it from the ground up using PyTorch and the Hugging Face ecosystem, avoiding "black-box" trainers like TRL.

**Key research questions:**
1.  **Stability**: How does the $\beta$ parameter affect training stability on different model scales?
2.  **Scalability**: Can insights from small models (GPT-2) transfer to larger, capable models (Mistral-7B)?
3.  **Performance**: Can we achieve significant alignment improvements (measured by win-rate) with a simple, transparent implementation?

---

## Implementation framework

All code is written in pure PyTorch/Python for maximum transparency and control.

### Core Components
*   **Infrastructure**: Designed for High-Performance Computing (HPC) environments (Mesonet Juliet cluster).
*   **Logging**: Custom streaming logger that writes metrics (loss, chosen/rejected margins, accuracy) to local `.jsonl` files, enabling offline analysis without external API dependencies.
*   **Dataset**: Uses **[HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)**, a standard benchmark for preference alignment.

### Key scripts
*   `train_dpo.py`: Main training loop implementing the DPO loss.
*   `dpo_loss.py`: The custom PyTorch loss module ($ \mathcal{L}_{DPO} = -\log \sigma (\beta (\dots)) $).
*   `analyze_sweep.py` / `analyze_mistral.py`: Tools for visualizing training dynamics (margin, accuracy, KL-divergence).
*   `judge.py`: LLM-as-a-Judge evaluation pipeline using Meta-Llama-3.

---

## Experiments

### 1. The stability sweep (GPT-2)
**Goal**: Efficiently determine the stable range for $\beta$ using a small proxy model (GPT-2, 124M).
**Findings**:
*   GPT-2 was **extremely unstable** for $\beta \ge 0.1$.
*   High $\beta$ caused the preference margin to collapse to large negative values.
*   **Hypothesis**: Small models require very high regularization (low $\beta \approx 0.01$) to remain stable.

### 2. Target run (Mistral-7B)
**Goal**: Train a capable instruction-tuned model (`Mistral-7B-Instruct-v0.3`) and verify the small-model hypothesis.
**Findings**:
*   **Hypothesis disproven**: Unlike GPT-2, Mistral-7B was highly stable even at $\beta=0.1$ and $\beta=0.2$.
*   **Optimal config**: $\beta=0.1$ provided the best balance between stability and preference separation (86% accuracy).
*   **Takeaway**: Hyperparameters for alignment are highly scale-dependent.

---

## Results & Evaluation

To rigorously evaluate the model, we implemented an **LLM-as-a-Judge** pipeline using **Llama-3-8B-Instruct**.

### Methodology
*   **Judge**: Meta-Llama-3-8B-Instruct (Deterministic generation, $T=0$).
*   **Protocol**: Pairwise comparison of DPO vs. SFT responses on 200 held-out prompts.
*   **Bias Mitigation**: Randomized position swapping (50% flip rate) to counter "position bias".

### Win-Rate Analysis
| Outcome | Count |
| :--- | :--- |
| **DPO Wins** | **117** |
| SFT Wins | 82 |
| Ties | 1 |
| **Win Rate** | **58.5%** |

The DPO model ($\beta=0.1$) demonstrates a clear improvement over the SFT baseline, validating the effectiveness of our scratch implementation.

---

## How to run

### 1. Setup environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run DPO training (Mistral-7B)
```bash
accelerate launch train_dpo.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --beta 0.1 \
    --lr 5e-7 \
    --epochs 1
```

### 3. Generate responses
```bash
python generate.py \
    --model_name_or_path "./output_mistral_beta_0.1" \
    --output_file "eval_results.jsonl"
```

### 4. Run evaluation (LLM-as-a-Judge)
```bash
python judge.py \
    --baseline_file "generations_sft.jsonl" \
    --candidate_file "eval_results.jsonl" \
    --judge_model "meta-llama/Meta-Llama-3-8B-Instruct"
```

---
*Created by Rebecca El Chidiac and Ruben Ifrah.*
