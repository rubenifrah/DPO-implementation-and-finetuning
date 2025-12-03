# Relevance and number of parameters of causal language models for DPO

This report lists relevant models supported by `AutoTokenizer` and `AutoModelForCausalLM` in the Hugging Face `transformers` library.

## 1. SOTA "small" models (7B - 9B)
These models are the current standard for experimentation and efficient deployment.

| Model | Model ID (use this in [train_dpo.py](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/train_dpo.py)) | Parameters | Context | Specificities |
| :--- | :--- | :--- | :--- | :--- |
| **Meta Llama 3 8B** | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | 8k | Standard architecture, GQA, excellent reasoning. |
| **Mistral 7B v0.3** | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 32k | Sliding Window Attention, GQA. Very efficient. |
| **Google Gemma 7B** | `google/gemma-7b-it` | 7B | 8k | RoPE embeddings, GeGLU. |
| **Qwen2 7B** | `Qwen/Qwen2-7B-Instruct` | 7B | 32k | GQA, SwiGLU. Exceptional coding capabilities. |

## 2. Efficient models (< 4B)
Ideal for rapid debugging, local testing, or edge deployment.

| Model | Model ID (use this in [train_dpo.py](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/train_dpo.py)) | Parameters | Context | Specificities |
| :--- | :--- | :--- | :--- | :--- |
| **Microsoft Phi-3 Mini** | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 4k | Trained on "textbook quality" data. |
| **StableLM 2 1.6B** | `stabilityai/stablelm-2-1_6b-chat` | 1.6B | 4k | Very small, fast. |
| **GPT-2** | `gpt2` | 124M | 1k | **Legacy**. Use only for pipeline verification. |

## 3. Large models (30B - 70B+)
With 8x A100s, you can fine-tune these using FSDP.

| Model | Model ID (use this in [train_dpo.py](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/train_dpo.py)) | Parameters | Context | Specificities |
| :--- | :--- | :--- | :--- | :--- |
| **Meta Llama 3 70B** | `meta-llama/Meta-Llama-3-70B-Instruct` | 70B | 8k | SOTA open-weights model. |
| **Mixtral 8x7B** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B | 32k | **MoE**. Fast and high quality. |
| **Qwen2 72B** | `Qwen/Qwen2-72B-Instruct` | 72B | 32k | Massive dense model. |

## 4. Specialized architectures

| Model | Model ID (use this in [train_dpo.py](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/train_dpo.py)) | Parameters | Specificities |
| :--- | :--- | :--- | :--- |
| **Yi 34B** | `01-ai/Yi-34B-Chat` | 34B | 200k context versions available. |
| **Command R** | `CohereForAI/c4ai-command-r-v01` | 35B | Optimized for RAG and Tool Use. |

### Summary

1.  **For development/debugging**: **`gpt2`** (instant feedback) or **`microsoft/Phi-3-mini-4k-instruct`** (fast, decent quality).
2.  **For standard DPO experiments**: **`meta-llama/Meta-Llama-3-8B-Instruct`** or **`mistralai/Mistral-7B-Instruct-v0.3`**.
3.  **For high performance runs**: **`meta-llama/Meta-Llama-3-70B-Instruct`**. With 8x A100s, you can train this efficiently using FSDP.

### How to use in [train_dpo.py](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/train_dpo.py)
Simply pass the Hugging Face ID to the `--model_name` argument:

```bash
# Debug
python train_dpo.py --model_name gpt2 ...

# Standard experiment
python train_dpo.py --model_name meta-llama/Meta-Llama-3-8B-Instruct ...

# High performance (requires multi-gpu setup)
accelerate launch train_dpo.py --model_name meta-llama/Meta-Llama-3-70B-Instruct ...
```
