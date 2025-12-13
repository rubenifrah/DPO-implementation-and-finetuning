# The DPO Cluster Deployment Adventure 🚀

A summary of our journey to get Direct Preference Optimization (DPO) running on the Mesonet cluster with 16 concurrent GPUs.

## 🎯 Objective
Run a hyperparameter sweep for the Beta parameter (0.01 to 1.0) using 16 A100 GPUs, training a 7B model on the UltraFeedback dataset.

## 🛠️ The Troubleshooting Log

### 1. The "Instant Success" Illusion
*   **Issue**: The Slurm job finished in 10 seconds and said "Completed!", but obviously didn't train anything.
*   **Cause**: The script was using the macOS `venv` copied via `scp`. Linux cannot run macOS binaries.
*   **Fix**: Deleted the remote `venv` and updated [run_sweep.sh](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/run_sweep.sh) to automatically create a fresh Linux-compatible environment.

### 2. The Silent Failure
*   **Issue**: The job still finished instantly, but now silently.
*   **Cause**: The script didn't have `set -e`, so it ignored errors and kept going.
*   **Fix**: Added `set -e` to fail fast and reveal the true error.

### 3. The Missing API Key
*   **Issue**: `wandb: No API key configured`.
*   **Cause**: W&B tries to sync to the cloud by default, but we weren't logged in on the server.
*   **Fix**: Set `export WANDB_MODE=offline` in the script to save logs locally.

### 4. The Path Confusion
*   **Issue**: `accelerate: command not found` (or using the wrong system version).
*   **Cause**: The script was calling `accelerate` directly, which might not be the one in our venv.
*   **Fix**: Changed the command to explicitly use `./venv/bin/accelerate`.

### 5. The Gated Model Block
*   **Issue**: `403 Client Error: Forbidden` for `meta-llama/Meta-Llama-3-8B-Instruct`.
*   **Cause**: Llama 3 is a gated model requiring license acceptance on Hugging Face.
*   **Fix**: Switched to **`mistralai/Mistral-7B-Instruct-v0.3`**, which is open and ungated.

### 6. The Missing Tokenizer Dependencies
*   **Issue**: `ValueError: ... make sure you have sentencepiece installed`.
*   **Cause**: Mistral's tokenizer relies on `sentencepiece` and `protobuf`, which were missing from [requirements.txt](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/requirements.txt).
*   **Fix**: Added them to [requirements.txt](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/requirements.txt) and re-uploaded.

### 7. The Flash Attention Hurdle
*   **Issue**: `ImportError: ... package flash_attn seems to be not installed`.
*   **Cause**: `flash_attn` is complex to compile and missing on the server.
*   **Fix**: Modified [train_dpo.py](file:///Users/ifrahruben/Desktop/github/DPO-implementation-and-finetuning/train_dpo.py) to force `attn_implementation="eager"` (standard PyTorch attention), bypassing the dependency.

## ✅ Final Status
**Success!** The jobs are now running. 
- **Model**: Mistral 7B v0.3
- **Infrastructure**: 16 concurrent jobs via Slurm Array
- **Method**: DPO with Beta sweep

## 📊 How to Monitor

**1. Check Job Status**
```bash
squeue -u rubifrah
```
*(Look for `R` in the `ST` column)*

**2. Watch Training Progress (Real-time)**
Pick a beta value (e.g., 0.1) and tail the log:
```bash
tail -f output_sweep/beta_0.1/train.log
```

**3. Check Slurm System Logs**
If a job crashes, check the main log:
```bash
cat logs/*.err
```
