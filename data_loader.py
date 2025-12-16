import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import Dict, List

class DPODataCollator:
    """
    A custom Data Collator for DPO (Direct Preference Optimization).
    
    What this does:
    1. Takes a list of samples (each containing "chosen" and "rejected" input IDs).
    2. Figures out the maximum sequence length in the batch.
    3. Pads all sequences (chosen AND rejected) to that same length.
    4. Returns a dictionary of tensors ready for the model.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Padding Token Handling:
        # Many modern models (like Llama-2 or Mistral) don't have a specific PAD token by default.
        # So we often use the EOS (End of Sentence) token as a fallback for padding.
        if tokenizer.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        This function is called by the DataLoader to create a batch.
        """
        # Extract the raw lists of integers (input IDs) from the batch
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        
        # We also need the masks (to tell the model which tokens are real and which are padding)
        chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
        rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
        
        # And the labels (for loss calculation - though DPO handles this inside the loss function too)
        chosen_labels = [item["chosen_labels"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch]

        # Determine the Global Maximum Length
        # To batch efficiently, both the "Chosen" and "Rejected" responses in a batch 
        # should ideally be padded to the same length so we can stack them nicely.
        # We find the longest sequence in the entire batch (across both chosen and rejected)
        max_len_chosen = max(len(x) for x in chosen_input_ids)
        max_len_rejected = max(len(x) for x in rejected_input_ids)
        
        global_max_len = max(max_len_chosen, max_len_rejected)
        
        # But we cap it at our hard limit (e.g., 2048) to avoid OOM
        global_max_len = min(global_max_len, self.max_length)

        # Pad everything to this calculated length
        batch_chosen = self._pad_batch(chosen_input_ids, chosen_attention_mask, chosen_labels, global_max_len)
        batch_rejected = self._pad_batch(rejected_input_ids, rejected_attention_mask, rejected_labels, global_max_len)

        # Combine into a single dictionary
        return {
            "policy_chosen_input_ids": batch_chosen["input_ids"],
            "policy_chosen_attention_mask": batch_chosen["attention_mask"],
            "policy_chosen_labels": batch_chosen["labels"],
            
            "policy_rejected_input_ids": batch_rejected["input_ids"],
            "policy_rejected_attention_mask": batch_rejected["attention_mask"],
            "policy_rejected_labels": batch_rejected["labels"],
        }

    def _pad_batch(self, input_ids, attention_mask, labels, max_len):
        """
        Helper function to actually perform the padding.
        """
        padded_ids = []
        padded_mask = []
        padded_labels = []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            # 1. Truncate if too long (just in case)
            ids = ids[:max_len]
            mask = mask[:max_len]
            lab = lab[:max_len]

            # 2. Calculate how much padding is needed
            pad_len = max_len - len(ids)
            
            # 3. Apply Padding
            # usage: [original_seq] + [PAD, PAD, PAD...]
            # Note: We use -100 for labels padding so 'CrossEntropyLoss' ignores them automatically.
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)

        # Convert simple python lists to PyTorch Tensors
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

def get_dpo_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "train_prefs",
    max_length: int = 1024,
):
    """
    Downloads and pre-processes the preference dataset.
    Ideally, we want it in a format like:
    
    Prompt: "How do I make a cake?"
    Chosen: "Here is a recipe..."
    Rejected: "Go buy one."
    
    This function handles the crucial step of applying the 'Chat Template'
    so that the model sees the special tokens (like <|user|>, <|assistant|>) it expects.
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)

    def process(examples):
        new_examples = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
        }
        
        # Iterate over the batch of raw examples
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            
            # Standardization: Ensure everything is a list of dictionaries (standard OpenAI format)
            if not isinstance(chosen, list):
                chosen = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
            if not isinstance(rejected, list):
                rejected = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
            
            # Apply the Tokenizer's Chat Template
            # This converts the list of dicts into a single string with correct special tokens.
            # E.g. [INST] prompt [/INST] response
            try:
                chosen_text = tokenizer.apply_chat_template(chosen, tokenize=False)
                rejected_text = tokenizer.apply_chat_template(rejected, tokenize=False)
            except:
                # Emergency fallback if template fails
                chosen_text = f"User: {prompt}\nAssistant: {chosen}"
                rejected_text = f"User: {prompt}\nAssistant: {rejected}"

            # Tokenize: Convert text string to Integers
            tokenized_chosen = tokenizer(chosen_text, truncation=True, max_length=max_length)
            tokenized_rejected = tokenizer(rejected_text, truncation=True, max_length=max_length)
            
            # Store the results
            new_examples["chosen_input_ids"].append(tokenized_chosen["input_ids"])
            new_examples["chosen_attention_mask"].append(tokenized_chosen["attention_mask"])
            new_examples["chosen_labels"].append(tokenized_chosen["input_ids"]) 
            
            new_examples["rejected_input_ids"].append(tokenized_rejected["input_ids"])
            new_examples["rejected_attention_mask"].append(tokenized_rejected["attention_mask"])
            new_examples["rejected_labels"].append(tokenized_rejected["input_ids"])

        return new_examples

    # Map the process function over the whole dataset in batches
    dataset = dataset.map(process, batched=True, remove_columns=dataset.column_names)
    return dataset

if __name__ == "__main__":
    # Simple sanity check to ensure the component loads correctly
    from transformers import AutoTokenizer
    print("Testing DPODataCollator...")
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    c = DPODataCollator(t)
    print("Collator initialization sequence complete.")
