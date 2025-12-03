import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional
import copy

class DPODataCollator:
    """
    Data collator for DPO.
    Pads the chosen and rejected sequences to the maximum length in the batch.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            # Set pad token to eos token if not defined (common in Llama)
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
        rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
        chosen_labels = [item["chosen_labels"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch]

        # Pad sequences
        batch_chosen = self._pad_batch(chosen_input_ids, chosen_attention_mask, chosen_labels)
        batch_rejected = self._pad_batch(rejected_input_ids, rejected_attention_mask, rejected_labels)

        return {
            "policy_chosen_input_ids": batch_chosen["input_ids"],
            "policy_chosen_attention_mask": batch_chosen["attention_mask"],
            "policy_chosen_labels": batch_chosen["labels"],
            "policy_rejected_input_ids": batch_rejected["input_ids"],
            "policy_rejected_attention_mask": batch_rejected["attention_mask"],
            "policy_rejected_labels": batch_rejected["labels"],
        }

    def _pad_batch(self, input_ids: List[List[int]], attention_mask: List[List[int]], labels: List[List[int]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(ids) for ids in input_ids)
        # Clamp to max_length
        max_len = min(max_len, self.max_length)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            # Truncate if necessary
            ids = ids[:max_len]
            mask = mask[:max_len]
            lab = lab[:max_len]

            # Pad (left padding is often better for generation, but for training right padding is standard usually)
            # However, for causal LM, right padding is tricky with attention masks if not handled carefully.
            # We will use right padding here as it's simpler for standard training loops.
            pad_len = max_len - len(ids)
            
            padded_ids = ids + [self.pad_token_id] * pad_len
            padded_mask = mask + [0] * pad_len
            padded_lab = lab + [-100] * pad_len # -100 ignores loss

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            padded_labels.append(padded_lab)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

def get_dpo_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    split: str = "train_prefs",
    max_length: int = 1024,
) -> Dataset:
    """
    Loads and processes the DPO dataset.
    """
    dataset = load_dataset(dataset_name, split=split)

    def process_function(examples):
        new_examples = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
        }
        
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Format: Prompt + Response
            # Note: Ultrafeedback binarized usually has 'prompt', 'chosen' (list of dicts role/content), 'rejected'
            # But the 'HuggingFaceH4/ultrafeedback_binarized' has 'prompt', 'chosen', 'rejected' as strings or lists of messages.
            # Let's assume they are lists of messages (chat format).
            
            # We need to apply the chat template.
            # If the tokenizer has a chat template, use it. Otherwise fall back to simple concatenation.
            
            try:
                # Chosen
                chosen_messages = chosen if isinstance(chosen, list) else [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
                rejected_messages = rejected if isinstance(rejected, list) else [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
                
                chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            except Exception:
                # Fallback if apply_chat_template fails or is not available
                chosen_text = f"User: {prompt}\nAssistant: {chosen}"
                rejected_text = f"User: {prompt}\nAssistant: {rejected}"

            # Tokenize
            tokenized_chosen = tokenizer(chosen_text, truncation=True, max_length=max_length, add_special_tokens=True)
            tokenized_rejected = tokenizer(rejected_text, truncation=True, max_length=max_length, add_special_tokens=True)
            
            new_examples["chosen_input_ids"].append(tokenized_chosen["input_ids"])
            new_examples["chosen_attention_mask"].append(tokenized_chosen["attention_mask"])
            # Labels are same as input_ids, but we mask the prompt part usually. 
            # For simplicity in this scratch implementation, we train on the whole sequence including prompt.
            # A more advanced version would mask the prompt tokens in the labels.
            new_examples["chosen_labels"].append(tokenized_chosen["input_ids"]) 
            
            new_examples["rejected_input_ids"].append(tokenized_rejected["input_ids"])
            new_examples["rejected_attention_mask"].append(tokenized_rejected["attention_mask"])
            new_examples["rejected_labels"].append(tokenized_rejected["input_ids"])

        return new_examples

    # Process dataset
    # Use batched=True for speed
    dataset = dataset.map(process_function, batched=True, remove_columns=dataset.column_names)
    
    return dataset
