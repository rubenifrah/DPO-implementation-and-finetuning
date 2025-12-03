import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import Dict, List

class DPODataCollator:
    """
    Simple collator for DPO that pads sequences.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Use eos as pad if pad is missing (common for Llama)
        if tokenizer.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Get lists of items
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
        rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
        chosen_labels = [item["chosen_labels"] for item in batch]
        rejected_labels = [item["rejected_labels"] for item in batch]

        # Pad everything
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

    def _pad_batch(self, input_ids, attention_mask, labels):
        # Find max length in this batch
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.max_length)

        padded_ids = []
        padded_mask = []
        padded_labels = []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            # Truncate
            ids = ids[:max_len]
            mask = mask[:max_len]
            lab = lab[:max_len]

            # Pad (right padding)
            pad_len = max_len - len(ids)
            
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)

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
        
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Handle different formats (list of dicts vs strings)
            if not isinstance(chosen, list):
                chosen = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
            if not isinstance(rejected, list):
                rejected = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
            
            # Apply template
            try:
                chosen_text = tokenizer.apply_chat_template(chosen, tokenize=False)
                rejected_text = tokenizer.apply_chat_template(rejected, tokenize=False)
            except:
                # Fallback
                chosen_text = f"User: {prompt}\nAssistant: {chosen}"
                rejected_text = f"User: {prompt}\nAssistant: {rejected}"

            # Tokenize
            tokenized_chosen = tokenizer(chosen_text, truncation=True, max_length=max_length)
            tokenized_rejected = tokenizer(rejected_text, truncation=True, max_length=max_length)
            
            new_examples["chosen_input_ids"].append(tokenized_chosen["input_ids"])
            new_examples["chosen_attention_mask"].append(tokenized_chosen["attention_mask"])
            new_examples["chosen_labels"].append(tokenized_chosen["input_ids"]) 
            
            new_examples["rejected_input_ids"].append(tokenized_rejected["input_ids"])
            new_examples["rejected_attention_mask"].append(tokenized_rejected["attention_mask"])
            new_examples["rejected_labels"].append(tokenized_rejected["input_ids"])

        return new_examples

    dataset = dataset.map(process, batched=True, remove_columns=dataset.column_names)
    return dataset

if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    c = DPODataCollator(t)
    print("Collator init success")
