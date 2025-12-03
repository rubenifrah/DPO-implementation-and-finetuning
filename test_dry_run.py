import torch
from dpo_loss import DPOLoss
from transformers import AutoTokenizer

def test_dpo_loss():
    print("Testing DPO Loss...")
    loss_fn = DPOLoss(beta=0.1)
    
    # Dummy logprobs
    # Batch size 2
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-1.0, -2.0])
    ref_rejected = torch.tensor([-3.0, -4.0])
    
    # With equal logprobs, log ratios are 0.
    # logits = beta * (0 - 0) = 0
    # loss = -log(sigmoid(0)) = -log(0.5) = 0.693
    
    loss, _, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    
    print(f"Loss: {loss}")
    assert torch.allclose(loss, torch.tensor([0.6931, 0.6931]), atol=1e-4)
    print("DPO Loss Test Passed!")

def test_data_loader():
    print("\nTesting Data Loader (Mock)...")
    # We won't load the real dataset to avoid downloading GBs, but we'll test the collator
    from data_loader import DPODataCollator
    
    # Mock tokenizer
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
    
    tokenizer = MockTokenizer()
    collator = DPODataCollator(tokenizer, max_length=10)
    
    batch = [
        {
            "chosen_input_ids": [1, 2, 3],
            "chosen_attention_mask": [1, 1, 1],
            "chosen_labels": [1, 2, 3],
            "rejected_input_ids": [4, 5],
            "rejected_attention_mask": [1, 1],
            "rejected_labels": [4, 5],
        },
        {
            "chosen_input_ids": [1, 2],
            "chosen_attention_mask": [1, 1],
            "chosen_labels": [1, 2],
            "rejected_input_ids": [4, 5, 6, 7],
            "rejected_attention_mask": [1, 1, 1, 1],
            "rejected_labels": [4, 5, 6, 7],
        }
    ]
    
    collated = collator(batch)
    print("Collated Batch Keys:", collated.keys())
    print("Policy Chosen Shape:", collated["policy_chosen_input_ids"].shape)
    
    # Check padding
    # Max len for chosen is 3. Second item has len 2, so should be padded by 1.
    assert collated["policy_chosen_input_ids"].shape == (2, 3)
    assert collated["policy_chosen_input_ids"][1, 2] == 0 # Pad token
    
    print("Data Loader Test Passed!")

if __name__ == "__main__":
    test_dpo_loss()
    test_data_loader()
