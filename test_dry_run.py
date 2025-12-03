import torch
from dpo_loss import DPOLoss
from data_loader import DPODataCollator

# this script serves as a dry run to test the data loader and loss function
# it does not train a model or save any checkpoints and can be run without a GPU

def test():
    print("Testing DPO Loss...")
    loss_fn = DPOLoss(beta=0.1)
    
    # Dummy data
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-1.0, -2.0])
    ref_rejected = torch.tensor([-3.0, -4.0])
    
    loss, _, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"Loss: {loss}")
    
    # Should be around 0.693
    assert torch.allclose(loss, torch.tensor([0.6931, 0.6931]), atol=1e-4)
    print("Loss test passed!")

    print("\nTesting Data Collator...")
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
    
    out = collator(batch)
    print("Collator output keys:", out.keys())
    print("Shape:", out["policy_chosen_input_ids"].shape)
    
    # Check padding
    assert out["policy_chosen_input_ids"].shape == (2, 3)
    assert out["policy_chosen_input_ids"][1, 2] == 0
    print("Collator test passed!")

if __name__ == "__main__":
    test()
