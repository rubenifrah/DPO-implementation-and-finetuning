import torch
from dpo_loss import DPOLoss
from data_loader import DPODataCollator

def run_sanity_check():
    """
    Runs a quick sanity check on the core DPO components.
    
    Why this exists:
    Before launching a massive training job on the cluster (which might queue for hours),
    it is crucial to verify that:
    1. The Loss function math is consistent.
    2. The Data Collator handles padding/batching correctly.
    """
    print("=" * 60)
    print("RUNNING DRY RUN / SANITY CHECK")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Test DPO Loss
    # -------------------------------------------------------------------------
    print("\n[1] Testing DPO Loss Function...")
    loss_fn = DPOLoss(beta=0.1)
    
    # Create dummy log probabilities
    # Case: Chosen has higher logp (-1.0) than Rejected (-3.0). Model is doing well.
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    # Reference model is identical for this test (ratios should cancel out if diff is same)
    ref_chosen = torch.tensor([-1.0, -2.0])
    ref_rejected = torch.tensor([-3.0, -4.0])
    
    loss, _, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    
    print(f"   Calculated Loss: {loss}")
    # With identical policy/ref, the logits should be 0, and loss should be -log(sigmoid(0)) = -log(0.5) = 0.693
    expected_loss = 0.6931
    if torch.allclose(loss, torch.tensor([expected_loss, expected_loss]), atol=1e-3):
        print("   ✅ Loss calculation is correct (approx 0.693).")
    else:
        print("   ❌ Loss calculation mismatch!")

    # -------------------------------------------------------------------------
    # 2. Test Data Collator
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Collator (Padding & Batching)...")
    
    # Mock a tokenizer object
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1
    
    tokenizer = MockTokenizer()
    collator = DPODataCollator(tokenizer, max_length=10)
    
    # Create a dummy batch with different lengths
    # Item 1: length 3
    # Item 2: length 2 (should be padded)
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
    
    print(f"   Output Keys: {list(out.keys())}")
    print(f"   Chosen Input Shape: {out['policy_chosen_input_ids'].shape} (Expected: [2, 3])")
    
    # Verification
    if out["policy_chosen_input_ids"].shape == (2, 3):
        # Check if padding was applied to the second item (index 1) at the end (index 2)
        if out["policy_chosen_input_ids"][1, 2] == tokenizer.pad_token_id:
             print("   ✅ Padding applied correctly.")
        else:
             print("   ❌ Padding logic failed.")
    else:
        print("   ❌ Batch shape mismatch.")

    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_sanity_check()
