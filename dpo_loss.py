import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta=0.1, label_smoothing=0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps):
        # Calculate log ratios
        pi_logratios = policy_chosen_logps - ref_chosen_logps
        ref_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO logits
        logits = self.beta * (pi_logratios - ref_logratios)
        
        # Loss
        losses = -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing

        # Rewards for tracking
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

def get_batch_logps(logits, labels, average_log_prob=False):
    # Shift labels and logits for next token prediction
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    loss_mask = (labels != -100)
    labels[labels == -100] = 0 

    # Gather log probs
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
