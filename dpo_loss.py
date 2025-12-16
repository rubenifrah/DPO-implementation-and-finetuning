import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    
    def __init__(self, beta=0.1, label_smoothing=0.0):
        """
        Args:
            beta (float): 
            label_smoothing : A regularizer that prevents the model from becoming overconfident.
                                     Instead of forcing the probability of the chosen answer to be exactly 1.0, 
                                     we might aim for 0.99 to add some noise robustness (not used in our study)
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing  

    def forward(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps):
        """
        Compute the DPO loss for a batch of preference pairs.

        Input:
            log probabilities of the chosen and rejected responses given by the policy and reference model
        """
        
        # we first calculate the log ratios of the policy and reference model
        pi_logratios  = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # then we compute the DPO logits and we scale it by Beta. 
        # This value effectively represents the "reward" implicit in the policy.
        logits = self.beta * (pi_logratios - ref_logratios)
        
        # finally we compute the loss
        # We use a logistic loss (sigmoid)
        losses = -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing

        # we compute the rewards for tracking (especially for the margin metric)
        # we detach them because we only use them for logging, not for backpropagation.
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

def get_batch_logps(logits, labels, average_log_prob=False):
    """
    Computes the log-probability of the ground-truth labels given the model's logits.
    """
    
    # Shift labels and logits to align prediction with target
    #removes the first label (since we don't have logits *for* the first token, only *from* it)
    labels = labels[:, 1:].clone()
    
    #removes the last logit (since we don't have a label for the token *after* the sequence ends)
    logits = logits[:, :-1, :]
    
    # Create a mask to ignore padding tokens (usually labeled as -100)
    loss_mask = (labels != -100)
    
    # Replace -100 with 0 temporarily to avoid index errors (the mask will zero out their contribution anyway)
    labels[labels == -100] = 0 

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # Sum up the log probabilities for each sequence (masking out padding)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
