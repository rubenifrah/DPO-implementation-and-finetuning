import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss.
    
    This loss function effectively turns a Language Model into its own Reward Model.
    Instead of training a separate reward model (as in RLHF), we optimize the policy 
    directly to satisfy human preferences.
    
    The Key Idea:
    We want the model (Policy) to assign a higher probability to the 'Chosen' response 
    compared to the 'Rejected' response, relative to a baseline reference model.
    """
    
    def __init__(self, beta=0.1, label_smoothing=0.0):
        """
        Args:
            beta (float): The 'temperature' of the preference optimization. 
                          - Higher beta (e.g., 0.5) forces the model to strictly adhere to preferences 
                            but might make it diverge too far from the reference.
                          - Lower beta (e.g., 0.1) keeps the model closer to the reference (SFT) model.
                          Think of it as the inverse of the KL-divergence penalty weight.
            
            label_smoothing (float): A regularizer that prevents the model from becoming overconfident.
                                     Instead of forcing the probability of the chosen answer to be exactly 1.0, 
                                     we might aim for 0.99 to add some noise robustness.
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing  

    def forward(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps):
        """
        Compute the DPO loss for a batch of preference pairs.

        Input:
            policy_chosen_logps:   Log-probabilities of the 'Chosen' response given by the Model being trained.
            policy_rejected_logps: Log-probabilities of the 'Rejected' response given by the Model being trained.
            ref_chosen_logps:      Log-probabilities of the 'Chosen' response given by the Reference (frozen) model.
            ref_rejected_logps:    Log-probabilities of the 'Rejected' response given by the Reference (frozen) model.
        """
        
        # 1. Calculate the 'Log Ratio' for both the Policy (trained) and Reference (frozen) models.
        # This measures "how much more likely" the model thinks the chosen response is compared to the rejected one.
        # Ideally, we want the Policy to have a HIGHER ratio than the Reference.
        pi_logratios  = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # 2. Compute the DPO Logits
        # We take the difference between the Policy's confidence and the Reference's confidence.
        # Then we scale it by Beta. 
        # This value effectively represents the "Reward" implicit in the policy.
        logits = self.beta * (pi_logratios - ref_logratios)
        
        # 3. Compute the Loss
        # We use a logistic loss (sigmoid) to encourage 'logits' to be positive (i.e., Policy prefers Chosen more than Ref does).
        # -F.logsigmoid(logits) is equivalent to -log(sigmoid(logits)).
        losses = -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing

        # 4. Compute 'Rewards' for tracking
        # These metrics help us understand if the model is actually learning to separate the chosen/rejected examples.
        # Note: We detach() them because we only use them for logging, not for backpropagation.
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

def get_batch_logps(logits, labels, average_log_prob=False):
    """
    Computes the log-probability of the ground-truth labels given the model's logits.
    
    Why this is tricky:
    Language Models predict the *next* token. So if we have a sequence:
    Input:  [A, B, C, D]
    Logits: [Pred_for_B, Pred_for_C, Pred_for_D, Pred_for_E]
    
    We need to align them so that we check the probability of:
    - Logits[0] (prediction for pos 1) vs Labels[1] (actual token B)
    - Logits[1] (prediction for pos 2) vs Labels[2] (actual token C)
    ... and so on.
    """
    
    # Shift labels and logits to align prediction with target
    # Remove the first label (since we don't have logits *for* the first token, only *from* it)
    labels = labels[:, 1:].clone()
    
    # Remove the last logit (since we don't have a label for the token *after* the sequence ends)
    logits = logits[:, :-1, :]
    
    # Create a mask to ignore padding tokens (usually labeled as -100)
    loss_mask = (labels != -100)
    
    # Replace -100 with 0 temporarily to avoid index errors (the mask will zero out their contribution anyway)
    labels[labels == -100] = 0 

    # Gather the log probabilities of the actual labeled tokens
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # Sum up the log probabilities for each sequence (masking out padding)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
