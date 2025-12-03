import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss module.
    
    Reference: https://arxiv.org/abs/2305.18290
    """
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Args:
            beta (float): Temperature parameter for the DPO loss, typically in range [0.1, 0.5].
                          Controls how much we deviate from the reference model.
            label_smoothing (float): Label smoothing parameter (default: 0.0).
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of preference pairs.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            losses: The DPO loss for each example in the batch.
            chosen_rewards: Implicit rewards for the chosen responses.
            rejected_rewards: Implicit rewards for the rejected responses.
        """
        
        # Calculate the log ratio of the policy to the reference model
        # pi_logratios = log(pi(y|x)) - log(ref(y|x))
        pi_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # The DPO objective is to maximize the likelihood of the preference:
        # P(chosen > rejected) = sigmoid(beta * (log(pi_chosen/ref_chosen) - log(pi_rejected/ref_rejected)))
        
        logits = self.beta * (pi_logratios - rejected_logratios)
        
        # The loss is negative log likelihood of the preference
        # loss = -log(sigmoid(logits))
        # We use F.logsigmoid which is numerically more stable than log(sigmoid(x))
        losses = -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing

        # Implicit rewards for tracking
        # reward = beta * (log(pi(y|x)) - log(ref(y|x)))
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probability (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per token. 
                          If False, return the sum of log probabilities.

    Returns:
        logps: Log probabilities of the labels (batch_size,)
    """
    assert logits.shape[:-1] == labels.shape

    # Shift labels and logits so that tokens < n predict n
    # We want to predict the next token, so we shift labels left by 1
    # and logits are already for the next token prediction? 
    # Standard causal LM training: logits[i] predicts labels[i] (if labels are targets)
    # Usually labels are input_ids.
    # If logits[t] is prediction for input_ids[t+1], then:
    
    # Standard implementation for CausalLM:
    # labels are usually input_ids. 
    # logits[:, :-1, :] predicts labels[:, 1:]
    
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    loss_mask = (labels != -100) # Ignore padding tokens

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0 

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
