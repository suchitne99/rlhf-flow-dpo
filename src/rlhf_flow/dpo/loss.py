import math
from typing import Tuple, Dict

def dpo_loss_fn(
    policy_chosen_logps: float,
    policy_rejected_logps: float,
    reference_chosen_logps: float,
    reference_rejected_logps: float,
    beta: float = 0.1
) -> Tuple[float, Dict[str, float]]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    scaled_logits = beta * logits
    loss = -scaled_logits + math.log(1.0 + math.exp(scaled_logits)) if scaled_logits <= 0 else math.log(1.0 + math.exp(-scaled_logits))
    margin = beta * (policy_chosen_logps - reference_chosen_logps) - beta * (policy_rejected_logps - reference_rejected_logps)
    return loss, {"reward_margin": margin, "accuracy": 1.0 if margin > 0 else 0.0}

# Batch tensorization

# Label smoothing

# Dynamic beta schedule

# In-place subtraction
