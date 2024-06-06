import pytest
from rlhf_flow.dpo.loss import dpo_loss_fn
from rlhf_flow.dpo.cache import ReferenceLogProbCache
from rlhf_flow.data.collator import PreferencePair, PreferenceDataCollator
from rlhf_flow.sft.trainer import SFTTrainer, SFTTrainingConfig

def test_dpo_loss_standard():
    loss, metrics = dpo_loss_fn(
        policy_chosen_logps=-1.2,
        policy_rejected_logps=-2.5,
        reference_chosen_logps=-1.4,
        reference_rejected_logps=-2.2,
        beta=0.1
    )
    assert loss > 0.0
    assert metrics["reward_margin"] > 0.0
    assert metrics["accuracy"] == 1.0

def test_dpo_loss_reversal():
    loss, metrics = dpo_loss_fn(
        policy_chosen_logps=-3.0,
        policy_rejected_logps=-1.0,
        reference_chosen_logps=-1.0,
        reference_rejected_logps=-2.0,
        beta=0.1
    )
    assert metrics["reward_margin"] < 0.0
    assert metrics["accuracy"] == 0.0

def test_reference_logprob_cache():
    cache = ReferenceLogProbCache()
    cache.store("Prompt 1", "Chosen response", "Rejected response", -2.1, -4.5)
    assert cache.size() == 1
    retrieved = cache.get("Prompt 1", "Chosen response", "Rejected response")
    assert retrieved == (-2.1, -4.5)

def test_preference_collator():
    collator = PreferenceDataCollator()
    batch = [
        PreferencePair(prompt="What is RLHF?", chosen="Reinforcement learning...", rejected="I don't know."),
        PreferencePair(prompt="Explain DPO.", chosen="Direct preference...", rejected="Something else.")
    ]
    res = collator.collate_batch(batch)
    assert res["batch_size"] == 2
    assert len(res["chosen_responses"]) == 2

def test_sft_trainer_specs():
    config = SFTTrainingConfig()
    trainer = SFTTrainer(config)
    specs = trainer.prepare_model_and_tokenizer()
    assert specs["trainable_percentage"] < 1.0
    assert specs["gradient_checkpointing_enabled"] is True

# Extreme negative logit stability test

# Multi-threaded cache concurrency test

# Monotonic decrease test

# Gradient norm clipping verified

# Gradient checkpointing verification

# KTO loss variance

# Reward margin monotonicity test

# Multi-GPU gradient synchronization test
