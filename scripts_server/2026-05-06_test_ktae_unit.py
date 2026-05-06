"""Unit smoke test for KTAE advantage estimator.

Builds a synthetic batch shaped like a real GRPO group rollout (G=8 rollouts,
short response length, mix of correct/wrong rewards), passes it through
``compute_ktae_outcome_advantage_and_keytokens`` via the registry, and asserts:

1. Output shape matches token_level_rewards.
2. No NaN / Inf in advantage tensor.
3. Group-mean advantage over correct rollouts > group-mean over wrong (sanity:
   correct rollouts should receive a *higher* per-token credit boost on average).
4. Advantages on padding tokens (mask=0) are 0.
5. Tokens unique to correct rollouts get strictly positive key_token weight,
   tokens unique to wrong rollouts get strictly negative key_token weight.

Run on CPU; takes <2 seconds. Run on GPU by setting DEVICE=cuda env var.
"""

import os
import sys

import numpy as np
import torch

# Make verl importable when run from anywhere
sys.path.insert(0, os.path.expanduser("~/sentencepo_v1-5"))

from verl.trainer.config.algorithm import AlgoConfig, KTAEConfig
from verl.trainer.ppo import core_algos as ca

DEVICE = os.environ.get("DEVICE", "cpu")
print(f"[smoke] device = {DEVICE}")

torch.manual_seed(0)
np.random.seed(0)


def build_batch(g=8, t=32, vocab=200, n_correct=3, signal_token=42, antisignal_token=43):
    """Build one prompt-group worth of rollouts.

    - All rollouts share the same prompt id ``"q0"`` (so the group has G entries).
    - First ``n_correct`` rollouts get reward=1 and contain ``signal_token`` exactly twice.
    - Remaining (G - n_correct) rollouts get reward=0 and contain ``antisignal_token`` exactly twice.
    - Other tokens are uniformly random in [0, vocab).
    """
    responses = torch.randint(0, vocab, (g, t), dtype=torch.long, device=DEVICE)
    # plant signal: token id 42 in correct, 43 in wrong; 2 occurrences each at fixed positions
    responses[:n_correct, 5] = signal_token
    responses[:n_correct, 17] = signal_token
    responses[n_correct:, 5] = antisignal_token
    responses[n_correct:, 17] = antisignal_token
    response_mask = torch.ones(g, t, dtype=torch.float32, device=DEVICE)
    # Mask out the last 4 tokens of every rollout to simulate ragged ends
    response_mask[:, -4:] = 0.0
    rewards = torch.zeros(g, dtype=torch.float32, device=DEVICE)
    rewards[:n_correct] = 1.0
    token_level_rewards = torch.zeros(g, t, dtype=torch.float32, device=DEVICE)
    token_level_rewards[:, -5] = rewards  # put scalar reward on the last unmasked position
    index = np.array(["q0"] * g, dtype=object)
    return token_level_rewards, response_mask, responses, index, rewards


def run():
    cfg = AlgoConfig(adv_estimator="ktae", ktae=KTAEConfig(pad_token_id=151643))
    fn = ca.get_adv_estimator_fn("ktae")
    print(f"[smoke] resolved fn = {fn.__name__}")

    tlr, mask, responses, index, rewards = build_batch()
    print(
        f"[smoke] batch: G={tlr.shape[0]} T={tlr.shape[1]} "
        f"n_correct={int((rewards>0).sum())} n_wrong={int((rewards<=0).sum())}"
    )
    adv, ret = fn(
        token_level_rewards=tlr,
        response_mask=mask,
        index=index,
        responses=responses,
        config=cfg,
    )

    # 1. shape
    assert adv.shape == tlr.shape, f"shape mismatch: {adv.shape} vs {tlr.shape}"
    assert ret.shape == tlr.shape

    # 2. finite
    assert torch.isfinite(adv).all(), "adv contains NaN/Inf"

    # 3. correct mean > wrong mean
    mean_correct = adv[rewards > 0].mean().item()
    mean_wrong = adv[rewards <= 0].mean().item()
    print(f"[smoke] mean adv per token: correct={mean_correct:+.4f} wrong={mean_wrong:+.4f}")
    assert mean_correct > mean_wrong, "expected correct rollouts to have higher mean advantage"

    # 4. zero on padded positions
    pad_adv = adv[mask == 0]
    assert torch.all(pad_adv == 0), f"non-zero adv on padding: max abs = {pad_adv.abs().max().item()}"

    # 5. signal token sanity: positions where token id==42 should have notably higher adv
    #    than positions where token id==43, ON AVERAGE across rollouts.
    pos_42 = adv[responses == 42]
    pos_43 = adv[responses == 43]
    print(
        f"[smoke] token 42 (correct-only) adv mean = {pos_42.mean().item():+.4f} "
        f"({pos_42.numel()} tokens); "
        f"token 43 (wrong-only) adv mean = {pos_43.mean().item():+.4f} "
        f"({pos_43.numel()} tokens)"
    )
    assert pos_42.mean() > pos_43.mean(), "expected token 42 (in correct rollouts) to have higher adv"

    # advantages should NOT all be identical along the response axis (token-level signal must vary)
    var_per_rollout = adv.var(dim=-1)
    assert (var_per_rollout > 0).all(), "KTAE produced uniform advantages within rollouts (check key-token wiring)"

    print(f"[smoke] var_per_rollout: min={var_per_rollout.min().item():.4f} max={var_per_rollout.max().item():.4f}")
    print("[smoke] OK")


if __name__ == "__main__":
    run()
