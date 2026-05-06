"""Verify the Hydra/OmegaConf -> AlgoConfig -> compute_advantage wiring for KTAE.

This closes a real gap I had: my earlier equivalence test bypassed Hydra entirely
by instantiating ``KTAEConfig()`` directly. This test simulates what the launcher
actually does at runtime:

    1. Load the same default ``ppo_trainer.yaml`` Hydra would load.
    2. Apply the same ``+algorithm.ktae.*`` command-line overrides our launcher passes.
    3. Convert the resulting OmegaConf DictConfig to an AlgoConfig dataclass via
       the same ``omega_conf_to_dataclass`` helper main_ppo.py calls.
    4. Pass that AlgoConfig into our ``compute_advantage`` (registry path) and
       verify the result equals what we get when constructing KTAEConfig directly.

If this passes, the entire shell -> Hydra -> dataclass -> registry -> kernel
chain is verified to work on CPU. The only remaining gap is GPU/FSDP/vLLM,
which can only be confirmed by a 1-step training smoke on a real GPU box.
"""
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, os.path.expanduser("~/sentencepo_v1-5"))

from verl.trainer.config.algorithm import AlgoConfig, KTAEConfig
from verl.trainer.ppo import core_algos as ca
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass

torch.manual_seed(0)
np.random.seed(0)


# ---------------- Step 1+2: load + override the way Hydra would --------------
config_dir = os.path.expanduser("~/sentencepo_v1-5/verl/trainer/config")
overrides = [
    "algorithm.adv_estimator=ktae",
    "+algorithm.ktae.alpha=1.0",
    "+algorithm.ktae.beta_ig=1.0",
    "+algorithm.ktae.gamma_tf=1.0",
    "+algorithm.ktae.top=1.0",
    "+algorithm.ktae.bottom=-1.0",
    "+algorithm.ktae.pad_token_id=151643",
]

with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=overrides)

print("=== Hydra after merge — algorithm subtree ===")
print(OmegaConf.to_yaml(cfg.algorithm))

# Confirm ktae field made it through
assert "ktae" in cfg.algorithm, "FAIL: +algorithm.ktae.* was not injected by Hydra"
assert float(cfg.algorithm.ktae.alpha) == 1.0
assert float(cfg.algorithm.ktae.beta_ig) == 1.0
assert int(cfg.algorithm.ktae.pad_token_id) == 151643
print("[hydra] command-line override successfully injected.")


# ---------------- Step 3: OmegaConf -> AlgoConfig dataclass ------------------
algo_dc = omega_conf_to_dataclass(cfg.algorithm, dataclass_type=AlgoConfig)
print(f"[dataclass] type(algo_dc) = {type(algo_dc).__name__}")
print(f"[dataclass] algo_dc.adv_estimator = {algo_dc.adv_estimator!r}")
print(f"[dataclass] algo_dc.ktae = {algo_dc.ktae}")

assert isinstance(algo_dc, AlgoConfig)
assert algo_dc.adv_estimator == "ktae"
assert isinstance(algo_dc.ktae, KTAEConfig), f"ktae field is {type(algo_dc.ktae)}, expected KTAEConfig"
assert algo_dc.ktae.alpha == 1.0
assert algo_dc.ktae.beta_ig == 1.0
assert algo_dc.ktae.pad_token_id == 151643
print("[dataclass] AlgoConfig and nested KTAEConfig populated correctly.\n")


# ---------------- Step 4: Run compute_advantage end-to-end -------------------
def build_data_proto(g=8, t=64, vocab=151700, n_correct=3):
    responses = torch.randint(0, vocab, (g, t), dtype=torch.long)
    sig_pos = [t // 8, t // 4, t // 2]
    responses[:n_correct, sig_pos] = 42
    responses[n_correct:, sig_pos] = 43
    responses[0, -1] = 151643
    response_mask = torch.ones(g, t, dtype=torch.float32)
    response_mask[:, -3:] = 0.0
    rewards = torch.zeros(g, dtype=torch.float32)
    rewards[:n_correct] = 1.0
    tlr = torch.zeros(g, t, dtype=torch.float32)
    tlr[:, -4] = rewards
    index = np.array(["q0"] * g, dtype=object)

    # DataProto: TensorDict for batched tensors + dict for non-tensor batch
    from tensordict import TensorDict
    batch = TensorDict(
        {
            "token_level_rewards": tlr,
            "responses": responses,
            "response_mask": response_mask,
        },
        batch_size=[g],
    )
    return DataProto(batch=batch, non_tensor_batch={"uid": index}, meta_info={})


torch.manual_seed(0); np.random.seed(0)
data = build_data_proto()
data_a = compute_advantage(data, adv_estimator="ktae", config=algo_dc)
adv_a = data_a.batch["advantages"].clone()

# Compare to direct-instantiation path (no Hydra). Re-seed so the synthetic
# batch is byte-identical to the previous call.
torch.manual_seed(0); np.random.seed(0)
data2 = build_data_proto()
data_b = compute_advantage(
    data2,
    adv_estimator="ktae",
    config=AlgoConfig(adv_estimator="ktae", ktae=KTAEConfig()),
)
adv_b = data_b.batch["advantages"].clone()

same = torch.allclose(adv_a, adv_b, atol=1e-6, rtol=0)
diff = (adv_a - adv_b).abs().max().item()
print(f"[end-to-end] Hydra-path advantage vs direct-instantiation advantage:")
print(f"             allclose = {same}   max_abs_diff = {diff:.2e}")
assert same, "Hydra-injected config produced different advantage from direct-construction"
print("\nOVERALL: PASS — Hydra config injection wired correctly through the registry.")
