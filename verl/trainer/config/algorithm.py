# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "SentenceAdvConfig", "SLPAConfig", "SCRConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class SentenceAdvConfig(BaseConfig):
    """Configuration for sentence-level semantic advantage.

    Args:
        enable (bool): Enable sentence-level semantic advantage.
        alpha (float): Weight to fuse sentence advantage into base advantages.
        temperature (float): Softmax temperature for similarity.
        pooling (str): Sentence embedding pooling: "mean" or "last".
        normalize (bool): Whether to z-normalize sentence advantages within bucket.
        eps (float): Numerical stability epsilon.
        correctness_threshold (float): Threshold on sequence reward to define correctness.
        bucket_count (int): Number of relative-position buckets per response.
        metrics_enable (bool): Whether to emit extra semantic metrics.
    """

    enable: bool = False
    alpha: float = 0.1
    temperature: float = 0.1
    pooling: str = "last"
    normalize: bool = True
    eps: float = 1e-8
    correctness_threshold: float = 0.0
    bucket_count: int = 3
    metrics_enable: bool = True


@dataclass
class SLPAConfig(BaseConfig):
    """Configuration for Sentence-Level Process Advantage (SLPA).

    Estimates V_k at each sentence boundary via kernel-weighted reward
    regression over group-internal rollouts, then computes temporal
    differences Δ_k = V_k - V_{k-1} as sentence-level credit.

    Args:
        enable (bool): Enable SLPA.
        alpha_correct (float): Fusion weight for correct rollouts.
        alpha_incorrect (float): Fusion weight for incorrect rollouts.
        tau_emb (float): Temperature for embedding cosine kernel.
        sigma_pos (float): Bandwidth for position Gaussian kernel.
        normalize (bool): Z-score normalize Δ within group.
        eps (float): Numerical stability epsilon.
        correctness_threshold (float): Threshold on sequence reward to define correctness.
        metrics_enable (bool): Whether to emit SLPA diagnostics.
        alpha_decay (str): Decay schedule for alpha ('none' or 'linear').
        alpha_min_ratio (float): Minimum alpha as fraction of initial (for linear decay).
    """

    enable: bool = False
    alpha_correct: float = 0.1
    alpha_incorrect: float = 0.1
    tau_emb: float = 0.1
    sigma_pos: float = 1.0
    normalize: bool = True
    eps: float = 1e-8
    correctness_threshold: float = 0.0
    metrics_enable: bool = True
    alpha_decay: str = "none"
    alpha_min_ratio: float = 0.1


@dataclass
class SCRConfig(BaseConfig):
    """Configuration for Sentence Contrastive Reward (SCR).

    Uses soft reward-weighted embedding centers with leave-one-out
    to compute per-sentence contrastive affinity scores.

    Args:
        enable (bool): Enable SCR.
        alpha_correct (float): Fusion weight for correct rollouts.
        alpha_incorrect (float): Fusion weight for incorrect rollouts.
        tau_reward (float): Temperature for reward softmax weighting.
        tau_sim (float): Temperature for cosine similarity scaling.
        normalize (bool): Z-score normalize SCR within group.
        eps (float): Numerical stability epsilon.
        correctness_threshold (float): Threshold on sequence reward to define correctness.
        metrics_enable (bool): Whether to emit SCR diagnostics.
        alpha_decay (str): Decay schedule for alpha ('none' or 'linear').
        alpha_min_ratio (float): Minimum alpha as fraction of initial (for linear decay).
    """

    enable: bool = False
    alpha_correct: float = 0.05
    alpha_incorrect: float = 0.05
    tau_reward: float = 1.0
    tau_sim: float = 0.1
    normalize: bool = True
    eps: float = 1e-8
    correctness_threshold: float = 0.0
    metrics_enable: bool = True
    alpha_decay: str = "none"
    alpha_min_ratio: float = 0.1


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_is_threshold (Optional[float]): Upper threshold for IS weights. null = disabled,
            float value = enabled (compute weights and metrics). This is the main on/off switch.
        rollout_is_threshold_lower (Optional[float]): Lower threshold for IS weights. If None, defaults to 1/upper.
        rollout_is_level (str): Aggregation level: "token", "sequence", or "geometric".
        rollout_is_mode (str): Bounding mode: "truncate" (cap upper only) or "mask" (zero outside bounds).
        rollout_is_veto_threshold (float): Per-token veto threshold for catastrophic outliers.
        rollout_is (bool): Whether to apply IS weights to policy loss. True = apply weights,
            False = compute metrics only (useful for monitoring before enabling correction). Default: False.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Importance Sampling (replaces legacy tis_imp_ratio_cap)
    # Controls computation of IS weights and mismatch metrics
    rollout_is_threshold: Optional[float] = None  # null = disabled, float = enabled
    rollout_is_threshold_lower: Optional[float] = None
    rollout_is_level: str = "token"
    rollout_is_mode: str = "truncate"
    rollout_is_veto_threshold: Optional[float] = 1e-4
    # Controls whether to apply IS weights to policy loss (only if rollout_is_threshold is set)
    # True = apply weights to loss, False = compute metrics only (no weight application)
    rollout_is: bool = False
    sentence_adv: Optional[SentenceAdvConfig] = None
    slpa: Optional[SLPAConfig] = None
    scr: Optional[SCRConfig] = None
