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

__all__ = [
    "AlgoConfig",
    "FilterGroupsConfig",
    "JudgeSFTConfig",
    "KLControlConfig",
    "SentenceAdvConfig",
    "SentenceJudgeAdvConfig",
]


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
class SentenceJudgeAdvConfig(BaseConfig):
    """Configuration for sentence-level judge advantage shaping.

    Args:
        enable (bool): Enable sentence-level judge advantage shaping.
        alpha (float): Weight to fuse judge advantage into base advantages.
        buckets (list[float]): Discrete advantage buckets.
        use_confidence_weight (bool): Whether to weight by judge confidence.
        confidence_floor (float): Confidence floor for keeping a sentence signal.
        normalize (str): Normalization mode: "none" or "zscore".
        correctness_threshold (float): Threshold on sequence reward to define correctness.
        judge_backend (str): Judge backend: "dummy", "callable", "self".
            every_n_steps (int): Run judge every N steps (1 = every step).
        judge_fn (Optional[str]): Import path of a callable judge function.
        max_sentences (int): Max sentences passed to judge (0 = no limit).
        max_chars (int): Max characters passed to judge (0 = no limit).
        judge_max_tokens (int): Max tokens to generate for judge output.
        rate_limit_qps (float): QPS rate limit for judge calls.
        debug_prompt (bool): Whether to include rendered prompt in debug.
    """

    enable: bool = False
    alpha: float = 0.05
    buckets: list[float] = field(default_factory=lambda: [0.75, 0.25, 0.0, -0.25, -0.75])
    use_confidence_weight: bool = True
    confidence_floor: float = 0.2
    normalize: str = "zscore"
    correctness_threshold: float = 0.0
    judge_backend: str = "dummy"
    judge_fn: Optional[str] = None
    every_n_steps: int = 1
    max_sentences: int = 0
    max_chars: int = 0
    judge_max_tokens: int = 256
    rate_limit_qps: float = 0.0
    debug_prompt: bool = False


@dataclass
class JudgeSFTConfig(BaseConfig):
    """Configuration for judge SFT mixed loss during RL training.

    When enabled, a cross-entropy SFT loss on pre-distilled judge data is added
    to the policy loss: total_loss = policy_loss + lambda_weight * judge_sft_loss.
    This allows the model's judge capability to evolve during RL training.

    Args:
        enable (bool): Enable judge SFT mixed loss.
        data_path (str): Path to distilled judge SFT parquet (columns: prompt, response).
        lambda_weight (float): Weight for judge SFT loss.
        micro_batch_size (int): Micro-batch size for judge SFT samples per gradient step.
        max_seq_len (int): Maximum sequence length for judge SFT samples.
    """

    enable: bool = False
    data_path: str = ""
    lambda_weight: float = 0.1
    micro_batch_size: int = 2
    max_seq_len: int = 2048


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
    sentence_judge_adv: Optional[SentenceJudgeAdvConfig] = None
    judge_sft: Optional[JudgeSFTConfig] = None
