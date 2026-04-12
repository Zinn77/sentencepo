# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.metric_utils import compute_sentencepo_metrics
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def apply_sentence_entropy_advantage(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    alpha_pos: float = 0.1,
    alpha_neg: float = 0.0,
    norm: str = "zscore",
    clip: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply sentence-level entropy scaling to token advantages.

    For each sentence, compute mean entropy and use it to scale the advantages
    for tokens belonging to that sentence. Positive and negative advantages can
    use different scaling strengths via ``alpha_pos`` and ``alpha_neg``.
    """

    if entropys is None or sentence_ids is None:
        return advantages
    if advantages.shape != entropys.shape or advantages.shape != sentence_ids.shape:
        return advantages

    with torch.no_grad():
        if advantages.numel() == 0:
            return advantages

        bs, seq_len = sentence_ids.shape
        device = sentence_ids.device

        sid_max = sentence_ids.max().item() if sentence_ids.numel() > 0 else -1
        sid_min = sentence_ids.min().item() if sentence_ids.numel() > 0 else -1
        needs_offset = sid_max < seq_len and sid_min >= -1
        if needs_offset:
            offset = torch.arange(bs, device=device).unsqueeze(1) * (seq_len + 1)
            sid_with_offset = sentence_ids + offset
        else:
            sid_with_offset = sentence_ids

        valid_mask = (response_mask > 0) & (sentence_ids >= 0)
        if not torch.any(valid_mask):
            return advantages

        flat_valid = valid_mask.view(-1)
        flat_sid = sid_with_offset.view(-1)[flat_valid]
        flat_ent = entropys.view(-1)[flat_valid]

        unique_sid, inv = torch.unique(flat_sid, return_inverse=True)
        ones = torch.ones_like(inv, dtype=flat_ent.dtype)
        sent_lens = torch.zeros_like(unique_sid, dtype=flat_ent.dtype)
        sent_lens.index_add_(0, inv, ones)
        ent_sum = torch.zeros_like(unique_sid, dtype=flat_ent.dtype)
        ent_sum.index_add_(0, inv, flat_ent)
        sent_entropy = ent_sum / (sent_lens + eps)

        norm_mode = (norm or "").lower()
        if norm_mode in {"zscore", "std", "standard"}:
            mu = sent_entropy.mean()
            sigma = sent_entropy.std(unbiased=False).clamp_min(eps)
            sent_entropy = (sent_entropy - mu) / sigma
        elif norm_mode in {"minmax", "min-max"}:
            minv = sent_entropy.min()
            maxv = sent_entropy.max()
            sent_entropy = (sent_entropy - minv) / (maxv - minv + eps)
            sent_entropy = sent_entropy * 2.0 - 1.0

        if clip is not None and clip > 0:
            sent_entropy = sent_entropy.clamp(min=-clip, max=clip)

        sent_entropy_token = sent_entropy[inv]
        entropy_token = torch.zeros_like(advantages, dtype=advantages.dtype).view(-1)
        entropy_token[flat_valid] = sent_entropy_token.to(advantages.dtype)
        entropy_token = entropy_token.view_as(advantages)

        alpha_tensor = torch.where(
            advantages >= 0,
            advantages.new_tensor(alpha_pos),
            advantages.new_tensor(alpha_neg),
        )
        scale = 1.0 + alpha_tensor * entropy_token
        advantages = advantages * scale

    return advantages


def compute_sentence_semantic_advantage(
    token_hidden_states: torch.Tensor | None,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    token_level_rewards: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    sentence_embeddings: torch.Tensor | None = None,
    sentence_unique_ids: torch.Tensor | None = None,
    sentence_sample_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute bucketed sentence-level semantic advantage with correct/incorrect contrast.

    Returns:
        sentence_advantages: (bs, seq_len)
        metrics: scalar metrics for bucket diagnostics
    """

    metrics: dict[str, float] = {}
    if config is None:
        return torch.zeros_like(response_mask, dtype=response_mask.dtype, device=response_mask.device), metrics
    sentence_adv_cfg = getattr(config, "sentence_adv", None)
    if sentence_adv_cfg is None or not getattr(sentence_adv_cfg, "enable", False):
        return torch.zeros_like(response_mask, dtype=response_mask.dtype, device=response_mask.device), metrics

    eps = float(getattr(sentence_adv_cfg, "eps", 1e-8))
    bucket_count = max(1, int(getattr(sentence_adv_cfg, "bucket_count", 3)))
    tau = float(getattr(sentence_adv_cfg, "temperature", 0.1))
    normalize_adv = bool(getattr(sentence_adv_cfg, "normalize", True))
    metrics_enable = bool(getattr(sentence_adv_cfg, "metrics_enable", True))

    device = response_mask.device
    sentence_ids = sentence_ids.to(device)
    response_mask = response_mask.to(device)

    bs, seq_len = sentence_ids.shape
    valid = (response_mask > 0) & (sentence_ids >= 0)
    if not torch.any(valid):
        return torch.zeros((bs, seq_len), device=device, dtype=response_mask.dtype), metrics

    flat_valid = valid.view(-1)
    flat_sid = sentence_ids.view(-1)[flat_valid]
    flat_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
    flat_pos = (flat_idx % seq_len).to(torch.long)
    flat_batch = (flat_idx // seq_len).to(torch.long)

    if sentence_embeddings is not None and sentence_unique_ids is not None and sentence_sample_idx is not None:
        sent_emb = sentence_embeddings.float().to(device)
        unique_sid = sentence_unique_ids.to(device)
        if unique_sid.numel() == 0:
            return torch.zeros((bs, seq_len), device=device, dtype=sent_emb.dtype), metrics
        inv = torch.searchsorted(unique_sid, flat_sid)
        sent_sample_idx = sentence_sample_idx.to(device)
    else:
        if token_hidden_states is None:
            return torch.zeros((bs, seq_len), device=device, dtype=response_mask.dtype), metrics

        token_hidden_states = token_hidden_states.float().to(device)
        bs, seq_len, hidden = token_hidden_states.shape
        flat_emb = token_hidden_states.view(-1, hidden)[flat_valid]

        unique_sid, inv = torch.unique(flat_sid, return_inverse=True)
        num_sent = unique_sid.numel()
        if num_sent == 0:
            return torch.zeros((bs, seq_len), device=device, dtype=token_hidden_states.dtype), metrics

        next_sid = torch.roll(sentence_ids, shifts=-1, dims=1)
        next_valid = torch.roll(valid, shifts=-1, dims=1)
        last_pos_mask = torch.zeros_like(valid)
        last_pos_mask[:, -1] = True
        boundary = last_pos_mask | (sentence_ids != next_sid) | (~next_valid)
        last_mask = valid & boundary

        flat_last = last_mask.view(-1)
        flat_sid_last = sentence_ids.view(-1)[flat_last]
        flat_emb_last = token_hidden_states.view(-1, hidden)[flat_last]

        if flat_sid_last.numel() == 0:
            return torch.zeros((bs, seq_len), device=device, dtype=token_hidden_states.dtype), metrics

        idx_last = torch.searchsorted(unique_sid, flat_sid_last)

        pooling = getattr(sentence_adv_cfg, "pooling", "last")
        if pooling == "mean":
            sum_emb = torch.zeros((num_sent, hidden), device=device, dtype=token_hidden_states.dtype)
            cnt = torch.zeros((num_sent, 1), device=device, dtype=token_hidden_states.dtype)
            sum_emb.index_add_(0, inv, flat_emb)
            cnt.index_add_(0, inv, torch.ones((flat_emb.size(0), 1), device=device, dtype=token_hidden_states.dtype))
            sent_emb = sum_emb / (cnt + eps)
        elif pooling == "last":
            sent_emb = torch.zeros((num_sent, hidden), device=device, dtype=token_hidden_states.dtype)
            sent_emb.index_copy_(0, idx_last, flat_emb_last)
        else:
            raise ValueError(f"Unknown sentence_adv.pooling: {pooling}")

        flat_sample_idx = torch.arange(bs, device=device).unsqueeze(1).expand(bs, seq_len).reshape(-1)
        flat_sample_last = flat_sample_idx[flat_last]
        sent_sample_idx = torch.zeros((num_sent,), device=device, dtype=torch.long)
        sent_sample_idx.index_copy_(0, idx_last, flat_sample_last)

    num_sent = sent_emb.shape[0]
    if num_sent == 0:
        return torch.zeros((bs, seq_len), device=device, dtype=sent_emb.dtype), metrics

    sent_first_pos = torch.full((num_sent,), seq_len, device=device, dtype=torch.long)
    try:
        sent_first_pos = sent_first_pos.scatter_reduce(0, inv, flat_pos, reduce="amin", include_self=True)
    except Exception:
        for i in range(num_sent):
            m = inv == i
            if torch.any(m):
                sent_first_pos[i] = flat_pos[m].min()

    sent_batch_sum = torch.zeros((num_sent,), device=device, dtype=torch.float)
    sent_batch_sum.index_add_(0, inv, flat_batch.to(torch.float))
    sent_counts = torch.zeros((num_sent,), device=device, dtype=torch.float)
    sent_counts.index_add_(0, inv, torch.ones_like(flat_pos, dtype=torch.float))
    sent_batch = (sent_batch_sum / (sent_counts + eps)).round().long()

    sent_bucket = torch.full((num_sent,), -1, device=device, dtype=torch.long)
    sent_count_per_sample = torch.zeros((bs,), device=device, dtype=torch.long)
    for b in range(bs):
        mask = sent_batch == b
        if not torch.any(mask):
            continue
        pos = sent_first_pos[mask]
        order = torch.argsort(pos)
        ranks = torch.zeros_like(order)
        ranks[order] = torch.arange(order.numel(), device=device)
        n_sent = int(order.numel())
        sent_count_per_sample[b] = n_sent
        bucket = (ranks * bucket_count) // max(n_sent, 1)
        sent_bucket[mask] = bucket

    scores = token_level_rewards.to(device).sum(dim=-1)
    correct = scores > float(getattr(sentence_adv_cfg, "correctness_threshold", 0.0))

    sent_emb = F.normalize(sent_emb, dim=-1)
    group_ids = as_torch_index(index, device=device)
    sent_group = group_ids[sent_sample_idx]
    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0

    sent_adv = torch.zeros((num_sent,), device=device, dtype=sent_emb.dtype)
    bucket_sent_count = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    pos_center_pos_sum = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    pos_center_pos_cnt = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    pos_center_neg_sum = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    pos_center_neg_cnt = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    neg_center_neg_sum = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    neg_center_neg_cnt = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    neg_center_pos_sum = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    neg_center_pos_cnt = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    center_cos_sum = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    center_cos_cnt = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    sep_sum = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    sep_cnt = torch.zeros((bucket_count,), device=device, dtype=torch.float)
    with torch.no_grad():
        for g in range(num_groups):
            group_mask = sent_group == g
            if not torch.any(group_mask):
                continue
            for b in range(bucket_count):
                mask = group_mask & (sent_bucket == b)
                if not torch.any(mask):
                    continue
                bucket_sent_count[b] += mask.float().sum()
                sample_idx_g = sent_sample_idx[mask]
                pos_mask = correct[sample_idx_g]
                neg_mask = ~pos_mask
                if not torch.any(pos_mask) or not torch.any(neg_mask):
                    continue
                E = sent_emb[mask]
                pos_center = F.normalize(E[pos_mask].mean(dim=0, keepdim=False), dim=-1)
                neg_center = F.normalize(E[neg_mask].mean(dim=0, keepdim=False), dim=-1)
                sim_pos_raw = (E * pos_center).sum(dim=-1)
                sim_neg_raw = (E * neg_center).sum(dim=-1)
                sim_pos = sim_pos_raw / tau
                sim_neg = sim_neg_raw / tau
                A = sim_pos - sim_neg
                if normalize_adv and A.numel() > 1:
                    A = (A - A.mean()) / (A.std(unbiased=False) + eps)
                sent_adv[mask] = A

                if metrics_enable:
                    pos_center_pos_sum[b] += sim_pos_raw[pos_mask].sum()
                    pos_center_pos_cnt[b] += pos_mask.float().sum()
                    pos_center_neg_sum[b] += sim_pos_raw[neg_mask].sum()
                    pos_center_neg_cnt[b] += neg_mask.float().sum()
                    neg_center_neg_sum[b] += sim_neg_raw[neg_mask].sum()
                    neg_center_neg_cnt[b] += neg_mask.float().sum()
                    neg_center_pos_sum[b] += sim_neg_raw[pos_mask].sum()
                    neg_center_pos_cnt[b] += pos_mask.float().sum()
                    center_cos_sum[b] += (pos_center * neg_center).sum()
                    center_cos_cnt[b] += 1.0

                    pos_mean = sim_pos_raw[pos_mask].mean()
                    neg_mean = sim_neg_raw[neg_mask].mean()
                    pos_to_neg_mean = sim_neg_raw[pos_mask].mean()
                    neg_to_pos_mean = sim_pos_raw[neg_mask].mean()
                    sep = 0.5 * (pos_mean + neg_mean) - 0.5 * (pos_to_neg_mean + neg_to_pos_mean)
                    sep_sum[b] += sep
                    sep_cnt[b] += 1.0

    flat_adv = torch.zeros_like(sentence_ids.view(-1), dtype=sent_emb.dtype)
    flat_adv[flat_valid] = sent_adv[inv]
    sent_adv_tokens = flat_adv.view(bs, seq_len) * response_mask

    if metrics_enable:
        metrics["sentence_adv/bucket_count"] = float(bucket_count)
        for b in range(bucket_count):
            mask = sent_bucket == b
            count = int(mask.sum().item())
            metrics[f"sentence_adv/bucket_{b}/sent_count"] = float(count)
            if count > 0:
                vals = sent_adv[mask]
                metrics[f"sentence_adv/bucket_{b}/adv_mean"] = float(vals.mean().item())
                metrics[f"sentence_adv/bucket_{b}/adv_std"] = float(vals.std(unbiased=False).item())
            else:
                metrics[f"sentence_adv/bucket_{b}/adv_mean"] = 0.0
                metrics[f"sentence_adv/bucket_{b}/adv_std"] = 0.0

            def _safe_div(num: torch.Tensor, den: torch.Tensor) -> float:
                if den.item() <= 0:
                    return 0.0
                return float((num / den).item())

            metrics[f"sentence_adv/bucket_{b}/sim_pos_center_pos_mean"] = _safe_div(
                pos_center_pos_sum[b], pos_center_pos_cnt[b]
            )
            metrics[f"sentence_adv/bucket_{b}/sim_pos_center_neg_mean"] = _safe_div(
                pos_center_neg_sum[b], pos_center_neg_cnt[b]
            )
            metrics[f"sentence_adv/bucket_{b}/sim_neg_center_neg_mean"] = _safe_div(
                neg_center_neg_sum[b], neg_center_neg_cnt[b]
            )
            metrics[f"sentence_adv/bucket_{b}/sim_neg_center_pos_mean"] = _safe_div(
                neg_center_pos_sum[b], neg_center_pos_cnt[b]
            )
            metrics[f"sentence_adv/bucket_{b}/center_cos"] = _safe_div(center_cos_sum[b], center_cos_cnt[b])
            metrics[f"sentence_adv/bucket_{b}/sep"] = _safe_div(sep_sum[b], sep_cnt[b])
        metrics["sentence_adv/adv_mean"] = float(sent_adv.mean().item()) if num_sent > 0 else 0.0
        metrics["sentence_adv/adv_std"] = float(sent_adv.std(unbiased=False).item()) if num_sent > 1 else 0.0

    return sent_adv_tokens, metrics


def compute_slpa_advantage(
    sentence_embeddings: torch.Tensor | None,
    sentence_unique_ids: torch.Tensor | None,
    sentence_sample_idx: torch.Tensor | None,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    token_level_rewards: torch.Tensor,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Sentence-Level Process Advantage (SLPA).

    Estimates a state-dependent baseline V_k at each sentence boundary via
    kernel-weighted reward regression over group-internal rollouts, then
    computes temporal differences Δ_k = V_k - V_{k-1} as sentence credit.

    This generalises GRPO's constant baseline μ_group to a sentence-level
    value function, reusing the N group rollouts as Monte Carlo samples
    at zero additional inference cost.

    Mathematical formulation:
        K_emb(k, l) = exp(cosine(h_k, h_l) / τ_emb)
        K_pos(k, l) = exp(-|pos_k - pos_l|² / (2σ²))
        V_k(i)      = Σ_{j≠i} Σ_l [K_emb·K_pos · r_j] / Σ_{j≠i} Σ_l [K_emb·K_pos]
        Δ_k(i)      = V_k(i) - V_{k-1}(i)   (V_{-1} = μ_group)

    Returns:
        slpa_advantages: (bs, seq_len) token-level process advantages.
        metrics: diagnostic scalars.
    """
    metrics: dict[str, float] = {}
    bs, seq_len = sentence_ids.shape
    device = sentence_ids.device

    slpa_cfg = getattr(config, "slpa", None)
    if slpa_cfg is None or not getattr(slpa_cfg, "enable", False):
        return torch.zeros((bs, seq_len), device=device), metrics

    if sentence_embeddings is None or sentence_unique_ids is None or sentence_sample_idx is None:
        return torch.zeros((bs, seq_len), device=device), metrics

    tau_emb = float(getattr(slpa_cfg, "tau_emb", 0.1))
    sigma_pos = float(getattr(slpa_cfg, "sigma_pos", 1.0))
    do_normalize = bool(getattr(slpa_cfg, "normalize", True))
    eps = float(getattr(slpa_cfg, "eps", 1e-8))
    metrics_enable = bool(getattr(slpa_cfg, "metrics_enable", True))

    sent_emb = F.normalize(sentence_embeddings.float().to(device), dim=-1)
    unique_sid = sentence_unique_ids.to(device)
    sent_sample = sentence_sample_idx.to(device)
    num_sent = sent_emb.shape[0]
    if num_sent == 0:
        return torch.zeros((bs, seq_len), device=device), metrics

    # Scalar rewards per sample
    scores = token_level_rewards.to(device).sum(dim=-1)  # (bs,)
    group_ids = as_torch_index(index, device=device)

    # ---- Compute local sentence rank within each sample (vectorised) ----
    sort_key = sent_sample.double() * (unique_sid.max().double() + 1) + unique_sid.double()
    _, perm = torch.sort(sort_key)
    sorted_sample = sent_sample[perm]

    sent_count = torch.zeros(bs, device=device, dtype=torch.long)
    sent_count.scatter_add_(0, sent_sample, torch.ones(num_sent, device=device, dtype=torch.long))

    global_rank = torch.arange(num_sent, device=device, dtype=torch.long)
    first_pos_sorted = torch.full((bs,), num_sent, device=device, dtype=torch.long)
    first_pos_sorted.scatter_reduce_(0, sorted_sample, global_rank, reduce="amin", include_self=True)
    local_rank_sorted = global_rank - first_pos_sorted[sorted_sample]

    sent_local_idx = torch.zeros(num_sent, device=device, dtype=torch.long)
    sent_local_idx[perm] = local_rank_sorted

    # Relative position in [0, 1]
    max_idx = (sent_count - 1).clamp(min=1)
    sent_rel_pos = sent_local_idx.float() / max_idx[sent_sample].float()

    # ---- Kernel-weighted V estimation per group ----
    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
    sent_group = group_ids[sent_sample]
    V_estimates = torch.zeros(num_sent, device=device, dtype=torch.float)
    delta = torch.zeros(num_sent, device=device, dtype=torch.float)

    with torch.no_grad():
        for g in range(num_groups):
            g_sent_mask = sent_group == g
            if not torch.any(g_sent_mask):
                continue

            g_sample_mask = group_ids == g
            mu_group = scores[g_sample_mask].mean()

            g_idx = torch.where(g_sent_mask)[0]
            g_emb = sent_emb[g_idx]
            g_sample_ids = sent_sample[g_idx]
            g_rel_pos = sent_rel_pos[g_idx]
            g_rewards = scores[g_sample_ids]
            n_g = g_idx.shape[0]
            if n_g < 2:
                continue

            # Embedding kernel: exp(cosine / τ)
            cos_sim = g_emb @ g_emb.T
            K_emb = torch.exp(cos_sim / tau_emb)

            # Position kernel: exp(-Δpos² / 2σ²)
            pos_diff = g_rel_pos.unsqueeze(1) - g_rel_pos.unsqueeze(0)
            K_pos = torch.exp(-pos_diff.pow(2) / (2 * sigma_pos ** 2))

            # Leave-one-out: mask sentences from the same rollout
            same_rollout = g_sample_ids.unsqueeze(1) == g_sample_ids.unsqueeze(0)
            K = K_emb * K_pos * (~same_rollout).float()

            # V_k(i) = Σ_j K[k,j]·r_j / Σ_j K[k,j]
            weighted_r = K * g_rewards.unsqueeze(0)
            denom = K.sum(dim=1).clamp(min=eps)
            V_k = weighted_r.sum(dim=1) / denom
            V_estimates[g_idx] = V_k

            # ---- Temporal difference per rollout ----
            g_local = sent_local_idx[g_idx]
            for s in g_sample_ids.unique():
                s_mask = g_sample_ids == s
                s_src = g_idx[s_mask]
                s_local = g_local[s_mask]
                s_V = V_k[s_mask]

                order = torch.argsort(s_local)
                V_sorted = s_V[order]
                V_prev = torch.cat([mu_group.unsqueeze(0), V_sorted[:-1]])
                d = V_sorted - V_prev

                inv_order = torch.argsort(order)
                delta[s_src] = d[inv_order]

        # ---- Optional z-score normalisation within group ----
        if do_normalize:
            for g in range(num_groups):
                g_mask = sent_group == g
                if not torch.any(g_mask):
                    continue
                g_d = delta[g_mask]
                if g_d.numel() > 1:
                    delta[g_mask] = (g_d - g_d.mean()) / (g_d.std(unbiased=False) + eps)

    # ---- Map to token level ----
    valid = (response_mask > 0) & (sentence_ids >= 0)
    flat_valid = valid.view(-1)
    flat_sid = sentence_ids.view(-1)[flat_valid]
    inv = torch.searchsorted(unique_sid, flat_sid)

    flat_adv = torch.zeros(bs * seq_len, device=device, dtype=torch.float)
    flat_adv[flat_valid] = delta[inv]
    slpa_adv = flat_adv.view(bs, seq_len) * response_mask

    if metrics_enable:
        metrics["slpa/V_mean"] = float(V_estimates.mean().item())
        metrics["slpa/V_std"] = float(V_estimates.std(unbiased=False).item()) if num_sent > 1 else 0.0
        metrics["slpa/delta_mean"] = float(delta.mean().item())
        metrics["slpa/delta_std"] = float(delta.std(unbiased=False).item()) if num_sent > 1 else 0.0
        metrics["slpa/delta_abs_mean"] = float(delta.abs().mean().item())
        metrics["slpa/num_sentences"] = float(num_sent)
        metrics["slpa/num_groups"] = float(num_groups)

    return slpa_adv, metrics


def compute_scr_advantage(
    sentence_embeddings: torch.Tensor | None,
    sentence_unique_ids: torch.Tensor | None,
    sentence_sample_idx: torch.Tensor | None,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    token_level_rewards: torch.Tensor,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Sentence Contrastive Reward (SCR).

    Builds soft reward-weighted embedding centres with leave-one-out,
    then scores each sentence by contrastive affinity:
        w_j^+ = softmax(r_j / τ_r),  w_j^- = softmax(-r_j / τ_r)
        C^+(i) = Σ_{j≠i} w_j^+ · ē_j,  C^-(i) = Σ_{j≠i} w_j^- · ē_j
        SCR_k  = cos(h_k, C^+) / τ_s - cos(h_k, C^-) / τ_s

    Unlike binary correct/incorrect centres, SCR uses reward-proportional
    softmax weights, producing a smooth discriminative signal.

    Returns:
        scr_advantages: (bs, seq_len) token-level contrastive scores.
        metrics: diagnostic scalars.
    """
    metrics: dict[str, float] = {}
    bs, seq_len = sentence_ids.shape
    device = sentence_ids.device

    scr_cfg = getattr(config, "scr", None)
    if scr_cfg is None or not getattr(scr_cfg, "enable", False):
        return torch.zeros((bs, seq_len), device=device), metrics

    if sentence_embeddings is None or sentence_unique_ids is None or sentence_sample_idx is None:
        return torch.zeros((bs, seq_len), device=device), metrics

    tau_reward = float(getattr(scr_cfg, "tau_reward", 1.0))
    tau_sim = float(getattr(scr_cfg, "tau_sim", 0.1))
    do_normalize = bool(getattr(scr_cfg, "normalize", True))
    eps = float(getattr(scr_cfg, "eps", 1e-8))
    metrics_enable = bool(getattr(scr_cfg, "metrics_enable", True))

    sent_emb = F.normalize(sentence_embeddings.float().to(device), dim=-1)
    unique_sid = sentence_unique_ids.to(device)
    sent_sample = sentence_sample_idx.to(device)
    num_sent = sent_emb.shape[0]
    if num_sent == 0:
        return torch.zeros((bs, seq_len), device=device), metrics

    scores = token_level_rewards.to(device).sum(dim=-1)  # (bs,)
    group_ids = as_torch_index(index, device=device)

    # Per-rollout mean embedding
    hidden_dim = sent_emb.shape[1]
    rollout_emb_sum = torch.zeros((bs, hidden_dim), device=device, dtype=torch.float)
    rollout_emb_sum.index_add_(0, sent_sample, sent_emb)
    sent_count = torch.zeros(bs, device=device, dtype=torch.long)
    sent_count.scatter_add_(0, sent_sample, torch.ones(num_sent, device=device, dtype=torch.long))
    rollout_emb = F.normalize(rollout_emb_sum / sent_count.unsqueeze(1).clamp(min=1).float(), dim=-1)

    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
    sent_group = group_ids[sent_sample]
    scr_scores = torch.zeros(num_sent, device=device, dtype=torch.float)

    with torch.no_grad():
        for g in range(num_groups):
            g_sample_mask = group_ids == g
            if not torch.any(g_sample_mask):
                continue

            g_sample_idx = torch.where(g_sample_mask)[0]
            n_rollouts = g_sample_idx.shape[0]
            if n_rollouts < 2:
                continue

            g_rewards = scores[g_sample_idx]           # (n_rollouts,)
            g_rollout_emb = rollout_emb[g_sample_idx]  # (n_rollouts, hidden)

            # Leave-one-out weighted centres for each rollout
            for ri in range(n_rollouts):
                sample_i = g_sample_idx[ri]
                loo_mask = torch.ones(n_rollouts, device=device, dtype=torch.bool)
                loo_mask[ri] = False
                loo_r = g_rewards[loo_mask]
                loo_e = g_rollout_emb[loo_mask]

                w_pos = torch.softmax(loo_r / tau_reward, dim=0)
                w_neg = torch.softmax(-loo_r / tau_reward, dim=0)

                C_pos = F.normalize((w_pos.unsqueeze(1) * loo_e).sum(dim=0), dim=-1)
                C_neg = F.normalize((w_neg.unsqueeze(1) * loo_e).sum(dim=0), dim=-1)

                sent_mask_i = sent_sample == sample_i
                if not torch.any(sent_mask_i):
                    continue
                h_i = sent_emb[sent_mask_i]

                sim_pos = (h_i * C_pos).sum(dim=-1) / tau_sim
                sim_neg = (h_i * C_neg).sum(dim=-1) / tau_sim
                scr_scores[sent_mask_i] = sim_pos - sim_neg

        # z-score within group
        if do_normalize:
            for g in range(num_groups):
                g_mask = sent_group == g
                if not torch.any(g_mask):
                    continue
                g_s = scr_scores[g_mask]
                if g_s.numel() > 1:
                    scr_scores[g_mask] = (g_s - g_s.mean()) / (g_s.std(unbiased=False) + eps)

    # Map to token level
    valid = (response_mask > 0) & (sentence_ids >= 0)
    flat_valid = valid.view(-1)
    flat_sid = sentence_ids.view(-1)[flat_valid]
    inv = torch.searchsorted(unique_sid, flat_sid)

    flat_adv = torch.zeros(bs * seq_len, device=device, dtype=torch.float)
    flat_adv[flat_valid] = scr_scores[inv]
    scr_adv = flat_adv.view(bs, seq_len) * response_mask

    if metrics_enable:
        metrics["scr/score_mean"] = float(scr_scores.mean().item())
        metrics["scr/score_std"] = float(scr_scores.std(unbiased=False).item()) if num_sent > 1 else 0.0
        metrics["scr/num_sentences"] = float(num_sent)

    return scr_adv, metrics


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("sentencepo")
def compute_policy_loss_sentencepo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    sentence_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sentence-level policy loss (SentencePO) with adaptive clipping.

    This implementation computes sentence-level log-ratio
    $\Delta_s = \frac{1}{|s|} \sum_{t \in s} (\log p_\theta - \log p_{\theta_{old}})$,
    and uses $\rho_s = \exp(\Delta_s)$ for PPO-style clipping.

    Adaptive log-clip radius $c_s$ is computed from sentence PPL (under old policy)
    and sentence length, following:
    $c_s = c_0 \cdot \text{clip}(1 + \lambda_{ppl} z_{ppl} - \lambda_{len} z_{len}, c_{min}, c_{max})$,
    where $z_{ppl}, z_{len}$ are tanh-normalized statistics over the batch.

    The training objective uses sentence-level averaging per sample.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # Prefer runtime-provided sentence_ids; fall back to config if absent.
    if sentence_ids is None and getattr(config, "policy_loss", None) is not None:
        sentence_ids = getattr(config.policy_loss, "sentence_ids", None)
    if sentence_ids is None:
        raise ValueError(
            "SentencePO requires non-empty `sentence_ids` tensor; "
            "please ensure dataset/rollout populates `sentence_ids`."
        )

    sentence_ids = sentence_ids.to(log_prob.device)

    policy_cfg = getattr(config, "policy_loss", None)
    eps_base = getattr(policy_cfg, "sentencepo_eps_base", 0.2)
    lambda_ppl = getattr(policy_cfg, "sentencepo_lambda_ppl", 0.5)
    lambda_len = getattr(policy_cfg, "sentencepo_lambda_len", 0.5)
    cmin = getattr(policy_cfg, "sentencepo_cmin", 0.5)
    cmax = getattr(policy_cfg, "sentencepo_cmax", 1.5)
    stats_eps = getattr(policy_cfg, "sentencepo_stats_eps", 1e-6)

    negative_approx_kl = log_prob - old_log_prob

    # Flatten for grouping; sentence_ids are expected to be already batch-offset upstream
    bs, seq_len = log_prob.shape
    flat_logp = log_prob.reshape(-1)
    flat_old = old_log_prob.reshape(-1)
    flat_mask = response_mask.view(-1)
    flat_sid = sentence_ids.view(-1)

    valid = (flat_mask > 0) & (flat_sid >= 0)
    if not torch.any(valid):
        pg_loss = log_prob.sum() * 0.0
        pg_metrics: dict[str, Any] = {
            "actor/pg_clipfrac": 0.0,  # Python float，不是 tensor
            "actor/ppo_kl": 0.0,
            "actor/pg_clipfrac_lower": 0.0,
            "sentencepo/valid_ratio": 0.0,
        }
        return pg_loss, pg_metrics

    flat_logp_valid = flat_logp[valid]
    flat_old_valid = flat_old[valid]
    flat_sid_valid = flat_sid[valid]

    unique_sid, inv = torch.unique(flat_sid_valid, return_inverse=True)
    num_sent = unique_sid.numel()
    ones = torch.ones_like(flat_logp_valid, dtype=flat_logp_valid.dtype)

    # Sentence length (token count)
    cnt = torch.zeros(num_sent, device=log_prob.device, dtype=flat_logp_valid.dtype)
    cnt.index_add_(0, inv, ones)

    # Sentence-level log-ratio: mean(logp_new - logp_old)
    delta_sum = torch.zeros(num_sent, device=log_prob.device, dtype=flat_logp_valid.dtype)
    delta_sum.index_add_(0, inv, flat_logp_valid - flat_old_valid)
    delta_sent = delta_sum / (cnt + 1e-8)
    rho_sent = torch.exp(delta_sent)

    # Sentence PPL under old policy: exp(-mean(logp_old))
    old_sum = torch.zeros(num_sent, device=log_prob.device, dtype=flat_logp_valid.dtype)
    old_sum.index_add_(0, inv, flat_old_valid)
    old_mean = old_sum / (cnt + 1e-8)
    ppl_sent = torch.exp(-old_mean)

    # Adaptive log-clip radius based on log(ppl) and log(length)
    log_ppl = (-old_mean).float()
    log_len = torch.log(cnt.float().clamp_min(1.0))

    mu_ppl = log_ppl.detach().mean()
    sigma_ppl = log_ppl.detach().std(unbiased=False).clamp_min(stats_eps)
    mu_len = log_len.detach().mean()
    sigma_len = log_len.detach().std(unbiased=False).clamp_min(stats_eps)

    z_ppl = torch.tanh((log_ppl - mu_ppl) / sigma_ppl)
    z_len = torch.tanh((log_len - mu_len) / sigma_len)
    scale = (1.0 + lambda_ppl * z_ppl - lambda_len * z_len).clamp(cmin, cmax)

    c0 = torch.log1p(torch.as_tensor(eps_base, device=log_prob.device, dtype=log_prob.dtype))
    c_sent = c0 * scale.to(log_prob.dtype)
    lower = torch.exp(-c_sent)
    upper = torch.exp(c_sent)

    # Sequence-level advantage broadcast to sentences
    if advantages.dim() == 2:
        seq_adv = verl_F.masked_mean(advantages, response_mask, axis=-1)
    else:
        seq_adv = advantages
    seq_adv = seq_adv.to(log_prob.dtype)

    # Map each sentence to its batch id
    flat_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    flat_batch = (flat_indices // seq_len).to(torch.float32)
    sent_batch_sum = torch.zeros(num_sent, device=log_prob.device, dtype=flat_batch.dtype)
    sent_batch_sum.index_add_(0, inv, flat_batch)
    sent_batch = (sent_batch_sum / cnt).round().long()

    sent_adv = seq_adv[sent_batch]

    obj1 = rho_sent * sent_adv
    obj2 = torch.clamp(rho_sent, lower, upper) * sent_adv
    sent_obj = torch.minimum(obj1, obj2)

    # Optional rollout importance weights (sentence-level mean)
    if rollout_is_weights is not None:
        flat_w = rollout_is_weights.view(-1)[valid]
        w_sum = torch.zeros(num_sent, device=log_prob.device, dtype=flat_w.dtype)
        w_sum.index_add_(0, inv, flat_w)
        w_mean = w_sum / (cnt + 1e-8)
        sent_obj = sent_obj * w_mean

    sent_loss = -sent_obj

    # Sentence-mean per sample, then batch mean
    loss_sum = torch.zeros(bs, device=log_prob.device, dtype=sent_loss.dtype)
    sent_count = torch.zeros(bs, device=log_prob.device, dtype=sent_loss.dtype)
    loss_sum.index_add_(0, sent_batch, sent_loss)
    sent_count.index_add_(0, sent_batch, torch.ones_like(sent_loss, dtype=sent_loss.dtype))
    loss_per_sample = loss_sum / (sent_count + 1e-8)
    pg_loss = loss_per_sample.mean()

    clipped = (rho_sent < lower) | (rho_sent > upper)
    pg_clipfrac = clipped.float().mean()
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    delta_mean = delta_sent.mean()
    delta_std = delta_sent.std(unbiased=False)
    delta_var = delta_sent.var(unbiased=False)
    kl_sent = -delta_sent
    kl_mean = kl_sent.mean()
    kl_std = kl_sent.std(unbiased=False)

    pg_metrics: dict[str, Any] = {
        "actor/pg_clipfrac": pg_clipfrac.item(),  # .item() 转成 Python float
        "actor/ppo_kl": ppo_kl.item(),
        "actor/pg_clipfrac_lower": 0.0,
        # SentencePO debug stats
        "sentencepo/sent_clip_fraction": pg_clipfrac.item(),
        "sentencepo/mean_abs_delta_sent": delta_sent.abs().mean().item(),
        "sentencepo/delta_sent/mean": delta_mean.item(),
        "sentencepo/delta_sent/std": delta_std.item(),
        "sentencepo/delta_sent/var": delta_var.item(),
        "sentencepo/delta_sent/min": delta_sent.min().item(),
        "sentencepo/delta_sent/max": delta_sent.max().item(),
        "sentencepo/kl_sent/mean": kl_mean.item(),
        "sentencepo/kl_sent/std": kl_std.item(),
        "sentencepo/kl_sent/min": kl_sent.min().item(),
        "sentencepo/kl_sent/max": kl_sent.max().item(),
        "sentencepo/mean_c_sent": c_sent.mean().item(),
        "sentencepo/ppl_sent_mean": ppl_sent.mean().item(),
        "sentencepo/len_sent_mean": cnt.mean().item(),
        "sentencepo/K_i_mean": sent_count.mean().item(),
    }

    # Sentence-level monitoring
    try:
        metrics_level = "full"
        if getattr(config, "policy_loss", None) is not None:
            metrics_level = getattr(config.policy_loss, "sentencepo_metrics_level", "full")
        sentencepo_metrics = compute_sentencepo_metrics(
            sentence_ids=sentence_ids,
            response_mask=response_mask,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            hist_enable=False,
            metrics_level=metrics_level,
        )
        # 确保这些 metrics 也转成 Python 标量
        for k, v in sentencepo_metrics.items():
            if isinstance(v, torch.Tensor):
                pg_metrics[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
            else:
                pg_metrics[k] = v
    except Exception:
        pass

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout importance sampling weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of rollout_is_level
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
