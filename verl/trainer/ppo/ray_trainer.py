# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_sentencepo_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import get_response_mask, masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


PUNCTUATION_CHARS = {".", "?", "!", "。", "？", "！"}


def build_sentence_ids_from_responses(
    tokenizer,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    min_sent_tokens: int = 6,
) -> torch.Tensor:
    """Generate sentence ids for response tokens based on punctuation heuristics.

    Args:
        tokenizer: HF tokenizer for decoding individual tokens.
        responses: Tensor of shape (bsz, response_len) containing token ids.
        response_mask: Tensor of same shape indicating valid response tokens (1 for valid, 0 otherwise).

    Returns:
        torch.LongTensor with same shape as ``responses`` assigning non-negative
        sentence ids to valid tokens and ``-1`` elsewhere.
    """

    assert responses.shape == response_mask.shape

    batch_size, seq_len = responses.shape
    sentence_ids = torch.full_like(responses, fill_value=-1, dtype=torch.long)

    min_sent_tokens = max(1, int(min_sent_tokens))

    for b in range(batch_size):
        valid_positions = (response_mask[b] > 0).nonzero(as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue

        sentences: list[list[int]] = []
        current: list[int] = []
        for idx in valid_positions.tolist():
            token_id = int(responses[b, idx].item())
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            current.append(idx)

            if "\n" in token_str or any(ch in token_str for ch in PUNCTUATION_CHARS):
                sentences.append(current)
                current = []

        if current:
            sentences.append(current)

        if not sentences:
            continue

        # Merge short sentences (< min_sent_tokens). Prefer merging into next; if last, merge into previous.
        i = 0
        while i < len(sentences):
            if len(sentences[i]) < min_sent_tokens and len(sentences) > 1:
                if i < len(sentences) - 1:
                    sentences[i + 1] = sentences[i] + sentences[i + 1]
                    sentences.pop(i)
                    continue
                else:
                    sentences[i - 1] = sentences[i - 1] + sentences[i]
                    sentences.pop(i)
                    i = max(i - 1, 0)
                    continue
            i += 1

        for sid, sent in enumerate(sentences):
            for idx in sent:
                sentence_ids[b, idx] = sid

    # Offset ids per sample to avoid cross-sample mixing when flattened
    for b in range(batch_size):
        mask = sentence_ids[b] >= 0
        if mask.any():
            sentence_ids[b, mask] += b * (seq_len + 1)

    return sentence_ids


def compute_response_mask(data: DataProto):
    """Construct a mask highlighting valid response tokens.

    Prefers EOS-aware masking when ``eos_token_id`` is available in ``meta_info``.
    Falls back to masking non-pad tokens otherwise.
    """

    responses = data.batch["responses"]
    eos_token_id = data.meta_info.get("eos_token_id") if data.meta_info is not None else None
    pad_token_id = data.meta_info.get("pad_token_id") if data.meta_info is not None else None

    if pad_token_id is None:
        pad_token_id = 0
    response_mask = (responses != pad_token_id).to(torch.int64)

    if eos_token_id is not None:
        eos_mask = get_response_mask(responses, eos_token=eos_token_id, dtype=torch.int64)
        response_mask = torch.minimum(response_mask, eos_mask)

    return response_mask


def _pool_sentence_embeddings_from_tokens(
    token_hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1e-8,
    *,
    pooling: str = "last",
    response_token_ids: torch.Tensor | None = None,
    token_entropy: torch.Tensor | None = None,
    punct_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Backward-compatible wrapper around verl.utils.sentence_repr.pool_sentence_embeddings."""
    from verl.utils.sentence_repr import pool_sentence_embeddings

    return pool_sentence_embeddings(
        token_hidden_states=token_hidden_states,
        sentence_ids=sentence_ids,
        response_mask=response_mask,
        pooling=pooling,
        response_token_ids=response_token_ids,
        token_entropy=token_entropy,
        punct_token_ids=punct_token_ids,
        eps=eps,
    )


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    progress: float = 0.0,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ---- Sentence-level advantage modules (bucket/compare, SLPA, SCR) ----
    # Shared tensors used by all embedding-based modules.
    # tensordict>=0.6 raises KeyError on missing key without explicit default,
    # which would crash plain GRPO+vanilla runs (no sentence_ids in batch).
    _sentence_ids = data.batch.get("sentence_ids", default=None)
    _response_mask = data.batch.get("response_mask", default=None)
    _index = data.non_tensor_batch.get("uid") if data.non_tensor_batch is not None else None
    _has_sent_data = _sentence_ids is not None and _response_mask is not None and _index is not None

    # (a) Bucket/compare sentence advantage (v1-3 legacy)
    sentence_adv_cfg = getattr(config, "sentence_adv", None) if config is not None else None
    if sentence_adv_cfg is not None and getattr(sentence_adv_cfg, "enable", False) and _has_sent_data:
        token_hidden_states = data.batch.get("token_hidden_states")
        sentence_embeddings = data.batch.get("sentence_embeddings")
        sentence_unique_ids = data.batch.get("sentence_unique_ids")
        sentence_sample_idx = data.batch.get("sentence_sample_idx")
        sentence_adv, sentence_adv_metrics = core_algos.compute_sentence_semantic_advantage(
            token_hidden_states=token_hidden_states,
            sentence_ids=_sentence_ids,
            response_mask=_response_mask,
            index=_index,
            token_level_rewards=data.batch["token_level_rewards"],
            config=config,
            sentence_embeddings=sentence_embeddings,
            sentence_unique_ids=sentence_unique_ids,
            sentence_sample_idx=sentence_sample_idx,
        )
        alpha = float(getattr(sentence_adv_cfg, "alpha", 0.1))
        data.batch["advantages"] = data.batch["advantages"] + alpha * sentence_adv
        data.batch["returns"] = data.batch["returns"] + alpha * sentence_adv
        if sentence_adv_metrics:
            if data.meta_info is None:
                data.meta_info = {}
            data.meta_info["sentence_adv_metrics"] = sentence_adv_metrics

    # Pool token_hidden_states into sentence embeddings for SLPA/SCR
    slpa_cfg = getattr(config, "slpa", None) if config is not None else None
    scr_cfg = getattr(config, "scr", None) if config is not None else None
    _need_pool = (
        _has_sent_data
        and (
            (slpa_cfg is not None and getattr(slpa_cfg, "enable", False))
            or (scr_cfg is not None and getattr(scr_cfg, "enable", False))
        )
    )
    _sent_emb = _sent_uid = _sent_sidx = None
    if _need_pool and "token_hidden_states" in data.batch.keys():
        _token_hs = data.batch["token_hidden_states"]
        # Resolve pooling from the first enabled module's repr config (SLPA wins
        # over SCR if both are on with mismatched configs — the ablation runs
        # them separately, so this only matters in the combined case).
        _pool_cfg = None
        for _c in (slpa_cfg, scr_cfg):
            if _c is not None and getattr(_c, "enable", False):
                _pool_cfg = getattr(_c, "repr", None)
                if _pool_cfg is not None:
                    break
        _pooling = getattr(_pool_cfg, "pooling", "last") if _pool_cfg is not None else "last"
        _resp_ids = data.batch.get("responses") if _pooling == "mean_no_punct" else None
        # Pad responses to full sequence length (sentence_ids covers full seq).
        if _resp_ids is not None and _resp_ids.shape[1] != _sentence_ids.shape[1]:
            _pad = _sentence_ids.shape[1] - _resp_ids.shape[1]
            _resp_ids = torch.nn.functional.pad(_resp_ids, (_pad, 0), value=0)
        _ent = data.batch.get("entropys") if _pooling == "entropy_weighted" else None
        if _ent is not None and _ent.shape[1] != _sentence_ids.shape[1]:
            _pad = _sentence_ids.shape[1] - _ent.shape[1]
            _ent = torch.nn.functional.pad(_ent, (_pad, 0), value=0.0)
        _punct_ids = (data.meta_info or {}).get("punct_token_ids") if _pooling == "mean_no_punct" else None
        with torch.no_grad():
            _pooled = _pool_sentence_embeddings_from_tokens(
                _token_hs,
                _sentence_ids,
                _response_mask,
                pooling=_pooling,
                response_token_ids=_resp_ids,
                token_entropy=_ent,
                punct_token_ids=_punct_ids,
            )
        if _pooled is not None:
            _sent_emb, _sent_uid, _sent_sidx = _pooled

    # (b) SLPA: Sentence-Level Process Advantage
    if slpa_cfg is not None and getattr(slpa_cfg, "enable", False) and _has_sent_data:
        slpa_adv, slpa_metrics = core_algos.compute_slpa_advantage(
            sentence_embeddings=_sent_emb,
            sentence_unique_ids=_sent_uid,
            sentence_sample_idx=_sent_sidx,
            sentence_ids=_sentence_ids,
            response_mask=_response_mask,
            index=_index,
            token_level_rewards=data.batch["token_level_rewards"],
            config=config,
        )
        # Asymmetric fusion: different α for correct vs incorrect rollouts
        _scores = data.batch["token_level_rewards"].sum(dim=-1)
        _thresh = float(getattr(slpa_cfg, "correctness_threshold", 0.0))
        _correct = _scores > _thresh
        _ac = float(getattr(slpa_cfg, "alpha_correct", 0.1))
        _ai = float(getattr(slpa_cfg, "alpha_incorrect", 0.1))
        # Alpha decay: reduce contribution as training progresses
        _decay_mode = str(getattr(slpa_cfg, "alpha_decay", "none"))
        if _decay_mode == "linear":
            _min_ratio = float(getattr(slpa_cfg, "alpha_min_ratio", 0.1))
            _decay_factor = max(_min_ratio, 1.0 - progress * (1.0 - _min_ratio))
            _ac *= _decay_factor
            _ai *= _decay_factor
        _alpha = torch.where(_correct, _ac, _ai).unsqueeze(-1)
        data.batch["advantages"] = data.batch["advantages"] + _alpha * slpa_adv
        data.batch["returns"] = data.batch["returns"] + _alpha * slpa_adv
        if slpa_metrics:
            if data.meta_info is None:
                data.meta_info = {}
            data.meta_info["slpa_metrics"] = slpa_metrics

    # (c) SCR: Sentence Contrastive Reward
    if scr_cfg is not None and getattr(scr_cfg, "enable", False) and _has_sent_data:
        scr_adv, scr_metrics = core_algos.compute_scr_advantage(
            sentence_embeddings=_sent_emb,
            sentence_unique_ids=_sent_uid,
            sentence_sample_idx=_sent_sidx,
            sentence_ids=_sentence_ids,
            response_mask=_response_mask,
            index=_index,
            token_level_rewards=data.batch["token_level_rewards"],
            config=config,
        )
        _scores = data.batch["token_level_rewards"].sum(dim=-1)
        _thresh = float(getattr(scr_cfg, "correctness_threshold", 0.0))
        _correct = _scores > _thresh
        _ac = float(getattr(scr_cfg, "alpha_correct", 0.05))
        _ai = float(getattr(scr_cfg, "alpha_incorrect", 0.05))
        # Alpha decay: reduce contribution as training progresses
        _decay_mode = str(getattr(scr_cfg, "alpha_decay", "none"))
        if _decay_mode == "linear":
            _min_ratio = float(getattr(scr_cfg, "alpha_min_ratio", 0.1))
            _decay_factor = max(_min_ratio, 1.0 - progress * (1.0 - _min_ratio))
            _ac *= _decay_factor
            _ai *= _decay_factor
        _alpha = torch.where(_correct, _ac, _ai).unsqueeze(-1)
        data.batch["advantages"] = data.batch["advantages"] + _alpha * scr_adv
        data.batch["returns"] = data.batch["returns"] + _alpha * scr_adv
        if scr_metrics:
            if data.meta_info is None:
                data.meta_info = {}
            data.meta_info["scr_metrics"] = scr_metrics

    # Cleanup temporary embedding tensors
    data.batch.pop("token_hidden_states", None)
    data.batch.pop("sentence_embeddings", None)
    data.batch.pop("sentence_unique_ids", None)
    data.batch.pop("sentence_sample_idx", None)

    policy_loss_cfg = None
    if config is not None and hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "actor"):
        policy_loss_cfg = getattr(config.actor_rollout_ref.actor, "policy_loss", None)

    if policy_loss_cfg is not None and getattr(policy_loss_cfg, "sentencepo_adv_entropy_enable", False):
        entropys = data.batch.get("entropys")
        sentence_ids = data.batch.get("sentence_ids")
        response_mask = data.batch.get("response_mask")
        if entropys is not None and sentence_ids is not None and response_mask is not None:
            advantages = core_algos.apply_sentence_entropy_advantage(
                advantages=data.batch["advantages"],
                entropys=entropys,
                sentence_ids=sentence_ids,
                response_mask=response_mask,
                alpha_pos=float(getattr(policy_loss_cfg, "sentencepo_adv_entropy_alpha_pos", 0.1)),
                alpha_neg=float(getattr(policy_loss_cfg, "sentencepo_adv_entropy_alpha_neg", 0.0)),
                norm=str(getattr(policy_loss_cfg, "sentencepo_adv_entropy_norm", "zscore")),
                clip=float(getattr(policy_loss_cfg, "sentencepo_adv_entropy_clip", 2.0)),
                eps=float(getattr(policy_loss_cfg, "sentencepo_adv_entropy_eps", 1e-6)),
            )
            data.batch["advantages"] = advantages
        data.batch.pop("entropys", None)
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        # Lazily-built cache of punctuation token ids (used by mean_no_punct pooling).
        self._punct_token_ids_cache: torch.Tensor | None = None
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _maybe_build_punct_token_ids(self) -> "torch.Tensor | None":
        """Build (and cache) punctuation token ids only if a module asks for mean_no_punct pooling."""
        if self._punct_token_ids_cache is not None:
            return self._punct_token_ids_cache
        algo = self.config.algorithm
        needed = False
        for cfg_name in ("sentence_adv", "slpa", "scr"):
            cfg = getattr(algo, cfg_name, None)
            if cfg is None or not bool(getattr(cfg, "enable", False)):
                continue
            repr_cfg = getattr(cfg, "repr", None)
            if repr_cfg is not None and getattr(repr_cfg, "pooling", "last") == "mean_no_punct":
                needed = True
                break
        if not needed:
            return None
        from verl.utils.sentence_repr import build_punct_token_ids

        self._punct_token_ids_cache = build_punct_token_ids(self.tokenizer)
        return self._punct_token_ids_cache

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _build_sentence_analysis_records(
        self,
        batch: DataProto,
        log_prob_new: torch.Tensor,
        old_log_prob: torch.Tensor | None,
        entropys: torch.Tensor | None,
        ref_log_prob: torch.Tensor | None,
        max_samples: int,
        loss_mode: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        sentence_ids = batch.batch.get("sentence_ids")
        if sentence_ids is None:
            return []

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"].to(bool)
        bs = responses.shape[0]
        n = min(bs, max_samples)

        has_old = old_log_prob is not None
        rewards = None
        if "token_level_scores" in batch.batch:
            rewards = (batch.batch["token_level_scores"] * response_mask).sum(-1)

        policy_cfg = getattr(self.config.actor_rollout_ref.actor, "policy_loss", None)
        eps_base = getattr(policy_cfg, "sentencepo_eps_base", 0.2)
        lambda_ppl = getattr(policy_cfg, "sentencepo_lambda_ppl", 0.5)
        lambda_len = getattr(policy_cfg, "sentencepo_lambda_len", 0.5)
        cmin = getattr(policy_cfg, "sentencepo_cmin", 0.5)
        cmax = getattr(policy_cfg, "sentencepo_cmax", 1.5)
        stats_eps = getattr(policy_cfg, "sentencepo_stats_eps", 1e-6)

        clip_ratio_low = self.config.actor_rollout_ref.actor.clip_ratio_low
        if clip_ratio_low is None:
            clip_ratio_low = self.config.actor_rollout_ref.actor.clip_ratio
        clip_ratio_high = self.config.actor_rollout_ref.actor.clip_ratio_high
        if clip_ratio_high is None:
            clip_ratio_high = self.config.actor_rollout_ref.actor.clip_ratio

        # Precompute sentence-level aggregates across batch
        flat_mask = response_mask.view(-1)
        flat_sid = sentence_ids.view(-1)
        valid = (flat_mask > 0) & (flat_sid >= 0)
        if not torch.any(valid):
            return []

        bs, seq_len = responses.shape
        flat_logp_new = log_prob_new.view(-1)[valid]
        flat_logp_old = old_log_prob.view(-1)[valid] if has_old else None
        flat_sid_valid = flat_sid.view(-1)[valid]

        unique_sid, inv = torch.unique(flat_sid_valid, return_inverse=True)
        num_sent = unique_sid.numel()
        ones = torch.ones_like(flat_logp_new, dtype=flat_logp_new.dtype)

        cnt = torch.zeros(num_sent, device=flat_logp_new.device, dtype=flat_logp_new.dtype)
        cnt.index_add_(0, inv, ones)

        delta = None
        delta_sum = None
        delta_mean = None
        old_mean = None
        ppl_sent = None
        if has_old and flat_logp_old is not None:
            delta = flat_logp_new - flat_logp_old
            delta_sum = torch.zeros(num_sent, device=flat_logp_new.device, dtype=flat_logp_new.dtype)
            delta_sum.index_add_(0, inv, delta)
            delta_mean = delta_sum / (cnt + 1e-8)

            old_sum = torch.zeros(num_sent, device=flat_logp_new.device, dtype=flat_logp_new.dtype)
            old_sum.index_add_(0, inv, flat_logp_old)
            old_mean = old_sum / (cnt + 1e-8)
            ppl_sent = torch.exp(-old_mean)

        max_abs = None
        kl_sent = None
        if delta is not None:
            try:
                max_abs = torch.full((num_sent,), -1e9, device=delta.device, dtype=delta.dtype)
                max_abs = max_abs.scatter_reduce(0, inv, delta.abs(), reduce="amax", include_self=True)
            except Exception:
                max_abs = torch.zeros(num_sent, device=delta.device, dtype=delta.dtype)
                for idx in range(num_sent):
                    m = inv == idx
                    if torch.any(m):
                        max_abs[idx] = delta[m].abs().max()

            kl_sent = -delta_mean

        ref_kl = None
        if ref_log_prob is not None:
            flat_ref = ref_log_prob.view(-1)[valid]
            ref_delta = flat_logp_new - flat_ref
            ref_sum = torch.zeros(num_sent, device=flat_logp_new.device, dtype=flat_logp_new.dtype)
            ref_sum.index_add_(0, inv, ref_delta)
            ref_mean = ref_sum / (cnt + 1e-8)
            ref_kl = -ref_mean

        # SentencePO clip bounds (require old log-prob)
        lower = None
        upper = None
        sent_clipped = None
        if has_old and old_mean is not None and delta_mean is not None:
            log_ppl = (-old_mean).float()
            log_len = torch.log(cnt.float().clamp_min(1.0))
            mu_ppl = log_ppl.detach().mean()
            sigma_ppl = log_ppl.detach().std(unbiased=False).clamp_min(stats_eps)
            mu_len = log_len.detach().mean()
            sigma_len = log_len.detach().std(unbiased=False).clamp_min(stats_eps)
            z_ppl = torch.tanh((log_ppl - mu_ppl) / sigma_ppl)
            z_len = torch.tanh((log_len - mu_len) / sigma_len)
            scale = (1.0 + lambda_ppl * z_ppl - lambda_len * z_len).clamp(cmin, cmax)
            c0 = torch.log1p(torch.as_tensor(eps_base, device=delta_mean.device, dtype=delta_mean.dtype))
            c_sent = c0 * scale.to(delta_mean.dtype)
            lower = torch.exp(-c_sent)
            upper = torch.exp(c_sent)
            sent_clipped = (torch.exp(delta_mean) < lower) | (torch.exp(delta_mean) > upper)

        records: list[dict[str, Any]] = []
        for i in range(n):
            row_mask = response_mask[i]
            row_sent = sentence_ids[i]
            row_resp = responses[i]
            row_logp_new = log_prob_new[i]
            row_logp_old = old_log_prob[i] if has_old else None
            row_entropy = entropys[i] if entropys is not None else None

            valid = row_mask & (row_sent >= 0)
            if not torch.any(valid):
                continue

            sent_ids = torch.unique(row_sent[valid])
            sent_ids = sent_ids.sort().values

            seq_len = int(row_mask.sum().item())
            seq_ratio = None
            response_clipped = None
            if row_logp_old is not None:
                seq_log_ratio = ((row_logp_new - row_logp_old) * row_mask).sum() / max(seq_len, 1)
                seq_ratio = float(torch.exp(seq_log_ratio).item())
                if loss_mode == "gspo":
                    response_clipped = (seq_ratio < (1 - clip_ratio_low)) or (seq_ratio > (1 + clip_ratio_high))
                elif loss_mode in {"vanilla", "grpo"}:
                    token_ratio = torch.exp(row_logp_new - row_logp_old)
                    token_ratio = token_ratio[row_mask]
                    response_clipped = bool(
                        torch.any(token_ratio < (1 - clip_ratio_low))
                        or torch.any(token_ratio > (1 + clip_ratio_high))
                    )

            sent_records = []
            sent_ids_list = sent_ids.tolist()
            for sid in sent_ids_list:
                m = (row_sent == sid) & row_mask
                if not torch.any(m):
                    continue
                token_ids = row_resp[m].tolist()
                sent_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

                # Map sentence to precomputed arrays
                sid_idx = (unique_sid == sid).nonzero(as_tuple=False).squeeze(-1)
                if sid_idx.numel() == 0:
                    continue
                sid_idx = int(sid_idx.item())

                sent_record = {
                    "sentence_id": int(sid),
                    "token_count": int(cnt[sid_idx].item()),
                    "text": sent_text,
                    "mean_log_ratio": float(delta_mean[sid_idx].item()) if delta_mean is not None else None,
                    "sum_log_ratio": float(delta_sum[sid_idx].item()) if delta_sum is not None else None,
                    "max_abs_log_ratio": float(max_abs[sid_idx].item()) if max_abs is not None else None,
                    "ppl": float(ppl_sent[sid_idx].item()) if ppl_sent is not None else None,
                    "entropy": float(row_entropy[m].mean().item()) if row_entropy is not None else None,
                    "kl_old": float(kl_sent[sid_idx].item()) if kl_sent is not None else None,
                }
                if ref_kl is not None:
                    sent_record["kl_ref"] = float(ref_kl[sid_idx].item())
                if loss_mode == "sentencepo":
                    if sent_clipped is not None:
                        sent_record["clipped"] = bool(sent_clipped[sid_idx].item())
                        sent_record["clip_lower"] = float(lower[sid_idx].item()) if lower is not None else None
                        sent_record["clip_upper"] = float(upper[sid_idx].item()) if upper is not None else None
                    else:
                        sent_record["clipped"] = None
                        sent_record["clip_lower"] = None
                        sent_record["clip_upper"] = None
                sent_records.append(sent_record)

            def _topk_by(key: str, k: int):
                values = [r.get(key) for r in sent_records]
                idxs = list(range(len(values)))
                idxs = [i for i in idxs if values[i] is not None]
                idxs.sort(key=lambda j: values[j], reverse=True)
                idxs = idxs[: k]
                return [
                    {
                        "sentence_index": int(j),
                        "sentence_id": int(sent_records[j]["sentence_id"]),
                        "value": float(values[j]),
                        "position": float(j / max(len(sent_records), 1)),
                    }
                    for j in idxs
                ]

            sent_lengths = [r["token_count"] for r in sent_records]
            sent_ppls = [r["ppl"] for r in sent_records if r.get("ppl") is not None]
            sent_ents = [r["entropy"] for r in sent_records if r.get("entropy") is not None]
            sent_deltas = [r["sum_log_ratio"] for r in sent_records if r.get("sum_log_ratio") is not None]
            sent_kls = [r["kl_old"] for r in sent_records if r.get("kl_old") is not None]

            response_sentence_clipped = None
            if loss_mode == "sentencepo" and sent_records:
                response_sentence_clipped = any(
                    (r.get("clipped") is True) for r in sent_records if "clipped" in r
                )

            resp_ppl = None
            if row_logp_old is not None:
                resp_ppl = float(torch.exp(-(row_logp_old[row_mask].mean())).item()) if seq_len > 0 else 0.0
            resp_entropy = None
            if row_entropy is not None:
                resp_entropy = float(row_entropy[row_mask].mean().item()) if seq_len > 0 else 0.0

            reward_val = float(rewards[i].item()) if rewards is not None else None

            records.append(
                {
                    "uid": str(batch.non_tensor_batch.get("uid", [""] * bs)[i]),
                    "reward": reward_val,
                    "response_len": seq_len,
                    "response_ppl": resp_ppl,
                    "response_entropy": resp_entropy,
                    "sentence_count": len(sent_records),
                    "seq_ratio": seq_ratio,
                    "response_clipped": response_clipped,
                    "response_sentence_clipped": response_sentence_clipped,
                    "sentence_stats": {
                        "len_mean": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
                        "len_var": float(np.var(sent_lengths)) if sent_lengths else 0.0,
                        "ppl_mean": float(np.mean(sent_ppls)) if sent_ppls else 0.0,
                        "ppl_var": float(np.var(sent_ppls)) if sent_ppls else 0.0,
                        "entropy_mean": float(np.mean(sent_ents)) if sent_ents else 0.0,
                        "entropy_var": float(np.var(sent_ents)) if sent_ents else 0.0,
                        "delta_sum_mean": float(np.mean(sent_deltas)) if sent_deltas else 0.0,
                        "delta_sum_var": float(np.var(sent_deltas)) if sent_deltas else 0.0,
                        "kl_old_mean": float(np.mean(sent_kls)) if sent_kls else 0.0,
                        "kl_old_var": float(np.var(sent_kls)) if sent_kls else 0.0,
                    },
                    "topk": {
                        "ppl": _topk_by("ppl", top_k),
                        "entropy": _topk_by("entropy", top_k),
                        "length": _topk_by("token_count", top_k),
                        "delta_sum": _topk_by("sum_log_ratio", top_k),
                        "kl_old": _topk_by("kl_old", top_k),
                    },
                    "sentences": sent_records,
                }
            )

        return records

    def _dump_sentence_analysis(
        self,
        batch: DataProto,
        log_prob_new: torch.Tensor,
        old_log_prob: torch.Tensor | None,
        entropys: torch.Tensor | None,
        ref_log_prob: torch.Tensor | None,
        dump_path: str,
        max_samples: int,
        loss_mode: str,
        top_k: int,
        group_by_uid: bool,
    ):
        os.makedirs(dump_path, exist_ok=True)
        split_name = "val" if batch.meta_info.get("validate", False) else "train"
        split_dir = os.path.join(dump_path, split_name)
        os.makedirs(split_dir, exist_ok=True)
        records = self._build_sentence_analysis_records(
            batch=batch,
            log_prob_new=log_prob_new,
            old_log_prob=old_log_prob,
            entropys=entropys,
            ref_log_prob=ref_log_prob,
            max_samples=max_samples,
            loss_mode=loss_mode,
            top_k=top_k,
        )
        if not records:
            return

        if group_by_uid:
            by_uid: dict[str, list[dict[str, Any]]] = {}
            for rec in records:
                by_uid.setdefault(rec.get("uid", "unknown"), []).append(rec)
            for uid, recs in by_uid.items():
                uid_dir = os.path.join(split_dir, uid)
                os.makedirs(uid_dir, exist_ok=True)
                filename = os.path.join(uid_dir, f"{self.global_steps}.jsonl")
                with open(filename, "w") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"Dumped sentence analysis to {filename}")
        else:
            filename = os.path.join(split_dir, f"{self.global_steps}.jsonl")
            with open(filename, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Dumped sentence analysis to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_resp_lens = []
        sample_prompt_lens = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            sample_prompt_lens.extend([int((ids != pad_id).sum().item()) for ids in input_ids])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            sample_resp_lens.extend([int((ids != pad_id).sum().item()) for ids in output_ids])

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            # attach rewards and tokenizer ids for downstream analysis
            test_batch.batch["token_level_scores"] = reward_tensor
            test_batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
            test_batch.meta_info["pad_token_id"] = self.tokenizer.pad_token_id

            # Optional sentence analysis on validation samples
            sentence_analysis_cfg = self.config.trainer.get("sentence_analysis", {}) or {}
            enable_sentence_analysis = bool(sentence_analysis_cfg.get("enable", False))
            if enable_sentence_analysis:
                loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
                top_k = int(sentence_analysis_cfg.get("top_k", 3) or 3)
                group_by_uid = bool(sentence_analysis_cfg.get("group_by_uid", True))
                min_sent_tokens = 6
                if hasattr(self.config, "data"):
                    min_sent_tokens = self.config.data.get(
                        "min_sent_tokens", self.config.data.get("sentencepo_min_sent_tokens", 6)
                    )
                response_mask = compute_response_mask(test_batch)
                test_batch.batch["response_mask"] = response_mask
                test_batch.batch["sentence_ids"] = build_sentence_ids_from_responses(
                    tokenizer=self.tokenizer,
                    responses=test_batch.batch["responses"],
                    response_mask=response_mask,
                    min_sent_tokens=min_sent_tokens,
                )
                # compute log-prob with padding to be divisible by dp size
                size_divisor = (
                    self.actor_rollout_wg.world_size
                    if not self.async_rollout_mode
                    else self.config.actor_rollout_ref.rollout.agent.num_workers
                )
                test_batch_padded, pad_size = pad_dataproto_to_divisor(test_batch, size_divisor)
                log_prob_new_dp_padded = self.actor_rollout_wg.compute_log_prob(test_batch_padded)
                log_prob_new_dp = unpad_dataproto(log_prob_new_dp_padded, pad_size=pad_size)
                if "log_probs" not in log_prob_new_dp.batch.keys():
                    raise KeyError(
                        "compute_log_prob output is missing 'log_probs'. Please ensure actor compute_log_prob "
                        "returns log_probs for analysis."
                    )
                log_prob_new = log_prob_new_dp.batch.get("log_probs")
                ent_new = log_prob_new_dp.batch.get("entropys")
                analysis_dir = sentence_analysis_cfg.get("dir", None)
                if analysis_dir is None:
                    analysis_dir = os.path.join(
                        self.config.trainer.get("validation_data_dir", ""), "sentence_analysis"
                    )
                if analysis_dir and log_prob_new is not None:
                    self._dump_sentence_analysis(
                        batch=test_batch,
                        log_prob_new=log_prob_new,
                        old_log_prob=None,
                        entropys=ent_new,
                        ref_log_prob=None,
                        dump_path=analysis_dir,
                        max_samples=int(sentence_analysis_cfg.get("max_samples", 8) or 8),
                        loss_mode=loss_mode,
                        top_k=top_k,
                        group_by_uid=group_by_uid,
                    )

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # response length bucket accuracy on validation
        if sample_resp_lens:
            resp_lens = np.array(sample_resp_lens)
            scores_np = np.array(sample_scores)
            bins = self.config.trainer.get("response_len_bins", [64, 128, 256, 512])
            prev = 0
            for b in bins:
                m = (resp_lens > prev) & (resp_lens <= b)
                if np.any(m):
                    metric_dict[f"val-aux/resp_len_bucket/{prev+1}-{b}/acc"] = scores_np[m].mean().item()
                    metric_dict[f"val-aux/resp_len_bucket/{prev+1}-{b}/count"] = int(m.sum())
                prev = b
            m = resp_lens > prev
            if np.any(m):
                metric_dict[f"val-aux/resp_len_bucket/{prev+1}+/acc"] = scores_np[m].mean().item()
                metric_dict[f"val-aux/resp_len_bucket/{prev+1}+/count"] = int(m.sum())

        # prompt length stratified response length accuracy
        if sample_prompt_lens and sample_resp_lens:
            prompt_lens = np.array(sample_prompt_lens)
            resp_lens = np.array(sample_resp_lens)
            scores_np = np.array(sample_scores)
            prompt_bins = self.config.trainer.get("prompt_len_bins", [64, 128, 256, 512])
            resp_bins = self.config.trainer.get("response_len_bins", [64, 128, 256, 512])
            prev_p = 0
            for pb in prompt_bins:
                mp = (prompt_lens > prev_p) & (prompt_lens <= pb)
                if np.any(mp):
                    prev_r = 0
                    for rb in resp_bins:
                        mr = mp & (resp_lens > prev_r) & (resp_lens <= rb)
                        if np.any(mr):
                            metric_dict[
                                f"val-aux/resp_len_by_prompt_len/{prev_p+1}-{pb}/{prev_r+1}-{rb}/acc"
                            ] = scores_np[mr].mean().item()
                            metric_dict[
                                f"val-aux/resp_len_by_prompt_len/{prev_p+1}-{pb}/{prev_r+1}-{rb}/count"
                            ] = int(mr.sum())
                        prev_r = rb
                    mr = mp & (resp_lens > prev_r)
                    if np.any(mr):
                        metric_dict[
                            f"val-aux/resp_len_by_prompt_len/{prev_p+1}-{pb}/{prev_r+1}+/acc"
                        ] = scores_np[mr].mean().item()
                        metric_dict[
                            f"val-aux/resp_len_by_prompt_len/{prev_p+1}-{pb}/{prev_r+1}+/count"
                        ] = int(mr.sum())
                prev_p = pb

        # within-prompt length vs correctness correlation
        if sample_uids and sample_resp_lens:
            uid_arr = np.array(sample_uids)
            resp_lens = np.array(sample_resp_lens)
            scores_np = np.array(sample_scores)
            corr_vals = []
            for uid in np.unique(uid_arr):
                m = uid_arr == uid
                if m.sum() < 2:
                    continue
                x = resp_lens[m]
                y = scores_np[m]
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                corr = np.corrcoef(x, y)[0, 1]
                if not np.isnan(corr):
                    corr_vals.append(corr)
            if corr_vals:
                metric_dict["val-aux/len_correct_corr/mean"] = float(np.mean(corr_vals))
                metric_dict["val-aux/len_correct_corr/std"] = float(np.std(corr_vals))
                metric_dict["val-aux/len_correct_corr/count"] = int(len(corr_vals))

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        sentencepo_monitor_cfg = self.config.trainer.get("sentencepo_monitor", {}) or {}
        try:
            sentencepo_monitor_cfg = OmegaConf.to_container(sentencepo_monitor_cfg, resolve=True)
        except Exception:
            pass
        sentencepo_monitor_cfg = sentencepo_monitor_cfg or {}
        sp_hist_enable = bool(sentencepo_monitor_cfg.get("hist_enable", False))
        sp_hist_every = int(sentencepo_monitor_cfg.get("hist_every", 0) or 0)
        sp_hist_max_points = int(sentencepo_monitor_cfg.get("hist_max_points", 2048) or 2048)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    sentence_analysis_cfg = self.config.trainer.get("sentence_analysis", {}) or {}
                    enable_sentence_analysis = bool(sentence_analysis_cfg.get("enable", False))
                    sentence_adv_cfg = getattr(self.config.algorithm, "sentence_adv", None)
                    enable_sentence_adv = bool(getattr(sentence_adv_cfg, "enable", False))
                    slpa_cfg = getattr(self.config.algorithm, "slpa", None)
                    scr_cfg = getattr(self.config.algorithm, "scr", None)
                    enable_slpa = bool(getattr(slpa_cfg, "enable", False))
                    enable_scr = bool(getattr(scr_cfg, "enable", False))
                    if (
                        loss_mode in {"sentencepo", "gspo"}
                        or enable_sentence_analysis
                        or enable_sentence_adv
                        or enable_slpa
                        or enable_scr
                    ):
                        if hasattr(self.config, "data"):
                            min_sent_tokens = self.config.data.get(
                                "min_sent_tokens", self.config.data.get("sentencepo_min_sent_tokens", 6)
                            )
                        else:
                            min_sent_tokens = 6
                        if hasattr(self.config, "actor_rollout_ref") and hasattr(
                            self.config.actor_rollout_ref, "actor"
                        ):
                            policy_loss_cfg = getattr(self.config.actor_rollout_ref.actor, "policy_loss", None)
                            if policy_loss_cfg is not None:
                                min_sent_tokens = getattr(policy_loss_cfg, "sentencepo_min_sent_tokens", min_sent_tokens)
                        batch.batch["sentence_ids"] = build_sentence_ids_from_responses(
                            tokenizer=self.tokenizer,
                            responses=batch.batch["responses"],
                            response_mask=batch.batch["response_mask"],
                            min_sent_tokens=min_sent_tokens,
                        )
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        sentence_adv_cfg = getattr(self.config.algorithm, "sentence_adv", None)
                        slpa_cfg = getattr(self.config.algorithm, "slpa", None)
                        scr_cfg = getattr(self.config.algorithm, "scr", None)
                        _need_sent_emb = (
                            (sentence_adv_cfg is not None and getattr(sentence_adv_cfg, "enable", False))
                            or (slpa_cfg is not None and getattr(slpa_cfg, "enable", False))
                            or (scr_cfg is not None and getattr(scr_cfg, "enable", False))
                        )
                        if _need_sent_emb:
                            batch.meta_info["return_hidden_states"] = True
                            batch.meta_info["sentence_adv_pool_only"] = True
                            batch.meta_info["return_last_hidden_state_only"] = True
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        if loss_mode in {"sentencepo", "gspo"} and "sentence_ids" in batch.batch:
                            sentencepo_stats = compute_sentencepo_metrics(
                                sentence_ids=batch.batch["sentence_ids"],
                                response_mask=response_masks,
                                entropys=entropys,
                                hist_enable=sp_hist_enable,
                                hist_every=sp_hist_every,
                                hist_max_points=sp_hist_max_points,
                                global_step=self.global_steps,
                                prefix="sentencepo" if loss_mode == "sentencepo" else "gspo_sentence",
                            )
                            metrics.update(sentencepo_stats)
                        policy_loss_cfg = getattr(self.config.actor_rollout_ref.actor, "policy_loss", None)
                        enable_sent_entropy_adv = bool(
                            getattr(policy_loss_cfg, "sentencepo_adv_entropy_enable", False)
                        ) if policy_loss_cfg is not None else False
                        if enable_sent_entropy_adv:
                            batch.batch["entropys"] = entropys
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        _progress = self.global_steps / max(self.total_training_steps, 1)
                        # Stash punctuation token ids in meta_info if mean_no_punct
                        # pooling is configured. Lazy-built once per trainer.
                        _punct_ids = self._maybe_build_punct_token_ids()
                        if _punct_ids is not None:
                            if batch.meta_info is None:
                                batch.meta_info = {}
                            batch.meta_info["punct_token_ids"] = _punct_ids
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            progress=_progress,
                        )
                        sentence_adv_metrics = batch.meta_info.pop("sentence_adv_metrics", None)
                        if sentence_adv_metrics:
                            metrics.update(sentence_adv_metrics)
                        slpa_metrics = batch.meta_info.pop("slpa_metrics", None)
                        if slpa_metrics:
                            metrics.update(slpa_metrics)
                        scr_metrics = batch.meta_info.pop("scr_metrics", None)
                        if scr_metrics:
                            metrics.update(scr_metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Optional: dump per-sample sentence analysis (after update for fresh log_probs)
                    sentence_analysis_cfg = self.config.trainer.get("sentence_analysis", {}) or {}
                    enable_sentence_analysis = bool(sentence_analysis_cfg.get("enable", False))
                    max_samples = int(sentence_analysis_cfg.get("max_samples", 8) or 8)
                    top_k = int(sentence_analysis_cfg.get("top_k", 3) or 3)
                    group_by_uid = bool(sentence_analysis_cfg.get("group_by_uid", True))
                    if enable_sentence_analysis and "sentence_ids" in batch.batch:
                        analysis_dir = sentence_analysis_cfg.get("dir", None)
                        if analysis_dir is None:
                            analysis_dir = os.path.join(
                                self.config.trainer.get("rollout_data_dir", ""), "sentence_analysis"
                            )
                        if analysis_dir:
                            with marked_timer("sentence_analysis", timing_raw, color="green"):
                                log_prob_new_dp = self.actor_rollout_wg.compute_log_prob(batch)
                                log_prob_new = log_prob_new_dp.batch.get("log_probs")
                                if log_prob_new is None:
                                    log_prob_new = log_prob_new_dp.batch.get("old_log_probs")
                                ent_new = log_prob_new_dp.batch.get("entropys")
                                ref_lp = batch.batch.get("ref_log_prob")
                                old_lp = batch.batch.get("old_log_probs")
                                if old_lp is None:
                                    old_lp = batch.batch.get("log_probs")
                                if log_prob_new is not None and old_lp is not None:
                                    self._dump_sentence_analysis(
                                        batch=batch,
                                        log_prob_new=log_prob_new,
                                        old_log_prob=old_lp,
                                        entropys=ent_new,
                                        ref_log_prob=ref_lp,
                                        dump_path=analysis_dir,
                                        max_samples=max_samples,
                                        loss_mode=loss_mode,
                                        top_k=top_k,
                                        group_by_uid=group_by_uid,
                                    )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
