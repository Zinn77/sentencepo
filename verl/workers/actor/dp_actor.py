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
Single Process Actor
"""

import logging
import os

import numpy as np

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _select_hidden_layer(
    hidden_states_tuple,
    layer_index: "int | list[int]",
) -> torch.Tensor:
    """Return one hidden-state tensor from a HF model's output.hidden_states.

    Accepts either a single int (use that layer) or a list/tuple/ListConfig of
    ints (mean of those layers). HF returns ``num_layers + 1`` tensors (input
    embeddings plus one per block); negative indices follow Python list
    semantics.
    """
    # Bool is a subclass of int — reject explicitly to catch config typos.
    if isinstance(layer_index, bool):
        raise TypeError(f"hidden_layer_index must not be bool, got {layer_index}")
    if isinstance(layer_index, int):
        return hidden_states_tuple[layer_index]
    # OmegaConf ListConfig does not inherit from list, so use a duck-typed check.
    try:
        idx_list = [int(i) for i in layer_index]
    except TypeError as e:
        raise TypeError(
            f"hidden_layer_index must be int or iterable of ints, got {type(layer_index)}"
        ) from e
    if len(idx_list) == 0:
        return hidden_states_tuple[-1]
    if len(idx_list) == 1:
        return hidden_states_tuple[idx_list[0]]
    stacked = torch.stack([hidden_states_tuple[i] for i in idx_list], dim=0)
    return stacked.mean(dim=0)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _find_transformer_blocks(self) -> "nn.ModuleList | list | tuple | None":
        module = getattr(self.actor_module, "_fsdp_wrapped_module", self.actor_module)
        candidates = (
            ("model", "layers"),
            ("model", "h"),
            ("transformer", "h"),
            ("transformer", "layers"),
            ("transformer", "blocks"),
            ("gpt_neox", "layers"),
            ("decoder", "layers"),
        )
        for path in candidates:
            obj = module
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if not ok:
                continue
            if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                return obj
        return None

    def _find_last_transformer_block(self) -> nn.Module | None:
        blocks = self._find_transformer_blocks()
        return blocks[-1] if blocks is not None else None

    def _find_transformer_block(self, layer_idx: int) -> nn.Module | None:
        blocks = self._find_transformer_blocks()
        if blocks is None:
            return None
        try:
            return blocks[layer_idx]
        except IndexError:
            return None

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_hidden_states: bool = False,
        return_last_hidden_state_only: bool = False,
        hidden_layer_index: "int | list[int]" = -1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            entropy: (bs, response_len) or None
            log_probs: (bs, response_len)
            hidden_states: (bs, response_len, hidden) or None
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            hidden_states = None
            hook_handle = None
            last_hidden = None
            use_hook = False
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            # Hook-based extraction is only valid when:
            #   - caller asked for last_hidden_state_only (memory optimization)
            #   - AND we want a single, fixed layer (int, not list)
            # For non-default layers or layer-ensemble, fall back to
            # output_hidden_states=True so we can index into the full stack.
            single_layer = isinstance(hidden_layer_index, int)
            if return_hidden_states and return_last_hidden_state_only and single_layer:
                block = self._find_transformer_block(hidden_layer_index)
                if block is not None:
                    use_hook = True

                    def _hook(_module, _inputs, output):
                        nonlocal last_hidden
                        if isinstance(output, tuple):
                            last_hidden = output[0]
                        else:
                            last_hidden = output

                    hook_handle = block.register_forward_hook(_hook)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if return_hidden_states and not use_hook:
                    extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                hidden_states_rmpad = None
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    if calculate_entropy:
                        entropy_rmpad = output.entropy.squeeze(0)
                    if return_hidden_states and not use_hook and hasattr(output, "hidden_states"):
                        hidden_states_rmpad = _select_hidden_layer(
                            output.hidden_states, hidden_layer_index
                        ).squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    if return_hidden_states and not use_hook and hasattr(output, "hidden_states"):
                        hidden_states_rmpad = _select_hidden_layer(
                            output.hidden_states, hidden_layer_index
                        ).squeeze(0)

                if return_hidden_states and use_hook and last_hidden is not None:
                    hidden_states_rmpad = last_hidden.squeeze(0)

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_hidden_states and hidden_states_rmpad is not None:
                        hidden_states_rmpad = gather_outputs_and_unpad(
                            hidden_states_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if return_hidden_states and hidden_states_rmpad is not None:
                    full_hidden_states = pad_input(
                        hidden_states=hidden_states_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if return_hidden_states and hidden_states_rmpad is not None:
                    hidden_states = full_hidden_states[:, -response_length - 1 : -1, :]
            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if return_hidden_states and not use_hook:
                    extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1] if calculate_entropy else None
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                if return_hidden_states and not use_hook and hasattr(output, "hidden_states"):
                    hidden_states = _select_hidden_layer(
                        output.hidden_states, hidden_layer_index
                    )[:, -response_length - 1 : -1, :]
                if return_hidden_states and use_hook and last_hidden is not None:
                    hidden_states = last_hidden[:, -response_length - 1 : -1, :]

            if hook_handle is not None:
                hook_handle.remove()

            return entropy, log_probs, hidden_states

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, return_hidden_states: bool = False):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        return_last_hidden_state_only = bool(data.meta_info.get("return_last_hidden_state_only", False))
        hidden_layer_index = data.meta_info.get("hidden_layer_index", -1)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        hidden_states_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, hidden_states = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_hidden_states=return_hidden_states,
                    return_last_hidden_state_only=return_last_hidden_state_only,
                    hidden_layer_index=hidden_layer_index,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if return_hidden_states and hidden_states is not None:
                hidden_states_lst.append(hidden_states)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        hidden_states = None
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if return_hidden_states and hidden_states_lst:
                hidden_states = restore_dynamic_batch(torch.concat(hidden_states_lst, dim=0), batch_idx_list)
        elif return_hidden_states and hidden_states_lst:
            hidden_states = torch.concat(hidden_states_lst, dim=0)

        return log_probs, entropys, hidden_states

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Optional pre-computed sentence ids for sentencepo loss
        if "sentence_ids" in data.batch.keys():
            select_keys.append("sentence_ids")
        # Optional token-level scores for CPR/CNR and clip-by stats
        if "token_level_scores" in data.batch.keys():
            select_keys.append("token_level_scores")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    # If using sentencepo loss, also fetch sentence_ids from batch
                    sentence_ids = None
                    if loss_mode == "sentencepo":
                        sentence_ids = model_inputs.get("sentence_ids", None)
                        if sentence_ids is None:
                            raise ValueError(
                                "sentencepo loss_mode requires 'sentence_ids' tensor in batch; "
                                "please ensure rollout populates batch['sentence_ids']."
                            )
                        if sentence_ids.dim() == 1:
                            sentence_ids = sentence_ids.unsqueeze(0)

                        if sentence_ids.shape != response_mask.shape:
                            raise ValueError(
                                "SentencePO expects sentence_ids shape to match response_mask; "
                                f"got {sentence_ids.shape} vs {response_mask.shape}."
                            )

                        sentence_ids = sentence_ids.long()

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    policy_loss_kwargs = dict(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    if loss_mode == "sentencepo":
                        policy_loss_kwargs["sentence_ids"] = sentence_ids

                    # Compute policy loss; some losses return (loss, metrics_dict), others 4-tuple
                    policy_loss_out = policy_loss_fn(**policy_loss_kwargs)
                    if isinstance(policy_loss_out, tuple) and len(policy_loss_out) == 2 and isinstance(
                        policy_loss_out[1], dict
                    ):
                        pg_loss, pg_metrics = policy_loss_out
                        pg_clipfrac = pg_metrics.get("actor/pg_clipfrac", torch.tensor(0.0, device=pg_loss.device))
                        ppo_kl = pg_metrics.get("actor/ppo_kl", torch.tensor(0.0, device=pg_loss.device))
                        pg_clipfrac_lower = pg_metrics.get(
                            "actor/pg_clipfrac_lower", torch.tensor(0.0, device=pg_loss.device)
                        )
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_out

                    # Clip diagnostics + CPR/CNR + correct/wrong response stats (for GSPO/GRPO/vanilla)
                    if loss_mode in {"gspo", "vanilla", "grpo"}:
                        response_mask_bool = response_mask.to(bool)
                        neg_kl = log_prob - old_log_prob
                        clip_ratio_low = (
                            self.config.clip_ratio_low if self.config.clip_ratio_low is not None else self.config.clip_ratio
                        )
                        clip_ratio_high = (
                            self.config.clip_ratio_high if self.config.clip_ratio_high is not None else self.config.clip_ratio
                        )

                        def _init_clip_by_keys(metrics_dict, prefix, bins, label):
                            prev = 0
                            for b in bins:
                                metrics_dict[f"{prefix}/clip_by_{label}/{prev+1}-{b}/rate"] = 0.0
                                metrics_dict[f"{prefix}/clip_by_{label}/{prev+1}-{b}/cpr"] = 0.0
                                metrics_dict[f"{prefix}/clip_by_{label}/{prev+1}-{b}/cnr"] = 0.0
                                prev = b
                            metrics_dict[f"{prefix}/clip_by_{label}/{prev+1}+/rate"] = 0.0
                            metrics_dict[f"{prefix}/clip_by_{label}/{prev+1}+/cpr"] = 0.0
                            metrics_dict[f"{prefix}/clip_by_{label}/{prev+1}+/cnr"] = 0.0

                        diag_metrics = {}
                        if loss_mode == "gspo":
                            prefix = "actor/gspo"
                            diag_metrics.update(
                                {
                                    f"{prefix}/seq_ratio/mean": 0.0,
                                    f"{prefix}/seq_ratio/max": 0.0,
                                    f"{prefix}/seq_ratio/min": 0.0,
                                    f"{prefix}/seq_ratio/std": 0.0,
                                    f"{prefix}/clipfrac_seq": 0.0,
                                    f"{prefix}/clip_reason_low": 0.0,
                                    f"{prefix}/clip_reason_high": 0.0,
                                    f"{prefix}/max_abs_logratio_token/mean": 0.0,
                                    f"{prefix}/max_abs_logratio_token/max": 0.0,
                                }
                            )
                        else:
                            prefix = "actor/grpo"
                            diag_metrics.update(
                                {
                                    f"{prefix}/clipfrac_resp": 0.0,
                                    f"{prefix}/clip_reason_low": 0.0,
                                    f"{prefix}/clip_reason_high": 0.0,
                                    f"{prefix}/max_abs_logratio_token/mean": 0.0,
                                    f"{prefix}/max_abs_logratio_token/max": 0.0,
                                }
                            )

                        # CPR/CNR and correct vs wrong response stats keys (always present)
                        diag_metrics.update(
                            {
                                f"{prefix}/cpr": 0.0,
                                f"{prefix}/cnr": 0.0,
                                f"{prefix}/resp_len_mean_correct": 0.0,
                                f"{prefix}/resp_len_mean_wrong": 0.0,
                                f"{prefix}/resp_ppl_mean_correct": 0.0,
                                f"{prefix}/resp_ppl_mean_wrong": 0.0,
                                f"{prefix}/resp_entropy_mean_correct": 0.0,
                                f"{prefix}/resp_entropy_mean_wrong": 0.0,
                                f"{prefix}/sent_count_mean_correct": 0.0,
                                f"{prefix}/sent_count_mean_wrong": 0.0,
                                f"{prefix}/max_sent_len_mean_correct": 0.0,
                                f"{prefix}/max_sent_len_mean_wrong": 0.0,
                                f"{prefix}/sent_ppl_mean_correct": 0.0,
                                f"{prefix}/sent_ppl_mean_wrong": 0.0,
                                f"{prefix}/sent_ppl_var_correct": 0.0,
                                f"{prefix}/sent_ppl_var_wrong": 0.0,
                                f"{prefix}/sent_ent_mean_correct": 0.0,
                                f"{prefix}/sent_ent_mean_wrong": 0.0,
                                f"{prefix}/sent_ent_var_correct": 0.0,
                                f"{prefix}/sent_ent_var_wrong": 0.0,
                            }
                        )

                        # Prepare clip-by buckets (always present)
                        bins_cfg = getattr(self.config, "policy_loss", None)
                        bins_cfg = bins_cfg.get("analysis_bins", {}) if bins_cfg is not None else {}
                        resp_bins = bins_cfg.get("response_len_bins", [64, 128, 256, 512])
                        sent_bins = bins_cfg.get("sentence_count_bins", [4, 8, 16, 32])
                        max_sent_bins = bins_cfg.get("max_sentence_len_bins", [32, 64, 128, 256])
                        _init_clip_by_keys(diag_metrics, prefix, resp_bins, "resp_len")
                        _init_clip_by_keys(diag_metrics, prefix, sent_bins, "sent_count")
                        _init_clip_by_keys(diag_metrics, prefix, max_sent_bins, "max_sent_len")

                        if loss_mode == "gspo":
                            seq_lens = response_mask_bool.sum(-1).clamp(min=1)
                            seq_log_ratio = (neg_kl * response_mask_bool).sum(-1) / seq_lens
                            seq_ratio = torch.exp(seq_log_ratio)
                            clip_resp = (seq_ratio < (1 - clip_ratio_low)) | (seq_ratio > (1 + clip_ratio_high))
                            clip_reason_low = seq_ratio < (1 - clip_ratio_low)
                            clip_reason_high = seq_ratio > (1 + clip_ratio_high)

                            diag_metrics.update(
                                {
                                    f"{prefix}/seq_ratio/mean": seq_ratio.mean().detach().item(),
                                    f"{prefix}/seq_ratio/max": seq_ratio.max().detach().item(),
                                    f"{prefix}/seq_ratio/min": seq_ratio.min().detach().item(),
                                    f"{prefix}/seq_ratio/std": seq_ratio.std(unbiased=False).detach().item(),
                                    f"{prefix}/clipfrac_seq": clip_resp.float().mean().detach().item(),
                                    f"{prefix}/clip_reason_low": clip_reason_low.float().mean().detach().item(),
                                    f"{prefix}/clip_reason_high": clip_reason_high.float().mean().detach().item(),
                                    f"{prefix}/max_abs_logratio_token/mean": neg_kl.abs()
                                    .masked_select(response_mask_bool)
                                    .mean()
                                    .detach()
                                    .item(),
                                    f"{prefix}/max_abs_logratio_token/max": neg_kl.abs()
                                    .masked_select(response_mask_bool)
                                    .max()
                                    .detach()
                                    .item(),
                                }
                            )
                        else:
                            ratio = torch.exp(neg_kl)
                            resp_any_low = torch.any((ratio < (1 - clip_ratio_low)) & response_mask_bool, dim=-1)
                            resp_any_high = torch.any((ratio > (1 + clip_ratio_high)) & response_mask_bool, dim=-1)
                            clip_resp = resp_any_low | resp_any_high
                            clip_reason_low = resp_any_low
                            clip_reason_high = resp_any_high

                            diag_metrics.update(
                                {
                                    f"{prefix}/clipfrac_resp": clip_resp.float().mean().detach().item(),
                                    f"{prefix}/clip_reason_low": clip_reason_low.float().mean().detach().item(),
                                    f"{prefix}/clip_reason_high": clip_reason_high.float().mean().detach().item(),
                                    f"{prefix}/max_abs_logratio_token/mean": neg_kl.abs()
                                    .masked_select(response_mask_bool)
                                    .mean()
                                    .detach()
                                    .item(),
                                    f"{prefix}/max_abs_logratio_token/max": neg_kl.abs()
                                    .masked_select(response_mask_bool)
                                    .max()
                                    .detach()
                                    .item(),
                                }
                            )

                        # CPR/CNR and correct vs wrong response stats (requires token_level_scores)
                        if "token_level_scores" in model_inputs:
                            rewards = (model_inputs["token_level_scores"] * response_mask_bool).sum(-1)
                            is_correct = rewards > 0
                            is_wrong = ~is_correct

                            clipped = clip_resp
                            clipped_count = clipped.float().sum().clamp(min=1)
                            cpr = (is_correct & clipped).float().sum() / clipped_count
                            cnr = (is_wrong & clipped).float().sum() / clipped_count

                            diag_metrics.update(
                                {
                                    f"{prefix}/cpr": cpr.detach().item(),
                                    f"{prefix}/cnr": cnr.detach().item(),
                                }
                            )

                            # Response-level aggregates (requires sentence_ids)
                            if "sentence_ids" in model_inputs:
                                resp_len = response_mask_bool.sum(-1).float()
                                resp_ppl = torch.exp(-(old_log_prob * response_mask_bool).sum(-1) / resp_len.clamp(min=1))
                                resp_ent = None
                                if entropy is not None:
                                    resp_ent = (entropy * response_mask_bool).sum(-1) / resp_len.clamp(min=1)

                                sentence_ids = model_inputs["sentence_ids"]
                                sent_counts = []
                                max_sent_lens = []
                                sent_ppl_mean = []
                                sent_ppl_var = []
                                sent_ent_mean = []
                                sent_ent_var = []

                                for i in range(sentence_ids.shape[0]):
                                    row_mask = response_mask_bool[i]
                                    row_sid = sentence_ids[i]
                                    row_valid = row_mask & (row_sid >= 0)
                                    if not torch.any(row_valid):
                                        sent_counts.append(0)
                                        max_sent_lens.append(0)
                                        sent_ppl_mean.append(0.0)
                                        sent_ppl_var.append(0.0)
                                        sent_ent_mean.append(0.0)
                                        sent_ent_var.append(0.0)
                                        continue

                                    sids = torch.unique(row_sid[row_valid])
                                    sent_counts.append(int(sids.numel()))
                                    sent_lens = []
                                    sent_ppl = []
                                    sent_ent = []
                                    for sid in sids.tolist():
                                        m = (row_sid == sid) & row_mask
                                        sent_lens.append(int(m.sum().item()))
                                        sent_ppl.append(float(torch.exp(-old_log_prob[i][m].mean()).item()))
                                        if entropy is not None:
                                            sent_ent.append(float(entropy[i][m].mean().item()))
                                    max_sent_lens.append(max(sent_lens) if sent_lens else 0)
                                    if sent_ppl:
                                        sent_ppl_mean.append(float(np.mean(sent_ppl)))
                                        sent_ppl_var.append(float(np.var(sent_ppl)))
                                    else:
                                        sent_ppl_mean.append(0.0)
                                        sent_ppl_var.append(0.0)
                                    if sent_ent:
                                        sent_ent_mean.append(float(np.mean(sent_ent)))
                                        sent_ent_var.append(float(np.var(sent_ent)))
                                    else:
                                        sent_ent_mean.append(0.0)
                                        sent_ent_var.append(0.0)

                                sent_counts_t = torch.tensor(sent_counts, device=resp_len.device, dtype=resp_len.dtype)
                                max_sent_lens_t = torch.tensor(max_sent_lens, device=resp_len.device, dtype=resp_len.dtype)

                                def _group_stats(mask, name):
                                    if torch.any(mask):
                                        diag_metrics[f"{prefix}/resp_len_mean_{name}"] = (
                                            resp_len[mask].mean().detach().item()
                                        )
                                        diag_metrics[f"{prefix}/resp_ppl_mean_{name}"] = (
                                            resp_ppl[mask].mean().detach().item()
                                        )
                                        if resp_ent is not None:
                                            diag_metrics[f"{prefix}/resp_entropy_mean_{name}"] = (
                                                resp_ent[mask].mean().detach().item()
                                            )
                                        diag_metrics[f"{prefix}/sent_count_mean_{name}"] = (
                                            sent_counts_t[mask].mean().detach().item()
                                        )
                                        diag_metrics[f"{prefix}/max_sent_len_mean_{name}"] = (
                                            max_sent_lens_t[mask].mean().detach().item()
                                        )
                                        diag_metrics[f"{prefix}/sent_ppl_mean_{name}"] = float(
                                            np.mean(np.array(sent_ppl_mean)[mask.cpu().numpy()])
                                        )
                                        diag_metrics[f"{prefix}/sent_ppl_var_{name}"] = float(
                                            np.mean(np.array(sent_ppl_var)[mask.cpu().numpy()])
                                        )
                                        diag_metrics[f"{prefix}/sent_ent_mean_{name}"] = float(
                                            np.mean(np.array(sent_ent_mean)[mask.cpu().numpy()])
                                        )
                                        diag_metrics[f"{prefix}/sent_ent_var_{name}"] = float(
                                            np.mean(np.array(sent_ent_var)[mask.cpu().numpy()])
                                        )

                                _group_stats(is_correct, "correct")
                                _group_stats(is_wrong, "wrong")

                                def _bin_metrics(values, bins, label):
                                    prev = 0
                                    for b in bins:
                                        m = (values > prev) & (values <= b)
                                        if torch.any(m):
                                            clip_m = clipped & m
                                            denom = clip_m.float().sum().clamp(min=1)
                                            diag_metrics[f"{prefix}/clip_by_{label}/{prev+1}-{b}/rate"] = (
                                                clip_m.float().mean().detach().item()
                                            )
                                            diag_metrics[f"{prefix}/clip_by_{label}/{prev+1}-{b}/cpr"] = (
                                                ((is_correct & clip_m).float().sum() / denom).detach().item()
                                            )
                                            diag_metrics[f"{prefix}/clip_by_{label}/{prev+1}-{b}/cnr"] = (
                                                ((is_wrong & clip_m).float().sum() / denom).detach().item()
                                            )
                                        prev = b
                                    m = values > prev
                                    if torch.any(m):
                                        clip_m = clipped & m
                                        denom = clip_m.float().sum().clamp(min=1)
                                        diag_metrics[f"{prefix}/clip_by_{label}/{prev+1}+/rate"] = (
                                            clip_m.float().mean().detach().item()
                                        )
                                        diag_metrics[f"{prefix}/clip_by_{label}/{prev+1}+/cpr"] = (
                                            ((is_correct & clip_m).float().sum() / denom).detach().item()
                                        )
                                        diag_metrics[f"{prefix}/clip_by_{label}/{prev+1}+/cnr"] = (
                                            ((is_wrong & clip_m).float().sum() / denom).detach().item()
                                        )

                                _bin_metrics(resp_len, resp_bins, "resp_len")
                                _bin_metrics(sent_counts_t, sent_bins, "sent_count")
                                _bin_metrics(max_sent_lens_t, max_sent_bins, "max_sent_len")

                        micro_batch_metrics.update(diag_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    def _to_scalar(x):
                        if isinstance(x, torch.Tensor):
                            return float(x.detach().item())
                        return float(x)

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": _to_scalar(pg_clipfrac),
                            "actor/ppo_kl": _to_scalar(ppo_kl),
                            "actor/pg_clipfrac_lower": _to_scalar(pg_clipfrac_lower),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
