# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import numpy as np
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.metric_utils import compute_sentencepo_metrics
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.roles.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        cu_seqlens = log_prob.offsets()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
        loss_mask_flatten[cu_seqlens[1:] - 1] = 0
        loss = -masked_mean(log_prob_flatten, loss_mask_flatten)
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_mean(log_prob, response_mask)

    return loss, {"loss": loss.detach().item()}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    log_prob = no_padding_2_padding(log_prob, data)  # (bsz, response_length)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)  # (bsz, response_length)

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
    )

    metrics.update(
        {
            "pg_loss": pg_loss.detach().item(),
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
    )

    # Clip diagnostics + CPR/CNR + correct/wrong response stats
    if loss_mode in {"gspo", "vanilla", "grpo"}:
        neg_kl = log_prob - old_log_prob
        clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
        clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

        if loss_mode == "gspo":
            seq_lens = response_mask.sum(-1).clamp(min=1)
            seq_log_ratio = (neg_kl * response_mask).sum(-1) / seq_lens
            seq_ratio = torch.exp(seq_log_ratio)
            clip_resp = (seq_ratio < (1 - clip_ratio_low)) | (seq_ratio > (1 + clip_ratio_high))
            clip_reason_low = (seq_ratio < (1 - clip_ratio_low))
            clip_reason_high = (seq_ratio > (1 + clip_ratio_high))

            metrics.update(
                {
                    "gspo/seq_ratio/mean": seq_ratio.mean().detach().item(),
                    "gspo/seq_ratio/max": seq_ratio.max().detach().item(),
                    "gspo/seq_ratio/min": seq_ratio.min().detach().item(),
                    "gspo/seq_ratio/std": seq_ratio.std(unbiased=False).detach().item(),
                    "gspo/clipfrac_seq": clip_resp.float().mean().detach().item(),
                    "gspo/clip_reason_low": clip_reason_low.float().mean().detach().item(),
                    "gspo/clip_reason_high": clip_reason_high.float().mean().detach().item(),
                    "gspo/max_abs_logratio_token/mean": neg_kl.abs().masked_select(response_mask).mean().detach().item(),
                    "gspo/max_abs_logratio_token/max": neg_kl.abs().masked_select(response_mask).max().detach().item(),
                }
            )
            prefix = "gspo"
        else:
            ratio = torch.exp(neg_kl)
            resp_any_low = torch.any((ratio < (1 - clip_ratio_low)) & response_mask, dim=-1)
            resp_any_high = torch.any((ratio > (1 + clip_ratio_high)) & response_mask, dim=-1)
            clip_resp = resp_any_low | resp_any_high
            clip_reason_low = resp_any_low
            clip_reason_high = resp_any_high

            metrics.update(
                {
                    "grpo/clipfrac_resp": clip_resp.float().mean().detach().item(),
                    "grpo/clip_reason_low": clip_reason_low.float().mean().detach().item(),
                    "grpo/clip_reason_high": clip_reason_high.float().mean().detach().item(),
                    "grpo/max_abs_logratio_token/mean": neg_kl.abs().masked_select(response_mask).mean().detach().item(),
                    "grpo/max_abs_logratio_token/max": neg_kl.abs().masked_select(response_mask).max().detach().item(),
                }
            )
            prefix = "grpo"

        # Sentence-level metrics for GSPO
        if loss_mode == "gspo" and "sentence_ids" in data:
            metrics_level = "full"
            policy_loss_cfg = getattr(config, "policy_loss", None)
            if policy_loss_cfg is not None:
                metrics_level = getattr(policy_loss_cfg, "sentencepo_metrics_level", "full")
            sentence_metrics = compute_sentencepo_metrics(
                sentence_ids=data["sentence_ids"],
                response_mask=response_mask,
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                entropys=entropy,
                hist_enable=False,
                metrics_level=metrics_level,
                prefix="gspo_sentence",
            )
            metrics.update(sentence_metrics)

        # CPR/CNR and correct vs wrong response stats
        if "token_level_scores" in data:
            rewards = (data["token_level_scores"] * response_mask).sum(-1)
            is_correct = rewards > 0
            is_wrong = ~is_correct

            clipped = clip_resp
            clipped_count = clipped.float().sum().clamp(min=1)
            cpr = (is_correct & clipped).float().sum() / clipped_count
            cnr = (is_wrong & clipped).float().sum() / clipped_count

            metrics.update(
                {
                    f"{prefix}/cpr": cpr.detach().item(),
                    f"{prefix}/cnr": cnr.detach().item(),
                }
            )

            # Response-level aggregates
            resp_len = response_mask.sum(-1).float()
            resp_ppl = None
            if old_log_prob is not None:
                resp_ppl = torch.exp(-(old_log_prob * response_mask).sum(-1) / resp_len.clamp(min=1))
            resp_ent = None
            if entropy is not None:
                resp_ent = (entropy * response_mask).sum(-1) / resp_len.clamp(min=1)
            if "sentence_ids" in data:
                sentence_ids = data["sentence_ids"]
                sent_counts = []
                max_sent_lens = []
                sent_ppl_mean = []
                sent_ppl_var = []
                sent_ent_mean = []
                sent_ent_var = []

                for i in range(sentence_ids.shape[0]):
                    row_mask = response_mask[i]
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
                        if old_log_prob is not None:
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
                        metrics[f"{prefix}/resp_len_mean_{name}"] = resp_len[mask].mean().detach().item()
                        if resp_ppl is not None:
                            metrics[f"{prefix}/resp_ppl_mean_{name}"] = resp_ppl[mask].mean().detach().item()
                        if resp_ent is not None:
                            metrics[f"{prefix}/resp_entropy_mean_{name}"] = resp_ent[mask].mean().detach().item()
                        metrics[f"{prefix}/sent_count_mean_{name}"] = sent_counts_t[mask].mean().detach().item()
                        metrics[f"{prefix}/max_sent_len_mean_{name}"] = max_sent_lens_t[mask].mean().detach().item()
                        metrics[f"{prefix}/sent_ppl_mean_{name}"] = float(np.mean(np.array(sent_ppl_mean)[mask.cpu().numpy()]))
                        metrics[f"{prefix}/sent_ppl_var_{name}"] = float(np.mean(np.array(sent_ppl_var)[mask.cpu().numpy()]))
                        metrics[f"{prefix}/sent_ent_mean_{name}"] = float(np.mean(np.array(sent_ent_mean)[mask.cpu().numpy()]))
                        metrics[f"{prefix}/sent_ent_var_{name}"] = float(np.mean(np.array(sent_ent_var)[mask.cpu().numpy()]))

                _group_stats(is_correct, "correct")
                _group_stats(is_wrong, "wrong")

                # stratify clipped responses by length/sentence count/max sentence length
                bins_cfg = getattr(config, "policy_loss", None)
                bins_cfg = bins_cfg.get("analysis_bins", {}) if bins_cfg is not None else {}
                resp_bins = bins_cfg.get("response_len_bins", [64, 128, 256, 512])
                sent_bins = bins_cfg.get("sentence_count_bins", [4, 8, 16, 32])
                max_sent_bins = bins_cfg.get("max_sentence_len_bins", [32, 64, 128, 256])

                def _bin_metrics(values, bins, label):
                    prev = 0
                    for b in bins:
                        m = (values > prev) & (values <= b)
                        if torch.any(m):
                            clip_m = clipped & m
                            denom = clip_m.float().sum().clamp(min=1)
                            metrics[f"{prefix}/clip_by_{label}/{prev+1}-{b}/rate"] = (
                                clip_m.float().mean().detach().item()
                            )
                            metrics[f"{prefix}/clip_by_{label}/{prev+1}-{b}/cpr"] = (
                                ((is_correct & clip_m).float().sum() / denom).detach().item()
                            )
                            metrics[f"{prefix}/clip_by_{label}/{prev+1}-{b}/cnr"] = (
                                ((is_wrong & clip_m).float().sum() / denom).detach().item()
                            )
                        prev = b
                    m = values > prev
                    if torch.any(m):
                        clip_m = clipped & m
                        denom = clip_m.float().sum().clamp(min=1)
                        metrics[f"{prefix}/clip_by_{label}/{prev+1}+/rate"] = (
                            clip_m.float().mean().detach().item()
                        )
                        metrics[f"{prefix}/clip_by_{label}/{prev+1}+/cpr"] = (
                            ((is_correct & clip_m).float().sum() / denom).detach().item()
                        )
                        metrics[f"{prefix}/clip_by_{label}/{prev+1}+/cnr"] = (
                            ((is_wrong & clip_m).float().sum() / denom).detach().item()
                        )

                _bin_metrics(resp_len, resp_bins, "resp_len")
                _bin_metrics(sent_counts_t, sent_bins, "sent_count")
                _bin_metrics(max_sent_lens_t, max_sent_bins, "max_sent_len")
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    vpreds = model_output["values"]
    vpreds = no_padding_2_padding(vpreds, data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
