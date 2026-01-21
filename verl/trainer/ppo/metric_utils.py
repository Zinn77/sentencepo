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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import math
import numpy as np
import torch
import torch.nn.functional as F

from verl import DataProto
from verl.utils import as_torch_index
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val


def compute_sentencepo_metrics(
    *,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    log_prob: torch.Tensor | None = None,
    old_log_prob: torch.Tensor | None = None,
    entropys: torch.Tensor | None = None,
    hist_enable: bool = False,
    hist_every: int = 0,
    global_step: int | None = None,
    hist_max_points: int = 2048,
) -> dict[str, Any]:
    """SentencePO 监控：句子数量、长度、比率与熵的统计，可选直方图。"""

    bs, seq_len = sentence_ids.shape
    device = sentence_ids.device

    offset = torch.arange(bs, device=device).unsqueeze(1) * (seq_len + 1)
    sid_with_offset = sentence_ids + offset

    valid_mask = (response_mask > 0) & (sentence_ids >= 0)
    if not torch.any(valid_mask):
        return {"sentencepo/valid_ratio": 0.0}

    flat_valid = valid_mask.view(-1)
    flat_sid = sid_with_offset.view(-1)[flat_valid]

    unique_sid, inv = torch.unique(flat_sid, return_inverse=True)
    ones = torch.ones_like(inv, dtype=torch.float)
    sent_lens = torch.zeros_like(unique_sid, dtype=torch.float)
    sent_lens.index_add_(0, inv, ones)

    sentence_counts = []
    for b in range(bs):
        row_valid = valid_mask[b]
        if torch.any(row_valid):
            n_sent = torch.unique(sentence_ids[b][row_valid]).numel()
            sentence_counts.append(n_sent)

    def _safe_stat(t: torch.Tensor) -> dict[str, float]:
        if t.numel() == 0:
            return {}
        return {
            "mean": t.mean().item(),
            "max": t.max().item(),
            "min": t.min().item(),
            "std": t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
        }

    metrics: dict[str, Any] = {
        "sentencepo/valid_ratio": flat_valid.float().mean().item(),
    }

    if sentence_counts:
        count_tensor = torch.tensor(sentence_counts, device=device, dtype=torch.float)
        metrics.update({f"sentencepo/count/{k}": v for k, v in _safe_stat(count_tensor).items()})

    metrics.update({f"sentencepo/len/{k}": v for k, v in _safe_stat(sent_lens).items()})

    if log_prob is not None and old_log_prob is not None:
        neg_kl = log_prob - old_log_prob
        flat_kl = neg_kl.view(-1)[flat_valid]
        kl_sum = torch.zeros_like(unique_sid, dtype=flat_kl.dtype)
        kl_sum.index_add_(0, inv, flat_kl)
        sent_kl_mean = kl_sum / (sent_lens + 1e-8)
        sent_ratio = torch.exp(torch.clamp(sent_kl_mean, max=10.0))

        metrics.update({f"sentencepo/ratio/{k}": v for k, v in _safe_stat(sent_ratio).items()})

        per_resp_std = []
        per_resp_range = []
        for b in range(bs):
            row_valid = valid_mask[b]
            if not torch.any(row_valid):
                continue
            row_sid = sid_with_offset[b][row_valid]
            row_kl = neg_kl[b][row_valid]
            row_unique, row_inv = torch.unique(row_sid, return_inverse=True)
            row_len = torch.zeros_like(row_unique, dtype=torch.float)
            row_len.index_add_(0, row_inv, torch.ones_like(row_inv, dtype=torch.float))
            row_kl_sum = torch.zeros_like(row_unique, dtype=row_kl.dtype)
            row_kl_sum.index_add_(0, row_inv, row_kl)
            row_ratio = torch.exp(torch.clamp(row_kl_sum / (row_len + 1e-8), max=10.0))
            if row_ratio.numel() > 1:
                per_resp_std.append(row_ratio.std(unbiased=False))
                per_resp_range.append((row_ratio.max() - row_ratio.min()))
            else:
                per_resp_std.append(torch.tensor(0.0, device=device))
                per_resp_range.append(torch.tensor(0.0, device=device))

        if per_resp_std:
            per_resp_std_t = torch.stack(per_resp_std)
            metrics.update({f"sentencepo/ratio_std_across_sent/{k}": v for k, v in _safe_stat(per_resp_std_t).items()})
        if per_resp_range:
            per_resp_range_t = torch.stack(per_resp_range)
            metrics.update(
                {f"sentencepo/ratio_range_across_sent/{k}": v for k, v in _safe_stat(per_resp_range_t).items()}
            )

        if hist_enable and (hist_every == 0 or (global_step is not None and global_step % hist_every == 0)):
            ratio_vals_cpu = sent_ratio.detach().cpu()
            if ratio_vals_cpu.numel() > hist_max_points:
                idx = torch.randperm(ratio_vals_cpu.numel())[:hist_max_points]
                ratio_vals_cpu = ratio_vals_cpu[idx]
            metrics["sentencepo/ratio_hist"] = ratio_vals_cpu.numpy()

    if entropys is not None:
        flat_ent = entropys.view(-1)[flat_valid]
        ent_sum = torch.zeros_like(unique_sid, dtype=flat_ent.dtype)
        ent_sum.index_add_(0, inv, flat_ent)
        sent_entropy = ent_sum / (sent_lens + 1e-8)
        metrics.update({f"sentencepo/entropy/{k}": v for k, v in _safe_stat(sent_entropy).items()})

        if hist_enable and (hist_every == 0 or (global_step is not None and global_step % hist_every == 0)):
            ent_cpu = sent_entropy.detach().cpu()
            if ent_cpu.numel() > hist_max_points:
                idx = torch.randperm(ent_cpu.numel())[:hist_max_points]
                ent_cpu = ent_cpu[idx]
            metrics["sentencepo/entropy_hist"] = ent_cpu.numpy()

    return metrics


def compute_sentencepo_semantic_metrics(
    *,
    token_hidden_states: torch.Tensor | None,
    sentence_ids: torch.Tensor | None,
    response_mask: torch.Tensor,
    index: np.ndarray,
    token_level_rewards: torch.Tensor,
    final_advantages: torch.Tensor | None = None,
    config: Any,
) -> dict[str, Any]:
    """Compute SentencePO semantic metrics (similarity, sentence_adv stats, divergence).

    This function is gated by sentence_adv.metrics_enable to avoid overhead.
    """

    sentence_adv_cfg = getattr(config, "sentence_adv", None)
    if sentence_adv_cfg is None or not getattr(sentence_adv_cfg, "metrics_enable", False):
        return {}

    if token_hidden_states is None or sentence_ids is None:
        return {}

    device = token_hidden_states.device
    token_hidden_states = token_hidden_states.float()
    sentence_ids = sentence_ids.to(device)
    response_mask = response_mask.to(device)

    bs, seq_len, hidden = token_hidden_states.shape
    valid = (response_mask > 0) & (sentence_ids >= 0)
    if not torch.any(valid):
        return {}

    flat_sid_all = sentence_ids.view(-1)
    flat_valid = valid.view(-1)
    flat_emb = token_hidden_states.view(-1, hidden)

    flat_sid_valid = flat_sid_all[flat_valid]
    flat_emb_valid = flat_emb[flat_valid]

    unique_sid, inv = torch.unique(flat_sid_valid, return_inverse=True)
    num_sent = unique_sid.numel()
    if num_sent == 0:
        return {}

    # compute last token mask for each sentence
    next_sid = torch.roll(sentence_ids, shifts=-1, dims=1)
    next_valid = torch.roll(valid, shifts=-1, dims=1)
    last_pos_mask = torch.zeros_like(valid)
    last_pos_mask[:, -1] = True
    boundary = last_pos_mask | (sentence_ids != next_sid) | (~next_valid)
    last_mask = valid & boundary

    flat_last = last_mask.view(-1)
    flat_sid_last = flat_sid_all[flat_last]
    flat_emb_last = flat_emb[flat_last]

    if flat_sid_last.numel() == 0:
        return {}

    idx_last = torch.searchsorted(unique_sid, flat_sid_last)

    # sentence embedding pooling
    pooling = sentence_adv_cfg.pooling
    eps = sentence_adv_cfg.eps
    if pooling == "mean":
        sum_emb = torch.zeros((num_sent, hidden), device=device, dtype=token_hidden_states.dtype)
        cnt = torch.zeros((num_sent, 1), device=device, dtype=token_hidden_states.dtype)
        sum_emb.index_add_(0, inv, flat_emb_valid)
        cnt.index_add_(0, inv, torch.ones_like(flat_sid_valid, dtype=token_hidden_states.dtype).unsqueeze(-1))
        sent_emb = sum_emb / (cnt + eps)
    elif pooling == "last":
        sent_emb = torch.zeros((num_sent, hidden), device=device, dtype=token_hidden_states.dtype)
        sent_emb.index_copy_(0, idx_last, flat_emb_last)
    else:
        return {}

    # map sentence to sample index (use last token positions)
    flat_sample_idx = torch.arange(bs, device=device).unsqueeze(1).expand(bs, seq_len).reshape(-1)
    flat_sample_last = flat_sample_idx[flat_last]
    sent_sample_idx = torch.zeros((num_sent,), device=device, dtype=torch.long)
    sent_sample_idx.index_copy_(0, idx_last, flat_sample_last)

    # correctness from token-level rewards (sum > threshold)
    scores = token_level_rewards.to(device).sum(dim=-1)
    correct = scores > sentence_adv_cfg.correctness_threshold

    # normalize sentence embeddings for cosine similarity
    sent_emb = F.normalize(sent_emb, dim=-1)

    group_ids = as_torch_index(index, device=device)
    sent_group = group_ids[sent_sample_idx]
    unique_groups = torch.unique(sent_group)

    max_sentences = max(1, int(getattr(sentence_adv_cfg, "metrics_max_sentences", 128)))
    max_pairs = max(1, int(getattr(sentence_adv_cfg, "metrics_max_pairs", 4096)))
    pos_bins = max(1, int(getattr(sentence_adv_cfg, "metrics_pos_bins", 4)))
    div_threshold = float(getattr(sentence_adv_cfg, "metrics_divergence_threshold", 0.1))

    cc_vals: list[torch.Tensor] = []
    ww_vals: list[torch.Tensor] = []
    cw_vals: list[torch.Tensor] = []

    sent_adv = torch.zeros((num_sent,), device=device, dtype=token_hidden_states.dtype)
    sent_correct = correct[sent_sample_idx]

    # local sentence id for position binning
    sent_local_id = torch.zeros((num_sent,), device=device, dtype=torch.long)
    sent_local_id.index_copy_(0, idx_last, flat_sid_last - flat_sample_last * (seq_len + 1))
    sent_local_id = torch.clamp(sent_local_id, min=0)

    max_local_per_sample = torch.zeros((bs,), device=device, dtype=torch.long)
    for b in range(bs):
        mask_b = sent_sample_idx == b
        if torch.any(mask_b):
            max_local_per_sample[b] = sent_local_id[mask_b].max()

    denom = torch.clamp(max_local_per_sample[sent_sample_idx], min=1).float()
    rel_pos = sent_local_id.float() / denom
    bin_idx = torch.clamp((rel_pos * pos_bins).long(), max=pos_bins - 1)

    with torch.no_grad():
        for g in unique_groups.tolist():
            mask = sent_group == g
            if not torch.any(mask):
                continue
            E = sent_emb[mask]
            sample_idx_g = sent_sample_idx[mask]
            pos_mask = correct[sample_idx_g]
            neg_mask = ~pos_mask

            if torch.any(pos_mask):
                E_pos = E[pos_mask]
                if E_pos.size(0) > max_sentences:
                    idx = torch.randperm(E_pos.size(0), device=device)[:max_sentences]
                    E_pos = E_pos[idx]
            else:
                E_pos = None

            if torch.any(neg_mask):
                E_neg = E[neg_mask]
                if E_neg.size(0) > max_sentences:
                    idx = torch.randperm(E_neg.size(0), device=device)[:max_sentences]
                    E_neg = E_neg[idx]
            else:
                E_neg = None

            if E_pos is not None and E_pos.size(0) >= 2:
                sims = E_pos @ E_pos.T
                tri = torch.triu_indices(sims.size(0), sims.size(1), offset=1, device=device)
                vals = sims[tri[0], tri[1]]
                if vals.numel() > max_pairs:
                    idx = torch.randperm(vals.numel(), device=device)[:max_pairs]
                    vals = vals.view(-1)[idx]
                cc_vals.append(vals.view(-1))

            if E_neg is not None and E_neg.size(0) >= 2:
                sims = E_neg @ E_neg.T
                tri = torch.triu_indices(sims.size(0), sims.size(1), offset=1, device=device)
                vals = sims[tri[0], tri[1]]
                if vals.numel() > max_pairs:
                    idx = torch.randperm(vals.numel(), device=device)[:max_pairs]
                    vals = vals.view(-1)[idx]
                ww_vals.append(vals.view(-1))

            if E_pos is not None and E_neg is not None and E_pos.numel() > 0 and E_neg.numel() > 0:
                sims = E_pos @ E_neg.T
                vals = sims.view(-1)
                if vals.numel() > max_pairs:
                    idx = torch.randperm(vals.numel(), device=device)[:max_pairs]
                    vals = vals[idx]
                cw_vals.append(vals.view(-1))

            # sentence_adv per sentence
            log_eps = math.log(sentence_adv_cfg.eps)
            tau = sentence_adv_cfg.temperature
            if E_pos is not None and E_pos.numel() > 0:
                sims_pos = E @ E_pos.T
                log_D_pos = torch.logsumexp(sims_pos / tau, dim=-1)
            else:
                log_D_pos = torch.full((E.size(0),), log_eps, device=device, dtype=token_hidden_states.dtype)

            if E_neg is not None and E_neg.numel() > 0:
                sims_neg = E @ E_neg.T
                log_D_neg = torch.logsumexp(sims_neg / tau, dim=-1)
            else:
                log_D_neg = torch.full((E.size(0),), log_eps, device=device, dtype=token_hidden_states.dtype)

            A = log_D_pos - log_D_neg
            if sentence_adv_cfg.normalize and A.numel() > 1:
                A = (A - A.mean()) / (A.std(unbiased=False) + sentence_adv_cfg.eps)
            sent_adv[mask] = A

    def _mean_p90(values: list[torch.Tensor]) -> dict[str, float]:
        if not values:
            return {}
        all_vals = torch.cat(values)
        if all_vals.numel() == 0:
            return {}
        return {
            "mean": all_vals.mean().item(),
            "p90": torch.quantile(all_vals, 0.9).item(),
        }

    metrics: dict[str, Any] = {}
    for name, vals in (
        ("correct_correct", cc_vals),
        ("wrong_wrong", ww_vals),
        ("correct_wrong", cw_vals),
    ):
        stats = _mean_p90(vals)
        if stats:
            metrics[f"sentencepo/sentence_sim/{name}_mean"] = stats["mean"]
            metrics[f"sentencepo/sentence_sim/{name}_p90"] = stats["p90"]

    if (
        "sentencepo/sentence_sim/correct_correct_mean" in metrics
        and "sentencepo/sentence_sim/correct_wrong_mean" in metrics
    ):
        metrics["sentencepo/sentence_sim/ratio_cc_vs_cw"] = (
            metrics["sentencepo/sentence_sim/correct_correct_mean"]
            - metrics["sentencepo/sentence_sim/correct_wrong_mean"]
        )
    if (
        "sentencepo/sentence_sim/wrong_wrong_mean" in metrics
        and "sentencepo/sentence_sim/correct_wrong_mean" in metrics
    ):
        metrics["sentencepo/sentence_sim/ratio_ww_vs_cw"] = (
            metrics["sentencepo/sentence_sim/wrong_wrong_mean"]
            - metrics["sentencepo/sentence_sim/correct_wrong_mean"]
        )

    # sentence_adv distribution (correct vs wrong)
    def _adv_stats(vals: torch.Tensor) -> dict[str, float]:
        if vals.numel() == 0:
            return {}
        return {
            "mean": vals.mean().item(),
            "std": vals.std(unbiased=False).item() if vals.numel() > 1 else 0.0,
            "p50": torch.quantile(vals, 0.5).item(),
            "p90": torch.quantile(vals, 0.9).item(),
        }

    adv_correct = sent_adv[sent_correct]
    adv_wrong = sent_adv[~sent_correct]
    stats_c = _adv_stats(adv_correct)
    stats_w = _adv_stats(adv_wrong)
    if stats_c:
        metrics.update({f"sentencepo/sentence_adv/{k}_correct": v for k, v in stats_c.items()})
    if stats_w:
        metrics.update({f"sentencepo/sentence_adv/{k}_wrong": v for k, v in stats_w.items()})

    # sentence_adv by position bins
    for b in range(pos_bins):
        mask_bin = bin_idx == b
        if not torch.any(mask_bin):
            continue
        m_correct = mask_bin & sent_correct
        m_wrong = mask_bin & (~sent_correct)
        if torch.any(m_correct):
            metrics[f"sentencepo/sentence_adv/pos_bin_{b}_correct"] = sent_adv[m_correct].mean().item()
        if torch.any(m_wrong):
            metrics[f"sentencepo/sentence_adv/pos_bin_{b}_wrong"] = sent_adv[m_wrong].mean().item()
        if torch.any(m_correct) and torch.any(m_wrong):
            metrics[f"sentencepo/sentence_adv/pos_bin_{b}_diff"] = (
                sent_adv[m_correct].mean() - sent_adv[m_wrong].mean()
            ).item()

    # divergence metrics per group
    div_first_pos = []
    div_strength = []
    for g in unique_groups.tolist():
        mask = sent_group == g
        if not torch.any(mask):
            continue
        diffs = []
        first_bin = None
        for b in range(pos_bins):
            m_bin = mask & (bin_idx == b)
            if not torch.any(m_bin):
                continue
            m_c = m_bin & sent_correct
            m_w = m_bin & (~sent_correct)
            if torch.any(m_c) and torch.any(m_w):
                diff = (sent_adv[m_c].mean() - sent_adv[m_w].mean()).abs()
                diffs.append(diff)
                if first_bin is None and diff.item() > div_threshold:
                    first_bin = b
        if diffs:
            div_strength.append(torch.stack(diffs).mean())
            if first_bin is not None:
                div_first_pos.append(torch.tensor((first_bin + 1) / pos_bins, device=device))

    if div_strength:
        div_strength_t = torch.stack(div_strength)
        metrics["sentencepo/divergence/strength_mean"] = div_strength_t.mean().item()
        metrics["sentencepo/divergence/strength_p90"] = torch.quantile(div_strength_t, 0.9).item()
    if div_first_pos:
        div_first_pos_t = torch.stack(div_first_pos)
        metrics["sentencepo/divergence/first_pos_mean"] = div_first_pos_t.mean().item()
        metrics["sentencepo/divergence/first_pos_p50"] = torch.quantile(div_first_pos_t, 0.5).item()
        metrics["sentencepo/divergence/first_pos_p90"] = torch.quantile(div_first_pos_t, 0.9).item()

    # advantage decomposition (final = grpo + alpha * sentence_adv)
    if final_advantages is not None:
        mask = response_mask > 0
        denom = mask.sum().clamp(min=1)
        final_mean = (final_advantages * mask).sum() / denom

        flat_adv = torch.zeros_like(flat_sid_all, dtype=token_hidden_states.dtype)
        flat_adv[flat_valid] = sent_adv[inv]
        sent_adv_tokens = flat_adv.view(bs, seq_len) * response_mask
        sent_adv_mean = sent_adv_tokens.sum() / denom

        alpha = sentence_adv_cfg.alpha
        metrics["sentencepo/sentence_adv/alpha_term_mean"] = (alpha * sent_adv_mean).item()
        metrics["sentencepo/sentence_adv/final_adv_mean"] = final_mean.item()
        metrics["sentencepo/sentence_adv/grpo_term_mean"] = (final_mean - alpha * sent_adv_mean).item()

    return metrics
