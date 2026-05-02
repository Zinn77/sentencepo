#!/usr/bin/env python
"""Phase A diagnostic for v1-5-hidden ablation.

Sweeps (hidden_layer × pooling) on a base model, scoring each combo on three
representation-health metrics for both the SCR-style and SLPA-style
sentence-advantage objectives. Output is a markdown table that ranks combos.

Two-stage flow:
  1. Rollout via vLLM (TP=8 by default), with sampling/template aligned to
     formal RL training in test_v1-5_hidden_phaseB.sh: chat_template +
     enable_thinking=False, temperature=1.0, top_p=1.0, top_k=-1, n=8.
  2. After freeing vLLM, load HF on GPU 0 and run a single forward pass with
     output_hidden_states=True over the collected rollouts. Pooling and metric
     computation are CPU-cheap.

Usage:
    python scripts_server/diagnose_sentence_repr.py \
        --model_path Qwen/Qwen3-4B-Base \
        --train_parquet $HOME/data/math/train.parquet \
        --output_md CCdocs/2026-05-02_phaseA_diagnostic_v2.md \
        --num_prompts 32 --rollouts_per_prompt 8 \
        --tensor_parallel_size 8 --max_prompt_tokens 1024 --max_new_tokens 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before `import vllm` / `LLM(...)`. We touch CUDA in the parent
# (torch.manual_seed → cuda.manual_seed_all → _lazy_init) before vLLM forks
# its TP workers; the default fork start method then trips
# "Cannot re-initialize CUDA in forked subprocess". 'spawn' fixes it.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.sentence_repr import VALID_POOLINGS, build_punct_token_ids, pool_sentence_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="HF model dir, e.g. Qwen3-4B-Base")
    p.add_argument("--train_parquet", required=True, help="math/train.parquet")
    p.add_argument("--output_md", required=True)
    p.add_argument("--num_prompts", type=int, default=32)
    # Defaults aligned with formal RL training (test_v1-5_hidden_phaseB.sh +
    # verl/trainer/config/rollout/rollout.yaml):
    #   max_prompt_length=1024, max_response_length=4096,
    #   rollout.n=8, temperature=1.0, top_p=1.0, top_k=-1,
    #   tensor_parallel_size=1 per replica (we use TP=8 in one process to
    #   saturate the 8 GPUs we're allocated to Phase A).
    p.add_argument("--rollouts_per_prompt", type=int, default=8)
    p.add_argument("--max_prompt_tokens", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    # HF forward stage. Aligned with formal RL: actor uses FSDP across all
    # GPUs in the node and processes ppo_micro_batch_size_per_gpu=4 sequences
    # per micro-batch. We do plain DDP (one model copy per GPU) which is
    # mathematically equivalent for inference; --forward_chunk_size mirrors
    # micro_batch_size, --forward_world_size mirrors n_gpus_per_node.
    p.add_argument("--forward_chunk_size", type=int, default=4)
    p.add_argument(
        "--forward_world_size",
        type=int,
        default=0,
        help="GPUs to use for HF forward (0 = auto = torch.cuda.device_count()).",
    )
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Pass enable_thinking=True into apply_chat_template (Qwen3 chat). "
        "Default off to match formal RL config.",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="-1,-4,-9,-18,-1|-9|-18",
        help="comma-separated layer indices; '|' inside an entry => list (ensemble mean).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_sent_tokens", type=int, default=6)
    return p.parse_args()


def parse_layers(spec: str) -> list[object]:
    out: list[object] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "|" in entry:
            out.append([int(x) for x in entry.split("|") if x])
        else:
            out.append(int(entry))
    return out


def load_prompts(parquet_path: str, num_prompts: int) -> tuple[list[list[dict]], list[str]]:
    """Return (messages_per_prompt, ground_truths).

    Preserves the parquet's chat-message structure so we can apply the same
    chat template formal RL training uses. Falling back to a synthetic single
    user-message wrap when the column is just a question string.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    prompts: list[list[dict]] = []
    answers: list[str] = []
    for _, row in df.head(num_prompts).iterrows():
        raw_prompt = row["prompt"] if "prompt" in row else None
        if hasattr(raw_prompt, "tolist"):
            raw_prompt = raw_prompt.tolist()
        messages: list[dict] = []
        if isinstance(raw_prompt, (list, tuple)) and len(raw_prompt) > 0:
            for m in raw_prompt:
                messages.append({"role": str(m["role"]), "content": str(m["content"])})
        elif "question" in row:
            messages = [{"role": "user", "content": str(row["question"])}]
        else:
            messages = [{"role": "user", "content": str(row.iloc[0])}]
        prompts.append(messages)
        if "reward_model" in row and isinstance(row["reward_model"], dict):
            answers.append(str(row["reward_model"].get("ground_truth", "")))
        elif "answer" in row:
            answers.append(str(row["answer"]))
        else:
            answers.append("")
    return prompts, answers


def build_sentence_ids(
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    min_sent_tokens: int,
) -> torch.Tensor:
    """Sentence segmentation matching verl/utils/dataset/rl_dataset.py:_build_sentence_ids."""
    bs, t = response_token_ids.shape
    sentence_ids = torch.full_like(response_token_ids, -1, dtype=torch.long)
    end_chars = {".", "?", "!", "。", "？", "！"}
    for b in range(bs):
        valid_pos = response_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        if not valid_pos:
            continue
        sentences: list[list[int]] = []
        current: list[int] = []
        for pos in valid_pos:
            token_id = int(response_token_ids[b, pos].item())
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            current.append(pos)
            if "\n" in token_str or any(ch in token_str for ch in end_chars):
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)
        if not sentences:
            continue
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
            for pos in sent:
                sentence_ids[b, pos] = sid
    # Match verl/utils/dataset/rl_dataset.py:65-69 collate_fn: add b*10000
    # offset so sentence_ids are globally unique across the batch. Without
    # this, torch.unique() in pool_sentence_embeddings collapses all "sentence
    # 0" across rollouts into a single bucket, killing the diagnostic.
    for b in range(bs):
        mask = sentence_ids[b] >= 0
        sentence_ids[b][mask] += b * 10000
    return sentence_ids


def score_math(answer: str, ground_truth: str) -> float:
    try:
        from verl.utils.reward_score.math_reward import compute_score
    except Exception:
        return 0.0
    try:
        return float(compute_score(answer, ground_truth))
    except Exception:
        return 0.0


def select_layer(hidden_states_tuple, layer_index) -> torch.Tensor:
    if isinstance(layer_index, int):
        return hidden_states_tuple[layer_index]
    return torch.stack(
        [hidden_states_tuple[int(i)] for i in layer_index], dim=0
    ).mean(dim=0)


def slpa_style_score(sent_emb: torch.Tensor, sent_sample: torch.Tensor, sent_reward: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Simplified per-sentence SLPA delta proxy for ranking only.

    For each sentence k of rollout i, compute
        V_k(i) = Σ_{j≠i} K(h_k, h_l_j) * r_j  /  Σ_{j≠i} K(h_k, h_l_j)
    where l_j is the *last* sentence index of rollout j (i.e. final state).
    Δ_k = V_k(i) - V_{k-1}(i) is the score; for the ranking metric we just
    return V_k - mean(V) per rollout (a stand-in for delta).
    """
    n = sent_emb.shape[0]
    V = torch.zeros(n, device=sent_emb.device)
    for i in range(n):
        same_sample = sent_sample == sent_sample[i]
        # Use rewards from samples *other* than i.
        kernel = torch.exp((sent_emb[i] @ sent_emb.t()) / tau)
        kernel = kernel * (~same_sample).float()
        denom = kernel.sum() + 1e-8
        V[i] = (kernel * sent_reward).sum() / denom
    # delta proxy: V minus per-sample V mean.
    out = torch.zeros_like(V)
    for s in torch.unique(sent_sample):
        mask = sent_sample == s
        if mask.any():
            out[mask] = V[mask] - V[mask].mean()
    return out


def scr_style_score(sent_emb: torch.Tensor, sent_sample: torch.Tensor, sent_reward: torch.Tensor, tau_r: float = 1.0, tau_s: float = 0.1) -> torch.Tensor:
    """Simplified SCR proxy: per-rollout pooled emb -> reward-weighted soft centers, leave-one-out."""
    bs = int(sent_sample.max().item()) + 1
    rollout_emb = torch.zeros((bs, sent_emb.shape[1]), device=sent_emb.device)
    cnt = torch.zeros(bs, device=sent_emb.device)
    rollout_emb.index_add_(0, sent_sample, sent_emb)
    cnt.index_add_(0, sent_sample, torch.ones_like(sent_sample, dtype=torch.float))
    rollout_emb = F.normalize(rollout_emb / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    # Per-sample reward.
    sample_reward = torch.zeros(bs, device=sent_emb.device)
    sample_reward.index_copy_(0, sent_sample, sent_reward.float())
    # Soft center (single group).
    w_pos = torch.softmax(sample_reward / tau_r, dim=0)
    w_neg = torch.softmax(-sample_reward / tau_r, dim=0)
    c_pos = (w_pos.unsqueeze(1) * rollout_emb).sum(dim=0)
    c_neg = (w_neg.unsqueeze(1) * rollout_emb).sum(dim=0)
    c_pos = F.normalize(c_pos, dim=-1)
    c_neg = F.normalize(c_neg, dim=-1)
    return ((sent_emb * c_pos).sum(-1) - (sent_emb * c_neg).sum(-1)) / tau_s


def repr_metrics(
    sent_emb: torch.Tensor,
    sent_score: torch.Tensor,
    sent_correct: torch.Tensor,
) -> dict[str, float]:
    s = sent_emb.shape[0]
    if s < 2:
        return {}
    out: dict[str, float] = {}
    n = min(s, 96)
    perm = torch.randperm(s, device=sent_emb.device)[:n]
    sub = sent_emb[perm]
    sim = sub @ sub.t()
    off = sim - torch.diag_embed(torch.diagonal(sim))
    out["cos_global"] = float(off.sum().item() / max(n * (n - 1), 1))
    if sent_correct.any() and (~sent_correct).any():
        c_pos = F.normalize(sent_emb[sent_correct].mean(dim=0), dim=-1)
        c_neg = F.normalize(sent_emb[~sent_correct].mean(dim=0), dim=-1)
        out["pos_neg_gap"] = float(1.0 - (c_pos * c_neg).sum().item())
    if sent_score.numel() > 1:
        out["score_var"] = float(sent_score.var(unbiased=False).item())
        ref_var = float(sent_correct.float().var(unbiased=False).item())
        out["snr"] = out["score_var"] / (ref_var + 1e-8)
    return out


def _forward_worker(
    rank: int,
    world_size: int,
    model_path: str,
    inputs_path: str,
    response_len: int,
    hidden_size: int,
    needed_pos_sorted: list[int],
    fwd_chunk: int,
    shard_dir: str,
) -> None:
    """One DDP rank: load model on cuda:rank, forward this rank's batch slice,
    extract needed-layer hidden states + token entropy, dump shard to disk.

    Mirrors formal RL's per-GPU forward (FSDP all-gather → forward → drop).
    """
    import torch
    import torch.nn.functional as F  # noqa: N812
    from transformers import AutoModelForCausalLM

    torch.cuda.set_device(rank)

    data = torch.load(inputs_path, weights_only=False)
    full_input: torch.Tensor = data["full_input"]
    full_attn: torch.Tensor = data["full_attn"]
    bs = full_input.shape[0]

    per_rank = (bs + world_size - 1) // world_size
    s = rank * per_rank
    e = min(s + per_rank, bs)
    out_path = os.path.join(shard_dir, f"fwd_{rank}.pt")
    if s >= e:
        torch.save({"rank": rank, "s": s, "e": e}, out_path)
        return

    if rank == 0:
        print(f"[rank {rank}] loading model on cuda:{rank}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda(rank).eval()

    bs_local = e - s
    needed_pos = set(needed_pos_sorted)
    response_hs_local: dict[int, torch.Tensor] = {
        idx: torch.zeros((bs_local, response_len, hidden_size), dtype=torch.bfloat16)
        for idx in needed_pos
    }
    token_entropy_local = torch.zeros((bs_local, response_len), dtype=torch.float32)

    ent_chunk = max(1, fwd_chunk // 2)
    for cs in range(0, bs_local, fwd_chunk):
        ce = min(cs + fwd_chunk, bs_local)
        gs, ge = s + cs, s + ce
        with torch.no_grad():
            out = model(
                input_ids=full_input[gs:ge].cuda(rank),
                attention_mask=full_attn[gs:ge].cuda(rank),
                output_hidden_states=True,
                use_cache=False,
            )
        rl_view = out.logits[:, -response_len - 1 : -1, :]
        for i in range(0, rl_view.shape[0], ent_chunk):
            lp = F.log_softmax(rl_view[i : i + ent_chunk].float(), dim=-1)
            token_entropy_local[cs + i : cs + i + lp.shape[0]] = (
                -(lp.exp() * lp).sum(dim=-1)
            ).cpu()
            del lp
        del rl_view
        for idx in needed_pos:
            response_hs_local[idx][cs:ce] = (
                out.hidden_states[idx][:, -response_len:, :].cpu()
            )
        del out
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"[rank 0] forward chunk {ce}/{bs_local} done", flush=True)

    torch.save(
        {
            "rank": rank,
            "s": s,
            "e": e,
            "response_hs": response_hs_local,
            "token_entropy": token_entropy_local,
        },
        out_path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    layers = parse_layers(args.layers)
    poolings = list(VALID_POOLINGS)

    print(f"Loading tokenizer from {args.model_path}", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading {args.num_prompts} prompts from {args.train_parquet}", flush=True)
    prompts_messages, answers = load_prompts(args.train_parquet, args.num_prompts)

    # Tokenize prompts (no truncation), then filter out overlong prompts to
    # match formal RL config: data.filter_overlong_prompts=True +
    # data.truncation='error' (see test_sentencepo_v1-5.sh:114-115). Silent
    # truncation here would cut the trailing "Please reason ... \boxed{}"
    # instruction and produce 0-reward rollouts.
    chat_template_kwargs: dict = {"enable_thinking": bool(args.enable_thinking)}
    prompt_token_ids_list: list[list[int]] = []
    kept_answers: list[str] = []
    n_dropped = 0
    for messages, gt in zip(prompts_messages, answers):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        ids = list(ids)
        if len(ids) > args.max_prompt_tokens:
            n_dropped += 1
            continue
        prompt_token_ids_list.append(ids)
        kept_answers.append(gt)
    if n_dropped > 0:
        print(
            f"Filtered {n_dropped}/{len(prompts_messages)} prompts that exceeded "
            f"max_prompt_tokens={args.max_prompt_tokens} after chat templating.",
            flush=True,
        )
    answers = kept_answers
    if not prompt_token_ids_list:
        raise RuntimeError(
            f"All {len(prompts_messages)} prompts exceed max_prompt_tokens="
            f"{args.max_prompt_tokens}. Increase --max_prompt_tokens or expand --num_prompts."
        )

    # ---- Phase 1: vLLM rollout (aligned with formal RL: vllm + TP, n=8, T=1.0). ----
    from vllm import LLM, SamplingParams

    print(
        f"Loading vLLM (TP={args.tensor_parallel_size}, "
        f"gpu_mem_util={args.gpu_memory_utilization})...",
        flush=True,
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        trust_remote_code=True,
        enforce_eager=False,
        seed=args.seed,
        # Custom all-reduce uses direct P2P which often isn't exposed inside
        # docker containers (autodl / similar). Falls back to NCCL all-reduce
        # which is ~10-20% slower for TP comms but 100% compatible.
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        n=args.rollouts_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"Generating {args.rollouts_per_prompt} rollouts × "
        f"{len(prompt_token_ids_list)} prompts via vLLM...",
        flush=True,
    )
    vllm_outputs = llm.generate(
        prompt_token_ids=prompt_token_ids_list,
        sampling_params=sampling,
    )

    all_input_ids: list[torch.Tensor] = []
    all_response_mask: list[torch.Tensor] = []
    all_response_ids: list[torch.Tensor] = []
    all_rewards: list[float] = []
    eos_id = tokenizer.eos_token_id
    # vLLM preserves input order, so prompt_idx ↔ output index.
    for prompt_idx, output in enumerate(vllm_outputs):
        prompt_token_ids = list(output.prompt_token_ids)
        gt = answers[prompt_idx]
        for completion in output.outputs:
            resp_ids = list(completion.token_ids)
            full_ids = prompt_token_ids + resp_ids
            response_text = completion.text
            r = score_math(response_text, gt)
            response_mask = torch.ones(len(resp_ids), dtype=torch.long)
            if eos_id is not None:
                resp_tensor = torch.tensor(resp_ids, dtype=torch.long)
                eos_pos = (resp_tensor == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    cut = int(eos_pos[0, 0].item()) + 1
                    response_mask[cut:] = 0
            all_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            all_response_ids.append(torch.tensor(resp_ids, dtype=torch.long))
            all_response_mask.append(response_mask)
            all_rewards.append(r)

    # Free vLLM before loading HF for the hidden-state forward pass. vLLM 0.8
    # holds GPU memory across all TP workers; explicit teardown is required.
    print("Freeing vLLM engine...", flush=True)
    try:
        del llm.llm_engine
    except Exception:
        pass
    del llm
    import gc

    gc.collect()
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        print(f"  vLLM cleanup partial: {e}", flush=True)
    torch.cuda.empty_cache()

    # ---- Phase 2: HF forward across all visible GPUs (DDP, equivalent to
    # formal RL's FSDP forward for inference; one full model copy per GPU,
    # each rank handles bs / world_size sequences). ----
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    total_layers = cfg.num_hidden_layers + 1  # +1 for input embedding
    hidden_size = cfg.hidden_size

    # Pad to common length.
    full_max = max(t.shape[0] for t in all_input_ids)
    resp_max = max(t.shape[0] for t in all_response_ids)
    bs = len(all_input_ids)
    full_input = torch.full((bs, full_max), tokenizer.pad_token_id, dtype=torch.long)
    full_attn = torch.zeros((bs, full_max), dtype=torch.long)
    response_ids = torch.full((bs, resp_max), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((bs, resp_max), dtype=torch.long)
    for i in range(bs):
        n_full = all_input_ids[i].shape[0]
        full_input[i, :n_full] = all_input_ids[i]
        full_attn[i, :n_full] = 1
        n_r = all_response_ids[i].shape[0]
        response_ids[i, :n_r] = all_response_ids[i]
        response_mask[i, :n_r] = all_response_mask[i]
    rewards = torch.tensor(all_rewards, dtype=torch.float)
    print(f"Reward stats: mean={rewards.mean():.3f} pos_frac={(rewards > 0).float().mean():.3f}", flush=True)

    # Build sentence ids on the response slice.
    print("Building sentence ids...", flush=True)
    sentence_ids = build_sentence_ids(response_ids, response_mask, tokenizer, args.min_sent_tokens)

    bs = full_input.shape[0]
    response_len = response_ids.shape[1]

    needed_layers: set[int] = set()
    for layer in layers:
        if isinstance(layer, int):
            needed_layers.add(layer)
        else:
            for x in layer:
                needed_layers.add(int(x))
    needed_pos = {(i if i >= 0 else total_layers + i) for i in needed_layers}

    # Persist inputs so each spawned worker can mmap them in.
    import shutil
    import tempfile

    shard_dir = tempfile.mkdtemp(prefix="phase_a_")
    inputs_path = os.path.join(shard_dir, "inputs.pt")
    torch.save({"full_input": full_input, "full_attn": full_attn}, inputs_path)

    # Give vLLM workers a moment to fully exit before spawning forward DDP.
    import time

    time.sleep(3)

    world_size = args.forward_world_size or torch.cuda.device_count()
    fwd_chunk = args.forward_chunk_size
    print(
        f"Spawning {world_size}-way DDP forward "
        f"(per-rank bs={(bs + world_size - 1) // world_size}, "
        f"fwd_chunk={fwd_chunk}, layers_kept={sorted(needed_pos)}/{total_layers})...",
        flush=True,
    )

    import torch.multiprocessing as mp

    mp.spawn(
        _forward_worker,
        args=(
            world_size,
            args.model_path,
            inputs_path,
            response_len,
            hidden_size,
            sorted(needed_pos),
            fwd_chunk,
            shard_dir,
        ),
        nprocs=world_size,
        join=True,
    )

    print("Aggregating forward shards...", flush=True)
    response_hs_per_layer: list[torch.Tensor | None] = [None] * total_layers
    for idx in needed_pos:
        response_hs_per_layer[idx] = torch.zeros(
            (bs, response_len, hidden_size), dtype=torch.bfloat16
        )
    token_entropy = torch.zeros((bs, response_len), dtype=torch.float32)
    for rank in range(world_size):
        shard_path = os.path.join(shard_dir, f"fwd_{rank}.pt")
        shard = torch.load(shard_path, weights_only=False)
        s, e = shard["s"], shard["e"]
        if s < e:
            for idx in needed_pos:
                response_hs_per_layer[idx][s:e] = shard["response_hs"][idx]
            token_entropy[s:e] = shard["token_entropy"]
    shutil.rmtree(shard_dir, ignore_errors=True)

    punct_ids = build_punct_token_ids(tokenizer)

    # Sweep.
    rows: list[dict] = []
    for layer in layers:
        layer_label = str(layer) if isinstance(layer, int) else "|".join(str(int(x)) for x in layer)
        if isinstance(layer, int):
            hs = response_hs_per_layer[layer].float()
        else:
            hs = torch.stack(
                [response_hs_per_layer[int(x)].float() for x in layer], dim=0
            ).mean(dim=0)
        for pooling in poolings:
            try:
                pooled = pool_sentence_embeddings(
                    token_hidden_states=hs,
                    sentence_ids=sentence_ids,
                    response_mask=response_mask,
                    pooling=pooling,
                    response_token_ids=response_ids,
                    token_entropy=token_entropy,
                    punct_token_ids=punct_ids,
                )
            except Exception as e:
                print(f"  layer={layer_label} pooling={pooling}: error {e}")
                continue
            if pooled is None:
                continue
            sent_emb_raw, _, sent_sample = pooled
            sent_emb = F.normalize(sent_emb_raw.float(), dim=-1)
            sent_reward = rewards[sent_sample.cpu()].to(sent_emb.device)
            sent_correct = sent_reward > 0.0

            # SCR-style metrics.
            scr_score = scr_style_score(sent_emb, sent_sample.to(sent_emb.device), sent_reward)
            scr_m = repr_metrics(sent_emb, scr_score, sent_correct)
            # SLPA-style metrics.
            slpa_score = slpa_style_score(sent_emb, sent_sample.to(sent_emb.device), sent_reward)
            slpa_m = repr_metrics(sent_emb, slpa_score, sent_correct)
            rows.append({
                "layer": layer_label,
                "pooling": pooling,
                "scr_cos_global": scr_m.get("cos_global", float("nan")),
                "scr_gap": scr_m.get("pos_neg_gap", float("nan")),
                "scr_snr": scr_m.get("snr", float("nan")),
                "slpa_cos_global": slpa_m.get("cos_global", float("nan")),
                "slpa_gap": slpa_m.get("pos_neg_gap", float("nan")),
                "slpa_snr": slpa_m.get("snr", float("nan")),
            })
            print(f"  layer={layer_label} pooling={pooling}  scr_gap={scr_m.get('pos_neg_gap', 0):.4f}  slpa_gap={slpa_m.get('pos_neg_gap', 0):.4f}", flush=True)

    # Rank.
    def composite(row, mod: str) -> float:
        gap = row[f"{mod}_gap"]
        snr = row[f"{mod}_snr"]
        cos = row[f"{mod}_cos_global"]
        if cos != cos or cos > 0.95:  # NaN or collapsed
            return float("-inf")
        if gap != gap or snr != snr:
            return float("-inf")
        return gap + 0.1 * snr

    scr_ranked = sorted(rows, key=lambda r: composite(r, "scr"), reverse=True)
    slpa_ranked = sorted(rows, key=lambda r: composite(r, "slpa"), reverse=True)

    # Output markdown.
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    lines: list[str] = []
    lines.append("# v1-5-hidden Phase A Diagnostic")
    lines.append("")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(f"- Prompts: {args.num_prompts}, rollouts/prompt: {args.rollouts_per_prompt}")
    lines.append(f"- Reward pos frac: {(rewards > 0).float().mean():.3f}")
    lines.append(f"- Total sentences across all rollouts: see per-row counts")
    lines.append("")
    lines.append("## Full sweep")
    lines.append("")
    lines.append("| layer | pooling | scr_cos_global | scr_gap | scr_snr | slpa_cos_global | slpa_gap | slpa_snr |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['pooling']} | "
            f"{r['scr_cos_global']:.4f} | {r['scr_gap']:.4f} | {r['scr_snr']:.4g} | "
            f"{r['slpa_cos_global']:.4f} | {r['slpa_gap']:.4f} | {r['slpa_snr']:.4g} |"
        )
    lines.append("")
    lines.append("## Top 5 for SCR (rank by gap + 0.1 * snr, drop cos_global > 0.95)")
    lines.append("")
    lines.append("| rank | layer | pooling | gap | snr | cos_global |")
    lines.append("|---|---|---|---:|---:|---:|")
    for i, r in enumerate(scr_ranked[:5]):
        lines.append(f"| {i + 1} | {r['layer']} | {r['pooling']} | {r['scr_gap']:.4f} | {r['scr_snr']:.4g} | {r['scr_cos_global']:.4f} |")
    lines.append("")
    lines.append("## Top 5 for SLPA")
    lines.append("")
    lines.append("| rank | layer | pooling | gap | snr | cos_global |")
    lines.append("|---|---|---|---:|---:|---:|")
    for i, r in enumerate(slpa_ranked[:5]):
        lines.append(f"| {i + 1} | {r['layer']} | {r['pooling']} | {r['slpa_gap']:.4f} | {r['slpa_snr']:.4g} | {r['slpa_cos_global']:.4f} |")
    lines.append("")

    Path(args.output_md).write_text("\n".join(lines))
    print(f"\nWrote {args.output_md}")


if __name__ == "__main__":
    main()
