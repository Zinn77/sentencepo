#!/usr/bin/env python
"""Phase A diagnostic for v1-5-hidden ablation.

Sweeps (hidden_layer × pooling) on a base model, scoring each combo on three
representation-health metrics for both the SCR-style and SLPA-style
sentence-advantage objectives. Output is a markdown table that ranks combos.

Self-contained: uses HF transformers (no vllm/Ray). One forward pass over the
generated rollouts, with output_hidden_states=True, supplies hidden states for
all candidate layers; pooling and metric computation are then CPU-cheap.

Usage:
    python scripts_server/diagnose_sentence_repr.py \
        --model_path /path/to/Qwen3-4B-Base \
        --train_parquet /path/to/math/train.parquet \
        --output_md CCdocs/2026-04-27_sentence_repr_diagnostic.md \
        --num_prompts 32 --rollouts_per_prompt 4 --max_new_tokens 512
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    p.add_argument("--rollouts_per_prompt", type=int, default=4)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
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


def load_prompts(parquet_path: str, num_prompts: int) -> tuple[list[str], list[str]]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    prompts: list[str] = []
    answers: list[str] = []
    for _, row in df.head(num_prompts).iterrows():
        # Try a few common column shapes used in verl math parquets.
        if "prompt" in row and isinstance(row["prompt"], (list, tuple)) and row["prompt"]:
            prompts.append(row["prompt"][0]["content"])
        elif "question" in row:
            prompts.append(str(row["question"]))
        else:
            prompts.append(str(row.iloc[0]))
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    layers = parse_layers(args.layers)
    poolings = list(VALID_POOLINGS)

    print(f"Loading model from {args.model_path}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    print(f"Loading {args.num_prompts} prompts from {args.train_parquet}", flush=True)
    prompts, answers = load_prompts(args.train_parquet, args.num_prompts)

    # Generate rollouts.
    print(f"Generating {args.rollouts_per_prompt} rollouts per prompt...", flush=True)
    all_input_ids: list[torch.Tensor] = []
    all_response_mask: list[torch.Tensor] = []
    all_response_ids: list[torch.Tensor] = []
    all_rewards: list[float] = []
    for p_idx, (prompt, gt) in enumerate(zip(prompts, answers)):
        prompt_text = prompt
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
        prompt_ids = enc["input_ids"].cuda()
        prompt_attn = enc["attention_mask"].cuda()
        for _ in range(args.rollouts_per_prompt):
            with torch.no_grad():
                gen = model.generate(
                    prompt_ids,
                    attention_mask=prompt_attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            full_ids = gen[0]
            response_ids = full_ids[prompt_ids.shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            r = score_math(response_text, gt)
            response_mask = torch.ones_like(response_ids)
            # Trim trailing pad.
            if tokenizer.eos_token_id is not None:
                eos_pos = (response_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    cut = int(eos_pos[0, 0].item()) + 1
                    response_mask[cut:] = 0
            all_input_ids.append(full_ids.cpu())
            all_response_mask.append(response_mask.cpu())
            all_response_ids.append(response_ids.cpu())
            all_rewards.append(r)
        if (p_idx + 1) % 4 == 0:
            print(f"  {p_idx + 1}/{len(prompts)} prompts done", flush=True)

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

    # Single forward with all-layer hidden states; we also reuse out.logits
    # for token entropy so there's no second forward pass.
    print("Running forward (with all hidden states)...", flush=True)
    response_len = response_ids.shape[1]

    # Only extract the layers we'll actually pool — keeps CPU memory bounded.
    needed_layers: set[int] = set()
    for layer in layers:
        if isinstance(layer, int):
            needed_layers.add(layer)
        else:
            for x in layer:
                needed_layers.add(int(x))

    with torch.no_grad():
        out = model(
            input_ids=full_input.cuda(),
            attention_mask=full_attn.cuda(),
            output_hidden_states=True,
            use_cache=False,
        )

    # Token entropy from logits, chunked over batch. Materializing the full
    # fp32 softmax (bs × T × |V|) was the OOM source; chunk_size=8 caps the
    # transient at ~5 GB.
    print("Computing token entropy from logits (chunked)...", flush=True)
    chunk_size = 8
    rl_view = out.logits[:, -response_len - 1 : -1, :]
    ent_parts: list[torch.Tensor] = []
    for i in range(0, rl_view.shape[0], chunk_size):
        lp = F.log_softmax(rl_view[i : i + chunk_size].float(), dim=-1)
        ent_parts.append((-(lp.exp() * lp).sum(dim=-1)).cpu())
        del lp
    token_entropy = torch.cat(ent_parts, dim=0)
    del rl_view

    # Extract only needed layers; keep bf16 on CPU (fp32 cast at use site).
    total_layers = len(out.hidden_states)
    needed_pos = {(i if i >= 0 else total_layers + i) for i in needed_layers}
    response_hs_per_layer: list[torch.Tensor | None] = [None] * total_layers
    for idx in needed_pos:
        response_hs_per_layer[idx] = out.hidden_states[idx][:, -response_len:, :].cpu()

    del out
    torch.cuda.empty_cache()

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
