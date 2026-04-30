# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared sentence-embedding pooling for v1-5-hidden ablation.

Six pooling strategies share boundary-detection and sample-index plumbing,
so the bucket / SLPA / SCR advantage modules and the offline diagnostic
script all consume sentence embeddings via one entry point.
"""

from __future__ import annotations

import torch

VALID_POOLINGS = (
    "last",
    "mean",
    "first",
    "mean_no_punct",
    "entropy_weighted",
    "diff",
)


def _sentence_boundaries(
    sentence_ids: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (first_token_mask, last_token_mask) over (B, T)."""
    next_sid = torch.roll(sentence_ids, shifts=-1, dims=1)
    next_valid = torch.roll(valid, shifts=-1, dims=1)
    last_pos_mask = torch.zeros_like(valid)
    last_pos_mask[:, -1] = True
    last_boundary = last_pos_mask | (sentence_ids != next_sid) | (~next_valid)

    prev_sid = torch.roll(sentence_ids, shifts=1, dims=1)
    prev_valid = torch.roll(valid, shifts=1, dims=1)
    first_pos_mask = torch.zeros_like(valid)
    first_pos_mask[:, 0] = True
    first_boundary = first_pos_mask | (sentence_ids != prev_sid) | (~prev_valid)

    last_mask = valid & last_boundary
    first_mask = valid & first_boundary
    return first_mask, last_mask


def pool_sentence_embeddings(
    token_hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
    response_mask: torch.Tensor,
    pooling: str,
    *,
    response_token_ids: torch.Tensor | None = None,
    token_entropy: torch.Tensor | None = None,
    punct_token_ids: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pool token-level hidden states into per-sentence embeddings.

    Args:
        token_hidden_states: (B, T, H) hidden states.
        sentence_ids: (B, T) sentence id per token (>=0 for valid tokens).
        response_mask: (B, T) response mask (1 for response tokens).
        pooling: one of VALID_POOLINGS.
        response_token_ids: (B, T) token ids; required for ``mean_no_punct``.
        token_entropy: (B, T) per-token entropy; required for ``entropy_weighted``.
        punct_token_ids: 1-D tensor of token ids treated as punctuation.

    Returns:
        ``(sent_emb, unique_sentence_ids, sample_index)`` or ``None`` if no
        valid sentences. Shapes: (S, H), (S,), (S,) where S is number of
        unique sentences.
    """
    if pooling not in VALID_POOLINGS:
        raise ValueError(f"Unknown pooling '{pooling}'. Valid: {VALID_POOLINGS}")

    device = token_hidden_states.device
    sentence_ids = sentence_ids.to(device)
    response_mask = response_mask.to(device)
    bs, seq_len, hidden = token_hidden_states.shape

    valid = (response_mask > 0) & (sentence_ids >= 0)
    if not torch.any(valid):
        return None

    flat_sid = sentence_ids.view(-1)
    flat_valid = valid.view(-1)
    flat_sid_valid = flat_sid[flat_valid]
    if flat_sid_valid.numel() == 0:
        return None

    unique_sid, inv = torch.unique(flat_sid_valid, return_inverse=True)
    num_sent = unique_sid.numel()
    if num_sent == 0:
        return None

    first_mask, last_mask = _sentence_boundaries(sentence_ids, valid)
    flat_first = first_mask.view(-1)
    flat_last = last_mask.view(-1)
    flat_sid_last = flat_sid[flat_last]
    flat_sid_first = flat_sid[flat_first]
    if flat_sid_last.numel() == 0 or flat_sid_first.numel() == 0:
        return None

    idx_last = torch.searchsorted(unique_sid, flat_sid_last)
    idx_first = torch.searchsorted(unique_sid, flat_sid_first)

    flat_emb = token_hidden_states.view(-1, hidden)
    dtype = token_hidden_states.dtype

    def _last_emb():
        out = torch.zeros((num_sent, hidden), device=device, dtype=dtype)
        out.index_copy_(0, idx_last, flat_emb[flat_last])
        return out

    def _first_emb():
        out = torch.zeros((num_sent, hidden), device=device, dtype=dtype)
        out.index_copy_(0, idx_first, flat_emb[flat_first])
        return out

    def _weighted_mean(weights: torch.Tensor) -> torch.Tensor:
        # weights: (num_valid,) non-negative; flat_emb_valid: (num_valid, H)
        flat_emb_valid = flat_emb[flat_valid]
        w = weights.to(dtype).view(-1, 1)
        sum_emb = torch.zeros((num_sent, hidden), device=device, dtype=dtype)
        sum_w = torch.zeros((num_sent, 1), device=device, dtype=dtype)
        sum_emb.index_add_(0, inv, flat_emb_valid * w)
        sum_w.index_add_(0, inv, w)
        return sum_emb / (sum_w + eps)

    if pooling == "last":
        sent_emb = _last_emb()
    elif pooling == "first":
        sent_emb = _first_emb()
    elif pooling == "mean":
        ones = torch.ones(int(flat_valid.sum().item()), device=device, dtype=dtype)
        sent_emb = _weighted_mean(ones)
    elif pooling == "diff":
        sent_emb = _last_emb() - _first_emb()
    elif pooling == "mean_no_punct":
        if response_token_ids is None or punct_token_ids is None:
            raise ValueError("mean_no_punct requires response_token_ids and punct_token_ids")
        flat_token_ids = response_token_ids.to(device).view(-1)[flat_valid]
        is_punct = torch.isin(flat_token_ids, punct_token_ids.to(device))
        weights = (~is_punct).to(dtype)
        # If a sentence is *all* punctuation, fall back to last-token to avoid /0.
        sent_emb_main = _weighted_mean(weights)
        # Detect zero-weight sentences.
        sum_w = torch.zeros((num_sent, 1), device=device, dtype=dtype)
        sum_w.index_add_(0, inv, weights.view(-1, 1))
        empty_mask = (sum_w.squeeze(-1) <= eps).unsqueeze(-1)
        if torch.any(empty_mask):
            sent_emb_main = torch.where(empty_mask, _last_emb(), sent_emb_main)
        sent_emb = sent_emb_main
    elif pooling == "entropy_weighted":
        if token_entropy is None:
            raise ValueError("entropy_weighted requires token_entropy")
        flat_ent = token_entropy.to(device).view(-1)[flat_valid].clamp(min=0.0)
        weights = 1.0 / (1.0 + flat_ent)
        sent_emb = _weighted_mean(weights)
    else:  # pragma: no cover
        raise ValueError(f"Unhandled pooling: {pooling}")

    flat_sample_idx = (
        torch.arange(bs, device=device).unsqueeze(1).expand(bs, seq_len).reshape(-1)
    )
    sent_sample_idx = torch.zeros((num_sent,), device=device, dtype=torch.long)
    sent_sample_idx.index_copy_(0, idx_last, flat_sample_idx[flat_last])

    return sent_emb, unique_sid, sent_sample_idx


def build_punct_token_ids(tokenizer) -> torch.Tensor:
    """Best-effort enumeration of punctuation/whitespace token ids in a tokenizer.

    Used by ``mean_no_punct`` pooling.
    """
    punct_chars = set(".,!?;:—-()[]{}\"'`\n\r\t ，。！？；：、（）【】《》「」“”‘’")
    ids: list[int] = []
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if vocab is None:
        return torch.tensor([], dtype=torch.long)
    for token_id in vocab.values():
        try:
            s = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except Exception:
            continue
        s_strip = s.strip()
        if not s_strip:
            ids.append(int(token_id))
            continue
        if all(c in punct_chars for c in s_strip):
            ids.append(int(token_id))
    return torch.tensor(sorted(set(ids)), dtype=torch.long)
