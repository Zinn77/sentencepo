"""Shared sentence splitting utilities for SentencePO.

All sentence splitting in training and data construction should use these
functions to guarantee consistent sentence boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

PUNCTUATION_CHARS = {".", "?", "!", "。", "？", "！"}


def split_token_positions(
    tokenizer,
    token_ids: list[int],
    positions: list[int],
    min_sent_tokens: int = 6,
) -> list[list[int]]:
    """Split a sequence of token positions into sentence groups.

    This is the canonical sentence splitting logic. It decodes each token and
    checks for punctuation or newline characters, then merges short sentences.

    Args:
        tokenizer: HuggingFace tokenizer (needs ``.decode()``).
        token_ids: Token id for each position (same length as *positions*).
        positions: Positional indices corresponding to each token.
        min_sent_tokens: Sentences shorter than this are merged into neighbours.

    Returns:
        List of sentence groups, where each group is a list of position indices.
    """
    min_sent_tokens = max(1, int(min_sent_tokens))

    sentences: list[list[int]] = []
    current: list[int] = []
    for tid, pos in zip(token_ids, positions):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        current.append(pos)
        if "\n" in token_str or any(ch in token_str for ch in PUNCTUATION_CHARS):
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    if not sentences:
        return sentences

    # Merge short sentences: prefer merging into next; if last, merge into previous.
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

    return sentences


def build_sentence_ids_1d(
    tokenizer,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    min_sent_tokens: int = 6,
) -> "torch.Tensor":
    """Assign sentence ids for a single 1-D token sequence.

    Used by ``rl_dataset.py`` during data loading (prompt + response).

    Returns:
        Tensor same shape as *input_ids* with sentence ids (>= 0) for valid
        tokens and ``-1`` elsewhere.
    """
    import torch

    sentence_ids = torch.full_like(input_ids, fill_value=-1)
    valid_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_positions.numel() == 0:
        return sentence_ids

    pos_list = valid_positions.tolist()
    tid_list = [int(input_ids[p].item()) for p in pos_list]

    sentences = split_token_positions(tokenizer, tid_list, pos_list, min_sent_tokens)
    for sid, sent in enumerate(sentences):
        for pos in sent:
            sentence_ids[pos] = sid

    return sentence_ids


def build_sentence_ids_batch(
    tokenizer,
    responses: "torch.Tensor",
    response_mask: "torch.Tensor",
    min_sent_tokens: int = 6,
) -> "torch.Tensor":
    """Assign sentence ids for a batch of response sequences.

    Used by ``ray_trainer.py`` during rollout to generate sentence ids for
    newly generated responses.

    Returns:
        ``torch.LongTensor`` of same shape as *responses*. Valid tokens get
        non-negative sentence ids (offset per sample to avoid collisions when
        flattened); invalid tokens get ``-1``.
    """
    import torch

    assert responses.shape == response_mask.shape
    batch_size, seq_len = responses.shape
    sentence_ids = torch.full_like(responses, fill_value=-1, dtype=torch.long)

    for b in range(batch_size):
        valid_positions = (response_mask[b] > 0).nonzero(as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue

        pos_list = valid_positions.tolist()
        tid_list = [int(responses[b, idx].item()) for idx in pos_list]

        sentences = split_token_positions(tokenizer, tid_list, pos_list, min_sent_tokens)
        for sid, sent in enumerate(sentences):
            for idx in sent:
                sentence_ids[b, idx] = sid

    # Offset ids per sample to avoid cross-sample mixing when flattened.
    for b in range(batch_size):
        mask = sentence_ids[b] >= 0
        if mask.any():
            sentence_ids[b, mask] += b * (seq_len + 1)

    return sentence_ids


def split_text_to_sentences(
    tokenizer,
    text: str,
    min_sent_tokens: int = 6,
    max_sentences: int = 0,
    max_chars: int = 0,
) -> list[str]:
    """Split plain text into sentences using the same token-level logic as RL training.

    This tokenizes the text, runs :func:`split_token_positions`, then decodes
    each sentence group back to text. The result is guaranteed to match the
    sentence boundaries used during RL training.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Raw response text.
        min_sent_tokens: Minimum tokens per sentence (short ones are merged).
        max_sentences: Keep at most this many sentences (0 = no limit).
        max_chars: Truncate each sentence text to this many chars (0 = no limit).

    Returns:
        List of sentence text strings.
    """
    text = text.strip()
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return [text[:max_chars] if max_chars > 0 else text] if text else []

    positions = list(range(len(token_ids)))
    sentences = split_token_positions(tokenizer, token_ids, positions, min_sent_tokens)

    sentence_texts: list[str] = []
    for sent_positions in sentences:
        sent_token_ids = [token_ids[p] for p in sent_positions]
        sent_text = tokenizer.decode(sent_token_ids, skip_special_tokens=True).strip()
        if sent_text:
            if max_chars > 0:
                sent_text = sent_text[:max_chars]
            sentence_texts.append(sent_text)

    if max_sentences > 0:
        sentence_texts = sentence_texts[:max_sentences]

    return sentence_texts
