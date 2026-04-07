"""Judge SFT data loading and iteration for mixed loss during RL training.

Pre-tokenizes distilled judge parquet data (prompt + response) and provides
a cycling micro-batch iterator used by dp_actor.py to compute judge SFT loss.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def prepare_judge_sft_data(
    tokenizer: Any,
    data_path: str,
    max_seq_len: int = 2048,
) -> list[dict[str, list[int]]]:
    """Load parquet and pre-tokenize into (input_ids, labels) pairs.

    The parquet must have ``prompt`` and ``response`` string columns.
    Labels are -100 for prompt tokens (no loss) and token ids for response tokens.
    """
    import pandas as pd

    df = pd.read_parquet(data_path)
    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError(f"Judge SFT parquet must have 'prompt' and 'response' columns, got {list(df.columns)}")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    samples: list[dict[str, list[int]]] = []

    for _, row in df.iterrows():
        prompt_text = str(row["prompt"])
        response_text = str(row["response"])

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)

        # Append EOS if the tokenizer has one
        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + response_ids
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            prompt_len = len(prompt_ids)
            response_ids = full_ids[prompt_len:]
        else:
            prompt_len = len(prompt_ids)

        # Labels: -100 for prompt tokens, actual ids for response tokens
        labels = [-100] * prompt_len + list(full_ids[prompt_len:])

        if len(response_ids) < 2:
            continue

        samples.append({"input_ids": full_ids, "labels": labels})

    if not samples:
        raise RuntimeError(f"No valid judge SFT samples from {data_path}")

    logger.info("Loaded %d judge SFT samples from %s (max_seq_len=%d)", len(samples), data_path, max_seq_len)
    return samples


class JudgeSFTIterator:
    """Cycling micro-batch iterator over pre-tokenized judge SFT data."""

    def __init__(self, samples: list[dict[str, list[int]]], micro_batch_size: int, pad_token_id: int = 0):
        self.samples = samples
        self.micro_batch_size = micro_batch_size
        self.pad_token_id = pad_token_id
        self.indices = list(range(len(samples)))
        random.shuffle(self.indices)
        self.pos = 0

    def next_batch(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Return a padded micro-batch on the given device."""
        if self.pos + self.micro_batch_size > len(self.indices):
            random.shuffle(self.indices)
            self.pos = 0

        batch_indices = self.indices[self.pos : self.pos + self.micro_batch_size]
        self.pos += self.micro_batch_size

        batch_samples = [self.samples[i] for i in batch_indices]
        max_len = max(len(s["input_ids"]) for s in batch_samples)

        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for s in batch_samples:
            pad_len = max_len - len(s["input_ids"])
            input_ids_list.append(s["input_ids"] + [self.pad_token_id] * pad_len)
            labels_list.append(s["labels"] + [-100] * pad_len)
            attention_mask_list.append([1] * len(s["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long, device=device),
            "labels": torch.tensor(labels_list, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long, device=device),
        }
