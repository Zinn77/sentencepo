#!/usr/bin/env python3
"""Clean rollout JSONL and split into train/val for judge distillation.

Responsibilities:
- Read rollout_debug/*.jsonl
- Clean responses (trim repetitive tails, compress emoji runs, cap length)
- Split by prompt group into train/val (same prompt never appears in both)
- Output clean_train.jsonl / clean_val.jsonl (with step field preserved for later phase splitting)

Phase splitting is NOT done here. After teacher distillation, phases can be
created from the distilled parquet by filtering on the ``step`` column.
"""
from __future__ import annotations

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import glob
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root is importable when running as a script.
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


FINAL_ANSWER_RE = re.compile(r"final\s+answer", flags=re.IGNORECASE)
CHECK_RUN_RE = re.compile(r"✅{4,}")
BOXED_RE = re.compile(r"\\boxed\{[^{}]*\}")


def normalize_prompt_key(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def parse_step(path: Path, row: dict[str, Any]) -> int:
    step = row.get("step", None)
    if isinstance(step, int):
        return step
    if isinstance(step, str) and step.isdigit():
        return int(step)

    stem = path.stem
    if stem.isdigit():
        return int(stem)

    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else -1


def split_sentences(tokenizer, text: str, min_sent_tokens: int = 6) -> list[str]:
    """Split text using the same token-level logic as RL training."""
    from verl.utils.sentence_utils import split_text_to_sentences

    return split_text_to_sentences(tokenizer, text, min_sent_tokens=min_sent_tokens)


def normalize_sentence_for_dedupe(sentence: str) -> str:
    s = sentence.lower()
    s = CHECK_RUN_RE.sub("✅", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def trim_output(
    tokenizer,
    text: str,
    max_sentences: int,
    max_chars: int,
    keep_repeat_tail_sentences: int,
    max_consecutive_same: int,
    min_sent_tokens: int = 6,
    pre_split_sentences: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    raw_text = text.strip()
    if pre_split_sentences:
        raw_sentences = [s.strip() for s in pre_split_sentences if str(s).strip()]
    else:
        raw_sentences = split_sentences(tokenizer, raw_text, min_sent_tokens)

    # Fallback for malformed content that fails sentence split.
    if not raw_sentences:
        cleaned = raw_text[:max_chars]
        return cleaned, {
            "raw_sentence_count": 0,
            "kept_sentence_count": 1 if cleaned else 0,
            "repeat_onset_idx": -1,
            "had_long_check_runs": bool(CHECK_RUN_RE.search(raw_text)),
            "boxed_count": len(BOXED_RE.findall(raw_text)),
            "final_answer_count": len(FINAL_ANSWER_RE.findall(raw_text)),
        }

    norm_sentences = [normalize_sentence_for_dedupe(s) for s in raw_sentences]

    # Detect repetitive onset by repeated normalized sentence blocks.
    repeat_onset = -1
    block_size = 3
    seen_blocks: dict[tuple[str, ...], int] = {}
    for i in range(0, max(0, len(norm_sentences) - block_size + 1)):
        block = tuple(norm_sentences[i : i + block_size])
        if block in seen_blocks:
            repeat_onset = i
            break
        seen_blocks[block] = i

    kept: list[str] = []
    same_run = 0
    last_norm = None

    # Keep head plus a short repetitive tail so teacher can learn to penalize redundancy.
    hard_limit = max_sentences
    if repeat_onset >= 0:
        hard_limit = min(max_sentences, repeat_onset + keep_repeat_tail_sentences)

    for sent, norm in zip(raw_sentences, norm_sentences):
        if len(kept) >= hard_limit:
            break

        if norm == last_norm:
            same_run += 1
        else:
            same_run = 1
            last_norm = norm

        if same_run > max_consecutive_same:
            continue

        s = CHECK_RUN_RE.sub("✅✅✅", sent)
        kept.append(s)

    cleaned = "\n\n".join(kept).strip()
    kept_sentence_count = len(kept)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
        # Avoid a second expensive token-level split by estimating how many kept sentences remain after character truncation.
        # 如果按 max_chars 截断，则 kept_sentence_count 是估算
        budget = max_chars
        sep_len = 2
        consumed = 0
        kept_sentence_count = 0
        for idx, sent in enumerate(kept):
            extra = len(sent)
            if idx > 0:
                extra += sep_len
            if consumed + extra > budget:
                break
            consumed += extra
            kept_sentence_count += 1

    if cleaned and not cleaned.endswith((".", "!", "?", "。", "！", "？")):
        cleaned += "。"

    return cleaned, {
        "raw_sentence_count": len(raw_sentences),
        "kept_sentence_count": kept_sentence_count,
        "repeat_onset_idx": repeat_onset,
        "had_long_check_runs": bool(CHECK_RUN_RE.search(raw_text)),
        "boxed_count": len(BOXED_RE.findall(raw_text)),
        "final_answer_count": len(FINAL_ANSWER_RE.findall(raw_text)),
    }


@dataclass
class Sample:
    input: str
    output: str
    output_sentences: list[str] | None
    score: float
    step: int
    source_file: str
    group_key: str
    meta: dict[str, Any]


def read_rollout_rows(glob_pattern: str) -> list[Sample]:
    samples: list[Sample] = []
    for file_path in sorted(glob.glob(glob_pattern)):
        p = Path(file_path)
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = str(row.get("input", "")).strip()
                output = str(row.get("output", "")).strip()
                raw_output_sentences = row.get("output_sentences", None)
                output_sentences = None
                if isinstance(raw_output_sentences, list):
                    output_sentences = [str(s).strip() for s in raw_output_sentences if str(s).strip()]
                if not output and output_sentences:
                    output = "\n\n".join(output_sentences)
                if not prompt or not output:
                    continue
                try:
                    score = float(row.get("score", 0.0))
                except Exception:
                    score = 0.0
                step = parse_step(p, row)
                samples.append(
                    Sample(
                        input=prompt,
                        output=output,
                        output_sentences=output_sentences,
                        score=score,
                        step=step,
                        source_file=str(p),
                        group_key=normalize_prompt_key(prompt),
                        meta={},
                    )
                )
    return samples


def split_groups(samples: list[Sample], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    groups = sorted({s.group_key for s in samples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_ratio)) if groups else 0
    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])
    return train_groups, val_groups


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(samples: list[Sample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        rows.append(
            {
                "input": s.input,
                "output": s.output,
                "output_sentences": s.output_sentences or [],
                "score": s.score,
                "step": s.step,
                "source_file": s.source_file,
                "prompt_hash": hashlib.md5(s.group_key.encode("utf-8")).hexdigest()[:12],
                "cleaning": s.meta,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean rollout JSONL and split train/val for judge distillation.")
    p.add_argument("--input-glob", required=True, help="Glob for rollout jsonl, e.g. /path/rollout_debug/*.jsonl")
    p.add_argument("--out-dir", required=True, help="Output directory for cleaned train/val jsonl")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-sentences", type=int, default=96)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--keep-repeat-tail-sentences", type=int, default=12)
    p.add_argument("--max-consecutive-same", type=int, default=2)

    p.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace tokenizer name or path (must match RL training model).",
    )
    p.add_argument(
        "--min-sent-tokens",
        type=int,
        default=6,
        help="Minimum tokens per sentence (must match RL training config).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print cleaning progress every N samples (0 to disable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"Loaded tokenizer: {args.tokenizer}")

    samples = read_rollout_rows(args.input_glob)
    if not samples:
        raise RuntimeError("No valid samples found. Check --input-glob.")
    print(f"Loaded samples: {len(samples)}")

    # Clean and annotate each response while keeping some repetitive tail for negative teaching signal.
    cleaned_samples: list[Sample] = []
    t0 = time.time()
    log_every = max(0, int(args.log_every))
    total = len(samples)
    for idx, s in enumerate(samples, start=1):
        cleaned_output, meta = trim_output(
            tokenizer,
            s.output,
            max_sentences=args.max_sentences,
            max_chars=args.max_chars,
            keep_repeat_tail_sentences=args.keep_repeat_tail_sentences,
            max_consecutive_same=args.max_consecutive_same,
            min_sent_tokens=args.min_sent_tokens,
            pre_split_sentences=s.output_sentences,
        )
        if not cleaned_output.strip():
            continue
        s.output = cleaned_output
        s.output_sentences = [x.strip() for x in cleaned_output.split("\n\n") if x.strip()]
        s.meta = meta
        cleaned_samples.append(s)

        if log_every > 0 and (idx % log_every == 0 or idx == total):
            elapsed = max(1e-6, time.time() - t0)
            speed = idx / elapsed
            remaining = max(0, total - idx)
            eta_sec = int(remaining / max(speed, 1e-6))
            print(
                f"Cleaning progress: {idx}/{total} "
                f"({idx / total:.1%}), kept={len(cleaned_samples)}, "
                f"speed={speed:.1f} samples/s, eta={eta_sec}s"
            )

    train_groups, val_groups = split_groups(cleaned_samples, val_ratio=args.val_ratio, seed=args.seed)

    train_samples = [s for s in cleaned_samples if s.group_key in train_groups]
    val_samples = [s for s in cleaned_samples if s.group_key in val_groups]

    all_rows = build_rows(cleaned_samples)
    train_rows = build_rows(train_samples)
    val_rows = build_rows(val_samples)

    write_jsonl(out_dir / "clean_all.jsonl", all_rows)
    write_jsonl(out_dir / "clean_train.jsonl", train_rows)
    write_jsonl(out_dir / "clean_val.jsonl", val_rows)

    steps = sorted({int(r["step"]) for r in train_rows if int(r["step"]) >= 0})

    stats = {
        "input_glob": args.input_glob,
        "total_raw_samples": len(samples),
        "total_clean_samples": len(cleaned_samples),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "unique_prompts": len({s.group_key for s in cleaned_samples}),
        "observed_steps": steps,
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
