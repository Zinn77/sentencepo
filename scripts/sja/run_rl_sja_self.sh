#!/usr/bin/env bash
set -euo pipefail

# Strict self-judge mode: actor and judge share the same model weights.
export sentence_judge_enable=${sentence_judge_enable:-true}
export sentence_judge_backend=${sentence_judge_backend:-self}
export sentence_judge_alpha=${sentence_judge_alpha:-0.1}

# Keep judge prompts tractable during early experiments.
export sentence_judge_max_sentences=${sentence_judge_max_sentences:-64}
export sentence_judge_max_chars=${sentence_judge_max_chars:-384}
export sentence_judge_max_tokens=${sentence_judge_max_tokens:-512}
export sentence_judge_truncate_prompt=${sentence_judge_truncate_prompt:-true}

# Use the distilled judge-SFT checkpoint as actor init (override as needed).
export MODEL_PATH=${MODEL_PATH:-$HOME/autodl-tmp/models_v1-4-metrics-sft/judge_sft_qwen3_4b_judge_sft_ep1_lr2e-6}

cd "$HOME/sentencepo_v1-4-metrics-sft"
bash test_sentencepo_v1-4.sh "$@"
