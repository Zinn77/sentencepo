#!/usr/bin/env bash
set -euo pipefail

# Split-judge mode: actor model and judge model are decoupled.
export sentence_judge_enable=${sentence_judge_enable:-true}
export sentence_judge_backend=${sentence_judge_backend:-callable}
export sentence_judge_fn=${sentence_judge_fn:-verl.utils.judge_self:judge_fn}
export sentence_judge_alpha=${sentence_judge_alpha:-0.1}

# Keep judge request size bounded.
export sentence_judge_max_sentences=${sentence_judge_max_sentences:-64}
export sentence_judge_max_chars=${sentence_judge_max_chars:-384}
export sentence_judge_max_tokens=${sentence_judge_max_tokens:-512}

# Judge server config consumed by verl.utils.judge_self:judge_fn.
export SJA_JUDGE_BASE_URL=${SJA_JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}
export SJA_JUDGE_MODEL=${SJA_JUDGE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
export SJA_JUDGE_API_KEY=${SJA_JUDGE_API_KEY:-${DASHSCOPE_API_KEY:-}}
export SJA_JUDGE_MAX_TOKENS=${SJA_JUDGE_MAX_TOKENS:-512}
export SJA_JUDGE_TIMEOUT_S=${SJA_JUDGE_TIMEOUT_S:-60}
export SJA_JUDGE_RETRIES=${SJA_JUDGE_RETRIES:-2}

cd "$HOME/sentencepo_v1-4-metrics-sft"
bash test_sentencepo_v1-4.sh "$@"
