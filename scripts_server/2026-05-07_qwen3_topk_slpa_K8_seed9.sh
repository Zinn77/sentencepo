#!/usr/bin/env bash
# 2026-05-07 / Qwen3 翻盘 C — SLPA top-K=8 × seed=9 ep1（multi-seed follow-up）
# 仅在 seed=42 peak ≥ 0.38 后启动
set -e
cd "$HOME/sentencepo_v1-5-topk"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_TOP_K=8

EXP_TAG="2026-05-07_qwen3_topk8_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=9 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
