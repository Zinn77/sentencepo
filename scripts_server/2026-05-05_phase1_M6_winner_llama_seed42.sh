#!/usr/bin/env bash
# Phase 1 / M6 — v1-5 winner_config × Llama × seed=42 × ep1
#
# 用户决策（2026-05-06）：DAPO 砍掉，腾出 M6 槽给 v1-5 Llama 多 seed。
# Phase 0 v1-5 Llama 跑的是 default config (slpa 0.05/0.05 + scr 0.02/0.02)，
# 不是 winner_config (smaller_alpha)。本次 fresh 重跑 winner_config + seed=42。
#
# Llama-3.2-3B 只有 28 层，hidden_layer_index=-18 等价于 layer 10
# （28-18=10），相对深度 ~36%，比 Qwen3 的 50% 偏浅。已知 caveat。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-05_phase1_winner_llama" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
