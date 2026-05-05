#!/usr/bin/env bash
# Phase 1 / M7 — v1-5 winner_config × Llama × seed=9 × ep3
#
# Llama 跨模型 multi-seed 第二个 seed（已有 phase0_M9 = seed=42）。
# 复用 winner_config（默认 = M2 default = L=-18 + slpa 0.05/0.05 + scr 0.02/0.02）。
# 注意：Llama-3.2-3B 只有 28 层，hidden_layer_index=-18 等价于 layer 10
# （28-18=10），相对深度 ~36%，比 Qwen3 的 50% 偏浅。这是已知 caveat，
# 但保持配置一致是公平比较的代价。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-05_phase1_winner_llama" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=9 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
