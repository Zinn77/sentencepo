#!/usr/bin/env bash
# 2026-05-07 / Llama slpa-bump s42 — winner_config + SLPA α 0.02→0.1 × Llama ep1
#
# Qwen3 上 slpa-bump (SLPA α=0.1) s42 peak=end=0.3817，新 winner config。
# Llama 之前用 winner (SLPA 0.02)，本次试 slpa-bump 是否 Llama 上也更强。
# 若 peak ≥ 0.20 + ep1_end > 0.18，论文双模型 winner 都用 slpa-bump 统一。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_ALPHA_C=0.1
export SLPA_ALPHA_I=0.1

EXP_TAG="2026-05-07_llama_v15_slpa_bump01" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
