#!/usr/bin/env bash
# 2026-05-07 / Llama alpha_decay s42 — winner + ALPHA_DECAY=linear × Llama ep1
#
# 备份方案：若 slpa-bump Llama s42 输 winner，用 alpha_decay 作 Llama 备选 winner。
# Qwen3 上 alpha_decay s42 peak=end=0.3751 不退化，stability 强。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export ALPHA_DECAY=linear
export ALPHA_MIN_RATIO=0.1

EXP_TAG="2026-05-07_llama_v15_alpha_decay" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
