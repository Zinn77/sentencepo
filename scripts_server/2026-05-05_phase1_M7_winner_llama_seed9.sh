#!/usr/bin/env bash
# Phase 1 / M7 — v1-5 winner_config × Llama × seed=9 × ep1
#
# Llama 跨模型 multi-seed 第二个 seed（M6 跑 seed=42）。
# winner_config = smaller_alpha (slpa 0.02 + scr 0.01).
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-05_phase1_winner_llama" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=9 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
