#!/usr/bin/env bash
# Phase 1 / M2 — v1-5 winner_config × Qwen3 × seed=37 × ep3
# multi-seed 主表的第三个 seed。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-05_phase1_winner_qwen3" \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=37 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
