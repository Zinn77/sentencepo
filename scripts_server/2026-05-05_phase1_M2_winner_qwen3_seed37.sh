#!/usr/bin/env bash
# Phase 1 / M2 — v1-5 winner_config × Qwen3 × seed=37 × ep1
#
# winner_config = smaller_alpha（slpa 0.02 + scr 0.01）。
# Phase 1 主表 ep1 multi-seed 第三个 seed
# (seed=42 from Phase 0, seed=9 from Phase 1/M1).
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-05_phase1_winner_qwen3" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=37 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
