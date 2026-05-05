#!/usr/bin/env bash
# Phase 1 / M1 — v1-5 winner_config × Qwen3 × seed=9 × ep3
#
# winner_config 由 _phase1_winner_config.sh 提供（Phase 0 完成后由 user 编辑）。
# 默认 = M2 default (slpa 0.05/0.05 + scr 0.02/0.02 + L=-18 + last)。
# multi-seed 主表的第二个 seed（已有 seed=42 = M2 default running）。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-05_phase1_winner_qwen3" \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=9 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
