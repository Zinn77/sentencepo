#!/usr/bin/env bash
# 2026-05-07 / Qwen3 翻盘 follow-up — SCR bump 0.1 × seed=37 ep1
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SCR_ALPHA_C=0.1
export SCR_ALPHA_I=0.1

EXP_TAG="2026-05-07_qwen3_v15_scr_bump01" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=37 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
