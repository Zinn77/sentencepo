#!/usr/bin/env bash
# Phase 2 b2 — SCR-only ablation × winner_base × seed=42
# Module ablation：保留 SCR，关闭 SLPA。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_ENABLE=false

EXP_TAG="2026-05-07_phase2_b2_scr_only_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
