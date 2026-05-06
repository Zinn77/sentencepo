#!/usr/bin/env bash
# Phase 2 a2 — Hidden layer ablation × L=-9 × winner_base × seed=42
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_LAYER=-9
export SCR_LAYER=-9

EXP_TAG="2026-05-07_phase2_a2_layer_neg9_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
