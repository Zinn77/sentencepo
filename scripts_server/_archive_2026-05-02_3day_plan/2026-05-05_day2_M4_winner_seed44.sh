#!/usr/bin/env bash
# Day 2 / Machine 4 — winner seed=44，方差估计 winner
#
# 同 M3，仅 SEED 改 44。winner config 通过 env 覆盖（见 M3 注释）。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_day2_M4_winner_seed44" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=${SLPA_ENABLE:-true} \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=${SLPA_LAYER:--18} \
  SLPA_POOL=${SLPA_POOL:-last} \
  SCR_ENABLE=${SCR_ENABLE:-true} \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=${SCR_LAYER:--18} \
  SCR_POOL=${SCR_POOL:-last} \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=44 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
