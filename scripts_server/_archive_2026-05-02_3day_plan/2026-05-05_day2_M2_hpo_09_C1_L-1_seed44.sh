#!/usr/bin/env bash
# Day 2 / Machine 2 — hpo_09_C1 (L=-1) seed=44，方差估计 baseline
#
# 与 Day 0 M1 (seed=42) + Day 2 M1 (seed=43) 凑 3-seed baseline。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_day2_M2_hpo_09_C1_L-1_seed44" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-1 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=-1 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=44 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
