#!/usr/bin/env bash
# Phase 0 / M5 — α 减半（slpa 0.02 + scr 0.01），ep1 调参
#
# 假说：当前 default α (slpa 0.05 + scr 0.02) 太大，导致 ep3 collapse；
# 减半看是否拉长有效训练窗口（ep1 / 早期 peak 应仍接近，ep2/ep3 应更稳）。
# 5.5h 跑完 ep1，立即比较 peak vs default M2 (peak 0.3797).
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase0_tune_smaller_alpha" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.02 \
  SLPA_ALPHA_I=0.02 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.01 \
  SCR_ALPHA_I=0.01 \
  SCR_LAYER=-18 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
