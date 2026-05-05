#!/usr/bin/env bash
# Phase 2 / M5 — default + per_sent_adv=OFF × seed=42 × ep1 (5.5h)
# Bug A 消融：关掉 sentencepo loss 内部的逐句 advantage（恢复到 v1-5 修 Bug A 之前）。
# 0428 实验：per_sent ON ep3 = 0.340，OFF ep3 = 0.263（差 7.7pp）。
# 本次跑 ep2 验证修 Bug A 是 v1-5 必要前提。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase2_persent_OFF" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=false \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=-18 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
