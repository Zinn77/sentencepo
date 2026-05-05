#!/usr/bin/env bash
# Phase 2 / M4 — SCR-only × L=-18 × seed=42 × ep1 (5.5h)
# 模块消融：关 SLPA，仅 SCR，看 SCR 单独是否还有效果。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase2_scr_only_L-18" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=false \
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
