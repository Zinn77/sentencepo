#!/usr/bin/env bash
# Phase 2 / M3 — SLPA-only × L=-18 × seed=42 × ep1 (5.5h)
# 模块消融：关 SCR，仅 SLPA，看 SLPA 单独是否还有效果。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase2_slpa_only_L-18" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=false \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
