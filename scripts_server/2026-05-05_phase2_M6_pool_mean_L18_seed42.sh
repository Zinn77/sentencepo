#!/usr/bin/env bash
# Phase 2 / M6 — combined L=-18 + pooling=Mean × seed=42 × ep2 (12h)
# Pooling 消融：替换 last 为 mean，验证 Phase A "pooling 在固定层后影响小" 的结论。
# Phase A: L=-18/Last gap=0.128, L=-18/Mean gap=0.110，差 0.018（噪声范围）。
# 在 RL 上是否同样小差异？
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase2_pool_mean_L-18" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-18 \
  SLPA_POOL=mean \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=-18 \
  SCR_POOL=mean \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=2 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
