#!/usr/bin/env bash
# Phase 2 / M1 — combined v1-5 × L=-9 × seed=42 × ep1 (5.5h)
# 层曲线消融点：L=-1 (M1 已有) / L=-9 (本) / L=-18 (M2 default 已有) / L=-27 (M2)
# 验证 Phase A "中间层 -18 是 sweet spot" 是否在 RL 上保持单峰形状。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase2_combined_L-9" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-9 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=-9 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
