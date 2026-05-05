#!/usr/bin/env bash
# Day 1 / Machine 2 — Hidden Phase B: combined SCR+SLPA at layer=-27
#
# 层消融第三个采样点。Phase A 显示 -27/last gap=0.0711，比 -18 低，
# 但仍高于 -1。验证 -18 是真正的甜点（而非 sweep 边缘伪冠军）。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-04_day1_M2_hidden_combined_L-27_ep3" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-27 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=-27 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
