#!/usr/bin/env bash
# Day 1 / Machine 1 — Hidden Phase B: combined SCR+SLPA at layer=-9
#
# 层消融第二个采样点（-1 控制 / -9 / -18 / -27 四点曲线）。
# Phase A 显示 -9/last gap=0.0792，显著低于 -18/last gap=0.1285，
# 但仍高于 -1/last gap=0.0568。预期：RL 表现 -9 介于 -1 和 -18 之间。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-04_day1_M1_hidden_combined_L-9_ep3" \
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
  EPOCHS=3 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
