#!/usr/bin/env bash
# Phase 0 / M6 — default + alpha_decay=linear，ep1 调参
#
# 假说：reward variance 在 ep2/ep3 消失后，SLPA/SCR 信号变成噪声放大器；
# linear decay 让 α 随训练 progress 线性衰减到 alpha_min_ratio (默认 0.1)。
# 0428 实验也跑过 decay，但当时 default α 偏大；本次小α + decay 一起看。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase0_tune_alpha_decay" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
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
  ALPHA_DECAY=linear \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
