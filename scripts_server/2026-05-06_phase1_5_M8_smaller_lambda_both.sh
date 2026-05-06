#!/usr/bin/env bash
# Phase 1.5 / M8 — smaller_alpha + LAMBDA_PPL=0.5 + LAMBDA_LEN=0.5（双开 + 激进）
#
# 假说：M6 (PPL) 和 M7 (len) 是单变量探针；M8 同时打开两个 λ 且用更激进
# 0.5 让自适应 clip 真正发挥作用 —— 难句 + 短句 → 双重放松；易句 + 长句
# → 双重收紧。如果 M6/M7 单独有效但不显著，M8 检验"两个机制是否互补"。
#
# clip scale 范围已 cmin=0.5 / cmax=1.5，0.5 配 z=±1 不会撞到 clamp。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M8_smaller_lambda_both" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  LAMBDA_PPL=0.5 \
  LAMBDA_LEN=0.5 \
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
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
