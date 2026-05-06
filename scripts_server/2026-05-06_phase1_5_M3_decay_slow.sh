#!/usr/bin/env bash
# Phase 1.5 / M3 — default α + alpha_decay + alpha_min_ratio=0.5
#
# 假说：min_ratio=0.1 太激进（信号几乎消失），min_ratio=0.5 衰减到一半，
# 保留中后期 SLPA/SCR 信号；如果 P0_alpha_decay (min_ratio=0.1, ep1_end 0.3751)
# 已是 ep1_end best，min_ratio 太低可能浪费了 SLPA/SCR 的中期价值。
#
# 与 M2 (min_ratio=0.0) 形成 0/0.1/0.5 三点曲线（已有 P0_alpha_decay=0.1）。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M3_decay_slow" \
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
  ALPHA_MIN_RATIO=0.5 \
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
