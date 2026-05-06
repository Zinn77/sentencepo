#!/usr/bin/env bash
# Phase 1.5 / M2 — default α + alpha_decay + alpha_min_ratio=0.0
#
# 假说：P0_alpha_decay 用 default α (0.05/0.02) + min_ratio=0.1，ep1_end 0.3751；
# 把 min_ratio 降到 0 让 SLPA/SCR 在 ep1 末期完全归零，等价于"前期 RL with sentence
# advantage，后期纯 GRPO"——理论上能保持 GRPO baseline ep1_end 水平。
#
# 对 ep1_end 是直接探针：如果 0.3751 → 0.385+ 说明 min_ratio=0.1 仍然有残留干扰。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M2_decay_to_zero" \
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
  ALPHA_MIN_RATIO=0.0 \
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
