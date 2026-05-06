#!/usr/bin/env bash
# Phase 1.5 / M5 — asym 反方向：incorrect ≫ correct（3x ratio）
#
# 假说：从未测过的 asym 反方向。直觉：错误 rollout 的句子特别需要被
# "推到正确中心"（SCR）+ 标注负向时序差分（SLPA），而正确 rollout 已经
# 在好状态、不需要太多句子级修正。incorrect=3·correct 给错误样本更强
# 信号但不极端（怕 5x 把错答推飞）。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M5_asym_incorrect_strong_3x" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.01 \
  SLPA_ALPHA_I=0.03 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.005 \
  SCR_ALPHA_I=0.015 \
  SCR_LAYER=-18 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
