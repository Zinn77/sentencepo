#!/usr/bin/env bash
# Phase 1.5 / M4 — asym：correct ≫ incorrect（5x ratio）
#
# 假说：P0_alpha_asym 已经测过 2.5x ratio (slpa 0.05/0.02 + scr 0.02/0.01)
# 但 ep1 peak 只有 0.3367 → 失败。我们怀疑 2.5x 力度不够极端，没真正
# 表达"错答几乎不动 + 正确强推"的语义。本次拉到 5x：incorrect 信号几乎
# 关闭，让 SLPA/SCR 几乎只在正确 rollout 上起作用。
#
# 与 M5 (反方向 asym, incorrect 强) 配对，完整探索 asym 维度。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M4_asym_correct_strong_5x" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.01 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.005 \
  SCR_LAYER=-18 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
