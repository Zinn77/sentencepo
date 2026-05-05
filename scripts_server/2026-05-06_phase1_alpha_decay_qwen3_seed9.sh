#!/usr/bin/env bash
# Phase 1 / 次优 winner — alpha_decay × Qwen3 × seed=9 × ep1
#
# 用户决策（2026-05-06）：smaller_alpha 是主 winner，但 ep1 内有退化。
# alpha_decay 是 Phase 0 中**唯一 ep1 trajectory 不退化**的配置（peak @ step 58
# 仍在升），作为"保险"二号 winner 多 seed。如果 smaller_alpha multi-seed
# 不复现，可降级成 alpha_decay 主表。
#
# 配置 = Phase 0 M6 alpha_decay 的 ep1 跑过的相同参数：
#   default α (slpa 0.05/0.05 + scr 0.02/0.02) + alpha_decay=linear。
# Phase 0 seed=42 ep1 peak = 0.3751 @ step 58 (trajectory 单调向上)。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_alpha_decay_qwen3" \
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
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=9 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
