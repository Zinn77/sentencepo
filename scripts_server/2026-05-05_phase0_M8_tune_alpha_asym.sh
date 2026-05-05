#!/usr/bin/env bash
# Phase 0 / M8 — α 不对称（slpa correct=0.05/incorrect=0.02），ep1 调参
#
# 假说：错答 rollout 的 SLPA/SCR 信号容易把模型推向更长更乱的回复（因为
# 错答的 sentence-level advantage 有更高方差），导致 collapse。
# 让 incorrect 用更小的 α，对错答的 sentence-level 修正更保守。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase0_tune_alpha_asym" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.02 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.01 \
  SCR_LAYER=-18 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
