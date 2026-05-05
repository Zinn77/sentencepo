#!/usr/bin/env bash
# Phase 0 / M7 — default + kl_loss_coef=0.005（5×），ep1 调参
#
# 假说：v1-5 ep3 退化的根本原因是 length bloat（CLAUDE.md Bug D），
# kl_loss_coef 大可以抑制 actor drift 远离 ref，从而抑制 length explosion。
# 直接打根因，可能比 alpha 调参更有效。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase0_tune_kl_strong" \
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
  ALPHA_DECAY=none \
  KL_LOSS_COEF=0.005 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
