#!/usr/bin/env bash
# Phase 2 / M7 — sentencepo loss + 全关 SLPA/SCR × seed=42 × ep2 (12h)
# 句子优势消融：sentencepo loss 自己（句子级 clip）是否独立有用，
# 还是必须配合 SLPA/SCR sentence-level advantage 才有效？
# 这是论文 Method 章节 ablation 的关键对比。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase2_sentencepo_only" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=false \
  SCR_ENABLE=false \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=2 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
