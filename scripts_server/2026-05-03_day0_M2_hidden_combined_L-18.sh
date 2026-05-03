#!/usr/bin/env bash
# Day 0 / Machine 2 — Hidden Phase B: combined SCR+SLPA at layer=-18 (Phase A top-1)
#
# 同 Day 0 M1 的 hpo_09_C1，仅把 SLPA / SCR 的 hidden_layer_index 从 -1 切到 -18。
# Pooling 都用 last（Phase A top-1：-18/last gap=0.1285，比 -1/last gap=0.0568 高 2.3×）。
#
# 目的：层消融的关键比较点。如果 ep3 显著高于 Day 0 M1 (-1)，
# 论文核心 finding：「sentence-level RL 用最后一层 hidden state 是次优选择」。
#
# 日志：generic wrapper 自带 tee → $HOME/autodl-tmp/models_v1-5/<EXP_TAG>_<...>/verl_v1-5-hidden.log
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-03_day0_M2_hidden_combined_L-18_ep3" \
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
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
