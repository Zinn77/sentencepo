#!/usr/bin/env bash
# Phase 1.5 / M1 — smaller_alpha + alpha_decay 组合（首选 candidate）
#
# 假说：smaller_alpha (P0 ep1 peak 0.3795) 和 alpha_decay (P0 ep1_end 0.3751)
# 是 Phase 0 唯二跑赢 baseline 的配置；plan §12.2 提议过组合，但 §13.1 决策
# 时为了"主表 peak"放弃了组合方案。Phase 1 multi-seed 暴露了 single-seed
# winner 不复现的问题，现在用 ep1 同时看 peak 和 ep1_end 两个指标。
#
# 期望：peak ≥ 0.3795（≥ smaller_alpha），ep1_end ≥ 0.3751（≥ alpha_decay）
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M1_smaller_decay" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
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
  ALPHA_DECAY=linear \
  ALPHA_MIN_RATIO=0.1 \
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
