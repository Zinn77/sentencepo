#!/usr/bin/env bash
# Phase 2 a1 — Hidden layer ablation × L=-1 × winner_base × seed=42
# Layer ablation：default winner 用 L=-18（Phase A diagnostic gap maximum）。
# 本 run 切换到 L=-1（最后一层 / SCR-SNR maximum / 也是大多数 prior work 的 implicit 选择）。
# 论文 Sec 4.2 必须对比 L=-1 vs L=-18。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_LAYER=-1
export SCR_LAYER=-1

EXP_TAG="2026-05-07_phase2_a1_layer_neg1_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
