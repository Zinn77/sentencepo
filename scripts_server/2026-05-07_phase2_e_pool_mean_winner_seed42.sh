#!/usr/bin/env bash
# Phase 2 e (low priority) — Pooling ablation × pool=mean × L=-18 × winner_base × seed=42
# Phase A diagnostic 显示 L=-18 上 last 与 mean gap 差 0.018（很小），
# RL 实测验证 last 是否仍是最佳。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_POOL=mean
export SCR_POOL=mean

EXP_TAG="2026-05-07_phase2_e_pool_mean_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
