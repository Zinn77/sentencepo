#!/usr/bin/env bash
# 2026-05-07 / Qwen3 翻盘尝试 A — SLPA α 从 0.02 上调到 0.1（5x），保持 SCR 0.01
#
# 假说：winner_config (SLPA 0.02) 的 SLPA 信号过弱，导致 ep1 慢热。Phase 1.5 测过
# default α (0.05) + decay 是 0.354（弱），但**没测过 SLPA 单独 bump 到 0.1**。
# 单 seed=42 先看，过 0.38 才扩 multi-seed (s9, s37)。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
# Override: bump SLPA only
export SLPA_ALPHA_C=0.1
export SLPA_ALPHA_I=0.1

EXP_TAG="2026-05-07_qwen3_v15_slpa_bump01" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
