#!/usr/bin/env bash
# 2026-05-07 / Qwen3 翻盘尝试 B — SCR α 从 0.01 上调到 0.1（10x），保持 SLPA 0.02
#
# 假说：winner_config 的 SCR 信号偏小，对比效果未充分发挥。
# 单 seed=42 先看，过 0.38 才扩 multi-seed。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
# Override: bump SCR only
export SCR_ALPHA_C=0.1
export SCR_ALPHA_I=0.1

EXP_TAG="2026-05-07_qwen3_v15_scr_bump01" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
