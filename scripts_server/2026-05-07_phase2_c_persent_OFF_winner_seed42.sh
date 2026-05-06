#!/usr/bin/env bash
# Phase 2 c — Bug A ablation × winner_base × seed=42
# 关闭 sentencepo loss 内部的 per-sentence advantage（恢复 Bug A bug）。
# 0428 测过 ep3 OFF=0.263 vs ON=0.340，相差 7.7pp。本 run 在 winner config 上
# 重测 ep1，作为论文 Sec 4 的"我们发现并修了一个隐藏 bug"消融的核心数据。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export PER_SENT_ADV=false

EXP_TAG="2026-05-07_phase2_c_persent_OFF_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
