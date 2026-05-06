#!/usr/bin/env bash
# Phase 2 d — Sentence-level loss only ablation × winner_base × seed=42
# 关闭 SLPA + SCR，只保留 sentencepo 句子级 clip loss + GRPO 序列级 advantage。
# 论文 paper.tex Sec 4.1 的 SentencePO-clip 行（loss 单独的贡献）。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_ENABLE=false
export SCR_ENABLE=false

EXP_TAG="2026-05-07_phase2_d_sentencepo_only_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
