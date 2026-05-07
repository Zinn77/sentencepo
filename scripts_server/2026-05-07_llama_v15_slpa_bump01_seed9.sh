#!/usr/bin/env bash
# 2026-05-07 / Llama slpa-bump s9 — preemptive parallel with s42
# 与 s42 同 config (SLPA 0.02→0.1)，同时跑节省 5h（若 s42 赢 winner，立刻有 n=2 多 seed）。
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_ALPHA_C=0.1
export SLPA_ALPHA_I=0.1

EXP_TAG="2026-05-07_llama_v15_slpa_bump01" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=9 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
