#!/usr/bin/env bash
# Phase 0 / M9 — v1-5 default config 在 Llama-3.2-3B-Instruct 上跑 ep3
#
# 🔥 论文最强卖点：GRPO Llama 因 length explosion 崩到 0（peak 0.20 → ep3 0.0），
# GSPO Llama 稳定但 peak 仅 0.177。如果 v1-5 在 Llama 上 peak ≥ GSPO 且不崩，
# 就是 sentence-level RL 解决 Llama length collapse 的故事。
#
# Llama 配置 = Qwen3 default 完全复用，唯一区别是 MODEL_PATH/MODEL_NAME。
# 这是公平对比（baseline GSPO Llama 也用 Qwen3 配置）。
#
# 时长 ~14h（Llama 3B 比 Qwen3 4B 略快），跨阶段一直跑到 Phase 2 早期。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_phase0_v15_default_llama" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
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
