#!/usr/bin/env bash
# Rerun / M13 — GRPO Llama × seed=42 × ep1 native
#
# 注意：原 GRPO Llama ep3 在 step 55 已 length collapse 到 0.090（peak 0.20 @ step 30）；
# ep1 native 重跑不一定能避免 collapse，但论文 cross-model claim 需要 step 58 数据点
# 跟 v1-5 / GSPO 同步对齐。
#
# 输出 dir：grpo_math_llama3.2-3b_ep1_rand42
set -e
cd "$HOME/sentencepo_v1-5"
SEED=42 EPOCHS=1 \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  bash test_GRPO-metrics2.sh
