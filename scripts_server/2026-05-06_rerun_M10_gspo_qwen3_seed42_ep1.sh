#!/usr/bin/env bash
# Rerun / M10 — GSPO Qwen3 × seed=42 × ep1 native
#
# 输出 dir：gspo_math_qwen3_4b_ep1_rand42（自动生成，与已有 ep3 dir 不冲突）
set -e
cd "$HOME/sentencepo_v1-5"
SEED=42 EPOCHS=1 bash test_GSPO-metrics2.sh
