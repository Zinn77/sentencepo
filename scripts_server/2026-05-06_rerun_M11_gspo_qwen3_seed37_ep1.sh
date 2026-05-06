#!/usr/bin/env bash
# Rerun / M11 — GSPO Qwen3 × seed=37 × ep1 native
#
# 输出 dir：gspo_math_qwen3_4b_ep1_rand37
set -e
cd "$HOME/sentencepo_v1-5"
SEED=37 EPOCHS=1 bash test_GSPO-metrics2.sh
