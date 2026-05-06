#!/usr/bin/env bash
# 2026-05-07 — KTAE baseline × Qwen3-4B-Base × seed=37 × ep1 native
# 输出 dir：ktae_math_qwen3_4b_ep1_rand37
set -e
cd "$HOME/sentencepo_v1-5"
SEED=37 EPOCHS=1 EXP_TAG=ktae bash test_KTAE.sh
