#!/usr/bin/env bash
# 2026-05-07 — KTAE baseline × Qwen3-4B-Base × seed=9 × ep1 native
# Multi-seed 第二个 seed（已有 seed=42 from 0506_baseline_ktae_qwen3_seed42_ep1.sh）。
# 输出 dir：ktae_math_qwen3_4b_ep1_rand9
set -e
cd "$HOME/sentencepo_v1-5"
SEED=9 EPOCHS=1 EXP_TAG=ktae bash test_KTAE.sh
