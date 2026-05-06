#!/usr/bin/env bash
# 2026-05-07 — KTAE baseline × Llama-3.2-3B-Instruct × seed=42 × ep1 native
#
# Llama EOS=128009，必须 override KTAE_PAD_TOKEN_ID（否则 KTAE mask 错位）。
# 输出 dir：ktae_math_llama3.2-3b_ep1_rand42
set -e
cd "$HOME/sentencepo_v1-5"
SEED=42 EPOCHS=1 EXP_TAG=ktae \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  KTAE_PAD_TOKEN_ID=128009 \
  bash test_KTAE.sh
