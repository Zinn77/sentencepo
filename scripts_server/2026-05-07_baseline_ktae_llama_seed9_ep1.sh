#!/usr/bin/env bash
# 2026-05-07 — KTAE baseline × Llama-3.2-3B-Instruct × seed=9 × ep1 native
set -e
cd "$HOME/sentencepo_v1-5"
SEED=9 EPOCHS=1 EXP_TAG=ktae \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  KTAE_PAD_TOKEN_ID=128009 \
  bash test_KTAE.sh
