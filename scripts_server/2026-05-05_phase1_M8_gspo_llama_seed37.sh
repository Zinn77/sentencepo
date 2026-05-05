#!/usr/bin/env bash
# Phase 1 / M8 — GSPO baseline × Llama × seed=37 × ep3
# Llama baseline multi-seed 第二个 seed（已有 seed=42）。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=37 EPOCHS=3 bash test_GSPO-metrics2-llama.sh
