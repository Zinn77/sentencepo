#!/usr/bin/env bash
# Phase 2 / M8 — DAPO baseline × Qwen3 × seed=9 × ep3 (跨阶段)
# DAPO multi-seed 第二个 seed（已有 phase1_M6 = seed=42）。
# 完整跑 ep3 = ~13h，跨 Phase 2 一直跑到 Phase 3 早期。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=9 EPOCHS=3 bash scripts_server/2026-05-05_phase1_M6_dapo_qwen3_seed42.sh
