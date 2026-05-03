#!/usr/bin/env bash
# Day 1 / Machine 3 — Hidden Phase B: pure SCR at layer=-18 (no SLPA)
#
# 模块拆解消融：combined（Day 0 M4）的增益来自 SCR 还是 SLPA？
# 选 SCR-only 而不是 SLPA-only 的依据：Phase A snr，
# SCR 在 -18/last 的 snr=0.319 远高于 SLPA 的 0.043（7.4×），
# 假设 SCR 是 -18 红利的主要受益者。
#
# 如果本 run ep3 ≈ Day 0 M4 → SCR-only 已足够，paper 推荐 single-channel 配置
# 如果本 run ep3 ≪ Day 0 M4 → 需要 combined（SLPA 也贡献）
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-04_day1_M3_hidden_scr_only_L-18_ep3" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=false \
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
