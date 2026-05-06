#!/usr/bin/env bash
# Phase 1.5 / M7 — smaller_alpha + LAMBDA_LEN=0.3（length 自适应 clip）
#
# 假说：c_s 公式中 -λ_len·z_len 项，长句 (z=+1) 的 clip 半径收紧 0.7x。
# Bug D（response length 暴增）是 ep3 退化主因；ep1 阶段 length 已开始
# 漂移（CLAUDE.md 511-1024+ 桶占比上升），lambda_len=0.3 让长句 ratio 更
# 难偏离 1，相当于"长句梯度更保守"，期望抑制 length explosion。
#
# 与 M6 (lambda_ppl) / M8 (双开) 配对。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M7_smaller_lambda_len" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  LAMBDA_PPL=0 \
  LAMBDA_LEN=0.3 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.02 \
  SLPA_ALPHA_I=0.02 \
  SLPA_LAYER=-18 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.01 \
  SCR_ALPHA_I=0.01 \
  SCR_LAYER=-18 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  KL_LOSS_COEF=0.001 \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
