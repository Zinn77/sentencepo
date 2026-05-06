#!/usr/bin/env bash
# Phase 1.5 / M6 — smaller_alpha + LAMBDA_PPL=0.3（PPL 自适应 clip）
#
# 假说：sentencepo loss 内部 c_s = c0 * clamp(1 + λ_ppl·z_ppl - λ_len·z_len,
# cmin, cmax)。CLAUDE.md 已记录 default λ_ppl=0「自适应 clip 实际未生效」
# 即"自适应 clip 从来没真开过"。z_ppl 是句子级 PPL z-score，正值表示该句
# 比 batch 平均更难（PPL 高）；λ_ppl=0.3 时，难句 (z=+1) 的 clip 半径
# 放大 1.3x，期望让 hard sentence 更激进更新。
#
# 与 M7 (lambda_len) / M8 (双开) 配对，完整探索 sentencepo 自适应 clip 维度。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-06_phase1_5_M6_smaller_lambda_ppl" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  LAMBDA_PPL=0.3 \
  LAMBDA_LEN=0 \
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
