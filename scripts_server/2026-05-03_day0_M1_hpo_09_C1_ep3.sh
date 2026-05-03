#!/usr/bin/env bash
# Day 0 / Machine 1 — hpo_09_C1 ep3 复现（最关键单实验）
# (Phase A 已 0503 跑完，M1 释放出来跑 hpo)
#
# sentencepo loss + per_sent_adv on（Bug A 修复）+ SLPA 0.05/0.05 + SCR 0.02/0.02
# + no decay。layer = -1, pooling = last（与 0428 HPO 对齐）。
#
# 0428 HPO 跑出的 hpo_09_C1 在 ep1 = 0.374，是 v1-5 路线 ep1 历史最高。
# 本 run 唯一目标：验证它在 ep3 是否能守住领先。
#
# 日志：generic wrapper 自带 tee → $HOME/autodl-tmp/models_v1-5/<EXP_TAG>_<...>/verl_v1-5-hidden.log
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-03_day0_M1_hpo_09_C1_ep3" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=true \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=-1 \
  SLPA_POOL=last \
  SCR_ENABLE=true \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=-1 \
  SCR_POOL=last \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
