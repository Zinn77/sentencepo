#!/usr/bin/env bash
# Day 1 / Machine 4 — Bug A 消融：hpo_09_C1 with per_sent_adv=false
#
# 同 Day 0 M3 hpo_09_C1，唯一区别 sentencepo_per_sentence_adv=false。
# 论文必备消融：证明 Bug A 修复（per_sent_adv=true）在 ep3 仍是必要条件。
# 0428 数据：per_sent on=0.340 vs off=0.263 (差 7.7pp)。
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-04_day1_M4_hpo_09_C1_persentOFF_ep3" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=false \
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
