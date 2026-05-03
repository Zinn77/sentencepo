#!/usr/bin/env bash
# Day 2 / Machine 3 — winner seed=43，方差估计 winner
#
# Day 0/1 跑完后，把 ep3 mean pass@1 最高的那个 hidden 配置选作 winner。
# 默认假设 winner 是 Day 0 M4（combined L=-18 P=last）。
# 如果实际 winner 是 Day 1 M1 (L=-9) / M2 (L=-27) / M3 (SCR-only L=-18)，
# launch 前覆盖 env：
#
#   # 例：winner 实际是 Day 1 M1 (L=-9 combined)
#   SLPA_LAYER=-9 SCR_LAYER=-9 \
#     bash scripts_server/2026-05-05_day2_M3_winner_seed43.sh
#
#   # 例：winner 是 Day 1 M3 (SCR-only L=-18)
#   SLPA_ENABLE=false \
#     bash scripts_server/2026-05-05_day2_M3_winner_seed43.sh
set -e
cd "$HOME/sentencepo_v1-5"

EXP_TAG="2026-05-05_day2_M3_winner_seed43" \
  LOSS_MODE=sentencepo \
  PER_SENT_ADV=true \
  SENTPO_EPS=0.03 \
  SLPA_ENABLE=${SLPA_ENABLE:-true} \
  SLPA_ALPHA_C=0.05 \
  SLPA_ALPHA_I=0.05 \
  SLPA_LAYER=${SLPA_LAYER:--18} \
  SLPA_POOL=${SLPA_POOL:-last} \
  SCR_ENABLE=${SCR_ENABLE:-true} \
  SCR_ALPHA_C=0.02 \
  SCR_ALPHA_I=0.02 \
  SCR_LAYER=${SCR_LAYER:--18} \
  SCR_POOL=${SCR_POOL:-last} \
  ALPHA_DECAY=none \
  MAX_RESP_LEN=4096 \
  EPOCHS=3 \
  SEED=43 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
