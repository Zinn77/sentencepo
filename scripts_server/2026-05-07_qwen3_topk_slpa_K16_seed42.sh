#!/usr/bin/env bash
# 2026-05-07 / Qwen3 翻盘 C — winner_config + SLPA top-K=16 × seed=42 ep1
#
# 假说：K=16 是更稳健的过滤强度（每个其他 rollout 平均取 ~2 句），如果 K=8 太激进
# 丢失重要远距离正样本，K=16 可能更优。
set -e
cd "$HOME/sentencepo_v1-5-topk"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_TOP_K=16

EXP_TAG="2026-05-07_qwen3_topk16_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
