#!/usr/bin/env bash
# 2026-05-07 / Qwen3 翻盘 C — winner_config + SLPA top-K=8 × seed=42 ep1
#
# 假说（zzx_slpa.md §3.1.3）：当前 SLPA V_k 用所有 leave-one-out 句子做核加权平均，
# 远距离低相似度句子带来的噪声稀释了 SLPA 信号。改成只取 top-K=8 个最相似句子能
# 抑制噪声 → ep1 peak 应过 0.38（GSPO baseline 0.3678 ± 0.006）。
#
# 在 worktree 分支 sentencepo_v1-5-hidden-topk 上跑，与主仓 ktae 并行，互不影响。
set -e
cd "$HOME/sentencepo_v1-5-topk"

source scripts_server/2026-05-05_phase1_winner_config.sh
export SLPA_TOP_K=8

EXP_TAG="2026-05-07_qwen3_topk8_winner" \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=42 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
