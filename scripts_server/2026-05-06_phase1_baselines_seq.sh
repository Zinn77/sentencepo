#!/usr/bin/env bash
# Phase 1 - 3 baseline runs sequential on a single 8-GPU machine.
#
# 用户决策（2026-05-06）：把 GRPO seed=9 + GRPO seed=37 + GSPO seed=9 放一台
# 机器串行跑，腾出 2 台机器给 alpha_decay multi-seed（次优配置保险）。
#
# 串行顺序选择：
#   1. GRPO seed=9   ~2.5h  ← 先短的，早出数据
#   2. GRPO seed=37  ~2.5h
#   3. GSPO seed=9   ~4h    ← 最长放最后
#   总计 ~9h
#
# 任一步失败不阻断后续（每步都 set +e 包起来）。
set -e
cd "$HOME/sentencepo_v1-5"

LOG_DIR="$HOME/autodl-tmp/models_v1-5"
SUMMARY_LOG="$LOG_DIR/2026-05-06_phase1_baselines_seq_summary.log"
mkdir -p "$LOG_DIR"

echo "===== Phase 1 baselines sequential start: $(date) =====" | tee "$SUMMARY_LOG"

# ----- 1. GRPO Qwen3 seed=9 -----
echo "[$(date)] [1/3] Launching GRPO seed=9..." | tee -a "$SUMMARY_LOG"
set +e
bash scripts_server/2026-05-05_phase1_M3_grpo_qwen3_seed9.sh 2>&1 | tail -50 >> "$SUMMARY_LOG"
echo "[$(date)] [1/3] GRPO seed=9 done (rc=$?)" | tee -a "$SUMMARY_LOG"
set -e

# ----- 2. GRPO Qwen3 seed=37 -----
echo "[$(date)] [2/3] Launching GRPO seed=37..." | tee -a "$SUMMARY_LOG"
set +e
bash scripts_server/2026-05-05_phase1_M4_grpo_qwen3_seed37.sh 2>&1 | tail -50 >> "$SUMMARY_LOG"
echo "[$(date)] [2/3] GRPO seed=37 done (rc=$?)" | tee -a "$SUMMARY_LOG"
set -e

# ----- 3. GSPO Qwen3 seed=9 -----
echo "[$(date)] [3/3] Launching GSPO seed=9..." | tee -a "$SUMMARY_LOG"
set +e
bash scripts_server/2026-05-05_phase1_M5_gspo_qwen3_seed9.sh 2>&1 | tail -50 >> "$SUMMARY_LOG"
echo "[$(date)] [3/3] GSPO seed=9 done (rc=$?)" | tee -a "$SUMMARY_LOG"
set -e

echo "===== Phase 1 baselines sequential end: $(date) =====" | tee -a "$SUMMARY_LOG"
