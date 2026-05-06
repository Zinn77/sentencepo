#!/usr/bin/env bash
# Rerun / M9 — GRPO Qwen3 × seed=42 × ep1 native（替代 ep3 截 step 55 的偏置数据）
#
# 原因：ep3 truncated 数据没有 step 58 val（trainer 末尾强制 val 仅在 ep1 native 触发），
# 导致 baseline ep1_end 只能用 step 55 来比；而 v1-5 ep1 native 用 step 58 引入
# 3-step 训练偏置（在 P0_alpha_decay seed=42 上看到 step 55→58 +5.6pp 漂移）。
# 本次 fresh ep1 native 让 baseline 也有 step 58 val，跟 v1-5 完全 apples-to-apples。
#
# 输出 dir：grpo_math_qwen3_4b_ep1_rand42（自动生成，与已有 ep3 dir 不冲突）
set -e
cd "$HOME/sentencepo_v1-5"
SEED=42 EPOCHS=1 bash test_GRPO-metrics2.sh
