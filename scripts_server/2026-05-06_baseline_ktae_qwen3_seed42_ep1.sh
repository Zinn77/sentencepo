#!/usr/bin/env bash
# 2026-05-06 — KTAE baseline × Qwen3-4B-Base × seed=42 × ep1 native
#
# 加 KTAE 作为论文第三 baseline（GRPO / GSPO / KTAE / SentencePO 四方对比）。
# KTAE 是 token-level non-uniform credit（Fisher exact + IG + Cohen's h），
# 移植自 https://github.com/ZNLP/KTAE （改成 verl-native 注册）。
#
# 与 GRPO/GSPO/v1-5 的 ep1 native run 完全同一 stack（同 rollout / 同 verifier /
# 同 6 数据集 eval / 同 lr / 同 train_batch_size=128 / 同 n=8 / 同 max_resp=4096），
# 只换 algorithm.adv_estimator=ktae + 5 个 KTAE 超参。
#
# 输出 dir：ktae_math_qwen3_4b_ep1_rand42
set -e
cd "$HOME/sentencepo_v1-5"
SEED=42 EPOCHS=1 EXP_TAG=ktae bash test_KTAE.sh
