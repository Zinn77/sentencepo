# Shared Phase 1 winner_config bundle.
#
# 2026-05-06 决策（advisor_report3 §12.2 的修订版）：
#   主表 = ep1 peak ± std multi-seed。
#   winner = P0_smaller_alpha（Phase 0 实测 ep1 peak 0.3795 = 7 个 v1-5 配置最高，
#   首次明确赢 GSPO multi-seed avg ep1 peak 0.3697 = +1.0pp）。
#
#   alpha_decay 没合并进 winner（虽然 ep1 trajectory 不退化），原因：
#     1. ep1 peak (0.3751) 低于 smaller_alpha (0.3795)
#     2. 既然主结果用 ep1 peak，alpha_decay 的"不退化"在 ep1 内部分体现
#     3. alpha_decay 留作 Phase 2 ablation（multi-seed 单测）
#
# Sourced by Phase 1 winner launchers (M1, M2, M6_llama_seed42, M7_llama_seed9).

export LOSS_MODE=sentencepo
export PER_SENT_ADV=true
export SENTPO_EPS=0.03

export SLPA_ENABLE=true
export SLPA_ALPHA_C=0.02      # ← smaller_alpha（原 default 0.05）
export SLPA_ALPHA_I=0.02
export SLPA_LAYER=-18
export SLPA_POOL=last

export SCR_ENABLE=true
export SCR_ALPHA_C=0.01       # ← smaller_alpha（原 default 0.02）
export SCR_ALPHA_I=0.01
export SCR_LAYER=-18
export SCR_POOL=last

export ALPHA_DECAY=none
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
