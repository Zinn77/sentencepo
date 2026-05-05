# Shared Phase 1 winner_config bundle.
#
# Sourced by all Phase 1 winner launchers. After Phase 0 (T+6h) finishes the 4
# tuning runs, edit this file ONCE to the actual winner config, then all Phase
# 1 winner launchers (M1, M2, M7) automatically pick up the new values.
#
# Default = current best (M2 v1-5 default: combined L=-18, slpa 0.05/0.05,
# scr 0.02/0.02, no decay, sentencepo loss, per_sent_adv on).

export LOSS_MODE=sentencepo
export PER_SENT_ADV=true
export SENTPO_EPS=0.03

export SLPA_ENABLE=true
export SLPA_ALPHA_C=0.05
export SLPA_ALPHA_I=0.05
export SLPA_LAYER=-18
export SLPA_POOL=last

export SCR_ENABLE=true
export SCR_ALPHA_C=0.02
export SCR_ALPHA_I=0.02
export SCR_LAYER=-18
export SCR_POOL=last

export ALPHA_DECAY=none
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
