#!/usr/bin/env bash
# ============================================================================
# Generic 8-GPU RL run wrapper — used by all Day 0 / Day 1 / Day 2 launchers
# in the 2026-05-03 v1-5-hidden 3-day plan.
#
# Superset of test_v1-5_hidden_phaseB.sh: supports loss_mode switching,
# per_sentence_adv toggle, both SLPA AND SCR enabled simultaneously each with
# their own (alpha, layer, pooling), variable max_response_length.
#
# Drives one 8-GPU 3-epoch run (~12h) on Qwen3-4B-Base + math.
#
# === Required env (set by per-machine launcher) ===
#   EXP_TAG          : free-form tag (used in dir name + tensorboard)
#
# === Optional env (defaults match GRPO baseline) ===
#   LOSS_MODE        : vanilla | sentencepo               [vanilla]
#   PER_SENT_ADV     : true | false (Bug A fix)           [false]
#   SENTPO_EPS       : sentencepo loss eps_base           [0.03]
#   SLPA_ENABLE      : true | false                       [false]
#   SLPA_ALPHA_C     : SLPA alpha for correct rollouts    [0.05]
#   SLPA_ALPHA_I     : SLPA alpha for incorrect rollouts  [0.05]
#   SLPA_LAYER       : -1 / -18 / "-1|-9|-18"             [-1]
#   SLPA_POOL        : last|mean|first|...                [last]
#   SCR_ENABLE       : true | false                       [false]
#   SCR_ALPHA_C      : SCR alpha for correct              [0.02]
#   SCR_ALPHA_I      : SCR alpha for incorrect            [0.02]
#   SCR_LAYER        : same format as SLPA_LAYER          [-1]
#   SCR_POOL         : same format as SLPA_POOL           [last]
#   MAX_RESP_LEN     : 4096 (default) or 1024 (length cut)
#   EPOCHS           : trainer.total_epochs               [3]
#   SEED             : random seed                        [42]
#   ALPHA_DECAY      : none | linear                      [none]
#   KL_LOSS_COEF     : actor.kl_loss_coef                 [0.001]
#   ROLLOUT_TEMP     : rollout sampling temperature       [1.0]
#
# === Path overrides (machine-specific) ===
#   REPO_DIR         : repo root                          [$HOME/sentencepo_v1-5]
#   DATA_DIR         : data parquet root                  [$HOME/data]
#   OUTPUT_DIR       : runs root                          [$HOME/autodl-tmp/models_v1-5]
#   MODEL_PATH       : HF hub id or local path            [Qwen/Qwen3-4B-Base]
#   MODEL_NAME       : short name in dir                  [qwen3_4b]
# ============================================================================
set -ex

if [ -z "${EXP_TAG:-}" ]; then
    echo "ERROR: EXP_TAG env var is required" >&2
    exit 1
fi

# Required experiment knobs.
LOSS_MODE=${LOSS_MODE:-vanilla}
PER_SENT_ADV=${PER_SENT_ADV:-false}
SENTPO_EPS=${SENTPO_EPS:-0.03}
SLPA_ENABLE=${SLPA_ENABLE:-false}
SLPA_ALPHA_C=${SLPA_ALPHA_C:-0.05}
SLPA_ALPHA_I=${SLPA_ALPHA_I:-0.05}
SLPA_LAYER=${SLPA_LAYER:--1}
SLPA_POOL=${SLPA_POOL:-last}
SCR_ENABLE=${SCR_ENABLE:-false}
SCR_ALPHA_C=${SCR_ALPHA_C:-0.02}
SCR_ALPHA_I=${SCR_ALPHA_I:-0.02}
SCR_LAYER=${SCR_LAYER:--1}
SCR_POOL=${SCR_POOL:-last}
MAX_RESP_LEN=${MAX_RESP_LEN:-4096}
EPOCHS=${EPOCHS:-3}
SEED=${SEED:-42}
ALPHA_DECAY=${ALPHA_DECAY:-none}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
DS=${DS:-math}

# Determinism / cache.
export PYTHONHASHSEED=$SEED
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/autodl-tmp/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}
REPO_DIR=${REPO_DIR:-$HOME/sentencepo_v1-5}
DATA_DIR=${DATA_DIR:-$HOME/data}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/autodl-tmp/models_v1-5}

# Data files.
math_train_path=${DATA_DIR}/math/train.parquet
math500_test_path=${DATA_DIR}/math500/test.parquet
aime2024_test_path=${DATA_DIR}/aime2024/test.parquet
aime2025_test_path=${DATA_DIR}/aime2025/test.parquet
amc23_test_path=${DATA_DIR}/amc23/test.parquet
minerva_test_path=${DATA_DIR}/minerva/test.parquet
olympiad_train_path=${DATA_DIR}/olympiad/train.parquet
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$minerva_test_path', '$olympiad_train_path']"

# Static training hparams (from test_sentencepo_v1-5.sh).
lr=1e-6
qwen3_enable_thinking=False
max_num_batched_tokens=8192
micro_batch_size=4
clip_ratio_low=0.2
clip_ratio_high=0.2
sentencepo_min_sent_tokens=6
sentencepo_lambda_ppl=0
sentencepo_lambda_len=0
sentencepo_cmin=0.5
sentencepo_cmax=1.5
sentencepo_stats_eps=1e-6
sentencepo_metrics_level=full

# Hydra layer literal: int → "-9", list → "[-1,-9,-18]".
to_hydra_layer() {
    local raw="$1"
    if [[ "$raw" == *"|"* ]]; then
        echo "[$(echo "$raw" | tr '|' ',')]"
    else
        echo "$raw"
    fi
}
SLPA_LAYER_ARG=$(to_hydra_layer "$SLPA_LAYER")
SCR_LAYER_ARG=$(to_hydra_layer "$SCR_LAYER")

NEED_SENTENCEPO=false
if [ "$SLPA_ENABLE" = "true" ] || [ "$SCR_ENABLE" = "true" ]; then
    NEED_SENTENCEPO=true
fi

TOT_DIR="${OUTPUT_DIR}/${EXP_TAG}_${DS}_${MODEL_NAME}_ep${EPOCHS}_rand${SEED}"
mkdir -p "$TOT_DIR"
mkdir -p "$TOT_DIR/verl_checkpoints_v1-5"

export PYTHONPATH="${REPO_DIR}:$PYTHONPATH"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +data.seed=$SEED \
    +critic.data_loader_seed=$SEED \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESP_LEN \
    +data.apply_chat_template_kwargs.enable_thinking=$qwen3_enable_thinking \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.enable_sentencepo=$NEED_SENTENCEPO \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
    +actor_rollout_ref.actor.policy_loss.sentencepo_min_sent_tokens=$sentencepo_min_sent_tokens \
    +actor_rollout_ref.actor.policy_loss.sentencepo_eps_base=$SENTPO_EPS \
    +actor_rollout_ref.actor.policy_loss.sentencepo_lambda_ppl=$sentencepo_lambda_ppl \
    +actor_rollout_ref.actor.policy_loss.sentencepo_lambda_len=$sentencepo_lambda_len \
    +actor_rollout_ref.actor.policy_loss.sentencepo_cmin=$sentencepo_cmin \
    +actor_rollout_ref.actor.policy_loss.sentencepo_cmax=$sentencepo_cmax \
    +actor_rollout_ref.actor.policy_loss.sentencepo_stats_eps=$sentencepo_stats_eps \
    +actor_rollout_ref.actor.policy_loss.sentencepo_metrics_level=$sentencepo_metrics_level \
    +actor_rollout_ref.actor.policy_loss.sentencepo_per_sentence_adv=$PER_SENT_ADV \
    +algorithm.slpa.enable=$SLPA_ENABLE \
    +algorithm.slpa.alpha_correct=$SLPA_ALPHA_C \
    +algorithm.slpa.alpha_incorrect=$SLPA_ALPHA_I \
    +algorithm.slpa.tau_emb=0.1 \
    +algorithm.slpa.sigma_pos=1.0 \
    +algorithm.slpa.normalize=true \
    +algorithm.slpa.eps=1e-8 \
    +algorithm.slpa.correctness_threshold=0.0 \
    +algorithm.slpa.metrics_enable=true \
    +algorithm.slpa.alpha_decay=$ALPHA_DECAY \
    +algorithm.slpa.alpha_min_ratio=0.1 \
    +algorithm.slpa.repr.hidden_layer_index=$SLPA_LAYER_ARG \
    +algorithm.slpa.repr.pooling=$SLPA_POOL \
    +algorithm.scr.enable=$SCR_ENABLE \
    +algorithm.scr.alpha_correct=$SCR_ALPHA_C \
    +algorithm.scr.alpha_incorrect=$SCR_ALPHA_I \
    +algorithm.scr.tau_reward=1.0 \
    +algorithm.scr.tau_sim=0.1 \
    +algorithm.scr.normalize=true \
    +algorithm.scr.eps=1e-8 \
    +algorithm.scr.correctness_threshold=0.0 \
    +algorithm.scr.metrics_enable=true \
    +algorithm.scr.alpha_decay=$ALPHA_DECAY \
    +algorithm.scr.alpha_min_ratio=0.1 \
    +algorithm.scr.repr.hidden_layer_index=$SCR_LAYER_ARG \
    +algorithm.scr.repr.pooling=$SCR_POOL \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
    critic.checkpoint.save_contents='["model"]' \
    critic.checkpoint.load_contents='["model"]' \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="verl_${MODEL_NAME}_${DS}_v1-5-hidden" \
    trainer.experiment_name="${EXP_TAG}_seed${SEED}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir=$TOT_DIR/verl_checkpoints_v1-5 \
    trainer.use_legacy_worker_impl=disable \
    "$@" 2>&1 | tee $TOT_DIR/verl_v1-5-hidden.log
