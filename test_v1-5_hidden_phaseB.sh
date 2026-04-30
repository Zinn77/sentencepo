#!/usr/bin/env bash
# Phase B parametric script for v1-5-hidden ablation.
#
#   MODULE   = scr | slpa | none      (which sentence-adv module to enable)
#   LAYER    = int (e.g. -1, -9, -18) or "-1|-9|-18" for ensemble mean
#   POOLING  = last | mean | first | mean_no_punct | entropy_weighted | diff
#   ALPHA    = fusion weight (default 0.05)
#
# Example:
#   MODULE=scr LAYER=-9 POOLING=mean ALPHA=0.05 SEED=42 EPOCHS=1 \
#     bash test_v1-5_hidden_phaseB.sh
set -ex

MODULE=${MODULE:-scr}
LAYER=${LAYER:--1}
POOLING=${POOLING:-last}
ALPHA=${ALPHA:-0.05}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-1}
DS=${DS:-math}
EXP_TAG_OVERRIDE=${EXP_TAG_OVERRIDE:-}

export PYTHONHASHSEED=$SEED
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOT_PATH=${TOT_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02}
MODEL_PATH=${MODEL_PATH:-${TOT_PATH}/huggingface.co/Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}

math_train_path=${TOT_PATH}/data/math/train.parquet
math500_test_path=${TOT_PATH}/data/math500/test.parquet
aime2024_test_path=${TOT_PATH}/data/aime2024/test.parquet
aime2025_test_path=${TOT_PATH}/data/aime2025/test.parquet
amc23_test_path=${TOT_PATH}/data/amc23/test.parquet
minerva_test_path=${TOT_PATH}/data/minerva/test.parquet
olympiad_train_path=${TOT_PATH}/data/olympiad/train.parquet
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$minerva_test_path', '$olympiad_train_path']"

lr=1e-6
qwen3_enable_thinking=False
max_response_length=4096
max_num_batched_tokens=8192
micro_batch_size=4

LOSS_MODE=vanilla
clip_ratio_low=0.2
clip_ratio_high=0.2

# === Module enable flags ===
if [ "$MODULE" = "scr" ]; then
    scr_enable=true
    slpa_enable=false
elif [ "$MODULE" = "slpa" ]; then
    scr_enable=false
    slpa_enable=true
else
    scr_enable=false
    slpa_enable=false
fi

NEED_SENTENCEPO=false
if [ "$slpa_enable" = "true" ] || [ "$scr_enable" = "true" ]; then
    NEED_SENTENCEPO=true
fi

# === Convert layer spec ===
# Hydra accepts a list literal like [-1,-9,-18] for list-typed fields,
# and an int like -9 for int-typed fields. We always pass a Python literal
# string; the SentenceReprConfig.hidden_layer_index field is `Any`.
if [[ "$LAYER" == *"|"* ]]; then
    LAYER_ARG="[$(echo "$LAYER" | tr '|' ',')]"
else
    LAYER_ARG="$LAYER"
fi

# === Experiment tag ===
LAYER_TAG=$(echo "$LAYER" | tr '|' '_' | tr '-' 'm')
EXP_TAG=${EXP_TAG_OVERRIDE:-"${MODULE}_L${LAYER_TAG}_P${POOLING}_a${ALPHA}"}
TOT_DIR="${TOT_PATH}/models_v1-5/${EXP_TAG}_${DS}_${MODEL_NAME}_ep${EPOCHS}_rand${SEED}"
mkdir -p $TOT_DIR
mkdir -p $TOT_DIR/verl_checkpoints_v1-5

# === Standard sentencepo loss params (loss=vanilla so most are inert) ===
sentencepo_min_sent_tokens=6
sentencepo_eps_base=0.01
sentencepo_lambda_ppl=0
sentencepo_lambda_len=0
sentencepo_cmin=0.5
sentencepo_cmax=1.5
sentencepo_stats_eps=1e-6
sentencepo_metrics_level=full
sentencepo_per_sentence_adv=false

# === SLPA defaults ===
slpa_alpha_correct=$ALPHA
slpa_alpha_incorrect=$ALPHA
slpa_tau_emb=0.1
slpa_sigma_pos=1.0
slpa_normalize=true
slpa_eps=1e-8
slpa_correctness_threshold=0.0
slpa_metrics_enable=true
slpa_alpha_decay=none
slpa_alpha_min_ratio=0.1

# === SCR defaults ===
scr_alpha_correct=$ALPHA
scr_alpha_incorrect=$ALPHA
scr_tau_reward=1.0
scr_tau_sim=0.1
scr_normalize=true
scr_eps=1e-8
scr_correctness_threshold=0.0
scr_metrics_enable=true
scr_alpha_decay=none
scr_alpha_min_ratio=0.1

export PYTHONPATH="${TOT_PATH}/sentencepo_v1-5:$PYTHONPATH"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +data.seed=$SEED \
    +critic.data_loader_seed=$SEED \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
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
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
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
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
    +actor_rollout_ref.actor.policy_loss.sentencepo_min_sent_tokens=$sentencepo_min_sent_tokens \
    +actor_rollout_ref.actor.policy_loss.sentencepo_eps_base=$sentencepo_eps_base \
    +actor_rollout_ref.actor.policy_loss.sentencepo_lambda_ppl=$sentencepo_lambda_ppl \
    +actor_rollout_ref.actor.policy_loss.sentencepo_lambda_len=$sentencepo_lambda_len \
    +actor_rollout_ref.actor.policy_loss.sentencepo_cmin=$sentencepo_cmin \
    +actor_rollout_ref.actor.policy_loss.sentencepo_cmax=$sentencepo_cmax \
    +actor_rollout_ref.actor.policy_loss.sentencepo_stats_eps=$sentencepo_stats_eps \
    +actor_rollout_ref.actor.policy_loss.sentencepo_metrics_level=$sentencepo_metrics_level \
    +actor_rollout_ref.actor.policy_loss.sentencepo_per_sentence_adv=$sentencepo_per_sentence_adv \
    +algorithm.slpa.enable=$slpa_enable \
    +algorithm.slpa.alpha_correct=$slpa_alpha_correct \
    +algorithm.slpa.alpha_incorrect=$slpa_alpha_incorrect \
    +algorithm.slpa.tau_emb=$slpa_tau_emb \
    +algorithm.slpa.sigma_pos=$slpa_sigma_pos \
    +algorithm.slpa.normalize=$slpa_normalize \
    +algorithm.slpa.eps=$slpa_eps \
    +algorithm.slpa.correctness_threshold=$slpa_correctness_threshold \
    +algorithm.slpa.metrics_enable=$slpa_metrics_enable \
    +algorithm.slpa.alpha_decay=$slpa_alpha_decay \
    +algorithm.slpa.alpha_min_ratio=$slpa_alpha_min_ratio \
    +algorithm.slpa.repr.hidden_layer_index=$LAYER_ARG \
    +algorithm.slpa.repr.pooling=$POOLING \
    +algorithm.scr.enable=$scr_enable \
    +algorithm.scr.alpha_correct=$scr_alpha_correct \
    +algorithm.scr.alpha_incorrect=$scr_alpha_incorrect \
    +algorithm.scr.tau_reward=$scr_tau_reward \
    +algorithm.scr.tau_sim=$scr_tau_sim \
    +algorithm.scr.normalize=$scr_normalize \
    +algorithm.scr.eps=$scr_eps \
    +algorithm.scr.correctness_threshold=$scr_correctness_threshold \
    +algorithm.scr.metrics_enable=$scr_metrics_enable \
    +algorithm.scr.alpha_decay=$scr_alpha_decay \
    +algorithm.scr.alpha_min_ratio=$scr_alpha_min_ratio \
    +algorithm.scr.repr.hidden_layer_index=$LAYER_ARG \
    +algorithm.scr.repr.pooling=$POOLING \
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
