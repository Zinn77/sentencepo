#!/usr/bin/env bash
# Phase 1 / M6 — DAPO baseline × Qwen3 × seed=42 × ep3
#
# DAPO 在 verl 里以 loss_mode=geo_mean 实现（见 verl/trainer/ppo/core_algos.py
# L2169，对齐 examples/gmpo_trainer/test_dapo_*.sh）。简化版：不启用 overlong
# buffer（需要 reward_kwargs 改造，单独再做）。核心 DAPO 元素：
#   - geo_mean loss (sequence-level geometric mean of token-level ratios)
#   - clip_ratio_low=0.2, clip_ratio_high=0.28 (clip-higher，比标准 PPO 0.2/0.2 不对称)
#   - loss_agg_mode=token-mean
#   - use_kl_loss=False（DAPO 不用 KL）
# 注意：完整 DAPO 还有 filter_groups + overlong shaping，本脚本是 minimal 版。
set -ex

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-3}
DS=${DS:-math}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}

export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

math_train_path=$HOME/data/math/train.parquet
math500_test_path=$HOME/data/math500/test.parquet
aime2024_test_path=$HOME/data/aime2024/test.parquet
aime2025_test_path=$HOME/data/aime2025/test.parquet
amc23_test_path=$HOME/data/amc23/test.parquet
minerva_test_path=$HOME/data/minerva/test.parquet
olympiad_train_path=$HOME/data/olympiad/train.parquet
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$minerva_test_path', '$olympiad_train_path']"

lr=1e-6
qwen3_enable_thinking=False
max_response_length=4096
max_num_batched_tokens=8192
micro_batch_size=4
clip_ratio_low=0.2
clip_ratio_high=0.28

TOT_DIR=$HOME/autodl-tmp/models_v1-5/dapo_${DS}_${MODEL_NAME}_ep${EPOCHS}_rand${SEED}
mkdir -p $TOT_DIR
mkdir -p $TOT_DIR/verl_checkpoints_dapo

cd $HOME/sentencepo_v1-5

PYTHONPATH=$HOME/sentencepo_v1-5 \
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
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=geo_mean \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
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
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
    critic.checkpoint.save_contents='["model"]' \
    critic.checkpoint.load_contents='["model"]' \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="verl_${MODEL_NAME}_${DS}_dapo" \
    trainer.experiment_name="dapo_ep${EPOCHS}_seed${SEED}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir=$TOT_DIR/verl_checkpoints_dapo \
    trainer.use_legacy_worker_impl=disable \
    "$@" 2>&1 | tee $TOT_DIR/verl_dapo.log
