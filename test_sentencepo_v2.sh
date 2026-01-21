#!/usr/bin/env bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
# Helps reduce CUDA fragmentation on large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 数据集设置
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet
math500_test_path=$HOME/data/math500/test.parquet
aime2024_test_path=$HOME/data/aime2024/test.parquet
aime2025_test_path=$HOME/data/aime2025/test.parquet
amc23_test_path=$HOME/data/amc23/test.parquet
minerva_test_path=$HOME/data/minerva/test.parquet
olympiad_train_path=$HOME/data/olympiad/train.parquet
train_files="['$math_train_path']"
# test_files="['$math_test_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$minerva_test_path', '$olympiad_train_path']"

# 训练设置
DS=${DS:-math}
EPOCHS=${EPOCHS:-5}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}
lr=1e-6
qwen3_enable_thinking=False # 关闭思考模式
max_response_length=4096
# max_num_batched_tokens >= max_prompt_length + max_response_length，或开启 enable_chunked_prefill
max_num_batched_tokens=8192    # 默认 8192
micro_batch_size=4
clip_ratio_low=0.01
clip_ratio_high=0.02
sentence_adv_pooling=${SENTENCE_ADV_POOLING:-mean}
sentence_adv_alpha=${SENTENCE_ADV_ALPHA:-0.1}
sentence_adv_temperature=${SENTENCE_ADV_TEMPERATURE:-0.2}
adv_estimator=${ADV_ESTIMATOR:-grpo_sentencepo}
sentence_adv_metrics=True
sentence_adv_metrics_max_sentences=${SENTENCE_ADV_METRICS_MAX_SENTENCES:-128}
sentence_adv_metrics_max_pairs=${SENTENCE_ADV_METRICS_MAX_PAIRS:-4096}
sentence_adv_metrics_pos_bins=${SENTENCE_ADV_METRICS_POS_BINS:-4}
sentence_adv_metrics_divergence_threshold=${SENTENCE_ADV_METRICS_DIVERGENCE_THRESHOLD:-0.1}

mkdir -p $HOME/autodl-tmp/models/sentencepo_${DS}_${MODEL_NAME}_ep${EPOCHS}

cd $HOME/sentencepo

PYTHONPATH=$HOME/sentencepo \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.sentence_adv.pooling=$sentence_adv_pooling \
    algorithm.sentence_adv.alpha=$sentence_adv_alpha \
    algorithm.sentence_adv.temperature=$sentence_adv_temperature \
    algorithm.sentence_adv.metrics_enable=$sentence_adv_metrics \
    algorithm.sentence_adv.metrics_max_sentences=$sentence_adv_metrics_max_sentences \
    algorithm.sentence_adv.metrics_max_pairs=$sentence_adv_metrics_max_pairs \
    algorithm.sentence_adv.metrics_pos_bins=$sentence_adv_metrics_pos_bins \
    algorithm.sentence_adv.metrics_divergence_threshold=$sentence_adv_metrics_divergence_threshold \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    +data.apply_chat_template_kwargs.enable_thinking=$qwen3_enable_thinking \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.enable_sentencepo=true \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=sentencepo \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
    critic.checkpoint.save_contents='["model"]' \
    critic.checkpoint.load_contents='["model"]' \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="verl_${MODEL_NAME}_${DS}" \
    trainer.experiment_name="sentencepo_ep${EPOCHS}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir=$HOME/autodl-tmp/models/sentencepo_${DS}_${MODEL_NAME}_ep${EPOCHS}/verl_checkpoints_sentencepo \
    "$@" 2>&1 | tee $HOME/autodl-tmp/models/sentencepo_${DS}_${MODEL_NAME}_ep${EPOCHS}/verl_sentencepo.log
