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
EPOCHS=${EPOCHS:-3}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}
lr=1e-6
qwen3_enable_thinking=False # 关闭思考模式
max_response_length=4096
# max_num_batched_tokens >= max_prompt_length + max_response_length，或开启 enable_chunked_prefill
max_num_batched_tokens=8192    # 默认 8192
micro_batch_size=4
clip_ratio_low=0.1
clip_ratio_high=0.1
# sentencepo_v1-1 超参数
sentencepo_min_sent_tokens=6
sentencepo_eps_base=0.01
sentencepo_lambda_ppl=0
sentencepo_lambda_len=0
sentencepo_cmin=0.5
sentencepo_cmax=1.5
sentencepo_stats_eps=1e-6
sentencepo_metrics_level=${sentencepo_metrics_level:-full} # 可选 full / basic / off，basic 模式下不记录分句相关指标

# v1-2 句子熵优势开关与参数
sentencepo_adv_entropy_enable=${sentencepo_adv_entropy_enable:-false}
sentencepo_adv_entropy_alpha_pos=${sentencepo_adv_entropy_alpha_pos:-0.1}
sentencepo_adv_entropy_alpha_neg=${sentencepo_adv_entropy_alpha_neg:-0.0}
sentencepo_adv_entropy_norm=${sentencepo_adv_entropy_norm:-zscore}
sentencepo_adv_entropy_clip=${sentencepo_adv_entropy_clip:-2.0}
sentencepo_adv_entropy_eps=${sentencepo_adv_entropy_eps:-1e-6}

# v1-5 SLPA（Sentence-Level Process Advantage）参数
slpa_enable=${slpa_enable:-true}
slpa_alpha_correct=${slpa_alpha_correct:-0.1}
slpa_alpha_incorrect=${slpa_alpha_incorrect:-0.1}
slpa_tau_emb=${slpa_tau_emb:-0.1}
slpa_sigma_pos=${slpa_sigma_pos:-1.0}
slpa_normalize=${slpa_normalize:-true}
slpa_eps=${slpa_eps:-1e-8}
slpa_correctness_threshold=${slpa_correctness_threshold:-0.0}
slpa_metrics_enable=${slpa_metrics_enable:-true}

# v1-5 SCR（Sentence Contrastive Reward）参数
scr_enable=${scr_enable:-true}
scr_alpha_correct=${scr_alpha_correct:-0.05}
scr_alpha_incorrect=${scr_alpha_incorrect:-0.05}
scr_tau_reward=${scr_tau_reward:-1.0}
scr_tau_sim=${scr_tau_sim:-0.1}
scr_normalize=${scr_normalize:-true}
scr_eps=${scr_eps:-1e-8}
scr_correctness_threshold=${scr_correctness_threshold:-0.0}
scr_metrics_enable=${scr_metrics_enable:-true}

# 诊断与可视化开关
enable_sentence_analysis=${enable_sentence_analysis:-true}
sentence_analysis_max_samples=${sentence_analysis_max_samples:-4096}
sentence_analysis_top_k=${sentence_analysis_top_k:-3}
sentence_analysis_group_by_uid=${sentence_analysis_group_by_uid:-true}
response_len_bins=${response_len_bins:-"[128,256,512,1024,2048,4096]"}
prompt_len_bins=${prompt_len_bins:-"[64,128,256,512,1024]"}
analysis_response_len_bins=${analysis_response_len_bins:-"[128,256,512,1024,2048,4096]"}
analysis_sentence_count_bins=${analysis_sentence_count_bins:-"[4,8,16,32,64]"}
analysis_max_sentence_len_bins=${analysis_max_sentence_len_bins:-"[32,64,128,256,512]"}

# 结果路径
TOT_DIR=$HOME/autodl-tmp/models_v1-5/sentencepo_${DS}_${MODEL_NAME}_ep${EPOCHS}_epsbase${sentencepo_eps_base}_slpa${slpa_enable}_scr${scr_enable}
mkdir -p $TOT_DIR
mkdir -p $TOT_DIR/verl_checkpoints_sentencepo

sentence_analysis_dir=${sentence_analysis_dir:-"$TOT_DIR/sentence_analysis"}

cd $HOME/sentencepo_v1-5-combined

PYTHONPATH=$HOME/sentencepo_v1-5-combined \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=sentencepo \
    +actor_rollout_ref.actor.policy_loss.sentencepo_min_sent_tokens=$sentencepo_min_sent_tokens \
    +actor_rollout_ref.actor.policy_loss.sentencepo_eps_base=$sentencepo_eps_base \
    +actor_rollout_ref.actor.policy_loss.sentencepo_lambda_ppl=$sentencepo_lambda_ppl \
    +actor_rollout_ref.actor.policy_loss.sentencepo_lambda_len=$sentencepo_lambda_len \
    +actor_rollout_ref.actor.policy_loss.sentencepo_cmin=$sentencepo_cmin \
    +actor_rollout_ref.actor.policy_loss.sentencepo_cmax=$sentencepo_cmax \
    +actor_rollout_ref.actor.policy_loss.sentencepo_stats_eps=$sentencepo_stats_eps \
    +actor_rollout_ref.actor.policy_loss.sentencepo_metrics_level=$sentencepo_metrics_level \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_enable=$sentencepo_adv_entropy_enable \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_alpha_pos=$sentencepo_adv_entropy_alpha_pos \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_alpha_neg=$sentencepo_adv_entropy_alpha_neg \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_norm=$sentencepo_adv_entropy_norm \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_clip=$sentencepo_adv_entropy_clip \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_eps=$sentencepo_adv_entropy_eps \
    +algorithm.slpa.enable=$slpa_enable \
    +algorithm.slpa.alpha_correct=$slpa_alpha_correct \
    +algorithm.slpa.alpha_incorrect=$slpa_alpha_incorrect \
    +algorithm.slpa.tau_emb=$slpa_tau_emb \
    +algorithm.slpa.sigma_pos=$slpa_sigma_pos \
    +algorithm.slpa.normalize=$slpa_normalize \
    +algorithm.slpa.eps=$slpa_eps \
    +algorithm.slpa.correctness_threshold=$slpa_correctness_threshold \
    +algorithm.slpa.metrics_enable=$slpa_metrics_enable \
    +algorithm.scr.enable=$scr_enable \
    +algorithm.scr.alpha_correct=$scr_alpha_correct \
    +algorithm.scr.alpha_incorrect=$scr_alpha_incorrect \
    +algorithm.scr.tau_reward=$scr_tau_reward \
    +algorithm.scr.tau_sim=$scr_tau_sim \
    +algorithm.scr.normalize=$scr_normalize \
    +algorithm.scr.eps=$scr_eps \
    +algorithm.scr.correctness_threshold=$scr_correctness_threshold \
    +algorithm.scr.metrics_enable=$scr_metrics_enable \
    +actor_rollout_ref.actor.policy_loss.analysis_bins.response_len_bins=$analysis_response_len_bins \
    +actor_rollout_ref.actor.policy_loss.analysis_bins.sentence_count_bins=$analysis_sentence_count_bins \
    +actor_rollout_ref.actor.policy_loss.analysis_bins.max_sentence_len_bins=$analysis_max_sentence_len_bins \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
    critic.checkpoint.save_contents='["model"]' \
    critic.checkpoint.load_contents='["model"]' \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="verl_${MODEL_NAME}_${DS}" \
    trainer.experiment_name="sentencepo_v1-5_ep${EPOCHS}_slpa${slpa_enable}_scr${scr_enable}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir=$TOT_DIR/verl_checkpoints_sentencepo \
    trainer.use_legacy_worker_impl=disable \
    +trainer.response_len_bins=$response_len_bins \
    +trainer.prompt_len_bins=$prompt_len_bins \
    +trainer.sentence_analysis.enable=$enable_sentence_analysis \
    +trainer.sentence_analysis.max_samples=$sentence_analysis_max_samples \
    +trainer.sentence_analysis.top_k=$sentence_analysis_top_k \
    +trainer.sentence_analysis.group_by_uid=$sentence_analysis_group_by_uid \
    +trainer.sentence_analysis.dir=$sentence_analysis_dir \
    "$@" 2>&1 | tee $TOT_DIR/verl_sentencepo.log
