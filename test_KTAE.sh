#!/usr/bin/env bash
set -ex

# 固定随机数种子
SEED=${SEED:-42}
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 数据集设置（与 GRPO/GSPO/v1-5 完全一致）
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet
math500_test_path=$HOME/data/math500/test.parquet
aime2024_test_path=$HOME/data/aime2024/test.parquet
aime2025_test_path=$HOME/data/aime2025/test.parquet
amc23_test_path=$HOME/data/amc23/test.parquet
minerva_test_path=$HOME/data/minerva/test.parquet
olympiad_train_path=$HOME/data/olympiad/train.parquet
train_files="['$math_train_path']"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$minerva_test_path', '$olympiad_train_path']"

# 训练设置（与 v1-5 / GRPO 对齐）
DS=${DS:-math}
EPOCHS=${EPOCHS:-1}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}
lr=1e-6
qwen3_enable_thinking=False
max_response_length=4096
max_num_batched_tokens=8192
micro_batch_size=4

# KTAE 超参 — 默认值匹配 KTAE upstream 的「真实训练值」（即 compute_ktae 函数
# body 里硬编码的 1/1/1/1/-1）。upstream shell run_qwen2.5_math_1.5b.sh 写
# beta_ig=2.0，但函数不读 config，所以 upstream 实际跑的 beta_ig=1.0。
KTAE_ALPHA=${KTAE_ALPHA:-1.0}
KTAE_BETA_IG=${KTAE_BETA_IG:-1.0}
KTAE_GAMMA_TF=${KTAE_GAMMA_TF:-1.0}
KTAE_TOP=${KTAE_TOP:-1.0}
KTAE_BOTTOM=${KTAE_BOTTOM:-(-1.0)}
# Qwen3-4B-Base pad/eos = 151643；Llama-3.2 EOS = 128009，需要 override
KTAE_PAD_TOKEN_ID=${KTAE_PAD_TOKEN_ID:-151643}

# 诊断（与 GRPO 对齐，关闭 sentence_analysis）
enable_sentence_analysis=${enable_sentence_analysis:-false}
sentence_analysis_max_samples=${sentence_analysis_max_samples:-4096}
sentence_analysis_top_k=${sentence_analysis_top_k:-3}
sentence_analysis_group_by_uid=${sentence_analysis_group_by_uid:-true}
response_len_bins=${response_len_bins:-"[128,256,512,1024,2048,4096]"}
prompt_len_bins=${prompt_len_bins:-"[64,128,256,512,1024]"}
analysis_response_len_bins=${analysis_response_len_bins:-"[128,256,512,1024,2048,4096]"}
analysis_sentence_count_bins=${analysis_sentence_count_bins:-"[4,8,16,32,64]"}
analysis_max_sentence_len_bins=${analysis_max_sentence_len_bins:-"[32,64,128,256,512]"}

# 结果路径
EXP_TAG=${EXP_TAG:-ktae}
TOT_DIR=$HOME/autodl-tmp/models_v1-5/${EXP_TAG}_${DS}_${MODEL_NAME}_ep${EPOCHS}_rand${SEED}
mkdir -p $TOT_DIR
mkdir -p $TOT_DIR/verl_checkpoints_ktae

sentence_analysis_dir=${sentence_analysis_dir:-"$TOT_DIR/sentence_analysis"}

cd $HOME/sentencepo_v1-5

PYTHONPATH=$HOME/sentencepo_v1-5 \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +data.seed=$SEED \
    +critic.data_loader_seed=$SEED \
    algorithm.adv_estimator=ktae \
    +algorithm.ktae.alpha=$KTAE_ALPHA \
    +algorithm.ktae.beta_ig=$KTAE_BETA_IG \
    +algorithm.ktae.gamma_tf=$KTAE_GAMMA_TF \
    +algorithm.ktae.top=$KTAE_TOP \
    +algorithm.ktae.bottom=$KTAE_BOTTOM \
    +algorithm.ktae.pad_token_id=$KTAE_PAD_TOKEN_ID \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    trainer.experiment_name="${EXP_TAG}_ep${EPOCHS}_rand${SEED}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.default_local_dir=$TOT_DIR/verl_checkpoints_ktae \
    trainer.use_legacy_worker_impl=disable \
    +trainer.response_len_bins=$response_len_bins \
    +trainer.prompt_len_bins=$prompt_len_bins \
    +trainer.sentence_analysis.enable=$enable_sentence_analysis \
    +trainer.sentence_analysis.max_samples=$sentence_analysis_max_samples \
    +trainer.sentence_analysis.top_k=$sentence_analysis_top_k \
    +trainer.sentence_analysis.group_by_uid=$sentence_analysis_group_by_uid \
    +trainer.sentence_analysis.dir=$sentence_analysis_dir \
    "$@" 2>&1 | tee $TOT_DIR/verl_ktae.log
