#!/usr/bin/env bash
# v1-5-hidden 消融的 Phase B 单 run 参数化脚本。
#
#   MODULE   = scr | slpa | none      （启用哪个句子级 advantage 模块）
#   LAYER    = int (例如 -1 / -9 / -18) 或 "-1|-9|-18" 表示多层 ensemble mean
#   POOLING  = last | mean | first | mean_no_punct | entropy_weighted | diff
#   ALPHA    = 融合权重（默认 0.05）
#
# 示例：
#   MODULE=scr LAYER=-9 POOLING=mean ALPHA=0.05 SEED=42 EPOCHS=1 \
#     bash test_v1-5_hidden_phaseB.sh
#
# 一般由 test_v1-5_hidden_pipeline.sh 调用；单独运行也行（例如失败后想重跑
# 某一个 run，或者用 PHASEB_ONLY 跑指定的某一个）。
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

# HF 缓存（MODEL_PATH 是 hub id 时，Qwen3-4B-Base 从缓存里解析加载）。
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/autodl-tmp/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}

# 本地路径（机器布局不同时通过 env 变量覆盖）。
DATA_DIR=${DATA_DIR:-$HOME/data}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/autodl-tmp/models_v1-5}
REPO_DIR=${REPO_DIR:-$HOME/sentencepo_v1-5}

math_train_path=${DATA_DIR}/math/train.parquet
math500_test_path=${DATA_DIR}/math500/test.parquet
aime2024_test_path=${DATA_DIR}/aime2024/test.parquet
aime2025_test_path=${DATA_DIR}/aime2025/test.parquet
amc23_test_path=${DATA_DIR}/amc23/test.parquet
minerva_test_path=${DATA_DIR}/minerva/test.parquet
olympiad_train_path=${DATA_DIR}/olympiad/train.parquet
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

# === 模块启用开关 ===
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

# === 转换 layer 规格 ===
# Hydra 对 list 类字段接收形如 [-1,-9,-18] 的 list literal，对 int 类字段接收
# -9 这样的整数。SentenceReprConfig.hidden_layer_index 字段是 Any，所以这里
# 统一传 Python literal 字符串即可。
if [[ "$LAYER" == *"|"* ]]; then
    LAYER_ARG="[$(echo "$LAYER" | tr '|' ',')]"
else
    LAYER_ARG="$LAYER"
fi

# === 实验 tag ===
LAYER_TAG=$(echo "$LAYER" | tr '|' '_' | tr '-' 'm')
EXP_TAG=${EXP_TAG_OVERRIDE:-"${MODULE}_L${LAYER_TAG}_P${POOLING}_a${ALPHA}"}
TOT_DIR="${OUTPUT_DIR}/${EXP_TAG}_${DS}_${MODEL_NAME}_ep${EPOCHS}_rand${SEED}"
mkdir -p $TOT_DIR
mkdir -p $TOT_DIR/verl_checkpoints_v1-5

# === sentencepo loss 标准参数（loss=vanilla 时大多数是惰性的，不会被用到） ===
sentencepo_min_sent_tokens=6
sentencepo_eps_base=0.01
sentencepo_lambda_ppl=0
sentencepo_lambda_len=0
sentencepo_cmin=0.5
sentencepo_cmax=1.5
sentencepo_stats_eps=1e-6
sentencepo_metrics_level=full
sentencepo_per_sentence_adv=false

# === SLPA 默认参数 ===
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

# === SCR 默认参数 ===
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

export PYTHONPATH="${REPO_DIR}:$PYTHONPATH"
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
