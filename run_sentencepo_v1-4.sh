#!/usr/bin/env bash
# 关键修复1：增加set -e 脚本出错立即退出，set -x打印调试日志，安全+调试双保障
set -ex

# 固定随机数种子
SEED=${SEED:-42}
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1

TOT_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02"

# 将 verl 仓库目录添加到 Python 模块搜索路径
export PYTHONPATH="${TOT_PATH}/sentencepo_v1-4-metrics-sft:$PYTHONPATH"

# 数据集设置
math_train_path=${TOT_PATH}/data/math/train.parquet
math500_test_path=${TOT_PATH}/data/math500/test.parquet
aime2024_test_path=${TOT_PATH}/data/aime2024/test.parquet
aime2025_test_path=${TOT_PATH}/data/aime2025/test.parquet
amc23_test_path=${TOT_PATH}/data/amc23/test.parquet
minerva_test_path=${TOT_PATH}/data/minerva/test.parquet
olympiad_train_path=${TOT_PATH}/data/olympiad/train.parquet
train_files="[\"${math_train_path}\"]"
test_files="[\"${math500_test_path}\", \"${aime2024_test_path}\", \"${aime2025_test_path}\", \"${amc23_test_path}\", \"${minerva_test_path}\", \"${olympiad_train_path}\"]"

# 训练设置
DS=${DS:-math}
EPOCHS=${EPOCHS:-3}
MODEL_PATH=${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai01/huggingface.co/Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3-4b-base}
lr=${lr:-1e-6}
qwen3_enable_thinking=False # 关闭思考模式
max_prompt_length=${max_prompt_length:-1024}
max_response_length=${max_response_length:-4096}
# max_num_batched_tokens >= max_prompt_length + max_response_length，或开启 enable_chunked_prefill
max_num_batched_tokens=${max_num_batched_tokens:-8192}
# self-judge prompt = prompt + numbered response sentences + template/json.
# rollout_max_model_len = self-judge prompt + judge_max_tokens
sentence_judge_template_json_tokens=${sentence_judge_template_json_tokens:-4096} # self-judge 输入模板和输出 JSON 的最大 token 数
rollout_max_model_len=${rollout_max_model_len:-$((max_prompt_length + max_response_length + sentence_judge_template_json_tokens))}
if (( max_num_batched_tokens < rollout_max_model_len )); then
	max_num_batched_tokens=$rollout_max_model_len
fi
micro_batch_size=${micro_batch_size:-4}
rollout_n=${rollout_n:-8}
rollout_tensor_model_parallel_size=${rollout_tensor_model_parallel_size:-1}
rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization:-0.7}
actor_param_offload=${actor_param_offload:-true}
ref_param_offload=${ref_param_offload:-true}
trainer_n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE:-8}
model_dtype=${model_dtype:-fp32} # 默认 fp32，可换成 bfloat16 或者其他支持的类型以节省显存
clip_ratio_low=0.1
clip_ratio_high=0.1
# sentencepo_clip
sentencepo_min_sent_tokens=6
sentencepo_eps_base=0.01
sentencepo_lambda_ppl=0
sentencepo_lambda_len=0
sentencepo_cmin=0.5
sentencepo_cmax=1.5
sentencepo_stats_eps=${sentencepo_stats_eps:-1e-6}
sentencepo_metrics_level=${sentencepo_metrics_level:-off} # 可选 full / basic / off，basic 模式下不记录分句相关指标

# v1-2 句子熵优势开关与参数
### entropy 一般 0.1 左右，v1-1 adv 均值 1e-6，min -2.5, max 2.5
sentencepo_adv_entropy_enable=${sentencepo_adv_entropy_enable:-false}
sentencepo_adv_entropy_alpha_pos=${sentencepo_adv_entropy_alpha_pos:-0.1}
sentencepo_adv_entropy_alpha_neg=${sentencepo_adv_entropy_alpha_neg:-0.0}
sentencepo_adv_entropy_norm=${sentencepo_adv_entropy_norm:-zscore}
sentencepo_adv_entropy_clip=${sentencepo_adv_entropy_clip:-2.0}
sentencepo_adv_entropy_eps=${sentencepo_adv_entropy_eps:-1e-6}

# 句子语义优势（分桶）开关与参数
sentence_adv_enable=${sentence_adv_enable:-false}
sentence_adv_alpha=${sentence_adv_alpha:-0.1}
sentence_adv_temperature=${sentence_adv_temperature:-0.1}
sentence_adv_pooling=${sentence_adv_pooling:-last}
sentence_adv_normalize=${sentence_adv_normalize:-true}
sentence_adv_bucket_count=${sentence_adv_bucket_count:-3}
sentence_adv_correctness_threshold=${sentence_adv_correctness_threshold:-0.0}
sentence_adv_metrics_enable=${sentence_adv_metrics_enable:-true}

# 句子 Judge 优势（SJA）开关与参数
sentence_judge_enable=${sentence_judge_enable:-true}
sentence_judge_alpha=${sentence_judge_alpha:-0.2}
sentence_judge_backend=${sentence_judge_backend:-self}
sentence_judge_fn=${sentence_judge_fn:-null}
sentence_judge_correctness_threshold=${sentence_judge_correctness_threshold:-0.0}
sentence_judge_use_confidence_weight=${sentence_judge_use_confidence_weight:-true}
sentence_judge_confidence_floor=${sentence_judge_confidence_floor:-0.2}
sentence_judge_normalize=${sentence_judge_normalize:-zscore}
sentence_judge_max_sentences=${sentence_judge_max_sentences:-0} # 每条样本最多处理句子数（0 表示不限制）
sentence_judge_max_chars=${sentence_judge_max_chars:-0} # 每条样本文本最大字符数（0 表示不限制）
sentence_judge_max_tokens=${sentence_judge_max_tokens:-2048} # self-judge 生成 JSON 的最大 token 数
sentence_judge_rate_limit_qps=${sentence_judge_rate_limit_qps:-0.0}
sentence_judge_debug_prompt=${sentence_judge_debug_prompt:-false}
sentence_judge_every_n_steps=${sentence_judge_every_n_steps:-1}
sentence_judge_truncate_prompt=${sentence_judge_truncate_prompt:-true}

# Judge SFT 混合损失（在 RL 训练中持续进化打分能力）
judge_sft_enable=${judge_sft_enable:-false}
judge_sft_data_path=${judge_sft_data_path:-""}  # 蒸馏后的 parquet 文件路径
judge_sft_lambda=${judge_sft_lambda:-0.1}
judge_sft_micro_batch_size=${judge_sft_micro_batch_size:-2}
judge_sft_max_seq_len=${judge_sft_max_seq_len:-2048}

# 诊断与可视化开关
enable_sentence_analysis=${enable_sentence_analysis:-false}
sentence_analysis_max_samples=${sentence_analysis_max_samples:-8}
sentence_analysis_top_k=${sentence_analysis_top_k:-3}
sentence_analysis_group_by_uid=${sentence_analysis_group_by_uid:-true}
response_len_bins=${response_len_bins:-"[128,256,512,1024,2048,4096]"}
prompt_len_bins=${prompt_len_bins:-"[64,128,256,512,1024]"}
analysis_response_len_bins=${analysis_response_len_bins:-"[128,256,512,1024,2048,4096]"}
analysis_sentence_count_bins=${analysis_sentence_count_bins:-"[4,8,16,32,64]"}
analysis_max_sentence_len_bins=${analysis_max_sentence_len_bins:-"[32,64,128,256,512]"}


# 创建tensorboard日志目录，不存在则创建，防止日志丢失
ts_dir="${TOT_PATH}/models_v1-4-metrics-sft/sentencepo_${DS}_${MODEL_NAME}_ep${EPOCHS}_epsbase${sentencepo_eps_base}_Lppl${sentencepo_lambda_ppl}_Llen${sentencepo_lambda_len}_cmin${sentencepo_cmin}_cmax${sentencepo_cmax}_min-token${sentencepo_min_sent_tokens}"
if [[ "$sentencepo_adv_entropy_enable" == "true" ]]; then
    ts_dir="${ts_dir}_adv-entropy-pos${sentencepo_adv_entropy_alpha_pos}-neg${sentencepo_adv_entropy_alpha_neg}-norm${sentencepo_adv_entropy_norm}"
fi
if [[ "$sentence_adv_enable" == "true" ]]; then
    ts_dir="${ts_dir}_adv-compare-alpha${sentence_adv_alpha}-temp${sentence_adv_temperature}-pool${sentence_adv_pooling}"
fi
if [[ "$sentence_judge_enable" == "true" ]]; then
    ts_dir="${ts_dir}_sja-alpha${sentence_judge_alpha}"
fi
if [[ "$judge_sft_enable" == "true" ]]; then
    ts_dir="${ts_dir}_jsft-lambda${judge_sft_lambda}"
fi
export TENSORBOARD_DIR="${ts_dir}"
mkdir -p ${TENSORBOARD_DIR}

sentence_analysis_dir=${sentence_analysis_dir:-"${TENSORBOARD_DIR}/sentence_analysis"}

# 是否开启 rollout 数据 dump 以便后续分析
enable_rollout_data_dump=${enable_rollout_data_dump:-true}
rollout_data_dir=${rollout_data_dir:-"${TENSORBOARD_DIR}/rollout_debug"}
rollout_data_dir_arg=""
if [[ "$enable_rollout_data_dump" == "true" ]]; then
	mkdir -p "$rollout_data_dir"
	rollout_data_dir_arg="trainer.rollout_data_dir=$rollout_data_dir"
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
	algorithm.adv_estimator=grpo \
	+data.seed=$SEED \
	+critic.data_loader_seed=$SEED \
	data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    +data.apply_chat_template_kwargs.enable_thinking=${qwen3_enable_thinking} \
    data.filter_overlong_prompts=True \
	data.truncation='error' \
	+data.enable_sentencepo=true \
	actor_rollout_ref.model.path=${MODEL_PATH} \
	actor_rollout_ref.actor.optim.lr=${lr} \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.use_dynamic_bsz=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=16 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size} \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.entropy_coeff=0 \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
	actor_rollout_ref.actor.fsdp_config.model_dtype=${model_dtype} \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
	actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
	actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
	actor_rollout_ref.rollout.enforce_eager=False \
	actor_rollout_ref.rollout.free_cache_engine=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
	actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tensor_model_parallel_size} \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
	actor_rollout_ref.rollout.n=${rollout_n} \
	actor_rollout_ref.rollout.enable_chunked_prefill=True \
	actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
	actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
	actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
	actor_rollout_ref.ref.fsdp_config.model_dtype=${model_dtype} \
	actor_rollout_ref.actor.policy_loss.loss_mode=sentencepo \
	+actor_rollout_ref.actor.policy_loss.sentencepo_min_sent_tokens=${sentencepo_min_sent_tokens} \
	+actor_rollout_ref.actor.policy_loss.sentencepo_eps_base=${sentencepo_eps_base} \
	+actor_rollout_ref.actor.policy_loss.sentencepo_lambda_ppl=${sentencepo_lambda_ppl} \
	+actor_rollout_ref.actor.policy_loss.sentencepo_lambda_len=${sentencepo_lambda_len} \
	+actor_rollout_ref.actor.policy_loss.sentencepo_cmin=${sentencepo_cmin} \
	+actor_rollout_ref.actor.policy_loss.sentencepo_cmax=${sentencepo_cmax} \
	+actor_rollout_ref.actor.policy_loss.sentencepo_stats_eps=${sentencepo_stats_eps} \
    +actor_rollout_ref.actor.policy_loss.sentencepo_metrics_level=$sentencepo_metrics_level \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_enable=$sentencepo_adv_entropy_enable \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_alpha_pos=$sentencepo_adv_entropy_alpha_pos \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_alpha_neg=$sentencepo_adv_entropy_alpha_neg \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_norm=$sentencepo_adv_entropy_norm \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_clip=$sentencepo_adv_entropy_clip \
    actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_eps=$sentencepo_adv_entropy_eps \
	+algorithm.sentence_adv.enable=$sentence_adv_enable \
	+algorithm.sentence_adv.alpha=$sentence_adv_alpha \
	+algorithm.sentence_adv.temperature=$sentence_adv_temperature \
	+algorithm.sentence_adv.pooling=$sentence_adv_pooling \
	+algorithm.sentence_adv.normalize=$sentence_adv_normalize \
	+algorithm.sentence_adv.bucket_count=$sentence_adv_bucket_count \
	+algorithm.sentence_adv.correctness_threshold=$sentence_adv_correctness_threshold \
	+algorithm.sentence_adv.metrics_enable=$sentence_adv_metrics_enable \
	algorithm.sentence_judge_adv.enable=$sentence_judge_enable \
	algorithm.sentence_judge_adv.alpha=$sentence_judge_alpha \
	algorithm.sentence_judge_adv.judge_backend=$sentence_judge_backend \
	algorithm.sentence_judge_adv.judge_fn=$sentence_judge_fn \
	algorithm.sentence_judge_adv.correctness_threshold=$sentence_judge_correctness_threshold \
	algorithm.sentence_judge_adv.use_confidence_weight=$sentence_judge_use_confidence_weight \
	algorithm.sentence_judge_adv.confidence_floor=$sentence_judge_confidence_floor \
	algorithm.sentence_judge_adv.normalize=$sentence_judge_normalize \
	algorithm.sentence_judge_adv.max_sentences=$sentence_judge_max_sentences \
	algorithm.sentence_judge_adv.max_chars=$sentence_judge_max_chars \
	algorithm.sentence_judge_adv.judge_max_tokens=$sentence_judge_max_tokens \
	algorithm.sentence_judge_adv.rate_limit_qps=$sentence_judge_rate_limit_qps \
	algorithm.sentence_judge_adv.debug_prompt=$sentence_judge_debug_prompt \
	algorithm.sentence_judge_adv.every_n_steps=$sentence_judge_every_n_steps \
	+algorithm.sentence_judge_adv.truncate_prompt=$sentence_judge_truncate_prompt \
	+algorithm.judge_sft.enable=$judge_sft_enable \
	+algorithm.judge_sft.data_path="$judge_sft_data_path" \
	+algorithm.judge_sft.lambda_weight=$judge_sft_lambda \
	+algorithm.judge_sft.micro_batch_size=$judge_sft_micro_batch_size \
	+algorithm.judge_sft.max_seq_len=$judge_sft_max_seq_len \
	+actor_rollout_ref.actor.policy_loss.analysis_bins.response_len_bins=$analysis_response_len_bins \
	+actor_rollout_ref.actor.policy_loss.analysis_bins.sentence_count_bins=$analysis_sentence_count_bins \
	+actor_rollout_ref.actor.policy_loss.analysis_bins.max_sentence_len_bins=$analysis_max_sentence_len_bins \
	actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
	actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
	algorithm.use_kl_in_reward=False \
	trainer.critic_warmup=0 \
	trainer.logger='["console","tensorboard"]' \
	trainer.project_name="verl_${MODEL_NAME}_${DS}" \
    trainer.experiment_name="sentencepo_ep${EPOCHS}_epsbase${sentencepo_eps_base}_Lppl${sentencepo_lambda_ppl}_Llen${sentencepo_lambda_len}_cmin${sentencepo_cmin}_cmax${sentencepo_cmax}_min-token${sentencepo_min_sent_tokens}" \
	trainer.n_gpus_per_node=${trainer_n_gpus_per_node} \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir=${TENSORBOARD_DIR} \
	trainer.use_legacy_worker_impl=disable \
	$rollout_data_dir_arg \
	+trainer.response_len_bins=$response_len_bins \
	+trainer.prompt_len_bins=$prompt_len_bins \
	+trainer.sentence_analysis.enable=$enable_sentence_analysis \
	+trainer.sentence_analysis.max_samples=$sentence_analysis_max_samples \
	+trainer.sentence_analysis.top_k=$sentence_analysis_top_k \
	+trainer.sentence_analysis.group_by_uid=$sentence_analysis_group_by_uid \
	+trainer.sentence_analysis.dir=$sentence_analysis_dir \
    "$@" 2>&1 | tee ${TENSORBOARD_DIR}/verl_sentencepo.log