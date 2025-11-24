#!/usr/bin/env bash
# 如果脚本被 sh/dash 调用，重新用 bash 执行以支持 [[ ]] 和 pipefail
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# ========== 控制线程数 ==========
# 严格控制 OpenMP/MKL 线程数
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# PyTorch 线程控制
export TORCH_NUM_THREADS=1

# Ray 配置
export RAY_DEDUP_LOGS=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# 限制 Python 线程池大小
export PYTHONTHREADSPERPROCESS=4
# ========== 结束添加 ==========



# 评测已训练的模型（以 GSM8K 测试集为例）
# 步骤：
# 1) 在 $HOME/sentencepo/models/verl_checkpoints_xxx 下寻找最新的 global_step_*（也可通过 CKPT_DIR 指定）
# 2) 将 FSDP 检查点合并为 Hugging Face 权重格式
# 3) 使用合并后的模型在 GSM8K 测试集上生成回答，得到 parquet 文件
# 4) 对生成的 parquet 进行离线评测，计算准确率

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# 路径设置
METHOD_NAME=${METHOD_NAME:-sentencepo} # 方法名称
TOT_DIR=${TOT_DIR:-$HOME/sentencepo/models/${METHOD_NAME}_gsm8k_hasval_qwen2_5_7b_ep1} # 模型总目录

CKPT_ROOT=${CKPT_ROOT:-${TOT_DIR}/verl_checkpoints_${METHOD_NAME}}
DATA_PATH=${DATA_PATH:-$HOME/data/gsm8k_hasval/test.parquet} # 或 $HOME/data/aime-2024/test.parquet
OUT_PATH=${OUT_PATH:-${TOT_DIR}/eval/${METHOD_NAME}_gsm8k_hasval_gen.parquet}
# 合并后的 Hugging Face 模型存放于：MERGED_HF_ROOT/<global_step_xxx>/actor_hf
MERGED_HF_ROOT=${MERGED_HF_ROOT:-${TOT_DIR}/verl_checkpoints_${METHOD_NAME}_merged}

# 也可以通过 CKPT_DIR 指定某个明确的检查点
CKPT_DIR=${CKPT_DIR:-}
if [[ -z "${CKPT_DIR}" ]]; then
  # 查找最新的 global_step_* 目录
  if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "Checkpoint root not found: ${CKPT_ROOT}" >&2
    exit 1
  fi
  latest_dir=$(ls -d ${CKPT_ROOT}/global_step_* 2>/dev/null | awk -F'_' '{print $NF" "$0}' | sort -nr | head -n1 | cut -d' ' -f2)
  if [[ -z "${latest_dir}" ]]; then
    echo "No global_step_* found under ${CKPT_ROOT}" >&2
    exit 1
  fi
  CKPT_DIR=${latest_dir}
fi

ACTOR_DIR=${CKPT_DIR}/actor
ckpt_base_name=$(basename "${CKPT_DIR}")
HF_DIR=${MERGED_HF_ROOT}/${ckpt_base_name}/actor_hf

echo "Using checkpoint: ${CKPT_DIR}"

# 1) 将 FSDP 检查点合并为 Hugging Face 权重（幂等）
# 若目录已存在但只有 tokenizer/config（没有权重文件），仍会执行合并。
need_merge=true
if [[ -d "${HF_DIR}" ]]; then
  has_safetensors=$(ls -1 "${HF_DIR}"/*.safetensors 2>/dev/null | head -n1 || true)
  has_bin=$(ls -1 "${HF_DIR}"/pytorch_model*.bin 2>/dev/null | head -n1 || true)
  if [[ -n "${has_safetensors}" || -n "${has_bin}" ]]; then
    need_merge=false
  fi
fi

if [[ "${need_merge}" == true ]]; then
  echo "正在将 FSDP 检查点合并为 HF 权重：${HF_DIR}"
  mkdir -p "${HF_DIR}"
  python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${ACTOR_DIR} \
    --target_dir ${HF_DIR}
else
  echo "检测到 HF 目录已有权重文件：${HF_DIR}（跳过合并）"
fi

# 2) 使用合并后的模型进行生成
# 根据需要调整 GPU 配置。对于 4 卡且较小模型，TP=1 通常足够。
python3 -m verl.trainer.main_generation \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  data.path=${DATA_PATH} \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${OUT_PATH} \
  model.path=${HF_DIR} \
  +model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_k=-1 \
  rollout.top_p=1.0 \
  rollout.prompt_length=512 \
  rollout.response_length=512 \
  +rollout.pipeline_model_parallel_size=1 \
  rollout.tensor_model_parallel_size=2 \
  rollout.gpu_memory_utilization=0.5

# 3) 对生成出来的 parquet 进行离线评测（并保存全部输出到日志文件，最后一行是准确率）
EVAL_DIR=${EVAL_DIR:-${TOT_DIR}/eval}
mkdir -p "${EVAL_DIR}"
EVAL_BASE=$(basename "${OUT_PATH}")
EVAL_BASE_NOEXT=${EVAL_BASE%.*}
EVAL_LOG=${EVAL_LOG:-${EVAL_DIR}/${EVAL_BASE_NOEXT}.eval.log}

python3 -m verl.trainer.main_eval \
  data.path=${OUT_PATH} \
  data.response_key=responses \
  data.data_source_key=data_source \
  data.reward_model_key=reward_model \
  custom_reward_function.path=/root/verl/scripts/eval_reward_wrappers.py \
  custom_reward_function.name=default_eval_fn \
  2>&1 | tee "${EVAL_LOG}"
