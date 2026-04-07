#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/autodl-tmp/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
MODEL_NAME=${MODEL_NAME:-qwen3_4b_judge_sft}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/sja_distill/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/sja_distill/val.parquet}

LR=${LR:-2e-6}
EPOCHS=${EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-8192}
MICRO_BATCH=${MICRO_BATCH:-1}
NGPU=${NGPU:-4}
DTYPE=${DTYPE:-bf16}

EXP_NAME=${EXP_NAME:-judge_sft_${MODEL_NAME}_ep${EPOCHS}_lr${LR}}
OUT_DIR=${OUT_DIR:-$HOME/autodl-tmp/models_v1-4-metrics-sft/${EXP_NAME}}
mkdir -p "$OUT_DIR"

cd "$HOME/sentencepo_v1-4-metrics-sft"

PYTHONPATH=$HOME/sentencepo_v1-4-metrics-sft \
PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length=$MAX_LENGTH \
  data.truncation=right \
  model.partial_pretrain="$MODEL_PATH" \
  model.fsdp_config.model_dtype=$DTYPE \
  model.enable_gradient_checkpointing=True \
  optim.lr=$LR \
  trainer.total_epochs=$EPOCHS \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  data.micro_batch_size_per_gpu=$MICRO_BATCH \
  trainer.project_name=judge_sft \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir="$OUT_DIR" \
  trainer.logger='["console","tensorboard"]' \
  trainer.test_freq=200 \
  "$@"
