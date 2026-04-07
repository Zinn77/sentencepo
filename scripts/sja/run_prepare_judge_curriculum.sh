#!/usr/bin/env bash
set -euo pipefail

# Pipeline: clean rollout data → teacher distill → SFT parquet
# Phase splitting is done AFTER distillation, not before.

INPUT_GLOB=${INPUT_GLOB:-"$HOME/autodl-tmp/rollout_debug/*.jsonl"}
OUT_DIR=${OUT_DIR:-"$HOME/data/sja_curriculum/qwen3-4b-instruct"}

# Tokenizer must match the RL training model to ensure consistent sentence boundaries.
TOKENIZER=${TOKENIZER:-"Qwen/Qwen3-4B-Instruct-2507"}

TEACHER_BASE_URL=${TEACHER_BASE_URL:-"https://dashscope.aliyuncs.com/compatible-mode/v1"}
TEACHER_MODEL=${TEACHER_MODEL:-"qwen3-max-2026-01-23"}
TEACHER_API_KEY=${TEACHER_API_KEY:-${DASHSCOPE_API_KEY:-}}

cd "$HOME/sentencepo_v1-4-metrics-sft"

# Step 1: Clean and split train/val (by prompt group, no data leakage).
python scripts/sja/prepare_judge_curriculum_data.py \
  --input-glob "$INPUT_GLOB" \
  --out-dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --val-ratio 0.05 \
  --max-sentences 96 \
  --max-chars 12000 \
  --keep-repeat-tail-sentences 12

# Step 2: Teacher distill train and val separately (--out-parquet = no internal split).
python scripts/sja/build_judge_sft_data.py \
  --input-glob "$OUT_DIR/clean_train.jsonl" \
  --tokenizer "$TOKENIZER" \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "$TEACHER_API_KEY" \
  --out-parquet "$OUT_DIR/distill_train.parquet"

python scripts/sja/build_judge_sft_data.py \
  --input-glob "$OUT_DIR/clean_val.jsonl" \
  --tokenizer "$TOKENIZER" \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "$TEACHER_API_KEY" \
  --out-parquet "$OUT_DIR/distill_val.parquet"

echo "Distillation finished: $OUT_DIR"
echo "  distill_train.parquet: $(python -c "import pandas as pd; print(len(pd.read_parquet('$OUT_DIR/distill_train.parquet')))" 2>/dev/null || echo '?') rows"
echo "  distill_val.parquet:   $(python -c "import pandas as pd; print(len(pd.read_parquet('$OUT_DIR/distill_val.parquet')))" 2>/dev/null || echo '?') rows"
echo ""
echo "Next: Judge-SFT → bash scripts/sja/run_judge_sft_qwen3_4b.sh TRAIN_FILE=$OUT_DIR/distill_train.parquet VAL_FILE=$OUT_DIR/distill_val.parquet"
