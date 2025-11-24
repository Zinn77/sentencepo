#!/usr/bin/env bash
set -euo pipefail

# 数据集放在 sentencepo/data 目录下，训练集为 train.parquet，验证集为 val.parquet，测试集为 test.parquet
DS=${DS:-gsm8k_hasval}
DS_TEST=${DS_TEST:-gsm8k_hasval} # math500
EPOCHS=${EPOCHS:-5}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}

# 训练，结果保存至 sentencepo/models
for file in test_sentencepo.sh test_GRPO2.sh test_GSPO2.sh
do
    DS=$DS EPOCHS=$EPOCHS MODEL_PATH=$MODEL_PATH MODEL_NAME=$MODEL_NAME \
    bash $file
done

# 合并模型并测试
for method in grpo gspo sentencepo
do
    METHOD_NAME=$method DS=$DS_TEST MODEL_NAME=$MODEL_NAME EPOCHS=$EPOCHS \
    bash test_eval.sh
done

echo "全部运行完毕"
