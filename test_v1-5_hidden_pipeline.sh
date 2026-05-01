#!/usr/bin/env bash
# ============================================================================
# v1-5-hidden 全流程消融 PIPELINE —— 单 GPU 节点，前台执行。
#
# 运行内容：
#   Phase A：scripts_server/diagnose_sentence_repr.py        （~30 分钟, 1 GPU）
#   Phase B：通过 test_v1-5_hidden_phaseB.sh 串行跑 7 个 RL 实验（每个 8 GPU）
#       1) phaseB_baseline       —— GRPO + vanilla loss，不开 SCR/SLPA
#       2) phaseB_scr_current    —— SCR  layer=-1 pooling=last（现状默认）
#       3) phaseB_scr_top1       —— SCR  Phase A top-1
#       4) phaseB_scr_top2       —— SCR  Phase A top-2
#       5) phaseB_slpa_current   —— SLPA layer=-1 pooling=last（现状默认）
#       6) phaseB_slpa_top1      —— SLPA Phase A top-1
#       7) phaseB_slpa_top2      —— SLPA Phase A top-2
#
# 日志：本脚本本身的 driver 输出 tee 到：
#   $OUTPUT_DIR/pipeline_<时间戳>.log
# 每个 phaseB run 额外 tee 一份到自己的目录：
#   $OUTPUT_DIR/<exp_tag>_<...>/verl_v1-5-hidden.log
# 所以可以直接在前台跑（不需要 nohup）；即便终端断了，所有日志都已落盘。
# 想抗 SSH 断连可以自己再套一层 tmux/screen。
#
# GPU 机器上需要的前置条件：
#   - 已激活 conda 环境 `verl2`：       conda activate verl2
#   - 仓库在 $HOME/sentencepo_v1-5
#   - 数据在 $HOME/data/{math,math500,aime2024,aime2025,amc23,minerva,olympiad}/
#   - HF 缓存在 $HOME/autodl-tmp/huggingface（Qwen3-4B-Base 已下载）
#   - 8 张 GPU 可见
#
# 用法：
#   bash test_v1-5_hidden_pipeline.sh                          # 全流程
#   SKIP_PHASE_A=1 bash test_v1-5_hidden_pipeline.sh           # 只跑 Phase B
#   SKIP_PHASE_B=1 bash test_v1-5_hidden_pipeline.sh           # 只跑 Phase A
#   PHASEB_ONLY=phaseB_scr_top1 bash test_v1-5_hidden_pipeline.sh
#   STOP_ON_FAIL=1 bash test_v1-5_hidden_pipeline.sh           # 首次失败即中止
#
# Phase B 的 top1/top2 通过环境变量传入（下面的默认值只是按 probing 文献写的
# 启发，跑 Phase B 之前一定要先看 Phase A 的 markdown 输出再覆盖）：
#   SCR_TOP1_LAYER  SCR_TOP1_POOL  SCR_TOP2_LAYER  SCR_TOP2_POOL
#   SLPA_TOP1_LAYER SLPA_TOP1_POOL SLPA_TOP2_LAYER SLPA_TOP2_POOL
#
# Layer 可以是单 int（"-9"）或 "|" 分隔的 ensemble（"-1|-9|-18"）。
# Pooling 取值：last | mean | first | mean_no_punct | entropy_weighted | diff
#
# 总耗时：8×A100 上约 8–24 小时（视 response 长度而定）。
# ============================================================================
set -e

# === 路径（机器布局不同时通过 env 覆盖） ===
REPO_DIR=${REPO_DIR:-$HOME/sentencepo_v1-5}
DATA_DIR=${DATA_DIR:-$HOME/data}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/autodl-tmp/models_v1-5}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
MODEL_NAME=${MODEL_NAME:-qwen3_4b}

# === Driver 级别 tee 日志（覆盖 Phase A 的 stdout + Phase B 的汇总） ===
mkdir -p "$OUTPUT_DIR"
PIPELINE_LOG=${PIPELINE_LOG:-${OUTPUT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "$PIPELINE_LOG") 2>&1
echo "Driver 日志：$PIPELINE_LOG"

# === HF 缓存 + 运行时环境变量 ===
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/autodl-tmp/huggingface}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTHONPATH=$REPO_DIR:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === 实验旋钮 ===
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-1}
ALPHA=${ALPHA:-0.05}
DS=${DS:-math}

# 把路径相关的覆盖透传给 test_v1-5_hidden_phaseB.sh
export REPO_DIR DATA_DIR OUTPUT_DIR MODEL_PATH MODEL_NAME

cd "$REPO_DIR"

# === 前置检查 ===
if [ ! -f "$DATA_DIR/math/train.parquet" ]; then
    echo "错误：$DATA_DIR/math/train.parquet 不存在" >&2; exit 1
fi
if [ ! -d "$REPO_DIR/verl" ]; then
    echo "错误：$REPO_DIR 看起来不像 sentencepo 仓库（缺 verl/ 目录）" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Phase A —— 句子表征诊断
# ---------------------------------------------------------------------------
PHASE_A_DATE=${PHASE_A_DATE:-$(date +%Y-%m-%d)}
PHASE_A_OUT=${PHASE_A_OUT:-${REPO_DIR}/CCdocs/${PHASE_A_DATE}_phaseA_diagnostic.md}

if [ "${SKIP_PHASE_A:-0}" != "1" ]; then
    echo "============================================================"
    echo "Phase A 诊断"
    echo "  输出：$PHASE_A_OUT"
    echo "============================================================"
    mkdir -p "$(dirname "$PHASE_A_OUT")"
    python3 scripts_server/diagnose_sentence_repr.py \
        --model_path "$MODEL_PATH" \
        --train_parquet "$DATA_DIR/math/train.parquet" \
        --output_md "$PHASE_A_OUT" \
        --num_prompts 32 \
        --rollouts_per_prompt 4 \
        --max_new_tokens 512 \
        --seed "$SEED"
    echo "Phase A 完成。请查看 $PHASE_A_OUT，若与默认 top1/top2 不一致，覆盖 SCR_*/SLPA_* 环境变量后再跑 Phase B。"
fi

if [ "${SKIP_PHASE_B:-0}" = "1" ]; then
    echo "SKIP_PHASE_B=1，Phase B 之前就退出。"
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase B —— RL 消融实验（串行）
# ---------------------------------------------------------------------------

# 这里的默认值只是按 probing 文献写的启发，实际请用 Phase A 选出的 winner 替换。
SCR_TOP1_LAYER="${SCR_TOP1_LAYER:--9}"
SCR_TOP1_POOL="${SCR_TOP1_POOL:-mean}"
SCR_TOP2_LAYER="${SCR_TOP2_LAYER:--18}"
SCR_TOP2_POOL="${SCR_TOP2_POOL:-mean_no_punct}"
SLPA_TOP1_LAYER="${SLPA_TOP1_LAYER:--9}"
SLPA_TOP1_POOL="${SLPA_TOP1_POOL:-mean}"
SLPA_TOP2_LAYER="${SLPA_TOP2_LAYER:--1|-9|-18}"
SLPA_TOP2_POOL="${SLPA_TOP2_POOL:-mean}"

echo ""
echo "============================================================"
echo "Phase B 配置"
printf "  SCR  top1 layer=%-12s pooling=%s\n" "$SCR_TOP1_LAYER"  "$SCR_TOP1_POOL"
printf "  SCR  top2 layer=%-12s pooling=%s\n" "$SCR_TOP2_LAYER"  "$SCR_TOP2_POOL"
printf "  SLPA top1 layer=%-12s pooling=%s\n" "$SLPA_TOP1_LAYER" "$SLPA_TOP1_POOL"
printf "  SLPA top2 layer=%-12s pooling=%s\n" "$SLPA_TOP2_LAYER" "$SLPA_TOP2_POOL"
echo "  alpha=$ALPHA seed=$SEED epochs=$EPOCHS"
echo "  输出目录：$OUTPUT_DIR"
echo "============================================================"

PIPELINE_START=$(date +%s)
declare -a SUCCEEDED=()
declare -a FAILED=()

run_phaseB() {
    local tag=$1 module=$2 layer=$3 pool=$4
    if [ -n "${PHASEB_ONLY:-}" ] && [ "$PHASEB_ONLY" != "$tag" ]; then
        echo "[跳过] $tag （PHASEB_ONLY=$PHASEB_ONLY）"
        return
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo "Phase B run：$tag"
    echo "  module=$module layer=$layer pooling=$pool"
    echo "  开始时间：$(date -Iseconds)"
    echo "------------------------------------------------------------"
    local t0=$(date +%s)
    if MODULE="$module" LAYER="$layer" POOLING="$pool" \
       ALPHA="$ALPHA" SEED="$SEED" EPOCHS="$EPOCHS" DS="$DS" \
       EXP_TAG_OVERRIDE="$tag" \
       bash "$REPO_DIR/test_v1-5_hidden_phaseB.sh"; then
        local dt=$(( $(date +%s) - t0 ))
        echo "[完成] $tag （${dt}s）"
        SUCCEEDED+=("$tag")
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        echo "[失败] $tag rc=$rc （${dt}s）"
        FAILED+=("$tag")
        if [ "${STOP_ON_FAIL:-0}" = "1" ]; then
            echo "STOP_ON_FAIL=1，中止 pipeline。"
            exit "$rc"
        fi
    fi
}

run_phaseB phaseB_baseline      none -1                  last
run_phaseB phaseB_scr_current   scr  -1                  last
run_phaseB phaseB_scr_top1      scr  "$SCR_TOP1_LAYER"   "$SCR_TOP1_POOL"
run_phaseB phaseB_scr_top2      scr  "$SCR_TOP2_LAYER"   "$SCR_TOP2_POOL"
run_phaseB phaseB_slpa_current  slpa -1                  last
run_phaseB phaseB_slpa_top1     slpa "$SLPA_TOP1_LAYER"  "$SLPA_TOP1_POOL"
run_phaseB phaseB_slpa_top2     slpa "$SLPA_TOP2_LAYER"  "$SLPA_TOP2_POOL"

echo ""
echo "============================================================"
echo "Pipeline 完成，总耗时 $(( ($(date +%s) - PIPELINE_START) / 60 )) 分钟"
echo "  成功 (${#SUCCEEDED[@]})：${SUCCEEDED[*]:-无}"
echo "  失败 (${#FAILED[@]})：${FAILED[*]:-无}"
echo "  实验输出位于：$OUTPUT_DIR"
echo "  driver 日志：$PIPELINE_LOG"
echo "============================================================"

[ ${#FAILED[@]} -eq 0 ]
