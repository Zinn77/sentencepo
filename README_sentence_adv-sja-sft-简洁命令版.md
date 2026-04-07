# SJA (Sentence Judge Advantage) 蒸馏与训练指南

## 背景

SJA 让模型在 rollout 后为每个句子打分（bucket + confidence），将句子级 advantage 融合到 token-level advantage 中。直接让小模型 self-judge 往往输出不稳定（非 JSON、乱码），因此采用**先蒸馏再 SFT**的方式提升 judge 质量。

## 三阶段流程

```
1. Distill:  Teacher 大模型对 rollout 样本做句子评分 → SFT parquet
2. Judge-SFT: 4B 模型 SFT，学习稳定输出严格 JSON 评分
3. RL + SJA:  self 模式（actor 兼 judge）或 split 模式（独立 judge 服务）
```

## 句子切分一致性

**重要**：数据构建和 RL 训练必须使用相同的句子切分逻辑。

所有切句统一走 `verl/utils/sentence_utils.py`，基于 token 级别切分：
- 逐 token decode，遇标点（`. ? ! 。 ？ ！`）或 `\n` 即切句
- 短句（< `min_sent_tokens=6` 个 token）合并到相邻句
- 数据构建脚本通过 `--tokenizer` 参数加载与 RL 训练相同的 tokenizer

## 两种 SJA 模式

| 模式 | 配置 | 特点 |
|------|------|------|
| **self** | `sentence_judge_backend=self` | actor 直接生成 judge JSON。部署简单，但 judge 随 actor 漂移 |
| **split** | `sentence_judge_backend=callable` | 独立 judge 服务（OpenAI 兼容接口）。更稳定，工程复杂度更高 |

## 快速开始

### 第一步：构建蒸馏数据

需要 rollout 数据（`rollout_debug/*.jsonl`，每行含 `input/output/score`）和 Teacher API。

有两种路径：

**路径 A：简单模式**（直接蒸馏，内部自动切分 train/val）

```bash
cd $HOME/sentencepo_v1-4-metrics-sft

python scripts/sja/build_judge_sft_data.py \
  --input-glob "$HOME/autodl-tmp/.../rollout_debug/*.jsonl" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --teacher-model "qwen3-max-2026-01-23" \
  --out-train "$HOME/data/sja_distill/train.parquet" \
  --out-val "$HOME/data/sja_distill/val.parquet"
```

**路径 B：课程化模式**（先清洗+按题防泄漏切分，再蒸馏）

先用 `prepare_judge_curriculum_data.py` 做数据清洗和 train/val 切分（按题分组，同题不同时出现在 train/val），再对已切分好的文件分别蒸馏：

```bash
# 步骤 1: 清洗 + 课程化切分
python scripts/sja/prepare_judge_curriculum_data.py \
  --input-glob "$HOME/autodl-tmp/.../rollout_debug/*.jsonl" \
  --out-dir "$HOME/data/sja_curriculum/qwen3-4b-instruct" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --num-phases 4 --val-ratio 0.05

# 步骤 2: 分别对 train 和 val 蒸馏（用 --out-parquet 单文件输出，不再内部切分）
python scripts/sja/build_judge_sft_data.py \
  --input-glob "$HOME/data/sja_curriculum/qwen3-4b-instruct/clean_train.jsonl" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --teacher-model "qwen3-max-2026-01-23" \
  --out-parquet "$HOME/data/sja_curriculum/qwen3-4b-instruct/distill_global_train.parquet"

python scripts/sja/build_judge_sft_data.py \
  --input-glob "$HOME/data/sja_curriculum/qwen3-4b-instruct/clean_val.jsonl" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --teacher-model "qwen3-max-2026-01-23" \
  --out-parquet "$HOME/data/sja_curriculum/qwen3-4b-instruct/distill_global_val.parquet"
```

也可以一键执行路径 B：`bash scripts/sja/run_prepare_judge_curriculum.sh`

`--tokenizer` **必须**与 RL 训练使用的模型一致，否则句子边界会错位。

### 第二步：Judge-SFT

```bash
bash scripts/sja/run_judge_sft_qwen3_4b.sh \
  TRAIN_FILE=$HOME/data/sja_distill/train.parquet \
  VAL_FILE=$HOME/data/sja_distill/val.parquet \
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
```

### 第三步：RL 接入

**self 模式**（推荐先用这个验证链路）：
```bash
bash scripts/sja/run_rl_sja_self.sh \
  MODEL_PATH=$HOME/autodl-tmp/models_v1-4-metrics-sft/your_judge_sft_ckpt
```

**split 模式**（需先启动 judge 服务）：
```bash
# 启动 judge 服务
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model $HOME/autodl-tmp/models_v1-4-metrics-sft/your_judge_sft_ckpt \
  --served-model-name qwen3-4b-judge \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 2

# 启动 RL
bash scripts/sja/run_rl_sja_split.sh \
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
  SJA_JUDGE_MODEL=qwen3-4b-judge \
  SJA_JUDGE_BASE_URL=http://127.0.0.1:8000/v1
```

### 可选：Judge SFT 混合损失（持续进化打分能力）

在 RL 训练中同时加入 judge SFT loss，让模型的打分能力随 RL 训练持续进化，无需调用 Teacher API：

```bash
judge_sft_enable=true \
judge_sft_data_path=$HOME/data/sja_distill/train.parquet \
judge_sft_lambda=0.1 \
  bash scripts/sja/run_rl_sja_self.sh \
  MODEL_PATH=$HOME/autodl-tmp/models_v1-4-metrics-sft/your_judge_sft_ckpt
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `judge_sft_enable` | false | 开关 |
| `judge_sft_data_path` | "" | 蒸馏后的 parquet 路径 |
| `judge_sft_lambda` | 0.1 | judge SFT loss 权重 |
| `judge_sft_micro_batch_size` | 2 | 每个 gradient step 采样的 judge 样本数 |
| `judge_sft_max_seq_len` | 2048 | 最大序列长度 |

监控指标：
- `judge_sft/loss` — cross-entropy loss（应随训练下降）
- `judge_sft/lambda` — 当前权重

## 没有 rollout 数据？先采样

```bash
sentence_judge_enable=false enable_rollout_data_dump=true \
  bash test_sentencepo_v1-4.sh
```

完成后在实验目录下得到 `rollout_debug/*.jsonl`，然后按上述流程继续。

## 脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/sja/prepare_judge_curriculum_data.py` | rollout 清洗 + 按题 train/val 切分 |
| `scripts/sja/build_judge_sft_data.py` | 切句 + 渲染 judge prompt + Teacher API 打分 → SFT parquet |
| `scripts/sja/run_prepare_judge_curriculum.sh` | 一键完成预处理 + Teacher 蒸馏（路径 B） |
| `scripts/sja/run_judge_sft_qwen3_4b.sh` | Judge-SFT 训练 |
| `scripts/sja/run_rl_sja_self.sh` | self 模式 RL |
| `scripts/sja/run_rl_sja_split.sh` | split 模式 RL |

## `build_judge_sft_data.py` 输出模式

| 参数 | 场景 | 说明 |
|------|------|------|
| `--out-train` + `--out-val` | 直接蒸馏（路径 A） | 内部按 `--val-ratio` 随机切分 train/val |
| `--out-parquet` | 搭配 `prepare`（路径 B） | 单文件输出，不做 train/val 切分（输入已由 `prepare` 分好） |

常用可观测性/容错参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log-every` | 20 | 每 N 条输出一次进度（0=关闭周期进度） |
| `--progress-log-file` | `""` | 进度日志路径；为空时自动使用 `<out-parquet>.progress.log` |
| `--records-log-file` | `""` | 增量结果 jsonl；为空时自动使用 `<out-parquet>.records.jsonl`，支持续跑 |
| `--raw-log-file` | `""` | Teacher 原始返回日志；为空时自动使用 `<out-parquet>.teacher_raw.jsonl` |
| `--num-workers` | 4 | Teacher API 并发线程数 |

`<out-parquet>.teacher_raw.jsonl` 可用于排障：
- `status="ok"`：看 `teacher_raw_content` / `teacher_raw_json`，定位 `missing_from_teacher`
- `status="error"`：看 `error`，定位 `skipped` 来源（超时/解析失败/网络）

## Teacher API 配置

支持任何 OpenAI 兼容 API。认证方式（按优先级）：
1. `--teacher-api-key` 命令行参数
2. `SJA_TEACHER_API_KEY` 环境变量
3. `DASHSCOPE_API_KEY` 环境变量
4. `OPENAI_API_KEY` 环境变量

DashScope 示例：
```bash
export DASHSCOPE_API_KEY="sk-xxx"
# --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1"
# --teacher-model "qwen3-max-2026-01-23-2026-01-23"
```

## 关键监控指标

- `sentence_judge/parse_fail_rate` — JSON 解析失败率（SFT 后应 < 5%）
- `sentence_judge/conf_mean` — 平均 confidence
- `sentence_judge/zero_frac` — advantage 为 0 的比例
- `sentence_judge/consistency_fix_rate` — 一致性修正比例
- `judge_sft/loss` — judge SFT cross-entropy loss（仅当 `judge_sft_enable=true` 时）
