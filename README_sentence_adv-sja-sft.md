# SJA (Sentence Judge Advantage) 完整指南

## 1. 背景与动机

SJA 让模型在 rollout 后对每个句子打分（bucket + confidence），生成**句子级 advantage**，与序列级 GRPO advantage 融合后用于 PPO 训练。

直接让小模型（4B）self-judge 往往输出不稳定（非 JSON、乱码、评分不一致），因此采用 **Teacher 蒸馏 → Judge-SFT → RL 接入**的三阶段流程，并可在 RL 训练中持续进化打分能力。

## 2. 核心算法

### 2.1 Judge Prompt 模板

数据蒸馏和 RL 推理时使用完全相同的 prompt 模板（`sentence_judge_adv.py:_build_prompt` / `build_judge_sft_data.py:build_prompt`）：

```
You are a sentence-level judge. Score each sentence with a discrete advantage bucket and a confidence in [0,1].
Allowed buckets: [0.75, 0.25, 0.0, -0.25, -0.75].
overall_correct = true

PROMPT:
{原始数学题}

RESPONSE (sentence numbered):
S1: Let me start by analyzing the equation...
S2: We can substitute x = 3...
S3: Therefore the answer is \boxed{42}.

Output strict JSON only:
{"sentences":[{"id":1,"bucket":0.25,"confidence":0.8,"reason":"..."}]}
```

**字段含义**：
- `bucket`：离散 advantage 值，正值表示该句子对正确答案有贡献，负值表示有害
- `confidence`：模型对自己评分的置信度 [0, 1]，低于 `confidence_floor`(0.2) 的信号被置零
- `overall_correct`：基于 `sequence_reward > correctness_threshold` 判定，作为 judge 的参考信号
- `reason`：评分理由（仅用于 debug/可解释性，不参与计算）

### 2.2 Advantage 计算与融合

```
1. Judge 输出 JSON → 解析每句 (bucket, confidence)
2. confidence 加权: adv_s = bucket_s * clamp(confidence_s, 0, 1)
   （confidence < confidence_floor 时 adv_s = 0）
3. zscore 归一化: adv = (adv - mean) / std
4. 一致性修正: 如果 mean(adv) 符号与 overall_correct 不一致，则修正
5. 映射到 token: sentence_ids tensor 记录每个 token 属于哪个句子，广播句子 advantage
6. 融合: final_advantages = grpo_advantages + alpha * judge_token_advantages
```

### 2.3 句子切分

**所有场景统一使用** `verl/utils/sentence_utils.py`，基于 token 级别切分：

1. 逐 token decode，遇标点（`. ? ! 。 ？ ！`）或 `\n` 即切句
2. 短句（< `min_sent_tokens=6` 个 token）合并到相邻句
3. 数据构建脚本通过 `--tokenizer` 参数加载 **与 RL 训练相同的 tokenizer**

这保证了数据蒸馏时的句子编号与 RL 训练时的 `sentence_ids` 完全对齐。

## 3. 三阶段流程

```
rollout_debug/*.jsonl  (input/output/score/step + optional output_sentences)
  │
  ├─ [可选] prepare_judge_curriculum_data.py  →  清洗 + 按题 train/val 切分
  │                                               (clean_train.jsonl, clean_val.jsonl)
  │
  ▼
build_judge_sft_data.py  →  切句 + 渲染 judge prompt + Teacher API 打分 + 后处理
  │                          (distill_train.parquet, distill_val.parquet)
  ▼
Judge-SFT (verl fsdp_sft_trainer)  →  4B 模型学习稳定输出严格 JSON
  │
  ▼
RL + SJA  →  self 模式 / split 模式 / 可选 judge SFT 混合损失
```

## 4. 第一步：构建蒸馏数据

### 4.1 数据来源

需要 RL rollout 数据 `rollout_debug/*.jsonl`，每行 JSON 含：
- `input`：数学题 prompt
- `output`：模型生成的完整回答
- `score`：0.0（错误）或 1.0（正确）
- `step`（可选）：采集时的训练步数
- `output_sentences`（可选）：按 RL 句子边界切好的句子数组

说明：
- 当 `sentence_judge_backend=self` 且 `trainer.rollout_dump_sentence_texts=true` 时，rollout 会**直接复用 self-judge 阶段已经提取好的句子文本**写入 `output_sentences`。
- 默认不会在 dump 阶段额外重新切句；没有可复用句子时，该字段可为空或缺失。

如果没有 rollout 数据，先运行一次 RL 采样：
```bash
sentence_judge_enable=false enable_rollout_data_dump=true \
  bash test_sentencepo_v1-4.sh
```

### 4.2 路径 A：简单模式（直接蒸馏）

适用于快速实验，`build_judge_sft_data.py` 内部自动随机切分 train/val：

```bash
python scripts/sja/build_judge_sft_data.py \
  --input-glob "$HOME/autodl-tmp/.../rollout_debug/*.jsonl" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --teacher-model "qwen3-max-2026-01-23" \
  --out-train "$HOME/data/sja_distill/train.parquet" \
  --out-val "$HOME/data/sja_distill/val.parquet"
```

### 4.3 路径 B：预处理 + 蒸馏（推荐）

先用 `prepare_judge_curriculum_data.py` 做数据清洗和 **按题分组**的 train/val 切分（同一道题的所有 rollout 不会同时出现在 train 和 val 中，避免数据泄漏），再对已切分的文件分别蒸馏：

```bash
# 步骤 1: 清洗 + train/val 切分
python scripts/sja/prepare_judge_curriculum_data.py \
  --input-glob "$HOME/autodl-tmp/.../rollout_debug/*.jsonl" \
  --out-dir "$HOME/data/sja_curriculum/qwen3-4b-instruct" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --val-ratio 0.05

# 步骤 2: 分别蒸馏 train 和 val（--out-parquet = 单文件输出，不内部切分）
python scripts/sja/build_judge_sft_data.py \
  --input-glob "$HOME/data/sja_curriculum/qwen3-4b-instruct/clean_train.jsonl" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --teacher-model "qwen3-max-2026-01-23" \
  --num-workers 4 \
  --log-every 20 \
  --out-parquet "$HOME/data/sja_curriculum/qwen3-4b-instruct/distill_train.parquet"

python scripts/sja/build_judge_sft_data.py \
  --input-glob "$HOME/data/sja_curriculum/qwen3-4b-instruct/clean_val.jsonl" \
  --tokenizer "Qwen/Qwen3-4B-Instruct-2507" \
  --teacher-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --teacher-model "qwen3-max-2026-01-23" \
  --num-workers 4 \
  --log-every 20 \
  --out-parquet "$HOME/data/sja_curriculum/qwen3-4b-instruct/distill_val.parquet"
```

一键执行：`bash scripts/sja/run_prepare_judge_curriculum.sh`

### 4.4 清洗逻辑（prepare_judge_curriculum_data.py）

| 操作 | 说明 |
|------|------|
| 重复尾句裁剪 | 检测连续重复的句子块，保留少量尾句供 teacher 学习"重复=低质量"信号 |
| 连续相同句子压缩 | 连续 > `max_consecutive_same`(2) 的相同句子只保留前 N 个 |
| emoji 压缩 | 连续 ✅ 符号压缩为 3 个 |
| 长度控制 | `--max-sentences`(96), `--max-chars`(12000) |
| 预切句复用 | 若存在 `output_sentences`，优先复用，不再走 token 级切句 |
| 按题切分 | 相同 prompt 的所有 rollout 归入同一组，组间随机分配到 train/val |

输出文件：
- `clean_all.jsonl` — 全部清洗后的数据
- `clean_train.jsonl` / `clean_val.jsonl` — 按题分组切分
- `stats.json` — 统计信息（含 `observed_steps` 列表）

备注（`cleaning.kept_sentence_count`）：
- 未触发 `max_chars` 截断时，`kept_sentence_count` 与输出句子数一致。
- 触发 `max_chars` 截断时，为避免二次 token 级切句开销，`kept_sentence_count` 是按字符预算估算的近似值。

### 4.5 蒸馏后处理（build_judge_sft_data.py）

Teacher API 返回的 JSON 会经过标准化：
- **句子来源优先级**：优先使用输入行里的 `output_sentences`，若缺失再对 `output` 进行切句
- **bucket 吸附**：将 teacher 返回的 bucket 值吸附到最近的允许值（如 0.3 → 0.25）
- **confidence 裁剪**：clamp 到 [0, 1]
- **缺失句子补默认**：teacher 漏掉的句子用 `{bucket: 0.0, confidence: 0.0, reason: "missing_from_teacher"}` 填充
- **reason 截断**：最多 256 字符

运行稳定性与可观测性：
- **并发调用**：`--num-workers` 可开启多线程并发调用 Teacher API（`1` = 单线程）
- **实时进度输出**：`--log-every` 控制打印频率（含 processed/distilled/skipped/speed/eta）
- **进度文件**：`--progress-log-file` 指定进度日志；若不指定且使用 `--out-parquet`，默认写 `<out-parquet>.progress.log`
- **逐条落盘**：每成功蒸馏 1 条都会立刻 append 到记录文件
- **断点续跑**：如果记录文件已存在，会自动读取已有记录并续跑，避免中断后全量重来（并发模式下按 `row_uid` 去重）
- **原始返回日志**：`--raw-log-file` 可保存 Teacher 原始返回（含成功/失败事件），用于排查 `missing_from_teacher` 和 `skipped`

最终 parquet 只有两列：
- `prompt`：渲染后的完整 judge prompt（含原始题目、编号句子、规则说明）
- `response`：标准化后的严格 JSON（用于 SFT 训练的 target）

默认会在 `--out-parquet` 同目录生成两个辅助文件：
- `<out-parquet>.progress.log`：进度日志
- `<out-parquet>.records.jsonl`：逐条增量结果（可用于续跑）
- `<out-parquet>.teacher_raw.jsonl`：Teacher 原始返回日志（debug 用）

`<out-parquet>.teacher_raw.jsonl` 每行是一个事件：
- `status="ok"`：包含 `teacher_raw_content`（模型原始文本）和 `teacher_raw_json`（提取后的 JSON 对象）
- `status="error"`：包含失败错误信息（例如超时、JSON 提取失败）

### 4.6 课程化 Phase（可选）

`prepare` 输出的 JSONL 保留了 `step` 字段。蒸馏后如果需要按训练阶段构建课程，可以从 distill parquet 过滤：

```python
import pandas as pd

df = pd.read_parquet("distill_train.parquet")
# 假设你有 step 信息可以关联（通过 prompt 匹配等方式）
# 或者直接对全量数据做随机子集作为不同 phase 的训练数据
phase1 = df.sample(frac=0.25, random_state=42)
phase2 = df.drop(phase1.index)
```

目前 phase 切分不由脚本自动完成——先蒸馏全部数据，再按需求手动或脚本化切分，更灵活。

### 4.7 Teacher API 配置

支持任何 OpenAI 兼容 API。认证方式（按优先级）：
1. `--teacher-api-key` 命令行参数
2. `SJA_TEACHER_API_KEY` 环境变量
3. `DASHSCOPE_API_KEY` 环境变量
4. `OPENAI_API_KEY` 环境变量

Teacher 请求格式：
```json
{
  "model": "qwen3-max-2026-01-23",
  "messages": [
    {"role": "system", "content": "You are a strict JSON generator."},
    {"role": "user", "content": "<渲染后的 judge prompt>"}
  ],
  "temperature": 0.0,
  "max_tokens": 512
}
```

## 5. 第二步：Judge-SFT

使用 verl 内置的 `fsdp_sft_trainer` 对 4B 模型做标准 SFT 训练。

### 5.1 训练流程

```bash
bash scripts/sja/run_judge_sft_qwen3_4b.sh \
  TRAIN_FILE=$HOME/data/sja_distill/train.parquet \
  VAL_FILE=$HOME/data/sja_distill/val.parquet \
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
```

### 5.2 训练细节

| 配置 | 默认值 | 说明 |
|------|--------|------|
| 基座模型 | Qwen3-4B-Instruct-2507 | 可替换为其他 causal LM |
| 学习率 | 2e-6 | |
| Epochs | 1 | 蒸馏数据量小，通常 1 epoch 就够 |
| 最大序列长度 | 8192 | judge prompt + response JSON |
| 精度 | bf16 | |
| 并行 | FSDP (4 GPU) | |

### 5.3 损失函数

标准 **Causal Language Modeling Cross-Entropy Loss**：

```
Loss = CrossEntropy(model(prompt + response), response_tokens)
```

parquet 的 `prompt` 列作为输入前缀（不参与 loss 计算），`response` 列（严格 JSON）作为训练 target。`fsdp_sft_trainer` 通过 `data.prompt_key=prompt` 和 `data.response_key=response` 指定这两列。

### 5.4 训练目标

让模型学会：
1. **格式稳定性**：始终输出严格 JSON，不夹杂自然语言
2. **评分一致性**：bucket 值在允许范围内，confidence 有意义
3. **句子覆盖**：为每个编号句子都给出评分，不遗漏

SFT 后 `sentence_judge/parse_fail_rate` 指标应从 >30% 降到 <5%。

## 6. 第三步：RL 接入

### 6.1 两种模式

| 模式 | 配置 | 特点 |
|------|------|------|
| **self** | `sentence_judge_backend=self` | actor 模型兼任 judge，在 rollout 阶段直接生成 judge JSON。部署简单，但 judge 随 actor 权重漂移 |
| **split** | `sentence_judge_backend=callable` | 独立 judge 服务（OpenAI 兼容接口），与 actor 解耦。更稳定，需额外部署 |

### 6.2 Self 模式 RL 流程

```
每个训练 step:
  1. Rollout: actor 生成 N 个 response
  2. Self-Judge: 用同一个 actor 模型对每个 response 做 sentence-level 评分
     - 构建 judge prompt (prompt + numbered sentences + template)
     - 用 rollout engine 生成 JSON 输出
     - 解析 JSON → sentence advantages
  3. Advantage 融合:
     final_adv = grpo_adv + alpha * judge_token_adv
  4. PPO 更新: 用融合后的 advantages 更新 actor
```

```bash
bash scripts/sja/run_rl_sja_self.sh \
  MODEL_PATH=$HOME/autodl-tmp/models_v1-4-metrics-sft/your_judge_sft_ckpt
```

### 6.3 Split 模式 RL 流程

```bash
# 1. 启动独立 judge 服务
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model $HOME/autodl-tmp/models_v1-4-metrics-sft/your_judge_sft_ckpt \
  --served-model-name qwen3-4b-judge \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 2

# 2. 启动 RL
bash scripts/sja/run_rl_sja_split.sh \
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
  SJA_JUDGE_MODEL=qwen3-4b-judge \
  SJA_JUDGE_BASE_URL=http://127.0.0.1:8000/v1
```

### 6.4 Judge SFT 混合损失（持续进化打分能力）

在 RL 训练中同时加入 judge SFT cross-entropy loss，让打分能力随 RL 训练持续进化，无需调用 Teacher API。

**原理**：在每个 micro-batch 的 backward 之前，从预蒸馏的 judge SFT 数据中采样一个 micro-batch，计算 causal LM cross-entropy loss，加权后与 policy loss 合并：

```
total_loss = policy_loss + lambda_judge * judge_sft_loss
```

**实现细节**（`dp_actor.py`）：
1. Worker 初始化时加载 parquet，pre-tokenize 全部样本（`judge_sft_data.py:prepare_judge_sft_data`）
2. 创建 `JudgeSFTIterator`，在数据用完时自动 shuffle 并循环
3. 每个 micro-batch 的 backward 前：
   - 从 iterator 取一个 judge SFT micro-batch
   - 前向传播得到 logits
   - 计算 causal LM cross-entropy（prompt tokens 用 label=-100 mask 掉）
   - `loss += lambda * judge_sft_loss`，然后统一 backward

**使用方式**：
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
| `judge_sft_data_path` | "" | 蒸馏后的 parquet 路径（需含 prompt, response 两列） |
| `judge_sft_lambda` | 0.1 | judge SFT loss 权重。过大会干扰 policy 学习，过小则进化不明显 |
| `judge_sft_micro_batch_size` | 2 | 每个 gradient step 采样的 judge 样本数 |
| `judge_sft_max_seq_len` | 2048 | 截断长度。judge prompt 一般较长，建议设为 2048+ |

## 7. 关键配置参数

### SJA Advantage

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sentence_judge_enable` | true | 总开关 |
| `sentence_judge_alpha` | 0.2 | judge advantage 融合权重 α |
| `sentence_judge_backend` | self | self / callable / dummy |
| `sentence_judge_fn` | null | callable 模式的 judge 函数导入路径 |
| `buckets` | [0.75, 0.25, 0.0, -0.25, -0.75] | 离散 advantage 桶 |
| `confidence_floor` | 0.2 | 低于此阈值的信号置零 |
| `normalize` | zscore | zscore / none |
| `sentence_judge_max_sentences` | 0 | 每条样本最多处理句子数（0=不限） |
| `sentence_judge_max_chars` | 0 | 每句最大字符数（0=不限） |
| `sentence_judge_max_tokens` | 2048 | self-judge 生成 JSON 的最大 token 数 |
| `sentence_judge_every_n_steps` | 1 | 每 N 步执行一次 judge（1=每步） |

### SentencePO Loss

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sentencepo_eps_base` | 0.01 | 句子级 clip 基础 epsilon |
| `sentencepo_lambda_ppl` | 0 | PPL 自适应 clip 权重 |
| `sentencepo_lambda_len` | 0 | 长度自适应 clip 权重 |
| `sentencepo_cmin` / `cmax` | 0.5 / 1.5 | clip scale 范围 |
| `sentencepo_min_sent_tokens` | 6 | 短句合并阈值 |

### Rollout Dump（可选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.rollout_dump_sentence_texts` | true（在 `test_sentencepo_v1-4.sh` 中） | 在 rollout jsonl 中输出 `output_sentences` |
| `trainer.rollout_dump_max_sentences` | 0 | dump 时每条样本最多保留多少句（0=不限） |
| `trainer.rollout_dump_max_chars` | 0 | dump 时每句最大字符数（0=不限） |

说明：`output_sentences` 采用“优先复用”策略，默认不会为了 dump 再额外切句。

## 8. 关键监控指标

| 指标 | 含义 | 参考范围 |
|------|------|----------|
| `sentence_judge/parse_fail_rate` | JSON 解析失败率 | SFT 后应 < 5% |
| `sentence_judge/conf_mean` | 平均 confidence | 0.5-0.8 |
| `sentence_judge/zero_frac` | advantage 为 0 的比例 | 过高说明信号太弱 |
| `sentence_judge/consistency_fix_rate` | 一致性修正比例 | 过高说明 judge 质量差 |
| `sentence_judge/adv_mean` | judge advantage 均值 | |
| `sentence_judge/adv_std` | judge advantage 标准差 | |
| `sentencepo/sent_clip_fraction` | 句子被 clip 的比例 | |
| `sentencepo/delta_sent/{mean,std}` | 句子级 log-ratio 统计 | |
| `judge_sft/loss` | judge SFT cross-entropy loss | 应随训练下降 |
| `judge_sft/lambda` | 当前 judge SFT loss 权重 | |

## 9. 脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/sja/prepare_judge_curriculum_data.py` | rollout 清洗 + 按题 train/val 切分 |
| `scripts/sja/build_judge_sft_data.py` | 切句 + 渲染 judge prompt + Teacher API 打分 → SFT parquet |
| `scripts/sja/run_prepare_judge_curriculum.sh` | 一键完成预处理 + Teacher 蒸馏 |
| `scripts/sja/run_judge_sft_qwen3_4b.sh` | Judge-SFT 训练 |
| `scripts/sja/run_rl_sja_self.sh` | self 模式 RL |
| `scripts/sja/run_rl_sja_split.sh` | split 模式 RL |

## 10. `build_judge_sft_data.py` 输出模式

| 参数 | 场景 | 说明 |
|------|------|------|
| `--out-train` + `--out-val` | 路径 A（简单模式） | 内部按 `--val-ratio` 随机切分 |
| `--out-parquet` | 路径 B（搭配 `prepare`） | 单文件输出，输入已由 `prepare` 分好 |

常用可观测性/容错参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log-every` | 20 | 每 N 条输出一次进度（0=关闭周期进度） |
| `--progress-log-file` | `""` | 进度日志路径；为空时自动使用 `<out-parquet>.progress.log` |
| `--records-log-file` | `""` | 增量结果 jsonl；为空时自动使用 `<out-parquet>.records.jsonl`，支持续跑 |
| `--raw-log-file` | `""` | Teacher 原始返回日志；为空时自动使用 `<out-parquet>.teacher_raw.jsonl` |
| `--num-workers` | 4 | Teacher API 并发线程数 |
