# Sentence Judge Advantage (SJA) - Summary

本文档详细说明新增的 Sentence Judge Advantage (SJA) 方法、优势计算流程、prompt 结构、接入位置与使用方式。

## 1. 方法概述

SJA 在原有 SentencePO 优势融合框架上新增一个“外部判别器”来源的句子级优势项。整体流程为：

- 对每个 response 做句子切分并对齐 `sentence_ids`。
- 对每个句子构造 judge prompt，获取离散档位（bucket）与置信度（confidence）。
- 将离散档位映射为句子优势，按置信度加权，并可选归一化（z-score）。
- 通过 `sentence_ids` 对齐到 token 级别，再与 base advantage 融合。

融合形式：

$$A_{final} = A_{base} + \alpha_{judge} \cdot A_{judge}$$

支持一键开关与可消融配置。

## 2. 优势计算细节

SJA 的优势计算分为句子级和 token 级两个阶段：

1) 句子切分与对齐
- 使用 `sentence_ids` 从 token 序列中恢复句子顺序，生成每句文本（必要时用 tokenizer 解码）。
- 可选 `max_sentences` 与 `max_chars` 对句子数和长度进行截断。

2) 构造 judge prompt 并请求判别
- 根据 `buckets` 生成允许的离散档位。
- 将 prompt、response、以及句子列表写入 judge prompt，要求返回严格 JSON。
- `self` backend 下会在 trainer 内部先生成 judge prompt，再通过 actor rollout 生成 JSON。

3) 解析 judge 输出
- 逐句读取 `bucket` 与 `confidence`。
- 若 `bucket` 不在允许集合内，按最近邻桶修正。
- 若 judge 返回非 JSON/脏文本（如空串、乱码、`!!!!`），会触发 fallback：该样本句子优势置 0，并继续训练（不因解析失败中断）。

4) 置信度加权
- 若 `use_confidence_weight=true`，则 $A_i = bucket_i \cdot confidence_i$。
- 若 `confidence_i < confidence_floor`，该句优势直接置 0。

5) 归一化
- 当 `normalize=zscore` 时，对句子优势执行 z-score 标准化。

6) 一致性约束
- 用 `sequence_reward` 与 `correctness_threshold` 判定整体正确性。
- 若平均优势与整体正确性符号不一致，执行平移与可能的整体翻转，保证方向一致。

7) 句子优势对齐到 token
- 将句子级优势按 `sentence_ids` 映射到 token 级，形成 $A_{judge}$。

该逻辑在 [verl/trainer/ppo/sentence_judge_adv.py](verl/trainer/ppo/sentence_judge_adv.py) 中实现。

## 3. Prompt 结构与输出格式

### 3.1 Prompt 模板

SJA 构造的 prompt 由以下部分组成：

- 角色与要求：说明需要对每句打分，并输出 JSON。
- 允许桶：`Allowed buckets: [...]`。
- `overall_correct`：由 reward 与阈值决定。
- 原始 `PROMPT` 与 `RESPONSE`（句子编号）。

简化模板如下（与实现保持一致）：

```text
You are a sentence-level judge. Score each sentence with a discrete advantage bucket and a confidence in [0,1].
Allowed buckets: [<bucket_1>, <bucket_2>, ...].
overall_correct = <true|false>

PROMPT:
<original prompt>

RESPONSE (sentence numbered):
S1: <sentence 1>
S2: <sentence 2>
...

Output strict JSON only:
{"sentences":[{"id":1,"bucket":0.25,"confidence":0.8,"reason":"..."}]}
```

### 3.2 输出 JSON 规范

JSON 结构必须为：

```json
{
	"sentences": [
		{"id": 1, "bucket": 0.25, "confidence": 0.8, "reason": "..."}
	]
}
```

- `id` 从 1 开始并与句子编号一致。
- `bucket` 为离散档位值。
- `confidence` 在 [0, 1]。
- `reason` 仅用于调试，不参与计算。

## 4. 新增与修改的文件

新增：
- [verl/utils/judge_client.py](verl/utils/judge_client.py) 统一 judge 访问接口（dummy / callable）。
- [verl/trainer/ppo/sentence_judge_adv.py](verl/trainer/ppo/sentence_judge_adv.py) SJA 核心逻辑。
- [tests/trainer/ppo/test_sentence_judge_adv.py](tests/trainer/ppo/test_sentence_judge_adv.py) 对齐与 dummy 后端测试。

修改：
- [verl/trainer/config/algorithm.py](verl/trainer/config/algorithm.py) 新增 `SentenceJudgeAdvConfig` 并加入 `AlgoConfig`。
- [verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml) 增加默认配置块。
- [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) 接入 SJA 融合逻辑与指标记录。

## 5. 核心配置项

`algorithm.sentence_judge_adv`：

- `enable`: 是否启用 SJA。
- `alpha`: 融合权重。
- `judge_backend`: `dummy` / `callable` / `self`。
- `judge_fn`: 当 `callable` 时的函数路径，例如 `your.module:judge_fn`。
- `correctness_threshold`: 判别正确性阈值。
- `use_confidence_weight`: 是否按置信度加权。
- `confidence_floor`: 置信度下限。
- `normalize`: 归一化方式（如 `zscore`）。
- `max_sentences`: 每条样本最多处理句子数（0 表示不限制）。
- `max_chars`: 每条样本文本最大字符数（0 表示不限制）。
- `judge_max_tokens`: self-judge 生成 JSON 的最大 token 数。
- `truncate_prompt`: `self` backend 下超预算时是否截断 judge prompt（默认 `false`）。
- `rate_limit_qps`: judge 调用限速（0 表示不限制）。
- `debug_prompt`: 是否输出 prompt 调试信息。

### 5.1 长度预算（self-judge）

`self` backend 的一次 judge 调用包含：

- 原始任务 prompt；
- 分句编号后的 response 文本（`S1: ...`）；
- 固定模板 + JSON 输出约束。

因此需满足：

$$
	\texttt{rollout.max\_model\_len} \ge \texttt{judge\_prompt\_tokens} + \texttt{judge\_max\_tokens}
$$

实践建议：

- 优先增大 `actor_rollout_ref.rollout.max_model_len`；
- 保证 `actor_rollout_ref.rollout.max_num_batched_tokens >= max_model_len`；
- 配合 `max_sentences` / `max_chars` 抑制 prompt 膨胀；
- 如需不中断可设 `truncate_prompt=true`（代价是 prompt 会被截断）。

当 `truncate_prompt=false` 且超预算时，会报错并打印：
`max_prompt_tokens / allowed_prompt_tokens / rollout.max_model_len / judge_max_tokens`。

## 6. 运行方式

示例脚本：
- [test_sentencepo_v1-4.sh](test_sentencepo_v1-4.sh)

默认说明：

- 框架配置默认 `judge_backend=dummy`；
- 示例脚本 [test_sentencepo_v1-4.sh](test_sentencepo_v1-4.sh) 默认 `sentence_judge_backend=self`。

若需要接真实 judge：

```bash
sentence_judge_backend=callable
sentence_judge_fn=your.module:your_callable
```

### 6.1 使用 actor 权重做 self-judge（无额外 GPU）

设置：

```bash
sentence_judge_backend=self
sentence_judge_every_n_steps=1
```

该模式会在训练过程中用当前 actor 权重生成 judge JSON（每 step 可配置）。

运行：

```bash
bash test_sentencepo_v1-4.sh
```

### 6.2 示例脚本中的预算参数

脚本里与 self-judge 长度相关的变量：

- `sentence_judge_prompt_overhead_tokens`：估算 self-judge prompt 额外开销；
- `rollout_max_model_len`：rollout 单请求上下文上限；
- `max_num_batched_tokens`：vLLM 批处理 token 预算（吞吐/显存）；
- `sentence_judge_truncate_prompt`：超预算时是否截断 judge prompt。

脚本会自动保证 `max_num_batched_tokens >= rollout_max_model_len`。

推荐起点（self backend）：

```bash
sentence_judge_backend=self \
sentence_judge_prompt_overhead_tokens=2048 \
sentence_judge_truncate_prompt=false \
bash test_sentencepo_v1-4.sh
```

### 6.3 self-judge 输出落盘开关

`test_sentencepo_v1-4.sh` 两个相关变量：

- `enable_rollout_data_dump`：是否开启 rollout 样本落盘（默认 `false`）；
- `rollout_data_dir`：落盘目录（默认 `$TOT_DIR/rollout_debug`）。

开启示例：

```bash
enable_rollout_data_dump=true \
rollout_data_dir=$HOME/autodl-tmp/models_v1-4-metrics-2/your_exp/rollout_debug \
bash test_sentencepo_v1-4.sh
```

开启后，每个 step 的 JSONL 会包含：

- `self_judge_output`
- `self_judge_prompt`

## 7. 指标与日志

新增指标前缀：`sentence_judge/`，包括但不限于：

- `adv_mean` / `adv_std`
- `conf_mean`
- `pos_frac` / `neg_frac` / `zero_frac`
- `consistency_fix_rate`
- `parse_fail_rate`（judge 解析失败比例，失败样本回退为零优势）

指标会被 trainer 记录到日志与 tensorboard。

### 7.1 在哪里查看 self-judge 生成结果

当前实现中，self-judge 的原始文本会先存入训练 batch：

- `batch.non_tensor_batch["judge_outputs"]`：self-judge 生成的 JSON 文本（或脏文本）；
- `batch.non_tensor_batch["judge_prompts"]`：对应发送给 self-judge 的完整 prompt。

若希望落盘查看，可通过以下任一方式开启：

```bash
trainer.rollout_data_dir=/your/path
```

或在脚本里设置：

```bash
enable_rollout_data_dump=true
```

开启后，每个 step 会写入 JSONL（见 `ray_trainer.py` 的 rollout dump 逻辑），其中会新增字段：

- `self_judge_output`
- `self_judge_prompt`

可直接用 `jq`/`grep` 检查 judge 是否输出了合法 JSON。

## 8. 设计要点

- 最小侵入：仅在优势融合入口增加一个可选分支。
- 可消融：开关可完全关闭，不影响原训练流程。
- 严格对齐：基于 `sentence_ids` 映射到 token 级优势。
- 可扩展：judge client 支持 dummy/可调用后端。
- 兼容输出差异：当 rollout 结果不含 `response_mask` 时，`self` judge 路径会从 `attention_mask` 尾段恢复 response mask（与 vLLM 输出语义一致）。

## 9. 小贴士

- 初次实验建议 `alpha` 取小值（如 0.01-0.05）。
- 若 judge 输出噪声较大，可启用 `normalize` 与 `confidence_floor`。
- 若有外部服务调用限制，请设置 `rate_limit_qps`。
