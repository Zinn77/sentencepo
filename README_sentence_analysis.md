# SentencePO / GSPO 诊断与可视化说明

本文档说明新增的诊断指标、输出文件与可视化脚本。

## 1. 指标概览

### 1.1 Clip 相关（GSPO / GRPO）
- CPR/CNR：
  - `gspo/cpr`, `gspo/cnr`（GSPO）
  - `grpo/cpr`, `grpo/cnr`（GRPO/vanilla）
- Clip 触发原因：
  - `gspo/clip_reason_low`, `gspo/clip_reason_high`
  - `grpo/clip_reason_low`, `grpo/clip_reason_high`
- Clip 分层统计：
  - 按 response 长度、句子数、最大句长分桶：
    - `gspo/clip_by_resp_len/*`, `gspo/clip_by_sent_count/*`, `gspo/clip_by_max_sent_len/*`
    - `grpo/clip_by_resp_len/*`, `grpo/clip_by_sent_count/*`, `grpo/clip_by_max_sent_len/*`

### 1.2 正确/错误 response 对比（仅 GRPO/GSPO）
- 平均 response 长度、句子数、最大句长
- 句子级 PPL/熵的均值与方差
- 指标前缀：`gspo/*` 或 `grpo/*`
> 说明：SentencePO 的“正确/错误 response 对比”不在训练指标里输出，但在 sentence_analysis JSONL 中可做离线对比。

### 1.3 句子级统计
- 句子长度分布、句子 PPL/熵
- 句内最大 `|log ratio|` 与平均 log-ratio
- 句长占比（句子长度 / response 长度）
- GSPO 句子指标前缀：`gspo_sentence/*`
- SentencePO 句子指标前缀：`sentencepo/*`
> 说明：GRPO 不输出句子级“标量指标”，但在开启 `trainer.sentence_analysis.enable` 时会输出句子级 JSONL。

## 2. 指标详解（含义说明）

### 2.1 训练阶段标量指标（TensorBoard）

#### GRPO / GSPO 通用
- `grpo/clipfrac_resp`：response 被 clip 的比例（GRPO/vanilla）
- `gspo/clipfrac_seq`：response 被 clip 的比例（GSPO，基于序列级 ratio）
- `grpo/clip_reason_low|high` / `gspo/clip_reason_low|high`：clip 触发原因占比（低/高阈值）
- `grpo/cpr` / `gspo/cpr`：被 clip 的 response 中，reward=1 的比例
- `grpo/cnr` / `gspo/cnr`：被 clip 的 response 中，reward=0 的比例
- `grpo/clip_by_*/*` / `gspo/clip_by_*/*`：按分桶的 clip 比例与 CPR/CNR（
  `resp_len`/`sent_count`/`max_sent_len` 三类分桶）

#### 正确/错误 response 对比（仅 GRPO/GSPO）
- `*/resp_len_mean_{correct|wrong}`：平均 response 长度
- `*/resp_ppl_mean_{correct|wrong}`：平均 response PPL（基于 old_log_probs）
- `*/resp_entropy_mean_{correct|wrong}`：平均 response 熵
- `*/sent_count_mean_{correct|wrong}`：平均句子数
- `*/max_sent_len_mean_{correct|wrong}`：平均最大句长
- `*/sent_ppl_mean_{correct|wrong}`、`*/sent_ppl_var_{correct|wrong}`：句子 PPL 的均值/方差
- `*/sent_ent_mean_{correct|wrong}`、`*/sent_ent_var_{correct|wrong}`：句子熵的均值/方差

#### 句子级统计（SentencePO / GSPO 标量）
- `sentencepo/count/*` / `gspo_sentence/count/*`：每条 response 的句子数统计
- `sentencepo/len/*` / `gspo_sentence/len/*`：句子长度统计
- `sentencepo/response_len/*` / `gspo_sentence/response_len/*`：response 长度统计
- `sentencepo/ppl/*` / `gspo_sentence/ppl/*`：句子 PPL 统计
- `sentencepo/entropy/*` / `gspo_sentence/entropy/*`：句子熵统计
- `sentencepo/delta_sent/*` / `gspo_sentence/delta_sent/*`：句子平均 log‑ratio（Δ）统计
- `sentencepo/kl_sent/*` / `gspo_sentence/kl_sent/*`：句子 KL 统计（= -Δ）
- `sentencepo/ratio_max_abs_log/*` / `gspo_sentence/ratio_max_abs_log/*`：
  句内最大 |log‑ratio|（m）统计
- `sentencepo/len_share/*` / `gspo_sentence/len_share/*`：句长占比统计
- `sentencepo/ratio_std_across_sent/*` / `gspo_sentence/ratio_std_across_sent/*`：
  同一 response 内各句 ratio 的标准差
- `sentencepo/ratio_range_across_sent/*` / `gspo_sentence/ratio_range_across_sent/*`：
  同一 response 内各句 ratio 的范围

### 2.2 验证阶段指标（TensorBoard）
- `val-aux/resp_len_bucket/*`：response 长度分桶准确率
- `val-aux/resp_len_by_prompt_len/*`：按 prompt 长度分层的 response 长度准确率
- `val-aux/len_correct_corr/*`：同一 prompt 下长度与正确性 Pearson 相关

### 2.3 句子级 JSONL 字段（离线可视化）
- response 级：`reward`, `response_len`, `response_ppl`, `response_entropy`,
  `sentence_count`, `seq_ratio`, `response_clipped`, `response_sentence_clipped`,
  `sentence_stats.*`
- sentence 级（`sentences` 数组）：`token_count`, `ppl`, `entropy`,
  `mean_log_ratio`（Δ）, `sum_log_ratio`, `max_abs_log_ratio`（m）, `kl_old`, `kl_ref`,
  `clipped`, `clip_lower`, `clip_upper`
- top‑k 位置：`topk.ppl`, `topk.entropy`, `topk.length`, `topk.delta_sum`, `topk.kl_old`

## 3. 句子级样本输出（可视化）

训练/验证可输出 JSONL：每个 response 一条，包含：
- response 是否被 clip（GSPO/GRPO）
- 句子数量、每句 PPL/熵/长度/Δlogπ/ KL
- top-k 句子位置（PPL/熵/长度/Δlogπ/KL）
- 若 SentencePO：每句是否被 clip

输出目录结构（按 prompt uid 分组，区分 train/val）：
```
<analysis_dir>/
  train/
    <uid_1>/
      <step>.jsonl
    <uid_2>/
      <step>.jsonl
  val/
    <uid_1>/
      <step>.jsonl
    <uid_2>/
      <step>.jsonl
```

## 4. 配置项

在配置中添加：
```yaml
trainer:
  sentence_analysis:
    enable: true
    max_samples: 8
    top_k: 3
    group_by_uid: true
    dir: /path/to/sentence_analysis   # 可选

  response_len_bins: [64, 128, 256, 512]
  prompt_len_bins: [64, 128, 256, 512]
```

推荐分桶（max_response_length=4096）：
- response_len_bins: [128, 256, 512, 1024, 2048, 4096]
- prompt_len_bins: [64, 128, 256, 512, 1024]
- analysis_bins.sentence_count_bins: [4, 8, 16, 32, 64]
- analysis_bins.max_sentence_len_bins: [32, 64, 128, 256, 512]

说明：response/prompt 分桶尽量覆盖主要长度分位数；句子数与最大句长分桶建议更稠密，便于观察短句/长句差异。

可选：在 actor 的 `policy_loss` 中配置分桶范围：
```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      analysis_bins:
        response_len_bins: [64, 128, 256, 512]
        sentence_count_bins: [4, 8, 16, 32]
        max_sentence_len_bins: [32, 64, 128, 256]
```
说明：需要 `analysis_bins` 已被 `PolicyLossConfig` 支持（当前已支持）。

## 5. 可视化脚本

脚本路径：
- scripts/analysis/view_sentence_analysis.py
- scripts/analysis/view.sh

示例：
```bash
python scripts/analysis/view_sentence_analysis.py \
  --input /path/to/sentence_analysis \
  --scatter /tmp/m_vs_delta.png \
  --bins 1,2,3,4,5,6,7,8,15,32,64 \
  --xlim 0,55 \
  --ylim 0,3.5
```

正确/错误 response 对比图示例：
```bash
python scripts/analysis/view_sentence_analysis.py \
  --input /root/autodl-tmp/models_v1-1-metrics/sentencepo_math_qwen3_4b_ep3_epsbase0.1_Lppo1.0_Llen0_cmin0.5_cmax1.5/sentence_analysis \
  --compare-out /root/autodl-tmp/models_v1-1-metrics/sentencepo_math_qwen3_4b_ep3_epsbase0.1_Lppo1.0_Llen0_cmin0.5_cmax1.5/correct_vs_wrong.png \
  --compare-metrics response_len,sentence_count,response_ppl,response_entropy,\
sentence_stats.ppl_mean,sentence_stats.ppl_var,sentence_stats.entropy_mean,sentence_stats.entropy_var
```

response 级散点图示例：
```bash
python scripts/analysis/view_sentence_analysis.py \
  --input /path/to/sentence_analysis \
  --response-scatter /tmp/resp_m_vs_delta.png \
  --response-bins 1,2,3,4,5,6,7,8,15,32,64 \
  --response-xlim 0,55 \
  --response-ylim 0,3.5
```

可视化能力（基于 JSONL 字段）：
- 正确/错误 response 摘要统计（`reward` + 各 response/句子统计字段）
- 句子级散点图：`max_abs_log_ratio` (m) vs `|mean_log_ratio|` (|Δ|)
- m 分桶的 reward=1 比例
- response 级散点图：response 内 m/Δ 的聚合（max/mean）
- response 级 m 分桶的 reward=1 比例
- 正确/错误对比柱状图（`--compare-metrics` 支持任意 JSONL 字段，含嵌套字段）

JSONL 主要字段列表：
- response 级：`reward`, `response_len`, `response_ppl`, `response_entropy`, `sentence_count`, `seq_ratio`,
  `response_clipped`, `response_sentence_clipped`, `sentence_stats.*`, `response_m`, `response_delta`
- sentence 级（`sentences` 数组）：`token_count`, `ppl`, `entropy`, `mean_log_ratio`, `sum_log_ratio`,
  `max_abs_log_ratio`, `kl_old`, `kl_ref`, `clipped`, `clip_lower`, `clip_upper`
- top-k 位置：`topk.ppl`, `topk.entropy`, `topk.length`, `topk.delta_sum`, `topk.kl_old`

## 6. 运行脚本参数说明（以 test_sentencepo_v1-1.sh 为例）

脚本路径：
- test_sentencepo_v1-1.sh

常用环境变量与含义：
- DS：数据集名字（影响输出目录命名）
- EPOCHS：训练轮数
- MODEL_PATH：模型路径
- MODEL_NAME：模型名称（影响输出目录命名）
- lr：学习率
- qwen3_enable_thinking：是否启用思考模式
- max_response_length：最大生成长度（响应长度上限）
- max_num_batched_tokens：最大批内 token 数（>= max_prompt_length + max_response_length）
- micro_batch_size：每 GPU 的 micro-batch
- clip_ratio_low / clip_ratio_high：PPO clip 的上下界

SentencePO 超参数：
- sentencepo_min_sent_tokens：句子最少 token 数，短句会被合并
- sentencepo_eps_base：自适应 clip 的基准 eps
- sentencepo_lambda_ppl：PPL 调整权重
- sentencepo_lambda_len：长度调整权重
- sentencepo_cmin / sentencepo_cmax：自适应 clip 缩放下/上界
- sentencepo_stats_eps：统计稳定项
- sentencepo_metrics_level：full/basic/off（basic 不记录分句相关指标）

诊断与可视化：
- enable_sentence_analysis：是否输出句子级 JSONL
- sentence_analysis_max_samples：每步最多保存多少条 response
- sentence_analysis_top_k：每个指标的 Top-K 句子
- sentence_analysis_group_by_uid：是否按 prompt uid 分目录保存
- sentence_analysis_dir：JSONL 输出目录

分桶参数：
- response_len_bins / prompt_len_bins：验证阶段长度分桶
- analysis_response_len_bins：训练阶段 clip 分层统计（按 response 长度）
- analysis_sentence_count_bins：训练阶段 clip 分层统计（按句子数）
- analysis_max_sentence_len_bins：训练阶段 clip 分层统计（按最大句长）

## 7. 验证阶段新增统计

- response 长度分桶准确率：
  - `val-aux/resp_len_bucket/*`
- prompt 长度分层后的 response 长度准确率：
  - `val-aux/resp_len_by_prompt_len/*`
- 同 prompt 下：长度与正确性相关性（Pearson）
  - `val-aux/len_correct_corr/*`

## 8. 注意事项
- CPR/CNR 与 clip 分层统计只在包含奖励信息的训练阶段输出。
- 句子切分规则仍使用标点/换行的启发式规则（`build_sentence_ids_from_responses`）。
