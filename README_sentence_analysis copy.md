# SentencePO / GSPO 诊断与可视化说明

本文档说明新增的诊断指标、输出文件与可视化脚本。

## 1. 新增指标概览

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

### 1.2 正确/错误 response 对比
- 平均 response 长度、句子数、最大句长
- 句子级 PPL/熵的均值与方差
- 指标前缀：`gspo/*` 或 `grpo/*`

### 1.3 句子级统计（SentencePO / GSPO）
- 句子长度分布、句子 PPL/熵
- 句内最大 `|log ratio|` 与平均 log-ratio
- 句长占比（句子长度 / response 长度）
- GSPO 句子指标前缀：`gspo_sentence/*`
- SentencePO 句子指标前缀：`sentencepo/*`

## 2. 句子级样本输出（可视化）

训练/验证可输出 JSONL：每个 response 一条，包含：
- response 是否被 clip（GSPO/GRPO）
- 句子数量、每句 PPL/熵/长度/Δlogπ/ KL
- top-k 句子位置（PPL/熵/长度/Δlogπ/KL）
- 若 SentencePO：每句是否被 clip

输出目录结构（按 prompt uid 分组）：
```
<analysis_dir>/
  <uid_1>/
    <step>.jsonl
  <uid_2>/
    <step>.jsonl
```

## 3. 配置项

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

## 4. 可视化脚本

脚本路径：
- scripts/analysis/view_sentence_analysis.py

示例：
```bash
python scripts/analysis/view_sentence_analysis.py \
  --input /path/to/sentence_analysis \
  --scatter /tmp/m_vs_delta.png \
  --bins 64,128,256,512
```

正确/错误 response 对比图示例：
```bash
python scripts/analysis/view_sentence_analysis.py \
  --input /root/autodl-tmp/models_v1-1-metrics/sentencepo_math_qwen3_4b_ep3_epsbase0.1_Lppo1.0_Llen0_cmin0.5_cmax1.5/sentence_analysis \
  --compare-out /root/autodl-tmp/models_v1-1-metrics/sentencepo_math_qwen3_4b_ep3_epsbase0.1_Lppo1.0_Llen0_cmin0.5_cmax1.5/correct_vs_wrong.png \
  --compare-metrics response_len,sentence_count,response_ppl,response_entropy,\
sentence_stats.ppl_mean,sentence_stats.ppl_var,sentence_stats.entropy_mean,sentence_stats.entropy_var
```

功能：
- 输出正确/错误 response 的摘要统计
- 生成 `m_{i,k}` vs `|Δ_{i,k}|` 散点图
- 按 m 分桶统计 reward=1 比例
- 生成正确/错误 response 的对比柱状图（支持自定义指标列表）

## 5. 运行脚本参数说明（以 test_sentencepo_v1-1.sh 为例）

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

## 5. 验证阶段新增统计

- response 长度分桶准确率：
  - `val-aux/resp_len_bucket/*`
- prompt 长度分层后的 response 长度准确率：
  - `val-aux/resp_len_by_prompt_len/*`
- 同 prompt 下：长度与正确性相关性（Pearson）
  - `val-aux/len_correct_corr/*`

## 6. 注意事项
- CPR/CNR 与 clip 分层统计只在包含奖励信息的训练阶段输出。
- 句子切分规则仍使用标点/换行的启发式规则（`build_sentence_ids_from_responses`）。
