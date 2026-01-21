# SentencePO 中间指标说明（TensorBoard）

本文档说明 SentencePO 相关的中间指标（TensorBoard）及其含义，并给出验证性实验与案例分析建议。

## 开关与性能控制

指标默认关闭，开启后会产生一定的额外计算开销（主要是句子向量相似度）。建议在小规模或抽样阶段开启。

配置位置：algorithm.sentence_adv
- `metrics_enable`: 是否记录 SentencePO 语义指标（默认 false）
- `metrics_max_sentences`: 每个问题组内用于相似度统计的最多句子数（默认 128）
- `metrics_max_pairs`: 每个问题组内用于相似度统计的最多成对相似度数（默认 4096）
- `metrics_pos_bins`: 句子位置分桶数量（默认 4）
- `metrics_divergence_threshold`: 分歧点检测阈值（默认 0.1）

## 指标命名规则

所有 SentencePO 指标统一在 `sentencepo/` 命名空间下。

### 现有指标（SentencePO 基础统计）
- `sentencepo/valid_ratio`: 句子 id 的有效比例（response_mask & sentence_ids >= 0）
- `sentencepo/count/*`: 每条 response 的句子数统计（mean/max/min/std）
- `sentencepo/len/*`: 句子长度统计（mean/max/min/std）
- `sentencepo/ratio/*`: 句子级 importance ratio 统计（mean/max/min/std）
- `sentencepo/ratio_std_across_sent/*`: 单条 response 内不同句子 ratio 的标准差统计
- `sentencepo/ratio_range_across_sent/*`: 单条 response 内不同句子 ratio 的范围统计
- `sentencepo/entropy/*`: 句子级熵统计（mean/max/min/std）
- `sentencepo/ratio_hist`, `sentencepo/entropy_hist`: 直方图（若开启直方图采样）

### 新增指标（SentencePO 语义相似度）
以下指标都在 `sentencepo/sentence_sim/` 下：
- `correct_correct_mean`: 正确回答之间的句子相似度均值
- `correct_correct_p90`: 正确回答之间的句子相似度 90 分位
- `wrong_wrong_mean`: 错误回答之间的句子相似度均值
- `wrong_wrong_p90`: 错误回答之间的句子相似度 90 分位
- `correct_wrong_mean`: 正确-错误回答之间的句子相似度均值
- `correct_wrong_p90`: 正确-错误回答之间的句子相似度 90 分位
- `ratio_cc_vs_cw`: (正确-正确均值) - (正确-错误均值)，用于衡量正确答案内部一致性与错对差异
- `ratio_ww_vs_cw`: (错误-错误均值) - (正确-错误均值)

说明：相似度基于句子向量 cosine，相似度统计在问题 group 内进行，并按 `metrics_max_sentences` / `metrics_max_pairs` 做采样限制。

### 新增指标（SentencePO 句子优势）
以下指标都在 `sentencepo/sentence_adv/` 下：
- `mean_correct`, `std_correct`, `p50_correct`, `p90_correct`: 正确回答中的句子优势分布
- `mean_wrong`, `std_wrong`, `p50_wrong`, `p90_wrong`: 错误回答中的句子优势分布
- `pos_bin_k_correct`: 第 k 个位置分桶的正确句子优势均值
- `pos_bin_k_wrong`: 第 k 个位置分桶的错误句子优势均值
- `pos_bin_k_diff`: 第 k 个位置分桶的正确-错误差值
- `alpha_term_mean`: 融合项 $\alpha \cdot A_{sentence}$ 的均值
- `grpo_term_mean`: 融合后反推的 GRPO 项均值
- `final_adv_mean`: 最终优势均值

说明：位置分桶基于句子相对位置（句子序号 / 最后句子序号），分桶数由 `metrics_pos_bins` 控制。

### 新增指标（分歧点 Divergence）
以下指标都在 `sentencepo/divergence/` 下：
- `first_pos_mean`, `first_pos_p50`, `first_pos_p90`: 分歧点位置的统计（归一化到 $[0,1]$）
- `strength_mean`, `strength_p90`: 分歧强度（按位置分桶的正确-错误差值绝对值）

## 验证性实验建议
1) 开关对照
- 配置 `adv_estimator=grpo` 与 `adv_estimator=grpo_sentencepo`，比较相似度指标是否能拉开“正确/错误”与“正确/错误交叉”。

2) 超参敏感性
- 扫 `sentence_adv.temperature` 与 `sentence_adv.alpha`，观察 `sentencepo/sentence_sim/*` 的变化是否符合预期（温度越高，分离度通常越弱）。

3) 轻量因果近似（可选）
- 对高相似度、高优势句子做遮盖/截断，比较最终 reward 变化。

## 案例分析建议
1) 同题多样本
- 取同一问题 group 的正确/错误回答，观察 `correct_correct`、`wrong_wrong`、`correct_wrong` 的相对关系。

2) 高相似度句子聚类
- 抽样查看高相似度句子，判断是否为模板化语句，并关注其是否真正提升 reward。

3) 反例分析
- 找到 `correct_wrong_mean` 高但 reward 仍为错误的样本，检查是否存在奖励噪声或句子切分问题。
