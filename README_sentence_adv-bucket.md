# SentencePO 句子语义优势（分桶版）

本文档说明在 SentencePO 中加入“句子级语义优势（按推理阶段分桶）”的实现、配置项与监控指标。

## 背景与动机

同一条正确回答内，不同句子对应不同推理阶段，语义差异较大。单中心对比会被平均化，导致优势信号变弱。
本实现按句子相对位置分桶（early/mid/late），在组内对“正确/错误”样本做对比，得到更稳定的句子级优势信号。

## 方法概述

对每个句子提取嵌入 $h_s$，基于组内正确/错误样本构造桶内中心 $C_b, W_b$，计算相似度差值并融合到优势：

$$
A_s^{sem} = \text{sim}(h_s, C_b) - \text{sim}(h_s, W_b)
$$

最终优势：

$$
A_s^{final} = A_s^{base} + \alpha \cdot A_s^{sem}
$$

- 句子按相对位置分桶，默认 3 桶（early/mid/late）。
- 组内只要“正确/错误”任一集合为空，则该桶语义优势置 0。
- 与句子熵优势可以同时开启，通过不同开关控制。

## 关键配置

配置项位于 `algorithm.sentence_adv.*`：

- `enable`: 是否开启句子语义优势（默认 false）
- `alpha`: 融合权重（默认 0.1）
- `temperature`: 相似度温度（默认 0.1）。用于缩放 cosine 相似度，值越小差值越放大、越敏感；值越大则更平滑、对噪声更稳健。
- `pooling`: 句子嵌入池化方式（`mean` 或 `last`）
- `normalize`: 是否对桶内 $A_s^{sem}$ 做 z-score（默认 true）。归一化发生在同一 group + 同一桶内，有助于消除不同组/桶的尺度差异。
- `eps`: 数值稳定项（默认 1e-8）
- `correctness_threshold`: 判定正确的序列奖励阈值（默认 0.0）
- `bucket_count`: 分桶数（默认 3）
- `metrics_enable`: 是否输出桶诊断指标（默认 true）

### 与句子熵优势组合

句子熵优势仍然通过 `actor_rollout_ref.actor.policy_loss.*` 控制，可独立开关：

- `sentencepo_adv_entropy_enable: true|false`
- `sentencepo_adv_entropy_alpha_pos`
- `sentencepo_adv_entropy_alpha_neg`

## 使用示例

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.sentence_adv.enable=true \
  algorithm.sentence_adv.alpha=0.1 \
  algorithm.sentence_adv.bucket_count=3 \
  algorithm.sentence_adv.pooling=last \
  algorithm.sentence_adv.temperature=0.1 \
  algorithm.sentence_adv.normalize=true \
  algorithm.sentence_adv.correctness_threshold=0.0 \
  actor_rollout_ref.actor.policy_loss.sentencepo_adv_entropy_enable=true
```

## 监控指标（判断分桶是否合理）

当 `metrics_enable=true`，会额外输出以下指标（以相似度与分离度为主）：

- `sentence_adv/bucket_{k}/sent_count`: 第 k 桶句子数
- `sentence_adv/bucket_{k}/adv_mean`: 语义优势均值
- `sentence_adv/bucket_{k}/adv_std`: 语义优势标准差
- `sentence_adv/bucket_{k}/sim_pos_center_pos_mean`: 正确回答句子到正确中心的平均相似度
- `sentence_adv/bucket_{k}/sim_neg_center_neg_mean`: 错误回答句子到错误中心的平均相似度
- `sentence_adv/bucket_{k}/sim_pos_center_neg_mean`: 错误回答句子到正确中心的平均相似度
- `sentence_adv/bucket_{k}/sim_neg_center_pos_mean`: 正确回答句子到错误中心的平均相似度
- `sentence_adv/bucket_{k}/center_cos`: 正确中心与错误中心的余弦相似度
- `sentence_adv/bucket_{k}/sep`: 桶内正负中心的分离度（越大表示聚类差异越明显）

建议观察：
- `center_cos` 是否在不同桶有明显变化
- `sep` 是否在某些桶显著更大，提示阶段性聚类差异

## 实现位置

- 句子语义优势：`verl/trainer/ppo/core_algos.py`
- 优势融合入口：`verl/trainer/ppo/ray_trainer.py`
- 句子嵌入池化：`verl/workers/fsdp_workers.py`

