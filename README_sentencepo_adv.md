# SentencePO 句子级优势估计说明

本文档说明本仓库中 SentencePO 句子级优势估计（Semantic-Aware Process Advantage）的实现改动与使用方式。

## 功能概述
- 在 GRPO 的基础上，引入句子级语义相似度优势估计。
- 句子优势通过 logsumexp 余弦相似度度量正/负集合支持度，和原始 GRPO 优势线性融合。
- 训练时使用句子级重要性采样聚合（SentencePO loss）。

## 主要改动位置
- 句子级 policy loss（SentencePO）实现：
  - [verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py)
- 句子级语义优势估计（logsumexp cosine）：
  - [verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py)
- 优势融合入口（通过 `adv_estimator=grpo_sentencepo` 启用）：
  - [verl/trainer/ppo/ray_trainer.py](../verl/trainer/ppo/ray_trainer.py)
- 隐状态回传（用于句子嵌入）：
  - [verl/workers/actor/dp_actor.py](../verl/workers/actor/dp_actor.py)
  - [verl/workers/fsdp_workers.py](../verl/workers/fsdp_workers.py)
- 算法配置新增项：
  - [verl/trainer/config/algorithm.py](../verl/trainer/config/algorithm.py)
- 训练默认配置补充 sentence_adv 结构（避免 Hydra Struct 报错）：
  - [verl/trainer/config/ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml)
  - [verl/trainer/config/ppo_megatron_trainer.yaml](../verl/trainer/config/ppo_megatron_trainer.yaml)

## 使用方式
通过算法配置选择优势估计类型：
- `algorithm.adv_estimator=grpo_sentencepo`

该设置会：
- 计算 GRPO 的 outcome advantage
- 额外计算句子级语义优势并按权重融合

## 脚本开关（test_sentencepo.sh）
在 [test_sentencepo.sh](../test_sentencepo.sh) 中可通过环境变量设置：
- `ADV_ESTIMATOR`（默认：`grpo_sentencepo`）：选择优势估计器。
- `SENTENCE_ADV_POOLING`（默认：`mean`）：句子向量 pooling，`mean` 或 `last`。
- `SENTENCE_ADV_ALPHA`（默认：`0.1`）：句子级优势融合权重。
- `SENTENCE_ADV_TEMPERATURE`（默认：`0.2`）：logsumexp 温度。

## 注意事项
- 句子嵌入依赖 actor 计算 log-prob 时的 token-level hidden states。
- 若 fused kernels 不返回 hidden states，请关闭 fused kernels 或确保模型支持 `output_hidden_states`。

## 设计要点（简版）
- 正确性判定：reward 为 1/0，因此用 `reward > 0.5` 作为正确/错误划分。
- 相似度：对句子向量做 `cosine`，并使用 logsumexp 聚合得到密度支持度。
- 优势融合：$A_{final} = A_{GRPO} + \alpha \cdot A_{sentence}$。
