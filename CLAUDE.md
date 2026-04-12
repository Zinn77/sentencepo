# SentencePO 项目指南

## 项目概述
基于 verl 框架对 GRPO 的改进，核心思想：**在句子级别（sentence-level）进行重要性采样的聚合和优势估计**。目标：投稿顶会。

## 版本迭代
| 版本 | 分支 | 核心内容 | 状态 |
|------|------|----------|------|
| v1-1 | `sentencepo-0.6-port_v1-1` | 句子级聚合 + 句子平均 + Lppl/Llen 自适应 clip | 基础版 |
| v1-1-metrics | `sentencepo_v1-1-metrics` | v1-1 + 监控指标 + 样本记录 | 诊断版 |
| v2 | — | 复杂版 sentence_adv（已弃用） | 弃用 |
| v1-2-test0 | — | verl-0.7 上简化 sentence_adv | 实验 |
| v1-3-metrics | `sentencepo_v1-3-metrics` | v1-1-metrics + entropy/bucket(compare) 两种优势估计 | **可作为新版基础** |
| v1-4-metrics | `sentencepo_v1-4-metrics` | v1-3 + self-judge 优势估计（actor 自身做 judge） | 复杂 |
| v1-4-metrics-sft | 当前分支 | v1-4 + 数据集 rollout、标注、judge SFT 混合训练 | **过于复杂，用户不想继续** |

演进主线：**v1-1 → v1-3 → v1-4 → v1-4-sft**。用户计划在 **v1-3** 基础上开展新改进。

## 架构总览：三层改进

### 第一层：SentencePO Loss（句子级 clip，v1-1 引入）
- 位置：`core_algos.py:compute_policy_loss_sentencepo`（L1374-1576）
- 在 policy loss 函数注册为 `"sentencepo"`
- 核心：句子级聚合 log-ratio → 句子级 PPO clip → 句子平均 → 样本平均 → batch 平均
- 自适应 clip 半径：基于句子 PPL（z_ppl）和句子长度（z_len）
```
delta_s = mean(logp_new - logp_old)  per sentence
rho_s = exp(delta_s)
c_s = c0 * clamp(1 + λ_ppl * z_ppl - λ_len * z_len, cmin, cmax)
sent_obj = min(rho_s * adv_s, clamp(rho_s, exp(-c_s), exp(c_s)) * adv_s)
```
- 参数：eps_base, lambda_ppl, lambda_len, cmin, cmax

### 第二层：句子级优势估计（v1-3 引入，在 advantage 计算后融合）
融合入口：`ray_trainer.py:compute_advantage()`（L270-428）
```
GRPO outcome advantage（序列级，广播到 token）
  + alpha_entropy * entropy_advantage（句子级熵调制，v1-2 引入）
  + alpha_bucket * semantic_advantage（句子级语义分桶对比，v1-3 引入）
```

#### 2a. 句子熵优势（entropy advantage）
- 位置：`core_algos.py:apply_sentence_entropy_advantage()`（L214-296）
- 对每句计算 token 平均熵 H_s → zscore/minmax 归一化 → 截断 → 缩放
- 正负优势分别用不同系数：A' = A * (1 + α_pos/neg * Ĥ_s)
- 开关：`sentencepo_adv_entropy_enable`

#### 2b. 句子语义优势（bucket/compare advantage）
- 位置：`core_algos.py:compute_sentence_semantic_advantage()`（L299-542）
- 按句子在 response 中的相对位置分桶（early/mid/late）
- 组内正确/错误样本构造 embedding 中心 → 计算 cosine 相似度差
- A_sem = sim(h_s, C_correct) - sim(h_s, C_wrong)，可选 z-score
- 需要 hidden states → 需额外 forward pass 获取 sentence embeddings
- 开关：`algorithm.sentence_adv.enable`

### 第三层：Self-Judge Advantage（v1-4 引入，**用户计划放弃**）
- SJA：让 actor 自身作为 judge，通过 prompt 让模型输出 JSON 评分
- Judge SFT：蒸馏 teacher 评分数据，在 RL 中加入 judge SFT 混合损失
- 用户认为太复杂，不够优美，且需要额外微调

## 核心文件
| 文件 | 作用 | 相关版本 |
|------|------|----------|
| `verl/utils/sentence_utils.py` | 共享句子切分逻辑 | all |
| `verl/trainer/ppo/core_algos.py` | SentencePO loss + entropy/bucket adv + GRPO adv | v1-1/v1-3 |
| `verl/trainer/ppo/ray_trainer.py` | 训练编排：advantage 融合入口 | all |
| `verl/trainer/ppo/metric_utils.py` | SentencePO 监控指标 | v1-1+ |
| `verl/trainer/ppo/sentence_judge_adv.py` | SJA 核心 | v1-4（可能弃用）|
| `verl/trainer/config/algorithm.py` | 配置：SentenceAdvConfig, SentenceJudgeAdvConfig, JudgeSFTConfig | all |
| `test_sentencepo_v1-3.sh` | v1-3 测试脚本（可作为新版模板） | v1-3 |

## 关键配置参数

### SentencePO Loss（v1-1）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sentencepo_eps_base` | 0.01 | 句子级 clip 基础 epsilon |
| `sentencepo_lambda_ppl` | 0 | PPL 自适应 clip 权重 |
| `sentencepo_lambda_len` | 0 | 长度自适应 clip 权重 |
| `sentencepo_cmin/cmax` | 0.5/1.5 | clip scale 范围 |
| `sentencepo_min_sent_tokens` | 6 | 短句合并阈值 |

### Entropy Advantage（v1-2）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sentencepo_adv_entropy_enable` | false | 开关 |
| `sentencepo_adv_entropy_alpha_pos` | 0.1 | 正优势缩放系数 |
| `sentencepo_adv_entropy_alpha_neg` | 0.0 | 负优势缩放系数 |

### Bucket/Compare Advantage（v1-3）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sentence_adv.enable` | false | 开关 |
| `sentence_adv.alpha` | 0.1 | 融合权重 |
| `sentence_adv.temperature` | 0.1 | cosine 相似度温度 |
| `sentence_adv.bucket_count` | 3 | 分桶数 |
| `sentence_adv.pooling` | last | 句子 embedding 池化方式 |

## GRPO Baseline（对比基线）
- 位置：`core_algos.py:compute_grpo_outcome_advantage()`（L597-661）
- 标准 GRPO：组内 z-score 归一化 reward → 广播到 token
- Vanilla PPO loss：`core_algos.py:compute_policy_loss_vanilla()`（L1217-1301）
- 用 GRPO + vanilla loss 作为 baseline 对比 SentencePO

## 实验结论（目前）
- 之前所有版本（v1-1 到 v1-4-sft）都 **没跑出超过 baseline GRPO 的效果**
- 用户认为 v1-4/sft 过于复杂，计划在 v1-3 基础上探索新方向
- 用户正在阅读新论文，有新的模糊想法待讨论

## 环境
- 生产路径前缀: `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yangfengkai02`
- 模型: Qwen3-4B-Base / Qwen3-4B-Instruct-2507
- 数据集: math/math500/aime2024/aime2025/amc23/minerva/olympiad
- v1-3 测试脚本用 n=8 rollout，train_batch_size=128
