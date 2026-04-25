# SentencePO v1-5: SLPA + SCR 方法说明文档

## 目录

1. [方法概述](#1-方法概述)
2. [背景与动机](#2-背景与动机)
3. [核心算法](#3-核心算法)
   - 3.1 [SLPA: Sentence-Level Process Advantage](#31-slpa-sentence-level-process-advantage)
   - 3.2 [SCR: Sentence Contrastive Reward](#32-scr-sentence-contrastive-reward)
   - 3.3 [非对称融合与总体公式](#33-非对称融合与总体公式)
4. [创新点与理论贡献](#4-创新点与理论贡献)
5. [代码改动详解](#5-代码改动详解)
6. [配置参数](#6-配置参数)
7. [运行指南](#7-运行指南)
8. [实验计划（NeurIPS 投稿）](#8-实验计划neurips-投稿)
9. [监控指标说明](#9-监控指标说明)

---

## 1. 方法概述

SentencePO v1-5 在 GRPO (Group Relative Policy Optimization) 的基础上，引入两个**句子级优势估计模块**，构成"双通道"句子级信用分配机制：

```
A_final(k, i) = A_GRPO(i) + α_V · SLPA(k, i) + α_C · SCR(k, i)
```

其中：
- **A_GRPO(i)**：标准 GRPO 优势（序列级 z-score reward，广播到所有 token）
- **SLPA(k, i)**：句子级过程优势（Value Channel）— 通过核回归估计句子边界的价值函数，计算时序差分
- **SCR(k, i)**：句子对比奖励（Discriminative Channel）— 通过 reward-weighted 软中心计算对比亲和度
- **α_V, α_C**：融合权重，支持正确/错误回答使用不同的 α（非对称融合）

整个方法**零额外推理成本**：复用 GRPO 已有的 N 条 rollout 作为蒙特卡洛样本，仅需一次额外的 hidden state 提取（在 log_prob 计算时顺带完成）。

---

## 2. 背景与动机

### 2.1 GRPO 的局限

GRPO 对每个 prompt 采样 N 条 rollout，计算 z-scored reward 作为优势：

```
A_GRPO(i) = (r_i - μ_group) / σ_group
```

这个优势是**序列级**的：同一条 rollout 中的所有 token 获得相同的优势值。这意味着 GRPO 无法区分一条回答中"哪些句子贡献了正确结果、哪些导致了错误"。

### 2.2 先前版本的问题

| 版本 | 方法 | 失败原因 |
|------|------|----------|
| v1-1 | 句子级自适应 clip | 只改了优化方式（clip），没改优势信号本身 |
| v1-2 | 句子熵优势 | 使用模型不确定性而非 outcome 信息，信号太弱 |
| v1-3 | 分桶+embedding 中心对比 | 位置分桶太粗糙，N=8 时中心噪声大，cosine 被非推理特征主导 |

**核心诊断**：先前方法都没有直接利用 reward 来分配句子级信用。

### 2.3 v1-5 的设计思路

借鉴 KTAE（Kernel Temporal Advantage Estimation）和 SegmentPO 的思想，但做了关键改进：

1. **用 reward 做句子级价值估计**（SLPA）：不是简单地把 reward 广播，而是通过核回归估计每个句子位置的"期望累积价值" V_k
2. **用 reward-weighted 软中心做对比**（SCR）：不是二元正确/错误划分，而是用 softmax 对 reward 加权，产生平滑的对比信号
3. **非对称融合**：正确/错误回答使用不同的融合权重，因为对"做对了的回答"和"做错了的回答"，信用分配的侧重不同

---

## 3. 核心算法

### 3.1 SLPA: Sentence-Level Process Advantage

**核心思想**：在每个句子边界估计一个"状态价值" V_k，然后计算相邻句子之间的价值差 Δ_k = V_k - V_{k-1} 作为该句子的信用。

#### 3.1.1 符号定义

对于一个 prompt 的 group g，有 N 条 rollout：

- 第 i 条 rollout 有 K_i 个句子，第 k 个句子的 embedding 为 h_k^(i)
- 第 i 条 rollout 的 reward 为 r_i
- 第 k 个句子在 rollout 中的相对位置为 pos_k = k / max(K_i - 1, 1) ∈ [0, 1]

#### 3.1.2 核函数

使用两个核函数衡量句子间的相似性：

**Embedding 核**（衡量内容相似性）：
```
K_emb(h_k^(i), h_l^(j)) = exp(cosine(h_k^(i), h_l^(j)) / τ_emb)
```

**Position 核**（衡量位置相近性）：
```
K_pos(k, l) = exp(-|pos_k - pos_l|² / (2σ²))
```

**组合核**：
```
K(k_i, l_j) = K_emb(h_k^(i), h_l^(j)) · K_pos(k, l)
```

#### 3.1.3 价值估计（Leave-One-Out）

对于 rollout i 的第 k 个句子，通过核加权回归估计其"状态价值"：

```
V_k(i) = Σ_{j≠i} argmax_l  [ K(k_i, l_j) · r_j  /  K(k_i, l_j) ]
```

**关键设计**：
- **Leave-one-out**：排除自身 rollout 的所有句子，避免信息泄漏
- **核加权**：内容相似 + 位置相近的句子获得更高权重
- **直接使用 reward**：r_j 是 rollout j 的 outcome reward

**直觉**：如果句子 k 的内容和位置与那些高 reward rollout 中的句子相似，则 V_k 高；反之则低。

#### 3.1.4 时序差分

```
Δ_k(i) = V_k(i) - V_{k-1}(i)
```

其中 V_{-1}(i) = μ_group（即 GRPO 的 baseline）。

**直觉**：Δ_k > 0 表示句子 k 使得"价值提升"（与高 reward 更对齐），Δ_k < 0 表示句子 k "偏离了好的方向"。

#### 3.1.5 归一化

对每个 group 内的所有 Δ_k 做 z-score 归一化：

```
Δ̂_k = (Δ_k - mean(Δ)) / std(Δ)
```

#### 3.1.6 与 GRPO 的理论关系

**GRPO 是 SLPA 的特殊情况**：当 σ → ∞（position kernel 退化为常数）且 τ_emb → 0（embedding kernel 退化为均匀权重）时，V_k(i) → μ_{-i}（leave-one-out group mean），Δ_k → 0，SLPA 退化为零，恢复标准 GRPO。

这意味着 SLPA 是 GRPO constant baseline 的**严格推广**：从 μ_group 推广到句子级状态依赖的 V_k(s)。

### 3.2 SCR: Sentence Contrastive Reward

**核心思想**：用 reward-proportional 的软权重构建"正向中心"和"负向中心"，衡量每个句子的内容与哪个中心更亲近。

#### 3.2.1 Rollout Embedding

对每条 rollout i，计算其所有句子 embedding 的均值作为 rollout 级表示：

```
ē_i = normalize(mean(h_1^(i), h_2^(i), ..., h_{K_i}^(i)))
```

#### 3.2.2 Reward-Weighted 软中心（Leave-One-Out）

对于 rollout i，用剩余 rollout 的 reward 构建软权重：

```
w_j^+ = softmax(r_j / τ_r)   for j ≠ i    (高 reward 得高权重)
w_j^- = softmax(-r_j / τ_r)  for j ≠ i    (低 reward 得高权重)
```

**正向中心**（"好的回答像什么"）：
```
C^+(i) = normalize( Σ_{j≠i} w_j^+ · ē_j )
```

**负向中心**（"差的回答像什么"）：
```
C^-(i) = normalize( Σ_{j≠i} w_j^- · ē_j )
```

#### 3.2.3 对比评分

对 rollout i 的第 k 个句子：

```
SCR_k(i) = cosine(h_k^(i), C^+(i)) / τ_s - cosine(h_k^(i), C^-(i)) / τ_s
```

SCR_k > 0 表示句子内容更接近"好回答"的模式，SCR_k < 0 表示更接近"差回答"的模式。

#### 3.2.4 与 v1-3 Bucket/Compare 的区别

| 维度 | v1-3 Bucket/Compare | v1-5 SCR |
|------|---------------------|----------|
| 权重 | 二元（正确/错误） | Softmax reward 软权重 |
| 分桶 | 位置分桶（early/mid/late） | 无分桶，全局中心 |
| 中心 | 组内句子均值 | Rollout 级均值 embedding |
| Leave-one-out | 无 | 有（排除自身 rollout） |

### 3.3 非对称融合与总体公式

最终优势的融合公式：

```
A_final(k, i) = A_GRPO(i) + α(i) · SLPA(k, i) + β(i) · SCR(k, i)
```

其中融合权重根据 rollout 的正确性分别设置：

```
α(i) = α_correct   if r_i > threshold
        α_incorrect  otherwise

β(i) = β_correct   if r_i > threshold
        β_incorrect  otherwise
```

**动机**：
- 对**正确回答**：重点放大"哪些句子贡献了成功"（α_correct 可以设大一些）
- 对**错误回答**：重点识别"哪些句子导致了失败"（α_incorrect 可以用不同值）
- 这种非对称性在先前工作中未被探索

---

## 4. 创新点与理论贡献

### 创新点 1: State-Dependent Baseline 推广 GRPO

**贡献**：GRPO 使用常数 baseline μ_group，我们将其推广为句子级状态依赖的 V_k(s)。理论上证明 GRPO 是 SLPA 的特殊情况（σ→∞, τ→0）。

**意义**：这不仅是工程上的改进，而是对 GRPO 优势估计的理论推广，建立了从 constant baseline 到 state-dependent baseline 的连续谱。

### 创新点 2: Zero-Cost Group-Internal Value Estimation

**贡献**：利用 GRPO 已有的 N 条 rollout 作为蒙特卡洛样本，通过核回归估计 V_k，无需额外推理（不需要 critic network 或 PRM）。

**对比**：
- GAE 需要训练 critic network → 额外计算成本
- PRM/ORM 需要训练过程奖励模型 → 额外标注 + 训练成本
- SLPA 复用已有 rollout → 零额外推理成本

### 创新点 3: Asymmetric Dual-Channel Credit Assignment

**贡献**：提出双通道信用分配（Value + Discriminative），并引入非对称融合（正确/错误回答使用不同权重）。

**意义**：
- Value Channel (SLPA)：回答"这个句子使价值提高了多少"
- Discriminative Channel (SCR)：回答"这个句子更像好回答还是差回答"
- 两个通道提供互补信息，非对称融合允许对正确/错误回答使用不同策略

---

## 5. 代码改动详解

### 5.1 文件变更摘要

```
verl/trainer/config/algorithm.py   |  +65 行  (新增 SLPAConfig, SCRConfig)
verl/trainer/ppo/core_algos.py     | +297 行  (新增 compute_slpa_advantage, compute_scr_advantage)
verl/trainer/ppo/ray_trainer.py    | +101/-42 行 (融合逻辑重构，新增 SLPA/SCR 融合块)
verl/workers/fsdp_workers.py       |  +12 行  (扩展 hidden state 提取触发条件)
test_sentencepo_v1-5.sh            |  新增    (完整测试脚本)
```

### 5.2 `verl/trainer/config/algorithm.py`

新增两个 dataclass 配置类：

**SLPAConfig** (L87-114):
```python
@dataclass
class SLPAConfig(BaseConfig):
    enable: bool = False
    alpha_correct: float = 0.1      # 正确回答的融合权重
    alpha_incorrect: float = 0.1    # 错误回答的融合权重
    tau_emb: float = 0.1            # embedding 核温度
    sigma_pos: float = 1.0          # position 核带宽
    normalize: bool = True          # 是否 z-score 归一化
    eps: float = 1e-8
    correctness_threshold: float = 0.0  # 正确/错误划分阈值
    metrics_enable: bool = True
```

**SCRConfig** (L117-144):
```python
@dataclass
class SCRConfig(BaseConfig):
    enable: bool = False
    alpha_correct: float = 0.05
    alpha_incorrect: float = 0.05
    tau_reward: float = 1.0         # reward softmax 温度
    tau_sim: float = 0.1            # cosine 相似度缩放温度
    normalize: bool = True
    eps: float = 1e-8
    correctness_threshold: float = 0.0
    metrics_enable: bool = True
```

在 **AlgoConfig** 中新增引用 (L195-196):
```python
slpa: Optional[SLPAConfig] = None
scr: Optional[SCRConfig] = None
```

### 5.3 `verl/trainer/ppo/core_algos.py`

**`compute_slpa_advantage()`** (L545-710, ~165 行):

算法流程：
1. L592-601: L2 归一化 sentence embeddings，计算 scalar rewards
2. L603-621: **向量化**计算每个句子在其 rollout 内的 local rank 和相对位置 [0,1]
   - 使用 `scatter_reduce_(reduce="amin")` 高效计算每个 sample 的 first position
   - 避免了显式 Python 循环
3. L629-679: 按 group 遍历，对每个 group:
   - L648-649: 计算 embedding 核矩阵 `K_emb = exp(cos_sim / τ)`
   - L652-653: 计算 position 核矩阵 `K_pos = exp(-Δpos² / 2σ²)`
   - L656-657: Leave-one-out 掩码：`K = K_emb * K_pos * (~same_rollout)`
   - L660-662: 核加权价值估计：`V_k = (K @ rewards) / K.sum()`
   - L666-679: 按 rollout 计算时序差分 `Δ_k = V_k - V_{k-1}`
4. L682-689: 可选 z-score 归一化
5. L692-699: 映射回 token 级（同一句子的所有 token 获得相同优势值）

**`compute_scr_advantage()`** (L713-839, ~125 行):

算法流程：
1. L766-771: 计算每条 rollout 的均值 embedding
2. L777-812: 按 group 遍历，对每条 rollout 做 leave-one-out:
   - 计算 softmax reward 权重 w+ 和 w-
   - 构建正向中心 C+ 和负向中心 C-
   - 对该 rollout 的每个句子计算 `SCR = sim(h, C+) - sim(h, C-)`
3. L815-822: z-score 归一化
4. L825-832: 映射回 token 级

### 5.4 `verl/trainer/ppo/ray_trainer.py`

**`compute_advantage()` 函数** (L344-434):

重构为三个连续的融合块，共享 tensor 引用：

```
(a) Bucket/Compare (v1-3 legacy) → 可选，默认关闭
(b) SLPA                         → 新增，默认开启
(c) SCR                          → 新增，默认开启
→ 清理临时 tensor (sentence_embeddings 等)
(d) Entropy advantage            → 保持不变
```

SLPA/SCR 融合块的核心逻辑 (以 SLPA 为例, L378-402):
```python
slpa_adv, slpa_metrics = core_algos.compute_slpa_advantage(...)

# 非对称融合
scores = data.batch["token_level_rewards"].sum(dim=-1)
correct = scores > threshold
alpha = torch.where(correct, alpha_correct, alpha_incorrect).unsqueeze(-1)  # (bs, 1)
data.batch["advantages"] = data.batch["advantages"] + alpha * slpa_adv
```

**训练循环** (L1730-1740):

扩展了 hidden state 提取触发条件：
```python
_need_sent_emb = (
    sentence_adv_cfg.enable or slpa_cfg.enable or scr_cfg.enable
)
if _need_sent_emb:
    batch.meta_info["return_hidden_states"] = True
    batch.meta_info["sentence_adv_pool_only"] = True
```

**Metrics 提取** (L1824-1832):
```python
slpa_metrics = batch.meta_info.pop("slpa_metrics", None)
scr_metrics = batch.meta_info.pop("scr_metrics", None)
```

### 5.5 `verl/workers/fsdp_workers.py`

**Hidden state 提取和 pooling** (L967-981):

扩展了判断是否需要 return hidden states 的条件：
```python
return_hidden_states = (
    sentence_adv_cfg.enable or slpa_cfg.enable or scr_cfg.enable
)
```

Pooling 模式回退逻辑：当 SLPA/SCR 开启但 sentence_adv 未配置时，默认使用 `"last"` pooling。

---

## 6. 配置参数

### 6.1 SLPA 参数

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `slpa.enable` | false | 开关 | — |
| `slpa.alpha_correct` | 0.1 | 正确回答融合权重 | 0.05 ~ 0.3 |
| `slpa.alpha_incorrect` | 0.1 | 错误回答融合权重 | 0.05 ~ 0.3 |
| `slpa.tau_emb` | 0.1 | Embedding 核温度 | 0.05 ~ 0.5。越小 → 越尖锐（只看最相似句子） |
| `slpa.sigma_pos` | 1.0 | Position 核带宽 | 0.3 ~ 2.0。越小 → 越局部（只看附近位置） |
| `slpa.normalize` | true | Z-score 归一化 | 建议保持 true |
| `slpa.correctness_threshold` | 0.0 | 正确/错误划分阈值 | 0.0（binary reward 用 0.5） |

### 6.2 SCR 参数

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `scr.enable` | false | 开关 | — |
| `scr.alpha_correct` | 0.05 | 正确回答融合权重 | 0.02 ~ 0.2 |
| `scr.alpha_incorrect` | 0.05 | 错误回答融合权重 | 0.02 ~ 0.2 |
| `scr.tau_reward` | 1.0 | Reward softmax 温度 | 0.5 ~ 5.0。越小 → 越极端（接近 argmax） |
| `scr.tau_sim` | 0.1 | Cosine 缩放温度 | 0.05 ~ 0.5 |
| `scr.normalize` | true | Z-score 归一化 | 建议保持 true |

### 6.3 SentencePO Loss 参数（保持不变）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sentencepo_eps_base` | 0.01 | 句子级 clip 基础 epsilon |
| `sentencepo_lambda_ppl` | 0 | PPL 自适应 clip 权重 |
| `sentencepo_lambda_len` | 0 | 长度自适应 clip 权重 |
| `sentencepo_min_sent_tokens` | 6 | 短句合并阈值 |

---

## 7. 运行指南

### 7.1 分支结构

```
sentencepo_v1-5-combined  ← SLPA + SCR 都开启（完整方法）
sentencepo_v1-5-slpa      ← 仅 SLPA（消融实验用）
sentencepo_v1-5-scr       ← 仅 SCR（消融实验用）
sentencepo_v1-3-metrics   ← 基线代码（v1-3 bucket/compare）
```

### 7.2 运行方式

**完整方法 (SLPA + SCR)**:
```bash
cd ~/sentencepo_v1-5-combined
bash test_sentencepo_v1-5.sh
```

**仅 SLPA**:
```bash
cd ~/sentencepo_v1-5-slpa
bash test_sentencepo_v1-5.sh
# 或在 combined 分支用环境变量覆盖:
slpa_enable=true scr_enable=false bash test_sentencepo_v1-5.sh
```

**仅 SCR**:
```bash
cd ~/sentencepo_v1-5-scr
bash test_sentencepo_v1-5.sh
# 或:
slpa_enable=false scr_enable=true bash test_sentencepo_v1-5.sh
```

**GRPO Baseline**（用 vanilla loss 替换 sentencepo loss）:
```bash
# 在 v1-3 分支上，关闭 sentencepo 相关模块
cd ~/sentencepo_v1-3-metrics
bash test_sentencepo_v1-3.sh \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    sentence_adv_enable=false \
    sentencepo_adv_entropy_enable=false
```

### 7.3 超参数覆盖示例

```bash
# 调整 SLPA 核温度和非对称权重
slpa_tau_emb=0.05 slpa_sigma_pos=0.5 \
slpa_alpha_correct=0.15 slpa_alpha_incorrect=0.05 \
bash test_sentencepo_v1-5.sh

# 调整 SCR reward 温度
scr_tau_reward=2.0 scr_alpha_correct=0.1 \
bash test_sentencepo_v1-5.sh
```

### 7.4 部署到生产环境

```bash
# 将代码同步到生产机器
rsync -av ~/sentencepo_v1-5-combined/ $PROD_HOST:~/sentencepo_v1-5-combined/

# 在生产机器上（假设路径前缀为 /mnt/dolphinfs/...）
# 修改 test_sentencepo_v1-5.sh 中的路径，然后运行
```

---

## 8. 实验计划（NeurIPS 投稿）

### 8.1 实验设计概览

```
Phase 1: 验证有效性（最关键）
Phase 2: 消融实验
Phase 3: 分析与可视化
Phase 4: 扩展性实验
```

### 8.2 Phase 1: 主实验 — 验证 SLPA+SCR 超过 GRPO

**目标**：在 MATH 数据集上证明方法有效。

| 实验名 | 方法 | Loss | 优势估计 |
|--------|------|------|----------|
| **Baseline 1** | GRPO | vanilla PPO clip | GRPO z-score |
| **Baseline 2** | SentencePO v1-1 | sentencepo clip | GRPO z-score |
| **Ours (full)** | SentencePO v1-5 | sentencepo clip | GRPO + SLPA + SCR |

**评测集**：MATH-500, AIME 2024, AIME 2025, AMC 2023, MINERVA, Olympiad

**关键超参数（首轮）**:
```bash
# Ours (full) - 默认超参数
slpa_enable=true slpa_alpha_correct=0.1 slpa_alpha_incorrect=0.1
slpa_tau_emb=0.1 slpa_sigma_pos=1.0
scr_enable=true scr_alpha_correct=0.05 scr_alpha_incorrect=0.05
scr_tau_reward=1.0 scr_tau_sim=0.1
```

**训练设置**：Qwen3-4B-Base, 3 epochs, N=8 rollout, batch_size=128

### 8.3 Phase 2: 消融实验

#### 8.3.1 模块消融

| 实验 | SLPA | SCR | SentencePO Loss |
|------|------|-----|-----------------|
| GRPO baseline | off | off | off (vanilla) |
| + SentencePO loss only | off | off | on |
| + SLPA only | **on** | off | on |
| + SCR only | off | **on** | on |
| + SLPA + SCR (full) | **on** | **on** | on |

这组实验直接证明每个模块的独立贡献和叠加效果。

#### 8.3.2 非对称 α 消融

| 实验 | α_correct | α_incorrect | 说明 |
|------|-----------|-------------|------|
| 对称 | 0.1 | 0.1 | α_c = α_i |
| 强化正确 | 0.15 | 0.05 | 重点放大正确回答的信用 |
| 强化错误 | 0.05 | 0.15 | 重点识别错误回答的原因 |
| 仅正确 | 0.1 | 0.0 | 只对正确回答做信用分配 |
| 仅错误 | 0.0 | 0.1 | 只对错误回答做信用分配 |

#### 8.3.3 SLPA 核参数消融

| 实验 | τ_emb | σ_pos | 说明 |
|------|-------|-------|------|
| 默认 | 0.1 | 1.0 | — |
| 尖锐 embedding | 0.05 | 1.0 | 更关注高度相似句子 |
| 平滑 embedding | 0.5 | 1.0 | 更均匀的权重 |
| 局部 position | 0.1 | 0.3 | 只看附近位置 |
| 全局 position | 0.1 | 5.0 | 接近无 position bias |
| 无 position kernel | 0.1 | ∞(100) | 纯 embedding similarity |
| 无 embedding kernel | ∞(100) | 1.0 | 纯 position proximity |

#### 8.3.4 Group Size (N) 消融

| N | 说明 |
|---|------|
| 4 | 最小 — 每组仅 4 条 rollout |
| 8 | 默认 |
| 16 | 更多样本 → V_k 估计更准 |
| 32 | 上限 — 计算成本高但估计最稳 |

### 8.4 Phase 3: 分析与可视化（论文 Figure 用）

#### 8.4.1 V_k 轨迹可视化

**Figure idea**: 对同一个 prompt 的 N 条 rollout，画出 V_k 随句子位置的变化曲线。正确回答（绿色曲线）vs 错误回答（红色曲线），展示 SLPA 能区分"价值上升"和"价值下降"的轨迹。

**数据来源**：`slpa/V_mean`, `slpa/V_std` metrics + 可在 `compute_slpa_advantage` 中加 per-sample V_k 记录。

#### 8.4.2 Δ_k 分布分析

**Figure idea**: 按句子位置（early/mid/late）分组，画 Δ_k 的分布，对比正确 vs 错误回答。期望看到：
- 正确回答的 Δ_k 在关键推理步骤处有明显正峰
- 错误回答的 Δ_k 在出错位置有明显负峰

#### 8.4.3 SCR Score 与 Reward 的相关性

**Figure idea**: Scatter plot of SCR_k vs rollout reward, 按句子位置着色。展示 SCR 信号的判别力。

#### 8.4.4 Case Study

选取 MATH/AIME 的具体样例：
- 展示同一 prompt 下正确/错误 rollout 的每个句子的 SLPA Δ_k 和 SCR 值
- 标注出"关键推理步骤"和"出错位置"
- 这种 case study 对 NeurIPS reviewer 非常有说服力

#### 8.4.5 训练曲线

**必须绘制的曲线**：
- 训练 reward 曲线 (GRPO vs SLPA vs SCR vs Full)
- 各评测集的 accuracy 曲线
- SLPA/SCR metrics 曲线（V_mean, delta_std, scr_score_std 等）

### 8.5 Phase 4: 扩展性实验

#### 8.5.1 不同模型规模

| 模型 | 参数量 | 说明 |
|------|--------|------|
| Qwen3-1.7B | 1.7B | 小模型是否有效 |
| Qwen3-4B-Base | 4B | 默认 |
| Qwen3-8B-Base | 8B | 更大模型 |

#### 8.5.2 不同数据集 / Domain

| 数据集 | Domain |
|--------|--------|
| MATH | 数学推理（主实验） |
| GSM8K | 小学数学 |
| CodeContests | 代码 |
| ARC-Challenge | 科学推理 |

#### 8.5.3 与其他方法对比

| 方法 | 来源 | 说明 |
|------|------|------|
| GRPO | DeepSeek | Baseline |
| Dr.GRPO | (2025) | 不除以 std 的 GRPO |
| DAPO | (2025) | Dynamic sampling |
| PRIME/FreePRM | (2025) | 隐式 PRM |
| KTAE | (2025) | 核时序优势（最相关 baseline） |

### 8.6 实验优先级（建议执行顺序）

```
优先级 1（必须做，1-2 周）：
  [1] GRPO baseline
  [2] SentencePO loss only (v1-1 等效)
  [3] SLPA + SCR full (v1-5)
  → 如果 [3] > [1]，继续下面的实验

优先级 2（消融，1 周）：
  [4] SLPA only
  [5] SCR only
  [6] 非对称 α 消融（2-3 组）
  [7] τ_emb / σ_pos 消融（2-3 组）

优先级 3（分析 + 扩展，1-2 周）：
  [8] V_k 轨迹可视化 + Case Study
  [9] 不同 N (4, 8, 16)
  [10] 不同模型规模

优先级 4（论文补充材料）：
  [11] 与 KTAE/PRIME 等方法对比
  [12] 不同数据集
```

### 8.7 论文结构建议

```
Title: "Sentence-Level Process Advantage for Group Relative Policy Optimization"
       (或 "Zero-Cost Sentence-Level Credit Assignment for RLVR")

1. Introduction
   - GRPO 的 token-uniform advantage 问题
   - 提出 SLPA+SCR：句子级信用分配，零额外推理成本

2. Related Work
   - GRPO/PPO/REINFORCE 系列
   - Token/step-level credit: KTAE, SegmentPO, PRIME
   - Contrastive reward: BiCC/RCC, SRPO

3. Method
   - 3.1 Preliminaries: GRPO
   - 3.2 SLPA: Kernel-Weighted Value Estimation
   - 3.3 SCR: Soft Contrastive Reward
   - 3.4 Asymmetric Dual-Channel Fusion
   - 3.5 Theoretical Analysis: GRPO as Special Case

4. Experiments
   - 4.1 Setup
   - 4.2 Main Results
   - 4.3 Ablation Study
   - 4.4 Analysis & Visualization
   - 4.5 Scaling Analysis

5. Conclusion
```

---

## 9. 监控指标说明

训练过程中会自动记录以下 metrics（在 TensorBoard 中可见）：

### SLPA Metrics

| Metric | 说明 | 健康范围 |
|--------|------|----------|
| `slpa/V_mean` | V_k 估计均值 | 接近 group mean reward |
| `slpa/V_std` | V_k 估计标准差 | > 0，表示有区分度 |
| `slpa/delta_mean` | Δ_k 均值 | z-score 后应接近 0 |
| `slpa/delta_std` | Δ_k 标准差 | z-score 后应接近 1 |
| `slpa/delta_abs_mean` | |Δ_k| 均值 | 反映信号强度 |
| `slpa/num_sentences` | 句子总数 | batch_size × avg_sents_per_sample |
| `slpa/num_groups` | Group 数 | batch_size / N |

### SCR Metrics

| Metric | 说明 | 健康范围 |
|--------|------|----------|
| `scr/score_mean` | SCR 评分均值 | z-score 后应接近 0 |
| `scr/score_std` | SCR 评分标准差 | z-score 后应接近 1 |
| `scr/num_sentences` | 句子总数 | 同 SLPA |

### 异常诊断

- `slpa/V_std ≈ 0`：核函数太平滑，所有句子得到近似相同的 V_k → 尝试减小 τ_emb 或 σ_pos
- `slpa/delta_abs_mean` 很大但训练不稳定：α 可能设太大 → 减小 alpha_correct/incorrect
- `scr/score_std ≈ 0`：所有句子得到相似的对比分 → 检查 τ_reward 是否太大（导致 w+ ≈ w-）
- 训练 reward 下降：SLPA/SCR 信号可能与 GRPO 方向冲突 → 先减小 α 观察
