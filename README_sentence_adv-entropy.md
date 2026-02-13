# SentencePO 句子熵优势（sentence-level entropy advantage）

本文档说明在 SentencePO 中加入“句子级熵调制优势”的实现、配置项与使用方式。

## 背景与动机

句子级熵反映模型在某个句子上的不确定性。对优势进行轻度熵调制可以：
- 对高熵句子提供适度探索激励（正优势更偏向保留/放大）。
- 对高熵负优势进行谨慎处理（避免过度惩罚噪声）。

本实现遵循“温和、可控、可消融”的原则：
- 按句子聚合熵（句内 token 平均）。
- 归一化并截断后作为缩放因子。
- 正负优势可分开设置不同强度。

## 方法概述

对每个句子计算平均熵 $H_s$，经归一化与截断得到 $\hat{H}_s$。对句子内每个 token 的优势 $A$，按正负不同系数做缩放：

- $A \ge 0$：$A' = A \cdot (1 + \alpha_{pos}\,\hat{H}_s)$
- $A < 0$：$A' = A \cdot (1 + \alpha_{neg}\,\hat{H}_s)$

其中 $\alpha_{pos}$ 通常大于等于 $\alpha_{neg}$。

### $\hat{H}_s$ 的实现

实现位于 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) 的 `apply_sentence_entropy_advantage()`，具体逻辑：

1. 句子级聚合：对每个句子做 token 熵均值：
	$$H_s = \frac{1}{|s|} \sum_{t \in s} H_t$$

2. Batch 内归一化（由 `sentencepo_adv_entropy_norm` 控制）：
	- `zscore`：
	  $$\tilde{H}_s = \frac{H_s - \mu}{\sigma + \epsilon}$$
	- `minmax`（映射到 $[-1, 1]$）：
	  $$\tilde{H}_s = 2\cdot\frac{H_s - H_{min}}{H_{max} - H_{min} + \epsilon} - 1$$
	- `none`：不做归一化，直接使用 $H_s$。

3. 截断（由 `sentencepo_adv_entropy_clip` 控制）：
	$$\hat{H}_s = \mathrm{clip}(\tilde{H}_s, -c, c)$$
	其中 $c$ 为 `sentencepo_adv_entropy_clip`。

4. 逐 token 广播：把 $\hat{H}_s$ 还原到句子内每个 token，得到对应的熵调制系数。

## 关键配置

配置项位于 `actor_rollout_ref.actor.policy_loss.*`。

- `sentencepo_adv_entropy_enable`：是否开启句子熵优势调制（默认 false）
- `sentencepo_adv_entropy_alpha_pos`：正优势缩放系数（默认 0.1）
- `sentencepo_adv_entropy_alpha_neg`：负优势缩放系数（默认 0.0）
- `sentencepo_adv_entropy_norm`：熵归一化方式（`zscore` / `minmax` / `none`）
- `sentencepo_adv_entropy_clip`：归一化后截断范围（默认 2.0）
- `sentencepo_adv_entropy_eps`：数值稳定项（默认 1e-6）

## 使用示例（脚本开关）

在 `test_sentencepo_v1-2.sh` 中可以直接控制：

- `sentencepo_adv_entropy_enable=true|false`
- `sentencepo_adv_entropy_alpha_pos=0.1`
- `sentencepo_adv_entropy_alpha_neg=0.0`
- `sentencepo_adv_entropy_norm=zscore`
- `sentencepo_adv_entropy_clip=2.0`
- `sentencepo_adv_entropy_eps=1e-6`

## 建议的起步设置

- 先从小强度开始：`alpha_pos=0.05~0.2`，`alpha_neg=0~0.1`
- 归一化推荐 `zscore`，并设置 `clip=2.0`
- 先做消融：关闭/开启对比，观察正确率与多样性变化

## 注意事项

- 本功能依赖句子划分与 token 级熵统计；若日志中不提供 entropys，则不会生效。
- 若训练早期不稳定，可降低 `alpha_pos/alpha_neg` 或直接关闭。

## 实现位置

- 句子熵优势缩放：`verl/trainer/ppo/core_algos.py`
- 优势计算调用：`verl/trainer/ppo/ray_trainer.py`
- 配置定义：`verl/workers/config/actor.py` 与 `verl/trainer/config/actor/actor.yaml`
- 脚本开关：`test_sentencepo_v1-2.sh`
