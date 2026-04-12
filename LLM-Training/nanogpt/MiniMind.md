**极低成本、极短时间**从零开始训练一个**超小型语言模型 (ultra-small LLM)**。

> **用 3 元人民币 + 2 小时 + 1 张消费级 GPU (如 RTX 3090)，从零训练一个约 64M 参数的 GPT-style 语言模型，并走完全部 LLM training pipeline。**

| 特性            | 描述                                                                                    |
| ------------- | ------------------------------------------------------------------------------------- |
| **参数量**       | ~64M (Dense) / ~201M-A67M (MoE，即激活参数仍为 67M)                                           |
| **训练成本**      | ~3 RMB 电力费                                                                            |
| **训练时间**      | ~2 小时 (单卡 3090)                                                                       |
| **架构风格**      | GPT-style decoder-only Transformer，同时支持 **Dense** 和 **MoE (Mixture-of-Experts)** 两种模式 |
| **模型维度**      | `hidden_size = 768`, 采用 "768+8" 统一架构                                                  |
| **Tokenizer** | 自训练 BPE Tokenizer（非直接复用他者）                                                            |
| **定位**        | 教育 + 研究：降低 LLM 学习门槛，让你能亲手理解每一个组件                                                      |

### 🔧 完整 Training Pipeline

MiniMind 覆盖了一个生产级 LLM 的**全部训练阶段**，从左到右依次为：

```
Tokenizer → Pretrain → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL
```

各阶段含义：

| 阶段 | 全称 | 作用 | 核心思路 |
|---|---|---|---|
| **Tokenizer** | Byte-Pair Encoding tokenizer 训练 | 将原始文本 → token IDs | BPE 算法，从零训练而非复用 |
| **Pretrain** | 预训练 | 让模型习得基础语言能力 | Next-token prediction on `pretrain_hq.jsonl` (~1.6GB) |
| **SFT** | Supervised Fine-Tuning | 让模型习得对话/指令跟随能力 | 在 `sft.jsonl` 上做 supervised instruction tuning |
| **LoRA** | Low-Rank Adaptation | 快速适配新任务（如医疗、身份） | 冻结主权重，仅训练低秩矩阵 $W_{\text{LoRA}} = A \cdot B$，其中 $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$，$r \ll d$ |
| **DPO** | Direct Preference Optimization | 对齐人类偏好，让输出更安全、更有帮助 | 绕过 reward model，直接用 preference pair $(y_w, y_l)$ 训练，loss 为 $\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$ |
| **PPO/GRPO/CISPO** | RLHF / Group Relative PO / CISPO | 进一步优化回答质量 | PPO 用 reward model 打分；GRPO 不用 critic，用 group 相对排名；CISPO 是改进版 |
| **Agentic RL** | Agentic Reinforcement Learning | 让模型学会使用工具（Tool Calling）和自主推理（Adaptive Thinking） | 模型与环境交互，通过 RL 优化工具调用策略 |

---

### 🧱 模型架构解析

MiniMind 的模型代码在 [`model_minimind.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)，核心参数配置如下：

```python
class MiniMindConfig:
    hidden_size = 768          # 隐藏维度 d_model
    num_hidden_layers = 16     # Transformer 层数 L
    num_attention_heads = 12   # 注意力头数 h
    intermediate_size = 2048    # FFN 中间维度 d_ff = 4/3 * d_model (近似)
    vocab_size = 6400          # 词表大小
    max_position_embeddings = 512  # 最大序列长度
```

**参数量估算（第一性原理）**：

对于 Dense 模型，每层参数量为：

$$N_{\text{layer}} = 4 \cdot d_{\text{model}}^2 + 2 \cdot d_{\text{model}} \cdot d_{\text{ff}}$$

- 第一项 $4d^2$：来自 Q/K/V/O 四个投影矩阵（各为 $d \times d$）
- 第二项 $2d \cdot d_{\text{ff}}$：来自 FFN 的两个线性层

代入 $d = 768, d_{\text{ff}} = 2048, L = 16$：

$$N \approx L \cdot (4 \times 768^2 + 2 \times 768 \times 2048) + V \times d \approx 16 \times (2.36M + 3.15M) + 6400 \times 768 \approx 88M + 4.9M \approx 93M$$

（与官方声称的 ~64M 略有出入，实际可能用了更小的 `num_hidden_layers` 或 `intermediate_size`，具体以代码为准）

**MoE 变体**：MiniMind-MoE 使用多个 Expert FFN，每次只激活 Top-2 Expert，从而在总参数量增大的同时保持激活参数量较小。

---

### 🌟 生态扩展

| 项目 | 描述 |
|---|---|
| **[MiniMind-V](https://github.com/jingyaogong/minimind-v)** | 多模态视觉语言模型 (VLM)，67M 参数，1.3 元 + 1 小时从零训练 |
| **[MiniMind2-Small](https://huggingface.co/jingyaogong/MiniMind2-Small)** | 更小参数版本（~25.8M），进一步降低门槛 |

---

### 💡 设计哲学（第一性原理）

MiniMind 的核心思想可以拆解为：

1. **可复现性**：一切从零开始（Tokenizer、数据、训练），不依赖预训练权重，让你能完整复现每一个环节。
2. **极简主义**：模型小到能在消费级硬件上跑，但架构与生产级 LLM 完全一致（Transformer + MoE），因此是理解 LLM 原理的**最小完整实例**。
3. **全流程覆盖**：不只是 Pretrain，而是涵盖 Pretrain → SFT → Alignment (DPO/PPO/GRPO/CISPO) → Agentic RL 的完整链路，这在大模型项目中极为罕见。

---

### 📎 参考链接

- **项目主页**：https://jingyaogong.github.io/minimind/
- **GitHub 仓库**：https://github.com/jingyaogong/minimind
- **英文 README**：https://github.com/jingyaogong/minimind/blob/master/README_en.md
- **MiniMind 文档**：https://minimind.readthedocs.io/
- **MiniMind-V (多模态)**：https://github.com/jingyaogong/minimind-v
- **HelloGitHub 介绍**：https://hellogithub.com/repository/3280977d37c142bd8ed65ce0f67e4e75
- **知乎专栏**：https://zhuanlan.zhihu.com/p/20815798807
