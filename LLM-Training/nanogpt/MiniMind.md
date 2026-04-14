
2 小时 + 1 张消费级 GPU (如 RTX 3090)，从零训练一个约 64M 参数的 GPT-style LLM

| 特性            | 描述                                                                                    |
| ------------- | ------------------------------------------------------------------------------------- |
| **参数量**       | ~64M (Dense) / ~201M-A67M (MoE，即激活参数仍为 67M)                                           |
| **架构风格**      | GPT-style decoder-only Transformer，同时支持 **Dense** 和 **MoE (Mixture-of-Experts)** 两种模式 |
| **模型维度**      | `hidden_size = 768`, 采用 "768+8" 统一架构                                                  |
| **Tokenizer** | 自训练 BPE Tokenizer（非直接复用他者）                                                            |

MiniMind 覆盖了一个生产级 LLM 的**全部训练阶段**，从左到右依次为：

```
Tokenizer → Pretrain → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL
```

| 阶段 | 全称 | 作用 | 核心思路 |
|---|---|---|---|
| **Tokenizer** | Byte-Pair Encoding tokenizer 训练 | 将原始文本 → token IDs | BPE 算法，从零训练而非复用 |
| **Pretrain** | 预训练 | 让模型习得基础语言能力 | Next-token prediction on `pretrain_hq.jsonl` (~1.6GB) |
| **SFT** | Supervised Fine-Tuning | 让模型习得对话/指令跟随能力 | 在 `sft.jsonl` 上做 supervised instruction tuning |
| **LoRA** | Low-Rank Adaptation | 快速适配新任务（如医疗、身份） | 冻结主权重，仅训练低秩矩阵 $W_{\text{LoRA}} = A \cdot B$，其中 $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$，$r \ll d$ |
| **DPO** | Direct Preference Optimization | 对齐人类偏好，让输出更安全、更有帮助 | 绕过 reward model，直接用 preference pair $(y_w, y_l)$ 训练，loss 为 $\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$ |
| **PPO/GRPO/CISPO** | RLHF / Group Relative PO / CISPO | 进一步优化回答质量 | PPO 用 reward model 打分；GRPO 不用 critic，用 group 相对排名；CISPO 是改进版 |
| **Agentic RL** | Agentic Reinforcement Learning | 让模型学会使用工具（Tool Calling）和自主推理（Adaptive Thinking） | 模型与环境交互，通过 RL 优化工具调用策略 |

**MoE 变体**：MiniMind-MoE 使用多个 Expert FFN，每次只激活 Top-2 Expert，从而在总参数量增大的同时保持激活参数量较小。
不只是 Pretrain，而是涵盖 Pretrain → SFT → Alignment (DPO/PPO/GRPO/CISPO) → Agentic RL 的完整链路。

| 项目 | 描述 |
|---|---|
| **[MiniMind-V](https://github.com/jingyaogong/minimind-v)** | 多模态视觉语言模型 (VLM)，67M 参数，1.3 元 + 1 小时从零训练 |
| **[MiniMind2-Small](https://huggingface.co/jingyaogong/MiniMind2-Small)** | 更小参数版本（~25.8M），进一步降低门槛 |
