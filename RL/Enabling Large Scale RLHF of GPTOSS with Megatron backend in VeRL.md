
# 深入解析：Enabling Large Scale RLHF of GPTOSS with Megatron backend in VeRL

这篇文章来自 Hugging Face Blog，由 LEI WANG（香港科技大学）等人撰写，发表于 2026 年 2 月 10 日。核心主题是**如何在 VeRL 框架中利用 Megatron 后端实现 GPTOSS 模型的大规模 RLHF（Reinforcement Learning from Human Feedback）后训练（post-training）**。下面我从第一性原理出发，逐层拆解。

---

## 一、核心问题与动机

### 1.1 背景：RLHF 的大规模瓶颈

在 2025 年 12 月之前，VeRL 框架中的 RLHF 要求 **colocated deployment**（共置部署）——即 inference（actor, ref, rollout）和 training（actor in GRPO）必须在**同一组 GPU 节点**上运行。这意味着：

- **推理阶段**：模型需要生成 rollout（即采样出的回复序列），这本质上是一个 **自回归解码（autoregressive decoding）** 过程，是 **memory-bandwidth bound** 的（显存带宽瓶颈型），GPU 利用率低。
- **训练阶段**：GRPO 更新 actor 模型参数，这是 **compute-bound** 的（计算瓶颈型），需要大量 FLOPs。

这两种工作负载对硬件资源的利用模式截然不同——**推理是 memory-bound，训练是 compute-bound**——强行共置会导致严重的资源浪费和扩展性瓶颈。

### 1.2 第一性原理分析

从第一性原理出发，RLHF 的训练循环可以抽象为：

$$\theta^{t+1} = \theta^t + \alpha \cdot \nabla_\theta \mathcal{L}_{\text{GRPO}}(\theta^t; \mathcal{D}_{\text{rollout}})$$

其中：
- $\theta^t$ 是第 $t$ 步的 actor 模型参数
- $\alpha$ 是学习率
- $\mathcal{L}_{\text{GRPO}}$ 是 GRPO 的损失函数
- $\mathcal{D}_{\text{rollout}}$ 是由当前策略 $\pi_{\theta^t}$ 生成的 rollout 数据

关键观察：**rollout 生成和参数更新是两个可分离的阶段**。如果 rollout 生成（推理）和参数更新（训练）必须共置，那么扩展性就受限于"木桶效应"——慢的那个阶段拖累整体。

---

## 二、GPTOSS 模型：为什么选择它？

### 2.1 GPTOSS 的定位

文章将 GPTOSS 定位为 **"agentic workflow 的轻量级推理 MoE 模型"**。关键数据：

| 模型 | 激活参数量 | 吞吐量 (toks/sec) | 排名/地位 |
|------|-----------|-------------------|----------|
| GPTOSS 20B | ~3B-5B (activated) | 409 (2025.08) | 开源最快 agentic-ready 模型 |
| GPTOSS 120B | ~3B-5B (activated) | 311 avg, 高达 906 (Together.ai API) | 开源最快，超 Gemini 2.5 Flash-Lite |

### 2.2 三大结构改进

文章总结了 GPTOSS 接近 **Pareto Frontier**（精度-速度的最优前沿）的三个关键改进：

1. **YaRN Rotation** [8]：将上下文长度扩展到 **100k+**，覆盖 agentic workflow 的所有场景需求。
   - YaRN（Yet another RoPE extensioN method）的核心思想是通过调整 RoPE（Rotary Position Embedding）的频率基底 $\theta_i$ 和缩放因子，在长上下文外推时保持注意力分布的稳定性。
   - RoPE 的原始公式：$\text{RoPE}(x_m, m) = x_m \cdot e^{i m \theta_m}$，其中 $\theta_m = 10000^{-2m/d}$。
   - YaRN 引入缩放因子 $s$ 和混合策略：对低频分量使用动态缩放，对高频分量保持原样，从而在扩展上下文时减少信息损失。

2. **Attention Side Project**（类似 [9][10]）：这涉及在注意力机制中添加额外的结构化偏置（可能是类似 **Sinks attention** 或 **attention sink** 的机制），用于处理长序列生成时的注意力稳定性问题。

3. **MoE 架构的优化**：GPTOSS 作为 MoE（Mixture of Experts）模型，激活参数仅 3B-5B，但总参数达 20B 或 120B，实现了"大模型能力、小模型成本"的效果。

### 2.3 性能对标

- GPTOSS 120B 的 906 toks/sec（Together.ai API）超越了 **Gemini 2.5 Flash-Lite (Sep)** 的 613 t/s
- 在 openRouter 上排名第四（非私有模型），仅次于 DeepSeek V3.2、MiniMax M2.1、Kimi K2.5

---

## 三、核心技术方案

### 3.1 Solution Bundle 架构

文章提到的技术栈组合：

```
┌─────────────────────────────────────────┐
│         Proprietary Slurm Platform      │
├─────────────────────────────────────────┤
│                 VeRL (框架层)             │
│  ┌──────────┐  ┌───────────┐            │
│  │  GRPO    │  │   PPO     │            │
│  │ Trainer  │  │  Trainer  │            │
│  └────┬─────┘  └─────┬─────┘            │
│       │              │                  │
├───────┼──────────────┼──────────────────┤
│       ▼              ▼                  │
│  ┌─────────┐  ┌──────────────┐          │
│  │  FSDP   │  │ Megatron-Core│          │
│  │(Training)│  │  (Training)  │          │
│  └─────────┘  └──────────────┘          │
│                                          │
│  ┌──────────┐  ┌──────────────┐          │
│  │  vLLM    │  │   SGLang     │          │
│  │(Inference)│  │ (Inference)  │          │
│  └──────────┘  └──────────────┘          │
│                                          │
│  ┌─────────────────────────────┐          │
│  │     Megatron-Bridge        │          │
│  │  (权重格式转换/通信桥接)     │          │
│  └─────────────────────────────┘          │
└─────────────────────────────────────────┘
```

- **VeRL**：字节跳动开源的 RLHF 训练框架，支持多种后端
- **Megatron-Core**：NVIDIA 的大模型分布式训练框架，支持 TP/PP/EP/DP 等并行策略
- **Megatron-Bridge**：连接 Megatron 格式和 vLLM/SGLang 格式的权重转换桥梁
- **FSDP**：PyTorch 的 Fully Sharded Data Parallelism
- **vLLM/SGLang**：高性能推理引擎

### 3.2 GRPO vs PPO

文章选择 **GRPO** [4]（Group Relative Policy Optimization）而非传统 PPO，原因如下：

#### 传统 PPO 的问题

PPO 的目标函数：

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是概率比，$\hat{A}_t$ 是优势函数估计。

PPO 需要**额外的 value network（critic）**来估计 $\hat{A}_t$，这在大规模 MoE 模型中意味着：
- 额外的显存开销（critic 模型参数）
- 额外的训练不稳定性
- Critic 本身的训练难度

#### GRPO 的改进

GRPO 的核心思想：**用 group 内的相对排名替代绝对优势估计**，无需 critic。

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[\frac{1}{G}\sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \tilde{A}_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)\tilde{A}_i\right) - \beta \cdot \text{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]\right]$$

其中：
- $G$ 是 group 大小（同一 prompt 生成的样本数）
- $\tilde{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$ 是**标准化后的相对奖励**，通过 group 内的奖励排名计算
- $r_i$ 是第 $i$ 个样本的 reward
- $\beta$ 是 KL 惩罚系数
- $\pi_{\text{ref}}$ 是参考策略（通常为 SFT 模型）

**GRPO 的关键优势**：
- ✅ 无需 critic network，节省 ~50% 显存
- ✅ 相对奖励更稳定，减少 reward hacking
- ✅ 天然适合 MoE 模型（不需要为 critic 维护 expert 并行）

### 3.3 Decouple Inference from Training（推理与训练解耦）

这是本文最核心的工程贡献。传统 VeRL 的架构：

```
旧架构 (Colocated):
┌──────────────────────────────────┐
│  GPU Node 0-N                    │
│  ┌──────────┐  ┌───────────────┐ │
│  │ Rollout  │  │   Training    │ │  ← 共享同一组 GPU
│  │ (推理)    │  │   (GRPO)      │ │
│  └──────────┘  └───────────────┘ │
│  Problem: 推理时GPU利用率低       │
│          训练时需要等推理完成      │
└──────────────────────────────────┘
```

新架构（Decoupled）：

```
新架构 (Decoupled):
┌─────────────────┐  ┌─────────────────┐
│ Inference Nodes  │  │ Training Nodes  │
│ (Rollout集群)    │  │ (GRPO集群)       │
│ ┌──────────┐    │  │ ┌──────────────┐│
│ │vLLM/SGLang│    │  │ │Megatron-Core ││
│ │BF16/FP8   │    │  │ │FSDP          ││
│ └──────────┘    │  │ └──────────────┘│
│ Memory-bound    │  │ Compute-bound   │
│ Low GPU util   │  │ High GPU util   │
└─────────────────┘  └─────────────────┘
        │                    │
        └────通过网络传递─────┘
              rollout data
```

**解耦的好处**：
1. **资源效率**：推理节点可以少一些 GPU（因为 memory-bound），训练节点可以用满所有 FLOPs
2. **独立扩展**：推理和训练可以独立 scale out
3. **消除等待**：训练不必等推理完成，可以流水线化

**关键工程挑战**：权重同步。Actor 模型在训练节点更新后，需要同步到推理节点。文章通过 **Megatron-Bridge** 解决格式转换问题（Megatron 的 TP/EP 分布式格式 ↔ vLLM/SGLang 的格式）。

### 3.4 FP8 for Post Training in Hopper Platform

在 NVIDIA Hopper（H800/H100）架构上，文章探索了 FP8 精度用于 post-training：

#### Hopper FP8 的硬件支持

H800 的 H100 芯片（Hopper 架构）引入了原生 FP8 支持：
- **FP8 E4M3**：4 bit 指数 + 3 bit 尾数，范围约 ±448，用于 forward pass（权重和激活）
- **FP8 E5M2**：5 bit 指数 + 2 bit 尾数，范围约 ±57344，用于 backward pass（梯度）

#### FP8 在 RLHF 中的应用策略

文章最终决定 **BF16 + FP8 混合精度** 作为 rollout 系统的主要数据类型：

- **Rollout（推理）**：FP8 量化，因为推理不需要梯度，FP8 的精度损失可接受
- **Training（GRPO 训练）**：BF16，保证参数更新的数值稳定性

#### Precision Verification（精度验证）

文章特别提到了精度验证环节，这很重要因为 FP8 的动态范围有限（E4M3 最大值 ~448），需要确保：
- 量化误差不会导致 rollout 质量下降
- GRPO 的相对奖励计算 $\tilde{A}_i$ 不会因精度损失而失去区分度

---

## 四、Main Experiment 核心结果

### 4.1 实验环境

- **硬件**：H800 DGX SuperPod（每台 DGX 有 8 张 H800 GPU）
- **规模**：64（8×8）到 512（64×8）张 GPU
- **模型**：GPTOSS-20B（MoE 模型，激活参数 ~3B-5B）

### 4.2 线性扩展性结果

| GPU 数量 | 配置 | 训练时间 | 吞吐量 | 每 iter 时间 |
|----------|------|---------|--------|-------------|
| 64 (8×8) | Baseline | 13 hrs | ~ | ~ |
| 512 (64×8) | Scaled | 2 hrs | 500-598 toks/sec | 108.62 s/it |

**线性扩展性**意味着：GPU 数量增加 8×，训练时间减少 ~6.5×（接近线性），这在大模型训练中是非常难得的。

### 4.3 关键发现

1. **16×8 GPU 以上时必须关闭 parameter offloading**：
   - Parameter offloading 是 ZeRO 优化中的技术，将暂时不用的参数 offload 到 CPU 内存
   - 在大规模训练中，offloading 的通信开销反而成为瓶颈
   - 计算公式：$T_{\text{offload}} = T_{\text{transfer}} + T_{\text{compute}}$，当 GPU 数量多到每个 GPU 的计算量减少时，transfer 开销占比增加

2. **推理与训练节点分离**：
   - 大规模时（16×8 以上），共置部署不可行
   - 原因：推理和训练对 NCCL 通信模式的需求不同，共置会导致通信冲突

3. **Week-zero support**：
   - 文章提到了"facilitating week-zero support"，意味着模型可以在发布后立即开始 post-training，无需等待基础设施搭建

---

## 五、第一性原理总结

从第一性原理理解这篇文章的核心贡献：

```
RLHF Post-Training 的时间复杂度：
T_total = T_rollout + T_training + T_communication

传统方法 (Colocated):
- T_rollout 受限于 GPU memory bandwidth（自回归解码是顺序的）
- T_training 受限于 GPU compute（矩阵乘法）
- 两者在同一 GPU 上串行执行 → 资源浪费

本文方法 (Decoupled + FP8):
- T_rollout → 在专用推理节点上，FP8 加速，2x 吞吐量提升
- T_training → 在专用训练节点上，Megatron-Core TP/PP/EP 充分利用 FLOPs
- T_communication → Megatron-Bridge 高效权重同步
- 三者并行化 → 近线性扩展
```

**核心 insight**：RLHF 的 rollout 和 training 阶段具有不同的 hardware utilization profile，decoupling 它们可以让每个阶段都运行在最优的硬件配置上，从而实现近线性扩展。

---

## 六、参考链接

- [1] VeRL 框架：https://github.com/volcengine/verl
- [2] GPTOSS 模型：https://huggingface.co/collections/GPTOSS
- [4] GRPO 论文（DeepSeekMath）：https://arxiv.org/abs/2402.03300
- [5] Artificial Analysis Leaderboard：https://artificialanalysis.ai/
- [6] Together.ai API：https://together.ai/
- [7] openRouter：https://openrouter.ai/
- [8] YaRN 论文：https://arxiv.org/abs/2309.00071
- [9] Attention Sink 相关研究：https://arxiv.org/abs/2309.17453
- [10] StreamingLLM：https://arxiv.org/abs/2309.17453
- Megatron-Core：https://github.com/NVIDIA/Megatron-LM
- vLLM：https://github.com/vllm-project/vllm
- SGLang：https://github.com/sgl-project/sglang

---

## 七、启发与延伸思考

1. **MoE + RLHF 的天然契合**：MoE 模型激活参数少（3B-5B），推理速度快，天然适合需要大量 rollout 的 RLHF 训练循环。这可能是 GPTOSS 在 agentic 场景下表现优异的根本原因。

2. **FP8 的正确使用场景**：FP8 不适合训练（数值不稳定），但在 rollout（推理）中完全可行。这种"混合精度分工"思路可以推广到其他场景。

3. **解耦是系统设计的基本原则**：正如 microservice 架构将不同负载解耦到不同服务中，RLHF 的推理-训练解耦也是同样的思想——**让每个组件运行在最适合自己的环境中**。

4. **线性扩展性 ≠ 理论最优**：8× GPU 减少 6.5× 时间（而非 8×），说明通信开销仍然存在，未来可能需要更高效的权重同步机制（如异步更新、stale synchronous parallelism）。