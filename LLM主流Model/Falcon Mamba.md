---
source_pdf: Falcon Mamba.pdf
paper_sha256: 6b87440ce671ce6f81716a0f79208731d2623a26a7263e5a7d5cd33da43fcd25
processed_at: '2026-08-04T06:29:50-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Falcon Mamba 7B：用人话讲一遍

## 1. 一句话概括

这篇 paper 说：**别管 attention 了，纯 Mamba 也能在 7B 规模打平甚至打赢 Llama3.1、Mistral 这种精心调过的 Transformer**。

这事以前没人做成。之前所有 scale up Mamba 到 7B+ 的尝试（Zamba、Samba、Jamba、Mamba2-8B）最后都妥协了，往 Mamba 里塞 attention 层变成 hybrid，才能追上 Transformer。Falcon 团队说不塞，就是纯 Mamba，结果还真追上了。

这本身就是 paper 最有价值的 contribution——一个存在性证明。

---

## 2. 背景：为什么大家觉得必须 hybrid？

Transformer 的核心是 attention：每个 token 去看所有之前的 token，决定怎么聚合信息。好处是 expressivity 强，坏处是计算和内存随 context 长度平方增长。

Mamba 是另一种思路：把历史信息全压进一个固定大小的 hidden state $h$，新 token 来了就 update 一下这个 state。好处是推理时内存和速度都是常数，坏处是大家怀疑"固定大小的 state 装不下复杂的长程依赖"。

近一年的 narrative 是：Mamba 单独不行，attention 单独也不行，**hybrid 最好**。Jamba、Zamba、Samba 全是这个套路——每隔几个 Mamba block 插一个 attention block。

Falcon 团队起了逆反心理：**到底是 Mamba 架构本身不行，还是之前训练 Mamba 的人没把 data 和 recipe 调好？**

他们赌后者。结果赌对了。

---

## 3. Mamba 到底在干嘛？用类比讲

想象你在读一本书，脑子里维护一个"当前理解状态" $h$。每读一个新词 $x_t$，你做两件事：

1. **决定这个新词有多重要**（用 $\Delta_t$ 控制）：重要的词让 state 大幅更新，不重要的词基本忽略
2. **决定怎么把这个新词的信息写进 state**（用 $B_t$ 控制），以及**从 state 里读出什么**（用 $C_t$ 控制）

数学上就是：

$$
h_t = \bar{A} h_{t-1} + \bar{B}_t x_t
$$

$$
y_t = C_t \cdot h_t
$$

- $h_t \in \mathbb{R}^N$：hidden state，N=16 在 Falcon Mamba 里。这是"脑子里的状态向量"
- $\bar{A}$：状态转移矩阵，控制"历史信息保留多少"
- $\bar{B}_t$：输入投影，控制"新信息写多少进 state"
- $C_t$：输出投影，控制"从 state 里读什么出来"
- $\Delta_t$：discretization step size，控制"对当前 token 反应多激烈"

Mamba 相比经典 SSM 的关键创新：**$B_t, C_t, \Delta_t$ 都依赖输入 $x_t$**。这就是 "selective" 的意思——模型学会根据当前 token 内容决定 forget 多少、memorize 多少。

对比 attention：attention 是每个 token 显式去 query 所有历史 token，表达力强但贵。Mamba 是把所有历史压进 state，表达力弱一点但便宜。**trade-off 的本质是 fixed-size state vs explicit key-value cache**。

参考原论文 Fig. 1 有完整架构图：https://arxiv.org/abs/2312.00752

---

## 4. Falcon Mamba 的架构细节

### 4.1 核心参数（Table 1）

| 参数 | 值 | 人话解释 |
|------|-----|---------|
| n_layers | 64 | 64 层 Mamba block 堆叠 |
| d_model | 4096 | 残差通道宽度 |
| expansion factor E | 2 | block 内部把宽度扩到 8192 |
| vocab_size | 65024 | 沿用 Falcon tokenizer |
| tied_embedding | False | **input embedding 和 output head 不共享参数** |
| d_conv | 4 | causal conv kernel，捕捉 local pattern |
| $\Delta$ proj. size | 16 | 投影到 $\Delta$ 的维度 |
| state dim N | 16 | 每 channel 的 state 大小 |
| 总参数 | 7.27B | |

### 4.2 Mamba block 的数据流

```
x (d_model=4096)
  ↓
LayerNorm (pre-norm)
  ↓
Linear: 4096 → 8192  (expansion E=2)
  ↓
Conv1d causal, kernel=4  (捕捉前 4 个 token 的 local pattern)
  ↓
SiLU activation
  ↓
分三路投影：
  ├── B_t = Linear_B(x) → RMSNorm → (N=16)
  ├── C_t = Linear_C(x) → RMSNorm → (N=16)
  └── Δ_t = softplus(Linear_Δ(x)) → RMSNorm → scalar per channel
  ↓
Discretize: Ā = exp(Δ·A), B̄ = f(Δ, A, B)
  ↓
SSM 递归: h_t = Ā·h_{t-1} + B̄_t·x_t ; y_t = C_t·h_t
  ↓
Linear: 8192 → 4096 (project back)
  ↓
+ residual
  ↓
output
```

### 4.3 关键工程改动：在 B/C/$\Delta$ 后加 RMSNorm

这是这篇 paper 最实用的 trick。Mamba 训练时很容易 loss spike，特别是大 LR 下。Jamba、Mamba2 都报告了同样问题。

Falcon 团队的解决方案：**在 $B_t, C_t, \Delta_t$ 三个 input-dependent projection 后面各加一个 RMSNorm**。

为什么这有效，我的直觉：
- $B_t$ 进入递归 $h_t = \bar{A} h_{t-1} + \bar{B}_t x_t$，如果 $B_t$ 数值不稳，会沿时间方向指数放大爆炸
- $C_t$ 直接乘 state 产生输出 logit，量级抖动会让 logit 漂移
- $\Delta_t$ 通过 $\exp(\Delta A)$ 离散化，对量级尤其敏感——$\Delta$ 太大 $\exp$ 直接爆

RMSNorm 把这三个量约束在稳定区间，等于在"信号进入递归前"加了个缓冲。

对比 Mamba2 的做法：把 RMSNorm 放在 block 末尾 output projection 前。Falcon 实验下来这个效果不如"局部 norm"。

参考 Jamba 的 normalization 讨论：https://arxiv.org/abs/2403.19887

---

## 5. 训练 recipe：这才是 paper 的隐形主角

Falcon Mamba 能赢，**架构只占 30%，data 和 recipe 占 70%**。这部分才是真正可迁移的 know-how。

### 5.1 WSD Learning Rate Schedule

WSD = Warmup-Stable-Decay，miniCPM 提出。Falcon 的具体配置：

- Warmup：1GT（1 GigaToken）从 0 线性到 $\eta_{\max}$
- Stable：保持 $\eta_{\max} = 6.4 \times 10^{-4}$
- Decay：指数衰减到 $\eta_{\min} = \eta_{\max}/256 = 2.5 \times 10^{-6}$

公式：

$$
\eta(t) = \eta_{\max} \exp\left[-\frac{t}{t_{\text{decay}}} \log\frac{\eta_{\max}}{\eta_{\min}}\right]
$$

- $t$：decay 阶段已训练的 token 数
- $t_{\text{decay}}$：decay 阶段总时长
- $\eta_{\max}, \eta_{\min}$：LR 上下界

**关键发现**：用约 10% 总 tokens 跑 decay，比 community 惯例长得多。这解释了 pre-decay checkpoint（57.29 avg）到 final checkpoint（64.09 avg）的 6.8 分飞跃。

直觉：decay 阶段不是"收尾"，是"把之前学到的 fuzzy knowledge sharpen 成 precise knowledge"。Stable 阶段是探索，decay 阶段是利用。给 decay 足够时间，模型才能把 stable 阶段积累的 capacity 真正释放出来。

参考 miniCPM：https://arxiv.org/abs/2404.06395

### 5.2 Batch Scaling：被低估的 elegant trick

这个我觉得是 paper 最聪明的地方。

#### 背景：gradient noise temperature

Malladi et al. 2022 推导出 Adam 的 effective noise temperature：

$$
T_{\text{noise}} = \frac{\eta}{\sqrt{b}}
$$

- $\eta$：learning rate
- $b$：batch size

这个量刻画训练过程的"随机性温度"。增大 batch 减小梯度方差（温度下降），减小 LR 也降温度。它们以 $\eta/\sqrt{b}$ 的形式耦合——注意是 $\sqrt{b}$ 不是 $b$，因为 Adam 的二阶矩归一化已经吸收了一个 $\sqrt{b}$ 因子。

#### Batch size rampup 的副作用

Falcon 用 batch size rampup：从 128 线性增到 2048，持续 50GT。前期小 batch 利于 exploration，后期大 batch 利于 throughput。

**但有个隐藏代价**：rampup 期间 $T_{\text{noise}}$ 持续下降。Stable 阶段结束进入 decay 时，温度已经偏低，decay 阶段本应提供的 sharpening 效果被削弱。

#### Batch scaling 的修复

做法：**rampup 结束后，保持 $T_{\text{noise}} = \eta/\sqrt{b}$ 恒定**。即调 batch 时同步调 LR：

$$
\eta \propto \sqrt{b}
$$

这与 SGD 的 "linear scaling rule"（$\eta \propto b$）不同，因为 Adam 的二阶矩已经吃掉一个 $\sqrt{b}$。

直觉：把"调 batch"和"调 LR"两个超参从 coupled 变成统一的 noise budget。这样 decay 阶段的 sharpening 不会被 rampup 阶段的 temperature 下降"预支"掉。

结果：decay 阶段 loss boost 更明显，final loss 更低。

参考 Malladi et al.：https://openreview.net/forum?id=F2mhzjHkQP

### 5.3 Optimizer

- AdamW，$\beta_1=0.9, \beta_2=0.95$（$\beta_2$ 比 0.999 小，让二阶矩更快适应分布变化）
- $\epsilon=10^{-8}$
- weight decay = 0.1
- **没用 Z-loss**，但 follow-up 实验显示 Z-loss 能进一步稳定（与 Wortsman et al. 一致）

Z-loss 形式：

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_z (\log Z)^2
$$

- $Z = \sum_i \exp(z_i)$：logit partition function
- $(\log Z)^2$：惩罚 logit 量级过大

对 Mamba 这种容易 spike 的架构特别 relevant，下次训练可以考虑加。

参考：https://openreview.net/forum?id=d8w0pmvXbZ

### 5.4 硬件

- 256 × H100 80GB
- 纯 Data Parallelism + ZeRO，没用 tensor/pipeline parallel

7B Mamba 单卡能放下（含 optimizer state 通过 ZeRO offload），DP 的通信开销低于 TP 的 all-reduce。这对 SSM 容易——SSM 的 state 远小于 KV cache，不需要 TP 切。

---

## 6. 数据策略

### 6.1 数据来源

| 类别 | 来源 | 占比策略 |
|------|------|---------|
| Web | RefinedWeb（5T tokens 英文） | 主体，多 stage 渐变 |
| Curated | books, arXiv, PubMed, USPTO, Reddit, StackExchange, Hackernews | 后期 stage 加大占比 |
| Code | The Stack | 渐进注入 |
| Math | Proof-Pile-2 + FastText 过滤的 web math | 后期加大 |

**关键决策**：排除 multilingual。理由是 7B 容量不够同时学好 English + 其他语言，宁可 English-only 把单语能力做强。

### 6.2 四阶段 curriculum + decay

| Stage | Seq len | Data 倾向 |
|-------|---------|----------|
| 1 | 2048 | 基础 web |
| 2 | ~4096 | 开始加 code |
| 3 | ~8192 | 加 math、curated |
| 4 | 8192 | 高质量科学密集 |
| Decay | 8192 | Fineweb-Edu + Cosmopedia synthetic + 少量 instruction (3.7%, 4 epochs) |

**Decay 阶段掺 instruction data** 是个微妙的决定。传统 base model 预训练严格不混 instruction，怕损害 fine-tuning 灵活性。Falcon 团队经验是：**少量 + 少 epoch 能 boost in-context retrieval，又不会 overfit instruction format**。

这与 miniCPM 的发现一致，也是 community 逐渐接受的"co-training"思路。

### 6.3 为什么 instruction data 对 Mamba 特别重要

Wen et al. 2024 证明 RNN-style 架构在 in-context retrieval（找上下文里的 [A][B]...[A] 预测 [B]）有结构性瓶颈。Falcon 团队承认这是 limitation，但声称高质量 CoT instruction data 能"部分"绕过。

机制猜测：CoT 数据让模型学会"显式复述上下文"而不是隐式 attention 检索。这绕过了架构限制，代价是输出更长。

Arora et al. 2024 "Just Read Twice" 也是类似思路：https://arxiv.org/abs/2407.05483

---

## 7. 评测结果：到底赢在哪、输在哪

### 7.1 HF Leaderboard v1（Table 2，越容易的 benchmark）

| Model | ARC-25 | HellaSwag | MMLU-5 | Winogrande | TruthfulQA | GSM8K | Avg |
|-------|--------|-----------|--------|------------|------------|-------|-----|
| **FalconMamba-7B** | **62.03** | 80.82 | 62.11 | 73.64 | **53.42** | 52.54 | **64.09** |
| Falcon2-11B | 59.73 | 82.91 | 58.37 | 78.30 | 52.56 | 53.83 | 64.28 |
| Gemma-7B | 61.09 | 82.20 | 64.56 | 79.01 | 44.79 | 50.87 | 63.75 |
| Llama3.1-8B | 58.53 | 82.13 | 66.43 | 74.35 | 44.29 | 47.92 | 62.28 |
| Mistral-7B | 59.98 | 83.31 | 64.16 | 78.37 | 42.15 | 37.83 | 60.97 |
| RecurrentGemma-9B (hybrid) | 52.00 | 80.40 | 60.50 | 73.60 | 38.60 | 42.60 | 57.95 |
| Zamba-7B (hybrid) | 56.14 | 82.23 | 58.11 | 79.87 | 52.88 | 30.78 | 60.00 |
| RWKV-v6-Finch-14B | 47.44 | 78.86 | 52.33 | 71.27 | 45.45 | 38.06 | 55.57 |
| mamba-7b-rw (pure SSM) | 51.25 | 80.85 | 33.41 | 71.11 | 32.08 | 4.70 | 45.52 |
| FalconMamba-7B (pre-decay) | 49.23 | 80.25 | 57.27 | 70.88 | 37.28 | 21.83 | 57.29 |

**三个关键观察**：

1. **Pre-decay → final 的 GSM8K 从 21.83 跳到 52.54**，2.4 倍。decay 阶段 sharpening 数学能力最直接的证据。

2. **与 mamba-7b-rw（45.52）对比，Falcon Mamba 高了 18.57 分**。同样架构、同样规模，pure data/recipe 优化能带来巨大差距。这是 paper 最 powerful 的论点。

3. **TruthfulQA 上 53.42 居首**。这有点 surprising，因为 TruthfulQA 通常依赖 in-context + knowledge recall，传统认为是 attention 强项。可能解释：Falcon Mamba 的 math/curated data 比例高，模型学到更"事实导向"的 distribution。

### 7.2 HF Leaderboard v2（Table 3，更难的 benchmark）

| Model | IFEval | BBH | Math-Lvl5 | GPQA | MuSR | MMLU-Pro | Avg |
|-------|--------|-----|-----------|------|------|----------|-----|
| **FalconMamba-7B** | **33.36** | 19.88 | 3.63 | 8.05 | **10.86** | 14.47 | **15.04** |
| Gemma-7B | 26.59 | 21.12 | 6.42 | 4.92 | 10.98 | 21.64 | 15.28 |
| Mistral-7B | 23.86 | 22.02 | 2.49 | 5.59 | 10.68 | 22.36 | 14.50 |
| Llama3.1-8B | 12.70 | 25.29 | 4.61 | 6.15 | 8.98 | 24.95 | 13.78 |
| RecurrentGemma-9B | 30.76 | 14.80 | 4.83 | 4.70 | 6.60 | 17.88 | 13.20 |
| Zamba-7B | 24.06 | 21.12 | 3.32 | 3.03 | 7.74 | 16.02 | 12.55 |
| RWKV-v6-Finch-14B | 29.81 | 12.89 | 1.13 | 5.01 | 3.16 | 11.3 | 10.55 |

**关键观察**：

1. **MuSR（multi-step soft reasoning，长上下文推理）上 10.86 居首**。这正中 Mamba 强项——SSM 把全部历史压进 fixed state，迫使模型学到"reasoning summarization"能力。

2. **IFEval（instruction following）33.36 遥遥领先**。可能因为 decay 阶段掺了 CoT instruction data。

3. **BBH、MMLU-Pro 略低于 Gemma / Llama3.1**。这两类任务需要 broad knowledge recall + in-context pattern matching，正是 attention 强项、Mamba 弱项。

4. **Zamba 12.55 < Falcon Mamba 15.04**。纯 Mamba 反而胜过 hybrid，这是 paper 核心论点的直接验证。

### 7.3 一句话总结 benchmark

- **Pure Mamba 在 reasoning-dense 任务（MuSR、GSM8K、TruthfulQA）上能胜过 Transformer**
- **Pure Mamba 在 retrieval-heavy 任务（MMLU-Pro、BBH）上有 gap**
- **Hybrid 设计（RecurrentGemma、Zamba）并未普遍优于 pure Mamba**

---

## 8. 推理效率：这才是 Mamba 真正的卖点

### 8.1 Transformer vs Mamba 的内存模型

**Transformer**：
- Prefill：$O(L^2)$ compute（attention 矩阵），$O(L \cdot d)$ KV cache 内存
- Decode：每 token 需 attend 所有历史 KV，per-token 时间 $O(L \cdot d)$，**内存随 $L$ 线性增长**

**Mamba**：
- Prefill（parallel）：parallel scan $O(L \log L)$ compute，但需存 hidden states，内存 $O(L \cdot d)$
- Decode：每 token 只 update 固定大小 state $h \in \mathbb{R}^{N \cdot d_{\text{inner}}}$，per-token 时间 $O(N \cdot d_{\text{inner}})$，**内存 $O(N \cdot d_{\text{inner}})$，与 $L$ 无关**

这是 Mamba 在长 output generation 场景的杀手锏。

### 8.2 Parallel Prefill vs Sequential Prefill

这里有个被忽视的问题：**SSM 的 prefill 阶段内存仍随 $L$ 增长**（hidden states 必须保存用于 backward）。

- **Parallel Prefill**：标准做法，整个 prompt 一次性 forward，最大化 GPU 利用，但内存 $O(L)$
- **Sequential Prefill**：token-by-token 处理 prompt，类似 sequence parallelism。Transformer 没好处（仍需 KV cache），但 SSM 可以"流式"输入，**内存 $O(1)$**

### 8.3 实验数据

**24GB A10 GPU**（Fig. 2）：
- Llama3.1-8B / Mistral-7B / Qwen2-7B：context 到某个长度就 OOM（KV cache 拖累）
- Falcon Mamba parallel prefill：能 fit 更长 context
- Falcon Mamba sequential prefill：**任意长度**，只受时间限制

**80GB H100，prompt=1, generate up to 130k tokens**（Fig. 3）：
- Mistral-7B：峰值内存线性增长，throughput 随 $L$ 下降
- Falcon Mamba：throughput **常数**，峰值内存 **常数**

对实际部署意义巨大：长 output generation（长文档摘要、代码生成、agent 长循环），Mamba 是唯一可行架构。

### 8.4 一个重要 caveat

训练 context length = 8192。**虽然推理能处理任意长输入，模型本身的 long-context understanding 能力是在 8k 上学到的**。Paper 第 6 节明确承认这点，列为 future work。

---

## 9. Batched Generation 的工程细节

### 9.1 SSM 的 padding 问题

Transformer 推理用 left padding，attention mask 屏蔽 padding token。Mamba 的 SSM 是递归，**没有 mask 机制**——padding token 也会进入 state 累积：

$$
h_t = \bar{A} h_{t-1} + \bar{B}_t x_t
$$

padding token（即使 EOS-like）进入 $h_t$ 会污染后续真实 token 的 state。

### 9.2 Falcon 的解决方案

三步处理：
1. **Left padding**（与 Transformer 一致）
2. **在 causal conv 之前** zero out padding token 的 hidden states
3. **在 causal conv 之后**再次 zero out

为什么 conv 前后都要 zero？Causal conv 是 $y_t = \sum_{k=0}^{d_{\text{conv}}-1} w_k \cdot x_{t-k}$，padding 区有非零值会 leak 到右边第一个真实 token。Zero 在 conv 前解决输入端 leak；zero 在 conv 后处理 conv 自身输出边界效应。

这个 trick 让 Mamba 支持批量推理，部署 critical。

---

## 10. 我的几个直觉联想

### 10.1 为什么 pure Mamba 能追平 Transformer？

我认为关键不是 architecture 本身的等价性，而是：

1. **Data quality & mixture 优化空间仍远未饱和**——RefinedWeb + curated + Fineweb-Edu + Cosmopedia 这套 recipe 任何架构都能受益
2. **WSD + batch scaling + 长 decay** 这套 training recipe 真的 work，与架构耦合度低
3. **7B 这个规模，capacity 不算 bottleneck**，架构差异容易被 data 抹平

这意味着 paper 的核心贡献可能不只是 "pure Mamba work"，更是 "**pure data/recipe 优化能 compensate 架构劣势**"——这对 community 是重要 reminder。

### 10.2 与 Mamba-2 的关系

Mamba-2 (Dao & Gu 2024) 通过 structured state space duality 把 SSM 和 attention 统一，理论上应该更优。但 Falcon Mamba 用的是 Mamba-1 架构。可能原因：

- Mamba-2 的 SSD 实现在 7B 规模上没明显优势
- 工程上 Mamba-1 更成熟，optimization 更稳定
- 或者 Falcon 团队就是想直接复用 Gu & Dao 的 reference impl

参考 Mamba-2：https://arxiv.org/abs/2405.21060

### 10.3 与 Zamba 的对比

Zamba 7B 在每 6 个 Mamba block 后插一个 shared attention。Falcon Mamba v1 上 64.09 vs Zamba 60.00，**纯 Mamba 反而胜过 hybrid**。

这间接说明：Zamba 的 attention 层可能没用足够数据训练好，或 hybrid 的额外参数没被有效利用。**Hybrid 不是 free lunch**——attention 层也要 learn，data budget 被分散了。

参考 Zamba：https://arxiv.org/abs/2405.16712

### 10.4 与 RecurrentGemma 的对比

RecurrentGemma 9B 用 Griffin 架构（local attention + linear recurrent），9B 比 7B 大，且是 hybrid，仍输给 Falcon Mamba（57.95 vs 64.09）。说明 Griffin 的 hybrid 设计在这个规模上不如 well-tuned pure Mamba。

参考 RecurrentGemma：https://arxiv.org/abs/2404.07839

### 10.5 与 RWKV-v6 Finch 的对比

RWKV-v6 用 matrix-valued state，14B 版本平均 55.57 仍输给 7B Falcon Mamba 64.09。这表明 Mamba 的 selective 机制（input-dependent $B, C, \Delta$）在表达力上优于 RWKV 的 fixed-state recurrence。

参考 RWKV-v6：https://arxiv.org/abs/2404.05892

### 10.6 Z-loss 没用但应该用

Paper 承认 follow-up 实验显示 Z-loss 能进一步稳定。我猜下次 release 会加上。Z-loss 对 Mamba 这种容易 spike 的架构特别 relevant，公式简单：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_z \cdot (\log Z)^2
$$

其中 $Z = \sum_i \exp(z_i)$ 是 logit partition function，$(\log Z)^2$ 惩罚 logit 量级过大。

### 10.7 Long-context training 是真正的 frontier

Paper 承认训练只用 8192 context length。但 Mamba 的架构优势在 long context 推理才显现——**训练时不见过长 context，推理时怎么用好？**

这是个 chicken-and-egg 问题。解决路径：
1. 训练时用更长 context（但 compute 成本高）
2. 用 Sequential prefill + chunk 训练，避免 $O(L^2)$
3. 或专门 fine-tune 一个 long-context stage

参考 long-context Mamba 的探索：https://arxiv.org/abs/2406.07887

---

## 11. 总结：这篇 paper 的真正贡献

按重要性排序：

1. **存在性证明**：pure Mamba 在 7B + 5.8T tokens 规模上能追平 Transformer SOTA。推翻"hybrid 是 SSM scale-up 唯一出路"的主流假设。

2. **Training recipe 的可迁移性**：WSD + batch scaling + 长 decay + 高质量 data mixture 是 architecture-agnostic 的。Mamba 团队的工程经验可被其他架构借鉴。

3. **Normalization hack**：B/C/$\Delta$ 上加 RMSNorm 是稳定 Mamba 训练的有效 trick。对所有用 Mamba 的人有 practical value。

4. **Sequential prefill 的 explicit 讨论**：明确指出 SSM prefill 阶段的内存问题及解法，对 inference framework 开发者重要。

5. **Open release**：pre-decay checkpoint 也开源，让社区能 continual training，少见的慷慨。

---

## 12. 给你的 actionable takeaways

如果你要训练自己的 Mamba LLM，记住这几条：

1. **B, C, $\Delta$ 后各加一个 RMSNorm**——这是稳定训练的最低成本 trick
2. **WSD schedule，给 decay 至少 10% tokens**——比常规长，但效果显著
3. **Batch scaling：$\eta \propto \sqrt{b}$**——rampup 之后保持 noise temperature 恒定
4. **Decay 阶段掺少量 instruction data（~3-4%）**——能 boost in-context 能力
5. **Untie embeddings**——7B 规模下 output head 需要独立参数空间
6. **$\beta_2 = 0.95$ 而不是 0.999**——让二阶矩更快适应分布变化
7. **考虑加 Z-loss**——Falcon 没加但建议加
8. **推理用 sequential prefill**——长 prompt 不会 OOM
9. **Batch 推理时 padding 三步处理**：left pad + conv 前 zero + conv 后 zero

---

## 参考资料汇总

主要论文：
- Falcon Mamba：https://arxiv.org/abs/2410.05355
- Mamba 原始：https://arxiv.org/abs/2312.00752
- Mamba-2 (SSD)：https://arxiv.org/abs/2405.21060
- Jamba (hybrid)：https://arxiv.org/abs/2403.19887
- Zamba (hybrid)：https://arxiv.org/abs/2405.16712
- Samba (hybrid)：https://arxiv.org/abs/2406.07522
- Griffin / RecurrentGemma：https://arxiv.org/abs/2402.19427
- RWKV-v6 Eagle/Finch：https://arxiv.org/abs/2404.05892
- Mamba-7B-RW (TRI)：https://arxiv.org/abs/2405.06640
- Empirical study of Mamba-based LMs：https://arxiv.org/abs/2406.07887
- Wen et al. "RNNs are not Transformers yet"：https://arxiv.org/abs/2402.18510
- Arora et al. "Just Read Twice"：https://arxiv.org/abs/2407.05483

训练相关：
- miniCPM (WSD schedule)：https://arxiv.org/abs/2404.06395
- Malladi et al. SDEs for adaptive gradient methods：https://openreview.net/forum?id=F2mhzjHkQP
- Wortsman et al. Z-loss & small-scale proxies：https://openreview.net/forum?id=d8w0pmvXbZ

数据集：
- RefinedWeb：https://arxiv.org/abs/2306.01116
- FineWeb-Edu：https://arxiv.org/abs/2406.17557
- Cosmopedia：https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- The Stack：https://arxiv.org/abs/2211.15533
- Proof-Pile-2 (Llemma)：https://arxiv.org/abs/2310.10631

Benchmarks：
- Open LLM Leaderboard v1：https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard
- Open LLM Leaderboard v2：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- lm-evaluation-harness：https://github.com/EleutherAI/lm-evaluation-harness
- lighteval：https://github.com/huggingface/lighteval

Code & Models：
- Falcon Mamba 7B on HF：https://huggingface.co/tiiuae/falcon-mamba-7b
- Transformers library：https://github.com/huggingface/transformers
- Optimum-Benchmark：https://github.com/huggingface/optimum-benchmark
- llama.cpp (Mamba support)：https://github.com/ggerganov/llama.cpp

Falcon 系列：
- Falcon2-11B：https://arxiv.org/abs/2407.14885
- Original Falcon：https://arxiv.org/abs/2311.16867

Transformer 基础：
- Attention is All You Need：https://arxiv.org/abs/1706.03762
- FlashAttention：https://arxiv.org/abs/2205.14135
- FlashAttention-2：https://arxiv.org/abs/2307.08691
- Longformer：https://arxiv.org/abs/2004.05150

想深入哪个部分告诉我，我可以再展开 parallel scan 的实现、SSD 的数学、或具体 inference framework 怎么改。

---

# Falcon Mamba 7B 深度解读

## 1. 这篇paper的核心主张

这篇paper要回答的核心问题是：**纯 Mamba 架构（无 attention）在 7B 规模、5.8T tokens 训练下，能否与精心优化的 Transformer LLM 抗衡？**

答案是 **yes**。Falcon Mamba 7B 在 Open LLM Leaderboard v1 上 average 64.09，v2 上 15.04，超越了 Llama3.1-8B (62.28 / 13.78)、Mistral-7B (60.97 / 14.50)、Falcon2-11B (64.28 / 13.78)，并且与 Gemma-7B 持平。这打破了此前 hybrid Mamba-Transformer 设计（如 Jamba、Zamba、Samba）才是 SSM scale-up 唯一出路的主流认知。

参考链接：
- 原论文：https://arxiv.org/abs/2410.05355
- HuggingFace model card：https://huggingface.co/tiiuae/falcon-mamba-7b
- Open LLM Leaderboard v2：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- Mamba 原论文：https://arxiv.org/abs/2312.00752
- Mamba-2 (structured state space duality)：https://arxiv.org/abs/2405.21060

---

## 2. Mamba 架构基础：从 SSM 到 Selective SSM

要理解为什么这个结果重要，必须先 internalize Mamba 的数学本质。

### 2.1 经典 SSM

State Space Model 是一个连续时间的线性时不变系统：

$$
h'(t) = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)
$$

- $h(t) \in \mathbb{R}^N$：隐藏状态向量，N 是 state dimension（Falcon Mamba 取 N=16）
- $x(t) \in \mathbb{R}$：输入信号（单个 scalar，channel-wise 独立处理）
- $y(t)$：输出
- $A \in \mathbb{R}^{N \times N}$：状态转移矩阵
- $B \in \mathbb{R}^{N \times 1}$、$C \in \mathbb{R}^{1 \times N}$：输入/输出投影

离散化（zero-order hold）后得到 RNN 形式：

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
$$

其中 $\bar{A} = \exp(\Delta A)$，$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$，$\Delta$ 是 step size。

关键点：**这是一个线性递归，所以可以用 parallel scan 在 $O(L \log L)$ 时间内训练，但推理时是 $O(L)$ 的，且每步只需固定大小的 state**。这正是 Mamba 在 long context generation 上常数内存的根源。

### 2.2 Mamba 的"Selective"创新

经典 SSM 的 $A, B, C, \Delta$ 是固定参数，与输入无关——这相当于"不动声色地"把所有历史信息压缩进一个固定大小的 state，对 language modeling 不够 expressive。

Mamba 的核心创新是让 $B, C, \Delta$ 依赖于输入 $x_t$（注意 $A$ 仍可保持为可学习参数，或也可输入相关）：

$$
B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))
$$

- $B_t \in \mathbb{R}^{N}$：决定"写入多少"到 state
- $C_t \in \mathbb{R}^{N}$：决定"读出多少"
- $\Delta_t \in \mathbb{R}_{>0}$：控制状态更新的"速率"，大 $\Delta$ 表示对当前 token 反应剧烈，小 $\Delta$ 表示保持历史

这个机制让模型能"选择性遗忘"无关 token（比如 filler words），把容量留给关键 token。这是 SSM 接近 attention 表达力的关键。

### 2.3 Mamba block 的完整结构

Falcon Mamba 用的就是 Gu & Dao (2023) 的标准 Mamba block，配合一些 normalization 调整：

```
input x
  ↓
  ┌──── LayerNorm (pre-norm)
  ↓
  ┌──── Linear expand: d_model → E*d_model  (E=2)
  ↓
  Conv1d (causal, kernel=d_conv=4)
  ↓
  SiLU
  ↓
  ┌──── branch x: input-dependent projection: B_t, C_t, Δ_t
  │       ↓ Δ_t → softplus → discretize A, B
  │       ↓
  │   SSM core: h_t = Ā h_{t-1} + B̄_t * x_t ; y_t = C_t · h_t
  ↓
  ┌──── RMSNorm on B, C, Δ (Falcon Mamba 的稳定性改动)
  ↓
  Linear project back: E*d_model → d_model
  ↓
  + residual (post-norm RMSNorm outside block)
```

参考架构图可看 Mamba 原论文 Fig. 1：https://arxiv.org/abs/2312.00752

---

## 3. Falcon Mamba 的关键设计决策

### 3.1 Model parameters 详解

| 参数 | 值 | 直觉 |
|------|-----|------|
| n_layers | 64 | 深度足以匹配 7B Transformer 的 representation 容量 |
| d_model | 4096 | hidden width |
| expansion factor E | 2 | 内部 MLP width = 8192 |
| vocab_size | 65024 | 沿用 Falcon tokenizer |
| tied_embedding | False | **关键决策**：untie input/output embeddings |
| d_conv | 4 | causal conv kernel，捕捉 local pattern |
| Δ proj. size | 16 | 输入到 Δ 的投影维度 |
| state dim N | 16 | 每个 channel 的 SSM state 大小 |
| 总参数 | 7.27B | |

**为什么 untie embedding 重要？** 在 Transformer 中，tied embedding 通常作为正则化手段，但在 7B 规模上，untie 让 output head 有独立参数空间，能学到更精确的 distribution。Falcon 团队的实验显示这带来 measurable 的性能提升。

### 3.2 Normalization 的稳定性 hack

这是这篇 paper 的工程亮点之一。Mamba 在大 scale 训练中观察到 random loss spikes，且对 LR 极度敏感（Jamba、Mamba2 也报告了同样问题）。

Falcon 团队的解决方案：**在 B, C, Δ 三个输入依赖的 projection 之后各加一个 RMSNorm**：

$$
B_t = \text{RMSNorm}(\text{Linear}_B(x_t)), \quad C_t = \text{RMSNorm}(\text{Linear}_C(x_t))
$$

为什么这有效？我的理解：
- $B_t$ 进入递归 $h_t = \bar{A} h_{t-1} + \bar{B}_t x_t$，如果 $B_t$ 数值不稳定，会沿时间累积爆炸
- $C_t$ 直接乘 state 产生输出，量级不稳会让 logit 抖动
- $\Delta_t$ 通过 $\exp(\Delta A)$ 离散化，对量级尤其敏感——RMSNorm 把它约束在稳定区间

相比之下，Mamba2 把 RMSNorm 放在 block 输出 projection 之前，效果不如 Falcon 这种"局部约束"。

参考 Jamba 论文：https://arxiv.org/abs/2403.19887

### 3.3 为什么坚持 pure Mamba 而非 hybrid？

这是 paper 的核心 motivation。近期 hybrid 派（Jamba、Zamba、Samba）认为 attention 的 exact retrieval 能力 + SSM 的 efficient sequence mixing 是 best of both worlds。但 hybrid 破坏了 SSM 的 linear scaling，让 long-context 仍然受 quadratic attention 拖累。

Falcon 团队假设：**之前 pure Mamba 表现差，可能是 data mixture 和 training recipe 不够好，而不是 architecture 的根本限制**。结果证明这个假设大致成立——只在 in-context retrieval 的某些极端场景下仍有 gap（见 §4 Wen et al. 2024 的发现）。

---

## 4. Training Strategy 数学详解

### 4.1 WSD Learning Rate Schedule

WSD (Warmup-Stable-Decay) 是 miniCPM 提出的 schedule：

$$
\eta(t) = \begin{cases}
\text{linear ramp from 0 to } \eta_{\max} & t \in [0, t_{\text{warmup}}] \\
\eta_{\max} = 6.4 \times 10^{-4} & t \in [t_{\text{warmup}}, t_{\text{stable}}] \\
\eta_{\max} \exp\left[-\frac{t - t_{\text{stable}}}{t_{\text{decay}}} \log \frac{\eta_{\max}}{\eta_{\min}}\right] & t \in [t_{\text{stable}}, T]
\end{cases}
$$

- $\eta_{\max} = 6.4 \times 10^{-4}$：stable 阶段 LR
- $\eta_{\min} = \eta_{\max} / 256 = 2.5 \times 10^{-6}$：decay 末期 LR
- $t_{\text{warmup}} = 1\text{GT}$（GT = GigaTokens）
- $t_{\text{decay}}$：decay 阶段时长

关键发现：**Falcon 团队用约 10% 总 tokens 跑 decay，比 community 惯例长**。这解释了为什么 pre-decay checkpoint（57.29 avg on v1）和 final checkpoint（64.09 avg）有 6.8 分的飞跃——decay 阶段不是收尾，而是"知识 sharpening"。

参考 miniCPM：https://arxiv.org/abs/2404.06395

### 4.2 Batch Scaling：一个被低估的 trick

这是这篇 paper 最有 intuition value 的部分之一。

#### 背景：gradient noise temperature

Malladi et al. 2022 在 Adam 的 SDE 推导中得到一个重要观察：**Adam 的 effective noise temperature** 是：

$$
T_{\text{noise}} = \frac{\eta}{\sqrt{b}}
$$

其中 $\eta$ 是 LR，$b$ 是 batch size。这个量刻画了 SGD/Adam 的 stochastic dynamics 的"温度"。

直觉：增大 $b$ 等比例减小梯度方差，等价于降低温度；减小 $\eta$ 也降低温度。它们以 $\eta/\sqrt{b}$ 的形式耦合（因为 Adam 的二阶矩归一化消除了 $\sqrt{b}$ 因子的一部分）。

#### Batch size rampup 的副作用

Falcon 用 batch size rampup：从 $b_{\min}=128$ 线性增到 $b_{\max}=2048$，持续 50GT。这能前期利用小 batch 的 exploration 优势、后期利用大 batch 的 throughput。

**但有个隐藏代价**：rampup 期间 $T_{\text{noise}}$ 持续下降，意味着 stable 阶段的 effective noise 已经偏低了，进入 LR decay 阶段时温度进一步下降，decay 阶段本应提供的"sharpening"效果被削弱。

#### Batch scaling 的修复

Falcon 的做法：**保持 $T_{\text{noise}} = \eta/\sqrt{b}$ 在 rampup 完成后恒定**。即每次调 batch size 时同步调 LR：

$$
\eta \propto \sqrt{b}
$$

这与"LR 线性 scaling rule"（$\eta \propto b$，for SGD）不同，因为 Adam 的二阶矩已经吸收了一个 $\sqrt{b}$。

结果：decay 阶段 boost 更明显，final loss 更低。这一点我觉得非常 elegant——它把"调 batch size"和"调 LR"两个超参从 coupled 变成统一的 noise budget。

参考 Malladi et al. 2022：https://openreview.net/forum?id=F2mhzjHkQP

### 4.3 Optimizer 细节

- AdamW: $\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$
- weight decay = 0.1
- **没有用 Z-loss**（但 follow-up 实验显示 Z-loss on logits 能进一步稳定，与 Wortsman et al. 2024 一致）

Z-loss 的形式：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_z \cdot (\log Z)^2
$$

其中 $Z = \sum_i \exp(z_i)$ 是 logit partition function。这个正则项惩罚 logit 量级过大，对训练稳定性有帮助，对 Mamba 这种易 spike 的架构特别 relevant。

参考：https://openreview.net/forum?id=d8w0pmvXbZ

### 4.4 硬件与并行

- 256 × H100 80GB
- **纯 Data Parallelism (DP=256) + ZeRO**
- 没有 tensor parallel 或 pipeline parallel

这说明 7B Mamba 单卡放得下（含 optimizer state via ZeRO offload），DP 的通信开销低于 TP 的 all-reduce。这点对 SSM 来说容易，因为 SSM 的 state 远小于 KV cache。

---

## 5. Data Strategy 细节

### 5.1 数据来源

| 类别 | 来源 | 备注 |
|------|------|------|
| Web | RefinedWeb (Penedo et al. 2023) | 5T tokens 英文 web 数据 |
| Curated | books, arXiv, PubMed, USPTO, Reddit, StackExchange, Hackernews | 处理 conversation trees 时强制 causal temporality |
| Code | The Stack (Kocetkov et al. 2022) | 与 web 同 pipeline |
| Math | Proof-Pile-2 + FastText 过滤的 web math | |

**关键决策**：排除 multilingual data，因为 7B 容量不足以同时学好 English + multilingual。这点与 Llama3.1 等同时刷多语言的做法不同。

### 5.2 四阶段 curriculum + decay 阶段

| Stage | Seq len | LR | Data 倾向 |
|-------|---------|----|----|
| Stage 1 | 2048 | $\eta_{\max}$ | 基础 web 数据 |
| Stage 2 | ~4096 | $\eta_{\max}$ | 增加 code |
| Stage 3 | ~8192 | $\eta_{\max}$ | 增加 math、curated |
| Stage 4 | 8192 | $\eta_{\max}$ | 高质量、科学密集 |
| Decay | 8192 | $\eta_{\max} \to \eta_{\min}$ | Fineweb-Edu + Cosmopedia synthetic + 少量 multitask instruction (4 epochs, 3.7% 占比) |

**Decay 阶段掺入 instruction data** 是个微妙的决定——传统 base model 预训练严格不混 instruction，怕损害 fine-tuning flexibility。Falcon 团队的经验是：少量（3.7%）+ 少 epoch（4）能 boost in-context retrieval 能力，且不会 overfit instruction format。

这与 miniCPM 的发现一致，也是 community 逐渐接受的"co-training"思路。

### 5.3 与 Mamba in-context retrieval 弱点的关系

Wen et al. 2024 (https://arxiv.org/abs/2402.18510) 证明 RNN-style 架构在 in-context retrieval（如 copying task、induction head 任务）上有结构性瓶颈。Falcon 团队承认这是 limitation，但声称高质量 CoT instruction data 能"部分"mitigate。

具体机制猜测：CoT 数据让模型学会"显式复述"上下文，而非依赖隐式 attention-based retrieval——这绕过了架构限制，但代价是输出更长。

Arora et al. 2024 (https://arxiv.org/abs/2407.05483) 的 "Just Read Twice" 也是类似思路。

---

## 6. 评测结果深度分析

### 6.1 HF Leaderboard v1（Table 2）

| Model | ARC-25 | HellaSwag-10 | MMLU-5 | Winogrande-5 | TruthfulQA-0 | GSM8K-5 | Avg |
|-------|--------|--------------|--------|--------------|--------------|---------|-----|
| **FalconMamba-7B** | **62.03** | 80.82 | 62.11 | 73.64 | **53.42** | 52.54 | **64.09** |
| Gemma-7B | 61.09 | 82.20 | 64.56 | 79.01 | 44.79 | 50.87 | 63.75 |
| Mistral-Nemo-12B | 57.94 | 82.82 | 64.43 | 73.72 | 49.14 | 55.27 | 63.89 |
| Mistral-7B | 59.98 | 83.31 | 64.16 | 78.37 | 42.15 | 37.83 | 60.97 |
| Llama3.1-8B | 58.53 | 82.13 | 66.43 | 74.35 | 44.29 | 47.92 | 62.28 |
| Falcon2-11B | 59.73 | 82.91 | 58.37 | 78.30 | 52.56 | 53.83 | 64.28 |
| RecurrentGemma-9B (hybrid) | 52.00 | 80.40 | 60.50 | 73.60 | 38.60 | 42.60 | 57.95 |
| Zamba-7B (hybrid) | 56.14 | 82.23 | 58.11 | 79.87 | 52.88 | 30.78 | 60.00 |
| RWKV-v6-Finch-14B | 47.44 | 78.86 | 52.33 | 71.27 | 45.45 | 38.06 | 55.57 |
| mamba-7b-rw (pure SSM) | 51.25 | 80.85 | 33.41 | 71.11 | 32.08 | 4.70 | 45.52 |
| FalconMamba-7B (pre-decay) | 49.23 | 80.25 | 57.27 | 70.88 | 37.28 | 21.83 | 57.29 |

**观察 1**：pre-decay 到 final 的 GSM8K 跳跃 21.83 → 52.54，几乎 2.4x。这是 decay 阶段 sharpening 数学能力最直接的证据。

**观察 2**：与 mamba-7b-rw（45.52）对比，Falcon Mamba 高了 18.57 分。这说明同样架构、同样规模，pure data/recipe 优化能带来巨大差距。这是这篇 paper 最 powerful 的论点。

**观察 3**：Falcon Mamba 在 TruthfulQA 上 53.42 居首，明显超过所有 Transformer baseline。这有点 surprising——通常 TruthfulQA 需要良好的 in-context + knowledge recall，传统认为是 attention 的强项。可能解释：Falcon Mamba 训练数据中 math/curated 比例高，让模型学到更"事实导向"的 distribution，而不是靠架构优势。

### 6.2 HF Leaderboard v2（Table 3）

v2 难度更高（MMLU-Pro、GPQA、MuSR 等），分数普遍很低：

| Model | IFEval-0 | BBH-3 | Math-Lvl5-4 | GPQA-0 | MuSR-0 | MMLU-Pro-5 | Avg |
|-------|----------|-------|-------------|--------|--------|------------|-----|
| **FalconMamba-7B** | **33.36** | 19.88 | 3.63 | 8.05 | **10.86** | 14.47 | **15.04** |
| Gemma-7B | 26.59 | 21.12 | 6.42 | 4.92 | 10.98 | 21.64 | 15.28 |
| Mistral-7B | 23.86 | 22.02 | 2.49 | 5.59 | 10.68 | 22.36 | 14.50 |
| Llama3.1-8B | 12.70 | 25.29 | 4.61 | 6.15 | 8.98 | 24.95 | 13.78 |
| RecurrentGemma-9B | 30.76 | 14.80 | 4.83 | 4.70 | 6.60 | 17.88 | 13.20 |
| Zamba-7B | 24.06 | 21.12 | 3.32 | 3.03 | 7.74 | 16.02 | 12.55 |
| RWKV-v6-Finch-14B | 29.81 | 12.89 | 1.13 | 5.01 | 3.16 | 11.3 | 10.55 |

**观察 4**：Falcon Mamba 在 IFEval（instruction following）和 MuSR（multi-step soft reasoning，长上下文推理）上特别突出。MuSR 的高分与 Mamba 在 long-context reasoning 上的天然优势一致——SSM 把全部历史压进 fixed state，迫使模型学到"reasoning summarization"能力。

**观察 5**：BBH、MMLU-Pro 上略低于 Gemma / Llama3.1。这两类任务需要 broad knowledge recall + in-context pattern matching，正是 attention 的强项、Mamba 的弱项。

### 6.3 关键 take-away

- Pure Mamba 在 reasoning-dense 任务（MuSR、GSM8K、TruthfulQA）上可以胜过 Transformer
- Pure Mamba 在 retrieval-heavy 任务（MMLU-Pro、BBH）上有 gap
- Hybrid 设计（RecurrentGemma、Zamba）并未普遍优于 pure Mamba——Zamba 平均分 12.55 < Falcon Mamba 15.04

---

## 7. 推理效率：Prefill vs Decode 的关键区分

这一节是 paper 的另一核心贡献，且对部署 Mamba 极其实用。

### 7.1 Transformer 与 SSM 的内存模型对比

**Transformer (with KV cache)**：
- Prefill：$O(L^2)$ compute（attention），$O(L \cdot d)$ KV cache memory
- Decode：每 token 需 attend 到所有历史 KV，per-token 时间 $O(L \cdot d)$，内存 $O(L \cdot d)$ 随 $L$ 线性增长

**Mamba (SSM)**：
- Prefill（parallel）：parallel scan $O(L \log L)$ compute，但需存储所有 hidden states 用于 backward/backprop，内存 $O(L \cdot d)$
- Decode：每 token 只需 update 固定大小 state $h \in \mathbb{R}^{N \cdot d_{\text{inner}}}$，per-token 时间 $O(N \cdot d_{\text{inner}})$，**内存 $O(N \cdot d_{\text{inner}})$，与 $L$ 无关**

这是 Mamba 的真正卖点：**decode 阶段常数内存**。

### 7.2 Parallel Prefill vs Sequential Prefill

Falcon 团队指出一个被忽视的问题：**SSM 的 prefill 阶段内存仍随 $L$ 增长**（hidden states 必须保存）。

- **Parallel Prefill**：标准做法，整个 prompt 一次性 forward，最大化 GPU 利用，但内存 $O(L)$。Optimum-Benchmark 默认用这个。
- **Sequential Prefill**：token-by-token（或 chunk）处理 prompt，类似 sequence parallelism。Transformer 没好处（仍需 KV cache），但 SSM 可以这样"流式"输入，**内存 $O(1)$**，处理任意长 prompt。

### 7.3 实验数据（Fig. 2 & 3）

- **24GB A10 GPU** 上 max context length：
  - Llama3.1-8B / Mistral-7B / Qwen2-7B：在某个 $L$ 后 OOM（受 KV cache 拖累）
  - Falcon Mamba parallel prefill：能 fit 更长的 context（因为没 KV cache）
  - Falcon Mamba sequential prefill：**任意长度**，只受时间限制

- **80GB H100，prompt=1, generate up to 130k tokens**：
  - Mistral-7B：峰值内存线性增长，throughput 随 $L$ 下降
  - Falcon Mamba：throughput **常数**，峰值内存 **常数**

这个数据对实际部署意义巨大：长 output generation（如长文档摘要、代码生成、agent 长循环）场景，Mamba 是唯一可行架构。

### 7.4 一个 caveat

注意：训练时用的 context length = 8192，所以虽然推理能处理任意长输入，**模型本身的"long-context understanding 能力"是在 8k 上学到的**。paper 第 6 节明确承认这点，认为这是 future work。

---

## 8. Batched Generation 的工程细节

这一节有个很 elegant 的 trick。

### 8.1 SSM 的 padding 问题

Transformer 推理用 left padding，因为 attention mask 能屏蔽 padding。Mamba 的 SSM 是递归，**没有 mask 机制**——padding token 也会进入 state 累积：

$$
h_t = \bar{A} h_{t-1} + \bar{B}_t x_t
$$

如果 padding token（即使是 EOS-like）进入 $h_t$，会污染后续真实 token 的 state。

### 8.2 Falcon 的解决方案

三步处理：
1. **Left padding**（与 Transformer 一致）
2. **在 causal conv 之前** zero out padding token 的 hidden states
3. **在 causal conv 之后**再次 zero out

为什么 conv 前后都要 zero？因为 causal conv 是 $y_t = \sum_{k=0}^{d_{\text{conv}}-1} w_k \cdot x_{t-k}$，如果 padding 区有非零值，会 leak 到右边第一个真实 token。Zero 在 conv 前解决输入端 leak；zero 在 conv 后处理 conv 自身输出的边界效应。

这个 trick 让 Mamba 支持批量推理，对部署 critical。

---

## 9. 局限性与 Open Questions

paper 第 6 节诚实讨论了几个 limitation：

### 9.1 In-context retrieval gap

Wen et al. 2024 (https://arxiv.org/abs/2402.18510) 证明 RNN 在 retrieval 任务（如 induction head：找 [A][B]...[A] → 预测 [B]）有结构性瓶颈。Mamba 通过 selective state 缓解，但理论 capacity 仍受限于 $N \cdot d$。

Falcon 团队的 acknowledge：CoT data 能 mitigate 但不能完全 close gap。这是为什么 BBH、MMLU-Pro 仍略逊于 Transformer。

### 9.2 Long-context 训练未充分探索

虽然架构支持任意长 context，但实际训练上限 8192。**长 context 的 ability 必须在训练时建立**——这是 paper 最重要的 future direction。

### 9.3 Hybrid 是否仍是更优解？

paper 留了个开放问题：Falcon Mamba 证明 pure Mamba 不输 hybrid，但 hybrid 在 retrieval 任务上可能仍有优势。Optimal mixing ratio 仍是 open problem。

---

## 10. 我的几个额外直觉联想

### 10.1 为什么 pure Mamba 能追平 Transformer？

我认为关键不是 architecture 本身的等价性，而是：
1. **Data quality & mixture 优化空间仍远未饱和**——RefinedWeb + curated + Fineweb-Edu + Cosmopedia 这套 recipe 任何架构都能受益
2. **WSD + batch scaling + 长 decay** 这套 training recipe 真的 work，与架构耦合度低
3. **7B 这个规模，capacity 不算 bottleneck**，架构差异容易被 data 抹平

这意味着 paper 的核心贡献可能不只是 "pure Mamba work"，更是 "**pure data/recipe 优化能 compensate 架构劣势**"——这对 community 是个重要 reminder。

### 10.2 与 Mamba-2 的关系

Mamba-2 (Dao & Gu 2024) 通过 structured state space duality 把 SSM 和 attention 统一，理论上应该更优。但 Falcon Mamba 用的是 Mamba-1 架构。这有点意外——可能是 Mamba-2 的 SSD 实现在 7B 规模上没明显优势，或工程上 Mamba-1 更成熟。

参考 Mamba-2：https://arxiv.org/abs/2405.21060

### 10.3 与 Zamba 的对比

Zamba 7B (https://arxiv.org/abs/2405.16712) 在每 6 个 Mamba block 后插一个 shared attention。Falcon Mamba 在 v1 上 64.09 vs Zamba 60.00，**纯 Mamba 反而胜过 hybrid**。这间接说明 Zamba 的 attention 层可能没用足够数据训练好，或者 hybrid 的额外参数没被有效利用。

### 10.4 与 RecurrentGemma 的对比

RecurrentGemma 9B (https://arxiv.org/abs/2404.07839) 用 Griffin 架构（local attention + linear recurrent）。9B 比 7B 大，且是 hybrid，仍输给 Falcon Mamba（57.95 vs 64.09）。这说明 Griffin 的 hybrid 设计在这个规模上不如 well-tuned pure Mamba。

### 10.5 与 RWKV-v6 Finch 的对比

RWKV-v6 (https://arxiv.org/abs/2404.05892) 是另一种 attention-free 设计，用 matrix-valued state。14B 版本平均 55.57 仍输给 7B Falcon Mamba 64.09。这表明 Mamba 的 selective 机制在表达力上优于 RWKV 的 fixed-state recurrence。

---

## 11. 总结：这篇 paper 的真正贡献

按重要性排序：

1. **存在性证明**：pure Mamba 在 7B + 5.8T tokens 规模上能追平 Transformer SOTA。这推翻了"hybrid 是 SSM scale-up 唯一出路"的主流假设。

2. **Training recipe 的可迁移性**：WSD + batch scaling + 长 decay + 高质量 data mixture 是 architecture-agnostic 的。Mamba 团队的工程经验可被其他架构借鉴。

3. **Normalization hack**：B/C/Δ 上加 RMSNorm 是稳定 Mamba 训练的有效 trick。这对所有用 Mamba 的人有 practical value。

4. **Sequential prefill 的 explicit 讨论**：明确指出 SSM prefill 阶段的内存问题及解法，对 inference framework 开发者重要。

5. **Open release**：pre-decay checkpoint 也开源，让社区能 continual training，是少见的慷慨。

---

## 参考资料汇总

主要论文：
- Falcon Mamba：https://arxiv.org/abs/2410.05355
- Mamba 原始：https://arxiv.org/abs/2312.00752
- Mamba-2 (SSD)：https://arxiv.org/abs/2405.21060
- Jamba (hybrid)：https://arxiv.org/abs/2403.19887
- Zamba (hybrid)：https://arxiv.org/abs/2405.16712
- Samba (hybrid)：https://arxiv.org/abs/2406.07522
- Griffin / RecurrentGemma：https://arxiv.org/abs/2402.19427
- RWKV-v6 Eagle/Finch：https://arxiv.org/abs/2404.05892
- Mamba-7B-RW (TRI)：https://arxiv.org/abs/2405.06640
- Empirical study of Mamba-based LMs (Mamba2 8B hybrid/non-hybrid)：https://arxiv.org/abs/2406.07887
- Wen et al. "RNNs are not Transformers yet"：https://arxiv.org/abs/2402.18510
- Arora et al. "Just Read Twice"：https://arxiv.org/abs/2407.05483

训练相关：
- miniCPM (WSD schedule)：https://arxiv.org/abs/2404.06395
- Malladi et al. SDEs for adaptive gradient methods：https://openreview.net/forum?id=F2mhzjHkQP
- Wortsman et al. Z-loss & small-scale proxies：https://openreview.net/forum?id=d8w0pmvXbZ

数据集：
- RefinedWeb：https://arxiv.org/abs/2306.01116
- FineWeb-Edu：https://arxiv.org/abs/2406.17557
- Cosmopedia：https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- The Stack：https://arxiv.org/abs/2211.15533
- Proof-Pile-2 (Llemma)：https://arxiv.org/abs/2310.10631

Benchmarks：
- Open LLM Leaderboard v1：https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard
- Open LLM Leaderboard v2：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- lm-evaluation-harness：https://github.com/EleutherAI/lm-evaluation-harness
- lighteval：https://github.com/huggingface/lighteval

Code & Models：
- Falcon Mamba 7B on HF：https://huggingface.co/tiiuae/falcon-mamba-7b
- Transformers library：https://github.com/huggingface/transformers
- Optimum-Benchmark：https://github.com/huggingface/optimum-benchmark
- llama.cpp (Mamba support)：https://github.com/ggerganov/llama.cpp

Falcon 系列：
- Falcon2-11B：https://arxiv.org/abs/2407.14885
- Original Falcon：https://arxiv.org/abs/2311.16867

Transformer 基础：
- Attention is All You Need：https://arxiv.org/abs/1706.03762
- FlashAttention：https://arxiv.org/abs/2205.14135
- FlashAttention-2：https://arxiv.org/abs/2307.08691
- Longformer：https://arxiv.org/abs/2004.05150

---

如果你想让我深入任何部分（比如 parallel scan 算法、Mamba vs Mamba-2 的 SSD 视角、或 WSD schedule 的理论解释），告诉我，我可以再展开。
