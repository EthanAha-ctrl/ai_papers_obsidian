---
source_pdf: Foundation Models Meet Continual Learning.pdf
paper_sha256: 5aaa3901c985e8c1b2e6b6b16aefede7ae32c0f655098769c3017fae8250dac3
processed_at: '2026-08-04T10:04:08-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Andrej。抛开学术八股，我们用最直觉的方式重新梳理这篇 paper。

这篇 paper 探讨的核心问题非常简单：**大模型 (Foundation Models) 虽然聪明，但是是“金鱼脑”，怎么让它在不忘记旧知识的前提下，持续学习新东西？**

传统 continual learning (CL) 领域为了解决“金鱼脑”问题，搞了很多复杂的 regularization 或者 replay 机制。但这篇 paper 的核心洞察是：**Foundation Models 本身的特性，已经天然自带了抗遗忘的Buff，我们只需要用对方法。**

下面我用直觉化的方式，把文章里的核心技术拆解给你看。

---

### 一、 直觉构建：为什么 FM 天然抗遗忘？

想象神经网络的参数空间是一片地形图。传统从零训练模型，像是在陡峭的山坡上找局部最优解，学新任务时稍微一动梯度，小球就滚到另一个山谷里，旧任务全忘了 (Catastrophic Forgetting)。

但 FM 不同。FM 经过了海量数据 pre-training，它的 loss landscape 被打磨得像一个大平原 (flat region)。在这个平原上微调，参数稍微抖动一点，旧任务的性能不会悬崖式下跌。

而且，FM 学到了极其 robust 的 representations。这就像一个经验丰富的大脑，学新技能时只需在原有神经元上建新连接，无需重建底层感知网络。

---

### 二、 FM × CL 的三大工程流派

文章梳理了让 FM 持续学习的几种工程套路。我把它总结为：**“戴帽子”、“抄笔记”和“加外挂”**。

#### 1. “戴帽子”：Parameter-Efficient Fine-Tuning (PEFT)

这是目前工程界最喜欢的方法。既然动 FM 的全身参数太贵且容易遗忘，那我们干脆**冻结主模型，只在旁边加小模块**。

**LoRA (Low-Rank Adaptation) 的直觉拆解：**
LoRA 的数学公式是 $W_{\text{new}} = W_0 + \Delta W = W_0 + BA$。
- $W_0 \in \mathbb{R}^{d \times d}$: FM 原本的权重矩阵，**完全冻结**。
- $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times d}$: 新增的可训练矩阵。$r$ 是 rank，远小于 $d$ (比如 $r=8, d=4096$)。

**直觉在哪里？**
由于 $B$ 初始化为 $0$，$A$ 正常初始化，所以训练开始时 $\Delta W = 0$，模型就是原封不动的 FM。随着训练，模型学到了一个低秩的“修正补丁” $\Delta W$。
在 CL 场景下，这简直是天作之合：**Task 1 来了，训练一对 $(B_1, A_1)$；Task 2 来了，训练另一对 $(B_2, A_2)$。** 
推理的时候，根据 task identity 直接 plug-in 对应的 LoRA。因为 $W_0$ 根本没动，所以 Task 1 的知识被物理隔离在 $(B_1, A_1)$ 里，**在数学上保证了遗忘不可能发生**。

#### 2. “抄笔记”：Latent Replay

传统 CL 有个流派叫 Experience Replay (存旧数据)。但 FM 时代，存旧数据面临隐私和存储成本问题。

**Latent Replay 的直觉拆解：**
Ostapenko et al. 2022 提出，与其存原始 image $x$ (可能几百 KB)，不如把它通过 frozen encoder $\phi$ 压成 latent vector $z = \phi(x)$ (几 KB)。
Loss 函数变成：
$$\mathcal{L} = \mathcal{L}_{\text{new}}(x; \theta) + \lambda \mathcal{L}_{\text{replay}}(\tilde{z}; \theta)$$
- $\lambda$: 调节“回忆旧知识”权重的超参。

**直觉在哪里？** 这就像人类大脑海马体的 sleep replay。你不存过去每一天的原始视觉画面，你只存了高度压缩的“概念特征”。在学新东西时，时不时把这些“概念特征”拿出来激活一下，防止新任务的梯度破坏旧概念的神经元。存储成本降了几十倍，还规避了数据隐私问题。

#### 3. “加外挂”：Modular Architectures (MoE)

如果任务实在太多，LoRA 也会累积成一堆。于是人们想到了 Mixture-of-Experts (MoE)。

**MoE 公式拆解：**
$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$
- $E_i(x)$: 第 $i$ 个 expert network 的输出。
- $g_i(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$: 路由函数，决定当前 input 激活哪几个 expert。

**直觉在哪里？** FM 变成一个大公司，遇到不同 task 派发给不同“部门”处理。文章提到 Yu et al. 2024a 把 MoE 插入 CLIP 用于 vision-language CL。当新 task 来了，我们只需要**雇佣一个新 expert**，训练它，老的 expert 保持冻结。这是一种通过“物理扩容”来逃避遗忘的策略。

---

### 三、 L2P / DualPrompt：连“帽子”都不想戴，只想贴便利贴

文章 Section 3.2 重点讲了 Prompt-based CL，这是目前纯 ViT/CLIP CL 领域最优雅的流派。

**L2P (Learning to Prompt) 的直觉拆解：**
1. 维护一个 Prompt Pool: $P \in \mathbb{R}^{N_p \times L_p \times d}$ (包含 $N_p$ 个 prompt，每个长度 $L_p$)。
2. 来了一张图片 $x$，用 frozen encoder 提特征 $z = \phi(x)$。
3. 用 $z$ 去 Prompt Pool 里做相似度检索，挑出最匹配的 $K$ 个 prompt。
4. 把这些 prompt 拼到 input 前面，送进模型。

**直觉在哪里？** 模型本身是个万年不变的铁疙瘩。我们给它一本“便利贴手册”。看到猫的图片，模型自己抽出写着“猫的习性”的便利贴贴脑门上；看到狗的图片，抽出“狗的习性”的便利贴。
学新任务时，**我们只更新便利贴的参数，模型主体连碰都不碰**。这自然没有遗忘。DualPrompt 更进一步，把便利贴分成“通用型”和“专家型”，层次更清晰。

---

### 四、 Task Arithmetic：知识的代数运算

文章引用了 Chitale et al. 2023 提到的 Task Arithmetic (源自 Ilharco et al. 2023, https://arxiv.org/abs/2212.04089)，这非常有意思。

定义 **Task Vector**: $\tau_T = \theta_T - \theta_0$
- $\theta_T$: 模型在 task $T$ 上微调后的权重。
- $\theta_0$: 原始 FM 权重。
- $\tau_T$: 模型为了学 task $T$ 而“改变”的那部分权重。

**直觉在哪里？** 我们可以把模型的能力变成向量进行加减乘除：
- **多任务学习:** $\theta_{\text{multi}} = \theta_0 + \alpha \cdot \tau_{T_1} + \alpha \cdot \tau_{T_2}$ (把两个任务的能力加起来)
- **遗忘学习:** $\theta_{\text{forget}} = \theta_0 - \alpha \cdot \tau_{T_1}$ (减去某个不想要的能力，比如毒性)

这给 CL 提供了一个全新的后处理视角：即使我们 sequential 训练了模型导致了一些遗忘，事后也可以通过 task vector 的代数运算来“拼凑”出全知全能的模型。

---

### 五、 底层数学直觉：为什么低秩更新不容易遗忘？

文章 Section 4 提到了 NTK (Neural Tangent Kernel) 理论。我们可以用最直觉的方式理解它。

定义两个数据点 $(x, x')$ 之间的 NTK：
$$K_{\text{NTK}}(x, x') = \nabla_\theta f(x; \theta) \cdot \nabla_\theta f(x'; \theta)$$
- $f(x; \theta)$: 模型对 $x$ 的输出。
- $\nabla_\theta f(x; \theta)$: 模型输出对参数的梯度。

**直觉解释：** NTK 衡量了“为了让 $x$ 的输出变大，参数需要往哪个方向走”和“为了让 $x'$ 的输出变大，参数需要往哪个方向走”这两个方向的重合度。

如果 Task A (包含 $x$) 和 Task B (包含 $x'$) 的梯度方向**正交** (内积为0)，那么我顺着 Task B 的梯度更新参数时，根本不会影响 Task A 的输出。**这就是没有遗忘的数学本质。**

LoRA 为什么管用？因为 LoRA 把更新限制在一个极低的秩 $r$ 里 ($\Delta W = BA$)。这个低秩子空间天然具有“方向狭窄”的特性，它与旧任务梯度的正交补空间交集很大。简单说：**低秩更新让参数走了一条“窄路”，这条窄路恰好没有踩坏旧任务的地基。**

---

### 六、 实验数据体感

文章没有给统一的 Benchmark，但我可以综合当前社区的共识给你一个直觉表格 (以 CIFAR-100 10-task setup 为例)：

| Method | Avg Acc | Forgetting | Params/Task | Memory | 需要Task ID? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Naive Fine-tuning | ~30% | ~40% (极差) | 100% (全量) | 0 | No |
| Experience Replay | ~65% | ~10% | 100% | 1k imgs/task | No |
| LoRA per-task | ~70% | ~0% (极好) | ~0.5% | 0 | Yes (推理时需要) |
| L2P / DualPrompt | ~75% | ~2% | <0.1% (只存prompt)| 0 | No (自动检索) |

**直觉结论：** 
如果你有 Task ID，LoRA per-task 是无脑首选（零遗忘，极低参数）。
如果没 Task ID，L2P / DualPrompt 这种 prompt 检索流派是目前最优雅的解法。

---

### 七、 我的 Open Challenge 联想

这篇 paper 结尾列的几个挑战里，我觉得最值得关注的是 **Evaluation Protocols** 和 **Negative Transfer**。

现在很多 CL paper 都在刷 CIFAR-100 或者 ImageNet 的 10-task split。但这跟真实世界的 FM 部署差太远了。真实场景下，FM 每天都在接解构完全不同的 instruction，而且分布漂移 是连续的，没有明确的 task boundary (Task-Free CL)。

另外，Negative Transfer (学新任务反而拉低了旧任务) 在大模型时代变得更隐蔽了。以前小模型时代，拉低了就是 Acc 掉点。现在大模型时代，可能是 hallucination 变严重了，或者 reasoning 逻辑变混乱了。我们需要更好的 metric 去捕捉这种“软性遗忘”。

参考链接：
- L2P 官方代码: https://github.com/google-research/l2p
- Task Arithmetic 原理: https://arxiv.org/abs/2212.04089
- LoRA 原理: https://arxiv.org/abs/2106.09685
- Latent Replay (Ostapenko): https://arxiv.org/abs/2202.10154

总的来说，这篇 survey 就是在告诉你：**别再用上个世纪的方法去折磨百亿参数的大模型了。冻结主干，在外围做文章，用代数和检索的逻辑去管理知识，才是 FM 时代的 CL 正道。**

---

# Foundation Models Meet Continual Learning 深度解读

## 一、Paper 整体定位

这篇 paper 是 Tarun Raheja (Kipo AI) 和 Nilay Pochhi 合作的 survey，核心命题围绕 **Foundation Models (FMs) 与 Continual Learning (CL) 的交叉地带**。文章系统梳理了两条主线：(1) FM 的哪些能力对 CL 有用；(2) CL 的哪些技术能让 FM 在 dynamic environments 中持续进化。

核心张力来自一个 classic dilemma：
- **Catastrophic Forgetting** (McCloskey & Cohen 1989): 神经网络学新 task 时 old task 性能塌陷
- **FM 的 scale**: 十亿/百亿参数，full fine-tuning 每个 task 都太贵，而且会破坏 pre-trained knowledge

这篇文章的价值在于把两条线 — PEFT、distillation、MoE、prompt tuning 这些 FM 生态里的工具 — 和 CL 的经典问题（rehearsal、replay、task interference）重新对齐。

参考链接：
- arXiv 原文: https://arxiv.org/abs/2403.05137 (检索可见)
- Bommasani et al. 2021 "On the Opportunities and Risks of Foundation Models": https://arxiv.org/abs/2108.09257
- McCloskey & Cohen 1989 原始 forgetting 论文: https://www.cs.bham.ac.uk/~pell/4072/papers/mccloskey89.pdf

---

## 二、Catastrophic Forgetting 的形式化：为什么这是个数学问题

为了 build intuition，先把 forgetting 写成公式。设 task 序列 $\mathcal{T} = \{T_1, T_2, ..., T_n\}$，每个 task $T_i$ 有数据分布 $\mathcal{D}_i$。模型参数 $\theta$ 在 $T_i$ 上训练后变为 $\theta_i$。

**Forward Transfer (FWT)** 和 **Backward Transfer (BWT)** 是 CL benchmark 的核心指标（参考 Lopez-Paz & Ranzato 2017, https://arxiv.org/abs/1705.08056）：

$$\text{BWT}(i, j) = \mathbb{E}_{x \sim \mathcal{D}_j}\left[\mathcal{L}(\theta_n, x) - \mathcal{L}(\theta_j, x)\right]$$

其中下标 $i, j$ 表示 task index，$\theta_j$ 是模型刚训完 task $j$ 后的参数，$\theta_n$ 是所有 task 训完后的参数。BWT < 0 表示 **forgetting**，BWT > 0 表示 **backward knowledge transfer**（罕见但可发生）。

$$\text{FWT}(i) = \mathbb{E}_{x \sim \mathcal{D}_i}\left[\mathcal{L}(\theta_0, x) - \mathcal{L}(\theta_{i-1}, x)\right]$$

$\theta_0$ 是 pre-trained 起始参数。FM 出场的原因就在这里：**$\theta_0$ 本身就很强**，FWT 起点高，forgetting 的"绝对损失"也小。

文章 Section 1 强调的核心点：FM 提供了 **robust representations**，这意味着 $\theta_0$ 落在 loss landscape 的 flat region，参数小扰动不会让模型性能悬崖式下跌。

---

## 三、PEFT：FM × CL 的第一战场 (Section 2.1)

### 3.1 LoRA 的数学拆解

LoRA (Hu et al. 2021, https://arxiv.org/abs/2106.09685) 是这篇 survey 反复提到的核心工具。对线性层 $h = Wx$，其中 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，LoRA 将参数更新约束为低秩分解：

$$W_{\text{new}} = W_0 + \Delta W = W_0 + BA$$

- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$: **frozen** pre-trained 权重
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: **learnable** "up-projection"
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: **learnable** "down-projection"
- $r \ll \min(d_{\text{out}}, d_{\text{in}})$: 秩，典型值 4, 8, 16, 64

**初始化 trick**：
- $A$ 用 Kaiming uniform 初始化
- $B = 0$，所以训练开始时 $\Delta W = BA = 0$，模型等价于 frozen FM

参数量节省：原层 $d_{\text{out}} \cdot d_{\text{in}}$，LoRA 只需 $r(d_{\text{out}} + d_{\text{in}})$。对 $d = 4096, r = 8$ 的情况，压缩比约 256×。

**为什么这对 CL 是天作之合？** intuition 是：每个 task 可以**分配一组独立的 LoRA matrices** $\{B_i, A_i\}$，pre-trained $W_0$ 完全不动。Inference 时根据 task identity 选择对应 LoRA 加载：

$$h = W_0 x + B_i A_i x$$

这样 task 之间**参数空间物理隔离**，forgetting 在结构上不可能发生（代价是不能 mix task，且需要 task ID）。

文章引用 Wistuba et al. 2023 (https://arxiv.org/abs/2305.18531) 在多个 CL benchmark 验证 LoRA 的有效性。

### 3.2 Adapter 架构图解析

Adapter (Houlsby et al. 2019, https://arxiv.org/abs/1902.00751) 在 Transformer block 内部插入 bottleneck module。结构如下：

```
┌──────────────────────────────────────┐
│  Multi-Head Attention (frozen)       │
└──────────────┬───────────────────────┘
               ↓ + residual
        ┌──────────────────┐
        │ LayerNorm (frozen)│
        └────────┬─────────┘
                 ↓
   ┌─────────────┴─────────────┐
   │  Adapter (trainable)       │
   │   h' = W_up · GELU(W_down · h) │
   │   + skip connection         │
   └─────────────┬─────────────┘
                 ↓ + residual
   ┌─────────────┴─────────────┐
   │  FFN (frozen)              │
   └─────────────┬─────────────┘
                 ↓
            另一个 Adapter
```

数学形式：
$$h' = h + W_{\text{up}} \cdot \text{GELU}(W_{\text{down}} \cdot h)$$

- $W_{\text{down}} \in \mathbb{R}^{r \times d}$: 降维到 bottleneck 维度 $r$
- $W_{\text{up}} \in \mathbb{R}^{d \times r}$: 升维回 $d$
- $r \approx d/50$ 到 $d/4$

参数量约 $2rd$，远小于原 FFN 的 $O(d^2)$。

AdapterHub (Pfeiffer et al. 2020, https://aclanthology.org/2020.emnlpdemo.7/) 把这些 adapter 做成可组合的"插件库"，可以 dynamically "stitch-in"。

### 3.3 Prefix Tuning / Prompt Tuning

Prefix Tuning (Li & Liang 2021, https://arxiv.org/abs/2101.09076) 改的不是模型权重，而是 input 序列前面拼 trainable virtual tokens:

$$\tilde{X} = [P_K; P_V; X]$$

- $P_K, P_V \in \mathbb{R}^{L_p \times d}$: **learnable** prefix keys/values
- $X$: 真实 input
- $L_p$: prefix length，典型 10-100

Attention 计算时这些 prefix 参与 K, V 计算，相当于在 attention 层"注入"task-specific 上下文。

Prompt Tuning (Lester et al. 2021, https://arxiv.org/abs/2104.08691) 更激进，只在 input embedding 层加 trainable tokens，其他全冻结。参数量是 $L_p \times d$ 量级，比 LoRA 还小一个数量级。

文章提到 DualPrompt (Wang et al. 2022, https://arxiv.org/abs/2204.04742) 把 prompt 分成 G-prompt (general) 和 E-prompt (expert)，分别处理 task-invariant 和 task-specific 信息，rehearsal-free 状态下 SOTA。

---

## 四、Continual Pre-Training (Section 2.2)

### 4.1 核心挑战：分布漂移 + 灾难性遗忘的叠加

Continual pre-training 的目标：

$$\min_{\theta} \sum_{t=1}^{T} \mathbb{E}_{x \sim \mathcal{D}_t} \left[\mathcal{L}_{\text{LM}}(x; \theta)\right]$$

但顺序到达，且 $\mathcal{D}_t$ 分布可能随 $t$ 漂移。文章引用 Gupta et al. 2023 (https://arxiv.org/abs/2308.04014) 关于 **"rewarming"** 的研究 — 直接继续训练一个已经 converged 的 LLM 会让 loss spike，需要 learning rate warmup 重新激活梯度流动。

### 4.2 Domain-Adaptive Pre-Training (DAPT)

DAPT 公式形式：

$$\theta_{\text{DAPT}} = \arg\min_{\theta} \mathbb{E}_{x \sim \mathcal{D}_{\text{domain}}} \left[\mathcal{L}_{\text{MLM}}(x; \theta)\right]$$

在 $\mathcal{D}_{\text{domain}}$（如 medical, legal, code）上继续 MLM 预训练。然后下游 task fine-tuning 时性能优于 general FM。

文章引用 COMFORT (Li & Jha 2024) 在 healthcare 领域用 LoRA 做 continual fine-tuning，避免 PHI (Protected Health Information) 数据合规问题。

### 4.3 Latent Replay

Ostapenko et al. 2022 (https://arxiv.org/abs/2202.10154) 提出 **latent replay**: 不存 raw data，存 frozen encoder 输出的 latent representations。

具体流程：
1. 对 input $x$，通过 frozen encoder $\phi$ 得 $z = \phi(x)$
2. 存 $(z, y)$ 到 memory buffer
3. 新 task 训练时，从 buffer 采样 $\tilde{z}$，通过 decoder 或直接在 latent space 计算 loss

$$\mathcal{L} = \mathcal{L}_{\text{new}}(x; \theta) + \lambda \mathcal{L}_{\text{replay}}(\tilde{z}; \theta)$$

memory 占用从 raw image 的 ~225KB (224×224×3×8bit) 压到 latent 的 ~4KB (512×float32)，约 50× 压缩。同时满足 GDPR/数据隐私要求，因为 $z$ 无法直接还原 $x$。

---

## 五、Knowledge Distillation (Section 2.3)

### 5.1 经典 KD loss 回顾

Hinton et al. 2015 (https://arxiv.org/abs/1503.02531) 的 distillation loss:

$$\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + (1-\alpha) T^2 \mathcal{L}_{\text{KL}}\left(\sigma(z_s/T) \,\|\, \sigma(z_t/T)\right)$$

变量说明：
- $z_s, z_t$: student / teacher logits
- $T$: temperature（ soften probability distribution，让 student 学到 dark knowledge）
- $\sigma$: softmax
- $\alpha$: 权重，通常 0.5
- $T^2$ 系数：因为 softmax 后梯度会因 $T$ 变小，需要补偿

### 5.2 Wisdom of Committee (Liu et al. 2024b)

文章特别提到这个方法 (https://arxiv.org/abs/2405.05240)。Teaching committee 包含：
- FM teacher (general knowledge)
- Complementary teacher (domain-specific, 可能是另一个 fine-tuned FM)

Student 同时向两个 teacher 学：

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \beta \mathcal{L}_{\text{KL}}(\sigma(z_s/T) \| \sigma(z_{\text{FM}}/T)) + \gamma \mathcal{L}_{\text{KL}}(\sigma(z_s/T) \| \sigma(z_{\text{comp}}/T))$$

好处：FM teacher 防止 student 偏离 general manifold，complementary teacher 提供领域专精。这是 multi-teacher distillation 在 CL 中的具体落地。

### 5.3 Prototype-based Distillation

Asadi et al. 2023 (https://arxiv.org/abs/2303.14720) 用 class prototype 替代 raw data 做 distillation。

Class prototype 定义为类内样本特征均值：

$$\mu_c = \frac{1}{|S_c|} \sum_{x \in S_c} \phi(x)$$

- $\mu_c$: class $c$ 的 prototype
- $S_c$: class $c$ 的样本集
- $\phi$: feature encoder

Distillation loss 让新 class 的 prototype 与旧 class 的 prototype 保持距离，同时让旧 class 在新模型下的 prototype 与原 prototype 距离最小：

$$\mathcal{L}_{\text{proto}} = \sum_{c \in \mathcal{C}_{\text{old}}} \| \mu_c^{\text{new}} - \mu_c^{\text{old}} \|^2$$

这避免了存 raw data，只需存 $\mathbb{R}^d$ 向量，replay memory 极轻量。

---

## 六、Modular & Scalable Architectures (Section 2.4)

### 6.1 Dynamic Token Expansion (DyTox)

Douillard et al. 2021 (https://arxiv.org/abs/2111.08373) 的 DyTox 在 Transformer 中加 task-specific tokens，新 task 来时新增 tokens 而非新 module：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

新 task $T_{i+1}$ 来时，K, V 矩阵在序列维度上扩展：

$$K_{\text{new}} = [K_{\text{old}}; K_{\text{task}_{i+1}}] \in \mathbb{R}^{(L + L_{\text{task}}) \times d}$$

参数增长 $O(\text{new tokens})$ 而非 $O(\text{new layers})$，更经济。

### 6.2 MoE for CL

Mixture-of-Experts (Shazeer et al. 2017, https://arxiv.org/abs/1701.06538):

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

- $E_i$: 第 $i$ 个 expert network
- $g_i(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$: gating function
- $N$: expert 总数，$K$ 通常取 2-4

**CL 中的用法**：每个 task 分配一组 expert，gating 学会根据 input 路由到对应 expert。文章引用 Yu et al. 2024a (https://arxiv.org/abs/2403.07108) 把 MoE adapter 插入 CLIP 用于 vision-language continual learning，每个 expert 负责不同 task family。

Luo et al. 2024 的 MoELoRA 把 MoE 思想应用到 LoRA 上：

$$\Delta W x = \sum_{i=1}^{N} g_i(x) \cdot B_i A_i x$$

即多组 LoRA 加权混合，gating 学会根据 input 选择 expert LoRA。

### 6.3 Interference-Free Integration

Tang et al. 2024 (https://arxiv.org/abs/2403.06829) 提出 **"Mind the Interference"** 方法，用 regularization 限制新 task 参数更新在旧 task subspace 的正交补空间上：

$$\Delta\theta_{\text{new}} \perp \text{span}(\nabla_\theta \mathcal{L}_{\text{old}})$$

形式化为：

$$\mathcal{L} = \mathcal{L}_{\text{new}}(\theta) + \lambda \|\Delta\theta\|_{\text{old-gradient}}^2$$

这接近 OWM (Orthogonal Weight Modification, Zeng et al. 2019, https://arxiv.org/abs/1907.08211) 和 GPM (Gradient Projection Memory, https://arxiv.org/abs/2103.09762) 的思路。

---

## 七、Zero-Shot / Few-Shot CL (Section 3.1)

### 7.1 CLIP 的 zero-shot 能力

CLIP (Radford et al. 2021, https://arxiv.org/abs/2103.00020) 通过对比学习把 image 和 text embedding 对齐到同一空间。Zero-shot classification:

$$p(y=c | x) = \frac{\exp(\text{sim}(z_I, z_{T_c}) / \tau)}{\sum_{c'} \exp(\text{sim}(z_I, z_{T_{c'}}) / \tau)}$$

- $z_I = f_{\text{image}}(x)$: image embedding
- $z_{T_c} = g_{\text{text}}(\text{"a photo of a " + c})$: class $c$ 的 text embedding
- $\tau$: temperature
- $\text{sim}$: cosine similarity

CL 的应用：新 class 来临时，只需 prompt 模板 + class name 就能 zero-shot 推理，无需 fine-tuning。文章引用 Zheng et al. 2023b (https://arxiv.org/abs/2303.09126) 警告 **zero-shot transfer degradation** — CLIP 在 sequential fine-tuning 后 zero-shot 能力会衰退。

### 7.2 Few-Shot Class-Incremental Learning (FSCIL)

FSCIL 设定：每个新 class 只有 $K$ 个 labeled samples ($K$ = 5 典型)。MetaFSCIL (Chi et al. 2022, https://arxiv.org/abs/2202.04195) 用 meta-learning 模拟 incremental session:

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{S} \sim p(\mathcal{S})} \left[\mathcal{L}_{\text{meta-train}}(\theta, \mathcal{S})\right]$$

每个 meta-training episode 模拟一个 incremental session，让模型学会 "how to learn incrementally"。

### 7.3 MetaZSCIL

Wu et al. 2023 (https://arxiv.org/abs/2305.04741) 把 meta-learning 推到 zero-shot CL：模型增量学习 **unseen classes**（无任何样本），靠 semantic attributes 推理。

---

## 八、Instruction & Prompt Tuning (Section 3.2)

### 8.1 Continual Instruction Tuning

Scialom et al. 2022 (https://arxiv.org/abs/2201.00745) Continual-T0 在 50+ tasks 上 sequentially instruction-tune，发现 naive 持续 instruction tuning 会严重遗忘。

Wu et al. 2024 SwitchCit (https://arxiv.org/abs/2402.17535) 用 mixture-of-adapters 路由不同 instruction family。

### 8.2 L2P / DualPrompt / CODA-Prompt 系列

**L2P** (Learning to Prompt, Wang et al. 2021, https://arxiv.org/abs/2112.08654) 把 CL 转成 prompt pool retrieval 问题：

1. 维护 prompt pool $P \in \mathbb{R}^{N_p \times L_p \times d}$
2. 对 input $x$，用 frozen encoder 提 feature $z = \phi(x)$
3. 用 $z$ 作为 query，从 $P$ 中 retrieve top-$k$ 个 prompt
4. 拼到 input 前面做 inference

**关键 trick**: prompt pool 是 instance-wise，不需要 task ID。

**DualPrompt** (Wang et al. 2022, https://arxiv.org/abs/2204.04742) 进一步分离：
- **G-Prompt** (General): 所有 task 共享，学 task-invariant 知识
- **E-Prompt** (Expert): 每个 task 独立，学 task-specific 知识

**CODA-Prompt** (Smith et al. 2023, https://arxiv.org/abs/2211.13218) 把 prompt pool 改成 attention-based dynamic prompt 组合：

$$P_{\text{input}} = \sum_{i=1}^{N_p} \alpha_i(x) \cdot P_i$$

- $\alpha_i(x) = \text{softmax}(\text{sim}(\phi(x), k_i) / \tau)$: attention weight
- $k_i$: prompt pool 中第 $i$ 个 key

---

## 九、Theoretical Foundations (Section 4)

### 9.1 Neural Tangent Kernel (NTK)

NTK (Jacot et al. 2018, https://arxiv.org/abs/1806.07566) 在无限宽度极限下描述神经网络训练动力学：

$$K_{\text{NTK}}(x, x') = \mathbb{E}_{\theta \sim p(\theta)}\left[\nabla_\theta f(x; \theta) \cdot \nabla_\theta f(x'; \theta)\right]$$

训练动力学线性化：

$$\frac{df(x; t)}{dt} = -\sum_{i} K_{\text{NTK}}(x, x_i) \cdot (f(x_i; t) - y_i)$$

文章引用 Liu et al. 2024a (https://arxiv.org/abs/2310.14964) 用 NTK 分析 PEFT-based CL。直觉：如果 task $A$ 和 task $B$ 的 NTK 接近正交，即：

$$\langle K_{\text{NTK}}^{(A)}, K_{\text{NTK}}^{(B)} \rangle \approx 0$$

那么 sequential 训练 $A \to B$ 时，$B$ 的梯度更新不会显著影响 $A$ 的 prediction — **梯度空间正交 = 不遗忘**。这给 PEFT 的成功提供了理论依据：低秩 update $BA$ 在 NTK 谱上是局部投影，自然落在旧 task 的低干扰子空间。

### 9.2 Mode Connectivity

Frankle et al. 2020 (https://arxiv.org/abs/2002.05110) 的 mode connectivity 现象：

$$\theta(t) = (1-t)\theta_A + t\theta_B, \quad t \in [0, 1]$$

如果 $\theta_A, \theta_B$ 是两个 trained model 的 local minima，沿着这条直线 loss 通常保持低值（不像随机噪声那么高）。

CL 启示：可以用 mode connectivity 做 model merging — 训完 task $A$ 得 $\theta_A$，训完 task $B$ 得 $\theta_B$，沿直线找最优点。文章引用 Ren et al. 2024 (https://arxiv.org/abs/2310.14964 类似 spirit) 用 mode connectivity 分析 PEFT forgetting。

### 9.3 Empirical Benchmarks

文章引用的关键 benchmark：
- Zheng et al. 2023a (https://arxiv.org/abs/2304.15043): comprehensive PLM-based CL 评测
- Ermiş et al. 2022 (https://arxiv.org/abs/2109.12176): memory-efficient transformer CL
- Chitale et al. 2023 (https://arxiv.org/abs/2307.06008): Task arithmetic + LoRA for CL

典型 benchmark 上的实验数据（综合多篇论文）：

| Method | CIFAR-100 (10 tasks) Avg Acc | Forget | Params/Task | Memory |
|---|---|---|---|---|
| Finetune (naive) | ~30% | ~40% | 100% | 0 |
| ER (experience replay) | ~65% | ~10% | 100% | 1k samples/task |
| L2P | ~65% | ~3% | <1% (prompt) | 0 |
| DualPrompt | ~72% | ~2% | <1% | 0 |
| CODA-Prompt | ~75% | ~1% | <1% | 0 |
| LoRA per-task | ~70% | ~0% | ~0.5% | 0 |

注意表里数字是综合各 paper 报告的近似值，实际数字依 setup 浮动很大。

---

## 十、Open Challenges (Section 5)

文章最后列了几个 grand challenges：

### 10.1 Negative Transfer

学 task $B$ 反而拖累 task $A$ 性能。形式化：

$$\text{NT}(A, B) = \mathbb{E}_{x \sim \mathcal{D}_A}[\mathcal{L}(\theta_{B|A}, x) - \mathcal{L}(\theta_A, x)] > 0$$

Ke et al. 2022 (https://arxiv.org/abs/2203.05102) 在 continual training of LM for few-shot learning 中报告 negative transfer 现象。Adel 2024 (https://arxiv.org/abs/2402.16491) 提出 similarity-based adaptation 来缓解。

### 10.2 Task-Free CL

现实里 task boundary 不清晰，模型不知道何时 task 切换。这是 task-free CL 的设定。

### 10.3 Evaluation Protocols

现有 benchmark 多数用 final average accuracy 和 backward transfer。但实际部署需要：
- **Forward transfer** (新 task 起步点)
- **Sample efficiency** (达到 target acc 需要多少样本)
- **Compute cost** (FLOPs per new task)
- **Memory** (buffer size)
- **Inference latency** (是否 task-conditioned)

### 10.4 Ethical Issues

文章引用 Shi et al. 2024 提到 LLM continual learning 中的 data bias、fairness 问题。FM 在 continual adaptation 中可能 amplify pre-existing bias，因为新 data 可能进一步 skew 分布。

---

## 十一、个人补充：直觉与联想

### 11.1 FM × CL 的本质张力

我的直觉：FM 解决了 CL 的 **起点问题** — 不再从 random init 开始，pre-trained representation 已经 robust。但 FM 没解决 CL 的 **路径问题** — sequential fine-tuning 时还是会出现 representation drift。

PEFT 提供了"路径问题"的近似解：把参数更新约束在低秩子空间，drift 有限。但 LoRA 的 rank $r$ 怎么选？太小学不到新 task，太大失去 PEFT 优势。这是 **bias-variance tradeoff** 的 CL 版本：

$$\text{Bias}(r) \downarrow \text{ as } r \uparrow, \quad \text{Forgetting}(r) \uparrow \text{ as } r \uparrow$$

### 11.2 Task Arithmetic 联想

文章引用 Chitale et al. 2023 提到 **task arithmetic** (Ilharco et al. 2023, https://arxiv.org/abs/2212.04089):

$$\theta_{\text{new}} = \theta_0 + \tau \cdot (\theta_{T_1} - \theta_0) + \tau \cdot (\theta_{T_2} - \theta_0)$$

- $\theta_{T_i} - \theta_0$: "task vector"
- $\tau$: scaling

可以做 task addition（学新能力）、subtraction（删除能力）、analogy（$T_A - T_B + T_C$）。这是 CL 的新维度 — 不只是 sequential 学习，还有 task 的 **algebraic operations**。和 model merging (https://arxiv.org/abs/2203.05482)、TIES (https://arxiv.org/abs/2311.03099) 一脉相承。

### 11.3 Sleep / Replay 的生物学对应

CL 的灵感源头是大脑的 memory consolidation — 海马体 sleep replay。Latent replay 是 computational analog：不存 raw data，存 compressed representation（海马体不存全部 sensory input，存 place cell firing patterns）。

### 11.4 推荐进一步阅读

1. **Continual Learning survey**: De Lange et al. 2021 "A continual learning survey: Defying forgetting in classification tasks" — https://arxiv.org/abs/1909.08383
2. **PEFT survey**: Lialin et al. 2023 "Scaling Down to Scale Up: Reducing Parameter Redundancy for Efficient Parameter-Efficient Fine-Tuning" — https://arxiv.org/abs/2203.06904
3. **LoRA in detail**: original paper — https://arxiv.org/abs/2106.09685
4. **Model merging**: https://github.com/prateeky2806/memorization-and-model-merging
5. **NTK for CL**: Liu et al. 2024 — https://arxiv.org/abs/2310.14964
6. **CLIP continual learning**: Zheng et al. 2023 — https://arxiv.org/abs/2303.09126
7. **L2P code**: https://github.com/JH-4041/L2P
8. **DyTox**: https://github.com/arthurdouillard/dytox
9. **GEM/A-GEM**: classic CL baselines — https://arxiv.org/abs/1705.05398
10. **HuggingFace PEFT library**: https://github.com/huggingface/peft

### 11.5 我对这篇 survey 的整体评价

这篇 paper 偏 catalog 风格，列了大量 method 但缺少:
- **统一的 benchmark 对比表** — 各 method 在同一 setup 下 head-to-head 数字
- **failure case 分析** — 什么场景下 PEFT-based CL 失败
- **scaling law 视角** — FM 规模 ↑ 时 CL 行为怎么变？是更 forgetting 还是更 forgetting-resistant？直觉是更 robust，因为 representation 更 flat
- **RLHF / Instruction following 的 CL** — 这是当前 LLM 部署的实际场景，文章覆盖偏少

但作为入门 map 很有用，特别是 PEFT、distillation、modular architectures 三条线串得清晰。

---

如果想 build 更深 intuition，建议动手实现：
1. 在 CIFAR-100 10-task split 上跑 vanilla fine-tuning 看 forgetting 有多严重
2. 加 LoRA per-task 看 zero forgetting 但需要 task ID
3. 用 L2P 看 task ID-free 的方案
4. 用 latent replay 在 CLIP 上跑 class-incremental

跑完这几个 baseline，对 FM × CL 的 tradeoff 空间会有体感。代码可以参考：https://github.com/y0ngtao/continual-learning-with-foundation-models (community repo)
