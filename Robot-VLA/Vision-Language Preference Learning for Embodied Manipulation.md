---
source_pdf: Vision-Language Preference Learning for Embodied Manipulation.pdf
paper_sha256: 907e30ede52fdb492ea72747d476440c058b1deef5292d480d622d248121b29e
processed_at: '2026-08-13T01:36:58-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLP 的核心 intuition：用人话讲透技术细节

Andrej，如果我们用最朴素的话来聊这篇 paper，它的出发点其实非常接地气。

想象一下你正在训练一个 robot arm 去关抽屉。在传统的 Reinforcement Learning (RL) 里，你需要给 environment 写一个 reward function，比如“抽屉离关闭状态每近 1 厘米给 0.1 分”。这种手写 reward 极其容易被 robot 钻空子，比如 robot 发现疯狂震动手臂能让距离传感器误判，它就会一直震动。这就是 reward hacking。

为了绕开这个坑，大家开始用 Preference-based RL (RLHF)。与其算绝对分数，不如每次给 robot 看两段 video，告诉它“左边这段比右边好”。这种相对比较更符合人类直觉，也极难被 hack。但问题是，请人类专家去标注成千上万对 video 实在太贵、太慢了。

既然人贵，用 Vision-Language Model (VLM) 比如 CLIP 来自动打分行不行？之前的人试过，发现 CLIP 的 cosine similarity 当 reward 噪声极大，因为它只懂“画面里的像素和文字在表征上对不对得上”，它完全不懂“哪个动作更接近完成任务”。

VLP (Vision-Language Preference Learning) 的精髓就在于：**它不要求 VLM 直接输出 absolute reward，而是训练一个专门的 preference model，让它学习“在给定语言指令的条件下，哪个 video 更好”**。为了让这个 model 真正理解任务、语言与执行质量的关系，作者设计了三种绝妙的对比关系，并用极低的成本造出了训练数据。

下面我们把这套思路拆解成数据、架构、公式和实验数据表，看看它具体是怎么运作的。

---

## 1. 数据集 MTVLP：怎么不花钱造出 preference 标签

要训 model，先得有数据。VLP 最实用的工程贡献之一就是 MTVLP 数据集的构建 pipeline。它完全不需要人工标注，全靠 Meta-World benchmark 的 scripted policy 和 GPT-4V 自动生成。

作者把 trajectory 按照执行水平分成了三个等级：
1.  **Expert**：用 scripted policy 完美执行，只加一点 Gaussian noise $\mathcal{N}(0, 0.1)$ 模拟真实误差。
2.  **Medium**：用 scripted policy 执行，但一旦完成前一半 subtask（比如抓到了把手但还没拉开），就强制终止。Meta-World 内部有个 `near_object` flag，正好用来判定这个半成功状态。
3.  **Random**：动作直接从 uniform distribution $\mathcal{U}[0,1]$ 采样，完全瞎动。

为了增强语言泛化能力，作者用 GPT-4V 给每个任务生成 40 条变体语言指令。比如对于 Drawer Close 任务，生成了 "close the drawer", "shut the cabinet slider", "secure the storage bin" 等等。这就构成了一个包含 4800 条轨迹的庞大隐式 preference 数据集。
(参考 Meta-World: https://meta-world.github.io/)

---

## 2. 核心创新：三种 Language-Conditioned Preferences

VLP 的灵魂在于把“preference”拆解成了三种基于语言条件的关系。普通 RLHF 只有一种比较：同一任务下谁更好。这种死板的关系没法泛化。VLP 的设计如下：

| 类型 | Video 来源 | Language 来源 | 比较准则 |
|------|---------|---------|------|
| **ITP** (Intra-Task Preference) | 同任务 τ¹ 的两个 video $v_1^b, v_2^b$ | 同任务的语言 $l^b$ | 看哪个 video 完成度更高 |
| **ILP** (Inter-Language Preference) | 同任务 τ¹ 的两个 video $v_1^b, v_2^b$ | 异任务的语言 $l^{\neq b}$ | 两者 equally preferred (打平) |
| **IVP** (Inter-Video Preference) | 不同任务 $v_1^b$ vs $v^{\neq b}$ | 当前任务的语言 $l^b$ | 当前任务的 $v_1^b$ 赢 |

我的直觉理解：
*   **ITP** 是主任务，教 model 认识什么是“做得好”。
*   **ILP** 是关键的正则项。如果给 model 看关抽屉的 video，但问它“哪个把门关得更好”，model 必须学会输出平局（logit 差为 0）。这逼迫 model 去理解语言的语义，而不是盲目地把任何 video 往语言上靠。
*   **IVP** 是任务判别器。关抽屉的 random 轨迹，配上 "close the drawer" 的指令，它的 preference score 也必须高于关门的 expert 轨迹。这教会了 model “任务对齐”才是第一位的。

---

## 3. 架构解析：CLIP backbone + Cross-modal Transformer

VLP 没有从头训视觉编码器，那是算力黑洞。它的架构设计极其高效：

```text
Video v = {v_1, ..., v_8} (均匀抽取 8 帧)
    │  CLIP ViT-B/16 image encoder
    ▼
Video tokens z ∈ R^{M×D_v}, M = (H/p)·(W/p), p=16 patch size

Language l
    │  CLIP text encoder
    ▼
Language tokens u ∈ R^{N×D_l}

z ──self-attn×2──► z'
u ──self-attn×2──► u'

Cross-attn (Q=z', K=V=u')  → language-related video feature
Cross-attn (Q=u', K=V=z')  → video-related language feature

mean-pool → 拼接 → MLP (512, 256)
    ▼
f_ψ(v | l) ∈ R   ← trajectory-level preference score
```

设计要点：
1.  **Trajectory-level 输出**：它不预测 frame-level 的 reward，而是直接给整段 video 打一个标量分数 $f_\psi(v|l)$。这直接对接了 RLHF 里的 segment preference。
2.  **对称 Cross-attention**：video 特征吸收 language 信息，language 特征也吸收 video 信息，双向 bridge 比单向信息更丰富。
3.  **算力友好**：因为用了 CLIP 现成权重，整个 preference model 在单张 NVIDIA RTX 4090 上只需 6 小时即可训完。相比之下，R3M 或 LIV 需要在 Ego4D 或 EpicKitchen 上做大规模 pretraining，成本极高。
(参考 CLIP: https://arxiv.org/abs/2103.00020)
(参考 R3M: https://arxiv.org/abs/2303.00905)

---

## 4. 公式精读：Loss Function 怎么运作的

VLP 的 loss function 极其精巧，把上述三种关系统一在一个 Bradley-Terry 模型里。

首先，给定一个 language 条件 $l$，两个 video $v_1, v_2$ 的 preference 概率分布定义为：

$$
P_\psi[v_1 \succ v_2 \mid l] = \frac{\exp\left(f_\psi(v_1 \mid l)\right)}{\sum_{k=1}^{2} \exp\left(f_\psi(v_k \mid l)\right)} \tag{3}
$$

*   $f_\psi(v \mid l) \in \mathbb{R}$：VLP 模型输出的标量 preference score。
*   $l$：语言条件。
*   这个公式就是比较两个 video 在同一语言 $l$ 下的 softmax 概率。

接下来是总训练 loss：

$$
\mathcal{L}_{\infty} = -\sum_{b\in B}\Big[
\underbrace{\mathrm{CE}\big(P_\psi[v_1^b \succ v_2^b \mid l^b], y^{\mathrm{ITP}}\big)}_{(a)\ \text{ITP}}
+ \lambda_1 \underbrace{\mathrm{CE}\big(P_\psi[v_1^b \succ v_2^b \mid l^{\neq b}], y^{\mathrm{ILP}}\big)}_{(b)\ \text{ILP}}
+ \lambda_2 \underbrace{\mathrm{CE}\big(P_\psi[v_1^b \succ v^{\neq b} \mid l^b], y^{\mathrm{IVP}}\big)}_{(c)\ \text{IVP}}
+ \lambda_2 \underbrace{\mathrm{CE}\big(P_\psi[v_2^b \succ v^{\neq b} \mid l^b], y^{\mathrm{IVP}}\big)}_{(c)\ \text{IVP}}
\Big] \tag{4}
$$

变量和上下标解释：
*   $b$：minibatch 内的样本 index。
*   上标 $b$（如 $v_1^b, l^b$）：从当前采样任务 $\tau$ 中来的数据。
*   上标 $\neq b$（如 $v^{\neq b}, l^{\neq b}$）：从其他任务采的数据。
*   $y^{\mathrm{ITP}}, y^{\mathrm{ILP}}, y^{\mathrm{IVP}}$：三种 preference 的 ground truth label (0, 0.5, 1)。
*   $\lambda_1, \lambda_2$：平衡权重，论文中设 $\lambda_1=0.1, \lambda_2=0.5$。
*   $\mathrm{CE}$：cross-entropy loss。

直觉解读：
*   **(a) ITP 项**：在同任务同语言下，preference 由 optimality 决定。这是主任务。
*   **(b) ILP 项**：在同任务但异任务语言下，两条 video 应 equally preferred，$y^{\mathrm{ILP}}=0.5$。这逼迫 $f_\psi(v_1^b|l^{\neq b}) \approx f_\psi(v_2^b|l^{\neq b})$，即输出 logit 差为 0。这教 model 学会忽略不匹配的语言。
*   **(c) IVP 项**：当前任务 video vs 异任务 video，配当前任务语言。即使 $v_1^b$ 是 random 轨迹，它配同任务语言的 score 也必须高于异任务 video。这教会 model task-identity。

---

## 5. 实验数据表：VLP 到底有多强

### 5.1 VLP label vs Scripted label (Table 2)

| Task | P-IQL (scripted) | P-IQL+VLP | CPL (scripted) | CPL+VLP | VLP Acc. |
|---|---|---|---|---|---|
| Button Press | 72.6 | **90.1** | 74.5 | 83.9 | 93.0 |
| Door Close | 79.2 | 79.2 | 98.5 | 98.5 | 100.0 |
| Drawer Close | 49.3 | 64.9 | 45.6 | 75.5 | 96.0 |
| Faucet Close | 51.1 | 51.1 | 80.0 | 80.0 | 100.0 |
| Window Open | 62.4 | 69.7 | 91.6 | 99.1 | 98.0 |
| **Average** | 62.9 | 71.0 | 78.0 | 83.8 | 97.4 |

观察：VLP label 甚至在多数任务上超过了基于 ground-truth reward 推导出来的 scripted label。这说明 Meta-World 的 ground-truth reward 在某些任务上设计有缺陷，而 VLP 直接对“video 反映 language instruction 的程度”建模，反而更符合任务语义。

### 5.2 VLP 对比 VLM rewards (Table 5: Correlation with ground truth)

| Task | R3M | VIP | LIV | CLIP | VLM-RM(0.0) | VLM-RM(1.0) | VLP |
|---|---|---|---|---|---|---|---|
| Button Press | 0.313 | 0.204 | -0.281 | 0.127 | 0.153 | -0.082 | **0.581** |
| Door Close | 0.735 | 0.125 | 0.600 | -0.309 | -0.152 | -0.492 | **1.000** |
| Drawer Close | -0.106 | 0.043 | 0.052 | -0.151 | -0.137 | -0.031 | **0.438** |
| Faucet Close | 0.676 | 0.851 | 0.563 | -0.301 | -0.291 | 0.084 | **1.000** |
| Window Open | 0.411 | 0.725 | -0.568 | 0.336 | 0.405 | -0.333 | **0.571** |
| **Average** | 0.406 | 0.390 | 0.073 | -0.060 | -0.005 | -0.171 | **0.718** |

观察：直接用 CLIP 做 zero-shot reward，correlation 接近于 0 甚至负数。R3M、VIP 这种预训练表征也只有 0.3-0.4 的 correlation。VLP 的 0.718 断层式领先。这证明了 preference 形式 + 三关系监督的绝对优势。

---

## 6. 理论联系：为什么 VLP 类似 "Negative Regret"

论文提到 VLP 在某种程度上逼近 "negative regret"。
*   Regret $\text{Regret}(\sigma) = V^* - \sum_{t} r(s_t,a_t)$，即 segment 相对最优策略的 return 差距。
*   如果 $f_\psi(v|l)$ 学到的是“该 segment 在语言 $l$ 描述的任务下的相对优劣”，并且最优轨迹 score 最高，那么 $f_\psi$ 实际上编码了 $-\text{Regret}$。

直觉上，preference model 实际上学到的是 "log-return gap"：
$$f_\psi(v_1|l) - f_\psi(v_2|l) \approx \beta (J(v_1|l) - J(v_2|l))$$
其中 $\beta$ 是温度，$J$ 是真实 expected return。这正是 DPO 在 LLM 中证明的隐式 reward 等价关系在 video domain 的对应物。
(参考 DPO: https://arxiv.org/abs/2305.18290)

---

## 7. 局限性与我想到的 Future Work

1.  **Trajectory-level vs Step-level**：VLP 只给 segment preference，对需要 fine-grained credit assignment 的任务（如 in-hand manipulation）可能力不从心。结合 Preference Transformer 的 attention aggregation 做帧级 reward 分配是个好方向。
2.  **Medium level 定义的依赖**：MTVLP 数据集的构建依赖 scripted policy 的 subtask flag (`near_object`)。真实世界里没有这种 flag，如何用 video diffusion model 生成 medium trajectory，或者用 unsupervised 方式聚类，是推广到 real-world 的瓶颈。
3.  **Language encoder 的语义理解**：CLIP text encoder 是 bag-of-phrases 级别，对复杂时序逻辑（“先开抽屉，等三秒，再放杯子”）理解有限。可以探索用 LLM-based text encoder 替换。
4.  **Self-supervised bootstrap**：当前依赖 scripted policy。如果用 RL agent 自己探索的轨迹 + 自一致 preference 训练，类似 self-rewarding language model，能否实现完全无监督的 preference learning？

VLP 这篇 paper 最 beautiful 的地方在于它的 ITP/ILP/IVP 三角约束。少一种，alignment 就不完整。它用极低的计算成本，把 VLM 从不可靠的 absolute reward predictor，变成了极其可靠的 relative preference annotator。这个思路在 embodied AI 领域有极大的启发意义。

---

# VLP: Vision-Language Preference Learning for Embodied Manipulation 深度讲解

## 1. 论文要解决的根本痛点

传统的 Reinforcement Learning (RL) 在 embodied manipulation 场景下被 reward engineering 卡住：手写 reward 容易被 agent hack，专家 demonstration 又贵。Preference-based RL（如 Christiano 2017 的 RLHF 原型）试图用 human preference 标注替代 reward，但标注成本依然高昂——线上 query 专家太慢，离线数据集又得人工标。最近的工作尝试用 VLM 直接 zero-shot 给 reward（CLIP、VLM-RM、RoboCLIP），但 reward 噪声大、方差高，因为 VLM 并没有真正学到"轨迹间相对优劣"这一关系。

VLP 的核心立意：**与其让 VLM 当 reward predictor，不如训练一个专门学"相对优劣 + 语言条件"的 preference model**。这个 model 一旦学好，就可以像 oracle 一样给下游任意 RLHF 算法做 annotator，并且能泛化到 unseen task、unseen language。

参考链接：
- Christiano et al. 2017, Deep RL from Human Preferences: https://arxiv.org/abs/1706.03741
- Meta-World benchmark: https://meta-world.github.io/
- CLIP: https://arxiv.org/abs/2103.00020

---

## 2. 关键 Intuition：把 preference 重新定义成"语言条件下的三角约束"

普通 RLHF 只定义了一种关系：同一任务下 σ¹ vs σ² 谁更好。这种关系是 task-bound、rigid 的，所以训出来的 reward model 没法泛化到新任务。

VLP 的精彩之处在于把"preference"拆成三种 language-conditioned 关系，构成一个三角形约束体系：

| 类型 | 视频来源 | 语言来源 | 准则 |
|------|---------|---------|------|
| ITP (Intra-Task Preference) | 同任务 τ¹ 的两个 video | 同任务的语言 l¹ | 看哪个 video 更符合 l¹（optimality） |
| ILP (Inter-Language Preference) | 同任务 τ¹ 的两个 video | 异任务的语言 l² | 二者 equally preferred（0.5） |
| IVP (Inter-Video Preference) | 不同任务 τ¹ vs τ² 各一个 video | τ¹ 的语言 l¹ | τ¹ 的 video 被 preferred |

这三个关系组合起来，等价于在隐空间里同时约束三件事：

1. **判别 optimality**：同一任务下要能区分 expert / medium / random 轨迹。
2. **language-invariance**：当语言和 video 不匹配时，输出应当退化为均匀分布（logit 差为 0），这逼着模型学到"忽略不相关语言"，而不是盲目地把 video 拉到任意 language 的 embedding 附近。
3. **task-identity / cross-modal alignment**：来自任务 τ¹ 的 video 配上 l¹ 的 score 要高于配上 τ² 的 video——这是最关键的 alignment 监督，让 model 学到"video 究竟描述了哪个任务"。

直觉上：CLIP 学的是 image-text 双塔的 cosine 相似度，是 absolute alignment；而 VLP 学的是 **conditional relative alignment**——给定语言 l，比较两个 video 谁更贴合 l。这种 relative form 天然抗噪声（因为只需要排序，不需要绝对标定），并且和 Bradley-Terry preference 模型兼容。

我的一个额外直觉：这三种关系本质上对应 metric learning 的三种 contrastive 结构——ITP 像 hard positive/negative within class，ILP 像 "ignore label" 的对称化，IVP 像 class-discriminative contrastive。所以 VLP 可以理解为 **video-text 的 supervised contrastive learning 用 preference 损失重写一遍**。

---

## 3. 数据集 MTVLP：怎么"免费"造出 preference 标签

这是论文最实用的工程贡献之一。它绕开了"人工标注"这个瓶颈，方法是利用 Meta-World 自带的 scripted policy + GPT-4V 数据增强：

- **50 个 Meta-World 任务**，其中 45 个训练、5 个测试（Button Press / Door Close / Drawer Close / Faucet Close / Window Open）。
- 每个任务用 scripted policy 滚出 3 个 optimality 级别：
  - **Expert**：scripted policy + Gaussian noise N(0, 0.1)。
  - **Medium**：scripted policy 但在完成"前一半 subtask"时终止。判定 subtask 用 Meta-World 内置的 `near_object` flag——这一步很关键，因为它给出了一种"半成功"的可识别锚点，否则 medium level 没法定义。
  - **Random**：动作从 U[0,1] 采样。
- 每种 32 条 → 每任务 96 条 → 总共 4800 条轨迹。
- **Language augmentation**：用 GPT-4V 给每个任务生成 40 条变体，通过两类变换：
  1. verb structure 变换（"close the drawer" / "shut the drawer" / "shift the drawer closed"…）。
  2. synonym noun 替换（"drawer" / "cabinet slider" / "storage bin"…）。

这样同一 video 配多条语言，自然产生 ILP 的训练样本（同一个 video 跨多种语言）。重要的是：**所有 preference label 都是基于 optimality 等级和 task identity 自动推导出来的，不需要人工比较两条轨迹**。Table 16 还显示用 GPT-3.5、Llama-3.1-8B 替代 GPT-4V 生成语言也几乎不掉点（97.0% vs 97.4%），说明这套数据 pipeline 不依赖顶级 LLM。

参考：
- LAMP (Adeniji et al. 2023) 语言增广思想来源: https://arxiv.org/abs/2308.12270
- Meta-World scripted policy: https://github.com/rlworkgroup/metaworld

---

## 4. 模型架构解析

```
Video v = {v_1, ..., v_|v|}     (|v|=8 frames, 每个 v_i ∈ R^{H×W×3})
       │  CLIP ViT-B/16 image encoder
       ▼
Video tokens z ∈ R^{M×D_v},  M = (H/p)·(W/p),  p=16 patch size, D_v=512

Language l
       │  CLIP text encoder
       ▼
Language tokens u ∈ R^{N×D_l},  N = token 数, D_l=512

z ──self-attn×2──► z'
u ──self-attn×2──► u'

Cross-attn (Q=z', K=V=u')  → language-related video feature
Cross-attn (Q=u', K=V=z')  → video-related language feature

mean-pool along 序列维 → 拼接 → w ∈ R^{D_v+D_l=1024}
       │  MLP (512, 256)
       ▼
f_ψ(v | l) ∈ R   ← trajectory-level preference score
```

设计上有几个值得注意的点：

1. **Video 和 language encoder 都用 CLIP 预训练权重**，避免从头训视觉编码器，节省大量算力（论文强调"6 小时一张 4090 训完"，对比 LIV/VIP 在 EpicKitchen / Ego4D 上的大规模 pretraining）。
2. **Cross-modal transformer 用对称 cross-attention**：既让 video feature 吸收 language 信息，也让 language feature 吸收 video 信息，这是双向 bridge，比单向 attention 信息更丰富。最后两个 modality 的 pooled feature 都拼进 MLP，避免只用 video-side feature 丢掉 language-side 的对齐信号。
3. **Preference score 是 trajectory-level 标量**，不是 frame-level reward。这一点和 Preference Transformer (Kim et al. 2023) 不同，PT 是 frame-level reward。Trajectory-level 的好处是直接对应 RLHF 里的 segment preference，坏处是无法直接给到 fine-grained step-level reward，但论文证明对 P-IQL/IPL/CPL 这类算法已经够用。
4. **8 帧均匀采样**：节省算力且足以表达 trajectory 语义。这是一个工程简化，可能与长程任务的时序建模能力有 trade-off。

附录 B 的 attention map 可视化（Figure 4）显示：cross-attention 在 Drawer Close 任务上集中在 drawer 边缘、Door Close 上集中在 door handle，说明 model 确实学到了"language 提到的物体在 video 哪里"。

参考：
- Preference Transformer: https://arxiv.org/abs/2305.01446
- LIV: https://arxiv.org/abs/2306.13995
- VIP: https://arxiv.org/abs/2210.03090

---

## 5. 公式精读

### Eq. (1) Bradley-Terry preference predictor（传统 RLHF）

$$
P_\psi[\sigma^1 \succ \sigma^2] = \frac{\exp\left(\sum_{t=1}^{H} \hat{r}_\psi(s_t^1, a_t^1)\right)}{\sum_{k=1}^{2} \exp\left(\sum_{t=1}^{H} \hat{r}_\psi(s_t^k, a_t^k)\right)}
$$

- $\sigma^1, \sigma^2$：两个 segment，长度为 H。
- $s_t^k, a_t^k$：第 k 条 segment 第 t 步的 state 和 action。
- $\hat{r}_\psi$：参数为 ψ 的 reward model。
- $\sum_{t=1}^{H} \hat{r}_\psi(\cdot)$：segment 累积 reward（return）。
- 分子是 σ¹ 的 return 的 softmax 分子；分母是两条 segment 的 return 的 softmax 和。
- 这是经典 Bradley-Terry (1952) model 的应用：把"return 高的更可能被 prefer"参数化。

### Eq. (2) Cross-entropy reward learning

$$
\mathcal{L}_{ce} = -\mathbb{E}_{(\sigma^1,\sigma^2,y)\sim\mathcal{D}}\left[(1-y)\log P_\psi[\sigma^1 \succ \sigma^2] + y\log P_\psi[\sigma^2 \succ \sigma^1]\right]
$$

- $y \in \{0, 0.5, 1\}$：标签。0 = σ¹ 更好，1 = σ² 更好，0.5 = 同等。
- 当 y=0.5 时，两项各占一半权重，相当于鼓励模型对两者输出相近概率。
- 这正是 Christiano 2017 RLHF 训 reward model 的 loss。

### Eq. (3) VLP 的语言条件 preference 分布

$$
P_\psi[v_1 \succ v_2 \mid l] = \frac{\exp\left(f_\psi(v_1 \mid l)\right)}{\sum_{k=1}^{2} \exp\left(f_\psi(v_k \mid l)\right)}
$$

- $f_\psi(v \mid l) \in \mathbb{R}$：新引入的 preference score function，输入 video + language，输出标量。注意它**不是 reward**，是直接的 preference score。
- $l$ 是 condition（语言指令）。整式表示"在语言 l 条件下，v₁ 比 v₂ 更被 prefer 的概率"。
- 这把 reward model 重写成了一个 **direct preference model**，类似于 DPO 的思路——跳过显式 reward 直接学 preference logit。相关思想可参考 Zhang et al. 2024 (FTB) 的论证。

### Eq. (4) VLP 总训练 loss

$$
\mathcal{L}_{\infty} = -\sum_{b\in B}\Big[
\underbrace{\mathrm{CE}\big(P_\psi[v_1^b \succ v_2^b \mid l^b], y^{\mathrm{ITP}}\big)}_{(a)\ \text{ITP}}
+ \lambda_1 \underbrace{\mathrm{CE}\big(P_\psi[v_1^b \succ v_2^b \mid l^{\neq b}], y^{\mathrm{ILP}}\big)}_{(b)\ \text{ILP}}
+ \lambda_2 \underbrace{\mathrm{CE}\big(P_\psi[v_1^b \succ v^{\neq b} \mid l^b], y^{\mathrm{IVP}}\big)}_{(c)\ \text{IVP}}
+ \lambda_2 \underbrace{\mathrm{CE}\big(P_\psi[v_2^b \succ v^{\neq b} \mid l^b], y^{\mathrm{IVP}}\big)}_{(c)\ \text{IVP}}
\Big]
$$

变量解释：

- $b$：minibatch 内的样本 index。
- 上标 $b$：从当前采样任务 $\tau$ 中来的数据（同任务内）。
- 上标 $\neq b$：从其他任务中采的数据。
- $v_1^b, v_2^b$：当前任务的两个 video。
- $v^{\neq b}$：其他任务的 video。
- $l^b$：当前任务的语言。
- $l^{\neq b}$：其他任务的语言。
- $y^{\mathrm{ITP}}, y^{\mathrm{ILP}}, y^{\mathrm{IVP}}$：三种 preference 的 ground truth label。
- $\lambda_1=0.1$：ILP 权重。
- $\lambda_2=0.5$：IVP 权重。
- $\mathrm{CE}$：cross-entropy。

直觉解读：

- **(a) ITP 项**：在同任务同语言下，preference 由 optimality 决定——专家 > medium > random。这是主任务，没有权重加成。
- **(b) ILP 项**：在同任务但**异任务语言**下，两条 video 应该 equally preferred，即 $y^{\mathrm{ILP}}=0.5$，CE 会鼓励 $P_\psi \approx 0.5$，即 $f_\psi(v_1^b|l^{\neq b}) \approx f_\psi(v_2^b|l^{\neq b})$。这等价于"语言不匹配 → 输出 logit 差为 0"的正则。
- **(c) IVP 项**：当前任务 video vs 异任务 video，配当前任务语言 → 当前任务 video 应当 win。注意这里**有两条**：$v_1^b \succ v^{\neq b}$ 和 $v_2^b \succ v^{\neq b}$，即无论当前任务用哪条 video（即使是 random 轨迹），它配同任务语言的 score 都要高于异任务 video。这一项是真正"教会"模型 task-identity 的。

Table 13 的消融很有说服力：$\lambda_2=0$ 时 IVP Acc 从 91.7 掉到 63.0，ILP Loss 也涨到 0.775——没有 IVP 项，整个语言条件结构就崩了，模型退化成"无语言条件的 vanilla preference model"。

---

## 6. 理论联系：为什么 preference model 类似 "negative regret"

论文提到 "the learned preference model resembles the negative regret of the segment under mild conditions"。直觉是：

- Regret $\text{Regret}(\sigma) = V^* - \sum_{t} r(s_t,a_t)$，即 segment 相对最优策略的 return 差距。
- 如果 $f_\psi(v|l)$ 学到的是"该 segment 在语言 l 描述的任务下的相对优劣"，并且最优轨迹 score 最高，那么 $f_\psi$ 实际上编码了 $-\text{Regret}$（或单调相关）。
- 关键的"mild condition"是：训练数据中 optimality 分层与真实 regret 单调对应，并且 language condition 足够 informative 来锁定 task。这两个条件在 MTVLP 数据集中都满足：expert/medium/random 由 scripted policy 自然分层，GPT-4V 生成的语言对每个任务是 informative 的。
- 这种联系解释了为什么 VLP label 可以直接当 reward 信号使用：对 P-IQL，可以把 $f_\psi$ 看作 implicit reward；对 CPL/IPL 这类 direct preference learning 方法，直接用 $f_\psi$ 计算 contrastive 优势即可。

更深一层的直觉：preference model 实际上学到的是 "log-return gap"，即 $f_\psi(v_1|l) - f_\psi(v_2|l) \approx \beta (J(v_1|l) - J(v_2|l))$，其中 $\beta$ 是温度、$J$ 是真实 expected return。这正是 DPO 在 LLM 中证明的隐式 reward 等价关系在 video domain 的对应物。

---

## 7. 实验数据详解

### 7.1 Q1: VLP label vs scripted label（Table 2）

5 个 test task，5 seeds：

| Task | P-IQL (scripted) | P-IQL+VLP | IPL (scripted) | IPL+VLP | CPL (scripted) | CPL+VLP | VLP Acc |
|---|---|---|---|---|---|---|---|
| Button Press | 72.6 | **90.1** | 50.6 | 56.0 | 74.5 | 83.9 | 93.0 |
| Door Close | 79.2 | 79.2 | 61.5 | 61.5 | 98.5 | 98.5 | 100.0 |
| Drawer Close | 49.3 | 64.9 | 64.3 | 63.2 | 45.6 | 75.5 | 96.0 |
| Faucet Close | 51.1 | 51.1 | 45.4 | 45.4 | 80.0 | 80.0 | 100.0 |
| Window Open | 62.4 | 69.7 | 54.1 | 61.4 | 91.6 | 99.1 | 98.0 |
| **Avg** | 62.9 | 71.0 | 55.2 | 57.5 | 78.0 | 83.8 | 97.4 |

观察：

- VLP label 在所有方法上**至少持平、多数超过 scripted label**。这是反直觉的——scripted label 是 ground-truth reward 推导的"理想标签"，VLP 用学习出来的 model 给标签，怎么会更好？
- 论文给的假说：Meta-World 的 ground-truth reward 在某些任务上不能准确反映 task goal（Xie 2024 / Ma 2024 / Sun 2024a 等指出 scripted reward 可能 sparse 或被 hack）。VLP 是直接对"video 反映 language instruction 的程度"建模，反而捕捉到 task 语义。
- 我的额外解读：VLP 是 trajectory-level 相对标签，天然是 ranked；scripted reward 是 dense reward 转成的 segment return，可能 reward shaping 引入 noise。两者误差来源不同，VLP 的 error 是"任务理解错"（罕见），scripted 的 error 是"reward 函数本身设计错"（可能）。

### 7.2 Q2: 对比 VLM rewards（Table 3, 4, 5）

Table 3 (用 VLM reward 直接训 IQL，VLP 用 P-IQL+VLP label)：

| Task | R3M | VIP | LIV | CLIP | VLM-RM(0.0) | VLM-RM(1.0) | VLP |
|---|---|---|---|---|---|---|---|
| Button Press | 10.1 | 68.4 | 56.3 | 59.5 | 60.3 | 64.3 | **90.1** |
| Door Close | 70.9 | 74.8 | 43.3 | 43.6 | 45.8 | 41.1 | **79.2** |
| Drawer Close | 46.6 | 70.4 | 61.8 | 69.4 | 69.4 | 73.5 | 64.9 |
| Faucet Close | 25.7 | 40.9 | 42.2 | 59.6 | 60.1 | 33.7 | 51.1 |
| Window Open | 39.0 | 42.7 | 33.8 | 26.4 | 23.9 | 23.7 | **69.7** |
| Avg | 38.5 | 59.4 | 47.5 | 51.7 | 51.9 | 47.3 | **71.0** |

Table 5 (correlation with ground truth)：

| Task | R3M | VIP | LIV | CLIP | VLM-RM(0.0) | VLM-RM(1.0) | VLP |
|---|---|---|---|---|---|---|---|
| Avg | 0.406 | 0.390 | 0.073 | -0.060 | -0.005 | -0.171 | **0.718** |

这是 paper 最强的实证证据。关键观察：

- VLM-RM (CLIP-based zero-shot reward) 平均 correlation 接近 0，甚至负相关，说明 CLIP 的 cosine similarity 直接当 reward 在 embodied 任务上几乎不可用。
- LIV、R3M 这种 pretrain 后 representation 也只有 ~0.4 correlation，远低于 VLP 的 0.718。
- VLP 的优势来源：**用 preference 形式 + 三种关系监督**，而不是简单的 embedding 相似度。它是 supervised contrastive，而 VLM-RM 是 unsupervised alignment。
- 还有一个工程意义：VLP 训练只需要 6 小时 1 张 4090，远低于 R3M/LIV 的大规模预训练。

参考链接：
- VLM-RM: https://arxiv.org/abs/2310.12921
- R3M: https://arxiv.org/abs/2303.00905
- RoboCLIP: https://arxiv.org/abs/2310.07699

### 7.3 Q3: 泛化能力（Table 6）

| Metric | Seen | Phrase | Description | Correct Color | Incorrect Color |
|---|---|---|---|---|---|
| ITP Acc | 97.4 | 95.8 | 97.0 | 97.0 | 97.0 |
| IVP Acc | 91.7 | 90.5 | 91.9 | 91.9 | 91.8 |
| ILP Loss | 0.705 | 0.704 | 0.704 | 0.705 | 0.705 |
| Avg Loss | 0.555 | 0.554 | 0.558 | 0.556 | 0.557 |

关键观察：

- 在 unseen tasks + unseen language instructions 上 VLP 几乎不掉点，correlation 仍 90%+。
- "Phrase"（短词组）的 ITP 略降（97.4 → 95.8），论文解释为 phrase 信息不足；"Description"（长描述）反而对 IVP 略涨，因为长描述给出更多 task 信号。
- **Color robustness**：correct vs incorrect color 几乎一样，说明 model 没有被颜色 shortcut 误导。这背后是 ILP + IVP 训练让 model 学到了"语言核心语义 vs 表层 lexical 差异"的区分。

附录 B 还扩展到 ManiSkill2（LiftCube / OpenCabinetDoor / PushChair），平均 97.9% accuracy（Table 12），说明 VLP 不止对 Meta-World 过拟合。

---

## 8. 与相关工作的差异图谱

| 方法 | 监督形式 | 需要 pretraining | 是否泛化到新任务 | 关键问题 |
|---|---|---|---|---|
| R3M | time-contrastive + vision-language | 是 (Ego4D) | 部分 | 表征不直接对应 task reward |
| VIP / LIV | value-implicit pretraining | 是 (EpicKitchen) | 部分 | reward 是 absolute，noise 大 |
| VLM-RM | CLIP zero-shot cosine | 是 (CLIP) | zero-shot | correlation ~0，几乎不可用 |
| RoboCLIP | S3D video-language + 单 demo | 是 (HowTo100M) | zero-shot | 依赖单 demo 质量 |
| RL-VLM-F | Gemini-Pro / GPT-4V 在线 query | 否 (API) | zero-shot | 需访问环境信息，label 不稳定 |
| CriticGPT | fine-tune MLLM | 是 (instruction data) | fine-tune | 训练昂贵 |
| PT / PEBBLE / SURF / RUNE | human preference | 否 | 单 task | 标注贵 |
| IPL / CPL | human preference | 否 | 单 task | 标注贵 |
| PEARL | cross-task preference transfer | 否 | 部分 | 仍需目标 task 偏好 |
| **VLP** | vision-language preference (ITP/ILP/IVP) | 用 CLIP backbone，但 preference model 从头训 6h | 是，到 unseen task + unseen language | — |

VLP 处在一个很巧的位置：**用 VLM 作 backbone，但用 RLHF 的 preference 形式监督**，同时引入 language condition 让监督信号跨 task 复用。它没有 R3M/LIV 的大规模 pretraining 负担，又比 VLM-RM zero-shot 强很多。

参考：
- PEBBLE: https://proceedings.mlr.press/v139/lee21i.html
- IPL: https://arxiv.org/abs/2310.04457
- CPL: https://arxiv.org/abs/2310.13039
- PEARL: https://arxiv.org/abs/2406.06354
- RL-VLM-F: https://arxiv.org/abs/2407.08293

---

## 9. 下游算法怎么用 VLP label

论文把 VLP 当作 "preference annotator"，可以插到三类 RLHF pipeline 里：

1. **P-IQL (Preference IQL)**：先用 VLP label 训 reward model $\hat{r}_\psi$（用 Eq. 2 的 CE loss），再用 IQL 在 offline buffer 上训 policy。IQL 的核心是 expectile regression + Q/V 网络，不需要 online 采样。
   - 参考 IQL: https://arxiv.org/abs/2110.06169
   
2. **IPL (Inverse Preference Learning, Hejna & Sadigh 2023)**：跳过 reward model，直接用 preference 对齐 Q-function。它假设 $\exp(Q(s,a))$ 应满足 BT 模型，VLP label 直接监督 Q。

3. **CPL (Contrastive Preference Learning, Hejna et al. 2024)**：把 preference 当作 supervised contrastive 信号，用 maximum entropy 原则直接学 policy，完全避免 RL。这是 VLP 的最佳拍档——因为 VLP 给的是 trajectory-level relative label，和 CPL 的 contrastive objective 天然对齐。

工程细节：对每个 test task，先用 K-means 把 trajectory 聚成 2 类，每类采 100 条 length-50 的 segment pair，用 VLP model 推断 preference label。整个 inference 在 RTX 4090 上 ~10 分钟训完下游 RL/RLHF。

---

## 10. 消融与敏感性分析

### λ1, λ2 的影响（Table 13）

| λ1 | λ2 | ITP Acc | IVP Acc | ILP Loss | Avg Loss |
|---|---|---|---|---|---|
| 0.0 | 0.5 | 95.4 | 74.1 | 0.728 | 0.618 |
| 0.5 | 0.5 | 85.8 | 74.7 | 0.702 | 0.578 |
| 0.1 | 0.0 | 96.2 | 63.0 | 0.775 | 0.646 |
| 0.1 | 1.0 | 95.8 | 96.5 | 0.699 | 0.554 |
| **0.1** | **0.5** | **97.4** | **91.7** | 0.705 | 0.555 |

直觉：

- **IVP ($\lambda_2$) 是关键**：去掉 ($\lambda_2=0$)，IVP Acc 从 91.7 掉到 63.0，ILP Loss 从 0.705 飙到 0.775。没有 IVP，模型完全没法区分 task identity，所有"语言条件"结构都失效。
- **ILP ($\lambda_1$) 是调味**：太大（0.5）会让 ITP 退化和 ILP 抢梯度（ITP Acc 从 97.4 掉到 85.8），太小（0.0）让 ILP Loss 升。0.1 是个甜点，相当于一个温和的 regularizer。
- $\lambda_2=1.0$ 在 IVP 上更好（96.5），但平均 loss 反而比 0.5 高（0.554 vs 0.555），且 ITP 略降。说明 IVP 权重过大会让模型过度聚焦 task-discrimination 而忽视 optimality。

### 数据规模（Table 14）

50% / 75% / 100% 数据：ITP 从 94.2 → 95.2 → 97.4，IVP 从 89.6 → 89.7 → 91.7。掉到 50% 时性能损失可控，说明 VLP 的 sample efficiency 不错。这背后是 CLIP backbone 提供了强 prior，preference model 只需要学相对关系。

---

## 11. 局限性与我的思考

论文自己承认的局限：

1. **任务必须能被 video + language 描述**：复杂 assembly、长程 spatial reasoning 任务（如 IKEA 组装）可能表达不出来。
2. **Language 信息不足时风险上升**：Table 6 中 phrase 输入会让 ITP 略降，因为信息量不够。

我额外想到的几个 issue：

1. **Trajectory-level vs step-level reward**：VLP 只给 segment preference，对需要 fine-grained credit assignment 的任务（比如 in-hand manipulation 中某一帧的关键微调）可能力不从心。一个可能的扩展是结合 Preference Transformer 的 attention aggregation，做 frame-level 的 reward 分配。
2. **Medium level 定义依赖 scripted policy 的 subtask flag**：对没有 `near_object` 这类信号的环境（如真实机器人），medium 轨迹的生成需要其他启发式。这是 pipeline 推广到 real-world 的瓶颈。
3. **Cross-task generalization 的边界**：Table 6 只测了 5 个 Meta-World test task + 3 个 ManiSkill2 task。如果 test task 和 train task 视觉差异巨大（如厨房任务训练 → 工厂装配测试），VLP 的 language alignment 可能不够。
4. **Language encoder 的语义理解**：CLIP text encoder 是 bag-of-phrases 级别，对复杂指令（"先开抽屉，等三秒，再把杯子放进去"）的时序逻辑理解有限。一个升级方向是用 LLM-based text encoder + VLP 的 cross-modal alignment 联合训练。
5. **Self-supervised bootstrap 可能性**：当前 VLP 依赖 scripted policy 造 optimality 分层。如果完全 unsupervised（用 random policy + RL agent 自己的探索轨迹 + 自一致 preference），是否能训出 model？类似 self-rewarding language model 的思路在 RL 中可以探索。
6. **Reward hacking 风险**：VLP 学到的是"video 看起来像不像语言描述"，可能被视觉上像但实际不完成的 trajectory 骗过。论文没有 adversarial 测试（如训练一个对抗 video 让 VLP 高分但实际失败），这是 future work 应该做的。

---

## 12. 这篇工作在更大版图中的位置

VLP 实际上把三个独立 trend 串了起来：

1. **RLHF 的多模态化**：把 LLM 里 DPO/RLHF 的思路搬到 video domain，但用 language 作 condition 解决多任务泛化。
2. **VLM as reward 的修复**：之前 CLIP-as-reward 失败，VLP 通过 preference 形式 + 三关系监督，把 VLM embedding 变成可靠的 relative supervision。
3. **Cross-task transfer in preference RL**：PEARL 等工作尝试跨任务 preference 迁移，VLP 给了一个更通用的方案——通过 language 作为 pivot 让 preference 跨 task 复用。

从这个角度看，VLP 是 RLHF + VLM + multi-task transfer 三者交叉点上的一个很自然的解。它的 contribution 不在某个单点突破，而在于这套组合：数据 pipeline（MTVLP）+ 三关系定义（ITP/ILP/IVP）+ 架构（CLIP backbone + cross-modal transformer）+ 理论联系（negative regret），形成一个完整闭环。

如果让我预测后续工作：很可能会看到 (a) VLP 应用到真实机器人视频（用 real-world 的 scripted/demo 轨迹），(b) 用 video diffusion model 生成 medium trajectory 替代 scripted policy，(c) 把 VLP 的 cross-modal transformer 替换成 LLaVA 式 multimodal LLM fine-tune，(d) 把 ITP/ILP/IVP 三关系扩展到更多关系类型（如 temporal-ordering preference、sub-goal preference）。

---

## 13. 总结：VLP 给我的三个核心 intuition

1. **Preference 比 absolute reward 更鲁棒**：相对标签天然抗噪声、抗 reward hacking，因为只需要排序正确，不需要绝对标定。这是 RLHF 在 LLM 上成功的根本原因，VLP 把它移植到 video domain。
2. **Language 是天然的 task interface**：把 language 作为 condition 而不是 input，让一个 preference model 可以跨无限多个 task 复用，前提是这些 task 都能用语言描述。
3. **三种关系构成最小完备的 alignment 监督**：ITP（同任务 optimality）+ ILP（语言不匹配时不变）+ IVP（任务判别）这三种关系，是教模型理解"video + language 在描述什么"的最小集合。少一种，alignment 就不完整。这是 VLP 最 beautiful 的设计。

参考 repo：
- VLP 应该有代码发布（论文是 NeurIPS/ICLR 风格），可以关注作者 Runze Liu / Chenjia Bai 的 GitHub。
- LAPP (Xie et al. 2023) 是 VLP 的实现基础: https://arxiv.org/abs/2310.04457 (IPL repo)
- CPL repo: https://github.com/jhejna/cpl

如果想 build intuition 进一步，我会建议：拿 Meta-World 跑一个 vanilla CLIP-as-reward 的 IQL，看看它怎么会失败（Table 5 显示 correlation ~-0.06），然后对照看 VLP 的三关系监督怎么把 0.718 correlation 拉起来。这种对比实验最能让人感受到"preference + language condition"两个改动各自贡献多少。
