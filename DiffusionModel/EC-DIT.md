---
source_pdf: EC-DIT.pdf
paper_sha256: d80288500f441fbf57dd0404c305cc4e2df889240ced98e1c81c93aad35e6ef5
processed_at: '2026-08-04T01:27:04-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EC-DIT

## TL;DR 一句话

传统 MoE 是"每个 token 都得被 k 个 expert 处理，不管你配不配"。EC-DIT 是"每个 expert 只挑它最想处理的 C 个 token，重要 patch 被 44% 的 expert 抢着处理，背景 patch 一个 expert 都不要"。结果：**97B 总参数，激活 8B，GenEval 71.68% SOTA**，比 SD3+DPO 还高。

---

## 类比一下

想象 token 是学生，expert 是食堂菜盘。

- **Token-choice**（传统 MoE）：每个学生必须去两个菜盘打菜，不管你饿不饿、爱不爱吃。菜盘不知道谁要来，只能傻等。
- **Expert-choice**（EC-DIT）：每个菜盘自己挑最饿的 C 个学生盛菜。热门菜盘排长队，冷门菜盘空着。每个菜盘恰好盛满 $C$ 份，不会浪费。

关键差异：决策权从 token 翻转到 expert。Token 说"我要你"，变成 expert 说"我要你"。

---

## 老 MoE 在 diffusion 上为什么别扭

[Switch Transformer](https://arxiv.org/abs/2101.03961)、[GShard](https://arxiv.org/abs/2006.16668) 这些都是为 LLM 设计的。LLM 是 autoregressive，每个 token 只看左边，所以"每个 token 自己挑 expert"很自然。但 diffusion 有三个 LLM 没有的特性，token-choice 全没利用上：

**1. Image patch 的难度天差地别**
月球表面的陨石坑、人脸细节、生成的文字——这些 patch 需要海量 compute。纯色背景、低频天空——给它们一个 FFN 都是浪费。Token-choice 强制每个 patch 都吃 $k$ 个 expert，这是均匀分配，违反生成任务的天然异质性。

**2. Diffusion 是 bidirectional 的**
每个 denoise step，模型同时看到全图所有 patch。LLM 的 token 看不到未来，所以 router 只能 local 决策。但 diffusion 的 router 完全可以"看一眼全图，再决定哪个 patch 该重点处理"。Token-choice 浪费了这个 global view。

**3. Load balancing 是个 pain**
Token-choice 必须加 auxiliary load-balance loss（[Shazeer 2017](https://arxiv.org/abs/1701.06538)），不然 router collapse 到几个 expert 上。这是工程麻烦、优化噪音、调参地狱。

EC-DIT 一次性解决这三个问题。

---

## EC-DIT 的三个核心改动

### 改动 1：Router 的输入是 cross-attention 后的表示

这是被低估的关键细节。看 paper 公式 4：

$$\mathbf{x}' = \mathbf{x}_s + \text{MHCA}(\mathbf{x}_s)$$

Router 不是在 raw image patch 上工作，而是在 **self-attention + cross-attention 之后**。这意味着 router 看到的表示已经融了：
- 图像 patch 的 spatial 信息（self-attention 贡献）
- 文本 prompt 的对齐信息（cross-attention 贡献，公式 2-3）
- Timestep 信息（通过 AdaLN 注入）

所以 router "看得懂"哪个 patch 对应文本里的关键 noun。你说"宇航员在月球上"，router 就知道哪个 patch 是宇航员、哪个是月球、哪个是黑背景。**这是 expert 能"挑对 token"的前提**。

### 改动 2：Expert 选 token，反过来

公式 5 算 affinity $\mathbf{A}_{s,i}$，注意 softmax 是沿 **expert 维**做的（每个 token 对所有 expert 的 preference 之和为 1）。然后公式 6：

$$\mathbf{G}_{s,i} = \begin{cases} \mathbf{A}_{s,i}, & \mathbf{A}_{s,i} \in \text{top-}C \\ 0, & \text{otherwise} \end{cases}$$

每个 expert $i$ 在所有 $S$ 个 token 的 affinity 里挑 top-$C$。$C = S \cdot f_c / E$，$f_c=2$ 是 capacity factor。

直觉：每个 expert 像猎头，扫描全图所有 patch 的"求职意愿分"（affinity），挑最匹配自己的 $C$ 个。被多个 expert 看上的 patch，会被多次处理，compute 自然聚集。

### 改动 3：偶数层 sparse，奇数层 dense

从第 2 层起，每隔一层换成 MoE。这是工程与性能的甜点：
- 不用每层都 router，减少 instability
- Dense 层保全局表示能力
- 跟 [Switch Transformer](https://arxiv.org/abs/2101.03961) 的 interleave 经验一致

---

## 为什么这么改 work——三个直觉

### 直觉 A：Diffusion 是 expert-choice 的"老家"

[原版 expert-choice paper](https://arxiv.org/abs/2202.09368) 在 LLM 上做，有个尴尬：autoregressive 推理时 expert 选 top-C token 需要"看到未来"，只能用在 training 或 non-causal attention 上。但 **diffusion 天生 bidirectional，每步全图都可见**。所以 expert-choice 在 diffusion 上没有 AR constraint，可以放心用。这是"把对的工具放到了对的地方"。

### 直觉 B：Heterogeneity 是免费的午餐

$f_c=2$ 跟 token-choice top-2 的总 compute 一样，但效果碾压。说明**瓶颈不在 compute 总量，在 compute 分配的 heterogeneity**。给月球 patch 4 个 expert、给背景 patch 0 个 expert，比给所有 patch 都 2 个 expert 强。Heterogeneity 是 image 生成任务天然给的，EC-DIT 只是把它捡起来用。

### 直觉 C：Router 通过 end-to-end 训练学会"看什么"

Paper Section 3.5 的可视化（Figure 6）是最直观证据：
- 早期 layer allocation 较均匀（处理低频 / 全局结构）
- 后期 layer allocation 高度集中（处理细节 / 文本相关 patch）
- 早期 timestep 较均匀，后期 timestep 集中
- 月球 patch 被多达 44% expert 选中

这些 allocation pattern 跟 [Lei et al. 2023](https://arxiv.org/abs/2304.04947) 在 pixel space 观察到的"early=low-freq, late=high-freq"完全吻合。**EC-DIT 没有手动编码这个 prior，而是 end-to-end 学出来了**。这种 emergent behavior 很 Karpathy-style。

---

## 关键数字

| 指标 | EC-DIT | 对比 |
|------|--------|------|
| Total params | 97.21B | 之前最大 sparse DiT 是 16B ([DiT-MoE](https://arxiv.org/abs/2407.08826)) |
| Activated params | 8.27B | 跟 SD3-Large 8B 几乎一样 |
| GenEval overall | **71.68%** | SD3-Large 68%, SD3+DPO 71%, DALL-E 3 67% |
| 推理 overhead | 20-28% | 理论 15%，实际因 model parallelism 略高 |
| EC-DIT-3XL-32E 推理时间 | 33.8% of 8B DENSE-M | 用更少 activated 拿更好结果 |
| 训练步数省一半 | EC-DIT-XXL 200K 步 ≈ DENSE-XXL 400K 步 | Figure 8 |

值得注意：EC-DIT-M-64E 在 **256×256** 分辨率下打 SD3-Large 在 **512×512** 下。一般分辨率翻倍 GenEval 会涨几个点，EC-DIT 用一半分辨率还赢——raw architecture 的优势很硬。

---

## 我觉得最 cool 的地方

**1. Expert collapse 问题不存在了**
Token-choice 最怕 router 把所有 token 都送到一个 expert 上，所以加 aux loss。Expert-choice 的 top-C 机制让每个 expert 恰好处理 $C$ 个 token，**by construction 完美平衡**。这是 paper 里最优雅的 design choice。

**2. $f_c=2$ 就够**
没调 capacity factor 到 4、8，就 $f_c=2$ 跟 token-choice top-2 公平对比，依然碾压。说明 expert-choice 的优势来自 **adaptive allocation**，不是 compute budget 翻倍。

**3. Layer 数从 46 降到 38 是工程妥协**
EC-DIT-M 为 fit HBM 把 layer 数砍了 8 层，结果还是赢。这是 scaling 路径 robust 的证据。如果将来有更大 HBM，加回 46 层会更强。

**4. GQA (#KV=6 或 12) 在 97B 上是必须的**
不然 KV cache 会爆。这跟 LLaMa、Gemini 路线一致，sparse diffusion 也得跟 LLM 学 memory-efficient attention。

---

## 一些联想 / Open Questions

**1. 推广到 video**
Paper Section A 明确提了。Video diffusion 每个 chunk 也是 bidirectional 的，expert-choice 直接适用。而且 video 的 token 复杂度异质性比 image 更强（运动物体 vs 静态背景），expert-choice 的 heterogeneity 收益会更大。可能结合 [SparseCtrl](https://arxiv.org/abs/2311.16933) 的思路。

**2. 推广到 AR + diffusion unified model**
[Transfusion](https://arxiv.org/abs/2408.11039)、[Chameleon](https://arxiv.org/abs/2405.09818) 这种 unify AR 和 diffusion 的架构里，diffusion 部分 natural fit expert-choice，AR 部分还是得 token-choice。混合 routing 是个 open direction。

**3. Position 任务还是差**
GenEval Position 21.33% 跟 SD3 一样烂。说明 sparse MoE 加 capacity 解决不了 spatial reasoning 的根本问题。这需要的是更 structured 的 inductive bias（如 slot attention、object-centric representation），不是堆 expert。

**4. Affinity softmax 维度的方向选择**
EC-DIT 沿 expert 维 softmax（每个 token 对所有 expert 之和为 1），然后 expert 选 top-C token。能不能反过来——沿 token 维 softmax（每个 expert 对所有 token 之和为 1），然后 token 选 top-k expert？这其实就是 token-choice 的变种。Paper 没做这个 ablation，可能值得试。

**5. Expert 之间的 specialization 长啥样**
Paper 可视化了 token-side allocation（哪个 patch 被多少 expert 选），但没可视化 expert-side specialization（expert $i$ 倾向处理什么）。如果跑 t-SNE 看 expert embedding，能不能看到"月球 expert"、"文字 expert"、"背景 expert"这种 emergent specialization？这能进一步证明 heterogeneity 是 learned 的。

**6. 跟 soft MoE 的关系**
[Soft MoE](https://arxiv.org/abs/2308.00951)（[Puigcerver 2023](https://arxiv.org/abs/2308.00951)）用 soft assignment 而不是 hard top-C，全部 token 都参与全部 expert 但权重不同。Soft MoE 没有 dispatch 通信开销，但失去了 hard sparsity 的 interpretability。Diffusion 上 soft MoE vs expert-choice 的 trade-off 没人做过系统比较。

**7. RMSProp 而不是 AdamW**
Paper 用 RMSProp + momentum。这有点反主流（LLM 都用 AdamW）。可能是 TPU memory 考虑，或者大 batch + 1.2B pair 数据下 RMSProp 已经够。如果能 ablation 一下 AdamW vs RMSProp 在 97B 上的 stability 差异，会很有意思。

---

## 我自己的吐槽

- Paper 标题"SCALING DIFFUSION TRANSFORMERS WITH ADAPTIVE EXPERT-CHOICE ROUTING"太长太学术，叫"EC-DIT: Let Experts Pick Tokens"就行。
- Figure 1(b) 的视觉对比是 cherry-picked 的（每个 prompt 选 best of 4）。真正能信的是 Table 2 的数字。
- DSG 用"internal Gemini"做 VQA 评估，reproducibility 差一点。但 GenEval 是公开 benchmark，那个 71.68% 是硬的。
- Masking ratio 0.5 跟 expert-choice 的交互没讲清楚——被 mask 的 token 是否参与 expert selection？我推测不参与，但 paper 没明说。这是 reproducibility 的小坑。

---

## 一句话总结

EC-DIT 把 [Zhou 2022](https://arxiv.org/abs/2202.09368) 的 expert-choice 从 LLM（它格格不入的地方）搬到 diffusion（它的天然 habitat），利用 diffusion 的 bidirectional + patch heterogeneity 两个特性，让 expert 主动抢 token，自然实现 adaptive compute + perfect load balance，一路 scale 到 97B 拿 SOTA。整个故事很干净，idea 不复杂，execution 很扎实。

References 都在前一条回复里，这里不重复了。需要 drill down 哪一块再问我。

---

# EC-DIT: Expert-Choice Routing for Scaling Diffusion Transformers — 深度解读

Andrej，这篇 paper 来自 Apple AI/ML + Georgia Tech，作者包括 Tao Lei、Bowen Zhang、Ruoming Pang、Bo Dai、Nan Du 等人。核心 idea 是把 **Expert-Choice Routing** ([Zhou et al., 2022](https://arxiv.org/abs/2202.09368)) 从 NLP 域迁移到 DiT ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) 上，把模型推到 **97B 参数**，在 GenEval 上拿到 **71.68% SOTA**。下面我从直觉、机制、公式、实验四个层次给你 build intuition。

---

## 1. Motivation: 为什么 token-choice MoE 在 diffusion 上不够用

现有 sparse DiT 工作（[DiT-MoE](https://arxiv.org/abs/2407.08826), [RAPHAEL](https://arxiv.org/abs/2305.18295), [SegMoE](https://huggingface.co/segmind/SegMoE-4x2-v0)）几乎都直接搬 NLP 域的 **token-choice routing**（[GShard](https://arxiv.org/abs/2006.16668), [Switch Transformer](https://arxiv.org/abs/2101.03961), [ST-MoE](https://arxiv.org/abs/2202.08906)）。这种 routing 有三个在 diffusion 场景下被忽略的弱点：

### 1.1 Uniform computation 的 mismatch
token-choice 中每个 token 都被恰好 $k$ 个 expert 处理。但 image patch 的复杂度是高度异质的：foreground object / 渲染中的文字 / 月球表面纹理 vs 单调背景 / 低频区域。给背景分配 $k$ 个 expert 是浪费，给人脸细节分配 $k$ 个 expert 是欠拟合。这与生成任务天然的 "compute heterogeneity" 相违背。

### 1.2 Local context 的 mismatch
token-choice 源自 autoregressive LM，每个 token 只能看 left context。但 DiT 每一步 denoise 都是 **full-sequence bidirectional** 的，全局信息天然可用。token-choice 没用上这个全局信号。

### 1.3 Load balancing 的痛苦
token-choice 必须加 auxiliary load-balance loss（[Shazeer 2017](https://arxiv.org/abs/1701.06538), [Zoph 2022](https://arxiv.org/abs/2202.08906)），否则 router collapse 到几个 expert 上。这是工程上的麻烦，也是优化上的噪音源。

EC-DIT 用 **expert-choice routing** 同时解决这三个问题。直觉：让 expert 主动挑 token，而不是让 token 找 expert。

---

## 2. Preliminaries: Rectified Flow 与 MoE

### 2.1 Rectified Flow 的训练目标

DiT 处理 latent space 的 patch 序列 $\mathbf{x} \in \mathbb{R}^{S \times d_x}$，$S$ 是序列长度，$d_x$ 是 hidden dim。Rectified flow ([Liu et al., 2022](https://arxiv.org/abs/2209.03003); [Lipman et al., 2022](https://arxiv.org/abs/2210.02747)) 在 data 分布 $p_0$ 和 noise 分布 $p_1 = \mathcal{N}(0, I)$ 之间用直线连接：

$$\mathbf{x}_t = t \mathbf{x}_1 + (1-t) \mathbf{x}_0$$

其中 $\mathbf{x}_0 \sim p_0$（干净图），$\mathbf{x}_1 \sim \mathcal{N}(0, I)$（纯噪声），$t \in [0, 1]$。速度场 $v_\theta(\mathbf{x}_t, t) = \mathbf{x}_1 - \mathbf{x}_0$ 由 DiT 参数化。训练目标：

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \pi(t), \mathbf{x}_1 \sim \mathcal{N}(0, I)} \left[ \left\| (\mathbf{x}_1 - \mathbf{x}_0) - v_\theta(t \mathbf{x}_1 + (1-t)\mathbf{x}_0, t) \right\|^2 \right] \quad (1)$$

变量含义：
- $\pi(t)$：timestep 采样密度，这里用 **logit-normal** ([Atchison & Shen, 1980](https://doi.org/10.1093/biomet/67.2.261))：$t = \log \frac{u}{1-u}$，$u \sim \mathcal{N}(\mu_t, \sigma_t)$。$\mu_t, \sigma_t$ 控制中间 timestep 被采样的频率。这个 trick 与 SD3 ([Esser et al., 2024](https://arxiv.org/abs/2403.03206)) 一致，目的是把训练 mass 集中在 mid-noise 区间。

### 2.2 Token-Choice MoE 的形式化

经典 MoE layer 有 $E$ 个 expert $\{\mathcal{E}_i\}_{i=1}^E$，router weight $\mathbf{W}_r \in \mathbb{R}^{d_x \times E}$。对 token $\mathbf{x}$ 计算 gating $g_i = \text{softmax}(\mathbf{x} \cdot \mathbf{W}_r)_i$，选 top-k experts，output 加权组合。问题：每个 token 独立决策，没有 sequence-level 优化。

---

## 3. EC-DIT 架构：核心机制

### 3.1 整体结构

EC-DIT 基于 [PixArt-α](https://arxiv.org/abs/2310.00426) 的 DiT 变体，每个 block 顺序为：

```
Self-Attention → Cross-Attention (text → image) → MoE-FFN (or dense FFN)
```

并加 AdaLN 注入 timestep embedding。**关键设计**：从第 2 层起，**偶数层**全部替换为 sparse MoE，奇数层保留 dense FFN。这种 interleave 设计的好处：
- Dense 层保留全局表示能力，sparse 层扩容
- 减少 router 数量，降低 router 训练 instability
- 类似 [Switch Transformer](https://arxiv.org/abs/2101.03961) 的 every-other-layer sparsity 经验

### 3.2 Cross-Attention 注入文本

文本 prompt $\mathbf{y}$ 通过 text encoder（这里用 670M CLIP-based encoder + T5 tokenizer）得 $\text{Enc}(\mathbf{y}) \in \mathbb{R}^{L \times d_y}$，$L$ 是文本 token 数，$d_y$ 是文本 hidden dim。Cross-attention 头 $h$：

$$\mathbf{Q}_x = \mathbf{x} \cdot \mathbf{W}_x^{\text{query}}, \quad \mathbf{K}_y = \text{Enc}(\mathbf{y}) \cdot \mathbf{W}_y^{\text{key}}, \quad \mathbf{V}_y = \text{Enc}(\mathbf{y}) \cdot \mathbf{W}_y^{\text{value}} \quad (2)$$

$$\mathbf{C}_h = \text{softmax}\left(\frac{\mathbf{Q}_x \cdot \mathbf{K}_y^\top}{\sqrt{d_x}}\right) \cdot \mathbf{V}_y \quad (3)$$

变量：
- $\mathbf{W}_x^{\text{query}} \in \mathbb{R}^{d_x \times d_x}$：image side query projection
- $\mathbf{W}_y^{\text{key}}, \mathbf{W}_y^{\text{value}} \in \mathbb{R}^{d_y \times d_x}$：text side key/value projection
- $\sqrt{d_x}$：标准 scaled dot-product 的温度
- $H$ 个头 concat 后经 $\mathbf{W}^{\text{out}}$ 投影

**关键 insight**：Cross-attention 后的表示 $\mathbf{x}' = \mathbf{x}_s + \text{MHCA}(\mathbf{x}_s)$ (公式 4) **同时携带了 image patch 信息 + 文本对齐信息 + timestep（通过 AdaLN）**。这正是 router 的输入——它看到的是 cross-modal fused representation，所以能"知道"哪个 patch 对应文本中的关键 noun。

### 3.3 Expert-Choice Routing 的核心数学

#### Step 1: Affinity 计算
对 router 输入 $\mathbf{x}' \in \mathbb{R}^{S \times d_x}$，与 router weight $\mathbf{W}_r \in \mathbb{R}^{d_x \times E}$ 做点积，再 softmax over expert 维度：

$$\mathbf{A}_{s,i} = \frac{\exp\left((\mathbf{x}' \cdot \mathbf{W}_r)_{s,i}\right)}{\sum_{i=1}^{E} \exp\left((\mathbf{x}' \cdot \mathbf{W}_r)_{s,i}\right)} \quad (5)$$

- $s$：token index ($1 \le s \le S$)
- $i$：expert index ($1 \le i \le E$)
- $\mathbf{A} \in \mathbb{R}^{S \times E}$：token-expert affinity tensor。**沿 expert 维 softmax** 意味着每个 token 对所有 expert 的 preference 之和为 1（这是与 token-choice 一致的方向）。

#### Step 2: Top-C token selection per expert
**这是 EC-DIT 与 token-choice 的根本分歧点**。每个 expert $i$ 在所有 $S$ 个 token 上看 $\mathbf{A}_{s,i}$，选 top-$C$ 个最大值：

$$\mathbf{G}_{s,i} = \begin{cases} \mathbf{A}_{s,i}, & \mathbf{A}_{s,i} \in \text{top-k}(\{\mathbf{A}_{s,i} | 1 \le s \le S\}, k=C) \\ 0, & \text{otherwise} \end{cases} \quad (6)$$

- $C = S \cdot f_c / E$：每个 expert 的容量，是个**平均容量**。
- $f_c$：**capacity factor**，反映每个 token 平均被多少 expert 处理。Paper 里 $f_c = 2.0$，对应每个 token 平均被 2 个 expert 处理（与 token-choice top-2 匹配，便于公平比较）。
- $\mathbf{G} \in \mathbb{R}^{S \times E}$：gating tensor，是稀疏的，只有 $E \times C$ 个非零元素（即 $S \cdot f_c$ 个非零元素）。

#### Step 3: Expert 处理与合并
对每个 expert $i$，定义其处理的 token 索引集 $\mathcal{T}_i = \{s | \mathbf{G}_{s,i} > 0\}$。每个 expert 是 2 层 FFN：

$$\mathcal{E}_i(\mathbf{x}) = \text{GeLU}(\mathbf{x} \cdot \mathbf{W}_1^i) \cdot \mathbf{W}_2^i$$

- $\mathbf{W}_1^i \in \mathbb{R}^{d_x \times d_x'}$, $\mathbf{W}_2^i \in \mathbb{R}^{d_x' \times d_x}$：expert $i$ 的 weight
- $d_x'$：FFN intermediate dim（paper 里 $d_x' = 4 d_x$，与 dense FFN 一致）

输出组合：

$$\mathbf{x}_{\text{out}} = \sum_{i=1}^{E} (\mathbf{G}_{\mathcal{T}_i, i})^\top \mathcal{E}_i(\mathbf{x}'_{\mathcal{T}_i, :}) \quad (8)$$

直觉：每个 token 的最终 output 是所有"选中它"的 expert 的加权和，权重就是 affinity 值。

### 3.4 为什么 EC-DIT 自然解决三大问题

| 问题 | EC-DIT 的解决方式 |
|------|------------------|
| Uniform compute | 一个 token 可能被 0 个、1 个、甚至 $E$ 个 expert 处理。Figure 6 显示某些 detail token 被 44% 的 expert 选中，而背景 token 可能跳过整层。 |
| Local context | Expert 在选 token 时看的是 $\mathbf{A}_{s,i}$ 整列（所有 $S$ 个 token），即全局视野。 |
| Load balance | 每个 expert 精确处理 $C$ 个 token，**by construction 完美 balance**。无需 auxiliary loss。 |

### 3.5 Algorithm 1 伪代码精读

```python
def ec_dit_routing(x_p, W_r, experts):
    # Step 1: Affinity (B, S, E)
    logits = einsum('bsd,de->bse', x_p, W_r)
    affinity = softmax(logits, dim=-1)
    affinity = einsum('bse->bes', affinity)   # transpose to (B, E, S)
    
    # Step 2: Top-C per expert
    gating, index = top_k(affinity, k=C, dim=-1)   # (B, E, C), (B, E, C)
    dispatch = one_hot(index, num_classes=S)        # (B, E, C, S)
    
    # Step 3: Gather, process, scatter
    x_in = einsum('becs,bsd->becd', dispatch, x_p) # (B, E, C, d)
    x_e = stack([experts[e](x_in[:, e]) for e in range(E)], dim=1)
    x_out = einsum('becs,bec,becd->bsd', dispatch, gating, x_e)
    return x_out
```

工程细节：
- `dispatch` 是 (B, E, C, S) 的 one-hot tensor，用于 gather/scatter
- `x_in` 是 (B, E, C, d)：每个 expert 拿到自己那 $C$ 个 token 的 representation
- 最终 `einsum('becs,bec,becd->bsd', ...)`：对每个 token $s$，把所有 expert $i$ 中选了它的输出按 gating 加权求和

这种 dispatch/combine 模式与 [GShard](https://arxiv.org/abs/2006.16668) 的 MoE 实现高度类似，只是 selection 方向反了。

---

## 4. 模型 Scaling 配置

Table 1 给出四种 backbone（XL / XXL / 3XL / M）的参数：

| Config | Hidden dim | #Layers | #Heads | #KV | Dense Params | 32E Params |
|--------|-----------|---------|--------|-----|--------------|------------|
| XL     | 1,152     | 28      | 18     | 6   | 1.47B        | 6.08B      |
| XXL    | 1,536     | 38      | 24     | 6   | 2.35B        | 13.47B     |
| 3XL    | 2,304     | 42      | 36     | 6   | 4.50B        | 32.15B     |
| M      | 3,072     | 38*/46  | 48     | 12  | 8.03B        | 97.21B (64E) |

要点：
- **GQA** (Grouped-Query Attention) 用 #KV=6 或 12，KV head 数远少于 query head，节省 KV cache。
- EC-DIT-M-64E 把 layer 数从 46 降到 38 是为了**fit HBM**——这是工程约束下的 trade-off。
- Activated params（每 token 实际激活的参数）只随 backbone 变化，与 expert 数量无关。EC-DIT-M-64E 有 97B 总参数但只激活 8.27B，与 dense 8B 持平。

### 4.1 Activated Parameters 计算（Algorithm 2）

```python
dense_ffn_size = 2 * hidden_dim * hidden_dim * ffn_factor  # FFN 两层
router_size = num_experts * hidden_dim
attn_size = hidden_dim * (num_heads + 2*num_kv_heads + num_heads) * attn_key_dim

activated_dense_total = (attn_size + dense_ffn_size) * num_sparse_layers
activated_sparse_total = (attn_size 
                        + 2*hidden_dim*hidden_dim*ffn_factor*capacity_factor 
                        + router_size) * num_sparse_layers
activated_increment = activated_sparse_total - activated_dense_total
```

关键：sparse FFN 的 activated size = `dense_ffn_size * capacity_factor`（即 $f_c=2$ 时 sparse 激活是 dense 的 2 倍）。这就是为什么 paper 说"less than 30% increase in computational overhead"——主要开销在 FFN，attention 部分不变。

---

## 5. 实验结果：全面碾压

### 5.1 GenEval SOTA (Table 2)

EC-DIT-M-64E 在 256×256 分辨率下达到 **71.68% overall GenEval**，超过：
- SD3-Large (68.00%, 512×512 分辨率)
- SD3-Large w/ DPO (71.00%)
- DALL-E 3 (67.00%)
- FLUX.1 (paper Figure 10 视觉比较也输给 EC-DIT)

子任务亮点：
- **Color attr.**：EC-DIT-M-64E 60.80% vs SD3-Large 43.00%（DPO 后 47.00%）——这是属性绑定能力的体现
- **Two obj.**：88.67% vs SD3 84.00%
- **Position**：21.33% 仍然偏低（这是 GenEval 的难点，所有模型都差）

EC-DIT-3XL-32E 只用 5.18B activated params（SD3-Large 的 64%），就达到 70.91% GenEval，几乎追平 SD3-Large w/ DPO。这说明 scaling 路径高效。

### 5.2 DSG 评估 (Figure 3, Table 3)

DSG ([Cho et al., 2024](https://arxiv.org/abs/2310.18235)) 用 Gemini 做 VQA 评估 fine-grained alignment。EC-DIT-M-64E 拿到 76.40 overall。从 Table 3 可见，**增加 expert 数量在所有 DSG 子任务上几乎单调提升**，特别是 Text 和 Counting。

### 5.3 Token-Choice vs Expert-Choice (Figure 5, Table 5)

这是最关键的 ablation。在相同 base model (XXL) 和相同 activation size（top-2 token-choice vs $f_c=2$ expert-choice）下：
- EC-DIT-XXL-8E ≈ GShard-XXL-16E（即 EC-DIT 用一半 expert 数量打平）
- EC-DIT-XXL-32E 显著超过 GShard-XXL-32E
- FID vs CLIP Score 曲线 (Figure 7) 显示 EC-DIT 全程收敛更快

原因分析（paper 的 hypothesis）：
1. **Adaptive compute**：detail area 得到更多 expert，background 得到更少
2. **Global routing**：expert 选 token 时有全局视野，能识别 "最该被处理的 patch"
3. **No load imbalance**：token-choice 容易出现 expert collapse，导致部分 expert 过载、部分空转

### 5.4 推理效率 (Figure 4)

EC-DIT 的实际 inference overhead 在 **20-28%** 之间，略高于理论 15%（Table 1）和 EC-DIT-M 的 3%。原因：
- Model parallelism 跨 8×H100 的通信开销
- Group size（推理时 512）影响 dispatch 效率
- 即便如此，EC-DIT-3XL-32E 的推理时间只有 8B DENSE-M 的 33.8%

### 5.5 Heterogeneous Compute Allocation (Figure 6) — 最直观的证据

Figure 6 的 heatmap 显示每个 image token 被多少 expert 选中：
- **(a) Layer-wise allocation**：早期层 allocation 较均匀，后期层高度集中
- **(b) Timestep-wise allocation**：denoise 早期较均匀，后期集中
- 月球 patch 被多达 44% 的 expert 选中，背景 patch 在某些层直接跳过

这与 [Lei et al., 2023](https://arxiv.org/abs/2304.04947) 的 pixel-space observation 一致：early layer 处理 low-frequency / 全局结构，late layer 处理 high-frequency / 细节。EC-DIT 通过 end-to-end training 自发学到了这个 inductive bias。

---

## 6. 我的 Intuition 解读

### 6.1 为什么 expert-choice 对 diffusion 特别合适

Diffusion 与 LLM 的根本差异是 **bidirectional + 全 patch 同时处理**。这意味着：
- 任意一步 denoise，模型都能"看到"整张图
- 不同 patch 的难度是高度异质的（细节物体 vs 平铺背景）
- 每一步 denoise 的"重要 patch" 还会随 timestep 变化（早期是全局结构，晚期是细节）

Token-choice 假设每个 token 都值得相同的 compute，这在 LLM 上还勉强成立（每个 token 都要预测下一个），但在 diffusion 上是浪费。

Expert-choice 的妙处在于：**它让 expert 像资源分配器**，自然把 compute 流向最有信息量的 token。这与 [DTR](https://arxiv.org/abs/2310.07138) 和 [Switch-DiT](https://arxiv.org/abs/2403.09176) 用 timestep 决定 expert 的思路不同——EC-DIT 让 router 在 end-to-end 训练中自己学会"看什么、忽略什么"。

### 6.2 为什么 capacity factor $f_c=2$ 就够

$f_c$ 决定每个 token 平均被几个 expert 处理。$f_c=2$ 对应 token-choice top-2 的 compute budget，但效果远超 token-choice top-2。这说明**问题不在 compute 总量，而在 compute 分配的 heterogeneity**。token-choice 强制均匀，expert-choice 允许异质——后者更贴近图像生成的真实需求。

### 6.3 与 LLM 中 expert-choice 的对比

[原版 expert-choice paper](https://arxiv.org/abs/2202.09368) 在 LM 上做，发现效果不错但有个尴尬问题：autoregressive 推理时 expert 选 top-C token 需要"看到未来"，所以只能用在 training 或 non-causal attention 上，或者用 approximate 策略。但 **diffusion 没这个问题**——它天生 bidirectional，每步都看到全图。EC-DIT 等于"在 expert-choice 的天然 habitat 上用它"。

### 6.4 失败模式与未来工作

Paper Section A 提到 future work：
- 加入 object semantics / compositional relationships 作为额外 routing signal
- Apply 到 mixed-modal early-fusion model（如 [Chameleon](https://arxiv.org/abs/2405.09818)）
- Apply 到 video generation（每个 chunk 的全局上下文）
- Apply 到 [Transfusion](https://arxiv.org/abs/2408.11039) 这种 unify AR + diffusion 的架构

潜在 failure mode（paper 没明说但可以推测）：
- **Image 的 token 顺序对 expert-choice 重要吗**？Affinity 是 token-wise 计算，但 top-C 选择是 sequence-wise。如果 shuffle token 顺序，selection 结果不变（top-C 是基于 score 排序），但 expert 内部处理的是无序集合，这是 OK 的。
- **极少数 token 被所有 expert 选中**：会不会过度拟合高频 token？Paper Figure 6 显示某些 token 被选中 44% expert，这是异质性的体现，但也可能带来梯度集中。
- **早期 layer 的 uniform allocation**：可能是因为早期 timestep / 早期 layer 的 cross-attention 还没充分 fuse 文本信号，router 还"看不清"哪些 patch 重要。

---

## 7. 与其他 sparse diffusion 工作的关系

| 工作 | Routing 策略 | Scale | 备注 |
|------|-------------|-------|------|
| [eDiff-I](https://arxiv.org/abs/2211.01324) | Timestep ensemble | ~10B | 每 timestep 只激活 1 expert，效率差 |
| [ERNIE-ViLG 2.0](https://arxiv.org/abs/2210.15257) | Timestep MoE | ~24B | 同上 |
| [SegMoE](https://huggingface.co/segmind/SegMoE-4x2-v0) | Token-choice top-k | ~4×SDXL | SDXL 上小规模验证 |
| [RAPHAEL](https://arxiv.org/abs/2305.18295) | Timestep + token-choice | ~3B | 复合 routing |
| [DiT-MoE](https://arxiv.org/abs/2407.08826) | Token-choice | 16B | 离 EC-DIT 最近，但 token-choice 限制性能 |
| [MoMa](https://arxiv.org/abs/2405.09818) | Expert-choice | 7B | 同 routing 但规模小 |
| [DTR / Switch-DiT](https://arxiv.org/abs/2403.09176) | Timestep-task routing | 小 | 把不同 timestep 当不同 task |
| **EC-DIT** | **Expert-choice** | **97B** | 目前最大 sparse DiT |

EC-DIT 与 MoMa 同属 expert-choice camp，但 EC-DIT 把 scale 推到 14× larger，并且加了大量 ablation 和 visualization 来解释为什么 expert-choice 在 diffusion 上 work。

---

## 8. 工程实现的注意点

### 8.1 Group 维度重组
Paper 提到把 (batch_size, seq_len) 重组为 (outer_batch, num_group, group_size)，group_size 训练 1024 / 推理 512。这允许 EC-DIT 在推理时**一次处理两张图**（concat 起来走一次 forward），提高 throughput。这是 expert routing 在小 batch 上的常见 trick——单张 256×256 图 latent 只有 128 token（mask 0.5 后），expert capacity 太小，并行度不够。

### 8.2 Mask 加速训练
用 [MaskDiT](https://arxiv.org/abs/2306.09305) 风格的 masking，masking ratio 0.5，输入序列长度从 256 降到 128。这是 diffusion 训练加速的标准做法，但 paper 没说 masking 是否与 expert-choice 有交互（被 mask 的 token 是否参与 expert selection？我推测是不参与的，否则会扭曲 affinity 分布）。

### 8.3 TPU 上的 expert sharding
"Expert dimension fully shared across TPU mesh"——这意味着 expert weight 按 E 维度切分到不同 device，dispatch/combine 用 all-to-all 通信。这与 [GShard](https://arxiv.org/abs/2006.16668) 的 MoE 实现一致。

### 8.4 Optimizer
RMSProp + momentum，lr=1e-4，20K warmup，800K steps，batch 4096。**没用 AdamW**——可能是 TPU 上的 memory 考虑，或者是 1.2B image-text pair 的大 batch 训练 RMSProp 已足够。

---

## 9. 总结：EC-DIT 给我们的启示

1. **MoE 在 diffusion 上的设计应该利用 diffusion 的特性**（bidirectional、global context、heterogeneous patch complexity），而不是机械搬运 NLP 的 token-choice。
2. **Expert-choice 的"天然 habitat" 恰恰是 diffusion**——LM 上它的 autoregressive constraint 是缺点，diffusion 上变成优点。
3. **97B 不是终点**——paper Section A 明确说可以加更多 signal（object semantics、compositional）。GenEval 71.68% 还有提升空间，特别是 Position（21.33%）这种 compositional reasoning 任务。
4. **Inference overhead 28% 是值得的**——相比 dense 8B，EC-DIT-3XL-32E 只用 33.8% 推理时间就达到几乎相同的 GenEval。
5. **Visualization 是 paper 的亮点**——Figure 6 的 heatmap 直接证明了 "router 学到了 adaptive compute allocation"，这种 interpretability 在 MoE paper 中少见。

---

## References

- [EC-DIT paper (本篇)](https://arxiv.org/abs/2503.09778) — 假设链接，实际 arxiv ID 未知
- [Expert Choice Routing (Zhou et al., 2022)](https://arxiv.org/abs/2202.09368)
- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)
- [PixArt-α (Chen et al., 2023)](https://arxiv.org/abs/2310.00426)
- [SD3 / Rectified Flow (Esser et al., 2024)](https://arxiv.org/abs/2403.03206)
- [Rectified Flow (Liu et al., 2022)](https://arxiv.org/abs/2209.03003)
- [Flow Matching (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [GShard (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668)
- [Switch Transformer (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)
- [ST-MoE (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906)
- [Sparsely-Gated MoE (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- [GLaM (Du et al., 2022)](https://arxiv.org/abs/2112.06905)
- [GenEval (Ghosh et al., 2023)](https://arxiv.org/abs/2310.11513)
- [DSG (Cho et al., 2024)](https://arxiv.org/abs/2310.18235)
- [CLIPScore (Hessel et al., 2022)](https://arxiv.org/abs/2104.08718)
- [Logit-Normal (Atchison & Shen, 1980)](https://doi.org/10.1093/biomet/67.2.261)
- [DTR (Park et al., 2023)](https://arxiv.org/abs/2310.07138)
- [Switch-DiT (Park et al., 2024)](https://arxiv.org/abs/2403.09176)
- [RAPHAEL (Xue et al., 2023)](https://arxiv.org/abs/2305.18295)
- [DiT-MoE (Fei et al., 2024)](https://arxiv.org/abs/2407.08826)
- [SegMoE](https://huggingface.co/segmind/SegMoE-4x2-v0)
- [eDiff-I (Balaji et al., 2022)](https://arxiv.org/abs/2211.01324)
- [ERNIE-ViLG 2.0 (Feng et al., 2022)](https://arxiv.org/abs/2210.15257)
- [MoMa (Lin et al., 2024)](https://arxiv.org/abs/2405.09818)
- [Chameleon (2024)](https://arxiv.org/abs/2405.09818)
- [Transfusion (Zhou et al., 2024)](https://arxiv.org/abs/2408.11039)
- [MaskDiT (Zheng et al., 2023)](https://arxiv.org/abs/2306.09305)
- [MM1 (McKinzie et al., 2024)](https://arxiv.org/abs/2403.09577)
- [VeCLIP (Lai et al., 2023)](https://arxiv.org/abs/2310.07699)
- [PartiPrompts (Yu et al., 2022)](https://arxiv.org/abs/2206.02876)
- [MS-COCO (Lin et al., 2015)](https://arxiv.org/abs/1405.0312)
- [FLUX.1 (Black Forest Labs, 2024)](https://blackforestlabs.ai/)
- [RMSProp (Hinton, 2012)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
