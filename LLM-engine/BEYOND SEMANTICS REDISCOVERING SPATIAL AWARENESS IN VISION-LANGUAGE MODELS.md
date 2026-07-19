---
source_pdf: BEYOND SEMANTICS REDISCOVERING SPATIAL AWARENESS IN VISION-LANGUAGE MODELS.pdf
paper_sha256: 974eaf9cd3eb75dfed90e98962ce3f8dfbdbf71ef5af1488e06b701cb88f7a58
processed_at: '2026-07-18T17:49:14-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Beyond Semantics: Rediscovering Spatial Awareness in VLMs — 深度解读

Andrej，这篇paper从 mechanistic interpretability 视角切入，回答一个你可能在 nanoGPT / LLaVA 讨论里反复琢磨过的问题：**为什么 VLMs 像 "bag-of-words" 一样对空间顺序不敏感？**作者给了一个相当干净的答案——**embedding norm skew 压制了 RoPE**。下面我尽量按你喜欢的"build intuition"方式逐层拆解。

---

## 1. Paper 的核心问题与起点

Vision-Language Models (VLMs) 比如 LLaVA-1.5-7B，在 object recognition、image captioning 上接近 SOTA，但在最简单的 "left/right/above/below" 这类 spatial reasoning 上跌得很惨。Kamath et al. 2023 (https://openreview.net/forum?id=RN5KLywTll) 和 Tong et al. 2024 (https://arxiv.org/abs/2401.06209) 都系统性地记录了这个现象。

作者借用神经科学的 "two-stream hypothesis" (Goodale & Milner 1992, https://www.sciencedirect.com/science/article/pii/0166223692903448)：
- **Ventral stream** (颞叶) → "what" 通路 → 识别物体
- **Dorsal stream** (顶叶) → "where" 通路 → 空间关系

VLMs 看起来 ventral 强、dorsal 弱，作者追问：**"Where does space live in VLMs?"**

空间信息在 LLaVA-style 架构里有两个来源：
1. **Vision encoder** (CLIP ViT) 的 feature map + token order
2. **LLM 的 position embedding** (RoPE)

直觉上两条路都应该把空间信息送进 decoder，但实际整合失败。作者先做了两个 motivating experiments 证明失败有多严重。

---

## 2. Motivating Experiments：VLM 是"bag-of-tokens"

### 2.1 Permutation Test

作者在 vision encoder + MLP projector 之后、LLM 之前，**随机打乱 vision token 的顺序**。等价于给每个 vision token 重新分配一个 RoPE 相位。结果 (Table 1):

| Dataset | Original | Permuted | Δ |
|---|---|---|---|
| VQAv2 | 78.20 | 77.35 | -0.85 |
| POPE | 87.30 | 87.10 | -0.20 |
| GQA | 61.36 | 58.62 | -2.74 |
| CV-Bench 2D | 56.59 | 56.26 | -0.33 |

掉点几乎可以忽略！这意味着 RoPE 对 vision tokens 来说几乎是无效的——你 shuffle 它，模型根本不在乎。

### 2.2 Compression Test

更极端：用 average pooling 把 vision tokens 从 576 → 256 → 64 → 16 → **1**。结果 (Figure 2) 最差只掉 8.5%。99.8% 的空间分辨率被抹掉，模型照样能回答 VQAv2 / POPE / GQA / CV-Bench 的问题。

这两个实验结合起来说明两件事：
1. **RoPE 在 LLM 一侧基本没起作用**；
2. **现有 benchmark 本身可能也不真的测空间**——很多问题用 "semantic shortcut" (object co-occurrence + language prior) 就能蒙对。

这个观察和 Yuksekgonul et al. 2022 "When and why VLMs behave like bags-of-words" (https://arxiv.org/abs/2210.01936) 的早期发现一致，只是后者是 CLIP-style contrastive model，这里是 generative LLaVA。

---

## 3. 关键诊断：Embedding Norm Suppression

这是 paper 的核心 finding。作者在 COCO val (5k) 上统计 **vision token 与 text token 的 $L_2$ norm 分布** (Figure 3)：

- Vision tokens: $10^1 \sim 10^3$
- Text tokens: $3 \times 10^{-1} \sim 10^0$

差了 **1-3 个数量级**。这一点很关键，因为 LLaMA 用的是 **pre-Norm** 架构：

```
h^(l+1) = r^(l) + a^(l)(φ)
       = h^(l) + Attn(RMSNorm(h^(l)), φ)
```

其中：
- $h^{(l)} \in \mathbb{R}^d$ 是 layer $l$ 的 hidden state，$d$ 是 model dimension (LLaMA-7B 里 $d=4096$)
- $r^{(l)} = h^{(l)}$ 是 residual carry（未归一化的原始 hidden state）
- $a^{(l)}(\phi)$ 是 attention sublayer 输出，$\phi$ 是 RoPE 相位
- RMSNorm 只在 attention 内部对 input 归一化，**残差本身没归一化**

Figure 4 给出 layer-wise 的 vision/text hidden state norm 和 ratio：在 early layers (0-15)，vision 的 norm 比 text 大一个数量级。这个比例直到 mid-depth 才衰减下来。

### 3.1 为什么这会压制 RoPE？

这里作者给了一个相当 elegant 的 directional lemma (Appendix A)。

设 $u(\phi) = h^{(l+1)} / \|h^{(l+1)}\|$ 是 hidden state 的方向（注意 logits 是 inner product，方向决定行为）。对 $\phi$ 求偏导：

$$
\frac{\partial u}{\partial \phi} = \frac{(I - uu^\top)}{\|h^{(l+1)}\|} \cdot \frac{\partial a^{(l)}}{\partial \phi}
$$

变量解释：
- $u \in \mathbb{R}^d$：归一化后的 hidden state 方向
- $I - uu^\top \in \mathbb{R}^{d \times d}$：投影矩阵，把变化投到 $u$ 的正交补空间（保证方向单位长度）
- $\|h^{(l+1)}\|$：hidden state 的 norm
- $\partial a^{(l)} / \partial \phi$：attention 输出对 RoPE 相位的敏感度

当 $\|r^{(l)}\| \gg \|a^{(l)}\|$ 时，$\|h^{(l+1)}\| \approx \|r^{(l)}\|$ 很大，于是 $\partial u / \partial \phi \propto 1/\|r^{(l)}\|$ 被压得很小。

直觉：**RoPE 改的是 attention 输出 $a^{(l)}$ 的方向，但这个变化被一个庞大的、未归一化的残差 $r^{(l)}$ 稀释掉了。** 就像往大海里倒一杯盐水——盐度变了，但测不出来。

这跟你在 nanoGPT 教程里讲 residual stream 时强调的 "residual stream 是 highway，sublayer 是 small updates" 的设计哲学正好相反——highway 太宽，update 太小，update 就消失了。

### 3.2 Attention-level 的版本

作者还推了一个 attention-level 的版本（Eq. 1, 4, Appendix B）。把 tokens 分成 vision $V$ 和 text $T$ 两组：

- $\alpha_v = \text{softmax}(\ell)_v$ 是 token $v$ 上的 attention weight
- $\alpha_V = \sum_{v \in V} \alpha_v$ 是 vision 组的总 attention mass
- $\alpha_T = 1 - \alpha_V$
- $g_V = \sum_{v \in V} \frac{\alpha_v}{\alpha_V} \frac{\partial \ell_v}{\partial \phi}$ 是 vision 组的 **group-average logit derivative**，衡量 vision logits 对相位的内在敏感度
- $g_T$ 同理

那么 vision attention mass 对相位的变化率是：

$$
\boxed{\frac{\partial \alpha_V}{\partial \phi} = \alpha_V \cdot \alpha_T \cdot (g_V - g_T)}
$$

推导（Appendix B）走的是标准 softmax 求导：

$$
\frac{\partial \alpha_v}{\partial \phi} = \alpha_v \left( \frac{\partial \ell_v}{\partial \phi} - \sum_k \alpha_k \frac{\partial \ell_k}{\partial \phi} \right)
$$

把所有 $v \in V$ 加起来：

$$
\frac{\partial \alpha_V}{\partial \phi} = \alpha_V \left( g_V - \sum_k \alpha_k \frac{\partial \ell_k}{\partial \phi} \right)
$$

注意到 $\sum_k \alpha_k \frac{\partial \ell_k}{\partial \phi} = \alpha_V g_V + \alpha_T g_T$，代入化简：

$$
\frac{\partial \alpha_V}{\partial \phi} = \alpha_V (g_V - \alpha_V g_V - \alpha_T g_T) = \alpha_V \alpha_T (g_V - g_T)
$$

这个公式非常重要，因为它告诉我们 attention 对相位的敏感度被三个因子乘起来：
1. $\alpha_V$：vision 已经拿到的 attention mass
2. $\alpha_T = 1 - \alpha_V$：text 拿到的 attention mass（"未被占满"的部分）
3. $g_V - g_T$：vision logits 与 text logits 的相位敏感度差

**关键 insight**：如果 vision norm 远大于 text norm，那么 vision 早就把 attention mass 抢光了（$\alpha_V \to 1, \alpha_T \to 0$），整个导数趋零。换句话说，norm skew 不仅在 residual 层压制 RoPE，还在 attention 层通过 saturation 压制 RoPE。**双重打击。**

这跟 attention sinks 现象 (Xiao et al. 2024, https://openreview.net/forum?id=NG7sS51zVF) 有点像——某些 token 不正常地吸走大量 attention，但这里是因为 norm 大而非语义重要。

---

## 4. Interpretability Toolkit：三个 probe

作者开发了三个互补的工具。

### 4.1 Position Sensitivity Index (PSI)

$$
\text{PSI} = \frac{\text{Acc}(\text{original}) - \text{Acc}(\text{permuted})}{\text{Acc}(\text{original})}
$$

- PSI = 0：完全 order-invariant（bag-of-tokens）
- PSI 大：依赖 token 顺序

PSI 同时是模型指标和数据集指标——固定模型可以横向比数据集，固定数据集可以横向比模型。

### 4.2 Cross-Modality Balance (CMB)

对 attention head $h$：

$$
\text{CMB}_h = \frac{\sum_{v \in V} a_{v,h}}{\sum_{i \in V \cup T} a_{i,h}} \in [0, 1]
$$

- $V, T$ 是 vision 和 text 的 token index 集合
- $a_{v,h}$ 是 head $h$ 在 token $v$ 上的 attention weight
- CMB 接近 1：head 几乎只看 vision
- CMB 接近 0：head 几乎只看 text

按 layer × head 做 heatmap (Figure 7)。

这里有一个 **非常重要的方法论 trick**：之前的工作 (Chen et al. 2024, 2025) 报告 "vision attention 很低"——其实是因为 system prompt 把 attention 吸走了（72.1%！）。一旦把 system prompt 从分析中剔除，原始 LLaVA 其实是 **vision-dominant** 的。但这不是 healthy dominance，而是 norm skew 导致的 "brute force" dominance——query 默认偏向 vision key，不是 question-conditioned retrieval。

这个发现让我想到 Anthropic 的 induction heads 工作 (Olsson et al. 2022, https://arxiv.org/abs/2209.11895)——functional behavior 是 head-specific 的，所以 head-level 分析比 layer-level 更 informative。

### 4.3 RoPE Sensitivity Probe

直接测公式 (4) 里的两个量：

1. **Attention-level**: 固定 query，对 vision keys 加一个额外的 RoPE rotation $\Delta$ 步，重新算 attention，看 $\Delta \alpha_V = \alpha_V(\Delta) - \alpha_V(0)$
2. **Logit-level**: 直接测 $\Delta g_V$，group-average logit derivative

这相当于公式 (4) 的 finite-difference 近似：

$$
\Delta \alpha_V \approx \alpha_V (1 - \alpha_V) \Delta g_V
$$

(这是 Eq. 11，把 $\alpha_T$ 用 $1-\alpha_V$ 代入，假设 $g_T$ 不变因为只 rotate vision keys)

### 4.4 2DS Synthetic Benchmark

作者自己造了一个数据集 2DS，去除 semantic shortcut：
- 2-6 个 colored shapes，随机放在画面里
- 问题遍历 {color, shape, color+shape} × {absolute, relative}
- 例："What color is at the bottom?" / "Is the circle below the square?"

共 500 张图、3000 个问题。2DS 的 PSI 显著高于标准 benchmark (Table 2: 41.07% vs VQAv2 的 1.09%)——说明 2DS 真的逼模型用 position。

---

## 5. 两个 Intervention

基于诊断，作者提出两个 minimal intervention：

### 5.1 +Normalize (针对 H1: norm skew)

在 projector 之后、LLM 之前，对 vision embedding 做 RMSNorm，calibrate 到 text token 的典型 norm：
- text token mean norm ≈ 0.83
- text token max norm ≈ 1.22

这相当于把 vision embedding 从 $10^1 \sim 10^3$ 压到 $O(1)$，让 $\|r^{(l)}\|$ 在 early layers 不再碾压 $\|a^{(l)}\|$。

### 5.2 +Normalize+Multilayer (针对 H2: mid-layer spatial richness)

把 vision encoder 的 layer 12, 16, 20, 24 的 features concat 起来（每个 1024-dim，合起来 4096-dim），再过 projector。依据是 Wang et al. 2023 (https://arxiv.org/abs/2305.12223) 和 Jiang et al. 2024 (https://arxiv.org/abs/2310.08825) 的观察：CLIP ViT 的 final layer 偏 semantic，intermediate layers 保留更多 geometry / local layout。

这与 "DINO vs CLIP for VLM" 那条研究线相关——DINO 的 dense feature 对 spatial grounding 更友好。

---

## 6. 实验结果：两个机制是互补的

### 6.1 PSI (Table 2)

| Dataset | LLaVA 1.5 | +Normalize | +Normalize+Multilayer |
|---|---|---|---|
| VQAv2 | 1.09 | 2.03 | 2.08 |
| POPE | 0.34 | 0.35 | 0.46 |
| GQA | 4.47 | 6.10 | 5.64 |
| CV-Bench 2D | 0.57 | 11.31 | 9.72 |
| **2DS** | **41.07** | **61.21** | **54.21** |

+Normalize 在 PSI 上提升最显著（CV-Bench 2D 从 0.57 跳到 11.31，差不多 20×）。这印证 H1：scale match 之后 RoPE 重新起作用。

但注意 +Normalize+Multilayer 的 PSI 反而比 +Normalize 略低（2DS: 54.21 < 61.21）——说明它走的不是 RoPE 这条路。

### 6.2 Accuracy (Table 3, 2DS)

| Category | LLaVA 1.5 | +Normalize | +Normalize+Multilayer |
|---|---|---|---|
| Color_abs | 79.60 | 88.00 (+8.40) | 87.80 (+8.20) |
| Color_rel | 37.00 | 42.40 (+5.40) | 47.20 (+10.20) |
| Shape_abs | 78.60 | 79.00 (+0.40) | 82.20 (+3.60) |
| Shape_rel | 34.40 | 32.80 (-1.60) | 40.80 (+6.40) |
| Shape_color_abs | 70.80 | 71.80 (+1.00) | 80.20 (+9.40) |
| Shape_color_rel | 39.40 | 41.80 (+2.40) | 50.60 (+11.20) |
| **Overall** | **56.63** | **59.30 (+2.67)** | **64.80 (+8.17)** |

模式很清晰：
- **+Normalize** 主要帮助 color-related（final layer 保留 color cue 好）
- **+Normalize+Multilayer** 在 relative relation 上跳得最猛（intermediate layers 提供 geometry）

PSI 和 accuracy 是 **互补** 的两个维度：
- PSI 测 "model 用不用 token order / RoPE"
- Accuracy 测 "model 的 spatial competence"

+Normalize 提 PSI 不提 accuracy 太多；+Multilayer 提 accuracy 不提 PSI——一个是让 LLM 重新接管 spatial，一个是让 vision encoder 接管 spatial。

### 6.3 RoPE Sensitivity (Figure 8)

在 layer 0-5，+Normalize 和 +Normalize+Multilayer 的 $\Delta \alpha_V$ 和 $\Delta g_V$ 都显著高于 baseline。这直接验证公式 (4)：**norm 降下来之后，early layer 的 RoPE 效应重新出现。**

### 6.4 CMB Heatmap (Figure 7)

- **Baseline**: vision-focused heads 散落各层，无清晰 pipeline
- **+Normalize**: early layers vision-heavy → deeper layers text-heavy（smooth vision→text shift）
- **+Normalize+Multilayer**: striking stage-like pattern——vision → text → vision（extract → align → integrate）

这个 stage-like pattern 让我想到 Bietti et al. 2023 "Birth of a Transformer" (https://arxiv.org/abs/2306.00802) 里讲的 memory → induction → function 的 layer-wise emergence。

### 6.5 Attention Entropy (Table 8)

| Dataset | LLaVA 1.5 | +Normalize | +Normalize+Multilayer |
|---|---|---|---|
| 2DS | 0.76 | 0.90 | **0.72** |
| COCO Val | 0.81 | 0.89 | **0.72** |
| CV-Bench 2D | 0.80 | 0.90 | **0.72** |

+Normalize 反而 entropy 更高（更 diffuse），+Multilayer 最低（最 focused）。这进一步说明两个机制不同：
- +Normalize 让 LLM "主动探索" spatial positions
- +Multilayer 让 vision encoder 已经把空间信息打包好，LLM 只需要 selectively retrieve

---

## 7. 与你 (Karpathy) 工作的联想

### 7.1 Residual Stream 的"highway"哲学

你在 nanoGPT 和 "Intro to LLMs" 视频里反复强调：**Transformer 的 residual stream 是一条 highway，每个 sublayer 只是 small additive update**。这篇 paper 给这条哲学敲了一个警钟——当不同 modality 的 residual scale 差 1-3 个数量级，highway 就被 dominant modality 占领了，small update 直接被淹没。

这其实是一个 general 的 multimodal 训练问题，不止 LLaVA。任何把两个 pre-trained encoder 接到一起的系统，如果没有显式的 scale alignment，都会遇到类似问题。CLIP-style contrastive training 里两边都被 L2-normalize 过，所以没这问题；LLaVA 直接把 raw embedding 投影过去，就中招了。

参考链接：
- nanoGPT: https://github.com/karpathy/nanoGPT
- "Intro to LLMs" video: https://www.youtube.com/watch?v=zjkBMFhNj_g

### 7.2 Pre-Norm 的代价

LLaMA / LLaVA 用 pre-Norm 是为了训练稳定，但它有个副作用——**残差流从未归一化**。Post-Norm (原始 Transformer 论文 Vaswani et al. 2017, https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) 在每个 sublayer 输出后归一化，反而能控制 norm 增长。

这跟 Xiong et al. 2020 "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/abs/2002.04745) 的理论分析吻合——pre-Norm 的 good conditioning 是以 "norm 可以任意增长" 为代价的。

### 7.3 "Bag of words" 和 CLIP 的老问题

Yuksekgonul et al. 2022 (https://arxiv.org/abs/2210.01936) 早就指出 CLIP-style encoders 有 bag-of-words 倾向。这篇 paper 把这个问题追踪到 LLM 一侧——**即使 LLM 有 RoPE，也被 norm skew 关掉了**。这给 "为什么 multimodal training 后空间信息会消失" 提供了一个新的、具体的 mechanism。

### 7.4 与 mechanistic interpretability 的连接

CMB heatmap 那种 "stage-like vision→text→vision" 模式，跟 Anthropic 的 circuits.thread (https://transformer-circuits.pub/) 和 induction heads 的分析思路完全一致。作者把 attention head 当 functional unit 来分析，比 layer-level 分析粒度更细。

Li et al. 2023b "Inference-time Intervention" (https://arxiv.org/abs/2306.03341) 也是在 head granularity 做 intervention。这篇 paper 的 +Normalize 是 representation-level intervention，但思路同源——**找到 mechanism，做最小干预，看行为变化。**

### 7.5 Vision encoder 的 layer hierarchy

Jiang et al. 2024 "From CLIP to DINO" (https://arxiv.org/abs/2310.08825) 指出 VLM 的 vision encoder 选择很关键：CLIP final layer 偏 semantic，DINO 偏 spatial。这篇 paper 给了另一条路——**不换 encoder，从 CLIP 中间层取 feature**。Layer 12, 16, 20, 24 的 concat 等于把 semantic 和 spatial 都喂给 LLM，让 LLM 自己挑。

这和 you-draw-it / micrograd 那种 "show me the gradient" 的直觉一致：**不同 layer 编码不同信息，混着用往往比单用 final layer 强。**

### 7.6 SpatialVLM / SpatialRGPT / Cambrian-1 的对比

最近的几个改进 spatial reasoning 的工作：
- SpatialVLM (Chen et al. 2024, https://arxiv.org/abs/2401.02388): 用 internet-scale spatial annotation 做 SFT
- SpatialRGPT (Cheng et al. 2024, https://openreview.net/forum?id=JKEIYQUSUc): 加 depth-aware feature
- Cambrian-1 (Tong et al. 2024, https://openreview.net/forum?id=Vi8AepAXGy): vision-centric 设计

这篇 paper 走了不同路线——**不动 data，不动 encoder，只做 representation-level intervention**。更像是 "diagnostic-first" 而非 "SOTA-chasing"。你之前在 "Software 2.0" 和 "Recipe for training neural networks" 里强调过，理解 failure mode 比堆 benchmark 重要，这篇 paper 的 taste 很接近。

### 7.7 Attention Sinks 的联系

System prompt 吸走 72% attention 这个发现，和 Xiao et al. 2024 "Efficient Streaming Language Models with Attention Sinks" (https://openreview.net/forum?id=NG7sS51zVF) 直接相关。System prompt 在 VLM 里相当于一个 "image-invariant attention sink"——它解释了为什么之前的工作看到 vision attention 很低。

CMB 把 system prompt 从 user text 里分离出来，这是一个方法论上的进步——**做 modality balance 分析时，必须把固定 prompt 和动态 user input 分开**，否则统计全错。

### 7.8 与 "Platonic Representation Hypothesis" 的连接

Huh et al. 2024 "The Platonic Representation Hypothesis" (https://proceedings.mlr.press/v235/huh24a.html) 论证不同模型收敛到同一个 representation。这篇 paper 的 norm skew 发现给了一个反例——**vision 和 text 即使方向 (cosine) 上 aligned，magnitude 不对齐也会破坏下游计算**。Platonic 假设关注 direction，这篇 paper 提醒我们 magnitude 也重要。

### 7.9 你可能想做的 follow-up

如果你看完想动手，几个方向：
1. **把 norm matching 推广到 LLaVA-NeXT / Qwen2.5-VL / InternVL**——这些模型 vision encoder 可能从 scratch 训练，norm skew 也许不同
2. **Per-head adaptive normalization**——paper 里 normalize 是 "blunt" 的，作者自己承认可能伤 3D cues。给每个 head 学一个 scale 参数，可能更好
3. **在 attention 里直接做 group-norm**——既然 $\alpha_V \alpha_T$ 是 saturation factor，可以在 softmax 前对 vision / text logits 分别 normalize，强行打破 saturation
4. **Token-level dynamic normalization**——某些 vision token (背景) 应该被压，某些 (foreground object) 应该保留。学习一个 per-token scalar
5. **在 nanoGPT 上 reproduce**——你可以用 nanoGPT + 一个小 vision encoder 复现这个 norm skew 现象，PSI / CMB / RoPE probe 三个工具都很轻量

---

## 8. 局限性与批评

作者自己承认：
- Normalize 是 "blunt" 的，可能损害 3D cues 的 dynamic range
- 只在 LLaVA-1.5 上验证，新模型 (Qwen2.5-VL, Cambrian-1) 可能没问题
- 没有在更大 scale 上验证 (70B+)

我额外想说的：
- **2DS 太简单**——只有 colored shapes，real-world spatial reasoning 复杂得多（occlusion, 3D, perspective）
- **Norm calibration 值 (0.83, 1.22) 是经验值**，没给理论依据。不同 LLM backbone 可能需要重新 calibrate
- **Multilayer concat 的 layer 选择 (12, 16, 20, 24) 也是经验值**，没做 ablation 说为什么这四层
- 没分析 **为什么** vision encoder 输出 norm 这么大——CLIP training 用 contrastive loss，本身没 norm constraint；projector MLP 学出来放大了 norm。这是 training dynamics 问题，paper 没深入

---

## 9. 一句话总结

**Vision tokens 的 L2 norm 比 text 大 1-3 个数量级，在 pre-Norm 架构下，这个 norm skew 同时在 (a) residual stream 层稀释 RoPE 对 hidden state 方向的影响，(b) attention 层通过 saturation 让 vision 抢光 attention mass，于是 RoPE 对 vision tokens 几乎无效，VLM 退化成 bag-of-tokens。RMSNorm 把 vision scale 拉回 text 量级，可以同时修复这两个问题；multilayer feature concat 则走另一条路，让 vision encoder 直接打包好 spatial 信息，绕过 RoPE。**

---

## 主要参考链接

- Paper website: https://user074.github.io/respatialaware/
- LLaVA: https://github.com/haotian-liu/LLaVA
- RoFormer (RoPE): https://arxiv.org/abs/2104.09864
- RMSNorm: https://arxiv.org/abs/1910.07467
- Attention Sinks: https://openreview.net/forum?id=NG7sS51zVF
- "Bag-of-words VLMs": https://arxiv.org/abs/2210.01936
- "What's up with VLMs": https://openreview.net/forum?id=RN5KLywTll
- "Eyes wide shut": https://arxiv.org/abs/2401.06209
- Induction heads: https://arxiv.org/abs/2209.11895
- Birth of a Transformer: https://arxiv.org/abs/2306.00802
- Inference-time Intervention: https://arxiv.org/abs/2306.03341
- Platonic Representation Hypothesis: https://proceedings.mlr.press/v235/huh24a.html
- From CLIP to DINO: https://arxiv.org/abs/2310.08825
- SpatialVLM: https://arxiv.org/abs/2401.02388
- Cambrian-1: https://openreview.net/forum?id=Vi8AepAXGy
- Goodale & Milner 1992: https://www.sciencedirect.com/science/article/pii/0166223692903448
- Attention is All You Need: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- On Layer Normalization in Transformer: https://arxiv.org/abs/2002.04745
- VLM-Visualizer (Zhang 2024): https://github.com/zjysteven/VLM-Visualizer
- CLIP: https://arxiv.org/abs/2103.00020
- nanoGPT: https://github.com/karpathy/nanoGPT
- Intro to LLMs video: https://www.youtube.com/watch?v=zjkBMFhNj_g
