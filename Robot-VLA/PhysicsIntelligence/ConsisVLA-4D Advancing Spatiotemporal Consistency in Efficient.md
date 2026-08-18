---
source_pdf: ConsisVLA-4D Advancing Spatiotemporal Consistency in Efficient.pdf
paper_sha256: 995398cc069303e5eb1d066dc5f7ea149ccabd93b8fab45dfbe6b67117350756
processed_at: '2026-08-18T03:54:01-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 你好。把这篇 paper 翻译成大白话，核心就是教 robot 像人一样“带着目的看东西，多角度拼凑立体感，并且在动手前能在脑子里预演动作后果”。

现在的 VLA 模型（比如 OpenVLA）看图就是死磕 2D 像素，这会导致两个致命问题：第一，换个角度看同一个杯子，模型就认不出这是同一个东西了；第二，模型只會预测下一帧长啥样，不会预测动作发生后 3D 场景会怎么变。

ConsisVLA-4D 的核心 philosophy 就是把人的感知过程拆成三步，对应三个 module：

### 1. CV-Aligner：听指令找重点，多视角认同一物体

**人话**：人看到一堆杂物，听到“拿碗”，眼睛会自动忽略背景，只盯着碗看。而且不管左眼看还是右眼看，大脑都知道这是同一个碗。CV-Aligner 就是干这个的。

**技术拆解**：
它先拿 instruction 去跟 image 的 patch token 算 cosine similarity，只保留最相关的 32 个 token（原本有 256 个，直接砍掉 7/8）。然后，它用一个叫 VGGT 的 frozen 3D 模型提取 3D feature，通过 cross-attention 注入进來。

公式表达：
$$\mathbf{z}_i^{\text{obj-3D}} = f_{\text{SF}}(f_{\text{ES-S}}(\mathbf{z}_i^{\text{sem}}, \mathbf{t}), \mathbf{z}_i^{\text{3D}})$$

- $i \in \{M, L, R\}$：代表 Main, Left, Right 三个视角的下标。
- $\mathbf{z}_i^{\text{sem}}$：SigLIP 提取的语义 feature。
- $\mathbf{t}$：instruction 的 text embedding。
- $f_{\text{ES-S}}$：Explicit Semantic Object Selection，也就是算 similarity 取 Top-K 的操作。
- $\mathbf{z}_i^{\text{3D}}$：VGGT 提取的 3D feature。
- $f_{\text{SF}}$：Single-Fusion，也就是 cross-attention 融合。

**Intuition**：VGGT 预训练时学过 point tracking，所以当 Main view 的“碗” token 和 Left view 的“碗” token 都去 attend VGGT 的 3D feature 时，它们会隐式地对齐到 3D 空间里的同一组点。这就实现了 cross-view 的 object identity consistency。

### 2. CO-Fuser：多角度拼凑，搞清物体间的空间关系

**人话**：一只眼看距离容易看错（scale ambiguity），双眼看就准了。CO-Fuser 就是把多个视角的几何信息揉在一起，搞清楚“碗在盘子左边，盘子在微波炉里面”这种相对空间关系。

**技术拆解**：
它把 DINOv2 的 geometric feature 和 VGGT 的 3D feature 逐层混合。混合时用了一个 cosine decay 权重 $\alpha_l$。

公式表达：
$$\alpha_l = \psi \cdot \left(\delta + (1-\delta) \cdot \frac{1 + \cos\left(\frac{l\pi}{\mathcal{L}'}\right)}{2}\right)$$
$$\mathbf{z}_l^{\text{geo-3D}} = (1 - \alpha_l) \odot \mathbf{z}_l^{\text{geo}} + \alpha_l \odot \mathbf{z}_l^{\text{3D}}$$

- $l$：transformer 的 layer 层数下标。
- $\mathcal{L}'$：总层数（24层）。
- $\psi$：最大权重（0.2），$\delta$：最小权重因子（0.01）。
- $\alpha_l$：随层数变化的融合权重。在浅层 ($l \to 0$) 和深层 ($l \to \mathcal{L}'$) 变化很慢，在中间层 ($l \approx \mathcal{L}'/2$) 变化最快。

**Intuition**：浅层吸收 low-level 的 3D prior（比如 depth edge），深层做 high-level 的 spatial relation abstraction，这两头都需要稳定的 prior 约束，所以 $\alpha_l$ 变化要慢。中间层是过渡区，变化快一点无妨。如果用 linear decay，整个网络都在均匀抖动，optimization 会不稳定。作者在 Table 8 做了实验，linear decay 会让 LIBERO 成功率掉 3.7%。

然后，它初始化了 64 个 learnable 的 aggregation token，用 block-wise causal attention 把多视角的几何信息全吸进这 64 个 token 里。Causal 的意思是 aggregation token 只能单向去看 geometric token，不能反过来污染源特征。

### 3. CS-Thinker：在脑子里彩排动作后果

**人话**：人伸手拿杯子前，大脑其实已经预演了杯子会被抓起来移动到哪。CS-Thinker 就是让模型在 training 时学会“脑补”未来的动态变化和深度变化，inference 时全凭直觉输出 action。

**技术拆解**：
它训练时加两个 auxiliary task：
1. 预测 action 发生后，目标 object 在某个固定视角的 feature（由 CoTracker 提供伪标签）。
2. 预测 action 发生后，全局的 depth feature（由 Depth Anything 提供伪标签）。

总 loss 公式：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{dyn-4D}} + \mathcal{L}_{\text{dep-4D}}$$

- $\mathcal{L}_{\text{action}}$：对 action chunk 的 L1 loss。
- $\mathcal{L}_{\text{dyn-4D}}$：预测 future dynamic object 的 L2 loss。
- $\mathcal{L}_{\text{dep-4D}}$：预测 future global depth 的 L2 loss。

**Intuition**：这就是一个 implicit world model。它在 latent space 里做预测，绕开了生成 2D image 的巨大计算开销。Inference 时，dynamic decoder 和 depth decoder 全砍掉，只保留它们在 SC-Attn (Spatiotemporal Consistency Attention) 里留下的 latent representation 来指导 action 生成。这就像人学会了骑车后，就不用脑子再刻意去想平衡公式了，肌肉记忆直接出。

---

### 数据说话：又快又好

这套设计带来的效率提升极其夸张。虽然加了 VGGT 的 2B 参数，但因为 visual token 从 768 降到了 84 左右，LLM 主干的计算量暴跌。

| Method | Latency | Throughput | FLOPs | LIBERO Avg |
| -------- | -------- | -------- | -------- | -------- |
| OpenVLA | 0.254s | 3.9 Hz | 8.48T | 76.5% |
| OpenVLA-OFT | 0.137s | 58.4 Hz | 8.45T | 97.1% |
| **ConsisVLA-4D** | **0.110s** | **72.7 Hz** | **4.59T** | **98.1%** |

FLOPs 直接砍半（8.48T $\to$ 4.59T），Throughput 翻倍。在真机双臂任务上，Throughput 甚至达到了 108.2 Hz，完全满足实时控制需求。

### 给你的 Intuition 联想

Karpathy，你可能觉得这套设计有点眼熟。CS-Thinker 的 "training-only auxiliary loss" 哲学，其实跟你在讲 LLM 时提到的 System 1 / System 2 思维很像。

Training 阶段，模型开启 System 2：费力去预测 future depth、future object trajectory，建立对世界运作规律的理解。Inference 阶段，模型切回 System 1：砍掉所有显式预测头，只靠之前学到的 latent representation “直觉”地输出 action。

这与 [V-JEPA 2](https://arxiv.org/abs/2506.09985) 的 latent space prediction 哲学一脉相承，但 ConsisVLA-4D 更进一步，它把 3D geometry (VGGT) 和 semantic (SigLIP) 解耦开，分别用不同的 pathway 去 reason dynamics 和 depth，最后在 SC-Attn 层汇合。这种 modular 的设计，比把所有东西扔进一个大 transformer 里硬学要高效得多。

参考链接：
- [ConsisVLA-4D GitHub](https://github.com/JiuTian-VL/ConsisVLA-4D)
- [VGGT](https://vgg-t.github.io/)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [OpenVLA-OFT](https://openvla-oft.github.io/)
- [Depth Anything V2](https://depth-anything.com/)
- [CoTracker 3](https://co-tracker.github.io/)

---

# ConsisVLA-4D: 从 3D-Perception 到 4D-Reasoning 的 VLA 架构深度解析

Karpathy 你好，这篇 paper 是 HIT-Shenzhen 团队的工作，发表于 arxiv 2025，核心是把 VLA 模型的 perception-reasoning 范式从 2D 推到 3D，再推到 4D（带时间维度）。我把它拆解成 intuition、mechanism、experiment 三层来聊，希望对你 build intuition 有帮助。

---

## 一、核心 motivation：为什么 2D VLA 不够用？

主流 VLA（OpenVLA, π0, RT-2）走的是 **image → VLM → action token** 这条路径，但有两个本质缺陷：

**Spatial 维度的瓶颈**：2D 像素丢失了 depth 和 viewpoint invariance。同一个杯子，从 main view 看是椭圆，从 wrist view 看是圆，模型如果不能 cross-view 对齐 object identity，就会在不同视角下产生 inconsistent representation，导致抓取轨迹漂移。

**Temporal 维度的缺失**：现有方法做 future-frame prediction（WorldVLA、V-JEPA 2），但生成的还是 **2D 像素图**，而非 **3D scene state 的演化**。机器人 action 改变 scene 后，模型对未来 scene 的推演和当前 instruction 不对齐，spatiotemporal consistency 就崩了。

**人类类比**（paper Section 3）：人眼 retina 抓 2D，但 brain 在做 manipulation 时持续整合 spatial cue（binocular disparity、motion parallax）并预测未来 scene state。Paper 把这个过程形式化为：

$$2D \xrightarrow{\text{construction}} 3D \xrightarrow{\text{prediction}} 4D \quad (Eq.2)$$

这个公式是整篇 paper 的 backbone。Construction 阶段是 CV-Aligner + CO-Fuser，prediction 阶段是 CS-Thinker。

参考：[V-JEPA 2](https://arxiv.org/abs/2506.09985), [OpenVLA](https://openvla.github.io/), [π0](https://www.physicalintelligence.company/blog/pi0)

---

## 二、整体架构：四个 Paradigm 的演进

Figure 1 把现有工作分成四个 paradigm：

| Paradigm | 代表 | 输入形式 | 问题 |
|---------|------|---------|------|
| Para. A | 3D-VLA, PointVLA | point cloud / depth map / history frames 显式输入 | 需要额外 sensor，computational overhead 大 |
| Para. B | SpatialVLA, BridgeVLA, Evo-0 | 2D → 3D 投影 | projection bias、occlusion error |
| Para. C | WorldVLA | 2D 预测 3D 表示 | 还停留在 image-level generation |
| **Para. D (本文)** | ConsisVLA-4D | 2D → 3D perception → 4D reasoning | unified framework，仅用 1/8 visual token |

ConsisVLA-4D 的关键 insight：**不需要显式 point cloud 输入**，靠 multi-view 2D 图像 + frozen pretrained 3D encoder (VGGT) 的 latent prior，就能 implicitly 推出 3D 结构，并进一步推 4D 演化。

参考：[VGGT](https://vgg-t.github.io/), [SpatialVLA](https://arxiv.org/abs/2501.15830)

---

## 三、Preliminary：三种 visual feature 的语义分工

Paper 用了三个 frozen visual encoder，分工很明确：

### 1. SigLIP → z^sem (semantic feature)
$$\mathbf{z}^{\text{sem}} = f_v^{\text{SigLIP}}(\mathbf{x})$$
SigLIP 用 sigmoid loss（不是 InfoNCE 的 softmax），每个 image-text pair 独立判别，所以每个 visual token **z^{sem,j} 都继承了 linguistic semantics**。这是 CV-Aligner 做 instruction-guided filtering 的前提。

参考：[SigLIP paper](https://arxiv.org/abs/2303.15343)

### 2. DINOv2 → z^geo (geometric feature)
$$\mathbf{z}^{\text{geo}} = f_v^{\text{DINOv2}}(\mathbf{x})$$
DINOv2 用 self-supervised contrastive loss 对齐同一图像的 augmented views，所以 z^geo 捕获 **geometric consistency across views**。这是 CO-Fuser 做 cross-object geometric aggregation 的基础。

参考：[DINOv2](https://arxiv.org/abs/2304.07193)

### 3. VGGT → z^{3D} (3D-aware feature)
$$\text{DPT}(f_v^{\text{VGGT}}(\mathbf{x}_i)_{i=1}^M) = (D_i, P_i, G_i)_{i=1}^M \quad (Eq.1)$$

变量含义：
- **M**: 输入 RGB 图像数（multi-view，本文用 3 个 viewpoint: Main, Left, Right）
- **D_i**: 第 i 个 view 的 depth map
- **P_i**: point map（3D point cloud 在 camera coordinate）
- **G_i**: feature grid，专门为 point tracking 设计
- **DPT(·)**: Dense Prediction Head，来自 [Ranftl et al.](https://arxiv.org/abs/2103.13413)

VGGT 的 key property 是它的 feature grid G_i **天然支持 cross-view point tracking**（这是它预训练时就学到的），所以 CV-Aligner 用它做 cross-view object identity alignment 是 free-lunch。

参考：[VGGT 项目主页](https://vgg-t.github.io/)

---

## 四、CV-Aligner：Cross-View Object Semantic Consistency

目标：从 multi-view 2D 图像里抽出 instruction-relevant object，并在不同 viewpoint 间对齐 identity，最终只用 1/8 visual token。

### Step 1: FiLM Modulation（公式 8）

$$\widetilde{\mathbf{z}}_{i,l}^{\text{sem}} = (1 + \gamma(\mathbf{t})) \odot \text{Self-Attn}(\mathbf{z}_{i,l}^{\text{sem}}) + \beta(\mathbf{t}) \quad (Eq.8)$$

变量：
- **i ∈ {M, L, R}**: viewpoint index
- **l**: transformer layer index in SigLIP
- **γ(t), β(t)**: FiLM 的 scale 和 shift vector，由 instruction t 通过 MLP 投影到 visual embedding space 得到
- **⊙**: element-wise multiplication
- **1**: all-ones vector，保证 1+γ 是 multiplicative residual

直觉：FiLM 让 instruction 在 **每一层** 都参与 modulation，比只在 input 层做 cross-attention 更 deep。这种 design 借鉴自 [FiLM 原文](https://arxiv.org/abs/1709.07871)，在 visual reasoning 任务上效果显著。

### Step 2: ES-Selection（公式 9-12）

公式 9 把 z_i^sem 拆成 N_i 个 token：
$$\mathbf{z}_i^{\text{sem}} = [\mathbf{z}_i^{\text{sem},1}, \ldots, \mathbf{z}_i^{\text{sem},N_i}] \in \mathbb{R}^{N_i \times d_v}$$
- **N_i**: i-th view 的 patch token 数（SigLIP 默认 256）
- **d_v**: visual feature 维度

公式 10 算每个 token 和 instruction 的 cosine similarity：
$$s_{i,j} = \text{sim}(\mathbf{z}_i^{\text{sem},j}, \mathbf{W}_t \cdot \mathbf{t}) = \frac{\mathbf{z}_i^{\text{sem},j} (\mathbf{W}_t \cdot \mathbf{t})^\top}{\|\mathbf{z}_i^{\text{sem},j}\|_2 \cdot \|\mathbf{W}_t \cdot \mathbf{t}\|_2}$$
- **W_t**: text → viewpoint dim 的 mapping matrix（learnable）
- **t**: instruction embedding from f_t^SigLIP

公式 11: Top-K 筛选
$$\mathcal{S}_i = \text{Top-K}(\{s_{i,1}, \ldots, s_{i,N_i}\}, K)$$
- **K = 32**（默认），从 256 → 32，压缩 8×

公式 12: 最终的 object tokens
$$\mathbf{z}_i^{\text{obj}} = \mathbf{z}_i^{\text{sem},j}\big|_{j \in \mathcal{S}_i} = f_{\text{ES-S}}(\mathbf{z}_i^{\text{sem}}, \mathbf{t})$$

**Intuition**：这本质是 query-based token pruning。和 FastV、SliME（[FastV](https://arxiv.org/abs/2403.06764), [SliME](https://arxiv.org/abs/2406.08487)）的固定 layer-based pruning 不同，ES-Selection 是 instruction-conditioned，所以能在不同任务下抽出不同的 object。Table 7 ablation 显示 FastV 在 1/8 压缩下 LIBERO 掉到 88.8%（-9.3%），ConsisVLA-4D 反而 98.1%，差距 9.3% 就来源于这种 instruction-awareness。

### Step 3: Single-Fusion（公式 13）

$$\mathbf{z}_i^{\text{obj-3D}} = \big(\text{FFN}(\text{Cross-Attn}(\mathbf{z}_i^{\text{obj}}, \mathbf{z}_i^{\text{3D}})) + \text{Res}(\mathbf{z}_i^{\text{obj}})\big)\big|_{\text{Layer}=1,\ldots,N}$$

- **z_i^obj**: Query (32 tokens)
- **z_i^{3D}**: Key 和 Value (from VGGT)
- **N = 4**: 4 层 Transformer，hidden size 1152，16 heads，FFN 2752 dim

**Why this works**：z_i^obj 已经 instruction-filtered，但还缺 3D 信息。VGGT 的 z^{3D} 含 point tracking prior G_i，所以 cross-attention 让每个 object token 去"查询"它在 3D 空间里的对应点。这就是 cross-view identity alignment 的 mechanism：同一 object 在 M view 和 L view 的 token 都 attend 到 VGGT 里相同的 3D point，于是 implicit 地对齐了 identity。

---

## 五、CO-Fuser：Cross-Object Spatial Geometric Consistency

CV-Aligner 关注 "object identity"，CO-Fuser 关注 **object 之间的空间几何关系**（比如杯子在碗左边，盘子在杯子后面）。这种 relation 在 single-view 下有 scale ambiguity，需要 multi-view 融合。

### Step 1: Group-Fusion（公式 14-15）

公式 14: 逐层融合 DINOv2 和 VGGT 的 feature
$$\mathbf{z}_l^{\text{geo-3D}} = (1 - \alpha_l) \odot \mathbf{z}_l^{\text{geo}} + \alpha_l \odot \mathbf{z}_l^{\text{3D}} = f_{\text{GF}}(\mathbf{z}_l^{\text{geo}}, \mathbf{z}_l^{\text{3D}})$$
- **z_l^{geo}**: DINOv2 第 l 层 feature（learnable，fine-tune）
- **z_l^{3D}**: VGGT 第 l 层 feature（**frozen**）
- **α_l**: layer-wise weight

公式 15: α_l 用 cosine decay
$$\alpha_l = \psi \cdot \left(\delta + (1-\delta) \cdot \frac{1 + \cos\left(\frac{l\pi}{\mathcal{L}'}\right)}{2}\right)$$
- **ψ = 0.2**: max weight（at l=0, α_0 = ψ）
- **δ**: min weight factor，使 α_{L'} = ψ·δ = 0.01
- **L' = 24**: total layers
- **l**: layer index, 0 到 L'

**Why cosine, not linear?** Paper 在 Table 8 给了 ablation：linear decay 让 LIBERO SR 掉到 94.4%（-3.7%），real-world 掉到 73.3%（-5.0%）。

直觉解释：cosine decay 的导数是
$$\frac{d\alpha_l}{dl} = -\psi(1-\delta) \cdot \frac{\pi}{2\mathcal{L}'} \cdot \sin\left(\frac{l\pi}{\mathcal{L}'}\right)$$
- 当 l→0 或 l→L' 时，sin→0，导数 →0，**α 在浅层和深层都接近常数**
- 当 l ≈ L'/2 时，sin=1，**导数最大，α 变化最快**

这反映了 feature learning 的一个普遍规律：浅层吸收 low-level geometric prior（depth edge、surface normal），深层做 high-level abstraction（spatial relation）。在浅层和深层都需要 **稳定的 prior 约束**，所以 α 在那里变化要慢；中层是 prior 向 learned feature 过渡的关键区，变化要快。Linear decay 是均匀 removal，会导致 optimization discontinuity。

参考：[Layer-wise feature fusion 分析](https://arxiv.org/abs/2304.07193)

### Step 2: Aggregation Token Concatenation
初始化一组 learnable aggregation tokens **z_0^{agg-3D}**（64 个），concatenate 到 z_0^{geo-3D}。这个数量是固定的，单臂 1/8 压缩（256→32），双臂 1/12 压缩。

### Step 3: IG-Aggregation with Block-wise Causal Self-Attention（公式 16）

$$\begin{aligned}
(\mathbf{z}_{l+1}^{\text{geo-3D}}, \mathbf{z}_{l+1}^{\text{agg-3D}}) &= \text{BC-Attn}(\mathbf{z}_l^{\text{geo-3D}} \oplus \mathbf{z}_l^{\text{agg-3D}}) \\
&= f_{\text{IG-A}}(\mathbf{z}_l^{\text{geo-3D}}, \mathbf{z}_l^{\text{agg-3D}})
\end{aligned}$$

- **⊕**: token set concatenation
- **BC-Attn**: Block-wise Causal self-attention，attention pattern 是：
  - z_l^{geo-3D} ↔ z_l^{agg-3D}: **causal**（agg 只能看 geo，geo 不能看 agg，单向）
  - geo-3D 内部: bidirectional
  - agg-3D 内部: bidirectional

**Why causal not bidirectional?** 因为 agg-3D 是"被填充"的，geo-3D 是"源信息"。Causal 让 agg 持续从 geo 抽信息但不污染 geo，这样 geo feature 保持稳定，agg 渐进聚合 multi-view geometric relation。

最终只保留 z_{L'}^{agg-3D}（64 个 token），它 implicitly 包含了 multi-view 之间的 geometric relation，不需要显式 point cloud 输入。

参考：[Causal attention in transformers](https://arxiv.org/abs/1706.03762)

---

## 六、CS-Thinker：Cross-Scene Spatiotemporal Consistency

这是从 3D 到 4D 的关键跃迁。Paper 把它叫 "Cross-Scene Thinker"，意思是 scene 随 action 变化，model 要预测 future scene state。

### 6.1 Multi-View Objects → Single-View Dynamic Objects（公式 17-18）

公式 17: 初始化 dynamic tokens
$$\forall i \in \mathcal{T}, \quad \mathbf{0}_i^{\text{dyn-4D}} \xrightarrow{\text{guide}} (\mathbf{z}_i^{\text{obj-3D}}, \mathbf{t})$$

- **0_i^{dyn-4D}**: learnable dynamic tokens，每个 viewpoint 一组，|T| = 3 组，每组 4 个，共 12 个
- "guide": 这些 token 通过 SC-Attn 被 z_i^{obj-3D} 和 t 引导

公式 18: Training-only loss，预测 action 后某 fixed viewpoint i* 的 dynamic object
$$\mathcal{L}_{\text{dyn-4D}} = \left\|\left(\hat{\mathbf{z}}_{i^*}^{\text{dyn-4D}} \odot \mathbf{m}_{i^*}^{\text{obj-3D}}\right) - \left(\mathbf{z}_{i^*}^{\text{dyn-4D}} \odot \mathbf{m}_{i^*}^{\text{obj-3D}}\right)\right\|_2^2$$

- **i*** ∈ T**: 固定 viewpoint（比如 Main view）
- **⊙**: element-wise multiplication
- **m_{i*}^{obj-3D}**: mask，localize object position 在 z_{i*}^{obj-3D} 里
- **ẑ_{i*}^{dyn-4D}**: predicted dynamic object feature
- **z_{i*}^{dyn-4D}**: ground truth，用 [CoTracker 3](https://co-tracker.github.io/) 提供监督

**Intuition**：人抓起杯子时，杯子在 M view 的视觉特征会变化（位置、形态）。CS-Thinker 学到的是 "给定当前 multi-view object feature + instruction，预测 action 后 fixed view 上的 object 状态"。这是个 **implicit world model**，但不开 image generator，而是预测 feature-level dynamics。这个 supervision 来自 CoTracker 的 point trajectory，cheap 且 dense。

### 6.2 Abstract Relation → Concrete Global Depths（公式 19-20）

公式 19:
$$\mathbf{0}^{\text{dep-4D}} \xrightarrow{\text{guide}} (\mathbf{z}_{\mathcal{L}'}^{\text{agg-3D}}, \mathbf{t})$$

- **0^{dep-4D}**: 1 组 learnable depth tokens，4 个
- 这组 token 只被 z_{L'}^{agg-3D}（aggregated geometric relation）和 t 引导
- **重要**：和 z_T^{obj-3D} 和 0_T^{dyn-4D} 是 **隔离的**，避免 semantic information 泄漏到 depth reasoning

公式 20: Depth loss
$$\mathcal{L}_{\text{dep-4D}} = \sum_{i=1}^{N_i} \mathcal{L}_{\text{dep-4D},i} = \sum_{i=1}^{N_i} \left\|\hat{\mathbf{z}}_i^{\text{dep-4D}} - \mathbf{z}_i^{\text{dep-4D}}\right\|_2^2$$

- 对每个 viewpoint（M, L, R）都解码一个 depth feature
- Ground truth 来自 [Depth Anything V2](https://depth-anything.com/)，但 depth 是 action 后 future state 的 depth
- 用 3 个独立 depth decoder，每个 8 层 Transformer，hidden 1024, 16 heads, FFN ratio 4

**Why this matters**：Single-view depth 有 scale ambiguity（[Monodepth 经典问题](https://arxiv.org/abs/1609.03677)），但 aggregated multi-view geometric relation 已经 implicit 解掉了 scale。所以从 agg-3D 解码 depth 比直接从 single-view 预测更 robust。

### 6.3 SC-Attn: Spatiotemporal Consistency Attention（公式 7）

$$\hat{\mathbf{A}} = \text{SC-Attn}\left(\mathbf{z}_{\{M,L,R\}}^{\text{obj-3D}}, \mathbf{z}_{\mathcal{L}'}^{\text{agg-3D}}, \mathbf{t}, \mathbf{0}_{\{M,L,R\}}^{\text{dyn-4D}}, \mathbf{0}^{\text{dep-4D}}, \mathbf{0}^A\right)$$

Attention pattern（Table 6 ablation 显示 SC-Attn 比 pure causal 和 pure bidirectional 都好）：
- z^{obj-3D} ↔ 0^{dyn-4D}: bidirectional（dynamic reasoning 需要看 object）
- z^{agg-3D} ↔ 0^{dep-4D}: bidirectional
- 0^{dyn-4D} ⊥ 0^{dep-4D}: **隔离**（semantic dynamics 和 geometric depth 不互相污染）
- **0^A** (action tokens): causal，append 在序列末尾

### 6.4 Training Objective（公式 21）
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{dyn-4D}} + \mathcal{L}_{\text{dep-4D}}$$

- **L_action**: L1 loss on action chunk (K=8 single-arm, 25 dual-arm)
- **L_dyn-4D**: dynamic object prediction loss
- **L_dep-4D**: global depth prediction loss

**Inference 时**：dynamic 和 depth decoder 都 **不用**，只让 implicit knowledge 在 SC-Attn 里通过 attention 影响 action tokens。这就是 paper 反复强调的 "training-only" 设计——supervision signal 在 training 时塑造 representation，inference 时零开销。

Table 6 的 ablation 很有意思：
- 去掉 dynamic objects（Dyn.O.）: LIBERO 掉 2.7-4.8%，real-world 掉 5.7-11.6%
- 去掉 global depth（Glob.D.）: 类似幅度
- 把 SC-Attn 换成 causal: LIBERO 掉 7.2%
- 换成 bidirectional: 掉 5.9%

real-world 掉得更多，说明 4D reasoning 在 spatial variation 大的场景下更关键。

---

## 七、实验数据深度解析

### 7.1 LIBERO 四个 suite（Table 1）

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 83.7 | 76.5 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| **ConsisVLA-4D** | **98.8** | **99.8** | 98.0 | **95.6** | **98.1** |

**分析**：
- ConsisVLA-4D 在 Spatial 和 Object 上 98.8% / 99.8%，证明 cross-view 和 cross-object consistency 直接帮助 spatial perception 任务
- Long-horizon 95.6%（比 OpenVLA-OFT 高 1.1%，比 π0 高 10.4%），证明 4D reasoning 对长序列 action 一致性收益大
- 比 SpatialVLA 高 20%，比 CoT-VLA 高 14.2%，证明 implicit 3D/4D 比显式 spatial modeling 更好

### 7.2 ManiSkill2（Table 2）

| Method | PickC | StackC | PushC | Avg |
|--------|-------|--------|-------|-----|
| OpenVLA | 67% | 64% | 71% | 67.3% |
| CogACT | 95% | 90% | - | 92.5% |
| OpenVLA-OFT | 85% | 93% | 88% | 88.7% |
| **ConsisVLA-4D** | 93% | 95% | **95%** | **94.3%** |

PushCube 上提升最明显（95% vs OFT 88%），因为 push 任务对 spatial relation 敏感（推的方向、距离），CO-Fuser 的 geometric relation aggregation 直接派上用场。

### 7.3 Efficiency（Table 3）—— 这是最亮眼的部分

**Simulation (单臂)**:
| Method | Latency | Throughput | FLOPs | Cost (10K steps) |
|--------|---------|-----------|-------|------------------|
| OpenVLA | 0.254s | 3.9 Hz | 8.48T | 11.7h |
| OpenVLA-OFT | 0.137s | 58.4 Hz | 8.45T | 12.3h |
| **ConsisVLA-4D** | **0.110s** | **72.7 Hz** | **4.59T** | **8.6h** |
| ConsisVLA-4D w/o E3D | 0.204s | 39.2 Hz | 16.83T | 22.3h |

**Real-world (双臂)**:
| Method | Latency | Throughput | FLOPs | Cost |
|--------|---------|-----------|-------|------|
| OpenVLA | 0.552s | 1.8 Hz | 16.30T | 12.8h |
| OpenVLA-OFT | 0.334s | 74.8 Hz | 14.95T | 13.7h |
| **ConsisVLA-4D** | **0.231s** | **108.2 Hz** | **9.68T** | **10.1h** |

**关键数字**：
- vs OpenVLA: 2.31× latency speedup, 2.3× throughput gain
- vs OpenVLA-OFT: 1.25× latency speedup, 1.36× training cost reduction
- FLOPs 从 8.48T → 4.59T（**降 46%**）—— 主要来自 visual token 从 256 → 32 的压缩
- 双臂 throughput 108.2 Hz，意味着 RTX 5090 上可以跑实时控制（一般 30Hz 就够）

**Why efficiency gains despite 2B extra params from VGGT**：因为 VGGT 的输出只在 3D perception 阶段用一次，进入 LLM 的 token 数从原 OpenVLA 的 256*3=768 降到 32*3 + 64 + 12 + 4 + 8 ≈ 84，**LLM 主干计算量减少 9×**。E3D（Efficient 3D-Perception）phase 是关键，去掉后 FLOPs 飙到 16.83T，比 OpenVLA 还慢。

参考：[OpenVLA-OFT](https://openvla-oft.github.io/)

### 7.4 Real-world long-horizon（Table 4）

四个 task：Microwave Operation、Banana Peeling、Drawer Arrangement、T-shirt Folding，都是 multi-stage long-horizon。

| Method | Galaxea R1 Lite Avg | AgileX Cobot Magic Avg |
|--------|---------------------|------------------------|
| OpenVLA | 28.5% | 30.0% |
| OpenVLA-OFT | 51.8% | 50.3% |
| **ConsisVLA-4D** | **70.0%** | **68.3%** |

跨平台一致性 ±1.7%，说明 sim-to-real 迁移稳定。比 OpenVLA-OFT 提升 18% absolute，主要来自 long-horizon 任务对 spatiotemporal consistency 的强依赖——T-shirt Folding 这种 fine-grained bimanual 操作，没有 4D reasoning 几乎不可能稳定完成。

参考：[Galaxea R1 Lite](https://galaxea-dynamics.com/), [ALOHA / Cobot Magic](https://tonyzhaozh.github.io/aloha/)

### 7.5 Sparsification ratio（Table 7）

| Spf.Ratio | z^{obj-3D} | z^{agg-3D} | 0^{4D} | LIBERO | Real |
|-----------|------------|------------|--------|--------|------|
| ≈1/4 | 128 | 128 | 30 | 98.0 | 80.0 |
| ≈1/8 | 64 | 64 | 18 | **98.1** | 78.3 |
| ≈1/16 | 32 | 32 | 12 | 94.9 | 68.3 |
| 1/8 (FastV) | - | - | - | 88.8 | 50.0 |
| 1/8 (SliME) | - | - | - | 85.6 | 46.7 |

**关键发现**：
- 1/8 是 sweet spot
- 1/16 太稀疏，丢 spatial info
- **同样 1/8 压缩，FastV 和 SliME 远差于 ConsisVLA-4D**——说明 instruction-aware + cross-view aggregation 是关键，单纯 layer-based pruning 不行

---

## 八、和 SOTA 对比的位置

| 维度 | ConsisVLA-4D 的优势 | 局限 |
|------|---------------------|------|
| **3D perception** | implicit，无 point cloud 输入 | 依赖 VGGT frozen feature quality |
| **4D reasoning** | training-only，inference 零开销 | reasoning supervision 依赖 CoTracker 和 Depth Anything 的伪标签 |
| **Efficiency** | 4.59T FLOPs，72.7Hz | 双臂训练 cost 仍需 10h/10K steps |
| **Generalization** | LIBERO/ManiSkill2/RoboTwin 都好 | Real-world 只测 4 个 task |

---

## 九、对你（Karpathy）的几个 takeaways

1. **Implicit 3D from 2D**：VGGT 作为 frozen 3D feature extractor 是个很 cost-effective 的设计，比端到端学 3D encoder 容易收敛。这有点像你之前在 [TinyLLM](https://www.youtube.com/watch?v=bZ06sn7cZtc) 讲过的——pretrained model 的 latent space 可以 reused。

2. **Token pruning with instruction conditioning**：Top-K selection 是最简单的，但 instruction-conditioned 比固定 layer-based（FastV）效果好 9.3%，说明 instruction signal 在 visual token selection 里非常重要。

3. **Cosine decay for layer-wise fusion**：这个细节很巧妙——浅层和深层保持稳定 prior，中层做过渡，比 linear 好很多。类似的 trick 在 [CogVLA](https://arxiv.org/abs/2508.21046) 系列工作里也出现过。

4. **Training-only auxiliary loss**：让 model 学一个 implicit world model（dynamic + depth），但不增加 inference cost。这和 [JEPA](https://arxiv.org/abs/2301.08243) 哲学类似——predict in latent space，但 ConsisVLA-4D 是 multi-modal supervised，不是 self-supervised。

5. **Block-wise causal attention for aggregation**：让 agg token 单向 attend geo token 这个设计，和 [GPT 的 causal mask](https://arxiv.org/abs/1706.03762) 思想相通，保证 source feature 不被污染。

---

## 参考链接汇总

- [ConsisVLA-4D GitHub](https://github.com/JiuTian-VL/ConsisVLA-4D)
- [OpenVLA](https://openvla.github.io/)
- [OpenVLA-OFT](https://openvla-oft.github.io/)
- [π0 Paper](https://arxiv.org/abs/2410.24164)
- [VGGT](https://vgg-t.github.io/)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [CoTracker 3](https://co-tracker.github.io/)
- [Depth Anything V2](https://depth-anything.com/)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [ALOHA / Mobile ALOHA](https://tonyzhaozh.github.io/aloha/)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [SpatialVLA](https://arxiv.org/abs/2501.15830)
- [Original Transformer paper](https://arxiv.org/abs/1706.03762)
- [Dense Prediction Transformer](https://arxiv.org/abs/2103.13413)
- [FastV](https://arxiv.org/abs/2403.06764)
- [SliME](https://arxiv.org/abs/2406.08487)
- [Galaxea R1 Lite](https://galaxea-dynamics.com/)

如果你想看某个 module 的更细节分析（比如 VGGT 的 point tracking prior 在 cross-view alignment 里到底怎么 work，或者 SC-Attn 的具体 mask 实现），告诉我，我可以再展开。
