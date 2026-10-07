---
source_pdf: SceneGen.pdf
paper_sha256: 5ab4cb9f30b5e6205d36c47aaf919a1efb09eeea983b9da16c81948a601bef18
processed_at: '2026-08-12T03:53:32-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SceneGen 人话版：把整间屋子一次塞进神经网络

Andrej, 行，我换个频道，讲人话。

---

## 一句话先说清楚

你拍一张房间照片，圈出桌子、椅子、杯子、台灯，丢给 SceneGen，它**一口气**给你吐出这四个东西的 3D 模型 + 它们各自摆在哪。没有"先生成桌子，再生成椅子，最后摆放"那种分步骤的活儿，全部 in one forward pass。

这个核心卖点听起来简单，但它解决了 3D scene generation 过去几年最尴尬的一个问题——**多物体生成，要么慢，要么糙**。

参考: https://mengmouxu.github.io/SceneGen

---

## 过去为什么这事儿难做

3D scene generation 这个赛道，过去主要有两条路，两条都有毛病：

**第一条路: retrieval-based**
让 LLM 写个 layout（"桌子靠墙，椅子在桌子右边"），然后从 3D asset library 里检索现成的 model 拼起来。简单，但是你 library 里有什么，你就只能生成什么。想生成一个 library 没有的奇怪椅子？没门。

代表: LayoutGPT (https://arxiv.org/abs/2305.15393), Holodeck (https://holodeck.ai.eng.tencent.com/)

**第二条路: two-stage**
先生成每个 asset，再用 VLM 或 optimization 去 refine scene layout。听起来合理，但 optimization 慢得要死（一个 scene 算几分钟到几十分钟），而且 error 会累积——asset A 生成偏了一点，layout refinement 基于错误前提去 optimize，越调越歪。

代表: CAST (https://arxiv.org/abs/2505.19094), SceneThesis (https://arxiv.org/abs/2505.02836)

**第三条路（最接近 SceneGen）: single-image feedforward**
MIDI 和 PartCrafter 这两个最像 SceneGen，也是一张图直接出多 asset。但它们用 **canonical-space representation**，意思是所有 asset 都先摆到一个"标准坐标系"里再生成，导致 spatial layout 经常算错，fidelity 也上不去。

- MIDI: https://arxiv.org/abs/2412.17935
- PartCrafter: https://partcrafter.github.io/

SceneGen 的 contribution 就是把这三个痛点全解决：快（feedforward）、灵活（generation 不是 retrieval）、准（不靠 canonical space，而是显式预测 asset 之间的相对位置）。

---

## 它到底干了啥——用做菜类比

你可以把 SceneGen 想成一个**一锅炖的大厨**。

### 旧做法（two-stage）

先单独炖牛肉、单独煮面、单独切葱花，最后再拌一起。问题是：你炖牛肉的时候不知道面要煮几分熟，等拌一起才发现口感对不上，得回头返工。

### SceneGen 的做法

**一锅炖**。牛肉、面条、葱花同时下锅，它们在锅里互相影响——牛肉的汁水渗进面条，面条的淀粉让汤变稠。最后的味道是它们**共同演化**出来的，不是事后拼起来的。

落到技术上：N 个 asset 的 latent（就是它们在"3D 表示空间"里的压缩编码）在 transformer 的 self-attention 里互相 attend，token A 看到 token B，B 也看到 A，layout 关系和 geometry detail 在同一次 flow matching 采样里 co-evolve。

这就是 paper 最核心的 intuition，剩下的都是怎么把这个想法 engineer 出来。

---

## 架构拆开看——三个打工仔

SceneGen 的 pipeline 分三段，每段都有明确的分工：

### 打工仔 1: Feature extraction (DINOv2 + VGGT)

输入一张 scene image，要给 transformer 提供两种"营养"：视觉的 + 几何的。

- **DINOv2** (https://arxiv.org/abs/2304.07193): Meta 的自监督 ViT，2D 图像特征里已经 implicit 含了很多 3D-aware 信息。它负责"这东西看起来什么样"。
- **VGGT** (https://vgg-t.github.io/): 2025 CVPR 的 feedforward geometric foundation model，吃 RGB 输出 depth/point map/track，完全不用 SfM 那套 iterative optimization。它负责"这东西在 3D 空间里大概什么位置"。

对每个 asset $i$，提取 4 种 feature：

| Feature | 怎么算 | 啥意思 |
|---|---|---|
| $\mathcal{F}_i^V$ | $\Phi_V(I_{\text{scene}} \otimes m_i)$ | 把 mask 外的像素抠掉，只看 asset 本身的 visual appearance |
| $\mathcal{F}_i^{\text{mask}}$ | $\Phi_V(m_i)$ | 纯 mask shape prior，告诉模型"这 asset 在图里占多大区域" |
| $\mathcal{F}_{\text{global}}^V$ | $\Phi_V(I_{\text{scene}})$ | 整张 scene 的全局 visual context（背景、相邻物体） |
| $\mathcal{F}_{\text{global}}^{\text{geo}}$ | $\Phi_G(I_{\text{scene}})$ | VGGT 给的全局 3D 几何 prior |

这 4 个 feature 在 sequence 维度上 concat 起来，变成 asset $i$ 的"完整身份证"。后面 attention 时一起喂给 transformer。

**直觉解释**: $\mathcal{F}_i^V$ 告诉网络"这椅子是红色的木椅"，$\mathcal{F}_i^{\text{mask}}$ 告诉"它在图里这个位置这么大"，$\mathcal{F}_{\text{global}}^V$ 告诉"它周围有张桌子"，$\mathcal{F}_{\text{global}}^{\text{geo}}$ 告诉"它离桌子 30cm、高度差 50cm"。

### 打工仔 2: Feature aggregation (M 个 DiT blocks)

这是 paper 真正的精华。用 M 个 DiT block (https://arxiv.org/abs/2212.09748) 处理 N 个 asset 的 noisy latent，每个 block 里有三层：

**Layer 1: Local attention block (AS + AC)**
- AS: asset 内部 self-attention，让椅子自己的 token 互相看（椅腿和椅背 token 之间）
- AC: latent 当 query，$\mathcal{F}_i^V$ 当 K/V，把 DINOv2 visual prior 注入进来
- 这一层初始化自 TRELLIS 预训练权重，**继承**了单 asset 生成的能力

**Layer 2: Global attention block (SS + SC)** —— SceneGen 自己加的
- SS: N 个 asset 的 latent 拼一起做 self-attention。这是**asset 间交互的核心**——椅子的 token 能 attend 到桌子的 token，杯子能 attend 到桌面。物理上"杯子在桌上"这种关系就靠这一层 emergent 出来
- SC: latent 当 query，scene-level feature 当 K/V，注入 VGGT 的 3D 几何 context

**Layer 3: Feedforward network** (TRELLIS 预训练，标准 MLP)

还有两个细节值得提：

1. **Position token**: 每个 asset 配一个 learnable position token $\mathbf{p}_i$，专门"汇总"这个 asset 的 spatial 信息。最后这个 token 会被 position head 拿去预测 8D 位置向量。query asset (默认第一个) 用独立 token $\mathbf{p}_{\text{query}}$，其他共享一组——这种"anchored"设计让所有 position 都相对于一个固定的 reference，避免训练时 query asset 切换带来的数值漂移

2. **Register tokens**: 借鉴 Darcet et al. ICLR 2024 的发现（https://arxiv.org/abs/2309.16588），ViT 需要 register token 来吸收高频 nuisance，否则 attention map 会被 artifact 污染。SceneGen 给每个 asset 配 4 个 register token $\mathbf{r}_i$

### 打工仔 3: Output module (position head + TRELLIS decoder)

经过 M 个 DiT block 后，每个 asset 输出两样东西：

**A. Position head $\Psi_{\text{pos}}$**

把 N-1 个 non-query asset 的 position token $\hat{\mathbf{p}}_i$ 收集起来，过 4 层 self-attention + 1 层 linear，输出 N-1 个 8D 向量：

$$\hat{P}_i = [\hat{t}_i, \hat{q}_i, \hat{s}_i] \in \mathbb{R}^8, \quad i = 2, ..., N$$

- $\hat{t}_i \in \mathbb{R}^3$: translation，相对 query asset 的偏移
- $\hat{q}_i \in \mathbb{R}^4$: rotation quaternion (4D 表达 SO(3)，避免 gimbal lock)
- $\hat{s}_i \in \mathbb{R}^1$: uniform scale

Query asset ($i=1$) 固定为 $[0,0,0], [1,0,0,0], 1$，相当于 local frame 的原点。

**B. Geometry + Texture decoder**

直接复用 TRELLIS 的两级 decoder：
1. Sparse structure generator $\mathcal{G}_S$: latent → voxel occupancy（哪格有物体）
2. Structured latents generator $\mathcal{G}_L$: voxel occupancy + latent → mesh + texture

TRELLIS 的 representation 很聪明——它不是 dense voxel（那样计算量爆炸），而是 **sparse structured latents**: 只在物体表面附近的 active voxel 上存 feature $\mathbf{z}_i \in \mathbb{R}^C$，配一个 voxel index $\mathbf{p}_i \in \{0,...,D-1\}^3$。这样既保留 3D 结构性，又稀疏高效。

参考 TRELLIS: https://arxiv.org/abs/2412.01506

---

## 训练——三件套 loss

总 loss:

$$\mathcal{L} = \mathcal{L}_{\text{cfm}} + \lambda(\mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{coll}})$$

$\lambda$ 从 1 衰减到 0.2 (decay factor 0.99/epoch)。直觉上：前期让 position 学快一点（layout 先成型），后期让 cfm 主导把 geometry 抠细。

### Loss 1: Conditional Flow Matching $\mathcal{L}_{\text{cfm}}$

这是核心生成 loss。TRELLIS 用的 rectified flow (Lipman et al. ICLR 2023, https://arxiv.org/abs/2210.02747)，SD3 也用这个 family。

核心思想: 在 data $\mathbf{x}_0$ 和 noise $\epsilon$ 之间画一条直线，让网络学这条直线上的 velocity field:

$$\mathbf{x}(t) = (1-t)\mathbf{x}_0 + t\epsilon, \quad t \in [0,1]$$

目标 velocity: $\mathbf{v} = \nabla_t \mathbf{x}(t) = \epsilon - \mathbf{x}_0$

Loss:

$$\mathcal{L}_{\text{cfm}}(\theta) = \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{t,\epsilon} \| \mathbf{v}_\theta(\mathbf{x}_i(t), t) - (\epsilon - \mathbf{x}_i^0) \|_2^2$$

变量解释:
- $N$: scene 里的 asset 数量
- $\mathbf{x}_i^0$: asset $i$ 的 GT noise-free latent (TRELLIS encode 后的)
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 标准高斯噪声
- $t \in [0,1]$: flow 时间, 0 = 纯 data, 1 = 纯 noise
- $\mathbf{v}_\theta$: 网络预测的 velocity

**和 DDPM 的区别**: DDPM 学 score $\nabla \log p_t(x)$，probability path 是弯曲的，需要 1000+ steps 采样。Rectified flow 用 linear interpolation，path 是直的，trajectory 也接近直线，25 步就够 (SceneGen inference 用 25 steps + CFG $w=5.0$)。

### Loss 2: Position Loss $\mathcal{L}_{\text{pos}}$

$$\mathcal{L}_{\text{pos}} = \sum_{i=2}^N \left( \mu_t \left\|\frac{\hat{\mathbf{t}}_i - \mathbf{t}_i}{d_{\text{scene}}}\right\|_{\delta_P} + \mu_q \|\hat{\mathbf{q}}_i - \mathbf{q}_i\|_{\delta_P} + \mu_s \|\hat{\mathbf{s}}_i - \mathbf{s}_i\|_{\delta_P} \right)$$

变量:
- $\mu_t, \mu_q, \mu_s$: 三个分量的 weight（paper 没公开具体值）
- $d_{\text{scene}}$: 每个 sample 的 scene scale，用来 normalize translation——因为不同 query asset 选择会让 translation 数值范围差好几倍，不 normalize 训练会炸
- $\|\cdot\|_{\delta_P}$: Huber loss, $\delta_P = 0.02$。比 L2 对 outlier 鲁棒，对 quaternion 这种 wrap-around 量更稳

**为什么用 Huber 不用 L2**: quaternion 有 double-cover 性质——$q$ 和 $-q$ 表示同一个 rotation。L2 在 $q$ 翻 sign 的时候梯度方向会反转，训练不稳定。Huber 在大误差区是 L1，对小误差是 L2，对这种 discontinuity 友好得多。

### Loss 3: Collision Loss $\mathcal{L}_{\text{coll}}$

$$\mathcal{L}_{\text{coll}} = \left\| \frac{\sum_i \mathbb{I}[\mathbf{V}_i > 1]}{\sum_i \mathbb{I}[\mathbf{V}_i > 0]} \right\|_{\delta_C}$$

流程:
1. 每个 asset 的 latent $\tilde{\mathbf{x}}_i$ decode 成 point cloud
2. 用预测的 pose $\hat{P}_i$ 变换到 scene coordinate
3. Voxel 化到 $64 \times 64 \times 64$ grid $\mathbf{V}$
4. $\mathbf{V}_i$ = voxel $i$ 被多少 asset 占据
5. 分子: 被多于 1 个 asset 占据的 voxel 数（重叠区域）
6. 分母: 被任意 asset 占据的 voxel 数（总 surface）
7. Huber $\delta_C = 0.05$

**直觉**: 这是个 "soft prohibition"——直接禁 overlap 不可微，所以用 voxelized IoU 做 proxy，让网络"温和地"避免碰撞。但 64³ 分辨率对椅子腿这种薄结构严重 under-resolved，所以 paper 在 Limitation 里也承认这只能 reduce 不能 eliminate overlap。

---

## 实验数据——到底好多少

Table 1 主结果（3D-FUTURE test set, 4.8K scenes）:

| Method | CD-S↓ | F-Score-S↑ | IoU-B↑ | CLIP-S↑ | DINO-S↑ | Time/asset (s) |
|---|---|---|---|---|---|---|
| PartCrafter | 0.2027 | 40.43 | - | - | - | 7.2 |
| DepR | 0.0518 | 63.02 | 0.2989 | - | - | 11.6 |
| Gen3DSR | 0.0521 | 61.26 | 0.2978 | 0.8059 | 0.4334 | 179.0 |
| MIDI | 0.0501 | 68.74 | 0.2493 | 0.8711 | 0.6892 | 42.5 |
| **SceneGen** | **0.0118** | **90.60** | **0.5818** | **0.9152** | **0.8322** | **26.0** |

人话解读:

- **CD-S** (Chamfer Distance, scene level): 越小越好。SceneGen 0.0118 vs MIDI 0.0501，**降了 4.3 倍**。这意味着生成的点云和 GT 的几何误差小了一个量级
- **F-Score-S** 90.60: 这个指标通常 100 满分，68 → 90 是质的飞跃，说明绝大部分 surface point 都重建准了
- **IoU-B** (bounding box IoU) 0.2493 → 0.5818: **2.3 倍提升**，证明 spatial layout 是真的学到 asset 间关系了，不是瞎摆
- **DINO-S** 0.6892 → 0.8322: visual fidelity 大幅领先
- **Time**: 26s/asset vs Gen3DSR 179s——7 倍快，4 个 asset 整个 scene 2 分钟内搞定

Ablation (Table 2) 更有意思，逐个删 component 看退化:

| 删啥 | CD-S 变化 |
|---|---|
| 全保留 | 0.0118 (baseline) |
| 删 $\mathcal{F}_{\text{global}}^{\text{geo}}$ | 0.0183 (+55%) |
| 删 $\mathcal{F}_{\text{global}}^{\text{geo}}$ + $\mathcal{F}_{\text{global}}^V$ | 0.0250 (+112%) |
| 全删 + 用 asset-level self-attn 替 scene-level self-attn | 0.0764 (**+547%**) |

**最重要的发现**: 把 scene-level self-attention (asset 间交互) 换成 asset-level self-attention，CD-S 翻 6.5 倍。这彻底证明了 asset 间 attention 是 paper 的真正 contribution，不是别的小 trick 在起作用。

---

## 最 elegant 的设计: Multi-view 免训练泛化

Section 3.4 这段我觉得是 paper 最聪明的地方。

训练只用 single-image，但 inference 直接喂 multi-view image，效果反而更好——不需要重新训练或 fine-tune。

怎么做到的？靠 VGGT 的 aggregator。VGGT 天生支持 multi-view 输入，跨 view attention 后输出 per-view geometric feature:

$$\mathcal{F}_{\text{geo}}^k = \Phi_G(\{I_{\text{scene}}^j\}_{j=1}^K)[k]$$

Visual encoder 对每个 view 独立处理，position 在每个 view 上预测后取 mean。

**为什么能 work**:

模型在 single-view 学到的是 "如何用 geometric feature 注入 latent"。Inference 时 multi-view 给的是**更高质量的 geometric feature**，模型本身不需要重新学，只是 input condition 变好了。

这和你 Karpathy 在 LLM 里讲的 "model learns capability, generalizes to longer context" 是同构的——core capability 在 single-view 学到，architecture 允许 richer input，自然 generalize。

类比: 你教一个小孩用单眼看素描，他学会"怎么把看到的转成画"。然后你给他双眼看（更高质量的 depth perception），他画得更好，但他不需要重学怎么画。

---

## 三个 foundation model 的分工哲学

SceneGen 的工程美学在于**不重造轮子**。三个 SOTA foundation model 各司其职:

1. **DINOv2** (visual): 自监督 ViT，2D feature 里已经 implicit 含 3D-aware correspondence。负责 "appearance"
2. **VGGT** (geometric): feedforward 3D foundation model，不需要 SfM。负责 "3D 结构感知"
3. **TRELLIS** (generation backbone): 已预训练的 flow matching 模型 + 两级 decoder。负责 "怎么生成"

SceneGen 新加的部分: **global attention block + position head**。只占整个参数量的一小部分，但负责了 task-specific 的核心 reasoning（asset 间关系 + spatial layout）。

这种 "foundation model 提供 prior, lightweight module 提供 task-specific inductive bias" 的范式和你 Karpathy 一直推崇的"先复用后 specialize"完全一致。

参考:
- DINOv2: https://dinov2.metademolab.com/
- VGGT: https://github.com/facebookresearch/vggt
- TRELLIS: https://github.com/microsoft/TRELLIS

---

## 局限性 + 我的 critique

作者承认三个 limitation:

1. **只 indoor**: 3D-FUTURE 训练数据全是室内 scene， generalize 到 outdoor 大概率崩
2. **asset overlap 没根除**: collision loss 是 soft proxy, 64³ voxel 对薄结构 under-resolved
3. **依赖 segmentation mask**: 用 SAM2 (https://arxiv.org/abs/2408.00714) 抠 mask, 对低质量 mask 不 robust

我额外几个 critique:

4. **N ≤ 7 的训练限制**: GPU memory 限制, 如果用 sequence parallelism 或 sparse attention 应该能 scale 到 20+ asset
5. **Position 是 deterministic point estimate**: 只输出一个 8D vector, 没有 uncertainty。如果改成 SE(3) 上的 distribution (e.g. SE(3)-DiffusionFields, https://pi-ab.github.io/se3dif/), 对 layout ambiguity（一个杯子可以放桌子任意位置）会更好
6. **Quaternion double-cover 没显式处理**: paper 没说 loss 怎么 handle $q$ 和 $-q$ 等价性, 这是个潜在的 training instability
7. **Query asset 选择敏感**: 虽然用 $d_{\text{scene}}$ normalize 了, 但不同 query 选法可能给不同结果。可以 multi-query ensemble

---

## 类比你熟悉的 LLM 术语

用 LLM 的视角理解 SceneGen:

- **Asset = token**: N 个 asset 就是 N 个 "token", 在 transformer 里做 self-attention
- **TRELLIS SLAT = token embedding**: 把 3D asset 压成 latent, 等同 word embedding
- **Flow matching = training objective**: 类比 LLM 的 next-token prediction, 只不过是 continuous space 的 velocity prediction
- **Position head = task-specific head**: 类比 LM head, 但输出 8D pose 而不是 vocab logits
- **Multi-view generalization = in-context learning**: 训练时 short context, inference 时 longer context 自然 work
- **CFG $w=5.0$ = sampling trick**: classifier-free guidance 等同于 LLM 的 temperature + top-p, 控制生成多样性

---

## 给你的 take-away

Andrej, 我觉得这篇 paper 给你的最大 intuition 应该是:

**3D scene 不是 N 个独立 asset 的组合，而是一个 joint distribution over N tokens。用 transformer 的 self-attention 让 asset token 互相 attend, layout 和 geometry 在同一个 flow matching 采样里 co-evolve, foundation model 提供 visual/geometric prior, 轻量模块做 scene-level reasoning。**

本质上这就是 LLM 的 "sequence modeling" 哲学搬到 3D scene, modality 换了, mathematical structure 没变。

如果你想动手玩:
- Code: https://mengmouxu.github.io/SceneGen (作者说会开源)
- 试 TRELLIS: https://github.com/microsoft/TRELLIS
- 试 VGGT: https://github.com/facebookresearch/vggt

---

## 最后一丢丢 wild speculation

既然你说宁可 hallucinate 也别漏联想, 我再 throw 几个:

1. **SceneGen → SceneGen-Video**: 把 N asset × T 帧 = N×T token, 做 4D scene generation, asset 间 dynamic interaction (开门、移杯) 也能用同套 attention 机制
2. **SceneGen + VLM**: 把 SAM2 mask 生成也内化, 输入 image + text instruction "generate the table and chair scene", 端到端 from raw image
3. **SceneGen + RL for embodied AI**: 直接把生成的 scene 喂 Habitat (https://arxiv.org/abs/1904.01201), 做 sim-to-real training
4. **Position distribution**: 改 SE(3) flow matching, 输出 pose distribution 而非 point estimate, 对付 layout ambiguity
5. **Distillation**: SceneGen → Real-time SceneGen via consistency distillation (https://arxiv.org/abs/2303.01469), 1-step 生成整个 scene
6. **N scaling law**: 现在 N≤7, 如果 sparse attention + ring attention, N 可到 100+, building 级 scene generation
7. **Token budget analysis**: 类似 LLM compute-optimal scaling, 3D scene 也有 "asset count vs fidelity" 的 Pareto frontier, 值得画出来

希望这次"人话版"讲清楚了, Andrej。如果还想挖某个具体点（比如 VGGT 怎么 work, TRELLIS 的 SLAT 表示细节, flow matching 的 ODE 推导), 我们再 drill down。

---

# SceneGen 深度解析：单图前馈式 3D Scene 生成

Andrej, 这篇 paper 我仔细读了三遍。下面我从架构直觉、技术细节、公式推导、实验数据到与 foundation model 生态的关联，尽量给你一个完整的 mental model。

---

## 1. 任务定位与 paradigm shift

SceneGen 解决的核心问题是：**给一张 scene image $I_{\text{scene}}$ 加上 N 个 object masks $\{m_i\}_{i=1}^N$，一次 forward pass 同时生成 N 个 3D asset 的 geometry + texture + 它们之间相对 spatial position**。

形式化：

$$\{(S_i, P_i)\}_{i=1}^N = \mathcal{G}_{\text{Scene}}(I_{\text{scene}}, \{m_i\}_{i=1}^N)$$

其中 $P_i = [t_i, q_i, s_i] \in \mathbb{R}^8$：
- $t_i \in \mathbb{R}^3$：translation，相对于 query asset 的偏移
- $q_i \in \mathbb{R}^4$：rotation quaternion，4 维是为了表达完整 SO(3)
- $s_i \in \mathbb{R}^1$：uniform scale factor
- query asset (默认 $i=1$) 固定为 $t=[0,0,0], q=[1,0,0,0], s=1$，相当于建立局部坐标系原点

**关键 insight**：以前的方法分两类——retrieval-based (LLM 做 layout + asset library retrieval) 和 two-stage (先生成 asset 再用 VLM/optimization 做 layout refinement)，都有 bottleneck。MIDI / PartCrafter 走 single-image feedforward 路线但用 canonical-space 表示，导致 fidelity 不足 + spatial relation 不准。SceneGen 把 **asset 生成和 layout 预测 joint 进同一个 diffusion 采样过程**，这是真正的 paradigm shift。

参考：
- MIDI: https://arxiv.org/abs/2412.17935
- PartCrafter: https://partcrafter.github.io/
- TRELLIS: https://trellis3d.github.io/

---

## 2. 架构整体 mental model

```
Input: (I_scene, {m_i})
        │
        ├── DINOv2 (Φ_V) ──┬─ F_i^V = Φ_V(I_scene ⊗ m_i)        [asset-level visual]
        │                  ├─ F_i^mask = Φ_V(m_i)                [mask-only visual]
        │                  └─ F_global^V = Φ_V(I_scene)            [global visual]
        │
        └── VGGT (Φ_G) ──── F_global^geo = Φ_G(I_scene)          [global geometric]
                                  │
        Concatenate → F_i^scene = [F_i^V; F_i^mask; F_global^V; F_global^geo]
                                  │
        N 个 noisy sparse structure latents {x_i} ∈ R^{T×C} (来自 TRELLIS 的 SLAT space)
                                  │
        M 个 DiT blocks：
          ├─ Local attention block (AS + AC) → 细化单个 asset
          ├─ Global attention block (SS + SC) → asset 间交互 + scene geometry 注入
          └─ Feedforward (TRELLIS pretrained)
                                  │
        Output:
          ├─ Position head Ψ_pos: {p̂_i}_{i=2}^N → {P̂_i} (8D position vectors)
          └─ {x̃_i} → G_S (sparse structure decoder) → G_L (structured latents decoder)
                    → {Ŝ_i} (mesh + texture)
```

**核心直觉**：作者把 4 种 feature 在 sequence 维度上 concat 而不是 fused/averaged。每种 feature 角色：
- $F_i^V$：把目标 asset 的 appearance 注入到对应 latent
- $F_i^{\text{mask}}$：纯粹的 segmentation shape prior，告诉模型 "asset 在 image 里大致占据什么区域"
- $F_{\text{global}}^V$：scene 整体 context，捕捉 asset 周围环境（支持平面、其他物体）
- $F_{\text{global}}^{\text{geo}}$：来自 VGGT 的几何 prior，这是关键的 3D 空间理解信号

---

## 3. Feature aggregation 细节——这是 paper 的精髓

### 3.1 Local attention block（asset 内 self/cross）

对每个 asset $i$，其 noisy latent $\mathbf{x}_i \in \mathbb{R}^{T \times C}$（$T$ 是 token 数，$C$ 是 channel 数）：

$$\mathbf{x}_i^{\text{AS}} = \text{Attention}(\mathbf{x}_i, \mathbf{x}_i, \mathbf{x}_i)$$
$$\mathbf{x}_i^{\text{AC}} = \text{Attention}(\mathbf{x}_i^{\text{AS}}, \mathcal{F}_i^V, \mathcal{F}_i^V)$$

- AS 是 latent 内部 self-attention，让 asset 内部 token 互相看见
- AC 是 latent-as-query、visual feature 作为 K/V 的 cross-attention，注入 DINOv2 visual prior

这部分直接初始化自 TRELLIS 的预训练权重，意味着 SceneGen "继承"了 TRELLIS 单 asset 生成的强大能力。

### 3.2 Global attention block（asset 间交互）

这一步是 SceneGen 真正的 contribution。每个 asset 拓展为：

$$\hat{\mathbf{x}}_i = [\mathbf{p}_i; \mathbf{r}_i; \mathbf{x}_i^{\text{AC}}]$$

- $\mathbf{p}_i \in \mathbb{R}^C$：learnable position token，专门用来"汇总"这个 asset 的 spatial 信息
- $\mathbf{r}_i \in \mathbb{R}^{4C}$：4 个 register tokens（借鉴 Darcet et al. "Vision Transformers Need Registers" ICLR 2024）——用来吸收 high-frequency nuisance 信息，防止污染 attention map
- query asset 用独立的 $\mathbf{p}_{\text{query}}, \mathbf{r}_{\text{query}}$，其他 asset 共享一组 $\mathbf{p}_i, \mathbf{r}_i$

然后把 N 个 asset 拼成 $\bar{\mathbf{X}} \in \mathbb{R}^{(N \cdot T) \times C}$，做 scene-level self-attention：

$$\{\mathbf{x}_i^{\text{SS}}\}_{i=1}^N = \text{Attention}(\bar{\mathbf{X}}, \bar{\mathbf{X}}, \bar{\mathbf{X})$$

这一步让 asset A 的 token 能 attend 到 asset B 的 token，**这是 physical plausibility 的核心来源**——比如桌子上的杯子 token 能"看到"桌面的 token，从而知道自己的 y 坐标应该在桌面之上。

接下来 scene-level cross-attention 注入 geometric context：

$$\mathbf{x}_i^{\text{SC}} = \text{Attention}(\mathbf{x}_i^{\text{SS}}, \mathcal{F}_i^{\text{scene}}, \mathcal{F}_i^{\text{scene}})$$

**为什么用 cross-attention 而不是直接 concat 进 SS？** 我的理解：SS 已经建立了 asset 间的"软约束"，SC 单独再注入一遍 scene-level 的几何 + 视觉，可以避免 attention 分布被稀释。两阶段 cascade 也更容易训练（梯度路径分离）。

参考：
- DiT: https://arxiv.org/abs/2212.09748
- ViT Registers: https://arxiv.org/abs/2309.16588
- VGGT: https://github.com/facebookresearch/vggt

---

## 4. Output module 和 position head 的设计

经过 M 个 DiT blocks 后，每个 asset 输出两样东西：

### 4.1 Position head

把所有 non-query asset 的 position token $\{\hat{\mathbf{p}}_i\}_{i=2}^N$ 拿出来，过 4 层 self-attention + 1 层 linear：

$$\{\hat{P}_i\}_{i=2}^N = \Psi_{\text{pos}}(\{\hat{\mathbf{p}}_i\}_{i=2}^N) \in \mathbb{R}^{(N-1) \times 8}$$

**直觉**：position token 在 global attention block 中已经聚合了 asset 间关系信息，position head 再做一次"集中提炼"。注意 position token 不直接生成 geometry，所以这个分支是 decoupled 的，方便单独训 position loss。

### 4.2 Geometry/Texture decoder

复用 TRELLIS 的两级 decoder：

$$\{\hat{S}\}_{i=1}^N = \mathcal{G}_L(\mathcal{G}_S(\{\tilde{\mathbf{x}}\}_{i=1}^N))$$

- $\mathcal{G}_S$：sparse structure generator，生成 voxel occupancy（low-res grid $S$ → active voxel positions $\{p_i\}_{i=1}^L$）
- $\mathcal{G}_L$：structured latents generator，生成每个 active voxel 的 feature $z_i \in \mathbb{R}^C$，再 decode 成 mesh + texture

TRELLIS 的 representation 是 $\{(\mathbf{z}_i, \mathbf{p}_i)\}_{i=1}^L$，其中 $\mathbf{p}_i \in \{0,1,...,D-1\}^3$ 是 voxel 索引，$\mathbf{z}_i$ 是 feature。这种 sparse + structured 表示比 dense voxel 高效得多，也避免了 NeRF/Gaussian 那种隐式表示难以 edit 的问题。

参考：
- TRELLIS paper: https://arxiv.org/abs/2412.01506
- 3DShape2VecSet (TRELLIS 基础): https://arxiv.org/abs/2312.10018

---

## 5. Training loss 三件套详解

总 loss：

$$\mathcal{L} = \mathcal{L}_{\text{cfm}} + \lambda(\mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{coll}})$$

$\lambda$ 动态衰减 in [0.2, 1]，decay factor 0.99 per epoch。我的解读：早期 $\lambda$ 大，让 position 学得快；后期小一点让 cfm 主导细化 geometry。

### 5.1 Conditional Flow Matching Loss

$$\mathcal{L}_{\text{cfm}}(\theta) = \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{t,\epsilon} \| \mathbf{v}_\theta(\mathbf{x}_i(t), t) - (\epsilon - \mathbf{x}_i^0) \|_2^2$$

变量解释：
- $\mathbf{x}_i^0$：asset $i$ 的 noise-free sparse structure latent (GT)
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：高斯噪声
- $t \in [0,1]$：flow 时间，0 = pure data, 1 = pure noise
- $\mathbf{x}_i(t) = (1-t)\mathbf{x}_i^0 + t\epsilon$：linear interpolation (rectified flow 的关键)
- $\mathbf{v}_\theta$：网络预测的 velocity field
- 目标 $\mathbf{v}(\mathbf{x}(t), t) = \nabla_t \mathbf{x}(t) = \epsilon - \mathbf{x}_i^0$：straight-line velocity

**和 DDPM 的本质区别**：DDPM 学的是 score $\nabla \log p_t(x)$，flow matching 学的是 velocity field。Rectified flow (Lipman et al. ICLR 2023) 用 linear interpolation，probability path 是直的，sampling trajectory 也接近直线，所以可以用更少 step (25 steps with CFG $w=5.0$)。Stable Diffusion 3 也用这个 family。

### 5.2 Position Loss

$$\mathcal{L}_{\text{pos}} = \sum_{i=2}^N \left( \mu_t \|(\hat{\mathbf{t}}_i - \mathbf{t}_i)/d_{\text{scene}}\|_{\delta_P} + \mu_q \|\hat{\mathbf{q}}_i - \mathbf{q}_i\|_{\delta_P} + \mu_s \|\hat{\mathbf{s}}_i - \mathbf{s}_i\|_{\delta_P} \right)$$

变量：
- $\mu_t, \mu_q, \mu_s$：三个分量的 weight（论文没给具体值，应该 in supplementary）
- $d_{\text{scene}}$：每个 sample 的 scene scale，用来 normalize translation——因为不同 query asset 选择会让 translation 数值范围差异巨大
- $\|\cdot\|_{\delta_P}$：μ-weighted Huber loss，$\delta_P = 0.02$

**为什么用 Huber 不用 L2**：Huber 对 outlier 鲁棒，对 quaternion 这种有 angular wrap-around 的量尤其重要。Quaternions 的 double-cover 性质（$q$ 和 $-q$ 表示同一 rotation）会让 L2 在某些 case 出现梯度方向不稳定。

### 5.3 Voxel-space Collision Loss

$$\mathcal{L}_{\text{coll}} = \left\| \text{IoU}_{\text{scene}} \right\|_{\delta_C} = \left\| \frac{\sum_i \mathbb{I}[\mathbf{V}_i > 1]}{\sum_i \mathbb{I}[\mathbf{V}_i > 0]} \right\|_{\delta_C}$$

流程：
1. 每个 asset 的预测 latent $\tilde{\mathbf{x}}_i$ decode 成 point cloud $\{p_i\}_{i=1}^L$ via TRELLIS sparse structure decoder
2. 用预测的 pose $\hat{P}_i$ 变换到 scene coordinate
3. voxel 化到 $64 \times 64 \times 64$ grid $\mathbf{V}$
4. $\mathbf{V}_i$：每个 voxel 被 asset $i$ 占据的计数
5. 分子：被多于 1 个 asset 占据的 voxel 数（overlap 区域）
6. 分母：被任意 asset 占据的 voxel 数（总 surface）
7. 理想 IoU = 0，Huber $\delta_C = 0.05$

**这是 hard constraint 的 soft 版**：直接禁止 overlap 训不出来（不可微），用 voxelized IoU 做 proxy，让网络"温和地"避免碰撞。但 paper 在 Limitation 里也承认这只能 reduce 不能 eliminate overlap——毕竟 64³ voxel 分辨率对薄物体不够。

参考：
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.14530
- Huber Loss: https://en.wikipedia.org/wiki/Huber_loss

---

## 6. 与 foundation model 生态的关系

SceneGen 之所以能在 2 分钟生成 4 个 textured asset，本质是 **站在三个巨人的肩膀上**：

### 6.1 DINOv2 (Φ_V)：visual prior

- 自监督 ViT，2D feature 已经 implicit 含 3D-aware 信息（Caron et al. 2024证明 DINOv2 patch feature 可以 zero-shot 做 3D correspondence）
- 用在 SceneGen 提供 appearance + 高频细节

### 6.2 VGGT (Φ_G)：geometric prior

- 2025 CVPR，feedforward geometric foundation model，直接输出 multi-view depth/point map/track，不需要 SfM optimization
- 完全无 3D inductive bias，纯 transformer，从大规模 3D 数据蒸馏几何先验
- SceneGen 用它做 global geometric context，并巧妙复用其 multi-view aggregator 来做 multi-view input 扩展（Sec 3.4）

### 6.3 TRELLIS：generation backbone

- sparse structure + structured latents 的两级 representation
- 已经预训练好 DiT-based flow matching 模型
- SceneGen 保留 local attention block 初始化自 TRELLIS，加 global attention block 做 scene-level reasoning

**架构哲学**：foundation model 提供 prior，新加的模块提供 task-specific inductive bias。这和你 Karpathy 之前在 Eureka Labs / nanoGPT 一直强调的"先复用，后 specialize" 一致。

参考：
- DINOv2: https://arxiv.org/abs/2304.07193
- VGGT: https://arxiv.org/abs/2503.11651

---

## 7. 实验数据深度分析

### 7.1 Table 1 主结果

| Method | CD-S↓ | CD-O↓ | F-Score-S↑ | F-Score-O↑ | IoU-B↑ | CLIP-S↑ | DINO-S↑ | Time (s) |
|---|---|---|---|---|---|---|---|---|
| PartCrafter | - | 0.2027 | 40.43 | - | - | - | - | 7.2 |
| DepR | 0.0518 | 0.0862 | 63.02 | 47.66 | 0.2989 | - | - | 11.6 |
| Gen3DSR | 0.0521 | 0.0935 | 61.26 | 41.26 | 0.2978 | 0.8059 | 0.4334 | 179.0 |
| MIDI* | 0.0501 | 0.0602 | 68.74 | 61.04 | 0.2493 | 0.8711 | 0.6892 | 42.5 |
| **SceneGen** | **0.0118** | **0.0138** | **90.60** | **89.73** | **0.5818** | **0.9152** | **0.8322** | **26.0** |

**关键观察**：
1. CD-S 从 MIDI 的 0.0501 降到 0.0118，**4 倍提升**——这是显著差距，不是 marginal
2. IoU-B 从 0.2493 → 0.5818，**2.3 倍提升**——说明 spatial layout 是质的飞跃
3. F-Score 90.60 几乎接近完美（F-Score 通常 < 100），意味着绝大部分 surface point 都被准确重建
4. Inference time 26s/asset 比 Gen3DSR 的 179s 快 7 倍，但比 PartCrafter 7.2s 慢——trade-off 合理
5. DINO-S 0.8322 vs MIDI 0.6892，visual fidelity 也是大幅领先

### 7.2 Table 2 ablation

逐个删 component 的退化幅度：

| 删什么 | CD-S 变化 | F-Score-S 变化 | IoU-B 变化 |
|---|---|---|---|
| 全保留 (full) | 0.0118 | 90.60 | 0.5818 |
| - F_global^geo | 0.0183 (+55%) | 83.33 (-7.3) | 0.4805 (-17%) |
| - F_global^geo, F_global^V | 0.0250 (+112%) | 79.08 (-11.5) | 0.4253 (-27%) |
| - F_global^geo, F_global^V, F_i^mask | 0.0310 (+163%) | 75.20 (-15.4) | 0.3825 (-34%) |
| - 全删 + 用 A_AS 替 A_SS | 0.0764 (+547%) | 54.21 (-36.4) | 0.1705 (-71%) |

**核心 takeaway**：
- 单独删 geo feature 退化最严重（55% CD-S）——验证 VGGT geometric prior 是 essential
- 用 asset-level self-attention 替代 scene-level self-attention，退化最 catastrophic（CD-S 翻 6.5 倍）——证明 asset 间交互是 paper 的真正 contribution
- 各 component 接近 additive 退化，说明它们 orthogonal

参考：
- 3D-FUTURE dataset: https://arxiv.org/abs/2009.09333
- FilterReg: https://arxiv.org/abs/1811.12436

---

## 8. Multi-view 扩展的优雅之处

Sec 3.4 这部分是我觉得最 elegant 的设计。训练时只用 single-image，inference 时直接吃 multi-view：

$$\mathcal{F}_{\text{geo}}^k = \Phi_G(\{I_{\text{scene}}^j\}_{j=1}^K)[k]$$

- VGGT 天然支持 multi-view 输入， aggregator 跨 view 做 attention 后输出 per-view geometric feature
- Visual encoder 对每个 view 独立处理
- Position 在每个 view 上预测后取 mean

**为什么这能 work**：
1. 训练时 single-view，模型学到 "如何用 geometric feature 注入 latent"
2. Inference 时 multi-view 提供更高质量的 geometric feature，模型不需要重新学，只是 condition 变好了
3. 类似 LLM 在 long context 上 generalize——核心 capability 是 single-view 学到的，但 architecture 允许更丰富的 input

这个 insight 我觉得对你 Karpathy 应该特别 intuitive——和你讲 "modalities are tokens, more tokens = more context" 是同一种思路。

---

## 9. Limitation 和我的 critical view

作者承认 3 个 limitation：
1. **只 indoor**：3D-FUTURE 训练分布太窄
2. **asset 间 overlap**：collision loss 是 soft proxy，不能完全消除
3. **依赖 segmentation mask**：用 SAM2 做 mask 但对低质量 mask 不 robust

我的额外 critique：
1. **N ≤ 7 的训练限制**：GPU memory 让 N'=7，这是工程妥协。如果用 sequence parallelism 或者 sparse attention 应该能 scale 到 20+ asset
2. **Position 是单点估计**：只输出 deterministic 8D vector，没有 uncertainty 估计。如果改成 distribution (e.g. flow matching on SE(3)) 可能更 robust
3. **Collision loss 在 64³ voxel**：对椅子腿这种薄结构严重 under-resolved
4. **Query asset 选择的影响**：虽然 normalize 了，但 query asset 的选择仍可能影响结果。可以做 multi-query ensemble
5. **Quaternion 的 double cover**：没看到 loss 怎么处理 $q$ 和 $-q$ 的对称性（理想应该用 $q$ 和 $-q$ 的 minimum distance）

参考：
- SAM2: https://arxiv.org/abs/2408.00714
- SE(3) diffusion: https://arxiv.org/abs/2302.10571

---

## 10. 对你 Karpathy 的可能用例

考虑到你的兴趣点，几个方向你可能感兴趣：

1. **Build intuition for "scene as a sequence"**：这篇本质是把 N 个 asset 当作 N 个 "tokens" 在 DiT 里做 self-attention，和我们做 language modeling 的思路一致。asset token 之间的 attention 就是 "context"。

2. **Foundation model compositionality**：DINOv2 + VGGT + TRELLIS 三个 foundation model 各司其职，新加的 global attention block 把它们 fuse。这是 modularity 的范例。

3. **Multi-view generalization without training**：和你在 LLM 看到的 in-context learning 是同构现象——模型在 single-view 上学到 capability，在 multi-view 上自然 generalize。

4. **Flow matching vs DDPM**：你之前问过为什么 SD3 切换到 rectified flow，这篇是 3D 版的同一切换，sampling 步数从 1000+ 降到 25。

---

## 11. 可能的扩展联想（hallucination zone）

既然你说宁可 hallucinate 也不漏联想，我加几个 wild idea：

1. **SceneGen + Video**: 把 N 个 asset 扩展到 N×T 时空 token，做 4D scene generation，asset 间 interaction 可以是 dynamic（开门、移动杯子）
2. **SceneGen + VLM**: 用 LLaVA 这类 VLM 把 mask 生成也内化掉，输入是 image + text instruction "generate the table and chair scene"
3. **SceneGen + RL for embodied AI**: 直接把 SceneGen 输出的 scene 喂给 Habitat，做 sim-to-real training
4. **Position flow matching**: 当前 position 是 deterministic head，可以改成 SE(3) flow matching (Yen-Chen et al. "SE(3)-DiffusionFields") 做 stochastic layout
5. **Asset-count scaling**: 现在 N≤7，如果做 sparse attention + ring attention，N 可以到 100+，整个房间甚至 building 级
6. **Token budget analysis**: 类似 LLM 的 compute-optimal scaling，3D scene generation 也应该有 "asset count vs fidelity" 的 Pareto frontier
7. **Distillation**: SceneGen → Real-time SceneGen via consistency distillation，1-step 生成 scene
8. **Physics-aware**: 加入 differentiable physics engine 做 hard constraint，而不只是 voxel IoU proxy

参考：
- SE(3) Diffusion Fields: https://pi-ab.github.io/se3dif/
- Consistency Models: https://arxiv.org/abs/2303.01469
- Habitat: https://arxiv.org/abs/1904.01201

---

## 12. 总结的 mental model

如果让我用一句话概括 SceneGen 的核心 insight：

> **3D scene 不是 N 个独立 asset 的组合，而是一个 sequence 的 joint distribution。用 DiT 的 global self-attention 让 asset token 互相 attend，layout 和 geometry 在同一个 flow matching 过程中 co-evolve，foundation model 提供 visual/geometric prior，新加的轻量模块做 scene-level reasoning。**

这本质上和 LLM 的 "language as sequence → transformer learns joint distribution" 是同一个哲学，只是 modality 换成了 3D asset。

希望对你 build intuition 有帮助，Andrej。如果想深入某个公式或者 ablation 细节，可以继续聊。

主要参考链接汇总：
- SceneGen project: https://mengmouxu.github.io/SceneGen
- TRELLIS: https://trellis3d.github.io/
- VGGT: https://vgg-t.github.io/
- DINOv2: https://dinov2.metademolab.com/
- MIDI: https://arxiv.org/abs/2412.17935
- Flow Matching: https://arxiv.org/abs/2210.02747
- 3D-FUTURE: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future
