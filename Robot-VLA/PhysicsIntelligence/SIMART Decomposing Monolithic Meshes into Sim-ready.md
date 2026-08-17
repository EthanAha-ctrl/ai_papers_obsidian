---
source_pdf: SIMART Decomposing Monolithic Meshes into Sim-ready.pdf
paper_sha256: 7c21816f58fba0b4cfe85b72331cda7846f8a3c7588cee811dac456c50e2ebd6
processed_at: '2026-08-12T06:24:57-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SIMART 用人话讲

## 一句话说清楚

现在 AI 能生成好看的 3D 模型（比如一把椅子），但这把椅子是"一坨"——没法开合抽屉、没法转椅背。SIMART 干的事就是：**给这坨 mesh "装上关节"，让它能动起来，能直接丢进机器人模拟器里玩**。

---

## 为什么这事需要一篇 paper

想象你是个做机器人训练的人，你需要大量 3D 物体来训练机器人开抽屉、开门、转方向盘。你去网上下 3D 模型，发现：

- 椅子就是一块 static mesh，没 metadata 说"椅背能绕哪个轴转"
- 想要 articulated model？得人工用 Blender / Maya 做，一个 model 几小时
- PartNet-Mobility 这种 dataset 才 2000 个 articulated object，远远不够训机器人

**gap**：3D generation 爆发了（Hunyuan3D、TRELLIS 都能生成漂亮 mesh），但生成出来的都是"死的"。机器人需要"活的"——需要知道哪里能转、哪里能拉、转多少度。

---

## 之前别人怎么解决，为什么不行

### 路线 A：多阶段 pipeline

Urdformer、Articulate-Anything 这类做法是：先 segment（把 mesh 切成 part），再估 joint（每个 part 的旋转轴），最后拼成 URDF。

问题：三步各自有误差，累乘起来很糟。特别是 segmentation 这步——用 SAM 之类的 2D 模型投影到 3D，边界糊得很；用 PartField、P3SAM 这类 3D segmentation，它们优化的是"表面一致"，不是"机械结构合理"。结果你把一个柜子切成 6 块，但切缝不在抽屉和柜体的交界处，joint 怎么估都错。

**人话类比**：像你切蛋糕，刀法很准但切的位置不在"该切的地方"，比如把蜡烛都切两半。

### 路线 B：从视频/多状态重建

ArtGS、ArticulatedGS 用 NeRF/3DGS 从同一物体的开/合多状态图像里推 articulation。

问题：你得先拍到同一个柜子开门和关门的照片，in-the-wild 哪来这种数据。

**人话类比**：你得先看一个人从婴儿长到老才能建模他的人生轨迹，但很多时候你只有一张照片。

### 路线 C：生成式 prior

CAGE、SINGAPO 用 diffusion 学一类物体的 articulation pattern。

问题：articulated 3D 数据集太小，严重 overfitting。学完只能生成"标准柜子"，遇到"带弧形玻璃门的柜子"直接崩。

### 路线 D：MLLM + dense voxel（最接近 SIMART 的思路）

PhysX-Anything、ShapeLLM-Omni 把 mesh voxel 化成 $64^3$ 的 grid，flatten 成 token 序列喂给 MLLM，让 MLLM 直接输出 URDF。

问题：$64^3 = 262144$ 个 voxel，绝大多数是空气（chair 的 voxel occupancy 可能只有 5-10%）。MLLM 的 context window 装不下。一个物体 4 个 part 就要 4×262144 token，直接 OOM。

**人话类比**：你描述一把椅子，结果 95% 的篇幅在说"这里是空气，那里是空气，那里还是空气"。

---

## SIMART 的核心 trick：把空气 skip 掉

SIMART 的核心 insight：**3D 物体大部分体积是空的，干嘛要花 token 去表示空气？**

具体做法：

1. 把 mesh 转成 $64^3$ voxel grid
2. 用 3D-Unet encoder 压成 $8^3$ 的 latent grid（每个 token 是 64 维 feature）
3. **关键**：latent grid 里没被占据的位置，直接 assign 一个专门的"zero token"（codebook 第 0 号），不去做 nearest neighbor search
4. 只对 occupied 的位置做 VQ quantization
5. 每个 occupied voxel serialize 成三个 token：`<voxel> [xyz坐标] [codebook索引]`

这样 token 数从 ~262144 降到 ~516，降了 87%。

**人话类比**：描述一把椅子时，你只说"腿在这里、座面在这里、靠背在这里"，而不是把整个房间每个立方厘米都描述一遍。

---

## 但这里有个 subtle 的问题

你可能会想：那直接把 empty voxel 丢掉不就行了，干嘛还要 zero token？

**因为 decoder 需要知道"哪里是空的"才能重建**。如果你只是 skip 掉 empty voxel，decoder 收到一堆 occupied voxel token，它不知道这些 voxel 之间和周围的空间结构。实验证明（Table 4）：

- Force Sparse（直接 skip，无 zero token）：Chamfer Distance = 56.10
- Zero Sparse（用 zero token）：Chamfer Distance = 4.19

差 13 倍。zero token 是个"占位符"，告诉 decoder "这些位置我已知是空的，你别乱填"。

**人话类比**：你跟别人描述一张地图，与其只说"这里有山、那里有河"，不如说"中间一大片是平原"——显式标记"空"本身是信息。

---

## 还有个有意思的发现

作者发现，即使你不显式 reserve zero token，训练 dense VQ-VAE 时 codebook 会自然涌现 2-4 个 entry 专门 represent empty space（比如 entry 1849）。这说明 empty space 的 distribution 很 distinct，VQ-VAE 会自发学到。

这跟 LLM 里 [PAD] token 的 emergent behavior 有点像。也跟 VQGAN 里 dead code 问题相关——通常被视为 bug，但这里被 reframe 成 feature。

参考 VQ-VAE original: https://arxiv.org/abs/1711.00937

---

## MLLM 部分

用 Qwen3-VL-8B 当 backbone。输入是三路 modality 拼接：

- Vision：物体的 45° isometric 渲染图（252×252），经过 ViT 提取 visual feature
- Geometry：sparse voxel tokens（就是上面说的 `⟨voxel⟩ xyz K` 序列）
- Text：任务指令

输出也是 hybrid：每个 part 的 voxel token + 一段 JSON 描述 kinematic structure。

JSON 长这样（Table 6）：

```json
{
  "parts_captions": {
    "1": {
      "type": "revolute",
      "parent": "0",
      "center": [100, 138, 101],
      "axis": [100, 0, 0],
      "limits": [-54, 45]
    }
  },
  "parts_voxels": {
    "1": "<voxel> 43 1930 <voxel> 44 13 ..."
  }
}
```

**人话**：模型同时输出"这个 part 长什么样"（voxel token）和"这个 part 怎么动"（JSON），在同一个 autoregressive sequence 里。这是 single-stage 的关键——不分开做，避免 error accumulation。

---

## 为什么 vision 这么重要

Ablation 显示加 vision 后 Type Accuracy 从 0.794 → 0.937，提升巨大。

原因：很多物体光看 geometry 分不出 articulation type。比如一个矩形盒子，可能是抽屉（prismatic，平移）、可能是门（revolute，旋转）、可能是盖子（revolute，另一种轴）。光看 mesh 你不知道，但看 texture + handle 形状 + world knowledge 就能推出来。

MLLM 的 visual reasoning 在这里是 key。Qwen3-VL 见过几十亿 image-text pair，它知道"带圆形把手的扁平方盒通常是抽屉"。

**人话**：你给机器人一个 3D 扫描的柜子，光看几何形状它不知道哪边能开。但给张照片，它看到把手在前面、铰链在侧面，就能猜出来"这是门，往前拉开"。

---

## 从 voxel token 到 mesh segmentation

MLLM 输出的 part-specific voxel token 经过 VQ-VAE decoder 解码成 sparse point cloud。但原始 mesh 是高分辨率三角形 mesh，怎么把 point cloud 的 part assignment 传播到 mesh 上？

做法是 Gaussian kernel + graph smoothing：

对 mesh 每个顶点 $v$，它属于 part $p$ 的初始概率：

$$P(v, p) \propto \exp\left(-\frac{d(v, S_p)^2}{2\sigma^2}\right)$$

- $d(v, S_p)$：顶点 $v$ 到 part $p$ 的最近 seed point 距离
- $\sigma$：尺度参数
- 距离越近概率越高，就是个 Gaussian RBF

然后在 mesh adjacency graph 上做 smoothing，让相邻顶点 label 一致。最后 majority voting 出 face label。

**人话**：voxel 是 64³ 的低分辨率，直接拿来切 mesh 会很糙。用 Gaussian + graph smoothing 把 part 信息从粗 voxel 平滑传播到细 mesh 上，相当于 super-resolution 的 segmentation。

---

## 实验数据解读

### 主实验（Table 1）

最强的 baseline 是 Particulate（https://arxiv.org/abs/2512.11798），IoU 0.643、CD 0.140。SIMART 到 IoU 0.690、CD 0.087。

最 impressive 的是 Axis Error 从 0.208 → 0.080，降 2.6 倍。说明 sparse voxel 保留了 surface detail，MLLM 能精确定位 joint axis。

### Part grounding（Table 2）

给一句话描述（比如"找出这个物体的盖子"），SIMART 的 IoU 是 0.807，而 P3SAM + Qwen3-VL（用 235B 模型！）只有 0.507。

这说明 coordinate-aware tokenization 让 MLLM 能直接把 functional description link 到 physical coordinate，不需要 separate segmentation module。8B 模型 + SIMART 设计 > 235B 模型 + 传统 segmentation。

### Ablation（Table 3）

| Method | Type↑ | Token Num↓ |
|---|---|---|
| Dense token | OOM | 4138 |
| Force Sparse | 0.661 | 862 |
| Zero Sparse | 0.794 | 516 |
| + Vision (full) | 0.937 | 516 |

Dense 直接 OOM。Force Sparse 能跑但丢 occupancy 信息性能差。Zero Sparse 性能大涨 + token 进一步减少（因为 MLLM 不用 output empty position）。加 Vision 再涨一截。

---

## 我觉得最 clever 的几个点

### 1. Zero token 的双重作用

它既是 computational trick（省 token），又是 information carrier（标记 empty space）。这两个目的在 dense representation 里是矛盾的——你想省 token 就得丢 empty 信息，但 decoder 需要知道哪是 empty。Zero token 同时解决两个问题。

参考 TRELLIS 的 sparse latent 思想：https://trellis3d.github.io

### 2. Coordinate-aware tokenization

Sparse sequence 是变长的，position 信息天然丢失。SIMART 把每个 voxel 的坐标显式 encode 进 token 序列：`⟨voxel⟩ xyz K`。这让 MLLM 能在变长序列上做几何 reasoning。

这跟 LLM 的 positional encoding 思路类似，但 position 是 3D 离散坐标。和 3D occupancy network 的思想也有关：https://arxiv.org/abs/2008.02212

### 3. Hybrid output

让 MLLM 在同一个 autoregressive sequence 里输出 symbolic（JSON）+ geometric（voxel token）。这避免了 multi-stage pipeline 的 error accumulation，也让 symbolic reasoning 和 geometric generation 互相 inform。

这是 end-to-end learning 在 3D articulation 上的体现，类似 LLM 里 reasoning 和 generation 的融合。

### 4. Emergent behavior 的 formalization

观察到 dense VQ-VAE 自然 emerge 出 empty space token，然后把它 explicit 成 zero token。这种 "发现 emergent behavior → formalize it" 的研究范式很 nice，类似 Anthropic 在 mechanistic interpretability 里的工作：https://transformer-circuits.pub

---

## 和你（Karpathy）的 work 的联想

### System 1 vs System 2

你在 e2e 演讲里讲过 LLM 要从 system 1（fast pattern matching）走向 system 2（deliberate reasoning）。SIMART 的 hybrid output 其实是让 MLLM 做 system 2 reasoning over 3D structure：

- 先识别 part（decompose）
- 再 assign property（joint type）
- 再 verify（physical consistency）
- 最后 generate（voxel token）

这和 chain-of-thought 精神一致，但 reasoning target 是 structured 3D 而不是 natural language。

### Token efficiency 是 scaling 的关键

你反复强调过 token efficiency 的重要性。SIMART 的 87% token reduction 直接决定了它能不能 scale 到 multi-part object、能不能 fit 进 8B 模型的 context window。这和 image token compression（像 native resolution ViT）、audio tokenization 的思路一脉相承。

参考你的 Software 2.0 观点：https://karpathy.medium.com/software-2-0-a63252b5e34b

### "数据是新的 PyTorch"

你在等多个场合讲过 "data is the new PyTorch"。SIMART 的 future work 里提到要用自己 generate 的 articulation prediction bootstrap 更大 dataset，这是典型的 data flywheel。和 STaR、self-improving LLM 的思路一致：https://arxiv.org/abs/2203.14465

---

## 对 embodied AI 的实际意义

### 1. Robotic manipulation training

现在 VLA model（RT-2、OpenVLA）的训练数据稀缺，scene diversity 不够。SIMART 能把海量 AIGC mesh 转成 sim-ready asset，直接 import 到 Isaac Sim 里，scale up training scene。

参考 OpenVLA: https://openvla.github.io
参考 RT-2: https://robotics-transformer2.github.io

### 2. Sim-to-real gap

SIMART 的 output 包含 physical property（density、friction、Young's Modulus），这对 sim-to-real 很关键。很多 sim 方法只输出 geometry，物理属性缺失导致 simulation 行为不真实。

### 3. VR/AR 交互

click-to-functionalize：用户点一下静态 mesh，SAM3D 生成 geometry，SIMART 加 articulation。这个 workflow 对 AR 场景 enrichment 很有用。参考 SAM3D: https://arxiv.org/abs/2511.16624

---

## 我觉得 paper 没明说但值得思考的

### Failure mode 猜测

- **极薄结构**：book page、fabric 这类薄结构的 $64^3$ voxelization 会丢 detail
- **多 DoF joint**：ball joint、universal joint 的 URDF 比单 axis 复杂，paper 里基本只测了 revolute/prismatic
- **长 kinematic chain**：折叠椅、眼镜蛇玩具这种长 chain 的 hierarchy reasoning 可能超 MLLM 能力
- **对称 ambiguity**：长方体盒子的 joint axis 可能有两个解，MLLM 怎么选？

### Scale 推断的 ambiguity

Paper 说 MLLM 用 visual cue 推断 real-world scale。但 scale 对物理 simulation 极度敏感（重力、惯性矩）。如果 scale 推断错 2 倍，整个 dynamics 就错了。这块的 robustness 值得单独 evaluate。

### Geometry fidelity 的上限

$8^3$ latent grid + 4096 codebook 是个 hard cap。对超精细 object（比如机械表）可能不够。作者 ablation 里提到 $16 \times 8 \times 8$ latent 能改善 reconstruction 但 OOM，说明 fidelity 和 context length 是 trade-off。

未来解法可能是 hierarchical tokenization（coarse-to-fine），或者 sliding window attention 让 MLLM 处理更长 sequence。

---

## 最 core 的 intuition 给你

SIMART 的故事其实就一句话：

> **3D 物体大部分是空的，显式标记空、只对非空部分做精细编码，token 数降 87%，让 MLLM 能在一个 sequence 里 jointly reason geometry + kinematics，single-stage 替代 multi-stage pipeline。**

这个 insight 和 LLM 里 sparse attention、MoE 的 sparse activation 思想是相通的——**显式建模 sparsity 比 dense 编码更 efficient，前提是你能 preserve structure**。

参考 Sparse Transformer: https://arxiv.org/abs/1904.10509
参考 MoE: https://arxiv.org/abs/1701.06538

如果你想 build 更深 intuition，建议看这几个：
- TRELLIS（SIMART 的 VQ-VAE 基础）: https://arxiv.org/abs/2412.01506
- Particulate（最强 baseline）: https://arxiv.org/abs/2512.11798
- ShapeLLM-Omni（被改进的 dense voxel 方法）: https://arxiv.org/abs/2506.01853
- VQ-VAE 原始 paper: https://arxiv.org/abs/1711.00937
- SAPIEN（articulated object simulator）: https://sapien.ucsd.edu

---

# SIMART: 把 Monolithic Mesh 变成 Sim-ready Articulated Asset

这篇 paper 来自 ByteDance Seed + NTU，核心想解决的问题是：现在 3D generation（Hunyuan3D、TRELLIS、DreamGaussian 等）能产高质量 static mesh，但这些 mesh 是 monolithic 的、没有 kinematic metadata、不能直接 drop 进 Isaac Sim / SAPIEN 里做 robotic manipulation。SIMART 想用一个 unified MLLM 把 part decomposition + joint parameter estimation 一起做掉，avoid 多 stage pipeline 的 error accumulation。

Project page: https://simart-mllm.github.io

---

## 1. 为什么这件事难：prior work 的 bottleneck

先 build intuition 关于为什么之前的方案都不行：

**Multi-stage pipeline 的问题**：比如 Urdformer、Articulate-Anything 这类方法，先做 part segmentation（用 SAM / PartField / P3SAM），再 separately 估 joint axis / origin，最后 assemble 成 URDF。问题在于 segmentation 不是 articulation-aware 的——PartField / P3SAM 优化的是 surface consistency，而不是 mechanical link boundary。结果你得到一个看起来 plausible 但 kinematically invalid 的 part split，joint 怎么估都对不上。这叫 error accumulation across decoupled modules。

**Reconstruction-based 的问题**：ArtGS、ArticulatedGS 这类用 NeRF / 3DGS 从多 state 观测里 extract kinematic structure。但它们需要同一 object 的 open/close 多个 state 的 multi-view 图像，in-the-wild 根本拿不到。

**Generative prior 的问题**：CAGE、SINGAPO 用 diffusion 学 category-level prior，但 articulated 3D dataset 太小（PartNet-Mobility 才 ~2000 个 articulated model），严重 overfitting，uncommon category 直接崩。

**MLLM + dense voxel 的问题**：PhysX-Anything、ShapeLLM-Omni 这类用 dense voxel tokenization 喂给 MLLM。64³ 的 voxel grid 展平就是 262144 个 token，绝大多数是 empty space。MLLM 的 context window 根本扛不住 multi-part assembly（一个 object 4 个 part 就要 4×262144 token），直接 OOM。即便能跑，也得 heavy downsample，geometric fidelity 丢失导致 joint axis localization 不准。

SIMART 的核心 insight：**用 sparse VQ-VAE 把 empty voxel 显式 skip 掉，token 数降 70%，让 MLLM 能在一个 stage 里 jointly reason geometry + kinematics**。

---

## 2. Problem Formulation

输入是 multimodal tuple $\mathcal{T} = \{I_{vis}, G_{geo}, T_{txt}\}$：
- $I_{vis}$：RGB image（252×252，45° isometric view）
- $G_{geo}$：raw input mesh
- $T_{txt}$：language instruction

输出 asset $\mathcal{A} = (\mathcal{M}_{seg}, \mathcal{P}_{sim})$：
- $\mathcal{M}_{seg} = \{m_1, m_2, ..., m_n\}$：part-segmented meshes
- $\mathcal{P}_{sim}$：simulation metadata，包含 joint type / axis / limits / scale / friction / density

关键点：output 既要有 geometry（每个 part 的 voxel tokens），又要有 symbolic structure（URDF 的 JSON）。这是 hybrid output sequence。

---

## 3. Sparse 3D VQ-VAE：核心贡献

这是整篇 paper 最关键的技术创新。让我拆开讲。

### 3.1 编码 pipeline

原始 mesh → voxelize 成 $64 \times 64 \times 64$ grid → 3D-Unet encoder → latent grid $Z \in \mathbb{R}^{16 \times 16 \times 16 \times C}$ → 进一步 aggregate 每 8 个相邻 token 沿 channel 维度合并 → $8 \times 8 \times 8$ latent grid，每个 token feature dim = 64 → vector quantization 成 codebook index。

Codebook $\mathcal{C}$ 有 4096 个 entry，其中 index 0 专门 reserved 为 zero token $\mathbf{e}_{zero}$，表示 unoccupied voxel。

### 3.2 Quantization 公式详解

对 latent feature $z_i$（index $i$ 处的 feature）：

$$\hat{z}_i = \begin{cases} \mathbf{e}_{zero}, & \text{if Voxel } i \text{ is unoccupied} \\ \operatorname{argmin}_{\mathbf{e}_j \in \mathcal{C} \setminus \{\mathbf{e}_{zero}\}} \|z_i - \mathbf{e}_j\|_2, & \text{otherwise} \end{cases}$$

变量解释：
- $z_i$：encoder 输出的第 $i$ 个 latent feature vector
- $\hat{z}_i$：quantize 之后的 discrete representation
- $\mathbf{e}_{zero}$：codebook 中专门表示 empty space 的 entry（index 0）
- $\mathbf{e}_j$：codebook $\mathcal{C}$ 中第 $j$ 个 entry
- $\|\cdot\|_2$：L2 距离，找 nearest neighbor
- $\mathcal{C} \setminus \{\mathbf{e}_{zero}\}$：从 codebook 里排除 zero token，避免 occupied voxel 被误 quantize 成 empty

关键 insight：传统 VQ-VAE 对每个 latent position 都做 quantization，不管 occupied 与否。这里显式地把 unoccupied 的 position 直接 assign 成 zero token，bypass 掉 nearest neighbor search。这样 MLLM 只需要 process occupied voxel 的 token。

### 3.3 Emergent Zero Token 现象

这是 paper 里一个很有意思的观察（Appendix B）：即使不显式 reserve zero token，dense VQ-VAE training 过程中会自然 emerge 出 2-4 个 codebook entry 专门 represent empty space（比如 entry 1849）。这说明 empty space 的 distribution 很 distinct，VQ-VAE 会自发学到用某些 entry 去 represent null distribution。SIMART 把这个 emergent behavior formalize 成 explicit zero token，更 robust 更 interpretable。

这个 observation 其实和 LLM 里 [PAD] token 的 emergent behavior 有点像，也和 VQGAN 里 dead code 问题相关。值得 follow up 的方向：能不能用类似的 mechanism 处理其他 modality 的 sparsity，比如 video 的 static region。

### 3.4 Coordinate-aware tokenization

Sparse token 有个问题：dense representation 靠固定 sequence length 隐式 encode 坐标（position $i$ 对应固定 3D 位置），但 sparse token 变长，position information 丢了。SIMART 的解法是每个 occupied voxel serialize 成 triplet：

```
⟨voxel⟩ [xyz] [K]
```

- `⟨voxel⟩`：start-of-voxel identifier（special token）
- `[xyz]`：coordinate token，用 linearized index $xyz = 64x + 8y + z$，其中 $x \in [0,15]$，$y \in [0,7]$，$z \in [0,7]$（注意 paper 里 Section 3.3 写的是 $x,y,z \in [0,7]$ 但 Appendix 的 system prompt 写的是 x: 0-15, y: 0-7, z: 0-7，应该以 Appendix 为准，因为 $8 \times 8 \times 8$ grid 里 x 方向被 expand 到 16）
- `[K]`：codebook index，$K \in [0, 4095]$（注意 system prompt 里写 8191 但 implementation 写 4096，可能有版本差异）

这个设计让 MLLM 能在 variable-length sequence 上做 fine-grained geometric reasoning，因为每个 token 都 explicitly 知道自己在 3D 空间的位置。

类比：这和 LLM 里 positional encoding 的思路类似，但这里 position 是 3D 离散坐标而不是 1D sequence position。和 3D scene representation 里的 Occupancy Network / Implicit Function 也有思想关联。

### 3.5 VQ-VAE training loss

$$\mathcal{L}_{total} = \mathcal{L}_{rec}(G_{geo}, \hat{G}_{geo}) + \|sg[E(G_{geo})] - \hat{z}\|_2^2 + \beta \|E(G_{geo}) - sg[\hat{z}]\|_2^2$$

- $\mathcal{L}_{rec}$：binary cross-entropy reconstruction loss，输入原始 voxel grid $G_{geo}$，重建 $\hat{G}_{geo}$
- $E(G_{geo})$：encoder 输出
- $\hat{z}$：quantized representation
- $sg[\cdot]$：stop-gradient operator，阻止梯度回传
- $\beta$：commitment loss 权重（standard VQ-VAE 用 $\beta=0.25$）
- 第二项是 codebook loss：让 codebook entry 靠近 encoder output
- 第三项是 commitment loss：让 encoder output commit 到 codebook entry

Pretrain 在 500k object subset 上，follow TRELLIS data distribution。初始化也从 TRELLIS VAE 来。

### 3.6 Reconstruction quality

Table 4 的 ablation：

| Configuration | MSE(×10⁵) | CD(×10⁵) |
|---|---|---|
| Sparse 8×8×8 (Ours) | 1.84 | 4.19 |
| Sparse 16×8×8 | 1.15 | 2.27 |
| Codebook 8192 | 1.84 | 4.56 |
| Force Sparse (no zero token) | 2.66 | 56.10 |

关键发现：
- Force Sparse（不显式用 zero token，直接丢弃 empty voxel）的 CD 暴涨到 56.10，因为 decoder 不知道哪些位置是 empty，重建时 occupation field 乱掉
- 显式 zero token 让 CD 从 56.10 降到 4.19，这是 13× 改善
- 加大 codebook 到 8192 没明显收益（4.56 vs 4.19，反而略差，可能 overfitting）
- 加大 latent grid 到 16×8×8 能改善 reconstruction（CD 4.19→2.27）但 token 数翻倍，MLLM 扛不住，所以选 8×8×8 做 trade-off

Intuition：zero token 不只是省 token，它还 preserves spatial structure 的 occupancy information，让 decoder 能正确 reconstruct empty region。这是个 non-trivial 的设计——光 skip empty voxel 会丢拓扑信息。

---

## 4. Unified MLLM

### 4.1 Backbone

用 Qwen3-VL-8B。选它的原因：
1. 大规模 image-text pretraining，有 physical world understanding 的 emergent capability
2. 能 reason 抽象物理属性（material、density、kinematic structure）
3. 8B size 在 32×A100 上能 fine-tune

### 4.2 Input 拼接

三路 modality feature concatenate 成 length $L = N_v + N_g + N_t$ 的 sequence：

- Vision：$I_{vis}$ → ViT encoder → $F_{vis} \in \mathbb{R}^{N_v \times D}$
- Geometry：$G_{geo}$ → voxelization → Sparse 3D VQ-VAE encoder + quantization → $F_{geo} \in \mathbb{R}^{N_g \times D}$（$N_g$ 是 occupied voxel 数，变长）
- Text：$T_{txt}$ → embedding → $F_{txt} \in \mathbb{R}^{N_t \times D}$

$D$ 是 hidden dimension。

### 4.3 Output 格式

MLLM 输出 hybrid sequence：
1. 每个 functional part 的 voxel tokens（用 ⟨voxel⟩ xyz K 格式）
2. Structured JSON（URDF metadata）

Table 6 给的 example：

```json
{
  "object_captions": {
    "name": "Storage Box with Frame",
    "scale": 40.0
  },
  "parts_captions": {
    "0": {
      "type": "fixed",
      "material": "Plastic",
      "density": "1.2 g/cm³",
      "Young's Modulus (GPa)": 2.5
    },
    "1": {
      "type": "revolute",
      "parent": "0",
      "center": [100, 138, 101],
      "axis": [100, 0, 0],
      "limits": [-54, 45]
    }
  },
  "parts_voxels": {
    "0": "<voxel> 0 1785 <voxel> 1 649 ...",
    "1": "<voxel> 43 1930 <voxel> 44 13 ..."
  }
}
```

注意坐标 encoding：
- `center: [x, y, z]` 整数 ∈ [0, 200]，resolution 0.005（所以 max 1.0m）
- `axis: [dx, dy, dz]` 整数 ∈ [0, 100]，方向向量
- `limits: [-val, val]`，revolute 的话 100 = 180°，prismatic 的话 100 = max distance

这个 output format 设计很关键：它让 MLLM 的 symbolic reasoning（JSON）和 geometric generation（voxel tokens）能在同一个 autoregressive sequence 里产生，jointly optimized。

---

## 5. Mesh segmentation：从 voxel token 到 surface mesh

MLLM 输出的 part-specific voxel tokens 经 VQ-VAE decoder 解码成 sparse point cloud $S_p$。但 point cloud 不是 mesh，需要 map 回原始 input mesh $G_{geo}$ 才能 preserve 高 fidelity texture 和 topology。

### 5.1 Gaussian kernel 初始化

对 mesh 上每个 vertex $v$，它属于 part $p$ 的初始概率：

$$P(v, p) \propto \exp\left(-\frac{d(v, S_p)^2}{2\sigma^2}\right)$$

- $d(v, S_p)$：vertex $v$ 到 part $p$ 的 nearest seed point cloud 的距离
- $\sigma$：scale hyperparameter，relative to mesh bounding box
- $\exp$：Gaussian kernel，让距离近的 vertex 概率高

这个就是标准的 Gaussian RBF，soft assignment。

### 5.2 Graph smoothing

初始化后用 iterative graph-smoothing operator 在 mesh adjacency matrix 上跑，保证 boundary coherence。最后 majority voting 出 face label $\mathcal{M}_{seg}$。

这个 trick 类似 GraphCut segmentation，但用 Gaussian initialization + graph smoothing 替代 energy minimization，更简单且 robust。原始 texture 直接保留。

Intuition：voxel 是 low-res 的（64³），直接拿来当 segmentation mask 会很粗糙。用 Gaussian + graph smoothing 把 voxel 的 part assignment 传播到 high-res mesh vertex 上，相当于一个 super-resolution 的 segmentation。

---

## 6. 实验

### 6.1 Dataset

- Training：39,600 个 3D object，来自 PhysXNet（5,600 articulated）+ PartNet-Mobility（34,000 static，用于 general shape comprehension）
- 每个 articulated model render 20 个 kinematic state，当独立 training instance
- 两个 instruction-following dataset：URDF generation（960k QA）+ part grounding（960k QA）
- SIMART-Bench：In-Domain（PartNet-Mobility）+ OOD（AIGC，用 Hunyuan3D-V3.1 生成），36 个 unified asset，10+ category

### 6.2 Main results（Table 1）

| Method | Type↑ | Axis↓ | Origin↓ | IoU↑ | CD↓ |
|---|---|---|---|---|---|
| Urdformer | 0.496 | 0.585 | 0.610 | 0.002 | 0.624 |
| Articulate-Anything | 0.891 | 0.315 | 0.174 | 0.202 | 0.239 |
| PhysX-Anything | 0.686 | 0.312 | 0.322 | 0.128 | 0.278 |
| Particulate | 0.822 | 0.208 | 0.204 | 0.643 | 0.140 |
| **SIMART** | **0.928** | **0.080** | **0.111** | **0.690** | **0.087** |

关键观察：
- Urdformer 的 IoU 只有 0.002，因为它不 process raw mesh，geometry 和 source 完全对不上
- Articulate-Anything / PhysX-Anything 的 geometry 都很差（IoU < 0.2），因为依赖 2D visual cue 重建 geometry
- Particulate 是最强 baseline（feed-forward 3D articulation），IoU 0.643 / CD 0.140
- SIMART 在所有 metric 上都 SOTA，尤其 Axis Error 从 0.208 降到 0.080（2.6× 改善），CD 从 0.140 降到 0.087

Axis Error 大幅改善的原因：sparse voxel token 保留了 surface detail，MLLM 能精确定位 joint axis；而 Particulate 用 standalone point segmentation，boundary 不够 articulation-aware。

### 6.3 Part grounding（Table 2）

| Method | IoU↑ | CD↓ |
|---|---|---|
| PhysX-Anything | 0.067 | 0.347 |
| P3SAM + Qwen3-VL | 0.507 | 0.234 |
| **SIMART** | **0.807** | **0.018** |

P3SAM + Qwen3-VL 是 strong baseline（用 P3SAM 做 segmentation，Qwen3-VL-235B 做 verification），但 IoU 只有 0.507。SIMART 到 0.807，CD 0.018（极低）。这说明 coordinate-aware tokenization 让 MLLM 能直接 link functional description 到 physical coordinate，不需要 separate segmentation module。

### 6.4 Ablation（Table 3）

| Method | Type↑ | Center↓ | IoU↑ | CD↓ | Token Num↓ |
|---|---|---|---|---|---|
| Dense token | OOM | - | - | - | 4138 |
| Force Sparse | 0.661 | 0.157 | 0.678 | 0.100 | 862 |
| Zero Sparse | 0.794 | 0.108 | 0.745 | 0.074 | 516 |
| + Vision (full) | 0.937 | 0.074 | 0.832 | 0.055 | 516 |

关键发现：
- Dense token 直接 OOM（4 part × 4138 token 平均，复杂 object 爆显存）
- Force Sparse（无 zero token）能跑但性能差（Type 0.661），因为丢失 occupancy 信息
- Zero Sparse 大幅改善（Type 0.794，token 数从 862 降到 516，因为 zero token 让 MLLM 不用 output empty position）
- 加 Vision 再提升（Type 0.937），说明视觉信息能 resolve geometric ambiguity——相似 morphology 但不同 articulation structure 的 object 需要视觉 cue 区分

Token 数从 4138（dense）→ 516（sparse），降了 87.5%（paper 说 70% 是 conservative 估计）。

### 6.5 为什么 Vision 这么重要

Table 3 显示加 vision 后 Type Accuracy 从 0.794 → 0.937，提升巨大。Intuition：很多 object 的 morphology 相似但 articulation 不同。比如一个 box 可能是 drawer（prismatic joint）也可能是 door（revolute joint），光看 geometry 分不出来，需要 visual appearance（texture、handle 形状）+ world knowledge 推断。MLLM 的 visual reasoning 能力在这里是 key。

---

## 7. Applications

### 7.1 Physics-based simulation

输出 URDF 直接 import 到 NVIDIA Isaac Sim。MLLM 还负责 estimate real-world scale（从 visual cue 推断 object 是 30cm 还是 1m），保证 physical consistency。

### 7.2 VR/AR

和 SAM3D 集成，click-to-functionalize：用户点一下静态 mesh，SAM3D 生成 geometry，SIMART 加 articulation。这个 workflow 对 mixed-reality scene enrichment 很有用。

---

## 8. Limitations & Future work

作者承认：articulated dataset 稀缺且质量参差，限制 open-world generalization。Future work 想用 SIMART 自己 generate pre-verified articulation prediction，bootstrap 出更大的 dataset。这是个 self-training / data flywheel 的思路，和 DAgger、STaR 的思想类似。

---

## 9. 我的 take 和延伸思考

**和 TRELLIS 的关系**：Sparse 3D VQ-VAE 初始化自 TRELLIS VAE，但 TRELLIS 用 structured latent 做 generation，SIMART 把它改造成 MLLM-compatible 的 token sequence。TRELLIS 的 sparse representation 思想被 SIMART 进一步压缩（8×8×8 + zero token）以适应 LLM context window。

**和 ShapeLLM-Omni 的对比**：ShapeLLM-Omni 也是 3D-native MLLM，但用 dense voxel token，context 爆炸。SIMART 的 zero token 机制是对它的直接改进。参考：https://arxiv.org/abs/2506.01853

**和 Particulate 的对比**：Particulate（https://arxiv.org/abs/2512.11798）是 feed-forward 3D articulation，用 point segmentation module。SIMART 的优势是 end-to-end MLLM，geometry + kinematics jointly reasoned。

**Token efficiency 的深层意义**：516 token / object 意味着 MLLM 可以在一个 sequence 里处理 multi-object scene（几个 object × 516 token 还是可接受的 context length）。这对 scene-level articulation（比如整个厨房的 cabinet + drawer + fridge）是 enabler。

**Zero token 的更广 implications**：这个 mechanism 本质上是 "显式 model sparsity 而不是让 model 自己学"。类似的思想在 video tokenization（static region 用 special token）、audio tokenization（silence token）里都有。可能是个 general principle for efficient modality tokenization。

**VLA 的 downstream 价值**：paper 提到生成的 asset 能 benchmark VLA model。这是 embodied AI 的关键 bottleneck——现在 RT-2、OpenVLA 等 VLA model 缺 diverse training scene，SIMART 能 scalable 生成 diverse articulated object 供 manipulation training。参考 OpenVLA: https://openvla.github.io

**潜在的 failure mode**（paper 没明说但值得思考）：
- 极薄 structure（比如 book page）的 voxelization 可能丢 detail
- 多 DoF joint（ball joint、universal joint）的 URDF representation 比单 axis 复杂
- Tangled / articulated chain（比如折叠椅）的 kinematic hierarchy 推理可能超出 MLLM 的 reasoning 能力

**和 my own work 的联想**（如果你真是 Karpathy）：你在 LinkedIn 上讲过 LLM 要从 "system 1 fast thinking" 走向 "system 2 reasoning"。SIMART 这种 hybrid output（symbolic JSON + geometric token）其实是让 MLLM 做 system 2 reasoning over 3D structure——先 decompose（part identification），再 assign property（joint type），再 verify（physical consistency）。这和 chain-of-thought 的精神一致，但 reasoning target 是 structured 3D 而不是 natural language。

---

## Reference links

- Project page: https://simart-mllm.github.io
- TRELLIS (VQ-VAE 基础): https://trellis3d.github.io
- ShapeLLM-Omni (对比方法): https://arxiv.org/abs/2506.01853
- Particulate (最强 baseline): https://arxiv.org/abs/2512.11798
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- PartNet-Mobility: https://sapien.ucsd.edu
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- Hunyuan3D: https://github.com/Tencent/Hunyuan3D-2
- SAM3D: https://arxiv.org/abs/2511.16624
- OpenVLA (VLA downstream): https://openvla.github.io

如果想 build deeper intuition，建议看 TRELLIS 的 sparse latent 设计（https://arxiv.org/abs/2412.01506）和 Particulate 的 feed-forward articulation（https://arxiv.org/abs/2512.11798），这两个是理解 SIMART technical contribution 的 prerequisite。
