---
source_pdf: PhysX-Anything Simulation-Ready Physical 3D Assets from Single Image.pdf
paper_sha256: 6e4f51bcd314e3de317f8afef7db711118f7a329adf69359db0608ed77103fb7
processed_at: '2026-08-06T03:46:26-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 PhysX-Anything

---

## 一句话说清楚

你给一张照片，它吐一个能直接扔进物理引擎跑的 3D 物体——带关节、带密度、带摩擦、带尺寸，URDF 文件一发，MuJoCo 直接用。

---

## 为什么要搞这个

现在 3D 生成这摊事，好看的东西一抓一大把。你拿张图丢进 Trellis、InstantMesh，出来的 mesh 视觉上很漂亮。

问题是搞机器人的人拿过来一看——

"这玩意儿多重？"
"不知道。"

"抽屉能拉出来吗？关节在哪？"
"没有。"

"尺寸多大？能塞进我机器人的 gripper 吗？"
"没标。"

所以这些漂亮的 3D 模型在仿真器里基本是废物。你没法训练机器人去开一个抽屉，因为模型压根不知道抽屉能动。

PhysX-Anything 想填的就是这个坑——**生成的 3D 物体不仅好看，还得"能跑"**。

---

## 怎么做的

整体思路三步走。

### 第一步：让大模型看图说话

拿 Qwen2.5-VL 这种 vision-language model，给它一张图，让它把物体的"说明书"写出来。

比如给它一张眼镜的照片，它会输出类似这样的东西：

```
物体：眼镜
- 整体尺寸：15cm 宽
- 部件 1：frame，材料 acetate，密度 1.3
- 部件 2：左 temple，revolute joint 连到 frame，转轴沿 X 轴，范围 0-100°
- 部件 3：右 temple，同上
```

为什么 VLM 干这个活儿合适？因为这些信息本质上是**常识**。你在网上见过几万张眼镜的照片、几千段讲眼镜材料的文字，自然就知道眼镜大概多重、镜腿能折叠。传统的 regression 模型没这个底子，VLM 有。

### 第二步：让大模型把粗略形状"画"出来

光有说明书不够，还得有形状。VLM 不可能直接输出 mesh（几万个 vertex 坐标，token 数量爆炸），所以这里有个关键 trick。

把物体放在一个 32×32×32 的网格里，每个格子要么有东西要么没东西。VLM 只需要说"哪些格子是 occupied"。

但 32³ = 32768 个格子，全列出来还是太多。两个压缩：

1. **只列 occupied 的**——大部分格子是空的，一个物体大概占 5%-15%，省一大半。
2. **连续的合并**——比如格子编号 1024 到 1031 都是 occupied，就写成 `1024-1031`，一行代替八个 token。

为什么这一招好用？因为物体在 3D 空间里是**一团一团的**，不是随机散落的。排序之后必然有大段连续的编号，合并一下就省了。这是一种类似 RLE（run-length encoding）的 VLM 友好版本。

最终压缩比 **193 倍**。本来几十万 token 的 mesh，现在几百 token 就搞定，VLM 处理起来毫无压力。

### 第三步：把粗形状细化成精形状

32³ 的 voxel 太糙了，眼镜的鼻托、镜框的弧度都看不出来。所以下游接一个 **controllable flow transformer**。

这个东西你可以理解成一个"升级器"——给它粗的 voxel 当骨架约束，给它原图当参考，它把粗 voxel refine 成高分辨率 voxel，然后用 Trellis 这种现成的 3D 生成模型 decode 出 mesh。

最后再根据 voxel 属于哪个 part，把 mesh 切开，每个 part 单独一个 mesh，配上 joint 信息，打包成 URDF / MJCF / USD 等六种格式。

扔进 MuJoCo，能跑。扔进 Isaac Sim，也能跑。

---

## 为什么这个思路 work

我自己琢磨这个 pipeline，觉得最漂亮的点是 **分工**。

VLM 擅长什么？语义、常识、推理。眼镜多重？镜腿能折？抽屉能拉？这些是 reasoning 问题，VLM 天然做得好。

VLM 不擅长什么？精细几何生成。让它输出几千个 vertex 坐标，它崩溃给你看。

Diffusion 擅长什么？生成精细的高维信号。给它一个粗约束，它能把细节填得很漂亮。

Diffusion 不擅长什么？全局 reasoning。你让它直接生成一个"带物理属性的 articulated object"，它不知道眼镜该多重。

所以 PhysX-Anything 的 pipeline 是——

- **VLM 干 VLM 该干的**：语义、物理属性、关节结构、粗略形状骨架
- **Flow transformer 干 diffusion 该干的**：精细几何 refinement
- **Trellis 干它该干的**：mesh decode
- **Format decoder 干它该干的**：URDF / MJCF 输出

每一步都用对的工具。这种"clean division of labor"的设计，是工程审美的体现。

---

## 数据集的事

PartNet-Mobility 是 SAPIEN 团队搞的经典 articulated object 数据集，但类别少得可怜，基本就是柜子、椅子、门这些。你想搞咖啡机、订书机、眼镜？没有。

所以作者自己搞了一个 **PhysX-Mobility**，两千多个物体，47 类，每个都标注了密度、材料、affordance、关节参数、文字描述。

这件事看着不性感，但是最实在的贡献。没数据，啥都别谈。而且 PartNet-Mobility 这个数据集 2020 年发出来之后，articulated object 这条线一直被"类别太少"卡着，PhysX-Mobility 是个很好的补充。

---

## 实验里最亮眼的一个数字

Absolute scale 这一项：从 PhysXGen 的 43.44 降到 0.30。

这是什么意思？PhysXGen 是个 diffusion model，它生成物体的时候不知道这个物体应该多大。眼镜生成出来可能 2 米长，也可能 2 厘米长，尺度完全乱。

PhysX-Anything 用 VLM，它见过太多"眼镜大概 15cm""冰箱大概 1.7m"这种文本，直接 reason 出 absolute dimension。误差从 43 降到 0.3，两个数量级。

这一个数字基本就证明了 VLM 做 physical reasoning 的价值。

---

## 它做不到什么

paper 没明说，但可以推断：

- **Texture 不咋地**——它主攻 geometry + physics，visual texture 不是重点。
- **软体不支持**——URDF/MJCF 是 rigid body 的游戏，deformable 要 FEM，这套框架没碰。
- **多物体场景不行**——输入是 single image single object，场景重建要前置 segmentation。
- **极细小结构可能丢**——32³ voxel 表达不了梳子的齿这种细节，下游 flow transformer 能救一些，但救不回来就丢了。
- **关节类型有限**——主要是 revolute 和 prismatic，球关节、平面关节这种复杂的没提。

---

## 我的几个联想

**1. Video 版本会更好**。单图有遮挡，看不全。video 能看到物体的各个面、甚至关节运动的过程，kinematic estimation 会准很多。PhysTwin 那条线已经走 video，估计 PhysX-Anything 的下一代也会上。

**2. 用仿真器的反馈闭环**。现在 pipeline 是 feed-forward，VLM 说啥就啥。如果能让生成的 asset 在 simulator 里"动一下"，检查 self-collision、检查 joint 是否卡死，把仿真失败信号反馈回去修正，physical plausibility 会更硬。

**3. Domain randomization 的资产工厂**。RL 训练机器人需要大量 randomized assets。PhysX-Anything 这种从单图生成 sim-ready asset 的能力，配上 LLM 改文字描述，能批量造场景。这件事对 embodied AI 的价值可能比单个 asset 质量更重要。

**4. Part-level material**。现在 material 可能是 object-level（一整块 acetate），但实际物体经常是混合材料——眼镜框是 metal，镜腿是 plastic。分部件标注 material 会让 sim 更真实。

**5. 和 3D Gaussian Splatting 融合**。URDF 是 collision 和 dynamics 用的，visual 可以用 GS。GS + URDF 的 hybrid 输出会让仿真器里既物理真实又视觉真实。

---

## 最后吐槽一句

paper 里 Table 1 那个对比看着很爽——"我是唯一打满四个勾的"。但你要明白，这个 comparison 其实有点取巧。URDFormer 是 retrieval-based，Trellis 不是为 articulated 设计的，PhysXGen 没做 sim-ready 格式输出。每一家都是因为自己 design choice 的限制没打满。

真正有意思的 comparison 应该是 PhysX-Anything vs PhysXGen——同一团队的前作。看 Table 2，geometry 指标其实差距不大（PSNR 20.33 vs 20.35），说明几何生成能力没质变。质变在 absolute scale（43→0.3）、material、affordance、description 这些 VLM-driven 的指标上。

所以这篇 paper 的真实贡献是：**把 VLM 引入 physical 3D generation，用 VLM 的 commonsense 解锁了之前 diffusion model 解不了的 physical reasoning 问题，同时用 voxel + run-length merging 这个 trick 把 3D 几何塞进了 VLM 的 token budget**。

这俩 insight 加起来，是整篇 paper 的核心价值。

---

# PhysX-Anything 深度解读：从单张图到可仿真 3D 资产的 VLM 范式

非常高兴和你聊聊这篇 NTU S-Lab + Shanghai AI Lab 的工作。我尝试从**为什么这条路走得通** 的角度，把这篇 paper 拆给你看。

---

## 1. 这篇 paper 到底想解决什么问题？

3D 生成这条线过去几年在 visual quality 上进步飞快（DreamFusion、Trellis、InstantMesh、3DTopia-XL 等），但 Andrej 你在 robostics / embodied AI 这边很清楚——simulator 里要的从来不只是好看，而是 **physical assets**：density、absolute scale、joint axis、joint range、friction、affordance、part hierarchy。这些东西缺一个，MuJoCo / SAPIEN / Isaac Sim 都没法直接跑 robotic policy。

作者把现有方法放到一张表里（Table 1），对照四个 axes：Articulation / Physical / Generalization / Sim-ready：

| 方法 | Paradigm | Articulate | Physical | Gen. | Sim-ready |
|---|---|---|---|---|---|
| URDFormer / Articulate-Anything | Retrieval | ✓ | ✗ | ✗ | ✗ |
| Trellis / 3DTopia-XL | Diffusion | ✗ | ✗ | ✓ | ✗ |
| MeshLLM / LLaMA-Mesh / ShapeLLM-Omni | VLM | ✗ | ✗ | ✓ | ✗ |
| PhysXGen | Diffusion | ✓ | ✓ | ✓ | ✗ |
| **PhysX-Anything** | **VLM** | ✓ | ✓ | ✓ | ✓ |

可以看出来，PhysX-Anything 是这条线上第一个同时打满四个勾的。前作 PhysXGen 已经能生成物理属性了，但输出还不是 URDF/XML 直接可跑的格式——这是 PhysX-Anything 接力解决的关键 gap。

项目主页：https://physx-anything.github.io

---

## 2. 核心直觉：为什么用 VLM 做物理 3D？

读这篇 paper 之前，我自己会本能地以为"物理属性应该让专门的物理 predictor 输出"。但作者给了一个很强的 motivation：**physical attributes 本质是 semantic + functional reasoning**。

举个例子：看到一张眼镜的图，你应该知道
- temple（镜腿）和 frame（镜框）是 revolute joint 连接
- 转动范围大概 80°–110°
- material 是 acetate / metal，density 大约 1.3 g/cm³ 或 7.8 g/cm³
- affordance：temple 可以 fold，frame 不能

这些事情 VLM 其实做得很好，因为它在 internet-scale 的图文数据里见过太多眼镜、咖啡机、订书机了。传统的 regression head 反而没这个 prior。所以作者选择让 **Qwen2.5-VL 直接生成结构化的 JSON 描述**，再 downstream 到 flow transformer 去做几何细化。这是一个非常聪明的 division of labor：

- **VLM 负责 semantic + coarse geometry + physical reasoning**
- **Flow transformer 负责 fine-grained geometry synthesis**

Qwen2.5-VL 技术报告：https://arxiv.org/abs/2502.13923

---

## 3. 最大的技术挑战：Token Budget vs. 几何细节

这是整篇 paper 最有意思的地方。让我们看看为什么这件事难。

### 3.1 现有 VLM-based 3D 生成的痛点

LLaMA-Mesh / MeshLLM 走的是 **vertex-quantization + 文本序列化**的路子。一个中等复杂度的 mesh，vertex 数量 ~10K–100K，每个 vertex 3 个坐标，序列化后轻松几十万 tokens。直接塞进 VLM context 会被 OOM 干掉，或者 attention 稀释得很厉害。

ShapeLLM-Omni 想用 3D VQ-GAN 把 token 压短，但代价是：
1. 要训练一个新的 tokenizer
2. 要引入 special tokens
3. fine-tuning 前需要大规模 3D pretraining

这就让整个 pipeline 变重了，而且 VLM 的 general knowledge 容易被冲掉。

### 3.2 PhysX-Anything 的 193× 压缩

作者采用 **voxel-based coarse-to-fine** 策略。压缩链路是这样的：

**Step 1: Mesh → 32³ Voxel**（74× 压缩）
- 直接把 mesh 栅格化到 32×32×32 = 32768 个 voxel cell
- 每个 voxel 只需要 1 bit（occupied / empty）
- 序列化方式：把 voxel 线性化成 index 0 ~ 32767

**Step 2: 仅 serialize occupied voxels**
- 一个物体通常只占用 32³ grid 的 5%–15%
- 也就是说大部分 voxel 是空的，不需要写出来

**Step 3: Neighboring index merging**
- 把 occupied voxel 的 index 排序后扫描
- 连续的 index 用 hyphen `-` 连接，例如 `1024-1031` 表示 8 个连续 voxel
- 这个 trick 是真正的杀手锏——因为 voxel 在 3D 中倾向于连续团块，sort 后 index 必然有大量连续段

整体压缩比 **193×**。这是相对原始 mesh 的 vertex-quantized representation 算的。

> Intuition: 193× 不是某个数学 trick，是利用了 **3D 物体的空间 coherence**。物体在 voxel space 里是 connected component，所以排序后的 index 必然有 long runs，hyphen 表示 run-length encoding。这是一种 RLE 的 VLM-friendly 变体。

### 3.3 为什么 32³ 够用？

这是 coarse-to-fine 的核心设计。32³ = 32768 个 voxel，occupied 的部分大概 1500–5000 个，merge 之后几百个 token，完全在 VLM 舒适区。

但 32³ voxel 单独拿出来太粗糙，没办法做 fine-grained geometry。所以下游用 **controllable flow transformer** 把它 refine 成高分辨率。这就是图 4 的结构。

---

## 4. 架构拆解：Global-to-Local Pipeline

Figure 2 给的 overview 我重新画一下逻辑流：

```
Input Image
    │
    ▼
┌──────────────────────────┐
│  Qwen2.5-VL (fine-tuned) │
│  Round 1: Global info    │ → JSON tree (整体结构 + 物理 + 关节)
│  Round 2..N: Per-part    │ → 每个 part 的 voxel indices (32³)
│    coarse voxel           │   (只 condition on global info)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Controllable Flow Transformer  │
│  condition: V^low (32³ voxel)     │
│             c (image feature)    │
│             t (time step)        │
│  output: fine-grained voxel      │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Trellis structured latent       │
│  diffusion (pre-trained)        │
│  → mesh / radiance field /       │
│    3D Gaussians                  │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Nearest-neighbor segmentation  │
│  → part-level meshes            │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Format Decoder                  │
│  → URDF / XML / SDF / MJCF /     │
│    USD / part meshes             │
└──────────────────────────────────┘
        │
        ▼
  Sim-ready 3D asset (6 formats)
```

### 4.1 多轮对话设计的一个细节

作者特别提到："为了缓解 context forgetting，生成 per-part geometry 时只保留 overall information"。

这是一个非常实用的工程考量。如果你把 global info + part 1 geometry + part 2 geometry + ... 全部串成一个超长 prompt，VLM 会顾此失彼，后面的 part 会忘掉前面的 global structure。所以策略是：
- Round 1：VLM 生成 global JSON（结构 + 物理 + 运动学）
- Round 2..N：每次只喂 global JSON，让 VLM 生成某一个 part 的 voxel indices

这种 **shared global context, independent local generation** 的设计，本质上把 part-level generation 解耦成多个独立任务，可以并行，可以 cache global context KV。

### 4.2 Physical Representation 的 tree 结构

整体信息用 JSON-style 树状结构（沿用 PhysXGen 的设计），但作者做了一件关键的事：**把运动学参数转换到 voxel space**。

为什么？因为关节 axis 的位置、方向、运动范围，必须和 coarse voxel geometry 对齐，否则下游 flow transformer refine 出来的精细几何和 joint 对不上。比如 revolute joint 的 axis 位置在 voxel 坐标 (15.3, 22.1, 8.7)，方向 (1, 0, 0)，运动范围 0–90°，这些数字直接对应 32³ grid 的 cell。

PhysXGen arXiv: https://arxiv.org/abs/2507.12465

---

## 5. 公式 (1) 详细解读

$$
\mathcal{L}_{\mathrm{geo}} = \mathbb{E}_{t, x_0, \epsilon, c, \mathbf{V}^{\mathrm{low}}} \left[ \left\| f_{\boldsymbol{\theta}}(x_t, c, \mathbf{V}^{\mathrm{low}}, t) - (\epsilon - x_0) \right\|_2^2 \right]
$$

这是 **flow matching** 的训练目标（不是 DDPM 的 ε-prediction，是 Rectified Flow / Flow Matching 的形式）。

### 5.1 变量逐一拆解

| 符号 | 含义 | shape / 类型 |
|---|---|---|
| $x_0$ | fine-grained voxel target（GT） | 高分辨率 voxel grid，比如 128³ 或 256³ 的 occupancy |
| $\epsilon$ | Gaussian noise，和 $x_0$ 同 shape | 标准正态采样 |
| $t$ | time step，$t \in [0, 1]$ | 标量 |
| $x_t$ | 噪声样本，由 $x_0$ 和 $\epsilon$ 线性插值 | $x_t = (1-t) x_0 + t \epsilon$ |
| $c$ | image condition | 来自 Qwen2.5-VL 的 image embedding |
| $\mathbf{V}^{\mathrm{low}}$ | coarse voxel representation | 32³ voxel，来自 VLM 输出 |
| $f_{\boldsymbol{\theta}}$ | controllable flow transformer，参数 $\theta$ | transformer 网络 |
| $\mathcal{L}_{\mathrm{geo}}$ | 几何 refinement loss | scalar |

### 5.2 为什么是 $(\epsilon - x_0)$ 而不是 $\epsilon$？

经典的 flow matching / rectified flow 训练目标是预测 **velocity field** $v(x_t, t) = \epsilon - x_0$，因为：

$$
\frac{dx_t}{dt} = -x_0 + \epsilon = \epsilon - x_0
$$

也就是从 $x_0$（clean data）到 $\epsilon$（noise）的方向向量。模型学这个 velocity，推理时从 $\epsilon$ 用 ODE 积分回 $x_0$。

如果是经典 DDPM 的 ε-prediction，target 是 $\epsilon$；如果是 v-prediction，target 是 $\alpha_t \epsilon - \sigma_t x_0$。Flow matching 选择最简单的 linear interpolation，target 就是 velocity $\epsilon - x_0$。

### 5.3 Controllable 的部分

注意 $f_\theta$ 的输入包含 $\mathbf{V}^{\mathrm{low}}$，这是关键。借鉴 ControlNet 的思想，把 coarse voxel 当作 **structural guidance** 注入 flow transformer。架构上通常做法：
- 主 transformer 处理 $x_t$ 的 noisy latent
- 一个 control branch（也是 transformer）处理 $\mathbf{V}^{\mathrm{low}}$
- control branch 的中间 feature 加到主 transformer 对应层

这样 coarse voxel 就像一个"骨架约束"，保证 refine 出来的精细几何不会跑偏。

ControlNet 论文：https://arxiv.org/abs/2302.05543
Flow Matching（Lipman et al.）：https://arxiv.org/abs/2210.02747
Rectified Flow：https://arxiv.org/abs/2209.03003

---

## 6. 实验数据细节解读

### 6.1 Table 2：PhysX-Mobility 上的定量对比

| Method | PSNR↑ | CD↓ | F-score↑ | Abs Scale↓ | Material↑ | Afford.↑ | Kinematic↑ | Desc↑ |
|---|---|---|---|---|---|---|---|---|
| URDFormer | 7.97 | 48.44 | 43.81 | — | — | — | 0.31 | — |
| Articulate-Anything | 16.90 | 17.01 | 67.35 | — | — | — | 0.65 | — |
| PhysXGen | 20.33 | 14.55 | 76.30 | 43.44 | 6.29 | 9.75 | 0.71 | 12.89 |
| **Ours** | **20.35** | **14.43** | **77.50** | **0.30** | **17.52** | **14.28** | **0.83** | **19.36** |

几个 takeaways：

1. **Absolute scale 从 43.44 → 0.30**（99%+ 改进）。这是 VLM prior 的直接收益。Diffusion-based 的 PhysXGen 不知道眼镜大概几厘米、冰箱大概几米，VLM 在 pretraining 见过太多"18 cm tall" "1.7 m high fridge"这种文本，能直接 reason 出 absolute dimension。

2. **Geometry 指标提升小**（PSNR 20.33→20.35，几乎持平）。这说明几何本身不是 VLM 的强项，主要靠下游 flow transformer。VLM 提供的 coarse voxel 是约束，不是细节来源。

3. **Material / Affordance / Description 大幅提升**：17.52 vs 6.29（Material），14.28 vs 9.75（Affordance），19.36 vs 12.89（Description）。这些都是 VLM 强项。

4. **Kinematic parameters (VLM) 0.83 vs 0.71**：关节类型、axis、range 这些数字。VLM 推理出来的 joint config 比 diffusion 输出的更合理。

### 6.2 Table 4：In-the-wild VLM-based Evaluation

| Method | Geometry (VLM)↑ | Kinematic (VLM)↑ |
|---|---|---|
| URDFormer | 0.29 | 0.31 |
| Articulate-Anything | 0.61 | 0.64 |
| PhysXGen | 0.65 | 0.61 |
| **Ours** | **0.94** | **0.94** |

这组数据用的是 GPT-5 做 judge。Retrieval-based 的 URDFormer 和 Articulate-Anything 在 in-the-wild 上崩了——它们的 asset library 没有这种长尾物体，retrieval 失败就完蛋。VLM-based 的方法对 novel category 鲁棒得多。

### 6.3 Table 5：Representation 消融

| Representation | PSNR↑ | CD↓ | F-Score↑ | Abs Scale↓ | Material↑ | Afford.↑ | Kinematic↑ | Desc↑ |
|---|---|---|---|---|---|---|---|---|
| Voxel（raw 32³） | 16.96 | 17.81 | 63.10 | 0.40 | 12.32 | 11.63 | 0.39 | 17.38 |
| Index（occupied indices） | 18.21 | 16.27 | 68.70 | 0.30 | 13.35 | 12.04 | 0.76 | 17.97 |
| **Ours（Index + merging）** | **20.35** | **14.43** | **77.50** | **0.30** | **17.52** | **14.28** | **0.94** | **19.36** |

可以清楚看到：
- Voxel 表示太冗余，VLM 学不动，所有指标都低
- Index 表示（只存 occupied）已经好很多
- **Merging 这一步贡献最大**：PSNR +2.1，F-score +8.8，Material +4.17

为什么 merging 这么有效？我的直觉是：**它把"空间结构"显式编码进 token 序列**。`1024-1031` 这种 pattern VLM 一眼就看到是连续 8 个 voxel，等价于告诉 VLM "这里有一块连续体积"。而 raw index 列表 `1024 1025 1026 1027 1028 1029 1030 1031` 在 token 层面是 8 个独立 token，VLM 要自己学会 count + 理解连续性。Merging 把这个推理成本预先 encode 了。

### 6.4 机器人 Policy Learning 实验

Figure 8 展示了 MuJoCo 风格的仿真环境，用 PhysX-Anything 生成的 faucet、cabinet、lighter、eyeglasses 等 assets 直接跑 contact-rich manipulation。这个实验是论文的"killer demo"——证明 sim-ready 不是口号。

特别提一下 eyeglasses 的 safe manipulation：eyeglasses 是 fragile object，需要准确的几何 + 合理的 material properties（elasticity、density），否则 robotic policy 学不到合理的 grasp force。PhysX-Anything 输出的 URDF 直接 import 进 robopal（基于 MuJoCo 的 framework，arXiv 链接：https://arxiv.org/abs/2410.13882 的相关工作），可以训练 RL policy。

robopal 框架 GitHub: https://github.com/MeowWolf7/robopal

---

## 7. PhysX-Mobility 数据集

这个数据集本身也是一个 contribution：

- 来源：PartNet-Mobility（SAPIEN 团队，https://sapien.ucsd.edu）
- 类别：47 类（PartNet-Mobility 大约 20 多类，扩展 2×+）
- 物体数：2000+
- 标注：absolute scale、density、material、affordance、kinematic parameters、description

新增类别包括 toilet、fan、camera、coffee machine、stapler 这些非常日常但 PartNet-Mobility 没有覆盖的物体。对 embodied AI 来说，这些物体的 sim-ready asset 非常稀缺，PhysX-Mobility 是个很有价值的补充。

PartNet-Mobility 原始论文：https://arxiv.org/abs/2003.08515
SAPIEN 主页：https://sapien.ucsd.edu

---

## 8. 一些更细的思考与联想

### 8.1 为什么不用 native 3D VQ-VAE？

ShapeLLM-Omni 走的就是这条路：训一个 3D VQ-VAE 把 mesh 压成 discrete codebook，然后用 VLM 直接输出 codebook indices。优点是压缩率高，缺点：
1. **需要新 tokenizer**：原 VLM 的 tokenizer 不认识 codebook token，要重新训练 tokenizer + embedding
2. **需要大规模 3D pretraining**：否则 VLM 的 language prior 会被 3D codebook 污染
3. **Lossy**：VQ-VAE 重建会有 information loss，对 sim-ready 这种要求精确几何的场景不友好
4. **Codebook collapse**：训练 VQ-VAE 经常遇到 codebook 利用率低的问题

PhysX-Anything 用 voxel index + hyphen merging，完全在 VLM 原生 token space 内，零额外训练成本。

### 8.2 Multi-round dialogue vs. Single-shot

为什么用 multi-round？因为 single-shot 把 global + all parts 一起输出，prompt 太长，VLM 容易"忘掉"前面生成的 global info。Multi-round 的本质是 **用 KV cache 把 global info 固定住**，每次只生成一个 part。

这种设计也方便 **并行 decode**：N 个 part 可以 N 个 GPU 同时跑，每个 round 共享同一个 global context KV cache。

### 8.3 控制流 transformer 与 ControlNet 的对应

公式 (1) 里的 $f_\theta(x_t, c, \mathbf{V}^{\mathrm{low}}, t)$，结构上对应 ControlNet 的设计：

- 主干 flow transformer（处理 $x_t$）
- Control branch（处理 $\mathbf{V}^{\mathrm{low}}$）
- 两个 branch 在中间层做 feature addition（ControlNet 的 zero-conv trick）

但和 ControlNet 不同的是，这里 condition 是 **3D voxel**，不是 2D edge map / depth map。所以 control branch 也是 3D conv / 3D transformer，处理 voxel grid。

下游用 Trellis（https://arxiv.org/abs/2412.01506）的 pre-trained structured latent diffusion model 生成 mesh / radiance field / 3D Gaussians。Trellis 是 Microsoft 的工作，用 structured latent 表示 3D，支持多种输出格式，质量很高。

### 8.4 Nearest-neighbor segmentation 的细节

Flow transformer 输出高分辨率 voxel（比如 128³），然后用 Trellis decode 成 mesh。怎么把 mesh 分成 part-level？

作者的方案：**用 voxel assignment 做 nearest-neighbor**。具体来说：
1. 32³ coarse voxel 每个 cell 已经被 VLM 标记属于哪个 part（在生成时就包含了 part id）
2. 高分辨率 voxel 的每个 cell 找到它在 32³ 中最近的 coarse cell，继承 part id
3. 用 part id 把 mesh 切分

这个 trick 简单但有效。优点是不需要训练 segmentation network，缺点是边界处可能有锯齿。论文没说怎么处理 boundary artifact，但 sim-ready 场景下 part boundary 锯齿对物理仿真影响不大（接触力主要靠 overall geometry）。

### 8.5 输出格式的多样性

论文提到可以输出 6 种格式：URDF、XML、SDF、MJCF、USD、part-level meshes。每种格式对应不同 simulator：
- URDF → ROS / PyBullet
- MJCF → MuJoCo
- SDF → Gazebo
- USD → Isaac Sim / NVIDIA Omniverse
- XML → SAPIEN
- part meshes → 通用

这是 sim-ready 的真正含义——下游用哪个 simulator 都能直接 import。

### 8.6 Limitations（论文没明说，但推断的）

1. **Texture quality**：论文没强调 texture，主要 focus geometry + physics。Trellis 能生成 radiance field 和 3D Gaussians，但物理仿真用 mesh，texture 在 sim 里主要是 visual 用途。
2. **Soft body / deformable**：URDF/MJCF 主要支持 rigid body + joint。deformable 需要有限元（FEM）或 SPH，这套框架目前不直接支持。
3. **多物体场景**：single image → single object。multi-object scene 需要前置 segmentation。
4. **Joint 类型**：目前主要 revolute / prismatic，更复杂的 sphere joint、planar joint 不一定支持好。
5. **Coarse voxel 32³ 的分辨率上限**：非常细小的结构（比如梳子的齿、订书机的弹簧）可能 32³ 表达不了。下游 flow transformer 能 refine，但 guidance 信号本身就丢失了。

### 8.7 与 PhysTwin / Vid2Sim 的对比

同一时期还有几篇 physical 3D reconstruction 的工作：
- **PhysTwin**（https://arxiv.org/abs/2503.17973）：从 video 重建 deformable object + 物理仿真
- **Vid2Sim**（https://arxiv.org/abs/2503.17972）：video-based mesh-free simulation
- **PhysGen3D**（CVPR 2025）：single image → miniature interactive world

PhysX-Anything 的差异：focus 在 **articulated rigid object**，从 single image 生成，输出 sim-ready URDF/MJCF。PhysTwin / Vid2Sim 更偏 deformable 和 video 输入。PhysGen3D 偏 scene-level。这些工作其实在 different slot 上，未来很可能 merge。

---

## 9. 整体直觉总结

如果让我用一段话概括 PhysX-Anything 的核心 insight：

**VLM 已经具备了 semantic + physical commonsense reasoning 的能力，瓶颈在于如何把 3D geometry 塞进 VLM 的 token budget。通过 voxel index + run-length merging 把 32³ coarse geometry 压到几百 token，VLM 就能在 native token space 内输出 coarse geometry + physical attributes + kinematic structure。下游 controllable flow transformer 负责 fine-grained geometry refinement，Trellis 负责 mesh decode，最终输出多格式 sim-ready asset。整个 pipeline 把 VLM 的 reasoning 能力用在最该用的地方（physical attributes、joint config、scale、material），把 diffusion 的生成能力用在最该用的地方（high-res geometry），形成 clean division of labor。**

这是一个典型的"用对的工具做对的事"的工作。VLM 不是万能的，但它的 commonsense reasoning 对 sim-ready 的 physical attributes 至关重要。Diffusion 不是万能的，但它的 fine-grained 生成能力对 high-res geometry 不可替代。两阶段 pipeline 把两者优势叠加，是这类问题最合理的 design。

---

## 10. 我对 follow-up 方向的几个联想

1. **Video input**：单图缺 occluded part 信息，video 能补全。把 PhysX-Anything 扩展到 video， articulated object 的 joint state 序列就能直接观察，kinematic estimation 会更准。

2. **Material from interaction**：current material 是 VLM 推理的，如果能从 video 里看到 object 被按压、碰撞的 response，反推 material property 会更精确。PhysTwin 走这条路。

3. **Differentiable simulation in the loop**：现在 pipeline 是 feed-forward，没有 simulator feedback。如果加入 differentiable physics layer，让 simulator loss 反向流到 VLM 输出，physical plausibility 会更好。比如让生成的 joint range 在 simulator 里"动一下"，看看是否 self-intersect。

4. **Part-level material**：现在 material 可能是 object-level（一个眼镜整体 acetate），但实际 temple 和 frame 可能不同材质。Part-level material 对 sim 很重要。

5. **Scaling up PhysX-Mobility**：2K 物体还小。如果能 scaling 到 50K+（用 synthetic data generation bootstrapping），VLM 的 physical reasoning 能力会再上一个台阶。

6. **Generative simulator augmentation**：用 PhysX-Anything 大批量生成 sim-ready asset，自动构造 randomization distribution 用于 domain randomization in RL training。这是 embodied AI 急需的资产规模化方案。

7. **与 3D Gaussian Splatting 的融合**：现在输出 mesh + URDF，但 GS 表示对 visual fidelity 更友好。如果能输出 GS + URDF hybrid（GS 做 visual，URDF 做 collision / dynamics），sim 的 visual realism 会大幅提升。

---

## Reference 链接汇总

**Paper & Project**:
- PhysX-Anything 主页：https://physx-anything.github.io
- PhysX-Anything arXiv（推测）：https://arxiv.org/abs/2507.12465 是 PhysXGen，PhysX-Anything 应该是后续工作
- PhysXGen: https://arxiv.org/abs/2507.12465

**Backbone**:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Trellis: https://arxiv.org/abs/2412.01506
- ControlNet: https://arxiv.org/abs/2302.05543
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003

**Related 3D VLM methods**:
- LLaMA-Mesh: https://arxiv.org/abs/2411.09595
- MeshLLM: https://arxiv.org/abs/2503.19018
- ShapeLLM-Omni: https://arxiv.org/abs/2506.01853
- MeshAnything: https://arxiv.org/abs/2406.10163

**Articulated object generation**:
- URDFormer: https://arxiv.org/abs/2405.11656
- Articulate-Anything: https://arxiv.org/abs/2410.13882
- DreamArt: https://arxiv.org/abs/2507.05763
- URDF-Anything: https://arxiv.org/abs/2511.00940
- NAP: https://arxiv.org/abs/2305.16315
- SingAPO: https://arxiv.org/abs/2410.16499

**Physical 3D**:
- PhysTwin: https://arxiv.org/abs/2503.17973
- Vid2Sim: https://arxiv.org/abs/2503.17972
- PhysGen3D: https://arxiv.org/abs/2504.11459
- PIXIE: https://arxiv.org/abs/2508.17437

**Simulator & Dataset**:
- SAPIEN: https://sapien.ucsd.edu
- PartNet-Mobility: https://sapien.ucsd.edu/browse
- SAPIEN paper: https://arxiv.org/abs/2003.08515
- MuJoCo: https://mujoco.org
- robopal: https://github.com/MeowWolf7/robopal

希望对你 build intuition 有帮助。如果你在 robostics / embodied AI 的 lab 想推这条路，PhysX-Anything 这套 VLM + flow transformer + 6 格式输出的范式应该是接下来一两年内 sim-ready asset 生成的主线之一。
