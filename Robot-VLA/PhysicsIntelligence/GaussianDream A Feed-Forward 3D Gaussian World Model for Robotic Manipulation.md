---
source_pdf: GaussianDream A Feed-Forward 3D Gaussian World Model for Robotic Manipulation.pdf
paper_sha256: eb80465d4f9316d97d7ac206f15d2a93d1e17a990c2399c73749ee962baf52f5
processed_at: '2026-08-04T12:49:34-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GaussianDream

## 1 这篇论文在解决什么问题

先说个直觉。你想象一下，你让一个机器人"把杯子放到左边的盘子上"。对人类来说这事儿太简单了，但现在的 VLA 模型（就是那种用 PaliGemma、Gemma 这类大模型来输出机器人动作的 policy）经常搞砸。为什么？

因为它们本质上是在 **2D pixel grid** 上做 pattern matching。它们从来没有真正"理解"过 3D 空间。你可以把当前主流的 VLA 想象成一个很聪明的鹦鹉——它能说出"往左移动 5 厘米"，但它脑子里并没有一个真正的 3D 世界模型，它只是学会了"看到这种 pixel pattern，就输出这种 action"的映射。

这就导致几个具体的 failure mode：
- **Grasp point 偏移**：杯子看起来差不多，但实际抓偏了
- **Contact 判断错误**：不知道夹爪离物体还有多远
- **交互后果预测缺失**：推完一个东西，不知道它会滚到哪儿去

论文里 [Sun et al., 2025](https://arxiv.org/abs/2508.09071) 和 [Li et al., 2025a](https://arxiv.org/abs/2510.12276) 都观察到这些问题。

## 2 别人是怎么尝试解决的

大概有两派思路，都有硬伤。

**第一派：3D-Enhanced Policy**

代表是 GeoVLA、StereoVLA、VLA-4D 这些。思路是给 policy 喂 depth map、point cloud、stereo 之类的几何信号。问题在于，它们只看 **当前这一帧** 的几何，相当于给鹦鹉戴上一副 3D 眼镜，但鹦鹉还是不会"想象"下一秒会发生什么。机器人推一个方块，policy 根本不知道推完之后方块会跑到哪里。

**第二派：World Model / World Action Model**

代表是 DreamZero、Cosmos Policy、Motus、WorldVLA 这些。思路是让模型预测未来——要么预测未来的 RGB video，要么预测未来的 latent state，要么直接预测 action sequence。这听起来很美，问题在于推理的时候太慢。你想，要预测 5 帧未来，就得 autoregressive rollout 5 次，每次都是一次 forward pass through 一个大模型。机器人控制要 10-20Hz，你 rollout 一次就要 700ms+，根本玩不转 high-frequency closed-loop control。

[Lu et al., 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.html) 的 GWM 和 [Lu et al., 2024](https://link.springer.com/chapter/10.1007/978-3-031-72615-8_21) 的 ManiGaussian 用 3D Gaussian 来做 world model，expressiveness 很强，但需要 iterative voxel optimization，推理开销巨大。

所以核心 tension 是：**3D 表达力 vs. 推理效率**。你要是想要 expressive 的 3D 表示，推理就慢；你要是想推理快，就只能用浅层的 2D feature。

## 3 GaussianDream 的核心 insight

GaussianDream 的 insight 非常 elegant，一句话概括：

> **训练的时候，让模型"学习"怎么重建 3D 世界、怎么预测 3D 世界的未来演化；推理的时候，把这些"重建器"和"预测器"全丢掉，只保留一个压缩好的 prefix token 序列，用来 condition action generation。**

这就像你学开车。教练在旁边告诉你"注意左边那辆车要变道"、"前面 50 米有红灯"，你在脑子里模拟整个交通场景的演化。但真正开熟了之后，你不需要每次都显式模拟，你的直觉就包含了这些知识——你看到某个场景，手就自然知道怎么动。

GaussianDream 就是把这个"训练时的显式模拟"压缩成"推理时的隐式直觉"。这个 prefix 就是那个"直觉"的载体。

## 4 整个 pipeline 走一遍

我按照数据流走一遍，把每一步在干什么讲清楚。

### 4.1 输入是什么

- **$\mathbf{o}_{t-K:t}$**：三帧历史观察，实际用的是 $\{t-10, t-5, t\}$。为什么稀疏采样？因为密集采样信息冗余，稀疏的 3 帧已经能提供足够的 motion cue，同时计算量小。
- **$\mathbf{l}$**：language instruction，比如"pick up the red cup"
- **$\mathbf{s}_t$**：robot proprioceptive state（joint position, gripper state 等）
- **$\mathbf{o}_t$**：当前帧的多视角 RGB（agent-view + wrist-view）

### 4.2 构建 GaussianDream Prefix

这是整个方法的灵魂。公式是：

$$\mathbf{Z}_t^{\mathrm{GD}} = \mathcal{F}_\omega(\mathbf{o}_{t-K:t}, \mathbf{Q}_{\mathrm{GD}})$$

拆开看：
- **$\mathbf{Q}_{\mathrm{GD}}$**：1024 个 learnable queries，reshape 成 $32 \times 32$ 的 grid。这个 grid 结构很关键，它让 queries 天然有 spatial layout，后面才能 decode 成 dense Gaussian map
- **$\mathcal{F}_\omega$**：由 TGE Module (Temporal Gaussian Evolution) 实现

TGE Module 内部干的事儿：

第一步，用 [VGGT](https://arxiv.org/abs/2406.07751) 提取每帧的 multi-scale 3D-aware feature：

$$\mathbf{P}_{t-K:t}^{(m)} = W_m \mathcal{P}_{32 \times 32}\left(\mathcal{E}_{\mathrm{VGGT}}^{(m)}(\mathbf{o}_{t-K:t})\right)$$

- $m$：feature scale 索引，multi-scale 提供 coarse-to-fine 的几何信息
- $\mathcal{E}_{\mathrm{VGGT}}^{(m)}$：VGGT 第 $m$ 层特征。VGGT 是个 visual geometry grounded transformer，通过 global frame attention 学到了很强的 static 3D prior
- $\mathcal{P}_{32 \times 32}$：adaptive average pooling 到 $32 \times 32$，和 query grid 对齐
- $W_m$：linear projection 到 512 维 temporal token space

第二步，queries 和 temporal tokens 在 TGE 里交互。TGE 有 12 个 attention block，每个 block 干两件事：

1. **Frame-wise spatial self-attention**：在每一帧内部，让 queries 和 patch tokens 做 self-attention。这一步是让 queries "看"到当前帧的几何结构
2. **Time-slot temporal attention**：在同一个 token slot 上，跨时间帧做 attention。这一步是让模型捕捉时序动态

直觉上，你可以把 queries 想象成 1024 个"探针"，它们先在空间维度上扫描每一帧的几何，然后在时间维度上追踪这些几何点的演化。最后，这些探针"吸收"了时空信息，变成了 prefix。

第三步，TGE 在最后一帧（$t$ 时刻）的输出 project 回 2048 维，append 到 multimodal prefix：

$$\mathbf{Z}_t^{\mathrm{GD}} = \mathrm{Proj}_{512 \to 2048}[\mathrm{TGE}(\mathrm{Proj}_{2048 \to 512}(\mathbf{Q}_{\mathrm{GD}}), \{\mathbf{P}_{t-K:t}^{(m)}\}_{m=1}^M)_t]$$

注意这个 prefix 是 2048 维的，和 PaliGemma/Gemma-2B 的 prefix space 对齐，所以可以无缝 insert 到 VLM 的 token sequence 里。

### 4.3 训练时：Current Reconstruction Branch

这部分强制 prefix 能被 decode 成一个 renderable 的 3D Gaussian scene。为什么要这么做？因为如果 prefix 只是个 free-floating 的 latent，模型完全可能学到一些无意义的 representation。通过强制它能 decode 成显式 3D Gaussians，并和真实 RGB/depth 对齐，就把 spatial structure 注入到 prefix 里了。

具体流程：

**Step 1**: 把 1024 个 prefix tokens reshape 成 $32 \times 32$ grid

**Step 2**: 用 Gaussian decoder backbone $\mathcal{B}_{\mathrm{G}}$ 上采样到 dense feature map：

$$\mathbf{F}_t^{\mathrm{G}} = \mathcal{B}_{\mathrm{G}}(\mathrm{Grid}(\mathbf{Z}_t^{\mathrm{GD}})), \quad \mathbf{F}_t^{\mathrm{G}} \in \mathbb{R}^{256 \times 256 \times 128}$$

这个 decoder 用 3 个 transposed-conv block（kernel 4, stride 2, padding 1），从 $32 \times 32$ 一路 upsample 到 $256 \times 256$。通道从 2048 → 512 → 256 → 128。每个 block 有 GroupNorm + GELU + 3×3 conv + bilinear residual skip。

**Step 3**: 两个 head 分别预测 geometry 和 appearance：

$$\mathbf{D}_t, \boldsymbol{\Theta}_t^{\mathrm{geo}} = \mathcal{H}_{\mathrm{geo}}(\mathbf{F}_t^{\mathrm{G}}), \quad \boldsymbol{\Theta}_t^{\mathrm{app}} = \mathcal{H}_{\mathrm{app}}(\mathbf{F}_t^{\mathrm{G}}, \mathbf{o}_t)$$

- **$\mathbf{D}_t$**：depth map，1 channel。用两层 3×3 conv + GroupNorm + GELU，最后 3×3 conv 出 1 channel
- **$\boldsymbol{\Theta}_t^{\mathrm{geo}}$**：8 channels = quaternion rotation (4) + scale (3) + opacity (1)。一个 3×3 conv 直接出 8 channel
- **$\boldsymbol{\Theta}_t^{\mathrm{app}}$**：9 channels = degree-1 spherical harmonics coefficients。先 7×7 conv fuse RGB $\mathbf{o}_t$，再 3×3 conv 出 9 channel。为什么要 fuse RGB？因为 appearance 强依赖于当前观察的颜色信息

**Step 4**: 用 depth unproject 成 Gaussian centers：

$$\mathcal{G}_t = \{(\boldsymbol{\mu}_i^t, \boldsymbol{\theta}_i^t)\}_{i=1}^{N_t}$$

其中 $N_t = 256 \times 256 = 65536$ 个 Gaussians。每个 Gaussian 有：
- $\boldsymbol{\mu}_i^t \in \mathbb{R}^3$：center，通过 depth + camera intrinsics back-project 得到
- $\boldsymbol{\theta}_i^t$：non-positional attributes（rotation, scale, opacity, SH）

这个 $\mathcal{G}_t$ 可以用 standard 3D Gaussian Splatting ([Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)) 渲染成 RGB 和 depth，然后和真实 observation 监督。

### 4.4 训练时：Future Prediction Branch

这是 GaussianDream 的另一个核心创新。只重建当前帧不够，还要让模型学会"预测"未来几帧的几何演化。为什么？因为机器人的 action 会改变环境，policy 需要隐式理解"我做了这个 action，环境会变成什么样"。

公式：

$$\boldsymbol{\nu}_t^{(\Delta)} = \mathcal{H}_{\mathrm{vel}}\left(\mathcal{B}_{\mathrm{pred}}(\mathbf{Z}_t^{\mathrm{GD}}), \mathbf{e}_\Delta\right), \quad \Delta\mathbf{X}_t^{(\Delta)} = \alpha_\Delta \boldsymbol{\nu}_t^{(\Delta)}$$

$$\hat{\boldsymbol{\mu}}_i^{t+\Delta} = \boldsymbol{\mu}_i^t + \Delta\mathbf{x}_i^{(\Delta)}$$

关键设计点：

**1. Horizon Embedding $\mathbf{e}_\Delta$**

这是一个 learnable embedding，告诉模型"你在预测 $t + \Delta$ 时刻的状态"。为什么需要这个？因为机器人的 motion 是 non-uniform 的。比如夹爪接近物体时运动慢，抓到后快速抬起。不同 horizon 的 dynamic mode 不一样，需要 horizon-aware 的预测。

**2. Temporal Scale Factor $\alpha_\Delta$**

这是一个标量，scale velocity。直觉上，预测 1 帧后的位移应该比预测 5 帧后的位移小。$\alpha_\Delta$ 让模型自动学习不同 horizon 的合适 scale。

**3. 只预测 Center 位移，Reuse 其他属性**

$$\hat{\mathcal{G}}_{t+\Delta} = \{(\hat{\boldsymbol{\mu}}_i^{t+\Delta}, \boldsymbol{\theta}_i^t)\}_{i=1}^{N_t}$$

未来 Gaussian state 的 rotation, scale, opacity, SH coefficients 全部 copy 自 current template，只更新 center。

这个设计非常聪明。为什么？
- **稳定性**：如果让模型同时预测所有属性，optimization 会非常不稳定
- **Short-horizon 假设**：t+1 到 t+5 这么短的时间窗口内，物体的 appearance 和 shape 基本不变，变的只是 position
- **降低 prediction 难度**：只预测 3-channel velocity map，比预测完整属性集简单一个数量级
- **聚焦 interaction-induced change**：robot interaction 主要引起几何位移，appearance 变化很少

**4. Velocity 经过 tanh**

$$\Delta\mathbf{X}_t^{(\Delta)} = \alpha_\Delta \tanh(\cdot)$$

tanh 把 velocity 限制在 $[-1, 1]$，然后用 $\alpha_\Delta$ scale。这避免了预测出极端位移，保证训练稳定。

### 4.5 监督信号从哪来

这里有个工程上的关键细节：**pseudo 3D scene flow**。机器人 demonstration 数据通常只有 RGB + action，没有 dense 3D geometry 标注。GaussianDream 怎么获得 depth 和 scene flow 监督？

**Depth**：用 [Depth Anything V2](https://arxiv.org/abs/2406.09414) 预测每帧的 dense depth。这是 pseudo ground truth，虽然不完美，但足够提供 metric geometry 约束。

**3D Scene Flow**：这是最巧妙的部分。用 [RAFT](https://arxiv.org/abs/2003.12039) 估计相邻帧的 2D optical flow $\mathbf{f}_{t \to t+1}$，然后结合两帧的 depth back-project 成 3D 点，计算 3D 位移：

$$\mathbf{x}_t = \Pi^{-1}(u, v, \mathbf{D}_t(u, v); \mathbf{K})$$

$$\mathbf{x}_{t+1} = \Pi^{-1}(u', v', \mathbf{D}_{t+1}(u', v'); \mathbf{K})$$

$$\mathbf{F}_{t \to t+1}^{3D}(u, v) = \mathbf{x}_{t+1} - \mathbf{x}_t$$

其中 $(u', v') = (u, v) + \mathbf{f}_{t \to t+1}(u, v)$ 是 warped pixel location。

这个 pseudo scene flow 直接监督 Gaussian center 的位移 $\Delta\mathbf{x}_i^{(\Delta)}$。validity mask 排除 warped 出 image boundary、depth invalid、超出 depth range 的 correspondence。

**Intuition**：把普通的 robot demonstration（只有 RGB + action label）转换成了 dense 的 spatial-temporal supervision。每个 pixel 都有了 depth + 3D flow 监督，而不是只有 sparse 的 action label。

### 4.6 两阶段训练

**Stage I: Pretraining**

只训练 reconstruction 和 prediction heads，不涉及 action learning。这一阶段让 prefix 学会 encoding 3D spatial structure 和 short-horizon dynamics。

Loss：

$$\mathcal{L}_{\mathrm{GD}} = \mathcal{L}_{\mathrm{cur}} + \mathcal{L}_{\mathrm{fut}}$$

其中：
- $\mathcal{L}_{\mathrm{cur}}$：当前帧 depth loss + RGB rendering loss
- $\mathcal{L}_{\mathrm{fut}}$：未来帧 depth loss + RGB rendering loss + scene flow loss，over horizon set $\mathcal{H} = \{1, 2, 3, 4, 5\}$

预测 horizon 逐渐扩大（curriculum learning），避免一开始就预测 long horizon 导致 instability。

**Stage II: Policy Learning**

联合训练 action policy 和 auxiliary Gaussian losses：

$$\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{GD}} \mathcal{L}_{\mathrm{GD}}$$

其中 $\mathcal{L}_{\mathrm{act}}$ 是基于 $\pi_{0.5}$ ([Physical Intelligence et al., 2025](https://arxiv.org/abs/2504.16054)) 的 flow-matching action loss：

$$\mathcal{L}_{\mathrm{act}} = \mathbb{E}_{\tau, \epsilon, \mathbf{a}_t^*}\left[\lVert \mathbf{v}_\theta(\tau\epsilon + (1-\tau)\mathbf{a}_t^*, \mathbf{c}_t, \tau) - (\epsilon - \mathbf{a}_t^*)\rVert_2^2\right]$$

- $\epsilon \sim \mathcal{N}(0, I)$：Gaussian noise
- $\tau$：flow-matching time，从 [0,1] 采样
- $\mathbf{v}_\theta$：policy 预测的 velocity field
- $\mathbf{c}_t = (\mathbf{o}_t, \mathbf{l}, \mathbf{s}_t; \mathbf{Z}_t^{\mathrm{GD}})$：conditioning context，**包含 GaussianDream prefix**

这一阶段，action loss 让 prefix 适应 executable control，auxiliary Gaussian loss 保持 prefix 的 spatial-temporal structure。两者 complementary。

### 4.7 推理时：Drop Everything

这就是 GaussianDream 最 elegant 的地方。推理的时候：

- **保留**：TGE Module + GaussianDream queries → 生成 $\mathbf{Z}_t^{\mathrm{GD}}$
- **丢弃**：Gaussian decoder $\mathcal{B}_{\mathrm{G}}$、reconstruction head $\mathcal{R}_\phi$、prediction head $\mathcal{D}_\psi$、所有 rendering 相关的 op

推理流程：
1. TGE Module 把 3 帧历史 observation 编码成 $\mathbf{Z}_t^{\mathrm{GD}}$（1024 个 prefix tokens）
2. Prefix tokens 和 image/language/state tokens 一起送入 VLM
3. VLM + action expert 用标准 flow-matching denoising 生成 action chunk
4. **没有** test-time Gaussian decoding
5. **没有** test-time rendering
6. **没有** test-time future rollout
7. **没有** additional planner

推理延迟：531 ms per action chunk，比 WAM baseline 的 700ms+ 快很多。

## 5 为什么这个设计 work

我从几个角度来 build intuition：

### 5.1 Prefix 是 3D 知识的"压缩包"

训练时，prefix 被强制要能 decode 成 3D Gaussians、能预测未来演化。这意味着 prefix 内部必须 encoding 了：
- 当前场景的 3D layout
- 物体的 spatial relation
- Short-horizon 的 dynamics

这些信息被压缩进 1024 个 token。推理时，即使不显式 decode，这些信息也已经"in"在 prefix 里了，action expert 可以直接利用。

这就像你学了物理，做题的时候不需要重新推导公式，直觉就告诉你答案。Prefix 就是那个"直觉"的载体。

### 5.2 Reconstruction 和 Prediction 是 Complementary 的

消融实验很说明问题：

| Current Recon | Future Pred | Rendering | Depth | LIBERO Avg |
|---|---|---|---|---|
| √ | × | × | × | 97.0 |
| √ | × | √ | √ | 97.3 |
| √ | √ | × | √ | 97.5 |
| √ | √ | √ | × | 97.2 |
| √ | √ | √ | √ | **98.4** |

- 只有 reconstruction：97.0%。说明 spatial grounding 本身就很有用
- 加上 rendering + depth：97.3%。dense pixel-level supervision 有帮助
- 加上 future prediction：97.5%。short-horizon dynamics 提供 additional signal
- 去掉 depth：97.2%。RGB alone 无法约束 metric geometry，depth 关键
- 全加上：98.4%。所有组件 complementary

**Intuition**：reconstruction 提供一个"static template"——告诉模型"世界长这样"。prediction 提供"dynamic update"——告诉模型"世界会怎么变"。两者结合，prefix 既知道当前 state，又知道 state 如何演化，这正是 precise manipulation 需要的。

### 5.3 为什么 Reuse Non-Positional Attributes

这是个反直觉但很聪明的设计。为什么不预测未来帧的 rotation、scale、opacity、SH？

想象一下：夹爪抓起杯子，从 $t$ 到 $t+5$，杯子的 position 变了（被举起来了），但 cup 的 shape、color、orientation 基本没变。预测这些属性的 delta 是 wasted capacity，而且增加 optimization 难度。

只预测 center displacement，相当于告诉模型"focus on what changes"——即 interaction-induced geometric change。这和 robot manipulation 的本质吻合：robot 主要引起物体位移，而非物体形变。

### 5.4 为什么用 VGGT 而非其他 backbone

[VGGT](https://arxiv.org/abs/2406.07751) 是个 visual geometry grounded transformer，通过 global frame attention 学到了很强的 static 3D prior。但它的 temporal dynamics 监督弱，因为它的设计目标是 single-view 或 multi-view 的 static reconstruction。

GaussianDream 用 learnable queries 来弥补这个缺陷。Queries 在 TGE Module 里通过 temporal attention 跨帧交互，学到 dynamic 信息。所以最终 prefix = VGGT 的 static 3D prior + TGE 学到的 temporal dynamics。这个组合很 powerful。

### 5.5 为什么 Pseudo Supervision 就够了

你可能会问：Depth Anything V2 预测的 depth 和 RAFT 预测的 flow 都不完美，这样的 pseudo supervision 真的够吗？

答案是：**够，因为目标是让 prefix encoding 3D structure，而非精确重建**。Pseudo depth 提供 metric scale 的 geometric cue，即使绝对值不完美，relative depth relation 是准确的。Pseudo scene flow 提供 motion direction 和 magnitude 的 cue，即使 noisy，average 之后的 signal-to-noise ratio 足够指导 representation learning。

而且，最终推理时根本不用这些 pseudo signal，它们只是 training-time 的 auxiliary supervision。只要它们能让 prefix 学到 useful 的 3D-aware representation，目的就达到了。

## 6 实验结果说明了什么

### 6.1 LIBERO 98.4%

| Method | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| π0.5 | 97.8 | 98.8 | 97.6 | 92.4 | 96.7 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| **GaussianDream** | **99.0** | 99.6 | **99.0** | 96.0 | **98.4** |

GaussianDream 在 Spatial 和 Goal 任务上 best。这完全符合预期：
- **Spatial 任务**需要精确的 3D grounding，GaussianDream 的 reconstruction branch 直接优化这个
- **Goal 任务**需要理解 goal state 和当前 state 的差异，future prediction branch 帮助模型"看到"未来应该变成什么样

LingBot-VA 平均分 98.5%，比 GaussianDream 高 0.1%，但它用了 larger autoregressive video-action pipeline，推理开销大得多。GaussianDream 用 prefix-based inference，效率高。

### 6.2 RoboCasa Human-50 54.8%

| Method | Pick&Place | Doors/Drawers | Others | Average |
|---|---|---|---|---|
| π0.5 | 36.0 | 46.5 | 39.5 | 40.1 |
| GeoPredict | 22.7 | 75.1 | 62.4 | 52.4 |
| Being-H0.5 | 36.0 | 71.7 | 57.6 | 53.9 |
| **GaussianDream** | **43.8** | 66.3 | 54.4 | **54.8** |

GaussianDream 在 Pick&Place 上优势最大（43.8% vs 第二名 36.0%）。Pick&Place 是最需要 spatially precise localization 的任务，这说明 reconstruction + prediction 对 localization-sensitive manipulation 帮助最大。

有趣的是，GeoPredict 在 Doors/Drawers 上更强（75.1%）。这可能是因为 Doors/Drawers 更多依赖 articulated object 的 kinematic structure，GeoPredict 用 future keypoint trajectory 可能更适配。

### 6.3 Real Robot 50.0%

| Method | Scene-A | Scene-B | Scene-C | Scene-D | Average |
|---|---|---|---|---|---|
| π0.5 | 42.5 | 50.0 | 25.0 | 20.0 | 34.4 |
| **GaussianDream** | 55.0 | 70.0 | 35.0 | 40.0 | **50.0** |

从 34.4% 到 50.0%，提升 15.6 个百分点。最大提升在 spatial-relation（Scene-B: 50.0 → 70.0）和 long-horizon（Scene-D: 20.0 → 40.0）。

这验证了 GaussianDream 在 real-world 的 sim-to-real transfer 能力。Pseudo depth 和 pseudo flow 虽然在 simulation 里生成，但学到的 representation 在 real robot 上依然 effective，因为 Depth Anything V2 和 RAFT 在 real image 上也能 work。

### 6.4 推理效率

从 Figure 8：
- **GaussianDream (asymmetric)**：531 ms per action chunk
- **GaussianDream (with decoder)**：569 ms per action chunk
- **WAM baseline**：700+ ms per action chunk

Asymmetric 设计只比 with-decoder 版本快 38 ms，这是因为 decoder 在 inference 时根本不跑，只多了 prefix tokens 的 attention 计算。但比 WAM 快 170 ms+，这个差距在 high-frequency control loop 里很显著。

而且，GaussianDream 的 trajectory 更 smooth（Figure 8 left），说明 prefix 提供的 spatial context 让 action generation 更稳定。

## 7 跟相关方法的关系

### 7.1 vs. GeoPredict

[GeoPredict](https://arxiv.org/abs/2512.16811) 也是预测未来几何，但用 keypoint trajectory + depth-supervised Gaussian。区别在于：
- GeoPredict 预测 sparse keypoints，GaussianDream 预测 dense 65536 个 Gaussians
- GeoPredict 在推理时可能保留 prediction head，GaussianDream 推理时全丢
- GeoPredict 在 Doors/Drawers 上更强，GaussianDream 在 Pick&Place 上更强

### 7.2 vs. GWM

[GWM](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.html) 也用 Gaussian world model，但推理时需要 iterative optimization。GaussianDream 的 prefix-based inference 避免了这个开销。

### 7.3 vs. LingBot-VA

[LingBot-VA](https://arxiv.org/abs/2601.21998) 用 autoregressive video-action pipeline，性能略高（98.5% vs 98.4%），但推理慢。GaussianDream 用 prefix-based inference，效率和性能的 trade-off 更好。

### 7.4 vs. Spatial Forcing

[Spatial Forcing](https://arxiv.org/abs/2510.12276) 用 implicit geometry alignment，没有显式 3D reconstruction。GaussianDream 的显式 reconstruction 提供更强的 spatial grounding。

## 8 我对这个工作的评价

### 8.1 优点

1. **Asymmetric design 非常 elegant**：训练时 expressive 3D supervision，推理时 lightweight prefix。这解决了 3D expressiveness vs. inference efficiency 的 tension
2. **Pseudo supervision 巧妙**：用 Depth Anything V2 + RAFT 把普通 demonstration 转成 dense 3D supervision，无需额外标注
3. **Future prediction 设计合理**：只预测 center displacement，reuse 其他属性，既稳定又聚焦
4. **VGGT + learnable queries 组合**：利用 VGGT 的 static 3D prior，用 queries 补充 temporal dynamics
5. **实验充分**：LIBERO、RoboCasa、real robot 都有，消融实验清楚

### 8.2 潜在局限

1. **Pseudo supervision 质量上限**：Depth Anything V2 和 RAFT 的误差会传导，虽然实验证明够用，但在极端场景可能 fail
2. **Short-horizon 限制**：只预测 t+1 到 t+5，无法建模 long-horizon dynamics。Long-horizon planning 仍需其他机制
3. **Static appearance 假设**：reuse non-positional attributes 对 deformable object（毛巾、衣物）可能不适用
4. **Single-view reconstruction**：虽然用 multi-view observation，但 reconstruction 是 per-frame 的，没有真正的 multi-view stereo fusion
5. **3D Gaussian 数量固定**：65536 个 Gaussians，对简单场景冗余，对复杂场景可能不够

### 8.3 未来方向

我能想到几个可能的 extension：

1. **Adaptive Gaussian number**：用 pruning + densification 让 Gaussian 数量自适应
2. **Deformable Gaussian**：引入 [4D Gaussian Splatting](https://arxiv.org/abs/2403.11154) 建模 non-rigid deformation
3. **Longer horizon**：用 hierarchical prediction，先预测 coarse long-horizon，再 refine short-horizon
4. **Active perception**：结合 next-best-view，让 robot 主动收集信息补全 3D
5. **Multi-view fusion**：在 TGE 里做真正的 multi-view stereo，而非 per-frame reconstruction
6. **Force-aware**：结合 [ForceVLA](https://arxiv.org/abs/2506.07967) 引入 force/tactile signal

## 9 一句话总结

GaussianDream 的核心 insight 是：**让模型在训练时"想象"3D 世界怎么重建、怎么演化，把这些想象压缩成一个 prefix；推理时只保留 prefix 指导 action generation，丢掉所有"想象器"**。这巧妙地把 3D world model 的 expressiveness 和 VLA 的 inference efficiency 结合起来，在 LIBERO、RoboCasa、real robot 上都取得了 competitive 的结果。

这让我想起一句话：**"The best way to predict the future is to have already imagined it."** GaussianDream 就是让 robot policy 在训练时已经"想象过"未来，推理时只需调用那个"想象的记忆"。

**代码**：[https://github.com/TuojingAI/GaussianDream](https://github.com/TuojingAI/GaussianDream)

**相关链接**：
- [VGGT](https://arxiv.org/abs/2406.07751)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [RAFT](https://arxiv.org/abs/2003.12039)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [RoboCasa](https://arxiv.org/abs/2406.02523)

---

# GaussianDream: Feed-Forward 3D Gaussian World Model 深度解析

## 1. 论文背景与动机

### 1.1 当前VLA范式的三大瓶颈

GaussianDream这篇论文瞄准了当前Vision-Language-Action (VLA)模型在精确机器人操作中的三个根本性缺陷。让我详细分析这些瓶颈:

**瓶颈一:Spatial-Geometric Underspecification**

预训练的VLMs主要在2D pixel grid上运行,3D空间结构和contact constraints只能隐式编码在visual latents和action labels中。这导致控制环路容易受到subtle geometric execution errors的影响,比如grasp point偏移。论文中提到了几篇相关工作都指出了这个问题 [Sun et al., 2025](https://arxiv.org/abs/2508.09071); [Li et al., 2025a](https://arxiv.org/abs/2510.12276)。

**瓶颈二:Dense Visual-Spatial Supervision的浪费**

机器人轨迹记录了丰富的object layouts、appearance cues和depth structures,但标准action-imitation objectives主要监督每个时间步的immediate control command,大量dense pixel-level geometric signals未被充分利用 [Ye et al., 2026](https://arxiv.org/abs/2602.15922); [Kim et al., 2026](https://arxiv.org/abs/2601.16163)。

**瓶颈三:缺乏显式的未来环境预测**

标准VLAs只能从action labels隐式获取环境动力学,缺少显式的internal environment emulation和future-state supervision机制。这限制了在执行结果依赖short-horizon state changes的复杂场景中的泛化能力。

### 1.2 现有解决方案的不足

论文指出当前两类改进方法都存在不足:

**3D-Enhanced Policies** ([Deng et al., 2025](https://arxiv.org/abs/2512.21970); [Zhou et al., 2025](https://arxiv.org/abs/2511.17199); [Ni et al., 2025](https://arxiv.org/abs/2512.00903)):
- 主要使用point clouds或depth maps静态锚定当前场景
- 缺少动态未来环境emulation机制

**Robotic World Models** ([Ye et al., 2026](https://arxiv.org/abs/2602.15922); [Li et al., 2026a](https://arxiv.org/abs/2601.21998); [Bi et al., 2025](https://arxiv.org/abs/2512.13030)):
- 在pixel、latent或action space预测未来
- 通常需要iterative voxel optimization或heavy visual autoregressive rollouts
- 推理时计算开销大,难以部署于高频机器人控制环路

## 2. GaussianDream核心创新

### 2.1 核心思想:非对称训练-推理架构

GaussianDream的关键insight在于将3D Gaussian world model作为training-time plug-in,而非inference-time generator。这通过一个巧妙的非对称设计实现:

- **训练时**:完整的3D Gaussian reconstruction + future prediction,通过dense RGB/depth/scene-flow监督
- **推理时**:丢弃所有auxiliary decoding heads,只保留compact prefix conditioning action generation

这种设计避开了test-time Gaussian decoding、geometric rendering、video rollout等昂贵操作。

### 2.2 三大创新贡献

1. **Unified 3D Gaussian World Model**:首次将language-conditioned VLA policies与structured 3D Gaussian representations统一集成
2. **Spatio-Temporal Representation + Dense Supervision**:在一个设计中同时解决3D空间grounding、dense pixel-level监督、short-horizon predictive emulation
3. **Efficient Asymmetric Plug-in**:训练时用完整Gaussian reconstruction/prediction,部署时丢弃auxiliary heads避免test-time开销

## 3. 架构详解

### 3.1 整体数据流

让我解析整个pipeline的数学形式:

**输入定义**:
- $\mathbf{o}_t$: 当前多视角observation
- $\mathbf{o}_{t-K:t}$: 短时序历史窗口(实践中用{t-10, t-5, t}三帧)
- $\mathbf{l}$: language instruction  
- $\mathbf{s}_t$: robot state
- $\mathbf{a}_t$: predicted action chunk

**GaussianDream Prefix构建**:

$$\mathbf{Z}_t^{\mathrm{GD}} = \mathcal{F}_\omega(\mathbf{o}_{t-K:t}, \mathbf{Q}_{\mathrm{GD}})$$

其中:
- $\mathbf{Q}_{\mathrm{GD}}$: 可学习的GaussianDream queries
- $\mathcal{F}_\omega$: 由Temporal Gaussian Evolution (TGE) module实现
- $\mathbf{Z}_t^{\mathrm{GD}}$: 1024个tokens,2048维(与PaliGemma/Gemma-2B prefix space对齐)

**训练时双头解码**:

$$\mathcal{G}_t = \mathcal{R}_\phi(\mathbf{Z}_t^{\mathrm{GD}}, \mathbf{o}_t), \quad \hat{\mathcal{G}}_{t+\Delta} = \mathcal{D}_\psi(\mathcal{G}_t, \mathbf{Z}_t^{\mathrm{GD}}, \Delta)$$

其中:
- $\mathcal{R}_\phi$: 重建解码器,生成当前Gaussian state $\mathcal{G}_t$
- $\mathcal{D}_\psi$: 预测解码器,生成horizon-conditioned未来Gaussian state $\hat{\mathcal{G}}_{t+\Delta}$
- $\Delta$: 预测horizon

**推理时动作生成**:

$$\mathbf{a}_t = \pi_\theta(\mathbf{o}_t, \mathbf{l}, \mathbf{s}_t; \mathbf{Z}_t^{\mathrm{GD}})$$

这里的关键insight是:训练时将reconstruction和prediction supervision转移到inference-time prefix,而不是用test-time simulator替换policy。

### 3.2 Current Gaussian Reconstruction Branch

这个分支强制GaussianDream prefix表示可渲染的3D场景状态,提供explicit 3D spatial grounding。

**Step 1: Latent Grid Reshaping**

将1024个GaussianDream tokens reshape为$32 \times 32$ latent grid,保留空间局部性和dense geometric structure。

**Step 2: Gaussian Decoder Upsampling**

$$\mathbf{F}_t^{\mathrm{G}} = \mathcal{B}_{\mathrm{G}}(\mathrm{Grid}(\mathbf{Z}_t^{\mathrm{GD}})), \quad \mathbf{F}_t^{\mathrm{G}} \in \mathbb{R}^{256 \times 256 \times 128}$$

- $\mathcal{B}_{\mathrm{G}}$: Gaussian decoder backbone
- 输出$256 \times 256$ spatial resolution对应dense Gaussian layout
- 128维channels编码shared geometric和appearance信息

**Step 3: Gaussian属性预测**

geometry head和appearance head分别预测:

$$\mathbf{D}_t, \boldsymbol{\Theta}_t^{\mathrm{geo}} = \mathcal{H}_{\mathrm{geo}}(\mathbf{F}_t^{\mathrm{G}}), \quad \boldsymbol{\Theta}_t^{\mathrm{app}} = \mathcal{H}_{\mathrm{app}}(\mathbf{F}_t^{\mathrm{G}}, \mathbf{o}_t)$$

其中:
- $\mathbf{D}_t$: depth map
- $\boldsymbol{\Theta}_t^{\mathrm{geo}}$: geometry attributes = (quaternion rotation (4), scale (3), opacity (1)),共8 channels
- $\boldsymbol{\Theta}_t^{\mathrm{app}}$: degree-1 spherical harmonics coefficients,共9 channels

**Step 4: Gaussian State构建**

$$\mathcal{G}_t = \boldsymbol{\mathcal{A}}(\mathbf{D}_t, \boldsymbol{\Theta}_t) = \{(\boldsymbol{\mu}_i^t, \boldsymbol{\theta}_i^t)\}_{i=1}^{N_t}$$

其中:
- $\boldsymbol{\mu}_i^t$: 第i个Gaussian的center(通过depth unprojection得到)
- $\boldsymbol{\theta}_i^t$: 第i个Gaussian的non-positional attributes
- $N_t = 256 \times 256 = 65536$: Gaussian总数

这个分支的intuition:通过强制prefix重建显式3D geometry,inject dense spatial structure到policy representation中,同时为future evolution prediction提供static Gaussian template。

### 3.3 Future Gaussian Prediction Branch

这个分支让模型学习short-horizon Gaussian evolution,捕捉interaction-induced geometric changes。

**VGGT特征提取**:

$$\mathbf{P}_{t-K:t}^{(m)} = W_m \mathcal{P}_{32 \times 32}\left(\mathcal{E}_{\mathrm{VGGT}}^{(m)}(\mathbf{o}_{t-K:t})\right)$$

其中:
- $m$: feature scale索引
- $\mathcal{E}_{\mathrm{VGGT}}^{(m)}$: VGGT ([Wang et al., 2025](https://arxiv.org/abs/2406.07751))的第m层特征
- $\mathcal{P}_{32 \times 32}$: adaptive average pooling到32×32
- $W_m$: linear projection到temporal token space

VGGT提供强static 3D priors但temporal interaction dynamics监督较弱,这就是为什么需要learnable GaussianDream queries。

**TGE Module处理**:

$$\mathbf{Z}_t^{\mathrm{GD}} = \mathrm{Proj}_{512 \to 2048}[\mathrm{TGE}(\mathrm{Proj}_{2048 \to 512}(\mathbf{Q}_{\mathrm{GD}}), \{\mathbf{P}_{t-K:t}^{(m)}\}_{m=1}^M)_t]$$

TGE Module的architecture:
- 12个attention blocks,8 heads
- 每个block交替:
  - Frame-wise spatial self-attention (query-patch tokens交互)
  - Time-slot temporal attention (相同token slot跨frames)
- 4× expansion MLP

**Horizon-Conditioned Future Prediction**:

$$\boldsymbol{\nu}_t^{(\Delta)} = \mathcal{H}_{\mathrm{vel}}\left(\mathcal{B}_{\mathrm{pred}}(\mathbf{Z}_t^{\mathrm{GD}}), \mathbf{e}_\Delta\right), \quad \Delta\mathbf{X}_t^{(\Delta)} = \alpha_\Delta \boldsymbol{\nu}_t^{(\Delta)}$$

$$\hat{\boldsymbol{\mu}}_i^{t+\Delta} = \boldsymbol{\mu}_i^t + \Delta\mathbf{x}_i^{(\Delta)}, \quad \hat{\mathcal{G}}_{t+\Delta} = \{(\hat{\boldsymbol{\mu}}_i^{t+\Delta}, \boldsymbol{\theta}_i^t)\}_{i=1}^{N_t}$$

关键设计要点:
- $\mathbf{e}_\Delta$: learnable horizon embedding,让模型区分不同dynamic modes across prediction horizons
- $\alpha_\Delta$: temporal scale factor
- $\boldsymbol{\nu}_t^{(\Delta)}$: 3-channel center velocity map(经过tanh)
- **关键**:未来状态reuse non-positional attributes $\boldsymbol{\theta}_i^t$,只预测中心位移

这个设计的intuition:robot interactions呈现non-uniform motion patterns和time-dependent uncertainty,horizon embedding帮助模型区分不同时间尺度的动态;只预测位移避免了重新预测所有属性的不稳定性。

### 3.4 Pseudo 3D Scene Flow构建

这是一个关键的技术细节,让我详细解析:

**Step 1: 2D Optical Flow Estimation**

用RAFT ([Teed & Deng, 2020](https://arxiv.org/abs/2003.12039))估计相邻帧$\mathbf{o}_t, \mathbf{o}_{t+1}$之间的dense 2D optical flow $\mathbf{f}_{t \to t+1}$:

$$(u', v') = (u, v) + \mathbf{f}_{t \to t+1}(u, v)$$

**Step 2: 3D Back-projection**

在$(u', v')$处bilinearly sample未来depth map,然后用camera intrinsics $\mathbf{K}$ back-project:

$$\mathbf{x}_t = \Pi^{-1}(u, v, \mathbf{D}_t(u, v); \mathbf{K}), \quad \mathbf{x}_{t+1} = \Pi^{-1}(u', v', \mathbf{D}_{t+1}(u', v'); \mathbf{K})$$

**Step 3: 3D Scene Flow计算**

$$\mathbf{F}_{t \to t+1}^{3D}(u, v) = \mathbf{x}_{t+1} - \mathbf{x}_t$$

**Step 4: Validity Mask**

排除以下correspondences:
- warped出image boundary
- invalid depth
- 超出configured depth range

这个pseudo监督的intuition:将普通robot demonstrations转换为dense spatial-temporal supervision,无需额外标注。

## 4. 训练策略详解

### 4.1 两阶段训练

**Stage I: GaussianDream Pretraining**

只训练reconstruction和prediction heads,不涉及action learning:

$$\mathcal{L}_{\mathrm{GD}} = \underbrace{\lambda_{\mathrm{curt}}^{\mathrm{depth}} \mathcal{L}_{\mathrm{curt}}^{\mathrm{depth}} + \lambda_{\mathrm{curt}}^{\mathrm{render}} \mathcal{L}_{\mathrm{curt}}^{\mathrm{render}}}_{\mathcal{L}_{\mathrm{cur}}} + \underbrace{\sum_{\Delta \in \mathcal{H}} w_\Delta\left(\lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}}^{(\Delta)} + \lambda_{\mathrm{render}} \mathcal{L}_{\mathrm{render}}^{(\Delta)} + \lambda_{\mathrm{flow}} \mathcal{L}_{\mathrm{flow}}^{(\Delta)}\right)}_{\mathcal{L}_{\mathrm{fut}}}$$

其中:
- $\mathcal{L}_{\mathrm{cur}}$: 当前帧depth + rendering重建损失
- $\mathcal{L}_{\mathrm{fut}}$: 未来帧depth + rendering + scene-flow预测损失
- $\mathcal{H}$: 预测horizon set(实践为t+1到t+5)
- $w_\Delta$: horizon-specific loss weight

预测horizon逐渐扩大以保证stable optimization。

**Stage II: GaussianDream-Conditioned Policy Learning**

联合训练policy和auxiliary Gaussian losses:

$$\mathcal{L}_{\mathrm{act}} = \mathbb{E}_{\tau, \epsilon, \mathbf{a}_t^*}\left[\lVert \mathbf{v}_\theta(\tau\epsilon + (1-\tau)\mathbf{a}_t^*, \mathbf{c}_t, \tau) - (\epsilon - \mathbf{a}_t^*)\rVert_2^2\right]$$

这是基于$\pi_{0.5}$ ([Physical Intelligence et al., 2025](https://arxiv.org/abs/2504.16054))的flow-matching action loss,其中:
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\tau$: flow-matching time,从flow-matching time distribution采样
- $\mathbf{v}_\theta$: policy预测的velocity field
- $\mathbf{c}_t = (\mathbf{o}_t, \mathbf{l}, \mathbf{s}_t; \mathbf{Z}_t^{\mathrm{GD}})$: action-conditioning context

**联合优化目标**:

$$\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{GD}} \mathcal{L}_{\mathrm{GD}}$$

action loss让prefix适应executable control,auxiliary Gaussian losses保持prefix的spatial-temporal structure。

### 4.2 Flow Loss详解

带validity mask的flow loss:

$$\mathcal{L}_{\mathrm{flow}}^{(\Delta)} = \frac{\sum_i \mathbf{M}_i^{(\Delta)} \lVert \Delta\mathbf{x}_i^{(\Delta)} - \mathbf{F}_i^{3D,(\Delta)}\rVert_1}{\sum_i \mathbf{M}_i^{(\Delta)} + \epsilon}$$

其中:
- $\mathbf{M}_i^{(\Delta)}$: validity mask
- $\mathbf{F}_i^{3D,(\Delta)}$: pseudo 3D scene-flow target在对应Gaussian/pixel location的采样
- $\epsilon$: 防止除零的小常数

## 5. 实验结果分析

### 5.1 LIBERO Benchmark结果

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | 97.8 | 98.8 | 97.6 | 92.4 | 96.7 |
| GeoPredict | 98.0 | 98.2 | 95.7 | 94.0 | 96.5 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| Spatial Forcing (PyTorch) | 98.6 | 98.4 | 98.2 | 95.4 | 97.6 |
| **GaussianDream** | **99.0** | 99.6 | **99.0** | 96.0 | **98.4** |

GaussianDream在Spatial和Goal任务上取得最佳成绩,这两个任务对spatial grounding和goal-conditioned execution要求最高。LingBot-VA平均分最高但使用更大的autoregressive video-action pipeline。

### 5.2 RoboCasa Human-50结果

| Method | Pick&Place | Doors/Drawers | Others | Average |
|--------|------------|---------------|--------|---------|
| π0 | 14.0 | 53.1 | 58.5 | 42.4 |
| π0.5 | 36.0 | 46.5 | 39.5 | 40.1 |
| GWM | 14.8 | 54.3 | 49.8 | 39.3 |
| GeoPredict | 22.7 | 75.1 | 62.4 | 52.4 |
| Being-H0.5 | 36.0 | 71.7 | 57.6 | 53.9 |
| **GaussianDream** | **43.8** | 66.3 | 54.4 | **54.8** |

GaussianDream在spatially precise pick-and-place任务上优势最明显,说明reconstruction + prediction提升localization-sensitive manipulation。

### 5.3 Real-Robot结果

| Method | Scene-A | Scene-B | Scene-C | Scene-D | Average |
|--------|---------|---------|---------|---------|---------|
| π0.5 | 42.5 | 50.0 | 25.0 | 20.0 | 34.4 |
| **GaussianDream** | 55.0 | 70.0 | 35.0 | 40.0 | **50.0** |

在physical execution中,GaussianDream将成功率从34.4%提升到50.0%,最大提升出现在spatial-relation和long-horizon场景。

### 5.4 推理效率分析

从论文Figure 8的数据:
- GaussianDream (asymmetric): 531 ms per action chunk
- GaussianDream (with decoder): 569 ms per action chunk  
- WAM/World Action Model baseline: >700 ms per action chunk

这验证了非对称设计的效率优势,即使保留decoder也只增加38ms,但比WAM快约170ms+。

## 6. 消融研究深度解析

### 6.1 Component Analysis

| Current Recon | Future Pred | Rendering | Depth | LIBERO Avg |
|---------------|-------------|-----------|-------|-------------|
| √ | × | × | × | 97.0 |
| √ | × | √ | √ | 97.3 |
| √ | √ | × | √ | 97.5 |
| √ | √ | √ | × | 97.2 |
| √ | √ | √ | √ | **98.4** |

关键发现:

1. **Current Reconstruction Alone (97.0%)**: 说明将observation重建为Gaussian state提供了强spatial prior

2. **Adding Rendering+Depth (97.3%)**: 渲染损失帮助align predicted Gaussians与observations,提升0.3%

3. **Adding Future Prediction (97.5%)**: short-horizon state-change supervision在current-frame grounding基础上额外贡献,提升0.5%

4. **Removing Depth with Rendering+Future (97.2%)**: RGB consistency alone无法完全约束metric geometry,depth supervision关键

5. **Full Model (98.4%)**: 所有组件complementary,达到最佳

### 6.2 关键Insight

消融研究揭示了一个重要的insight:reconstruction提供spatial template,prediction学习interaction-induced changes,两者complementary。Rendering提供image-level consistency,depth提供metric geometry约束,两者也是complementary的。

## 7. 技术细节深挖

### 7.1 Architecture Implementation Details

**GaussianDream Token Path**:
- Dense 32×32 query grid → 1024 learnable queries
- 每个query先project到512 channels
- TGE Module: 12 blocks, 8 heads, 4× expansion MLP
- VGGT features: 32×32 adaptive average pooling + linear projection to 512 channels
- Output: project回2048维,append到multimodal prefix

**Gaussian Decoder**:
- Reshape $\mathbf{Z}_t^{\mathrm{GD}}$到32×32 token grid
- 3个transposed-conv upsampling blocks (kernel 4, stride 2, padding 1)
- 通道:VLM width → 512 → 256 → 128
- 每个block: GroupNorm + GELU + 3×3 conv + bilinear residual skip
- DPT-style feature fusion blocks with 3×3和1×1 convs

**Current Reconstruction Head**:
- Geometry branch: 3×3 conv预测8 channels (quaternion 4 + scale 3 + opacity 1)
- Depth branch: 2层3×3 convs with GroupNorm+GELU + final 3×3 conv
- Appearance branch: 7×7 conv fuse RGB + 3×3 conv预测9 SH channels

**Future Prediction Head**:
- 3×3 projection to 128 channels + horizon embedding
- Residual block: 2×3×3 convs with GroupNorm
- Velocity head: 3×3 conv + GroupNorm + GELU + 1×1 conv → 3-channel
- tanh + horizon scale → update only centers

### 7.2 训练超参数

- 60K optimization steps
- Global batch size: 24
- Optimizer: AdamW
- Peak learning rate: $5 \times 10^{-5}$
- Schedule: cosine with 10K warmup steps
- Gradient clipping: max norm 1.0
- EMA decay: 0.999
- Hardware: NVIDIA A100 GPUs, mixed-precision

### 7.3 Pseudo Depth Generation

使用Depth Anything V2 ([Yang et al., 2024](https://arxiv.org/abs/2406.09414))生成dense pseudo depth maps,resize回原始image resolution,与episode data一起存储。这些pseudo targets只在training时使用,允许将普通demonstration sequences转换为dense spatial-temporal supervision。

## 8. 与相关工作的对比

### 8.1 3D-Enhanced VLA方法对比

| Method | 3D Representation | Future Prediction | Inference Cost |
|--------|------------------|-------------------|----------------|
| GeoVLA | depth maps | × | low |
| StereoVLA | stereo cues | × | low |
| VLA-4D | 4D features | × | medium |
| SwiftVLA | spatiotemporal | × | low |
| QDepth-VLA | quantized depth | × | low |
| Spatial Forcing | implicit geometry | × | low |
| GeoPredict | keypoints + Gaussian | √ (future keypoints) | medium |
| **GaussianDream** | **full 3D Gaussians** | **√ (Gaussian evolution)** | **low (asymmetric)** |

### 8.2 World Model方法对比

| Method | Prediction Space | Inference Rollout | 3D Structure |
|--------|-----------------|------------------|--------------|
| DreamZero | latent | autoregressive | × |
| Cosmos Policy | video latents | autoregressive | × |
| Motus | latent actions | autoregressive | × |
| WorldVLA | actions | autoregressive | × |
| GWM | Gaussians | iterative | √ |
| ManiGaussian | Gaussians | iterative | √ |
| **GaussianDream** | **Gaussians** | **× (training-only)** | **√** |

GaussianDream的独特之处在于:training时用full Gaussian reconstruction + prediction,但inference时只保留prefix,避免了GWM和ManiGaussian的iterative optimization开销。

## 9. 设计Intuition总结

### 9.1 为什么用3D Gaussian Splatting?

1. **可微渲染**:支持end-to-end训练,可以用RGB rendering loss监督
2. **密集表示**:256×256 = 65536个Gaussians,比keypoints密集得多
3. **自然建模运动**:通过中心位移即可建模动态变化
4. **完整属性集**:rotation, scale, opacity, SH coefficients完整描述场景
5. **高效渲染**:比voxel representation更高效

### 9.2 为什么用Prefix而非Test-Time Generation?

1. **推理效率**:一个forward pass即可,无需iterative optimization
2. **知识压缩**:训练时密集监督压缩到紧凑token序列
3. **保留VLA接口**:不改变base policy的action生成流程
4. **避免误差累积**:test-time generation会有rollout误差

### 9.3 为什么需要Future Prediction?

1. **理解交互后果**:模型学习action如何改变环境
2. **Short-horizon planning**:预测t+1到t+5的Gaussian evolution
3. **Horizon-aware dynamics**:不同时间尺度有不同dynamic modes
4. **Complementary to reconstruction**:static template + dynamic changes

### 9.4 为什么Reuse Non-Positional Attributes?

未来Gaussian state只预测中心位移,rotation/scale/opacity/SH coefficients直接copy from current template。这个设计的intuition:
1. **稳定性**:避免重新预测所有属性的不稳定性
2. **聚焦变化**:只预测interaction-induced geometric changes
3. **短horizon假设**:short-horizon内appearance变化不大
4. **降低预测难度**:3-channel velocity map比完整属性预测简单

## 10. 局限性与未来方向

基于论文内容,我可以推测几个潜在局限:

1. **Pseudo监督质量**:依赖Depth Anything V2和RAFT的预测质量
2. **Short-horizon限制**:只预测t+1到t+5,无法建模long-horizon dynamics
3. **Static Appearance假设**:reuse attributes可能不适合deformable objects
4. **Single-view reconstruction**:虽然用multi-view observation,但reconstruction仍是per-frame的

未来可能方向:
1. 扩展到longer horizons
2. 引入deformable Gaussian representations
3. 结合主动感知
4. 扩展到mobile manipulation

## 11. 代码与资源

- **官方代码**: [https://github.com/TuojingAI/GaussianDream](https://github.com/TuojingAI/GaussianDream)
- **VGGT**: [Visual Geometry Grounded Transformer](https://arxiv.org/abs/2406.07751)
- **Depth Anything V2**: [https://arxiv.org/abs/2406.09414](https://arxiv.org/abs/2406.09414)
- **RAFT**: [Recurrent All-Pairs Field Transforms](https://arxiv.org/abs/2003.12039)
- **3D Gaussian Splatting**: [原始论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **π0.5**: [Physical Intelligence](https://arxiv.org/abs/2504.16054)
- **LIBERO**: [Benchmark](https://arxiv.org/abs/2306.03310)
- **RoboCasa**: [Large-scale simulation](https://arxiv.org/abs/2406.02523)

## 12. 总结

GaussianDream的核心贡献在于巧妙地将3D Gaussian world model的expressiveness与VLA的高效推理结合。通过非对称训练-推理架构,训练时获得dense 3D spatial-temporal supervision,推理时只保留compact prefix。这种方法在多个benchmark上取得SOTA或competitive performance,同时保持531ms的inference latency,适合real-time closed-loop control。

论文的intuition可以概括为:**让模型在训练时"想象"3D世界如何重建和演化,但推理时只需"记住"这种想象的compact representation来指导动作生成**。这巧妙地解决了3D-awareness与inference efficiency之间的tension,为未来VLA与world model的融合提供了一个elegant的范式。
