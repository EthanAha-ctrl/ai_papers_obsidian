---
source_pdf: DynaRend Learning 3D Dynamics via Masked Future Rendering for Robotic
  Manipulation.pdf
paper_sha256: 04b8a83486893a6eeb2c3350687cb24e6875243de9c998654305bed0230ed0df
processed_at: '2026-08-04T00:53:39-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我用最直白的话把 DynaRend 的逻辑拆解一遍。抛开那些学术包装，这篇 paper 的核心就是：**让 robot 在动手之前，先在脑子里建立一个 3D 场景，并且能想象出这个场景下一步会变成什么样。**

### 1. 解决什么痛点？

以前的 representation learning 面临一个困境。
如果用 2D pretraining（比如 MAE 或者 CLIP），模型只懂像素层面的 semantics，缺乏 3D geometry 和 physics 的概念，抓取时不知道深度。
如果用 video prediction（比如 GR-1），模型能预测未来，但是局限在 2D 平面上，不知道物体在 3D 空间里的真实遮挡关系。
如果用显式的 3D 表征（比如 3D Gaussians），结构太复杂，很难直接接上 policy head 去输出 action。

DynaRend 找到了一个 sweet spot：用 triplane 这种轻量级、结构化的 3D 表示，把所有优点揉在一起。

### 2. DynaRend 的"人话"逻辑

我们可以把 robot 的工作空间想象成一个立方体。DynaRend 的第一步，是把这个立方体从正面、侧面、顶面拍扁，得到三张 2D feature map（triplane）。

任何 3D 点的特征，就是把它的 x, y, z 坐标投影到这三张图上，通过 bilinear interpolation 查出三个向量，然后加起来。

公式直观感受：
$$ \mathbf{f}(x,y,z) = \text{Interp}(\mathbf{f}_{xy}, x, y) + \text{Interp}(\mathbf{f}_{xz}, x, z) + \text{Interp}(\mathbf{f}_{yz}, y, z) $$

这里 $\mathbf{f}_{xy}, \mathbf{f}_{xz}, \mathbf{f}_{yz}$ 就是三个平面的特征图。这种表示法内存占用极小，而且可以直接塞进 Transformer 处理。

### 3. 两个自监督游戏

训练模型的时候，DynaRend 让 Transformer 玩了两个游戏。

**游戏一：Reconstruction（遮挡补全）**
随机盖住 triplane 的一部分 feature，让 Transformer $\mathcal{E}_{\mathrm{recon}}$ 猜出完整的当前场景。这让模型理解空间结构和遮挡关系。

**游戏二：Future Prediction（预测未来）**
把补全好的当前场景喂给另一个 Transformer $\mathcal{E}_{\mathrm{pred}}$，让它预测下一个 keyframe 的 triplane。这让模型理解 dynamics，比如 drawer 被拉开后是什么样。

### 4. 怎么算 Loss？

模型不能瞎猜。DynaRend 借用了 NeRF 的 volumetric rendering 技术，把预测出来的 3D triplane "画" 回 2D 图片。

公式核心：
$$ \hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}_i $$

这里 $\mathbf{r}$ 是相机射线，$w_i$ 是 NeRF 那套累积透射率权重，$\mathbf{c}_i$ 是采样点的 RGB 颜色。模型把 3D triplane 渲染出 RGB、depth、semantic feature，然后和真实的 camera view 做对比算 loss。这就构成了一个完全 self-supervised 的闭环。

### 5. 最聪明的 Trick：用 AI 脑补新视角

这套 rendering 监督有个致命弱点：需要很多不同的 camera view 来防止模型过拟合到特定视角。Simulation 里随便加相机，但是 real-world 里 robot 旁边通常就固定俩相机。

DynaRend 的骚操作是：用 See3D（一个 multi-view diffusion model）来"脑补"新视角。

给定现有的 RGB-D 输入，稍微扰动一下相机 pose，让 See3D 生成这个新视角的逼真图片，再用 Depth Anything v2 估计深度。这样凭空造出了大量新视角的 supervision。

这一步的效果在 Table 3 的 ablation 里很明显：去掉这个 augmentation，success rate 直接掉 3.4%。

### 6. 实验结果直观解读

我们看 Table 1 的 RLBench 实验。
RVT baseline 只有 62.9%，加上各种 2D pretraining 撑死也就 67% 左右。DynaRend 达到了 83.2%。

更关键的是 inference speed。3D Diffuser Actor 虽然也强（81.3%），但是 1.4 Hz，太慢了。DynaRend 保持了 19.6 Hz 的实时速度。因为 triplane 本质上还是 2D feature map 的集合，计算开销远小于纯 3D 卷积。

在 real-world 的 Table 4 里，加了 distractor 物体之后，RVT-2 的成功率跌到 16%，DynaRend 还有 45%。因为 pretraining 阶段蒸馏了 foundation model（RADIOv2.5）的 semantic feature，模型对什么是目标物体、什么是干扰物体分得很清。

### 7. Karpathy 式的 Intuition

从你一直强调的 "predict next token" 角度来看，DynaRend 就是把 GPT 的自回归预测搬到了 3D 空间里。

输入是：当前状态的 3D 表征（masked）。
预测目标是：当前状态完整 3D 表征 + 未来状态的 3D 表征。
Loss 函数就是渲染成 2D 图片后的 photometric + semantic + depth 误差。

这里完全抛弃了人类标注，纯靠不同的可微渲染头去逼迫 Transformer 学会 3D geometry 和 physics dynamics。这和 NanoGPT 里用 next-token prediction 统一一切任务的哲学是一样的：找一个强大的、可微的、能自监督的 target，剩下的交给 scaling。

---
**参考链接：**
- RLBench (Simulation Benchmark): https://github.com/stepjam/RLBench
- See3D (View Augmentation Model): https://arxiv.org/abs/2412.06699
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- RVT-2 (Baseline Policy): https://robotic-view-transformer.github.io/

---

# DynaRend 深度解析：用 Masked Future Rendering 学 3D Dynamics for Robotic Manipulation

Andrej，这篇 paper 我仔细读完了，正好契合你一直关注的几个方向——representation learning for robotics、self-supervised pretraining、neural rendering 在 interactive scenario 里的应用。我把核心 idea 拆开讲一下，包括数学公式、架构 intuition、实验数据，以及和邻近工作的 connection。

---

## 1. Motivation 和核心 positioning

这篇 paper 的出发点非常清晰：**现有 robotic manipulation 的 representation learning 范式各有缺陷**。

| 范式 | 代表方法 | 缺什么 |
|------|---------|--------|
| 2D masked image modeling | VC-1 [1], Voltron [2], 3D-MVP [3] | 只有 static semantics/2D geometry，没有 3D，没有 dynamics |
| 2D video prediction | VPP [4], VidMan [5], GR-1/2 [6,7] | 有 dynamics 但是 2D 的，缺 3D 结构 |
| Explicit 3D dynamics | ManiGaussian [8], Imagination Policy [9] | 用 dynamic Gaussians / point clouds，结构复杂，scalability 差，需要 dense novel view supervision |

DynaRend 的 thesis 是：**用 differentiable volumetric rendering 同时把 geometry、semantics、future dynamics 学到一个 unified triplane representation 里**，pretraining 在 multi-view RGB-D video 上完成，不需要额外标注。

参考链接：
- VC-1: https://eiclr.cc/virtual/2023/poster/18398
- 3D-MVP: https://arxiv.org/abs/2411.06942
- ManiGaussian: https://arxiv.org/abs/2406.08428
- VPP: https://arxiv.org/abs/2412.14803
- RVT / RVT-2: https://robotic-view-transformer.github.io/, https://arxiv.org/abs/2406.08545

---

## 2. 整体架构

整个 pipeline 分三个 stage（参考 Figure 2）：

### Stage A: Triplane 构建
1. Multi-view RGB-D $\mathcal{O} = \{I_1, I_2, \cdots, I_n\}$ → depth back-projection → scene point cloud
2. MLP encode 每个 point 的 feature
3. 把 3D workspace 切成 $H \times W \times D$ 的 voxel grid（实际用 $16\times 16\times 16$）
4. Axis-aligned max pooling 投影到三个正交平面 → triplane features

### Stage B: Pretraining (Masked Future Rendering)
1. 随机 mask 一部分 triplane features，换成 learnable mask embedding
2. Concat CLIP language embedding
3. Reconstructive network $\mathcal{E}_{\mathrm{recon}}$ 重建当前 scene
4. Predictive network $\mathcal{E}_{\mathrm{pred}}$ 在重建结果上预测 future keyframe
5. 对两个 triplane 都做 volumetric rendering，监督 RGB、depth、semantic
6. 用 See3D 生成 novel view 增强 supervision

### Stage C: Finetuning
1. 把 $\mathcal{E}_{\mathrm{recon}}$ 和 $\mathcal{E}_{\mathrm{pred}}$ 串联起来当 triplane encoder（不带 mask）
2. 加 action decoder，预测 multi-view action value maps
3. 用 expert demonstration finetune

---

## 3. 技术细节深挖

### 3.1 Triplane Representation (Eq. 1)

$$\mathbf{f}_{xy} \in \mathbb{R}^{H \times W \times C}, \quad \mathbf{f}_{xz} \in \mathbb{R}^{H \times D \times C}, \quad \mathbf{f}_{yz} \in \mathbb{R}^{W \times D \times C}$$

变量解释：
- $H, W, D$：workspace 在 x, y, z 三个轴上的 voxel 分辨率
- $C$：feature channel dimension（实际用 768）
- $\mathbf{f}_{xy}, \mathbf{f}_{xz}, \mathbf{f}_{yz}$：三个正交平面的 2D feature map

**Intuition**: Triplane 最早在 EG3D [10] 和 K-Planes [11] 里提出来，作为 NeRF 的 efficient 替代。它本质上是把 3D scene 用三个 2D 特征图压缩表示——query 任意 3D 点 $(x, y, z)$ 时，在三个平面分别 bilinear interpolate 一次，再把结果 sum 起来。相比 voxel grid 是 $O(HWD)$ 内存，triplane 只有 $O(HW + HD + WD)$，而且 2D convolution 可以直接用。

DynaRend 选 triplane 而不是 voxel/pointcloud/Gaussian 的原因：**scalability + structural simplicity**。PerAct [12] 用 voxel 在 $100^3$ 分辨率已经要爆显存；ManiGaussian 用 dynamic Gaussians 难以和 policy head 直接接；GNFactor [13] 用 voxel + NeRF distillation。Triplane 是个 sweet spot。

参考：
- EG3D: https://nvlabs.github.io/eg3d/
- K-Planes: https://arxiv.org/abs/2301.10241
- PerAct: https://arxiv.org/abs/2209.05451
- GNFactor: https://arxiv.org/abs/2310.17061

### 3.2 Masked Future Prediction (Eq. 2, 3)

$$\mathcal{V}_{\mathrm{now}} = \{\mathbf{f}^{\mathrm{now}}_{xy}, \mathbf{f}^{\mathrm{now}}_{xz}, \mathbf{f}^{\mathrm{now}}_{yz}\} = \mathcal{E}_{\mathrm{recon}}(\bar{\mathcal{V}}, \mathbf{l})$$

$$\mathcal{V}_{\mathrm{future}} = \{\mathbf{f}^{\mathrm{future}}_{xy}, \cdots\} = \mathcal{E}_{\mathrm{pred}}(\mathcal{V}_{\mathrm{now}}, \mathbf{l})$$

变量：
- $\bar{\mathcal{V}}$：masked triplane，subset of $\mathcal{V}$ 被 learnable mask embedding 替换
- $\mathbf{l}$：CLIP text encoder 输出的 language embedding，concat 到 triplane 上
- $\mathcal{E}_{\mathrm{recon}}, \mathcal{E}_{\mathrm{pred}}$：两个 Transformer，4 层，带 SwiGLU [14] + QK Norm [15] + RoPE [16]

**Intuition**: 这是把 MAE [17] 的思想从 2D patches 迁移到 3D triplane 上。但有几个 critical 修改：
1. **Reconstruction 和 prediction 分离**——避免一个网络既要补全又要外推，任务复杂度太大。Reconstruction 学 spatial completion（像 MAE），prediction 学 temporal extrapolation（像 video prediction）。
2. **两个网络串联**——$\mathcal{E}_{\mathrm{pred}}$ 的输入是 $\mathcal{V}_{\mathrm{now}}$ 而不是 $\bar{\mathcal{V}}$，意味着先 reconstruct 出干净 current state，再 predict future。这比直接从 masked 输入预测 future 更稳。
3. **共享 architecture 但是不共享 weights**——这样 finetune 时它们能形成 hierarchical encoder。

Ablation Table 3 证实：
- w/o pretraining: 76.7%（baseline RVT-2 from scratch）
- w/o reconstruction: 80.7%（-2.5%）
- w/o future prediction: 78.9%（-4.3%）
- 完整 DynaRend: 83.2%

**Future prediction 比 reconstruction 贡献更大**——这暗示 robotic manipulation 更需要 temporal dynamics 而非纯 spatial completion。

参考：
- MAE: https://arxiv.org/abs/2111.06377
- SwiGLU: https://arxiv.org/abs/2002.05202
- RoPE: https://arxiv.org/abs/2104.09864

### 3.3 Volumetric Rendering (Eq. 4)

这是最技术化的部分，需要细讲。

对于每个 target view 的 pixel，对应相机射线：
$$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$$

变量：
- $\mathbf{o}$：camera origin（相机光心）
- $\mathbf{d}$：view direction（射线方向，单位向量）
- $t \in [t_{\mathrm{near}}, t_{\mathrm{far}}]$：沿射线的深度参数

沿射线采样 $N$ 个点 $\{\mathbf{p}_i = \mathbf{o} + t_i \mathbf{d}\}_{i=1}^N$，每个点投影到 triplane 的三个平面，bilinear interpolation 查询特征，sum 聚合得到 $\mathbf{v}_i$。

然后 MLP head 解码三个属性：
1. Density: $\sigma(\mathbf{v}_i): \mathbb{R}^C \to \mathbb{R}_+$（用 ReLU 软plus）
2. RGB: $\mathbf{c}(\mathbf{v}_i, \mathbf{d}): \mathbb{R}^{C+3} \to \mathbb{R}^3$（concat viewing direction 做 view-dependent color）
3. Semantic feature: $\mathbf{s}(\mathbf{v}_i, \mathbf{d}): \mathbb{R}^{C+3} \to \mathbb{R}^{C'}$（蒸馏 RADIOv2.5 [18] feature）

体渲染积分：

$$\hat{\mathbf{C}}(\mathbf{r}, \psi) = \sum_{i=1}^N w_i \mathbf{c}(\mathbf{v}_i, \mathbf{d})$$

$$\hat{\mathbf{S}}(\mathbf{r}, \psi) = \sum_{i=1}^N w_i \mathbf{s}(\mathbf{v}_i, \mathbf{d})$$

$$\hat{\mathbf{D}}(\mathbf{r}, \psi) = \sum_{i=1}^N w_i t_i$$

变量：
- $\hat{\mathbf{C}}, \hat{\mathbf{S}}, \hat{\mathbf{D}}$：渲染出来的 RGB、semantic feature、depth
- $\psi$：triplane parameters（隐含参数集）
- $w_i = T_i (1 - \exp(\sigma(\mathbf{v}_i) \delta_i))$：第 $i$ 个采样点的权重
- $T_i = \exp(-\sum_{j=1}^{i-1} \sigma(\mathbf{v}_j) \delta_j)$：累积 transmittance（光线没被前面遮挡的概率）
- $\delta_i = t_{i+1} - t_i$：相邻采样点距离

**Intuition**: 这就是 NeRF [19] 那套体渲染公式，把 3D 表征"渲染"成 2D image。但 DynaRend 的关键区别：
1. **Supervision 信号丰富**——不只渲染 RGB，还渲染 depth（用 SiLog loss）和 semantic feature（蒸馏 RADIOv2.5）。Semantic supervision 让 triplane 里编码 high-level semantics，而不只是 photometric reconstruction。
2. **Random pixel sampling**——每次 iteration 只 sample $K$ 个 pixels（具体值未明确，但 GNFactor 用 4096），加速训练。
3. **Coarse-to-fine**——Appendix A 提到 follow GNFactor 用 hierarchical rendering，fine network 用 depth-guided sampling，能在细节上提升。

Table 3 的 ablation：
- w/o RGB loss: 78.2%（-5.0%）← 最关键
- w/o semantic loss: 80.4%（-2.8%）
- w/o depth loss: 82.0%（-1.2%）← 因为 triplane 本身就来自 depth projection

参考：
- NeRF: https://arxiv.org/abs/2003.08934
- RADIOv2.5: https://arxiv.org/abs/2501.01206
- SiLog loss (depth): https://arxiv.org/abs/1406.2283

### 3.4 Target View Augmentation（关键创新点）

这是我觉得 paper 里最聪明的设计之一。问题在于：

**现有 rendering-based pretraining 方法（GNFactor, ManiGaussian）需要大量 calibrated novel views 做 supervision**。Simulation 里没问题（可以加任意 camera），但 real-world 不行——你装几个 RGB-D 相机就固定几个 view。

DynaRend 的解法：
1. 给定一组 calibrated multi-view RGB-D（fixed cameras）
2. 选一个 base view，perturb 相机 pose（±30 度随机偏移）
3. 把 multi-view point cloud warp 到 target pose（projection + back-projection）
4. 用 **See3D** [20]——pretrained visual-conditioned multi-view diffusion model——生成 realistic novel view image
5. 用 **Depth Anything v2** [21] 估计 depth
6. 生成 25 帧的 camera trajectory，每个 keyframe 采样 4 条 trajectory，总共 100 个 augmented views per keyframe

**Intuition**: 这是个 bootstrapping 思路——既然有 generative model 能 synthesize plausible 新视角，那就用它来增强 rendering supervision 的 view diversity。本质上是用 foundation model 的 prior 补 fixed-camera setup 的不足。

Ablation Table 3: w/o novel view augmentation → 79.8%（-3.4%），贡献不小。

Appendix A 提到关键细节：**只 augment keyframe**（trajectory 里中间帧不 augment），原因应该是 diffusion 生成耗时。

参考：
- See3D: https://arxiv.org/abs/2412.06699
- Depth Anything v2: https://arxiv.org/abs/2406.09414

### 3.5 Loss Functions (Eq. 5, 6)

$$\mathcal{L}_{\mathrm{recon}} = \lambda_c \|\mathbf{C}(\mathbf{r}) - \hat{\mathbf{C}}(\mathbf{r}, \gamma_{\mathrm{now}})\| + \lambda_s \|\mathbf{S}(\mathbf{r}) - \hat{\mathbf{S}}(\mathbf{r}, \gamma_{\mathrm{now}})\| + \lambda_d \mathrm{SiLog}(\mathbf{D}(\mathbf{r}), \hat{\mathbf{D}}(\mathbf{r}, \gamma_{\mathrm{now}}))$$

$$\mathcal{L}_{\mathrm{pred}} = \text{同上，但用 future target view 监督}$$

$$\mathcal{L}_{\mathrm{pretrain}} = \lambda_{\mathrm{recon}} \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{pred}} \mathcal{L}_{\mathrm{pred}}$$

变量：
- $\mathbf{C}(\mathbf{r}), \mathbf{S}(\mathbf{r}), \mathbf{D}(\mathbf{r})$：target view 的 ground truth RGB、semantic、depth
- $\gamma_{\mathrm{now}}, \gamma_{\mathrm{future}}$：当前和预测的 triplane 参数（注意 paper 里写法有些混乱，应该都是 $\psi$）
- $\lambda_c, \lambda_s, \lambda_d$：RGB/semantic/depth loss 的权重
- $\lambda_{\mathrm{recon}}, \lambda_{\mathrm{pred}}$：reconstruction 和 prediction 的权重
- SiLog: scale-invariant log loss $\mathrm{SiLog}(D, \hat{D}) = \frac{1}{N} \sum_i d_i^2 - \frac{\lambda}{N^2}(\sum_i d_i)^2$，$d_i = \ln \hat{D}_i - \ln D_i$

**Intuition**: 为什么 semantic loss 重要？因为 RGB reconstruction 会过拟合到 photometric detail，semantic feature（RADIOv2.5 是 agglomerative foundation model）则强调 object-level、affordance-level 的语义。下游 manipulation 需要的是 "哪里是 handle、哪里是 drawer" 这种语义，而不是精确的 pixel color。这是把 vision foundation model 的 prior 蒸馏进 triplane 的关键。

### 3.6 Action Prediction (Eq. 7)

Finetune 阶段 follow RVT 的 multi-view action value map 范式：

$$\mathcal{L}_{\mathrm{finetune}} = \lambda_{\mathrm{trans}} \mathrm{CE}(\mathbf{a}_{\mathrm{trans}}, \hat{\mathbf{a}}_{\mathrm{trans}}) + \lambda_{\mathrm{rot}} \mathrm{CE}(\mathbf{a}_{\mathrm{rot}}, \hat{\mathbf{a}}_{\mathrm{rot}}) + \lambda_{\mathrm{gripper}} \mathrm{CE}(\mathbf{a}_{\mathrm{gripper}}, \hat{\mathbf{a}}_{\mathrm{gripper}})$$

变量：
- $\mathbf{a}_{\mathrm{trans}} = \{a_x, a_y, a_z\}$：end-effector translation（连续值 → 投影到 triplane 三个平面做 heatmap target）
- $\mathbf{a}_{\mathrm{rot}}$：discretized Euler angles（classification）
- $\mathbf{a}_{\mathrm{gripper}} \in \{0, 1\}$：binary gripper state
- $\mathrm{CE}$：cross-entropy

**Inference 流程**:
1. 把 triplane 三个平面的 translation heatmap 通过 broadcasting + summing 融合成一个 3D heatmap over workspace
2. 取 argmax 作为 translation
3. 用 translation 位置 query triplane feature，过 MLP 预测 rotation + gripper state

**Intuition**: 这是 PerAct / RVT 那套 keyframe policy + heat map decoding 的做法。Heatmap 把连续 regression 变成 voxel classification，更稳，而且能可视化 attention。DynaRend 的好处是 triplane 已经是 3D-aware + dynamics-aware 的，policy head 只需要 "读" 而不是从 2D feature 重建 3D 理解。

参考：
- PerAct: https://arxiv.org/abs/2209.05451
- RVT: https://robotic-view-transformer.github.io/

### 3.7 Architecture 细节

Appendix A 给的 hyperparameters：
- Triplane resolution: $16 \times 16 \times 16$（注意：三个 plane 分别是 $16\times 16$, $16\times 16$, $16\times 16$）
- Transformer depth: 8 层（recon + pred 各 8 层？还是总 8 层，paper 正文说 4 层，appendix 写 8 层，可能是 paper 不一致）
- Width: 768
- Attention heads: 12
- MLP ratio: 4.0
- Render head: 2 层 MLP with residual，width 768
- Batch size: 256
- Ray batch size: 32
- Optimizer: AdamW, lr $10^{-4}$, cosine decay

**SE(3) Augmentation**：
- Translation: 随机 ±0.125 m 沿 x, y, z
- Rotation: 随机 ±45° 绕 z 轴
- 同时 augment point cloud、camera pose、action label

**M-RoPE**: 把 feature dimension 切三份，分别对三个 spatial dimension 加 RoPE，让 Transformer 学习 triplane 三个平面之间的 relative position。

---

## 4. 实验数据深度分析

### 4.1 RLBench-18 (Table 1)

| Method | Avg S.R. ↑ | Avg Rank ↓ | Inf Speed ↑ |
|--------|-----------|-----------|-------------|
| C2F-ARM-BC | 20.1 | 6.4 | - |
| PerAct | 49.4 | 5.1 | 4.9 Hz |
| RVT | 62.9 | 4.3 | 11.6 Hz |
| 3D-MVP | 67.5 | 3.2 | 11.6 Hz |
| 3D Diffuser Actor | 81.3 | 2.2 | 1.4 Hz |
| RVT-2 | 81.4 | 2.2 | 20.6 Hz |
| **DynaRend** | **83.2** | **1.5** | **19.6 Hz** |

关键观察：
1. DynaRend 比 RVT-2（current SOTA）高 1.8%，rank 更好
2. Inference speed 19.6 Hz 接近 RVT-2，远超 3D Diffuser Actor（1.4 Hz，太慢）
3. 比 RVT (62.9%) 提升 **20.3 个百分点**，比 from-scratch baseline 提升 32.3%

具体任务表现：
- Slide Block: 100% (RVT-2: 92%)
- Sweep to Dustpan: 93.6% (RVT-2: 100%, 这里 DynaRend 略输)
- Close Jar: 91.2% (RVT-2: 76%, **大胜**)
- Stack Cups: 82.4% (RVT-2: 69%, **大胜**)
- Stack Blocks: 71.2% (RVT-2: 80%, 略输)
- Place Cups: 25.6% (RVT-2: 38%, **输了**)
- Sort Shape: 44.8% (RVT-2: 35%, 胜)
- Insert Peg: 31.2% (RVT-2: 40%, 输)

**Intuition**: DynaRend 在 long-horizon、多步骤任务（Stack Cups, Close Jar）上明显占优，因为这些任务更依赖 dynamics reasoning。在 Place Cups 这种需要精细 count-aware 任务上输——可能是 language instruction 里 "2 cups" "3 cups" 这种 count 语义没被 CLIP 编码好。

### 4.2 RLBench-71 (Table 2)

| Method | Group 1 (35 tasks) | Group 2 (36 tasks) | Avg |
|--------|-------------------|---------------------|------|
| RVT (single-stage) | 71.9 | 50.4 | 61.1 |
| MoCov3 (two-stage) | 73.7 | 54.2 | 63.9 |
| MAE | 78.3 | 57.7 | 68.0 |
| DINOv2 | 78.2 | 56.1 | 67.1 |
| CLIP | 76.8 | 55.7 | 66.2 |
| MVP | 76.2 | 56.3 | 66.2 |
| VC-1 | 80.1 | 55.7 | 67.9 |
| SPA | 80.5 | 61.2 | 70.8 |
| **DynaRend** | **81.4** | **71.8** | **76.6** |

**关键 insight**:
1. Group 2 更难（50.4 vs 71.9 for RVT），DynaRend 在 Group 2 上提升巨大（71.8 vs SPA 61.2 = +10.6）
2. 平均提升 RVT 25.2%，比 SPA（另一个 3D-aware pretraining）提升 5.8%
3. SPA 也用 differentiable rendering，但是只做 reconstruction 不做 prediction——所以**future prediction 是关键 differentiation**

### 4.3 Colosseum Benchmark (Generalization)

Colosseum 测 12 类 perturbation：MO color/size/texture, RO color/size/texture, light, table color, table texture, bg texture, camera pose, distractor。

DynaRend 在 **MO Texture** 和 **Background Texture** 类别上提升最大——这暗示 3D-aware + dynamics pretraining 学到的是 object-level、affordance-level 的鲁棒表示，而不是 surface texture 过拟合。

### 4.4 Real-world Experiments (Table 4)

5 个 task，30 demonstrations each。

| Method | Put Item in Drawer | Stack Blocks | Sort Shapes | Close Pot | Stack Cups | Avg |
|--------|--------------------|--------------|--------------|------------|--------------|------|
| RVT-2 | 45 | 60 | 10 | 60 | 15 | 37 |
| DynaRend | 65 | 60 | 35 | 85 | 40 | **57** |

**With distractors**:
| Method | Avg |
|--------|------|
| RVT-2 | 16 |
| DynaRend | **45** |

**Intuition**: Distractor 场景下 DynaRend 比 RVT-2 高 29 个百分点（45 vs 16）——这是巨大的 robustness 差距。原因应该是 semantic distillation (RADIOv2.5) 让模型对 distractor 不敏感。

---

## 5. Ablation 深度分析 (Table 3, Figure 3)

### 5.1 Mask Ratio (Figure 3)
- 0% mask: 性能最低（容易过拟合 fixed view）
- 中等 mask（应该 30-50%）：最优
- 高 mask（>70%）：性能下降（reconstruction 不出来）

这和 MAE 在 ImageNet 上的发现一致——mask ratio 是个 sweet spot，过高信息丢失太多，过低又没有 regularization。

### 5.2 各 Loss 贡献
- w/o RGB: -5.0% （最关键）
- w/o semantic: -2.8%
- w/o depth: -1.2%（贡献小，因为 triplane 来自 depth）

### 5.3 各 Pretraining 目标
- w/o pretraining: -6.5%
- w/o reconstruction: -2.5%
- w/o future prediction: -4.3%

---

## 6. 我的 Intuition 和思考

### 6.1 为什么 work？

我认为 DynaRend 的成功来自三个 self-supervised 信号的"乘法效应"：

1. **Spatial completion**（reconstruction）：让模型理解 occlusion 和 object boundaries
2. **Temporal extrapolation**（future prediction）：让模型理解 physics 和 affordance
3. **Multi-modal rendering supervision**（RGB + depth + semantic）：让 representation 同时编码 photometric、geometric、semantic 信息

单做任何一个都不够：MAE 只有 (1)，video prediction 只有 (2)，NeRF reconstruction 只有 (3)。

### 6.2 为什么是 Triplane 而不是 NeRF/Gaussian？

- NeRF: 不可解析，policy head 难接
- Voxel: 内存爆炸
- 3D Gaussian: 结构太复杂，dynamics 难处理
- Point cloud: 无 regular structure，Transformer 难处理
- **Triplane**: compact、structured、Transformer-friendly、rendering-efficient

### 6.3 为什么 Target View Augmentation 这么重要？

这其实是个 **chicken-and-egg** 问题：rendering pretraining 需要多视角，real-world 没多视角，怎么办？用 generative prior（See3D）合成新视角来 bootstrap。这等价于把 foundation model 的 3D prior 蒸馏到 task-specific representation 里。我觉得这个思路可以推广——以后任何需要多视角监督但实际没多视角的场景都可以用这个 trick。

### 6.4 Limitations 和潜在改进

Paper 自己提到：依赖 external low-level motion planner 做 keyframe 之间的执行。未来方向是 end-to-end action sequence prediction。

我看其他 limitation：
1. **CLIP 文本编码弱**：Place Cups 任务表现差暗示 count-aware 指令处理不好，需要更强的 language grounding
2. **See3D 质量决定上限**：augmented view 质量差会 hurt pretraining，需要 monitor
3. **Pretraining data 假设 multi-view RGB-D**：单目 setup 不行，限制了部署场景
4. **Triplane resolution 16³ 较粗**：精细 manipulation（Insert Peg 只有 31.2%）受限

### 6.5 和 VLA (Vision-Language-Action) models 的关系

你最近关注的 $\pi_0$ [22], OpenVLA [23], RT-2 [24] 这些 VLA model 是 end-to-end 范式。DynaRend 是 complementary 路线——**先把 perception 做 robust，再用轻量 policy head**。两者可以结合：用 DynaRend-style representation pretraining 给 VLA model 提供 3D-aware visual encoder。

参考：
- $\pi_0$: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818

### 6.6 和你的 NanoGPT / Eureka Labs 思路的 connection

DynaRend 的 pretraining pipeline 其实是个非常 elegant 的 self-supervised 学习案例——**用 rendering 作为 self-supervised 信号学 representation**，类似 MAE 用 reconstruction 学 vision。如果用你的 teaching 风格讲，这正好是：
1. 给定 input（masked triplane + language）
2. Predict target（RGB/depth/semantic of current and future views）
3. Differentiable rendering 是 forward pass，loss 是 backward pass
4. Transformer 是 backbone，learnable mask token + RoPE + SwiGLU 是 building block

可以看成是一个 **"domain-specific nano-GPT"**：tokens 是 triplane positions，target 是 rendered views，architecture 是标准 Transformer。

---

## 7. 总结

DynaRend 的 contribution 可以浓缩成一句话：**用 masked future rendering 把 geometry、dynamics、semantics 三个 self-supervised 信号统一到 triplane representation 里，再通过 differentiable rendering + foundation model distillation 实现 pretraining**。

技术上最 elegant 的部分：
1. Reconstruction + Prediction 串联（避免任务冲突）
2. Multi-modal rendering supervision（RGB+depth+semantic）
3. See3D-based target view augmentation（解决 real-world 多视角不足）
4. Triplane 作为 scalable 中间表示

实验上最 striking 的部分：
1. 比 RVT-2 在 18-task 上 +1.8%，71-task 上 +5.8%
2. Real-world with distractors +29% absolute
3. Inference speed 19.6 Hz 实时

我觉得这篇 paper 是 **robotic representation learning 从 2D pretraining 到 3D-aware + dynamics-aware pretraining 的一个标志性工作**，未来会被很多 follow-up 引用——尤其是 target view augmentation 这个 trick，应该会成为 real-world robot learning 的标配。

参考链接汇总：
- Paper（推断）: CVPR 2026 或类似 venue
- Related codebases: 
  - RVT-2: https://github.com/roborange-ai/rvt_2
  - GNFactor: https://github.com/YanjieZe/GNFactor
  - RLBench: https://github.com/stepjam/RLBench
  - Colosseum: https://github.com/andrewsilv/colosseum
- Foundation models:
  - DINOv2: https://github.com/facebookresearch/dinov2
  - See3D: https://arxiv.org/abs/2412.06699
  - Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2

如果你想 dig deeper 某一块（比如 volumetric rendering 数学推导、triplane 的 architecture choice、或者怎么把这套迁移到 VLA model 上），告诉我，我可以展开讲。
