---
source_pdf: Not All Tasks Quantize Equally Fisher-Guided.pdf
paper_sha256: e469f5fb94ef12ff765e38d1702e0b7300a755b48f0b4ee6fdb0bdc08526ea87
processed_at: '2026-08-05T22:45:37-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 FGQ

---

## 一句话版本

**VGGT 这个模型同时干三件事，但量化的时候大家之前都是一刀切，导致最怕出错的那件事（点云重建）崩了。这篇 paper 的核心就是：先算清楚哪个 channel 对哪个任务最要命，然后量化的时候重点保护它们。**

---

## 背景是什么

VGGT 是一个挺火的 3D 重建模型。你给它几张照片，它一次 forward pass 就吐出来三样东西：

- **Camera pose**：这些照片是从哪个角度拍的
- **Depth map**：每个像素离相机多远
- **Point map**：把所有像素拼成 3D 点云

三件事共用同一个 backbone（一个 transformer），最后通过不同的 head 输出。

问题：这个模型有 1.2 billion 参数。要部署到手机、机器人、AR 眼镜上，太大了。所以要做 **quantization**（量化），把 16-bit 的数字压成 4-bit，省内存、省算力。

---

## 量化的老办法为什么不work

之前的量化方法（FlatQuant、QuaRot、QuantVGGT）思路大概是这样：

1. 在 linear layer 前面插一个可学的矩阵 $P$，把 activation 和 weight 都做个变换
2. 变换后分布更"平"，更适合 4-bit 量化
3. 校准（calibration）的时候，让量化后的 block 输出尽量接近原始 full-precision 的输出

**关键问题在第 3 步**。校准时用的 loss 是 **plain MSE**——每个 channel 的误差权重都一样。等于说："我不管这个 channel 后面要喂给哪个 task，反正你每个 channel 都给我尽量还原就完事了"。

但 VGGT 是 **multi-task** 的。三个 task 对量化误差的耐受度完全不一样：

- **Camera pose**：是 global 的，几个数字描述全局几何，很 robust，W4A4 下几乎不掉点
- **Depth**：dense 的，但每个像素独立算，中等敏感
- **Point map**：dense 的，而且要跨视图对齐，误差会 spatially accumulate，W4A4 下直接 NPD 飙到 200%~500%，基本崩了

**之前的方法把所有 task 一视同仁，结果 calibration 的优化目标被那些"不在乎的 channel"主导，真正要紧的 channel 没被保护好。**

这就好比你给三个人体检，一个人的痛点在心脏，一个人的痛点在膝盖，一个人的痛点在眼睛，结果医生给三个人都开了同一剂量的止痛药，最后心脏病的人没救过来，膝盖和眼睛的人倒是没事。

---

## 他们怎么解决的

核心 idea：**在 calibration 的 MSE loss 里，给每个 channel 加一个权重，这个权重反映"这个 channel 对所有 task 的整体重要性"。**

那这个权重怎么算？这就是这篇 paper 的数学核心。

### 第一步：直觉——为什么用二阶信息

你量化一个 block，输出从 $\mathbf{h}$ 变成了 $\mathbf{h} + \Delta\mathbf{h}$。这个 $\Delta\mathbf{h}$ 会怎么影响 task loss？

对 task loss 做 Taylor 展开：

$$\Delta\mathcal{L} \approx \mathbf{g}^\top \Delta\mathbf{h} + \frac{1}{2}\Delta\mathbf{h}^\top \mathbf{H} \Delta\mathbf{h}$$

- 第一项是 gradient（一阶）
- 第二项是 Hessian（二阶）

模型已经训练好了，在 local optimum 附近，**gradient 期望为零**（这个结论 LeCun 1989 年就用了）。所以一阶项消失，剩下的就是 Hessian 项。

**Hessian 告诉你：在哪个方向上，扰动会被放大成大量 loss。Hessian 的大 eigenvalue 方向就是"敏感方向"，量化时要特别小心。**

### 第二步：Hessian 太贵，用 Fisher 替代

直接算 Hessian 灾难——计算需要二阶 autograd，存储是 $C \times C$（$C = 1024$ 就是一百万个数 per block per task）。

但有个经典的数学恒等式（Bartlett identity）说：**当模型 fit 了数据分布时，Hessian 的期望等于 Fisher 信息矩阵**。而 Fisher 只需要 first-order gradient 就能算：

$$F_c = \mathbb{E}\left[\left(\frac{\partial\mathcal{L}}{\partial h_c}\right)^2\right]$$

也就是说，**你做一次普通 backward，把每个 channel 的 gradient 平方累加起来，就近似了 Hessian 在这个 channel 上的 curvature**。

这个 trick 不新——pruning 领域用了 30 年了（Optimal Brain Damage 就是这思路）。新的是把它用在了 **multi-task 3D 量化** 上。

### 第三步：再简化成 diagonal Fisher

完整 Fisher 是 $C \times C$，还是太大。所以只保留对角线——每个 channel 一个标量，存储从 $O(C^2)$ 降到 $O(C)$。

论文验证了这个 diagonal Fisher 跟实际量化 loss 的 Pearson correlation 是 **0.88**。虽然丢了 cross-channel 信息，但对 ranking channel importance 已经够用。

### 第四步：三个 task 的 Fisher 怎么合

每个 task 给每个 channel 一个 Fisher 值。三个 task 怎么聚合？

1. **每个 task 先除以自己的均值**（防止 gradient magnitude 大的 task 霸占权重）
2. **三个 task 等权相加**
3. **每个 block 内再归一化到均值 1**
4. **设一个 0.01 的 floor**，防止某些 channel 权重被压到 0 直接被忽略

最后得到每个 channel 的权重 $w_l[c]$。

### 第五步：加进 calibration loss

把原来 FlatQuant 的 plain MSE 换成 **weighted MSE**：

$$\mathcal{L}_{\text{FGQ}} = \frac{1}{|\mathcal{T}|C}\sum_t \sum_c w_l[c] \cdot (\text{quantized} - \text{full precision})^2$$

就这么简单。**其他什么都没改**，FlatQuant 的 affine transformation 结构、learnable clipping、quantizer 放置位置统统不动。只改了 calibration loss 的 weighting scheme。

---

## 为什么这个思路 work

直觉上，**量化时保护什么，取决于"误差往下游传播时会被放大多少"**。Fisher 正好量化了"这个 channel 的误差会被 task loss 放大多少"。

- Fisher 大的 channel：误差传到 task head 会变成大量 loss → 量化时要重点保护
- Fisher 小的 channel：误差传到 task head 影响不大 → 可以适当放宽

Plain MSE 不区分，把 Fisher 大和小的 channel 一视同仁，结果 calibration 的优化能力被浪费在不重要的 channel 上。Fisher-weighted MSE 让 calibration 优先保住要命的 channel。

---

## 效果如何

**Point map reconstruction（最敏感的 task，之前最崩的）**：
- ETH3D accuracy mean：QuantVGGT 0.312 → FGQ **0.275**，相对改进 ~12%
- 7-Scenes completeness：QuantVGGT 0.085 → FGQ **0.059**，相对改进 ~30%
- DTU completeness：QuantVGGT 1.933 → FGQ **1.879**

这就是 paper 标题说的 "up to 39% relative improvement"。

**Depth estimation**（KITTI Mono）：
- AbsRel：0.088 → **0.079**（~10% 改进）
- SqRel：0.446 → **0.389**（~13% 改进）

**Camera pose**（最不敏感的 task）：
- 改进 modest，Co3Dv2 AUC@15：0.8840 → 0.8887
- 这很合理——pose 本来就不敏感，Fisher-guided 给它的边际收益自然就小

**在 $\pi^3$ 模型上泛化**：
- Co3Dv2 AUC@3：FlatQuant 0.2686 → FGQ **0.5032**（几乎恢复到 FP16 的 0.5140）
- 说明这个方法不是 VGGT 专属的 trick

---

## Ablation 的关键发现

最有意思的 ablation 是 **multi-task Fisher 互补性**：

在 Co3Dv2（测 pose）上：
- 只用 point Fisher：0.8868
- point + depth：0.8875
- point + depth + camera：**0.8952**

在 DTU（测 point map）上：
- 只用 camera Fisher：1.465
- camera + depth：1.458
- camera + depth + point：**1.446**

**结论：与 evaluation metric 对齐的那个 task 的 Fisher 贡献最大，但加上其他 task 的 Fisher 总是更好**。这说明不同 task 的 gradient 提供了互补的重要性信号——不同 head 从 backbone 不同角度看 importance，综合起来比单一视角更准确。

---

## 成本

- **Calibration 时间**：3.54GB → 3.62GB，0.87h → 0.92h。内存 +2.26%，时间 +5.73%
- **推理速度**：W4A4 真实低比特推理比 RTN 慢约 10%（因为额外的 affine transform），但精度大幅领先
- **Fisher 估计本身**：64 clips × 4 frames = 256 个样本，一次估计，calibration 期间固定。非常便宜

---

## 这篇 paper 真正的贡献

数学上，diagonal Fisher 替代 Hessian 这个 trick 不新，pruning 圈子 1989 年就在用。Engineering 上，把 MSE 换成 weighted MSE 也很 trivial。

**真正的贡献是 insight**：他们发现并量化了一个具体问题——**multi-task 模型量化时，不同 task/block/channel 的 sensitivity 严重不对称，uniform calibration 会被 insensitive task 主导，把 sensitive task 搞崩**。然后他们用 Fisher 这个成熟工具，针对性地解决了这个问题。

这个 insight 可以推广到很多地方：
- 任何 multi-task 模型的量化（LLM + 多个 head 的场景）
- 任何 multi-modal 模型（视觉+语言+音频）
- Mixed-precision allocation（用 Fisher 决定哪些 block 用 8-bit、哪些用 4-bit）
- 甚至 pruning 和 distillation——任何"哪些参数更重要"的决策都可以 Fisher-guided

---

## 一句话总结

**别再 uniform 量化 multi-task 模型了。先花点 cheap 的代价算一下 Fisher，然后 calibration 时按 Fisher 加权保护关键 channel，dense task 能多救回十几个百分点，成本几乎为零。**

---

## 相关 reference

- [VGGT 原始 paper (CVPR 2025)](https://arxiv.org/abs/2503.19551)
- [FlatQuant (FGQ 的 base method, ICML 2025)](https://arxiv.org/abs/2410.09926)
- [QuantVGGT (前 SOTA, 被 FGQ 超越)](https://arxiv.org/abs/2509.21302)
- [SpinQuant (learnable rotation 的代表作)](https://arxiv.org/abs/2405.16406)
- [Optimal Brain Damage (LeCun 1989, Fisher pruning 的鼻祖)](https://papers.nips.cc/paper/1989/hash/6c9882bbd1e0b2f3e6f1c6e1c6e1c6e1-Abstract.html)
- [Fisher-aware quantization for DETR (Yang 2024, 直接前驱)](https://arxiv.org/abs/2407.03442)
- [PCGrad (multi-task gradient conflict)](https://arxiv.org/abs/2001.06782)
- [FGQ 代码](https://github.com/ypzhng/FGQ)

---

# Fisher-Guided Quantization for VGGT — 深度技术讲解

Andrej, 这篇paper处理的是一个非常specific的问题，但其背后的数学结构和直觉相当优雅。让我从底层开始一层一层build up your intuition。

---

## 1. 问题背景：VGGT 的多任务架构

**VGGT (Visual Geometry Grounded Transformer)** 是一个 feed-forward 3D reconstruction 模型，核心 idea 是用一个 shared transformer backbone 在 single forward pass 中联合预测多个几何任务：
- **Camera pose estimation** (相机位姿)
- **Depth estimation** (深度图)
- **Point map reconstruction** (点云地图重建)

### 1.1 架构细节

输入是 N 张图像 $\{I_i\}_{i=1}^N$，每张图像通过 DINOv2 encoder patchified 成 K 个 tokens：

$$\mathbf{T} = [t_1; t_2; \ldots; t_N] \in \mathbb{R}^{N \times K \times C}$$

其中：
- $N$ = 图像数量 (views)
- $K$ = 每张图像的 patch tokens 数量
- $C$ = hidden channel dimension (在 VGGT-1.2B 中 $C = 1024$)

backbone 由 $L$ 个 **Alternating-Attention (AA) blocks** 组成，每个 AA block 包含两个 attention 操作（公式1-2）：

$$\mathbf{T}_i^{(\ell+\frac{1}{2})} = \text{FrameAttention}(\mathbf{T}_i^{(\ell)}), \quad i = 1, \ldots, N$$

$$\mathbf{T}^{(\ell+1)} = \text{GlobalAttention}(\mathbf{T}^{(\ell+\frac{1}{2})})$$

这里 $\ell \in \{0, 1, \ldots, L-1\}$ 是 block index，$\mathbf{T}^{(\ell)}$ 是第 $\ell$ 个 block 输入处的 token state。**FrameAttention** 在每张图像的 K 个 tokens 内独立操作（保留 per-view 结构），**GlobalAttention** 在所有 $N \times K$ tokens 上联合操作（跨视图传播信息）。VGGT-1.2B 共有 $L = 24$ 个 AA blocks，每个 AA block 含一个 frame block 和一个 global block，所以 calibrated blocks 总数是 48。

Reference: [VGGT paper (Wang et al., CVPR 2025)](https://arxiv.org/abs/2503.19551)

---

## 2. 核心观察：不同 task 的 quantization sensitivity 严重不对称

这是整篇 paper 的 motivation。作者定义了一个 **Normalized Performance Degradation (NPD)** metric (公式3)：

$$\text{NPD}_k(b) = \frac{|\text{Metric}_k(b) - \text{Metric}_k(\text{FP16})|}{\text{Metric}_k(\text{FP16})} \times 100\%$$

其中：
- $k$ = task index (depth, pose, point map)
- $b$ = bit-width (例如 4)
- $\text{Metric}_k(\cdot)$ = task $k$ 在某个 bit-width 下的 evaluation metric

### 2.1 Task-level asymmetry (Figure 2)

在 W4A4 quantization 下：
- **Camera pose** 保持 robust
- **Depth** 中等退化
- **Point map reconstruction** NPD 高达 200% ~ 500%（崩溃！）

这个 asymmetry 的物理直觉是：camera pose 是 global prediction（少量参数描述全局几何），而 point map 是 dense prediction（每个空间位置都有几何输出），所以 quantization noise 在 dense task 上 spatially accumulate。

### 2.2 Block 和 channel-level asymmetry (Figure 3a, 3b)

更细致的观察：分别测量 camera 和 depth 任务的 per-block, per-channel quantization loss。**两个 task 在 block 维度和 channel 维度上产生明显不同的 loss profile**——对某个 task 至关重要的 block/channel，对另一个 task 可能完全不重要。

**直接结论**：任何把所有 task 同等对待的 uniform quantization policy，都会被 insensitive task 主导（因为它们 loss 量级大），从而牺牲 sensitive task 的精度。

---

## 3. 方法：Fisher-Guided Quantization (FGQ)

### 3.1 从二阶 Taylor 展开说起 (公式4-5)

设 $\mathbf{h}_l \in \mathbb{R}^C$ 是 block $l$ 的输出，$\mathcal{L}_k$ 是 task $k$ 的 loss。Quantization 引入 perturbation $\Delta \mathbf{h}_l$。对 $\mathcal{L}_k$ 在 full-precision 输出附近做二阶 Taylor 展开：

$$\Delta \mathcal{L}_k \approx \mathbf{g}_k^\top \Delta \mathbf{h}_l + \frac{1}{2} \Delta \mathbf{h}_l^\top \mathbf{H}_k \Delta \mathbf{h}_l$$

变量含义：
- $\mathbf{g}_k = \nabla_{\mathbf{h}_l} \mathcal{L}_k$：task $k$ loss 对 $\mathbf{h}_l$ 的 gradient
- $\mathbf{H}_k = \nabla^2_{\mathbf{h}_l} \mathcal{L}_k$：task $k$ loss 对 $\mathbf{h}_l$ 的 Hessian
- $\Delta \mathbf{h}_l$：quantization 造成的 perturbation

**关键观察**：pre-trained VGGT 处于 local optimum 附近，所以 $\mathbb{E}[\mathbf{g}_k] \to \mathbf{0}$（一阶项期望消失，这是 [LeCun et al., 1989 - Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbd1e0b2f3e6f1c6e1c6e1c6e1-Abstract.html) 的经典结论）。于是：

$$\mathbb{E}[\Delta \mathcal{L}_k] \approx \frac{1}{2} \Delta \mathbf{h}_l^\top \mathbf{H}_k \Delta \mathbf{h}_l$$

**Intuition**：Hessian 的 eigenvalues 描述了不同 perturbation direction 上的 "loss 增益"。大 eigenvalue 方向上 quantization error 会被放大成大量 task loss，小 eigenvalue 方向上误差被吸收。

### 3.2 为什么不能用 Hessian？Fisher 信息作为 surrogate

直接计算 $\mathbf{H}_k$ 有两个问题：
1. **计算成本**：二阶 autograd 在每个 block、每个 task 上都有 huge overhead
2. **存储成本**：$\mathbf{H}_k \in \mathbb{R}^{C \times C}$，每个 block 每 task 需要 $\mathcal{O}(C^2)$ 内存

**解决方案**：用 Fisher Information Matrix 代替 Hessian。这是基于 Proposition 3.1 (second Bartlett identity 的一个实例)。

### 3.3 Proposition 3.1 的证明直觉 (公式6)

设 $p_z(y|x)$ 是 normalized conditional model（$z$ 可以是参数或中间 activation），$\ell(z; x, y) = -\log p_z(y|x)$ 是 NLL loss。如果模型在 $z^\star$ 处 well-specified，即 $q(y|x) = p_{z^\star}(y|x)$，那么：

$$\mathbb{E}_{x \sim q(x), y \sim q(y|x)}[\nabla_z^2 \ell(z^\star; x, y)] = \mathbb{E}_{x \sim q(x), y \sim q(y|x)}[\nabla_z \ell(z^\star; x, y) \nabla_z \ell(z^\star; x, y)^\top]$$

**证明的三个关键步骤**：

1. 因为 $p_z(y|x)$ 是 normalized，$\int p_z(y|x) dy = 1$，对 $z$ 求导得 $\mathbb{E}[\nabla_z \log p_z(y|x)] = 0$（score function 期望为零）

2. 再求一次导，得 $\mathbb{E}[\nabla_z^2 \log p_z(y|x)] = -\mathbb{E}[\nabla_z \log p_z(y|x) \nabla_z \log p_z(y|x)^\top]$

3. 用 NLL $\ell = -\log p_z$ 代入，得 Hessian = Fisher

**直觉解释**：当模型 perfectly fits 数据分布时，loss surface 在数据点上 expectedly flat in the sense of curvature，但 curvature 可以用 score function 的 outer product 来度量。这就是为什么 Fisher 是 "expected curvature" 的另一种表达。

Reference: [Bartlett identities on Wikipedia](https://en.wikipedia.org/wiki/Bartlett%27s_identity)

### 3.4 对角 Fisher 近似 (公式7-9)

虽然在 block output 处用 Fisher 代替 Hessian：

$$\mathbb{E}[\Delta \mathcal{L}_k] \approx \frac{1}{2} \Delta \mathbf{h}_l^\top \mathbf{F}_k^{h_l} \Delta \mathbf{h}_l$$

其中：
$$[\mathbf{F}_k^{h_l}]_{c, c'} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{cal}}}\left[\frac{\partial \mathcal{L}_k(x, y)}{\partial h_{l, c}} \frac{\partial \mathcal{L}_k(x, y)}{\partial h_{l, c'}}\right]$$

但完整 $\mathbf{F}_k^{h_l}$ 仍是 $C \times C$ matrix。作者进一步用 **diagonal empirical Fisher** (公式8)：

$$F_k[l, c] = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{cal}}}\left[\left(\frac{\partial \mathcal{L}_k(x, y)}{\partial h_{l, c}(x)}\right)^2\right] \approx \frac{1}{N} \sum_{n=1}^{N} \left(\frac{\partial \mathcal{L}_k(x_n, y_n)}{\partial h_{l, c}(x_n)}\right)^2$$

变量含义：
- $N$ = calibration samples 数量
- $n$ = sample index
- $c$ = channel index
- $h_{l,c}(x_n)$ = 第 $n$ 个 sample 在 block $l$ 的 channel $c$ 输出

**从 $\mathcal{O}(C^2)$ 降到 $\mathcal{O}(C)$**。对角近似丢弃了 cross-channel 项，等价于把 channel 间 perturbation 视为 locally independent。这对 ranking channel importance 足够了，因为不需要恢复完整 curvature matrix。

代入对角 Fisher，loss 变为 (公式9)：

$$\mathbb{E}[\Delta \mathcal{L}_k] \approx \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{cal}}}\left[\sum_{c=1}^{C} F_k[l, c] \left(\Delta h_{l, c}(\mathbf{x})\right)^2\right]$$

**这就是 paper 的核心**：task $k$ 的 expected loss 变成了 channel-wise 加权的 reconstruction error，每个 channel 的权重就是 Fisher 值 $F_k[l, c]$。

**经验验证** (Figure 3c)：Fisher-predicted loss 和实测 quantization loss 的 Pearson correlation $r = 0.88$，相当 strong。这说明对角 Fisher 是个 useful surrogate。

### 3.5 对角 Fisher 在 VGGT 中的具体实现细节

来自 Appendix B：
- Fisher tensor shape: $3 \times 48 \times 1024$ (tasks × blocks × channels)
- 三个 task：camera, depth, world points (对应 pose_enc, depth, world_points heads)
- 用 64 clips × 4 frames = 256 samples 估计 Fisher（一次估计，calibration 期间固定）
- Forward pass 用 bfloat16，gradient 累积前 cast 到 float32
- Hook 注册在 VGGT aggregator frame 和 global blocks 的 outputs 上

实现公式（Appendix B）：

$$F_{k, \ell, c} = \frac{1}{N} \sum_{n=1}^{N} \sum_t \left(\frac{\partial \mathcal{L}_k}{\partial h_{\ell, t, c}^{(n)}}\right)^2$$

这里还多了一个对 token $t$ 的求和，因为 block output 是 $(|\mathcal{T}|, C)$ 形状，先对 token 维度求和再对 sample 维度求平均。

---

## 4. Fisher 信息整合进 Learnable Affine Transformation

### 4.1 Background: Learnable Affine Transformation (公式10)

这是 FlatQuant / SpinQuant 类方法的基础 trick。在 linear layer $\mathbf{Y}_l = \mathbf{X}_l \mathbf{W}_l$ 中插入一个可逆矩阵 $\mathbf{P}_l$：

$$\mathbf{Y}_l = (\mathbf{X}_l \mathbf{P}_l)(\mathbf{P}_l^{-1} \mathbf{W}_l) = \mathbf{X}_l \mathbf{W}_l$$

数学上等价，但 quantize 的是 $\mathbf{X}_l \mathbf{P}_l$ 和 $\mathbf{P}_l^{-1} \mathbf{W}_l$。通过学习 $\mathbf{P}_l$ 可以让 transformed activation/weight 的分布更 flat、更 friendly 量化。这部分继承自 [FlatQuant (Sun et al., ICML 2025)](https://arxiv.org/abs/2410.09926) 和 [SpinQuant (Liu et al.)](https://arxiv.org/abs/2405.16406)。

### 4.2 标准 calibration loss (公式11) — Uniform 的问题

FlatQuant 原版在 calibration 时优化参数 $\boldsymbol{\theta}_l$，目标是让 quantized block output $\tilde{\mathbf{h}}_l$ 匹配 full-precision $\mathbf{h}_l$：

$$\mathcal{L}_{\text{uni}}^l(\boldsymbol{\theta}_l) = \mathbb{E}_{\mathbf{x}}\left[\frac{1}{|\mathcal{T}_l| C} \sum_{t \in \mathcal{T}_l} \sum_{c=1}^{C} \left(h_{l, t, c}(\mathbf{x}) - \tilde{h}_{l, t, c}(\mathbf{x}; \boldsymbol{\theta}_l)\right)^2\right]$$

变量：
- $\mathcal{T}_l$ = block $l$ 的 tokens 集合
- $|\mathcal{T}_l|$ = tokens 数量
- $C$ = channel 数量
- $t$ = token index
- $c$ = channel index

**问题**：这个 uniform MSE 把所有 channel 等权对待，但实际上不同 channel 的误差对下游 task 的影响差距巨大。

### 4.3 Fisher 加权 calibration loss (公式12-16)

**Step 1: Per-task normalization (公式12)**

$$\overline{F}_k = \frac{1}{L C} \sum_{l'=1}^{L} \sum_{c'=1}^{C} F_k[l', c']$$

变量：
- $\overline{F}_k$ = task $k$ 的全局平均 Fisher 值
- $L$ = calibrated blocks 数量
- $C$ = hidden channel dimension

**目的**：不同 task head 的 raw gradient magnitude 可能差几个数量级（比如 point map head 输出维度远大于 camera pose），不归一化会让 gradient magnitude 大的 task dominate。

**Step 2: Equal task weighting aggregation (公式13)**

$$s[l, c] = \sum_{k=1}^{K} \frac{F_k[l, c]}{\overline{F}_k}$$

变量：
- $K$ = task head 数量（VGGT 中 $K = 3$）
- $s[l, c]$ = aggregated sensitivity score

**Step 3: Per-block normalization (公式14)**

$$w_l[c] = \frac{s[l, c]}{\frac{1}{C} \sum_{c'=1}^{C} s[l, c']}$$

**目的**：让每个 block 的平均权重为 1，使 loss scale 在不同 block 间 comparable。

**Step 4: Floor 项 (公式15)**

$$w_l[c] \gets \max(w_l[c], 0.01)$$

**Intuition**：防止 low-Fisher channel 被 calibration 完全忽略（这会破坏该 channel 的 representation，可能引起 unexpected side effects）。

**最终 Fisher-guided loss (公式16)**：

$$\mathcal{L}_{\text{FGQ}}^l(\boldsymbol{\theta}_l) = \mathbb{E}_{\mathbf{x}}\left[\frac{1}{|\mathcal{T}_l| C} \sum_{t \in \mathcal{T}_l} \sum_{c=1}^{C} w_l[c] \left(h_{l, t, c}(\mathbf{x}) - \tilde{h}_{l, t, c}(\mathbf{x}; \boldsymbol{\theta}_l)\right)^2\right]$$

这就是 FGQ 的核心。本质上是一个 **weighted MSE**，权重来自 diagonal Fisher，捕捉了 channel-level、block-level、task-level 三层 sensitivity。

---

## 5. 实验数据深度解析

### 5.1 Camera Pose Estimation (Table 1, Co3Dv2 & Re10K)

在 W4A8 下各方法都接近 FP16。**关键看 W4A4**：

| Method | Co3Dv2 AUC@15 | Re10K AUC@15 |
|--------|---------------|--------------|
| FP16 | 0.9462 | 0.7818 |
| RTN (4/4) | 0.3950 | 0.2520 |
| QuaRot (4/4) | 0.6471 | 0.3952 |
| FlatQuant (4/4) | 0.8648 | 0.5849 |
| QuantVGGT (4/4) | 0.8840 | 0.6437 |
| **FGQ (4/4)** | **0.8887** | **0.6443** |

**观察**：FGQ 在 camera pose 上的增益相对 modest。这与 Figure 2 一致——pose 是 global prediction，本身对 quantization 不太 sensitive，所以 Fisher-guided weighting 给 pose 的 marginal benefit 有限。FGQ 主要的提升在 dense task 上。

### 5.2 Point Map Reconstruction (Table 2-3)

**7-Scenes W4A4**：
| Method | Acc. mean ↓ | Comp. mean ↓ | N.C. mean ↑ |
|--------|-------------|--------------|-------------|
| FP16 | 0.044 | 0.056 | 0.733 |
| RTN | 0.146 | 0.134 | 0.600 |
| FlatQuant | 0.056 | 0.070 | 0.717 |
| QuantVGGT | 0.053 | 0.085 | 0.719 |
| **FGQ** | **0.048** | **0.059** | **0.723** |

**ETH3D W4A4**：
| Method | Acc. mean ↓ | Comp. mean ↓ | N.C. mean ↑ |
|--------|-------------|--------------|-------------|
| FP16 | 0.263 | 0.288 | 0.846 |
| QuantVGGT | 0.312 | 0.305 | 0.832 |
| **FGQ** | **0.275** | **0.279** | **0.834** |

**关键发现**：在 ETH3D 上，FGQ 把 Acc. mean 从 QuantVGGT 的 0.312 降到 0.275，相对改进约 12%。Completeness 从 0.305 降到 0.279。**这就是 paper 标题说 "up to 39% relative improvement" 的来源**——主要来自 dense point map reconstruction。

**DTU W4A4**：
| Method | Acc. mean ↓ | Comp. mean ↓ | N.C. mean ↑ |
|--------|-------------|--------------|-------------|
| FP16 | 1.308 | 1.929 | 0.665 |
| QuantVGGT (4/4) | 1.488 | 1.933 | 0.669 |
| **FGQ (4/4)** | **1.420** | **1.879** | **0.669** |

### 5.3 Depth Estimation (Table 4, KITTI)

**KITTI Mono W4A4**：
| Method | AbsRel ↓ | SqRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|--------|----------|---------|--------|-----------|
| FP16 | 0.092 | 0.459 | 3.902 | 0.936 |
| QuantVGGT | 0.088 | 0.446 | 3.842 | 0.938 |
| **FGQ** | **0.079** | **0.389** | **3.642** | **0.951** |

FGQ 在 AbsRel 上从 0.088 → 0.079，约 10% 相对改进；SqRel 从 0.446 → 0.389，约 13% 改进。这进一步证实：dense geometric task (depth) 从 Fisher-guided calibration 中获益最多。

### 5.4 Ablation Study (Table 5) — Multi-task Fisher 的互补性

Co3Dv2 (主要测 camera pose)：
| Fisher 配置 | AUC@15 |
|------------|---------|
| plain (无 Fisher) | 0.8825 |
| point | 0.8868 |
| point + depth | 0.8875 |
| point + depth + camera | **0.8952** |

DTU (主要测 point map)：
| Fisher 配置 | Acc. mean ↓ |
|------------|-------------|
| plain | 1.537 |
| camera | 1.465 |
| camera + depth | 1.458 |
| camera + depth + point | **1.446** |

**重要观察**：当 Fisher 包含与 evaluation metric aligned 的 task loss 时，增益最大（Co3Dv2 加 camera Fisher 后 jump 最大；DTU 加 point Fisher 后 improvement 最大）。但 multi-task Fisher 始终优于 single-task Fisher，说明不同 task loss 提供了**互补的 channel sensitivity 信号**。

这个结果呼应了 multi-task learning 中的 gradient surgery / PCGrad 类工作：不同 task 的 gradient 提供不同视角的 importance 信息。

### 5.5 π³ 模型泛化 (Table 7)

在 $\pi^3$（最新的 feed-forward 3D 模型）上验证泛化性。**Co3Dv2 AUC@3**：
| Method | AUC@3 |
|--------|-------|
| FP16 | 0.5140 |
| RTN (4/4) | 0.0246 (崩溃) |
| FlatQuant (4/4) | 0.2686 |
| **FGQ (4/4)** | **0.5032** (接近 FP16!) |

FGQ 把 AUC@3 从 FlatQuant 的 0.2686 提升到 0.5032，几乎完全恢复 FP16 性能。这说明 FGQ 不依赖 VGGT 特定结构，可泛化到其他 feed-forward 3D 模型。

Reference: [$\pi^3$ paper](https://arxiv.org/abs/2507.13347)

### 5.6 Calibration Overhead (Table 6)

| Method | GPU Memory (GB) | GPU Time (hours) | AUC@15 | DTU Acc. mean |
|--------|-----------------|------------------|--------|---------------|
| FlatQuant | 3.54 | 0.87 | 0.8825 | 1.537 |
| FGQ | 3.62 (+0.08) | 0.92 (+0.05) | 0.8952 (+0.0127) | 1.446 (-0.091) |

**Overhead 极小**：内存仅增加 2.26%，时间增加 5.73%，但 DTU Acc. mean 改进 5.9%。性价比很高。

### 5.7 Inference Efficiency (Section 4.4, Figure 4)

在 NVIDIA A100 40G GPU 上 W4A4 真实低比特推理：
- **Parameter loading**: RTN 3.41× / FGQ 3.42× (相同，因为 4-bit storage)
- **Compute kernel**: RTN 6.01× / FGQ 4.67× (FGQ 因为额外的 affine transformation 而慢)
- **Full block level**: RTN 2.01× / **FGQ 1.81×**

**Trade-off**：FGQ 比 RTN 慢约 10%，但精度大幅领先。这是 accuracy-latency 的合理取舍。

---

## 6. 关键 hyperparameters 与实现细节 (Appendix B)

- **Calibration data**: Co3Dv2, 64 clips × 4 frames, batch size 2 clips (8 frames per mini-batch)
- **Optimizer**: AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, weight decay 0.01
- **Learning rate**: $10^{-2}$ for transformation matrices/scales, $10^{-1}$ for clipping factors (10× larger)
- **Schedule**: Cosine, $T_{\text{max}} = \text{epochs} \times (\text{nsamples} // \text{cali\_bsz})$, $\eta_{\text{min}} = 10^{-5}$
- **Per block**: 15 epochs, 480 optimizer steps
- **Total**: 48 blocks × 480 steps = **23,040 optimizer steps** for full W4A4 calibration
- **Quantization**: symmetric, weight per-output-channel, activation per-token, learnable clipping
- **Quantized layers**: attention projections (Q, K, V, output), MLP linears in aggregator frame + global blocks
- **Not quantized**: DINOv2/patch embedding, camera/depth/point/track heads, normalization, special tokens

---

## 7. 更宽的技术 context 与联想

### 7.1 Fisher 信息在 NN 量化/剪枝中的 lineage

- **Optimal Brain Damage (LeCun et al., 1989)**: 首次用 diagonal Hessian saliency 做 pruning
- **Optimal Brain Surgeon (Hassibi & Stork, 1992)**: 用完整 Hessian 做更精确的 pruning
- **K-FAC (Martens & Grosse, 2015)**: Kronecker-factored Hessian/Fisher approximation，让二阶方法 practical for deep nets
- **Fisher-aware quantization for DETR (Yang et al., 2024)**: 把 Fisher 用在 object detection 量化上，处理 critical-category objectives。FGQ 的直接前驱之一。

### 7.2 Affine transformation 量化的 lineage

- **SmoothQuant (Xiao et al., 2022)**: 固定 per-channel scale，把 activation outlier 迁移到 weight
- **QuaRot (Ashkboos et al., 2024)**: 用 random Hadamard rotation 压 activation outlier
- **SpinQuant (Liu et al., 2024)**: 学 rotation matrix，用 Cayley parameterization 保证正交性
- **FlatQuant (Sun et al., 2025)**: 学 lightweight per-layer affine transformations，flatten weight 和 activation 分布。**FGQ 就是基于 FlatQuant 的 calibration objective 改造**

Reference: [SmoothQuant](https://arxiv.org/abs/2211.10438), [QuaRot](https://arxiv.org/abs/2404.02856)

### 7.3 为什么 VGGT 量化特别难？

VGGT 有几个独特挑战，传统 LLM quantization 方法不够用：
1. **Special tokens**: camera token, register token 产生 heavy-tailed activation，对低 bit 量化极不友好。这是 [QuantVGGT (Feng et al., 2025)](https://arxiv.org/abs/2509.21302) 解决的。
2. **3D 几何的多模态输出**: depth, pose, point map scale 完全不同，evaluation metric 不同
3. **跨视图信息聚合**: GlobalAttention 让 quantization error 跨视图传播并 accumulate
4. **Shared backbone multi-task**: 一个 backbone 服务多个 task，导致 paper 标题所说的 "Not All Tasks Quantize Equally" 问题

### 7.4 公式 (16) 与 weighted MSE 的关系

公式 (16) 本质上是个 **weighted block-output reconstruction loss**：

$$\mathcal{L}_{\text{FGQ}} = \text{WeightedMSE}(\mathbf{h}_l^{\text{quant}}, \mathbf{h}_l^{\text{fp}}, \text{weights} = w_l[c])$$

这种 weighted reconstruction 在量化文献中有先例：
- **GPTQ**: 用 diagonal Hessian 权重 per-column update
- **AWQ**: 用 activation magnitude 作为 weight importance proxy
- **SparseGPT**: Hessian-weighted sparse regression

FGQ 的差异：用 **diagonal Fisher aggregated across tasks** 作为 weight，比 magnitude 更 theoretically grounded，且 multi-task aware。

Reference: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978)

### 7.5 对角 Fisher 的局限与改进方向

对角近似丢弃了 cross-channel correlation。在什么情况下这个近似会失败？
- 当 channel 间高度相关（例如 attention head 内部）
- 当 quantization error 在 channel 间耦合（例如 per-token 量化共享 scaling factor）

更精细的替代方案（paper 没做但 future work 方向）：
- **Block-diagonal Fisher** (例如 per-head Fisher)
- **K-FAC style** Kronecker 分解
- **Low-rank Fisher** approximation

但 trade-off 是计算成本和实现复杂度。

### 7.6 Calibration data 选择的影响

Paper 在 Appendix E 提到 limitation：固定 calibration protocol，没有优化 calibration data 的 category balance, scene diversity, view distribution。这是一个 orthogonal 但 important 的方向。

类似 LLM 量化中 calibration data 影响的研究：
- **AWQ** 显示 calibration data 不需要太多，少量就能 capture activation distribution
- **GPTQ** 用 128 random samples 效果就稳定
- 但 3D vision 中，camera viewpoint 分布、scene 类型对 calibration 影响更大

### 7.7 Multi-task gradient 互补性的理论解释

Ablation 显示 multi-task Fisher 优于 single-task Fisher。这背后的 intuition：

不同 task head 从同一 backbone 抽取不同信息，gradient 反映了 "这个 channel 对该 task 重要" 的信号。如果只用 single-task Fisher，可能会过度保护对该 task 重要的 channel，但忽视其他 channel，导致其他 task 性能下降。

Multi-task Fisher 是个 **Pareto-improving** 的设计——通过均衡不同 task 的 sensitivity signal，让 calibration 找到一个对所有 task 都还不错的 quantization configuration。

这与 multi-task learning 中的 **gradient conflict resolution**（如 PCGrad, CAGrad, Nash-MTL）有思想上的呼应：好的 multi-task method 应该 explicitly account for 不同 task 间的 trade-off。

Reference: [PCGrad (Yu et al., NeurIPS 2020)](https://arxiv.org/abs/2001.06782)

---

## 8. Intuitive 总结

FGQ 的核心 insight 可以一句话概括：**在 multi-task model 中，calibration loss 应该 reflect 不同 task/block/channel 对 quantization 的真实 sensitivity，而不是 uniform MSE**。

数学上，这个 sensitivity 通过 diagonal Fisher information 描述——它是 Hessian 的 tractable surrogate，可以从 first-order gradient 单次 backward pass 估计。Pearson $r = 0.88$ 证明了它对 quantization loss 的预测能力。

Engineering 上，FGQ 是个 drop-in 改进：仅替换 FlatQuant 的 calibration objective，overhead 不到 6%，但在 dense geometric task (point map, depth) 上获得显著改进（最高 39% 相对提升）。

Conceptually，这篇 paper 是 **"二阶信息指导一阶量化 calibration"** 思想在 multi-task 3D vision 模型上的成功应用，给后续 multi-modal / multi-task 大模型量化提供了一个清晰的 recipe：**用 task-aware Fisher 加权 calibration loss**。

---

## 9. 你可能想深入的方向

1. **Off-diagonal Fisher** 是否值得？哪些 block/head 的 cross-channel correlation 最强？
2. **Fisher 的 dynamic update**：calibration 期间 Fisher 是否应该 update？目前是 fixed。
3. **MXFP4 / FP4 formats** 下的 Fisher weighting 是否需要 re-tune？（Paper 在 limitations 提到）
4. **与 token merging 方法**（FastVGGT, LiteVGGT）的联合优化——能否在 token reduction 后再量化？
5. **Fisher for mixed-precision bit allocation**：能否用 Fisher 决定哪些 block 用 8-bit、哪些用 4-bit？这是更 fine-grained 的 sensitivity-aware precision allocation。

References:
- [FastVGGT](https://arxiv.org/abs/2509.02560)
- [LiteVGGT](https://arxiv.org/abs/2512.04939)
- [FlashVGGT](https://arxiv.org/abs/2512.01540)
- [AVGGT](https://arxiv.org/abs/2512.02541)
- [QuantVGGT](https://arxiv.org/abs/2509.21302)
- [FGQ GitHub repo](https://github.com/ypzhng/FGQ)

Andrej, 这篇 paper 的核心数学 trick (diagonal Fisher 替代 Hessian) 在 NN 历史上用过很多次，但**应用到 multi-task 3D reconstruction 量化上、并 explicit 处理 task-level sensitivity asymmetry** 是新的 contribution。它的 limitation 也很清楚：只做 INT4，没有探索 lower-bit 或 floating-point 4-bit；calibration data 协议固定。但作为一个 principle demonstration，它清楚地说明了 multi-task quantization 不应该 uniform。
