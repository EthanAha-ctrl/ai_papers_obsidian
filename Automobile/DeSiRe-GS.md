---
source_pdf: DeSiRe-GS.pdf
paper_sha256: b7256a776a002d22ef3331a6369980b8a5006c36cb638f6fac16d712557f8b78
processed_at: '2026-08-03T20:00:51-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeSiRe-GS 人话版

## 一句话概括

**让 3DGS 自己暴露动态物体在哪，然后把这些地方用 4D Gaussian 重新建模，顺带加一堆正则化让几何质量过得去。**

---

## 要解决什么问题

自动驾驶场景重建，三个老大难：

1. **有动的物体**：原始 3DGS 没时间维度，一辆车开过去，它就在车的整条轨迹上撒一堆 Gaussian，渲染出来就是半透明拖影（ghost artifact）。

2. **视角稀疏**：不像 object-centric 那种 100 张图均匀包围，driving 就 50 帧前向 3 相机，Gaussian 很容易过拟合 training views，图看着 OK 但几何是塌的。

3. **之前 SOTA 都偷懒用 3D bbox**：StreetGaussians / OmniRe / HUGS 都依赖检测器 + tracker 给的 bbox，把动态物体框出来单独建模。问题是 bbox 在实际中不总是有，pipeline 还复杂。

DeSiRe-GS 想干的事情：**不用任何 3D 标注，纯 self-supervised，把静态动态分开，同时把表面重建得像样。**

---

## 核心洞察（最漂亮的地方）

**3DGS 拟合不好的地方，就是动态物体。**

这个观察乍看 trivial，但细想很优雅：
- 3DGS 只能建模静态部分
- 所以渲染图和 GT 在静态区域一致，在动态区域不一致
- 不一致 = 动态物体位置

等于 3DGS 自带一个 "dynamic detector"，failure mode 直接变成了 supervision signal。类似 RobustNeRF 用 uncertainty，EmerNeRF 用 scene flow，但 DeSiRe-GS 这个更直接。

---

## Pipeline 拆解

两阶段，coarse-to-fine。

### Stage I：找出动态区域在哪

训练一个 vanilla 3DGS，同时训练一个 MLP decoder 预测 motion mask。

具体怎么算 mask：
1. 用预训练的 FiT3D（GS-aware 的 DINOv2）提取 rendered image 和 GT image 的特征
2. 算 per-pixel cosine distance $D = (1-\cos(\hat{F}, F))/2$
3. 训练一个 MLP decoder $\delta$，loss 是 $\delta \odot D$ —— 让 decoder 在 $D$ 高的地方输出低值
4. 二值化得到 mask $M = \mathbb{I}(\delta > \varepsilon)$

**关键 trick**：渲染 loss 只在 mask 标为静态的区域算（masked rendering）：
$$\mathcal{L}_{masked-render} = M \odot \|\hat{I} - I\|$$

这造成正反馈循环：
- Mask 动态区域 → 3DGS 不拟合动态区域
- 动态区域 ghost 更明显 → mask 更准
- 越迭代越好

这是个 bootstrapping 思路，很 elegant。

### Stage II：把 2D mask 信息蒸到 4D Gaussian 里

用 PVG 作为 4D 表示。PVG 的核心是让 Gaussian 在时间维度上"振动 + 衰减"：

$$\tilde{\mu}(t) = \mu + \frac{l}{2\pi}\sin\left(\frac{2\pi(t-\tau)}{l}\right) \cdot \mathbf{v}$$

$$\tilde{o}(t) = o \cdot e^{-\frac{1}{2}(t-\tau)^2/\beta^2}$$

人话解释：
- 每个 Gaussian 有"生命周期峰值" $\tau$，在 $\tau$ 时刻最活跃
- 位置在 $\mu$ 附近以速度 $\mathbf{v}$ 振动（sin 振幅 $\frac{l}{2\pi}\mathbf{v}$）
- opacity 在 $\tau$ 处最大，远离 $\tau$ 按 Gaussian 衰减
- 多了三个 learnable 参数：$\tau, \beta, \mathbf{v}$

**问题**：PVG 直接学，velocity map 在静态区域（路面、建筑）也有非零 velocity，没人告诉它"这里该是 0"。

**Solution**：用 Stage I 的 mask 监督 velocity map：
$$\mathcal{L}_v = \mathbf{V} \odot M$$

在 mask 标为静态的像素位置，惩罚 velocity 不为 0。梯度通过 alpha blending 反传到每个 Gaussian 的 $\mathbf{v}$，等于把 2D image space 的 motion info 蒸到 3D Gaussian space。

最后静态/动态分解就是 threshold $|\mathbf{v}|$。

---

## 几何正则化（为了 sparse view 不塌）

这块比较工程化，但每个都有道理。

### 1. Flattening：把 3D ellipsoid 压成 2D disk

物体表面是 2D manifold，所以 Gaussian 应该扁平贴在表面上：
$$\mathcal{L}_s = \|\min(s_1, s_2, s_3)\|$$

直接最小化最短轴的 scale。

### 2. Normal 直接从 scale 推导（创新点）

之前方法（2DGS）给每个 Gaussian 额外 append 一个 normal 向量，独立学习。问题是 normal 和 scale 解耦，优化时互不影响。

DeSiRe-GS 的 trick：
$$\mathbf{n} = \mathbf{R} \cdot \arg\min(s_1, s_2, s_3)$$

Gaussian 压扁后，最薄那个方向就是法向。这样 normal loss 的梯度会反传到 scale 和 rotation，让 Gaussian 真正"贴"在表面上。

### 3. Giant Gaussian 惩罚

3DGS / PVG 会产生 oversized Gaussian（很大但 opacity 低，图看着没事，但几何被破坏）：
$$\mathcal{L}_g = s_g \cdot \mathbb{I}(s_g > \epsilon), \quad s_g = \max(s_1, s_2, s_3)$$

只惩罚超过阈值的，避免干扰小 Gaussian。

### 4. Cross-view temporal-spatial consistency（最关键）

稀疏 view 容易过拟合，单视角 photometric loss 在远距离 textureless 区域不可靠。

核心假设：**静态区域在不同时间、不同视角下应该几何一致**。

公式逻辑：
1. Reference frame 像素 $(u_r, v_r)$，深度 $d_r$，投影到 neighbor frame 得到 $(u_n, v_n)$
2. 在 neighbor frame 查询 depth $d_n$，再投影回 reference frame 得到 $(u_{nr}, v_{nr})$
3. 如果深度一致，$(u_{nr}, v_{nr})$ 应该回到 $(u_r, v_r)$
4. Loss：$\mathcal{L}_{uv} = \|(u_r, v_r) - (u_{nr}, v_{nr})\|_2$

注意 loss 在 pixel space 算，不在 3D space 算，避免深度尺度问题。

这个对 depth L1 提升巨大：ablation 显示从 0.115 → 0.071（降低 38%）。

---

## 实验结果

### Waymo / KITTI 主表

- 比 PVG 提升约 1.1 PSNR（significant，因为 PVG 已经很强）
- FPS 36（vs PVG 50），因为多渲染 normal/depth/velocity
- NeRF 方法（EmerNeRF, SUDS）FPS < 1，远不及 GS

### 与 bbox-dependent 方法比较

DeSiRe-GS **不用 bbox，却超过了用 bbox 的 HUGS 和 StreetGS**。仅次于 OmniRe（但 OmniRe 用了 SMPL 等额外建模，bbox 标注完整）。这是相当强的结果。

### Ablation 几个关键发现

1. **Cross-view consistency 对 depth 提升最大**（38% 降误差）
2. **Motion mask 对整体提升显著**（证明 Stage I → Stage II 蒸馏必要）
3. **Scale reg 对 PSNR 影响小但对几何质量重要**（PSNR 不敏感不代表几何好）
4. **FiT3D 优于 DINOv2**（GS-aware feature 在 driving scene 更准）

---

## 我的评价

### 优点
- **核心 insight 漂亮**：3DGS failure → supervision signal，emergent behavior 的好例子
- **Pipeline 设计合理**：Stage I (mask) → Stage II (4D modeling)，coarse-to-fine
- **几何正则齐全**：覆盖了 sparse view driving 的核心问题
- **FiT3D vs DINOv2 ablation** 是个有用经验

### 局限
- **PVG 的 sine 振动假设太强**：真实车辆运动更像 linear motion 不是周期振动。50 帧短序列可能没暴露问题。
- **Cross-view 假设完全静态**：如果 mask 漏检动态物体，cross-view loss 会错误地把动态物体"钉"住。
- **两阶段训练复杂**：30K + 50K = 80K iter，还要切换 loss 和模块，工程上 fragile。
- **依赖 LiDAR 初始化**：$6 \times 10^5$ 个点来自 LiDAR，纯 vision 不适用。
- **FiT3D 依赖**：FiT3D 本身是 GS 训练的，有点循环依赖味道。

### 一句话 essence

> **"3DGS 拟合不好的地方就是动态物体 —— 用它的 failure mode 做 self-supervised detector，再用 4D Gaussian 重建。"**

这是 self-supervised learning 里很 elegant 的思路：把 model 的 limitation 转成 feature。

---

# DeSiRe-GS 深度解析：4D Street Gaussians for Static-Dynamic Decomposition

这是一篇来自 UC Berkeley 的工作（一作 Chensheng Peng, Chengwei Zhang，作者包括 Wei Zhan, Masayoshi Tomizuka 等自动驾驶领域的资深研究者），发表于 2024 年底。论文的核心是 **在自动驾驶场景下，无 3D bounding box 标注的情况下，实现高质量的静态-动态分解和表面重建**。让我从直觉开始构建，然后深入到每个技术细节。

---

## 1. 直觉构建：为什么需要 DeSiRe-GS？

### 1.1 问题本质

自动驾驶场景重建有几个非常本质的难点，论文抓住的就是这几个：

**Difficulty 1 - Dynamic Objects**：原始 3DGS [Kerbl et al., 2023] 是 time-independent 的，每个 Gaussian 只有静态属性 $\{\mu, s, r, o, c\}$，对一辆开过的车会"拖影"（ghost-like artifacts）。因为同一个 Gaussian 要在不同时间点拟合车的不同位置，只能糊成一团。

**Difficulty 2 - Sparse Views**：Object-centric 重建（比如 NeRF Synthetic）通常有 100+ 视角均匀覆盖；自动驾驶 50 帧，前向 3 相机，视角高度冗余且 forward-facing。这意味着 photometric loss 很容易 overfit 到 training views，几何质量一塌糊涂。

**Difficulty 3 - Bounding Box 依赖**：之前 SOTA 的方法 StreetGaussians [Yan et al., 2024]、OmniRe [Chen et al., 2024]、HUGS [Zhou et al., 2024] 都用 3D bbox 来"作弊"——把动态物体框出来单独建模，问题被显著简化。但 bbox 在实际中需要 tracker, 检测器，pipeline 复杂。

DeSiRe-GS 的洞察是：**3DGS 自己就是一种天然的 dynamic detector**。因为 3DGS 只能拟合静态部分，所以"3DGS 拟合得不好的地方"就是动态区域。

---

## 2. 方法详解：两阶段 Pipeline

### 2.1 整体架构

```
Stage I: 3DGS + Motion Mask Decoder (30K iter)
    ↓ 提取 2D motion masks
Stage II: PVG + 几何正则 + Cross-view 一致性 (50K iter)
    ↓ 输出 4D 场景 + 表面
```

这是典型的 coarse-to-fine 思路：Stage I 把"哪里是动的"这件事搞清楚，Stage II 再把运动信息"蒸"到 4D Gaussian space 里去。

---

### 2.2 PVG 复习：动态 Gaussian 的数学形式

DeSiRe-GS 直接采用 PVG [Chen et al., 2023] 作为 4D 表示。先理解 PVG 才能理解 Stage II。

**原始 3DGS**:
$$G(\mathbf{x}) = \exp\left\{ -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) \right\} \tag{1}$$

- $\mathbf{x} \in \mathbb{R}^3$：query 的 3D 点
- $\boldsymbol{\mu} \in \mathbb{R}^3$：Gaussian 中心
- $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$：协方差矩阵，$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$，其中 $\mathbf{S}$ 为 diagonal scaling matrix（参数化成 vector $\mathbf{s} = (s_1, s_2, s_3)$），$\mathbf{R}$ 为 rotation matrix（参数化成 quaternion $\mathbf{r} \in \mathbb{R}^4$）

**PVG 修改**：让 Gaussian 在时间维度上"振动"+ 衰减：

$$\tilde{\boldsymbol{\mu}}(t) = \boldsymbol{\mu} + \frac{l}{2\pi} \cdot \sin\left(\frac{2\pi(t-\tau)}{l}\right) \cdot \mathbf{v} \tag{3}$$

- $t \in [0,1]$：归一化时间（论文把 50 帧的时间跨度 rescale 到 [0,1]）
- $\tau$：life peak（"生命周期峰值"），Gaussian 在此时刻"最活跃"
- $l$：oscillation period（预定义场景级参数，控制振动周期）
- $\mathbf{v}$：peak velocity（瞬时速度，learnable，每个 Gaussian 独立）

直觉：位置 $\boldsymbol{\mu}$ 在以 $\tau$ 为中心、振幅 $\frac{l}{2\pi}\mathbf{v}$ 的 sine 函数上振动，模拟物体运动。注意这不是线性运动，而是周期振动——这是 PVG 的一个 trick，避免显式 deformation field。

$$\tilde{o}(t) = o \cdot e^{-\frac{1}{2}(t-\tau)^2 \beta^{-2}} \tag{4}$$

- $o$：base opacity（静态 opacity）
- $\beta$：decay rate（learnable，每个 Gaussian 独立）
- Gaussian 的 opacity 在 $\tau$ 处达到峰值 $o$，远离 $\tau$ 时按 Gaussian 形式衰减

直觉：每个 PVG 有"出生-活跃-消失"的生命周期，类似 particle system。这是为了处理 transient objects（一辆车开进又开出视野）。

**完整 PVG 参数**：
$$\mathscr{G}(t) = \{\tilde{\mu}(t), s, r, \tilde{o}(t), c, \tau, \beta, \mathbf{v}\} \tag{5}$$

对比 3DGS 多了 $\tau, \beta, \mathbf{v}$ 三个 learnable 参数。

---

### 2.3 Stage I: Motion Mask Extraction

这是论文最有趣的部分，也是 self-supervised 的关键。

#### 2.3.1 核心观察

3DGS 在动态场景中，**静态区域会被清晰重建**，**动态区域会出现 ghost artifacts**（一辆车开过，3DGS 在它经过的整条轨迹上都铺一些低 opacity Gaussian，渲染出来就是半透明的拖影）。

所以：**3DGS 渲染图 与 GT 的"特征差异"≈ 动态物体的位置**。

#### 2.3.2 具体公式

Step 1: 用预训练 foundation model 提取特征（论文 ablation 比较了 DINOv2 和 FiT3D，FiT3D 更好）。

- $\hat{F}$：rendered image $\hat{I}$ 的特征
- $F$：GT image $I$ 的特征

Step 2: 计算像素级不相似度：
$$D = \frac{1 - \cos(\hat{F}, F)}{2} \tag{6}$$

- $D \in \mathbb{R}^{H \times W}$：每像素 dissimilarity，范围 $[0, 1]$
- $D \to 0$：相似，静态区域
- $D \to 1$：不相似，动态区域
- $\cos(\cdot, \cdot)$：cosine similarity

Step 3: 不是直接阈值化 $D$（因为 foundation model 在 road/sky 等区域 noisy），而是训练一个 MLP decoder $\delta$：

$$\mathcal{L}_{dyn} = \delta \odot D \tag{7}$$

- $\delta \in \mathbb{R}^{H \times W}$：decoder 输出的 dynamicness
- $\odot$：element-wise multiplication
- 直觉：让 decoder 在 $D$ 高的地方输出低值（minimize loss）—— 等价于 decoder 学着"承认动态区域是动态的"

Step 4: 二值化：
$$M = \mathbb{I}(\delta > \varepsilon) \tag{8}$$

- $M$：最终 motion mask
- $\varepsilon$：固定阈值
- $\mathbb{I}$：indicator function

#### 2.3.3 关键 trick：Masked Rendering

$$\mathcal{L}_{masked-render} = M \odot \|\hat{I} - I\| \tag{9}$$

只在静态区域计算渲染 loss！这造成了正反馈循环：
1. Mask 出动态区域 → 3DGS 不再尝试拟合动态区域
2. 动态区域 ghost artifacts 更明显 → mask 更准确
3. 反复迭代

这个 bootstrapping 思路很关键，类似于 RobustNeRF, EmerNeRF 的思路，但实现更简洁。

#### 2.3.4 为什么用 FiT3D 而非 DINOv2？

论文在 ablation 中明确指出 DINOv2 在 road / sky 上 noisy，而 **FiT3D** [Yue et al., ECCV 2024] 是用 Gaussian Splatting fine-tune 过的 DINOv2，3D-aware features 更 clean。这是 ablation 里 (b) 项的关键发现。

---

### 2.4 Stage II: 蒸馏到 4D Gaussian Space

#### 2.4.1 问题：PVG 直接学的 velocity map 是 noisy 的

如果直接用 PVG + image loss 训练，得到的 velocity map $\mathbf{V} \in \mathbb{R}^{H \times W}$ 在静态区域（road, building）也有非零 velocity，因为没有任何信号告诉它"这里应该是 0"。

#### 2.4.2 Solution: 用 Stage I mask 监督 velocity

$$\mathcal{L}_v = \mathbf{V} \odot M \tag{10}$$

- $\mathbf{V}$：渲染出来的 velocity map（从每个 Gaussian 的 $\mathbf{v}$ 通过 alpha blending 得到）
- $M$：Stage I 学到的 mask
- 直觉：在 mask 标为静态的像素位置，惩罚 velocity 不为 0

这相当于把 2D image space 的 motion information "propagate" 到 3D Gaussian space 的 $\mathbf{v}$ 参数上。因为渲染是 differentiable 的，梯度从 $\mathcal{L}_v$ 经过 alpha blending 反传到每个 Gaussian 的 $\mathbf{v}$。

最终的静态/动态分解：对每个 Gaussian，threshold $|\mathbf{v}|$ 即可。

---

### 2.5 Surface Reconstruction：几何正则化

这是论文的另一大块贡献。原始 3DGS / PVG 在稀疏 view 下几何质量很差，作者加了几条正则化。

#### 2.5.1 Flattening 3D Gaussian（受 2DGS 启发）

3D Gaussian 是 ellipsoid（三个轴 $s_1, s_2, s_3$）。2DGS [Huang et al., 2024] 的洞察是：物体表面应该是 2D manifold，所以应该把 Gaussian 压扁成 disk。

$$\mathcal{L}_s = \|\min(s_1, s_2, s_3)\| \tag{11}$$

- $s_1, s_2, s_3$：scale vector 的三个分量
- 直接最小化最短轴的 scale
- 直觉：让 Gaussian 在一个方向上越来越薄，最终接近 2D disk

#### 2.5.2 Normal Derivation（关键创新）

之前的方法（如 2DGS）是给每个 Gaussian **额外 append 一个 normal vector** $\mathbf{n}_i \in \mathbb{R}^3$，作为独立 learnable parameter。但这样 normal 和 scale 是解耦的，优化时不会相互影响。

DeSiRe-GS 的 trick：**normal 直接从 scale 导出**：

$$\mathbf{n} = \mathbf{R} \cdot \arg\min(s_1, s_2, s_3) \tag{12}$$

- $\mathbf{R}$：rotation matrix
- $\arg\min(s_1, s_2, s_3)$：scale 最小的那个轴的方向（在 local frame 中是 one-hot 向量，比如 $(0, 0, 1)$）
- 旋转到 world frame 得到 normal

直觉：Gaussian 被压扁后，"最薄"那个方向就是法向。这样 normal 监督的梯度会反传到 scale 和 rotation 上，让 Gaussian 真正"贴"在表面上。

$$\mathcal{L}_n = \|\mathcal{N} - \hat{\mathcal{N}}\|_2 \tag{13}$$

- $\mathcal{N} \in \mathbb{R}^{H \times W}$：渲染的 normal map
- $\hat{\mathcal{N}}$：pseudo-GT normal，来自预训练模型 OmniData [Eftekhar et al., 2021]（从单目图像预测 normal）

#### 2.5.3 Giant Gaussian Regularization

观察：3DGS / PVG 在 unbounded driving 场景会产生 oversized Gaussian（很大但 opacity 很低），渲染图看起来 OK，但几何被严重破坏（Floater）。

$$s_g = \max(s_1, s_2, s_3); \quad \mathcal{L}_g = s_g \cdot \mathbb{I}(s_g > \epsilon) \tag{14}$$

- $s_g$：最大 scale 方向
- $\epsilon$：预定义阈值
- 只有当 $s_g$ 超过阈值时才惩罚（避免对小 Gaussian 产生干扰）

---

### 2.6 Temporal Spatial Cross-view Consistency

这是为了对抗 sparse view overfitting 的核心模块。

#### 2.6.1 核心思想

静态区域在不同时间、不同视角下应该几何一致。给定 reference frame 中的一个静态像素 $(u_r, v_r)$，深度 $d_r$，应该能在 neighboring frame 中找到对应点，且 3D 位置一致。

#### 2.6.2 数学公式

**Forward projection** (reference → neighbor):
$$[u_n, v_n, 1]^\top = K T_n T_r^{-1} \left( d_r \cdot K^{-1} [u_r, v_r, 1]^\top \right) \tag{15}$$

- $K$：camera intrinsics (3×3)
- $T_r$：reference frame 的 extrinsics（world-to-camera）
- $T_r^{-1}$：camera-to-world
- $T_n$：neighbor frame 的 extrinsics
- $d_r$：reference frame 的深度
- $K^{-1}[u_r, v_r, 1]^\top$：像素 $(u_r, v_r)$ 在归一化相机坐标下的 ray direction
- $d_r \cdot K^{-1}[u_r, v_r, 1]^\top$：3D 点（在 reference camera frame 下）
- $T_n T_r^{-1}$：reference camera frame → neighbor camera frame
- 整个 RHS：3D 点投影到 neighbor 的像素坐标
- LHS $[u_n, v_n, 1]^\top$：neighbor frame 中对应像素

**Backward projection** (neighbor → reference, round-trip check):
$$[u_{nr}, v_{nr}, 1]^\top = K T_r T_n^{-1} \left( d_n \cdot K^{-1}[u_n, v_n, 1]^\top \right) \tag{16}$$

- $d_n$：在 neighbor frame 查询到的 depth（由当前 Gaussian 渲染）
- 如果深度一致，则 $[u_{nr}, v_{nr}]$ 应该回到 $[u_r, v_r]$

**Loss**:
$$\mathcal{L}_{uv} = \|(u_r, v_r) - (u_{nr}, v_{nr})\|_2 \tag{17}$$

注意：这里 loss 是在 pixel space 算的，不是在 3D space 算的。这是因为 pixel space 的误差更 stable，避免深度尺度问题。

#### 2.6.3 实现细节

- 在 Stage II 第 20K iterations 之后才开启（前面 Gaussian 还在粗调，开启太早会损坏训练）
- 每次采样 102400 pixels
- 用 nearest neighboring view（最大 overlap）

---

### 2.7 总 Loss

**Stage I Loss**:
$$\mathcal{L}_{stage1} = M \odot \mathcal{L}_I + \mathcal{L}_{dyn} \tag{19}$$

其中：
$$\mathcal{L}_I = (1-\lambda_{ssim})\|I - \tilde{I}\|_1 + \lambda_{ssim} \mathrm{SSIM}(I, \tilde{I}) \tag{18}$$

- $I$：GT image
- $\tilde{I}$：rendered image
- $\lambda_{ssim}$：SSIM 权重

**Stage II Loss**:
$$\mathcal{L}_{stage2} = \mathcal{L}_I + \mathcal{L}_D + \mathcal{L}_n + \mathcal{L}_v + \mathcal{L}_s + \mathcal{L}_g + \mathcal{L}_{uv} \tag{22}$$

七项 loss，比较复杂。

**Depth supervision**:
$$\mathcal{L}_D = \|\mathcal{D} - D_{gt}\|_1 \tag{21}$$

- $\mathcal{D}$：rendered depth map
- $D_{gt}$：sparse LiDAR projection

**Rendering multiple attributes** via alpha blending:
$$\{\mathcal{D}, \mathcal{N}, \mathcal{V}\} = \sum_{i \in N} \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \{d_i, n_i, v_i\} \tag{20}$$

- $d_i, n_i, v_i$：第 $i$ 个 Gaussian 的 depth, normal, velocity
- $\alpha_i$：第 $i$ 个 Gaussian 的 alpha
- 与 color 渲染同样的 alpha blending 公式

---

## 3. 实验数据深度解读

### 3.1 Waymo / KITTI 主表 (Table 1)

| Method | Waymo PSNR↑ | Waymo SSIM↑ | Waymo LPIPS↓ | Waymo NVS PSNR↑ | KITTI PSNR↑ | KITTI NVS PSNR↑ | FPS |
|--------|-------------|-------------|--------------|-----------------|-------------|-----------------|-----|
| 3DGS | 27.99 | 0.866 | 0.293 | 25.08 | 21.02 | 19.54 | 125 |
| EmerNeRF | 28.11 | 0.786 | 0.373 | 25.92 | 26.95 | 25.24 | 0.28 |
| SUDS | 28.83 | 0.805 | 0.317 | 25.36 | 28.83 | 26.07 | 0.29 |
| PVG | 32.46 | 0.910 | 0.229 | 28.11 | 32.83 | 27.43 | 50 |
| **DeSiRe-GS** | **33.61** | **0.919** | **0.204** | **29.75** | **33.94** | **28.87** | 36 |

观察：
1. 比 PVG 提升约 **1.1 PSNR**（这是 significant 的，因为 PVG 已经很强）
2. FPS 36（vs PVG 50），是因为渲染了额外 attribute（normal, depth, velocity）
3. NeRF-based 方法（EmerNeRF, SUDS）FPS < 1，远不及 GS 方法

### 3.2 与 bbox-dependent 方法比较 (Table 2)

| Method | Box required? | PSNR (reconst)↑ | PSNR (nvs)↑ |
|--------|---------------|-----------------|-------------|
| EmerNeRF | ✗ | 31.93 | 29.67 |
| PVG | ✗ | 32.37 | 30.19 |
| HUGS | ✓ | 28.26 | 27.65 |
| StreetGS | ✓ | 29.08 | 28.54 |
| OmniRe | ✓ | 34.25 | 32.57 |
| **DeSiRe-GS** | ✗ | **33.82** | **31.49** |

关键 takeaway：**DeSiRe-GS 不需要 bbox，却超过了需要 bbox 的 HUGS 和 StreetGS**，仅次于 OmniRe（但 OmniRe 用了 SMPL 等额外建模，且 bbox 标注完整）。

### 3.3 Ablation Studies (Table 3)

| Setting | PSNR↑ | DPSNR↑ | Depth L1↓ |
|---------|-------|--------|-----------|
| (a) w/o Stage I motion mask | 34.71 | 34.72 | 0.102 |
| (b) w/o FiT3D (use DINOv2) | 34.96 | 34.97 | 0.098 |
| (c) w/o GT normal supervision | 35.45 | 35.49 | 0.091 |
| (d) w/o GT normal (use depth-grad) | 35.24 | 35.53 | 0.085 |
| (e) w/o min scale reg | 35.29 | 35.33 | 0.094 |
| (f) w/o max scale reg | 35.69 | 35.73 | 0.080 |
| (g) w/o multi-view consistency | 35.33 | 35.37 | **0.115** |
| **Full model** | **35.76** | **35.78** | **0.071** |

几个关键观察：
1. **Cross-view consistency 对 depth L1 提升最大**：从 0.115 → 0.071（降低 38%）。这是几何质量的提升，PSNR 提升不显著因为图像质量本身已经很好。
2. **Motion mask 对整体提升显著**，证明 Stage I → Stage II 的蒸馏是必要的。
3. **Scale regularization 对 PSNR 影响小但对几何质量重要**（论文说"虽然 PSNR 变化不大，但 Gaussian 的 3D 结构明显改善"）。
4. **FiT3D 优于 DINOv2**：因为 FiT3D 是 GS-aware 的，在 driving scene 上更准确。

---

## 4. 与相关工作的关系

### 4.1 PVG → DeSiRe-GS

PVG 已经提出了 4D Gaussian 表示（$\tau, \beta, \mathbf{v}$ 三个新参数），但 PVG 本身有缺陷：
- Velocity map noisy（静态区域也有 velocity）
- 没有几何正则化，depth 质量差
- Overfit 到 training views

DeSiRe-GS 是在 PVG 表示基础上加：
1. **Self-supervised mask extractor**（PVG 完全没有）
2. **Velocity regularization**（用 mask 监督 velocity）
3. **Geometry regularization**（flattening, normal derivation, giant Gaussian）
4. **Cross-view consistency**

### 4.2 S3Gaussian [Huang et al., 2024]

S3Gaussian 用 HexPlane encoder + 多头 decoder 做 canonical-to-observation deformation。问题：
- HexPlane 在 object-level 好，但在 unbounded driving scene 上 struggle（论文 intro 明确指出）
- 计算量大（dense deformation field）
- 分解不准确

DeSiRe-GS 用 PVG 的 simple formulation 替代 HexPlane，更高效。

### 4.3 EmerNeRF [Yang et al., 2023]

EmerNeRF 是 NeRF-based 的 self-supervised 方法，思路类似：
- Static field + Dynamic field 分解
- Scene flow estimation

但 NeRF 表示本身慢（FPS < 1），渲染质量也低于 GS 方法。

---

## 5. 我对这篇 paper 的评价与 intuition

### 5.1 优点

1. **Insight 漂亮**：利用 3DGS "本身不能拟合动态" 这一特性做 self-supervised detection，很 elegant。
2. **Pipeline 设计合理**：Stage I (mask) → Stage II (4D modeling) 是经典的 coarse-to-fine。
3. **几何正则化齐全**：flattening + normal derivation + giant Gaussian + cross-view，覆盖了 sparse view driving 的几个核心问题。
4. **FiT3D vs DINOv2 的 ablation** 表明在 GS pipeline 中，"GS-aware feature" 优于 generic feature，这是个有用的经验。

### 5.2 局限 / 可能的问题

1. **PVG 的 sine 振动假设太强**：现实中车辆运动不一定是周期振动。一辆车从远处驶来到近处离开，它的实际轨迹更接近 linear motion 或 constant velocity。用 sine 拟合可能在长序列上有 artifact。论文里 sequence 只有 50 帧，可能没暴露这个问题。

2. **Cross-view consistency 假设完全静态**：公式 (15)-(17) 只对 static pixel 应用。但实际中如何决定哪些 pixel 是 static？论文用 Stage I 的 mask。但如果 mask 有错（漏检动态物体），会把动态 pixel 也当成 static，cross-view loss 会错误地把动态物体"钉"在某处。

3. **两阶段训练复杂**：30K + 50K = 80K iterations，且需要切换 loss 函数和模块。工程上比较 fragile。

4. **依赖 LiDAR 初始化**：$6 \times 10^5$ 个点来自 LiDAR，$4 \times 10^5$ 随机采样。在纯 vision-only 设定下不适用。

5. **FiT3D 依赖**：用了 FiT3D 预训练模型，这是个额外的 dependency。FiT3D 本身是用 GS 训练的，所以这里有点"循环依赖"的意味。

### 5.3 直觉总结

如果让我用一句话概括 DeSiRe-GS 的 essence：

> **"3DGS 自己就是最好的 motion detector —— 把它拟合不好的地方挑出来，再用 4D Gaussian 把这些地方重新建模。"**

这是 emergent behavior 的好例子：3DGS 的 failure mode（不能拟合动态）反而成了 self-supervised 分解的信号源。

---

## 6. Web References

- **DeSiRe-GS GitHub**: https://github.com/chengweialan/DeSiRe-GS
- **3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **2D Gaussian Splatting (Huang et al., SIGGRAPH 2024)**: https://surh.github.io/2d-gaussian-splatting/
- **PVG (Periodic Vibration Gaussian, Chen et al., 2023)**: https://arxiv.org/abs/2311.18561
- **S3Gaussian (Huang et al., 2024)**: https://arxiv.org/abs/2405.20396
- **EmerNeRF (Yang et al., 2023)**: https://arxiv.org/abs/2311.02077
- **StreetGaussians (Yan et al., 2024)**: https://arxiv.org/abs/2401.01339
- **OmniRe (Chen et al., 2024)**: https://arxiv.org/abs/2408.16760
- **HUGS (Zhou et al., CVPR 2024)**: https://arxiv.org/abs/2403.12737
- **FiT3D (Yue et al., ECCV 2024)**: https://arxiv.org/abs/2407.20293
- **DINOv2 (Oquab et al., 2023)**: https://arxiv.org/abs/2304.07193
- **OmniData (Eftekhar et al., ICCV 2021)**: https://omnidata.vision/
- **Waymo Open Dataset**: https://waymo.com/open/
- **KITTI Dataset**: http://www.cvlibs.net/datasets/kitti/

---

## 7. 额外联想与开放问题

这篇工作让我想到几个有趣的延伸方向：

1. **Emergent supervision 类比**：这个"用 model failure 做 supervision"的思路，和 RobustNeRF 用 photometric uncertainty 检测 dynamic 很像，和 EmerNeRF 用 scene flow 也是同类思路。但 DeSiRe-GS 的"3DGS failure"信号更直接。

2. **与 4DGS [Wu et al., 2024] 的关系**：4DGS 用 HexPlane，论文里说 HexPlane 在 unbounded scene 不好，所以选 PVG。但 4DGS 在 D-NeRF 等小场景上很强。能否做一种 hybrid：HexPlane for object, PVG for background?

3. **Normal derivation 的 generalization**：把 normal 从 scale 推导出来这个 idea，其实可以推广到任何 explicit representation。比如 Neural SDF 也可以从 gradient 推 normal。这里用 Gaussian 的 scale argmin 推 normal 是个很简洁的实现。

4. **关于 self-driving scene 的特殊性**：driving scene 有几个独特性被这篇 paper 利用：
   - Camera forward-facing，几乎无背面观测
   - Static background 占大部分像素
   - Dynamic objects 是 rigid（车）或 articulated（人）
   - LiDAR 提供 sparse 但准确的 depth supervision
   这些特性使得 PVG + mask + cross-view 这套组合恰好 work。换到 indoor scene 或 object-centric 4D 重建，可能就不一定 work。

5. **Scale regularization 在 geometry vs appearance 上的不同效果**：Ablation 表明 scale reg 对 PSNR 影响小，但对几何质量影响大。这呼应了一个 generally known fact：Gaussian splatting 的 PSNR 优化很容易，但几何质量优化难，因为 photometric loss 对几何不敏感。

6. **关于 future work**：如果不用 PVG 的 sine 假设，而是直接学一个 deformation field（类似 Deformable GS），但用 mask 监督，可能可以处理更复杂的运动。代价是计算量增加。

希望这个讲解能 build 你的 intuition。核心 take-away 是：**self-supervised decomposition 的关键是找到 model failure mode 和 supervision signal 之间的对应关系**，这篇 paper 用 3DGS 的 ghost artifact 对应 dynamic object 是个很漂亮的实例。
