---
source_pdf: EDGS Eliminating Densification for Efficient Convergence of 3DGS.pdf
paper_sha256: 594456c7924cf6a95caf427f8ca06dbd9680335be846c38f012fe93c03a12478
processed_at: '2026-08-04T01:35:14-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直观的人话来讲，这篇 paper 戳中了 3DGS 的一个“历史包袱”。

### 1. 一句话总结
传统的 3DGS 很像是在玩“扫雷”：一开始只用 SfM 算出几百个稀疏的 3D 点，然后依靠 loss gradient 慢慢去猜“哪里细节不够，需要加新的 splat”。EDGS 的做法极其粗暴且有效：直接用 pretrained 的 dense matcher（比如 RoMa）把所有图片的 pixel 对齐，算出几百万个准确的 3D 点，一开局就把整个 scene 铺满。因为开局就铺满了，后面就不需要再玩“扫雷”（densification）了，只需微调参数即可。

参考链接：
- EDGS 项目主页: https://compvis.github.io/EDGS/
- 3DGS 原始论文: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

### 2. 3DGS 的“痛点”：为什么要 Densification？

在讲 EDGS 之前，需要先建立 3DGS 的 intuition。

3DGS 把 3D 场景表示为很多个 3D 的 Gaussian splat。每个 splat 有 position（位置）、covariance（形状大小）、color（颜色）和 opacity（透明度）。渲染的时候，按照 front-to-back 的顺序把 splats 投影到 pixel 上叠加。看 paper 里的渲染公式（Eq. 1）：

$$
C(\mathbf{p}) = \sum_{i=1}^{N} \mathbf{g}_i^c \, \sigma_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - \mathbf{g}_j^\alpha)
$$

变量和上下标解析：
- $C(\mathbf{p})$：像素 $\mathbf{p}$ 最终的 RGB 颜色。
- $N$：覆盖当前 pixel 的 splats 总数。
- $\mathbf{g}_i^c$：第 $i$ 个 splat 的 RGB 颜色。
- $\sigma_i(\mathbf{p})$：第 $i$ 个 splat 对当前 pixel $\mathbf{p}$ 的贡献权重（受距离和 opacity 影响）。
- $\mathbf{g}_j^\alpha$：第 $j$ 个 splat 的 opacity。
- $\prod_{j=1}^{i-1}(1-\mathbf{g}_j^\alpha)$：这是 transmittance 项，表示前面的 splat 挡住了多少光线。如果前面的 splat 不透明度很高，后面的 splat 就看不见了。

**痛点在哪里？**
传统 3DGS 的初始化用的是 COLMAP (SfM) 算出来的 sparse point cloud。对于一个复杂 scene，SfM 可能只给你几万个点。这就导致一开始渲染出来的图全是黑洞，loss 极大。3DGS 的做法是：在 optimization 过程中，监控每个 splat 周围的 photometric loss gradient。如果某个区域的 gradient 一直很大（说明现有的 splat 拟合不好），它就会把那个地方的 splat 切开或者复制，变成两个 splat（这就是 densification）。

这就像是在一片荒地上盖房子。你先随机扔几块砖，发现哪里漏风就在哪里补砖。这种方式很慢，且容易在高频细节（比如草地、树叶）处失败，因为 gradient 太杂乱，系统不知道该在哪里精确地“补砖”。

---

### 3. EDGS 的核心思路：提前做好功课

EDGS 认为，既然 densification 是为了补几何的漏洞，为什么我们不在 optimization 开始前，直接用现成的 CV 工具把几何算准呢？

**它的 pipeline 极其符合直觉：**
1. 选一张 reference image $I^i$。
2. 找它附近的 2 张 neighbor images $I^j$。
3. 用一个强大的 pretrained 网络 $\mathcal{M}$（默认用 RoMa），把 $I^i$ 和 $I^j$ 做 dense matching，得到每个 pixel 的匹配关系和 confidence。
4. 拿着这些匹配好的 pixel pairs，用经典的三角测量算出 3D 坐标。
5. 把这些 3D 坐标作为 splat 的初始 position，把 reference image 上的 pixel 颜色直接赋给 splat 的 color。
6. 关掉 3DGS 的 densification 开关，直接跑 optimization。

这就好比盖房子前，先用无人机扫一遍地形，直接把所有的砖头精确地放到承重位置。后面只需要稍微抹点水泥（微调参数）就行了。

参考链接：
- RoMa (Robust Dense Feature Matching): https://github.com/Parskatt/RoMa

---

### 4. 技术细节与直觉建立

虽然思路简单，但工程实现上有很多巧思。这里讲讲几个关键的技术点。

#### 4.1 Splats 三角测量

假设我们在 image $I^i$ 的 pixel $(u_k^i, v_k^i)$ 和 image $I^j$ 的 pixel $(u_k^j, v_k^j)$ 找到了一个 match。我们要算出它在 3D 空间里的位置 $\mathbf{g}_k^x = (x_k, y_k, z_k)$。

Paper 里的公式（Eq. 6 和 7）：

$$
\mathbf{g}_k^x := \arg\min_{\mathbf{x}} \|A\mathbf{x} + b\|^2
$$

这里的 intuition 非常优美：在理想情况下，从 camera $i$ 发出的射线和从 camera $j$ 发出的射线会在 3D 空间中精确相交。但现实是，dense matcher 算出来的 pixel 坐标有误差，这两条射线在 3D 空间里通常是“异面直线”，不会相交。
变量解释：
- $A$：这是一个矩阵，由两个 camera 的 projection matrices（$P^i, P^j$）和 matched pixel 坐标 $(u, v)$ 构成。
- $\mathbf{x}$：我们要求解的 3D 坐标。
- $b$：常数向量。

这个公式其实就是在求：哪个 3D 点离这两条射线的距离都最近？也就是经典的 DLT (Direct Linear Transform) 算法。通过 least squares 找一个最接近交点的位置。

#### 4.2 采样策略：如何避免一堆点挤在一起？

Dense matcher 会给出成百上千万个匹配点，全部拿来当 splat 会让显存爆炸。所以需要 sampling distribution $\mathbf{p}^i$ 来挑选。

Paper 提出了两个维度的过滤：
1. **Confidence-based ($\mathbf{p}_{corr}$)**：matcher 给出的 confidence score 要大于阈值 $\tau_{corr}$。
2. **Geometry-based ($\mathbf{p}_{proj}$)**：reprojection error 要小于阈值 $\tau_{proj}$。

Reprojection error（Eq. 8）：
$$
\varepsilon_k^i = \|\pi(P^i, \mathbf{g}_k^x) - (u_k^i, v_k^i)\|_2
$$
变量解释：
- $\pi(P^i, \cdot)$：把 3D 点 $\mathbf{g}_k^x$ 用 camera $P^i$ 重新投影回 2D image 的函数。
- $(u_k^i, v_k^i)$：原本 matcher 算出来的 2D 坐标。
- $\varepsilon_k^i$：两者之间的距离。

Intuition：你算出来的 3D 点，重新投影回图片里，如果和原来 matcher 指认的 pixel 差了十万八千里，说明这个匹配是错的，或者三角测量算错了，直接扔掉。

最后把这两者结合起来（Eq. 11）：
$$
\mathbf{p}^i(k) \propto \max_{j \in \mathbb{I}_i}\big(\mathbf{p}_{corr}^{ij}(k) \cdot \mathbf{p}_{proj}^{ij}(k)\big)
$$
这保证选出来的点既在语义上可靠，又在几何上自洽。关键一点，paper 强调要在 thresholded set 上做 **uniform** sampling。如果只按 confidence 采样，点全聚集在纹理强烈的边缘，平坦区域就没点了，这就背离了 dense initialization 的初衷。

#### 4.3 Spherical Harmonics (SH) 初始化

3DGS 用 Spherical Harmonics 来建模 view-dependent color（比如从不同角度看玻璃反光不同）。SH 有 16 个系数。

EDGS 在初始化时，每个 splat 通常只能看到 2 个 view（reference view 和 neighbor view）。2 个方程解 16 个未知数，这是欠定的。

Paper 的做法是解一个 least squares 问题（Eq. 12）：
$$
\hat{\mathbf{H}}_k = \arg\min_{\mathbf{H} \in \mathbb{R}^{16 \times 3}} \|\mathbf{Y}_k \mathbf{H} - \mathbf{O}_k\|_F^2
$$
变量解释：
- $\mathbf{H} \in \mathbb{R}^{16 \times 3}$：16 个 SH 系数，对应 3 个 RGB 通道。
- $\mathbf{Y}_k \in \mathbb{R}^{n \times 16}$：SH basis 在 $n$ 个 view directions 上的取值。
- $\mathbf{O}_k \in \mathbb{R}^{n \times 3}$：从 $n$ 个 view 观测到的 RGB 颜色。

因为方程数不够，这里用 Moore-Penrose pseudoinverse（伪逆）求 minimum-norm 解：
$$
\hat{\mathbf{H}}_k = \mathbf{Y}_k^+ \mathbf{O}_k
$$
直觉：给高阶 SH 系数赋一个尽量小、尽量平滑的初始值，剩下的让 3DGS 后续优化去慢慢学。这个操作对 indoor scenes（复杂反光）提升极大，paper 里的 ablation study (Table 6) 显示加了 SH init，LPIPS 从 0.175 降到了 0.141。

参考链接：
- Moore-Penrose pseudoinverse: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse

---

### 5. 实验数据的直观解读

看实验数据，最能体现 EDGS 威力的是两处。

#### 5.1 Gaussian Motion 分析

Paper 4.5 节算了一笔账：从 optimization 开始到结束，每个 Gaussian 到底跑了多远？

定义两个量（Eq. 14 和 15）：
1. **Displacement**（起点到终点的直线距离）：
$$ \|\mathbf{g}_i^x(0) - \mathbf{g}_i^x(T)\|_2 $$
2. **Trajectory length**（整个优化路径的累计长度）：
$$ \sum_{t=0}^{T} \|\mathbf{g}_i^x(t) - \mathbf{g}_i^x(t+1)\|_2 $$

结果极其震撼：
- EDGS 的 coordinate displacement 比 3DGS 减少了 **50 倍**。
- EDGS 的 trajectory length 比 3DGS 减少了 **30 倍**。

这就是 EDGS 的 "smoking gun"（铁证）。传统 3DGS 之所以慢，是因为它的 splats 一直在 3D 空间里疯狂“漂流”，寻找归宿。而 EDGS 的 splats 一开局就已经站在了正确的几何表面上，optimization 只是在原地微调。这直接解释了为什么 EDGS 训练 5000 步就能超过别的模型训练 30000 步。

#### 5.2 核心实验对比

在 Mip-NeRF360 dataset 上：
- **原版 3DGS**：PSNR 27.49，LPIPS 0.215，2.8M 个 Gaussians，耗时 26 分钟。
- **3DGS-MCMC**（SOTA quality 方法）：PSNR 28.15，LPIPS 0.176，3.2M 个 Gaussians，耗时 20 分钟。
- **EDGS + 3DGS**：PSNR 28.02，LPIPS **0.141**，**1.9M** 个 Gaussians，耗时 27 分钟。

重点看 LPIPS 和 #Gaussians。LPIPS 衡量感知质量，0.141 是极其优秀的成绩。而且 EDGS 只用了不到 200 万个 splats，比别的方法少了一大截。更少的 splats 意味着渲染速度更快，对下游应用（VR、游戏）极度友好。

另外，Table 4 做了一个极端测试：把 3DGS 的 densification 关掉。原版 3DGS 的 PSNR 直接从 27.49 暴跌到 25.60。而 EDGS 关掉 densification，PSNR 依然维持在 28.02。这彻底证明了 densification 对于 EDGS 来说是完全多余的。

---

### 6. 更深层的直觉与联想

EDGS 这篇工作，在哲学层面给人很多启发。

**1. Explicit Geometry Prior 的回归**
近几年的趋势是 end-to-end learning，啥都丢给网络去学，什么 SfM、multi-view geometry 都显得“老掉牙”了。3DGS 本身也是从 sparse SfM 起步，让网络在 optimization 里去“发现”几何。EDGS 证明了，如果把传统的 dense matching、triangulation 这些 explicit geometry 工具拿出来，作为强大的先验注入到 representation 里，效果反而比让网络自己苦哈哈地去学要好。这是一种“老派 CV 与现代 Differentiable Rendering 结合”的胜利。

**2. Optimization Landscape 的重塑**
从 optimization 的角度看，3DGS 的 loss landscape 像是一片布满坑洼的荒地。Sparse initialization 导致很多区域没有 gradient signal，densification 就是在这片荒地上盲目探索。EDGS 直接把起点放到了 global minimum 附近，把非凸优化问题降维成了一个 local convex refinement 问题。这启示我们：很多 deep learning 里的 optimization trick（比如复杂的 scheduler、regularization），可能本质上都是在弥补 initialization 拉胯的问题。

**3. 对其他领域的启发**
这个思路完全可以迁移到其他任务。比如 NeRF 的加速（虽然 NeRF 已经快被 3DGS 淘汰了），或者 4D Gaussian Splatting for dynamic scenes。在 dynamic 场景里，时间维度的 correspondence 更难找，如果引入时序 dense matcher（类似 video optical flow），提前做好 4D triangulation，是不是也能省掉 dynamic 3DGS 里的 deformation field 学习成本？甚至在未来 text-to-3D 的生成里，如果 generative model 能直接生成 dense correspondence 而不仅仅是图片，3D 资产的构建会瞬间提速。

参考链接：
- 4D Gaussian Splatting: https://arxiv.org/abs/2402.03307

**4. 局限性**
当然，EDGS 也有它的 soft spot。它极其依赖 dense matcher 的质量。如果遇到 textureless region（比如白墙），或者极端光照变化，RoMa 给出的 match 就会崩掉，triangulation 就会算出飞在天上的“噪点 splat”。虽然 paper 里展示说 optimization 能把这些噪点 suppress 掉，但如果 init 阶段噪点过多，显存开销和优化时间依然是个问题。另外，初始化需要 120 秒，对于要求极低延迟的 SLAM 场景还是偏慢。

总而言之，EDGS 是一篇极其 elegant 的 paper。它没有发明复杂的数学公式，也没有搞庞大的网络架构，仅仅是换了一个视角：“与其让 optimization 苦苦探索，不如提前把功课做好”。这种靠直觉和工程美学直击痛点的工作，非常 Karpathy 风格。

---

# EDGS: Eliminating Densification for Efficient Convergence of 3DGS — 深度解析

## 1. Paper 的核心 thesis

EDGS 提出一个非常 elegant 的观点：3DGS [Kerbl et al. 2023] 里那套 incremental densification 机制（split / clone / prune）本质上是一种 **lazy 的 geometry discovery**。它通过 photometric gradient 反复探测哪里需要更多 splats，然后才行动。这个过程：

- 在 high-frequency regions（grass、stones、textures）容易失败，因为 gradient norm metric 与人眼感知对齐差
- optimization path 长：每个 Gaussian 要经历多次 refinement 才到 final state
- 整个 scene 收敛被 densification 的 "等待 → split → 再优化" 循环拖慢

EDGS 的 solution：用 dense 2D correspondences（来自 RoMa [Edstedt et al. 2023]）做 triangulation，**一步到位**地把 splats 铺满 scene 的几何，让 optimization 一开始就 supervised by rich per-pixel signal。

Paper 的关键论点：*densification 是被默认的 sparse SfM initialization 逼出来的；如果一开始就 dense + geometrically accurate，densification 就是 redundant 的。*

参考链接：
- EDGS 项目页: https://compvis.github.io/EDGS/
- 3DGS 官方: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- RoMa: https://github.com/Parskatt/RoMa

---

## 2. 3DGS 背景 — 为什么 densification 必要又昂贵

### 2.1 渲染公式（Eq. 1 解析）

3DGS 把 scene 表示为 Gaussians 集合 $\mathbb{G} = \bigcup_{i=1}^{N} \mathbf{g}_i$，每个 Gaussian 编码：

$$
\mathbf{g}_i = \{\underbrace{\mathbf{g}_i^x}_{\text{center } \in \mathbb{R}^3},\ \underbrace{\Sigma_i}_{\text{covariance } \in \mathbb{R}^7},\ \underbrace{\mathbf{g}_i^c}_{\text{RGB } \in \mathbb{R}^3},\ \underbrace{g_i^\alpha}_{\text{opacity } \in \mathbb{R}}\}
$$

每个下标的含义：
- $i \in \{1, \ldots, N\}$ — 第 $i$ 个 Gaussian
- $x$ — spatial position
- $c$ — color
- $\alpha$ — opacity (alpha)

像素 $p$ 的渲染颜色：

$$
C(\mathbf{p}) = \sum_{i=1}^{N} \mathbf{g}_i^c \, \sigma_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - \mathbf{g}_j^\alpha)
$$

变量解释：
- $C(\mathbf{p})$ — 像素 $\mathbf{p}$ 的最终渲染 RGB
- $\sigma_i(\mathbf{p}) = g_i^\alpha \exp(-\frac{1}{2}(\mathbf{p}' - \mathbf{g}_i^x)^T \Sigma_i^{-1}(\mathbf{p}' - \mathbf{g}_i^x))$ — Gaussian $i$ 在像素 $\mathbf{p}$ 上的 contribution
- $\mathbf{p}' - \mathbf{g}_i^x$ — pixel projection line 与 Gaussian center 的最短距离
- $\prod_{j=1}^{i-1}(1-\mathbf{g}_j^\alpha)$ — front-to-back alpha compositing 的 transmittance，表示前 $i-1$ 个 Gaussian "挡掉" 的比例

Covariance reparameterization（Eq. 2）保证 positive semi-definiteness：

$$
\Sigma_i = R_i S_i S_i^T R_i^T
$$

其中 $R_i$ 是 rotation matrix，$S_i$ 是 scaling matrix。这种分解让 optimization 可以直接做 SGD 而不会破坏 PSD 性质。

### 2.2 Densification 的流程与问题

3DGS 的 densification loop：
1. 渲染 → 算 photometric loss → backward
2. 累积每个 Gaussian 位置空间的 gradient norm
3. 若 gradient norm 超阈值（典型 $\tau_{pos} = 0.0002$），split 或 clone
4. 若 opacity 过低或 scale 过大，prune

问题：gradient norm 是个 **lagging indicator**。要等到 loss 在某区域"卡住"很久，累积出足够大的 gradient，densification 才会被触发。这种 lag 导致：
- 高频纹理区域 gradient 噪声大，threshold 难调
- 收敛 trajectory 长，Gaussians 反复"漂流"后才稳定

---

## 3. EDGS 方法详解

### 3.1 Pipeline overview

整个 EDGS 流程：

```
Input images + camera poses
        ↓
For each reference view I^i:
    ├── Pick 2 nearest neighbors I^j
    ├── Run dense matcher M (RoMa) → warp field W^{ij}, confidence c^{ij}
    ├── Triangulate matched pixels → 3D points g_k^x
    ├── Filter via reprojection error + confidence
    └── Sample 20K splats from sampling distribution p^i
        ↓
Initialize SH coefficients from multi-view colors
        ↓
Standard 3DGS optimization (densification disabled)
```

### 3.2 Dense correspondence extraction (Sec 3.2, Eq. 3)

定义 matcher $\mathcal{M}$：

$$
\mathcal{M}(I^i, I^j) \mapsto (\mathcal{W}^{ij}, \mathbf{c}^{ij})
$$

变量：
- $I^i, I^j$ — reference 和 neighbor images
- $\mathcal{W}^{ij} \in \mathbb{R}^{2 \times H \times W}$ — dense forward warp field，把 $I^i$ 中每个 pixel map 到 $I^j$ 中对应位置
- $\mathbf{c}^{ij} \in \mathbb{R}^{H \times W}$ — 每个 correspondence 的 confidence（match quality score）
- $H, W$ — image 高度和宽度

paper 默认用 RoMa，但 ablation (Tab. 5) 也试了 LoFTR、DKM、RAFT。RAFT 效果最差，因为它是为 optical flow 设计的（小 baseline），不适合大 viewpoint 变化。

**Intuition**：与其让 photometric loss 在 optimization 过程中"发现"哪些位置需要更多 splats，不如先用一个强大的 pretrained matcher 直接告诉我们哪些像素在多视图中是 consistent 的——这些位置必然对应真实几何表面。

### 3.3 Triangulation (Sec 3.3, Eqs. 4-7)

给定两个 camera 的 projection matrices $P^i, P^j \in \mathbb{R}^{4 \times 3}$（这里 notation 略 unusual，常见是 $3 \times 4$），以及 matched pixel pair $(u_k^i, v_k^i)$ 和 $(u_k^j, v_k^j)$，求 Gaussian center $\mathbf{g}_k^x = (x_k, y_k, z_k)$。

Projection 方程：

$$
\begin{cases}
[\mathbf{g}_k^x \ \ 1] P^i = w_k^i [u_k^i \ v_k^i \ 1] \\
[\mathbf{g}_k^x \ \ 1] P^j = w_k^j [u_k^j \ v_k^j \ 1]
\end{cases}
$$

变量：
- $[\mathbf{g}_k^x \ \ 1]$ — homogeneous 3D coordinates of Gaussian $k$
- $P^i, P^j$ — camera $i$ 和 $j$ 的 projection matrices
- $w_k^i, w_k^j$ — homogeneous normalization scalars（project 到 image plane 时的 depth scale）

把 $w$ 消掉后得到 4 个 linear equations（Eq. 5），重排成 $A\mathbf{g}_k^x = -b$ 的形式（Eq. 6），其中 $A \in \mathbb{R}^{4 \times 4}$，$b = \mathbf{0}$（齐次形式）。

求 least squares 解：

$$
\mathbf{g}_k^x := \arg\min_{\mathbf{x}} \|A\mathbf{x} + b\|^2
$$

这是经典的 DLT (Direct Linear Transform) triangulation。**Intuition**：两条 viewing rays 在 3D 空间中可能不严格相交（因为 matcher 有误差），最小化两条 rays 的"距离平方"得到最佳近似交点。

### 3.4 Sampling distribution (Sec 3.4, Eqs. 8-11)

直接用所有 triangulated points 不可行（数量太大）。所以定义一个 sampling distribution $\mathbf{p}^i$ 来挑选"高质量且空间均匀"的子集。

**Reprojection error**（Eq. 8）：

$$
\varepsilon_k^i = \|\pi(P^i, \mathbf{g}_k^x) - (u_k^i, v_k^i)\|_2
$$

变量：
- $\pi(P, \cdot)$ — 用 camera matrix $P$ 做 projection
- $\varepsilon_k^i$ — triangulated 3D point $\mathbf{g}_k^x$ 重新 project 回 image $I^i$ 时与原 matched pixel 的距离
- $\varepsilon_k^{ij} := \max(\varepsilon_k^i, \varepsilon_k^j)$ — 取两个 view 的 worst case

**两个 sampling distributions**：

$$
\mathbf{p}_{corr}^{ij} \sim \mathcal{U}\big(\{k \mid \mathbf{c}^{ij}(u_k^i, v_k^i) > \tau_{corr}\}\big)
$$

$$
\mathbf{p}_{proj}^{ij} \sim \mathcal{U}\Big(\{k \mid \varepsilon_k^{ij} < \tau_{proj}\}\Big)
$$

变量：
- $\tau_{corr} = 0.05$ — confidence threshold
- $\tau_{proj} = 0.01$ — reprojection error threshold（NDC units）
- $\mathcal{U}$ — uniform distribution over a set

**Combined sampling distribution**（Eq. 11）：

$$
\mathbf{p}^i(k) \propto \max_{j \in \mathbb{I}_i}\big(\mathbf{p}_{corr}^{ij}(k) \cdot \mathbf{p}_{proj}^{ij}(k)\big)
$$

变量：
- $\mathbb{I}_i$ — reference image $I^i$ 的 neighbor 集合
- $\max_{j}$ — 对所有 neighbor 取最大（保证一个 correspondence 只要在某一个 neighbor pair 中高质量就能被采样）

**Intuition**：confidence 反映 matcher 的 semantic reliability（这个 match 看起来对不对），reprojection error 反映 geometric consistency（triangulation 出来的点重新 project 回去是否还对得上）。两者乘积 = 既要 match 看起来对，又要几何自洽。Uniform sampling 在 thresholded set 上避免偏向 high-confidence 的 spatial cluster（如 edges）。

**Global aggregation**：

$$
\mathbf{p}(k) \propto \bigcup_i \mathbf{p}^i_k
$$

跨所有 reference views aggregate，保证 scene 全覆盖。

### 3.5 Spherical harmonics initialization (Sec 3.5, Eqs. 12-13)

每个 splat 采样后，从多视图收集 $n$ 个 RGB observations $\mathbf{O}_k \in \mathbb{R}^{n \times 3}$（$n$ 行，每行一个 view 的 RGB），同时记录 view directions $\mathbf{v}_1, \ldots, \mathbf{v}_n \in \mathbb{R}^3$。

构建 SH basis matrix $\mathbf{Y}_k \in \mathbb{R}^{n \times 16}$：每行是 16 个 real SH basis functions（degree $l=3$）在某个 view direction $\mathbf{v}_i$ 处的取值。SH 系数：

$$
\hat{\mathbf{H}}_k = \arg\min_{\mathbf{H} \in \mathbb{R}^{16 \times 3}} \|\mathbf{Y}_k \mathbf{H} - \mathbf{O}_k\|_F^2
$$

变量：
- $\mathbf{H} \in \mathbb{R}^{16 \times 3}$ — 16 个 SH coefficients × 3 个 RGB channel
- $\|\cdot\|_F$ — Frobenius norm
- $\hat{\mathbf{H}}_k$ — 学到的 SH coefficients

通常 $n < 16$（每个 splat 只有 2 个 observation），系统欠定，所以用 Moore-Penrose pseudoinverse：

$$
\hat{\mathbf{H}}_k = \mathbf{Y}_k^+ \mathbf{O}_k
$$

其中 $\mathbf{Y}_k^+$ 是 pseudoinverse，给出 minimum-norm 解。

**Implementation 细节**：第 0 个 SH coefficient（即 DC 项，平均颜色）直接从 reference view 的 pixel color 初始化，保证 appearance 一致；其余 15 个 coefficients 用 pseudoinverse 估计。Optimization 时每 1000 步逐步 unfreeze 更高阶的 coefficients，类似 3DGS 原版的 progressive unfreezing 策略。

**Intuition**：SH 初始化让 splat 一开始就有合理的 view-dependent appearance，特别对 indoor scenes（complex lighting、reflections）帮助大。Tab. 6 显示加 SH init 把 LPIPS 从 0.175 降到 0.141（对比 w/o SH init），降幅明显。

---

## 4. 实验数据深度解读

### 4.1 主结果（Tab. 1, Mip-NeRF360）

| Method | Densif.-free | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Train time | #G (M) |
|---|---|---|---|---|---|---|
| Mip-NeRF360 [Barron et al. 2022] | ✓ | 0.792 | 27.69 | 0.237 | 48h | – |
| 3DGS* (retrained) | ✗ | 0.816 | 27.49 | 0.215 | 26m | 2.8 |
| AbsGS-0004 [Ye et al. 2024] | ✗ | 0.818 | 27.41 | 0.198 | 20m | 3.1 |
| Mip-Splatting [Yu et al. 2024] | ✗ | 0.838 | 27.97 | 0.179 | 26m | 4.0 |
| 3DGS-MCMC [Kheradmand et al. 2024] | ✗ | 0.842 | 28.15 | 0.176 | 20m | 3.2 |
| ScaffoldGS [Lu et al. 2024] | ✗ | 0.812 | 27.60 | 0.222 | 22m | 6.0 |
| **EDGS + 3DGS** | **✓** | **0.839** | **28.02** | **0.141** | 27m | 1.9 |

**关键观察**：
1. **LPIPS 0.141** 是所有方法中最低的，比 3DGS-MCMC (0.176) 低 20%，比原 3DGS (0.215) 低 34%。LPIPS 反映 perceptual quality，说明 EDGS 在 high-frequency details 上优势最大。
2. **#Gaussians 1.9M** 也是最低的之一，比 3DGS (2.8M) 少 32%。Dense initialization 让每个 splat 都"有用"，不需要冗余 splats 来填补 gap。
3. **Train time 27m** 包含 initialization（约 2m），与 3DGS 的 26m 相当。注意 EDGS 是 densification-free 的，关闭了 gradient 累积这一步。
4. **Densif.-free 一栏**：所有 baseline 都依赖 densification，只有 EDGS 不依赖。Tab. 4 显示关掉 densification 后 3DGS PSNR 从 27.49 掉到 25.60（跌 1.89 dB），而 EDGS 几乎不变（28.02 vs 28.08）。

### 4.2 早期停止对比（Tab. 2, 效率导向）

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Time |
|---|---|---|---|---|
| gsplat | 0.818 | 27.51 | 0.215 | 18m |
| Taming 3DGS | 0.820 | 27.71 | 0.207 | 14m |
| MiniSplatting | 0.820 | 27.25 | 0.217 | 12m |
| EAGLES | 0.809 | 27.20 | 0.232 | 16m |
| **EDGS + 3DGS 10K** | **0.834** | 27.54 | **0.154** | 12m |
| **EDGS + 3DGS 5K** | 0.825 | 26.88 | 0.166 | 8m |

**关键观察**：
- EDGS 5K 在 8 分钟内达到 LPIPS 0.166，比 Taming 3DGS (14m, 0.207) 还低
- EDGS 10K 在 12 分钟内 SSIM 0.834，比所有 efficiency baseline 都高
- **加速的核心来源**：dense initialization 让 optimization 从一个有几何先验的点开始，而非从 sparse SfM 点云缓慢生长

### 4.3 与 ADC 方法结合（Tab. 3, plug-and-play）

| ADC Method | EDGS Init | SSIM | PSNR | LPIPS | Time | #G |
|---|---|---|---|---|---|---|
| AbsGS-0004 | ✗ | 0.818 | 27.41 | 0.198 | 20m | 3.1 |
| AbsGS-0004 | ✓ | 0.822 | 27.53 | 0.187 | 19m | 3.0 |
| 3DGS-MCMC | ✗ | 0.842 | 28.15 | 0.176 | 20m | 3.2 |
| 3DGS-MCMC | ✓ | **0.847** | **28.29** | **0.159** | 20m | 3.2 |
| Taming 3DGS | ✗ | 0.820 | 27.71 | 0.207 | 14m | 3.2 |
| Taming 3DGS | ✓ | 0.842 | 28.07 | 0.179 | 11m | 3.2 |

**关键观察**：EDGS 作为 plug-in initialization 给所有 ADC 方法都带来提升，且不增加 Gaussian count 或 training time。最有意思的是 Taming 3DGS + EDGS：从 14m 缩到 11m，SSIM 还涨了 0.022。这说明 EDGS 的 dense init 让 ADC 的 densification 更高效——它不需要从头长出 splats，而是在已经合理的初始集合上做 refinement。

### 4.4 Gaussian motion 分析（Sec 4.5, Fig. 4, Eqs. 14-15）

这是 paper 中最 insightful 的 ablation。定义两个量：

**Displacement**（从 init 到 final 的位移）：

$$
\|\mathbf{g}_i^x(0) - \mathbf{g}_i^x(T)\|_2 \in \mathbb{R}^2
$$

**Full trajectory length**（整个 optimization 路径长度）：

$$
\sum_{t=0}^{T} \|\mathbf{g}_i^x(t) - \mathbf{g}_i^x(t+1)\|_2 \in \mathbb{R}^2
$$

变量：
- $T$ — 总 optimization steps
- $\mathbf{g}_i^x(t)$ — Gaussian $i$ 在 step $t$ 的 position
- $\mathbf{g}_i^c(t)$ — Gaussian $i$ 在 step $t$ 的 color

**实验结果**：
- EDGS 把 coordinate displacement 减少 **50 倍**
- EDGS 把 coordinate trajectory length 减少 **30 倍**
- Color trajectory length 减少约 **2 倍**

**Intuition**：这是 paper 的"smoking gun"。它直接证明了 EDGS 的核心 thesis——optimization 难是因为起点差。给一个好的几何先验，每个 Gaussian 不需要"漂流"到对的位置，它一开始就在那儿附近。Color 减少没那么 dramatic 是因为 view-dependent effects 的微调仍然需要 optimization。

### 4.5 Robustness to noise (Fig. 7)

给 initialized Gaussian 参数加 Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma)$，分别注入到 spatial coordinates 或 color：

**关键观察**：
- 对 color noise 极 robust（即使 $\sigma = 0.1$ 也几乎不退化）
- 对 coordinate noise 较敏感
- 但 EDGS 本身的 init 已经"自然 noisy"（来自 triangulation 误差），所以对 moderate noise 不敏感

**Intuition**：这印证了 3.4 节的设计——sampling distribution 通过 reprojection filtering 保证了 geometric consistency，但 color 信号是从单 view pixel 直接读取的，相对容易通过 optimization 修正。这也解释了为什么 SH initialization 对 indoor scenes 帮助大（多 view 观测让 color 解析度提升）。

### 4.6 不同初始化策略对比（Tab. 7）

| Init type | Densif.-free | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Random | ✗ | 22.19 | 0.704 | 0.313 |
| COLMAP | ✗ | 27.49 | 0.816 | 0.215 |
| DepthFM | ✓ | 26.99 | 0.810 | 0.202 |
| DepthFM + densif | ✗ | 27.18 | 0.819 | 0.197 |
| VGGT | ✗ | 26.40 | 0.782 | 0.177 |
| VGGT-X | ✗ | – | – | – |
| **EDGS** | **✓** | **28.02** | **0.839** | **0.141** |

**关键观察**：
- Random init 失败：证明 init quality 极重要
- DepthFM (monocular depth) 不如 EDGS：monocular depth 有 scale ambiguity across views
- VGGT (feed-forward 3D prediction) 也不如 EDGS：可能是 VGGT 的预测在 fine details 上不够 sharp
- EDGS 是唯一同时达到 densif-free 和 SOTA quality 的方法

### 4.7 Sparse-view 实验（Tab. A1）

| Method | 12-view SSIM | 24-view SSIM |
|---|---|---|
| 3DGS | 0.499 | 0.588 |
| SparseGS | 0.577 | 0.713 |
| **EDGS + 3DGS** | **0.594** | 0.699 |

**Intuition**：EDGS 不是为 sparse-view 设计的，但在 12-view 时 SSIM 比 SparseGS 还高。这暗示一个有趣的方向——dense correspondence 本身就是一种"几何 prior"，某种程度上替代了 diffusion-based score distillation 的作用。

---

## 5. 为什么 EDGS work — 几个 deep intuition

### 5.1 Optimization landscape 的"地形图"视角

3DGS 的 loss landscape 在 sparse SfM init 下是个 badlands——大量 local minima，每个对应"用错的 splats 拟合某个区域"。Densification 的 split/clone 相当于在 landscape 上"加新 explorer"，但每个新 explorer 都从原点附近开始，需要时间 drift 到对的位置。

EDGS 的 dense init 把所有 explorer 直接放到接近 global minimum 的位置。Optimization 变成"local refinement"，几乎不需要 landscape navigation。这就是为什么 Fig. 4 的 trajectory length 减少 30 倍。

### 5.2 Dense supervision 信号

3DGS 在 init 后早期，每个 pixel 只被少数 splats 覆盖（因为 SfM 点稀疏）。Photometric gradient 流到这些少数 splats 上，但 neighboring 区域没有 splats 接收信号——所以需要 densification 来"扩展"覆盖。

EDGS 一开始就让每个 pixel 都有对应的 splat，photometric loss 直接 inform 所有 spatial regions，gradient signal 几乎处处 non-trivial。这相当于把"哪里需要 splats"的决策从 optimization feedback loop 抽出来，用 geometric matcher offline 解决。

### 5.3 为什么 densification 在 high-frequency 区域失败

3DGS 用 position-space gradient norm 触发 densification。但 high-frequency regions（grass、stones）的 photometric error 来自 fine spatial variations，gradient norm 可能均匀地大或均匀地小，导致 threshold 难以选择。

EDGS 用 dense correspondence 解决：matcher 直接在每个 pixel 位置给出一个 match（如果有），triangulation 出一个 3D point。不需要 gradient 来"发现"这些区域——它们 by construction 都被覆盖了。

### 5.4 与 NeRF 的呼应

这有点像 NeRF → Instant-NGP → 3DGS 的演化：从 coarse grid → multi-resolution hash → explicit primitives。EDGS 的 contribution 是在"explicit primitives + 好的 spatial prior"之间找到了新的 sweet spot。RadSplat [Niemeyer et al. 2024] 也探索类似 idea，但用 pretrained NeRF 做 init，需要 9 小时——比 EDGS 的 120s initialization 慢 270 倍。

---

## 6. Limitations 和潜在问题

1. **Dense matcher 的 inductive bias**：RoMa 在哪些场景可能 fail？Textureless regions、repetitive patterns、剧烈 viewpoint 变化。Paper 没有详细讨论 matcher failure mode 下的表现。

2. **Memory overhead during init**：peak GPU memory ~15GB（matcher forward + intermediate correspondences）。对 embedded / mobile 场景不友好。

3. **Initialization 时间约 2 分钟**：对于需要快速 preview 的应用仍偏慢。可以探索 lightweight matcher（如/lighter versions of RoMa）。

4. **Indoor scenes 中 SH init 的局限**：当 splat 只有 2 个 view observation 时，SH 的 view-dependent modeling 很 underdetermined。pseudoinverse 给出 minimum-norm 解，但物理上可能不正确。可以引入 lighting prior 或 learned SH prior。

5. **与 Mip-Splatting 的 anti-aliasing 关系**：EDGS 没有显式处理 aliasing。Tab. 1 显示 Mip-Splatting SSIM (0.838) 与 EDGS (0.839) 接近——可能 EDGS 的 dense init 隐式缓解了 aliasing（更多 splats → 每个 splat 更小 → less aliasing），但这没在 paper 中分析。

6. **Dynamic scenes 推广**：paper 只做 static scenes。Dynamic 场景下 correspondence 更难（需要 disentangle motion），dense matcher 可能给出错误 match。

---

## 7. 与相关工作的联系

- **3DGS-MCMC [Kheradmand et al. 2024]** (https://arxiv.org/abs/2406.04251): reformulates densification as MCMC sampling. EDGS + MCMC (Tab. 3) 比单独 MCMC 还好——MCMC 的"采样"过程在 EDGS 的好 init 上更高效。

- **ScaffoldGS [Lu et al. 2024]** (https://arxiv.org/abs/2311.14091): anchor points + neural Gaussian generation. EDGS 的 dense init 与 anchor-based 结构正交，可以组合。

- **Taming 3DGS [Mallick et al. 2024]** (https://arxiv.org/abs/2406.15643): pixel-error-driven densification, 资源受限场景。Tab. 3 显示 EDGS init 让 Taming 3DGS 训练时间从 14m 降到 11m。

- **AbsGS [Ye et al. 2024]** (https://arxiv.org/abs/2404.02390): 解决 gradient collision 问题。EDGS 把 gradient collision 问题从根本上缓解——dense init 让 gradient 分布更均匀。

- **Mip-Splatting [Yu et al. 2024]** (https://arxiv.org/abs/2311.16493): anti-aliasing via multi-scale。EDGS SSIM (0.839) 略超 Mip-Splatting (0.838)，可能两者结合会更好。

- **DepthFM [Gui et al. 2024]** (https://arxiv.org/abs/2403.13788): monocular depth 估计。Tab. 7 显示 depth-based init 不如 correspondence-based，因为 monocular depth 的 scale 不一致。

- **RoMa [Edstedt et al. 2023]** (https://arxiv.org/abs/2305.15404): EDGS 用的 dense matcher。RoMa 是 detector-free matcher，可以给出 dense correspondences 而非 sparse keypoints。

- **Mip-NeRF360 dataset [Barron et al. 2022]** (https://jonbarron.info/mipnerf360/): 主要 benchmark。

- **Tanks & Temples [Knapitsch et al. 2017]** (https://www.tanksandtemples.org/): 大场景 benchmark。

- **Deep Blending [Hedman et al. 2018]** (https://github.com/Phog/DeepBlending): 室内场景 benchmark。

---

## 8. 个人 take-away 与未来方向

### 8.1 方法学 insight

EDGS 的核心贡献是把 3DGS 的 densification 重新 frame 为"initialization 不够好的 symptom"。这类似于 deep learning 里 batch norm 的故事——很多 optimization trick 其实是在 compensate initialization / data distribution 的问题。

更广义的 lesson：**explicit geometry prior 仍然有价值**。在 end-to-end learning 时代，用 SfM、COLMAP、dense matcher 这些"old school" CV 工具提取几何信号，再 feed 给 modern learned representation，往往是 win-win。

### 8.2 未来方向联想

1. **Learned dense matcher 的 specialization**：训练一个专门为 3DGS init 设计的 matcher，可能在 fine details 上比 RoMa 更好。比如加入 multi-view consistency loss 训练。

2. **Streaming / online EDGS**：当前 init 是 batch 的（一次性处理所有 views）。可以探索 incremental 版本——新 view 加入时只 update 局部 splats。

3. **Combination with neural fields priors**：把 EDGS 与 triplane representations 或 multi-resolution hash 结合，可能进一步压缩 Gaussian count。

4. **Dynamic scene extension**：4D 场景下，dense correspondence 变成 dense optical flow + scene flow。可以把 EDGS 推广到 4D Gaussian splatting [Wu et al. 2024]。

5. **Differentiable triangulation**：当前 triangulation 是 one-shot 的，不可微。如果 end-to-end 训练 matcher + 3DGS，可能 jointly optimize geometric prior 和 photometric loss。

6. **Generative 3D**：text-to-3D 任务里，EDGS 可以作为更好的 init point 替代 random point clouds。DreamGaussian [Tang et al. 2023] 之类的方法可能受益。

7. **SLAM application**：GS-SLAM [Yan et al. 2024] 类工作里，dense init 可以让 mapping module 更高效。

### 8.3 一些可能被忽视的细节

- Paper Sec. A 提到 outdoor scenes 4× downsample，indoor 2× downsample。这个 preprocessing choice 对应 3DGS 的标准 protocol，但 EDGS 的 dense init 可能对 resolution 更敏感——高 resolution 下 dense matcher 的输出更 fine-grained，可能进一步提升 quality。

- Fig. A1 的 nearest neighbor aggregation 显示，单个 neighbor 只覆盖 reference image 的一部分。这暗示 EDGS 的 scene coverage 严重依赖 neighbor selection 策略。Future work 可以探索更智能的 view selection（如基于 coverage 的 greedy selection）。

- Tab. A2 显示单纯增加 COLMAP points 10× 或 50× 都没用（甚至略降），证明 EDGS 的 gains 来自 geometric accuracy 而非 point count。这是个 very clean 的 ablation，强烈支持 paper 的核心 thesis。

- EDGS 的 #Gaussians 是 1.9M（Mip-NeRF360），低于大多数 baseline。这对 rendering 性能有直接好处——更少的 splats 意味着更快的 rasterization。

---

## 9. 总结

EDGS 是一个 elegant 的工作：用 dense correspondence triangulation 一步到位地初始化 3DGS，bypass densification。它的 strength 在于：
- 简单：不修改 optimization algorithm，只改 init
- 有效：LPIPS 降 34%，#G 降 32%，train time 相当
- 兼容：plug-and-play with ADC methods
- Principled：reprojection + confidence 双重 filter 保证 init quality

Methodologically，它代表了一种 "use strong geometric priors to shortcut expensive optimization" 的思路，可能在 more general learning 场景里有 inspiration。Limitation 主要是依赖 matcher quality 和 init time overhead，但这些都是可以工程上优化的。

对 3DGS community 而言，EDGS 提供了一个新的 baseline——以后任何 densification-based 方法都应该和 EDGS 比较，因为 EDGS 证明了 "densification is not strictly necessary"。

---

## References (with web links)

- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- RoMa: https://github.com/Parskatt/RoMa
- EDGS project: https://compvis.github.io/EDGS/
- Mip-NeRF360 dataset: https://jonbarron.info/mipnerf360/
- Tanks & Temples: https://www.tanksandtemples.org/
- Deep Blending: https://deepblending.github.io/
- 3DGS-MCMC: https://arxiv.org/abs/2406.04251
- ScaffoldGS: https://city-super.github.io/scaffold-gs/
- Mip-Splatting: https://neuralradiancefields.io/mip-splatting/
- AbsGS: https://arxiv.org/abs/2404.02390
- Taming 3DGS: https://arxiv.org/abs/2406.15643
- DepthFM: https://arxiv.org/abs/2403.13788
- LoFTR: https://zju3dv.github.io/loftr/
- DKM: https://arxiv.org/abs/2202.00667
- RAFT: https://arxiv.org/abs/2003.12039
- COLMAP (SfM): https://colmap.github.io/
- Moore-Penrose pseudoinverse: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
- DreamGaussian: https://dreamgaussian.github.io/
- GS-SLAM: https://arxiv.org/abs/2311.14100
- RadSplat: https://arxiv.org/abs/2403.06505
