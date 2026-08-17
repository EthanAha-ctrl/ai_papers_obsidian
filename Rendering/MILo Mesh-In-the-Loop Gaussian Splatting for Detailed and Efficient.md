---
source_pdf: MILo Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient.pdf
paper_sha256: e22db09e608af23d288503f65875431799e79e27c26f37da1c33ab09bb3875bf
processed_at: '2026-08-05T18:11:34-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MILo

Andrej, 你想听人话我就用大白话讲，但公式和直觉还是会留着，因为这些恰恰是 build intuition 的关键。

---

## 一句话讲清楚这篇 paper 在干啥

之前所有从 Gaussian Splatting 提 mesh 的方法都是**先训完 Gaussians，再提 mesh**。这就好比你让一个小孩随便画一幅画，画完之后你再拿橡皮去擦多余的线条，擦完发现关键的细节也被擦没了，或者有些地方画得根本不对，但小孩已经走了，你只能干看着。

MILo 干的事是：**在训练的每个 iteration 里，都从 Gaussians 里提一个 mesh 出来，然后让 image loss 同时作用在 Gaussians 和 mesh 上**。这样 Gaussians 在训练过程中始终知道"我最后要变成 mesh"，它就没动机去作弊了。最后出来的 mesh 又干净、又轻、又准。

paper 项目页：<https://anttwo.github.io/milo/>

---

## 为什么"先训再提"这个套路根本就是 broken 的

3D Gaussian Splatting 这个 representation 是为渲染服务的，不是为提几何服务的。你给它一组 photometric loss，它会想尽一切办法把图 fit 出来。问题在于，fit 图的方式有很多种：

- 老老实实把 Gaussian 摆在物体表面 — 几何对，渲染也对
- 弄一个高 opacity 的 floater 漂在镜头前面，颜色调对 — 几何错，渲染也能过
- 在 thin structure（栅栏、车辐条）两侧各摆一个 Gaussian，让它们的颜色互相 compensate — 几何有点错，但渲染凑合

Gaussian Splatting 不会主动选第一种，它会随便选哪个能降 loss 用哪个。这就是 paper 里说的 "cheating"。

问题来了：等训练完，你要提 mesh 了。Floater 在 isosurface 之外，提不出来；thin structure 两侧的 Gaussian 因为 opacity 被 photometric loss 拉成各管各的，isosurface 一过直接把两侧合并，几何就糊了；cavity 里那些本来不该有的 Gaussian 还赖在那儿，提出来一堆杂碎。

**post-hoc extraction 一次性把一个已经"被惯坏"的 volumetric 表示拍扁成 surface，丢什么细节、出什么 artifact 全靠运气。** 而且 Gaussian 是连续椭球，naive isosurface 在 thin 部分会 over-inflate 或 erode — 这是数学性质决定的，不是 bug。

MILo 的 insight 就是：**与其等 Gaussians 作弊完再想办法补救，不如训练时就让它没法作弊**。怎么没法作弊？每个 iteration 都从 Gaussians 提一个 mesh 出来，渲染这个 mesh 跟 GT 比，gradient 流回 Gaussians。Gaussian 想摆到 floater 位置？mesh 一提出来发现这地方深度不对，loss 立刻惩罚它，它就老实了。

这个 idea 在 NeRF 时代根本做不了，因为 NeRF 的 geometry 全锁在 MLP 里，mesh vertex 没法 chain rule 回去。GS 的 $\mu, R, s$ 都是 explicit 参数，mesh vertex 位置可以直接 chain 回到 Gaussian 的中心、朝向、尺度上。这是 representation 选择决定 algorithm possibility 的典型例子。

---

## 整个 pipeline 用大白话走一遍

每个 training iteration 干 5 件事：

1. **挑一批 "Gaussian pivot"** — 从几 million 个 Gaussians 里按重要性采样出 0.1–0.5M 个，这些会当 Delaunay 顶点
2. **算 Delaunay 四面体化** — 把这些 pivot 点连成四面体网络
3. **拿 SDF 值** — 每个 Gaussian pivot 自带 9 个 learnable SDF 值，对应它生成的 9 个 Delaunay 顶点
4. **跑 Marching Tetrahedra** — 在四面体上找 SDF 符号变化的边，插值出 mesh vertex，连成 triangle
5. **同时渲染 Gaussians 和 mesh**，两个跟 GT 比，loss 流回 Gaussian 参数

关键：Delaunay 算法本身不需要 differentiable。每 500 个 iteration 才重新算一次 connectivity，gradient 不走 connectivity 这条路，走的是 vertex 坐标 → Gaussian $\mu, R, s$ 这条连续通道。Delaunay 只是个 discrete 框架，套在上面，几何在里面微调。

---

## 几个关键技术 piece 的人话解释

### (1) 为什么每个 Gaussian 要生成 9 个 Delaunay 顶点

直接用 Gaussian 中心当 Delaunay 顶点行不行？不行，两个原因。

**第一个原因**：Marching Tetrahedra 需要四面体的顶点**跨越** surface — 一部分顶点在 surface 外（SDF 正），一部分在 surface 内（SDF 负），才能在边上插值出 mesh 顶点。如果所有顶点都落在 surface 上（Gaussian 中心正好在 surface 上），算法直接退化，提不出 mesh。

**第二个原因**：要适应 Gaussian 的 anisotropic shape。一个 thin Gaussian（沿某方向 scale 很小）应该对应一个 thin tetrahedra 结构，flat Gaussian 对应 flat tetrahedra。

MILo 的解决方案是：每个 Gaussian $\mathcal{G}_k$ 生成 9 个点，公式是：

$$p_{k,i} = \mu_k + R_k \times (s_k \odot b_i), \quad i = 0 \dots 8$$

变量解释：
- $\mu_k \in \mathbb{R}^3$：第 $k$ 个 Gaussian 的中心位置
- $R_k \in \mathbb{R}^{3\times 3}$：Gaussian 的旋转矩阵（从 quaternion $q_k$ 算出来）
- $s_k \in \mathbb{R}^3$：Gaussian 沿 3 个主轴的 scale
- $b_i \in \mathbb{R}^3$：9 个标准点，即 $\{(0,0,0), (\pm 1, \pm 1, \pm 1)\}$，也就是中心 + 8 个 unit cube 的角
- $\odot$：Hadamard 积，逐元素相乘

**人话**：把 Gaussian 想象成一个 oriented bounding box，box 的中心和 8 个角点就是这 9 个 Delaunay 顶点。因为 box 是用 Gaussian 的旋转和尺度变换出来的，所以它完美跟随 Gaussian 的形状 — thin Gaussian 就给 thin box，flat Gaussian 就给 flat box。box 的角在 Gaussian "外面"，自然就跨越了 surface。

这 9 个点的位置会随 Gaussian $\mu, R, s$ 变化而变化。**Gradient 从 mesh vertex 流回这 9 个点，再 chain rule 回 $\mu, R, s$**。这是 mesh 反过来 sculpt Gaussian 的物理通道。

参考 GOF 原文：<https://arxiv.org/abs/2406.01467>

---

### (2) 为什么要 importance sampling 选一部分 Gaussian 当 pivot

用全部 Gaussians 当 Delaunay 顶点的话，几 million 个点跑 Delaunay，CGAL 库会慢到爆炸。而且大场景里很多 Gaussian 是 background filler，对几何贡献不大。

借鉴 Mini-Splatting2，给每个 Gaussian 算一个 importance score = 它在所有 training view 渲染时 blending coefficient 的平均幅度。这个 score 高的 Gaussian 就是对图像真正有贡献的，用它们当 pivot。

采样出 0.1–0.5M 个当 pivot。其他 Gaussian 不直接被 mesh loss 约束，但通过 **volume-to-surface consistency loss**（渲染 Gaussians 和 mesh 比 depth/normal），它们也被间接约束 — Gaussian 渲出来的 depth 要跟 mesh 对齐，mesh 是从 pivot 提的，所以 non-pivot Gaussian 也得跟着 pivot 走。

Mini-Splatting2: <https://arxiv.org/abs/2411.12788>

---

### (3) SDF 值跟 Gaussian 其他参数解耦 — 这是个很关键的设计

每个 Gaussian $\mathcal{G}_k$ 关联一个可训练向量 $f_k \in \mathbb{R}^9$，9 个 SDF 值对应 §(1) 里的 9 个 Delaunay 顶点。这个 $f_k$ 跟 Gaussian 的 opacity、scale、rotation、color **完全独立**，单独被 mesh loss 优化。

**为什么不直接用 Gaussian 的 opacity 当 SDF？** SuGaR 早期就这么干过。问题是 opacity 在训练时被 photometric loss 疯狂推，它 fit 图的目标和 fit 几何的目标**互相打架**。opacity 为了渲染可能被推到很低（让某个 Gaussian 不影响图像），但这并不代表这个位置几何上就不存在。

解耦之后，两套 gradient 各管各的：
- Gaussian 的 $\mu, R, s, \alpha, c$ 被 photometric loss 优化 — 负责把图渲染对
- $f_k$ 被 mesh supervision 优化 — 负责把几何摆对

它们通过 consistency loss 互相 *soft 通信*，但不会互相 *hard 干扰*。这个 dual-representation co-optimization 设计很干净。

实操上优化的是 tanh-normalized truncated SDF 在 $[-1, 1]$ 之间，远离 surface 时值饱和，避免梯度爆炸。

---

### (4) Marching Tetrahedra 公式拆解

给定 Delaunay 四面体化，每个 tetrahedron 有 4 个顶点，每个顶点带 SDF 值。算法遍历所有 tetrahedron，找 SDF 符号改变的边，在边上插值出 mesh vertex。

考虑一条边，两端是 Delaunay 顶点 $\mathcal{P}_{k,i}$ 和 $\mathcal{P}_{k',j}$，SDF 值 $f_{k,i}$ 和 $f_{k',j}$ 符号相反。Mesh vertex $v_n$ 落在这条边上：

$$v_n = \frac{f_{k,i} \, p_{k',j} - f_{k',j} \, p_{k,i}}{f_{k,i} - f_{k',j}}$$

变量：
- $f_{k,i} \in \mathbb{R}$：Gaussian $k$ 的第 $i$ 个 SDF 值
- $p_{k,i} \in \mathbb{R}^3$：对应的 Delaunay 顶点坐标（公式 1）
- $v_n \in \mathbb{R}^3$：提取出的 mesh triangle 顶点坐标

**人话**：就是 1D 线性插值。SDF 是符号距离函数，0 等值面就是 surface。两个端点一正一负，0 等值面肯定在中间某处。$v_n = p_{k,i} + t \cdot (p_{k',j} - p_{k,i})$，其中 $t = f_{k,i} / (f_{k,i} - f_{k',j})$。

**Gradient 通道**：
- $\partial v_n / \partial f_{k,i}$ — mesh loss 可以调 SDF 值
- $\partial v_n / \partial p_{k,i}$ — mesh loss 可以调 Delaunay 顶点位置，进而 chain 回 $\mu_k, R_k, s_k$

第二条通道是关键。Mesh 渲出来的 depth/normal 不对 → mesh vertex 位置需要调 → Delaunay 顶点位置需要调 → Gaussian 的中心、旋转、尺度需要调。**Gaussian 被推向能让 mesh 干净的位置**。这就是 in-loop 的核心机制。

GPU 实现用 PyTorch custom kernel，Delaunay 用 CGAL 每 500 iter 跑一次。

CGAL: <https://www.cgal.org/>

---

## Loss 函数的人话拆解

总 loss：

$$\mathcal{L} = \mathcal{L}_{\text{vol}} + \mathcal{L}_{\text{mesh}} + \mathcal{L}_{\text{reg}}$$

每一项干啥：

### $\mathcal{L}_{\text{vol}}$ — 让 Gaussians 把图渲染对

$$\mathcal{L}_{\text{vol}} = (1 - \lambda_{\text{RGB}}) \mathcal{L}_1 + \lambda_{\text{RGB}} \mathcal{L}_{\text{D-SSIM}} + \lambda_{\text{N}} \mathcal{L}_{\text{N}}$$

- $\mathcal{L}_1$：rendered image vs GT pixel-wise L1
- $\mathcal{L}_{\text{D-SSIM}}$：结构相似度，$\lambda_{\text{RGB}} = 0.2$
- $\mathcal{L}_{\text{N}}$：让相邻 Gaussian 渲的法向一致

$$\mathcal{L}_{\text{N}} = \sum_i \left( 1 - \mathbf{N}(i) \cdot \tilde{\mathbf{N}}(i) \right)$$

- $\mathbf{N}(i)$：pixel $i$ 通过 volume rendering 算出的 expected normal
- $\tilde{\mathbf{N}}(i)$：pixel $i$ 通过 rendered depth map 做 finite difference 算出的法向

**人话**：让 Gaussian 自己声明的法向和它实际深度图体现的法向一致。逼 Gaussians 排成局部平面，别东倒西歪。

---

### $\mathcal{L}_{\text{mesh}}$ — 让 Gaussians 和 mesh 互相贴着

$$\mathcal{L}_{\text{mesh}} = \lambda_{\text{MD}} \mathcal{L}_{\text{MD}} + \lambda_{\text{MN}} \mathcal{L}_{\text{MN}}$$

**Depth consistency**：

$$\mathcal{L}_{\text{MD}} = \sum_i \log\left(1 + |D(i) - D_M(i)|\right)$$

- $D(i)$：pixel $i$ 渲自 Gaussians 的 depth
- $D_M(i)$：pixel $i$ 渲自 mesh 的 depth

**人话**：你 Gaussian 看到的深度和 mesh 看到的深度要一样。Charbonnier penalty (log(1+·)) 比 L1 在小 error 处平滑、比 L2 在大 error 处鲁棒，防 outlier 主导 gradient。

**Normal consistency**：

$$\mathcal{L}_{\text{MN}} = \sum_i \left(1 - \tilde{\mathbf{N}}(i) \cdot N_M(i)\right)$$

- $\tilde{\mathbf{N}}(i)$：Gaussian depth map 算出的法向
- $N_M(i)$：mesh face 的法向

**人话**：Gaussian 体现的法向和 mesh 法向要朝同方向。Depth 防位置偏离，normal 防方向偏离。

Ablation 里有个有意思的现象：只用 depth loss 时 F1=0.46，加上 normal loss 反而 F1=0.44。看似变差，但看 Fig. 7 视觉对比就明白 — normal term 把 mesh 上的 noise 砍掉了，F1 这个 metric 对 noise 不敏感，没反映出这个改善。

---

### $\mathcal{L}_{\text{erosion}}$ — 防止 thin structure 被冲掉

$$\mathcal{L}_{\text{erosion}} = \sum_{g \in G_{\text{Del}}} \max(0, f_{\mu_g})$$

- $G_{\text{Del}}$：被选作 Delaunay pivot 的 Gaussian 集合
- $f_{\mu_g}$：Gaussian $g$ 中心点（即 $b_0 = (0,0,0)$ 那个 pivot）的 SDF 值

**这是 hinge loss**：Gaussian 中心 SDF 是正的（在 surface 外）就惩罚，是负的（在 surface 内）就不惩罚。

**直觉**：什么叫 erosion？想象一个栅栏的 thin bar，两侧各有一个 Gaussian。如果两个 Gaussian 的 9 个 pivot SDF 全变正，整个 tetrahedron 内没有负 SDF 顶点，Marching Tetrahedra 在这个区域直接没有 mesh 输出 — 栅栏消失了。更糟的是，mesh 一消失，渲染 mesh 的 depth/normal loss 在这个区域也没 signal 了，gradient 流不回去，几何就再也恢复不了。**Erosion 是个不可逆的失败模式**。

这个 loss 说：每个被选作 pivot 的 Gaussian，它的中心点的 SDF 必须是负的（在 mesh 内部）。这就保证每个 pivot Gaussian 至少贡献一个负 SDF 顶点给 Delaunay 网络，surface 在这个 Gaussian 周围肯定存在。

**类比**：像沙滩上插旗子。每个 Gaussian 是一面旗，要求旗杆必须插在沙里（SDF 负），不能漂在空中。旗子插住了，海浪（gradient）就冲不掉这块沙。

只对 pivot Gaussian 的中心点应用，不对其他 8 个 corner pivot 应用，避免 mesh 整体 collapse 到 Gaussian 中心上。

---

### $\mathcal{L}_{\text{interior}}$ — 防止 mesh 内部藏杂碎

$$\mathcal{L}_{\text{interior}} = \sum_p H\left(\sigma(-f_p), o_p\right) \cdot o_p$$

- $p$：Delaunay site
- $f_p$：site $p$ 的 SDF 值
- $o_p \in \{0, 1\}$：site $p$ 是否在 mesh 内部
- $\sigma$：sigmoid
- $H$：cross-entropy

展开就是 $H(\sigma(-f_p), 1) = -\log(\sigma(-f_p))$，当 $f_p$ 越大（越在 surface 外）这个值越大。乘 $o_p$ 表示只对内部点应用，逼内部点的 SDF 变负（确实在 surface 内）。

**$o_p$ 怎么算**：
1. 用当前 mesh 渲所有 training view 的 depth map
2. 对每个 Delaunay site $p$，如果它在所有能看见它的 view 里都被 depth map 遮挡（即在 rendered depth 后面），那 $o_p = 1$（在内部）
3. 否则 $o_p = 0$（在外部或 visible）

每 200 iter 才更新一次 $o_p$，因为这步要遍历所有 views × 所有点，慢。但 200 iter 内 mesh 变化不大，lazy update 够用。

**直觉**：Depth/normal loss 只能监督看得见的 surface，看不见的 mesh 内部没监督。内部容易出现 chaotic cavities（Fig. 8a）— 想象一个表面看起来好好的气球，里面塞了一堆杂碎表面。这对下游 physics simulation 是灾难 — 流体在内部会绕着这些杂碎打转，结果完全不对。

这个 loss 把内部点拉成 SDF 负（内部应该是 solid，在 surface 内），让 mesh 内部 watertight 且 empty。这是一个 self-supervised feedback loop — mesh 自己决定哪些点在内部，再用这个信息约束 SDF 生成下一个 mesh，类似 EM 算法，E-step 算 label，M-step 用 label 优化参数。

---

## 训练 schedule 的人话版本

```
Iter 0 - 3000:    光用 photometric loss 训 Gaussians，aggressive densification
                  （Gaussians 数量从初始增长到几 million）

Iter 3000:        加上 normal consistency loss，开始让 Gaussians 排齐

Iter 3000 - 8000: 让 Gaussians refine，稳定下来

Iter 8000:        关键转折点！
                  - 停止 densification 和 pruning
                  - Base model 在这里 prune 到 0.1-0.5M（importance-weighted）
                  - 开始 mesh extraction，full loss 上场

Iter 8000 - 18000: 10k iter 的 mesh-in-the-loop 优化
                   - 每 500 iter 更新 Delaunay connectivity
                   - 每 200 iter 更新 interior occupancy label
                   - 每 1 iter 都提 mesh、渲 mesh、算 loss、backprop
```

总 18k iter，base model 在 DTU 上 25 分钟跑完，T&T 上 40–50 分钟。Dense 上 unbounded scene 最多 2 小时。单 4090。

超参数：$\lambda_{\text{RGB}} = 0.2$，$\lambda_N = \lambda_{MD} = \lambda_{MN} = 0.05$，$\lambda_{\text{erosion}} = \lambda_{\text{interior}} = 0.005$。注意 anti-erosion 和 interior 权重是 0.005，比其他小一个量级 — 因为这两个 loss 是 "护栏"，不是主 driving force，权重太大反而会跟主 loss 打架。

---

## 实验数据的直觉解读

### Resource 对比 (Tab. 1) — 这是 paper 标题 "Detailed and Efficient" 的核心证据

| Method | #Gauss (M) | GPU Mem (GB) | Time | #Verts | #Tris | Size (MB) |
|---|---|---|---|---|---|---|
| 2DGS | 0.98 | 4.7 | 29m | 16.39M | 21.68M | 557 |
| GOF | 1.55 | 10.6 | 93m | 16.49M | 33.17M | 600 |
| RaDe-GS | 1.56 | 12.4 | 42m | 14.75M | 29.59M | 592 |
| **MILo base** | **0.28** | 10.0 | 50m | **4.36M** | **8.97M** | **180** |
| MILo dense | 2.11 | 16.5 | 110m | 6.89M | 13.79M | 276 |

MILo base 用 GOF 1/6 的 Gaussians，1/4 的 vertices，1/3 的存储，F1 还比 GOF 高。

**为什么 vertex 这么少 quality 反而高？** Post-hoc TSDF / Marching Cubes 在 regular grid 上跑，grid resolution 固定，不管 surface 简单还是复杂都均匀 dense。MILo 的 Delaunay 在 Gaussian 密集处 dense（复杂几何）、稀疏处 sparse（flat background），自然 adaptive。再加上 in-loop 优化让 Gaussians 主动 reposition 到真正需要的地方，不浪费 vertex 在 flat 区域。

---

### T&T F1 (Tab. 2) — 在真实复杂场景上 MILo 显著领先

平均 F1：
- 2DGS = 0.30
- GOF = 0.46
- MILo base = **0.47**（已经超过 GOF，即使 vertex 数量只有 1/4）
- MILo dense = **0.49**（explicit methods 里最佳）
- Neuralangelo = 0.50（implicit SOTA，但要 >24h 训练）

---

### DTU Chamfer Distance (Tab. 3) — 在 object-centric 场景上跟 SOTA 持平

DTU 是受控的 object-centric 数据集，post-hoc 本来就 work 得好。MILo base mean CD = 0.68，跟 GOF (0.74)、RaDe-GS (0.68) 同档。

MILo 在 DTU 上没 T&T 上那么突出，因为 DTU 的主要挑战在 fit object 几何细节，post-hoc 在那里已经做得不错。MILo 的优势在 large complex full-scene reconstruction。

---

### Mesh-Based Novel View Synthesis (Tab. 4) — 这是个新提出的 evaluation protocol

**问题**：现有 benchmark 都有局限 — DTU 只 object GT，T&T 只 foreground GT，MipNeRF360 完全没 geometry GT。怎么评 background 几何质量？

**MILo 的方案**：用一个 neural color field 给 mesh 上色，然后渲染 test view 跟 GT 比 PSNR/SSIM/LPIPS。

具体做法：
1. 对每个被评 mesh，训练一个 $F_{\text{color}}: \mathbb{R}^3 \to [0,1]^3$（用 TensoRF backbone，5k iter）
2. 渲 mesh 到 test view，每个 pixel 的 3D 位置 backproject 出来，query color field 得 RGB
3. 跟 GT 比

**关键 design**：用 neural color field 而非 vertex color。Vertex color 会 bias dense mesh — 稀疏 mesh 的 vertex 颜色分辨率低，image 质量必然差，但这不代表几何差。Neural color field decouples color from mesh resolution，纯评几何对图像的 alignment。

**直觉**：如果 mesh geometry 错位、缺失、有 erosion，color field 怎么训都 fit 不好 GT image，PSNR 低。这是 image-space 的 geometry 评估 proxy。

**结果** (MipNeRF360)：
- GOF: PSNR 20.78
- MILo base: PSNR **24.09**（最佳），SSIM 0.6885（最佳），LPIPS 0.3235（最佳）

MILo base 在 MipNeRF360 上甚至比 GOF 好 3 分多，即使 vertex 数量是 GOF 的 1/5 (6.73M vs 32.80M)。强烈支持 in-loop 优化带来的几何质量提升。

TensoRF: <https://apchenstu.github.io/TensoRF/>

---

### Ablation (Tab. 5) — 各 loss 贡献

| Method | F1 |
|---|---|
| Baseline | 0.41 |
| + $\mathcal{L}_{\text{MD}}$ | 0.46 |
| + $\mathcal{L}_{\text{mesh}}$ (depth + normal) | 0.44 |
| + $\mathcal{L}_{\text{erosion}}$ | 0.44 |
| Full | 0.47 |

Normal 和 erosion loss 在 F1 上几乎不涨甚至略降，但视觉上 (Fig. 5, Fig. 7) 改善巨大。这暴露了 F1 score 对 mesh noise 和 thin structure 缺失不敏感的弱点。Paper 提的 Mesh-Based NVS metric 就是为了补这个弱点。

**Take-away**：单一 metric 评 mesh 质量是不够的，要多个角度交叉验证。

---

### 跟 TSDF fusion 对比 (Fig. 9)

TSDF 在 fixed 3D grid 上融合 depth map，内存随 resolution 立方增长，large scene 直接 OOM（Courthouse 跑不了）。MILo 的 learnable SDF 在 Delaunay sites 上，数量随 Gaussian 数量线性，scalable。

而且 TSDF fusion 要遍历所有 training view 才能提一次 mesh，计算重，没法塞进 training loop。MILo 的 SDF 是 learnable 参数，跟 Gaussians 一起优化，每次 iteration 提 mesh 几乎免费（GPU Marching Tetrahedra 是 ms 级）。

---

## 这 paper 的核心 conceptual contribution（我的看法）

不是 mesh 质量提升（虽然显著），是 **把 surface extraction 从 "conversion task" 变成 "regularization task"**。

Post-hoc mesh extraction 是一次性 conversion，错了就错了。In-loop mesh extraction 是持续的 geometric prior，它把 Gaussians 拉向一个 "surface-friendly" 的 manifold，同时让 mesh 也能跟着 Gaussians 走。**两个 representation 互相 sculpt**，最终结果比任何一个单独优化都好。

这种 bidirectional coupling 的思路在 graphics 里见过（mesh + image joint optimization、differentiable physics + neural net），但在 GS surface reconstruction 这个具体场景里被组合得很干净。Loss 设计上（erosion 防 thin structure 消失、interior 防 cavities）处理了几个非常 specific 的 failure mode，工程上很扎实。

参考一些相关工作：
- SuGaR (先训再提，有 refine stage 但 topology fixed): <https://github.com/Anttwo/SuGaR>
- GOF (post-hoc 但用 Delaunay): <https://github.com/autonomousvision/gaussian-opacity-fields>
- Radiant Foam (Delaunay in-loop，只做 NVS 不做 surface): <https://arxiv.org/abs/2502.01157>
- FlexiCubes (类似的 differentiable extraction 思路): <https://research.nvidia.com/labs/toronto-ai/flexicubes/>

---

## 可以接着挖的方向（如果你想 build on top）

1. **Dynamic scene**: 当前 framework static。Dynamic Gaussian (Deformable 3DGS) 要 in-loop mesh 需要时间一致的 Delaunay，每帧重算太慢。潜在方向：时间一致的 tetrahedral structure，让它能 warp 而不重算。

2. **Differentiable rendering on mesh**: nvdifrast 是 rasterization-based，处理半透明（植被、玻璃）弱。Mitsuba 之类的 path tracing differentiable renderer 在 mesh 上会更准但慢 10-100x。

3. **Eikonal regularization**: SDF 是 Gaussian-wise 独立学的，缺全局 SDF 一致性约束。理论上 SDF 应满足 $\|\nabla f\| = 1$，在 tetrahedra 上算 spatial gradient 工程不 trivial，但加上可能让 SDF 学得更稳。

4. **Adaptive pivot sampling**: 现在 importance 是渲染贡献。更聪明的策略是 "几何复杂度" — 在 high-curvature 区域多采样 pivot，flat 区域少采样。可以用 rendered normal 的方差当 curvature proxy。

5. **Open surface**: 当前假设 watertight closed surface。 vegetation、cloth、open shell 几何需要 unsigned distance field 或 manifold mesh representation。这套 framework 改一改应该能做 open surface。

6. **MILo for 4D**: 把 mesh-in-loop 思路用到 dynamic scene，让每帧的 mesh 之间有一致 topology 和 correspondence，对 animation 和 simulation 都是宝。

---

## 我的整体 take

MILo 这 paper 我觉得是 GS surface reconstruction 这条线上的一个**架构转变**，而不是 incremental improvement。它把"训完再提"这个 community 默认的 paradigm 推翻了，用 in-loop bidirectional coupling 替代。这个转变让 vertex 数量降一个量级、内部变 watertight、thin structure 保住，这些 quantitative 改善都是 paradigm 转变的 *必然副产品* — 当你让 Gaussians 始终为 mesh 服务，它就不会形成 floater、cavity 这些 artifact，mesh 自然就干净了。

而且这个 idea 实际上挺通用的，可以 plug 到任何能渲染 depth/normal 的 GS 方法里（paper 里就 demo 了 plug 到 GOF 和 RaDe-GS）。这种 plug-and-play 特性让它容易被 community 接受。

Erosion loss 和 interior loss 这两个看似 minor 的工程细节，其实是 paper 的精髓 — 它们处理的是 in-loop 优化特有的 failure mode（一旦 erosion 发生就不可逆、内部无监督产生杂碎）。这种对 failure mode 的精准识别和针对性 design，是好研究的标志。

希望这版"人话"版本帮到你 build intuition。如果你想深挖某个具体点，比如 Marching Tetrahedra 在 GPU 上怎么并行、SDF 学习的 dynamics、occupancy label 计算的并行算法，或者跟 FlexiCubes / DMTet 那套 differentiable extraction 的对比，我都可以继续展开。

---

# MILo: Mesh-In-the-Loop Gaussian Splatting 深度技术拆解

Andrej, 这篇 paper 我看完之后的直觉是: 它把 3DGS 的 surface reconstruction 这个老问题做了一个非常关键的架构转变 — 把 mesh extraction 从 *post-hoc 的单向操作* 改成 *training loop 内的双向 differentiable 操作*。这个转变带来一系列 downstream 好处, 包括 vertex 数量降低一个量级, 内部 cavities 消除, thin structure 不被 erosion 掉。下面我尽量把每个技术 piece 的数学和 intuition 都挖透。

paper 项目页: <https://anttwo.github.io/milo/>

---

## 1. 问题本质: 为什么 post-hoc mesh extraction 是 fundamentally broken

之前所有 GS-based surface reconstruction (SuGaR, 2DGS, GOF, RaDe-GS, NeuSG...) 的 pipeline 都是:

1. Optimize Gaussians with photometric loss (+ some regularizer)
2. 用某种 isosurface / TSDF / Poisson 把 volume 转成 mesh

这里有个 deep issue: Gaussian 在 step 1 优化时, 它**根本不知道**后面要被转成 mesh。它的 opacity、scale、view-dependent color 这些参数被 photometric loss 推着走, 完全可以 "cheating" — 用一个高 opacity 的 floater 配上正确颜色去 fit 一个 pixel, 而不需要把 geometry 摆在 surface 上。一旦这种 inconsistency 被 baked into representation, post-hoc 的 isosurface extraction 会:
- 在 thin structure 上 over-inflate 或 erode (Gaussian 是连续椭球, naive isosurfacing 在 thin 部分会把两侧 surface 合并)
- 生成 cavities 和 floaters
- 丢失 background 几何 (因为 background 上的 Gaussian 通常 opacity 不高, isosurface threshold 过它们时直接消失)

MILo 的核心 insight 是: **在每一个 training iteration 都 differentiably extract 一个 mesh 出来, 让 image-based loss 直接作用在 mesh 上, gradient 再流回 Gaussians**。这样 Gaussians 在整个 training 过程中, 始终被一个 explicit surface representation "约束" 着, 它没有动机去形成 floater、cavity 这类 artifact, 因为这些 artifact 在 mesh 上立刻会被惩罚。

这个 idea 在 NeRF 时代几乎不可能 (因为 NeRF 的 MLP + volume rendering 对 surface 没有显式参数化), 但 GS 的显式 Gaussian 参数恰好可以让 mesh extraction 的不同iable链路建立起来。

参考:
- 3DGS 原文: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- SuGaR: <https://arxiv.org/abs/2406.01467>
- 2DGS: <https://arxiv.org/abs/2403.17888>
- GOF: <https://arxiv.org/abs/2406.01467>

---

## 2. 整体 Pipeline: 5 步循环

每个 training iteration:

1. **Fetch Delaunay vertices** — 从 importance-sampled Gaussian pivots 派生
2. **Update Delaunay triangulation** — 每 500 iter 更新一次 (因为 Delaunay 本身 non-differentiable, 但 gradient 不需要经过它)
3. **Fetch SDF values** — 每个 Gaussian pivot 对应 9 个 learnable SDF values
4. **GPU Marching Tetrahedra** — 提取 triangle mesh
5. **Render + backprop** — Gaussians 和 mesh 都被渲染, image loss + consistency loss 双向 backprop 到 Gaussians

关键 design choice: Delaunay 算法本身不需要 differentiable。Gradient 从 mesh vertex 流回 Gaussians 是通过 (a) SDF values $f_{k,i}$ 和 (b) Delaunay vertex 坐标 $p_{k,i}$ (后者由 Gaussian 的 $\mu_k, R_k, s_k$ 表达)。Delaunay connectivity 只是一个 *discrete combinatorial structure*, 它每 500 iter 更新一次即可, connectivity 不参与 gradient flow。这跟 FlexiCubes / DMTet 的思路一致 — 把 topology 离散, geometry 连续。

---

## 3. Gaussian Pivots: 不是所有 Gaussians 都当 Delaunay vertex

### 3.1 为什么不能直接用所有 Gaussian centers

两个原因:
- (i) Gaussian centers 通常**正好落在 surface 上**, 但 Marching Tetrahedra 需要四面体**跨越** surface (即在 surface 两侧都有 vertex), 才能算出 sign change。如果所有 vertex 都在 surface 上, 算法退化。
- (ii) Large scene 有几 million Gaussians, 全部塞进 Delaunay 计算成本爆炸 (Delaunay 是 $O(N \log N)$ 到 $O(N^2)$ 之间, 实测 CGAL 在 1M 点以上明显变慢)。

### 3.2 第一个问题的解决: 9 个采样点 per Gaussian

借鉴 GOF, 每个 Gaussian $\mathcal{G}_k$ 生成 9 个 Delaunay vertices:

$$p_{k,i} = \mu_k + R_k \times (s_k \odot b_i), \quad i = 0 \dots 8$$

变量含义:
- $\mu_k \in \mathbb{R}^3$: $k$-th Gaussian 的 center
- $R_k \in \mathbb{R}^{3\times 3}$: 由 quaternion $q_k$ 构造的 rotation matrix, 表示 Gaussian 的主轴朝向
- $s_k \in \mathbb{R}^3$: Gaussian 的 scale vector (沿 3 个主轴的 std)
- $b_i \in \mathbb{R}^3$: 9 个 unit bounding box 标定点, 即 $\{(0,0,0), (\pm 1, \pm 1, \pm 1)\}$, 也就是 center + 8 corners
- $\odot$: Hadamard (element-wise) product

Intuition: 这 9 个点把 Gaussian 想象成一个 oriented bounding box 的 center 和 8 个 corner, 这些 corner 落在 Gaussian 椭球的 "外面", 保证 Delaunay 在 surface 周围形成 tetrahedra 跨越 surface。由于 corner 是用 $R_k, s_k$ 变换出来的, 它们 follow Gaussian 的 anisotropic shape, 所以 thin Gaussian (沿某一方向 scale 很小) 也会产生对应 thin 的 tetrahedra 结构。这跟 anisotropic meshing 的思路一致。

Gradient flow: $\partial L / \partial p_{k,i}$ 可以通过这个公式直接 chain rule 到 $\mu_k, R_k, s_k$ — 这是 mesh → Gaussian gradient 的主路径之一。

### 3.3 第二个问题的解决: Importance-weighted sampling

借鉴 Mini-Splatting2, 对每个 Gaussian 计算一个 importance score = average magnitude of blending coefficients across all training views。这个 score 反映了这个 Gaussian 对渲染的实际贡献。然后用 score 作为 sampling probability, 抽 subset (大约 0.1–0.5M 个) 当 pivots。

注意: 不是所有 Gaussians 都直接被 mesh loss 约束, 但通过 volume-to-surface consistency loss (渲染 Gaussians 和 mesh 比较 depth/normal), 所有 Gaussians 都被间接约束 — 因为 Gaussians 渲染的 depth 要跟 mesh 对齐, mesh 是从 sampled pivots 来的, 所以 non-sampled Gaussians 也得跟着 sampled 的走。

### 3.4 Base vs Dense 两种变体

- **Base**: 在 iteration 8000 时 prune 掉所有 non-sampled Gaussians, 只保留 0.1–0.5M, 直接当 Delaunay vertices。轻量, 但 render 质量略低。
- **Dense**: 保留所有 2–5M Gaussians 做 Gaussian splatting rendering, 但只用 sampled subset 作 Delaunay pivots。SDF 值学习更稳 (因为有更多 Gaussians 提供 rendering supervision), 但训练时间长。

这个区分很重要: 它把 "渲染表示" 和 "几何表示" 在数量上解耦了。Dense 的 SDF 学得好, 是因为 Gaussian field 更 dense, supervision 更强, 但 mesh vertex 数量依然保持低。

Mini-Splatting2: <https://arxiv.org/abs/2411.12788>

---

## 4. SDF Values: 每个 Gaussian 9 个 learnable scalars

### 4.1 设计

每个 Gaussian $\mathcal{G}_k$ 关联一个可训练向量 $f_k \in \mathbb{R}^9$, 9 个 SDF 值对应 9 个 Delaunay vertices (即 §3.2 中的 $p_{k,0..8}$)。

关键 design: **$f_k$ 与 Gaussian 的 opacity / scale / rotation / color 解耦**。Gaussian 自己的参数被 photometric loss 优化, SDF 是单独的一组参数被 mesh loss 优化。两套 gradient 不打架。

为什么不直接用 Gaussian opacity 当 SDF (像 SuGaR 那样)? 因为 opacity 在 training 中被 photometric loss 强烈 push, 它 fit image 的目标跟 fit geometry 的目标**不一致**。Decoupled SDF 可以被 mesh 一致性 loss 优化到一个跟几何真正对齐的状态, 而不受 opacity 的干扰。这是 paper 一个细节但关键的 design。

### 4.2 SDF normalization

实际上优化的是 truncated SDF normalized 到 $[-1, 1]$ via tanh。这样做有数值稳定性好处 — SDF 在 surface 附近可以非常大 (远离 surface 时值饱和到 ±1), 避免梯度爆炸。

### 4.3 SDF initialization

paper 提了一个 custom scalable depth-fusion 算法:
- 对每个 Delaunay site $p$, 渲染所有 training view 的 depth map
- 对每个 view, 计算 $p$ 在该 view 下的 depth 与 rendered depth 的差 (signed, 视点方向)
- 跨所有 view average, 作为初始 SDF

这是 TSDF fusion 的变种, 但作用在 Delaunay sites 而不是 regular grid 上。初始化是 over-smoothed, 但提供一个合理起点给 mesh-in-the-loop 优化。

---

## 5. Differentiable Marching Tetrahedra: 公式拆解

### 5.1 算法逻辑

给定 Delaunay tetrahedralization (每个 tetrahedron 有 4 个 vertices, 每个有 SDF 值):
- 遍历所有 tetrahedron
- 检查 4 个 vertex 的 SDF 符号
- 如果符号不全相同 (有正有负), surface 穿过这个 tetrahedron
- 在符号改变的 edge 上, linear interpolation 找到 surface 交点 (即 mesh vertex)
- 把这些交点连成 triangle(s)

每个 tetrahedron 根据符号 pattern 有 16 种 case, 但 GPU 实现时通常用 lookup table 一次性处理。

### 5.2 Vertex 位置公式

考虑一个 tetrahedron 的两个 Delaunay vertices $\mathcal{P}_{k,i}$ 和 $\mathcal{P}_{k',j}$ (来自 Gaussian $k$ 的第 $i$ 个 pivot 和 Gaussian $k'$ 的第 $j$ 个 pivot), SDF 值 $f_{k,i}$ 和 $f_{k',j}$ 符号相反。Mesh vertex $v_n$ 落在这条 edge 上:

$$v_n = \frac{f_{k,i} \, p_{k',j} - f_{k',j} \, p_{k,i}}{f_{k,i} - f_{k',j}}$$

变量含义:
- $f_{k,i} \in \mathbb{R}$: Gaussian $k$ 的第 $i$ 个 SDF 值 (在 $[-1, 1]$ 之间, 经过 tanh)
- $p_{k,i} \in \mathbb{R}^3$: Gaussian $k$ 的第 $i$ 个 Delaunay vertex 坐标 (公式 1)
- $v_n \in \mathbb{R}^3$: 提取出的 mesh triangle vertex 坐标

这是 1D linear interpolation: $v_n = p_{k,i} + t \cdot (p_{k',j} - p_{k,i})$, 其中 $t = f_{k,i} / (f_{k,i} - f_{k',j})$。SDF=0 的等值面就在这条 edge 上这个位置。

### 5.3 Gradient 路径

对 $v_n$ 求 gradient:
- $\partial v_n / \partial f_{k,i}$ 和 $\partial v_n / \partial f_{k',j}$ — 让 mesh loss 可以调整 SDF 值
- $\partial v_n / \partial p_{k,i}$ 和 $\partial v_n / \partial p_{k',j}$ — 让 mesh loss 可以调整 Delaunay vertex 位置, 进而 chain 到 $\mu, R, s$

后者是关键: mesh 渲染的 depth/normal loss → mesh vertex 位置 → Gaussian 中心/朝向/尺度。**这是 mesh 反过来 sculpt Gaussian 的物理通道**。Gaussian 被推向能让 mesh surface 干净的位置。

### 5.4 GPU 实现

paper 用 PyTorch 写的 custom GPU Marching Tetrahedra。Delaunay 用 CGAL (<https://www.cgal.org/>) 算, 每 500 iter 一次。CGAL 是非 differentiable 的, 但因为 connectivity 在 gradient 路径之外, 没问题。

---

## 6. Loss 函数全面拆解

总 loss:

$$\mathcal{L} = \mathcal{L}_{\text{vol}} + \mathcal{L}_{\text{mesh}} + \mathcal{L}_{\text{reg}}$$

### 6.1 Volumetric loss $\mathcal{L}_{\text{vol}}$ (公式 3, 4)

$$\mathcal{L}_{\text{vol}} = (1 - \lambda_{\text{RGB}}) \mathcal{L}_1 + \lambda_{\text{RGB}} \mathcal{L}_{\text{D-SSIM}} + \lambda_{\text{N}} \mathcal{L}_{\text{N}}$$

- $\mathcal{L}_1$: rendered image vs ground truth pixel-wise L1
- $\mathcal{L}_{\text{D-SSIM}}$: structural similarity, λ_RGB = 0.2
- $\mathcal{L}_{\text{N}}$: normal consistency, 让相邻 Gaussian 渲出的法向一致

$$\mathcal{L}_{\text{N}} = \sum_i \left( 1 - \mathbf{N}(i) \cdot \tilde{\mathbf{N}}(i) \right)$$

- $\mathbf{N}(i)$: pixel $i$ 通过 volume rendering 得到的 expected normal (Gaussian 法向按 opacity 加权)
- $\tilde{\mathbf{N}}(i)$: pixel $i$ 通过 finite difference on rendered depth map 得到的法向 (类似 2DGS)

这个 term 鼓励 Gaussian 排列成局部平面 — depth-based normal 和 splat-based normal 一致。

### 6.2 Volume-to-Surface consistency $\mathcal{L}_{\text{mesh}}$ (公式 5–7)

$$\mathcal{L}_{\text{mesh}} = \lambda_{\text{MD}} \mathcal{L}_{\text{MD}} + \lambda_{\text{MN}} \mathcal{L}_{\text{MN}}$$

**Depth consistency**:

$$\mathcal{L}_{\text{MD}} = \sum_i \log\left(1 + |D(i) - D_M(i)|\right)$$

- $D(i)$: pixel $i$ 渲自 Gaussians 的 depth
- $D_M(i)$: pixel $i$ 渲自 extracted mesh 的 depth (用 nvdifrast rasterize)

log(1 + ·) 是 Charbonnier penalty, 比 L1 在 small error 处更平滑, 比 L2 在 large error 处更 robust, 防止 outlier 主导 gradient。

**Normal consistency**:

$$\mathcal{L}_{\text{MN}} = \sum_i \left(1 - \tilde{\mathbf{N}}(i) \cdot N_M(i)\right)$$

- $\tilde{\mathbf{N}}(i)$: pixel $i$ 渲自 Gaussians 的 normal (finite difference on Gaussian depth)
- $N_M(i)$: pixel $i$ 渲自 mesh face 的 normal

这个 term 鼓励 Gaussian depth map 的法向和 mesh face 法向一致 — Gaussian 应该 "贴" 在 mesh 上, 法向对齐。

这两个 term 一起, depth 防止位置偏离, normal 防止方向偏离。Ablation (Tab. 5) 显示只用 depth 时 F1 = 0.46, 加上 normal 反而 F1 = 0.44 — 看似变差, 但 Fig. 7 视觉对比清楚说明 normal term 把 mesh noise 砍掉了, F1 这个 metric 对 noise 不敏感。

### 6.3 Anti-erosion loss $\mathcal{L}_{\text{erosion}}$ (公式 8)

$$\mathcal{L}_{\text{erosion}} = \sum_{g \in G_{\text{Del}}} \max(0, f_{\mu_g})$$

- $G_{\text{Del}}$: 被选作 Delaunay pivot 的 Gaussian 集合
- $f_{\mu_g}$: Gaussian $g$ 的 center (即 $b_0 = (0,0,0)$ 对应的 pivot) 的 SDF 值

**Intuition**: hinge loss。如果 Gaussian center 的 SDF 是正的 (在 surface 外), 给 penalty; 是负的 (在 surface 内), no penalty。

为什么需要这个? **Erosion 问题** — 当一个 tetrahedron 内所有 SDF 值变正, mesh 在这个 tetrahedron 内消失, thin structure (fence, spoke, leaf) 直接没了。Gradient 也变弱 (mesh 不在了, depth/normal 渲染都没 signal)。一旦发生, 难以恢复。

把 Gaussian center 拉进 surface 内 (SDF < 0), 保证每个 tetrahedron 至少有一个负 SDF vertex, 给 Marching Tetrahedra 提供 "锚点", surface 在这区域肯定存在。

注意只对 pivot Gaussian 的 center 应用, 不对其他 8 个 corner pivot 应用 — 否则可能让整个 mesh collapse 到 Gaussian center 上。

### 6.4 Interior regularization $\mathcal{L}_{\text{interior}}$ (公式 9)

$$\mathcal{L}_{\text{interior}} = \sum_p H\left(\sigma(-f_p), o_p\right) \cdot o_p$$

- $p$: Delaunay site
- $f_p$: site $p$ 的 SDF 值
- $o_p \in \{0, 1\}$: occupancy label, 1 表示 site 在 mesh 内部
- $\sigma$: sigmoid
- $H$: cross-entropy

展开: $H(\sigma(-f_p), 1) = -\log(\sigma(-f_p))$, 当 $f_p$ 越大 (越在 surface 外), 这个值越大。乘以 $o_p$ 表示只对 inside 点应用。

**Occupancy label 怎么算**: 
- 用 mesh 渲所有 training view 的 depth map
- 对每个 Delaunay site $p$, 如果它在所有包含它的 view 里都被 depth map 遮挡 (i.e. site 在 rendered depth 后面), 则 $o_p = 1$ (在内部)
- 否则 $o_p = 0$ (在外部或 visible)

每 200 iter 更新一次 occupancy label (因为这步要遍历所有 views 和所有 sites, 比较慢)。

**Intuition**: depth/normal loss 只能监督 visible surface, 看不见的内部没监督。内部容易出现 chaotic cavities (Fig. 8a)。这个 loss 把内部点拉成 SDF < 0 (内部应该是 solid, 在 surface 内), 让 mesh 内部 watertight 且 empty。对下游 physics simulation 关键 — 内部 cavities 会让 fluid simulation 出 weird 结果。

### 6.5 训练 schedule

- **Iter 0–3000**: aggressive densification (Mini-Splatting2 风格), 只有 photometric loss
- **Iter 3000**: 引入 $\mathcal{L}_N$
- **Iter 3000–8000**: refine Gaussians, $\mathcal{L}_{\text{vol}}$
- **Iter 8000**: 停止 densification 和 pruning。Base model 在这里 prune 到 0.1–0.5M (importance-weighted)
- **Iter 8000**: 开始 mesh extraction, 应用 full $\mathcal{L}$
- **Iter 8000–18000**: 10k iter 的 mesh-in-the-loop 优化

总 18k iter, base model 在 DTU 上 25 min, T&T 上 40–50 min (单 4090)。Dense 上 unbounded scene 最多 2 小时。

超参数: $\lambda_{\text{RGB}} = 0.2$, $\lambda_N = \lambda_{MD} = \lambda_{MN} = 0.05$, $\lambda_{\text{erosion}} = \lambda_{\text{interior}} = 0.005$。

---

## 7. 实验数据细致解读

### 7.1 Resources (Tab. 1)

| Method | #Gauss (M) | GPU Mem (GB) | Time | #Verts | #Tris | Size (MB) |
|---|---|---|---|---|---|---|
| 2DGS | 0.98 | 4.7 | 29m | 16.39M | 21.68M | 557 |
| GOF | 1.55 | 10.6 | 93m | 16.49M | 33.17M | 600 |
| RaDe-GS | 1.56 | 12.4 | 42m | 14.75M | 29.59M | 592 |
| **MILo base** | **0.28** | 10.0 | 50m | **4.36M** | **8.97M** | **180** |
| MILo dense | 2.11 | 16.5 | 110m | 6.89M | 13.79M | 276 |

**关键观察**: MILo base 用了大约 GOF 1/6 的 Gaussians, 1/4 的 vertices, 1/3 的存储, 同时 F1 更高。Vertex 数量是 1/4, 这是 paper 标题 "Detailed and Efficient" 的核心 quantitative 证据。

为什么 vertex 这么少还能 quality 更高? 因为 in-loop 优化让 mesh topology 动态适应 surface 真实结构, 而不是 post-hoc 的 fixed-grid TSDF 那样不管 surface 复杂度均匀 dense。

### 7.2 T&T F1 score (Tab. 2)

平均 F1: MILo base = 0.47, GOF = 0.46, 2DGS = 0.30, Neuralangelo = 0.50 (implicit SOTA, 但 >24h 训练)。

MILo dense = 0.49, 是 explicit methods 里最好的。注意 MILo base 已经超过 GOF, 即使 vertex 数量只有 1/4。

### 7.3 DTU Chamfer Distance (Tab. 3)

DTU 是 object-centric controlled scenes, post-hoc 本来就 work 得好。MILo base mean CD = 0.68, 跟 GOF (0.74)、RaDe-GS (0.68) 同档, 比 2DGS (0.80) 好。MILo 在 DTU 上没有 T&T 上那么突出, 因为 DTU 的 challenge 主要不在 mesh extraction, 而在 fit object 的几何细节, 那里 GOF 之类方法已经做得很好。MILo 的优势在 large complex scene。

### 7.4 Mesh-Based Novel View Synthesis (Tab. 4) — 这是个新 metric

paper 提了一个新 evaluation protocol 来衡量 mesh 质量 (尤其 background), 因为:
- DTU: 只有 object GT, 没有 background
- T&T: 只有 foreground GT
- MipNeRF360: 完全没有 geometry GT

方法:
1. 对每个 mesh, 训练一个 neural color field $F_{\text{color}}: \mathbb{R}^3 \to [0,1]^3$ (用 TensoRF backbone, 5k iter)
2. 渲 mesh + color field 到 test view, 比 PSNR/SSIM/LPIPS
3. **关键**: color field 而非 vertex color, decouple color from mesh resolution — 避免稀疏 mesh 因 vertex 颜色分辨率低而吃亏

直觉: 如果 mesh geometry 错位/缺失/有 erosion, color field 怎么训都 fit 不好 reference image, PSNR 低。这是一个 image-space 的 geometry 评估 proxy。

结果: MILo base 在 MipNeRF360 上 PSNR=24.09 (best), SSIM=0.6885 (best), LPIPS=0.3235 (best)。在 DeepBlending 上 PSNR=28.04, SSIM=0.8336, LPIPS=0.2285, 全面最佳。

注意 MILo base 在 MipNeRF360 上甚至比 GOF 更好 (24.09 vs 20.78), 即使 vertex 数量是 GOF 的 1/5 (6.73M vs 32.80M)。这强烈支持 in-loop 优化带来的几何质量提升。

TensoRF: <https://apchenstu.github.io/TensoRF/>

### 7.5 Novel View Synthesis (Tab. 6)

GS rendering 质量 (非 mesh): MILo base indoor PSNR 29.96, outdoor 24.47。略低于 SOTA (GOF 30.74 indoor / 25.17 outdoor), 因为 base 用了更少 Gaussians (0.46M vs 2.99M)。Dense 提升到 30.76 / 24.81, 与 GOF 持平。

**重要 insight**: mesh-in-loop regularization 让 Gaussian 表示为几何 "服务", 渲染质量略降但几何质量大幅提升。这是 trade-off, 用户根据下游应用选 base 或 dense。

### 7.6 Ablation (Tab. 5)

- Baseline (无 mesh loss): F1 = 0.41
- + $\mathcal{L}_{\text{MD}}$: 0.46 (depth supervision 大涨)
- + $\mathcal{L}_{\text{mesh}}$ (depth + normal): 0.44 (F1 略降, 但视觉 Fig. 7 显示 noise 显著降低)
- + $\mathcal{L}_{\text{erosion}}$: 0.44 (F1 几乎不变, 但 thin structure 视觉恢复, Fig. 5)
- + $\mathcal{L}_{\text{reg}}$ (full): 0.47
- Full with GOF backbone: 0.49

Normal 和 erosion loss 在 F1 上贡献不大, 但视觉质量贡献大 — 暴露了 F1 score 对 mesh noise / thin structure 缺失不敏感的弱点。Paper 提的 Mesh-Based NVS metric 是对这弱点的补充。

### 7.7 跟 TSDF fusion 对比 (Fig. 9)

TSDF 在 fixed 3D grid 上融合 depth maps, 内存随 resolution 立方增长, 在 large scene 上 OOM (Courthouse 场景直接跑不了)。MILo 的 learnable SDF 在 Delaunay sites 上, 数量随 Gaussian 数量线性, 完全 scalable。在不同 vertex 预算下 MILo 都比 TSDF 高 (Fig. 9 ablation)。

---

## 8. 核心直觉总结

### 8.1 为什么这个 idea 现在才 work?

NeRF 时代做不了, 因为 NeRF 的 geometry 隐式编码在 MLP 里, 没有显式参数可以让 mesh gradient "推回去"。GS 出现后, Gaussian 的 $\mu, R, s$ 是 explicit 参数, mesh vertex position 可以直接 chain rule 回去。这是一个 representation 选择决定 algorithm possibility 的好例子。

### 8.2 为什么 Delaunay 不需要 differentiable?

Delaunay connectivity 是 discrete combinatorial decision, 每 500 iter 重新算一次。Gradient 通过 mesh vertex 位置 (连续) → Gaussian 参数 (连续), 不经过 connectivity (离散)。这跟 DMTet、FlexiCubes 的思路一致。如果硬要让 Delaunay differentiable, 需要处理 topology change 的 gradient, 极其复杂且通常没有意义的 gradient signal。

FlexiCubes: <https://research.nvidia.com/labs/toronto-ai/flexicubes/>

### 8.3 SDF decoupling 是关键 design

如果 SDF 跟 opacity 耦合 (像 SuGaR 用 opacity 当 indicator), Gaussian 优化图像时 SDF 也被推着变, 几何被图像 loss 污染。Decoupled SDF 只被 mesh supervision 优化, 图像 loss 只优化 Gaussian 自己的参数, 两套 gradient 互不干扰, 各自负责各的事。这是一个 dual-representation co-optimization 的 clean 设计。

### 8.4 Interior regularization 的 feedback loop

$\mathcal{L}_{\text{interior}}$ 用当前 mesh 算 occupancy label, 再用 label 监督 SDF。这是一个 *self-supervised feedback loop*: mesh 自己决定哪些点在内部, 然后用这个信息约束 SDF 生成下一个 mesh。类似于 EM 算法, E-step 算 label, M-step 用 label 优化参数。

每 200 iter 才更新 label 是因为这个 step 涉及全 views × 全 sites 的 ray test, 计算重。但 label 在 200 iter 内变化不大, 这个 lazy update 在效果和效率间取得平衡。

### 8.5 为什么 vertex 数量大幅降低?

Post-hoc isosurface extraction (Marching Cubes on regular grid) 在 surface 附近均匀 dense, 因为 grid resolution 固定。Delaunay + Marching Tetrahedra 在 Gaussian 密集处 dense (complex geometry)、稀疏处 sparse (flat background)。In-loop 优化还让 Gaussians 主动 reposition 到 surface 真正需要的地方, 而不是被图像 loss 推到各处当 floater 用。

---

## 9. Limitations 和潜在 extension

paper 自己提的:
- 训练时间增加 (虽然仍 manageable)
- 初始 Gaussian 分布影响质量

我额外看到几个:
- **Dynamic scene**: 现在 framework 是 static 的。Dynamic Gaussian (Deformable 3DGS 之类) 要 in-loop mesh 需要每帧重新 Delaunay, 计算量大。潜在方向: 时间一致的 Delaunay。
- **Differentiable rendering on mesh**: nvdifrast 是 rasterization-based, 处理 transparency / semi-transparency (植被、玻璃) 弱。Path tracing based differentiable renderer 在 mesh 上会更准但更慢。
- **SDF learning stability**: SDF 是 Gaussian-wise 独立学习, 缺乏全局 SDF 一致性约束 (monotonic to surface)。一个潜在 extension 是加 Eikonal regularization $\|\nabla f\| = 1$, 但需要在 tetrahedra 上算 spatial gradient, 工程上不 trivial。
- **Adaptive sampling for pivots**: 现在 importance 是渲染贡献, 但更聪明的策略可以是 "几何复杂度" (curvature 估计), 让 pivots 集中在 high-curvature 区域。

---

## 10. 跟相关工作的位置关系

| 类别 | 代表 | 跟 MILo 关系 |
|---|---|---|
| Pure post-hoc GS | 2DGS, GOF, RaDe-GS, SuGaR | MILo 是 in-loop 版, 把它们的 mesh extraction 提前到 training |
| NeRF + SDF | NeuS, VolSDF, Neuralangelo | 完全 implicit, 训练慢, MILo 训练快 50x |
| GS + implicit SDF | GaussianUDF, GS-Pull, GSDF | 两个表示同时优化, MLP bottleneck; MILo 用 explicit SDF per Gaussian |
| Surface-based NVS | BakedSDF, Binary Opacity Grids | 先 mesh 再 bake color, 两阶段; MILo 启发了它们的 mesh-based NVS evaluation |
| Delaunay-based NVS | Radiant Foam | Delaunay in-loop, 但只做 NVS 不做 surface reconstruction; MILo 借鉴这个 idea 但加 mesh supervision |

关键 differentiation: MILo 是**第一个**把 differentiable mesh extraction 嵌入 GS training loop 并做 surface reconstruction 的工作。之前的 GS-mesh 工作 (SuGaR, GOF) 都是先训 GS 再 extract, 即使有 refinement stage (SuGaR-Refine), mesh topology 也是 fixed 的 post-extraction, 不能动态适应。

Radiant Foam: <https://arxiv.org/abs/2502.01157>
Binary Opacity Grids: <https://research.nvidia.com/labs/toronto-ai/binary-opacity-grids/>
GaussianUDF: <https://arxiv.org/abs/2406.14351>

---

## 11. 工程上值得学习的细节

1. **CGAL for Delaunay**: 用 CGAL 的 3D Delaunay, robust 且 efficient。Python binding 通过 SWIG 或 PyCGL。
2. **nvdifrast for mesh rasterization**: NVIDIA 官方 differentiable rasterizer, 支持 depth/normal 渲染 + backward。
   - nvdifrast: <https://nvlabs.github.io/nvdifrast/>
3. **Lazy Delaunay update**: 500 iter 一次, connectivity 不变期间 gradient 仍然有效。
4. **Lazy occupancy update**: 200 iter 一次, 平衡准确度和效率。
5. **SDF via tanh normalization**: 数值稳定性, 截断到 $[-1, 1]$。
6. **TensoRF for mesh-based NVS eval**: 一个 neural color field, 跟 mesh 几何 decouple, 不 bias dense mesh。

---

## 12. 我的 take

这篇 paper 我觉得最有价值的 conceptual contribution 不是 mesh 质量提升 (虽然显著), 而是把 **surface extraction from a "conversion" task 变成 a "regularization" task**。Post-hoc mesh extraction 是一次性 conversion, 错了就错了。In-loop mesh extraction 是持续的 geometric prior, 它把 Gaussians 拉向一个 "surface-friendly" 的 manifold, 同时让 mesh 也能跟着 Gaussians 走。这种 bidirectional coupling 让两个 representation 互相 sculpt, 最终结果比任何一个单独优化都好。

这种思路其实在 graphics 和 ML 里都见过 (mesh + image joint optimization, differentiable physics + neural net 等等), 但在 GS surface reconstruction 这个具体场景里被组合得很干净, 而且 loss 设计上 (erosion, interior) 处理了几个非常 specific 的 failure mode, 工程上很扎实。

如果你 (Andrej) 想做类似直觉的 downstream, 我会想到: 同样的 in-loop differentiable extraction 思路可以用在 **point cloud → mesh** (open surface), **NeRF → mesh** (但需要把 NeRF distill 到 Gaussian intermediate), 或者 **动态场景的 mesh sequence** (consistent topology across time)。MILo 给了一个 framework 模板, 关键是找到合适的 differentiable extraction primitive。

希望这个拆解 build 了你想要的 intuition。如果想深挖某个具体 piece (比如 Marching Tetrahedra 的 GPU 实现, 或者 SDF 学习的 dynamics, 或者 occupancy label 计算的并行算法), 我可以继续展开。

补充参考:
- MILo paper (arXiv 应该会出): <https://anttwo.github.io/milo/>
- 3DGS original repo: <https://github.com/graphdeco-inria/gaussian-splatting>
- 2DGS repo: <https://github.com/hbb1/2d-gaussian-splatting>
- GOF repo: <https://github.com/autonomousvision/gaussian-opacity-fields>
- SuGaR repo: <https://github.com/Anttwo/SuGaR>
- nvdifrast: <https://github.com/NVlabs/nvdifrast>
- CGAL: <https://www.cgal.org/>
