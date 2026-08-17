---
source_pdf: EDGS.pdf
paper_sha256: 15261558501d9a0c218c0f66b98ae4e876d43e38c884db85e16c778143843a08
processed_at: '2026-08-04T01:37:17-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EDGS 说人话版

## 1. 原来 3DGS 怎么干活的

打个比方, 3DGS 像雕塑家捏泥人:
- 先搭铁丝骨架 (SfM sparse 点云, 大概几万个点)
- 开始糊泥巴 (优化 Gaussian 参数)
- 发现哪里缺料, 补一块 (densification)
- 反复糊反复补, 直到成型

这个"发现缺料再补"就是 densification。痛点在于:
- 你得先糊一阵子, 才能发现哪里料不够 (gradient 信号积累需要时间)
- 补完新料还得继续糊, 让它融入
- 整个过程像挤牙膏, 慢

每个 Gaussian 在 parameter space 里走一条很长的弯路, 才最终落到合适位置。

## 2. EDGS 的核心 insight

作者有个漂亮的类比:
- **Sculptor** 是减法思维, 从粗到细逐步 refine
- **Camera** 是一次性 capture, 所有光线同时到达 sensor

相机并不"逐步发现"细节, 一次曝光全部进来。所以 3DGS 模仿 sculptor 的方式其实是反直觉的 — 既然 2D images 里已经有全部 dense 信息, 为什么不直接"投影"到 3D 作为初始化, 让 photometric optimization 只做 refinement?

## 3. 具体怎么干

EDGS 的 pipeline 用人话讲就四步:

**Step 1**: 拿一张参考图 $I^i$, 找它附近几张图 $I^j$ (相机位置近的, overlap 大)

**Step 2**: 用一个预训练的 dense matcher (叫 RoMa) 找这两张图里哪些 pixel 对应同一个 3D 点。输出是个 dense warp field $\mathcal{W}^{j \to i}$, 告诉你图 $I^j$ 里每个 pixel $(u_k^j, v_k^j)$ 对应到 $I^i$ 里哪个位置

**Step 3**: 对每对 matched pixel, 用三角化算它在 3D 空间的坐标 $\mathbf{g}_k^x = (x_k, y_k, z_k)$

**Step 4**: 把这些 3D 点变成 Gaussian, 一次性有几百万个, 整个场景铺满, 然后开始标准优化, 但禁掉 densification

## 4. 三角化到底是啥

你两只眼睛看同一个点, 因为两眼位置不同, 点在两眼视网膜上位置不同, 大脑用这个差异反推深度。

EDGS 干的是机器版:
- 图 A 里的 pixel $(u_k^i, v_k^i)$ 和图 B 里的 pixel $(u_k^j, v_k^j)$ 对应同一个 3D 点
- 从相机 A 画条射线穿过这个 pixel
- 从相机 B 画条射线穿过那个 pixel  
- 两条射线在 3D 空间的交点就是 3D 点位置

数学上, 这变成解一个线性方程组。每个相机给出两个约束 (来自 homogeneous 坐标消去分母后的 $u$ 和 $v$ 方程), 两个相机共 4 个方程, 但 $\mathbf{g}_k^x$ 只有 3 个 unknown, 所以 over-determined, 用 least squares 解:

$$\mathbf{g}_k^x = \arg\min_{\mathbf{g}_k^x} \|\mathbf{A} \mathbf{g}_k^x + \mathbf{b}\|^2$$

这里 $\mathbf{A} \in \mathbb{R}^{4 \times 3}$ 的每一行来自一个 pixel 的 projection equation, $\mathbf{A}^T$ 的四行分别是:
- $\mathbf{P}_{col,0}^i - u_k^i \mathbf{P}_{col,2}^i$ (相机 A 的 $u$ 约束)
- $\mathbf{P}_{col,1}^i - v_k^i \mathbf{P}_{col,2}^i$ (相机 A 的 $v$ 约束)
- $\mathbf{P}_{col,0}^j - u_k^j \mathbf{P}_{col,2}^j$ (相机 B 的 $u$ 约束)
- $\mathbf{P}_{col,1}^j - v_k^j \mathbf{P}_{col,2}^j$ (相机 B 的 $v$ 约束)

其中 $\mathbf{P}_{col,m}$ 是 projection matrix $\mathbf{P}$ 的第 $m$ 列。实际两条 ray 因噪声通常 skew (不精确相交), 解的是两 ray 公垂线中点。EDGS 不要求这个点精确, 只要"足够好"让后续 optimization refine 就行。

这就是 textbook 的 DLT triangulation, Hartley & Zisserman 的 *Multiple View Geometry* Chapter 12 有完整推导。

## 5. 其他参数怎么 init

- **Color** $\mathbf{g}_k^c$: 两个 matched pixel RGB 值的 average
- **Scale** $\mathbf{g}_k^s$: Gaussian 到两个 camera 的最小距离 (离 camera 远的 Gaussian 应该更大, 因为同样 angular 范围对应更大 spatial extent)
- **Rotation** $\mathbf{R}_k$: identity matrix (偷懒)
- **Opacity** $\alpha_k$: 3DGS 默认 (0.1 经 sigmoid)

## 6. 为什么这个 init 这么 work

论文做了个特别 informative 的 ablation: 量每个 Gaussian 从开始到结束走了多远。

两个量:
- **Final displacement**: $\|\mathbf{g}_i^x(0) - \mathbf{g}_i^x(T)\|_2$ (起点到终点直线距离)
- **Total path length**: $\sum_{t=0}^{T} \|\mathbf{g}_i^x(t) - \mathbf{g}_i^x(t+1)\|_2$ (整个优化轨迹长度)

结果惊人:
- 3DGS coordinate travel distance 比 EDGS 大 **50 倍**
- 3DGS coordinate total path length 比 EDGS 长 **30 倍**
- Color path length 只短约 **2 倍**

**Intuition**: 原版 3DGS 的 Gaussian 大部分时间在"漂移找位置", densification 反复 clone/split 让位置和 scale 大幅震荡。EDGS 一开始就在 surface 附近, 几乎直线收敛到附近的 minimum。

Color 只短 2 倍的原因: EDGS 用 2-view average init color, 但 surface 真实 radiance 在 view-dependent reflection 区域差异大, color 还得自己学。Coordinate 是几何确定性的, init 准了 optimization 几乎不动。

## 7. Noise robustness 验证了 thesis

作者在 init 上加 $\epsilon \sim \mathcal{N}(0, \sigma)$:
- **Color noise**: 极其鲁棒, $\sigma=0.15$ 都几乎不掉点
- **Coordinate noise**: 敏感, $\sigma=0.05$ 就开始 degrade

这与 motion ablation 完美一致 — coordinate 决定 geometric structure, 一旦偏了 photometric loss 难拉回; color 是 photometric loss 直接 supervise 的, optimization 容易修正。

**关键 takeaway**: EDGS 的核心价值是 **geometric 初始化**, color init 只是 bonus 加速。

## 8. 效果数据 (Mip-NeRF360)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Time | #G (M) |
|--------|-------|-------|--------|------|--------|
| 3DGS (original) | 27.21 | 0.815 | 0.214 | 42 min | 3.5 |
| 3DGS* (re-trained) | 27.49 | 0.816 | 0.215 | 26 min | 2.8 |
| Mip-Splatting | 27.97 | 0.842 | 0.176 | 26 min | 4.0 |
| 3DGS-MCMC | 28.15 | 0.838 | 0.176 | 20 min | 3.2 |
| **EDGS + 3DGS 30K** | 27.80 | 0.840 | **0.175** | 29 min | **1.9** |
| **EDGS + Taming 3DGS 30K** | **28.06** | 0.839 | **0.174** | **16 min** | 3.2 |
| Taming 3DGS | 27.71 | 0.820 | 0.207 | 14 min | 3.2 |
| MiniSplatting | 27.25 | 0.820 | 0.217 | 12 min | 0.5 |
| **EDGS + 3DGS 5K** | 26.70 | 0.820 | 0.202 | 8 min | 2.9 |
| **EDGS + Taming 3DGS 5K** | 26.89 | 0.825 | **0.195** | **6 min** | 2.8 |

翻译成人话:
- EDGS 5K 步就达到 3DGS 30K 步的 LPIPS, 训练快约 **3 倍**
- 用 Gaussian 数量少 **40%**, 推理时内存和速度更好
- LPIPS/SSIM 提升明显 (感知/结构指标), PSNR 提升小 (pixel-level)

PSNR 提升小的原因: EDGS 用 2-view color average init, 对 view-dependent 反射表面 (镜面、玻璃、水) 建模不力, 这影响 PSNR。但 high-frequency texture 重建更好, 所以 LPIPS 大幅改善。

## 9. Densification 影响的 paradox

Table 2 有个有意思的对比:
- 3DGS (Random init, **w/ Densification**): 22.19 PSNR — 远差
- 3DGS (COLMAP init, **w/ Densification**): 27.49 PSNR
- EDGS (RoMa init, **w/o Densification**): 27.80 PSNR
- EDGS (RoMa init, **w/ Densification**): 27.84 PSNR (只提升 0.04, 但 splat 翻倍)

两点 takeaway:
1. Random init + Densification 远不如 COLMAP + Densification — densification 是 local operation, 拉不回完全错位的 init
2. EDGS 加 densification 几乎没收益, 但 splat 翻倍 — diminishing return 极陡

这强烈支持论文 thesis: **densification 只是个 hack, 真正缺的是 dense geometric prior; 有了它, densification 是冗余甚至有害的**。

## 10. 这篇 paper 真正厉害的三层

**第一层 (表面)**: 训练快 3 倍, 省 40% 内存

**第二层 (方法)**: 用 2D pre-trained matcher 作为 3D prior, 把传统 MVS (Multi-View Stereo) 的 triangulation 思想引入 3DGS 初始化

**第三层 (insight)**: 挑战了 3DGS 的隐藏假设 — "必须从 sparse 开始慢慢长"。3DGS 社区一直在研究"怎么让 densification 更好" (AbsGS 改 gradient, MCMC 重新 formulate, MiniSplatting 改 sampling), 但 EDGS 说"你们都在改一个根本不需要的步骤"。

这就像大家都在研究怎么让马鞍更舒服, 有人突然说不如造个汽车。

## 11. 几个有意思的 connection

**与 MVS 的关系**: 传统 MVS pipeline 是 SfM 给 sparse 点 → dense MVS 给 dense depth map → 重建 mesh。3DGS 实际上把 dense MVS 步骤省了, 用 densification 代替。EDGS 本质是把 dense MVS 补回来, 但直接输出 Gaussian 而非 depth map。

**与 RAIN-GS 的对比**: RAIN-GS 证明 random init 也能 work, 乍看与 EDGS 矛盾。但 Table 1 显示 RAIN-GS 在 Mip-NeRF360 上只有 22.23 PSNR (远低于 EDGS 的 27.80)。RAIN-GS 仍然依赖 densification 把 random Gaussians 拉到正确位置, 还是 sculptor 思维。EDGS 给 dense geometric prior, 直接 teleport 到正确位置。

**与 Rad-Splat 对比**: Rad-Splat 用 pretrained NeRF 提取点云做 init, 但需要 **9 小时** NeRF 预训练。EDGS 直接用 2D matcher (RoMa) 跳过 NeRF, 时间从 9h 降到几分钟。这反映一个 design philosophy: NeRF 隐式表示对 init 是 overkill, 2D correspondence 已经包含足够 geometric 信息。

**2D prior 是 cheat code**: EDGS, MVSplat, DepthSplat 都在 leverage 2D pretrained models 作为 3D prior。这与 SfM 点云作为 prior 是 paradigm shift。2D models 见过海量数据, 直接做 dense correspondence / depth 几乎 free lunch。

## 12. 局限性

1. **Reflective / view-dependent surfaces**: 用 2-view color average, 反光物体不同角度看颜色不同, 平均值错。可能的改进: 用更多 view 的 color 拟合 spherical harmonics 系数

2. **Sparse view**: RoMa 在 baseline 大时性能下降。Supplementary 提到 16 帧能 work, 更稀疏可能崩, 因为 triangulation 的数值 conditioning 在 ray 角度小时恶化

3. **Under-covered regions**: 如果 keypoint sampling 没覆盖到某区域, 该区域无法收敛。论文强调 uniform sampling 比 high-confidence mining 重要 — 这是 dense MVS 的 wisdom

4. **Memory**: Dense init 一次性分配几百万 Gaussians, 早期训练 memory footprint 更大, 但 final 更少

## 13. 值得想的 open questions

1. **Uncertainty-aware init**: RoMa 本来就输出 confidence $\mathbf{c}^{ij}$, 可以做 weighted triangulation 或 weighted init opacity (低 confidence 区域 init opacity 低, 让 optimization 自由调整)

2. **SH coefficients init**: 现在只 init DC component (color), SH 高阶项从零开始。可以用 multi-view matched colors 反推 SH, 解决反光表面问题

3. **Init rotation/scale**: 现在用 identity rotation 和距离-based scale。如果用 local surface normal 估 rotation, 用 depth gradient 估 scale, 会不会更进一步?

4. **Dynamic scene**: 4DGS 能否用 EDGS-style init? 时间维 correspondence 更复杂

5. **Adaptive sampling**: 现在 uniform sampling, 但 high-frequency 区域应该更密。用 image gradient 或 frequency 分析做 importance sampling

## 14. 给你的 takeaway, Andrej

我猜你作为教育者会关注几个点:

1. **Photometric loss landscape 的 geometry**: motion ablation 间接揭示 3DGS optimization 的 difficulty 在"找到正确位置"而非"refine"。给 lecture 讲 3DGS 时这是好 motivation 例子

2. **Simple idea + careful execution**: 数学上是 textbook DLT triangulation, 价值在 system-level insight 和验证 (matcher 选择, robustness study, 与 acceleration methods 组合)。好 research 不需要 novel math, 需要 novel question ("can we skip densification?")

3. **2D prior 是 cheat code**: 把 2D pretrained models 当 3D prior 用, 是这一波 3DGS 加速工作的共同 pattern

---

## Web References

- **EDGS project page**: https://compvis.github.io/EDGS/
- **3DGS original**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **RoMa matcher**: https://github.com/Parskatt/RoMa
- **Mip-NeRF360 dataset**: https://jonbarron.info/mipnerf360/
- **Taming-3DGS**: https://github.com/mallick-sashwat/taming-3dgs
- **3DGS-LM**: https://github.com/uhugl/3DGS-LM
- **EAGLES**: https://github.com/Sharath-girish/efficient-gaussian-splatting
- **AbsGS**: https://github.com/MonsPla/3DGS-with-Absolute-Precision
- **Scaffold-GS**: https://city-super.github.io/scaffold-gs/
- **3DGS-MCMC**: https://maturk.github.io/3DGS-MCMC/
- **DepthFM**: https://github.com/CompVis/depth-fm
- **LoFTR**: https://zju3dv.github.io/LoFTR/
- **DKM**: https://github.com/Parskatt/DKM
- **RAFT**: https://github.com/princeton-vl/RAFT
- **RAIN-GS**: https://maincold2.github.io/raings/
- **Rad-Splat**: https://research.nvidia.com/labs/toronto-ai/radsplat/
- **MVSplat**: https://github.com/donydchen/mvsplat
- **DepthSplat**: https://arxiv.org/abs/2410.13862
- **gsplat**: https://github.com/nerfstudio-project/gsplat
- **Tanks&Temples**: https://www.tanksandtemples.org/
- **Deep Blending**: https://hdr-2018-22.github.io/
- **Hartley & Zisserman (DLT triangulation textbook)**: https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/
- **Spec-Gaussian**: https://github.com/ingra14a/Spec-Gaussian

想深挖某块 (DLT 在 sparse view 下的 numerical conditioning, 或 SH init 推导), 喊一声。

---

# EDGS: Eliminating Densification for Efficient Convergence of 3DGS — 深度技术解析

Andrej, 这篇 paper 我觉得核心 insight 很有意思, 它挑战了 3DGS 一个被默认接受的"基本设定": 必须从 sparse SfM 点云开始, 通过 iterative densification 逐步加密。EDGS 直接用 dense 2D correspondences + triangulation 一步到位初始化, 然后跳过 densification。让我把里面的技术细节、math、实验数据都拆解给你, 帮你 build intuition。

---

## 1. 核心 Motivation 的 Intuition

3DGS 的 densification 流程本质上是一个 **"discovery-then-fill"** 的过程:
1. 从 sparse SfM keypoints 初始化 (~10^4 个 Gaussian)
2. 训练若干步, 累积 photometric loss 的 gradient norm
3. gradient norm 超过 threshold 的区域被标记为"under-reconstructed"
4. 在这些区域 split / clone Gaussians
5. 重复 2-4

这里有个根本性的 efficiency 问题: Gaussian 必须先"调整自己一段时间"才能让 gradient signal 暴露出"我这里需要更多 splat"。这意味着每个 Gaussian 在 parameter space 中走过一条很长的优化路径, 最终才落到合适的位置。

EDGS 作者提了一个非常好的哲学类比:
- **Sculptor** 是减法思维: 从 rough block 开始, 逐步 refine 细节
- **Camera** 是一次性 capture: 所有光线同时到达 sensor, 一次曝光获取全部细节

Camera 并不"逐步发现"细节, 它一次 capture 所有。所以 3DGS 模仿 sculptor 的方式是反直觉的, 应该利用 2D images 中已经存在的 dense 信息, 把它们"投影"到 3D 空间作为初始化, 然后让 photometric optimization 做 refinement。

这是一个非常简单的 idea, 但 execution 里有几个关键 engineering 决策让它 work。

---

## 2. 方法 Pipeline 详解

### 2.1 Dense Correspondence Extraction

对每个 reference image $I^i$, 找到 K 个最近邻 $\mathbb{I} = \{I^1, \dots, I^J\}$, 用 pretrained matcher $\mathcal{M}$ (RoMa [1]) 计算 dense warp field:

$$\mathcal{M}(I^i, I^j) \rightarrow \mathcal{W}^{j \to i}, \mathbf{c}^{ij}$$

其中:
- $\mathcal{W}^{j \to i} \in \mathbb{R}^{2 \times H \times W}$: dense warp field, 对 $I^j$ 中每个 pixel $(u_k^j, v_k^j)$ 给出在 $I^i$ 中对应位置 $\mathcal{W}^{j \to i}(u_k^j, v_k^j)$
- $\mathbf{c}^{ij} \in \mathbb{R}^{H \times W}$: 每 pixel 的 correspondence confidence

**Engineering 细节**: 相机邻近性用 projection matrix 的 Frobenius norm 距离, 因为 intrinsics 一致所以只看 extrinsics 差异。

### 2.2 Triangulation — 数学核心

给定一对 matched pixels $(u_k^i, v_k^i)$ 和 $(u_k^j, v_k^j)$, 对应 3D point $\mathbf{g}_k^x = (x_k, y_k, z_k)$, 我们用 **DLT (Direct Linear Transform)** triangulation:

对每个 camera 的 projection equation:
$$[\mathbf{g}_k^x, 1]^T \mathbf{P}^i = w_k^i [u_k^i, v_k^i, 1]^T$$

其中 $\mathbf{P}^i \in \mathbb{R}^{3 \times 4}$ 是 3x4 projection matrix (intrinsics × extrinsics), $w_k^i$ 是 homogeneous scale。

把第三行 (z 分量, 即 $w$) 提出来消掉, 得到 per-camera 的 2 个 linear equations:
$$[\mathbf{g}_k^x, 1]^T \mathbf{P}_{col,0}^i - u_k^i [\mathbf{g}_k^x, 1]^T \mathbf{P}_{col,2}^i = 0$$
$$[\mathbf{g}_k^x, 1]^T \mathbf{P}_{col,1}^i - v_k^i [\mathbf{g}_k^x, 1]^T \mathbf{P}_{col,2}^i = 0$$

其中 $\mathbf{P}_{col,m}$ 表示 $\mathbf{P}$ 的第 $m$ 列 (index 0-based)。这是因为 homogeneous 形式下 $u = \frac{P_0 \cdot X}{P_2 \cdot X}$, $v = \frac{P_1 \cdot X}{P_2 \cdot X}$, cross-multiply 消掉分母。

两个 camera 给出 4 个 equations, 但 $\mathbf{g}_k^x$ 只有 3 个 unknowns $(x, y, z)$, 所以是 over-determined system:

$$\mathbf{A}^T = \begin{bmatrix} \mathbf{P}_{col,0}^i - u_k^i \mathbf{P}_{col,2}^i \\ \mathbf{P}_{col,1}^i - v_k^i \mathbf{P}_{col,2}^i \\ \mathbf{P}_{col,0}^j - u_k^j \mathbf{P}_{col,2}^j \\ \mathbf{P}_{col,1}^j - v_k^j \mathbf{P}_{col,2}^j \end{bmatrix}, \quad \mathbf{b} = \mathbf{0}$$

求解:
$$\mathbf{g}_k^x = \arg\min_{\mathbf{g}_k^x} \|\mathbf{A} \mathbf{g}_k^x + \mathbf{b}\|^2$$

这是个标准 least-squares, 用 SVD 解 $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, 取 $\mathbf{V}$ 最后一列 (对应最小 singular value) 作为 solution (DLT triangulation 的标准做法, 因为本质上是找 null space)。

**Intuition**: 这相当于在两个 camera 的 viewing rays 之间找最短距离点。理想情况下两条 ray 在 3D 空间相交, 但因 matcher 噪声和 calibration 误差, 它们通常 skew, 找的是两 ray 之间公垂线的中点。EDGS 不要求这个点精确, 它只要"足够好"让后续 optimization refine。

### 2.3 其他 Gaussian 参数初始化

- **Color** $\mathbf{g}_k^c$: 两 matched pixels RGB 值的 average
- **Scale** $\mathbf{g}_k^s$: Gaussian 到两个 camera 的最小距离 (作为初始 spatial extent)
- **Rotation** $\mathbf{R}_k$: identity matrix
- **Opacity** $\alpha_k$: 标准 3DGS 默认值 (0.1 + sigmoid)

Scale 用 camera 距离是个有趣的 heuristic: 离 camera 远的 Gaussian 应该更大 (因为视角覆盖的 angular 范围对应更大的 spatial extent), 这样视觉上 splatting 出来的 size 合理。

---

## 3. 关键 Ablation 数据深度分析

### 3.1 Gaussian 运动距离 — 这个 ablation 最 informative

论文 Section 4.5 测量了两个量:
- **Final displacement**: $\|\mathbf{g}_i^x(0) - \mathbf{g}_i^x(T)\|_2$, 起点到终点直线距离
- **Total path length**: $\sum_{t=0}^{T} \|\mathbf{g}_i^x(t) - \mathbf{g}_i^x(t+1)\|_2$, 整个优化轨迹长度

数据:
- 3DGS coordinate travel distance 比 EDGS 大 **50 倍**
- 3DGS coordinate total path length 比 EDGS 长 **30 倍**
- Color path length 比 EDGS 长约 **2 倍**

**这给我们什么 intuition?**

传统 3DGS 的 Gaussian 大部分时间在"漂移": 它们从 SfM 点 (粗糙位置) 出发, 通过 densification 在中间被 clone / split, 位置和 scale 大幅震荡。Optimization path 长意味着 optimization landscape 有很多局部 minimum 和 plateau, Gaussian 在其间徘徊。

EDGS 因为 init 已经接近 surface, Gaussian 只需小幅 refine, optimization 几乎是直线收敛到附近 minimum。这暗示 photometric loss 的 landscape 在 "正确位置" 附近是 well-conditioned 的, 真正 difficulty 在于 "如何到达正确位置" — 而 EDGS 用 2D prior 直接 teleport 过去。

Color 路径只短 2 倍而 coordinate 短 30 倍, 这个不对称很有意思。原因: EDGS 的 color init 只是 average of 2 views, 而 surface 真实 radiance 在 view-dependent reflection 区域可能差异很大; 但坐标更确定, 因为 triangulation 利用 camera geometry 给出的位置信息是几何确定性的 (up to noise)。

### 3.2 Noise robustness — 这个 ablation 验证了核心 thesis

作者在 init 上加 $\epsilon \sim \mathcal{N}(0, \sigma)$, 分别加到 coordinate 和 color:
- 对 color noise: 极其鲁棒 (σ 增大到 0.15 都几乎不掉点)
- 对 coordinate noise: 较敏感 (σ = 0.05 就开始 degrade)

**Intuition**: 这与上面 motion ablation 完美一致 — coordinate 决定 Gaussian 最终位置 (geometric structure), 一旦偏了 photometric loss 难以拉回; 而 color 是 photometric loss 直接 supervise 的, optimization 容易修正。这告诉我们: **EDGS 的核心价值是 geometric 初始化**, color init 只是 bonus 加速。

### 3.3 Densification 影响的 paradox

Table 2 中:
- 3DGS (COLMAP init, w/ Densification): 27.49 PSNR / 0.215 LPIPS
- 3DGS (Random init, w/ Densification): 22.19 PSNR / 0.313 LPIPS
- EDGS (RoMa init, w/o Densification): 27.80 / 0.175
- EDGS (RoMa init, w/ Densification): 27.84 / 0.173
- DepthFM init (w/ Densification): 27.15 / 0.198
- DepthFM init (w/o Densification): 26.75 / 0.209

注意几件事:
1. Random init + Densification 远不如 COLMAP + Densification — 3DGS 依赖 SfM 先验并非 "free lunch", densification 是 local operation, 无法 global 拉回完全错位的 init
2. EDGS w/o Densification 已经超过 3DGS w/ Densification, 证明 dense init 替代了 densification 的功能
3. EDGS + Densification 只提升 0.04 PSNR, 但 splat 数量几乎翻倍 — diminishing return 极陡
4. DepthFM (monocular depth-based init) 比 EDGS 差, 关键原因是 monocular depth 有 scale ambiguity, 不同 view 之间的 depth scale 不一致, triangulation 反而用 multi-view geometry 给出 metric scale

**这个 ablation 强烈支持论文 thesis**: densification 只是个 hack, 真正缺的是 dense geometric prior; 一旦有了 dense prior, densification 就是冗余甚至有害 (大幅增加 splat 数量、降低 controllability)。

### 3.4 Matcher 选择 — Table 3

| Matcher | PSNR↑ | SSIM↑ | LPIPS↓ | Notes |
|---------|-------|-------|--------|-------|
| RoMa | 27.80 | 0.840 | 0.175 | 主方法 |
| LoFTR | 27.71 | 0.828 | 0.185 | 接近 |
| DKM | 27.69 | 0.829 | 0.190 | 接近 |
| RAFT | 26.98 | 0.802 | 0.218 | 显著差 |

RAFT 差的原因: 它是为 optical flow 设计的, 假设 view 间 motion 小 (视频相邻帧); 而 3DGS 场景 view 间 baseline 大, optical flow 假设破裂。这是 task mismatch 导致性能下降的典型案例。

---

## 4. 主结果 Table 1 深度阅读

挑几个 Mip-NeRF360 上有代表性的对比:

| Method | SSIM↑ | PSNR↑ | LPIPS↓ | Time | #G (M) |
|--------|-------|-------|--------|------|--------|
| Mip-NeRF360 [2] | 0.792 | 27.69 | 0.237 | 48 h | N/A |
| 3DGS (original) | 0.815 | 27.21 | 0.214 | 42 min** | 3.5 |
| 3DGS* (re-trained) | 0.816 | 27.49 | 0.215 | 26 min | 2.8 |
| AbsGS | 0.818 | 27.41 | 0.198 | 20 min | 3.1 |
| Mip-Splatting | 0.842 | 27.97 | 0.176 | 26 min | 4.0 |
| 3DGS-MCMC | 0.838 | 28.15 | 0.176 | 20 min | 3.2 |
| ScaffoldGS | 0.812 | 27.60 | 0.222 | 22 min | 0.6 anchors |
| **Ours + 3DGS 30K** | **0.840** | 27.80 | **0.175** | 29 min | **1.9** |
| **Ours + Taming 3DGS 30K** | 0.839 | **28.06** | **0.174** | **16 min** | 3.2 |
| Taming 3DGS | 0.820 | 27.71 | 0.207 | 14 min | 3.2 |
| MiniSplatting | 0.820 | 27.25 | 0.217 | 12 min | 0.5 |
| EAGLES | 0.820 | 27.20 | 0.232 | 16 min | 1.3 |
| **Ours + 3DGS 5K** | 0.820 | 26.70 | 0.202 | 8 min | 2.9 |
| **Ours + Taming 3DGS 5K** | 0.825 | 26.89 | **0.195** | **6 min** | 2.8 |

**几个关键观察**:

1. **5K steps 的 EDGS 已经达到 30K steps 3DGS* 的 LPIPS (0.195 vs 0.215)** — 论文标题声称"6× fewer optimization steps"是经过谨慎设计的对比

2. **Splat 数量减少 40%** (1.9M vs 3.5M for original 3DGS, 1.9M vs 2.8M vs re-trained 3DGS*) — 这意味着不仅训练快, 推理时内存和速度也更好

3. **PSNR 提升相对小 (27.80 vs 27.49), 但 LPIPS 大幅下降 (0.175 vs 0.215)** — 论文坦承这是 reflective surface 处理弱的副作用: EDGS 用 2-view color average 初始化, 对 view-dependent 反射表面建模不力, 这影响 PSNR (pixel-level metric), 但 SSIM/LPIPS (perceptual / structural metrics) 大幅提升, 说明 high-frequency texture 重建更好

4. **EDGS + Taming 3DGS 是最强组合** (28.06 PSNR, 16 min) — 显示 EDGS 的 init 与其他 acceleration techniques 正交, 可叠加

### 4.1 Per-scene 数据有意思的细节 (Table A1-A5)

看 Mip-NeRF360 per-scene:
- **flowers** scene: EDGS+3DGS 21.57 PSNR, Taming version 21.87 — flowers 是 high-frequency texture 区域, 体现 EDGS 的 dense init 优势
- **kitchen**: 32.31 PSNR (3DGS) → 32.52 (Taming) — 室内场景本来就有较好 baseline
- **garden**: 27.61 → 28.09 — 户外 unbounded 场景提升明显

Time per scene 也透露信息:
- room (33 min for 3DGS, 13 min for Taming) — 室内场景 Taming 加速效果最显著
- counter (40 → 13 min) — 类似

---

## 5. 与我之前理解的一些连接

### 5.1 与 RAIN-GS [29] 的对比

RAIN-GS 证明 random init 也能达到 3DGS 性能, 这乍看与 EDGS thesis 矛盾 (如果 random 都行, 为什么要精心 init?)。但 Table 1 显示 RAIN-GS 是 22.23 PSNR (Mip-NeRF360), 远低于 EDGS 的 27.80。

EDGS 的贡献是:**dense init 不仅够用, 而且 superior**, 因为它给每个 Gaussian 提供 geometric 信号, 让 optimization 不用 waste 时间在 "discovery" 上。RAIN-GS 依赖 densification 把 random Gaussians 拉到正确位置, 仍然是"sculptor"思维。

### 5.2 与 Rad-Splat [53] 对比

Rad-Splat 用 pretrained NeRF 提取点云做 init, 但需要 **9 小时** NeRF 预训练。EDGS 直接用 2D matcher (RoMa) 跳过 NeRF 中间步骤, 时间从 9h 降到几分钟。这反映一个 design philosophy: NeRF 隐式表示对 init 是 overkill, 2D correspondence 已经包含足够 geometric 信息。

### 5.3 与 MVS (Multi-View Stereo) 的关系

EDGS 本质上是把 MVS 的 patch-match / depth triangulation 思想用到 3DGS 初始化。传统 MVS (COLMAP dense, OpenMVS) 也做 per-pixel depth triangulation, 但目标是 mesh; EDGS 把这个 pipeline 直接输出 Gaussian primitives。

这让我想到: 3DGS 社区可能一直忽略了一个事实 — SfM sparse 点云是 MVS pipeline 的中间产物, 之后再做 dense MVS 才是完整 pipeline。3DGS 把 dense MVS 步骤省掉, 用 densification 替代, 但 densification 本质上是个 "blind" 的 MVS, 用 photometric loss gradient 当 signal。EDGS 实际是恢复 dense MVS 步骤, 但直接初始化 Gaussian 而非 depth map。

### 5.4 关于 view-dependent color 的 limitation

论文承认 EDGS 在 reflective surface 弱。这是 EDGS design 的内在限制: 用 2-view RGB average 初始化 color, 在 view-dependent 反射区域 (镜面、玻璃、水) 是错的。3DGS 用 spherical harmonics (SH) 处理 view-dependent color, 但 SH 系数 init 也是从 sparse SfM 点开始通过 densification 学到。

可能的改进方向:
- 用更多 view 的 color cluster 而非 average, 捕捉 view-dependent variation
- 用 SH 直接 init (从 matched pixels 在不同 view 的 color 反推 SH 系数)
- 结合近期 Spec-Gaussian [75] 等 view-dependent 3DGS 工作

---

## 6. Hyperparameter 选择 (Section 4.5)

Figure 7 显示:
- 15000 keypoints per reference frame 是 sweet spot (蓝色曲线)
- 180 reference cameras 是 sweet spot (橙色曲线)

更多 keypoints / cameras 给 diminishing returns。这暗示 init quality 在某点后 saturate, 而 photometric optimization 的 capacity 是 bottleneck。也意味着 EDGS init 的"足够好"门槛不高。

Supplementary Section C 提到 nearest neighbors 数量影响: 增加邻居数量显著增加 init 时间但 minimally 影响 final performance。所以策略是 top-1 nearest neighbor + 更多 reference views, 而非每个 reference 用多邻居。这暗示 overlap redundancy 很高, 多邻居只在 coverage gap 时才增量补充。

---

## 7. Failure modes 和局限性

从 ablation 推断:

1. **Sparse view 场景**: EDGS 假设 dense correspondence 可靠, 极稀疏 view 下 matcher 性能下降, triangulation baseline 不足 (ray angle小, triangulation 数值不稳定)。Supplementary A 提到 16 frames 可工作, 但更稀疏可能退化

2. **Reflective / view-dependent surfaces**: 上面讨论过

3. **Under-covered regions** (Supplementary E): 若 keypoint sampling 没覆盖到某区域, 该区域无法收敛。论文强调 "uniform sampling 比 high-confidence mining 重要" — 这是 dense MVS 的 wisdom

4. **Memory**: Dense init 一次性分配大量 Gaussians (1.6M-3.2M), 相比 3DGS 逐步增长到 3.5M 的内存模式不同。早期训练 memory footprint 更大, 但 final 更少 (1.9M)

---

## 8. 实验设计上的几个聪明选择

1. **包含 init time 在 total training time**: Table 1 footnote 明确说"reported training time for our method includes initialization", 这是诚实做法, 避免"作弊"对比

2. **Re-train 3DGS (3DGS\*)**: 原版 3DGS 在不同硬件上时间不同 (RTX 3090 vs A100), 重新训练保证 apples-to-apples

3. **报告 ScaffoldGS 在 3DGS resolution 设置下的结果**: 原 ScaffoldGS paper 只报告 7 个 Mip-NeRF360 scene, 这里补齐 9 scene 并用相同 resolution, 这是负责任的对比

4. **5K vs 30K steps 的 dual reporting**: 既对比 final quality (30K) 又对比 speed (5K), 完整 Pareto frontier

---

## 9. 我的 takeaways for you, Andrej

我猜你作为教育者/研究者会关注几个点:

1. **Photometric loss landscape 的 geometry**: 这个工作的 motion ablation 间接揭示了 3DGS optimization 的 difficulty 在"找到正确位置"而非"refine"。如果你要给 lecture 讲 3DGS, 这个 ablation 是 motivation 好例子

2. **2D prior 是 cheat code**: EDGS, MVSplat, DepthSplat 等近期工作都在 leverage 2D pretrained models 作为 3D prior, 这与 SfM 点云作为 prior 是 paradigm shift。2D models 见过海量数据, 可以直接做 dense correspondence / depth, 把这信息投影到 3D 几乎 free lunch

3. **Densification 是 over-engineered**: 3DGS 的 densification 是 elegant engineering, 但本质上是 photometric gradient signal 的 hack。EDGS 用 "direct dense init + skip densification" 显示这个 hack 可以 bypass, 这对 future 3DGS-style methods 设计有启发 — 应该思考 "我们想 achieve 什么" 而非 "怎么 make densification better"

4. **Simple idea + careful execution**: 整个 method 在数学上是 DLT triangulation, 几乎是 textbook 内容。价值在 system-level insight + 验证 execution (matcher 选择, robustness study, 与 acceleration methods 组合)。这反映好的 research 不需要 novel math, 需要 novel question (here: "can we skip densification?")

---

## 10. 可能的延伸方向 (open questions)

1. **Uncertainty-aware init**: 现在 EDGS 对所有 matched pixels 一视同仁, 但 confidence $\mathbf{c}^{ij}$ 已有, 可以 weighted triangulation 或 weighted init opacity (低 confidence 区域 init opacity 低, 让 optimization 自由调整)

2. **Dynamic scene**: 4DGS 类工作能否用 EDGS-style init? 时间维 correspondence 更复杂

3. **SH coefficients init**: 现在只 init DC component (color), SH 高阶项从零开始。可以用 multi-view matched colors 反推 SH

4. **Adaptive sampling**: 现在 uniform sampling, 但 high-frequency 区域应该更密。用 image gradient 或 frequency 分析做 importance sampling

5. **更激进的 init**: 既然 init 这么有效, 能否 init rotation / scale 也从 local surface normal 估计? 用 multi-view normal estimation 或 depth gradient 推 surface orientation

---

## References & Web Links

- **3DGS (original)**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
  Paper: Kerbl et al., ACM TOG 2023

- **EDGS project page**: https://compvis.github.io/EDGS/  

- **RoMa matcher (主用)**: https://github.com/Parskatt/RoMa  
  Edstedt et al., CVPR 2024

- **Mip-NeRF360 dataset**: https://jonbarron.info/mipnerf360/

- **Taming-3DGS (组合加速)**: https://github.com/mallick-sashwat/taming-3dgs  
  Mallick et al., 2024

- **3DGS-LM**: https://github.com/uhugl/3DGS-LM  
  Hölllein et al., 2024

- **EAGLES**: https://github.com/Sharath-girish/efficient-gaussian-splatting  
  Girish et al., 2023

- **MiniSplatting**: https://github.com/longxiaofeng/mini-splatting (path 推断)

- **AbsGS**: https://github.com/MonsPla/3DGS-with-Absolute-Precision  

- **Scaffold-GS**: https://city-super.github.io/scaffold-gs/  
  Lu et al., CVPR 2024

- **3DGS-MCMC**: https://maturk.github.io/3DGS-MCMC/  
  Kheradmand et al., NeurIPS 2024

- **DepthFM (depth init baseline)**: https://github.com/CompVis/depth-fm  

- **LoFTR**: https://zju3dv.github.io/LoFTR/  
  Sun et al., CVPR 2021

- **DKM**: https://github.com/Parskatt/DKM  
  Edstedt et al., CVPR 2023

- **RAFT (optical flow baseline)**: https://github.com/princeton-vl/RAFT  
  Teed & Deng, ECCV 2020

- **RAIN-GS**: https://maincold2.github.io/raings/  
  Jung et al., 2024

- **Rad-Splat**: https://research.nvidia.com/labs/toronto-ai/radsplat/  
  Niemeyer et al., 2024

- **MVSplat (sparse-view GS)**: https://github.com/donydchen/mvsplat  
  Chen et al., ECCV 2024

- **DepthSplat**: https://arxiv.org/abs/2410.13862  

- **Tanks&Temples dataset**: https://www.tanksandtemples.org/  

- **Deep Blending dataset**: https://hdr-2018-22.github.io/  

- **gsplat (fast reimplementation)**: https://github.com/nerfstudio-project/gsplat  

- **DLT triangulation theory**: Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, Cambridge University Press, 2003 (Chapter 12 on Triangulation) — 这是 EDGS 公式 (4)-(7) 的 textbook 出处

- **Spec-Gaussian (view-dependent 3DGS)**: https://github.com/ingra14a/Spec-Gaussian  

如果你想要更深入聊某一个部分 (比如 DLT 的 numerical conditioning 在 sparse view 下怎么退化, 或 SH init 的具体推导), 让我知道, 我可以再展开。
