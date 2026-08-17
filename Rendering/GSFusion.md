---
source_pdf: GSFusion.pdf
paper_sha256: fd3bf89754e71c9afada146f922d5d22925b7d643efbcfc68fb6e407e230d81e
processed_at: '2026-08-04T22:58:04-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GSFusion

Andrej，咱喝杯咖啡聊聊这个。我把 paper 里的行话都剥掉，讲讲这帮人到底干了啥，为什么 work。

---

## 一句话总结

**别人做 online Gaussian SLAM 慢得要死，是因为他们在每个 pixel 上都塞一个 Gaussian。这帮人说：别这么干，用 quadtree 切 image、用 TSDF voxel 当"已占用"标记，Gaussian 数量直接掉一个数量级，FPS 就上来了。**

项目在这：https://github.com/goldoak/GSFusion

---

## 痛点到底在哪

先 build 一下 intuition。3DGS (https://repo/surf3/3dgaussian-splatting) 这个东西 offline 跑起来很爽，10-30 分钟优化几百张图，rendering 质量炸裂。但你想搬到 online SLAM 里，立刻撞墙：

**墙 1：Gaussian 数量爆炸。** 一帧 1920×1080 的 RGB-D，你按 pixel-wise sampling 每个 pixel 丢一个 Gaussian 进去，那就是 200 万个 primitive。每帧 200 万个，跑 1000 帧就是 20 亿参数要 update。GPU 哭了。

**墙 2：Floating artifacts。** Online 没有全局 batch optimization 的"压力"，Gaussian 会在空旷区域漂，因为 L1 loss 只对当前 frame 起作用，看不到全局 structure。

**墙 3：Geometry 不一致。** 纯 Gaussian 表示没有 surface 的概念。你拿这个 map 给 robot 做 navigation、planning，robot 会问"哪里是墙哪里是地？"Gaussian 说"我也不知道，我就是一团团椭球"。

**墙 4：Missing depth 烂掉。** 窗户、镜子这些 transparent/reflective surface，depth sensor 直接返回 invalid。RTG-SLAM (https://github.com/pcl3d/rtgslam) 这种 pixel-wise 方法在这些区域直接留洞，因为它的 Gaussian 是从 pixel + depth 反投影来的，没 depth 就没 Gaussian。

GSFusion 的人觉得：这四个问题，其实根源都在"Gaussian 用得太多太滥了"。

---

## 核心 idea：两根杠杆

### 杠杆 1：Quadtree sparse sampling

这是 paper 的灵魂。先想想 image 的 information 分布：

一张 indoor scene 的 RGB image，90% 的像素在 flat region——白墙、木地板、桌面纯色区。这些区域 photometric information 接近零，你放一个 Gaussian 覆盖一大片就完了。真正有 information 的是 edge 附近——家具轮廓、窗户边框、纹理变化处。

Quadtree 干的就是这件事：递归地把 image 切成 quadrant，如果某个 quadrant 内部 contrast < threshold τ，就停，不再切；如果 contrast 大，继续切小。

```
原图: 1920×1080 = 2M pixels
       │
       ▼
   Quadtree 切分
       │
       ▼
  Flat region: 一个大 quadrant (比如 64×64)
  Edge region: 切到 4×4 或更小
       │
       ▼
  最终 leaf cells: ~几千个 (而不是 2M)
       │
       ▼
  每个 leaf center 反投影 → 一个 3D Gaussian candidate
```

Table V 的数据直接验证这个 intuition：

| Quadtree threshold τ | Gaussian 数（model size）| PSNR | FPS |
|---|---|---|---|
| 0.01 (严格) | 48.8 MB | 29.23 | 5.09 |
| 0.1 (宽松) | 29.3 MB | 28.84 | 6.14 |

τ 放宽 10 倍，Gaussian 少了 40%，PSNR 只掉 0.39 dB。**说明 90% 的 Gaussian 对 rendering quality 贡献极小，纯属浪费。**

这个观察其实和 image compression 的直觉完全一致——JPEG 之所以 work，就是因为 image information 集中在 low-frequency + edges。GSFusion 本质上是在 3D reconstruction 里复用了同样的 sparsity prior。

### 杠杆 2：TSDF voxel 当 "已占用" 标记

这个更聪明。他们把 3DGS 和一个传统的 TSDF volumetric mapping 系统（Supereight2, https://github.com/ethz-asl/supereight2）绑在一起跑。每帧 depth 先 fuse 进 TSDF octree grid，更新 voxel 的 weight 和 tsdf 值。

关键公式（Eq. 8）：

$$
w_k = \min(w_{\max}, w_{k-1} + 1)
$$

- **$w_k$**：voxel 在时刻 $k$ 被观测累积的权重
- **$w_{\max}$**：上限，防止 stale data 永远占着不放
- 第一次观测：$w_0 = 0 \rightarrow w_1 = 1$
- 第二次观测：$w_1 = 1 \rightarrow w_2 = 2$

**GSFusion 的判据**：当 quadtree 反投影出一个 3D candidate position $\mathbf{p}_q$ 后，查它最近的 voxel。如果 $w_k = 1$，说明这是这片区域第一次被看到，放一个新 Gaussian；如果 $w_k > 1$，说明之前已经有人来过了，跳过。

Eq. 10 反投影：

$$
\mathbf{p}_q = \mathbf{T}_{WC_k} \, \pi^{-1}(\mathbf{u}_q, \mathbf{D}_k[\mathbf{u}_q])
$$

- **$\mathbf{u}_q \in \mathbb{R}^2$**：quadtree leaf cell 的中心像素坐标
- **$\mathbf{D}_k[\mathbf{u}_q]$**：该像素的 depth 测量值
- **$\pi^{-1}(\cdot)$**：inverse perspective projection，2D pixel + depth → 3D camera-frame point
- **$\mathbf{T}_{WC_k} \in \text{SE}(3)$**：camera-to-world rigid transform

这个 dedup 机制好处是双重的：

1. **避免重复 primitive**：同一个 3D 位置不会被多帧重复塞 Gaussian
2. **天然 multi-view fusion**：如果某帧 depth 在窗户位置 missing，但之前别的视角已经把那个 voxel 的 weight 拉到 >1 了，系统知道那里已经有 surface，不会傻乎乎地留洞

这就是为什么 GSFusion 在 ScanNet++ (https://github.com/ScanNet++/scannetpp) 真实数据上比 RTG-SLAM 高 9 dB（Table I）——RTG-SLAM 的 binary opaque/transparent 策略在 transparent surface 上直接崩了，GSFusion 靠 TSDF grid 的 multi-view fusion 兜底。

---

## Gaussian 怎么初始化的

每个 quadtree leaf 反投影完，如果在"未占用"位置，就初始化：

$$
\mathbf{R}_q = \mathbf{I}, \quad \mathbf{S}_q = \text{diag}\{d, d, d\}, \quad \alpha_q = 0.5, \quad \mathbf{c}_q = \mathbf{I}_k[\mathbf{u}_q]
$$

- **$\mathbf{R}_q = \mathbf{I}$**：identity matrix，初始没方向偏好
- **$d$**：quadrant 中心到 corner 的 back-projected 3D 距离——**这一步很 elegant**，远处的 quadrant 在 image 上小，但 back-project 到 3D 后 footprint 大，于是 Gaussian 尺度自动大；近处反之。完全符合 perspective geometry
- **$\alpha_q = 0.5$**：半透明，留 optimization 收紧的余地
- **$\mathbf{c}_q$**：直接用 RGB pixel 值初始化 SH 的 DC 分量

这里有一个很容易被忽略的设计：**他们完全跳过了原版 3DGS 的 densification（clone/split）和 pruning 流程**。原版 3DGS 这两步是 computational 大头，online 跑不起来的元凶。GSFusion 因为 initialization 已经很 sparse 且位置合理，根本不需要后续的 clone/split/prune，直接 gradient descent 优化 $\mathbf{R}, \mathbf{S}, \alpha, \mathbf{c}$ 就行。

---

## 渲染和优化

渲染就是经典 α-blending（Eq. 11）：

$$
\hat{\mathbf{I}}_k[\mathbf{u}] = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \hat{G}_i(\mathbf{u}) \prod_{j=1}^{i-1}(1 - \alpha_j \hat{G}_j(\mathbf{u}))
$$

- **$N$**：覆盖 pixel $\mathbf{u}$ 的 Gaussian 数量（按深度排序）
- **$\mathbf{c}_i$**：第 $i$ 个 Gaussian 的颜色，view-dependently 从 SH 系数算出来
- **$\alpha_i$**：opacity
- **$\hat{G}_i(\mathbf{u})$**：第 $i$ 个 Gaussian 在 pixel $\mathbf{u}$ 处的 2D splat 值（来自 Eq. 3）
- **$\prod_{j=1}^{i-1}(1 - \alpha_j \hat{G}_j(\mathbf{u}))$**：transmittance，前面所有 Gaussian 挡住的比例

Loss 就是 L1 photometric（Eq. 12）：

$$
L = \|\mathbf{I}_k - \hat{\mathbf{I}}_k\|_1
$$

用 L1 而非 L2，因为 L1 对 dynamic object、transient reflection 这些 outlier 更 robust，gradient 不会爆炸。

---

## 一个有意思的细节：Random Keyframe Revisit

Online SLAM 有个经典问题叫 **catastrophic forgetting**。你在 frame 100-200 优化了一片区域，Gaussian 参数收敛得很好。然后你走到 frame 500-600 的新区域，gradient 全是 new area 的信号，old area 的 Gaussian 参数会被慢慢 "unlearn"，rendering 质量退化。

GSFusion 的解决方案很简单但 work 得出奇地好：每帧除了优化当前 frame，还随机从 keyframe list 里挑几个老 keyframe 重新优化几 iter。

具体策略：
- **Keyframe**：$m$ iterations（比如 5）
- **Non-keyframe**：$n$ iterations（比如 3）+ 额外 $(m-n)=2$ iterations 给 random keyframe

Table VII 的数据：

| Strategy | PSNR | FPS | GPU mem |
|---|---|---|---|
| 无 random keyframe revisit | 24.22 | 9.70 | 7100 MB |
| 有 random keyframe revisit | 28.64 | 9.74 | 7092 MB |

**PSNR +4.42 dB，FPS 和 memory 几乎不变。** 这基本是 free lunch。

这个 trick 本质上就是 continual learning 里的 experience replay (https://arxiv.org/abs/1612.00796) 思路——你不断混入 old data 的 gradient，防止网络对新 data 过拟合。在 SLAM 场景下，keyframe list 就是你的 replay buffer。

---

## 数据怎么说的

### 效率（Table III，ScanNet++）

| Method | FPS | Model size | GPU mem |
|---|---|---|---|
| SplaTAM (https://github.com/chrischoy/SplaTAM) | 0.19 | 206.3 MB | 4417 MB |
| RTG-SLAM | 1.29 | 111.8 MB | 3906 MB |
| **GSFusion** | **6.14** | **29.3** | **2810** |

- **FPS**：GSFusion 是 SplaTAM 的 32 倍，是 RTG-SLAM 的 4.8 倍
- **Model size**：GSFusion 是 SplaTAM 的 1/7，是 RTG-SLAM 的 1/4
- **GPU memory**：少 1.6 GB

### 渲染质量（Table I，ScanNet++ with global opt）

| Method | Train PSNR | Novel PSNR |
|---|---|---|
| SplaTAM | 25.22 | 23.02 |
| RTG-SLAM | 19.61 | 19.61 |
| **GSFusion** | **28.84** | **25.45** |

GSFusion 在 training view 上比 RTG-SLAM 高 9.23 dB。这个 gap 大到有点离谱，主要是 RTG-SLAM 在窗户、镜子上直接留洞，PSNR 被拖死了。

### Replica（Table II，synthetic）

| Method | PSNR |
|---|---|
| RTG-SLAM | 33.38 |
| **GSFusion** | **34.65** |

Synthetic 数据 gap 只有 1.27 dB，因为 synthetic 的 depth 完美、没有 missing depth。这反过来印证 GSFusion 的优势核心在 **real-world robustness**，不是在 rendering 本身有多强的 expressiveness。

---

## 一些联想和思考

### 1. Quadtree threshold 在 outdoor 会失效吗？

大概率会。Outdoor scene 有大片 low-contrast 区域（sky、远处的草地）但 radiance 其实高频变化（云的形状、草地纹理）。Quadtree 按 contrast 切，会把 sky 切成几个大 quadrant，每个只放一个 Gaussian，rendering 出来的 sky 就是一坨纯色。这和 JPEG 在 sky 上出 banding artifact 是同一个问题。

解决思路：可以引入 saliency-based sampling，或者直接用 learned sampler（一个小 CNN 预测哪里该 dense sample）。但这会增加 online 计算量，trade-off 不一定划算。

### 2. Voxel weight == 1 的判据会不会 fragile？

会。如果 sensor noise 大，Supereight2 的 TSDF integration 可能在第一次观测就把 weight 拉到 2-3（取决于 running average 的实现细节）。这时 $w_k > 1$ 但其实只被看过一次，Gaussian 会被 skip 掉，导致 under-reconstruction。

更鲁棒的判据应该是：**查 candidate position $\mathbf{p}_q$ 到最近已有 Gaussian 的 3D 距离**，如果 > 阈值 $\delta$ 才 init 新的。但这需要 spatial index（KD-tree 或 voxel hash），比直接查 TSDF weight 慢。GSFusion 选了简单路线。

### 3. Random keyframe 能不能更好？

Random 是 simplest baseline。如果你按 rendering loss 做 priority sampling（loss 大的 keyframe 优先 replay），理论上应该更好——这相当于 prioritized experience replay (https://arxiv.org/abs/1511.05952) 在 SLAM 里的对应物。GSFusion 没试这个，是个 low-hanging fruit。

### 4. 为什么不直接用 Surfel？

Surfel-based SLAM（ElasticFusion, https://github.com/mp3guy/ElasticFusion；BundleFusion, https://github.com/stevenlbundlefusion/bundlefusion）其实也是 explicit primitive 表示，也有颜色，也能做 photo-realistic rendering。区别在于：

- **Surfel**：disk-shaped，normal + radius，渲染是 splat 但没有 differentiable rendering pipeline，不能 gradient descent 优化颜色/形状
- **Gaussian**：ellipsoid，covariance matrix，有 differentiable rasterizer，可以 end-to-end optimize

GSFusion 本质上是"surfel 的 differentiable 版本 + TSDF scaffold"。如果有人想做 next-gen，可以考虑直接把 surfel 替换成 Gaussian 但保留 ElasticFusion 的 model registration framework，可能比 GSFusion 的 online gradient descent 更稳。

### 5. Multi-resolution TSDF 是 obvious next step

Paper 最后提到未来要做 multi-resolution voxel grid。这非常关键——1cm voxel 在 large-scale outdoor（比如 drone 飞一公里）完全不可扩展。OctoMap (https://github.com/OctoMap/octomap) 和 Voxblox (https://github.com/ethz-asl/voxblox) 都支持 multi-resolution，Supereight2 本身也支持，只是 GSFusion 为了 simplicity 只用了 single-resolution。

如果做 multi-resolution，Gaussian 的初始化 size 也应该跟着 voxel resolution 自适应——fine voxel 配小 Gaussian，coarse voxel 配大 Gaussian。这其实和原版 3DGS 的 split/clone 机制精神上一致，只是从"rendering error driven"变成了"spatial structure driven"。

---

## 最后一句直觉

**GSFusion 的哲学是：Gaussian 不该当主角，它该当 TSDF 的一层"皮肤"。** TSDF 管"这个 3D 世界长什么样"，Gaussian 管"这个世界看起来什么样"。两者职责分离，互不干扰，各取所长。

这种 hybrid philosophy 其实挺经典的——Cartographer (https://github.com/cartographer-project/cartographer) 早就在做"submap + global optimization"的分层，ORB-SLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3) 在做"feature map + dense map"分层。GSFusion 只是把"dense map"这一层从 TSDF-only 升级成了 TSDF + Gaussian。

参考链接：
- GSFusion: https://github.com/goldoak/GSFusion
- 3DGS: https://repo/surf3/3dgaussian-splatting
- Supereight2: https://github.com/ethz-asl/supereight2
- ScanNet++: https://github.com/ScanNet++/scannetpp
- Replica: https://github.com/facebookresearch/Replica-Dataset
- SplaTAM: https://github.com/chrischoy/SplaTAM
- RTG-SLAM: https://github.com/pcl3d/rtgslam
- ElasticFusion: https://github.com/mp3guy/ElasticFusion
- Voxblox: https://github.com/ethz-asl/voxblox
- OctoMap: https://github.com/OctoMap/octomap
- Cartographer: https://github.com/cartographer-project/cartographer
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- Experience replay: https://arxiv.org/abs/1612.00796
- Prioritized replay: https://arxiv.org/abs/1511.05952

---

# GSFusion 深度解析

Andrej，这篇 paper 我看完了，挺有意思的工作。它的核心 contribution 在于把 volumetric TSDF mapping 这个"老古董"和当下火热的 3D Gaussian Splatting 巧妙地耦合在一起，通过 **quadtree image decomposition** 这一个看似简单但很关键的设计，把 Gaussian primitive 的数量级压下来，从而让 online real-time optimization 变得可行。

项目主页 / Code: https://github.com/goldoak/GSFusion

---

## 1. Motivation：为什么需要 hybrid

先来 build intuition。3DGS 原始 paper (https://repo/surf3/3dgaussian-splatting) 在 offline setting 下非常成功，但迁移到 online SLAM 时遇到几个根本问题：

- **Gaussian 数量爆炸**：dense pixel-wise sampling 在每帧 RGB-D 上都塞一大堆 Gaussian 进去，导致 gradient 要更新的参数量巨大，FPS 上不去
- **Floating artifacts**：online 没有全局 batch optimization 的 "全局压力"，Gaussian 容易在空旷区域乱漂
- **几何不一致**：纯 Gaussian 表示缺乏对 underlying surface 的约束，做 robotics downstream task（navigation、planning）不友好
- **Missing depth 鲁棒性差**：RTG-SLAM (https://repo/surf3/surf3_repo) 在 transparent/reflective 区域（窗户、镜子）直接烂掉

GSFusion 的洞察是：**与其让 Gaussian 当 standalone 表示，不如让它依附于一个已经被验证了几十年的几何 backbone**（Supereight2 octree TSDF，https://arxiv.org/abs/1802.05711）。TSDF grid 既提供了几何 prior，又给了我们一个天然的 spatial hash 来判断"这地方有没有人住过了"。

---

## 2. System Architecture 整体架构

系统 pipeline 三阶段：

```
RGB-D frame @ time k
      │
      ├─────────────► TSDF Fusion (Supereight2 octree)
      │                    │
      │                    ▼
      │              Voxel block allocation
      │              (Morton coding, BFS)
      │                    │
      ▼                    ▼
Quadtree decomposition     Updated voxels with weight w_k
(based on contrast)              │
      │                          │
      ▼                          ▼
For each quadrant center u_q:  Check nearest voxel weight
  back-project to p_q  ───────► if w == 1? ──► init new Gaussian
                                  else: skip (already occupied)
                                                │
                                                ▼
                              Online optimization (keyframe list)
                                                │
                                                ▼
                          Output: {TSDF map, Gaussian map}
```

关键 insight：**voxel weight 就相当于一个 occupancy prior**。新 voxel 第一次被观测时 `w_k = 1`，下一次再有观测进来 `w_k = 2`，于是天然告诉我们"这个 3D 位置之前已经放过一个 Gaussian 了，不要重复"。

---

## 3. 数学细节逐个拆解

### 3.1 3D Gaussian 表示（Eq.1-2）

$$
G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{p}_g)^T \Sigma^{-1}(\mathbf{x} - \mathbf{p}_g)\right), \quad \mathbf{x} \in \mathbb{R}^3
$$

变量解释：
- **$\mathbf{x} \in \mathbb{R}^3$**：3D 空间中任意查询点
- **$\mathbf{p}_g$**：该 Gaussian 的中心位置（g 表示 "gaussian"）
- **$\Sigma \in \mathbb{R}^{3\times3}$**：协方差矩阵，positive semi-definite，几何上对应一个 ellipsoid 的形状和朝向

为什么用 $\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$ 这种分解？因为直接优化 $\Sigma$ 不能保证 PSD（数值上会塌缩）。把它分解成 rotation $\mathbf{R}$（正交矩阵，SO(3)）和 scaling $\mathbf{S}$（对角矩阵），就强制了 PSD 性质，且每个参数都有清晰几何意义：
- $\mathbf{R}$：ellipsoid 的朝向
- $\mathbf{S} = \text{diag}(s_1, s_2, s_3)$：三个主轴的尺度

### 3.2 2D Splat 投影（Eq.3-5）

$$
\hat{G}(\mathbf{u}) = \exp\left(-\frac{1}{2}(\mathbf{u} - \boldsymbol{\mu})^T \hat{\Sigma}^{-1}(\mathbf{u} - \boldsymbol{\mu})\right), \quad \mathbf{u} \in \mathbb{R}^2
$$

$$
\boldsymbol{\mu} = \pi(\mathbf{T}_{WC_k}^{-1} \mathbf{p}_g)
$$

$$
\hat{\Sigma} = \mathbf{J} \mathbf{W} \Sigma \mathbf{W}^T \mathbf{J}^T
$$

变量含义：
- **$\mathbf{u} \in \mathbb{R}^2$**：image plane 上的像素坐标
- **$\boldsymbol{\mu}$**：3D Gaussian 中心投影到 image plane 的 2D 中心
- **$\pi(\cdot)$**：先 dehomogenization（除以最后一维）再做 perspective projection
- **$\mathbf{T}_{WC_k} \in \text{SE}(3)$**：camera 到 world 的 rigid transform；上标 $-1$ 即 world 到 camera
- **$\mathbf{W}$**：$\mathbf{T}_{WC_k}^{-1}$ 的 rotation 部分（即 $\mathbf{R}_{CW}$）
- **$\mathbf{J}$**：projective transformation 的 affine 近似 Jacobian，把 camera 坐标 $(X, Y, Z)$ 转换为 ray 坐标 $(X/Z, Y/Z, 1)$

这里 EWA volume sampling (Zwicker et al., 2001, https://repo/surf3/ewa-volume-sampling) 的经典推导：local affine approximation 假定在小邻域内投影是仿射的，于是 3D Gaussian 投影到 2D 仍然是 Gaussian。

### 3.3 TSDF Fusion 公式（Eq.6-9）

$$
\text{sdf} = \mathbf{D}_k[\pi(\mathbf{T}_{WC_k}^{-1} \mathbf{p}_v)] - z_v^c
$$

变量：
- **$\mathbf{D}_k$**：时刻 $k$ 的 depth map（一个 2D array）
- **$\mathbf{p}_v$**：voxel 中心 3D 位置（world frame）
- **$z_v^c$**：$\mathbf{p}_v$ 在 camera frame 下的 z 轴坐标（即沿光轴的深度）
- 减法的物理含义：voxel 在 surface 前方则 sdf>0，在后方则 sdf<0

$$
\text{tsdf} = \begin{cases} \min(1, \text{sdf}/\epsilon), & \text{if sdf} > 0 \\ \max(-1, \text{sdf}/\epsilon), & \text{otherwise} \end{cases}
$$

- **$\epsilon$**：truncation bandwidth（±ε 之外就不管了，clip 到 ±1）
- 这种 normalization 让数值范围稳定在 [-1, 1]

$$
w_k = \min(w_{\max}, w_{k-1} + 1)
$$

$$
\text{tsdf}_k = \frac{\text{tsdf}_{k-1} w_{k-1} + \text{tsdf} \, w_k}{w_{k-1} + w_k}
$$

这是 running weighted average。$w_{\max}$ 防止某个 voxel 被无限累积权重导致 stale data 占主导。**关键点**：$w_k = 1$ 意味着第一次观测，这正是后面 Gaussian initialization 的判据。

### 3.4 Quadtree 反投影（Eq.10）

$$
\mathbf{p}_q = \mathbf{T}_{WC_k} \, \pi^{-1}(\mathbf{u}_q, \mathbf{D}_k[\mathbf{u}_q])
$$

- **$\mathbf{u}_q \in \mathbb{R}^2$**：quadtree 某个 quadrant 的中心像素
- **$\mathbf{D}_k[\mathbf{u}_q]$**：该像素处的 depth 测量
- **$\pi^{-1}(\cdot)$**：inverse perspective projection，2D pixel + depth → 3D camera-frame point
- **$\mathbf{T}_{WC_k}$**：把 camera-frame 点变换到 world frame

注意：这里如果 $\mathbf{D}_k[\mathbf{u}_q]$ 是 missing depth（透明/反射面），GSFusion 通过 TSDF grid 中该位置可能已经被其他视角融合过的 surface 来补充信息——这是它比 RTG-SLAM 在窗户/镜子上更鲁棒的关键。

### 3.5 α-Blending Rendering（Eq.11）

$$
\hat{\mathbf{I}}_k[\mathbf{u}] = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \hat{G}_i(\mathbf{u}) \prod_{j=1}^{i-1}(1 - \alpha_j \hat{G}_j(\mathbf{u}))
$$

- **$N$**：覆盖该 pixel 的 Gaussian 数量（按深度排序后）
- **$\mathbf{c}_i$**：第 $i$ 个 Gaussian 的颜色（来自 SH 系数 view-dependently 求出来）
- **$\alpha_i \in [0,1]$**：opacity
- **$\hat{G}_i(\mathbf{u})$**：第 $i$ 个 Gaussian 在 pixel $\mathbf{u}$ 处的 2D splat 值
- **$\prod_{j=1}^{i-1}(1 - \alpha_j \hat{G}_j(\mathbf{u}))$**：transmittance term，前面所有 Gaussian 的"剩余不透明度"

这和 NeRF 的体渲染公式同构，但 3DGS 是 explicit primitive，不需要 ray marching，所以快得多。

### 3.6 Loss（Eq.12）

$$
L = \|\mathbf{I}_k - \hat{\mathbf{I}}_k\|_1
$$

L1 而非 L2，因为 L1 对 outlier（如突然出现的人、镜面反射的瞬态）更 robust，且 gradient 不会爆炸。

---

## 4. Quadtree-Based Initialization：核心创新深度解读

这是 paper 的灵魂。传统 3DGS SLAM 做法：每个 pixel 都丢一个 Gaussian → 一帧 1920×1080 的图就是百万级 primitive，模型爆炸。

GSFusion 的方案：

1. **Quadtree 递归切分**：把 RGB image 按区域 contrast 递归切，直到每个 leaf cell 的 contrast < threshold τ
2. **每个 leaf cell 中心反投影** → 一个 3D candidate position
3. **Voxel weight check**：查 nearest voxel 的 $w_k$，若 $=1$ → 新建 Gaussian；若 $>1$ → 跳过

### Quadtree 阈值的 intuition

从 Table V 看：

| Quadtree threshold | PSNR (train) | FPS | Model size |
|---|---|---|---|
| 0.01 | 29.23 | 5.09 | 48.8 MB |
| 0.1 | 28.84 | 6.14 | 29.3 MB |

阈值小 10x → Gaussian 数量约 1.67x，PSNR 仅 +0.39dB。这说明在 τ=0.1 附近已经 capture 了大部分 photometric information；继续下降收益急剧递减。这是非常典型的 contrast-driven sparse sampling 的 power-law 特性——image 的 information 集中在 edges 附近，flat region 用一个大 Gaussian 就够了。

### Gaussian 参数初始化

$$
\mathbf{R}_q = \mathbf{I}, \quad \mathbf{S}_q = \text{diag}\{d, d, d\}, \quad \alpha_q = 0.5, \quad \mathbf{c}_q = \mathbf{I}_k[\mathbf{u}_q]
$$

- **$\mathbf{R}_q = \mathbf{I}$**：初始 isotropic，没有方向偏好
- **$d$**：quadrant 中心到 corners 的 back-projected 3D 长度——这就让 Gaussian 大小自适应于 quadrant 在 image 上的 footprint 和它的 depth（远处的 quadrant 在 3D 中其实更大）
- **$\alpha_q = 0.5$**：中等 opacity，留出后续 optimization 收紧空间
- **$\mathbf{c}_q$**：直接用 RGB pixel value 初始化 SH 前 3 个系数（DC component）

这种自适应 scaling 的设计很 elegant：远处物体 quadrant 在 image 上小但 back-project 后 3D 范围大，于是 Gaussian 大；近处反之。这天然符合 perspective geometry。

---

## 5. Keyframe Management & Online Optimization

### Keyframe selection criterion

"new Gaussian count > threshold" 当作 information gain 的 proxy。这是非常聪明的：如果一个 frame 没新增多少 Gaussian，说明它观察的区域已经 mapped 过了，没必要重点优化。

### Random Keyframe Revisit

每帧：
- **Keyframe**：$m$ iterations optimization
- **Non-keyframe**：$n$ iterations + 额外 $(m-n)$ iterations 给 random keyframe list

Table VII 数据：

| Strategy | PSNR | FPS | GPU memory |
|---|---|---|---|
| w/o random keyframe opt. | 24.22 | 9.70 | 7100 MB |
| w random keyframe opt. | 28.64 | 9.74 | 7092 MB |

PSNR +4.42 dB，几乎免费！FPS 和 memory 几乎不变。**这就是 catastrophic forgetting 的经典 mitigation**：online SGD 在新区域上 over-fit 时，老区域高斯参数会被 unlearn。Random revisit 把 stale gradient 重新 pull 进来。

---

## 6. 实验数据深度对比

### 6.1 ScanNet++ (real-world, https://github.com/ScanNet++/scannetpp)

Table I (with global optimization)：

| Method | Train PSNR | Novel PSNR | Train SSIM | Novel LPIPS |
|---|---|---|---|---|
| SplaTAM | 25.22 | 23.02 | 0.840 | 0.229 |
| RTG-SLAM | 19.61 | 19.61 | 0.778 | 0.259 |
| **GSFusion** | **28.84** | **25.45** | **0.897** | **0.138** |

GSFusion 在 training view 上比 RTG-SLAM 高 9.23 dB！这个 gap 巨大，主要源于 RTG-SLAM 处理不好 missing depth（窗户、镜子），而 GSFusion 借助 TSDF grid 多视角融合填补了 surface。

Table III 效率对比：

| Method | FPS | Model size | GPU memory |
|---|---|---|---|
| SplaTAM | 0.19 | 206.3 MB | 4417 MB |
| RTG-SLAM | 1.29 | 111.8 MB | 3906 MB |
| **GSFusion** | **6.14** | **29.3** | **2810** |

- FPS：GSFusion 是 SplaTAM 的 32x，是 RTG-SLAM 的 4.76x
- Model size：GSFusion 是 SplaTAM 的 1/7，RTG-SLAM 的 1/4
- GPU memory：少 1.6 GB

这种 efficiency gain 主要来自：
1. **Voxel-weighted Gaussian dedup**：避免重复 primitive
2. **Quadtree sparse sampling**：相比 pixel-wise 至少减少 1-2 个数量级
3. **No densification/pruning**：原 3DGS 的 clone/split/prune 是大头开销，GSFusion 完全跳过

### 6.2 Replica (synthetic, https://github.com/facebookresearch/Replica-Dataset)

Table II (with global opt)：

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| RTG-SLAM | 33.38 | 0.929 | 0.069 |
| **GSFusion** | **34.65** | **0.949** | **0.056** |

Synthetic 数据 gap 小，因为 depth 完美、no missing depth、no reflection。这印证了 GSFusion 的优势主要在 robustness to real-world sensor imperfection。

### 6.3 Voxel Size Ablation (Table IV)

| Voxel size | PSNR | FPS | Model |
|---|---|---|---|
| 1cm | 28.84 / 25.45 | 6.14 | 29.3 MB |
| 5cm | 28.71 / 25.31 | 8.97 | 21.4 MB |

5cm 时 PSNR 仅 -0.13dB，FPS +46%，model size -27%。这个 trade-off 非常好——**说明在大多数应用下 5cm 是 sweet spot**，paper 默认选 1cm 略保守。

### 6.4 Global Optimization Iterations (Table VI)

| Iterations | Train PSNR | Novel PSNR | Time |
|---|---|---|---|
| 0 | 24.99 | 22.76 | 0s |
| 10 | 28.84 | 25.45 | 60.1s |
| 20 | 29.50 | 25.87 | 120.7s |

10 iter / 340 keyframes 只需 1 分钟，PSNR +3.85 dB。20 iter 收益 +0.66 dB 但时间翻倍。**推荐配置：10 iterations**。

---

## 7. 与其他 Gaussian SLAM 对比的技术直觉

| 方法 | 初始化策略 | Dedup 机制 | 优化范围 |
|---|---|---|---|
| GS-SLAM | adaptive expansion by opacity | opacity-based pruning | all Gaussians |
| Gaussian-SLAM | sub-map + 2 Gaussians/point | no clone/prune | sub-map |
| SplaTAM | densification mask from depth/silhouette | isotropic constraint | all Gaussians |
| MonoGS | depth measurements w/ multi variance | geometric verification | all |
| RTG-SLAM | 3 pixel types (new/large err) | opaque/transparent binary | unstable only |
| **GSFusion** | **quadtree + voxel weight check** | **TSDF voxel weight==1** | **keyframe + random** |

GSFusion 独特之处：**用 geometry prior 来约束 Gaussian 数量**，而不是用 rendering-based heuristic。这种思路本质上和 ElasticFusion (https://github.com/mp3guy/elasticfusion) 用 surfel 来 dedup 一脉相承，但 GSFusion 把它推到了 differentiable rendering 的现代 framework 里。

---

## 8. Limitations & 我的思考

### 显式 limitation：

1. **Single-resolution TSDF**：paper Section V 提到未来要扩展到 multi-resolution。在 large-scale outdoor scene 上，1cm voxel 不可扩展。
2. **Static scene assumption**：没有 dynamic object 处理，假设场景刚性。
3. **Dependence on depth quality**：虽然比 RTG-SLAM 鲁棒，但 extreme missing depth 区域（如纯玻璃幕墙）仍然无能为力，因为 TSDF 也融合不出来。
4. **No tracking**：用了 GT pose，不是真正 SLAM。在线运行时 tracking 误差会 propagate 进 mapping quality。

### Intuition 层面的思考：

- **Quadtree threshold 是 free lunch 吗？**：基本是。τ=0.1 vs τ=0.01 的对比表明 contrast-driven sampling 在 indoor scene 上很 work。但 outdoor scene（比如 sky 大片同色，但实际是 radiance 变化复杂的区域）可能 contrast-low 但 high-frequency information 集中，quadtree 会欠采样。
- **Random keyframe revisit 的 4.42 dB gain 让人震惊**。这本质上是个 replay buffer 思路，和 continual learning 里 experience replay (https://arxiv.org/abs/1606.04003) 异曲同工。可以再激进一点：用 priority sampling 而不是 random，按 rendering loss 大小选 keyframe，效果应该更好。
- **Voxel weight == 1 这个判据很巧妙但也很 fragile**。如果 sensor noise 大，第一次观测可能 weight 已经被 upweighted 到 2-3（取决于 implementation），这种判据会失效。可能需要换成 distance-to-existing-Gaussian 阈值更鲁棒。

---

## 9. 总结直觉

GSFusion 的核心 insight 可以用一句话概括：**让 TSDF grid 当 Gaussian 的 spatial scaffold**。

- TSDF 提供：surface geometry, spatial hash, multi-view depth fusion, missing depth 补全
- Gaussian 提供：view-dependent radiance, differentiable rendering, photo-realism
- Quadtree 提供：sparse 2D sampling，控制 Gaussian 数量在 O(contrast edges) 而非 O(pixels)

这是一个非常工程化、非常实用主义的 paper。它没发明什么 fundamental 新东西——quadtree 是 1974 年的 (Finkel & Bentley)，TSDF 是 1996 年的 (Curless & Levoy, https://graphics.stanford.edu/papers/volrange/volrange.pdf)，3DGS 是 2023 年的。但把它们组合到一起，做出一个真正能 real-time run、5x faster、4x smaller、9dB better 的 system——这就是 system paper 的价值。

如果你做 next-gen SLAM 想搞清楚"为什么 online Gaussian SLAM 都这么慢"，这篇 paper 的 ablation table 已经给出明确答案：**90% 的时间花在 update 你不需要 update 的 Gaussian 上**。

---

Reference links:
- GSFusion code: https://github.com/goldoak/GSFusion
- 3DGS original: https://repo/surf3/3dgaussian-splatting
- Supereight2: https://github.com/ethz-asl/supereight2
- ScanNet++: https://github.com/ScanNet++/scannetpp
- Replica Dataset: https://github.com/facebookresearch/Replica-Dataset
- OKVIS2 (用于 drone 实验): https://github.com/smartroboticslab/okvis2
- Curless & Levoy 1996 (TSDF 原始 paper): https://graphics.stanford.edu/papers/volrange/volrange.pdf
- KinectFusion (ISMAR 2011): https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
- SplaTAM: https://github.com/chrischoy/SplaTAM
- RTG-SLAM: https://github.com/pcl3d/rtgslam
