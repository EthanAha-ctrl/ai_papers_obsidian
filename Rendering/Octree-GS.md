---
source_pdf: Octree-GS.pdf
paper_sha256: 6d07bd7b289032072f28bb08df40507c076da362b7f7a007e68e322062f7e1c3
processed_at: '2026-08-05T22:58:09-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Octree-GS 人话版

## 一句话讲清楚

3D Gaussian Splatting 在大场景里会爆——远看也把所有细节 Gaussian 都渲染一遍，慢得要死。Octree-GS 给 Gaussian 加了个"远看用粗的、近看用细的"的开关，类似游戏里 LOD (Level of Detail) 的老 trick，但用 differentiable 的方式训练出来。

---

## 1. 这事为什么值得做

想象你在 GTA 里开车。远处的摩天大楼就是个方块，根本看不清窗户；等你开近了，窗户一层层"长"出来，栏杆、门把手依次出现。这就是 LOD——**让物体的表示复杂度跟你和它的距离挂钩**。这个 idea 在游戏引擎里用了 30 年了（Progressive Meshes, Hoppe 1996）。

3D Gaussian Splatting (3D-GS) 反过来——它不管你站哪儿，都把那几百万个 Gaussian 全部 alpha-blend 一遍。你在 500 米外俯瞰城市，那栋楼上一个窗户里有 200 个小 Gaussian 拟合玻璃反光，全都要算一遍，每个投影到屏幕上才 0.01 pixel。结果就是远看时 FPS 从 100+ 掉到 10。

NeRF 系（Mip-NeRF、Zip-NeRF、BungeeNeRF）早解决了这事——用 cone casting + pyramid structure，远的 pixel 自动用粗的 sampling。但 GS 没有 sampling 这个步骤，只有 primitive blending，所以 LOD 没法照搬。

Octree-GS 的核心问题: **怎么在 GS 这个"选 primitives 然后 blend"的框架里塞进 LOD?**

参考:
- 3D-GS 原版: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Mip-NeRF (NeRF 系 LOD 范本): https://jonbarron.info/mipnerf/
- Progressive Meshes (LOD 老祖宗): https://en.wikipedia.org/wiki/Progressive_meshes

---

## 2. 核心 idea: 把 anchor grid 换成 octree

先回忆 Scaffold-GS 干了什么。3D-GS 直接优化几百万个 free-floating Gaussian，Scaffold-GS 觉得太乱了，引入了 **anchor**——在空间里铺一个均匀网格，每个网格点 (anchor) 负责它附近一片区域，emit 出 k 个 neural Gaussian。这就像把自由散漫的 Gaussian 雇佣兵收编成正规军，每个 anchor 是个班长管 10 个兵。

Scaffold-GS: https://city-super.github.io/scaffold-gs/

Octree-GS 的核心 move: **把 Scaffold-GS 的均匀网格换成 octree (多分辨率网格)**。

最粗那一层 (LOD 0) voxel 是 $\delta$ 米见方，每深一层 voxel 减半。LOD 0 的 anchor 管"这一大片区域大概长什么样"，LOD 3 的 anchor 管"这个窗框具体长什么样"。

这就是 3D-GS 时代的 LOD——同一个区域，远看时只用 LOD 0 那一个 anchor 表达（一个粗糙 blob），近看时 LOD 0 + 1 + 2 + 3 全部累加（粗 blob + 细节 = 高清）。

### 关键公式: 怎么决定一个 anchor 该不该渲染

对于 viewpoint $i$ 看 anchor $j$:

$$\hat{L}_{ij} = \lfloor \Phi(\log_2(d_{max} / (d_{ij} \cdot s))) + \Delta L_j \rfloor$$

变量人话版:
- $d_{ij}$: 相机 $i$ 到 anchor $j$ 的距离 (米)
- $s$: focal scale，处理不同焦距的相机。同样 10m 距离，长焦镜头 footprint 小，等效"看起来更远"
- $d_{max}$: 场景里最远的 observation distance (用来归一化)
- $\Phi(\cdot)$: clamp 到 $[0, K-1]$ 的范围
- $\Delta L_j$: anchor $j$ 自己的 learnable LOD bias——这玩意儿是论文的精华之一，下面单独讲
- $\hat{L}_{ij}$: 该 viewpoint 对该 anchor 需要的 LOD level (整数)

**直觉**: 距离越大，$\log_2(d_{max}/d)$ 越小 (接近 0)，只需要 LOD 0；距离越小，这个值越大，需要更细的 LOD。这就是 mip-map 选 texture level 的逻辑，只不过把 pixel footprint 换成 observation distance。

### Cumulative LOD (累加式)

这是 Octree-GS 区别于其他 LOD 方法的核心设计。**渲染 LOD $K$ = 渲染 LOD 0 到 $K$ 所有 anchor 的并集**。

为什么这样设计? 如果是 hard switch (只渲染某一层)，切换时会有 popping——一个窗户突然从"几个色块"变成"清晰的窗框"。Cumulative 保证 coarse 永远在，fine 是在 coarse 基础上"叠加细节"，过渡平滑。

再加 opacity blending: 边界处 (fractional LOD = 1.3 之类) 还把 LOD $\hat{L}+1$ 那层的 opacity 乘以 0.3，做 piecewise linear 插值，完全抹平 popping。

### $\Delta L_j$: learnable LOD bias 的妙处

光靠距离算 LOD 有个问题: 一个 building 的边缘 (sharp edge) 在中等距离就开始模糊，因为距离阈值太死板。$\Delta L_j$ 给每个 anchor 一个 learnable residual——"如果你这个 anchor 经常欠拟合 (gradient 高)，就让自己在更近的距离才被激活，逼模型用更细的 LOD 表示你"。

具体更新规则: 训练时如果 anchor 的 average gradient $\nabla_v > \tau_g^L \cdot 0.25$，就 $\Delta L \mathrel{+}= 0.01$。简单粗暴但有效——Figure 13 的 ablation 显示这个 0.01 的累加能让建筑白线变连续。

---

## 3. 训练怎么搞: Grow, Prune, Progressive

光有结构不够，得让 anchor 在训练中"流"到正确的 LOD 层。

### Next-level Grow (生长到更细层)

标准 Scaffold-GS 的 densification: 哪个 Gaussian 的 gradient 大，就在它附近 grow 新 anchor。Octree-GS 要问: 这个新 anchor 该在 LOD $L$ 还是 $L+1$?

答案是: 设个递增阈值，越细的 LOD 越难进。

$$\tau_g^L = \tau_g \cdot 2^{\beta L}$$

- $\tau_g$: base threshold (0.0002)
- $\beta$: 0.2
- $\tau_g^L$: 在 LOD $L$ 层 grow 新 anchor 需要的 gradient 阈值

规则: gradient 超过 $\tau_g^{L+1}$ 才 promote 到 LOD $L+1$；只超过 $\tau_g^L$ 就停在 LOD $L$。

**直觉**: 升一层 voxel volume 缩 8 倍，需要 8 倍以上的"信号"才值得用细 LOD。这防止训练初期一窝蜂涌到 fine LOD、coarse LOD 没人管的烂摊子。

### View-frequency Prune (按"被看见的频率"剪枝)

标准剪枝看 opacity。但作者发现一个 bug: fine LOD 的 anchor 只在近处 view 才被选中，而训练集里近处 view 数量少，导致这些 anchor 长期不更新、学坏、变 floater (飘在空中的脏 Gaussian)。

定义 view-frequency = anchor 在训练 view 中被选中的概率。规则: view-frequency < $\tau_v$ 就删。

$\tau_v$ 得按 dataset 调:
- 密集拍摄 (Mip-NeRF360): $\tau_v = 0.7$ (严)
- 稀疏拍摄 (MatrixCity): $\tau_v = 0.01$ (松，不然删太多)
- 多尺度 (BungeeNeRF): $\tau_v = 0.2$

Ablation 数据很硬: 关掉 view-freq 后，storage 从 139.6M 涨到 244.4M，PSNR 还从 28.05 掉到 27.74。脏 anchor 又占地方又拖质量。

### Progressive Training (从粗到细训练)

如果一开始所有 LOD 都开训，会出现 LOD entanglement——所有 LOD 的 anchor 都试图表达整个场景，互相抢活，coarse LOD 在远距离 view 时烂得一塌糊涂 (Figure 3 第二行)。

解决: 
- 从 LOD $\lfloor K/2 \rfloor$ 开始 (不是 LOD 0，太粗会 underfitting 太久)
- 每 $N_i$ iter 激活下一层
- 越粗训越久: $N_{i-1} = 1.5 \cdot N_i$

就像画画先打底稿再画细节，别一上来就抠眼睛。

---

## 4. 实验里几个最 punchy 的数字

### 4.1 渲染速度 (Table V, MatrixCity Block_All)

| Method | $T_1$ FPS | $T_2$ FPS | $T_3$ FPS |
|---|---|---|---|
| 3D-GS | 13.81 | 11.70 | 13.50 |
| Scaffold-GS | 6.69 | 7.37 | 8.04 |
| Hierarchical-GS ($\tau_1$ finest) | 16.14 | 13.26 | 14.79 |
| Hierarchical-GS ($\tau_3$ coarsest) | 24.33 | 19.59 | 18.94 |
| **Our-3D-GS** | **57.08** | **56.85** | **56.07** |

注意两点:
1. 比 Scaffold-GS 快 ~8 倍
2. 三个 trajectory FPS 几乎不变 (~56)，而 baselines 在不同 trajectory 上抖来抖去。这就是论文标题 "Consistent Real-time Rendering" 的实证——无论怎么飞都稳。

### 4.2 质量 + 存储 (Table II, Block_All)

| Method | PSNR | #GS(k) | Mem |
|---|---|---|---|
| Scaffold-GS | 26.30 | 690 | 2272.2M |
| CityGaussian | 26.26 | 235 | 4316.6M |
| Hierarchical-GS | 26.00 | 492 | 4874.2M |
| **Our-Scaffold-GS** | **27.31** | **344** | **1648.6M** |

PSNR 高 1 dB，storage 比 Hierarchical-GS 省 66%。质量好还省地方，因为 LOD 把"远距离也需要的粗表示"和"近距离才需要的细表示"分开了，没冗余。

### 4.3 BungeeNeRF 的 #Gaussian 跨 scale 增长 (Table IV)

BungeeNeRF 数据集有 4 个 scale (从地面到卫星视角)。

| Method | scale-1 #GS | scale-4 #GS | 增长倍数 |
|---|---|---|---|
| 3D-GS | 522k | 5821k | **11.2×** |
| Scaffold-GS | 303k | 3876k | **12.8×** |
| **Our-Scaffold-GS** | 486k | 2167k | **4.5×** |

baselines 从地面看到卫星，#Gaussian 暴涨 11-13 倍；Octree-GS 只涨 4.5 倍——因为远看时 coarse anchor 一个顶 fine anchor 一片。

### 4.4 训练时间 (Mip-NeRF360, 40k iter)

- 3D-GS: 34 min
- Scaffold-GS: 29 min
- Hierarchical-GS: 69 min (3 stage, 第一 stage 就 38 min)
- **Our-Scaffold-GS: 23 min** (single stage)

Hierarchical-GS 慢是因为它要 train → build hierarchy → fine-tune 三步。Octree-GS 用 progressive training 把这三步压成一步，反而比 Scaffold-GS 还快——因为 progressive 早期 #primitives 小。

---

## 5. 跟 concurrent works 比，赢在哪

| 维度 | Octree-GS | Hierarchical-GS | CityGaussian |
|---|---|---|---|
| LOD 怎么定 | explicit octree + learnable bias | tree hierarchy (post-hoc) | distance interval (手动阈值) |
| 训练 | end-to-end single stage | 3 stages | multi-stage |
| LOD 切换 | opacity blending (平滑) | hard switch | fusion (有 stroboscopic) |
| Storage | cumulative (coarse 共享) | independent per LOD | independent |
| 训练时间 | 23 min | 69 min | ~50 min |

关键差异:
1. **Hierarchical-GS** 是先训完 base，再 post-hoc 建 hierarchy，再 fine-tune。Octree-GS 是训练时就让 anchor 自然分配到 LOD——progressive training 强制 coarse-to-fine，自然形成 hierarchy。
2. **CityGaussian** 用 distance interval 切换 LOD，切点处有闪烁。Octree-GS 用 cumulative + opacity blending，无闪烁。
3. **存储**: 其他方法每个 LOD 独立表示整个场景，Octree-GS 的 coarse LOD 是 fine LOD 的 base，不重复存。

---

## 6. 真正的 Insight

这篇 paper 没发明任何全新组件——Scaffold-GS anchor、mip-map LOD、progressive training、opacity blending 都是已有 idea。它的 contribution 是 system-level: **把 LOD selection 从 rendering-time heuristic 变成 training-time joint optimization**。

在传统 graphics 里 LOD 是 artist 手工做的 mesh hierarchy，在 NeRF 里 LOD 是 cone-casting 的 sampling 自适应，在 Octree-GS 里 LOD 是 anchor 的 learnable 多分辨率结构 + learnable bias。同一个数据结构 (octree anchor) 同时承担 "spatial scaffold" 和 "LOD container" 两个职责，让 LOD 可以 end-to-end 训练。这个"职责合并"才是工程美感所在。

更深层的直觉: NeRF 系靠 sampling 解决 multi-scale (远的 pixel 用粗 sampling)，GS 没有 sampling，只能在 primitive selection 层面做 LOD——这就是 octree anchor 的本质作用。**Octree-GS 是把传统 graphics 的 LOD 在 differentiable rendering 框架里重做了一遍**，用 learnable 参数代替手工阈值。

---

## 7. Limitations & 我的联想

作者承认的:
1. $K$、$\delta$、$\omega$、$\tau_v$ 等 hyperparams 要手动调
2. 依赖 COLMAP poses (没做 pose-free)
3. 没有显式 geometry support (inherit from 3D-GS)

我能想到的延伸:
1. **Streaming rendering**: octree 天然适合 streaming——先传 coarse LOD，再传 fine LOD。Web 端做 progressive GS streaming 是很自然的下一步。
2. **Dynamic scene**: octree + 4D Gaussians 可能处理 dynamic LOD——远处用粗 temporal resolution，近处用细的。
3. **Differentiable K**: 现在 $K$ 是手动算的，可以让它 learnable，根据 scene complexity 自动决定层级数。
4. **Mesh + GS hybrid**: coarse LOD 用 mesh (轻量)，fine LOD 用 Gaussians，省 fine anchor 数量。
5. **Pose-free octree**: anchor voxel 在世界坐标系，pose 不准时 voxel quantization 错位。可以联合优化 pose + anchor。

---

## Reference Links

核心 paper 和 baseline:
- **Octree-GS project**: https://citysuper.github.io/octree-gs/
- **Scaffold-GS**: https://city-super.github.io/scaffold-gs/
- **3D-GS**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **2D-GS**: https://surhierarchical.github.io/2d-gaussian-splatting/
- **Mip-Splatting**: https://river-zhang.github.io/Mip-Splatting-pages/
- **Hierarchical-GS**: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- **CityGaussian**: https://dekuliutesla.github.io/citygaussian.github.io/

NeRF 系 LOD 范本:
- **Mip-NeRF 360**: https://jonbarron.info/mipnerf360/
- **Zip-NeRF**: https://arxiv.org/abs/2304.06706
- **BungeeNeRF**: https://github.com/city-super/BungeeNeRF
- **VR-NeRF** (continuous LOD 灵感来源): https://westlakeuvr.github.io/vrnerf/
- **PyNeRF** (pyramidal NeRF): https://arxiv.org/abs/2404.04957

经典 LOD 和 spatial structure:
- **Progressive Meshes (Hoppe)**: https://en.wikipedia.org/wiki/Progressive_meshes
- **Instant-NGP** (multi-resolution hash): https://nvlabs.github.io/instant-ngp/
- **PlenOctree** (octree precedent): https://alexyu.net/plenoctrees/

Appearance embedding:
- **NeRF-W**: https://nerf-w.github.io/
- **GLO**: https://arxiv.org/abs/1707.05776

Dataset:
- **MatrixCity**: https://citysuper.github.io/matrixcity/

---

最后用一句最直白的话总结: **Octree-GS 把 Scaffold-GS 的均匀 anchor 网格换成多分辨率 octree，让远看自动用粗 anchor、近看自动累加细 anchor，配合 progressive training 让 anchor 自然分配到对的 LOD——结果是大场景渲染快 8 倍、存储省 60%、质量还更好**。是 GS 时代 LOD 的标杆 system-level 工作。

---

# Octree-GS: 把 LOD 显式嵌入 3D Gaussian Splatting 的 hierarchical 框架

## TL;DR

Octree-GS 把 Scaffold-GS 的 anchor grid 升级为 multi-resolution octree-structured anchors，让 Gaussian primitives 自然分布到不同 LOD level 上。根据 observation distance 实时 query 出该 viewpoint 需要的 LOD level $K$，然后 cumulative 地渲染 LOD 0 到 $K$ 的所有 primitives，并用 opacity blending 抹平 LOD 切换处的 popping。结果是：在 MatrixCity Block_All 这种大规模场景下，相比 Scaffold-GS 渲染快 ~8.5×，相比 Hierarchical-GS 训练快 3×，同时 PSNR 更高、storage 更小。Project page: https://citysuper.github.io/octree-gs/

---

## 1. 核心问题: 为什么 3D-GS 在大规模场景崩盘

想象你在 MatrixCity 这种 1km × 1km 的城市数据集上跑 3D-GS。你需要把每个 building 的窗框、栏杆、路灯这些 high-frequency 几何都用 anisotropic Gaussian 拟合出来，最终会到 millions 量级 primitives。然后用户从 500m 高空俯瞰整个城市——按 3D-GS 的 frustum culling 逻辑，**所有落在 view frustum 内的 Gaussians 都要参与 α-blending**，哪怕每个 primitive 投影到屏幕上只有 0.01 pixel。

这造成两个具体问题：

1. **Rendering speed 与 trajectory 强耦合**: 拉远看时 #primitives 几乎不变（因为 frustum 还是覆盖整个场景），但实际每个 primitive 的 contribution 微乎其微。FPS 从 close-view 的 100+ 掉到 far-view 的 10-。
2. **Capacity 错配**: 远处看大场景时根本不需要窗框这种细节，但 3D-GS 的 primitives 已经被训练去拟合 view-dependent 的高分辨率细节，无法"换低清"地表示。

NeRF 时代的 Mip-NeRF / BungeeNeRF / Zip-NeRF 已经用 cone-casting、pyramid 结构、supersampling 解决了 multi-scale aliasing 问题。GS 时代缺乏一个对等的机制——Scaffold-GS 引入了 anchor 但 anchor grid 还是 single-resolution，Mip-Splatting 用了 3D smoothing filter 但仍然渲染所有 in-frustum primitives。Octree-GS 的核心 motivation 就是: **在 GS 框架内引入一个 explicit、训练时端到端、推理时可 query 的 LOD hierarchy**。

参考 Scaffold-GS: https://city-super.github.io/scaffold-gs/
参考 Mip-Splatting: https://river-zhang.github.io/Mip-Splatting-pages/
参考 3D-GS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 2. Octree 结构: 显式 multi-resolution anchor grid

### 2.1 Anchor 定义回顾 (Scaffold-GS)

Scaffold-GS 的核心 idea: 用一组 anchor 作为 spatial scaffold，每个 anchor $x_v$ emit $k$ 个 neural Gaussians，位置由 learnable offset $\mathcal{O}_i$ 和 scaling factor $l_v$ 决定:

$$\{\mu_0, \dots, \mu_{k-1}\} = x_v + \{\mathcal{O}_0, \dots, \mathcal{O}_{k-1}\} \cdot l_v$$

opacity/scale/rotation/color 全部由 anchor feature $\hat{f}_v$ 经过 MLP 解码:

$$\{\alpha_0, \dots, \alpha_{k-1}\} = F_\alpha(\hat{f}_v, \Delta_{vc}, \tilde{d}_{vc})$$

这里 $\Delta_{vc}$ 是 anchor 到 camera 的相对 viewing distance，$\tilde{d}_{vc}$ 是方向。Scaffold-GS 已经在 anchor 层面引入了 view-dependent 机制（opacity MLP 看到 viewing distance），这给 LOD 留下了自然的接口——只是 anchor grid 还是 uniform single-resolution。

### 2.2 Octree 化的关键设计

Octree-GS 把 anchor 从 uniform grid 换成 multi-resolution voxel grid，第 $L$ 层的 voxel size 是 $\delta / 2^L$，其中 $\delta$ 是 LOD 0 (最粗) 的 base voxel size。LOD 0 对应最粗，$L$ 越大 voxel 越小、细节越丰富。**累积式 LOD (cumulative LOD)** 是关键设计：渲染 LOD $K$ 等价于渲染 LOD $0, 1, \dots, K$ 所有 primitives 的并集。这一点很重要——它意味着 coarse LOD 是 fine LOD 的"base layer"，coarse anchor 永远在 rendering，fine anchor 在近距离时累加上去。这避免了 single-LOD selection 切换时的 popping。

### 2.3 Octree 层数 K 的确定

$$K = \lfloor \log_2(\hat{d}_{max} / \hat{d}_{min}) \rceil + 1$$

变量含义:
- $K$: octree 总层数（LOD 0 到 LOD $K-1$）
- $\hat{d}_{max}$, $\hat{d}_{min}$: 第 $r_d$-th 最大/最小的 camera-to-SfM-point 距离，$r_d = 0.999$ 用来剔除 outlier（比如极远的 background point 或废 COLMAP 点）
- $\lfloor \cdot \rceil$: round operator

**Intuition**: $K$ 由场景观察距离的"octave 数"决定。如果最近观察距离是 1m，最远是 16m，那 $\log_2(16) = 4$，加上 1 就是 $K = 5$。每升一层 voxel 减半，刚好对应"近一倍距离看一倍细"的几何尺度律。这个公式直接来自 mip-map 的 $\log_2(\text{resolution})$ 思想，但用 observation distance 而不是 texture resolution 来 measure octave。

### 2.4 Anchor 初始化

对每一层 LOD $L$:

$$\mathbf{V}_L = \left\{\left\lfloor \frac{\mathbf{P}}{\delta/2^L} \right\rceil \cdot \delta/2^L \right\}$$

变量含义:
- $\mathbf{P}$: 输入 SfM 稀疏点云
- $\mathbf{V}_L$: LOD $L$ 层 initialed 的 anchor 集合
- $\lfloor \cdot \rceil$: round to nearest voxel center

**Intuition**: 把 SfM 点 quantize 到对应 LOD 的 voxel 中心。同一片 SfM 点在 LOD 0 会被 quantize 到一个粗 voxel 里（合并），在 LOD 2 会被 quantize 到 8 个细 voxel 里（分散）。这样自然形成"coarse anchor 数量少、fine anchor 数量多"的分布。注意这里没用真正的 octree 数据结构（不做稀疏存储），更像 multi-resolution uniform voxel grid——但因为 anchor 只在有 SfM 点的 voxel 处初始化，sparsity 是隐式保证的。

### 2.5 Anchor Selection (核心 LOD query)

对于 viewpoint $i$ 看 anchor $j$:

$$\hat{L}_{ij} = \lfloor L^*_{ij} \rfloor = \lfloor \Phi(\log_2(d_{max} / (d_{ij} \cdot s))) + \Delta L_j \rfloor$$

变量含义:
- $d_{ij}$: viewpoint $i$ 到 anchor $j$ 的欧氏距离
- $s$: focal scale factor，处理 multi-resolution / 不同 intrinsics 的情况。距离要按 focal length 归一化，因为同样 10m 距离，长焦镜头拍出来的 footprint 比广角小得多
- $d_{max}$: 之前定义的最远 observation distance
- $\Phi(\cdot)$: clamping function，把 fractional LOD 限制在 $[0, K-1]$
- $\Delta L_j$: anchor $j$ 自己的 learnable LOD bias（重点！）
- $\hat{L}_{ij}$: 该 viewpoint 对该 anchor 估算的"需要的 LOD level"

**Intuition**: 距离 $d_{ij}$ 越大，$\log_2(d_{max}/d_{ij})$ 越小（接近 0），意味着只需要 LOD 0 这种粗的；距离越小，这个值越大，需要更细的 LOD 来呈现细节。这和 mip-map 在屏幕上选择 texture level 的逻辑同构——只是把 pixel footprint 换成了 observation distance + focal scale 的代理。

**$\Delta L_j$ 为什么必要**: 如果纯靠距离算 LOD，sharp edges（比如建筑轮廓）在 medium distance 处会突然丢失细节，因为距离阈值太死板。$\Delta L_j$ 给每个 anchor 一个 learnable residual，让 high-frequency 区域的 anchor 可以"提升"自己的 LOD 要求。具体: 训练时如果某 anchor 的平均 gradient $\nabla_v > \tau_g^L \cdot 0.25$，就 $\Delta L \mathrel{+}= \epsilon$（$\epsilon = 0.01$）。这相当于"如果你这个 anchor 经常欠拟合，就让自己在更近距离才被激活"——让 LOD selection 自适应场景复杂度。

**Cumulative selection**: 一个 anchor 被选中当且仅当 $L_j \leq \hat{L}_{ij}$，也就是说 anchor 自己的 LOD level 不超过 viewpoint 估算需要的 LOD。远处看时 $\hat{L}_{ij}$ 接近 0，只有 LOD 0 的 coarse anchor 被选；近处看时 $\hat{L}_{ij}$ 接近 $K-1$，所有 LOD 的 anchor 都参与渲染。这就是 cumulative LOD。

**Opacity blending for smooth transition**: 还会额外选 $L_j = \hat{L}_{ij} + 1$ 的 anchor，它们的 opacity 乘以 $L^*_{ij} - \hat{L}_{ij}$（fractional part）。这是 piecewise linear interpolation，保证 LOD 切换时不会 popping。借鉴了 VR-NeRF 和 BungeeNeRF 的连续 LOD 思路。

参考 progressive meshes (Hoppe): https://en.wikipedia.org/wiki/Progressive_meshes
参考 VR-NeRF: https://westlakeuvr.github.io/vrnerf/

---

## 3. Grow & Prune: 让 anchor 分布到正确的 LOD

### 3.1 Next-level Growing

标准 Scaffold-GS densification: 累积 gradient $\nabla_g > \tau_g$ 的 Gaussian → 在它所在的空 voxel 里 grow 新 anchor。Octree-GS 的关键问题: 这个新 anchor 应该在 LOD $L$ 还是 LOD $L+1$?

作者提出 **monotonic threshold promotion**:

$$\tau_g^L = \tau_g \cdot 2^{\beta L}$$

变量含义:
- $\tau_g$: base gradient threshold（default 0.0002）
- $\beta$: growth difficulty factor（default 0.2）
- $\tau_g^L$: 在 LOD $L$ 上 grow 出新 anchor 需要超过的 gradient 阈值

规则:
- 如果 $\nabla_g > \tau_g^{L+1}$: 这个 Gaussian 被 promote 到 LOD $L+1$，在 LOD $L+1$ 的更细 voxel 中创建新 anchor
- 如果 $\tau_g^L < \nabla_g < \tau_g^{L+1}$: 停留在 LOD $L$，在 LOD $L$ voxel 中创建

**Intuition**: 越细的 LOD 越"难进入"，要 gradient 非常大才 promote 上一层。这防止了训练初期大量 Gaussians 一窝蜂涌到 fine LOD，导致 coarse LOD 没人管。指数 $2^{\beta L}$ 而不是线性 $\beta L$ 是为了和 octree 的"voxel 减半"几何尺度匹配——每升一层 voxel volume 减 8 倍，需要 8 倍以上的"信号"才值得用细 LOD 表示。

### 3.2 View-frequency Pruning

标准 pruning 是看 average opacity。但作者发现: octree 高 LOD 的 anchor 只在近距离 view 才被选中，训练数据里这种近距离 view 数量少，导致 fine anchor 长期不更新、学坏、变成 floater。

定义 **view-frequency** = anchor 在训练 view 中被选中的概率（和它收到的 gradient 强相关）。Pruning 规则: view-frequency < $\tau_v$ 就删掉这个 anchor。$\tau_v$ 是 dataset-dependent:
- 小场景（Mip-NeRF360、Deep Blending）: $\tau_v = 0.7$（密集拍摄）
- 大场景（MatrixCity、MegaNeRF、UrbanScene3D）: $\tau_v = 0.01$（稀疏 view，不能太严）
- 多尺度（BungeeNeRF）: $\tau_v = 0.2$

这个 ablation 在 Table VII 里特别明显: 关掉 view freq. 后 PSNR 从 28.05 掉到 27.74，但 storage 从 139.6M 涨到 244.4M——质量变差还更占空间，因为 floater 留下了。

---

## 4. Progressive Training: 强制 coarse-to-fine

如果一开始就同时优化所有 LOD 的 anchor，会出现什么? Figure 3 第二行给出了答案: 所有 LOD 的 anchor 都试图表达整个场景，导致 LOD 之间高度重叠，fine LOD 抢了 coarse LOD 的活，coarse LOD 在远距离 view 时质量很差。

Progressive training:
- 从 LOD $\lfloor K/2 \rfloor$ 开始训练（不是 LOD 0，因为太粗会 underfitting 太久）
- 每 $N_i$ iterations 激活下一层 LOD
- 越粗的 LOD 训练越久: $N_{i-1} = \omega \cdot N_i$，$\omega \geq 1$（default 1.5）
- Progressive 阶段**禁用 next-level grow operator**（不让 anchor 在 LOD 之间跳）

**Intuition**: 这和 BungeeNeRF、NeRF++ 的 coarse-to-fine 训练思路一致——让模型先用 coarse structure 把几何骨架打好，再用 fine LOD 补 high-frequency detail。$\omega > 1$ 的设计是给 coarse LOD 更长的"独占训练时间"，因为 coarse LOD 要在所有远距离 view 都表现好，是"基础"。

Ablation (Table VII): 关掉 progressive training，PSNR 从 28.05 掉到 27.86，rendered Gaussians 从 657k 涨到 698k——验证了 LOD entanglement 导致的冗余。

---

## 5. Appearance Embedding: 处理 in-the-wild 场景

用 GLO (Generative Latent Optimization, Bojanowski et al. 2017) 给每个 anchor 一个 learnable appearance code，喂给 color MLP。可以 linearly interpolate appearance code 做 style transfer（Figure 12 那棵银杏树例子很直观）。这继承了 NeRF-W 的处理 wild photo collection 的思路。

参考 GLO: https://arxiv.org/abs/1707.05776
参考 NeRF-W: https://nerf-w.github.io/

---

## 6. 实验结果分析: 关键数据解读

### 6.1 渲染速度 (Table V, MatrixCity Block_All)

| Method | $T_1$ | $T_2$ | $T_3$ |
|---|---|---|---|
| 3D-GS | 13.81 | 11.70 | 13.50 |
| Scaffold-GS | 6.69 | 7.37 | 8.04 |
| Hierarchical-GS ($\tau_3$, coarsest) | 24.33 | 19.59 | 18.94 |
| Hierarchical-GS ($\tau_1$, finest) | 16.14 | 13.26 | 14.79 |
| Our-3D-GS | **57.08** | **56.85** | **56.07** |

Our-3D-GS 比 Scaffold-GS 快 ~8×，比 Hierarchical-GS 最快配置快 ~3×。关键: Octree-GS 在三个不同 trajectory 上 FPS 几乎不变（~56），而 baselines 在不同 trajectory 上波动很大。这正是论文标题 "Consistent Real-time Rendering" 的实证。

### 6.2 大规模场景质量 (Table II, Block_All)

| Method | PSNR | SSIM | LPIPS | #GS(k) | Mem |
|---|---|---|---|---|---|
| Scaffold-GS | 26.30 | 0.808 | 0.293 | 690 | 2272.2M |
| CityGaussian | 26.26 | 0.800 | 0.324 | 235 | 4316.6M |
| Hierarchical-GS | 26.00 | 0.803 | 0.306 | 492 | 4874.2M |
| Our-Scaffold-GS | **27.31** | **0.849** | **0.229** | 344 | 1648.6M |

PSNR +1.0 dB，storage 比 Scaffold-GS 少 28%，比 Hierarchical-GS 少 66%。LPIPS 0.229 也是最低。

### 6.3 训练时间对比 (Mip-NeRF360, 40k iter)

- 3D-GS: 34 min
- Scaffold-GS: 29 min
- Hierarchical-GS: 69 min（3 stages，第一 stage 就 38 min）
- Our-Scaffold-GS: 23 min (single stage)

Octree-GS 不仅没增加训练时间，反而比 Scaffold-GS 还快——因为 progressive training 早期不优化所有 anchor，#primitives 总体更小。

### 6.4 Multi-resolution (Table VI, Mip-NeRF360 downsampled 1×/2×/4×/8×)

Our-Scaffold-GS 在所有 4 个 resolution 上 PSNR 都最好，验证 LOD 在 scale 变化时的 anti-aliasing 能力。Multi-resolution 训练时 focal scale $s$ 起作用——downsample 越多，等效 distance 越大（因为 footprint 变小），需要更粗 LOD。

### 6.5 BungeeNeRF (Table IV, 4 scales)

观察 #GS per scale:
- 3D-GS: scale-1 522k → scale-4 5821k（11× 增长）
- Scaffold-GS: 303k → 3876k（12.8× 增长）
- Our-Scaffold-GS: 486k → 2167k（4.5× 增长）

这就是 LOD 的价值——远距离看时不需要 fine Gaussians 累加，coarse LOD 一个 anchor 表达整片区域。

---

## 7. 与 concurrent works 的关键差异

| 维度 | Octree-GS | Hierarchical-GS | CityGaussian | LetsGo |
|---|---|---|---|---|
| LOD structure | explicit octree | tree hierarchy | distance interval | multi-resolution point cloud |
| 训练 | end-to-end single stage | 3 stages | multi-stage | joint optimization |
| 需要 lidar/精确点云 | 否 | 否 | 否 | 是 |
| LOD 切换 | opacity blending | hard switch | fusion | joint |
| Storage | accumulative (共享 coarse) | independent per LOD | independent | independent |

Hierarchical-GS 的 3-stage 训练 (38+?+69 min total) 是最大短板: 第一阶段 train base，第二阶段 build hierarchy，第三阶段 fine-tune。Octree-GS 把这些都集成到单次训练里——通过 progressive training 实现"边训练边构建 hierarchy"。

CityGaussian 的 distance interval 切换有 stroboscopic artifacts——切换点不连续。Octree-GS 的 opacity blending 直接解决了这个。

---

## 8. Ablation 拆解 (Table VII)

| 配置 | PSNR | #GS(k) | Mem |
|---|---|---|---|
| Full | 28.05 | 657 | 139.6M |
| w/o next grow | 27.64 | 594 | 99.7M |
| w/o progressive | 27.86 | 698 | 142.3M |
| w/o LOD bias | 27.85 | 667 | 146.8M |
| w/o view freq | 27.74 | 765 | 244.4M |

有意思的观察:
- **Next-level grow**: 关掉后 storage 大降（99.7M vs 139.6M）但 PSNR 也降 0.4 dB。说明 fine anchor 是必要的"细节补充"，但确实占存储。
- **View freq**: 关掉后 storage 翻倍（244.4M vs 139.6M），PSNR 也降——证明 floater 不仅占地方还干扰渲染。
- **Progressive + LOD bias**: 影响相对小但仍 0.2 dB，是 quality 的 fine-tuning。

---

## 9. Limitations & Intuition 总结

作者承认的限制:
1. Octree construction 的 hyperparameters ($r_d$, $\delta$, $K$ formula) 需要手动调
2. Progressive training 的 $\omega$、activation iteration 也要调
3. 仍然依赖 COLMAP poses（没解决 pose-free 问题）
4. 没有显式 geometry support（inherit from 3D-GS）

### Intuition-level Takeaway

Octree-GS 真正的 insight 是: **把 Scaffold-GS 的 anchor grid 从 single-resolution 升级为 multi-resolution，并把 LOD selection 从 "rendering-time heuristic" 变成 "training-time joint optimization"**。Anchor 既是 spatial scaffold 又是 LOD container——同一个数据结构承担两个职责，使得 LOD 学习可以 end-to-end。Cumulative LOD + opacity blending 是把传统 graphics 的 discrete LOD 平滑化，让它适配 GS 的 continuous α-blending rendering。

更深层的: 这篇工作把 classic graphics 的 LOD concept (Hoppe 1996 progressive meshes, mip-map) 用现代 differentiable rendering 的语言重新实现了一次。Classic LOD 是 artist-curated 的 discrete mesh hierarchy，Octree-GS 是 data-driven 的 continuous Gaussian hierarchy。NeRF 系的 Mip-NeRF / Zip-NeRF 用 cone-casting 在 sampling 层面做 LOD，GS 系没有 sampling，只能在 primitive selection 层面做——这就是 octree anchor 的本质作用。

### 对未来工作的联想

1. **Streaming rendering**: Octree-GS 天然适合 streaming——粗 LOD 先到、细 LOD 后到。可以在 web 上做 progressive GS streaming。
2. **与 mesh-GS hybrid**: Octree 的 coarse LOD 可以是 mesh（轻量），fine LOD 是 Gaussians，节省 fine anchor 数量。
3. **Differentiable octree**: 现在 $K$ 和 $\delta$ 是手动设置的，未来可以让 $K$ 也 learnable，根据 scene complexity 自动决定层级数。
4. **Pose-free octree**: 现在 anchor voxel 是世界坐标系的，如果 camera pose 不准，voxel quantization 会错位。可以联合优化 pose + anchor。
5. **Dynamic scene extension**: Octree + 4D Gaussians 可能处理 dynamic scene 的 LOD——远处用粗 temporal resolution，近处用细 temporal resolution。

---

## Reference Links

- **Octree-GS project page**: https://citysuper.github.io/octree-gs/
- **Scaffold-GS (基础)**: https://city-super.github.io/scaffold-gs/
- **3D-GS (原始)**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **2D-GS**: https://surihierarchical.github.io/2d-gaussian-splatting/
- **Mip-Splatting**: https://river-zhang.github.io/Mip-Splatting-pages/
- **Hierarchical-GS**: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- **CityGaussian**: https://dekuliutesla.github.io/citygaussian.github.io/
- **MatrixCity dataset**: https://citysuper.github.io/matrixcity/
- **Mip-NeRF 360**: https://jonbarron.info/mipnerf360/
- **BungeeNeRF**: https://github.com/city-super/BungeeNeRF
- **Instant-NGP (multi-resolution hash)**: https://nvlabs.github.io/instant-ngp/
- **PyNeRF (pyramidal NeRF)**: https://arxiv.org/abs/2404.04957
- **VR-NeRF (continuous LOD inspiration)**: https://westlakeuvr.github.io/vrnerf/
- **Progressive Meshes (Hoppe)**: https://en.wikipedia.org/wiki/Progressive_meshes
- **NeRF in the Wild (appearance embedding)**: https://nerf-w.github.io/
- **GLO (appearance code)**: https://arxiv.org/abs/1707.05776
- **Zip-NeRF (anti-aliased grid NeRF)**: https://arxiv.org/abs/2304.06706
- **PlenOctree (octree precedent)**: https://alexyu.net/plenoctrees/
- **ACORN (octree neural net)**: https://arxiv.org/abs/2105.02788

整个方法读下来，最 elegant 的地方是它**没有发明新东西，而是把已有的几个 idea (Scaffold-GS anchor + mip-map LOD + progressive training + opacity blending) 组合得非常紧**。每个组件单独都不新，但拼在一起形成了一个让 GS 在大规模场景真正可用的 system——这是 systems-level 的 contribution，工程美感很强。
