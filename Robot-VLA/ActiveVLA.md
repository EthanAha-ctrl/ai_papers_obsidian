---
source_pdf: ActiveVLA.pdf
paper_sha256: 6a46d2643e9c83abd5db2ecb7d7da30a7b387a824eff66434437bc0012f1db22
processed_at: '2026-07-18T01:33:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ActiveVLA: 把 Active Perception 注入 VLA 的深度解析

Andrej，这篇paper我读得很过瘾，因为它直接戳中了一个我一直觉得尴尬的事实：现存的 VLA 模型本质上是 **passive perceivers**——相机绑在 wrist 上或者固定在前方，模型只能"等"图像进来再反应。而生物视觉是 saccadic、hypothesis-driven 的。ActiveVLA 试图把这个 active loop 显式做出来，并且嵌入到 PaliGemma 这种 VLM backbone 里。下面我把所有技术细节掰开讲，重点是 build your intuition。

---

## 1. 为什么这件事 hard：传统 VLA 的 perception bottleneck

传统 VLA 的 setup（公式1）:

$$\pi: (\mathbf{o}, l) \mapsto \mathbf{a}$$

- $\mathbf{o}$: observation，通常是 RGB-D images，从一个或多个固定的 end-effector-centric viewpoint 拍摄
- $l$: language instruction
- $\mathbf{a}$: action，包含 $T \in SE(3)$（6-DoF end-effector pose）、gripper state $g \in \{0,1\}$、collision flag $c \in \{0,1\}$
- $\tau^i = \{l^i, (\mathbf{o}_1^i, \mathbf{a}_1^i), \dots, (\mathbf{o}_H^i, \mathbf{a}_H^i)\}$: 第 i 条 expert demo trajectory

问题出在 $\mathbf{o}$：camera 是 fixed 的，所以：
- 物体被 drawer / 其他物体 occlude 时，无论怎么 train 都看不到
- 小物体（比如 welding gun 要对准的孔）在低分辨率下像素不够，policy 输出 jitter
- Long-horizon task 中前面 stage 把 workspace 弄乱之后，原 viewpoint 失效

这正是 Richard Gregory 那句"perception is an active process of hypothesis testing"在 robotics 语境下的具象化。

参考：[Active perception (Bajcsy 1988)](https://ieeexplore.ieee.org/document/5901) / [Gregory's perception theory](https://en.wikipedia.org/wiki/Richard_Gregory_(psychologist))

---

## 2. 整体架构：Coarse-to-Fine 的 Active Perception Pipeline

Figure 2 的 pipeline 我重新画一遍给你看清楚逻辑流：

```
[RGB-D inputs] 
      │
      ▼ (reconstruct point cloud)
[3D Point Cloud]
      │
      ├──► [Coarse Stage] ─────────────────────────────┐
      │     - 3 orthographic projections (top/front/right)│
      │     - 每个view 7 channels: RGB(3) + Depth(1) + XYZ(3)│
      │     - PaliGemma backbone → patch tokens         │
      │     - Convex upsampling → 2D heatmaps           │
      │     - Back-project to 3D → crucial region p_f   │
      │                                                  │
      ▼                                                  │
[3D Crucial Region p_f ∈ R^3] ◄────────────────────────┘
      │
      ├──► [Fine Stage] ────────────────────────────────┐
      │     Active View Selection:                       │
      │     - Geodesic sampling on icosahedron sphere    │
      │     - Score by (visibility, distance, diversity) │
      │     - Top-K viewpoints selected                  │
      │                                                   │
      │     Active 3D Zoom-in:                            │
      │     - Narrow FoV re-render at p_f                │
      │     - Higher effective resolution R              │
      │                                                   │
      ▼                                                   │
[Refined multi-view images] ◄───────────────────────────┘
      │
      ▼
[PaliGemma (fine stage)]
      │
      ├──► Heatmaps for end-effector keypoints
      ├──► Back-projection → 3D score volume → translation t*
      ├──► Global tokens (max-pool per view) 
      ├──► Local tokens (ROI-aware sampler)
      │
      ▼
[MLP action head]
      │
      ▼
[6-DoF pose T, gripper g, collision c]
```

Intuition：先用一个 cheap 的 global pass 找到"哪里值得看"（coarse），再花算力在那个 region 周围生成 hypothesis viewpoints，从中挑出信息量最大的几个，最后在 virtual renderer 里 zoom in 把分辨率堆上去（fine）。这是经典的 exploration → exploitation 分层。

---

## 3. Coarse Stage: 3D Crucial Area Perception

### 3.1 Multi-View Rendering（公式2）

$$I^{(v)}(u_x, u_y) = \sum_{i=1}^{N} \mathbf{c}_i \cdot \delta\big((u_x, u_y) - \pi^{(v)}(\mathbf{p}_i)\big)$$

变量含义：
- $I^{(v)}(u_x, u_y)$: 视角 $v \in \{\text{top, front, right}\}$ 下像素 $(u_x, u_y)$ 的渲染值
- $N$: point cloud 中点的数量
- $\mathbf{c}_i \in \mathbb{R}^7$: 第 i 个点的 channel vector，包括 RGB(3) + depth(1) + world-frame XYZ(3)
- $\pi^{(v)}(\cdot)$: 视角 $v$ 的 orthographic projection 函数
- $\mathbf{p}_i \in \mathbb{R}^3$: 第 i 个点的 3D 坐标
- $\delta(\cdot)$: Kronecker delta，把 3D 点 splat 到 2D 像素上

**关键 trick**：XYZ 通道不是冗余的，而是 cross-view correspondence 的桥梁。如果两个不同 view 的像素都包含同一个 $(x,y,z)$，说明它们对应 3D 空间里的同一个点。这给后面 back-projection 留了钩子。

**Occlusion handling**: 同一像素上多个点投影时，取 minimal depth $z_i^{(v)}$，这样前面的物体覆盖后面的，几何上正确。

实现用 PyTorch3D，参考 [PyTorch3D paper](https://arxiv.org/abs/2007.08501)。

### 3.2 Heatmap Prediction（公式3）

VLM 的 global representation 不够 spatially precise，所以作者加了一个 heatmap head：

$$\mathbf{H} = \mathcal{U}\Big(\mathrm{Rearrange}\big(\{\mathbf{t}_i\}_{i=1}^M\big)\Big)$$

- $\{\mathbf{t}_i\}_{i=1}^M$: VLM backbone 输出的 $M$ 个 patch tokens
- $\mathrm{Rearrange}(\cdot)$: 把 token sequence 重新排列成 $H_p \times W_p$ 的 2D feature grid（PaliGemma 用的是 SigLIP encoder，patch token 本来就是空间排列的，这一步是 undo flatten）
- $\mathcal{U}(\cdot)$: convex upsampling block

**Convex upsampling 的 intuition**（来自 [Lite-HRNet](https://arxiv.org/abs/2104.06403)）：不是简单的 bilinear interpolation，而是对每个低分辨率 pixel 学一组 weight，用来 weighted-sum 它邻域的 high-resolution pixels。这相当于学一个 sparse 4-tap filter，比转置卷积参数少且更稳定。Fine-grained manipulation 需要 sub-pixel 精度，这一步是必要的。

Loss 是 cross-entropy，target 是 GT end-effector position 在 2D 视图下的 Gaussian peak（softmax 化之后）。

### 3.3 Back-projection 到 3D

3 个 view 的 heatmap 都投影到一个共享的 discretized 3D grid $\mathcal{G}$ 上，每个 grid point $\mathbf{g}$ 累积所有 view 的 score，最大值点就是 coarse stage 估计的 crucial region centroid $\mathbf{p}_f$。

这一步在 paper 里属于 coarse → fine 的 hand-off，没有显式公式但可以理解为：

$$\mathbf{p}_f \approx \arg\max_{\mathbf{g} \in \mathcal{G}} \sum_{v \in \{\text{top,front,right}\}} h_v(\pi_v(\mathbf{g}))$$

---

## 4. Fine Stage: Active 3D Perception

这是 paper 的真正 novelty。两个 module 串行：先选 viewpoint，再 zoom-in。

### 4.1 Active Viewpoint Selection

#### 4.1.1 Candidate generation: Geodesic sampling（公式4）

给定 $\mathbf{p}_f$，要在以它为中心的球面上均匀采样 candidate camera positions。如果直接用 latitude-longitude 参数化会有 pole-singularity（两极采样密集）。作者用 **icosahedron recursive subdivision**：

$$V(k) = 12 + 30k + \frac{20}{3}(4^k - 1)$$

- $V(k)$: 第 $k$ 次 subdivision 后的 vertex 数
- $k \in \mathbb{N}$: subdivision level
- $k=0$: 原始 icosahedron，12 个顶点
- $k=1$: 42 个顶点
- $k=2$: 162 个顶点
- 公式来源：每次 subdivision 把每条边二分，每个 triangle 拆成 4 个，所以 face 数 $F(k) = 20 \cdot 4^k$，由 Euler formula $V - E + F = 2$ 配合 $E = 3F/2$ 推出

这是球面 uniform sampling 的经典 trick，参考 [Geodesic polyhedron](https://en.wikipedia.org/wiki/Geodesic_polyhedron)。Intuition：subdivision 让三角形近似为球面等边三角形，顶点分布近似均匀。

#### 4.1.2 Multi-objective scoring

每个 candidate $c_i$ 算三个 score：

**(a) Visibility score**（公式5）

从 $c_i$ 到 $\mathbf{p}_f$ 的射线方向上均匀采 $N$ 个点 $\{q_k\}$，对每个点算到观测点云 $\mathcal{S}$ 的最近距离：

$$d_k = \min_{s \in \mathcal{S}} \|q_k - s\|$$

- $d_k$: 采样点 $q_k$ 到点云 $\mathcal{S}$ 的 closest-surface distance
- $q_k$: 射线上第 k 个采样点
- $\mathcal{S}$: 当前已经观测到的点云
- $s$: $\mathcal{S}$ 中的点

如果 $\forall k, d_k \geq r$（threshold），则 $v(c_i, \mathbf{p}_f) = 1$（unoccluded），否则 $0$。

**Intuition**：这是 ray-casting 的简化版。KDTree 加速 nearest-neighbor 查询到 $O(\log |\mathcal{S}|)$ per query，整体 $O(N \log |\mathcal{S}|)$ per candidate。$r$ 是个 hinge，相当于"允许穿过多少 noise"，防止点云本身有 hole 导致误判 occlusion。

**(b) Distance score**

$\|c_i - \mathbf{p}_f\|$ 归一化。太近 field-of-view 狭窄看不到 context，太远 resolution 低，所以偏好 moderate distance。Z-normalized 跨 candidate 标准化。

**(c) Diversity score**（公式6）

$$S_{\mathrm{div}}(c_i) = \sum_{j \neq i} \arccos(\mathbf{v}_i \cdot \mathbf{v}_j)$$

- $S_{\mathrm{div}}(c_i)$: candidate $c_i$ 相对所有其他 candidate 的 angular spread
- $\mathbf{v}_i, \mathbf{v}_j \in \mathbb{S}^2$: 单位 viewing direction vectors
- $\arccos(\mathbf{v}_i \cdot \mathbf{v}_j)$: 两个 viewing direction 之间的 geodesic distance on unit sphere

**Intuition**：选中的 top-K viewpoints 之间要 angular spread 大，这样 multi-view 才能提供 complementary 信息（类似 multi-view stereo 里的 baseline constraint）。这一步是相对于"贪婪选 top-K 单独最优"的改进。

#### 4.1.3 Unified score（公式7）

$$s_i = w_{\mathrm{vis}} \cdot s_{\mathrm{vis}} + w_{\mathrm{dis}} \cdot s_{\mathrm{dis}} + w_{\mathrm{div}} \cdot s_{\mathrm{div}}$$

- $s_i$: candidate $i$ 的最终 score
- $w_{\mathrm{vis}}, w_{\mathrm{dis}}, w_{\mathrm{div}}$: 三个权重，$w_{\mathrm{vis}} + w_{\mathrm{dis}} + w_{\mathrm{div}} = 1$
- $s_{\mathrm{vis}}, s_{\mathrm{dis}}, s_{\mathrm{div}}$: 分别是上面 Z-normalized 之后的 visibility、distance、diversity score

Top-K ($K=3$ in experiments) 选出来作为 next observation poses。每个 camera 用 look-at formulation：eye $= c_i$, target $= \mathbf{p}_f$, up vector 动态调整防 gimbal lock。

### 4.2 Active 3D Zoom-in（公式8）

这一步是"既然找到了好视角，那就把像素堆上去"：

$$W(z) = 2d \tan\left(\frac{\alpha}{2z}\right)$$

- $W(z)$: 渲染图像在垂直 viewing direction 方向上的 spatial coverage 宽度（米）
- $z > 1$: zoom-in factor
- $d$: camera 到 ROI 的距离
- $\alpha$: 原始 FoV（弧度）

而像素分辨率 $R$ 保持不变（因为 image width 是固定的 pixel 数），所以 effective spatial resolution:

$$R = \frac{\text{image width (pixels)}}{W(z)}$$

$z$ 增大 → $W(z)$ 减小 → $R$ 增大。

**关键 insight**：这是 virtual optical zoom，不是物理移动 camera。Virtual renderer 可以从同一 pose 用 narrowed FoV 重新渲染场景，相当于把 3D point cloud 在那个 region "放大"显示。物理 camera 做不到无损 zoom，但 virtual renderer 可以——这是 sim-to-real VLA 一个被低估的优势。

**Ablation 显示** $z=4$ 是甜点，再大会丢失 context。

### 4.3 Exploration vs Exploitation 的分离

paper 里我特别喜欢的一句总结：

> "By separating exploration (view selection) from exploitation (zoom-in), ActiveVLA forms a hierarchical perception strategy."

View selection 解决 "where to look"，zoom-in 解决 "how closely to look"。这种解耦在 RL 里对应 explore-exploit 的经典分离，但在 VLA 里被显式做出来还是第一次（据我所知）。

---

## 5. Action Prediction

### 5.1 Translation via Multi-view Score Volume（公式9）

$$S(\mathbf{g}) = \sum_{v=1}^{3} w_v h_v(\pi_v(\mathbf{g}))$$

- $S(\mathbf{g})$: grid point $\mathbf{g}$ 的累积 score
- $w_v$: 视角 $v$ 的 weight（可学，或者 uniform）
- $h_v(\cdot)$: 视角 $v$ 的 attention heatmap（fine stage PaliGemma 输出）
- $\pi_v(\mathbf{g})$: grid point $\mathbf{g}$ 在视角 $v$ 下的 2D projection

Translation target:

$$\mathbf{t}^* = \arg\max_{\mathbf{g} \in \mathcal{G}} S(\mathbf{g})$$

**Intuition**：3 个 view 都"亮"起来的 grid point 最有可能是真目标。这相当于 soft multi-view triangulation。比起直接 regress 6-DoF，discretize + voting 在 fine-grained 任务上稳定性高很多——这也是 PerAct、RVT 系列验证过的 design choice。

### 5.2 Rotation, Gripper, Collision via Hierarchical Fusion

Rotation 用 Euler angles $(\phi, \theta, \psi)$，每个 angle 离散成 72 bins（即 5° per bin）。

Feature fusion 分两层：
- **Global context**: 每个 orthographic view 的 vision encoder output 做 max-pool，得到 3 个 global tokens
- **Local context**: ROI-aware sampler 在 fine-grained view 上提取 local tokens

Concat 之后过 MLP head 输出 rotation（3 × 72 = 216 维分类）、gripper state（binary）、collision flag（binary）。

参考这种 global-local fusion 的设计可追溯到 [ViT + ROI pooling 思路](https://arxiv.org/abs/2010.11929)。

---

## 6. 实验结果深度分析

### 6.1 RLBench（Table 1）

| Method | Avg SR (%) | Avg Rank |
|---|---|---|
| HiveFormer | 45.3 | 8.22 |
| PerAct | 49.4 | 7.33 |
| Act3D | 65.0 | 5.28 |
| RVT | 62.9 | 5.39 |
| 3D Diffuser Actor | 81.3 | 3.39 |
| RVT-2 | 81.4 | 3.00 |
| BridgeVLA | 88.2 | 2.44 |
| **ActiveVLA** | **91.8** | **1.22** |

ActiveVLA 在 18 个 task 里 10 个排第一。一些有意思的 case：
- **Insert Peg**: 92.4%（precision-demanding，得益于 zoom-in）
- **Place Cups**: 65.6%（heavy occlusion，得益于 active view selection）
- **Sweep to Dustpan**: 100.0%（这个 task occlusion 严重，active perception 直接打通）
- **Sort Shape**: 63.3%（最低，因为这个 task 本身定义复杂）

**vs BridgeVLA 的差距来源**：BridgeVLA 也是 PaliGemma-based，但只用 fixed orthographic views。ActiveVLA 多了 active view + zoom-in 两个 module，所以差距直接反映 active perception 的价值。

### 6.2 COLOSSEUM（Table 2）

COLOSSEUM 是 RLBench 的 robustness benchmark，加 14 类 perturbation：

| Method | Avg SR (%) | Avg Rank |
|---|---|---|
| RVT-2 | 56.7 | 2.86 |
| BridgeVLA | 64.0 | 2.07 |
| **ActiveVLA** | **65.9** | **1.07** |

具体类别看：
- **Camera Pose**: 76.3%（vs BridgeVLA 73.8%）— 这一项最直接反映 active perception 的价值，因为 camera pose 改变就是 viewpoint shift，ActiveVLA 能再 active 选 view 补偿
- **Distractor**: 54.3%（只小幅领先）— distractor 是 object-level 而非 view-level challenge，active perception 帮助有限
- **Background Texture**: 75.2%（小幅领先）— VLM 预训练已经处理了大部分 texture invariance

### 6.3 GemBench（Table 3）

GemBench 是 hierarchical generalization benchmark，L1-L4 难度递增：

| Method | Avg | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| 3D-LOTUS++ | 48.0 | 68.7 | 64.5 | 41.5 | 17.4 |
| BridgeVLA | 50.0 | 91.1 | 65.0 | 43.8 | 0.0 |
| **ActiveVLA** | **51.3** | 92.4 | 66.3 | 45.1 | 1.2 |

**L4 的崩盘很有意思**：3D-LOTUS++ 在 L4 有 17.4%，ActiveVLA 只有 1.2%。L4 是 long-horizon compositional generalization。说明 active perception 解决的是"看不清"的问题，但解决不了"task graph 从未见过的组合"问题——那是 high-level reasoning 的范畴。这其实界定了 active perception 的能力边界。

### 6.4 Real-World（Table 5）

真实机器人实验（KINOVA GEN2 + RealSense D455, eye-to-hand setup）：

| Method | Towel | Red→Green | Banana | Cup | Overall |
|---|---|---|---|---|---|
| Diffusion Policy | 26 | 38 | 42 | 35 | 35.3 |
| VPP | 52 | 48 | 58 | 64 | 55.5 |
| TriVLA | 68 | 54 | 62 | 72 | 64.0 |
| RVT-2 | 77 | 63 | 72 | 78 | 72.5 |
| **ActiveVLA** | **92** | **95** | **91** | **89** | **91.8** |

Real-world 的提升比 sim 更大（24% / 41% / 29% / 17% over TriVLA），说明 sim-to-real gap 里很大一部分确实是 perception 的问题——real sensor 噪声 + 非程序化 occlusion 让 fixed view 更脆，而 active view 自带 robustness。

参考 [TriVLA](https://arxiv.org/abs/2507.01424) / [RVT-2](https://robotic-view-transformer-2.github.io/) / [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

### 6.5 Ablation（Table 4）

| A-VS | A-3Z | RLBench | COLOSSEUM | GemBench | Time (s) |
|---|---|---|---|---|---|
| ✗ | ✗ | 87.6 | 63.6 | 48.9 | 0.26 |
| ✓ | ✗ | 89.4 | 64.5 | 49.4 | 0.45 |
| ✓ | ✓ | 91.8 | 65.9 | 51.3 | 0.53 |

- A-VS 单独加：+1.8% on RLBench，cost +0.19s
- A-3Z 在 A-VS 之上加：+2.4% on RLBench，cost +0.08s

A-3Z 的 marginal cost 比 A-VS 小（zoom-in 只是一次 re-render，不需要再 sample candidate + score），但 marginal gain 更大。这说明 "看清"比"找对角度"对 fine-grained 任务更关键。

### 6.6 Hyperparameter（Figure 5）

- **Number of views**: 1 view → 82.2%, 2 → ~88%, 3 → 91.8%, 4+ 饱和。3 是甜点
- **Zoom factor**: $z=1$ baseline, $z=2/3/4$ 持续提升, $z>4$ 下降（context 丢失）

参考信息论 intuition：multi-view 的边际信息量随 view 数递减（views 之间相关性变高），而 zoom 的边际信息量在 pixel density 超过 object feature size 后饱和。

---

## 7. 训练细节

backbone 是 [PaliGemma](https://arxiv.org/abs/2407.07726)（SigLIP vision encoder + Gemma language decoder），预训练在 120K-image [RoboPoint](https://arxiv.org/abs/2406.10721) subset。

| Stage | LR | Batch | Steps | GPUs | Hours |
|---|---|---|---|---|---|
| RLBench | 8e-5 | 192 | 90k | 16×H100 | ~20 |
| COLOSSEUM | 8e-5 | 192 | 90k | 16×H100 | ~20 |
| GemBench | 8e-5 | 160 | 50 epochs | 32×H100 | ~8 |
| Real-robot | 2e-5 | 192 | 400 epochs | 8×H100 | ~2 |

**关键 trick**：SigLIP vision encoder 和 language token embeddings 全程 frozen。理由是 active viewpoint 会持续改变 input distribution，如果 vision encoder 不 frozen，embedding space 会 drift，破坏 VLM 预训练知识。只有 action head、viewpoint selection module、3D zoom-in module 更新。

Optimizer 是 AdamW + cosine LR decay + gradient clip (max norm 1.0)，no warmup。

---

## 8. 与相关工作的 positioning

| 方法 | 2D/3D | Viewpoint | Backbone |
|---|---|---|---|
| [OpenVLA](https://openvla.github.io/) | 2D | Fixed wrist | Prismatic VLM |
| [3D-VLA](https://3d-vla.github.io/) | 3D | Fixed | LLaMA + 3D tokenizer |
| [PointVLA](https://arxiv.org/abs/2503.07511) | 2D+3D | Fixed | VLM + point encoder |
| [SpatialVLA](https://spatial-vla.github.io/) | 2D+Ego3D | Fixed | VLM + 3D pos encoding |
| [Lift3D](https://arxiv.org/abs/2411.18623) | 2D→3D | Fixed | DINOv2 + 3D lift |
| [RVT-2](https://robotic-view-transformer-2.github.io/) | 3D | Fixed multi-view | PerAct transformer |
| [BridgeVLA](https://arxiv.org/abs/2506.07961) | 3D | Fixed ortho | PaliGemma |
| **ActiveVLA** | 3D | **Active + zoom** | PaliGemma |

ActiveVLA 的不同点：**第一个**把 viewpoint selection 和 zoom-in 显式做成 active module 嵌进 VLA 的 paper（在 VLA 范式下）。在 active vision 经典领域有类似工作（[Next-Best-View](https://en.wikipedia.org/wiki/Next-best-view)），但和大规模 VLM 融合是新的。

---

## 9. 我对这篇 paper 的几条 takeaway

1. **Virtual renderer 是 VLA 的隐藏能力**：sim-trained VLA 因为可以 re-render 任意 viewpoint + 任意 FoV，本质上比 real-robot direct IL 多一个 "perception action space"。ActiveVLA 把这个 action space 显式 model 出来。

2. **Coarse-to-fine 是工程友好的**：如果不分两阶段，直接对 162 个 candidate viewpoints 都跑一次 PaliGemma，compute cost 不可接受。Coarse stage 用便宜的 ortho projection 找 ROI，再 fine stage 在小集合上做 active selection，是必要的工程 compromise。

3. **Active perception 的能力边界**：GemBench L4 的崩盘说明 active perception 解决的是 "sensing insufficient" 而非 "reasoning insufficient"。Long-horizon compositional reasoning 还是要靠 high-level policy（比如 LLM planner）。

4. **Discretize + vote > regress**：translation 用 grid voting、rotation 用 72-bin classification，这种 design choice 在 fine-grained 任务上比直接 regress 6-DoF 稳定得多。PerAct 系列已经验证过，ActiveVLA 延续了这个 tradition。

5. **冻结 vision encoder 的 importance**：active viewpoint 切换会破坏 VLM 的 embedding 稳定性。Frozen backbone + 只训 action head 是个简单但关键的 trick，建议未来 sim-to-real VLA 都考虑。

参考链接汇总：
- [PaliGemma](https://arxiv.org/abs/2407.07726)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [Gemma](https://arxiv.org/abs/2403.08295)
- [RLBench](https://arxiv.org/abs/1909.12271)
- [COLOSSEUM](https://arxiv.org/abs/2402.08191)
- [GemBench](https://arxiv.org/abs/2410.01345)
- [BridgeVLA](https://arxiv.org/abs/2506.07961)
- [RVT](https://robotic-view-transformer.github.io/)
- [RVT-2](https://robotic-view-transformer-2.github.io/)
- [PerAct](https://peract.github.io/)
- [Act3D](https://act3d.github.io/)
- [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)
- [RoboPoint](https://robotpoint.github.io/)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Active perception (Bajcsy 1988)](https://ieeexplore.ieee.org/document/5901)
- [Next-Best-View综述](https://www.mdpi.com/2218-6574/9/4/69)

如果你想看 video demos，project page 应该在 ZhenyangLiu.github.io/ActiveVLA。
