---
source_pdf: GAUSSGYM.pdf
paper_sha256: b7099e68dbef4ee0b3f88c2278ab051bb8f6a6e276ba9b2fd2870386c5e2cfba
processed_at: '2026-08-04T12:25:03-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GaussGym 用人话讲

## 一句话版

现在训 robot RL，大家都不用 RGB 当输入，只用 depth，因为 simulator 渲染 RGB 太慢而且不够真实。GaussGym 把 3D Gaussian Splatting 塞进 IsaacGym 当 renderer，一下子把"画面漂亮 + 速度快"两个看似矛盾的属性都拿到了——单卡 RTX 4090 跑 4096 个并行环境，每秒 10 万步，渲染 640×480 RGB。然后它还证明：RGB 确实比 depth 多了语义信息，policy 能学会"避开黄色地板"这种 depth 根本看不到的事。

---

## 为什么要做 —— 一个很尴尬的现状

你想想现在 legged robot RL 的标配 pipeline：ANYmal、Unitree A1、H1 这些，policy 的 observation 是什么？base angular velocity、joint positions、加上一个 depth map 或者 elevation map。**没有 RGB**。

为什么不用 RGB？不是不想，是技术栈不支持：

1. **IsaacLab 的 raytracing 太慢** — 768 envs 在 RTX 4090 上才 800 FPS，per-env 1 FPS。训 RL 要 10 亿步，这速度训到天荒地老。
2. **mesh asset 视觉质量差** — textured mesh 看起来像 PS2 画面，跟真实摄像头拍到的差太远，policy 学了也 transfer 不过去。
3. **场景太少** — 你能找到多少带真实 texture 的 URDF 场景？ReplicaCAD 那种规模撑死几十个。

所以大家就妥协了，用 depth。depth 至少 geometry 准，而且 raycast 快。但 depth 丢掉了所有**语义**：它知道前面有个东西挡着，但不知道那是水坑（要绕开）还是草地（可以踩）。

GaussGym 的立论就是：这个妥协没必要了，3DGS 给了我们"又快又好看"的渲染。

---

## 怎么做的 —— pipeline 走一遍

假设你拿 iPhone 扫了一段视频，想变成 robot 训练场景。

### Step 1: 视频进来，VGGT 处理

VGGT 是 2025 年 CVPR 的工作，一个 feed-forward transformer，吃一组 RGB 图像，吐出：
- 每张图的 camera pose（extrinsic + intrinsic）
- 整个场景的 dense point cloud
- 每个 point 的 normal

相当于把 COLMAP 那套 SfM pipeline 压到一个 network 里，秒级出结果，不用几十分钟。

### Step 2: 点云变成两样东西

**一样是 3DGS**：直接拿 VGGT 的点云当 gaussian 的初始位置，normal 决定 gaussian 椭球的朝向，然后优化一下 opacity 和 SH color coefficients。收敛快，因为初始化好。

**另一样是 mesh**：用 NKSR（Neural Kernel Surface Reconstruction）从点云+normal 重建一个干净 mesh，给 physics engine 当 collision geometry。

然后 splat 和 mesh 自动对齐到同一个世界坐标系。这就搞定了 LucidSim 之前要手动对齐的麻烦事。

### Step 3: 开训

IsaacGym 管 physics（contact、joint、gravity），3DGS 管 rendering（给每个 env 的 camera 出 RGB）。两者共享同一个 global frame，所以 robot 在 mesh 上走，camera 拍到的是 splat 渲染的画面，完美同步。

---

## 为什么 3DGS 这么快

这是整个 paper 的核心。让我对比一下：

**NeRF** 渲染一个 pixel：从 camera 发 ray，沿 ray 采 100 个点，每个点过 MLP 查 density+color，alpha composite。一个 640×480 图要 30 万 ray × 100 点 × MLP forward = 慢得要死。

**IsaacLab raytracing**：对每个 pixel，跟 mesh 求 intersection。mesh 复杂了也慢。

**3DGS**：完全不一样的思路。场景不是 implicit function，就是**一堆椭球**（几百万个），每个椭球有：
- 中心位置 μ
- 形状（covariance Σ，存成 rotation + scale）
- 颜色（spherical harmonics 编码，view-dependent）
- 不透明度 α

渲染时干嘛？把 3D 椭球投影到 2D 屏幕，变成 2D 椭圆，然后按深度排序，alpha blend 叠加。这跟 OpenGL 的 rasterization 思路一样，区别是可微。

**关键速度优势**：
- 没有 ray march，没有 per-pixel MLP query
- 就是投影 + 排序 + blend，GPU 天生擅长
- 可以用 CUDA tile-based rasterizer，跟传统图形管线一样高效
- 跨 env 向量化：4096 个 env 用不同 splat，同一套 kernel，GPU 打满

所以 per-env 能到 25 FPS，而 LucidSim 用 ControlNet diffusion 生成 RGB，单 env 才 3 FPS。

**额外福利**：depth 是渲染的副产品。因为每个 pixel 命中最近的 gaussian 时，那个 gaussian 的 camera-space z 就是 depth。RGB 和 depth 一起出，时间成本几乎一样。

---

## 关键 trick：Voxel prediction head

这是 paper 里我觉得最有 insight 的设计。

问题是这样的：你给 policy 一个 RGB 图，让它学走路。但 RL reward 是什么？"跟踪 commanded velocity""别摔倒""别绊倒"。这些 reward 太稀疏了——policy 完全可以 ignore RGB，只靠 proprioception 混过去。因为 RGB → "该怎么落脚" 这个映射太难学了，reward 信号又不够强。

Table 2 的 ablation 印证了这点："Vision w/o voxel" 在 tall stairs 上成功率从 94.4 掉到 80.8。

GaussGym 的解法：加一个 auxiliary head，让网络**同时预测前方场景的 voxel occupancy**。Ground truth 来自 NKSR mesh 的 voxelization。

网络结构长这样：

```
RGB → DinoV2 → features
                      ↓
proprioception → concat → LSTM encoder → latent z
                                          ↓                    ↓
                              Voxel Head (3D deconv)    Policy Head (LSTM)
                              预测 occupancy grid       输出 joint action
```

**为什么这管用**？因为要预测 voxel，latent z 就**必须**编码场景的 3D geometry。这个 representation 同时被 policy head 用，所以 policy 也受益。等于说用监督学习的方式，把 "RGB → geometry" 这个 hard inverse rendering 问题塞进 representation learning 里，不指望 RL reward 从零学会。

这个思路其实很通用：RL reward 信号弱的时候，加 auxiliary supervised task 来 shape representation。DeepMind 的 world model、Google 的 unsupervised representation learning for RL 都是这个 family。

---

## Motion blur 那个 trick

特别简单但特别合理。

真实摄像头在 robot 走路时会拍出 motion blur——因为曝光时间内 camera 在动。但 simulator 渲染的都是"瞬时清晰帧"，policy 训出来没见过模糊图像，一到真实就傻。

GaussGym 的做法：已知 camera velocity vector v 和 shutter speed T，在 [0, T] 区间取几个时间点，把 camera pose 沿 v 方向偏移，渲染几帧，alpha blend 成一帧。

$$
I_{\text{blur}} = \frac{1}{K} \sum_{k=0}^{K-1} I(\mathbf{T}_0 + \mathbf{v} \cdot t_k)
$$

尤其是在 stair climbing 时，脚一接触台阶，camera 突然顿挫，blur 就很明显。这个 domain randomization 思路非常对路。

---

## 实验里最有意思的点

### 1. Scene diversity 极其重要

用 1/10 的场景训，tall stairs 成功率从 94.4 掉到 67.3。用 1/2 掉到 83.9。这说明 2500 个场景不是噱头，是真有用。视觉泛化需要海量不同场景。

### 2. RGB > Depth 的语义实验（Figure 8）

设计了一个 navigation 任务：goal 在障碍物后面，地上有个黄色 patch，踩上去扣分。

- RGB policy：看到黄色，绕开了
- Depth policy：depth 看不出颜色，直接踩过去

这个实验设计得很巧妙，直接证明了 RGB 携带 depth 没有的语义信息。而且这种 task 在传统 depth-based simulator 里**根本无法定义**——你连"黄色 patch"这个 asset 都放不进去。

### 3. Sim-to-real 的初步证据

A1 上训的 stair climbing policy，zero-shot 迁移到真实 A1，能在真实楼梯上走。虽然 paper 自己承认精度下降了，但这是个 proof of concept。

---

## 几个我觉得值得多想的点

### 生成式 video model 当 scene source

这是 paper 最 forward-looking 的部分。Veo 能从 text prompt 生成多视角一致的视频，GaussGym 把这个视频 → VGGT → 3DGS → 训练场景。

这意味着什么？**你可以用 text prompt 无限生成训练环境**。"一个废弃的工厂走廊""火星表面""洪水后的街道"——这些真实世界根本扫不到的地方，video model 能生成，GaussGym 能用。

虽然 Veo 现在还不够稳定（paper 说有时要重 prompt），但这个方向太诱人了。相当于把 generative prior 转成 simulation asset，把 video model 从"慢推理的 simulator"变成"离线 scene generator"。

### LSTM 而不是 Transformer

policy 网络用 LSTM 融合 visual + proprio，而不是 transformer。原因很实在：robot onboard 推理要快，LSTM 单步 O(1)，transformer 要 attend over history。A1 上 Jetson Orin，DinoV2 ViT-S 已经吃掉 10ms，再加 transformer 就超 20ms budget 了。

但这也有代价——LSTM 的 memory capacity 有限，长时序推理可能不如 transformer。未来如果 onboard 算力上去（比如 H1 上的 Orin AGX），换 transformer 可能更好。

### Memory 问题 paper 没讲清楚

2500 scenes，每个 scene 几百万 gaussian，每个 gaussian ~60 bytes，总共可能几十 GB。4096 envs 是怎么 share 的？paper 说"128 unique scenes × 32 replication"，但 splat data 是 read-only，所以应该是 GPU 上 cache 128 份 splat，不同 env 用不同 camera pose 投影同一份 data。这点等 code release 要仔细看。

### Physical parameter uniform 的问题

所有 splat 对应的 mesh 都是同一个 friction coefficient。这意味着 policy 看到草地和看到冰面，物理表现一样——这显然不对。Chen et al. 2024a 的工作（https://ieeexplore.ieee.org/document/10604968）已经在做"从视觉预测 terrain physical parameter"，GaussGym 未来应该集成这类能力，让"看起来滑"的东西"真的滑"。

---

## 跟你（Andrej）可能关心的联系

你在多个场合讲过 "software 2.0"、neural network 作为可微程序的思想。3DGS 某种程度就是这个故事在 3D rendering 上的实例化——从 hardcoded rasterization pipeline（OpenGL）到 learned explicit representation（3DGS），保持可微和 GPU 友好。

GaussGym 把这个可微 renderer 接到 RL loop 里，等于在说：rendering 不再是 simulator 的瓶颈组件，而是 policy 的 rich observation source。这对 robot learning 的意义类似于"计算机视觉从 ImageNet classification 走向 dense prediction"——observation space 从 sparse geometric signal 走向 dense photorealistic signal。

另外，auxiliary voxel prediction head 的思路跟你在 Stanford 讲 cs231n 时强调的 "representation learning shapes what the network sees" 完全一致。RL reward 不够强时，监督信号来补，逼着 representation 编码对的东西。

---

## 一句话总结

GaussGym = 3DGS（快又好看的 renderer）+ IsaacGym（快 physics）+ VGGT（万能数据入口）+ voxel auxiliary head（逼 representation 学 geometry）+ video model 生成场景（无限数据源）。第一次让 visual locomotion RL 达到 depth-based 方法的训练规模，打开了一堆之前做不了的任务（语义导航、social norm walking 等）。

项目主页：https://escontrela.me/gauss_gym/

等 code 开了值得跑一遍，看看 3DGS 跨 env vectorized 的实现细节，那个 CUDA kernel 肯定有很多工程宝藏。

---

# GaussGym: Photorealistic Real-to-Sim Locomotion Learning

你好 Andrej，这篇 GaussGym paper 来自 UC Berkeley 的 Pieter Abbeel 组（Escontrela, Kerr, Allshire 等），核心想法是把 **3D Gaussian Splatting (3DGS)** 作为 drop-in renderer 嵌入到 vectorized physics simulators (IsaacGym/IsaacLab) 中，从而把 photorealistic RGB rendering 和 high-throughput contact physics 统一在一个训练 loop 里。我下面尽量讲细一点，build intuition。

---

## 1. 为什么需要这篇 paper —— Gap 在哪

经典的 sim-to-real locomotion pipeline（如 ANYmal, Unitree A1 上的工作）几乎全部依赖 **depth / elevation map / proprioception** 作为 policy observation。原因是：

1. **Visual sim-to-real gap** — 传统 simulators (IsaacLab raytracing, MuJoCo) 的 rendering 要么 photorealism 不够（textured mesh 的 shading artifacts），要么 throughput 太低（IsaacLab 在 RTX 4090 上 768 envs 只有 800 FPS vectorized, Table 1）。
2. **Asset scarcity** — 真实场景的 URDF/textured mesh 资产极少，procedural terrain 又缺乏语义信息。
3. **Training cost** — RL 需要 10⁶–10⁹ steps，per-env FPS 决定了能不能训完。

GaussGym 给出的 throughput 数字是 **100,000 steps/sec across 4,096 envs on a single RTX 4090**，per-env ~25 FPS。这个数量级的提升来自 3DGS 的 explicit, rasterization-friendly 表示——它绕开了 raytracing 的 per-pixel scene intersection，直接用 alpha-compositing 投影 gaussian ellipsoids 到屏幕空间。

参考链接：
- 3DGS original paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Isaac Gym: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/28dd2c7955ce926456240b2ff0100bde-Abstract-round2.html

---

## 2. 整体 Pipeline 架构解析 (Figure 2)

数据流分三段：

### (a) Data Ingestion
支持的数据源极其多样：
- **ARKitScenes** (https://openreview.net/forum?id=tjZjv_qh_CE) — iPhone RGB-D 室内扫描
- **GrandTour** (Frey et al. 2025) — 大规模 SLAM scans，含 20m² 级别场景
- **Smartphone casual scans** (带 intrinsic calibration)
- **Veo** (Google DeepMind video model, https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf) — 直接从 text prompt 生成多视角一致视频，再用作 scene source
- **Hand-held videos**

这种"video model → VGGT → 3DGS"的链路很有想象力，因为它把生成式 video model 当成 "scene prior" 而不是直接的 simulator——绕开了 video model 推理慢的问题。

### (b) VGGT 统一前处理
**VGGT (Visual Geometry Grounded Transformer)** (Wang et al. 2025, https://arxiv.org/abs/2503.00569 是 CVPR 2025) 是 Meta/ETH/UCL 的工作，给定一组 unposed RGB images，前向一次输出：
- camera intrinsics K (每个 view)
- camera extrinsics (R, t) 
- dense point cloud P ∈ ℝ^{N×3}
- per-point normals n ∈ ℝ^{N×3}

这是把传统 SfM/SLAM pipeline 压到一个 feed-forward transformer 里。好处是统一不同数据源格式，坏处是 VGGT 本身有 inaccuracy（点云噪声、depth scale ambiguity），后面 NKSR 来兜底。

### (c) 双输出
- **3DGS** 直接用 VGGT 的 point cloud 作为 gaussian 初始化（位置 + 法向决定 covariance 的旋转轴），大大加速收敛
- **NKSR (Neural Kernel Surface Reconstruction)** (Huang et al. 2023, https://arxiv.org/abs/2305.19240) 从 point cloud + normals 重建高质量 mesh，用作 collision geometry

两个 asset 自动对齐到同一个 gravity-aligned global frame，这就解决了 LucidSim 那种需要 manual mesh+splat registration 的痛点。

---

## 3. 3DGS 作为 Drop-in Renderer — 为什么快

### 3DGS 渲染方程回顾

每个 3D gaussian 写成：

$$
\mathcal{G}_i(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)\right)
$$

其中：
- $\boldsymbol{\mu}_i \in \mathbb{R}^3$ 是第 $i$ 个 gaussian 的中心位置 (mean)
- $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times3}$ 是 covariance matrix，物理上控制 gaussian 的椭球形状和朝向

为了保证 $\boldsymbol{\Sigma}_i$ 半正定（PSD），实际存储的是它的 Cholesky 分解 $\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^\top \mathbf{R}_i^\top$，其中 $\mathbf{R}_i$ 是 quaternion 表示的旋转，$\mathbf{S}_i = \text{diag}(s_x, s_y, s_z)$ 是 scale。

渲染时把 3D gaussian 投影到 2D 屏幕：

$$
\boldsymbol{\Sigma}'_i = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma}_i \mathbf{W}^\top \mathbf{J}^\top
$$

其中 $\mathbf{W}$ 是 view transformation (world→camera), $\mathbf{J}$ 是 projection 的 Jacobian (projective approximation)。$\boldsymbol{\Sigma}'_i \in \mathbb{R}^{2\times2}$ 是 2D 屏幕空间的 covariance。

最终 pixel color 通过 alpha compositing（front-to-back sorting）：

$$
C(\mathbf{u}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j<i}(1-\alpha_j)
$$

其中：
- $\mathbf{u}$ 是像素坐标
- $\mathbf{c}_i \in \mathbb{R}^3$ 是第 $i$ 个 gaussian 的 spherical harmonics-encoded view-dependent color
- $\alpha_i$ 是 opacity × 2D gaussian evaluated at $\mathbf{u}$
- $\prod_{j<i}(1-\alpha_j)$ 是 transmittance（前面 gaussian 还没吸收多少光的剩余比例）

**关键速度优势**：
1. **Explicit representation** — 没有 MLP query（对比 NeRF 每次 ray march 要 forward 一个 network 几百次），3DGS 就是 lookup + tile-based rasterization
2. **GPU-friendly** — 类似 OpenGL rasterizer，但可微，可以用 CUDA/PyTorch 写 multi-threaded kernels
3. **Vectorizable across envs** — 4,096 envs 用不同 splats 但同一 rendering codepath，GPU 充分利用

### Depth as a byproduct
Depth 直接从 gaussian splatting 的 z-sort 中得到（每个 pixel 命中的最近 gaussian 的 camera-space z），所以 **Figure 5 所示 RGB + Depth 同帧渲染时间几乎相同**。这对比 IsaacLab 的 depth raycast，是一个免费的好处。

参考：3DGS rendering details https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 4. High-Throughput 优化 (Section 3.3)

### Decoupling rendering from control
- Control frequency: 50 Hz（policy 输出 joint commands 的速率）
- Camera frequency: 10 Hz（实际 RGB 输入到 policy 的速率）
- Simulator physics step: 通常更高 (200 Hz+)

策略只用 10Hz 的 image，因为摄像头物理上就是这个速率；不需要每个 control step 都 render。这是把 "real robot perception latency" 当成 feature 而不是 bug 来建模。

### Motion Blur Simulation (Figure 10)
非常简单但聪明的 trick。给定：
- shutter speed $T_s$（曝光时间）
- camera linear velocity $\mathbf{v}_c$（在世界系）

在 $[0, T_s]$ 区间内取 $K$ 个等距时间点 $t_k = k \cdot T_s / K$，camera pose 偏移：

$$
\mathbf{T}_k = \mathbf{T}_0 + \mathbf{v}_c \cdot t_k
$$

每个 $\mathbf{T}_k$ render 一帧 RGB $\mathbf{I}_k$，最后 alpha blend：

$$
\mathbf{I}_{\text{blur}} = \frac{1}{K}\sum_{k=0}^{K-1} \mathbf{I}_k
$$

intuition：foot 接触 stair 时 camera velocity 突然 jerky，会模拟出方向性 blur streaks，这跟真实摄像头拍到的 motion blur 一致。Sim2Real gap 中一个常被忽视的点就是 motion blur 让网络见过的"清晰图像"分布 ≠ 真实 fast motion 下的模糊图像分布。

---

## 5. Neural Architecture (Figure 7) — 详解

整个 network 是 **asymmetric actor-critic + auxiliary task** 结构。

### 5.1 Observation space (Table 6)
- Base angular velocity $\boldsymbol{\omega}_b \in \mathbb{R}^3$
- Projected gravity angle $\alpha$（base 上 vector 在世界系投影的角度，给姿态信息）
- Joint positions $\mathbf{q} \in \mathbb{R}^{n_{\text{dof}}}$（A1 是 12 DOF, T1 humanoid 是更多）
- Joint velocities $\dot{\mathbf{q}}$
- Swing phase $\phi$（gait phase，[0,1] 循环量）
- Image $\mathbf{I} \in \mathbb{R}^{640 \times 480 \times 3}$

### 5.2 Vision encoder: DinoV2
RGB image 先过 **DinoV2** (https://arxiv.org/abs/2304.07193) — Meta 的 self-supervised ViT，输出 dense patch embeddings。DinoV2 的关键特性是它的 features 在 image patches 上 dense 且对应到 scene geometry，类似一个 implicit "what's there" map。

为什么 DinoV2 而不是 CLIP？DinoV2 是 self-distillation 训的，不接 language supervision，所以 features 更纯粹是 visual geometry/semantics，对 dense prediction 友好。CLIP features 偏 global image-text alignment，对 spatial resolution 不友好。Table 2 ablation "Vision w/o DINO" 显示掉了 ~5-10% success rate。

### 5.3 Fusion: LSTM encoder
$$
\mathbf{z}_t = \text{LSTM}_{\text{enc}}\left([\mathbf{o}_t^{\text{prop}}, \mathbf{e}_t^{\text{dino}}], \mathbf{h}_{t-1}\right)
$$

其中：
- $\mathbf{o}_t^{\text{prop}}$ 是 proprioception vector
- $\mathbf{e}_t^{\text{dino}}$ 是 DinoV2 embedding（flatten 或 pooled）
- $\mathbf{h}_{t-1}$ 是上一时刻 hidden state
- $\mathbf{z}_t$ 是 fused latent

为什么 LSTM 不用 Transformer？robot onboard inference 速度限制——LSTM 单步推理是 O(1) memory + 一次矩阵乘，而 vanilla transformer 要 attend over history。Mobile robot controller 通常 1kHz 推理。

### 5.4 Dual heads

**(a) Voxel prediction head (auxiliary)**
$\mathbf{z}_t$ unflatten 成 coarse 3D grid $(N_x, N_y, N_z)$，然后过 3D transposed convolution stack 上采样到 dense voxel grid $\hat{\mathbf{V}}_t \in \mathbb{R}^{H \times W \times D \times C}$，预测：
- occupancy probability per voxel
- terrain height per (x,y) cell

Ground truth 来自 NKSR mesh 的 voxelization。Loss 是 BCE on occupancy + L2 on heights。

**这是 paper 最关键的设计 choice**：强制 latent $\mathbf{z}_t$ 编码 3D geometry。没有这个 auxiliary task，policy 完全可能 ignore vision（Table 2 "Vision w/o voxel" 列：stairs tall 掉到 80.8 vs 94.4）。这个发现本质上是在说 "RL reward 信号太稀疏，无法逼着 CNN 从 RGB 学出 geometry 来；必须显式监督"。

类似思想在很多 visual RL 工作里都有 — 比如 DeepMind 的 world model 学习，或者 RL with representation learning auxiliary tasks。

**(b) Policy head**
第二个 LSTM consume $\mathbf{z}_t$ + 自己的 recurrent state：

$$
\mathbf{h}_t^{\pi}, \boldsymbol{\mu}_t, \boldsymbol{\sigma}_t = \text{LSTM}_{\pi}(\mathbf{z}_t, \mathbf{h}_{t-1}^{\pi})
$$

输出 Gaussian distribution over joint position offsets：

$$
\pi(\Delta\mathbf{q}_t | \mathbf{o}_{\le t}) = \mathcal{N}(\boldsymbol{\mu}_t, \text{diag}(\boldsymbol{\sigma}_t^2))
$$

实际 commanded joint position：

$$
\mathbf{q}_t^* = \mathbf{q}_{\text{default}} + \Delta\mathbf{q}_t
$$

注意是 PD controller target，不是直接 torque——这是 legged robot RL 的标准做法。

### 5.5 为什么 asymmetric actor-critic 而不是 student-teacher
ANYmal parkour (Hoeller et al. 2024, https://www.science.org/doi/abs/10.1126/scirobotics.adi7566) 用的是 privilege-info teacher → vision student 两阶段 distillation。GaussGym 主张直接 end-to-end 训练，因为：
- 一阶段省事
- Gaussian renderer 足够快，可以直接训 vision policy
- 避免 distillation 中的 mode collapse

参考 asymmetric actor-critic 原始 paper (Pinto et al.): https://arxiv.org/abs/1710.06542

---

## 6. Reward 设计详解 (Tables 3-5)

### 6.1 General rewards (Table 3)

| Reward | Expression | Weight | 变量解释 |
|---|---|---|---|
| Ang Vel XY | $\|\boldsymbol{\omega}_{xy}\|^2$ | -0.2 | $\boldsymbol{\omega}$ 是 base angular velocity, 下标 $xy$ 表示水平分量，惩罚摇晃 |
| Orientation | $\|\boldsymbol{\alpha}\|^2$ | -0.5 | $\boldsymbol{\alpha}$ 是 world up vector $\hat{\mathbf{z}}_w$ 与 policy base up vector 的夹角，惩罚翻倒 |
| Action Rate | $\|\mathbf{q}_t^* - \mathbf{q}_{t-1}^*\|^2$ | -1.0 | $\mathbf{q}_t^*$ 是 timestep $t$ commanded action，惩罚抖动 |
| Pose Deviation | $\|\mathbf{q}_t - \bar{\mathbf{q}}\|^2$ | -0.5 | $\bar{\mathbf{q}}$ 是 default standing pose，避免 strange posture |
| Feet Distance | $\mathbb{1}[(\mathbf{f}_{\text{left},xy} - \mathbf{f}_{\text{right},xy}) < 0.1]$ | -10.0 | $\mathbf{f}$ 是 foot position，避免双脚重合（self-collision） |
| Feet Phase | $\mathbb{1}_{f,\text{contact}} \cdot \mathbb{1}[\phi \le 0.25]$ | 5.0 | $\phi \in [0,1]$ 是 gait phase, 0.25 是 swing 中段；鼓励 contact 在 stance phase |
| Stumble | $\mathbb{1}[\|\mathbf{F}_{f,xy}\| \ge 2\|\mathbf{F}_{f,z}\|]$ | -3.0 | $\mathbf{F}_f$ 是 foot contact force, 横向力大于 2 倍垂直力意味着绊倒 |

### 6.2 Velocity tracking (Table 4)

$$
r_{\text{lin}} = \exp\left(-\frac{\|\mathbf{v}_{xy} - \mathbf{v}_{xy}^*\|^2}{0.25}\right), \quad r_{\text{ang}} = \exp\left(-\frac{\|\omega_z - \omega_z^*\|^2}{0.25}\right)
$$

- $\mathbf{v}_{xy}, \mathbf{v}_{xy}^*$: current 和 desired base linear velocity
- $\omega_z, \omega_z^*$: current 和 desired yaw rate
- 0.25 是 bandwidth constant，控制 reward 衰减 sharpness
- 用 exp 而不是 L2 是为了让 small error 时 reward 接近 1（饱和），避免 policy 在已经很准时还过度优化

### 6.3 Goal tracking (Table 5)

$$
r_{\text{pos}} = \mathbb{1}_{t<1}\left(1 - 0.5\|\mathbf{r}_{xy} - \mathbf{r}_{xy}^*\|\right), \quad r_{\text{yaw}} = \mathbb{1}_{t<1}\left(1 - 0.5\|\psi - \psi^*\|\right)
$$

- $\mathbb{1}_{t<1}$: indicator that remaining time $t < 1$ 秒（goal 临近时才激活高权重 tracking）
- $\mathbf{r}, \mathbf{r}^*$: current 和 desired base position
- $\psi, \psi^*$: current 和 desired yaw
- 参考 ANYmal parkour reward 设计

---

## 7. 实验结果分析 (Table 2)

主实验在 4 个 scenario × 2 个 robot (A1 quadruped, T1 humanoid from Booster) × 6 个 ablation。

### 关键数字

| Scenario | Vision (full) | Blind | Vision w/o voxel | Vision w/o DINO | Vision 1/10 scenes | Vision 1/2 scenes |
|---|---|---|---|---|---|---|
| Flat | 100.0 / 100.0 | 98.1 / 97.2 | 100.0 / 98.3 | 100 / 96.7 | 94.3 / 99.2 | 99.0 / 99.2 |
| Steep | 99.3 / 97.1 | 89.4 / 87.6 | 91.9 / 87.0 | 95.6 / 91.5 | 88.1 / 88.3 | 95.5 / 94.1 |
| Stairs (short) | 98.7 / 97.4 | 80.8 / 72.3 | 85.2 / 82.7 | 92.3 / 87.5 | 79.7 / 74.8 | 86.3 / 84.9 |
| Stairs (tall) | 94.4 / 92.5 | 74.0 / 60.5 | 80.8 / 76.3 | 88.3 / 82.8 | 67.3 / 58.2 | 83.9 / 75.2 |

(每列 A1 / T1)

### Intuition 提取

1. **Flat 上 blind 也接近 100%** — 因为平地不需要 vision，proprioception 足够。
2. **Stairs (tall) 上 vision vs blind 差 20-30%** — 高台阶必须看准 foot placement。
3. **Voxel auxiliary loss 贡献最大**（tall stairs: 94.4 → 80.8 without）— 印证了 "RGB → geometry 是 ill-posed，需要显式监督"。
4. **Scene diversity 极其重要** — 1/10 scenes 比 full 掉 ~27% on tall stairs。这是 GaussGym 2,500 scenes 的核心卖点。
5. **DINO pretraining 也关键** — 没有 DINO 的 from-scratch CNN 学不出 good features。

### Semantic reasoning experiment (Figure 8)
最有说服力的实验：goal-reaching + yellow floor patch penalty。
- **RGB policy** 绕开 yellow patch
- **Depth-only policy** 直接穿过

这直接证明了 RGB 携带 depth 没有的 semantic information（颜色、texture），policy 可以学会 "yellow = bad"。这种 task 在传统 depth-based simulators 里根本无法定义。

---

## 8. Limitations — Paper 自己承认的

1. **Sim2Real gap 仍存在** — 真实部署时 foot placement precision 下降
2. **No semantic reward automation** — yellow patch 是手工标的；未来需要 LLM/VLM 自动从 scene 生成 cost function（参考 Langsplat https://arxiv.org/abs/2312.16084, URDformer https://arxiv.org/abs/2405.11656）
3. **Uniform physical parameters** — 所有 splat 都是同一 friction，不能模拟 ice/mud/sand 的差异（参考 Chen et al. 2024a https://ieeexplore.ieee.org/document/10604968 的工作）
4. **Veo inconsistent** — 需要重 prompt
5. **Static scenes only** — 没有动态物体、流体、deformable

---

## 9. 我的 Intuition / 扩展思考

### 9.1 为什么这套方案能 work
GaussGym 本质上是把 "scene representation learning" 和 "policy learning" 解耦，但用同一个 simulator bridge 它们。3DGS 提供了：
- **photorealistic observation** for policy training
- **geometric ground truth** (via NKSR mesh) for auxiliary supervision
- **fast rendering** for vectorized RL

这三个属性同时满足才能做大规模 visual RL。NeRF2Real (https://ieeexplore.ieee.org/document/10161544) 之前卡在第一个+第三个——NeRF 太慢。

### 9.2 与 LucidSim 的对比 (Table 1)
LucidSim (https://arxiv.org/abs/2411.17033 是 Yu et al. CoRL 2024) 用 ControlNet 从 depth+semantic mask 生成 RGB，单 env rendering 3 FPS。GaussGym 用 3DGS 直接 rasterize，单 env 25 FPS。但 LucidSim 的好处是 can generate arbitrary scenes on-the-fly via diffusion，GaussGym 必须预先 reconstruct。

未来可能：用 generative video model (Veo, Genie 3) 离线生成海量 scenes → VGGT → 3DGS → GaussGym 训练，把 generative prior 转成 simulation asset。这正是 paper Section 4.1 暗示的方向。

### 9.3 Auxiliary task 的更深含义
Voxel prediction head 让我想到 **perception-as-prediction** paradigm。本质上是在说：好的 visual latent 必须 "renderable" 回 3D。这跟 world model 思路（Dreamer, Genie）有亲缘关系，但更轻量——只 predict 静态 geometry，不 predict dynamics。

如果再加一个 head predict next-frame RGB（给定 action），就接近 world model 了。这可能是 GaussGym 的自然 extension——把 NKSR mesh + 3DGS 渲染换成 learned world model，从而支持 deformable/dynamic scenes。

参考 Dreamer V3: https://arxiv.org/abs/2301.04104
参考 Genie: https://arxiv.org/abs/2401.02924

### 9.4 Cross-embodiment 想象
Paper 测了 A1 (quadruped) 和 T1 (humanoid)，但同一个 GaussGym framework 可以扩展到 wheeled robot, drone (https://arxiv.org/abs/2503.03984 的 GradNav 已经做了 drone)。3DGS rendering 跟 embodiment 无关，只跟 camera pose 有关。这就允许 cross-embodiment training，policy 在不同 robot 上 share visual encoder。

### 9.5 为什么 VGGT 而不是其他 SfM
VGGT (https://vgg-t.github.io/) 是 2025 年初的工作，相比 COLMAP (https://colmap.github.io/) 的优势：
- 前向一次（秒级），COLMAP 要分钟到小时
- 不需要 matching 等启发式步骤
- 对 casual video 鲁棒（handheld, low texture）
- 输出 normals（用于 NKSR）

但 VGGT 在 large-scale scene (>100 images) 上可能不稳定，这是为什么 GrandTour 这种 SLAM-captured 数据可能还需要 hybrid 处理。

### 9.6 Memory footprint 的隐忧
2,500 scenes × 3DGS 每 scene 几十万到几百万 gaussians × 每个 gaussian ~60 bytes (position + rotation quaternion + scale + opacity + SH coefficients) = 几十 GB GPU memory。Vectorized across 4,096 envs 意味着每个 env 实例化一份 splat？还是共享？Paper 没说清楚。可能的实现：
- 4,096 envs 用 128 unique scenes（Section 3.3 提到），所以 32× env replication per scene
- Splats 是 read-only during simulation，所以可以在 GPU 上 cached + 不同 env 用不同 camera pose 投影同一份 splat data
- 这跟 IsaacGym 的 URDF instancing 类似

### 9.7 跟近期 3DGS robotics 工作的关系
- **VR-Robo** (Zhu et al. 2025, https://arxiv.org/abs/2503.03984 类似): real-to-sim-to-real for navigation，没有 contact physics
- **HAMMER** (https://arxiv.org/abs/2501.14147): multi-robot semantic gaussian splatting
- **LucidSim** (https://arxiv.org/abs/2411.17033): 最接近，但 single-env rendering only
- **NeRF2Real** (https://ieeexplore.ieee.org/document/10161544): bipedal motion 用 NeRF

GaussGym 的差异化是 **vectorized rendering + 自动 mesh/splat alignment + generative video input**。

### 9.8 公式补遗：3DGS forward 的可微性
为了 RL training 中可能需要 gradient through renderer（比如 differentiable simulation 或 learned rendering），3DGS 的所有步骤都可微：

$$
\frac{\partial C(\mathbf{u})}{\partial \boldsymbol{\theta}_i} = \frac{\partial C}{\partial \alpha_i} \cdot \frac{\partial \alpha_i}{\partial \boldsymbol{\theta}_i}
$$

其中 $\boldsymbol{\theta}_i = \{\boldsymbol{\mu}_i, \mathbf{q}_i, \mathbf{s}_i, \alpha_i, \mathbf{c}_i\}$ 是第 $i$ 个 gaussian 的所有参数。但 GaussGym 里 splat 是 fixed asset（不训练），只 train policy network，所以 renderer 只需要 forward，不需要 backward。这又是速度优势的来源——可以 disable autograd。

### 9.9 实际部署 latency budget
A1 control 50 Hz = 20ms per cycle。LSTM + DinoV2 ViT-S 推理在 Jetson Orin 上大约：
- DinoV2 ViT-S forward: ~10ms @ 640×480
- LSTM forward: ~1ms
- Total: ~11ms，留 9ms margin

这是可行的，但 DinoV2 ViT-B 就不行了。可能需要 distill 到 smaller encoder。

### 9.10 Future work 可能方向
1. **Deformable 3DGS** (4D Gaussians, https://arxiv.org/abs/2305.12788 类) — 支持可变形物体
2. **VLM-guided reward** — GPT-4V 自动从 scene screenshot 生成 cost function
3. **Closed-loop world model** — 把 3DGS 渲染 + action-conditioned next-frame prediction 合并
4. **Multi-agent** — 当前 4096 envs 是独立 robot，未来可能 multi-robot coordination in shared photorealistic scene
5. **Tactile/force sensing integration** — NKSR mesh 已经给了 collision geometry，可以扩展到 contact-rich manipulation

---

## 10. 总结一句话

GaussGym 的核心 insight 是：**3DGS explicit representation 既给了 photorealism 又给了 vectorizability**，使得 visual RL on legged robots 第一次达到 depth-based simulators 的 throughput 量级。Auxiliary voxel prediction head 是关键 trick，解决了 "RL reward 太稀疏无法逼出 geometry-aware visual representation" 的问题。Generative video model → 3DGS 链路开启了 "infinite scenes for training" 的可能。

项目主页：https://escontrela.me/gauss_gym/
代码（承诺 open-source）：同上

---

如果你（Andrej）想再深挖某个点，比如：
- 3DGS rasterizer 的 CUDA kernel 实现细节
- VGGT 的 transformer 架构
- asymmetric actor-critic 的具体 loss formulation
- NKSR 的 neural kernel 数学

我可以展开讲。这篇 paper 信息密度很高，但很多工程细节（比如 splat memory management across 4096 envs）藏在 code 里没写出来，等 code release 后会值得深读。
