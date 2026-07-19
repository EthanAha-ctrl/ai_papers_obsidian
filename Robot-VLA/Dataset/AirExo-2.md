---
source_pdf: AirExo-2.pdf
paper_sha256: f17048aa3b65c5eff586ad0358b7eb5c84705366294f6562678d2e9d7b1ed248
processed_at: '2026-07-18T07:09:28-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AirExo-2: Scalable In-the-Wild Demonstration Collection + Generalizable Policy 深度解析

Andrej，这篇paper本质上是想解决一个 scaling 问题：imitation learning 的 bottleneck 在 data collection。Teleoperation 太贵（一台 robot platform 约 $60k），而且 operator learning curve 陡峭。作者团队（SJTU 的 Cewu Lu 组，之前做过 AnyGrasp、GraspNet-1Billion、RH20T 等）的思路是：用 $600 的 exoskeleton 在野外采集 human motion，再通过一套 visual adaptor pipeline 把它"翻译"成 pseudo-robot demonstrations，最后训练一个 view-invariant 的 RISE-2 policy，zero-shot 部署到真实 robot 上。

项目主页：http://airexo.tech/airexo2

---

## 1. Motivation：为什么 prior in-the-wild 方案不够

In-the-wild demonstration 设备分两派：

| 类别 | 代表工作 | 核心问题 |
|---|---|---|
| Handheld | UMI (Chi et al. 2024), AnyTeleop, Robot Utility Models | 依赖 visual SLAM 做 pose estimation，action 误差大（paper 测出 8.86mm）；in-hand camera FOV 受限，depth perception 差 |
| Whole-body | AirExo (Fang et al. 2024), DexCap (Wang et al. 2024), ARCap | 需要 robot data fine-tuning 来弥合 domain gap |

UMI paper: https://universal-manipulation-interface.github.io/
DexCap: https://dexcap.github.io/
ARCap: https://arcap-2024.github.io/

AirExo-2 同时解决这两个问题：(1) 机械结构 + forward kinematics 取代 SLAM，把 action error 降到 1.74mm；(2) visual adaptor 把 in-the-wild image 显式 transform 成 pseudo-robot image，让 policy 直接 train 不需要 fine-tuning。

---

## 2. AirExo-2 Hardware：从 PLA 到 Aluminum Profile

原版 AirExo 有五个 hardware 问题（H1-H5）：
- H1：PLA 3D-printed 部件 rigidity 差，易变形
- H2：joint 能转 360°+，超出 robot 可达范围
- H3：操作时身体移动导致 base 漂移
- H4：gripper 卡顿
- H5：encoder 直连 shaft，线缆侧出易磨损

AirExo-2 的改进：

```
- 20×20 欧标 aluminum profile 替换 3D-printed links
- PLA-CF（碳纤维增强 PLA）做 joint 外壳
- 更大 bearings
- Hollow rotating disc + 侧装 encoder（齿轮传动，线缆穿中空）
- Joint angle limit + 可调 friction pad（grooved track + screw 压紧）
- Linear guide gripper（参考 UMI、ForceMimic 的设计）
- Mobile aluminum stand（解决 H3，base 稳定）
```

成本分解：dual-arm platform（不含 camera）= $600。相比 $60k teleoperation platform，是 100× cost reduction。这个数字非常 key，因为整个 scaling argument 建立在这个 cost ratio 上。

**Intuition**：1:1 scale + kinematic isomorphism 意味着 human joint angle 直接等于 robot joint angle（经过 zero-position offset 校准后）。这就避开了 inverse kinematics 求解，也避开了 SLAM drift。机械精度天然比视觉 SLAM 高一个数量级，这是 action accuracy 的根本来源。

---

## 3. Calibration：Two-Stage Differentiable Rendering

这是 paper 里我觉得技术上最 elegant 的部分之一。

### Stage 1: Initial Calibration
- Joint zero position：用 3D-printed 工具对齐，读 encoder 值得到 $\{\tilde{q}_{\text{calib}}^{\text{left}}, \tilde{q}_{\text{calib}}^{\text{right}}\}$
- Camera-base transform：ArUco marker board 光学定位

$$[\tilde{\mathbf{t}}_{\text{base}}^{\text{camera}} | \tilde{\mathbf{r}}_{\text{base}}^{\text{camera}}] \stackrel{\text{def}}{=} \tilde{\mathbf{T}}_{\text{base}}^{\text{camera}} = \mathbf{T}_{\text{marker}}^{\text{camera}} (\tilde{\mathbf{T}}_{\text{marker}}^{\text{base}})^{-1}$$

变量解释：
- $\tilde{\mathbf{t}}_{\text{base}}^{\text{camera}}$：base 在 camera 坐标系下的 translation（3D vector），上标"camera"表示参考系，下标"base"表示被描述的物体
- $\tilde{\mathbf{r}}_{\text{base}}^{\text{camera}}$：base 在 camera 坐标系下的 rotation（6D format）
- $\mathbf{T}_{\text{marker}}^{\text{camera}}$：marker 在 camera 下的 pose（由 detection 得到）
- $\tilde{\mathbf{T}}_{\text{marker}}^{\text{base}}$：marker 相对 base 的已知 pose（marker 板物理固定在 base 上）

### Stage 2: Differentiable Rendering Refinement

定义待优化参数：

$$p \stackrel{\text{def}}{=} \{\Delta\mathbf{t}_{\text{base}}^{\text{camera}}, \mathbf{r}_{\text{base}}^{\text{camera}}, \Delta q_{\text{calib}}^{\text{left}}, \Delta q_{\text{calib}}^{\text{right}}\}$$

- $\Delta\mathbf{t}$：translation 的 delta 修正（3D）
- $\mathbf{r}$：rotation 用 6D 表示（参考 Zhou et al. 2019，"On the Continuity of Rotation Representations in Neural Networks"），因为 Euler/quaternion 在网络学习中不连续
- $\Delta q$：joint zero offset 的 delta（每臂一个 vector）

初始值 $p_0 = \{\mathbf{0}, \tilde{r}_{\text{base}}^{\text{camera}}, \mathbf{0}, \mathbf{0}\}$，即 rotation 用 stage 1 结果，其他 delta 从 0 开始。

Loss function（公式 8）：

$$\mathcal{L}(p) = \frac{1}{N_c} \sum_{i=1}^{N_c} \left(\beta \cdot \|M_i^a - \hat{M}_i^a\|^2 + \|d_i - \hat{d}_i\|^2 \circ M_i^d\right)$$

- $N_c = 40$：校准用样本数（来自一条 human play trajectory）
- $M_i^a$：SAM-2 标注的 AirExo-2 mask（pseudo-ground-truth）
- $\hat{M}_i^a = \mathcal{R}(p; q_i^{\text{left}}, q_i^{\text{right}}, \tilde{\mathbf{T}}, \tilde{q})$：可微渲染引擎 Redner 渲染出的 mask
- $d_i$：camera 观测的 depth
- $\hat{d}_i$：渲染 depth
- $M_i^d \subseteq M_i^a$：depth valid mask（排除噪声大的 depth 区域）
- $\beta = 5$：mask loss 权重
- $\circ$：mask-apply（element-wise multiply）

**Intuition**：这个 calibration 本质是 analysis-by-synthesis。给定 joint angles，渲染出 AirExo-2 应该长什么样，和真实观测对齐。误差通过 differentiable rendering 反传到 calibration 参数。由于 exoskeleton 是 chain structure，单 joint 误差会累积到 end-effector 放大，所以必须 joint-level refine。Redner 是 Li et al. 2018 提出的 edge-sampling differentiable ray tracer，参考：https://www.cs.cornell.edu/projects/diff-render/

Calibration 结果（Table 5）：

| Method | Mask Diff (%) | Depth Err (mm) |
|---|---|---|
| Initial Calibration | 1.71±0.37 | 21.6±5.2 |
| Human Annotation | 2.31±0.31 | 31.2±6.4 |
| Two-Stage (mask only) | 1.10±0.26 | 17.6±4.1 |
| **Two-Stage (mask+depth)** | **0.78±0.25** | **14.0±2.9** |

有意思的是 Human Annotation 比 Initial Calibration 还差——因为人靠 2D visual cue 调，depth perception 不行。这印证了 3D 不同模态 supervision 的必要性。

---

## 4. Visual Adaptors：把 In-the-Wild 翻译成 Pseudo-Robot

这是整个 pipeline 最"重"的部分，分三个 adaptor。

### 4.1 Operation Space Adaptor
所有 states 和 actions 投影到 global camera coordinate。这借鉴了 RISE (Wang et al. 2024) 和 iDP3 (Ze et al. 2024) 的思路——camera frame 是 AirExo-2 和 robot 之间的 universal coordinate frame，避免 base frame 不一致问题。

RISE paper: https://arxiv.org/abs/2406.14531
iDP3: https://humanoid-ai.github.io/

### 4.2 Image Adaptor
流程：
1. **Render AirExo-2 mask**：用 joint encoder readings + calibration + Open3D 渲染出当前 AirExo-2 的 RGB-D + mask
2. **Hand mask**：SAM-2 分割人手（如果可见还有 head）
3. **Inpainting**：ProPainter 把 human-related regions（手 + AirExo-2 本身）填补成背景
4. **Render robot image**：用 joint mapping 在 Open3D 中渲染出对应 robot pose 的 RGB-D
5. **ControlNet refinement**：Stable Diffusion 1.5 + ControlNet 把渲染的"假"robot image 变成 photo-realistic
6. **Composition**：refined robot image 叠加到 inpainted background 上

ControlNet 训练细节很关键：
- Training data：teleoperation 收集少量 robot 在 empty workspace 随机运动的数据（RGB-D + joint states）
- **Platform-specific but task-invariant**：一次训练，所有 task 通用
- Batch size 88，lr 1e-5
- 50 DDPM steps，guidance scale 9.0
- Prompt: "robotic arms, dual arm, industrial robotic manipulator, metallic silver color, mechanical joints, precise mechanical details, gripper end effector, high-quality photo, photorealistic, clear and sharp details"

ControlNet paper: https://arxiv.org/abs/2302.05543
Stable Diffusion: https://arxiv.org/abs/2112.10752
ProPainter: https://github.com/sczhou/ProPainter
SAM-2: https://sam2.metademolab.com/

**Intuition**：这一步本质是 image-to-image translation，把"人 + exoskeleton"的 visual domain 转到"纯 robot"的 visual domain。关键是 platform-specific but task-invariant——意味着收集一次 ControlNet 训练数据（在空 workspace 上 robot 随机动几下），就能 transform 任意 task 的 in-the-wild data。这是 data reusability 的关键。

Semi-automatic SAM-2：先手动标 ~50 scenes，fine-tune SAM-2，之后自动化。这让整个 adaptation pipeline 几乎 fully automated。

### 4.3 Depth Adaptor
1. Capture reference depth of empty workspace
2. Combine with static object depth from first frame → reference depth per demonstration
3. Inpaint human regions（用 image adaptor 的 combined mask）with reference depth
4. Merge inpainted depth + rendered robot depth

**Intuition**：depth 不能像 RGB 那样让 diffusion 生成（diffusion 不保证 metric accuracy），所以用 reference depth + rendered robot depth 这种几何方法。这保留了 spatial consistency，对 3D policy 很重要。

计算开销：1 分钟场景（~600 frames）渲染需 40 秒/single GPU，主要瓶颈在 ControlNet。可以多 GPU 并行。作者强调 human effort 是真瓶颈，computation 可 scale。

---

## 5. RISE-2 Policy：3D Geometric + 2D Semantic 的 Spatial Alignment

这是 paper 的另一大贡献，也是我觉得设计上最 thoughtful 的部分。

### 5.1 整体架构（Figure 3）

```
RGB-D Input
   ├──> Dense Encoder (DINOv2 + LoRA) ──> F_s (2D semantic, h×w) + C_s (3D coords)
   │
   └──> Sparse Encoder (MinkowskiEngine) ──> F_g (3D geometric) + C_g (seed points)
                                                          │
                                     Spatial Aligner (weighted interpolation)
                                                          │
                                                  Fused features
                                                          │
                                  Sparse Conv (MinkowskiEngine)
                                                          │
                                  Transformer (decoder-only, 4 blocks)
                                                          │
                                              Diffusion Head (CNN, 100 train / 20 infer steps)
                                                          │
                                              Action chunk (horizon=20)
```

### 5.2 Sparse Encoder: Pure 3D Geometric
$$\text{E}_s: (D, K) \to (\mathbf{F}_g, \mathbf{C}_g)$$

- $D$：depth image
- $K$：camera intrinsics
- $\mathbf{F}_g$：sparse geometric features（每个 seed point 一个 feature vector）
- $\mathbf{C}_g = \{c_g^i \in \mathbf{P}\}$：seed points 的 3D 坐标，$\mathbf{P}$ 是 down-sampled point cloud

**关键设计 choice**：相比原版 RISE，**移除了 color input**，只用 xyz。为什么？因为 color 信号和 coordinate 信号在 sparse conv 里会 entangle。3D geometric feature 应该只编码 shape/spatial structure，semantic 让 2D encoder 负责。这种 disentanglement 让 fusion 更 clean。

架构：ResNet-like，基于 MinkowskiEngine（Minkowski Convolutional Neural Networks, Choy et al. 2019）。MinkowskiEngine: https://github.com/NVIDIA/MinkowskiEngine

具体 layers（Table 6）：
- init_conv: kernel [3,3,3], channels 32, dilation 1, stride 1, + 2× mean pooling
- conv1: k[3,3,3], c=32, d=1, s=1
- conv2: k[3,3,3], c=64, d=2, s=1
- conv3: k[3,3,3], c=128, d=4, s=1
- conv4: k[3,3,3], c=128, d=8, s=2
- final_conv: k[1,1,1], c=128, d=1, s=1

Voxel size 5mm。

### 5.3 Dense Encoder: 2D Semantic via DINOv2
$$\text{E}_d: (I, D, K) \to (\mathbf{F}_s, \mathbf{C}_s)$$

- $I$：color image
- $\mathbf{F}_s = \{f_s^i\}$：semantic feature map，shape $h \times w$（DINOv2-base 是 32×18）
- $\mathbf{C}_s = \{c_s^i\}$：对应的 3D coordinates，由 depth $D$ + intrinsics $K$ 反投影得到，再经过 2D adaptive average pooling 到 $h \times w$

DINOv2 (Oquab et al. 2024) 是 Meta 的 self-supervised vision foundation model，在 LVD-142M 上训练。DINOv2: https://dinov2.metademolab.com/

用 LoRA (Hu et al. 2022) fine-tune，rank 没明说但通常 r=8 或 16。LoRA: https://arxiv.org/abs/2106.09685

**Intuition**：point cloud 的 texture 质量差（depth sensor noise + sparse），直接从 point cloud 学 semantic 不如从 2D image 学。DINOv2 在大规模无监督数据上 pretrain，semantic representation 已经非常 generalizable，正好解决 in-the-wild → robot 的 domain gap。

### 5.4 Spatial Aligner：核心创新
这是 2D-3D fusion 的关键。问题是：sparse encoder 输出 $N_g$ 个 seed points（几千个），dense encoder 输出 $h \times w = 576$ 个 feature vectors。怎么 fuse？

**Naive 方案 1**：global average pool 各自，concatenate。问题：丢失 fine-grained local info。

**Naive 方案 2**（3D Policies 风格，如 Enerdex-Wang 等）：upsample $\mathbf{F}_s$ 到 image resolution，project 到 point cloud，downsample 到 seed points。问题：compute expensive。

**RISE-2 方案**：对每个 seed point $c_g^i$，找它在 $\mathbf{C}_s$ 中的 M=3 nearest neighbors $\mathbf{N}_i = \{n_1^i, n_2^i, n_3^i\}$，做 inverse-distance weighted interpolation：

$$f_{s*}^i = \frac{\sum_{j=1}^M f_s^j / \text{dist}(c_g^i, n_j^i)}{\sum_{j=1}^M 1 / \text{dist}(c_g^i, n_j^i)}$$

变量：
- $f_{s*}^i$：seed point $c_g^i$ 的 aligned semantic feature
- $f_s^j$：dense encoder 输出的第 $j$ 个 neighbor 的 semantic feature
- $c_g^i$：sparse encoder 输出的第 $i$ 个 seed point 的 3D 坐标
- $n_j^i \in \mathbf{C}_s$：dense encoder 输出中 $c_g^i$ 的第 $j$ 近的 neighbor
- $\text{dist}(c_i, c_j)$：Euclidean distance
- $M = 3$：最近邻数量

然后 fused feature：$f_i = \text{Concat}(f_g^i, f_{s*}^i)$，再过 sparse conv 聚合。

**Intuition**：这本质是 3D 空间里的 bilinear interpolation，只不过作用在 feature space。key insight 是——dense encoder 的 feature map 虽然 low-res（32×18），但每个 pixel 都有 3D coordinate，所以可以在 3D space 里做 nearest neighbor query。这样：
1. 不需要 upsampling（compute cheap）
2. 不需要 2D pixel-space alignment（处理了 camera viewpoint 变化）
3. 保留了 fine-grained local info（每个 seed point 都有自己的 semantic feature）

Figure 10 的可视化很漂亮——把 sparse semantic feature PCA 降维到 RGB 显示，能看到不同 object 有明显不同的颜色，target object 在 task progression 中 feature 变化显著。这说明 DINOv2 feature 确实 encode 了 task-relevant semantic。

### 5.5 Action Generator
- Decoder-only Transformer（Vaswani et al. 2017），4 blocks
- $d_{\text{model}} = 512$，$d_{\text{ff}} = 2048$
- Readout token channel 512
- Positional encoding：sparse（来自 RISE）
- Diffusion head：CNN implementation（Diffusion Policy 风格），100 denoising steps train / 20 infer
- Action horizon = 20
- Action representation：camera frame，translation absolute，rotation 6D format（Zhou et al. 2019）

Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
6D rotation: https://arxiv.org/abs/1812.07035

**Intuition**：action chunk + diffusion 解决了 multi-modality 问题（同一个 observation 可能有多个合理 action）。Transformer 的 readout token 把 variable-length point tokens 聚合成 fixed-length latent，condition diffusion head。

---

## 6. 实验结果深度解析

### 6.1 In-Domain Evaluation（Figure 4）
4 个 tasks：Collect Toys, Lift Plate, Open Lid, Pour Water

| Method | Avg Success Rate |
|---|---|
| ACT | 32.5% |
| Diffusion Policy | 40.0% |
| CAGE | 65.0% |
| RISE | 72.5% |
| π0 | 85.0% |
| RISE-2 (ResNet-18) | 77.5% |
| **RISE-2 (DINOv2)** | **95.0%** |

**Observations**：
- RISE → RISE-2 (ResNet-18)：72.5% → 77.5%。说明分离 2D/3D encoder + spatial alignment 本身有提升
- RISE-2 (ResNet-18) → RISE-2 (DINOv2)：77.5% → 95.0%。说明 foundation model 的 semantic prior 巨大
- RISE-2 (DINOv2) > π0：95% vs 85%。π0 是 VLA flow model，pretrain 了大规模 robot data，但在这个 narrow domain 上不如 RISE-2。这有点 surprising——可能是因为 π0 的 pretraining data 和这些 specific tasks 不够 aligned，而 RISE-2 的 3D perception 对 manipulation 更直接有效。

ACT paper: https://tonyzhaozh.github.io/aloha/
π0: https://www.physicalintelligence.company/blog/pi0

### 6.2 Generalization Evaluation（Table 1）
Collect Toys task， disturbances：novel backgrounds (5种) + novel objects (1种)

| Method | In-Domain | Bg. | Obj. | Both |
|---|---|---|---|---|
| ACT | 32.5% | 0.0% | 2.5% | 0.0% |
| Diffusion Policy | 40.0% | 12.5% | 5.0% | 2.5% |
| CAGE | 65.0% | 45.0% | 42.5% | 10.0% |
| RISE | 72.5% | 42.5% | 12.5% | 40.0% |
| π0 | 85.0% | 77.5% | 82.5% | 82.5% |
| RISE-2 (ResNet-18) | 77.5% | 47.5% | 47.5% | 37.5% |
| **RISE-2 (DINOv2)** | 95.0% | **85.0%** | **85.0%** | 62.5% |

**Interesting findings**：
- π0 在 "Both" disturbance 下表现最好（82.5%），说明大规模 pretraining 的 compositional generalization 强
- RISE-2 (DINOv2) 在 single-axis disturbance 下和 π0 持平或更好，但 combined disturbance 下 drop 更大（95→62.5%）
- CAGE 是 prior SOTA generalizable policy，被 RISE-2 显著超越

CAGE paper: https://arxiv.org/abs/2410.14944

### 6.3 AirExo-2 Pseudo-Robot Demonstrations（Figure 5, Table 7）
同等数据量（50 demos/task），比较 teleoperation vs AirExo-2：

| Data Source | Policy | Avg Success Rate |
|---|---|---|
| Teleoperation | RISE | 74.375% |
| Teleoperation | RISE-2 | 93.750% |
| AirExo-2 | RISE | 60.625% (-13.75%) |
| AirExo-2 | RISE-2 | 85.000% (-8.75%) |

**Key takeaway**：RISE-2 在 AirExo-2 data 上的 drop（-8.75%）比 RISE（-13.75%）小，说明 generalizable policy 更能 handle pseudo-robot data 的 residual domain gap。这印证了 paper 的 thesis：scalable data collection + generalizable policy 必须 co-design。

### 6.4 Complex Task: Serve Steak（Figure 6）
- Long-horizon, contact-rich
- 50 in-the-wild demos via AirExo-2
- RISE-2 zero-shot deployment：50% overall success rate
- Scoop Steak phase（contact-rich）表现也不错

这个 result 很 impressive——用 $600 设备 + 50 demos，能在 contact-rich long-horizon task 上达到 50% success，没有任何 robot data。

### 6.5 Ablation on Visual Adaptors（Table 2）
Collect Toys：

| Method | w/o adaptors | w/ adaptors |
|---|---|---|
| RISE | 30.0% | 57.5% |
| RISE-2 | 52.5% | 90.0% |

**Insight**：即使没有 adaptor，RISE-2 (52.5%) 也接近 RISE with adaptor (57.5%)。这说明 RISE-2 的 generalization 能跨 embodiment——这是 DINOv2 + 3D perception 的功劳。但有 adaptor 后大幅提升到 90%，说明 adaptor 仍然 critical。

### 6.6 User Study（Table 3）
20 participants, Collect Toys task：

| Method | Time (s) | Intui. Rank | Learn. Rank | Rating |
|---|---|---|---|---|
| EE Teleop | 46.06±27.21 | 3.00/3 | 2.95/3 | 29.75 |
| Joint Teleop (AirExo) | 17.31±5.05 | 1.80/3 | 2.00/3 | 49.58 |
| **AirExo-2** | **5.66±1.97** | **1.20/3** | **1.05/3** | **83.00** |

Welch's t-test: p = 8.32e-10（极显著）。完成时间快 8×。这是 scalability argument 的核心证据——同样时间能收集 8× 的 data。

### 6.7 Action Accuracy（Table 4）
3 tracks evaluation：

| Device | Avg±std (mm) | Max (mm) |
|---|---|---|
| UMI | 8.855±3.228 | 20.002 |
| **AirExo-2** | **1.737±1.713** | **6.134** |

5× 精度提升。这是 mechanical vs SLAM 的根本差异。

UMI: https://universal-manipulation-interface.github.io/

### 6.8 Scalability Analysis（Figure 7）
- 相同 collection time：AirExo-2 throughput >> teleoperation
- Cost：$0.6k vs $60k（100×）
- 相同时间下，AirExo-2 训练的 policy 性能超过 teleop data 训练的 policy

这个 result 意味着：即使在"控制时间变量"后，AirExo-2 data 质量也不逊于 teleop data。可能因为：
1. Human 用 exoskeleton 操作更 natural，demonstration 质量更高
2. Adapted pseudo-robot data 经过了 visual normalization，更 consistent

---

## 7. Implementation Details 速览

### Training
- 4× NVIDIA A100
- 60k steps
- Batch size 240
- Initial lr 3e-4，warmup 2000 steps
- Cosine scheduler
- 20% color jitter augmentation (brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

### Inference
- RTX 2060 SUPER
- 5Hz
- 远程 server 推理 script 也提供

### Data Processing
- Color image: 448×252 (DINOv2) / 640×360 (ResNet-18)
- Depth: 640×360
- Voxel size 5mm
- Teleop point cloud crop: x[-0.70, 0.70], y[-0.30, 0.55], z[0.90, 1.55]
- AirExo-2 point cloud crop: x[-0.70, 0.70], y[-0.30, 0.45], z[0.75, 1.40]

注意 AirExo-2 的 crop range 略小（z 下限 0.75 vs 0.90），因为 mobile base 高度和 robot mount 不同。

### Trajectory Sampling
Threshold-based deduplication：
- Translation threshold: 5mm
- Rotation threshold: π/24 ≈ 7.5°
- Gripper width threshold: 5mm

这去掉静止帧，让 action distribution 更 balanced。

---

## 8. Limitations 和 Future Work

作者诚实地列出：
1. **Pseudo-robot data 主要适用于 generalizable policy**：对 narrow policy 可能不够，可以加 novel view synthesis augmentation（RoVi-Aug, ZeroNVS 等）
2. **没有 in-hand camera**：虽然设计了 connector，但 in-hand camera 和 exoskeleton 的 calibration 难。Appendix C.3 case study 显示 in-hand camera 单独不够，要配合 global camera
3. **只有 parallel gripper**：没支持 dexterous hand。Future work 可以集成 LEAP Hand、EyeSight Hand 等，以及对应 exoskeleton（DEXOS、HOMIE 等）
4. **没 force/torque, tactile, sound**：这些 modality 对 contact-rich task 重要（FoAR, ViTaMIn, ManiWAV 等工作）

HOMIE: https://homie-robot.github.io/
DEXOS: https://sites.google.com/view/dexos-dexterous
FoAR: https://arxiv.org/abs/2501.12581

---

## 9. 我的思考：这篇 paper 在大图景中的位置

### Data Scaling 视角
Lin et al. 2025 的 "Data Scaling Laws in Imitation Learning for Robotic Manipulation"（ICLR 2025）证明 imitation learning 也有 scaling law。这意味着 data collection efficiency 是第一性的。AirExo-2 把 marginal cost of one demonstration 从 ~(60k amortized + operator time) 降到 ~(0.6k amortized + operator time / 8)，是 scaling curve 上的 step function。

Data scaling paper: https://arxiv.org/abs/2406.15400

### Embodiment Gap 视角
这个工作其实是在做 embodiment transfer：human embodiment → robot embodiment。和以下工作形成谱系：
- **Video pretraining**（MimicPlay, R3M, LIV, VIP, Gen2Act）：从 internet video 学 representation，gap 大
- **Handheld devices**（UMI, AnyTeleop）：end-effector 一致，gap 中
- **Exoskeleton**（AirExo, AirExo-2, DexCap, HOMIE）：kinematic isomorphism，gap 小
- **Direct teleoperation**：gap 0

AirExo-2 的 visual adaptor 是把 exoskeleton 的 residual gap（visual appearance）显式 close 掉，所以能达到接近 teleop 的性能。

MimicPlay: https://mimic-play.github.io/
R3M: https://arxiv.org/abs/2203.12601

### 3D vs 2D Policy 视角
Manipulation policy 一直在 2D 和 3D 之间摇摆：
- 2D 阵营：ACT, Diffusion Policy, RT-1, RT-2, OpenVLA, π0
- 3D 阵营：RISE, 3D Diffusion Policy, Act3D, RVT, PolarNet
- **Hybrid**：Enerdex-Wang 等（PerAct 风格），RISE-2

RISE-2 的 hybrid 设计——sparse 3D for geometry + dense 2D for semantic——很 principled。3D 给 view-invariance（解决 camera viewpoint gap），2D foundation model 给 semantic generalization（解决 object/background variation）。Spatial aligner 的 inverse-distance interpolation 是 elegant 的 fusion mechanism。

3D Diffusion Policy: https://3d-diffusion-policy.github.io/
Act3D: https://act3d.github.io/
RVT: https://robot-vision-transformer.github.io/

### 为什么不直接用 VLA？
π0 在 in-domain 上输给 RISE-2 (85% vs 95%)，但在 combined generalization 上赢 (82.5% vs 62.5%)。这暗示：
- VLA 的优势在 compositional generalization（见过很多 task + object 组合）
- 3D policy 的优势在 spatial precision 和 in-domain efficiency

对 industrial deployment（特定 task、固定场景），RISE-2 这种方案可能更 practical。对 general-purpose robot，VLA 仍是 long-term 方向。两者可能 converge——把 3D perception 加进 VLA（Lift3D Foundation Policy 的方向）。

Lift3D: https://arxiv.org/abs/2411.18623

### Calibrated Data Pipeline 视角
AirExo-2 最 deep 的贡献可能是：它把 data collection 变成一个 **calibrated, reproducible pipeline**。每个 demonstration 都经过：
1. Mechanical calibration（forward kinematics）
2. Differentiable rendering calibration
3. Visual adaptation（mask + inpainting + ControlNet）
4. Depth adaptation（reference + render）

这和 ML 里的 data curation 思路一致——高质量 data 比 quantity 重要（"Data-Centric AI" movement）。ControlNet 的 platform-specific but task-invariant 性质意味着：一次性投入训练 ControlNet，之后所有 task 的 data 都自动 transform。这是 infrastructure 思维。

---

## 10. 可能的延伸阅读

- **AirExo (原版)**: https://airexo.github.io/
- **UMI**: https://universal-manipulation-interface.github.io/
- **RISE**: https://arxiv.org/abs/2406.14531
- **DINOv2**: https://dinov2.metademolab.com/
- **MinkowskiEngine**: https://github.com/NVIDIA/MinkowskiEngine
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **ControlNet**: https://arxiv.org/abs/2302.05543
- **SAM-2**: https://sam2.metademolab.com/
- **ProPainter**: https://github.com/sczhou/ProPainter
- **Redner (differentiable rendering)**: https://www.cs.cornell.edu/projects/diff-render/
- **6D Rotation Representation**: https://arxiv.org/abs/1812.07035
- **LoRA**: https://arxiv.org/abs/2106.09685
- **Open3D**: https://github.com/isl-org/Open3D
- **Data Scaling Laws in IL**: https://arxiv.org/abs/2406.15400
- **CAGE**: https://arxiv.org/abs/2410.14944
- **DexCap**: https://dexcap.github.io/
- **ARCap**: https://arcap-2024.github.io/
- **π0**: https://www.physicalintelligence.company/blog/pi0
- **3D Diffusion Policy**: https://3d-diffusion-policy.github.io/
- **RoVi-Aug**: https://arxiv.org/abs/2403.02115
- **HOMIE**: https://homie-robot.github.io/

---

## 11. 总结：Build Your Intuition

如果要用一句话 capture 这篇 paper 的 essence：

**Scalable imitation learning requires co-design of (1) low-cost hardware that mechanically aligns with robot kinematics, (2) visual adaptation that closes the embodiment gap via differentiable rendering + diffusion, and (3) generalizable policy that leverages 3D geometric invariance + 2D semantic priors.**

更深层的 intuition：
- **Mechanical precision >> visual precision**：能用机械结构解决的精度问题，就别用 SLAM
- **Disentangle modalities in feature space**：3D 学 geometry，2D 学 semantic，spatial alignment 做 fusion
- **Platform-specific adaptation, task-invariant data**：把 domain gap 的 cost 集中到一次性 infrastructure 投入上
- **Generalization comes from pretraining**：DINOv2 的 semantic prior 是 RISE-2 超越 prior 3D policies 的关键
- **Data quality > data quantity, but both matter**：AirExo-2 同时提升了 quality（mechanical accuracy）和 quantity（8× faster collection）

这篇 paper 是一个完整的 system paper，从 hardware design 到 calibration 到 visual adaptation 到 policy architecture，每一环都经过 careful design。最后的 Serve Steak experiment——50 demos + $600 设备 + zero-shot deployment 在 contact-rich long-horizon task 上达到 50% success——是这个 system thinking 的 payoff。这种 end-to-end system approach 是 SJTU Cewu Lu 组一贯风格（参考 AnyGrasp, GraspNet-1Billion, RH20T, RoboMIND 等），值得 study。

希望这个 walkthrough 帮你 build 起对 scalable imitation learning 的 intuition。如果你对某个具体 module（比如 differentiable rendering calibration 的梯度传播、或 spatial aligner 的复杂度分析）想深入，可以继续聊。
