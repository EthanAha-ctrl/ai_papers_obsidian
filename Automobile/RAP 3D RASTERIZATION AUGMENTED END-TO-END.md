---
source_pdf: RAP 3D RASTERIZATION AUGMENTED END-TO-END.pdf
paper_sha256: 79b5a3f272aa454c7d00435a7fa5ef44e9268c80fbcaf406d0fab0d5cadeb8d4
processed_at: '2026-08-11T20:53:18-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 RAP

## 这篇 paper 在解决什么问题

E2E autonomous driving 的训练方式很简单：拿一堆人类开车的视频，让神经网络模仿。这叫 imitation learning。

问题在于，人类开车都是 "正常状态下的正常操作"。比如你永远不会有一个训练样本是 "车已经偏到隔壁车道了，怎么打方向盘救回来"。因为人类 expert 根本不会开到那种状态去。

所以 model 一旦在 deployment 时犯个小错，稍微偏一点点，它就进了一个训练数据里从没见过的状态，然后不知道怎么 recover，然后错误越滚越大，最后撞了。这就是 closed-loop failure。

这个病在 open-loop 评估里完全看不出来，因为 open-loop 就是 "给一帧，看预测轨迹和 ground truth 差多少"，model 偏一点无所谓。只有 closed-loop eval（model 真的按自己预测的开，开完下一步再预测）才能暴露。

---

## 现有的解决方案有什么问题

要让 model 见过 "偏离状态"，就得造数据。现有方案有几种：

**方案 A：用 CARLA 这种游戏引擎模拟器**
缺点：看起来太假，跟 real image 差太多，model 在模拟器里学的东西迁移不到现实。

**方案 B：用 NeRF / 3D Gaussian Splatting 重建真实场景，然后从新视角 render**
缺点：慢得要死。每个场景要 optimize 好久，render 一帧要几秒。要造几十万 counterfactual 样本根本不现实。所以这些方法基本只能用来做 evaluation，不能用来 scale 训练数据。

大家 implicit 的 assumption 是：**模拟器越 photorealistic，训练效果越好**。

---

## 这篇 paper 的 core insight

作者站出来说：photorealism 根本不重要。

开车靠的是什么？是前面有没有车、车在哪个车道、红绿灯什么颜色、路怎么弯的。这些是 **geometry、dynamics、semantics**。至于路面的 texture 是沥青还是水泥、天上的云长什么样、车的漆面反光如何 —— 这些对 planning 决策 **完全无关**。

你想想，人类从 GTA 里学开车，画面完全是 game engine 渲染的，但 skills 照样迁移到现实世界。说明人脑 extract 的是 task-relevant 的 latent representation，不是 pixel-level appearance。

既然 self-supervised pre-trained vision encoder（比如 DINOv3）本来就在做 "保留 structure 丢弃 texture" 的事，那我们干嘛还要费劲去 render photorealistic image？直接 render 一个 "只有 geometry 和 semantics 的简化版" 就够了。

---

## RAP 的具体做法

### 第一步：3D Rasterization

拿 nuPlan 数据集的 annotation（每个 log frame 都标注了车道线、车辆位置、红绿灯等等），用最简单的 graphics pipeline 把这些东西画出来：

- 车辆 = 一个 cuboid（长方体），按类别上色
- 车道线 = polyline
- 红绿灯 = 一个小 cuboid，按状态涂红/黄/绿
- 背景 = 纯黑

就这些。没有 texture，没有 lighting，没有天空，没有树，没有路面纹理。看起来像 80 年代的 wireframe game。

但关键点：这个 rasterizer **快到飞起**。NeRF render 一帧要几秒，这个 render 一帧大概毫秒级。所以可以从 1200 小时的 nuPlan logs 里 rasterize 出 **50 万+ 样本**。

### 第二步：两种 augmentation

有了这么快的 rasterizer，就可以造训练数据里没有的 scenario：

**Recovery-oriented perturbation**：把 expert 的轨迹故意偏一点，重新 render 这个 "ego 偏离了" 的视角，让 model 学 "从偏离状态怎么开回来"。这直接打 covariate shift 的痛点。

**Cross-agent view synthesis**：nuPlan 每个场景有 N 辆车，每辆车都有自己的轨迹。原本只用 ego 视角，现在把每辆车的轨迹都当作 "ego 轨迹" render 一遍，等于从一个 log 里榨出 N 个不同视角不同行为的训练样本。这是 50 万样本的大头。

### 第三步：Raster-to-Real Alignment

现在有 50 万 raster 样本 + 8 万 real 样本。问题：raster 长得跟 real 差太多了（一个是黑底 wireframe，一个是真实照片），model 直接混着训练会不会 domain shift 严重？

作者的发现：用 frozen DINOv3 提 feature，做 PCA 可视化，real image 和 raster image 的 feature map 结构 **高度一致**。因为 DINOv3 已经把 texture 信息丢掉了，只保留 structure。

所以 sim-to-real gap 在 pixel space 很大，但在 feature space 很小。那就直接在 feature space 对齐：

**Spatial alignment**：对每个有 paired (real, raster) 的样本，让 real feature 向 raster feature 靠拢（MSE loss）。为什么是这个方向？因为 raster 来自高质量 annotation，feature 干净，是 "clean structural scaffold"。让 real feature 学它等于教 encoder 丢弃无关 texture。

**Global alignment**：大量 raster 样本是没有 paired real 的。用 DANN 的经典 trick —— 装个 domain classifier 试图分辨 "这是 real 还是 raster"，然后前面加 gradient reversal layer 让 encoder 反着学，让 classifier 分不出来。这就是 domain adversarial training。

两者一起用，sim-to-real gap 就被拉近了。

---

## 实验：四个 benchmark 全 SOTA

- **NAVSIM v1**（open-loop）：93.8 PDMS，第一
- **NAVSIM v2**（带 3DGS counterfactual 的 closed-loop）：36.9 EPDMS，第一
- **WOD-E2E**（Waymo 的 long-tail benchmark）：RFS 8.04，第一，beat 掉 Poutine 这种 3B 参数的 VLM 方法
- **Bench2Drive**（CARLA closed-loop）：37.27% success rate，第一

---

## 几个特别有意思的 ablation

**Background 是 natural（有天空有地面）vs 纯黑**：纯黑更好（MinADE 0.91 vs 1.33）。加 photorealistic 的 background 反而变差。这定量证明 "photorealism 不仅不必要，甚至有害"。

**Recovery perturbation 在 NAVSIM v1 上完全没效果（92.5 → 92.5），但在 v2 上 +4.4（32.5 → 36.9）**。说明这种 augmentation 是 closed-loop specific 的，open-loop 评估根本看不出价值。

**Cross-agent synthesis 的 scaling law**：MinADE = -0.021 × ln(sample_count) + 1.2173，R²=0.9942。完美的 log scaling。意味着 rasterized synthetic data 满足 scaling law，可以靠加数据继续涨点。

**50% synthetic + 50% real 比 100% real 还好**。说明 synthetic data 不只是 "凑数"，是真正有 task-relevant 信息的 augmentation。

---

## 为什么这个 work

我的理解是，这篇 paper 其实是在说一件很 deep 的事：**self-supervised representation learning 已经把 "photorealism → task performance" 这条路径短路了**。

传统思路：photorealistic simulator → 真实感图像 → encoder 提 feature → planner 决策。中间有个 "让图像看起来真实" 的环节，很贵很慢。

RAP 的思路：annotation → 简化 raster → encoder 提 feature → planner 决策。跳过 "看起来真实" 这个环节，因为 encoder 本来就会把 texture 丢掉。

所以本质上，RAP 是利用了 modern vision foundation model 的 invariance property：DINOv3 这种 model 对 appearance 不敏感，对 structure 敏感。那输入端就只需要提供 structure。

---

## 局限

- 还是 IL paradigm，causal confusion 的根没解决，只是缓解
- 只能 rasterize annotation 里有的东西，没标注的 cue（特殊路牌、LED 显示屏）raster 抓不到，这些得靠 real image 补
- 假设其他 agent 的轨迹是 reasonable driving behavior，但有些 aggressive driver 的轨迹可能教坏 model
- 还是 frame-wise 重建，动态遮挡处理可能有 artifact

---

## 一句话

放弃 photorealism，因为 DINOv3 这种 encoder 本来就把 texture 丢了；改用超快的 annotation-driven rasterizer，scale 出 50 万训练样本；在 feature space 做 sim2real alignment；四个 benchmark 全 SOTA。核心 insight 是 "task-relevant structure 才是 bottleneck，texture 是 over-kill"。

---

# RAP: 3D Rasterization Augmented End-to-End Planning — 深度拆解

## 1. 一句话直觉

这篇 paper 的核心 thesis 可以浓缩成一句：**driving 是个 geometry + dynamics + semantics 的问题，不是 photorealism 的问题**。人类能从 GTA 里把开车技能迁移到现实世界，说明 task-relevant 的 latent feature alignment 比 pixel-level appearance 重要得多。基于这个观察，作者放弃 NeRF / 3DGS / CARLA 这条 "追求 photorealism" 的路线，转而用 annotation-driven 的 lightweight 3D rasterization 生成 500k+ 训练样本，再用 feature-space alignment 把 sim 和 real 拉到一起。

这其实是把 "rendering for training" 这个问题重新 frame 成 "what is the minimal sufficient representation for a planner to learn"，类似于 self-supervised learning 里 "丢弃 label 保留 structure" 的哲学。

---

## 2. Why this paper matters: IL 的 covariate shift 病根

E2E driving 的 IL 训练有个 well-known failure mode（Ross et al. 2011, DAgger）：

```
expert 只演示 "好状态 → 好 action"
policy 在 deployment 一旦偏离 expert 分布 → 没见过 → 不知道怎么 recover → 错误 compound
```

这个病在 closed-loop eval 下暴露无遗，open-loop metric（ADE / FDE）完全看不出来。NAVSIM v2 引入的 3DGS 两阶段 evaluation 就是专门把这个 gap 揭露出来。

**解决路径**：
- **DAgger**：在线 query expert，但 expert 是 human 不可能
- **Adversarial scene generation**（ChauffeurNet, KING, RegentS）：在 BEV space 造 hard case，但只用在 mid-to-end
- **Photorealistic digital twin**（NeuroNCAP, HUGSIM, RealEngine, RAD）：用 NeRF/3DGS 重建 log，可以 counterfactual replay，但慢、贵，主要用作 evaluation

RAP 选择第四条路：**lightweight rasterization + feature alignment**，目标是把训练数据从 "100k expert 演示" 扩展到 "500k counterfactual + cross-agent views"。

参考：
- DAgger: https://arxiv.org/abs/1011.0686
- ChauffeurNet: https://arxiv.org/abs/1812.03079
- NeuroNCAP: https://arxiv.org/abs/2408.00615
- HUGSIM: https://arxiv.org/abs/2412.01718
- RealEngine: https://arxiv.org/abs/2505.16902

---

## 3. 3D Rasterization — 公式级讲解

### 3.1 Scene Representation

每个 log frame 用 annotation 重建，分两类 primitives：

**Map elements (静态)** — polylines：
$$
\mathcal{M} = \{\mathbf{P}_k\}, \quad \mathbf{P}_k \in \mathbb{R}^{n_k \times 3}
$$
- $\mathcal{M}$：整个 map 的 polyline 集合
- $\mathbf{P}_k$：第 k 条 polyline
- $n_k$：第 k 条 polyline 的 vertex 数
- $3$：world coordinates $(x, y, z)$

**Dynamic actors** — oriented cuboids：
$$
\mathcal{B}_i = (l_i, w_i, h_i, \mathbf{T}_i)
$$
$$
\mathbf{C}_i = \mathbf{T}_i \begin{bmatrix} \pm l_i/2 & \pm w_i/2 & 0, h_i \end{bmatrix}^\top
$$
- $l_i, w_i, h_i$：actor i 的长宽高
- $\mathbf{T}_i \in SE(3)$：actor i 在 world 的 rigid-body pose（rotation + translation）
- $\mathbf{C}_i \in \mathbb{R}^{8 \times 3}$：cuboid 的 8 个 3D corner points

注意 traffic lights 是固定尺寸的 upright cuboid，按 state (red/yellow/green) 上色 —— 这种 semantic color coding 把一个原本需要 perception 的 task 直接 bake 进 input。

### 3.2 Pinhole Camera Projection (公式 1)

$$
\mathbf{u}_{uv} = \pi(\mathbf{p}_w) = K \mathbf{T}_{w \to c} \tilde{\mathbf{p}}_w
$$
- $K \in \mathbb{R}^{3 \times 3}$：camera intrinsics (fx, fy, cx, cy)
- $\mathbf{T}_{w \to c} \in SE(3)$：world to camera extrinsics
- $\tilde{\mathbf{p}}_w = [\mathbf{p}_w^\top, 1]^\top \in \mathbb{R}^4$：homogeneous coordinates
- $\mathbf{u}_{uv} \in \mathbb{R}^3$：投影后的 homogeneous pixel coords

Perspective division：
$$
(u, v) = \left( \frac{u_x}{u_z}, \frac{u_y}{u_z} \right)
$$
- $u_z$：depth（z 轴沿光轴方向）

Discard if $u_z < z_{\text{near}}$ —— 防止 near plane 后的点把 z-buffer 搞乱。

### 3.3 Rasterization — Depth-aware Compositing

输出 RGB canvas $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$，每个 fragment 存：
- depth $d$
- fading weight $\alpha = \max(0, 1 - d/d_{\max})$

这里 $\alpha$ 是 **depth-based intensity decay**：远处物体变淡。这个设计是 ablation Table 5 里 critical 的一项（No decay: MinADE 1.05 vs Yes: 0.91）—— 直觉是 far objects 信息量低且 noisy，weight 衰减能让模型 focus 在近处高信息密度区域。

Occlusion resolution：single depth buffer (painter's algorithm 的简化版)。
View boundary clipping: **Sutherland-Hodgman** polygon clipping (Sutherland & Hodgman 1974) —— 这是经典 computational geometry 算法，把超出 view frustum 的 polygon 切掉，避免边界处出现伪影。

### 3.4 关键 insight：DINOv3 features 自洽

Figure 4 是整篇 paper 最 intuitive 的 figure 之一：把 real image 和 rasterized image 各过 frozen DINOv3-H，对 output feature 做 PCA 可视化。结果两个 domain 的 feature map 结构高度一致。

这给了一个很强的 claim：**self-supervised pre-trained vision encoder 本来就在做 "丢弃 texture，保留 structure" 的事**。rasterized input 把这件事推到极端（直接把 texture 全去了），DINOv3 还是能提取出同构的 feature。这跟最近 representation alignment 文献（Yu et al. 2024, "Representation Alignment for Generation"）的观察吻合 —— aligned latent space 比 raw pixels 更易迁移。

参考：
- DINOv3: https://arxiv.org/abs/2508.10104
- Representation alignment: https://arxiv.org/abs/2410.06940

---

## 4. Data Augmentation — 两种关键 trick

### 4.1 Recovery-oriented Perturbations

$$
\tilde{\tau}(t) = \tau^*(t) + \delta_{\text{lat}}(t) + \delta_{\text{long}}(t) + \varepsilon_t
$$
- $\tau^*(t)$：expert ground truth trajectory
- $\delta_{\text{lat}}, \delta_{\text{long}}$：从 predefined range 采样的 lateral / longitudinal offset
- $\varepsilon_t$：Gaussian noise

把 perturbed $\tilde{\tau}$ 重新 rasterize，得到 "ego 偏离 expert path" 的 counterfactual scene。训练时让 model 看 "从偏离状态恢复回正轨" 的 demonstration，直接打 IL brittleness 的痛点。

**关键 ablation (Table 6)**：
- NAVSIM v1: 92.5 → 92.5（无变化）
- NAVSIM v2: 32.5 → 36.9（+4.4）

为什么 v1 没动？因为 v1 是 open-loop PDMS，本身就反映不出 closed-loop brittleness。v2 的两阶段 evaluation + 3DGS counterfactual 才暴露问题，perturbation 的价值也才显现。这是评估协议跟方法 design 必须耦合的好例子。

### 4.2 Cross-agent View Synthesis

nuPlan 每个 scenario 有 $n$ 个 agent trajectory。原方法只 render ego view。RAP 的 trick：把 ego trajectory **替换成其他 agent 的 trajectory**，但保持 camera intrinsics/extrinsics 不变。这等于 "假装我是那辆车，开他们的路径"。

这样从每个 log 可以榨出 n 个不同的视角 + n 个不同的 "ego 行为" 的训练样本。500k synthetic samples 的 majority 来自这里。

**Scaling law (Figure 6)**：
$$
y = -0.021 \ln(x) + 1.2173, \quad R^2 = 0.9942
$$
- $x$：synthetic sample count (从 1k 到 1000k)
- $y$：MinADE

这是经典的 log-scaling law（参考 Baniodeh 2025 scaling laws report, Zheng 2024）。注意 R² = 0.9942 非常高，说明 **cross-agent synthetic data 满足 scaling law**，意味着 secondary viewpoint 不是噪声，是有 task-relevant 信息的训练 signal。

参考：
- Scaling laws for motion forecasting: https://arxiv.org/abs/2503.01975
- E2E data scaling: https://arxiv.org/abs/2412.02689

---

## 5. Raster-to-Real (R2R) Alignment — Sim2Real 在 feature space 的解法

这是 paper 的第二个核心 contribution。核心 idea：sim2real gap 在 pixel space 难解决（raster 是黑底、无 texture、cuboid 化），但在 feature space 容易（DINOv3 已经把两者拉到同构了）。

### 5.1 Spatial-level Alignment (公式 2)

$$
F^r = \phi(x^r), \quad F^s = \phi(x^s), \quad F^r, F^s \in \mathbb{R}^{N \times d'}
$$
- $x^r$：real image
- $x^s$：paired rasterized rendering（同 frame 同 viewpoint）
- $\phi(\cdot)$：visual encoder (frozen DINOv3-H for RAP-DINO, ResNet34 for RAP-ResNet)
- $N$：spatial location 数量（ViT patch tokens 或 CNN feature map positions）
- $d'$：projected feature dim

$$
\mathcal{L}_{\text{spatial}} = \frac{1}{N} \sum_{j=1}^{N} \| F^r_j - F^s_j \|_2^2 \tag{2}
$$

关键 design choice：**freeze $F^s$，update $F^r$ 朝 $F^s$ 对齐**。这个方向叫 "Real-to-Raster" (在 appendix Table 7 验证)，但 paper 主文叫 "Raster-to-Real" 是因为目的是让 real features 学到 raster features 的 clean structure。这是 representation alignment 里的标准做法 —— align toward the cleaner / more structured space。

**为什么这个方向 work**：raster features 来自高质量 annotation，没有 distracting texture 噪声，是 task-relevant structure 的 "clean proxy"。让 real features 向它对齐 = 让 encoder 学会丢弃无关 texture，提取 driving-relevant structure。

Table 7 ablation 验证：
- Real-to-Raster: MinADE 1.02 (best)
- Symmetric: 1.14
- Raster-to-Real: 1.12

### 5.2 Global Alignment (公式 3)

Spatial alignment 需要 paired (real, raster) sample。但 raster-only 数据远多于此（500k synthetic vs 85k paired）。要利用 unpaired raster data，用经典 DANN 思路 (Ganin & Lempitsky 2015)：

$$
g = \text{AvgPool}(F), \quad g \in \mathbb{R}^{d'}
$$
$$
\mathcal{L}_{\text{global}} = -\mathbb{E}_{(g, y)} \big[ y \log D(g) + (1-y) \log(1 - D(g)) \big] \tag{3}
$$
- $g$：global feature（average pool 整个 feature map）
- $D$：domain classifier (lightweight MLP)
- $y \in \{0, 1\}$：domain label (real vs raster)

前面加 **Gradient Reversal Layer (GRL)**：forward 时 identity，backward 时乘 $-\lambda$，这样 encoder 被优化去 **maximize domain confusion**（让 D 分不出 real vs raster），同时 D 自己被优化去 minimize classification error。

GRL schedule (公式附录)：
$$
\lambda(p) = 0.1 \cdot \left( \frac{2}{1 + \exp(-\gamma p)} - 1 \right), \quad p \in [0, 1], \gamma = 10
$$
- $p$：training progress (0 to 1)
- $\gamma$：annealing sharpness

这是个 sigmoid-based ramp：开始 $\lambda \approx 0$（不强制 align，先学 task），后期 $\lambda \to 0.1$（强化 domain confusion）。这种 schedule 是 DANN 的标准做法，避免早期 adversarial signal 把 representation 搞坏。

**Why global**：raster 有大量纯黑背景，real 没有。这种 systematic bias 在 spatial alignment 看不到（黑区域 spatial token 自然 close），但在 global distribution 上会偏。global alignment 强制 encoder 学 domain-invariant representation，缓解这种 distribution shift。

参考：
- DANN (Ganin & Lempitsky): https://arxiv.org/abs/1409.7495
- DINOv3 features PCA 分析类似 MAE 做法

### 5.3 Overall Objective

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_s \mathcal{L}_{\text{spatial}} + \lambda_g \mathcal{L}_{\text{global}}
$$
- $\lambda_s = 0.002$（spatial weight，小因为 MSE 量级大）
- $\lambda_g = 0.1$（global weight）

$\mathcal{L}_{\text{task}}$ 是 multi-modal trajectory head + PDMS scoring head（参考 iPad Guo et al. 2025）。

---

## 6. Architecture 细节

### 6.1 RAP-DINO (主模型，~888M params)

```
[Real image x^r]  →  [Frozen DINOv3-H backbone]  →  F^r
                                                      ↓
                                                  [MLP projector]  →  →  [iterative deformable
[Raster x^s]     →  [Frozen DINOv3-H backbone]  →  F^s       (learnable)        attention decoder]
                                                      ↓                     ↓
                                                  frozen                [trajectory head]
                                                                        [PDMS scoring head]
```

Components：
- **DINOv3-H**：frozen，作为 perception backbone（参数大概 700M+）
- **MLP projector**：learnable，把 DINOv3 features project 到 planner 工作的 dim
- **Iterative deformable attention decoder**：从 iPad (Guo et al. 2025) 改造而来，做 iterative proposal refinement
- **Multi-modal trajectory head**：supervised by future trajectories
- **PDMS scoring head**：supervised by PDMS scores

### 6.2 RAP-ResNet (~29M params)

- ResNet34 backbone
- 主要用于 Bench2Drive 因为要 closed-loop inference，需要 fast
- 用于 ablation 加速

### 6.3 Model-agnostic 性质

Paper 把 RAP framework 套到三个不同 planner 上验证 generalizability：
- RAP-iPad (Guo et al. 2025)：+0.7 PDMS
- RAP-DiffusionDrive (Liao et al. 2025)：+3.2 PDMS
- RAP-DINO：+from scratch SOTA

参考：
- iPad: https://arxiv.org/abs/2505.15111
- DiffusionDrive: https://arxiv.org/abs/2412.20124

---

## 7. Experiments — 四个 benchmark 全部 SOTA

### 7.1 NAVSIM v1 (Table 1)

RAP-DINO PDMS = **93.8**，第一。Sub-metrics:
- NC (No at-fault Collision): 99.1
- DAC (Drivable Area Compliance): 98.9
- TTC (Time-to-Collision): 96.7
- EP (Ego Progress): 90.3

对比：
- Centaur (Sima et al. 2025, test-time training): 92.1
- iPad: 91.7
- DiffusionDrive Camera-only: 86.0 → RAP-DiffusionDrive 89.2 (+3.2)

### 7.2 NAVSIM v2 (Table 2)

NAVSIM v2 是真正测 closed-loop robustness 的 benchmark，引入 3DGS counterfactual + EPDMS metric。

RAP-DINO EPDMS = **36.93**，vs LTF 23.12（+13.8）。Stage 1 vs Stage 2：
- Stage 1: 36.93 (normal)
- Stage 2: 36.93 (with 3DGS counterfactual view synthesis)

LTF Stage 1 → Stage 2: 23.12 → 23.12（其实 LTF 是 constant across stages 的奇怪现象，可能 report 是 stage-combined score）。

注意 v2 引入的新 metrics：
- TLC (Traffic Light Compliance)
- DDC (Driving Direction Compliance)
- LK (Lane Keeping)
- EC (Extended Comfort)
- EPDMS = weighted aggregate

### 7.3 WOD-E2E Driving (Table 3)

WOD-E2E 是 Waymo 专门 curate 的 long-tail benchmark (construction detours, pedestrian accidents, freeway obstacles)，这些 case 在 daily driving 中 frequency < 0.003%。

RAP-DINO:
- ADE@5s: 2.65 (lowest)
- ADE@3s: 1.17 (lowest)
- RFS (Spotlight): 7.20 (highest)
- RFS (Overall): 8.04 (highest)

对比 Poutine (Rowe et al. 2025) 是 3B VLM-based 方法，RAP-DINO 888M params 反而 beat 它。这印证 paper 的 claim：lightweight geometric reasoning + feature alignment > 通用大模型 brute force。

参考：
- WOD-E2E: https://arxiv.org/abs/2510.26125
- Poutine: https://arxiv.org/abs/2506.11234

### 7.4 Bench2Drive (Table 4)

Bench2Drive 是 CARLA-based closed-loop benchmark，220 routes 每个含 safety-critical event。

RAP-ResNet (29M params):
- Success Rate: **37.27%** (highest)
- Driving Score: **66.42** (highest)
- Efficiency: 165.47 (highest)
- Comfortness: 23.63 (中等，trade-off 了 comfort 换 efficiency)

对比 DriveTransformer: 63.46 DS / 35.01% SR；iPad: 65.02 DS / 35.91% SR。

参考：
- Bench2Drive: https://arxiv.org/abs/2406.07492
- CARLA: https://carla.org/

---

## 8. Ablations — 拆解每个 design choice

### 8.1 Rasterization Design (Table 5)

| ID | Face Rendering | Depth Decay | Background | MinADE↓ |
|----|---------------|-------------|------------|---------|
| A | Colored | Yes | Black | **0.91** |
| B | Transparent | Yes | Black | 0.98 |
| C | Colored | No | Black | 1.05 |
| D | Colored | Yes | Natural | 1.33 |

关键发现：
- **Colored faces > Transparent** (0.91 vs 0.98)：solid color 给 semantic cue，transparent 只剩 wireframe，信息量低
- **Depth decay > No decay** (0.91 vs 1.05)：far 区域降权，focus near high-info
- **Black bg > Natural bg** (0.91 vs 1.33)：这个差距最大！natural sky-ground split 引入无关 texture，干扰学习

D 是最 surprising 的 ablation —— 加回 natural background 反而严重变差。说明 paper 的 "photorealism unnecessary" claim 不只是 "不需要"，甚至是 "加了会变差"。

### 8.2 Recovery Perturbation (Table 6)

- v1: 92.5 → 92.5（open-loop 看不出）
- v2: 32.5 → 36.9（+4.4）

这个 gap 说明：recovery perturbation 是 closed-loop specific augmentation，必须在 closed-loop metric 下评估才能看到价值。也说明 NAVSIM v1 (PDMS) 是 insufficient 评估协议，v2 (EPDMS) 才 capture 到这点。

### 8.3 R2R Alignment (Figure 5)

变量：real data ratio {1%, 5%, 20%, 50%, 100%}，剩余用 raster 替换。

发现：
1. **任何 alignment > no alignment**：spatial 和 global 都改善
2. **Spatial + Global > Spatial alone**：两者 complementary
3. **50% synthetic 比 100% real 好**：synthetic data 不只是 substitute，是 powerful augmentation

第 3 点是最重要的 scaling insight —— 说明 rasterized data 在 quality 上不仅 OK，还能 exceed 等量 real data。这给未来 scaling E2E driving 训练数据提供了一条低成本路径。

### 8.4 Cross-agent Scaling (Figure 6)

Log law: $y = -0.021 \ln(x) + 1.2173$，R² = 0.9942。这跟 LLM scaling laws 的形式完全一致（loss ∝ log(data)）。说明 E2E driving 的 data scaling law 不只在 real data 上 hold，在 synthetic rasterized data 上也 hold，且 secondary viewpoints 同样 contribute。

---

## 9. Implementation Details (Table 8)

- Hardware: 4× H100
- Training time: ~80 hours (full pretraining)
- Optimizer: AdamW (Loshchilov & Hutter 2017)
- LR: 1e-4 initial, cosine decay
- Weight decay: 1e-4
- Batch size: 128 (NAVSIM) / 64 (WOD/B2D)
- Epochs: 20 pretrain + 20 finetune
- Dropout: 0.1
- No gradient clipping (0.0)

WOD finetuning 特殊：unfreeze visual encoder，lr 1e-5，两阶段（train split → val split），NMS ensembling 2 checkpoints。

Bench2Drive mixed training：nuPlan 和 Bench2Drive 数据格式统一（camera view reorder, resize to 576×1024, calibration matrix adjust），保证 joint optimization 稳定。

参考：
- AdamW: https://arxiv.org/abs/1711.05101

---

## 10. Discussion & Limitations

### 10.1 Rasterization 是否丢失关键 visual cue？

Appendix Figure 7 给出 ablation：同一个 fully trained RAP-DINO，分别 condition on real image 和 rasterized image。两个 case：
- **Scenario A**：unannotated "Keep Left" sign —— real image 下 model 正确反应，raster 下失败
- **Scenario B**：OOD dynamic LED arrow —— real image 下 model 识别并安全变道，raster 下失败

这证明 model **在 inference 时** 能感知 raster ontology 之外的 cue，因为训练时同时看了 real 和 raster。Raster 提供 geometric scaffold，real 提供 texture/semantic richness，两者 complementary。

但同时也证明：raster 单独不足以处理所有 case，**必须搭配 real data**。这跟 paper 的整体 framing 一致 —— raster 是 augmentation 不是 replacement。

### 10.2 Real-to-Raster 是否信息损失？

这是 reviewer 必问的问题：把 real features 拉向 raster features，会不会 suppress 掉 unannotated cue？

Table 7 验证 Real-to-Raster (1.02) < Symmetric (1.14) < Raster-to-Raster (1.12)。Real-to-Raster 最好。

作者的解释：multi-task learning 保护了信息。Real-to-Raster alignment 提供 geometric prior，planning loss + perception loss 强制 real features 保留 task-relevant 信息。这跟 Yu et al. 2024 representation alignment 的观察一致 —— align to well-structured space improves abstraction without discarding perceptual detail。

### 10.3 局限

Paper 自己 ack 的局限：还是 IL paradigm，继承 causal confusion 问题。未来工作：把 3D rasterization 扩展成 full simulator 支持 closed-loop RL。

我的 additional thoughts：
- **Causal confusion** 是 IL 的根本病，feature alignment 缓解但不解决。需要 causal reasoning 或 RL 才能根治
- **Rasterization ontology 限制**：只能渲染 annotation 里有的东西。corner case 里没 annotation 的 cue（特殊 sign, LED display）raster 抓不到
- **Distribution shift**：cross-agent view synthesis 假设其他 agent 的 trajectory 是 reasonable driving behavior，但 nuPlan 里有些 aggressive driver 的 trajectory 可能让 model 学到危险行为
- **Static scene assumption**：rasterization 是 frame-wise 的，dynamic occlusion / deocclusion 处理可能有问题

---

## 11. 跟其他工作的关系 — Intuition 联想

### 11.1 vs NeRF / 3DGS-based 方法 (NeuroNCAP, HUGSIM, RealEngine)

NeRF/3DGS 走 pixel-space fidelity 路线，目标是 "render 出来的图和 real 看起来一样"。优点：训练和 eval 完全可复用。缺点：慢（每帧秒级），optimization 贵，view 偏离大时 artifact。

RAP 反其道而行：放弃 pixel fidelity，要 feature-space fidelity。speed 提升 1000×，scale 到 500k samples。代价：需要 feature alignment module 处理 domain gap。

### 11.2 vs VISTA (Amini et al. 2020, 2022)

VISTA 是 real-image reprojection：把 real image 用 depth + ego offset 重新 warp 到新 viewpoint。优点：photorealistic。缺点：只能 small ego deviation，大 deviation 就 broken。

RAP 用 cuboid 重建 scene，可以从任意 viewpoint render（cross-agent），不受 reprojection 距离限制。

### 11.3 vs ChauffeurNet (Bansal et al. 2019)

ChauffeurNet 的 "synthesizing the worst" 在 BEV space 生成 hard case。RAP 在 perspective view 做类似的事，但走 rasterization 路线，能直接 feed E2E camera-based planner。

### 11.4 vs DANN (Ganin & Lempitsky 2015)

RAP 的 global alignment 直接复用 DANN 的 GRL 思路。这是 sim2real 经典做法，RAP 把它跟 spatial alignment 结合，hybrid 处理 paired + unpaired 数据。

### 11.5 vs Diffusion Models 的 Representation Alignment (Yu et al. 2024)

Yu et al. 2024 发现 aligning diffusion DiT 的 representation 到预训练 encoder (CLIP/DINO) 大幅加速训练。RAP 的 R2R alignment 是这个 idea 在 sim2real 上的应用 —— align target 是 "clean, structured" 的 raster features。

### 11.6 vs Occupancy / Voxel Reconstruction (Symphonize, SelfOcc, VoxDet)

Occupancy methods 重建 dense 3D voxel grid。RAP 走 sparse primitive (cuboid + polyline) 路线，更轻量，annotation-driven 不需要 dense label。

### 11.7 vs VAD / UniAD / DriveTransformer / Transfuser

这些都是 E2E planner 架构创新。RAP 是 data augmentation framework，model-agnostic，可以叠加到这些方法上。Paper 验证了叠加到 iPad 和 DiffusionDrive 上都有正收益，理论上也能叠加到 UniAD/VAD。

参考：
- UniAD: https://arxiv.org/abs/2208.04353
- VAD: https://arxiv.org/abs/2303.12077
- DriveTransformer: https://arxiv.org/abs/2503.07656
- Transfuser: https://arxiv.org/abs/2205.15997
- VoxDet: https://arxiv.org/abs/2506.04623

---

## 12. 我的 Intuition 总结

读完这篇 paper 我 build 出来的几个 intuition：

1. **Driving 的本质是 geometry + dynamics + semantics，texture/lighting 是冗余的**。这个 claim 在 ablation Table 5 D (natural bg 反而更差) 上得到 quantitative 验证。

2. **Feature space 比 pixel space 更易做 sim2real alignment**。因为 self-supervised encoder (DINOv3) 已经在做 "保留 structure 丢弃 texture" 的工作，feature space 里 sim 和 real 天然接近。

3. **Scaling law 在 synthetic raster data 上同样 hold**（Figure 6, R²=0.9942）。这意味着 E2E driving 的数据瓶颈可以靠 rasterization 突破，不需要 photorealistic simulator。

4. **Closed-loop 鲁棒性需要 closed-loop-specific augmentation**（Table 6, perturbation 在 v2 才显效）。open-loop metric 完全看不出这个价值，必须用 NAVSIM v2 / Bench2Drive 这种 closed-loop benchmark。

5. **Multi-task learning 保护信息不丢**（Table 7, Real-to-Raster alignment + task loss 不会 suppress unannotated cue）。这给了 representation alignment 一个安全使用的范本。

6. **Lightweight method 可以 beat large VLM**（WOD-E2E 上 RAP-DINO 888M beat Poutine 3B）。说明在 task-specific 场景，inductive bias + 数据策略 比 scale 更重要。

7. **Raster + Real 是 complementary 不是替代**（Figure 7 Scenario A/B）。Raster 提供 clean geometric scaffold，real 提供 perceptual richness。两者必须一起用。

---

## 13. 项目链接 & 资源

- **Project page**: https://alan-lanfeng.github.io/RAP/
- **NAVSIM benchmark**: https://github.com/autonomousvision/navsim
- **nuPlan dataset**: https://www.nuscenes.org/nuplan
- **Bench2Drive**: https://github.com/Thinklab-SJTU/Bench2Drive
- **WOD-E2E**: https://arxiv.org/abs/2510.26125
- **DINOv3**: https://arxiv.org/abs/2508.10104
- **DANN (GRL)**: https://arxiv.org/abs/1409.7495
- **CARLA**: https://carla.org/
- **3D Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **NeuroNCAP**: https://arxiv.org/abs/2408.00615
- **HUGSIM**: https://arxiv.org/abs/2412.01718
- **VISTA**: https://arxiv.org/abs/2011.12019
- **ChauffeurNet**: https://arxiv.org/abs/1812.03079
- **iPad**: https://arxiv.org/abs/2505.15111
- **DiffusionDrive**: https://arxiv.org/abs/2412.20124
- **Centaur**: https://arxiv.org/abs/2503.11650
- **UniAD**: https://arxiv.org/abs/2208.04353
- **VAD**: https://arxiv.org/abs/2303.12077
- **DriveTransformer**: https://arxiv.org/abs/2503.07656
- **Poutine**: https://arxiv.org/abs/2506.11234
- **DAgger**: https://arxiv.org/abs/1011.0686
- **AdamW**: https://arxiv.org/abs/1711.05101
- **Sutherland-Hodgman clipping**: https://en.wikipedia.org/wiki/Sutherland–Hodgman_algorithm
- **Representation Alignment for Generation (Yu et al.)**: https://arxiv.org/abs/2410.06940
- **E2E Driving Survey (Chen et al. 2024)**: https://arxiv.org/abs/2306.16927
- **Data scaling laws for E2E driving**: https://arxiv.org/abs/2412.02689
- **Scaling laws for motion forecasting (Waymo)**: https://arxiv.org/abs/2503.01975

---

## 14. 一句话 take-away

> 把 "sim2real for E2E driving" 这个问题从 pixel space 转到 feature space，用 annotation-driven rasterization 替代 photorealistic rendering，靠 DINOv3 + DANN-style alignment 把 synthetic 和 real 拉到同构 —— 这样既拿到了 photorealism 路线拿不到的 scale（500k samples vs 100 hours logs），又保住了 closed-loop robustness（4 个 benchmark SOTA）。

Andrej 这篇 paper 的 framing 你应该会喜欢 —— 它本质上是在 challenge "more photorealism = better simulator = better training" 这个 implicit assumption，用 data + ablation 证明 "task-relevant structure" 才是真正的 bottleneck，texture 是 over-kill。这跟你之前在 micrograd / 神经网络教学里强调的 "找最 minimal sufficient representation" 是一致的哲学。
