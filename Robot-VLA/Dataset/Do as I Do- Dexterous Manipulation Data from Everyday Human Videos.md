---
source_pdf: Do as I Do- Dexterous Manipulation Data from Everyday Human Videos.pdf
paper_sha256: fc774e7c1815615652ddfdc843cc9a8204eb8c8cffe1a5bde7f55e837801c09e
processed_at: '2026-08-03T22:50:30-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DO AS I DO

## 一句话概括

Internet 上人做事的视频一大堆，但 robot 没法直接学——因为中间隔着两道墙：第一道是"从视频里看懂手和物体到底怎么动的"，第二道是"把人的动作翻译成 robot 手能执行的动作"。这篇 paper 给出了一个能同时翻过这两道墙的完整 pipeline，第一次让 in-the-wild internet 视频走到 real robot 上真的能跑出来。

---

## 为什么这个问题 hard

你做了很多 robot learning 的工作，肯定比谁都清楚 data 的痛。Teleoperation 慢、贵、operator 难训。Simulated exploration 又要 design environment、design reward。真正大规模、几乎 zero-cost 的 observational data 就躺在 YouTube、Ego4D、100DOH 里——人做事的视频。但 robot 用不了，因为：

1. **"看"和"做"不是一回事**：小孩看一遍大人做事就能模仿，robot 看了视频连"手在哪、物体在哪、怎么动的"都没完全搞清楚，更别说做了
2. **人手和 robot 手长得不一样**：人有 5 根手指、21 个 DoF、fingertip 形状圆润、有 friction skin；Sharpa Wave 这种 robot hand 是 22-DoF、link lengths 不同、关节 articulation 不同。机械地"摆同样的姿势"很容易穿模、指尖滑掉、抓不住
3. **Internet 视频又脏又乱**：motion blur、occlusion、low resolution、相机抖、切镜头、物体出画面、手不露全——这些在 lab dataset 里基本不存在，但在 internet 视频里是家常便饭

之前的工作要么假设有 depth、有 3D scan、有 hand keypoints（DexMV, DexImit），要么假设 clean lab 视频（MCC-HO, G-HOP），要么假设 pick-and-place only 的简单动作（H2Sim2Robot）。这些 assumption 一旦拿掉，方法就崩。DO AS I DO 的卖点就是不假设这些——任意 rigid object、任意 hand-object interaction 行为、任意 video source（ego/exo/internet/generated）。

---

## Pipeline 全景

整个方法就两步，直觉上很 clean：

```
RGB video
   ↓
[Reconstruction] 用 vision foundation models 把 video 重建为 4D hand-object trajectory
   ↓
[Retargeting] 用 GPU-parallel physics simulator 把 trajectory 翻译成 robot 可执行的控制序列
   ↓
Real-world rollout on UR3e arms + Sharpa Wave hands
```

每一步都靠最近 1-2 年的 foundation model breakthrough 才能做出来，所以这 paper 出现在 2026 年初一点都不意外——6 个月前 SAM 3D、MoGe-2、HaWoR 都还没出，这个 pipeline 物理上不可能。

---

## 第一步：Reconstruction 的人话版

### 现成组件先拿来用

Hand tracking 直接用 HaWoR（CVPR 2025, Potamias et al.）——egocentric 视频→world-space 3D hand motion，对 motion blur 和 occlusion 已经够 robust。Depth 和 camera intrinsics 用 MoGe-2（arxiv 2507.02546）——单目 RGB→metric depth。Segmentation 用 SAM 3。这三个都是 plug-and-play。

### 难点在 object pose tracking

HaWoR 给手，MoGe 给 depth，SAM 3 给 mask，但**物体在每一帧的 6-DoF pose 怎么追**？这是 reconstruction 真正的 bottleneck。

现有方案两种都不行：
- **Joint hand-object reconstruction**（HO, IHOI, HORSE, MCC-HO, G-HOP）：在 lab 数据上还行，in-the-wild 上天花板低，Table 2 里 F-5 最高 0.36 vs object tracker 的 0.69，差一倍
- **6-DoF object tracker**（FoundationPose, Any6D）：clean 视频上很好，但 in-the-wild 一遇到 motion blur/occlusion 就 lose lock、drift，re-acquire 也做不到——见 Fig 5 右边的对比，FPose 稍微模糊一下就丢了，整个后半段全乱

### 关键 insight：把 image-to-3D 生成模型当 video tracker 用

SAM 3D 是个 image-to-3D 生成模型，输入 single image + mask，输出 mesh + 6-DoF pose。它学的是 shape $x^s$ 和 pose $x^p$ 的联合分布 $p_\theta(x^s, x^p | c)$，对 occlusion 和 low res 都 robust（因为训练时见过）。

**核心观察**：shape 和 pose 在 SAM 3D 里 share 同一个 latent space。Rigid object 的 shape 是不变的，所以可以在 anchor frame 把 shape $\bar{x}^s$ 钉死，每帧只更新 pose。这就把"per-frame 生成不同 mesh + incoherent pose"的悲剧变成了"per-frame 只追 pose + shape temporal coherent"的可控问题。

### 怎么"只更新 pose"——Guided Diffusion

SAM 3D 用 flow matching 做 generation。采样从 noise $x_0 \sim \mathcal{N}(0, I)$ 开始，沿线性路径 $x_t = (1-t)x_0 + t x_1$ 积分 ODE $\dot{x} = v_\theta(x_t, t, c)$ 到达 sample $x_1$。每一步是 free Euler update：$x_t \leftarrow x_{t-\Delta} + \Delta \cdot v_\theta$。

但 free update 会把 shape 也一起更新，因为生成模型不知道你"想 fix shape"。借鉴 RePaint（diffusion inpainting 的经典方法）的思路：每一步 free denoise 之后，把 shape block 和 pose block 各自 blend 向一个 target interpolant：

$$
x_t^s = (1-\alpha_s) \underbrace{(x_{t-\Delta}^s + \Delta v_\theta^s)}_{\text{free denoise}} + \alpha_s \underbrace{z_{\text{ref}}^s(t)}_{\text{blend target}}
$$

$$
z_{\text{ref}}^s(t) = (1-t)\epsilon^s + t \bar{x}^s
$$

变量含义：
- $x_t^s$：shape block 在 diffusion time $t$ 的状态
- $v_\theta^s$：velocity field 的 shape 分量
- $\alpha_s$：blend strength，paper 取 0.9-1.0
- $\epsilon^s$：shape block 的初始 noise
- $\bar{x}^s$：anchor frame 的 shape（fix 死）

pose block 完全对称，只是把 $\bar{x}^s$ 换成前一帧的 pose $x_{k-1}^p$，blend strength 用 $\alpha_p$。

**直觉**：每一步生成模型自由探索，但探索完之后被"拉回"已知信息附近。Shape 拉向 anchor，pose 拉向前一帧——这样 shape temporal coherent，pose 也 temporal coherent。$\alpha$ 大就是"我更信已知信息"，$\alpha$ 小就是"我更信生成模型"。

### Adaptive $\alpha_p$：从数据里读出来

固定 $\alpha_p$ 麻烦：太大→快速旋转时跟丢（rigidly 锁住前一帧 pose，但物体其实转了），太小→静止时 pose 漂移（生成模型 free 探索，noise 散到 SE(3) 各处）。

作者从 2D point tracks（BootsTAPIR）估计物体在 image plane 的 in-plane rotation $\Delta\theta_k$，然后：

$$
\alpha_p(k) = \max(0.1, 0.7 - 0.09|\Delta\theta_k|)
$$

旋转大→$\alpha_p$ 小→让生成模型自由探索；旋转小→$\alpha_p$ 大→lock 住。代价是每 video 多一次 offline point tracking pass，但收益明显（Table 5 adaptive > fixed）。

### Per-frame 采样选最好的

Guided diffusion 是 stochastic 的，每帧跑一次得到一个 pose，但可能这一帧的 sample 不太对。每帧采 $N=25$ 个候选 pose，挑一个最好的。

"最好的"理论上该按生成模型自己的 conditional log-density 排：

$$
\log p_\theta(x_{k,i}^p | \bar{x}^s, c_k) = \log p_0(x_0^p) + \int_0^1 \text{tr}\left(\frac{\partial v_\theta^p(x_t, t, c_k)}{\partial x_t^p}\right) dt
$$

但 flow model 的 density 计算要 vector-Jacobian product per pose coordinate per Euler step——$D_p=13$ 维 pose × $T=25$ 步 × $N=25$ candidates ≈ 8700 forward+backward passes per frame，比 generation 本身慢 100 倍，video scale 上 prohibitive。

作者用 clustering 启发式：25 个 samples 在 weighted SE(3) distance 下 cluster，丢掉小 cluster（outliers），剩下按 2D silhouette IoU 排序选最佳。

$$
d(x_i^p, x_j^p) = w_t \|t_i - t_j\|_2 + w_r \cdot 2\arccos|\langle q_i, q_j\rangle|
$$

**直觉**：高置信度的 samples 会聚到同一 mode（因为正确答案只有一个），estimator noise 会散到 SE(3) 各处。Cluster 大小 = confidence proxy。这比 likelihood ranking 快 30×，质量相当（Table 5）。

### Hand-Object Alignment

HaWoR 输出 near-metric 的 hand mesh，SAM 3D 输出的 object 在另一个 scale。怎么 align？

把 hand scale 当 ground truth，scale object 到 match。核心算式：

$$
k = \frac{z_{\text{hand}}^H}{z_{\text{hand}}^M}, \quad \mathbf{obj}_{\text{target}} = \mathbf{c}_{\text{hand}}^H + k(\mathbf{c}_{\text{obj}}^M - \mathbf{c}_{\text{hand}}^M)
$$

- $\mathbf{c}_{\text{hand}}^H$：HaWoR hand mesh visible 部分的 centroid
- $\mathbf{c}_{\text{hand}}^M, \mathbf{c}_{\text{obj}}^M$：MoGe pointmap 下 hand pixels 和 object pixels 的 3D centroid
- $k$：从两个 space 的 hand 深度比得到 scale ratio
- $\mathbf{obj}_{\text{target}}$：把 MoGe space 下 hand-to-object 的相对位移 rescale 到 HaWoR space 后的目标位置

然后沿 viewing ray 滑动 object mesh 直到 match target depth（closed-form least squares，1D 优化）。每帧独立做，最后 GeoCalib 对齐重力。

---

## 第二步：Retargeting 的人话版

### 整体框架

Reference 是 4D hand-object trajectory——但 incomplete：
- 人手到 robot hand 形态不同，fingertip 位置不能直接 copy
- 没有 contact、force 信息，纯 kinematic signal
- Reconstruction 出来的 trajectory 有 noise、有 temporal discontinuity、hand-object 可能 misalign

之前两条路都不够好：
- **Kinematic retargeting**（mink, Pyroki, AnyTeleop, Geometric Retargeting）：解几何/joint 优化，但不考虑 force，容易 penetration / fingertip sliding / grasp 不稳
- **Dynamics-aware via RL**（ManipTrans, DexMachina, Dexplore）：需要 per-task reward engineering，不通用
- **Dynamics-aware via sampling**（SPIDER, ExoStart, Trajectory Optimization）：general-purpose，但之前都假设 clean MoCap reference

DO AS I DO 在 SPIDER 的 MPPI-style sampling-based optimization 框架上加了三个新组件，专门解决 noisy reference 的问题。

### MPPI 基本结构（recall 一下）

每 0.5s planning 一次，horizon 3s，sim_dt 0.005s（200 Hz）。每次 sampling 1024 个 candidate control sequences，roll out 在 MuJoCo Warp（GPU-parallel implementation built on NVIDIA Warp）上，eval reward，32 iterations 优化。Kernel annealing：早期 broad exploration，后期 local refinement。

### 三个新组件

#### 1. Warmup Steps——解决"第一帧就死"

Reference trajectory horizon $H$ 的第一帧可能 noisy 得离谱——比如手和物体 reconstruction 给的相对位置完全错，object 实际上根本没被抓在手里。如果直接从这个 state 开始 track reference，optimizer 怎么探索都救不回来。

而且 annealed sampling 对前 $H$ 步 exploration 不够——因为它们只出现在 rollout horizon 的开头，kernel 已经 anneal 到 local refinement 时这些 step 早就过去了。

**解决方案**：prepend 额外 $H$ 步 warmup 到 reference 前。Warmup 期间 object 用 weld constraint 固定在空中（mid-air），robot hand 自由移动；warmup 结束后 weld release，正常 sim。

**直觉**：让 robot 先"找到"一个 stable grasp pose 把 object 抓好，再开始 track 后面的 reference。Optimizer 在 warmup 期间自然把 hand pose 调整到能稳稳抓住 mid-air object 的位置，drop weld 之后 hand 已经稳 hold object，后续 track 就顺利了。

**不需要 grasp sampling 或 task-specific heuristics**，纯靠 core optimization procedure 自己 discover。从 Table 3 看，reconstruction 数据上 success rate 从 25% 直接到 66%——单 component 最大贡献。

#### 2. Random Force Perturbation——解决"看似稳其实不稳"

Optimizer 可能找到 local minima：用 fingertip 顶住 object 中心保持平衡，看起来在 track reference，但稍微扰动一下就崩。这种 solution 不 robust，real-world 跑一定失败。

借鉴 sim-to-real domain randomization（OpenAI Rubik's cube, Rudin et al. walking in minutes），在 sampling rollout 中 inject random force/torque：

- `perturb_force_scale`: 0.5
- `perturb_torque_scale`: 0.5  
- `perturb_prob`: 0.05（per timestep）
- `perturb_continue_prob`: 0.95（force 持续概率）

**直觉**：stable grasp 应该对 minor disturbance 不敏感。如果一个 grasp 一推就掉，说明它根本不是 grasp，是 balancing trick。Explicit inject perturbation 让 optimizer 被迫找 robust 的 solution，而不是 fragile 的 local minima。

Quantitative metrics 提升小（0.66→0.67），但 qualitative 上"grasp 更 natural"——这种 metric 不容易量化但 deployment 上重要。

#### 3. Transition Reward——解决"关键 binary event 没发生"

Object 从"rest 状态"到"in-hand 状态"的 transition 是 task 的 critical inflection point——pickup 成功了吗？place 成功了吗？但 pure tracking reward 是 L2 距离，太 soft，对这种 binary event 表达力不够。Reference noisy 时更糟：tracking reward 看着还 OK，但 object 其实没被 pickup。

**解决方案**：加 constant penalty：
- Reference timestep 是 rest state（hand-object distance > $\epsilon$）但 object-floor 没 contact → penalty
- Reference timestep 是 in-hand state（hand-object distance < $\epsilon$）但 hand-object 没 contact → penalty

Penalty scale = 3000.0，远大于其他 reward terms（pos_rew_scale=1.0, rot_rew_scale=0.3, joint_rew_scale=0.01）。

**直觉**：transition 是 hard binary event，应该 hard 约束。Tracking reward 是 smooth L2，对"object 在地上还是被拿起 5cm"这种 binary 区分力不够——5cm 的 L2 error 在 tracking reward 里很小，但语义上是质的差别。

### 其他细节

1. **Reference Blending**：每次 planning 把 previous plan 的 controls 和 next chunk of reference concatenate。Naive concatenation 会 jerky，用 interpolation blend 过渡
2. **Robust Kinematic Retargeting**：先用 mink 做 kinematic retargeting（匹配 fingertip positions，不考虑 force）生成 reference。从不同 random initial poses 跑多次取最好的，避免 kinematic local minima
3. **Object Base Plate**：有些 video 里 object 不 lying flat（spoon 直立在容器里）。给 object mesh 底部加 flat "plate"（只和 floor contact，不和 robot contact），让任意 initial pose 都能稳定 sim 起步
4. **Object Mesh Processing**：CoACD 做 convex decomposition，对 many-contact task mesh 加厚膨胀 2mm 稳定 contact

---

## 实验结果——数字说了什么

### Reconstruction：SOTA on 标准 benchmark

DexYCB（160 videos）和 HOI4D（12 videos），用 GT hands isolate object performance：

| Method | DexYCB F-5 | DexYCB F-10 | DexYCB CD | HOI4D F-5 | HOI4D F-10 | HOI4D CD |
|---|---|---|---|---|---|---|
| Joint methods（HO/MCC-HO/G-HOP）| 0.24-0.36 | 0.48-0.60 | 3.74-8.11 | 0.28-0.69 | 0.51-0.91 | 0.63-3.86 |
| FoundationPose | 0.69 | 0.89 | 0.89 | 0.71 | 0.91 | 0.49 |
| **Ours** | **0.71** | **0.93** | **0.66** | **0.72** | 0.91 | 0.49 |

Joint reconstruction 方法天花板低，dedicated tracker 在 lab 数据上 OK 但 in-the-wild 崩。Ours 在两个维度都好。

### In-the-wild 人类评估：FPose 被吊打

150 videos，3 raters per video，Ours 67% preferred vs FPose 18%，ties 15%。Non-tie win rate 79%。Fleiss' $\kappa = 0.65$（substantial agreement）。

**为什么 FPose 在 in-the-wild 不行**：它依赖 visual evidence + 已有 reference，occlusion/motion blur 时 visual evidence degrade 它就 lose lock 重新 acquire 不了。Generative prior（SAM 3D）的 strong shape+pose prior 在 weak signal 时反而能 maintain——prior 帮你"猜出"被挡住的物体 pose。

### Retargeting：三个组件逐步加，逐步涨

| Method | Recon Success | OakInk2 Success |
|---|---|---|
| Annealed Sampling (SPIDER baseline) | 0.25 | 0.72 |
| + Warmup | 0.66 (+41) | 0.77 (+5) |
| + Perturbation | 0.67 | 0.79 |
| + Transition Reward | **0.71** | **0.81** |

OakInk2 是 clean MoCap，baseline 已经 72%——说明 warmup 不仅对 noisy reference 有用，对 clean reference 也 boost。Reconstruction 数据 baseline 25% 说明 noisy reference 真的难，warmup 是 game-changer。

### Real-World Deployment：500 trajectories，10 tasks 跑真机

10 个 tasks：whisking, pouring, dusting, squeezing, tamping, erasing, stirring, hammering, spreading, picking。Dual UR3e arms + Sharpa Wave hands (22-DoF), 50 Hz commanded。500 个 human-verified trajectories 来源：53% internet, 31% egocentric, 16% generated。

这是**第一次**从 internet video → real dexterous hand rollout 的完整 pipeline。

---

## 最 practical 的部分——Human Data Filtering Playbook

Section 4.5 是 hidden gem，对做 robot data scaling 的人极有价值。

从 100DOH（CVPR 2020，已经 filtered 过 hand-object interaction 的 dataset）采 2000 个 10s clips：

| 失败原因 | clips |
|---|---|
| 没有真正 hand-object interaction | 1813 (91%) |
| Hand/object 出画面 | 41 |
| 无 activity 或跨 shot | 29 |
| Camera motion 太剧烈 | 14 |
| SAM 3D 失败 | 10 |
| 其他 | 10 |
| **Survive** | **83 (4%)** |
| 最乐观估计可用 | ~107 (5%) |

**Implication**：从 internet video 学 robot manipulation，**20× data penalty**。1000 小时 filtered internet video ≈ 50 小时 teleoperation data 的 utility。

如果你在做 robot data scaling 的 planning，这个数字得算进去。Internet video 不是"免费的"——filtering labor、compute、failure mode 处理都是成本。但即使 20× penalty，internet video 还是比 teleoperation 便宜一个量级（teleoperation $100/h 量级，filtered internet video 估计 $5-10/h 量级）。

而且这个 penalty 主要来自现有 foundation model 的 limit（SAM 3D 失败、camera motion 处理差）。这些会随 model 改进降低，penalty 会缩小。3 年前这个 penalty 可能 100×，现在 20×，3 年后可能 5×——trend 是 internet data 越来越 usable。

---

## 这个 paper 的真正 contribution 是什么

不是某个 algorithm 特别 novel——guided diffusion tracking、MPPI、warmup、perturbation 都是 known techniques 的新组合。真正 contribution 是：

1. **第一个 end-to-end pipeline 把 internet video 跑到 real dexterous hand**——这个 milestone 之前没人 reach
2. **系统性地诊断了每个环节的 failure mode**，并给出针对性 fix——warmup 对应 first frame 不可恢复、perturbation 对应 fragile grasp、transition reward 对应 binary event
3. **Filtering playbook 给出了实用的 data scaling 数字**——20× penalty 不是 hand-wavy 估计，是从 100DOH 实际测出来的
4. **Modular design 让每个 component 可独立改进**——SAM 3D 改进了，object tracking 直接 benefit；MoGe 改进了，alignment 直接 benefit；physics simulator 改进了，retargeting 直接 benefit

这种 paper 的价值在"打通"而非"发明"。Robot learning 从 teleoperation-only 走向 internet-video-as-first-class-data-source 的过程中，这种 pipeline-level 工作 critical。

---

## 我自己的几点 intuition

你之前讲过 "Software 2.0" 和 "Software 3.0" 的想法。这个 paper 在某种意义上是 "Robot Data 2.0" 的 case study——robot data 不再来自 curated teleoperation/sim dataset，而是来自 in-the-wild internet video，通过一 stack 的 foundation model "编译"成 executable form。

Pipeline 的每一环都依赖 2024-2025 的 model：
- 没有SAM 3D，image-to-3D 生成质量不够 robust
- 没有 MoGe-2，metric depth 不可靠，alignment 就崩
- 没有 HaWoR，in-the-wild hand tracking 不存在
- 没有 MuJoCo Warp，1024 samples × 32 iterations × 3s horizon 的 MPPI 跑不动
- 没有 BootsTAPIR，adaptive $\alpha_p$ 没法做
- 没有 SAM 3，segmentation 在 occlusion 下不 robust

每个 component 半年前都不可用。这种 paper 是 model 进步的"集成释放"——某个时间点，所有 component 都过了可用阈值，pipeline 就突然可行了。类似的 trend 在 vision-language model、robot VLA 也在发生。

**关于 hand-object alignment 的 ambiguity**：公式 $\mathbf{obj}_{\text{target}} = \mathbf{c}_{\text{hand}}^H + k(\mathbf{c}_{\text{obj}}^M - \mathbf{c}_{\text{hand}}^M)$ 假设 MoGe pointmap 的 hand-to-object 相对距离正确。但单目 depth 在 hand 遮挡区域最 unreliable——这正是 occlusion 最严重的地方。可能下个 paper 会用 multi-view、temporal consistency、或 contact-based 优化来 refine 这部分。

**关于 scaling**：500 trajectories 听起来不多，但 pipeline 第一次跑通后，scaling 是工程问题——更多 compute、更多 video、更好 foundation model 就能线性 scale。从 500 到 50k 到 500k trajectories，每个阶段解锁的下游 capability 不一样。500k in-the-wild dexterous trajectories 足以 train 一个非常 generalist 的 dexterous policy。

**关于 scene-level reasoning 的 limit**：作者老实承认只重建 hand+object 不重建 scene。但很多 task 的 intention 通过 hand-scene interaction 表达（杯子放到架子特定槽位、刀切菜用砧板支撑）。下个 obvious paper 是把 scene reconstruction 加进 pipeline——NeRF/Gaussian Splatting、scene graph、articulated object 都已经在快速发展。

**关于 sim2real gap**：作者说 MuJoCo/Isaac 是 approximate dynamics，给 achievable performance 设了 upper bound。但 paper 中 real-world deployment 看起来挺顺——可能因为这些 task 主要是 kinematic tracking，contact dynamics 不极端。如果要加 high-force task（hammering hard, drilling），sim2real gap 会暴露。

---

## 参考链接

- **项目主页**: https://do-as-i-do.com
- **SAM 3D**: https://arxiv.org/abs/2511.16624
- **SPIDER (retargeting framework)**: https://arxiv.org/abs/2511.09484
- **FoundationPose**: https://arxiv.org/abs/2312.08344
- **HaWoR**: https://arxiv.org/abs/2411.17494 (CVPR 2025)
- **MoGe-2**: https://arxiv.org/abs/2507.02546
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **RePaint (diffusion inpainting)**: https://arxiv.org/abs/2201.09865
- **MuJoCo Warp**: https://mujoco.readthedocs.io/en/latest/mjwarp/
- **NVIDIA Isaac Sim**: https://developer.nvidia.com/isaac/sim
- **SAM 3**: https://arxiv.org/abs/2511.16719
- **BootsTAPIR**: https://arxiv.org/abs/2402.00847
- **GeoCalib**: https://arxiv.org/abs/2409.06704
- **DexYCB**: https://dex-ycb.github.io
- **HOI4D**: http://www.hoi4d.github.io
- **OakInk2**: https://arxiv.org/abs/2403.19417
- **100DOH**: http://fouheylab.eecs.umich.edu/~dandeng/projects/100doh.html
- **CoACD**: https://arxiv.org/abs/2104.04569
- **mink (MuJoCo IK)**: https://github.com/kevinzakka/mink
- **Any6D**: https://arxiv.org/abs/2501.08970
- **G-HOP**: https://arxiv.org/abs/2403.00031
- **MCC-HO**: https://arxiv.org/abs/2404.06507
- **EgoScale**: https://arxiv.org/abs/2602.16710
- **EgoVLA**: https://arxiv.org/abs/2507.12440
- **Being-H0**: https://arxiv.org/abs/2507.15597
- **EgoDex**: https://arxiv.org/abs/2505.11709
- **DreamDojo**: https://arxiv.org/abs/2602.06949

一句话总结这个 paper：它把"robot data 从哪来"这个 bottleneck 问题的答案从 teleoperation 转向 internet video，并给出了第一个完整可跑通的 pipeline。每个环节都不算 breakthrough，但串在一起解锁了之前做不到的事——这才是真正的 system-level contribution。

---

# Do as I Do: 从日常人类视频到灵巧机械手操作数据

## 核心动机与问题定位

这篇paper试图回答一个非常fundamental的问题：robot learning的data bottleneck怎么破？传统方案teleoperation受限于operator expertise、cost、mechanical transparency；simulated exploration需要设计diverse environments和reward functions。Internet上的人手操作视频是规模最大、最便宜的observational data，但存在两个critical gap：

1. **Recognition/Reconstruction gap**：从monocular RGB重建hand-object interaction难，特别是in-the-wild noisy videos下object pose tracking容易drift
2. **Embodiment gap**：人手和robot hand的kinematics/dynamics差距巨大，kinematic retargeting不考虑forces容易导致penetration、fingertip sliding、grasp instability

DO AS I DO给出的方案是两阶段pipeline：先把RGB video通过vision foundation models重建为4D hand-object trajectory，再通过GPU-parallel physics simulator (MuJoCo Warp / Isaac)做sampling-based optimization，最终生成可执行的robot trajectories。

项目主页：https://do-as-i-do.com

---

## Reconstruction部分技术详解

### 模块化分解的设计选择

整个reconstruction pipeline做了modular decomposition：
- **Hand tracking**: HaWoR (world-space hand motion reconstruction from egocentric videos, CVPR 2025)
- **Object mesh generation**: SAM 3D (image-to-3D generative foundation model)
- **Object pose tracking**: 作者自己开发的guided diffusion方法
- **Depth + camera intrinsics**: MoGe-2 (monocular geometry with metric scale)
- **Segmentation**: SAM 3

为什么用modular而非joint hand-object reconstruction？Joint methods如G-HOP、MCC-HO在lab数据上有效，但in-the-wild视频上各模块的failure mode不一样，modular设计可以分别替换/改进每个组件。

### Guided Diffusion Object Tracking的核心创新

**关键观察**：SAM 3D学的是shape $x^s$ 和pose $x^p$ 的联合分布 $p_\theta(x^s, x^p | c)$，其中 $c$ 是single 2D image + object mask。Shape和pose share同一个latent space。这意味着可以fix shape为anchor frame的 $\bar{x}^s$，然后inference时只更新pose。

直接per-frame独立运行SAM 3D会得到per-frame不同的mesh（shape都不一样）和time-incoherent pose sequence。作者把image-to-3D生成模型重写为video object tracker。

### Flow Matching的Guidance机制

Flow matching（Lipman et al. 2023）通过积分ODE $\dot{x} = v_\theta(x_t, t, c)$ 采样，从 $x_0 \sim \mathcal{N}(0, I)$ 沿线性路径 $x_t = (1-t)x_0 + t x_1$ 到达样本 $x_1$。

借鉴RePaint (Lugmayr et al. 2022)和SDE-based inpainting的思想，作者在每个Euler step做"denoising + blending"混合：

$$
x_t^s = \underbrace{(1-\alpha_s)(x_{t-\Delta}^s + \Delta v_\theta^s)}_{\text{denoising}} + \underbrace{\alpha_s z_{\text{ref}}^s(t)}_{\text{blending}}
$$

$$
x_t^p = \underbrace{(1-\alpha_p)(x_{t-\Delta}^p + \Delta v_\theta^p)}_{\text{denoising}} + \underbrace{\alpha_p z_{\text{ref}}^p(t)}_{\text{blending}}
$$

变量含义：
- $x_t^s, x_t^p$：shape block和pose block在diffusion time $t \in [0,1]$ 的状态
- $\Delta$：Euler step size
- $v_\theta^s, v_\theta^p$：velocity field $v_\theta(x_{t-\Delta}, t-\Delta, c)$ 在shape和pose block上的分量
- $\alpha_s, \alpha_p \in [0,1]$：guidance strength，越大越向reference靠拢
- $z_{\text{ref}}^s(t) = (1-t)\epsilon^s + t\bar{x}^s$：shape target interpolant，$\epsilon^s$ 是shape block初始噪声，$\bar{x}^s$ 是anchor shape
- $z_{\text{ref}}^p(t) = (1-t)\epsilon^p + t x_{k-1}^p$：pose target interpolant，$\epsilon^p$ 是pose block初始噪声，$x_{k-1}^p$ 是前一帧pose

**Intuition**：这相当于classifier-free guidance的一种变体。Free Euler update是生成模型自己的prior预测，blend向 $z_{\text{ref}}$ 是把已知信息（fixed shape或前一帧pose）通过interpolant inject回去。$\alpha$ 越大越保守，越小越trust生成模型。

### Adaptive Guidance Parameters

固定 $\alpha_p$ 容易over-rigid（快速旋转时跟丢）或spurious flips（静止时仍允许pose漂移）。作者从2D point tracks (BootsTAPIR)估计object的in-plane rotation $\Delta\theta_k$：

$$
\alpha_p(k) = \max(0.1, 0.7 - 0.09|\Delta\theta_k|)
$$

旋转大时降低 $\alpha_p$，让生成模型自由探索；旋转小时提高 $\alpha_p$，保持temporal coherence。Shape guidance $\alpha_s \in [0.9, 1]$ 因为rigid object assumption。

### Per-frame Pose Sampling与Consensus Filtering

每帧采样 $N=25$ 个候选pose $\{x_{k,i}^p\}_{i=1}^N$。理想做法是按生成模型自己的conditional log-density排序。Flow model的density通过instantaneous change-of-variables精确计算（公式2）：

$$
\log p_\theta(x_{k,i}^p | \bar{x}^s, c_k) = \log p_0(x_0^p) + \int_0^1 \text{tr}\left(\frac{\partial v_\theta^p(x_t, t, c_k)}{\partial x_t^p}\right) dt
$$

变量含义：
- $x_0^p$：从 $x_{k,i}^p$ backward integrate ODE (从 $t=1$ 到 $t=0$) 得到的noise pre-image
- $\log p_0(x_0^p)$：base Gaussian distribution的log density
- 被积函数：velocity field对pose block的Jacobian的trace
- $D_p = 13$：pose dimensions（3 translation + 4 quaternion + 6 DoF冗余表示）
- $T = 25$：ODE steps
- $N = 25$：candidates per frame

成本：$N \cdot T \cdot (1 + D_p) \approx 8700$ forward+backward passes per frame，比generation本身慢两个数量级，video scale上prohibitive。

改用clustering-based heuristic，weighted SE(3) distance：

$$
d(x_i^p, x_j^p) = w_t \|t_i - t_j\|_2 + w_r \cdot 2\arccos|\langle q_i, q_j\rangle|
$$

变量含义：
- $t_i, t_j \in \mathbb{R}^3$：translations
- $q_i, q_j \in S^3$：unit quaternions
- $\langle q_i, q_j\rangle$：quaternion dot product
- $2\arccos|\langle q_i, q_j\rangle\rangle$：SO(3)上的geodesic angle（绝对值保证double cover一致性）

**Intuition**：confident samples聚集到同一pose mode，estimator noise散布到SE(3)各处。cluster+discard小cluster+按mask IoU排序，就恢复mode-best pose。比likelihood ranking快30×，质量相当（见Table 5）。

### Hand-Object Alignment

SAM 3D用MoGe pointmaps训练，inference时也用MoGe pointmap。HaWoR hand和SAM 3D object在不同scale上，需要align。

记号：
- $\mathbf{c}_{\text{hand}}^M, \mathbf{c}_{\text{obj}}^M$：MoGe pointmap space下的hand和object centroids
- $\mathbf{c}_{\text{hand}}^H$：HaWoR mesh space下visible hand部分的centroid
- $z_{\text{hand}}^H, z_{\text{hand}}^M$：两个space下hand centroid的z分量

Scale ratio: $k = z_{\text{hand}}^H / z_{\text{hand}}^M$

Target object position: $\mathbf{obj}_{\text{target}} = \mathbf{c}_{\text{hand}}^H + k(\mathbf{c}_{\text{obj}}^M - \mathbf{c}_{\text{hand}}^M)$

然后优化single scalar $s$ (translation scale)，沿viewing ray slide object mesh：

$$
s^* = \arg\min_s \|\mathbf{obj}_{\text{pos}}(s) - \mathbf{obj}_{\text{target}}\|^2 = \frac{\mathbf{t}^\top(\mathbf{obj}_{\text{target}} - \mathbf{c}_{\text{mesh}})}{\mathbf{t}^\top \mathbf{t}}
$$

其中 $\mathbf{obj}_{\text{pos}}(s) = \mathbf{c}_{\text{mesh}} + s\mathbf{t}$，$\mathbf{c}_{\text{mesh}}$ 是visible mesh vertices的centroid，$\mathbf{t}$ 是mesh的camera-frame translation。Closed-form解，每帧独立求解，最后用GeoCalib对齐重力。

---

## Retargeting部分技术详解

### 整体框架

基于Pan et al.的SPIDER (Scalable Physics-Informed Dexterous Retargeting, arxiv 2511.09484) 的MPPI-style sampling-based optimization。Simulation timestep 0.005s (200 Hz)，planning every 0.5s (2 Hz)，horizon 3s，每planning step evaluate 1024 samples，optimize 32 iterations。在MuJoCo Warp（GPU-parallel implementation built on NVIDIA Warp）上运行。

Kernel annealing: across both iterations和prediction horizon，从broad exploration退火到local refinement。

### 三个novel组件解决noisy reference问题

**Problem Statement**: 之前的dynamics-aware retargeting（SPIDER等）假设clean MoCap references。DO AS I DO处理reconstruction的noisy references，有temporal discontinuities和severe hand-object misalignments。

#### Component 1: Warmup Steps

Reference trajectory horizon $H$ 存在两个问题：
1. Noisy first frame可能让hand和object起始在unrecoverable state（例如object没被抓到）
2. Annealed sampling对前 $H$ 步exploration不充分，因为这些steps只在rollout horizon开始处出现

解决方案：prepend额外 $H$ warmup steps到reference前。Warmup期间object用weld constraint固定在空中（例如mid-air），robot hand自由移动；warmup结束后weld release，simulation正常进行。

**Intuition**：让robot先"找到"稳定grasp再开始tracking reference。不需要grasp sampling或task-specific heuristics，直接复用core optimization procedure。从Table 3看，warmup把reconstruction数据上的success rate从25%拉到66%，是最大的single-component贡献。

#### Component 2: Random Force Perturbation

Rollout horizon可能让optimization陷入local minima——例如短暂的unstable interaction能track但不可持续（用fingertip顶住object保持平衡）。

借鉴sim-to-real domain randomization（OpenAI Rubik's cube, Rudin et al.），在sample rollouts中inject random forces/torques。

Hyperparameters (Table 4):
- `num_perturb_samples`: 4
- `perturb_force_scale`: 0.5
- `perturb_torque_scale`: 0.5
- `perturb_prob`: 0.05 (per timestep)
- `perturb_continue_prob`: 0.95 (force持续probability)

**Intuition**：robustness应该被显式optimize。如果interaction对minor disturbance敏感，那它就不是stable grasp。General-purpose solution，不assume high-fidelity references（vs SPIDER的contact guidance）。

#### Component 3: Transition Reward

Object在"rest"和"in-hand"之间的transition是critical inflection points。但noisy reference下pure tracking reward太soft，无法鼓励step-function式interaction（rest时object-floor contact，in-hand时hand-object contact）。

添加constant penalty：
1. Reference timestep是rest state但object-floor没contact → penalty
2. Reference timestep是in-hand state但hand-object没contact → penalty

Reference stage由hand-object distance under threshold $\epsilon$ 划分。Penalty scale = 3000.0（Table 4），远大于其他reward terms（pos_rew_scale=1.0, rot_rew_scale=0.3）。

**Intuition**：transition是task的关键milestone，应该被hard约束。Tracking reward是smooth的L2距离，对于"成功pickup"这种binary event表达力不够。Hard penalty补偿这个gap。

### 其他implementation details

1. **Reference Blending**: 每个planning timestep，samples centered around previous plan的controls + next chunk of reference。Naive concatenation导致jerky motions，用interpolation blend optimized controls into reference。

2. **Robust Kinematic Retargeting**: Kinematic retargeting（用mink匹配fingertip positions）从不同random initial poses计算多次，避免local minima。Computational cheap，可以做多个seeds。

3. **Object Base Plate**: 有些video里object不lying flat on surface（例如容器里直立的spoon）。给object mesh底部加flat base "plate"，只和floor有contact不和robot contact。允许任意initial pose without task-specific assumptions。

4. **Object Mesh Processing**: 用CoACD (Approximate Convex Decomposition)做convex decomposition，对many-contact tasks mesh加厚/膨胀2mm稳定hand-object interaction。

---

## 实验结果深度解读

### Reconstruction (Table 2)

DexYCB (160 videos) 和 HOI4D (12 videos)，isolate object-level performance by supplying ground-truth hands。

| Method | DexYCB F-5 | DexYCB F-10 | DexYCB CD | HOI4D F-5 | HOI4D F-10 | HOI4D CD |
|---|---|---|---|---|---|---|
| HO [48] | 0.24 | 0.48 | 4.76 | 0.28 | 0.51 | 3.86 |
| MCC-HO [51] | 0.36 | 0.60 | 3.74 | 0.52 | 0.78 | 1.36 |
| G-HOP [54] | 0.31 | 0.49 | 8.11 | 0.69 | 0.91 | 0.63 |
| FoundationPose [17] | 0.69 | 0.89 | 0.89 | 0.71 | 0.91 | 0.49 |
| Any6D [47] | 0.69 | 0.88 | 0.97 | 0.71 | 0.91 | 0.50 |
| **Ours** | **0.71** | **0.93** | **0.66** | **0.72** | 0.91 | 0.49 |

F-5/F-10是Add-threshold metrics（5cm/5°和10cm/10°下的accuracy），CD是Chamfer Distance。

Joint reconstruction methods（HO, IHOI, HORSE, MCC-HO, G-HOP）显著弱于object trackers（FPose, Any6D）和Ours。这说明joint reconstruction在lab data上competitive但approach的天花板低。Ours比dedicated trackers在DexYCB上明显领先（CD从0.89降到0.66，F-10从0.89到0.93），HOI4D上持平。

### In-the-wild Human Evaluation (Fig 5)

150 videos (internet + egocentric + generated)，3 raters per video。Ours: 67% preferred, FPose: 18%, ties: 15%。Non-tie judgments下win rate 79%。Fleiss' $\kappa = 0.65$ (substantial agreement), 75% videos获unanimous agreement。

**Intuition**：FPose依赖visual evidence和reference，in-the-wild motion blur/occlusion下容易lose lock。Guided diffusion的strong prior在弱signal时也能maintain pose。Dedicated tracker的优势在clean数据下放大，noisy数据下反而不如generative prior。

### Retargeting (Table 3)

Reconstruction dataset (655 trajectories) 和 OakInk2 (1352 bimanual MoCap trajectories)：

| Method | Recon Success | Recon Pos | Recon Rot | OakInk2 Success | OakInk2 Pos | OakInk2 Rot |
|---|---|---|---|---|---|---|
| Annealed Sampling | 0.25 | 0.08 | 0.40 | 0.72 | 0.08 | 0.32 |
| + Warmup | 0.66 | 0.06 | 0.28 | 0.77 | 0.06 | 0.25 |
| + Perturbation | 0.67 | 0.06 | 0.30 | 0.79 | 0.03 | 0.14 |
| + Transition Reward | **0.71** | **0.05** | **0.28** | **0.81** | 0.03 | 0.15 |

Success定义：mean position error $E_{\text{pos}} < 0.1$m 且 mean rotation error $E_{\text{rot}} < 0.5$ rad。

**Key insight**：在reconstruction数据上baseline只有25%，warmup直接拉到66%（+41%）——noisy reference的最大问题是initial state不可恢复，warmup解决这个。OakInk2上baseline已经72%（clean MoCap data），warmup仍能加5%——说明warmup对所有reference都有用，noisy references上贡献更大。Perturbation在quantitative metrics上marginal但在qualitative上明显改善（更natural grasps）。Transition reward专门解决pickup/place的binary event失败。

### Ablation: Object Tracking (Table 5)

| Pose Guidance | Candidate Selection | DexYCB F-10 | HOI4D F-10 |
|---|---|---|---|
| Fixed | Clustering | 0.91 | 0.91 |
| Adaptive | Random | 0.91 | 0.87 |
| Adaptive | Log-likelihood | 0.93 | 0.91 |
| Adaptive | Clustering | 0.93 | 0.91 |

Adaptive guidance > Fixed (在HOI4D上0.91 vs 0.91，但CD和F-5略好，detailed见原文)；Clustering ≈ Log-likelihood，但快30×。

### Human Data Filtering Playbook (Section 4.5)

从100DOH (CVPR 2020的hand-object interaction dataset)采样2000个10秒clips：
- 只有187 (9%) 有meaningful hand-object interaction
- 41 clips hand/object超出video boundary
- 29 clips 没activity或activity跨越shot boundaries
- 14 clips 因camera motion失败
- 10 clips 因SAM 3D失败
- 10 clips 其他原因
- 最终83 (4%) 通过quality check
- 最乐观估计107 clips (5%) directly relevant

**Implication**：不做preprocessing直接从internet videos学，有20×的data penalty。这对于scaling robot data是critical practical insight。

### Real-World Deployment

500个human-verified trajectories (53% internet, 31% egocentric, 16% generated)。10个tasks部署到dual UR3e arms + Sharpa Wave hands (22-DoF)，50Hz commanded。Grasp types包括writing tripod, power, ventral, parallel extension。

Manual align初始pose (x, y, z, yaw) with robot workspace in simulation，然后arm IK + real-world execution。10个tasks: whisking, pouring, dusting, squeezing, tamping, erasing, stirring, hammering, spreading, picking。

---

## 与相关工作的定位对比

Table 1总结了related work：

| Method | Reconstruction | Retargeting | Self | Gen | Ego | Internet |
|---|---|---|---|---|---|---|
| H2Sim2Robot | LiDAR Scan + FPose | RL | √ | | | |
| VideoManip | MeshyAI + FPose | DRO + DP3 | √ | √ | | |
| DexMan | TRELLIS + FPose + SpaTrack | RL | √ | √ | | |
| DexImit | SAM 3D + FPose++ | Motion planning | √ | √ | | |
| **Ours** | SAM 3D + Guided Diffusion | Sampling-based opt. | √ | √ | √ | √ |

DO AS I DO是唯一支持全部四种data sources（self-collected, generated, egocentric, internet）的方法。Reconstruction用了guided diffusion而非现成FPose，retargeting用sampling-based opt.而非RL或motion planning。

**RL-based retargeting** (H2Sim2Robot, DexMan, ManipTrans) 通常需要reward engineering，对每个task设计奖励；**Sampling-based** (SPIDER, DO AS I DO) 更general-purpose，对reference tracking的reward足够。

**Motion planning** (DexImit) 不ensure physical plausibility，容易lose generality。

### 与EgoZero, EgoScale, EgoDex, EgoVLA, Being-H0的对比

近期大量egocentric human video工作：
- EgoZero (arxiv 2505.20290): smart glasses数据
- EgoScale (arxiv 2602.16710): scaling dexterous manipulation with egocentric data
- EgoVLA (arxiv 2507.12440): VLA from egocentric videos
- Being-H0 (arxiv 2507.15597): VLA pretraining from large-scale human videos
- EgoDex (arxiv 2505.11709): dexterous manipulation from large-scale egocentric video

这些通常用smart glasses或egocentric rig，做representation/policy pretraining。DO AS I DO不同：直接产生playable trajectories，且支持exocentric internet videos，data source更广。

### 与DreamDojo, World Models for Dexterous的对比

- DreamDojo (arxiv 2602.06949): generalist robot world model from human videos
- World Models for Dexterous (arxiv 2512.13644): LeCun等人的world model路线

World model路线learn dynamics而非直接产生playable trajectories。DO AS I DO是data generation pipeline，输出executed controls。

---

## Limitations和未来方向

作者明确列出的limits：

1. **Rigid object assumption**: SAM 3D生成rigid mesh，articulated/deformable object不行。这是foundation model层面的limit，需要articulated 3D generation或neural object representations。

2. **Semi-accurate metric depth**: 依赖MoGe-2的metric depth。Extreme场景（low texture, transparent surface）depth fail会propagate到alignment。

3. **Monocular contact ambiguity**: 从单目RGB区分visual occlusion和physical contact很难。会导致retargeting把"靠近"误解为"接触"。

4. **Scene-level reasoning缺失**: 只重建hand和object，不重建full scene（obstacles, articulations, table geometry）。Human intention很多通过hand-scene interaction表达（例如把杯子放到架子特定位置）。这是limit，因为scene-level reasoning对真正的manipulation task critical。

5. **Simulator fidelity**: MuJoCo Warp和Isaac对real world dynamics只是approximate。Contact dynamics、friction、deformation等sim2-real gap给achievable performance设了上限。

### 我的额外思考

**关于hand-object alignment的ambiguity**: 公式 $\mathbf{obj}_{\text{target}} = \mathbf{c}_{\text{hand}}^H + k(\mathbf{c}_{\text{obj}}^M - \mathbf{c}_{\text{hand}}^M)$ 假设MoGe pointmap的相对距离正确。但单目深度在hand-object interaction区域往往unreliable（occlusion最严重的地方）。可能需要multi-view或temporal consistency约束。

**关于Retargeting的scability**: MPPI每planning step 1024 samples × 32 iterations × 3s horizon × 200Hz = ~2M simulator steps per planning step。GPU-parallel simulator是必须的。整个pipeline从video到playable trajectory可能几分钟到几十分钟per video，scaling到millions of videos需要进一步优化。

**关于Evaluation的simplification**: Reconstruction在DexYCB/HOI4D用GT hands isolate object performance。但real pipeline中hand估计也有noise，会propagate到object alignment。End-to-end evaluation可能更informative但更难设计。

**关于Foundation Models的role**: SAM 3D, MoGe-2, HaWoR, SAM 3, BootsTAPIR, GeoCalib都是2024-2025的model。这反映modern robot learning pipeline越来越依赖vision foundation models。任何component的改进直接benefit整个pipeline。Pace of progress非常快——半年前SOTA还不可能做到in-the-wild reconstruction。

**关于Filtering Playbook的implication**: 20× data penalty意味着要从internet学robot manipulation，1000小时filtered internet video ≈ 50小时teleoperation data的utility。如果teleoperation cost是$100/h，filtered internet video cost约$5/h（算上filtering labor和compute）。还是便宜，但不是"free data"那么便宜。

---

## 参考链接

- 项目主页: https://do-as-i-do.com
- SAM 3D: https://arxiv.org/abs/2511.16624
- SPIDER (Pan et al., retargeting framework): https://arxiv.org/abs/2511.09484
- FoundationPose: https://arxiv.org/abs/2312.08344
- HaWoR: CVPR 2025, world-space hand motion reconstruction
- MoGe-2: https://arxiv.org/abs/2507.02546
- Flow Matching: https://arxiv.org/abs/2210.02747
- RePaint (diffusion inpainting): https://arxiv.org/abs/2201.09865
- MuJoCo Warp: https://mujoco.readthedocs.io/en/latest/mjwarp/
- NVIDIA Isaac Sim: https://developer.nvidia.com/isaac/sim
- SAM 3: https://arxiv.org/abs/2511.16719
- BootsTAPIR (point tracking): https://arxiv.org/abs/2402.00847
- GeoCalib: https://arxiv.org/abs/2409.06704 (ECCV 2024)
- DexYCB benchmark: https://dex-ycb.github.io
- HOI4D: CVPR 2022
- OakInk2: https://arxiv.org/abs/2403.19417
- 100DOH: http://fouheylab.eecs.umich.edu/~dandeng/projects/100doh.html
- CoACD (convex decomposition): https://arxiv.org/abs/2104.04569 (作者引用[78])
- mink (MuJoCo IK): https://github.com/kevinzakka/mink
- Any6D: CVPR 2025, https://arxiv.org/abs/2501.08970
- G-HOP: CVPR 2024, https://arxiv.org/abs/2403.00031
- MCC-HO: https://arxiv.org/abs/2404.06507

这篇paper代表了2025-2026 robot learning领域一个清晰的trend：用vision foundation models把observational data (internet videos) 转化为experiential data (robot trajectories)，用GPU-parallel physics simulator解决retargeting的feasibility。Pipeline的每个环节都依赖最近的model breakthrough，整个pipeline在6个月前还不可能。Filtering playbook揭示了"internet video是robotics的free data"这个narrative的实际代价——20× penalty，但仍比teleoperation便宜得多。Limitations部分诚实指出rigid object和scene-level reasoning的gaps，这些是下一个paper的obvious target。
