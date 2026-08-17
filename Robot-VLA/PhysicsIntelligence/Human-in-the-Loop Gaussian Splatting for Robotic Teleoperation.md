---
source_pdf: Human-in-the-Loop Gaussian Splatting for Robotic Teleoperation.pdf
paper_sha256: 53700c0bd94aac571717f38339b8d6355ed98c80fadbf78e1453bf3fff322580
processed_at: '2026-08-05T07:42:31-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 HIL-GS

## 一句话总结

让操作员戴上 VR 头显，看着机器人边走边建出来的 3D Gaussian Splatting 地图，用手势告诉机器人"下一个该去哪拍照"，机器人照做，地图越来越清晰——这个过程一直循环，直到任务完成。

---

## 为什么传统 teleoperation 很难用

想象你在操控一个危险环境里的机器人。你手头只有一个 live camera stream，就像拿着一个手电筒在黑暗仓库里摸索：

- 你只能看到正前方一小块区域，旁边的 obstacle 完全 invisible
- 屏幕是 2D 的，你根本判断不出"那个阀门离机械臂还有几厘米"
- camera 焊死在 robot body 上，你想绕到目标物体背后看看？没门

这种 tunnel vision 在民用场景勉强能用，在工业现场比如炼油厂、核电站就是灾难——operator 碰一下管道都可能引发事故。

---

## 为什么用 3D Gaussian Splatting

NeRF 也能做 photorealistic 重建，但有两个致命问题：

1. **太慢**：offline optimization 要几分钟，rendering 一个新视角要几秒
2. **太 rigid**：要求你从很多角度拍一圈，这在 teleoperation 里根本做不到

3DGS [1] 解决了速度问题——explicit Gaussian primitive 可以 real-time rendering。但 dense GS 仍然需要很多 multi-view image，在 teleoperation 里收集这些 view 是真正的 bottleneck。

这就是 HIL-GS 要解决的问题。

---

## 核心思路：把 view selection 的决策权交给人类

Autonomous active-sensing planner 听起来很美——让算法自己决定下一个最优观测点。但在真实 cluttered industrial environment 里：

- Splat-Nav [2] 这种 state-of-the-art planner 在密集管道环境里频繁 path planning failure，需要 human 手动加 intermediate waypoint
- ActiveSplat [3] 假设 2.5D flat floor，sim-only 代码，deploy 困难
- Algorithm 不知道哪个 valve 是 task-critical，可能在 irrelevant region 浪费 view

Human operator 有 semantic understanding。他能看到当前 GS map 里哪块重建得糊，知道任务目标在哪里，能判断下一个 viewpoint 既要 informative 又要 safe。这正是 human-in-the-loop 的价值。

---

## 系统怎么跑起来的

### Step 1: 机器人先建一个粗糙的初始地图

RGB-D camera 2Hz 传图像，proprioceptive sensor（IMU + joint encoder）提供 motion 信息，motion-aware GS reconstruction module 实时建图。

### Step 2: 操作员在 VR 里观察这个地图

Meta Quest 3 头显里，operator 看到：
- 实时更新的 3DGS 地图
- 灰色 voxel 表示还没观测到的区域
- 机器人的 ghost（半透明预测姿态）跟着右手 pinch 手势移动
- 如果 ghost 位置会 collision，整个 ghost 变红色

### Step 3: 操作员用手指选择下一个 viewpoint

- 右手 pinch-drag：translate view
- 双手 pinch-twist：rotate view
- 右手 pinch：触发 end-effector 跟随手指，MoveIt [4] 解 IK 显示 ghost
- 左手 pinch：fix ghost（冻结候选 pose）
- 再右手 pinch：execute；左手 pinch：cancel

### Step 4: 机器人执行，获取新数据，refine 地图

回到 Step 1，loop 继续。

---

## 技术上最关键的一块：Motion-Aware Sensor Fusion

### 问题：vision-only SLAM 在 teleoperation 里经常 fail

Teleoperation 的 robot motion pattern 跟 SLAM benchmark 完全不同：

- **Pure rotation**：operator 原地转 camera 扫一圈，frame-to-frame translation 几乎为零，parallax 消失，photometric loss 在 rotation 方向太 flat，optimizer 卡住
- **Fast linear motion**：motion blur + frame overlap 突降，correspondence 搜不到
- **Distant background**：炼油厂那种远处的天空、远山，depth noise 极大，parallax 几乎为零，triangulation 不可靠

GS-ICP SLAM [5] 是 base，但在这些 condition 下会 drift 甚至 diverge。

### Solution：用 proprioception 给 vision 打辅助

IMU 和 joint encoder 是**独立于 vision 的信号源**。在 vision fail 的时候，proprioception 仍然可靠。Paper 的核心 idea 是把 proprioception 用在两个 stage：

#### Stage 1: 给 G-ICP 一个好的 initial guess

公式(1)：
$$\Delta \xi_{\text{prop}} = \log(T_{\text{prev}}^{-1} T_{\text{prop}})$$

- $T_{\text{prev}} \in SE(3)$：上一 frame 的 refined pose
- $T_{\text{prop}} \in SE(3)$：当前 frame 通过 IMU + encoder forward kinematics 推算的 pose
- $\Delta \xi_{\text{prop}} \in \mathbb{R}^6$：这个 relative transform 的 Lie algebra 表示，前 3 维是 translation $\Delta \xi_t$，后 3 维是 rotation $\Delta \xi_R$

公式(2)：
$$X_{t^*} = \exp(\Delta \xi_{\text{prop}}) \cdot X_t$$

- $X_t$：当前 frame 的 raw point cloud（camera local frame）
- $\exp(\cdot): \mathbb{R}^6 \to SE(3)$：matrix exponential
- $X_{t^*}$：predicted target point cloud，已经 pre-align 到上一 frame frame 下

**Intuition**：G-ICP 是 iterative registration，需要 good initialization 才能收敛到 right basin。Vision-only 时 initial guess 是 identity，aggressive motion 下直接飞出 convergence basin。Proprioception 给的 prior 把 search 起点拉回正确区域。

#### Stage 2: G-ICP 跑出 $T_{\text{icp}}$ 后做 adaptive fusion

公式(5)算 discrepancy：
$$\Delta \xi = \log(T_{\text{icp}}^{-1} T_{\text{prop}}) \in \mathbb{R}^6$$

这表示 vision 结果和 proprioception 结果的 disagreement。分解为 $\Delta \xi_t$（translation error）和 $\Delta \xi_R$（rotation error）。

公式(6)：proprioception 的 uncertainty covariance
$$\Sigma_{\text{prop}} = \text{diag}(\Sigma_t, \Sigma_R)$$

- $\Sigma_t \in \mathbb{R}^{3\times3}$：translation covariance
- $\Sigma_R \in \mathbb{R}^{3\times3}$：rotation covariance

公式(7)：rotation reweighting
$$\lambda_R = \frac{\alpha}{\|\Delta \xi_t\| + \varepsilon}$$

- $\alpha$：scaling constant
- $\varepsilon$：avoid division by zero
- Intuition：translation 小的时候（near-pure rotation），$\lambda_R$ 被放大，rotation term weight 更高。因为 vision 在 pure rotation 下 rotation 估计特别差，这时让 proprioception 的 rotation 信息 dominate

公式(8)：distance-dependent overall trust
$$\lambda = \lambda_0 \cdot \exp(\beta D)$$

- $\lambda_0$：base weight
- $\beta$：controls how fast trust shifts with distance
- $D$：average distance to observed points
- Intuition：场景远的时候 vision depth noise 大，exponential decay 让 proprioception weight 更高

公式(9)：final pose
$$T^* = T_{\text{icp}} \cdot \exp\left(\lambda \begin{bmatrix} \Sigma_t^{-1} \Delta \xi_t \\ \lambda_R \Sigma_R^{-1} \Delta \xi_R \end{bmatrix}\right)$$

**Read in plain English**：从 G-ICP 的结果出发，沿着 proprioception 指示的方向（Mahalanobis normalized）做一步 Lie algebra update。Step size 是 $\lambda$，rotation 方向额外乘 $\lambda_R$。

这是一个 hand-tuned adaptive sensor fusion，类似 visual-inertial navigation 的思路，但 applied to GS-ICP 的 registration 阶段。

---

## Collision Warning 怎么做到 real-time

### 问题

要在 20Hz control loop 里检查 robot ghost 和整个 GS map（200 万 Gaussian）的 collision。如果每个 Gaussian 都做精确 collision test，根本不可能实时。

### Solution：Conservative Ellipsoid + SAT

每个 robot link 和每个 Gaussian splat 都用一个 ellipsoid 包起来。Ellipsoid $i$ 的参数 $\{p_i, R_i, U_i\}$：
- $p_i \in \mathbb{R}^3$：center position
- $R_i \in SO(3)$：orientation
- $U_i \in \mathbb{R}^3$：axis scale

基于 Separating Axis Theorem，两个 ellipsoid 不 overlap 当且仅当存在某个 axis $d$ 使它们的 support function 之差大于 center distance。

公式(10)定义的 test function：
$$G(d) = c \cdot d - \|S^\top d\|$$

- $d \in \partial B(0,1) \subset \mathbb{R}^3$：unit ball 表面上的点（候选 separating axis）
- $c = U_2^{-1} R_2^{-1}(p_1 - p_2)$：在 ellipsoid 2 local frame 下的 center offset vector
- $S = U_2^{-1} R_2^{-1} R_1 U_1$：composite transform matrix

如果 $\max_{d \in \partial B(0,1)} G(d) > 1$，则两个 ellipsoid 不 overlap（safe）；否则 overlap（collision risk）。

**为什么 ellipsoid 而不是更 tight 的 bound**：保守一点的 bound 会产生 false positive（实际不碰但 warning 触发），但 teleoperation 里 false positive 只是 operator 看到红色 ghost 调整一下，成本很低。False negative 才是真灾难——operator 看着绿色 ghost 提交命令，实际碰撞。

GPU parallel 后单次 collision check 在 sub-millisecond 完成，百万级 ellipsoid pair test 完全不影响 20Hz control loop。

---

## 系统拓扑的 engineering 细节

两台 PC 分工：

| Component | Hardware | Role |
|---|---|---|
| GS-PC | i9-13900K + RTX 4090 + 32GB | 跑 GS-SLAM pipeline |
| Interface-PC | Ryzen 9900x + RTX 4090 + 32GB | 处理手势 + 渲染 VR + overlays |
| Meta Quest 3 | – | 仅作 display，wired tether |

三路 MQTT 通信：

| Stream | Rate | Bandwidth | Content |
|---|---|---|---|
| Robot → GS-PC | 2 Hz | 110 Mbps | 1280×720 RGB-D raw |
| GS-PC → Interface-PC | 2 Hz | 7 Mbps | 10k Gaussian splat 增量更新 |
| Interface-PC ↔ Robot | 20 Hz | low | Control command + pose feedback |

**关键 insight**：GS map 传的是 incremental update 而不是整个 map。这是 deploy 的关键——110 Mbps raw RGB-D 在跨洋场景可能不可得，但 7 Mbps 的 incremental Gaussian stream 是 feasible 的。这也是 3DGS 相比 NeRF 的 innate advantage：explicit representation 可以 delta update，implicit NeRF 要么传 weight 要么重 encode。

---

## 实验数据的故事

### Experiment 1：HIL-GS vs. Splat-Nav（不公平的对比）

| Metric | Splat-Nav | HIL-GS |
|---|---|---|
| Total time | 14m 35s | 10m 48s (±2m 32s) |
| Movement time | 14m 35s | 7m 19s (±2m 10s) |
| PSNR | 16.38 | 17.70 |
| SSIM | 0.567 | 0.602 |
| LPIPS | 0.429 | 0.420 |

**这个 comparison 的关键 caveat**：Splat-Nav 拿到的是 ideal condition——pre-built complete map + manually-added intermediate waypoints（因为 planner 在 clutter 环境频繁 failure）。HIL-GS operator 拿到的是 zero prior——no map, no target location。

在这么 unfair 的 disadvantage 下，HIL-GS 仍然全面 outperform。这强烈说明 human operator 的 task-aware view selection 比 autonomous planner 的 information-gain-driven view selection 更 efficient。

5 个 non-expert subject 标准差只有 2m 32s（相对 10m 48s 总时长），说明 framework 对 non-expert 也 highly usable，learning curve 平缓。

### Experiment 2：Sensor fusion 的价值

| Scene | GS-SLAM [6] | GS-ICP SLAM [5] | HIL-GS |
|---|---|---|---|
| Pumpjack PSNR | 12.41 | 13.17 | **24.74** |
| Oil Tank PSNR | 12.79 | 14.81 | **27.18** |

PSNR 从 13 跳到 25 是 **10dB 以上提升**——相当于 pixel error RMS 降低 3 倍以上。这不是 incremental improvement，是 categorical difference。

Vision-only pipeline 在 aggressive motion + distant background 下直接 tracking failure，重建出扭曲的 geometry。Proprioceptive prior 让 system 在这些 condition 下 maintain stable tracking。

### Real-world Experiment

- Franka Panda + RealSense D530i + joint encoder：PSNR 24.15
- LASDRA [7] 5 米 modular aerial manipulator + motion capture：PSNR 23.27

LASDRA 是 aerial platform 有 rapid jitter，但 PSNR 仍 23+，证明 motion-aware fusion 在真实 aggressive motion 下也 work。

---

## 这篇 paper 的真正 contribution

技术创新性其实不算高——adaptive sensor fusion 是 standard 思路，ellipsoid collision 是 conservative engineering choice，VR interaction 也是 existing idea 的组合。真正价值在 **system integration 让 3DGS 在 real robotics teleoperation 里跑起来**。

具体来说：

1. **第一次把 3DGS 从 offline benchmark dataset 拉到 online robot-embodied reconstruction**
2. **第一次把 human-in-the-loop 的 next-best-view decision 用 VR finger interaction 实现**
3. **第一次证明 incremental GS streaming（7Mbps）足以支撑远距离 teleoperation**

这是 **robotics first, vision second** 的设计哲学。Vision 社区追求 PSNR/SSIM 上的 SOTA，robotics 场景追求 robustness 和 real-time deploy。HIL-GS 选择了后者，engineering 上做了一系列 trade-off：conservative ellipsoid vs. tight bound、hand-tuned adaptive weight vs. learned fusion、incremental streaming vs. batch optimize。

---

## 我的几个 takeaways

### 1. GS 从 rendering 技术变成 robotics sensing modality

3DGS 已经不只是 graphics 工具，而是 robot state estimation 的 intermediate representation——比 point cloud dense expressive，比 NeRF fast incremental。这意味着未来 robotics + radiance field 的 cross-area 会越来越多。

### 2. Proprioception 是 vision SLAM 的 anchor

在 aggressive motion 下，vision-only SLAM 的 failure mode 是 lost track。Proprioception 提供 strong motion prior 让 correspondence search 始终在 right basin。这跟 visual-inertial odometry [8] 的 motivation 一致，但 applied to GS-ICP 更 specific。

### 3. HIL 的本质是 distribution mismatch

Autonomous planner 的 objective 是 geometry information gain。Operator 的真实 objective 是 task success。这两者的 distribution mismatch 在 cluttered/occluded industrial scene 尤其大。HIL 的 "loop" 就是把 task prior inject 回 view selection。

### 4. Incremental streaming 是 deploy 关键

7 Mbps 的 incremental Gaussian stream vs. 110 Mbps raw RGB-D。这是 3DGS 相比 NeRF 的 innate advantage——explicit representation 可以 delta update，implicit NeRF 要么传 weight 要么重 encode。Teleoperation 在低带宽环境很常见，这个 advantage 是 practical value 的核心。

### 5. Human operator 仍然不可替代

即使在 Splat-Nav 拿到 complete prior map 的情况下，HIL-GS 的 human operator 仍然更 efficient。这暗示在复杂、未知、task-critical 场景，human 的 holistic scene understanding 仍是 autonomous planner 难以 replicate 的。

---

## 可能的延伸方向

### Dynamic scene extension

Paper 明确说 limitation 是 static scene assumption。如果有 moving target（如旋转的 valve），GS 会 fuse 不同时刻的 geometry 产生 artifact。可以借鉴 Dynamic 3D Gaussians [9] 或 4DGS 的 deformation field，给每个 Gaussian 加 temporal deformation。

### Cognitive load 优化

Multi-stage preview-evaluate-execute 在长任务下可能有 cognitive load。Semi-autonomous 模式：operator 给 high-level goal（"reconstruct that valve"），robot 做 fine-grained motion planning，collision warning 作为 hard constraint。这是 paper 提到的 future direction。

### Edge-compressed streaming

110 Mbps 在跨洋 disaster response 不可得。考虑 semantic-aware streaming：只传 task-critical region 的高质量 depth，background 用低质量 proxy。或者 event-based camera 作为补充 modality。

### Learning-based fusion

公式(7)(8)的 adaptive weight 是 hand-tuned。可以 learn 一个 fusion policy，input 是 vision/proprioception confidence estimate，output 是 fusion weight。甚至可以 end-to-end learn 整个 sensor fusion module，loss 用 reconstruction fidelity + tracking robustness。

---

## References

- [1] 3D Gaussian Splatting (TOG 2023): https://repo.videocutting.net/source/3dgs.pdf
- [2] Splat-Nav (T-RO 2025): https://arxiv.org/abs/2403.02751
- [3] ActiveSplat: https://arxiv.org/abs/2410.21955
- [4] MoveIt Task Constructor: https://moveit.github.io/moveit_tutorials/doc/task_constructor/task_constructor.html
- [5] RGBD GS-ICP SLAM (ECCV 2024): https://link.springer.com/chapter/10.1007/978-3-031-73423-8_11
- [6] Gaussian Splatting SLAM (CVPR 2024): https://arxiv.org/abs/2312.06741
- [7] LASDRA (ICRA 2018): https://arxiv.org/abs/1806.06750
- [8] VINS-Mono (T-RO 2018): https://arxiv.org/abs/1708.03852
- [9] Dynamic 3D Gaussians: https://dynamic3dgaussians.github.io/
- [SplaTAM (CVPR 2024)](https://arxiv.org/abs/2403.02751)
- [VR-Splat (ACM CGIT 2025)](https://dl.acm.org/doi/10.1145/3720424)
- [VR-GS (SIGGRAPH 2024)](https://zju3dv.github.io/vr-gs/)
- [Luma AI Unreal Engine Plugin](https://lumaai.notion.site/Luma-Unreal-Engine-Plugin-0-41-8005919d93444c008982346185e933a1)
- [NeRF (ECCV 2020)](https://arxiv.org/abs/2003.08934)
- [Generative model-based predictive display (ICRA 2021)](https://arxiv.org/abs/2011.06264)

---

# HIL-GS: Human-in-the-Loop Gaussian Splatting for Robotic Teleoperation

## 一、核心问题与动机

这篇paper的出发点可以提炼为一个intuition：teleoperation的核心瓶颈是**operator的situational awareness不足**。传统live camera stream有三个本质缺陷：

- **Tunnel vision**：FOV受限，operator看不到camera视野边缘的obstacle
- **Weak depth cues**：单目video流缺乏parallax和accommodation cue，operator无法判断collision clearance
- **Fixed viewpoint**：camera rigid-mounted在robot body上，operator无法像NeRF/GS那样free-viewpoint去"绕到背后看"

GS-ICP SLAM和3DGS已经解决了**rendering quality**问题，但是把它们真正deploy到teleoperation中有两个关键gap：

1. **View acquisition bottleneck**：dense GS需要多视角，但teleoperation中robot trajectory是为manipulation/inspection服务而非为scanning优化，trajectory是aggressive、unstructured、rotation-only的——这会让vision-only SLAM直接drift甚至fail
2. **Autonomy vs. expertise gap**：autonomous active-sensing planner（如ActiveSplat [1]）在cluttered non-convex环境中planner failure频繁，且缺乏task-critical semantic prior

HIL-GS的核心idea就是用**human expert的scene understanding**来close this loop——operator通过VR观察incremental GS map，用手指交互选择next-best-viewpoint，同时collision warning module保证safety。这本质上是把**active SLAM的next-best-view规划问题转化为人机协同的decision problem**。

---

## 二、系统架构解析

Paper的Fig. 2展示了tight coupling的三个module：

### Module 1: Motion-Aware GS Reconstruction

这是技术含量最高的部分。它基于GS-ICP SLAM [2]做了两项extension：

#### (a) 数学推导

记pose $T \in SE(3)$，其Lie algebra表示 $\xi \in \mathbb{R}^6$（前3维translation $\xi_t$，后3维rotation $\xi_R$，对应 $\mathfrak{se}(3)$的twist coordinate）。

**Step 1: Proprioceptive prior generation**

公式(1)：
$$\Delta \xi_{\text{prop}} = \log(T_{\text{prev}}^{-1} T_{\text{prop}})$$

- $T_{\text{prev}} \in SE(3)$：上一frame的robot pose（已经refine过的final estimate）
- $T_{\text{prop}} \in SE(3)$：当前frame通过proprioception（IMU + joint encoder + forward kinematics）推算的pose
- $\log(\cdot): SE(3) \to \mathbb{R}^6$：matrix logarithm把relative pose mapping到Lie algebra

这一步的intuition是：用proprioception给出一个motion prior $\Delta \xi_{\text{prop}}$，作为G-ICP的initial guess。

公式(2)：
$$X_{t^*} = \exp(\Delta \xi_{\text{prop}}) \cdot X_t$$

- $X_t$：当前frame的target point cloud（depth camera读入的local frame下）
- $\exp(\cdot): \mathbb{R}^6 \to SE(3)$：matrix exponential
- $X_{t^*}$：predicted target point cloud，已经pre-aligned到上一frame的frame下

这样G-ICP就不用从identity开始搜索correspondence，搜索convergence basin大大扩大。在aggressive motion下，vision-only的G-ICP会因initial misalignment过大而fail to converge。

**Step 2: G-ICP registration**

公式(3)(4)就是standard Generalized-ICP [3]的cost：

$$J_{\text{icp}}(T) = \sum_{i=1}^{N} (x_i^t - T x_i^s)^\top (C_i^t + T C_i^s T^\top)^{-1} (x_i^t - T x_i^s)$$

- $x_i^s \in X_s$：source point cloud的第$i$个点
- $x_i^t \in X_{t_{\text{cur}}}$：target point cloud对应点
- $C_i^s, C_i^t \in \mathbb{R}^{3\times 3}$：每个点的local covariance（planarity-based）
- $T \in SE(3)$：待优化的relative transform
- $N$：correspondence对数

G-ICP vs. point-to-point ICP vs. point-to-plane ICP的区别在于cost function中的Mahalanobis distance用full anisotropic covariance，这一般更robust。

**Step 3: Adaptive fusion**

这是paper最关键的创新。先算discrepancy：

公式(5)：
$$\Delta \xi = \log(T_{\text{icp}}^{-1} T_{\text{prop}}) \in \mathbb{R}^6$$

分解为 $\Delta \xi_t \in \mathbb{R}^3$ (translation error) 和 $\Delta \xi_R \in \mathbb{R}^3$ (rotation error)。

公式(6)定义proprioception的uncertainty block-diagonal covariance：
$$\Sigma_{\text{prop}} = \text{diag}(\Sigma_t, \Sigma_R)$$

公式(7)：rotation reweighting factor
$$\lambda_R = \frac{\alpha}{\|\Delta \xi_t\| + \varepsilon}$$

- $\alpha > 0$：scaling constant
- $\varepsilon$：避免除零的small constant
- Intuition：当translation error小（near-pure rotation场景）时，$\lambda_R$变大，rotation term被trust更多——因为vision在rotation-only motion下特别脆弱（视差变化小，photometric loss太平坦），所以这时proprioception的rotation信息更有价值

公式(8)：distance-dependent weight
$$\lambda = \lambda_0 \times \exp(\beta D)$$

- $\lambda_0$：base weight
- $\beta$：controls distance dependence
- $D$：average distance to observed points
- Intuition：场景远的Gaussian对应low photometric signal和high depth noise，这时vision可信度下降，需要trust proprioception更多。$\exp(\beta D)$是exponential的trust shift

公式(9)：final pose refinement
$$T^* = T_{\text{icp}} \cdot \exp\left(\lambda \begin{bmatrix} \Sigma_t^{-1} \Delta \xi_t \\ \lambda_R \Sigma_R^{-1} \Delta \xi_R \end{bmatrix}\right)$$

这是一个iterative update——从 $T_{\text{icp}}$ 出发，沿着proprioception prior指示的direction（Mahalanobis-normalized）做一步Lie algebra update。$\lambda$作为step size和trust weight的combined factor。

#### (b) Keyframe selection的modification

原GS-ICP SLAM用overlap ratio或fixed interval做keyframe决策。HIL-GS改用**proprioceptive tracking**：当累计的 $T_{\text{prop}}$ motion超过threshold就插keyframe。Intuition是：在fast motion下frame-to-frame overlap可能因为motion blur或depth missing值而misjudge，但proprioception是稳定信号。

#### (c) Depth masking

对过近/过远的depth value直接mask掉不生成Gaussian。这个细节在outdoor场景尤其重要——RealSense这类camera对天空、远景返回的depth noise极大，不mask会让GS-ICP在那些区域fit出spurious Gaussians。

---

### Module 2: VR-based Informative Display

#### (a) 系统拓扑

- **GS-PC**（Intel i9-13900K, RTX 4090, 32GB RAM）：跑GS-SLAM pipeline
- **Interface-PC**（Ryzen 9900x, RTX 4090, 32GB RAM）：处理手势、渲染VR + overlays
- **Meta Quest 3**：仅作display，wired tether到Interface-PC
- **MQTT**协议三路通信：
  1. Robot → GS-PC：1280×720 RGB-D @ 2Hz, 110 Mbps
  2. GS-PC → Interface-PC：**incremental** GS update 10k splats @ 2Hz, 7 Mbps——这里很关键，不传整个map只传delta
  3. Interface-PC ↔ Robot：control command @ 20Hz, low bandwidth

为什么RGB-D只2Hz？因为teleoperation要bandwidth-conservative，远距离网络条件下高帧率难sustain。GS map的2Hz update对应一次新keyframe就refine一次。

#### (b) Voxel observation map

workspace 10m radius, 0.5m resolution voxel grid。每个voxel初始为unobserved（灰色）。当从某个viewpoint看过去，depth值落在voxel内且不被已有Gaussian occlude，voxel flip为observed（transparent）。

这里的subtlety是**用现有GS作为occlusion model**——即voxel观测性是相对于当前重建状态的，而不是绝对的。这避免了在occluded区域虚假标记observed。

#### (c) Predictive robot display

operator右手pinch触发end-effector constraint到右手pose，通过MoveIt [4]逆运动学解出整robot joint configuration，作为半透明ghost显示。同时显示camera FOV frustum。

#### (d) Collision warning——数学细节

这是paper中比较有技术含量的几何部分。每个robot link和每个Gaussian splat都用**conservative ellipsoid**包围。

Ellipsoid $i$ 的参数：$\{p_i, R_i, U_i\}$
- $p_i \in \mathbb{R}^3$：center
- $R_i \in SO(3)$：orientation
- $U_i \in \mathbb{R}^3$：scale（轴对角线）

基于**Separating Axis Theorem (SAT)**，两个ellipsoid的overlap test归结为判断函数 $G: \partial B(0,1) \to \mathbb{R}$ 的max是否>1：

公式(10)：
$$G(d) = c \cdot d - \|S^\top d\|$$

其中：
- $d \in \partial B(0,1) \subset \mathbb{R}^3$：unit ball boundary上一点（separating axis candidate）
- $c = U_2^{-1} R_2^{-1}(p_1 - p_2)$：在ellipsoid 2 local frame下的center offset
- $S = U_2^{-1} R_2^{-1} R_1 U_1$：composite transform

这是一个transformed support function problem。$G(d) > 1$意味着存在separating axis $d$使得两个ellipsoid分离。

**关键计算bottleneck**：单次collision check需要做几百万次ellipsoid pair test（每个robot link × 每个map Gaussian）。作者用GPU parallel化到sub-millisecond级别，这对20Hz control loop的real-time需求至关重要。

---

### Module 3: Finger-Based Control Interface

#### (a) Hand tracking

paper测了两套：
- **VIST** [5]：visual-inertial skeletal tracking，sub-mm accuracy，可扩展haptic
- **Meta Quest 3 built-in hand tracking**：自然无wearable

#### (b) VR viewpoint maneuvering

- 右手pinch-drag：translate view
- 前后motion：zoom（改变camera距离）
- 双手pinch-twist：rotate view
- 整体6-DoF reorientation

#### (c) Multi-stage robot motion selection

这是HIL的关键interaction design：
1. 右手pinch → ghost实时跟随fingertip（通过MoveIt IK）
2. 左手pinch → fix ghost（freeze candidate pose）
3. 评估：surrounding layout、joint limits、singularities、self/scene occlusion
4. 右手pinch → execute；左手pinch → cancel

这个preview-evaluate-execute的多阶段设计是paper的UX贡献。它让operator可以explore alternative poses而无需commit，类似"action shadowing"的概念。

---

## 三、实验数据深度解读

### Experiment 1: HIL-GS vs. Splat-Nav [6]

| Metric | Splat-Nav | HIL-GS |
|---|---|---|
| Total time | 14m 35s | 10m 48s (2m 32s std) |
| Movement time | 14m 35s | 7m 19s (2m 10s std) |
| Interaction time | – | 3m 29s (30s std) |
| PSNR | 16.38 | 17.70 |
| SSIM | 0.567 | 0.602 |
| LPIPS | 0.429 | 0.420 |

**Critical caveat**：Splat-Nav是在**ideal condition**下跑的——pre-built map + manually-added intermediate waypoints（因为planner在clutter环境频繁failure）。HIL-GS operator**没有pre-built map，没有target location信息**，完全online。在这种unfair disadvantage下HIL-GS仍outperform，这是paper的核心claim。

PSNR +1.32在GS reconstruction里已经是significant gap，特别是考虑到baseline有完全地图advantage。

为什么HIL-GS的movement time少7分钟？因为operator看到live GS map能立即判断哪些区域还poorly reconstructed，directly navigate过去，而Splat-Nav的discrete waypoint path是segmented的，path非convex导致detour。

注意5个subject的标准差很小（2m 32s on 10m 48s），说明framework对non-expert也highly usable。

### Experiment 2: Sensor fusion的重要性

| Method | Pumpjack PSNR | Oil Tank PSNR |
|---|---|---|
| Gaussian Splatting SLAM [7] | 12.41 | 12.79 |
| GS-ICP SLAM [2] | 13.17 | 14.81 |
| **HIL-GS (Ours)** | **24.74** | **27.18** |

PSNR从13跳到25是**10dB以上**的提升——这相当于pixel error RMS降低约3倍。这种量级的提升在SLAM领域不是incremental improvement而是categorical difference。

Vision-only pipeline在以下场景fail：
- Pure rotation：photometric loss在rotation下ambiguity高
- Fast linear motion：motion blur + frame-to-frame overlap低
- Distant sparse background：parallax小，depth triangulation噪声大

Proprioceptive prior在所有这些场景都给出stable initial guess，让G-ICP不会lost track。Adaptive fusion（公式7-9）则保证在vision-reliable时让vision dominate，在vision-degraded时让proprioception补位。

### Real-world Experiment

- Franka Panda + RealSense D530i + joint encoder
- LASDRA [8] 5-meter modular aerial manipulator + motion capture system

Real-world数据：Franka PSNR 24.15，LASDRA PSNR 23.27。LASDRA是aerial platform有rapid jitter，但PSNR仍23+，证明motion-aware fusion在真实aggressive motion下也work。

---

## 四、与related work的关系网络

### 4.1 Reconstruction-based teleoperation的演进

```
SLAM-based (sparse feature/volumetric)
    ↓ 缺photorealism
NeRF-based [9, 10, 11]
    ↓ 离线优化 + 秒级rendering
3DGS-based [3, 4]
    ↓ 但需要dense多视角，autonomous planner在cluttered环境fail
HIL-GS (本文)
```

### 4.2 Human-in-the-Loop SLAM的演进

```
A-SLAM [14]: AR headset手势refine 2D occupancy grid
    ↓ 
Human-in-the-Loop SLAM [15]: GUI调object placement
    ↓
Interactive 3D Graph SLAM [16]: RViz mouse实时align point cloud
    ↓
Semantic HIL-SLAM [17, 18]: operator标记ROI/hazard
    ↓
HSS-SLAM [19]: operator online refine superquadric object model
    ↓ (限制: 都是point cloud/voxel, coarse input)
HIL-GS: 第一个用live 3DGS + finger-based VR interaction
```

### 4.3 GS + VR工作的定位

- VR-Splat [12]：优化VR rendering pipeline
- VR-GS [13]：Gaussian转mesh做physical simulation
- 这些都是static/pre-captured scene的interaction
- HIL-GS是第一个**online reconstruction within teleoperation loop**

---

## 五、Intuition层面的几个关键点

### 5.1 为什么HIL比autonomous active-sensing好？

Active SLAM的next-best-view通常用information gain（如Shannon entropy reduction over voxel grid）+ path cost的trade-off。但这有几个fundamental limitation：

1. **Semantic blindness**：autonomous planner不知道哪个valve是task-critical，可能waste view在irrelevant region
2. **Local minima**：greedy information gain容易陷入local最优，特别是在non-convex空间
3. **Safety unconstraint**：information-optimal viewpoint可能在物理上unreachable或collision-prone

Human operator通过VR实时观察可以holistically考虑semantic priority、geometric informativeness、path safety。Paper的Experiment 1在Splat-Nav有manual intermediate waypoint assistance下仍outperform，正是这个human prior的价值体现。

### 5.2 为什么sensor fusion要做adaptive而不是fixed weight？

Fixed weight在某个regime下能work但会fail在其他regime：
- 纯rotation：vision unreliable，要trust proprioception的rotation
- Large translation + close-range：vision reliable，要trust vision
- Distant background：vision depth noise大，要trust proprioception

公式(7)的 $\lambda_R = \frac{\alpha}{\|\Delta \xi_t\| + \varepsilon}$实际上是一个**adaptive gating**——translation小的时候rotation weight被放大。公式(8)的 $\lambda = \lambda_0 \exp(\beta D)$是distance-dependent的整体trust shift。

### 5.3 Conservative ellipsoid的trade-off

用ellipsoid包robot link和Gaussian splat是conservative的——会有false positive（实际不collision但ellipsoid overlap）。但in teleoperation这个trade-off合理，因为：
- False positive只是触发warning，operator可以override
- False negative会导致实际碰撞
- GPU parallel化让百万级测试sub-ms，对交互率没bottleneck

---

## 六、可能的延伸与limitation

### 6.1 Dynamic scene

Paper明确说limitation是static scene assumption。Teleoperation中如果有moving target（如旋转的valve在动），GS会fuse不同时刻的geometry产生artifact。Future work提到的"dynamic target object segmentation and tracking"是关键方向，可以借鉴Dynamic 3DGS [11]或4DGS的deformation field。

### 6.2 Cognitive load

虽然paper展示non-expert用户也能consistent操作，但multi-stage preview-evaluate-execute可能在长时间任务下有cognitive load。Semi-autonomous的hybrid模式（operator给high-level goal，robot做fine-grained motion planning with collision warning as hard constraint）是paper提到的future direction。

### 6.3 Bandwidth requirement

110 Mbps的RGB-D stream在真实remote operation（如跨洋disaster response）可能不可得。可以考虑edge-compressed传输或semantic-aware streaming——只传task-critical region的高质量depth。

### 6.4 与NeRF-based predictive display [10]的关系

Xie et al. 2021用NeRF做predictive display，但offline optimization要几分钟，rendering每view要几秒。HIL-GS用3DGS将这个latency降到real-time。但NeRF在view extrapolation上可能更robust，GS在unobserved region会fail to render。HIL的voxel observation map部分mitigate这个——operator能看到哪里unobserved。

### 6.5 与SplaTAM [12]、MonoGS [13]等RGB-D GS-SLAM的对比

HIL-GS的base是GS-ICP SLAM [2]，但更近期有SplaTAM（CVPR 2024）和Gaussian Splatting SLAM [7]。这些method在static benchmark dataset上可能更高，但paper没比较。可能的reason是这些method在aggressive teleoperation motion下也会fail，且在real-time incremental + streaming场景下不稳定。HIL-GS的contribution主要在motion-aware fusion和HIL loop，base SLAM可替换。

---

## 七、对robotics + radiance field的更广context

这篇paper实际上指向一个更大的trend：**radiance field从offline capture转向online robot-embodied reconstruction**。这带来几个独特挑战：

1. **Trajectory non-optimality**：robot motion是为task服务不是为scan服务
2. **Sensor constraint**：单一forward camera，不能像multi-camera rig那样dense capture
3. **Real-time + incremental**：不能等所有data收集完再optimize
4. **Robustness over accuracy**：teleoperation中tracking failure的代价远大于sub-optimal reconstruction

HIL-GS通过两件事回答这些：(1) sensor fusion补vision的robustness，(2) human-in-loop补view selection的semantic understanding。这是**robotics first, vision second**的设计哲学。

---

## References (web links)

- [1] ActiveSplat: https://arxiv.org/abs/2410.21955
- [2] RGBD GS-ICP SLAM (ECCV 2024): https://link.springer.com/chapter/10.1007/978-3-031-73423-8_11
- [3] Generalized-ICP (Segal et al. RSS 2009): https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf
- [4] MoveIt Task Constructor: https://moveit.github.io/moveit_tutorials/doc/task_constructor/task_constructor.html
- [5] VIST hand tracking (Science Robotics 2021): https://www.science.org/doi/10.1126/scirobotics.abe1315
- [6] Splat-Nav (IEEE T-RO 2025): https://arxiv.org/abs/2403.02751
- [7] Gaussian Splatting SLAM (CVPR 2024): https://arxiv.org/abs/2312.06741
- [8] LASDRA (ICRA 2018): https://arxiv.org/abs/1806.06750
- [9] NeRF (ECCV 2020): https://arxiv.org/abs/2003.08934
- [10] Generative model-based predictive display (ICRA 2021): https://arxiv.org/abs/2011.06264
- [11] Dynamic 3D Gaussians: https://dynamic3dgaussians.github.io/
- [12] SplaTAM (CVPR 2024): https://arxiv.org/abs/2403.02751
- [13] MonoGS / Gaussian Splatting SLAM: https://arxiv.org/abs/2312.06741
- [14] A-SLAM (ICRA 2019): https://ieeexplore.ieee.org/document/8793917
- [15] Human-in-the-Loop SLAM (AAAI 2018): https://ojs.aaai.org/index.php/AAAI/article/view/12023
- [16] Interactive 3D Graph SLAM (RAL 2021): https://arxiv.org/abs/2011.00954
- [17] Semantic SLAM with HIL: https://doi.org/10.1007/978-3-030-96007-3_21
- [18] Immersive interface for teleoperation: https://doi.org/10.1007/978-3-319-64107-2_3
- [19] HSS-SLAM (IROS 2024): https://arxiv.org/abs/2409.07083
- [3DGS original paper (TOG 2023)](https://repo.videocutting.net/source/3dgs.pdf)
- [VR-Splat (ACM CGIT 2025)](https://dl.acm.org/doi/10.1145/3720424)
- [VR-GS (SIGGRAPH 2024)](https://zju3dv.github.io/vr-gs/)
- [Luma AI Unreal Engine Plugin](https://lumaai.notion.site/Luma-Unreal-Engine-Plugin-0-41-8005919d93444c008982346185e933a1)

---

## 八、给你的intuition takeaways

1. **GS从rendering技术变成robotics sensing modality**：3DGS已经不仅是graphics工具，而是robot state estimation的intermediate representation——既比point cloud dense expressive，又比NeRF fast incremental。

2. **Proprioception是vision SLAM的"anchor"**：在aggressive motion下，vision-only SLAM的failure mode是lost track，proprioception提供了strong motion prior让correspondence search始终在right basin。这跟visual-inertial odometry的motivation一致，但用在GS-ICP上更specific。

3. **HIL的本质是distribution mismatch**：autonomous planner的objective是geometry information gain，但operator的真实objective是task success。这两者的distribution mismatch在cluttered/occluded industrial scene尤其大。HIL的"loop"是把task prior inject回view selection。

4. **Collision warning的SAT-based ellipsoid test是practical engineering的胜利**：理论上tighter bounding volume（如convex hull）更准确，但ellipsoid的overlap test可以closed-form + GPU parallel，对20Hz control loop的关键。Engineering-wise，conservative bound + operator override是合理设计。

5. **Incremental GS streaming（7Mbps vs 110Mbps raw RGB-D）**：这是deploy的关键。GS map的compact representation让远距离teleoperation变可行。这点上GS比NeRF有innate优势——explicit Gaussian可以直接delta update，implicit NeRF要传weight或重新encode。

整体上，这篇paper是**3DGS真正走出lab benchmark走向real robotics teleoperation**的重要一步，其价值主要在system integration和motion-aware sensor fusion的engineering robustness上。理论新意不算高（adaptive weight是standard sensor fusion思路），但把所有piece拼起来让它在real robot platform work是关键contribution。
