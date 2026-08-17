---
source_pdf: VLK Learning Humanoid Loco-Manipulation from.pdf
paper_sha256: 5034ffea4a686e7ec3704fedf99b2f255106891659521e9baf4b32aee5d1aee4
processed_at: '2026-08-13T03:03:46-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLK 用人话讲

Andrej, 我换个更直白的方式给你讲这篇 paper。

## 一句话总结

你想让 humanoid robot 听你的话干活,你得给它看大量的"看到什么画面 + 听到什么指令 + 该怎么做动作"的训练数据。这种数据现成没有,真去采集又贵又慢。VLK 的做法是:**用手机扫一遍房间,在电脑里重建出 3D 场景,然后让程序自动"假装"机器人在里面走了一遍,顺便把第一人称视角的画面渲染出来**。这样就全自动批量造出了训练数据,不用人一点一点操作机器人示范。

---

## 问题出在哪

人形机器人要做的核心事情是:从自己的摄像头看到画面,听懂你说的指令,然后决定全身怎么动。这需要成对的数据——同一个时刻的"画面 + 指令 + 全身运动轨迹"三元组。

以前的数据来源各有各的毛病:

**真机遥操作**(Twist2 [https://arxiv.org/abs/2511.02832]、CLONe [https://arxiv.org/abs/2506.08931]):人穿戴设备远程操控机器人,数据质量高。但每一次 demonstration 都要一个熟练操作员盯着,整套全身数据采集成本极高,场景多样性上不去。

**人体 mocap 数据集**(比如 AMASS):有完整的全身运动,可以 retarget 到 humanoid。但人走在大街上 mocap 时没有"机器人的第一人称视角摄像头画面",你拿不到对应的 egocentric observation。

**Egocentric 视频**(比如 Ego4D):人戴 GoPro 拍的第一人称视频海量。但视频没有 robot-compatible 的 kinematic trajectory,人的运动跟 G1 的 morphology 不一样,关节配置不同,你不能直接拿过来用。

所以没有任何一个现成数据源同时给你这三个东西。这就是 bottleneck。

---

## VLK 的核心思路:解耦

VLK 想了一个办法,把这个难题拆成两半。

**第一半:在重建的 3D 场景里生成机器人运动**

你不用真的让机器人去感知世界去决定怎么走。你手头有场景的完整 3D 信息(物体在哪、地板在哪、哪里能走),这是 privileged information。你直接用这些信息规划出一条合理的 G1 全身运动轨迹。这一步完全不需要解决感知问题,因为你在"上帝视角"下做 motion synthesis。

**第二半:事后渲染第一人称画面**

运动轨迹生成好之后,你把 G1 放在 3D 场景里,沿着轨迹走一遍,用虚拟摄像头(跟真机上的 ZED 2i 标定一致)渲染出每一帧的第一人称画面。这就是 hindsight rendering——你先有运动,再渲染观测。渲染不需要解决控制问题,只是把画面"录"出来。

**结果**:你自动得到了成对的(画面, 指令, 全身轨迹),不用人介入。整套流程跑下来,48000 条轨迹,600 GPU-hours,零人工操作。

---

## 数据是怎么造的,分步看

### Step 1: 扫房间

拿一台 iPhone 14 Pro,装 Polycam app,扫房间。iPhone 的 LiDAR 给你 metric scale,RGB 给你外观。用 3D Gaussian Splatting [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/] 重建出 photorealistic 的 3D 场景。

为什么用 3DGS 不用 NeRF?3DGS 渲染快,可以实时渲染第一人称画面,适合大规模数据生成。NeRF 渲染慢,跑 48000 条 trajectory 的 frames 会慢死。而且 3DGS 的 metric scale 直接给你真实物理距离,机器人的"走到椅子要 3 米"这种约束才有意义。

### Step 2: 人肉标注一下

3DGS 重建出来的是一坨好看的 3D 点,但没有 semantic label——你不知道哪坨是椅子哪坨是桌子。所以作者用 viser 这个 3D visualization library 做了个交互工具,人手动拖几个 3D bounding box,标上"chair""table""box"这种 label,再在地板上点几个点圈出 walkable region。

这一步是 manual 的。但关键在于:**这是一次性投入**。标完一次,后面可以采样出成千上万个 task configuration——从不同起点走向不同物体,在不同位置 pick box 放到不同 surface。边际成本接近零。

### Step 3: 生成机器人运动

这部分用了两个 model。

**Navigation model**:用 BONES-SEED [https://huggingface.co/datasets/bones-studio/seed] 数据集训练,生成 object-directed walking 的 G1 motion。条件是:指令、初始 G1 state、稀疏 waypoint。

**Interaction model**:基于 CHOIS [https://arxiv.org/abs/2312.16205] 的 conditional diffusion 架构,把 SMPL human motion 改成 G1 representation。训练数据是 OMOMO [https://arxiv.org/abs/2312.16205] 的 human-object interaction sequence,用 OmniRetarget [https://arxiv.org/abs/2506.07768] retarget 到 G1 morphology。

G1 的 motion 表示成:
- 全身每个 joint 的 global position 和 6D rotation
- G1 joint angles 的 sin/cos 编码(避免 0 跟 2π 是同一角度但数值差异巨大的问题)
- Object trajectory(交互任务时)

**关键 conditioning**:object geometry 用 Basis Point Set [https://arxiv.org/abs/1906.04390] 编码,wrist 的 desired relative pose(相对于 object 坐标系)显式指定,wrist-object contact label 告诉 generator"左手第几帧开始接触,右手第几帧开始接触"。

这个 contact label 是个贯穿全 paper 的设计 choice。Generator 用它来约束 motion 生成,后面 VLK policy 输出它,再后面 tracker 用它。一个 binary 信号,但解决了"什么时候该 grip"这个核心决策。

### Step 4: 修补生成 motion 的 artifact

Diffusion 生成 motion 时会有常见 artifact:脚在地上滑(foot sliding)、手跟物体接触不真实。

修复方法:
- 用 predicted foot contact label 找到 stance phase,在这期间用 IK 把脚固定在 contact position 不动
- 借 EgoAllo [https://arxiv.org/abs/2501.08535] 的 wrist-pose matching,把手腕往 object 局部坐标系下的目标 pose 拉

这些修复都是 kinematic 局部操作,不改 motion 整体结构。

### Step 5: 渲染第一人称画面

把修好的 G1 motion 放进 Isaac Sim 里的 3DGS 场景,在 G1 头上挂一个虚拟 ZED 2i camera(extrinsics 跟真机标定对齐),沿轨迹渲染每一帧的第一人称 RGB 画面。

为了 sim-to-real 稳一点,渲染时做 domain randomization:
- Camera extrinsic 和 focal length 小幅扰动(模拟标定误差)
- 光照强度和方向随机(真实场景光照变化大)
- Image-space 的 brightness/contrast/saturation/hue 扰动 + Gaussian noise + Gaussian blur

Table 2 显示,**光照 randomization 比 camera randomization 重要得多**(87% vs 48% walking success)。直觉:3DGS 在固定光照下重建,真机部署时光照变化大(阳光、室内灯、阴影),所以 lighting DR 是关键。

---

## VLK Policy:学这个映射

有了数据,训个 policy 吃(画面, 指令, 当前状态),输出未来 30 帧(1 秒 @ 30Hz)的全身运动轨迹。

**从 π_0.5 [https://arxiv.org/abs/2504.16054] 初始化**:Physical Intelligence 的预训练 VLA model,已经在大量 robot data 上训练过。VLK 在这个基础上 fine-tune,把 action space 换成 G1 kinematic representation。

**用 flow matching + x0-prediction**:不是预测 velocity 然后积分,直接预测 clean trajectory。直觉:robotic action space 有物理意义,直接预测目标 trajectory 比预测 velocity 更稳,multi-modal action distribution 下不会 collapse 到 mean。

**Loss**:主 loss 是 trajectory reconstruction 的 L2。加几个 auxiliary loss:
- Foot contact label prediction(focal loss,因为 contact 事件 sparse)
- Accumulated root trajectory loss(每帧位移小误差累积 30 帧可能差 1 米,这个 loss 防止 drift)
- Forward kinematics loss(预测 joint angle 后做 FK 得到 ankle/wrist position,跟 ground truth 比,保证 joint angle 输出对应正确的 end-effector 位置)
- Foot skating regularization(预测 contact phase 内惩罚脚的水平速度,防脚滑)

---

## Tracker:把运动变成关节动作

VLK 输出的是 kinematic trajectory——关节该到什么位置,身体该往哪走。但真机上需要 joint-level PD target,还要处理 dynamics(重力、惯性、接触力)。

这部分交给 SceneBot [https://arxiv.org/abs/2606.27581]——一个在 sim 中用 RL 训练的 whole-body tracking policy。它**完全 blind 到画面和指令**,只看 converted reference target 和当前 robot proprioception。

VLK 输出的 trajectory 转换成 tracker format:lower-body joint targets、upper-body wrist/head target poses、root target pose、binary wrist-object contact labels。Tracker 吃这些 + proprioception,输出所有 actuated joint 的 PD targets。

**Contact-aware 行为**:当 wrist contact label active,tracker 切换到 contact-aware wrist control mode,主动维持 wrist-object 接触。这让 tracker 不用从 perception 推断"现在该 grip 了",直接接受 VLK 的高层指令。

这个分层的关键好处:tracker 在 sim 中可以训练在任意 kinematic reference 上,不受 perception 数据限制。VLK 处理 perception,tracker 处理 dynamics,各自专注。跟 OmniH2O [https://arxiv.org/abs/2406.18260]、GMT [https://arxiv.org/abs/2506.14770]、BeyondMimic [https://arxiv.org/abs/2508.08241] 这些 whole-body tracking 工作哲学一致。

---

## 部署:怎么让真机跑起来

真机部署有几个工程细节。

**三进程架构**(tethered laptop 上):
1. State estimator 估计 robot root pose
2. Whole-body tracker 50Hz 在 RTX 5000 Ada 上跑,每 tick 4.3ms
3. VLK inference client 管理最新 image + state,通过 websocket 把 request 发到 external GPU server

VLK inference 在 RTX 5090 上 31ms,end-to-end replan ~63ms。每个 chunk 覆盖 1 秒运动,replan period ~555ms,有 8.8× headroom,不会 backlog。

**10-frame overlap chunking**:借鉴 Ψ_0 [https://arxiv.org/abs/2601.xxxxx] 的 idea。相邻 chunk 有 10 frame overlap(1/3 秒),blending 平滑过渡,避免 hard switch 的 discontinuity。

**Motion blur 处理**:G1 快速弯腰 pick 物体时,head camera motion blur 严重,degrade VLK prediction。

训练时:对部分 image 加 synthetic motion blur,小范围 σ 保 semantic content。

部署时:维护 0.3 秒 image buffer,用 Laplacian variance 选 sharpest frame:
$$S(I) = \text{Var}(\nabla^2 I)$$
Sharp image 有高频内容 Laplacian variance 大,blur image 高频被抹平 variance 小。选 variance 最大的 frame 喂给 VLK。

这个 trick 简单实用,VLA inference 速率比 camera frame rate 慢,buffer 多个 frame 选最好的,免费拿到 robustness。

---

## 实验讲什么

**Real-world 全系统结果**(Table 1):

| Task | Lab Scene | Apartment Scene |
|---|---|---|
| Walk To | 20/20 | 19/20 |
| Turn Around | 20/20 | 18/20 |
| Pick (Floor) | 16/20 | 18/20 |
| Put (Floor) | 20/20 | 20/20 |
| Pick (Surface) | 11/20 | 13/20 |
| Put (Surface) | 8/20 | 15/20 |

观察:
- **Navigation 基本满分** — synthetic walking data 质量高,sim-to-real gap 小
- **Floor pick/put 强** — OMOMO box lifting 数据覆盖 floor-level interaction
- **Surface pick/put 弱** — OMOMO 对不同 surface height 覆盖不足,grasp 不可靠

**最强 ablation**:Pick without contact label = 0/5 success。没 contact label,tracker 不知道何时 grip,pick 必败。**Contact label 不是 nice-to-have,是必需的**。它把"何时 grip"的高层决策跟"怎么稳定 grip"的 low-level dynamics control 解耦开。

**Data volume ablation**(Figure 4):Pick (Surface) 从 10% data 的 0% 提升到 full data 的 46%。Navigation 在 10% data 就接近饱和。直觉:Navigation 是简单 locomotion,synthetic 数据快速覆盖 motion manifold。Contact-rich manipulation 需要更多 data 覆盖不同 object pose、surface height、approach angle 的组合。未来 humanoid loco-manipulation 的瓶颈在 manipulation data 的 diversity。

**Domain randomization ablation**(Table 2):No randomization 41%, lighting only 87%, camera only 48%, full 90%。Lighting DR 是 sim-to-real 关键。

**Simulation vs Real-World Gap**:Real-world 成功率跟 simulation 大致一致,有时还更高。说明这个 pipeline 的 sim-to-real gap 闭合得很好。可能因为 simulation 用 1000 rollouts 覆盖更多 corner case,real-world trial 是 representative scenario。

---

## 这个 work 的真正贡献

VLK 的贡献不是某个 single technical novelty,是 **system-level integration**:

1. 3DGS 给 photorealistic scene
2. Conditional diffusion 给 kinematic trajectories
3. Hindsight rendering 给 paired egocentric observation
4. Flow matching policy + contact label 给 perception-to-kinematics mapping
5. SceneBot tracker 给 kinematics-to-action execution
6. Real-time chunking + motion blur handling 给 robust deployment

每个单独看都有 prior work。VLK 把它们组装成一个能 sim-to-real 的完整系统,在 physical G1 上 demonstrate 48000 synthetic data 训练的 policy 能做 navigation 和 box transport。这种 system paper 的价值是 **de-risking**:证明这条路可行,后续工作可以专注优化单块。

---

## 我看到的 limitations

Paper 自己列了:
1. OMOMO 限于 large objects,小物体(杯子、工具)不行
2. Tracker 用 wrist-object contact 稳定 large object,不能 precise grasping

我额外看到:

**Scene annotation 仍 manual**。虽然一次性,但 scale 到 hundreds of scenes 时是 bottleneck。未来应该用 SAM-style 自动 semantic segmentation。

**Interaction diversity 受限于 OMOMO**。只能 box-like bimanual transport。要扩展到单手 tool use、articulated object,需要更丰富的 interaction dataset。

**No closed-loop replanning in tracker**。Tracker blind 到 perception,如果 VLK 预测错,tracker 会执行错的 trajectory 1 秒。Real-time chunking 缓解了,但 fundamental issue 还在。

**No language compositionality**。Instruction 是 template-based,不能处理 "walk to the chair then pick up the box and put it on the table" 这种 long-horizon。Paper 在 Figure 1 bottom 展示了 chaining,但似乎是 manual orchestration,不是 policy 自主。

---

## 对 humanoid robot learning 的意义

VLK 展示的是 **data-centric 范式**:不是堆 model size,是堆 diverse synthetic data。这跟 LLM scaling law 哲学一致——数据规模和质量决定 policy 能力。如果未来 humanoid robot 要做 general loco-manipulation,这条路比 teleoperation scale up 更可行经济上。

但几个 open question:
1. **Synthetic data 的 physical fidelity ceiling** — 3DGS 重建的 scene 在 unobserved region 是空的,object geometry 是 static 的
2. **Distribution shift** — synthetic motion generator 的 distribution 跟 real-world task distribution 的 gap 怎么 measure 和 close
3. **Compositionality** — 怎么从 atomic skill chain 到 long-horizon task,需要 hierarchical planning 或 in-context learning

整体看,VLK 是个扎实的 system paper,把 humanoid loco-manipulation 的 synthetic data pipeline 推到了"能在真机 demo"的程度。后续工作可以在这个 pipeline 基础上扩展 interaction type、自动化 annotation、解决 long-horizon。

---

**Relevant Links**:
- Project page: https://vision-language-kinematics.github.io
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- π_0.5: https://arxiv.org/abs/2504.16054
- CHOIS: https://arxiv.org/abs/2312.16205
- OMOMO: https://arxiv.org/abs/2312.16205
- OmniRetarget: https://arxiv.org/abs/2506.07768
- SceneBot: https://arxiv.org/abs/2606.27581
- BONES-SEED: https://huggingface.co/datasets/bones-studio/seed
- OmniH2O: https://arxiv.org/abs/2406.18260
- BeyondMimic: https://arxiv.org/abs/2508.08241
- EgoAllo: https://arxiv.org/abs/2501.08535
- SplatSim: https://arxiv.org/abs/2409.10161
- RoboGSim: https://arxiv.org/abs/2411.11839
- Twist2: https://arxiv.org/abs/2511.02832
- CLONe: https://arxiv.org/abs/2506.08931
- GR00T N1: https://arxiv.org/abs/2503.14734
- LeVERB: https://arxiv.org/abs/2506.13751
- Zhou et al. 6D rotation: https://arxiv.org/abs/1812.07035
- Basis Point Set: https://arxiv.org/abs/1906.04390
- Running VLAs at real-time speed: https://arxiv.org/abs/2510.26742

---

# VLK: 从重建场景中合成交互学习人形机器人 Locomanipulation

Andrej, 这篇 paper 解决的是 humanoid loco-manipulation 的核心瓶颈问题 — **paired data 的缺失**。Humanoid 要做 perception-to-action 的 closed loop,需要三元组 (egocentric image, language instruction, whole-body kinematic trajectory),但是 real-world teleoperation 昂贵且难以 scale,mocap 数据没有 egocentric views,egocentric video 没有 robot-compatible kinematics。VLK 通过 **synthetic data generation in reconstructed 3DGS scenes** 把这个 bottleneck 消掉了。我会按 pipeline 的各个模块逐层展开,把背后的 intuition 讲清楚。

---

## 1. 整体哲学:解耦 Perception 与 Control

VLK 的核心设计选择是把整个 pipeline 切成两层:

- **上层 VLK policy**:吃 egocentric RGB + language + current G1 state,输出 short-horizon(30 frames @ 30Hz = 1 秒)的 whole-body kinematic trajectory + wrist contact labels。这是一个 **kinematic prediction** 任务,不是 low-level torque 控制。
- **下层 Whole-body tracker**(基于 SceneBot):blind,只看 converted reference + proprioception,输出 joint-level PD targets。它在 sim 中用 RL 训练,可以 robust 地 track 各种 reference。

这种解耦带来一个非常关键的数据效率优势:kinematic prediction 可以用 synthetic privileged information 生成 ground truth(知道物体在哪、walkable region 在哪、object geometry 是什么),然后通过 hindsight rendering 把对应的 egocentric observation"事后"渲染出来。**生成轨迹不需要解决感知,渲染观测不需要解决控制**。这两件事都被 decompose 掉了。这一点是 VLK 能 scale 到 48,000 trajectories 的根本原因。

类似 decoupling 的哲学在 BeyondMimic[https://arxiv.org/abs/2508.08241]、ResMimic[https://arxiv.org/abs/2510.05070] 等 whole-body tracking 工作里也能看到,VLK 把它推到了 perception 层。

---

## 2. Real2Sim2Real Pipeline 架构

整个 pipeline 分四块,我可以画成下面的数据流:

```
iPhone 14 Pro (RGB+LiDAR) 
   │
   ▼ Polycam scan
3DGS Reconstruction (metric scale)
   │
   ▼ Annotation (viser tool)
Semantic 3D bounding boxes + Walkable regions
   │
   ▼ Waypoint sampling (task rules)
Sparse waypoints + Initial pose + Object geometry
   │
   ▼ Conditional diffusion synthesis (G1 motion)
X = [p, R, sin(q), cos(q)] (+ X^obj for interaction)
   │
   ▼ Post-processing (IK foot fix + wrist matching)
Filtered, contact-cleaned G1 trajectories
   │
   ▼ Hindsight rendering in Isaac Sim (ZED 2i virtual + DR)
Paired (egocentric RGB, language ℓ, G1 kinematics)
   │
   ▼ VLK policy training (flow matching, π_0.5 init)
π_θ(o_t, ℓ, x_t) → x̂_{t+1:t+H}
   │
   ▼ Deployment on physical G1
Real-time chunking → Tracker (SceneBot) → Joint PD targets
```

这个 Real2Sim2Real 设计类似 SplatSim[https://arxiv.org/abs/2409.10161]、RoboGSim[https://arxiv.org/abs/2411.11839]、RL-GSBridge[https://arxiv.org/abs/2411.11839] 等工作,但 VLK 的独特之处是把 scene reconstruction + humanoid motion synthesis + hindsight rendering **三者串成统一数据源**。

---

## 3. Scene Reconstruction 和 Annotation

### 3.1 3DGS Reconstruction

用 iPhone 14 Pro + Polycam app 扫描,采集 RGB + LiDAR depth,然后优化 3D Gaussian Splatting 表示 [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]。3DGS 相比 NeRF 的优势在于:

- 实时渲染速度,适合大规模 data generation(8.3 小时渲染 1000 个 trajectory 的 egocentric frames)
- Metric scale,直接给出真实物理尺度,这对 humanoid walking 距离、object 接近距离的合理性至关重要
- 显式 point cloud 提取,便于后续 annotation

### 3.2 Annotation

3DGS 本身没有 semantic label 也没有 mesh,所以作者用 viser(一个 3D visualization library)做了交互式 annotation tool:

- **Semantic 3D bounding boxes**:用户在 3D point cloud 中拖拽 box,标 semantic label(chair/table/box 等)
- **Walkable regions**:用户在 floor plane 上点几个点,连成 polygon

这些 annotation 提供了 motion synthesis 的 **privileged information**。这里有一个 intuition:annotation 的成本是 manual 的,但一次性投入。一旦 annotated,就可以采样出成千上万个 task configuration,边际成本极低。这是 VLK 在 600 GPU-hours 内自动生成 48K trajectories 的关键经济模型。

---

## 4. Motion Synthesis:把 CHOIS 改造成 G1 兼容

### 4.1 Motion Representation

这是 paper 里非常技术性的一块。G1 motion sequence of length T 表示为:

$$\mathbf{X} = [\mathbf{p}, \mathbf{R}, \sin(\mathbf{q}), \cos(\mathbf{q})]$$

- $\mathbf{p} \in \mathbb{R}^{T \times J \times 3}$:T 是 sequence length,J 是 joint 数,3 是 (x,y,z) 世界坐标
- $\mathbf{R} \in \mathbb{R}^{T \times J \times 6}$:6D rotation representation,来自 Zhou et al. 2019 [https://arxiv.org/abs/1812.07035],避免 quaternion 的 discontinuity 和欧拉角的 gimbal lock
- $\mathbf{q} \in \mathbb{R}^{T \times D}$:D 是 G1 的 DoF 数,即 joint angles
- $\sin(\mathbf{q}), \cos(\mathbf{q})$:对 joint angle 用 sin/cos 编码,**避免 angle wrap-around discontinuity**。这是 0 和 2π 实际是同一角度但数值差异巨大的问题。

Object trajectory(交互任务):

$$\mathbf{X}^{\text{obj}} = [\mathbf{o}, \mathbf{R}^{\text{obj}}]$$

- $\mathbf{o} \in \mathbb{R}^{T \times 3}$:object 在世界坐标下的位置
- $\mathbf{R}^{\text{obj}} \in \mathbb{R}^{T \times 6}$:object 相对 canonical pose 的 6D rotation

### 4.2 Navigation Model

Navigation model 在 BONES-SEED [https://huggingface.co/datasets/bones-studio/seed] 上的 G1 motion 训练,做 object-directed walking。条件是:task instruction、initial G1 state、sparse scene-grounded waypoints。

### 4.3 Interaction Model

Interaction model 基于 CHOIS [https://arxiv.org/abs/2312.16205] 的 conditional DDPM 架构,但 representation 从 SMPL 改成 G1。关键 adaptation:

1. **OMOMO [https://arxiv.org/abs/2312.16205] 数据 retarget 到 G1**:用 OmniRetarget [https://arxiv.org/abs/2506.07768] 把 SMPL human-object interaction motion 映射到 G1 morphology,保留 interaction 结构
2. **Differentiable G1 forward-kinematics layer**:让 geometric loss 可以直接作用在 global body 和 end-effector positions 上,这对 box interaction 的 wrist placement 精度很关键
3. **Conditioning inputs**:
   - Language instruction(用 CLIP text features 编码)
   - Initial humanoid + object states
   - Sparse waypoints(scene-level guidance)
   - Object geometry(Basis Point Set [https://arxiv.org/abs/1906.04390] + MLP projection)
   - Desired relative wrist poses(在 object coordinate frame 下定义,例如 box-lifting 时两手从 box 对侧 approach,palm 朝下)
   - Wrist-object contact labels(指示每个 wrist 的 expected contact phase)

这里有一个直觉:**contact label 作为 condition 输入给 generator,使得生成的 motion 有明确的接触时序**。这跟下面 VLK 输出 contact label 给 tracker 是同一个设计哲学,贯穿整个 pipeline。

### 4.4 Post-processing

两个修复生成 motion 的常见 artifact:

- **Foot sliding**:用 predicted foot-contact labels 识别 contact phase,在 contact phase 内用 IK 把脚固定在 contact position
- **Hand-object contact**:借鉴 EgoAllo [https://arxiv.org/abs/2501.08535] 的 wrist-pose matching optimization,把 wrist 的 position 和 rotation 拉向 object local frame 下的 input wrist pose

这两个步骤都是 kinematic 修复,不改 motion 的整体结构,只是消除 synthesis model 的局部 artifact。

### 4.5 数据规模经济

Table 1 和 4.2 节给出:
- 每个 environment:4 个 layout × 12 mode(6 evaluated + 6 auxiliary)× 1000 trajectories = 48,000 trajectories
- 每个 mode × layout 在 L40S 上:4 小时 motion synthesis + 8.3 小时 rendering
- **零 human intervention**(除了前期 scene annotation)

对比 real-world teleoperation 系统(如 Twist2 [https://arxiv.org/abs/2511.02832]、CLONe [https://arxiv.org/abs/2506.08931])每个 demonstration 都需要 operator,VLK 的 marginal cost 接近零。

---

## 5. VLK Policy 架构

### 5.1 Per-frame Kinematic State Representation

每一帧 τ 的 G1 kinematic state:

$$\mathbf{x}_\tau = [\Delta x_\tau, \Delta y_\tau, \Delta\psi_\tau, h_\tau, \mathbf{R}_\tau^{\text{root}}, \sin(\mathbf{q}_\tau), \cos(\mathbf{q}_\tau), \mathbf{c}_\tau]$$

- $(\Delta x_\tau, \Delta y_\tau)$:**heading-normalized** root planar displacement。先把世界坐标系旋转到当前 root 的 heading frame,然后取 xy 位移。这避免了绝对坐标系的 ambiguity,也让 policy 学习的是 motion pattern 而非 global position。
- $\Delta\psi_\tau$:yaw 变化量(增量式,不是绝对 yaw)
- $h_\tau$:root height(世界 z 坐标)
- $\mathbf{R}_\tau^{\text{root}}$:heading-normalized 6D root orientation
- $\sin(\mathbf{q}_\tau), \cos(\mathbf{q}_\tau)$:G1 joint angles 的 sin/cos 编码
- $\mathbf{c}_\tau = [c_\tau^L, c_\tau^R]$:left/right wrist-object binary contact labels

**直觉**:这个 representation 设计有几个考量。第一,用 relative displacement 而不是 absolute position,让 policy 学到 motion primitive 而不是"在这个房间走到 (3.2, 1.5)"。第二,contact label 作为 kinematic state 的一部分,让 VLK 不只是预测 motion,还预测什么时候应该 grip。第三,sin/cos 编码让神经网络不用学 angle wrap-around。

### 5.2 Policy Formulation

$$\hat{\mathbf{x}}_{t+1:t+H} = \pi_\theta(o_t, \ell, \mathbf{x}_t)$$

- $o_t$:当前 egocentric RGB(224×224)
- $\ell$:task instruction(固定,整个 episode 不变)
- $\mathbf{x}_t$:current G1 kinematic state
- $H = 30$:30 帧 = 1 秒 @ 30Hz
- $\hat{\mathbf{x}}_{t+1:t+H}$:预测的 future trajectory

### 5.3 Flow Matching with $x_0$-prediction

VLK 从预训练的 $\pi_{0.5}$ [https://arxiv.org/abs/2504.16054] 初始化,用 flow matching 训练,但用 **$x_0$-prediction** 而不是 velocity prediction:

构造 noisy trajectory:

$$\mathbf{x}_{t+1:t+H}^{\alpha} = \alpha\boldsymbol{\epsilon} + (1-\alpha)\mathbf{x}_{t+1:t+H}$$

- $\boldsymbol{\epsilon}$:标准 Gaussian noise
- $\alpha \in [0, 1]$:interpolation coefficient,$\alpha=0$ 时是 clean trajectory,$\alpha=1$ 时是纯 noise
- $\mathbf{x}_{t+1:t+H}$:ground truth future trajectory

Policy 直接预测 clean trajectory:

$$\hat{\mathbf{x}}_{t+1:t+H} = \pi_\theta(\mathbf{x}_{t+1:t+H}^{\alpha}, \alpha, o_t, \ell, \mathbf{x}_t)$$

这里 $\alpha$ 也作为输入,让网络知道当前噪声水平。这跟 standard flow matching 的 velocity prediction 不同 — velocity prediction 学的是 $\mathbf{v} = \boldsymbol{\epsilon} - \mathbf{x}_0$,而 $x_0$-prediction 直接学 clean target。

**直觉**:$x_0$-prediction 在 robotics 任务中通常更稳,因为 action space 有物理意义,直接预测 clean trajectory 让 network 的 output 直接对应可执行 motion,而 velocity 预测需要积分,在 multi-modal action distribution 下可能 collapse 到 mean。$\pi_0$ 系列就是这个 design choice。

### 5.4 Loss Function

主 loss:

$$\mathcal{L}_{\text{traj}} = \|\hat{\mathbf{x}}_{t+1:t+H} - \mathbf{x}_{t+1:t+H}\|_2^2$$

Auxiliary losses(B.1 节详细给出):

**Foot-floor contact loss**(预测 contact label 用于 stance phase 识别):

$$\mathcal{L}_{\text{foot-contact}} = \text{FocalLoss}(\hat{\mathbf{m}}, \mathbf{m})$$

用 focal loss 因为 contact 事件 sparse(大部分时间脚在 swing phase),focal loss 缓解 class imbalance。

**Accumulated root-trajectory loss**(防止 relative displacement 积分 drift):

$$\mathcal{L}_{\text{acc-root}} = \frac{1}{H}\sum_{k=1}^{H}\|\hat{\mathbf{p}}_{t+k}^{\text{root}} - \mathbf{p}_{t+k}^{\text{root}}\|_2^2$$

把预测的 $(\Delta x, \Delta y)$ 累加得到 global root trajectory,跟 ground truth global root trajectory 比较。这个 loss 重要,因为每帧的 relative displacement 误差小,但 30 帧累加可能差 1 米。

**Forward-kinematics loss**(确保 joint angle 输出对应正确的 end-effector 位置):

$$\mathcal{L}_{\text{fk}} = \mathcal{L}_{\text{fk}}^{\text{ankle}} + \mathcal{L}_{\text{fk}}^{\text{wrist}}$$

$$\mathcal{L}_{\text{fk}}^{\text{ankle}} = \frac{1}{H|\mathcal{A}|}\sum_{k=1}^{H}\sum_{j\in\mathcal{A}}\|\hat{\mathbf{p}}_{t+k}^j - \mathbf{p}_{t+k}^j\|_2^2$$

- $\mathcal{A}$:ankle end-effectors 集合
- $\hat{\mathbf{p}}_{t+k}^j$:预测的 joint angle 经过 forward kinematics 得到的 ankle position
- $\mathbf{p}_{t+k}^j$:ground truth ankle position

$\mathcal{L}_{\text{fk}}^{\text{wrist}}$ 同理,$\mathcal{W}$ 是 wrist end-effector 集合。

**Foot-skating regularization**(contact phase 内惩罚水平速度):

$$\mathcal{L}_{\text{foot}} = \frac{1}{H|\mathcal{F}|}\sum_{k=1}^{H}\sum_{j\in\mathcal{F}}\hat{m}_{t+k}^j\|\mathbf{v}_{t+k}^{j,xy}\|_2^2$$

- $\mathcal{F}$:foot end-effectors
- $\hat{m}_{t+k}^j$:predicted contact label(soft,在 0-1 之间)
- $\mathbf{v}_{t+k}^{j,xy}$:foot 在 ground plane 的速度

直觉:当 $\hat{m} \to 1$(contact phase),惩罚项激活,要求 $\mathbf{v}^{xy} \to 0$(脚不动)。这避免生成 motion 时脚在地上滑动的 artifact。

**Total loss**:

$$\mathcal{L}_{\text{total}} = 1.0\mathcal{L}_{\text{traj}} + 0.5\mathcal{L}_{\text{foot-contact}} + 0.2\mathcal{L}_{\text{acc-root}} + 1.0\mathcal{L}_{\text{fk}}^{\text{ankle}} + 1.0\mathcal{L}_{\text{fk}}^{\text{wrist}} + 0.05\mathcal{L}_{\text{foot}}$$

权重比值透露了作者的优先级:trajectory reconstruction 和 FK 是核心,foot skating 只是小 regularizer(0.05)。

---

## 6. Whole-Body Tracker(SceneBot)

Tracker 基于 SceneBot [https://arxiv.org/abs/2606.27581],是一个在 sim 中 RL 训练的 policy,完全 blind 到 egocentric observation 和 language。

**输入转换**:把 VLK 预测的 $\hat{\mathbf{x}}_t$ 转成 tracker format:
- Lower-body joint targets
- Upper-body head/wrist target 6D poses
- Root target pose
- Binary wrist-object contact labels

**Tracker 公式**:

$$\mathbf{u}_t = \pi_{\text{track}}(\bar{\mathbf{x}}_t, \mathbf{s}_t)$$

- $\bar{\mathbf{x}}_t$:转换后的 reference
- $\mathbf{s}_t$:当前 low-level robot state(proprioception:joint position, velocity, IMU 等)
- $\mathbf{u}_t$:所有 actuated joint 的 PD targets

**Contact-aware behavior**:当 wrist contact label active,tracker 切换到 contact-aware wrist control mode,主动维持 wrist-object contact。这个设计让 tracker 不需要从 perception 推断"现在该 grip 了",而是直接接受 VLK 的高层指令。

**直觉**:这种分层让 tracker 可以在 sim 中用大量 diverse motion 训练(因为 reference 可以是任意 kinematic trajectory),不受 perception 数据限制。VLK 在 real-world 处理 perception,tracker 处理 dynamics。这跟 humanoid teleoperation 工作(OmniH2O [https://arxiv.org/abs/2406.18260]、GMT [https://arxiv.org/abs/2506.14770])的 philosophy 一致。

---

## 7. Deployment 系统

### 7.1 实时性架构

部署架构很精细(Figure 7 和 Table 4),有三个 concurrent process 在 tethered laptop 上:

1. **State estimator**:估计 robot root pose
2. **Whole-body tracker**:50Hz on RTX 5000 Ada,per-tick 4.3ms
3. **VLK inference client**:管理最新 image + robot state,通过 websocket 把 request 发到 external GPU server

VLK inference 在 RTX 5090 上 31ms,end-to-end replan ~63ms。因为每个 chunk 覆盖 1 秒(30 帧),replan period ~555ms,所以有 8.8× headroom,不会 backlog。

### 7.2 Real-time Chunking with 10-frame Overlap

借鉴 $\Psi_0$ [https://arxiv.org/abs/2601.xxxxx] 的 chunking idea。相邻 chunk 有 10 frame overlap(1/3 秒),用某种 blending 让 transition 平滑。这避免了 hard switch 时的 discontinuity。

### 7.3 Motion Blur 处理

这是个很实用的 trick。G1 在快速弯腰 pick 物体时,head camera 会 motion blur,degrade VLK prediction。

**训练时**:对部分 image 加 synthetic motion blur:

$$I' = \mathcal{B}_\sigma(I), \quad \sigma \sim p(\sigma)$$

- $I$:原始 image
- $\mathcal{B}_\sigma$:blur operator(σ 控制模糊强度)
- $p(\sigma)$:限制在小范围,保留 semantic content

**部署时**:维护 0.3 秒 image buffer,选 sharpest frame:

$$S(I) = \text{Var}(\nabla^2 I)$$

- $\nabla^2 I$:image Laplacian operator
- $S(I)$:Laplacian 的 variance,sharp image 有高频内容,variance 大;blur image 高频被抹平,variance 小

$$I_t^* = \arg\max_{I_t^{(i)}} S(I_t^{(i)})$$

这是经典的 sharpness measure,在 computer vision 里常用。VLK 把它用到 robotics deployment,因为 VLA inference 速率比 camera frame rate 慢,可以 buffer 多个 frame 选最好的。

### 7.4 Latency Breakdown(Table 4)

| Stage | Latency (ms) | Share |
|---|---|---|
| Image fetch from camera buffer | 5.4 | 9% |
| Observation packing | 7.5 | 12% |
| Server roundtrip + GPU flow-matching sampling | 37.0 | 59% |
| Output denormalization | 4.2 | 7% |
| World-frame transform + FK | 7.2 | 11% |
| Reference merge into tracker stream | 3.4 | 5% |
| **Total** | **63.0** | **100%** |

GPU flow-matching sampling 占 59%,是 bottleneck。这暗示未来优化方向:distill flow matching 到 fewer-step inference,或者用 consistency model。

---

## 8. Experiments 深入解析

### 8.1 Full-system Evaluation(Table 1)

Real-world 结果:

| Task | Lab Scene | Apartment Scene |
|---|---|---|
| Walk To | 20/20 | 19/20 |
| Turn Around | 20/20 | 18/20 |
| Pick (Floor) | 16/20 | 18/20 |
| Put (Floor) | 20/20 | 20/20 |
| Pick (Surface) | 11/20 | 13/20 |
| Put (Surface) | 8/20 | 15/20 |

几个观察:
1. **Navigation 几乎完美** — 这说明 synthetic walking data 质量高,sim-to-real gap 小
2. **Floor pick/put 强** — 因为 OMOMO 的 box lifting 数据覆盖了 floor-level interaction
3. **Surface pick/put 弱** — paper 解释:retargeted OMOMO 数据对 different support-surface heights 覆盖不足,grasp 不可靠
4. **Pick without contact label = 0/5** — 这个 ablation 验证了 contact label 的必要性。没有 contact label,tracker 不知道何时 grip,无法稳定 pick

这个 last point 是 paper 最强的 ablation 之一:**contact label 作为 kinematic state 的 auxiliary component,不是 nice-to-have,是必需的**。它把 high-level perception 决策(何时 grip)和 low-level dynamics control 解耦,让 tracker 可以专注于执行。

### 8.2 Data Volume Ablation(Figure 4)

Pick (Surface) 从 10% data 的 0% success 提升到 full data 的 46% success。Navigation 在 10% data 就已经接近饱和。

**直觉**:Navigation 是相对简单的 locomotion task,synthetic 数据快速覆盖 motion manifold。Pick/put 涉及 contact-rich manipulation,需要更多 data 覆盖不同 object pose、surface height、approach angle 的组合。这暗示未来 humanoid loco-manipulation 的瓶颈在 manipulation data 的 diversity,而不是 navigation。

### 8.3 Domain Randomization Ablation(Table 2)

Walking mode success under visual perturbation:

| Configuration | Success Rate |
|---|---|
| No randomization | 41% |
| Camera randomization only | 48% |
| Lighting randomization only | 87% |
| Full randomization | 90% |

**Lighting 比 camera 重要得多**(87% vs 48%)。直觉:3DGS 重建的 scene 在固定 lighting 下渲染,real-world deployment 时 lighting 变化大(阳光、室内灯、阴影)。Camera extrinsic 在 ZED 2i 标定后相对稳定。所以 lighting DR 是 sim-to-real transfer 的关键。

这个结果对未来工作有指导意义:在 3DGS-based synthetic data pipeline 中,lighting variation 应该是默认配置,不能省。

### 8.4 Simulation vs Real-World Gap

Simulation(Lab Scene)vs Real-World(Lab Scene)对比:

| Task | Simulation | Real-World |
|---|---|---|
| Walk To | 994/1000 (99.4%) | 20/20 (100%) |
| Turn Around | 843/1000 (84.3%) | 20/20 (100%) |
| Pick (Floor) | 731/1000 (73.1%) | 16/20 (80%) |
| Put (Floor) | 991/1000 (99.1%) | 20/20 (100%) |
| Pick (Surface) | 458/1000 (45.8%) | 11/20 (55%) |
| Put (Surface) | 569/1000 (56.9%) | 8/20 (40%) |

Real-world success rate 跟 simulation 大致一致,有时甚至更高(Walk To、Turn Around)。这说明 **sim-to-real gap 在这个 pipeline 上被很好地闭合了**。这有点反直觉(通常 sim > real),可能因为:
1. Simulation evaluation 用了 1000 rollouts,覆盖更多 corner case
2. Real-world trial 选择的是 representative scenario,不是 worst case
3. Domain randomization 在 sim 中训练时已经让 policy robust 到 visual variation

---

## 9. 我的 Intuition 和 Critique

### 9.1 VLK 的核心贡献是什么?

我认为 VLK 的真正贡献不是某个 single technical novelty,而是 **system-level integration**:

1. 3DGS 提供 photorealistic scene
2. Conditional diffusion 提供 kinematic trajectories
3. Hindsight rendering 提供 paired egocentric observation
4. Flow matching policy + contact label 提供 perception-to-kinematics mapping
5. SceneBot tracker 提供 kinematics-to-action execution
6. Real-time chunking + motion blur handling 提供 robust deployment

每块单独看都有 prior work,VLK 把它们组装成一个可以 sim-to-real 的完整系统,并在 physical G1 上 demonstrate 48K synthetic data 训练的 policy 能做 navigation 和 box transport。这种 system paper 的价值在于 **de-risking**:证明了这条路可行,后续工作可以专注优化单块。

### 9.2 关键设计选择的 Intuition

**为什么用 kinematic trajectory 而不是直接预测 joint torque 或 end-effector pose?**

Joint torque 需要 dynamics model,synthetic data 没有真实 dynamics。End-effector pose 损失 whole-body coordination(legs 怎么配合 reach)。Kinematic trajectory 是 **embodiment-agnostic 的中间表示**,可以 retarget,可以让 tracker 处理 dynamics。这是 decoupling 的关键。

**为什么 contact label 是 binary 而不是 continuous force?**

Binary label 足以指示"现在 grip / release",而 continuous force 需要 force sensing 和 precise dynamics model。Tracker 在 sim 中训练时可以用 privileged force info,但 real-world deployment 没有 wrist force sensor。Binary label 是 minimal sufficient signal。

**为什么 hindsight rendering 而不是 forward simulation with sensor?**

Forward simulation 需要 sensor model,渲染慢,且 perception noise 难以 model。Hindsight rendering 直接从 3DGS 渲染 view,质量高、速度快(8.3 小时 1000 trajectories 的所有 frames)。

### 9.3 Limitations 和未来方向

Paper 自己列了几个:
1. OMOMO 限制于 large objects,小物体(杯子、工具)不行
2. Tracker 用 wrist-object contact 稳定 large object,不能 precise grasping

我额外看到的:
1. **Scene annotation 仍是 manual** — 虽然一次性,但 scale 到 hundreds of scenes 时仍是 bottleneck。未来应该用 SAM-style 自动 semantic segmentation
2. **Interaction diversity 受限于 OMOMO** — 只能 box-like bimanual transport。要扩展到单手 tool use、articulated object,需要更丰富的 interaction dataset
3. **No closed-loop replanning in tracker** — Tracker blind 到 perception,如果 VLK 预测错,tracker 会一直执行错的 trajectory 1 秒。Real-time chunking 缓解了这个问题,但 fundamental issue 还在
4. **No language compositionality** — Instruction 是 template-based,不能处理 "walk to the chair then pick up the box and put it on the table" 这种 long-horizon。Paper 在 Figure 1 bottom 展示了 chaining,但似乎是 manual orchestration

### 9.4 跟相关工作的 Positioning

- **WholeBodyVLA [https://arxiv.org/abs/2610.xxxxx]** 和 **$\Psi_0$**:用 real teleoperation data,数据 quality 高但 scale 受限。VLK 用 synthetic data,quality 受 generator 限制但 scale 大。两者 complementary
- **GR00T N1 [https://arxiv.org/abs/2503.14734]**:NVIDIA 的 foundation model,大规模预训练。VLK 可以视为 GR00T N1 的 specialized fine-tuning pipeline
- **LeVERB [https://arxiv.org/abs/2506.13751]**:latent VL instruction,把 language 嵌入 latent space。VLK 用 explicit kinematic trajectory,更 interpretable 但 less flexible

### 9.5 对你(Andrej)的 community 的意义

VLK 展示了一个 **data-centric 范式**:不是堆 model size,而是堆 diverse synthetic data。这跟 LLM 的 scaling law 哲学一致 — 数据规模和质量决定 policy 能力。如果未来 humanoid robot 要做 general loco-manipulation,这条路比 teleoperation scale up 更可行。

但是这条路有几个 open question:
1. Synthetic data 的 **physical fidelity ceiling** — 3DGS 重建的 scene 在 unobserved region 是空的,object geometry 是 static 的
2. **Distribution shift** — synthetic motion generator 的 distribution 跟 real-world task distribution 的 gap 怎么 measure 和 close
3. **Compositionality** — 怎么从 atomic skill(chain 起来)到 long-horizon task,需要 hierarchical planning 或 in-context learning

---

## 10. 总结

VLK 的核心 thesis:**humanoid loco-manipulation 的 perception policy 可以从 synthetic interactions in reconstructed scenes 学到**。它通过 Real2Sim2Real pipeline(3DGS reconstruction + conditional diffusion motion synthesis + hindsight rendering)自动生成 48K paired (vision, language, kinematics) 数据,训练 flow-matching VLK policy 预测 short-horizon whole-body kinematic trajectory + contact labels,然后用 contact-aware SceneBot tracker 在 physical G1 上执行。

关键 technical 亮点:
- **Decoupling**:VLK 做 perception → kinematics,tracker 做 kinematics → action,各自优化
- **Hindsight rendering**:用 privileged info 生成 motion,事后渲染观测,绕开 active perception
- **Contact label 作为 kinematic state**:binary 信号解耦 grip 决策和 dynamics 执行
- **$x_0$-prediction flow matching**:从 $\pi_{0.5}$ init,直接预测 clean trajectory
- **Real-time chunking + motion blur selection**:deployment robustness

Experiments 在 physical Unitree G1 上 demo 了 navigation 和 box transport,sim-to-real gap 通过 domain randomization(lighting 为主)显著缩小。Limitations 在 small object manipulation 和 long-horizon compositionality,这些是未来工作方向。

---

**Relevant Links**:
- Project page: https://vision-language-kinematics.github.io
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\pi_0$: https://arxiv.org/abs/2410.24164
- CHOIS: https://arxiv.org/abs/2312.16205
- OMOMO: https://arxiv.org/abs/2312.16205
- OmniRetarget: https://arxiv.org/abs/2506.07768
- SceneBot: https://arxiv.org/abs/2606.27581
- BONES-SEED: https://huggingface.co/datasets/bones-studio/seed
- Unitree G1: https://www.unitree.com/g1/
- SplatSim: https://arxiv.org/abs/2409.10161
- RoboGSim: https://arxiv.org/abs/2411.11839
- GaussGym: https://arxiv.org/abs/2409.10161
- OmniH2O: https://arxiv.org/abs/2406.18260
- BeyondMimic: https://arxiv.org/abs/2508.08241
- Zhou et al. 6D rotation: https://arxiv.org/abs/1812.07035
- Basis Point Set: https://arxiv.org/abs/1906.04390
- Twist2: https://arxiv.org/abs/2511.02832
- CLONe: https://arxiv.org/abs/2506.08931
- GR00T N1: https://arxiv.org/abs/2503.14734
- LeVERB: https://arxiv.org/abs/2506.13751
- EgoAllo: https://arxiv.org/abs/2501.08535
- Running VLAs at real-time speed: https://arxiv.org/abs/2510.26742
