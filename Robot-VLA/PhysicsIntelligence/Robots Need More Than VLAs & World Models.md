---
source_pdf: Robots Need More Than VLAs & World Models.pdf
paper_sha256: 34c17cc1389cc35ea952a7eb159df855d2a0f11b526e2cc1300e13fc24f3ce41
processed_at: '2026-08-12T02:03:06-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

现在的 robotics 大家都在拼命 collect robot 数据、train 大 VLA、scale up model，但作者说你们搞错重点了 —— 真正的瓶颈是**怎么把世界上已经存在的海量"物理行为"（人类视频、simulation、穿戴设备记录、robot 自己 rollout 的失败）转换成 robot 能学的 supervision**。VLA 只是 stack 里一层，光 scale 它没用。

---

## 为什么现在这条路走不远

想象一下 LLM 的成功是因为什么：internet 上有海量 text，天然数字化，天然有人类 supervision，scaling law 一放大就 work。

Robotics 呢？世界上海量的"物理行为"其实存在 —— YouTube 上有人做饭、工厂有 workflow、家里有人收拾东西、robot 自己每天在 lab 里成功失败无数次。**但这些东西 robot 没法直接学**，因为它们缺了几个关键 ingredient：
- 没有 robot action label（视频里看不到 motor command）
- 没有 embodiment 信息（人类的手 ≠ robot gripper）
- 没有明确的 task phase / reward / success 信号
- 没有 contact、force 这些物理细节

所以现在大家只能去 lab 里 teleop collect demos，一条一条来。这个数据量跟 internet 相比是 nine orders of magnitude 的差距。**你 scale VLA 也好，collect 更多 demos 也好，天花板就在那 —— curated robot data 的总量就那么多**。

---

## 作者的 vision：把"整个世界的物理行为"变成 supervision

这篇 paper 真正的 contribution 是提出一个 **compounding system**，由四个 pillar 组成。我挨个讲。

### Pillar 1: Physical Data Engine（自动给物理行为打标签）

现在 human video 在 robotics 里的用法基本是"pretrain 一个 visual representation"就完事了。作者说这太浪费了 —— 一个人在 YouTube 上切菜的视频，里面其实包含了：
- 什么时候开始 reach knife
- 什么时候 contact 到 knife handle
- grasp 稳了没
- 切下去的时候刀的角度、力道
- 切到第几片了（task progress）
- 最后成没成功

这些信息全在 video 里，但没人把它们 extract 出来变成 robot 能用的 label。

作者提出的 physical data engine 就是干这个的：输入是异构的 raw data（video + motion capture + tactile + language），输出是一串 latent physical events，每个 event 带 object state、contact、task phase、latent action、reward。

**直觉类比**：这就像给 video 做"物理 captioning"。普通 captioning 说"一个人在切菜"；physical data engine 说"frame 30-55: reach-to-knife，contact-begins @ 1.8s，grasp-stable @ 2.1s，knife pose = [x,y,z,quat]，grasp force ≈ 15N，task phase = preparation，progress = 0.2"。

这个 label 就可以直接拿来 train perception model、reward model、world model、甚至 policy。

### Pillar 2: Task-Preserving Retargeting（把人类动作翻译成 robot 动作）

即使你把人类行为 parse 出来了，还有一个 embodiment gap：人类的手 ≠ robot gripper ≠ dexterous hand ≠ humanoid arm。

现在大家做 retargeting 基本就是"copy joint trajectory"或者"match end-effector pose"。作者说这太 weak 了 —— 你应该 preserve 的是**task-relevant physical effect**，不是 pose。

举例：人类开 drawer 是手指勾住 handle 往外拉。Robot gripper 没有手指，但它可以用 parallel jaw 夹住 handle 往后拉。pose 完全不一样，但 **drawer 被打开了**这件事是一样的。这才是 retargeting 真正要 preserve 的 invariant。

作者把 retargeting 分成四个 level：

1. **Pose level**：直接 copy 关节角度 —— 最弱
2. **Contact level**：保证 gripper 接触对的 surface、对的 timing
3. **Object-state level**：保证 drawer 真开了、cup 真被 lift 了
4. **Intent / Skill level**：motion 完全不同，但 task 完成了 —— 最强

Generalist robotics 要走到 level 3-4，而不是停在 level 1。

### Pillar 3: Physics-Grounded World Model（预测"物理后果"而不是"视觉未来"）

现在 world model 这块很火，大家都在 generate video。但作者说 robotics 的 WM **不是 video generator**，而是 **consequence predictor**。

区别在哪？Video generator 关心"未来看起来像不像真的"。Robot WM 关心"如果我这么 action，drawer 会不会开？cup 会不会掉？peg 能不能插进去？"

这意味着：
- WM 应该 predict 的是 task-relevant physical variables（object pose、contact、force、deformation），不是 pixel
- WM 应该 **task-conditioned**：开 drawer 的时候不用关心 background 长啥样，倒水的时候不用关心 table 纹理
- WM 应该 **physics-grounded**：不能 object 互相穿透、contact 没 force、rigid object 莫名变形

作者提到几个 promising 方向：3D Gaussian Splatting + differentiable physics、object-centric world model、V-JEPA 2 路线（representation space prediction）。

还有一个关键 point：**WM 必须有 uncertainty quantification**。如果 WM 在 OOD 区域 hallucinate，robot 会基于错误 prediction 做错误 action，把自己推到更 OOD 的区域，WM 进一步 hallucinate —— vicious cycle。这是 model-based RL 的经典坑，但现在 VLA + WM 时代又回来了。

### Pillar 4: Self-Improving Deployment Loop（部署时持续学习）

这是我觉得最 underexplored 也最有意思的一个 pillar。

现在的 pipeline：train policy → deploy → evaluate → 发论文。Deployment 只是 evaluation。

作者说 deployment 应该是 **supervision source**。每次 robot 跑一个 episode，不管成功失败，都应该被 parse 成 structured supervision 反哺给整个 system。

但这里有个关键问题：**component-level credit assignment**。Robot 失败了，到底是谁的锅？
- 是 policy 选错了 action？→ update policy
- 是 WM 预测错了后果？→ update WM
- 是 retargeting 没保持住 task effect？→ update retargeting
- 是 reward model 把成功标成失败了？→ update reward model

如果分不清就瞎 update，系统会越来越 confused。比如 WM 预测 drawer 会开但实际卡住了，这时候 update policy 没用 —— policy 已经执行了"正确"的 action，是 WM 错了。

所以 self-improving loop 的核心是 **diagnosing failure root cause**，然后 route 到正确的 component 去更新。

---

## 整个 system 长啥样

```
世界上的物理行为（video, wearable, sim, robot rollout）
        ↓
   [Physical Data Engine] ← 自动打标签
        ↓
   latent physical events: object state, contact, phase, action, reward
        ↓
   [Retargeting] ← 翻译成 robot 能执行的 action
        ↓
   [World Model] ← 预测这个 action 的物理后果
        ↓
   [VLA Policy] ← 执行
        ↓
   real-world deployment
        ↓
   [Reward Grounding + Credit Assignment] ← 诊断成功/失败，分配 credit
        ↓
   feedback 回到 Data Engine, WM, Retargeting, Policy
        ↓
   下一轮部署更强
```

关键：这不是 feedforward stack，是 **closed loop**。四个 pillar 互相监督、互相修正。每一次 deployment 都让整个 system 变强一点，compounding 上去。

---

## 作者真正想说的

**下一个 robotics foundation model 不会是一个 monolithic VLA，而是一个 compounding system**。这个 system 把世界上所有物理行为都变成 supervision source，通过 grounding stack 转换成 robot-usable signal，部署时持续 self-improve。

VLA 是这个 stack 里的 policy interface，很重要，但只是其中一层。光 scale VLA 而不 build grounding stack，就像只 scale decoder 而不 pretrain encoder —— 天花板很快就到。

---

## 我的直觉

这篇 paper 之所以重要，是因为它精确指出了现在 robotics 圈的一个集体盲点：大家都在比谁的 VLA 更大、谁的 dataset 更多、谁的 benchmark 更难，但很少有人问"我们的 supervision source 是不是 fundamentally limited"。

作者说 yes，而且给出了一个 coherent 的 alternative vision。这就像当年 ImageNet 之前大家都在手工设计 feature，突然有人说"我们应该让 model 从 raw pixel 学 feature"。现在 robotics 就是在手工 design supervision（teleop demos），作者说我们应该让 system 从 raw physical behavior 学 supervision。

从 engineering 角度看，四个 pillar 里我觉得最 promising 的是：
1. **Wearable sensing → autolabelling**（最 cheap 的 grounding signal，比 teleop 便宜两个数量级）
2. **3DGS + differentiable physics WM**（geometry + contact + dynamics 在一个 representation 里）
3. **V-JEPA 2 路线**（representation-space prediction + small robot fine-tuning，最有 scaling potential）

最 underexplored 但 critical 的是 **component-level credit assignment in deployment** —— 这是一个非常 hard 的问题，但如果解决了，self-improving loop 就能真正跑起来。

---

# Robots Need More Than VLAs & World Models — 深度解读

## 一、这篇 paper 的核心 thesis

这是 Motoniq.ai 联合 Stanford、ETH Zurich、IIT、TU Darmstadt、UCL 的一篇 position paper。核心论点可以浓缩为一句话：**generalist robotics 的瓶颈不在 policy 本身，而在缺少 grounding 机制把世界上大量非结构化的 physical experience 转换成 robot-usable supervision**。

论文构造了一个非常有意思的对比：

| LLM/Vision 时代 | Robotics 时代 |
|---|---|
| text/images 自然数字化 + 密集 human supervision | physical interaction 数据海量但无法直接用 |
| internet ≈ 自然 corpus | robot dataset ≠ internet，每条轨迹必须物理可执行 |
| scaling law 直接放大效果 | scaling 受 grounding 瓶颈约束 |

作者们提出 robotics 缺一个 "robotics internet" —— 不是数据量问题，而是数据"可消化性"问题。VLA 和 world model 都只是这个 stack 中的某一层。

参考链接：
- Position paper trend: https://arxiv.org/abs/2506.09985 (V-JEPA 2，类似 thesis 的代表)
- Physical Intelligence π₀: https://arxiv.org/abs/2410.24164
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## 二、Section 2：当前 robot learning 的三个 supervision regime

作者把现有工作分成三类，每一类都暴露出 grounding bottleneck 的一个面：

### 2.1 Robot-Native Supervision（最有用但最贵）

**定义**：数据已经 expressed in coordinate system of robot learning problem — observation 配 robot action + task label + reward。

这一段相当于一个 robot foundation models 的"百科全书式 survey"，我把它整理成一个 scaling 时间线：

| 系统 | Year | Scale | 关键贡献 |
|---|---|---|---|
| RoboNet | 2019 | 15M frames, 7 platforms | 第一次跨平台 dataset sharing |
| BC-Z | 2022 | 100+ tasks | zero-shot task generalization，video/language conditioning |
| RT-1 | 2022 | 130K episodes, 13 robots, 700+ tasks | transformer policy at scale |
| RT-2 | 2023 | VLM + robot | action tokenization，web knowledge transfer |
| Open X-Embodiment | 2024 | 1M+ trajectories, 22 embodiments | cross-embodiment 标准化格式 |
| Octo | 2024 | 800K trajectories | open-source generalist，可适配新 obs/action space |
| OpenVLA | 2024 | 7B params, 970K demos | 开源 7B VLA |
| π₀ | 2024 | flow-matching on VLM | continuous action via flow matching |
| SpatialVLA | 2025 | 1.1M episodes | 显式 spatial representation |
| RDT-1B | 2024 | 1M+ episodes | diffusion-transformer for bimanual |
| GR00T N1 | 2025 | egocentric video + robot + sim | humanoid dual-system VLA |
| Gemini Robotics | 2025 | Gemini-based VLA | on-device variant，cross-platform |
| Helix | 2025 | Figure humanoid | full upper-body unified VLA |
| LeVERB / WholeBodyVLA / HuMI / HEX | 2025-26 | humanoid-specific | latent action vocab，loco-manipulation |

**作者的关键 observation**：所有这些进展都依赖于已经 grounded 的 supervision。actions、task labels、embodiment constraints、success signals 都是事先 curated 的。VLA scaling 本质上 still bounded by "已 grounded data 的总量"。

我的 intuition：这就像 LLM 如果只能从精标注的 instruction tuning data 训练，而没法 pretrain on internet text。Robotics 需要 pretraining equivalent，但要解决 grounding。

### 2.2 Learning from Weakly Grounded Physical Observations

这是 paper 中最技术化的一段。作者把"video → robot supervision"的转换用一个非常清晰的 latent variable formulation 表达：

观测：$o_{1:T} = \langle o_1, \ldots, o_T \rangle$
目标 action 序列：$a_{1:T} = \langle a_1, \ldots, a_T \rangle$（不可观测）
latent action：$z_t \sim q(\cdot \mid o_t, o_{t+1}, L_t, L_{t+1})$

**变量解释**：
- $o_t$：第 t 帧 observation（RGB/depth/feature）
- $L_t$：第 t 时刻的 language caption / metadata
- $z_t$：latent action，解释从 $o_t$ 到 $o_{t+1}$ 的 transition
- $q$：inference distribution（approximate posterior）

**为什么这是个漂亮的 formulation**：它把 "video 缺 action 标签" 问题转化为一个 latent variable inference 问题。z 是任务相关的 physical change 的 compressed code —— 抓、移动、放置、对齐 —— 但还没有 tie 到具体 embodiment。

作者把现有工作分成四类 passive video supervision：

**(1) Representation Learners**（不直接出 action，只学 feature）
- R3M (Ego4D pretrain, time-contrastive + video-language alignment + sparsity)
- VIP (value-implicit pretraining, temporal distance = task progress)
- MVP (masked visual pretraining)
- VC-1 (大规模 visual pretraining as substrate)

参考：R3M https://arxiv.org/abs/2203.12601，VIP https://arxiv.org/abs/2210.00030

**(2) Latent-Action Approaches**（直接从 video 学 action-like code）

LAPA 是这一类最重要的代表，方法三步走：
1. 在 video transitions 上用 VQ-VAE-style objective 学 discrete latent action space
2. 训 latent VLA model：输入 observation + task description，预测 latent action
3. 用少量 robot data fine-tune：map latent → executable robot action

UniVLA：把这套机制扩展到 cross-embodiment，task-centric latent actions 无监督学习，不依赖 action labels。

但作者强调了一个 subtle 但关键的 point：**这些 latent variables 本质上是 "transition codes" 或 "physical-change descriptors"，只有当一个 embodiment-conditioned decoder 能把它们 map 到实际 robot command 并 reproduce intended physical change 时，才变成真正的 "action"**。这是一个非常重要的概念切割，避免大家把 latent action 直接等同于 robot action。

**(3) Task-Progress Signals**（从 video 推 reward）
- PROGRESSOR: task-agnostic reward from unlabelled video + self-supervised refinement
- Adapt2Reward: VLM → language-conditioned reward function
- ReWiND: video rewinding + misaligned video-language pairs as negative supervision
- TimeRewarder: temporal distance between frame pairs
- SARM: stage-aware reward with dense subtask labels

**(4) Behavioural Priors**（affordance, contact structure, temporal task structure）

**作者的关键 takeaway**：weak physical supervision **没有消除** grounding 问题，只是**把它 relocate 了**。Latent action ≠ robot command，progress signal ≠ reward for new embodiment，human strategy 未必 robot executable。

### 2.3 Generating Physical Experience

第三条路：自己 generate 数据。作者用一个非常 functional 的视角定义了 simulation/world model 的价值 —— 不是 visual realism，而是 **counterfactual experience 的物理保真度**。

四条 sub-route：

**(1) Simulation Route**：RLBench (100 tasks), Meta-World, ManiSkill, CALVIN, LIBERO (130 tasks)

**(2) Data Generation Route**：
- MimicGen：从 <200 seed demos 自动生成 50K+ demos，跨 18 tasks
- RoboCasa → RoboCasa365：365 tasks, 2500 kitchens, 2000+ hours
- RoboGen：foundation models 自动 construct tasks/scenes/data

**(3) Real-to-Sim-to-Real Route**：
- RialTo：digital twin + RL robustify
- RL-GSBridge：3D Gaussian Splatting + zero-shot sim-to-real
- Real-is-Sim：Embodied Gaussians throughout pipeline
- RoboGSim：interactive real2sim2real Gaussian-splatting platform
- SOUS VIDE / SINGER / GRaD-Nav / GRaD-Nav++：navigation-focused，利用 3DGS end-to-end differentiability

**(4) World-Model Route**（最技术密集的一段）：

历史脉络：
- Schmidhuber "making the world differentiable"（早期 RL+planning with RNN）
- Ha & Schmidhuber "World Models" (2018)：generative recurrent model + policy in dreamed rollout
- PlaNet, Dreamer, DreamerV3, DayDreamer：latent dynamics from pixels，dreamer-style works on real robots

最新 generation：
- RoboDreamer：compositional video world models，factorize by language-derived primitives
- UniSim：universal interactive simulator from heterogeneous data
- Genie (DeepMind)：spatiotemporal tokenizer + autoregressive dynamics + latent action model from unlabelled video

但作者强调 robotics 比 visual plausibility 严格得多 —— 必须保持 3D geometry, object permanence, contact, material, force, constraints。所以引出：

**Object-centric / 3D / Physics-Grounded World Models**：
- FOCUS：object-centric for manipulation
- Language-guided object-centric world models
- PointWorld：unify state+action in shared spatial domain，full-scene 3D point flow from RGB-D
- ParticleFormer：transformer-based 3D pointcloud world model，multi-object multi-material，supports MPC

**Uncertainty Quantification**（作者单独强调，这点很关键）：
- Mei et al.：latent uncertainty via VAE，statistically calibrated
- Li et al.：uncertainty for WM used as RL training environment
- Ward et al.：calibrated uncertainty in latent space detects runtime VLA policy failures

这里作者点出一个**致命的"vicious cycle"风险**：WM 在 OOD 时 hallucinate → bad control choice → 把 system 推到更 OOD 区域 → 进一步 hallucinate → 进一步 degenerate control。这是 model-based RL 的经典陷阱，作者用 modern context 重新强调，非常准确。

**Physics-Informed Structured Models**：
- PILCO (GP dynamics, 2011)：经典 model-based RL
- Deep Lagrangian Networks：impose Lagrangian mechanics structure on neural dynamics
- Hamiltonian Neural Networks：learn Hamiltonian function，use Hamilton's equations
- Lagrangian Neural Networks：parameterize Lagrangian，derive via Euler-Lagrange
- Symplectic ODE-Net
- Interaction Networks, Neural Physics Engines, Graph Networks as Physics Engines
- Graph-network simulators scaling to particle-based fluids/materials

**JEPA Family**：
- LeCun's "A Path Towards Autonomous Machine Intelligence" 提出 WM for planning under uncertainty
- I-JEPA：image-based joint-embedding predictive learning
- V-JEPA：video，predict masked parts in representation space
- V-JEPA 2：combine internet video + small robot interaction data，show prediction/planning/zero-shot robot control —— **作者明确说这是 WM 和 embodied control 之间最清晰的 link 之一**

**Gaussian-Splatting-based WMs**：
- Physically Embodied Gaussian Splatting：dual Gaussian-particle representation，couple visual rendering with particle-based physics + online correction
- Gaussian World Models：3DGS as dynamic world representation，action-conditioned 3D video prediction
- ContactGaussian-WM：unified Gaussian for appearance + collision geometry，differentiable contact dynamics，closed-form physical reasoning，support MPC
- PIN-WM：physics-informed WM for non-prehensile / deformable manipulation

**作者核心 insight**：world model 的真正价值不是生成 visually plausible future，而是让 robot 可以做 **counterfactual reasoning** —— "如果我换一个 push point 会怎样？换 grasp orientation 会怎样？insert 角度稍微偏一点会怎样？"

---

## 三、Section 3：四个 Missing Pillars（这篇 paper 的核心贡献）

这是作者真正提出的 constructive vision。我把它看成一个 **grounding-centric pipeline** 的 spec：

### 3.1 Physical Data Engine & Embodied Autolabelling

**Raw episode 的形式化**：

$$\mathbf{x} = \{(v_i, \tau_i^{(v)})_{i=1}^{T_v}, (m_j, \tau_j^{(m)})_{j=1}^{T_m}, (h_k, \tau_k^{h})_{k=1}^{T_h}, (r_l, \tau_l^{(r)})_{l=1}^{T_r}, L\}$$

**变量解释**：
- $v_i$：第 i 个 video frame，$\tau_i^{(v)}$ 是其 timestamp
- $m_j$：第 j 个 motion-capture / wearable / body-pose measurement
- $h_k$：第 k 个 tactile / force / contact / hand sensor reading
- $r_l$：第 l 个 raw robot log entry（proprioception, deployment metadata）
- $L$：language（instruction, caption, task description, human correction）
- $T_v, T_m, T_h, T_r$：各 modality 的长度，**通常不同且异步**

**Alignment variable**：

$$\mathcal{A}: \{\tau_i^{(v)}, \tau_j^{(m)}, \tau_k^{(h)}, \tau_l^{(r)}\} \to \{1, \ldots, Z\}$$

$\mathcal{A}$ 把所有 modality 的 timestamp map 到 latent event timeline $\zeta \in \{1, \ldots, Z\}$。例：video frames 30-55 + motion readings 102-180 + tactile spike @1.8s → 都对应 $\zeta = 2$ "contact-begins"。

**Per-event latent structure**：

$$\mathbf{z}_\zeta = [\mathbf{s}_\zeta, \mathbf{c}_\zeta, \phi_\zeta, \mathbf{u}_\zeta, \mathbf{r}_\zeta]$$

- $\mathbf{s}_\zeta$：object-centric physical state（pose, relation, velocity）
- $\mathbf{c}_\zeta$：contact / interaction label（where, what type, force magnitude proxy）
- $\phi_\zeta$：task phase（reach, contact, grasp, transport, place, release, verify）
- $\mathbf{u}_\zeta$：latent physical action / transition code（task-relevant change）
- $\mathbf{r}_\zeta$：task-conditioned progress / reward scalar

**Episode-level latent**：$\mathbf{z} = [\mathbf{z}_{1:Z}, \mathbf{g}, \mathbf{y}]$
- $\mathbf{g}$：goal（inferred from language or behavior）
- $\mathbf{y}$：outcome（success / failure / partial / unsafe）

**总 inference model**：

$$q_\theta(\mathbf{z}, \mathcal{A} \mid \mathbf{x})$$

**这是一个 joint inference problem**，不是 perception 加 post-processing。$q_\theta$ 同时做 temporal alignment, event segmentation, object-state estimation, contact inference, phase recognition, latent-action discovery, reward grounding, outcome prediction。

**为什么这是关键 innovation**：现有的 pipeline 是 sequential 的（detect → track → segment → label → ...），错误会 propagate。这里把它 formulate 成一个 joint inference，所有 latent variable 互相约束 —— contact 解释 object state，object state 解释 task phase，task phase 解释 reward。这很像现代多模态 LLM 中 token-level joint distribution 的思路，但 lift 到 physical event level。

**例子**：human 穿 sensing suit 把 cup 放到 tray 上。Inferred event sequence：
- $\zeta=1$: reach-to-cup
- $\zeta=2$: contact-begins  
- $\zeta=3$: grasp-stable
- $\zeta=4$: lift
- $\zeta=5$: transport-to-tray
- $\zeta=6$: place
- $\zeta=7$: release

每个 event 都附带物理描述（cup pose, contact normal, grasp force 等），而不只是 caption "person places cup on tray"。

### 3.2 Task-Preserving Retargeting

**形式化**：

$$\mathbf{a}_\zeta^{(\text{embodied})} = f_\psi(\mathbf{u}_\zeta, \mathbf{s}_\zeta, \text{embodiment})$$

使得：

$$\Delta_{\mathbf{g}}(\mathbf{s}_\zeta, \mathbf{a}_\zeta^{(\text{embodied})}) \approx \Delta_{\mathbf{g}}(\mathbf{s}_\zeta, \mathbf{u}_\zeta)$$

**变量解释**：
- $f_\psi$：retargeting function with parameters $\psi$
- $\Delta_{\mathbf{g}}$：task-relevant effect function under goal $\mathbf{g}$
  - drawer opening：$\Delta_{\mathbf{g}}$ = drawer displacement
  - placing：$\Delta_{\mathbf{g}}$ = object pose change
  - insertion：$\Delta_{\mathbf{g}}$ = relative alignment
  - packing：$\Delta_{\mathbf{g}}$ = containment relation
  - grasping：$\Delta_{\mathbf{g}}$ = contact state

**Hierarchy of invariants**（这是我觉得 paper 中最 elegant 的概念结构之一）：

| Level | Preserves | Weakness | Example |
|---|---|---|---|
| Pose | joint trajectory | 最弱 | 直接 copy human joints |
| Contact | touch surface + timing | 中 | 确保 gripper 接触正确 surface |
| Object-state | physical transition | 强 | drawer 真打开 / cup 真被 lift |
| Intent / Skill | task effect | 最强 | 完全不同 motion，完成同样 task |

**关键 insight**：generalist robotics 需要从 pose-preserving imitation 走到 task-effect-preserving translation。Wearable sensing 之所以有价值，正是因为它 expose 的是 hierarchy 中 higher level 的 invariant（contact, force, object state, phase boundary），这些比 raw joint angle 更 transferable。

### 3.3 Physics-Grounded World Models for Consequence Prediction

**形式化**：

Task-level consequence prediction:
$$\mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta, \mathbf{u}_\zeta, \mathbf{g})$$

Embodiment-specific prediction:
$$\mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta, \mathbf{a}_\zeta^{(\text{embodied})}, \text{embodiment}, \mathbf{g})$$

**变量解释**：
- $p_\omega$：world model with parameters $\omega$
- $\mathbf{s}_\zeta$：object-centric state at event $\zeta$
- $\mathbf{u}_\zeta$：latent physical action（task-level）
- $\mathbf{a}_\zeta^{(\text{embodied})}$：embodiment-specific executable action
- $\mathbf{g}$：goal（task-conditioning）

**两个用途对应两种 reasoning mode**：
- 第一式：task-level reasoning —— "如果 intended action 是 pull/lift/insert/place，应该产生什么物理 transition？"
- 第二式：embodiment-specific planning —— "如果这个 robot with this morphology + controller 执行这个 action，会怎样？"

**Task-Conditioning 是关键**：作者反复强调 WM 不需要均匀地 predict 所有 future detail，而是 predict **task-relevant aspects**。
- 开 drawer：drawer displacement + handle contact 比 background texture 重要
- 倒水：liquid state + container pose 比 table appearance 重要
- 折布：deformable geometry + contact points 比 pixel-perfect reconstruction 重要

**这意味着 WM 的 objective function 应该 aligned with downstream control，不只是 visual reconstruction**。这呼应了 LeCun JEPA 的 philosophy（predict in representation space）但 push 到 task-conditioned physical variables。

**WM 在 stack 中的角色（4 种 use）**：
1. **Pre-action evaluation**：在 retargeting 提出多个候选 motion 时，WM 评估哪个最可能 establish correct contact / avoid collision / produce desired displacement
2. **Planning**：search over alternatives
3. **Post-failure explanation**：peg insertion 失败时，WM 区分 alignment 不够 / force 不够 / grasp 不稳 / object geometry 问题 / wrong task phase
4. **Training data generation**：counterfactual experience

**WM 和其他 pillar 的耦合**：
- Autolabelled contact + object-state transitions → supervision for WM
- WM → 检测 inconsistent labels / impossible transitions
- Retargeting → use WM rollout 选择 task-effect-preserving action
- Deployment failures → feed back into data engine 作为 WM 错误预测的 example

**Open Challenge — Representation Choice**：

| Representation | Pros | Cons |
|---|---|---|
| Pixel space | broad coverage | weak physical abstraction |
| Object-centric | exposes entities + relations | 需要可靠 perception/tracking |
| 3D (point cloud, mesh, NeRF, 3DGS) | geometry | struggle with contact/force/material |
| Mechanics-based | physical laws | brittle for unknown/deformable env |
| **Hybrid (3D + object-centric + physics constraints + residual)** | actionable + general | hardest to build |

作者的 bet：**hybrid 是最有希望的方向** —— 学习的 models combine 3D scene representation + object-centric structure + physics-inspired constraints + data-driven residual dynamics。

### 3.4 Self-Improving Deployment Loops / Task-Conditioned Reward Grounding

**Task-conditioned reward 形式化**：

$$\mathbf{r}_\eta(\mathbf{s}_\zeta, \mathbf{g}, \phi_\zeta)$$

**变量解释**：
- $\mathbf{r}_\eta$：reward model with parameters $\eta$
- $\mathbf{s}_\zeta$：inferred physical state at event $\zeta$
- $\mathbf{g}$：task / goal
- $\phi_\zeta$：task phase

**关键 insight**：**physical state 本身没有 intrinsic success/failure**。
- cup 静止 on table → "put down cup" 的 success
- cup 静止 on table → "pick up cup" 的 failure
- cup 静止 on table → "open drawer" 的 irrelevant

所以 reward 不能是 state-only function，必须 condition on (state, goal, phase)。这是一个非常精确的 formulation —— reward 是 "physical progress under a goal" 的 interpretation。

**Self-Improving Loop 的完整闭环**：

```
deploy policy 
  → observe outcome 
  → infer task-conditioned progress/success/failure 
  → explain failure or correction 
  → add grounded supervision to data engine 
  → update reward model / world model / retargeting / policy 
  → redeploy
```

**Component-level credit assignment**（这是关键创新，区别于简单 "did it work?"）：
- **Policy update**：when the action was poor
- **World-model update**：when the consequence prediction was wrong
- **Retargeting update**：when the physical effect was not preserved
- **Reward-model update**：when success/failure was misclassified

**为什么这个 credit assignment 重要**：如果没有它，系统只知道 "rollout 失败了" 但不知道改什么。比如，如果 WM 预测 drawer 会开但实际卡住，应该 update WM，而 update policy 是没用的 —— policy 已经 execute 了"正确"的 action。如果 reward model 把一个 success state 标成 failure，反复 update policy 只会让 policy 越来越 confused。所以 **diagnosing 失败的根源** 是 self-improving 的核心。

**Three capabilities required**：
1. **Monitor execution**：detect contacts, state changes, subgoal completion, anomalies, safety violations
2. **Evaluate relative to task**：produce progress / reward / failure labels
3. **Route supervision**：to correct component (policy / WM / retargeting / reward model)

---

## 四、整篇 paper 的 pipeline 总览（我重画一遍）

```
┌─────────────────────────────────────────────────────────────────────┐
│  Heterogeneous Physical Experience                                 │
│  (human video, wearable, internet video, sim, robot rollout, ...)  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────┐
                │  (1) Physical Data Engine        │
                │      q_θ(z, A | x)                │
                │  → temporal alignment            │
                │  → event segmentation            │
                │  → object-state estimation        │
                │  → contact inference              │
                │  → phase recognition              │
                │  → latent-action discovery        │
                │  → reward grounding               │
                │  → outcome prediction             │
                └────────────────┬──────────────────┘
                                 │
                                 ▼  z_ζ = [s_ζ, c_ζ, φ_ζ, u_ζ, r_ζ]
                ┌─────────────────────────────────┐
                │  (2) Task-Preserving Retargeting │
                │      f_ψ(u_ζ, s_ζ, embodiment)   │
                │  → preserve Δ_g invariant       │
                │  → pose / contact / state / intent │
                └────────────────┬──────────────────┘
                                 │
                                 ▼  a_ζ^(embodied)
                ┌─────────────────────────────────┐
                │  (3) Physics-Grounded WM         │
                │      p_ω(s_{ζ+1} | s_ζ, a, g)     │
                │  → consequence prediction        │
                │  → task-conditioned              │
                │  → counterfactual reasoning       │
                └────────────────┬──────────────────┘
                                 │
                                 ▼  predicted outcome
                ┌─────────────────────────────────┐
                │  VLA Policy (execution)          │
                │  π(action | obs, language)        │
                └────────────────┬──────────────────┘
                                 │
                                 ▼  real-world rollout
                ┌─────────────────────────────────┐
                │  (4) Self-Improving Deployment   │
                │      r_η(s_ζ, g, φ_ζ)             │
                │  → monitor execution              │
                │  → task-conditioned evaluation    │
                │  → credit assignment              │
                │    (policy / WM / retarget /     │
                │     reward model)                 │
                └────────────────┬──────────────────┘
                                 │
                                 └────── feedback loop ──────┐
                                                          │
                                                          ▼
                                              (back to Physical Data Engine)
```

---

## 五、我的 critical commentary / intuition building

### 5.1 这篇 paper 真正的 contribution

表面上是 survey，实际上是 **架构 manifesto**。它把 robotics 的未来 foundation model 定义为 **a compounding system**，而不是一个 monolithic model。这是和目前 VLA 主流叙事（"scale up model + scale up data"）的一个明确分歧。

### 5.2 为什么 grounding-centric 比 robot-data-centric 更 fundamental

考虑一个 thought experiment：假设我们有无限人力 collect robot demos，VLA 也无限 scale。会怎样？

答案：仍然 stuck。因为：
- long-horizon task 上的失败模式 infinite
- novel object 上的 affordance 无法 enumerate
- humanoid 级别的 whole-body skill 不可能 teleop 收集

grounding-centric 把"已经在 robot coordinate system"的假设拿掉，让 supervision source 从 robot demos 扩展到 **all physical behavior in the world**。这是数量级的扩展。

### 5.3 四个 pillar 之间的 dependency graph

我重新整理它们的 dependency：

```
Physical Data Engine (1)
        │
        ├──→ Retargeting (2): 需要 u_ζ, s_ζ, φ_ζ
        │
        ├──→ World Model (3): 需要 s_ζ, c_ζ, u_ζ 作 supervision
        │
        └──→ Reward Model (4): 需要 s_ζ, φ_ζ, g

World Model (3)
        │
        └──→ 检测 Physical Data Engine 的 inconsistent labels
                ↓
        反过来监督 Engine

Retargeting (2)
        │
        └──→ 使用 World Model rollout 选择 best action

Deployment (4)
        │
        ├──→ 产生新 episode → Data Engine
        │
        └──→ credit assignment → 更新对应 component
```

所以这不是一个 feedforward stack，是一个 **closed loop with cross-component supervision**。

### 5.4 与 LeCun JEPA 的关系

LeCun 的 JEPA philosophy："predict in abstract representation space, not pixel space" 这篇 paper 完全 embrace 并 extend：
- representation space 应该包含 **task-relevant physical variables**（不只是 abstract latent）
- prediction 应该 **task-conditioned**（不只 generic future prediction）
- WM 应该和 grounding system 耦合，互为 supervision

V-JEPA 2 已经 show "internet video + small robot interaction → zero-shot robot control"，这是 thesis 的第一个 strong existence proof。

参考：V-JEPA 2 https://arxiv.org/abs/2506.09985

### 5.5 与我的"Software 2.0" / "Software 3.0" 思路的关系

我之前讲过 Software 1.0（人写 code）→ Software 2.0（neural net weights）→ Software 3.0（natural language as programming language）。这篇 paper 在某种意义上是 **Robotics 2.0 → Robotics 3.0** 的 manifest：

- Robotics 1.0：hand-crafted controller
- Robotics 2.0：trained policy（包括 VLA）
- Robotics 3.0：compounding system that turns the world itself into supervision

数据闭环让 robot 在部署中持续变强 —— 这是 self-improving agent 的具体化。

### 5.6 这篇 paper 没有充分讨论的几点

1. **Sample efficiency of grounding models themselves**：q_θ（Data Engine）、f_ψ（Retargeting）、p_ω（WM）、r_η（Reward Model）这些 model 都需要训练，他们的 supervision 从哪来？bootstrapping？这是一个 cold-start 问题。
2. **Compute cost**：joint inference over (z, A) 是非常 expensive 的，特别是 alignment 变量 A 是 latent。
3. **Failure mode cascading**：如果 Data Engine 错标 contact，会污染 WM、reward、retargeting。如何 detect + recover？
4. **Safety constraints**：deployment loop 中如何保证 robot 不会因为 mislabelled reward 反复 try dangerous action？

### 5.7 我觉得最 promising 的几个具体 research direction

基于 paper 的 framework，我会优先投资：

1. **3DGS + differentiable physics 的 hybrid WM**（如 ContactGaussian-WM 方向）—— 因为它把 geometry + contact + dynamics 在一个 representation 里 unify。
2. **V-JEPA 2 路线**：representation-space prediction + small robot fine-tuning —— 最有 scaling potential。
3. **Wearable sensing → autolabeling** pipeline —— 这是最 cheap 的 grounding signal 来源，比 teleop 便宜两个数量级。
4. **Component-level credit assignment in deployment** —— 这是一个 underexplored 但 critical 的问题。

---

## 六、相关 reference 链接汇总

**核心 references（论文中提及）**：
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- π₀ (Physical Intelligence): https://arxiv.org/abs/2410.24164
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- OpenVLA: https://arxiv.org/abs/2406.09246
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ALOHA / Mobile ALOHA: https://arxiv.org/abs/2401.02117
- R3M: https://arxiv.org/abs/2203.12601
- VIP: https://arxiv.org/abs/2210.00030
- LAPA: https://arxiv.org/abs/2410.11758
- UniVLA: https://arxiv.org/abs/2505.06111
- MimicGen: https://arxiv.org/abs/2310.17596
- RoboCasa: https://arxiv.org/abs/2406.02523
- RoboGen: https://arxiv.org/abs/2310.12284
- DreamerV3: https://arxiv.org/abs/2304.12273
- Genie: https://arxiv.org/abs/2402.15391
- RoboDreamer: https://arxiv.org/abs/2404.12377
- UniSim: https://arxiv.org/abs/2310.06114
- DayDreamer: https://arxiv.org/abs/2206.14176
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- LeCun JEPA paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Hamiltonian Neural Networks: https://arxiv.org/abs/1906.01563
- Lagrangian Neural Networks: https://arxiv.org/abs/2003.04630
- Interaction Networks: https://arxiv.org/abs/1612.00222
- Deep Lagrangian Networks: https://arxiv.org/abs/1907.04490
- Physically Embodied Gaussian Splatting: https://arxiv.org/abs/2311.12198
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- Figure Helix: https://www.figure.ai/news/helix
- RialTo: https://arxiv.org/abs/2403.03949
- RL-GSBridge: https://arxiv.org/abs/2409.20291
- Real-is-Sim: https://arxiv.org/abs/2504.03597
- RoboGSim: https://arxiv.org/abs/2411.11839
- SpatialVLA: https://arxiv.org/abs/2501.15830
- RDT-1B: https://arxiv.org/abs/2410.07864
- 3D-VLA: https://arxiv.org/abs/2503.09631
- CogACT: https://arxiv.org/abs/2411.19650
- RoboMamba: https://arxiv.org/abs/2409.06406
- FAST: https://arxiv.org/abs/2501.09747
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952
- RH20T: https://arxiv.org/abs/2307.00595

**背景 / 相关思考**：
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a6eb52a30490
- Karpathy Software 3.0 (recent talks on LLM as OS)
- LeCun A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 七、一句话总结

这篇 paper 把 robotics foundation model 的未来从 "scale up VLA" 重新 frame 为 **"build the grounding stack that converts the world's physical experience into robot-usable supervision"**。VLA 是 stack 中的一层，world model 也是一层，但真正 missing 的 pillar 是：physical data engine（autolabelling）、task-preserving retargeting、physics-grounded consequence prediction、和 task-conditioned self-improving deployment loop。这四者构成一个 closed-loop compounding system，让 robot 不再受限于 curated robot-native data，而能从整个世界的 physical behavior 中学习。

这是一个非常 ambitious 的 vision，paper 本身没有 propose 具体的训练 algorithm 或 benchmark，但作为一个 research agenda，它精确定位了 robotics 当前的 scaling bottleneck，并提供了一个 coherent 的 architectural framework 让 community 可以在此基础上分工推进。
