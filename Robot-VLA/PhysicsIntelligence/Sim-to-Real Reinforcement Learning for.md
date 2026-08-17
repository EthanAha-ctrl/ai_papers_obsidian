---
source_pdf: Sim-to-Real Reinforcement Learning for.pdf
paper_sha256: 4310f02064f966bd803001dfa70a0470335239a9ddabe9fa29ce1b55f01f1df9
processed_at: '2026-08-12T06:20:30-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话总结

一群人让一个 humanoid 机器人（有两条胳膊两只多指手）**纯靠在 simulation 里瞎试**，学会了几样挺难的活儿：抓东西举起来、两只手抬大箱子、两只手互相递东西。训完之后**直接搬到真实机器人上就能用**，不需要真机演示数据，面对没见过的东西也能抓。这是 sim-to-real RL 第一次在 humanoid + 多指手 + 双手 + vision-based 这个组合上跑通。

项目主页：https://toruowo.github.io/recipe

---

## 为什么这事难

我先把难点拆开讲，你才能理解他们为什么每一步都做了看似奇怪的选择。

### 难点 1：sim 跟 real 对不上

你建一个机器人模型扔进 Isaac Gym，物理参数（friction、joint damping、link inertia）填厂商给的 URDF，跑出来的动作跟真机对不齐。这事所有人都遇到过。OpenAI 当年解 Rubik's cube 是雇人手调参数调到崩溃。这次他们用的是 Fourier GR1，是个便宜 humanoid，motor 噪声比工业级 arm 大得多，手调更痛苦。

### 难点 2：两只手怎么配合，reward 怎么写

单手抓东西，reward 就是"指尖离物体近 + 物体到目标位置"。两只手递东西，你得说清楚：**先哪只手碰、再举到哪、另一只手什么时候接、最后放哪**。这四步如果只用一个稀疏 final reward，RL agent 一辈子也撞不上一次成功，永远学不到东西。

### 难点 3：探索爆炸

两只手 + 多指，action dimension 几十。Horizon 长（递东西要好几步）。Sparse reward。纯 RL 在这个空间里随机探索，命中成功轨迹的概率比单手任务低几个数量级。这是 RL 老大难的 exploration 问题在 robotics 上的具体放大版。

### 难点 4：vision-based sim-to-real 的感知 gap

如果你用 RGB 图像训练，sim 里渲染的塑料杯跟 real 里的塑料杯长得不像，policy 立刻挂。用纯 3D position 又信息不够（不知道物体形状怎么抓）。这是 representation 的两难。

---

## 他们怎么一个个解的

### 解决 1：把"手调参数"变成 4 分钟自动搜索

核心 idea 极其朴素：**同一串 joint target 在真机和 sim 里跑出来的轨迹应该一样**。

具体做法：
1. 在真机上跑一串预设的 joint target，记录每帧 actual joint position
2. 在 sim 里 random sample 一堆物理参数组合，跑同样的 target
3. 选 MSE 最小的那组参数

就这么简单。4 分钟，2000 步，搞定。MSE 越低，后面 sim-to-real 成功率越高（Table 1 很干净：MSE 最低 8/10，最高 0/10）。

**聪明在哪**: 这给了一个 sim-to-real readiness 的 proxy metric。你不用真部署才知道 sim 够不够准，autotune MSE 本身就是信号。

underactuated joint（tendon-driven 手指那种被动关节）也用线性近似 $q_u = k \cdot q_a + b$，$k, b$ 一起 search。

参考类似思路：
- ASID: https://arxiv.org/abs/2404.12308
- Reconciling Reality through Simulation: https://arxiv.org/abs/2403.03934

### 解决 2：用 "contact stickers" 定义接触目标

这是这篇 paper 我个人觉得最 elegant 的设计。

他们在 sim 里的物体表面贴一堆虚拟 "contact markers"（3D 坐标点）。Reward 写成：

$$r_{contact} = \sum_i \left[ \frac{1}{1 + \alpha \cdot d(\mathbf{X}^L, \mathbf{F}_i^L)} + \frac{1}{1 + \beta \cdot d(\mathbf{X}^R, \mathbf{F}_i^R)} \right]$$

翻译成人话：
- $\mathbf{X}^L, \mathbf{X}^R$：左右手上贴的 marker 位置
- $\mathbf{F}_i^L, \mathbf{F}_i^R$：第 $i$ 个指尖位置
- $d$ = min distance，每个指尖找最近的 marker
- $\frac{1}{1+\alpha d}$ 是个 soft indicator，指尖贴上 marker 时 reward 趋近 1，远了掉到 0

**为什么用 min 不用 average**：因为一个指尖只需要贴上一个 marker 就构成 grasp，不需要碰所有 marker。min 自动完成 "fingertip 到 marker 的 assignment"。

**最骚的地方**：marker 放哪决定 emergent grasp pattern。
- 放箱子左右中心 → policy 学会从侧面抓
- 放上下中心 → 学会从上下抓
- 放底部边缘 → 学会抠底

reward 不只是说"抓到就行"，而是用 marker 的几何位置**隐式编码 "怎么抓"**。比写一堆 if-else 的 grasp pose 灵活得多。

对于 bimanual handover，reward 加 stage variable $a \in \{0, 1\}$：

$$r = (1-a) \cdot (r_{contact}^{A} + r_{goal}^{A}) + a \cdot (r_{contact}^{B} + r_{goal}^{B})$$

每个 stage 有自己的 contact + object goal reward，stage 之间有 bonus 鼓励推进。

参考：NVIDIA Eureka（用 LLM 自动写 reward）https://eureka-research.github.io/，这篇走的是更结构化的 keypoint 路线。

### 解决 3：分而治之 + 蒸馏

这个 trick 解决 exploration。

**问题**：训一个 policy 同时学 10 个不同物体，sparse reward 下，10 个一起的 success 概率是每个相乘，极低。

**做法分两步**：

**第一步：训 specialist**。把任务拆开。10 个 object 拆成 3 组（mix：每组内部 shape 多样），每组训一个 PPO policy。每组 sample efficiency 高很多，因为 reward signal 密集。

**第二步：distill 成 generalist**。每个 specialist 在 sim 里跑 5000 步，**filter 出成功的 trajectory**，当 demonstration 存下来。然后用 Diffusion Policy 在这些 demonstration 上做 supervised learning 训一个 generalist。

**关键 insight**: specialist 扮演了 sim 里的 "teleoperator"。Divide-and-conquer 把 hard exploration 问题转化为 supervised learning 问题。RL 负责解决每个 sub-task 的探索，distillation 负责聚合。

Ablation 结果（Figure 4 右 + 4.5）：
- single（10 个 policy 一对一）：sim 训练最快，但 sim-to-real 只有 40%（overfit 单一 geometry）
- mix（3 个 policy 多样分组）：sim-to-real 90%（最佳）
- shape（3 个 policy 相似分组）：63.3%
- all（1 个 policy 学全部）：sim 训练最差，sim-to-real 23.3%

**最佳策略**：group 数量适中（保 sample efficiency）+ group 内部多样（保 generalization）。

还有一个加速探索的小 trick：用 < 30 秒 human "play"（不是 demo，就是随便玩玩）记录几个 task-relevant 的 hand-object 初始状态，RL episode 从这些状态开始随机采样。把探索 baseline 抬高，避免 agent 在 "手离物体八丈远" 的无用区域里转圈。这个跟 DemoStart 的 full demo 路线不同，更轻量。

参考：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DemoStart: https://arxiv.org/abs/2409.06613

### 解决 4：3D position + depth 的 hybrid 表示

vision-based sim-to-real 的 representation trade-off：

- **3D position**: 低维，sim-to-real gap 小，但物体形状信息丢失
- **depth image**: 高维，shape 信息丰富，但 sim-to-real visual gap 大
- **RGB**: 最高维，gap 最大

他们选了 **3D object position（来自 third-view camera）+ depth image（来自 egocentric camera，且先 segment 出物体区域）**。

**为什么这样组合**：
- 3D position 提供 global localization 锚点，sim-to-real 抗噪
- Depth 提供 local geometry detail，segment 后 gap 大幅减小（不用 render photorealistic RGB，只要 depth shape 对就行）

Real-world perception pipeline 用 SAM2：
1. 第一帧用 SAM2 分割物体（zero-shot，foundation model 没见过的物体也能分）
2. SAM2 的 tracking 把 mask 跟到后续所有帧
3. 从 mask center + depth camera 得 3D position
4. 5 Hz 跑，匹配 policy 控制频率

Ablation（Table 3）非常 striking：
- Depth + Position: 几乎 100% 成功
- Depth Only: 几乎 0% 成功

**纯 depth 完全失败，加个 3D position 立刻 work**。说明 3D position 是 stable anchor，弥补了 depth 在 real 世界里 viewpoint shift / noise 带来的不稳定。

参考：SAM2 https://ai.meta.com/sam2/

### Domain randomization（Table 4）

老一套但很全：物体 mass、friction、shape scale、initial pose；手 friction；PD gain（×0.8~1.1）；random force；observation noise；action noise；frame lag；action lag；depth camera pos/rot/FOV noise。

PD gain randomization 跟 autotune 互补 — autotune 让 sim 更准，PD randomization 让 policy 对剩余不准的部分 robust。

---

## 整体 pipeline

把它串起来：

1. **Autotune**: 真机跑 4 分钟 calibration → 自动调好 sim 参数
2. **Reward design**: 用 contact markers + object goals 写 reward
3. **Specialist training**: 拆任务，PPO 训每个 specialist
4. **Distillation**: Filter 成功 trajectory，Diffusion Policy 蒸馏成 generalist
5. **Sim-to-real**: Hybrid representation (3D pos + depth) + domain randomization
6. **Deploy**: SAM2 给 perception，policy zero-shot 部署

---

## 结果

- Grasp-and-reach: 62.3%（seen object 90%, novel 60-80%）
- Box lift: 80%
- Bimanual handover: 52.5%（最难，最 dynamic）

Force perturbation（knock / pull / push / drag）下 robust。

Cross-embodiment：从 Fourier hands 换 Inspire hands（完全不同的 morphology）也 work，autotune module 自动适配。

可以跟 FSM / teleoperation framework 串联做 longer-horizon 任务。

---

## 我读完的几个 intuition

### 1. Sim-to-real 不是一招的事

很多人把 sim-to-real 等同于 domain randomization。这篇 paper 强烈反驳了这点。它把 sim-to-real 拆成一个 pipeline：real-to-sim 建模、reward 设计、policy 训练、perception 表示，每一步都有 explicit 的 intervention。**盲目 randomize 是懒得思考**。

### 2. Contact marker 是一种 reward 语言

传统 reward 是连续函数（distance、velocity），contact marker 是 spatial keypoint 语言。它把 "where to grasp" 这种 implicit knowledge 显式化成 3D 点。这种 representation 可以扩展到 tool use（marker 在 tool handle）、articulated object（marker 在 joint）、deformable object（marker 在 mesh 顶点）。这是个新范式。

### 3. Distillation from RL 是 RL 的 "作弊"

纯 RL 的 exploration bottleneck 在高维 sparse reward 下没法解。把 hard exploration 拆成多个 easy exploration（specialist），再 distill 成 generalist。这等于让 RL agent 互相教。这其实是 imitation learning from RL agents —— specialist 是 teleoperator，generalist 是 student。一个不稳定（RL）的方法被一个稳定（supervised）的方法吸收。

这跟 AlphaGo 的 pipeline 精神类似：RL 训 expert，distillation 得 compact policy。

### 4. Hybrid representation 的哲学

不要追求 "end-to-end from pixels"，也不要退回 "pure state-based"。Mid-level representation（segmented depth + 3D position）在 sim-to-real 上是 sweet spot。Bitter lesson 说 scale + general method 胜过 hand-crafted feature，但 bitter lesson 没考虑 sim-to-real domain gap。**在 domain gap 是 bottleneck 的场景，mid-level representation 有独特价值**。

### 5. SAM2 当 perception backbone

Foundation vision model 已经悄悄成为 robotics perception 的 default building block。这篇 paper 用 SAM2 做 zero-shot segmentation + tracking，省去了训一个 robust segmentation model 的麻烦。未来更多 VLM / foundation model 会进 robotics perception front-end，让 policy 专注 control。

### 6. Bimanual handover 52.5% 暴露的问题

最 dynamic 的任务成功率最低。作者归因于 dynamics gap。这暗示：**对 contact-rich + dynamic + bimanual，纯 domain randomization 可能不够**。未来可能需要：
- Online system identification for object dynamics
- Real-world fine-tuning（RHT / RLHF for robots）
- Tactile sensing（这篇没用 tactile，但作者 prior work Lin et al. visuotactile 探索过 https://arxiv.org/abs/2404.16823）

### 7. "No demonstration" 的 nuance

Paper 主打 "无需大量 human demo"，但实际用了：
- < 30 秒 human play for init
- SAM2 第一帧 mask（人 click 一下）

不算完全 zero human，但远少于 ALOHA 几百条 demo 的规模。这是 honest framing。

### 8. 跟 GR00T N1 的关系

NVIDIA 内部两条 humanoid 路线并行：
- GR00T N1（foundation model, VLA + imitation）
- 这篇（sim-to-real RL）

未来很可能 hybrid：foundation model 高层规划，sim-to-real RL 提供 low-level dexterity。

GR00T N1: https://arxiv.org/abs/2503.14734

---

## 这篇 paper 的真正贡献

不是算法 novelty。每个 module（autotune, contact marker, divide-and-conquer distillation, hybrid representation）单独看都不复杂。

**贡献是 engineering recipe 级别**：第一次把 vision-based + bimanual + multi-fingered + humanoid + sim-to-real RL 这个组合跑通，并给出每个 component 的具体 trick 和 ablation 证据。

这是 sim-to-real RL 在 humanoid dexterous manipulation 上的 milestone paper。未来 1-2 年会看到大量 follow-up 在这条 pipeline 上做改进：tactile 加入、object foundation model 替代 primitive geometry、real-world online adaptation、torque-level control、更多 DoF 的 hand hardware。

---

## 对你做 research 的启示

如果你在做 robotics RL，这篇 paper 的几个 trick 直接可借鉴：
- **Autotune 思路**: 任何 sim-to-real 项目都能用，把 manual tuning 换成 black-box search on tracking error
- **Contact marker reward**: 任何 contact-rich task 都能用，比手写 grasp pose 灵活
- **Divide-and-conquer distillation**: 任何 multi-task RL 探索困难时都能用，specialist + Diffusion Policy 是 stable combo
- **Hybrid representation**: 任何 vision-based sim-to-real 都该考虑，不要无脑 end-to-end

更深层的启示：**sim-to-real 是系统工程问题，每个 component 都要 polish，不要寄望一招通杀**。这是 RL research community 应该内化的态度。

---

# Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids — Technical Deep Dive

## 0. Paper 的核心命题

这篇论文要解决的问题是：**如何用纯 RL（无 human demonstration）训练一个 humanoid robot 配 multi-fingered hands 做 vision-based、contact-rich、bimanual dexterous manipulation，并 zero-shot transfer 到 real world**。

作者识别出四个被 prior work 没有充分解决的 bottleneck：
- (A) Low-cost hardware 的 sim-to-real gap（motor noise 大）
- (B) Bimanual coordination 的 reward design
- (C) Long-horizon high-dim exploration
- (D) Object perception 在 sim-to-real 下的 domain shift

整个 recipe 的 flow 如下：**Real-to-Sim Autotune → Reward Design (contact + object goals) → Task-aware init + Divide-and-conquer distillation → Hybrid representation + Domain randomization → Zero-shot sim-to-real**。

项目主页：https://toruowo.github.io/recipe

---

## 1. Real-to-Sim Autotune Module — 把 manual tuning 变成 4 分钟 black-box search

### 1.1 动机

Manufacturer-supplied URDF + 默认 simulator physics 通常不能直接用于 sim-to-real。Prior work（如 OpenAI Rubik's cube [9]、Dextreme [10]）都需要大量 manual tuning。对于低成本 humanoid（比如 Fourier GR1 + 轻量级 hand），motor noise、joint backlash、underactuation 都更严重，manual tuning 更痛苦。

### 1.2 Algorithm 1 详解

```
Require: E (env params), N (calibration action sequences), R (real robot), M (initial URDF)
1: P ← InitializeParameterSpace(E, M)
2: S ← {}
3: for i ← 1 to K do  (K 是 population size)
4:     p_i ← RandomSample(P)
5:     S_i ← CreateSimEnvironment(p_i)
6:     S ← S ∪ {S_i}
7: end for
8: J ← GenerateJointTargets(N)
9: R_track ← GetTrackingErrors(R, J)  // real robot 执行 J，记录 tracking error
10: best params ← null; min error ← ∞
11: for S_i ∈ S do
12:     S_track ← GetTrackingErrors(S_i, J)  // 在 sim 中跑同样的 J
13:     error ← ComputeMSE(S_track, R_track)
14:     if error < min error: update best
15: end for
16: return best params
```

**Intuition**: 这是一个 model-free 的 system identification。我们不知道真实的 friction coefficient 是多少，但我们知道 **同一个 joint target sequence 在 real 和 sim 上跑出来的轨迹应该一致**。所以把 tracking error 当作 loss，用 random search 在 parameter space 里找最小化 MSE 的那组参数。

时间复杂度：4 分钟 = 2000 simulated steps at 10 Hz。这是一个非常聪明的 budget — 对于 10 Hz 控制频率，2000 步约 200 秒的 trajectory，足以覆盖 joint dynamics 的主要 mode。

### 1.3 Underactuated joint 的建模

Fourier hand 有 6 actuated DoF + 5 underactuated DoF。Isaac Gym 不直接支持 underactuation，所以用线性近似：

$$q_u = k \cdot q_a + b$$

- $q_u$: underactuated joint angle
- $q_a$: actuated joint angle  
- $k, b$: 线性拟合系数

这里 $k, b$ 也作为 autotune 的 search parameter。这种 linear coupling 是 tendon-driven hand 的常见近似 — 真实硬件里 underactuated joint 通过 tendon 被 actuated joint 拖动，近似线性关系在 small range 内合理。

### 1.4 Table 1 的关键观察

| Autotune MSE | Lowest | Median | Highest |
|---|---|---|---|
| Grasp Success | 8/10 | 3/10 | 0/10 |
| Reach Success | 7/10 | 3/10 | 0/10 |

这个 correlation 极其干净：**MSE 越低，sim-to-real success 越高**。这说明 autotune 找到的参数确实在减小 reality gap，可以作为 sim-to-real readiness 的 metric。这给了一个 practical 的 stopping criterion — 不用等到部署才知道 sim 是否够准，autotune MSE 本身就是 proxy。

Reference: 类似思路见 ASID [38] (Active exploration for system identification), Reconciling Reality through Simulation [37]。

---

## 2. Generalizable Reward Design — Contact + Object Decomposition

### 2.1 核心 insight

Human manipulation task 可以分解为 **hand-object contact transition** + **object state change** 的序列。比如 bimanual handover:
1. Hand A contacts object
2. Object lifted near Hand B  
3. Hand B contacts object
4. Object transferred to target

### 2.2 Contact goal reward 公式

$$r_{contact} = \sum_i \left[ \frac{1}{1 + \alpha \cdot d(\mathbf{X}^L, \mathbf{F}_i^L)} + \frac{1}{1 + \beta \cdot d(\mathbf{X}^R, \mathbf{F}_i^R)} \right]$$

变量解释：
- $\mathbf{X}^L \in \mathbb{R}^{n \times 3}$: left hand 的 $n$ 个 contact marker 的 3D 位置（"contact stickers"）
- $\mathbf{X}^R \in \mathbb{R}^{m \times 3}$: right hand 的 $m$ 个 contact marker 位置
- $\mathbf{F}^L \in \mathbb{R}^{4 \times 3}$: left hand 4 个 fingertip 的位置
- $\mathbf{F}^R \in \mathbb{R}^{4 \times 3}$: right hand 4 个 fingertip 的位置
- $\mathbf{F}_i^L, \mathbf{F}_i^R$: 第 $i$ 个 fingertip
- $\alpha, \beta$: scaling hyperparameters，控制 reward 的 sharpness（越大越 sharp）
- $d(\mathbf{A}, \mathbf{x}) = \min_i \|\mathbf{A}_i - \mathbf{x}\|_2$: 取 $\mathbf{A}$ 中距离 $\mathbf{x}$ 最近的那个 marker 的距离

**Intuition**: $\frac{1}{1+\alpha d}$ 是 Lorentzian function，类似一个 soft indicator — 当 fingertip 离最近 marker 很近时 reward 接近 1，远离时 reward 衰减到 0。比 Gaussian 或 exp 更 robust，不会有 gradient explosion。

为什么要 $\min$ 而非 average？因为一个 fingertip 只需要 match 一个 marker 就构成 grasp，不需要 match 所有的。$\min$ 让每个 fingertip "选择" 最近的 marker，自动实现 assignment。这有点像 Chamfer distance 的 asymmetric 版本。

### 2.3 Object goal reward

对于 grasp-and-reach 和 box lift：
$$r(s_h, s_o) = r_{contact}(s_h, s_o) + r_{goal}(s_o)$$

- $s_h$: hand state，包括 fingertip positions
- $s_o$: object state，包括 center-of-mass position + contact marker positions
- $r_{goal}$: penalize 当前 object state 与 target object state (e.g. xyz position) 的偏差

### 2.4 Handover 的 staged reward

$$r = (1-a) \cdot (r_{contact}(s_{h_A}, s_{o_A}) + r_{goal}(s_{o_A})) + a \cdot (r_{contact}(s_{h_B}, s_{o_B}) + r_{goal}(s_{o_B}))$$

- $a \in \{0, 1\}$: stage variable，0 表示 stage A（hand A 主动），1 表示 stage B（hand B 主动）
- $s_{h_A}, s_{h_B}$: 各阶段 engaged hand 的 fingertip positions
- $s_{o_A}, s_{o_B}$: 各阶段 object state
- 每完成一个 stage 有 bonus，bonus scale 随 stage index 递增（鼓励往后期走）

**Intuition**: 这是 reward shaping 的经典思路 — 用 stage variable 把长 horizon task 切成几段，每段有独立的 dense reward。stage 切换可以基于 object position 阈值或 contact 检测自动判断。

### 2.5 Contact markers 的 procedural generation

Contact markers 可以基于 object geometry 程序化生成。Figure 5 展示了 box lift 任务里不同 marker placement 导致不同 emergent behavior:
- Markers 在左右中心 → 侧面 grasp
- Markers 在上下中心 → 顶部底部 grasp
- Markers 在底部边缘 → 底部 edge grasp

这是一个非常 elegant 的设计 — **reward function 既定义了 "what to achieve"，也通过 marker placement 隐式定义了 "how to achieve"**。这跟 NVIDIA Eureka（LLM 自动生成 reward code）思路不同，这里用更 structured 的 keypoint 语言来 encode human prior。

Reference: 
- Eureka reward design: https://eureka-research.github.io/
- DexCap (human hand priors): https://dex-cap.github.io/

---

## 3. Sample Efficient Policy Learning — Exploration 的两个 trick

### 3.1 Task-aware hand pose initialization

从 human teleoperation in sim 收集 task-relevant hand-object configurations（<30 秒），作为 episode 的初始条件随机采样。

**关键区别**: prior work（DemoStart [55]）用 full demonstration trajectory，这里只需 human "casually play around"。Recorded states 包括 object poses + robot joint positions。

**Intuition**: 纯 RL 的 exploration bottleneck 在于初期 agent 几乎不可能随机探索到 "object 在手里" 这种 rare state。如果 episode 总是从零开始，sparse reward 永远触发不了。把 initial state distribution 偏向 "task-relevant region"，相当于把探索的 baseline 抬高，让 agent 从 "已经接近成功" 的状态开始 learn。这跟 hindsight experience replay (HER) 的精神类似，但用 human prior 而非 random replay。

### 3.2 Divide-and-conquer distillation

核心 insight：**不要试图让一个 policy 同时学多个 object，而是先训练 specialist，再 distill 成 generalist**。

#### Specialist 训练

- 每个 sub-task（比如 single object 或 shape-similar object group）独立训练一个 PPO policy
- Observation: object position + robot joint position
- Action: robot joint angles
- Asymmetric actor-critic: critic 有 privileged info（arm/hand joint velocities, fingertip positions, object orientation/velocity/angular velocity, mass/friction/shape randomization scale）
- Actor & critic: 3-layer MLP, (512, 512, 512)

#### Distillation 流程

1. 每个 specialist policy 在 100 个 parallel envs 跑 5000 步
2. Filter 成功的 trajectory，存到 disk
3. 把这些 trajectory 当作 "demonstrations"
4. 用 Diffusion Policy [42] 训练 generalist

**Intuition**: 这里 specialist 扮演 "simulation 里的 teleoperator" 角色。Divide-and-conquer 把 hard exploration 问题转化为 supervised learning 问题。RL 负责解决每个 sub-task 的 exploration，distillation 负责 aggregate。

### 3.3 Figure 4 右图的 ablation

四种 decompose 策略在 10 个 object 的 multi-object task 上比较：
- **all**: 一个 policy 学所有 10 个 object — 最差
- **shape**: 3 个 policy，按 shape similarity 分组
- **mix**: 3 个 policy，按 shape diversity 分组（强迫每个 group 内部多样）
- **single**: 10 个 policy，每个 object 一个 — sample efficiency 最高

Sim-to-real 成功率：
- mix: 90.0% ← 最好
- shape: 63.3%
- single: 40.0%（overfit 具体 geometry）
- all: 23.3%

**解读**: single 在 sim 里训练快但 sim-to-real 差，因为 overfit 单一 geometry，domain randomization 不够。all 在 sim 里训练慢且差，exploration 太难。**mix 是最佳折中 — group 内部多样保证 generalization，group 数量适中保证 sample efficiency**。这本质上是一个 curriculum / diversity vs specificity 的 trade-off。

---

## 4. Vision-Based Sim-to-Real — Hybrid Representation

### 4.1 Representation spectrum 的问题

Prior work 在 object representation 上有一个 spectrum:
- 3D position [14]: 低维，sim-to-real gap 小，但信息少
- 6D pose [9]: 中等
- Depth image [17, 12]: 高维，信息多，但 sim-to-real gap 大
- Point cloud [57]: 高维
- RGB [10]: 最高维，gap 最大

**Trade-off**: 高维 representation 编码更多 object 信息提升 task performance，但同时放大 sim-to-real gap。低维反之。

### 4.2 Hybrid 方案

作者的方案：**3D object position (from third-view) + Depth image (from egocentric view, segmented)**。

- Sparse 3D position: 通过 SAM2 [59] segmentation + depth recovery 得到 object center-of-mass 的 noisy 3D position
- Dense depth: egocentric camera 的 depth，先 segment 出 object 区域再使用

**Intuition**: 
- 3D position 提供 **global localization**，sim-to-real gap 小（一个 3D vector），noise 可控
- Depth 提供 **local geometry detail**，segment 后 gap 大幅减小（不需要 render photorealistic RGB，只需 depth shape 对）

两者互补：sparse 给 coarse global，dense 给 fine local。

### 4.3 Perception pipeline (real world)

1. SAM2 在 trajectory 初始帧生成 object segmentation mask
2. SAM2 的 tracking capability 跟踪 mask 到所有后续帧
3. 从 mask 在 image plane 的 center + depth camera 的 noisy reading 恢复 3D position
4. Pipeline 运行 5 Hz（匹配 policy control frequency）

**SAM2 的角色**: 省去了训练一个 robust segmentation model 的需求。SAM2 是 foundation model，零样本分割能力很强，特别适合 sim-to-real 场景下 object 外观未知的情况。

Reference:
- SAM2: https://ai.meta.com/sam2/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 4.4 Table 3 的关键 ablation

| Task | Grasping Pickup | Grasping Success | Lifting Pickup | Lifting Success | HandoverA Pickup | HandoverA Success | HandoverB Pickup | HandoverB Success |
|---|---|---|---|---|---|---|---|---|
| Depth + Pos | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 | 5/10 |
| Depth Only | 2/10 | 2/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

**Depth only 几乎完全失败**，Depth + Pos 几乎全过。差距在 task horizon 长的任务（handover B 阶段）更大。

**解读**: Depth only 的 policy 在 sim 里可能能 work，但 transfer 到 real 后 depth noise + viewpoint shift 导致 object localization 不稳。3D position 提供了一个 stable anchor。这印证了 hybrid design 的核心 hypothesis。

### 4.5 Domain randomization (Table 4)

物理 randomization:
- Object mass [0.03, 0.1] kg
- Object friction [0.5, 1.5]
- Object shape scale × U(0.95, 1.05)
- Object initial position + U(-0.02, 0.02) m
- Object initial z-orientation + U(-0.75, 0.75) rad
- Hand friction [0.5, 1.5]
- PD controller: P gain × U(0.8, 1.1), D gain × U(0.7, 1.2)
- Random force: scale 2.0, probability 0.2, decay 0.99 every 0.1s

非物理 randomization:
- Object position observation noise: 0.02
- Joint observation noise + N(0, 0.4)
- Action noise + N(0, 0.1)
- Frame lag probability 0.1
- Action lag probability 0.1
- Depth camera pos noise 0.005 m
- Depth camera rot noise 5.0°
- Depth camera FOV noise 5.0°

**Intuition**: 注意 PD gain randomization — 这相当于让 policy 适应 motor 的不确定性，跟 autotune module 互补。Frame lag / action lag 模拟通信延迟，是 sim-to-real 的常见 hidden gap。

---

## 5. Policy Architecture & Training Details

### 5.1 Specialist (RL) 阶段

- **Algorithm**: PPO [60] with asymmetric actor-critic
- **Actor**: 3-layer MLP (512, 512, 512), input = object position + robot joint position
- **Critic**: 同样 3-layer MLP，但 input 加 privileged info
- **Action**: robot joint angles (absolute desired position)

Asymmetric actor-critic 的核心：critic 训练时可以用 ground truth privileged info（object velocity, mass, friction 等），actor 部署时没有这些 info。critic 用 privileged info 降低 value estimation variance，让 actor 的 gradient 更 clean。这是 OpenAI Rubik's cube [9] 就用的 trick。

### 5.2 Generalist (Distillation) 阶段

- **Policy class**: Diffusion Policy [42]
- **Proprioception + 3D position encoder**: 3-layer MLP, ELU activation, (512, 512, 512), output 64-dim feature
- **Depth encoder**: ResNet-18 [61]，所有 BatchNorm [62] 替换为 GroupNorm [63]
- **Diffusion model**: 100 steps, square cosine noise schedule
- **Output**: 7-DoF absolute joint position per arm + 6-DoF normalized (0-1) joint position per hand
- **Optimizer**: AdamW [65], lr=1e-4, weight decay=1e-5, batch size=128
- **EMA**: 维护 model weights 的 exponential moving average，用于 eval/deploy

**为什么 ResNet-18 用 GroupNorm 而非 BatchNorm**: BatchNorm 在 batch size 小或非 i.i.d.（RL 数据有时序相关）时统计量不稳。GroupNorm 不依赖 batch，更稳定。这是 Diffusion Policy 原论文 [42] 的实践经验。

**为什么 output 是 absolute 而非 delta**: absolute desired joint position 对每个 timestep 独立预测，policy 更易 interpret，也更易与 scripted high-level controller (FSM) 串联。delta 方式需要积分，error 累积。

Reference: PPO https://arxiv.org/abs/1707.06347, Diffusion Policy https://diffusion-policy.cs.columbia.edu/

---

## 6. Experimental Results Summary

### 6.1 总体 success rate (Section 4.7)

- Grasp-and-reach: 62.3%（seen object 90%, novel object 60-80%）
- Box lift: 80%
- Bimanual handover: 52.5%（最难，因为 most dynamic + 最长 horizon）

### 6.2 Robustness (Figure 6)

Policy 在 knock / pull / push / drag 四种 force perturbation 下保持 robust。这是 domain randomization 中 random force 训练的直接成果。

### 6.3 Cross-embodiment (Inspire hands)

Inspire hands 跟 Fourier hands 有显著差异：mass、surface friction、finger/palm morphology、thumb actuation 都不同。Autotune module 自动适配这些 hardware，验证了 cross-embodiment generalizability。

### 6.4 System extension

Learned RL policy 可以与 high-level FSM / teleoperation framework 串联，做 long-horizon task（比如 pick-and-drop）。这是 modular design 的好处。

---

## 7. Limitations & Open Problems (Section 6)

作者诚实地指出：

1. **Reward design 还可以更强**: 可以 integrate human teleoperation demonstration 作为更强的 prior
2. **Controller 只用 position**: torque controller 可以探索
3. **Sim-to-real dynamics gap**: 只用 naive domain randomization，没做 advanced dynamics adaptation。这可能是 bimanual handover 成功率低的原因（最 dynamic 的任务对 dynamics mismatch 最敏感）
4. **Hardware dexterity 限制**: 当前 multi-fingered hand 的 active DoF 远少于人手。Policy dexterity 可能被 hardware 限制，而非 approach 限制

---

## 8. 与 Related Work 的定位

### 8.1 vs OpenAI Rubik's Cube [9]

- OpenAI: single hand, state-based (block pose 6D), domain randomization 重度
- This work: bimanual, vision-based (depth + 3D pos), humanoid platform, automated system ID

### 8.2 vs Dextreme [10]

- Dextreme: single hand, RGB input, heavy domain randomization on visual
- This work: bimanual, depth + sparse 3D, structured reward with contact markers

### 8.3 vs Chen et al. [33] (Object-centric dexterous manipulation from human motion data)

- Chen: 用 human hand motion capture 作为 prior
- This work: 从 scratch 学 full hand-arm joint control，不需要 motion capture

### 8.4 vs Lin et al. [14] (Twisting lids off with two hands)

- Lin et al.: state-based bimanual
- This work: vision-based bimanual，更 general

### 8.5 vs ALOHA / Mobile ALOHA / Diffusion Policy [42, 44]

- ALOHA: imitation learning, 需要大量 demo
- This work: 纯 RL，无需 real-world demo（除了 < 30s human play for init）

### 8.6 vs GR00T N1 [1]

- GR00T N1: foundation model for humanoid，imitation-based
- This work: focused recipe for dexterous manipulation via sim-to-real RL

---

## 9. 我的 Intuition Building

这篇 paper 给我的几个 key takeaways:

### 9.1 Sim-to-real 不只是 domain randomization

很多人把 sim-to-real 等同于 domain randomization，但这篇展示了 sim-to-real 是一个 pipeline，每一步都要 explicit 处理：
- **Real-to-sim**: autotune 让 sim 本身更准（vs 盲目 randomize）
- **Reward**: structured reward 让 sim 里的 learning 更 sample efficient
- **Policy learning**: divide-and-conquer 让 sim 里的 exploration 更可行
- **Perception**: hybrid representation 让 sim-to-real 的 visual gap 可控

### 9.2 Divide-and-conquer distillation 是 RL 的 "trick"

把 hard exploration 拆成多个 easy exploration，再 distill 成 generalist。这其实是一种 **imitation learning from RL agents** 的思路。Specialist 是 "teleoperator in simulation"，generalist 是 "student"。这让 RL 的 exploration bottleneck 被 supervised learning 的 stability 补偿。

这个思路其实跟 AlphaGo 的 pipeline 有点像 — 先用 RL 训 expert，再用 distillation 得到更 compact/general 的 policy。

### 9.3 Contact marker 是 reward design 的新语言

传统 reward design 用 distance / velocity / orientation，这里引入 **contact marker** 作为 spatial keypoint。这把 "where to grasp" 这个 implicit knowledge 变成 explicit 3D points，reward function 直接 reference 这些 points。

这种 structured representation 可移植性很强 — 比如可以扩展到 tool use（marker 在 tool handle 上），articulated object（marker 在 joint 处），甚至 deformable object（marker 在 surface mesh 顶点）。

### 9.4 Hybrid representation 的哲学

不要追求 "end-to-end from pixels"，也不要退回 "pure state-based"。Mid-level representation (segmented depth + 3D position) 在 dexterous manipulation 上是 sweet spot。

这让我想到 Levity 的 "the bitter lesson" — 但 bitter lesson 说的是 scale + general method 胜过 hand-crafted feature。这里的情况不同：sim-to-real 的 bottleneck 是 domain gap，不是 representation power。所以 mid-level representation 在 sim-to-real 里有独特价值。

### 9.5 SAM2 作为 perception backbone

这篇 paper 把 SAM2 用作 zero-shot segmentation + tracking。这是 foundation vision model 进入 robotics perception pipeline 的典型例子。未来更多 vision-language foundation model 会作为 robot 的 perception front-end，让 policy 专注于 control。

Reference: 
- SAM2 in robotics: https://ai.meta.com/sam2/
- Foundation models for robotics: https://octo-models.github.io/, https://octo-models.github.io/

### 9.6 一个 practical concern

Bimanual handover 只有 52.5% 成功率，作者归因于 dynamics gap。这暗示对于 contact-rich + dynamic + bimanual 的任务，纯 domain randomization 可能不够。Future work 可能需要：
- System identification for object dynamics
- Real-world fine-tuning (RHT, RLHF for robots)
- Tactile sensing integration (这篇没用 tactile，但 Lin et al. [3] 的 prior work 用了 visuotactile)

### 9.7 与 GR00T N1 / humanoid foundation model 的关系

GR00T N1 [1] 是 NVIDIA 自家的 humanoid foundation model，走 imitation + VLA 路线。这篇 paper 走 sim-to-real RL 路线。两条路线在 NVIDIA 内部并行，可能未来 hybrid: foundation model 提供高层规划，sim-to-real RL 提供 low-level dexterity。

### 9.8 关于 "No demonstrations" 的 nuance

Paper claim 是 "without extensive human demos"，但实际上用了：
- < 30 秒 human play data for task-aware init
- SAM2 的初始 frame mask（需要人 click 一下 object）

这不算 "完全 zero human"，但确实远少于 ALOHA 那种 hundreds of demos 的规模。这是一个重要的 honest framing。

---

## 10. 公式与变量的 Quick Reference

| Symbol | Meaning |
|---|---|
| $\mathbf{X}^L \in \mathbb{R}^{n \times 3}$ | Left hand 的 $n$ 个 contact marker 3D 位置 |
| $\mathbf{X}^R \in \mathbb{R}^{m \times 3}$ | Right hand 的 $m$ 个 contact marker 3D 位置 |
| $\mathbf{F}^L \in \mathbb{R}^{4 \times 3}$ | Left hand 4 个 fingertip 位置 |
| $\mathbf{F}^R \in \mathbb{R}^{4 \times 3}$ | Right hand 4 个 fingertip 位置 |
| $\alpha, \beta$ | Reward sharpness scaling |
| $d(\mathbf{A}, \mathbf{x})$ | $\min_i \|\mathbf{A}_i - \mathbf{x}\|_2$，最近 marker 距离 |
| $a \in \{0, 1\}$ | Handover stage variable |
| $s_h$ | Hand state (fingertip positions) |
| $s_o$ | Object state (COM + markers) |
| $q_u, q_a$ | Underactuated / actuated joint angle |
| $k, b$ | Linear coupling coefficients for underactuation |

---

## 11. 相关 reference 汇总

- **Project page**: https://toruowo.github.io/recipe
- **Isaac Gym**: https://arxiv.org/abs/2108.10470
- **PPO**: https://arxiv.org/abs/1707.06347
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **SAM2**: https://ai.meta.com/sam2/ , https://arxiv.org/abs/2408.00714
- **ResNet**: https://arxiv.org/abs/1512.03385
- **GroupNorm**: https://arxiv.org/abs/1803.08494
- **OpenAI Rubik's Cube**: https://arxiv.org/abs/1910.07113
- **Dextreme**: https://arxiv.org/abs/2304.13653
- **Eureka (NVIDIA)**: https://eureka-research.github.io/
- **ASID**: https://arxiv.org/abs/2404.12308
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **Lin et al. Twisting Lids**: https://arxiv.org/abs/2403.02338
- **Lin et al. Visuotactile**: https://arxiv.org/abs/2404.16823
- **ALOHA Unleashed**: https://arxiv.org/abs/2410.13126
- **DemoStart**: https://arxiv.org/abs/2409.06613
- **Reconciling Reality through Simulation**: https://arxiv.org/abs/2403.03934
- **DexCap**: https://dex-cap.github.io/
- **AdamW**: https://arxiv.org/abs/1711.05101
- **BatchNorm**: https://arxiv.org/abs/1502.03167
- **Champion Drone Racing (RL)**: https://www.nature.com/articles/s41586-023-06462-7
- **Agile Locomotion (Science Robotics)**: https://www.science.org/doi/10.1126/scirobotics.adi8022

---

## 12. Final Thoughts

这篇 paper 的贡献是 **engineering recipe level** 而非 fundamental algorithmic novelty。每个 module（autotune, contact marker reward, divide-and-conquer distillation, hybrid representation）单独看都不复杂，但组合起来解决了 vision-based bimanual dexterous manipulation on humanoid 这个 prior work 没碰透的 setting。

对 research community 的启示：
1. Sim-to-real 是一个系统工程问题，每个 component 都要 polish
2. Structured inductive bias（contact markers, hybrid representation）在 sim-to-real 里依然 valuable，bitter lesson 还没完全奏效
3. Distillation from RL specialists 是连接 RL exploration 和 supervised learning stability 的实用桥梁
4. Foundation vision models (SAM2) 已经成为 robotics perception 的 default building block

对未来的展望：
- Tactile sensing 加入后会怎样？(Lin et al. [3] 已经探索)
- Object foundation model（3D generative model）替代 primitive geometry?
- Real-world online adaptation（meta-learning, RHT）补 domain randomization 的不足
- Torque-level control 替代 position-level，开启动态更丰富的 task
- Hardware evolution: 更多 active DoF 的 hand（比如 Shadow Hand 级别的 humanoid-sized hand）

这是 sim-to-real RL 在 humanoid dexterous manipulation 上一个 milestone 工作，未来 1-2 年会看到大量 follow-up 在这条 pipeline 上做改进。
