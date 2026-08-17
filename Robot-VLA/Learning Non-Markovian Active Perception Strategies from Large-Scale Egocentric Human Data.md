---
source_pdf: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric
  Human Data.pdf
paper_sha256: a63a4ecf3a6193318f58405cb136a90bfd341a424984ccd9f567cf735ca26942
processed_at: '2026-08-05T13:27:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好,我换个说法,像跟同事 coffee chat 那样聊。

---

## 这篇 paper 到底在解决什么问题

先说一个 observation:现在的 robot policy,不管 OpenVLA、$\pi_0$ 还是 Diffusion Policy,本质上都是**给定一张图,输出一个 action**。这个 framing 有一个隐含假设——observation 是 world 给你的,agent 只是被动接收。

但现实里 robot 干活,observation 是 agent **自己构造出来的**。你让 robot 去 toolbox 里找一把 wrench,wrench 被 stuff 盖住了,robot 必须先伸手挪开别的东西,才能看见 wrench。这个"挪开"的 action 不是为了抓 wrench,而是**为了让自己能看见 wrench**。Action 在这里有两个 role:一个是 manipulation,一个是 **epistemic**(获取信息)。

现有 policy 完全 handle 不了这个——它们只学了"看到 wrench → 抓 wrench"这个 mapping,但如果 wrench 不在视野里,policy 就傻了,要么 stuck,要么 random walk。

paper 的核心 claim:**active perception 应该被 formalize 成一个 non-Markovian process,因为 agent 需要记住自己看过哪里、需要根据 observation 结果 branch 出不同 action**。

---

## 他们怎么 formalize 的

用两个 concept 解释 active perception:

### Concept 1: Information Gain

action 应该 reduce uncertainty。比如你转头看左边,如果你已经知道左边没东西,这个 action 的 information gain 就是 0;如果左边是个 unexplored region,information gain 就高。paper 用 mutual information $I(o_{t+1}; \mathbf{A}_t \mid h_t, o_t)$ 来 formalize 这个——"采取 action $\mathbf{A}_t$ 之后,我对 world 的了解增加多少"。

但他们没有真去算这个 mutual information(高维 RGB 上算不了),而是训了一个 **cognitive auxiliary head** 当 proxy。这个 head 输出一个 binary label:"当前 sub-task 的信息 uncertainty 是否已经被 resolve"。比如 target 从隐藏变成可见,label 翻 1。这个 label 用来 trigger sub-task 切换。

### Concept 2: Decision Branching

不同的 observation 结果应该触发不同 action。比如 target 在左 → 左手抓;target 在右 → 右手抓。这看起来很 obvious,但现有 policy 学不到,因为它们是 Markovian 的——只看当前 frame,无法根据 history branch。

paper 把 branching 分两类:
- **Exploratory branching**:没找到 target 继续扫,找到了就 gaze
- **Exploitative branching**:根据 target 位置切换 manipulation strategy

这两类合起来,就是 "Act, Sense, Act" 这个闭环。

---

## 他们怎么分类 active perception 任务

三种 paradigm,很 clean:

1. **Viewpoint Discovery (VD)**:target 不在视野里,需要移动视角找。例:转头找桌上的 croissant,转 chassis 找房间里的垃圾桶。

2. **Manipulation Discovery (MD)**:target 被遮挡,需要物理交互揭示。例:开 cabinet 才能看见 bottle,掀开碗才能看见 cylinder。

3. **Information Enrichment (IE)**:target 看得见但不够清晰,需要靠近。例:peg 太远看不清精确位置,先抓近再插入。

这个 taxonomy 很 useful,因为它直接对应不同能力——VD 需要 chassis/head 控制,MD 需要 manipulation-as-sensing,IE 需要 zoom-in。

---

## 数据策略:用 human data 当 prior

这是 paper 的另一个 big bet。

问题:robot data 太贵,尤其 active perception 的 teleop 需要 VR + "blind protocol"(operator 不知道 target 在哪),效率很低。

他们的 solution:**用大量 human egocentric video 当 prior**。选了 CaptainCook4D(厨房场景,HoloLens 采集,1.07M samples)和 Ego-Exo4D(多场景,Aria glasses 采集,421k samples),共 800k human samples。

但 human 和 robot 的 morphology 不同,怎么对齐?他们设计了一个 **unified egocentric action space**:每条 trajectory 的第一帧 head pose 当 base frame,所有后续 pose 都 relative to 这个 frame。hand pose 用 MANO 补全,gripper 用 thumb-to-fingertip 平均距离压缩成 scalar。这样 human 和 robot 的 trajectory 在同一个 body-centric frame 里表达,embodiment gap 被大幅 reduce。

直觉上:human 在厨房找东西时,头怎么转、手怎么探、眼睛怎么 gaze,这些 **eye-hand coordination prior** 是 embodiment-agnostic 的。paper 的实验验证了这个 intuition——800k human data + 100k robot data 就能达到 83.3% SR,而 0 human + 400k robot data 只有 42.7% SR。Human data 的 scaling effect 非常明显。

---

## 模型架构:CoMe-VLA

backbone 是 Qwen3-VL-2B,加了几个 module:

### 1. Cognitive Auxiliary Head

输入一个 special token `[COG]`,让它 cross-attention query 前面所有 token 的 hidden state,输出一个 scalar score。这个 score 判断"当前 sub-task 是否完成"。inference 时 threshold 0.7,要求 3 consecutive frames 都高于 threshold 才 trigger sub-task 切换。

这本质是一个 **learnable option termination function**——hierarchical RL 里的概念,但用 supervised 方式训。annotation:每个 sub-task 最后 90 帧(robot)/ 30 帧(human)标 1,其它标 0。

### 2. Dual-Track Memory

两条 track 并行:
- **Visual track**:当前 frame + 5 个历史 frame(1s 间隔,覆盖 5s window),用 Qwen3-VL 的 multi-image capacity 编码
- **Proprioceptive track**:transformer encoder 编码历史 robot state

flow-matching decoder 在每一层都 cross-attend 这两个 context。

为什么 dual-track?ablation 显示:visual-only 能判断"target 找到没",但 action 抖;proprio-only action 流畅,但不知道 target 找到没,会反复扫同一区域。两个解耦建模再 fuse,比 single coupled memory 涨 10% SR。

### 3. Flow-Matching Action Decoder

照搬 $\pi_0$ 的设计,输出 30 步 future action chunk,每步 29 维(viewpoint pos+rot + 双臂 pos+rot + 双 gripper)。inference 5 步 denoising,执行前 10 步再 replan。

---

## 三阶段训练

这个 staging 很关键:

**Stage 1**:只训 VLM + cognitive head,让 model 先学会"判断 task 进度"。用 focal loss 监督 cognitive label。

**Stage 2**:解冻所有 module,在 human data 上 joint train cognitive + action。cognitive loss 权重降到 0.5,让 action loss 主导。

**Stage 3**:切到 robot data,继续 full-model finetune。cognitive loss 权重拉回 0.8,让 cognitive grounding 在 robot 上 refine。

为什么要 staging?ablation 显示跳过 Stage 1 直接 joint train,SR 从 83.3% 跌到 52.0%。直觉:如果 model 同时从零学 high-level cognition 和 low-level motor control,两者互相干扰。先 establish cognitive awareness,再用它 guide motor control,效果好得多。这跟 curriculum learning 的直觉一致。

---

## 实验结果

最 striking 的数据:

**vs baselines**:
- OpenVLA-OFT: 12.7% SR
- $\pi_{0.5}$: 16.0% SR
- ACT: 2.7% SR
- Diffusion Policy: 16.7% SR
- CoMe-VLA: 83.3% SR

**data scaling**:
- 0 human + 400k robot: 42.7% SR
- 400k human + 400k robot: 72.0% SR
- 800k human + 400k robot: 87.3% SR
- 800k human + 100k robot: 83.3% SR

最后两行最关键——human data 从 400k 翻倍到 800k 涨 15%,但 robot data 从 400k 砍到 100k 只跌 4%。说明 **human data 提供的 exploratory prior 非常 dense**,robot data 主要用于 embodiment grounding,不需要太多。

**memory ablation**:
- No memory: 40.7% SR
- Visual-only: 52.7% SR
- Proprio-only: 61.3% SR
- Single coupled: 73.3% SR
- Dual-track: 83.3% SR

Dual-track 比 single coupled 涨 10%,说明 visual 和 proprio 的 temporal context 需要分别建模,不能简单 fuse。

**cognition ablation**:
- w/o cognition: 22.7% SR
- w/o Stage 1: 52.0% SR
- Full: 83.3% SR

Cognition 贡献 60% SR,Stage 1 贡献 31% SR。这两个 component 都 essential。

---

## 最让我兴奋的点

1. **Cognitive head 作为 learnable option termination**:hierarchical RL 里 option termination 一直很难学,paper 用 supervised label 简化了。未来如果结合 RL 让这个 head 自主探索,可能 unlock 更 complex 的 long-horizon task。

2. **Human data scaling law**:如果 human data 真能 dense transfer 到 robot,robot learning 的 bottleneck 会从 robot data collection 转向 human data curation。这跟 LLM 用 human text pretrain 再 align 的范式平行。

3. **Dual-track memory 呼应 neuroscience**:dorsal/ventral stream 双通路在 robot 上也 work,说明 visual 和 motor 的解耦是 inductive bias 而非 over-engineering。

4. **Non-Markovian formalization**:把 active perception formalize 成 NMDP 而非 MDP,这个 framing 会让后续 work 更 serious 地处理 history。当前 VLA 主流还是 single-frame reactive,这个 paper 是一个 push。

---

## 局限

paper 自己也承认:
- Memory 窗口固定 5s,对 minutes-scale task 不够
- 还是需要 on-robot teleop data,没完全 eliminate
- Cognitive label 需要 supervised annotation,未来应该用 RL 自主学

---

## 一句话总结

**把 active perception 当作 intentional act 而非被动 sensing,用 cognitive head 监督信息 uncertainty resolution,用 dual-track memory 处理 non-Markovian history,用 human egocentric data 提供 exploratory prior,用 staged training 避免 cognition 和 motor 互相干扰——这套组合让 wheel-based humanoid 在 5 个 long-horizon active perception task 上达到 83.3% SR,比 baselines 高 5 倍。**

核心 insight:robot policy 需要 **memory + cognition + active sensing**,缺一不可。现有 VLA 只有 reactive mapping,所以在 active perception 任务上 fail 是必然的。

---

project page: https://jern-li.github.io/asa/

这篇 paper 读完,我最大的 takeaway 是:**robot learning 的下一个 frontier 不是更大的 reactive VLA,而是把 perception 和 action 真正 couple 起来的 cognitive agent**。CoMe-VLA 是这个方向的一个 solid step。

---

# Act, Sense, Act: 深度解析

## 1. 整体 intuition：为什么这篇 paper 重要

这篇工作本质上是在向一个根本问题发起攻击：**当前 robot learning 把 perception 当作被动 exogenous input，而真正的 active perception 应该是 intentional act 的一部分**。Karpathy 你一定很熟悉这个 framing——它呼应了你常说的一件事：policy 不应该仅仅是 `observation → action` 的 stationary mapping，而是 `history → action` 的非平稳过程，其中 observation 本身是 agent 通过先前的 action 主动构造出来的。

paper 把 active perception formalize 为一个 **Non-Markovian Decision Process (NMDP)**，并把它的驱动力拆成两个机制：
1. **Information gain**——action 应该减少 task-relevant 的不确定性
2. **Decision branching**——不同的 perceptual outcome 应该触发不同的后续 action

这非常 elegant，因为它把"找东西"和"用东西"统一到了同一个 decision-theoretic framework 下，并且非常诚实地承认了：这两个机制本质上要求 agent 有 memory。

project page: https://jern-li.github.io/asa/

---

## 2. Active Perception 的形式化：NMDP 视角

### 2.1 Passive vs Active Perception（Fig. 2 的本质）

被动 perception 的 flow 是：
$$o_t \xrightarrow{\pi} a_t \rightarrow o_{t+1} \xrightarrow{\pi} a_{t+1} \dots$$

active perception 的 flow 是闭合的：
$$a_{t-1} \rightarrow o_t \xrightarrow{\pi(\cdot | h_t, o_t)} a_t \rightarrow o_{t+1} \dots$$

**关键直觉**：action 不仅改变 world state，也改变下一个 observation 的内容（视角、occlusion 关系、距离）。observation 是 action history 的函数，而 policy 又是 observation history 的函数。这就是 non-Markovian 的根源——你不能只看当前 $o_t$，你必须看 $h_t = \{o_0, A_0, o_1, A_1, \dots, o_{t-1}, A_{t-1}\}$。

### 2.2 Eq. (1) Information Gain 详解

$$I(o_{t+1}; \mathbf{A}_t \mid h_t, o_t)$$

变量含义：
- $o_{t+1} \in \mathcal{O}$：下一个时刻的 observation（高维 RGB frame）
- $\mathbf{A}_t \in \mathbb{A}$：当前 action chunk（这里写成 chunk 而非 single action，是因为 CoMe-VLA 用 flow-matching decoder 输出 $K$ 步未来 action，呼应 $\pi_0$ 的设计）
- $h_t = \{o_0, \mathbf{A}_0, o_1, \mathbf{A}_1, \dots, o_{t-1}, \mathbf{A}_{t-1}\}$：完整 trajectory history
- $o_t$：当前 observation
- $I(\cdot ; \cdot \mid \cdot, \cdot)$：conditional mutual information

直觉上这个量度量的是："**给定我现在所知道的一切 $(h_t, o_t)$，采取 action chunk $\mathbf{A}_t$ 能让我对未来 observation $o_{t+1}$ 多多少新信息**"。在 active perception 的语境下，最大化这个量意味着 action 不仅是 manipulation 工具，也是 **epistemic tool**——例如转头去看桌子左边、打开抽屉揭示内部。

注意，paper 没有显式 optimize Eq. (1)（因为高维 visual space 算 mutual information 不可行），而是用 cognitive auxiliary head 作为 proxy（见下文 Eq. (4)）。这是一个重要的设计取舍：**信息论给出 principle，neural network 给出 approximation**。

### 2.3 Eq. (2) Decision Branching

$$\mathbf{A}_t = \pi(h_t, o_t)$$

变量：
- $\pi$：policy
- $\mathbf{A}_t \in \mathbb{R}^{K \times 29}$：K 步 future action chunk，每维 29 = (viewpoint pos 3 + 6D rot 6) + (left EE pos 3 + 6D rot 6) + (right EE pos 3 + 6D rot 6) + (left gripper 1 + right gripper 1) = 3+6+3+6+3+6+1+1 = 29

paper 把 branching 分成两类：
- **Exploratory branching**：根据 perceptual evidence 切换 exploration 策略（找到 target → gaze；没找到 → 继续扫描）
- **Exploitative branching**：根据 resolved perceptual state 切换 manipulation 策略（target 在左 → 左手抓；target 在右 → 右手抓）

这两个 branching 形式实际上对应了 RL 中两个不同 type 的 non-stationarity：exploration 时的 belief update，exploitation 时的 conditional policy。

### 2.4 两种 Paradigm（Section III-B）

paper 给出了一个很 clean 的 taxonomy：

**Information Discovery (ID)**：target 一开始不在 view 里
- **Viewpoint Discovery (VD)**：agent 改变自己 viewpoint（head pan-tilt、base movement）来定位
  - 例：Croissant Search（桌上找 croissant）、Can Disposal（房间找 dustbin）
- **Manipulation Discovery (MD)**：agent 物理交互 environment 来揭示信息
  - 例：Bottle Retrieval（打开 cabinet 看 shelf）、Cylinder Hunt（掀开两个倒扣的碗）

**Information Enrichment (IE)**：target 在 view 里但信息不够
- 例：Ring Peg（peg 太远，需要先抓近才能精确插入）

这个 taxonomy 很 useful，因为它直接映射到 evaluation 的 5 个 task，并且每个 paradigm 需要不同的 active perception 能力：VD 主要需要 chassis/head 控制，MD 需要把 manipulation 当作 sensing tool，IE 需要精细的 zoom-in。

---

## 3. 数据：从 human egocentric data 到 unified action space

### 3.1 Human Data 选择

paper 选了两个 dataset（Table II）：

| Dataset | 场景 | Takes | Samples | Unique Lang. | Curation |
|---|---|---|---|---|---|
| CaptainCook4D | Cooking | 257 | 1,071,604 | 349 | HoloLens GT |
| Ego-Exo4D | Cook, Bike, Covid | 249 | 421,582 | 2,730 | MANO optimization + strict visibility filtering |

CaptainCook4D 用 HoloLens [28] 采集，提供 high-quality head + hand pose，环境相对 structured。Ego-Exo4D 用 Aria glasses [26] 采集，环境 unconstrained，但 raw annotation 只有 wrist position 缺 3D rotation，所以 paper 用 MANO [34] optimization 补全 6DoF hand pose。

选择标准：
1. Fine-grained annotations
2. Synchronized hand and head/camera poses
3. Active perception 相关场景（clutter, occlusion, viewpoint adjustment）

总 human data：800k samples（10 Hz, 22.2 hrs）。CaptainCook4D ref: https://arxiv.org/abs/2410.14665 ，Ego-Exo4D ref: https://egoexo4ddata.org/ 。

### 3.2 Egocentric Robot Teleoperation

硬件：Corenetic Monte02 [11]，wheel-based humanoid，23 DoFs（实际用 chassis 2 + head 2 + 双臂 7+7 + 双 gripper 1+1 = 20 DoFs；waist 3 在实验中 disabled）。

Teleop：Meta Quest 3 [27] VR headset，operator 只看 robot 头戴相机的 egocentric stream，**不知道 target 位置**。这是非常关键的 "blind protocol"——确保 operator 必须主动 search，这样 trajectory 里嵌入的就是 genuine information-seeking behavior，不是 memorized path。

robot data: per task 100k-400k samples（30 Hz, 0.9 hrs/task）。

### 3.3 Human-Robot Data Alignment: Unified Egocentric Action Space

这是 paper 的一个核心技术贡献。核心问题：human 和 robot 的 morphology 不同，如何把它们对齐到同一个 action space？

paper 的答案是 **structural isomorphism**：定义 episode base frame $B$ 为 trajectory 第一帧的 head pose，所有后续 pose 都表达成相对于 $B$ 的 body-centric local frame $\mathcal{L}$（x forward, y left, z up）。

**Eq. (3) Frame transformation**:

$$T_i^B(t) = (T_B^W)^{-1} \cdot T_i^W(t) \cdot T_{i_j \mathcal{L}}$$

变量：
- $T_i^B(t) \in SE(3)$：部件 $i$ 在 t 时刻、以 base frame $B$ 为参考的 pose
- $T_B^W \in SE(3)$：base frame $B$ 在 world frame $W$ 中的 pose（在 Eq. (10) 中定义为 $T_H^W(0)$）
- $T_i^W(t) \in SE(3)$：部件 $i$ 在 t 时刻、world frame $W$ 中的 pose
- $T_{i_j \mathcal{L}} \in SE(3)$：从 local frame $\mathcal{L}$ 到部件 $i$ 的本征 frame 的 static transform（per dataset $j$ 不同的 offset）
- $i \in \{H, L, R\}$：head、left wrist、right wrist
- $j \in \{\text{CaptainCook4D}, \text{Ego-Exo4D}, \text{Robot}\}$
- $(\cdot)^{-1}$：SE(3) 上的逆变换
- $\cdot$：matrix multiplication

**直觉**：第一步 $(T_B^W)^{-1}$ 把 world 坐标系的原点放到 episode 起始 head 位置；第二步 $T_i^W(t)$ 把部件 pose 拉到 world；第三步 $T_{i_j \mathcal{L}}$ 校准不同 dataset 的 local convention 偏移。这样无论 human 还是 robot，所有 trajectory 都表达在同一个 "starting-from-head-pose" 的 body-centric frame 里。

**Hand → gripper 映射**：human hand 高维 configuration 被 abstract 成一个 scalar gripper width——thumb tip 到其他 fingertips 距离的 average。这把 21-joint hand pose 压成 1D 抓握信号，类似 DexCap [37] 和 DexUMI [45] 的简化思路。

**Chassis + Head 复合 pose**：robot 的 head pose = chassis yaw + head gimbal。Eq. (13):

$$R_H^O(t) = \text{Rot}_z(\psi_C(t) + \psi_H(t)) \cdot \text{Rot}_y(\theta_H(t))$$

变量：
- $R_H^O(t) \in SO(3)$：head 在 odometry frame $O$ 中的 rotation
- $\text{Rot}_z$：绕 z 轴的 yaw rotation
- $\text{Rot}_y$：绕 y 轴的 pitch rotation
- $\psi_C(t)$：chassis yaw 在 odometry 中
- $\psi_H(t)$：head yaw 相对 chassis
- $\theta_H(t)$：head pitch 相对 chassis

inference 时通过 threshold-based heuristic 分解回 chassis + head gimbal 命令（$|\psi| > 0.7$ rad 用 chassis，否则用 gimbal）。这个设计很简洁，把 mobile manipulation 的 long-range motion 和 tabletop 的 fine motion 用一个统一 representation 表达。

---

## 4. CoMe-VLA 架构

### 4.1 整体结构（Fig. 4）

backbone：Qwen3-VL-2B [2]（去掉 LM head，2.79B 总参数）。Qwen3-VL ref: https://arxiv.org/abs/2511.21631

输入：
- Temporal visual observations（current + 5 个历史 frame，1s 间隔）
- Task description（text）
- Cognitive token `[COG]`
- Temporal proprioceptive states

输出：
- Action chunk $\mathbf{A}_t \in \mathbb{R}^{K \times 29}$（$K = 30$）
- Cognitive label $c_t \in \{0, 1\}$

### 4.2 Cognitive Auxiliary Head

paper 的核心 claim：active perception 由 information gain 驱动，但高维 visual space 算 mutual information 不可行。所以引入一个 learnable proxy。

**Eq. (4)**:
$$c_t \approx f(I(o_{t+1}; \mathbf{A}_t \mid h_t, o_t))$$

变量：
- $c_t \in \{0, 1\}$：binary cognitive label（"是否一个 sub-task 完成 / 信息 uncertainty 是否 resolved"）
- $f(\cdot)$：implicit learned mapping
- $I(\cdot)$：mutual information（来自 Eq. (1)）

实现细节（附录 III-A）：
- 用 Qwen3-VL tokenizer 中 token 151621 作为 cognitive input token（一个 rarely used token）
- 不直接把 cognitive latent token 喂入 MLP，而是让它通过 cross-attention（8 heads, hidden dim 2048）query 前面所有 token 的 hidden state
- 然后过 two-layer MLP 投影到 scalar cognitive score
- inference 时 threshold $\tau = 0.7$，要求至少 3 consecutive timesteps 高于 threshold 才有效

**作用**：当 cognitive label 翻转时，sub-task textual instruction 切换，policy 被 recondition 到新的 goal 上。这相当于一个 learned sub-goal detector，避免了 hardcoded 状态机。

这是一个非常 elegant 的设计，呼应了 hierarchical RL 中的 option termination function，但用 supervised 方式训练。

### 4.3 Dual-Track Memory

paper 强调 active perception 是 non-Markovian 的，需要 history。但全 raw history 太贵，所以用 dual-track：

**Track 1: Visual memory**
- 利用 Qwen3-VL 的 multi-image capacity
- 当前 frame + 5 个历史 frame（1s 间隔，覆盖过去 5s）
- 保留 spatial awareness

**Track 2: Proprioceptive memory**
- Transformer-based encoder（1-layer MLP + LayerNorm + 2-layer transformer encoder）
- 8-head self-attention, hidden 2048, FFN 8192
- 编码 temporal dynamics of motor behavior

**Cross-attention 融合**：flow-matching action decoder 在每一层都对 dual context 做 cross-attention。这相当于让 action generation 同时 grounded 在 visual scene 和 motor history 上。

ablation（Table VII）显示：
- No memory: SR 40.7%, ST 485.6s
- Visual-only: SR 52.7%, ST 340.9s
- Proprio-only: SR 61.3%, ST 170.0s
- Single coupled: SR 73.3%, ST 143.1s
- Dual-track: SR 83.3%, ST 97.9s

**直觉解读**：visual-only 能维持认知判断但 motor 执行差；proprio-only motor 连续性好但认知差（不知道 target 是否找到，无法根据 perceptual feedback 切换策略）。Dual-track 让两者解耦建模，再在 decoder 层 cross-attention 融合，这是 paper 的关键 architectural insight。

### 4.4 Flow-Matching Action Decoder

参考 $\pi_0$ [5] 和 $\pi_{0.5}$ [6]，用 flow-matching 而非 diffusion。架构（附录 III-A）：
- 两个 one-layer MLP 分别投影 viewpoint action noise 和 manipulation action noise
- Concatenate 但保留 modality identity 用于 self-attention masking
- 6 个 transformer block，每 block = 1 self-attention + 2 cross-attention + FFN
- Hidden 2048, 8 heads, FFN 8192
- Residual connection + AdaRMSNorm [44]

inference：5 个 denoising steps（实验发现再增加没收益），action chunk $K=30$，receding-horizon 执行前 10 步再 replan。

$\pi_0$ paper: https://arxiv.org/abs/2410.24164
$\pi_{0.5}$ paper: https://arxiv.org/abs/2504.16054

### 4.5 Implicit Data-Driven Decisions

paper 的一个反直觉点：**没有显式 modeling branching as discrete symbolic choices**，而是让 policy 在 large-scale human data 上学习，implicit capture branching 作为 perception-action coupling 的 continuous variations。

这个设计哲学跟 classic 的 active vision 工作（如 next-best-view [13, 7]、gaze control [18, 35, 10]）形成对比——后者把 branching 显式化、要求 exhaustive iteration。CoMe-VLA 选择把 branching 隐含在 policy 的 multimodal action distribution 里，让 flow-matching 自然表达。

---

## 5. 三阶段训练策略

paper 的训练 pipeline 是 staged 的，每个 stage 解锁不同 module：

### Stage 1: Cognitive State Pretraining
- 只更新 VLM + cognitive auxiliary head
- 其它 module frozen
- 用 focal loss 监督 cognitive label

**Eq. (5) Focal Loss**:
$$\mathcal{L}_1 = \mathcal{L}_{cog} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

变量：
- $p_t$：模型预测的 cognitive label 为 1 的概率
- $\alpha_t$：balancing factor（class imbalance 调节，paper 中 = 0.25）
- $\gamma$：focusing parameter（让难分样本获得更高权重，paper 中 = 2.0）
- $(1 - p_t)^\gamma$：modulating factor，对容易分类的样本（$p_t \to 1$）降低 loss 权重

**直觉**：Stage 1 先把 model 训成会"判断 task 进度"的 cognitive module，建立 high-level awareness，再让它去学 motor control。这样避免了 cognition 和 action 互相干扰（消融实验显示跳过 Stage 1 会让 SR 跌 31.3%，ST 涨 213.9s）。

### Stage 2: Cognition-Action Joint Pretraining
- 解冻所有 module
- 在 human data 上同时训 cognitive head + action decoder
- 引入 action MSE loss

**Eq. (6) Action MSE Loss**:
$$\mathcal{L}_{action} = \sum_{i \in \{vr, vp, er, ep, g\}} \lambda_i \|\hat{\mathbf{v}}_i - \mathbf{v}_i^*\|_2^2$$

变量：
- $\hat{\mathbf{v}}_i$：模型预测的速度序列
- $\mathbf{v}_i^*$：ground-truth 速度序列
- $\|\cdot\|_2^2$：L2 norm 的平方
- $\lambda_i$：每个 action component 的权重
- $i \in \{vr, vp, er, ep, g\}$：viewpoint rotation, viewpoint position, end-effector rotation, end-effector position, gripper state

权重设置（Table IV）：$\lambda_{vr} = 1.5$（viewpoint rotation 权重最高，强调 active perception 的视角控制），其他都是 1.0。

**Eq. (7) Joint Loss**:
$$\mathcal{L}_{joint} = \lambda_t \mathcal{L}_{cog} + \mathcal{L}_{action}$$

变量：
- $\lambda_t$：cognitive loss 的权重（Stage 1: 1.0，Stage 2: 0.5，Stage 3: 0.8）
- $\mathcal{L}_{cog}$：cognitive focal loss
- $\mathcal{L}_{action}$：action MSE loss

### Stage 3: Robot Data Finetuning
- 切换到 robot data
- 同样的 loss structure
- grounding 之前学到的 priors 到具体 embodiment

整个 3-stage 训练在 8x H100 HBM3 80G 上跑 3 天。

**Training hyperparameters**（Table IV）：

| Stage | LR | Schedule | Epoch | Chunk | $\alpha_t$ | $\gamma$ | $\lambda_t$ | $\lambda_{vr}$ | $\lambda_{vp}$ | $\lambda_{er}$ | $\lambda_{ep}$ | $\lambda_g$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2e-5 | Cosine | 10 | - | 0.25 | 2.0 | 1.0 | - | - | - | - | - |
| 2 | 1e-4 | Cosine | 20 | 30 | 0.25 | 2.0 | 0.5 | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| 3 | 2e-5 | Cosine | 50 | 30 | 0.25 | 2.0 | 0.8 | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 |

注意 Stage 1 的 $\lambda_t = 1.0$ 是因为只有 cognitive loss；Stage 2 降到 0.5 让 action loss 占主导；Stage 3 拉回 0.8 让 cognitive grounding 在 robot data 上继续 refine。

---

## 6. 实验

### 6.1 五个 task

每个 task 都有 uncertain initial condition（target 位置 randomize 在多个 configuration 之间）：

1. **Croissant Search (ID-VD)**：croissant 一开始在 view 外（左或右），需要 head scan 定位，然后 pick-and-place
2. **Can Disposal (ID-VD)**：抓两个 can，chassis 转动 search 房间里的 dustbin（在 behind/right/right-back 三个位置）
3. **Bottle Retrieval (ID-MD)**：cabinet 里有 bottle（upper 或 lower shelf），需要打开 cabinet 才能看见，再 grab
4. **Cylinder Hunt (ID-MD)**：cylinder 隐藏在两个倒扣碗中的一个下面，需要掀开搜索
5. **Ring Peg (IE)**：peg 一开始可能近或远，需要先抓近再精确插入

### 6.2 主对比结果（Table I）

| Method | Croissant SR | Can SR | Bottle SR | Cylinder SR | RingPeg SR | Mean SR | Mean ST |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT [20] | 3/30 | 7/30 | 1/30 | 2/30 | 6/30 | 12.7% | 740.1s |
| $\pi_{0.5}$ [6] | 5/30 | 6/30 | 4/30 | 2/30 | 7/30 | 16.0% | 616.0s |
| ACT [49] | 0/30 | 0/30 | 0/30 | 0/30 | 4/30 | 2.7% | 859.8s |
| DP [9] | 9/30 | 0/30 | 3/30 | 5/30 | 8/30 | 16.7% | 710.7s |
| CoMe-VLA (0+400k) | 6/30 | 15/30 | 10/30 | 13/30 | 20/30 | 42.7% | 329.0s |
| CoMe-VLA (400k+400k) | 24/30 | 18/30 | 16/30 | 22/30 | 28/30 | 72.0% | 144.3s |
| CoMe-VLA (800k+400k) | 28/30 | 24/30 | 21/30 | 28/30 | 30/30 | 87.3% | 93.4s |
| **CoMe-VLA (800k+100k)** | 28/30 | 22/30 | 18/30 | 27/30 | 30/30 | **83.3%** | **97.9s** |

**几个关键观察**：

1. **Illusory Exploration**：$\pi_{0.5}$ 在 Croissant Search 上 ST 116.6s 看似不错，但实际是 stochastic random-walk 行为，target 进入视野后无法稳定 gaze。说明 ST 单独不可信，必须和 SR 一起看。

2. **Weak Visuo-Motor Grounding**：$\pi_{0.5}$ 有时出现"gaze 向左但抓取向右"的 misaligned perception-action，说明它的 visual attention 和 motor control 没有正确 ground 起来。

3. **Data scaling 效应**：从 (0+400k) 到 (400k+400k) 跳跃 30% SR，从 (400k+400k) 到 (800k+400k) 又涨 15%，从 (800k+400k) 到 (800k+100k) 只跌 4%。这说明 **human data 提供的 exploratory prior 是 dense 的**，极大降低了 robot data 的 sample complexity。这对未来 robot data 稀缺的场景非常关键。

4. **ACT 全军覆没**（除了 Ring Peg 4/30）：ACT 是 single-task 模型，没有 sub-task switching，面对 long-horizon active perception 完全不行。这是 paper 想强调的——active perception 不是 single-task reactive control 问题。

### 6.3 Cognition Ablation（Table V）

| Method | Mean SR | Mean ST |
|---|---|---|
| w/o Cog. | 22.7% | 616.4s |
| w/o Stage-1 | 52.0% | 311.8s |
| Ours | 83.3% | 97.9s |

**直觉**：cognition 的作用是把 "high-level semantic reasoning"（target 是否找到）和 "low-level visuo-motor grounding"（精确控制）解耦。如果硬塞到一起，会出现 severe instability。Stage 1 先建立 cognitive representation，再让 model 用 high-level awareness refine low-level motor command——这是 curriculum 的精髓。

### 6.4 Memory Length Ablation（Table VI）

| History Frames | Mean SR | Mean ST |
|---|---|---|
| 1 | 44.6% | 518.8s |
| 3 | 59.3% | 218.2s |
| 5 (Ours) | 83.3% | 97.9s |

5s 的 temporal context 是 sweet spot——更长会让 VLM 的 visual token 序列爆炸，但更短不足以支撑 active perception 的 non-Markovian 特性。

### 6.5 Memory Architecture Ablation（Table VII）

| Memory | Mean SR | Mean ST |
|---|---|---|
| No memory | 40.7% | 485.6s |
| Visual-only | 52.7% | 340.9s |
| Proprio-only | 61.3% | 170.0s |
| Single coupled | 73.3% | 143.1s |
| **Dual-track** | **83.3%** | **97.9s** |

**直觉解读**：
- Visual-only 让 model 知道"我在哪、看过什么"，但没有 motor continuity，action 执行抖动
- Proprio-only 让 motor 流畅，但 model 不知道 perceptual outcome，会反复搜索同一区域
- Single coupled 把两者 feature 简单 fuse，比 dual-track 差 10% SR
- Dual-track 在 decoder 层 cross-attention 解耦融合——visual 和 proprio 各自建模 temporal context，再让 action decoder ground 在两者之上

这个结果呼应了 neuroscience 中 dorsal (motor) / ventral (perception) stream 的双通路假设：visual processing 和 motor control 在 brain 中是部分独立的 pathway，再在更高层 integrate。

### 6.6 Robustness Analysis

10 分钟 adversarial test on Croissant Search（grasping 禁用，纯 search mode）：
- **Sudden Disappearance**：target 被完全移走，robot 持续 stochastic scanning，不陷入 fixed viewpoint trap
- **Relocation**：target 被移动到 out-of-sight 位置，robot 切换回 search mode
- **Reintroduction**：target 重新出现并进入视野，model 立即 visual locking 并 sustained gaze
- **再次 Disappearance**：立即回到 search mode

这种 search ↔ gaze 的 seamless 切换说明 model 学到的是真正的 active perception 策略，不是 memorized trajectory。

---

## 7. Robot Platform 和 Inference 细节

### 7.1 Gripper Discretization（Eq. 18）

$$w_t = \begin{cases} w_{max} & \text{if } a_{grip} > 0.7 \quad (\text{Open}) \\ 0 & \text{if } a_{grip} < 0.3 \quad (\text{Close}) \\ w_{t-1} & \text{otherwise} \quad (\text{Hold}) \end{cases}$$

变量：
- $w_t$：实际发给 robot 的 gripper 宽度
- $a_{grip} \in [0, 1]$：policy 输出的 normalized gripper value
- $w_{max} = 0.1$ m：最大张开宽度
- $w_{t-1}$：上一时刻命令（hold mode）

dead-zone 设计避免 jitter，这是简单但实用的工程细节。

### 7.2 Chassis Control（3-stage 状态机）

给定 target pose $\mathbf{p}_{tgt} = (x_t, y_t, \theta_t)$ 和 current pose $\mathbf{p}_{curr} = (x, y, \theta)$：

1. **Heading Alignment**：原地转向对齐 bearing line，bidirectional 选最短转角
   - $\omega = k_\omega e_\psi$，其中 $e_\psi = \text{atan2}(y_t - y, x_t - x) - \theta$
2. **Linear Approach**：直线前进
   - $v = k_v e_d$，其中 $e_d = \|(x_t, y_t) - (x, y)\|_2$
3. **Final Orientation**：到达后旋转匹配 target yaw
   - $\omega = k_\omega e_\theta$，其中 $e_\theta = \theta_t - \theta$

参数：$k_v = 0.2$, $k_\omega = 0.5$, $v \in [0.08, 0.15]$ m/s, $\omega \in [0.15, 0.3]$ rad/s。dead-zone: $\epsilon_p = 0.015$ m, $\epsilon_\psi = 0.5$ rad, $\epsilon_\theta = 0.05$ rad。

### 7.3 Cognitive Label 部署

threshold $\tau = 0.7$，要求至少 3 consecutive timesteps 高于阈值才有效。这避免 noise 导致的 false trigger。

annotation：每个 sub-task 的最后 90 帧（robot）/ 30 帧（human）标 $c_t = 1$，其它 $c_t = 0$。

---

## 8. 与 concurrent work 的对比

paper 在附录 V-B 提到几个 concurrent work：
- **ViA [43]** (https://arxiv.org/abs/2506.15666)：VR-based teleop，把 active perception 当 extra action dimension
- **ActiveUMI [48]** (https://arxiv.org/abs/2510.01607)：UMI-based data collection
- **EgoMI [47]** (https://arxiv.org/abs/2511.00153)：有 SPARKS memory module，但专门针对 egocentric human demo 的 viewpoint shift 问题

CoMe-VLA 的差异点：
1. 不是把 head movement 当 extra action dim，而是把它当作 **strategic epistemic action**
2. 用 large-scale public human data 而非专用 teleop 设备
3. 双机制 formalization（information gain + decision branching），并 categorize paradigm
4. Cognitive auxiliary head + dual-track memory 显式 modeling non-Markovian 特性

---

## 9. Limitations 和未来方向

paper 自己承认：
1. **Memory 窗口固定**（5s）——对 minutes-scale long-horizon 不够，naive 扩展会引入过时信息。未来方向：video streaming input + belief-aware memory representation。
2. **仍需 on-robot teleop data**——human data 降低了 sample complexity 但没完全消除。未来：用 RL 让 robot 自主探索 active perception 策略。

---

## 10. 我的联想和延伸思考

### 10.1 跟你的 Micrograd / nanoGPT 教学风格的呼应

这篇 paper 实际上做了一个非常 elegant 的事：**把 active perception 的信息论 principle（Eq. 1）和 deep learning 的实践（Eq. 4 cognitive head proxy）bridge 起来**。这跟你经常强调的"先 build intuition 再写 code"的思路一致——paper 没有真去 maximize mutual information，而是承认 explicit computation intractable，用 learnable proxy。这种"用 inductive bias 编码 principle，用 SGD 学具体 mapping"的思路是现代 deep learning 的精髓。

### 10.2 跟 $\pi_0$ 家族的对比

CoMe-VLA 借用了 $\pi_0$ 的 flow-matching action decoder，但加了三个东西：
- Cognitive auxiliary head（sub-task termination detector）
- Dual-track memory（visual + proprio）
- Stage-wise training（cognition 先，joint 后，robot grounding 末）

$\pi_{0.5}$ 已经有 memory 但没有 explicit cognitive signal。CoMe-VLA 的 ablation 显示 explicit cognitive head 比纯 memory 涨 30% SR，说明 **hierarchical decomposition 在 long-horizon 任务中关键**。这跟你之前在 podcast 谈到的 "Hierarchical RL is hard, but maybe explicit option termination helps" 的直觉一致。

### 10.3 跟 world model / JEPA 路线的关联

LeCun 的 JEPA (https://arxiv.org/abs/2304.10546) 强调 latent predictive world model。CoMe-VLA 的 cognitive head 实际上学了一个 latent "is information uncertainty resolved" predictor，这本质上是 world model 的一部分。如果 paper 进一步把 cognitive head 扩展成 latent state predictor（预测 next observation 的 abstract representation），就接近 JEPA 范式。这是 paper 提到的 "belief-aware memory representation" 的自然延伸。

### 10.4 数据 scaling law

Table I 最后一行 (800k+100k) 比 (800k+400k) 只差 4% SR——这是非常强的 scaling signal。如果 human data 真的能 transfer 到 robot，那么未来 robot learning 的 bottleneck 可能从 robot data collection 转向 human data curation。这跟 EgoMimic [17] (https://arxiv.org/abs/2510.00099) 和 EMMA [50] (https://arxiv.org/abs/2509.04443) 的发现一致。

### 10.5 VLA 范式的演化

OpenVLA [19] (https://arxiv.org/abs/2406.09246) 是 single-frame reactive VLA
$\pi_0$ 引入 flow-matching action chunk
$\pi_{0.5}$ 加 open-world generalization
CoMe-VLA 再加 cognitive + dual-track memory + active perception formalization

可以看出 VLA 演化路线正在从 "reactive imitation" 走向 "cognitive, memory-augmented, active agent"。这是迈向 AGI-level robot 的必要路径。

### 10.6 一个值得深挖的问题

paper 没有讨论 active perception 的 **planning horizon** 问题。Eq. (1) 中的 $\mathbf{A}_t$ 是 action chunk，但 information gain 实际上应该是 multi-step lookahead——比如"先走到桌子另一边再看"vs"先开抽屉"哪个 information gain 更高。当前 paper 通过 supervised learning 隐式 learning 这个，但 RL 框架下显式 optimize 可能更 powerful。这也是 paper limitation 里提到的 RL 方向。

---

## 11. 总结

这篇 paper 的核心贡献，从 build intuition 的角度：

1. **Formalization**：把 active perception formalize 为 NMDP，由 information gain 和 decision branching 双驱动，并提供 ID-VD/ID-MD/IE 的 taxonomy。这是 concept 层面的 contribution。

2. **Architecture**：CoMe-VLA = Qwen3-VL + flow-matching decoder + cognitive auxiliary head + dual-track memory。Cognitive head 是 information gain 的 learnable proxy，dual-track memory 处理 non-Markovian 特性。

3. **Data strategy**：通过 unified egocentric action space 把 800k human data 和 robot data 对齐，stage-wise training 让 cognitive awareness 先于 motor control。Scaling law 显示 human data 能显著降低 robot data 需求。

4. **Empirical validation**：5 个 task 覆盖三种 paradigm，与 OpenVLA-OFT、$\pi_{0.5}$、ACT、DP 对比，SR 从 baselines 的 <17% 提升到 83.3%。Ablation 验证 cognition 和 dual-track memory 各自的贡献。

这是个非常 solid 的工作，从 principle formalization 到 architecture design 到 empirical validation 闭环完整。最让我兴奋的是 cognitive auxiliary head 的设计——它实际上是在 VLA 框架内引入了 **learnable option termination**，这是把 hierarchical RL 的关键 concept 用 supervised 方式实现。未来的 VLA 演化大概率会沿着这条路继续走，结合 RL 让 cognitive signal 不再依赖 supervised annotation。

---

**主要 references**：
- Project page: https://jern-li.github.io/asa/
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Ego-Exo4D: https://egoexo4ddata.org/
- CaptainCook4D: https://arxiv.org/abs/2410.14665
- ACT: https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ViA: https://arxiv.org/abs/2506.15666
- ActiveUMI: https://arxiv.org/abs/2510.01607
- EgoMI: https://arxiv.org/abs/2511.00153
- EgoMimic: https://arxiv.org/abs/2510.00099
- EMMA: https://arxiv.org/abs/2509.04443
- Focal Loss: https://arxiv.org/abs/1708.02002
- JEPA: https://arxiv.org/abs/2304.10546
- 6D rotation representation: https://arxiv.org/abs/2202.12713
