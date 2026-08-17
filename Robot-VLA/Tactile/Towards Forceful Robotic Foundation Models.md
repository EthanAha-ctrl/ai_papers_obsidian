---
source_pdf: Towards Forceful Robotic Foundation Models.pdf
paper_sha256: 80e4243508c4a64c06d24e12f88668ae9607ecd445654b70f1e33d8e400ae567
processed_at: '2026-08-12T17:08:26-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我用最接地气的方式重新讲一遍这篇paper的核心故事。

---

## 一句话总结

**现在的机器人基础模型全是"瞎子摸象"——只有眼睛（vision）和手脚（position control），但没有触觉。这篇paper就说：兄弟，你要让机器人真正能干精细活儿，光靠看是不够的，你得让它能"感觉到"自己在用力。**

---

## 为什么要搞力觉？一个最直觉的例子

你闭上眼睛，伸手去拿桌上的鸡蛋。你眼睛看不到，但你的手指头能感觉到压力在变大，你就知道"差不多该停了，再用力就碎了"。

现在的机器人呢？它只能看到"鸡蛋在那儿"，然后算出一个目标位置，伸手过去夹住。但是 position 的微小误差在 stiff system 里会导致巨大的 force 误差。你差了1毫米，可能 force 就多了5牛顿，鸡蛋就碎了。

从物理上说：

$$F = m\ddot{x}$$

Force 是 position 的二阶导数。这意味着 force 里面包含了 position 没有的高频信息。你只学 position，就像只听一首歌的低音部分，高音全丢了。那些 contact-rich 的精细操作——倒水、插销钉、拿易碎物品——全都是高频动态的事儿，你低频信号根本 capture 不到。

---

## 人类是怎么干的？生物学启发

人的手上有四种 mechanoreceptor，你可以理解成四种不同的"触觉传感器"：

| 类型 | 俗名 | 频率范围 | 干什么用的 |
|------|------|----------|------------|
| FA-I | Meissner corpuscles | 20-200Hz | 检测纹理、滑移，receptive field 小而清晰 |
| FA-II | Pacinian corpuscles | 到1500Hz | 高频振动，用工具时关键，receptive field 大而模糊 |
| SA-I | Merkel cells | 持续压力 | 形状和纹理识别，空间精度高 |
| SA-II | Ruffini endings | 皮肤拉伸 | 手指位置和姿势的本体感觉 |

然后人还有 proprioception——肌肉里的 muscle spindles 感知肌肉长度，Golgi tendon organs 感知力的大小（超阈值就自动抑制收缩防止拉伤），joint receptors 感知关节角度。

有意思的是，实验表明：你把人的手指局部麻醉了（触觉没了），人还能完成任务，但会严重变笨拙。你把人的 proprioception 弄没了（有这种罕见神经病例），人也能重新学会控制肌肉，但得全程盯着自己的手看。

**核心 insight：人类的传感系统有巨大的冗余性。触觉不是"必须有"才能完成任务，但有触觉会让任务完成得快得多、好得多、鲁棒得多。** 机器人也一样——纯 vision + position 也能干很多事，但加了 force 之后性能会有质的飞跃。

---

## 25篇论文的全景图

这篇 survey 梳理了 25 篇用 transformer 或 diffusion 架构做 tactile robot policy 的论文。为什么只看 transformer 和 diffusion？因为这俩架构有 scaling properties，有潜力长成 foundation model。之前的 LSTM + impedance control 那些工作 [27-30] 不具备 scale 的潜力。

64 个实验任务，53 个独特任务，横跨两个数量级的 force（0.1N 到 10N+）和两个数量级的时间（0.1s 到 20s）。但是——没有一个统一的 benchmark task，也没有一个 policy 能搞定超过少数几个任务。这说明这个领域还在非常早期的"各自探索"阶段。

任务分布大概是：
- 6% 短时间 + 高力（>10N）——比如 hammer peg
- 22% 短时间 + 中等力（1-10N）——比如 peg insertion
- 17% 短时间 + 小力（0.1-1N）——比如 delicate grasp
- 40% 长时间 + 中等力——比如 wiping, pouring
- 6% 长时间 + 高力——比如 door open

---

## 传感器：百花齐放但也意味着乱七八糟

25篇论文用了14种不同的传感器产品。大类分6种：

1. **Visuotactile (finger)** — 36%的论文，全是 GelSight 家族的。GelSight 就是手指头上装个小摄像头，拍一个软胶的变形。好看、信息量大，但贵且 dimension 高。
2. **Whole-arm joint torque** — Franka Panda 或 Flexiv Rizon 自带的关节力矩传感。
3. **Wrist F/T sensor** — 手腕处装六维力传感器（ATI Mini 45 等）。
4. **Finger force sensor** — CoinFT 之类的指尖力传感器。
5. **Finger audio** — 手指上装麦克风，听接触的声音来推断纹理和事件。
6. **Combined** — 手指 + 关节力矩组合使用。

这个多样性本身就说明了问题：**没有标准化的传感硬件，数据无法跨平台对齐，这从根本上阻碍了大规模 pretraining。** 你想搞一个 tactile 的 Open-X dataset？传感器都不一样，怎么对齐？

---

## 数据采集：最大的瓶颈

76% 的论文用了不到 200 个 demonstration。这说明大家还在小规模摸索阶段。数据采集方法五大类：

### 1. Teleoperation（36%）
人用 VR、joystick、或者 leader-follower 机械臂来遥控。最常见但也最粗糙——人通过遥控器操作，感受到的力和机器人实际感受到的力完全不一样，embodiment gap 巨大。

### 2. Kinesthetic Teaching
人直接抓着机器人（处于 free-drive / gravity compensation 模式），手把手教它做任务。机器人身上的传感器能直接记录人施加的力。好处是力信号真实，坏处是累、慢、容易把贵重的传感器搞坏。比如 Hou et al. [78] 让人抓着 UR5E 的手腕教擦花瓶和翻转物体。Zhao et al. [92] 用这个方法教机器人插电容到电路板上，用 GelSight Wedge 间接感知力。

还有人更直接——把 GelSight Mini [91] 或 CoinFT [65] 绑在人的手指上，人直接用手指做任务，传感器记录人手指的力。这比抓着机器人更自然。

### 3. Bilateral Teleoperation（增强版遥操作）
给遥操作加力反馈。机器人感知到的力回传给人（通过 haptic device 或振动马达），人能"感觉"到机器人在碰什么。这是个折中方案：比纯 teleoperation 好很多，比 kinesthetic teaching 省力，但系统极其复杂、难做、难 scale。而且反馈给人体的力是经过处理的信号，不是原始的。

### 4. Handheld Grippers（最有潜力的方向！）
把夹爪从机械臂上拆下来，人直接拿在手里操作。夹爪上有 F/T sensor 和相机。人做任务，夹爪记录所有力交互数据。完事之后，数据可以直接迁移到任何机械臂上（因为数据里只有 end-effector 级别的力交互，没有机械臂运动学信息）。

Liu et al. [83] 更巧妙——在 UMI gripper [98] 的指尖装了接触式麦克风，通过 audio 信号来近似力反馈。能区分钩毛带的表面纹理！

这个方向的直觉非常清晰：**如果你要 scale 数据采集，就得让普通人也能做。你不能指望每个数据采集员都会操作 VR 遥控一个7自由度机械臂。但你确实可以让任何人拿着一个装了传感器的夹爪去抓东西。**

### 5. Simulation
Wang et al. [88] 在 Drake simulator 里生成了 50000 个 contact-rich 技能的轨迹，加上 400 个真实机器人 demonstration，用 domain randomization 来泛化。问题是：模拟器对动态摩擦和大力的建模还不够准。Sim2real 在触觉领域比 vision 领域难得多。

---

## Action Space：输出什么决定了你能干什么

64% 的 policy 输出 Cartesian position 指令——就是"去到这个xyz位置"。底层是个 position controller。

但最亮眼的几个工作都用了 **explicit force control**：

### Hou et al. [78] 的 Adaptive Compliance Policy
任务：擦花瓶、翻转物体。
做法：从 kinesthetic teaching 数据里提取 wrist F/T 和 position，后处理生成 admittance controller 的输入（virtual target pose + stiffness matrix K）。
启发式规则：力反馈方向上用低 stiffness（高 compliance），其他方向用高 stiffness。用 1秒移动平均滤波生成 future-contact-informed stiffness。
结果：相比用固定高/低 stiffness 的 ablation，成功率提高 81%。

直觉：擦花瓶时，你贴着花瓶表面的方向要"软"（否则花瓶碎了或擦不匀），垂直于表面的方向要"硬"（否则没压力擦不干净）。policy 学会了根据接触情况动态调整这个 stiffness。

### Wu et al. [89] 的 TacDiffusion
任务：peg insertion（公差 <0.5mm）。
做法：policy 直接输出 50-500Hz 的 target external wrench，底层接 1000Hz impedance controller。用 dynamic filter 做频率插值。
结果：平均 <2秒完成，成功率 >90%。同类 position control 方法至少要两倍时间。

直觉：peg insertion 这种任务，你需要的不是"去到某个位置"，而是"用力推，感觉到了就微调方向再推"。直接输出 wrench 让控制器能以 1000Hz 的频率做这种 reactive 微调，policy 只需要在 50Hz 给出"大方向"。

### Explicit force control 的三大优势

1. **Performance**：控制器跑 500-1000Hz，反应速度远超人，也远超 policy 的 10-50Hz 输出频率。
2. **Interpretability**：你输出的是一个标量 stiffness 或一个六维 wrench，人能看懂、能 debug。输出 7 个关节角度你完全不知道它在干嘛。
3. **Dimension reduction**：policy 不用学高频的力调节细节了，只要学"运动意图"和"刚度参数"。经典控制论帮你搞定剩下的。这跟人脑分工一样——大脑规划去哪儿，脊髓反射负责肌肉微调。

---

## Policy 架构：Diffusion 称王

52% 用 diffusion policy，20% 用 transformer（多为 ACT），16% 还在用 MLP。

**为什么 diffusion 这么受欢迎？**

人类示教数据是 multi-modal 的——你可以从左边绕过障碍物，也可以从右边绕。如果你用 MSE loss 训一个 MLP 或 transformer，它会把这俩平均起来，生成一个直接撞上障碍物的轨迹（mode collapse）。Diffusion 通过学习去噪过程来建模 score function，能自然地处理这种多模态分布。

Diffusion 的缺点：容易 overfit 到绝对的 robot state，换个位置就不行了。解决办法是去掉 robot state，用相对 position action 而不是绝对 position [71, 80]。

**ACT (Action Chunking Transformer)** [47] 解决另一个问题：long-horizon reasoning。它把动作切成时间上连贯的 chunk，一次预测一整段动作序列。这降低了高频 action prediction 的负担，同时保持控制平滑。特别适合需要长序列协调运动的触觉任务。

---

## Representation Learning：怎么编码触觉信号

### Visuotactile（GelSight 图像）
GelSight 输出的是高分辨率图像，需要像 vision 一样做 representation learning。

Jones et al. [80] 的 FuSe 是最有意思的工作。他们微调了 Octo foundation model，加了 GelSight DIGIT 和 microphone 数据。但直接加 MSE loss 不行——Octo 会忽略 touch 数据，只用它熟悉的 vision 和 position。

他们用 language 作为"胶水"设计了两个辅助 loss：

1. **Contrastive loss** $L_{contrast}$：最大化同一场景下不同模态和语义之间的互信息。比如"pick up the squishy object"这个指令和你的 touch embedding 应该对齐。

2. **Generative loss** $L_{gen}$：从 embedding 生成对应的 ground truth language 描述。迫使模型理解"我摸到的东西是软的"这种 high-level 语义。

结果：FuSe 能做 compositional reasoning，比如"pick the object that has the same color as the button that plays piano"——先视觉找颜色，再触觉确认材质。

直觉：**language 是连接不同模态的天然桥梁。你说"软的"，视觉能看到弹性形变，触觉能感受到 compliance，这俩通过"软"这个语义概念对齐了。**

### Force（低维力信号）
Force 数据是低维的（6维 wrench 或 n维 joint torque），跟 motion 有显式因果关系，可以直接编码进 observation space。或者用 FFT 转成 2D spectrogram [78] 来显式提取高频特征。或者用小 MLP [66, 77, 93]。

直觉：**Force 数据的低维性是它的优势。在 long-horizon、物理上复杂的任务里，GelSight 图像可能太冗余了，而 force 信号的紧凑性可能更有利于 reasoning。**

---

## 最大的 open question：到底需不需要显式的 force？

这篇 paper 最诚实的部分是 Discussion 章节。它承认：

1. **生理学角度**：触觉和 proprioception 都不是 strict 必需的。人没了触觉也能完成任务（虽然更笨拙），没了 proprioception 也能重新学（虽然得全程看手）。
2. **控制角度**：Impedance control 本身就是力控的隐式实现。很多 contact-rich task 可能只需要 inference 出合适的 mass/spring/damping 参数，剩下的交给 position control。
3. **机械设计角度**：Soft robotics 和 compliant mechanism 可以在硬件层面实现 implicit touch sensing [103, 104]。Bhatt et al. [103] 证明纯开环 + 好的机械设计就能做 in-hand manipulation。

但 paper 的立场是：**这些"替代方案"只是证明了人类感觉运动系统的冗余性，不应该被用来论证"vision alone 就够了"。** 对于真正 high-performance、fine-grained 的动态任务，显式的 force representation 仍然是必要的。

---

## 我的 take（作为 Karpathy 你可能会关心的）

这篇 paper 暴露了一个根本矛盾：

**Foundation models 的力量来自 scale——大规模、标准化、跨平台的数据。但 force/touch sensing 的本质是 hardware-specific、high-frequency、physics-coupled 的。你怎么把一个 ATI Mini 45 的六维 wrist F/T 信号和一个 GelSight 的 224x224 图像对齐？你怎么把 Franka Panda 的 7维 joint torque 和一个 magnetometer 的 15维 magnetic field 信号 token化到同一个 sequence 里？**

这比 vision + language 的对齐难多了。Vision 和 language 都是 discrete/semantic 的，可以用 contrastive learning 对齐。但 force 是 continuous/dynamic 的，它的语义在于"时间序列的动态变化"而不在于"某一时刻的值"。

几个可能的突破口：

1. **Handheld gripper + standardization**：UMI [98] 已经在 vision 领域证明了 handheld data collection 的威力。加上标准化的 F/T sensor（比如 CoinFT），可能形成第一个可 scale 的 tactile data collection pipeline。参考：https://arxiv.org/abs/2402.10329

2. **Language as bridge**：FuSe [80] 证明了 language 可以把 force/touch 锚定到 semantic space。如果每个 tactile trajectory 都有 language description，就可以用类似 CLIP 的方式做跨模态对齐。参考：https://arxiv.org/abs/2501.04693

3. **Frequency decomposition + tokenization**：Force 信号的高频特性意味着传统的 10-50Hz policy frequency 不够。可能需要类似 audio processing 的方式，把 force 信号分成不同频段的 tokens（就像人手有 FA-I 20-200Hz 和 FA-II up to 1500Hz）。Spectrogram [78] 是一个初步尝试。

4. **Compositional architecture**：$\pi_0$ [5], Helix [105], Gemini Robotics [6] 都在走 compositional 路线——低频大模型做 reasoning + skill selection，高频小模型做 low-level control。Force/touch 最自然的插入点就是 high-frequency low-level controller 那一层。低频大模型输出"意图+刚度参数"，高频 controller 做 reactive force regulation。参考：https://www.figure.ai/news/helix

5. **Sim2real for force**：Wang et al. [88] 的 Drake simulation 路线虽然现在效果有限，但如果 simulator 对摩擦和接触的建模足够好（可能需要 differentiable physics 或 particle-based simulation），sim data 可以解决 scaling 问题。参考：https://arxiv.org/abs/2402.02511

---

## 底线

这篇 paper 的核心 message 是：**robot foundation models 目前停在 "看得到" 但 "摸不着" 的阶段。Force 是 position 的二阶导数，携带了高频动态信息，对于 contact-rich manipulation 不可或缺。但 force/touch data 的异构性、高频率、硬件耦合性使得它无法简单地像 vision 一样被塞进 transformer。未来需要标准化传感器、language-grounded 多模态对齐、compositional 控制架构、以及可能的 simulation 辅助来打破这个僵局。**

现在这个领域有点像 2012 年的 computer vision——大家都知道方向对，但还没等到 ImageNet 和 ResNet 那样的 breakthrough。

---

Andrej, 这篇paper是一篇非常精彩的 literature survey，它系统性地梳理了当前 robotics foundation models 在迈向 "forceful"（具备力觉/触觉交互能力）过程中的核心挑战、现有方案以及未来演进路径。当前的 VLA (Vision-Language-Action) models，诸如 OpenVLA, RT-1, 乃至 Google 最新的 Gemini Robotics，虽然展现了惊人的 zero-shot manipulation 能力，但它们都有一个致命的盲点：纯视觉与位置控制。

从物理第一性原理来看，Force 是 position 的二阶导数（Newton's second law of motion $F = m \ddot{x}$）。这意味着 Force 蕴含了比 position 高得多的频域信息。在 stiff systems 中，微小的 position error 会导致巨大的 force error。仅依赖 position 的 imitation learning 既不 sample-efficient，也无法捕捉接触交互中的高频动态。

为了 build your intuition，我将从底层物理公式、生物学启发、data collection 范式、action space 设计，一直到 policy architecture 进行深度解构。

---

### 1. Biological Inspiration: Human Tactile & Proprioception Apparatus

要构建 forceful robot foundation models，最直觉的起点是理解人类的 sensing apparatus。人类的触觉系统依靠皮肤下的 mechanoreceptors，分为四类，它们在空间分辨率和带宽上形成了完美的互补：

*   **FA-I (Fast-Adapting Type I) / Meissner corpuscles**: 主要分布在指尖无毛区，响应 20-200Hz 的低频振动与动态皮肤形变。 receptive field 小且清晰，对于检测 texture 和 slip 极其关键。
*   **FA-II (Fast-Adapting Type II) / Pacinian corpuscles**: 分布全手，敏感于高达 1500Hz 的高频振动及压力突变。receptive field 大且模糊，能够检测远端振动，这对于 tool use 感知至关重要。
*   **SA-I (Slowly-Adapting Type I) / Merkel cell-neurite complexes**: 集中在指尖，检测 sustained pressure 和 fine spatial details。receptive field 小，是 form 和 texture discrimination 的核心。
*   **SA-II (Slowly-Adapting Type II) / Ruffini endings**: 检测 skin stretch 和 sustained pressure，receptive field 大，为 hand shape 和 finger position 的 proprioceptive feedback 提供信号。

此外，人类的 proprioception（本体感觉）依赖肌肉内的 muscle spindles（检测肌肉长度和拉伸速率）、Golgi tendon organs / GTOs（检测肌肉张力/force，超过阈值会抑制收缩以防止损伤）以及 joint receptors（检测关节角度）。

**Intuition Build**: 人类的高维度 dexterity 依赖于具有极宽频带和极高信息密度的多模态传感网络。机器人领域的 proprioception 通常指 joint encoders 和 joint torque sensors，而 tactile sensors 属于 exteroception。目前的机器人触觉传感器在 information density 和 shear force 测量上远不及人类水平。这直接导致了 robot learning 中 force 数据的严重异构性，阻碍了大规模 foundation model 的形成。

### 2. Force Control Mechanics: The Math Behind Interaction

在深入 policy learning 之前，我们需要明确 robot action space 中的数学约束。假设一个具有 $n$ 个关节的机械臂，其关节角为 $q \in \mathcal{R}^n$，末端执行器位姿为 $x \in \mathcal{R}^6$。

末端速度 $\dot{x}$ 与关节速度 $\dot{q}$ 通过 Jacobian matrix $J \in \mathcal{R}^{6 \times n}$ 联系：
$$ \dot{x} = J \dot{q} \quad \text{(Equation 1)} $$
其中，$J$ 是末端位姿对关节角的偏导矩阵。

如果我们要控制末端输出的力（即 spatial wrench，包含 3 个平动力和 3 个旋转力矩），所需的关节力矩 $\tau \in \mathcal{R}^n$ 可以通过 Jacobian 的转置计算：
$$ \boldsymbol{\tau} = J^T \boldsymbol{F} \quad \text{(Equation 2)} $$
这里 $F$ 是六维 spatial wrench。

**Impedance & Admittance Control**
由于很多机器人不支持直接控制 joint torques，Impedance control 提供了一种通过 position 控制来间接实现力控的方法。其公式如下：
$$ F = M \ddot{x} + D \dot{x} + K (x - x_d) \quad \text{(Equation 3)} $$
变量解释：
*   $F$: 末端执行器施加的 wrench。
*   $M, D, K$: 虚拟的 mass matrix, damping matrix, stiffness matrix。
*   $x, \dot{x}, \ddot{x}$: 当前的 pose, velocity, acceleration。
*   $x_d$: 期望的 pose。

**Intuition Build**: 你可以把 Impedance control 想象成在机器人的末端连接了一个虚拟的 spring-damper system。如果当前 pose $x$ 偏离了期望 pose $x_d$，虚拟弹簧就会产生一个回复力 $F$。如果忽略 $M$ 和 $D$ ($M=0, D=0$)，就退化为 compliance control（纯刚性控制）。
Admittance control 则是公式 (3) 的逆过程：已知期望施加的力 $F$，求解出需要的加速度 $\ddot{x}$，然后数值积分得到位移指令，发送给底层 position controller。

对于 policy learning 而言，如果 action space 是 position/velocity，就需要 inverse kinematics (基于公式1)；如果 action space 是 wrench，就需要 inverse dynamics (基于公式2) 或走 admittance control 的弯路。这直接影响了 learning 的难度和 robustness。

### 3. Data Collection Paradigms: The Embodiment Gap

这篇 paper 梳理了 25 篇使用 transformer/diffusion 架构的相关论文，涉及 64 个 manipulation experiments。在 sensor modalities 上（Fig 5），GelSight 类型的 visuotactile sensors 占据了 36% 的 plurality。GelSight 通过相机观察弹性聚合物的形变来测量 touch，提供了高分辨率的 contact deformation imaging。其他还包括 whole-arm joint torque (如 Franka Panda), Wrist F/T sensors, 甚至 finger audio sensing (surface microphones)。

在 data collection 方面，论文揭示了一个残酷的现实：76% 的研究使用少于 200 个 demonstrations (Fig 7)。数据采集方法分为以下几类：

1.  **Teleoperation (36%)**: 使用 VR headset, joystick 或 leader-follower 机械臂。embodiment gap 较大，且在 forceful tasks 中难以反映真实的交互动态。
2.  **Kinesthetic Teaching**: 人类直接抓着处于 free-drive (gravity compensation) 模式的机械臂进行示教。这种方法让 robot sensors 能直接捕捉人施加的 nuanced forces。例如 Hou et al. [78] 使用 UR5E 示教 vase cleaning；Zhao et al. [92] 使用 GelSight Wedge 进行低力矩的 capacitor insertion 示教。缺点是体力消耗大，容易损坏传感器。
3.  **Bilateral Teleoperation (Augmented)**: 引入 haptic feedback。将机器人感知的力镜像回传给人类操作者，形成闭环。这减轻了人的体力负担，但反馈信号经过了 processing，且硬件系统极其复杂、难以 scale。
4.  **Handheld Grippers**: 这是一个极具启发性的方向。例如 UMI (Universal Manipulation Interface) [98] 的变体，将夹爪从机械臂上彻底分离，只保留 finger-level force sensing 和 camera。人直接拿着这个夹爪完成任务。它捕捉了最纯粹的 end-effector force interaction，彻底绕过了 robot arm 的 embodiment gap，极大地提升了可扩展性。

**Intuition Build**: Foundation models 的核心是 scaling law。当前 tactile data 缺乏统一的 sensor hardware，各种 F/T sensors, GelSight, magnetometers 导致数据无法在异构平台上对齐。Handheld grippers (如 UMI 加上触觉) 可能是打破这个僵局的突破口，它将 data collection 从昂贵的 robot teleop 降维成了日常动作的录制。

### 4. Action Space & Explicit Force Control

统计显示（Fig 8），64% 的 learned policies 输出 Cartesian space position 指令。但是，explicit force control 展现出了压倒性的性能优势。

例如 Hou et al. [78] 的 **Adaptive Compliance Policy**：
它从示教数据中提取 wrist F/T 和 position 数据，通过后处理重构出 admittance controller 的输入（即 virtual target pose 和 stiffness matrix $K$）。其 heuristic 是：在力反馈方向上允许低 stiffness (高 compliance)，其他方向保持高 stiffness。他们用一个 1-second moving average filter 对 wrist wrenches 进行滤波，生成 future-contact-informed stiffness inputs，进而生成平滑的 contact-engaging virtual target trajectories。策略输出 virtual target 和 stiffness value。在 vase wiping 任务中，相比于使用均匀高/低 stiffness 的 ablation 版本，成功率提升了 81%。

又如 Wu et al. [89] 的 **TacDiffusion**：
策略直接输出 50-500Hz 的高频 target external wrenches，底层接 1000Hz 的 impedance controller。通过 dynamic filtering 进行频率插值。在 peg insertion (公差 <0.5mm) 任务上，平均耗时 <2s，成功率 >90%。而基于 position control 的同类 peg insertion 策略通常需要至少两倍的时间。

**Intuition Build**: 纯 position control 的策略即使以 force 为输入条件，它本质上依然是一个 model-free 的隐式 force-position controller。Explicit force control（输出 stiffness, wrench, 或 compliance 参数）有三个核心优势：
1.  **Performance**: 控制器运行在极高频率 (500Hz-1000Hz) 上，能够实现超人类反应速度的 reactive consistency。
2.  **Interpretability**: 低维的控制参数（如标量 compliance）易于监控和 debug。
3.  **Dimension Reduction**: 策略只需学习 "运动意图" 和 "刚度参数"，将高频的 force regulation 工作交给了经典控制论模块。这种分层架构与人类的 "大脑规划+脊髓反射" 极其相似。

### 5. Policy Learning Architectures & Representation

在架构选择上（Fig 9），52% 使用 **Diffusion policy** [11]，20% 使用 **Transformer**（多为 ACT [47]），16% 仍在用 **MLP**。

Diffusion policy 之所以流行，在于它能够优雅地处理 multi-modal action distributions。人类示教数据充满随机性和多种可行路径，传统的 MSE loss 会导致 mode collapse（将向左绕和向右绕平均成直接撞上的轨迹）。Diffusion 通过去噪过程学习 score function，能够完美捕捉这种 stochastic nature。

在 representation learning 层面：
*   **Visuotactile Representation**: GelSight 产生高维图像。Jones et al. [80] 提出的 **FuSe** 是目前唯一成功微调大型 robot foundation model (Octo [71]) 的 tactile policy。他们发现如果仅用 naive MSE loss，Octo 会过度依赖 pre-training 中的 vision 和 robot position 数据，忽略 touch 数据。为此，他们引入了两个多模态 loss，用 language (如 "pick up the squishy object") 作为 glue：
    1.  **Contrastive loss** ($L_{contrast}$): 最大化同一场景下不同 modalities 和 semantics 之间的 mutual information。
    2.  **Generative loss** ($L_{gen}$): 从 embedding 生成对应的 ground truth language，迫使模型理解 high-level semantics。
*   **Force Representation**: Force 数据是低维的且与 motion 存在显式的因果联系。可以直接喂入 observation space，或者通过 fast Fourier transform (FFT) 转换为 2D spectrogram [78] 来显式提取高频特征，也可用简单的 MLP 编码 [66, 77, 93]。Force 的低维特性使其在 long-horizon, physically intricate tasks 中极具潜力，而 visuotactile data 在这些场景下可能过于冗余。

**Intuition Build**: LLM 时代的基础是 tokenization 和统一的 next-token prediction。Robotics foundation model 要整合 force，难点在于 force 的时间序列特征极强，且与底层 controller 的频率耦合严重。FuSe 的多模态 loss 设计展示了通过 language grounding 将高频 force 信号锚定到语义空间的巨大潜力。

### 6. Future Directions: Tactile Robot Foundation Models

Paper 最后探讨了一个深刻的问题：Explicit representations for force 究竟是否必需？
从机械设计角度看，通过 soft robotics 和 compliant mechanisms 可以实现 **implicit touch sensing** [104]，无需触觉传感器也能完成复杂的 in-hand manipulation。然而，这种开环鲁棒性在 fine-grained manipulation 面前依然会崩溃。

未来的 robot foundation models（如 Physical Intelligence 的 $\pi_0$, Figure AI 的 Helix, Google 的 Gemini Robotics）都在走向 **compositional control**：低频大模型负责 high-level reasoning 和 skill decomposition，高频小模型/policy 负责底层执行。Force 和 tactile sensing 必须无缝融入这个频谱。为了实现这一目标，我们需要：
1.  跨越异构硬件的 force/touch tokenization 方法。
2.  大规模自动化的 tactile data collection pipeline（摆脱纯 human teleop 依赖，如 sim2real 结合 LLM-guided controllers [67]）。
3.  结合大模型推理的 hybrid force-position policies，既能处理 "倒液体" 这种连续力控制，也能处理 "插拔销钉" 这种瞬态高频力反馈。

### References & Further Reading

为了方便你深入阅读，这里列出上述讨论中提及的关键论文及其官方链接：

1.  **Diffusion Policy**: *Visuomotor Policy Learning via Action Diffusion* by Chi et al. 
    https://arxiv.org/abs/2311.01248
2.  **ACT (Action Chunking Transformer)**: *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* by Zhao et al. 
    https://arxiv.org/abs/2304.13705
3.  **Adaptive Compliance Policy**: *Learning Approximate Compliance for Diffusion Guided Control* by Hou et al. 
    https://arxiv.org/abs/2410.09309
4.  **TacDiffusion**: *Force-domain Diffusion Policy for Precise Tactile Manipulation* by Wu et al. 
    https://arxiv.org/abs/2409.11047
5.  **FuSe & Octo Fine-tuning**: *Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding* by Jones et al. 
    https://arxiv.org/abs/2501.04693
6.  **Octo Model**: *An Open-Source Generalist Robot Policy* by Octo Model Team. 
    https://arxiv.org/abs/2405.12213
7.  **OpenVLA**: *An Open-Source Vision-Language-Action Model* by Kim et al. 
    https://arxiv.org/abs/2406.09246
8.  **AnySkin (Magnetometer Tactile)**: *Plug-and-play skin sensing for robotic touch* by Bhirangi et al. 
    https://arxiv.org/abs/2409.08276
9.  **UMI (Universal Manipulation Interface)**: *In-the-wild robot teaching without in-the-wild robots* by Chi et al. 
    https://arxiv.org/abs/2402.10329
10. **Gemini Robotics**: *Bringing AI into the physical world* by Google DeepMind. 
    https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf
11. **Just Add Force**: *Just add force for delicate robot policies* by Xie et al. 
    https://openreview.net/pdf?id=GSEs7MCnoi
12. **FoAR**: *Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation* by He et al. 
    https://arxiv.org/abs/2411.15753
13. **GelSight**: *High-resolution robot tactile sensors for estimating geometry and force* by Yuan et al. 
    https://arxiv.org/abs/1810.08272

这篇 survey 揭示了 robotics 向前演进的关键拼图。纯粹的 V++ (Vision + Position) 无法跨越动态交互的鸿沟，引入 Force 这个高频的二阶导数信号，结合 diffusion 的多模态轨迹建模能力与经典阻抗控制的高频反应能力，将是构建下一代 Generalist Humanoid Robots 的核心路径。
