---
source_pdf: FLOW EQUIVARIANT WORLD MODELS.pdf
paper_sha256: 5074620500fa1e74b52db44260c0d4d26b98a761803975b4f26456587c2a1e7c
processed_at: '2026-08-04T09:30:38-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最接地气的人话来说，这篇 paper 戳中了当前所有 video world models（包括 Sora 这类巨无霸）最致命的软肋：**它们全是金鱼脑，并且完全没有真正的空间认知。**

Imagine 一下现在的 video diffusion model 怎么工作：它就像一个看着前方墙壁的监控摄像头。如果一只狗从画面左边跑到右边，然后跑出了视野，接着你把这个摄像头的镜头转向天花板，过了十秒钟再转回来。Sora 这类模型会看到什么？它大概率会 hallucinate 出一只全新的猫，或者把刚才那只狗随机画在某个位置。为什么？因为狗的像素离开 attention window 后，信息就被丢弃了。Transformer 架构本身没有任何机制去维持一个“世界在你身后继续运转”的概念。

FloWM (Flow Equivariant World Models) 这篇 paper 就是为了彻底解决这个问题。它给 neural network 装上了一个符合物理几何规律的“大脑海马体”。

---

### 1. 核心直觉：Memory 必须长在“世界”上，并且要跟着物理定律动

人类的大脑是怎么处理这个问题的？你的大脑里有一张关于周围环境的绝对坐标地图。
*   **Self-Motion (你自己动了)**: 当你转头时，你的视野转了，但是你大脑里的那张地图没有转。地图里的桌子还是在那儿。这就叫 **Self-Motion Equivariance**。Agent 做了一个动作 $a_t$（比如向左转），那么 latent map 必须向右做逆向旋转 $T_{a_t}^{-1}$，从而保持 latent map 和绝对世界对齐。
*   **External Motion (外部物体动了)**: 狗在跑，哪怕你背对着狗，你大脑里代表狗的那部分特征也在根据狗的速度平滑移动。这就叫 **Flow Equivariance**。

为了实现这两点，作者用了一个极其优雅的数学框架：**One-parameter Lie Group Flows**。

---

### 2. 神来之笔：Velocity Channels (速度通道)

FloWM 最惊艳的设计就是这个 Velocity Channels。
传统的 RNN 只有一个 hidden state $h_t$。FloWM 的 hidden state 切成了很多层，每一层对应一个预设的速度向量 $\nu$（比如往左跑、往右跑、静止）。

想象你有 25 个平行的宇宙（25 个 velocity channels），每个宇宙里的物体运动速度假设不同。
*   如果现实世界里，狗以速度 $\hat{\nu}$ 往右跑。
*   在“速度也是 $\hat{\nu}$ 往右跑”的那个 channel 里，狗的特征是相对静止的，因此它会被完美地保留在那个 channel 里，并且随着时间推移跟着地图一起平移。
*   在“速度是往左跑”的 channel 里，狗的特征会以极快的相对速度飘走，迅速模糊消散。

模型不需要去“猜”狗往哪跑，它只是把当前看到的画面 broadcast 到所有的 velocity channels 里。由于 equivariance 的数学性质，真实物体的特征会自动在匹配其速度的 channel 里稳定存在并流动。这简直是数学上的魔法。

---

### 3. 公式拆解：数学与物理直觉的完美统一

看看这个 Simple Recurrent FloWM 的核心公式 (Eq 8)：

$$
h_{t+1}(\nu) = \psi_1(\nu - a_t) \cdot \sigma\big(\mathcal{W} \star h_t(\nu) + \mathrm{pad}(\mathcal{U} \star f_t)\big)
$$

我们逐个拆解，你会发现每个符号都在描述极其具体的物理动作：

*   $h_t(\nu)$: 当前时刻的 hidden state。注意这个下标或隐式索引 $\nu$，它表示这是在“速度假设为 $\nu$”的那个 channel 里的地图状态。
*   $f_t$: 当前时刻 agent 看到的局部图像。
*   $\mathcal{U} \star f_t$: Encoder 操作，用卷积核 $\mathcal{U}$ 提取图像特征。
*   $\mathrm{pad}(\cdot)$: 因为 agent 只能看到局部视野，所以把提取出的局部特征用 0 填充到整个世界地图的尺寸，准备写进全局地图。
*   $\mathcal{W} \star h_t(\nu)$: 对上一时刻的全局地图做卷积处理，更新状态。
*   $\sigma(\dots)$: 非线性激活函数，把新观察到的信息和旧的记忆融合起来。
*   **最核心的 $\psi_1(\nu - a_t)$**: 这就是物理引擎！$\psi$ 是 Lie group 的 flow 操作，$\nu$ 是物体自身的速度，$a_t$ 是 agent 自身的动作速度。两者的差 $\nu - a_t$ 就是物体在 agent 参考系下的**相对速度**。这个 flow 操作把刚刚更新好的地图特征，按照这个相对速度在空间上平移一格。

这一个公式直接统一了 self-motion 和 external dynamics。时间流逝的本质，在这个框架下就变成了群作用在 latent space 上的不断复合。

---

### 4. 架构怎么跑的：从 2D 卷积到 3D ViT

**对于 2D 的 MNIST World**:
就是上面那个公式。Agent 在 $50 \times 50$ 的世界里移动，视野只有 $32 \times 32$。网络把当前看到的 $32 \times 32$ 写进 hidden state 的对应区域，然后整个 hidden state 按照各个 velocity channel 的速度 $\nu - a_t$ 整体平移。预测未来时，从 hidden state 中心 crop 出 $32 \times 32$，在 velocity 维度做 max-pool（提取最显著的运动特征），再 decode 成图像。

**对于 3D 的 Block World**:
由于输入是第一人称的 RGB 图像，不能直接简单卷积。作者用了一个 Transformer-based 架构。
1.  **Hidden state**: 是一个 2D 的 top-down 的 token map，尺寸为 $32 \times 32$。Agent 永远在地图正中心。
2.  **Encoder ($\mathrm{E}_\theta$)**: 一个 ViT。它接收当前的图像 patch tokens，同时从 latent map 中根据当前的视锥角裁剪出对应的 tokens。两者 concat 后过 self-attention，相当于把当前的视觉信息反投影到 top-down 地图上。
3.  **Action Transform ($T_{a_t}^{-1}$)**: 如果 agent 做了 turn left，那么整个 $32 \times 32$ 的 latent map 就向右旋转 90 度。这保证了地图始终与世界对齐。
4.  **Internal Flow ($\psi_1(\nu)$)**: 每个 velocity channel 的地图按照 $\nu$ 各自平移。
5.  **Decoder ($\mathrm{D}_\theta$)**: 需要预测下一帧时，从更新后的 latent map 中提取 FoV 区域的 tokens，通过 cross-attention 还原成第一人称的 RGB 图像。

---

### 5. 实验结果最震撼的点：Length Generalization (长度外推)

现在的 generative model 通常训练时见过 20 帧的未来，测试时超过 20 帧误差就指数级爆炸。FloWM 的实验结果极其震撼：

在 3D Dynamic Block World 数据集上：
*   训练时只预测 70 帧。
*   测试时让它预测 210 帧（训练长度的 3 倍）。
*   DFoT (Diffusion Forcing Transformer) 在 210 帧时 MSE 飙升到 0.02，画面全乱。
*   **FloWM 在 210 帧时的 MSE 依然只有 0.0015**，几乎没有任何性能衰减！

为什么能 length generalization？因为群作用是可结合且确定的。时间再长，也只是同样的 flow 操作 $\psi$ 在不断复合。网络并没有在“学习时间流逝”，它只是在执行几何变换。Inductive bias 把物理定律硬编码进了网络结构里。

---

### 6. 更深层的 Intuition 与联想

这篇 paper 透露出几个极其深刻的信号，作为做 AI 架构的人，你一定会产生共鸣：

**6.1 Neuroscience 的完美印证**
作者提到，哺乳动物的视觉皮层会在 agent 自身运动时产生预测信号，并且海马体中的 place cells 和 grid cells 的放电相位会随着 agent 移动发生 equivariant shift。FloWM 的 latent map 简直就是人工版的 grid cells。这暗示着，生物大脑为了在部分可观测的动态环境中生存，进化出的必然是某种满足 Lie group equivariance 的表征结构。

**6.2 对 "Scale is all you need" 信仰的挑战**
OpenAI 等公司试图去掉所有 inductive bias，纯靠 scale 压缩世界规律。FloWM 证明了，只要把对称性写进架构里，区区 10M 参数的模型（DFoT 有 95M 参数）在长程动态一致性上能秒杀百倍于己的 Diffusion Transformer。未来的 AGI world model，极大概率是 Scale + Geometric Priors 的结合。

**6.3 与 SLAM 和 Robotics 的合流**
机器人领域搞了几十年的 SLAM，维护 explicit 的 occupancy grid。FloWM 本质上是做了一个 deep learning 版本的 Dynamic SLAM。未来如果把这个 equivariant latent map 挂载到机器人控制环里，机器人的 planning 和 exploration 效率将会产生质的飞跃。

**6.4 扩展到 JEPA 与 Latent Dynamics**
目前 FloWM 还在做 pixel-level 的重建（用 MSE loss），这其实浪费了算力。未来的终极形态，是把 FloWM 的 equivariant recurrence 拿来作为 JEPA 的 predictor backbone。在 latent space 里做 flow equivariant dynamics prediction，彻底摆脱像素生成的包袱，专注于抽象状态的推演。

### References & Further Reading

如果想顺着这个思路深挖，推荐几篇极其相关的工作：

1.  **FloWM 的理论基础** - T. Anderson Keller 的 Flow Equivariance:
    https://arxiv.org/abs/2507.14793
2.  **Grid Cells 涌现于 Equivariance 约束** - Dorrell et al.:
    https://arxiv.org/abs/2209.15563
3.  **经典 Spatial Memory 架构 Neural Map** (FloWM 的前序思想):
    https://arxiv.org/abs/1702.08360
4.  **Baseline 对比对象 History-Guided Diffusion Forcing**:
    https://arxiv.org/abs/2502.06764
5.  **Memory-augmented 3D World Models** (WORLDMEM):
    https://arxiv.org/abs/2504.12369

这篇 paper 堪称架构设计里的艺术品，它告诉你：在这个领域里，数学美感与 SOTA performance 完全可以合二为一。

---

这篇 paper 提出了一个极其优雅且具有深刻理论意义的框架：**Flow Equivariant World Models (FloWM)**。作为做 embodied AI 和 world model 的人，你一定会非常欣赏这篇工作里蕴含的数学结构美感和 neuroscientific inspiration。当前主流的 video diffusion models (如 Sora, DFoT) 本质上是把时间维度当作 spatial patch 来暴力扩展 attention window，这在 partially observed environments（视野受限、有遮挡）下注定会崩溃，因为一旦 object 滑出 attention window，信息就彻底丢失了，模型必定开始 hallucination。FloWM 则从 Lie group 和 flow equivariance 的角度出发，给 world model 的 memory 赋予了几何与动力学的归纳偏置。

以下我为你详细拆解这篇 paper 的核心技术细节、架构设计、公式推导以及更广泛的联想。

---

### 1. Core Intuition: 从 "Symphony of Flows" 到 Equivariant Memory

要理解 FloWM，我们先建立直觉。想象你在一个房间里，一条狗从你面前跑过，跑出了你的视野，此时你转过身去看墙壁，过了几秒钟你再转回来。你的大脑知道那条狗大概率还在某个位置按照之前的轨迹继续跑。这意味着你的大脑里维持了一个 **allocentric (世界中心坐标系) 的 dynamic latent map**，这个 map 满足两个对称性：
1.  **Self-Motion Symmetry**: 当你的头部转动时，这个 map 在你大脑中的表示会做逆旋转（保持世界不动，视角在动）。
2.  **External Motion Symmetry**: 狗在移动时，map 中代表狗的特征会按照狗的速度平滑流动，哪怕你当时没在看它。

FloWM 将这两类运动统一为 **one-parameter Lie group flows**，并强制 neural network 的 hidden state 严格 equivariant 于这些 flow。

---

### 2. Mathematical Formulation: Flow Equivariance 拆解

Equivariance 的核心意思是：输入经过变换 $g$ 后输入网络，等价于把原始输入通过网络后再输出做 $g$ 变换，即 $\phi(g \cdot f) = g \cdot \phi(f)$。这保证了网络内部的特征表示与外部物理空间的几何结构同构。

#### 2.1 Flow 与 Lie Group
在 FloWM 中，运动被定义为时间参数化的 Lie group flow $\psi_t(\nu) \in G$。
*   $\nu \in \mathfrak{g}$: Lie algebra 的元素，对应物理世界中的**速度向量**（如 2D 平移速度 $(v_x, v_y)$）。
*   $t \in \mathbb{R}$: 时间步长。
*   $\psi_t(\nu)$: 由速度 $\nu$ 积分出来的 Lie group 元素（即 $t$ 时间后的位移量）。
*   公式 $\psi_t(\nu) \cdot g_0 = g_t$ 表示初始 group element $g_0$ 经过流动后到达 $g_t$。

#### 2.2 Generalized Flow Equivariant Recurrence (核心架构公式)
普通的 RNN 是 $h_{t+1} = \sigma(W h_t + U f_t)$，没有任何几何结构。FloWM 提出了 Generalized Flow Equivariant Recurrence (Eq 5)：

$$
h_{t+1}(\nu) = \psi_1(\nu) \cdot \mathrm{U}_\theta \big[ h_t(\nu); E_\theta[f_t; h_t](\nu) \big]
$$

*   $h_t(\nu) \in \mathbb{R}^{|V| \times C_{hid} \times H \times W}$: t 时刻的 hidden state。注意这个上标/隐式索引 $\nu$，它表示 **velocity channel**。网络不只是有一个 hidden state，而是有 $|V|$ 个，每个对应一个离散的速度假设。
*   $E_\theta[f_t; h_t](\nu)$: Encoder 函数。它将当前的观测 $f_t$（通常是 image patch）和先前的 hidden state $h_t$ 作为输入。关键在于它必须满足 **trivial lift** 条件，即将输入特征均匀地广播到所有的 velocity channel $\nu$ 上。
*   $\mathrm{U}_\theta[h_t(\nu); o_t(\nu)]$: Update 函数。负责将 encoded observation 融合到 hidden state 中。它必须对 spatial transformations 保持 equivariant。
*   $\psi_1(\nu)$: 应用在 $\mathrm{U}_\theta$ 外部的 flow operator。它将更新后的 hidden state 按照其对应的速度 $\nu$ 在空间上平移一个 step。

**这里最绝妙的数学性质是 (Eq 4) 中的 velocity permutation**：
当输入序列被速度为 $\hat{\nu}$ 的 flow 作用时，网络内部的各个 velocity channel 会发生排列，并且跟着流动：
$$
h_t[\psi(\hat{\nu}) \cdot f](\nu) = \psi_{t-1}(\hat{\nu}) \cdot h_t[f](\nu - \hat{\nu})
$$
直觉解释：如果现实世界中有一个物体以速度 $\hat{\nu}$ 移动，那么在代表"绝对速度 $\nu$"的 channel 里，这个物体的特征会随着时间慢慢飘走；但是在代表"相对速度 $\nu - \hat{\nu}$"的 channel 里，这个物体是静止的，它的特征会被完美地保留并随着输入流一起平移。这就是 equivariance 带来的信息保持能力。

#### 2.3 Self-Motion Equivariance (Eq 7)
为了处理 agent 自身的移动，FloWM 引入了动作变量 $a_t$。Agent 移动会导致整个世界的相对移动。

$$
h_{t+1}(\nu) = T_{a_t}^{-1} \cdot \psi_1(\nu) \cdot \mathrm{U}_\theta[h_t(\nu); E_\theta[f_t; h_t](\nu)]
$$

*   $a_t$: Agent 的 action（如 2D 中的相对位移，或 3D 中的左转/右转/前进）。
*   $T_{a_t}$: Action 在 latent space 中的 group representation。
*   $T_{a_t}^{-1}$: 逆变换。如果 agent 向左转，那么 latent map 必须向右旋转，从而维持 latent map 与绝对世界坐标系的对齐。

在 2D MNIST World 中，action $a_t$ 就是平移，公式可以合并简化为 (Eq 8)：
$$
h_{t+1}(\nu) = \psi_1(\nu - a_t) \cdot \sigma\big(\mathcal{W} \star h_t(\nu) + \mathrm{pad}(\mathcal{U} \star f_t)\big)
$$
这里 $\psi_1(\nu - a_t)$ 完美统一了 self-motion 和 external flow。物体的相对速度变成了 $\nu - a_t$。这个数学形式简直是艺术品。

---

### 3. 架构设计解析

Paper 中给出了两种具体的 instantiation：

#### 3.1 Simple Recurrent FloWM (For 2D MNIST World)
*   **Dataset**: 2D 黑底画布，多个 MNIST 数字以随机恒定速度移动，agent 只能通过 $32 \times 32$ 的窗口观察 $50 \times 50$ 的世界。
*   **Architecture**: 纯卷积 RNN。$\mathcal{W}$ 和 $\mathcal{U}$ 是 $3 \times 3$ 卷积核。
*   **Partial Observability Handling**:
    *   **Write-in**: Encoder 输出 $\mathcal{U} \star f_t$ 被 `pad` 到 $50 \times 50$ 的世界尺寸，并加到 agent 当前 FoV 对应的位置上。
    *   **Read-out**: 从 hidden state 中心 crop 出 $32 \times 32$ 窗口，然后在 velocity channel 维度做 `max-pool`（提取最显著的运动特征），最后通过 decoder $g_\theta$ 预测下一帧。

#### 3.2 Transformer-Based FloWM (For 3D Dynamic Block World)
*   **Dataset**: Miniworld 3D 环境，带颜色的 block 以随机速度移动并在墙壁反弹。Agent 视角是第一人称 RGB 图像，动作是离散的（turn left/right, forward, nothing）。
*   **Hidden State**: $h_t$ 是一个 2D top-down 的 egocentric token map，形状为 $\mathbb{R}^{|V| \times 256 \times 32 \times 32}$。注意，虽然是 3D 环境，但 memory map 是 2D top-down 的（类似 Ha & Schmidhuber 的 World Models 或 SLAM 中的 occupancy grid）。因为 agent 始终在 map 中心，所以当 agent 做动作时，直接对 map 做 `roll` (平移) 或 90 度旋转即可实现 $T_{a_t}^{-1}$。
*   **Encoder $\mathrm{E}_\theta$**: 是一个 ViT。它接收两个输入：
    1.  `patchify(f_t)`: 当前视角的 image patch tokens。
    2.  `FoV(h_t)`: 从 latent map 中根据当前视角的三角视锥裁剪出的 tokens。
    将它们 concat 后过 self-attention。
*   **Update $\mathrm{U}_\theta$**: Gated update (Eq 9)。
    $$ \mathrm{U}_\theta[h_t; o_t]^{(x,y)} = (1 - \alpha) * h_t^{(x,y)} + \alpha * o_t^{(x,y)} $$
    其中 $\alpha = \sigma(\mathbf{W} \text{concat}[h_t^{(x,y)}; o_t^{(x,y)}])$。
*   **Decoder $\mathrm{D}_\theta$**: Cross-attention。Query 是可学习的 image patch tokens，Key/Value 是更新后的 `FoV(h_{t+1})`，输出预测的下一帧图像。

---

### 4. 实验数据与长度外推

实验结果证明了 FloWM 在 partially observable dynamic environment 中的压倒性优势。

**Table 1 (2D MNIST World)**:
*   FloWM 在 20 步训练长度下达到 MSE 0.0005，PSNR 32.99。
*   **长度外推 (150 steps)**: FloWM 的 MSE 仅为 0.0018，几乎没有任何性能衰减。而 DFoT (Diffusion Forcing Transformer) 和 DFoT-SSM (带 State Space Model 的 memory) 的 MSE 在 0.16 左右，甚至不如 All-Black baseline。
*   **Ablation**: 去掉 VC (Velocity Channels) 会导致 150 步时 MSE 上升到 0.0334。去掉 SME (Self-Motion Equivariance) 则模型彻底崩溃 (MSE 0.12)。

**Table 2 (3D Dynamic Block World)**:
*   在 Textured 3D Block World 中 (见 Appendix Table 4)，FloWM 在 210 步预测时 PSNR 达到 30.33，而 DFoT-SSM 只有 19.15。
*   Figure 6 展示了 Error vs Timestep 曲线，baselines 的 error 随时间呈发散趋势，而 FloWM 误差几乎是一条水平的直线。

**为什么能 length generalization？** 因为 Lie group 的代数性质保证了只要 $a_t$ 和 $\nu$ 是恒定的，hidden state 的 trajectory 就是一个精确的群作用轨道，网络参数不需要学习 "如何随时间平移"，时间再长也只是同样的 group action 重复应用。

---

### 5. 更广阔的联想与 Intuition Building

这篇 paper 让人产生非常多深层的联想：

#### 5.1 Neuroscientific Connections: Grid Cells & Place Cells
Paper 提到了 Dorrell et al. (2023) 的工作，证明 grid-like activations 是通过 enforcing equivariant responses 自动涌现的。FloWM 的 $h_t(\nu)$ 某种程度上类似于 entorhinal cortex 中的 grid cells，它们对位置和速度敏感，并且维持一个 allocentric 的 metric map。当 agent 移动时，place cell 的 firing phase 会发生 equivariant shift，这与 $T_{a_t}^{-1}$ 作用于 latent map 的数学过程惊人地一致。

#### 5.2 Classical SLAM vs Neural SLAM
传统的 SLAM（如 ORB-SLAM）维护一个 explicit 的 occupancy grid，用 Bayesian filter 更新。FloWM 可以被视为一种 **Neural SLAM**，但它不仅能记忆静态地图，还能通过 velocity channels 预测未观测部分的 dynamic evolution。这与 Robotics 中的 Kalman Filter 追踪动态目标非常类似，只不过这里是在 latent space 中用 equivariant convolutions/attention 实现的。

#### 5.3 Inductive Bias vs Scale (Connection to Sora/Genie)
现在业界 (OpenAI, Google) 的风气是去掉所有的 inductive bias，纯靠 transformer 架构和海量数据暴力压缩世界动态。Sora 和 Genie 证明了 scale 可以做惊人的事情，但它们在需要长程一致性和部分可观测逻辑推演时（如转个圈看到之前的东西依然在动）依然表现挣扎。FloWM 走了一条相反的路：通过注入极度硬核的 Lie group inductive bias，在极小的数据量（180k 视频）和算力下，达到了完美的 long-horizon dynamic consistency。这暗示了构建通用 World Model 的另一条可行路径：**Scale + Geometric Priors**。

#### 5.4 JEPA 与 Latent Dynamics
Yann LeCun 极力推崇的 JEPA 架构是在 latent space 中做预测，避免 pixel-level reconstruction 的信息浪费。FloWM 目前的 loss 还是 pixel-level 的 MSE，但作者在 Future Work 中明确提到可以将 FloWM 的 equivariant latent memory 作为 JEPA 的 backbone。想象一下，如果将 $h_t$ 作为 JEPA 的 predictor state，所有的 flow equivariance 都在 latent space 中发生，这将是一个极其强大的 predictive world model。

#### 5.5 State Space Models (SSMs) 与 Recurrent Convolution
FloWM 面临的一个质疑是 recurrent 结构无法 parallelize over sequence length。虽然 Mamba/S4 等基于 associative scan 的 SSMs 解决了长序列训练问题，但它们缺乏空间结构。值得注意的是，FloWM 的 recurrence $\psi_1(\nu) \cdot h_t$ 在离散网格上本质就是一个 shift-convolution。如果将其扩展到 continuous velocity field 并写成 linear recurrence，可能可以用类似于 S4 的 kernel parameterization 来并行化。

#### 5.6 OOD Hallucination 与 Memory Bank
目前 3D reconstruction 领域的 memory-augmented diffusion (如 WORLDMEM) 是用过去视角的图像做 retrieval。这种基于 appearance retrieval 的机制无法处理 "视角未覆盖但物体已经移动" 的情况。FloWM 强调 world-centric state，memory 不存图像，只存 latent map，并让 latent map 随着 physics laws 动起来。

#### 5.7 Non-Rigid Dynamics 扩展
Paper 局限在于假设了 rigid motion 和已知速度离散集合 $V$。如果遇到非刚性形变（如人体运动），或者连续的复杂动力学，离散的 Lie group flow 会失效。一个直觉的 hallucination 是：能否引入 **Equivariant Neural ODE**，或者把 $\psi_1(\nu)$ 换成 latent space 中的 learned diffeomorphic flow field，并通过算子学习来隐式满足 equivariance？另外，semantic actions (如 "pick up") 属于离散群作用，可以构建 Hierarchical Lie Groups 来处理。

### References & Further Reading

如果你想顺着这个方向深挖，强烈推荐看以下链接：

1.  **T. Anderson Keller 的前序工作 (Flow Equivariance 创始)**:
    *   *Flow Equivariant Recurrent Neural Networks*: https://arxiv.org/abs/2507.14793
2.  **Diffusion Forcing Baselines (DFoT)**:
    *   *Diffusion Forcing: Next-token prediction meets full-sequence diffusion*: https://arxiv.org/abs/2407.01392
    *   *History-Guided Video Diffusion*: https://arxiv.org/abs/2502.06764
3.  **Equivariance 与 Grid Cells 联系**:
    *   *Actionable neural representations: Grid cells from minimal constraints*: https://arxiv.org/abs/2209.15563
4.  **Memory-Augmented World Models 对比**:
    *   *WorldMem: Long-term consistent world simulation with memory*: https://arxiv.org/abs/2504.12369
    *   *Learning 3D persistent embodied world models*: https://arxiv.org/abs/2505.05495
5.  **SLAM 启发的 Neural Mapping**:
    *   *Neural Map: Structured memory for deep reinforcement learning*: https://arxiv.org/abs/1702.08360

FloWM 的贡献绝不局限于一个简单的 video prediction benchmark 刷榜。它深刻地指出了：**Current generative world models lack a structured memory that respects the symmetries of the physical world**。把 action 和 dynamics 统一在 group theory 框架下进行 equivariant recurrence，这不仅是一个数学技巧，更触及了 embodied intelligence 的本质。构建具备 generalization 和 reasoning 能力的 world model，必然需要这种将时间、空间与动作编织在一起的代数结构。
