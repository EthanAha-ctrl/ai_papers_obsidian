---
source_pdf: VisionLanguageActionModelsinRoboticManipulation.pdf
paper_sha256: 881935edbd0c6bbc66f6dd9145f5e8cee2b29abc62ef7fca66d07f8d54770f5b
processed_at: '2026-08-13T01:49:18-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我们抛开那些学术套话，用最直白的方式聊聊VLA（Vision Language Action）到底在干什么，以及为什么这篇review里提到的那些设计选择其实非常巧妙。

---

## 1. VLA的根本动机：给LLM装上一个物理身体

传统的机器人控制特别像是在写一堆复杂的`if-else`规则。你有一个视觉模块负责找物体坐标，一个规划模块负责算路径，一个控制模块负责发关节力矩。这三个模块互相不说话，全靠人类工程师在中间定接口。只要光线变了一下，或者桌子上的杯子换了个位置，整个系统就可能崩溃。

VLA的核心直觉其实非常简单：既然Transformer在处理文本（GPT）和图像时已经展现出了惊人的泛化能力，为什么我们不直接把机器人的“眼睛看到的画面”、“耳朵听到的指令”和“身体要做的动作”全部塞进一个巨大的Transformer里，让模型自己从数据里学怎么把它们对应起来？

这就是End-to-End learning在机器人上的终极形态。模型不再需要人类去定义“什么是抓取”，它只看几百万次人类演示或者仿真数据，自己就把语言里的“grab the apple”、视觉里的“红色圆圈”和机械臂的“关节角度序列”映射到了同一个latent space里。

参考: [RT-1](https://arxiv.org/abs/2212.06817), [Open X-Embodiment](https://arxiv.org/abs/2310.08864)

---

## 2. 架构的Intuition：为什么要这么拼模块

这篇paper里的Figure 7画了典型的VLA架构，其实就是三路输入汇合到一个Backbone，最后接一个Action Decoder。我们拆解一下背后的工程直觉。

### 2.1 Vision Encoder的纠结：Semantic vs Spatial
机器人要看图，通常用ViT。但在选ViT的时候有两个流派：
- **CLIP / SigLIP路线**：它们是用图文对比学习训练出来的。给一张图，它输出的是“这是一个苹果，不是杯子”这种high-level semantic。这对于理解指令很有用，因为指令是文字，天然和语义对齐。
- **DINOv2路线**：它是self-supervised训练的，特别擅长抓取图像的local细节和深度信息。机器人要抓杯子把柄，它需要知道把柄在像素里的确切位置，DINOv2这种dense feature就特别重要。

所以你看OpenVLA或者HybridVLA，它们干脆把SigLIP和DINOv2的token拼在一起喂给模型。因为机器人既需要懂“这是什么”，又需要懂“具体在哪”，缺了任何一个任务都会搞砸。

### 2.2 State Encoder：为什么机器人要知道自己的关节在哪
这点很多搞纯CV的人容易忽略。机器人不是一个漂浮在空中的相机，它是一个有物理结构的机械臂。如果不把当前的joint angle或者gripper status告诉模型，模型规划出来的轨迹可能会跟自己的底座撞上。

把proprioceptive state通过一个小MLP变成几个token塞进去，本质上是在让模型Implicitly学习正向运动学。模型在生成下一步动作前，必须知道“我现在在这个姿态，我还能往哪伸”。

### 2.3 Attention机制：怎么把文字和画面绑起来
Self-Attention的公式 $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$ 听起来很玄乎，其实直觉上就是个“软查找”。

- $Q$ (Query) 是当前token在找的信息。比如文字token“苹果”发出的Query是“我在画面里哪呢？”。
- $K$ (Key) 是每个token能提供的标签。画面里某个红色圆形patch发出的Key是“我是一个红色的圆”。
- $V$ (Value) 是实际传递的信息。

$QK^T$ 一乘，“苹果”这个token和画面里那个红色patch的相似度分数就爆表了。再经过 $softmax$ 归一化，最后乘以 $V$，画面里那个红色patch的特征就流进了“苹果”这个token里。这样一通操作，语言和视觉就在同一个embedding space里对齐了。除以 $\sqrt{d_k}$ 只是为了防止点积太大导致softmax梯度过平。

参考: [Attention is All You Need](https://arxiv.org/abs/1706.03762), [DINOv2](https://arxiv.org/abs/2304.07193)

---

## 3. Action Decoder：为什么Diffusion干掉了自回归

这篇review里反复提到Diffusion Policy和Flow Matching。为什么不用传统的直接回归或者像GPT那样的自回归生成action token？这是VLA里最核心的工程突破之一。

### 3.1 MSE平均化的灾难
假设你要让机器人去抓一个杯子。桌上有杯子，你可以从左边抓，也可以从右边抓，甚至可以从上面抓。这些都是完全合理的action trajectory。

如果你用最简单的MSE去训练网络，网络为了最小化loss，会把从左抓、从右抓、从上抓的轨迹全部平均起来。结果就是它输出一个直接往前捅的中间值，这根本不是一个可行的动作。Action distribution是高度多峰的。

### 3.2 Diffusion怎么解决的
Diffusion Policy把生成动作变成了去噪过程。公式描述的是一个前向加噪过程：
$q(a_t | a_0) = \mathcal{N}(a_t; \sqrt{\bar{\alpha}_t} a_0, (1 - \bar{\alpha}_t) I)$

- $a_0$ 是干净的、真实的人类演示动作。
- $a_t$ 是加了 $t$ 步高斯噪声后的动作。
- $\bar{\alpha}_t$ 是一个随时间变小的系数，控制保留多少原始信号。

模型要做的，就是接收一个完全的随机噪声（$t=T$时），一步一步预测噪声是什么，然后把它减掉，最终还原出 $a_0$。因为这是一个采样过程，你从不同的随机噪声开始，就能采样出从左抓或者从右抓的不同mode，完美避开了平均化问题。轨迹在时间上也是连续平滑的。

### 3.3 Flow Matching的崛起
Pi-0和SmolVLA用了Flow Matching。Diffusion还是太慢了，生成一个动作要迭代几十步去噪，机器人控制频率顶多几十赫兹。

Flow Matching的公式 $\frac{da_t}{dt} = v_\theta(a_t, t, c)$ 就是把Diffusion的离散去噪变成了一个连续的向量场。它直接学习一个速度场，把随机噪声沿着直线“流”向真实数据分布。因为路径更直，它只需要很少的步数就能生成高质量动作，这让Pi-0能跑到200Hz的超高频控制。这对于需要快速反应的动态任务（比如接住抛过来的物体）是决定性的。

参考: [Diffusion Policy](https://arxiv.org/abs/2303.04137), [Pi-0](https://arxiv.org/abs/2410.24164)

---

## 4. 数据集的真相：缺的不是数据，是“复杂且丰富”的数据

这篇review里最棒的一个贡献是那个 $C_{task}$ 和 $C_{mod}$ 的二维评价框架。我们来看看它的公式到底在暗示什么：

$C_{task}(D) = \alpha_1 \log(1 + T) + \alpha_2 S + \alpha_3 D + \alpha_4 L$

- $T$ 是每个episode的动作数。加了个 $\log$ 说明动作多一点确实更复杂，但多到一定程度边际效应递减。
- $S$ 是高级技能数量。
- $D$ 是任务顺序依赖度。
- $L$ 是语言复杂度。

作者其实是在抱怨：我们现在的VLA数据集，要么是像R2R这种只有导航的简单任务，要么是像ALFRED这种只有单一pick-and-place的。根本没有那种“去厨房把水烧开，顺便把地扫了，然后再切个洋葱”的长程、多技能交织的数据集。

$C_{mod} = \beta_1 M + \beta_2 Q + \beta_3 A + \beta_4 R$

- $M$ 是模态数量。现在大部分数据集只有RGB和Language，最多加个Depth。
- $Q$ 是信号质量。
- $A$ 是时间对齐。
- $R$ 是有没有场景图这种高级语义标签。

这里暴露的问题是：**触觉和力觉数据极度稀缺**。机器人光靠看是不够的，你让它插个USB或者拧个螺丝，视觉是有遮挡的，必须靠手上的力传感器去感受阻力。但你看图表，只有Kaiwu这种极少数数据集把EMG、gaze、IMU、audio全放进去了。这也是为什么TLA（Tactile-Language-Action）这种工作在Table 5里被重点提出来，因为这是一个巨大且未被填满的坑。

参考: [TLA](https://arxiv.org/abs/2503.08548), [Kaiwu](https://arxiv.org/abs/2503.05231)

---

## 5. Sim-to-Real的鸿沟：物理引擎的谎言

在Simulation那一节，作者一针见血地指出了为什么仿真环境生成的数据放到真机上总是崩。

1. **Coulomb Friction的谎言**：Isaac Sim或者MuJoCo底层用的都是简单的库仑摩擦模型和点接触近似。真实世界里，橡胶抓手捏住一个软杯子，接触面是会变形的，摩擦力是非线性的。仿真里因为没有这种软体变形，模型在仿真里学会了用某种力气去捏，放到真机上杯子直接飞了或者被捏扁了。
2. **缺乏Language Grounding API**：你在仿真里想让机器人执行一个任务，你得自己写脚本去生成“把杯子放到左边”这句话，然后硬把它跟代码动作绑起来。没有一个统一的API让仿真器自己理解自然语言并自动生成场景逻辑。

这就是为什么现在很多顶会论文都在喊Hybrid Pipeline。你用仿真生成海量的domain randomization数据去教模型大致的几何和空间感，然后用几百条真实世界的昂贵遥操数据去做最后的微调。

参考: [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim), [SAPIEN](https://arxiv.org/abs/2003.08515)

---

## 6. 总结：VLA的未来是“大脑做规划，小脑做控制”

这篇paper在讲Future Directions时，隐隐约约指向了一个Hybrid架构的未来。

完全End-to-End的一个巨型Transformer去输出每个时刻的关节力矩，这听起来很酷，但在工程上极其难调试，且不安全。

未来的VLA大概率会分化成两层：
1. **System 2 (大模型)**：一个LLM级别的VLM，它很慢，但在做long-horizon planning。它接收画面和指令，输出“下一步应该去抓那个红色块”的子目标或者 affordance map。
2. **System 1 (小模型)**：一个轻量级的Diffusion Policy或者MPC控制器，它很快，专门负责把“抓那个红色块”这个抽象目标转化为极其平滑的、避开障碍物的关节轨迹。

像SayCan、HiRT、OneTwoVLA其实已经在走这条路了。把语义理解和低级控制解耦，既保留了LLM的常识推理能力，又保证了机器人控制的实时性和安全性。这才是能真正走进千家万户的机器人架构。

---

# VLA Models in Robotic Manipulation: 深度技术解读

这篇paper是Khalifa University的Muhayy Ud Din等人写的一篇系统性综述，覆盖了102个VLA模型、26个数据集、12个仿真平台。我下面从架构、数学、数据、仿真四个层面深入拆解，重点build intuition about why these design choices work.

---

## 1. VLA的根本动机：为什么要把Vision + Language + Action统一

传统robotic manipulation的pipeline是分层的：perception module（CNN/object detector）→ planning module（symbolic planner/MPC）→ control module（PID/operational space controller）。每个module独立训练，中间通过hand-crafted interfaces（如object pose、waypoint）传递信息。这种设计有几个fundamental problems：

- **Error propagation**：上游perception的误差会放大到下游control，且没有end-to-end的gradient信号让perception适应control的需求。
- **Semantic gap**：language instruction（"把苹果放到盘子上"）和motor command（joint torque sequence）之间隔了好几层symbolic abstraction，每层都需要manual engineering。
- **Generalization failure**：task-specific programming在dynamic、unstructured environment中脆弱，因为hand-crafted rules无法覆盖long-tail场景。

VLA的核心insight是：用一个single可微的transformer把这三个modality压到一个shared latent space，让gradient从action loss一直回传到vision和language encoder。这本质上是在做**multimodal representation alignment + end-to-end policy learning**的联合优化。

参考：[RT-1 paper](https://arxiv.org/abs/2212.06817)、[SayCan paper](https://arxiv.org/abs/2204.01691)、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)

---

## 2. VLA架构解析：Three-Stream Encoder + Fusion Backbone + Action Decoder

Figure 7展示的architecture是当前SOTA VLA系统（RT-2、OpenVLA、Octo、Pi-0、CLIP-RT）的canonical form。我把它拆成三个子模块讲。

### 2.1 Visual Encoder

输入是raw RGB（optionally depth/semantic mask），通常224×224或更高分辨率。处理方式有两种主流路线：

**ViT-based（dominant choice）**：CLIP-ViT、SigLIP、DINOv2是三个最常用的backbone。ViT把image切成non-overlapping patches（通常16×16），每个patch linearly project成一个token embedding，加上learnable positional encoding，然后过standard transformer encoder。

数学上，对于image $x \in \mathbb{R}^{H \times W \times C}$，patch size $P \times P$，得到 $N = \frac{H \cdot W}{P^2}$ 个patches。每个patch flatten后乘以projection matrix $E \in \mathbb{R}^{(P^2 \cdot C) \times d}$ 得到token embedding。这里 $d$ 是model dimension（如768 for ViT-B）。

- **CLIP-ViT**：用contrastive loss和text encoder联合训练，visual feature天然和language aligned。适合需要强semantic grounding的任务。
- **DINOv2**：self-supervised训练，feature对dense prediction（segmentation、depth）更强。适合需要fine-grained spatial understanding的manipulation。
- **SigLIP**：用sigmoid loss替代softmax loss，batch size可以更小，训练更稳定。OpenVLA和Pi-0都用它。

**Hybrid方案**：OpenVLA、CogACT、HybridVLA同时用SigLIP + DINOv2，把两路visual token concatenate。这是为了同时拿到semantic alignment（来自CLIP family）和dense spatial feature（来自DINO family）。

**CNN-based（legacy但仍在用）**：ResNet、EfficientNet出现在RT-1、CLIPort、ACT里。CNN的inductive bias（locality、translation equivariance）在小数据量时更sample efficient，但long-range dependency建模不如ViT。

### 2.2 Language Encoder

输入是natural language instruction，从短指令（"pick up the apple"）到long-horizon多步指令（"first open the drawer, then put the bowl inside, then close it"）。

主流选择：
- **LLaMA-2/3 family**：OpenVLA、RevLA、HybridVLA、ECoT都用。优势是pretrained on internet-scale text，有strong reasoning和instruction following能力。7B参数是常见配置。
- **T5/T5-XXL**：VIMA、Octo、RDT-1B用。encoder-decoder结构，适合需要explicit cross-attention的任务。
- **Qwen2/Qwen2-VL**：Chain-of-Affordance、Edge VLA、DexVLA用。在中文和multimodal场景表现好。
- **CLIP text encoder**：CLIPort、PerAct用。轻量，但reasoning能力弱。
- **Gemma-2B**：Pi-0、FAST、SpatialVLA用。和PaliGemma配合，参数量小适合real-time inference。

关键设计选择：language token和visual token要在同一个 $d$-dimensional space里，这样cross-attention才能work。通常通过linear projection把不同encoder的输出对齐。

### 2.3 State Encoder

这是被很多review忽略但实际很重要的模块。输入是robot的proprioceptive state：joint angles $\mathbf{q} \in \mathbb{R}^n$、end-effector pose $\mathbf{x}_{ee} \in SE(3)$、gripper status $g \in \{0, 1\}$、joint velocities $\dot{\mathbf{q}}$。

用一个小MLP或small transformer把这些continuous value embed成几个token。为什么必须做这个？因为：

1. **Reachability reasoning**：模型需要知道"我的arm现在在这个configuration，能不能reach到那个object"。
2. **Closed-loop correction**：execution过程中有disturbance，模型需要知道current state才能生成corrective action。
3. **Embodiment awareness**：不同robot的kinematics不同，state token让模型implicitly learn forward kinematics。

OpenVLA、Octo、Pi-0都显式incorporate proprioceptive token。

### 2.4 Fusion Backbone

三个stream的token concatenate成一个long sequence，送进transformer做cross-modal attention。这里是整个VLA的"brain"。

有两种主要paradigm：

**Early fusion**：所有token从第一层就在一起attend。OpenVLA、Gato、RT-2用这种。优点是信息融合最彻底，缺点是computation expensive（sequence length = visual tokens + language tokens + state tokens，通常几百到上千）。

**Late fusion / Modular**：vision和language先用一个小VLM处理，输出一个compact task embedding，再和state token一起送进action decoder。SayCan、VoxPoser、HiRT用这种hierarchical结构。优点是latency低、模块可替换，缺点是信息损失。

### 2.5 Action Decoder

这是VLA区别于普通VLM的关键模块。有四种主流设计：

#### (a) Autoregressive Token Decoder

把action离散化成token，让LLM autoregressive地生成。RT-1、RT-2、OpenVLA、Gato用这种。

RT-1把每个action dimension分成256个bin，7-DoF action + gripper + terminate flag = 9个token，每个inference step生成9个token。

优点：可以直接reuse LLM的next-token prediction machinery，training简单。
缺点：discretization损失精度，autoregressive generation慢（9个token要9次forward pass）。

#### (b) Diffusion Policy

把action generation看成denoising process。从一个noisy action trajectory $\mathbf{a}_T \sim \mathcal{N}(0, I)$ 开始，迭代denoise $T$ 步得到clean action $\mathbf{a}_0$。

Diffusion Policy的forward process：
$$q(\mathbf{a}_t | \mathbf{a}_0) = \mathcal{N}(\mathbf{a}_t; \sqrt{\bar{\alpha}_t} \mathbf{a}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 是cumulative product of noise schedule，$\mathbf{a}_0$ 是clean action trajectory，$\mathbf{a}_t$ 是step $t$ 的noisy version。

Reverse process用网络 $\epsilon_\theta$ 预测noise：
$$\mathbf{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{a}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{a}_t, t, \mathbf{c}) \right) + \sigma_t \mathbf{z}$$

其中 $\mathbf{c}$ 是conditioning（visual + language + state features），$\mathbf{z} \sim \mathcal{N}(0, I)$，$\sigma_t$ 是variance。

为什么diffusion对manipulation特别合适？因为action distribution是**multimodal**的——同一个task（比如"把杯子放到桌上"）可以有多个valid trajectory（从左边抓、从右边抓、用不同gait）。普通的MSE regression会average这些mode，产生无意义的中间trajectory。Diffusion能sharp地model每个mode。

Octo、Diffusion Policy、RDT-1B、CogACT、DexVLA、HybridVLA都用diffusion head。RDT-1B是1.2B参数的diffusion foundation model，在bimanual manipulation上SOTA。

#### (c) Flow Matching

Pi-0、$\pi$-0.5、SmolVLA用的variant。Flow matching是diffusion的generalization，学习一个vector field把noise distribution transport到action distribution：

$$\frac{d\mathbf{a}_t}{dt} = v_\theta(\mathbf{a}_t, t, \mathbf{c})$$

训练目标：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{a}_0, \mathbf{a}_1} \| v_\theta(\mathbf{a}_t, t, \mathbf{c}) - (\mathbf{a}_1 - \mathbf{a}_0) \|^2$$

其中 $\mathbf{a}_t = (1-t)\mathbf{a}_0 + t\mathbf{a}_1$ 是linear interpolation，$\mathbf{a}_0 \sim \mathcal{N}(0, I)$，$\mathbf{a}_1$ 是real action。

Flow matching比diffusion的优势：path更straight，sampling step更少，适合200Hz+的高频控制。Pi-0能跑到200Hz就是这个原因。

#### (d) MLP / Token Predictor Head

最简单的方案，直接用MLP把fused feature map到action space。CLIPort、TraceVLA、RoboMamba用。适合simple task或latency-critical场景。

参考：[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[Pi-0](https://arxiv.org/abs/2410.24164)、[RDT-1B](https://arxiv.org/abs/2410.07864)、[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

---

## 3. Self-Attention和Multi-Head Attention的数学

这是整个transformer的基础，我详细讲变量含义。

### 3.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

变量含义：
- $Q \in \mathbb{R}^{n \times d_k}$：Query matrix，$n$ 是sequence length，$d_k$ 是key/query dimension。每个row $\mathbf{q}_i$ 表示token $i$ "想要找什么信息"。
- $K \in \mathbb{R}^{n \times d_k}$：Key matrix。每个row $\mathbf{k}_j$ 表示token $j$ "能提供什么信息"。
- $V \in \mathbb{R}^{n \times d_v}$：Value matrix，$d_v$ 是value dimension。每个row $\mathbf{v}_j$ 是token $j$ 的实际content。
- $d_k$：key的dimension，用于scaling。除以 $\sqrt{d_k}$ 是为了控制dot product的方差——当 $d_k$ 大时，$QK^\top$ 的element会变大，softmax会进入saturation region，gradient消失。
- $\text{softmax}$：沿last dimension做normalization，让attention weight求和为1。

Intuition：$QK^\top$ 计算每个query和每个key的similarity（dot product），softmax把这些similarity变成probability distribution，然后用这个distribution加权aggregate values。结果是每个token得到一个context-aware representation。

### 3.2 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

变量含义：
- $h$：head数量（如8、16、32）。
- $W_i^Q \in \mathbb{R}^{d \times d_k/h}$、$W_i^K \in \mathbb{R}^{d \times d_k/h}$、$W_i^V \in \mathbb{R}^{d \times d_v/h}$：第 $i$ 个head的projection matrix。每个head看到input的不同subspace。
- $W^O \in \mathbb{R}^{(h \cdot d_v) \times d}$：output projection，把concatenated heads压回 $d$ 维。
- 总参数量：$3 \times d \times d + d \times d = 4d^2$（假设 $d_k = d_v = d$）。

为什么multi-head有用？不同的head可以specialize到不同的pattern——有的head关注syntactic dependency，有的关注semantic similarity，有的关注spatial locality。在VLA里，有的head可能专门做vision-language alignment，有的做action sequence modeling。

参考：[Attention is All You Need](https://arxiv.org/abs/1706.03762)

---

## 4. VLA模型Taxonomy：从Table 1看到的Trends

Table 1列了102个模型，我按action decoder类型分类讲trends。

### 4.1 Diffusion-based（最popular）

Octo、RDT-1B、CogACT、DexVLA、HybridVLA、DexGraspVLA、Diffusion-VLA、MoLe-VLA、EnerVerse、3D-VLA都用diffusion head。

**Octo**是第一个generalist diffusion policy，在Open X-Embodiment上训练，覆盖22个robot platform、4M+ trajectory。架构是CNN vision encoder + T5 language encoder + Diffusion Transformer head。关键创新是**diffusion transformer**——把diffusion的denoising network换成transformer，利用attention的long-range建模能力。

**RDT-1B**是1.2B参数的bimanual manipulation foundation model，用SigLIP + T5-XXL做encoder，Diffusion Transformer + MLP做decoder。在ALOHA bimanual task上zero-shot transfer表现强。

### 4.2 Autoregressive Token-based

RT-1、RT-2、OpenVLA、Gato、Pi-0（部分）、FAST用这种。

**RT-2**的关键创新是**co-finetuning**——同时在internet-scale VQA data和robot trajectory data上训练。这让model从web knowledge中获得emergent capability，比如能理解"move the object to the Taylor Swift picture"这种需要world knowledge的instruction。

**OpenVLA**是开源版RT-2，用DINOv2 + SigLIP做vision encoder，LLaMA-2 7B做language backbone，LoRA fine-tuning。在OXE + DROID上训练，性能match RT-2但完全open。

**FAST**（Fast Action Sequence Tokenization）是Karl Pertsch等人的工作，把action sequence用DCT（Discrete Cosine Transform）压缩到frequency domain再tokenize，能把inference速度提升15×。这是很clever的工程创新——action trajectory在frequency domain是sparse的，大部分energy集中在low frequency。

### 4.3 Flow Matching

Pi-0、$\pi$-0.5、SmolVLA、Hi Robot用。Physical Intelligence的Pi-0是landmark work，3B参数，用PaliGemma（SigLIP + Gemma-2B）做VLM backbone，flow matching做action expert。能跑200Hz，zero-shot transfer到新robot。

### 4.4 Hierarchical / Modular

SayCan、VoxPoser、HiRT、Hi Robot、HAMSTER、OneTwoVLA用hierarchical design——high-level planner（LLM）生成subgoal，low-level controller（small policy）执行。

**SayCan**的经典设计：LLM生成candidate action sequence，每个action有一个affordance value（从value function学），用 $P(\text{action}) \propto P(\text{LLM}|\text{action}) \times V(\text{state})$ 选action。这把semantic plausibility和physical feasibility解耦。

**OneTwoVLA**用decision token在reasoning和control之间动态切换，很有意思的设计。

参考：[OpenVLA](https://arxiv.org/abs/2406.09246)、[FAST](https://arxiv.org/abs/2501.09747)、[Pi-0](https://arxiv.org/abs/2410.24164)、[SayCan](https://arxiv.org/abs/2204.01691)

---

## 5. 数据集Benchmarking Framework：C_task和C_mod

这是这篇review最novel的贡献。作者提出一个2D framework来quantitatively评估VLA dataset。

### 5.1 Task Complexity Score

$$C_{\text{task}}(D) = \alpha_1 \log(1 + T) + \alpha_2 S + \alpha_3 D + \alpha_4 L$$

变量含义：
- $T$：average number of low-level actions per episode。比如pick-and-place可能 $T=50$，long-horizon cooking task可能 $T=500$。用 $\log(1+T)$ 是因为complexity随action数量sub-linearly增长（很多action是repetitive的）。
- $S$：number of distinct high-level skills。比如"open drawer"、"pick object"、"pour water"是3个skill。更多skill意味着model需要learn更多diverse capability。
- $D \in [0, 1]$：degree of sequential task dependency。如果task必须按严格顺序执行（如"先开门再拿东西"），$D$ 接近1；如果skill可以parallel或任意order，$D$ 接近0。
- $L \in \mathbb{R}^+$：linguistic abstraction level。用vocabulary size或syntactic depth量化。简单指令"pick apple"的 $L$ 低，复杂指令"carefully place the fragile glass on the left side of the table"的 $L$ 高。
- $\alpha_i > 0$：weights，paper里都设为1，但可以tune。

### 5.2 Modality Richness Score

$$C_{\text{mod}} = \beta_1 M + \beta_2 Q + \beta_3 A + \beta_4 R$$

变量含义：
- $M$：number of distinct modalities。RGB、depth、language、proprioception、tactile、audio、force/torque是常见modality。
- $Q = \frac{1}{M} \sum_{i=1}^{M} Q_{m_i}$：mean quality score，$Q_{m_i} \in [0.6, 0.95]$。由expert annotation、SNR analysis或documentation决定。
- $A \in [0, 1]$：temporal alignment fidelity。frame-accurate alignment的 $A=1$，approximate alignment的 $A$ 低。
- $R \in \{0, 1\}$：presence of reasoning-critical modalities（object mask、scene graph等）。这些modality让model能做higher-level reasoning。
- $\beta_i > 0$：weights，都设为1。

### 5.3 Normalization和Visualization

两个score分别normalize到 $[1, 5]$ 和 $[2, 5]$，然后在2D plane上plot，bubble size表示dataset scale。

### 5.4 Key Finding：Coverage Gap

Figure 10揭示了一个critical gap：**high task complexity + comprehensive modality的dataset极度稀缺**。

具体来说：
- EmbodiedQA、R2R、RoboSpatial：low complexity, minimal modality（navigation/QA主导）。
- RLBench、TEACh、Ego4D：low-medium complexity, medium modality。
- ALFRED、DialFRED、CoVLA：medium complexity, rich modality。
- IRef-VLA、Robo360、TLA、CALVIN、Open X-Embodiment：high complexity, rich modality。
- **Kaiwu**：唯一一个very high complexity + most comprehensive modality的dataset，集成vision、depth、language、proprioception、haptics、EMG、gaze、IMU、audio、motion capture。
- **AgiBot World**：very high complexity, medium modality（emphasize scale over sensor diversity）。

这个gap的implication：要build truly generalist robot，我们需要combine Kaiwu的modality richness + AgiBot World的scale + CALVIN的long-horizon complexity。这是future dataset development的方向。

参考：[Kaiwu dataset](https://arxiv.org/abs/2503.05231)、[AgiBot World](https://arxiv.org/abs/2503.06669)、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)、[DROID](https://droid-dataset.github.io/)、[CALVIN](https://arxiv.org/abs/2112.03227)

---

## 6. Simulation Platforms对比

Table 3列了12个simulator。我按use case分类。

### 6.1 Photorealistic Navigation

**AI2-THOR**、**Habitat**、**iGibson**：这三个是embodied navigation的主力。提供photorealistic RGB、depth、semantic segmentation。

- AI2-THOR有120个indoor scene，支持physics-based interaction。ALFRED、TEACh、DialFRED都基于它。
- Habitat支持大规模3D scene（Matterport3D、HM3D），rendering速度快（1000+ FPS），是R2R、CVDN、EmbodiedQA的backend。
- iGibson强调interactive object和dynamic scene。

### 6.2 High-Fidelity Manipulation

**NVIDIA Isaac Sim**、**SAPIEN**、**MuJoCo**：

- Isaac Sim用PhysX + RTX rendering，是Open X-Embodiment、Isaac Gym的backend。physic accuracy和rendering quality都top-tier，但GPU demand高。
- SAPIEN是Stanford的part-based simulator，特别适合articulated object（drawer、door、scissor）。DexGraspNet、TLA用它。
- MuJoCo是DeepMind的classic，contact dynamics准确，analytic gradient可用。Meta-World、RoboSuite基于它。

### 6.3 Lightweight / High-Throughput

**PyBullet**、**Unity ML-Agents**、**Webots**：

- PyBullet是Python wrapper around Bullet，real-time physics，cross-platform。适合rapid prototyping。
- Unity ML-Agents用Unity engine的rendering，visual fidelity好但physics一般。
- Webots开源，支持多robot、多sensor，GUI友好。

### 6.4 Multi-Robot / Cloud-Native

**Gazebo**、**UniSim**、**CoppeliaSim**：

- Gazebo是ROS生态的主力，支持URDF/SDF import，多robot coordination强。
- UniSim是unified multi-sensor API的cloud-native simulator。
- CoppeliaSim（原V-REP）支持多个physics engine，适合multi-robot coordination。

### 6.5 Simulator的关键Limitation

Paper指出4个challenge：

1. **Physics accuracy**：大多数simulator用Coulomb friction + point contact approximation，无法model soft-body deformation、variable friction、compliance。这是sim-to-real gap的主要来源。
2. **Visual realism vs throughput tradeoff**：photorealistic rendering慢，high-throughput rendering不realistic。
3. **Lack of language grounding API**：大多数simulator没有native support把language instruction map到agent behavior，需要custom pipeline。
4. **Multi-robot support不一致**：Isaac Sim、Gazebo支持arbitrary URDF，Webots、RoboSuite只优化特定robot family。

参考：[AI2-THOR](https://ai2thor.allenai.org)、[Habitat](https://arxiv.org/abs/1904.01201)、[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)、[SAPIEN](https://arxiv.org/abs/2003.08515)、[MuJoCo](https://mujoco.org/)

---

## 7. Evaluation：10个SOTA模型对比

Table 5选了10个manipulation-focused模型详细比较。我重点讲几个insight。

### 7.1 RT-2

- Benchmark：Open X-Embodiment、BridgeData V2
- Success rate：High（≥90%）
- Zero-shot：High
- Real-robot：Yes
- 核心贡献：co-finetuning on internet VQA + robot data，unlock emergent multi-robot zero-shot capability。

RT-2的intuition很深刻——web-scale VQA data给model提供了vast world knowledge（object affordance、spatial relation、commonsense），robot data给model提供了motor control capability。两者synergy让model能generalize到unseen task。

### 7.2 Pi-0

- Benchmark：Pi-Cross-Embodiment
- Success rate：Medium
- Zero-shot：Medium
- Real-robot：Yes
- 核心贡献：3B参数，200Hz+ control，flow matching action expert。

Pi-0证明了一点：**不需要巨型model也能做generalist robot control**。3B参数通过careful architecture design（flow matching + PaliGemma + efficient inference）就能match甚至超过更大model。

### 7.3 Octo

- Benchmark：RLBench、Open X-Embodiment
- Success rate：Medium
- Zero-shot：Medium
- Real-robot：Yes
- 核心贡献：first diffusion-based generalist，4M+ trajectory，22 robot platform。

Octo的意义在于证明diffusion policy可以scale到generalist level。之前diffusion policy主要在single-task、single-robot上work，Octo把它推到cross-embodiment。

### 7.4 OpenVLA

- Benchmark：Open X-Embodiment、DROID
- Success rate：Medium
- Zero-shot：Medium
- Real-robot：Yes
- 核心贡献：open-source，LoRA fine-tuning，match RT-2 performance。

OpenVLA的价值是democratization——让academic lab也能用SOTA VLA。DINOv2 + SigLIP的hybrid vision encoder也成了后续很多model的标准配置。

### 7.5 CLIPort

- Benchmark：Ravens pick-and-place suite
- Success rate：Medium
- Zero-shot：Low
- Real-robot：Yes
- 核心贡献：CLIP semantic grounding + Transporter Network dense transport。

CLIPort是早期VLA的classic work。它的insight是：CLIP pretraining提供strong semantic feature，Transporter Network提供spatial precision，两者combine既懂"what"又懂"where"。

### 7.6 TLA

- Benchmark：TLA benchmark
- Success rate：Medium
- Zero-shot：High
- Real-robot：Yes
- 核心贡献：first language-tactile VLA，85%+ success on contact-rich task。

TLA是这篇review里highlight的novel direction——把tactile sensing引入VLA。Contact-rich manipulation（peg-in-hole、assembly）纯vision不够，需要tactile feedback做force-controlled insertion。

### 7.7 两大Trajectory

Paper总结了两条main trajectory：

1. **Large generalist**（RT-2、Octo、Gato、OpenVLA）：massive transformer + diffusion/autoregressive decoder + million-trajectory pretraining。强zero-shot generalization。
2. **Modular specialist**（DexVLA、CLIPort、TLA、RoboAgent）：targeted module（object-centric ViT、tactile encoder、LoRA adapter、semantic augmentation）。强precision和data efficiency。

这两条路线不mutually exclusive——未来的方向可能是large generalist backbone + task-specific adapter的hybrid。

参考：[RT-2](https://arxiv.org/abs/2307.15818)、[Pi-0](https://arxiv.org/abs/2410.24164)、[Octo](https://arxiv.org/abs/2405.12213)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[CLIPort](https://arxiv.org/abs/2109.12098)、[TLA](https://arxiv.org/abs/2503.08548)

---

## 8. Challenges和Future Directions的Deep Dive

### 8.1 Architectural Challenges

**Tokenization问题**：BPE对text好，但visual patch和continuous proprioception怎么tokenize？BLIP-2的Q-Former是one solution——用learned query token去attend visual feature，输出fixed-length visual token。Perceiver IO用shared latent array。但如何处理high-dimensional sensor stream（如tactile的1000+ sensing element）仍是open problem。

**Modality Fusion问题**：简单concatenation或cross-attention不够。Paper提到"align-then-fuse" paradigm——先用contrastive learning把vision和language align到same space，再fuse。VLMo的mixture-of-modality-expert是另一个方向——每个modality有dedicated expert layer。

**Cross-embodiment generalization**：不同robot的action space不同（7-DoF arm vs 6-DoF vs quadruped），如何transfer？PaLM-E用explicit hardware embedding。DexVLA用plug-in diffusion expert。但zero-shot transfer到completely novel robot仍open。

**Motion smoothness**：discrete action token可能产生jerky motion。Diffusion policy通过iterative denoising生成temporally coherent trajectory，但real-time inference难。如何balance smoothness和reactivity是open problem。

### 8.2 Dataset Challenges

**Task diversity**：大多数dataset是narrow domain（ALFRED只pick-and-place，R2R只navigation）。需要combine locomotion + manipulation + social interaction的long-horizon dataset。

**Modality imbalance**：RGB + text是标配，但depth、tactile、force/torque、audio经常missing或unsynchronized。Kaiwu和REASSEMBLE是少数exception。

**Annotation cost**：6-DoF pose、frame-aligned multi-sensor、detailed language explanation都需要expensive manual annotation。Self-supervised和auto-labeling还不够reliable。

**Realism vs scale tradeoff**：real-world data（Open X-Embodiment）realistic但expensive，simulated data scalable但有domain gap。Hybrid synthetic-real pipeline（neural rendering + physics-aware domain randomization）是promising direction。

### 8.3 Simulation Challenges

**Physics accuracy**：Coulomb friction + point contact不够。需要differentiable、multi-scale contact model，blend classical solver和data-driven calibration。这能better handle soft-body deformation、friction variability、compliance。

**Visual realism vs throughput**：Hybrid rendering pipeline——general frame用high-throughput rasterization，key scene用neural/ray-traced rendering。

**Language grounding API**：需要simulator-agnostic API把language instruction map到scene graph和agent behavior。这是ecosystem-level的infrastructure需求。

**Multi-robot support**：需要auto-import URDF/SDF + shared simulation protocol，让policy能在heterogeneous robot platform上pretrain。

### 8.4 Future Directions的Technical Spec

Paper给出具体technical recommendation：

1. **Modality-aware tokenizer**：VQ-VAE或neural dictionary把continuous sensor stream（proprioception、force/torque）和visual、textual input jointly discretize。
2. **Dynamic fusion block**：gating network、mixture-of-experts、conditional attention根据task demand reweight modality。
3. **Hierarchical architecture**：lightweight CNN/RNN frontend downsample high-frame-rate input，sparse transformer layer做long-range modeling。
4. **Diffusion + safety filter**：diffusion trajectory generator + differentiable safety/collision-avoidance filter，生成smooth compliant motion。
5. **Procedural task grammar**：simulator embed task grammar，auto-generate long-horizon open-ended scenario。
6. **Standardized multimodal capture pipeline**：synchronize RGB-D、tactile、force/torque、audio、language at compatible sampling rate。
7. **Cross-modal synthesis**：missing modality通过synthesis augment（如monocular depth estimation补depth）。
8. **Self-supervised annotation**：unsupervised segmentation、vision-language co-training、active learning自动extract object mask、6-DoF trajectory、language explanation。
9. **Differentiable multi-scale contact model**：blend classical solver和data-driven calibration。
10. **Hybrid rendering pipeline**：rasterization + neural/ray-traced rendering。
11. **Simulator-agnostic language grounding API**。
12. **Multi-robot multi-agent support**：auto-import URDF/SDF + shared protocol。

参考：[BLIP-2](https://arxiv.org/abs/2301.12597)、[Perceiver IO](https://arxiv.org/abs/2107.14795)、[VLMo](https://arxiv.org/abs/2111.02358)、[PaLM-E](https://arxiv.org/abs/2303.03378)、[DexVLA](https://arxiv.org/abs/2502.05855)、[VLA Repository](https://github.com/Muhayyuddin/VLAs)

---

## 9. 我的Intuition总结

读完这篇review，我对VLA field的intuition是：

**VLA的本质是multimodal representation alignment + end-to-end policy learning**。传统的modular pipeline（perception → planning → control）通过hand-crafted interface传递信息，每个module独立优化。VLA用一个可微transformer把所有module压到一个shared latent space，让gradient从action loss回传到perception，实现真正的end-to-end optimization。

**Diffusion/Flow Matching是action decoder的主流选择**，因为action distribution是multimodal的。Autoregressive token decoder虽然simple但discretization损失精度且inference慢。Flow matching比diffusion更适合real-time high-frequency control。

**Data是bottleneck**。Figure 10的coverage gap说明我们需要combine Kaiwu的modality richness + AgiBot World的scale + CALVIN的long-horizon complexity。没有这样的dataset，truly generalist robot无法实现。

**Simulation是scalability的关键**，但physics accuracy和visual realism的tradeoff、language grounding API的缺失、multi-robot support的不一致是三大瓶颈。Future simulator需要differentiable contact model + hybrid rendering + unified language API + auto-import URDF。

**Architectural趋势是hybrid**：large generalist backbone（提供broad capability）+ task-specific adapter（提供precision）。OpenVLA的LoRA、DexVLA的plug-in diffusion expert、Pi-0的flow matching expert都是这个方向。纯粹的monolithic model和纯粹的modular pipeline都不是future——hybrid才是。

**Tactile和contact-rich manipulation是underexplored frontier**。TLA、REASSEMBLE、ARIO是少数incorporate tactile的dataset/model。要实现general-purpose robot（能做cooking、cleaning、assembly），tactile sensing必须成为first-class modality。

这篇review的价值在于它systematic地map了整个VLA landscape，并通过C_task/C_mod framework揭示了data coverage gap。这对future research direction有clear guiding意义——我们需要more multimodal、more long-horizon、more contact-rich的dataset和model。
