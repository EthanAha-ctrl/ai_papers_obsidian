---
source_pdf: IGOR Image-GOal Representations.pdf
paper_sha256: 9c1b0720a29f36f52a227de7b8bbd06d4a7ee8d9cb47d782942b4644a975d2f1
processed_at: '2026-08-05T09:00:22-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# IGOR 用人话讲

## 一句话版本

IGOR 干的事：**把视频里"帧和帧之间发生了啥变化"压成几个离散 token，管它叫 latent action，然后用这个 token 当桥梁，让互联网上的人做事视频能给机器人当训练数据。**

---

## 为什么这事重要

做 embodied AI 最大的痛点就一个字：**数据少**。

LLM 能 scale 是因为 internet 上 text 几乎无限。robot 呢？Open-X-Embodiment 拼命凑也就 0.8M 条 trajectory，DROID 也是这个量级。但 YouTube 上人做饭、开抽屉、拿东西的视频有**几亿**条。

问题来了：人做饭的视频里**没有 robot action label**。你看到一个手把杯子从左边挪到右边，你知道"发生了移动"，但这个"移动"怎么映射到某个具体 robot 的 7 维 action vector？没答案。

IGOR 的回答是：**别映射了，根本不需要真实 action label。** 你只要能从视频里 recover 出一个"语义层面描述变化"的 token，这个 token 对人对机器人通用，就够了。然后让 foundation policy 在 token 层面做规划，low-level policy 再把 token 翻译成具体 robot action。

---

## 核心类比：人脑怎么看视频

你看到一段"手把抽屉拉开"的视频，大脑不会逐帧分析像素怎么变化，你脑子自动蹦出一个词：**"open"**。

IGOR 就是想让神经网络自动蹦出这种词。只不过它不用人类语言，而是自己发明一套离散 code。paper 里 codebook 是 $N=4$ 个 slot，每个 slot 有 $|C|=32$ 种选择，所以一个 latent action 是 4 个 token、每个 token 有 32 种可能。

这个"词表"是网络自己学的——没人告诉它 "open" "close" "move left" 这些概念，它从数据里 emergent 出来。

---

## 三个模型怎么串起来

整个 IGOR pipeline 是三个模型接力：

```
Video frames
    ↓
[1. Latent Action Model]  →  给每对 image-goal pair 打 latent action label
    ↓
[2. Foundation Policy]    →  给定文字指令 + 当前画面，预测下一个 latent action
    ↓
[3. Low-level Policy]     →  把 latent action 翻译成具体 robot 的 7 维 action
```

外加一个 **[2'. Foundation World Model]**：给定当前画面 + latent action，生成未来视频帧——用来验证 latent action 真的能控制画面变化，也用来 rollout 看 policy 效果。

---

## Latent Action Model 怎么训的

这是整个 paper 的核心，讲细一点。

### 架构：IDM + FDM 两个网络

- **IDM (Inverse Dynamics Model)**：看一对图 $(o_t, o_{t+1})$，输出 latent action $a_t$
- **FDM (Forward Dynamics Model)**：看单帧 $o_t$ 和 $a_t$，重建 $o_{t+1}$

两个一起训，loss 是 reconstruction $\|o_{t+1} - \hat{o}_{t+1}\|^2$ 加上 VQ 的 commitment loss。

### 三个关键设计 choice，每个都有 why

**Choice 1: VQ 把 action quantize 成离散 token**

如果不 quantize，latent action 就是个 continuous vector，网络可以随便编码，相似动作可能映射到完全不同的 vector。quantize 到 32 个桶 ×4 个 slot，信息瓶颈逼着网络把"语义相似的视觉变化"压到同一个 code。

这是 VQ-VAE 的老套路，但用在 action space 上有意思——相当于自创一套"动作语言"。

**Choice 2: FDM 只看单帧 $o_t$，不看历史**

这是反 shortcut 的关键。如果 FDM 看整个 $o_{1:t}$，网络会直接从历史外推 $o_{t+1}$，绕过 $a_t$，那 $a_t$ 就学不到东西了。改成只看单帧 $o_t$，要重建 $o_{t+1}$ **必须**靠 $a_t$ 携带信息——信息被迫流过 latent action。

直觉：你想知道"下一步会发生啥"，只给你现在的画面，那唯一能帮你预测的就是"要做什么动作"这个信号。

**Choice 3: IDM 和 FDM 用不同的 random crop**

IDM 看的视野和 FDM 看的视野空间错位。如果 latent action 真的是"语义变化"而非"像素位置变化"，那 crop 一下不该改变 $a_t$ 的语义。这是一种 implicit augmentation，逼着 latent action 学到 translation-invariant 的东西。

---

## World Model：基于 Open-Sora 改的

paper 不自己从头训 video generator，拿 Open-Sora 当 base 改。

关键改动：**把原来的 text condition 换成 latent action condition**。

具体注入方式：
1. latent action $a_{t:T-1}$ 每帧映射成单 token，通过 **cross-attention** 喂进 ST-RFT（spatial-temporal rectified flow transformer）
2. FDM 输出的粗粒度预测 $\hat{o}_{t+1:T}$，用 3D VAE 编码后 **element-wise 加到 noisy input 上**

第 2 点很巧妙——相当于给 world model 一张"草图"，让它不用从噪声里凭空 hallucinate 目标长啥样，只需 refine 细节。类似 ControlNet 但更轻量。

### Rectified Flow 公式讲一下

paper 用的是 velocity parameterization：

$$L = \mathbb{E}_{n, x_0, x_1}\|(x_1 - x_0) - v_\theta(x_n, n, a, \hat{o})\|^2$$

变量含义：
- $n \in [0,1]$：flow time，0 对应 clean data，1 对应纯噪声
- $x_0$：clean latent（VAE 编码后的视频 latent）
- $x_1$：标准高斯噪声
- $x_n = (1-n)x_0 + n x_1$：插值
- $v_\theta$：网络预测的 velocity
- $(x_1 - x_0)$：ground-truth velocity（因为 $\frac{dx_n}{dn} = x_1 - x_0$）

为什么用 RF 不用 DDPM？RF 把扩散路径"拉直"，推理步数少很多，视频生成效率高。Stable Diffusion 3 也是这套（https://arxiv.org/abs/2403.03206）。

---

## Policy Model：两阶段 hierarchy

**Stage 1 - Foundation Policy 预训练**：

输入文字 $s$ + 画面 $o_{1:t}$，输出 latent action $a_t$。loss 是 L2：
$$L = \|P([s; o_{1:t}]) - a_t\|^2$$

这就是在 latent action 空间做行为克隆。训练数据是 2.8M 条 video（robot + human），都用 LAM 打了 latent action label。

**Stage 2 - Low-level Policy fine-tuning**：

每个 latent action $a_t$ 对应 $\tau=3$ 个真实 robot action $u_t^{1:\tau}$（每个是 7 维：3 位置 + 3 旋转 + 1 夹爪）。

低层 policy 把 latent action 当 sub-task embedding，concat 到观测上，预测具体 7 维 action：
$$L = \|P_f([s; P([s; o_{1:t}]); o_{1:t}]) - u_t^{1:\tau}\|^2$$

只训低层 head，foundation policy 冻住。用 RT-1 的 **1% 数据** fine-tune。

### 这个 hierarchy 的 intuition

高层 policy 负责"做什么 sub-task"——这是 embodiment-agnostic 的，可以从人视频学。
低层 policy 负责"怎么具体执行"——这是 embodiment-specific 的，需要少量真实 robot data。

latent action 是两层之间的 **interface**。这种分解让 foundation model 的预训练知识能 transfer 到具体机器人，只 fine-tune 一个 small head——low-data regime 下的关键。

---

## 实验里几个让人信服的点

**1. Figure 3 - 相似 latent action → 相似视觉变化**

在 OOD 的 RT-1 上 retrieve latent action 最近的 image-goal pair，发现"open gripper" "move left" "close gripper" 这些 sub-task 自动聚到一起，而且这些 sub-task 出现在不同 raw language task 里。说明 latent action **被复用、被泛化**。

**2. Figure 2 - Human → Robot 迁移**

最惊艳的实验。从人视频提 latent action，apply 到 robot 初始图上，world model 能 generate 出 robot arm 执行相同动作的视频。人把手往左移的 latent action，用到 robot 上也能让 robot 往左移。

这证明 latent action **跨 embodiment 语义一致**。

**3. Figure 4 - Multi-object control**

同一张图有 apple/tennis/orange，apply 6 个不同 latent action，world model 能让**指定物体**动。说明 latent action 是 object-centric 的控制，不是全局 scene 变化。

**4. Figure 6b - Latent action 对真实 action 的 predictiveness**

在 latent action 空间找 nearest neighbor，看邻居们的真实 robot action std。N 越小 std 越低（且都 <1.0），说明**相似 latent action 对应相似真实 action**。

有意思的发现：latent action 对 **movement** 比 **rotation/gripper** 更 predictive。这可能因为 $|C|=32$ 太小，对连续精细动作（旋转角度）表达能力不够。

**5. Table 2 - Human data 帮 robot task**

Robot data only：OOD loss 0.145
加 Human data：0.112（↓22.8%）

证明 human video 通过 IGOR 框架确实能帮 robot——cross-embodiment transfer 不是空话。

---

## 我的几个看法

**1. Codebook 太小可能是 bottleneck**

$N=4 \times |C|=32 = 128$ token 的组合空间。描述"往左移 5cm"和"往左移 10cm"这种连续差异，32 个 code 不够。Figure 6b 显示 rotation/gripper predictiveness 弱，可能就是这个原因。扩 codebook 到 256/512 会怎样？paper 没试。

**2. World model 的物理一致性存疑**

Open-Sora base 本质是 video generation 模型，不是物理引擎。Sora 报告里（https://openai.com/research/video-generation-models-as-world-simulators）自己承认 long roll-out 会 break 物理一致性。IGOR 用 FDM 输出当 conditioning 部分缓解，但 long horizon rollout 误差累积问题没解决。

**3. Cross-embodiment claim 主要 qualitative**

Figure 2 视觉上惊艳，但缺 quantitative 的 human→robot transfer 实验（比如真机部署成功率）。只在 SIMPLER sim 上做了 1% data 的 success rate，那是 robot→robot。

**4. 最值得借鉴的技术点**

不是整体框架，是几个局部 trick：
- **Single-frame FDM 防 shortcut**：任何 predictive learning 都可能遇到这问题
- **Different crop for IDM/FDM**：implicit invariance learning 的轻量做法
- **FDM 输出 element-wise 加到 world model input**：比 cross-attention 更强的 conditioning 方式

这些可以单独抽出来用到别的 task。

---

## 最后再压缩一句

IGOR = **"给互联网视频自动打 action label，打的 label 跨人对机器人通用，然后在这套 label 上训 foundation policy"**。

核心 bet 是：**视觉变化的语义压缩空间是 embodiment-agnostic 的**。如果这个 bet 成立，互联网视频就是 embodied AI 的"text data"。

参考链接：
- Project: https://aka.ms/project-igor
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Rectified Flow: https://arxiv.org/abs/2209.03003
- SD3: https://arxiv.org/abs/2403.03206
- Genie: https://arxiv.org/abs/2401.15445
- VPT: https://arxiv.org/abs/2206.11795
- UniPi: https://arxiv.org/abs/2302.00111
- OpenVLA: https://arxiv.org/abs/2406.09246
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864
- SIMPLER: https://arxiv.org/abs/2405.05941

---

# IGOR: Image-Goal Representations 深度讲解

## 1. 核心动机与高层直觉

IGOR 想解决一个根本问题：**embodied AI 缺 interaction data**。Open-X-Embodiment、DROID 这些机器人数据集相比 internet 上的 text/video 数据，量级差好几个数量级。但 internet 上有海量 human activity video——如果能把这些"看人做事"的视频"翻译"成机器人能用的控制信号，就打开了 scaling 的大门。

IGOR 的核心 idea 用一句话讲：**把 image-goal pair 之间的视觉变化压缩成一个 discrete latent action**，这个 latent action 跨 embodiment 通用，相当于在视觉空间里定义了"atomic control units"。

类比人脑看视频：你不是逐帧分析，而是把帧间变化模块化成 "move / open / close" 这种压缩 token。IGOR 就是想让网络从 image-goal pair 里自动 recover 出这种 latent action，从而给互联网视频打 action label。

paper 里三个关键 insight 我重点强调：

1. **Latent action = sub-task embedding**：每个 latent action $a_t$ 描述从 $o_t$ 到 $o_{t+1}$ 的变化信息，本质上是"从初始图到达 goal 的子任务"。
2. **Single-frame FDM 是反 shortcut 的关键**：如果用整个 context $o_{1:t}$ 重建 $o_{t+1}$，网络会绕过 latent action 直接外推；改成只看单帧 $o_t$，逼着信息流过 $a_t$。
3. **Image-goal representations 是除 text embedding 和 image embedding 之外的第三个 building block**——这是 paper 想立起来的核心 thesis。

paper link: https://aka.ms/project-igor  
arXiv 链接需要从作者列表搜，相关引用都来自 Microsoft Research：https://www.microsoft.com/en-us/research

---

## 2. Latent Action Model (LAM) 架构详解

### 2.1 总体结构：IDM + FDM 对偶

LAM 由两部分组成：

- **IDM (Inverse Dynamics Model)** $I$: 给定观测序列 $o_{1:t+1}$，预测 latent action $a_t$
- **FDM (Forward Dynamics Model)** $F$: 给定单帧 $o_t$ 和 $a_t$，重建 $o_{t+1}$

### 2.2 IDM 内部

IDM 流程：
$$a_t = I(c_1[o_{1:t+1}])$$

其中 $c_1$ 是 random cropping。架构是 **ViT (DINO-v2 frozen)** 提每帧 feature，再过 **ST-transformer with temporal causal mask**，最后用 **learnable readout tokens** 压缩成 N 个 token。每个 token 经过 **Vector Quantization (VQ)**，限制在 codebook $|C|$ 内。

最终 $a_t \in \mathbb{R}^{N \times D}$，paper 用 $N=4, |C|=32, D=128$。

**直觉**: VQ 这步是关键——强制把视觉变化 quantize 到 32 个离散桶里 ×4 个 slot，逼着网络把"语义上类似的变化"映射到同一个 code。这就是为什么后续 retrieval 实验里相似动作会聚到一起。

### 2.3 FDM 内部与 random cropping 技巧

FDM 用 **single-frame ViT** 重建 $o_{t+1}$：
$$\hat{o}_{t+1} = F(c_2[o_t], a_t)$$

注意 $c_1$ 和 $c_2$ 是**不同的 cropping**——IDM 看的视野和 FDM 看的视野空间错位。这个设计的 intuition：如果 latent action 真的捕捉了"语义变化"而非"具体像素位置"，那 cropping 不应该改变 $a_t$ 的语义。相当于一种 implicit augmentation / invariance learning。

### 2.4 训练目标

Joint 训练，loss 由两部分组成：

- **Reconstruction loss**: $\|c_2[o_{t+1}] - \hat{o}_{t+1}\|^2$
- **Commitment loss** (VQ 标准)：让 encoder 输出贴近 codebook 里的 code

**为什么这个设计能学到一致的 latent action space**？因为如果压缩足够狠（N=4×D=128，相对于整张图的 pixel 信息量很小），那"相似视觉变化"的 image-goal pair 必然被压到相近的 code，否则 reconstruction 会失败。信息瓶颈逼出了语义一致性。

---

## 3. Foundation World Model (基于 Rectified Flow)

### 3.1 为什么用 Rectified Flow 而非 DDPM

paper 借用 Open-Sora 作为 base。Rectified Flow (Liu et al. 2023b, Esser et al. 2024) 的核心 idea 是把 DDPM 的曲扩散路径"拉直"，让训练目标变成回归 velocity field，推理步数可以少很多。

参考：
- Rectified Flow: https://arxiv.org/abs/2209.03003  
- Stable Diffusion 3 (Esser): https://arxiv.org/abs/2403.03206  
- Open-Sora: https://github.com/hpcaitech/Open-Sora

### 3.2 公式逐个拆解

**公式 (1) — 插值路径**：
$$x_n = (1-n)x_0 + n \cdot x_1$$

变量解释：
- $n \in [0,1]$：时间参数（flow matching 的 "time"），相当于扩散模型的 timestep 归一化
- $x_0$：clean data（在 IGOR 里是 VAE 编码后的 latent，不是 raw pixel）
- $x_1$：从标准高斯采样的 noise
- $x_n$：在 noise 和 data 之间做线性插值得到的 noisy sample

**公式 (2) — Naive loss（不实际用）**：
$$\mathbb{E}_{n, x_0, x_1}\|x_0 - x_\theta(x_n, n, a_{t:T-1}, \hat{o}_{t+1:T})\|^2$$

这里 $x_\theta$ 直接回归 clean data $x_0$。条件输入包括：
- $x_n$：当前 noisy latent
- $n$：flow time
- $a_{t:T-1}$：**latent action sequence**（来自 LAM），每个 action 映射成单 token 通过 cross-attention 注入 ST-RFT
- $\hat{o}_{t+1:T}$：FDM 给的粗粒度预测，通过 3D VAE 编码后 **element-wise 加到 noisy input 上**——这是一个很强的 conditioning signal，让 world model 不用从零猜目标长什么样

**公式 (3) — Velocity parameterization（实际用）**：
$$L_{\text{world}}(\theta) = \mathbb{E}_{n, x_0, x_1}\|(x_1 - x_0) - v_\theta(x_n, n, a_{t:T-1}, \hat{o}_{t+1:T})\|^2$$

变量：
- $v_\theta$：神经网络参数化的 velocity field
- $(x_1 - x_0)$：从 clean data 到 noise 的"真实速度向量"（因为 $x_n = (1-n)x_0 + nx_1$，对 $n$ 求导 $\frac{dx_n}{dn} = x_1 - x_0$，所以这就是 ODE 的 ground-truth velocity）

**直觉**: 训练时让网络预测"每一步该往哪个方向走"，推理时从 $x_1$（纯噪声）开始，按 ODE $\frac{dx}{dn} = v_\theta(\cdot)$ 积分到 $n=0$ 就得到 clean latent。

### 3.3 Open-Sora 修改

paper 对 Open-Sora 做了两个改动：
1. **替换 text input 为 latent action $a_{1:T}$**，最后一个 action 用 zero-padding。每帧的 latent action 映射到单 token，通过 cross-attention 喂进 ST-RFT。
2. **额外 conditioning 在 FDM 输出 $\hat{o}_{t+1:T}$**：用同一个 3D VAE 编码后 element-wise 加到 noisy input。

第二个改动很关键——相当于给 world model 一个 "草图"（FDM 的 low-fidelity 重建），让它 focus 在 refine 细节而不是从零 hallucinate 目标。这类似 ControlNet 的思路但更简单。

3D VAE 下采样：spatial $8\times 8 = 64\times$，temporal $4\times$。

---

## 4. Foundation Policy Model + Low-level Policy 两阶段

### 4.1 Stage 1: Foundation Policy Pretraining

输入：观测 $o_{1:t}$ + 文字描述 $s$，预测 latent action $a_t$（由 IDM 标注）。

架构：ST-transformer + frozen DINO-v2 ViT encoder + frozen CLIP text encoder。文本表示和观测表示 concat 后过 12 层 spatial/temporal attention。

**公式 (4)**：
$$L_{\text{policy}} = \|P([s; o_{1:t}]) - a_t\|^2$$

变量：
- $P(\cdot)$：policy 网络
- $[s; o_{1:t}]$：text embedding $s$ 和观测 token sequence 的 concatenation
- $a_t$：IDM 标注的 latent action 作为 ground truth

直觉：这是在 latent action 空间做行为克隆。因为 $a_t$ 是 quantized 到 codebook 的，predict 的是 continuous embedding，用 L2 loss 而非 cross-entropy——这是个折衷，便于 fine-tune 时和 low-level policy 对接。

### 4.2 Stage 2: Low-level Policy Finetuning

每个 latent action $a_t$ 对应 $\tau$ 个真实机器人 action $u_t^{1:\tau}$（paper 用 $\tau=3$）。低层 policy 把 latent action 当作"sub-task embedding"，concat 到观测 patch 上：

**公式 (5)**：
$$L_{ft} = \|P_f([s; P([s; o_{1:t}]); o_{1:t}]) - u_t^{1:\tau}\|^2$$

变量：
- $P_f$：低层 policy（只训练这部分，其他 frozen）
- $P([s; o_{1:t}])$：foundation policy 预测的 latent action，作为 sub-task embedding
- $u_t^{1:\tau}$：真实 7 维 action 序列（$\tau=3$ 步）

**核心直觉**: 高层 policy 在 "做什么 sub-task" 层次规划，低层 policy 在 "怎么具体执行" 层次 fine-grained 控制。latent action 是这两层之间的 interface。这种 hierarchy 让 foundation model 的预训练知识能 transfer 到具体机器人，只 fine-tune 一个 small head。

---

## 5. 实验设置与数据细节

### 5.1 数据组成

总共 **2.8M trajectories**：
- **0.8M robot trajectories**：来自 Open-X-Embodiment 的子集，**RT-1 留作 OOD 评估**，不用作训练
- **2.0M human video clips**：Something-Something v2, EGTEA, Epic Kitchen, Ego4D

数据混合权重参考 Octo 和 OpenVLA。Ego4D 占 32.1%，Something-Something v2 占 9.5%——human video 是大头。

参考：
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864  
- DROID: https://arxiv.org/abs/2403.12945  
- OpenVLA: https://arxiv.org/abs/2406.09246  
- Octo: https://arxiv.org/abs/2405.12213

### 5.2 数据预处理两个 trick

1. **Camera motion 过滤**：剔除约 40% 的 open-world video，因为相机抖动会引入与 agent action 无关的视觉变化，污染 latent action 学习。剩下的做 stabilization。
2. **Frame interval 调优**：
   - Robot data: 取 $s_t$ 和 $s_{t+3}$ 作为 image-goal pair（3 帧间隔）
   - Real-world video: 0.1s 到 0.5s 间隔

这是关键工程细节——帧太近没变化 latent action 学不到东西，帧太远变化太大模型学不动。

### 5.3 Fine-tuning 设置

- 用 RT-1 的 **1% episode**（low-data regime）
- 7 维 action: $\Delta\text{Pos}(3) + \Delta\text{Rot}(3) + \Delta\text{Grp}(1)$
- $\tau = 3$（每个 latent action 对应 3 个 7 维 action）
- 评估在 SIMPLER simulator 上做（https://arxiv.org/abs/2405.05941）

### 5.4 训练超参

| Model | Batch | Steps | LR |
|---|---|---|---|
| LAM | 512 | 140K | 1.5e-4 |
| World Model | 12 | 48K | 1e-4 |
| Foundation Policy | 128 | 124K | 1e-4 |
| Low-level Policy | 128 | 32K | 1e-4 |

World model 的 batch size 只有 12，是因为高分辨率视频生成显存吃紧。

---

## 6. 关键实验结果

### 6.1 Qualitative: Latent Action 一致性 (Figure 3)

在 OOD RT-1 上随机选 image-goal pair，retrieve latent action embedding 上最近的 3 个 pair。结果：相似 latent action 对应相似视觉变化，且这些相似 sub-task（"open gripper", "move left", "close gripper"）出现在不同的 raw language task 里——说明 latent action **被复用、被泛化**。

### 6.2 Cross-embodiment transfer (Figure 2)

这是 paper 最惊艳的实验：从 human video 提取 latent action，apply 到 robot arm 初始图上，world model 能 generate 出 robot arm 执行相同 sub-task 的视频。**Human → Robot transfer** 在 latent action 层面直接成立。

### 6.3 Multi-object control (Figure 4)

同一张初始图（有 apple/tennis/orange），apply 6 个不同 latent action，world model 能让**指定物体**移动——说明 latent action 学到了 object-centric 的控制，不只是全局 scene 变化。

### 6.4 Quantitative: SIMPLER 评估 (Figure 6a)

三个任务：Pick Coke Can, Move Near, Open/Close Drawer。baseline 是相同 ST-transformer 架构但没有 latent action embedding。IGOR 在 1% data 下 success rate **更高或持平**。

### 6.5 Latent action 对真实 action 的 predictiveness (Figure 6b)

X 轴：$\log(N)$，N 是 latent action embedding 空间的 nearest neighbor 数量。  
Y 轴：normalized std（真实 robot action 在 N 个邻居里的标准差 / 整个 RT-1 上的标准差）。

结果：N 越小 std 越低，且都 <1.0。说明**相似的 latent action 对应相似的真实 action**，latent action 确实 predictive。还发现 latent action 对 **movement** 比 **rotation/gripper** 更 predictive——说明 IGOR 学到的 action space 里 movement 信息更密集。

### 6.6 Ablation: Robot data only vs Mixed (Table 2)

在 RT-1 上做 OOD validation loss：
- Robot only: 0.145
- Robot + Human: **0.112** (↓ 22.8%)

加 human video 大幅降低 OOD loss——证明 human data 通过 IGOR 框架确实能帮助 robot task。

---

## 7. 与相关工作的精细对比

### 7.1 vs Genie / LAPO

Genie (https://arxiv.org/abs/2401.15445) 和 LAPO (https://arxiv.org/abs/2312.10812) 也学 latent action，但只在 2D 平台游戏上，latent action 对应具体按钮。IGOR 处理开放世界，latent action 不对应任何具体 underlying action——这点根本不同。

### 7.2 vs VPT

VPT (https://arxiv.org/abs/2206.11795) 用 IDM 从**交互数据**学真实 action label，需要 labeled interaction data。IGOR 完全 unsupervised，不需要 action label，这是它能 scale 到 internet video 的关键。

### 7.3 vs UniPi

UniPi (https://arxiv.org/abs/2302.00111) 先预测 goal image 再用 IDM 推 action。IGOR 先预测 latent action 再用它 specify goal。paper 论证：在 latent action 空间做 forward prediction 比在 image 空间好，因为 (a) 可以做 sub-task understanding，(b) 压缩后的 latent 更好预测。

### 7.4 vs RT-2 / OpenVLA

RT-2 (https://arxiv.org/abs/2307.15818) 和 OpenVLA 用 internet text + VQA 数据训 VLM。IGOR 用 internet video + text label。一个是"看图说话"，一个是"看人/机器人做事"——后者更接近 embodied AI 的本质。

### 7.5 vs iVideoGPT / SiamMAE / Voltron

这些都是 image-goal representation learning：
- SiamMAE (https://arxiv.org/abs/2305.14344)：siamese encoder 学视觉对应
- Voltron (https://arxiv.org/abs/2302.12766)：language-guided 表示
- iVideoGPT (https://arxiv.org/abs/2405.15223)：image-conditioned goal representation 作为 state

IGOR 的独特点是把 image-goal 表示**作为 latent action / sub-task**——不仅是 state representation，而是 action 层面的控制单元。

---

## 8. 局限性与未来工作

paper 自陈的 limitation：

1. **无法区分 visual change 来源**：camera shake / 其他 agent（如狗）/ 自己的动作。mitigation：用 ego-centric video + stabilization + 排除其他 agent。
2. **数据/模型 scaling** 没到极致。
3. **Future work**：加 object segmentation、tune world model 到真实场景、扩展到 multi-agent。

我个人补充的几个思考方向：
- Latent action 是 $N=4$ token × $|C|=32$——这个 codebook 太小，可能限制了对连续精细动作（如 rotation）的表达能力。Figure 6b 也显示 rotation/gripper 的 predictiveness 较弱，可能和这个 bottleneck 有关。
- World model 用 Open-Sora base 是一个工程妥协。Sora 这类模型对物理一致性其实并不强（参考 Sora 技术报告里的 failure cases: https://openai.com/research/video-generation-models-as-world-simulators），IGOR 用 FDM 输出作 conditioning 部分缓解了这个问题，但 long roll-out 误差累积还在。
- "Latent action 跨 embodiment 通用" 这个 claim 主要在 qualitative 上展示，quantitative 的 cross-embodiment transfer（比如真机部署 human→robot）还没做。

---

## 9. 给 Karpathy 的核心 takeaway

如果用你常用的话总结：IGOR 本质上是在做 **"next-token prediction in latent-action space"**——把视频帧之间的"delta"显式 quantize 成离散 token，从而让 foundation policy 可以像 LLM 一样在 token 空间做自回归预测，同时这个 token space 是 embodiment-agnostic 的。

三个最值得借鉴的技术点：

1. **IDM/FDM 用不同 cropping + single-frame FDM**——这是防止 representation collapse 到 shortcut 的关键技巧，可以推广到任何 contrastive/predictive learning。
2. **FDM 输出作为 world model 的 element-wise conditioning**——比 cross-attention 更强，相当于给生成模型一个"先验草图"。在 SD3/Flux 上也见过类似思路（https://arxiv.org/abs/2403.03206）。
3. **Latent action 作为高层-低层 policy 的 interface**——这种 hierarchical decomposition 让 foundation model 的预训练可以 freeze，只 fine-tune 一个 small low-level head，是 low-data regime 下的范式。

主要参考链接汇总：
- Project page: https://aka.ms/project-igor  
- Open-Sora: https://github.com/hpcaitech/Open-Sora  
- Rectified Flow: https://arxiv.org/abs/2209.03003  
- Stable Diffusion 3 (SD3): https://arxiv.org/abs/2403.03206  
- Genie: https://arxiv.org/abs/2401.15445  
- VPT: https://arxiv.org/abs/2206.11795  
- UniPi: https://arxiv.org/abs/2302.00111  
- RT-2: https://arxiv.org/abs/2307.15818  
- OpenVLA: https://arxiv.org/abs/2406.09246  
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864  
- DROID: https://arxiv.org/abs/2403.12945  
- SIMPLER: https://arxiv.org/abs/2405.05941  
- Sora: https://openai.com/research/video-generation-models-as-world-simulators  
- iVideoGPT: https://arxiv.org/abs/2405.15223  
- SiamMAE: https://arxiv.org/abs/2305.14344  
- Voltron: https://arxiv.org/abs/2302.12766  
- LAPO: https://arxiv.org/abs/2312.10812  
- Ego4D: https://arxiv.org/abs/2110.02558 (Ego4D paper)  
- Something-Something v2: https://arxiv.org/abs/1706.04261  
- Epic-Kitchens: https://arxiv.org/abs/1804.02748
