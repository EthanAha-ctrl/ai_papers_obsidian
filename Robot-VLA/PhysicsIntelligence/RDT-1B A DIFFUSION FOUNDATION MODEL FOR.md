---
source_pdf: RDT-1B A DIFFUSION FOUNDATION MODEL FOR.pdf
paper_sha256: fabfa885b63f2d68b8e7868549563c346d4323a596d24ee3e43d0a62b4ef9667
processed_at: '2026-08-11T21:00:38-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RDT-1B：大白话拆解

Karpathy，咱用最直白的话把这篇 paper 捋一遍。核心直觉就是：把大模型那套 scaling laws 硬核搬到双臂机器人上，但是因为机器人数据的物理特性跟图像视频完全不一样，直接套用现成的 architecture 会死得很惨，所以必须得做几个“反直觉”的魔改。

---

## 1. 一句话概括

这篇 paper 做了啥？它造了一个 1.2B 参数的 Diffusion Transformer foundation model，专门用来做双臂机器人的 imitation learning。为了让它能 work，他们解决了两个大坑：一是双臂动作的 multi-modality（同一任务有多条路径，直接回归会得到一个废动作），二是跨机器人 pre-training 时的数据异构问题。

---

## 2. 痛点：为啥双臂机器人难搞？

单臂机器人抓杯子，手伸过去抓就行了。双臂呢？抓一个 cube，左手先动可以，右手先动可以，两只手同时上也可以。这在统计学上叫 action distribution 是 multi-modal 的（多峰分布）。

如果你用最传统的 MSE 回归方法去训 policy，模型会学到什么？它会学到所有路径的算术平均数。想象一下，一条路径是左手向左伸，另一条是右手向右伸，平均下来就是两只手都往中间伸，结果谁都没抓到，机械臂还可能撞在一起。这就是为什么传统的 deterministic policy 在双臂任务上完全不行。

以前的人怎么解决呢？用 VAE（比如 ACT 模型），或者把连续动作离散化（比如 OpenVLA）。但这俩都有问题：VAE 表达能力不够，离散化在高维 14-DoF 空间里会有 quantization error，精度不够。

---

## 3. 破局点 1：为啥是 Diffusion？

既然分布是多峰的，我们就需要一个能拟合复杂分布的模型。Diffusion model 就是干这个的。它在图像上很贵（要采样 1000 步），但是在机器人 action 上，因为维度低得多，采样 5 步就能搞定，成本可以忽略不计。

### 公式直觉拆解

Diffusion 的训练过程（公式 2）长这样：
$$\mathcal{L}(\theta) := \text{MSE}\left(\mathbf{a}_t, f_\theta(\ell, o_t, \tilde{\mathbf{a}}_t, k)\right)$$

- $\mathbf{a}_t$：真实的 clean action（机械臂应该执行的完美动作）。
- $f_\theta$：你的神经网络。
- $\ell, o_t$：language instruction 和当前观察。
- $\tilde{\mathbf{a}}_t$：加噪后的 noisy action，等于 $\sqrt{\bar{\alpha}^k}\mathbf{a}_t + \sqrt{1-\bar{\alpha}^k}\epsilon$。这里 $k$ 是随机的 diffusion timestep，$\bar{\alpha}^k$ 是信号保留率，$\epsilon$ 是高斯噪声。
- 输出：网络直接预测 clean action $\mathbf{a}_t^0 \approx f_\theta(\ell, o_t, \tilde{\mathbf{a}}_t, k)$。

这里有个关键选择：网络是 **predict $x_0$**（直接预测干净动作），而不是像最早的 DDPM 那样 predict noise $\epsilon$。直觉上，在低维数据上直接预测目标比预测噪声要容易收敛得多，这也是参考了 Diffusion Policy 里的做法。

另外，他们不预测单步 action，而是预测一个 action chunk（一段长度为 $T_a=64$ 的连续动作序列 $\mathbf{a}_{t:t+T_a}$）。为啥？因为单步预测会导致误差累积，机械臂走几步就漂移出训练分布了。一次预测一段，能保持时间连贯性，机械臂不会抽搐。

---

## 4. 破局点 2：为什么直接拿图像的 DiT 来用会爆炸？

这是这篇 paper 最有技术含量的地方。作者选了 Diffusion Transformer (DiT) 作为 backbone，因为 Transformer 适合 scale up。但是，直接用原版 DiT，loss 会直接爆炸。为什么？因为机器人数据和图像数据的物理特性完全相反：

- **图像**：空间连续，像素值在 [0, 255] 之间，很稳定。
- **机器人数据**：极度非线性，碰到东西瞬间速度就变了（高频）；数值范围极不稳定（传感器可能有极端值）；绝对位置有意义（关节零位是绝对的，不能随便减掉）。

针对这些，作者做了三个“魔改”：

### 魔改 A：QKNorm + RMSNorm（解决数值爆炸）

原版 Transformer 用 LayerNorm。LayerNorm 有个操作叫 centering（减均值）。对于机器人来说，关节的绝对位置信息很重要，你把均值减掉，相当于把 DC component 拿掉了，这会破坏时间序列的对称性，导致 token shift 和 attention shift。

所以他们换成了 RMSNorm（公式看这）：
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$
注意这里 **没有减均值**，只是除以了 RMS（Root Mean Square）。$\gamma$ 是可学习的缩放参数。

同时，attention 里的 Q 和 K 做了 L2 normalization（QKNorm），防止点积过大导致数值溢出。Fig. 4a 实验证明，没这俩，大模型根本训不动。

### 魔改 B：MLP Decoder（解决非线性拟合）

原版 DiT 最后一层是个 linear layer，把 latent 投影回像素空间。图像像素之间的映射比较平滑，linear 够用。但机器人动作有强非线性（碰撞、阻尼），linear 根本拟合不了。

所以换成了 nonlinear MLP decoder。Fig. 4b 里，控制机器狗推摇杆的任务，没有 MLP decoder，成功率直接从 32% 掉到 0%，因为推摇杆差一毫米狗就跑偏了。

### 魔改 C：Alternating Condition Injection (ACI)（解决多模态打架）

条件输入（图像和文本）怎么注入到 Transformer 里？传统 DiT 用 adaptive layer norm，把条件压缩成一个 token。但这里图像有几百个 patch token，文本有几十个 token，信息量很大，压成一个全丢了。

所以用 cross-attention。但问题来了：图像 token 太多，文本 token 太少。如果每层都同时 cross-attend 两者，文本信息会被图像淹没，模型就不听人话了。

解决方案就是 ACI：奇数层只注入图像，偶数层只注入文本。交替进行。Fig. 4b 显示，没这个交替，"倒三分之一水"这种需要严格遵循指令的任务，成功率断崖式下跌。这个 insight 对以后做 VLA 模型太重要了：token budget 不平衡时，必须有机制去保护弱者。

---

## 5. 破局点 3：Unified Action Space（跨机器人训练的核心）

要训 foundation model，单臂那点数据完全不够。得用多机器人数据 pre-train。但是，不同机器人的 action space 长得千奇百怪：有的 6 DoF，有的 7 DoF，有的有夹爪，有的带移动底盘。

以前的处理方法是丢掉不一致的数据，或者只保留大家都有的特征。这太浪费了。作者的做法是：搞一个 128 维的“物理可解释统一动作空间”。

这 128 维怎么分配的？看 Table 4：
- [0, 50)：右臂的关节位置、速度、末端位姿等。
- [50, 100)：左臂的镜像。
- [100, 103)：底盘速度。
- 剩下的：Reserved。

对于一个具体的机器人，比如只有 7 DoF 单臂，就把它的 7 个关节位置填到 [0, 7) 里，[7, 10) 补 0，剩下维度全补 0。

这里有个绝妙的 trick：补 0 会不会让模型困惑？因为速度为 0 可能代表“静止”，也可能代表“没这个零件”。为了消除歧义，他们 concat 了一个 128 维的 0-1 mask 向量，指示哪些维度是真的，哪些是 padding 的。最终输入是 256 维。

另一个反直觉的点：他们**坚决不 normalize 数据到 [-1, 1] 或 $\mathcal{N}(0,1)$**。以前大家都这么干，但作者说：保留真实的物理单位（米、弧度等）。因为“1 米”在不同数据集里代表同一个真实长度，一旦 normalize，这种跨机器人的共享物理意义就没了。模型就学不到跨机器人的物理 prior 了。旋转表示用的是 6D representation（Zhou et al. 2019），避免了四元数的 gimbal lock 和不连续性问题。

---

## 6. 数据与规模

### Pre-training 数据
46 个数据集，1M+ 条轨迹，21TB。包括 Open X-Embodiment 里的 RT-1, DROID, RH20T 等。Sampling 权重用 $\sqrt{N_j}$（数据集 $j$ 的大小为 $N_j$），防止大数据集 dominate。

### Fine-tuning 数据
自己用 Mobile ALOHA 采了 6K+ 条轨迹，300+ 任务，100+ 物体，15+ 房间。为了增加 language 多样性，用 GPT-4-Turbo 把每条指令改写成 100 个扩展版 + 1 个简化版。训练时 1/3 概率取原始，1/3 取扩展，1/3 取简化。

---

## 7. 训练与推理细节

- 48 张 H100 80GB，pre-train 一个月跑 1M steps，fine-tune 三天跑 130K steps。
- 优化器：AdamW，learning rate $1 \times 10^{-4}$，bf16 混合精度。
- 推理加速：用 DPM-Solver++，把 1000 步 diffusion 压缩到 5 步。在 RTX 4090 上能跑到 6Hz 的 action chunk 频率（每秒输出 6 个动作序列），平均 action 频率 381Hz。
- 没用 Classifier-Free Guidance (CFG)。作者发现 CFG 在机器人控制上没卵用，反而让机械臂行为不稳定。这跟图像生成里 CFG 是标配完全不同，可能因为 action space 的多峰性不需要那种 explicit guidance。

---

## 8. 实验结果有多猛？

对比 baselines：ACT（VAE），OpenVLA（离散化，7B 参数），Octo（diffusion，93M 参数）。

- **Unseen Object/Scene**：RDT 在没见过的杯子和房间里，成功率 50-62.5%，baselines 基本全 0。
- **Instruction Following**：RDT 能听懂“倒三分之一水”这种没见过的词，成功率 100%，baselines 全 0。
- **1-Shot/5-Shot Learning**：只给 1 个 demo 学折短裤，RDT 68%，baselines 0%。
- **Dexterity**：控制机器狗走直线，RDT 32%，baselines 0%。

Ablation study 也很扎实：
- 去掉 diffusion 用回归：instruction following 从 100% 掉到 12.5%。
- 去掉 pre-training：unseen object 从 50% 掉到 0%。
- 把模型缩到 166M：unseen object 从 50% 掉到 37.5%。

有个很有意思的发现（Appendix H）：OpenVLA 和 Octo 在双臂数据上 fine-tune 根本不收敛，action accuracy 在 60% 左右震荡。只有按 task-specific 给几百条数据 fine-tune 才能勉强用。这说明它们 architecture 根本 hold 不住 14-DoF 的高维双臂动作空间。

---

## 9. 直觉联想与 Open Questions

1. **Architecture 适配数据分布**：这篇最大的 insight 就是，图像那套架构不能直接搬。Robotics data 是低维、非线性、绝对位置敏感的。RMSNorm 保 DC、MLP decoder 处理非线性、ACI 平衡多模态，这三个改动都是针对这些特性的精准打击。
2. **Unified Action Space 的通用性**：这套 128 维 padding 机制目前只测了 ALOHA。如果换成一个带灵巧手（20 DoF 手）的机器人，128 维还够不够？这种设计能不能扩展到 ANYmal 这种足式机器人？
3. **Diffusion 的 latency 瓶颈**：5 步 DPM-Solver 已经很快了，但 6Hz 对打乒乓球这种任务肯定不够。未来如果要做快速动态任务，可能得考虑 Consistency Models 或者 Flow Matching 把采样步数压到 1 步。
4. **Pre-training 数据偏置**：目前 pre-training 数据里双臂数据极少，绝大部分是单臂。模型在 pre-train 时学到的 prior 可能偏向单臂的 motion pattern。如果未来 Open X-Embodiment v2 里有大量双臂数据，这些 architectural trick 是否还需要，这是个 open question。

References:
- RDT-1B Paper: https://arxiv.org/abs/2410.07864
- Project Page: https://rdt-1b.github.io/
- Code: https://github.com/thu-ml/RoboticsDiffusionTransformer
- Diffusion Policy (Chi et al.): https://diffusion-policy.cs.columbia.edu/
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- 6D Rotation Representation: https://arxiv.org/abs/1812.07035
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

# RDT-1B 深度技术解析

非常荣幸为 Karpathy 详细拆解这篇 Tsinghua 的工作。这篇 paper 的核心 insight 是把 diffusion model + DiT scaling laws 直接搬到 bimanual manipulation 上，但关键 contribution 是对 robotic data 物理特性的若干"反直觉"修改。下面按 architecture → data → training → experiments 的顺序展开。

---

## 1. Problem Setup 与为什么是 Diffusion

**形式化**: 给定 language instruction $\ell$ 和 observation $o_t := (X_{t-T_{\text{img}}+1:t+1}, z_t, c)$，policy 需要输出 action $\mathbf{a}_t$ 控制双臂完成目标。其中 $X_{t-T_{\text{img}}+1:t+1}$ 是 RGB 历史帧序列，$z_t$ 是 proprioception，$c$ 是 control frequency。

**关键观察 — multi-modality**：双臂 manipulation 的 action distribution 是天然 multi-modal 的。Fig. 2b 给出了一个 toy example：用两只手抓一个 cube，可能左手先动、右手先动、两手同时动、或者一只手辅助另一只手 — 这些都是合法的 mode。如果用 deterministic regression $(\ell, o_t) \mapsto a_t$，模型会学到所有 mode 的 arithmetic mean，结果是完全 infeasible 的 action（比如两只手都伸到中间，谁都没抓到）。

**为什么选 diffusion 而非 VAE 或 discretization**：
- VAE (ACT 用的)：expressiveness 不足，latent space 难以 capture 所有 mode
- Discretization (OpenVLA, RT-2 用的)：quantization error，对于双臂 14-DoF 这种高维空间，bin 数量爆炸
- Diffusion：在 low-dim continuous control 上 sampling cost 很小（不像图像要 1000 steps，action 只需 5 steps via DPM-Solver++），expressiveness 强

---

## 2. Diffusion Formulation 细节

论文用的是 DDPM (Ho et al. 2020) 的 formulation，但训练目标是 **predict $x_0$ directly** 而非 predict noise $\epsilon$（这是 key choice，参考 Nichol & Dhariwal 的 improved DDPM）。

### Forward process (加噪)
对 clean action $\mathbf{a}_t$ 加噪得到 $\tilde{\mathbf{a}}_t$：
$$\tilde{\mathbf{a}}_t := \sqrt{\bar{\alpha}^k}\mathbf{a}_t + \sqrt{1-\bar{\alpha}^k}\epsilon$$

变量含义：
- $k \sim \text{Uniform}(\{1,...,K\})$：diffusion timestep，$K=1000$ 在训练时
- $\alpha^k$：noise schedule 第 k 步保留信号的比例
- $\beta^k := 1 - \alpha^k$：第 k 步添加的噪声比例
- $\bar{\alpha}^k := \prod_{i=1}^{k}\alpha^i$：累积保留比例，k 越大 $\bar{\alpha}^k$ 越小
- $\epsilon \sim \mathcal{N}(\mathbf{0}, I)$：采样噪声

### Reverse process (去噪，公式 1)
$$\mathbf{a}_t^{k-1} = \frac{\sqrt{\bar{\alpha}^{k-1}}\beta^k}{1-\bar{\alpha}^k}\mathbf{a}_t^0 + \frac{\sqrt{\alpha^k}(1-\bar{\alpha}^{k-1})}{1-\bar{\alpha}^k}\mathbf{a}_t^k + \sigma^k \mathbf{z}$$

变量含义：
- $\mathbf{a}_t^{k-1}$：第 k-1 步去噪后的 action
- $\mathbf{a}_t^k$：第 k 步的 noisy action
- $\mathbf{a}_t^0$：网络预测的 clean sample（不是真实 ground truth，是 $f_\theta$ 估出来的）
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ if $k > 1$，else $\mathbf{z} = \mathbf{0}$（最后一步不加噪声）
- $\sigma^k$：pre-defined by noise schedule

### Training objective (公式 2)
$$\mathcal{L}(\theta) := \text{MSE}\left(\mathbf{a}_t, f_\theta(\ell, o_t, \tilde{\mathbf{a}}_t, k)\right)$$

这里 $f_\theta$ 直接预测 clean action $\mathbf{a}_t^0 \approx f_\theta(\ell, o_t, \tilde{\mathbf{a}}_t, k)$。Intuition：相比 $\epsilon$-prediction，$x_0$-prediction 在 low-dim 数据上收敛更快，且更稳定（参考 Diffusion Policy, Chi et al. 2023）。

### Action Chunking
不预测单步 action，而是预测 chunk $\mathbf{a}_{t:t+T_a} := (\mathbf{a}_t, ..., \mathbf{a}_{t+T_a-1})$，其中 $T_a = 64$。这样：
- 减少 trajectory 中决策次数（缓解 covariate shift / error accumulation，参考 DAgger Ross et al. 2011）
- 鼓励 temporal consistency，避免相邻 action 跳变导致机械臂损坏

---

## 3. Architecture 三大关键修改

论文以 DiT (Peebles & Xie, 2023) 为 backbone，做了 **三项针对 robotic data 的关键修改**。这三点都通过 ablation 验证了必要性（Fig. 4）。

### 3.1 QKNorm + RMSNorm

**动机**：robotic physical quantities 有 unstable numerical range（传感器可能有 extreme values），加上大模型训练容易数值爆炸。

- **QKNorm** (Henry et al. 2020): 在 attention 的 $Q$ 和 $K$ 上加 L2 normalization
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q'K'^T}{\sqrt{d_k}}\right)V, \quad Q' = \frac{Q}{\|Q\|_2}, K' = \frac{K}{\|K\|_2}$$
  
  这把 attention logit 的 scale 控制住了，避免数值爆炸。

- **RMSNorm 替代 LayerNorm** (Zhang & Sennrich 2019):
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$
  
  注意 RMSNorm **没有 mean-subtraction (centering)**。论文的理由是：robotic data 类似 time series forecasting，centering 会造成 token shift 和 attention shift，破坏 time series 对称性（参考 Huang et al. 2024, UnitNorm paper）。

**Intuition**: LayerNorm 的 centering 相当于在 feature 维度上做高通滤波，把 DC component 拿掉。但对 robot joint position 这种 absolute 位置信息，DC component 本身就有物理意义（机械臂关节零位是 absolute 的，不是 relative 的）。所以 RMSNorm 保留 magnitude 信息更适合。

Fig. 4a 显示，没有这两个改动，pre-training loss 直接爆炸或剧烈震荡。

### 3.2 MLP Decoder

把 DiT 原本的 linear decoder 换成 nonlinear MLP decoder。

**动机**：robotic action 有强非线性（碰撞、约束、damping），linear projection 难以拟合。Image pixel 之间的映射相对 smooth（locality），但 joint acceleration、contact force 之间是高度非线性的。

Fig. 4b 的 Robot Dog 任务里，没有 MLP decoder 时，success rate 从 32% 掉到 0%。这个任务需要精确控制 joystick 角度，差一点就偏离直线，对非线性拟合要求极高。

### 3.3 Alternating Condition Injection (ACI)

**问题**：传统 DiT 用 adaptive layer norm (adaLN) 注入 class label 条件，但这里条件是 image tokens 和 text tokens，都是高维变长的。如果都压缩成 single token 会丢信息。

**方案**：用 cross-attention 注入条件。但发现 **image tokens 数量远多于 text tokens**（SigLIP 输出几百个 patch token，T5-XXL 输出几十个 text token），如果每层同时注入两者，text 信息会被 image 淹没。

**ACI 策略**：在 successive layers 的 cross-attention 中 **交替注入** image 和 text。比如奇数层注入 image，偶数层注入 text。

**Intuition**：这本质上是 modality balancing。如果不交替，language instruction following 能力会显著下降（Fig. 4b 中 Pour Water-L-1/3 的 correct amount 子任务，没有 ACI 时 success rate 严重下降）。这个 insight 对 VLA 模型设计很关键 — image 和 text 的 token budget 极度不平衡时，需要 explicit 的 balancing 机制。

---

## 4. Multi-Modal Input Encoding

### 4.1 Low-Dimensional Inputs
- Proprioception $z_t$, noisy action chunk $\tilde{\mathbf{a}}_{t:t+T_a}$, control frequency $c$, diffusion step $k$
- 用 MLP with **Fourier features** (Tancik et al. 2020) 编码

**Fourier features 的作用**：MLP 在低维输入上有 spectral bias，难以学习 high-frequency 函数。Fourier features $\gamma(x) = [\sin(2^0 \pi x), \cos(2^0 \pi x), ..., \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x)]$ 把输入 lift 到高维，让 MLP 能拟合 high-frequency changes（robotic data 的 collision、contact 时刻就有 high frequency）。

### 4.2 Image Inputs
- Frozen SigLIP encoder (Zhai et al. 2023)
- MLP projection 到 token space
- 多维 grid positional embedding: $(T_{\text{img}}, N_{\text{cam}}, N_{\text{patch}}, D)$
  - $T_{\text{img}} = 2$ (历史长度)
  - $N_{\text{cam}} = 3$ (exterior + right wrist + left wrist)
  - $N_{\text{patch}}$: ViT patch 数
  - $D$: embedding dimension

**为什么 freeze SigLIP**：节省 GPU memory（1.2B 模型 + 48 H100 80GB 已经很满）。SigLIP 的 image-text aligned 表示对 robotics 任务的 semantic understanding 有帮助。

### 4.3 Language Inputs
- Frozen T5-XXL (Raffel et al. 2020)
- MLP projection
- Language attention mask 处理 padding tokens

### 4.4 Stochastic Independent Masking
每个模态独立以 10% 概率被 mask。**防止 shortcut learning**：模型可能 over-rely on exterior camera（信息量大）而忽略 wrist camera（信息量小但关键）。Masking 强制模型学习从所有视角提取信息。

---

## 5. Physically Interpretable Unified Action Space

这是论文的 **第二大 contribution**，解决 cross-robot training 的 heterogeneity 问题。

### 设计思路
1. 同一个机器人的 proprioception $z_t$ 和 action $\mathbf{a}_t$ 共享一个空间（因为 action 通常是 desired $z_{t+1}$ 的 subset）
2. 设计一个 128 维的 unified space，涵盖所有 gripper-arm 机器人的主要物理量

### 128 维分配（Table 4）

| Index Range | 物理量 |
|---|---|
| [0, 10) | Right arm joint positions |
| [10, 15) | Right gripper joint positions |
| [15, 25) | Right arm joint velocities |
| [25, 30) | Right gripper joint velocities |
| [30, 33) | Right EEF positions |
| [33, 39) | Right EEF 6D pose |
| [39, 42) | Right EEF velocities |
| [42, 45) | Right EEF angular velocities |
| [45, 50) | Reserved |
| [50, 100) | Left arm 对应镜像 |
| [100, 102) | Base linear velocities |
| [102, 103) | Base angular velocities |
| [103, 128) | Reserved |

### Embedding 策略
对于具体机器人，按物理含义把原始 action 填入对应位置，其余位置 padding。单臂机器人映射到 "right" arm。

### 关键 trick: Padding indicator
直接 pad 0 有歧义（速度 0 可能表示静止，也可能是 padding）。所以额外 concat 一个 0-1 vector 表示每个维度是否 padded，最终输入是 256 维 = 128 (action) + 128 (availability mask)。

### 为什么不 normalize 到 [-1, 1] 或 $\mathcal{N}(0, 1)$
论文 explicit 地说：保留真实物理单位，因为 "1 (m)" 在不同 dataset 对应同一个真实长度，normalize 会破坏这种 cross-robot shared property。这与 Chi et al. 和 Ghosh et al. 的做法相反。

### EEF 6D rotation representation
用 6D representation (Zhou et al. 2019) 而非 quaternion 或 Euler，避免 gimbal lock 和 continuity 问题。

---

## 6. 数据规模

### Pre-training 数据
- **46 datasets, 1M+ trajectories, 21TB**
- 包括 RT-1, DROID, RH20T, Mobile ALOHA, BridgeData V2, Open X-Embodiment 等
- Sampling weight: $\sqrt{N_j}$ for dataset $j$ with size $N_j$，防止大数据集 dominate
- 数据清洗：去重复 episodes、失败 episodes、blank images、过短 trajectory、过长 trajectory 下采样

### Fine-tuning 数据
- **6K+ trajectories, 3M+ frames, 300+ tasks**
- 用 Mobile ALOHA robot 采集
- 100+ objects (rigid + non-rigid)
- 15+ scenes, 不同光照
- GPT-4-Turbo 扩展指令：每个 task 生成 100 个扩展指令 + 1 个简化指令
- 训练时 1/3 概率取原始指令，1/3 取扩展，1/3 取简化

---

## 7. Training 细节

### 硬件与时长
- 48 H100 80GB GPUs
- Pre-training: 1M steps, ~1 个月
- Fine-tuning: 130K steps, ~3 天
- **重要**: fine-tune 是从 500K checkpoint 开始，不是 1M（scheduling 原因）

### Hyper-parameters (Table 10)
- Batch size: 32 × 48 = 1536
- Learning rate: $1 \times 10^{-4}$, constant scheduler
- Optimizer: AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $1 \times 10^{-2}$
- Mixed precision: bf16
- Warm-up: 500 steps

### Architecture (Table 9)
- 28 layers, hidden size 2048, 32 heads
- 1.2B params
- Language token dim: 4096 (T5-XXL output)
- Image token dim: 1152 (SigLIP output)
- RDT token dim: 2048

### Diffusion scheduler
- Training: DDPM with **glide cosine scheduler** (squaredcos cap v2), 1000 steps
- Sampling: **DPM-Solver++** (Lu et al. 2022) with glide cosine, **5 steps**
- 推理速度: 6 Hz action chunks, 381 Hz actions on RTX 4090 24GB

### Data Augmentation
- Image: color jitter, image corruption
- Proprioception: Gaussian noise, SNR = 40dB
- Language: GPT-4-Turbo 扩展

### Monitoring trick
论文发现 training set 上 sampled action vs ground truth 的 MSE 与 real robot deployment 性能 **正相关**。当 MSE 收敛时可以停止训练。但 MSE 太低可能 overfitting。

### 重要: 没有 Classifier-Free Guidance (CFG)
论文说 CFG 没有提升性能，反而导致 unstable robot arm behavior。这与 image generation 中 CFG 几乎是标配不同 — 可能因为 action space 的 multi-modality 不像 image 那么需要 explicit guidance。

---

## 8. 实验结果深度分析

### 8.1 主结果 (Table 3)

7 个任务覆盖 5 个维度：
- **Unseen Object** (Wash Cup): RDT 在 unseen cup 上 50% success，ACT/OpenVLA/Octo 几乎全 0
- **Unseen Scene** (Pour Water): RDT 在 3 个 unseen rooms 上 62.5% total，baselines 全部失败
- **Instruction Following** (Pour Water-L-1/3, -R-2/3): RDT correct amount 100%/75%，baselines 全 0
- **5-Shot Learning** (Handover): RDT 56% (从 Table 3 数字估算)，baselines ~0
- **1-Shot Learning** (Fold Shorts): RDT 68%，baselines ~0
- **Dexterity** (Robot Dog): RDT 32%, ACT 0%, 其他全 0

整体比 baselines 提升 **56%** success rate。

### 8.2 Ablation (Table 2)

| Variant | Unseen Object | Unseen Scene | Instruction Following |
|---|---|---|---|
| RDT (regress) - no diffusion | 12.5 | 50 | 12.5 |
| RDT (small) - 166M params | 37.5 | 62.5 | 25 |
| RDT (scratch) - no pre-training | 0 | 25 | 62.5 |
| RDT (ours) | 50 | 62.5 | 100 |

**关键 insight**：
- Diffusion 对 instruction following 至关重要（12.5 → 100），因为 instruction 决定的 action mode 是离散选择
- Pre-training 对 unseen object/scene generalization 至关重要（0 → 50），因为需要 prior knowledge
- Model size 影响中等，但 small model 在 instruction following 上明显不行（25 → 100）

### 8.3 为什么 OpenVLA 和 Octo fine-tune 不收敛

Appendix H 提到一个重要现象（Fig. 8, 9）：
- OpenVLA full fine-tune：action token accuracy 在 60% 附近震荡，不收敛
- Octo full fine-tune：test MSE 在 $10^{-1}$ 附近震荡，不收敛
- 只有 per-task fine-tune（~100 episodes）才能达到 deployment 要求（95% accuracy / $10^{-3}$ MSE）

**Intuition**: 这暗示 OpenVLA 和 Octo 的 architecture 不适合 bimanual 14-DoF 高维 action space。OpenVLA 用 discretization，bin 数量爆炸；Octo 的 diffusion head 容量太小（93M params）。这两个 baseline 在双臂设置下其实有 fundamental limitation。

---

## 9. 我的 Intuition 拆解

### 9.1 为什么这个工作重要
1. **Scaling laws for robotics**: 第一次在 robotic manipulation 上验证了 diffusion + Transformer + large data 的 scaling 路径，类比 GPT-3 之于 NLP
2. **Architecture matters for robotics**: DiT 直接搬过来不行，需要针对 robotic data 物理特性做修改（RMSNorm 保 DC、MLP decoder 处理非线性、ACI 平衡 modality）
3. **Unified action space 设计**: physical interpretability 比 naive normalization 好，这是 cross-robot transfer 的关键

### 9.2 局限与 open questions
1. **Pre-training data 偏 single-arm**: 46 个 dataset 里大部分是单臂，双臂只有 ALOHA 和 Mobile ALOHA。Pre-training 学到的 prior 可能偏向单臂 motion pattern
2. **Evaluation 只有 ALOHA**: 没在其他双臂平台验证，cross-embodiment 能力未证明
3. **Sample efficiency 未量化**: 1-shot 和 5-shot 都测了，但没有 learning curve 显示需要多少 shots 才能稳定
4. **Failure mode 分析缺失**: 没有 qualitative analysis 失败案例
5. **Diffusion sampling latency**: 6 Hz action chunks 对快速任务可能不够（比如 ping-pong），但 paper 没讨论
6. **Long-horizon planning**: 任务都是 short-horizon（<1 分钟），没有 long-horizon 评估

### 9.3 与其他 foundation model 对比

| 模型 | Size | Action modeling | Pre-training data | Bimanual |
|---|---|---|---|---|
| RT-1 | 35M | Discretization | 130K | No |
| RT-2 | 55B | Discretization (VLM) | Web + robotics | No |
| OpenVLA | 7B | Discretization | Open X-Embodiment | No (fine-tune fails) |
| Octo | 93M | Diffusion | Open X-Embodiment subset | No (fine-tune fails) |
| ACT | ~1M | VAE | - | Yes (task-specific) |
| **RDT-1B** | **1.2B** | **Diffusion** | **1M+ trajectories** | **Yes** |

### 9.4 与 Diffusion Policy (Chi et al. 2023) 关系
Diffusion Policy 用 U-Net backbone，~10M params，task-specific。RDT 把它 scale up 100×，换成 Transformer backbone，加 pre-training，从 task-specific 变成 foundation model。

### 9.5 关于 6D rotation representation
Zhou et al. 2019 的 6D 表示是用 rotation matrix 的前两列（6 个数），比 quaternion（4 个）和 Euler（3 个）在连续性上都好。Neural network 学 6D 表示更稳定，避免 gimbal lock。这个细节在 robotics policy 里很关键，但很多 VLA paper 忽略。

---

## 10. References & 进一步阅读

**核心论文**:
- RDT-1B paper: https://arxiv.org/abs/2410.07864
- Project page: https://rdt-1b.github.io/
- Code: https://github.com/thu-ml/RoboticsDiffusionTransformer

**Architecture references**:
- DiT: https://arxiv.org/abs/2212.09748
- Diffusion Policy (Chi et al.): https://diffusion-policy.cs.columbia.edu/
- DDPM: https://arxiv.org/abs/2006.11239
- Improved DDPM (predict $x_0$): https://arxiv.org/abs/2105.05233
- QKNorm: https://arxiv.org/abs/2010.04245
- RMSNorm: https://arxiv.org/abs/1910.07467
- Fourier Features: https://arxiv.org/abs/2006.10739
- 6D rotation representation: https://arxiv.org/abs/1812.07035

**Robotics foundation models**:
- ALOHA: https://arxiv.org/abs/2304.13705
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- OpenVLA: https://openvla.github.io/
- Octo: https://octo-models.github.io/
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- Open X-Embodiment: https://robotics-transformer-x.github.io/

**Diffusion sampling acceleration**:
- DPM-Solver++: https://arxiv.org/abs/2211.01095

**Pre-training datasets**:
- DROID: https://droid-dataset.github.io/
- RH20T: https://rh20t.github.io/
- BridgeData V2: https://rail-berkeley.github.io/bridgedata/
- RT-1 dataset: https://robotics-transformer1.github.io/

**相关 Tsinghua 工作**:
- ViT backbone for diffusion (U-ViT, Bao et al.): https://arxiv.org/abs/2209.12152

---

## 11. 总结

RDT-1B 是一个 milestone 性的工作，把 NLP/CV 的 scaling laws 真正带到了 bimanual manipulation。三个 architectural modification (QKNorm+RMSNorm, MLP decoder, ACI) 都是针对 robotic data 物理特性的精心设计，不是随便搬过来的。Unified action space 的 physical interpretability 思路很可能成为未来 cross-robot foundation model 的标准。

最大的 open question 是：当 pre-training data 里双臂占比提升（比如未来 Open X-Embodiment v2 有更多双臂数据），这些 architectural modification 是否还需要？以及能否进一步 scale 到 10B+ params，是否会遇到新的 instability？这些都值得期待后续工作。

希望这个拆解对 build intuition 有帮助，Karpathy！如果想深入任何一个细节（比如 DPM-Solver++ 的数学推导、SigLIP vs CLIP 的差异、或者具体某个 task 的 failure mode），可以继续讨论。
