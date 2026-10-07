---
source_pdf: HUMAN MOTION DIFFUSION MODEL.pdf
paper_sha256: cfe20558dfb4670e9a23e52cff289bb305ebace9b079333e9541e3dfa2b2669c
processed_at: '2026-08-19T11:40:20-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MDM 用人话说

OK 我重新来，把技术 jargon 翻译成 intuition。

**参考链接**：
- Paper: https://arxiv.org/abs/2209.14991
- Project: https://guytevet.github.io/mdm-page/
- Code: https://github.com/GuyTevet/motion-diffusion-model
- DDPM: https://arxiv.org/abs/2006.11239

---

## 1. 这篇 paper 在搞什么

让 computer 从文字（"一个人走路然后摔倒"）生成一段 3D 人体动作。

听起来简单，实际上很难。难在哪？

**Motion 是个 many-to-many 问题**。你说"kick"，可能是足球踢法、空手道踢法、高踢、低踢、踢向左、踢向右……同一个描述对应一大堆合理动作。反过来，同一个踢腿动作，可以用很多种文字描述。

之前的主流方法（T2M, TEMOS, MotionCLIP, JL2P）都基于 **VAE**。VAE 的哲学是：把所有 motion 压进一个 Gaussian 分布的 latent space，然后从 Gaussian 采样解码。问题来了——Gaussian 是 single-peak 的、平滑的、unimodal 的。你硬把 multimodal 的真实分布塞进 Gaussian，结果就是**生成的动作都是"平均脸"**——所有 kick 都长得差不多，那种 average karate-soccer 混合体。

Diffusion models 不做这种假设。它直接学"怎么从噪声里一步步还原出真实数据"，能表达任意形状的分布。这就是为什么 MDM 选 diffusion。

---

## 2. Diffusion 怎么工作——直觉版

想象你把一张照片慢慢加 noise，加 1000 步，最后变成纯噪声。训练时让神经网络学会"给定第 t 步的 noisy 图，预测原图长什么样"。

生成的时候反过来：从纯噪声出发，一步步 denoise，1000 步后得到干净样本。

每一步加多少 noise 由 **noise schedule** $\alpha_t$ 决定（公式 1）：

$$q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\, x_{t-1},\; (1-\alpha_t) I)$$

这里 $\sqrt{\alpha_t}$ 控制"保留多少原 signal"，$(1-\alpha_t)$ 控制"加多少 noise"。$\alpha_t$ 从接近 1 衰减到接近 0。MDM 用 cosine schedule，$T=1000$。

---

## 3. 最关键的设计决策：Predict Sample, Not Noise

标准 DDPM 让网络预测 noise $\epsilon$：

$$\hat{\epsilon} = G(x_t, t)$$

MDM 让网络直接预测最终干净样本 $\hat{x}_0$：

$$\hat{x}_0 = G(x_t, t, c)$$

对应的 simple loss（公式 2）：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, t} \left[ \| x_0 - G(x_t, t, c) \|_2^2 \right]$$

- $x_0$：真实干净 motion
- $t$：diffusion 步数
- $c$：condition（CLIP 文本 embedding）
- $G$：transformer

### 为什么这个选择颠覆一切？

在 image domain 大家普遍认为 predict noise 更稳定。为什么？因为 noise 是 isotropic Gaussian，分布简单，loss surface 平滑。predict sample 方差大，收敛慢。

但 motion 不一样。Motion 的维度低（一帧 ~66 维 vs image ~50万 维），方差问题没那么严重。更关键的是——**只有在 sample space 才能施加 motion-specific 几何约束**。

你想加"脚不能贴地滑动"这个约束（foot sliding 是 animation 里超级明显的 artifact）。如果你预测的是 noise，noise 本身没有任何 motion 结构，foot contact loss 加在 noise 上毫无意义。但如果你预测 $\hat{x}_0$，每一步输出都是"对真实动作的猜想"，你就可以直接拿 forward kinematics 把它转成 joint positions，然后检查"接触地面的脚速度是不是零"。

这个 design choice 打开了 "diffusion + motion domain knowledge" 的大门。**这是整篇 paper 的灵魂**。

---

## 4. 三个 Geometric Losses

### 4.1 Position Loss（公式 3）

$$\mathcal{L}_{\text{pos}} = \frac{1}{N} \sum_{i=1}^N \left\| FK(x_0^i) - FK(\hat{x}_0^i) \right\|_2^2$$

- $i$：frame index
- $FK(\cdot)$：forward kinematics，把 joint rotations 转成 joint positions
- $x_0^i$：第 $i$ 帧 GT

如果你预测的是 rotations（更物理），但约束要作用在 positions 上，FK 就是那个桥梁。

### 4.2 Foot Contact Loss（公式 4）—— 防止 foot sliding

$$\mathcal{L}_{\text{foot}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left\| \left( FK(\hat{x}_0^{i+1}) - FK(\hat{x}_0^i) \right) \cdot f_i \right\|_2^2$$

- $f_i \in \{0,1\}^J$：第 $i$ 帧的 foot contact mask，脚接触地面时对应位是 1
- $FK(\hat{x}_0^{i+1}) - FK(\hat{x}_0^i)$：相邻帧位置差 = velocity

人话：**当脚踩在地上时，它的速度必须为零**。否则你会看到角色像在冰面上滑冰一样脚贴地滑动——这是 motion generation 里最经典的破绽。

### 4.3 Velocity Loss（公式 5）—— 防止 jitter

$$\mathcal{L}_{\text{vel}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left\| (x_0^{i+1} - x_0^i) - (\hat{x}_0^{i+1} - \hat{x}_0^i) \right\|_2^2$$

- $x_0^{i+1} - x_0^i$：GT 的 velocity
- $\hat{x}_0^{i+1} - \hat{x}_0^i$：预测的 velocity

人话：**每帧单独位置对了还不够，帧间的运动趋势也要对**。否则你会看到角色像得了帕金森一样帧间抖动。

### 4.4 总 loss（公式 6）

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\lambda} \mathcal{L}_{\text{vel}} + \lambda_{\text{foot}} \mathcal{L}_{\text{foot}}$$

$\lambda$ 是权重超参。

---

## 5. 架构：为什么用 Transformer 不用 U-Net

所有主流 diffusion model（DDPM, GLIDE, Stable Diffusion, Imagen）都用 **U-Net**。U-Net 为 image 设计——convolution 假设 spatial locality，down/up-sampling 假设 hierarchical spatial structure。

Motion 不是 spatial grid。Motion 是 **temporal sequence of joints**，joints 之间是 skeletal hierarchy 不是 grid 邻接。动作长度变化大（20 帧到 500 帧）。

Transformer 的 self-attention 天生处理 temporal + arbitrary token relations，更契合 motion 的本质。

### 具体架构

输入组装：
1. diffusion step $t$ → FFN → dimension $d$
2. condition $c$（CLIP text embedding）→ FFN → dimension $d$
3. 两者相加 → conditioning token $z_{tk}$
4. 每个 frame $x_t^i \in \mathbb{R}^{J \times D}$ flatten + 线性投影 + positional embedding → frame token

送进 transformer encoder：
- Tokens = $[z_{tk}, \text{frame}_1, ..., \text{frame}_N]$
- 8 层 self-attention
- 第一个 output token 丢弃（对应 $z_{tk}$）
- 剩下 $N$ 个 token 投影回 $\mathbb{R}^{J \times D}$ = $\hat{x}_0$

参数：batch 64，8 层，latent dim 512，500K steps，单张 RTX 2080 Ti 训 3 天。**轻得离谱**。

---

## 6. Classifier-Free Guidance——多样性 vs 准确性的旋钮

训练时 10% 概率把 condition 替换成 $\emptyset$，模型同时学 $p(x_0|c)$ 和 $p(x_0)$。

Sampling 时（公式 7）：

$$G_s(x_t, t, c) = G(x_t, t, \emptyset) + s \cdot \left( G(x_t, t, c) - G(x_t, t, \emptyset) \right)$$

- $G(x_t, t, \emptyset)$：unconditioned 预测，给"平均走向"
- $G(x_t, t, c) - G(x_t, t, \emptyset)$：conditioning direction
- $s$：guidance scale

直觉：$s=1$ 就是正常 conditional sampling。$s>1$ 沿 condition 方向外推，"用力推向 text 描述"，accuracy 升但 diversity 降。$s<1$ 更随机。$s=0$ 完全 unconditioned。

论文 sweep 出 $s=2.5$ 是 sweet spot。Figure 4b 展示了 FID 和 R-precision 随 $s$ 的 trade-off 曲线。

---

## 7. Editing——白送的福利

Diffusion 的 inpainting 思路直接搬到 motion，**不需要训练**。

### Temporal In-Betweening
固定动作的开头 25% 和结尾 25%，让模型生成中间 50%。

每步 sampling：
1. 模型预测 $\hat{x}_0$
2. 用 GT 的 prefix/suffix **overwrite** $\hat{x}_0$ 对应帧
3. 把 overwrite 后的 $\hat{x}_0$ 重新加噪到 $x_{t-1}$
4. 重复

### Body Part Editing
固定不想动的 joints（如下半身），让模型按新 text 重新生成上半身。同样 overwrite 机制，mask 在 joint 维度。

VAE 类方法做 editing 要专门设计 latent space（如 MotionCLIP 用 CLIP space）。Diffusion 天然支持，因为 sampling 过程就是"给定约束 denoise"。

---

## 8. 实验数字直觉版

### Text-to-Motion (HumanML3D, Table 1)

| Method | FID ↓ | Diversity → | MultiModality ↑ | R-Precision ↑ |
|---|---|---|---|---|
| Real | 0.002 | 9.503 | – | 0.797 |
| T2M | 1.067 | 9.188 | 2.090 | **0.740** |
| **MDM** | **0.544** | **9.559** | **2.799** | 0.611 |

- **FID 0.544**：T2M 的一半，分布距离大幅降低
- **Diversity 9.559**：甚至略高于 Real (9.503)，说明没压缩多样性
- **MultiModality 2.799**：远超 T2M 的 2.090，这就是 many-to-many 的直接体现——同一个 prompt 下 MDM 生成的动作分布更宽
- **R-Precision 0.611 vs T2M 0.740**：这是 MDM 的弱点。R-Precision 测"text-motion retrieval accuracy"，VAE 的 one-to-one 倾向让它更"精准"。MDM 用 fidelity 换了 diversity

### User Study (Figure 4a)

31 个用户 side-by-side 比较：
- MDM vs T2M：MDM 胜率约 60%
- **MDM vs Real motion：MDM 胜率 42.3%**

人话：**42% 的情况下人类觉得生成的动作比真实动作更像真实动作**。这数字惊人。

### Action-to-Motion (HumanAct12, Table 3)

| Method | FID ↓ | Accuracy ↑ |
|---|---|---|
| Real | 0.050 | 0.990 |
| ACTOR | 0.120 | 0.955 |
| INR | 0.088 | 0.973 |
| **MDM** | 0.100 | **0.990** |

Accuracy 0.990 跟 Real 持平。即使对手是专门为 action-to-motion 设计的方法（ACTOR, INR），通用框架 MDM 仍领先。

---

## 9. 一句话总结

VAE 把 motion 多模态分布压扁成 Gaussian，生成 average motion。MDM 用 diffusion 不假设分布形式，配合三个关键决策——**predict sample（而非 noise）让 geometric losses 可施加、transformer encoder（而非 U-Net）匹配 motion 的 temporal+per-joint 结构、classifier-free guidance 调节 diversity vs fidelity**——在单张 GPU 三天训练下达到 SOTA，42% 时候骗过人类评估者。

每个选择都由 motion domain 的结构属性反推出来。这就是 "carefully adapted" 的真正含义。

---

## 10. 局限和后续

**局限**：
- 推理慢：1000 步 forward pass，motion 维度小所以 ~1 分钟可接受
- R-Precision 输给 VAE 类（fidelity vs diversity trade-off）

**后续影响**：
- 启发了一大批 motion diffusion 工作：MotionDiffuse (https://arxiv.org/abs/2208.15001), MoFusion, OmniControl, FineMoGen, MotionLCM
- latent diffusion for motion 的方向（类似 Stable Diffusion 把 DDPM 移到 latent space）
- 至今仍是 baseline：训练便宜、代码清晰、通用、可编辑

Paper 的真正贡献：**证明 diffusion 在低维、temporal、几何约束丰富的 domain 里，需要为该 domain 重新设计 diffusion 的每个组件**。这个思路适用于任何把 diffusion 迁移到新 domain 的工作——audio, 3D shapes, protein, robot trajectories, 你 name it。

---

# Motion Diffusion Model (MDM) 深度解析

这篇 paper 是 Guy Tevet 等人 (Tel Aviv University) 在 2022 年 ICLR 2023 上的工作，把 diffusion models 第一次成功带到 human motion generation 域。我接下来从动机、设计决策、公式细节、架构、实验数据几个层面来 build 你的 intuition。

**资源链接**：
- Paper: https://arxiv.org/abs/2209.14991
- Project page: https://guytevet.github.io/mdm-page/
- Code: https://github.com/GuyTevet/motion-diffusion-model
- DDPM (Ho et al.): https://arxiv.org/abs/2006.11239
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- CLIP: https://arxiv.org/abs/2103.00020
- HumanML3D: https://github.com/Eric-Guo/webs

---

## 1. 为什么 motion generation 需要 diffusion？—— many-to-many 问题

Human motion generation 的核心难点是 **many-to-many mapping**：
- 同一个 text prompt "kick" 可以对应 soccer kick / karate kick / 高踢 / 低踢
- 同一个 kick 动作可以由多种 text 描述（"用左脚踢"、"猛烈踢"、"踢向右边"）

之前的 SOTA 方法（JL2P, TEMOS, T2M, MotionCLIP）大多基于 **auto-encoder** 或 **VAE**。VAE 的 latent space 是 **Gaussian**，本质上假设了一种 unimodal、平滑的潜空间分布，这种假设压抑了真实 motion 分布的多样性。Auto-encoder 更极端，是 deterministic one-to-one mapping。

Diffusion models 的关键优势：**不假设 target distribution 的形式**，通过 score matching / denoising 直接学习数据流形，自然能表达 many-to-many。这也是为什么 MDM 在 MultiModality 指标上能碾压 VAE 类方法（见下文 Table 1）。

---

## 2. 核心设计决策 #1：Predict Sample, Not Noise

这是 MDM 最关键也最反直觉的设计。标准 DDPM (Ho et al. 2020) 重新参数化后预测的是 noise $\epsilon_t$：

$$\hat{\epsilon} = G_\theta(x_t, t, c)$$

MDM 跟随 Ramesh et al. 2022 (DALL-E 2 / Hierarchical Text-Conditional Image Generation) 的路线，**直接预测 sample 本身**：

$$\hat{x}_0 = G_\theta(x_t, t, c)$$

对应的 simple loss（公式 2）：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0 \sim q(x_0|c),\; t \sim [1,T]} \left[ \| x_0 - G(x_t, t, c) \|_2^2 \right]$$

**变量含义**：
- $x_0$：从数据分布 $q(x_0|c)$ 采样的 clean motion（条件 $c$ 下的真实动作）
- $t$：从 $\{1, 2, \ldots, T\}$ 均匀采样的 diffusion time step
- $G$：transformer 模型（参数 $\theta$ 隐去）
- $x_t$：第 $t$ 步加噪后的 motion
- $c$：condition（CLIP text embedding / action class embedding / $\emptyset$）

### 为什么这个选择如此关键？

直觉：**只有在 sample space 才能施加几何约束**。

如果你预测 $\epsilon$，geometric losses 作用在 noise 上毫无物理意义（noise 是高维 isotropic Gaussian，没有任何 motion 几何结构）。但预测 $\hat{x}_0$ 后，模型每一步输出都是"对真实 motion 的猜想"，可以直接套用 forward kinematics、velocity、foot contact 这类 motion domain 久经验证的 geometric losses。

这个决策打开了"diffusion + domain knowledge"的结合通道。MDM 论文标题里 "carefully adapted" 就是指这种"为 motion 域重塑 diffusion"的工程努力。

### 副作用：sample prediction 通常被认为比 noise prediction 更难

在 image domain 中，predict-$x_0$ 通常方差更大、收敛更慢（见 DDPM 原文 $\mathcal{L}_{\text{simple}}$ 的讨论）。但 motion 的维度远小于 image（一帧 motion 约 $J \times D \approx 22 \times 3 \approx 66$ 维，而 image 是 $H \times W \times 3$），所以 sample prediction 的方差问题没那么严重，反而换来了 geometric losses 的可施加性。这是 motion domain 的 sweet spot。

---

## 3. Diffusion Framework 细节

前向 Markov noising process（公式 1）：

$$q(x_t^{1:N} \mid x_{t-1}^{1:N}) = \mathcal{N}\left( \sqrt{\alpha_t}\, x_{t-1}^{1:N},\; (1-\alpha_t)\, I \right)$$

**变量含义**：
- $x_t^{1:N}$：在 diffusion step $t$ 的完整 motion 序列（$1:N$ 表示帧 index 1 到 N）
- $\alpha_t \in (0, 1)$：常数超参，控制"保留多少原 signal + 加多少 noise"
- 当 $\alpha_t \to 0$，方差项 $(1-\alpha_t) \to 1$，$x_t$ 趋于纯噪声
- 当 $t = T$ 足够大，$x_T \sim \mathcal{N}(0, I)$（标准高斯）

逆向 sampling：从 $x_T \sim \mathcal{N}(0, I)$ 出发，迭代 $t = T \to 1$：
1. 模型预测 $\hat{x}_0 = G(x_t, t, c)$
2. 把 $\hat{x}_0$ 重新 "diffuse" 回 $x_{t-1}$（用 closed-form posterior）
3. 重复直至得到 $x_0$

注意 MD M 用 $T = 1000$ 步，cosine noise schedule（Nichol & Dhariwal 2021）。

---

## 4. 核心设计决策 #2：Geometric Losses

这是 MDM 把 motion domain 传统智慧嫁接到 diffusion 上的关键。论文实验了三种 geometric losses：

### 4.1 Position Loss（公式 3）

$$\mathcal{L}_{\text{pos}} = \frac{1}{N} \sum_{i=1}^{N} \left\| FK(x_0^i) - FK(\hat{x}_0^i) \right\|_2^2$$

**变量含义**：
- $i$：frame index（$1$ 到 $N$）
- $x_0^i$：第 $i$ 帧 GT motion（joint rotations 表示）
- $\hat{x}_0^i$：模型预测的第 $i$ 帧
- $FK(\cdot)$：Forward Kinematics 函数，把 joint rotations 转成 joint positions

**直觉**：当模型预测的是 rotations（更物理、更易 retarget），但训练数据中很多约束是定义在 positions 上的。FK 把 rotations 投影到 position space，在那里施加约束。如果直接预测 positions，$FK$ 就是 identity。

### 4.2 Foot Contact Loss（公式 4）

$$\mathcal{L}_{\text{foot}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left\| \left( FK(\hat{x}_0^{i+1}) - FK(\hat{x}_0^i) \right) \cdot f_i \right\|_2^2$$

**变量含义**：
- $f_i \in \{0, 1\}^J$：第 $i$ 帧的 binary foot contact mask（每 joint 一个 0/1，仅脚部 joints 会有 1）
- $FK(\hat{x}_0^{i+1}) - FK(\hat{x}_0^i)$：相邻帧的位置差 = 瞬时 velocity
- $\cdot$：element-wise product（广播到 D 维）

**直觉**：脚接触地面时（$f_i = 1$），velocity 应当为零。否则会出现经典的 **foot sliding** artifact（脚贴着地面滑动，看着像在水冰上走路）。

这个 loss 通过 zeroing out velocity when grounded 来强制物理约束，来自 Shi et al. 2020 (MotionNet)。

### 4.3 Velocity Loss（公式 5）

$$\mathcal{L}_{\text{vel}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left\| \left( x_0^{i+1} - x_0^i \right) - \left( \hat{x}_0^{i+1} - \hat{x}_0^i \right) \right\|_2^2$$

**变量含义**：
- $x_0^{i+1} - x_0^i$：GT 的相邻帧差 = GT velocity
- $\hat{x}_0^{i+1} - \hat{x}_0^i$：预测的 velocity

**直觉**：防止 jitter（抖动）。每帧单独位置对了（$\mathcal{L}_{\text{simple}}$ 保证），但帧间不连续就会抖。Velocity loss 强制模型在"运动趋势"层面也对齐。

### 4.4 总训练 loss（公式 6）

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}} + \lambda_{\text{foot}} \mathcal{L}_{\text{foot}}$$

**变量**：$\lambda_{\text{pos}}, \lambda_{\text{vel}}, \lambda_{\text{foot}}$ 是权重超参。

注意 HumanML3D 已经显式包含了 joint positions / velocities / foot contact labels 在 representation 中，所以 text-to-motion 实验里没单独加 geometric losses（被 $\mathcal{L}_{\text{simple}}$ 隐式覆盖）。Action-to-motion 用 rotation 表示时才显式启用。

---

## 5. 核心设计决策 #3：Transformer Encoder-Only Backbone

标准 diffusion（DDPM, GLIDE, Imagen, Stable Diffusion）几乎都用 **U-Net**（Ronneberger et al. 2015）。MDM 偏离这个惯例，用 **transformer encoder-only**（Vaswani et al. 2017）。

### 为什么 U-Net 不适合 motion？

U-Net 设计哲学是为 **spatial grid data** 优化的：convolution 假设 spatial locality + translation invariance，down/up-sampling 假设 hierarchical spatial structure。

Motion data 的本质是：
- **Temporal**：序列结构
- **Per-joint**：joints 之间有 skeletal hierarchy，但不是 spatial grid
- **Variable length**：动作长度从 20 帧到 500 帧不等

Transformer 的 self-attention 天生处理 temporal + arbitrary token relations，更契合。

### 架构 forward 流程

输入组装：
1. $t$（标量）经过一个 FFN 投影到 transformer dimension $d$
2. $c$（CLIP embedding，约 512 维）经过另一个 FFN 投影到 $d$
3. 两者相加得到 conditioning token $z_{tk} \in \mathbb{R}^d$
4. 每个 frame $x_t^i \in \mathbb{R}^{J \times D}$ flatten 成 $\mathbb{R}^{JD}$，线性投影到 $d$，加 sinusoidal positional embedding

送入 transformer encoder：
- Input tokens = $[z_{tk}, \text{frame}_1, \text{frame}_2, \ldots, \text{frame}_N]$
- 经过 $L=8$ 层 self-attention

输出处理：
- 第一个 token（对应 $z_{tk}$）丢弃
- 其余 $N$ 个 token 投影回 $\mathbb{R}^{J \times D}$ = $\hat{x}_0^{1:N}$

**关键架构参数**：
- Batch size = 64
- Layers = 8（GRU variant 用 2）
- Latent dimension = 512
- 500K training steps for text-to-motion
- 单卡 RTX 2080 Ti，3 天训完

---

## 6. Classifier-Free Guidance（公式 7）

训练时 10% 的样本随机把 $c$ 替换为 $\emptyset$，模型同时学 $p(x_0|c)$ 和 $p(x_0)$。

Sampling 时通过 scale $s$ 在两者间插值/外推：

$$G_s(x_t, t, c) = G(x_t, t, \emptyset) + s \cdot \left( G(x_t, t, c) - G(x_t, t, \emptyset) \right)$$

**变量含义**：
- $G(x_t, t, \emptyset)$：unconditioned prediction
- $G(x_t, t, c) - G(x_t, t, \emptyset)$：conditioning direction vector
- $s$：guidance scale

**几何直觉**：在 sample space 中，$G(\emptyset)$ 给出"平均走向"，$G(c)$ 给出"按 condition 走向"。两者的差就是"conditioning 方向"。$s > 1$ 沿这个方向外推，相当于"用力推向 condition"，fidelity 升高但多样性下降。$s < 1$ 则更随机。

论文 sweep 出 $s = 2.5$ 是 HumanML3D 上的 sweet spot（Figure 4b）。

**和 classifier-guided diffusion 的区别**：classifier guidance 需要单独训练一个 classifier $p(c|x_t)$，对噪声 $x_t$ 也要鲁棒。Classifier-free 把 condition 信息直接 bake 进 generative model，更简洁。

---

## 7. Editing via Diffusion Inpainting

无需训练，只在 sampling 时操作。

### Temporal In-Betweening
固定 motion 的 prefix 和 suffix（如前 25% + 后 25%），让模型生成中间 50%。

每步 sampling：
1. 模型预测 $\hat{x}_0$
2. 用 input 的 prefix/suffix **overwrite** $\hat{x}_0$ 的对应帧
3. 把 overwrite 后的 $\hat{x}_0$ 重新 noise 到 $x_{t-1}$
4. 重复

这相当于在每步"提醒"模型两端是固定的，中间要连贯。

### Body Part Editing（Spatial Inpainting）
固定不想改的 joints（如下半身），让模型按 text 重新生成其他 joints（如上半身）。

同样的 overwrite 机制，只是 mask 在 joint 维度而非 frame 维度。

这种"训练好后即用"的编辑能力是 diffusion 模型相比 VAE 的重要优势——VAE 的 latent space editing 要专门设计（如 MotionCLIP）。

---

## 8. 实验数据深度解读

### 8.1 Text-to-Motion: HumanML3D (Table 1)

| Method | R-Precision ↑ | FID ↓ | Multimodal Dist ↓ | Diversity → | MultiModality ↑ |
|---|---|---|---|---|---|
| Real | 0.797 | 0.002 | 2.974 | 9.503 | – |
| T2M | **0.740** | 1.067 | **3.340** | 9.188 | 2.090 |
| **MDM (encoder)** | 0.611 | **0.544** | 5.566 | **9.559** | 2.799 |
| MDM (decoder) | 0.608 | 0.767 | 5.507 | 9.176 | **2.927** |
| MDM (decoder+token) | 0.621 | 0.567 | 5.424 | 9.425 | 2.834 |
| MDM (GRU) | 0.645 | 4.569 | 5.325 | 7.688 | 1.264 |

**解读**：
- **FID = 0.544**，约为 T2M (1.067) 的一半，分布距离显著降低
- **Diversity = 9.559**，甚至略高于 Real (9.503)，说明 MDM 没有压缩多样性
- **MultiModality = 2.799**，远超 T2M (2.090) —— 这是 many-to-many 优势的直接体现：同一个 prompt 下生成的 motion 分布更宽
- **R-Precision 落后 T2M**（0.611 vs 0.740）：这是 MDM 的弱点。R-Precision 测的是"生成 motion 与 text 的匹配度（用 retrieval 模型评估）"，VAE 类方法因为 one-to-one 倾向，反而更"精准"。MDM 用 diversity 换了部分 fidelity。
- **GRU backbone FID 飙到 4.569**：架构确实重要，但 transformer 系（encoder/decoder/decoder+token）都在 0.5-0.8 之间，说明 diffusion 框架对 transformer 变体相对 robust

### 8.2 User Study (Figure 4a)

31 个用户，KIT 测试集上 side-by-side 比较：
- MDM vs T2M: MDM 胜率约 60%
- **MDM vs Real motion: MDM 胜率 42.3%**

这个数字惊人。人类评估者 42% 的情况下选生成样本而非真实样本，说明 perceptual quality 已接近 photorealistic motion 的水平。

### 8.3 Action-to-Motion: HumanAct12 (Table 3)

| Method | FID ↓ | Accuracy ↑ | Diversity → | MultiModality → |
|---|---|---|---|---|
| Real | 0.050 | 0.990 | 6.880 | 2.590 |
| ACTOR | 0.120 | 0.955 | 6.840 | 2.530 |
| INR | 0.088 | 0.973 | 6.881 | 2.569 |
| **MDM** | 0.100 | **0.990** | 6.860 | 2.520 |
| MDM w/o foot | **0.080** | **0.990** | 6.810 | **2.580** |

**解读**：
- Accuracy = 0.990，与 Real 持平，action class 极准
- 即使没专门为 action-to-motion 设计（ACTOR 和 INR 都是专门方法），MDM 仍然领先

### 8.4 Action-to-Motion: UESTC (Table 4)

| Method | FID_train ↓ | FID_test ↓ | Accuracy ↑ |
|---|---|---|---|
| Real | 2.92 | 2.79 | 0.988 |
| ACTOR | 20.49 | 23.43 | 0.911 |
| INR | **9.55** | 15.00 | 0.941 |
| **MDM** | 9.98 | **12.81** | 0.950 |

UESTC 比 HumanAct12 大得多（25K 样本，40 类），FID 整体偏高。MDM 在 test FID 上领先 INR 约 15%。

### 8.5 Unconstrained Generation (Table 5)

| Method | FID ↓ | KID ↓ | Precision/Recall | MultiModality ↑ |
|---|---|---|---|---|
| ACTOR | 48.80 | 0.53 | 0.72/0.74 | 14.10 |
| MoDi | **13.03** | **0.12** | 0.71/0.81 | 17.57 |
| MDM | 31.92 | 0.36 | 0.66/0.62 | 17.00 |

MoDi 是专门为 unconstrained 设计的，MDM 是通用模型顺带做 unconstrained，差距合理。MDM 仍然超过 ACTOR（一个 conditioned 模型硬塞到 unconstrained）。

---

## 9. Limitations 与后续影响

**Limitations**：
- 推理慢：1000 步 forward pass，从 <1s 涨到 ~1 分钟（motion 维度小，所以还算可接受）
- R-Precision 落后 VAE 类方法（fidelity vs diversity 的 trade-off）

**后续影响**：MDM 之后涌现了大量 motion diffusion 工作：
- **MotionDiffuse** (Zhang et al. 2022, 同期): https://arxiv.org/abs/2208.15001
- **FLOOR**: 基于 MDM 改进的 layout-to-motion
- **MotionDiffuse**, **MoFusion**, **OmniControl**, **FineMoGen**, **MotionLCM** (Latent Consistency Models for motion)
- 启发了 latent diffusion for motion（类似 Stable Diffusion 把 DDPM 移到 latent space）

**为什么 MDM 至今仍是 baseline**：
- 训练成本低（单 GPU 3 天）
- 代码开源且清晰
- 通用框架支持任意 condition
- Geometric losses 设计哲学被后续工作广泛继承

---

## 10. 一句话直觉总结

MDM 的核心 insight 可以压缩成一句：**在 motion 这个低维、temporal、几何约束丰富的 domain 中，predict-sample 形式的 diffusion 加 transformer encoder 加 geometric losses 加 classifier-free guidance，构成一个轻量、可编辑、多样的 SOTA generator**。每个选择都不是 ad-hoc，而是由 motion domain 的结构属性反向推导出来的——这就是"carefully adapted"的真正含义。
