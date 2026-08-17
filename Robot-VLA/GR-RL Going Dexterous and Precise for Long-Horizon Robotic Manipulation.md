---
source_pdf: GR-RL Going Dexterous and Precise for Long-Horizon Robotic Manipulation.pdf
paper_sha256: 2a010e98364a3e99ec8054f4041646dbe6774891e6e3befabf3416c28ece92bf
processed_at: '2026-08-04T22:17:33-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 你好，我们来抛开学术黑话，直接用最 bottom-up 的直觉来聊聊这篇 paper 到底在干嘛。

这个 work 的核心出发点是一个很尴尬的现实：现在的大模型 VLA（比如 GR-3）很聪明，什么都会一点，BUT 让它去做那种“需要双手配合、拉扯软体物体、还要毫米级穿针引线”的长链条任务（比如系鞋带），它就崩了。

Paper 里揭示了 VLA 失败的两个根本原因，这也是整个 pipeline 要解决的痛点：

1. **Human demo 太脏了**：人在遥操作穿鞋带时，手会抖，会穿歪，会犹豫，会退回来重试。如果你用最朴素的 Behavior Cloning (BC) 让模型去 mimic 这些 demo，模型就会把这些“错误动作和犹豫”也学进去。在简单任务里这没啥，BUT 在毫米级精度的任务里，一个微小的多余动作就会导致鞋带穿不过去。
2. **训练和执行的对不齐**：模型训练时，看到的是一段一段的 raw action chunk。BUT 在真实部署时，为了让机械臂动作平滑，系统层会加各种滤波和优化（temporal ensembling）。这就导致模型脑子里想的东西，和机械臂实际做出来的东西，存在 mismatch。

为了解决这两个问题，GR-RL 设计了一个三步走的流水线。我用大白话加上底层的技术细节给你拆解一下。

---

### Stage 1: 用 RL Critic 当“剪辑师”，把脏数据洗干净

**直觉**：你拍 vlog 不可能把所有 NG 镜头都放进去，你得剪辑。怎么让机器知道哪些是 NG 镜头？GR-RL 的 trick 是：训练一个 RL critic，让它来判断“当前这一步到底有没有推进任务进度”。如果有，保留；如果发现进度倒退了（比如鞋带掉出来了），就把这一段 demo 删掉。

**技术深挖**：
他们没有用普通的 regression 去预测 progress，因为 regression 只会一条直线平滑到底，捕捉不到“为了更好地抓取而故意把鞋带放下”这种 long-term 的好动作。他们用了 **Distributional Offline RL** (TD3+BC)。

核心 Reward 设计公式：
$$
r(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t) = \begin{cases} 
\gamma^{T-t} \mathbb{I}(\tau), & t > T - k \\
0, & t \leq T - k 
\end{cases} \tag{1}
$$
变量解释：
- $\mathbf{o}_t$: 当前观测的图像
- $l$: language instruction
- $\mathbf{s}_t$: robot 的本体感觉状态
- $\mathbf{a}_t$: 当前 action chunk
- $T$: 整条 trajectory 的总长度
- $k$: action chunk 的长度
- $\gamma$: discount factor
- $\mathbb{I}(\tau)$: 判断这条轨迹最终是否成功的 indicator (1 或 0)

这个公式的逻辑是：只在轨迹的最后 $k$ 步给 reward，并且越靠近终点 reward 越大（通过 $\gamma^{T-t}$ shaping）。因为大部分 demo 都是成功的，为了告诉 critic 什么是失败，他们用了一个叫 **Hindsight Experience Replay (HER)** 的 trick，人工标出 teleoperator 犹豫的 keyframe，把一条成功轨迹硬生生截断成多条失败轨迹丢给 critic 学。

因为用了 distributional critic，输出的 Q-value 天然 bound 在 $[0,1]$ 之间。那么 progress 计算就是取期望：
$$
\rho_t := \text{mean}(Q_\phi(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)) \tag{2}
$$
如果 $\rho_t$ 在接下来的几步里骤降，系统就判定这一段是 suboptimal，直接从训练集里剔除。

---

### Stage 2: 左右镜像复制，白嫖一倍数据量

**直觉**：双手操作有天然的对称性。你用左手拿鞋带穿右边的孔，和右手拿鞋带穿左边的孔，在物理上是对称的。既然如此，把所有数据镜像翻转一下，数据量直接翻倍，还能强迫模型学到真正的空间 invariance，BUT 绝对不是简单地把图片 flip 一下就行。

**技术深挖**：
这个叫 Morphological Symmetry Augmentation。具体操作：
1. **Vision**: 水平翻转图像，并且交换 left wrist camera 和 right wrist camera 的输入。
2. **Action & State**: 在 world frame 下做镜像变换，然后再 transform 回 local wrist frame。这要求机器人的运动学结构是严格对称的。
3. **Language**: 把 prompt 里的“左”和“右”对调。比如 "thread the left hole" 变成 "thread the right hole"。

这个 simple trick 直接让 success rate 涨了 10% 以上。在数据稀缺的高精度任务里，这种强 inductive bias 极其管用。

---

### Stage 3: 在 Latent Space 里做 Online RL，对齐真实执行环境

**直觉**：模型在干净数据上 offline 学会了，BUT 一上真机，因为那些平滑滤波器，动作还是有点变形。要解决这个 mismatch，必须让模型在真实环境里自己试错。BUT 如果直接在 action space（比如关节角度）加随机噪音去 explore，在系鞋带这种毫米级任务里，随便加噪音永远穿不进去，根本拿不到 reward。

GR-RL 的解法是：**在生成动作的源头——Noise Space 里做探索**。Diffusion/Flow policy 生成动作时，第一步是采样一个高斯噪音 $\epsilon$，然后再逐步 denoise 出 action。他们选择去 steer 这个噪音 $\epsilon$。

**技术深挖**：
在冻结的 VLM backbone 后面，挂了一个极小的 noise predictor $\pi_{\theta'}$ (只有 51.5M 参数)。这个 predictor 输出初始噪音 $\epsilon_t$，喂给 Action DiT。

Loss 函数设计：
$$
\mathcal{L}(\pi_{\theta'}) = \mathbb{E} \left[ -Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t) + c \cdot \max\left(\frac{1}{2}\|\epsilon_t\|^2 - \beta, 0\right) \right] \tag{3}
$$
变量解释：
- $\epsilon_t$: noise predictor 采样的初始噪音
- $Q_{\phi'}(\dots)$: 在 noise space 训练的另一个 critic，评估这个噪音能生成多好的 action
- $-Q_{\phi'}$: 第一项是 policy improvement，试图寻找能产生高价值动作的噪音
- $\frac{1}{2}\|\epsilon_t\|^2 - \beta$: 第二项是 penalty，强制 $\epsilon_t$ 不要偏离标准正态分布 $\mathcal{N}(0, \mathbf{I})$ 太远。如果偏离太大，生成的 action 就会 out-of-distribution (OOD)，导致机器人乱动。
- $\beta$: 允许偏离的阈值
- $c$: penalty 系数

这里有个非常 interesting 的细节，他们在训练 noise space 的 critic $Q_{\phi'}$ 时，采样 $\epsilon_t$ 有 0.5 的概率从标准正态分布采，0.5 的概率从 noise predictor 采。这是为了保证 critic 对整个 noise space 都有 coverage，防止 policy 钻牛角尖。

---

### 最终效果与技术联想

这套组合拳打下来，数据非常漂亮：

| Training Stage | Success Rate | Intuition |
| :--- | :--- | :--- |
| Base GR-3 (直接 BC) | 45.7% | 师傅怎么乱来它就怎么乱来 |
| + Stage 1 (Data Filtering) | 61.6% | 剪掉了 NG 镜头，基础扎实了 |
| + Stage 2 (Symmetry Aug) | 72.7% | 领悟了左右互搏，泛化变强 |
| + Stage 3 (Online RL) | 83.3% | 闭环微调，克服了真机执行误差 |

**我的联想：**
1. **关于 RL 的复兴**：过去几年 RL 在 robotics 里被 BC 压着打，因为 RL 太难调，reward 太难设。这篇 paper 展示了 RL 的一个全新用法：用来做 data curation。Offline RL critic 在这里充当了一个极其敏锐的“状态评估器”。这跟 LLM 里的 Reward Model 本质上是一回事，BUT 这里的 critic 是基于 Bellman equation 的，能看懂 long-term 的因果，而不是单点的 preference。
2. **Latent Space Steering 的潜力**：在 diffusion policy 里做 RL 一直是个大难题，因为 backprop 通过整个 denoising step 太贵。在 initial noise 上做文章，类似于给模型的大脑植入一个“潜意识引导”，让它在 OOD 边界内做最大程度的探索。这跟最近很火的 RLHF 里在 latent representation 上做 preference optimization 有异曲同工之妙。
3. **Sim2Real 的新解法**：以前我们总想着在仿真器里加 domain randomization 来对齐真实世界。GR-RL 提供了另一种思路：直接在真实环境里做极少步数的 online RL，只要你的 offline base 足够强，并且探索空间限制得足够巧妙（latent noise + KL penalty），real-world RL 的 sample efficiency 是可以接受的。

**参考链接：**
- Distributional RL 原始 paper (C51): https://arxiv.org/abs/1707.06887
- TD3+BC (Offline RL 基础): https://arxiv.org/abs/2106.01345
- Hindsight Experience Replay (HER): https://arxiv.org/abs/1707.01495
- Latent Space Steering for Diffusion Policy: https://arxiv.org/abs/2506.15799
- 并行工作 π*0.6: https://arxiv.org/abs/2511.14759

总结一句话：GR-RL 告诉我们，让大模型做精细活儿，光看视频瞎学是不够的，得有人帮它剪掉废片（Filtering），得让它懂得左右对称，最后还得让它在真实环境里闭着眼睛摸几把，它才能真正长手眼协调的脑子。

---

# GR-RL 深度技术解析

Karpathy 你好，这篇 ByteDance Seed 的 paper 是 VLA 领域一个相当有分量的小 step。它瞄准的不是 generalization，而是 **specialization** —— 把一个 generalist policy (GR-3) 锻造成能自动系鞋带 (lace up a shoe) 的 expert，做到 **83.3% success rate**，穿过多个 eyelets，millimeter-level precision。让我从底层逻辑一路拆到顶层 trick。

---

## 1. 核心洞察：为什么 Human Demonstrations 在 Dexterous 任务里是 Suboptimal 的

这是一个关键 reframe。传统 imitation learning 默认 human demos 是 optimal，然后 BC（behavior cloning）去 mimic。但 GR-RL 的 claim 是：

- **在 high-precision dexterous manipulation** 下，teleoperator 会 hesitate、retry、shake，demos 里充满 multimodal noisy action fragments
- **Training/inference mismatch**：训练时 policy 见到的是 raw action chunks，但 deployment 时用了 temporal ensembling [Zhao et al., 2023]、receding horizon control [Black et al., 2024]、jerk constraint refinement，这导致 policy 学的 和 执行的 不一致
- 在 long-horizon + high-precision 场景下，这两个问题会复合放大

直觉上：你教一个学生模仿一个手忙脚乱的师傅，结果他还手忙脚乱地执行（因为 smooth filter），那他能做对才怪。GR-RL 的 solution：**先清洗 demos，再 online RL 让 model 自己闭环体验真实部署的动作**。

---

## 2. Model Architecture：Mixture-of-Transformer (MoT)

```
                    ┌─────────────────────────────────┐
   language l ─────▶│                                 │
   vision o_t ─────▶│   VLM Backbone                  │── KV cache (latter half)
   robot state s_t ─▶│   (Qwen2.5-VL-3B-Instruct)      │         │
                    │                                 │         ▼
                    └─────────────────────────────────┘   Shared tokens
                                          │                    │
                          ┌───────────────┴────────────┐       │
                          ▼                            ▼       │
                  Policy π_θ (action DiT)        Critic Q_φ    │
                  flow-matching objective        distributional│
                  outputs a_t:t+k                outputs Q-chunk
                                                          (k values)
```

- **Policy π_θ**: 输入 `(l, o_t, s_t)`，输出 k-length action chunk `a_t = a_{t:t+k}`，用 **action diffusion transformer (DiT) + flow matching** [Lipman et al., 2022; Liu, 2022]
- **Critic Q_φ**: 同样是 causal transformer，用 **Q-chunking** [Seo & Abbeel, 2024; Li et al., 2025] 预测一个 chunk of Q-values，并用 **distributional RL** [Bellemare et al., 2017; Farebrother et al., 2024]
- 总参数 **5B**

**Distributional critic 的关键 trick**：把 value 看作 bounded discrete distribution，上下界设为 1 和 0。这样 Q-value **天然就是 task progress**（0=刚开始，1=完成）。这是后面 data filtering 的关键。

为什么不直接用 non-distributional critic？因为 sparse reward + long horizon 下，non-distributional critic 会严重 **over-estimate** early states（reward signal 很弱），导致 progress 预测完全失序。Paper Figure 7 给了直观对比。这点跟 π*0.6 [Physical Intelligence, 2025] 的 observation 一致。

参考链接：
- Distributional RL 原始 paper: https://arxiv.org/abs/1707.06887
- Q-chunking: https://arxiv.org/abs/2411.12155
- Flow matching: https://arxiv.org/abs/2210.02747

---

## 3. Stage 1: Offline RL 作为 Data Filter

这是 paper 最 elegant 的 trick。让我详细讲讲公式。

### 3.1 Sparse Reward 设计

$$
r(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t) = \begin{cases} 
\gamma^{T-t} \mathbb{I}(\tau), & t > T - k \\
0, & t \leq T - k 
\end{cases} \tag{1}
$$

变量解释：
- $\mathbf{o}_t$: 第 t 步的 observation (多视角 RGB images)
- $l$: language instruction
- $\mathbf{s}_t$: proprioception state (robot 自己的 joint positions 等)
- $\mathbf{a}_t$: action chunk
- $\mathbb{I}(\tau)$: indicator function，trajectory τ 成功为 1，失败为 0
- $T$: trajectory 总长度
- $k$: action chunk 长度
- $\gamma$: discount factor (通常 0.99 左右)
- $\gamma^{T-t}$: 这是个关键的 shaping term — 越接近终点 reward 越大（因为 $\gamma < 1$，$t$ 越大，$T-t$ 越小，$\gamma^{T-t}$ 越接近 1）

所以只有 **最后 k 步** 才有 nonzero reward，并且 reward 随时间衰减。

### 3.2 Hindsight Failed Trajectories

关键 trick：**most demos 都是成功的**，但我们需要 failed trajectories 才能让 critic 学到 "什么导致失败"。

方法：在每条成功 trajectory 里人工标注 **retry keyframes** $m_0, m_1, \ldots, m_{M-1}$（即 teleoperator 犯错/犹豫/重试的时刻），然后把 trajectory 截断到 $m_i$ 之前当作 failed trajectory：

$$
\tau_{0:m_i}, \quad 0 \leq i < M
$$

所以一条成功 trajectory 经过 hindsight augmentation 后变成 M+1 条 (1 个成功 + M 个失败)。这跟 HER [Andrychowicz et al., 2017] 的哲学一脉相承。

参考：HER 原始 paper https://arxiv.org/abs/1707.01495

### 3.3 Critic as Progress Function

用 **TD3+BC** [Fujimoto & Gu, 2021] 训练 critic 后，对每个 transition 计算：

$$
\rho_t := \text{mean}(Q_\phi(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)) \tag{2}
$$

注意：因为 distributional critic 输出的是 categorical distribution，所以 $\rho_t$ 是这个 distribution 的 **期望值**（mean），落在 [0, 1] 区间。

### 3.4 Filtering Rule

定义 sample $(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)$ 为 **suboptimal** 的条件：

> 在序列 $\rho_{t:t+k}$（即当前 chunk 的 progress 序列）中，存在大于阈值 δ 的 drop

直觉：如果 teleoperator 在这个 chunk 里犯错（比如 shoelace slipped out），progress 就会骤降，这个 transition 就被剔掉。

Figure 3 给了非常直观的例子：
- (a) shoelace missed eyelet → progress 突降 → thread 成功 → progress 上升
- (b) handover 失败重试 → progress 在多次 oscillation
- (c) **intentionally put down shoelace to regrasp** → progress 暂时降但长期升 — 这是 regression-based predictor 抓不到的，因为它只看 short-term

### 3.5 跟 Regression-based Progress Predictor 的对比

Ablation 实验里训了一个 baseline：直接回归 $dt/dT$（当前步占总进度的比例）。

| 维度 | RL-based critic | Regression predictor |
|------|-----------------|---------------------|
| 处理 hindsight failed data | 天然 | 不能（没失败数据可回归） |
| 捕捉 long-term effect (e.g. 放下 shoelace 重抓) | ✓ 突变响应 | ✗ 平滑无反应 |
| 对 millimeter-level subtle failure 敏感 | ✓ | ✗ |
| 输出 bound | [0,1] 自然 | unbounded，容易飘 |

这是 paper 的关键 contribution 之一：**Q-value under sparse reward ≈ progress function**，比直接监督学习 progress 更 robust。

---

## 4. Stage 2: Morphological Symmetry Augmentation

非常 simple 但 effective 的 trick。Bimanual manipulation 有天然的 mirror symmetry：

```
原始 scene:        Augmented scene:
  ┌──────────┐        ┌──────────┐
  │ L wrist  │        │ R wrist  │ (image flipped)
  │ R wrist  │   →    │ L wrist  │
  │ language │        │ "right"  │ (left→right)
  │ "left"   │        │           │
  └──────────┘        └──────────┘
```

具体操作：
1. **Image**: 水平 flip 所有图像 + swap left wrist camera 和 right wrist camera
2. **Proprioception** $\mathbf{s}_t$ 和 **action** $\mathbf{a}_t$：在 world frame 做 mirror symmetry，再 transform 回 local wrist frame
3. **Language** $l$：spatial description flip，"the hole on the left" → "the hole on the right"

效果：success rate 从 61.6% → 72.7% (+11.1%)，single simple trick 带来的 gains。

直觉：bimanual manipulation 中左右手是 symmetric role，mirror 让 model 见到 2x 数据，并且强迫它学 invariance，而不是 spurious correlation "left hand 总做抓"。这跟 **data augmentation in CV** (random crop, flip) 的逻辑一样，只是要做 robot-specific 的 transform。

---

## 5. Stage 3: Online RL via Latent Space Steering

这是 paper 最 tricky 的部分。直接在 action space 加 noise 几乎不可能成功（因为 millimeter precision + long horizon，随机扰动永远 hit 不到 eyelet）。所以 GR-RL 用 **structured exploration in latent space**，借用了 [Wagenmaker et al., 2025] 的思路：steer the flow policy's initial noise。

参考: https://arxiv.org/abs/2506.15799

### 5.1 Architecture: Noise Predictor

在 shared VLM backbone 后面挂一个 lightweight **noise predictor** $\pi_{\theta'}$ (只有 **51.5M 参数**)，输出 action DiT 的 initial noise $\epsilon_t$。

```
VLM Backbone (frozen) ──► tokens
                          │
              ┌───────────┼────────────┐
              ▼                        ▼
        action DiT (π_θ)         noise predictor (π_θ')
        input: ε_t (from π_θ')   output: ε_t
        output: a_t (action chunk)
```

### 5.2 Loss for Noise Predictor

$$
\mathcal{L}(\pi_{\theta'}) = \mathbb{E}_{(\mathbf{o}_t, l, \mathbf{s}_t) \sim \mathcal{D}} \left[ -Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t) + c \cdot \max\left(\frac{1}{2}\|\epsilon_t\|^2 - \beta, 0\right) \right]
$$

变量：
- $\epsilon_t \sim \pi_{\theta'}(\mathbf{o}_t, l, \mathbf{s}_t)$：noise predictor 采样的 initial noise
- $Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t)$：**noise space Q function**，评估某个 noise 对应的 action 的 return
- 第一项：maximize Q (policy improvement)
- 第二项：**KL-like penalty**，防止 $\epsilon_t$ 偏离 $\mathcal{N}(0, \mathbf{I})$ 太远，避免 OOD action
- $\beta$：偏离阈值
- $c$：penalty 系数

直觉：policy gradient 在 noise space 而非 action space，因为 noise 是连续可微的输入，扰动它就等于扰动整条 generated trajectory 的 manifold，但 manifold 是 structured 的（被 flow model 的训练 distribution 约束）。

### 5.3 Loss for Noise Space Q

$$
\mathcal{L}(Q_{\phi'}) = \text{cross\_entropy}\left(Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t), \ Q_{\phi}(\mathbf{o}_t, l, \mathbf{s}_t, \pi_{\theta}(\mathbf{o}_t, l, \mathbf{s}_t | \epsilon_t))\right) \tag{4}
$$

with $\epsilon_t$ sampled from:
$$
\epsilon_t \sim \begin{cases} 
\mathcal{N}(\mathbf{0}, \mathbf{I}) & \text{w.p. 0.5} \\
\pi_{\theta'}(\cdot) & \text{w.p. 0.5}
\end{cases}
$$

关键设计：**0.5 概率从原 normal distribution sample，0.5 概率从 noise predictor sample**。这是 GR-RL 对原始 Wagenmaker 方法的改进 — 为了保证 **noise space coverage**（原方法只从 predictor sample，会偏向 high-return region，distributional critic 在没探索到的区域估值不可靠）。

cross_entropy 因为是 distributional Q（categorical distribution），所以用 cross-entropy 而不是 MSE。

### 5.4 Buffer Strategy

- **Off-policy buffer**: warm-up 阶段用 offline checkpoint rollout 673 trajectories，注入 buffer
- **On-policy buffer**: 只存最近 2 个 checkpoint 的 trajectories，stale 数据 push 到 off-policy buffer
- **不 mix teleoperated trajectories**！这点非常重要：因为 teleop 数据的 dynamics 和 policy rollout 不一致（前面讲的 training/inference mismatch）

参考 Warm-start RL: https://arxiv.org/abs/2412.07762

### 5.5 为什么这套设计在 Real-world RL 里能 work

Sample efficiency 是 real-world RL 的命门。GR-RL 的 sample efficiency 来自：

1. **强 offline base** (filtered BC + augmentation 72.7%)，online RL 只需在局部 refine
2. **Latent space exploration**：探索空间维度大幅降低，structured
3. **Distributional critic**：robust to sparse noisy reward
4. **Off-policy + on-policy mix**：充分利用历史 + 紧跟当前 policy
5. **50 step per 12 episodes** update：避免 policy 漂移过快

---

## 6. 实验数据表

### 6.1 Main Result (Success Rate)

| Stage | Success Rate | Δ |
|-------|--------------|---|
| GR-3 baseline (BC on all demos) | 45.7% | — |
| + Data filtering (offline RL critic) | 61.6% | +15.9% |
| + Symmetry augmentation | 72.7% | +11.1% |
| + Online RL (500 steps) | 83.3% | +10.6% |

Online RL 训练中，success rate 短期 dip (distribution shift)，然后迅速恢复并超过 90%（moving average）。最终 eval 选 500-step checkpoint，因为后期有 behavior drift (paper 提到的 limitation)。

### 6.2 Failure Mode Breakdown (Figure 6)

主要瓶颈是 **threading shoelace into correct eyelet**。Data filtering 和 online RL 在这一步改善最多。Augmentation 在所有 stage 都有改善但 magnitude 小。

这跟直觉一致：threading 是 millimeter-precision 关键步骤，最依赖 demo 质量 + inference consistency。Augmentation 是 uniform improvement，因为对称性是 inductive bias，对所有 stage 都 generalize。

---

## 7. Robot Hardware: ByteMini-v2

- Wheeled mobile manipulation robot
- **7-DoF dual arms** + 球形 wrist joint [ByteWrist, Tian et al., 2025]
- 改进点：
  - **Elbow actuator**: peak torque 17 Nm → **35 Nm**，peak load 1.4 kg → **3.15 kg**
  - **Chassis**: 500×720 mm → **450×650 mm** (更 compact，confined space 友好)
  - Servo steering wheels：yaw + pitch 同步调整 [Wu et al., 2023]
  - ID 优化：monitor 从 chassis 移到 shoulder

参考 ByteWrist: https://arxiv.org/abs/2509.18084

---

## 8. 行为智能

Figure 8 展示了 GR-RL 的 robust behaviors，从中能看出它确实在 long-horizon reasoning 上做得不错：

- **(a)** 不同颜色 shoe generalize ✓
- **(b)** shoelace 掉落 → 自动 regrasp ✓
- **(c)** 没穿准 eyelet → retry ✓
- **(d)** 抓 shoelace 离 tip 太远 → intentionally 放下 → 在 shoe 表面 regrasp closer to tip (long-term positive action!)
- **(e)** shoe 角度不好 → 先 reorient → 再 threading
- **(f)** shoe 在 far side → pull near → 调整 → threading
- **(g)** 两根 shoelace 交叉，正确那根被压住 → pull out the correct one

(d) 这个行为特别有意思：它要 model 理解 "短期放下 shoelace 是好的，因为长期能更容易穿入 eyelet"。这正是 regression-based progress predictor 抓不到的 long-horizon reasoning，而 distributional critic 能 capture (Figure 3c)。

---

## 9. Limitations

Paper 自己承认的：
- **Behavior drift during online RL**：sparse + noisy reward，online training 时 policy 不稳定。可能因为 noise predictor 容量小 (51.5M)，或 latent action space 大，credit assignment 难
- **Distillation 回 base VLA** 没做，未来方向

我补充几点 paper 没明说但隐含的：
- 单 task specialist (shoe lacing)。Generalization 到其他 precise dexterous task 未知
- Teleop 标注 retry keyframes 需要人工，scalability 存疑
- Reward 是 binary sparse + manual annotation（"shoelace 通过正确 eyelet 并放回 table"），自动化程度有限
- Long online training 时间（500 steps × 12 episodes per update ≈ 6000 episodes，real-world 里这个 sample 量在鞋带这种 long-horizon task 上是相当大）

---

## 10. 跟 π*0.6 的对比

π*0.6 [Physical Intelligence, 2025] 是 concurrent work，也是 high-precision manipulation + real-world RL：

| 维度 | GR-RL | π*0.6 |
|------|-------|-------|
| Critic | Distributional | Distributional |
| Policy improvement | Filtered BC + online steering (latent noise) | Advantage-conditioned denoising |
| Exploration space | Latent noise | Latent (via advantage conditioning) |
| Base model | GR-3 | π0.5 |
| 实验任务 | Shoe lacing (long-horizon) | High-precision (more general) |

两者都 hit 到 distributional critic 这个 idea。GR-RL 的 contribution 主要在 **filtered BC pipeline**：通过 critic 把 demos 清洗掉 suboptimal fragments，offline 阶段就拿到 72.7% base，然后 online RL 才能 efficient (因为 search space 小)。

参考 π*0.6: https://arxiv.org/abs/2511.14759

---

## 11. Build Intuition：为什么这套 pipeline 对

让我帮你 build 起直觉。

**Core mental model**: Generalist VLA 是一个 "知道大概怎么做" 的 policy，但 demo 里的 noise + training-inference mismatch 让它在 high-precision 区域 collapse。要把 generalist 变 specialist，需要：

1. **蒸馏 demo** (Stage 1)：用 offline RL critic 当 judge，过滤掉 teleop 时的 hesitation 和 error
2. **Exploit symmetry** (Stage 2)：inductive bias 注入，2x data + invariance
3. **Closed-loop self-improvement** (Stage 3)：让 policy 在真实部署的 action distribution 下 explore，而不是 teleop 的 distribution

这三个 stage 是 **递进且互相依赖** 的：
- 没有 Stage 1，base policy 就 noisy，online RL 起点低
- 没有 Stage 2，policy 泛化性差，online RL 探索效率低
- 没有 Stage 3，policy 永远在 mismatch 下部署，再 clean 的 demo 也无法接近 deployment-time 的 optimal

**类比 human learning**：
- Stage 1 = 把师傅的错误动作筛掉
- Stage 2 = 通过镜像练习加强 motor program
- Stage 3 = 在真实任务里反复试错，把"心里想的"和"手做的"对齐

这就是为什么这套 pipeline 能让系鞋带这种 task 跨过 success threshold 的根本原因。

---

## 12. 联想和 Open Questions

- **Critic as Progress** 这个 idea 是否能 generalize 到其他 long-horizon task？比如 multi-step cooking、assembly。如果任务结构更复杂（多 sub-task），单 Q function 可能不够，需要 hierarchical critic 或 subgoal detection
- **Distributional critic 的 bound [0,1]** 选得巧妙。如果 task 有 partial credit (e.g., 穿过 5 个 eyelets 中的 3 个)，bound 是否应该 [0, 5]？这是一个 natural extension
- **Noise space exploration** vs **advantage-conditioned denoising**：哪个更适合 VLA backbone 很大的场景？GR-RL 选择冻结 VLM 只训 51.5M noise predictor，是 sample efficiency 的妥协。如果 VLM 不冻结，可能能学到更 complex behavior 但 sample efficiency 大降
- **Multi-task specialist distillation**：GR-RL 一个 task 一个 model。能不能把 N 个 specialist 蒸回 generalist？这是 paper limitation 提的方向，跟 LoRA-style adaptation / multi-task distillation 相关
- **Auto-retry keyframe labeling**：现在 retry keyframe 是人工标注。能否用 critic 的 progress drop 自动 detect？bootstrapping 出来。这跟 active learning 相关
- **Behavior drift 的 mitigation**：能否用 KL constraint against offline policy (类似 PPO 的 clip)？GR-RL 用了 penalty term $\frac{1}{2}\|\epsilon_t\|^2 - \beta$，是 implicit KL，但可能不够

---

## 13. 相关 Reference 链接汇总

**GR 系列基础**:
- GR-3: https://arxiv.org/abs/2507.15493
- GR-2: https://arxiv.org/abs/2410.06158

**Offline RL / Critic-based Progress**:
- TD3+BC: https://arxiv.org/abs/2106.01345
- HER: https://arxiv.org/abs/1707.01495
- Distributional RL: https://arxiv.org/abs/1707.06887
- Stop Regressing (classification for value): https://arxiv.org/abs/2403.03950

**Diffusion / Flow Policy**:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π*0.6: https://arxiv.org/abs/2511.14759
- Diffusion Policy: https://arxiv.org/abs/2304.13705 (ACT, temporal ensembling 出处)
- Flow Matching: https://arxiv.org/abs/2210.02747

**Online RL Steering**:
- Wagenmaker et al. latent steering: https://arxiv.org/abs/2506.15799
- Warm-start RL: https://arxiv.org/abs/2412.07762
- SERL: https://arxiv.org/abs/2401.16087

**VLA Foundation Models**:
- OpenVLA: https://arxiv.org/abs/2406.09246
- RDT-1B: https://arxiv.org/abs/2410.07864
- CogACT: https://arxiv.org/abs/2411.19650
- Octo: https://arxiv.org/abs/2405.12213

**Robot Hardware**:
- ByteWrist: https://arxiv.org/abs/2509.18084

---

## 14. 总结

GR-RL 是 VLA specialist 化的 **第一个完整 pipeline**，并且在 **shoe lacing** 这种 long-horizon + dexterous + millimeter-precision 的 task 上做到 83.3%。三个核心 trick:

1. **Offline RL critic as progress filter**：巧妙地把 sparse reward 下的 Q-value 当作 task progress，filter 掉 suboptimal demos
2. **Morphological symmetry augmentation**：bimanual mirror symmetry，2x data + invariance
3. **Latent noise steering for online RL**：在 noise space 做 structured exploration，避免 action space 的 random perturbation 失败

整个 pipeline 的 intuition 是 **把 generalist policy 的 noise 过滤 + symmetry 加强 + online deployment 对齐**，三步递进。这套思路对所有想用 VLA 做 high-precision dexterous manipulation 的 researcher 都很有参考价值。Limitation 在 multi-task distillation 和 behavior drift，未来工作方向也清晰。

希望这个 breakdown 对你 build intuition 有帮助, Karpathy。如果你想深挖某个 stage (e.g., distributional critic 的 categorical representation 细节，或 noise space KL penalty 的 tuning)，我可以再展开。
