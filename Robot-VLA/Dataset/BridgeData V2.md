---
source_pdf: BridgeData V2.pdf
paper_sha256: aa5fdac48bf9122b22f88150d45575cb0685f2a2f761ca6fc570be743022819b
processed_at: '2026-08-03T14:24:14-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# BridgeData V2 人话版：到底干了啥

Andrej，我换个节奏，把这篇 paper 用大白话再走一遍。这篇 paper 本质上做了一件特朴素的事——**攒了一大堆 robot data，然后验证说"数据多了，policy 就能 generalize"**。但魔鬼在细节里，我把它掰开揉碎讲。

---

## 1. 这篇 paper 想解决啥痛点

Robotics 长期有个尴尬：CV 和 NLP 早就靠 "大数据 + 大模型" 横扫一切，GPT 嘛，scaling law 嘛。但 robot learning 死活 scale 不起来。原因很傻——**没数据**。

更精确地说，有几个 specific 的痛点：

**痛点 A：以前的 dataset 只有 1 个 environment**

你拿 MIME 或 BC-Z 训个 policy，到你自己 lab 用，scene 对不上，policy 就废了。你得 1:1 复现原 lab 的桌面、灯光、物体位置。这等于数据根本没法复用。

**痛点 B：以前的 dataset 一个 scene 只有一个 task**

这个更阴险。假设你 scene 里只有 "把杯子放桌上" 这一种 task，policy 学到的其实是 "看到这个画面就输出这个动作"，**根本没 attend 到 task instruction**。你给它个 language command，它照样按画面 hardcode 的动作走。Multi-task learning 名存实亡。

**痛点 C：数据采集太贵**

Google 的 RT-1 用 Everyday Robot，那个 robot 一个就几万刀，学术 lab 根本买不起。BC-Z 也是 Google 内部搞的，外人复现不了。

**痛点 D：方法之间数据不通用**

RT-1 的数据只能给 language-conditioned BC 用，RoboNet 的数据只能给 goal-conditioned RL 用。你做 research 想试不同方法，得自己重新采数据。

BridgeData V2 想把这四个痛点一起解决。

---

## 2. 他们具体怎么搞的

### 2.1 硬件：便宜 + 公开

- Robot: **WanderX 250**，约 $4000
- Camera: Intel RealSense D435 (RGBD over-the-shoulder) + 2× Logitech C920 (randomized pose) + Raspberry Pi camera (wrist)
- Teleoperation: Meta Quest 2 VR
- Control: 5 Hz, 640×480 image
- 所有零件公开可买，两周到货

这是 democratization 的关键。任何 academic lab 都能复现这套 setup。

### 2.2 采集 protocol 的关键设计

**Design 1：每个 scene 有多个 feasible task**

比如摆个 toy kitchen，有 sink、stove、drawer、各种食物。采集者随便选 task 来 demo——可以开 drawer、可以把 fork 放 sink、可以关 stove。这样 policy 必须 attend task spec 才能区分行为。

**Design 2：不 reset scene**

每条 trajectory 跑完不归位物体，直接接着采下一条。这大大加速采集，同时让 data distribution 更接近真实 deployment（真实世界也不 reset）。

**Design 3：每 50 条 trajectory 大 random 一次**

Random 什么：
- 两个 alternative camera 的 pose
- Scene 里的物体
- Workspace 相对 robot 的位置

这个 randomization 是 cross-institution generalization 的 source。你 lab 的 camera 偏 5 度？policy 见过类似的，没事。

**Design 4：混入 scripted autonomous data**

作者训了个 heavily randomized pick-and-place scripted policy，让它自己跑采了 9731 条。这 policy 经常失败。为什么保留这些失败数据？**因为 offline RL 方法（CRL, IQL）反而需要 suboptimal 数据学 robust behavior**。BC 方法用不上可以 exclude，但 dataset 要给你 flexibility。

**Design 5：post-hoc language 标注**

采集时不标 task name，采完用 crowdsourcing 平台补 language label。这避免采集者分心，加速采集速度。

---

## 3. Dataset composition 拆解

### 3.1 Skill vs Task 的精确定义

这俩词在 robotics literature 里一直乱用，这篇 paper 给出清晰定义：

- **Skill** = motion pattern 的聚类
  - 例子：pick-and-place、pushing、sweeping
  - 同一个 skill 可以对应不同 object、不同位置
  
- **Task** = language instruction 的聚类
  - 例子："put fork in sink" 和 "put bowl in sink" 是同一个 skill (pick-and-place)，不同 task
  - 同一个 skill 也能对应不同 task

**Intuition**: Skill 是 "怎么动"，Task 是 "动到哪儿、动什么"。Policy 要学的是 "skill 是共享的 motor primitive，task 是具体的 instantiation"。

### 3.2 13 个 Skills 列表

| 类别 | Skills |
|---|---|
| Foundational (大头) | Pick-and-place, Pushing, Reorienting |
| Tool use | Sweeping (颗粒物), Wiping |
| Articulated obj | Open/close drawer, door, box flap |
| Deformable | Folding cloth |
| Precision | Stacking blocks |
| Small mechanism | Twist knob, Flip switch, Zip zipper, Turn lever |

**为啥 foundational 占大头**？因为 pick-and-place / pushing 是 manipulation 的 "alphabet"，其他复杂 skill 都能拆成这些 primitive 的组合。先把 alphabet 学扎实，complex skill 才能 transfer。

### 3.3 24 个 Environments

- 7 个 toy kitchens（含 sink, stove, microwave 的不同组合）
- 多个 tabletop
- 几个 standalone toy sink
- 一个 toy laundry machine
- 等等

**100+ 个 object**，跨 texture、shape、weight 多样。

---

## 4. 6 个方法的人话版

### 4.1 GCBC（最 baseline）

输入：current image + goal image，channel-wise stack 一下。

```
[obs, goal] → ResNet-34 → 3×FC → 7D action
```

Loss 就是 MSE：

$$\mathcal{L} = \mathbb{E}\left[\|a - \pi_\theta(s, g)\|^2\right]$$

- $a$ = demonstrator 的真实 action（7D：6D EE pose + 1D gripper）
- $\pi_\theta(s,g)$ = policy 网络输出的 action
- $s$ = current observation
- $g$ = goal image（从 trajectory 未来 timestep 随机采样）

**问题**: L2 regression 输出 single deterministic action，对 multi-modal 任务无能为力。"杯子放左放右都行" 这种 case，L2 会输出中间位置，action 直接 invalid。

### 4.2 D-GCBC（Diffusion 版 BC）

不输出 single action，输出一个 action distribution。用 DDPM 建模。

**前向加噪**（采数据时用）：

$$x_t = \sqrt{\bar{\alpha}_t}\, a + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

- $a$ = 真实 action（clean）
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ = 随机高斯噪声
- $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$ = noise schedule 累计
- $x_t$ = timestep $t$ 的 noisy action

**反向去噪**（inference 时用）：

从纯噪声 $x_T$ 开始，一步步去噪得到 $a$：
$$x_{t-1} = \mu_\theta(x_t, t, c) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

- $c = \text{ResNet-34}([s, g])$ = conditioning
- $\mu_\theta$ = 神经网络预测的 mean

**训练 loss**（简化版）：
$$\mathcal{L} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

- $\epsilon_\theta$ = 网络预测的 noise
- 让网络学会 "给你 noisy action，告诉你加了什么 noise"

**Intuition**: Diffusion 把 action distribution 当成 image 来生成。multi-modal 直接天然支持——sample 两次得到不同 valid action。比 L2 强太多。

但代价是 inference 慢（要迭代去噪 T 步），且 sample 出 OOD action 的风险也高。

### 4.3 ACT（Action Chunking with Transformers）

核心 idea：**不预测下一步 action，预测下 5 步 action 一起**。

为啥这么做？传统 BC 每步误差累积，跑 50 步就漂了。Chunk 让 policy "看见未来 5 步"，相当于 implicit planning，error 不累积。

架构是 Conditional VAE：

- **Encoder** $q_\phi(z | s, a_{t:t+5})$：训练时用，把 "未来 5 步 action" 压成 latent $z$
- **Decoder** $p_\theta(a_{t:t+5} | s, z)$：inference 时用，给个 $z$ 生成 5 步 action

Loss:
$$\mathcal{L} = \|a - \mu_\theta(s, z)\|^2 + \beta D_{KL}(q_\phi \| p(z))$$

- 第一项：重建 action
- 第二项：KL regularizer，让 $z$ 别太离谱
- $\beta$ = KL weight

**Intuition**: $z$ 是 "style variable"，捕捉 multi-modality。"快折叠" vs "慢折叠" 对应不同 $z$。Inference 时从 prior sample $z$，得到不同 mode 的 action。

ACT chunk size = 5（原 paper 是 100，因为 Aloha 50Hz 高频控制，本文 5Hz 低频，chunk 缩小）。

### 4.4 CRL（Contrastive RL）

这哥们最特殊，不是 BC，是 RL，但 offline 跑。

核心 idea：把 "reach goal" 这件事翻译成 "future obs 的 representation 要靠近 goal 的 representation"。

Value function 写成 log-linear：
$$V(s, g) = \phi(s)^\top \psi(g)$$

- $\phi(s)$ = obs encoder（ResNet-34）
- $\psi(g)$ = goal encoder（ResNet-34，和 $\phi$ 共享 backbone）

Loss 是 InfoNCE：
$$\mathcal{L} = -\log \frac{\exp(\phi(s)^\top \psi(g^+) / \tau)}{\sum_{g'} \exp(\phi(s)^\top \psi(g') / \tau)}$$

- $g^+$ = 真实 future obs（正样本）
- $g'$ = batch 里其他 trajectory 的 goal（负样本）
- $\tau$ = temperature

**Intuition**: 这就是 contrastive learning 那套。让 "当前 obs" 和 "真实 future goal" 在 embedding space 靠近，和 "别人家的 goal" 拉远。学完之后 $\phi(s)$ 自动 encode "我在 trajectory 哪个阶段、离 goal 多远" 的信息。

Policy extraction 用 GCBC 做 regularizer（weight 0.2），避免 sample 出 OOD action。

### 4.5 LCBC（Language-Conditioned BC）

输入：image + language instruction。

流程：
1. Language $\ell$ → frozen MUSE encoder → 2× FC → embedding $e_\ell$
2. Image $s$ → ResNet-34，每个 block 末尾用 FiLM 调制

**FiLM 公式**：
$$\text{FiLM}(x | \gamma, \beta) = \gamma \odot x + \beta$$

- $x$ = ResNet block 输出的 feature map
- $\gamma, \beta$ = 从 $e_\ell$ 通过 MLP 预测出来的 scale 和 shift
- $\odot$ = element-wise 乘

**Intuition**: FiLM 让 language 在网络每一层都 modulate visual feature。比 simple concat 强，因为 language 影响是分层的。

但 LCBC 用 frozen MUSE，对 "eggplant"、"corn cob" 这种 rare word grounded 不充分。Table 2 里 LCBC 在 "put corn in pot" 上 0.0，因为 MUSE embedding 里 "corn cob" 这个 phrase 和 "corn" 不太一样，没 fine-tune 接不上。

### 4.6 RT-1（Google 的 Robotics Transformer）

最强的方法。架构：
1. Image (320×256，比其他方法大) → EfficientNet → image tokens
2. Language → language model → language tokens
3. 所有 token → decoder-only Transformer → 预测 action tokens

**关键设计**：
- **Action discretization**: 7D action 每维切 256 bin，tokenize 成 vocabulary 256
- **Observation history**: 15 个 timestep（原 RT-1 是 6，本文加大）
- **Loss**: next-token prediction (cross-entropy)

**Intuition**: RT-1 把 robot control 当成 "sequence modeling" 问题。Transformer 学 "given 历史 obs 和当前 instruction，下一个 action token 是什么"。Discretization 让 transformer 在 discrete space 学得稳，避免 continuous action 的 unimodal collapse。

为啥 RT-1 比 LCBC 强那么多？三个 reason：
1. 图像大（320×256 vs 128×128），visual detail 丰富
2. Action discretization，transformer 学 discrete token 更稳
3. Observation history，能 modeling temporal context（pause、delay）

---

## 5. 实验结果人话版

### 5.1 Seen Tasks (Table 2)

**RT-1 在 "open drawer" 上 1.0 完爆全场**——drawer handle 需要极精确 6-DOF insertion，RT-1 的 discretization + history 学得最稳。

**Stack block 所有人都跪**——block stacking 需要 delicate balance，goal image 区分不出 "成功 stack" 和 "没 stack"，task design 本身有盲区。

**LCBC 在 corn/eggplant 上 0.0**——object name 在 MUSE 里 grounded 不充分，unseen word 直接挂。

### 5.2 Unseen Tasks (Table 3) — 真正测 generalization

**核心 finding**: Goal-conditioned 方法（GCBC, D-GCBC, CRL）平均 0.5+，language-conditioned 方法（LCBC）只有 0.08。

**Intuition**: Goal image 直接 grounded 在视觉，"看到目标长这样" 就是 task spec。Language 受 vocabulary coverage 限制，unseen object name 直接失灵。

**RT-1 在 seen obj + unseen env 上 0.83**——observation history + EfficientNet 给它足够 visual robustness 吸收 environment shift。

**D-GCBC 和 ACT 在 unseen 上反而不如 GCBC**——diffusion / CVAE 的 expressiveness 在 OOD 下变成 liability，sample 出 OOD action。

### 5.3 Cross-Institution (Table 4) — Lab 1 → Lab 2

**RT-1 几乎不掉**（0.47 → 0.40），其他方法掉一半。

**Intuition**: Large transformer + history 对 domain shift 极其 robust。这是 robotics cross-lab 可用性的 key evidence。

### 5.4 Scaling Analysis (Figure 5) — 最 strong 的 finding

三个结论：

**(1) Model capacity**: ResNet encoder 越大越好。small model 学不会 wrist orientation。

**(2) Data size**: 数据越多 seen 和 unseen task 都越好。scaling law 在 robotics 也成立。

**(3) Skill diversity positive transfer** — 最重要

控制数据量近似相等（28k vs 27k），但 skill 数从 3 变 13：

| 训练 skills | unseen pick-and-place success rate |
|---|---|
| 3 skills (pick-and-place, pushing, wiping) | 0.30 |
| 13 skills (全部) | **0.65** |

**Intuition**: 不同 skill 之间有 shared "physics common sense"。Wiping 让 policy 学到 cloth 的 affordance，sweeping 让 policy 学到 tool use，这些 knowledge 反过来 improve pick-and-place robustness。

这等于在 robotics 上验证了 "diverse pretraining → downstream transfer" 这套 LLM 的 recipe。

---

## 6. Limitations 人话版

1. **Low-precision task**: 没 force-controlled insertion、没 dynamic task（throwing、heavy lifting）。这是 robot foundation model 的下一个 frontier。

2. **Single institution**: 数据全 Berkeley 采，distribution 偏 US academic lab。

3. **Single robot**: WanderX 形态固定，没法直接 transfer 到 Franka、UR5。Multi-robot dataset 是 obvious next step。

---

## 7. 对你 Build Intuition 的关键 Take-aways

1. **Multi-task per scene 是 dataset design 硬性要求**——否则 policy 不 attend task spec
2. **Random camera pose + random workspace = cross-institution 的 source**——不是 nice-to-have，是必要条件
3. **Goal image > Language instruction for generalization**——image grounded 在视觉，language 受 vocabulary 限制
4. **Diffusion / CVAE 在 seen 上强，在 OOD 上可能反而差**——expressiveness 是双刃剑
5. **Skill diversity 产生 positive transfer**——这是 robot foundation model 路线的 empirical 基石
6. **Large transformer + history + discretization 是最 robust policy 范式**——RT-1 设计 cross-institution 掉最少

---

## 8. 后续影响

这篇 paper 是 RT-X (Open X-Embodiment) 的前奏。后来 Google 联合 22 个 institution 把 BridgeData V2、RT-1 data、Jaco data 等全合并，搞了 RT-X dataset (https://robotics-transformer-x.github.io/)，验证 cross-robot generalization。

你最近在搞的那套 "robot data scaling law" 直觉，BridgeData V2 是最早的 empirical 证据之一。

---

## 9. Web Links

- BridgeData V2: https://rail-berkeley.github.io/bridgedata/
- RT-1: https://robotics-transformer.github.io/
- RT-X: https://robotics-transformer-x.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- ACT (Aloha): https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Contrastive RL: https://contrastive-rl.github.io/
- IDQL: https://arxiv.org/abs/2304.10573
- FiLM: https://arxiv.org/abs/1709.07871
- DDPM: https://arxiv.org/abs/2006.11239

总结一句：这篇 paper 朴素但关键——它证明 $4000 robot + 60k diverse trajectories 就能让 6 个 SOTA method 在 cross-institution 上 work，给 robot foundation model 路线奠定了 dataset design 的方法论基础。

---

# BridgeData V2: 大规模 Robot Learning 数据集深度解析

Andrej, 这篇 paper 我细看了一下，核心动机非常清晰：把 vision 和 NLP 那套 "大模型 + 大数据 → broad generalization" 的 recipe 搬到 robot manipulation 上来。但 robotics 长期卡在数据规模和 diversity 上，BridgeData V2 就是想 fill 这个 gap。下面我把技术细节和 intuition 展开讲。

---

## 1. Dataset 核心数字与设计哲学

| 指标                    | 数值                                            |
| --------------------- | --------------------------------------------- |
| Total trajectories    | 60,096                                        |
| Expert demos          | 50,365 (84%)                                  |
| Scripted (autonomous) | 9,731 (16%)                                   |
| Skills                | 13                                            |
| Environments          | 24                                            |
| Objects               | 100+                                          |
| Robot                 | WidowX 250 (~$4,000)                          |
| Control freq          | 5 Hz                                          |
| Image res             | 640×480                                       |
| Action dim            | 7D (6D Cartesian EE pose + 1D gripper binary) |

**设计哲学的关键 insight（这里我 build 一下你的 intuition）：**

前人的 dataset（如 MIME, RoboTurk, BC-Z, RT-1）普遍只覆盖 1~3 个 environment，意味着你要拿这些数据训练 policy，必须精确复现采集时的 scene。BridgeData V2 反其道而行——**每个 environment 都同时存在多种可行 task**。这解决了 imitation learning 里一个 subtle 但要命的问题：如果每个 scene 只有单一 task，policy 学到的就是 "看到这种 image 就输出这种 action"，完全没必要 attend 到 task specification（goal image 或 language）。Multi-task 数据强迫 policy 把 task spec 当成 input 的一部分来用，否则没法区分。

这点和 RT-1 的发现一致（https://robotics-transformer.github.io/），但 BridgeData V2 把它放到了 academic-scale 公开数据层面。

---

## 2. 硬件 Setup 与 Sensing

硬件图里能看到：
- **Primary camera**: Intel RealSense D435 RGBD，over-the-shoulder 固定视角
- **Randomized cameras**: 2× Logitech C920，每 50 条 trajectory 随机一下 pose
- **Wrist camera**: 自制 3D-printed mount + Raspberry Pi camera module
- **Teleoperation**: Meta Quest 2 VR controller

**为什么强调 randomized camera pose？** 这是 build intuition 的关键：传统 dataset 把 camera 固死，policy 也就学死了那个视角的 visual feature。如果别家 lab 的 camera 稍微偏 5 度，policy 就挂。Random camera pose 强制 policy 学到 perspective-invariant 的 representation，是 cross-institution generalization 的核心 enabler。

**为什么用 $4000 的 WidowX？** 公开、可采购、两周到货，学术 lab 能复现。这是和 BC-Z（Google 内部 robot）和 RT-1（Google Everyday Robot）的本质差别——democratization of robot learning。

---

## 3. Dataset Composition：Skill vs Task 的精细定义

paper 在 Section 3.3 把 "skill" 和 "task" 区分得非常清楚，这对理解后续 generalization experiment 至关重要：

- **Skill** = 动作模式的聚类（如 pick-and-place, pushing, sweeping），允许不同 object 不同 arrangement
- **Task** = language instruction 的聚类（如 "put the fork in the sink" vs "put the bowl in the sink" 都是 pick-and-place skill，但不同 task）

13 个 skills 包括：
1. Pick-and-place（含 in-place reorienting）
2. Pushing
3. Wiping
4. Sweeping（颗粒物）
5. Stacking
6. Folding cloths
7. Opening/closing drawers
8. Opening/closing doors
9. Opening/closing cardboard box flaps
10. Twisting knobs
11. Flipping switches
12. Zipping/unzipping
13. Turning levers

**Foundational skills（pick-and-place, pushing, reorienting）占大头**——因为它们 transfer 到更复杂 task 的可能性最高，是 manipulation 的 "alphabet"。

**9,731 scripted trajectories 的来历**：作者训练了一个 heavily randomized pick-and-place scripted policy，让它自己跑。这个 policy 经常失败，但失败的 trajectory 对 offline RL 有用——因为 offline RL（如 IQL, CRL）反而能从 suboptimal 数据中学到 robust behavior，不会像 BC 那样被噪声拖死。这是为什么 BridgeData V2 同时支持 BC 和 offline RL 两类方法的关键设计。

Reference: IQL (Kostrikov et al., 2021) https://arxiv.org/abs/2110.06169

---

## 4. 6 个 Offline Learning Methods 的技术拆解

### 4.1 Goal-Conditioned Methods

**(a) GCBC (Goal-Conditioned Behavioral Cloning)** — baseline

```
π(a | s, g) = MLP(ResNet-34([s, g]))
```

其中：
- $s$ = current observation (128×128 RGB)
- $g$ = goal image (从 trajectory 未来 timestep 均匀采样)
- $[s, g]$ = channel-wise stack (6 channels)
- ResNet-34 → 3×FC(256) → 7D action

Loss 是 standard MSE regression：

$$\mathcal{L}_{GCBC} = \mathbb{E}_{(s,a,g) \sim \mathcal{D}} \left[ \| a - \pi_\theta(s, g) \|^2 \right]$$

Data augmentation: random crop + random resize + color jitter。线性 warmup 2000 steps，Adam，lr=3e-4。

**Intuition**: GCBC 用 L2 regression 输出 single deterministic action，对 multi-modal action distribution 完全无能为力。比如 "把杯子放桌上" 既可放左边也可放右边，L2 会 collapse 到中间（其实是 invalid action）。

---

**(b) D-GCBC (Diffusion Goal-Conditioned BC)** — IDQL 去掉 value function 的简化版

DDPM 前向加噪：

$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$

- $x_t$ = 在 timestep $t$ 的 noisy action
- $x_{t-1}$ = 在 timestep $t-1$ 的 noisy action
- $\beta_t$ = variance schedule 第 $t$ 步的 noise 增量

反向去噪：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t, c)\right)$$

其中 $c = \text{ResNet-34}([s, g])$ 是 conditioning embedding。

训练 objective（重参数化形式）：

$$\mathcal{L}_{DDPM} = \mathbb{E}_{t \sim \mathcal{U}(1,T),\, x_0,\, \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

- $\epsilon$ = 采样自 standard Gaussian 的 noise
- $\epsilon_\theta$ = 神经网络预测的 noise
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$（重参数化）
- $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$

**Intuition**: Diffusion 直接对 action distribution 建模，能捕捉 multi-modal。这就是为什么 Table 2 中 D-GCBC 在 "fold blue cloth"（0.7 vs GCBC 0.4）上明显更强——cloth folding 有多种 valid 折叠方式，unimodal model 学不出来。

Reference: IDQL (Hansen-Estruch et al., 2023) https://arxiv.org/abs/2304.10573
Reference: DDPM (Ho et al., 2020) https://arxiv.org/abs/2006.11239

---

**(c) ACT (Action Chunking with Transformers)** — Aloha paper 的方法

核心：不预测 single action，而是预测一整段 action sequence $a_{t:t+k}$。chunk size 原版 100，本文改成 5（因为 control freq 是 5Hz 不是 50Hz，trajectory 也只有 50-100 步）。

架构是 Conditional VAE：

**Encoder**（训练时用）：
$$q_\phi(z | s_{t:t+k}, a_{t:t+k}) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$$

**Decoder**（推理时用）：
$$p_\theta(a_{t:t+k} | s_{t:t+k}, z) = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$$

- $z$ = latent style variable，捕捉 multi-modality
- $s_{t:t+k}$ = observation history
- $a_{t:t+k}$ = action chunk (k=5)

Loss:
$$\mathcal{L}_{ACT} = \underbrace{\| a - \mu_\theta(s, z) \|^2}_{\text{reconstruction}} + \beta \underbrace{D_{KL}\left(q_\phi(z|s,a) \| p(z)\right)}_{\text{KL}}$$

- $\beta$ = KL 重量，控制 latent 的 entropy
- $p(z)$ = prior，standard Gaussian

本文修改：observation + goal 在 ResNet-18 上 channel-wise stack，让 ACT 变成 goal-conditioned。

**Intuition**: ACT 通过 chunk 抑制 compounding error——传统 BC 每步误差累积，chunk 让 policy "看见未来 5 步"，相当于 implicit planning。同时 CVAE 的 latent $z$ 提供 multi-modality。这就是为什么 ACT 的行为有 "pauses"——人在演示时手停顿，chunk 把停顿也学了。

Reference: ACT (Zhao et al., 2023) https://arxiv.org/abs/2304.13705

---

**(d) CRL (Contrastive RL)** — Eysenbach 那条线

把 goal-conditioned RL 转化为 contrastive representation learning。Value function 写成 log-linear 形式：

$$V(s, g) = \log p(g | s) \approx \phi(s)^\top \psi(g)$$

- $\phi(s)$ = observation encoder 输出
- $\psi(g)$ = goal encoder 输出
- 都用 ResNet-34

Contrastive loss（InfoNCE 形式）：

$$\mathcal{L}_{CRL} = -\mathbb{E}_{(s,g^+) \sim \mathcal{D},\, g^- \sim \text{batch}} \left[ \log \frac{\exp(\phi(s)^\top \psi(g^+) / \tau)}{\sum_{g' \in \{g^+, g^-\}} \exp(\phi(s)^\top \psi(g') / \tau)} \right]$$

- $g^+$ = 真实 future goal（positive sample）
- $g^-$ = batch 中其他 trajectory 的 goal（negative samples）
- $\tau$ = temperature

Policy extraction 用 GCBC-style regularization（$\lambda=0.2$）：
$$\mathcal{L}_{policy} = \mathcal{L}_{CRL} + 0.2 \cdot \mathcal{L}_{GCBC}$$

本文实现细节：ResNet-34 backbone 在 value 和 policy 之间 shared，加速 learning。TD-style target 用了 Stabilizing CRL 的技巧。

**Intuition**: CRL 本质是把 "reach goal" 这件事翻译成 "future observation representation 和 goal representation 在 embedding space 接近"，绕开了 reward shaping 和 explicit value backup。Offline 训练时不需要任何 reward 标注，只要 trajectory 本身。

Reference: Contrastive RL (Eysenbach et al., 2022) https://arxiv.org/abs/2201.07827
Reference: Stabilizing CRL (Zheng et al., 2023) https://arxiv.org/abs/2306.07848

---

### 4.2 Language-Conditioned Methods

**(e) LCBC (Language-Conditioned BC)** — baseline

架构：
1. Language instruction $\ell$ → frozen **MUSE encoder** → 2× FC → embedding $e_\ell$
2. Image $s$ → **ResNet-34** with **FiLM conditioning** on $e_\ell$
3. Final encoding → FC → 7D action

**FiLM (Feature-wise Linear Modulation)** 公式：

$$\text{FiLM}(x | \gamma, \beta) = \gamma \odot x + \beta$$

- $x$ = 每个 ResNet block 的 feature map
- $\gamma, \beta$ = 从 $e_\ell$ 通过两个小 MLP 预测出来的 scale 和 shift
- $\odot$ = element-wise product

FiLM 在每个 ResNet block 末尾施加，让 language 在网络深层不断 influence visual feature。

**Intuition**: FiLM 比 simple concatenation 强在——language 在每个层级 modulate visual feature，类似 attention 但计算更便宜。但 LCBC 用 MUSE 这种 sentence embedding 没有 fine-tune，对 unseen object name（如 "marker", "rice"）泛化弱——Table 3 中 LCBC 在 unseen object 上平均 0.08，几乎学不会。

Reference: FiLM (Perez et al., 2017) https://arxiv.org/abs/1709.07871
Reference: MUSE (Yang et al., 2019) https://arxiv.org/abs/1907.04307

---

**(f) RT-1 (Robotics Transformer)** — Google 那条线

架构：
1. Image (320×256 RGB) → **EfficientNet** (pre-trained) → image tokens
2. Language instruction → pre-trained language model → language tokens
3. All tokens → **decoder-only Transformer** → predict discretized actions

Action discretization：每维 action 切成 256 个 bin，tokenize 成 vocabulary size 256 的 token。

History: 15 个 timestep（原 RT-1 是 6，本文增加以适应更长 episode）。

Loss: standard next-token prediction (cross-entropy) on action tokens。

**Intuition**: RT-1 比其他方法强（Table 2 平均 0.49，和 GCBC 持平但更稳定；Table 3 average 0.50，远超 LCBC 0.08）的原因有三个：
1. **Larger image (320×256 vs 128×128)** — visual detail 更丰富，特别是 "open drawer" 这种需要精确插入 gripper 的 task
2. **Action discretization** — transformer 在 discrete token 上学得更稳，避免 continuous action 的 unimodal collapse
3. **Observation history** — 能建模 temporal context，对 pause/delay 的 modeling 关键

但 RT-1 训练成本高，需要 TPU Research Cloud 支持。

Reference: RT-1 (Brohan et al., 2022) https://arxiv.org/abs/2212.06817

---

## 5. Experimental Results 详细解读

### 5.1 Seen Tasks (Table 2)

| Task | GCBC | D-GCBC | ACT | CRL | LCBC | RT-1 |
|---|---|---|---|---|---|---|
| Open drawer | 0.4 | 0.6 | 0.5 | 0.4 | 0.5 | **1.0** |
| Sweep beans | 0.9 | 0.9 | 0.9 | 0.7 | 0.4 | 0.6 |
| Fold blue cloth | 0.4 | 0.7 | 0.7 | 0.5 | 0.5 | **0.9** |
| Stack block | 0.4 | 0.2 | 0.3 | 0.6 | 0.0 | 0.0 |
| Put corn in pot | 0.9 | 0.8 | 0.8 | 0.8 | 0.0 | 0.0 |
| Put carrot on plate | 0.7 | 0.4 | 0.1 | 0.0 | 0.0 | 0.8 |
| Flip pot upright | 0.1 | 0.1 | 0.0 | 0.4 | 0.4 | 0.4 |
| Put eggplant in pot | 0.1 | 0.2 | 0.0 | 0.0 | 0.0 | 0.2 |
| **Avg** | 0.49 | 0.49 | 0.41 | 0.42 | 0.23 | 0.49 |

**值得注意的 pattern**：
- **Open drawer** 上 RT-1 = 1.0 完爆其他所有方法 — drawer handle insertion 需要极精确的 6-DOF control，RT-1 的 discretization + history 把这个 precision 学得最稳。
- **Stack block** 上 RT-1 和 LCBC 都 0.0 — block stacking 需要 delicate balance，所有方法都不行，goal image 也区分不出 "stack 成功" 和 "没 stack 成功"，是 dataset task design 的盲区。
- **Put corn/eggplant in pot** — LCBC 全 0，因为 "corn cob"、"eggplant" 这种 object 在 language label 里 grounded 不充分，frozen MUSE 不能 transfer 这种 rare word 的 semantics。

---

### 5.2 Unseen Tasks (Table 3) — Generalization 关键测试

| Task 类型 | GCBC | D-GCBC | ACT | CRL | LCBC | RT-1 |
|---|---|---|---|---|---|---|
| Unseen obj, unseen env (marker in bowl) | 0.6 | 0.6 | 0.2 | 0.7 | 0.0 | 0.0 |
| Unseen obj, seen env (rice sweep, thick cloth) | 0.45 avg | 0.30 | 0.50 | 0.15 | 0.20 | 0.25 |
| Seen obj, unseen env (wipe, mushroom, spoon) | 0.70 | 0.70 | 0.17 | 0.70 | 0.17 | 0.83 |
| **Avg** | 0.60 | 0.55 | 0.28 | 0.52 | 0.08 | 0.50 |

**核心 insight**: 
- **Goal-conditioned 方法的 generalization 显著强于 language-conditioned** — goal image 直接告诉 policy 终态视觉，不依赖 language grounding。LCBC 在 unseen object 上 0.08 几乎是废的。
- **RT-1 在 unseen environment + seen object 上仍有 0.83** — observation history + EfficientNet 给 RT-1 足够的 visual robustness 来吸收 environment shift。
- **D-GCBC 和 ACT 在 unseen 上反而不如 GCBC** — diffusion 和 CVAE 在分布外条件下 expressiveness 变成 liability，sample 出 OOD action。

---

### 5.3 Cross-Institution (Table 4) — Lab 1 → Lab 2

| Task | GCBC | D-GCBC | ACT | CRL | LCBC | RT-1 |
|---|---|---|---|---|---|---|
| Put carrot on plate | 0.7→0.3 | 0.4→0.0 | 0.1→0.0 | 0.0→0.3 | 0.0→0.0 | 0.8→0.4 |
| Flip pot upright | 0.1→0.0 | 0.1→0.2 | 0.0→0.3 | 0.4→0.2 | 0.4→0.1 | 0.4→0.6 |
| Put eggplant in pot | 0.1→0.1 | 0.2→0.2 | 0.0→0.0 | 0.0→0.1 | 0.0→0.0 | 0.2→0.2 |

**Critical observation**: RT-1 几乎不掉 (0.47 → 0.40)，其他方法掉一半。说明 large transformer + observation history 对 domain shift 极其 robust——这是 robotics cross-lab 可用性的关键证据。

---

### 5.4 Scaling Analysis (Figure 5)

三个 finding：

**(1) Model capacity scaling**: GCBC 用不同大小 ResNet encoder，larger model 严格更好。small model 在 "rotate spoon horizontally" 这种 case 上学不会 wrist joint 的正确 orientation。

**(2) Data size scaling**: 随机下采样子集（stratified by task），数据越多 seen 和 unseen task 都越好。

**(3) Skill diversity positive transfer**: 用 28k trajectories + 3 skills vs 27k trajectories + 13 skills，**控制数据量近似相等**。在 unseen pick-and-place 上：
- 3 skills: 0.30
- 13 skills: **0.65**

这是 paper 最 strong 的 finding：**不同 skill 之间有 positive transfer**。Wiping 让 policy 学到 cloth 的 affordance，sweeping 让 policy 学到 tool use，这些 knowledge 反过来 improve pick-and-place 的 robustness。

**Intuition build**: 这意味着 manipulation skill 之间有 shared 的 "physics common sense"。Pre-training on diverse skill 让 policy 学到 world dynamics，类似 LLM 在 diverse text 上学 syntax 和 reasoning。

---

## 6. 局限与未来方向

paper 自己坦白：
1. **Low-precision tasks** — 没有高精度 force-controlled insertion、没 dynamic task（throwing, heavy object）
2. **Single institution** — 数据全在 Berkeley 采，虽然 cross-lab eval 但 distribution 仍偏 US academic lab
3. **Single robot** — WidowX 形态固定，无法 generalize 到 Franka, UR5 等 industrial arm

未来方向最 promising 的是 multi-robot dataset，让 policy 学到 morphology-invariant 的 manipulation representation。这其实就是 RT-X 和 Open X-Embodiment 后来的方向（https://robotics-transformer-x.github.io/）。

---

## 7. 对你 Build Intuition 的关键 Take-aways

1. **Multi-task per scene 是 dataset design 的硬性要求** — 否则 policy 不会 attend to task spec
2. **Random camera pose + random workspace position = domain robustness 的 source** — 不是 nice-to-have，是 cross-institution 的必要条件
3. **Goal image > Language instruction for generalization** — 因为 image 直接 grounded 在视觉，language 受限于 vocabulary coverage
4. **Diffusion / CVAE 在 seen task 上强，在 OOD 上可能反而差** — expressiveness 是 double-edged sword
5. **Skill diversity 产生 positive transfer** — 这是 "robot foundation model" 路线的 empirical 基石
6. **Large transformer + history + discretization 是目前最 robust 的 policy 范式** — RT-1 的设计在 cross-institution 上掉得最少

---

## 8. 相关 Reference 链接汇总

- BridgeData V2 project page: https://rail-berkeley.github.io/bridgedata/
- RT-1: https://robotics-transformer.github.io/
- RT-X (后续 multi-robot): https://robotics-transformer-x.github.io/
- ACT (Aloha): https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Contrastive RL: https://contrastive-rl.github.io/
- IDQL: https://arxiv.org/abs/2304.10573
- FiLM: https://arxiv.org/abs/1709.07871
- DDPM: https://arxiv.org/abs/2006.11239
- Open X-Embodiment: https://robotics-transformer-x.github.io/

这篇 paper 本质上是 robotics 走向 "data-driven foundation model" 路线的 critical stepping stone——它证明了一个 $4000 robot + 60k trajectories 就能让 6 个 SOTA method 在 cross-institution 上 work，给后来 RT-X 和 π₀ 等 work 奠定了 dataset 设计的方法论。
