---
source_pdf: Humanoid-VLA Towards Universal Humanoid Control with Visual Integration.pdf
paper_sha256: 19dd35d0ff83c44a70a7a1bc893464b47c68edb2aa818ad295118ee45fbf13f8
processed_at: '2026-08-05T08:04:58-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Humanoid-VLA

## 一句话概括

让 humanoid robot 第一次能"自己看、自己想、自己动" —— 不再是被动听话的木偶，而是能看懂眼前有啥、自己决定怎么动的"有眼睛的机器人"。

---

## 1. 这篇 paper 在解决什么"人话"问题？

### 现状：robot 是瞎子 + 傀儡

想象你雇了个工人，他只能这样工作：

- 你说"举手"，他举手
- 你拽他的手，他跟着动
- 但他**看不见**面前的桌子上有杯子，也不会自己想"我应该去拿那个杯子"

这就是现在所有 humanoid 控制方法的状态 —— Exbody、H2O、OmniH2O、HARMON、UH-1 全都这样。它们本质上是 **reactive**（被动反应）的：你给 input，它给 output，中间没有"理解"。

为什么卡住了？因为要给 robot 装"眼睛+脑子"，需要三类数据同时存在：

| 需要的数据 | 现状 |
|-----------|------|
| 第一人称视频（egocentric vision）| 几乎没有 |
| 文字描述（language caption）| 很少且贵 |
| 全身 motion 数据 | 有一点 |

**teleop** 能采集这三类数据，但成本高到爆炸 —— 你得让人戴着 VR 头显操控 robot 干几千小时活，还容易出事。

### 这篇 paper 的"作弊"思路

作者的 insight 是：**把问题拆开**。

- 先让 model 学会"语言 ↔ 动作"的通用关系（不需要眼睛）
- 再用少量"带眼睛的数据"微调一下
- 最后接上 RL controller 让动作能真正执行

这就像教小孩：先教他"举手是什么意思"（不戴眼镜），再戴眼镜教他"看到球就踢"。

---

## 2. 最聪明的 trick：自己给自己出题

### 核心难题

Llama3-70B 这种 LLM 要训得好，得有大量"指令-回答"对。但 7500 小时的 online video motion **没有 caption**。

怎么办？传统做法是请人标注，或者用 VLLM 自动标注。但 VLLM 看不懂 fine-grained motion detail，标注质量差。

### 作者的解法：从 motion 里挖"内在结构"当 supervision

Motion data 自带可计算的结构特征：

- **轨迹**：root joint 在空间画的曲线
- **时长**：这段动作持续几秒
- **遮挡**：如果把左手的 joint 抹掉，剩下的还能不能复原
- **状态**：第 N 帧的 pose 长啥样

把这些特征当作"条件"，把完整 motion 当作"答案"，就自动生成了 QA pair：

> Q: "Please move your center along the trajectory of `<Track>`"
> A: 完整 motion sequence

这个 trick 妙在哪 —— **原本是 reconstruction 任务（motion→motion），改头换面成 conditional generation（language→motion）**，直接套进 LLM 的 next-token prediction 框架。

而且 motion feature 本身被 encode 成 token，**塞进 language instruction 里**，跟普通英文单词一起被 LLM 处理。这就把 motion 和 language 真正统一到一个 vocabulary 里了。

### 为什么这个 trick 重要

- 不需要任何人工标注
- 比用 VLLM 标注精确得多（VLLM 看不懂 motion 细节，但数学计算的 `<Track>` 是精确的）
- 可组合、可扩展（`<Track>` + `<Time>` + `<State>` 能组合出复杂条件）
- 数据规模直接放大 **25 倍**（vs Mao et al. UH-1）

---

## 3. 第二个聪明设计：把人体拆成 5 块

### 传统做法的毛病

T2M-GPT、MDM 这些 text-to-motion 模型，把 22 个 SMPL joint 一起丢进一个 VQ-VAE 量化。问题是：

- Torso 的运动模式（root 平移）和 left arm 的运动模式（挥手）在 latent space 根本不是一回事
- 共享 codebook 导致 cross-contamination
- 想编辑某个 body part？做不到 —— token 是整体的

### Humanoid-VLA 的拆法

把 pose 分成 5 个 part，**各自独立训练 encoder 和 codebook**：

```
Body Pose → [Left Leg] [Right Leg] [Torso] [Left Arm] [Right Arm]
              ↓           ↓          ↓        ↓          ↓
            VQ_1         VQ_2       VQ_3     VQ_4       VQ_5
           (1024)       (1024)     (1024)   (1024)     (1024)
              ↓           ↓          ↓        ↓          ↓
            z_1           z_2        z_3      z_4        z_5
```

公式形式（变量解释）：

$$\hat{z}_t = \mathcal{E}_m(c_t), \quad \hat{z}_t = \{\hat{z}_b\}_{b=1}^{5}$$

- $c_t$: timestep $t$ 的 body pose 数据
- $\mathcal{E}_m = \{\mathcal{E}_b\}_{b=1}^{5}$: 5 个 part 各自的 encoder
- $\hat{z}_b$: 第 $b$ 个 part 在 codebook $V_b$ 中查到的最近 token
- 每个都 $\in \mathbb{R}^{5}$（5 个 part 各 1 个 token）

### 为什么这么拆有道理

1. **Token-level 可编辑**：想训练 `<Occlusion>` 任务，直接把 left leg 的 token 替换成 `<Occlusion>` 占位符，模型学会"补全缺失部位"。整体量化做不到。

2. **各 part 学独立的 motion 语义**：torso codebook 学的是 locomotion pattern，arm codebook 学的是 gesture pattern，互不干扰。

3. **Cross-embodiment 通用**：任何 humanoid robot 都有这 5 个 part（loco-manipulation 的基本单元）。换 robot 不用改 model，只改最后的 joint mapping。

Loss 用的标准 VQ-VAE 三件套：

$$\mathcal{L}_{hvq} = \underbrace{\|c_t - \hat{c}_t\|_2}_{\text{重建}} + \underbrace{\|\mathbf{sg}(z_t) - \hat{z}_t\|_2}_{\text{codebook 拉近}} + \underbrace{\|z_t - \mathbf{sg}(\hat{z}_t)\|_2}_{\text{encoder commit}}$$

- $\mathbf{sg}(\cdot)$: stop-gradient，控制 gradient 只流到该流的地方
- 第一项让 decoder 输出逼近原始 input
- 第二项让 codebook embedding 跟上 encoder
- 第三项让 encoder 输出 commit 到 codebook

---

## 4. 训练流程：两阶段 = 脏数据 + 干净数据

### Stage 1a: 大规模 dirty pre-train

用从 video 提取的 motion（不准确但量大），配合 self-supervised augmentation 训 Llama3-70B。

Loss 就是标准 autoregressive NLL：

$$\mathcal{L}_{LLM} = -\sum_i \log p(x_o^i \mid x_o^{<i}, x_d)$$

- $x_o^i$: 第 $i$ 个 output motion token
- $x_o^{<i}$: 前面所有 output token（causal mask，只能看前面）
- $x_d$: 完整 input description（language + embedded motion tokens 混在一起）

### Stage 1b: 小规模 clean refine

用 motion capture 数据（AMASS 这种，精确但量少）继续训。让 model 符合真正的人类运动学。

### Ablation 的关键发现

| Low-quality | High-quality | w/ aug | FID↓ | DIV↑ |
|------------|--------------|--------|------|------|
| ✓ | ✗ | ✓ | 0.698 | 4.576 |
| ✓ | ✓ | ✗ | 0.557 | 3.867 |
| ✓ | ✓ | ✓ | **0.467** | **4.585** |

读懂这张表：

- 只用脏数据 + augmentation：FID 0.698，但 diversity 很高（4.576）—— aug 给的多样性起作用了
- 加干净数据但不用 aug：FID 降到 0.557，但 diversity 暴跌到 3.867 —— 干净数据让 motion 精确但"死板"
- 两者都用：FID 0.467，DIV 4.585 —— **sweet spot**

这跟 LLM 的 pretrain + SFT 是一个道理：pretrain 用 dirty web data 学广度，SFT 用 clean instruction data 学精度。

---

## 5. Vision 怎么接进来：不动原 model，只加一个小模块

### 设计哲学

Stage 1 学到的"语言↔动作"关系是宝贵的 broad knowledge。如果直接 fine-tune 全部参数，会把这块知识 catastrophic forget 掉。

作者的做法：**冻结原 transformer，复制一份，中间插 cross-attention**。

### Cross-attention 的角色分配

在每一层 $l$：

- **Query** 来自 language token：$Q_l = X_d^l W_Q^l$
- **Key/Value** 来自 vision token：$K_l = X_v^l W_K^l$, $V_l = X_v^l W_V^l$
- **Output**：$X_u^l = \text{Softmax}\left(\frac{Q_l K_l^T}{\sqrt{D}}\right) V_l$

变量含义：
- $X_d^l$: 第 $l$ 层 language feature
- $X_v^l$: 第 $l$ 层 vision feature
- $W_Q^l \in \mathbb{R}^{D_d \times D}$: language → hidden 的投影
- $W_K^l, W_V^l \in \mathbb{R}^{D_v \times D}$: vision → hidden 的投影
- $D$: hidden dimension
- $\sqrt{D}$: scaled dot-product 的标准缩放，防止内积过大

### 为什么是 language 作 Q，vision 作 KV

这是反直觉但合理的设计。一般的 image captioning 是 vision 作 Q，language 作 KV（因为要"看着图说话"）。

但这里反过来，因为：

- **Language 是 intent**："kick the ball" 是用户意图，需要被 grounded 到场景
- **Vision 是 context**：球在哪、多远、什么颜色 —— 是给 intent 服务的"证据"
- Language token 去"查询" vision token，获取它需要的 grounding info

如果反过来（vision 作 Q），vision 会主导，language 沦为被动描述，user intent 就被淹没了。

### Parameter-efficient 的好处

只训练 cross-attention 的 $W_Q, W_K, W_V$，参数量极小。Stage 1 的 motion manifold 完全保留，vision 只是"叠加"上去的 contextual signal。

---

## 6. 最后一步：从 token 到 robot 关节扭矩

VLA 输出的是 15 个 universal joint 的 pose sequence。但 Unitree G1 有 24 个 joint，还要变成 torque 才能动。

### 两步映射

**15 → 24 joint mapping**：用 Adam optimizer 做 optimization，保持 end-effector position 尽量对齐，把 15 个 keypoint 的位置映射到 24 个 robot joint。

**Pose → Torque**：goal-conditioned RL policy，用 PPO 训：

$$j_t = \mathcal{P}(s_t, p_t)$$

- $j_t \in \mathbb{R}^{24}$: 24 维 joint torque
- $s_t$: VLA 输出的 target pose
- $p_t$: robot proprioception（当前状态）
- $\mathcal{P}$: RL policy

Reward function $R(\mathcal{O}, \mathcal{G})$ 根据 observation $\mathcal{O}$ 和 goal $\mathcal{G}$ 输出 PD controller 的 target position。

### 为什么这个 design 重要

**Cross-embodiment universality**。换 robot 时只改 mapping 和 RL policy，VLA model 不动。这是"universal humanoid control"的真正含义。

---

## 7. 实验结果讲人话

### Motion 质量（Table 3）

跟 MDM（diffusion）和 T2M-GPT（transformer + VQ）比：

| | HumanML3D FID↓ | Humanoid-S FID↓ |
|---|---|---|
| MDM | 0.889 | 2.351 |
| T2M-GPT | 0.531 | 1.101 |
| **Humanoid-VLA** | **0.467** | **1.037** |

- FID 越低越好（生成分布跟真实分布越接近）
- Humanoid-VLA 比 MDM 提升 47.5%，比 T2M-GPT 提升 12%
- 为什么赢？因为 25 倍数据 + self-supervised aug 让 language-motion alignment 学得更好

### Physical plausibility（Table 2）

在 IsaacGym 仿真里测 motion 的物理可行性：

- Global MPJPE（关节位置误差）全部 < 40mm，最好 31.07mm
- PA-MPJPE（Procrustes 对齐后）最低 1.18mm —— shape accuracy 极好
- Acceleration error 最低 27.84 mm/s² —— motion 很平滑

**D+T 组合（描述+时长）效果最好**，D+A（有缺失 body part 要补全）最难。

### Real robot 实验（Table 5）

Unitree G1 上 8 个任务，每个做 10 次：

| Task | 成功率 |
|------|--------|
| Turn to an object | 10/10 |
| Hold an object | 9/10 |
| Wave to people | 10/10 |
| Avoid an obstacle | 9/10 |
| Jump over an object | 9/10 |
| Dance with a partner | 8/10 |
| Punch an obstacle | 10/10 |
| Kick a ball | 9/10 |

平均 93.75%。"Dance with a partner" 最低（8/10），因为要持续追踪 partner 位置，vision grounding 难度最高。

---

## 8. 整篇 paper 的核心 intuition

用最朴素的话讲：

### Intuition 1: 分开学，别一口吃成胖子

先学"语言和动作怎么对应"（不要眼睛），再学"看到东西怎么调整动作"。两阶段解耦，各学各的，避免小数据 overfit，也避免大数据把小信号淹没。

### Intuition 2: Motion 是 language 的一部分

把 motion 量化成 token，跟英文单词一起塞进 LLM 词表。所有 motion 操作变成 next-token prediction，self-supervised augmentation 自然 fall out —— 因为 motion 内在结构（轨迹、时长、状态）都能当"条件"输入。

### Intuition 3: 结构决定能力

Compositional VQ（拆 5 块）是 token-level 编辑的前提。没有这个拆分，`<Occlusion>` augmentation 设计不出来。Tokenization 的粒度决定了你能做什么 operation。

### Intuition 4: 脏数据给广度，干净数据给精度

7500 小时 video motion 不精确但量大，建立 broad alignment；29K mocap 精确但量小，做 refinement。这跟 LLM pretrain + SFT 是一个哲学。

### Intuition 5: Language 主导 intent，vision 服务 intent

Cross-attention 里 language 作 Q，vision 作 KV。意图是主线，视觉是辅助。反过来的话视觉会喧宾夺主。

### Intuition 6: Universal representation 锁在中间层

15 个 universal joint 是 human + humanoid 共有的最小集。向上接 VLA（不变），向下接各 robot 的 mapping（可变）。这是 cross-embodiment 的关键。

---

## 9. 我看到的局限和未来方向

### Paper 自己承认的

1. **RL policy 还不够 robust**，complex loco-manipulation 上会失败
2. **High-quality data 太少**，现有 dataset（如 Mimicking-Bench）限特定 robot，不能直接用
3. **训练方法简单**，没充分利用数据

### 我额外看到的

1. **Self-supervised aug 的天花板**：四种 aug 都从 motion 内部派生，model 学到的"language"其实是 motion 的结构描述，跟真正 human language 的 semantic richness 有 gap。未来可以混 VLLM-generated caption + structural aug。

2. **Cross-attention 可能不够深**：只 fine-tune cross-attention 是 parameter-efficient 的无奈选择。更深的 vision-motion fusion（MoE adapter、perceiver-style late fusion）可能更好。

3. **15→24 mapping 的 loss**：post-hoc Adam optimization 是个 hack。End-to-end 学习 mapping 可能更优，但需要 differentiable physics。

4. **Sim-to-real 的 quantitative gap**：Table 2 是仿真测的精细误差，Table 5 是 real 上的成功率，两者之间没有 bridge。Real 上的 motion quality 仍然是 open question。

5. **Egocentric data 的多样性**：paper 没说收集了多少 ego vision 数据。如果只有几百小时，cross-attention 能学到的 grounding 有限。

6. **Long-horizon 任务**：现在的 task 都是 single-step（踢球、避障）。Multi-step loco-manipulation（"去厨房拿杯水"）能不能做？paper 没测。

---

## 10. 这篇 paper 在大图景中的位置

### VLA 范式的演进

```
RT-2 (2023) ────► OpenVLA (2024) ────► π₀ (2024)
   │                  │                   │
   └── arm only       └── arm only        └── multi-robot
                                              but not humanoid focus

QUAR-VLA (2025) ──► QUART-Online (2024)
   │                    │
   └── quadruped        └── quadruped online

HumanVLA (2024) ──► Humanoid-VLA (本 paper)
   │                    │
   └── physical humanoid└── first universal humanoid VLA
       specific platform
```

### 关键转变

- **Arm VLA** (RT-2, OpenVLA, GR-2)：只管机械臂，manipulation only
- **Quadruped VLA** (QUAR-VLA)：四足，locomotion only
- **Humanoid VLA** (本 paper)：第一次把 whole-body locomotion + manipulation + vision + language 全融合

Humanoid 难在哪？自由度高（24+ DoF）、要同时维持平衡和做 manipulation、motion space 是 arm 的几十倍、sim-to-real 难度更高。

### 跟 LLM 范式的对应

| LLM 时代 | Humanoid-VLA 对应 |
|---------|------------------|
| Web text pretrain | Video motion pretrain (dirty) |
| Instruction tuning (SFT) | Mocap refine (clean) |
| Tokenization | Compositional VQ (5-part) |
| RLHF | Whole-body RL controller |
| Vision-language (LLaVA) | Vision-conditioned cross-attention |
| Multi-token prediction | Motion token autoregressive |

这篇 paper 本质上是把 LLM 的成功 recipe 搬到 humanoid control 上：dirty pretrain + clean SFT + efficient adapter for new modality + RL for execution。

---

## 11. Web Links 汇总

### Core
- Llama 3: [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)
- PPO: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- VQ-VAE: [https://arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937)

### Humanoid baselines
- Exbody: [https://arxiv.org/abs/2402.16796](https://arxiv.org/abs/2402.16796)
- Exbody2: [https://arxiv.org/abs/2412.13196](https://arxiv.org/abs/2412.13196)
- H2O: [https://arxiv.org/abs/2403.04436](https://arxiv.org/abs/2403.04436)
- OmniH2O: [https://arxiv.org/abs/2406.08858](https://arxiv.org/abs/2406.08858)
- UH-1: [https://arxiv.org/abs/2412.14172](https://arxiv.org/abs/2412.14172)

### VLA family
- RT-2: [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- OpenVLA: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- GR-2: [https://arxiv.org/abs/2410.06158](https://arxiv.org/abs/2410.06158)
- RDT-1B: [https://arxiv.org/abs/2410.07864](https://arxiv.org/abs/2410.07864)
- π₀: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- QUAR-VLA: ECCV 2025
- QUART-Online: [https://arxiv.org/abs/2412.15576](https://arxiv.org/abs/2412.15576)
- HumanVLA: [https://arxiv.org/abs/2406.19972](https://arxiv.org/abs/2406.19972)

### Motion
- AMASS: [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/)
- HumanML3D: [https://arxiv.org/abs/2105.02679](https://arxiv.org/abs/2105.02679)
- Motion-X: [https://arxiv.org/abs/2307.00272](https://arxiv.org/abs/2307.00272)
- MDM: [https://arxiv.org/abs/2209.14915](https://arxiv.org/abs/2209.14915)
- T2M-GPT: [https://arxiv.org/abs/2301.06098](https://arxiv.org/abs/2301.06098)
- TRAM: [https://arxiv.org/abs/2407.07182](https://arxiv.org/abs/2407.07182)

### Vision-Language
- LLaVA: [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- VILA: [https://arxiv.org/abs/2312.07533](https://arxiv.org/abs/2312.07533)

### Sim
- IsaacGym: [https://arxiv.org/abs/2108.10470](https://arxiv.org/abs/2108.10470)

---

## 最后一句人话总结

**这篇 paper 的本质**：把 humanoid control 从"被动提线木偶"升级成"有眼睛的自主 agent"。核心 trick 是用 self-supervised augmentation 把海量无标注 video motion 变成有监督训练数据，用 compositional VQ 让 motion 可编辑，用 cross-attention 高效注入 vision，最后用 RL 接地到 hardware。

**它告诉我们一件事**：当 data 不够时，别死磕标注，去挖数据内在的结构。Motion 的轨迹、时长、状态都是免费 supervision signal。这个 insight 在其他 data-scarce 领域同样适用。

---

# Humanoid-VLA 深度解析

## 1. 核心问题与动机

Humanoid robot 控制领域长期被 **reactive framework** 主导。具体讲，现有的 Exbody、H2O、OmniH2O、HARMON、UH-1 等方法本质上是把人类 motion capture 数据 retarget 到 humanoid，让 robot 被动地"翻译"输入信号(text / keypoints / teleop)为 motor output。这种范式缺少 **autonomous perception** —— robot 无法主动感知周围物体并推断该与什么交互。

要实现 **universal humanoid control**，需要三件事 fused：(1) language intent，(2) egocentric scene context，(3) whole-body motion control。瓶颈在于 **data scarcity**：

| Category | Text | Motion | Clips | Hours |
|----------|------|--------|-------|-------|
| Motion capture | ✓ | ✓ | 29K | 4.1 |
| Online video | ✗ | ✓ | 0.8M | 7515.7 |
| Synthetic | ✓ | ✓ | 100K | 227.7 |

带 text 标注的 motion capture data 只有 4 小时，video 有 7500+ 小时却没 caption。Teleop 成本过高无法 scale。Humanoid-VLA 的核心 contribution 是设计了一套 self-supervised augmentation pipeline 把 video motion 转化成 pseudo-labeled 训练数据，再用 parameter-efficient cross-attention 注入 egocentric vision，最终接到 RL-based whole-body controller 上。

---

## 2. 整体架构概览

```
                    Stage 1: Language-Motion Pre-Alignment
                    ┌────────────────────────────────────────┐
Video motion ──►   │ 1. Compositional VQ (5 body parts)      │
(no caption)       │ 2. Self-supervised Augmentation         │  ──► 伪标注 QA pairs
                   │    (<Track>, <Time>, <Occlusion>, <State>)│
                   │ 3. Llama3-70B autoregressive training     │
                   └────────────────────────────────────────┘
                                          │
                                          ▼ frozen
                    Stage 2: Vision-Conditioned Fine-tuning
                    ┌────────────────────────────────────────┐
Egocentric RGB ──►  │ Vision encoder + Cross-attention        │
                   │ Q from language, K/V from vision         │
                   └────────────────────────────────────────┘
                                          │
                                          ▼ motion tokens
                    Stage 3: Whole-Body Controller (RL/PPO)
                    ┌────────────────────────────────────────┐
                    │ Goal-conditioned policy π              │
                    │ Output: 24-D joint torques              │
                    └────────────────────────────────────────┘
                                          │
                                          ▼
                       Unitree G1 real-world execution
```

这个 design 的关键 insight：把 universal motion knowledge 和 egocentric grounding 分两阶段学习。Stage 1 学的是 "language ↔ motion" 的 broad statistical relationship，没有 visual 依赖；Stage 2 才把 visual context 作为 residual signal 注入。这种 decoupling 让我们能在 7500 小时 video motion 上 pre-train，再用少量 egocentric 数据 fine-tune。

---

## 3. 技术细节：Compositional Motion Quantization

### 3.1 设计动机

VQ-VAE 是 text-to-motion 的标准做法 (T2M-GPT, MDM)。但是 single codebook 把整个 22-joint SMPL pose 一起量化，粒度太粗。Humanoid-VLA 的关键设计是 **decompositional compression**：把 pose 分成 5 个 part 独立量化：

- left leg
- right leg
- torso
- left arm
- right arm

### 3.2 公式详解

对每个 body part $b \in \{1,2,3,4,5\}$ 训练独立 encoder $\mathcal{E}_b$ 和 codebook $V_b$。在 timestep $t$，body part data $c_t$ 被编码：

$$\hat{z}_t = \mathcal{E}_m(c_t), \quad \mathcal{E}_m = \{\mathcal{E}_b\}_{b=1}^{5}$$

其中 $\hat{z}_t = \{\hat{z}_b\}_{b=1}^{5} \in \mathbb{R}^5$，每个 $\hat{z}_b$ 是从 codebook $V_b$ (size 1024) 中查到最接近 quantization 的离散 token。

Decoder 把 latent 投回 action space：

$$\hat{c}_t = \mathcal{D}_m(\hat{z}_t)$$

完整 loss 借鉴 VQ-VAE 三项：

$$\mathcal{L}_{hvq} = \underbrace{\|c_t - \hat{c}_t\|_2}_{\mathcal{L}_{rec}} + \underbrace{\|\mathbf{sg}(z_t) - \hat{z}_t\|_2}_{\mathcal{L}_{emb}} + \underbrace{\|z_t - \mathbf{sg}(\hat{z}_t)\|_2}_{\mathcal{L}_{com}}$$

- $z_t$ 是 encoder 的连续输出 (pre-quantization)
- $\hat{z}_t$ 是 quantized token (post-quantization)
- $\mathbf{sg}(\cdot)$ 是 stop-gradient operator
- $\mathcal{L}_{rec}$: 重建 L2 loss，让 decoder 输出逼近 input $c_t$
- $\mathcal{L}_{emb}$: 把 codebook embedding 拉近 encoder output (gradient 只流到 codebook)
- $\mathcal{L}_{com}$: 让 encoder output commit 到 codebook (gradient 只流到 encoder)

### 3.3 Intuition：为什么 decompose？

这个设计有 3 个深层好处：

1. **Token-level 编辑能力**：可以精确 mask 掉 "left leg" 的 token，对应 prompt "missing left leg <Occlusion> motion data, please complete"。如果整体量化，无法精细操作单个 body part。

2. **独立 codebook 学习 part-specific 语义**：torso 的运动模式 (root trajectory) 与 left arm 的运动模式 (gesture) 在 latent space 几乎不重叠。共享 codebook 会导致 cross-contamination。

3. **Cross-embodiment generalization**：5 个 part 对应到任何 humanoid robot 都通用 (loco-manipulation 的基本单元)，而 SMPL 22-joint 是 human skeleton specific。

---

## 4. 技术细节：Self-Supervised Data Augmentation

这是这篇 paper 最有 insight 的部分。本质思想：**用 motion 自身的内在结构作为 supervision signal，把 reconstruction 任务伪装成 conditional generation 任务**。

### 4.1 四种 Augmentation 类型

| Type | 操作 | 示例 Prompt |
|------|------|-------------|
| `<Track>` | 提取 root joint 时间轨迹 → 编码成 motion token | "Please move your center position along the trajectory of `<Track>`" |
| `<Time>` | motion 持续时间约束 | "Show me a motion lasting `<Time>` seconds" |
| `<Occlusion>` | mask 掉某些 body joints，模型重建完整 motion | "Missing left arm `<Occlusion>` data. Please complete the motion" |
| `<State>` | 特定 timestep 的 motion pose | "Plan a sequence of actions ending with `<StateN>` over `<Time>` seconds" |

### 4.2 关键 trick：Motion token 嵌入到 language

Augmentation 生成的 motion feature (e.g. `<Track>` 对应的 root 轨迹 token 序列) 直接嵌入到 textual instruction 里，**变成 unified vocabulary** $V = \{V_l, V_m\}$ 中的一部分。LLM 把它当成普通 token 处理。

例如 instruction $l$ 形式化为：

> "Plan a sequence of actions ending with `<State>` over `<Time>` seconds."

其中 `<State>` 是从 $c_t$ 在 timestep $t$ 提取的离散 token $z_t$，`<Time>` 是 duration。

这个设计的妙处：把 "motion → motion" 的 reconstruction 改造成 "language(with motion embedded) → motion" 的 standard conditional generation，直接套用 LLM 的 next-token prediction 框架。

### 4.3 多 Augmentation 组合

Table 6 (Appendix) 展示了 59 种 subtask 模板，每种用 GPT-4 rephrase 成 N 种语言表达，最终扩展为几千个模板。可以组合成 complex tasks：

- `<State> + <Track> + <Caption> → <Motion>`
  - "Starting from `<State1>`, follow the direction of your root guided by `<Track>` to perform the dynamic described by `<Caption>`"

这种组合性让 model 学到 multi-condition grounding，远比单纯 text-to-motion 强。

### 4.4 数据规模

最终 dataset 是 Mao et al. (UH-1) 的 **25 倍**，达到 0.929M clips / 7790 小时。这是 humanoid motion-language interleaved dataset 迄今最大规模。

---

## 5. 技术细节：LLM Training

### 5.1 Foundation model 与 tokenization

用 **Llama3-70B** 作为 backbone。motion codebook $V_m$ (5 × 1024 = 5120 tokens) 与 language codebook $V_l$ 合并为 $V = \{V_l, V_m\}$。所有 instruction tokens $X_d = \{x_d^i\}_{i=1}^N$，$x_d \in V$；所有 motion output tokens $X_o = \{x_o^i\}_{i=1}^L$，$x_o \in V$。

### 5.2 Loss 函数

标准 autoregressive negative log-likelihood：

$$\mathcal{L}_{LLM} = -\sum_i \log p(x_o^i \mid x_o^{<i}, x_d)$$

- $x_o^i$: 第 $i$ 个 output motion token
- $x_o^{<i}$: 之前所有 output tokens (causal mask)
- $x_d$: 完整 input description (motion+language mixed)

### 5.3 两阶段训练策略

| Stage | 数据 | 作用 |
|-------|------|------|
| 1 | Low-quality video motion (大量，不准确) | 建立 broad alignment，学 statistical relationship |
| 2 | High-quality Mocap (小量，精确) | Refinement，确保符合 human kinematics |

Ablation (Table 4) 显示：仅 low-quality + aug → FID 0.698；加 high-quality (no aug) → 0.557；两者 + aug → **0.467** (16% 提升)。证明 dirty data 的 diversity 与 clean data 的 precision 缺一不可。

### 5.4 Implementation

- 8× NVIDIA H100 GPU
- 216 GPU hours
- batch size 4 / device
- learning rate 2e-5
- cosine scheduler
- warmup ratio 0.01
- codebook size 1024 per part

---

## 6. 技术细节：Vision-Conditioned Fine-Tuning

### 6.1 设计动机

Stage 1 学到的是 "language + embedded motion → full motion"。但缺少真实世界感知，无法 react 到第一人称视野下的物体。直接 fine-tune 全部参数会 catastrophic forgetting Stage 1 学到的 broad alignment。

### 6.2 Cross-attention 架构

冻结 Stage 1 的 transformer decoder layers，复制一份并 add cross-attention。在第 $l$ 层：

**Query** 来自 language tokens：
$$Q_l = X_d^l W_Q^l, \quad W_Q^l \in \mathbb{R}^{D_d \times D}$$

**Key / Value** 来自 vision tokens：
$$K_l = X_v^l W_K^l, \quad V_l = X_v^l W_V^l, \quad W_K^l, W_V^l \in \mathbb{R}^{D_v \times D}$$

**Cross-attention output**：
$$X_u^l = \text{Softmax}\left(\frac{Q_l K_l^T}{\sqrt{D}}\right) V_l$$

变量解释：
- $X_d^l$: 第 $l$ 层的 language token feature
- $X_v^l$: 第 $l$ 层的 visual token feature (来自 vision encoder)
- $D$: hidden dimension
- $D_d, D_v$: input dimension of language / vision
- $\sqrt{D}$: scaled dot-product 的标准 scaling

### 6.3 Intuition：Q-K-V 角色分配的哲学

这里 language 作 Query、vision 作 Key/Value 是 reverse of standard image captioning (那里 vision 作 Q)。原因：

- Language 是 **high-level intent** ("kick the ball")，需要被 grounded 到具体场景
- Vision 是 **contextual evidence** (球的位置、距离)
- Language token "queries" 视觉 token 获取 grounding info
- 反过来 (vision 作 Q) 会让 vision 主导，language 沦为被动描述，无法表达 user intent

只训练 cross-attention 层的 $W_Q^l, W_K^l, W_V^l$ 是 **parameter-efficient**，保留 Stage 1 学到的 motion manifold 不被破坏。

---

## 7. 技术细节：Whole-Body Controller

### 7.1 控制方程

$$j_t = \mathcal{P}(s_t, p_t)$$

- $j_t \in \mathbb{R}^{24}$: 24-D joint torques at time $t$
- $s_t$: target body pose (来自 VLA 的 motion prediction)
- $p_t$: humanoid proprioception
- $\mathcal{P}$: goal-conditioned RL policy

### 7.2 RL Setup

- **Algorithm**: PPO (Proximal Policy Optimization, Schulman et al. 2017)
- **Reward**: $R(\mathcal{O}, \mathcal{G})$ 输出 PD controller target position in action space $\mathcal{A}$
- **Optimization**: Adam optimizer 用于把 15 keypoints 映射到 24 humanoid joints (end-effector position alignment)

### 7.3 15 → 24 joint mapping

VLA 生成的是 15 个 universal joints (human + humanoid 共有)，下游执行需要 24 joints。这个 mapping 通过 optimization 保持 end-effector positions 尽量 aligned。这个设计保证 **cross-embodiment universality**：换 robot 时只改 mapping，不动 model。

---

## 8. 实验详解

### 8.1 Kinematic Fidelity (Table 3)

| Method | HumanML3D FID↓ | HumanML3D DIV↑ | Humanoid-S FID↓ | Humanoid-S DIV↑ |
|--------|----------------|----------------|------------------|------------------|
| MDM | 0.889±0.026 | 3.855±0.053 | 2.351±0.590 | 4.111±0.261 |
| T2M-GPT | 0.531±0.020 | 4.555±0.058 | 1.101±0.189 | 4.199±0.218 |
| **Humanoid-VLA** | **0.467±0.018** | **4.585±0.086** | **1.037±0.147** | **4.466±0.213** |

- vs MDM：FID 降低 47.5%
- vs T2M-GPT：FID 降低 12%
- Humanoid-S 是 paper 自己收集的 4646 video clips 含复杂 action，DIV 4.466 比 MDM 高 6%
- FID 越低，分布匹配越好；DIV 越高，多样性越强

### 8.2 Physical Plausibility (Table 2)

四类 metric：

- **$E_{mpjpe}^g$ (global MPJPE, mm)**: 全局 mean per-joint position error
- **$E_{mpjpe}^{pa}$ (PA-MPJPE, mm)**: Procrustes-aligned MPJPE，消除 global scale 和 rotation
- **$E_{accel}$ (mm/s²)**: 加速度误差
- **$E_{vel}$ (mm/s)**: 速度误差

输入条件组合 (D=Description, T=Time, A=Absent parts, $S_n$=State at time n)：

| Difficulty | Input | $E_{mpjpe}^g$ | $E_{mpjpe}^{pa}$ | $E_{accel}$ | $E_{vel}$ |
|------------|-------|---------------|-------------------|--------------|------------|
| Easy | D | 36.13 | 1.53 | 34.42 | 18.73 |
| Easy | T | 36.57 | 1.48 | 35.10 | 18.53 |
| Easy | A | 39.02 | 1.32 | 34.32 | 17.91 |
| Easy | $S_n$ | 36.29 | 1.55 | 34.93 | 18.88 |
| Medium | D+T | **31.07** | 1.18 | **27.84** | **14.76** |
| Medium | D+A | 36.98 | 1.30 | 34.87 | 18.16 |
| Medium | D+$S_n$ | 35.75 | 1.18 | 33.41 | 17.18 |
| Hard | D+$S_1$+$S_N$ | 37.14 | 1.34 | 34.69 | 18.08 |

- 全部 $E_{mpjpe}^g < 40mm$，最佳 31.07mm
- $E_{mpjpe}^{pa}$ 最低 1.18mm，shape accuracy 极高
- $E_{accel}$ 最低 27.84 mm/s²，motion smooth
- **D+T 表现最好** —— 给定 description + duration 时 motion 生成最稳；而 D+A (有缺失 body part) 任务最难

### 8.3 Ablation on Data Augmentation (Table 4)

| Low-quality | High-quality | w/ aug | FID↓ | DIV↑ |
|------------|--------------|--------|------|------|
| ✓ | | ✓ | 0.698±0.037 | 4.576±0.098 |
| ✓ | ✓ | | 0.557±0.016 | 3.867±0.062 |
| ✓ | ✓ | ✓ | **0.467±0.018** | **4.585±0.086** |

关键 takeaway：
- 仅 dirty data + aug 已经能到 0.698 (DIV 4.576 高) — aug 提供 diversity
- 加 clean data 没 aug 到 0.557 (DIV 降到 3.867) — clean data 提供 precision 但 loss of diversity
- 两者 + aug 达到 0.467 / 4.585 — sweet spot

### 8.4 Vision Integration (Table 5, Real Robot)

Unitree G1 上的 real-world 实验，10 次重复：

| Task | Success Rate |
|------|--------------|
| Turn to an object | 10/10 |
| Hold an object | 9/10 |
| Wave to people | 10/10 |
| Avoid an obstacle | 9/10 |
| Jump over an object | 9/10 |
| Dance with a partner | 8/10 |
| Punch an obstacle | 10/10 |
| Kick a ball | 9/10 |

平均 ~93.75% 成功率。"Dance with a partner" 最低 (8/10)，因为需要持续 spatial tracking partner 位置，对 vision grounding 要求最高。

---

## 9. Intuition 总结

把整篇 paper 抽象成几条核心 intuition：

1. **Decoupled learning**：先学 broad statistical motion-language alignment (无需 ego vision)，再学 ego grounding。这避免在小规模 ego dataset 上 overfit，也避免 broad motion manifold 被 ego 数据 bias 掉。

2. **Motion as part of language vocabulary**：通过把 motion token 加入 LLM 词表，所有 motion operation 变成 LLM 的 next-token prediction。这套 design 让 self-supervised augmentation 自然 fall out —— motion 内在结构变成 conditional generation 的输入部分。

3. **Compositional VQ 是 fine-grained control 的 enabler**：5-part 分解让 token-level 编辑成为可能，没有它 `<Occlusion>` augmentation 无法设计。这呼应了 LLM tokenization 的核心哲学 ——"discrete + composable"。

4. **Self-supervision through structural priors**：`<Track>`、`<State>`、`<Time>` 都是从 motion 内部可计算的特征，不需要外部 caption。把 reconstruction 改写成 generation 的 trick 是把无标注数据塞进 supervised learning framework 的关键。

5. **Q from language, KV from vision 的 cross-attention 哲学**：language 主导 intent，vision 提供 grounding。这避免了 vision feature 主导而 language 沦为被动描述。

6. **Two-stage training = dirty + clean**：dirty data 提供 statistical diversity (broad alignment)，clean data 提供 physical precision (refinement)。这是 LLM pretrain + SFT 范式在 motion generation 上的对应。

---

## 10. 与相关工作的对比

| Method | 类型 | 关键限制 |
|--------|------|----------|
| Exbody (Cheng et al. 2024) | upper-body retargeting | 只 upper-body，reactive |
| Exbody2 (Ji et al. 2024) | expressive whole-body | reactive，无 vision |
| HARMON (Jiang et al. 2024) | language→motion | reactive |
| H2O (He et al. 2024b) | teleop whole-body | 需要 teleop，reactive |
| OmniH2O (He et al. 2024a) | universal teleop | reactive |
| UH-1 (Mao et al. 2024) | massive video learning | reactive，无 ego vision |
| RT-2 (Brohan et al. 2023) | VLA for arm | 只 arm |
| OpenVLA (Kim et al. 2024) | VLA for arm | 只 arm |
| GR-2 (Cheang et al. 2024) | VLA for arm | 只 arm |
| RDT-1B (Liu et al. 2024b) | diffusion VLA bimanual | 只 arm |
| QUAR-VLA (Ding et al. 2025) | VLA quadruped | quadruped，非 humanoid |
| QUART-Online (Tong et al. 2024) | quadruped online LMM | quadruped |
| π₀ (Black et al. 2024) | VLA flow multi-robot | 多机器人但非 humanoid focus |
| HumanVLA (Xu et al. 2024) | VLA for physical humanoid | 限特定 platform |
| **Humanoid-VLA (本 paper)** | **VLA universal humanoid** | **首个 humanoid VLA** |

---

## 11. Limitations 与启示

Paper 自己提到的 (Appendix E)：

1. **RL policy robustness 不够**：whole-body controller 在 complex loco-manipulation 上还脆。
2. **High-quality data 有限**：Mimicking-Bench 这类 dataset 限特定 robot config，不能直接用。
3. **训练方法相对简单**：未充分利用现有数据。

我读出的额外 insight：

- **Self-supervised augmentation 的天花板**：四种 augmentation 都是从 motion 自身派生的 structural feature。这意味着 model 学到的 "language" 其实是 motion 的内部描述，与真正 human language 的 semantic richness 还有 gap。如何引入真实 caption（如 VLLM-generated）+ augmentation caption 的混合训练是一个方向。
- **Cross-attention 的限制**：只 fine-tune cross-attention 简单有效，但 vision 与 motion 的 deep binding 可能需要更深的 fusion。可以考虑 MoE-style adapter 或 perceiver-style late fusion。
- **15 vs 22 vs 24 joints 的 mapping loss**：从 universal 15 joints 映射到 robot-specific 24 joints 用 Adam optimizer，这个 step 是 post-hoc optimization。end-to-end 学习 mapping 可能更优。
- **Sim-to-real gap**：Table 2 的 physical plausibility 是 IsaacGym 仿真；real-world (Table 5) 只测成功率，没有 quantitative tracking error。real 上的 motion quality 还是 open question。

---

## 12. References & Web Links

### Core Paper
- Humanoid-VLA (本篇): arXiv search "Humanoid-VLA Towards Universal Humanoid Control"
- Llama 3: [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)
- PPO: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- VQ-VAE: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

### Humanoid Control Baselines
- Exbody: [https://arxiv.org/abs/2402.16796](https://arxiv.org/abs/2402.16796)
- Exbody2: [https://arxiv.org/abs/2412.13196](https://arxiv.org/abs/2412.13196)
- H2O: [https://arxiv.org/abs/2403.04436](https://arxiv.org/abs/2403.04436)
- OmniH2O: [https://arxiv.org/abs/2406.08858](https://arxiv.org/abs/2406.08858)
- UH-1 (Mao et al.): [https://arxiv.org/abs/2412.14172](https://arxiv.org/abs/2412.14172)
- HARMON: CoRL 2024 Workshop
- Mobile-television: [https://arxiv.org/abs/2412.07773](https://arxiv.org/abs/2412.07773)
- PHC: [Perpetual Humanoid Control](https://arxiv.org/abs/2304.01150)

### VLA Models for Manipulation & Quadruped
- RT-2: [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- OpenVLA: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- GR-2: [https://arxiv.org/abs/2410.06158](https://arxiv.org/abs/2410.06158)
- RDT-1B: [https://arxiv.org/abs/2410.07864](https://arxiv.org/abs/2410.07864)
- RoboMamba: NeurIPS 2024
- QUAR-VLA: ECCV 2024
- QUART-Online: [https://arxiv.org/abs/2412.15576](https://arxiv.org/abs/2412.15576)
- π₀ (pi-zero): [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- HumanVLA (Xu et al. 2024): [https://arxiv.org/abs/2406.19972](https://arxiv.org/abs/2406.19972)
- Mimicking-Bench: [https://arxiv.org/abs/2412.17730](https://arxiv.org/abs/2412.17730)

### Motion Datasets & Methods
- AMASS: [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/)
- HumanML3D: [Generating Diverse 3D Human Motions from Text](https://arxiv.org/abs/2105.02679)
- Motion-X: [NeurIPS 2023](https://arxiv.org/abs/2307.00272)
- Human3.6M: [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/)
- SMPL: [SMPL Paper](https://smpl.is.tue.mpg.de/)
- MDM (Motion Diffusion Model): [ICLR 2023](https://arxiv.org/abs/2209.14915)
- T2M-GPT: [CVPR 2023](https://arxiv.org/abs/2301.06098)
- TRAM (Wang et al. 2025): [ECCV 2024](https://arxiv.org/abs/2407.07182)
- VILA: [CVPR 2024](https://arxiv.org/abs/2312.07533)

### Vision-Language Models
- Visual Instruction Tuning (Liu et al. 2023): [LLaVA](https://arxiv.org/abs/2304.08485)
- Video-LLaMA: [https://arxiv.org/abs/2306.02858](https://arxiv.org/abs/2306.02858)
- GPT-4: [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)

### Simulation
- IsaacGym: [https://arxiv.org/abs/2108.10470](https://arxiv.org/abs/2108.10470)

---

## 最后的思考

这篇 paper 的核心 contribution 不是单点突破，而是 **system-level 整合**：

1. Compositional VQ 让 fine-grained motion editing 成为可能
2. Self-supervised augmentation 把 7500 小时无标注 video motion 转化为有监督训练对
3. Cross-attention parameter-efficient tuning 在保留 Stage 1 知识的前提下注入 vision
4. PPO-based whole-body controller 把 LLM 输出接地到 robot hardware
5. 15-joint universal representation 保证 cross-embodiment generalization

它本质上展示了 LLM 时代的 humanoid control 范式：把 motion 看作 language 的一部分，用 next-token prediction 统一 text+motion+vision 三模态。这跟 RT-2 / OpenVLA 的思路在 manipulation 上的成功是一致的，只是 humanoid 的 whole-body + loco-manipulation 难度更高。

值得 follow 的方向：(1) 把 augmentation 与 VLLM captioning 结合；(2) end-to-end joint mapping 学习；(3) 在 cross-attention 之外探索更深 vision-motion fusion (e.g. perceiver, MoE adapter)；(4) 把 RL controller 与 LLM policy joint train，让 physics constraint 反向影响 motion generation。
