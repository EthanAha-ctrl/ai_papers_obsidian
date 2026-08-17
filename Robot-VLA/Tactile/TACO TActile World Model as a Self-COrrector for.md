---
source_pdf: TACO TActile World Model as a Self-COrrector for.pdf
paper_sha256: 7ba1b4e646524a54f753e7ecc1cce8d3c834173c2848725f7d6a9302ab19e9c7
processed_at: '2026-08-12T12:12:39-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TACO 用人话说一遍

## 一句话说清楚

机器人做"需要用力气、需要精确接触"的任务时，经常会卡在一种很尴尬的失败上：**它知道该干嘛，位置也对，但接触那一下出问题了**——擦白板力度不够擦不掉、拧瓶盖打滑拧不动、插花对不准插不进去。光看摄像头完全看不出问题在哪，但力传感器上信号天差地别。TACO 就是让一个"能同时想象视频和力的 world model"自己去生成纠正数据，把机器人从这种失败里救回来，不需要人再来示范。

---

## 这到底是个什么问题

VLA 模型（vision-language-action，像 π0.5、OpenVLA 这些）在一般抓取任务上已经很能打了，但遇到 contact-rich manipulation（需要精细物理接触的任务）就开始拉胯。

举个具体例子，Twist Bottle Cap 任务：
- 机械手正确地抓到了瓶盖
- 视觉上：机械手稳稳握住 cap，位置完美
- 但：没有产生有效的 twisting torque，cap 纹丝不动
- 结果：任务失败，而且从 RGB 帧序列里你根本看不出哪一步开始坏的

这种失败有个特点：**localized 而非 semantic**。policy 在 semantic 层面全对（知道要抓、要拧、要提），只是在 contact transition 那一瞬间，物理接触出了问题——slippage（打滑）、insufficient pressure（压力不够）、abnormal torque（力矩异常）。这些信息在 RGB 里几乎不可见，在 force-torque 信号里却非常明显。

---

## 为什么之前的办法都不行

### 1. 让人来纠正（DAgger 那一套）

DAgger、IntervenGen、MILE、Hi-WM 这些方法的核心思路：机器人一卡住，人就用 SpaceMouse 接管，给出 recovery demonstration。问题是这事儿 scale 不起来——你得一直盯着机器人，一失败就介入，人力成本极高。

- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- IntervenGen: https://arxiv.org/abs/2404.13872

### 2. 用 vision-only world model 想象未来

RoboDreamer、DreamGen、VLAW 这些工作用 world model 想象"如果执行某个 action 会怎样"，然后用想象出来的数据 post-train policy。问题在于：**一个视频看起来 plausible，接触力可能完全违反物理**。生成的视频里 eraser 看起来擦掉了 mark，但 force sequence 可能只有 0.1N，根本擦不掉。vision-only 的想象在 contact-rich 场景下不可靠。

- RoboDreamer: https://arxiv.org/abs/2404.12377
- DreamGen: https://arxiv.org/abs/2505.12705

### 3. 直接把 force 喂给 VLA 做 fine-tune

ForceVLA、Tactile-VLA 这些工作尝试给 VLA 加 force input。但一旦你做 full fine-tune，tactile 的 gradient 会回流到 PaliGemma 这种 vision-language backbone，把原本的视觉语义 prior 搞坏——结果就是：机器人学会用力了，但连"东西在哪"都看不准了。

- ForceVLA: https://arxiv.org/abs/2507.06409
- Tactile-VLA: https://arxiv.org/abs/2507.09160

---

## TACO 的核心思路

**让 world model 同时想象 video 和 force，自己生成"纠正数据"，然后只把这些数据喂给 action expert，不去碰 VLM backbone。**

整个流程是一个闭环：

```
real rollout 采失败数据
  → recognize 找到失败的 state
  → imagine 想象 49 步正确的 video+force 轨迹
  → label 给想象出来的轨迹标上 action
  → post-train 只训 action expert
  → 部署新 policy 再采失败数据
  → 循环
```

这个 loop 的关键在于：**所有"纠正"都是 world model 自己想出来的，没有人在里面**。real world 只负责提供"哪里失败了"这个信号，imagination 负责生成"应该怎么做"，policy 负责学。

---

## 三步走：Recognize–Imagine–Label

### Step 1: Recognize（识别哪里开始坏的）

部署当前 policy 去采 rollout。对每个 timestep，用 unified progress-action model 预测一个 progress score $p_t \in [0,1]$，表示"任务完成了多少"。

当 progress 停滞或下降时，那个 state 就是 failure-adjacent（即将失败）：

$$\mathcal{S}_{\text{anchor}}^{(k)} = \{(\tau, t) \mid p_{t+\Delta} - p_t < \epsilon\}$$

- $\tau$: 一条 rollout
- $t$: timestep
- $\Delta$: lookahead 窗口（看未来几步 progress 有没有涨）
- $\epsilon$: 阈值，progress 涨幅小于这个就算 stalled
- $\mathcal{S}_{\text{anchor}}^{(k)}$: 第 $k$ 轮所有"需要纠正的锚点"

每条失败轨迹最多取 10 个 anchor，让纠正集中在 contact transition 附近。

同时分配 advantage label：
- 失败轨迹第一个 anchor 之前：$y_t = 1$（还在正轨上）
- 第一个 anchor 及之后：$y_t = 0$（已经偏了）
- expert demo 全程：$y_t = 1$
- 想象出来的 correction：$y_t = 1$（因为这是"正确做法"）

### Step 2: Imagine（想象 49 步正确的 video+force）

从每个 anchor state 出发，让 visuo-tactile generation model 采样：

$$(\hat{I}_{t:t+T}, \hat{F}_{t:t+T}) \sim G_\psi(\cdot \mid I_t, F_t, l)$$

- $I_t$: 当前 RGB 帧
- $F_t$: 当前 12 维 force-torque（左右手指各 6-DoF）
- $l$: language instruction
- $T = 49$: 想象 49 步未来
- 输出：$\hat{I}$（想象的视频）和 $\hat{F}$（想象的力）

这 49 步同时包含视觉演化和接触力演化，物理上是一致的——因为它们是联合 denoise 出来的。

### Step 3: Label（给想象轨迹标 action）

用同一个 progress-action model 对想象出来的 $(\hat{I}, \hat{F})$ 预测 action：

$$(\hat{a}_{t:t+T}, \hat{p}_{t:t+T}) = U_\phi(\hat{I}_{t:t+T}, \hat{F}_{t:t+T})$$

- $\hat{a}_t \in \mathbb{R}^7$: 7-DoF end-effector action
- $\hat{p}_t \in [0,1]$: progress（用来 verify 想象的轨迹确实在推进任务）

这样想象出来的轨迹就变成了可执行的监督数据：$(\hat{I}, \hat{F}, \hat{a}, y=1)$。

---

## World Model 怎么同时生成 video 和 force

### Backbone

用的是 **Wan2.2-TI2V-5B**，腾讯出的 5B 参数 text-to-video 模型。先在 DROID（20 万条机器人轨迹）、AgiBot、RoboMIND 这些大数据集上预训练，让模型有 visual realism 和 robot-scene consistency 的 prior，再在 contact-rich demo 上 fine-tune。

- Wan2.2: https://arxiv.org/abs/2503.20314
- DROID: https://arxiv.org/abs/2403.12945

### Token 拼接

- Video latent tokens: $X^v \in \mathbb{R}^{B \times 12 \times 3072}$（12 个 video latent token）
- Force tokens: $X^f = T_\eta(F) \in \mathbb{R}^{B \times 49 \times 3072}$（49 个 force token，比 video 时间密 4 倍）

为什么 force 比 video 密？因为接触力的变化频率远高于视觉帧。你擦白板时，1 帧视频里 eraser 看起来没动，但 force 可能已经从 2N 涨到 8N 又掉回 1N 了。

拼接：$X = [X^v; X^f]$，沿 token 维拼。关键是 video 和 force token 在 DiT self-attention 里**双向交互**——force 不是单纯的外部 condition，而是参与 attention 计算。这就保证了生成的视频和力在物理上一致。

### Joint flow matching loss

$$\mathcal{L}_{\text{joint}} = \|u_\psi^v - (\xi_1^v - \xi_0^v)\|_2^2 + \lambda_f \|u_\psi^f - (\xi_1^f - \xi_0^f)\|_2^2$$

- $u_\psi^v$: DiT 预测的 video flow velocity
- $u_\psi^f$: tactile head 投影后的 force flow velocity
- $\xi_1^v, \xi_1^f$: clean video latent 和 clean force
- $\xi_0^v, \xi_0^f$: Gaussian noise
- $(\xi_1 - \xi_0)$: flow matching 的 target velocity（从 noise 到 clean 的直线方向）
- $\lambda_f$: force 项的权重

video 和 force **共享同一个 denoising timestep** $\sigma$，保证两者在同一 noise level 同步去噪。如果 video 已经接近 clean 而 force 还是噪声，两者会 mismatch，生成出来的视频和力对不上。

### Temporal RoPE alignment（这里有个很聪明的细节）

Wan2.2 原生的 RoPE 作用在 3D video latent grid $(t, h, w)$ 上。force token 只有时间维度，没有空间维度。如果直接拼进去，RoPE 的 spatial 部分会乱掉。

TACO 的做法：把 force token 的 temporal position 映射到 video latent 的 temporal axis：

$$\rho(i) = \text{round}\left(\frac{i}{T-1}(f-1)\right), \quad i = 0, ..., T-1$$

- $i$: force token 的 index（0 到 48）
- $T = 49$: force token 总数
- $f$: video latent 的 temporal length
- $\rho(i)$: force token $i$ 对应到 video latent 的时间位置

每个 force token 用 $\rho(i)$ 处的 temporal RoPE，spatial RoPE 设为 $1 + 0j$（复数单位元，不引入任何 spatial 偏置）。

人话翻译：告诉 attention 机制"force token 在空间上是统一的，只在时间上和 video 对齐"。第 $i$ 个 force token "属于" video latent 的第 $\rho(i)$ 个时间槽，和那个槽里所有 spatial token 都参与 attention。

### First-frame force anchoring

保留 $F_0$（第一帧 force）clean，不注入 noise。这给模型一个明确的"现在接触状态是什么"的锚点，不用从 noise 里猜起始 force。在 flow matching 里，如果第一帧 force 也是噪声，模型要猜这是 slip 还是 insufficient pressure，这是 ambiguous 的——视觉上两种情况一样。

---

## Unified Progress-Action Model（UPA）：一个模型干两件事

UPA 同时负责：
1. 预测 corrective action（给想象轨迹标 action 用）
2. 预测 dense progress（识别 failure-adjacent state 用）

### 架构

- **Visual pathway**: DINOv2-with-Registers backbone，输出 $37 \times 37$ patch features（dim=768），再接 direction-aware decoder（4 个 dilated conv branch + angle-sensitive pooling），输出 1024 维 visual embedding $z_t^v$
- **Tactile pathway**: 2 层 MLP（hidden=128, output=256），输出 256 维 tactile embedding $z_t^f$
- **Fusion**: $[z_t^v; z_t^f] \in \mathbb{R}^{1280}$ 拼接
- **Action head**: 512 维 hidden，输出 $\hat{a}_t \in \mathbb{R}^7$
- **Progress head**: 256 维 hidden + sigmoid，输出 $\hat{p}_t \in [0,1]$

- DINOv2: https://arxiv.org/abs/2304.07193

### Joint loss

$$\mathcal{L}_{\text{UPA}} = \text{SmoothL1}(\hat{a}_t, a_t) + m_t \|\hat{p}_t - p_t\|_2^2$$

- $\hat{a}_t, a_t$: 预测和真实 action
- $\hat{p}_t, p_t$: 预测和真实 progress
- $m_t \in \{0, 1\}$: progress label 的 mask（有些帧没标注就置 0）

为什么 action 和 progress 联合训？因为 contact cue 同时影响"该做什么动作"和"任务进展到哪了"，两个 task 互相正则。action prediction 让 visual feature 学到 actionable 表示，progress prediction 让 feature 学到 task-stage 结构。一个 model 干两件事，还省了训两个 model 的开销。

---

## Knowledge-Insulated Tactile Adaptation：保护 VLM prior 的关键

### 问题

想象出来的 correction 数据是 tactile-heavy 的（force 信号主导）。如果直接 fine-tune 整个 VLA，force 的 gradient 会回流到 PaliGemma (2B) vision-language backbone，把 visual-language prior 搞坏。结果：机器人学会用力了，但连"东西在哪"都看不准。

### 解法

基于 π0.5 架构（PaliGemma 2B + 300M action expert）：

1. Image + language + state 编码为 VLM prefix token $z_t$
2. **Stop-gradient**: $z_t = \text{sg}[\text{VLM}(\text{image, language, state})]$，VLM backbone 完全冻结
3. Force history（长度 8，12 维）+ advantage 通过 encoder 编为 256 维 embedding
4. 这些 condition 通过 **adaRMSNorm** 注入 action expert
5. 只训：tactile encoder + adaptation layer + action expert，VLM 一个参数都不动

- π0.5: https://arxiv.org/abs/2504.16054

人话翻译：action expert 可以"看到" VLM 抽出来的视觉特征，但 gradient 不会回流到 VLM。VLM 的视觉语言能力完全保留，action expert 只学"在 frozen VLM feature 之上 + force/advantage condition 下如何预测 action"。

### Advantage-Conditioned Training

Action expert 的 flow matching loss：

$$\mathcal{L}_\pi = \mathbb{E}\left[\|u_\theta(x_\sigma, \sigma \mid z_t, \tilde{c}_{\text{adaRMS}}) - (\epsilon - a_t)\|_2^2\right]$$

conditioning：

$$c_{\text{adaRMS}} = c_t + \lambda_f c_f + \lambda_a c_a$$

- $x_\sigma = \sigma \epsilon + (1-\sigma) a_t$: noisy action chunk
- $\sigma \in [0,1]$: noise level
- $\epsilon \sim \mathcal{N}(0,I)$: noise
- $a_t$: clean action
- $c_t$: flow timestep condition
- $c_f$: force condition（8 步 history，256 维）
- $c_a$: advantage condition（256 维，0 或 1）
- $\lambda_f, \lambda_a$: 权重

**Classifier-Free Guidance (CFG)**: 训练时 condition 以 0.1 概率被替换成 null embedding（让模型同时学 conditional 和 unconditional）。推理时 condition on positive advantage $y=1$，把 action expert 推向 high-progress recovery behavior。

这本质上是把 offline RL 的思路用 binary advantage label 实现了：正样本（成功/纠正）推 policy 向前，负样本（失败）让 policy 远离 stalled state。推理时用 CFG "推"一把，让 policy 倾向 recovery behavior。

- π0.6（advantage training 的 inspiration）: https://arxiv.org/abs/2511.14759

---

## 实验结果

### Setup

- 硬件：Franka Research 3 (FR3) 单臂 + Xense tactile sensor（gripper 指尖，6-DoF force/torque）+ Intel RealSense D455 前视相机
- 6 个 task：Insert Flower、Wipe Whiteboard、Twist Bottle Cap、Play Xylophone、Toast Bread、Move Hanoi Rings
- 每个 task 50 个 SpaceMouse teleop demo
- Base policy：π0.5 warm-start
- 评估：每个 task 40 个 episode，随机化物体位置

### 主结果（Table 1）

| 方法 | Insert | Wipe | Twist | Xylo | Toast | Hanoi | Ave SR |
|------|--------|------|-------|------|-------|-------|--------|
| Base Policy | 0.50 | 0.51 | 0.45 | 0.46 | 0.30 | 0.08 | 0.38 |
| Filtered BC (Iter2) | 0.52 | 0.57 | 0.48 | 0.51 | 0.36 | 0.11 | 0.43 |
| TACO w/o KI (Iter2) | 0.62 | - | 0.65 | - | 0.78 | 0.50 | 0.50 |
| **TACO (Iter2)** | **0.93** | **0.65** | **0.98** | **0.52** | **0.81** | **0.51** | **0.82** |

**TACO 相对 base policy: +44% absolute success rate**

### 三个 baseline 各自为什么不行

**Filtered BC**: 只在成功的 rollout 上做 supervised fine-tune。但成功的 rollout 里**没有 recovery behavior**——成功的轨迹一路顺畅，不存在"卡住又恢复"的模式。所以 Filtered BC 只能 reinforce 已有的 narrow action manifold，遇到 contact 失败就无能为力。这正是 DAgger 经典论断的体现：在 policy 自己的成功分布上训练，分布必然 narrow。

**TACO w/o KI**: 有 imagined correction，但 full VLA end-to-end fine-tune，tactile gradient 侵蚀 VLM prior。pre-contact perception 退化，机械手连"东西在哪"都看不准。Wipe Whiteboard 从 0.55 掉到 0.33 就是这个原因——失去 spatial grounding 后，抓 eraser 和定位 star 都变差。

**TACO full**: 两者兼得。VLM prior 保护 pre-contact perception（可靠接近目标），tactile correction 提供接触恢复能力（有效调整力）。

### Ablation: tactile 的双重作用（Figure 5）

| 设置 | Force Val Loss | Action Val Loss | VOC | FL | Real SR |
|------|----------------|-----------------|-----|-----|---------|
| w/o tactile generation | 0.004 | 0.025 | 0.78 | 0.87 | **0.28** |
| w/o tactile labeling | 0.002 | 0.038 | 0.88 | 0.90 | **0.65** |
| TACO full | 0.002 | 0.019 | 0.94 | 0.95 | **0.82** |

- VOC: video frame-wise progress rank correlation（shuffle 帧后预测 progress 和真实时序的 rank correlation）
- FL: failure localization accuracy（手动标注 failure frame，看 UPA 能不能检测到）

**两个关键结论**：

1. **w/o tactile generation**（vision-only world model，UPA label 时用 force）：SR 掉到 28%。想象的 video 本身没有 force 约束，contact dynamics 已经错了，给 label 用的 force 也是错的。光看视频想不出正确的 contact transition。

2. **w/o tactile labeling**（generate force 但 label 时不用 force）：SR 掉到 65%。Force 已经生成出来了，但 UPA label action 时只看 video，force 信号被浪费。Force 必须直接参与 corrective action prediction，仅作 auxiliary observation 不够。

### Scaling of Imagined Correction Data（Figure 5 右）

Insert Flower 上 real-to-imagined ratio：
- 1:2 → SR 70%
- 1:4 → SR 93%
- 1:8 → SR 97%

Toast Bread 上：
- 1:2 → 55%
- 1:5 → 81%
- 1:10 → 90%

**想象的数据越多，SR 越高，而且没饱和**。1:8 还比 1:4 好，说明 broader coverage of failure-adjacent contact state 比 narrow high-quality 更有效。这非常符合 world model 作 self-corrector 的核心论点：**想象的边际成本几乎为零，可以无限 scale，这正是替代 human intervention 的关键**。

### Advantage-Conditioned Ablation（Figure 10b）

| 方法 | Insert Flower | Wipe Whiteboard |
|------|---------------|-----------------|
| TACO | 93% | 65% |
| TACO w/o Advantage | 83% | 56% |

去掉 advantage-conditioned，把 failed rollout 全丢掉，只在 success + imagined correction 上做 supervised fine-tune，SR 掉 9-10 个点。失败 rollout 提供 "negative signal"——哪些 contact state 应该 avoid。没有 advantage label，policy 分不清 high-progress correction 和 stalled failure，CFG 推理时也没法用 positive advantage 推向 recovery。

### Anchor Selection Ablation（Figure 10c）

| 方法 | Insert Flower | Wipe Whiteboard |
|------|---------------|-----------------|
| TACO | 93% | 65% |
| TACO Uniform Anchor | 78% | 25% |

Uniform anchor（从 failed rollout 随机取同样多 anchor）在 Insert Flower 掉 15 个点，在 Wipe Whiteboard 掉 40 个点。Random anchor 会浪费 imagination budget 在远离 failure 的 state 或已经 unrecoverable 的 state 上，产出的 correction 与真实 failure 不匹配。**Recognize step 的 progress-guided anchor selection 至关重要**。

### Action Distribution（Figure 6）

Insert Flower 上固定初始位置，评估 40 个 successful rollout，投影 end-effector pose 到 XY 平面：

- Expert demo：主流形 + 少量扰动
- Base policy：narrow 集中在 demo manifold
- Filtered BC：与 base 几乎一致，不能扩展
- TACO Iter1：broader 分布
- TACO Iter2：进一步拓宽

Base policy 和 Filtered BC 都困在 narrow demonstration manifold，对 execution error 敏感，一偏就无 recovery。TACO 通过 imagined correction 暴露 policy 于 diverse successful recovery behavior，显著 broaden action distribution。这呼应了 RL 中 "exploration broaden policy distribution" 的思想，但 TACO 用 world model imagination 替代真实 exploration，极低成本实现 distribution broadening。

### OOD Generalization（Figure 7, 12）

Wipe Whiteboard 三个 OOD 设置：

| Setting | Base Policy | TACO (1 iter) |
|---------|-------------|---------------|
| Unseen Background | 20.0% | 80.5% |
| Unseen Object (sponge 替代 eraser) | 35.0% | 85.5% |
| Unseen Position | 13.5% | 55.0% |

这些 OOD setting 在 vision、tactile、action 三个维度都超出了 world model 的 training distribution，但 tactile-aware world model 仍能 generate effective correction。这说明 **tactile dynamics 的 physical regularity 比 visual appearance 更 transferable**——force-torque 信号对 object geometry/texture shift 的 robustness 比 RGB 高。

### Failure Case（Figure 11）

三个 task 上对比 Filtered BC、TACO w/o KI、TACO：

- **Filtered BC**: 接触前正常，contact 时 stall。Wipe 没 force，Hanoi 没 align，Twist 没 torque。困在同一个 narrow behavior。
- **TACO w/o KI**: 有 imagined correction 学到 recovery，但 VLM prior 被侵蚀，pre-contact perception 差。off-target wiping, misaligned insertion, unstable engagement。
- **TACO**: reliable approach（VLM prior 完整）+ effective contact recovery（tactile correction）。三个 task 全部完成。

**结论**: imagined visuo-tactile correction 和 knowledge-insulated adaptation 是互补的，缺一不可。

---

## 几个值得细想的设计

### 1. 为什么 force 用 12 维而非 tactile image

Xense sensor 输出 6-DoF force-torque，两个 sensor 拼成 12 维。相比 GelSight 的高分辨率 tactile image，12-D F/T 更紧凑，更容易 tokenize，也更接近物理力学的直接表示。Flow matching 在低维度上更稳定，adaRMSNorm conditioning 也更轻量。

- 3D-ViTac（tactile image based）: https://arxiv.org/abs/2410.24091

### 2. Binary advantage 为什么够用

复杂的 advantage（如 RL 中的 $A(s,a) = Q(s,a) - V(s)$）需要 value function 估计，在 manipulation 中很难学好。Binary advantage 等价于"成功/失败"的简单 reward，配合 CFG 就能把 policy 推向 positive 区域。简单、稳定、有效。

### 3. 为什么要用 π0.5 作 base

π0.5 是 flow matching VLA，PaliGemma 2B + 300M action expert 的架构天然适合 knowledge insulation——VLM backbone 和 action expert 是分开的 module，stop-gradient 很自然。如果用一个 monolithic transformer，knowledge insulation 就难做。

### 4. 为什么 offline generation 而非 online

Wan2.2-TI2V-5B 推理慢，online generation（部署时实时跑 world model）不现实。TACO 选择 offline 生成 correction 数据，post-train 后部署的还是单步 inference 的 policy。这是工程考量——牺牲了 runtime correction 能力，换来部署时的低延迟。

---

## 和相关工作的对比

| 对比对象 | 区别 |
|----------|------|
| RoboDreamer / ManipDreamer | vision-only world model，contact-rich 上不可靠。TACO 加 force joint denoise + UPA 联合 action/progress |
| DAgger / IntervenGen | 需要 human 介入，scale 不起来。TACO world model 自己 generate correction |
| ForceVLA / Tactile-VLA | 直接把 force 喂 VLA，full fine-tune 损害 VLM prior。TACO knowledge-insulated，force 只到 action expert |
| π0.6 | 真实环境 rollout + binary success 上做 RL。TACO 在 world model imagination 上做 advantage training，real world 只用于 failure detection |
| DreamGen / VLAW | world model 作 full simulator 跑 RL。TACO 只 generate local correction segment，更轻量，避免长 horizon 累积误差 |
| Hi-WM | human 提供 correction signal。TACO 完全 autonomous |

---

## 局限与未来方向

### 论文承认的局限

Imagined correction 是 offline generation，部署时没有 runtime correction 能力。

### 我觉得可以探索的方向

1. **Online correction with distilled world model**: 把 Wan2.2-TI2V-5B 蒸馏成小模型，部署时实时跑，实现 MPC + world model 风格的 online correction。

2. **World model 与 policy tighter coupling**: 让 policy 直接在 world model latent space 中规划（latents as actions），类似 DreamerV3 的 latent imagination。当前 world model 和 policy 还是分开训的。
   - DreamerV3: https://arxiv.org/abs/2301.04104

3. **Tactile representation learning**: 当前 force 用 12-D F/T，信息密度有限。引入 GelSight tactile image + 自监督 representation learning，可以捕捉更精细的 contact geometry（slip direction, contact patch, surface texture）。

4. **Multi-finger / bimanual**: 单臂 + parallel gripper 扩展到 multi-finger hand 或 bimanual，contact state dimensionality 暴增，tactile world model 复杂度也增加。

5. **Long-horizon task**: 49 步 correction horizon 对 multi-stage task（如 Hanoi Rings 4 个 stage）可能不够，需要 hierarchical imagination。

6. **Iteration divergence**: 论文只跑 2 轮迭代，多轮 post-train 是否会 collapse 到 narrow recovery mode？长期 dynamics 未知。

---

## 我的整体 take

TACO 的核心 insight 浓缩成一句：**contact-rich manipulation 的失败是 localized tactile dynamics 失败，所以 correction 应该 localized 在 contact transition 附近，supervision 应该是 visuo-tactile，post-training 应该保护 pre-contact perception。**

这套设计有几个 nice property：

1. Failure 是 self-identified 的——progress stall 检测，不需要人标
2. Correction 是 self-generated 的——world model 想象，不需要人 demo
3. Action 是 self-labeled 的——UPA 在 imagined segment 上预测，不需要单独的 inverse dynamics model
4. VLM prior 是 self-protected 的——stop-gradient，不需要 LoRA 折中
5. 整个 loop 是 autonomous 的——real → imagine → real 闭环

Karpathy 你在 Tesla 讲过 "data is the bottleneck"。TACO 正是 attack 这个 bottleneck：不再依赖 human teleop demo，而是 build 一个 self-corrector 把 failure 转 correction。这与 self-driving 的 "shadow mode + auto-labeling" 思路相通——把"想象"任务交给 world model，让数据飞轮自己转起来。

潜在风险：
- World model hallucination——imagination 可能 physically inconsistent，需要 verification mechanism
- Distribution shift——OOD 实验只测了 1 轮，更 extreme OOD（unseen action space）可能失败
- Iteration divergence——多轮 post-train 的长期 dynamics 未知

但作为 VLA post-training 的新范式，TACO 把 tactile world model + advantage RL + knowledge insulation 三件事整合得很优雅，6 个真实 contact-rich task 上 +44% SR 是 solid 的实证。

参考链接汇总：
- TACO project page: https://taco-wm.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0.6: https://arxiv.org/abs/2511.14759
- Wan2.2: https://arxiv.org/abs/2503.20314
- DINOv2: https://arxiv.org/abs/2304.07193
- DROID: https://arxiv.org/abs/2403.12945
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboDreamer: https://arxiv.org/abs/2404.12377
- DreamGen: https://arxiv.org/abs/2505.12705
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- Knowledge Insulating VLA: https://arxiv.org/abs/2507.05060
- DreamerV3: https://arxiv.org/abs/2301.04104
- 3D-ViTac: https://arxiv.org/abs/2410.24091

---

# TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training 深度解析

## 1. 核心问题与 motivation 的来源

VLA (Vision-Language-Action) 模型如 π0、π0.5、OpenVLA、RDT-1B 在 general manipulation 上取得 impressive 进展, 但在 contact-rich manipulation (Insert Flower、Twist Bottle Cap、Wipe Whiteboard、Move Hanoi Rings 这类需要稳定物理接触的任务) 上仍然 fragile. Karpathy 你一定熟悉 VLA 在 visual grounding 上的强势表现, 但 contact-rich 任务的失败模式有个非常微妙的特点: **failure 是 localized 而非 task-level semantic**.

具体场景:
- Wipe Whiteboard: eraser 盖住 mark 但 force 不够 → mark 没擦掉
- Twist Bottle Cap: gripper 对齐 cap 但没产生有效 twisting torque → cap 拧不开
- Insert Flower: flower 靠近 vase 但 stem 和 narrow opening 没对齐 → 插不进去

在这些情况中, policy 知道 "要做什么" (semantic level 正确), 但在 contact transition 附近一旦 contact shift (slippage、insufficient pressure、abnormal torque), 视觉上观察几乎不变, force-torque 信号却剧烈变化. 这就导致了一个尴尬: **vision-only 的 detection 看不出失败, vision-only 的 correction 又想象不出正确的 contact dynamics**.

参考资料:
- Project page: https://taco-wm.github.io/
- π0.5: https://arxiv.org/abs/2504.16054
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246

## 2. 为什么之前的方法不够好

### 2.1 Human intervention 难以 scale
DAgger、IntervenGen、MILE、Hi-WM 这些方法依赖 human 在 failure state 处介入, 提供 recovery demonstration. 这种 monitoring + manual recovery 模式难以 scale 到大量 task.

参考:
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- IntervenGen: https://arxiv.org/abs/2404.13872

### 2.2 Vision-only world model 在 contact-rich 场景下不可靠
RoboDreamer、ManipDreamer、DreamGen、VLAW 这类工作用 world model 想象 future rollout 来 post-train policy. 但 vision-only generation 容易产生 "visually plausible yet contact-inconsistent" 的轨迹. 例如视频看起来擦掉了 mark, 但 force sequence 完全不符合物理接触. 这就是 TACO 引入 tactile world model 的直接 motivation.

参考:
- RoboDreamer: https://arxiv.org/abs/2404.12377
- DreamGen: https://arxiv.org/abs/2505.12705

### 2.3 Naively 加 tactile input 会 erode VLM priors
直接把 force 喂到 VLA 全参数 fine-tune, tactile-action gradient 会回流到 PaliGemma vision-language backbone, 损害 pre-contact perception 和 spatial grounding. AT-VLA、ForceVLA 等工作都遇到类似问题.

参考:
- ForceVLA: https://arxiv.org/abs/2507.06409
- Tactile-VLA: https://arxiv.org/abs/2507.09160

## 3. TACO 框架总览: Recognize–Imagine–Label 闭环

TACO 的设计哲学是: **不收集更多 human demo, 而是让 world model 自己当 self-corrector, 把 real rollout 中的 failure 转换为 visuo-tactile corrective supervision**.

迭代循环:
1. **Recognize**: 部署当前 policy $\pi_\theta^{(k)}$ 采 real rollout $\mathcal{D}_{roll}^{(k)}$, 用 unified progress-action model 预测 dense progress $p_t$, 识别 failure-adjacent state (progress stall/decrease 处) 作为 correction anchor
2. **Imagine**: 从每个 anchor state 出发, visuo-tactile generation model 联合 denoise 未来 video + force 序列, 生成 49 步 local correction segment
3. **Label**: 同一个 progress-action model 对 imagined segment 预测 corrective action $\hat{a}_t$ 和 progress $\hat{p}_t$, 同时赋 binary advantage label $y_t \in \{0, 1\}$
4. **Post-train**: 用 knowledge-insulated tactile adaptation + advantage-conditioned training 把 imagined correction 喂回 policy, 完成一次迭代

这个闭环是 **real → imagine → real** 的: real rollout 提供 failure signal, imagine 生成 correction, post-train 后回到 real world 再 rollout, 逐步消除 contact-sensitive failure.

## 4. Tactile-Aware World Model 架构细节

### 4.1 Visuo-Tactile Generation Model: joint video-force denoising

Backbone 选的是 **Wan2.2-TI2V-5B** (Tencent 出的开源视频生成大模型, 5B 参数). 这是个有意思的选择, Wan2.2 是 text-to-video 模型, TACO 在此基础上做 video-force joint diffusion. 先在 DROID (201,119 trajectories)、AgiBot (3,017)、RoboMIND (1,721,985) 这些 broad robot dataset 上预训练, 再用 sliding window 适配到 contact-rich demo.

参考:
- Wan: https://arxiv.org/abs/2503.20314
- DROID: https://arxiv.org/abs/2403.12945
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877

#### 4.1.1 Tokenizer 与 token 拼接

- Video latent tokens: $X^v \in \mathbb{R}^{B \times N_v \times d}$, 其中 $N_v = 12$, $d = 3072$ (DiT hidden dim)
- Force sequence: $F \in \mathbb{R}^{B \times T \times 12}$, 12 个维度对应 left/right 各 6-DoF force-torque 信号 (左右两个 Xense tactile sensor, 每个 6-DoF)
- Force tokenizer $T_\eta$: $X^f = T_\eta(F) \in \mathbb{R}^{B \times T \times d}$, $T = 49 = 4 N_v + 1$ (force 用更高 temporal resolution)

注意这里 video 是 12 个 latent token (压缩后), force 是 49 个 raw timestep, **force 在时间维度上比 video dense 4 倍**. 这是因为 contact force 变化比 visual frame 快得多, 高频信号需要更密的采样.

拼接: $X = [X^v; X^f] \in \mathbb{R}^{B \times (N_v + T) \times d}$, 沿 token 维拼接. 这个设计的关键是 force token 与 video token 在 DiT self-attention 中 **bidirectional 交互**, force 不是单纯的 condition, 而是参与 attention 计算. 这与 vision-only world model 用 cross-attention 注入 condition 的方式本质不同.

#### 4.1.2 Joint flow-matching loss

公式:
$$
\mathcal{L}_{\text{joint}} = \|u_\psi^v - (\xi_1^v - \xi_0^v)\|_2^2 + \lambda_f \|u_\psi^f - (\xi_1^f - \xi_0^f)\|_2^2
$$

变量含义:
- $u_\psi^v \in \mathbb{R}^{B \times N_v \times d}$: video flow velocity field, 由 DiT 预测
- $u_\psi^f \in \mathbb{R}^{B \times T \times 12}$: force flow velocity field, 由 tactile head $H_\eta$ 投影后得到
- $\xi_1^v$: clean video latent (来自真实 demo 的 VAE 编码)
- $\xi_0^v \sim \mathcal{N}(0, I)$: Gaussian noise
- $\xi_1^f \in \mathbb{R}^{B \times T \times 12}$: clean force segment
- $\xi_0^f \sim \mathcal{N}(0, I)$: force 的 Gaussian noise
- $\lambda_f$: force denoising term 的权重系数, 控制两个 modal 之间的平衡

注意 $(\xi_1 - \xi_0)$ 是 flow matching 中的 target velocity (从 noise 到 clean data 的常数速度), 这是 rectified flow / flow matching 的标准形式. **video 和 force 共享同一个 sampled denoising timestep**, 这保证两个 modal 在同一 noise level 同步去噪, 不会出现 video 已经接近 clean 而 force 还在 noise 的 mismatch.

#### 4.1.3 Temporal RoPE alignment: 关键设计

这是 TACO 最聪明的细节之一. Wan2.2 原生是 3D video latent grid, RoPE 作用在 $(t, h, w)$ 三个轴上. Force token 只有 temporal 信息, 没有 spatial 维度. 如果直接拼进去, RoPE 的 spatial 部分会乱套.

TACO 的做法: 把 force token 的 temporal 位置映射到 video latent 的 temporal axis:

$$
\rho(i) = \text{round}\left(\frac{i}{T-1}(f-1)\right), \quad i = 0, ..., T-1
$$

变量:
- $i$: force token 的 index, $i \in [0, T-1] = [0, 48]$
- $T$: force token 总长 = 49
- $f$: video latent 的 temporal length
- $\rho(i)$: force token $i$ 对应到 video latent 的时间位置

这样 49 个 force token 被均匀映射到 video latent 的 $f$ 个时间槽位上. 每个 force token 用 $\rho(i)$ 处的 temporal RoPE, spatial RoPE 设为 $1 + 0j$ (单位复数, 即不引入任何 spatial 偏置).

**First-frame force anchoring**: 保留 $F_0 \in \mathbb{R}^{12}$ 作为 clean anchor (不注入 noise), 这是给模型一个明确的 contact-state 起点, 减少 contact-state ambiguity. 这个 trick 让模型不必从 noise 中"猜"起始 force 状态, 而是从真实 force 出发预测演化.

### 4.2 Unified Progress-Action Model (UPA): 双头监督

UPA 是 TACO 的 "大脑", 同时负责:
1. 预测 corrective action (用于 label imagined segment)
2. 预测 dense task progress (用于 recognize failure-adjacent state + 给 advantage label)

#### 4.2.1 架构

输入:
- RGB frame $I_t$
- Force-tactile signal $F_t \in \mathbb{R}^{12}$

输出:
- Corrective action $\hat{a}_t \in \mathbb{R}^{7}$ (7-DoF end-effector)
- Task progress $\hat{p}_t \in [0, 1]$

**Visual pathway**: DINOv2-with-Registers backbone, 输出 $37 \times 37$ patch features, dim=768. 之后接 **direction-aware decoder**: 4 个 dilated convolution branch + angle-sensitive pooling, 产出 1024 维 visual embedding $z_t^v$.

**Tactile pathway**: 2-layer MLP, hidden=128, output=256, 产出 tactile embedding $z_t^f$.

Fusion: $[z_t^v; z_t^f] \in \mathbb{R}^{1280}$ 拼接.

两个 head:
- Action head: $h_a$ 含 512 维 hidden, 输出 $\hat{a}_t = h_a([z_t^v; z_t^f])$
- Progress head: $h_p$ 含 256 维 hidden + sigmoid, 输出 $\hat{p}_t = \sigma(h_p([z_t^v; z_t^f]))$

参考:
- DINOv2: https://arxiv.org/abs/2304.07193

#### 4.2.2 Joint action-progress loss

$$
\mathcal{L}_{\text{UPA}} = \text{SmoothL1}(\hat{a}_t, a_t) + m_t \|\hat{p}_t - p_t\|_2^2
$$

变量:
- $\hat{a}_t$: 预测 action
- $a_t$: ground truth action
- $\hat{p}_t$: 预测 progress
- $p_t$: ground truth progress (从手动标注的 task-stage label 归一化得到)
- $m_t \in \{0, 1\}$: valid progress label 的 mask (有的 frame 没有标注就置 0)

为什么把 action 和 progress 联合训? **contact cue 同时引导 corrective action 和 progress 估计**, 两个 task 互相正则化: action prediction 让 visual feature 学习到 actionable 表示, progress prediction 让 feature 学习到 task-stage 结构. 这种 multi-task 设计让 single model 在 Recognize 和 Label 阶段共享同一份 representation, 避免训两个 model 的开销和不一致.

## 5. TACO 迭代 correction 框架

### 5.1 Recognize: 找到 failure-adjacent anchor

部署当前 policy $\pi_\theta^{(k)}$ 采 rollout $\tau \in \mathcal{D}_{roll}^{(k)}$. 对每个 timestep, UPA 预测 progress $p_t$. 当 progress 停滞或下降时, 该状态是 failure-adjacent:

$$
\mathcal{S}_{\text{anchor}}^{(k)} = \{(\tau, t) \mid \tau \in \mathcal{D}_{roll}^{(k)}, \; p_{t+\Delta} - p_t < \epsilon\}
$$

变量:
- $\Delta$: 短窗口 (lookahead 步数)
- $\epsilon$: progress threshold (小于这个阈值就视为 stalled)
- $\mathcal{S}_{\text{anchor}}^{(k)}$: 第 $k$ 轮迭代的所有 correction anchor

**实践细节**: 每条 failed trajectory 最多取 10 个 anchor, 让 anchor 集中在 contact-sensitive stage 而非均匀分布全轨迹.

**Advantage label 分配**:
- 失败 rollout 中第一个 anchor 之前的 timestep: $y_t = 1$ (on-track, positive)
- 第一个 anchor 及之后的 timestep: $y_t = 0$ (failed segment, negative)
- Expert demo 全程 $y_t = 1$
- Imagined correction 全程 $y_t = 1$ (因为是 recovery supervision)

### 5.2 Imagine: 49 步 visuo-tactile correction

从 anchor state $(I_t, F_t)$ 出发, 用 language instruction $l$, 让 generation model 采样:
$$
(\hat{I}_{t:t+T}, \hat{F}_{t:t+T}) \sim G_\psi(\cdot \mid I_t, F_t, l)
$$

$T = 49$ 步, 产出 local plausible correction segment. 这 49 步同时包含 visual evolution 和 contact-force dynamics.

### 5.3 Label: 用 UPA 给 imagined segment 标 action

$$
(\hat{a}_{t:t+T}, \hat{p}_{t:t+T}) = U_\phi(\hat{I}_{t:t+T}, \hat{F}_{t:t+T})
$$

注意这里 UPA 输入的是 **imagined video + imagined force**, 不是真实观测. 这相当于 inverse dynamics + progress estimation 一起做. 输出的 $\hat{a}_{t:t+T}$ 是 corrective action chunk, $\hat{p}_{t:t+T}$ 用来 verify imagined segment 是否真的向 task completion 推进.

## 6. Knowledge-Insulated Tactile Adaptation

### 6.1 核心矛盾

想象出来的 correction 数据是 tactile-heavy 的 (force 信号主导). 如果直接 fine-tune 整个 VLA, tactile-action gradient 会回流到 PaliGemma (2B) vision-language backbone, 损害 pre-contact perception. 但如果不 fine-tune VLM, 又学不到 visual context. 

TACO 的方案: **stop-gradient 隔离 VLM backbone, 把 tactile learning 路由到 action expert**.

### 6.2 具体实现

Policy base 是 **π0.5**:
- PaliGemma (2B) 作为 VLM backbone
- 300M action expert (flow matching, action chunk horizon=30, zero-pad 到 32 维)

训练流程:
1. Image + language + state 编码为 VLM prefix token $z_t$
2. **关键**: $z_t = \text{sg}[\text{VLM}(\text{image, language, state})]$, stop-gradient! VLM backbone 完全冻结
3. Force history (长度 8, 12-D, 通过 force encoder 编为 256 维) + advantage (通过 advantage encoder 编为 256 维) 注入 action expert, 通过 adaRMSNorm path
4. 只训: tactile encoder + adaptation layer + action expert

### 6.3 Advantage-Conditioned Post-Training (flow matching + CFG)

Action expert 训练目标:
$$
\mathcal{L}_\pi = \mathbb{E}\left[\|u_\theta(x_\sigma, \sigma \mid z_t, \tilde{c}_{\text{adaRMS}}) - (\epsilon - a_t)\|_2^2\right]
$$

其中 conditioning:
$$
c_{\text{adaRMS}} = c_t + \lambda_f c_f + \lambda_a c_a
$$

变量:
- $x_\sigma = \sigma \epsilon + (1 - \sigma) a_t$: noisy action chunk (flow matching interpolation)
- $\sigma \in [0, 1]$: noise level
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $a_t$: clean action (来自 demo 或 imagined correction)
- $u_\theta$: action expert 预测的 flow velocity
- $z_t$: VLM prefix representation (stop-gradient)
- $c_t$: flow timestep condition
- $c_f$: force condition (8 步 history, 256 维)
- $c_a$: advantage condition (256 维, 0 或 1)
- $\lambda_f, \lambda_a$: 平衡权重

**Classifier-Free Guidance (CFG)**: 训练时 $\tilde{c}_{\text{adaRMS}}$ 以概率 0.1 被替换成 null condition (让模型同时学 conditional 和 unconditional). 推理时 condition on positive advantage $y=1$, 引导 action expert 生成 high-progress tactile recovery behavior.

这是 offline RL 的视角: advantage label $y \in \{0, 1\}$ 把 trajectory 分为 success/correction (positive) 和 failure (negative), 训练时正样本推动 policy 向 high-progress 区域, 负样本让 policy 远离 stalled state. 推理时用 positive CFG 把 policy "推"向 recovery behavior.

参考:
- Knowledge Insulating VLA (Driess et al.): https://arxiv.org/abs/2507.05060 (推测, paper [30] 引用)
- π0.6 (advantage training): https://arxiv.org/abs/2511.14759

## 7. 实验结果深度分析

### 7.1 Real-world 6 个 contact-rich task

Task list:
1. **Insert Flower**: 拿起花插入花瓶
2. **Wipe Whiteboard**: 拿 eraser 擦掉 star
3. **Twist Bottle Cap**: 抓 cap + twist + lift
4. **Play Xylophone**: 用 mallet 依次敲 1, 3, 5, 8 号键
5. **Toast Bread**: 拿两片面包依次放入 toaster
6. **Move Hanoi Rings**: 把 middle peg 顶部环移到 left, 下一环移到 right

Hardware: Franka Research 3 (FR3) 单臂 + Xense tactile sensor (gripper 指尖) + Intel RealSense D455 前视相机. 每个 task 50 个 SpaceMouse teleop demo.

### 7.2 主结果表 (Table 1)

| 方法 | Insert SR | Wipe SR | Twist SR | Xylo SR | Toast SR | Hanoi SR | Ave SR | Ave CS |
|------|----------|---------|----------|---------|----------|----------|--------|--------|
| Base Policy | 0.50 | 0.51 | 0.45 | 0.46 | 0.30 | 0.08 | 0.38 | 185.5 |
| Filtered BC (Iter1) | 0.55 | 0.54 | 0.50 | 0.49 | 0.32 | 0.07 | 0.41 | 148.8 |
| TACO w/o KI (Iter1) | 0.55 | 0.33 | 0.58 | 0.42 | 0.49 | 0.32 | 0.49 | 154.8 |
| TACO (Iter1) | 0.70 | 0.55 | 0.85 | 0.63 | 0.48 | 0.51 | 0.66 | 141.8 |
| Filtered BC (Iter2) | 0.52 | 0.57 | 0.48 | 0.51 | 0.36 | 0.11 | 0.43 | 155.5 |
| TACO w/o KI (Iter2) | 0.62 | - | 0.65 | - | 0.78 | 0.50 | 0.50 | 146.5 |
| TACO (Iter2) | 0.93 | 0.65 | 0.98 | 0.52 | 0.81 | 0.51 | **0.82** | **127.7** |

**关键观察**:
- TACO 相对 Base Policy 改善 **+44% absolute SR**
- TACO 相对 Filtered BC 改善 **+39%** (Filtered BC 只能 reinforced narrow demo manifold)
- TACO 相对 w/o KI 改善 **+32%** (KI 保护 VLM prior 至关重要)
- TACO 在 Twist Bottle Cap 上达到 0.98 SR, 在 Insert Flower 上达到 0.93 SR
- CS (completion steps) 在成功 episode 上平均更短, 表明更 smooth execution, fewer pause/redundant motion

Filtered BC 失败原因: 只在 successful rollout 上 fine-tune, 但 successful rollout 里 **没有 recovery behavior** (成功的 trajectory 不包含 stall-and-recover 模式). 所以 Filtered BC 只能 reinforced 已有的 narrow action manifold, 在 contact 失败时无能为力. 这印证了 DAgger 经典论断: 仅在 policy 自己的成功分布上训练必然 narrow.

TACO w/o KI 失败原因: full VLA end-to-end fine-tune, tactile gradient 侵蚀 VLM prior, 导致 **pre-contact perception 退化**. 在 Wipe Whiteboard 上从 0.55 降到 0.33 就是这个原因: 失去了 spatial grounding, 抓 eraser 和定位 star 都变差.

### 7.3 Ablation: tactile 的双重作用 (Figure 5)

| 设置 | Visuo-Tactile Gen Input | Output | UPA Input | Output | Force Val Loss F↓ | Action Val Loss A↓ | VOC↑ | FL↑ | Real SR↑ |
|------|----|----|----|----|----|----|----|----|----|
| w/o tactile generation | V | V | V | A+R+F | 0.004 | 0.025 | 0.78 | 0.87 | 0.28 |
| w/o tactile labeling | V+F | V+F | V | A+R | 0.002 | 0.038 | 0.88 | 0.90 | 0.65 |
| TACO (full) | V+F | V+F | V+F | A+R | 0.002 | 0.019 | 0.94 | 0.95 | 0.82 |

**V = Video, F = Force, A = Action, R = progress Reward (这里指 progress)**

Metric 含义:
- **VOC** (Video frame-wise progress rank correlation): shuffle 帧后, 预测 progress 与真实时序的 rank correlation. 越高说明 UPA 越懂 task 进展
- **FL** (Failure Localization accuracy): 手动标注 failure-adjacent frame, 检查 UPA 通过 progress stall 检测这些 frame 的 accuracy

**两个关键 ablation 结论**:
1. **w/o tactile generation (vision-only world model)**: SR 掉到 28%. 即使 UPA 在 label 时用 force, 但 imagined video 本身没有 force 约束, contact dynamics 已经是 wrong 的, 给 label 提供的 force 也是错的. **证明**: 视觉 alone 想不出 contact transition.
2. **w/o tactile labeling (generate force 但 label 时不用)**: SR 掉到 65%. Force 已经生成, 但 UPA label action 时只看 video, force 信号被浪费. **证明**: tactile 必须直接参与 corrective action prediction, 仅作 auxiliary observation 不够.

完整 TACO: 82%. 两者协同提升 17-54 个百分点.

### 7.4 Scaling of Imagined Correction Data (Figure 5 右)

Insert Flower 上, real-to-imagined ratio:
- 1:2 → SR 70%
- 1:4 → SR 93%
- 1:8 → SR 97%

Toast Bread 上 (Appendix C.2):
- 1:2 → 55%
- 1:5 → 81%
- 1:10 → 90%

**结论**: imagined correction 越多, SR 越高, 而且没有饱和. 1:8 > 1:4 说明 broader coverage of failure-adjacent contact state 比 narrow high-quality 更有效. 这非常符合 world model 作 self-corrector 的核心论点: 想象的边际成本低, 可以 scale, 这正是替代 human intervention 的关键.

实际 ratio: 每条 failed trajectory 取 10 个 anchor, 每个产 1 个 49 步 imagined segment, 整体 real-to-imagined ratio 约在 1:4 到 1:5 之间.

### 7.5 Advantage-Conditioned Training 的 ablation (Figure 10b)

| 方法 | Insert Flower SR | Wipe Whiteboard SR |
|------|------------------|---------------------|
| TACO | 93% | 65% |
| TACO w/o Advantage | 83% | 56% |

去掉 advantage-conditioned, 即把 failed rollout 全部丢弃, 只在 success + imagined correction 上做 supervised fine-tune, 不区分 positive/negative, SR 掉 10-9 个点.

**Intuition**: 失败 rollout 提供 "negative signal" — 哪些 contact state 应该 avoid. 没有 advantage label, policy 把所有 trajectory 视为同质, 学不到 "high-progress correction" vs "stalled failure" 的区分. CFG 在推理时也无法用 positive advantage 推向 recovery.

### 7.6 Failure-Adjacent Anchor Selection 的 ablation (Figure 10c)

| 方法 | Insert Flower SR | Wipe Whiteboard SR |
|------|------------------|---------------------|
| TACO | 93% | 65% |
| TACO Uniform Anchor | 78% | 25% |

Uniform anchor (从 failed rollout 随机取同样多 anchor) 在 Insert Flower 掉 15 个点, 在 Wipe Whiteboard 掉 40 个点. **证明**: recognize step 的 progress-guided anchor selection 至关重要. 随机 anchor 会浪费 imagination budget 在远离 failure 的 state 或已经 unrecoverable 的 state 上, 产出的 correction 与 failure 不匹配.

### 7.7 Action Distribution Analysis (Figure 6)

Insert Flower 上, 固定 flower 和 vase 的初始位置, 评估 40 个 successful rollout, 投影 end-effector pose 到 world-frame XY 平面.

| 配置 | 分布形态 |
|------|----------|
| (a) Expert demo | 主流形 + 少量扰动 |
| (b) Base policy | narrow 集中在 demo manifold |
| (c) Filtered BC | 与 (b) 几乎一致, 不能扩展 |
| (d) TACO Iter1 | broader 分布 |
| (e) TACO Iter2 | 进一步拓宽 |

**Intuition**: Base policy 和 Filtered BC 都困在 narrow demonstration manifold, 对 execution error 敏感, 一旦偏离就无 recovery. TACO 通过 imagined correction 暴露 policy 于 diverse successful recovery behavior, 显著 broaden action distribution, 即使遇到未见场景也能 recover.

这呼应了 RL 中 "exploration broaden policy distribution" 的思想, 但 TACO 用 world model imagination 替代真实 exploration, 极低成本实现 distribution broadening.

### 7.8 OOD Generalization (Figure 7, 12)

Wipe Whiteboard 三个 OOD 设置:
| Setting | Base Policy | TACO (1 iter) |
|---------|-------------|---------------|
| Unseen Background | 20.0% | 80.5% |
| Unseen Object (sponge 替代 eraser) | 35.0% | 85.5% |
| Unseen Position (远位置) | 13.5% | 55.0% |

Insert Flower 同样三个 OOD 设置, base 从 50% 降到 20-35-13.5%, TACO 一轮迭代后恢复到 80.5-85.5-55%.

**关键**: 这些 OOD setting 在 vision、tactile、action 三个维度都超出了 world model 的 training distribution, 但 tactile-aware world model 仍能 generate effective correction. 这说明 **tactile dynamics 的 physical regularity 比 visual appearance 更 transferable**, force-torque 信号对 object geometry/texture shift 的 robustness 比 RGB 高.

### 7.9 Failure Case Analysis (Figure 11)

三个 task 上比较 Filtered BC、TACO w/o KI、TACO:

- **Filtered BC**: 接触前正常, contact 时 stall. Wipe 没 force, Hanoi 没 align, Twist 没 torque. 始终困在同一个 narrow behavior.
- **TACO w/o KI**: 有 imagined correction 学到 recovery, 但 VLM prior 被侵蚀, pre-contact perception 差, 接触点偏离 (off-target wiping, misaligned insertion, unstable engagement).
- **TACO**: 两者兼得 — reliable approach (VLM prior 完整) + effective contact recovery (tactile correction). 三个 task 全部完成.

**结论**: imagined visuo-tactile correction 和 knowledge-insulated adaptation 是互补的, 缺一不可. 单独一个只解决一半问题.

## 8. Algorithm 1 完整 pseudocode 逐行解读

```python
# Require: Base VLA policy π; visuo-tactile gen model G_ψ; UPA U_φ
# Expert demo D_demo; iter K_iter; horizon T; window Δ; threshold ε
# Weights λ_f, λ_a

# Step 1: Warm-start
Warm-start π on D_demo

# Step 2: 给 expert demo 全程赋 y=1 (positive)
for t in D_demo: y_t = 1

for k in 1..K_iter:
    # (1) Collect real rollouts
    D_roll^(k) = deploy π_θ in real world
    S_anchor = ∅
    
    # (2) Recognize failure-adjacent states
    for τ in D_roll^(k):
        p̂_t = U_φ(τ)  # dense progress
        if success(τ):
            y_t = 1 for all t
        else:
            A_τ = {(τ,t) | p̂_{t+Δ} - p̂_t < ε, t+Δ ≤ |τ|}
            if A_τ non-empty:
                t* = min{t | (τ,t) ∈ A_τ}  # failure onset
                y_t = 1 for t < t*
                y_t = 0 for t >= t*
                S_anchor ∪= A_τ
            else:
                y_t = 0 for all t  # unrecoverable
    
    # (3) Imagine + Label
    D_corr = ∅
    for (τ, t) in S_anchor:
        (Î, F̂) ~ G_ψ(I_t, F_t, l)  # joint video-force denoise
        (â, p̂) = U_φ(Î, F̂)  # label action + progress
        ŷ = 1 for all in imagined segment
        D_corr ∪= {(Î, F̂, â, ŷ)}
    
    # (4) Knowledge-insulated tactile adaptation
    D_train = D_demo ∪ D_roll ∪ D_corr
    for minibatch:
        z_t = sg[VLM(image, lang, state)]  # STOP-GRADIENT
        c_f, c_a = encode(force_hist), encode(advantage)
        (c̃_f, c̃_a) = ConditionDropout(c_f, c_a)  # CFG
        c̃_adaRMS = c_t + λ_f c̃_f + λ_a c̃_a
        update(tactile_encoder, adaptation, action_expert) to minimize L_π
```

整个流程是 real → imagine → real 闭环, 第 $k$ 轮的 real failure 生成 imagined correction, post-train 后第 $k+1$ 轮再 collect real failure, 渐进消除 contact-sensitive failure.

## 9. 几个值得深入的细节

### 9.1 为什么 force 用 12 维而不是 raw tactile image

Xense tactile sensor 输出 6-DoF force-torque, 两个 sensor (left + right finger) 拼成 12 维. 相比 GelSight 之类的高分辨率 tactile image, 12-D F/T 更紧凑, 更容易 tokenize, 也更接近物理力学的直接表示. 这让 force 的 flow matching 在低维度上更稳定, 也让 adaRMSNorm conditioning 更轻量.

参考:
- 3D-ViTac (tactile image based): https://arxiv.org/abs/2410.24091
- Visuo-Tactile World Models: https://arxiv.org/abs/2602.06001

### 9.2 Temporal RoPE alignment 的 intuition

Wan2.2 的 RoPE 原生是 3D grid $(t, h, w)$. Force token 没有 $h, w$, 只有 $t$. 如果直接拼, RoPE 计算时 spatial 部分会产生无意义的相位, 干扰 attention. TACO 把 force token 沿 video 的 temporal axis 对齐: $\rho(i) = \text{round}(i/(T-1) \cdot (f-1))$, 让 force token 的 temporal position 落在 video latent 的对应时间槽. Spatial RoPE 设为 $1+0j$ (复数单位元), 即不引入 spatial 偏置.

这相当于告诉 attention: "force token 在空间上是统一的, 只在时间上和 video 对齐". 一个 force token "属于" video latent 的某个时间槽, 与该槽内所有 spatial token 都参与 attention.

### 9.3 First-frame force anchoring 的 intuition

Flow matching 中, noise level $\sigma \in [0, 1]$, $\sigma=0$ 是 clean, $\sigma=1$ 是 noise. 如果第一帧 force 也被注入 noise, 模型要从 noise 推断起始 contact state, 这是 ambiguous 的 (slip vs insufficient force 视觉上一样, 起始 force 不同).

保留 $F_0$ clean 作为 anchor, 给模型一个明确的 "现在 contact 状态是什么" 的 reference, 从此出发预测未来 force 演化. 这相当于 conditional generation 中的 "first frame conditioning", 是 video diffusion 常用 trick 在 force domain 的迁移.

### 9.4 adaRMSNorm 与 stop-gradient 的协同

adaRMSNorm 是 DiT 的标准 conditioning 机制: 用 condition embedding 调制每个 transformer block 的 scale 和 shift. TACO 把 force + advantage embedding 通过 adaRMSNorm 注入 action expert, action expert 与 VLM backbone 通过 cross-attention 交互.

关键: $z_t = \text{sg}[\text{VLM}(...)]$, VLM 输出被 stop-gradient. 这意味着 action expert 可以 "看到" VLM representation, 但 gradient 不会回流到 VLM. VLM 的 visual-language prior 完全保留, action expert 学到的是 "在 frozen VLM feature 之上 + force/advantage condition 下如何预测 action".

这是 "modular adaptation" 思路: 不同 modality 用不同 module 学习, base model 保持 frozen. 与 LoRA、adapter 等思路类似但更激进 — 完全切断 gradient, 只在 expert 内训.

### 9.5 Advantage label 的设计: binary vs continuous

TACO 用 binary advantage $y \in \{0, 1\}$, 简单粗暴. 复杂的 advantage (如 RL 中的 $A(s,a) = Q(s,a) - V(s)$) 需要 value function 估计, 在 manipulation 中很难学好. Binary advantage 等价于 "成功/失败" 的简单 reward, 配合 CFG 可以把 policy 推向 positive 区域.

这与 π0.6 (paper [32], Physical Intelligence 2025) 的 "learns from experience" 思路一致, 但 TACO 把 advantage source 从真实环境 feedback 换成了 world model imagination + progress signal, 更低成本.

参考:
- π0.6: https://arxiv.org/abs/2511.14759

## 10. 局限与未来方向

### 10.1 已承认局限

TACO 的 imagined correction 是 **offline generation**, 不是 online deployment. 这意味着 correction 在训练时生成, 部署时还是用 post-trained policy 单步 inference, 没有 runtime correction.

### 10.2 可探索方向

1. **Online correction**: 部署时实时 run world model, 在 detected failure 时即时 generate correction, 类似 MPC + world model. 但 Wan2.2-TI2V-5B 推理慢, 需要蒸馏或轻量化.

2. **World model 与 policy tighter coupling**: 当前 world model 与 policy 是分开训的. 让 policy 直接在 world model latent space 中规划 (latents as actions), 或 world model 直接作为 policy 的 imagination engine (类似 DreamerV3), 可以 tighter 绑定.

参考:
- DreamerV3: https://arxiv.org/abs/2301.04104
- WorldVLA-Learning: https://arxiv.org/abs/2602.06508

3. **Tactile representation learning**: 当前 force 用 12-D F/T, 信息密度有限. 引入 tactile image (GelSight) + 学习 tactile representation, 可以捕捉更精细的 contact geometry (slip direction, contact patch, surface texture).

4. **Multi-finger / bimanual**: TACO 单臂 + parallel gripper. 扩展到 multi-finger hand 或 bimanual, contact state dimensionality 暴增, tactile world model 的复杂度也增加.

5. **Long-horizon task**: 49 步 correction horizon 是否够长? 对于 multi-stage task (如 Hanoi Rings 4 个 stage), 可能需要 hierarchical imagination.

## 11. TACO 与相关工作的对比

### 11.1 vs RoboDreamer / ManipDreamer (vision-only world model + policy)
- RoboDreamer: 视频生成 + inverse dynamics, vision-only. contact-rich 上不可靠.
- TACO: 加 force joint denoise + UPA 联合 action/progress, 解决 contact dynamics.

### 11.2 vs DAgger / IntervenGen (human intervention)
- DAgger: 需要 human 在 failure state 介入, 难以 scale.
- TACO: world model 自动 generate correction, 无需 human, scale 到任意多 task.

### 11.3 vs ForceVLA / Tactile-VLA (naive tactile VLA)
- ForceVLA: 直接把 force 喂 VLA, full fine-tune, 损害 VLM prior.
- TACO: knowledge-insulated, force 只路由到 action expert, 保护 VLM prior.

### 11.4 vs π0.6 (real experience learning)
- π0.6: 在真实环境 rollout + binary success 上做 RL, 需要 real world feedback loop.
- TACO: 在 world model imagination 上做 advantage training, real world 只用于 failure detection, 无需 real-world success label.

### 11.5 vs DreamGen / VLAW (world model as simulator for RL)
- DreamGen: 用 video world model 作 simulator, 在其中 rollout RL.
- TACO: world model 不作 full simulator, 只 generate local correction segment, 然后 label + post-train. 更轻量, 也避免长 horizon imagination 的累积误差.

### 11.6 vs Hi-WM (Human-in-the-world-model)
- Hi-WM: human 提供 correction signal 给 world model.
- TACO: world model 自己 generate correction, 完全 autonomous.

## 12. 我的整体 intuition

TACO 的核心 insight 可以浓缩为一句话: **contact-rich manipulation 的失败是 localized tactile dynamics 失败, 不是 task semantic 失败, 所以 correction 应该 localized 在 contact transition 附近, supervision 应该是 visuo-tactile 而非 vision-only, post-training 应该保护 pre-contact perception 而非全参数 fine-tune**.

这套设计有几个 nice property:

1. **Failure 是 self-identified 的** — 通过 progress stall 检测, 不需要 human 标注 failure.
2. **Correction 是 self-generated 的** — world model 想象, 不需要 human demo recovery.
3. **Action 是 self-labeled 的** — UPA 在 imagined segment 上预测 action, 不需要 inverse dynamics model.
4. **VLM prior 是 self-protected 的** — stop-gradient 隔离, 不需要 LoRA 之类的折中.
5. **整个 loop 是 autonomous 的** — real rollout → recognize → imagine → label → post-train → real rollout, 闭环.

Karpathy 你在 Tesla 讲过 "data is the bottleneck", TACO 正是 attack 这个 bottleneck 的方法: 不再依赖 human teleop demo, 而是 build 一个 self-corrector 把 failure 转 correction. 这与 self-driving 的 "shadow mode + auto-labeling" 思路相通, 只是把"想象"任务交给 world model.

潜在风险:
- **World model hallucination**: imagined correction 可能 physically inconsistent (虽然 joint video-force denoise 缓解). 需要 verification mechanism.
- **Distribution shift**: OOD 实验显示 1 轮迭代后 SR 恢复, 但更 extreme OOD (unseen action space) 可能失败.
- **Iteration divergence**: 多轮 post-train 是否会 collapse 到某些 narrow recovery mode? 论文只跑 2 轮, 长期 dynamics 未知.

但作为 VLA post-training 的一个新范式, TACO 把 tactile world model + advantage RL + knowledge insulation 三件事整合得很优雅, 而且实验在 6 个真实 contact-rich task 上 +44% SR 是非常 solid 的实证.

参考链接汇总:
- TACO project page: https://taco-wm.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0.6: https://arxiv.org/abs/2511.14759
- Wan2.2: https://arxiv.org/abs/2503.20314
- DINOv2: https://arxiv.org/abs/2304.07193
- DROID: https://arxiv.org/abs/2403.12945
- OpenVLA: https://arxiv.org/abs/2406.09246
- RoboDreamer: https://arxiv.org/abs/2404.12377
- DreamGen: https://arxiv.org/abs/2505.12705
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- Knowledge Insulating VLA: https://arxiv.org/abs/2507.05060
- DreamerV3: https://arxiv.org/abs/2301.04104

希望这个深度解析能 build up 你的 intuition, Andrej. 如果你想进一步讨论某个细节 (比如 RoPE alignment 的实现, 或 advantage CFG 的具体推理), 可以再问.
