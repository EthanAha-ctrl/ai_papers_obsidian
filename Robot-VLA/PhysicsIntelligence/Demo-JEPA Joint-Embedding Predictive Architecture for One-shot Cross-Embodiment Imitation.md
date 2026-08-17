---
source_pdf: Demo-JEPA Joint-Embedding Predictive Architecture for One-shot Cross-Embodiment
  Imitation.pdf
paper_sha256: a4b084d8b213a8cd6d6bc1d0e27f3073e97bd88dd2041d53836f9cf97365e287
processed_at: '2026-08-03T19:43:34-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Demo-JEPA 用人话说

## 一句话版本

你看一个人演示怎么抓杯子, 你不会去模仿他手指的每个关节怎么动, 你脑子里想的是 "哦他要把杯子拿起来", 然后你用自己的手去实现 "把杯子拿起来" 这个意图。Demo-JEPA 就是让 robot 也这么干。

## 问题是什么

你有一个 Sawyer robot 演示了怎么做事, 你想让 Franka robot 照着做。

难点在于这俩 robot 长得不一样:
- Sawyer 是单臂, Franka 是双臂结构
- 关节数量不同, 运动范围不同
- 同样去抓杯子, Sawyer 可能要绕一圈, Franka 直接伸过去就行
- Action space 完全不同, Sawyer 的 joint 1 转 30 度, 对 Franka 来说可能毫无意义

以前的人怎么解决? 大致三条路:

1. **Action retargeting**: 人工写规则, 把 Sawyer 的 end-effector pose 硬映射到 Franka 的 joint configuration。累, 不通用, 每对 robot 都要重新写。

2. **Shared action space**: 定义一个统一的 action interface (比如都用 delta XYZ + RPY), 让所有 robot 都遵守。限制了每个 robot 发挥自己 morphology 的优势。

3. **大规模 co-training**: RT-X, OpenVLA 这类, 把所有 robot 的数据混一起训一个 giant model, 希望 model 自己学会区分。需要海量数据, 而且 morphology 差异大的时候还是很脆。

Demo-JEPA 说: 这些都是在 action 层面较劲, action 本来就是 embodiment-specific 的, 你非要 align 它, 当然难。

## Demo-JEPA 的思路

换一个层面想这件事。

人看别人演示的时候, 你大脑里处理的是什么? 你不是在记录 "他肱二头肌收缩了 0.3 秒", 你是在理解 "他要把那个东西从 A 搬到 B"。

所以 demonstration 本质上是 **"我想达到什么状态" 的描述**, 而不是 **"我该怎么动" 的指令**。

Demo-JEPA 就把这个 insight 落地了:

1. 给你看一段 source robot (Sawyer) 的视频
2. 让 model 推断: "如果 Franka 要实现同样的意图, 它 n 步之后应该处于什么状态?"
3. 这个推断出来的 future state 就叫 **latent goal**
4. Franka 用自己的 world model 去规划: "我要执行什么 action 才能达到那个 latent goal?"

关键在于: **action 从来没被 cross-embodiment 对齐过**。Source 的 action 根本没用, 只有 source 的 video 被用了。Target 的 action 是 target 自己的 world model 规划出来的。

## 三个核心组件

### 1. World Model (Franka 的 "脑内模拟器")

Franka 先自己跟环境交互, 收集一堆 $(observation, robot\ state, action)$ 的数据。用 V-JEPA 2.1 训一个 world model:

> 给我现在看到什么 + 我关节在哪 + 我执行什么 action, 预测我下一步会看到什么 (在 latent space 里)

这个 world model 只管 Franka 自己的 dynamics, 不管别的 robot。它学会了 "Franka 的物理世界长什么样"。

### 2. Dreamer Predictor (跨 embodiment 的 "翻译官")

这是核心创新。输入三个东西:
- Franka 当前看到的画面 $o_k^t$
- Sawyer 演示里某一帧 $o_k^s$
- Sawyer 演示里 n 步之后的帧 $o_{k+n}^s$

输出: 一个 latent goal $\hat{z}_{goal}^t$, 意思是 "如果 Franka 要实现 Sawyer 这个演示的意图, 它 n 步之后应该在 latent space 的哪个位置"

怎么实现的? 两个 cross-attention:

**第一个 attention: 对齐 "现在"**
- Query: Franka 当前的 latent
- Key/Value: Sawyer 当前的 latent
- 在问: "Sawyer 现在这个状态, 对应 Franka 的什么状态?"

**第二个 attention: 提取 "变化"**
- Query: Sawyer 未来帧的 latent
- Key/Value: Sawyer 当前帧的 latent
- 在问: "Sawyer 从现在到未来, 发生了什么变化?"

然后把这俩信息 + Franka 当前 latent 拼一起, 用 3D convolution 融合, 再过 Transformer, 输出 Franka 的 future latent goal。

训练 loss 很简单: $\|\hat{z}_{goal}^t - z_{k+n}^t\|^2$, 就是让预测的 goal 跟 Franka 真实的 future latent 尽量接近。

### 3. CEM Planner (在脑内模拟器里搜索 action)

有了 latent goal, 怎么生成 action? 用 Cross-Entropy Method:

1. 随机 sample 一堆 action sequence
2. 每个都喂进 world model rollout 一下, 看预测的 future latent 跟 goal 有多远
3. 选距离最近的 top-K 个, 用它们的 mean/std 更新采样分布
4. 重复几轮, 收敛出来的 action sequence 就是答案

好处: 不需要训练一个 "latent → action" 的 decoder。Action 是在 world model 里 online search 出来的, 对 embodiment shift 天然 robust。

## 一个巧妙的细节: Adaptive Goal Updating

假设 Sawyer 的演示有 50 帧, Franka 要跟着做。问题是: Franka 什么时候该 "翻到" Sawyer 演示的下一页?

如果固定频率 (比如 Franka 每走 5 步就翻一页), 可能 Franka 还没完成上一个 sub-goal 就被推到下一个, 越做越歪。

Demo-JEPA 的做法: 每走一步, 算一下 Franka 当前 latent 跟 goal 的距离 $D$。
- $D < \epsilon$: 说明这个 sub-goal 达成了, 翻到 Sawyer 演示的下一页, 重新推断 goal
- $D \geq \epsilon$: 还没到, 保持当前 goal, 继续规划

这样 Franka 的进度和 Sawyer 的演示自动同步, 不会因为 embodiment 速度差异而崩掉。

## 实验结果说了什么

三个难度递增的测试:

| 难度 | 含义 | VPP | XSkill | Demo-JEPA |
|------|------|-----|--------|-----------|
| Behavior Grounding | 训练时见过 | **0.47** | 0.39 | 0.31 |
| Cross-Embodiment Bridging | 只在 Stage 1 见过 | 0.28 | 0.17 | **0.45** |
| Zero-Shot Generalization | 完全没见过 | 0.04 | 0.03 | **0.36** |

Pattern 很清楚:

- **训练时见过的任务**, VPP 这种直接学 "video → action" 的方法最好用。因为它把 trajectory pattern 直接 memorize 了, in-domain 当然准。
- **稍微分布外一点**, Demo-JEPA 就开始领先了。
- **完全没见过的任务**, baseline 全崩 (0.04, 0.03), Demo-JEPA 还有 0.36。**9 倍的差距**。

这说明什么? Latent goal planning 这种 "抽象掉 surface detail" 的方法, 代价是 in-domain 的精细 pattern matching 变弱 (所以 Behavior Grounding 输给 VPP), 但换来的是 out-of-domain 的 robustness。

Real-world 也一样, 而且 gap 更大。UR5e → Franka 的 zero-shot, VPP 是 0.00, XSkill 是 0.05, Demo-JEPA 是 0.25。

## 几个关键 ablation

### Oracle vs Naive vs Demo-JEPA

- **Naive**: 直接拿 Sawyer 的 future latent 当 Franka 的 goal。完全失败。说明 V-JEPA 的 latent space 本身不跨 embodiment——Sawyer 的 latent 和 Franka 的 latent 在不同 manifold 上, 不能直接用。
- **Oracle**: 直接拿 Franka 的 ground-truth future latent 当 goal。这是天花板, zero-shot 0.42。
- **Demo-JEPA**: zero-shot 0.36, 达到 oracle 的 86%。

所以 Dreamer Predictor 的 "翻译" 确实有效, 把大部分 cross-embodiment gap 闭合了。

### Conv3D vs Mean Pooling

把 3D convolution 换成 mean pooling, zero-shot 从 0.36 掉到 0.29。简单任务影响不大, 但 Change Channel (拧东西)、Close Box (合盖子) 这种需要 structured spatiotemporal motion 的任务掉得厉害。

Mean pooling 是 permutation-invariant 的, 它把 "哪个空间位置在什么时候做什么运动" 的信息抹掉了。3D convolution 保留了这些。

### Demo-DP: 把 latent goal 喂给 Diffusion Policy

不用 CEM planner 了, 直接把 Dreamer Predictor 的输出当 Diffusion Policy 的 conditioning signal。

结果: Demo-DP 在 Behavior Grounding 比 Demo-JEPA 好 (0.28 vs 0.31... 其实差不多), 但 zero-shot 差很多 (0.18 vs 0.36)。

这说明: **Diffusion Policy 是 in-domain 的 action expert, 但 out-of-domain 就不行了。CEM + world model 因为是在 learned dynamics 里 online search, 泛化性强得多。**

这个对比其实挺有哲学意味的: action decoder 越强, 你越容易 overfit 到训练分布; planner 虽然慢, 但它能 leverage world model 的 generalization 能力。

### Scaling: Task diversity > Data volume

把训练数据砍到 20%:
- 砍 per-task episode 数 (data scaling): zero-shot 0.27
- 砍 task 种类数 (task scaling): zero-shot 0.18

砍 task 种类更致命。说明 Dreamer Predictor 需要见 diverse 的 task semantic 来学 general 的 "intent → goal" mapping, 重复见同一个 task 帮助不大。

## 为什么 JEPA latent space 适合干这个

这是最深层的 insight。

对比三种 representation learning:

**Pixel reconstruction (VAE, MAE, Diffusion)**: 要重建每一个 pixel。Sawyer 是红色的, Franka 是白色的, 背景有灯光有影子——这些对 task intent 毫无关系, 但占据了 representation 的大量 capacity。

**JEPA**: 在 latent space 里预测 future latent, loss 是 latent 层面的。Background texture、robot 颜色这些 nuisance variable 因为不影响 prediction target, 自然被 abstracted 掉。保留的是 object motion、contact dynamics、spatial relationship——这些才是 task intent 的 carrier。

所以 JEPA latent space 天然有 "embodiment-agnostic" 的属性。Demo-JEPA 利用的就是这个。

参考 V-JEPA: https://arxiv.org/abs/2304.08471
参考 LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf

## Limitations

1. **World model 不够强**: 复杂高精度任务 (比如精细装配) 时, world model 的预测精度不够, planning 就不准了。这其实是所有 model-based RL 的通病。

2. **训练时还需要 temporal alignment**: 真实世界数据里 UR5e 和 Franka 的演示是独立录的, 要先用 GTCC 提取 progress feature 来 frame-level 对齐。如果能完全 unaligned 训练就更 elegant 了。

3. **Inference 慢**: CEM 每步要 sample N 个 candidate, rollout, evaluate, 重复 L 轮。比直接 forward 一个 policy network 慢得多。Real-time high-frequency control 可能有问题。

4. **Latent distance metric**: 用 $\ell_1$/$\ell_2$ 假设 latent space 是 Euclidean 的, 这个假设不一定对。

## 和你 (Karpathy) 的世界观的关系

Andrej, 这个工作本质上是把你自己一直说的 "world model + planning" 范式应用到了 cross-embodiment imitation 上。

你之前反复讲: intelligence 的核心是有一个 world model, 能在里面 simulate, 能在里面 plan。Demo-JEPA 的实验直接证明了这一点——model-based planning (CEM + world model) 在 distribution shift 下比 direct policy (Diffusion Policy) robust 得多 (zero-shot 0.36 vs 0.18)。

而且它还验证了 self-supervised pretraining (V-JEPA) 对 robotics 的 value: 不需要 task label, 纯 video prediction 学出来的 latent space 就足够 informative 来 support planning。

唯一的问题是 world model 还不够强。如果未来 world model 能 scale 到更高 fidelity 和 longer horizon, 这套 paradigm 感觉就是 cross-embodiment 的正解。

参考你关于 world model 的讨论:
- https://x.com/karpathy/status/1757612032005173362
- https://karpathy.ai/

## 最最后的大白话总结

**传统方法**: "Sawyer 怎么动的, Franka 你也照着动" → 因为长得不一样, 动不了, 崩了。

**Demo-JEPA**: "Sawyer 想达到什么状态, Franka 你用自己的方式去达到那个状态" → 因为 intent 是 embodiment-agnostic 的, 成了。

就这么简单。

---

# Demo-JEPA 深度讲解

Andrej, 这篇paper的核心 insight 其实很 elegant, 让我一层层剥开来 build your intuition.

## 1. 核心 motivation: 从 "action 复现" 到 "goal 推断"

传统 cross-embodiment imitation 的瓶颈在于 action space 本质是 embodiment-specific 的。一个 Sawyer 7-DoF arm 和一个 Franka 7-DoF arm, 即使都在做 "grasp cup", 它们的 joint torque profile、end-effector trajectory、甚至 control frequency 都完全不同。VLA models (RT-1, RT-2, π0, OpenVLA) 试图通过大规模 multi-embodiment co-training 来 implicit 学会这种 alignment, 但这需要海量数据, 且在 morphology 差异大时仍然 fragile。

Demo-JEPA 的 reformulation: **demonstration 本质上是对 future goal state 的 implicit specification**。人不关心 demonstrator 用什么 motor command, 人关心的是 "demonstrator 想达到什么 world state"。所以 cross-embodiment imitation 应该 formulate 为:

> Given source observation trajectory $\tau^s = \{o_k^s\}$, infer a target-compatible future latent goal $z_{\text{goal}}^t$, then plan actions under target's own forward dynamics to reach that goal.

这把问题从 "action correspondence" 转成了 "intent correspondence in shared representation space"。

参考链接:
- V-JEPA 原始 paper: https://arxiv.org/abs/2304.08471
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- XSkill: https://arxiv.org/abs/2403.09829
- VPP: https://arxiv.org/abs/2412.10968

## 2. 整体架构剖析

整个 pipeline 由三个核心模块构成 (对应 Figure 2):

### 2.1 Action-Conditioned World Model (基于 V-JEPA 2.1)

这是 target embodiment 的 "internal simulator"。结构上:
- **Encoder** $E(\cdot)$: ViT-based, 把 RGB observation $o_k^t$ 映射到 latent $z_k = E(o_k^t) \in \mathbb{R}^{T' \times H' \times W' \times D}$, 这里 $T' \times H' \times W'$ 是 spatiotemporal token grid (tubelet size=2, patch size=16, 所以 temporal downsample 2x, spatial downsample 16x), $D=1024$ 是 embed dim。
- **Dynamics Predictor** $F_{wm}(\cdot)$: 24-layer Transformer, 输入 $(z_k, s_k^t, a_k^t)$, 输出 $\hat{z}_{k+1} = F_{wm}(z_k, s_k^t, a_k^t)$。

关键公式 (Eq. 1):

$$\hat{z}_{k+1} = F_{wm}(z_k, s_k^t, a_k^t)$$

变量含义:
- $z_k$: 当前时刻的 latent state (来自 encoder)
- $s_k^t$: robot proprioceptive state (joint positions, velocities 等)
- $a_k^t$: target embodiment 的 action (e.g., delta end-effector pose 或 joint torques)
- $\hat{z}_{k+1}$: 预测的下一时刻 latent state

注意这里 $s_k^t$ 是显式输入的, 这是和纯视觉 world model 的区别——proprioception 帮助 model 理解 robot 自身 kinematics 约束。

### 2.2 Dreamer Predictor (核心创新)

这是 cross-embodiment 的 "翻译器"。输入是:
- 当前 target observation $o_k^t$
- Source demonstration 的 frame pair $(o_k^s, o_{k+n}^s)$, 其中 $n$ 是 temporal offset

输出是 target-compatible 的 future latent goal $\hat{z}_{\text{goal}}^t$。

**架构细节** (对应 Figure 2 底部):

Step 1: Shared encoder 编码三个 frame:
$$z_k^t = E(o_k^t), \quad z_k^s = E(o_k^s), \quad z_{k+n}^s = E(o_{k+n}^s)$$

Step 2: 两个 cross-attention module 分别建模两个 factor:

$$f_{\text{emb}} = \text{Attn}(Q=z_k^t, K=z_k^s, V=z_k^s)$$
$$f_{\text{mot}} = \text{Attn}(Q=z_{k+n}^s, K=z_k^s, V=z_k^s)$$

这两个 attention 的设计意图很关键:
- $f_{\text{emb}}$: **embodiment correspondence**。Query 是 target 当前 latent, Key/Value 是 source 当前 latent。这本质上是在问 "source demonstrator 现在所处的 world state, 对应 target embodiment 的什么状态?" 这个 attention 学到了 cross-embodiment 的 spatial alignment。
- $f_{\text{mot}}$: **motion/temporal correspondence**。Query 是 source 的 future latent, Key/Value 是 source 的 current latent。这是在 source 自己的 representation 空间内建模 "demonstrator 想要达到的 future state 相对于现在的 motion delta"。这把 temporal evolution 从 source 轨迹里提取出来, 但还在 source 的 frame of reference 里。

Step 3: 3D Convolution fusion:

$$f_{\text{fused}} = \phi([z_k^t \oplus f_{\text{emb}} \oplus f_{\text{mot}}])$$

这里 $\oplus$ 是 channel-wise concatenation, $\phi$ 是 3D conv。为什么用 3D conv 而不是 mean pooling? 因为 spatiotemporal 结构需要被 preserve。Ablation (Table 4, 5) 显示在 Change Channel (twisting motion)、Close Box (articulated closure)、Remove Plate (coordination-heavy) 这类需要 structured temporal motion 的任务上, Conv3D 显著优于 mean pooling。Mean pooling 是 permutation-invariant 的, 它丢失了 "哪个 spatial token 在哪个时刻做什么 motion" 的信息。

Step 4: Transformer decoder $\mathcal{T}$ 输出 latent goal:

$$\hat{z}_{\text{goal}}^t = \mathcal{T}(f_{\text{fused}})$$

这个 $\hat{z}_{\text{goal}}^t$ 是在 target embodiment 的 latent space 里的 future state, 它既保留了 source demonstration 的 semantic intent (通过 $f_{\text{mot}}$), 又被 grounded 到 target 的当前状态 (通过 $f_{\text{emb}}$ 和 $z_k^t$)。

### 2.3 CEM Planner

给定 $\hat{z}_{\text{goal}}^t$, 用 Cross-Entropy Method 在 world model 里 search action sequence:

$$\mathbf{a}_{k:k+H-1}^{t*} = \arg\min_{\mathbf{a}} d\big(F_{wm}(z_k, s_k^t, \mathbf{a}), z_{\text{goal}}^t\big)$$

变量:
- $\mathbf{a}_{k:k+H-1}^t$: planning horizon $H$ 内的 action sequence
- $d(\cdot, \cdot)$: latent distance metric (paper 里用 $\ell_1$ 或 $\ell_2$)
- $F_{wm}(z_k, s_k^t, \mathbf{a})$: 递归 rollout, 把 action sequence 逐 step 喂进 dynamics predictor, 得到 predicted future latent

CEM 的具体 procedure (Algorithm 1):
1. 初始化 Gaussian 分布 $\mathcal{N}(M, S)$, $M \in \mathbb{R}^{H \times \dim(a)}$, $S \in \mathbb{R}^{H \times \dim(a)}$
2. 每次 iteration sample $N$ 个 candidate action sequences
3. 每个 candidate 在 world model 里 rollout, 计算 planning loss $\mathcal{L}_i = d(\hat{z}_{t+H}^{(i)}, z_{gt})$
4. 选 top-$K$ lowest loss 的作为 elites
5. 用 elites 的 mean 和 std 更新采样分布, 带 momentum $\beta$:
   $$M \leftarrow \beta M + (1-\beta) M_{\text{elite}}$$
   $$S \leftarrow \beta S + (1-\beta) S_{\text{elite}}$$
6. 最后 return $M$ 作为 optimized action sequence

CEM 的优势: 不需要训练一个 inverse dynamics model 或 action decoder。Action 直接通过 online optimization 在 learned latent dynamics 里 search。这在 embodiment shift 下更 robust, 因为 action space 的具体 semantics 不需要被 explicitly learned。

## 3. 训练 pipeline: 两阶段 + 一个 Stage 0

### Stage 0: Pretrain action-conditioned world model

这个 stage 不是 paper 的重点, 用 V-JEPA 2.1 的 setup 训练。8×A100, 7 days。数据是 target embodiment (Franka) 自己的 interaction trajectories, 包含 $(o_t, s_t, a_t)$。这个 stage 让 world model 学会 "Franka 的 physics 和 kinematics"。

### Stage 1: Train Dreamer Predictor (latent goal inference)

这是 cross-embodiment 的核心。数据是 paired visual trajectories: source (Sawyer 或 UR5e) 和 target (Franka) 的同步 video。注意——**只有 observation, 没有 action**。

Loss function (Eq. 4):

$$\mathcal{L}_{\text{pred}} = \|\hat{z}_{\text{goal}}^t - z_{k+n}^t\|_2^2$$

变量:
- $\hat{z}_{\text{goal}}^t$: Dreamer Predictor 输出的 predicted future latent goal
- $z_{k+n}^t = E(o_{k+n}^t)$: target embodiment 在 $k+n$ 时刻的真实 future latent (作为 supervised target)

这个 loss 的意义: 让 Dreamer Predictor 学会 "给定 source 的 $(o_k^s, o_{k+n}^s)$ 和 target 的 $o_k^t$, 预测 target 在 $k+n$ 时刻应该处于的 latent state"。

**Temporal perturbation trick**: 训练时不直接用 $(o_k^s, o_{k+n}^s)$, 而是 perturb 当前 timestamp: $(o_{k+\delta}^s, o_{k+n}^s)$, 其中 $\delta \sim \mathcal{U}(-r, r)$, $r$ 是 perturbation radius。

这个 trick 的 intuition: inference 时, target 和 source 的 execution speed 不一致 (因为 morphology 不同), 所以 "source 的 frame $k$" 和 "target 的 frame $k$" 在 semantic progress 上不严格对齐。通过 perturbation, 强制 predictor 学会 robust 到 source frame 的 temporal offset, 不要 overfit 到严格的时间对齐。这相当于 data augmentation in temporal domain。

8×A100, 2.5 days。

### Stage 2: Action Co-Training

Freeze Dreamer Predictor, unfreeze world model 的 dynamics predictor $F_{wm}$。Loss (Eq. 5):

$$\mathcal{L}_{\text{plan}} = \|F_{wm}(z_k^t, s_k^t, \mathbf{a}_{k:k+n-1}^t) - \hat{z}_{\text{goal}}^t\|_2^2$$

变量:
- $\mathbf{a}_{k:k+n-1}^t$: 从 dataset 采样的真实 action sequence
- $F_{wm}(z_k^t, s_k^t, \mathbf{a}_{k:k+n-1}^t)$: world model 用真实 action rollout 得到的 predicted future latent
- $\hat{z}_{\text{goal}}^t$: frozen Dreamer Predictor 输出的 goal

这个 stage 的目的: 让 world model 的 latent rollout 分布和 Dreamer Predictor 输出的 goal 分布对齐。为什么需要这个? 因为 Stage 0 训练的 world model 只见过 "自然 execution" 的 latent trajectories, 但 inference 时 goal 是 Dreamer Predictor 推断出来的, 可能在 latent space 里有 slight distribution shift。Stage 2 把这个 gap 闭合。

8×A100, 1 day。

## 4. Inference: Adaptive Goal Updating

这个机制是 long-horizon 任务的关键。Algorithm 3 的核心 logic:

1. 初始化: 从 source demonstration 取第一对 reference frames $(o_i^s, o_{i+\Delta}^s)$, $i=1$, $\Delta$ 是 temporal offset
2. 用 Dreamer Predictor 推断 $\hat{z}_{\text{goal}}^t$
3. CEM planning, execute first action $a_0^*$
4. Observe 新的 $o_{\text{next}}^t$, encode 成 $z_{\text{next}}^t$
5. 计算 discrepancy: $D = d(z_{\text{next}}^t, \hat{z}_{\text{goal}}^t)$
6. **如果 $D < \epsilon$**: 认为 sub-goal 达成, advance 到 source 的下一对 reference frames, 重新推断 $\hat{z}_{\text{goal}}^t$
7. **如果 $D \geq \epsilon$**: sub-goal 未达成, 保持当前 goal, 继续往这个 goal planning

这个 adaptive mechanism 解决的问题: source 和 target 的 execution speed 不同。如果固定频率 advance source reference, target 可能还没达到前一个 sub-goal 就被强行 push 到下一个, 导致 compounding error。Adaptive updating 让 source 的 progress 和 target 的实际 progress 同步。

## 5. 实验结果深度分析

### 5.1 三个 evaluation suite 的设计

这个设计很关键, 它体现了 paper 想测的 generalization hierarchy:

1. **Behavior Grounding**: Stage 1 + Stage 2 都见过的 task。测 in-domain execution。
2. **Cross-Embodiment Bridging**: Stage 1 见过但 Stage 2 没见过的 task。测 "Dreamer Predictor 推断的 goal 是否足够好, 让 world model 能 plan 出没见过的 action"。
3. **Zero-Shot Generalization**: 完全没见过的 task configuration。测 extreme distribution shift 下的 robustness。

### 5.2 Simulation results (Table 2)

| Method | Behavior Grounding | Cross-Embodiment Bridging | Zero-Shot Generalization |
|--------|-------------------|--------------------------|-------------------------|
| VPP | 0.47 | 0.28 | 0.04 |
| XSkill | 0.39 | 0.17 | 0.03 |
| Demo-JEPA | 0.31 | **0.45** | **0.36** |

有趣的 pattern:
- **Behavior Grounding**: VPP 最好 (0.47)。这合理, 因为 VPP 直接学 visual prediction → action mapping, in-domain 时 trajectory regularity 强, 这种 direct mapping 最 efficient。
- **Cross-Embodiment Bridging**: Demo-JEPA 大幅领先 (0.45 vs 0.28)。这说明 Dreamer Predictor 推断的 latent goal 确实能在 world model 里 plan 出 reasonable action, 即使这个 action 没在 Stage 2 被直接 supervised。
- **Zero-Shot**: Demo-JEPA 0.36, baselines 基本崩溃 (0.04, 0.03)。这是 9x 的 gap! 说明 latent goal planning 在 extreme shift 下远比 action-level transfer robust。

这个 trend——"in-domain 略弱, out-of-domain 大幅强"——是 representation learning 方法典型的 signature。Latent space 抽象掉了 surface statistics, 代价是 in-domain 的精细 pattern matching 变弱, 但换来的是 out-of-domain 的 robustness。

### 5.3 Real-world results (Table 3)

| Method | Behavior Grounding | Cross-Embodiment Bridging | Zero-Shot |
|--------|-------------------|--------------------------|-----------|
| VPP | 0.65 | 0.53 | 0.00 |
| XSkill | 0.45 | 0.40 | 0.05 |
| Demo-JEPA | 0.43 | **0.55** | **0.25** |

Real-world 的 gap 更明显, 因为 UR5e → Franka 的 morphology 差异 + reality gap 让 action-level 方法在 zero-shot 完全失效 (VPP 0.00, XSkill 0.05), 而 Demo-JEPA 还能到 0.25。

### 5.4 Oracle vs Naive vs Demo-JEPA (Table 4, 5)

这个 ablation 非常 informative:

- **V-JEPA 2.1 (Naive)**: 直接用 source 的 future latent 作为 target 的 goal。完全失败 (table 里基本是空白)。这证明: V-JEPA 的 latent space 本身**不**是 cross-embodiment transferable 的。Source 和 target 的 latent 即使在 "相同 task" 下也是 different manifold。
- **V-JEPA 2.1 (Oracle)**: 用 target 的 ground-truth future latent 作为 goal。这是 upper bound。Simulation zero-shot 0.42, real-world zero-shot 0.28。
- **Demo-JEPA**: 0.36 (sim), 0.25 (real)。

Demo-JEPA 在 zero-shot 下达到 oracle 的 ~86% (sim) 和 ~89% (real)。这说明 Dreamer Predictor 的 cross-embodiment translation 相当 effective, 把大部分 "source intent → target goal" 的 gap 闭合了。

### 5.5 Conv3D ablation

| Method | Cross-Embodiment Bridging Avg | Zero-Shot Avg |
|--------|------------------------------|---------------|
| Demo-JEPA | 0.45 | 0.36 |
| w/o Conv3D | 0.44 | 0.29 |

Cross-embodiment bridging 几乎没差 (因为这些 task motion 相对简单), 但 zero-shot 差很多 (0.36 → 0.29)。这说明 Conv3D 的 spatiotemporal modeling 在 unseen task 上更重要——简单的 in-distribution motion 可以靠 attention 隐式建模, 但 novel motion pattern 需要 explicit spatiotemporal structure。

### 5.6 Demo-DP extension (Table 6, 7)

这个实验设计很巧妙: 把 Dreamer Predictor 的输出作为 Diffusion Policy 的 conditioning signal, 而不是用 CEM planning。

| Method | Behavior Grounding | Cross-Embodiment Bridging | Zero-Shot |
|--------|-------------------|--------------------------|-----------|
| DP | 0.23 | — | — |
| Demo-DP | 0.28 | 0.44 | 0.18 |
| Demo-JEPA | 0.31 | 0.45 | 0.36 |

观察:
- Demo-DP > DP: 证明 Dreamer Predictor 的 latent goal 是 strong conditioning signal, 即使在 standard imitation learning 框架里也有用。
- Demo-DP > VPP, XSkill in zero-shot: latent goal 作为 conditioning 比 visual prediction 或 skill prototype 更 robust。
- Demo-JEPA > Demo-DP in zero-shot (0.36 vs 0.18): **CEM planner 在 world model 里 search 比 Diffusion Policy 直接生成 action 更 robust 到 distribution shift**。Diffusion Policy 是 "local action expert", in-domain 强但 out-of-domain 弱; CEM + world model 是 "model-based planning", 可以 leverage world model 的 generalization能力。

这个对比揭示了 representation learning 的一个深层 trade-off: **action decoder 的 capacity 越强, in-domain 越好, 但 out-of-domain 越 overfit**。Planner-based execution 因为是在 learned dynamics 里 online search, 本身就有更强的 compositional generalization。

### 5.7 Scaling study (Table 8)

| Scaling | Ratio | Cross-Embodiment Bridging | Zero-Shot |
|---------|-------|--------------------------|-----------|
| Data | 20% | 0.27 | 0.27 |
| Task | 20% | 0.18 | 0.18 |
| Data | 50% | 0.38 | 0.29 |
| Task | 50% | 0.33 | 0.25 |
| Full | 100% | 0.45 | 0.36 |

关键 finding: **task diversity 比 per-task data volume 更重要**。20% task scaling (0.18) 比 20% data scaling (0.27) 差很多, 50% 也是同样 pattern。

这个 finding 和 LLM pretraining 的 scaling law 有意思的对比: 对 representation learning, diversity (coverage of semantic space) 比 repetition (more samples of same semantics) 更重要。Dreamer Predictor 需要学到 "source intent → target goal" 的 general mapping, 这个 mapping 的 generalization 来自见过 diverse 的 task semantic, 而不是对同一 task 的更多 samples。

## 6. 为什么 JEPA latent space 适合 cross-embodiment?

这是 paper 最深层的 insight, 我觉得值得展开。

JEPA (Joint-Embedding Predictive Architecture, LeCun 2022) 的核心思想: **在 latent space 里做 prediction, 而不是在 pixel space 里做 reconstruction**。

对比三种 representation learning paradigm:
1. **Pixel reconstruction (MAE, VAE, diffusion)**: 必须重建所有 pixel, 包括 background texture、lighting、robot hardware appearance。这些对 task intent 是 nuisance variable, 但占据了 representation capacity。
2. **Contrastive (SimCLR, MoCo)**: 学习 invariant representation, 但不直接学习 predictive dynamics。
3. **JEPA**: 在 latent space 里预测 future latent, loss 是 latent-level 的 $\|\hat{z} - z\|^2$。Nuisance variable 因为不影响 prediction target, 自然被 abstracted away。

对 cross-embodiment, 这意味着:
- Sawyer 的红色外观 vs Franka 的白色外观 → 被 abstracted away
- 不同的 camera angle → 被 abstracted away  
- 不同的 background → 被 abstracted away
- **保留的是**: object motion, contact dynamics, spatial relationship——这些是 task intent 的 carrier。

这就是为什么 Demo-JEPA 的 latent goal 能跨 embodiment transfer: latent space 已经 implicit 地把 "task-relevant structure" 和 "embodiment-specific appearance" 分离开了。

参考: LeCun 的 JEPA position paper https://openreview.net/pdf?id=BZ5a1r-kVsf

## 7. Limitations 和我的思考

Paper 自己提的 limitations:
1. World model fidelity 限制 complex, high-precision task 的表现。
2. 训练时仍需 temporal/progress-aware alignment (GTCC [47] for real-world data)。

我的额外观察:
- **Inference latency**: CEM 是 iterative optimization, 每个 step 要 sample $N$ 个 candidate, rollout, evaluate。虽然 paper 没报 latency, 但这肯定比 direct policy inference 慢。Real-time control 可能是 challenge。
- **Latent distance metric $d$**: paper 用 $\ell_1$ 或 $\ell_2$, 但 latent space 的 geometry 不一定是 Euclidean 的。更好的 metric (比如 learned metric 或 cosine) 可能 help。
- **Goal inference 的时间 horizon**: 固定 $n$ 或 $\Delta$ 可能不 optimal。Adaptive horizon based on task complexity 可能 better。
- **Multi-modal demonstrations**: 如果 source demonstration 有 multiple ways 做同一 task, Dreamer Predictor 怎么处理? Paper 没讨论。

## 8. 和你之前 work 的 connection

Andrej, 这篇 paper 的思路和你一直 advocate 的 "world model + planning" 路线高度一致。从你的 LLM 'hallucination' is just compression paper 到最近对 world model 的讨论, 核心 thesis 都是: **intelligent behavior 来自对 world dynamics 的 internal model + 在 model 里的 search/planning**。

Demo-JEPA 把这个 thesis 应用到 cross-embodiment imitation, 而且 empirical 证明了: model-based planning (CEM + world model) 在 distribution shift 下比 direct policy (Diffusion Policy) 更 robust。这支持了 "world model 是 generalization 的 key" 这个 hypothesis。

另外, V-JEPA 2.1 作为 backbone, 也 validate 了 self-supervised video pretraining 对 robotics task 的 value——不需要 task-specific supervision, 纯 video prediction 的 latent space 就足够 informative 来 support planning。

参考你的相关讨论:
- https://karpathy.ai/
- 你的 software 2.0 / software 3.0 思路: https://x.com/karpathy/status/1757612032005173362

## 9. 总结

Demo-JEPA 的贡献可以总结为三个 reformulation:

1. **Cross-embodiment imitation = latent goal-conditioned planning** (不是 action alignment)
2. **Demonstration = implicit future goal specification** (不是 motor primitive sequence)
3. **JEPA latent space = embodiment-agnostic intent representation** (不是 pixel-level visual prediction)

Empirical evidence 强支持这三个 reformulation, 尤其是 zero-shot generalization 的 9x gap over baselines。

方法的 limitation 主要在 world model fidelity 和 training-time alignment 需求, 但 core insight——"在 predictive latent space 做 cross-embodiment transfer"——我认为是 right direction。未来如果 world model 能 scale 到更 complex task 和 longer horizon, 这个 paradigm 有潜力成为 cross-embodiment imitation 的 standard framework。

---

如果你想深入某个具体方面 (比如 Dreamer Predictor 的 attention pattern 可视化、CEM 的 hyperparameter sensitivity、或者和 RT-X / Open X-Embodiment 的对比), 我可以继续展开。
