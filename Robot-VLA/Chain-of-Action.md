---
source_pdf: Chain-of-Action.pdf
paper_sha256: 32f423d1f9b384bb554a1d07e866f9f122b82b47fd87c45b8021d93ed945ee46
processed_at: '2026-08-03T15:26:07-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Chain-of-Action 用人话讲

好,我换个讲法,用更直觉的方式把这篇 paper 的精髓讲出来。

## 一句话版本

传统机器人 policy 是"走一步看一步",CoA 是"先定终点,再倒着规划路线"。

就这么简单。但这个反转背后有很深的道理。

---

## 1. 一个生活类比:从家到餐厅

想象你要从家走到一家餐厅吃饭。

**Forward method (ACT, Diffusion Policy) 的做法**:
你站在家门口,看一眼周围,决定"先迈左脚往东走两步"。走到那里再看一眼,"嗯,往北走三步"。每一步都只看眼前,根据当下 observation 决定下一个 action。走多了,累积误差越来越大,最后你可能走到隔壁小区去了。

这就是 **compounding error**。每一步都有一点点偏差,走到第 50 步时,你已经在完全错误的地方了。Forward method 的训练 objective 只管"下一步对不对",不管"最终到没到餐厅"。

**CoA 的做法**:
你先确定餐厅在哪——这就是 **keyframe action**(task-specific goal)。然后从餐厅往回想:要到餐厅,我得先到那个路口;要到那个路口,我得先经过那个公园;要到公园,我得先出家门。

这样倒着规划,每一步都"知道"它要去往哪里。即使中途有点小偏差,因为终点 anchor 一直在那里 pull 你,不会跑偏太多。

这就是 **global-to-local structure**。Goal 给整个 trajectory 提供了一个 anchor,local action 被 global goal tightly constrained。

---

## 2. 为什么 Forward Method 有根本问题

传统的 visuo-motor policy 训练时,loss 是这样:

$$\mathcal{L} = \|\hat{a}_{t+1} - a_{t+1}\|^2$$

变量解释:
- $\hat{a}_{t+1}$: 模型预测的下一个 action
- $a_{t+1}$: ground-truth 下一个 action
- $\|\cdot\|^2$: squared L2 distance

这个 loss 只看**一步**。模型学的是"给定当前画面,下一步应该怎么动"。它根本不知道这个 task 要 30 步以后才能完成,也不在乎 30 步后是否到达 goal。

ACT 用 action chunking 缓解这个问题——一次预测 20 步。但这只是 symptom treatment,没有改变 forward 的 myopic nature。Chunk 内部 20 步依然是从前往后预测的,第 20 步的错误依然来自前面 19 步的累积。

Diffusion Policy 用 denoising 过程建模 multimodal distribution,但 denoising 的方向是 noise→action,不是 time forward/backward,所以也没有解决时间维度上的 compounding error。

**根本问题**: forward method 从来没有显式地告诉模型"你要去哪里"。它只知道"现在这样,下一步那样"。这是 **missing the forest for the trees**。

---

## 3. CoA 的反向分解

CoA 把 trajectory 的 joint probability 反过来分解:

$$p(a_{1:T} | O) = \underbrace{p(a_T | O)}_{\text{先预测终点}} \cdot \underbrace{\prod_{t=T-1}^{2} p(a_t | a_{t+1:T}, O)}_{\text{从终点倒着生成}} \cdot \underbrace{p(a_1 | a_{2:T}, O)}_{\text{最后生成当前要执行的}}$$

变量解释:
- $a_{1:T}$: 从当前 step 到 keyframe 的完整 action 序列,下标 1 是最早执行的,$T$ 是 keyframe(goal)
- $O$: observation(图像 $I$ + proprioceptive state $S$)
- $a_T$: keyframe action,就是"餐厅位置"
- $p(a_T | O)$: 第一步,先从画面预测餐厅在哪
- $p(a_t | a_{t+1:T}, O)$: 已知未来的 action,反推前一步。condition on $a_{t+1:T}$ 意味着"知道后面要去哪"
- $p(a_1 | a_{2:T}, O)$: 最后一步,生成当前立刻要执行的 action

注意 generation 顺序: 先生成 $a_T$(goal),然后 $a_{T-1}, a_{T-2}, ..., a_1$。但 **execution 顺序** 是反过来的: 先执行 $a_1$,再 $a_2$,直到 $a_T$。

这就是 "generate backward, execute forward"。

### Keyframe 是什么

Keyframe 定义很朴素: gripper state 改变(比如从张开到闭合,完成 grasp),或者 joint velocity 接近零(停下来,完成一个 phase)。这些时刻对应 task 的语义转折点。

比如"拿杯子放到架子上":
- Keyframe 1: 抓住杯子的那一刻
- Keyframe 2: 杯子放到架子上松手那一刻

每个 keyframe 之间的 sub-trajectory 就是一个独立的训练 sample。CoA 学的是: 给定某个 phase 起点,生成到达下一个 keyframe 的完整 trajectory(反向生成)。

---

## 4. 四个必要的设计细节

光有"反向生成"这个 idea 不够,实际做起来会遇到四个坑,作者说这四个 design 是 **necessary**,不是 nice-to-have。

### 坑 1: Continuous Action Token

如果你把 action 离散化成 bins(像 LLM tokenize 那样),每个 bin 有 resolution loss。Forward model 里这还好,但 CoA 是 backward generation——量化误差**从 goal 往回累积**。到 $a_1$ 的时候,误差已经放大了 $T$ 倍。

所以 CoA 用 continuous representation: $\mathbf{x}_t = W_{enc}\mathbf{a}_t + b_{enc}$,线性投影到 latent space。

但 continuous token 训练有个问题: latent space 没有 regularization,encoder 可能学到乱七八糟的表示,导致 autoregressive decoding 不稳定。

解决: **Latent Consistency Loss**

$$\mathcal{L}_{consistency} = \|\hat{\mathbf{x}}_t - f_{enc}(\mathbf{a}_t)\|^2$$

- $\hat{\mathbf{x}}_t$: decoder 预测的 latent token
- $f_{enc}(\mathbf{a}_t)$: ground-truth action 通过 encoder 得到的 latent
- 这个 loss 强制 decoder 输出的 latent 和 encoder 的 latent 对齐

Ablation 数据很吓人:
| Loss type | Success Rate |
|-----------|-------------|
| Latent consistency | 0.756 |
| Action reconstruction (替代) | 0.212 |

换成直接 reconstruct action,s success 从 75.6% 暴跌到 21.2%,而且 trajectory 出现"unnatural curling"——轨迹会卷起来,完全不像正常动作。

**Intuition**: Latent space 需要时间上的 consistency,才能让 autoregressive decoding 一步一步稳定推进。直接 reconstruct action 只约束了"一个 action 对不对",没约束"latent space 的时间结构"。

### 坑 2: Multi-Token Prediction (MTP)

反向 autoregressive 传播了 high-level intent,但 local 的连续性没被显式建模。比如 $a_{t}$ 到 $a_{t-1}$ 之间的 smoothness,光靠 reverse chain 不够。

解决: decoder 最后 $K$ 层,每层预测不同 future step。Layer $k$ 预测 $\hat{x}_{t+k}$,$k=1,...,K$。一次 forward pass,model 同时"看到"接下来 $K$ 步的依赖关系。

Ablation:
| K (MTP heads) | Success Rate |
|--------------|-------------|
| 1 | 0.710 |
| 2 | 0.704 |
| 4 | 0.720 |
| 5 | **0.756** |
| 8 | 0.672 |
| 10 | 0.660 |

$K=5$ 最佳。太少 → underutilize local context;太多 → disrupt causal structure。这是个 sweet spot。

### 坑 3: Dynamic Stop

Continuous action space 没有 EOS token,模型不知道什么时候该停。如果一直 generate,会 over-generate,生成一堆没用的 action。

解决: **distance-based stop**

$$\text{STOP} \iff \|\hat{a}_t - S_{current}\| < \epsilon$$

- $\hat{a}_t$: 当前预测的 action
- $S_{current}$: 当前真实的 gripper state(end-effector pose)
- $\epsilon$: 距离阈值

**Intuition**: backward generation 从 keyframe 往回走。当预测的 action 离当前真实 state 很近时,说明"倒着走"已经走到现在了,可以停了。

### 坑 4: Reverse Temporal Ensemble

ACT 的 temporal ensemble 是 forward 的——多次 rollout 在时间上对齐求平均。CoA 反过来,需要对齐策略。

CoA 用 **predicted keyframe $a_T$ 作为 anchor**。多次 backward rollout 都从同一个 $a_T$ 开始 decode,在 anchor 附近对齐。这有一个妙处: 因为每条 trajectory 的 error 都被 keyframe accuracy 约束,提升 keyframe accuracy(通过 ensemble)就等于收紧所有 trajectory 的 error bound。

Ablation:
| Setting | Success Rate |
|---------|-------------|
| No ensemble | 0.660 |
| Reverse ensemble | 0.756 |

+9.6%,非常显著。

---

## 5. 实验里最 telltale 的数字

### Overall(RLBench-60)

| Method | Avg Success |
|--------|------------|
| **CoA** | **0.756** |
| Octo (finetuned) | 0.644 |
| ACT | 0.488 |
| Diffusion Policy | 0.416 |

CoA vs ACT **+16.3%**, CoA vs DP **+23.2%**。

**关键**: CoA 和 ACT 用**完全一样的 architecture**(4-layer encoder + 7-layer decoder + ResNet-18 vision backbone),只是 modeling paradigm 从 forward 改成 backward。这是一个 clean controlled experiment,performance gap 完全归因于 paradigm 本身。

### Spatial Generalization(最有意思的部分)

作者用 object coordinate variance 衡量"物体摆放有多分散",然后看 success rate 和这个 variance 的关系。

**Interpolation vs Extrapolation**(Push Button task):

| Method | Interpolation (in-dist) | Extrapolation (out-of-dist) |
|--------|------------------------|---------------------------|
| CoA | 0.94 | 0.48 |
| ACT | 0.54 | 0.08 |
| DP | 0.18 | 0.04 |

ACT 从 interpolation 到 extrapolation 成功率掉到 15%,DP 掉到 22%,CoA 只掉到 51%。这说明 forward method 在 out-of-distribution spatial position 上几乎完全崩溃,而 CoA 还有相当强的 robustness。

**Intuition**: Forward model 学到的是"看到画面 X → 做 action Y"的 mapping。这个 mapping 对 spatial location 极度敏感——button 出现在 training 时没见过的位置,mapping 直接失效。

CoA 反过来: 先预测 goal(button 在哪里,作为 keyframe action),goal 是 task-level 的,对 spatial location 的 generalization 更好(因为 keyframe action 本身就是"去按那个 button"的终点位置)。然后从 goal 反推轨迹,每一步被 goal anchor,即使 button 位置变了,只要 goal 预测对了,整条 trajectory 都会被 pull 到正确位置。

### Attention Map

Decoder 的 self-attention 显示两个 pattern:

1. **Local chain**: 每个 token attend 最近的几个 predecessor —— 这是 local continuity
2. **Long-range to keyframe**: 后面的 token 强烈 attend 到第一个 token(keyframe) —— 这是 goal anchoring

这直接证实了 CoA 的 hypothesis: 模型确实在用 keyframe 作为 anchor guide 整个 trajectory generation。

### Modeling Paradigm Ablation(最 clean 的 ablation)

| Variant | Success Rate | 描述 |
|---------|-------------|------|
| **Reverse (CoA)** | **0.756** | 反向 + keyframe anchor |
| Forward | 0.668 | 正向 autoregressive,无 keyframe |
| Hybrid | 0.600 | 有 keyframe,但从 keyframe 正向生成 |
| ACT baseline | 0.488 | 正向 + fixed-length chunk |

三个 insight:

1. **Reverse > Forward** (0.756 > 0.668): 反向本身有 +8.8% gain,因为 goal anchoring
2. **Forward > ACT** (0.668 > 0.488): autoregressive modeling 整个 trajectory joint distribution 比 fixed-length chunk 强 +18%,因为 CoA 的 forward variant 也是 autoregressive 整条 trajectory,只是没有 keyframe anchor
3. **Hybrid 最差** (0.600): 光有 keyframe 不够,必须保持 chain-style backward reasoning。Hybrid 从 keyframe 开始正向生成,丢失了 temporal continuity

**核心 insight**: keyframe anchoring 和 chain-style backward reasoning,两者缺一不可。光往 forward model 里塞 keyframe signal(Hybrid / ACT+KF)没用,必须从根本上改成 backward generation。

### ACT+KF 补充实验

作者还做了一个 ACT+KF: 在 ACT 的 action chunk 末尾 append 一个 keyframe action。

| Method | Success Rate |
|--------|-------------|
| ACT | 0.488 |
| ACT+KF | 0.516 |

只提升 +2.8%,非常 marginal。这说明:**光往 forward model 里 inject keyframe signal 没用,必须从 modeling paradigm 上改成 backward generation**。

---

## 6. 真实世界实验

8 个 kitchen tasks,Fetch robot,单 RGB camera:

| Method | Avg Success |
|--------|------------|
| **CoA** | **0.613** |
| ACT | 0.463 |
| DP | 0.363 |

CoA vs ACT **+15%**。Real-world gap 比 simulation 略小(可能 real-world spatial variation 没那么极端),但依然显著。

部署: 10Hz policy + 1000Hz PD controller,ROS 通信,absolute end-effector pose command。

---

## 7. 局限性

Keyframe detection 用的是 heuristic(gripper state change + joint velocity ≈ 0)。这个 heuristic 对 typical manipulation task 很有效,但可能不 generalize 到所有 task type。Future work 可以探索 unsupervised keyframe learning。

另外,CoA 在 RLBench-18 上依然不如 3D-based hierarchical methods(RVT-2 0.814 vs CoA 0.373)。但这是 apples-to-oranges 比较: RVT-2 用 3D point cloud + motion planner + open-loop execution,CoA 是 RGB-only + end-to-end + closed-loop。作者 argue RLBench-18 对 RGB-only policy discriminative power 不足(很多 task 都是 0 success rate),所以他们提出 RLBench-60 作为更公平的 benchmark。

---

## 8. 我的 Intuition 总结

这篇 paper 让我最 excited 的点:

**1. Conceptually clean experiment**

保持 architecture 和 training setup 完全一致,只改 modeling paradigm(forward → backward),拿到 +16% gain。这种 controlled experiment 在 robotics 里很少见,通常大家改 architecture、改 dataset、改 training trick,很难 isolate 一个 variable。CoA 这个实验让"backward generation"这个 idea 的 causal effect 非常清晰。

**2. 解决 root cause,不是 symptom**

Action chunking、temporal ensemble、image goal conditioning 这些都是在 forward paradigm 内打补丁。CoA 直接质疑 paradigm 本身——forward prediction 的 myopic nature 是 root cause,那就把 direction 反过来。这种从根本出发重新思考 problem formulation 的工作,我觉得是 research 最有价值的形式。

**3. 和 LLM CoT 的深层对应**

LLM 的 Chain-of-Thought 是在 token space 里引入显式中间 reasoning step。CoA 的 Chain-of-Action 是在 action space 的 time dimension 上引入显式中间 reasoning step(从 goal 到 current 的 backward chain)。两者本质都是: **给模型一个显式的 intermediate structure,让它 not just predict output, but reason toward output**。

**4. 和 Diffusion 的对比**

Diffusion Policy 的 denoising 也是某种 "backward"(noise → data),但那是 **signal space** 的 backward,不是 **time space** 的 backward。CoA 的 backward 是 time dimension 上的,有显式 goal anchor。Diffusion 的 denoising 没有 task goal 作为 anchor,它只是一个 distribution modeling tool。这是两种不同的 "backward" philosophy。

**5. Potential for VLA scaling**

CoA 的 formulation 是 single autoregressive framework,理论上可以无缝 scale 到 VLA setting: language instruction + visual observation → backward action generation。想象一下 OpenVLA 但 action generation 是 backward from keyframe...这可能是一个很有潜力的方向。Language 可以用来 condition keyframe prediction("把杯子放到架子上" → keyframe = 杯子在架子上的 pose),然后 backward chain 生成 trajectory。

**6. 和 Hindsight 的关系**

Reverse generation 本质是一种 hindsight reasoning——把未来的 goal 当 conditioning。这和 HER (Hindsight Experience Replay) 思想相通,只是 HER 在 RL 里 relabel goal,CoA 在 imitation learning 里用 keyframe 作为 goal。整个 robotics 里 "用未来指导现在" 的思想一直在以不同形式出现。

References:
- Project page: https://chain-of-action.github.io
- ACT paper: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- MTP (Meta, multi-token prediction): https://arxiv.org/abs/2404.19737
- C2F-ARM (keyframe definition): https://arxiv.org/abs/2204.01571
- RLBench: https://arxiv.org/abs/1809.00726
- Octo: https://octo-models.github.io/
- PerAct: https://peract.github.io/
- RVT-2: https://rv-2.github.io/
- OpenVLA: https://openvla.github.io/
- Waypoint BC: https://arxiv.org/abs/2307.14326
- Hierarchical Diffusion Policy: https://hd-policy.github.io/
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- ChainedDiffuser: https://chaineddiffuser.github.io/
- HER (Hindsight Experience Replay): https://arxiv.org/abs/1707.01495
- CoT-VLA (visual chain-of-thought for VLA): https://arxiv.org/abs/2503.22020
- TraceVLA: https://arxiv.org/abs/2412.10345

---

这篇 paper 给我的 meta-level 启示: 很多时候一个领域的 "obvious" paradigm(forward prediction for robotics)可能恰恰是 performance bottleneck 的根源。质疑这个 "obvious" 选择,尝试反方向,可能拿到出乎意料的 gain。当然,backward generation 在 implementation 上有一堆坑(continuous token、dynamic stop、reverse ensemble...),但 root idea 是 clean 的——**先知道要去哪,再规划怎么去**。这大概是 planning 的最朴素直觉,只是 forward end-to-end learning 把这个直觉丢了。

---

# Chain-of-Action: Trajectory Autoregressive Modeling 深度解析

Andrej，这篇 paper 挺有意思，它对 robotic manipulation 里 visuo-motor policy 的建模范式做了一次比较根本性的反转。让我把它拆开来讲，尽量 build 你的 intuition。

## 1. Core Insight: Forward Prediction 的根本问题

传统 visuo-motor policy（ACT、Diffusion Policy、OpenVLA 等）基本都是 **forward prediction paradigm**：给定当前 observation $O_t$，预测 next action $a_{t+1}$ 或者 next action chunk $a_{t+1:t+k}$。

这个范式有一个根本性的问题——**compounding error**。其根源在于 training objective：模型被优化去 minimize $\|a_{t+1} - \hat{a}_{t+1}\|^2$，这个 loss 在每一步独立计算，并不保证 long-horizon task completion。Action chunking（ACT 用的）和 image goal conditioning（Waypoint BC）只是缓解症状，并没有触及 root cause——**forward prediction 的 myopic nature**。

CoA 的核心 insight 是：**把 action generation 过程反转过来**。从 keyframe action（task-specific goal）开始，backward autoregressive 生成整个 sub-trajectory，最终到达当前 state。这样每个 local action 都被 final goal tightly constrained，形成一个 **global-to-local** 的结构。

## 2. 数学 Formulation: 反向 Chain 分解

CoA 把 trajectory distribution 分解为：

$$p(a_{1:T} | O) = \underbrace{p(a_T | O)}_{\text{Keyframe Action}} \cdot \underbrace{\prod_{t=2}^{T-1} p(a_t | a_{t+1:T}, O)}_{\text{Reverse Reasoning Actions}} \cdot \underbrace{p(a_1 | a_{2:T}, O)}_{\text{Executed Action}}$$

变量解释：
- $a_T$: keyframe action，即 sub-trajectory 末端的 goal action。这里 $T$ 是 sub-trajectory 长度，下标 $T$ 表示末位（goal 位置）
- $a_{1:T}$: 从当前 step 到 keyframe 的完整动作序列
- $O$: observation context，包含 visual input $I$ 和 proprioceptive state $S$
- $p(a_T | O)$: 第一项，直接从 observation 预测 keyframe（goal）
- $p(a_t | a_{t+1:T}, O)$: 中间项，**反向** autoregressive，conditioning on 未来所有 action
- $p(a_1 | a_{2:T}, O)$: 最后一项，这是**最先执行**的 action（因为生成是反向的，最后生成的是最早执行的）

这里有一个很关键的直觉：**reverse causal dependency**。每个 $a_t$ 都 condition 于 $a_{t+1:T}$，即"未来"。这在 forward model 里是 impossible 的——forward model 只能 condition on past。但 CoA 的 reverse 设置让每个 action 都"知道"它要去往哪里。

### Keyframe 的定义

CoA 借用 C2F-ARM 的 keyframe 定义：**gripper state 改变** 或者 **joint velocities 接近零** 的时刻。这捕捉了语义上有意义的 phase transition（grasp completion、object placement 等）。Keyframe 作为一个 action（不是 embedding），可以 share 同一个 action embedding space，实现 seamless backward generation。

### Training Sample 构造

对每个 expert demonstration，随机 sample 一个起始 time step，然后从该 step 到下一个 keyframe 之间形成一个 sub-trajectory。Observation $O$ 取自起始 step，$a_{1:T}$ 是从起始 step 到 keyframe（含 keyframe）的 action 序列。这样每个 $(O, a_{1:T})$ 是一个独立 training example。

## 3. 四个 Essential Designs

光有 reverse autoregressive 的 idea 是不够的，作者强调这四个 design 是 **necessary**（not optional）for stable training 和 reliable closed-loop execution。

### 3.1 Continuous Action Token Representation

**为什么不能用 discrete bins**：Long-horizon autoregressive generation 中，quantization error 会累积。在 CoA 的 backward 设置里，这个累积方向是**从 goal 往回**——即使每步 quantization 误差很小，反向累积到 $a_1$ 时可能已经显著偏离。

**具体实现**：用 linear projection 把 action $\mathbf{a}_t \in \mathbb{R}^8$ 映射到 latent token $\mathbf{x}_t = W_{enc}\mathbf{a}_t + b_{enc}$。这里 action 是 8-dim（3 position + 4 quaternion + 1 gripper）。

**Latent Consistency Loss**：
$$\mathcal{L}_{consistency} = \|\hat{\mathbf{x}}_t - f_{enc}(\mathbf{a}_t)\|^2$$

变量：
- $\hat{\mathbf{x}}_t$: Transformer decoder 预测的 latent token at step $t$
- $f_{enc}(\mathbf{a}_t) = W_{enc}\mathbf{a}_t + b_{enc}$: ground-truth action $\mathbf{a}_t$ 通过 encoder 得到的 latent
- $\|\cdot\|^2$: L2 norm squared

这个 loss 的作用是**regularize latent space**，让它和 action space 的时间动态对齐。Ablation 显示，换成直接 action reconstruction loss，success rate 从 0.756 暴跌到 0.212，trajectory 出现 unnatural curling。这说明 latent consistency 对 autoregressive decoding 的稳定性至关重要。

### 3.2 Multi-Token Prediction (MTP) for Locality Modeling

Reverse autoregressive 传播了 high-level intent，但**不显式建模 local action dependencies** within sub-trajectory。CoA 借鉴 Meta 的 MTP 工作（Gloeckle et al., 2024），把 transformer decoder 的最后 $K$ 层分配给不同 future step 的 prediction。

具体：layer $k$ predicts token $\hat{x}_{t+k}$, $k=1, ..., K$。这样一次 forward pass 模型就能"aware"接下来 $K$ 步的 mutual dependencies。这是一个 **temporal locality** 的 inductive bias，只在 training 时用，inference 时移除。

Ablation 显示 $K=5$ 最佳（0.756），$K=1$ 降到 0.710，$K=10$ 降到 0.660。太少 underutilize local context，太多 disrupt causal structure。

### 3.3 Dynamic Stop via Distance Criterion

Continuous action space 里没有 discrete EOS token。CoA 设计了一个 **distance-based stop mechanism**：

$$\text{STOP}(\hat{a}_t, S) \iff \|\hat{a}_t - S\| < \epsilon$$

变量：
- $\hat{a}_t = f_{dec}(\hat{x}_t)$: 当前预测的 action
- $S$: 当前 gripper 的 proprioceptive state（end-effector pose）
- $\epsilon$: 距离阈值

直觉：backward generation 从 keyframe 往回走，当预测的 action 充分接近**当前真实 state**时，说明 backward trajectory 已经"走到现在"，可以停止。这个 criterion 对 action representation agnostic——delta action 或 joint-space 都能用，只需调整 reference point。

### 3.4 Reverse Temporal Ensemble

ACT 原版的 temporal ensemble 是基于 forward assumption 设计的——多次 forward rollout 在时间上对齐求平均。CoA 反过来，所以需要一个 reverse-compatible variant。

核心 idea：**用 predicted keyframe action $a_T$ 作为 anchor point**。多次 backward rollout 都从同一个 $a_T$ 开始 decode，在 $a_T$ 附近对齐求平均。

这有一个独特优势：每个 trajectory 的 compounding error 都被 keyframe accuracy 约束。通过 ensembling 提升 keyframe accuracy，这个约束被进一步收紧。

Ablation：non-ensemble 0.660 → reverse ensemble 0.756，提升明显。

## 4. 网络架构细节

**Encoder**：4-layer Transformer encoder
**Decoder**：7-layer Transformer decoder，最后一层包含 multiple parallel heads for MTP

**Vision**：每路 RGB 通过 ResNet-18 提取 visual tokens（4 个 cameras：wrist, front, left shoulder, right shoulder）
**State**：gripper state 通过 learnable linear layer 投影成 token
**Token assembly**：vision tokens + state token concatenate 后过 Transformer encoder，产出 context features
**Decoder input**：learnable SOS token（对应 final keyframe action）
**Autoregressive**：decoder 一步一步反向生成 action tokens，加 sinusoidal positional embeddings 提供时间顺序 hint
**Action embedding**：linear projection layers（encoder $f_{enc}$, decoder $f_{dec}$），share latent space

## 5. 完整 Loss Function

$$\mathcal{L}_{total} = \sum_{t=1}^{T} \sum_{k=1}^{K} \|\hat{\mathbf{a}}_{t+k-1}^k - \mathbf{a}_{t+k-1}\|^2 + \lambda_1 \|\hat{x}_{t+k-1}^k - f_{enc}(\mathbf{a}_{t+k-1})\|^2$$

变量详解：
- $t$: decoding step，从 1 到 $T$（sub-trajectory 长度）
- $k$: MTP head index，从 1 到 $K$
- $\hat{\mathbf{a}}_{t+k-1}^k$: 第 $k$ 个 MTP head 在 decoding step $t$ 预测的 action。上标 $k$ 表示哪个 head，下标 $t+k-1$ 表示预测的是哪一步的 action
- $\mathbf{a}_{t+k-1}$: ground-truth action at step $t+k-1$
- $\hat{x}_{t+k-1}^k$: 第 $k$ 个 head 在 step $t$ 预测的 latent embedding
- $f_{enc}(\mathbf{a}_{t+k-1})$: ground-truth action 的 encoded latent
- $\lambda_1$: latent consistency loss 的权重

**Masking**：当 $t + k - 1 > T$ 时，对应 term 被 mask out（不贡献 loss），避免预测超出 trajectory horizon。Batch training 时，$T_{max}$ 是 dataset 中最长 sub-trajectory 的长度，短的 sequence zero-pad，padded step 的 loss 也 mask out。

## 6. 实验结果深度解读

### 6.1 RLBench-60 Overall

| Method | Avg Success Rate |
|--------|-----------------|
| CoA | 0.756 |
| ACT | 0.488 |
| Diffusion Policy | 0.416 |
| Octo (finetuned) | 0.644 |

CoA vs ACT: **+16.3%**，在 81.7% tasks 上更好
CoA vs DP: **+23.2%**，在 80.0% tasks 上更好

关键观察：**CoA 和 ACT 共享相同的 Transformer encoder-decoder 架构和 training setup**，performance gap 完全来自 modeling paradigm 的改变。这是一个非常干净的 controlled experiment。

### 6.2 Spatial Generalization Analysis（这是最有意思的部分）

作者用 object coordinate variance 衡量 spatial distribution difficulty。三个发现：

**Finding 1: Pearson correlation**
| Method | Pearson r (success vs spatial variance) |
|--------|---------|
| CoA | -0.1679 |
| ACT | -0.2471 |
| DP | -0.2455 |

所有方法都 negative correlation（spatial variance 越大，success 越低），但 CoA 的负相关**最弱**——意味着对 spatial perturbation 更 robust。

**Finding 2: Interpolation vs Extrapolation**（Push Button task）

| Method | Interpolation | Extrapolation |
|--------|---------------|---------------|
| CoA | 0.94 | 0.48 |
| ACT | 0.54 | 0.08 |
| DP | 0.18 | 0.04 |

CoA 在 extrapolation 上的 advantage 极其显著。Forward method（ACT, DP）从 interpolation 到 extrapolation 几乎崩塌（ACT: 0.54→0.08, DP: 0.18→0.04），而 CoA 的衰减相对温和（0.94→0.48）。

**Intuition**: Forward model 学到的是 "current observation → next action" 的 mapping，这个 mapping 对 spatial location 非常 sensitive。当 object 出现在 training distribution 之外的位置，mapping 直接失效。CoA 反过来——先预测 goal（keyframe），goal 的 spatial 位置可以 generalize（因为 goal 是 task-level 的），然后从 goal 反推轨迹，每一步都被 goal 约束，因此更 robust。

**Finding 3: Attention Map Analysis**

Decoder self-attention 显示两种 pattern：
1. **Chain-like local dependencies**: 每个 action token 主要 attend 最近的几个 predecessor tokens（局部连贯性）
2. **Long-range dependencies to keyframe**: 后期 tokens 强烈 attend 到 initial keyframe token（layer 1 红框, layer 6 大面积）

第二个 pattern 尤其重要——它直接证实了 CoA 的 hypothesis：**模型确实在用 keyframe 作为 anchor 来 guide 整个 trajectory generation**。

### 6.3 Ablation 关键对比

**Modeling Paradigm ablation**:
| Variant | Avg SR | 描述 |
|---------|--------|------|
| Reverse (CoA) | 0.756 | 完整反向 autoregressive + keyframe anchoring |
| Forward | 0.668 | 保留 autoregressive，去掉 keyframe anchoring，正向预测 |
| Hybrid | 0.600 | 保留 keyframe anchoring，去掉 chain-style reasoning（从 keyframe 开始正向生成） |

三个发现：
1. **Reverse > Forward**: 0.756 vs 0.668，reverse ordering 本身有 +8.8% gain
2. **Forward > ACT**: 0.668 vs 0.488，autoregressive modeling 整个 trajectory 的 joint distribution 比 fixed-length chunk 强 +18%
3. **Hybrid 最差**: 0.600，说明光有 keyframe 没 chain-style reasoning 不够，**temporal continuity 是关键**

**Latent consistency loss ablation**:
- Latent consistency: 0.756
- Action consistency (替代): 0.212

这是最 dramatic 的 ablation。直接 reconstruct action 而不约束 latent space，导致 trajectory 出现 "unnatural curling"。这说明 latent space 的 temporal consistency 对 autoregressive decoding 的 stability 是**必需的**。

## 7. 真实世界实验

8 个 kitchen tasks 在 Fetch robot 上：

| Method | Avg Success |
|--------|-------------|
| CoA | 0.613 |
| ACT | 0.463 |
| DP | 0.363 |

CoA 相对 ACT **+15%**，相对 DP **+25%**。Real-world 的 gap 比 simulation 略小，可能是因为 real-world 任务 spatial variation 不如 RLBench-60 那么大。

部署细节：10Hz policy on 4070 GPU + 1000Hz PD controller on robot，通过 ROS 通信。绝对 end-effector pose command。

## 8. 与 Hierarchical Methods 的对比

RLBench-18 上，3D-based hierarchical methods（PerAct、3D Diffuser Actor、RVT-2）依然领先。CoA 在 RLBench-18 上 avg 0.373，而 RVT-2 达到 0.814。

但 CoA 是 RGB-only + end-to-end，而 hierarchical methods 需要 3D point cloud 和 motion planner，open-loop execution between keyframes。CoA 的优势是 closed-loop + 不需要高保真 3D 输入。

作者特别强调 RLBench-18 的局限性：很多 task RGB-only policy 成功率为 0，discriminative power 不足，所以他们提出 RLBench-60 作为更公平的 RGB-only policy benchmark。

## 9. 局限性和 Future Work

**Keyframe heuristic 依赖**: 当前用 gripper state change + joint velocity ≈ 0 的 heuristic 来 split trajectory，可能不 generalize 到所有 task type。作者提到 future work 可以探索 unsupervised keyframe learning。

## 10. 我的 Intuition 总结

这篇 paper 让我想到几个更深的问题：

1. **Backward reasoning 和 LLM 的 CoT 的对应**：CoA 的 backward generation 是一种 action-level CoT。LLM 里 CoT 是从 premise 到 conclusion 的中间 reasoning step；CoA 是从 goal 反推到 current state 的中间 action step。两者都引入了**显式的中间结构**。

2. **和 Hindsight Relabeling 的关系**：Reverse generation 本质上是一种 hindsight-style reasoning——把未来的 goal 当作 conditioning。这和 HER（Hindsight Experience Replay）的思想相通，只是 HER 用在 RL，CoA 用在 imitation learning。

3. **和 Diffusion 的对比**：Diffusion Policy 通过 denoising process 建模 multimodal distribution，本质上也是某种 "backward"（从 noise 到 data）。但 diffusion 的 backward 是在 **noise space**，CoA 的 backward 是在 **action space** 的时间维度上。CoA 的 backward 有显式的 task goal 作为 anchor，diffusion 的 denoising 没有。

4. **Potential scaling**: CoA 的 formulation 是 single autoregressive framework，理论上可以 scale 到 VLA setting——language instruction + visual observation → backward action generation。这可能是 VLA 的下一个 paradigm shift。

References:
- Project page: https://chain-of-action.github.io
- ACT paper: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- MTP (Meta): https://arxiv.org/abs/2404.19737
- C2F-ARM (keyframe definition): https://arxiv.org/abs/2204.01571
- RLBench: https://arxiv.org/abs/1809.00726
- Octo: https://octo-models.github.io/
- PerAct: https://peract.github.io/
- RVT-2: https://rv-2.github.io/
- OpenVLA: https://openvla.github.io/
- Waypoint BC (Shi et al.): https://arxiv.org/abs/2307.14326
- Hierarchical Diffusion Policy: https://hd-policy.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- ChainedDiffuser: https://chaineddiffuser.github.io/

---

这篇 paper 让我最 excited 的点是它做了一个**conceptually clean**的实验：保持 architecture 和 training setup 一致，只改 modeling paradigm（forward → backward），就拿到 +16% 的 gain。这种 controlled experiment 在 robotics 里挺罕见的，也让结论非常有说服力。Backward reasoning 这个 idea 看起来 simple，但它解决的是 forward prediction 的 root cause，而不是 symptom。这种从根本出发重新思考 problem formulation 的工作，我觉得是值得鼓励的方向。
