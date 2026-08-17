---
source_pdf: ProgVLA Progress-Aware Robot Manipulation Skill.pdf
paper_sha256: 3b15b51995a0a32a9beef2a27ffee9a2f88ed6a629bab5ebe5a0af58bc7b4a9f
processed_at: '2026-08-06T06:46:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ProgVLA 人话版

---

## 这篇 paper 在讲什么

一句话：**一个 100M 参数的小模型，没用任何大规模 robot pretraining，在长程任务上打赢了 2.25B 的 SmolVLA 和 7B 的 OpenVLA。**

这听起来有点反直觉。当前 VLA 领域的共识是"大模型 + 大数据 + cross-embodiment pretraining"。ProgVLA 说：在 robot manipulation 这种 task 范围比较窄的场景下，与其堆参数堆数据，不如把 architecture 设计好。

---

## 三个核心 trick

### Trick 1: 漏斗压信息（贡献最大）

机器人看到的世界很"嘈杂"——两个 camera 的图像加起来几百个 patch tokens，语言指令几十个 tokens，还有 proprioception。如果让 action expert 直接 attend 这一大堆东西，它容易分心，尤其在长程任务里会积累错误。

ProgVLA 的做法是**两级漏斗**：

- 第一级：每个 modality 先各自压一遍。vision 压成 8 个 tokens，language 压成 16 个。
- 第二级：所有 modality fuse 之后再压一次，最终只留 **4 个 tokens**。

这 4 个 tokens 就是 policy 看到的"整个世界"。极度 lossy，但极度 task-relevant。

Ablation 显示，去掉第二级漏斗，long-horizon success rate 从 88.6 直接崩到 51.2。这是整篇 paper 最戏剧性的数字。

**人话比喻**：就像你让一个人蒙眼只通过 4 个关键词来理解整个厨房场景。逼他学会抓重点，忽略无关细节。这种"被迫 extractive"的 representation 在 data 少的时候反而更 robust。

### Trick 2: 视觉 backbone 必须微调（贡献第二大）

用 DUNE 这个通用视觉 encoder 当起点，但必须跟 policy 一起 fine-tune。冻结的话掉 13.5 个点。

原因很简单：robot 的 wrist-camera 视角太特殊了——近距离、运动模糊、奇怪的角度。ImageNet 或 self-supervised pretrain 的 visual prior 跟这个 distribution 差很远。你必须让 backbone 在实际任务上 adapt。

**人话比喻**：好比一个人视力很好但从来没近距离看过桌面。你得让他适应"从机器人手腕角度看世界"这个新视角，不然他认不出 gripper 跟物体的接触关系。

### Trick 3: Progress heads（贡献最小但最有意思）

这是 paper 标题里的卖点，但 ablation 显示只贡献 2.3 个点。不过它在 long-horizon 上的 gain 集中且 consistent。

做法表面上借了 offline RL 的 Q/V/S 结构，实际上是一个**伪装成 RL 的 prioritized supervised learning**。

核心 idea 很朴素：
- 给每个时间步打一个"进度分数" $r_t$，离终点越近分数越高
- 训三个小 head 去预测这个分数
- 用预测出来的分数当 weight，乘到 imitation loss 上

效果是：**trajectory 末尾的 frames 被 upweight，开头的 frames 被 downweight**。

为什么这对 long-horizon 有效？因为长程任务的成败往往决定于最后几步——precise grasp、final placement、sub-task transition。一个微小的 action error 就毁掉整个 episode。把 gradient capacity 推向这些 critical frames，等于"在考试重点上多花时间复习"。

**人话比喻**：想象你在学投篮。大部分练习动作是运球跑向篮筐，但真正决定进球的是最后出手那一下。Progress heads 的作用就是告诉模型"最后出手那一下最重要，多练那个"。

---

## 为什么这三个 trick 叠加有效

单看任何一个都不惊艳。但叠在一起，它们解决了一个共同问题：**small data regime 下的 overfitting 和 capacity allocation**。

- 漏斗压缩 → 防止 model 被 patch-level noise 淹没
- 视觉微调 → 让 visual prior 适配任务分布
- Progress reweighting → 把有限的 gradient budget 用在 critical decisions 上

这三个都是 *strong inductive bias*。在 data 少的时候，inductive bias 比 data 量更重要。在 data 多的时候，inductive bias 反而可能限制 model。ProgVLA 赌的就是 manipulation benchmark 属于前者。

---

## 结果有多 impressive

看 LIBERO Long-horizon：

| Model | Params | Long SR |
|---|---|---|
| OpenVLA | 7B | 53.7 |
| SmolVLA | 2.25B | 77 |
| π0 | 3.3B | 73 |
| **ProgVLA** | **0.1B** | **88.6** |

0.1B 打 7B，差距 34.9 个点。这不是微小优化，是 architecture 选择带来的质变。

Meta-World 的 very hard 也类似：ProgVLA 79.6，SmolVLA 2.25B 只有 64。

**越难的任务，ProgVLA 优势越大**。这跟预期一致——长程和困难任务最需要 temporal coherence 和 critical frame focus，正好是 progress heads + bottleneck 的强项。

---

## 作者诚实承认的 caveat

这篇 paper 让我印象深的一点是作者很诚实，不 oversell：

1. **Progress target 是纯时间的**，不包含 semantic subgoal 信息。一个长 episode 的中间和一个短 episode 的末尾可能 $r_t$ 一样。作者明说这不是"verified internal progress estimator"。

2. **Advantage 在 offline RL 意义下 weakly identified**。因为只用 successful demos，每个 context 只配一个 action，Q 和 V 学的几乎是同一个东西的两种 smoothing。更准确的解释是"trajectory-phase reweighting"，不是真正的 action quality evaluation。

3. **没验证 progress heads 是否真的校准**。没做 calibration curve，没做 monotonicity 分析。heads 到底学到了什么，定性上理解但不定量验证。

4. **单 seed，没 sweep hyperparameters**。$T_M=500, \rho=0.8, \beta=0.5$ 都是拍脑袋定的。

5. **Baseline 不严格 matched**。SmolVLA 的数字是从另一个 paper 抄来的，用的 LIBERO release 不同（filtered vs 原版，high-res vs low-res）。作者论证这些差异不足以解释 10+ 点的 gap，但严格 head-to-head 没做。

6. **Real-world 太窄**。单 arm，10 task，in-distribution eval。不测 cross-environment。

---

## 跟主流路线的哲学分歧

当前 VLA 主流 ([π0](https://arxiv.org/abs/2410.24164), [OpenVLA](https://arxiv.org/abs/2406.09246), [RT-2](https://arxiv.org/abs/2307.15818)) 走的是 **"scale backbone + scale robot data"**。背后的信念是：internet-scale vision-language pretraining 给的 broad prior 会 transfer 到 manipulation。

[SmolVLA](https://arxiv.org/abs/2506.08450) 已经质疑过：450M 也能打 10x 大的，但仍依赖 cross-embodiment pretraining。

ProgVLA 推得更极端：**100M，完全不用 robot pretraining，也能打**。

这挑战的是 "scale is all you need" 的 naive 版本。它的潜台词是：在 *narrow task distribution*（比如一个 benchmark 的 40 个 task）下，broad pretraining 的 marginal value 在递减——Open X-Embodiment 的大部分 data 跟 LIBERO 的具体 task 没直接关系。真正起作用的是 *task-relevant inductive bias*。

**但**——这个 claim 的 scope 是有限的。ProgVLA 的 eval 全是 in-distribution。如果你要 generalization to held-out task family 或 cross-environment，broad pretraining 的优势可能才显现。ProgVLA 证明的是"在 narrow distribution 上 small model + strong bias 能赢"，没证明"在 broad distribution 上也能赢"。

---

## 给你的 takeaway

如果你要记住一件事：

**Bottleneck 是 long-horizon 的命脉。** 4 个 context tokens 的设计，去掉它 long-horizon 从 88.6 崩到 51.2。这个 gap 比所有其他因素加起来都大。

直觉上，长程任务对 context length 极度敏感——attention 在长序列上会累积 spurious correlation。Bottleneck 强制 representation 只保留 task-relevant 信息，相当于一个 supervised 的 information bottleneck。这在 data 少的时候是 regularizer，在 data 多的时候可能变成限制。ProgVLA 恰好处于 data 少的 regime，所以 bottleneck 是 pure win。

如果你想follow up，最值得做的实验：
1. 把 progress target 换成 semantic subgoal（用 VLM 打标签），看 advantage 是否变成真正 informative 的 signal
2. 加入 failed demos，让 Q 真正学到 action quality
3. 做 cross-environment real-world eval，看 small-model + strong-bias 在 distribution shift 下是否还 robust

这三个里任何一个，都比当前 paper 的 in-distribution success rate 更有 scientific value。

---

# ProgVLA 深度解读 — 给 Karpathy 的直觉向分析

Andrej, 这篇paper的核心赌注其实蛮有意思: **在 small-model regime, 与其靠 scale 和 cross-embodiment robot pretraining, 不如做精细的 architectural engineering + temporal supervision**. 0.1B 参数, 完全 from-scratch 在 2000 条 LIBERO demos 上训, 长horizon 打赢了 2.25B 的 SmolVLA 和 7B 的 OpenVLA. 这挑战了 "scale is all you need" 的naive版本, 值得拆开看为什么 work.

---

## 1. Big picture: paper 在对抗什么

当前 VLA 主流 ([RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246), [π0](https://arxiv.org/abs/2410.24164), [π0.5](https://physical-intelligence.medium.com/)) 路线是: 大 VL backbone + 大规模 robot pretraining (Open X-Embodiment, DROID). [SmolVLA](https://arxiv.org/abs/2506.08450) 已经把参数压到 450M 级别, 但仍依赖 cross-embodiment pretraining, 在 long-horizon 上掉得厉害 (LIBERO Long 只有 63-77%).

ProgVLA 想问: **如果只给一个 benchmark 的 demos (2000条), 一个 0.1B 模型能不能达到甚至超过那些巨无霸?** 答案是能, 但需要三个设计杠杆同时拉满:

| 杠杆 | 作用 | Ablation 贡献 |
|---|---|---|
| Two-stage Perceiver resampling | 序列长度压缩, 把 heterogeneous modality 压成 control-ready tokens | -16.0 SR (最大) |
| Fine-tune DUNE vision backbone | 让通用 visual prior 适应 wrist-camera 视角 | -13.5 SR (第二) |
| Progress heads (auxiliary) | Temporal reweighting + representation regularizer | -2.3 SR (第三, 但在 Long 上集中) |

这个排序很关键 — 见 [Table 2 ablation](#). 单看 progress heads 只贡献 2.3 个点, 但它们是 paper 的标题卖点, 因为它们代表了"用 offline RL 形式做 supervised reweighting"这种 hybrid 设计哲学.

---

## 2. Architecture walk-through: 四阶段 multimodal encoder

整体 information flow 是一个 **漏斗 + 漏斗** 的双瓶颈结构. 这是这篇 paper 最有工程价值的地方.

### 2.1 Stage 1: Modality-specific encoders

- **Vision**: [DUNE](https://arxiv.org/abs/2504.13136) ViT-S (CVPR 2025), 输入图像 resize 到 112×112. DUNE 是从多个 specialist vision models (包括 2D 和 3D/depth-style tasks) 蒸馏出来的 universal encoder. 相比 [DINOv2/v3](https://arxiv.org/abs/2304.07193) 纯 contrastive SSL, DUNE 多了 geometric/depth prior, 这点在 ablation 里体现得很明显: DINOv3 替换 DUNE 掉 2.4 个点, 其中 Long-horizon 掉 7.6 个点 — manipulation 任务需要 geometric 理解.
- **Language**: frozen [T5](https://arxiv.org/abs/1910.10683) text encoder. 不训, 只当 feature extractor.
- **Proprioception** q_t (joint angles, velocities, gripper status): 一个 MLP 投影到 d_model=384 共享空间.

直觉上, 这一步是 "每个模态用自己的语言先说出话来", 但每个模态输出的 token 数量天差地别: ViT-S 在 112×112 + patch 16 大概产生 ~50 patch tokens, T5 可能产生几十个 language tokens, proprioception 只有 1 个 token.

### 2.2 Stage 2: Per-modality Perceiver Resamplers

用 [Perceiver Resampler](https://arxiv.org/abs/2204.14198) (Flamingo 那个) 把每个模态压到固定数量的 latent tokens:
- Vision: 每个 camera → 8 tokens
- Language: → 16 tokens
- (proprioception 直接进 fusion)

Perceiver Resampler 的核心是: 一组 learnable latent queries $Q_{lat} \in \mathbb{R}^{K \times d}$ 通过 cross-attention 反复读 input tokens. 2 个 cross-attention blocks, 8 heads, d_model=384, MLP ratio=4, dropout=0.1.

公式上 (Perceiver Resampler):
$$\text{output}_k = \text{CrossAttn}(Q_{lat,k}, K=\text{input}, V=\text{input})$$
经过 L 层 stack. 这里 K=8 (vision) 或 K=16 (language).

**为什么先 per-modality 压一遍, 而不是直接 fusion?** Appendix B.2 给了诚实的理由: 单一 bottleneck 会让 capacity 不公平分配 — vision 几百个 patch tokens 会 dominate attention, 把 language 和 proprioception 淹掉. 这跟 [Perceiver IO](https://arxiv.org/abs/2107.12395) 原始 paper 的 motivation 类似, 但 ProgVLA 把它分两层, 让 "平衡模态" 和 "准备 control tokens" 这两个 function 解耦.

### 2.3 Stage 3: Fusion Transformer

把所有模态的 tokens concat 起来, 过一个 2-layer Pre-LN Transformer encoder (d_model=384, 8 heads, MLP ratio 4, dropout 0.1). 这里做 cross-modal self-attention, 让 vision 和 language tokens 互相 attend, 实现 grounding.

直觉: "把红色杯子放到水槽里" 这条 instruction 的 'red cup' token 需要 attend 到 vision 里对应物体的 patch tokens. 这一步是真正的 cross-modal binding.

### 2.4 Stage 4: Post-fusion Perceiver Resampler (最关键!)

第二个 Perceiver Resampler, 把 fusion 后的全部 tokens 蒸馏成 **4 个 context tokens** $c_t \in \mathbb{R}^{4 \times 384}$.

这 4 个 tokens 就是整个 policy 和 progress heads 看到的全部 "世界状态". Ablation 显示去掉这一步, SR 从 91.1 跌到 75.1, Long-horizon 直接从 88.6 崩到 51.2.

**直觉**: 这 4 个 tokens 是一个极度 lossy 但极度 task-relevant 的 bottleneck. 类似于 VAE 的 latent, 但是 supervised 学的, 不是 generative 学的. 它强迫 fusion Transformer 把所有 grounding 信息压进极小空间, 这反而让下游 action expert 和 progress heads 看到的 context 高度结构化, 不被噪声 patch 淹没.

---

## 3. Flow-matching action expert

这部分直接复用 [SmolVLA](https://arxiv.org/abs/2506.08450) 的设计, 没有架构创新, 但值得回顾公式.

### 3.1 训练目标

给定 context $c_t$ 和 ground-truth action chunk $\mathbf{A}_t = [a_t, ..., a_{t+H-1}]$ ($H=16$):

- 采样噪声 $\epsilon = \mathbf{y}_0 \sim \mathcal{N}(0, I)$
- 采样 flow time $\tau \sim \text{Beta}(\alpha=2, \beta=2)$ (注意, 不是 uniform, 这是 SmolVLA 的 trick, 让中间步采样更多)
- 构造 interpolant: $\mathbf{y}_\tau = (1-\tau)\mathbf{y}_0 + \tau \mathbf{y}_1$, 其中 $\mathbf{y}_1 = \mathbf{A}$ (data)
- 训练网络预测 velocity field:

$$\mathcal{L}_{FM} = \mathbb{E}_{\tau, \epsilon} \left[ \| \pi_\theta(\mathbf{y}_\tau; \tau, \mathbf{c}) - (\mathbf{y}_1 - \epsilon) \|_2^2 \right] \tag{1}$$

变量说明:
- $\mathbf{y}_\tau$: 在 noise ($\mathbf{y}_0$) 和 data ($\mathbf{y}_1$) 之间的线性插值点
- $\tau \in [0,1]$: flow time, 0=纯噪声, 1=纯 data
- $\pi_\theta(\mathbf{y}_\tau; \tau, \mathbf{c})$: 网络预测的 velocity, 把 $\mathbf{y}_\tau$ 推向 $\mathbf{y}_1$
- $(\mathbf{y}_1 - \epsilon)$: ground-truth velocity, 因为 $\mathbf{y}_1 - \mathbf{y}_0 = \mathbf{y}_1 - \epsilon$
- $\mathbf{c}$: 4 个 context tokens (条件)

直觉上, 这是在学一个 ODE: $\frac{d\mathbf{y}}{d\tau} = \pi_\theta(\mathbf{y}_\tau; \tau, \mathbf{c})$, 从 noise 积分到 data. 跟 [diffusion policy](https://arxiv.org/abs/2303.04137) 思路类似但更干净 (flow matching 是 deterministic ODE, diffusion 是 SDE; flow matching 的 velocity 是直线, diffusion 是 score).

### 3.2 推理

从 $\mathbf{y}_0 \sim \mathcal{N}(0, I)$ 出发, 用 **10 步 Heun (explicit trapezoidal)** 积分. Heun 比 Euler 多一次 corrector step, 精度更高, 在同样 step count 下基本免费. 执行 chunk 的前 8 步 (execution horizon $H_e=8$), 然后 replan.

### 3.3 Action expert 架构

12 blocks, 交替:
- Cross-attention to 4 个 post-fusion context tokens
- Self-attention over action sequence (16 actions × action_dim)

d_model=384, 8 heads, MLP ratio 4. Time embedding $\tau$ 用 sinusoidal continuous embedding, 通过 **AdaLN-Zero modulation** 注入. AdaLN-Zero 是 [DiT](https://arxiv.org/abs/2212.09748) 的 trick — scale 和 shift 都用 $\tau$ 条件化, 而且 zero-init, 训练初期等价于 identity, 稳定优化.

---

## 4. Progress heads — paper 的核心创新

这是最值得细读的部分. 名义上借用 offline RL 的 Q/V/S 结构, 实际上是一个 hybrid: 用 offline RL 的 *形式* + supervised learning 的 *实质*.

### 4.1 Progress target (纯 temporal shaping)

$$r_t = \max\left(0, 1 - \frac{T_n - t}{T_M}\right), \quad T_M = 500 \tag{2}$$

变量:
- $T_n$: episode $n$ 的终止时间 (假设是 success state)
- $t$: 当前时间步
- $T_M$: 固定 horizon cap, 500
- $r_t \in [0, 1]$: shaped progress signal

直觉: 这是一个 **纯粹基于 trajectory phase 的 open-loop 信号**. 离终点越近, $r_t$ 越大. 它不来自 subgoal detector, 不来自 environment reward, 不来自 semantic segmentation — 完全是 $(T_n - t)$ 的函数.

作者在 Sec 5 和 Appendix B.8 非常诚实地承认: "progress-aware" 这里指 *training objective*, 不指 *verified internal progress estimator*. 他们没有做 calibration 验证, 没做 monotonicity 分析, 没有 sweep $T_M$.

### 4.2 三个 head 的架构

3 个 head 共享同样的 4 个 context tokens $c_t$ 作为输入, 但分两个 trunk:

- **Shared value trunk** (for $\hat{V}$ and $\hat{S}$): Perceiver pooler 把 4 tokens pool 成 1 个, 然后分两个 scalar head
- **Q trunk** (for $\hat{Q}$): 单独的 Perceiver pooler, 把 $a_t$ 通过 MLP 投成 1 个 token, concat 到 $c_t$ 上, 然后 pool

为什么 $\hat{Q}$ 单独 trunk? Appendix B.2 解释: 不想让 action conditioning leak 进 $\hat{V}$ 和 $\hat{S}$ 的输入, 否则 advantage 就没意义了.

### 4.3 Q loss (Monte-Carlo, 不 bootstrap)

$$\mathcal{L}_Q = \mathbb{E}_{(c_t, a_t, r_t) \sim B} \left[ \text{Huber}(\hat{Q}(c_t, a_t) - r_t) \right] \tag{3}$$

关键设计选择: **不用 TD target $r + \gamma \hat{V}(c')$, 直接回归 Monte-Carlo return-to-go $r_t$**.

为什么? 标准 offline RL ([CQL](https://arxiv.org/abs/2006.04779), [IQL](https://arxiv.org/abs/2110.06169), [BEAR](https://arxiv.org/abs/1911.11033)) 都用 Bellman backup, 会有 OOD action extrapolation 问题. ProgVLA 只用 successful demos, 每个 context 只对应一个 logged action, 没必要 bootstrap. 用 Huber (smooth-L1) 比 L2 抗 outlier, 在 long-horizon 轨迹里噪声大.

直觉: 这不是真正的 Q-learning, 是一个 "state-action conditioned progress regressor". $\hat{Q}(c_t, a_t)$ 学的是 "在这个 context 下, 走这个 action, 还剩多远到终点".

### 4.4 V loss (IQL-style expectile regression)

$$\mathcal{L}_V = \mathbb{E}_{(c_t, r_t) \sim B} \left[ \rho(r_t - \hat{V}(c_t))_+^2 + (1-\rho)(r_t - \hat{V}(c_t))_-^2 \right] \tag{4}$$

变量:
- $(\cdot)_+$: positive part, $\max(\cdot, 0)$
- $(\cdot)_-$: negative part, $\min(\cdot, 0)$
- $\rho = 0.8$: expectile parameter, 偏向 high residuals

这是 [IQL](https://arxiv.org/abs/2110.06169) 的核心 trick, 但 ProgVLA 把它 apply 到 Monte-Carlo $r_t$ 而不是 bootstrapped Bellman target.

直觉: 普通 MSE 让 $\hat{V}$ 拟合 $r_t$ 的均值. Expectile regression with $\rho > 0.5$ 让 $\hat{V}$ 拟合 $r_t$ 的 **optimistic envelope** — 也就是在同一个 context 下, 假设采取 *比平均更好* 的 action, return 会是多少. 这是一个 in-dataset 的乐观 baseline.

为什么需要这个? 因为 advantage $A_t = \hat{Q} - \hat{V}$ 需要对 *好 action* 为正, 对 *差 action* 为负. 如果 $\hat{V}$ 是 mean, $\hat{Q}$ 对所有 in-dataset action 都学同一个值 (因为只有一个 logged action), advantage 就退化. Expectile bias 让 $\hat{V}$ 比均值高, 于是 logged action 的 $\hat{Q}$ (它确实在那条成功轨迹里) 相对 $\hat{V}$ 形成有意义的差.

但作者也诚实承认 (Appendix B.1): 这种 advantage 在严格 offline RL 意义下 **weakly identified**, 更自然的解释是 "trajectory-phase reweighting term". 因为 successful demo 里每个 context 只有一个 action, $\hat{Q}$ 和 $\hat{V}$ 学的几乎是同一个东西的两种不同 smoothing.

### 4.5 Success classification loss

$$\mathcal{L}_S = \mathbb{E} \left[ \text{BCEWithLogits}(\hat{S}(c_t), y_t^{succ}) \right] \tag{6}$$
$$y_t^{succ} = \mathbb{I}\{r_t \geq r_{succ}\}, \quad r_{succ} = 1 - 17/T_M \approx 0.966 \tag{7}$$

只让 $\hat{S}$ 在 trajectory 的最后 17 步 fire. 这是个 binary near-completion detector.

为什么 17 步? 大概对应一个 grasp-and-place 的 critical phase. 没解释, 应该是 empirical tuned.

### 4.6 Reweighting (关键的一步)

把 advantage 和 success prob 转成 detached per-sample weights:

$$A_t = \hat{Q}(c_t, a_t) - \hat{V}(c_t), \quad \text{detached}$$
$$w_{A,t} = \min\{\exp(A_t / \beta), C\}, \quad \beta = 0.5, C = 20 \tag{5}$$
$$w_{S,t} = 0.5 + 0.5 \cdot \sigma(\hat{S}(c_t))$$
$$w_{S,t} \in [0.5, 1.0]$$

注意 $w_{S,t}$ 的设计很温和 — 最小 0.5, 最大 1.0, 永远不会 zero out 任何 sample. 这是 "soft upweight close-to-completion frames" 而不是 hard filtering.

### 4.7 Total loss

$$\mathcal{L}_{total} = \mathbb{E} \left[ w_{A,t} \cdot w_{S,t} \cdot \ell_{FM}(c_t, a_t) \right] + \lambda_V \mathcal{L}_V + \lambda_Q \mathcal{L}_Q + \lambda_S \mathcal{L}_S \tag{8}$$

权重: $\lambda_V = 0.5, \lambda_Q = 1.0, \lambda_S = 0.2$.

**最关键的设计: $w_{A,t}$ 和 $w_{S,t}$ 都 detached from autograd graph**. 它们只 *reweight* imitation loss, 不传 gradient 回 Q/V/S heads. 所以 Q/V/S 的训练完全靠 $\mathcal{L}_Q, \mathcal{L}_V, \mathcal{L}_S$ 自己, 跟 imitation loss 解耦.

直觉: 这避免了 reward shaping 中常见的 feedback loop — 如果 advantage 同时驱动 policy 和被 policy 影响, 训练会不稳定. ProgVLA 让 progress heads 当 *pure auxiliary supervisor* + *fixed per-sample weighter*, 不参与 policy gradient.

---

## 5. 为什么 progress heads 帮 long-horizon? — 直觉解释

Appendix B.1 给了 qualitative 解释, 我觉得这是 paper 最 insight-dense 的部分:

1. **Critical decisions 在 trajectory 末尾聚集**. Long-horizon 任务 (LIBERO Long, Meta-World hard/very-hard) 的 fragile 时刻往往是: 最后的 precise grasp, 最终的 placement, sub-task 之间的 transition. 这些时刻一个小 action error 就毁掉整个 episode.

2. **Multiplicative weight 把 gradient capacity 推向末尾**. 因为 $r_t$ 单调递增, $\hat{V}, \hat{Q}$ 在末尾更大, $A_t$ 也更大, $w_{A,t}$ 更大; $w_{S,t}$ 在末尾接近 1.0, 在开头接近 0.5. 乘起来, 末尾 frame 的 imitation loss 权重比开头可能大 5-10 倍.

3. **Expectile bias 在 phase change 处 sharpen advantage**. 比如 approach → grasp 的过渡时刻, return-to-go 分布是非对称的 (有些 demo 抓得快, 有些慢), expectile 把 $\hat{V}$ 推到 fast 那端, 于是慢的 demo 在这个 context 下 advantage 小, 快的 advantage 大. 这相当于 *prefer fast progress*.

4. **Success head 把末端 17 步特别标出**. 这 17 步的 imitation loss 被 $w_{S,t} \approx 1.0$ upweight, 其他 frame 被 0.5 downweight.

定性上, 这跟 [prioritized experience replay](https://arxiv.org/abs/1511.05952) 和 [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) 的精神相通 — 用某种 signal 优先学 critical transitions. 但 ProgVLA 的 signal 是 *temporal*, 不是 *TD-error* 或 *goal-achievement*.

---

## 6. Experimental results — 数字解读

### 6.1 [LIBERO](https://arxiv.org/abs/2306.03310) (Table 1, top)

| Model | Params | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| ProgVLA | 0.1B | 87.6 | 96.0 | 92.0 | 88.6 | **91.1** |
| [Octo](https://arxiv.org/abs/2405.12213) | 0.09B | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| π0 (PaliGemma-3B) | 3B | 87 | 63 | 89 | 48 | 71.8 |
| π0 (3.3B) | 3.3B | 90 | 86 | 95 | 73 | 86.0 |
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SmolVLA | 0.24B | 87 | 93 | 88 | 63 | 82.75 |
| SmolVLA | 2.25B | 93 | 94 | 91 | 77 | 88.75 |

关键 gap:
- ProgVLA 0.1B vs OpenVLA 7B: **+14.6 overall, +34.9 on Long**
- ProgVLA 0.1B vs SmolVLA 2.25B: **+2.4 overall, +11.6 on Long**

Long-horizon 的 gap 是最 informative 的, 因为这正好是 cross-embodiment pretraining 都救不了的场景 — 长程需要 temporal coherence, 而不是更多的 visual-language prior.

### 6.2 [Meta-World](https://arxiv.org/abs/1910.10897) MT50 (Table 1, bottom)

| Model | Easy | Medium | Hard | Very hard | Avg |
|---|---|---|---|---|---|
| ProgVLA 0.1B | 84.9 | 72.7 | 77.0 | 79.6 | **78.5** |
| π0 PaliGemma-3B | 80.4 | 40.9 | 36.7 | 44.0 | 50.5 |
| π0 3.3B | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA 0.24B | 86.43 | 46.36 | 35 | 60 | 56.95 |
| SmolVLA 2.25B | 87.14 | 51.82 | 70 | 64 | 68.24 |

ProgVLA 在 easy 上略输 SmolVLA 2.25B (-2.2), 但 medium +20.9, hard +7.0, very hard +15.6. **越难的任务 gap 越大**. 这跟 LIBERO Long-horizon 的 pattern 一致, 都指向 "progress heads + bottleneck architecture 在 long-horizon/difficult 上 scale 得好".

### 6.3 Ablation (Table 2)

| Variant | Spatial | Object | Goal | Long | Avg | Δ |
|---|---|---|---|---|---|---|
| Full ProgVLA | 87.6 | 96.0 | 92.0 | 88.6 | 91.1 | — |
| w/o progress | 87.0 | 90.6 | 90.2 | 85.1 | 88.8 | -2.3 |
| w/o context resampler | 84.4 | 77.2 | 87.4 | 51.2 | 75.1 | **-16.0** |
| DINOv3 backbone | 88.2 | 92.4 | 93.2 | 81.0 | 88.7 | -2.4 |
| frozen DUNE | 79.0 | 85.2 | 85.4 | 60.6 | 77.6 | -13.5 |

观察:
- **w/o context resampler**: Long 从 88.6 暴跌到 51.2. 这说明 4-token bottleneck 是 long-horizon 的命脉. 没有它, action expert 要直接 attend 上百个 tokens, 在长程下 spurious correlation 累积.
- **frozen DUNE**: Long 从 88.6 跌到 60.6. 说明 wrist-camera 视角太特殊, 通用 backbone 必须适配. 
- **DINOv3 vs DUNE**: 总体差距小 (-2.4), 但 Long 上掉 7.6. DUNE 的 multi-teacher distillation (含 depth-style targets) 比 DINOv3 的纯 contrastive 更适合 manipulation.
- **w/o progress**: 影响最小但集中在 Object (-5.4) 和 Long (-3.5). 验证了 "progress heads 帮 multi-object 和 long-horizon" 的假设.

---

## 7. Real-world 实验

6-DOF PiPER arm + RealSense D405 (wrist) + D435 (agent). 10 个 task, 每个 50 demos, 同样 hyperparameters 训 8.5 小时 on H100. 在便携笔记本 (Intel Ultra 7 + RTX 3500) 上实时跑.

100 trials, 平均 68% SR. 失败模式 (Table 4 + Appendix B.6):
- 10 次场景杂物遮挡 (clutter obstruction)
- 8 次 gripper-opening timeout
- 4 次抓错物体
- 其他零散

注意这是 **in-distribution evaluation** — train 和 eval 用同样的两个环境 (toy kitchen + white-table). 不测试 cross-environment generalization. 作者诚实承认这一点.

直觉上 68% 听起来不算高, 但考虑到: (a) 0.1B 模型, (b) from scratch (没用 Open X-Embodiment), (c) 只 50 demos/task, (d) 在笔记本上跑 — 这个 baseline 已经很有价值了. 主要失败模式 (clutter, timeout) 都不是 architecture-specific 的, 是 robotic deployment 的共性问题.

---

## 8. 跟 related work 的关系 — 一些联想

### 8.1 跟 self-monitoring 工作的区别

[SuccessVQA](https://arxiv.org/abs/2312.02462), [Luo et al.](https://arxiv.org/abs/2402.07874), [Video-Language Critic](https://arxiv.org/abs/2402.18213), [Generative Value Learning](https://arxiv.org/abs/2402.06476) 都把 progress/success estimation 当 **post-hoc external module**: 训一个独立 VLM 评估 trajectory, 然后用于 reward shaping 或 evaluation. ProgVLA 的差异是: progress heads **share context tokens with policy**, **trained jointly**, **feed scalar predictions back as loss weights**. 部署时 progress heads 可以扔掉, 只留 policy.

这跟 [value-based RL](https://arxiv.org/abs/2110.06169) 里 "critic 只在训练时用, 部署时只要 actor" 的 philosophy 一致, 但用在 imitation learning 上.

### 8.2 跟 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 和 [π0](https://arxiv.org/abs/2410.24164) 的关系

Flow matching 是 diffusion 的表亲 — 同样从 noise sample 到 data, 但用 deterministic ODE 而不是 SDE, 训练目标更简洁 (predict velocity 而不是 score). π0 把这个引入 VLA, SmolVLA 把它小型化, ProgVLA 沿用 SmolVLA 的 action expert 但加上 progress reweighting.

直觉上, flow matching + multiplicative weight 类似于 **conditional flow matching with non-uniform sampling** — progress heads 决定哪些 $(c_t, a_t)$ pair 在 mini-batch 里 "权重更高", 间接影响哪个 $\tau$ 的 velocity 学得更准.

### 8.3 跟 [IQL](https://arxiv.org/abs/2110.06169) 的关系

IQL 的核心是 expectile regression 避免 OOD action extrapolation. ProgVLA 借了这个 trick 但用在 Monte-Carlo return 而不是 Bellman target. 这相当于 "IQL without bootstrapping", 一个 *strictly simpler* 的变种. 可以理解成: 当你的 dataset 全是 successful demos, bootstrapping 没什么可 bootstrap 的, 直接 MC return 就够.

### 8.4 跟 [DUNE](https://arxiv.org/abs/2504.13136) / [DINOv3](https://arxiv.org/abs/2506.14545) 的对比

DUNE 是 multi-teacher distillation, 从多个 specialist (含 2D 和 3D tasks) 蒸馏出 universal encoder. DINOv3 是纯 self-supervised contrastive. 在 manipulation 上 DUNE 略胜, 主要在 Long-horizon — 暗示 geometric/depth prior 对 long-horizon planning 更有用, 因为长程需要更稳定的空间理解.

### 8.5 跟 VLA scaling law 的关系

[RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246), [π0](https://arxiv.org/abs/2410.24164) 走的是 "scale backbone + scale robot data" 路线. SmolVLA 已经质疑过这条: 450M 也能打 10x 大的. ProgVLA 把这个 questioning 推得更极端: 100M, 不用 robot pretraining, 也能打. 

直觉: 在 robot manipulation 这种 *task-distribution narrow* 的 setting, 大规模 pretraining 的 marginal value 在递减 — Open X-Embodiment 里的大部分 data 跟 LIBERO 的具体 task 没直接关系. 真正起作用的是 **任务相关的 inductive bias** (bottleneck, temporal supervision, 适配过的 vision). 这是 small-data + strong-bias 的胜利.

但要注意 caveat: ProgVLA 的 evaluation 都是 in-distribution (LIBERO/Meta-World 任务 family). 真正的 generalization test (held-out task, held-out environment) 还没做. SmolVLA/OpenVLA 的 pretraining 优势可能在那种 setting 才显现.

---

## 9. 局限性 — 作者自己承认的 (很诚实)

读 Appendix B.8 我对作者的诚实度评价很高. 几个关键 caveat:

1. **Progress target 是纯 temporal, 不是 semantic**. $r_t$ 只看 trajectory phase, 不看 subgoal. 一个长 episode 的中间 phase 和短 episode 的末尾 phase 可能 $r_t$ 一样. 作者明说 "progress-aware 这里指 training objective, 不指 verified internal progress estimator".

2. **没验证 progress heads 的 calibration**. 没做 $p_{succ}$ 的 calibration curve, 没做 $\hat{V}$ 的 monotonicity 分析, 没做 success/failure rollout 的 separation 验证. 这是 follow-up 的 low-hanging fruit.

3. **Advantage 在 offline RL 意义下 weakly identified**. 因为只用 successful demos, 每个 context 只配一个 logged action, $\hat{Q}(c_t, a_t) - \hat{V}(c_t)$ 严格说不是 action-quality signal, 是 trajectory-phase reweighting.

4. **单 seed, 没 sweep hyperparameters**. $T_M, \rho, \beta, C$ 都没扫. Run-to-run variance ±1-2 SR.

5. **Baseline protocol 不严格 matched**. SmolVLA baseline 来自另一个 LIBERO release (filtered 1693 demos + higher-res images), ProgVLA 用原版 2000 demos + 低分辨率. Image resolution 偏向 SmolVLA, demo count 偏向 ProgVLA, 但作者论证 gap 远超这些 confounders 能解释的范围.

6. **Real-world 太窄**. 单 arm, 10 task, in-distribution. 不测 cross-platform 或 cross-environment.

7. **没 efficiency benchmark**. 只说 "在笔记本上实时跑", 没测 latency / FLOPs / peak memory.

---

## 10. 一些可能延伸的联想

- **Progress target 换成 semantic subgoal**: paper 留了这个 future direction. 想象用 [VLM as progress estimator](https://arxiv.org/abs/2402.06476) 给每帧打 subgoal achievement score, 替换 $r_t$. 这会让 advantage 真正 informative 而不是 trajectory-phase reweighting.

- **Failed trajectory 利用**: 当前只用 successful demos. 如果加入 failed demos, $\hat{Q}$ 可以学到 "这个 action 会导致失败", advantage 就真正变成 action-quality signal. 这是 [AWAC](https://arxiv.org/abs/2006.09359) 或 [IQL](https://arxiv.org/abs/2110.06169) 的标准 setup.

- **On-policy refinement**: progress heads 可以做 actor-critic 式的 on-policy update. 但这违反了 paper "保持 supervised flow matching 简洁性" 的设计哲学.

- **Bottleneck 的信息论分析**: 4 个 context tokens 是极度 aggressive 的压缩. 信息瓶颈 (information bottleneck) 视角下, 这强迫 representation 只保留 task-relevant 信息, 丢弃 task-irrelevant visual detail. 跟 [VAE](https://arxiv.org/abs/1312.6114) 和 [SIMCLR](https://arxiv.org/abs/2002.05709) 的 contrastive bottleneck 精神相通, 但 supervised. 是否存在最优 token 数? paper 没扫这个.

- **Flow matching + reweighting 的理论联系**: $w_{A,t} w_{S,t} \cdot \ell_{FM}$ 等价于在 weighted empirical distribution 上做 flow matching. 这跟 [importance sampling](https://arxiv.org/abs/1804.09077) 在 diffusion 中的应用, [classifier-free guidance](https://arxiv.org/abs/2207.12598) 的 conditioning, 都有形式上的联系 — 都是 *modify the training distribution to bias generation toward desired regions*.

- **DUNE vs DINOv3 的 ablation 暗示**: 如果 manipulation 真需要 depth/geometric prior, 那 *distill from depth-predictor teachers* 是一个 cheap substitute for *real depth input*. 这跟 [Depth Anything](https://arxiv.org/abs/2401.10891) 路线呼应 — 用 distilled depth prior 替代 explicit depth sensor.

- **Two-stage resampling 的 generalization**: 这个 pattern (per-modality normalize → fuse → post-fusion bottleneck) 可以推广到任何 multi-modal sequence 模型. 比如 video-language model, audio-text model. 跟 [Perceiver IO](https://arxiv.org/abs/2107.12395) 和 [Flamingo](https://arxiv.org/abs/2204.14198) 的 single-stage bottleneck 比, 多了一层 "balance before fuse" 的解耦.

- **LIBERO Long 88.6 vs 51.2 的 -37.4 暴跌**: 这是 paper 最戏剧性的 ablation 数字. 暗示 long-horizon 对 context length 极度敏感 — attention 在长程下累积 spurious correlation, bottleneck 强制 attention focus 在 distilled task-relevant tokens. 这跟 [Transformer context length scaling](https://arxiv.org/abs/2205.01089) 的失败模式理论一致.

---

## 11. 给 Karpathy 的 intuition 总结

如果让我用一句话概括 ProgVLA: **"在 small-data small-model regime, 用 information bottleneck 和 temporal reweighting 替代大规模 pretraining"**.

更细的 intuition:

1. **Bottleneck 是 representation regularizer**. 4-token post-fusion resampler 强迫 fusion Transformer 学会 *extractive* representation, 而不是 *expansive*. 这在 data-scarce setting 下避免 overfitting 到 patch-level noise.

2. **Vision fine-tune > vision pretrain**. 通用 backbone 给 prior, 但 manipulation 的 wrist-camera 视角太特殊, 必须在任务上 adapt. 这跟 [ViT fine-tune protocol](https://arxiv.org/abs/2106.10270) 的发现一致, 但在 robot setting 下放大了.

3. **Progress heads 是 prioritized supervised learning 的 disguise**. 名义上 offline RL, 实质是 prioritized sampling + auxiliary representation supervision. 诚实地说, paper 的 advantage 没有真正的 RL 内容 — 它是一个 detached scalar reweighting. 但这个 *伪装* 让它跟 imitation learning 文献区别开来, 也确实 long-horizon 上 work.

4. **Critical transition upweighting** 是 long-horizon 的关键. 长程任务的成败往往决定于几个 critical frames (grasp, place, transition). Multiplicative weight 把 gradient capacity 推向这些 frames. 这跟 [HMM 的 Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) 在 critical state 上投入更多计算, [AlphaZero 的 MCTS](https://arxiv.org/abs/1712.01815) 在 critical position 上 expand 更多, 都是 *concentrate compute where it matters* 的精神.

5. **诚实的 paper > 漂亮的 paper**. 作者反复承认 limitation: temporal target 不是 semantic progress, advantage weakly identified, 单 seed, 不严格的 baseline matching, real-world 太窄. 这种诚实让 reader 能 trust ablation 的定性结论, 即使绝对数字可能因 protocol 差异有所偏移.

---

## Web links 参考

- [ProgVLA (this paper, NAVER LABS)](https://www.naverrlabs.com/) — 暂用 NAVER LABS 官网, arXiv 链接待发布
- [SmolVLA](https://arxiv.org/abs/2506.08450) — HuggingFace Matic, 流匹配 action expert 来源
- [π0 — Physical Intelligence](https://arxiv.org/abs/2410.24164) — Flow matching VLA 开创
- [π0.5](https://physical-intelligence.medium.com/) — Open-world VLA extension
- [OpenVLA](https://arxiv.org/abs/2406.09246) — 7B 开源 VLA baseline
- [RT-2](https://arxiv.org/abs/2307.15818) — Google 的 VLA 早期工作
- [Octo](https://arxiv.org/abs/2405.12213) — 0.09B 通用 robot policy
- [Perceiver](https://arxiv.org/abs/2103.03206) — 原始 Perceiver
- [Perceiver IO](https://arxiv.org/abs/2107.12395) — Perceiver IO 扩展
- [Flamingo (Perceiver Resampler)](https://arxiv.org/abs/2204.14198) — Resampler 架构来源
- [IQL](https://arxiv.org/abs/2110.06169) — Expectile regression trick
- [DUNE](https://arxiv.org/abs/2504.13136) — Multi-teacher distilled vision encoder
- [DINOv2](https://arxiv.org/abs/2304.07193) — Self-supervised vision baseline
- [DINOv3](https://arxiv.org/abs/2506.14545) — Updated DINO
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310) — Lifelong robot learning benchmark
- [Meta-World](https://arxiv.org/abs/1910.10897) — MT50 multi-task benchmark
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) — Diffusion-based action generation
- [Flow Matching for generative modeling](https://arxiv.org/abs/2210.02747) — Flow matching 理论基础
- [T5](https://arxiv.org/abs/1910.10683) — 文本 encoder
- [DiT (AdaLN-Zero)](https://arxiv.org/abs/2212.09748) — AdaLN-Zero modulation
- [Generative Value Learning (Ma et al.)](https://arxiv.org/abs/2402.06476) — VLM as value learner
- [Video-Language Critic](https://arxiv.org/abs/2402.18213) — Transferable reward functions
- [SuccessVQA](https://arxiv.org/abs/2312.02462) — VLM as success detector
- [Depth Anything](https://arxiv.org/abs/2401.10891) — Distilled depth prior
- [CQL](https://arxiv.org/abs/2006.04779) — Conservative Q-learning
- [AWAC](https://arxiv.org/abs/2006.09359) — Advantage-weighted actor-critic
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — TD-error prioritization
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) — Conditional diffusion guidance
- [AlphaZero](https://arxiv.org/abs/1712.01815) — MCTS compute concentration

---

## 12. 一句话直觉

**ProgVLA 是一个 evidence that, 在 robot manipulation 这种 narrow-distribution setting, "bottleneck architecture + temporal reweighting + adapted vision" 这三个 strong inductive bias 可以 substitute for "10x more parameters + 1000x more robot data"**. 它没否定 scaling, 但指出 scaling 在 *narrow task family* 上的 marginal utility 可能比想象的低, 而 careful architecture 在 *small-data regime* 上的 marginal utility 可能比想象的高. 长程上的 +11.6 / +34.9 gap 是这个 thesis 的最强 evidence — 因为 long-horizon 正好是 pretraining 最难救援的场景, 也是 temporal supervision 最有效的场景.

但作为一个 *scientific claim*, 它还差几步: 严格的 baseline-matched comparison, multi-seed evaluation, progress head 的 calibration 验证, cross-environment real-world generalization. 当前 paper 是一个 strong *existence proof* (这样训能 work), 还不是 strong *scientific explanation* (为什么 work 的机理是什么).
