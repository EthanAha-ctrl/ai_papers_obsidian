---
source_pdf: EvoScene-VLA Evolving Scene Beliefs Inside the.pdf
paper_sha256: c0a96d5ab9c5d9e424dfa5d8d99fee241ba2d3bcea99df21933b83dbc91325e9
processed_at: '2026-08-18T11:46:48-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EvoScene-VLA

## 先说这 paper 在 solve 什么问题

想象你在用一个 chunked VLA policy——就是它看一眼场景，然后一口气预测未来 50 步 action。听起来挺高效，但有个要命的 bug：**这 50 步里的每一步，它脑子里想的全是 50 步前那个场景**。

举个例子（paper Fig.1 的 Open Microwave）：robot 看到 microwave 门，开始规划整个 chunk——伸手、抓 handle、拉门。但 chunk 执行到一半，gripper 已经 retract 了，根本没够到 handle。为什么？因为整个 chunk 是基于"门在初始位置"规划的，robot 自己的动作改变了场景状态（手伸过去挡住了视线、或者 kinematic 上已经偏离），但 policy 不知道，它还在追那个 stale target。

这就是 paper 标题里 "Evolving Scene Beliefs" 想解决的：**scene 在 action 下演化，但 policy 的 belief 不演化**。

---

## 为什么现有的 fix 都不够

现有工作大致三类，每类都只解决一半问题：

**Spatial VLA**（[SpatialVLA](https://arxiv.org/abs/2501.15830), [QDepth-VLA](https://arxiv.org/abs/2510.14836), [Spatial Forcing](https://arxiv.org/abs/2510.12276)）让 VLM 对单帧 image 的 geometry 理解更好——depth supervision、3D encoding 之类。但它们只看当前一帧，action 一执行 scene 变了，它们就茫然了。

**Temporal VLA**（[MemoryVLA](https://arxiv.org/abs/2508.19236), [HAMLET](https://arxiv.org/abs/2510.00695), [TraceVLA](https://arxiv.org/abs/2412.10345)）记得过去看到的 observation。但 history 是 observation 的 history，不是 "我的 action 把 scene 改成什么样了" 的 history。observation 是结果，action 是原因，两者不一样。

**Action-conditioned prediction**（[FLARE](https://arxiv.org/abs/2505.15659), [UP-VLA](https://arxiv.org/abs/2501.18867), [DreamerV3](https://arxiv.org/abs/2301.04104)）能预测未来——给定 action，预测未来 scene 长什么样。但它们预测完就在当前决策里用掉，**用完即弃**，下个 chunk 重新从零开始。

Paper 的论点很直接：你需要一个 scene representation 同时满足三个性质——**persist 跨 chunk、update under actions、correct against new observation**。三缺一都不行。缺 persist 就是 temporal VLA 的老问题；缺 action update 就是 spatial VLA 的老问题；缺 correction 就会 error accumulate。

---

## EvoScene-VLA 的核心 idea：给 action decoder 配个 "scene 记事本"

最 elegant 的 insight 在 Discussion 里那句话：**action decoder 是更新 scene state 的天然位置，因为它已经在 generate 改变 scene 的 action 序列了**。

既然 action decoder 已经在 denoise 一个 50 步的 action chunk，为什么不让它顺手也 denoise 一个 "scene 在这 50 步里怎么演化" 的 compact 记录？然后把这个记录传给下个 chunk 作为 prior？

这就是 "recurrent scene prefix" 的全部 idea。具体实现上，VLM 的 prefix 被扩展成：

$$[x_t, \, s_{\mathrm{obs}}^{(1:V)}, \, \bar{s}_t, \, \ell]$$

变量讲清楚：
- $x_t$：multi-view image（head cam + 两个 wrist cam，共 $V=3$ 个视图）
- $s_{\mathrm{obs}}^{(v)} \in \mathbb{R}^{16 \times 2048}$：第 $v$ 个视图的 observation slots，16 个 token，每个 2048 维（Qwen2.5-VL-3B 的 hidden dim）——它们从当前 image "采集" 几何证据
- $\bar{s}_t \in \mathbb{R}^{16 \times 2048}$：**prior slots**，从上个 chunk 继承来的 scene state——这是 "记事本"
- $\ell$：language instruction

关键设计是 **asymmetric attention mask**。规则很 simple：
- image tokens 和 language tokens 完全 ignore scene slots——预训练的 vision-language pathway 一点不被污染
- observation slots 只看自己视图的 image tokens + 同组其他 slots
- prior slots 看所有 observation slots + 自己，**不看 image/language**，也**没有任何 token 看回 prior slots**

这个 mask 干了三件事：
1. **保护预训练 pathway**：scene slots 不污染 image/language，所以 fine-tuning 友好
2. **Information bottleneck**：current image 到 prior 必须经过 observation slots，强制 prior 被 new observation correct
3. **单向流**：prior 不写回 image/language，prediction error 不会污染 perception

VLM forward 后，在 prior slots 位置读出 $s_p \in \mathbb{R}^{16 \times 2048}$——这就是 "用新 observation 修正过的、跨视图融合的、继承了 prior 的 scene representation"。

---

## 怎么让 prior 真的编码 geometry，而不是退化成压缩 image features

如果你只靠后面的 action loss 训练，scene slots 会自由地吸收任何有用的 feature——很可能就是 generic appearance，而不是 3D structure。那 prior 就没意义了。

Paper 用两级 Geometric Anchor 强制 ground 这个 representation：

### Local Anchor：cross-view masked depth reconstruction

把一个视图 mask 掉，强迫一个 head 从其他视图 + $s_p$ 重建被 mask 视图的 depth。公式（Eq.3-4）：

$$\hat{f}_{t,i}^d = g_{\mathrm{depth}}\left(q_{\mathrm{tmpl}}, \left[\tilde{h}_{t,v_1}^{\mathrm{img},(i)}, \ldots, \tilde{h}_{t,v_V}^{\mathrm{img},(i)}, s_p\right]\right)$$

$$\mathcal{L}_{\mathrm{geo}} = \frac{1}{V} \sum_{i=1}^{V} \mathrm{SmoothL1}\left(\hat{f}_{t,i}^d, \mathrm{MDT}(x_{t,v_i})\right)$$

变量：
- $g_{\mathrm{depth}}$：lightweight cross-attention head
- $q_{\mathrm{tmpl}} \in \mathbb{R}^{256 \times 2048}$：256 个 template query
- $\tilde{h}_{t,v_j}^{\mathrm{img},(i)}$：第 $j$ 个视图的 image tokens，当 $j=i$ 时被替换成 learned mask embedding $m$
- MDT = Monocular Depth Teacher，是 [MoGe-2](https://arxiv.org/abs/2503.21712) + LingBot-Depth 的组合（[ref](https://arxiv.org/abs/2601.17895)）

mask target view 切断了 "直接 copy 自己 image features" 的 shortcut——你必须从其他视图 + cross-view representation $s_p$ 推断。这隐含 multi-view stereo 的归纳偏置。

### Global Anchor：distill 3D foundation model

Local 只到 depth（2.5D），Global 进一步到 metric 3D。它 distill 一个 frozen 3D foundation model——具体是 [$\pi^3$](https://arxiv.org/abs/2503.18704)（permutation-equivariant visual geometry learning）。公式（Eq.5-6）：

$$P_t = W_{\mathrm{proj}} \cdot g_{\mathrm{3D}}(q_{\mathrm{dec}}; s_p)$$

$$\mathcal{L}_{\mathrm{rep}} = \frac{1}{V} \sum_{v=1}^{V} \lVert P_t^{(v)} - Z_t^{(v)} \rVert_1, \quad Z_t = \pi^3(x_t)$$

变量：
- $g_{\mathrm{3D}}$：view-conditioned cross-attention decoder，以 $s_p$ 为 keys/values
- $q_{\mathrm{dec}}$：learnable view-aware queries
- $W_{\mathrm{proj}}$：linear projector 到 $\pi^3$ feature space
- $Z_t$：frozen $\pi^3$ 在同 multi-view 上的 features
- $\ell_1$ loss：dense token-level supervision

**关键：这个 decoder $(g_{\mathrm{3D}}, W_{\mathrm{proj}})$ 会被 Scene Predictor 复用**——current 和 future representation 共享同一个 "解释器"，确保它们在同一个 feature space 对齐。这是后面 co-denoising 能收敛的前提。

---

## 怎么教 action decoder 更新 prior：Scene Predictor

现在 prior slots 知道编码什么了，但 action decoder 还不知道怎么更新它。需要一个 teacher 教它："给定当前 scene + action 序列，未来 scene 长什么样"。

这就是 Scene Predictor，**只在训练时存在**。它是个 causal Transformer，输入（Eq.7）：

$$[r_t, \, s_p, \, a_t, \ldots, a_{t+H}, \, q_1, \ldots, q_K], \quad q_i = \mathrm{copy}(s_p)$$

变量：
- $r_t$：robot proprioceptive state
- $s_p$：当前 corrected scene representation
- $a_t, \ldots, a_{t+H}$：ground-truth action chunk（$H=49$）
- $q_1, \ldots, q_K$：$K=3$ 个 key-frame query groups，每个 shape $\mathbb{R}^{16 \times 2048}$，都 init 自 $s_p$ 的 copy
- key-frame offsets $\{k_1, k_2, k_3\} \subseteq \{1, \ldots, 49\}$：chunk 内部稀疏选 3 个时刻

Causal mask 让每个 $q_i$ 只 attend 到 $r_t, s_p$、action prefix 到 $a_{t+k_i}$ 为止、以及更早的 query groups。所以预测 $t+k_i$ 时刻的 scene 只用 $t+k_i$ 之前的 action——causality 正确。

输出 $\hat{s}_{t+k_1:t+k_3}$，每个 $\hat{s}_{t+k_i} \in \mathbb{R}^{16 \times 2048}$。

Loss（Eq.8）把这些预测 future scene latents 经 **同一个** $(g_{\mathrm{3D}}, W_{\mathrm{proj}})$ decoder 投影，匹配 future frame 上 frozen $\pi^3$ 的 features：

$$\mathcal{L}_{\mathrm{pred}} = \frac{1}{K \cdot V} \sum_{i=1}^{K} \sum_{v=1}^{V} \lVert \tilde{P}_{t+k_i}^{(v)} - Z_{t+k_i}^{(v)} \rVert_1$$

---

## 核心：Joint Action-Scene Denoising

现在 action decoder 要学会同时 denoise action chunk 和 scene chunk。这里用 [flow matching](https://arxiv.org/abs/2210.02747)（[$\pi_0$](https://arxiv.org/abs/2410.24164) 同款）。

**最关键的设计：action 和 scene 共享同一个 flow-matching time $\tau$**（Eq.9）：

$$a_{t:t+H}^\tau = \tau \epsilon_a + (1-\tau) a_{t:t+H}$$

$$z^\tau = \tau \tilde{\epsilon}_s + (1-\tau) z_0, \quad \tilde{\epsilon}_s := \sigma \epsilon_s$$

变量：
- $\tau \in [0,1]$：flow-matching time，**两个 path 共享**
- $\epsilon_a, \epsilon_s$：action path 和 scene path 的独立高斯噪声
- $\sigma$：scene noise rescale factor，让 noise 量级匹配 $z_0$ 的量级（因为 $z_0$ 经 LayerNorm，scale 和 action 不同）
- $z_0 \in \mathbb{R}^{K \times N \times D} = \mathbb{R}^{3 \times 16 \times 2048}$：stacked future-scene targets（来自 Scene Predictor，detach gradient）

为什么 share $\tau$？这强制 action 和 scene 在 **同一个 denoising schedule 上协同**。$\tau=1$ 都从 noise 出发，$\tau=0$ 都到达 target。中间每步，action expert 必须输出 consistent 的 $(v^{(a)}, v^{(s)})$——隐式让 action 每步 denoising 都考虑对应 scene 状态。如果独立采样 $\tau_a, \tau_s$，两个 path 就解耦了，co-denoising 退化成两个独立 flow matching。

Action expert 接收 suffix（Eq.10）：

$$[\boldsymbol{r}_t \mid z^\tau \mid a_{t:t+H}^\tau]$$

在 causal suffix mask 下 attend 到 VLM prefix cache（含 image, language, observation, prior slots）。预测两个 velocity（Eq.11）：

$$\mathcal{L}_{\mathrm{sceneFM}} = \lVert v_\theta^{(s)}(z^\tau, \tau) - (\tilde{\epsilon}_s - z_0) \rVert_2^2$$

$$\mathcal{L}_{\mathrm{actFM}} = \lVert v_\theta^{(a)}(a^\tau, \tau) - (\epsilon_a - a_{t:t+H}) \rVert_2^2$$

$\mathcal{L}_{\mathrm{actFM}}$ 是 $\pi_{0.5}$ 的标准 action loss。$\mathcal{L}_{\mathrm{sceneFM}}$ 把 future scene 蒸馏进 action expert——这就是 "build world model into policy decoder" 的实现。

注意 $\lambda_4 = 0.01$ 很小（Tab.4），说明 scene FM loss 是 soft distillation，不抢 action branch 的 capacity。

---

## Inference：闭环就形成了

Algorithm 2 的人话版：

1. VLM forward 一遍，读出 $s_p$，cache prefix KV
2. Init action noise $a^1 \sim \mathcal{N}(0,I)$，scene noise $z^1 \sim \mathcal{N}(0,I)$
3. 跑 10 步 Euler denoising：每步 action expert 同时输出 $(v^{(a)}, v^{(s)})$，更新 $(a^\tau, z^\tau)$
4. 拿到 denoised action chunk $\hat{a}_{t:t+H}$ 和 denoised scene chunk $\hat{s}_{t+k_1:t+k_3}$
5. 执行 $\hat{a}_{t:t+H}$
6. **把 $\hat{s}_{t+k_3}$ 写回作为下个 chunk 的 prior $\bar{s}_{t+1}$** ← 这是 recurrence 的核心
7. Re-observe $x_{t+1}$，回 step 1

为什么取 $k_3$（最后一个 key frame）？因为 $k_3$ 对应 chunk 执行结束时的 scene state——这正是下个 chunk 开始时 robot 真正面对的 scene。取 $k_1$ 的话 prior 反映 chunk 中间状态，与下个 chunk start 时刻不匹配。

**Geometric Anchor 和 Scene Predictor 在 inference 时全部丢弃**。只剩 recurrent scene prefix + action-scene co-denoising。这是 "training scaffolding" pattern：用复杂训练模块塑造简洁 inference 模型。

---

## 实验证据

### RoboTwin 31 任务主结果（Tab.1）

| Method | Clean | Rand |
|--------|-------|------|
| $\pi_{0.5}$ | 81.2 | 75.9 |
| LingBot-VLA | 85.3 | 84.1 |
| LingBot-VLA* (depth-augmented) | 87.2 | 86.1 |
| **EvoScene-VLA** | **89.1** | **88.5** |

- Clean 提升 +1.9，**Rand 提升 +2.4 更大**
- Rand 下初始 layout 随机化更大，单 observation perception 误差更大，跨 chunk 累积更严重——recurrent scene prefix 正好 mitigate 这两个问题

### Ablation（Tab.2，RoboTwin-5Task）

| Variant | Clean | Rand |
|---------|-------|------|
| baseline | 81.6 | 75.8 |
| + $\mathcal{L}_{\mathrm{pred}}$ & $\mathcal{L}_{\mathrm{rep}}$ | 89.3 | 86.2 |
| + $\mathcal{L}_{\mathrm{geo}}$ | 90.1 | 86.5 |
| + prior info at inference | **90.8** | **87.8** |

最后一行最关键：它对比 "每个 chunk 重新 init prior from learnable embedding" vs "从上个 chunk 继承 denoised scene token"。**纯 recurrence 贡献 +0.7/+1.3**——证明跨 chunk scene prior 确实有用。

### Real-robot（Tab.3，Galaxea R1-Lite）

| Task | $\pi_{0.5}$ | LingBot-VLA* | EvoScene-VLA |
|------|-------------|--------------|--------------|
| Mirror | 28 | 26 | 29 |
| Sink | 42 | 49 | 51 |
| Cutting-board | 44 | 37 | **46** |
| **Avg** | 38.0 | 37.3 | **42.0** |

Cutting-board 提升最大（+9）：擦拭任务 surface state 随 action 演化最明显，后续 control 必须瞄准未擦拭区域——这正是 EvoScene-VLA 设计针对的场景。

Real-robot 数据来自 [Galaxea Open-World Dataset](https://arxiv.org/abs/2509.00576)：7 sessions，439 episodes，48,419 frames，~9 小时 bimanual demos @ 15fps。Inference 单卡 RTX 4090。

### Trajectory 质量（Fig.3, Fig.5）

3D end-effector trajectory 显示 EvoScene-VLA 明显更平滑。Paper 的解释：co-denoising 把每个 action tie 到 consistent scene context，避免 action-only baseline 的 abrupt corrections。这跟 [RDT-1B](https://arxiv.org/abs/2410.07864) 的 diffusion 平滑性类似，但来源不同——RDT 是 trajectory-level modeling，EvoScene 是 action-scene co-conditioning。

---

## 直觉总结：这 paper 干了什么

一句话：**给 chunked VLA policy 配了一个 16-token 的 "scene 记事本"，记事本每次 action chunk 执行后被更新，下个 chunk 开局带着这个记事本，并用新 observation 修正它**。

三个设计选择各司其职：
1. **Recurrent scene prefix** 定义记事本在哪——prefix 的 prior slots，通过 asymmetric attention mask 保护预训练 pathway 同时建立 information bottleneck
2. **Two-level Geometric Anchor** 定义记事本记什么——per-view depth + cross-view 3D foundation model features，确保记的是 geometry 不是 appearance
3. **Scene Predictor + scene FM loss** 定义记事本怎么 update——先让 Scene Predictor 学会从 (scene, action) 预测 future scene，再蒸馏进 action expert 的 scene branch，让 action expert 自己成为 scene updater

最 elegant 的地方在 inference：所有 supervision infrastructure（Geometric Anchor、Scene Predictor）都丢弃，只剩 recurrent scene prefix + action-scene co-denoising。Action decoder 身兼两职——既 generate motor command 又 update scene prior——因为 "它已经在 generate 改变 scene 的 action 序列了，顺手更新 scene state 是最自然的"。

实验证据指向同一个 failure mode：**chunked control 的 mid-chunk stall**——baseline 追 stale target 中途卡住，EvoScene-VLA 带着演化的 prior 继续执行。Rand 比 Clean 提升更大说明 prior 对 pose/layout variation 鲁棒；real-robot cleaning 提升最大说明 prior 对 evolving surface state 敏感；trajectory 更平滑说明 co-denoising 让 action 与 scene 一致。

---

## 我觉得最值得 follow up 的方向

Paper Discussion 最后那句其实很有想象力：**用 observation slots 和 prior slots 的 mismatch 作为 uncertainty signal**。

什么意思？observation slots 来自新 image，prior slots 来自上个 chunk 的 prediction。如果两者差异大，说明 prediction error 大——下个 chunk 应该提前 re-observe 或 replan，甚至缩短 chunk length。这就把 EvoScene-VLA 从 "passive prior" 升级成 "active perception" policy，类似 active inference 框架里 prediction error 驱动行为的概念。

更进一步，如果这个 mismatch 能驱动 **adaptive chunk execution**——chunk 内部检测到 prediction error 超阈值就提前 break chunk 重新 observe——那 chunked control 就从 "open-loop within chunk" 变成 "closed-loop within chunk"。这会解决 chunked policy 最根本的 trade-off：长 chunk 高效但 stale，短 chunk responsive 但昂贵。

其他联想：
- **Longer chunk 的双刃剑**：Paper 提到 longer chunk 让 scene drift 更大（recurrence 更有价值），但 Scene Predictor target 更难预测。$H=50$ 已经不短，$H=100$ 会怎样是个 open question
- **与 [DreamerV3](https://arxiv.org/abs/2301.04104) world model 的关系**：EvoScene 把 scene updater 嵌入 action decoder，没单独 online world model。但 Dreamer 风格 world model 在 long-horizon planning 上可能更强——EvoScene 的 prior 是 chunk-level，Dreamer 是 step-level
- **与 [CUT3R](https://arxiv.org/abs/2501.12387) persistent state 的对比**：CUT3R 用 state token 做 dense 3D reconstruction，EvoScene 用 state token 做 policy-facing compact prior。同样 "persistent state across calls" 思想，但目标不同——dense reconstruction vs task-relevant geometry
- **Latent interpretability**：paper 自陈 prior 是 latent，geometric content 只能间接判断。如果能 force prior slots 对应 explicit 3D entities（像 [Embodied-SlotSSM](https://arxiv.org/abs/2511.11478) 的 object-centric slots），prior 就可解释了

---

## References

- [EvoScene-VLA paper](https://arxiv.org/abs/2601.18692)（实际是 LingBot-VLA 系列）
- [π0 / π0.5: VLA flow model](https://arxiv.org/abs/2410.24164)
- [SpatialVLA](https://arxiv.org/abs/2501.15830)
- [MemoryVLA](https://arxiv.org/abs/2508.19236)
- [HAMLET: History-aware VLA](https://arxiv.org/abs/2510.00695)
- [TraceVLA](https://arxiv.org/abs/2412.10345)
- [FLARE: Implicit world modeling](https://arxiv.org/abs/2505.15659)
- [UP-VLA](https://arxiv.org/abs/2501.18867)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03022)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [π³: Permutation-equivariant visual geometry](https://arxiv.org/abs/2503.18704)
- [DUSt3R](https://arxiv.org/abs/2312.14132) / [MASt3R](https://arxiv.org/abs/2406.09756) / [CUT3R](https://arxiv.org/abs/2501.12387)
- [RoboTwin benchmark](https://arxiv.org/abs/2409.02920)
- [Galaxea Open-World Dataset](https://arxiv.org/abs/2509.00576)
- [RDT-1B](https://arxiv.org/abs/2410.07864)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Embodied-SlotSSM](https://arxiv.org/abs/2511.11478)

---

# EvoScene-VLA 深度技术解析

这篇 paper 直击 chunked VLA control 的一个 fundamental gap：**action 会改变 scene，但 policy 在 chunk 内部和跨 chunk 边界都缺乏对这个 changed scene 的紧凑记忆**。让我从 problem motivation、architecture、loss design、experimental evidence 四个层面展开，并尽可能多地 connect 到相关 work。

---

## 1. Problem: Chunked Control 的 Scene Drift

### 1.1 失败模式的形式化

Chunked policy 在 time $t$ 接收 observation $x_t$，然后预测 $H+1$ 步 action chunk $a_{t:t+H}$（这里 $H=49$，chunk length = 50）。问题是：**chunk 内部所有 $a_{t+k}$ 都 conditioning on $x_t$ 这一个 observation**，但 $a_t$ 执行后 scene 已经变化了。

Paper 给了三个 qualitative 例子（Fig.1）：
- **Open Microwave**：gripper 还没到 door handle 就 retract 了——因为整 chunk 是基于 start state 规划的，gripper trajectory 在 chunk 内部 "commit 到 stale target"
- **Block stacking / Cabinet placement**：同样 mid-chunk stall

这个失败模式可以形式化为：设 $S_t$ 是 time $t$ 的 true scene state，policy 实际 access 的是 $S_t$（通过 $x_t$），但 chunk 内第 $k$ 步的 optimal action 应该 conditioning on $S_{t+k} \neq S_t$。**误差源**是 $\Delta_k = S_{t+k} - S_t$，其中 $\Delta_k$ 由 robot 自己的 action 产生。

### 1.2 现有 work 为什么不够

| 类别 | 代表工作 | 缺陷 |
|------|---------|------|
| **Spatial VLA** | [SpatialVLA](https://arxiv.org/abs/2501.15830), [QDepth-VLA](https://arxiv.org/abs/2510.14836), [Spatial Forcing](https://arxiv.org/abs/2510.12276), [3DS-VLA](https://arxiv.org/abs/2403.03954), [VLA-4D](https://arxiv.org/abs/2511.17199) | improve 单帧 geometry，但 action 改变 scene 后不更新 |
| **Temporal VLA** | [MemoryVLA](https://arxiv.org/abs/2508.19236), [HAMLET](https://arxiv.org/abs/2510.00695), [HiF-VLA](https://arxiv.org/abs/2512.09928), [Embodied-SlotSSM](https://arxiv.org/abs/2511.11478), [TraceVLA](https://arxiv.org/abs/2412.10345), [AVA-VLA](https://arxiv.org/abs/2511.18960) | 保留 observed history，但 history ≠ "action 会如何改变 scene" 的 prior |
| **Action-conditioned prediction** | [FLARE](https://arxiv.org/abs/2505.15659), [UP-VLA](https://arxiv.org/abs/2501.18867), world models ([DreamerV3](https://arxiv.org/abs/2301.04104), [Day-Dreamer](https://arxiv.org/abs/2204.04905)) | predict future for current decision，**用完即弃**，不跨 chunk 持久化 |

EvoScene-VLA 的论点：**scene representation 应该 persist 跨 chunk、在 action 下 update、并被新 observation correct**。三性缺一不可。

这让我联想到 [CUT3R](https://arxiv.org/abs/2501.12387) 的 persistent state tokens——但 CUT3R 是 dense 3D reconstruction 系统，不是 policy-facing 的紧凑 latent；而 [DUSt3R](https://arxiv.org/abs/2312.14132)/[MASt3R](https://arxiv.org/abs/2406.09756) 也是 reconstruction-oriented。EvoScene-VLA 借鉴了 "state token 跨调用持久化" 的思想，但把它做成了 **policy-facing 的紧凑 latent**（仅 $N=16$ tokens，$D=2048$，约 33K float）。

---

## 2. Architecture: Recurrent Scene Prefix

### 2.1 Prefix 结构

VLM prefix 被扩展为（Eq.1）：

$$[x_t, \, s_{\mathrm{obs}}^{(1:V)}, \, \bar{s}_t, \, \ell]$$

变量含义：
- $x_t \in \mathbb{R}^{V \times \cdots}$：多视图图像输入（$V=3$：head, left wrist, right wrist，分辨率 $224 \times 224$）
- $s_{\mathrm{obs}}^{(v)} \in \mathbb{R}^{N \times D}$：第 $v$ 个视图的 observation slots，$N=16$, $D=2048$（Qwen2.5-VL-3B 的 hidden dim）
- $\bar{s}_t \in \mathbb{R}^{N \times D}$：**prior slots**，从上一个 chunk 的 action expert denoising 结果继承（首个 chunk 用 learnable embedding 初始化）
- $\ell$：language instruction

总共新增的 token 数：$V \cdot N + N = 3 \times 16 + 16 = 64$ 个 scene tokens，相对 Qwen2.5-VL 的 image tokens（每视图约 256 tokens × 3 = 768）规模合理。

### 2.2 Asymmetric Attention Mask——这是设计的关键

这是整个 architecture 最 elegant 的部分。Attention routing 规则：

| Source \ Target | Image tokens | Language tokens | Observation slots (own view) | Observation slots (other views) | Prior slots |
|-----------------|-------------|-----------------|------------------------------|----------------------------------|-------------|
| Image tokens | ✓ | ✓ | ✗ | ✗ | ✗ |
| Language tokens | ✓ | ✓ | ✗ | ✗ | ✗ |
| Observation slots (own view) | ✓ | ✗ | ✓ | ✗ | ✗ |
| Observation slots (other views) | ✓ | ✗ | ✗ | ✓ | ✗ |
| Prior slots | ✗ | ✗ | ✓ (all views) | ✓ (all views) | ✓ (self) |

关键性质：
1. **Image/language 路径完全保留**：scene slots 不污染预训练的 vision-language pathway。这是 few-shot fine-tuning 友好的关键。
2. **Information bottleneck**：current image evidence 到达 prior slots **只能通过** observation slots。这是 "correct against new observation" 的 architectural enforcement。
3. **单向流**：prior slots 不写回 image/language tokens，避免 prior 的预测误差污染 perception。

这让我想到 [HPT](https://arxiv.org/abs/2409.20537) 的 heterogeneous pre-training 中也强调不要破坏 pretrained pathway——但 HPT 用的是 latent alignment，EvoScene-VLA 用的是 architectural isolation。

### 2.3 VLM 输出：corrected scene representation

VLM forward 后，在 prior-slot 位置读出（Eq.2）：

$$s_p = \mathrm{VLM}_{\mathrm{scene}}\left(x_t, \ell, s_{\mathrm{obs}}^{(1:V)}, \bar{s}_t\right)_{\mathrm{prior}} \in \mathbb{R}^{N \times D}$$

$s_p$ 的语义：**用新 observation 修正后的、跨视图融合的、继承了 prior 的 scene representation**。它是 recurrent state 的 "current-frame 版本"。

注意：prefix 定义了 state **在哪里**，不定义它 **编码什么**。编码内容由后续的 Geometric Anchor + Scene Predictor + co-denoising 共同塑造。

---

## 3. Two-Level Geometric Anchor（训练时）

### 3.1 为什么需要 geometric supervision

如果只靠 action expert 的 co-denoising loss，scene slots 会自由地吸收 generic appearance features，而不是 3D structure。这会导致 prior slots 退化为 "压缩的 image features"，失去跨 chunk 的几何意义。

Geometric Anchor 分两级：**local**（per-view depth）和 **global**（cross-view 3D foundation model）。

### 3.2 Local Anchor: Cross-View Masked Depth Reconstruction

核心思想：**mask 掉一个视图，强迫 head 从其他视图 + $s_p$ 重建被 mask 视图的 depth**。这强制 observation slots 编码 per-view geometry，且 $s_p$ 必须编码 cross-view geometry。

具体（Eq.3）：

$$\hat{f}_{t,i}^d = g_{\mathrm{depth}}\left(q_{\mathrm{tmpl}}, \left[\tilde{h}_{t,v_1}^{\mathrm{img},(i)}, \ldots, \tilde{h}_{t,v_V}^{\mathrm{img},(i)}, s_p\right]\right)$$

其中：
- $q_{\mathrm{tmpl}} \in \mathbb{R}^{256 \times D}$：256-token template query bank（也用于 $g_{\mathrm{depth}}$ 的 query）
- $g_{\mathrm{depth}}$：lightweight cross-attention head
- $\tilde{h}_{t,v_j}^{\mathrm{img},(i)}$：第 $j$ 个视图的 VLM image tokens，但当 $j=i$ 时被替换为 learned mask embedding $m \in \mathbb{R}^D$
- $s_p$：Eq.2 的输出

Loss（Eq.4）：

$$\mathcal{L}_{\mathrm{geo}} = \frac{1}{V} \sum_{i=1}^{V} \mathrm{SmoothL1}\left(\hat{f}_{t,i}^d, f_{t,i}^d\right), \quad f_{t,i}^d = \mathrm{MDT}(x_{t,v_i})$$

- MDT = Monocular Depth Teacher，由 [MoGe-2](https://arxiv.org/abs/2503.21712) ViT-B + LingBot-Depth masked-depth-modeling teacher 组成（[ref 32](https://arxiv.org/abs/2601.17895), [ref 29](https://huggingface.co/robbyant/lingbot-depth)）
- SmoothL1 对 outlier 鲁棒

**关键设计：mask target view 切断了 "直接 copy 自己 VLM features" 的 shortcut**，迫使 $g_{\mathrm{depth}}$ 从未 mask 的视图和 $s_p$ 聚合证据。这隐含一个 multi-view stereo 的归纳偏置。

### 3.3 Global Anchor: 3D Foundation Model Decoding

Local anchor 只到 depth（2.5D），global anchor 进一步到 metric 3D。它 distill 一个 frozen multi-view 3DFM——具体是 [$\pi^3$](https://arxiv.org/abs/2503.18704)（permutation-equivariant visual geometry learning，ICLR 2026）。

公式（Eq.5-6）：

$$H_t = g_{\mathrm{3D}}(q_{\mathrm{dec}}; s_p), \quad P_t = W_{\mathrm{proj}} H_t$$

$$\mathcal{L}_{\mathrm{rep}} = \frac{1}{V} \sum_{v=1}^{V} \lVert P_t^{(v)} - Z_t^{(v)} \rVert_1, \quad Z_t = \mathrm{3DFM}(x_t)$$

- $g_{\mathrm{3D}}$：view-conditioned cross-attention decoder，以 $s_p$ 为 keys/values
- $q_{\mathrm{dec}}$：learnable view-aware queries
- $W_{\mathrm{proj}}$：linear projector 到 3DFM feature space
- $Z_t$：frozen 3DFM 在同 multi-view 输入上的 features
- $\ell_1$ loss：dense token-level supervision，对 outlier 鲁棒，保留方向和量级

**这个 decoder 会被 Scene Predictor 复用**（见下节），这是 "shared decoder handles both current and future representations" 的实现，减少了训练参数和 supervision 不一致。

### 3.4 Local vs Global 的互补性

| 维度 | Local Anchor | Global Anchor |
|------|-------------|---------------|
| 监督粒度 | per-view depth | cross-view 3D features |
| 归纳偏置 | multi-view stereo | metric 3D scene embedding |
| 训练信号 | dense per-pixel | dense per-token |
| 作用对象 | 每个 $s_{\mathrm{obs}}^{(v)}$ + $s_p$ | $s_p$（aggregated） |

Ablation（Tab.2）显示：
- 仅 baseline：81.6% / 75.8%
- + $\mathcal{L}_{\mathrm{pred}}$ & $\mathcal{L}_{\mathrm{rep}}$：89.3% / 86.2%（+7.7/+10.4）
- + $\mathcal{L}_{\mathrm{geo}}$：90.1% / 86.5%（+0.8/+0.3）

Local anchor 的边际贡献小，但 consistent——它 ground 了 per-view geometry，complement 了 global anchor 的 cross-view 监督。

---

## 4. Scene Predictor（训练时）

### 4.1 任务

Scene Predictor 产生 **future scene-token targets**，让 action expert 的 scene branch 学会 "给定当前 scene + action 序列，预测未来 scene"。

输入序列（Eq.7）：

$$[r_t, \, s_p, \, a_t, \ldots, a_{t+H}, \, q_1, \ldots, q_K], \quad q_i = \mathrm{copy}(s_p), \, i=1,\ldots,K$$

- $r_t$：robot proprioceptive state
- $s_p$：当前 corrected scene representation
- $a_t, \ldots, a_{t+H}$：ground-truth action chunk
- $q_1, \ldots, q_K$：K=3 个 key-frame query groups，每个 shape $\mathbb{R}^{N \times D}$，都 init 自 $s_p$ 的 copy
- key-frame offsets $\{k_1, k_2, k_3\} \subseteq \{1, \ldots, 49\}$

### 4.2 Causal Mask 的设计

每个 $q_i$ 在 causal mask 下 attend 到：
- $r_t, s_p$
- action prefix $a_{t:t+k_i^\ast}$（到其 target step 为止）
- 更早的 query groups $q_1, \ldots, q_{i-1}$

这意味着：**预测 step $t+k_i$ 的 scene 只 conditioning 于到 $t+k_i$ 之前执行的 actions**。这正确反映 causal 结构——你不能用未来 action 预测中间 scene。

输出：$\hat{s}_{t+k_1:t+k_K}$，每个 $\hat{s}_{t+k_i} \in \mathbb{R}^{N \times D}$。

### 4.3 Loss 与 decoder 复用

Loss（Eq.8）：

$$\mathcal{L}_{\mathrm{pred}} = \frac{1}{K \cdot V} \sum_{i=1}^{K} \sum_{v=1}^{V} \lVert \tilde{P}_{t+k_i}^{(v)} - Z_{t+k_i}^{(v)} \rVert_1$$

$$\tilde{P}_{t+k_i} = W_{\mathrm{proj}}\left(g_{\mathrm{3D}}(q_{\mathrm{dec}}; \hat{s}_{t+k_i})\right)$$

- $\tilde{P}_{t+k_i}$：predicted future scene latent 经 **同一个** $(g_{\mathrm{3D}}, W_{\mathrm{proj}})$ decoder 投影
- $Z_{t+k_i} = \mathrm{3DFM}(x_{t+k_i})$：future-frame multi-view 上 frozen 3DFM 的 features

**Decoder sharing 的意义**：current 和 future scene latents 共享同一个 "解释器" decoder，因此它们在同一个 3D feature space 中被对齐。这避免了 "current representation 学一个 space，future representation 学另一个 space" 的不一致——这种不一致会让 co-denoising 难以收敛。

### 4.4 与 FLARE / UP-VLA 的对比

- [FLARE](https://arxiv.org/abs/2505.15659)：插入 learnable future tokens 到 VLM，但 future token 只服务当前 decision，不跨 chunk 持久
- [UP-VLA](https://arxiv.org/abs/2501.18867)：action-conditioned image/feature prediction，也是 in-decision consumption
- EvoScene-VLA Scene Predictor：**只在训练时存在**，目的是给 action expert 的 scene branch 提供 target；inference 时完全丢弃

这是 "teacher-student distillation into the action expert" 的模式，类似 [Consistency Policy](https://arxiv.org/abs/2405.07503) 把 diffusion sampler 蒸馏成 consistency model——但这里蒸馏的是 "scene predictor" 到 "action expert 的 scene branch"。

---

## 5. Joint Action-Scene Denoising（核心）

### 5.1 Flow Matching 基础

[Flow matching](https://arxiv.org/abs/2210.02747)（[Lipman et al.](https://arxiv.org/abs/2210.02747), [Liu et al. Rectified Flow](https://arxiv.org/abs/2209.03022)）定义一个 probability path 从 noise $\epsilon$ 到 data $z_0$：

$$z^\tau = \tau \cdot \epsilon + (1-\tau) \cdot z_0, \quad \tau \in [0,1]$$

velocity field 目标：

$$v^\star(z^\tau, \tau) = \epsilon - z_0$$

训练 loss：$\lVert v_\theta(z^\tau, \tau) - (\epsilon - z_0) \rVert^2$

[$\pi_0$](https://arxiv.org/abs/2410.24164)/[$\pi_{0.5}$](https://arxiv.org/abs/2410.24164) 把这个用于 robot action 的 chunk denoising。

### 5.2 EvoScene-VLA 的扩展：shared flow-matching time

**核心创新**：action 和 scene 两个 path **共享同一个 $\tau$**（Eq.9）：

$$a_{t:t+H}^\tau = \tau \epsilon_a + (1-\tau) a_{t:t+H}$$

$$z^\tau = \tau \tilde{\epsilon}_s + (1-\tau) z_0, \quad \tilde{\epsilon}_s := \sigma \epsilon_s$$

- $\epsilon_a$：action path 的高斯噪声
- $\epsilon_s$：scene path 的高斯噪声
- $\sigma$：rescale factor，让 scene noise 的量级匹配 $z_0$ 的经验量级（因为 $z_0 = \mathrm{LayerNorm}(\hat{s}_{t+k_1:t+k_K})$ 的 scale 与 action targets 不同）
- $z_0 \in \mathbb{R}^{K \times N \times D}$：stacked future-scene targets，经 LayerNorm 标准化

**为什么 share $\tau$？** 这强制 action 和 scene 在 **同一个 denoising schedule 上协同**。在 $\tau=1$（纯噪声）时两者都从 noise 出发；在 $\tau=0$（data）时两者都到达 target。中间每一步，action expert 必须输出 consistent 的 $(v^{(a)}, v^{(s)})$——这隐式地让 action 的每步 denoising 都考虑对应的 scene 状态。

如果不 share $\tau$（例如 action 用 $\tau_a$，scene 用 $\tau_s$ 独立采样），action 和 scene 的 denoising trajectory 就解耦了，co-denoising 退化为两个独立 flow matching。

### 5.3 Action Expert 的输入

Suffix（Eq.10）：

$$[\boldsymbol{r}_t \mid z^\tau \mid a_{t:t+H}^\tau]$$

- $r_t$：robot state
- $z^\tau$：当前 denoising 步的 scene noise
- $a_{t:t+H}^\tau$：当前 denoising 步的 action noise

Action expert 在 causal suffix mask 下 attend 到 VLM prefix cache（包含 image, language, observation, prior slots）。

### 5.4 Velocity Losses

Eq.11：

$$\mathcal{L}_{\mathrm{sceneFM}} = \lVert v_\theta^{(s)}(z^\tau, \tau) - (\tilde{\epsilon}_s - z_0) \rVert_2^2$$

$$\mathcal{L}_{\mathrm{actFM}} = \lVert v_\theta^{(a)}(a^\tau, \tau) - (\epsilon_a - a_{t:t+H}) \rVert_2^2$$

- $v_\theta^{(s)}$：scene path 的 predicted velocity
- $v_\theta^{(a)}$：action path 的 predicted velocity
- **stop gradient through both targets**：$z_0$ 和 $a_{t:t+H}$ 都被 detach，避免梯度通过 Scene Predictor 的预测 target 回传——这是 distillation 标准做法

$\mathcal{L}_{\mathrm{actFM}}$ 是 $\pi_{0.5}$ 的标准 action FM loss。$\mathcal{L}_{\mathrm{sceneFM}}$ 把 future scene representations 蒸馏进 action expert。

### 5.5 Inference 时的 recurrence 闭环

Algorithm 2 的关键步骤：

```
1. VLM forward → 读出 s_p，cache prefix KV
2. Init a^1 ~ N(0,I), z^1 ~ N(0,I)
3. for τ ∈ {1, 1-Δτ, ..., Δτ} (10 Euler steps):
     v^(a), v^(s) = ActionExpert([r_t | z^τ | a^τ]; prefix_cache)
     a^(τ-Δτ) = a^τ - Δτ·v^(a)
     z^(τ-Δτ) = z^τ - Δτ·v^(s)
4. â_{t:t+H} = a^0; ŝ_{t+k_1:t+k_K} = z^0
5. Execute â_{t:t+H} on robot
6. s̄_{t+1} ← ŝ_{t+k_K}   # 关键：最后 key-frame 的 scene token 成为下个 chunk 的 prior
7. Re-observe x_{t+1}，回到 step 1
```

**第 6 步是 recurrence 的核心**：denoised scene token 在 executed key-frame offset $k_K$ 处被写回，成为下个 chunk 的 $\bar{s}_{t+1}$。下个 chunk 的 VLM call 用新 observation $x_{t+1}$ 来 correct 它。

**为什么取 $k_K$（最后一个 key frame）而不是中间的？** 因为 $k_K$ 对应 chunk 执行结束时的 scene state——这正是下个 chunk 开始时 robot 真正面对的 scene。如果取 $k_1$（最早 key frame），prior 反映的是 chunk 中间状态，与下个 chunk 的 start 时刻不匹配。

### 5.6 与 Diffusion Policy / $\pi_0$ 的对比

| 方面 | [Diffusion Policy](https://arxiv.org/abs/2303.04137) | [$\pi_0$](https://arxiv.org/abs/2410.24164)/[$\pi_{0.5}$](https://arxiv.org/abs/2410.24164) | EvoScene-VLA |
|------|---------------------|------------------|--------------|
| Denoise 对象 | action chunk | action chunk | **action chunk + scene chunk（joint）** |
| Recurrent state | 无 | 无 | **scene token 跨 chunk** |
| 条件 | observation | VLM prefix | VLM prefix + recurrent prior |
| Training-only modules | 无 | 无 | Geometric Anchor + Scene Predictor |
| Inference 复杂度 | 多步 denoising | 多步 denoising | 多步 denoising（同 $\pi_{0.5}$，但 scene branch 几乎无额外开销） |

关键：EvoScene-VLA 在 inference 时 **没有额外的 online predictor**——它复用 action expert 的 scene branch 作为 scene updater。这是 "build the world model into the policy decoder" 的设计哲学。

---

## 6. Total Loss

Eq.12：

$$\mathcal{L} = \mathcal{L}_{\mathrm{actFM}} + \lambda_1 \mathcal{L}_{\mathrm{geo}} + \lambda_2 \mathcal{L}_{\mathrm{rep}} + \lambda_3 \mathcal{L}_{\mathrm{pred}} + \lambda_4 \mathcal{L}_{\mathrm{sceneFM}}$$

权重（Tab.4）：
- $\lambda_1 = 0.04$（local depth anchor）
- $\lambda_2 = 0.10$（global 3DFM anchor）
- $\lambda_3 = 0.10$（scene predictor）
- $\lambda_4 = 0.01$（scene flow matching distillation）

四个 scene-side loss 的角色：
1. $\mathcal{L}_{\mathrm{geo}}, \mathcal{L}_{\mathrm{rep}}$：ground **current** scene representation in geometry
2. $\mathcal{L}_{\mathrm{pred}}$：train **future** representations in the same coordinate
3. $\mathcal{L}_{\mathrm{sceneFM}}$：**transfer** future representations into action expert

注意 $\lambda_4 = 0.01$ 很小——这意味着 scene FM loss 是 "soft distillation"，不是 hard target。这避免了 scene branch 抢占 action branch 的 capacity。

训练配置：
- Optimizer: AdamW, lr = $1 \times 10^{-4}$, constant schedule
- Effective batch size: 256
- 20,000 update steps
- Mixed precision: bf16 storage / fp32 reductions（action expert held in fp32）
- Hardware: 8×A800
- Single-stage end-to-end training

---

## 7. Experiments

### 7.1 RoboTwin 31 tasks 主结果（Tab.1）

| Method | Clean avg | Rand avg |
|--------|----------|----------|
| $\pi_{0.5}$ | 81.2 | 75.9 |
| LingBot-VLA | 85.3 | 84.1 |
| LingBot-VLA* (depth-augmented) | 87.2 | 86.1 |
| **EvoScene-VLA** | **89.1** | **88.5** |

- Clean 提升：+1.9（vs LingBot-VLA*）
- **Rand 提升：+2.4**——比 Clean 更大

Paper 的解读：Rand 设置下初始 layout 变化更大，单 observation 的 perception 误差更大，跨 chunk 累积更严重——recurrent scene prefix 正好 mitigate 这两个问题。

这让我想到 [OpenVLA](https://arxiv.org/abs/2406.09246) 也在 distribution shift 下表现下降——但 OpenVLA 是单步 action，没有 chunk drift 问题。EvoScene-VLA 在 chunked setting 下专门解决 drift。

### 7.2 Ablation（Tab.2，RoboTwin-5Task）

| Variant | Clean | Rand |
|---------|-------|------|
| LingBot-VLA* | 87.8 | 84.6 |
| baseline（无任何 scene 模块） | 81.6 | 75.8 |
| + $\mathcal{L}_{\mathrm{pred}}$ & $\mathcal{L}_{\mathrm{rep}}$ | 89.3 | 86.2 |
| + $\mathcal{L}_{\mathrm{geo}}$ | 90.1 | 86.5 |
| + prior info at inference | **90.8** | **87.8** |

最后一行 "+prior info at inference" 是关键的 ablation：它对比 "每个 chunk 重新初始化 $\bar{s}_t$ from learnable embedding" vs "从上个 chunk 继承 denoised scene token"。**单独这一项贡献 +0.7/+1.3**——这是 recurrence 的纯增益，证明跨 chunk 的 scene prior 确实有用。

注意 baseline（81.6/75.8）比 LingBot-VLA*（87.8/84.6）还低——这是因为 baseline 是 "LingBot-VLA without depth" 的 EvoScene-VLA 训练 setting，不是 LingBot-VLA* 本身。所以正确的 ablation reading 是从 baseline 起算：
- + global anchor & scene predictor：+7.7/+10.4（巨大）
- + local depth anchor：+0.8/+0.3
- + recurrent prior at inference：+0.7/+1.3

### 7.3 Real-Robot（Tab.3，Galaxea R1-Lite）

| Task | $\pi_{0.5}$ | LingBot-VLA | LingBot-VLA* | EvoScene-VLA |
|------|------------|-------------|--------------|--------------|
| Mirror | 28 | 27 | 26 | 29 |
| Sink | 42 | 44 | 49 | 51 |
| Cutting-board | 44 | 34 | 37 | 46 |
| **Avg** | 38.0 | 35.0 | 37.3 | **42.0** |

- Avg 提升：+4.7（vs LingBot-VLA*）
- **Cutting-board 提升最大：+9**（37→46）

Cutting-board 任务最依赖 scene evolution——robot 用工具擦拭表面，surface state 随 action 演化，后续 control 必须瞄准未擦拭区域。这正是 EvoScene-VLA 设计针对的场景。Mirror 和 Sink 也是 cleaning 任务，但 surface state 变化没那么 subtle。

Real-robot 数据集（[Galaxea Open-World Dataset](https://arxiv.org/abs/2509.00576)）：
- 7 recording sessions
- 439 episodes
- 48,419 frames
- 1,756 video clips
- ~9 hours bimanual demos @ 15 fps
- Inference: single NVIDIA RTX 4090

### 7.4 Trajectory Quality（Fig.3, Fig.5）

Paper 还展示了 3D end-effector trajectories（Fig.3 四个 episode，Fig.5 四个 RoboTwin 任务：grab_roller, place_bread_skillet, place_burger_fries, place_cans_plasticbox）。

Qualitative observation：EvoScene-VLA 的 trajectory 明显 **更平滑**。Paper 的解释：co-denoising 把每个 predicted action tie 到 consistent scene context，避免 action-only chunked baseline 的 "abrupt corrections"。

这让我想到 [RDT-1B](https://arxiv.org/abs/2410.07864) 也观察到 diffusion-based policy 的 trajectory 比 action regression 平滑——但 RDT 的平滑来自 diffusion 的 trajectory-level modeling，EvoScene-VLA 的平滑来自 action-scene 的 co-conditioning。

---

## 8. Limitations & Open Questions

Paper 自陈的 limit：
1. **Latent state 不可解释**：recurrent state 是 latent，geometric content 只能通过 downstream behavior + ablation 间接判断。无法像 dense reconstruction 那样 visualize。
2. **Target quality upper-bound**：future scene supervision 来自 frozen 3DFM 在 future frames 上的 features。如果 3DFM 本身对未来帧的预测不准（occlusion、动态物体），prior 质量受限。
3. **Key-frame offset 固定**：训练只在 $\{k_1, k_2, k_3\}$ 上监督。如果 deployment 的 re-observation 间隔与这些 offset 不匹配，prior 在下个 chunk 会 temporally misaligned。
4. **Real-robot 评估局限**：只测了 3 个 indoor-cleaning 任务在单一 dual-arm 平台。category shift、novel scenes、multi-step task chaining 未测。

我自己的几个 questions：
- **Longer chunk 的影响**？Paper 在 Discussion 提到 "longer chunks may increase the value of recurrence by creating larger scene changes, but also push Scene Predictor targets farther into the future"。这是个 trade-off——$H=50$ 已经不算短，如果 $H=100$，scene drift 更大但 prediction target 更难。
- **Uncertainty signal**？Discussion 提到 "use the mismatch between observation slots and prior slots as an uncertainty signal for replanning or adaptive chunk execution"。这是个很有意思的方向——如果 observation slots（from 新 image）和 prior slots（from 上个 chunk 的 prediction）差异大，说明 prediction error 大，应该提前 re-observe 或 replan。这类似 active inference 的 prediction error 信号。
- **与 world model 的关系**？EvoScene-VLA 把 scene updater 嵌入 action decoder，没有单独的 online world model。但 [DreamerV3](https://arxiv.org/abs/2301.04104) 风格的 world model 在更 long-horizon 任务上可能更有优势——EvoScene-VLA 的 prior 是 chunk-level 的，Dreamer 是 step-level 的。

---

## 9. Broader Context & 个人联想

### 9.1 与 PerAct / RVT 系列的对比

[PerAct](https://arxiv.org/abs/2209.05451) 和 [RVT/RVT-2](https://arxiv.org/abs/2306.17896) 用 3D voxel 或 multi-view transformer 做 manipulation，但它们是 single-step action policies，没有 chunked control 的 drift 问题。EvoScene-VLA 的 recurrent scene prefix 可以看作 "在 chunked policy 中恢复 step-level scene awareness" 的努力——但用的是 latent prior 而非 dense voxel。

### 9.2 与 video prediction policies 的对比

[GR-2](https://arxiv.org/abs/2410.06158) 和 [Video Prediction Policy](https://arxiv.org/abs/2412.14803) 用 video generation 作为 policy 的 future conditioning。这些方法 generate pixel-level futures，信息量大但计算贵。EvoScene-VLA 选择 **compact latent**（16 tokens）而非 pixel-level future——这是 "policy-facing" 的关键：policy 不需要看见 future，只需要知道 future 的几何 state。

### 9.3 与 persistent reconstruction 系列的对比

[DUSt3R](https://arxiv.org/abs/2312.14132) → [MASt3R](https://arxiv.org/abs/2406.09756) → [CUT3R](https://arxiv.org/abs/2501.12387) 这条线 maintain state tokens for dense 3D reconstruction。EvoScene-VLA 借鉴了 "persistent state token across calls" 的思想，但：
- CUT3R 是 dense reconstruction，state token 编码 dense pointmap
- EvoScene-VLA 是 policy-facing，state token 编码 compact geometry prior

EvoScene-VLA 的 ablation 显示即使 16 tokens 也够用——因为 policy 不需要 dense geometry，只需要 task-relevant 的几何信息（gripper 与 target 的相对位姿、occluded region 的存在性等）。

### 9.4 Recurrence in VLA——更广的 trend

EvoScene-VLA 是 "把 recurrence 引入 VLA decoder" 的一个 instance。对比：
- [HAMLET](https://arxiv.org/abs/2510.00695)：history-aware attention，recurrence 在 attention 层
- [Embodied-SlotSSM](https://arxiv.org/abs/2511.11478)：slot state-space model，recurrence 在 SSM
- EvoScene-VLA：recurrence 在 action decoder 的 scene branch

三者解决不同问题：HAMLET 解决 observation history 的 attention 衰减，Embodied-SlotSSM 解决 object-centric state dynamics，EvoScene-VLA 解决 **action-updated scene prior 的跨 chunk 持久化**。

### 9.5 与 π3 的关系

[$\pi^3$](https://arxiv.org/abs/2503.18704)（ICLR 2026，permutation-equivariant visual geometry learning）是 EvoScene-VLA 的 global anchor teacher。$\pi^3$ 本身是一个 3D foundation model，permutation equivariant 意味着对输入视图顺序不变——这对 multi-view 3D 很自然。EvoScene-VLA 把 $\pi^3$ 的 features 蒸馏到 $s_p$ 的 16 tokens 中，让 policy 拿着一个紧凑的 $\pi^3$-aligned latent 作为 prior。

### 9.6 与 LingBot-VLA 的关系

EvoScene-VLA 是 [LingBot-VLA](https://arxiv.org/abs/2601.18692) 的扩展。LingBot-VLA 已经有 depth-augmented 版本（LingBot-VLA*）。EvoScene-VLA 在此基础上加 recurrent scene prefix，保持相同 training data / chunk length / compute budget——所以 gain 是 "纯 architectural" 的，不来自数据或算力。

---

## 10. 总结直觉

EvoScene-VLA 的核心 insight：**action decoder 是更新 scene state 的天然位置，因为它已经 generate 改变 scene 的 action 序列**。

这个 insight 重新定位了 action decoder 的角色：从 "motor command generator" 到 "motor command + scene state co-generator"。co-denoising 让两者在同一个 flow matching schedule 上协同，避免了独立预测的 inconsistency。

三个设计选择共同实现了这个 insight：
1. **Recurrent scene prefix**：定义 state 在哪里（prefix 的 prior slots）
2. **Two-level geometric anchor**：定义 state 编码什么（per-view depth + cross-view 3D）
3. **Scene Predictor + scene FM loss**：定义 state 如何跨时间 update（蒸馏 future scene 到 action expert）

Inference 时只保留 1 和 action expert 的 scene branch——所有 supervision infrastructure 在部署时丢弃。这是 "training scaffolding" 的典型 pattern：用复杂的训练时模块塑造一个简洁的 inference 模型。

实验证据支持这个设计：
- RoboTwin Rand 提升比 Clean 大（+2.4 vs +1.9）——prior 对 pose/layout variation 鲁棒
- Real-robot cleaning 任务提升最大——prior 对 evolving surface state 敏感
- Trajectory 更平滑——co-denoising 让 action 与 scene 一致

Paper 留下的最有趣 open question 是 **uncertainty signal from observation-prior mismatch**——如果这个 mismatch 能驱动 adaptive chunk execution 或 replanning，就把 EvoScene-VLA 从 "passive prior" 升级为 "active perception" 的 policy，那会是很自然的下一步。

---

## References

- [π0: A VLA flow model for general robot control](https://arxiv.org/abs/2410.24164)
- [OpenVLA: An open-source VLA model](https://arxiv.org/abs/2406.09246)
- [RT-2: VLA models transfer web knowledge to robotic control](https://arxiv.org/abs/2307.15818)
- [SpatialVLA: Exploring spatial representations for VLA](https://arxiv.org/abs/2501.15830)
- [QDepth-VLA: Quantized depth prediction as auxiliary supervision](https://arxiv.org/abs/2510.14836)
- [3D Diffusion Policy](https://arxiv.org/abs/2403.03954)
- [RoboTwin: Dual-arm robot benchmark](https://arxiv.org/abs/2409.02920)
- [Diffusion Policy: Visuomotor policy learning via action diffusion](https://arxiv.org/abs/2303.04137)
- [Flow matching for generative modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03022)
- [DUSt3R: Geometric 3D vision made easy](https://arxiv.org/abs/2312.14132)
- [MASt3R: Grounding image matching in 3D](https://arxiv.org/abs/2406.09756)
- [CUT3R: Continuous 3D perception model with persistent state](https://arxiv.org/abs/2501.12387)
- [MemoryVLA: Perceptual-cognitive memory in VLA](https://arxiv.org/abs/2508.19236)
- [HAMLET: History-aware VLA policy](https://arxiv.org/abs/2510.00695)
- [TraceVLA: Visual trace prompting for generalist robot policies](https://arxiv.org/abs/2412.10345)
- [AVA-VLA: Active visual attention for VLA](https://arxiv.org/abs/2511.18960)
- [FLARE: Robot learning with implicit world modeling](https://arxiv.org/abs/2505.15659)
- [UP-VLA: Unified understanding and prediction model](https://arxiv.org/abs/2501.18867)
- [RDT-1B: Diffusion foundation model for bimanual manipulation](https://arxiv.org/abs/2410.07864)
- [Octo: Open-source generalist robot policy](https://arxiv.org/abs/2405.12213)
- [Consistency Policy: Accelerated visuomotor policies](https://arxiv.org/abs/2405.07503)
- [HPT: Heterogeneous pre-trained transformers](https://arxiv.org/abs/2409.20537)
- [Galaxea Open-World Dataset and G0 dual-system VLA model](https://arxiv.org/abs/2509.00576)
- [DreamerV3: Mastering diverse domains through world models](https://arxiv.org/abs/2301.04104)
- [GR-2: Generative video-language-action model](https://arxiv.org/abs/2410.06158)
- [LingBot-VLA: A pragmatic VLA foundation model](https://arxiv.org/abs/2601.18692)
- [π³: Permutation-equivariant visual geometry learning](https://arxiv.org/abs/2503.18704)
- [PerAct: Perceiver-Actor multi-task transformer](https://arxiv.org/abs/2209.05451)
- [RVT-2: Learning precise manipulation from few demonstrations](https://arxiv.org/abs/2406.08545)
- [Masked Depth Modeling for Spatial Perception](https://arxiv.org/abs/2601.17895)
