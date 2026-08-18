---
source_pdf: Auto-JEPA A Latent World Model of Continuous Intent for End-to-End Autonomous
  Driving.pdf
paper_sha256: 961ec1e0009319aa3e5b5839a1857d520a2dddaf0ce4bbd5bd3a1a734a2aae58
processed_at: '2026-08-18T01:40:27-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换一种讲法，用更像聊天的语气把这篇 paper 的核心 idea 重新 build 一遍 intuition。

---

## 这群人到底想解决什么问题

你先想想现在 autonomous driving 的 world model 都在干嘛。主流做法是：给 model 当前帧，让它 predict 未来 4 秒的 BEV / occupancy / future video / multi-agent motion。听起来很合理——你 predict 出未来世界长啥样，planner 在这个未来里找一条好 trajectory。

但问题在于，**你逼 model 把整个未来场景都 reconstruct 出来，model 的大部分 capacity 被浪费在跟 ego 决策完全无关的东西上**。路边停着的车、远处天桥、对面车道那辆跟 ego 没交互的 truck——这些 pixel 在 future prediction 里占比巨大，但它们对 ego 接下来 4 秒怎么开几乎没影响。更要命的是，这些无关 prediction 的 error 会 propagate 进 planner，你 predict 错了一辆远处车的位置，planner 可能就乱了。

清华这波人提出一个很 LeCun 式的 thesis：

**planning 真正需要的信息，全在这条未来 ego trajectory 里。** 你不需要知道未来场景长什么样，你只需要知道"ego 接下来会怎么动"。因为这条 trajectory 已经把所有"会影响 ego 的 scene 信息"压缩进去了——前方 lead vehicle 距离、要 yield 的行人、navigation command、当前速度——这些全都会体现在 ego 未来 4 秒的 8 个 waypoint 上。

所以与其 predict future scene，不如直接 predict **future ego trajectory 的 latent representation**。这个 latent 就是 "driving intent"——ego 接下来的连续运动意图。

---

## 核心赌注：让 action target 自动 supervise perception

这是整篇 paper 最聪明的地方，我用一个例子讲。

假设你训一个 model，input 是当前 image + ego history + navigation command，output 是 8 个 waypoint 的 latent。loss 是让 predicted latent 跟 ground-truth trajectory 的 latent 对齐。

现在你想：前方 lead vehicle 的 pixel 对这个 prediction 有影响吗？有，而且很大——因为 lead vehicle 距离决定了 ego 要不要减速，减速就改变了 future trajectory，trajectory 改变了 target latent，target latent 改变了 loss。所以 lead vehicle pixel 的 gradient 自然大。

那旁边车道那辆跟 ego 无交互的车呢？它的 pixel 变化会改变 future trajectory 吗？不会。所以 gradient 接近 0，model 学到 ignore 它。

**你完全没给 model 任何 object box、agent identity、interaction label。但 model 自发地学到了 "lead vehicle 重要，旁边无关车不重要"。** 这种 selective attention 完全是从 action target 的 gradient 里 emerge 出来的。

paper 第 7 节做了 occlusion 实验验证这个赌注，结果非常干净：mask 掉所有 dynamic agent 的 region，predicted intent 的变化是等面积 random mask 的 2.97 倍。而且 71.1% 的 scene 里，mask 动态 agent 的影响比 random 大。Figure 1 的 qualitative 例子更直观——同一帧画面，你 occlude 掉跟 ego 有交互的 lead vehicle，predicted intent 和最终 trajectory 大幅 shift；你 occlude 掉旁边那辆不交互的车，intent 几乎不变。

这就是 "learning to see by learning to act" 的实证——action-relevant 的 perception 可以从 action supervision 里被 implicit supervise，不需要显式 annotation。

---

## 整个 pipeline 长啥样

我用最简单的话过一遍 dataflow：

### 训练阶段

**Stage 1：先训一个 trajectory autoencoder。**

输入是 ground-truth 的 8 个 waypoint $\mathbf{Y} = [(x_1,y_1), \dots, (x_8,y_8)] \in \mathbb{R}^{8\times 2}$，覆盖未来 4 秒，每 0.5 秒一个点。坐标先除以 64 归一化。

Trajectory encoder $E_{\text{traj}}$ 把它编码成 8 个 latent token $\mathbf{Z}^+ \in \mathbb{R}^{8\times 1024}$，decoder 再 reconstruct 回来。Loss 包括坐标、终点、速度、加速度四项：

$$\mathcal{L}_{\text{traj}} = \mathcal{L}_{xy} + 2.0\mathcal{L}_{\text{end}} + 0.5\mathcal{L}_{\text{vel}} + 0.2\mathcal{L}_{\text{acc}}$$

- $\mathcal{L}_{xy}$：8 个 waypoint 坐标 error
- $\mathcal{L}_{\text{end}}$：终点 FDE，权重最大（2.0），因为 planning 终点准最关键
- $\mathcal{L}_{\text{vel}}$：速度 consistency
- $\mathcal{L}_{\text{acc}}$：加速度 consistency，隐含 comfort 约束

训完之后 **decoder 扔掉，encoder freeze 住**。这个 frozen encoder 接下来干两件事：定义训练的 target latent，以及编码 trajectory memory 里的每条 candidate。

**Stage 2：训 visual intent predictor。**

Input 三样东西：
- $\mathbf{I}$：4 帧 front camera 256×256
- $\mathbf{H} \in \mathbb{R}^{4\times 2}$：4 个 historical ego position
- $\mathbf{C} \in \mathbb{R}^4$：navigation command

Visual encoder 用 **frozen V-JEPA 2**（LeCun 那篇 https://arxiv.org/abs/2506.09985），输出 visual token。history 和 command 各过一个 MLP 投到 1024 维。然后一个 24-layer Transformer predictor（hidden 1024, 16 heads）fuse 这三路，输出 8 个 future temporal token：

$$\hat{\mathbf{Z}} = P_\theta(\mathbf{F}_v, \mathbf{F}_h, \mathbf{F}_c) \in \mathbb{R}^{8\times 1024}$$

注意这 8 个 token **不是 8 个离散 maneuver class**（左转/右转/直行…），而是 8 个 continuous time step 的 latent，jointly 描述一条连续 future realization。这是 "continuous intent" 这个名字的来源——它是一个 continuous latent sequence，不是 discrete action token。

训练 loss 三个部分：

$$\mathcal{L}_{\text{intent}} = 0.1\mathcal{L}_{\text{feat}} + 2.0\mathcal{L}_{\text{cos}} + \mathcal{L}_{\text{NCE}}$$

**Feature alignment**（权重 0.1）：
$$\mathcal{L}_{\text{feat}} = \text{SmoothL1}(\text{Norm}(\hat{\mathbf{Z}}), \text{Norm}(\mathbf{Z}^+))$$
对齐 magnitude，SmoothL1 对 outlier 鲁棒。

**Token-wise cosine**（权重 2.0，最重要）：
$$\mathcal{L}_{\text{cos}} = \frac{1}{8}\sum_{t=1}^{8}\left(1 - \frac{\hat{\mathbf{z}}_t^\top \mathbf{z}_t^+}{\|\hat{\mathbf{z}}_t\|_2 \|\mathbf{z}_t^+\|_2}\right)$$
- 下标 $t$ 是 time step index
- 强制每个 time step 的方向对齐，preserve temporal order
- 没这一项的话，"先减速再转弯" 和 "先转弯再减速" 这种 temporal order 不同的 trajectory 会被洗成相似 latent

**Batch-level InfoNCE**（权重 1.0，防 collapse）：
$$\mathcal{L}_{\text{NCE}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\hat{\mathbf{q}}_i^\top \mathbf{k}_i / \tau)}{\sum_{j=1}^{B}\exp(\hat{\mathbf{q}}_i^\top \mathbf{k}_j / \tau)}$$
- $\hat{\mathbf{q}}_i$ 是 scene $i$ 的 predicted intent flatten + normalize
- $\mathbf{k}_j$ 是 scene $j$ 的 ground-truth latent flatten + normalize
- $\tau = 0.07$ temperature，distributed training 时 $\mathbf{k}_j$ 跨 GPU gather 扩大 negative pool
- **为啥需要这个？** 光做 positive alignment 会让所有 scene 映射到同一个点（loss 也是 0）。InfoNCE 强制不同 scene 的 trajectory latent 必须可区分，是防 representation collapse 的标准配置

### 推理阶段

1. 拿 $\hat{\mathbf{Z}}$ flatten + L2 normalize 成 query $\mathbf{q}$
2. 在 **110,335 条 ground-truth trajectory** 的 memory 里做 flat cosine top-300
3. Scene-conditioned scorer $S_\phi$ 给每个 candidate 打分（从 CLOVER scorer 初始化，https://arxiv.org/abs/2605.15120）
4. DAC gate $G_\psi$ 预测每个 candidate 的 drivable-area failure probability，threshold $\tau_{\text{DAC}}=0.2$ mask 掉高风险 candidate
5. 剩下 candidate 里 argmax scorer 分数，输出对应 waypoint

公式：
$$r_n = \mathbf{q}^\top \mathbf{m}_n, \quad \mathcal{C} = \text{TopK}(\{r_n\}, K=300)$$
$$s_k = S_\phi(\mathbf{F}_{\text{scene}}, \mathbf{e}, \mathbf{Y}_k)$$
$$p_k^{\text{DAC}} = G_\psi(\mathbf{F}_{\text{scene}}, \mathbf{e}, \mathbf{Y}_k)$$
$$m_k = \mathbf{1}[p_k^{\text{DAC}} \le 0.2]$$
$$k^* = \arg\max_{k: m_k=1} s_k, \quad \mathbf{Y}^* = \mathbf{Y}_{k^*}$$

**关键点：没有 learned trajectory generator，没有 iterative diffusion sampling，没有 waypoint regression head。** Trajectory geometry 完全来自 memory，model 只负责 "选哪条"。如果所有 candidate 都被 gate 拒了，fallback 到 ungated scorer ranking，保证 output non-empty。

---

## 为什么把 pipeline 拆成四块

这是我看完最欣赏的设计哲学。四个模块各有正交责任：

1. **Intent predictor** 决定 "what kind of future motion is appropriate"——latent 的、抽象的、scene-conditioned
2. **Memory retrieval** 提供 explicit trajectory geometry——non-parametric，无需 learn，可解释
3. **Scorer** estimate scene-level driving quality——collision、TTC、comfort、progress
4. **Gate** hard feasibility filter——drivable-area violation 直接 mask

为什么不让 intent predictor 直接 regress waypoint？因为 image→coordinate 的 precise mapping 很难，perspective、occlusion 都会引入 error。让 predictor 只学 latent，把 precise geometry 交给 memory，是分工。

为什么不用 scorer 生成 trajectory？因为 scorer 是 evaluator，逼它 generate 会 mode collapse（典型 GAN discriminator 问题）。memory 天然 multi-modal（ground-truth 数据里各种 maneuver 都有），scorer 只做 ranking。

为什么 gate 必须独立？如果把 DAC 检测塞进 scorer，scorer 可能为了一条 high-utility 但 borderline DAC 的 candidate 给很高分。独立 gate + hard threshold 保证 safety 不被 utility trade off。

---

## 结果好到什么程度

### NAVSIM v1 navtest（12,146 scenarios）

Auto-JEPA **只用 1 个 front camera** 拿到 **91.3 PDMS**。对比一下：

| Method | Sensors | PDMS |
|---|---|---|
| TransFuser | 3×C+L | 84.0 |
| Hydra-MDP | 3×C+L | 86.5 |
| DiffusionDrive | 3×C+L | 88.1 |
| AutoVLA | 3×C | 89.1 |
| Curious-VLA | 1×C | 90.3 |
| **Auto-JEPA** | **1×C** | **91.3** |
| Human | — | 94.8 |

几个亮点：
- 单 camera 干翻一堆 3 camera + LiDAR 的方法
- DAC 98.3 几乎是表里最高（gate + memory 防御很强）
- EP 87.1 接近 human 87.5（没学到 "原地不动" 的 degenerate 解，progress 积极）
- Comfort 100.0（memory 里都是 ground-truth，天然 smooth）

PDMS 公式：$\text{PDMS} = \text{NCDAC} \times \frac{5(\text{EP}+\text{TTC}) + 2\text{C}}{12}$

- NCDAC 是 NC 和 DAC 的复合 hard gate，乘性。一旦撞车或越界，整体分数被压到接近 0
- EP 和 TTC 各占 5/12，C 占 2/12
- NAVSIM 故意把 EP 权重设大，防 model 学 "原地不动最安全"

### NAVSIM v2 EPDMS

updated official implementation + human-behavior filtering 下 **89.1 EPDMS**。对比 CLOVER 90.4（用 learned generator + scorer pipeline），Auto-JEPA 无 parametric generator 差距只有 1.3 分。

### Ablation

| Intent | Scorer | Gate | PDMS |
|---|---|---|---|
| ✗ | ✓ | ✓ | 52.6 |
| ✓ | ✗ | ✓ | 87.6 |
| ✓ | ✓ | ✗ | 91.0 |
| ✓ | ✓ | ✓ | 91.3 |

- **关掉 intent（用固定 codebook medoid 当 query）→ 52.6**：暴跌 38.7 分。证明 predicted intent 确实 scene-conditioned，不是 random
- **关掉 scorer → 87.6**：scorer 贡献 3.7 分。retrieval 给的 300 candidate 包含正确答案，scorer 把它挑出来
- **关掉 gate → 91.0**：gate 只贡献 0.3 分 PDMS，但 DAC 从 98.3 降到 97.9。gate 主要修 safety edge case

Candidate pool size：
- K=1：87.6
- K=200：91.1
- K=300：91.3

K=1→200 提升 3.5 分，200→300 只提升 0.2 分。**Saturation 在 K≈200**。这告诉我 bottleneck 不是 scorer，是 **memory recall**——再放大 pool 也捞不到新的正确 candidate，因为 memory 里就没有。

---

## 这套思路跟别人的区别

**vs. V-JEPA 2 / I-JEPA / V-JEPA**：原版 JEPA 的 target 是 future image/video patch 的 latent。Auto-JEPA 把 target 推到 future ego trajectory latent，从 scene-centric 变成 action-centric。而且 JEPA prediction 直接当 retrieval key 参与推理，不只是 pretraining backbone。

**vs. Drive-JEPA**（https://arxiv.org/abs/2601.22032）：同样 JEPA + driving，但 Drive-JEPA 主要做 video pretraining + trajectory distillation，JEPA prediction 在推理时只是 representation。Auto-JEPA 的 prediction 就是 planner 输出。

**vs. LAW / World4Drive / DeepSight / DriveWorld-VLA**（latent world model 系）：这些方法 predict future **scene** latent，target 描述 "surrounding scene 怎么 evolve"。Auto-JEPA 的 target 是 "ego 怎么 move"，scene dynamics 只通过它对 ego trajectory 的影响被 retain。

**vs. DiffusionDrive / GoalFlow / VADv2**（parametric candidate generation）：这些用 diffusion / flow matching / vectorized vocabulary 生成 candidate。Auto-JEPA 用 non-parametric retrieval 替代 generator。好处是无需训 generator、可解释、易扩展；坏处是 memory coverage 是 hard bound。

**vs. VLA**（AutoVLA / DriveVLA / Curious-VLA）：VLA 把 action tokenize 成 discrete token autoregressive decode。Auto-JEPA predict continuous intent latent + non-parametric retrieval，trajectory geometry 不经过 quantization error。

**vs. CLOVER**（https://arxiv.org/abs/2605.15120）：CLOVER 是 learned generator + scorer。Auto-JEPA 直接 reuse CLOVER 的 scorer 当初始化，去掉 generator 用 retrieval 替代。差 1.3 EPDMS 但省了 generator 训练成本。

---

## 我觉得 limitation 在哪

paper 自己提的：
1. **Memory coverage bound**：feasible maneuver 不在 memory 就 retrieve 不到。K=200→300 saturation 佐证
2. **Selection calibration**：scorer 可能 misrank，gate 可能误拒
3. **No scene-level forecast**：不能做 interactive simulation / counterfactual

我自己加几个 intuition：

**Memory 是离散 manifold 的 sample，intent 是 continuous。** 当 predicted intent 落在 memory 两个 candidate 之间时，retrieval 只能给最近那个，丢了 interpolate 能力。一个 fix 是 retrieve top-1 后用 predicted intent 当 condition 做 small diffusion / flow 把 candidate 微调到 intent 附近，突破 memory coverage limit 同时保持 explicit geometry 可解释性。

**InfoNCE 的 negative 是 batch 内其他 scene，但同一 scene 的多 modal trajectory（左转 vs 直行都可能）会被当 negative push 开。** 这跟 multi-modal planning 初衷有 tension。可以用 cluster-based negative 或 hard negative mining，把 "不同 scene 但 similar intent" 当 negative，"同 scene 不同 modal" 不当 negative。

**Visual encoder frozen 是双刃剑。** V-JEPA 2 的 self-supervised representation 对 general visual feature 好，但对 driving-specific 的 fine-grained（车道线、traffic light、small obstacle）可能不够。NAVSIM v2 的 TL 97.2→99.7 主要是 evaluator change，但 TL 本身不是最高分，可能跟 frozen encoder 对 traffic light sensitivity 不足有关。Unfreeze 最后几层做 LoRA fine-tune 可能 help。

**跟 LeCun H-JEPA 的联想。** LeCun 一直推 hierarchical predictive model：高层 predict abstract plan，低层 predict pixel。Auto-JEPA 只做了 "高层"——predict intent latent。如果想加 simulation / counterfactual 能力，得把 "predict future video given intent" 的 low-level head 加回来，就是完整 H-JEPA。

**Retrieval 跟 RAG 的类比。** 这其实是 trajectory 版的 retrieval-augmented generation。Query = intent latent，memory = ground-truth trajectory，retriever = flat cosine。跟 LLM RAG 的区别是 LLM RAG retrieve text chunk 后 concatenate 进 prompt 再 generate；Auto-JEPA retrieve trajectory 后直接用。一个 hybrid 是 retrieve top-K 然后 scorer 当 reranker——跟 LLM RAG 的 reranker pipeline 几乎同构。未来可以用 learned retriever、cross-encoder reranker、HNSW ANN upgrade。

**"Continuous intent" 跟 motor cortex / affordance 的类比。** 8-token continuous latent 很像神经科学里的 "motor program"——连续的、time-indexed 的运动计划 latent。Driving 这种 continuous control task 天然适合 continuous intent，比 discrete action token（VLA 那套）的 "discrete motor command" 更 smooth、可 interpolate、对 small motion change 敏感。

---

## 一句话 takeaway

Auto-JEPA 把 driving world model 的 prediction target 从 "future scene" 收窄到 "future ego trajectory latent"，让 JEPA predictor 只学 "这个 scene 意味着 ego 怎么动"。然后用 predicted latent 当 retrieval key 从 ground-truth trajectory memory 捞 300 个 candidate，再用 scorer + gate 做 scene-conditioned selection。

整个 pipeline 没有 dense future reconstruction、没有 learned generator、没有 explicit perception annotation，但通过 action-target supervision 自发 emerge 出 planning-relevant selective attention，单 camera 在 NAVSIM v1 拿 91.3 PDMS。

这个工作给 "world model 不需要 reconstruct 完整世界，只需要 predict action-relevant latent" 这个哲学提供了一个非常干净的 driving 实证。代码在 https://github.com/NoctYang/Auto-JEPA 开源。

---

## 关键 reference 链接

- Auto-JEPA 代码：https://github.com/NoctYang/Auto-JEPA
- V-JEPA 2（visual encoder backbone）：https://arxiv.org/abs/2506.09985
- I-JEPA：https://arxiv.org/abs/2301.08243
- V-JEPA：https://arxiv.org/abs/2404.08471
- NAVSIM benchmark：https://arxiv.org/abs/2406.13359
- NAVSIM repo：https://github.com/autonomousvision/navsim
- CLOVER（scorer 初始化来源）：https://arxiv.org/abs/2605.15120
- Drive-JEPA（对照）：https://arxiv.org/abs/2601.22032
- LAW（latent world model baseline）：https://arxiv.org/abs/2412.15215
- Hydra-MDP（candidate scoring baseline）：https://arxiv.org/abs/2406.06978
- DiffusionDrive：https://arxiv.org/abs/2411.15139
- VADv2：https://arxiv.org/abs/2402.13243
- InfoNCE / CPC：https://arxiv.org/abs/1807.03748
- AutoVLA：https://arxiv.org/abs/2412.14349
- DriveVLA-W0：https://arxiv.org/abs/2601.01403
- Curious-VLA：https://arxiv.org/abs/2603.06049
- PRANK（latent retrieval for motion forecasting）：https://arxiv.org/abs/2010.06945
- Vista（generative driving world model）：https://arxiv.org/abs/2401.02977
- GAIA-1：https://arxiv.org/abs/2309.17080
- TransFuser：https://arxiv.org/abs/2205.15997

---

先给你一句话的 thesis，然后慢慢 build up intuition。

**Auto-JEPA 的核心 thesis：driving world model 根本不需要 reconstruct 完整的未来世界（future video / occupancy / BEV / multi-agent motion），它只需要 predict 一个"未来 ego trajectory 的 latent representation"，因为这个 latent 已经压缩了所有会影响 ego action 的 scene 信息。把 prediction target 收窄到 future ego motion 本身，model 就会自发地把 capacity 分配给 planning-relevant 的 visual feature，而不是浪费在 reconstruct 那些跟 ego 决策无关的 background / 远处车辆上。**

这个 idea 非常 LeCun-style JEPA，但做了一个关键 shift：原版 I-JEPA / V-JEPA 的 target 是 "future image/video patch 的 latent"，Auto-JEPA 把 target 进一步推到 "future ego trajectory 的 latent"。换句话说，predictor 学的是 "这个 scene 意味着 ego 接下来会怎么动"，而不是 "这个 scene 接下来会长什么样"。这是从 scene-centric world model 到 action-centric world model 的转变。

---

## 1. The big intuition：为什么 predict future trajectory latent 就够了

你想想，planning 真正需要的信息是什么？是 ego 未来 4 秒要走的那条曲线 $\mathbf{Y} \in \mathbb{R}^{8\times 2}$。这条曲线完全由"会影响这条曲线的 scene feature"决定——前方 lead vehicle 的距离/速度、要 yield 的行人、navigation command、ego 当前速度。路边停着的车、远处跟 ego 无交互的车、天空纹理，对 $\mathbf{Y}$ 几乎没影响。

如果你让 model 去 reconstruct future BEV / occupancy / video，你强制它把 capacity 分给那些 irrelevant pixel。error 会从这些 irrelevant prediction propagate 进 planner。Auto-JEPA 的赌注是：**直接把 $\mathbf{Y}$ 的 latent 当 prediction target，loss gradient 会自动 teach visual encoder "哪些 pixel 对 $\mathbf{Y}$ 有贡献"**。第 7 节的 occlusion 实验就是在验证这个赌注——结果非常漂亮：mask 掉 dynamic agent 的 region，intent latent 的 change 是等面积 random mask 的 2.97×，而且 71.1% 的 scene 里 dynamic-agent mask 影响更大。关键点是：**model 从来没见过 object box / agent identity / interaction label**，这种 selective attention 完全是从 "predict future ego trajectory" 这个 supervision 里 emerge 出来的。

这跟你以前在 "learning to see by learning to act" 上的 intuition 是一致的——action-relevant 的 perception 不需要显式 annotation，它可以从 action target 的 gradient 里被 implicit supervise。

---

## 2. 架构总览：两个 stage + 一个非参数 retrieval

整个 pipeline 可以拆成 4 个模块，我按 dataflow 讲：

### Stage 1: Trajectory latent space pretraining（task-specific target space）
先用一个 trajectory autoencoder 学一个 $\mathbf{Y} \to \mathbf{Z}^+ \to \hat{\mathbf{Y}}$ 的 bottleneck。训完 discard decoder，freeze encoder $E_{\text{traj}}$。这个 frozen encoder 干两件事：
1. 定义 training 的 target latent $\mathbf{Z}^+ = E_{\text{traj}}(\mathbf{Y}_{\text{gt}})$；
2. 把 trajectory memory 里每条 ground-truth trajectory 编成 key latent。

这是非常关键的 design choice：**target space 和 retrieval space 用同一个 encoder**，保证 predicted intent $\hat{\mathbf{Z}}$ 和 memory key $\mathbf{Z}_n$ 在同一个 metric space 里，cosine similarity 才有意义。

### Stage 2: Visual intent predictor（JEPA predictor）
Input：
- $\mathbf{I}$：4 帧 front camera（256×256）；
- $\mathbf{H} \in \mathbb{R}^{4\times 2}$：4 个 historical ego position；
- $\mathbf{C} \in \mathbb{R}^4$：navigation command（one-hot-ish）。

Visual encoder 用 **frozen V-JEPA 2**（LeCun 那篇 arXiv:2506.09985），输出 visual token $\mathbf{F}_v$。History 和 command 各过一个 MLP 投到 1024 维。然后一个 24-layer Transformer predictor（hidden 1024, 16 heads）fuse 这三路，输出 8 个 future temporal token $\hat{\mathbf{Z}} \in \mathbb{R}^{8\times 1024}$。

注意这 8 个 token **不是 8 个离散 maneuver class**（左转/右转/直行/...），而是 8 个 continuous time step 的 latent， jointly 描述一条 continuous future realization。这是 "continuous intent" 这个名字的来源——intent 是一个 continuous latent sequence，不是 discrete action token。

### Inference: Non-parametric retrieval + scorer + gate
1. 拿 $\hat{\mathbf{Z}}$ flatten + L2 normalize 成 query $\mathbf{q}$；
2. 在 110,335 条 ground-truth trajectory 的 memory 里做 flat cosine top-300；
3. Scene-conditioned scorer $S_\phi$ 给每个 candidate 打分（从 CLOVER scorer 初始化）；
4. DAC gate $G_\psi$ 预测每个 candidate 的 drivable-area failure probability，threshold $\tau_{\text{DAC}}=0.2$ mask 掉高风险 candidate；
5. 在剩下的 candidate 里 argmax scorer 分数，输出对应 waypoint $\mathbf{Y}^*$。

**没有 learned trajectory generator，没有 iterative diffusion sampling，没有 waypoint regression head。** Trajectory geometry 完全来自 memory，model 只负责 "选哪条"。

---

## 3. 公式逐个拆：变量、上下标、设计意图

### Trajectory 表示
$$\mathbf{Y} = [(x_1,y_1),\dots,(x_8,y_8)] \in \mathbb{R}^{8\times 2}$$
- 下标 $t \in \{1,\dots,8\}$：time step，覆盖 4s horizon，间隔 0.5s；
- $(x_t, y_t)$：ego 中心在 ego-centric planar coordinate 的位置；
- 坐标先除以 scale factor 64 归一化（supplementary 里写的），让数值进 O(1) 量级，便于 Transformer 处理。

### Trajectory autoencoder
$$\mathbf{Z}^+ = E_{\text{traj}}(\mathbf{Y}) \in \mathbb{R}^{8\times 1024}, \quad \hat{\mathbf{Y}} = D_{\text{traj}}(\mathbf{Z}^+)$$
- $\mathbf{Z}^+$：上标 $+$ 表示 "target / ground-truth side"（JEPA 论文里习惯用 $+$ 标 target context）；
- 8 个 token 对应 8 个 time step，1024 维 hidden；
- $E_{\text{traj}}$ 是 4 个 Transformer block + 8 个 Fourier frequency band 做 coordinate encoding（Fourier band 帮助 encode 低 amplitude 高频的 trajectory shape 细节，避免 MLP 对 low-frequency bias）；
- $D_{\text{traj}}$ 预测 waypoint **increment**，cumulative sum 回 trajectory（这种 residual / delta parameterization 在 trajectory prediction 里很常见，能让 model 学相对运动而不是绝对坐标，对 ego frame 的 translation invariance 更好）。

### Trajectory reconstruction loss
$$\mathcal{L}_{\text{traj}} = \mathcal{L}_{xy} + \lambda_e \mathcal{L}_{\text{end}} + \lambda_v \mathcal{L}_{\text{vel}} + \lambda_a \mathcal{L}_{\text{acc}}$$
- $\mathcal{L}_{xy}$：8 个 waypoint 坐标的 reconstruction error；
- $\mathcal{L}_{\text{end}}}$：final endpoint（FDE-style，强调长 horizon 准确性）；
- $\mathcal{L}_{\text{vel}}$：finite-difference velocity consistency；
- $\mathcal{L}_{\text{acc}}$：finite-difference acceleration consistency（隐含 comfort / jerk 约束）；
- supplementary 给的实际权重：$\lambda_e=2.0, \lambda_v=0.5, \lambda_a=0.2$。endpoint 权重最大，因为 planning 的最终判断往往看终点是否正确（NAVSIM 的 EP metric 也跟 endpoint 强相关）。

### Intent predictor output
$$\hat{\mathbf{Z}} = P_\theta(\mathbf{F}_v, \mathbf{F}_h, \mathbf{F}_c) \in \mathbb{R}^{8\times 1024}$$
- hat 表示 predicted；
- $P_\theta$：24-layer Transformer，$\theta$ 是 trainable 参数；
- 输出 shape 故意跟 $\mathbf{Z}^+$ 完全对齐，方便 token-wise alignment。

### Feature alignment loss
$$\mathcal{L}_{\text{feat}} = \text{SmoothL1}(\text{Norm}(\hat{\mathbf{Z}}), \text{Norm}(\mathbf{Z}^+))$$
- Norm 是 L2 normalize（把每个 8×1024 张量按 element-wise 或整体 normalize，paper 没完全说清，从 context 看是整体 flatten 后 normalize）；
- SmoothL1 比 L2 对 outlier 更鲁棒，比 L1 在 near-zero 更 differentiable。

### Token-wise cosine alignment
$$\mathcal{L}_{\text{cos}} = \frac{1}{8}\sum_{t=1}^{8}\left(1 - \frac{\hat{\mathbf{z}}_t^\top \mathbf{z}_t^+}{\|\hat{\mathbf{z}}_t\|_2 \|\mathbf{z}_t^+\|_2}\right)$$
- 下标 $t$：time step index；
- 这一项保证 **每个 time step 的方向对齐**，不光是整体 flatten 对齐。这很关键，因为 trajectory 的 temporal structure（先直行再左转 vs 先左转再直行）需要在 token level 区分，flatten 的 feature alignment 会把 temporal order 洗掉。

### Batch-level InfoNCE
$$\hat{\mathbf{q}}_i = \text{Norm}(\text{vec}(\hat{\mathbf{Z}}_i)), \quad \mathbf{k}_j = \text{Norm}(\text{vec}(\mathbf{Z}_j^+))$$
$$\mathcal{L}_{\text{NCE}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\hat{\mathbf{q}}_i^\top \mathbf{k}_i / \tau)}{\sum_{j=1}^{B}\exp(\hat{\mathbf{q}}_i^\top \mathbf{k}_j / \tau)}$$
- 下标 $i, j$：batch 内 scene index；
- $\tau = 0.07$：temperature，越小 contrastive 越尖锐（focus on hard negative）；
- distributed training 时 $\mathbf{k}_j$ 跨 GPU gather，扩大 negative pool——这是 SimCLR / MoCo 系的标准 trick，对避免 representation collapse 至关重要。

**为什么需要 InfoNCE？** 单纯的 positive alignment（feature + cosine）会 collapse——所有 scene 映射到同一个 latent point 也能让 alignment loss = 0。InfoNCE 强制 "different scene 的 trajectory latent 必须可区分"，相当于一个 contrastive regularizer。这是 JEPA 系方法防 collapse 的标准配置（I-JEPA 也靠 EMA target + 高维空间 + batch negative）。

### Total intent loss
$$\mathcal{L}_{\text{intent}} = 0.1\mathcal{L}_{\text{feat}} + 2.0\mathcal{L}_{\text{cos}} + \mathcal{L}_{\text{NCE}}$$
- cosine 权重 2.0 最大，说明作者认为 temporal-token-level 的方向对齐是核心；
- NCE 权重 1.0 作为 regularizer；
- feat 权重 0.1 最小， magnitude 信息没那么重要（因为后面 retrieval 也 normalize）。

### Retrieval
$$\mathbf{q} = \text{Norm}(\text{vec}(\hat{\mathbf{Z}})), \quad \mathbf{m}_n = \text{Norm}(\text{vec}(\mathbf{Z}_n))$$
$$r_n = \mathbf{q}^\top \mathbf{m}_n$$
$$\mathcal{C} = \text{TopK}(\{r_n\}_{n=1}^{N}, K), \quad N=110{,}335, K=300$$
- 下标 $n$：memory entry index；
- flat cosine：把 8×1024 flatten 成 8192 维向量再算 cosine。这意味着 temporal order 信息被 encode 在 8192 维的位置里（因为 vec 是按 (t, dim) 顺序 flatten 的），normalize 后 cosine 度量的是 "两个 trajectory 在这个 8192 维超球上的夹角"。

### Scorer + Gate + Final selection
$$s_k = S_\phi(\mathbf{F}_{\text{scene}}, \mathbf{e}, \mathbf{Y}_k)$$
$$p_k^{\text{DAC}} = G_\psi(\mathbf{F}_{\text{scene}}, \mathbf{e}, \mathbf{Y}_k)$$
$$m_k = \mathbf{1}[p_k^{\text{DAC}} \le \tau_{\text{DAC}}], \quad \tau_{\text{DAC}}=0.2$$
$$k^* = \arg\max_{k: m_k=1} s_k, \quad \mathbf{Y}^* = \mathbf{Y}_{k^*}$$
- 下标 $k$：candidate index in retrieved pool；
- $\mathbf{e}$：ego context（速度、历史位姿等）；
- $\mathbf{1}[\cdot]$：indicator function，gate 通过的 candidate 才进 argmax；
- 如果所有 candidate 都被 gate 拒绝，fallback 到 ungated scorer ranking——保证 non-empty output。

### PDMS metric（NAVSIM v1）
$$\text{PDMS} = \text{NCDAC} \times \frac{5(\text{EP}+\text{TTC}) + 2\text{C}}{12}$$
- NCDAC：NC（no-at-fault collision）和 DAC（drivable-area compliance）的复合 hard gate，乘性。一旦撞车或越界，整体分数被压到接近 0；
- EP：ego progress（不能为了安全原地不动）；
- TTC：time-to-collision（跟其他 agent 的最小 TTC）；
- C：comfort（jerk / acceleration 平滑度）；
- 权重 5,5,2 归一化到 12，意思是 EP 和 TTC 各占 5/12，C 占 2/12。NAVSIM 故意把 EP 权重设大，防止 model 学到 "原地不动最安全" 的 degenerate 解。

### Intent change metric（occlusion 实验）
$$\Delta_{\text{intent}} = 1 - \cos(\hat{\mathbf{Z}}, \hat{\mathbf{Z}}_m)$$
- $\hat{\mathbf{Z}}_m$：masked input 的 predicted intent；
- 1 - cosine：0 表示完全不变，2 表示完全反向。0.080 vs 0.027 的 absolute 数值不大，但 ratio 2.97× 说明 selective sensitivity。

---

## 4. Training objectives 的分工直觉

这三个 loss 我觉得设计得很精巧，各有分工：

| Loss | 作用 | 不加会怎样 |
|---|---|---|
| $\mathcal{L}_{\text{feat}}$ (SmoothL1) | 对齐 magnitude | magnitude 漂移，retrieval scale 不稳 |
| $\mathcal{L}_{\text{cos}}$ (token-wise) | 对齐每个 time step 的方向 | temporal order 丢失，先左转再直行 vs 先直行再左转 分不开 |
| $\mathcal{L}_{\text{NCE}}$ (batch) | 防止 representation collapse | 所有 scene 映射到同一点，retrieval 退化 |

我特别想强调 $\mathcal{L}_{\text{cos}}$ 的 token-wise 设计。如果只 flatten 后做 cosine，"先减速再转弯" 和 "先转弯再减速" 这种 temporal order 不同的 trajectory 可能 cosine 相似度很高（因为同样的 token set，只是顺序不同），但它们是两条不同的 candidate。token-wise cosine 强制 $t=1$ 的 predicted token 对齐 $t=1$ 的 target token，temporal structure 才被 preserve。

---

## 5. Retrieval + Scorer + Gate 的分工直觉

这是我最喜欢的部分，因为它把 planning 拆成了四个正交 responsibility：

1. **Intent predictor**：决定 "what kind of future motion is appropriate"——这一步是 latent 的，抽象的，scene-conditioned；
2. **Memory retrieval**：提供 explicit trajectory geometry——non-parametric，无需 learn，可解释；
3. **Scorer**：estimate scene-level driving quality——collision、TTC、comfort、progress，是 "在这条 candidate 上跑会怎样" 的 value estimation；
4. **Gate**：hard feasibility filter——drivable-area violation 直接 mask。

为什么要这么拆？我的 intuition 是：

**Intent predictor 不擅长 fine-grained geometry。** 它是从 image token 到 latent 的 mapping，让它直接 regress waypoint 坐标会逼它学一个 high-precision decoder，但 image→coordinate 的 precise mapping 很难（perspective、occlusion 都会引入 error）。让它只 predict latent，把 precise geometry 交给 memory，是分工。

**Scorer 不擅长 "想出" trajectory。** Scorer 是个 evaluator，给它一条 trajectory 它能估 quality，但你让它 generate trajectory 它会 mode collapse（典型 GAN discriminator 问题）。所以 generate 交给 memory（ground-truth trajectory 天然 multi-modal），scorer 只做 ranking。

**Gate 必须独立。** 如果把 DAC 检测塞进 scorer，scorer 可能为了一条 high-utility 但 borderline DAC 的 candidate 给很高分（utility 和 feasibility 是两个 axis）。独立 gate + hard threshold 保证 safety 不被 utility trade off。

这个 paradigm 跟 Hydra-MDP / DiffusionDrive / VADv2 的 candidate-scoring 思路同源，但 Auto-JEPA 用 **retrieval 替代 parametric generator**。好处是：
- 无需训 generator（省一个 diffusion / flow matching 训练）；
- memory 可解释、可审计（出问题可以看是哪条 ground-truth trajectory 被选了）；
- memory 容易扩展（加新 data 就 encode 进 memory，无需 retrain）。

坏处是：
- memory coverage 是 hard bound——如果某个 maneuver 没在 training data 里出现过，永远 retrieve 不到（论文 limitation 里承认了这点）；
- K=300 后 saturation，说明 memory recall 是 bottleneck。

---

## 6. 实验结果表解读

### Table 1: NAVSIM v1 navtest (12,146 scenarios)

| Method | Sensors | NC↑ | DAC↑ | TTC↑ | C↑ | EP↑ | PDMS↑ |
|---|---|---|---|---|---|---|---|
| Human | — | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| TransFuser | 3×C+L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| Hydra-MDP | 3×C+L | 98.3 | 96.0 | 94.6 | 100.0 | 78.7 | 86.5 |
| DiffusionDrive | 3×C+L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| LAW (world model) | 1×C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| AutoVLA | 3×C | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| Curious-VLA | 1×C | 98.4 | 96.9 | 97.9 | 98.1 | 88.5 | 90.3 |
| **Auto-JEPA** | **1×C** | **98.4** | **98.3** | **95.0** | **100.0** | **87.1** | **91.3** |

几个观察：
1. **只用 1 个 front camera** 就到了 91.3，跟用 3 camera + LiDAR 的 DiffusionDrive（88.1）拉开 3.2 分。这说明 latent target 比 dense sensor fusion 更高效；
2. **DAC 98.3 是表里最高的之一**（仅次于 DriveVLA-W0 的 99.1），说明 gate + memory 的 drivable-area 防御很有效；
3. **EP 87.1 接近 human 87.5**，说明 model 没有学到 "原地不动" 的 degenerate 解，progress 很积极；
4. **C 100.0**，因为 memory 里都是 ground-truth trajectory，天然 smooth，scorer 又有 comfort supervision；
5. **TTC 95.0 不是最高**（AutoVLA 98.0、Curious-VLA 97.9 更高），说明 collision avoidance 还有提升空间——这可能跟 memory 里某些 candidate 本身就 close-call 有关。

### Table 2: NAVSIM v2 EPDMS

Auto-JEPA 在 updated official implementation 下 89.1 EPDMS，跟 CLOVER（90.4，但用 learned generator）接近，但 Auto-JEPA 无 parametric generator。evaluator change 主要影响 TL（traffic light）和 LK（lane keeping），从 TL 97.2→99.7、LK 84.0→94.7，说明 official v2 的 human-behavior filtering 对这两项更友好。

### Table 3: Component ablation

| Intent | Scorer | Gate | PDMS |
|---|---|---|---|
| ✗ | ✓ | ✓ | 52.6 |
| ✓ | ✗ | ✓ | 87.6 |
| ✓ | ✓ | ✗ | 91.0 |
| ✓ | ✓ | ✓ | 91.3 |

- **关掉 intent（用固定 codebook medoid 当 query）→ 52.6**：暴跌 38.7 分。这是最关键的 ablation，证明 predicted intent 确实是 scene-conditioned 的，不是 random。52.6 大概是 "永远 retrieve 同一条 trajectory" 的分数；
- **关掉 scorer（只 retrieval + gate）→ 87.6**：scorer 贡献 3.7 分。retrieval 给的 300 candidate 已经包含正确答案，但需要 scorer 把它挑出来；
- **关掉 gate（只 retrieval + scorer）→ 91.0**：gate 只贡献 0.3 分 PDMS，但 DAC 从 98.3 降到 97.9。gate 主要修 safety edge case，对平均分数影响小但对 failure mode 重要。

### Table 4: Candidate pool size K

| K | PDMS |
|---|---|
| 1 | 87.6 |
| 200 | 91.1 |
| 300 | 91.3 |

K=1→200 提升 3.5 分，200→300 只提升 0.2 分。**Saturation 在 K≈200**。这告诉我 bottleneck 不是 scorer（如果 scorer 是 bottleneck，K 越大越应该好，因为更多 candidate 包含正确答案），而是 **memory recall**——再放大 pool 也捞不到新的正确 candidate，因为 memory 里就没有。这指向 future work：要么扩 memory，要么用 intent-conditioned diffusion refinement 生成 memory 外的 candidate。

---

## 7. Selective occlusion 实验的深层含义

这个实验我觉得是全文最漂亮的 validation。设置：

- 对 15,364 个 validation scene，做两类 mask：
  1. **Dynamic-agent mask**：把所有 visible traffic participant 的 projected region mask 掉，4 帧一致；
  2. **Random mask**：mask 等面积 的 random region；
- 保持 ego history 和 navigation command 不变，只改 visual；
- 测 $\Delta_{\text{intent}} = 1 - \cos(\hat{\mathbf{Z}}, \hat{\mathbf{Z}}_m)$。

结果：
- Dynamic-agent mask：mean $\Delta_{\text{intent}} = 0.080$；
- Random mask：mean $\Delta_{\text{intent}} = 0.027$；
- Ratio **2.97×**；
- 71.1% 的 scene 里 dynamic-agent mask 影响更大。

Figure 1 / 4 / 5 的 qualitative 例子更直观：在同一个 scene 里，occlude 掉 **跟 ego 有交互的 lead vehicle**，predicted intent 和 selected trajectory 大幅 shift；occlude 掉 **旁边不交互的车**，intent 几乎不变。

**为什么这个结果 surprising 且 important？**

Model 从来没见过 object box、agent identity、interaction label。训练 signal 只有 "predict future ego trajectory latent"。但 model 自发地学到了 "lead vehicle 的 visual feature 对 prediction 重要，旁边车的 visual feature 不重要"。这是 **planning-relevant attention 从 action supervision 里 emerge** 的证据。

我的 intuition 是：gradient flow 自动做了一种 implicit attention。Lead vehicle 的 pixel 通过 visual encoder 影响 $\hat{\mathbf{Z}}$，$\hat{\mathbf{Z}}$ 通过 alignment loss 跟 $\mathbf{Z}^+$ 对齐，$\mathbf{Z}^+$ 由 future trajectory 决定，future trajectory 受 lead vehicle 影响。所以 lead vehicle pixel 的 gradient magnitude 自然大。而旁边不交互的车，它的 pixel 变化不改变 future trajectory，gradient 接近 0，model 学到 ignore 它。

这跟你在 supervised learning 里观察到的 "attention 跟 task relevance 相关" 是一回事，但这里没有显式 attention label，全靠 action target 反传。这给 "用 action target 当 implicit perception supervision" 提供了很强的证据。

---

## 8. 跟相关 work 的 positioning

### vs. V-JEPA 2 / I-JEPA / V-JEPA
- I-JEPA：image patch → future patch latent；
- V-JEPA：video patch → future video patch latent；
- V-JEPA 2：自监督 video representation，可 downstream 做 understanding/prediction/planning；
- **Auto-JEPA 把 target 从 "future video latent" 换成 "future ego trajectory latent"**，并且这个 prediction 直接参与 inference（retrieval key），不只是 pretraining。

### vs. Drive-JEPA (Wang et al. 2026, arXiv:2601.22032)
Drive-JEPA 也是 JEPA + driving，但它主要做 **video pretraining + trajectory distillation**，JEPA prediction 在 inference 时只是 representation backbone。Auto-JEPA 的 prediction **就是 planner 的输出**，直接当 retrieval key。这是 operational role 的根本区别。

### vs. LAW / World4Drive / DeepSight / DriveWorld-VLA（latent world model 系）
这些方法 predict future **scene** latent（BEV / occupancy / scene feature），target 描述 "surrounding scene 怎么 evolve"。Auto-JEPA 的 target 是 "ego 怎么 move"——scene dynamics 只通过它对 ego trajectory 的影响被 retain。这是 **scene-centric → action-centric** 的 shift。

### vs. DiffusionDrive / GoalFlow / VADv2（parametric candidate generation）
这些方法用 diffusion / flow matching / vectorized vocabulary 生成 candidate trajectory。Auto-JEPA 用 **non-parametric retrieval** 替代 generator。好处是无需训 generator、可解释、易扩展；坏处是 memory coverage 是 hard bound。

### vs. VLA (AutoVLA / DriveVLA / Curious-VLA / RecogDrive)
VLA 把 action tokenize 成 discrete token，autoregressive decode。Auto-JEPA predict **continuous intent latent** + non-parametric retrieval。VLA 的好处是 language grounding、reasoning chain；Auto-JEPA 的好处是 trajectory geometry 不经过 quantization error，且 memory 提供高保真 candidate。

### vs. CLOVER (Ang et al. 2026, arXiv:2605.15120)
CLOVER 是 learned generator + scorer pipeline。Auto-JEPA 直接 reuse CLOVER 的 scorer 当初始化，但 **去掉 generator，用 retrieval 替代**。Auto-JEPA 89.1 vs CLOVER 90.4 EPDMS，差距 1.3 分，说明 retrieval-only 跟 learned generator 接近，但少了 generator 训练成本。

### vs. PRANK (Biktairov et al. 2020)
PRANK 在 motion forecasting 里用过 latent nearest-neighbor。Auto-JEPA 把这个 idea 用到 **ego planning**，并且 query 是从 visual predict 出来的 intent latent，不是从 history motion encode 的。

---

## 9. 我的 intuition on limitation & future

论文自己提的 limitation：
1. **Memory coverage bound**：如果 feasible maneuver 不在 memory，retrieval 失败。K=200→300 saturation 佐证；
2. **Selection calibration**：scorer 可能 misrank，gate 可能误拒；
3. **No scene-level forecast**：不能做 interactive simulation / counterfactual。

我加几个自己的 intuition：

**a. Memory 是离散 manifold 的 sample，intent 是 continuous。** 当 predicted intent 落在 memory 两个 candidate 之间时，retrieval 只能给最近的那个，丢失了 interpolate 的能力。Future work 可以做 **intent-conditioned local refinement**：retrieve top-1 然后 用 predicted intent 当 condition 做 small diffusion / flow 把 candidate 微调到 intent 附近。这能突破 memory coverage limit 同时保持 explicit geometry 的可解释性。

**b. InfoNCE 的 negative 是 batch 内其他 scene，但同一 scene 的多 modal trajectory（左转 vs 直行都可能）会被当 negative push 开。** 这跟 multi-modal planning 的初衷有点 tension。一个 fix 是用 cluster-based negative 或 hard negative mining，把 "不同 scene 但 similar intent" 当 negative，"同 scene 不同 modal" 不当 negative。

**c. Visual encoder frozen 是双刃剑。** V-JEPA 2 的 self-supervised representation 对 general visual feature 好，但对 driving-specific 的 fine-grained（车道线、traffic light、small obstacle）可能不够。Paper 里 NAVSIM v2 的 TL 97.2→99.7 主要是 evaluator change，但 TL 本身不是最高分，可能跟 frozen encoder 对 traffic light 的 sensitivity 不足有关。Unfreeze 最后几层做 LoRA fine-tune 可能 help。

**d. Selective attention 实验的 2.97× 是平均，但 28.9% 的 scene 里 random mask 影响更大。** 这些 scene 可能是 "lead vehicle 不在视野内 / 全是 background 决定 trajectory"（比如空旷直道，road geometry 主导）。这提示 model 其实也关注 road / lane feature，不只是 agent。这是好事，说明 selectivity 不是 "只看车" 而是 "看所有 planning-relevant feature"。

**e. 跟 LeCun 的 H-JEPA (Hierarchical JEPA) 的联想。** LeCun 一直推 hierarchical predictive model：高层 predict abstract plan，低层 predict pixel。Auto-JEPA 实际上只做了 "高层"——predict intent latent。如果把 retrieval 出来的 trajectory 当 low-level target，再加一个 "predict future video given intent" 的 low-level head，就是完整 H-JEPA。这个 low-level head 在 Auto-JEPA 里被 intentional drop 掉了，因为 planning 不需要它。但如果你想加 simulation / counterfactual，就得把它加回来。

**f. Retrieval 跟 RAG 的类比。** 这其实是 trajectory 版的 retrieval-augmented generation。Query = intent latent，memory = ground-truth trajectory，retriever = flat cosine。跟 LLM RAG 的区别是：LLM RAG retrieve text chunk 然后 concatenate 进 prompt 再 generate；Auto-JEPA retrieve trajectory 然后 直接用（不 generate）。一个 hybrid 是 retrieve top-K 然后 用 scorer 当 "reranker"——这跟 LLM RAG 的 reranker pipeline 几乎同构。所以未来可以用 LLM RAG 那套（learned retriever、cross-encoder reranker、HNSW ANN）来 upgrade。

**g. "Continuous intent" 跟 motor cortex / affordance 的类比。** 8-token continuous latent 很像神经科学里的 "motor program"——一个连续的、time-indexed 的运动计划 latent。这跟离散 action token（VLA 那套）的 "discrete motor command" 形成对比。Continuous 的好处是 smooth、可 interpolate、对 small motion change 敏感；discrete 的好处是 language-grounded、easy to compose。Driving 这种 continuous control task 天然适合 continuous intent。

---

## 10. Web links for reference

- **Auto-JEPA code & models**: https://github.com/NoctYang/Auto-JEPA
- **V-JEPA 2** (visual encoder backbone, Assran et al. 2025): https://arxiv.org/abs/2506.09985
- **I-JEPA** (Assran et al. 2023, ICCV): https://arxiv.org/abs/2301.08243
- **V-JEPA** (Bardes et al. 2024): https://arxiv.org/abs/2404.08471
- **NAVSIM benchmark** (Dauner et al. 2024, NeurIPS): https://arxiv.org/abs/2406.13359
- **NAVSIM repo**: https://github.com/autonomousvision/navsim
- **CLOVER** (scorer 初始化来源, Ang et al. 2026): https://arxiv.org/abs/2605.15120
- **Drive-JEPA** (Wang et al. 2026, 对照): https://arxiv.org/abs/2601.22032
- **LAW** (Li et al. 2025b, ICLR, latent world model baseline): https://arxiv.org/abs/2412.15215
- **Hydra-MDP** (Li et al. 2024, CVPR, candidate scoring baseline): https://arxiv.org/abs/2406.06978
- **DiffusionDrive** (Liao et al. 2025, CVPR): https://arxiv.org/abs/2411.15139
- **VADv2** (Jiang et al. 2024): https://arxiv.org/abs/2402.13243
- **InfoNCE / CPC** (van den Oord et al. 2018): https://arxiv.org/abs/1807.03748
- **Vista** (Gao et al. 2024, generative driving world model): https://arxiv.org/abs/2401.02977
- **GAIA-1** (Hu et al. 2023, generative world model): https://arxiv.org/abs/2309.17080
- **TransFuser** (Chitta et al. 2023, TPAMI): https://arxiv.org/abs/2205.15997
- **AutoVLA** (Zhou et al. 2025b, NeurIPS): https://arxiv.org/abs/2412.14349
- **DriveVLA-W0** (Li et al. 2026a, ICLR): https://arxiv.org/abs/2601.01403
- **Curious-VLA** (Chen et al. 2026, CVPR Findings): https://arxiv.org/abs/2603.06049
- **PRANK** (Biktairov et al. 2020, NeurIPS, latent retrieval for motion forecasting): https://arxiv.org/abs/2010.06945

---

## 11. 一句话总结直觉

Auto-JEPA 把 driving world model 的 prediction target 从 "future scene" 收窄到 "future ego trajectory latent"，让 JEPA predictor 只学 "这个 scene 意味着 ego 怎么动"，再用 predicted latent 当 retrieval key 从 ground-truth trajectory memory 捞 candidate，最后用 scorer + gate 做 scene-conditioned selection。整个 pipeline 没有 dense future reconstruction、没有 learned generator、没有 explicit perception annotation，但通过 action-target supervision 自发 emerge 出 planning-relevant selective attention，在 NAVSIM v1 拿到 91.3 PDMS。这个工作给 "world model 不需要 reconstruct 完整世界，只需要 predict action-relevant latent" 这个哲学提供了一个非常干净的 driving 实证。
