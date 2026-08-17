---
source_pdf: ENACT.pdf
paper_sha256: fa243a0e57530b77198d33de536c651ca336843fc3703b91dab45c0fae88e1a7
processed_at: '2026-08-04T04:17:44-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ENACT 的人话版

## 一句话概括

这篇论文其实就是问一个问题：**现在的 VLM 到底有没有"世界模型"，还是只是个会说话的 pattern matcher？**

答案很残酷：基本没有。人类能轻松做到的事情，GPT-5 这种 frontier model 在多步推理下直接崩溃。

## 他们到底干了什么

想象你在厨房做菜。你看到一张照片：冰箱门开着，手在抓一盒牛奶。下一张照片：牛奶放在桌上。你能推断中间发生了什么——手把牛奶从冰箱拿出来放桌上。这叫 **inverse world modeling**——从结果倒推动作。

反过来，如果告诉你"手抓住牛奶，把牛奶放桌上"，让你从一堆打乱的照片里挑出正确顺序，这就是 **forward world modeling**——从动作正推结果。

ENACT 就是让 VLM 玩这两种排列组合游戏，步数从 3 步到 10 步不等。就这么简单，但结果很扎心。

## 核心发现，用大白话说

### 发现 1：VLM "倒推" 比 "正推" 强

这个很反直觉。你想想，forward 是给你 action script 让你 match 图像，inverse 是给你图像让你推断 action。直觉上 forward 更容易（都有剧本了），但实际上 inverse 普遍比 forward 高 5-15 个百分点。

为什么？因为 VLM 本质上还是 LLM——它擅长**文字推理**。给它看图像序列问"发生了什么"，它能用语言 chain 思考。但让它**想象** action 执行后的视觉结果，它就废了。这暴露了一个根本问题：VLM 的 vision 部分是个"挂件"，真正干活的还是 language 部分。

Karpathy 你之前就吐槽过这个——VLMs 本质上是"带 vision adapter 的 LLM"。这篇 paper 用实验证明了你的直觉。

### 发现 2：步数一长，全崩

这是最 striking 的结果。看 GPT-5 的 Task Accuracy：

- 3 步任务：80%
- 7 步任务：20%
- 10 步任务：5%

而人类？从 3 步到 10 步，稳定在 84%-90%。

这意味着什么？VLM 没有 **persistent state tracking**。每推一步，它都要从头看图重新推理，没有"累积信念"的概念。人脑里有个 running mental model，每步动作只更新局部，VLM 每步都像第一次看这个场景。

这就是 POMDP 里 **belief state** 的核心思想（[Kaelbling 1998](https://arxiv.org/abs/cs/9803107)）。人脑天然有 belief update 机制，VLM 架构上完全没有。Autoregressive Transformer 没 recurrent state，每个 token 重新 attend 所有 prior tokens——这对 language 还行，对 evolving world state 就力不从心了。

### 发现 3：VLM 有"人手偏好"

这个发现太有意思了。他们统计了模型对 LeftGrasping 和 RightGrasping 的预测准确率：

- Right hand: precision 0.50, recall 0.50
- Left hand: precision 0.45, recall 0.41
- 左手被误判为右手的比例：9.38%
- 右手被误判为左手的比例：4.67%

VLM 居然有"惯用手"偏好！而且这个偏好与人类 ~89% 右撇子的统计 ([Papadatou-Pastou 2020](https://psycnet.apa.org/record/2020-71343-001)) 完全吻合。

这只能有一个解释：**VLM 从人类数据中继承了人类 motor prior**。它不是"理解"了左右手，它只是见过更多右手操作的图像，所以更"熟悉"右手。

这是 embodied bias 的铁证。模型并不知道自己是个 robot，它只是带着人类数据的偏见在 pattern match。

### 发现 4：VLM 不认识自己的身体

他们还做了个实验：改变 robot 的外观颜色（白色、随机色、肤色），看模型表现。结果：**完全没影响**，p > 0.10。

这说明什么？VLM 对自己的 embodiment 完全无 awareness。它不知道"我是个 robot"，它只是把 robot arm 当作图像里的一个 object 来识别。改颜色无所谓，因为颜色不改变动作语义。

如果换个真正的 embodied agent，应该对自己的身体敏感——因为 body 就是 action 的 effector，是 self-awareness 的核心。VLM 完全没有这种 self-model。

### 发现 5：图像真实度不重要

他们测试了四种渲染质量：photorealistic (GPT-image-1 转换)、path tracing (最高 fidelity)、ray tracing (baseline)、ray tracing only (关闭高级效果)。

结果：**全部 p ≥ 0.2，性能差异 < 5%**。

这太关键了。这说明 VLM 的瓶颈**根本不在图像识别**，而在 **multi-step reasoning**。你把图像搞得更逼真，模型也不会突然学会推理。问题在 cognitive 层面，不在 perceptual 层面。

这也 validates 了 ENACT 的设计——把 world modeling 评估与 photorealism 解耦，直接 probe reasoning 能力。

### 发现 6：错误主要是"没看到"和"瞎编"

他们做了精细的错误分析。把 model 预测与 ground truth 的 scene graph 变化做集合比较，分五类错误：

| 错误类型 | Forward | Inverse |
|---------|---------|---------|
| Hallucination (瞎编) | 43.9% | 41.8% |
| Omission (漏掉) | 37.1% | 41.8% |
| Polarity Inversion (方向反) | 12.4% | 9.2% |
| Entity Substitution (对象错) | 6.3% | 5.4% |
| Predicate Substitution (关系错) | 0.3% | 1.9% |

Hallucination + Omission 占了 80%+。这说明 model 的核心问题**不是搞错了细节**，而是**根本不知道发生了什么**。

它要么瞎编一个不存在的动作，要么漏掉一个真实发生的动作。它不是"误解了世界"，它是"没看到世界"。

这跟 LLM hallucination 的本质是一样的——autoregressive generation 倾向于 generate "plausible" content，而不是 faithful-to-evidence content。VLM 也染上了这个毛病。

## 数据是怎么生成的

这部分工程上很 clever。他们用 BEHAVIOR simulator ([Li et al. 2024](https://arxiv.org/abs/2403.09227)) 跑机器人轨迹，然后：

1. 找出所有 scene graph 发生变化的时刻（segmented frames）
2. 构建 DAG：节点是 frame，边是 valid transition
3. 用 dynamic programming 数所有 length-L 的 valid path
4. 加权回溯采样，保证 uniform 采样所有 valid trajectory

这样从一个 episode 能生成 $\binom{M}{L}$ 个 QA（M 通常 30+，L ≤ 10），scalable 到 millions of QAs。

这比手动标注 scene graph 不知道高效多少倍。Simulator 提供 ground-truth physics state，这是 real-world data 永远做不到的——你没法精确知道真实世界每个 object 的 pose 和 contact 关系。

## 为什么这个 paper 重要

Karpathy 你应该从这个 paper 里看到几件事：

**第一，evaluation 的艺术。** ENACT 把 embodied cognition 评估从"capability checklist"（模型声称能做 X/Y/Z）转成"behavioral test"（模型必须 demonstrate 能力）。这就像你之前讲 LLM 时说的——做 benchmark 和真懂之间差距巨大。ENACT 提供了一个 behavioral probe，让 model 躲不掉。

**第二，暴露了 VLM 的 architecture limit。** Autoregressive Transformer 没有 persistent state，每步从零推理。这限制了 long-horizon reasoning。Mamba ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752))、RWKV ([Peng et al. 2023](https://arxiv.org/abs/2305.13048)) 这种 recurrent architecture 可能更适合。

**第三，揭示了 training data bias。** Right-handed bias 直接从 human data 继承。如果我们想要真正 general 的 embodied AI，需要 diverse embodiment training data——不同 camera、不同视角、不同 handedness。

**第四，指明了未来方向。** 真正的 world model 需要：explicit belief state tracking + action-conditioned dynamics learning + counterfactual reasoning + embodiment self-awareness。当前 VLM 全部缺失。

## 最终的 intuition

ENACT 告诉我们：当前 VLM 在 embodied cognition 上的表现，就像一个只看过烹饪视频但从没下过厨的人——它能描述刀怎么拿、菜怎么切，但真让它按顺序执行 10 步做一道菜，它会忘步骤、瞎编动作、分不清左右手。

真正的 embodied intelligence 需要 **enactment**——通过 interaction 学习，不是通过 observation 学习。这跟 Smith & Gasser 2005 ([link](https://direct.mit.edu/artl/article/11/1-2/13/9052)) 说的 embodied cognition 理论完全一致：intelligence 通过 sensorimotor interaction 涌现，不是被动观察习得。

VLM 现在就是个被动观察者。让它变成 active enactor，是下一阶段 AI 的核心挑战。

## 参考链接

- [ENACT 项目主页](https://enact-embodied-cognition.github.io)
- [BEHAVIOR-1K](https://behavior.stanford.edu)
- [Ha & Schmidhuber, World Models](https://arxiv.org/abs/1803.10122)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [Genie](https://arxiv.org/abs/2402.15391)
- [Kaelbling POMDP](https://arxiv.org/abs/cs/9803107)
- [Smith & Gasser Embodied Cognition](https://direct.mit.edu/artl/article/11/1-2/13/9052)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [RWKV](https://arxiv.org/abs/2305.13048)
- [Human Handedness Meta-Analysis](https://psycnet.apa.org/record/2020-71343-001)
- [GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)
- [Gemini 2.5 Pro](https://storage.googleapis.com/model-cards/documents/gemini-2.5-pro.pdf)

---

# ENACT: Egocentric Interaction World Modeling Benchmark 深度解析

## 1. 核心动机与问题定位

ENACT 的核心命题：把 embodied cognition 评估转化为 **world modeling** 问题，通过 egocentric interaction 的 sequence reordering VQA 形式来度量 VLMs 是否真的"理解"了 action 与 environment 之间的因果关系。这避开了 photorealistic video generation 的 confounding factor，直接探查 model 的 transition reasoning 能力。

论文立足于几个关键假设：
1. **Embodied cognition ≠ passive observation**：intelligence 通过 sensorimotor interaction 而非被动观察涌现（参考 Smith & Gasser, 2005, https://direct.mit.edu/artl/article/11/1-2/13/9052）
2. **Current VLMs 的 disembodied training** 是否真的产生了 embodied reasoning？这点目前没有 unified objective 来测试
3. **World model as sandbox for reasoning**（参考 Xing et al. 2025, https://arxiv.org/abs/2507.05169）：world model 应该支持 counterfactual rollout，而非单纯的 video fidelity

**My intuition build**: Karpathy 你应该立刻感受到 ENACT 的设计哲学与 classical world models (Ha & Schmidhuber 2018, https://arxiv.org/abs/1803.10122) 的差异——它把 world modeling 退化为 **symbolic sequence reordering**，这样既能 force model 进行 causal reasoning，又能 scalable 地自动 grading，避开了 video prediction benchmark 里那些"看起来对就行"的虚假信号。

## 2. POMDP Formulation 的数学细节

### 2.1 三元组定义

ENACT 形式化定义在 POMDP (Åström 1965, https://www.sciencedirect.com/science/article/pii/0022247X65900813) 框架上：

- **State space** $\mathcal{S}$: 元素是 symbolic scene graphs $\mathcal{G}$，由 simulator low-level state 抽取
- **Observation space** $\mathcal{O} \subset \mathbb{R}^{H \times W \times 3}$: 机器人 egocentric RGB views，$H$ 是 image height，$W$ 是 image width
- **Action space** $\mathcal{A}$: 元素是 scene-graph differences $a_t = \delta(s_t, s_{t-1})$

其中 $\delta(\cdot, \cdot)$ 是 scene-graph difference operator，输出形如：
$$\delta(s_i, s_j) = \{ \text{add: } \{e_1 \xrightarrow{\rho} e_2\}, \text{remove: } \{e_1 \xrightarrow{\rho'} e_2\} \}$$

### 2.2 Visible Action 提取

由于 POMDP 是 partially observable，不是所有 scene graph 变化都能在 egocentric view 中看到（比如冰箱内部的变化当冰箱门关着时）。论文定义 visibility predicate $\mathrm{Vis}(\cdot)$ 与 visible-change extractor $\Delta_{\mathrm{Vis}}$:

$$a_k := \Delta_{\mathrm{Vis}}(s_{i_{k+1}}, s_{i_k}) \subseteq \delta(s_{i_{k+1}}, s_{i_k})$$

只有当涉及的 object 在前后两帧图像里都可见时，state change 才进入 action space。

### 2.3 Forward 与 Inverse 任务的 permutation formulation

**Forward World Modeling**:
- 输入: 当前观察 $o_0$，有序 action sequence $(a_0, \ldots, a_{L-2})$，shuffled observations $O' = (o'_1, \ldots, o'_{L-1})$
- 输出: permutation $\sigma \in \mathrm{Sym}([L-1])$，使得
$$(o'_{\sigma(1)}, \ldots, o'_{\sigma(L-1)}) = (o_1, \ldots, o_{L-1})$$

**Inverse World Modeling**:
- 输入: 当前观察 $o_0$，有序 observation sequence $(o_1, \ldots, o_{L-1})$，shuffled actions $A' = (a'_0, \ldots, a'_{L-2})$
- 输出: permutation $\tau \in \mathrm{Sym}([L-1])$，使得
$$(a'_{\tau(1)}, \ldots, a'_{\tau(L-1)}) = (a_0, \ldots, a_{L-2})$$

这里 $\mathrm{Sym}([L-1])$ 表示 $\{1, 2, \ldots, L-1\}$ 上的对称群，即所有 $(L-1)!$ 种 permutation。

**Intuition**: Forward 任务相当于"我有动作脚本，请按顺序展示结果"——要求 prospective visual simulation；Inverse 任务相当于"我有结果，请倒推动作"——要求 retrospective causal inference。这种**对称设计**让作者能 clean 地分离 model 的 prospective vs retrospective reasoning 能力。

## 3. Key-Frame Trajectory Synthesis (KFTS) 算法

这是 paper 的 engineering 亮点：从 single trajectory 通过 combinatorial sampling 生成 up to millions of QAs。

### 3.1 Segmented Frames 抽取

Raw trajectory 中大量帧没有 semantic change（比如 gripper 移动到 toolbox 之前）。论文通过以下筛选：

1. **Temporal stability filter**: state change 至少持续 40 帧（≈1.3s @ 30Hz），呼应 cognitive science 中 attentional sub-event ~1s 的发现（Wyble et al. 2009, https://psycnet.apa.org/record/2009-08467-009; Gavazzi et al. 2013, https://www.nature.com/articles/srep01168）
2. **Predicate-level change signature**: 将 scene-graph difference 编码为 one-hot vector $c_j$，通过 cosine similarity 阈值 0.97 去重
3. **Acceptance criterion**: $\cos(c_j, c_{j-1}) < 0.97$

输出 chronologically ordered segmented frames $\mathcal{K} = \{t_1 < \cdots < t_M\}$。

### 3.2 DAG 构建

把 $M$ 个 segmented frames 看作 graph 节点，构建 directed acyclic graph (DAG)：
- 节点 $i$ 对应 segmented frame $(o_i, s_i)$
- 边 $i \to j$（要求 $i < j$）存在当且仅当 $\mathrm{Vis}(\delta(s_i, s_j)) = 1$
- 邻接矩阵 $E_{ij} = [\mathrm{Vis}(\delta(s_i, s_j))]$

注意这里允许 **frame skipping**：$i$ 与 $j$ 不需要相邻，只要中间状态变化 visible。这相当于把原始 trajectory 抽象成 semi-MDP（Sutton et al. 1999, https://arxiv.org/abs/cs/9905014）。

### 3.3 Dynamic Programming Path Counting

为了 uniform sampling 所有 length-$L$ 的 valid trajectories，论文用 DP table $DP[\ell, i]$ 表示以 frame $i$ 结尾的 length-$\ell$ 路径数量：

$$DP[1, i] = 1 \quad \text{(base case)}$$

$$DP[\ell, i] = \sum_{j < i} DP[\ell-1, j] \cdot E_{ji} \quad \text{for } \ell = 2, \ldots, L$$

变量解释：
- $\ell$: 当前 path length（上标维度）
- $i$: path 终点 frame index（下标维度第二位）
- $j$: 前驱 frame index
- $E_{ji}$: adjacency matrix entry，表示 $j \to i$ 边是否存在
- $DP[\ell-1, j]$: 以 $j$ 结尾的 length-$(\ell-1)$ 路径数量

**Complexity**: $\mathcal{O}(L \cdot M^2)$，远比 brute-force $\binom{M}{L}$ 的指数复杂度高效。

### 3.4 Weighted Backtracking Sampling

有了 $DP$ table，可以无偏采样：

1. **End-node sampling**: 按 $w_i = DP[L, i]$ 为权重，sample end-node $i_L^{(r)} \sim \mathrm{Categorical}(w)$，这样 frame 参与 trajectory 越多越容易被选作终点
2. **Backward reconstruction**: 对 $\ell = L, \ldots, 2$，从前驱集合 $\mathcal{P} = \{j < \mathrm{cur} \mid E_{j, \mathrm{cur}} = 1 \land DP[\ell-1, j] > 0\}$ 中以 $DP[\ell-1, j]$ 为权重采样 $j^*$
3. 重复 $R$ 次得到 trajectory set $\Pi$

**Intuition**: 这个采样方式保证了**所有 valid length-$L$ trajectory 被采到的概率均等**。如果你直接均匀 sample end-nodes 然后 uniform backtrack，会偏向那些前驱少的 path。Weighted sampling 修正了这种 bias。

## 4. Evaluation Metrics 的语义设计

### 4.1 Online Verifier

这是 ENACT 评估的核心 trick：不直接比对 permutation index，而是比对**语义上的 state change**。

对 reference sequence 提取：
- $C_i$: 第 $i$ 步的 visible change subset
- $F_i$: 第 $i$ 步的 full change set（含不可见变化）

对 prediction 提取 $\tilde{C}_i$（full diff，由 model 输出的 permutation 推算出来的 implied action sequence）。

**Forward acceptance rule**:
$$\text{match} = (\sigma = \tau) \lor \left( \forall i, C_i \subseteq \tilde{C}_i \right)$$

即：predicted step 必须 **覆盖** reference 中可见的变化（多预测可以，少不行）。

**Inverse acceptance rule**:
$$\text{match} = (\tau = \tau^*) \lor \left( \forall i, \tilde{C}_i \subseteq F_i \right)$$

即：predicted action 可以是 reference full transition 的 **子集**（少预测可以，多不行）。

**Intuition**: 这种 asymmetric subset rule 反映了 forward/inverse 任务的本质差异——forward 中图像信息有限，模型允许多预测（实际没看到的可能也合理）；inverse 中 model 给的是 action 描述，应该精炼，不能幻觉出没发生的事。

### 4.2 Metrics 公式

**Task Accuracy (TA)**:
$$\mathrm{TA} = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \mathbf{1}\{\text{accepted}(x)\}$$

其中 $\mathbf{1}\{\cdot\}$ 是 Iverson bracket（条件真则 1，假则 0），$\mathcal{D}$ 是 dataset，$x$ 是单个 QA。

**Pairwise Accuracy (PA)**:
$$\mathrm{PA}(x) = \frac{1}{L} \sum_{i=1}^{L} \mathbf{1}\{C_i \subseteq \tilde{C}_i \text{ (forward) } \lor \tilde{C}_i \subseteq F_i \text{ (inverse)}\}$$

$$\mathrm{PA} = \frac{\sum_x \#\text{correct pairs in } x}{\sum_x L_x}$$

这里 $L_x$ 是 sample $x$ 的 step length，PA 是 micro-average，相当于固定 $L$ 时的 per-item average。

## 5. 实验结果的核心发现

### 5.1 Forward vs Inverse 的 Asymmetry

Table 1 的 Pairwise Accuracy 显示一致 pattern：**Inverse > Forward**，且 gap 随 $L$ 增长而扩大。例如 GPT-5：

| L | Forward PA | Inverse PA | Gap |
|---|-----------|-----------|-----|
| 3 | 84.62 | 86.28 | 1.66 |
| 6 | 64.18 | 68.78 | 4.60 |
| 10 | 46.93 | 55.33 | 8.40 |

**Interpretation**: Models 在 retrospective textual reasoning（从 observation 倒推 action）上比 prospective visual simulation（从 action 正推 observation）更强。这与 LLMs 的训练 paradigm 一致——文本 generation 是 well-defined 的，而 visual synthesis 不是 LLMs 的 native 能力。

Karpathy 你应该联想到：这呼应了你之前在 tweets 里提到的 VLMs 本质是"LLM with vision adapter"的问题——它们依然以文本 reasoning 为主导。

### 5.2 Horizon Length 的 Degradation

几乎所有 model 在 $L \geq 8$ 时 TA 接近 0：

| Model | L=3 Forward TA | L=10 Forward TA |
|-------|---------------|------------------|
| GPT-5 | 80.59 | 5.00 |
| Gemini 2.5 Pro | 81.99 | 3.60 |
| GLM-4.5V | 66.08 | 0.00 |

**Intuition**: VLMs 没有"persistent spatial memory"——它们每一步都重新从 image-encoded features 推理，无法像人一样 maintain evolving world state 的 mental model。这与 Kaelbling's POMDP belief state (https://arxiv.org/abs/cs/9803107) 形成鲜明对比：人类通过 interaction 维护 belief，VLMs 没有 belief update 机制。

### 5.3 Human-Model Gap

Human performance 在所有 horizon 上都保持 90%+：

| L | Human Forward TA | GPT-5 Forward TA |
|---|-----------------|------------------|
| 3 | 90.38 | 80.59 |
| 7 | 88.31 | 20.24 |
| 10 | 84.00 | 5.00 |

Inter-annotator agreement Krippendorff's $\alpha = 0.83$（[0.79, 0.87] 95% CI）—— 数据非常 solid。

Krippendorff's alpha 公式：
$$\alpha = 1 - \frac{D_o}{D_e}$$

$D_o$ 是 observed disagreement，$D_e$ 是 expected disagreement by chance。$\alpha > 0.8$ 表示 high reliability。

## 6. Anthropocentric Bias 分析

### 6.1 Camera FOV Sensitivity

Baseline aperture = 40（接近 human FOV）。实验设置：

| Aperture | Forward $\Delta$ PA | Significance |
|----------|---------------------|--------------|
| 30 | small | $p > 0.1$ |
| 60 | $< -0.05$ | $p \leq 0.01$ |
| 80 | $< -0.05$ | $p \leq 0.01$ |
| Fisheye | $< -0.05$ | $p \leq 0.01$ |

**Intuition**: VLMs 在训练时几乎只见过 human-like FOV（30-50 度 aperture），偏离这个分布性能显著下降。这暗示了**数据集的 anthropocentric bias**——VLM 训练数据来自 human-captured images/web data，缺乏 robotic embodiments 的多样化视角。

### 6.2 Camera Height Sensitivity

| Height | Forward $\Delta$ PA | Inverse $\Delta$ PA |
|--------|---------------------|---------------------|
| High (+0.5m) | -0.13 (significant) | -0.06 (n.s.) |
| Low (-0.25m) | small | small |

Low 设置下 object 仍 visible，所以变化不大；High 设置下 perspective 显著不同，forward 受影响严重。

### 6.3 Robot Appearance Insensitivity

测试 White Color, Random Color, Skin Color 三种 robot appearance 变体——**全部 $p > 0.10$，$|\Delta| < 0.05$**。

**Key finding**: VLMs 对自己的 embodiment appearance 几乎无 awareness。这意味着它们不"知道自己是 robot"，只是 pattern matching。

### 6.4 Handedness Asymmetry

这是 paper 最 striking 的发现之一。定义 metrics:

- **Precision**: 正确匹配 / 预测总数
- **Recall**: 正确匹配 / ground truth 总数
- **Mixing rate**: ground truth 中某只手被预测为另一只手的比例

GPT-5 Forward task:

| Hand | Precision | Recall | Mixing |
|------|-----------|--------|--------|
| Left | 0.4483 ± 0.0076 | 0.4087 ± 0.0072 | 0.0938 |
| Right | 0.4976 ± 0.0055 | 0.4958 ± 0.0055 | 0.0467 |

Right hand 全面 better，left-to-right mixing rate (9.38%) 显著高于 right-to-left (4.67%)。

**Intuition**: 这与人类 ~89% right-handed 的统计 (Papadatou-Pastou et al. 2020, https://psycnet.apa.org/record/2020-71343-001) 完全一致。VLMs 从 human data 中"继承"了 right-handed prior。这是 embodied bias 的直接证据——data 决定了 model 的 motor prior。

## 7. Error Analysis 的五维分类

### 7.1 Structural Errors

将 ground truth 与 prediction 的 signature components 做集合比较，得到三类基础 outcome：
- **Correct**: 在两边都存在
- **Omission**: 只在 ground truth 中
- **Hallucination**: 只在 prediction 中

然后细分五类：

| Error Type | Definition |
|-----------|-----------|
| **Entity Substitution** | predicate 对，object 错 |
| **Polarity Inversion** | object + predicate 对，但 add/remove 反了 |
| **Predicate Substitution** | object 对，predicate 错 |
| **Hallucination** | 预测了不存在的 state change |
| **Omission** | 漏掉了真实 state change |

GPT-5 错误分布（Figure 6）：

| Error Type | Forward % | Inverse % |
|-----------|----------|----------|
| Hallucination | 43.9 | 41.8 |
| Omission | 37.1 | 41.8 |
| Polarity Inversion | 12.4 | 9.2 |
| Entity Substitution | 6.3 | 5.4 |
| Predicate Substitution | 0.3 | 1.9 |

**Key insight**: Omission + Hallucination 合计 81% (forward) / 84% (inverse)。这说明 model 的核心 failure 不是"搞错了 state change 的细节"，而是"根本不知道哪些变化发生了"。

Karpathy 这里你应该立刻联想到 LLM hallucination 的根源——autoregressive generation 倾向于 generate "plausible-sounding" content，而非 faith-to-ground-truth content。VLMs 在这里也是一样：它们 generate 看起来合理的 action description，但不严格 track visual evidence。

### 7.2 Semantic Error Categories

进一步把 errors 按 predicate 语义分类：

| Category | Example Predicates |
|---------|-------------------|
| Spatial Relations | OnTop, Inside, Under |
| Functional States | Open, ToggledOn, Cooked |
| Material States | Covered, Transition |
| Agent Interactions | LeftGrasping, RightGrasping |

发现：errors 主要集中在 **Spatial Relations** 与 **Agent Interactions**，且 task-dependent asymmetry:
- Forward task: spatial-relation errors 更多
- Inverse task: agent-interaction errors 更多

**Intuition**: Forward 需要从 action 推出 object 位置变化——这天然是 spatial reasoning；Inverse 需要从 observation 推出 agent 做了什么——这天然是 agent interaction reasoning。Error distribution 反映了 task 的 reasoning bottleneck。

## 8. Sim-to-Real Consistency 验证

为了验证 BEHAVIOR simulator 的发现是否 transfer 到 real world，论文在三个场景（kitchen, dinner table, workspace）手动采集 real videos 并 annotate scene graphs，生成 960 real-world QA pairs。

InternVL3.5-241B 在 real-world 上的表现（Table 2）：

| L | Forward TA (real) | Forward PA (real) |
|---|------------------|------------------|
| 3 | 73.33 | 80.00 |
| 6 | 13.33 | 49.00 |
| 10 | 0.00 | 26.88 |

**与 simulator 结果 qualitatively consistent**:
- Inverse > Forward
- Performance 随 L 单调下降
- TA 在 long horizon 崩溃

**Min sim-to-real gap**: 模拟器可以 faithful proxy for real-world embodied cognition 评估。

## 9. Image Realism Ablation

测试 4 种 rendering 设置（baseline = Ray Tracing with global effects）：

1. **Realistic**: GPT-image-1 把 segmented frames 转换为 photorealistic style
2. **Path Tracing**: 最高 fidelity rendering (NVIDIA Isaac Sim 内置)
3. **Ray Tracing Only**: 关闭 reflections, DLSS, ambient occlusion 等高级效果

结果：**所有变体 $p \geq 0.2$ vs baseline**，$|\Delta| < 0.05$。

**Critical insight**: VLMs 的 failure **不是 image realism 问题**，而是 multi-step interaction reasoning 问题。这进一步 validates ENACT 的设计哲学——把 world modeling 评估与 photorealism 解耦。

Karpathy 这点你应该深有共鸣：你之前讲过 LLMs 的核心 reasoning limitation 与 input modality fidelity 不直接相关，更深层的是 architecture + training paradigm 的限制。

## 10. Predicate Encoding Ablation

为了排除 "inverse 优势仅是 language prior" 假设，论文测试三种 action representation:

1. **Vanilla (Natural Language)**: 原始 NL 描述
2. **Symbolic Predicates**: 结构化 predicates
3. **Emoji-Style Encodings**: emoji 表示

InternVL3.5-241B 在 2,304 QA subset 上的结果（Table 12）：

| Encoding | L=3 Fwd TA | L=3 Inv TA |
|---------|-----------|-----------|
| Vanilla NL | 68.97 | 83.45 |
| Symbolic | 67.59 | 79.86 |
| Emoji | 65.52 | 77.24 |

**Inverse > Forward 在所有 encoding 下都成立**——这排除了 language prior 是 inverse 优势主因的假说。Inverse 优势源于 task structure 本身。

## 11. Contact Predicate Ablation

为了验证主结论不是 "semantic scene graph only" 抽象的 artifact，论文 augment predicate set 加入 binary contact relations（touch/no-touch），重新 evaluate InternVL3.5-241B（Table 11）：

| L | Forward TA (with contact) | Inverse TA (with contact) |
|---|--------------------------|--------------------------|
| 3 | 86.67 | 90.00 |
| 7 | 3.45 | 16.67 |

**结论依然 robust**: Inverse > Forward，长 horizon degradation 依然显著。

## 12. 与相关 Work 的差异

### 12.1 vs Classical World Models (Ha & Schmidhuber 2018)

| Aspect | Ha & Schmidhuber | ENACT |
|--------|-----------------|-------|
| Representation | Latent dynamics $z_t = f(z_{t-1}, a_{t-1})$ | Symbolic scene graph + RGB |
| Evaluation | Reconstruction + RL reward | VQA permutation accuracy |
| Action space | Continuous | Symbolic deltas |
| Generative | Yes (decoder) | No (avoids synthesis confounding) |

### 12.2 vs Embodied Benchmarks

- **EmbodiedQA** (Das et al. 2018, https://arxiv.org/abs/1902.08045): navigation QA，不测试 world modeling
- **Physion** (Bear et al. 2021, https://arxiv.org/abs/2106.08261): passive physics prediction，无 action conditioning
- **CLEVRER** (Yi et al. 2019, https://arxiv.org/abs/1910.01442): collision reasoning，primitive objects only
- **GVL** (Ma et al. 2024, https://arxiv.org/abs/2411.16064): value estimation as reordering，但无 explicit action space

ENACT 的 unique combination:
1. Egocentric observation
2. Explicit scene-graph action space
3. Forward AND Inverse dual-task
4. Long-horizon ($L$ up to 10)
5. Scalable auto-generation pipeline

### 12.3 vs Video World Model Benchmarks

- **Aurora-Bench** (Qiu et al. 2025, https://arxiv.org/abs/2506.06006): short-horizon video generation evaluation
- **WorldSimBench** (Qin et al. 2024, https://arxiv.org/abs/2410.18072): video generation quality
- **EWMBench** (Yue et al. 2025, https://arxiv.org/abs/2505.09694): scene/motion/semantic quality

这些 benchmark 关注 **generative fidelity**，ENACT 关注 **reasoning fidelity**——直接 decoupling。

## 13. Cosmos-Reason1 的特殊地位

Cosmos-Reason1-7B (Azzolini et al. 2025, https://arxiv.org/abs/2503.15558) 在 embodied data 上专门 trained。论文发现：

- 当 $L > 5$ 时，Cosmos-Reason1-7B 比 similar-sized VLMs 更稳定且 generally better
- 但是仍然显著弱于 GPT-5 与 Gemini 2.5 Pro

**Implication**: Embodied-specific training data 可以提升 model 的 embodied reasoning robustness，但 scale + general data 仍是主导因素。

## 14. Limitations 与 Open Questions

### 14.1 Scope 限制

- 测试的 bias 维度有限（FOV, height, appearance, handedness）
- 只 evaluate subset of models due to compute cost
- 没有探索 VLM finetuning
- 没有评估 video generative models（BAGEL 除外）

### 14.2 数据生成的 Sim-to-Real Generalization

虽然 real-world 实验 validate 了 simulator trends，但 BEHAVIOR 的 29 个 activities 仍有限。更多样化的 real-world environments 测试是 future work。

### 14.3 Open Questions for Intuition Building

1. **Belief state tracking**: 如何给 VLM 加 explicit belief update 机制？这是 POMDP 的核心，但 VLM architecture 完全没有
2. **Affordance learning**: ENACT 隐式测试了 affordance（哪些 action 在哪些 state 下可行），但 model 是否真的"理解" affordance？
3. **Counterfactual reasoning**: ENACT 测试的是 factual rollout，counterfactual（"如果换一个 action 会发生什么"）需要扩展
4. **Long-horizon memory**: 是否需要 explicit memory module（如 episodic memory + working memory 分离）？
5. **Embodiment-aware training**: 如何构造 training paradigm 让 VLMs 真正"知道自己是 robot"？目前的 self-awareness 测试显示 VLMs 完全无 embodiment awareness

## 15. 对 VLM 未来发展的 Implications

### 15.1 Architecture 层面

- **需要 recurrent state**: Autoregressive Transformer 没有 persistent state，每步从零推理。可能需要 Mamba-style (https://arxiv.org/abs/2312.00752) 或 RWKV-style (https://arxiv.org/abs/2305.13048) 的 recurrent architecture
- **需要 world model module**: 类似 Genie (Bruce et al. 2024, https://arxiv.org/abs/2402.15391) 或 DreamerV3 (Hafner et al. 2023, https://arxiv.org/abs/2301.04104) 的 latent dynamics module
- **需要 action embedding**: VLMs 当前只 encode image + text，没有 action tokens 的 native representation

### 15.2 Training Paradigm 层面

- **Self-supervised world modeling loss**: 让 model 在训练时 explicitly predict next state given action
- **Diverse embodiment training**: 不能只用 human-captured images，需要 synthetic data 涵盖不同 camera intrinsics / viewpoints
- **Counterfactual augmentation**: 给 model 看"如果做不同 action 会怎样"的数据，强制 causal reasoning

### 15.3 Evaluation 层面

ENACT 的方法论可以扩展到：
- **Multi-agent settings**: 多个 agent 的 world modeling
- **Deformable object interactions**: 论文已涉及，但可以 deeper
- **Tool use and causal chains**: 长链工具使用
- **Real-time interactive settings**: 不仅是 offline VQA，而是真正 online interaction

## 16. 总结的 Mental Model

ENACT 的核心 contribution 是把 embodied cognition 评估从"capability checklist"转向"world modeling behavioral test"。这就像 Karpathy 你之前对 LLMs 的"做 benchmark"和"真懂"之间 gap 的思考——ENACT 提供了一个**behavioral probe**，让 model 必须 demonstrate world modeling 能力，而不是声称拥有 capability。

**Key takeaways for your intuition**:

1. **Inverse > Forward asymmetry** 是 LLM-centric reasoning 的 fingerprint——models 强在 retrospective text reasoning，弱在 prospective visual simulation
2. **Long-horizon degradation** 是 missing belief state tracking 的直接证据——VLMs 没有 POMDP-style belief update
3. **Anthropocentric biases** 揭示了 training data 决定 model prior 的本质——right-handed bias 是直接从 human data 继承的
4. **Omission + Hallucination dominance** 表明 model 的核心 failure 是不知道发生了什么，而不是搞错了细节——这是 faithfulness 而非 precision 的问题
5. **Image realism insensitivity** 进一步 validates reasoning 是 bottleneck，而非 perception

ENACT 是一个 **scalable, reproducible, sim-grounded** 的测试床，为未来 embodied AI 发展提供了清晰的 diagnostic 工具。下一步的关键问题是：什么样的 architecture + training paradigm 能让 VLMs 真正"enact"而非"observe"？

## 参考链接

- [ENACT 项目主页](https://enact-embodied-cognition.github.io)
- [BEHAVIOR-1K Benchmark](https://behavior.stanford.edu)
- [Ha & Schmidhuber, World Models (2018)](https://arxiv.org/abs/1803.10122)
- [Åström, POMDP Optimal Control (1965)](https://www.sciencedirect.com/science/article/pii/0022247X65900813)
- [Kaelbling et al., Planning in POMDPs (1998)](https://arxiv.org/abs/cs/9803107)
- [Sutton et al., Semi-MDPs (1999)](https://arxiv.org/abs/cs/9905014)
- [Smith & Gasser, Embodied Cognition (2005)](https://direct.mit.edu/artl/article/11/1-2/13/9052)
- [Hafner et al., DreamerV3 (2023)](https://arxiv.org/abs/2301.04104)
- [Bruce et al., Genie (2024)](https://arxiv.org/abs/2402.15391)
- [Azzolini et al., Cosmos-Reason1](https://arxiv.org/abs/2503.15558)
- [Papadatou-Pastou et al., Human Handedness Meta-Analysis (2020)](https://psycnet.apa.org/record/2020-71343-001)
- [Krippendorff's Alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha)
- [Yi et al., CLEVRER (2019)](https://arxiv.org/abs/1910.01442)
- [Bear et al., Physion (2021)](https://arxiv.org/abs/2106.08261)
- [Ma et al., GVL (2024)](https://arxiv.org/abs/2411.16064)
- [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim)
- [GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)
- [Gemini 2.5 Pro Model Card](https://storage.googleapis.com/model-cards/documents/gemini-2.5-pro.pdf)
- [Llama 4 Model Card](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
