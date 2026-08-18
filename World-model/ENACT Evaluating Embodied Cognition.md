---
source_pdf: ENACT Evaluating Embodied Cognition.pdf
paper_sha256: fa243a0e57530b77198d33de536c651ca336843fc3703b91dab45c0fae88e1a7
processed_at: '2026-08-18T11:03:47-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 ENACT

好，咱们抛开学术腔，像在 Stanford 白板前聊天那样说说这篇 paper 到底在干嘛。

---

## 一句话概括

**给 VLM 看一堆 robot 第一人称操作画面，然后让它玩"排排序"游戏——看你能不能搞清楚"做了什么动作"和"画面怎么变了"之间的因果关系。**

就这么简单。但背后的 insight 很深。

---

## 为什么要做这个？

你知道现在 VLM 的问题——大家都在说 "GPT-5 能看图能聊天，它是不是已经 understand 世界了？" 但你仔细想，这些模型训练的时候看的是什么？是互联网上的图片和文字。它从来没碰过任何东西。它不知道 "把杯子放到桌子上" 之后画面会变成什么样。

这就像一个人看了 10 万小时烹饪视频，但你让他闭眼描述 "打完蛋之后碗长什么样"，他可能答得上来（因为视频里见过），但你让他推理 "如果把打好的蛋倒进锅里，下一秒画面是什么样"，他就傻了。

Embodied cognition 理论说：**智能是在和世界互动中" enact"出来的，不是看出来的**。那问题就来了——VLM 是"看"出来的，它到底有没有 embodied cognition？

以前的 benchmark 要么测 instruction following（"把杯子拿起来"），要么测 spatial reasoning（"杯子在桌子左边还是右边"），但都 confound 了一堆东西。ENACT 想要一个 **干净的诊断工具**，只测 reasoning 本身。

---

## 怎么测？——排序游戏

这个设计真的很漂亮，简单到让人拍大腿。

### Forward task：给动作，排画面

你给模型：
- 一张当前画面（比如 robot 站在冰箱前）
- 一串 actions（"打开冰箱门" → "拿出牛奶" → "关上冰箱门"）
- 几张 shuffled 的未来画面

让它排出正确的画面顺序。

**这测的是"想象力"**——你得能在脑子里 simulate，"开了冰箱之后画面里会多一扇打开的门，拿牛奶的时候手上会多一个牛奶盒"。

### Inverse task：给画面，排动作

你给模型：
- 一张当前画面
- 一串 ordered 的未来画面
- 几个 shuffled 的 actions

让它排出正确的 action 顺序。

**这测的是"归因能力"**——你看到画面里冰箱门开了，牛奶不见了，你得反推 "哦，这步是打开冰箱门，这步是拿牛奶"。

### 为什么用排序而不用 video generation？

因为 video generation 的评估太 messy 了。你让模型生成下一帧画面，它生成的像素模糊一点，你就算它错？它颜色暗一点，你就算它错？这不 fair。

排序游戏把 photorealism 和 reasoning **完全解耦**。画面都是 simulator 给好的真画面，你只需要判断 "哪个画面先来哪个后来"。评估信号超级 clean。

---

## 数据哪来的？——BEHAVIOR simulator

Stanford 自己的 BEHAVIOR benchmark，1000 个 everyday activities，物理引擎模拟的 robot 操作。

关键 trick：他们不是用 raw trajectory（里面 90% 都是无意义的 gripper 移动），而是 **只提取 semantic state change 的 keyframes**。比如从 "没抓" 到 "抓了"，从 "没开" 到 "开了"。这些才是有意义的 decision points。

然后从一个长 trajectory 里，用 dynamic programming 采样不同长度的 sub-trajectory（3 到 10 步），拼成 QA pairs。一条 trajectory 能生成几百个 questions，非常 scalable。

---

## 核心发现：四个 "啊哈" 时刻

### 发现一：Inverse 比 Forward 容易——而且差距随步数变大

这个结果非常 robust，所有模型都一样。

你想，Inverse 是 "看图说话"——画面里微波炉从关变开，你 match 到 "open microwave" 这个 action。这是 visual-to-textual mapping，VLM 的 comfort zone。

Forward 是 "看文画图"——给你 "open microwave"，你得想象下一帧画面长什么样。这需要真正的 visual simulation，VLM 基本做不到。

**这说明 VLM 的 language prior 很强，但 visual imagination 很弱。它能"读"世界，但不能"演"世界。**

### 发现二：步数一长，模型就崩了

L=3 的时候 GPT-5 还能跟人类差不多（84% vs 93%）。到了 L=10，GPT-5 掉到 47%，人类还是 95%。

这个 gap 太 telling 了。人类做这个任务根本不费劲——我们看到 10 步操作，自然就在脑子里 maintain 一个 "现在冰箱开着、牛奶在桌上、碗已经洗了" 的 mental model。VLM 没有这个 persistent state tracking 能力。每看一张新图，它就从零开始理解，前面的 context 就散了。

**这就是 embodied cognition 的核心——在 partial observation 下 maintain long-horizon spatial memory。VLM 完全缺这个。**

### 发现三：右手偏好

这个发现特别 striking。分析 LeftGrasping 和 RightGrasping 的错误，发现模型对右手的判断明显比左手准。9.38% 的真实左手操作被误判为右手，反过来只有 4.67%。

人类 89% 是 right-handed，训练数据里右手操作远多于左手。VLM 把这个 distribution prior 学进去了。

**VLM 不是通用 embodied agent，它是 human-embodiment mimic。** 如果你的 robot 是双臂对称的，VLM 的 grounding 会变差，因为它 priors 里 "右手是主力"。

### 发现四：视野偏好

把 camera aperture 从人类正常的 40 改成 60、80 或者 fisheye，模型性能显著下降。把 camera height 从 1.75m 抬高 0.5m，forward task 也显著变差。

**VLM 假设自己是个 1.75m 高、FOV 正常的人类。** 如果你的 robot 装了个 GoPro（fisheye），或者装在 2m 高的杆子上，VLM 的 visual understanding 就 degrade。

---

## 错误分析：VLM 在 forward task 里是 "hallucinator"

他们把模型的 predicted ordering 转成 implied action sequence，然后和 ground truth 对比。五类错误：

1. **Entity Substitution**: 动作对了，物体搞错了（"打开冰箱" 变成 "打开柜子"）
2. **Polarity Inversion**: 方向反了（"add" 变成 "remove"）
3. **Predicate Substitution**: 物体对了，关系搞错了（"OnTop" 变成 "Inside"）
4. **Hallucination**: 预测了根本没发生的动作
5. **Omission**: 漏掉了真实发生的动作

Forward task 里 **Hallucination 占 43.9%，是最大错误类型**。

这意味着什么？模型在"想象未来"时，不是在做 faithful simulation，而是在 **prior-driven 脑补**。它看到 "robot 在厨房"，就自动补 "应该会开冰箱、拿东西、关冰箱"，不管实际 action sequence 是什么。

这跟你说的 "LLM 是有损压缩" 完全对上了。模型压缩了训练数据里的 statistical pattern（厨房里经常开冰箱），但没压缩物理因果。被要求 extrapolate 时，它 fall back 到统计 prior，而不是 simulate 物理 reality。

Inverse task 里 Hallucination 和 Omission 平衡（各 41.8%），说明 retrospective 时模型既会脑补也会漏，更 "随机"。

---

## 我觉得这个工作最 clever 的地方

### 1. Evaluation philosophy 的 shift

不是又一个 leaderboard。它是一个 **diagnostic tool**，帮我们理解 VLM 内部的 "world model" 长什么样。

以前的 benchmark 告诉你 "GPT-5 在 ALFRED 上 60% success rate"。然后呢？你不知道它为什么失败。ENACT 告诉你 "GPT-5 在 10 步 forward task 里 47% PA，主要错误是 hallucination，右手偏好 9% mixing rate"。这些是 actionable insights。

### 2. Symbolic action space 的选择

用 scene graph delta 而不是 continuous motor command。这很 smart——VLM 的 native abstraction 就是 semantic level 的，用 symbolic predicate (OnTop, Inside, Grasping) 让 evaluation 信号和 VLM 的 representation space 对齐。

代价是你 capture 不到 fine-grained motor skill（怎么 grasp、力度多大），但那是另一层 evaluation 的事。

### 3. DP-based trajectory sampling

从 M 个 keyframes 采 L 长度的 trajectory，暴力枚举 $\binom{M}{L}$ 会爆炸。他们转成 DAG path counting + backtracking sampling，polynomial 复杂度，还能保证 uniform sampling。这个工程做得漂亮。

---

## 更大的图景：这对 embodied AI 意味着什么？

### VLM 离真正的 embodied agent 还很远

人类做 ENACT 95% 准确率，几乎不随步数下降。GPT-5 在 3 步时 84%，10 步时 47%。这不是 "差一点"，是 **fundamentally different cognitive architecture**。

人类有 working memory、有 spatial mental model、有 causal reasoning。VLM 有 attention、有 statistical correlation、有 language prior。前者是 systemic 的，后者是 pattern matching 的。

### Training data 的方向

VLM 的 anthropocentric bias 说明：光喂人类第一人称视频不够。需要 diverse embodiment 的 egocentric data——robot 视角、不同 FOV、不同 height。不然你的 "embodied VLM" 永远是人类模仿者，不是通用 embodied agent。

### Architecture 的方向

VLM 的 long-horizon degradation 说明：transformer 的 attention 机制不适合 maintain persistent spatial state。可能需要 external memory（像 DVRAct 那种）、或 recurrent state（像 Dreamer 的 latent dynamics）、或显式的 scene graph representation。

光 scale up transformer 大概解决不了这个问题。这是 architectural 的 bottleneck。

---

## 如果我来 follow up

1. **Probe 内部 representation**：用 ENACT 的 QA pairs 做 mechanistic interpretability，看 VLM 哪一层开始 encode action-conditioned dynamics。如果找不到，说明 VLM 根本没有 internal world model，纯靠 surface pattern。

2. **Finetune intervention**：用 ENACT 数据 finetune，看 forward task 能否提升。如果能大幅提升，说明是 data 问题；如果提升有限，说明是 architecture 问题。这会直接指导下一代 embodied VLM 的设计。

3. **Cross-embodiment**：用 humanoid、quadruped、dual-arm 不同形态生成 ENACT，测 VLM 的 embodiment transfer。如果 transfer 很差，那 "one VLM for all robots" 的 dream 还很遥远。

4. **和 video generation 对比**：虽然 paper 说没评估 video model，但我觉得值得试试。如果 Genie、Cosmos 这类 video world model 在 forward task 上比 VLM 好很多，说明 latent dynamics model 是对的方向。如果差不多烂，说明问题更深。

---

## 最后一句

ENACT 不复杂，但它问了一个好问题，用了一个 clean 的方法，得到了 actionable 的结论。这种 paper 比那些刷 SOTA 的 paper 有价值得多——它帮我们 **understand** 模型，而不只是 **rank** 模型。

作为 Karpathy 你应该 appreciate 这种 taste。这不是 "我的模型比你的大"，是 "我的 probe 比你的 insightful"。

---

# ENACT: 用 Egocentric World Modeling 评估 Embodied Cognition

## 核心直觉

这篇 paper 来自 Stanford (Li Fei-Fei, Jiajun Wu, Ruohan Zhang) 和 Northwestern (Manling Li) 团队，核心 idea 非常 elegant：**把 embodied cognition 的评估，cast 成一个 world modeling 问题，用 sequence reordering VQA 来 probe VLMs 是否具备 embodied intelligence**。

其背后 philosophical grounding 来自 embodied cognition 理论 (Smith & Gasser, 2005; Varela et al., 2017)——intelligence 不是 passive observation 习得的，是 sensorimotor interaction 中 enacted 出来的。那么问题是：现代 VLMs 主要在 disembodied data 上训练，它们是否涌现出了 embodied cognition？

参考链接：
- 项目主页: https://enact-embodied-cognition.github.io
- BEHAVIOR benchmark: https://behavior.stanford.edu
- Ha & Schmidhuber World Models: https://worldmodels.github.io
- POMDP 经典综述 Kaelbling et al. 1998: https://www.sciencedirect.com/science/article/pii/S000437029800023X

---

## 问题形式化：POMDP + Scene Graph

### 数学架构

ENACT 把 underlying embodied task 建模为一个 POMDP (Partially Observable Markov Decision Process, Åström 1965)：

- **State space** $\mathcal{S}$: 元素是 symbolic scene graphs，从 low-level simulator state $\mathcal{G}$ 导出
- **Observation space** $\mathcal{O} \subset \mathbb{R}^{H \times W \times 3}$: robot 的 egocentric RGB 视图
- **Action space** $\mathcal{A}$: 元素是 scene-graph differences $a_t = \delta(s_t, s_{t-1})$

这里的符号含义：
- $H, W$: image height 和 width
- $s_t \in \mathcal{G}$: 时刻 $t$ 的 scene graph
- $o_t$: 时刻 $t$ 的 RGB observation
- $\delta(\cdot, \cdot)$: scene graph difference operator，summarize 两帧之间 semantic changes (objects, relations, attributes)

Scene graph 的结构化表示（参考 Figure 7 的 JSON 例子）：
```json
{
  "nodes": [{"name": "robot_r1", "category": "agent", "states": []}, ...],
  "Edges": [{"from": "robot_r1", "to": "plate_93", "states": ["RightGrasping"]}, ...]
}
```

Scene graph difference（Figure 8）包含 `add` 和 `remove` 两个部分，描述 edge 和 node 的变化。

### Action 的语义化抽象

关键设计：**action 不是 continuous motor trajectory，而是 scene graph 的 symbolic delta**。这是 deliberate 的选择：

$$a_k := \Delta_{\text{Vis}}(s_{i_{k+1}}, s_{i_k})$$

其中 $\Delta_{\text{Vis}}$ 返回 $\delta(s_{i_{k+1}}, s_{i_k})$ 中在两帧图像里都 visible 的 difference subset。

为什么这样设计？因为 VLM 的 native abstraction 是 semantic-level 的，不是 continuous control。用 symbolic predicates (RightGrasping, OnTop, Inside, Open, Cooked 等 11 类，见 Table 6) 可以避免 confounding——评估的是 reasoning 能力，不是 photorealistic video prediction。

---

## 两个互补任务

### Forward World Modeling（前瞻）

**输入**: 当前 image $o_0$ + 正确顺序的 action sequence $(a_0, \dots, a_{L-2})$ + shuffled future images $O' = (o'_1, \dots, o'_{L-1})$

**输出**: permutation $\sigma \in \text{Sym}([L-1])$ 使得：
$$(o'_{\sigma(1)}, \dots, o'_{\sigma(L-1)}) = (o_1, \dots, o_{L-1})$$

直觉：给定 action，想象未来状态怎么演化。这需要 prospective visual simulation——在脑中"演"一遍物理过程。

### Inverse World Modeling（回溯）

**输入**: 当前 image $o_0$ + 正确顺序的 observation images $(o_1, \dots, o_{L-1})$ + shuffled actions $A' = (a'_0, \dots, a'_{L-2})$

**输出**: permutation $\tau \in \text{Sym}([L-1])$ 使得：
$$(a'_{\tau(1)}, \dots, a'_{\tau(L-1)}) = (a_0, \dots, a_{L-2})$$

直觉：给定观察到的状态变化，反推是什么 action 导致的。这更像 retrospective textual reasoning。

这两个 task 的不对称性很有意思——forward 需要 visual imagination（"如果右抓微波炉，下一步画面什么样"），inverse 需要 visual-to-textual mapping（"画面里 microwave 从 closed 变 open，对应哪个 action"）。Table 7 列出了它们各自所需的 cognitive constructs：

| Task | Action-Effect Reasoning | Causal Inference | Affordance Recognition | Embodied Awareness | Temporal Abstraction |
|------|---|---|---|---|---|
| Forward | ✓ | ✗ | ✓ | ✓ | ✓ |
| Inverse | ✗ | ✓ | ✓ | ✓ | ✓ |

---

## 数据生成 Pipeline：从 BEHAVIOR simulator 自动生成

这是 paper 的第二个 contribution——scalable 数据生成。

### Step 1: Segmented Frames 提取

Raw robot trajectory $\mathcal{T} = \{(o_t, s_t)\}_{t=1}^T$ 通常有大量无意义帧（gripper 移动但不改变 semantic state）。筛选规则：

1. **Temporal stability filter**: state change 必须持续至少 40 frames (~1.3s at 30Hz)。这个数字来自 cognitive science——人类 attentional sub-events 更新周期约 1s (Wyble et al. 2009; Gavazzi et al. 2013)。
   
   参考: https://doi.org/10.1037/0096-1523.35.3.787

2. **Similarity check**: 计算 change signature 的 cosine similarity，threshold 0.97，过滤 near-duplicate frames。

最终得到 segmented frames $\mathcal{K} = \{t_1 < \dots < t_M\}$。

### Step 2: Key-Frame Trajectory Synthesis (KFTS)

这是最 elegant 的算法部分（Algorithm 1）。问题转化为：从 $M$ 个 segmented frames 中采样长度为 $L$ 的 valid trajectory $\pi = (i_1, \dots, i_L)$，相邻 frames 之间必须有 visible state change。

朴素方法：枚举 $\binom{M}{L}$ 组合，复杂度爆炸。

ENACT 的解法：**转化为 DAG path sampling + dynamic programming**。

#### DAG 构建
节点 = segmented frames。有向边 $E_{ij} = [\text{Vis}(\delta(s_i, s_j))]$，当且仅当 $i < j$ 且 state change visible。

#### DP 路径计数
$$DP[\ell, i] = \sum_{j < i} DP[\ell-1, j] \cdot E_{ji}$$

变量含义：
- $DP[\ell, i]$: 长度为 $\ell$ 且终止于 frame $i$ 的 valid trajectory 数量
- Base case: $DP[1, i] = 1$（任何单 frame 都是长度 1 的 valid path）
- $E_{ji}$: adjacency matrix 元素，表示从 frame $j$ 到 frame $i$ 是否有 valid transition

#### Weighted Backtracking Sampling
1. 采样 end-node $i_L \sim \text{Categorical}(w)$，其中 $w_i = DP[L, i]$
2. 反向回溯：选择 predecessor $j^*$ 的概率 $\propto DP[\ell-1, j^*]$
3. 这样保证 uniform sampling over all valid length-$L$ trajectories

复杂度是 polynomial in $M$ 和 $L$，比 brute-force 高效得多。从单条 trajectory 可以生成最多 $\binom{M}{L}$ 个 candidates，实践中 $L \leq 10$ 而 $M \gtrsim 30$，所以单条 trajectory 可以生成大量 QA pairs。

### Step 3: QA 生成

每个 key-frame trajectory 转成一个 forward QA 和一个 inverse QA。最终数据集：
- 29 个 BEHAVIOR activities
- Step lengths $L \in \{3, \dots, 10\}$
- 每个 $L$ 约 560 items per QA type
- 总计 **8,972 QA pairs**

---

## 评估设计：Online Verifier

这部分很重要——不能用 brittle 的 exact index matching，因为多个 valid answers 可能存在。

### Action Signature
$$a^{\text{sig}}(s_{i-1}, s_i) = \{c = (\gamma, e, \rho)\}$$

变量：
- $\gamma \in \{\text{add, remove}\}$: operation
- $e$: entity 涉及的 object
- $\rho$: predicate (如 RightGrasping, OnTop)

### Forward acceptance rule
对于 predicted ordering $\sigma$ 和 ground-truth $\tau$：
- **Exact**: $\sigma = \tau$
- **Semantic** (length matched): $\forall i, C_i \subseteq \tilde{C}_i$

其中 $C_i$ 是 ground-truth visible change，$\tilde{C}_i$ 是 predicted change。直觉：predicted step 必须覆盖 reference 的 visible change。

### Inverse acceptance rule
- **Semantic** (length matched): $\forall i, \tilde{C}_i \subseteq F_i$

其中 $F_i$ 是 full reference transition。直觉：predicted action 可以是 reference 的 concise subset。

### Metrics

**Task Accuracy (TA)**:
$$\text{TA} = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \mathbf{1}\{\text{accepted}(x)\}$$

**Pairwise Accuracy (PA)**（对相邻 pair 给 partial credit）:
$$\text{PA}(x) = \frac{1}{L} \sum_{i=1}^{L} \mathbf{1}\{C_i \subseteq \tilde{C}_i \text{ (forward) or } \tilde{C}_i \subseteq F_i \text{ (inverse)}\}$$

如果 length mismatch，用 monotone alignment 最大化 subset-satisfying pairs。

---

## 实验结果：核心发现

### Table 1 解读：Pairwise Accuracy

这个表是 paper 的核心数据。我挑几个关键 cell 分析：

**GPT-5**（最强 proprietary）:
- Forward L=3: 84.62, L=10: 46.93
- Inverse L=3: 86.28, L=10: 55.33
- 始终 Inverse > Forward，gap 随 $L$ 增大而扩大

**Gemini 2.5 Pro**:
- Forward L=3: 86.10, L=10: 36.98
- Inverse L=3: 87.94, L=10: 56.62

**Human**:
- Forward L=3: 93.62, L=10: 95.13
- Inverse L=3: 92.05, L=10: 96.29

**关键观察**：

1. **Inverse consistently > Forward**。这是 paper 最 robust 的 finding。作者的解释：models 的语言 retrospective reasoning 强于 prospective visual simulation。Forward 需要"想象"画面，Inverse 只需要"读"画面。

2. **Performance 随 horizon 单调下降**。VLMs 在 $L \geq 8$ 时近乎 collapse（Table 9 显示 TA 接近 0）。人类却保持 ~95%。这说明 VLMs 缺乏 long-horizon interactive spatial memory。

3. **Human-model gap 随 $L$ 急剧扩大**。$L=3$ 时 GPT-5 接近人类，$L=10$ 时差距 50 个百分点。

4. **Open-weight 模型表现**：InternVL3.5-241B-A28B、GLM-4.5V、Qwen2.5-VL-72B 都很有竞争力，有时甚至超过 Claude Sonnet 4。Cosmos-Reason1（用 embodied data 训练）在 $L > 5$ 时比同 size 模型更稳定。

参考 InternVL: https://github.com/OpenGVLab/InternVL
参考 Qwen-VL: https://github.com/QwenLM/Qwen2.5-VL
参考 Cosmos-Reason1: https://github.com/nvidia-cosmos/cosmos-reason1

---

## Ablation 实验：揭示 VLM 的偏见

### 1. Image Realism（Section 3.2）

测试四种渲染：
- **Ray Tracing (baseline)**: BEHAVIOR 默认
- **Realistic**: 用 GPT-image-1 把 sim frame 转 photorealistic 风格
- **Path Tracing**: 更高保真渲染 (Kajiya 1986 rendering equation)
- **Ray Tracing Only**: 关闭 reflections, DLSS, ambient occlusion 等

结果：**所有变体 $p \geq 0.2$，无统计显著差异**。说明 VLM 的 bottleneck 是 multi-step interaction reasoning，不是 low-level image realism。

Real-world 实验（960 QA pairs from 3 个真实场景）也证实 **sim-to-real gap 极小**（Table 2）。InternVL3.5-241B 在真实视频上的 inverse > forward 趋势、horizon degradation 趋势都和 sim 一致。

### 2. Camera FOV（Section 3.3）

Baseline aperture 40。测试 aperture 30, 60, 80, Fisheye。

结果：
- Aperture 30: 无显著差异
- Aperture 60, 80, Fisheye: **$p \leq 0.01$，显著下降**

直觉：VLMs 对 human-like FOV 有强烈 prior。非标准 optics 会破坏它们的 visual grounding。

### 3. Camera Height（Section 3.3）

Baseline 1.75m（人眼高度）。测试 High (+0.5m) 和 Low (-0.25m)。

结果：
- High: forward 显著下降 $\Delta = -0.13$
- Low: 无显著差异（仍在正常人类身高范围内）

### 4. Robot Appearance（Section 3.4）

测试 White Color, Random Color, Skin Color。

结果：**全部无显著差异**（$|\Delta| < 0.05$, $p > 0.10$）。说明 VLM 对自己的 embodiment 形式不敏感——它理解 interaction，不依赖特定 body representation。

### 5. Handedness Asymmetry（Section 3.4）——最 striking 的发现

分析 LeftGrasping 和 RightGrasping predicate 的错误（Figure 5 C.2）：

| Task | Hand | Precision | Recall | Mixing Rate |
|------|------|-----------|--------|-------------|
| Forward | Left | 0.4483 | 0.4087 | 0.0938 |
| Forward | Right | 0.4976 | 0.4958 | 0.0467 |
| Inverse | Left | 0.4040 | 0.4040 | 0.1858 |
| Inverse | Right | 0.4618 | 0.4618 | 0.0949 |

**右手 consistently 更准确，左手 mixing rate 更高**。Forward 中 9.38% 的真实左手变化被误判为右手，而只有 4.67% 反过来。这与人类 ~89% right-handed 分布 (Papadatou-Pastou et al. 2020) 一致——VLMs 从训练数据中学到了人类的 handedness prior。

参考: https://doi.org/10.1037/bul0000229

---

## Error Analysis：五个类别

Section 3.5 的 error analysis framework 很精细。把 model 的 predicted permutation 转成 predicted action sequence $\hat{a}_k := \Delta_{\text{Vis}}(s'_{\sigma(k+1)}, s'_{\sigma(k)})$，然后和 ground truth $a_k$ 做 set-level 对比。

### 五类错误

1. **Entity Substitution**: predicate 对，object 错
2. **Polarity Inversion**: object 和 predicate 对，add/remove 反了
3. **Predicate Substitution**: object 对，predicate 错
4. **Hallucination**: 预测了 ground truth 里没有的 change
5. **Omission**: 漏掉了 ground truth 里的 change

### GPT-5 错误分布（Figure 6）

**Forward task**:
- Hallucination: 43.9%（最多！）
- Omission: 37.1%
- Polarity Inversion: 12.4%
- Entity Substitution: 6.3%
- Predicate Substitution: 0.3%

**Inverse task**:
- Hallucination: 41.8%
- Omission: 41.8%（完美平衡）
- Polarity Inversion: 9.2%
- Entity Substitution: 5.4%
- Predicate Substitution: 1.9%

**关键洞察**：

- **Hallucination 主导**：模型依赖 learned textual priors，不是 faithful visual grounding。它"脑补"了应该发生但实际没发生的 action。
- **Omission 高发**：在 egocentric partial observation 下，模型 track 不到 object persistence。物体被遮挡后，模型 forget 了它的 state。
- Forward 的 Hallucination > Omission，说明模型倾向"多想"——over-imagine 未来状态。Inverse 的两者平衡，说明 retrospective 时 model 既会脑补也会漏。

---

## 为什么这个工作重要：Build Your Intuition

Karpathy，让我帮你 connect the dots。

### 1. 评估范式的 shift

传统 embodied AI benchmark（ALFRED, TEACh, CALVIN, RoboVQA）大多评估 instruction following 和 goal-conditioned control。它们 confound 三个东西：(a) visual perception, (b) action execution, (c) reasoning。

ENACT 的 clever 之处在于用 **sequence reordering** 隔离了 reasoning。模型不需要输出 pixel-level video（避免 video generation 的 evaluation nightmare），只需要 order 现成的 frames/actions。这让评估信号非常 clean。

参考 ALFRED: https://askforalfred.com
参考 TEACh: https://github.com/alexpashevich/teach
参考 CALVIN: https://calvinrobot.github.io
参考 RoboVQA: https://robovqa.github.io

### 2. Forward vs Inverse 的认知科学映射

这个 asymmetry 让我想到认知科学里的 **mental simulation vs mental rotation** 区别。Forward world modeling 类似 "episodic future thinking"——人类在 hippocampus 里 pre-play 未来场景。Inverse 类似 "episodic memory retrieval"——reconstruct 过去的 cause。

VLMs 的 Inverse > Forward 暗示：它们的 language prior 帮助 retrospective reasoning（"画面里 microwave 开了，所以 action 是 open microwave"），但 prospective visual imagination 很弱（"如果 open microwave，下一帧画面什么样"需要真正 simulate 像素）。

这呼应了 Hafner 的 Dreamer 系列工作——world model 在 latent space 里 roll out。VLMs 没有显式的 latent dynamics model，它们的 "world model" 是隐式的、fragile 的。

参考 DreamerV3: https://danijar.com/project/dreamerv3/
参考 Genie: https://sites.google.com/view/genie-2024

### 3. Embodied Cognition 的 Operational Definition

Paper 的 philosophical stance 很明确：embodied cognition 不是单一能力，是一束能力的 emergent property。ENACT 通过两个 task 隐式评估了：
- **Affordance recognition**: 识别 object 可以被怎么 interact
- **Action-effect reasoning**: 理解 action 的 physical consequence
- **Embodied awareness**: 理解自己的 body 在 scene 中的角色
- **Long-horizon interactive memory**: 在 partial observation 下 maintain 多步 state

这比 Yang et al. 2025 的 EmbodiedBench（用 subjective criteria catalogue 能力）更 objective。

参考 EmbodiedBench: https://embodiedbench.github.io

### 4. Anthropocentric Bias 的启示

VLMs 的 right-handed bias 和 human-vision FOV preference 揭示了一个 deep issue：**这些模型不是通用 embodied agent，它们是 human-embodiment mimic**。

如果未来要让 robot 用 VLM 做 policy，而 robot 有 6 个 DOF 的 arm、fisheye camera、非人眼高度——VLM 的 visual grounding 会 degrade。这说明：

(a) Training data 需要更多 diverse embodiment 的 egocentric video（不只是人类 first-person）
(b) 或者需要 architecture-level 的 viewpoint-invariant representation

这让我想到 Meta 的 Ego-Exo4D 和 EPIC-Kitchens——它们采集了大量人类 egocentric 视频，但恰恰强化了这种 anthropocentric prior。

参考 Ego-Exo4D: https://ego-exo4d-data.org/
参考 EPIC-Kitchens: https://epic-kitchens.github.io

### 5. 与 World Model 文献的对话

ENACT 和 recent world model 工作（Cosmos, Genie, Aurora-Bench）形成有趣对话：

- **Cosmos** (NVIDIA): 物理常识 + embodied reasoning，但用 generative video 评估
- **Genie** (DeepMind): latent space interactive environment
- **Aurora-Bench** (Qiu et al. 2025): short-horizon forward/inverse world modeling

ENACT 的差异化：**explicit action space (scene graph delta) + long-horizon (up to 10 steps) + egocentric + symbolic evaluation**。这避免了 video generation 的 photorealism confound。

Aurora-Bench: https://arxiv.org/abs/2506.06006
Cosmos: https://github.com/nvidia-cosmos/cosmos-predict

### 6. 局限性与未来方向

Paper 自己承认的 limitations：
- 没有评估 video generative models（因为 rollout 物理不一致 + metric 设计难）
- 没有做 VLM finetuning（但 dataset 可以支持）
- Ablation 只在 representative subset 上做

我会加几个：
- **Symbolic action 的 expressiveness 上限**：scene graph delta 捕捉不到 fine-grained motor skill（怎么 grasp、grasp 力度）。对于需要 dexterous manipulation 的 task，这个 abstraction 会漏掉关键信息。
- **BEHAVIOR 的 activity 分布偏差**：29 个 home-scale activities，可能不覆盖 industrial、outdoor、social 场景。
- **Sim-to-real 的 hidden gap**：虽然 Table 2 显示小 gap，但只有 960 QA pairs，且只测了 InternVL3.5。更大规模、更多模型的真实世界评估会更有说服力。

---

## 总结：这篇 paper 的 contribution 在我心中的位置

ENACT 是一个 **conceptually clean, methodologically sound, empirically rich** 的 benchmark。它的核心 contribution 不是某个 SOTA number，而是：

1. **提供了一个评估 lens**：world modeling as embodied cognition proxy
2. **揭示了 VLM 的 fundamental limitation**：long-horizon interactive reasoning 崩溃
3. **发现了 anthropocentric bias**：handedness、FOV、viewpoint
4. **提供了一个 scalable 数据生成 pipeline**：simulator-based, DP sampling

作为 Karpathy，你可能会特别欣赏这个工作的 **evaluation philosophy**——它不是又一个 leaderboard，而是一个 diagnostic tool，帮我们理解 VLM 内部的 "world model" 到底是什么形态。

我最兴奋的 finding 是 **Hallucination > Omission in forward task**。这暗示 VLM 在 "想象未来" 时，不是 faithful simulator，而是 prior-driven hallucinator。这和你在 "State of GPT" 演讲里提到的 "LLM 是有损压缩" 的直觉一致——模型压缩了训练数据里的 statistical pattern，但没压缩物理动力学。当被要求 extrapolate 时，它 fall back 到 prior，而不是 simulate reality。

参考你自己的 talk: https://karpathy.ai/stateofgpt/

如果要 push 这个工作进一步，我会想做：
1. **Latent world model probing**：用 ENACT 的 QA pairs probe VLM 内部 representation，看哪一层开始 encode action-conditioned dynamics
2. **Finetuning intervention**：用 ENACT 数据 finetune，看 forward task 能否提升（如果能，说明是 data 问题；如果不能，说明是 architecture 问题）
3. **Cross-embodiment transfer**：用不同 robot morphology（dual-arm, quadruped, humanoid）生成 ENACT，测 VLM 的 embodiment generalization

这是 embodied AI evaluation 的一个重要 milestone，值得仔细读和 follow。
