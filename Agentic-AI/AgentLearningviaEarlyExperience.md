---
source_pdf: AgentLearningviaEarlyExperience.pdf
paper_sha256: 6db07231ddf4b40df55acbb1d8fd0922bc979f4aba2c6a36ebf8d0bef2e6b01d
processed_at: '2026-07-18T05:32:54-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Early Experience：在 IL 与 RL 之间架一座桥

## 1. 这篇 paper 要解决的核心矛盾

Karpathy 你应该很熟悉 Silver 和 Sutton 2025 年那篇 "Welcome to the Era of Experience"（https://ai.googleblog.com/2025/era-of-experience.html）的论调：真正的智能体应该靠 trial-and-error 从自身经验中学习。但落到 language agent 上，RL 这条路现在卡在两件事：

- **Reward 不可获得**：web、tool-use 这类环境大多数没有 verifiable reward。一个 form 提交成功不代表每一项都填对了，平台不返回 ground truth。
- **Long-horizon rollout 代价高**：multi-turn tool use 涉及长交互链，credit assignment 不稳定，simulator/reset 基础设施几乎没有。

于是大家退回到 SFT/behavior cloning，但 SFT 有两个本质缺陷：(i) agent 从不观察自己 non-expert action 的后果，无法从 failure 中学习；(ii) expert data 是静态、窄分布的，distribution shift 一上来就 compounding error（Ross et al. 2011, DAgger https://arxiv.org/abs/1011.0686）。

这篇 paper 提出的 **early experience** 就是要在这两个极端之间找到一个 scalable 且 reward-free 的中间态。核心 insight 其实很朴素但被严重忽视：**agent 自己 propose 的 action 在 environment 中执行后得到的 next state，本身就是 supervision signal，不需要外部 reward**。Environment 的 textual response（DOM 变化、error message、API output）隐含了 action 质量的反馈。

---

## 2. 形式化：MDP 与 reward-free 学习

他们把问题放进标准 MDP 框架：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, R, \gamma, \rho_0)$$

变量含义：
- $\mathcal{S}$：state space，对应 webpage DOM、tool output、文本场景描述等
- $\mathcal{A}$：action space，离散动作（click、invoke tool、生成文本）
- $T: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$：transition function，$\Delta(\mathcal{S})$ 是 $\mathcal{S}$ 上的 probability simplex（即输出的是 next-state 分布）
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$：reward function，**可能在训练时未知或不可验证**——这是 paper 的关键设定
- $\gamma \in [0,1]$：discount factor
- $\rho_0 \in \Delta(\mathcal{S})$：initial state distribution
- $\pi_\theta: \mathcal{S} \to \Delta(\mathcal{A})$：参数为 $\theta$ 的 policy

经典 IL 目标：

$$\mathcal{L}_{\mathrm{IL}}(\theta) = -\sum_{i=1}^{N} \log \pi_\theta(a_i \mid s_i) \tag{1}$$

这是 negative log-likelihood，最小化等价于在 expert state $s_i$ 下最大化 expert action $a_i$ 的概率。问题在于训练时 agent 永远见不到 "$a \neq a_i$ 时 $s_{i+1}$ 长什么样"，于是对 consequences 毫无 awareness。

---

## 3. Early Experience 的数据构造

对每个 expert state $s_i$，从当前 policy $\pi_\theta(\cdot \mid s_i)$ 采样 $K$ 个 alternative action，构成候选集 $\mathcal{A}_i = \{a_i^1, a_i^2, \ldots, a_i^K\}$。上标 $j \in [K]$ 表示第 $j$ 个 alternative。

把每个 $a_i^j$ 在真实 environment 里执行一次，得到 $s_i^j \sim T(s_i, a_i^j)$。这就构成了 rollout dataset：

$$\mathcal{D}_{\mathrm{rollout}} = \{(s_i, a_i^j, s_i^j) \mid i \in [N], j \in [K]\} \tag{2}$$

关键：所有 $a_i^j \neq a_i$，让 agent 真正经历"非 expert 路径"的后果。这些 next state $\{s_i^j\}$ 编码了 environment 对 action 质量的隐式反馈——错误点击产生 error page、错误 API 调用返回 exception、错误搜索 query 检索不到相关文档。这种 supervision 不依赖任何外部 reward function。

直觉上，这就是把 **hindsight experience replay**（https://arxiv.org/abs/1707.01495）的思想从 "relabel goal" 改成 "直接把 transition 当 supervision"，但完全去掉了 reward 这一层。

---

## 4. 方法一：Implicit World Modeling (IWM)

把 world modeling 当作 auxiliary next-token prediction 任务，让 policy 自己内化 environment dynamics。

### 4.1 损失函数

对每个 rollout triple $(s_i, a_i^j, s_i^j)$，构造预测任务：输入 $(s_i, a_i^j)$，预测 $s_i^j$。

$$\mathcal{L}_{\mathrm{IWM}} = -\sum_{(s_i, a_i^j, s_i^j) \in \mathcal{D}_{\mathrm{rollout}}} \log p_\theta(s_i^j \mid s_i, a_i^j) \tag{3}$$

变量解释：
- $p_\theta$：language model 的 output token distribution
- $s_i^j$：上标 $j$ 标识第 $j$ 个 alternative action 产生的 next state
- **关键设计**：state prediction 和 action prediction 共享同一套参数 $\theta$，没有独立 world model 模块

### 4.2 为什么这样做 work

这与 Ha & Schmidhuber 的 "World Models"（https://arxiv.org/abs/1803.10122）和 Dreamer 系列（https://arxiv.org/abs/1912.01603）有渊源，但关键区别在于：传统 world model 是 **standalone simulator**，用于 model-based planning；而 IWM 把 dynamics prediction 直接 **塞进 policy 的参数里**，类似 mid-training（Zhang et al. 2025 https://arxiv.org/abs/2504.10127），让 policy 在做决策前先"暖身"理解环境规律。

具体来说，IWM 让模型学到：
- 哪些 action 会导致 error state（"输入无效日期 → 弹出错误提示"）
- transition 的 regularity（"点击 'Buy Now' 后进入 checkout 页"）
- side effect 和 invalid action 的边界

### 4.3 两阶段 pipeline

实际训练分两步：
1. 先用 $\mathcal{L}_{\mathrm{IWM}}$ 跑 1 epoch，让 model 粗略内化 dynamics
2. 再在 $\mathcal{D}_{\mathrm{expert}}$ 上用 $\mathcal{L}_{\mathrm{IL}}$ fine-tune

总 update step budget 和纯 IL 一致，不增加额外训练 step——只是把一部分 budget 用于 "warm-up"。

---

## 5. 方法二：Self-Reflection (SR)

让 agent 把对比性的 outcome 转化为可迁移的 natural language reasoning。

### 5.1 数据构造

对每个 expert state $s_i$：
1. 执行 expert action $a_i$，得到 $s_{i+1}$
2. 对每个 alternative $a_i^j$（$j \in \{1,\ldots,K\}$），得到 $s_i^j$
3. 用 LLM 在 prompt 下生成 chain-of-thought $c_i^j$，解释为什么 $a_i$ 优于 $a_i^j$，grounding 在 $s_{i+1}$ 和 $s_i^j$ 的差异上

数据集：

$$\mathcal{D}_{\mathrm{refl}} = \{(s_i, a_i^j, c_i^j)\}$$

### 5.2 损失函数

$$\mathcal{L}_{\mathrm{SR}} = -\sum_{(s_i, a_i^j, c_i^j) \in \mathcal{D}_{\mathrm{refl}}} \log p_\theta(c_i^j, a_i \mid s_i) \tag{4}$$

变量：
- $c_i^j$：上标 $j$ 表示针对第 $j$ 个 alternative 生成的 reflection chain
- $a_i$：expert action（被预测的目标）
- 联合预测 $c_i^j \circ a_i$（拼接序列），即先输出 reasoning 再输出 action

### 5.3 与 STaR 的关键区别

STaR（Zelikman et al. 2022, https://arxiv.org/abs/2203.14465）也让 model 生成 rationale 再 SFT，但 STaR 的 rationale **从未在 environment 中验证过**，只是 "post-hoc rationalization"，容易 hallucinate 工具或事实。

SR 的 rationale 是 **grounded** 的：它必须基于实际观察到的 $s_i^j$ 和 $s_{i+1}$ 的差异来论证。这把 LLM 的语言推理能力锚定到了 environment 的真实反馈上。

### 5.4 训练 setup

$\mathcal{D}_{\mathrm{refl}}$ 和 $\mathcal{D}_{\mathrm{expert}}$ 混合训练。Expert data 保留原始 CoT（如果有的话），SR data 用新生成的 $c_i^j$。这平衡了 "imitation 的 grounded decision-making" 和 "exploration 的 contrastive insight"。

### 5.5 一个直觉例子

WebShop 里，expert action 是 "click on the $15 blue shirt"，alternative 是 "click on the $30 red shirt"。生成的 reflection 可能是："虽然 red shirt 满足颜色偏好，但超出 $20 budget 约束。Blue shirt 同时满足 style 和 budget。" 

这教给模型的不是一个具体 item，而是 "**prioritize budget constraint over color preference**" 这种可迁移的 decision principle。

---

## 6. 实验全景

### 6.1 八个 environment 的覆盖

| Domain | Environment | # Traj. | # $\mathcal{D}_{\mathrm{expert}}$ |
|---|---|---|---|
| Embodied | ALFWorld | 3,553 | 21,031 |
| Scientific sim | ScienceWorld | 1,000 | 14,506 |
| Long-horizon planning | TravelPlanner | 45 | 1,395 |
| Multi-turn tool use | BFCLv3 | 125 | 1,264 |
| Customer service API | Tau-Bench | 452 | 5,239 |
| Multi-hop QA | SearchQA | 2,082 | 7,691 |
| Web shopping | WebShop | 1,571 | 15,464 |
| Web navigation | WebArena-Lite | 554 | 7,044 |

模型：Llama-3.2-3B、Qwen-2.5-7B、Llama-3.1-8B，外加 Llama-3.3-70B 做 scaling 实验。

### 6.2 In-domain 结果（Table 2 关键数字）

| Benchmark | Model | IL | IWM | SR |
|---|---|---|---|---|
| ALFWorld | 3B | 78.1 | 83.6 (+5.5) | 85.9 (+7.8) |
| ScienceWorld | 8B | 54.7 | 57.0 (+2.3) | 68.0 (+13.3) |
| TravelPlanner | 8B | 17.2 | 25.0 (+7.8) | 32.2 (+15.0) |
| BFCLv3 | 3B | 21.3 | 25.3 (+4.0) | 29.3 (+8.0) |
| Tau-Bench | 8B | 35.9 | 40.8 (+4.9) | 41.7 (+5.8) |
| WebShop | 3B | 41.8 | 60.2 (+18.4) | 52.7 (+10.9) |
| WebArena-Lite | 8B | 4.9 | 8.5 (+3.6) | 8.5 (+3.6) |

观察到的 pattern：
- **IWM 在 transition 稳定、observation 简单的环境强**（WebShop +18.4，ALFWorld +5.5）——因为 dynamics 可预测，next-state prediction 学得到东西
- **SR 在需要多步推理和约束满足的环境强**（TravelPlanner +15.0，ScienceWorld +13.3）——因为 failure 主要来自 reasoning error，contrastive reasoning 能直接修补
- **Open action space 环境（SearchQA、WebArena）增益小但稳定**——action space combinatorial，但仍能从 rollout 中提取 dense signal

### 6.3 OOD 泛化（Table 3）

| Benchmark | Model | IL | IWM | SR |
|---|---|---|---|---|
| ALFWorld | 8B | 63.3 | 78.1 (+14.8) | 72.7 (+9.4) |
| BFCLv3 | 3B | 5.3 | 8.9 (+3.6) | 13.8 (+8.5) |
| SearchQA | 8B | 47.4 | 49.6 (+2.2) | 50.7 (+3.3) |

关键 insight：**OOD 增益经常超过 in-domain 增益**。这说明 agent 自己的 experience 提供的 supervision 比 expert demonstration 更具泛化性——因为 expert 只覆盖 narrow scenario，而 self-exploration 触及了更宽的 state 分布。

### 6.4 RL warm-start（Figure 3，最关键的实验）

在 WebShop、ALFWorld、SearchQA 三个有 verifiable reward 的环境上跑 GRPO（Shao et al. 2024, https://arxiv.org/abs/2402.03300）。唯一变量是 RL 前的 initialization：

- **IL → GRPO**：标准 SFT warm start
- **IWM → GRPO** / **SR → GRPO**：early experience warm start
- **Raw pretrained → GRPO**：无 SFT 直接 RL（最差，训练不稳定）

结果：early experience 初始化的 RL ceiling **始终高于** IL 初始化。在 ALFWorld 上 gap 在 RL 过程中甚至放大；在 WebShop 上 gap 缩小但从不反转。具体数字（Llama-3.2-3B）：

- WebShop succ rate: IL+GRPO 82.0 vs IWM+GRPO 92.2 vs SR+GRPO 89.8
- ALFWorld: IL+GRPO 92.2 vs IWM+GRPO 97.7 vs SR+GRPO 99.2
- SearchQA (MuSiQue F1): IL+GRPO 43.6 vs IWM+GRPO 44.6 vs SR+GRPO 43.1

这是 paper 最 strong 的 claim：**early experience 是 IL 到 RL 的 practical bridge**。RL 收益被 initialization 质量放大，early experience 提供的 initialization 让 RL 起点更高、收敛更好。

### 6.5 Data efficiency（Figure 4a）

在 WebShop 上，**只用 1/8 的 expert trajectory** 配合 early experience，就能超过用全部 expert data 的纯 IL。ALFWorld 上 1/2 即可。这说明 early experience 提供的 supervision 量级远超 expert demonstration 本身——rollout 数据通常是 expert data 的 10 倍量级。

### 6.6 Branching factor K 的影响（Figure 4b）

- **IWM 单调偏好大 K**：更多 alternative → 更丰富的 transition 覆盖
- **SR 最佳在 K=2-4**：太大时，alternative 里可能混入其他 success-leading action，削弱与 expert 的 contrast；且 context window 限制了一次 reasoning 多个 alternative 的能力

### 6.7 Model scaling（Figure 5）

WebArena 上从 3B → 8B → 70B（LoRA），early experience 相对 IL 的 gap 在每个 scale 上都保持。这说明 early experience 提供的 supervision **与 model capacity 互补**，不会被规模抵消。

---

## 7. 与两个关键 baseline 的对比（Table 4，Llama-3.1-8B）

| Method | WebShop | ALFWorld |
|---|---|---|
| Prompt + Long CoT | 0.0 | 25.0 |
| IL | 47.3 | 80.5 |
| IL + Long CoT | 0.0 | 25.8 |
| IL + STaR | 25.0 | 74.2 |
| IWM | 58.6 | 85.9 |
| SR | 58.2 | 85.2 |

两个 negative result 很有启发：

1. **Long CoT 在 fine-tuned model 上失效**：一旦在缺乏 rationale 的 expert data 上 SFT，模型失去了维持长链推理的能力。Truncation at `` 强行延长反而 drift 到 invalid action。这印证了 Chu et al. 2025（https://arxiv.org/abs/2505.01781）的发现——SFT memorizes, RL generalizes，但 SFT 也会破坏 pre-trained 的 reasoning 能力。

2. **STaR-style 反而 degrade**：rationale 没有在 environment 中验证，match rate 低，留下的 rationale 经常 hallucinate 工具或事实。Fine-tune 在这些 ungrounded rationale 上会让 model 学到错误因果关系。

这两个对比强化了 paper 的核心论点：**grounding in actual environment outcome 是 essential 的**，单纯加 reasoning length 或 ungrounded rationale 都不行。

---

## 8. 局限与 open question

1. **Short-horizon trace 限制**：当前 IWM 和 SR 都聚焦于单步 transition。Long-horizon credit assignment without reward 仍是 open problem——比如一个 10 步 trajectory 第 3 步的 suboptimal action 如何归因到最终 outcome。
2. **K 的 trade-off**：SR 在大 K 下退化，说明 current model 的 context 处理多 alternative 能力有限。未来可能需要 hierarchical reflection。
3. **Environment simulator 依赖**：rollout 需要可重置的 environment。对真实 web（不可重置、有 side effect）如何 scale 仍是问题。Paper 提到希望未来用 organically collected interaction data 做 continual learning。
4. **与 RL 的衔接机制**：目前是 sequential（early experience → RL），没有探索 interleaved 或 joint training。

---

## 9. 我的 intuition 构建

把这篇 paper 放在更大的图景里看：

- **Pre-training**：model 学到了 language 和世界知识
- **SFT on expert**：model 学到了 "在 state $s$ 下做 action $a$" 的 mapping，但不知道为什么
- **Early experience**：model 学到了 "在 state $s$ 下做 action $a$ 会导致 $s'$" 的因果模型，以及 "为什么 $a$ 比 $a'$ 好" 的 reasoning
- **RL**：model 学到了 "在 state $s$ 下做 action $a$ 能最大化累积 reward" 的最优策略

Early experience 填补的是 SFT 和 RL 之间的 **representation gap**：SFT 给的是 state→action 的 lookup table，RL 需要的是 state→value→action 的规划能力。Early experience 通过 world modeling 和 contrastive reasoning，让 model 在没有 reward 的情况下，先建立起对 environment dynamics 和 decision principle 的内部表征。这个表征是 RL 的好起点——RL 不再需要从零探索 dynamics，只需要 refine value 估计。

这与你在 NanoGPT 和教育内容里反复强调的 "理解机制比堆 trick 重要" 一致：early experience 本质上是让 agent 在 RL 之前先 "理解" environment，而不是盲目 trial-and-error。

---

## 参考链接

- Silver & Sutton, "Welcome to the Era of Experience" (2025): https://blog.google/research/2025/era-of-experience/
- DAgger (Ross et al. 2011): https://arxiv.org/abs/1011.0686
- Hindsight Experience Replay: https://arxiv.org/abs/1707.01495
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dreamer V2: https://arxiv.org/abs/2010.02193
- STaR (Zelikman et al. 2022): https://arxiv.org/abs/2203.14465
- Self-Refine (Madaan et al. 2023): https://arxiv.org/abs/2303.17651
- Reflexion (Shinn et al. 2023): https://arxiv.org/abs/2303.11366
- GRPO / DeepSeekMath (Shao et al. 2024): https://arxiv.org/abs/2402.03300
- WebShop: https://arxiv.org/abs/2207.01296
- ALFWorld: https://arxiv.org/abs/2010.03768
- WebArena: https://arxiv.org/abs/2307.13854
- TravelPlanner: https://arxiv.org/abs/2402.01622
- Search-R1: https://arxiv.org/abs/2503.09516
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html
- Tau-Bench: https://arxiv.org/abs/2404.04428
- ScienceWorld: https://arxiv.org/abs/2203.07740
- LLaMA-Factory: https://arxiv.org/abs/2404.02348
- OSWorld (concurrent work on world modeling): https://arxiv.org/abs/2411.06559
- Dyna-Think (concurrent): https://arxiv.org/abs/2506.00320
- Chu et al. "SFT memorizes, RL generalizes": https://arxiv.org/abs/2505.01781
