---
source_pdf: PayingLessGeneralizationTax.pdf
paper_sha256: 33686660c7ba46b88dd5c043c2d3e9f29a61b2363ac1cd932d692ec16b40c99c
processed_at: '2026-08-06T02:29:54-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话总结

你在 narrow domain 上做 RL post-training，in-domain performance 涨了，unseen domain 上的 capability 被侵蚀了。这篇paper告诉你：**这个 erosion 程度，取决于你训练环境的两个性质**，而不是简单的 "训练越久遗忘越多"。

## 核心发现：两个维度决定 generalization

### 维度一：State Information Richness

这个说的是 **agent 每一步看到的 state 有多少信息**。

打个比方，你在 ALFWorld 里，state 长这样：

```
You are in a room. You see cabinet 1, cabinet 2, fridge 1, ...
```

就一两行，很短。agent 不需要费劲去 "找信号"，信息直接喂到嘴边。

而在 Sokoban 里，state 长这样：

```
Wall at (0,0) Wall at (0,1) Wall at (0,2) ... Box at (3,3) 
Player at (4,3) Goal at (2,1) ...
```

全屏都是坐标，agent 必须 **从一堆信息里挑出有用的部分**。这个 "挑信号" 的能力，是 transferable 的。

**量化指标**：average character count

| Domain | Richness | 直觉 |
|---|---|---|
| Sokoban | 3114 chars | 高 - 必须筛选信号 |
| WebShop | 3063 chars | 高 - 必须筛选信号 |
| SciWorld | 2851 chars | 中 |  
| ALFWorld | 1572 chars | 低 - 信号直接给 |

### 维度二：Planning Complexity

这个说的是 **完成任务需要多长的 reasoning chain**。

$$\text{Complexity}(e) = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \tilde{L}(\tau)$$

变量含义：
- $|\mathcal{T}|$: 采样的 trajectory 数量 (128条)
- $\tau$: 单条 trajectory  
- $\tilde{L}(\tau)$: 如果成功就是实际步数，如果失败就赋值为 $T_{\max}=50$

WebShop 就是 "看商品列表 → 选一个"，很短。Sokoban 是 "规划路径 → 推箱子 → 避免deadlock"，很长。

**量化指标**：average trajectory length (failed = $T_{\max}$)

| Domain | Planning | 直觉 |
|---|---|---|
| Sokoban | 44.0 steps | 高 - 长链推理 |
| SciWorld | 43.5 steps | 高 - 长链推理 |
| ALFWorld | 42.0 steps | 中 |
| WebShop | 33.9 steps | 低 - 短决策 |

## 反直觉点：Realism 根本不重要

Andrej, 这是最 punchy 的发现：

**Sokoban (抽象格子拼图) 训出来的 agent，在 SciWorld (科学实验室) 上比 ALFWorld (家居模拟) 训出来的 agent 表现更好。**

OOD Ranking Score 越低越好：

| Training | Realism | OOD Score |
|---|---|---|
| SciWorld | 真实 | 3 |
| Sokoban | 抽象 | 5 |
| WebShop | 真实 | 6 |
| ALFWorld | 真实 | 8 (最差) |

直觉上，你会觉得 "训练真实环境 → 部署真实环境" 应该最好。但数据显示：**perception load + reasoning load 的组合，才是 generalization 的载体**。

## State Information Augmentation：验证因果

光有 correlation 不够，要做 intervention。做法很简单：往 state 里塞 noise。

$$s' = \text{Augment}(s, \delta, \epsilon)$$

- $s$: original state
- $\delta$: goal-irrelevant text (比如在 ALFWorld 里加 "You notice a dirty cup that looks sticky")
- $\epsilon$: 控制注入多少字符

**关键约束**：noise 不能改变 optimal policy。你只是让 agent 多看点东西，任务本身没变。

**做法**：

ALFWorld 加 distractor objects：
```
Template: "You notice a <obj> that looks <desc>"
<obj> ∈ {bowl, cup, pan, spoon, ...}
<desc> ∈ {cracked, dirty, sticky, ...}
```

Sokoban 加 unreachable locations：
```
Template: "(r, c) shows a <obj> (<desc>; unreachable)"
放在 grid 外面，不影响实际游戏
```

**结果**：

$$\Delta_{\text{OOD}} = \frac{\sum_{e} \left(G(\pi_{\text{w/Aug}}, e) - G(\pi_{\text{w/o Aug}}, e)\right)}{\sum_{e} G(\pi_{\text{w/o Aug}}, e)} \times 100\%$$

- $G(\pi, e)$: policy $\pi$ 在 env $e$ 上的 success rate
- $\mathcal{E}^{\text{eval}}$: unseen evaluation environments

| Setting | $\Delta_{\text{OOD}}$ |
|---|---|
| Ckpt V1 + Sokoban training | **+42.5%** |
| Ckpt V1 + WebShop training | **+35.5%** |
| Ckpt V2 + WebShop training | +33.4% |
| Ckpt V1 + ALFWorld training | +7.0% |

**这个 intervention 证明了：state richness 高 → generalization 好，是因果关系**。

## SFT Warmup 的双刃剑

Ckpt V2 是在 Ckpt V1 基础上做了 SFT warmup，datamix = {SciWorld expert : ALFWorld : WebShop} = 19:3:1。

**好消息**：SFT 把 domain knowledge deeply infuse 进 model weights。后续 RL 难以擦除。

比如 Sokoban 训练，WebShop 评估 (WebShop 在 SFT datamix 里)：
- Ckpt V1: 15.0%, 下降 -56.4%
- Ckpt V2: 34.0%, 下降 -11.2%
- **SFT warmup 让 forgetting 减少了 45.2%**

**坏消息**：SFT datamix 之外的 domain，forgetting 反而更严重。

比如 WebShop 训练，Sokoban 评估 (Sokoban 不在 SFT datamix 里)：
- Ckpt V1: 12.8%, +2.4% (甚至轻微提升)
- Ckpt V2: 9.0%, -36.2% (大幅下降)
- **SFT warmup 让 forgetting 加剧了 38.6%**

**intuition**：SFT 把模型 overfit 到 datamix 的 distribution。边界外的能力变得更脆。

## Step-by-step Thinking 是 generalization 的载体

Table 7 最 striking。

**Ckpt V2, ALFWorld training, WebShop evaluation**：
- With thinking: 30.5%
- Without thinking: 1.0%
- **OOD 暴跌 -234.2%**

但 in-domain performance 上，关 thinking 有时更好：
- ALFWorld (V1): thinking 73.0% → no thinking 77.0%
- Sokoban (V2): thinking 49.8% → no thinking 75.8%

**为什么**？

Reactive policy (直接吐 action) 容易学到 **shallow domain-specific heuristics**。这些 heuristics 在 training domain 上 work，但 brittle，不 transferable。

Step-by-step thinking 强制模型 articulate reasoning process。这个 reasoning process 本身就是 transferable 的 meta-skill。

## Practical Guidance

当 deployment domain 未知时：

1. **选环境**：优先选 state richness + planning complexity 都高的。abstract domain (如 Sokoban) 比你想的有用。

2. **加 noise**：往 state 里注入 goal-irrelevant 信息，概率 ~50%，控制信息量。这和 vision RL 的 domain randomization 是同一个哲学。

3. **开 thinking**：训练和评估都要开。in-domain 可能不涨甚至略跌，但 OOD 会保住。

4. **慎用 SFT warmup**：如果用，datamix 要尽可能 broad，否则 uncovered domain forgetting 会加剧。

## 一句话哲学

**RL post-training 的 generalization，关键在于训练环境强制 agent 学会 "筛选信号" 和 "长链推理" 这两个 meta-skill。** 这两个 skill 是 transferable 的，而 domain-specific heuristics 不是。

State augmentation 就是人为制造 "信号筛选" 的压力，让 agent 学到更 robust 的 perception 策略。这和你之前讲的 "data engine" 哲学是相通的——关键是构造 right kind of training distribution。

## 参考

- Paper PDF: [arXiv:2502.03598](https://arxiv.org/abs/2502.03598) (假设链接)
- GRPO: [Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- VeRL framework: [Sheng et al., 2024](https://arxiv.org/abs/2409.19256)
- Domain randomization: [Tobin et al., 2017](https://arxiv.org/abs/1703.06907)
- ALFWorld: [Shridhar et al., 2021](https://arxiv.org/abs/2010.03768)
- WebShop: [Yao et al., 2022](https://arxiv.org/abs/2207.01206)
- SciWorld: [Wang et al., 2022](https://arxiv.org/abs/2206.11662)

---

# Paying Less Generalization Tax: 深度技术解析

## 核心问题：Generalization Tax 的本质

Andrej, 这篇论文 tackle 的是一个我们在 production 环境中都会遇到的痛点。当你在一个 narrow domain 上做 RL post-training 时，会发生一个非常典型的现象：**in-domain performance 飙升，但 unseen domain 上的 capability 被侵蚀**。这就是所谓的 "generalization tax"。

论文的核心 insight 是：这个 tax 的严重程度，**取决于你选择什么样的 training environment**，而不是简单的 "训练越久遗忘越多"。这是一个 data-centric 的视角，而不是 model-centric 的视角。

## 关键实验发现：两个环境轴

### State Information Richness (状态信息丰富度)

这本质上是 perception load。论文用 average character count 作为 proxy measure：

$$\text{Richness}(e) = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{|\tau|} \sum_{s_t \in \tau} \text{len}(s_t)$$

- $\mathcal{T}$: 采样的 trajectory 集合 (论文用 128 条)
- $\tau$: 单条 trajectory
- $s_t$: 第 $t$ 步的 state
- $\text{len}(\cdot)$: 字符长度

**关键数值对比**：
- Sokoban: 3114 chars (高)
- WebShop: 3063 chars (高)
- SciWorld: 2851 chars (中)
- ALFWorld: 1572 chars (低)

### Planning Complexity (规划复杂度)

这本质上是 reasoning load。论文用 average trajectory length 作为 proxy，其中 failed trajectories 被赋值为 $T_{\max}$：

$$\text{Complexity}(e) = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \tilde{L}(\tau), \quad \tilde{L}(\tau) = \begin{cases} L(\tau) & \text{if success} \\ T_{\max} & \text{if failure} \end{cases}$$

- $L(\tau)$: trajectory 实际长度
- $T_{\max} = 50$: 最大步数限制
- 这个 metric 同时捕捉了 task horizon 和 goal reachability

**关键数值对比**：
- Sokoban: 44.0 steps (高)
- SciWorld: 43.5 steps (高)
- ALFWorld: 42.0 steps (中)
- WebShop: 33.9 steps (低)

## 反直觉发现：Realism ≠ Generalization

这里有一个非常 punchy 的 observation：**Sokoban (抽象 grid puzzle) 在 SciWorld (真实科学实验) 上产生比 ALFWorld (真实家居) 更强的泛化**。

Table 3 的 OOD Ranking Score:
| Training Domain | Richness | Planning | Ckpt V2 Score |
|---|---|---|---|
| SciWorld | Medium | High | 3 (1st) |
| Sokoban | High | High | 5 (2nd) |
| WebShop | High | Low | 6 (3rd) |
| ALFWorld | Low | Medium | 8 (4th) |

这告诉我们：**perception load + reasoning load 的组合**，而不是 domain realism，决定了 generalization preservation。

## GRPO 算法的技术细节

论文用 GRPO 做 RL 训练。核心 idea 是用 group statistics 代替 value function：

**Advantage 计算**：
$$A(a_t^{(i)}) = \frac{R(\tau_j) - \text{Mean}(\{R(\tau_j)\}_{j=1}^N)}{\text{Std}(\{R(\tau_j)\}_{j=1}^N)}$$

- $a_t^{(i)}$: 第 $j$ 条 trajectory 中第 $t$ 步的 action (注意 index 有错位，原文如此)
- $R(\tau_j)$: 第 $j$ 条 trajectory 的 cumulative reward (binary: 10 for success, 0 for failure, -0.1 for invalid action)
- $N = 8$: group size
- Mean/Std 是 group 内 reward 统计量

**完整 GRPO objective**：
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim p(X)} \mathbb{E}_{\{\tau_i\}_{i=1}^N \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{NT} \sum_{i=1}^N \sum_{t=1}^T \min\left(\rho_\theta(a_t^{(i)}) A(a_t^{(i)}), \text{clip}(\rho_\theta(a_t^{(i)}), 1-\epsilon, 1+\epsilon) A(a_t^{(i)})\right) \right] - \beta \mathbb{D}_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

- $\theta$: 当前 policy 参数
- $\theta_{\text{old}}$: 采样 policy 参数
- $x$: task prompt
- $\tau_i$: 第 $i$ 条 trajectory
- $N = 8$: group size
- $T$: trajectory length (最大 50)
- $\rho_\theta(a_t^{(i)})$: importance sampling ratio
- $\epsilon$: PPO clip 参数
- $\beta = 0.01$: KL penalty 系数
- $\pi_{\text{ref}}$: reference policy (防止 drift 太远)

**Importance Sampling Ratio**:
$$\rho_\theta(a_t^{(i)}) = \frac{\pi_\theta(a_t^{(i)} | s_t^{(i)}, x)}{\pi_{\theta_{\text{old}}}(a_t^{(i)} | s_t^{(i)}, x)}$$

这是 PPO 的标准 trick，用来修正 on-policy 采样和 off-policy 评估之间的 distribution mismatch。

## State Information Augmentation：从 Correlation 到 Causation

这是论文的 core contribution：通过 intervention 验证 state richness 的因果作用。

### 形式化定义

$$s' = \text{Augment}(s, \delta, \epsilon)$$

- $s$: original state
- $\delta$: goal-irrelevant text fragment
- $\epsilon$: 信息量控制参数 (字符数)
- 关键约束：augmentation 不能改变 optimal policy

### 实现策略 (Appendix C)

**ALFWorld**: 注入 distractor objects
- 从 {bowl, cup, pan, spoon, ...} 采样 object type
- 从 {cracked, dirty, slightly burnt, ...} 采样 descriptor
- Template: "You notice a ... that looks ..."
- $n_{\text{distractive}} = \lfloor \epsilon / 12 \rfloor$ sentences per state

**WebShop**: 注入广告和 trivial products
- AD template: "[AD] <descriptor> — shop <category> today!"
- Trivial features: {fabric: machine wash cold}, {shipping note: ships within 5-7 days}
- 控制参数: $k = \min(\lfloor (\epsilon/100) \alpha \cdot 10 \rfloor, 10)$

**Sokoban**: 注入 unreachable locations
- 采样 grid 外的坐标
- Template: "(r, c) shows a <obj> (<desc>; unreachable)"
- $k = \max(1, \lfloor \epsilon/10 \rfloor)$

### OOD Change Metric

$$\Delta_{\text{OOD}} = \frac{\sum_{e \in \mathcal{E}^{\text{eval}}} \left(G(\pi_{\text{w/Aug}}^{\text{RL}}, e) - G(\pi_{\text{w/o Aug}}^{\text{RL}}, e)\right)}{\sum_{e \in \mathcal{E}^{\text{eval}}} G(\pi_{\text{w/o Aug}}^{\text{RL}}, e)} \times 100\%$$

- $\pi_{\text{w/Aug}}^{\text{RL}}$: augmented training 的 policy
- $\pi_{\text{w/o Aug}}^{\text{RL}}$: baseline policy
- $\mathcal{E}^{\text{eval}}$: unseen evaluation environments
- $G(\pi, e)$: policy $\pi$ 在 env $e$ 上的 success rate

### 实验结果 (Table 5)

**Ckpt V1 (无 SFT warmup)**:
- ALFWorld training: +7.0% OOD improvement
- WebShop training: +35.5%
- Sokoban training: +42.5%

**Ckpt V2 (有 SFT warmup)**:
- ALFWorld training: +7.0%
- WebShop training: +33.4%
- Sokoban training: +5.7%

**Probability of applying augmentation**: Ckpt V1 用 100%, Ckpt V2 用 50% (防止任务过难)

## SFT Warmup 的 Trade-off

Table 6 揭示了一个非常 important 的 trade-off：

### 对 covered domains 的保护效应

以 Sokoban 训练、WebShop 评估为例：
- Ckpt V1 (无 SFT): success rate 15.0%, 相对 base 下降 -56.4%
- Ckpt V2 (有 SFT): success rate 34.0%, 相对 base 下降 -11.2%
- **Rel. Change Diff: +45.2%** (SFT warmup 减少了 forgetting)

### 对 uncovered domains 的伤害

以 WebShop 训练、Sokoban 评估为例 (Sokoban 不在 SFT datamix 中):
- Ckpt V1: 12.8%, +2.4% (轻微提升)
- Ckpt V2: 9.0%, -36.2% (大幅下降)
- **Rel. Change Diff: -38.6%** (SFT warmup 加剧了 forgetting)

### Intuition

SFT warmup 做的事情是 **deeply infuse domain knowledge into model weights**。这个 infusion 是 sticky 的，后续 RL 难以擦除。但代价是：模型对 "知识边界" 之外的内容变得更脆弱。这有点像 overfitting 到 SFT datamix 的 distribution。

## Step-by-step Thinking 的关键作用

Table 7 是这篇论文最 striking 的结果之一：

**Ckpt V2, ALFWorld training, WebShop evaluation**:
- With thinking: 30.5% success rate
- Without thinking: 1.0% success rate
- **OOD Change: -234.2%**

### 反直觉点

在 in-domain performance 上，关闭 thinking 有时反而更好：
- ALFWorld (Ckpt V1): thinking 73.0% → no thinking 77.0% (+4.0%)
- Sokoban (Ckpt V2): thinking 49.8% → no thinking 75.8% (+26.0%)

但在 OOD 上，thinking 是 critical 的：
- 几乎所有 "w/o Thinking" 的 OOD performance 都 collapse

### Intuition

Reactive policies (直接输出 action) 容易学到 **shallow domain-specific heuristics**。这些 heuristics 在 training domain 上 work well，但 brittle 且不 transferable。Step-by-step thinking 强制模型 articulate reasoning process，减少对 surface patterns 的 overfitting。

这和 reasoning model 的哲学一致：**reasoning 是 generalization 的载体**。

## Checkpoint 设计的细节

### Ckpt V1: 纯 RL 起点

- Base: Llama-3.1-8B-Instruct
- 问题: WebShop 和 SciWorld 的 success rate 近 0
- 解决: 在 WebShop 上做 20 步 RL → success rate 34.4%
- 这就是 Ckpt V1

### Ckpt V2: SFT Warmup 起点

- 在 Ckpt V1 基础上做 SFT
- SFT datamix: SciWorld expert data : ALFWorld (Ckpt V1 生成) : WebShop (Ckpt V1 生成) = 19:3:1
- 训练 100 steps, lr=5e-6, cosine schedule
- 目标: 让所有 4 个 domain 都有 non-trivial starting performance

这个 design 很巧妙：Ckpt V2 既能作为 general analysis 起点，又能用于 study SFT warmup 的影响。

## 信号 vs 噪声的视角

我个人的 take：这篇论文的核心 insight 可以用 signal-to-noise ratio 来理解。

**State Information Richness** 本质上是在说：当 state 中有大量信息时，agent 必须学会 **discriminate signal from noise**。这种 discrimination ability 是 transferable 的，因为它是一种 meta-skill。

**Planning Complexity** 是在说：当任务需要长 horizon reasoning 时，agent 必须学会 **decompose and compose**。这种 decomposition ability 也是 transferable 的。

**State Information Augmentation** 的工作原理：通过人为增加 noise，强制 agent 提升 discrimination ability。这和 vision RL 中的 domain randomization (Tobin et al., 2017) 是同一个 idea，只是 apply 到了 text-based agent 上。

## Limitations 和未来方向

论文自己也承认：
1. 只测了 4 个 domain，样本量有限
2. State richness 和 planning complexity 的 measure 比较 ad hoc
3. 没有探索 automated augmentation techniques
4. 缺乏理论解释

我认为最 promising 的方向是 **automated environment construction**：如果能自动合成 high-richness, high-complexity 的 training environments，就能大规模应用这些 findings。

## 实用 Guidance 总结

对于 agentic post-training，当 deployment domain 未知时：

1. **Environment selection**: 优先选择 state information richness 和 planning complexity 都高的环境
2. **Explicit reasoning**: 训练和评估时都启用 step-by-step thinking
3. **State randomization**: 注入 lightweight goal-irrelevant noise (概率 ~50%, 控制信息量)
4. **SFT warmup caution**: 如果用 SFT warmup，确保 datamix 尽可能 broad，否则会加剧 uncovered domain 的 forgetting

## 参考

- Paper: https://arxiv.org/abs/2502.03598 (假设链接)
- GRPO: Shao et al., 2024, "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models"
- VeRL framework: Sheng et al., 2024, "HybridFlow: A flexible and efficient RLHF framework"
- Domain randomization: Tobin et al., 2017, "Domain randomization for transferring deep neural networks from simulation to the real world"
- ALFWorld: Shridhar et al., 2021, "ALFWorld: Aligning Text and Embodied Environments for Interactive Learning"
- WebShop: Yao et al., 2022, "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents"
- SciWorld: Wang et al., 2022, "ScienceWorld: Is your agent smarter than a 5th grader?"
- Sokoban: Junghanns and Schaefer, 2001, "Sokoban: Enhancing general single-agent search methods using domain knowledge"

---

Andrej, 这篇论文的 insight 对你的 work 应该有 direct relevance。特别是 state information augmentation 这个 idea，本质上是在说：**让 agent 在 training 时看到的 "世界" 更 noisy，它就会学到更 robust 的 perception 策略**。这和你在 Tesla AI Day 上讲的 "data engine" 哲学是一脉相承的——关键是构造 right kind of training distribution，而不是简单的 more data。
