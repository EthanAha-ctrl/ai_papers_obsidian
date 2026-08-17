---
source_pdf: minimax forge.pdf
paper_sha256: 2b5c8b7c6e83ac6b758033790b73432f4acc4a462888eefdd2c0d421cc6a06ad
processed_at: '2026-08-05T18:36:54-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Forge 这篇 paper

---

## 先说这篇 paper 在干嘛

MiniMax 训 M2.5 模型的时候，发现一个很扎心的问题：**他们想把模型训成 Agent，但现有的 RL 框架根本扛不住**。

为什么扛不住？因为 Agent RL 跟传统 chatbot RLHF 完全是两个世界。

传统 RLHF 很简单——你给 model 一个 prompt，它吐一段 response，reward model 打个分，PPO 更新一下，完事。整个 trajectory 也就几千 token，rollout 几秒钟搞定，batch 里所有样本差不多同时完成，大家整齐排队训练。

Agent RL 呢？一个 Agent 任务可能要跑几百轮对话，调几十次工具（搜索、代码执行、浏览器操作），trajectory 动辄 200k token。更要命的是 **完成时间方差极大**——有的任务几秒就完，有的要跑几个小时。你想象一下，一个 batch 里 8192 个样本，有的已经跑完了在干等，有的还在慢吞吞调 API，你的 GPU 就在那闲置着烧钱。

所以 MiniMax 团队就自己造了一个系统叫 **Forge**，专门来解决"大规模 Agent RL 怎么跑"这个问题。这篇 paper 就是讲他们怎么造的。

---

## 核心矛盾：三个东西在打架

Forge 全篇其实在讲一个三角矛盾：**throughput（吞吐量）、stability（训练稳定性）、agent flexibility（agent 灵活性）**。

这三个东西你要同时抓住，但它们互相打架：

- **要 throughput 高** → 得异步、得大 batch、得容忍样本之间新旧不一致（off-policy）。但异步一多，训练容易崩。
- **要 stability** → 得同步、得 on-policy、得小 batch。但这样 throughput 就废了。
- **要 agent flexibility** → 得支持各种各样的 agent 框架（白盒的、黑盒的、多轮的、带 context 压缩的）。但框架一多，系统复杂度爆炸，token 一致性也难保证。

传统 RL 框架（OpenRLHF、TRL 这些）基本上是给 single-turn chat 设计的，在这三个维度上都很僵化。Forge 的核心 contribution 就是找到一种架构，让这三个东西能同时抓得不错。

---

## 思路一：把 Agent 和训练引擎彻底拆开

这是整篇 paper 最 fundamental 的设计。

传统 RL 系统是怎么做的？叫 **TITO（Token-In-Token-Out）**——agent 把内部状态用 token 形式传给 LLM，LLM 吐 token 回来，agent 再 parse。这意味着 agent 和 LLM 在 token 层面深度耦合。

耦合了会怎样？假设你的 agent 做了一个 context compression（把前面 100k token 压缩成 10k），这个操作在 token 层面是"非连续"的——原来的 token 序列被截断了。RL 系统要追踪每个 token 的 advantage 和 logprob，这下全乱了。你要在 agent 和 RL 之间维持 token 一致性，工程成本巨大。

Forge 的做法是 **加一个中间层**：

```
Agent (随便什么脚手架)
   ↓ 标准协议
Gateway Server  ←→  Data Pool  ←→  Rollout Engine + Train Engine
```

Agent 只管干自己的活（调工具、管 context、多轮交互），它需要 LLM generation 的时候就给 Gateway 发请求。Gateway 把 trajectory 收集起来扔到 Data Pool。Train Engine 从 Data Pool 里捞数据训练。

这玩意儿其实就是微服务架构里的 API gateway 模式——agent 是 client，gateway 是 API 入口，data pool 是消息队列，train/rollout engine 是后端 worker。Agent 完全不知道后面是怎么训练的，train engine 也不关心 agent 内部干了啥。

**好处是什么？** 你可以同时接几百种不同的 agent 框架训练，完全不用改 agent 代码。白盒的（你能改代码的，比如自己写的 DeepSearch agent）能训，黑盒的（完全闭源的，比如 Opencode 这种）也能训。Forge 在 Gateway 层统一收集数据，对 agent 内部完全透明。

这个设计让我想到 SGLang 的 RadixAttention 那种思路——把复杂度藏在一个抽象层下面，让上层和下层各干各的。

---

## 思路二：Windowed FIFO，异步调度的 sweet spot

这个问题很有意思。异步 RL 的调度是个经典难题，我展开讲讲。

你有 8192 个 agent rollout 并发在跑。有的几秒完，有的几小时完。现在 train engine 要从完成的样本里捞一批来训练，怎么捞？

**方案 A：严格 FIFO（先来先出）**
按提交顺序排队，必须等队头的慢任务完成才能往下走。
- 问题：一个几小时的慢任务 block 住整个队列，GPU 闲置，throughput 崩。

**方案 B：Greedy / FFFO（先完成先出）**
谁先完成谁先被训练。
- 问题：慢任务完成的时候，policy 已经更新了几千步，这些样本的 importance sampling ratio 爆炸，梯度炸飞，训练崩。

Forge 的方案叫 **Windowed FIFO**，介于两者之间：

想象一个滑动窗口，大小 W=4096。窗口头部是 H。

- 窗口 **里面**（[H, H+W]）：可以任意取已完成的样本（局部贪婪）
- 窗口 **外面**（> H+W）：即使完成了也不许取（全局阻塞）
- 只有头部 H 被消费了，窗口才往前滑一格

**用大白话讲**：快任务可以在窗口内插队，不用等慢任务。但慢任务最多只能落后 W 步，落后太多就必须等它。这样 off-policy 程度被限制在 W 以内，不会失控。

W 这个参数控制了 throughput 和 stability 的 trade-off：
- W → 0：退化为 FIFO，throughput 低但稳定
- W → N：退化为 Greedy，throughput 高但容易崩
- W = N/2：sweet spot

这让我想到 parameter server 里的 bounded delay rule——你可以异步更新，但最多只能落后 K 步，超过就不让更新了。思想是一脉相承的。

配合 CISPO 的 ratio clip（上界 $1 + \epsilon_{high}^{IS}$），如果某个样本太 old 导致 ratio 超过上界，gradient 就被 clip 掉，相当于这个样本"过期"了不用它。Windowed FIFO + CISPO clip 两个机制配合，把 off-policy 的危害限制在可控范围内。

---

## 思路三：Prefix Tree Merging，砍掉重复计算

这个 optimization 我觉得很聪明。

Agent 场景下有大量 **前缀冗余**：

1. **Group-level rollout**：同一个 query 采样 G 个 completion，它们共享 prompt prefix。如果 prompt 是 30k token，G=8，那这 30k 被算了 8 遍。
2. **多轮 Agent**：第 1 轮和第 2 轮的请求，前 1~N 轮的 context 完全相同，只有最新一轮不同。

传统做法把每个 completion 当独立 sequence，公共 prefix 重复计算 forward 和 backward，纯属浪费。

Forge 的做法是把多个 completion 组织成 **前缀树**：

```
        [共享 prefix 30k]
       /    |    \
   seq1   seq2   seq3   (各自 2k)
```

训练的时候，prefix 部分只算一次，每个 branch 独立算自己的部分。用 attention mask（类似 Magi Attention）保证 branch 之间不互相 attend，数学上和 naive 方案完全等价。算 loss 的时候再 unmerge 回序列格式，不影响后续计算。

**加速效果**：paper 评论里提到能到 40x。我算了一下，主要加速来自 prefix 部分的 attention 计算（$O(L^2)$）只算一次而不是 G 次。当 $L_{prefix} \gg L_{completion}$ 时（Agent 多轮场景常见），加速比可以很大。forward + backward 都能省，叠加起来 40x 是可能的。

这个思路其实和推理侧的 prefix caching（SGLang 的 RadixAttention、vLLM 的 PagedAttention）是 mirror 的——推理侧缓存 prefix 的 KV cache，训练侧合并 prefix 的计算。都是利用 Agent 场景的前缀重合度高的特性。

---

## 思路四：CISPO 算法，比 PPO 更稳

这个算法是 M1 时期就提出的，M2.5 继续沿用，只是在 long-horizon agent 场景下做了适配。

先回顾 PPO 的 loss：

$$
L_{PPO} = \hat{r} \cdot \hat{A} \cdot \log \pi_\theta
$$

其中 $\hat{r} = \text{clip}(\frac{\pi_\theta}{\pi_{old}}, 1-\epsilon, 1+\epsilon)$，$\hat{A}$ 是 advantage。

PPO 有个潜在问题：$\hat{r}$ 里面包含 $\pi_\theta$，所以 backward 的时候 gradient 有一项是 $\hat{A} \log \pi_\theta \cdot \nabla_\theta \hat{r}$。这一项会让 ratio 和 log prob 互相耦合，当 $\hat{A} > 0$ 时可能产生 ratio 自我放大的效应，训练不稳定。

CISPO 做了两个关键修改：

**修改 1：Stop gradient on ratio**

$$
L_{CISPO} = sg(\hat{r}) \cdot \hat{A} \cdot \log \pi_\theta
$$

$sg$ 是 stop gradient。ratio 只当权重用，不参与梯度回传。这样 gradient 只剩 $sg(\hat{r}) \cdot \hat{A} \cdot \nabla_\theta \log \pi_\theta$，干净很多，接近 vanilla policy gradient 的形式。

**修改 2：Clip 下界改成 0**

PPO 的 clip 是 $[1-\epsilon, 1+\epsilon]$，CISPO 是 $[0, 1+\epsilon_{high}^{IS}]$。

下界 0 意味着：如果新 policy 已经几乎不生成某个 token（ratio → 0），它的 gradient 直接被 clip 到 0。这对 agent 场景下"强力抑制 bad action"有利——某个 bad action 的 token 该压就压到底，不受 $1-\epsilon$ 下界保护。

**Intuition**：CISPO 本质上是个 "stop-gradient 版本的 PPO"。它的稳定性来自 clip，它的简洁性来自 stop gradient。在 200k context + 长 horizon 场景下，ratio 的二阶效应（ratio gradient 依赖整个 sequence 的 log prob）很危险，stop gradient 把这个雷拆掉了。

和 GRPO 的对比：GRPO 去掉了 value function，用 group-relative advantage。CISPO 在 GRPO 基础上又加了 stop gradient + 下界 0 的改进。可以说 CISPO = GRPO + stop gradient + 更激进的 clip。

---

## 思路五：Reward 设计，dense 化 + speed 化

长 horizon agent 的 credit assignment 是噩梦。trajectory 几千步，只有最后才给一个 reward，中间步骤全是 0。signal-to-noise ratio 极低，gradient variance 爆炸。

Forge 的复合 reward 有三部分：

**1. Process Reward（过程奖励）**
中间步骤给 dense reward——工具调用对不对、推理质量好不好、子任务完成度。这把稀疏 reward 变成 dense reward，credit assignment 友好很多。

**2. Speed Reward（速度奖励）**
这个很有意思。R1 出来之后大家发现 RL 会鼓励模型生成超长 reasoning（reward 和正确性挂钩，模型学会"想得多一点"来提正确率）。但在真实 agent 场景下，用户对执行时间非常敏感——你让 agent 帮我订个机票，它给我推理 10 分钟，用户体验极差。

Speed reward 让模型主动选择最短执行路径、利用并行工具调用。形式上大概是给完成时间一个负 reward（完成越快 reward 越高）。

**3. Reward-to-Go + Group Baseline**
$$
\hat{A}_{i,t} = \sum_{p=t}^T (r_p^{speed} + r_p^{perf}) - B_i
$$

从 step t 到结束的累计 reward，减去 group 内的 mean reward 作为 baseline。这是降低 variance 的标准操作，不需要 value function（在 agent 场景下 value function 很难学准）。

---

## 思路六：把 Context Management 变成 Agent Action

这个我单独拎出来讲，因为我觉得是 paper 里最有 idea 的一部分。

DeepSearch 这种长程任务有个问题：agent 多轮搜索，每轮都积累 observation，context 越来越长。到后面，关键的早期信息被大量冗余 observation 稀释，模型"注意力分散"，性能下降。

传统解法是 inference-time 做 context management（CM）——压缩历史、截断、重写。但训练时没用 CM，推理时用 CM，导致 train-test distribution shift。模型没见过被压缩的 context，表现就差。

Forge 的思路是 **把 CM 当成 agent 的一个 action**，直接放进 RL 训练循环里。

具体说：状态转换 $S_t \to S_{t+1}$ 不只是"生成一个 token"，还隐式包含"context 被压缩了"这个操作。模型在训练时就学会在被压缩的 context 上仍然表现好——它内化了这种 distribution shift。

更进一步的 emergent behavior：模型在生成时会 **预见**可能的 CM 操作，主动保留 task-relevant 信息，丢弃无关 context。它学会了"context 会被压缩，所以我现在就只记关键的"。

这个思路让我想到 curriculum learning 或者 self-play——你把困难（context 被压缩）直接放进训练，让模型自己适应，而不是 inference 时硬塞。

---

## 思路七：推理侧的一堆优化

Forge 在推理侧也做了不少事，我快速过一下：

**Dynamic MTP**：DeepSeek-V3 的 MTP（Multi-Token Prediction）思路，训练时预测多个 future token，推理时当 draft model 做 speculative decoding 加速。但 RL 训练会让 policy 分布变化，MTP head 不更新的话 draft 接受率会降。所以他们用 Top-K KL loss 在 RL 过程中持续训练 MTP head，同时 detach gradient 不让 MTP head 干扰主 model。

**PD 分离**：Prefill 和 Decode 物理分离到不同 instance。MoE 模型上 prefill 是 compute-bound、decode 是 memory-bound，混在一起会互相干扰。分离后各自优化并行策略，长尾样本不会 block scheduler。

**全局 L3 KV Cache Pool**：Agent 多轮请求间共享 prefix 比例极高，但单 instance 的 KV cache 容量有限，RL batch size 大的时候会发生 eviction 重算。Forge 做了三级 cache（GPU HBM → CPU memory → 全局共享），并用 cost-aware scheduler 路由请求到缓存命中的 instance。

这三个优化叠起来，rollout 阶段的算力占比降到了 60% 左右，给训练腾出空间。

---

## 我的整体 take

这篇 paper 读下来，我感觉 MiniMax 团队是真的在 production 场景里摸爬滚打出来的。每个 optimization 都有明确的 pain point 在驱动，不是学术论文那种"我发明了一个新方法"，而是"我踩了一个坑，填了，再踩，再填"。

几个让我印象深的点：

**1. "解耦"是核心 design philosophy**
Agent 和引擎解耦，rollout 和 train 解耦（通过 Data Pool），prefill 和 decode 解耦。每一层解耦都让系统复杂度下降一个数量级。这个思路和 SGLang、vLLM 这些 inference framework 的演进是一脉相承的——把 monolithic 的设计拆成模块化。

**2. Bounded off-policy 是关键 insight**
完全 on-policy 没吞吐，完全异步会崩。Windowed FIFO 的 bounded off-policy 思想在系统领域很常见（bounded staleness、bounded delay），但用在 RL 调度上很自然。$W = N/2$ 这个 sweet spot 具体怎么选的需要更多实验数据，但方向是对的。

**3. CISPO 的 stop gradient 是个小但关键的 trick**
PPO 的 ratio gradient 二阶效应在长 sequence 场景下很危险，stop gradient 一刀切掉。这个 trick 简单但有效，我觉得会逐渐成为 long-context RL 的标配。

**4. Speed reward 是被低估的设计**
大家都在追求 reasoning 长、agent 强，但真实用户要的是"又快又好"。Forge 把 speed 放进 reward function，让模型学会 trade off 深度和速度。这个方向我觉得以后会有更多探索，比如 user-simulated reward、latency-aware decoding。

**5. CM as action 的思路有想象力**
把 inference-time trick 变成 training-time policy 的一部分，让模型内化 distribution shift。这个 idea 可以推广到很多其他 inference trick 上——quantization-aware training、sliding window attention、early exit 等等，都可以变成 policy 的一部分来训。

---

## 几个我想吐槽的地方

**1. 缺 ablation**
paper 没给 Windowed FIFO 的 W 扫描结果，没给 Prefix Tree Merging 的具体加速比 breakdown，没给 CISPO vs PPO 的 head-to-head 对比。这些数据对理解每个 contribution 的价值很关键。可能是 blog post 篇幅限制，希望后续有完整 paper。

**2. "40x 加速"没说清楚**
评论里有人问 40x 怎么来的，作者说主要来自同一 traj 内多个 completion 的共享前缀。但具体场景（prefix 多长、G 多大、forward only 还是 forward+backward）没说清。我估算 8x~40x 都有可能，取决于场景。

**3. 黑盒 RL 的 credit assignment 还是粗糙**
黑盒 agent 内部 state 不可见，只能做 trajectory-level reward。虽然 process reward 部分缓解了，但本质上还是 group-relative advantage 在撑。如果 agent 内部有 complex branching（multi-agent 协作），这个方案的天花板在哪没讨论。

---

## 最后

这篇 paper 的价值在于它把"大规模 agent RL 怎么做"这个问题的 solution space 给勾勒出来了。Windowed FIFO、Prefix Tree Merging、CISPO、CM-as-action、复合 reward——这些单独看都是具体 engineering trick，但组合在一起就是一个完整的 design pattern。

如果你要 build 自己的 agent RL 系统，Forge 这篇基本是必读的 reference design。期待 MiniMax 后续放出更完整的 paper 和 ablation 数据，也期待 M2.7 的表现。

---

**参考阅读**：
- GRPO (DeepSeek): https://arxiv.org/abs/2402.03300
- PPO: https://arxiv.org/abs/1707.06347
- DeepSeek-V3 MTP: https://arxiv.org/abs/2412.19437
- SGLang RadixAttention: https://arxiv.org/abs/2312.07104
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- veRL 开源 Agent RL 框架: https://github.com/volcengine/verl

---

# Forge: MiniMax 大规模 Agent RL 系统深度解析

这篇 paper 是 MiniMax 团队（岳览等）关于 **Forge** 系统的技术分享，记录了他们训练 **MiniMax M2.5** 模型时在大规模 Agent RL 上踩过的坑和做出的工程/算法创新。我会从 motivation、架构、工程优化、算法四个维度逐一拆解，并 build 出背后的 intuition。

---

## 1. 核心问题：为什么 Agent RL 比传统 RLHF 难一个数量级

传统 RLHF（如 InstructGPT、Llama-2 RLHF）的 setting 是：单轮 chat，input → output，trajectory 长度几百到几千 token，rollout 时间方差小（秒级），reward 相对 dense（preference model 给 scalar）。这个 setting 下 PPO/GRPO 都能 work。

但 **Agent RL** 完全不同：

| 维度 | 传统 RLHF | Agent RL |
|---|---|---|
| Trajectory 长度 | 1k-4k token | 50k-200k token（多轮 + 工具调用） |
| Rollout 时间方差 | 秒级，方差小 | 几秒到几小时，方差极大 |
| Reward 稀疏度 | Dense（preference model 逐 token） | Sparse（只有 task 完成才给 reward） |
| Agent 框架多样性 | 固定 chat template | 数百种 scaffold，黑盒 + 白盒 |
| Off-policy 程度 | 低（同步 batch） | 高（异步 + 长尾） |

Forge 这篇 paper 本质上是在回答一个问题：**如何让一个 RL 系统同时 handle 这五个维度的放大**。

---

## 2. 问题建模：J(θ) = Throughput × Sample Efficiency

paper 一开始就把优化目标形式化为：

$$
\max_\theta J(\theta) = \text{Throughput}(\mathcal{A}) \times \text{Sample Efficiency}(\mathcal{A})
$$

subject to:
$$
\forall \mathcal{A} \in \Omega_{\text{agent}} \quad \text{(Arbitrary Agent)}
$$
$$
\mathbb{E}[\text{Update Variance}] < \delta \quad \text{(Stability)}
$$
$$
\mathbb{E}[\|J^{(T)} - J^*\|] < \epsilon \quad \text{(Convergence)}
$$

**变量解析**：
- $\theta$：policy model 参数
- $\mathcal{A}$：某个具体的 Agent scaffold（脚手架）
- $\Omega_{\text{agent}}$：所有可能的 Agent scaffold 集合
- $J^{(T)}$：第 $T$ 步的训练收益
- $J^*$：最优训练收益
- $\delta, \epsilon$：稳定性和收敛性的容忍上界

**Intuition**：传统 RL paper 通常只优化 sample efficiency（sample 复杂度），但 Forge 强调 **throughput 和 sample efficiency 是乘积关系**，且两者之间存在 trade-off：

- 要提高 throughput，需要异步、batch 大、off-policy 程度高
- 要提高 sample efficiency，需要 on-policy、batch 小、分布稳定

这个 trade-off 是 Forge 后面所有工程优化（Windowed FIFO、Prefix Tree Merging）和算法设计（CISPO、Dense Reward）的出发点。

---

## 3. 系统架构：三层解耦

### 3.1 核心设计哲学：Agent 与 Engine 彻底解耦

传统 RL 系统的耦合方式（TITO, Token-In-Token-Out）：

```
[Agent Logic] ←→ [Tokenizer] ←→ [LLM Engine] ←→ [RL Trainer]
     ↑                                    ↑
     └──── token-level 耦合 ─────────────┘
```

问题：Agent 内部做 context compression、history rewrite、tool call parsing 时，会破坏 token 序列的连续性。RL 系统需要严格追踪每个 token 的 advantage，但 Agent 的操作让 token 边界变得模糊。

Forge 的解耦方式：

```
[Agent (任意脚手架)] 
    ↓ (标准化协议)
[Gateway Server] ←→ [Data Pool] ←→ [Rollout Engine + Train Engine]
```

三个核心模块：

**1. Agent 抽象层（Trajectory Producer）**
- Agent 只负责产生 trajectory，不关心底层训练/推理细节
- Agent 可以是白盒（我们能改代码）或黑盒（如 Opencode Agent 这种闭源工具）

**2. 中间件抽象层**
- **Gateway Server**：标准化通信网关，Agent 通过统一协议请求 LLM generation
- **Data Pool**：分布式数据存储，异步收集 trajectory 和 process signal，作为生成-训练的缓冲区

**3. 训练与推理引擎**
- **Rollout Engine+**：高吞吐量 token 生成
- **Train Engine+**：从 Data Pool fetch 数据，更新 model，并与 rollout engine 同步策略

**Intuition**：这种设计类似微服务架构中的 API gateway 模式。Agent 像 client，Gateway 像 API gateway，Data Pool 像消息队列（Kafka），Train/Rollout Engine 像后端 worker。这种解耦让 RL 系统可以同时训练数百种不同 Agent scaffold，而不需要为每种 scaffold 改代码。

---

## 4. 白盒 Agent RL：Context Management 的例子

### 4.1 问题：注意力稀释 + 训推不一致

在 DeepSearch 这类长程任务中，Agent 会进行多轮搜索，每轮都积累 observation。问题：

1. **注意力稀释**：随着交互轮次增加，中间推理和冗余 observation 积累，关键信息被稀释。模型在 200k context 内对关键 token 失去焦点。

2. **训推不一致**：传统做法是训练时用完整 context，推理时用 context management（如截断、压缩）。这导致推理时分布偏移（distribution shift），模型被迫处理训练时没见过的 context 模式。

### 4.2 解决方案：将 CM 建模为 Agent Action

paper 的关键 insight：**把 context management 视为一种 functional action**，状态转换 $S_t \to S_{t+1}$ 隐式包含 context switching 逻辑。

形式化：
$$
S_{t+1} = f(S_t, a_t^{CM}, a_t^{gen}, \text{env})
$$

其中：
- $a_t^{CM}$：context management action（如压缩、截断、重写）
- $a_t^{gen}$：generation action（LLM 输出）
- $\text{env}$：环境反馈

这样，RL 训练目标自然包含了对 context 适应的优化：
$$
\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t r_t \right]
$$

模型学会 **内化分布偏移**——在生成时就预见可能的 context 管理操作，主动保留 task-relevant 信息，丢弃无关 context。

**Intuition**：这类似于 self-distillation 的思想，但用在了 context management 上。传统做法是 inference-time trick（如 sliding window attention），但 Forge 把它变成 training-time 的 policy 的一部分，让模型学会"怎么在 context 被压缩后仍然表现好"。

---

## 5. 黑盒 Agent RL：跨框架鲁棒性

### 5.1 挑战

真实用户用的 Agent 很多是闭源的（如 Opencode、Truncate BC 等），内部 loop 逻辑完全不可见。传统 RL 系统需要知道每一步的 state、action、reward，但黑盒 Agent 只暴露最终结果。

### 5.2 Forge 的非侵入式集成

**核心做法**：Agent 内部只需要把 LLM 请求打到 Gateway Server，Forge 在 Gateway 层做数据收集和训练。Agent 内部的 memory compression、history rewrite、multi-agent coordination 等，对 RL 系统是透明的。

这意味着：
- 任意 context 操作（记忆压缩、历史重写）都能兼容
- 任意 Agent loop（Deep Think、Multi-Agent）都能训练
- 完全不透明的黑盒系统也能带来稳定提升

**Intuition**：这其实是把 RL 系统从 "state-action-reward" 范式降级到 "trajectory-level reward" 范式。我们放弃精细的 credit assignment，转而用 group-relative advantage（GRPO 思想）和 dense reward（speed + process）来补偿。这个 trade-off 在黑盒场景下是合理的——你不知道内部 state，所以只能 optimize 整体 trajectory 质量。

---

## 6. 工程优化一：Windowed FIFO+ 调度策略

### 6.1 问题：吞吐 vs Off-policy 的 trade-off

异步 RL 的核心调度难题：

| 策略 | 吞吐量 | Off-policy 程度 | 稳定性 |
|---|---|---|---|
| FIFO（严格同步） | 低（被长尾 block） | 低 | 高 |
| Greedy/FFFO（先完成先出） | 高 | 高（分布偏移） | 低，易崩溃 |

FFFO 的问题：长尾样本（几小时的 trajectory）完成后，policy 已经更新了很多步，这些样本的 importance sampling ratio 极大，导致梯度爆炸。

### 6.2 Windowed FIFO 的设计

设定：
- $N = 8192$：最大并发量
- $Q$：生成队列
- $H$：当前头部索引
- $W = 4096$：可见窗口大小

调度规则：

1. **受限可见性**：调度器只能从 $[H, H+W]$ 范围获取已完成 trajectory
2. **局部贪婪（窗口内）**：窗口内可任意提取已完成 trajectory，避免 HoL（Head of Line）阻塞
3. **全局严格阻塞（窗口外）**：即使 $H+W+1$ 完成，也不能提取
4. **约束推进**：只有头部 $H$ 被消费时，窗口才滑动 $H \leftarrow H+1$

**Intuition**：这本质上是 **bounded staleness** 的概念，类似 parameter server 中的 bounded delay rule。$W$ 控制了 off-policy 的上界：

- 最大 off-policy step $\approx W$（窗口内最旧样本相对当前 policy 的更新步数）
- 当 $W \to 0$：退化为 FIFO
- 当 $W \to N$：退化为 Greedy/FFFO

$W = 4096$（$N$ 的一半）是一个 sweet spot：既能让快任务不必等待慢任务，又能限制慢任务的 staleness 不超过 4096 个 policy update。

**与 PPO clip 的配合**：CISPO 的 clip 上界 $1 + \epsilon_{high}^{IS}$ 正好用来 handle 这种 bounded off-policy。如果 importance sampling ratio $r_{i,t} > 1 + \epsilon_{high}^{IS}$，说明样本太 old，gradient 被 clip 掉。

---

## 7. 工程优化二：Prefix Tree Merging+

### 7.1 问题：前缀冗余

Agent 场景下的冗余来源：

1. **Group-level rollout**：同一 query 采样 $G$ 个 completion，共享 prompt prefix
2. **Multi-turn Agent**：多轮请求间，前几轮的 context 完全相同
3. **Tokenizer encode-decode 不一致**：context management 后 retokenize，但 prefix 大部分相同

传统方法把每个 completion 当独立 sequence，重复计算公共 prefix 的 forward/backward。

### 7.2 Prefix Tree Merging 的加速原理

假设：
- Group size $G = 8$
- 共享 prefix 长度 $L_p = 30k$ token（多轮 Agent 场景常见）
- 每个 completion 平均长度 $L_c = 2k$ token

**Naive 方案计算量**（forward only，FLOPs $\propto$ sequence length² for attention）：
$$
\text{FLOPs}_{\text{naive}} = G \times (L_p + L_c)^2 = 8 \times 32k^2 \approx 8.19 \times 10^9
$$

**Tree Merge 方案**：
- Prefix 只算一次：$L_p^2$
- 每个 completion 独立 branch：$G \times L_c^2 + 2 \times L_p \times L_c \times G$（cross-attention 部分）

实际上 attention 的加速主要在 prefix 部分只算一次。简化估算：
$$
\text{FLOPs}_{\text{merge}} \approx L_p^2 + G \times L_c^2 \approx 9 \times 10^8 + 3.2 \times 10^7 \approx 9.3 \times 10^8
$$

**加速比**：
$$
\frac{\text{FLOPs}_{\text{naive}}}{\text{FLOPs}_{\text{merge}}} \approx \frac{8.19 \times 10^9}{9.3 \times 10^8} \approx 8.8\times
$$

但 paper 评论里提到能加速 40x。这需要考虑：
- forward + backward 都加速（backward 也共享 prefix 的梯度计算）
- 实际 FLOPs 还包括 FFN 层（$\propto$ sequence length），这部分加速比为 $G \times (L_p + L_c) / (L_p + G \times L_c) \approx 8 \times 32k / 46k \approx 5.6\times$
- 再叠加 attention 部分的高加速比，整体 40x 是可能的（尤其在 $L_p \gg L_c$ 的极端 case）

### 7.3 实现细节

- **数据结构**：将多个 completion 组织成前缀树（trie），共享节点
- **Attention Mask**：使用 Magi Attention 之类的原语表示 branch 间的依赖关系——同一个 branch 内 token 互相 attend，不同 branch 间不 attend
- **Loss 计算**：训练时 unmerge 为序列格式，每个 completion 的 loss 独立计算（保证数学等价性）

**Intuition**：这本质上是把 **data parallelism 中的 batch dimension 部分转换为 sequence dimension 的 sharing**。类似于 RadixAttention / PagedAttention 在推理侧做的 prefix caching，但 Forge 是在训练侧做，并且用 attention mask 保证数学正确性。

实现上的难点：
- 需要 custom kernel 实现 tree-structured attention
- micro-batch 内合并，跨 micro-batch 需要重算 prefix（评论中确认）

---

## 8. 推理加速

### 8.1 Dynamic MTP（Multi-Token Prediction）

MTP 来自 DeepSeek-V3 的设计：训练时预测多个 future token，推理时用 draft model 加速（speculative decoding）。

Forge 的创新：
1. **Dynamic MTP**：推理时动态决定 draft length
2. **Top-K KL Loss**：在 RL 过程中持续训练 MTP head，保持与 RL policy 对齐
3. **Detached MTP**：MTP head 的梯度不回传主模型（stop gradient）

公式形式（推测，paper 没给具体公式）：
$$
\mathcal{L}_{\text{MTP}} = \text{KL}_{\text{Top-K}}(\pi_{\text{MTP}}(\cdot|x_{<t}) \| \pi_\theta^{\text{stop-grad}}(\cdot|x_{<t}))
$$

其中 Top-K KL 只在 vocab top-K 上计算 KL，避免 full softmax 的计算开销。

**Intuition**：RL 训练会让主 policy 分布不断变化，如果 MTP head 不更新，它的 draft 接受率会下降，加速效果退化。Top-K KL loss 让 MTP head 跟随主 policy 的 top-K 分布，同时 detached 保证主 policy 不被 MTP head 干扰。

### 8.2 Rollout 侧 PD 分离

PD 分离（Prefill-Decode 分离）在 MoE 模型上尤为重要：
- Prefill 是 compute-bound（长 prompt），需要高 FLOPs 利用率
- Decode 是 memory-bound（逐 token），需要低延迟

如果 prefill 和 decode 混在一个 instance 上，会发生 scheduling 干扰——一个长 prefill 会 block 后续 decode。

Forge 的做法：物理分离 prefill 和 decode 实例，各自优化并行策略。这对长尾样本尤其重要——极端长 trajectory 不会 block FIFO scheduler。

### 8.3 全局 L3 KV Cache Pool

传统 KV cache 是 instance-local 的，但 Agent 场景下：
- 多轮请求间共享 prefix 比例极高（前几轮 context 完全相同）
- RL batch size 极大（如 N=8192），局部 KV cache 容量不够
- 发生大量 eviction 导致重算

Forge 的 L3 KV Cache 设计：
- **L1**：GPU HBM（instance-local）
- **L2**：CPU memory（instance-local）
- **L3**：全局共享（跨 instance，可能用 RDMA 或 NVMe）

**Cost-aware scheduling**：权衡排队延迟和缓存传输时间，动态路由请求到缓存命中的 instance。

**Intuition**：这类似于 web caching 的多级架构（browser cache → CDN → origin）。L3 cache 让 prefix 命中率从 instance 级别提升到 global 级别，在 200k context 长度 + 数千并发 Agent 的场景下，命中率提升可能带来数倍 throughput 提升。

---

## 9. CISPO 算法

### 9.1 公式解析

$$
\mathcal{L}_{\text{CISPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} sg(\hat{r}_{i,t}(\theta)) \hat{A}_{i,t} \log \pi_\theta(o_{i,t}|q, o_{i,<t}) \right]
$$

其中：
$$
\hat{r}_{i,t}(\theta) = \text{clip}\left(r_{i,t}(\theta), 0, 1 + \epsilon_{high}^{IS}\right)
$$
$$
\hat{A}_{i,t} = \sum_{p=t}^T (r_p^{\text{speed}} + r_p^{\text{perf}}) - B_i
$$

**变量解析**：
- $q$：query（prompt）
- $a$：answer（注意：这个变量在公式中没出现，可能是 paper 笔误，或者 $a$ 指代 group-level 的 answer 集合 $\{o_i\}$）
- $o_i$：group 内第 $i$ 个 completion
- $G$：group size
- $|o_i|$：第 $i$ 个 completion 的 token 数
- $r_{i,t}(\theta) = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})$：importance sampling ratio
- $\epsilon_{high}^{IS}$：ratio 上界（如 1.2）
- $sg(\cdot)$：stop gradient 操作
- $\hat{A}_{i,t}$：advantage，从 step $t$ 到 $T$ 的 reward-to-go 减去 baseline
- $r_p^{\text{speed}}$：step $p$ 的速度奖励
- $r_p^{\text{perf}}$：step $p$ 的性能奖励
- $B_i$：baseline（通常 = group mean reward）

### 9.2 CISPO vs PPO vs GRPO

| 特性 | PPO | GRPO | CISPO |
|---|---|---|---|
| Ratio clip | $[1-\epsilon, 1+\epsilon]$ | $[1-\epsilon, 1+\epsilon]$ | $[0, 1+\epsilon_{high}^{IS}]$ |
| Stop gradient on ratio | No | No | Yes |
| Advantage | GAE + value function | Group-relative, no value | Group-relative + reward-to-go |
| Clip 下界 | $1-\epsilon$ | $1-\epsilon$ | 0 |

**关键差异**：

1. **下界 clip 到 0 而非 $1-\epsilon$**：PPO 下界 $1-\epsilon$ 会限制负 advantage 样本的更新幅度。CISPO 下界 0 意味着如果某个 token 的 probability ratio 已经接近 0（即新 policy 几乎不生成这个 token），它的梯度完全被 clip。这对 Agent 场景下处理 "bad action that should be suppressed aggressively" 有利。

2. **Stop gradient on clipped ratio**：这是最关键的区别。PPO 中 clip 后的 ratio 仍然参与 backward（clip 区间内梯度正常传递）。CISPO 中 $sg(\hat{r})$ 意味着 **ratio 完全不参与梯度计算**，只有 $\log \pi_\theta$ 和 $\hat{A}$ 参与梯度。

形式化看 PPO 的 gradient：
$$
\nabla_\theta \mathcal{L}_{\text{PPO}} = \hat{r} \hat{A} \nabla_\theta \log \pi_\theta + \hat{A} \log \pi_\theta \nabla_\theta \hat{r}
$$

CISPO 的 gradient：
$$
\nabla_\theta \mathcal{L}_{\text{CISPO}} = sg(\hat{r}) \hat{A} \nabla_\theta \log \pi_\theta
$$

第二项 $\hat{A} \log \pi_\theta \nabla_\theta \hat{r}$ 在 PPO 中会导致 **ratio 自我放大**——当 $\hat{A} > 0$ 时，gradient 会同时增大 ratio 和 $\log \pi_\theta$，可能不稳定。CISPO 移除这个 term，让优化更接近 vanilla policy gradient 的形式，但保留了 ratio clip 的 stability benefit。

**Intuition**：CISPO 本质上是 "stop-gradient 版本的 PPO"。它的 stability 来自 clip，它的 simplicity 来自 stop gradient。在 200k context + 长 horizon Agent 场景下，这个设计避免了 ratio 的二阶效应（ratio 的 gradient 依赖于整个 sequence 的 log prob），让 training 更稳定。

### 9.3 Advantage 设计

$$
\hat{A}_{i,t} = \sum_{p=t}^T (r_p^{\text{speed}} + r_p^{\text{perf}}) - B_i
$$

这是 **reward-to-go** 形式：
- $\sum_{p=t}^T r_p^{\text{speed}}$：从 step $t$ 到结束的总速度奖励
- $\sum_{p=t}^T r_p^{\text{perf}}$：从 step $t$ 到结束的总性能奖励
- $B_i$：group-level baseline（第 $i$ 个 group 的 mean reward）

**Intuition**：reward-to-go 相比 GAE 的优势在于不需要 value function。在 200k context + 数千步 trajectory 场景下，value function 很难学准（信用分配太稀疏）。直接用 reward-to-go + group baseline 是更鲁棒的选择。

---

## 10. Dense & Process Reward 设计

paper 提到三部分复合奖励：

### 10.1 Process Reward（过程奖励）
对 trajectory 中间步骤给 dense reward，缓解稀疏奖励问题。可能的形式：
- 工具调用正确性
- 中间推理质量
- 子任务完成度

### 10.2 任务完成时间奖励（Speed Reward）
$$
r^{\text{speed}} = -\frac{t_{\text{completion}}}{t_{\text{max}}}
$$
或相对形式：
$$
r^{\text{speed}} = \frac{t_{\text{median}} - t_{\text{completion}}}{t_{\text{median}}}
$$

**Intuition**：Agent 场景下，用户对执行时间敏感。纯 task-completion reward 会鼓励模型生成冗长 reasoning（刷榜强但 UX 差）。Speed reward 让模型主动选择最短执行路径、利用并行工具调用。

### 10.3 Reward-to-Go（降低方差）
长 horizon 任务的稀疏 reward 导致梯度方差极大。Reward-to-go 标准化回报：
$$
\hat{R}_t = \sum_{p=t}^T r_p - \bar{R}
$$
其中 $\bar{R}$ 是 baseline（group mean 或 running mean）。

---

## 11. Multi-Domain 混合训练

paper 提到同时混合训练：
- Reasoning（数学、逻辑）
- General QA
- Code Agent
- General Agent

**Intuition**：分阶段训练（先 reasoning 再 agent）会导致 catastrophic forgetting。混合训练类似 multi-task learning，每个 batch 内包含多个 domain，gradient 互相 regularize。这类似于 GPT-4 的 mixed-data training，但在 RL 阶段做。

比例控制是关键——不同 domain 的 sample efficiency 不同，需要动态调整采样比例（可能用类似 priority experience replay 的机制）。

---

## 12. 实验数据与效果

paper 提到：
- 每天百万级样本量吞吐
- 200k context 长度
- 数十万真实 Agent scaffold
- 持续稳定的 reward 上涨
- MiniMax M2.5 性能突破

可惜 paper 没给详细的 ablation 数据（如 Windowed FIFO 的 $W$ 扫描、Prefix Tree Merging 的具体加速比、CISPO vs PPO 的稳定性对比）。这些数据对理解每个 contribution 的贡献度很重要。

---

## 13. 我的思考与 Intuition

### 13.1 这篇 paper 的核心贡献是什么？

我认为 Forge 的核心贡献不是单个 trick（如 Windowed FIFO 或 Prefix Tree Merging），而是 **系统级的 design philosophy**：

1. **解耦**：Agent 逻辑与训推引擎解耦，让 RL 系统可以 scale 到任意 scaffold
2. **Bounded off-policy**：用 Windowed FIFO 控制 off-policy 程度，而非完全 on-policy 或完全异步
3. **Dense reward 补偿**：用 speed + process + reward-to-go 补偿 black-box 场景下的稀疏 reward

### 13.2 与其他系统的对比

- **OpenRLHF / TRL**：传统 RLHF 框架，假设 single-turn chat，无法 handle Agent 场景
- **veRL**：近期开源的 Agent RL 框架，思路类似但规模小
- **DeepSeek R1 的 GRPO**：算法层面的创新（group-relative advantage），CISPO 是 GRPO 的 evolution
- **AgentGym / AgentTrek**：Agent 数据/环境层面，与 Forge 互补

### 13.3 未来方向

paper 暗示 M2.7 即将发布，未来可能探索：
- Tree search（MCTS）与 RL 结合
- 更长的 horizon（million-step trajectory）
- Multi-agent RL（多个 Agent 协作）
- Hierarchical RL（sub-agent 调用）

---

## 参考链接

- PPO 原始 paper: https://arxiv.org/abs/1707.06347
- GRPO (DeepSeek): https://arxiv.org/abs/2402.03300
- DeepSeek-V3 MTP: https://arxiv.org/abs/2412.19437
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- RadixAttention (SGLang): https://arxiv.org/abs/2312.07104
- PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
- veRL (开源 Agent RL 框架): https://github.com/volcengine/verl
- MagiAttention (推测相关): https://arxiv.org/abs/2407.01506
- Speculative Decoding: https://arxiv.org/abs/2211.17192

---

## 总结

Forge 这篇 paper 是目前看到的最完整的大规模 Agent RL 系统设计。它的价值在于：

1. **系统性**：从问题建模到工程优化到算法设计，每一层都有清晰的 trade-off 分析
2. **实用性**：所有 trick 都在实际 200k context + 数十万 scaffold 的 production 场景验证过
3. **可迁移性**：Windowed FIFO、Prefix Tree Merging、CISPO 这些技术可以独立应用到其他 RL 系统

对于想 build intuition 的读者，我建议重点理解三个 trade-off：
1. **Throughput vs Off-policy**（Windowed FIFO 的 $W$）
2. **Sample efficiency vs Stability**（CISPO 的 stop gradient + clip）
3. **Dense reward vs Sparse reward**（Process + Speed + Reward-to-go）

这三个 trade-off 是所有 Agent RL 系统都会遇到的核心问题，Forge 的解法是当前 SOTA 的 reference design。
