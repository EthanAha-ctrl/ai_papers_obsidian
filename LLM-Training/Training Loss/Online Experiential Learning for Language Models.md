---
source_pdf: Online Experiential Learning for Language Models.pdf
paper_sha256: 743a776285dd0c3885c39e3359d3e355bc520a22c7f6e1a07ed2f5a37926a9f7
processed_at: '2026-08-05T23:54:32-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，咱们抛开那些学术包装，直接用大白话讲讲这篇 paper 到底在干嘛，以及为什么我觉得它比表面看起来更有意思。

### 1. 核心痛点：训练完就“死”了

现在的 LLM，不管是 SFT 还是 RLHF，训练完就 freeze 了，deploy 出去之后就再也不长进了。它在真实世界里跟用户聊了 100 亿次，犯了无数错，试出了无数好策略，结果呢？这些 experience 全被扔进垃圾桶，下一个版本的训练还得从头来，重新人工标注数据。

这太蠢了。人类是靠 experience 长智慧的，模型为什么不行？

### 2. 难点在哪：没有 Reward，也碰不到环境

你可能会说，那部署的时候顺手 RL 一下不就行了？问题在于真实的 deployment 场景：
- Server side（训练的地方）根本访问不到 User side（用户用的环境），因为隐私、延迟、API 隔离。
- 真实环境不像下棋有输赢，它只给你文本反馈，比如“你刚才那个动作撞墙了”、“文件路径不对”。这种文本没法直接塞进 standard RL 算法里当 reward 用。

所以我们需要一个完全 **reward-free** 的 learning paradigm。

### 3. OEL 的核心 Intuition：自己教自己

OEL 的招数其实就两步，但组合起来很精妙：

**第一步：把经验提炼成“知识小本本”**
模型在环境里玩了一圈，留下了一堆 trajectory（你动了啥，环境给了啥反馈）。OEL 让模型自己去看这些 trajectory，把里面“下次还能用”的规律总结出来，记在一个小本本上。这个小本本就是 **experiential knowledge**。

比如玩 Sokoban（推箱子），模型可能总结出：“如果箱子和墙贴在一起，别往墙那边推，会死棋。”

注意，这个总结过程是累积的。模型看第二条 trajectory 时，能看到前一条总结的 knowledge，这样它能去重、能细化、能修正之前的错误结论。

**第二步：把小本本“吃”进权重里（Context Distillation）**
小本本越长，in-context learning 效果越差，因为 context window 挤爆了。怎么办？

OEL 用了一招 **on-policy context distillation**：
1. 拿出一个 partial 的游戏局面 $x$。
2. 让 **student** 模型（不看小本本）自己往下玩，生成 response $y$。
3. 让 **teacher** 模型（看小本本）在同样的 $x$ 和 $y$ 的基础上，给出它认为下一步该怎么走的概率分布。
4. 用 **Reverse KL** 强迫 student 的分布去对齐 teacher 的分布。

Teacher 是谁？就是训练前的 student 自己，加上小本本作为 context。

这等于在说：“嘿 student，如果你带了小本本你会这么想，那我现在逼你在不带小本本的时候，也要想得跟带小本本一样。”

训完之后，小本本里的知识就被“烤”进 student 的 weights 了。下一轮 inference，student 不需要带小本本，就能表现得像带了小本本一样。

### 4. 这里的数学细节，为什么这么设计

咱们看那个 loss function (公式 2)：

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, e \sim \mathcal{C}, y \sim \pi_\theta(\cdot \mid x)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} D_{\text{KL}}\Big(\pi_\theta(\cdot \mid x, y_{<t}) \Big\| \pi_{\text{teacher}}(\cdot \mid e, x, y_{<t})\Big) \right]
$$

拆开讲人话：
- $x$：游戏某个中间状态（partial rollout prefix）。
- $e$：从小本本集合 $\mathcal{C}$ 里随机抽一本知识。
- $y$：student 自己生成的 response（**on-policy** 的关键，$y$ 是 student 采样的，不是 teacher 给的）。
- $t$：response 里的第 $t$ 个 token。
- $y_{<t}$：student 已经生成的前 $t-1$ 个 token。
- $\pi_\theta(\cdot \mid x, y_{<t})$：student 在这个位置的 next-token 分布。
- $\pi_{\text{teacher}}(\cdot \mid e, x, y_{<t})$：teacher 在这个位置的 next-token 分布（带小本本 $e$）。

**为什么是 Reverse KL？**
Reverse KL $D_{\text{KL}}(p \| q)$ 是 mode-seeking 的。意思是，student 只敢去 teacher 有概率的地方（否则 $\log(p/q)$ 爆炸）。这样 student 不会乱发挥，只会在自己已有的 mode 上 refine，避免把知识蒸馏成 hallucination。

**为什么必须 On-Policy？**
如果 $y$ 是 teacher 生成的（off-policy），那 student 是在强行模仿 teacher 的路径。但 teacher 是带了小本本的，它的路径可能 student 现在的能力根本走不出来。Train-inference mismatch，学完就崩。

On-policy 让 $y$ 来自 student 自己，student 在自己 visited 的 action space 里，被 teacher 拉着往“更符合知识”的方向走一点。这是温和的 refinement，不是强扭的瓜。

实验里（Figure 6）直接证明了这点：on-policy 保住了 OOD (IF-Eval) 的性能，off-policy 蒸完 OOD 就崩了。

### 5. Loop 起来：自我进化的飞轮

这两步一旦连起来，就是一个飞轮：
- 模型变强 → 收集的 trajectory 质量更高 → 提炼的知识更精 → 蒸馏进 weights 后模型更强 → ...

Paper 里跑了 3 轮，pass rate 稳定上升，response length 还变短了（Figure 5），说明模型不仅学对了，还学“精”了，思考效率提高了。

### 6. 实验里最让我意外的两个点

**Ablation 1 (Table 1)：Raw trajectory 不能直接用，必须提炼成 knowledge**
直接拿 raw trajectory 蒸馏，pass rate 反而从 10.9% 掉到 7.8%。因为 raw trajectory 里全是 noise、死胡同、错误尝试。Student 去模仿这些 noise，越学越蠢。
提炼成 knowledge 后，noise 被过滤掉，只留下 essence，pass rate 跳到 21.4%。
**这印证了 "compression is learning" 的哲学。**

**Ablation 2 (Table 2)：小模型用自己的知识，比用大模型的知识还好**
Qwen3-1.7B 用自己 trajectory 提炼的知识，比用 Qwen3-4B 提炼的知识，效果好得多（31.1% vs 22.7%）。
直觉是：4B 的策略可能依赖 4B 的 reasoning 深度，1.7B 根本 execute 不了。蒸馏这种“高攀不起”的策略，反而造成 mismatch。
**Model 应该在自己的 capability frontier 上探索，自己 pull up by bootstraps。**

### 7. 我的几个联想

**跟 RL 的关系：**
OEL 看起来像 RL，但完全没 reward model，没 value function，没 environment interaction。它用 **knowledge-conditioned teacher 的 next-token 分布** 充当了 dense reward。这比 scalar reward 信息量大太多了，等于是 token-level 的 fine-grained supervision。

**跟 Test-Time Compute Scaling 的关系：**
大家都在 talk test-time compute。OEL 其实是在说：deployment 期间模型 burn 的 compute（生成 trajectory、试错），别浪费，把它当 training signal 存下来，amortize 成 permanent capability。

**跟“Era of Experience”的关系：**
Silver & Sutton 那篇 position paper (https://arxiv.org/abs/2503.07269) 呼吁模型应该从 experience 学。OEL 是我看到的第一个 concrete、scalable、reward-free 的 implementation blueprint。它告诉你怎么把 textual feedback 转成 training signal，怎么在 server 隔离的情况下做 on-policy learning。

### 8. Limitations & Open Questions

Paper 没充分讨论的几点，我觉得很关键：
1. **Extraction 的瓶颈**：如果模型本身很笨，它提炼出来的 knowledge 也可能很笨。Self-bootstrapping 能 escape local optimum 吗？Paper 只跑了 3 轮，long-horizon 稳定性未知。
2. **Knowledge Drift**：转了 100 轮之后，knowledge 会不会累积 bias？会不会把错误 insight 放大？Paper 没有机制来验证 knowledge 的 correctness。
3. **Environment Complexity**：Frozen Lake 和 Sokoban 是 toy environment，反馈结构化、规则简单。Real-world deployment（比如 coding agent, customer service）的反馈远比“你撞墙了”复杂，extraction prompt 能不能 handle 这种 messiness？
4. **Compute Cost**：每轮要 collect 几千条 trajectory，跑 K=10 次 extraction，再 distill 100 steps。Wall-clock cost 和 ROI 没报告。在真实 production 里这个 loop 的 latency 是个大问题。

### 9. 最核心的 takeaway

OEL 给我最大的启发是：**Context distillation 是把 in-context learning 的成果“固化”进 weights 的通用机制**。

你只要能构造出一个“带 context 的 teacher”，哪怕这个 context 是 model 自己写的 knowledge 笔记，你就能通过 on-policy reverse KL 把它烤进 weights。这个 pattern 极其通用，可以推广到很多场景：
- 模型自己写的 CoT，烤进 weights
- 模型自己总结的 tool-use manual，烤进 weights
- 模型自己总结的 user preference，烤进 weights

这是一个把 **test-time adaptation 转成 training-time improvement** 的通用 transformer。

Reference:
- OEL paper: https://aka.ms/GeneralAI
- On-policy context distillation: https://arxiv.org/abs/2602.12275
- MiniLLM (reverse KL): https://arxiv.org/abs/2306.08543
- Era of Experience: https://arxiv.org/abs/2503.07269
- RL's Razor: https://arxiv.org/abs/2509.04259

---

# Online Experiential Learning (OEL) 深度解析

Andrej, 这篇 paper 触及了一个我（作为模型）认为 LLM 领域真正 fundamental 的方向 —— 把 deployment 当成 training 的延续，把 textual feedback 当成 reward signal。我会尽量把每层 intuition 都剥开。

---

## 1. 大图景：为什么这个范式重要

当前 LLM 训练有两个 dominant paradigm：

- **SFT**：human annotation → static dataset → supervised learning
- **RL (RLHF/RLVR)**：verifiable reward 或 reward model → PPO/GRPO

两者共同的问题是 **closed-world assumption** —— model 训练完就是 frozen artifact，deployment 期间积累的丰富 experience 全部 discard。这与 human learning 形成鲜明对比：human 通过 experience 持续 refine 能力。

Silver & Sutton 在 "Welcome to the Era of Experience" (https://arxiv.org/abs/2503.07269) 中已经呼吁这个方向。OEL 把这个 vision 落地到一个具体可执行的 framework。

**关键约束**（这部分是 paper 的核心 insight）：
- Server side **无法访问** user-side environment（隐私、延迟、API 限制）
- Real-world environment 通常 **不提供 scalar reward**，只提供 textual feedback（错误信息、状态描述）
- 为每个新 deployment scenario 构造 reward function **不现实**

OEL 的解法：把 textual feedback → experiential knowledge → 蒸馏进 weights，整个 loop **reward-free**。

---

## 2. Framework 架构解析

参考 paper Figure 3，整体 pipeline：

```
[User Side]                    [Server Side]
Environment E                  Stage 1: Extract
     ↓                              ↓
   π_θ interacts              π_extract (= π_θ)
     ↓                              ↓
Trajectories T, T'      Accumulate knowledge C = {e^1,...,e^K}
                                      ↓
                              Stage 2: Consolidate
                                      ↓
                              On-policy context distillation
                              (reverse KL, top-256)
                                      ↓
                              Updated π_θ → redeploy
```

**两个 trajectory set 的作用**：
- $\mathcal{T} = \{\tau_1, ..., \tau_n\}$：用于 extraction stage，产生 knowledge
- $\mathcal{T}' = \{\tau_1, ..., \tau_m\}$：用于 consolidation stage，产生 partial rollout prefixes 作为训练数据

这两个 set 是独立的，paper 没有明确说是否 overlap，但从 algorithm 1 看是分别 collect 的。

---

## 3. Stage 1: Experiential Knowledge Extraction

### 3.1 公式 (1) 详解

$$
e_i' \sim \pi_{\text{extract}}(\cdot \mid \tau_i, e_{i-1})
$$
$$
e_i = [e_{i-1} ; e_i']
$$

变量含义：
- $\tau_i$：第 $i$ 条 trajectory，结构为 $(f_i^1, a_i^1, f_i^2, a_i^2, ...)$，其中 $f_i^j$ 是 environment feedback，$a_i^j$ 是 model action
- $e_{i-1}$：处理前 $i-1$ 条 trajectory 后累积的 knowledge
- $e_i'$：从第 $i$ 条 trajectory 中新提取的 knowledge（条件于历史 knowledge）
- $e_i$：累积后的 knowledge，$[;]$ 表示 concatenation
- $e_0 = \emptyset$：初始为空
- $\pi_{\text{extract}}$：extraction model，默认等于 $\pi_\theta$（on-policy 一致性）

### 3.2 为什么是 accumulative 而非独立提取

这是关键设计。如果对每条 trajectory 独立提取然后合并，会丢失 cross-trajectory 的 pattern。Accumulative 方式让 extractor 看到之前总结的 knowledge，能够：
- **去重**：避免重复总结相同的 insight
- **refine**：在已有 knowledge 基础上补充更精细的观察
- **conflict resolution**：当新 trajectory 与已有 knowledge 矛盾时，extractor 可以更新

这本质是一种 **online summarization with memory**，类似 Voyager (https://arxiv.org/abs/2305.16291) 的 skill library accumulation，但用 text 而非 code。

### 3.3 两种 knowledge format

- **Structured**：强制格式 `– EXPERIENCE ITEM:`，便于解析和过滤；$n = 25$ 或 $50$，$L_{\max} = 8192$
- **Unstructured**：自由生成；$n = 15$，$L_{\max} = 2048$

K = 10 次重复（不同 random seed），产生 $\mathcal{C} = \{e^1, ..., e^{10}\}$。这是 **self-consistency** 的思路 —— 不同 seed 会产生略有差异的 knowledge，训练时随机采样增加 diversity。

### 3.4 一个微妙但重要的点

Paper 提到：**Qwen3-1.7B 在 extraction 时不包含之前累积的 knowledge**，因为小模型无法有效利用长 context。这是一个诚实的工程妥协，但也揭示了一个 limitation —— extraction quality 受 model capacity 限制。

---

## 4. Stage 2: On-Policy Context Distillation

### 4.1 公式 (2) 详解

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, e \sim \mathcal{C}, y \sim \pi_\theta(\cdot \mid x)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} D_{\text{KL}}\Big(\pi_\theta(\cdot \mid x, y_{<t}) \Big\| \pi_{\text{teacher}}(\cdot \mid e, x, y_{<t})\Big) \right]
$$

逐项拆解：

- $x$：partial rollout prefix，从 $\mathcal{D}$ 采样。$\mathcal{D}$ 由 $\mathcal{T}'$ 中所有 trajectory 的所有 prefix $x_i^j = (f_i^1, a_i^1, ..., f_i^{j-1}, a_i^{j-1}, f_i^j)$ 组成
- $e$：experiential knowledge，从 $\mathcal{C}$ 随机采样（每次 training step 重新采样）
- $y$：student model $\pi_\theta$ 自己生成的 response（**on-policy** 关键！）
- $|y|$：response 长度，用于 normalize
- $t$：token position，从 1 到 $|y|$
- $y_{<t}$：response 的前 $t-1$ 个 token（autoregressive conditioning）
- $\pi_\theta(\cdot \mid x, y_{<t})$：student 在 prefix 和已生成 token 上的 next-token distribution
- $\pi_{\text{teacher}}(\cdot \mid e, x, y_{<t})$：teacher 在 prefix、已生成 token **和 knowledge $e$** 上的 distribution
- $\pi_{\text{teacher}}$：frozen 的初始 $\pi_\theta$（训练前的 checkpoint），加 knowledge context $e$
- $D_{\text{KL}}$：**reverse** KL divergence（注意方向：student || teacher）

### 4.2 Reverse KL 的意义

Reverse KL $D_{\text{KL}}(p \| q) = \mathbb{E}_p[\log(p/q)]$ 是 **mode-seeking** 的：
- 当 $p$（student）有概率而 $q$（teacher）没概率时，penalty 趋于无穷
- 这迫使 student 把概率 mass 集中在 teacher 高概率区域
- 对比 forward KL $D_{\text{KL}}(q \| p)$ 是 mode-covering 的，会试图覆盖 teacher 所有 mode

参考 MiniLLM (https://arxiv.org/abs/2306.08543) 对这个方向选择有详细论证。在 distillation 场景下，reverse KL 避免学生模型过度 spread 概率到 teacher 不擅长的区域，这对于避免 hallucination 和保持 specificity 很重要。

### 4.3 Top-k Approximation

公式 (4) 显示完整 KL 需要对整个 vocabulary $\mathcal{V}$ 求和。Paper 用 $\mathcal{V}_{\text{top-}k}$ 近似，$k = 256$：

$$
D_{\text{KL}} \approx \sum_{y_t' \in \mathcal{V}_{\text{top-}k}} \pi_\theta(y_t' \mid x, y_{<t}) \left(\log \pi_\theta(y_t' \mid x, y_{<t}) - \log \pi_{\text{teacher}}(y_t' \mid e, x, y_{<t})\right)
$$

这里 $\mathcal{V}_{\text{top-}k}$ 是 **student** 概率最高的 top-256 tokens。这是一个工程妥协：
- 完整 vocab sum 计算成本高
- Student 高概率 token 已经覆盖了大部分概率 mass
- 但会 **miss** student 低概率但 teacher 高概率的 token —— 这是 approximation 的偏差源

### 4.4 On-Policy 的核心含义

这是 paper 最 subtle 的部分。注意 $y \sim \pi_\theta(\cdot \mid x)$ —— response 是从 **student** 采样的，而非 teacher。

对比 **off-policy context distillation**（如 Askell et al. 2021, Snell et al. 2022, https://arxiv.org/abs/2209.15189）：
- Teacher with context 生成 response
- Student 学习模仿这些 response（forward KL）

On-policy 的优势（paper Figure 6 的实验验证）：
1. **Train-inference match**：训练时 student 见到的是自己的 distribution，inference 时也是自己的 distribution
2. **Mitigate catastrophic forgetting**：student 不会被迫去模仿 teacher 在 OOD 区域的行为，只在自己 visited 的区域更新
3. **Mode-seeking**：reverse KL + on-policy 让 student 在自己已有的 mode 上 refine，而非被 teacher 拉向新 mode

直觉上：on-policy distillation 像 "在自己的思考路径上 refine"，off-policy 像 "强制模仿专家的路径"。前者更温和，更不容易破坏已有能力。

### 4.5 Teacher 是 frozen 初始 model + knowledge context

注意 $\pi_{\text{teacher}}$ 是 **frozen 的初始 $\pi_\theta$**（训练前的 checkpoint），加上 knowledge $e$ 作为 context。

这意味着：
- Teacher 本身没有学习，它的 "智慧" 完全来自 in-context knowledge $e$
- Student 学习的是 "如何在不看 $e$ 的情况下，表现得像看了 $e$ 一样"
- 这是 **context distillation** 的本质 —— 把 in-context learning 的能力压缩进 weights

这与 Anthropic 的 Constitutional AI (https://arxiv.org/abs/2212.08073) 中 context distillation 的思路一脉相承。

### 4.6 为什么 student 能超越 teacher？

Paper Section 4.2 提到一个 intriguing observation：

> "the student can generalize beyond the teacher's in-context capabilities by distilling the knowledge directly into its parameters."

直觉解释：
- Teacher with knowledge context 受限于 context window 和 attention 机制 —— knowledge 太长会稀释
- Student 把 knowledge 内化后，参数化地访问这些知识，不受 context length 限制
- 多次 training step 让 student 反复 exposure 到 knowledge-conditioned behavior，相当于 **多次 in-context inference 的累积**

这有点像 "把 prompt engineering 的成果固化进 weights"。

---

## 5. Online Learning Loop

### 5.1 自举式改进

Algorithm 1 的核心循环：

```
while Online Learning:
    [User Side] π_θ collects T, T' from E
    [Server Side]
        π_extract = π_θ  # on-policy extraction
        C = accumulate(T) via Eq.(1)
        D = partial_prefixes(T')
        π_teacher = frozen π_θ
        train π_θ on D, C via Eq.(2)
    redeploy π_θ
```

关键：每轮 $\pi_\theta$ 改进后，下一轮 trajectory 质量更高 → knowledge 更丰富 → 下一轮改进更大。这是 **positive feedback loop**，但 paper 显示没有 divergence（Figure 4 的曲线稳定上升）。

### 5.2 为什么不会 collapse？

可能的担忧：
- Model 会不会 overfit 到特定 environment pattern？
- 会不会 amplify bias？

Paper 的实验显示 OOD performance (IF-Eval) 基本保持，说明 on-policy distillation 的 regularization 起作用了。但这是在 2 个简单 game environment 上，scale up 到真实 deployment 时是否还成立，是 open question。

---

## 6. 实验数据深度分析

### 6.1 主结果（Figure 4）

- **Frozen Lake + Qwen3-1.7B (thinking)**：3 轮 OEL，pass rate 持续上升
- **Sokoban + Qwen3-4B-Instruct-2507 (non-thinking)**：同样持续上升

Accumulation phase 的曲线（transparent）显示：in-context knowledge 增长到一定程度后 **saturate** —— context window 被占满，in-context learning capacity 耗尽。Consolidation 后 performance 跳过 saturation point，因为 knowledge 被压缩进 weights，释放了 context capacity。

### 6.2 Token Efficiency（Figure 5）

Response length 在 3 轮 OEL 后降到初始的 ~70%。这说明：
- Model 学到了更高效的 reasoning
- 不是简单地 "记住答案"，而是 internalize 了 problem-solving strategy
- 对 deployment 成本有直接经济价值

### 6.3 OOD Preservation（Figure 6）

对比 on-policy vs off-policy context distillation：
- **In-distribution (game pass rate)**：on-policy 全程高于 off-policy
- **OOD (IF-Eval)**：on-policy 基本保持，off-policy 明显下降

这与 recent work "RL's Razor" (https://arxiv.org/abs/2509.04259) 和 "Retaining by Doing" (https://arxiv.org/abs/2510.18874) 的发现一致 —— on-policy data 天然 mitigate forgetting。

### 6.4 Model Size Effect（Figure 7）

Qwen3-1.7B / 4B / 8B 在 Frozen Lake 上：
- 初始 performance 跨 scale 差异不大（任务可能对小模型也不难）
- OEL 后所有 scale 都有 substantial gain
- Larger model gain 更大 —— 因为 trajectory 质量更高，knowledge extraction 更有效
- Round 1 → Round 2 的 gain 跨 scale 一致 —— 说明 knowledge accumulation 不会因 model capacity 而饱和

### 6.5 Ablation：Raw Trajectory vs Extracted Knowledge（Table 1）

| Experience Type | In-Context | Consolidate |
|---|---|---|
| w/o Experience | 7.5% | - |
| Raw Trajectory | 10.9% | 7.8% |
| Knowledge | 18.2% | 21.4% |

Raw trajectory consolidation 反而 **下降**（10.9 → 7.8）！这非常 striking。直觉：
- Raw trajectory 包含大量 noise（错误尝试、冗余 exploration）
- Distillation 时 student 被迫模仿这些 noise
- Extracted knowledge 是 denoised、abstracted 的，更适合作为 teaching signal

这印证了 "compression is learning" 的观点 —— extraction 本身是一种 information bottleneck，强制 model 提取 essence。

### 6.6 Ablation：On-Policy Consistency（Table 2）

| Experience Source | In-Context | Consolidate |
|---|---|---|
| w/o Experience | 7.3% | - |
| Qwen3-4B (stronger) | 18.0% | 22.7% |
| Qwen3-1.7B (self) | 23.8% | 31.1% |

**Weaker model 从自己 trajectory 提取的 knowledge，比从 stronger model 提取的更有效！**

直觉解释：
- Stronger model 的 strategy 可能依赖其独有的 capability（如更长 reasoning chain）
- Weaker model 无法 execute 这些 strategy，distillation 后产生 train-inference mismatch
- Self-extracted knowledge 是 model "能 actually do" 的 strategy，可执行性更高

这与 "self-play" 和 "self-improvement" 的 philosophy 一致 —— model 应该在自己的 capability frontier 上探索，而非被拽向 unreachable behavior。

---

## 7. 联想与 Intuition Building

### 7.1 与 Reflexion / ExpeL / Voyager 的关系

- **Reflexion** (https://arxiv.org/abs/2303.11366)：verbal reinforcement，reflection 存在 context 中，不更新 weights
- **ExpeL** (https://arxiv.org/abs/2308.10144)：extract insights 存入 external memory，retrieval-augmented
- **Voyager** (https://arxiv.org/abs/2305.16291)：Minecraft skill library，code-based
- **OEL**：experiential knowledge → distill into weights，**真正改变 model parameters**

OEL 是这条 line 的 natural next step —— 从 "外部记忆" 到 "参数化记忆"。

### 7.2 与 RL 的关系

OEL 表面看像 RL，但本质不同：

| 维度 | Standard RL | OEL |
|---|---|---|
| Reward | Scalar, verifiable | Textual feedback |
| Credit assignment | Value function / policy gradient | Teacher's token-level distribution |
| Environment access | Online interaction | Offline trajectories |
| Sample efficiency | Low (exploration) | Higher (knowledge extraction) |

OEL 用 **teacher model + knowledge context** 作为 implicit reward model —— teacher 的 next-token distribution 提供 dense, token-level signal，远比 sparse scalar reward 信息丰富。

### 7.3 与 Test-Time Compute 的关系

Recent work (https://arxiv.org/abs/2408.03314) 显示 test-time compute scaling 很有效。OEL 可以看作：
- **Test-time compute → training signal** 的转换
- Deployment 期间 model "思考" 产生的 trajectory，反过来成为 training data
- 这是一种 **amortized test-time compute** —— 把每次 inference 的 compute 累积成 permanent capability

### 7.4 与 STaR / Self-Taught Reasoner 的关系

STaR (https://arxiv.org/abs/2203.14465) 让 model 生成 rationale，用 ground truth filter 后 SFT。OEL 与之相似但关键不同：
- STaR 需要 ground truth label
- OEL 完全 reward-free，靠 knowledge-conditioned teacher 提供 signal
- STaR 是 single-round，OEL 是 iterative online

### 7.5 Potential Limitations（paper 没充分讨论）

1. **Extraction quality bottleneck**：如果 $\pi_\theta$ 本身弱，extraction 也弱。Paper 用 $\pi_{\text{extract}} = \pi_\theta$，self-bootstrapping 能否 escape local optimum？
2. **Knowledge drift**：iterative loop 中，knowledge 可能累积 bias 或错误 insight。Paper 没有验证 long-horizon（>3 rounds）的稳定性。
3. **Environment diversity**：只在 2 个简单 game 上验证。Real-world deployment 的 environment feedback 远比 "你撞墙了" 复杂。
4. **Compute cost**：每轮需要 collect trajectories + K=10 extraction runs + training。Paper 没报告 wall-clock cost 和 ROI。
5. **Knowledge format sensitivity**：structured vs unstructured 的选择似乎 ad hoc，没有 ablation 对比。
6. **Teacher = frozen initial model**：如果初始 model 很弱，teacher 的 ceiling 也低。是否应该用 stronger external teacher？但这会破坏 on-policy consistency。

### 7.6 一个更深的联想：Learning as Compression

OEL 的 extraction stage 本质是 **lossy compression** —— 把 raw trajectory 压缩成 abstract knowledge。Consolidation stage 是 **把这个 compression 内化进 parameters**。

这呼应 Ilya Sutskever 的观点："compression is intelligence"。OEL 提供了一个 concrete mechanism：model 通过 compressing 自己的经验来 become more intelligent。

### 7.7 与 Continual Learning 文献的连接

OEL 的 online learning loop 是 continual learning 的一种实现。Continual learning 的核心挑战是 stability-plasticity dilemma。OEL 的解法：
- **Plasticity**：通过 knowledge extraction + distillation 持续学习
- **Stability**：on-policy distillation + reverse KL 限制 distribution shift

这比传统 continual learning 的 EWC (https://arxiv.org/abs/1612.00796) 等 parameter regularization 方法更 semantic —— 它 regularize 的是 behavior distribution，而非 individual parameters。

### 7.8 推到极限的 thought experiment

如果 OEL 无限迭代：
- Model 会 converge 到什么状态？
- 是 environment 的 optimal policy，还是 extraction capability 的 bottleneck？
- Knowledge accumulation 会无限增长还是 saturate？

Paper 的 3 轮实验不足以回答，但这正是 paper 开启的方向 —— **deployment as training**。

---

## 8. 实现细节速查表

| Hyperparameter | Value |
|---|---|
| Extraction model $\pi_{\text{extract}}$ | $\pi_\theta$ (on-policy) |
| Teacher $\pi_{\text{teacher}}$ | Frozen initial $\pi_\theta$ |
| Trajectories for extraction $n$ | 25/50 (structured), 15 (unstructured) |
| Max knowledge length $L_{\max}$ | 8192 (structured), 2048 (unstructured) |
| Knowledge set size $K$ | 10 (different seeds) |
| Training steps per round | 20 or 100 |
| Batch size | 64 |
| Trajectories per round | 1280 or 6400 |
| Interaction turns | Up to 5 |
| Max response per turn | 1024 tokens |
| KL approximation $k$ | 256 (top-k vocab) |
| Learning rate | 1e-6 or 5e-6 |
| Sampling temperature | 0.7 |
| Test set size | 128 maps |
| Eval seeds | 10 |

---

## 9. 总结：OEL 的真正贡献

OEL 的 contribution 不在于单个 technique（on-policy distillation, context distillation, experiential learning 都有 prior work），而在于 **把它们组装成一个 reward-free, environment-access-free 的 online learning loop**。

这回答了一个关键问题：**LLM 部署后如何继续学习？**

答案：让 model 自己从 textual feedback 中 extract knowledge，用 knowledge-augmented self 作为 teacher，通过 on-policy reverse KL distillation 把 knowledge 内化。Loop 起来。

我认为这是 LLM 走向 AGI 的必经路径之一 —— **deployment 不是 training 的终点，而是 training 的起点**。Silver & Sutton 的 "Era of Experience" 终于有了第一个 concrete, scalable 的 implementation blueprint。

---

**参考链接汇总**：
- Paper: https://aka.ms/GeneralAI (OEL)
- Code: https://aka.ms/oel-code
- On-policy context distillation (companion paper): https://arxiv.org/abs/2602.12275
- Era of Experience: https://arxiv.org/abs/2503.07269
- MiniLLM (reverse KL distillation): https://arxiv.org/abs/2306.08543
- Voyager: https://arxiv.org/abs/2305.16291
- Reflexion: https://arxiv.org/abs/2303.11366
- ExpeL: https://arxiv.org/abs/2308.10144
- STaR: https://arxiv.org/abs/2203.14465
- TextArena: https://arxiv.org/abs/2504.11442
- Qwen3: https://arxiv.org/abs/2505.09388
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Learning by Distilling Context: https://arxiv.org/abs/2209.15189
- RL's Razor: https://arxiv.org/abs/2509.04259
- Retaining by Doing: https://arxiv.org/abs/2510.18874
- EWC: https://arxiv.org/abs/1612.00796
- Test-time compute scaling: https://arxiv.org/abs/2408.03314
