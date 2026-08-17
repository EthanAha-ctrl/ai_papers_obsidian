---
source_pdf: RubricEM Meta-RL with Rubric-guided Policy.pdf
paper_sha256: 0eef6034a96d6d8c7d3f1cb1535d30de4b6988b68ab8e5c75b071063244b6ab2
processed_at: '2026-08-12T02:27:53-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RubricEM

## 这篇 paper 到底在干嘛

想象你在训练一个 AI 研究助手，给它一个问题比如 "对比一下 Transformer 和 Mamba 在长序列任务上的优劣"，它要自己上网搜、读论文、整合证据、写一篇有引用的长报告。你想用 RL 让它越练越好。

问题来了：**你怎么给它打分？**

数学题可以对答案，代码题可以跑测试。但长篇研究报告没有标准答案——你说它写得好不好，全靠主观判断。这叫 **beyond verifiable rewards**，reward 不可验证。

传统做法是找个 LLM 当 judge，给最终报告打个分（比如 0.7/1.0），然后把这个分数广播到整条 trajectory 的每个 token 上做 RL update。但这有三个大麻烦：

1. **分数太粗**：一个 0.7 分推到几千个 token 上，plan 阶段写得好不好、search 查得准不准、review 做得对不对，全混在一起，policy 不知道哪里该改进。

2. **经验不复用**：这次查 "Transformer vs Mamba" 学到的教训（比如"应该先查 benchmark 数据再查理论分析"），下次查 "RNN vs Transformer" 用不上，因为参数 update 把知识压进了 weight，没有显式的 textual memory。

3. **长 horizon 探索低效**：一条 trajectory 几千个 token、十几次 tool call，flat autoregressive generation 不知道自己现在该 plan 还是该 search 还是该 review，容易在错误的 mode 上 compounding error。

RubricEM 的核心 insight 是一句话：**rubric 不只是用来打分的，它是整个 RL 循环的 shared interface**——同一个 rubric 同时 condition agent 的决策、judge 的评分、memory 的蒸馏。

paper: [RubricEM on arXiv](https://arxiv.org/abs/2511.19399)

---

## 三个组件，各治一个病

### 病1: trajectory 太 flat，policy 迷失 → 药: Structured Scaffold

**Intuition**：你让一个 agent 自由发挥写长报告，它一开始在干嘛？在 plan。中间在干嘛？在 search。最后在干嘛？在 write。但你如果不告诉它"你现在在哪个阶段"，它只能从上下文猜——同样的上下文（"我刚收到一段搜索结果"），在 search 阶段该决定下一个 query，在 review 阶段该判断证据够不够。这叫 **state aliasing**：不同的决策点看起来一样，但需要不同的 action。

Theorem 1 用 Jensen 不等式严格证明了：只要存在两个 stage 的最优 action 冲突，显式告诉 policy 当前 stage 就严格比让它猜要好。

$$
V_{\text{stage}} = \mathbb{E}\Big[\max_a \mathbb{E}[U(H,a) | C, Z]\Big] \ge \max_a \mathbb{E}\Big[\mathbb{E}[U(H,a) | C, Z]\Big] = V_{\text{flat}}
$$

- $H$：当前决策点的 history
- $C = \phi(H)$：压缩后的 context
- $Z \in [K]$：stage label（Plan/Research/Review/Answer）
- $U(H,a)$：在 $H$ 做 action $a$ 后继续 rollout 的 expected value
- $V_{\text{stage}}$：知道 stage 时的最优 value
- $V_{\text{flat}}$：不知道 stage、只能看 $C$ 时的最优 value

不等式来自 max 是 convex 函数 + Jensen：$\mathbb{E}[\max_a f(a,Z)] \ge \max_a \mathbb{E}[f(a,Z)]$。直觉就是"知道更多信息（stage）再做选择，不会比不知道更差"。

**具体做法**：用 XML tag 强制切成四段：

```
Plan: <structured_plan> 内含 <deep_analysis>, <rubric>, <research_plan>
  ↓
Research: <call_tool> → <state_evaluation> → 可能修订 Plan → 再 <call_tool>...
  ↓
Review: <review> 内含 <rubric_review>, <writing_plan>
  ↓
Answer: <answer> with <cite id="S_123">...</cite>
```

关键约束：第一轮必须 search（不能直接 answer），每轮恰好一个 action（要么 tool call 要么 answer），rubric 在 Plan 阶段生成后贯穿全程。

rubric 包含三类内容：
- **knowledge checklist**：要查什么具体事实
- **analytical criteria**：最终回答要达成什么 intellectual connection
- **negative constraints**：要避免什么（比如"不要把博客当学术共识"）

这个 scaffold 通过 SFT distillation（teacher 是 Gemini-3.1-Pro）注入 Qwen3-8B，做 rejection sampling 过滤违规 trajectory。

参考 [Constitutional AI](https://arxiv.org/abs/2212.08073) 用 principles 做 AI feedback，[Panadero 2017](https://www.frontiersin.org/articles/10.3389/fpsyg.2017.00422/full) 的 self-regulated learning rubric 理论。

### 病2: credit 太稀疏 → 药: Stage-Structured GRPO (SS-GRPO)

**Intuition**：传统 GRPO 把 final answer score 广播到所有 token。plan 写得好但 answer 烂了，plan 的 token 拿负 advantage；plan 烂了但 answer 碰巧好，plan 的 token 拿正 advantage。signal 被 dilute。

SS-GRPO 的 idea：既然有四个 stage，就给每个 stage 单独打分，然后让每个 stage 的 token 拿"自己的分数 + 下游 stage 的部分 credit"。

**公式**：stage $k$ 的 return 是

$$
G_{i,k}^{\Lambda} = \sum_{j=k}^{K} \lambda_{k,j} R_{i,j}
$$

- $i$：rollout index（同一 query 采样 $n=8$ 个 rollout）
- $k$：stage index（1=Plan, 2=Research, 3=Review, 4=Answer）
- $R_{i,j}$：rollout $i$ 在 stage $j$ 的 judge score $\in [0,1]$
- $\lambda_{k,j}$：stage weight matrix 的元素，$\lambda_{k,j}=0$ if $j<k$（因果：前面的 stage 不接收后面 stage 的 reward），$\lambda_{k,k}=1$（每个 stage 至少拿自己的分）

实验用的 $\Lambda$：

$$
\Lambda = \begin{pmatrix} 1.0 & 0.4 & 0.6 & 0.8 \\ 0 & 1.0 & 0.4 & 0.8 \\ 0 & 0 & 1.0 & 0.8 \\ 0 & 0 & 0 & 1.0 \end{pmatrix}
$$

看第一行：Plan 的 return = $1.0 \times R_{\text{Plan}} + 0.4 \times R_{\text{Research}} + 0.6 \times R_{\text{Review}} + 0.8 \times R_{\text{Answer}}$。Plan 好的话，Research/Review/Answer 大概率也好，所以 Plan 的 token 也 share 一部分下游 credit。但 Plan 自己的 score 权重最大（1.0），因为 Plan 本身的质量最直接反映在 $R_{\text{Plan}}$ 上。

**Per-stage normalization**（GRPO 的精髓保留）：

$$
A_{i,k} = \frac{G_{i,k}^{\Lambda} - \frac{1}{n}\sum_{i'} G_{i',k}^{\Lambda}}{\text{Std}_{i'}[G_{i',k}^{\Lambda}] + \epsilon}
$$

- $A_{i,k}$：rollout $i$ 在 stage $k$ 的 advantage
- 分子：这个 rollout 的 stage return 减去同组 rollout 的平均 stage return
- 分母：同组 rollout stage return 的标准差

关键是 normalize 的是 $G_{i,k}^{\Lambda}$（per-stage），不是 terminal score。这让每个 stage 有自己的 baseline——Plan 阶段的 token 和 Plan 阶段的 token 比，Answer 阶段的 token 和 Answer 阶段的 token 比，apple to apple。

**Loss**：

$$
\mathcal{L}_{\text{SS-GRPO}} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K \sum_{t \in \mathcal{B}_{i,k}} \min(\rho_{i,t} A_{i,k}, \text{clip}(\rho_{i,t}, 1-\eta, 1+\eta) A_{i,k}) + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

- $\mathcal{B}_{i,k}$：rollout $i$ 的 stage $k$ 的 token 集合
- $\rho_{i,t} = \pi_\theta(a_{i,t}|h_{i,t}) / \pi_{\theta_{\text{old}}}(a_{i,t}|h_{i,t})$：PPO importance ratio
- $\eta$：clip ratio
- $\beta = 0.001$：KL 系数
- 同一 stage block 内所有 token 共享 $A_{i,k}$

**Stagewise evolving rubric judge**：judge 自己有个 rubric buffer，分 stage 维护。每步对比同 query 的 8 个 rollout，找 discriminative criteria（"为什么有的好有的差"），加入 buffer，buffer cap = (3,2,2,3)。打分后按 score variance 剪枝低 discrimination 的 rubric。judge 可以参考 agent 自己生成的 rubric 作为 signal，但打分用 judge-side rubric（不奖励 follow 弱的 self-rubric）。

这是 **agent-judge co-evolution**：agent 通过参数 + rubric bank 演化，judge 通过 rubric buffer 演化。

**Theorem 4 给 dense credit 严格优于 terminal broadcast 的条件**：

定义 oracle gradient（真实想估计的）：

$$
g_k^\star = \sum_{j=k}^K \lambda_{k,j} \mathbb{E}[\Gamma_k Y_j]
$$

- $\Gamma_k = \sum_{t \in \mathcal{B}_k} \nabla_\theta \log \pi_\theta(a_t|H_t)$：stage $k$ 的 score function sum
- $Y_j$：stage $j$ 的 latent true score（不可观测）

Terminal broadcast 的 gradient 是 $g_k^{\text{term}} = \mathbb{E}[\Gamma_k R_K]$，它有两个误差来源：

1. **Omission**：它只看 final stage $Y_K$，完全忽略 $Y_1, \dots, Y_{K-1}$。被忽略的真实信号量：

$$
M_k^\Lambda = \big\|\mathbb{E}[\Gamma_k Y_K] - \sum_{j=k}^K \lambda_{k,j} \mathbb{E}[\Gamma_k Y_j]\big\|_2
$$

2. **Judge noise**：$R_K$ 对 $Y_K$ 有 noise $\epsilon_{k,K}$。

Stage-weighted 的 gradient 是 $g_k^\Lambda = \sum_j \lambda_{k,j} \mathbb{E}[\Gamma_k R_j]$，它消掉了 omission，但引入了每个 stage 的 cumulative judge noise $\sum_j \lambda_{k,j} \epsilon_{k,j}$。

结论：

$$
\|g_k^\Lambda - g_k^\star\|_2 \le \sum_{j=k}^K \lambda_{k,j} \epsilon_{k,j} \quad \text{(stage-weighted 误差上界)}
$$
$$
\|g_k^{\text{term}} - g_k^\star\|_2 \ge M_k^\Lambda - \epsilon_{k,K} \quad \text{(terminal broadcast 误差下界)}
$$

当 $\sum_j \lambda_{k,j} \epsilon_{k,j} < M_k^\Lambda - \epsilon_{k,K}$，即 **recovered intermediate signal 大过 cumulative judge noise** 时，stage-weighted 严格更接近 oracle。直觉就是：只要 judge 在每个 stage 上 sufficiently aligned（noise 不爆炸），dense signal 损失的 omission 价值就盖过引入的 noise。

参考 [Math-Shepherd](https://aclanthology.org/2024.acl-long.510/)、[Let's Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi) 是 verifiable domain 的 process supervision，RubricEM 推广到 LLM-judge noisy semantic signal。

### 病3: experience 不复用 → 药: Reflection Meta-Policy

**Intuition**：传统 RL 把 judged attempt 变成参数 update，知识压进 weight。下次相关 query 来了，weight 里有隐含知识但没有显式的 "上次我犯的错是什么" 的 textual memory。

RubricEM 的做法：用同一个 backbone $\pi_\theta$ 同时当 task policy 和 reflection generator。每次 task rollout 完、judge 打完分后，采样一条 trajectory，让 backbone 生成 8 个 reflection 候选（格式是 `<reflection_rubrics>` + `<reflection_takeaways>`），judge 对每个 candidate 打分（diagnostic accuracy 0.4 + specificity 0.3 + scope 0.3），最高分的写入 **rubric bank**（textual memory），所有 candidate 的分数作为 RL reward 更新 reflection tokens 的参数。

**Rubric bank 两种 retrieval**：
- **Within-episode**：同一 query 再来一次，retrieve 上次的 reflection 作为 "avoid past mistakes" 的直接 guidance
- **Cross-episode**：新 query，用 FAISS + Qwen3-Embedding-0.6B 检索 top-2 语义相似 past reflection，作为 "adapt to current question" 的 few-shot exemplar

**Windowed curriculum with K=3**：每 6 个 step 一个 window，前 3 个 fresh query（cross-episode retrieval），后 3 个 replay 对应 batch（within-episode retrieval）。3-step gap 保证异步 reflection pipeline 完成。

**Theorem 5 证明共享 backbone 是 co-evolution 的必要条件**：

设 task gradient $g = \nabla J_{\text{task}}$，memory gradient $h = \nabla U$。Assumption 4 要求 accepted reflection 的 conditional expected gradient 与 task gradient 正向 align：

$$
\langle g, \mathbb{E}[\Psi | A=1] \rangle \ge \mu > 0, \quad \mathbb{P}(A=1) \ge p_0 > 0
$$

- $\Psi = \Gamma^{\text{ref}}(\beta_w \Delta^w + \beta_c \Delta^c)$：reflection 的 score-gradient
- $A \in \{0,1\}$：judge acceptance indicator
- $p_0$：acceptance rate 下界
- $\mu$：alignment 强度下界

结论是 mutual improvement：

$$
J_{\text{task}}(\theta + \eta h) - J_{\text{task}}(\theta) \ge \eta p_0 \mu - \frac{L_J \eta^2}{2}\|h\|_2^2
$$

即 reflection update 也能提升 task performance（一阶项 $\eta p_0 \mu$ 是正的）。

**关键 Remark 5**：如果 task policy 和 reflection generator 用 disjoint parameters $\theta = (\theta_\pi, \theta_{\text{ref}})$，则 $\nabla J_{\text{task}} = (g_\pi, 0)$，$\nabla U = (0, h_{\text{ref}})$，内积 $\langle g, h \rangle = 0$，mutual improvement 在一阶完全消失。

**Intuition**：如果 reflection model 是独立训练的，它的 update 在 task policy 参数空间没有 projection，两个 model 互相不影响。共享 backbone 把它们耦合到同一个 loss landscape，reflection training 的 gradient 在 task 参数空间有 nontrivial component，才能互相 benefit。这给 "为什么不用 separate reflection model + inference-time retrieval" 一个 sharp 理论回答。

参考 [Reflexion](https://arxiv.org/abs/2303.11366)、[Self-Refine](https://arxiv.org/abs/2303.17651) 是 inference-time reflection，不更新参数。RubricEM 通过 parameter sharing + judge-gated RL reward 把 reflection 训进 meta-policy。参考 [MAML](https://arxiv.org/abs/1703.03400)、[PEARL](https://arxiv.org/abs/1903.08254)、[RL2](https://arxiv.org/abs/1611.02779) 是经典 meta-RL。

### 工程巧思：异步 reflection pipeline

同步实现会阻塞：task rollout → judge → reflection generation → reflection judge → meta-policy update 全做完才能开始下一步 task rollout。这 sequential bottleneck 是 Meta-RL 的老问题（参考 [Jiang et al. 2026](https://openreview.net/forum?id=4GiBscHW1k)）。

RubricEM 用 **one-step deferred** 三线程架构：
- Main thread: 交替 Phase A（训练 step $t-1$ 的 deferred reflection）和 Phase B（训练 step $t$ 的 task rollout）
- Inference thread: Phase A 期间生成 step $t+1$ 的 rollout
- Data prep thread: judge scoring 同步返回，reflection generation + judging 异步

step $t$ 的 reflection 在 step $t+1$ 训练。trade 掉 exact sync 换 infrastructure utilization，几乎零 wall-clock overhead。

---

## EM 视角

名字 RubricEM 的 EM 是 Expectation-Maximization 的致敬（参考 [Dempster et al. 1977](https://www.jstor.org/stable/2984875)），但不是标准 EM 算法，是 EM-inspired 的 view：

- **E-step (Estimate)**：给定当前 policy 分布，estimate latent structure——judge 对比 rollout 找 discriminative rubric（"what matters"），agent 在 Plan 阶段自己生成 rubric（"what should I do"）。
- **M-step (Maximize)**：给定 rubric，maximize task policy（SS-GRPO 用 rubric score 做 advantage）和 reflection meta-policy（用 reflection score 做 advantage）。

两步交替，agent 和 judge co-evolve。比传统 EM 多一个 textual memory 维度——rubric bank 本身也在 EM 循环里 evolve。

---

## 结果有多强

### Long-form（4 个 benchmark 平均）

| Model | Avg |
|---|---|
| GPT-5 + Search | 62.2 |
| OpenAI Deep Research | 59.9 |
| **RubricEM-8B (RL, 1400 steps)** | **55.5** |
| DR Tulu-8B (RL, 1900 steps) | 53.6 |
| Gemini 3.1 Pro + Search | 53.9 |
| Tongyi DeepResearch-30B-A3B | 50.8 |
| RubricEM-8B (SFT) | 49.2 |

8B 模型用 1400 步 RL 超过 1900 步的 DR Tulu，超过 30B 的 Tongyi DeepResearch，在 DRB 上甚至超过 OpenAI Deep Research（47.8 vs 46.9）。SFT → RL 提升 6.3 点，还超过了 teacher Gemini-3.1-Pro。

### Short-form transfer（完全 out-of-domain）

| Model | SimpleQA | 2Wiki | WebWalker | DSQA | Avg |
|---|---|---|---|---|---|
| DR Tulu-8B (RL, 1900) | 80.1 | 68.0 | 39.1 | 8.3 | 49.0 |
| **RubricEM-8B (RL, 1400)** | **92.3** | **78.8** | **70.0** | **53.0** | **73.5** |

Short-form 完全没用 RL data 训练，但 RubricEM 大幅胜出。说明 long-form RL 教的是 transferable 的 tool-use 和 evidence-grounding skill，long-form structure 强制更系统的 search 行为。

### Ablation

600 步下对比四个 recipe：
- Baseline-RL（standard answer-only GRPO）
- SS-GRPO（只换 stage credit）
- Meta-Policy（只加 reflection）
- Full RubricEM（组合）

两者都 improve，组合最好。说明 stage credit 和 reusable experience 提供**互补**的 gain。

Scaffold ablation：structured vs unstructured SFT，structured 在 distillation quality 和后续 RL gain 都更好。无 scaffold 的 600 步 RL 几乎不增长。同样 Gemini-3.1-Pro + 同 search backend，structured scaffold 比 ReAct prompt 在 DRB 上更高——结构本身在 prompt level 就提升 deep research behavior。

Inference-time reuse：RubricEM 在 DRB 上受益于 cross-episode 和 within-episode 两种 retrieval；Baseline-RL 在同样 retrieval 设置下不受益。说明 meta-policy 学到了 actionable guidance，单纯加 context 不够。

---

## 我的 Intuition 和联想

### 这 paper 的核心 contribution

两个独立但协同的 insight：

1. **Rubric 不只是 evaluator**，是 **shared interface**：condition policy、condition judge、condition memory。把 rubric 从 reward shaping 工具提升为整个 RL loop 的 organizing principle。

2. **Stage 结构不只是 format**，是 **credit assignment unit**。Theorem 1 用 state aliasing + Jensen 严格证明 stage 信息有 value，Theorem 4 证明 stage-weighted credit 在 judge noise < recovered signal regime 下严格胜出。

### 与 DR Tulu 的关键差别

[DR Tulu](https://arxiv.org/abs/2511.19399) 的 RLER 只对 final answer 用 evolving rubric。RubricEM 的差别：
- **Stagewise rubric**：从 final-answer judging 推广到 process-level feedback
- **Joint meta-policy training**：把 reflection 从 inference-time prompting 提升为 RL objective
- **Asynchronous execution**：避免 Meta-RL 的 sequential bottleneck

### 联想：Hindsight Credit Assignment

[Tan et al. 2026](https://arxiv.org/abs/2603.08754) 的 hindsight credit assignment 也解决 long-horizon credit，但用 hindsight 视角 back-propagate credit。RubricEM 用 prospective + rubric-guided decomposition——plan 阶段就制定 rubric，forward-looking。两者互补。

### 联想：Constitutional AI

[Constitutional AI](https://arxiv.org/abs/2212.08073) 用 principles 做 AI feedback。RubricEM 把这个 idea 推广到 multi-stage trajectory + experience reuse。rubric 既是 evaluator 也是 generator 的 condition。

### 联想：Reflexion / Self-Refine

[Reflexion](https://arxiv.org/abs/2303.11366)、[Self-Refine](https://arxiv.org/abs/2303.17651) 是 inference-time reflection，不更新参数。RubricEM 通过共享 backbone 和 judge-gated RL reward，把 reflection 训成 meta-policy 的 parameter——reflection 既是 inference-time 的 textual memory，又是训练时的 auxiliary loss。Theorem 5 严格证明 co-evolution 必须靠 parameter sharing。

### 联想：Multi-task gradient

[Gradient Surgery (Yu et al. 2020)](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)、[Conflict-averse gradient (Liu et al. 2021)](https://openreview.net/forum?id=_61Qh8tULj_) 研究 multi-task parameter sharing 的 positive/negative transfer。Theorem 5 的 Assumption 4（judge-gated local positive transfer）正是这个 framework 的应用：helpful auxiliary objective 的 gradient 与 main-task gradient align，misaligned gradient 诱导 negative transfer。

### 联想：AlphaZero-like self-play

agent 和 judge co-evolution 是 self-play 的弱化版——agent 生成 trajectory，judge 生成 rubric，两者互相 challenge。未来可以探索更对称的 self-play，比如 judge 也用 RL 训练（现在 judge 只更新 rubric buffer 不更新参数）。

### 8B 逼近 GPT-5 的意义

最 striking 的结果是 8B model 在 long-form research 上逼近 GPT-5 + Search（62.2 vs 55.5），在 DRB 上超过 OpenAI Deep Research。这表明：

- **Structure 比 scale 更 efficient**：8B + 强 structure + dense credit + memory 接近 175B+ 系统的 behavior
- **Open deep research 可达**：不需要海量 imitation data 或 proprietary infra，正确的 RL recipe + 适度 SFT 足以 competitive
- **RL 不只是 verifiable reward 的专属**：通过 rubric 作为 dense semantic signal + judge co-evolution，RL 在 open-ended domain 也能 work

### 未来方向猜想

1. **Multi-agent shared memory**：rubric bank 扩成多 agent 共享，类似 [MetaClaw](https://arxiv.org/abs/2603.17187) 的 skill library，但保持 RL co-training
2. **Hierarchical rubric**：当前 flat 4-stage，可以扩成 hierarchical（meta-plan → sub-plan → action），参考 [HiPer](https://arxiv.org/abs/2602.16165)
3. **Verifiable + unverifiable hybrid**：citation grounding 等 verifiable signal 作为 auxiliary reward，rubric 作为 primary semantic reward，用 gradient surgery 平衡
4. **Stronger judge = stronger agent**：Theorem 4 表明 judge noise 是瓶颈，scaling judge 可能比 scaling policy 更 effective
5. **Test-time scaling via reflection retrieval**：rubric bank 是 inference-time 可扩展 memory，retrieve 多个相关 reflection 做 ensemble
6. **Graph-based memory**：当前用 embedding retrieval，可以探索 graph-based 或 hierarchical memory 结构，让 cross-episode transfer 更精确

### 局限

- Judge 用 Gemini-3-Flash，cost-effective 但能力有限，subtle long-form 任务的 stage credit 和 reflection reward 可能不够准
- 训练对 infrastructure 稳定性敏感：API delay、network jitter 引入额外 staleness
- Citation grounding 不是直接优化目标（SQA-v2 被排除），未来可结合 verifiable citation reward
- Rubric bank 用 embedding retrieval，cross-episode transfer 还比较粗

---

## 一句话总结

RubricEM 的本质是**把 rubric 提升为 RL 的 latent organizing structure**。三个组件构成 EM-inspired 的循环：scaffold 暴露 task structure（解开 state aliasing），SS-GRPO 给 structure 分配 credit（dense signal 胜过 terminal broadcast），reflection meta-policy 把 judged attempt 蒸馏成 reusable memory（共享 backbone 让 co-evolution 在一阶成立）。8B 模型靠这套 recipe 在 long-form research 上逼近 GPT-5 + Search，证明 open-ended RL beyond verifiable rewards 是可行的，关键在 expose structure、assign credit to structure、convert judged attempts into reusable experience。

---

# RubricEM: 用 Rubric 作为 RL 共享接口训练 Deep Research Agent

## 1. 这篇 paper 想解决什么问题

Deep research agent 涉及 plan、search、evidence evaluation、long-form synthesis 这一长串 tool-augmented 决策。传统 RL post-training 在这类任务上面临三个困难：

- **Reward 不可验证**：长篇 answer 没有 ground truth，只能靠 LLM judge 给 rubric score，是 open-ended 的。
- **Credit assignment 稀疏且延迟**：terminal reward 被广播到整条 trajectory 上千上万个 token，plan 阶段的 token 和 answer 阶段的 token 拿到同样的 advantage，gradient signal 高度 noisy。
- **Experience 不可复用**：传统 post-training 把 judged attempt 转成 parametric update，没有显式的 textual memory 留下来供下次相关 query 使用。

核心问题作者用一句话总结：*How can RL train deep research agents beyond verifiable rewards, while enabling long-horizon credit assignment and learning from experience?*

RubricEM 给的答案是 EM-inspired 的 view：把 rubric 当成 **latent structure**——它是 agent 决策的条件、judge 打分的标准、memory 蒸馏的格式。E-step 估计 rubric（哪些 criterion 对当前 policy 分布有 discrimination），M-step maximize task policy 和 reflection meta-policy。

Paper 链接：[arXiv 2511.19399 (DR Tulu)](https://arxiv.org/abs/2511.19399)，[Gemini Deep Research blog](https://blog.google/products/gemini/google-gemini-deep-research/)，[OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)。

---

## 2. 三大组件技术详解

### 2.1 Rubric-guided Structured Reasoning Scaffold

核心 idea：把 flat autoregressive rollout 切成四个 stage，每段用 XML tag 标记，由 agent 自己在 Plan 阶段生成的 rubric 贯穿始终。

**四阶段 schema：**

```
Plan → Research → Review → Answer
```

- **Plan** (`<structured_plan>` 内含 `<deep_analysis>`, `<rubric>`, `<research_plan>`)：agent 自己写 (i) knowledge checklist (要查什么), (ii) analytical & synthesis criteria (最终回答要达成什么 intellectual connection), (iii) negative constraints (要避免什么)。
- **Research** (`<call_tool>` × N 配合 `<state_evaluation>`)：每次 tool call 后做 state evaluation，对比 accumulated evidence 和 rubric；如果 initial assumption 失效，可以 in-place 修订 Plan。
- **Review** (`<review>` 内含 `<rubric_review>` 和 `<writing_plan>`)：强制 self-evaluation，把 evidence 映回 rubric 各项；写一个 writing outline（thesis, value proposition, narrative architecture, citation mapping）。
- **Answer** (`<answer>`)：按 writing plan 写 long-form 报告，用 `<cite id="S_123">...</cite>` 做 inline citation。

**Scaffold 的几个关键约束：**

- Mandatory starting tag: `🤔`（保留 Qwen3 的 thinking 行为）
- First turn 必须 end with `</call_tool>`，禁止第一轮就 answer
- One action per turn: 每个 generation 末尾要么 `</call_tool>` 要么 `</answer>`，禁止 hallucinate tool output
- Adaptive cognitive effort: 简单 query 简短 plan，复杂 query 用全部 machinery

**SFT distillation 流程**：用 Gemini-3.1-Pro 当 teacher，prompt 拆成 first-round（只 Phase 1）和 later-rounds（只 Phase 2）两个 variant，每轮单独调 API。生成后做 rejection sampling：缺 `</answer>`、第一轮没 valid tool call、缺 structural elements、连续两次 tool error 都丢弃。`<scratchpad>` 后处理转换成 `🤔`。最终 ~11K samples（比 DR Tulu 少 2K）。

### 2.2 Stage-Structured GRPO (SS-GRPO)

GRPO 把同一个 query 的多个 rollout 放在一组，用 group-relative advantage 去掉 critic。SS-GRPO 在此基础上分 stage 打分。

**核心定义：**

设 query $q$，采样 $n$ 个 rollout $\{\tau_i\}_{i=1}^n$，每个 rollout 分成 $K=4$ 个 stage。$\mathcal{B}_{i,k}$ 是 rollout $i$ 的 stage $k$ 的 token 集合。

每个 stage 的 judge 分数 $R_{i,k} \in [0,1]$（由 LLM judge 用 stage-specific rubric 给）。

**Stage-return** 用 causal stage-weight matrix $\Lambda = (\lambda_{k,j})$：

$$
G_{i,k}^{\Lambda} = \sum_{j=k}^{K} \lambda_{k,j} R_{i,j}
$$

- $\lambda_{k,j} = 0$ if $j < k$（stage $k$ 的 token 不接收它前面 stage 的 reward，符合因果）
- $\lambda_{k,k} = 1$（每个 stage 至少保留自己的 score）
- $\lambda_{k,j}$ for $j > k$ 控制 downstream stage 的影响传播（stage $k$ 的好决策使得后续 stage 也变好，所以 stage $k$ 的 token 也该 share 一部分 $R_{i,j}$ 的 credit）

实验中实际用的 $\Lambda$ 是 lower-triangular 加权：

$$
\Lambda = \begin{pmatrix} 
1.0 & 0.4 & 0.6 & 0.8 \\ 
0 & 1.0 & 0.4 & 0.8 \\ 
0 & 0 & 1.0 & 0.8 \\ 
0 & 0 & 0 & 1.0 
\end{pmatrix}
$$

可以看出 Answer 的 weight 对每个上游 stage 都是 0.8（最大），Review 对 Plan/Research 是 0.4/0.6，Plan 只对 Research 是 0.4。这是 hand-tuned 的，但直觉合理：最终 answer 质量是所有上游 stage 的最终 outcome。

**Stage-wise normalization**（GRPO 关键）：在 rollout group 内 per-stage normalize：

$$
A_{i,k} = \frac{G_{i,k}^{\Lambda} - \frac{1}{n}\sum_{i'=1}^n G_{i',k}^{\Lambda}}{\text{Std}_{i'}[G_{i',k}^{\Lambda}] + \epsilon}
$$

注意这里 normalize 的是 $G_{i,k}^\Lambda$（同一 stage 内的所有 rollout 的 stage return），不是 terminal reward broadcast。这让每个 stage 有自己的 baseline，dense reward 在 stage 内 comparable。

**Loss**：

$$
\mathcal{L}_{\text{SS-GRPO}} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K \sum_{t \in \mathcal{B}_{i,k}} \min(\rho_{i,t} A_{i,k}, \text{clip}(\rho_{i,t}, 1-\eta, 1+\eta) A_{i,k}) + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

- $\rho_{i,t} = \pi_\theta(a_{i,t}|h_{i,t}) / \pi_{\theta_{\text{old}}}(a_{i,t}|h_{i,t})$ 是 PPO importance ratio
- 同一个 stage block $\mathcal{B}_{i,k}$ 内的所有 token 共享 advantage $A_{i,k}$
- $\beta = 0.001$（KL 系数很小）
- critic-free：因为 stage reward 是 judge-defined、evolving、从 expensive rollout 收集来的，加 critic 会爆炸复杂度

**Stagewise evolving-rubric judge**：judge 自己有一个 rubric buffer，分 stage 维护。每步对每个 query：
1. Generate adaptive rubrics：对比同一 query 的多个 rollout，找 discrimination 高的 criteria（"为什么有的 trajectory 好有的差"）。可以参考 agent 自己生成的 rubric 作为 signal，但打分用 judge-side rubric，不奖励 follow 弱的 self-rubric。
2. 加入 buffer，buffer cap = (3, 2, 2, 3) for stages 1-4
3. 对每个 trajectory 打分（positive 0/1/2，negative inverted）
4. 剪枝：按 score variance 跨 rollout group 排序，移除 discrimination 低的 rubric

这是一个 **agent–judge co-evolution**：agent 通过 policy 参数 + rubric bank 演化，judge 通过 rubric buffer 演化（不更新参数）。

### 2.3 Reflection Meta-Policy Training

把 "experience reuse" 变成显式 RL objective。task policy 和 reflection meta-policy **共享 backbone** $\pi_\theta$。

**Reflection 流程（单步）：**

1. Task rollout 完毕，judge 给完 stage-wise scores。
2. 从 rollout group 中 uniform random 选一个 trajectory $\tau_s$（不基于 score 选，模拟 inference 条件）。
3. 用同一 backbone 生成 $m=8$ 个 reflection 候选 $s_\ell \sim \pi_\theta^{\text{refl}}(\cdot | q, \tau_s)$。Reflection 输出格式：`<reflection_rubrics>` + `<reflection_takeaways>`，要 grounded in trajectory 但 transferable。
4. Judge 对每个 candidate 打分，三维度：
   - Diagnostic Accuracy (weight 0.4)：是否正确识别 strengths/weaknesses
   - Specificity (weight 0.3)：是否 concrete、actionable、tailored to this question
   - Scope & Balance (weight 0.3)：是否对类似问题有用、takeaways 是否互补
5. Reflection score 提供 RL reward 给 reflection tokens（meta-policy update）。
6. 最高分且 valid 的 candidate 写入 **rubric bank**（textual memory），其他丢弃。

**Rubric bank 两种 retrieval mode：**

- **Within-episode refinement**：同一 query 的重复 attempt，retrieve 上次的 reflection 作为直接 guidance（"avoid past mistakes"）。
- **Cross-episode transfer**：新 query，retrieve top-2 语义相似 past reflection（FAISS 索引 + Qwen3-Embedding-0.6B）作为 few-shot exemplar（"adapt to current question's specific needs"）。

**Windowed curriculum with K=3：**

每个 window 6 个 step，前 3 个 fresh query（cross-episode retrieval），后 3 个 replay 对应的 batch（within-episode retrieval）。这个 3-step gap 保证异步 reflection pipeline 完成（generate + judge + bank insert + deferred training），不会 retrieve 还没 ready 的 reflection。

**异步 reflection training（核心工程贡献）：**

三个并发线程：
- Main thread（training engine）：交替 Phase A（meta-policy update on step $t-1$ 的 deferred reflections）和 Phase B（task-policy SS-GRPO update on step $t$ 的 rollouts）
- Inference thread（vLLM）：在 Phase A 期间生成下一批 rollouts
- Data preparation thread：judge scoring 同步返回，reflection rollout + judging 异步

One-step deferred：step $t$ 的 reflections 在 step $t+1$ 的 Phase A 训练。这 trade 掉 exact sync 换 infrastructure utilization，几乎零 wall-clock overhead。这解决了 Meta-RL literature 里 sequential rollout–reflection–update bottleneck 的痛点（参考 [Jiang et al. 2026, Meta-RL induces exploration](https://openreview.net/forum?id=4GiBscHW1k)）。

---

## 3. 三个 Theorem 的 Intuition

### 3.1 Theorem 1: Value of Stage Information

设 $H$ 是 random decision point，$c = \phi(H)$ 是 compressed context，$z = \psi(H) \in [K]$ 是 stage label，$U(H,a)$ 是 continuation utility。

定义两种 value：

$$
V_{\text{flat}} = \mathbb{E}\Big[\max_{a \in \mathcal{A}} \mathbb{E}[U(H,a) | C]\Big], \quad V_{\text{stage}} = \mathbb{E}\Big[\max_{a \in \mathcal{A}} \mathbb{E}[U(H,a) | C, Z]\Big]
$$

- $V_{\text{flat}}$：policy 只看 compressed context $c$，stage 信息被 averaged 掉
- $V_{\text{stage}}$：policy 还知道当前 stage $z$，可以根据 stage 选不同 action

**核心不等式**（aliasing gap）：

$$
\Delta_{\text{alias}}(c) := \sum_{z=1}^K p(z|c) \max_a q(c,z,a) - \max_a \sum_{z=1}^K p(z|c) q(c,z,a) \ge 0
$$

- 第一项：知道 stage 后 expected max utility（先按 stage 选 action 再取期望）
- 第二项：不知道 stage 时 max expected utility（先按 marginal 选 action 再取 max）

这是 **Jensen 不等式应用于 convex function $\max$**：$\mathbb{E}[\max_a f(a, Z)] \ge \max_a \mathbb{E}[f(a, Z)]$。Max 是 convex 的，所以 $V_{\text{stage}} \ge V_{\text{flat}}$。

**Strict 的条件**：存在正概率集 $C_0$ 和两个 stage $z \ne z'$，使得 $\arg\max_a q(c,z,a) \cap \arg\max_a q(c,z,a') = \emptyset$。即两个 stage 的最优 action 完全冲突。这是 long-horizon agent 的常态：同样的 c（"我刚收到一段证据"），在 Research stage 应该决定下一个 query，在 Review stage 应该判断 evidence 是否充分。

**Intuition**：这就是经典 RL 中的 **state aliasing problem**。Flat autoregressive LM 在长 trajectory 中只能从 local context 推 stage，导致 policy 在 aliased context 上 average 化。显式 stage tag 把 aliasing 解开。

完整证明见 [Mathematics of Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) 和 [POMDP state aliasing literature](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)。

### 3.2 Theorem 3/4: Judge-Aligned Stage-Weighted Credit

这个 theorem 给 dense reward 的 trade-off 严格刻画。

设 rollout $\tau \sim p_\theta(\tau)$，stage $k$ 的 token set $\mathcal{B}_k$。每个 stage 的 score-function sum：

$$
\Gamma_k = \sum_{t \in \mathcal{B}_k} \nabla_\theta \log \pi_\theta(a_t | H_t)
$$

定义三种 stage-$k$ gradient signal：

$$
g_k^\Lambda = \sum_{j=k}^K \lambda_{k,j} \mathbb{E}[\Gamma_k R_j] \quad \text{(judge-induced stage signal)}
$$
$$
g_k^{\text{term}} = \mathbb{E}[\Gamma_k R_K] \quad \text{(terminal broadcast)}
$$
$$
g_k^\star = \sum_{j=k}^K \lambda_{k,j} \mathbb{E}[\Gamma_k Y_j] \quad \text{(oracle, true process gradient)}
$$

- $R_j$：observed judge score
- $Y_j$：latent true process score
- $g_k^\star$ 是我们真正想估计的"理想 process gradient"

**Omitted signal**（被 terminal broadcast 忽略的真实中间信号）：

$$
M_k^\Lambda = \Big\|\mathbb{E}[\Gamma_k Y_K] - \sum_{j=k}^K \lambda_{k,j} \mathbb{E}[\Gamma_k Y_j]\Big\|_2
$$

**Assumption 2 (Judge alignment)**：对每对 $k \le j$，存在 $\epsilon_{k,j} \ge 0$ 使得

$$
\big\|\mathbb{E}[\Gamma_k (R_j - Y_j)]\big\|_2 \le \epsilon_{k,j}
$$

即 judge score $R_j$ 与 latent true $Y_j$ 在 $\Gamma_k$ 张成的 gradient 方向上有 bounded 误差。

**结论**：

$$
\|g_k^\Lambda - g_k^\star\|_2 \le \sum_{j=k}^K \lambda_{k,j} \epsilon_{k,j} \quad \text{(stage-weighted error)}
$$
$$
\|g_k^{\text{term}} - g_k^\star\|_2 \ge M_k^\Lambda - \epsilon_{k,K} \quad \text{(terminal broadcast error)}
$$

只要 $\sum_{j=k}^K \lambda_{k,j} \epsilon_{k,j} < M_k^\Lambda - \epsilon_{k,K}$，stage-weighted credit **严格** 比 terminal broadcast 更接近 oracle gradient。

**Intuition**：Terminal broadcast 有两类误差——(a) 它根本没看到 intermediate stage 的真实信号 $Y_1, \dots, Y_{K-1}$，这部分误差是 $M_k^\Lambda$（deterministic 的 omission），(b) judge 在 final stage 的 noise $\epsilon_{k,K}$。Stage-weighted 把 omission 消掉了，但换来 cumulative judge noise $\sum_j \lambda_{k,j} \epsilon_{k,j}$。当 (a) - (b) > cumulative noise，dense credit 胜出。

这是 dense reward shaping 在 unverifiable domain 上的理论 support：不需要 oracle process reward，只要 judge 在每个 stage 上 sufficiently aligned，intermediate signal 的 recovered value 就 outweigh cumulative noise。

参考 [Math-Shepherd: Verify and reinforce LLMs step-by-step](https://aclanthology.org/2024.acl-long.510/)、[Let's Verify Step by Step (Lightman et al.)](https://openreview.net/forum?id=v8L0pN6EOi) 是 verifiable domain 的工作；RubricEM 把它推广到 LLM-judge 给的 noisy semantic signal。

### 3.3 Theorem 5: Judge-Gated Co-Evolution

这个 theorem 严格说明 **parameter sharing** 是 co-evolution 的必要条件。

**Setup：**

- Task objective: $J_{\text{task}}(\theta) = \mathbb{E}[R(Q,T;\mathcal{M})]$，期望对 $Q \sim \mathcal{D}$, $T \sim p_\theta(\cdot | Q, \mathcal{M})$
- Memory objective: $U(\theta) = \mathbb{E}[A(\tilde Q, \tilde T, S; \mathcal{M})(\beta_w \Delta^w + \beta_c \Delta^c)]$
  - $A \in \{0,1\}$：acceptance indicator（judge gate）
  - $\Delta^w$：within-episode usefulness score
  - $\Delta^c$：cross-episode usefulness score

Score-function gradients：

$$
g = \nabla J_{\text{task}} = \mathbb{E}[\Gamma^{\text{traj}} R], \quad h = \nabla U = \mathbb{E}[A \Psi]
$$

其中 $\Psi = \Gamma^{\text{ref}}(\beta_w \Delta^w + \beta_c \Delta^c)$ 是 reflection 的 score-gradient。

**Assumption 4 (Judge-gated local positive transfer)**：存在 $p_0 \in (0,1]$ 和 $\mu > 0$ 使得

$$
\mathbb{P}(A=1) \ge p_0, \quad \langle g, \mathbb{E}[\Psi | A=1] \rangle \ge \mu
$$

- $p_0$：acceptance rate 至少有下界
- $\mu$：accepted reflections 的 conditional expected gradient 与 task gradient 内积为正

**结论：**

(a) **Mutual improvement**：

$$
U(\theta + \eta g) - U(\theta) \ge \eta p_0 \mu - \frac{L_U \eta^2}{2} \|g\|_2^2
$$
$$
J_{\text{task}}(\theta + \eta h) - J_{\text{task}}(\theta) \ge \eta p_0 \mu - \frac{L_J \eta^2}{2} \|h\|_2^2
$$

即 task update 也提升 memory objective，reflection update 也提升 task objective（一阶 + small step size）。

(b) **Dominance over task-only with static memory**：

$$
J_{\text{task}}(\theta_{\text{co}}^+) - J_{\text{task}}(\theta_{\text{stat}}^+) \ge \eta p_0 \mu - L_J \eta^2 \|g\|_2 \|h\|_2 - \frac{L_J \eta^2}{2} \|h\|_2^2
$$

其中 $\theta_{\text{stat}}^+ = \theta + \eta g$，$\theta_{\text{co}}^+ = \theta + \eta(g+h)$。当 $p_0 \mu > L_J \eta (\|g\|_2 \|h\|_2 + \frac{1}{2} \|h\|_2^2)$ 时，co-evolution 严格胜出 task-only + static memory。

**关键 Remark 5（为什么 parameter sharing 必要）：**

如果 task policy 和 reflection meta-policy 用 disjoint parameters $\theta = (\theta_\pi, \theta_{\text{ref}})$，则

$$
\nabla J_{\text{task}}(\theta) = (g_\pi, 0), \quad \nabla U(\theta) = (0, h_{\text{ref}})
$$
$$
\langle \nabla J_{\text{task}}, \nabla U \rangle = 0
$$

**Mutual improvement 在一阶完全消失**。共享 backbone 让 $\nabla U$ 在 task 参数空间有 nontrivial projection，reflection training 才能改善 task performance。

**Intuition**：这给 "为什么不用 separate reflection model + inference-time retrieval" 一个 sharp 回答。如果 reflection model 独立训练，它的 update 和 task policy 的 update 在参数空间正交，二者无法互相 benefit。共享 backbone 把它们耦合到同一个 loss landscape，一阶 gradient 互相 align。

参考 [MAML (Finn et al. 2017)](https://arxiv.org/abs/1703.03400)、[PEARL (Rakelly et al. 2019)](https://arxiv.org/abs/1903.08254)、[Gradient surgery for multi-task (Yu et al. 2020)](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)、[Conflict-averse gradient descent (Liu et al. 2021)](https://openreview.net/forum?id=_61Qh8tULj_) 都是 multi-task parameter sharing 相关工作。

---

## 4. 工程实现细节

### 4.1 RL Hyperparameters

- Base: Qwen3-8B
- Algorithm: GRPO / SS-GRPO
- Rollouts per prompt: 8
- Unique prompts per step: 32
- Effective batch: 256
- LR: $5 \times 10^{-7}$
- KL coefficient: 0.001 (KL3 estimator)
- Max response: 18432 tokens, max prompt: 8192, max total pack: 26624
- Max tool calls per trajectory: 10
- DeepSpeed ZeRO-3 with CPU offloading
- Judge: Gemini-3-Flash
- Rubric buffer cap: (3, 2, 2, 3) for stages 1-4
- Bank retrieval top-k: 2
- Reflection candidates per trajectory: 8
- Windowed curriculum K: 3
- Bank save: every 10 steps

### 4.2 Search Tools

- `google_search`: Gemini-3-Flash + Google Search grounding，返回 AI-synthesized summaries with grounding snippets
- `snippet_search`: Semantic Scholar API，返回 academic paper excerpts

### 4.3 Training Data

- SFT: ~11K samples（蒸馏自 Gemini-3.1-Pro，rejection sampling 过滤违规）
- RL: ~4.9K deep research queries，来自 SearchArena 和 OpenScholar

### 4.4 异步 Pipeline 的巧思

One-step deferred reflection training：step $t$ 的 task rollouts 在 step $t$ Phase B 训练，step $t$ 的 reflection 在 step $t+1$ Phase A 训练。Step $t+1$ 的 rollouts 在 Phase A 期间并行生成。这避免 reflection generation + judging + update 阻塞下一步 rollout 的 sequential bottleneck。

---

## 5. 实验结果

### 5.1 Long-form benchmarks

| Model | HealthBench | ResearchQA | DRB | ResearchRubrics | Avg |
|---|---|---|---|---|---|
| GPT-5 + Search | 59.5 | 78.2 | 50.7 | 60.5 | 62.2 |
| OpenAI Deep Research | 53.8 | 79.2 | 46.9 | 59.7 | 59.9 |
| Gemini 3.1 Pro + Search | 47.5 | 74.5 | 44.4 | 49.1 | 53.9 |
| DR Tulu-8B (RL, 1900) | 50.2 | 74.3 | 43.4 | 46.4 | 53.6 |
| Tongyi DeepResearch-30B-A3B | 46.2 | 66.7 | 40.6 | 49.5 | 50.8 |
| WebThinker-32B-DPO | 39.4 | 74.2 | 40.6 | 41.9 | 49.0 |
| **RubricEM-8B (RL, 1400)** | **49.3** | **74.5** | **47.8** | **50.3** | **55.5** |
| RubricEM-8B (SFT) | 39.0 | 71.8 | 43.0 | 42.8 | 49.2 |
| Qwen3-8B + Search | 24.5 | 58.4 | 28.2 | 24.5 | 33.9 |

关键观察：
- 8B backbone 达到 55.5 average，**比 1900-step 的 DR Tulu-8B-RL 高 1.9 点且少用 500 steps**
- 在 DRB 上甚至超过 OpenAI Deep Research (47.8 vs 46.9)
- 比 Tongyi DeepResearch-30B-A3B（30B 模型）还高 4.7 点
- SFT → RL 提升 6.3 点（49.2 → 55.5），且超越 teacher Gemini-3.1-Pro（53.9）

### 5.2 Short-form transfer (out-of-domain)

| Model | SimpleQA | 2Wiki | WebWalker | DSQA | Avg |
|---|---|---|---|---|---|
| DR Tulu-8B (RL, 1900) | 80.1 | 68.0 | 39.1 | 8.3 | 49.0 |
| **RubricEM-8B (RL, 1400)** | **92.3** | **78.8** | **70.0** | **53.0** | **73.5** |

Short-form 没用任何 RL data，但 RubricEM 大幅胜出 DR Tulu。说明 long-form RL 教会了 transferable tool-use 和 evidence-grounding skill，而 long-form structure 强制更系统的 search 行为。

### 5.3 Ablation (600 steps)

四个 recipe 在 600-step budget 下比较：
- Baseline-RL: standard answer-only GRPO
- SS-GRPO: 只换 stage credit
- Meta-Policy: 只加 reflection training
- Full RubricEM: 组合

结果：两者都 improve，组合最好。说明 stage credit 和 reusable experience 提供互补 gain。

### 5.4 Scaffold ablation

- Structured vs unstructured SFT: structured 在 distillation quality 和后续 RL gain 都更好。无 scaffold 的 600-step RL 几乎不增长。
- 同样 Gemini-3.1-Pro + 同 search backend，structured scaffold 比 ReAct prompt 在 DRB 上更高——结构本身就在 prompt level 提升 deep research behavior。

### 5.5 Inference-time reuse

RubricEM 在 DRB 上受益于 both cross-episode 和 within-episode retrieval；Baseline-RL 在同样 retrieval 设置下不受益。说明 meta-policy 学到了 actionable、reusable guidance，单纯加 context 不够。

---

## 6. 我的 Intuition 和思考

### 6.1 这篇 paper 的核心 contribution

**两个独立但协同的 insight**：

1. **Rubric 不只是 evaluator**，而是 **shared interface**：condition policy、condition judge、condition memory。这把 rubric 从 reward shaping 工具提升为整个 RL loop 的 organizing principle。

2. **Stage 结构不只是 format**，而是 **credit assignment unit**。Paper 用 Theorem 1（state aliasing + Jensen）严格证明 stage 信息有 value，用 Theorem 4 证明 stage-weighted credit 在 judge noise < recovered signal regime 下严格胜出。

### 6.2 与 DR Tulu 的关键差别

[DR Tulu (Shao et al. 2025)](https://arxiv.org/abs/2511.19399) 引入 RLER（Reinforcement Learning with Evolving Rubrics）只对 final answer 用 evolving rubric。RubricEM 的差别：
- **Stagewise rubric**：从 final-answer judging 推广到 process-level feedback（Theorem 4 给理论 support）
- **Joint meta-policy training**：把 reflection 从 inference-time prompting 提升为 RL objective（Theorem 5 给理论 support）
- **Asynchronous execution**：避免 Meta-RL 的 sequential bottleneck

### 6.3 EM 视角的妙处

E-step: 给定当前 policy 分布，estimate rubric——judge 对比 rollout 找 discriminative criteria（rubric 估计 latent structure "what matters"）。同时 agent 在 Plan stage 自己生成 rubric（self-estimate "what should I do"）。

M-step: 给定 rubric，maximize task policy（SS-GRPO 用 rubric score 做 advantage）和 reflection meta-policy（用 reflection score 做 advantage）。

两步交替，agent 和 judge co-evolve，rubric buffer 和 rubric bank 都动态更新。这比传统 EM 多了一个 textual memory 维度——memory 本身也在 EM 循环里 evolve。

### 6.4 联想：与 Hindsight Credit Assignment 的关系

[Tan et al. 2026 (Hindsight credit assignment for long-horizon LLM agents)](https://arxiv.org/abs/2603.08754) 也在解决 long-horizon credit assignment，但用的是 hindsight 视角。RubricEM 用的是 prospective + rubric-guided decomposition——plan 阶段就制定 rubric，是 forward-looking 的。两者互补：hindsight 利用 outcome information back-propagate credit，rubric 用 prior structure forward-propagate。

### 6.5 联想：与 Constitutional AI 的关系

[Constitutional AI (Bai et al. 2022)](https://arxiv.org/abs/2212.08073) 用 principles/rubric 做 AI feedback。RubricEM 把这个 idea 推广到 multi-stage trajectory + experience reuse。rubric 既是 evaluator 也是 generator 的 condition，这是 Constitutional AI 思想的 RL 训练版。

### 6.6 联想：与 Reflexion / Self-Refine 的关系

[Reflexion (Shinn et al.)](https://arxiv.org/abs/2303.11366) 和 [Self-Refine (Madaan et al.)](https://arxiv.org/abs/2303.17651) 是 inference-time 的 reflection 机制，不更新参数。RubricEM 通过共享 backbone 和 judge-gated RL reward，把 reflection 训练成 meta-policy 的 parameter——reflection 既是 inference-time 的 textual memory，又是训练时的 auxiliary loss。Theorem 5 严格证明这个 co-evolution 必须靠 parameter sharing，separate reflection model 拿不到 mutual improvement。

### 6.7 局限和开放问题

- Judge 用 Gemini-3-Flash，cost-effective 但能力有限。Stronger judge ensemble 可能给更准确的 stage credit。
- 训练对 infrastructure 稳定性敏感：API delay、network jitter 会引入额外 staleness。
- Citation grounding 不是直接优化目标（SQA-v2 被排除），未来工作可结合 verifiable citation reward。
- Rubric bank 用 embedding retrieval，cross-episode transfer 还比较粗，可以考虑 graph-based 或 hierarchical memory。

### 6.8 8B 模型逼近 GPT-5 + Search 的意义

这 paper 最 striking 的结果是 8B model 在 long-form research 上逼近 GPT-5 + Search (62.2 vs 55.5)，并在 DRB 上超过 OpenAI Deep Research。这表明：

- **Structure 比 scale 更 efficient**：8B + 强 structure + dense credit + memory 接近 175B+ 系统的 behavior。
- **Open deep research 是可达的**：不需要海量 imitation data 或 proprietary infra，正确的 RL recipe + 适度 SFT 足以 competitive。
- **RL 不只是 verifiable reward 的专属**：通过 rubric 作为 dense semantic signal + judge co-evolution，RL 在 open-ended domain 也能 work。

### 6.9 对未来方向的猜想

1. **Multi-agent co-evolution**：可以把 rubric bank 扩成多 agent 共享 memory，类似 [MetaClaw (Xia et al. 2026)](https://arxiv.org/abs/2603.17187) 的 skill library，但保持 RL co-training。
2. **Hierarchical rubric**：当前是 flat 4-stage，可以扩成 hierarchical（meta-plan → sub-plan → action），可能用 [HiPer (Peng et al. 2026)](https://arxiv.org/abs/2602.16165) 的 hierarchical RL 思路。
3. **Verifiable + unverifiable hybrid**：把 citation grounding 等 verifiable signal 作为 auxiliary reward，rubric 作为 primary semantic reward，类似 multi-task learning 的 gradient surgery。
4. **Stronger judge = stronger agent**：Theorem 4 表明 judge noise 是瓶颈，scaling judge 可能比 scaling policy 更 effective。
5. **Test-time scaling via reflection retrieval**：rubric bank 是 inference-time 可扩展的 memory，可以 retrieve 多个相关 reflection 做 ensemble。
6. **Connection to AlphaZero-like self-play**：agent 和 judge co-evolution 是 self-play 的弱化版本，未来可以探索更对称的 self-play。

---

## 7. 相关文献

- [RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards](https://arxiv.org/abs/2511.19399) - 本 paper（DR Tulu 系列）
- [DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](https://arxiv.org/abs/2511.19399) - 直接 prior
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) - GRPO 原始方法
- [Rubrics as Rewards (Gunjal et al. 2025)](https://arxiv.org/abs/2507.17746) - rubric as reward
- [Constitutional AI (Bai et al. 2022)](https://arxiv.org/abs/2212.08073) - rubric-based AI feedback
- [Math-Shepherd (Wang et al. 2024)](https://aclanthology.org/2024.acl-long.510/) - process supervision without annotations
- [Let's Verify Step by Step (Lightman et al. 2024)](https://openreview.net/forum?id=v8L0pN6EOi) - PRM
- [MAML (Finn et al. 2017)](https://arxiv.org/abs/1703.03400) - meta-RL
- [PEARL (Rakelly et al. 2019)](https://arxiv.org/abs/1903.08254) - meta-RL with latent context
- [RL2 (Duan et al. 2016)](https://arxiv.org/abs/1611.02779) - meta-RL with recurrence
- [Gradient Surgery (Yu et al. 2020)](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) - multi-task gradient
- [Conflict-averse gradient (Liu et al. 2021)](https://openreview.net/forum?id=_61Qh8tULj_) - multi-task
- [Asynchronous RLHF (Noukhovitch et al. 2024)](https://arxiv.org/abs/2410.18252) - 异步 RL infra
- [WebThinker (Li et al. 2025c)](https://arxiv.org/abs/2504.21776) - deep research agent
- [Search-R1 (Jin et al. 2025)](https://arxiv.org/abs/2503.09516) - search RL
- [HealthBench (Arora et al. 2025)](https://arxiv.org/abs/2505.08775) - benchmark
- [ResearchQA (Yifei et al. 2025)](https://arxiv.org/abs/2509.00496) - benchmark
- [DeepResearchBench (Du et al. 2025)](https://arxiv.org/abs/2506.11763) - benchmark
- [ResearchRubrics (Sharma et al. 2026)](https://openreview.net/forum?id=ErnvfmSX0P) - benchmark
- [SimpleQA (Wei et al. 2024)](https://arxiv.org/abs/2411.04368) - short-form benchmark
- [OpenScholar (Asai et al. 2024)](https://arxiv.org/abs/2411.14199) - training data source
- [SearchArena (Miroyan et al. 2025)](https://arxiv.org/abs/2506.05334) - training data source
- [Qwen3 Technical Report (Yang et al. 2025)](https://arxiv.org/abs/2505.09388) - base model
- [EM Algorithm (Dempster et al. 1977)](https://www.jstor.org/stable/2984875) - EM 经典
- [POMDP / State aliasing](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) - Theorem 1 的理论背景
- [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) - Theorem 1 的核心
- [Meta-RL induces exploration (Jiang et al. 2026)](https://openreview.net/forum?id=4GiBscHW1k) - meta-RL
- [HiPer (Peng et al. 2026)](https://arxiv.org/abs/2602.16165) - hierarchical RL
- [Hindsight credit assignment (Tan et al. 2026)](https://arxiv.org/abs/2603.08754) - hindsight credit
- [AgentPRM (Xi et al. 2026)](https://arxiv.org/abs/2604.09459) - process reward for agents
- [MetaClaw (Xia et al. 2026)](https://arxiv.org/abs/2603.17187) - meta-learning agent

---

## 8. 总结

RubricEM 的本质是**把 rubric 提升为 RL 的 latent organizing structure**。三个组件构成 EM-inspired 的循环：

- **Scaffold**（structure）：把 flat trajectory 切成 rubric-conditioned stages，解开 state aliasing（Theorem 1）
- **SS-GRPO**（assign）：用 stagewise rubric judge 给 dense credit，在 judge noise < recovered signal regime 下严格胜出 terminal broadcast（Theorem 4）
- **Reflection meta-policy**（evolve）：共享 backbone 训练 textual memory，parameter sharing 让 mutual improvement 在一阶成立（Theorem 5）

工程上靠 one-step deferred 异步 pipeline 避免瓶颈，windowed curriculum 保证 retrieval consistency。

结果：8B 模型在 long-form research 接近 GPT-5 + Search、超过 OpenAI Deep Research（DRB）、short-form 大幅 transfer。这给 "open-ended RL beyond verifiable rewards" 一个具体可行的 recipe：expose task structure、assign credit to that structure、convert judged attempts into reusable experience。
