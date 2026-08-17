---
source_pdf: The MiniMax-M2 Series Mini Activations Unleashing Max Real-World Intelligence.pdf
paper_sha256: 4c090a07b73dade56ed3e90f7fdf56a183601c12d2f2d1f81eb46f24cff311fd
processed_at: '2026-08-12T14:37:22-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 聊聊 MiniMax-M2：用人话拆解一下

Andrej 咱们坐下来慢慢聊。这篇 paper 我看完最大的感受是——它表面上是 model paper，骨子里是 **post-training 的 engineering manifesto**。我尽量把里面那些 jargon 翻译成"为什么这么干"的直觉。

## 一、这篇 paper 到底想讲啥

30 秒版：MiniMax 团队搞了个 229.9B total / 9.8B activated 的 MoE，然后在它上面搭了 **agent-native 的 post-training stack**，最后用 ~10B 的 per-token compute 在 agentic coding、deep search、office task、reasoning 上打平或接近 Claude Opus 4.6、GPT 5.4、Gemini 3.1 Pro 这些 frontier 闭源。

但这个 surface number 其实不重要。重要的是它揭示了 **2025-2026 这一代 frontier model 团队真正在拼什么**——base model 早就不是 bottleneck，agentic data pipeline + RL infra + self-evolution loop 才是。

参考一下背景：https://arxiv.org/abs/2412.19437 (DeepSeek-V3)，https://arxiv.org/abs/2401.06066 (DeepSeekMoE)，https://arxiv.org/abs/2506.13585 (MiniMax-M1)

## 二、为什么是 mini-activation MoE，这个 thesis 真正的含义

paper 反复说 "mini activations unleash max real-world intelligence"。表面上看是"小模型也能打大模型"，但这个 framing 其实太浅了。真正的逻辑链是这样：

**agent 时代的 cost 结构变了**。传统 chatbot 一个 request 几百 token，cost 主要在 prefill。但 agent 一个 episode 跑几千 step、上下文 192K、中间无数次 tool call、生成上百万 token——inference cost 几乎完全在 decode。这种场景下，per-token activated parameters 直接决定你能不能 **deploy 到 production**。

所以 mini-activation 的真正意义不是"省点 GPU 钱"，是**把省下来的 compute redirect 到 long-horizon rollout 的经济可行性上**。一个 192K context、几千 action 的 SWE agent episode，如果用 dense 100B activated model，单次 cost 可能几十美金起；9.8B activated 把这个 cost 压到可以反复 rollout 的量级。

而反复 rollout 是 RL training 的 prerequisite，self-evolution 又需要 model 自己反复 rollout 自己——这个 **compute footprint → trajectory count → self-evolution economic feasibility** 的传导链，才是 "mini activations unleash max intelligence" 的真意。

我扯远一点联系：这跟 OpenAI o1 之后整个 field 的转向是一致的。2023 之前大家拼 base model scale，2024 开始拼 reasoning RL 的 test-time compute，2025 开始拼 agent 的 long-horizon RL。每一步都是把 compute 的 bottleneck 从 training 挪到 inference rollout。MoE 是这个 trend 下的天然选择，因为它让 inference compute 可控。

DeepSeek 在 V3/R1 上已经走通了这条路，MiniMax 这篇是把这条路推到 agent-native 的极致——不仅是 reasoning RL，是 agent loop 的 RL。

## 三、Architecture 部分的几个细节为什么这么选

### 3.1 Fine-Grained Experts 的真正动机

paper 用 256 个小 expert，top-8 activated。Table 1 的 ablation 显示比 32 expert top-2 在 MATH 上 +4.5、HumanEval +2.8。

但我觉得 paper 没说清楚的是 **routing 的 combinatorial diversity** 这个事到底意味着什么。传统 32 expert top-2 的 routing 空间是 $C(32,2) = 496$ 种组合，256 expert top-8 是 $C(256,8) \approx 1.6 \times 10^{14}$ 种。差了 11 个数量级。

这意味着什么？意味着 model 可以学到 **极其细粒度的 specialization**。一个 token 不仅能选"Python expert"，还能同时选"Python + sorting + recursion + memory layout + ..."的精细组合。这对 reasoning-heavy 任务特别重要，因为 reasoning 本质是组合性的——你需要同时调动多个 skill。

还有个 paper 没强调的工程考量：**multi-GPU 分片的负载均衡**。大 expert 容易出现 hot expert（某个 expert 被 50% token 选中），分片时这个 GPU 成 bottleneck。小 expert 的 utilization variance 自然小，pipeline parallelism 更平滑。这是 256 这个数字背后的工程直觉。

DeepSeekMoE paper: https://arxiv.org/abs/2401.06066

### 3.2 Sigmoid Gating vs Softmax Top-k

这个改动看着小，其实是个 **routing philosophy 的转变**。

softmax top-k 的本质是 zero-sum game——一个 expert 概率上升必然压低其他 expert。这在 training 早期容易导致 **expert collapse**：几个 expert 抢走所有 token，其他 expert 饿死。

sigmoid 把每个 expert 独立打分，移除 zero-sum 约束。一个 token 可以让"Python expert"和"sorting expert"同时 high confidence，而不是被迫二选一。

这个对 long-horizon agent 特别重要。agent 一个 step 可能需要同时调用多种 capability（比如"理解代码语义"+"调用 git"+"写 commit message"），sigmoid gating 让 routing 更接近这种"多 capability 并发"的真实需求。

paper 把这个叫 "smoother routing dynamics"——我理解是 training 时 expert utilization 的演化更平滑，不容易出现"某 epoch 某 expert 突然爆发"的 instability。

参考 Shazeer 2017 原 MoE: https://arxiv.org/abs/1701.06538

### 3.3 Expert Bias 做隐式负载均衡

传统 MoE 用 auxiliary load-balancing loss 强行均匀化 expert utilization。这个 loss 有个问题：它跟主 loss 是 conflict 的——主 loss 想让 token 去最合适的 expert，auxiliary loss 想让 token 去最闲的 expert，两个 gradient 互相打架。

M2 的解法：给每个 expert 加 learnable bias $b_e$，作为 routing score 的 shift。如果某 expert 被过度使用，optimizer 会自动降低它的 bias，"价格"上涨，流量自然分流到其他 expert。

这是把 **load balancing 从 loss constraint 转化成 parameter optimization**——更平滑、更 implicit、跟主 loss 不冲突。本质是把市场机制引进 routing。

参考 Auxiliary-loss-free load balancing: https://arxiv.org/abs/2408.15664

### 3.4 放弃 Hybrid Attention 的真实原因

paper Section 2.2.2 写得比较委婉，但我读出来的潜台词是：**linear/efficient attention 的 infrastructure 太不成熟了**，是真正的工程 blocker，不是 algorithm 的问题。

他们尝试了 SWA（Sliding Window Attention）+ full attention 的 hybrid。Table 2/3 显示：
- Pretraining 时 SWA 在 RULER 128K 上从 90.0 掉到 72.0，MTOB translation 从 60.0 掉到 45.0——长上下文严重退化
- SFT 后 32K 以内 SWA 甚至略好（IFBench 23.1 → 27.2，XBench-ds 58.0 → 63.0），但超过 32K 的 agent task 严重退化（SWE-verified 54.7 → 50.2，BrowseComp-zh 32.8 → 28.7）

但 paper 自己承认真正的 blocker 是 infra：
1. Linear attention 在 training 时就 memory-bound
2. Inference 时对 low-precision storage 敏感
3. 没有 native prefix cache 支持
4. 跟 speculative decoding 集成 unclear

这些 infra gap 在 192K context 的 agent 场景下是 deal-breaker。Prefix cache 对 multi-turn agent 太关键——每个 turn 都共享前面所有 turn 的 KV，没有 prefix cache 等于每个 turn 重算 192K prefill，cost 爆炸。

这个观察其实是给整个 linear attention 社区的一个 warning：**algorithmic elegance 不能弥补 infrastructure gap**。MiniMax-Text-01 用 Lightning Attention 踩过坑，M2 退回 full attention 是个务实选择。

MiniMax-Text-01: https://arxiv.org/abs/2501.01464

### 3.5 MTP 的 Weight Copying 扩展

MTP（Multi-Token Prediction）本身是 DeepSeek-V3 和 Gloeckle 的工作，M2 的创新在 **continued pre-training 阶段从 k=1 扩到 k=3 的初始化方式**。

随机初始化新 MTP module 会"污染"main model——新 module 初始 loss 高，gradient 会 backprop 回 main model，短暂 degrade 主模型 representation。

M2 用 weight copying：把 main model 的权重复制到新 MTP module。这个 init 让新 module 起始就接近 main model 的 representation，不污染主模型。

训练 schedule 也讲究：先 freeze main model 只训 MTP，等 MTP loss 稳定后再 joint training。如果一直 frozen main model，MTP 收敛到 worse quality——必须 co-evolve。

这个 insight 让我想到一个更普遍的 principle：**auxiliary module 的初始化要尽量"贴近"main model 的 representation manifold**，否则会破坏已学到的 feature。这跟 LoRA init 用 zero init B matrix 是一个 family 的 intuition。

MTP 原 paper: https://arxiv.org/abs/2404.19737
DeepSeek-V3: https://arxiv.org/abs/2412.19437
Speculative decoding: https://arxiv.org/abs/2211.17192

## 四、Post-Training Data Pipeline——这才是 paper 的真正核心

paper 的 Section 4 占了很大篇幅，我读完的感受是：**MiniMax 团队真正在 build 的是"verifiable reward 工厂"**。

传统 RLHF 的 reward 来自 human preference 或 reward model——这两种 reward 都有 hallucination、reward hacking、不可复现的问题。M2 的每一条 post-training trajectory 都 ground 在 **executable workspace + verifiable reward**。这是让 RL training 在 agent 场景 work 的 prerequisite。

我逐个拆：

### 4.1 SWE-Scaling Pipeline：从 GitHub PR 反向构造训练数据

这个 pipeline 的优雅之处在于它 **把 GitHub 这个天然 verifiable signal source 工程化**。

GitHub PR 本身就包含：problem statement（issue）、code diff（solution）、test cases（verification）。理论上直接拿来训 SWE agent 就行。但实际有三个问题：
1. **Diversity 不够**：90% 的 PR 是琐碎的 typo fix / dependency bump
2. **Verifiability 弱**：很多 PR 没有测试，或者测试跟 issue 关系模糊
3. **Volume 不够**：高质量 PR 数量不足以支撑大规模 RL

M2 的 6-stage pipeline 解决这三个问题。我最想讲的是 Stage 2 和 Stage 4。

**Stage 2: Agent-Synthesized Multi-Language Docker Environments**

非 Python 语言的 Docker env 合成是个公认的 hard problem。Java/Go/Rust/C++ 各有各的 build system、dependency 难题、version conflict。

M2 的解法：**让 LLM agent 自己迭代生成 build script**，用 execution feedback 引导修复。这本质是把 "Docker environment synthesis" 当成一个 agent task 本身来解，用 LLM 的代码能力 bootstrap 出环境构造能力。

这有个很妙的 self-reference：用 agent 能力构造 agent 训练数据。如果 agent 能力不够强，构造的 env 质量差；env 质量差，训练出来的 agent 更弱。这是个 chicken-and-egg。但 MiniMax 的 base model 已经够强，能 break in 这个 loop——先有 weak agent 造 weak env，训出 stronger agent，再造 stronger env，迭代。

这个 pattern 我觉得是 agent data pipeline 的未来形态——**用 agent 造 agent data**。Terminal-Gym 的 Stage 1 也是这个 pattern。

**Stage 4: Test-Based Verifiable Reward Construction**

不同类型 PR 用不同 reward function，这个设计很精细：

- **Bug fix**: 用 F2P（Fail-to-Pass）+ P2P（Pass-to-Pass）。F2P 验证 bug 真的被修了，P2P 验证没引入新 bug。两者缺一不可——只看 F2P 会让 agent 学到"删掉测试就能通过"的 reward hacking。
- **Feature addition**: F2P/P2P 不适用（新功能引入新测试）。改用"newly added test points"——golden patch 必须通过新加的测试点。
- **Performance optimization**: 没有修复过程，reward 来自"P2P 测试验证的稳定性能差异"。需要 before/after 性能对比，且差异要 statistically significant。

这个分类型 reward 设计是 **anti-reward-hacking 的关键**。统一的 reward function 容易被 model 钻空子（比如删测试、注释代码）。type-specific reward 把 model 的"作弊空间"压缩到最小。

参考 SWE-Smith 的 commit merging: https://arxiv.org/abs/2412.21132
SWE-bench: https://arxiv.org/abs/2310.06770

### 4.2 AppDev 的 Agent-as-a-Verifier (AaaV)：我最欣赏的设计

传统 LLM-as-judge 评分方式是看 static code 或 screenshot。AaaV 把 generated app **部署到 sandbox**，让 verifier agent 用 Playwright **实际 click、interact、跑 workflow**。

三层 evaluation：
1. Execution Layer：file 存在 + syntax 有效 + build 成功 + HTTP 200 + 无 JS error。失败立即 reject。
2. Interaction Layer：Playwright 检查按钮响应、表单交互、端到端 workflow 完成。
3. Visual Aesthetics Layer：layout、hierarchy、color、UI 标准。

**这个设计的哲学是"用代码而不是看代码"**。verifier 不读 code，直接 deploy + 用，跟真人 QA 一样。

为什么这个比 LLM-as-judge 强？因为 static code 评分会漏掉 runtime bug——代码看起来对，跑起来崩。LLM-as-judge 会被表面 syntactic correctness 误导，AaaV 不会被误导。

更深一层：AaaV 提供的 reward signal 是 **environment-grounded** 的，跟 RL 训练需要的 reward 性质一致。RL reward 必须是"agent 在 environment 中真实行为的结果"，不能是"另一个 model 对 code 的 opinion"。AaaV 把 reward 从 opinion 升级为 observation。

我大胆联想一下：这个 pattern 可以扩展到很多 domain。Web app 用 Playwright，数据分析用真实 DB query，document generation 用 LibreOffice headless render，game dev 用真实 engine run——每个 domain 都能构造一个 "runtime verifier agent"。这是 **verifiable reward 的通用化 pattern**。

### 4.3 Terminal-Gym：从 Stack Overflow 合成 Terminal 任务

Stack Overflow 是另一个天然 verifiable source——accepted answer 提供 ground truth。M2 把 SO post 转成结构化 task，自动生成 Docker env 和 test。

Stage 2 的 query evolution 是关键创新：把 task instruction 里的 explicit hints、file paths 抽象掉。为什么？因为如果不抽象，test 会 overfit 到 description style——model 学会"看 task description 里提到 `/var/log/nginx/access.log` 就去 grep 这个 path"，而不是真的理解任务逻辑。

抽象后，test 必须验证 **underlying logic** 而非 surface pattern。这把 task 从"按图索骥"升级到"理解意图"。

Stage 3 的 difficulty calibration 也很 important：保留 hint 少 + zero-shot pass rate 低的 variants。这保证最终训练集是"hard but solvable"——RL 最有效的 difficulty sweet spot。

paper 提到这个系统 evolving 成 **Anything2Docker**，目标是 zero-intervention 把任意 task 转成 Docker env。如果能做出来，这是 agent data pipeline 的 holy grail——任意 verifiable task 都能自动 wrap 成 RL training instance。

### 4.4 Agentic Cowork：Search / Office / Financial / Slide 四个 domain

四个 domain 共享同一 design pattern：**runnable workspace + rotating teacher distillation + artifact-aligned reward**。

Deep Search 那个 evidence specification 设计特别有意思。每个 task 显式声明需要哪些 evidence，trajectory 只有在 answer grounded in 实际 retrieved evidence 时才 accept。这防 model 学会"fabricate plausible-sounding answer"——这是 search agent 最常见的 failure mode。

Rotating teacher + scaffold perturbation 也是 important。如果只用一个 teacher、一个 scaffold，student 会 overfit 到 teacher style 和 scaffold layout。rotating 让 student 学到 task-invariant policy。

Financial Analysis 那个 evidence-driven synthesis 很 clever：**先跑真实 financial tool 收集 traces，反向 derive task**。这样 task 必然可执行可验证，因为 task 本身就是从 execution trace 反推出来的。这把"task authoring"和"task verification"统一成一件事。

Slide Generation 用 multiple slide-generation libraries 防 overfit 也是个细节亮点——单 toolkit 训练会让 model 变成"python-pptx wrapper"，多 toolkit 让 model 学到 slide 的 abstract 概念。

## 五、RL Algorithm：CISPO + Composite Reward

### 5.1 Agent RL 的 MDP 形式化

paper 把 LLM 当 policy，把 LLM 外的一切（context management、memory、agent state transition）当 environment。这个 boundary 划在 **model generation interface** 上。

这个 abstraction 看起来 trivial，但意义深远。传统 RL 里的 environment 是 fixed game engine，state transition 是 deterministic function。Agent RL 里 environment 包含 scaffold 代码、sub-agent、memory system——这些都可以变化、可以 learning、可以 non-stationary。

把 boundary 划在 generation interface 意味着：**policy gradient 只关心 $(s_t, a_t)$ atomic pair，不关心 state 怎么演化**。$s_t$ 是怎么来的（append message、aggressive truncation、complete rewrite）不影响 policy 的 update rule。

这让 RL framework 可以 support 任意复杂的 agent architecture——因为 environment dynamics 是 framework 的外部黑盒。这是 Forge 能 support white-box + black-box agent 的 algorithmic foundation。

### 5.2 CISPO 的 asymmetric clipping

CISPO 的核心是公式3的 asymmetric clip：
$$\hat{r}_{i,t}(\theta) = \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}, 0, 1+\epsilon_{\text{high}}^{\text{IS}}\right)$$

变量解释：
- $\pi_\theta$：当前 policy
- $\pi_{\theta_{\text{old}}}$：rollout 时的旧 policy
- $o_{i,t}$：trajectory $i$ 第 $t$ 个 token
- $q$：prompt
- $o_{i,<t}$：前 $t-1$ 个 token
- $\epsilon_{\text{high}}^{\text{IS}}$：上界 clip 参数

PPO 的 symmetric clip 是 $[1-\epsilon, 1+\epsilon]$，CISPO 是 $[0, 1+\epsilon]$。lower bound 从 $1-\epsilon$ 变成 0。

这个差异意味着什么？

PPO 的 $1-\epsilon$ lower bound 是"温和的 down-weight"——即使某 action 在当前 policy 下几乎不可能，它的 importance ratio 还是 $1-\epsilon$，gradient 还在反向 update policy 去"轻微地 forget"这个 action。

CISPO 的 0 lower bound 是"激进的 down-weight"——某 action 一旦变得不可能，importance ratio 归 0，gradient 直接 zero out，policy **立即停止学习这个 action**。

对 long-horizon agent 这个设计很重要。agent trajectory 几千 step，早期某些 action（比如错误的 tool call format）应该被快速 forget，而不是缓慢 decay。Symmetric clip 会让这些 bad action 的 gradient 持续污染 training。

Upper bound $1+\epsilon$ 防止 over-upweight——已经 gain 很多的 action 停止 update，避免 policy collapse 到单一 trajectory。

**Stop-gradient on clipped ratio** 是个很 subtle 的设计。传统 PPO clip 在 backward 时让 importance ratio 的梯度参与计算，引入 second-order term。CISPO 用 $sg(\cdot)$ 让 ratio 只起 **modulate gradient magnitude** 的作用，不引入二阶项。这让 update rule 是 first-order，更稳定，实现更简单。

我觉得 CISPO 反映了对 RL 训练稳定性的一个理解：**forget fast, reinforce slow**。这个哲学跟 agent 训练的实际需求匹配——agent trajectory 长、exploration 空间大，需要快速 prune bad policy，但 reinforce good policy 要谨慎防 collapse。

MiniMax-M1 的 CISPO 原始 paper: https://arxiv.org/abs/2506.13585

### 5.3 Composite Reward 的三个 component

公式7：
$$r_t = \alpha \cdot r_t^{\text{process}} + \beta \cdot r_t^{\text{speed}} + r_t^{\text{perf}}$$

- $r_t^{\text{process}}$：dense behavioral reward（penalty for language mixing / format error, reward for well-structured reasoning）
- $r_t^{\text{speed}}$：completion time reward
- $r_t^{\text{perf}}$：primary task performance（如 SWE 的 test pass）
- $\alpha, \beta$：coefficients

**Process reward** 解决 credit assignment 稀疏性问题。192K token trajectory 只有最终 outcome reward 的话，gradient signal 太稀疏。Process reward 在每个 $(s_t, a_t)$ 提供 fine-grained feedback。

**Speed reward**（公式5）解决一个被传统 RL 忽略的问题：**functionally equivalent trajectory 的 wall-clock latency 差几个量级**。Sequential tool execution vs parallel tool execution、sub-agent invocation overhead，这些都会让"功能等价"的 trajectory 实际 efficiency 差很多。传统 RL 只 optimize correctness，model 学不到 parallelism。

$$r_t^{\text{speed}} = h\left(\frac{T_{\text{completion}}}{T_{\text{baseline}}}\right)$$

- $h$：单调递减 shaping function
- $T_{\text{completion}}$：rollout wall-clock time
- $T_{\text{baseline}}$：参考时间

这个 reward 让 policy 发现 parallelism 机会，产出"既正确又高效"的 trajectory。

**Reward-to-Go with Baseline**（公式6）解决 long-horizon variance 问题：
$$G_t = \sum_{\tau=t}^T \gamma^{\tau-t} r_\tau$$

- $\gamma$：discount factor
- $r_\tau$：step $\tau$ 的 reward

把 gradient signal 集中到"consequences 还没被 accounted for"的 action 上，提高 credit assignment precision。

这三个 component 组合起来是个很 thoughtful 的设计：process reward 解决稀疏性，speed reward 解决 efficiency，reward-to-go 解决 variance。每个 component 都对应一个 long-horizon agent RL 的具体 failure mode。

### 5.4 Mixed-Domain RL：防 forgetting + 防 negative transfer

这个 section 我觉得是 RL 训练 practitioners 最容易踩坑的地方。

Single-domain RL（只在 agent task 上 fine-tune）会 catastrophic forgetting——model 把 reasoning、general knowledge 全忘掉。Sequential multi-stage（先 reasoning 再 coding 再 agent）会 negative transfer——后一 stage 的 gradient 把前一 stage 学到的 capability erode 掉。

M2 的解法：**每 stage 同时从四个 domain（reasoning / coding / agent / general）抽数据**，joint optimization。每 step policy gradient informed by diverse task distribution，optimizer 不 overfit 单 domain reward landscape。

这个策略的 intuition 是：**用 data mixing 代替 sequential training**。数据混合自然实现 multi-task regularization，类似 multi-task learning 的 hard parameter sharing。

Across stages 的三轴 curriculum：
1. Domain mixing ratio：early 重 reasoning+general，late 重 agent+coding
2. Context length：per-domain 扩展，先短后长
3. Difficulty distribution：每 domain 内逐步偏 hard

这种 curriculum 反映了一个 developmental view：**先建立 base competence，再 sharpen task-specific performance**。跟人类学习路径一致——先学基础数学再学算法竞赛。

## 六、Forge RL Infrastructure：The Impossible Triangle

### 6.1 三个互相矛盾的目标

paper 形式化了 RL infra 的"不可能三角"：
1. **System Throughput**：max tokens/sec
2. **Training Stability**：$\mathbb{E}[\text{Var}(\nabla_\theta J)] < \delta$
3. **Agent Flexibility**：support arbitrary $\mathcal{A} \in \Omega_{\text{agent}}$

为什么这三个互相 conflict？

**Throughput vs Stability**：agent rollout completion time 方差巨大（秒级到小时级）。Max throughput 要 greedy scheduling（谁完成就 fetch 谁），但这样早期 batch 都是 short easy task，hard task 集中后期，distribution shift 导致 gradient oscillation，stability 破坏。

**Throughput vs Flexibility**：Max throughput 要 tight coupling（agent state 跟 training logic 强绑定，减少 overhead），但 flexibility 要 support 任意 agent architecture，必须 loose coupling。

**Stability vs Flexibility**：Stable credit assignment 要长 trajectory 的 reward propagation，flexibility 要 support dynamic context management，但 dynamic context 会 break trajectory 的 temporal coherence，让 credit assignment 不稳定。

公式8：
$$\max_\theta J(\theta) = \text{Throughput}(\mathcal{A}) \times \text{SampleEfficiency}(\mathcal{A})$$
$$\text{s.t.} \quad \forall \mathcal{R} \in \Omega_{\text{agent}}, \mathbb{E}[\text{Var}(\nabla_\theta J)] < \delta, \mathbb{E}[\|J^{(T)} - J^*\|] < \epsilon$$

Throughput × Sample Efficiency = Effective Training Yield。两个 constraint 保证 stability 和 convergence。

这个 formulation 很 systems-thinking，把 RL infra 当成 multi-objective optimization problem。

### 6.2 Three Decoupled Modules

Forge 的解法是 **三个 decouple 的 module 通过 middleware 通信**：

1. **Agent Side**：纯 trajectory producer，agnostic to training/inference。执行 tool、管 context、访问 memory，记录 $(s_t, a_t, o_t)$。
2. **Middleware**：Gateway Server（标准化通信）+ Data Pool（distributed trajectory storage，async 收集）。
3. **Training/Inference Side**：Rollout Engine（高吞吐 generation）+ Train Engine（CISPO gradient + weight sync）。

关键：**generation 和 training pipeline 完全 decouple**，独立 scale。Data Pool async 收集让两边互不阻塞。

这个 architecture 让我想到 microservice 的 pattern。每个 module 独立部署、独立 scale、通过标准接口通信。比 monolithic RL framework（比如 Ray RLlib 的 tightly coupled design）更灵活。

**White-Box vs Black-Box Agent** 的统一也是个亮点：

- White-Box: 暴露 context management logic 给 framework，training 能 reconstruct exact training state。适合 well-defined CM（sliding window、periodic summarization）。
- Black-Box: opaque trajectory producer，framework 只看 externally visible $(s_t, a_t, o_t)$。适合 deep thinking loop、aggressive context rewriting、hierarchical multi-agent。

两种 paradigm 通过 Gateway-based abstraction 统一——white-box 只是多了 CM operations 的 registration。这个设计让 Forge 能 support 数百个不同 scaffold 而不需要 framework 改动。

这个 abstraction 的深远意义：**让 agent 架构可以 free evolve，不用每次改 scaffold 都重训 framework**。agent 团队可以快速迭代 scaffold，training team 用同一 framework 训。这是 organizational scaling 的 key。

### 6.3 Windowed FIFO Scheduling

这是我看完最拍案叫绝的 engineering 设计。

问题：agent rollout completion time 方差巨大。
- Strict FIFO：保 distribution consistency，但 head-of-line blocking——一个慢 task 卡住整个 queue
- Fully greedy：max throughput，但 distribution shift——早期 batch 全 short easy task

Windowed FIFO（Figure 5）：
- Generation queue $Q = [T_0, T_1, \dots, T_{N-1}]$，head index $i$
- Scheduler 只在 sliding window $[T_i, T_{i+W-1}]$ 内 fetch completed trajectories
- Window 内：greedy，缓解 HoL blocking
- Window 外：strict FIFO
- Window 只在 head-of-window task 被 consume 时 advance

$W = 0.3N$ 在实践中是 sweet spot——接近 FIFO 分布特性，cluster idle time 大幅减少。

这个 idea 让我想到数据库 query optimizer 的 **latency-bounded reordering**——允许小范围 reorder 提吞吐，但保整体 distribution 不偏。也像 MapReduce shuffle 的 sort-merge：小 buffer 内自由排序，整体保 order。

这是个典型的 **system design 中"almost FIFO"比"strict FIFO"好得多**的 pattern。Strict FIFO 看起来 clean，但 variance 杀死 throughput；greedy 看起来 efficient，但 variance 杀死 stability。中间地带的 sliding window 是工程上的最优解。

### 6.4 Prefix Tree Merging

这个设计直接把 training speedup 40×，我觉得是 paper 最 impressive 的工程数字之一。

Multi-turn agent trajectory 里，同一 rollout group 的 G 个 trajectory 共享 system prompt + early turns。传统 training 每个 sample 独立 recompute common prefix，long-context agent 下 early turns 几乎完全相同，重复计算严重浪费。

Prefix tree merging 把共享 prefix 的 completions 合并成 single prefix tree：
1. Forward pass：shared prefix 只算一次，到 branch 处分叉
2. Forward 后用 metadata deconstruct tree
3. Per-sample 独立计算 loss

关键保证：**数学上等价于 independent-sample training**。Causal attention 在 shared prefix 上产生 identical activations，无论算 1 次还是 N 次。Zero approximation error。

这本质是 **把 beam search 的 prefix sharing 思想用到 training forward pass**。RL 的 G 个 rollout 共享 prompt 是天然 prefix sharing 场景，特别是 long-context agent 下 early turns 几乎完全相同。

40× speedup 这个数字让我震惊。这意味着同样 hardware 可以训 40× 大的 batch、40× 长的 trajectory、40× 多的 rollout。对 long-horizon agent RL 这个 speedup 直接 unlock 之前不可行的 training scale。

我联想一下：这个 pattern 可以扩展到所有 "shared prefix" 场景。Few-shot learning 的 N-shot examples、chain-of-thought 的多个 reasoning path、Best-of-N sampling——所有这些都有 shared prefix，都能 benefit from tree merging。这是个 general technique。

### 6.5 Inference Acceleration 三个优化

**MTP-based Speculative Decoding with Co-training**：MTP module 在 RL training 中跟 policy co-train，通过 top-k KL divergence loss 保持 draft acceptance rate。这个 co-training 是 key——如果不 co-train，policy 在 RL 中演化，draft model 跟不上，speculative decoding 性能退化。

**Heterogeneous Prefill-Decode Disaggregation**：Prefill 和 decode 分到独立 instance，消除 MoE mixed scheduling 互相干扰。每 phase 用各自优化的 parallelism。这个 pattern 跟 vLLM、SGLang 最近的 prefill-decode disaggregation trend 一致。

**Global L3 KV Cache Pool**：distributed、DFS-backed global KV cache，cost-aware request router 平衡 queuing delay vs cache migration cost。最大化 prefix cache hit rate。

这三个优化组合起来，让 inference throughput 在 multi-turn agent 场景下最大化。Multi-turn agent 每 turn 共享前面所有 KV，没有 prefix cache 等于每 turn 重算 prefill，cost 爆炸。Global KV cache pool 解决这个问题。

## 七、Interleaved Thinking：为什么这个 simple idea 重要

### 7.1 Trajectory 结构

公式9：
$$\tau = (r_1, a_1, o_1, r_2, a_2, o_2, \dots, r_T, a_T, o_T)$$

- $r_t$：reasoning tokens（thinking block）
- $a_t$：action tokens（tool call）
- $o_t$：observation（tool output）

每个 $r_t$ conditioned on full history。

对比两个 alternative：
1. **Front-loaded reasoning**：所有 reasoning 在 actions 前。不能 adapt 到 intermediate observation。
2. **Stateless per-turn reasoning**：每 turn 把 $r_{<t}$ strip 掉。不能 build on earlier analysis。

### 7.2 Reasoning State Persistence

公式10：
$$\mathcal{H}_{t+1} = \mathcal{H}_t \oplus [\text{assistant}(r_t, a_t)] \oplus [\text{tool}(o_t)]$$

如果不 persist（strip thinking blocks）：
$$\mathcal{H}_{t+1}^{(\text{drop})} = \mathcal{H}_t \oplus [\text{assistant}(a_t)] \oplus [\text{tool}(o_t)]$$

每 turn 都要重新 derive context/constraints/partial conclusions，导致 cumulative state drift + degraded self-correction。

### 7.3 Plan-Act-Reflect Loop

每个 turn 内：
1. **Plan**：review accumulated state，formulate/refine strategy
2. **Act**：select & execute tool call
3. **Reflect**：evaluate observation vs expectation，update world model，决定 revise or proceed

这个 loop 让 model 能 self-correct through reflection——遇到 unexpected observation 能 revise plan，而不是一路按初始 plan 跑到底。

### 7.4 为什么这个 simple idea 重要

Interleaved thinking 本质是把 ReAct (Yao et al. 2022) 升级——不仅 Reason-Act 交替，而是 **reasoning state 完整 persist 到下一 turn 的 context**，让 model 可以"复盘自己之前的思考"。

Ablation 显示 strip thinking blocks 后，agentic benchmark 全面下降，deep search 和 SWE 任务下降最大。这证明 sustained planning + iterative refinement 的任务最依赖 reasoning state persistence。

这个 insight 跟 DeepMind、Anthropic 最近的工作一致——thinking tokens 是 first-class state，不能当一次性消耗品。Anthropic 的 Claude extended thinking、DeepMind 的 Gemini thinking 都是类似 design philosophy。

我觉得这里有个更深的 observation：**long-horizon agent 的 bottleneck 不是单步 reasoning 能力，是跨步 reasoning state 的累积和 revision**。Base model 的单步 reasoning 已经很强（AIME 94 分），但如果不 persist thinking state，跨步 planning 就会 drift。Interleaved thinking + state persistence 解决的是这个跨步 coherence 问题。

ReAct 原 paper: https://arxiv.org/abs/2210.03629

## 八、Self-Evolution：M2.7 的最 future-facing 部分

### 8.1 Model Iteration System

paper Section 7.2 描述了一个让我震撼的系统。

Principle："humans steer while models build"。
- Researcher 配置 goals、通过 chat guide agent、review outputs
- Agent 在 "Agent Harness" 内运行——**harness 完全由内部 M2.7 生成，zero human-written code**
- Harness 配备 hierarchical skills：action chaining、persistent memory、safety guardrails、evaluation infrastructure

"zero human-written code" 这个细节让我震了一下。Agent 的 scaffold 自己写自己——这是真正的 self-referential system。

### 8.2 Dual-Loop Workflow

- Human-led experiment planning
- M2.7 autonomous execution：profile ongoing runs、read logs、diagnose anomalies
- Auto-debug code + 调整 configurations
- 直接 intervene 自己的训练 loop
- **吸收 30-50% 的 daily iteration workload**
- Human review 触发 major iteration decisions
- Agent 可 auto-continue bounded analysis between reviews

30-50% 的 daily iteration workload 被 agent 接管——这是非常具体的 productivity gain 数字。

### 8.3 Recursive Scaffold Upgrades

M2.7 被给定"优化内部 programming scaffold"任务，跑了 **100-round 全自动 iteration cycle**：
- Analyze failures
- Modify code
- Evaluate changes

引入了 loop detection 机制，发现更好参数组合，**in-house evaluation 上 30% 性能提升**。

### 8.4 这意味着什么

我觉得这个 section 是 paper 最 future-facing 的部分，但也是最难评估的部分。

一方面，**这是 self-improvement 的 early operational form**。Model 修改自己 scaffold、debug 自己训练——这是 ELK、AlphaGo 之外的另一条 self-improvement path。AlphaGo 是 game self-play，这里是 ML engineering self-improve。

另一方面，paper 没详细说 **safety boundary**。Model 修改自己 scaffold 可能引入 subtle bug、reward hacking、甚至 safety regression。100-round autonomous iteration 如果没有 robust evaluation harness，可能 evolve 到 bad local optimum。paper 说有 "safety guardrails" 但没展开。

我大胆推测这个 direction 的下一阶段：
1. **Recursive depth 增加**：M2.7 是 100-round，下一代可能 1000-round、10000-round
2. **Self-modification scope 扩大**：从 scaffold 扩展到 training recipe、reward function、architecture search
3. **Multi-agent self-evolution**：多个 model 互相 review、互相 improve，类似学术界的 peer review

这个 direction 如果 scale 起来，frontier model development 的 human-in-the-loop bottleneck 会被大幅 compress。RL 团队的工作从"写 training code、调 hyperparameter"变成"set goal、review outcome、set next goal"。

MLE-bench: https://arxiv.org/abs/2410.13276

## 九、Evaluation 的几个 highlight

### 9.1 Agentic Coding 上跟 frontier 持平

| Benchmark | M2.7 | Opus 4.6 | GPT 5.4 | Gemini 3.1 Pro |
|---|---|---|---|---|
| SWE-bench Pro | 56.2 | 57.2 | 54.2 | – |
| SWE-bench Multilingual | 76.5 | 77.8 | 70.5 | – |
| Multi-SWE-bench | 52.7 | 50.3 | 49.0 | – |
| Terminal-Bench 2.0 | 57.0 | 65.4 | 75.1 | 68.5 |

M2.7 在 Multi-SWE-bench 上第一，SWE-bench Pro/Multilingual 接近 Claude Opus。但 Terminal-Bench 2.0 跟 GPT 5.4 差距大（57.0 vs 75.1）——说明 terminal operation 这种长 horizon、多 step 的 task，GPT 5.4 还是有优势。

### 9.2 NL2Repo 上 M2.5 → M2.7 跳了 13 分

39.8 vs 26.6——直接印证 §4.1 引入的 full-stack repository data 起作用。这是 **data pipeline 决定 capability frontier** 的直接证据。

### 9.3 Office Task 是 M2.5 → M2.7 增长最大 block

GDPval-AA +15.0、MEWC v2 +13.5、Finance Modeling Pro +23.2。这些都是 §4.2 新引入的 task family。再次印证 data pipeline thesis。

### 9.4 Reasoning 上 AIME 2026 94.2 超 Claude

这个数字 impressive，94.2 超过 Opus 4.6 (92.5) 和 Sonnet 4.6 (92.7)。9.8B activated parameters 在数学竞赛上打 frontier，说明 base model + RL recipe 已经成熟到可以用 mini-activation 达到 reasoning frontier。

### 9.5 Within-Series Progression 的 Pattern

Figure 9 显示 M2 → M2.5 → M2.7 在 11 个 benchmark 上 all improve。引入新 task family 的 benchmark 涨得最猛：
- BrowseComp +33.8
- Wide Search +12.9
- Toolathlon +27.5
- GDPval-AA +16.0
- MLE Bench Lite +26.6

原本就强的 benchmark incremental：
- SWE-bench Multilingual
- Multi-SWE-bench

Reasoning benchmark 稳步上升：
- AIME 2025 +16.0
- GPQA-Diamond +11.8
- AA-LCR +11.0

**Pattern：数据 pipeline 引入新 task family 直接对应 capability jump，model 规模不变**。

这是 agentic post-training era 的核心 insight：**不是 base model 决定 capability frontier，是 data pipeline 决定 capability frontier**。Base model 决定的是"能学会什么"，data pipeline 决定的是"实际学到什么"。

### 9.6 MLE Bench Lite 的 Self-Evolution Case Study

M2.7 在 MLE Bench Lite 上 66.6% medal rate，tie Gemini 3.1 Pro。

实验设置：
- 22 competitions
- 每 run 24 小时 iterative evolution
- Simple autonomous harness（short-term memory + self-feedback，**harness 内 zero human-written code**）
- 每 iteration 后 model 写 memory file + rigorous self-criticism

Best run = 9 gold + 5 silver + 1 bronze。

Figure 10 显示 medal rate 随 trial 累积上升——证明 model 能 build on accumulated feedback chain，而不是 plateau。

这个 case study 是 self-evolution 的 direct evidence。Model 不只是完成 ML task，是 **iteratively improve 自己完成 ML task 的能力**。

## 十、我的 Takeaways 和 Meta-Level 观察

### 10.1 这篇 paper 真正的贡献

不是 base model。229.9B/9.8B MoE 是 incremental 改进，不是革命。Full attention 放弃 hybrid 是务实工程，不是 algorithm 突破。

真正贡献是 **agentic post-training stack 的工程化方法论**：

1. **Verifiable reward engineering**：每个 domain 设计 domain-specific 但 automated 的 verifiable reward。SWE 的 F2P/P2P、AppDev 的 AaaV、Terminal-Gym 的 unified test、Deep Search 的 evidence specification。这是 RL scaling 的 prerequisite。

2. **RL infrastructure 的 decoupled architecture**：Forge 通过 three-module decoupling + windowed FIFO + prefix tree merging 同时 achieve throughput / stability / flexibility。这是 systems engineering 优雅。

3. **Self-evolution 的 early operational form**：M2.7 能 debug 自己训练、改自己 scaffold、100-round 自主迭代。这是 model-on-model recursion 的早期实现。

### 10.2 Mini Activations 的真正意义

"Mini activations unleash max real-world intelligence" 不是"小模型也能打"，是 **per-token compute 的节省可以 redirect 到 long-horizon agentic rollout 的经济可行性**。

更深层：**self-evolution** 让 model 用自己 capability 改进自己训练。这个 loop 每一步都需要大量 agent rollout。如果 per-token cost 是 100B activated，这个 loop 经济不可行；mini activations 让 self-evolution loop 经济可行。

这是 **compute footprint → trajectory count → self-evolution economic feasibility** 的传导链。

### 10.3 Post-Training Era 的核心矛盾

Base model era（GPT-3 到 GPT-4）拼的是 pretraining scale。Post-training era（o1 之后）拼的是 RL recipe。Agentic era（2025+）拼的是 **verifiable reward + RL infra + self-evolution**。

这三件事的难度递增：
- Verifiable reward：需要 domain-specific engineering，每个 task family 都要构造 verifier
- RL infra：需要 support long-horizon trajectory、arbitrary agent architecture、stable training
- Self-evolution：需要 model 能力达到能 debug 自己 training 的水平，且需要 safety boundary

MiniMax 这篇 paper 是三件事都做到 production-grade 的早期 example。DeepSeek 在 reasoning RL 上做了类似事（R1），MiniMax 在 agentic RL 上做了对应工作。

### 10.4 跟其他 lab 的对比

**vs OpenAI**：OpenAI 的 o1/o3 focus on reasoning RL，GPT-5 开始往 agent 走。MiniMax 的 M2 是直接从 agent 切入，跳过 pure reasoning RL 的中间阶段。这个 path 可能更 efficient——agent trajectory 自然包含 reasoning，agent RL subsume reasoning RL。

**vs Anthropic**：Claude Opus 4.6 在 agentic coding 上还是 frontier，但 Anthropic 没公开他们的 post-training stack。MiniMax 这篇是 agentic post-training stack 的公开 example，可能反映 Claude 背后的类似 design。

**vs DeepSeek**：DeepSeek R1 是 reasoning RL 的 open example，MiniMax M2 是 agentic RL 的 open example。两个 lab 都在 push RL 的边界，但 focus 不同。DeepSeek 的 fine-grained MoE + MTP 被 MiniMax 直接借鉴——这种 open 研究生态的相互 learning 很 healthy。

**vs Google**：Gemini 3.1 Pro 在 long-context + multi-modal 上是 frontier，但在 agentic coding benchmark 上没出现在所有 row。Google 的 infra 优势（TPU、own datacenter）让他们能 scale dense model，但 mini-activation MoE 这条 path 没走。

### 10.5 对未来的 predictions

基于这篇 paper 我大胆预测：

1. **Anything2Docker 会成为 agent data pipeline 的标准组件**。任意 verifiable task 自动 wrap 成 RL training instance 是 data engineering 的 holy grail。

2. **Prefix Tree Merging 会成为 long-context training 的 standard optimization**。40× speedup 太香了，所有 long-context RL training 都会 adopt。

3. **Self-evolution 会成为 frontier model 团队的 standard practice**。30-50% daily iteration workload 被 agent 接管是 too good to ignore。下一阶段是 recursive depth 增加 + scope 扩大。

4. **Mini-activation MoE 会成为 agent model 的 default architecture**。Dense model 在 long-horizon agent rollout 的 cost 不可承受，MoE 的 compute-capacity 解耦是必须。

5. **Verifiable reward engineering 会分化成独立 discipline**。每个 domain 都需要构造 verifier，这会催生专门的"reward engineering"角色，类似 MLE 之于 ML。

### 10.6 几个值得深挖的 open question

1. **Agent-as-a-Verifier 的 scaling law**：AaaV 提供 ground-truth reward，但 AaaV 自身的 reliability 怎么 scale？如果 AaaV agent 自己有 bias，训出来的 student 会不会 inherit bias？这是 self-distillation 的经典问题在 agent 时代的新形式。

2. **Self-evolution 的 safety boundary**：100-round autonomous iteration 引入了 loop detection，但 paper 没详细说怎么防止 model 学到 reward hacking 的 scaffold 修改。这是 alignment 的 hard problem。

3. **Prefix Tree Merging 的极限**：40× speedup 是 shared prefix 比例高的情况。如果 trajectory 之间 prefix sharing 少（比如 diverse prompt），speedup 退化多少？这个 technique 的 applicability 边界在哪里？

4. **Interleaved Thinking 的 context cost**：persist thinking state 让 model 能 plan across turns，但 thinking tokens 占 context window。192K context 下 thinking state 会不会 saturate？需要 dynamic thinking state compression 吗？

## 十一、参考链接大汇总

### Architecture & Algorithm
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- MTP (Gloeckle): https://arxiv.org/abs/2404.19737
- GQA: https://arxiv.org/abs/2305.13245
- RoPE: https://arxiv.org/abs/2104.09864
- Auxiliary-loss-free load balancing: https://arxiv.org/abs/2408.15664
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Shazeer MoE 2017: https://arxiv.org/abs/1701.06538
- MiniMax-Text-01: https://arxiv.org/abs/2501.01464
- MiniMax-M1 (CISPO): https://arxiv.org/abs/2506.13585

### Benchmarks
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-bench Multilingual: https://arxiv.org/abs/2508.02153
- SWE-Smith: https://arxiv.org/abs/2412.21132
- Terminal-Bench: https://arxiv.org/abs/2502.08200
- MLE-bench: https://arxiv.org/abs/2410.13276
- BrowseComp: https://arxiv.org/abs/2504.13716
- GDPval: https://arxiv.org/abs/2503.12631
- GAIA: https://arxiv.org/abs/2310.02357
- GPQA-Diamond: https://arxiv.org/abs/2311.12022
- HLE: https://arxiv.org/abs/2501.04909
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- IFBench: https://arxiv.org/abs/2506.10406
- Toolathlon: https://arxiv.org/abs/2510.01257
- RULER: https://arxiv.org/abs/2404.06654
- HELMET: https://arxiv.org/abs/2410.18976
- SciCode: https://arxiv.org/abs/2410.03784
- KorBench: https://arxiv.org/abs/2507.13139
- CVE-Factory: https://arxiv.org/abs/2603.11076

### Agent Systems
- WebExplorer: https://arxiv.org/abs/2509.06501
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- Induction Heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- Longformer (SWA): https://arxiv.org/abs/2004.05150
- ReAct: https://arxiv.org/abs/2210.03629

---

最后总结一下我对这篇 paper 的 overall take：

MiniMax-M2 series 的核心 thesis 是用 **mini-activation MoE 的 compute efficiency** 换取 **long-horizon agentic RL 的 economic feasibility**。围绕这个 thesis，他们 build 了 verifiable reward factory（domain-specific data pipeline）、Forge RL infra（decoupled architecture + windowed FIFO + prefix tree merging）、interleaved thinking（reasoning state persistence）、self-evolution（model-on-model recursion）。

这篇 paper 的价值不在某个 algorithm，在 **整套 agentic post-training 的工程化方法论**。它是 base model era 结束后 frontier model 团队真正在拼的东西的公开 example。DeepSeek 开源了 reasoning RL 的 playbook，MiniMax 这篇是 agentic RL 的 playbook。

看完最大的感受：**我们正在进入 model development 的 self-loop 时代**。Model 越强，越能 generate 自己的训练数据、debug 自己的训练、修改自己的 scaffold。这个 loop 的 speed 取决于 per-token compute cost——mini activations 让这个 loop 经济可行。这是 compute × algorithm × self-reference 的三重复合，可能是通往 AGI 的最 concrete path。

---

# MiniMax-M2 Series: Mini Activations, Max Real-World Intelligence 深度技术解析

## 0. 一句话定位

MiniMax-M2 series 的核心 thesis 是：用 **~10B activated parameters**（229.9B total，9.8B activated）的 sparse MoE，配合 agent-native 的 data pipeline + RL system + self-evolution loop，逼近 frontier 闭源模型（Claude Opus 4.6、GPT 5.4、Gemini 3.1 Pro）在 agentic coding、cowork、reasoning 上的表现。这不是又一个 base model paper，而是一篇 **agentic post-training stack 的工程论文**。

参考链接：
- MiniMax-M2 paper (Hugging Face): https://huggingface.co/papers
- MiniMax 官方: https://www.minimaxi.com
- DeepSeekMoE (fine-grained experts 灵感来源): https://arxiv.org/abs/2401.06066
- DeepSeek-V3 (MTP 设计来源): https://arxiv.org/abs/2412.19437
- MTP 原始论文 (Gloeckle et al.): https://arxiv.org/abs/2404.19737
- GQA (Ainslie et al.): https://arxiv.org/abs/2305.13245
- RoPE (Su et al.): https://arxiv.org/abs/2104.09864
- SWE-bench: https://www.swebench.com/
- BrowseComp: https://arxiv.org/abs/2504.13716
- GDPval: https://arxiv.org/abs/2503.12631
- MLE-bench: https://arxiv.org/abs/2410.13276
- Terminal-Bench: https://arxiv.org/abs/2502.08200
- CISPO (MiniMax-M1): https://arxiv.org/abs/2506.13585

---

## 1. 设计哲学与整体架构

### 1.1 为什么选 mini-activation MoE

paper 反复强调一个 trade-off：agentic 任务需要 **ultra-long context**（192K）+ **deep multi-step reasoning**，这天然带来两个 cost bottleneck：
1. Training/inference 的 FLOPs 随激活参数线性增长
2. Long-horizon rollout 让 RL 的 wall-clock 极度膨胀

MoE 的本质就是把 **capacity（total params）** 和 **compute per token（activated params）** 解耦。M2 把这个 ratio 推到 ~23:1（229.9B / 9.8B），相当于让模型拥有"大模型的知识储备"却只用"小模型的算力"。

Intuition：这就像一个公司雇了 230 个专家但每次开会只请 10 个相关的人出席——开会成本（inference）低，但每次都能从 230 人的 pool 里挑出最合适的组合。

### 1.2 Architecture 规格

| 项目 | M2 规格 |
|---|---|
| Total params | 229.9B |
| Activated params | 9.8B |
| Layers | 62 |
| Hidden dim | 3,072 |
| Vocab | 200,064 |
| Context window | 192K |
| Pre-training tokens | 29.2T (19.9T constant + 9.3T decay) |
| Attention | Full MHA + GQA (48 query heads, 8 KV heads) |
| Position encoding | RoPE |
| MoE | 256 fine-grained experts, top-8 activated |
| Gating | Sigmoid + per-expert bias |
| MTP | k=1 pre-training → k=3 continued pre-training |
| Pre-training loss weight for MTP | 0.3 → annealed to 0.1 |

**关键设计选择**：
- **放弃 hybrid attention**：之前 MiniMax-Text-01 用 Lightning Attention + full attention 混合，M2 全部用 full attention。理由是 hybrid 在 standard benchmark 上"看起来一样"，但在 complex multi-hop reasoning 上有 clear deficit，且这个 deficit 在 scale up 后才显现。
- **放弃 SWA**：Table 2/3 显示 SWA 在 RULER 128K、MTOB translation、SWE-verified、BrowseComp-zh 等长上下文任务上显著退化（RULER 128K CWE 从 90.0 → 72.0），但 32K 内的 instruction-following 上甚至更好。结论是 SWA 的 coverage 限制对 long-context 致命。

---

## 2. MoE 设计细节：Fine-Grained + Sigmoid Gating + Expert Bias

### 2.1 Fine-Grained Experts

把传统的"32 个大 expert, top-2"改成"128/256 个小 expert, top-8"（参考 DeepSeekMoE, Dai et al. 2024）。

Intuition：这把 routing 从"32 选 2"的离散决策变成"256 选 8"的组合空间，routing 的 combinatorial diversity 指数级上升。同时每个 expert 更小，单个 expert 的 utilization variance 在 multi-GPU 分片下更小（负载更均匀）。

Table 1 的 ablation（500B tokens, 17.8B total / 2B activated 的小模型）：

| Benchmark | Baseline (32 experts, top-2) | w/ MTP | w/ Fine-Grained (128 experts, top-8) |
|---|---|---|---|
| MATH | 19.6 | 21.3 | **24.1** |
| MMLU | 39.8 | 39.7 | **40.2** |
| ARC-Challenge | 27.4 | 27.5 | **27.8** |
| KorBench | 14.1 | **15.0** | 14.8 |
| HumanEval | 29.7 | 30.1 | **32.5** |

Fine-Grained 在 reasoning-heavy 任务上提升最明显（MATH +4.5, HumanEval +2.8）。

### 2.2 Sigmoid Gating（而非 softmax top-k）

传统 Shazeer 2017 的 MoE gating 是 softmax over experts 后取 top-k。这有 zero-sum 约束：一个 expert 的概率上升必然压低其他 expert。

M2 改用 sigmoid：每个 expert 独立计算 activation score：
$$g_e(x) = \sigma(w_e \cdot x + b_e)$$

其中 $w_e$ 是 expert-specific 的 routing weight，$b_e$ 是 learnable bias term。

Intuition：sigmoid 让"多个 expert 可以同时 high confidence"。例如：一个 token 可能同时触发"Python expert"和"sorting algorithm expert"，两个都高置信激活，而不是被迫二选一。这导致更 smooth 的 routing dynamics。

### 2.3 Expert Bias 隐式负载均衡

传统 MoE 用 auxiliary load-balancing loss（如 DeepSeek 之前的 batch-wise assignment loss）来防止 expert collapse。M2 引入 per-expert learnable bias $b_e$：
- 如果某 expert 被过度使用，optimizer 会自动降低它的 bias
- 反之，under-utilized expert 的 bias 上升

这把 load balancing 从 explicit auxiliary loss 转化为 implicit parameter optimization，paper 说 "allows the auxiliary load-balancing loss to be greatly reduced"。

Intuition：这就像市场机制——热门 expert 的"价格"（bias）自动上涨，把流量挤给冷门 expert。比硬性的 loss 约束更平滑。

参考：Auxiliary-loss-free load balancing (Wang et al. 2024a) https://arxiv.org/abs/2408.15664

---

## 3. Multi-Token Prediction (MTP) 模块

### 3.1 训练时：k=1 的辅助 head

参考 DeepSeek-V3 和 Gloeckle et al. 2024。pre-training 阶段加一个 MTP module（k=1，即预测下一个 token 之外再预测一个），loss weight 从 0.3 退火到 0.1。

MTP module 的架构（Figure 2）：把 main model 的 last hidden state 作为输入，经过一个独立的 transformer block + projection head，预测下一个 token。

训练 loss 形式化：
$$\mathcal{L}_{MTP} = -\sum_t \log p_{MTP}(x_{t+2} | x_{\leq t+1}, h_{t+1})$$

其中 $h_{t+1}$ 是 main model 在 $t+1$ 位置的 hidden state。

### 3.2 Continued Pre-Training：Weight Copying 扩展到 k=3

为了 inference 时支持 multi-step speculative decoding，M2 在 decay phase 把 MTP module 从 1 个扩展到 3 个（k=3）。

关键 trick：**copy initialization**。把 main model 的权重复制到新的 MTP modules，而不是随机初始化。理由：
1. 随机初始化的 MTP module 起始 loss 很高，会"污染"main model 的 representation
2. Copy init 让新 module 快速收敛，且 minimally disrupt main model

训练 schedule：
- Phase 1: Freeze main model，只训 MTP modules，直到 loss 稳定
- Phase 2: Joint training（main model + MTP modules 一起训）

paper 还尝试了 "main model 一直 frozen" 的纯 MTP-only schedule，发现 MTP 收敛到 worse final quality——说明 main model 和 MTP 需要 co-evolve。

### 3.3 Inference：Speculative Decoding Draft Path

inference 时 3 个 MTP modules 串行生成 draft tokens，main model 在单次 forward pass 里 verify（参考 Leviathan et al. 2023, https://arxiv.org/abs/2211.17192）。

直觉：这相当于 main model 带着 3 个"小弟"——小弟们基于 main model 的 hidden state 快速猜下一几个 token，main model 一次 forward 批量验证。throughput 提升，output quality 不变（因为是 main model 自己 verify）。

---

## 4. Post-Training Data Pipeline：Agent-Native 的核心

这是 paper 最重的部分。M2 series 把 post-training data 看成 **agent data pipeline** 而非传统 SFT/RLHF 数据集。每个 task 都 ground 在 executable workspace + verifiable reward。

### 4.1 Agentic Coding

#### 4.1.1 SWE-Scaling Pipeline（从 GitHub PR 反向构造）

这是我最欣赏的设计。直接用 GitHub PR 训 SWE agent 有三大问题：
1. Task diversity 不够（大部分 PR 是琐碎的 typo fix）
2. Verifiability 弱（很多 PR 没有测试）
3. Volume 不够大规模 RL

M2 的 6-stage pipeline：

**Stage 1: PR Collection & Filtering**
- 爬 permissive license 的 GitHub repos
- 只保留"已 merge"的 PR
- 规则过滤：必须有 relevant test cases

**Stage 2: Agent-Synthesized Multi-Language Docker Environments**
关键创新：非 Python 语言的 Docker environment synthesis 不可靠（heterogeneous dependencies、version conflicts）。M2 用 **agent-driven execution loop**——让 LLM agent 迭代生成 build script，用 execution feedback 引导修复。

三个维度挑战：
- Build system orchestration（Java/Go/Rust/C++ 的 toolchain）
- Heterogeneous testing interfaces
- Repository-level structural variability

**Stage 3: PR Tagging & Task Diversification**
给每个 PR 打标签路由：bug fix / feature / perf / refactor / test construction。不同类型用不同的 reward function。

**Stage 4: Test-Based Verifiable Reward Construction**
- **Bug fix**：提取 F2P（Fail-to-Pass）和 P2P（Pass-to-Pass）测试。golden patch 必须通过两者。P2P 尤其关键——确保 fix 不引入新 bug。
- **Feature addition**：F2P/P2P 不适用，因为新代码引入新测试。focus 在 newly added test points。
- **Performance optimization**：没有 bug fix 过程，提取能验证稳定性能差异的 P2P 测试。

**Stage 5: Model-Based Task Validation**
raw PR 的 description 往往 under-specified。用 LLM 验证 problem description 和 test cases 的一致性，并 enrich 缺失信息。

**Stage 6: Task Transformations & Augmentation**
- Bug injection：往 codebase 注入额外 bug，增加 task 难度分布
- Commit merging：相邻 commits 合并成 multi-step 修复任务（类似 SWE-Smith, Yang et al. 2024, https://arxiv.org/abs/2412.21132）
- SWE-Test conversion：把 bug-fix PR 反转——agent 要写一个在 pre-patch fail、post-patch pass 的测试
- Code review tasks：静态分析任务，不需要 runnable env，由 secondary LLM 验证

最终产出：每个 instance = (problem statement, test-based reward, runnable Docker env)。覆盖 10+ 编程语言。

#### 4.1.2 AppDev Pipeline（Expert-in-the-Loop + Agent-as-a-Verifier）

AppDev 是从 0 构建完整应用，挑战：
1. 不能从现有 codebase 抽取
2. 质量评估不能只看 code，要看 runtime
3. 评判标准包含 functional correctness + subjective design

**Expert-in-the-Loop Query Synthesis**
- Domain expert 设计 meta query（template），specifying framework（e.g., React + Zustand + Tailwind）、architectural constraints、realistic use cases
- 自动 sample 不同 tech stack / styling / functional requirements 组合
- MinHash deduplication
- LLM-as-judge 三维评分：Tech Stack Rationality / Feature Feasibility / Requirement Clarity

**Trajectory Sampling with Expert System Prompts**
- Expert 写 system prompt 编码 best practices（比如"不要过度使用 gradient background"）
- **Prompt distillation**：sampling 时给 full system prompt，training 时部分 drop——强迫 model 把 expert 知识 internalize 为 default behavior，减少 inference 时对 explicit prompt 的依赖

Intuition：这本质是 **behavioral cloning with knowledge distillation**——把 expert 的 implicit knowledge 通过 system prompt 显式注入，然后训练让 model 在没有 prompt 时也表现一致。

**Agent-as-a-Verifier (AaaV)**：这是 paper 的关键创新之一。传统 LLM-as-judge 看 static code 或 screenshot 评分。AaaV 把 generated app **部署到 sandbox**，让 verifier agent 用 Playwright **实际交互**：

三层 evaluation：
1. **Execution Layer**：file 存在 + syntax 有效 + dependency resolution + build 成功 + HTTP status + 无 JS error。失败立即 reject。
2. **Interaction Layer**：用 Playwright 检查 interactive element 存在、按钮响应、端到端 workflow 完成。
3. **Visual Aesthetics Layer**：layout professionalism、visual hierarchy、color harmony、modern UI standards。

总 pass rate 作为 rejection sampling 的 reward signal。

Intuition：这把"代码质量评估"从"看代码"升级到"用代码"。verifier 不读 code，直接 deploy + click，跟真人 QA 一样。这种 ground-truth signal 是 RL 训练最缺的。

#### 4.1.3 Terminal-Gym：从 Stack Overflow 自动合成 Terminal 任务

Terminal-Bench 这类任务需要 LLM 在真实 terminal 环境调试、配置系统。Terminal-Gym 是数据合成 pipeline：

1. **Seed Dataset**：完整 Stack Overflow dataset，按时间排序重建完整 posts，规则过滤（无 accepted answer、低分、过长都丢弃）。按 tag 筛 terminal-relevant threads。每个 post 标注 problem quality、task type、verifiability、复杂度、env requirements。
2. **Query Synthesis**：把 SO question rewrite 成结构化 task description，包含 execution context、tools、I/O format、success criteria。按 testability/completeness/clarity 分 4 tier，保留 top-2。
3. **Three-Stage Synthesis**：
   - Stage 1: Agent 生成 Dockerfile + test script，test 失败则 agent 迭代修复
   - Stage 2: Query evolution——把 task instruction 里的 explicit hints、file paths 抽象掉，强迫 test 验证 underlying logic 而非 description style overfitting
   - Stage 3: Difficulty calibration——保留 hint 少、zero-shot pass rate 低的 variants

paper 还提到 evolving 成 **Anything2Docker** zero-intervention 系统，以及扩展 **CVE-Factory** (Luo et al. 2026) 做自主 cybersecurity 研究。CVE-Factory: https://arxiv.org/abs/2603.11076

### 4.2 Agentic Cowork

四个 domain 共享同一 design pattern：runnable workspace + distilled trajectories from rotating teacher models + artifact-aligned reward。

#### 4.2.1 Deep Search
- Guide-and-rewrite synthesis：seed question → 迭代 rewrite + obscure entities → 直到难度能 discriminate strong/weak agents
- **Evidence specification**：每个 task 显式声明需要哪些 evidence，trajectory 只有在 answer grounded in 实际 retrieved evidence 时才 accept。防止 model 学会 fabricate plausible-sounding answers。
- Report-style query 用 rubric-based judge（factual accuracy、transparency、uncertainty handling、risk disclosure）
- Scaffold perturbation across runs，policy 不 overfit 单一 tool layout

#### 4.2.2 Knowledge-Worker Office Tasks
- Anchor 到 GDPval benchmark (Patwardhan et al. 2025, https://arxiv.org/abs/2503.12631)
- Hierarchical synthesis：occupational categories → fine-grained subdivisions（含文化/地域多样性）→ concrete tasks + workspace + multi-granularity query versions
- Multi-axis rubric：positive/negative behaviors、critical errors、regional appropriateness、depth of reasoning
- Typed cleanup：移除 fabricated data/references/entities

#### 4.2.3 Financial Analysis & Spreadsheet
两族任务：
- **Evidence-Driven Synthesis**：先执行真实 financial tools 收集 traces，反向 derive task。grounding by construction——每个 task 都必然可执行可验证。
- **Workbook-Walk Synthesis**：agent 在 seed workbook 上跑 atomic spreadsheet operations，中间 states 作为新 seeds，反序合成 task。

Acceptance：deterministic value-level match（cell value 对比 ground truth workbook）；form 自由的用 rubric/agent-based judging。

#### 4.2.4 Slide Generation
两 stream：
- Open-ended generation：diverse source documents → vary description granularity/length/register
- Localized editing：real decks 作为 seeds，按 granularity（element → page → document）/ intent（content/style/structure）/ complexity 三轴 diversity

Acceptance 多 signal：execution success + functional correctness (agent) + rule-based layout aesthetics + visual scorer (render 后作为 image 评判)。混入多个 slide-generation libraries 防止 overfit。

### 4.3 Reasoning-Intensive

三轴 scaling：
- **Query-Side**：扩展 unique problems，尤其 underrepresented 难度段
- **Response-Side**：每个 query 生成多个 correct solution paths。OOD generalization 随 responses per query 增加而持续提升——证明 diverse solution paths 教 transferable reasoning strategies 而非 memorization
- **Training-Side**：研究 query expansion vs response expansion 的最优 mixture ratio，per-stage 动态调整

QA pipeline：multi-stage cleaning、boundary case coverage、cross-model agreement check、rubric-based scoring。

### 4.4 Role-Play

形式化为 long-horizon conditional generation over {Worlds} × {Stories} conditioned on {User Preferences}。核心目标是 maintain physical/narrative/stylistic coherence。

关键 insight："misalignment is objectively detectable while alignment is subjective"——所以 Role-Play Bench penalize specific failure modes（OOC break、logic error），而不是给 positive 分数。

Self-play 生成 + dispersion sampling（四轴）+ Best-of-N + segment-level rewriting。RLHF + causal inference 去 bias + entropy monitoring 防 reward hacking。

---

## 5. RL Algorithm：CISPO + Composite Reward

### 5.1 Agent RL 形式化

把 LLM 视为 policy $\pi_\theta$，model 外的一切（context management、memory、agent state transition）视为 environment。这个 boundary 划在 **model's generation interface**——所有 processing/transform/respond 模型输出的 component 都是 environment dynamics。

MDP：$M = (S, A, T, R, \gamma)$
- $s_t \in S$：当前 context window content（task instruction + history + tool outputs + artifacts）
- $a_t \in A$：单步 LLM completion（可含 NL reasoning、tool invocation、context management op、sub-agent communication，任意组合）
- $o_t$：observation（tool 执行后返回）
- 状态转移：$s_{t+1} = f_{\text{trans}}(s_t, a_t, o_t)$（公式1）

$f_{\text{trans}}$ 可以是任意复杂函数——append message、aggressive context truncation、complete history rewrite 都行。policy 不需要知道 $s_t$ 是怎么来的。

Intuition：这相当于把 RL 的"环境"从"游戏物理引擎"扩展到"整个 agent scaffolding"——agent 内部的 memory、planner、sub-agent dispatch 都是 environment 的一部分。policy gradient 只关心 $(s_t, a_t)$ atomic pair，不关心 state 是怎么演化的。

### 5.2 CISPO Objective

MiniMax 自家的 algorithm（M1 paper, https://arxiv.org/abs/2506.13585），adapt 到 M2。

公式2：
$$J_{\text{CISPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} sg(\hat{r}_{i,t}(\theta)) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} | q, o_{i,<t}) \right]$$

变量解释：
- $q$：prompt（task instruction）
- $a$：参考答案（用于 reward）
- $G$：每个 prompt 采样的 rollout trajectory 数
- $|o_i|$：trajectory $i$ 的 token 长度
- $o_{i,t}$：trajectory $i$ 的第 $t$ 个 token
- $o_{i,<t}$：trajectory $i$ 前 $t-1$ 个 token
- $\hat{r}_{i,t}(\theta)$：importance sampling ratio（见公式3）
- $\hat{A}_{i,t}$：advantage estimate（见公式4）
- $sg(\cdot)$：stop-gradient operator，阻止梯度流过 importance weight
- $\pi_{\theta_{\text{old}}}$：rollout 时的旧 policy
- $\pi_\theta$：当前要优化的 policy

公式3（asymmetric clipping）：
$$\hat{r}_{i,t}(\theta) = \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}, 0, 1+\epsilon_{\text{high}}^{\text{IS}}\right)$$

Intuition 关键点：
- **Lower bound = 0**（不是 PPO 的 $1-\epsilon$）：当某 action 在当前 policy 下变得很不可能时，importance ratio 趋近 0，gradient 被完全 zero out。这允许 **aggressive down-weighting**——policy 不再支持某 action 时，可以快速 forget。
- **Upper bound = $1+\epsilon_{\text{high}}^{\text{IS}}$**：防止过大 policy update。PPO 的对称 clip（$1\pm\epsilon$）会保留"应该 upweight 但被 clip"的 gradient，CISPO 的不对称设计让"已经 gain 太多"的 action 停止 update。

**Stop-gradient on clipped ratio**：传统 PPO 的 clip 在 backward 时让 importance ratio 的梯度参与计算，引入 second-order term。CISPO 的 $sg(\cdot)$ 让 importance ratio 只起 **modulate gradient magnitude** 的作用，不引入二阶项，得到 first-order 稳定 update rule。

公式4（advantage estimate）：
$$\hat{A}_{i,t} = \sum_{p=t}^T r_p - B_i$$

- $r_p$：step $p$ 的 composite reward
- $B_i$：trajectory $i$ 的 baseline（用于 variance reduction）

这是 reward-to-go 减去 trajectory-level baseline。$B_i$ 是整个 trajectory 的平均 reward（不是 token-level critic），简化实现且对 long-horizon 稳定。

### 5.3 Composite Reward 设计

paper 指出 standard outcome-based reward 在 192K token、数千 action 的 trajectory 上 insufficient for credit assignment。三个 component：

**Process Reward** $r_t^{\text{process}}$：
- penalty for language mixing（中英文混杂）
- penalty for tool invocation format errors
- reward for well-structured intermediate reasoning

**Task Completion Time Reward** $r_t^{\text{speed}}$（公式5）：
$$r_t^{\text{speed}} = h\left(\frac{T_{\text{completion}}}{T_{\text{baseline}}}\right)$$

- $h(\cdot)$：单调递减 shaping function
- $T_{\text{completion}}$：rollout 的 wall-clock time
- $T_{\text{baseline}}$：参考 completion time

Intuition：传统 RL 只 optimize correctness，忽略 efficiency。但 agentic 任务里 sequential vs parallel tool execution、sub-agent invocation overhead 会让"功能等价"的 trajectory 的 latency 差几个量级。这个 reward 让 policy 发现 parallelism 机会。

**Reward-to-Go with Baseline**（公式6）：
$$G_t = \sum_{\tau=t}^T \gamma^{\tau-t} r_\tau$$

- $\gamma$：discount factor
- $r_\tau$：step $\tau$ 的 reward

这把 gradient signal 集中到"consequences 还没被 accounted for"的 action 上，improving credit assignment precision。

**Composite reward**（公式7）：
$$r_t = \alpha \cdot r_t^{\text{process}} + \beta \cdot r_t^{\text{speed}} + r_t^{\text{perf}}$$

- $\alpha, \beta$：coefficients
- $r_t^{\text{perf}}$：primary task performance signal（如 SWE 的 test pass）

### 5.4 Mixed-Domain RL Training

避免两个 failure mode：
1. Single-domain RL → catastrophic forgetting
2. Sequential multi-stage → negative transfer

策略：**每 stage 同时从四个 domain（reasoning / coding / agent / general）抽数据**，joint optimization。每 step 的 policy gradient 都 informed by diverse task distribution。

三轴 curriculum（across stages）：
1. **Domain mixing ratio**：早期 reasoning+general 重，后期 agent+coding 重
2. **Context length**：per-domain 扩展
3. **Difficulty distribution**：每 domain 内逐步偏 hard

---

## 6. Forge RL Infrastructure

### 6.1 The Impossible Triangle

paper 形式化三个互相矛盾的 desiderata：
1. **System Throughput**：max tokens/sec
2. **Training Stability**：$\mathbb{E}[\text{Var}(\nabla_\theta J)] < \delta$，单调提升收敛
3. **Agent Flexibility**：support arbitrary $\mathcal{A} \in \Omega_{\text{agent}}$，从 simple single-turn 到 multi-agent dynamic context management

优化目标（公式8）：
$$\max_\theta J(\theta) = \text{Throughput}(\mathcal{A}) \times \text{SampleEfficiency}(\mathcal{A})$$
$$\text{s.t.} \quad \forall \mathcal{R} \in \Omega_{\text{agent}}, \mathbb{E}[\text{Var}(\nabla_\theta J)] < \delta, \mathbb{E}[\|J^{(T)} - J^*\|] < \epsilon$$

Intuition：throughput × sample efficiency = effective training yield。两个 constraint 保证 stability 和 convergence。

### 6.2 Three Decoupled Modules

Figure 4 的架构：
1. **Agent Side**：纯 trajectory producer，完全 agnostic to training/inference mechanics。执行 tool call、管理 context、访问 memory，记录 $(s_t, a_t, o_t)$ tuples。
2. **Middleware**：Gateway Server（标准化通信接口，route completion requests）+ Data Pool（distributed trajectory storage，async 收集 rollout data）。
3. **Training/Inference Side**：Rollout Engine（高吞吐 token generation，作为 $\pi_\theta$ 响应 Gateway requests）+ Train Engine（消费 trajectory，compute CISPO gradient，sync 回 Rollout Engine）。

关键设计：**generation 和 training pipeline 完全 decoupled**，可以独立 scale。Data Pool 异步收集让两边 throughput 互不阻塞。

### 6.3 White-Box vs Black-Box Agent Support

**White-Box Agents**：暴露 context management logic 给 framework。state transition $s_{t+1} = f_{\text{CM}}(\text{concat}(s_t, a_t, o_t))$ 在 framework 内实现，training pipeline 能直接 observe 和 backprop through context transformation。

适合：well-defined CM strategies（sliding-window truncation、periodic summarization）。framework 能 reconstruct exact training states。

**Black-Box Agents**：agent 是 opaque trajectory producer。所有 internal context management / memory compression / multi-agent coordination 不暴露。framework 只看 externally visible $(s_t, a_t, o_t)$ tuples。

适合：deep thinking loops、aggressive context rewriting、hierarchical multi-agent systems。**不需要 agent-side 任何修改**。

Intuition：white-box 像编译器优化（framework 能看 AST），black-box 像调外部 service（只看 I/O）。两种 unified 在 Gateway-based abstraction 下。

### 6.4 Windowed FIFO Scheduling

问题：agent rollout completion time variance 巨大（秒级到小时级）。
- Strict FIFO：保 distribution consistency，但 head-of-line blocking——一个慢 task 卡住整个 queue
- Fully greedy：max throughput，但 distribution shift——早期 batch 都是 short easy task，hard task 集中在后期，gradient oscillation

**Windowed FIFO**（Figure 5）：
- Generation queue $Q = [T_0, T_1, \dots, T_{N-1}]$，current head index $i$
- Scheduler 只能在 sliding window $[T_i, T_{i+W-1}]$ 内 fetch completed trajectories
- Window 内：greedy（任何完成的都能 fetch），缓解 HoL blocking
- Window 外：strict FIFO（beyond window 的 trajectory 无论是否完成都 block）
- Window 只在 head-of-window task 被 consume 时 advance

trade-off parameter：$W$
- $W \to 0$：strict FIFO（max distributional consistency）
- $W \to N$：fully greedy（max throughput）

实践中 $W = 0.3N$ 接近 FIFO 分布特性同时大幅减少 cluster idle time。

Intuition：这像 batch processing 系统的 "latency-bounded reordering"——允许小范围 reorder 提吞吐，但保整体 distribution 不偏。MapReduce 的 shuffle 优化有类似思想。

### 6.5 Prefix Tree Merging

Multi-turn agent trajectory 里 sequential message append + context management 产生大量 shared prefix（同一 rollout group 内的 G 个 trajectory 共享 system prompt + early turns）。

传统 training：每个 sample 独立 recompute common prefix，long-context agent 场景下严重浪费。

**Prefix tree merging**（Figure 6）：
1. 把共享 prefix 的多个 completion 合并成 single prefix tree
2. Forward pass：shared prefix 只算一次，到 branch 处分叉到 individual response segments
3. Forward 后用 stored metadata deconstruct tree
4. Per-sample 独立计算 loss

关键保证：**数学上等价于 independent-sample training**——causal attention 在 shared prefix 上产生 identical activations，无论算 1 次还是 $N$ 次。**zero approximation error**。

实测：up to **40× training speedup** + 对应 memory reduction。

Intuition：这是把 beam search 的 prefix sharing 思想用到 training forward pass。RL 的 G 个 rollout 共享 prompt 是天然 prefix sharing 场景，特别是 long-context agent 下 early turns 几乎完全相同。

### 6.6 Inference Acceleration

三个 optimization：

**MTP-based Speculative Decoding**：MTP modules 在 RL training 中通过 top-k KL divergence loss 与 policy co-train，确保 draft acceptance rate 在 non-stationary RL 优化过程中保持高。否则 policy 演化会让 draft model 跟不上，speculative decoding 性能退化。

**Heterogeneous Prefill-Decode Disaggregation**：Prefill 和 decode 操作 decouple 到独立 scheduled instances。消除 MoE 架构下 mixed scheduling 的相互干扰。每 phase 用各自优化的 parallelism strategy。

**Global L3 KV Cache Pool**：distributed、DFS-backed global KV cache，通过 group-level rollout scheduling 最大化 prefix cache hit rate。cost-aware request router 平衡 queuing delay vs cache migration cost。

---

## 7. Agentic Mechanism：Interleaved Thinking + Self-Evolution

### 7.1 Interleaved Thinking

公式9（trajectory 结构）：
$$\tau = (r_1, a_1, o_1, r_2, a_2, o_2, \dots, r_T, a_T, o_T)$$

- $r_t$：reasoning tokens（thinking block）
- $a_t$：action tokens（tool call）
- $o_t$：observation（tool output）

每个 $r_t$ conditioned on full history $(r_1, a_1, o_1, \dots, r_{t-1}, a_{t-1}, o_{t-1})$。

对比两个 alternative：
1. **Front-loaded reasoning**：所有 reasoning 在 actions 前——不能适应 intermediate observation
2. **Stateless per-turn reasoning**：每 turn 把 $r_{<t}$ 从 context strip 掉——不能 build on earlier analysis

公式10（reasoning state persistence）：
$$\mathcal{H}_{t+1} = \mathcal{H}_t \oplus [\text{assistant}(r_t, a_t)] \oplus [\text{tool}(o_t)]$$

如果不 persist：$\mathcal{H}_{t+1}^{(\text{drop})} = \mathcal{H}_t \oplus [\text{assistant}(a_t)] \oplus [\text{tool}(o_t)]$，每 turn 都要重新 derive context/constraints/partial conclusions，导致 cumulative state drift + degraded self-correction。

**Plan-Act-Reflect loop**（Figure 7）：
1. **Plan**：review accumulated state，formulate/refine strategy
2. **Act**：select & execute tool call grounded in plan
3. **Reflect**：evaluate observation vs expectation，update world model，决定 revise plan or proceed

Ablation：strip thinking blocks from prior turns → agentic benchmark 全面下降，deep search 和 SWE 任务下降最大——说明 interleaved thinking 在需要 sustained planning + iterative refinement 的任务上最 impactful。

Intuition：这本质是把 ReAct (Yao et al. 2022) 升级——不只是 Reason-Act 交替，而是把 reasoning state 完整 persist 到下一 turn 的 context，让 model 可以"复盘自己之前的思考"。

### 7.2 Self-Evolution

paper 最 future-leaning 的部分。M2.7 已经能：
1. Autonomous triage failed training runs
2. Edit its own agent scaffold
3. Multi-round self-improvement on ML engineering tasks

**Model Iteration System**（Figure 8A）：
- Principle："humans steer while models build"
- Researcher 配置 goals、通过 chat guide agent、review outputs 决定下一步
- Agent 在 "Agent Harness" 内运行——harness 完全由内部 M2.7 生成，**zero human-written code**
- Harness 配备 hierarchical skills：action chaining、persistent memory、safety guardrails、evaluation infrastructure

**Dual-Loop Workflow**（Figure 8B）：
- Human-led experiment planning
- M2.7 autonomous execution：profile ongoing runs、read logs、diagnose metric anomalies
- Auto-debug code + 调整 configurations
- 直接 intervene 自己的训练 loop，吸收 **30-50% 的 daily iteration workload**
- Human review 触发 major iteration decisions
- Agent 可 auto-continue bounded analysis between reviews

**Recursive Scaffold Upgrades**：
M2.7 被给定"优化内部 programming scaffold"任务，执行了 **100-round 全自动 iteration cycle**：analyze failures → modify code → evaluate changes。这引入了 loop detection 机制，发现更好参数组合，**in-house evaluation 上 30% 性能提升**。

Intuition：这是把 model 的 capability 用回自己的开发上。RL 团队的 daily 工作变成"高 level goal setting + final review"，agent 把 30-50% 的 mechanical iteration 接过去。这条路径如果走通，frontier model development 的 human-in-the-loop bottleneck 会被大幅 compress。

---

## 8. Evaluation

### 8.1 Main Results (Table 4)

跟 4 个 frontier 闭源 baseline 对比：Claude Opus 4.6、Claude Sonnet 4.6、GPT 5.4、Gemini 3.1 Pro。

**Agentic Coding**：
| Benchmark | M2.7 | M2.5 | Opus 4.6 | Sonnet 4.6 | GPT 5.4 | Gemini 3.1 Pro |
|---|---|---|---|---|---|---|
| SWE-bench Pro | 56.2 | 55.4 | 57.2 | 57.7 | 54.2 | – |
| SWE-bench Multilingual | 76.5 | 74.1 | 77.8 | 75.9 | 70.5 | – |
| Multi-SWE-bench | **52.7** | 51.3 | 50.3 | 51.0 | 49.0 | – |
| NL2Repo | 39.8 | 26.6 | 43.7 | 43.3 | 46.8 | 35.9 |
| Terminal-Bench 2.0 | 57.0 | 51.7 | 65.4 | 59.1 | 75.1 | 68.5 |
| MLE Bench Lite | 66.6 | 51.5 | 75.7 | 72.7 | 71.2 | 66.6 |

M2.7 在 Multi-SWE-bench 上拿到第一，SWE-bench Pro/Multilingual 跟 frontier 持平。NL2Repo 上 M2.7 比 M2.5 跳了 13.2 分——印证 §4.1 引入的 full-stack repository data 起作用。

**Agentic Cowork - Search**：
| Benchmark | M2.7 | M2.5 |
|---|---|---|
| BrowseComp | 77.8 | 59.4 |
| Wide Search | 75.2 | 70.3 |
| RISE | 64.3 | 50.2 |

BrowseComp +18.4，RISE +14.1——deep search 上 M2.7 进步最显著，符合 §4.2 的 evidence-grounded synthesis 设计。

**Agentic Cowork - Office**：
| Benchmark | M2.7 | M2.5 |
|---|---|---|
| GDPval-AA | 50.0 | 35.0 |
| Toolathlon | 46.3 | 38.3 |
| MM Claw | 62.7 | 75.4 |
| MEWC v2 | 63.3 | 49.8 |
| Finance Modeling Pro | 57.0 | 33.8 |

GDPval-AA +15.0、MEWC v2 +13.5、Finance Modeling Pro +23.2——office task 是 M2.5 → M2.7 增长最大的 block。

**Reasoning & Knowledge**：
| Benchmark | M2.7 | M2.5 | Opus 4.6 | Sonnet 4.6 | GPT 5.4 | Gemini 3.1 Pro |
|---|---|---|---|---|---|---|
| AIME 2026 | 94.2 | 87.2 | 92.5 | 92.7 | 97.0 | 88.7 |
| GPQA-Diamond | 89.8 | 85.2 | 89.6 | 87.5 | 92.0 | 94.1 |
| SciCode | 47.0 | 43.0 | 51.9 | 46.8 | 56.6 | 58.9 |
| IFBench | 76.0 | 72.0 | 53.1 | 56.6 | 73.9 | 77.1 |
| AA-LCR | 72.0 | 65.0 | 70.7 | 70.7 | 74.0 | 72.7 |
| HLE | 28.0 | 19.0 | 36.7 | 30.0 | 41.6 | 44.7 |
| MMLU-Pro | 81.8 | 85.2 | 89.1 | 87.3 | 87.5 | 91.2 |

M2.7 在 AIME 2026 上 94.2，超 Opus 4.6 和 Sonnet 4.6。IFBench 76.0 远超 Claude 系列——instruction following 的 multi-domain RL 起作用。

### 8.2 Within-Series Progression (Figure 9)

M2 → M2.5 → M2.7 在 11 个 benchmark 上 all improve，gain 从 +11（AA-LCR、GPQA-Diamond）到 +33.8（BrowseComp）。

**Pattern**：
- 引入新 task family 的 benchmark 涨得最猛：BrowseComp +33.8、Wide Search +12.9、Toolathlon +27.5、GDPval-AA +16.0、MLE Bench Lite +26.6
- 原本就强的 benchmark incremental：SWE-bench Multilingual、Multi-SWE-bench
- Reasoning benchmark 稳步上升：AIME 2025 +16.0、GPQA-Diamond +11.8、AA-LCR +11.0

Intuition：这印证了 paper 的核心论点——**数据 pipeline 决定 capability frontier**。引入新 task family 直接对应 capability jump，而 model 规模（同样 ~10B activated）没变。这是 agentic post-training era 的关键 insight：不是 base model 决定上限，是 data pipeline 决定上限。

### 8.3 Self-Evolution Case Study：MLE Bench Lite

M2.7 在 OpenAI 的 MLE Bench Lite (Chan et al. 2025, https://arxiv.org/abs/2410.13276) 上达到 66.6% medal rate，**tie Gemini 3.1 Pro**。

实验设置：
- 22 competitions
- 每 run 24 小时 iterative evolution
- Simple autonomous harness（short-term memory + self-feedback，**harness 内 zero human-written code**）
- 每 iteration 后 model 写 memory file + rigorous self-criticism，建立 explicit optimization directions

结果：best run = 9 gold + 5 silver + 1 bronze。M2.7 demonstrated clear cumulative improvement over time。

Figure 10 显示 medal rate 随 trial 累积上升——证明 model 能 build on accumulated feedback chain，而不是 plateau。

---

## 9. 我的 Intuition 与 Takeaways

### 9.1 这篇 paper 真正的 contribution

**不是 base model**：229.9B/9.8B 的 MoE 架构虽然 fine-grained + sigmoid + bias 是 incremental 改进，但不是革命性。Full attention 放弃 hybrid 也是 engineering trade-off，不是 algorithmic 突破。

**真正 contribution 是 agentic post-training stack 的工程化**：
1. **Verifiable reward engineering**：SWE 的 F2P/P2P、AppDev 的 AaaV 三层 verification、Terminal-Gym 的 unified test suite、Deep Search 的 evidence specification——每个 domain 都设计 domain-specific 但 automated 的 reward signal。这是 RL scaling 的 prerequisite。
2. **RL infrastructure 的"不可能三角"解构**：Forge 通过 decoupling + windowed FIFO + prefix tree merging 同时 achieve throughput / stability / flexibility。Windowed FIFO 是经典 systems 的 latency-bounded reordering 思想，prefix tree merging 是 beam search 思想用到 training forward，两个都是工程优雅。
3. **Self-evolution 的早期 operational form**：M2.7 能 debug 自己的训练 run、改自己的 scaffold、100-round 自主迭代。这是 paper 最 future-facing 的部分——把 model 的 capability 直接 feed 回自己的开发 loop。

### 9.2 Mini Activations 的真正意义

paper 的 thesis "mini activations unleash max real-world intelligence" 不是说小模型也能打——而是说 **per-token compute 的节省 可以 redirect 到 long-horizon agentic rollout**。

192K context 的 agent rollout 一个 episode 几千 action、上百万 token，inference cost 远超单次 chat。mini activations 让这个 cost 在 production deployment 可承受。这是 "compute footprint → trajectory length" 的转换。

更深一层：**self-evolution** 让 model 用自己的 capability 改进自己的训练——这个 loop 的每一步都需要大量 agent rollout（debug 训练、改 scaffold、evaluate）。如果 per-token cost 是 100B activated，这个 loop 经济上不可行；mini activations 让 self-evolution loop 经济可行。

### 9.3 关于 CISPO 的设计哲学

CISPO 的 asymmetric clipping（lower bound = 0, upper bound = $1+\epsilon$）反映了对 RL 训练稳定性的理解：
- **允许快速 forget**：lower bound 0 让 policy 能 aggressively down-weight 已不可能的 action，这对 long-horizon agent 重要——避免被早期 bad trajectory 锁死
- **防止快速 over-upweight**：upper bound 限制过大 update，防 policy collapse
- **Stop-gradient on ratio**：避免 second-order term 让优化更稳定，简化实现

跟 PPO 比，CISPO 更"aggressive on forgetting, conservative on reinforcing"——符合 agent training 的实际需求。

### 9.4 对未来的 hint

paper 最后说："each axis—data, RL system, and self-evolution—remains far from saturation"。这暗示 M2.x 后续会继续 scale 三轴。

值得关注的后续方向：
1. **Anything2Docker**：把 Terminal-Gym 的 Docker env synthesis 推到 zero-intervention，可能扩展到任意 task → env 自动构造
2. **CVE-Factory 扩展**：autonomous cybersecurity research，可能涉及 autonomous vulnerability discovery + defense
3. **Recursive self-evolution 的 scaling**：M2.7 是 100-round，下一代可能是 1000-round，model 修改 model 修改 model 的 deeper recursion

---

## 10. 参考链接汇总

### Architecture & Algorithm
- DeepSeekMoE (fine-grained experts): https://arxiv.org/abs/2401.06066
- DeepSeek-V3 (MTP 设计): https://arxiv.org/abs/2412.19437
- MTP 原始 (Gloeckle et al.): https://arxiv.org/abs/2404.19737
- GQA (Ainslie et al.): https://arxiv.org/abs/2305.13245
- RoPE (Su et al.): https://arxiv.org/abs/2104.09864
- Auxiliary-loss-free load balancing (Wang et al.): https://arxiv.org/abs/2408.15664
- Speculative decoding (Leviathan et al.): https://arxiv.org/abs/2211.17192
- Sigmoid gating MoE (Shazeer et al. 2017): https://arxiv.org/abs/1701.06538
- MiniMax-Text-01 (Lightning Attention hybrid 前身): https://arxiv.org/abs/2501.01464
- MiniMax-M1 (CISPO 原始): https://arxiv.org/abs/2506.13585

### Benchmarks
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-bench Multilingual: https://arxiv.org/abs/2508.02153
- SWE-Smith (commit merging): https://arxiv.org/abs/2412.21132
- Terminal-Bench: https://arxiv.org/abs/2502.08200
- MLE-bench: https://arxiv.org/abs/2410.13276
- BrowseComp: https://arxiv.org/abs/2504.13716
- GDPval: https://arxiv.org/abs/2503.12631
- GAIA: https://arxiv.org/abs/2310.02357
- GPQA-Diamond: https://arxiv.org/abs/2311.12022
- HLE (Humanity's Last Exam): https://arxiv.org/abs/2501.04909
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- IFBench: https://arxiv.org/abs/2506.10406
- Toolathlon: https://arxiv.org/abs/2510.01257
- RULER: https://arxiv.org/abs/2404.06654
- HELMET: https://arxiv.org/abs/2410.18976
- SciCode: https://arxiv.org/abs/2410.03784
- KorBench: https://arxiv.org/abs/2507.13139
- CVE-Factory: https://arxiv.org/abs/2603.11076

### 相关 Agent/System 工作参考
- WebExplorer (Liu et al. 2025b): https://arxiv.org/abs/2509.06501
- Attention Sinks (StreamingLLM): https://arxiv.org/abs/2309.17453
- Induction Heads (Olsson et al.): https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- Longformer (SWA): https://arxiv.org/abs/2004.05150
- ReAct (Reason+Act, Yao et al.): https://arxiv.org/abs/2210.03629

---

总结一句话：MiniMax-M2 series 是把 **MoE 的 compute efficiency**、**verifiable agentic data pipeline 的 reward engineering**、**Forge RL infrastructure 的 decoupled architecture**、**interleaved thinking 的 reasoning state persistence**、**self-evolution 的 model-on-model recursion** 五件事 stack 在一起，用 ~10B activated parameters 逼近 frontier 闭源模型。它的核心贡献不在某个 algorithm，而是一整套 **agentic post-training 的工程化方法论**——这是 base model era 结束后，frontier model 团队真正在拼的东西。
