---
source_pdf: GLM-5.pdf
paper_sha256: 4a97eab759d4576b8a93cac0cdf8e25b8fc4c01d71c61f759548eab6796e6ac7
processed_at: '2026-08-04T21:46:52-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GLM-5 用人话讲

## 一句话总结

GLM-5是个744B参数的MoE模型，核心卖点是把"写代码"从人prompting模型（vibe coding）推向模型自己规划、实现、迭代的agentic engineering，靠的是三件事：**更高效的attention架构**、**异步RL训练框架**、**大规模可验证环境**。

---

## Architecture部分

### MLA + Muon Split

MLA就是DeepSeek发明的把KV cache压到低维的trick，省显存。但GLM团队发现MLA配上他们自己的Muon optimizer打不过GQA-8。原因是Muon会对up-projection矩阵做orthogonalization，如果所有attention head共享一个矩阵，更新scale是耦合的，不同head没法独立调整。

解法很直觉：把up-projection矩阵按head切开，每个head独立做orthogonalization。这样不同head可以以不同速度学习，性能就追上来了。

还有个MLA-256的小trick：原版MLA head dim=192，decoding时要做576维点积，在某些硬件上不划算。GLM-5把head dim加到256，head数砍1/3，训练FLOPs和参数不变，但decoding更快。这是针对非H800硬件的调优。

### MTP共享参数

Multi-token prediction就是训练时让模型一次predict好几个token，既能提升base model质量，也能做speculative decoding的draft model。

问题是如果要predict n个token就得有n个MTP layer，显存linearly涨。DeepSeek-V3用单layer predict 2个token，但训练-推理不一致，第2个token accept rate低。

GLM-5的解法：3个MTP layer共享同一组参数。这样显存cost和DeepSeek-V3一样，但accept length从2.55涨到2.76。这个在小batch decoding（RL rollout的典型场景）特别有用。

### DSA — 真正的核心

DSA的intuition非常simple：128K context下dense attention太贵了，sliding window会丢信息，那怎么办？让模型自己看内容，决定哪些token重要，只attend那些。

具体来说有个Lightning Indexer，对每个query token算relevance score，选top-2048个key-value entry，然后只在这些上做attention。

关键insight是DSA不需要从头训，而是从dense base model做continued pre-training：先warmup 1000步只训indexer，再sparse adaptation 20B tokens。因为90%的attention entry在long context下本来就是redundant的，所以sparse attention理论上lossless。

实验也验证了：DSA在long-context benchmark上几乎不掉点，而且比SWA、linear attention这些方法好得多。那些方法多多少少都丢信息，DSA是纯selection，不丢。

---

## RL部分 — 这是paper的真正亮点

### 为什么异步RL这么重要

传统同步RL的问题：agent rollout有严重的long-tail。一个trajectory可能跑几小时，如果同步等所有rollout完成再update，GPU大量idle。这对agentic training是致命的，因为agent任务天然就有巨大的duration variance。

GLM-5的解法是彻底异步：inference engine不停生成trajectory，攒够一批就送training engine update，training engine每K步把新weights推回inference engine。代价是不同trajectory可能来自不同model version，引入off-policy bias。

### 怎么稳定异步RL

这里有三个关键mechanism：

**TITO (Token-in-Token-out)**：训练端直接消费inference端的token IDs，不走text round-trip。为什么这重要？因为如果inference端输出text再让训练端re-tokenize，token boundary可能不一致，action和reward的对齐就乱了。在streaming、truncation的场景下这个mismatch是致命的。

**Direct double-sided importance sampling**：异步RL没法track历史所有policy checkpoint，所以直接用rollout时的log probability做behavior proxy。importance ratio超出的token直接mask掉，不参与gradient。比标准PPO的clipping更简单更稳定。

**Dropping off-policy samples**：记录每个trajectory生成时的model version，如果version太老（超过阈值τ），直接扔掉。环境crash的sample也扔掉，因为那是环境问题不是模型问题。

### DSA在RL里的坑

这个细节很值得注意。DSA的indexer要选top-2048个KV entry，这个选择必须deterministic。如果用CUDA的非deterministic top-k，训练时和推理时选的entry不一致，整个attention就乱了，RL几步后entropy就崩了。

解法：用`torch.topk`，虽然慢一点但deterministic。整个RL过程freeze indexer参数。

这跟MoE的routing replay是一个道理：训练时要reuse推理时的routing决策，否则training-inference mismatch会destabilize RL。

### General RL的三维度

GLM-5把General RL拆成三个目标：
- **Foundational correctness**：不犯instruction following、逻辑、事实、hallucination的错
- **Emotional intelligence**：empathy、insight、自然人类沟通风格
- **Task-specific quality**：writing、QA、role-playing、translation各自的细粒度质量

Reward用hybrid system：rule-based（精确但局限）+ ORM（高效但容易reward hacking）+ GRM（robust但variance高）。三者互补。

还有个很有意思的design choice：explicit引入人类专家写的response作为stylistic anchor。因为纯模型生成容易收敛到verbose、formulaic的"model-like"风格。

### On-Policy Cross-Stage Distillation

多阶段RL有个经典问题：训完Reasoning RL再训Agentic RL，reasoning能力可能degrade。

解法是最后做一轮on-policy distillation，用前面各阶段的final checkpoint当teacher。advantage就是teacher和student的log probability ratio，teacher stop gradient不更新，student通过match teacher distribution来恢复之前的能力。

group size设为1（不需要大group estimate advantage），batch size 1024，throughput很高。

---

## 环境部分

### SWE环境

基于RepoLaunch框架，自动分析repo的installation和dependency，生成test commands，用LLM生成log-parsing function提取F2P和P2P test case。最终构建了10k+可验证环境，覆盖9种语言。

### Terminal环境

两条pipeline：
1. 从seed task出发，LLM brainstorm生成task draft，construction agent变成Harbor format（Docker + test script），refine agent迭代优化
2. 从web corpus出发，coding agent合成terminal task，然后自己做first-pass验证，失败就迭代修改直到通过

这个closed-loop self-verification很clever——构造数据的agent本身就是自己的evaluator。

### Search环境

建了个Web Knowledge Graph，2M+网页，LLM做entity extraction和relation consolidation。然后采样low-frequency entity做multi-hop neighborhood expansion，生成multi-entity关系链问题。

三阶段难度过滤：tool-free模型能答的不要、early-stage agent几步能解的不要、verification agent做bidirectional validation。

### Slide generation的reward设计

三层reward很直觉：
1. 静态HTML属性：position、color、typography等，规则约束
2. 运行时rendering属性：DOM的实际width/height/bounding box，需要真的render才能拿到
3. 视觉感知特征：abnormal whitespace detection等

为什么需要runtime rendering？因为静态检查会被reward hack——模型可能hard truncate内容或manipulate spacing来骗metric。真的render才能抓到这些。

---

## 评估的key insight

### CC-Bench-V2的设计哲学

paper反复强调：static benchmark（如SWE-bench）是single-commit isolated edits，不能评估真实engineering的long-horizon、state-recursive、incremental特性。

CC-Bench-V2的三个维度：
- **Frontend**: 用Agent-as-a-Judge（GUI agent + Playwright）自动验证，94%和人类expert agreement
- **Backend**: 85个真实open-source项目任务，6种语言，all-or-nothing unit test
- **Long-horizon**: Repo Exploration（找文件）+ Chained Tasks（multi-commit任务链）

Chained Tasks的设计特别有意思：从merged PR里挖掘3-15个commit的PR，用dynamic programming把commit分成coherent task groups，agent顺序执行，每个task完成后apply下一个task的auto-apply patch。这直接测试long-context consistency和self-correction。

### GLM-5 vs Claude Opus 4.5的gap在哪

从Table 8能看出来：
- Frontend ISR：GLM-5 vs Claude Opus 4.5有gap（HTML 38.9 vs 52.2），说明end-to-end完成度不够
- Chained Tasks：52.3 vs 61.6，error compounding across chain是主要问题
- Repo Exploration：65.6 vs 64.5，GLM-5反而更好

这build了intuition：GLM-5在单步能力上接近frontier，但在需要维持long-horizon consistency和error recovery的场景还有gap。

---

## 整体故事的intuition

GLM-5的narrative其实是：**LLM正在从knowledge repository变成active problem solver，bottleneck不再是参数量或数据量，而是训练效率和真实环境feedback的scale**。

他们解决这个bottleneck的方式是full-stack co-design：
- Architecture level: DSA + MLA + MTP降低长context的计算cost
- System level: 异步RL + TITO + DP-aware routing提升训练throughput
- Data level: 10k+ verifiable environments提供真实engineering feedback

最值得注意的positioning：GLM-5在BrowseComp（75.9）上SOTA across所有模型包括proprietary，在SWE-bench Multilingual上beats Gemini 3 Pro和GPT-5.2。open-source model在agentic engineering上首次compete with frontier proprietary models。

最大的limitation：chained long-horizon tasks的error compounding，需要更好的long-context consistency和self-correction能力。

---

# GLM-5: From Vibe Coding to Agentic Engineering 深度技术解析

## 0. 核心intuition: 从"vibe coding"到"agentic engineering"的paradigm shift

GLM-5这篇paper的核心thesis是：当前LLM的评估范式（如SWE-bench这种single-commit isolated edits）已经无法capture真实软件工程的能力，因为真实的engineering是**long-horizon、state-recursive、incremental**的过程。GLM-5试图把模型从"被动knowledge repository"推向"active problem solver"，and关键bottleneck变成了：

1. **Computational cost**: 训练和推理cost在long-context reasoning下爆炸
2. **Real-world adaptability**: 静态benchmark与真实agentic workflow的gap

这俩问题的解法构成了paper的technical contribution主线：
- DSA + MLA + MTP组合解决cost问题
- Asynchronous RL + 环境scaling解决adaptability问题

参考链接：
- DeepSeek Sparse Attention原paper: https://arxiv.org/abs/2512.02556
- GLM-4.5 technical report (前作): https://arxiv.org/abs/2507.15xxx
- GRPO原始paper (DeepSeekMath): https://arxiv.org/abs/2402.03300
- IcePop: https://arxiv.org/abs/2509.xxxxx (MoE RL)

---

## 1. Architecture: 三大核心创新

### 1.1 Model scaling: 744B total / 40B activated

GLM-5相比GLM-4.5（355B/32B activated）scale了大约2x：
- **256 experts** (vs GLM-4.5的160)
- **80 layers** (减少，vs GLM-4.5的92)，目的是minimize expert parallelism communication overhead
- 8 routed experts + 1 shared expert

为什么减少layer count？因为MoE的expert parallelism在pipeline parallel里，layer数多了会造成更多communication boundary。这里trade-off是depth vs width——他们选择了wider但shallower的架构，使得EP通信在更少的pipeline stage之间进行。

### 1.2 Multi-latent Attention (MLA) + Muon Split

**MLA的核心思想**（来自DeepSeek-V2）：把KV cache压缩到一个低维latent space。

标准attention：
$$\text{Attn}(q, K, V) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}}\right)V$$

MLA把key和value投影到latent dimension：
- $c_K = W^{DK} h$ (down-projection到576维latent)
- $c_V = W^{DV} h$
- $k = W^{UK} c_K$ (up-projection回high dim，仅在需要时计算)
- $v = W^{UV} c_V$

cache里只存$c_K$和$c_V$（576维），decode时再up-project，这样KV cache大小大幅减少。

**Muon optimizer的问题**：在GLM-4.5的recipe里，Muon会对up-projection矩阵 $W^{UQ}, W^{UK}, W^{UV}$ 做 matrix orthogonalization。问题是所有attention heads共享同一个up-projection矩阵时，orthogonalization的scale对所有heads是耦合的，导致MLA无法match GQA-8的性能。

**Muon Split的解法**：把 $W^{UQ}, W^{UK}, W^{UV}$ 按head split成多个小矩阵，对每个head独立做orthogonalization：

$$W^{UQ} \rightarrow \{W^{UQ}_1, W^{UQ}_2, \ldots, W^{UQ}_H\}$$

每个 $W^{UQ}_h$ 独立做 Newton-Schulz orthogonalization，使得不同heads的projection weights可以以不同scale更新。

Table 1的结果显示：
- GQA-8: MMLU 61.2, BBH 53.3
- MLA (naive): MMLU 61.5, BBH 48.9 (BBH掉很多)
- MLA + Muon Split: MMLU 62.5, BBH 51.8 (恢复)
- MLA-256 + Muon Split: MMLU 62.0, BBH 51.3

**MLA-256的额外trick**：原版MLA head dim=192，decoding时576维点积计算量大。GLM-5把head dim从192增加到256，同时attention heads数减少1/3，保持training FLOPs和params不变，但decoding计算下降。

这是个H800 roofline vs 其他硬件的trade-off：DeepSeek-V3的head数选H800 roofline optimal，但在其他硬件上不optimal。

### 1.3 Multi-Token Prediction with Parameter Sharing

**MTP motivation**：训练时predict next n tokens可以improve base model quality，and serve as draft model for speculative decoding。

**原版MTP的问题**：predict next n tokens需要n个独立MTP layer，memory和KV cache linearly scale with speculative steps。DeepSeek-V3用单MTP layer predict next 2 tokens，但training-inference discrepancy降低了第2个token的accept rate。

**GLM-5的解法**：3个MTP layer share parameters during training。

$$\theta_{MTP}^{(1)} = \theta_{MTP}^{(2)} = \theta_{MTP}^{(3)} = \theta_{MTP}$$

这意味着3层共享同一组weight。好处：
- Memory cost = 单个MTP layer (同DeepSeek-V3)
- Accept rate提升，因为training时多个MTP step的prediction都来自同一个model

Table 2: Accept Length比较
- DeepSeek-V3.2: 2.55
- GLM-5: 2.76 (4个speculative steps)

这个改进对小batch decoding（RL rollout的典型场景）特别重要，因为speculative decoding的speedup在小batch时更显著。

### 1.4 DeepSeek Sparse Attention (DSA) — 核心architectural innovation

**DSA的intuition**：传统dense attention $O(L^2)$ 在128K context下prohibitively expensive。Fixed patterns（如sliding window）有信息损失。DSA的核心是**content-based dynamic selection**：

DSA结构：
1. **Lightning Indexer**：对每个query token，计算对所有key token的relevance score，选出top-k个most relevant的key-value entries
2. **Sparse Attention**：仅在selected subset上计算attention

形式化：
$$\text{DSA}(q_i, K, V) = \text{softmax}\left(\frac{q_i K_{S_i}^T}{\sqrt{d}}\right)V_{S_i}$$

其中 $S_i = \text{TopK}(\text{Indexer}(q_i, K), k)$，k=2048。

**Critical engineering insight**：DSA不是train from scratch，而是**Continued Pre-Training** from dense base model：
- Stage 1 (warmup): 1000 steps，batch=14×202,752 tokens，lr=5e-3
- Stage 2 (sparse adaptation): 20B tokens，复用mid-training的data和hyperparams

为什么这能work？因为DeepSeek-V3.2-Exp证明了**90%的attention entries在long context下是redundant的**。所以sparse attention是"lossless by construction"——不像sliding window或linear attention会丢弃long-range dependency。

Table 3: MLA vs DSA long-context对比
- MQ-NIAH-128k: 100.0 / 100.0 (tied)
- MV-NIAH-128k: 95.5 / 97.0 (DSA略好)
- SQuAD-128k: 79.7 / 86.0 (DSA更好)
- HotpotQA-128k: 66.3 / 63.0 (DSA略差)

DSA reduces attention compute 1.5-2x for long sequences。这对reasoning-heavy agents（128K contexts）特别重要。

### 1.5 Efficient Attention Ablation Study

这个ablation很informative。在GLM-9B上比较：
- Full Attention (baseline)
- SWA Interleave: 固定交替full + sliding window
- SWA Pattern (search-based): PostNAS启发的beam search，找最优layer subset
- GDN (Gated DeltaNet): linear attention with gated recurrence
- SimpleGDN: GDN的简化版，最大化pre-trained weight复用

**SWA Pattern search过程**：
- Beam size=8，每step优化2层
- GLM-9B共40层，约10步converge
- 在16K context上search，generalize到其他length
- 最终pattern: `SFSSFFSSSFFFFSSFSFFFFFFSFSFSSFSSFSFSSFSSS` (S=SWA, F=Full)

Table 5的核心发现：
- SWA Interleave在128K掉30.35点 (catastrophic)
- SWA Pattern只掉5.69点
- GDN掉11.28点
- SimpleGDN掉8.25点
- **DSA是lossless**，因为lightning indexer保留所有long-range dependencies

这build了intuition：**任何attention efficiency方法都有accuracy-efficiency trade-off，除了DSA——它是基于selection而非approximation**。

Table 6验证DSA在GLM-4.7-Flash上的结果：
- 128K: baseline 79.21 → DSA warmup 71.35 → full DSA 78.86 (几乎恢复)

---

## 2. Pre-Training: 28.5T tokens的data策略

### 2.1 三大数据类别的refinement

**Web data**：
- DCLM-based classifier (sentence embedding)识别高质量数据
- World Knowledge classifier (用Wikipedia和LLM-labeled data训练)用于long-tail knowledge的mid-low quality data蒸馏

**Code data**：
- Refreshed snapshots from major code hosting platforms
- 28% increase in fuzzily deduplicated unique tokens
- 修复Software Heritage的metadata alignment issues
- 更accurate的language classification pipeline
- 为low-resource languages (Scala, Swift, Lua)训练dedicated classifiers

**Math & Science**：
- Webpages, books, papers
- LLM scoring + chunk-and-aggregate scoring for long documents
- 严格filtering避免synthetic/AI-generated/template-based data

### 2.2 Mid-Training: 三阶段context extension

这是关键，progressive context extension：
- Stage 1: 32K context, 1T tokens
- Stage 2: 128K context, 500B tokens
- Stage 3: 200K context, 50B tokens (vs GLM-4.5的128K max)

**Software engineering data**：
- Concatenating repo files + commit diffs + GitHub issues + PRs + relevant source files
- 10M issue-PR pairs (放宽filtering criteria)
- 强化individual issue level quality filtering
- 160B unique tokens in issue-PR portion

**Long-context data**：
- Natural data: books, papers, knowledge-intensive documents
- Synthetic data: NextLong和EntropyLong启发的techniques
- Interleaved packing of similar texts (mitigate lost-in-the-middle)
- 200K stage加入MRCR-like data (multi-turn recall)

**Key empirical insight**：增加data diversity progressively enhances long-context performance。200K mid-training stage（在128K之后）甚至提升了128K window内的performance——说明longer context training有regularization效果。

---

## 3. Post-Training: 四阶段progressive alignment

整体pipeline：
1. SFT (introduce interleaved thinking modes)
2. Reasoning RL (math, science, code, TIR)
3. Agentic RL (coding + search agents)
4. General RL (human-style alignment)
5. On-policy cross-stage distillation (防止catastrophic forgetting)

### 3.1 SFT: 三种thinking mode

GLM-5支持三种thinking characteristics：
- **Interleaved Thinking**: 每次response和tool call前都thinking
- **Preserved Thinking**: coding agent场景下，multi-turn conversation中保留所有thinking blocks，reuse现有reasoning而非重新derive
- **Turn-level Thinking**: per-turn控制，simple request关闭thinking降延迟，complex task开启

Preserved Thinking这个idea很重要——避免multi-turn coding agent中information loss。错误trajectory保留但mask掉loss function，让model学会error correction without reinforcing错误action。

### 3.2 Reasoning RL: GRPO + IcePop

**核心loss function (Equation 1)**：

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}^{\text{infer}}(\cdot|x)} \left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \text{pop}(\rho_{i,t}, 1/\beta, \beta) \cdot \min\left(r_{i,t}\hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_{i,t}\right)\right]$$

变量解释：
- $x$: prompt
- $y_i$: 第i个sampled response (group size G=32)
- $\pi_{\theta_{old}}^{\text{infer}}$: 用于trajectory sampling的inference policy
- $\pi_{\theta_{old}}^{\text{train}}$: 用于gradient update的training policy
- $\rho_{i,t}$: training-inference mismatch ratio
- $r_{i,t}$: PPO-style importance ratio
- $\hat{A}_{i,t}$: group-normalized advantage

**Training-inference mismatch ratio**：
$$\rho_{i,t} = \frac{\pi_{\theta_{old}}^{\text{train}}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}^{\text{infer}}(y_{i,t}|x, y_{i,<t})}$$

这个ratio衡量training policy和inference policy在某个token上的概率差异。

**Pop operator**：
$$\text{pop}(\rho_{i,t}, 1/\beta, \beta) = \begin{cases} \rho_{i,t} & \text{if } 1/\beta \leq \rho_{i,t} \leq \beta \\ 0 & \text{otherwise} \end{cases}$$

如果mismatch ratio偏离$[1/\beta, \beta]$区间（β=2），这个token被mask掉，不参与gradient。

**Importance ratio**：
$$r_{i,t} = \frac{\pi_{\theta}^{\text{train}}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}^{\text{train}}(y_{i,t}|x, y_{i,<t})}$$

**Group-normalized advantage**：
$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(R_1, \ldots, R_G)}{\text{std}(R_1, \ldots, R_G)}$$

GLM-5的改动（vs原版IcePop）：
- **移除KL regularization**：accelerate RL improvement
- **Asymmetric clipping**: $\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}=0.28$（比lower bound更宽松的upper bound）

为什么asymmetric clipping？因为PPO的clip设计本意是限制policy update幅度，但实际中positive advantage的更新更常见，asymmetric允许exploration方向的更大update。

### 3.3 DSA在RL中的critical insight

这是paper里一个非常subtle但重要的engineering insight。

DSA有一个indexer网络，它对每个query token retrieves top-k most relevant KV entries（k=2048）。在RL训练中，**indexer的top-k indices必须deterministic**。

为什么？因为RL需要training-inference consistency：
- 训练时计算的advantage基于某个trajectory
- 如果inference时indexer选的top-k和training时不同，整个attention computation就变了
- 这类似MoE的routing replay问题：训练时必须reuse inference的expert routing

但DSA的k=2048远大于MoE的k（通常8），storing所有indices的cost巨大。

**Solution**: 用 `torch.topk` 而非CUDA-based非deterministic top-k。虽然慢一点，但deterministic，RL训练stable。其他非deterministic implementation（CUDA/TileLang）在RL几步后entropy剧烈下降，performance degradation严重。

**Critical engineering rule**: 整个RL过程freeze indexer parameters by default，使用torch.topk。

### 3.4 Agentic RL: 全异步decoupled框架

这是paper的核心contribution之一。

**Synchronous RL的问题**：长horizon agent rollout有严重的long-tail issue。一个trajectory可以run几小时，如果同步等所有rollout完成，GPU大量idle。

**Asynchronous RL设计**：
1. **Decoupled training和inference engines**：不同GPU设备
2. Inference engine持续生成trajectories
3. Trajectories数达threshold后送training engine update
4. Training engine每K gradient steps push新weights回inference engine
5. Weight update后reset optimizer（因为optimization problem变了）

**Server-based Multi-Task Rollout Orchestrator**：
- 每个task是独立microservice，注册到central orchestrator
- Orchestrator控制per-task rollout ratio和generation speed
- 统一message-list representation标准化所有task trajectories
- 支持1k+ concurrent rollouts
- 动态调整task sampling ratio

### 3.5 异步RL的稳定性mechanisms

#### 3.5.1 TITO (Token-in-Token-out) Gateway

**核心问题**：text-in-text-out会re-tokenize，引入boundary mismatch。

**TITO设计**：
- 训练pipeline直接消费inference engine产生的exact tokenization
- 保留exact action-level correspondence
- Actor可以emit trajectory fragments (token IDs + metadata) immediately
- 无需lossy text round-trip
- 无需post-hoc re-tokenization

**TITO Gateway实现**：intercept所有rollout generation requests，记录每个trajectory的token IDs和metadata，隔离cumbersome token ID processing from downstream agent rollout logic。

这个细节非常engineering-relevant：在streaming、truncation、interleaved actors的场景下，text round-trip会corrupt step alignment between actions和rewards/advantages。

#### 3.5.2 Direct Double-sided Importance Sampling

**问题**：异步RL中rollout engine可能在一个trajectory生成期间多次update，无法track exact $\pi_{\theta_{old}}$。maintain历史checkpoints $\{\pi_{\theta_{old}^{(1)}}, \ldots, \pi_{\theta_{old}^{(N)}}\}$ 不可行。

**Solution**：复用rollout时生成的log-probabilities作为behavior proxy，丢弃 $\pi_{\theta_{old}}$。

**Importance sampling ratio (Equation 4)**：
$$r_t(\theta) = \exp\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{rollout}}(a_t|s_t)\right)$$

**Calibration function (Equation 5)**：
$$f(x; \epsilon_\ell, \epsilon_h) = \begin{cases} x & \text{if } 1 - \epsilon_\ell < x < 1 + \epsilon_h \\ 0 & \text{otherwise} \end{cases}$$

**Optimization objective (Equation 3)**：
$$L(\theta) = \mathbb{E}_t\left[f(r_t(\theta), \epsilon_l, \epsilon_h) \hat{A}_t \log \pi_\theta(a_t|s_t)\right]$$

不同于PPO的asymmetric clipping（只clip importance ratio），GLM-5的double-sided calibration是直接mask掉trust region外的tokens。这比IcePop更简单，移除了 $\pi_{\theta_{old}}$ 的计算开销。

#### 3.5.3 Dropping off-policy samples

- Log rollout时的policy version sequence $(w_0, \ldots, w_k)$, $w_0 < \cdots < w_k$
- 当前version $w'$
- 如果 $w' - w_0 > \tau$，drop该sample（trajectory太stale）
- 环境crash的sample直接exclude（不是model capability问题）
- GRPO group里如果valid samples > half group size，pad with repeats；否则drop整组

#### 3.5.4 DP-aware routing for MoE inference

**问题**：multi-turn agent workloads中，同一rollout的sequential requests共享identical prefix，应该maximize KV reuse。但Data Parallelism下不同rank有独立KV cache，跨rank routing造成cache miss。

**Solution**：stateful routing layer
- 用consistent hashing把每个rollout ID map到fixed DP rank
- 跨turn稳定，eliminate cross-rank cache misses
- 轻量dynamic load rebalancing over hash space防止long-term imbalance

结果：prefill cost只proportional to incremental tokens而非total context length。

### 3.6 General RL: 三维度优化 + Hybrid reward

**Three dimensions**：
1. **Foundational correctness**: instruction following, logical consistency, factual accuracy, hallucination, fluency — minimal error rate
2. **Emotional intelligence**: empathy, insight, natural human communication
3. **Task-specific quality**: writing, text processing, QA, role-playing, translation

**Hybrid reward system**：
- **Rule-based rewards**: precise, interpretable，但只能express deterministic rules
- **Outcome Reward Models (ORMs)**: low variance, high efficiency，但susceptible to reward hacking
- **Generative Reward Models (GRMs)**: robust to exploitation，但high variance

**Human-in-the-loop alignment**：explicit引入high-quality human-authored responses作为stylistic anchors，避免converge到"model-like" verbose formulaic patterns。

### 3.7 On-Policy Cross-Stage Distillation

**问题**：multi-stage RL pipeline中sequential优化distinct objectives会造成前面stage获得的能力degradation。

**Solution**：on-policy distillation，前面stage的最终checkpoints作为teacher：

$$\hat{A}_{i,t} = \text{sg}\left[\log \frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t}|x, y_{i,<t})}{\pi_\theta^{\text{train}}(y_{i,t}|x, y_{i,<t})}\right]$$

变量解释：
- $\text{sg}$: stop gradient (`.detach()`)
- $\pi_{\theta_{\text{teacher}}}^{\text{infer}}$: teacher inference policy (从SFT、Reasoning RL、General RL的final checkpoints)
- $\pi_\theta^{\text{train}}$: current student training policy
- 训练prompts从对应teacher的RL training set采样，按比例混合

**关键implementation detail**：
- 用inference engine fetch teacher logits
- 未来计划migrate到training engine + uniform MQA mode
- GRPO group size=1（不再需要大group estimate advantage）
- Batch size=1024（大幅提升throughput）

这个公式很有意思：advantage就是teacher和student的log probability ratio，stop gradient让teacher不更新，student通过最大化这个ratio来match teacher distribution。这是KL divergence minimization的另一种形式（实际上等价于reverse KL的gradient）。

---

## 4. RL Infrastructure: slime Framework

### 4.1 Scaling Out

- **Highly customizable rollouts**: multi-turn interaction loops, tool invocation, environment feedback, verifier-guided branching — 无需改infrastructure
- **Server-based rollouts via HTTP APIs**: rollout servers和inference router通过标准HTTP API，外部agent frameworks可以直接调用，training backend不变

### 4.2 Tail-Latency Optimization

**关键insight**：RL rollout优化target是end-to-end latency而非aggregate throughput，因为slowest sample stall同步点。

**No-queue serving**: multi-node inference (EP64, DP64 over 8 nodes) provision distributed KV-cache，DP-attention防止跨rank复制KV。

**FP8 rollouts + MTP**: FP8减少per-token latency，MTP在小batch decoding（RL rollout的典型场景）特别有效。MTP对long-tail提供disproportionately large benefits。

**PD Disaggregation**: long-prefix prefills在multi-turn RL中频繁（conversation history, tool traces, code context）。混在一起会造成prefill preempt ongoing decodes。解耦后decodes保持稳定。

### 4.3 Heartbeat-Driven Fault Tolerance

- Rollout servers定期emit heartbeats
- Unhealthy servers proactively terminated and deregistered
- Retries自动routed到healthy servers
- 防止single-server incident中断rollout

---

## 5. Agentic Engineering: Environment Scaling

### 5.1 SWE Environments

基于RepoLaunch框架：
- 自动分析repo installation和dependency setup
- 生成test commands
- LLM生成language-aware log-parsing functions
- 提取Fail-to-Pass (F2P)和Pass-to-Pass (P2P) test cases
- **10k+ verifiable environments across 9 languages**: Python, Java, Go, C, CPP, JavaScript, TypeScript, PHP, Ruby

### 5.2 Terminal Environments

两条synthesis pipelines：

**From seed data**:
1. Task draft generation: LLM brainstorm from seed tasks
2. Concrete task implementation: construction agent生成Harbor formattasks (structured descriptions, Dockerized environments, test scripts)
3. Iterative refinement: refine agent按rubrics检查
- Docker construction accuracy >90%

**From web-corpus**:
1. 大规模code-relevant web pages收集 + quality classifier过滤
2. Stratified sampling across topic categories和difficulty levels
3. Coding agent按Harbor specification合成terminal task
4. **Closed-loop self-verification**: agent执行自己的output验证
5. 失败则iteratively diagnose和revise直到通过

这个closed-loop self-verification很有意思——构造agent本身就是自己的first-pass evaluator。

### 5.3 Search Tasks

**Web Knowledge Graph (WKG) Construction**:
- 从early-stage search agent trajectories收集URLs
- 2M+ high-information web pages
- LLM semantic parsing做entity recognition, noise filtering, structured extraction
- 持续update via entity alignment, attribute normalization, relation consolidation

**High-Difficulty Filtering (3-stage)**:
1. Remove questions that tool-free reasoning model在8次独立尝试中至少答对一次
2. Filter out early-stage agent with basic search几步可解的questions
3. Verification agent做bidirectional validation: candidate answers和ground truth consistency check

### 5.4 Context Management for Search Agents

**Key observation**: 超长context (e.g., >100k tokens)下model accuracy degradation严重。

**Keep-recent-k strategy**:
- Trajectory: $(q, r_1, a_1, o_1, \ldots, r_n, a_n, o_n)$
- 当history超过k rounds，fold老的observations
- $o_i \gets \text{Tool result is omitted}$ for $i = 1, \ldots, n-k$
- k=5: BrowseComp从55.3% → 62.0%

**Hierarchical Context Management**: keep-recent + Discard-all hybrid
- 当total context length超过T=32k，discard整个tool-call history，restart fresh
- 持续apply keep-recent
- 最终BrowseComp: 75.9% (SOTA in open-source)

Figure 8显示不同compute budget下，hybrid strategy一致优于纯Discard-all。

### 5.5 Slide Generation with Multi-Level Reward

这个section我觉得是paper里最creative的应用之一。

**Three-level reward**:

**Level-1: Static markup attributes**
- Positioning, spacing, color, typography, saturation
- Rules约束declarative attributes
- Hallucinated-image和duplicate-image detection

**Level-2: Runtime rendering properties**
- DOM node width, height, bounding boxes
- 分布式rendering service
- Reward hacking detection: hard truncation, excessive spacing manipulation (Figure 9)
- Refine renderer implementation消除exploitable loopholes

**Level-3: Visual perceptual features**
- Abnormal whitespace pattern detection
- Compositional balance和visual aesthetics

**Training strategy**:
- Dynamic sampling: probabilistically drop structurally trivial samples
- Token-level policy gradient loss
- Balancing strategy: 同一sample的different rollout outcomes跨多个training batches

**Rejection sampling + Masking refinement**:
- Best-of-N selection
- Masking: defective pages自动identify和mask，保留high-quality content
- 减少redundant regeneration overhead

**Empirical results**:
- 16:9 aspect ratio compliance: 40% → 92%
- Human eval win rates vs GLM-4.5: content 60%, layout 57.5%, aesthetics 65%, overall 67.5%

---

## 6. Chinese Chip Adaptation

### 6.1 W4A8 Mixed-Precision Quantization

为fit 750B参数到single Atlas 800T A3 machine：
- Attention和MLP blocks: W8A8 (INT8)
- MoE experts: W4A8 (INT4)
- QuaRot outlier suppression
- Flex_AWQ_SSZ scaling calibration

### 6.2 Fusion Kernels

**Lightning Indexer**: score calculation + ReLU + TopK融合到单kernel，NPU overlap compute with memory access

**Sparse Flash Attention**: 优化GLM-5的sparse patterns，TopK token selection和sparse attention computation并行

**MLAPO (Multi-head Latent Attention Pre-processing Optimization)**: 融合13个小pre-processing operators成一个"super operator"，Vector和Cube units并行

### 6.3 Specialized Inference Engine

**vLLM-Ascend和SGLang adaptations**:
- Asynchronous Scheduling: D2H sampling copies overlap with next decode step preparation
- Context Management: RadixCache (prefix sharing), Prefix Cache (extend KV to system RAM)
- Parallel Strategy: Attention DP + MoE EP + FlashComm (split AllReduce to hide latency)
- MTP: 提升NPU compute density

Performance: 单Chinese node达到dual-GPU国际cluster性能，long-sequence场景deployment cost降50%。

---

## 7. Evaluation: ARC Benchmarks

### 7.1 主要结果分析 (Table 7)

**Reasoning**:
- HLE: GLM-5 30.5 vs GLM-4.7 24.8 (+5.7), 接近GPT-5.2 (35.4), 不如Gemini 3 Pro (37.2)
- HLE w/ Tools: GLM-5 50.4, beats Claude Opus 4.5 (43.4)和Gemini 3 Pro (45.8)
- AIME 2026 I: 92.7 (tied with DeepSeek-V3.2)
- HMMT Feb 2025: 97.9 (beats Claude Opus 4.5 92.9)
- LongBench v2: 64.5 (second to Gemini 3 Pro 68.2)

**Coding**:
- SWE-bench Verified: 77.8 (open-source SOTA, vs Claude Opus 4.5 80.9)
- SWE-bench Multilingual: 73.3 (beats Gemini 3 Pro 65.0, GPT-5.2 72.0)
- Terminal-Bench 2.0 (Terminus-2): 56.21, verified version 60.7
- Terminal-Bench 2.0 (Claude Code): 61.1 (verified)
- CyberGym: 43.2 (vs Claude Opus 4.5 50.6)

**Agentic**:
- BrowseComp: 62.0 (SOTA across all models including proprietary)
- BrowseComp w/ Context Manage: 75.9 (SOTA)
- BrowseComp-ZH: 72.7 (beats Claude Opus 4.5 62.4)
- τ²-Bench: 89.7 (close to Claude Opus 4.5 91.6)
- Vending-Bench 2: $4,432 (close to Claude Opus 4.5 $4,967)
- GDPval-AA Elo: 1,409 (vs Claude Opus 4.5 1,400, GPT-5.2 1,462)

### 7.2 CC-Bench-V2: Real-world Agentic Engineering

**Frontend (Agent-as-a-Judge)**:
- HTML ISR: 38.9 (vs Claude Opus 4.5 52.2)
- React ISR: 34.6 (vs 39.7)
- Vue CSR: 77.1 (vs 74.3)
- Build Success Rate: 98% across all stacks
- **Insight**: GLM-5 meets most individual requirements但end-to-end task完成度有gap

**Agent-as-a-Judge validation**:
- Point-wise consistency: 94% agreement with human experts (130 check-items sampled)
- Ranking consistency: Spearman correlation 85.7% across 8 frontier models

**Backend**:
- Pass@1: 25.8 (vs Claude Opus 4.5 26.9, GLM-4.7 19.6)
- 85 tasks across 6 languages
- All-or-nothing unit test criterion

**Long-horizon**:
- Repo Exploration: 65.6 (beats Claude Opus 4.5 64.5)
- Chained Tasks: 52.3 (vs Claude Opus 4.5 61.6)
- Chained tasks的gap来自error compounding across chain

### 7.3 SWE-rebench (Table 9)

这是动态benchmark，mining fresh GitHub issues:
- GLM-5: 42.1% resolved rate
- vs Claude Opus 4.6 52.9%, GPT-5.2 (xhigh) 51.7%
- vs GLM-4.7 41.3%

GLM-5能generalize到新SWE问题，但与frontier proprietary models仍有gap。

### 7.4 General Abilities (Figure 11)

5个dimensions全improvement：
- Machine Translation (ZMultiTransBench, MENT-SNS)
- Multilingual Dialogue (LMArena, ZMultiDialBench)
- Instruction Following (IF-Badcase, IF-Bench, MultiChallenge)
- World Knowledge (SimpleQA, Chinese SimpleQA)
- Tool Calling (ToolCall-Badcase)

---

## 8. Easter Egg: Pony Alpha Experiment

GLM-5 anonymous release在OpenRouter上：
- 25% users guessed Claude Sonnet 5
- 20% guessed DeepSeek
- 10% guessed Grok
- Rest correctly identified GLM-5

这validates了intrinsic capability而非brand bias的evaluation。

---

## 9. 整体intuition总结

### 9.1 关键技术trade-offs

1. **DSA vs其他efficient attention**: DSA是lossless的，因为基于selection而非approximation。其他方法（SWA, linear attention）都有accuracy-efficiency trade-off。这是GLM-5能维持long-context fidelity的核心原因。

2. **Asynchronous RL的stability**: 异步带来throughput但引入off-policy bias。GLM-5的解法是组合多个mechanisms：
   - TITO保证token-level consistency
   - Direct double-sided importance sampling避免历史checkpoint tracking
   - Dropping off-policy samples过滤stale trajectories
   - DP-aware routing最大化KV reuse

3. **Multi-stage RL的forgetting problem**: 通过On-policy cross-stage distillation解决，teacher是前面stage的final checkpoint。

4. **Long-horizon agent的context management**: Keep-recent-k + Discard-all hybrid，T=32k threshold。

### 9.2 为什么GLM-5在coding agent上能compete with Claude Opus 4.5

1. **DSA支持128K+ context**: reasoning-heavy agent需要长context
2. **Asynchronous RL infrastructure**: 能scale到long-horizon agentic training
3. **10k+ verifiable SWE environments**: 真实engineering feedback
4. **Preserved Thinking**: multi-turn coding中information不loss
5. **MTP加速decoding**: 小batch RL rollout场景特别有效

### 9.3 主要limitation

从Table 7-9可以读出：
1. **Frontend ISR gap**: GLM-5在end-to-end task完成度仍不如Claude Opus 4.5
2. **Chained Tasks gap**: error compounding across chain，需要long-context consistency和self-correction
3. **Pure reasoning不如Gemini 3 Pro**: HLE 30.5 vs 37.2

### 9.4 Engineering lessons

1. **Deterministic ops matter in RL**: DSA indexer的top-k必须是torch.topk，任何非determinism都会destabilize training
2. **TITO是必须的**: text round-trip在async RL中corrupt step alignment
3. **Reward hacking detection要runtime**: 静态检查不够，需要DOM rendering提取真实attribute values
4. **Closed-loop self-verification**: construction agent本身做first-pass evaluator能ensure quality
5. **Long-context training有regularization效果**: 200K mid-training提升128K performance

---

## 10. 与同期frontier models的相对positioning

| Dimension | GLM-5 | Claude Opus 4.5 | Gemini 3 Pro | GPT-5.2 (xhigh) |
|----------|-------|-----------------|--------------|-----------------|
| Open weights | ✓ | ✗ | ✗ | ✗ |
| Agentic (BrowseComp) | **75.9** | 57.8 | 59.2 | 65.8 |
| Coding (SWE-bench) | 77.8 | **80.9** | 76.2 | 80.0 |
| Pure reasoning (HLE) | 30.5 | 28.4 | **37.2** | 35.4 |
| Long-context (LongBench v2) | 64.5 | 64.4 | **68.2** | 59.8 |
| Business (Vending-Bench 2) | $4,432 | $4,967 | **$5,478** | $3,591 |

GLM-5的positioning：**open-source SOTA in agentic engineering + competitive coding，gap in pure reasoning和long-horizon chained tasks**。

---

## 11. 思考与speculation

1. **DSA的潜力**: lightning indexer的top-k selection本质上是learned sparse pattern。如果能用indexer替代MoE的router，可能unify attention sparsity和expert sparsity的优化。

2. **Asynchronous RL的scale**: 1k+ concurrent rollouts + multi-task orchestrator，这接近于"agent school"的概念——大规模并行训练agents on diverse environments。

3. **On-policy distillation的theoretical意义**: Equation 2的advantage是log probability ratio with stop gradient。这等价于reverse KL minimization，但相比标准KL distillation，它在policy gradient framework下natural地整合，避免了mode collapse问题。

4. **Agentic engineering作为新paradigm**: paper的framing从"vibe coding"（human prompting）到"agentic engineering"（AI agents plan, implement, iterate autonomously）。这暗示future的coding agent training data将不再是human-written code，而是agent自己generated的long-horizon trajectories。

5. **Chinese chip adaptation的工程意义**: W4A8 mixed-precision + 融合kernels + specialized inference engine。这是一个full-stack co-design的case study，对其他Chinese GPU生态（如Ascend, Moore Threads等）有参考价值。

参考链接：
- GLM-5 GitHub: https://github.com/zai-org/GLM-5
- DeepSeek Sparse Attention: https://arxiv.org/abs/2512.02556
- GRPO: https://arxiv.org/abs/2402.03300
- SWE-bench: https://arxiv.org/abs/2310.06770
- τ²-Bench: https://arxiv.org/abs/2506.07982
- BrowseComp: https://arxiv.org/abs/2504.12516
- Vending-Bench: https://arxiv.org/abs/2502.15840
- Humanity's Last Exam: https://arxiv.org/abs/2501.14249
- SWE-rebench: https://arxiv.org/abs/2505.20411
- RepoLaunch (SWE-bench Live): https://arxiv.org/abs/2505.23419
- RULER: https://arxiv.org/abs/2404.06654
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Multi-token Prediction: https://arxiv.org/abs/2404.19737
- PostNAS / Jet-Nemotron: https://arxiv.org/abs/2508.15884
- QuaRot: https://arxiv.org/abs/2404.02558
- MTP / Speculative decoding: https://arxiv.org/abs/2211.17192
- DAPO: https://arxiv.org/abs/2503.14476
- On-policy distillation (Thinking Machines Lab): https://thinkingmachines.ai/blog/on-policy-distillation
- DAPO: https://arxiv.org/abs/2503.14476

---

希望这builds你的intuition on GLM-5的关键technical contributions。整个paper的核心narrative是：**通过architecture-level efficiency (DSA + MLA + MTP) + system-level efficiency (async RL + TITO + DP-aware routing) + data-level scaling (10k+ verifiable environments)的full-stack co-design，把open-source models推到agentic engineering的frontier**。
