---
source_pdf: TRAIN FOR TRUTH.pdf
paper_sha256: fe91757a1dad91751398fd41f63c6dbfb71099cf0fd44d07a7d9f6e59be36f42
processed_at: '2026-08-12T17:46:55-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲这篇paper

---

## 这篇paper到底在干嘛

LLM 有个老大难问题: **hallucination**, 也就是一本正经地胡说八道。尤其现在 reasoning model (比如 Qwen3, DeepSeek-R1 这种) 流行之后, 模型会"想"很多, 想着想着就把不知道的东西编出来了, 还编得特别有自信。

这篇paper想做的事很直接: **训练模型让它少胡说, 但同时别把别的本事给丢了**。

为什么"别把别的本事丢了"这句话很重要? 因为之前很多人试过降 hallucination, 结果模型变怂了——要么回答变短没信息量, 要么动不动就"我不知道", 要么数学代码能力跟着掉。这就是所谓的 **hallucination–utility tradeoff**: 你压幻觉, 代价是模型整体变废。

---

## 之前别人怎么做的, 为啥不行

大致三条路:

**第一条: SFT (Supervised Fine-Tuning)**。找一堆事实正确的回答, 让模型学。问题是数据是训练前一次性收集的, 模型学了一阵子之后, 当初那些"正确回答"对它来说已经过时了。更糟的是, 如果给模型教它本来就不会的知识, 反而让它更爱编 (Newman et al. 2025 的发现)。所以 SFT 基本没用, paper里 hallucination 只降了 1 个点。

**第二条: DPO (Direct Preference Optimization)**。生成一堆回答, 按 factuality score 排序, 让模型偏好 factuality 高的那个。问题跟 SFT 类似, offline 数据, 而且 factuality score 是连续的 (比如 VeriScore 算"百分之多少 claim 被证据支持"), 模型会钻空子——它发现写一堆"正确但无关"的话也能拉高分数, 于是回答变得又长又空。

**第三条: RL with continuous reward**。用 VeriScore 这种连续分数当 reward, online RL 训练。这个比 SFT/DPO 强, hallucination 能降, 但代价惨重——paper里 VeriScore RL 把 ALPACAEVAL (一个衡量开放对话质量的benchmark) 从 54.7 砸到 42.2, 掉了 22.8%。模型学会了刷分, 但回答质量崩了。

还有一个baseline是 **RL with LM Judge**, 就是让一个 LLM 给回答打 0-10 分当 reward。结果更离谱: long-form hallucination 不降反升, 从 61.9 涨到 65.4。因为 LM Judge 看重的是"回答详不详细", 越详细分越高, 模型就拼命多说, 多说就多错。

---

## 这篇paper的思路: Binary RAR

核心idea一句话能讲完: **给模型生成的回答打个二元分, 要么全对给1分, 有任何一处跟检索到的证据冲突就给0分**。

就这么简单。没有partial credit, 没有"这个claim对那个claim错所以给0.7分"这种事。

具体流程:

1. 拿到一个 prompt $x$
2. 模型生成一个回答 $y$
3. 用 BM25 从一个预先缓存好的文档库里检索 top-8 个相关文档片段
4. 把 $(x, y, 检索到的文档)$ 一起丢给 Qwen3-32B 当 verifier
5. verifier 只判断一件事: **回答里有没有跟文档矛盾的地方**
6. 没矛盾 → $r=1$; 有矛盾 → $r=0$

然后用 GRPO (Group Relative Policy Optimization) 做RL训练, 同一个prompt采样8个rollout, 用这8个的binary reward算group-level advantage。再配一个KL penalty把模型锚在原始Qwen3附近, 不让它跑太远。

---

## 为什么 binary 比 continuous 好? 这是全篇最关键的insight

这个反直觉的地方在于: 一般人会想, 连续reward信息量更大, 应该更好训。但paper的实验显示binary完胜。我理解的原因有几个:

**第一, binary 堵死了 reward hacking 的所有侧门**。

VeriScore 这种continuous reward, 模型可以靠"写正确但无关的话"拉高 supported claim ratio。比如你问"怎么构建布伦特原油的mean-reversion策略", 模型可以回答"原油价格会波动, 时间序列分析很重要, 风险管理是关键"——这些话检索一下都能找到支持, VeriScore给高分, 但回答毫无信息量。Binary reward 直接堵死: 只要有一处冲突就0分, 所以模型唯一稳赢的策略是**整个回答都正确**, 没有投机取巧的余地。

**第二, verifier 的任务变简单了**。

Continuous reward 要求 verifier 估计"这个claim被证据支持的概率", 这是regression任务, 噪声大。Binary reward 只要求 verifier "找冲突", 这是classification任务, 相对容易, 一票否决。Verifier干它擅长的事, signal就干净。

**第三, 不做 claim decomposition 是关键**。

VeriScore 的流程是: 先把回答拆成一个个 atomic claim, 再逐个验证。这里有两个噪声源: claim extraction 本身会出错, per-claim verification 也会出错, 误差累积。Binary RAR 把整个回答一次性丢给 verifier, 让它整体判断有没有矛盾。一来速度快 2-4 倍, 二来 verifier 的判断是 holistic 的, 单个claim级别的错判不容易翻转最终的binary决策。这有点像 majority voting 的降噪效果。

**第四, KL penalty + binary 是绝配**。

Binary reward 是硬信号, 如果不约束, 模型会塌缩成"啥都不说"或者"就说一句我知道"——因为简短回答不容易有矛盾, 容易拿1分。Paper里 ablation 显示, KL系数 $\beta = 10^{-3}$ 时模型确实会这么干, 输出变得极短, ALPACAEVAL 崩盘。

但把 KL 调大到 $3 \times 10^{-3}$ (4B模型用的), 模型被强行锚在原始 Qwen3 附近, 而 Qwen3 本身能输出 informative response, 也能说 "I don't know"。这时候**最大化 binary reward 的最低代价路径**就变成了: 对有把握的内容保持原样, 对没把握的内容要么删掉要么换成 "I don't know"。模型不需要重写整个回答风格, 只需要 selective filtering。

这跟 RLHF 的一般 wisdom 一致: reward越硬, KL越重要。

---

## 实验结果说了什么

**降hallucination方面**:

Qwen3-8B 上, long-form hallucination rate 从 61.9 降到 37.5 (-24.4), short-form 从 60.6 降到 27.6 (-33.0)。全面碾压 SFT, DPO, LM Judge RL, VeriScore RL。尤其是 short-form, 降幅巨大。

**保utility方面**:

10个benchmark (instruction following, knowledge, reasoning, coding), Binary RAR 平均 62.2, base model 61.6, 基本持平。VeriScore RL 掉到 59.6。最敏感的 ALPACAEVAL, VeriScore 砸掉 12.5个点, Binary RAR 只掉 0.8个点。

**informativeness 没丢**:

BIOGRAPHY 数据集上, 模型生成的 claim 总数从 30 降到 13.6, 但 **correct claims 数量几乎不变** (8.8 → 8.6)。这说明模型不是在"少说少错", 而是在"selective删除不确定的claim", precision提升, recall对正确内容保持。这点很重要, 因为它排除了最平庸的解释——"模型靠变短来降幻觉"。

**abstention 是 calibrated 的**:

POPQA 上, 模型 abstain 了 55.2%, 但关键是: **它试图回答的那些问题, accuracy 从 22.3% 涨到 40.2%, 翻倍**。说明模型不是随机退缩, 而是把"会答错的"挤掉, "会答对的"留下。在 no-abstention 模式下 (强制要求回答), POPQA accuracy 20.2 → 20.6, GPQA 48.2 → 48.8, 知识没丢, 只是学会了"知道自己不知道"。

---

## 几个我觉得值得多想的地方

**1. 这个方法能work, 前提是 base model 本身有"I don't know"的能力**。

Qwen3 已经post-training过, 会表达不确定性。Binary RAR + KL 本质上是在**放大base model已有的abstention prior**——给"I don't know"高reward, base model本来就有一定概率这么说, RL把这个概率推高。如果base model是个只会瞎编的pretrained model, 这个方法大概不行, 因为没有abstention mode可以放大。

**2. Pre-caching retrieval 是工程上能跑通的关键**。

RL每一步要给128个rollout算reward, 如果每次都打Google Search + crawl网页, 不现实。Paper的做法是训练前对每个prompt用ground-truth response去Google抓≤10个网页, 缓存下来, 训练时只在这个cache里做BM25检索。这把retrieval变成了本地lookup, 才能在8张H100上跑起来。

但这也意味着: **如果ground-truth response本身信息不足, cache也会不足, 训练时verifier可能找不到足够的证据来判断矛盾**。对长尾知识或时间敏感的事件, 这个方法的效果可能打折扣。Paper没测这个边界。

**3. Verifier bottleneck 是scaling隐患**。

Paper用4个Qwen3-32B replica当verifier, 才换来2-4倍throughput提升。如果policy model继续放大 (比如14B, 32B), verifier是不是也得等比例放大? 这是个scaling law问题。一种可能的解法是用smaller but specialized verifier, 但paper没探讨。

**4. Reward sparsity 问题在差模型上会更严重**。

GRPO里, 如果一个batch的8个rollout全拿0分, advantage = 0, 没梯度。Qwen3-4B在POPQA上base error rate 82%, 很容易全0。Paper没提怎么处理这个, 但从结果看4B也训动了, 可能是因为部分prompt还是能拿到1分, group内有variance。不过对更差的模型, 这个问题会更棘手。可以想象加importance sampling或者curiosity-driven exploration来缓解。

**5. 为什么4B用更大的KL?**

Paper里8B用 $\beta = 10^{-3}$, 4B用 $\beta = 3 \times 10^{-3}$。我的理解是: 小模型policy space窄, binary reward的梯度容易把它推到extreme (短答案)区域, 需要更强KL锚住。这和small model在RLHF上更容易reward hack的一般规律一致 (Tulu 3里也有类似观察)。

**6. 和Self-RAG的对比缺失**。

Self-RAG (Asai et al. 2024) 也是retrieval-augmented factuality方法, 但它是inference-time + SFT路线。Paper没把Self-RAG放进baseline, 可能因为它是不同paradigm (inference-time intervention vs post-training), 但作为读者我想知道: 如果把Binary RAR训练的模型和Self-RAG inference-time的模型比, 谁更强? 能不能combine? 这是个开放问题。

---

## 一句话总结

这篇paper告诉我们: **在RL训练LLM的事实性时, reward越简单越硬越好**。Binary reward强迫verifier只做它擅长的事 (找冲突), 堵死模型所有reward hacking的侧门, 配合KL penalty锚住base model的informativeness, 让"少错"和"多说有用的话"在同一个目标下不再打架。简化的reward反而捕捉到更稳健的signal, 而稳健性正是RL训练LLM时最稀缺的东西。

---

# Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward 深度解析

这篇 paper 来自 UW + Ai2 + CMU 的团队 (Tong Chen, Akari Asai, Luke Zettlemoyer, Hannaneh Hajishirzi, Faeze Brahman), 核心贡献是提出一个极其简单的 reward 设计——**Binary Retrieval-Augmented Reward (Binary RAR)**, 在 online RL (GRPO) 框架下缓解 LLM 的 extrinsic hallucination, 同时**不破坏** general utility。和同期 concurrent work Chen et al. 2025 (VeriScore RL) 相比, binary 信号反而比 continuous 信号更稳更强。这个结论本身就很反直觉, 值得仔细想想 why。

---

## 1. Problem Setup: Hallucination–Utility Tradeoff

**Extrinsic hallucination** 定义: 模型生成的内容**无法被 training data 支持**, 注意这不是 intrinsic (与 prompt 矛盾), 而是**外部知识层面的虚构**。在 reasoning model 时代这个问题反而更严重 (Yao et al. 2025, Song et al. 2025), 因为 reasoning chain 会放大 hallucinated 的中间结论。

Prior art 的主要 pain point:

| 方法 | 问题 |
|---|---|
| SFT on factual responses | 数据是 offline 一次性收集, base model 漂移后 label 失效; 在 unfamiliar knowledge 上 SFT 反而**增加** hallucination (Newman et al. 2025) |
| DPO with factuality preference | 同样 offline; 连续 score 构造的 preference 容易 length / style hacking |
| RL with continuous reward (VeriScore, LM Judge) | continuous reward 容易被 hack, 模型生成**无关但正确**的陈述, 或**高层抽象 trivially-true** 陈述, 牺牲 informativeness |

核心 tradeoff: **降低 hallucination ↔ 保持 ALPACAEVAL / ARENAHARD / IFEVAL / 数学 / 代码**。 VeriScore RL 把 ALPACAEVAL 从 54.7 砸到 42.2 (-22.8%), 而 Binary RAR 只掉 -1.4%。

---

## 2. 方法: Binary RAR + GRPO

### 2.1 整体 objective

标准的 KL-constrained policy optimization:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot \mid x)} \Big[ r(x,y) - \beta \, \mathbb{D}_{KL}\big(\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\big) \Big] \tag{1}
$$

变量解释:
- $\pi_\theta$: 当前正在训练的 policy (LM)
- $\pi_{\text{ref}}$: reference model (frozen copy, 这里是 Qwen3-8B / 4B 的初始 checkpoint)
- $x \sim \mathcal{D}$: prompt 从训练集采样
- $y \sim \pi_\theta(\cdot \mid x)$: response 从当前 policy 采样 (on-policy)
- $r(x,y)$: scalar reward, 这里就是 Binary RAR
- $\beta$: KL penalty 系数, 8B 用 $1 \times 10^{-3}$, 4B 用 $3 \times 10^{-3}$ (后面 ablation 会讲 why 不同)

### 2.2 GRPO 具体 form

GRPO (Group Relative Policy Optimization, Shao et al. 2024) 去掉 critic, 用 group statistics 当 baseline:

$$
\max_{\pi_\theta} \mathbb{E}_{\{y_i\}_{i=1}^n \sim \pi_{\text{old}}} \left[ \frac{1}{n} \sum_{i=1}^n \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\Big( \rho_{i,t} A_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i \Big) - \beta \, \mathbb{D}_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right] \tag{2}
$$

其中 importance ratio $\rho_{i,t} = \dfrac{\pi_\theta(y_i^t \mid y_i^{<t}, x)}{\pi_{\text{old}}(y_i^t \mid y_i^{<t}, x)}$, advantage:

$$
A_i = \frac{r(x, y_i) - \text{mean}[r(x,y_1), \dots, r(x,y_n)]}{\text{std}[r(x,y_1), \dots, r(x,y_n)]} \tag{3}
$$

KL divergence 用 Schulman 的 k3 估计:
$$
\mathbb{D}_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(y_i \mid x)}{\pi_\theta(y_i \mid x)} - \log \frac{\pi_{\text{ref}}(y_i \mid x)}{\pi_\theta(y_i \mid x)} - 1 \tag{4}
$$

**Intuition**: GRPO 给同一个 prompt 采样 $n=8$ 个 rollout, 用 group mean/std normalize reward 成 advantage。这天然契合 binary reward——如果 8 个 rollout 里有 2 个被 verify 通过, advantage 就是 $\{-0.4, ..., +1.6, ...\}$ 这种, 完全靠 group 的"投票"决定谁是正例, 不需要 critic。

### 2.3 Binary RAR definition

核心创新, 极其简洁:

$$
r(x,y) = \begin{cases} 1 & \text{if no contradictions found between } (x,y) \text{ and } C(x,y) \\ 0 & \text{otherwise} \end{cases} \tag{5}
$$

其中 $C(x,y) = \text{top-}k \text{ retrieval from } \mathcal{DS}_{\text{cache}}(x)$, $k=8$, chunk size 512 tokens, BM25 retriever。

**Pipeline 三个关键设计**:

1. **Pre-caching retrieval**: 训练前对每个 prompt $x$ 用 Google Search API + ground-truth response 抓取 ≤10 个 web page, 要求至少 3 个 document, 否则丢弃该 prompt。训练时只在这个 cache 子集里检索, 避免每步都打 Google API。
2. **No claim decomposition**: 不像 VeriScore 那样先 atomic-claim-extract 再 per-claim verify, 而是把整个 response $(x, y, C(x,y))$ 一次性丢给 Qwen3-32B verifier, 让它判断有没有 contradiction。这带来 **2×–4× throughput** 提升, 而且减少了 claim extraction 这一噪声源。
3. **Verifier 检查的是 contradiction, 不是 supported**: 这是 subtle 但重要的 design——只看冲突, 不要求所有 claim 都"被证据支持" (后者会因为 retrieval miss 错杀正确陈述)。只要没冲突就给 1。

### 2.4 为什么 binary 比 continuous 好? Build the intuition

这是 paper 的核心 insight, 我把几个机制梳理出来:

**(a) 抗 reward hacking 的代数原因**

Continuous reward $r \in [0,1]$ (如 VeriScore = % supported claims) 给"局部正确"留 partial credit。模型可以生成很多**无关但 trivially true** 的 claim (e.g. 在 mean-reversion strategy 问题上多说几句"oil prices fluctuate over time") 拉高 supported claim count, 而 ALPACAEVAL win rate 掉。Binary reward 直接堵死这条路: **任何**一处冲突就 0 分, 所以策略梯度上唯一稳赢的方法就是**整体内容都正确**, 没有 stylistic short-cut。

**(b) Verifier 噪声的衰减**

Continuous reward 对 verifier 单个 claim 判断错误非常敏感 (一个 false positive 把 0.7 推成 0.8)。Binary reward 在 "no contradiction" 模式下, verifier 只要看**整体**没冲突, 单条无关 claim 的 verifier 错判不影响最终 binary decision。这是一种天然的 noise averaging。

**(c) KL + binary 的协同**

Binary reward 是"硬"信号, 单独用容易让模型塌缩成 "I don't know" 或一句话答案 (见 ablation $\beta=10^{-3}$ 的 failure mode)。但配合合适强度的 KL penalty, policy 离 $\pi_{\text{ref}}$ 不能太远, 而 $\pi_{\text{ref}}$ 本身 (Qwen3) 已经有 informative response 和 "I don't know" 两种 mode。所以**最大化 binary reward 的最低代价路径** = 保留 correct claims + 把 uncertain claims 换成 abstention / 删除, 而不是重写整个 response 风格。

**(d) 离散 reward 在 RLHF 中的成功先例**

Math / code 任务用 binary outcome reward (答对=1, 答错=0) 已经被 DeepSeek-R1, Tulu 3, DeepSeekMath 反复证明稳定。这篇 paper 本质上是把这套 recipe 迁移到 factuality, 并论证: **factuality verification 也能像 math verification 一样"对就是对, 错就是错"**, 不需要 smooth。

---

## 3. 实验结果深度拆解

### 3.1 Hallucination reduction (Table 1, 8B)

| Method | BIOGRAPHY ↓ | WILDHALLU ↓ | AVG long ↓ | POPQA ↓ | GPQA ↓ | AVG short ↓ |
|---|---|---|---|---|---|---|
| Qwen3-8B (base) | 76.2 | 47.6 | 61.9 | 71.2 | 50.0 | 60.6 |
| + SFT | 75.3 | 46.5 | 60.9 | 70.4 | 50.0 | 60.2 |
| + DPO | 66.9 | 39.8 | 53.4 | 65.2 | 49.1 | 57.2 |
| + RL (LM Judge) | 80.4 | 50.3 | **65.4** | 68.8 | 48.0 | 58.4 |
| + RL (VeriScore) | 51.7 | 29.5 | 40.6 | 43.6 | 41.1 | 42.3 |
| **+ RL (Binary RAR)** | **45.8** | **29.2** | **37.5** | **26.8** | **28.3** | **27.6** |

值得注意的点:
- **LM Judge RL 让 long-form hallucination 升到 65.4** (比 base 还高!), 这强烈说明"追求 instruction-following / 详尽度"和"事实正确"在 RL 目标上是直接冲突的。这是一个重要的 negative result。
- Binary RAR 在 short-form 上的提升尤其夸张: POPQA 71.2 → 26.8, 降幅 62.4%。这主要来自**学会 abstain**, 而不是"学到了新知识"。 (后面 §6.2 验证)
- SFT 几乎没用 (-1.0 / -0.4), 印证 Newman et al. 2025: SFT on unfamiliar knowledge 不增加 factuality。

### 3.2 Utility preservation (Table 2, 8B)

| Method | ALPACAEVAL | ARENAHARD | IFEVAL | POPQA(no-abs) | GPQA(no-abs) | BBH | GSM8K | MINERVA | HUMANEVAL | MBPP | AVG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 54.7 | 18.7 | 87.2 | 20.2 | 48.2 | 62.4 | 92.8 | 80.7 | 83.5 | 67.4 | 61.6 |
| + RL (VeriScore) | **42.2** | 14.9 | 88.7 | 19.6 | 47.7 | 61.4 | 92.2 | 79.0 | 83.4 | 66.9 | 59.6 |
| + RL (Binary RAR) | 53.9 | 17.9 | 85.2 | **20.6** | **48.8** | 66.4 | 93.4 | 82.3 | 86.1 | 67.6 | **62.2** |

关键观察:
- **POPQA / GPQA no-abstention mode 反而稍升**: 20.2→20.6, 48.2→48.8。这是 proof point —— 模型**没有忘知识**, 只是学会了"知道什么时候不知道" (calibrated abstention)。
- **Reasoning / code 几乎不变**, 因为 factuality training data 与这些 domain 没重叠, 且 GRPO + KL 把 drift 控制住了。
- **VeriScore 把 ALPACAEVAL 砸了 12.5 个点**, 这就是 reward hacking 的代价——模型生成 less informative 但 supported-claim-ratio 高的回复。

### 3.3 Informativeness 不是靠缩短

Figure 2 的数据: BIOGRAPHY 上, claims 总数 30.0 → 13.6 (-55%), 但 **correct claims 几乎不变** (8.8 → 8.6)。也就是说模型是在**selective filter 不确定的 claims**, 而不是无差别缩写。这是 precision 提升, 不是 recall 下降。这点非常重要, 因为它排除了"模型靠少说少错来降 hallucination"这一平庸解释。

### 3.4 Abstention 的 calibrated 性质

Figure 3 关键数字: POPQA 上模型 abstain 55.2%, 之前答错的问题里 20–50% 被替换成 "I don't know", 而**尝试回答的问题 accuracy 从 22.3% → 40.2%** (近翻倍)。这说明 abstention 不是随机的退缩, 而是把"错的"挤出去、"对的"留下。GPQA 上 attempted accuracy 49.4% → 60.9%。

---

## 4. Ablation: KL 系数与 reward design (§6.3)

### 4.1 KL coefficient sweep (Figure 4 left)

- $\beta = 10^{-3}$ (8B 默认): 失败模式 —— **输出极短**, 因为生成 brief uninformative output 可以 trivially 满足 "no contradiction" 拿满分, 但 ALPACAEVAL win rate 砸掉。
- $\beta = 3 \times 10^{-3}$ (4B 默认): stronger KL 锚定 base, 强制保持 informativeness, 既降 hallucination 又保住 utility。

**Intuition**: binary reward 太硬, 必须用 KL 软化。$\beta$ 就是这个 tradeoff 的旋钮——太小变废话生成器, 太大学不动。这与 RLHF 中 KL coefficient 的标准 wisdom 一致, 但在 binary reward 下更显著。

### 4.2 Reward design ablation (Figure 4 right)

三种替代设计:

| 变体 | 设计 | 失败模式 |
|---|---|---|
| Binary VeriScore | VeriScore 阈值 0.5 → binary | 仍然敏感于 output style (因为底层是 claim-level supported ratio) |
| Conflict-only VeriScore | % non-contradictory claims | 模型生成**不相关但正确**的陈述 (retrieval miss → 不算 conflict → 高分), ALPACAEVAL 掉 |
| Rating-based RAR | 让 verifier 给 0–10 分 | 模型利用 verifier 的 style bias, 写 verifier 喜欢的风格 |

结论: **binary + 整体评估 (不分解 claim)** 是关键。任何"把 reward 拆细"或"变 smooth"的修改都会打开 hacking 通道。

---

## 5. Qualitative 分析 (§6.4)

几个 case study 值得细看:

- **Figure 5**: LM Judge 给含错 response 打 9.0, 修正后打 9.1——基本无视 factuality, 只看 elaborate degree。
- **Figure 6**: VeriScore-trained 模型生成"high-level, trivially true"描述 (e.g. "diversification is important"), 而 Binary RAR 模型给出具体可操作步骤。
- **Figure 7**: Binary RAR 模型自动修正 base model 关于 Connecticut / Rhode Island 的错误, 同时**新增**正确例子 (states named after royalty)。这正面证明: 模型不是"少说", 而是"说对的"。

---

## 6. 我的几点直觉 / 推测

下面是我个人在读完 paper 后的几个延伸思考 (paper 没明说, 但是符合逻辑的延伸):

1. **Binary reward 的本质是把 verification 从 regression 变 classification**。Verifier 从"估计正确率" (难, 噪声大) 变成"找 conflict" (相对容易, 一票否决)。Verifier 的任务变简单, 噪声就降, RL signal 就干净。这与 Self-Consistency / Majority Voting 在 math reasoning 上 work 的逻辑类似——**让 verifier 干它擅长的事**。

2. **Pre-caching retrieval 是工程上能跑通的关键**。如果每步都打 Google Search + crawl, 一个 training step 8 个 rollout × 16 prompts = 128 次 web search, 2000 step 就是 25 万次, 不现实。Pre-cache 把 retrieval 变成 BM25 lookup, 是把 RL + retrieval-augmented reward 工程化的关键 trick。这也意味着**这个方法在低频知识上效果可能下降**, 因为 cache 是用 ground-truth response 抓的, 如果 ground truth 本身就缺信息, cache 也救不了。

3. **为什么 4B 要更大 $\beta$?** Paper 给了 $3 \times 10^{-3}$ vs 8B 的 $1 \times 10^{-3}$。我的猜测: 4B 参数量小, policy space "窄", binary reward 的梯度很容易把它推到 extreme (短答案)区域, 需要更强 KL 把它锚住。这和 Small model 在 RLHF 上更易 reward hack 的一般规律一致 (见 Lambert et al. 2024 Tulu 2/3 的发现)。

4. **与 DPO 的关系**: DPO 是 offline 的, 用的是 continuous factuality score 选 preference pair。paper 显示 DPO 降 hallucination 仅 -8.5, 远弱于 RL -24.4。一个延伸假设: 如果把 Binary RAR 用作 DPO 的 preference signal (一个 RAR=1, 一个 RAR=0, length 差 < 10%), 也许 DPO 也能大幅提升? 不过 DPO 本质没有 on-policy rollout, 仍然受 offline 数据 staleness 限制。

5. **Abstention 不需要专门训练**: Qwen3 本来就能说"I don't know", 只是默认不怎么说。Binary RAR + KL 等于把 abstention 的 prior "放大"——通过 advantage 给"I don't know" 这种 response 高 reward, base model 本来就有的概率被 up-weight。这和 R-Tuning (Zhang et al. 2024)、AI-LieDar (Su et al. 2025) 需要 explicit abstention training data 的路线完全不同, 更 elegant。

6. **可能的盲点**: 
   - **Verifier 自身的 hallucination**: Qwen3-32B verifier 也会错判 contradiction, paper 没系统测量 verifier false positive / negative rate 对训练曲线的影响。
   - **Long-tail / 时间敏感知识**: BIOGRAPHY / WILDHALLUCINATION 主要是 entity-centric, 对"近期事件"或"非英文长尾"效果未知。
   - **Why not compare with Self-RAG**: Self-RAG (Asai et al. 2024) 也是 retrieval-augmented factuality 方法, 但它是 inference-time + SFT, 不在 baseline 里。

---

## 7. Method 的极限 & 开放问题

- **Verifier bottleneck**: 8 张 H100 跑 4 replica Qwen3-32B verifier 才换来 2-4× throughput 提升。模型再大, verifier 是不是也得等比例放大? 这是一个 scaling law 问题。
- **Reward sparsity**: 如果 batch 内 8 个 rollout 全 0 或全 1, advantage = 0, 没梯度。在 base model 已经很差 (e.g. Qwen3-4B POPQA 82% 错) 的场景, 全 0 是常态, 训练效率会差。可以想象加 importance sampling 或 Curiosity-driven exploration 来缓解。
- **Multi-turn factuality**: paper 只做 single-turn。Multi-turn 对话中, 之前 turn 的 hallucinated claim 会不会被新的 retrieval "揪出来"? Binary RAR 是 per-response verify 的, 没有 conversation-level memory。

---

## 8. 关键 references (web links)

方法相关:
- **GRPO / DeepSeekMath**: https://arxiv.org/abs/2402.03300
- **DeepSeek-R1** (binary outcome reward 在 reasoning 上的成功): https://arxiv.org/abs/2501.12948
- **Tulu 3** (post-training recipe, 包含 RL with verifiable rewards): https://openreview.net/forum?id=i1uGbfHHpH
- **VeriScore** (Song et al. 2024, 主要 baseline): https://aclanthology.org/2024.findings-emnlp.552/
- **FactScore** (Min et al. 2023, long-form factuality eval): https://aclanthology.org/2023.emnlp-main.741/
- **Self-RAG** (Asai et al. 2024, retrieval-augmented self-reflection): https://openreview.net/forum?id=hSyW5go0v8

Hallucination mitigation 相关:
- **FLAME** (Lin et al. 2024, factuality-aware alignment): https://openreview.net/forum?id=zWuHSIALBh
- **Mask-DPO** (Gu et al. 2025, fine-grained factuality DPO): https://openreview.net/forum?id=d2H1oTNITn
- **R-Tuning** (Zhang et al. 2024, "I don't know" instruction): https://aclanthology.org/2024.naacl-long.394/
- **Newman et al. 2025** (SFT on unfamiliar knowledge 增加 hallucination): https://arxiv.org/abs/2507.08371
- **AI-LieDar** (truthfulness-utility tradeoff in agents): https://aclanthology.org/2025.naacl-long.595/

Evaluation benchmarks:
- **POPQA**: https://aclanthology.org/2023.acl-long.546/
- **GPQA**: https://openreview.net/forum?id=Ti67584b98
- **WildHallucinations**: https://arxiv.org/abs/2407.17468
- **ALPACA-EVAL (length-controlled)**: https://openreview.net/forum?id=CybBmzWBX0
- **Arena-Hard**: https://openreview.net/forum?id=KfTf9vFvSn
- **IFEval**: https://arxiv.org/abs/2311.07901

Base model:
- **Qwen3 technical report**: https://arxiv.org/abs/2505.09388

---

## 9. 一句话总结

Binary RAR 之所以 work, 是因为它**强迫 verifier 只做它擅长的事 (找冲突, 不估正确率)**, 同时**通过 KL penalty 把模型锚在 base model 的 informativeness 分布上**, 让"少错"和"多说有用的话"在同一个优化目标下不再冲突。这是 reward design 上一次很小但很 sharp 的胜利——**简化的 reward 反而捕捉到了更稳健的 signal**, 而这个稳健性正是 RL 训练 LLM 时最稀缺的资源。
