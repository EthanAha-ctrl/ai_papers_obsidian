---
source_pdf: DeReason A Difficulty-Aware Curriculum Improves Decou.pdf
paper_sha256: b655b0ca3fe74b4d074a8970591624b88c67a786babd2a43d3e6aaf1b1b5945a
processed_at: '2026-08-03T19:58:10-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DeReason

## 这篇 paper 在纠结啥

最近两年大家一窝蜂搞 RLVR（用可验证的 reward 训 reasoning），DeepSeek-R1、o1 都是这个路子。于是冒出一堆工作想把 RLVR 从数学推到 general STEM（物理、化学、生物等等）。问题是这些工作基本都直接在 base model 上跑 RL，跳过了 SFT。

作者就觉得：等等，你们是不是太乐观了？DeepSeek-R1 自己都是先 SFT cold-start 再 RL 的，你们凭啥觉得纯 RL 就够？

## Motivation：一个小实验打脸

作者做了个对照实验：同样的题目，同样的数据量，分别用 SFT 和 RL 训。SFT 的答案是用一个中等水平的小 model（Qwen3-4B-Instruct）生成的，不是用 GPT-4 这种强 model——故意让 SFT 不占便宜。

结果（Figure 1）：**SFT 在所有数据规模上都赢 RL**。general STEM 和数学两个 domain 都一样。

为啥？道理其实很朴素：

- SFT 是手把手教，每个 token 都有 gradient 信号
- RL 只告诉"对/错"，模型得自己瞎试，small base model exploration cost 太高
- STEM 题需要 domain knowledge（物理公式、生物事实），这种 declarative knowledge 通过 trial-and-error 学太慢，SFT 一遍 distill 就行

## DeReason 的核心 idea

既然 SFT 和 RL 各有所长，那就按难度分数据：

- **容易的题给 SFT**：这类题主要是知识 recall，teacher 已经能答对，直接模仿最 efficient
- **困难的题给 RL**：这类题需要多步推理，teacher 自己也不一定答对，RL 通过 exploration 能跳出 teacher 的天花板

具体怎么做？用一个同 size 的 instruct LLM（Qwen3-4B-Instruct）给每道题打 1-5 分：
- 1 分 = 单一事实回忆
- 5 分 = 多步推导 + 深 domain knowledge

然后 $\tau = 3$ 一刀切：score ≤ 3 的给 SFT，score ≥ 4 的给 RL。

$$
\mathcal{D}_{\mathrm{SFT}} = \{(x_i, a_i^*) \in \mathcal{D} \mid d_i \leq \tau\}, \quad \mathcal{D}_{\mathrm{RL}} = \{(x_i, a_i^*) \in \mathcal{D} \mid d_i > \tau\}
$$

就这一行公式，$d_i$ 是题 $i$ 的难度分，$\tau$ 是 threshold（取 3）。Pipeline 是 SFT 先跑完得到 $\pi_{\mathrm{SFT}}$，然后从 $\pi_{\mathrm{SFT}}$ 初始化做 GRPO。

## Reward 怎么搞

数学题用 rule-based matcher 就行，general STEM 不行——答案经常是 free-form explanation，没法 regex match。所以用一个 LLM-based verifier：

$$
R(x, o) = \begin{cases} 1 & \text{if } \mathcal{V}_{\theta}(\mathrm{EXTRACT}(o), a^*, x) = \mathrm{True} \\ 0 & \text{otherwise} \end{cases}
$$

$\mathcal{V}_{\theta}$ 是一个 LLM verifier，给定题 $x$、ground truth $a^*$ 和模型提取出的答案，判断语义上对不对。$\mathrm{EXTRACT}$ 就是从 response $o$ 里把最终答案抠出来。这套是直接抄 General Reasoner (Ma et al. 2025, https://openreview.net/forum?id=pBFVoll8Xa) 的。

## 主结果：谁赢谁

WebInstruct-Verified 上 4B model（Table 1）：

| 方法 | MMLU-Pro | GPQA-D | SuperGPQA | BBEH | AVG |
|---|---|---|---|---|---|
| RL only | 62.8 | 42.9 | 32.5 | 12.2 | 37.6 |
| SFT only | 68.6 | 46.8 | 38.4 | 13.5 | 41.8 |
| SFT then random RL | 68.6 | 47.8 | 39.4 | 15.8 | 42.9 |
| **DeReason (SFT easy + RL hard)** | 68.4 | **50.0** | **40.2** | **16.7** | **43.8** |

四个 takeaway：

1. **SFT only 把 RL only 按地上摩擦**——4B 上纯 RL 完全打不过 SFT
2. **DeReason 又把 SFT only 摩擦一遍**——说明 RL 在 hard subset 上确实能加分
3. **DeReason > random split**——难度切分有用，不是单纯"多加一个 RL 阶段"的功劳
4. **加分的 benchmark 不一样**：MMLU-Pro（knowledge-heavy）几乎没涨，BBEH（reasoning-heavy）涨 +3.2。完全符合预期——knowledge recall 题 SFT 已经到顶，RL 在 reasoning 题上才有空间

最有意思的 negative result：把 hard data 也拿来 SFT（"SFT then selected SFT"），avg 41.4，居然比 SFT only (41.8) 还低。这说明 hard data 上 teacher 自己也答得烂，硬模仿反而学到错误 pattern。RL 的价值就在于能从 noisy rollouts 里靠 verifier 筛出对的路子——noise 让 verifier 干掉，不让 teacher 坑 model。

## 跟大模型对比

最骚的 result：4B 的 DeReason model，AVG 43.8，**超过 32B 的 QwQ-32B (43.2)**，超过 7B 的 Open-Reasoner-Zero (35.3) 8.5 个点。这说明 post-training recipe 比 model size 重要得多——你把 data 分好了，小 model 也能打大 model。

## 那些 analysis figures 说的故事

### Figure 2：难度分布的真相

按难度 1-5 切片看 category：
- Score 1-2：History、Biology、Business、Psychology 各种都有
- Score 4：Math 占 78%
- Score 5：Math 占 96%

**难度这个 1-5 scale 其实就是在区分"知识题"和"数学题"**。容易的全是事实回忆，难的全是数学推导。所以 DeReason 等价于"非数学给 SFT，数学给 RL"——这恰好匹配 SFT 擅长 distill knowledge、RL 擅长 explore reasoning 的分工。

### Figure 3：reward 曲线

从 SFT checkpoint 出发跑 RL：
- 容易子集（score 1-3）上 reward **下降**——model 本来就会，RL exploration 反而引入 noise
- 困难子集（score 4-5）上 reward **维持或上升**

这直接证明了 DeReason 的 thesis：**RL 在已经掌握的题上跑是有害的，只在 challenging 题上才有用**。

### Figure 4：response length 的故事

这是最 intuition-rich 的一张图。

从 **SFT checkpoint** 出发：所有 response 起始都很长（~4200 tokens，因为 SFT 学了 instruct model 的 verbose 风格）。RL 期间长度普遍下降，high-reward output 从 4200 降到 3000。RL 在做 **compression**——把 verbose 部分压掉，保留 length-quality hierarchy。

从 **base model** 出发：所有 response 起始都短（~1200 tokens）。RL 期间长度快速 **bifurcate**——答对的越写越长，答错的越写越短，40 步内 gap 超过 1000 tokens。这是 base model 从零学到 "答对要长 CoT，答错要 short-circuit"。

这两种 mechanism 完全不同。SFT 先把 model 拉到 verbose 的状态，RL 再做减法；base model 直接从空白学，RL 同时学"该长还是该短"和"内容对不对"。

### Figure 5：entropy 的反直觉

Policy entropy（log scale）：
- Base init：起始 ~2.0，前 20 步陡降，稳定在 <0.10
- SFT init：起始 ~0.30，缓慢下降
- **Base 最终 entropy < SFT 最终 entropy**

反直觉吧？直觉上 SFT 应该把 policy 收窄，RL 从 SFT 开始应该更 sharp。结果反过来——RL 从 base 开始通过 reward-driven exploration 最终收得更窄。说明 **RL 的 narrowing 是 selective 的（只 narrow 到 reward-positive trajectory），比 SFT 的 uniform imitation narrowing 更精准**。

## 一句话总结

**Post-training 里 data 怎么切比 algorithm 怎么调重要**。DeReason 就一句 rule：用 LLM-judge 给题打 1-5 分，≤3 给 SFT，≥4 给 RL，SFT 先跑完再 RL。就这，4B model 超过 32B 的 QwQ-32B。

R1-Zero 那套"纯 RL from base"在小 model 上其实根本不是最优，很多被归功给 RL 的效果，其实归功给 data selection 才对。

---

## 想深挖的话

- DeReason（估计 ID）: https://arxiv.org/abs/2602.06375
- General Reasoner（verifier 设计来源）: https://openreview.net/forum?id=pBFVoll8Xa
- WebscaleRL（另一个数据集）: https://arxiv.org/abs/2510.06499
- DeepSeek-R1: https://www.nature.com/articles/s41586-025-09422-z
- GRPO: https://arxiv.org/abs/2402.03300
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- RL squeezes, SFT expands: https://arxiv.org/abs/2509.21128
- Hard examples are all you need: https://arxiv.org/abs/2508.14094
- E2H Reasoner: https://arxiv.org/abs/2506.06632

我觉得最值得 follow-up 的几个点：
1. **τ 是不是 3 最优**？没系统 ablation
2. **LLM judge 选 score 4-5 实际就是选 math 题**，这等于隐含假设"数学最需要 RL"。这假设换 domain 是否成立？
3. **在 70B+ 上是否还成立**？大 model base 本身 reasoning prior 强，可能 pure RL 就够 efficient
4. **Verifier accuracy vs final performance** 没做 ablation——verifier 错了怎么办？
5. **AIME25 上 Webscale 的反例**：Ours (20.7) 居然低于 SFT only (23.3)，啥情况？
6. **Difficulty score 能不能换别的**？比如 pass rate、base model 的 logprob？LLM judge 是不是最优 partitioning criterion？

---

# DeReason 深度解读：A Difficulty-Aware Curriculum for Decoupled SFT-then-RL

这篇 paper 来自 University of Zurich 的 Rico Sennrich 组（加上 ETH、HKUST、UPenn 合作者），核心问题是：**在 general STEM reasoning 上，post-training 中 SFT 和 RL 应该如何分配数据？** 答案是用一个 1-5 的 reasoning intensity 把数据切两半——容易的、knowledge recall 类的给 SFT；困难的、multi-step reasoning 类的留给 RL。方法叫 **DeReason** (Difficulty-aware Decoupling)。

paper: https://arxiv.org/abs/2602.06375 (注：从 references 看 DeReason 引用了 DEPO 2602.06375，应该是近似 ID)

---

## 1. Paper 想回答的根本问题

最近 RLVR (Reinforcement Learning with Verifiable Rewards) 在 math 和 code 上 success 后，大家开始把它推到 general STEM reasoning（General Reasoner、WebscaleRL）。这股潮流隐含一个**共识**：RL 比 SFT 强。但作者说这个共识可能 premature，需要在 controlled 实验下重新审视。

具体质疑点有三个：

- **Pure RL on base model 在 general STEM 上是不是真的 efficient?** DeepSeek-R1、TuluV3 这些 production pipeline 其实都是 cold-start SFT 然后 RL，但 General Reasoner / WebscaleRL 这种学术工作却跳过了 SFT。
- **如果 SFT 和 RL 都要，data 怎么分?** 之前 curriculum learning 工作（E2H Reasoner、DEPO、SEC）都只讨论 RL 内部如何挑难度，没讨论跨阶段的数据分配。
- **SFT 和 RL 各自塑造 model behavior 的方式有什么本质区别?** 通过 response length、entropy、reward dynamics 来刻画。

---

## 2. Motivation 实验：SFT vs RL 的 controlled comparison

### 2.1 实验设计

为了公平比较，作者做了一个非常 careful 的对照实验：

- **同样的 problem set** 训练 SFT 和 RL
- **varying 训练数据量**（看 scaling 行为）
- SFT 的 responses **不是** frontier model 生成的，而是用 moderate-capability model (Qwen3-4B-Instruct-2507) 生成，避免 "SFT 占便宜因为 teacher 太强" 的 confound
- RL 用 GRPO from base model 直接训

evaluation:
- general STEM: GPQA-Diamond pass@1 averaged over 8 runs
- math: AIME24 + AIME25 + MATH500 的 pass@1

### 2.2 关键发现 (Figure 1)

两个 domain 都呈现同样的 pattern：

```
General STEM (GPQA-D):
  Data size → SFT 单调超过 RL
  RL 即使数据增加也 catch up 不上 SFT

Math (AIME24/25 + MATH500):
  SFT 用 moderate-quality responses 已经达到或超过 RL trained on same problems
```

**核心 insight**: 对 small base model 而言，SFT 的 sample efficiency 显著优于 RL，原因是

- SFT 给的是 **dense token-level gradient signal**——每个 token 都是 supervised target
- RL 只给 **sparse outcome-based reward**——模型必须通过 noisy exploration 自己 discover effective reasoning paths
- 没有先验 reasoning skill 的小 base model，exploration cost 极高
- 更关键：STEM 需要 **domain knowledge** (physics formulae, algebraic identities, biology facts)，这类 knowledge 通过 trial-and-error 很难 acquire，但 SFT 可以直接 distill

这个 finding 其实呼应了 Chu et al. 2025 的 "SFT memorizes, RL generalizes" (https://arxiv.org/abs/2501.17161) 和 Matsutani et al. 2025 的 "RL squeezes, SFT expands" (https://arxiv.org/abs/2509.21128)——但 DeReason 把它推到了数据分配的层面。

---

## 3. DeReason 方法详解

### 3.1 Pipeline 三阶段

设训练集 $\mathcal{D} = \{(x_i, a_i^*)\}_{i=1}^N$，每个 problem $x_i$ 带 ground-truth answer $a_i^*$。

**Stage 1: Difficulty Estimation**
用一个同规模的 instruct LLM（论文里是 Qwen3-4B-Instruct，避免依赖 external 强模型）对每个 problem 打分 $s_i \in \{1, 2, 3, 4, 5\}$。Prompt 考虑：
- number of reasoning steps
- prerequisite domain knowledge
- potential for error

完整的 prompt 在 Appendix A.1 / Table 3，大致结构是：先描述评分标准（1 分=单一 fact recall；5 分=multi-step derivation requiring deep domain expertise），然后让模型对 `{question}` 输出 integer score。

**Stage 2: Data Partitioning**

$$
\mathcal{D}_{\mathrm{SFT}} = \{(x_i, a_i^*) \in \mathcal{D} \mid d_i \leq \tau\}, \quad \mathcal{D}_{\mathrm{RL}} = \{(x_i, a_i^*) \in \mathcal{D} \mid d_i > \tau\}
$$

公式 (3) 中：
- $\mathcal{D}_{\mathrm{SFT}}$, $\mathcal{D}_{\mathrm{RL}}$ 是两个互不相交的子集
- $d_i$ 是 problem $i$ 的 difficulty score
- $\tau$ 是 difficulty threshold，论文里实际取 $\tau = 3$，即把 score 4 和 5 的 problem 分给 RL

对 $\mathcal{D}_{\mathrm{SFT}}$ 用 moderate teacher (Qwen3-4B-Instruct) 生成 reference response $y_i$，构造 SFT pairs。

**Stage 3: Curriculum Training**
1. 在 $\mathcal{D}_{\mathrm{SFT}}$ 上做 SFT 得到 $\pi_{\mathrm{SFT}}$
2. 从 $\pi_{\mathrm{SFT}}$ 初始化，在 $\mathcal{D}_{\mathrm{RL}}$ 上做 GRPO

### 3.2 Verification for general reasoning

general STEM 的答案往往是 free-form explanations 或 qualitative reasoning，rule-based matcher 搞不定。所以 reward function 用 model-based verifier（遵循 Ma et al. 2025, General Reasoner）：

$$
R(x, o) = \begin{cases} 1 & \text{if } \mathcal{V}_{\theta}(\mathrm{EXTRACT}(o), a^*, x) = \mathrm{True} \\ 0 & \text{otherwise} \end{cases}
$$

公式 (2) 解析：
- $x$ 是 prompt
- $o$ 是 model 的 response
- $\mathrm{EXTRACT}(\cdot)$ 从 response $o$ 中提取最终答案（比如解析 `\boxed{}` 或者最后一段）
- $a^*$ 是 ground-truth answer
- $\mathcal{V}_{\theta}$ 是一个 LLM-based verifier，给定问题 $x$、ground truth $a^*$ 和 extracted answer，判断语义等价性
- 输出 binary reward：1 表示正确，0 表示错误

这个 verifier 比传统 rule-based matcher 更通用，能处理 approximate numerical values、multi-part derivations 等复杂情况。但代价是：verifier 本身可能出错，且引入额外的 inference cost。

### 3.3 GRPO 训练目标

公式 (1) 是 GRPO 的标准 loss：

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}_{\mathrm{RL}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(\rho_i \hat{A}_i, \ \mathrm{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) - \beta D_{\mathrm{KL}}\left(\pi_{\theta} \| \pi_{\mathrm{ref}}\right) \right]
$$

逐项解析：

- $\theta$: policy network 的参数
- $x \sim \mathcal{D}_{\mathrm{RL}}$: 从 RL 数据集采 prompt
- $G$: 每个 prompt 采样的 response 数量
- $i \in \{1, \dots, G\}$: group 内 response 索引
- $o_i$: 第 $i$ 个 sampled response
- $r_i = R(x, o_i)$: reward from verifier
- $\hat{A}_i$: group-normalized advantage，由 $\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}$ 得到（group-relative baseline，省掉 value model）
- $\rho_i = \frac{\pi_{\theta}(o_i \mid x)}{\pi_{\mathrm{ref}}(o_i \mid x)}$: importance sampling ratio，衡量 current policy 相对于 reference policy 的概率变化
  - $\pi_{\theta}$: current policy
  - $\pi_{\mathrm{ref}}$: reference policy（通常 frozen，防止 policy 漂移太远）
- $\varepsilon$: clipping 参数（PPO 标准 trick），限制 $\rho_i$ 在 $[1-\varepsilon, 1+\varepsilon]$ 区间内，避免 importance ratio 爆炸
- $\beta$: KL regularization 强度
- $D_{\mathrm{KL}}(\pi_{\theta} \| \pi_{\mathrm{ref}})$: KL divergence，惩罚 policy 远离 reference
- min 和 clip 组合是 PPO 的 standard clipped surrogate objective
- 负号因为我们要 minimize loss = maximize expected advantage minus KL penalty

**Intuition**: GRPO 把 group 内的 relative reward 作为 advantage，省掉 critic。一个 group 里答得对的 response 拿正 advantage、答得错的拿负 advantage，模型通过 importance-weighted gradient 朝前者靠拢。clip + KL 是 stabilization。

---

## 4. 实验设置

### 4.1 Training details

- **Base model**: Qwen3-4B-Base
- **SFT framework**: Llama-Factory (https://aclanthology.org/2024.acl-demos.38/), batch size 128, lr 1e-5
- **RL framework**: VeRL (HybridFlow, https://arxiv.org/abs/2409.19256), max response length 8192, batch size 128, mini-batch 64, lr 1e-6
- **SFT teacher**: Qwen3-4B-Instruct-2507 (same size, moderate capability——这是个非常 deliberate 的设计选择，避免 SFT 看起来占便宜)
- **两个 dataset**:
  - WebInstruct-Verified (Ma et al. 2025, General Reasoner): https://openreview.net/forum?id=pBFVoll8Xa
  - Webscale-RL (Cen et al. 2025): https://arxiv.org/abs/2510.06499

### 4.2 Evaluation benchmarks

- **MMLU-Pro** (Wang et al. 2024, https://openreview.net/forum?id=y10DM6R2r3): 10-way MC，比原 MMLU 推理要求更高
- **GPQA-Diamond** (Rein et al. 2024, https://openreview.net/forum?id=Ti67584b98): PhD-level 专家出题，random baseline ≈ 34%
- **SuperGPQA** (Du et al. 2025, https://openreview.net/forum?id=6WgflzYQpf): 285 disciplines
- **BBEH** (Kazemi et al. 2025, BIG-Bench Extra Hard, https://aclanthology.org/2025.acl-long.1285/): 难度大幅升级的 BIG-Bench Hard
- **数学**: AIME24, AIME25, MATH500

四个 benchmark 的 spectrum 是从 **knowledge recall** (MMLU-Pro) 渐变到 **complex multi-step reasoning** (BBEH)——这正好契合 DeReason 的 difficulty 假设。

---

## 5. 主实验结果

### 5.1 General reasoning (Table 1)

**4B Models on WebInstruct-Verified**:

| Setting | MMLU-Pro | GPQA-D | SuperGPQA | BBEH | AVG |
|---|---|---|---|---|---|
| Qwen3-4B-Base | 51.6 | 26.5 | 25.4 | 8.1 | 27.9 |
| RL only | 62.8 | 42.9 | 32.5 | 12.2 | 37.6 |
| SFT only | 68.6 | 46.8 | 38.4 | 13.5 | 41.8 |
| SFT then selected SFT (hard data also via SFT) | 68.6 | 45.6 | 38.2 | 13.0 | 41.4 |
| SFT then random RL | 68.6 | 47.8 | 39.4 | 15.8 | 42.9 |
| **Ours (SFT easy + RL hard)** | 68.4 | **50.0** | **40.2** | **16.7** | **43.8** |

**4B Models on Webscale-RL**:

| Setting | MMLU-Pro | GPQA-D | SuperGPQA | BBEH | AVG |
|---|---|---|---|---|---|
| RL only | 55.4 | 34.0 | 30.9 | 10.1 | 32.6 |
| SFT only | 60.7 | 39.2 | 37.3 | 13.4 | 37.7 |
| **Ours** | 60.3 | **43.7** | **38.8** | **15.7** | **39.6** |

**几个重要观察**：

1. **SFT only > RL only**：在两个 dataset、所有 benchmark 上都成立，验证 Figure 1 的 controlled finding。
2. **SFT then selected SFT (41.4) < SFT only (41.8)**：把 hard data 也拿去 SFT 反而轻微 hurt 性能。这是个非常有意思的 negative result——说明 hard data 用 SFT 是 suboptimal 的，因为 SFT 只能模仿 teacher 的 demonstration，但 teacher (Qwen3-4B-Instruct) 在 hard problem 上自己也可能 not optimal，强行 mimic 反而 constrain 了 model 的 reasoning potential。
3. **SFT then random RL (42.9) < Ours (43.8)**：difficulty-aware partitioning 比 random split 好约 1 point on average。
4. **Benchmark-dependent gap**: 
   - MMLU-Pro 上 Ours (68.4) ≈ SFT only (68.6)，几乎无提升甚至略降
   - BBEH 上 Ours (16.7) >> SFT only (13.5)，提升 +3.2
   - 这印证了 DeReason 的核心 thesis：knowledge-heavy benchmark 上 SFT 已经够好，RL 的增益主要在 reasoning-heavy benchmark 上。

### 5.2 Math (Table 2)

| Model | AIME24 | AIME25 | MATH500 |
|---|---|---|---|
| WebIns-V (RL only) | 20.0 | 15.4 | 80.6 |
| WebIns-V (SFT only) | 22.0 | 17.6 | 82.6 |
| WebIns-V (Ours) | **22.1** | **18.0** | **84.1** |
| Webscale (RL only) | 21.3 | 14.0 | 81.6 |
| Webscale (SFT only) | 26.3 | 23.3 | 87.5 |
| Webscale (Ours) | **27.7** | 20.7 | **88.1** |

注意 Webscale 上 AIME25 Ours (20.7) 居然低于 SFT only (23.3)——这是个值得探讨的反例。可能 AIME25 这种 extreme hard 题目上，selected RL subset 的覆盖度不够，或者 RL 训练的 exploration 在 certain distribution 上导致 regression。

### 5.3 与更大模型的对比 (Table 1 上半部分)

- GPT-4o: AVG 48.3
- QwQ-32B: AVG 43.2
- DeepSeek-R1: AVG 62.6
- Qwen2.5-7B-Base: AVG 27.9
- Qwen2.5-7B-Instruct: AVG 33.4
- Open-Reasoner-Zero (7B): AVG 35.3
- SimpleRL-Qwen2.5-7B-Zoo: AVG 29.4
- **DeReason on 4B (WebIns-V): AVG 43.8**——比所有 7B baseline 都好，接近 QwQ-32B 的 43.2

这是个很强的 result：4B 的 DeReason model 超过 32B 的 QwQ-32B，比 7B 的 Open-Reasoner-Zero 高 8.5 points。这暗示了**post-training recipe 比 model size 更重要** 的可能。

---

## 6. Analysis 深挖：build your intuition

这部分是 paper 最有价值的部分，提供了 SFT vs RL training dynamics 的细粒度刻画。

### 6.1 Data distribution across difficulty (Figure 2)

按 difficulty score 1-5 切片看 category 分布：

- **Score 1-2**: 分布在 History, Biology, Business, Psychology, etc.，broad coverage
- **Score 3**: 开始倾向 Math/Physics
- **Score 4**: Math 占 ~78%
- **Score 5**: Math 占 ~96%

**Intuition**: 难度这个 1-5 scale 其实是在 measure "knowledge recall vs multi-step derivation"。容易问题都是 "事实题"——比如 "X 是哪一年发生的"、"Y 的定义是什么"；困难问题几乎全是 "证明/推导" 类——这本质就是 math/physics reasoning 的范畴。

这个 distribution 也解释了为什么 DeReason works：把 broad knowledge 给 SFT 做 distillation，把 math-heavy 的硬骨头给 RL 做 exploration，正好匹配两个阶段各自的 strength。

### 6.2 Reward dynamics across difficulty subsets (Figure 3)

作者从两个 init checkpoint（SFT 和 base）出发，分别在 difficulty 1/2/3/4/5 的 subset 上跑 RL，看 reward 曲线：

**From SFT checkpoint (左)**:
- 初始 reward 普遍较高（因为 SFT 已经学过）
- 后续 improvement 较慢
- 在 score 4 和 5 subset 上能持续提升
- 在 score 1, 2, 3 subset 上反而有 gradual decline——可能因为 RL 在已经掌握的题目上 exploration 反而引入 noise

**From base checkpoint (右)**:
- 初始 reward 很低（base model 没见过这种 format）
- 前 40 步 dramatic improvement（exploration phase）
- 40 步后 plateau（exploitation phase）
- 所有 difficulty 上都缓慢提升

**Intuition**: SFT 后的 model 已经处于 local minimum 附近，RL 的边际收益小；base model 从零开始 explore，初期 reward gain 巨大但很快 saturate。这呼应了 RLVR 文献里 "early-burst then plateau" 的现象（参考 Liu et al. 2025 Dr. GRPO, https://arxiv.org/abs/2503.20783）。

### 6.3 Response length evolution (Figure 4)

这是最 fascinating 的 figure。按 reward score 切片看 response 长度随 training step 变化：

**From SFT checkpoint (左)**:
- 初始所有 score 的 response 都很长（~4200 tokens）——SFT 继承了 instruct model 的 verbose behavior
- RL 期间 length 单调下降
- High-scoring outputs 下降最明显：~4200 → ~3000 tokens
- 各 score level 的 length 大致保持平行，gap 不 dramatic
- RL 的作用像 **compression mechanism**：保留 length-quality hierarchy，uniformly 压缩 verbosity

**From base checkpoint (右)**:
- 初始所有 score 的 response 都类似长度（~1200 tokens）
- RL 期间 length 快速 **bifurcate**：
  - High-scoring (Score 5): length 保持甚至增长
  - Low-scoring (Score 1): 缩到 <500 tokens
- 前 40 步内 Score 5 和 Score 1 的 gap 超过 1000 tokens
- Bifurcation 远比 SFT case 明显

**Intuition**: 
- SFT-initialized model 已经内化了 "对难题要长 chain-of-thought" 的 prior，RL 只是把 verbose 部分压缩掉
- Base model 通过 RL 从头学到 "答对就要长，答错就要短"——这是 RLVR 的 emergent behavior，通过 reward signal 直接塑造 length-quality correlation
- 这呼应了 Matsutani et al. 2025 "RL squeezes, SFT expands" (https://arxiv.org/abs/2509.21128)：SFT 让 trajectory expand 到 verbose reasoning，RL 让 trajectory squeeze 到 essential steps

### 6.4 Actor entropy (Figure 5)

Policy entropy 在 RL 训练中的演化（log scale）：

- **Base init**: entropy 起始 ~2.0，前 20 步陡降，stabilize 在 <0.10
- **SFT init**: entropy 起始 ~0.30，缓慢下降
- **Base 最终 entropy < SFT 最终 entropy**

**Intuition**: 
- Base model 输出 distribution 是 broad/unconstrained，SFT 已经预 narrow 过
- RL 从 base 开始会通过 reward-driven exploration 达到 **sharper specialization**，比 SFT 还要 deterministic
- SFT 起的 "narrowing" 作用是 uniform 的（imitation）；RL 的 narrowing 是 reward-oriented 的（只 narrow 到 correct trajectories）

这个 base entropy 最终低于 SFT entropy 的现象很反直觉，说明 **RL 的探索-利用循环可以产生比 supervised imitation 更 sharp 的 policy**——只要 reward signal 足够 informative。这跟 DeepSeek-R1 报告里 R1-Zero 的 emergent behavior 一致。

### 6.5 GPQA-D during training (Figure 6)

从 SFT checkpoint 出发的 RL 在 GPQA-D 上：
- 大部分 difficulty subsets 上 performance gradual decline
- **只有 score 4 和 5 subset 上能维持或提升**

从 base model 出发：
- 初始 performance 低
- 之后慢慢 increase，但 final 不如 SFT init

**Intuition**: 给已经 SFT 过的 model 跑 RL，如果在"已经掌握"的子集上训，会出现 **catastrophic forgetting / over-optimization**，performance decline。只有在真正 challenging 的子集上训才能 maintain 或 gain。这是 DeReason 设计的根本依据——hard data 才是 RL 的 sweet spot，easy data 反而会 hurt。

---

## 7. Intuition 综合：为什么 DeReason works?

把以上 analysis 拼起来：

### 7.1 SFT 和 RL 的 division of labor

**SFT 的优势**:
- Dense token-level supervision
- 能直接 distill **declarative knowledge** (physics formulae, biology facts, history events)
- Sample-efficient: 一个 (x, y) pair 给 N tokens 的 gradient
- 对于 base model 来说是 **cold-start**: 让 model 学会基本的 reasoning format 和 domain knowledge

**RL 的优势**:
- 能 push performance **beyond teacher's demonstration**——teacher 不是 frontier model 时尤其重要
- 通过 exploration discover novel reasoning paths
- 直接 optimize task reward，不受 teacher sub-optimality 限制
- 适合 **procedural knowledge** / **multi-step derivation**: 这里 exploration cost 可以被 reward signal amortize

### 7.2 Difficulty 作为划分 criterion 的依据

Difficulty score 实际上 measure 了两个 dimension：
1. **Knowledge content**: 容易题是 recall，难题是 derivation
2. **Reasoning depth**: 容易题 1-2 步，难题 5-10+ 步

这两个 dimension 共同决定了：
- 容易题 → teacher 已经答得不错 → SFT 直接 mimic 最 efficient
- 难题 → teacher 也可能 sub-optimal → RL exploration 价值高

### 7.3 Negative result: 为什么 "SFT then selected SFT" 表现差?

这个 baseline (WebIns-V: 41.4) 比 "SFT only" (41.8) 还略差。说明：

- Hard data 上 teacher 的 demonstration 本身就 noisy（teacher Qwen3-4B-Instruct 在 score 5 题目上正确率可能 50% 都不到）
- 把 noisy demonstration 作为 SFT target，会让 model 学到错误的 reasoning pattern
- RL 用 reward signal 反而能从 noisy exploration 中筛出正确的 path，**noise 由 verifier filter 掉**

这是 DeReason 的核心洞察：**hard data 的 signal 不在 demonstration 里，而在 verification 里**。

### 7.4 Curriculum vs Joint Training

DeReason 的 sequential curriculum 跟 Huang et al. 2025 (https://arxiv.org/abs/2507.01679) 和 Yan et al. 2025 (https://arxiv.org/abs/2504.14945) 的 algorithmic blending 不同——它在 data level 操作，所以 orthogonal to 任何 algorithmic improvement。如果未来有一个更好的 SFT+RL joint algorithm，DeReason 的 difficulty partitioning 仍然可以作为 pre-processing 步骤加上去。

---

## 8. 相关工作脉络

### 8.1 RLVR 主线

- **GRPO** (Shao et al. 2024, DeepSeekMath): https://arxiv.org/abs/2402.03300 —— group-relative advantage，省 value model
- **DeepSeek-R1** (Guo et al. 2025, Nature): https://www.nature.com/articles/s41586-025-09422-z —— 大规模 RLVR 的标杆
- **TinyZero** (Pan et al. 2025): https://github.com/Jiayi-Pan/TinyZero —— 小规模复现 R1-Zero
- **Dr. GRPO** (Liu et al. 2025): https://arxiv.org/abs/2503.20783 —— 修正 GRPO 的 length bias
- **DAPO** (Yu et al. 2025): https://arxiv.org/abs/2503.14476 —— open-source 大规模 RL 系统
- **Tulu 3** (Lambert et al. 2025): https://arxiv.org/abs/2411.15124 —— production pipeline，SFT + RL + DPO
- **o1 system card** (OpenAI 2024): https://arxiv.org/abs/2412.16720

### 8.2 General STEM reasoning 扩展

- **General Reasoner** (Ma et al. 2025): https://openreview.net/forum?id=pBFVoll8Xa —— model-based verifier 解锁 general domain RL
- **WebscaleRL** (Cen et al. 2025): https://arxiv.org/abs/2510.06499 —— 自动化 data pipeline
- **Mammoth2** (Yue et al. 2024): https://arxiv.org/abs/2405.03548 —— 从 web scale instruction
- **Nemotron CrossThink** (Akter et al. 2025): https://arxiv.org/abs/2504.13941 —— cross-domain self-learning
- **RLPR** (Yu et al. 2025b): https://arxiv.org/abs/2506.18254 —— verifier-free RL via likelihood
- **Cross-domain study** (Cheng et al. 2025): https://arxiv.org/abs/2506.14965 —— RL transfer 的 domain-dependence

### 8.3 Difficulty-based curriculum (在 RL 内部)

- **Pikus et al. 2025**: https://arxiv.org/abs/2508.14094 —— 只用 base model 失败最多的 10% hard examples，能涨 47%；easy examples 只涨 3-15%
- **E2H Reasoner** (Parashar et al. 2025): https://arxiv.org/abs/2506.06632 —— curriculum from easy to hard within RL
- **DEPO** (Zhao et al. 2026): https://arxiv.org/abs/2602.06375 —— online difficulty estimator + 过滤 trivial/complex
- **SEC** (Chen et al. 2025): https://arxiv.org/abs/2505.14970 —— multi-armed bandit 自适应选难度
- **Tang et al. 2025**: https://arxiv.org/abs/2509.01321 —— offline curation + online explorability
- **Sun et al. 2026**: https://arxiv.org/abs/2506.05316 —— attention-based difficulty estimator 选 moderate 难度

这一支的共同盲点：**只考虑 RL 阶段内部如何挑难度，没考虑 SFT vs RL 的跨阶段分配**。DeReason 填的就是这个 gap。

### 8.4 SFT vs RL 的 mechanistic study

- **SFT memorizes, RL generalizes** (Chu et al. 2025): https://arxiv.org/abs/2501.17161
- **RL squeezes, SFT expands** (Matsutani et al. 2025): https://arxiv.org/abs/2509.21128 —— DeReason 的 Figure 4 直接验证这个
- **Non-decoupling of SFT and RL** (Niu et al. 2026): https://arxiv.org/abs/2601.07389 —— 理论上 SFT 和 RL 在 parameter space coupled
- **Math RL preserves general representations** (Huan et al. 2025): https://arxiv.org/abs/2507.00432 —— RL 比 SFT 在 math-only training 上更少 distribution drift
- **Blending SFT and RL** (Huang et al. 2025): https://arxiv.org/abs/2507.01679 —— prefix sampling 算法层面 blend
- **Off-policy guidance** (Yan et al. 2025): https://arxiv.org/abs/2504.14945

---

## 9. DeReason 的局限与 open questions

paper 自己没 explicit 讨论 limitations，但从 analysis 能推出几个：

### 9.1 Difficulty score 依赖 LLM judge

用 Qwen3-4B-Instruct 打 1-5 分，问题是：
- Judge 自身有 bias，可能对某些 category（比如 math）系统性偏高/低
- Figure 2 显示 score 4/5 几乎全是 math，这意味着 DeReason 实际上把 "math 给 RL, 其他给 SFT"——这隐含假设 math 推理比其他 STEM 更需要 exploration。这个假设不一定普适。
- 不同 judge model 可能给不同 partition，sensitivity 没分析

### 9.2 Threshold τ = 3 是 arbitrary

paper 没说明为什么选 τ=3 而不是 2 或 4。Figure 3 实际显示 score 4 和 5 在 SFT-init 上是唯二能 maintain performance 的 subset，所以 τ=3 是经验上合理，但没 systematic ablation over τ。

### 9.3 Verifier reliability

general STEM 的 model-based verifier $\mathcal{V}_{\theta}$ 本身有 error rate。reward noise 会 propagate 到 RL training，但 paper 没分析 verifier accuracy 对 final performance 的影响。

### 9.4 Generalization 到更大 model

实验只在 4B 上做。直觉上 larger model 的 base 已经有更强 reasoning prior，pure RL 可能就够 efficient（DeepSeek-R1 在 671B 上做 cold-start SFT + RL，但 R1-Zero 在更小规模上也能 emerge reasoning）。DeReason 在 70B / 671B 上是否仍然 beat pure RL 是个 open question。

### 9.5 AIME25 反例

Webscale 上 Ours (20.7) 在 AIME25 上低于 SFT only (23.3)。这说明 selected RL subset 在某些 distribution 上可能 miss 掉关键题型，RL exploration 反而 hurt performance。需要更多 ablation 理解什么时候 RL 会 regression。

### 9.6 Why does "SFT then selected SFT" underperform "SFT only"?

Table 1 里 41.4 < 41.8。这意味着在 hard subset 上做额外 SFT 反而 hurt model。一个可能解释：hard subset 的 teacher response 错误率高，model 在错误的 demonstration 上 over-fit。但这需要量化——比如分析 teacher 在 score 5 subset 上的 correctness。

---

## 10. 给 Karpathy 的直觉总结

你曾经强调过 "input/output 行为的可解释性 比 内部机制更重要" (e.g., Software 2.0 思路)。DeReason 这个 paper 本质上是在做 **post-training data 的 program synthesis**：

1. **Data as curriculum**: SFT 和 RL 不是两个独立算法，而是同一条 curriculum 上的两个阶段。SFT 教 "知识"，RL 教 "推理"——这跟人类学习从 "背诵" 到 "解题" 的 progression 同构。
2. **Difficulty as a low-dimensional projection of data**: 一个 1-5 score 把高维的 (problem, domain, reasoning_type) 压到一个 scalar，然后用这个 scalar 做 partition。Simple but effective。
3. **RL as exploration beyond teacher**: 当 teacher 自己 weak 时，RL 的价值在于从 noisy rollouts 中筛出 reward-positive trajectory——这个 noise-filtering 能力是 SFT 不具备的。
4. **Sharp vs Uniform narrowing**: base model 通过 RL 训练最终 entropy 比 SFT 还低——这是个 deep 的观察。SFT 的 narrowing 是 uniform 的（imitation 全部 narrow），RL 的 narrowing 是 selective 的（只 narrow 到 reward-positive trajectories）。后者的最终 policy 可以比前者更 sharp。
5. **Compression mechanism**: SFT-init 的 RL 像是 lossy compression——保留 length-quality hierarchy 但 squeeze verbose。Base-init 的 RL 像是 bifurcation——同时学会"长而正确"和"短而错误"两种模式。这两个 mechanism 非常不同。

DeReason 不是 algorithmic breakthrough，是个 **data engineering insight**。它的价值在于把 SFT-then-RL 这条 recipe 从 "随便切数据" 升级到 "用 difficulty 切数据"。Tulu 3、DeepSeek-R1 这种 production pipeline 都用 sequential SFT+RL，但都是手工 curate data。DeReason 给了一个 principled rule: **用 LLM-judge difficulty 1-5 score 切数据，score ≤ 3 给 SFT，score ≥ 4 给 RL**——就这么简单。

更深远的 implication: post-training 里 **data selection 比 algorithm 重要**。DeReason 用 vanilla GRPO + vanilla SFT，但通过 data partition 拿到 4B model 超过 7B baseline 的 result。这说明 R1-Zero-style "纯 RL from base" 在 small model 上可能根本不是最优——很多效果归因给 RL 的，其实归因给 data selection 才对。

---

## References

- DeReason paper 链接 (推测): https://arxiv.org/abs/2602.06375
- DeepSeek-R1: https://www.nature.com/articles/s41586-025-09422-z
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- Tulu 3: https://arxiv.org/abs/2411.15124
- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- o1 system card: https://arxiv.org/abs/2412.16720
- General Reasoner: https://openreview.net/forum?id=pBFVoll8Xa
- WebscaleRL: https://arxiv.org/abs/2510.06499
- GPQA: https://openreview.net/forum?id=Ti67584b98
- SuperGPQA: https://openreview.net/forum?id=6WgflzYQpf
- MMLU-Pro: https://openreview.net/forum?id=y10DM6R2r3
- BBEH: https://aclanthology.org/2025.acl-long.1285/
- E2H Reasoner: https://arxiv.org/abs/2506.06632
- Pikus et al. (Hard examples): https://arxiv.org/abs/2508.14094
- DEPO: https://arxiv.org/abs/2602.06375
- SEC: https://arxiv.org/abs/2505.14970
- Tang et al. (data efficiency): https://arxiv.org/abs/2509.01321
- Sun et al. (difficulty-targeted): https://arxiv.org/abs/2506.05316
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- RL squeezes, SFT expands: https://arxiv.org/abs/2509.21128
- Non-decoupling of SFT and RL: https://arxiv.org/abs/2601.07389
- Math RL transferability: https://arxiv.org/abs/2507.00432
- Blending SFT and RL (prefix sampling): https://arxiv.org/abs/2507.01679
- Off-policy guidance: https://arxiv.org/abs/2504.14945
- Nemotron CrossThink: https://arxiv.org/abs/2504.13941
- RLPR (verifier-free): https://arxiv.org/abs/2506.18254
- Mammoth2: https://arxiv.org/abs/2405.03548
- Cross-domain RL: https://arxiv.org/abs/2506.14965
- VeRL (HybridFlow): https://arxiv.org/abs/2409.19256
- Llama-Factory: https://aclanthology.org/2024.acl-demos.38/
- Qwen3 technical report: https://arxiv.org/abs/2505.09388

如果你想进一步 dig in，最有价值的 follow-up 方向是：
1. **在 70B+ model 上验证** DeReason 是否还成立
2. **τ 的 systematic ablation** 找 sweet spot
3. **替换 difficulty score** 用 pass rate 或 embedding-based metric，看 LLM-judge 的 score 是不是 optimal
4. **结合 algorithmic SFT+RL blending** (e.g., Huang 2025 prefix sampling) 看是否 orthogonal 提升
5. **Verifier accuracy vs final performance** 的 ablation
6. **Difficulty partition 的可解释性**：是否可以 visualize 出 partition 决策的 boundary
