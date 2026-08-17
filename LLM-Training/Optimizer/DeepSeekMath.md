---
source_pdf: DeepSeekMath.pdf
paper_sha256: 6cc20b3c5b8d25b8b53868fc4ec1792c144f07d67bdf7395138efd4422197e7b
processed_at: '2026-08-03T18:53:44-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepSeekMath 用人话讲

好，我把这篇 paper 用大白话拆一遍，重点讲 intuition。

**Paper link**: https://arxiv.org/abs/2402.03300
**Code**: https://github.com/deepseek-ai/DeepSeek-Math

---

## 1. 一句话总结

DeepSeek 团队让一个 7B 的 model 在数学竞赛题（MATH benchmark）上拿到 51.7%，接近 GPT-4 的 52%。秘诀就两条：**挖数据** 和 **改 RL 算法**。

---

## 2. 他们到底干了啥

想象你要训一个数学牛人。传统思路：把 Llama 拿来，喂一堆 arXiv 论文，再做 instruction tuning。结果发现——没用，甚至变笨。

DeepSeek 的做法完全不同。

### 2.1 数据：Common Crawl 是个金矿

大家都在用 arXiv 训数学模型。DeepSeek 说：算了吧，Common Crawl 里有的是数学内容，看你能不能挖出来。

问题是怎么挖。你想用 classifier 去捞数学网页，但 classifier 要 training data。OpenWebMath 有 13.6B tokens 的高质量数学网页，可以当 seed。但 seed 太 narrow，classifier 只能捞到和 OpenWebMath "长得像" 的东西，会漏掉一大堆数学领域。

他们的招叫 **iterative domain-based expansion**：

1. 拿 OpenWebMath 训一个 fastText classifier
2. 从 40B 网页里 recall，按分数排序，留 top 40B tokens
3. 把 Common Crawl 按 base URL 分成 domains
4. 看哪些 domain 有 >10% 的网页被捞上来了——这些 domain 大概率是数学相关的（比如 mathoverflow.net）
5. 人工去这些 domain 里找具体的数学 URL pattern（比如 mathoverflow.net/questions）
6. 把这些 URL 下还没被捞的网页加入 seed
7. 重新训 classifier，再来一轮

4 轮迭代后得到 120B tokens，比 OpenWebMath 大 9 倍，比 Minerva 用的数据大 7 倍。

**Intuition**: 这本质上就是 active learning + human-in-the-loop。Classifier 有盲区，你用 domain-level 的统计发现盲区在哪，然后人工补。这比单纯放大 seed 高效得多，因为不同 URL pattern 对应不同写作风格、不同数学子领域。

**Decontamination**: 10-gram exact match 去掉包含 benchmark 的网页。短于 10-gram 的用 3-gram exact match。

### 2.2 数据质量验证

他们很诚实地做了 ablation。用 1.3B 小模型，在不同 corpus 上各训 150B tokens，结果：

| Corpus | GSM8K | MATH | CMATH |
|---|---|---|---|
| MathPile (85% arXiv) | 2.7% | 3.3% | 1.2% |
| OpenWebMath | 11.5% | 8.9% | 16.8% |
| Proof-Pile-2 | 14.3% | 11.2% | 19.9% |
| DeepSeekMath Corpus | **23.8%** | **13.6%** | **41.5%** |

注意 CMATH 是中文 benchmark——他们收了多语言数据，所以中文也强。

更细的：Figure 3 显示 DeepSeekMath Corpus 训到 50B tokens 时已经超过 Proof-Pile-2 的完整 epoch，说明 **单位 token 的信息密度更高**。

---

## 3. 两个反直觉发现

### 3.1 arXiv 没用（甚至有害）

这是最反直觉的。所有人都觉得 arXiv 是数学数据的天花板。DeepSeek 说：看 Table 8。

用 DeepSeek-LLM 1.3B 训 arXiv-only corpus：
- GSM8K: 2.9% → 2.7%（MathPile）或 3.3%（ArXiv-RedPajama）
- MATH: 3.0% → 3.3% 或 3.4%

基本没涨。用 DeepSeek-Coder 7B 训：
- GSM8K: 29.0% → 23.6% 或 28.1%（**下降**）
- MATH: 12.5% → 11.5% 或 11.1%（**下降**）

Table 9 的 miniF2F（形式化证明）也一样下降。

**Intuition**: arXiv 主要是 LaTeX 符号 manipulation 和高能物理、CS theory 等专业领域。competition math 是 structured problem solving，两者分布差太远。LaTeX parsing 本身消耗 model capacity，训完模型反而把已有的 reasoning 能力污染了。

作者很谨慎，列出三种未探索的可能：informalization 任务、与其他数据混合、更大模型。但至少在 1.3B 和 7B 上，arXiv 确实没用。

### 3.2 Code pre-training 帮 math

这是对 "code training improves reasoning" 这个 folk hypothesis 的第一个干净 ablation。

用 1.3B 模型做实验：

**Two-stage**:
- Code 400B → Math 150B: GSM8K 21.9%, MATH 15.3%, GSM8K+Python 17.4%
- General 400B → Math 150B: GSM8K 19.1%, MATH 14.4%, GSM8K+Python 14.3%

Code pre-training 不仅帮 tool-use math（明显，因为模型会写 Python），也帮 non-tool math（弱但显著）。

**One-stage mixed**:
- Code 400B + Math 150B 混着训: GSM8K 17.6%, MATH 12.1%

Mixed 反而更差。推测：1.3B 模型容量不够，同时吸收 code 和 math 的 reasoning gain 会互相干扰。更大模型可能两者兼得。

**Intuition**: Code 本质上是离散数学推理——状态追踪、edge case 处理、symbolic manipulation。即使没有 explicit math，coding 也在训练 reasoning 能力。Code → Math 的迁移比 General → Math 更顺。

---

## 4. DeepSeekMath-Base 7B 训练配方

- 初始化：DeepSeek-Coder-Base-v1.5 7B（不是 general LLM，是 code 模型）
- 500B tokens 继续训
- Data mix: 56% math web + 4% AlgebraicStack + 10% arXiv + 20% GitHub code + 10% NL CC
- AdamW, β₁=0.9, β₂=0.95, weight_decay=0.1
- LR: 2000 warmup → peak 4.2e-4 → 80% 时降到 31.6% peak → 90% 时降到 10% peak
- Batch 10M tokens, context 4K

结果（Table 2）：

| Model | Size | GSM8K | MATH |
|---|---|---|---|
| Minerva | 540B | 58.8% | 33.6% |
| Mistral | 7B | 40.3% | 14.3% |
| Llemma | 34B | 54.0% | 25.3% |
| **DeepSeekMath-Base** | **7B** | **64.2%** | **36.2%** |

7B 干过 540B 的 Minerva。这个 scaling 效率说明 **数据质量 > 模型规模**。

意外收获：MMLU 从 49.1% 涨到 54.9%，BBH 从 55.2% 涨到 59.5%。Math training 带动了 general reasoning。这说明 reasoning 是个 transferable 的 latent skill，不是 domain-specific 的 narrow 技能。

---

## 5. SFT 阶段

776K 训练样本，三种格式：
- **CoT** (Chain-of-Thought): 自然语言分步
- **PoT** (Program-of-Thought): 写 Python 求解
- **Tool-integrated**: 自然语言 + 程序混合

数据：English 用 GSM8K、MATH、MathInstruct subset、Lila-OOD；Chinese 用 K-12 76 个 sub-topics。

Training: examples 随机 concatenate 到 4K context，500 steps，batch 256，constant lr 5e-5。

结果（Table 5）：DeepSeekMath-Instruct 7B 在 MATH (CoT) 上 46.8%，超过 Qwen 72B 的 35.2%。MATH (tool-integrated) 57.4%，接近 GPT-4 Code Interpreter 的 69.7%。

---

## 6. GRPO：这篇 paper 的算法灵魂

### 6.1 PPO 哪里不好

PPO 是 RLHF 的标配。它的 objective 长这样（Eq 1）：

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}_{q, o} \frac{1}{|o|} \sum_t \min[\rho_t A_t, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) A_t]$$

变量：
- $\theta$: policy model 参数
- $\rho_t = \pi_\theta(o_t|q,o_{<t}) / \pi_{\theta_{old}}(o_t|q,o_{<t})$: importance ratio
- $A_t$: advantage，需要 value function $V_\psi$ 来算
- $\varepsilon$: clip 范围

PPO 的 reward（Eq 2）：

$$r_t = r_\varphi(q, o_{\leq t}) - \beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{ref}(o_t|q, o_{<t})}$$

问题三个：
1. $V_\psi$ 通常和 policy 一样大 → **memory 翻倍**
2. LLM RL 里 reward 一般只在 last token 给（outcome supervision），训 token-level 准确的 $V_\psi$ 很难
3. KL penalty 加在 reward 里，advantage 计算变复杂

### 6.2 GRPO 的核心 idea

**砍掉 value function，用 group statistics 当 baseline。**

对每个 question $q$，从 old policy $\pi_{\theta_{old}}$ 采样一组 outputs $\{o_1, ..., o_G\}$（论文 G=64）。Reward model 给每个 output 打分 $\mathbf{r} = \{r_1, ..., r_G\}$。

然后做 group normalization：

$$\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

这个 $\tilde{r}_i$ 就是 advantage 的核心。

**Intuition**: 对同一个题采 64 个答案，reward model 给它们排序。高于平均的强化，低于平均的抑制。这相当于 REINFORCE with batch-wise baseline，比 learned $V_\psi$ 简单太多。

更妙的是：reward model 的训练数据本来就是 pairwise comparison，group-relative 设置正好匹配。所以 RM 在 group-relative 下更可靠。

### 6.3 GRPO Objective (Eq 3)

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_t \left\{ \min[\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\}$$

关键改动：
- **KL penalty 直接加在 loss 里**，不加在 reward 里。这样 advantage 估计干净，不污染。
- 用 Schulman 的 unbiased KL estimator (Eq 4)：

$$\mathbb{D}_{KL} \approx \frac{\pi_{ref}}{\pi_\theta} - \log \frac{\pi_{ref}}{\pi_\theta} - 1$$

这个 estimator 保证非负（Schulman 2020: http://joschu.net/blog/kl-approx.html）。

### 6.4 Outcome vs Process Supervision

**Outcome Supervision (OS)**: reward 只在 output 末尾给。整个 output 的所有 token 共享同一个 normalized reward。

$$\hat{A}_{i,t} = \tilde{r}_i \quad \forall t$$

**Process Supervision (PS)**: reward 在每个 reasoning step 末尾给。对第 $i$ 个 output 的 $K_i$ 个 step，每个 step 给一个 reward。

Advantage 是当前 token 起所有后续 step reward 之和：

$$\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \tilde{r}_i^{\text{index}(j)}$$

**Intuition**: token $t$ 在 step $j$ 里，它对后续所有 step 的 reward 都有贡献。越靠后的 step advantage 越小（只含自己 step 的 reward）。这相当于 step 粒度的 Monte Carlo return。

实验（Figure 5）显示 GRPO+PS > GRPO+OS，跟 OpenAI "Let's verify step by step" (https://arxiv.org/abs/2305.20050) 的结论一致。

### 6.5 Iterative GRPO (Algorithm 1)

每个外层 iteration：
1. 把当前 policy 设成新的 reference model（防止 policy 漂太远 KL 失效）
2. 用当前 policy 采 64 个 output，reward model 打分
3. 算 group-relative advantage
4. 更新 policy
5. 用新数据 + 10% replay 持续训练 reward model

两轮 iteration（Figure 6），第一轮提升巨大，第二轮还有小幅提升。说明 RM 和 policy 的 co-evolution 有持续收益。

### 6.6 超参数

- RM 初始：DeepSeekMath-Base 7B，lr 2e-5
- Policy lr: 1e-6
- β (KL coef): 0.04
- G (group size): 64
- max length: 1024
- batch size: 1024 questions × 64 samples = 65K outputs
- Single policy update per exploration
- RL 训练数据：144K questions，只用 GSM8K + MATH 的 CoT 格式（故意限制 scope 看 OOD 效果）

### 6.7 结果

| Model | GSM8K | MATH | MGSM-zh | CMATH |
|---|---|---|---|---|
| Instruct 7B | 82.9% | 46.8% | 73.2% | 84.6% |
| RL 7B | **88.2%** | **51.7%** | **79.6%** | **88.8%** |

只在 GSM8K + MATH 上训 RL，MGSM-zh 和 CMATH 也涨——OOD 提升。

---

## 7. 统一范式：把所有方法塞进一个公式

作者给了一个很优雅的 unified view (Eq 5)：

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{(q,o) \sim \mathcal{D}} \left(\frac{1}{|o|} \sum_t GC(q, o, t) \nabla_\theta \log \pi_\theta(o_t|q, o_{<t})\right)$$

三个 axis：
1. **Data Source** $\mathcal{D}$: 数据从哪来（offline from SFT vs online from current policy）
2. **Reward Function**: reward 从哪来（rule-based vs model-based）
3. **Gradient Coefficient** $GC$: reward 怎么转成 per-token gradient weight

| Method | Data | Reward | GC |
|---|---|---|---|
| SFT | SFT data | — | 1 |
| RFT | offline, SFT model | rule | $\mathbb{I}(o)$ (对错 0/1) |
| Online RFT | online, current policy | rule | $\mathbb{I}(o)$ |
| DPO | offline pairs | preference | $\sigma(\beta \log \frac{\pi_\theta(o^-)}{\pi_{ref}(o^-)} - ...)$ |
| PPO | online | model | $A_t$（需要 $V_\psi$） |
| GRPO | online, group | model | $\hat{A}_{i,t} + \beta(\frac{\pi_{ref}}{\pi_\theta} - 1)$ |

### 7.1 关键 ablation (Figure 5)

用 DeepSeekMath-Instruct 1.3B 做 baseline：

- **RFT vs Online RFT**: 早期相当，后期 Online RFT 大幅领先。初期 policy ≈ SFT model，数据分布接近；后期 policy 漂移，online sampling 提供 in-distribution 数据。

- **Online RFT vs GRPO**: GRPO 领先。Online RFT 对正确答案统一 +1，对错误答案 +0（不惩罚）。GRPO 用 reward model 给 soft 分数，**对错误答案有 negative gradient**。这个差别很关键——光奖励正确的不够，还得主动抑制错误的。

- **GRPO+OS vs GRPO+PS**: PS 领先，step-level reward 提供更细粒度的 credit assignment。

---

## 8. 最有价值的 insight：RL 到底干了啥

Figure 7 这个诊断特别重要。

- **Pass@K**: 采 K 次，至少一次对就算对。衡量 "模型能不能生成正确答案"
- **Maj@K**: 采 K 次，majority vote。衡量 "模型是否把正确答案当主选项"

结果：
- RL 显著提升 Maj@K
- RL **不提升 Pass@K**

**Interpretation**: RL 没有提升 fundamental capability（生成正确答案的能力），只是把正确答案的概率 mass 从 "在 top-K 内" 移到 "在 top-1"。RL 是 **distribution shaping**：让正确答案变成 mode，但不动 mode 的 absolute capability。

类比 AlphaGo：RL 把 MCTS 探索出的好策略变成 default policy。在 LLM 里，SFT 已经把 capability 灌进去了，RL 把 "正确答案在 top-K" 变成 "正确答案在 argmax"。

**Practical implication**: 如果你的 use case 允许 self-consistency voting，RL 收益小；如果是 single-shot 推理，RL 收益大。

---

## 9. 局限

作者诚实地说：
- Geometry 和 theorem-proof 仍弱（pre-training 和 SFT 数据 bias）
- 7B 规模限制 few-shot 能力（GPT-4 能用 few-shot 提升，DeepSeekMath zero-shot ≈ few-shot）

未来方向（Section 5.2.3）：
1. **Data Source**: OOD prompts、tree search (https://arxiv.org/abs/2305.10601)、speculative decoding (https://arxiv.org/abs/2401.07851)、vLLM (https://arxiv.org/abs/2309.06180)
2. **Algorithms**: Noisy reward robustness（PRM800K 有 ~20% 错误标注），weak-to-strong generalization (https://arxiv.org/abs/2312.09390)
3. **Reward Function**: 泛化到 OOD、uncertainty quantification、高质量 process reward model

---

## 10. 给你的 take-away

1. **数据 pipeline 比数据 scale 重要**。120B 不是因为大才好，是因为 domain-aware iterative expansion 把 classifier 盲区补上了。OpenWebMath 13.6B 和 Proof-Pile-2 51.9B 都打不过。

2. **GRPO 本质是 REINFORCE with batch-wise baseline**。理论 high variance，但实践中：
   - LLM RL batch 巨大（65K outputs），baseline 估计够稳
   - RM 在 group-relative 下更准（训练数据就是 pairwise）
   - 省掉 $V_\psi$ 让 memory 减半，可以训更大 policy

3. **RL 不提升 capability，只 shaping distribution**。SFT 灌 capability，RL 把正确答案推到 argmax。如果你能 afford self-consistency voting，RL 的边际收益小。

4. **Math training 提升 general reasoning**（MMLU, BBH 都涨）。Reasoning 是 transferable latent skill。

5. **Code pre-training 帮 math reasoning**——第一个干净 ablation 证实 folk hypothesis。Tool-use math 上最强，non-tool math 上弱但显著。

6. **arXiv 反直觉无效**——data scale ≠ data quality，domain match > source authority。arXiv 的 LaTeX 符号 manipulation 和 competition math 分布差太远，训完反而污染已有 reasoning。

7. **统一范式很优雅**：SFT、RFT、DPO、Online RFT、PPO、GRPO 都可以塞进 $\nabla \log \pi \cdot GC$ 的形式，只是 Data Source、Reward、GC 三个 axis 不同。这让方法比较变得清晰。

参考链接：
- DeepSeekMath paper: https://arxiv.org/abs/2402.03300
- OpenWebMath: https://arxiv.org/abs/2310.06786
- Llemma/Proof-Pile-2: https://arxiv.org/abs/2310.10631
- Schulman KL approx: http://joschu.net/blog/kl-approx.html
- PPO: https://arxiv.org/abs/1707.06347
- Let's verify step by step (PRM800K): https://arxiv.org/abs/2305.20050
- Math-Shepherd: https://arxiv.org/abs/2312.08935
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Weak-to-strong generalization: https://arxiv.org/abs/2312.09390
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
- Speculative decoding survey: https://arxiv.org/abs/2401.07851

---

# DeepSeekMath 深度技术讲解

这篇 paper 来自 DeepSeek-AI，发表于 2024 年初。核心贡献有两个：(1) 通过精心设计的数据 pipeline 从 Common Crawl 中挖出 120B math tokens，让 7B 模型在 MATH benchmark 上达到 51.7%，接近 GPT-4 水平；(2) 提出 GRPO (Group Relative Policy Optimization)，省掉 PPO 中的 value model。我会从数据、pre-training、SFT、RL 四个层面展开，重点 build 你的 intuition。

**Paper link**: https://arxiv.org/abs/2402.03300  
**Code**: https://github.com/deepseek-ai/DeepSeek-Math

---

## 1. 核心定位：为什么这篇 paper 重要

GPT-4 在 MATH benchmark 上 ~52%，开源模型长期落后。DeepSeekMath 7B 用一个 7B 模型逼近这个数字，且不依赖 external tools 和 voting。更关键的是，他们把方法拆解得非常清楚：data pipeline、code pre-training 的作用、arXiv 的（反直觉）无效、GRPO 的设计——每一步都有 ablation 支撑，这种工程透明度在 2024 年的 LLM paper 中很罕见。

Figure 1 展示了关键结果：DeepSeekMath 7B 在 MATH 上 Top1 达 51.7%，self-consistency@64 达 60.9%。

---

## 2. Data Pipeline: 从 Common Crawl 挖出 120B math tokens

### 2.1 Iterative FastText Classifier

这是这篇 paper 最有工程价值的部分。流程（对应 Figure 2）：

**Iteration 0 (Seed)**:
- Seed corpus = OpenWebMath (13.6B tokens, 高质量 math web text)
- 训练 fastText classifier，参数：
  - vector dimension = 256
  - learning rate = 0.1
  - word n-gram max length = 3
  - min word occurrence = 3
  - epochs = 3
- 500K positive samples (from OpenWebMath) + 500K negative samples (random CC pages)
- 从 40B deduplicated HTML pages (URL dedup + near-dup) 中 recall
- 按 fastText score 排序，保留 top-40B tokens

**Iteration k (k≥1)**:
关键问题是 fastText 只能 recall 与 seed "长得像" 的数据，会漏掉很多 math domain。解决方案是 **domain-based seed expansion**：

1. 把整个 CC 按 base URL 组织成 disjoint domains
2. 对每个 domain，统计第 k 轮被收集的网页比例
3. 比例 > 10% 的 domain 视为 math-related（例如 mathoverflow.net）
4. 人工标注这些 domain 内具体的 math URL（例如 mathoverflow.net/questions）
5. 把这些 URL 对应的未收集网页加入 seed corpus
6. 重新训练 fastText，进入下一轮

4 轮后得到 35.5M math web pages，共 120B tokens。第 4 轮时 98% 的数据已经在第 3 轮收集完，所以停止。

**Intuition**: 这是 classic 的 bootstrapping——classifier 的 recall 受限于 seed 的 diversity。通过 domain-level 统计发现 "高 math 密度 domain"，再人工挑出该 domain 内的具体 URL pattern，相当于用人工先验补充 classifier 的盲区。这比简单放大 seed corpus 更高效，因为新 URL pattern 往往对应不同写作风格、不同数学子领域。

### 2.2 Decontamination

- 10-gram exact match 移除包含 benchmark 的网页（GSM8K, MATH, CMATH, AGIEval）
- 短于 10-gram 但 ≥ 3-gram 的 benchmark 文本，用 exact match 过滤

### 2.3 数据质量验证 (Table 1, Figure 3)

他们用 DeepSeek-LLM 1.3B 在不同 corpus 上各训 150B tokens 对比：

| Corpus | Size | GSM8K | MATH | CMATH |
|---|---|---|---|---|
| No Math | - | 2.9% | 3.0% | 12.3% |
| MathPile | 8.9B | 2.7% | 3.3% | 1.2% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 16.8% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 19.9% |
| DeepSeekMath Corpus | 120.2B | **23.8%** | **13.6%** | **41.5%** |

关键观察：
1. DeepSeekMath Corpus 在所有 benchmark 上领先
2. Multilingual: 中文 benchmark (CMATH) 也大幅领先，因为他们没有只收 English
3. Figure 3 显示 DeepSeekMath Corpus 在 50B tokens 时已超过 Proof-Pile-2 的完整 epoch，说明 **平均质量更高**

---

## 3. DeepSeekMath-Base 7B 的训练

### 3.1 配方

- 初始化：DeepSeek-Coder-Base-v1.5 7B（这是个重要选择，见 §3.3）
- 500B tokens 继续训练
- Data mix：
  - 56% DeepSeekMath Corpus (math web)
  - 4% AlgebraicStack (math code)
  - 10% arXiv
  - 20% GitHub code
  - 10% NL CC (English + Chinese)
- Optimizer: AdamW, β₁=0.9, β₂=0.95, weight_decay=0.1
- LR schedule: 2000 warmup steps → peak 4.2e-4 → 80% 训练时降到 31.6% peak → 90% 时降到 10% peak
- Batch size: 10M tokens, context 4K

### 3.2 结果 (Table 2, 3, 4)

| Model | Size | GSM8K | MATH | OCW | MMLU-STEM | CMATH |
|---|---|---|---|---|---|---|
| Minerva | 540B | 58.8% | 33.6% | 17.6% | 63.9% | - |
| Mistral | 7B | 40.3% | 14.3% | 9.2% | 51.1% | 44.9% |
| Llemma | 34B | 54.0% | 25.3% | 10.3% | 52.9% | 56.1% |
| **DeepSeekMath-Base** | **7B** | **64.2%** | **36.2%** | **15.4%** | **56.5%** | **71.7%** |

7B 模型在 MATH 上超过 540B 的 Minerva 2.6 个点。这个 scaling 效率说明：**数据质量 > 模型规模**，至少在 math domain 成立。

Table 4 的 MMLU/BBH 也提升（MMLU 49.1% → 54.9%, BBH 55.2% → 59.5%），说明 math training 不只是 narrow skill，而是带动 general reasoning。

### 3.3 为什么从 Code 模型初始化

Table 6 & 7 是这篇 paper 最有 insight 的 ablation 之一。用 DeepSeek-LLM 1.3B 做：

**Two-stage**:
- General 400B → Math 150B: GSM8K 19.1%, MATH 14.4%, GSM8K+Python 14.3%
- Code 400B → Math 150B: GSM8K 21.9%, MATH 15.3%, GSM8K+Python 17.4%

**One-stage (mixed)**:
- Math 150B only: GSM8K 20.5%, MATH 13.1%
- Code+Math mixed (400B code + 150B math): GSM8K 17.6%, MATH 12.1%, 但 HumanEval 29.3%, MBPP 39.4% (远超 two-stage 的 12.2%/17.0%)

**关键 insight**:
1. Code pre-training 同时帮助 tool-use math（明显）和 non-tool math（弱但显著）
2. Two-stage code→math 在 reasoning 上最优，但 code 能力灾难性遗忘
3. Mixed training 缓解遗忘，但 1.3B 模型容量不足以同时吸收 code+math 的 reasoning gain
4. 推测：在更大模型上 mixed training 可能两者兼得

**为什么 code 帮 math**：Code 包含大量 symbolic manipulation、状态追踪、edge case 处理。即使没有 explicit math，coding 本质上就是离散数学推理。这与 "code training improves reasoning" 的 folk hypothesis 一致，但首次在 math domain 给出干净 ablation。

### 3.4 arXiv 的反直觉发现 (Table 8, 9)

| Base | arXiv Corpus | GSM8K | MATH | MMLU-STEM |
|---|---|---|---|---|
| DeepSeek-LLM 1.3B | No math | 2.9% | 3.0% | 19.5% |
| DeepSeek-LLM 1.3B | MathPile (85% arXiv) | 2.7% | 3.3% | 15.7% |
| DeepSeek-LLM 1.3B | ArXiv-RedPajama | 3.3% | 3.4% | 9.0% |
| DeepSeek-Coder 7B | No math | 29.0% | 12.5% | 38.1% |
| DeepSeek-Coder 7B | MathPile | 23.6% | 11.5% | 35.8% |
| DeepSeek-Coder 7B | ArXiv-RedPajama | 28.1% | 11.1% | 35.2% |

arXiv 论文训练后，**MATH benchmark 反而下降**。作者诚实地列出三种未探索的可能：informalization 任务、与其他数据混合、更大模型。我的解读：arXiv 主要是 LaTeX 符号 manipulation 和专业领域（高能物理、CS theory），与 competition math（GSM8K, MATH 这种 structured problem solving）分布差距大。LaTeX parsing 本身消耗 model capacity。

---

## 4. SFT 阶段

776K 训练样本，三种 reasoning format：
- **CoT** (Chain-of-Thought): 自然语言分步推理
- **PoT** (Program-of-Thought): 写 Python 程序求解
- **Tool-integrated reasoning**: 自然语言 + 程序混合

数据组成：
- English: GSM8K, MATH (annotated with tool-integrated solutions), MathInstruct subset, Lila-OOD
- Chinese: K-12 problems，76 个 sub-topics，CoT + tool-integrated 双标注

Training: examples 随机 concatenate 到 4K context，500 steps，batch 256，constant lr 5e-5。

DeepSeekMath-Instruct 7B 结果 (Table 5):
- MATH (CoT, no tool): 46.8%，超过 Qwen 72B (35.2%), MetaMath 70B (26.6%)
- MATH (tool-integrated): 57.4%，接近 GPT-4 Code Interpreter (69.7%)

---

## 5. GRPO: 这篇 paper 的算法核心

### 5.1 PPO 的问题

PPO 的 objective (Eq 1):

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)} \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left[\rho_t A_t, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) A_t\right]$$

变量含义：
- $\theta$: policy model 参数
- $\theta_{old}$: 上一轮 exploration 时的 policy 参数（frozen during update）
- $q$: question，从 question distribution $P(Q)$ 采样
- $o$: output 序列，长度 $|o|$，token $o_t$ 在位置 $t$
- $\pi_\theta(o_t|q, o_{<t})$: 当前 policy 给出 token $o_t$ 的概率（给定 question 和已生成 prefix）
- $\rho_t = \pi_\theta(o_t|q,o_{<t}) / \pi_{\theta_{old}}(o_t|q,o_{<t})$: importance sampling ratio
- $\varepsilon$: clipping 范围（典型 0.1-0.2），限制单步 policy 更新幅度
- $A_t$: advantage，通过 GAE 基于 reward $\{r_{\geq t}\}$ 和 value function $V_\psi$ 计算

PPO 的 reward (Eq 2):

$$r_t = r_\varphi(q, o_{\leq t}) - \beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{ref}(o_t|q, o_{<t})}$$

- $r_\varphi$: reward model
- $\pi_{ref}$: reference model（通常是 SFT 模型）
- $\beta$: KL penalty 系数

**问题**:
1. $V_\psi$ 通常和 policy 同等大小 → memory 翻倍
2. LLM RL 中 reward 通常只在 last token 给出（outcome supervision），训练 token-level 准确的 $V_\psi$ 困难
3. KL penalty 加在 reward 里，让 advantage 计算复杂

### 5.2 GRPO 的核心 idea (Figure 4)

**抛弃 value function，用 group statistics 当 baseline。**

对每个 question $q$，从 $\pi_{\theta_{old}}$ 采样一组 outputs $\{o_1, o_2, ..., o_G\}$（论文 G=64）。然后用 reward model 给每个 output 打分 $\mathbf{r} = \{r_1, ..., r_G\}$。

**Group-normalized reward**:
$$\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

这就是 advantage 的核心。这相当于在 batch 内做 baseline subtraction，理论上等价于 REINFORCE with batch-wise baseline，但比 learned $V_\psi$ 简单得多。

**Intuition**: 对同一个 question，采样 64 个回答，reward model 给它们排序。reward 高于平均的 → 强化；低于平均的 → 抑制。group 内的 relative comparison 正好匹配 reward model 的训练数据形式（pairwise comparison），所以 reward model 在 group-relative 设置下更可靠。

### 5.3 GRPO Objective (Eq 3)

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)\right] \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left[\rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}\right] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\}$$

- $\rho_{i,t} = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})$: 第 $i$ 个 output 在 token $t$ 的 importance ratio
- $\hat{A}_{i,t}$: group-relative advantage，具体见 §5.4 和 §5.5
- $\beta$: KL coefficient (论文 0.04)
- **KL penalty 直接加在 loss 里**，不加在 reward 里。这避免了 advantage 估计的偏差

### 5.4 KL 估计 (Eq 4)

$$\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \approx \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1$$

这是 Schulman 2020 (http://joschu.net/blog/kl-approx.html) 提出的 unbiased estimator：

设 $r = \pi_{ref}(x) / \pi_\theta(x)$，则
$$\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] = \mathbb{E}_{x \sim \pi_\theta}\left[\log \frac{\pi_\theta(x)}{\pi_{ref}(x)}\right] = \mathbb{E}_{x \sim \pi_\theta}[-\log r]$$

直接估计 $-\log r$ 是 high variance（当 $r$ 很小时）。Schulman 的技巧：
$$-\log r = -(r - 1) + (r - 1) - \log r = -(r - 1) + \log(1/r \cdot e^{r-1})$$

近似 $f(r) = r \log r - r + 1 \geq 0$（KL 的标准非负形式），于是
$$\mathbb{E}[-\log r] \approx \mathbb{E}[-(r-1) + f(r)]$$

实践中用 $\hat{D}_{KL} = r - \log r - 1$，即 $r - \log r - 1 \geq 0$ 当 $r > 0$。这保证 KL 项始终非负。

### 5.5 Outcome Supervision (OS) vs Process Supervision (PS)

**Outcome Supervision**: reward 只在 output 末尾给出。

对一组 outputs $\{o_1, ..., o_G\}$，reward model 给出 $\mathbf{r} = \{r_1, ..., r_G\}$。

$$\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})} \quad \forall t$$

整个 output 的所有 token 共享同一个 normalized reward。

**Process Supervision**: reward 在每个 reasoning step 末尾给出。

对第 $i$ 个 output，有 $K_i$ 个 step。step $j$ 末尾 token 的 index 记为 $\text{index}(j)$。

reward model 给出每个 step 一个分数：
$$\mathbf{R} = \{\{r_1^{\text{index}(1)}, ..., r_1^{\text{index}(K_1)}\}, ..., \{r_G^{\text{index}(1)}, ..., r_G^{\text{index}(K_G)}\}\}$$

normalize:
$$\tilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$$

advantage 是从当前 token 起的所有后续 step reward 之和：
$$\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \tilde{r}_i^{\text{index}(j)}$$

**Intuition**: token $t$ 在 step $j$ 内，它对后续所有 step 的 reward 都有贡献（credit assignment）。越靠后的 step，advantage 越小（只包含自己 step 的 reward）。这相当于 token-level 的 Monte Carlo return，但 step 粒度而非 token 粒度。

实验（Figure 5）显示 GRPO+PS > GRPO+OS，符合 OpenAI "Let's verify step by step" (https://arxiv.org/abs/2305.20050) 的发现。

### 5.6 Iterative GRPO (Algorithm 1)

```
Input: π_θ_init, r_φ, dataset D, hyperparams ε, β, μ

1: π_θ ← π_θ_init
2: for iteration = 1, ..., I do
3:   π_ref ← π_θ  // 当前 policy 作为新的 reference
4:   for step = 1, ..., M do
5:     Sample batch D_b from D
6:     π_θ_old ← π_θ
7:     Sample G outputs {o_i} ~ π_θ_old(·|q) for each q ∈ D_b
8:     Compute rewards {r_i} via r_φ
9:     Compute Â_{i,t} via group relative advantage estimation
10:    for GRPO iter = 1, ..., μ do
11:      Update π_θ by maximizing GRPO objective
12:      Update r_φ via continuous training (10% replay buffer)
```

关键设计：
- **Reference refresh** (line 3): 每个外层 iteration 重置 reference model = 当前 policy，避免 policy drift 太远后 KL 变成主要 objective
- **Reward model 持续训练** (line 12): 用当前 policy 生成的新 data 训练 RM，加 10% replay 防止 catastrophic forgetting

### 5.7 训练超参数

- RM 初始：DeepSeekMath-Base 7B，lr 2e-5
- Policy lr: 1e-6
- β (KL coef): 0.04
- G (group size): 64
- max length: 1024
- batch size: 1024 (即 1024 个 question，每个 64 个 sample，共 64K outputs per batch)
- Single policy update per exploration（即 line 10 的 μ=1）
- RL 训练数据：144K questions，只用 GSM8K + MATH 的 CoT 格式（**故意限制数据 scope**，看 OOD 效果）

### 5.8 结果 (Table 5)

| Model | GSM8K | MATH | MGSM-zh | CMATH |
|---|---|---|---|---|
| DeepSeekMath-Instruct 7B | 82.9% | 46.8% | 73.2% | 84.6% |
| DeepSeekMath-RL 7B | **88.2%** | **51.7%** | **79.6%** | **88.8%** |

只在 GSM8K + MATH 上 RL 训练，MGSM-zh 和 CMATH 也提升（OOD 提升），说明 RL 学到的不只是 narrow skill。

---

## 6. 统一范式：SFT, RFT, DPO, Online RFT, PPO, GRPO 的统一视角

### 6.1 通用梯度公式 (Eq 5)

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}\left[\underbrace{(q, o) \sim \mathcal{D}}_{\text{Data Source}}\right] \left(\frac{1}{|o|} \sum_{t=1}^{|o|} \underbrace{GC_{\mathcal{R}}(q, o, t, \pi_{rf})}_{\text{Gradient Coefficient}} \nabla_\theta \log \pi_\theta(o_t | q, o_{<t})\right)$$

三个 axis：
1. **Data Source** $\mathcal{D}$: 训练数据从哪来（offline from SFT model vs online from current policy）
2. **Reward Function** $\pi_{rf}$: reward 从哪来（rule-based vs model-based）
3. **Algorithm → Gradient Coefficient** $GC$: 怎么把 reward 转成 per-token gradient weight

### 6.2 各方法对比 (Table 10)

| Method | Data Source | Reward | Gradient Coefficient |
|---|---|---|---|
| SFT | $q, o \sim P_{sft}(Q, O)$ | — | 1 |
| RFT | $q \sim P_{sft}(Q), o \sim \pi_{sft}(O\|q)$ | Rule | $\mathbb{I}(o)$ (Eq 10) |
| DPO | $q \sim P_{sft}(Q), o^+, o^- \sim \pi_{sft}$ | Rule/Preference | $\sigma(\beta \log \frac{\pi_\theta(o^-)}{\pi_{ref}(o^-)} - \beta \log \frac{\pi_\theta(o^+)}{\pi_{ref}(o^+)})$ (Eq 14) |
| Online RFT | $q \sim P_{sft}(Q), o \sim \pi_\theta(O\|q)$ | Rule | $\mathbb{I}(o)$ |
| PPO | $q \sim P_{sft}(Q), o \sim \pi_\theta$ | Model | $A_t$ (Eq 18) |
| GRPO | $q \sim P_{sft}(Q), \{o_i\}_{i=1}^G \sim \pi_\theta$ | Model | $\hat{A}_{i,t} + \beta(\frac{\pi_{ref}}{\pi_\theta} - 1)$ (Eq 21) |

### 6.3 关键对比实验 (Figure 5)

用 DeepSeekMath-Instruct 1.3B 做 baseline：

- **RFT vs Online RFT**: 早期相当，后期 Online RFT 大幅领先。因为初期 policy ≈ SFT model，offline 和 online 数据分布接近；后期 policy 漂移后，online sampling 提供 in-distribution 数据
- **Online RFT vs GRPO**: GRPO 领先。Online RFT 的 GC = $\mathbb{I}(o)$，对正确答案统一 +1，对错误答案 +0（不惩罚）。GRPO 用 reward model 给 soft 分数，**对错误答案有 negative gradient**
- **GRPO+OS vs GRPO+PS**: PS 领先，因为 step-level reward 提供更细粒度的 credit assignment

### 6.4 Iterative RL (Figure 6)

两轮 iteration，第一轮提升巨大，第二轮仍有小幅提升。说明 RM 和 policy 的 co-evolution 有持续收益。

---

## 7. 为什么 RL 有效：Maj@K vs Pass@K (Figure 7)

这是个非常重要的诊断。

- **Pass@K**: 采样 K 次，只要至少一次正确就算对。衡量 "模型能不能生成正确答案"
- **Maj@K**: 采样 K 次，取多数投票，看 majority 是否正确。衡量 "模型是否把正确答案当作主选项"

Figure 7 显示：
- RL 提升 Maj@K（显著）
- RL **不提升 Pass@K**

**Interpretation**: RL 没有提升 fundamental capability（生成正确答案的能力），而是把正确答案的概率 mass 从 "在 top-K 内" 移到 "在 top-1"。换句话说，RL 是 **distribution shaping**：让正确答案变成 mode，但不动 mode 的 absolute capability。

这与 Singh et al. (https://arxiv.org/abs/2306.17492) 的 "alignment tax" / "preference alignment" 视角一致：SFT 模型的 distribution 与 "把正确答案放第一" 的 preference 不对齐，RL 修复这个 misalignment。

**Practical implication**: 如果你的 use case 允许 self-consistency voting，RL 收益小；如果是 single-shot 推理，RL 收益大。

---

## 8. 局限和未来方向

### 8.1 已知局限

- Geometry 和 theorem-proof 仍弱于 GPT-4（pre-training 和 SFT 数据 bias）
- 受限于 7B 规模，few-shot 能力不如 GPT-4（GPT-4 能用 few-shot 提升，DeepSeekMath zero-shot ≈ few-shot）

### 8.2 未来方向 (Section 5.2.3)

1. **Data Source**: 
   - OOD prompts（扩展 beyond GSM8K/MATH）
   - Advanced sampling: tree search (https://arxiv.org/abs/2305.10601), speculative decoding (https://arxiv.org/abs/2401.07851)
   - Efficient inference: vLLM/PagedAttention (https://arxiv.org/abs/2309.06180)

2. **Algorithms**:
   - Noisy reward robustness（PRM800K 有 ~20% 标注错误，参考 https://arxiv.org/abs/2305.20050）
   - Weak-to-strong generalization (https://arxiv.org/abs/2312.09390)

3. **Reward Function**:
   - Generalization to OOD questions
   - Uncertainty quantification（连接 weak RM 和 weak-to-strong algorithms）
   - 高质量 process reward model

---

## 9. 对你的 Take-away

如果你是 Karpathy 在思考这些：

1. **GRPO 的本质是 REINFORCE with batch-wise baseline**。理论上是 high variance 的（baseline 是 batch mean 而非 learned $V_\psi$），但实践中：
   - LLM RL 的 batch 通常很大（1024 questions × 64 samples = 65K outputs）
   - Reward model 在 group-relative 设置下更准（其训练数据就是 pairwise）
   - 省掉 $V_\psi$ 让 memory 减半，可以训更大 policy

2. **RL 不提升 capability，只 shaping distribution**。这与 AlphaGo 的 RL 经验类似——RL 主要是把 MCTS 探索出的好策略变成 default policy。在 LLM 中，SFT 已经把 capability 灌进去了，RL 把 "正确答案在 top-K" 变成 "正确答案在 argmax"。

3. **Math training 提升 general reasoning**（MMLU, BBH）暗示 reasoning 是一个 transferable 的 latent skill，而非 domain-specific。

4. **Code pre-training 帮 math reasoning** 给 "code improves reasoning" folk hypothesis 提供了第一个干净的 ablation。这个 effect 在 tool-use math 上最强，non-tool math 上弱但显著。

5. **arXiv 反直觉无效** 提醒我们 data scale ≠ data quality，且 domain match > source authority。

参考链接：
- DeepSeekMath paper: https://arxiv.org/abs/2402.03300
- OpenWebMath: https://arxiv.org/abs/2310.06786
- Proof-Pile-2 / Llemma: https://arxiv.org/abs/2310.10631
- Schulman KL approx: http://joschu.net/blog/kl-approx.html
- PPO: https://arxiv.org/abs/1707.06347
- Let's verify step by step (PRM800K): https://arxiv.org/abs/2305.20050
- Math-Shepherd: https://arxiv.org/abs/2312.08935
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Weak-to-strong generalization: https://arxiv.org/abs/2312.09390
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
- Speculative decoding survey: https://arxiv.org/abs/2401.07851
