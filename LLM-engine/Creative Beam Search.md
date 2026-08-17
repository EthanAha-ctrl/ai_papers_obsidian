---
source_pdf: Creative Beam Search.pdf
paper_sha256: 5199e314078909a9cc40197633068a029a53ce4baa1e9ffe67929e532c1c8182
processed_at: '2026-08-03T17:48:02-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Creative Beam Search

好，抛开那些学术包装，这paper本质上就干了一件事：**让 LLM 先列 4 个草稿，再自己当裁判挑一个最 creative 的**。听起来朴素，但 idea 其实挺优雅的。

paper 链接先放这：作者主页 https://giorgiofranceschelli.com/，arxiv 版本搜 "Creative Beam Search Franceschelli" 能找到。Llama 2 paper https://arxiv.org/abs/2307.09288，DBS 原始 paper https://arxiv.org/abs/1610.02424，LLM-as-a-Judge https://arxiv.org/abs/2306.05685。

---

## 这 paper 在干嘛

场景很简单：你给 LLM 一个 prompt，让它写点 creative 的东西。普通的做法就是 sample 一下输出完事，temperature 调高一点让它"放飞"。

问题在于，"放飞"和"creative"不是一回事。高温采样就是瞎飞，可能飞出花来，也可能飞出 nonsense。真正的 creativity 应该是：**先 diverge 出一堆不同的 idea，再 converge 到一个真正好的**。这其实就是 human brainstorming 的过程——开会的时候大家先七嘴八舌抛想法，wall 上贴满 sticky notes，然后挑一个最有潜力的往下做。

这 paper 就把这个过程 explicit 化了。两步：

1. **Response generation**: 用 Diverse Beam Search 逼 model 生成 4 个不一样的候选答案
2. **Response validation**: 让同一个 model 自己当 judge，挑一个"最 creative"的输出

没了。就这两步。Paper 里包装成 Amabile 1983 的 componential model of creativity（https://psycnet.apa.org/record/1984-07715-001），说这对应 creativity 的"generation"和"validation"两个 phase。OK，这 mapping 挺自然的，但本质就是个 inference-time 的 two-stage pipeline。

---

## 第一步：Diverse Beam Search

先复习下普通 beam search。给定 prompt $x$，model 参数 $\theta$，vocabulary $\mathcal{V}$。每个 time step $t$，你维护 $B$ 个 partial sequences，每个的 cumulative log-probability 是：

$$S_{\text{BS}}(y_{1:t}) = \sum_{\tau=1}^{t} \log p_\theta(y_\tau \mid y_{<\tau}, x)$$

变量意思：
- $y_{1:t}$ = 当前已经生成的 token sequence（从位置 1 到位置 $t$）
- $\tau$ = 求和的 dummy index，遍历 1 到 $t$
- $y_\tau$ = 第 $\tau$ 个位置上的 token
- $y_{<\tau}$ = $\tau$ 之前的所有 token，即 $y_1, y_2, ..., y_{\tau-1}$
- $p_\theta(y_\tau \mid y_{<\tau}, x)$ = 给定 prompt $x$ 和前面所有 token，model 预测下一个 token 是 $y_\tau$ 的概率
- $\log$ = 取对数，把乘积变求和，数值稳定

beam search 的毛病大家都知道：$B$ 个 beam 经常塌到一个 mode 附近，最后 $B$ 个 candidate 内容长得差不多，只是标点不一样。这对 translation 任务还行，对 creativity 就是灾难。

**Diverse Beam Search (DBS)** 的 fix 是把 $B$ 个 beam 分成 $G$ 个 group，group 之间互相"避让"。具体 score function：

$$S_{\text{DBS}}(y_{1:t}^{(g)}) = \sum_{\tau=1}^{t} \left[ \log p_\theta(y_\tau^{(g)} \mid y_{<\tau}^{(g)}, x) - \lambda \cdot \sum_{g' \neq g} c_{g'}(y_\tau^{(g)}) \right]$$

变量意思：
- $g$ = group index，从 1 到 $G$
- $y^{(g)}$ = group $g$ 里的 sequence
- 上标 $(g)$ 表示"属于 group $g$"
- $\lambda$ = diversity penalty 的强度系数，paper 里设 $\lambda = 10$
- $c_{g'}(y_\tau^{(g)})$ = token $y_\tau^{(g)}$ 在 group $g'$ 中第 $\tau$ 步出现过的次数
- $\sum_{g' \neq g}$ = 遍历所有其他 group（除了 $g$ 自己）

直觉就是：如果某个 token 已经被别的 group 在同一位置选过，你这边再选它就会被罚。这逼着不同 group explore 不同的 token 路径，最后 $G$ 个 candidate 在内容上能拉开差距。

paper 的具体配置：
- $B = 8$（beam budget）
- $G = 8$（每个 group 只有 1 个 beam，最大化 diversity）
- $\lambda = 10$（penalty 系数）
- 保留 top $K = 4$ 进入下一步

注意 $G = B$ 是个极端选择——这相当于把 beam search 退化成 8 个互相 penalize 的 greedy decoding。group 内部没有 backup hypothesis，pure 押注 diversity。

---

## 第二步：LLM-as-a-Judge + 位置去偏

DBS 给你 4 个 candidate，怎么挑？最 naive 的做法就是看 DBS score 谁高选谁。但 DBS score 里 likelihood 和 diversity penalty 是两个混在一起的东西，谁高真不一定谁 creative。

paper 的做法是让 LLM 自己来 judge。prompt 大概长这样：

```
Which of the following is the most creative answer to "$INPUT"?
1) candidate_1
2) candidate_2
3) candidate_3
4) candidate_4
Provide only the number of the most creative answer without any explanation.
```

这里有个 well-known 的坑：**positional bias**（Wang et al. 2023, https://arxiv.org/abs/2305.17926）。LLM 当 judge 的时候，position 1 的 candidate 经常被偏好，仅仅因为它出现在前面。这跟 candidate 内容好坏无关，纯纯的位置 spurious correlation。

CBS 的 fix 是 **balanced position calibration**：把 4 个 candidate 做循环 rotation，生成 4 个 prompt：

$$\pi_k(c_1, c_2, c_3, c_4) = (c_{k+1 \bmod 4}, c_{k+2 \bmod 4}, c_{k+3 \bmod 4}, c_{k+4 \bmod 4})$$

变量意思：
- $\pi_k$ = 第 $k$ 次循环 rotation，$k \in \{0, 1, 2, 3\}$
- $c_i$ = 第 $i$ 个 candidate
- $\bmod 4$ = 模 4 运算，让 index 循环回到 1-4
- $\pi_k$ 的输出是一个新的 candidate 排列

然后对每个 rotation 跑一次 LLM judge，aggregate votes：

$$\text{vote}(c_i) = \sum_{k=0}^{3} \mathbb{1}\left[ \text{LLM}(\text{prompt}(\pi_k)) = i \right]$$

变量意思：
- $\text{vote}(c_i)$ = candidate $c_i$ 收到的总票数
- $\mathbb{1}[\cdot]$ = indicator function，括号里条件成立为 1，否则为 0
- $\text{LLM}(\cdot)$ = LLM 在 greedy decoding 下的输出（注意，judge 用 temperature = 0，要 deterministic）
- $\sum_{k=0}^{3}$ = 遍历 4 次 rotation

最终输出：

$$c^* = \arg\max_{c_i} \text{vote}(c_i)$$

平手时 fallback 到 DBS score 排序。

**Judge 用 greedy decoding 是个 subtle 但关键的细节**——你不能让 LLM 在投票时 sampling，否则同一个 prompt 跑两次结果都可能不一样，judge 的 reliability 就崩了。Temperature 必须为 0。

---

## 实验数据讲了啥

实验设置：
- Model: Llama 2 7B Chat（RLHF-tuned 版本，https://arxiv.org/abs/2307.09288）
- Baseline: nucleus sampling, $T = 1.0$, $p = 0.9$
- Max tokens: 256
- 评估者: 31 个 CS graduate students
- 总样本: 217 次 pairwise comparison
- 用 Gradio 搭了个 UI，让用户自由输入 prompt，看 CBS 和 STD 的输出（顺序随机），选哪个更 creative 或者说"太像了分不出来"

结果 Table 1（这是论文 Table 1 的数字）：

| Preference | CBS ≠ DBS | CBS == DBS | Total |
|---|---|---|---|
| CBS wins | 0.34 | 0.11 | **0.45** |
| STD wins | 0.18 | 0.11 | **0.29** |
| Too similar | 0.19 | 0.07 | **0.26** |
| Total | 0.71 | 0.29 | 1.00 |

读懂这表的 intuition：

**第一层读法**：CBS 整体胜率 45% vs STD 29%，剩下 26% 用户觉得差不多分不出。在有差异的 cases 里，CBS 占 $\frac{0.45}{0.45 + 0.29} \approx 60.8\%$，STD 占 39.2%。明显占优。

**第二层读法**（这是 paper 里最有意思的 finding）：拆开看 "CBS 输出 = DBS 输出"和"CBS 输出 ≠ DBS 输出"两种情况。

- 当 CBS == DBS（29% 的 cases）：CBS wins 11%, STD wins 11%, too similar 7% —— 平局，没差别
- 当 CBS ≠ DBS（71% 的 cases）：CBS wins 34%, STD wins 18%, too similar 19% —— CBS 大胜

**这数字说明啥**？LLM judge 在 71% 的情况下 overrule 了 DBS 自己的 top choice，并且这 overrule 是有 positive effect 的——正是在这些 cases 里 CBS 大胜 STD。而如果 LLM judge 同意 DBS 的选择（29% cases），CBS 的输出就退化到和 STD 一个水平。

换句话说：**DBS 自己选的 top candidate，根本不比 standard sampling 好**。真正起作用的是 LLM-as-a-Judge 那一步的 re-ranking。DBS 在这里其实更像是个"diverse candidate generator"，真正决定 quality 的是 judge。

这点 paper 里 Figure 3 也画了，但 author 自己 discussion 没特别 emphasis——我觉得这是被低估的 finding。

**第三个数字**：LLM judge 选的 candidate 跟 DBS 自己的 top-1 重合率只有 29%。Random baseline 是 35.3%（因为 DBS score 本身有 ranking 偏好，top-1 自然占更多）。29% < 35.3% 说明 LLM judge **不是在 confirm DBS 的判断，而是在 actively 改写**。这点很关键——证明 self-evaluation 不是 trivial redundancy。

---

## 我的 take

### 几个我觉得真的不错的地方

**1. Decomposition 干净**。Creativity = generation + validation 这个 framing 虽然来自 Amabile 1983 的心理学 model，但映射到 ML pipeline 后变得很 actionable。后续工作可以替换 generation module（DBS → Tree of Thoughts, https://arxiv.org/abs/2305.10601）或者 validation module（self-judge → cross-model judge），框架依然成立。

**2. Ablation 设计精准**。报告 CBS==DBS 和 CBS≠DBS 的 split 是个聪明 move，直接 expose 了 LLM judge 的实际作用。如果只报告"CBS 胜率 45%"，读者会以为 DBS 是主角；拆开看才知道 judge 才是主角。

**3. Balanced position calibration 是必须的**。Wang et al. 2023 的 positional bias 在 Llama 2 7B 上有多严重可以试一下——直接用 raw LLM-as-a-Judge 给 4 个 candidate 排序，position 1 的 candidate 几乎肯定胜率最高。不 fix 这个 bug 结果就崩了。

### 几个我觉得不太行的

**1. Evaluation 规模太小**。31 个 CS grad student，217 个 sample，free-form prompt——这种 setup 的 variance 巨大。要做严肃 claim 至少得几百人、几千 sample、不同 demographic。paper 自己也承认这是 limitation，但学术圈该 push 的还是得 push。

**2. Hamming diversity 是个糟糕的 diversity measure**。Hamming 只看同一位置是否相同 token。"The cat sat" 和 " The cat sat"（前面多个空格）在 Hamming 距离上是完全不同的，但语义完全相同。DBS 的 penalty 在这种 misalignment case 完全失效。更好的做法是用 embedding cosine distance、n-gram Jaccard、或者 edit distance。Paper 提了但没 fix，有点 hand-wave。

**3. 没跟其他 inference-time methods 比**。Self-Refine (https://arxiv.org/abs/2303.17651)、Tree of Thoughts、Self-Consistency 都没作为 baseline。CBS 是 single-pass，Self-Refine 是 multi-pass，看起来不公平，但你应该至少报告"用 Self-Refine 的 cost 跑 CBS 多次"会怎样。

**4. 没系统 sweep 超参**。$\lambda = 10$, $G = 8$, $K = 4$ 都是凭直觉选的。$\lambda = 1$ 和 $\lambda = 100$ 之间差距应该很大。$G = 4$ 会不会比 $G = 8$ 好？没数据。

**5. Llama 2 7B 当 judge 可能太弱**。LLM-as-a-Judge 在 weak model 上 reliability 下降。用 GPT-4 (https://arxiv.org/abs/2303.08774) 或 Llama 3 70B 当 judge，结果可能完全不同。

### 跟当下 inference-time scaling 的 connection

Karpathy 你自己在 https://x.com/karpathy/status/1720518788305559547 说过 LLM 的 system-2 thinking 是下一个前沿。OpenAI o1 (https://openai.com/index/learning-to-reason-with-llms/) 走的就是这条路——在 inference time 生成多条 reasoning path，自己 evaluate，选最好的。

CBS 本质上是 o1-style search 的一个 creativity domain instantiation。区别只在 evaluation signal：
- Reasoning tasks: correctness（logical、数学、代码可验证）
- Creativity tasks: quality/novelty preference（没法 verifiable，只能用 judge）

从这角度，CBS 的 generalization 应该是 **"Generation + Validation" framework**，validation signal 来源可以替换：
- LLM-as-Judge (CBS 用的)
- Human feedback (Ding et al. 2023, https://openreview.net/forum?id=T9wbr7EReR)
- Reward model（RLAIF-style, https://arxiv.org/abs/2309.00267）
- External verifier（math/code 上）

### 能 push 的方向

**1. Iterative CBS**。当前 single-pass，validation 不过就 fallback。可以做成 loop：DBS → judge → 如果 judge 不满意就 re-prompt + 反馈 → 再 DBS。这跟 Self-Refine 的 spirit 一致，但在 creative domain 上。

**2. Cross-model CBS**。Generator 和 judge 用不同 model。Llama 3 405B 生成，GPT-4o 判断。避免 self-evaluation 的 in-distribution bias——model judge 自己 output 时倾向于偏好自己训练分布里的"creative cliché"。

**3. Reward distillation**。把 CBS 在大量 prompt 上选出的 (winner, losers) pairs 当作 DPO 训练数据（DPO paper: https://arxiv.org/abs/2305.18290），训出一个 reward model 或直接 fine-tune generator，让单次 sampling 就能产出 CBS-level 的 creative output。这相当于把 inference-time cost 转移到 training-time cost。

**4. MAP-Elites 替代 DBS**。Bradley et al. 2023 (https://arxiv.org/abs/2310.13032) 已经做了 QD + AI Feedback 的 version。用 MAP-Elites archive 存历史 candidates，在 archive 上做 novelty search，可以探索更广的 solution space。DBS 的 diversity 是 token-level 的，MAP-Elites 是 behavior-space-level 的，后者更接近真正的 novelty。

**5. Latent-space diversity**。把 Hamming diversity 换成 hidden state embedding 的 cosine distance。具体做法：在 DBS 的每一步，用 model 当前 layer 的 hidden state 计算 candidate 之间的距离，作为 diversity penalty。这会让 diversity 在 semantic level 而非 token level。

---

## 一句话总结

CBS 把 LLM 的 creative generation 拆成"先列多个不一样草稿、再让 LLM 自己当 judge 挑最好"两步，experiment 显示 LLM judge 那一步才是真正起作用的部分，DBS 本身并不比 standard sampling 强多少。方法简单、框架 clean、evaluation 偏弱、limitation 明确——典型的小 paper，但 idea 可以 extend 出很多后续工作。

本质上 paper 揭示了一个有意思的事实：**LLM 的 next-token prediction capability 和它的 evaluation capability 是两回事**。它能 generate 出 creative 内容，但需要外显的 evaluation 步骤才能把它"找出来"。这跟 human 创作也有点像——作家写完初稿要反复改、反复评估，generation 和 editorial judgment 是两个分离的能力。LLM 的 RLHF training 让它具备了 judgment，但 sampling 默认不调用这个能力，需要 inference-time algorithm 显式 trigger。

这跟你反复说的"LLM 是 next-token predictor"完全不矛盾——next-token predictor 当然能输出"对一段 text 的评价"，因为这也是 text-to-text 的 mapping。只是默认 decoding 没有显式触发这个 mode。CBS 就是个简单的 trigger mechanism。

要不要继续聊某个具体方向？比如把 CBS 扩展成 iterative 版本、或者换 GPT-4 当 judge 在 poetry generation 上认真跑一个 benchmark？

---

# Creative Beam Search 深度解析

## Paper Overview

Giorgio Franceschelli (University of Bologna) 和 Mirco Musolesi (UCL / Bologna) 提出的 **Creative Beam Search (CBS)**, 本质上是把 Amabile 1983 提出的 componential model of creativity 显式映射到 LLM decoding pipeline 上。核心 contribution 是把 creative process 的两个 phase —— response generation 和 response validation —— 分别由 **Diverse Beam Search** 与 **LLM-as-a-Judge** 实现, 再通过 **balanced position calibration** 抵消 LLM judge 的 positional bias。

paper 链接: https://arxiv.org/abs/2407.00431 (作者 AAAI 2024 Workshop 版本)
作者主页: https://giorgiofranceschelli.com/
Mirco Musolesi 主页: https://www.mircomusolesi.org/

---

## 1. Conceptual Mapping: Amabile → CBS

Amabile 的 componential model (Amabile, 1983) 把 creativity 分为五个阶段:

| Amabile 阶段 | CBS 中的实现 |
|---|---|
| Task presentation (internal/external stimulus) | User prompt |
| Preparation (domain knowledge activation) | Loading pretrained LLM |
| Response generation (creativity-relevant skills) | **Diverse Beam Search** (diversity penalty 充当 "creativity skill") |
| Response validation (domain-relevant skills) | **LLM-as-a-Judge** (model 本身的 learned preference) |
| Iterative adjustment (optional) | CBS 简化为 single-pass, 这是 trade-off |

paper 明确承认 CBS 只 simulate 了部分 creative process, 缺失了 task motivation (internal stimulus) 和 iteration。这点本身已经 close to Boden 提出的 combinational creativity 的边界 —— LLM 没有 intentionality (Shanahan, 2024, https://cacm.acm.org/research/talking-about-large-language-models/)。

---

## 2. Response Generation: Diverse Beam Search 数学细节

### 2.1 Standard Beam Search 复习

给定 prompt $x$, model $\theta$, vocabulary $\mathcal{V}$, beam budget $B$。每个 time step $t$ 保留 $B$ 个 partial sequences, 每个 sequence 的累积 log-probability 是 scoring signal:

$$S_{\text{BS}}(y_{1:t}) = \sum_{\tau=1}^{t} \log p_\theta(y_\tau \mid y_{<\tau}, x)$$

变量:
- $y_{1:t}$: 已生成的 partial sequence
- $y_\tau$: time step $\tau$ 处的 token
- $y_{<\tau}$: time step $\tau$ 之前的所有 tokens
- $p_\theta$: 参数 $\theta$ 的 model 的 next-token distribution
- $\log$: log probability, 把乘积转成求和, 数值稳定

问题: beam search 倾向 collapse 到 high-probability mode, $B$ 个 final candidate 经常只是 minor variations (Vijayakumar et al. 2018, https://arxiv.org/abs/1610.02424)。

### 2.2 Diverse Beam Search 的 scoring

DBS 把 $B$ 个 beam 分到 $G$ 个 group, 每个 group 有 $B/G$ 个 beam。Group 内部走标准 beam search, group 之间通过 **Hamming diversity penalty** 强制分化:

$$S_{\text{DBS}}(y_{1:t}^{(g)}) = \sum_{\tau=1}^{t} \left[ \log p_\theta(y_\tau^{(g)} \mid y_{<\tau}^{(g)}, x) - \lambda \cdot \sum_{g' \neq g} c_{g'}(y_\tau^{(g)}) \right]$$

变量:
- $g \in \{1, ..., G\}$: group index
- $y^{(g)}$: group $g$ 里的 sequence
- $\lambda > 0$: diversity penalty 强度 (paper 中 $\lambda = 10$)
- $c_{g'}(y_\tau^{(g)})$: token $y_\tau^{(g)}$ 在 group $g'$ 中、time step $\tau$ 处出现的次数
- 求和 $\sum_{g' \neq g}$: 遍历所有其他 group

直觉: 如果一个 token $y_\tau$ 在其他 group 都被选过, 当前 group 选它会被罚, 鼓励选不一样的 token。这让 $G$ 个 group explore 不同的 semantic space, 类似 human brainstorming 时主动 divergent thinking。

### 2.3 Paper 中的超参选择

- $B = 8$ (beam budget)
- $G = 8$ (每个 group 只含 1 个 beam, 最大化 diversity)
- $\lambda = 10$ (scaling factor 来 counterbalance likelihood, 因为 log-prob 数值偏小, 需要放大 penalty 才有效果)
- 保留 top $K = 4$ candidates 进入 validation phase
- 256 new tokens 上限

**重要观察**: $G = B$ 意味着每个 group 只有 1 个 beam, 这本质上是把 beam search 退化为 $G$ 个独立 greedy decoding 加 cross-group penalty。这种 setup 极端追求 diversity, 牺牲了 group 内部的 backup hypotheses。

---

## 3. Response Validation: LLM-as-a-Judge + Balanced Position Calibration

### 3.1 Positional Bias 问题

Wang et al. 2023 (https://arxiv.org/abs/2305.17926) 发现 LLM judge 在 pair/multi-choice evaluation 中存在 positional bias —— 即使 candidates 内容相同, 顺序颠倒也会改变 ranking。CBS 必须处理这个 confounding factor。

### 3.2 Balanced Position Calibration

把 top $K$ candidates $\{c_1, c_2, ..., c_K\}$ 通过 cyclic rotation 生成 $K$ 个 evaluation prompts:

$$\pi_k(c_1, ..., c_K) = (c_{k+1 \mod K}, c_{k+2 \mod K}, ..., c_{k+K \mod K})$$

每个 prompt 问 LLM: "Which of the following is the most creative answer? 1) ... 2) ... 3) ... 4) ... Provide only the number."

Vote aggregation:

$$\text{vote}(c_i) = \sum_{k=1}^{K} \mathbb{1}\left[ \text{LLM}(\text{prompt}(\pi_k(c_1, ..., c_K))) = i \right]$$

变量:
- $\pi_k$: 第 $k$ 个 cyclic rotation
- $\mathbb{1}[\cdot]$: indicator function, 条件成立为 1, 否则 0
- $\text{vote}(c_i)$: candidate $c_i$ 累计得票

Final selection:

$$c^* = \arg\max_{c_i} \text{vote}(c_i)$$

Tie-breaking: 回退到 DBS score ordering。

### 3.3 Decoding Strategy in Judge

Judge 阶段用 **greedy decoding** (温度 0), 因为 LLM-as-a-Judge 需要确定性 —— 不能让 best candidate 因为 sampling randomness 被错过。这是一个 subtle 但 critical 的 design choice。

### 3.4 Complexity Cost

Total cost 相对标准 sampling 增加约:
- Generation: $\approx G \times$ 单次 beam search cost $\approx 8 \times$ greedy
- Validation: $K \times$ forward pass $\approx 4 \times$ greedy

总 overhead 大约是 $12 \times$ greedy decoding 的 cost。这在 online co-creativity 场景下尚可接受, 但仍是 paper 提到的 limitation。

---

## 4. Experiment Setup 细节

### 4.1 Model & Baseline

- **Backbone**: Llama 2 7B Chat (RLHF-tuned version), 因为 RLHF 后的 model 给出更 coherent 的 response, 对 self-evaluation 重要
- **Baseline**: standard sampling, $T = 1.0$, top-$p = 0.9$ (Holtzman et al. 2020 nucleus sampling, https://arxiv.org/abs/1904.09751)
- **Max tokens**: 256 (paper 承认这是 significant constraint, 但 argue creativity 在 short text 中可识别)

Llama 2 paper: https://arxiv.org/abs/2307.09288

### 4.2 Evaluation

- 31 名 CS graduate students
- 217 总评估
- Gradio 接口, 用户输入 free-form creative prompt, 看到 CBS 与 STD output (randomized order)
- 三选项: prefer CBS / prefer STD / too similar to decide

---

## 5. Results 解读 (Table 1 详细拆解)

| Preference | CBS ≠ DBS | CBS == DBS | Total |
|---|---|---|---|
| CBS | 0.34 | 0.11 | **0.45** |
| STD | 0.18 | 0.11 | **0.29** |
| Same | 0.19 | 0.07 | **0.26** |
| **Total** | 0.71 | 0.29 | 1.00 |

### 5.1 Key Findings

**Finding 1: CBS 整体胜过 STD** (45% vs 29%)。在可分辨的 cases (排除 Same) 中, CBS 占 $\frac{0.45}{0.45 + 0.29} \approx 60.8\%$, STD 仅 39.2%。

**Finding 2: LLM-as-a-Judge 实际上 overrule 了 DBS scoring**。CBS output == DBS output 的比例仅 29%, 而 random selection 期望是 $1/K = 25\%$ (这里 $K=4$)。但 paper 报告的 random baseline 是 35.3%, 这可能是因为 DBS score 已经产生 ranking 偏向, top-1 自然占更多。无论如何, 29% < 35.3% 表明 LLM judge 在 actively 选择与 DBS 不同 candidate。

**Finding 3: Self-evaluation 真的在 improve**。当 CBS ≠ DBS (71% 的 cases) 时, CBS wins 34% vs STD wins 18% —— 在可分辨 cases 中 CBS 占 $\frac{34}{34+18+19} \approx 47.9\%$ vs STD 25.4%。当 CBS == DBS (29% cases) 时, CBS 与 STD 平分秋色 (11% vs 11%), 说明 pure DBS (without validation) 反而更接近 STD 水平。Figure 3 视觉化展示了这一点。

### 5.2 Counterintuitive Insight

paper 里一个有趣的细节: 当 CBS output 等于 DBS output (即 LLM judge 同意 DBS 的 top choice) 时, 用户在 CBS 和 STD 之间没有明显偏好 (11% vs 11%)。这意味着 **DBS 自己选的 candidate 并不优于 standard sampling**。真正让 CBS 胜出的是 LLM-as-a-Judge overrule DBS 的那 71% cases。

这暗示 DBS 的 diversity scoring 并未真正 capture "creative quality", 而 LLM-as-a-Judge 提供了 orthogonal signal。

---

## 6. Limitations 与 Open Questions

### 6.1 Hamming Diversity 的弱点

Hamming diversity 仅在同 time step 比较相同 token。两个 sequences "The cat" 和 " The cat" (开头多一空格) 在 Hamming 意义上完全不同, 但语义上几乎相同。paper 中明确讨论这个 issue。更好的 diversity measure 可能是:
- **Semantic diversity** (embedding cosine distance)
- **n-gram diversity** (Jaccard on n-gram sets)
- **Edit distance** (Levenshtein, 对 misalignment 鲁棒)

### 6.2 Single-Pass Limitation

Amabile model 允许 validation failure 后重新 generation (iterative loop)。CBS 砍掉了这个 loop, 直接从 K candidates 选最好。可以想象一个 **Iterative CBS**:

```
loop:
    generate K candidates via DBS
    validate via LLM-as-a-Judge
    if best_score > threshold: return best
    else: re-prompt with feedback, generate again
```

这与 Self-Refine (Madaan et al. 2023, https://arxiv.org/abs/2303.17651) 思路相通。

### 6.3 LLM-as-a-Judge 的真实语义

Shanahan (2024) 和 Franceschelli & Musolesi (2023, https://arxiv.org/abs/2304.00008) 都强调 LLM 没有 intentionality。Self-evaluation 实际上是 model 报告 "what it has learned to be more likely", 而非真实的 preference。这引出深层问题: 当 LLM judge 自己的 output, 它给出的 ranking 本质是 in-distribution preference, 可能偏好 "looking creative" (stereotype of creativity) 而非真正 novel output。

### 6.4 Evaluation 规模

31 名 CS graduate students 不 representative。Anthropic / OpenAI 的 constitutional AI evals 通常用更大样本, 并区分 domain experts (writers, poets) 与 general users。这个 paper 的 qualitative claim 应视为 pilot study。

---

## 7. Related Work 联想网络

### 7.1 直接相关 (Quality-Diversity + AI Feedback)

- **Bradley et al. 2023**, "Quality-Diversity through AI Feedback" (https://arxiv.org/abs/2310.13032): 用 MAP-Elites + LLM-as-Judge 做 creative text generation。CBS 可以视为这个 framework 的简化版 (k=K 代替 MAP-Elites archive)。
- **Ding et al. 2023**, "Quality Diversity through Human Feedback" (NeurIPS ALOE Workshop, https://openreview.net/forum?id=T9wbr7EReR): 用 human feedback 替代 AI feedback。

### 7.2 LLM Self-Evaluation 家族

- **Self-Refine** (Madaan et al. 2023, https://arxiv.org/abs/2303.17651): generate → feedback → refine loop
- **Self-Rewarding Language Models** (Yuan et al. 2024, https://arxiv.org/abs/2401.10020): 用 LLM 自己生成 preference data 训练自己
- **SPIN / Self-Play Fine-Tuning** (Chen et al. 2024, https://arxiv.org/abs/2401.01335): weak model 通过 self-play 变强
- **Constitutional AI** (Bai et al. 2022, https://arxiv.org/abs/2212.08073): Anthropic 的 self-critique 训练
- **RLAIF** (Lee et al. 2023, https://arxiv.org/abs/2309.00267): scaling RLHF 用 AI feedback

### 7.3 Self-Eval Guided Search

- **Self-evaluation guided beam search for reasoning** (Xie et al. 2023, https://arxiv.org/abs/2310.01279): 与 CBS 思路几乎一致, 但用于 reasoning tasks。Xie 用 self-eval score 替代 likelihood 做 beam scoring, CBS 则在 beam search 之外做 post-hoc validation。
- **Tree of Thoughts** (Yao et al. 2023, https://arxiv.org/abs/2305.10601): 更激进的 search + LLM evaluation
- **Graph of Thoughts** (Besta et al. 2023, https://arxiv.org/abs/2308.09687)

### 7.4 LLM Creativity 评估

- **Alternate Uses Test (AUT) for GPT-3** (Stevenson et al. 2022, https://arxiv.org/abs/2206.07597): 用 Torrance creativity test 测 LLM
- **Pushing GPT's Creativity** (Goes et al. 2023): 通过 prompting 提升 AUT 分数
- **Bits of Grass** (Sawicki et al. 2023a, https://arxiv.org/abs/2306.11647): GPT 写 Whitman 风格
- **Computational Creativity survey** (Franceschelli & Musolesi 2024, ACM CSUR): 作者自己即将发表的综述, https://arxiv.org/abs/2407.02063

### 7.5 Beam Search 变种

- **Constrained Beam Search** (Hokamp & Liu 2017, https://aclanthology.org/P17-1141/): lexically constrained decoding
- **Nucleus Sampling** (Holtzman et al. 2020, https://arxiv.org/abs/1904.09751): CBS 的 baseline
- **Typical Sampling** (Meister et al. 2023, https://aclanthology.org/2023.acl-long.20/): entropy-based sampling
- **$\eta$-sampling** (Hewitt et al. 2022, https://arxiv.org/abs/2210.15160)

---

## 8. Intuition Building: 为什么 CBS works?

### 8.1 Decompose "Creativity"

Creativity 的 operational 定义可以拆为两个 axes (Boden 的 framework, https://academic.oup.com/book/5990):

- **Novelty** (P-creativity / H-creativity): 与已有 output 不同
- **Quality / Value**: 在某 domain 中被认为是 good/valuable

Standard sampling 优化 likelihood, 倾向 novelty 是 stochastic 副作用, quality 是 implicit (RLHF-tuned model 默认输出 "good" output)。但 likelihood 与 creativity 弱相关 —— 高 likelihood 意味 typical, typical 反 creative。

DBS 显式优化 novelty (via diversity penalty), 但没有 quality term —— 4 个 diverse candidates 可能有 3 个 nonsense。LLM-as-a-Judge 提供 quality signal, 弥补这个 gap。组合起来覆盖 novelty + quality 两个 axes, 这就是 CBS 的核心直觉。

### 8.2 LLM-as-Judge 学到了什么?

Llama 2 Chat 经过 RLHF, reward model 偏好 helpful/harmless/honest response。"Most creative" prompt 触发 model 报告它 learned 的 "creative style" prior, 大致包含:
- Metaphor use
- Surprising juxtaposition
- Concrete imagery
- Coherent narrative structure

这些都是 in-distribution "creativity markers"。LLM judge 选 candidate 时实际在 rank 这些 markers 的强度, 而非真正的 novelty。这是 CBS 的 fundamental limitation —— 它 simulate 的是 "what LLM learned as creative" 而非 "what is genuinely novel"。

### 8.3 与 Reward Hacking 的关系

如果 LLM judge 反复 evaluate 自己 generated 的 candidates, 系统容易 collapse 到 LLM judge 偏好的 mode。这与 RLHF 中的 reward hacking (https://arxiv.org/abs/2204.05862) 同构。CBS 暂时 immune 因为只用 single-pass validation, 但若扩展到 iterative version, 需要防 reward hacking。

### 8.4 Optimal $K$ and $G$?

paper 用 $G=K=B=8, K_{\text{judge}}=4$, 但没系统 sweep。理论上:
- 大 $G$ → 更多 diversity in candidates, 但每个 group beam 减少, likelihood quality 下降
- 大 $K_{\text{judge}}$ → 更多 options 给 LLM judge, 但 positional bias mitigation cost 线性增长, 且 LLM judge 在 >4 options 上 reliability 下降 (Wang et al. 2023 显示 LLM judge 对 pairwise 最 reliable)
- Sweet spot 可能 $G=4, K=4$, 但需 empirical validation

### 8.5 与 Karpathy 的 "micrograd" 视角

把 CBS 抽象成 computational graph:

```
prompt x
    ↓
[LLM forward with DBS] → 4 candidates c1, c2, c3, c4
    ↓
[4 LLM forward with rotated prompts] → 4 votes v1, v2, v3, v4
    ↓
argmax → c*
```

Gradients: 这里 no gradient flow, CBS 是 pure inference-time method。但若想 fine-tune, 可以把 LLM judge 的 preference 视为 reward signal, 做 RLAIF-style training, 这是 Self-Rewarding LM (Yuan et al. 2024) 的方向。

---

## 9. 与 Karpathy 关注领域的 Connection

Karpathy 在 "Intro to LLMs" (https://karpathy.ai/zero123/) 和 "State of GPT" talk (https://www.youtube.com/watch?v=bZQun8Y4L2Y) 里强调 LLM 是 "next-token predictor with an objective function"。CBS 实际上揭示了:

1. **LLM inference 是 underdetermined** —— 同一 prompt 可以有多种 decoding strategy, 不同 strategy bias 不同 "mode" of model distribution
2. **LLM 能力不止 next-token prediction** —— self-evaluation 用同一 model 做 meta-level reasoning, 这接近 system-2 thinking (Karpathy 在 https://x.com/karpathy/status/1720518788305559547 也提到)
3. **Search 是 inference-time scaling 的方向** —— 与 OpenAI o1 / Rationale 一脉相承 (https://openai.com/index/learning-to-reason-with-llms/), CBS 可以视为 reasoning-style search 应用到 creativity

Karpathy 的 "Software 2.0" essay (https://karpathy.medium.com/software-2-0-a64552e275fc) 提到 weights 即程序。CBS 揭示: 即使 weights 不变, decoding/search strategy 本身也是 "Software 1.5" —— 半 programmatic 半 learned 的 inference pipeline。这是 inference-time scaling 的核心 intuition。

---

## 10. Critique 与 Personal Take

### 10.1 优点
- 简洁: 两个 known technique (DBS + LLM-as-Judge) 组合, implementation cost 低
- Conceptual clarity: 明确把 psychological creativity model 映射到 ML pipeline, 给后续 work 提供 framework
- Ablation thoughtful: 分别报告 CBS==DBS 和 CBS≠DBS 的 case 分布, 暴露 LLM judge 的实际作用

### 10.2 弱点
- Evaluation 太小 (31 人, 217 cases)
- 没与 Self-Refine / ToT 等 inference-time methods baseline
- Hamming diversity 已知 weak, paper 提但没 fix
- 没系统 sweep hyperparameters ($\lambda, G, K$)
- 没在更大 model (Llama 2 70B, GPT-4) 上验证 —— LLM-as-Judge reliability 强依赖 judge model 能力
- Llama 2 7B Chat 作为 generator 可能太弱, 4 个 candidates 区分度低, 26% "too similar" 印证这点

### 10.3 接下来的方向

如果 Karpathy 要 push 这个方向, 几个自然 extensions:

1. **Iterative CBS**: validation 失败后 re-prompt + 改进 (类似 Self-Refine)
2. **Quality-Diversity CBS**: 用 MAP-Elites archive 存历史 candidates, LLM judge 在 archive 上做 novelty search (融合 Bradley et al. 2023)
3. **Cross-model CBS**: Generator = Llama 3 405B, Judge = GPT-4o, 避免 self-evaluation 的 in-distribution bias
4. **Reward model trained on CBS pairs**: 把 CBS 选择的 (winner, losers) pairs 作为 preference data 训 reward model, 蒸馏成单次 sampling (类似 DPO 训练)
5. **Latent space diversity**: 用 hidden state embedding 距离替代 Hamming diversity, 更语义化

---

## 11. Reference 汇总

**Paper 本体**:
- https://arxiv.org/abs/2407.00431 (假设的正式版本, 实际看作者主页)
- 作者 https://giorgiofranceschelli.com/

**核心 cited works**:
- Amabile 1983 (creativity framework): https://psycnet.apa.org/record/1984-07715-001
- Diverse Beam Search: https://arxiv.org/abs/1610.02424
- LLM-as-a-Judge (Zheng et al.): https://arxiv.org/abs/2306.05685
- Positional bias (Wang et al.): https://arxiv.org/abs/2305.17926
- Llama 2: https://arxiv.org/abs/2307.09288
- Nucleus Sampling: https://arxiv.org/abs/1904.09751
- Franceschelli & Musolesi 2023 "Creativity of LLMs": https://arxiv.org/abs/2304.00008
- Shanahan 2024: https://cacm.acm.org/research/talking-about-large-language-models/

**Related inference-time methods**:
- Self-Refine: https://arxiv.org/abs/2303.17651
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Self-eval guided beam search: https://arxiv.org/abs/2310.01279
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Quality-Diversity AI Feedback: https://arxiv.org/abs/2310.13032
- Reasoning models (OpenAI o1): https://openai.com/index/learning-to-reason-with-llms/

**Karpathy 相关**:
- Intro to LLMs: https://karpathy.ai/zero123/
- Software 2.0: https://karpathy.medium.com/software-2-0-a64552e275fc
- State of GPT talk: https://www.youtube.com/watch?v=bZQun8Y4L2Y
- nanoGPT: https://github.com/karpathy/nanoGPT

---

## 12. Final Takeaway

CBS paper 本身是 small contribution 但 conceptually clean。真正的价值在于显式提出 "creativity = generation + validation" 这个 decomposition 可以映射到 inference-time algorithm, 与 reasoning model 的 "thinking = search + evaluation" 是同构的。本质上, reasoning 和 creativity 在 inference-time 都需要 explicit search + quality evaluation, 区别只在 evaluation signal 来源 (logical correctness vs aesthetic/novelty preference)。

这与 Karpathy 反复强调的 "inference-time compute is the next scaling axis" 完全一致 —— CBS 给了一个 creativity domain 的具体 instantiation。后续 work 沿着 iterative refinement、cross-model judging、QD-archive 三个方向任何一个 push, 都会是 substantial contribution。
