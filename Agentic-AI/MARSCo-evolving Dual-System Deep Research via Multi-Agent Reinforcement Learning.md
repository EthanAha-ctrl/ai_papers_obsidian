---
source_pdf: MARSCo-evolving Dual-System Deep Research via Multi-Agent Reinforcement
  Learning.pdf
paper_sha256: 1ef96eabc79e21f2a043496542f5babe4ba89df99e0582eff47d0272bf6814b1
processed_at: '2026-08-05T16:27:03-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MARS 用人话讲：让 LLM 学会"该快则快，该慢则慢"

## 一句话总结

现有的 reasoning model 像个"只会死磕的书呆子"——遇到"2+3=?"也要 deep think 三分钟，遇到"帮我读 20 篇 paper"直接 context window 爆炸。MARS 做的事情就是让一个 model 学会**两种人格来回切换**，而且这两种人格通过 RL 一起进化，互相磨合出默契。

---

## 问题到底出在哪

我先讲个 scenario 你就懂了。

假设你让 DeepSeek-R1 去查一个问题，它调了 Google Search，返回 10 个 web pages，每个 page 平均 5000 tokens。现在 model 要怎么办？

**WebThinker 的做法**：用一个写死的 prompt 去总结每个 page。问题是这个 prompt 不知道你 reasoning 到哪一步了，不知道你需要什么细节，只能给你一个 generic summary。关键数字丢了，你就答错了。

**Search-R1 的做法**：直接把 10 个 raw page 塞进 context。50,000 tokens 直接进去，reasoning model 的 context 被 noise 淹没，dilution 严重。

**人类的做法**：你读到 reasoning 的某一步，心想"我现在需要知道 2023 年 Transformer 架构的 parameter count"，然后去查，查到的 paper 你只看 parameter table 那一页，其他跳过。

MARS 就是想模拟这个过程。关键 insight：**你怎么 extract 信息，取决于你在 reason 什么**。这个 dependency 不能 hardcode，必须让两个 system 一起 train，一起 evolve。

参考: https://arxiv.org/abs/2504.21776 (WebThinker), https://arxiv.org/abs/2503.09516 (Search-R1)

---

## MARS 的设计：一个 model，两个 mode

### 角色 division

| | System 1 | System 2 |
|---|---|---|
| 像什么 | 你的"扫视眼"——快速 scan 一页 paper 抓重点 | 你的"思考脑"——坐下来慢慢 reason |
| 干什么 | 吃 raw tool output，吐 distilled snippet | 拿 distilled info 做 multi-step reasoning |
| 记忆 | 无状态——每 turn 只看当前 tool 输出 | 有记忆——整个 reasoning history 都在 context 里 |
| Context budget | 长 prompt (23K) + 短 response (8K) | 短 prompt (3K) + 长 response (28K) |

### 关键桥梁：purpose $p_i$

System 2 调 tool 时不能只说"给我搜 X"，必须说"我搜 X 是为了搞清楚 Y 这一步 reasoning"。这个 $p_i$ 就是给 System 1 的 brief，让它知道该 extract 什么。

这一步是整个 paper 的灵魂。你看公式 (1):

$$s_i, (t_i, p_i) = \pi_{sys_2}(c_i)$$

意思就是 System 2 一口气吐出三样东西:
- $s_i$ = 这一步的 reasoning 内容
- $t_i$ = 要调什么 tool，参数是什么
- $p_i$ = 我调这个 tool 是想搞清楚什么（给 System 1 的 brief）

然后 System 1 干的事情（公式 2）:

$$\tilde{o}_{t_i} = \pi_{sys_1}\left(\text{Bin-Packing}(o_{t_i}^{(1)}, ..., o_{t_i}^{(n_{t_i})}), p_i\right)$$

变量逐个讲:
- $o_{t_i}^{(j)}$ = tool $t_i$ 返回的第 j 个 raw 结果（比如第 j 个搜索结果）
- $n_{t_i}$ = 这次 tool call 返回了多少个结果
- Bin-Packing(...) = 把这些变长结果打包成几个 chunk
- $p_i$ = System 2 给的 purpose brief
- $\tilde{o}_{t_i}$ = System 1 蒸馏出来的精华，喂回给 System 2

然后 context 更新（公式 3）:

$$c_{i+1} = c_i \oplus \{s_i, t_i, p_i, \tilde{o}_{t_i}\}$$

就是把这一 turn 的所有东西（reasoning step, tool call, purpose, 蒸馏结果）拼到 System 2 的 context 后面，下一 turn 继续。

---

## 训练：最 tricky 的部分

### 难题：shared reward 怎么做 credit assignment

整个 trajectory 用 Qwen2.5-72B 当 judge 给一个 binary reward（公式 8）:

$$r(c_N, \text{groundtruth}) = \begin{cases} 1 & \text{if correct} \\ 0 & \text{otherwise} \end{cases}$$

但是 trajectory 里既有 System 2 的 reasoning tokens，又有 System 1 的 distillation tokens。如果直接一起算 loss，两个 system 的 gradient 会纠缠在一起，谁也学不好。

### 解法：decoupled gradients（解耦梯度）

**核心 trick**：同一个 trajectory，造两份 training sample，loss 在不同的 token 上计算。

**System 2 的 sample** 长这样:

```
[question][s_1][t_1][p_1][ũo_t1][s_2][t_2][p_2][ũo_t2]...[answer]
              ↑loss↑      ↑MASK↑              ↑MASK↑
```

也就是说 System 1 产生的 $\tilde{o}_{t_i}$ 部分**在 System 2 的 loss 里被 mask 掉**，System 2 只在 $\{s_i, t_i, p_i, \text{answer}\}$ 这些 token 上算 loss。

**System 1 的 sample** 完全独立，每个 sample 就是一个 (输入 bin, 输出 distillation) pair:

```
[bin-packed content][purpose] -> [distilled output]
                        ↑ loss 只在这里 ↑
```

**为什么这能 work？** Shared reward 保证两个 system 都朝"答对题"这个共同目标努力；decoupled gradients 保证每个 system 收到的学习信号是 role-specific 的:
- System 2 学到: "我怎么 reason，怎么 plan tool purpose，能让整体答对"
- System 1 学到: "在同一个 trajectory 的所有 System 1 sample 中，什么样的 distillation 比 others 更有助于答对"

这就形成了 co-evolution 的动态: System 2 学会 generate 更明确、更 actionable 的 purpose $p_i$；System 1 学会根据 purpose extract 更 task-relevant 的信息。两边互相适应。

### Advantage 计算（公式 5）

GRPO-style 的 group normalization:

$$A_{sys_2}^k = \frac{r_{sys_2}^k - \text{mean}(\mathbf{r}_{sys_2})}{\text{std}(\mathbf{r}_{sys_2})}$$

$$A_{sys_1}^{k,j} = \frac{r_{sys_1}^{k,j} - \text{mean}(\mathbf{r}_{sys_1})}{\text{std}(\mathbf{r}_{sys_1})}$$

变量讲清楚:
- $k$ = 第 k 个 rollout trajectory（group size G=16）
- $j$ = 第 k 个 trajectory 里的第 j 个 System 1 sample
- $\mathbf{r}_{sys_2}$ = 16 个 System 2 reward 的集合
- $\mathbf{r}_{sys_1}$ = 所有 System 1 sample 的 reward 集合（每个都等于所在 trajectory 的 reward）
- $A_{sys_2}^k$ = 第 k 个 System 2 sample 的 normalized advantage
- $A_{sys_1}^{k,j}$ = 第 k 个 trajectory 里第 j 个 System 1 sample 的 normalized advantage

注意 System 1 的 advantage 是**在自己 group 内** normalize 的，意思就是 System 1 在比较"我这次 distillation 比 group 里其他 distillation 好多少"，不是"整体 trajectory 成不成功"。这就是 decoupled 的精髓。

### Balanced sampling（平衡采样）

每个 question 做 16 个 rollout，System 2 恰好产生 16 个 sample，但 System 1 产生 $M = \sum_{k=1}^{16} n_k$ 个 sample，$n_k$ 是 trajectory k 里的 System 1 sample 数（取决于 tool call 次数和 bin-packing 结果），完全不可控。

**问题**: 如果 $M \gg 16$，System 1 主导学习；如果 $M \ll 16$，System 2 主导。都会破坏 co-evolution。

**解法**: 先 pre-compute 所有 advantage，然后:
- $M > 16$: 随机下采样到 16
- $M < 16$: 随机复制上采样到 16

**为什么先 pre-compute 再 sample？** 因为如果先 sample 再算 advantage，normalization 的统计特性就被采样破坏了。先算后采样，保证所有 sample 的 advantage 信息都参与了 normalization，statistical integrity 保持完整。

---

## Bin-Packing：被忽视的工程细节

HLE 数据每个问题平均要处理 22 个 web pages + 0.17 篇 papers。如果每个 page 单独让 System 1 处理一次，那要 22 次 generation，效率炸了。

**FFD (First Fit Decreasing) 算法**:

```
1. 对每个 tool output 数 token
2. 超过 System 1 max context 的，truncate 后单独一个 bin
3. 剩下的按 token 长度降序排序
4. 每个输出放进第一个能装下的 bin，装不下就开新 bin
```

举例: 22 个 page，平均 5000 tokens，System 1 max context 23552 tokens。FFD 大概能打包成 5-6 个 bin，每个 bin 3-4 个 page。这样 System 1 只需要 5-6 次 generation，并行处理。

为什么选 FFD 而不是 BFD (Best Fit Decreasing)？paper 说实践中 FFD 效率更好。approximation ratio 是 $\frac{11}{9}OPT + \frac{6}{9}$，对于这个问题完全够用。

参考: https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

---

## 数据 Pipeline：从 5M 到 40K

这一步很多 paper 不会重点讲，但 MARS 的数据 curation 其实很有意思。

**Best-of-16 过滤的 intuition**: 用 Qwen2.5-72B + Google Search 对每个 question 做 16 次 attempt，统计正确次数:
- 0 次正确: 可能 question 本身就 ambiguous，没 definitive answer
- 13-16 次正确: 太 trivial，没训练价值
- **1-12 次正确: moderate difficulty，有 verifiable answer，留用**

这个过滤很聪明——既保证 difficulty 合适，又保证 answer 可验证。RL 训练最怕的就是 reward signal 不可靠，这一步从源头掐掉了 noise。

最终 40K curated + 一些公开 QA dataset 凑成 5050 training examples。对，你没看错，**总共就 5050 examples**，8B model 训出来打平 32B with SFT。这说明 RL framework 的 efficiency 远超 SFT。

---

## 实验结果：让我震惊的几个数字

### HLE (Humanity's Last Exam)

| Model | Size | SFT | HLE Accuracy |
|---|---|---|---|
| GPT-4o | - | - | 2.32% |
| Qwen3-8B (base) | 8B | - | 4.31% |
| WebThinker | 32B | ✓ | 6.87% |
| Claude 3.7 Sonnet | - | - | 7.89% |
| o1 | - | - | 7.75% |
| DeepSeek-R1 | 671B | - | 8.54% |
| **MARS (Qwen3-8B)** | **8B** | **✗** | **8.17%** |

8B Zero RL 打过 32B with SFT，接近 Claude 3.7 和 o1。如果这不让你震惊，你对 scale 的信仰可能太重了。

### System 1 ablation

| Config | HLE |
|---|---|
| Base model | 3.52% |
| MARS full | 7.38% |
| MARS w/o System 1 | 5.47% |

去掉 System 1 掉 1.91 个点。注意这里 System 2 还是有 tool access 的，只是把 raw tool output 直接喂给它，不做 distillation。这 1.91 个点完全是 co-evolution 带来的——证明 System 1 学到了 task-relevant extraction，不是被动 summarizer。

### Tool ablation

| Config | HLE |
|---|---|
| All three | 7.38% |
| w/o Python | 6.21% |
| w/o Google Search | 5.99% |
| w/o Scholar | 6.72% |

Google Search 贡献最大（broad coverage），Python 对 Math/Physics 关键（exact computation），Scholar 对 CS/AI 和 Other 类关键（academic evidence）。三个 tool 不是简单 additive，有 orchestration 的必要。

### Multi-hop QA 的巨大提升

| Task | C-3PO (72B) | MARS (8B) | Gain |
|---|---|---|---|
| Single-hop avg | 59.73 | 69.53 | +9.8% |
| Multi-hop avg | 48.90 | 61.60 | **+12.7%** |

Multi-hop 提升更大的 intuition: errors 在 multi-hop 里 compound。如果你第二步漏了一个 entity，后面全错。System 1 的 purpose-conditioned distillation 保留 entity/relation/intermediate constraint，对 multi-hop 是救命级的。

参考: https://arxiv.org/abs/2501.14249 (HLE)

---

## 训练动态：两个阶段

Figure 3 里的曲线讲了个故事:

1. **早期 (~step 50)**: model 不敢调 tool，每个 question 平均调 1 个 tool，reward 低，HLE 慢慢爬
2. **中后期 (~step 150+)**: model 学会了"遇到难题多调几次 tool"，每个 question 平均 2+ 个 tool call，reward 稳定在 0.4，HLE 突破 10%

**关键观察**: Tool call 数量增加的同时 HLE 也增加，说明 model 不仅学会了"多调 tool"，还学会了"调对 tool"和"用 tool 结果做更好的 reasoning"。如果只是盲目多调 tool，context 被 noise 淹没，HLE 应该下降才对。这里 System 1 的 distillation 起了关键的降噪作用。

---

## 为什么单 checkpoint 共享 System 1/2

这一点 paper 在 Appendix C.6 解释了，我觉得很重要:

1. **认知科学对齐**: Kahneman 的 dual-process theory 里 System 1/2 不是两个 brain，是同一个 brain 的两种 mode。单 model 设计反映了这一点
2. **Synergistic optimization**: 共享参数让 System 2 学会 generate "System 1 能听懂的 purpose"，因为它们是同一个 model
3. **工程效率**: 两个 model 双倍 memory，部署不便
4. **实证验证**: WebThinker 的 two-component 设计（独立 reasoner + 固定 extractor）32B 都打不过 MARS 8B

参考: Kahneman "Thinking, Fast and Slow" https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

## 我作为 Karpathy 的 take

### 这篇 paper 真正的创新点

不是 System 1/System 2 概念本身——这个 idea 大家都想过。真正的贡献是:

1. **把 co-evolution 落地为可训的 RL objective**: decoupled gradients + balanced sampling 解决了 multi-agent RL 在 LLM 上的工程难题
2. **Purpose $p_i$ 作为 bridge**: 这个设计很 elegant，让两个 system 的 communication 有了 explicit 的 interface
3. **Zero RL setting**: 证明 SFT 不是必需的，RL co-evolution 能 discover 互补策略

### 与我关注的 overthinking 问题的关联

之前有篇 paper "Do Not Think That Much for 2+3=?" (https://arxiv.org/abs/2412.21187) 讲 LRMs overthinking 问题。MARS 给了一个自然的解法: System 1 处理简单信息，System 2 只处理复杂 reasoning，token 消耗自然 partition。

### 与 System 1/2 distillation 的关联

Yu et al. 2024 "Distilling System 2 into System 1" (https://arxiv.org/abs/2407.06023) 讲的是把 slow reasoning 压进 fast intuition。MARS 反过来——让 System 1 和 System 2 共存且 co-evolve。两个方向其实可以结合: 先 co-evolve，再 distill System 2 的部分能力进 System 1。

### 对未来 reasoning model 设计的启示

1. **Context budget 应该 adaptive**: System 1 长 prompt 短 response，System 2 短 prompt 长 response——这种 asymmetric 设计比单纯扩展 context window 更 efficient
2. **Tool use 不是 add-on，是 reasoning 的一部分**: $p_i$ 让 tool call 有了 semantic purpose，这是 "tool as reasoning primitive" 的范例
3. **Multi-agent RL 在单 model 上是可行的**: 通过 prompt switching 实现 heterogeneity，避免多 model 的 communication overhead

### 我会想做的 next step

1. **Hierarchical System 1**: 当前 System 1 是 flat 的，所有 bin 平等。可以加一层 meta-System 1 先决定哪些 bin 值得深入 extract
2. **Soft switching**: 当前 hard switch（tool call 才触发 System 1），可以让 model 自己 decide 何时用哪个 mode
3. **Process reward**: 当前只有 outcome reward，加入中间 reward（retrieval relevance, distillation faithfulness）可能加速 co-evolution
4. **Multi-modal System 1**: 扩展到 image/table understanding，处理 paper 里的 figure
5. **Long-horizon scaling**: 当前 max 10 turns，扩展到 100+ turns 看看 co-evolution 是否还 stable

---

## 最后的 takeaway

MARS 这篇 paper 给我的最大启示: **specialization can emerge from collaborative pressure**。你不用预先定义 System 1 应该做什么、System 2 应该做什么，只要给它们 shared reward 和 decoupled gradient，它们自己会 evolve 出 complementary strategies。

这跟生物 co-evolution 很像——花和蜜蜂不是谁先设计好对方的角色，而是长期互动中互相 shape。MARS 在 LLM 内部复现了这个 dynamic，用 8B model 打出了 32B 的效果。

如果你只记一个 thing: **purpose $p_i$ 是灵魂**。它让 tool call 有了 semantic intent，让 System 1 有了 extraction brief，让两个 system 的 communication 有了 explicit interface。这个设计 choice 是整个 framework 的 keystone。

参考汇总:
- MARS paper (推测 arxiv): https://arxiv.org/abs/2507.06774
- GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300
- HLE: https://arxiv.org/abs/2501.14249
- WebThinker: https://arxiv.org/abs/2504.21776
- Search-R1: https://arxiv.org/abs/2503.09516
- C-3PO: https://openreview.net/forum?id=hlpwAmQ4wr
- Search-o1: https://arxiv.org/abs/2501.05366
- Qwen2.5: https://arxiv.org/abs/2412.15115
- Qwen3: https://arxiv.org/abs/2505.09388
- Overthinking paper: https://arxiv.org/abs/2412.21187
- Distilling System 2 to System 1: https://arxiv.org/abs/2407.06023
- Dual-process theory: https://en.wikipedia.org/wiki/Dual_process_theory
- Kahneman book: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- Bin-packing FFD: https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
- VeRL framework: https://github.com/volcengine/verl
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- Qwen-Agent: https://github.com/QwenLM/Qwen-Agent
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- AutoGen (multi-agent): https://arxiv.org/abs/2308.08155
- MetaGPT: https://arxiv.org/abs/2308.00352
- Toolformer: https://arxiv.org/abs/2302.04761
- WebGPT: https://arxiv.org/abs/2112.09332

---

# MARS 深度技术解析：Co-evolving Dual-System Deep Research via Multi-Agent RL

## 1. 论文核心直觉 (Build Intuition)

这篇 paper 解决了一个我（Karpathy）长期关注的根本性矛盾：**Large Reasoning Models (LRMs)** 如 o1, DeepSeek-R1 在 System 2 deliberate reasoning 上很强，但在 **处理简单信息处理任务时严重 over-analyze**，导致 token 爆炸，无法 digest 大量外部 retrieved documents。

传统 RAG 的 architectural mismatch 在于：
- **WebThinker** 使用固定 extraction prompt 处理 retrieved docs
- **Search-R1** 直接把 raw search results 喂给 reasoning model
- 信息处理器无法学到 reasoning model 真正需要什么细节
- 要么 information overload（context dilution），要么 critical detail loss

MARS 的核心洞察：**optimal distillation strategy 依赖于 reasoning task**，两个 system 必须 co-adapt 才能形成 complementary strategies。这受启发于 Kahneman 的 dual-process theory。

参考链接：
- https://en.wikipedia.org/wiki/Dual_process_theory
- https://arxiv.org/abs/2501.14249 (HLE benchmark)
- https://arxiv.org/abs/2402.03300 (DeepSeekMath/GRPO)

---

## 2. Dual-System Collaborative Framework 架构详解

### 2.1 角色分工

| Component | Role | Cognitive Mode | Context Handling |
|-----------|------|----------------|------------------|
| **System 2** ($\pi_{sys_2}$) | Deliberate reasoning & tool planning | Slow, analytical | Maintains full history $c_i$ |
| **System 1** ($\pi_{sys_1}$) | Information distillation from tool outputs | Fast, intuitive | Stateless per turn, processes current outputs only |

关键设计：**"purpose" $p_i$ 作为两个 system 之间的 bridge**——System 2 在调用 tool 时同时指定 purpose，System 1 根据 purpose 进行 task-relevant extraction。

### 2.2 生成过程形式化

公式 (1) - System 2 的 per-turn generation：

$$s_i, (t_i, p_i) = \pi_{sys_2}(c_i)$$

变量解释：
- $s_i$: i-th turn 的 reasoning step
- $t_i$: tool parameters（可为空）
- $p_i$: purpose 描述（可为空）
- $c_i$: 累积上下文，$c_i = c_{i-1} \oplus \{s_{i-1}, t_{i-1}, p_{i-1}, \tilde{o}_{t_{i-1}}\}$

公式 (2) - System 1 的信息蒸馏，使用 bin-packing：

$$\tilde{o}_{t_i} = \pi_{sys_1}\left(\text{Bin-Packing}(o_{t_i}^{(1)}, o_{t_i}^{(2)}, ..., o_{t_i}^{(n_{t_i})}), p_i\right)$$

变量解释：
- $o_{t_i}^{(j)}$: tool $t_i$ 返回的第 j-th raw output
- $n_{t_i}$: outputs 数量
- Bin-Packing(...): FFD 算法打包成 chunks
- $\tilde{o}_{t_i}$: 蒸馏后的信息

公式 (4) - 整体 generative process 的因式分解：

$$\mathcal{P}(\text{answer}|q) = \prod_{i=1}^{N}\left[\underbrace{\pi_{sys_2}(s_i, t_i, p_i | c_i)}_{\text{System 2: Reasoning}} \cdot \underbrace{\pi_{sys_1}(\tilde{o}_{t_i} | \text{Bin-Packing}(...), p_i)}_{\text{System 1: Info Processing}}\right]$$

这个 factorization 揭示了 co-evolution 的本质：两个 policy 的乘积决定最终答案质量，必须 joint optimize。

参考链接：
- https://arxiv.org/abs/2504.21776 (WebThinker)
- https://arxiv.org/abs/2503.09516 (Search-R1)
- https://arxiv.org/abs/2411.19443 (C-3PO)

---

## 3. 三大技术创新深度解析

### 3.1 Decoupled Gradient Computation（解耦梯度计算）

这是 paper 最精妙的设计。挑战在于：**System 1 和 System 2 共享 trajectory-level reward，但学习行为完全不同**。

**关键设计：Non-overlapping token sets**

对于 System 2 的训练 sample，sequence 是完整 reasoning context：
$$c_N = \{s_i, t_i, p_i, \tilde{o}_{t_i}\}_{i=1}^{N}$$

但 loss 计算时，**System 1 的 output tokens $\tilde{o}_{t_i}$ 被 masked**：

```
System 2 training sample:
  [s_1][t_1][p_1][MASK: ũo_t1][s_2][t_2][p_2][MASK: ũo_t2]...[answer]
                                          ↑ loss only on {s_i, t_i, p_i, answer}
```

对于 System 1 的训练 sample，每个 sample 是独立的 (bin-packed input, output) pair：
$$(b, \tilde{o})$$
loss 只在 $\tilde{o}$ 上计算。

**为什么这能 work？** Shared reward 提供共同的优化目标（correctness），decoupled gradients 提供角色特异的学习信号：
- System 2 学到："如何 plan 更好的 purpose $p_i$ 给 System 1"
- System 1 学到："什么样的 distillation 在 group 内比其他 distillation 更有效"

公式 (7) 的 loss：

$$\mathcal{L}_{sys_i} = \mathbb{E}_{(x,y) \sim \mathcal{D}_i}\left[\mathcal{L}_{policy}(x, y, A_{sys_i}) + \lambda \mathcal{L}_{KL}(x, y)\right]$$

公式 (9) 的 policy loss（PPO-style clip）：

$$\mathcal{L}_{policy}(x, y, A_{sys_i}) = \frac{1}{|y|}\sum_{j=1}^{|y|}\min\left[\frac{\pi_{sys_i}(y_j|x, y_{<j})}{\pi_{sys_i}^{old}(y_j|x, y_{<j})}A_{sys_i}, \text{clip}\left(\frac{\pi_{sys_i}(y_j|x, y_{<j})}{\pi_{sys_i}^{old}(y_j|x, y_{<j})}, 1-\epsilon, 1+\epsilon\right)A_{sys_i}\right]$$

变量解释：
- $y_j$: output 的第 j-th token
- $y_{<j}$: prefix tokens
- $\pi_{sys_i}^{old}$: behavior policy（rollout 时的旧 policy）
- $A_{sys_i}$: group-normalized advantage
- $\epsilon$: PPO clip ratio

### 3.2 Bin-Packing Optimization（装箱优化）

**为什么需要 bin-packing？** 每个 question 平均处理 22 个 web pages + 0.17 papers（HLE 数据），naive 处理需要 22+ 次 System 1 generation，效率极低。

**FFD (First Fit Decreasing) 算法**：

```
1. Token counting: 对每个 o_t_i^(j) 计算 token 数
2. Large output handling: 若 > max context，truncate 并放入独立 bin
3. Sorting: 剩余 outputs 按 token 长度降序排序
4. Bin assignment: 每个 output 放入第一个能容纳的 bin，否则创建新 bin
```

FFD 的 approximation ratio 是 $\frac{11}{9}OPT + \frac{6}{9}$，在实践中优于 BFD。

参考链接：
- https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
- https://link.springer.com/chapter/10.1007/978-1-4613-3672-7_3 (Coffman et al. 1984)

### 3.3 Advantage Pre-Computation & Balanced Sampling

**问题**：每个 question 做 G 个 rollout，产生恰好 G 个 System 2 samples，但 System 1 samples 数量 $M = \sum_{k=1}^{G} n_k$ 是可变的（取决于每条 trajectory 的 tool call 次数和 bin-packing 产生的 chunk 数）。这种 imbalance 会让一个 system 主导学习。

**解决方案**：先 pre-compute 所有 advantages，然后 balanced sampling。

公式 (5) - GRPO-style advantage normalization：

$$A_{sys_2}^k = \frac{r_{sys_2}^k - \text{mean}(\mathbf{r}_{sys_2})}{\text{std}(\mathbf{r}_{sys_2})}, \quad A_{sys_1}^{k,j} = \frac{r_{sys_1}^{k,j} - \text{mean}(\mathbf{r}_{sys_1})}{\text{std}(\mathbf{r}_{sys_1})}$$

变量解释：
- $\mathbf{r}_{sys_2} = \{r_{sys_2}^1, ..., r_{sys_2}^G\}$: 所有 G 个 System 2 rewards
- $\mathbf{r}_{sys_1} = \{r_{sys_1}^{k,j} | k \in [1,G], j \in [1,n_k]\}$: 所有 System 1 rewards
- $r_{sys_1}^{k,j} = r_{sys_2}^k$: trajectory k 中的 System 1 sample 共享该 trajectory 的 reward
- $n_k$: trajectory k 中的 System 1 samples 数量

**Balanced sampling 策略**：
- 若 $M > G$: 随机 downsample 到恰好 G 个
- 若 $M < G$: 随机 duplication upsample 到 G 个

**为什么先 pre-compute 再 sample？** 两个关键好处：
1. 所有 sample 的 advantage 信息都贡献到 normalization，最大化数据利用
2. 保持 advantage 分布的 statistical integrity（避免 sampling 扭曲 normalization）

---

## 4. Reward Design 详解

公式 (8) - 简单但有效的 binary reward：

$$r(c_N, \text{groundtruth}) = \begin{cases} 1, & \text{if Eval}_{LLM} = \text{Correct} \\ 0, & \text{otherwise} \end{cases}$$

使用 Qwen2.5-72B-Instruct 作为 judge，遵循 HLE 官方 evaluation prompt。所有 System 1 和 System 2 samples 共享这个 trajectory-level reward。

**设计哲学**：避免复杂 reward engineering，让两个 system 专注 ultimate goal——correctness。System 2 学会 generate 更好的 reasoning steps 和 tool plans，System 1 学会更准确、更简洁地 distill 信息。

参考链接：
- https://arxiv.org/abs/2501.14249 (HLE official evaluation)
- https://arxiv.org/abs/2412.15115 (Qwen2.5)

---

## 5. 训练数据 Pipeline

### 5.1 数据 curation 流程

从 5M 候选 examples → 40K 高质量 training prompts：

| Stage | Filter | Pool Size |
|-------|--------|-----------|
| 1 | Academic level (undergrad/grad) | 5M → 237K |
| 2 | Deduplication | 237K → 155K |
| 3 | Clarity assessment (LLM judge) | 155K → 99K |
| 4 | Graduate-level filtering | 99K → 81K |
| 5 | Best-of-16 challenge verification | 81K → 40K |

**Best-of-16 的精妙设计**：保留正确 1-12 次的 questions
- 正确 0 次：可能 ambiguous 或无 definitive solution
- 正确 >12 次：trivial，无训练价值
- 1-12 次：moderate difficulty，有 verifiable answer

### 5.2 训练数据组成

| Data Type | Samples | Percentage |
|-----------|---------|------------|
| Curated complex reasoning | 2000 | 39.6% |
| Single-Hop QA (TriviaQA, PopQA) | 800 | 15.8% |
| Multi-Hop QA (HotpotQA, 2Wiki, Musique) | 1500 | 29.7% |
| Biology & Medicine (PubMedQA, CUPCase) | 750 | 14.9% |
| **Total** | **5050** | **100%** |

---

## 6. 实验结果深度分析

### 6.1 HLE 主结果

| Method | Model Size | SFT | HLE Accuracy |
|--------|-----------|-----|--------------|
| Claude 3.7 Sonnet | - | - | 7.89% |
| o1 | - | - | 7.75% |
| DeepSeek-R1 | 671B | - | 8.54% |
| WebThinker | 32B | ✓ | 6.87% |
| C-3PO | 72B | ✓ | 5.79% |
| **MARS** | **8B** | **✗** | **8.17%** |

**令人震惊的结果**：8B Zero RL 超过 32B with SFT，接近 Claude 3.7 Sonnet 和 o1。

**Per-category 分析**（MARS Qwen3-8B）：
- CS/AI: 8.84%（最强单项）
- Other: 8.57%
- Bio/Med: 13.12%
- Physics: 7.92%
- Math: 7.17%

### 6.2 Knowledge-intensive Tasks 结果

MARS (Qwen3-8B) 在 7 个 benchmark 上平均 65.00%，比 C-3PO 提升 8.95%。

| Task Type | MARS (8B) | C-3PO (72B) | Gain |
|-----------|-----------|-------------|------|
| Single-Hop avg | 69.53 | 59.73 | +9.8% |
| Multi-Hop avg | 61.60 | 48.90 | +12.7% |
| **Overall avg** | **65.00** | **53.42** | **+8.95%** |

**Multi-hop 提升更大的 intuition**：errors 在 multi-hop 中 compound——missing 一个 entity 早期会 derail 整条 chain。System 1 学到的 purpose-conditioned distillation 保留 entities, relations, intermediate constraints，对 multi-hop 至关重要。

### 6.3 Ablation Studies

**Tool Ablation (HLE)**：

| Config | Accuracy |
|--------|----------|
| All three tools | 7.38% |
| w/o Python | 6.21% (-1.17) |
| w/o Google Search | 5.99% (-1.39) |
| w/o Scholar | 6.72% (-0.66) |
| Only Python | 4.64% |
| Only Search | 6.12% |
| Only Scholar | 5.61% |

**Key insights**：
- Google Search 是最强 single tool（broad, high-recall）
- Python 专精 Math/Physics（exact arithmetic, symbolic manipulation）
- Scholar 补充 Search（technical terminology, academic evidence）

**System 1 Co-evolution Ablation**：

| Method | HLE Accuracy |
|--------|--------------|
| Base model | 3.52% |
| MARS (Full) | 7.38% |
| w/o System 1 (raw outputs) | 5.47% (-1.91) |

**这 1.91% 的 drop 证明 System 1 是 active co-evolver**，passive summarizer 无法达到。Maintains all tools 但 removes learned collaboration——说明 gains 来自 co-evolution，tool availability 单独不够。

### 6.4 训练动力学分析

Figure 3 显示三个关键趋势：
- **HLE score**: ~2% → ~10%（consistent improvement）
- **Training reward**: 稳定在 ~0.4
- **Tools per question**: ~1 → ~2+（从 single-shot retrieval 转向 iterative evidence gathering）

**两个 qualitative phases**：
1. **Early training**: conservative tool use, under-explores evidence, low reward
2. **Late training**: frequent tool calls on hard questions, longer trajectories, higher reward

---

## 7. 实现细节关键点

### 7.1 Asymmetric Context Configuration

| System | Max Prompt | Max Response |
|--------|-----------|--------------|
| System 1 | 23,552 tokens | 8,192 tokens |
| System 2 | 3,072 tokens | 28,672 tokens |

**设计 intuition**：System 1 需要处理 large volumes → 长 prompt，短 response（distillation）。System 2 需要 multi-step reasoning → 短 prompt，长 response。

### 7.2 Single Checkpoint Design

**为什么 System 1 和 System 2 共享同一个 LLM（通过不同 prompts 激活）？**

1. **Alignment with dual-process theory**: 在认知科学中，System 1/2 是同一个 mind 的不同 modes，不是 separate brains
2. **Synergistic optimization**: 共享参数使 System 2 学会 generate System 1 能 interpret 的 purpose，反之亦然
3. **Practical efficiency**: 避免双倍 memory 和 inference complexity
4. **Empirical validation**: WebThinker 的 two-component design（separate reasoner + fixed extractor）即使 32B 也 underperform 8B 的 unified approach

### 7.3 Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 1e-6 |
| Batch size | 32 |
| Group size G | 16 |
| Temperature | 1.0 |
| Top-p | 0.95 |
| KL coefficient λ | 0 |
| Entropy coefficient | 0 |
| Max interaction turns | 10 |

---

## 8. 与相关工作的对比

### 8.1 vs WebThinker

| Aspect | WebThinker | MARS |
|--------|-----------|------|
| Model size | 32B | 8B |
| Training | SFT + RL | Zero RL (no SFT) |
| Info processor | Fixed extraction prompt | Learned co-evolving System 1 |
| Scholar tool | ✗ | ✓ |
| HLE | 6.87% | 8.17% |

### 8.2 vs Search-R1

| Aspect | Search-R1 | MARS |
|--------|-----------|------|
| Info processing | Raw search results → reasoner | Bin-packing → System 1 distillation → System 2 |
| Context dilution | High risk | Mitigated by purpose-conditioned distillation |
| Co-evolution | ✗ | ✓ |

### 8.3 vs C-3PO

| Aspect | C-3PO | MARS |
|--------|-------|------|
| Model size | 72B | 8B |
| Training | SFT with curated trajectories | Zero RL |
| Co-evolution | ✗ (independent processor) | ✓ |

---

## 9. Limitations & Open Questions

### 9.1 潜在问题

1. **Binary reward 的局限性**：只有 correct/incorrect，没有 partial credit，可能对 multi-step reasoning 的中间步骤 credit assignment 不够精细。但作者通过 group normalization 缓解了这个问题。

2. **Single LLM 共享参数的风险**：System 1 和 System 2 的 gradient 可能互相干扰。Decoupled gradients 通过 non-overlapping token sets 缓解，但参数层面仍共享。

3. **Bin-packing 的 truncation**：超过 max context 的 output 被 truncate，可能丢失关键信息。对于 very long papers 这是潜在问题。

4. **Eval scale**：WebThinker 在原 paper 用 500-question sample，MARS 用完整 2,154 questions。Score discrepancy 部分源于此。

### 9.2 对 LLM reasoning 的 broader implications

**作为 Karpathy 的视角**：

1. **System 1/System 2 separation 在 LLM 中的实际意义**：不同于人类大脑的 hardwired division，MARS 显示这种 separation 可以 emerge from RL training with shared reward。这暗示 dual-process theory 可能是 efficient computation 的自然产物，biological necessity。

2. **Co-evolution vs pre-training specialization**：传统方法用 SFT 预先定义 specialization，MARS 证明 RL co-evolution 能 discover 更互补的 strategies。这与 AlphaZero 的 self-play 哲学一致。

3. **Tool use as first-class citizen**：MARS 把 tool calls 视为 reasoning 的一部分，System 2 的 purpose $p_i$ 是 tool call 的 integral component，决定 System 1 的 extraction strategy。这比单纯 "call tool, get result" 更接近人类的 research behavior。

4. **Context window economics**：Bin-packing + System 1 distillation 实际上是一种 context compression 策略，让 System 2 能在 fixed context budget 内处理更多信息。这对未来 long-context model 的设计有启示——也许应该 train model 学会 compress 而不是单纯 extend context。

5. **Zero RL 的 significance**：8B Zero RL beating 32B with SFT 强烈暗示——很多 SFT-based 方法可能 overfit to specific trajectory patterns，限制了 generalization。RL 让 model discover 自己的 strategy。

---

## 10. 未来方向推测

1. **Hierarchical System 1**：当前 System 1 是 flat 的，未来可以 hierarchical——高层 System 1 决定哪些 information 值得深入，低层 System 1 做细节 extraction。

2. **Dynamic System 1/2 switching**：当前是 hard switch（tool call 触发 System 1），未来可以 soft switching，model 自己 decide allocation。

3. **Multi-modal extension**：当前限于 text，可以扩展到 image/table understanding，System 1 处理 visual information。

4. **Memory mechanism**：System 1 当前 stateless per turn，加入 episodic memory 可能改善 multi-turn coherence。

5. **Reward shaping**：除了 final answer correctness，可以加入 intermediate rewards（e.g., retrieval relevance, distillation faithfulness）。

---

## 11. 总结

MARS 的核心贡献是证明了 **dual-system co-evolution via multi-agent RL** 可以让 8B model 达到接近 SOTA proprietary model 的 performance。三个技术创新（decoupled gradients, bin-packing, balanced sampling）解决了 multi-agent RL 在 LLM setting 中的具体挑战。

**最深层的 insight**：optimal information processing strategy 依赖于 reasoning task，两个 system 必须 co-adapt。这不仅是工程优化，反映了一个深层 principle——**specialization emerges from collaborative pressure**，类似于 biological co-evolution。

参考链接汇总：
- Paper: https://arxiv.org/abs/2507.06774 (推测)
- GRPO: https://arxiv.org/abs/2402.03300
- HLE: https://arxiv.org/abs/2501.14249
- WebThinker: https://arxiv.org/abs/2504.21776
- Search-R1: https://arxiv.org/abs/2503.09516
- C-3PO: https://openreview.net/forum?id=hlpwAmQ4wr
- Qwen2.5: https://arxiv.org/abs/2412.15115
- Qwen3: https://arxiv.org/abs/2505.09388
- Dual-process theory: https://en.wikipedia.org/wiki/Dual_process_theory
- Bin-packing: https://en.wikipedia.org/wiki/Bin_packing_problem
- VeRL framework: https://github.com/volcengine/verl
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- Qwen-Agent: https://github.com/QwenLM/Qwen-Agent

这篇 paper 对我（Karpathy）来说最有启发的点是——它把 cognitive science 的 dual-process theory 落地为可训练的 RL framework，并通过 co-evolution 让 specialization emerge，rather than prescribe。这种 "let the system discover its own division of labor" 的哲学，可能是通向 AGI-level reasoning 的关键 path。
