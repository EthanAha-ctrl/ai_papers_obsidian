---
source_pdf: Small Language Models are the Future of Agentic AI.pdf
paper_sha256: ee2dbe19d09a857b94aeb3ea8c7e35516178bf6eb290caa5461ff19ada517083
processed_at: '2026-08-12T07:54:07-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Andrej, 我换个说法, 咱们就当在咖啡店里聊天。

---

## 这篇 paper 到底在嚷嚷什么

一句话: **现在所有 agent 都调 GPT-4 这种大模型, 其实大部分调用换成 7B 以下的小模型就够了, 而且更便宜、更快、更可控。**

NVIDIA 的这帮人是在给整个 industry 喊话: "兄弟们, 你们用大模型用过头了。"

---

## 他们为什么这么想

你想想现在一个 typical agent 是怎么跑的。用户说一句 "帮我订下周三飞东京的机票", 然后 agent 就去调 GPT-4o, 让它输出一个 JSON 格式的 tool call, 再去调航班 API。整个过程里, LLM 干的活其实就两件:

1. **理解用户意图** — "下周三飞东京" → 具体日期 + 出发地 + 目的地
2. **吐一个结构化输出** — `{"tool": "search_flights", "date": "2026-08-19", "from": "...", "to": "NRT"}`

你说这玩意儿需要 175B 参数吗? 显然不需要。一个 7B 的模型 fine-tune 一下, 吐 JSON 吐得比 GPT-4 还稳, 因为 GPT-4 还会偶尔 "犯花痴" 用 Markdown 给你包一层, 你的 parser 就炸了。

这就是 paper 里 A4 和 A5 的核心直觉: **agent 把 LLM 关在一个很窄的笼子里用, 那干嘛不直接养一只专门住这个笼子的小动物。**

---

## 三个核心论点 (V1, V2, V3)

论文写成很 formal 的样子, 我翻译成人话:

**V1 — 小模型已经够强了**
两年前 7B 模型还傻乎乎的, 现在 Phi-3 7B 能打 70B, DeepSeek-R1-Distill 7B 能打 Claude 3.5 Sonnet, xLAM-2 8B 在 function calling 上直接干翻 GPT-4o。这个进步速度, 你自己 train 过 NanoGPT 你懂的, 数据质量 + 架构创新比单纯堆参数管用多了。

**V2 — 小模型天然更适合 agent**
agent 要的不是 "什么都能聊两句", 是 "每次都按规矩办事"。小模型 fine-tune 一下, 输出格式死死钉住, 不会跑偏。大模型反而要靠 prompt 去压, 压不住就 hallucinate。

**V3 — 小模型必然更便宜**
7B 在一张 A100 上就能 serve, 175B 要 8 卡 TP + PP, 光运维就是两个世界。paper 说 10-30x 的成本差距, 我觉得这个数字还保守了, 因为没算上 KV cache 内存、batch 调度、cross-node 通信的开销。

---

## 他们的论辩结构挺有意思

这 paper 不是那种 "我证明了一个 theorem" 的 paper, 是 **position paper**, 就是站队表态。但 NVIDIA 的人写得很认真, 列了:

- 7 个支持论点 (A1-A7)
- 3 个反对意见 (AV1-AV3), 还自己反驳自己
- 3 个 adoption barrier (B1-B3)
- 6 步转换算法 (S1-S6)
- 3 个 case study

这种 "我把正反方都摆出来, 但我站这边" 的写法, 在 ML 圈不太常见, 倒是 philosophy paper 常这么干。附录 A.2 那个 "银河系尺度 intelligence" 的 thought experiment 真的很 philosophical, 我读的时候笑了 — 这帮人写 paper 写出 existential crisis 了。

---

## 最硬的反驳: LLM 不就是永远更强吗

AV1 是最能打动人的反对意见: **"同一个 task, 同一代模型, LLM 永远比 SLM 强。"** 这是 scaling law 的直接推论, 你很难反驳。

但作者们给了四个反击, 我觉得 A11 最 sharp:

> Agent 本身就是在做 task decomposition。如果复杂任务被拆成 10 个简单子任务, 每个 sub-task 都简单到不需要 "general world understanding", 那 LLM 的 "semantic hub" 优势就废了。

这跟你自己的直觉应该是一致的 — 你在 Tesla 讲自动驾驶的时候说过, 大 monolithic network 想做 everything 往往不行, 拆成几个 specialist 反而每个都能 train 得更 robust。agent 也是一样, **decomposition 是 generalization 的天敌**。

---

## 转换算法 (S1-S6) 其实就是个 data flywheel

人话版本:

1. **先记录**: 给现有 agent 的每次 LLM call 加 logger, 记下 (prompt, output, tools, latency)
2. **清洗**: 攒到 10k-100k 条, 把 PII / PHI 抹掉
3. **聚类**: 看哪些 call 长得像, 归成几类 — "哦 30% 是 intent recognition, 20% 是 code gen, 15% 是 summarization..."
4. **选 base model**: 每类挑一个 SLM 底座, 比如 Phi-3 或者 Hymba
5. **fine-tune**: 用 LoRA 或者直接 distill LLM 的输出
6. **轮换**: 隔段时间用新数据再 train 一遍

说白了就是: **先用 LLM 跑业务, 跑着跑着攒数据, 然后把 LLM 替换成 fine-tuned SLM。** LLM 在这里变成了 "数据标注员" 而不是 "生产引擎"。

这个 idea 其实 industry 里很多人都在做, 只是没写成 paper。Anthropic 内部估计也有类似 playbook, 只不过他们不会公开说 "少用我们的 API"。

---

## Case Study 的数字值得记一下

| Agent | 可替换比例 |
|-------|----------|
| MetaGPT (模拟软件公司) | 60% |
| Open Operator (workflow automation) | 40% |
| Cradle (GUI 自动操作) | 70% |

平均 ~57%。假设 SLM 的成本是 LLM 的 1/20 (paper 说的 10-30x 中位数), 那整体成本能砍到原来的 50% 左右。如果 100% 替换, 就是 95% 成本下降。

这个数字对 enterprise CTO 来说是非常 aggressive 的。paper 里引的 market data 说 agentic AI 市场 2024 年 5.2bn, 2034 年预计 200bn。如果整个 industry 能省 50% inference cost, 那就是 100bn 级别的 story。

---

## 我觉得他们说得对的地方

1. **"Capability 不是 binding constraint, parameter count 才是"** — 这句话 hit the nail on the head。现在 SLM 在 narrow task 上已经够用了, 瓶颈是大家不知道怎么 deploy。

2. **Heterogeneous system 是 natural evolution** — Figure 1 右边那个 "code agency" 的架构, 一个 controller code 调多个 specialist SLM, 这就是 future。不是 "SLM 替代 LLM", 是 "SLM 做 90% 的活, LLM 做 10% 的活"。

3. **Agent 天然产 training data** — A7 这个点很妙。agent 每次调 LLM 都在产 (prompt, output) pair, 这就是 SLM 的训练数据。agent 用得越多, SLM 训练越充分, 替换越容易。这是一个 positive feedback loop。

---

## 我觉得他们说得不够的地方

1. **Barrier B1 (57bn USD sunk cost) 被轻描淡写了**

paper 说这是 "inertia", 但实际上这是 **capital lock-in**。AWS, Azure, GCP 投了几百亿建 data center, 他们不可能主动推 "用小模型跑你本地"。这不是技术问题, 是 incentive structure 问题。你想让 SLM 真正起飞, 得等一批新的 vendor (比如 NVIDIA 自己卖 edge GPU, 或者 Apple / Qualcomm 推 on-device) 打破现有格局。

2. **Benchmark 问题 (B2) 是 root cause**

paper 在 B2 里说 "SLM 评估用 generalist benchmark 不公平", 但没给解决方案。MMLU, HumanEval, GSM8K 这些 benchmark 全是 generalist 的, 你要让 industry 接受 SLM, 得有一套 **agentic-native benchmark**。Berkeley Function Calling Leaderboard 是个好开始, 但远远不够。需要有人搞一个 "AgentBench" 之类的, 专测 agent 在 real workflow 里的 end-to-end success rate。

3. **Conversion algorithm 太乐观**

S1-S6 看起来 linear, 但实际部署中, 你跑到 S5 发现 fine-tuned SLM 在某些 case 上崩了, 得回到 S3 重新 cluster, 或者发现 S2 的 data 有 distribution shift, 得回到 S1。这是个 **messy iterative process**, paper 给的 framework 太 clean。

4. **Edge inference 的真实困难没谈**

paper 提了 ChatRTX, 但那是 desktop GPU。手机上 SLM 部署还远没成熟 — iOS 的 RAM 限制、Android 的 fragmentation、电池、模型更新、隐私合规... 这些都是 hard problems。Apple Intelligence 用 ~3B 模型在 A17 Pro / M1+ 上跑, 但那也是 Apple 全栈控制才做到的。对 Android 阵营, 这个 future 还很远。

5. **Test-time scaling 的 caveat**

paper 在 A10 里说 "reasoning 让小模型也能 scale up at inference"。这个 claim 在 math / code 上成立 (DeepSeek-R1 证明了), 但在 open-domain conversation 上, test-time compute 的 ROI 不明确。你让 1.5B 模型 chain-of-thought 想半天, 它想出来的东西可能不如 70B 直接 one-shot。**test-time compute 不是 free lunch**, 它对小模型的提升有 ceiling。

6. **Cold start 问题**

新 agent 上线, 前 10k-100k 条数据哪来? 难道先用 LLM 跑半年再切 SLM? 这中间的成本和 delay paper 没谈。实际 industry 里, 这个 cold-start 期往往是 deal-breaker — CTO 不会接受 "我们先花 6 个月 LLM 成本, 然后才能切 SLM"。

---

## 跟你自己工作的关联

Andrej, 你这几年一直在推 "small, understandable, educational" 的路线 — NanoGPT, LLM101n, makemore, micrograd。这篇 paper 从 industry 角度论证了你在教学角度一直在做的事: **你不需要 175B 才能做有用的 LM work**。

只不过 NVIDIA 这帮人是从 "agentic deployment economy" 切入, 你是从 "让人类理解 LM 内部" 切入, 终点都是 **"democratize LM capability by going small"**。

你们俩家的交集在 **Phi 系列** — Microsoft 的 Phi-2 / Phi-3 用合成数据 + curriculum 训练 2.7B / 7B 达到大模型水平, 这既是 "SLM is enough" 的证据, 也是 "data quality > parameter count" 的证据, 跟你一直强调的 "data is the new code" 完全一致。

---

## 我的预测

我赌这篇 paper 的核心论点 **2-3 年内会被 industry 广泛接受**, 但路径不会是 "LLM → SLM 替换", 会是:

1. **2026-2027**: Heterogeneous agent framework 成为标配 — LangChain / LlamaIndex 之类会内置 router, 自动决定 invoke SLM 还是 LLM。NVIDIA Dynamo 之类的 inference OS 会支持多模型混部。

2. **2027-2028**: On-device SLM 在 iOS / high-end Android 上 mass deploy, Apple Intelligence 2.0 / Google Gemini Nano 2 会真正 usable。Cloud agent 和 edge agent 会形成 two-tier 生态。

3. **2028-2030**: "Agent-native benchmark" 出现并成为 standard, MMLU 之类被边缘化。SLM 在 agentic benchmark 上全面碾压 LLM (per FLOP), industry 重新校准 "什么是 SOTA"。

4. **2030+**: LLM 变成 "backend reasoning engine for hard cases", 日常 agent call 90%+ 是 SLM。Cloud inference revenue 结构重塑, NVIDIA 自己的 GPU 卖法可能从 "datacenter H100" 向 "edge RTX / Grace" 倾斜。

这篇 paper 可能会像 [Ray Kurzweil 的 prediction](https://singularity.com/predictions) 一样, 方向对, 但 timeline 偏乐观。

---

## 最后吐槽一句

NVIDIA 出这篇 paper 挺微妙的 — 他们卖 H100 给云厂商建 LLM inference 集群, 同时又发 paper 说 "大家别用 LLM, 用 SLM"。左手卖铲子, 右手说金矿快挖完了。

但这也合理 — NVIDIA 的 edge GPU 产品线 (RTX, Jetson, Grace) 同样需要 story。这篇 paper 本质上是在为 **"edge AI / on-device AI"** 这个下一波市场做 narrative 铺垫。NVIDIA 不会只靠 datacenter, 他们要覆盖从 cloud 到 edge 的全栈。

这篇 paper 的真正 audience, 是那些还在犹豫 "要不要自己 serve 小模型" 的 enterprise decision maker。NVIDIA 在跟他们说: "放心, 这不是 fringe idea, 这是 future, 我们 NVIDIA 都背书了。"

参考:
- Paper 全文: [research.nvidia.com/labs/lpr/slm-agents](https://research.nvidia.com/labs/lpr/slm-agents)
- Hymba: [arxiv.org/abs/2411.13676](https://arxiv.org/abs/2411.13676)
- NVIDIA Dynamo: [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
- Berkeley Function Calling Leaderboard: [gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- Apple Intelligence: [apple.com/apple-intelligence](https://www.apple.com/apple-intelligence/)
- DeepSeek-R1: [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

---

# SLMs are the Future of Agentic AI — 深度解读

Andrej 朋友, 这篇 paper 是 NVIDIA Research 的一篇 position paper, 2025年的观点性论文。我读完之后的直觉是: 作者们戳中了一个真实的问题, 但表达得过于克制。让我把这个 paper 撕开, 把里面的骨架和血肉都摊出来给你看。

---

## 1. Paper 的核心命题

作者们把整个 position 压缩成三个 view (V1–V3), 用七个 argument (A1–A7) 支撑, 又列出三个 alternative view (AV1–AV3) 和三个 barrier (B1–B3) 反驳自己, 最后给一个六步的 LLM→SLM 转换算法 (S1–S6)。这是一种非常 formal 的论辩结构, 在 position paper 里很罕见。

核心命题:

$$\text{SLMs} \;\succeq\; \text{LLMs} \quad \text{in agentic context}$$

其中 " $\succeq$ " 表示 "at least as suitable as", 这个偏序关系沿 three axes 展开:

- **V1 (Capability)**: $\exists$ capability threshold $\tau$ s.t. SLMs 已超过 $\tau$ → "sufficiently powerful"
- **V2 (Suitability)**: SLMs 与 agentic architecture 的 alignment 比 LLMs 更紧 → "inherently more operationally suitable"
- **V3 (Economy)**: $\text{cost}(\text{SLM}) \ll \text{cost}(\text{LLM})$ under realistic serving scenarios → "necessarily more economical"

我注意到作者们刻意用 "necessary" 而不是 "preferable" — 他们声称这是一个**因果必然性** (causal necessity) 而不是偏好选择。这是一种比较强的 claim。

---

## 2. Definition 上的微妙处理

WD1 把 SLM 定义成 "能 fit 进 consumer device + 实用 latency" 的 LM, 而 WD2 直接用 "not SLM" 定义 LLM。这种**互补定义** trick 蛢明:

- Timelessness: 避开 "参数 < 10B" 这种会被 Moore's law 淘汰的定义
- Limit argument (Appendix A.2): 用银河系尺度的 "super-intelligent system" 和 "infinitely small intelligent system" 两个极端做 thought experiment, 类比到 human brain-to-body mass ratio

这个 limit argument 我觉得有点 over-stylized, 但背后的直觉是对的: intelligence 不是 size 的单调函数。这部分像是 anthropic principle 的味道。

---

## 3. 技术论点深度剖析

### 3.1 A1: SLMs 已经足够强大

作者们列举了一系列 SLMs 的 "competitive off-the-shelf performance":

| Model | Params | Comparison |
|-------|--------|------------|
| Phi-2 | 2.7B | matches 30B on commonsense, runs ~15× faster |
| Phi-3 small | 7B | matches 70B contemporaries |
| Nemotron-H | 2/4.8/9B | matches 30B dense, 1 order less FLOPs |
| SmolLM2 | 125M–1.7B | matches 70B from 2 years ago |
| Hymba-1.5B | 1.5B | outperforms 13B |
| DeepSeek-R1-Distill-Qwen-7B | 7B | beats Claude-3.5-Sonnet-1022, GPT-4o-0513 |
| RETRO-7.5B | 7.5B | matches GPT-3 175B, 25× fewer params |
| xLAM-2-8B | 8B | SOTA on tool calling, beats GPT-4o |

这里的关键 insight 是: **scaling laws 曲线的 steepening**。作者们引 [Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556) 和 [Kaplan et al.](https://arxiv.org/abs/2001.08361) 之后, 指出新 SLM 的 scaling curve 比旧 LLM 的陡。

具体地, 如果我们写出 scaling law 的形式:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中:
- $L$ 是 loss, $N$ 是 parameter count, $D$ 是 data tokens
- $E$ 是 irreducible loss (entropy floor)
- $\alpha, \beta$ 是 scaling exponents, $A, B$ 是常数
- 上下标都没有特殊含义, 就是幂律拟合系数

新 SLMs 之所以能 "punch above their weight", 是因为:
1. **训练数据质量**: Phi 用合成 + filtered data, SmolLM2 强调 data-centric training
2. **架构创新**: Mamba-attention hybrid (Hymba, Nemotron-H), retrieval augmentation (RETRO)
3. **Test-time compute**: DeepSeek-R1-Distill 把 reasoning 蒸馏到小模型

### 3.2 A2: 经济性论证

作者们给了一个数字: 7B SLM vs 70–175B LLM 的 inference 成本是 **10–30× 便宜** (latency, energy, FLOPs 三个维度)。

这里我补一个点: 真正的经济模型应该用 **dollar per token** 而不是 FLOPs。作者们略过了 batch size, KV cache 内存、GPU utilization 等等。对 7B 模型, 一个 A100 / H100 单卡就能 hold, 而 175B 需要多卡 TP + PP, 复杂度完全不同。

形式化:

$$\text{Cost}_{\text{token}} = \frac{C_{\text{hw}} \cdot T_{\text{token}}}{\text{Throughput}} + C_{\text{op}}$$

其中 $C_{\text{hw}}$ 是硬件折旧成本, $T_{\text{token}}$ 是每 token 处理时间, $C_{\text{op}}$ 是运维 overhead。

对 SLM, $C_{\text{op}} \to 0$ (单机部署), 对 LLM, $C_{\text{op}}$ 显著 (分布式 serving, load balancing, model parallelism)。

### 3.3 A4–A5: Agentic interactions 暴露 LM 的是 narrow 功能

这是 paper 里我觉得最有说服力的部分。

作者们说: 一个 agent 本质上是 "a heavily instructed and externally choreographed gateway to a language model"。LLM 袄训练的 **generality** 大部分被 prompt template 和 tool schema 框死。

A5 强调 **behavioral alignment**: agent 要严格遵循 JSON / XML / Python schema。对 LLM, 这是一种 "format suppression"; 对 fine-tuned SLM, 这是 native behavior。

经验上, 我会说这跟 tool use / function calling benchmark 上的 SLM 表现强势一致 — 比如 Berkeley Function Calling Leaderboard ([gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)) 上 xLAM-2-8B 打败 GPT-4o 就是例证。

### 3.4 A6: Heterogeneous agentic system

Figure 1 描述了两种 mode of agency:

**Left (Language Model Agency)**:
```
User ←→ [Orchestrator LM] ←→ [Tool 1, Tool 2, Tool 3]
```
这里 LM 既当 HCI 又当 orchestrator。

**Right (Code Agency)**:
```
User ←→ [Controller Code] ←→ [HCI LM, Sub-LM, Tool 1, Tool 2]
```
这里 controller code 主导, LMs (可以是多个 SLMs) 各司其职。

作者们主张 right side 是 SLM 的天然栖息地 — 一个 SLM 负责 HCI, 另一个负责 tool calling, 第三个负责 format conversion。

这跟我自己最近一两年在 [Tesla 自动驾驶上的直觉](https://www.youtube.com/watch?v=oBkll0KHoA0) 很像: 大 monolithic network 想要做 everything 的往往不行, 但如果分解成几个 specialist, 反而每个都能 train 到更 robust。

---

## 4. AV1 (最有意思的反驳): LLM generalists 永远更强

作者们把最硬的反驳 AV1 列出来:

> "Let T be a single task using general language and let L, S be a large and a small language model of the same generation, respectively. The performance of L on T will always trump that of S."

这里有个 quantifier 的问题: "**always**" 是不是真的成立? 作者们的反驳有四点:

- **A8**: scaling law 研究假设架构不变, 但实际新 SLM 用了新架构 (Mamba, hybrid heads)
- **A9**: SLM 的 flexibility 让它能 fine-tune 到 task T 上, 这种 post-training gain scaling law 研究不 capture
- **A10**: test-time compute (reasoning) 让小模型也能 scale up at inference
- **A11**: 复杂任务在 agent 里会被 decomposition 拆解, "semantic hub" 在 sub-task 粒度下用不上

A11 这个反驳我觉得是最 sharp 的: 如果 agent 本身就是在做问题分解, 那 LLM 的 "semantic hub" advantage 在 sub-task 粒度下就没用了。这是把 "generalization 优势" 转换成了 "decomposition 责任"。

引用 [MIT 2025 关于 semantic hub 的研究](https://news.mit.edu/2025/study-large-language-models-encode-world-knowledge-across-languages-0211): LLM 确实有跨语言 / 跨模态的 shared semantic representation, 但其 utility 依赖于 task complexity 本身没有被 decomposed away。

---

## 5. LLM→SLM 转换算法 (S1–S6)

这是 paper 最 actionable 的部分, 让我重写一下, 因为作者的写法太抽象:

**S1 — Usage Data Collection**:
在 agent 的 non-HCI call 处加 logger, 记录 $(p_i, o_i, t_i, l_i)$ — input prompt, output, tools, latency。要做 encryption + anonymization (引 [WorkOS guide](https://workos.com/docs/build-secure-ai-agents))。

**S2 — Data Curation**:
收集到 10k–100k 条后做 PII/PHI scrubbing, 并 paraphrase application-specific inputs。这里引了 [Yu et al. 2024](https://arxiv.org/abs/2402.13659) 的 privacy-preserving instruction alignment。

**S3 — Task Clustering**:
对 $(p_i, t_i)$ 做 unsupervised clustering, 比如 k-means on sentence embeddings (text-embedding models 上的 $k$ clusters)。Cluster 定义 candidate specialization task。

**S4 — SLM Selection**:
对每个 cluster 选 base SLM。Selection criteria 包括 context window, license, footprint。

**S5 — Specialized Fine-tuning**:
用 LoRA ([arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)) / QLoRA ([arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)) / full FT。LoRA 的更新可以形式化为:

$$W' = W + \Delta W = W + BA$$

其中 $W \in \mathbb{R}^{d \times k}$ 是 frozen pre-trained weight, $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $r \ll \min(d, k)$ 是 low-rank 约束。$A$ 用 Gaussian init, $B$ 用 zero init 保证 $\Delta W = 0$ at start。

也可以用 knowledge distillation: specialist SLM 学 LLM 在 task-specific data 上的 output distribution:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \text{KL}(p_S \Vert p_T) + (1-\alpha) \cdot \text{CE}(y, p_S)$$

其中 $p_S$ 是 student logits, $p_T$ 是 teacher logits, $\alpha$ 是 distillation weight。

**S6 — Iteration**:
Retrain SLM + router (something like a routing network that decides which expert SLM to invoke for a given input query) 周期性更新。

---

## 6. Case Studies 里的数字

作者给了三个 open-source agent 的 SLM 替换率估计:

| Agent | Estimated Replaceable % |
|-------|------------------------|
| MetaGPT | 60% |
| Open Operator | 40% |
| Cradle | 70% |

平均 ~57%。如果整个 industry 50% 的 agentic LLM calls 能被 SLMs 取代, 而 SLM 的 cost 是 LLM 的 1/10 — 1/30, 那 overall cost 能降到原来的:

$$\text{Cost}_{\text{new}} = 0.5 \cdot 1 + 0.5 \cdot \frac{1}{20} \approx 0.525$$

也就是接近 2× 成本节省, 或者说 90% 的成本归约如果整 100% 替换。这个数字非常 significant。

---

## 7. 我 (Karpathy) 的一些看法

读完之后我有几个直觉上的反应:

**支持的地方**:
1. **"Inverted scaling" 直觉**: 这个 paper 的核心论点跟我的直觉是一致的 — 在 narrow, well-defined task 上, 越大的模型边际收益递减甚至负收益。我在 [NanoGPT](https://github.com/karpathy/nanoGPT) 上 train 过小 GPT, 在 [LLM101n](https://github.com/karpathy/LLM101n) 的教学经历都让我看到: 你不需要 175B 才能让模型 useful。

2. **Specialization > Generality**: E2E 的单一巨模型在窄域上经常被 specialist 小模型打败 — 这是 ML 历史反复出现的规律, 从 mixture-of-experts 到 GPT-4 本身 (GPT-4 估计是 MoE)。

**保留意见**:
1. **Barriers B1–B3 太轻描淡写了**: 作者们把 B1 (云厂商 sunk cost 57bn USD) 当作 "inertia", 但实际上是 economic lock-in。这不是技术问题, 是 capital allocation 问题。一旦资本做出选择, 想扭转需要新的资本势力进场, 不只是 "技术证明 SLMs 更优" 就够。

2. **Generalist benchmark 问题 (B2) 严重**: 我自己一直诟病现在的 eval。如果大家用 MMLU 评 SLM, SLM 永远打不过 LLM, 这不能说明 SLM 不适合 agentic use case。作者们引 [Hymba paper](https://arxiv.org/abs/2411.13676) 说 "在 agentic benchmarks 上 SLMs 反超 LLMs" 是个关键 insight, 但 paper 没给详细数据。

3. **Conversion algorithm 过于乐观**: S1–S6 看起来 linear, 但实际部署中 S6 的 iteration 往往会发现 S3 的 clustering 错了, 需要回到 S1。需要更现实的 feedback loop 描述。

**他们没说但我认为重要的**:

4. **Edge inference 的真实障碍**: paper 提了 ChatRTX, 但移动端 (iOS / Android) 上 LLM 部署还远未成熟。电池、RAM、隐私、模型更新都是 hard problems。如果 SLM 不能 mass-deploy 到 edge, 那这个 "future" 还很远。

5. **Test-time scaling 的 caveat**: 我自己最近在 [DeepSeek-R1 系列工作](https://github.com/deepseek-ai/DeepSeek-R1) 上看到, test-time compute 能让小模型在 math 上达到大模型水平, 但这是非常 task-specific 的。Open-domain conversation 上 test-time scaling 的 ROI 不明朗。

6. **数据飞轮 (S1–S2) 的 cold start**: 一个新部署的 agent system, 前 10k–100k 条数据从哪来? 难道先用 LLM 跑 6 个月再切换? 这中间的成本和延迟 paper 没谈。

---

## 8. 相关工作延伸联想

这篇 paper 与几个研究方向可以互为支撑:

- **Mixture of Agents**: [arxiv.org/abs/2406.04692](https://arxiv.org/abs/2406.04692) "MoA: Mixture of Agents" 论 multiple LLMs 协作, 这正是 heterogeneous agentic system 的一种形式化。

- **On-device LLM**: Apple Intelligence ([apple.com/apple-intelligence](https://www.apple.com/apple-intelligence/)) 和 Google Pixel AI 都是端侧 SLM 部署的实例, 都用 ~3B 模型 — 这是 paper 所描述 "SLM for agentic" 的工业实现。

- **Toolformer** ([arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)): paper 提到 Toolformer 6.7B 通过 tool use 击败 GPT-3 175B, 这是 "SLM + tooling > LLM alone" 的直接证据。

- **EvoLLM / Evolutionary optimization of model architectures**: 我自己在关注 [Syneron](https://arxiv.org/abs/2402.15058) 类工作, 用进化算法搜 SLM 架构 — 这跟 paper 强调的 "different architectures for different sizes" 一致。

- **Router-based routing**: PolyRouter / FrugalGPT 类工作 ([arxiv.org/abs/2305.05176](https://arxiv.org/abs/2305.05176)) 给了 "何时 invoke SLM, 何时 invoke LLM" 的 router 形式化, 是 paper 的 S1–S6 缺失的一环。

---

## 9. 总结

Paper 的核心 claim — SLMs 是 agentic AI 的未来 — 我认为方向是对的, 但 path 不是 "replacement", 而是 **"heterogeneous co-design"**: 系统设计师在 agent 粒度上 mix-and-match, 让 SLMs 拿大多数 volume, LLMs 拿少数需要 generality 的 calls。

作者们把 V1, V2, V3 framing 为 "**necessary**" 比 "preferable" 更强, 这招 paper 受得住 — 但只在他们承认 "idealized agent decompositions" 的前提下。一旦 agent 自身要求 emergent reasoning (比如 long-horizon planning on unseen environment), SLMs 还需要赶上。

我会 follow-up 关注 NVIDIA 后续的 [research.nvidia.com/labs/lpr/slm-agents](https://research.nvidia.com/labs/lpr/slm-agents) 上有没有更多实测数据放出来, 因为这篇 paper 几乎没有 quantitative ablation, 这对于一个 position paper 来说可以接受, 但对 NVIDIA 这种有 compute 资源的 lab 来说, 是个 missed opportunity。

参考链接:
- Paper PDF: [arxiv.org/abs/2505.11407](https://arxiv.org/abs/2505.11407) (预印本)
- NVIDIA Research: [research.nvidia.com/labs/lpr](https://research.nvidia.com/labs/lpr)
- Berkeley Function Calling Leaderboard: [gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- Hymba paper: [arxiv.org/abs/2411.13676](https://arxiv.org/abs/2411.13676)
- DeepSeek-R1: [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- Toolformer: [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)
- LoRA: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- QLoRA: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- SmolLM2: [arxiv.org/abs/2502.02736](https://arxiv.org/abs/2502.02736)
- Phi-3 tech report: [arxiv.org/abs/2404.14219](https://arxiv.org/abs/2404.14219)
- RETRO: [arxiv.org/abs/2112.04426](https://arxiv.org/abs/2112.04426)
- xLAM: [arxiv.org/abs/2409.03215](https://arxiv.org/abs/2409.03215)
