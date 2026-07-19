---
source_pdf: ACE-RTL When Agentic Context Evolution Meets.pdf
paper_sha256: 5d1c89cb959a7842bfb2f6c8b4becb32c153030982dce9247a6dd5bd780db071
processed_at: '2026-07-18T00:16:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ACE-RTL 深度解析：当 RTL-Specialized LLM 遇上 Agentic Framework

Andrej，这篇来自 NVIDIA 的 paper 我读完之后的整体直觉是：它本质上是在做一个 **"分工"** 的工作 —— 把"领域知识"（domain knowledge）和"通用推理"（general reasoning）这两个能力分别交给最擅长它的 model，然后用一个 agentic loop 把它们 glue 起来。下面我尽量 build 你的 intuition。

---

## 1. 核心洞察：两条正交路线的互补性

paper 开篇就画了一张 Figure 1，对比 prior work 的两条独立 path：

**(a) Domain-adapted RTL models**（RTLCoder, CraftRTL, ScaleRTL, OriGen, CodeV）：
- 在 RTL 数据上 SFT，内化 hardware semantics
- 问题：general capability 弱（long-context reasoning、multi-step planning、instruction following 都不够）

**(b) Agentic systems with frontier LLMs**（VerilogCoder, MAGE）：
- 用 GPT-4-Turbo / Claude3.5-Sonnet + simulator/waveform toolchain
- 问题：缺乏大规模 RTL 训练带来的 deep hardware semantics

这两条路线的限制是 **正交的**（orthogonal）—— 一个擅长"know-how"但不擅长"think-through"，一个擅长"think-through"但不懂"know-how"。ACE-RTL 的核心 thesis 就是：**一个 model 不需要同时具备两种能力，只要系统设计能把它们组合起来**。这个观察其实和 AlphaCode 风格的 "specialized generator + test-time search" 思路在哲学上一脉相承。

参考链接：
- ScaleRTL: https://arxiv.org/abs/2506.05566
- VerilogCoder (AAAI'25): https://doi.org/10.1609/aaai.v39i1.32007
- MAGE: https://arxiv.org/abs/2412.07822
- CraftRTL: https://arxiv.org/abs/2409.12993
- CVDP benchmark: https://arxiv.org/abs/2506.14074

---

## 2. 系统架构：Generator / Reflector / Coordinator 三件套

### 2.1 Generator —— "懂 RTL 的工匠"

**Base model**：Qwen2.5-Coder-32B-Instruct  
**Training**：32 nodes × 8 A100 GPUs，3 epochs，context window 32,768 tokens，global batch size 128，cosine-annealing LR scheduler  
**Serving**：vLLM for low-latency inference，temperature = 1.2

#### 数据 pipeline 的关键技术细节

这是整个工作里我觉得最值得拆解的部分。1.7M 样本不是凭空来的，pipeline 长这样：

**Step 1: Raw collection → 5M RTL scripts**  
从 public repos + open hardware projects（比如 OpenCores、chipyard 之类）抓取。

**Step 2: Aggressive filtering → 157K 高质量 scripts**

过滤规则非常 aggressive，我列一下：
- 去重（deduplication）
- 去除机器生成代码（netlist、HLS output）—— **这步很关键**，因为 netlist 是综合后的，没有 abstraction 价值；HLS output 是 C→RTL 编译器产物，code pattern 偏机械
- 行数限制：30 ≤ lines ≤ 2000 —— 排除 trivial snippet 和巨型 monster file
- **Icarus Verilog (iverilog) syntax validation**：丢弃所有语法错误代码。iverilog 是 open-source Verilog simulator (https://github.com/steveicarus/iverilog)，paper 中多次出现，因为它是 lightweight 且 portable 的，整个 agentic loop 都依赖它
- **Benchmark contamination 检测**：用 Jaccard similarity 衡量 collected code 与 benchmark golden solution 的相似度，阈值 0.8 以上丢弃

Jaccard similarity 公式：

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

其中：
- $A$ = 当前 RTL snippet 的 token / n-gram 集合
- $B$ = 某 benchmark golden solution 的 token / n-gram 集合
- $|A \cap B|$ = 两个集合的交集大小
- $|A \cup B|$ = 并集大小

Jaccard ∈ [0, 1]，0 表示完全不同，1 表示完全相同。0.8 阈值相当严格。ScaleRTL 也用了同样的 metric（reference [12]）。

**Step 3: Specification–Code pair generation → 1.7M pairs**

这里用了一个 **in-context learning pool**：手动构造 32 个高质量 (spec, RTL) 例子，覆盖 code generation / modification / debugging 三类任务。对于每个 raw RTL script，随机从 pool 里采样一个 example 作为 ICL context，然后 prompt **多个 LLMs**（GPT-OSS-120B、DeepSeek-R1、以及一些 proprietary models）生成 task-specific specification 和对应 golden RTL。

这里有个细节我特别想强调 —— **modification 和 debugging 任务的构造方式**：specification 里包含 golden code 的 simplified 或 intentionally faulty 版本，模拟真实 design correction workflow。这正是为什么 Table 1 里 ACE-RTL-Generator 在 **Code Modification (cid004)** 上能拿到 65.09% Pass@1，超过 GPT-5 21.45 个百分点。**数据分布和评估分布对齐了**。

参考链接：
- Qwen2.5-Coder technical report: https://arxiv.org/abs/2409.12186
- GPT-OSS model card: 见 reference [1]
- DeepSeek-R1: https://arxiv.org/abs/2501.14448
- vLLM (PagedAttention): https://arxiv.org/abs/2309.06180

### 2.2 Reflector —— "通用推理诊断师"

**Model**：Claude4-Sonnet  
**Trigger**：Generator 产出的代码 iverilog compile/simulate 失败时  
**Input**：specification + buggy RTL code + structured simulation feedback  
**Output**：structured diagnostic report，包含 (a) root cause explanation，(b) high-level fix guidance

#### Feedback 结构化的工程细节

paper 里说 "Automated scripts invoke the simulator and convert the raw logs into a structured format that captures the error message and the expected versus actual signal behavior."

我推测这个 structured format 大概是这样的 JSON-ish 结构：

```
{
  "error_type": "mismatch" | "timeout" | "compile_error" | ...,
  "failing_test_vectors": [
    {"input": {...}, "expected": {...}, "actual": {...}, "cycle": N}
  ],
  "assertion_violations": [...],
  "compile_errors": [...]
}
```

Reflector 的核心价值在于：它 **不直接改代码**，而是产出 high-level fix guidance 给 Generator。这是一个 **抽象层级跃迁** —— 把"波形 mismatch"这种低层信号 upgrade 成"missing alignment transformation on the data stream"这种 design-intent 层面的描述。

### 2.3 Coordinator —— "记忆与控制中心"

这个组件最容易被忽视但其实是 paper 的灵魂。它做两件事：

**Function 1: Structured history aggregation**  
跨 iteration 跟踪：哪些 error 被 identify、什么 fix 被 suggest、fix 是否 work。目的是 **prevent regression** —— Generator 不会在下一轮重新陷入已经被 fix 过的坑。

**Function 2: Adaptive restart mechanism**  
监控跨 iteration 的 progress。如果同一个 error 在多轮里持续存在（即进入 plateau），Coordinator 判定当前 trajectory 难以修复，**discard 当前实现，prompt Generator 从 spec 重新生成**。这本质上是 **simulated annealing 中的"re-heating"** 操作，跳出 local minimum。

paper 里 Case Study III 完美展示了这个机制 —— Clock Jitter Detection 模块卡在"jitter detection 在 valid baseline interval 建立之前就触发"这个 fundamental design flaw 上，所有 syntactic variant 都保留了这个 early-trigger issue。Coordinator 触发 restart 后，从 failed attempts 里 distill 出"need to validate the first measurement interval before performing detection"这个 high-level insight，注入到 restart 的 prompt 里，最终通过 validity flag 解决问题。

**这里 build 的 intuition 是**：LLM 代码生成的失败模式不只是"语法错"或"小逻辑错"，更常见的是"整体架构选错了"。对于架构层面的错误，**local refinement 永远修不好**，必须 restart from scratch 但带着 learned insight。这是 Coordinator 设计的根本理由。

---

## 3. Parallel Scaling：把 stochasticity 变成优势

这个 idea 简单但 powerful。设 K = 5（并行进程数），每个进程独立从同一 spec 生成不同 initial RTL（因为 Generator temperature = 1.2 引入 stochasticity），然后各自独立跑 Generator→Reflector→Coordinator loop。**任何进程成功 → 立即终止其他所有进程**。

#### 为什么 work？数学直觉

假设单进程在 N 轮内 pass 的概率为 $p_N$，K 个独立进程至少一个 pass 的概率是：

$$P_K(N) = 1 - (1 - p_N)^K$$

对于 K=5，即使单进程 $p_N = 0.5$，$P_5(N) = 1 - 0.5^5 = 0.96875$。

更关键的是 **expected iterations to first success** 大幅下降，因为不同 initial code 的"可修复性"差异巨大 —— 有的 initial 一两轮就 pass，有的死循环到上限。Parallel scaling 把"最 lucky 的 trajectory"挑出来，相当于 **best-of-K over trajectories**。

#### 实验数据（Figure 6）

| CVDP Category | Serial iterations | Parallel (K=5) iterations | Speedup |
|---|---|---|---|
| cid002 Code Completion | 11.33 | 3.95 | 2.87× |
| cid003 Spec-to-RTL | 11.25 | 4.23 | 2.66× |
| cid004 Code Modification | 9.25 | 3.75 | 2.47× |
| cid016 Code Debugging | 13.36 | 4.37 | 3.06× |

平均 **~4 iterations** 就能 converge，这非常 impressive，因为 RTL debugging 的 search space 巨大。

**Build intuition**：这和 AlphaCode 的 "massive sampling + filter by execution" 哲学类似，只不过 ACE-RTL 是 "massive trajectories + filter by simulator"。差别是 AlphaCode 是 sample-and-prune，ACE-RTL 是 sample-and-debug-in-parallel。

---

## 4. CVDP Benchmark：为什么 VerilogEval 已经 saturated

paper 在 Section 2.3 给了一个挺 sharp 的 critique：

> VerilogEval 和 RTLLM mostly evaluate small, self-contained RTL problems, emphasizing short specification-to-RTL snippets rather than sustained reasoning over realistic design contexts.

CVDP-v1.0.2 包含 4 类任务：

| CID | Task | # Problems |
|---|---|---|
| cid002 | Code Completion | 94 |
| cid003 | Spec-to-RTL Generation | 78 |
| cid004 | Code Modification | 55 |
| cid016 | Code Debugging | 35 |

覆盖的 hardware domain：arithmetic units、on-chip protocols、memory hierarchies、DSP kernels、communication blocks、control logic。

#### 评估指标

**Pass@1**（standalone models）：

$$\text{Pass@1} = \mathbb{E}_{\text{problems}} \left[ \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{sample } i \text{ passes all tests}] \right]$$

其中 $n$ = 独立运行次数（paper 用 5 runs），$\mathbb{1}[\cdot]$ 是 indicator function。

**APR (Agentic Pass Rate)** —— paper 自己引入的指标：

$$\text{APR} = \frac{|\{p \in P : \exists \text{ iteration } t, \text{code}(t) \text{ passes all tests of } p\}|}{|P|}$$

其中 $P$ = 全部 problem 集合，$|P|$ = 问题总数。APR 衡量"agent 在多次 retry 中能 cover 多少 unique problem"。

paper 强调：**对于 agentic methods，Pass@1 等于 APR**（因为只要 agent 在某轮 pass 了就算 solved），所以用 APR 才能公平对比 standalone LLM。这是一个挺 thoughtful 的 metric design。

---

## 5. 主结果：Table 1 的关键解读

完整数据（百分比）：

| Type | Model | cid002 P@1 / APR | cid003 P@1 / APR | cid004 P@1 / APR | cid016 P@1 / APR |
|---|---|---|---|---|---|
| Open | Llama4-Maverick | 26.81 / 28.72 | 29.49 / 32.05 | 36.36 / 38.18 | 36.00 / 37.14 |
| Open | DeepSeek-v3.1 | 32.34 / 37.23 | 41.79 / 48.72 | 36.73 / 41.82 | 34.86 / 40.00 |
| Open | DeepSeek-R1 | 34.89 / 39.36 | 39.23 / 42.31 | 37.45 / 43.64 | 48.57 / 51.43 |
| Open | Kimi-K2 | 23.40 / 25.53 | 26.67 / 29.49 | 29.09 / 32.73 | 29.71 / 31.43 |
| Open | Qwen3-Coder-480B | 30.43 / 31.91 | 33.33 / 35.90 | 35.27 / 41.82 | 39.43 / 42.86 |
| Proprietary | 4o-mini | 35.10 / 37.23 | 41.56 / 45.45 | 41.48 / 44.44 | 50.00 / 58.82 |
| Proprietary | GPT-5 | 36.17 / 39.36 | 42.31 / 47.44 | 43.64 / 45.45 | 54.28 / 60.00 |
| Proprietary | Claude4-Sonnet | 37.94 / 39.36 | 49.49 / 51.28 | 42.91 / 49.09 | 51.43 / 54.29 |
| RTL | RTLCoder-v1.1-7B | 0.43 / 1.06 | 3.33 / 5.13 | 1.09 / 1.82 | 0.57 / 2.86 |
| RTL | CodeV-7B | 3.83 / 6.38 | 5.38 / 7.69 | 0.00 / 0.00 | 0.00 / 0.00 |
| RTL | OriGen-7B | 18.30 / 21.28 | 18.97 / 21.79 | 14.18 / 16.36 | 6.86 / 11.43 |
| RTL | CraftRTL-15B | 8.09 / 11.70 | 12.31 / 17.95 | 13.45 / 16.36 | 5.14 / 8.57 |
| RTL | ScaleRTL-32B | 25.32 / 27.66 | 28.97 / 33.33 | 25.82 / 30.91 | 30.86 / 37.14 |
| RTL | ScaleRTL†-32B | — / 29.79 | — / 35.90 | — / 32.73 | — / 40.00 |
| Ours | ACE-RTL-Generator | 39.57 / 40.43 | 49.74 / 52.56 | **65.09** / 67.27 | 56.00 / 57.14 |
| Ours | ACE-RTL (Claude4) | — / 80.85 | — / 89.74 | — / 81.82 | — / 88.57 |
| Ours | **ACE-RTL** | — / **80.85** | — / **96.15** | — / **90.91** | — / **91.43** |

#### 我读出几个关键 takeaways

**Takeaway 1：Prior RTL models 在 CVDP 上几乎崩溃。** RTLCoder、CodeV、OriGen、CraftRTL 的 Pass@1 基本是 0–18% 范围，远低于 GPT-5、Claude4。这说明 **早期 benchmark（VerilogEval、RTLLM）已经 saturated，不能 differentiate 真正强模型**。ScaleRTL 是唯一在 CVDP 上还行的 prior RTL model，但也只有 25–33% Pass@1。

**Takeaway 2：ACE-RTL-Generator 单独就已经超越所有 baselines。** Code Modification 上 65.09% vs GPT-5 的 43.64%，**绝对提升 21.45 个百分点**。这是 SFT 数据质量（spec-code pair 的 distribution 与 evaluation 对齐）的直接体现。

**Takeaway 3：Agentic framework 的增益巨大且 consistent。** 
- Generator 单独：40–67% APR
- ACE-RTL (Claude4，即 Generator 换成 Claude4)：80–90% APR  
- **ACE-RTL (我们的 Generator)**：80–96% APR

ACE-RTL vs ACE-RTL (Claude4) 的对比证明了 **specialized Generator 比 frontier generic LLM 在这个 loop 里更 effective**。直觉是：在迭代 refinement 中，Generator 需要能稳定产出 hardware-correct code，而 Claude4 虽然 reasoning 强但 RTL pattern 掌握不如 fine-tuned model。

**Takeaway 4：相对 best baseline 的最大提升是 44.87%。** 具体在哪一项 paper 没明说，但从 Table 1 推测应该是 cid003 Spec-to-RTL：96.15 vs Claude4-Sonnet 的 51.28，相对提升 (96.15-51.28)/51.28 ≈ 87.5%… 嗯不对。或者 cid004：90.91 vs 49.09 = 85.2%。可能 paper 算的是相对 ScaleRTL† 的 (90.91-32.73)/32.73 ≈ 177%... 我觉得 44.87% 可能是某种 averaged relative improvement，paper 没给精确定义，这算是一个 minor 的 reporting ambiguity。

---

## 6. 三个 Case Study 的技术深度

### Case Study I: RS232 Transmitter

**问题**：实现 baud-rate generator，需要 fractional clock divider 处理 system clock 和 baud rate 之间的非整数比。

**Claude4-Sonnet 的错误**：把 accumulator 和 pulse 耦合在 single assignment 里，丢失了 overflow-based pulse signal，导致 unstable timing。

**ACE-RTL-Generator 的正确实现**：accumulator-based divider，只 update lower bits，用 MSB 作为 stable pulse output。

这反映了一个 **canonical RTL pattern** —— "用 counter MSB 作为 pulse" 是 hardware designer 的 muscle memory，但 general LLM 没内化这个 pattern。Generator 通过 SFT 学到了这种 pattern。

直觉：hardware design 中有大量"idioms"（类似 software 中的 design patterns），比如：
- Clock divider 用 counter MSB
- FIFO 用格雷码 pointer
- Handshake 用 valid/ready 双向 signaling
- CDC 用 synchronizer chain

这些都是 SFT 能 capture 但 general pretraining 难以深度掌握的。

### Case Study II: 64b/66b Decoder

**问题**：bit-aligned extraction of mixed-mode data streams。

**关键现象**：ACE-RTL 早期快速 improve 但卡在很多 iteration 才 pass 最后一个 test case。

**Reflector 的诊断**：比较 expected vs actual outputs，infer 出 decoding logic 需要一个 **implicit alignment transformation**，这个 transformation 没在 spec 里明说。

**Build intuition**：这是 spec ambiguity 的典型案例。硬件 spec 经常隐含 assumption（比如"输入已经 word-aligned"），general LLM 不会主动 infer，而 Reflector 通过 **systematic failure pattern** 反推出这个 missing requirement。这是 **abductive reasoning** —— 从观察反推最可能的 latent cause。

### Case Study III: Clock Jitter Detection

**问题**：通过比较 measured edge intervals 与 timing threshold 评估 clock stability。

**关键现象**：early progress 然后 long plateau，所有 attempt 都 fail。

**根本原因**：每个 iteration 都在"valid baseline interval 建立之前就进行 jitter detection"，这是 **architectural flaw**，不是 local bug。

**Coordinator 的作用**：
1. 识别"连续多轮同样 assertion pattern + 无 meaningful improvement"
2. 触发 restart
3. 从 failed attempt 里 distill high-level insight："need to validate the first measurement interval before performing detection"
4. 用 clarified constraints prompt Generator 重新生成
5. 经过 2 次 restart，Generator 引入 validity flag 延迟 jitter detection，立即 pass

**Build intuition**：这是 paper 最 important 的 case。它揭示了 LLM debugging 的一个 fundamental limit —— **local search 无法 fix architectural error**。Coordinator 的 restart-with-distilled-insight 机制本质上是在 **search space 之间跳转**，而不是在单个 search space 内探索。这和 program synthesis 中的 "restart with learned constraint" 思路相通。

---

## 7. VerilogEval-Human-v2 上的对比（Table 2）

| CraftRTL P@1 | Claude4-Sonnet P@1 | ACE-RTL-G P@1 | ScaleRTL† APR | VerilogCoder APR | ACE-RTL APR |
|---|---|---|---|---|---|
| 68.0 | 73.0 | 73.8 | 93.6 | 94.2 | **95.5** |

这个 table 的价值是证明 **ACE-RTL 没有过拟合到 CVDP** —— 在 VerilogEval-Human-v2 这种更简单的 benchmark 上依然是 SoTA，说明 Generator 学到的是 transferable RTL capability。

参考：VerilogEval (original): https://arxiv.org/abs/2309.07544，VerilogEval v2: https://arxiv.org/abs/2408.11053

---

## 8. 对 Karpathy 视角的几个 personal observation

1. **"Generator–Reflector–Coordinator" 三件套是 RLHF 的 actor–critic–reward 的远房表亲**。Generator 是 actor，Reflector 是 critic（提供 value-like feedback），Coordinator 是 reward shaping + exploration controller（restart ≈ exploration noise injection）。这个 analogy 帮助 build intuition。

2. **Data scaling law 的应用很 classical**（reference [15] Hoffmann et al. Chinchilla, [19] Kaplan et al. scaling laws）。从 5M → 157K → 1.7M 的 pipeline 是 textbook-level data curation。1.7M 是 ScaleRTL 的"数十亿 reasoning tokens"之外的另一个量级参考点，但 ACE-RTL 更注重 **diversity 和 task coverage**（generation/modification/debugging 三类）。

3. **Parallel scaling 本质是 self-consistency 的 execution-level 版本**。Self-consistency 在 reasoning task 里 sample 多个 CoT 然后 majority vote；这里 sample 多个 trajectory 然后 filter by execution test。Execution test 是 perfect verifier，所以不需要 vote。

4. **Coordinator 的 restart-with-insight 机制让我想起 "hindsight experience replay"** —— 把 failed trajectory 的 insight 提取出来 conditioning 下一次 sample，本质是 sample-efficient off-policy learning 在 inference time 的化身。

5. **一个潜在的 limitation paper 没明说**：Coordinator 依赖 "consecutive iteration 同 assertion pattern" 作为 stagnation signal。但对于"每轮都 fail 但 fail 在不同地方"的情况，这个 trigger 可能太弱。一个改进方向是用 LLM-based progress detector。

6. **APR vs Pass@1 的区分很关键**。Agentic 方法的 Pass@1 ≈ APR（因为 agent 总是会重试到 succeed 或 exhausted budget），这导致 standalone LLM 的 Pass@1 和 agent 的 APR 严格说不是同一 metric。Paper 引入 APR 是 honest 的做法。

---

## 9. 几个值得追的 future direction

1. **Tool integration 扩展**：目前只用 iverilog。加入 waveform analyzer（VerilogCoder 风格）、formal verifier（如 JasperGold）、synthesis tool（DC）反馈，能 capture 更多 error type。
2. **Multi-modal spec**：hardware spec 经常是 waveform diagram + 文字描述。Reflector 可以处理 image input。
3. **Testbench generation**：paper 明说 "leave testbench generation to future work"。如果 agent 能自己写 testbench，就能在没有 golden test 的 real design 上工作。
4. **Coordinator 的 LLM 化**：目前 Coordinator 用 scripts + Claude4-Sonnet 的混合，可以更彻底地 LLM 化，做 meta-reasoning about debugging strategy。

---

## 总结性 intuition

ACE-RTL 的核心 message 我觉得可以浓缩成一句：

> **在 inference-time agentic loop 中，把"领域 pattern memory"放在 Generator（SFT 学到），把"通用 reasoning & memory management"放在 Reflector + Coordinator（frontier LLM 提供），两者通过 structured context 通信，比试图把两种能力塞进同一个 model 更 sample-efficient 且更 effective。**

这其实是 Mixture-of-Experts 的 agentic 版本 —— 不是参数级 MoE，而是 capability-level MoE。对 hardware design 这种"idiom-heavy + reasoning-heavy"的 domain，这个 decomposition 比单一 model 更 work。

paper 链接（CVDP）：https://arxiv.org/abs/2506.14074  
ScaleRTL 链接：https://arxiv.org/abs/2506.05566  
VerilogCoder 链接：https://doi.org/10.1609/aaai.v39i1.32007  
MAGE 链接：https://arxiv.org/abs/2412.07822  
CraftRTL 链接：https://arxiv.org/abs/2409.12993  
Icarus Verilog: https://github.com/steveicarus/iverilog  
vLLM: https://arxiv.org/abs/2309.06180  
Qwen2.5-Coder: https://arxiv.org/abs/2409.12186  
VerilogEval v2: https://arxiv.org/abs/2408.11053  
RTL-Repo: https://arxiv.org/abs/2405.17378  
Scaling laws (Kaplan): https://arxiv.org/abs/2001.08361  
Chinchilla (Hoffmann): https://arxiv.org/abs/2203.15556  
Self-Consistency (Wang et al.): https://arxiv.org/abs/2203.11171  
AlphaCode: https://arxiv.org/abs/2203.07814
