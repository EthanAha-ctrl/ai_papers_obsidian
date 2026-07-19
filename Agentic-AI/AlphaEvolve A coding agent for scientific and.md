---
source_pdf: AlphaEvolve A coding agent for scientific and.pdf
paper_sha256: f092c8cbd65da89951ee6496374da9a54963f9035758c25ac6e4625f506df9ad
processed_at: '2026-07-18T07:50:27-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AlphaEvolve 深度解析

Andrej, 这篇 paper 我觉得是 FunSearch 的"成年版", 信号很强, 因为它把 LLM-guided evolution 从 toy domain 真正推到了 production-grade 的科学发现 + 工程优化。下面我从 intuition 开始一层层剥开。

---

## 1. Core Intuition: 为什么 Evolution + LLM 是天然配对

经典的 genetic programming 用手写的 mutation/crossover operator, 在 symbolic space 上盲搜, operator 设计本身就是 bottleneck。而 LLM 本质上是一个 "diff 操作的 prior"——它见过海量代码 diff、看过 commit history、知道"什么样的修改能修 bug、什么样的修改能加速", 这就是一个**learned mutation operator**。

AlphaEvolve 的核心 move 就是: **把 evolution 的 mutation/crossover 整个外包给 frontier LLM, 而 fitness 完全交给 deterministic code execution**。后者是关键, 它绕开了 LLM hallucination——LLM 可以瞎说, 但 evaluate 是 ground truth, 不通过就丢掉。这就把 "LLM 不可靠" 这个最大问题消化掉了。

intuition 上, 可以这样想: AlphaEvolve = MAP-Elites (quality-diversity) + island model (diversity maintenance) + LLM as mutation operator + machine-graded fitness。它是 test-time compute scaling 的一种极致形式, 把 LLM 的 single-shot capability 通过 evolution 撑开成持续数百小时的搜索。

参考: [FunSearch paper (Romera-Paredes et al., 2023)](https://www.nature.com/articles/s41586-023-06924-6), [MAP-Elites (Mouret & Clune, 2015)](https://arxiv.org/abs/1504.04909)。

---

## 2. 系统架构拆解

整条 pipeline 是一个 asyncio-based asynchronous DAG, 优化的是 throughput 而非 single-computation latency。节点包括 controller / LLM sampler / evaluation node, 互相 block on dependency。这是工程上很重要的一点——因为 evolution 是 iterative, 你必须 maximize ideas/sec, 而不是单次评估快。

### 2.1 Task specification

用户必须提供两样东西:

1. **Evaluation function h**: `solution → Dict[str, scalar]`, 所有 metrics 最大化。可以是简单的 "生成 graph → check property → return size", 也可以是 "evolve search algorithm → run 1000s → return best score found"。
2. **Code skeleton with EVOLVE-BLOCK markers**: 在源码里用 `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` 圈出要 evolve 的块, 其他部分作为 skeleton 调用这些块。初始实现可以是 `return constant` 的 placeholder。

这里有一个非常重要的 abstraction choice: 同一个问题可以不同方式 encode。比如要找一个对称 graph, 可以直接 evolve graph 本身 (string representation), 可以 evolve constructor function (FunSearch 的做法, 对对称解友好), 可以 evolve custom search algorithm (AlphaEvolve 数学部分的主打做法), 甚至 co-evolve search + intermediate solution。Paper 在 §6 discussion 里特别强调这点——不同 problem 适合不同 abstraction, 对称性强的用 constructor function (更简洁), 非对称的用 custom search algorithm。

### 2.2 Prompt sampling

Prompt 包含:
- 多个之前发现的 solution, 从 program database 采样
- system instruction on how to propose changes
- **explicit context**: 问题相关的人写说明、方程、代码片段、PDF (literature)
- **stochastic formatting**: 模板占位符 + 概率分布 → prompt diversity
- **rendered evaluation results**: program + execution result + scores
- **meta prompt evolution**: LLM 自己 propose prompt instructions, 在单独的 database 里 co-evolve (类似 [PromptBreeder](https://arxiv.org/abs/2309.16797))

这个 rich context 是和 FunSearch 最大区别之一——FunSearch 只把 previous solutions 塞进 prompt。

### 2.3 Creative generation: LLM ensemble + diff format

**Model ensemble**: Gemini 2.0 Flash (高 throughput, 多采样) + Gemini 2.0 Pro (高质量, 偶尔突破)。这是经典的 explore-exploit 在 LLM 层面落地。

**Diff 输出格式**: 用 `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` 的 git-merge-style format。这点很关键——evolve 整个 codebase 时, 必须精准定位修改点, 不能让 LLM 重写整个 file (会丢掉已有的好结构)。但对短代码也可以切到 full-rewrite mode。

intuition: diff format 是把 "incremental programming" 显式 encode 到 LLM 输出空间里。LLM 在 diff 上的 prior 是非常强的——它训练数据里看过 billions 行 commit diff, 知道小步迭代长什么样。

### 2.4 Evaluation: cascade + LLM feedback + parallelism

- **Evaluation cascade (hypothesis testing)**: 测试集按难度分级, 过不了简单的就别跑难的, 省算力。新程序先跑 small-scale sanity check。
- **LLM-generated feedback**: 对于 "simplicity" 这种难以 formalize 的指标, 单独用 LLM call 给分, 加进 score dict。
- **Parallelized evaluation**: 单次 evaluation 可以花 100 compute-hours, 但 embarrassingly parallel (多 random seed), 通过 async cluster 调用, 把 wall-clock 时间压下来。

**Multi-objective optimization**: AlphaEvolve 同时优化多个 metric。Paper 里给了一个很 insightful 的观察——即便只关心一个 metric, 加上 auxiliary metric 也常常提升主 metric, 因为不同 metric 下的 high-performers 结构不同, 喂回 prompt 增加 diversity。

### 2.5 Evolution: program database = MAP-Elites + islands

Database 平衡 exploration 和 exploitation。Inspired by:
- **MAP-Elites**: 按 behavior descriptor 分 cell, 每 cell 保留 elite。鼓励 quality-diversity。
- **Island model**: 多个 sub-population 独立 evolve, 偶尔 migrate。维持 genotype diversity。

这俩组合是个 hybrid: 用 MAP-Elites 的 cell structure 保 quality-diversity, 用 island 的隔离防 premature convergence。这是和 [island-based GA (Tanese, 1989)](https://deepblue.lib.umich.edu/handle/2027.42/29528) 的 connection。

### 2.6 Distributed pipeline

asyncio, controller + samplers + evaluators concurrent, 优化 throughput。这点对 scalability 至关重要——因为 evolution 需要数百-thousands 的 LLM samples + evaluation, 串行跑根本不可能。

---

## 3. 关键数学公式和变量

### 3.1 Tensor rank decomposition (matrix multiplication)

矩阵乘法 $C = AB$, $A \in \mathbb{F}^{m \times n}$, $B \in \mathbb{F}^{n \times p}$, 对应 multiplication tensor $\langle m, n, p \rangle \in \mathbb{F}^{m \times n \times p} \otimes \mathbb{F}^{n \times p \times m} \otimes \mathbb{F}^{p \times m \times n}$ (具体是三阶张量)。其 rank-$r$ decomposition:

$$T_{\langle m,n,p \rangle} = \sum_{i=1}^{r} u_i \otimes v_i \otimes w_i$$

变量解释:
- $r$ = tensor rank = 算法需要的 scalar multiplication 次数 (乘法是 bottleneck, 加法 cheap)
- $u_i \in \mathbb{F}^m$, $v_i \in \mathbb{F}^n$, $w_i \in \mathbb{F}^p$ (或对应维度, 取决于具体 ordering) = rank-1 component vectors
- $r$ 越小 → 算法越快

Strassen (1969) 给出 $\langle 2,2,2 \rangle$ rank-7 算法, 递归应用到 $\langle 4,4,4 \rangle$ 得到 rank $7^2 = 49$。[Fawzi et al. 2022 (AlphaTensor)](https://www.nature.com/articles/s41586-022-05172-4) 在 $\mathbb{F}_2$ 上找到 rank-47, 但 characteristic 0 (复数域) 56 年无人突破 49。

AlphaEvolve 找到 **rank-48 复数域**算法, 这是 paper 最 striking 的 result。在 Table 2 里看到 ⟨4,4,4⟩: 49 → 48, ⟨3,4,7⟩: 66 → 63, ⟨4,4,8⟩: 98 → 96 等共 14 项突破。

evaluation function 设计:
- target tensor $T_{\langle m,n,p \rangle}$ 固定
- decomposition 是 trainable complex tensor, 用 Adam-style 优化 reconstruction loss
- 关键 trick: round 到 nearest integer/half-integer 保 exactness, 同时加 discretization loss 鼓励 near-integral
- 评分: 最低达到的 rank + 达到该 rank 的 seed 比例 (signal for hill-climbing)

Figure 9a-c 展示了 AlphaEvolve 改了什么——它自己引入了:
- AdamW with weight decay (vs 原 Adam)
- smaller init scale (encourage low-rank)
- gradient noise injection (exploration)
- cyclical annealing of clip threshold (sinusoidal)
- **hallucination loss**: 随机替换部分值, 强迫 robustness
- **discretization loss** to half-integers (保 exact)
- cosine annealing of half-integer multiplier
- large value penalty (stability)
- 一堆 hyperparameter sweep

这些都是 LLM 自己"想出来"的 tricks——15 个 mutations 累积而成。这是 paper 里最 amazing 的图之一。

### 3.2 Kissing number (11 维)

定义: $d$ 维空间最多能放多少个不重叠单位球同时 tangent 到中心单位球, 记为 $\tau(d)$。$\tau(11)$ 之前 best known lower bound 是 592, AlphaEvolve 找到 593。

关键 lemma: 找一个 set $C \subset \mathbb{R}^d$ with $0 \notin C$ 满足:

$$\min_{x \neq y \in C} \|x - y\| \geq \max_{x \in C} \|x\|$$

变量解释:
- $x, y$ = $C$ 中点
- $\|x - y\|$ = 两点距离
- $\|x\|$ = 点到原点距离
- 条件含义: 任意两点距离 ≥ 最远点到原点距离

那么 $\{2x/\|x\| : x \in C\}$ 形成有效 kissing configuration (单位球中心在 $2x/\|x\|$, 都距原点 2, 互不重叠)。证明:
$$2\langle x, y \rangle \leq \|x\|^2 + \|y\|^2 - \max\{\|x\|^2, \|y\|^2\} = \min\{\|x\|^2, \|y\|^2\} \leq \|x\| \cdot \|y\|$$

所以 $\|\frac{2x}{\|x\|} - \frac{2y}{\|y\|}\| \geq 2$, 球不重叠。AlphaEvolve 找到了 593 个 11 维整数坐标点满足这条件。

参考: [Boyvalenkov et al. survey on kissing numbers](https://www.math.bas.bg/serdica/2012/2012-4.pdf), [Ganzhinov 2022](https://arxiv.org/abs/2207.08266)。

### 3.3 Autocorrelation inequalities

Autoconvolution:
$$f * f(t) := \int_{\mathbb{R}} f(t-x) f(x) \, dx$$

变量解释:
- $f$ = 非负函数 (在 $[-1/4, 1/4]$ 上)
- $t$ = shift 参数
- $f*f(t)$ = $f$ 与自己 shift $t$ 后的 overlap integral

$C_1$ 是最大常数使:
$$\max_{-1/2 \leq t \leq 1/2} f*f(t) \geq C_1 \left(\int_{-1/4}^{1/4} f(x) dx\right)^2$$

对所有非负 $f$ 成立。已知 $1.28 \leq C_1 \leq 1.5098$, AlphaEvolve 用 600-interval step function 找到 $C_1 \leq 1.5053$, 微微改进 upper bound。

这类问题用 step function construction, AlphaEvolve evolve 的是 search algorithm 来找 step function 的值。

### 3.4 Uncertainty principle

$$A(f) := \inf\{r > 0 : f(x) \geq 0 \text{ for all } |x| \geq r\}$$

$A(f)$ = $f$ "终于变正"的最小半径。$A(f)A(\hat{f}) \geq C_4$, $C_4$ 的 upper bound 之前是 0.3523 (来自 [Gonçalves, Silva, Steinerberger 2017](https://www.sciencedirect.com/science/article/pii/S0022247X17300907))。

AlphaEvolve 找 test function $f(x) = P(x) e^{-\pi x^2}$, 其中 $P(x) = \sum c_{4k} H_{4k}(x)$, $H_{4k}$ 是 Hermite polynomial。利用 Fourier 性质 $\widehat{H_n e^{-\pi x^2}} = i^n H_n e^{-\pi \xi^2}$, 对 even polynomial 得到 $A(f) = A(\hat{f}) = $ largest positive root of $P$。

AlphaEvolve 找到 $[c_0, c_1, c_2] \approx [0.32925, -0.01159, -8.9216 \times 10^{-5}]$, 给出 $C_4 \leq 0.3521$, 几乎 confirm [33] 的 construction 接近 optimal。

### 3.5 Erdős minimum overlap problem

$$\sup_{x \in [-2,2]} \int_{-1}^1 f(t) g(x+t) dt \geq C_5$$

变量解释:
- $f, g: [-1,1] \to [0,1]$, $f+g=1$, $\int f = 1$
- $g$ 可以理解为 $1-f$
- 这衡量两个 function 在 worst-case shift 下的最小 overlap

之前 $0.379005 \leq C_5 \leq 0.380927$, AlphaEvolve 改进到 $C_5 \leq 0.380924$。差别极小, 但确实是 new SOTA。

### 3.6 Sums and differences

$$C_6 \geq 1 + \frac{\log \frac{|U-U|}{|U+U|}}{\log(2 \max(U) + 1)}$$

变量解释:
- $U$ = 有限整数集, 包含 0
- $|U-U|$ = 差集大小, $|U+U|$ = 和集大小
- $\max(U)$ = $U$ 中最大元素
- 想要 $|U-U|$ 相对 $|U+U|$ 大, 同时 $\max(U)$ 小

AlphaEvolve 找到 size 2003 的 set 给 $C_6 \geq 1.1479$, size 54265 的 set 给 $C_6 \geq 1.1584$。

---

## 4. 工程 application 拆解

### 4.1 Google Borg 数据中心调度

问题: 在线 job scheduling, 形式化为 vector bin-packing with 2 维 (CPU, memory)。Machine = bin, job = item。Heuristic function 输入 (job CPU/mem 要求, machine CPU/mem 余量), 输出 priority score, Borg 把 job 分给 score 最高的 machine。

为什么用 AlphaEvolve 不用 deep RL: interpretability、debuggability、predictability、deployment simplicity。Mission-critical 系统必须能 audit。AlphaEvolve 输出是一段 Verilog/Python, 人能看懂。

Result: 0.7% fleet-wide compute 回收——在 Google 规模下是天文数字的算力。

### 4.2 Gemini kernel tiling heuristic

Pallas kernel for matrix multiplication on TPU。Tiling 把大矩阵乘切成小 subproblem, 平衡 compute 和 data movement。问题是: tile size $(b_m, b_k, b_n)$ 怎么随 input shape 变化选——这取决于 hardware 的 optimal shape、memory、fused ops、compiler intricacies, 手工 tune 几个月。

AlphaEvolve evolve 一个 heuristic function, 输入 input shape 输出 tile size。evaluate 用真实 TPU 跑 runtime。23% 平均 kernel speedup, 整体 Gemini training time 降 1%。从几个月工程变成几天自动实验。**这是 Gemini 通过 AlphaEvolve 优化自己的 training——self-improvement loop 出现了**。

### 4.3 TPU 算术电路 Verilog 优化

Optimize Verilog RTL of 一个 highly-optimized TPU arithmetic circuit in matrix multiply unit。目标: 减 area + power, 保 functional correctness。AlphaEvolve 找到一个简化——去掉 unnecessary bits。虽然是 downstream synthesis tool 也能抓到, 但 AlphaEvolve 在 RTL 阶段就介入了, 早期暴露问题。**首次 Gemini 通过 AlphaEvolve 直接贡献到 TPU 电路设计**。

### 4.4 XLA compiler IR 优化 FlashAttention

直接 optimize XLA 生成的 IR (中间表示), wrapping FlashAttention kernel + pre/post-processing。挑战:
- IR 是给 debugging 用的, 不是给手编辑的
- 已被 compiler 高度优化过

数值 correctness 通过 random input 对比 reference 检验, 最后人工 confirm 全输入正确。FlashAttention kernel 32% speedup, pre/post-processing 15% speedup。这意味着 AlphaEvolve 能进 compiler 工作流, 长期可能直接嵌入 compiler pipeline。

参考: [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135), [JAX](https://github.com/jax-ml/jax), [XLA](https://github.com/openxla/xla)。

---

## 5. Ablation 拆解

Figure 8 在两个 task (tensor decomposition 和 kissing number) 上做 ablation, 各项都 significant:

- **No evolution** (重复喂 initial program 给 LLM): 大幅退化, evolution 是骨架
- **No context in prompt**: 退化很多, rich context 重要
- **No meta prompt evolution**: 退化, LLM 自己写 prompt 也 contribute
- **No full-file evolution** (只 evolve loss function): 退化, full-file evolution 比 single-function 强很多
- **Small base LLM only** (不用 Pro): 退化, frontier model 质量不可替代

这五个 component 全是 additive 的, 没有 "silver bullet", 是 stack 整体强。

---

## 6. 与 FunSearch、AlphaTensor、AI Co-Scientist 对比

| 维度 | FunSearch | AlphaEvolve | AlphaTensor | AI Co-Scientist |
|------|-----------|-------------|-------------|-----------------|
| Evolve 单位 | 单个 Python 函数 | 整个 codebase, 任何语言 | RL policy over tensor actions | 自然语言 hypothesis |
| Code 规模 | 10-20 行 | 数百行 | N/A | N/A |
| 评估 | 快 (≤20min CPU) | 数小时, 可并行 on accelerator | exact tensor game | LLM 评估 |
| LLM samples | millions | thousands | N/A | LLM-heavy |
| LLM 模型 | small, code-only | frontier (Gemini Pro + Flash) | N/A | frontier |
| Context | 仅 previous solutions | rich (literature, code, meta-prompt) | task-specific | literature + ranking agents |
| 目标 | 单 metric | multi-objective | single (rank) | multi-criteria ranking |
| Hallucination 风险 | 低 (eval 是 truth) | 低 | 低 | 高 (LLM 评估) |

AlphaEvolve 最大的 conceptual move: 把 "code 是 solution" 升级成 "code 是 search algorithm / 是 solution 的描述 / 是 heuristic"。这种 abstraction flexibility 让它能 cover 远超 FunSearch 的领域。

和 [AI Co-Scientist (Gottweis et al., 2025)](https://arxiv.org/abs/2502.18864) 的对比很关键: AI Co-Scientist 用 natural language hypotheses + LLM evaluation, 高灵活但有 hallucination 风险, 不能长跑。AlphaEvolve 用 code + deterministic eval, 低灵活但能跑数千步 evolution。Paper 在 §6 提到未来 combine 两者: LLM 评估 high-level idea → 转入 code execution 评估 implementation。

---

## 7. 局限 + 未来方向

Paper 自己提的主要 limitation: **需要 automated evaluator**。自然科学里很多实验不能 simulate, 这就 out of scope。但 LLM-based evaluation 可以补充——只是 AlphaEvolve 没 optimize 这个方向。

我想到的其他 limitations (paper 没明说):
1. **冷启动成本**: 写一个 robust evaluate function 不简单, 尤其 multi-objective 时
2. **Codebase 必须足够 modular**: EVOLVE-BLOCK 边界要画对, 否则 LLM 改不动
3. **Compute 不便宜**: 每个 evaluation 100 compute-hours, 整个 evolution 跑下来是巨大 footprint
4. **Discovery 是 incremental**: 大部分是 tiny SOTA improvements (e.g., 0.380927 → 0.380924), 不是 paradigm-shifting。但这其实是诚实的——真正的科学突破本来就 rare
5. **Reward hacking 风险**: multi-objective 时 metric 之间 trade-off 可能被 LLM 利用漏洞

未来方向 paper 提的:
- **Distill AlphaEvolve-augmented performance 回 base LLM**, 用下一代 base model 升级 AlphaEvolve (recursive self-improvement, 但慢, 月级 feedback loop)
- **更多 environments with robust evaluation functions** (设置问题本身是 bottleneck)
- **和 AI Co-Scientist 类方法 combine**: LLM 评估 high-level → code execution 评估 low-level

---

## 8. 我的 intuition takeaway

几个我觉得最重要的 insight:

**Insight 1: Evolution 是 LLM test-time compute 的"长跑"形式。** Repeated sampling 是单次扩展, AlphaEvolve 是有 memory、有方向的 multi-step 扩展, 数百小时持续 hill-climb。这是 inference-time scaling 的一种结构化形态。

**Insight 2: Code 作为 hypothesis representation 是关键。** 自然语言 hypothesis 难 verify、难 compose; 代码可以执行、可以 compose、可以 diff。代码是科学发现的"formal language substrate"。

**Insight 3: Diff format 是 evolution 与 LLM 的接口。** LLM 对 commit diff 有 strong prior, 用 diff 作输出空间等于 free 利用这个 prior。这比让 LLM 重写整个 program 高效得多。

**Insight 4: Multi-objective 不仅是为了 multi-objective。** 多个 metric 即使只关心一个, 也通过 diversity injection 帮主 metric。这是个反直觉但重要的实证观察。

**Insight 5: AlphaEvolve 能 self-improve 自己的 infrastructure (Borg scheduling 释放 0.7% 算力, Gemini kernel 加速 23%)。** 这是最早期但真实的 recursive AI self-improvement 信号——虽然 feedback loop 是月级, 但 loop 闭合了。

**Insight 6: "Evolving search algorithm" 比 "evolving solution" 强。** Paper 反复强调这点。Search algorithm 比 solution 更 transferable、更 compressible、能利用 budget。这是从 instance-level search 到 meta-search 的跃迁。

**Insight 7: MAP-Elites + island 是 quality-diversity 的关键。** Evolution 最怕 premature convergence, 这俩组合维持 diversity, 让 LLM 看到不同 high-performer 风格, 产生不同 children。

---

## 9. 思考题 / 可能的延伸

1. 如果把 AlphaEvolve 用到神经网络 architecture search 上 (像 [EvoPrompting](https://papers.nips.cc/2023/file/...)), 但 evaluate 是真实 training loss 而非 proxy, 会怎样?
2. 把 AlphaEvolve 嵌入 PyTorch / JAX compiler, 作为 IR-level optimizer (像 §4.4), 让 compiler 自己 evolve 出 hardware-specific fused kernel, 这是不是 future of compiler design?
3. 数学发现的 abstraction choice: paper 说对称问题用 constructor, 非对称用 search algorithm。但很多 open math problem 不知道有没有对称性, 是否可以 co-evolve abstraction choice 本身?
4. AlphaEvolve 找到 4×4 复数 rank-48 但实数 rank 还是 49 不变——复数算法是否有 deep reason?AlphaEvolve 是真的"理解"了什么还是只是搜到了?
5. 如果给 AlphaEvolve 一个 theorem prover 作为 evaluator, 能否 evolve 出新的 proof tactics?这是 AlphaEvolve + Lean/Coq 的方向, 跟 [LeanDojo](https://arxiv.org/abs/2310.04705) 类工作结合。

---

## 关键 references

- [AlphaEvolve paper PDF (DeepMind)](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/doc/alphaevolve/AlphaEvolve_A_coding_agent_for_scientific_and_algorithmic_discovery.pdf)
- [FunSearch (Romera-Paredes et al., Nature 2023)](https://www.nature.com/articles/s41586-023-06924-6)
- [AlphaTensor (Fawzi et al., Nature 2022)](https://www.nature.com/articles/s41586-022-05172-4)
- [AI Co-Scientist (Gottweis et al., 2025)](https://arxiv.org/abs/2502.18864)
- [MAP-Elites (Mouret & Clune, 2015)](https://arxiv.org/abs/1504.04909)
- [PromptBreeder (Fernando et al., 2023)](https://arxiv.org/abs/2309.16797)
- [Strassen 1969 original](https://link.springer.com/article/10.1007/BF02165411)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [JAX](https://github.com/jax-ml/jax), [XLA](https://github.com/openxla/xla)
- [Erich's Packing Center (Friedman)](https://erich-friedman.github.io/packing/)
- [DeepMind blog: AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-coding-agent-for-scientific-and-algorithmic-discovery/)

---

总结一句: AlphaEvolve 是把 LLM 当作 "learned mutation operator"、把 code execution 当作 "deterministic fitness oracle"、用 MAP-Elites + island 维持 diversity、用 frontier LLM ensemble 平衡 explore-exploit, 在数小时级 evaluation budget 上跑数百步 evolution 的系统。它把 FunSearch 从玩具推到了 SOTA-breaking science 和 production infra。最让我 excited 的是它能在 Gemini 自己的训练 pipeline 里找到 23% kernel speedup——recursive self-improvement 闭环的最早信号。
