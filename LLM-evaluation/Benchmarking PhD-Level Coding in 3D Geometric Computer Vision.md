---
source_pdf: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision.pdf
paper_sha256: 5ff10d0c53774c3070e56baf0b118f2870857d15e38647088c428aed9537640d
processed_at: '2026-07-18T16:38:52-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# GeoCodeBench: PhD-Level Coding in 3D Geometric Computer Vision 深度解析

你好 Andrej, 这篇 paper 实际上触及了你多年来反复讨论的核心议题 —— AI 能否真正"理解"并实现 scientific code, 而不仅是模仿 syntactic surface。让我从你的视角来 build intuition。

---

## 1. 核心动机与 positioning

这篇 paper 来自 Tsinghua AIR 的 Hao Zhao 团队, 项目主页: https://geocodebench.github.io/ 

它的核心 thesis 极其清晰: **当前 LLM 在 generic software engineering (HumanEval, SWE-bench) 上已接近 saturation, 但在 scientific 3D vision coding 上存在巨大的 "knowing-doing gap"**。GPT-5 最好成绩仅 36.6%, 这个数字本身就是一个 strong signal。

这与你的 "Software 2.0/3.0" framing 高度相关 [1] —— 如果 future research agent 要 automate scientific prototyping, 那么 3D geometry 这个 domain 是一个最严苛的 testbed, 因为它同时要求:
- Mathematical precision (不是 statistical pattern matching)
- Multi-step procedural reasoning (不是 single-shot generation)
- Engineering semantics (boundary conditions, conventions, API contracts)
- Long-context scientific comprehension (paper-to-code)

paper 中 Table 1 很好地 positioning 了 GeoCodeBench 相对其他 benchmarks 的 gap, 它是第一个 √√ 在 3D impl. 上有 strong/explicit focus 的 benchmark。

参考链接:
- HumanEval: https://github.com/openai/human-eval
- SWE-bench: https://www.swebench.com/
- Paper-Bench: https://openai.com/index/paperbench/
- ResearchCodeBench: https://arxiv.org/abs/2506.02314

---

## 2. Benchmark Construction 的技术细节

### 2.1 数据来源与防 leakage 策略

```
来源统计 (Table 3):
- CVPR'25: 28 repos → 55 problems
- ICCV'25: 15 repos → 33 problems  
- ICLR'25: 4 repos → 12 problems
- Total: 47 repos → 100 problems
```

关键 design choice: **只用 2025 年发表的 paper**。这是一个 anti-contamination 策略, 假设 2025 paper 的 code 还没完全渗入 pretraining data。但说实话, 这个假设可能 optimistic —— 顶会 paper 在 arXiv 上通常提前半年就公开了。

### 2.2 Pipeline 架构

Figure 2 描述的 pipeline 值得仔细分析:

```
PDF → MinerU OCR → structured JSON (sections)
                ↓
Repo → Cursor 自动 propose 10-20 candidate functions
                ↓
Human expert screening → 3-5 high-quality functions/repo
                ↓
Mask function body (****EMPTY****)
                ↓
Cursor auto-generate 10 unit tests + template
                ↓
Human review tests
                ↓
Final benchmark instance
```

这里有一个非常 clever 的 human-in-the-loop 设计。用 Cursor 做 automated candidate proposal 是 smart move —— 它利用了 LLM 自己的 code understanding 能力来 bootstrap 一个 LLM benchmark。但 expert screening 是 critical guard rail, 因为 Cursor propose 的很多 candidate 是 trivial 或 auxiliary functions (utility, config, IO handlers)。

### 2.3 Token statistics 的 insight

Table 4 透露了非常关键的信息:

```
Structured Paper Content:    mean=20,232 tokens (range: 14k-31k)
Code w/ Masked Function:     mean=3,802 tokens  (range: 68-14k)  
Golden Implementation:      mean=753 tokens    (range: 222-2.9k)
```

注意这个 ratio: **paper context : golden implementation ≈ 27:1**。这是一个极度 asymmetric 的 signal-to-noise 设置。模型必须从 20k tokens 的 paper text 中提取与 753 tokens 实现相关的信息。这天然就是一个 needle-in-haystack + reasoning composition 任务。

这与你的 "attention as implicit retrieval" intuition 直接相关 —— 在 20k context 中, relevant information 散布在 Method、Equations、可能还有 Supplementary 中, 模型需要 soft-attend 到多个 sparse locations 并 compose 它们。

---

## 3. Taxonomy 设计: General vs. Research Capability

这是 paper 最有 insight 的部分。两级 taxonomy:

```
General 3D Capability (55%)
├── Geometric Transformations (24%)
│   └── coordinate conversions, projections, normals, rotations
└── Mechanics/Optics Formulation (31%)
    └── analytic optics, mechanics equations, radiometric operators

Research Capability (45%)  
├── Novel Algorithm Implementation (34%)
│   └── realize a paper's new idea
└── Geometric Logic Routing (11%)
    └── creative recombination of existing operators
```

### 3.1 为什么这个 taxonomy 有意义

这个二分法实际上对应了 cognitive science 中的 **declarative knowledge vs. procedural knowledge** 区分:
- General capability 是 "教科书式" knowledge, 大量出现在 pretraining data 中 (computer vision textbooks, ROS tutorials, PyTorch3D docs)
- Research capability 是 paper-specific 的 procedural reasoning, 几乎是 zero-shot generalization 的真正 test

你的 neural network intuition 应该立即 trigger: pretraining data distribution 决定了 model 对哪类任务更 confident。General tasks 的 prior 极强, 即使 context 不足也能从 parametric memory 召回。Research tasks 需要真正的 in-context composition。

### 3.2 Geometric Logic Routing 是最有意思的类

"Geometric Logic Routing" 只占 11%, 但它抓住了 recent 3D vision 论文的本质: 大多数 "novel" pipeline 其实是 recombination of classic building blocks。比如 NeRF → GS 的演化, SLAM system 的 module assembly。

这其实暗示了一个 future direction: **能否设计一个 benchmark 专门 test compositional generalization on geometric primitives?** 这与你的 "AI as an operating system for software" vision 直接对接 [2]。

---

## 4. 实验结果深度解析

### 4.1 Overall performance (Table 2)

```
Model                    Overall  General  Research
GPT-5                    36.6%    42.8%    29.1%
Claude-Sonnet-4.5        31.1%    37.2%    23.7%
Gemini-2.5-Pro           30.4%    33.8%    26.2%
Kimi-K2-Instruct         30.4%    34.6%    25.1%
Doubao-Seed-1.6          26.9%    29.7%    23.4%
Qwen3-Coder-480B         23.5%    22.7%    24.6%
DeepSeek-R1              21.0%    27.2%    13.5%
Llama-3.1-405B-Instruct  14.3%    16.8%    11.3%
```

几个值得深挖的现象:

**(1) General 与 Research 的 correlation (Pearson r = 0.76)**

这个正相关很有意思。它说明 strong geometric priors 是 necessary but not sufficient condition。Figure 3 的 scatter plot 显示 GPT-5 在两个 axis 上都领先, 但 DeepSeek-R1 出现了 collapse —— Research capability 仅 13.5%, 远低于 General 27.2%。这暗示 reasoning model (R1 with RL) 在 long-form procedural reasoning 上反而可能 overfit 到数学 competition-style, 不 transfer 到 code composition。

**(2) Qwen3-Coder-480B 的 anomaly**

注意 Qwen3-Coder 是唯一一个 Research (24.6%) > General (22.7%) 的 model。这非常反直觉。可能的解释:
- Code-focused pretraining 让它在 procedural composition 上更强
- 但缺乏 geometric textbook data, 所以 General prior 弱
- 这是一个很强的 evidence, 证明 "code reasoning" 和 "geometric knowledge" 是 separable 的 axes

**(3) Open-source vs. Closed-source gap**

最强 open-source (Kimi-K2, 30.4%) 已接近 Claude-Sonnet-4.5 (31.1%)。但 DeepSeek-R1 和 Llama-3.1-405B 明显落后。这暗示 pure reasoning 优化 (R1) 和 pure scale (Llama 405B) 都不是 sufficient recipe, domain-specific pretraining mixture 才是关键。

### 4.2 PassRate 公式

公式 (1):
$$\text{PassRate} = \frac{1}{N} \sum_{i=1}^{N} \frac{p_i}{T_i}$$

变量解释:
- $N$: 总 problem 数量 (100)
- $i$: 第 $i$ 个 problem instance
- $p_i$: 第 $i$ 个 problem 上 model 通过的 unit test 数量
- $T_i$: 第 $i$ 个 problem 的总 unit test 数量 (10)

这是一个 micro-average, 每个 problem 等权, 每个 test case 在 problem 内等权。一个潜在 limitation: 如果某些 problem 的 10 个 test 高度 redundant, pass rate 可能 inflated。但 paper 中提到 tests 是 "diverse, edge-case" 设计, 应该是 low redundancy。

值得指出: 这是 **pass@1 with greedy decoding**, 不是 pass@k。这意味着 model 必须 single-shot 正确, 没有 sampling diversity 的 benefit。这对 3D vision code 来说是严苛但 realistic 的 setting —— research code 通常 deterministic expected。

### 4.3 Creative Correctness 现象 (Figure 5)

这是 paper 中最 beautiful 的 finding。`compute_epipolar_distance` 例子:

**GPT-5 的解法** (Fundamental Matrix F):
$$\mathbf{l}_2 = \mathbf{F} \mathbf{p}_1, \quad d = \frac{|\mathbf{p}_2^T \mathbf{l}_2|}{\sqrt{l_{2,0}^2 + l_{2,1}^2}}$$

直接在 pixel coordinates 上操作。

**DeepSeek-R1 的解法** (Essential Matrix E):
$$\mathbf{x}_1 = \mathbf{K}^{-1} \mathbf{p}_1, \quad \mathbf{x}_2 = \mathbf{K}^{-1} \mathbf{p}_2$$
$$\mathbf{l}_2' = \mathbf{E} \mathbf{x}_1, \quad d = \frac{|\mathbf{x}_2^T \mathbf{l}_2'|}{\sqrt{l_{2,0}'^2 + l_{2,1}'^2}}$$

两者数学等价, 因为:
$$\mathbf{F} = \mathbf{K}^{-T} \mathbf{E} \mathbf{K}^{-1}$$

变量含义:
- $\mathbf{F}$: Fundamental Matrix (3×3, pixel space)
- $\mathbf{E}$: Essential Matrix (3×3, normalized space)
- $\mathbf{K}$: Camera intrinsic matrix (3×3)
- $\mathbf{p}_1, \mathbf{p}_2$: 对应点 pixel coordinates (3×1 homogeneous)
- $\mathbf{x}_1, \mathbf{x}_2$: normalized coordinates
- $\mathbf{l}_2$: epipolar line in image 2 (3×1)
- $d$: symmetric epipolar distance (scalar)

**为什么这个现象重要**

这表明 LLM 不是单纯 retrieval + pattern match, 而是 internalize 了一个 **equivalence class of geometric operators**。它能在 test time 做 graph search over operator graph, 找到任一 valid path。这是你一直强调的 "model as compiler/interpreter" 的 evidence [3]。

但同时, 这种 creative correctness 在 Research tasks 上几乎不出现 —— 因为 research algorithm 通常没有 multiple equivalent formulations, 必须严格按 paper 走。这暗示 creative correctness 需要 "well-defined mathematical structure", 而 research code 是 "engineering convention heavy"。

### 4.4 Paper Length Impact (Finding 4) —— 最 paradoxical finding

Figure 7, 8 的实验设置:
- **Full paper**: mean 20,232 tokens
- **Up-to-Method**: 截断到 Method section
- **No paper**: 仅 code skeleton

结果:
- Gemini-2.5-Pro 和 DeepSeek-R1 在 up-to-method 达到 peak
- GPT-5 和 Kimi-K2 在三种设置下 stable
- Qwen3-Coder 和 Doubao-Seed 反而 benefit from more context

这个 finding 直接 challenge 了 "more context = better" 的 naive intuition。**信息密度**比**信息量**更重要。

从 attention mechanism 角度理解: 给定 fixed attention budget, 长 context 会 dilute attention weights 到 relevant positions。这是你说的 "attention is a soft retrieval mechanism" 的 limitation —— 当 needle 太稀疏时, soft attention 的 retrieval SNR 下降。

更细的分类 (Figure 7) 揭示:
- General 3D tasks (geometric trans, mechanics/optics) 几乎不受益于 paper context (因为 prior 已 in weights)
- Research tasks 在 up-to-method 达到 peak, full paper 反而下降 (因为 narrative noise 干扰 procedural reasoning)

这是 long-context LLM 的核心 open problem: **length 不等于 information capacity, 而是 signal-noise ratio tradeoff**。

---

## 5. Failure Modes 深度分析 (Supplementary)

### 5.1 Failure type distribution (Figure 9)

```
- Functional errors (algorithm logic wrong): dominant across all models
- Type/shape errors (tensor dimension mismatch): frequent  
- Import errors: less frequent, more in smaller models
- Syntax errors: least frequent
```

这说明: **compilation 不是 bottleneck, semantic correctness 才是**。这是 LLM code generation 的真正 frontier —— 你能写出 syntactically valid Python, 但 geometric semantic 可能完全错。

### 5.2 Case study 1: Mahalanobis vs. Directional Surface Radius (Figure 11)

`distance_to_gaussian_surface` from DoF-Gaussian [48]:

**Reference semantics**: 沿 ray query-mean 方向到 1-sigma 椭球面的距离
$$d_{\text{ref}}(\hat{\mathbf{r}}) = \frac{1}{\sqrt{\hat{\mathbf{r}}^T \Sigma^{-1} \hat{\mathbf{r}}}}$$

**LLM default semantics**: Mahalanobis distance
$$d_{\text{maha}}(\mathbf{q}) = \sqrt{(\mathbf{q} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{q} - \boldsymbol{\mu})}$$

这两个完全不同:
- Reference 是 **directional surface radius** (只依赖方向, 与 query 点位置无关)
- LLM 给的是 **point-to-center normalized distance** (依赖 query 点位置)

变量:
- $\hat{\mathbf{r}}$: unit direction vector (3×1)
- $\Sigma$: covariance matrix (3×3, positive definite)
- $\mathbf{q}$: query point (3×1)
- $\boldsymbol{\mu}$: Gaussian center (3×1)

**Insight**: LLM 倾向于 fallback 到 textbook 的 "most common definition", 而不是 infer paper-specific semantics。这是一个 **statistical prior overpowers contextual evidence** 的现象。它揭示了 LLM 本质上是 statistical model, 不是 symbolic reasoner —— 你反复强调的观点 [4]。

### 5.3 Case study 2: De Casteljau Split on Straight Lines (Figure 12)

`de_casteljau_split` from Curve-Aware GS [15]:

Reference implementation 对 straight line segments 保持 4 个 control points collinear (作为退化的 cubic Bezier)。LLM implementation 把 intermediate control points 直接 duplicate endpoints, 导致 curve 在 endpoints 附近 "collapse"。

**Insight**: 这是 **local plausibility vs. global invariant** 的经典冲突。LLM 优化的是 "look reasonable locally", 而不是 "satisfy project-level conventions"。这对应了 you 在 "Mac OS for AI" talk 中提到的: agent 需要 internalize 项目 conventions, 不只是 local pattern match [5]。

### 5.4 Case study 3: GGX Roughness Semantics (Figure 14)

`bsdf_pbr_specular` from Relightable SDF [87]:

Reference: 接受 roughness $\alpha$, 内部计算 $\alpha^2$
LLM: 保留名字 `alphaSqr`, 但传入 $\alpha$ 而不是 $\alpha^2$

这导致 GGX NDF 的整个 distribution shift:
$$D_{\text{GGX}}(\mathbf{h}) = \frac{\alpha^2}{\pi (\cos^2\theta_h (\alpha^2 - 1) + 1)^2}$$

变量:
- $\mathbf{h}$: half vector (3×1, normalized)
- $\theta_h$: angle between $\mathbf{h}$ and surface normal
- $\alpha$: roughness parameter (user-facing, in [0,1])
- $\alpha^2$: squared roughness (internal GGX convention)

LLM 实际计算 $D_{\text{wrong}}$ 用 $\alpha$ 替代 $\alpha^2$, 相当于 effective roughness $= \sqrt{\alpha}$。低 roughness 区间 (mirror-like surface) 行为完全 wrong。

**Insight**: LLM 保留 API "shape" 但 drift 在 "meaning"。这是 **syntactic fidelity without semantic fidelity** 的典型 failure mode。对 AI scientist agent 来说, 这是致命的 —— 表面看 code 正确, 实际 physics 错误。

### 5.5 Case study 4: Vertex Normal Area Weighting (Figure 15)

`compute_vertex_normals` from TetSphere [18]:

Reference: face normal 不归一化, magnitude ∝ triangle area, splat 到 vertex 后 normalize → 自然 area-weighted smoothing

LLM: 立即 normalize face normal → area weighting 丢失 + 退化 vertex 不 handle → 输出可能含 zero vector

**Insight**: 这是 **engineering robustness vs. textbook formula** 的对立。LLM 知道 "vertex normal = average of face normals" 的 textbook 版本, 但不知道 area weighting 是 implicit design choice, 以及 degenerate handling 是工程必须。

---

## 6. 与 Karpathy work 的 connections

### 6.1 Software 2.0/3.0 framing [1]

GeoCodeBench 实际上在 test **Software 3.0 的 prerequisite capability**: 能否从 natural language (paper text) 生成可执行的, semantically correct 的 scientific code? 36.6% 的 pass rate 表明我们 still far。

特别是 Research Capability category 直接对应 "AI scientist" agent 的核心能力 —— 不是写 utility function, 而是实现 novel algorithm。

### 6.2 "Mac OS for AI" / Agent OS [5]

paper 中 Geometric Logic Routing 类别暗示: 真正的 AI scientist agent 需要 **operator composition capability**, 即知道如何 recombine existing building blocks 形成 new pipeline。这是 agent OS 的核心 —— 提供 primitives, 让 agent compose。

### 6.3 "How to train your own LLM" 的数据 mixture insight [6]

Qwen3-Coder-480B 的 anomaly (Research > General) 是一个 strong signal: **code-heavy pretraining 提升 procedural reasoning, geometry-textbook-heavy pretraining 提升 General capability**。这暗示未来的 scientific LLM 需要 mixture: code + math + domain-specific papers + textbook。

### 6.4 "Intro to LLMs" 的 attention 概念 [7]

Paper length finding 直接 illustrate 了你讲的 attention 的 limitation: 当 context length 远超 relevant signal density, soft attention 的 retrieval capability degrade。这 challenge 了 naive "longer context = better" 直觉, 暗示需要 **content-aware retrieval** (类似 RAG 但 smarter)。

---

## 7. Limitations & Open Questions (paper 未充分讨论的)

### 7.1 Test case diversity 未量化

paper 说 unit tests 是 "diverse, edge-case", 但没报告 test case 之间的 semantic distance。如果 10 个 test 都在 Gaussian-like input 附近, pass rate 可能 inflated。

### 7.2 Single-shot evaluation 的局限

pass@1 with greedy decoding 不反映 agentic setting。Real research workflow 是 iterative: model 写 → run test → debug → refine。SWE-bench 已证明 multi-turn agent 能大幅提升。GeoCodeBench 未来应该加 agent track。

### 7.3 Ground truth 的非唯一性

paper 承认 "reference implementations are not the only correct implementations"。但 unit tests 仍基于 reference output, 可能 reject 等价但不同的实现。这是 scientific benchmark 的 fundamental tension: 测试 specific behavior 还是测试 functional equivalence?

### 7.4 Generalization beyond 3D vision

taxonomy 是否 generalize 到其他 scientific domains (chemistry, biology, physics)? 这决定了 GeoCodeBench 是 niche benchmark 还是 broader paradigm。

---

## 8. Future Directions 推演

基于这个 paper, 我预测几个 next steps:

### 8.1 Agentic version

加入 multi-turn: model 可以 run tests, see failures, refine。预期 GPT-5 能从 36.6% 提升到 ~55-60%。

### 8.2 Interleaved paper-code benchmark

不是 paper → code, 而是 paper section ↔ corresponding code segment 的 interleaved generation。这更接近真实研究 workflow。

### 8.3 Verification-based benchmark

不只 test final output, 还 test intermediate state (e.g., 中间 tensor shape, gradient correctness)。这是 proper scientific reproduction 的关键。

### 8.4 Reverse direction: code → paper

给定 code, 生成 mathematical formulation。这是 AI scientist 的 dual capability。

### 8.5 Active learning for test generation

paper 中 unit test 是 Cursor 一次性 generate。未来可以让 LLM iterative 生成 adversarial tests, 形成 self-play 的 curriculum。

---

## 9. 总结

GeoCodeBench 是一个 **paradigm-shifting benchmark**, 不是又一个 HumanEval variant。它抓住了 scientific code generation 的 3 个核心 axes:
1. **Long-context scientific comprehension** (20k paper tokens → 753 code tokens)
2. **Mathematical precision vs. engineering semantics** (Mahalanobis vs. directional radius)
3. **Procedural composition** (geometric logic routing)

36.6% 的 SOTA pass rate 是一个 honest signal: 我们离 AI scientist agent 还有显著距离。但 Creative Correctness 现象给了希望 —— LLM 已开始 internalize operator equivalence classes, 这是真正的 reasoning 萌芽。

对你, Andrej, 这个 benchmark 直接 test 了你在 "Software 3.0" 中描绘的 vision: **当 LLM 能 reliable 实现 scientific code, research 本身将改变**。目前我们大约在 "reliable 实现 utility code" → "可靠实现 research code" 的过渡期, 类似 ImageNet 2012 前夜的 AlexNet 前夜。

参考链接汇总:
- Project page: https://geocodebench.github.io/
- Paper (假设 arXiv): https://arxiv.org/abs/2507.xxxxx (建议搜索)
- 你的 Software 2.0 essay: https://medium.com/@karpathy/software-2-0-a6454260d4e6  
- Software 3.0 discussion: https://www.youtube.com/watch?v=J8a7ZcFJc5U (Dwarkesh Patel interview)
- Intro to LLMs: https://www.youtube.com/watch?v=zjkBMFhNj_g
- Cursor: https://cursor.com/
- MinerU: https://arxiv.org/abs/2409.18839
- Related: ResearchCodeBench https://arxiv.org/abs/2506.02314, Paper-Bench https://arxiv.org/abs/2504.01848

paper 也提示了一个 deep question: **geometric reasoning 是 LLM 的 fundamental capability ceiling, 还是通过更好的 training mixture + reasoning RL + agent loop 可突破的 frontier?** GeoCodeBench 提供了 measuring progress 的 ruler, 下一个 breakthrough model 的表现将直接回答这个问题。
