---
source_pdf: Dynamo Dynamic Skill-Tool Evolution for.pdf
paper_sha256: 208b6ed14257a68473a601ff076f346c6dd5862cffe689e0be877017f0977c22
processed_at: '2026-08-04T00:46:09-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Dynamo 用人话说

## 一句话概括

让一个 frozen VLM 看自己做对和做错的题目，然后自己总结出 reusable 的 reasoning skill 和 Python tool，存到一个 library 里，下次用。全程 zero gradient updates。

## 为什么这事儿重要

现在给 VLM 做 task adaptation 的主流路径很重：每个新 task family 都要人工 curate SFT data，或者搭一个 RL pipeline，10^4-10^5 rollouts，80-240 gradient updates。**compute 成本巨大，而且每个 task 都要来一遍**。

Dynamo 问了一个很 fundamental 的问题：**agent 能不能自己 inspect 自己的 behaviour，从 mistakes 里提取 reusable capability，而且不 update weights？**

答案是：在 visual reasoning 这个 domain，大部分 task adaptation 不需要 gradient updates，只需要让 agent 在 capability space 上做 search。compute 成本从几万 trajectories 降到几千 API calls，几个 order of magnitude 的 savings。

参考这个跟 Voyager (https://arxiv.org/abs/2305.16291) 的思路对比：Voyager 在 Minecraft 里让 LLM 积累 skill library，Dynamo 把这个 idea 搬到了 vision-language domain，还加了 tool generation。

## 核心机制：Evolution Loop

想象你教一个学生解题。你不给他 weights 更新（没法改他大脑结构），但你能让他：

1. **先做一批题**，收集 reasoning trace（包括做对和做错的）
2. **自我诊断**：看自己做对和做错的 case，找 root cause。关键是同时看 correct 和 incorrect——只看 correct 没 target，只看 error 没 boundary，contrast 才给方向。而且必须看 image，不能只看 reasoning trace——因为 "crop 对没对"、"label 清不清楚" 这种 perceptual 问题只能从 image 判断。
3. **生成候选 capability**：诊断出是 cognitive bottleneck（reasoning weak）还是 perceptual bottleneck（需要 transformed visual input），然后生成 M 个 candidate skill/tool combinations。比如 ChartQA 生成 "Edit-Then-Read" skill + 一个 PIL-based highlighting tool；HRBench4K 生成 "Coarse-to-Fine Zoom" skill + 一个递归 split-and-zoom tool。
4. **Validation guard**：每个 candidate 在 training set 上 evaluate，只有 strictly improve over current capability set 才 promote。这保证 monotonic training-set accuracy，防止 regression。

公式化表达（Eq 3）：

$$c^\star = \arg\max_{c \in \{c_1, \ldots, c_M\}} \mathrm{Acc}(f_{C \cup c}; \mathcal{D}_{\mathrm{train}})$$

变量含义：
- $c_m = (s_m, t_m)$：第 $m$ 个 candidate，$s_m$ 是 skill，$t_m$ 是 tool
- $f_{C \cup c}$：把 candidate 加进 capability set $C$ 之后的 agent
- $\mathrm{Acc}(\cdot; \mathcal{D}_{\mathrm{train}})$：在 training set 上算 accuracy
- Guard: $C \gets C \cup \{c^\star\}$ only if $\mathrm{Acc}(f_{C \cup c^\star}) > \mathrm{Acc}(f_C)$

intuition：这就是一个贪心的 coordinate ascent on capability space，每个 iteration 可能 promote 0 或 1 个 capability。跑 $N=3$ iterations 结束。

## Skills vs Tools 的分工

这是 paper 里最 clean 的 dichotomy：

**Skill = structured Markdown SOP**（cognitive bottleneck）
- 四个 field：When-to-Use, Strategy, Common Pitfalls, Worked Example
- 适合 ChartQA / MathVista 这类——image 信息都看得见，但 reasoning procedure 弱
- Inference 时用 BM25 retrieve top-3 skills, append 到 system prompt

**Tool = Python function ≤150 lines**（perceptual bottleneck）
- Input image paths, output processed images 或 extracted data
- 适合 V* / HRBench4K 这类——需要 zoom, crop, search, contrast adjustment 才能让 evidence legible
- 用 OpenCV, PIL, NumPy

Table 1 的 ablation 完美对应这个 dichotomy：
- ChartQA 上 Skill Only competitive（cognitive bottleneck）
- V* 上 Tool Only / Full dominate（perceptual bottleneck）

## Mastery Skill：Tool 的 "使用说明书"

这俩必须配对，tool 不能裸跑。Paper 里最 insight 的设计。

AnalyzerDecider 如果 action=both，Generator 会 emit 一个 skill + 一个 tool，这个 **paired skill 就是 tool 的 mastery SOP**：
- When-to-Use: tool 在哪些 input pattern helpful
- Common Pitfalls: tool 在哪些 pattern 会 hurt
- Strategy: 怎么 invoke，怎么 consume output

**Inference 时 tool invocation 由 retrieving paired skill 来 gate**。意思是 tool 不是无条件 expose，而是先 retrieve skill 判断 When-to-Use 是否 match。

这个设计的必要性在 Section B.4 的 ablation 里被验证：unpaired tool 虽然 executable 且 locally plausible，但 exposed without mastery skill 后在 subset 上 regressed（HRBench4K-cross 从 0.5625 掉到 0.4375）。

intuition：tool 是一个 function with applicability distribution，mastery skill 是在 approximate 这个 distribution。没有 gate 的 tool 就像一个 if statement 没写 condition——只要能跑就跑，结果瞎跑反而 hurt accuracy。Prior tool-creation methods（CREATOR https://arxiv.org/abs/2305.14318 , LATM https://arxiv.org/abs/2305.17126）都缺这层 gating。

## 最 clever 的 Case Study：Interleaved Visual Skill

Appendix D.3 那个 Case C 真的很 elegant。Agent 在 ChartQA bar chart 上第一次 crop 太激进，x-axis labels 被 cut off，答错了 year。AnalyzerDecider 诊断："crop too aggressive: target context lost"。

Generator emit 的不是单纯 text skill，是 **skill + 两张 reference images**：
- `crop_good.png`: 正确 crop，year labels (2016, 2015) 都在
- `crop_bad.png`: 错误 crop，year-label column 被 mask 掉

Skill 的 Strategy step 4 literally 写着：
> If your crop looks like `crop_bad.png` → re-think bounding box, expand ~30px, re-crop  
> If your crop looks like `crop_good.png` → proceed to read off values

这相当于把 few-shot demonstration 存进了 library。下次 retrieve 这条 skill 时，good/bad reference images 一起 retrieve。Agent 不用 re-discover 这个 failure mode，直接 load 这个 bundle。

这跟 in-context learning 有 deep connection：传统的 ICL 是把 examples 塞进 prompt，Dynamo 是把 examples 存进 persistent library，按需 retrieve。

## RL Comparison：为什么 training-free 能 match RL

Table 4 的数据：

**ChartQA (VTool-R1 protocol, Qwen2.5-VL-7B)**：
- Direct: 56.05
- RL: 75.18
- Dynamo: 75.06
- Dynamo + RL: 76.88

**V* (DeepEyes protocol, Qwen2.5-VL-7B)**：
- Direct: 79.06
- RL: 84.82
- Dynamo: 85.86 (beat RL!)
- Dynamo + RL: 92.67

**Cost**:
- RL: VTool-R1 ~20,480 trajectories + 80 gradient updates
- RL: DeepEyes ~122,880 trajectories + 240 gradient updates
- Dynamo: 几千 frozen-VLM API calls, zero gradient updates

intuition：RL 在 weight space 做 search，Dynamo 在 capability space 做 search。两者目标相同但 search space 不同，所以 compose additively——Dynamo + RL 在所有 cells 上都 beat RL alone 1-3 points。

这给一个 deep insight：**RL 学的是 when to take which action（policy），Dynamo 用 skill retrieval 来 approximate 这个 policy，但 explicit 且 interpretable**。RL 的 policy 是 implicit 在 weights 里，Dynamo 的 policy 是 explicit 在 Markdown + Python 里。

## Online Adaptation：应对 Distribution Shift

Real deployment 里 query distribution 会 shift。比如一会儿是 high-resolution perception，一会儿是 MathVista reasoning。

Dynamo 用一个 rolling-window 来 detect：

$$\hat{\phi}_t = \arg\max_\phi \sum_{i=t-W+1}^{t} \mathbf{1}[\mathrm{fam}(x_i) = \phi]$$

变量：
- $W$: rolling-window length
- $\mathrm{fam}(\cdot)$: query family classifier
- $\hat{\phi}_t$: step $t$ 时 dominant task family
- $\mathbf{1}[\cdot]$: indicator function

当 $\hat{\phi}_t \neq \hat{\phi}_{t-1}$（dominant family flip）时，触发下一次 evolution iteration on new sub-stream。

Figure 3 / Table 10 的结果非常 compelling：
- GPT-5.4 stress protocol: Direct 0.026 → Static 0.391 → Online 0.947 → Oracle 0.995
- Direct 几乎 collapse，Online adapt 接近 Oracle upper bound

需要一个 per-case correctness signal 来驱动——可以用 human-in-the-loop review, automated QA pipeline, 或 LLM verifier。Framework agnostic to feedback source。

## Mechanism Ablations：Rules Out Alternative Explanations

Paper 里最严谨的部分，每个 ablation 都排除一种 alternative explanation。

**Table 6 (HRBench4K, 700 cases)**：
- No Capability: 0.8557
- Generic Skill (hand-neutral prompt): 0.8486 (甚至 lower!)
- Evolved No-Tool Skill: 0.8529 (tied with no-capability)
- Evolved Tool Only: 0.8929
- Evolved Skill + Tool: 0.8986

结论：generic prompt injection 解释不了 gain，validated visual artifact 才是 active ingredient。不是 "more prompt text"，是 "validated image transformations that expose hard-to-read local evidence"。

**Table 8 (ChartQA polarity stress test)**：
- Correct-Only Diagnosis: degenerate, no failure cluster, loop 停在 generation 之前
- Error-Only Diagnosis: active 但 unstable，每 round 生成新的 broad overlay tool 但都不 promote（tie baseline selection score）

结论：**contrast between solved 和 unsolved cases 是 evolution 的 directional signal**。这跟 contrastive learning 的 intuition 相通。

**Table 7 (Text-only Diagnosis)**：
- Text-only candidate 被 reject，因为 proposed tool 没产生 visual artifact

结论：text reflection 能说 "answer wrong"，但不能说 "crop misaligned" 或 "labels illegible"。Visual diagnosis 是 materialize usable image-processing tool 的必要条件。

## 我的几个直觉

### 1. Capability space vs Weight space

Dynamo 本质上是在说：**VLM 的 task-specific weakness 不一定要通过改 weights 来 fix，很多时候通过给对 context 就能 fix**。Skill 是 structured context, tool 是 transformed visual context, mastery skill 是 gating context。

这跟 in-context learning 有 connection：ICL 是 temporary 的 context，Dynamo 是 persistent 的 context library。从 software 1.0 / 2.0 的框架看，weights 是 software 2.0（learned），skills/tools 是 software 1.0（written, 但这里 agent 自己 write）。Dynamo 在探索两者怎么 compose。

### 2. Contrastive Diagnosis 的深层意义

Correct + Incorrect contrast 是 evolution 的 directional signal，这跟很多 learning algorithm 都相通：
- Contrastive learning: positive-negative pairs 学 representation
- RL: advantage = reward - baseline
- Active learning: uncertainty sampling 需要对比

Dynamo 用这个 contrast 来 diagnose bottleneck type 和 generate targeted capability。

### 3. Training-Free Adaptation 的 Scope

Paper 是 honest 的：Limitations 里说需要 per-case correctness signal。Fully unsupervised deployment outside scope。但 paper 也说 framework agnostic to feedback source，可以用 LLM verifier 或 self-consistency。

这留下一个 open question：能不能完全 unsupervised？这跟 Sun et al. 2025 的 "self-improvement paradox" paper 直接相关（https://aclanthology.org/2025.findings-acl.588/ ）——LM 能不能 bootstrap reasoning capabilities without external scaffolding？Dynamo 给了一个 partial constructive answer：在 visual reasoning domain，with small labeled subset，可以。

### 4. RL 和 Self-Evolution 不矛盾

Dynamo + RL > RL alone，这 suggest 两个 intervention target 不同 failure modes。RL 优化 intrinsic reasoning 和 tool invocation，Dynamo 提供 external SOPs 和 visual transformations。两者 complementary，所以 additive composition。

这给一个 practical implication：**production 里如果 compute budget 充裕，Dynamo + RL 是最强配置；如果 compute 紧张，Dynamo alone 就能 recover 65-99% 的 RL gain**。

## Bottom Line

Dynamo 这个工作给人最大的 intuition 是：**task-specific VLM adaptation 的大部分收益，可以通过让 agent 在 capability space 上做 discrete search 来获得，而不需要在 weight space 上做 gradient descent**。这个 claim 有 20 个 model-benchmark cells 的 direct inference improvement、GTA 的 tool mastery、以及跟 RL 的 comparison 来支撑。

最 clever 的地方在于 mastery skill 作为 tool 的 gate——这解决了 tool creation 领域一个长期被忽视的问题：generated tools 不能裸跑，需要 learned applicability boundaries。这个 insight 估计不仅适用于 visual reasoning，在其他 tool-use domain 也应该 generalize。

参考链接汇总：
- Voyager: https://arxiv.org/abs/2305.16291
- Reflexion: https://arxiv.org/abs/2303.11366
- VTool-R1: https://arxiv.org/abs/2505.19255
- DeepEyes: https://arxiv.org/abs/2505.14362
- V*: https://arxiv.org/abs/2312.14135
- ChartQA: https://arxiv.org/abs/2203.10244
- MathVista: https://arxiv.org/abs/2310.02255
- ViperGPT: https://arxiv.org/abs/2303.08128
- VisProg: https://arxiv.org/abs/2211.08343
- Chameleon: https://arxiv.org/abs/2304.09842
- CREATOR: https://arxiv.org/abs/2305.14318
- LATM: https://arxiv.org/abs/2305.17126
- ExpeL: https://arxiv.org/abs/2308.10144
- PixelReasoner: https://arxiv.org/abs/2505.15966
- V-Thinker: https://arxiv.org/abs/2511.04460
- EcoAlign: https://arxiv.org/abs/2511.11301
- EvolveR: https://arxiv.org/abs/2510.16079
- ToolLLM: https://arxiv.org/abs/2307.16789
- Self-improvement paradox: https://aclanthology.org/2025.findings-acl.588/

---

# Dynamo: Training-Free Capability Evolution for VLMs

## 1. Paper的整体定位

这篇paper来自Alibaba Qwen Team, 核心insight其实非常elegant: **与其为每个new task family手工curate SFT data或设计RL pipeline, 不如让frozen VLM自己inspect自己的behaviour, 从small labeled training subset构建capability set**。这跟传统的per-task adaptation pipeline形成了一个范式转换。

Karpathy你肯定会喜欢这个angle, 因为这本质上是在问: "VLM adaptation到底需要多少gradient updates?" Dynamo给出的答案是: 在很多情况下, zero gradient updates就够了, 只要你让agent学会从自己的mistakes中提取reusable的capability。

paper里Figure 1那个对比图非常clever:
- Top: 每个new task family都需要hand-curated SFT data + custom RL pipeline (per-task cost)
- Bottom: Dynamo从small labeled training subset evolve出task-family-specific skill和tool library, VLM保持frozen

reference: 这跟Voyager (Wang et al., 2024a)在Minecraft中积累skill library的思路有神似, 但Voyager是text-only game domain, Dynamo把它迁移到了vision-language domain, 而且引入了tool generation: https://arxiv.org/abs/2305.16291

## 2. Core Formulation深度解析

### 2.1 Problem Setup

训练集定义:
$$\mathcal{D}_{\mathrm{train}} = \{(x_i, y_i, \mathbf{v}_i)\}_{i=1}^{k}$$

变量含义:
- $x_i$: 第$i$个case的question
- $y_i$: ground-truth answer
- $\mathbf{v}_i$: associated visual input (一张或多张image)
- $k$: training subset大小 (默认是benchmark官方train split的10%)

Capability set定义:
$$C = (S, \mathcal{T})$$

- $S$: skill library (structured reasoning SOPs, Markdown格式)
- $\mathcal{T}$: tool library (executable Python programs, ≤150 lines)

### 2.2 Agent的inference公式 (Eq 1)

$$f_C(x, \mathbf{v}; \pi_\theta) = \pi_\theta(\cdot \mid x, \mathbf{v}, \mathrm{Retrieve}(x, \mathbf{v}; C)) \tag{1}$$

变量含义:
- $\pi_\theta$: frozen VLM backbone (参数$\theta$固定, $\nabla_\theta = 0$)
- $\mathrm{Retrieve}(\cdot)$: 返回与当前input相关的skill/tool子集
- 输出: VLM在augmented context下的answer distribution

intuition: 这本质上是把inference分解成两个step: retrieval (找到relevant capability) + reasoning (用capability solve)。Retrieve用的是BM25 cosine similarity, top-$K$=3 skills。

### 2.3 Optimization Objective (Eq 2)

$$C^\star = \arg\max_C \mathrm{Acc}(f_C; \mathcal{D}_{\mathrm{val}}) \tag{2}$$

关键约束: 优化变量只有$C$, $\pi_\theta$保持frozen。这跟standard fine-tuning有本质区别: 我们在capability space上做search, 而不是在weight space上做gradient descent。

## 3. Evolution Loop: 三阶段深度剖析

这是paper的核心algorithm, Algorithm 1给了pseudocode。每个iteration有三个phase:

### 3.1 Diagnose Phase

Step 1: 用当前$C$ solve $\mathcal{D}_{\mathrm{train}}$, 每个case产生reasoning trace $\tau_i$。

Step 2: 采样 $\mathcal{D}_{\mathrm{sub}} \subseteq \mathcal{D}_{\mathrm{train}}$, **关键设计**: 同时包含correct和incorrect attempts (当两者都存在时)。
- 如果全部correct: skip iteration (no bottleneck to fix)
- 如果全部incorrect: 在failures上单独处理

Step 3: AnalyzerDecider检查 $\mathcal{D}_{\mathrm{sub}}$ 的:
- questions $x_i$
- ground-truth answers $y_i$  
- original images $\mathbf{v}_i$
- reasoning traces $\tau_i$
- intermediate tool outputs

输出三件事:
1. Root-cause analysis (grounded in visual evidence)
2. Bottleneck type: **cognitive** vs **perceptual**
   - Cognitive: visual evidence available但reasoning procedure weak
   - Perceptual: agent needs transformed visual input before evidence becomes legible
3. Action $a \in \{\text{skill}, \text{tool}, \text{both}\}$

这里有一个非常subtle的设计: **visual input对diagnosis是essential的**。paper里明确说: "deciding whether a crop is misaligned, whether labels are legible, or whether a processed image introduced artefacts requires inspecting the image, not just the reasoning trace."

### 3.2 Explore Phase

Generator基于diagnosis提出$M$个candidate skill+tool combinations, 每个用不同strategy/implementation解决同一个diagnosed bottleneck。这是multi-candidate exploration, 避免single-candidate的variance。

hyperparameter: Generator temperature=0.7, max tokens=2048。

### 3.3 Validate and Promote Phase (Eq 3)

$$c^\star = \arg\max_{c \in \{c_1, \ldots, c_M\}} \mathrm{Acc}(f_{C \cup c}; \mathcal{D}_{\mathrm{train}}) \tag{3}$$

Promotion guard:
$$c \gets C \cup \{c^\star\} \quad \text{if } \mathrm{Acc}(f_{C \cup c^\star}) > \mathrm{Acc}(f_C)$$

变量含义:
- $c_m = (s_m, t_m)$: 第$m$个candidate, 包含skill $s_m$和tool $t_m$
- $M$: candidate数量 (paper保持single-digit budget)
- Guard保证monotonic training-set accuracy

这个guard有两个requirement:
1. **Origin requirement**: 新capability必须在它target的cases上improve over previous $f_C$
2. **Regression requirement**: 新capability不能degrade currently-correct cases

**关键insight**: Eq 3是Eq 2的training-set proxy。因为$\mathcal{D}_{\mathrm{train}}$和$\mathcal{D}_{\mathrm{val}}$ disjoint, 这个proxy不会contaminate held-out numbers。唯一residual concern是selection bias in $c^\star$ over discrete candidate set, 但这scale logarithmically in $M$, 所以paper保持$M$在single-digit。

### 3.4 整体loop

默认$N=3$ iterations。每iteration可能promote 0或1个capability。如果某iteration没有candidate strictly improve over current $f_C$, $C$保持unchanged。

## 4. Skills和Tools的设计

### 4.1 Skill的结构

Skill是structured Markdown document, 四个fields:
1. **When-to-Use**: trigger predicate over question和image type
2. **Strategy**: numbered step-by-step SOP
3. **Common Pitfalls**: 常见陷阱
4. **Worked Example**: 工作示例

如果S中已有skill覆盖同一problem class, 新insights merge进existing skill, 防止library bloat。

At inference, Retrieve返回top-$K$=3 skills by BM25 similarity to question, append到system prompt。

### 4.2 Tool的设计

Tool是Python function, ≤150 lines, 接收image paths, 返回processed images或extracted data。用OpenCV, PIL, NumPy。

paper里提到的representative examples:
- Chart re-rendering with enlarged axis labels
- Saliency-guided sub-region extraction  
- Contrast enhancement for low-quality documents

### 4.3 Mastery as paired skill (核心创新)

当AnalyzerDecider的action是both时, Generator emit skill alongside tool。这个**paired skill就是tool的mastery SOP**:
- When-to-Use predicate: 指定tool在哪些input pattern下helpful
- Common Pitfalls: 标记tool会hurt的pattern
- Strategy: 指导Solver如何invoke tool和consume output

**关键设计**: Tool invocation由retrieving paired skill来gate。这意味deployment是selective by construction, 而不是indiscriminate。

这跟prior tool-creation methods (CREATOR, LATM)有本质区别: 那些方法expose generated tools without learned applicability boundaries。reference: https://arxiv.org/abs/2305.14318 , https://arxiv.org/abs/2305.17126

### 4.4 Mode B: External tool sets

当curated tool set $\mathcal{T}_0 \neq \emptyset$提供时, Dynamo可以skip tool generation, 为每个provided tool $t_j \in \mathcal{T}_0$ synthesize mastery skill。这learn when to invoke each tool from agent自己的behaviour on $\mathcal{D}_{\mathrm{train}}$。

这对应Experiment II的GTA setting。

## 5. Online Adaptation to Distribution Shift

这是paper里我觉得最elegant的设计之一。real deployment中query distribution会shift over time。

### 5.1 Feedback Signal

需要per-case correctness signal。paper列出三个practical channels:
1. Small fraction of queries reviewed by humans in the loop
2. Automated QA pipeline
3. LLM-based verifier scoring agent's answer post-hoc

experiments用benchmark ground-truth labels作为stand-in。framework本身agnostic to feedback source。

### 5.2 Rolling-window detection (Eq 4)

$$\hat{\phi}_t = \arg\max_\phi \sum_{i=t-W+1}^{t} \mathbf{1}[\mathrm{fam}(x_i) = \phi] \tag{4}$$

变量含义:
- $W$: rolling-window length
- $\mathrm{fam}(\cdot)$: query family classifier (从question和image features分类)
- $\hat{\phi}_t$: step $t$时的dominant task family
- $\mathbf{1}[\cdot]$: indicator function

当$\hat{\phi}_t \neq \hat{\phi}_{t-1}$时, 触发下一次evolution iteration on new sub-stream。

### 5.3 实验结果

Figure 3比较四个policy: Direct, Static, Online adapt, Oracle。ordering是:
$$\text{Direct} \leq \text{Static} < \text{Online adapt} \approx \text{Oracle}$$

GPT-5.4在stress protocol下: Direct collapses to 0.03, Online adapt still recovers to 0.95。Shift detection latency: 2-7 cases per phase。

Table 10给出per-backbone numerical aggregates, 这里extract几个关键数据点:

GPT-5.4:
- Natural: Direct 0.761, Static 0.800, Online 0.830, Oracle 0.833
- Capability-relevant: Direct 0.479, Static 0.636, Online 0.870, Oracle 0.899
- Stress: Direct 0.026, Static 0.391, Online 0.947, Oracle 0.995

Qwen3.5-27B:
- Stress: Direct 0.122, Static 0.488, Online 0.940, Oracle 0.984

Doubao-Seed-2.0:
- Capability-relevant: Direct 0.635, Static 0.773, Online 0.900, Oracle 0.917

intuition: Static在shift时lag严重, Online adapt几乎match Oracle, 证明rolling-window detection + re-evolution的有效性。

## 6. 实验数据深度分析

### 6.1 Experiment I: Autonomous Evolution from Scratch

Table 1是core result, 5 backbones × 4 benchmarks × 4 configurations。我extract几个关键patterns:

**Pattern 1: V*和HRBench4K上Tool Only和Full dominate**

o4-mini V*: None 0.7285 → Tool Only 0.8387 → Full 0.8698 (+14.1)
o4-mini HRBench4K: None 0.7300 → Tool Only 0.7457 → Full 0.8571 (+12.7)

这对应perceptual bottleneck: 需要transformed visual input (zoom, crop, search)。

**Pattern 2: ChartQA和MathVista上Skill Only competitive**

GPT-4o ChartQA: None 0.7552 → Skill Only 0.7785 → Full 0.7799
GPT-4o MathVista: None 0.6711 → Skill Only 0.7122 → Full 0.7189

这对应cognitive bottleneck: visual evidence present但reasoning procedure weak, skill的SOP帮助decomposition和calculation。

**Pattern 3: 平均+5.6 accuracy points across all 20 cells**

Full configuration在所有20个model-benchmark cells上都improve over None baseline, 这是非常consistent的结果。

**Pattern 4: Larger gains on weaker backbones**

o4-mini V* +14.1 vs GPT-5.4 V* +7.7。这suggests weaker backbones有更多room for capability evolution。

### 6.2 Training Set Size Sensitivity (Table 9)

ChartQA上sweep $k \in \{10, 25, 50, 100, 200\}$:

| k | Promoted | Train replay | Val[:200] |
|---|----------|--------------|-----------|
| 10 | 0 | 9/10 (0.900) | 162/200 (0.810) |
| 25 | 0 | 23/25 (0.920) | 159/200 (0.795) |
| 50 | 1 skill | 45/50 (0.900) | 159/200 (0.795) |
| 100 | 1 skill | 88/100 (0.880) | 158/200 (0.790) |
| 200 | 1 skill + 1 tool | 180/200 (0.900) | 161/200 (0.805) |

intuition: Held-out accuracy在0.790-0.810区间stable, 支持saturation interpretation。Small subsets已经expose common ChartQA reading patterns。只有$k=200$ promote了generated visual tool + mastery skill, 但差异仍在2 accuracy points内。

### 6.3 Multi-benchmark Joint Evolution (Table 11)

GPT-4o, 单一unified skill on 10% joint training subset:

| Benchmark | Held-out cases | Correct | Accuracy |
|-----------|----------------|---------|----------|
| ChartQA | 1,920 | 1,488 | 0.775 |
| MathVista | 900 | 616 | 0.684 |
| HRBench4K | 700 | 476 | 0.680 |
| V* | 151 | 101 | 0.669 |
| Combined | 3,671 | 2,681 | 0.730 |

intuition: 单一library可以generalize across visually distinct benchmark families, 不需要per-benchmark evolution。这在deployment中很practical。

### 6.4 Experiment II: GTA (Mode B)

Table 2比较Base Agent, XSkill-style, Dynamo Mode B。Doubao-Seed-2.0的gain最dramatic:

| Method | ToolAcc | ArgAcc | InstAcc | AnsAcc |
|--------|---------|--------|---------|--------|
| Base | 67.7 | 45.5 | 41.6 | 76.9 |
| XSkill-style | 72.8 | 48.4 | 44.4 | 76.9 |
| Dynamo | 86.4 | 61.3 | 57.7 | 81.8 |

Gain: +18.7 ToolAcc, +15.8 ArgAcc, +16.1 InstAcc, +4.9 AnsAcc。

**关键insight**: XSkill-style baseline用的是generic visual-reasoning prompt, 没有GTA-specific tool-chain protocol。在GPT-4o和GPT-5.4上, XSkill-style甚至drop below Base on step-level metrics——adding generic prompt without learning environment's tool protocol反而confuse tool selection。

这证明: gain不是来自skill injection per se, 而是来自mastery skills encode target environment的specific tool chains, argument formats, answer normalisation rules。

### 6.5 Controlled Tool-Policy Analysis (Table 3)

Doubao-Seed-2.0-Pro on GTA:

**Tool strength study** (fixed OCR+Calc env):
- Direct, no tools: 30 cases, 83.3% acc, 0 tool use
- Atomic OCR+Calc: 86.7%, 100% tool use, 1.83 avg calls
- Composite VAS: 90.0%, 100% tool use, 1.00 avg calls

**OCR role study**:
- OCR+Calc env: 80.0%, 2.20 avg calls (OCR→Calculator chain)
- OCR+Search env: 85.0%, 2.35 avg calls (OCR→Search→OCR→Search)

intuition: 同一个OCR tool在不同environment中play不同role。Composite tool compress policy (1 call vs 1.83 calls)。

### 6.6 Experiment III: RL Comparison

Table 4是Karpathy你可能会最感兴趣的comparison。Qwen2.5-VL 3B/7B, 两个task families。

**ChartQA/TableQA** (VTool-R1 protocol):

| Backbone | Variant | ChartQA | TableQA |
|----------|---------|---------|---------|
| 3B | Direct | 45.40 | 43.09 |
| 3B | RL | 66.95 | 56.25 |
| 3B | Dynamo | 59.56 | 51.97 |
| 3B | Dynamo+RL | 68.28 | 59.54 |
| 7B | Direct | 56.05 | 56.91 |
| 7B | RL | 75.18 | 68.42 |
| 7B | Dynamo | 75.06 | 67.43 |
| 7B | Dynamo+RL | 76.88 | 69.41 |

Gap recovery fraction (Dynamo−Direct)/(RL−Direct):
- ChartQA 3B: 65.7%
- ChartQA 7B: 99.4%
- TableQA 3B: 67.5%
- TableQA 7B: 91.4%

**V*/HRBench4K** (DeepEyes protocol):

| Backbone | Variant | V* | HRBench4K |
|----------|---------|----|-----------| 
| 3B | Direct | 71.20 | 63.62 |
| 3B | RL | 78.01 | 66.50 |
| 3B | Dynamo | 84.29 | 70.50 |
| 3B | Dynamo+RL | 85.34 | 71.25 |
| 7B | Direct | 79.06 | 70.50 |
| 7B | RL | 84.82 | 76.88 |
| 7B | Dynamo | 85.86 | 75.12 |
| 7B | Dynamo+RL | 92.67 | 81.25 |

**关键insight**: 
1. V*上Dynamo alone beat RL at both scales (+6.3 at 3B, +1.0 at 7B)
2. HRBench4K上Dynamo beat RL at 3B (+4.0), trail at 7B (−1.8)
3. Dynamo+RL在所有cells上beat RL alone 1-3 points → **additive composition**

### 6.7 Cost Comparison

RL baselines: $10^4$-$10^5$ rollouts, 80-240 gradient updates per backbone。
- VTool-R1: 80 GRPO steps × 32 prompts × 8 rollouts = ~20,480 trajectories
- DeepEyes: 240 RL steps × 128 prompts × 4 rollouts = ~122,880 trajectories

Dynamo: NO gradient updates, ~few thousand frozen-VLM API calls per benchmark。这是几个order of magnitude的compute savings。

## 7. Mechanism Ablations深度剖析

这section是paper最有intellectual rigor的部分, 每个ablation都rules out一个alternative explanation。

### 7.1 Generic Skill vs Evolved Capability (Table 6)

HRBench4K, 700 cases, Doubao-Seed-2.0-Pro, tools disabled:

| Variant | Tools | Correct/Total | Accuracy |
|---------|-------|---------------|----------|
| No Capability | off | 599/700 | 0.8557 |
| Generic Skill | off | 594/700 | 0.8486 |
| Evolved No-Tool Skill | off | 597/700 | 0.8529 |
| Evolved Tool Only | on | 625/700 | 0.8929 |
| Evolved Skill + Tool | on | 629/700 | 0.8986 |

**关键结论**: 
1. Generic Skill (0.8486)甚至slightly below No Capability (0.8557)——adding neutral task prompt不recover tool-enabled gains
2. Evolved No-Tool Skill (0.8529)与No Capability tied within noise
3. Tool-enabled rows jump 4-5 points → active ingredient是validated visual artifacts, 不是"more prompt text"
4. Evolved Skill + Tool (0.8986) > Evolved Tool Only (0.8929) → paired mastery skill提供额外boost

### 7.2 Correct-Only vs Error-Only Diagnosis (Table 8)

ChartQA, $k=50$, $N=3$:

| Variant | Rounds | Promoted | Train replay |
|---------|--------|----------|--------------|
| Correct-Only | 0 | 0 | 45/50 |
| Error-Only | 3 | 0 | 45/50 |

**Correct-Only**: degenerate, no failure cluster formed, evolution loop has no target。
**Error-Only**: active但unstable, 每round generate fresh broad overlay tool (chart_data_point_highlighter, chart_series_value_overlay, chart_data_point_overlay_generator), 但none promotes because tie baseline selection score。

intuition: **contrast between solved和unsolved cases is what gives AnalyzerDecider stable target**。这跟contrastive learning的intuition相通。

### 7.3 Text-Only Diagnosis (Table 7)

Guarded HRBench4K pilot, 30 validation cases:

| Variant | Baseline | Candidate | Δ |
|---------|----------|-----------|---|
| Text-only Diagnosis | 0.6667 | 0.6667 | 0.0000 |

Text-only candidate被rejected because proposed tool produced no visual artifact。

**关键insight**: Text reflection能说answer wrong, 但不能tell whether crop misaligned, whether zoomed region missed target, whether processed image introduced artefacts。Visual diagnosis is necessary to materialize usable image-processing tool。

### 7.4 No Paired Mastery Skill

| Variant | Baseline | Candidate | Δ |
|---------|----------|-----------|---|
| No Paired Mastery | 0.5667 | 0.5333 | -0.0333 |

Unpaired tool passed smoke validation, 但once exposed without mastery skill, regressed on subset。Targeted HRBench4K-cross family dropped from 0.5625 to 0.4375。

**关键insight**: Tool本身可以executable和locally plausible, 但missing when-to-use gate makes it harmful at subset scale。这证明paired mastery skill的设计是必要的。

## 8. Qualitative Case Studies

Appendix D给三个case studies, 非常illustrative。

### 8.1 Case A: ReFocus-Style Skill and Tool

reference: ReFocus https://arxiv.org/abs/2411.11342

Dynamo在ChartQA上autonomously生成Edit-Then-Read capability:
- Skill: "Edit-Then-Read for Structured Images"
- When-to-Use: question refers to specific chart/table element visually close to distractors
- Strategy: identify visual entity → create edited view marking relevant region → re-read → arithmetic
- Tool: `mark_relevant_region(image_path, region, mode="highlight")` 用PIL ImageDraw

跟ReFocus区别: ReFocus定义visual editing as reasoning interface up front, Dynamo只在diagnosis identifies visual disambiguation as bottleneck时才emit这个skill/tool pair。

### 8.2 Case B: ZoomEye-Style Skill and Tool

reference: ZoomEye https://arxiv.org/abs/2506.13610 (可能)

HRBench4K上生成Coarse-to-Fine Region Search:
- Skill: "Coarse-to-Fine Region Search"
- Tool: `coarse_to_fine_zoom(image_path, target_description, score_tile, grid=2, depth=2)`
- 递归split image into 2×2 grid, score all 4 tiles, zoom into best one, recurse

跟ZoomEye区别: ZoomEye是general tree-search algorithm fixed up front, Dynamo只在diagnosis identifies perceptual-resolution bottleneck时emit, 且只实现diagnosis warrant的zoom/search subset。

### 8.3 Case C: Interleaved Visual Skill (最interesting)

这是paper里最clever的设计。artefact存储partly visual, visual part教by demonstration:

- Skill: "Conservative Cropping with Axis-Label Preservation"
- Trigger: ChartQA bar chart, agent第一次crop太aggressive, x-axis labels cut off
- AnalyzerDecider诊断: "crop too aggressive: target context lost"
- Generator emit paired skill+tool with **two reference images**:
  - `crop_good.png`: target appearance, year labels (2016, 2015)和category headers fully visible
  - `crop_bad.png`: failure appearance, year-label column masked out

Skill Strategy step 4 literally instructs agent:
- If your crop looks like `crop_bad.png` → re-think bounding box, expand ~30px, re-crop
- If your crop looks like `crop_good.png` → proceed to read off values

**关键insight**: 这不是text mentions picture, 是Markdown body whose Strategy step 4 names two specific stored images and prescribes different agent behaviour depending on which one agent's own intermediate crop matches。Good reference, bad reference, diagnostic line, conservative-cropping rule全部由同一evolution iteration emit, 一起retrieve。

## 9. Limitations

paper里诚实列出三个limitations:

### 9.1 Diagnosis bounded by backbone

AnalyzerDecider的bottleneck classification和Generator的candidates由同一个frozen VLM产生。Weak self-introspection的backbone可能mis-classify bottleneck, propagate到mistargeted skill/tool。Promotion guard rejects under-performing candidates但不correct upstream diagnosis errors。Smaller/less capable backbones可能converge更slowly或到less useful library。

### 9.2 Benchmark scope

只evaluate image-based VLMs across four visual reasoning benchmarks + GTA + VTool-R1/DeepEyes。Other modalities (3D spatial reasoning, medical imaging) outside scope。Whether cognitive/perceptual decision rule和executable-tool interface transfer to those settings留给future work。

### 9.3 Reliance on per-case correctness signal

Offline evolution loop和online adaptation都需要per-case correctness signal。Experiments用ground-truth作为stand-in, 对应deployment with human-in-loop review, automated QA, 或LLM verifier。Fully unsupervised production traffic outside scope。Extending to that setting需要confidence-based或self-consistency-based feedback substitute。

## 10. 与相关work的关系网

让我构建一个intellectual map:

### 10.1 Self-improving agents lineage

- **Reflexion** (Shinn et al., 2023): verbal reflection buffer resets between episodes。Dynamo的skill/tool library是persistent的。https://arxiv.org/abs/2303.11366
- **Voyager** (Wang et al., 2024a): Minecraft中persistent skill library。Dynamo迁移到vision-language domain + tool generation。https://arxiv.org/abs/2305.16291
- **ExpeL** (Zhao et al., 2024): experiential learners。https://arxiv.org/abs/2308.10144
- **AutoManual** / **EvolveR**: refine instruction manuals/reasoning strategies via self-play。https://arxiv.org/abs/2510.16079
- **Trace2Skill** (Ni et al., 2026): distil trajectory-local lessons into transferable skills。

### 10.2 Tool creation for LLMs

- **CREATOR** (Qian et al., 2023): LLM写Python helpers for unsolvable problems。Dynamo区别: paired mastery skill gates invocation。https://arxiv.org/abs/2305.14318
- **LATM** (Cai et al., 2024): scale to large API collections。https://arxiv.org/abs/2305.17126
- **ToolLLM** (Qin et al., 2024): 16000+ real-world APIs。https://arxiv.org/abs/2307.16789

### 10.3 Visual reasoning with tools

- **VTool-R1** (Wu et al., 2025a): RL with curated trajectories teach VLMs invoke fixed tool inventory。https://arxiv.org/abs/2505.19255
- **DeepEyes** (Zheng et al., 2025): RL for "thinking with images"。https://arxiv.org/abs/2505.14362
- **PixelReasoner** (Su et al., 2025): pixel-space reasoning with curiosity-driven RL。https://arxiv.org/abs/2505.15966
- **V-Thinker** (Qiao et al., 2025): interactive thinking with images。https://arxiv.org/abs/2511.04460
- **ZoomEye** (Shen et al., 2025): training-free但fixed zoom/search interface。
- **ReFocus** (Fu et al., 2025): training-free但fixed editing interface。
- **Lever LM** (Yang et al., 2024): in-context example sequences。

### 10.4 Program synthesis lineage

- **VisProg** (Gupta & Kembhavi, 2023): compositional visual reasoning without training。https://arxiv.org/abs/2211.08343
- **ViperGPT** (Surís et al., 2023): visual inference via Python execution。https://arxiv.org/abs/2303.08128
- **Chameleon** (Lu et al., 2023): plug-and-play compositional reasoning。https://arxiv.org/abs/2304.09842

这些compose hand-curated tool libraries of detectors, OCR, arithmetic modules。Dynamo区别: tool library是evolved而不是hand-curated。

### 10.5 Benchmarks

- **ChartQA** (Masry et al., 2022): multi-step numerical reasoning over charts。https://arxiv.org/abs/2203.10244
- **MathVista** (Lu et al., 2024): math reasoning in figures/plots。https://arxiv.org/abs/2310.02255
- **HRBench4K** (Wang et al., 2025): perception on 4K-resolution images。
- **V*** (Wu & Xie, 2024): visual search for object's property。https://arxiv.org/abs/2312.14135
- **GTA** (Wang et al., 2024b): general tool agents benchmark。
- **MMR-Bench** (Ma et al., 2026): multimodal LLM routing benchmark。

### 10.6 作者自己的相关工作

- **Self-improvement paradox** (Sun et al., 2025): "Can language models bootstrap reasoning capabilities without external scaffolding?" 这是同一first author的paper, 探讨self-improvement的fundamental limits。Dynamo可以看作这个paradox在vision-language domain的一个constructive answer。
- **EcoAlign** (Cheng et al., 2025): economically rational framework for LVLM alignment。https://arxiv.org/abs/2511.11301

## 11. Karpathy视角的Critical Thinking

### 11.1 为什么training-free adaptation可行? Intuition

这其实跟LLM的in-context learning有deep connection。当VLM是frozen的, 我们能tune的只有context。Dynamo本质上是在context space上做structured search:
- Skills是structured prompts (SOPs)
- Tools是executable programs that transform visual input before it enters context
- Mastery skills是gate controlling what enters context when

这跟RL在weight space上做search有本质区别, 但两者目标相同: maximize accuracy on task family。Dynamo的insight是: 在capability space上做discrete search可以recover大部分RL gain, at fraction of compute。

### 11.2 Skill-Tool coevolution的必要性

Table 1的Skill Only和Tool Only ablation揭示一个dichotomy:
- Cognitive bottleneck (ChartQA, MathVista): Skill Only competitive
- Perceptual bottleneck (HRBench4K, V*): Tool Only和Full dominate

这suggests瓶颈类型决定了哪种capability更effective。Full configuration让AnalyzerDecider per-case决定action, 这是为什么Full在所有cells上都improve。

### 11.3 Contrastive Diagnosis的深层意义

Table 8的Correct-Only vs Error-Only ablation非常insightful。这跟contrastive learning的intuition相通: 

- Correct cases定义了现有capability的boundary (what we can already do)
- Incorrect cases暴露了missing behaviour (what we cannot do)
- Contrast between them给出了evolution的direction signal

单看任何一polarity都不够:
- Correct-only: no target for evolution
- Error-only: over-focus on visible failures without enough successful neighboring cases to constrain proposed capability

这跟self-supervised learning中的positive-negative pair construction有异曲同工之妙。

### 11.4 Mastery Skill作为Tool Gate的深层意义

Section 7.4的No Paired Mastery ablation证明: tool本身可以executable和locally plausible, 但missing when-to-use gate makes it harmful at subset scale。

这给了一个deep insight: **indiscriminate tool deployment是dangerous的**。Prior tool-creation methods (CREATOR, LATM) expose generated tools without learned applicability boundaries。Dynamo的paired mastery skill本质上是在学习tool的applicability distribution, 这是一个classifier over input patterns。

这跟RL中的policy learning有相通之处: RL学的也是when to take which action, 只是Dynamo用skill retrieval来approximate这个policy。

### 11.5 Dynamo + RL的Additive Composition

Table 4的Dynamo+RL在所有cells上beat RL alone 1-3 points。这suggests两个intervention target不同的failure modes:
- RL: 优化weight space, 改善VLM的intrinsic reasoning和tool invocation
- Dynamo: 优化capability space, 提供external SOPs和visual transformations

两者互补, 所以compose additively。这跟ensemble methods的intuition相通: diverse error modes的methods compose better。

### 11.6 Online Adaptation的Rolling-Window设计

Eq 4的rolling-window detection非常elegant。$\hat{\phi}_t$是dominant task family, 当flip时trigger evolution。这相当于一个change-point detection algorithm。

intuition: 这跟online learning中的concept drift detection有相通之处。Dynamo的contribution是把这个idea应用到capability evolution上, 且decoupled from feedback signal source。

### 11.7 跟AGI路径的关系

Karpathy你可能会对这个angle感兴趣。Dynamo代表了一种alternative path to AGI:
- 传统path: scaling laws + more data + more compute → better weights
- Dynamo path: frozen weights + capability evolution + tool creation → better agent

两者不矛盾, 可以compose。但Dynamo path有一些attractive properties:
1. Training-free: no GPU, no gradient updates
2. Interpretable: skills是Markdown, tools是Python, 都是human-readable
3. Composable: 跟RL additively compose
4. Online adaptable: handles distribution shift
5. Persistent: library accumulates across episodes

这跟你的"software 2.0 vs software 1.0"框架有connection: weights是software 2.0 (learned), skills/tools是software 1.0 (written, 但这里由agent自己write)。Dynamo是在探索两者如何compose。

### 11.8 开放问题

读完paper我有几个open questions:

1. **Multi-candidate exploration的$M$选择**: paper保持single-digit, 但没系统study $M$的sensitivity。更大的$M$是否能在weaker backbones上recover more gain?

2. **Skill library的容量**: 随着evolution iterations增加, library如何避免bloat? paper提到merge same problem class的skills, 但没给quantitative bound。

3. **Tool composition**: 能否evolve出compose multiple tools的skill? 比如crop → enhance → OCR的chain。Table 3的OCR+Search env hint了这个方向。

4. **Cross-modal transfer**: 同一skill能否transfer across modalities? 比如3D spatial reasoning的skill能否从2D analogies evolve?

5. **Fully unsupervised adaptation**: Limitations里提到需要per-case correctness signal。能否用self-consistency或confidence-based substitute? 这跟self-improvement paradox paper的theme相关。

6. **Capability library的遗忘**: 长期running中, obsolete capabilities如何prune? Online adaptation的rolling-window是否handle这个?

7. **Adversarial robustness**: Generated tools是否robust to adversarial inputs? 一个mis-targeted tool能否被adversarial example trigger?

## 12. 总结

Dynamo这篇paper的core contribution可以概括为三点:

1. **Problem formulation**: 把per-task VLM adaptation recast为capability evolution。Frozen VLM从small labeled training subset构建自己的skill/tool library。

2. **Multi-candidate evolution loop**: Diagnose (correct + incorrect contrast) → Explore (M candidates) → Validate and Promote (monotonic guard)。Mastery skill paired with tool gates invocation。

3. **Empirical evidence**: 
   - 4 benchmarks × 5 backbones, +5.6 avg accuracy on all 20 cells
   - GTA: +18.7 ToolAcc on Doubao
   - RL comparison: 65-99% gap recovery on ChartQA/TableQA, matches/beats RL on V*/HRBench4K
   - Additive composition with RL
   - Online adaptation handles distribution shift, stays close to Oracle

这paper给人的intuition是: **much of task-specific VLM adaptation can be obtained without gradient updates, by letting agent shape its capability library**。这是一个strong claim, 但实验数据support它。

reference链接汇总:
- Paper本身: 应该在arxiv上, 但从内容看是2026年的paper (引用了2026年的papers)
- Voyager: https://arxiv.org/abs/2305.16291
- Reflexion: https://arxiv.org/abs/2303.11366
- VTool-R1: https://arxiv.org/abs/2505.19255
- DeepEyes: https://arxiv.org/abs/2505.14362
- V*: https://arxiv.org/abs/2312.14135
- ChartQA: https://arxiv.org/abs/2203.10244
- MathVista: https://arxiv.org/abs/2310.02255
- ViperGPT: https://arxiv.org/abs/2303.08128
- VisProg: https://arxiv.org/abs/2211.08343
- Chameleon: https://arxiv.org/abs/2304.09842
- CREATOR: https://arxiv.org/abs/2305.14318
- LATM: https://arxiv.org/abs/2305.17126
- ExpeL: https://arxiv.org/abs/2308.10144
- PixelReasoner: https://arxiv.org/abs/2505.15966
- V-Thinker: https://arxiv.org/abs/2511.04460
- EcoAlign: https://arxiv.org/abs/2511.11301
- EvolveR: https://arxiv.org/abs/2510.16079
- ToolLLM: https://arxiv.org/abs/2307.16789

Karpathy, 希望这个deep dive能build你的intuition about Dynamo。这paper的beauty在于它把VLM adaptation从weight space解放到capability space, 而且用一个非常clean的evolution loop实现了这个idea。我认为这是一个值得follow的方向, 特别是online adaptation和cross-modal transfer还有很多room to explore。
