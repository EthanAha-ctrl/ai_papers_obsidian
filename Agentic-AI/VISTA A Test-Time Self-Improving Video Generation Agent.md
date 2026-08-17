---
source_pdf: VISTA A Test-Time Self-Improving Video Generation Agent.pdf
paper_sha256: 45541bba2dbd2aab44b7bcb30c5bc63365db3b157f771f4116dc73323259a853
processed_at: '2026-08-13T01:58:30-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VISTA 这篇 paper

---

## 一句话概括

现在的 AI 生成视频已经很强了（Veo 3、Sora 这种），但有个烦人的问题：**你得反复改 prompt 才能出好视频**。VISTA 干的事就是**让 AI 自己当导演 + 评审 + 编剧，自动帮你改 prompt，改完再生成，循环往复，直到视频真的好为止**。

听起来像 self-refine？确实是，但 video 比 text 难太多了，naive 的 self-refine 在 video 上会越改越糟（paper 里 VPO 在 multi-scene 上 -16.3% win rate，比啥都不做还差）。VISTA 的贡献是搞清楚**video prompt 到底该怎么改才不会改崩**。

---

## 为什么这事难——类比一下

想象你让一个朋友帮你描述一个场景，你要他画出来。

**文本场景**：你说"写个悲伤的故事"，他写完你给反馈"再 sad 一点"，他改改就行。单一维度，feedback 容易给。

**图像场景**：你说"画只猫"，他画完你说"猫耳朵太小"——视觉问题，但好歹是静态的一张图，feedback 还是 actionable 的。

**视频场景**：你说"画个飞船进入超光速，星星往后飞"。他画完。你发现问题：
- 飞船是**垂直飞**的（不是水平加速，违反直觉）
- 背景星星**匀速划过**（没有 parallax 变化）
- 引擎**没有喷射效果**（违反物理）
- 画面切换**硬切**（没有 dissolve）
- 音频里**有风噪**（室外拍摄但有 noise）

这五个问题分属**视觉、运动、物理、剪辑、音频**五个维度。你要 feedback 给他，他改 prompt 的时候不能把别的维度搞坏。这就是 VISTA 要解决的事。

---

## VISTA 的整体流程——用一个电影制作团队来类比

Paper 里 Algorithm 1 描述的流程，翻译成人话：

### 第一阶段：开机准备（Initialization）

**Step 1：编剧拆剧本（PromptPlanner）**

用户说："一个男人在户外问 trivia，然后给答案。"

编剧 agent（其实是 MLLM）把这个 flat prompt 拆成 structured 的场景卡：

```
Scene 1 (0-5.5s):
  - Scene Type: man asking trivia outdoors
  - Characters: man wearing red baseball cap, black sunglasses...
  - Actions: speaks, gestures
  - Dialogues: "Which comedian is known for deadpan delivery?"
  - Visual Environment: outdoor, sunny day
  - Camera: medium shot
  - Sounds: ambient street noise
  - Mood: casual, humorous

Scene 2 (5.5-8s):
  - Scene Type: outro screen with branding
  - ...
```

为什么这么做？因为 Veo 3 这种 model 看到 structured 的、有时间戳的、每帧该干啥的 prompt，比看到一坨 flat 文字，生成质量好太多。这个 observation 很 empirical 但很关键——Ablation 里 w/o PromptPlanner 在 single-scene 上 Init 就从 35.5% 掉到 25.2%，说明 prompt 结构本身贡献了 10% 的 win rate。

**Step 2：执行 + 选最佳 take（Pairwise Tournament）**

根据上面拆出来的 prompt 变体，生成一堆 video（paper 里每轮 30 个），然后像选秀节目一样两两 PK，赢的晋级，最后决出冠军。

但这里有个讲究——**不能直接让 MLLM 打分**。原因 paper 里讲得挺清楚：没有 ground truth，absolute scoring 不可靠。Pairwise comparison 更接近人类判断方式。

PK 流程（Algorithm 2）：
1. 先让 MLLM 给每个 video 写一份" probing critique"——单独分析每个 video 的问题（不比较，只诊断）
2. 然后两两 PK，让 MLLM 看两个 video 加上各自的 critique，决定谁赢
3. 关键 trick：**正向比较一次，swap 顺序再比较一次**。如果两次结论一致就采纳，不一致就 tie。这是为了消除 LLM 的 position bias（先看到的 video 容易被偏好）

打分公式（人话版）：
- 每个 video 在 5 个维度上 PK（Visual Fidelity、Physical Commonsense、Text-Video Alignment、Audio-Video Alignment、Engagement）
- 赢了某维度 +1，平手 +0.5，输了 0
- 如果 video 在某维度有违规（比如突然出现/消失物体），扣 penalty
- 总分高者晋级

### 第二阶段：自我改进（Self-Improvement）

**Step 3：三人评审团 critique（MMAC）**

冠军 video 出来后，要给它写"改进意见"。但直接让 MLLM 写 critique 会有问题——Veo 3 已经很强了，surface-level 看不出毛病，MLLM 会给一堆"挺好挺好"的废话。

VISTA 的解法是借鉴**法庭陪审团制度**（paper 引用 Klevorick & Rothschild 1979 的 Jury Decision Process，https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1711&context=cowles-discussion-paper-series）：

对每个维度（Visual / Audio / Context），派三个 judge：

- **Normal Judge（正常评审）**：尽力找出优点，客观打分
- **Adversarial Judge（魔鬼评审）**：尽力找茬，专挑毛病
- **Meta Judge（首席评审）**：看前两个的意见，综合仲裁

为什么要两个对立的 judge？看 Ablation 数据就明白了：

| 配置 | Single-scene Init→Iter5 | Multi-scene Init→Iter5 |
|---|---|---|
| 只用 Normal Judge | 35.0 → 17.2 | 35.3 → 33.3 |
| 只用 Adversarial | 35.0 → 42.0 | 35.3 → 26.7 |
| 两个都用 | 35.5 → 45.9 | 37.8 → 46.3 |

只 Normal 在 single-scene 上**崩盘**到 17.2%——因为 MLLM 看不出 Veo 3 的问题，给一堆 "looks great" 的 critique，optimizer 不知道改啥，越改越乱。

只 Adversarial 在 single-scene work（42%），但 multi-scene **停滞**在 26.7%——因为 over-critical，optimizer 把 prompt 越改越复杂，最后 violate 了 Realism constraint。

两个一起用，Meta judge 平衡，才稳定 work。

举个具体例子（Table 8）：
- Prompt: "A spaceship entering hyperdrive, stars streaking past as it accelerates"
- Self-Refine（传统方法）说："The video is highly successful..."
- VISTA 的 Adversarial Judge 说："the spaceship moves **vertically**, which conflicts with viewer expectations of horizontal acceleration. Additionally, ... lack of micro-dynamics (e.g., rotational drift, buildup phases) and unrealistic exhaust behavior"

你看，Adversarial 找出来的问题是"飞船方向错了"——这是 motion 语义层面的问题，Normal Judge surface level 根本看不到。

**Step 4：编剧修订（DTPA - Deep Thinking Prompting Agent）**

拿到 critique 后，不能直接让 MLLM 改 prompt——它会 over-engineer，把 prompt 越搞越复杂。Paper Section 4.3 提到这个 failure mode。

VISTA 让 MLLM 做**六步 introspection**，每步至少 200 字 reasoning：

1. **Review Issues**：哪些 metric 分数 ≤8？把 qualitative critique 也吸收进来。如果没 major issue 就 early-exit，别瞎改
2. **Define Objectives**：用户到底想要啥？explainer？promo？tutorial？success criteria 是啥？
3. **Identify Model Limitations**：哪些问题是 model 固有的（比如生成 reflection 困难），哪些是 prompt 没说清
4. **Identify Prompt Issues**：prompt 里有没有 vague term（"engaging"）、scope 太宽、矛盾约束（"short but detailed"）、missing info
5. **Propose Revisions**：综合以上提出修改建议
6. **Revise**：复查修改建议是否覆盖所有问题，否则再改

关键设计：**DTPA 输出的是"修改动作列表"，不是新 prompt**。然后另一个 MLLM call 根据 actions 采样多个新 prompt。这样分离"诊断"和"改写"——DTPA 负责 surgical 诊断，prompt sampler 负责创意改写。

Ablation 显示 w/o DTPA：
- Single-scene: 35.5 → 37.8（只 +2.3，full 版 +10.4）
- Multi-scene: 37.8 → 45.2（接近 full 的 46.3）

所以 single-scene 上 DTPA 贡献巨大，multi-scene 上 MMAC 贡献更大——场景越复杂，critique quality 越 critical。

---

## 数据上到底好多少

### 主实验（Table 2）

VISTA vs Direct Prompting（啥都不做），iteration 5：
- Single-scene: Win 45.9% / Tie 50.2% / Loss 13.9%，**净赢 +32%**
- Multi-scene: Win 46.3% / Tie 47.2% / Loss 13.4%，**净赢 +35%**

对比 baselines（都做了 fairness scaling）：
- vs VSR++（最强的 self-refine baseline）：VISTA 多赢 18-26%
- vs Rewrite：VISTA 多赢 21-23%
- vs VPO：VISTA 多赢 17-28%

### 传统指标（Table 5, 6）

最能说明问题的几个数：

| Metric | DP | VISTA | 改善 |
|---|---|---|---|
| Dynamic Degree (single) | 75.95 | **89.87** | +14 |
| Aesthetic Quality | 61.86 | **64.53** | +2.7 |
| CLIP-Score | 0.310 | **0.358** | +15% relative |
| Audio Discontinuity (multi) | 2.62 | 2.69 | 略升 |

Dynamic Degree +14 是很显著的——说明 VISTA 不只是让画面"更漂亮"，而是让**运动更丰富真实**，这是 video 相对 image 的本质特征。

### Human Eval（Figure 4）

- 50 个 prompt，5 个 annotator
- VISTA vs 最强 baseline：**66.4% win rate**
- Self-improvement score（1-5）：VISTA 3.78 vs VSR++ 3.33
- Visual quality：VISTA 3.77 vs DP 3.36
- Audio quality：VISTA 3.47 vs DP 3.21

人类一致偏好 VISTA 输出。

### 成本（Figure 5）

每 iteration：
- ~0.7M tokens
- ~28 videos

5 iteration = 3.5M tokens + 140 videos per prompt。这在 production 上不算便宜，但 paper 显示 token 和 video 增加时性能持续提升——证明 test-time scaling 在 video 上 work。

### 弱模型也能用（Table 4）

Veo 2 + VISTA：
- Single: 15.0 → 23.8（+8.8）
- Multi: 27.6 → 33.3（+5.7）

Gain 比 Veo 3 小（Veo 3 +10.4 / +8.5），说明 VISTA 的优化需要 model 有足够的 instruction-following 能力才能 leverage。但弱模型也能受益，证明 method 的 robustness。

---

## 关键 insights 总结

### Insight 1：Video test-time optimization 的瓶颈是 prompt，不是 model

Veo 3 已经很强，DP 下输给 VISTA 35%——说明 model capability 没被充分利用。VISTA 通过 prompt refinement 释放了 model 潜力。

这和你之前在 LLM reasoning 上的直觉一致——**test-time scaling 在 inference 空间而非 parameter 空间**。

### Insight 2：Multi-modal 必须联合优化

VPO 在 multi-scene 上 -16.3% collapse，因为它只优化 helpfulness/accuracy/harmlessness，没考虑 audio 和 visual 的 trade-off。VISTA 的三维分解（Visual / Audio / Context）+ 独立 critic + meta judge 综合是关键。

### Insight 3：Adversarial + Normal 是必须的

Normal judge alone 在 single-scene collapse 到 17.2%，Adversarial alone 在 multi-scene 停滞在 26.7%。两者结合才稳定 work。这个 observation 很 fundamental——可能不只适用于 video，任何 SOTA output 的 critique 都需要这种对抗结构。

### Insight 4：Surgical edit 比 rewrite 好

DTPA 输出 modification actions 而不是新 prompt，preserve 原 prompt content 的同时 patch specific failure。这防止了 prompt drift——Figure 14 显示 VISTA 的输出内容仍然 faithful to original prompt，但 quality 大幅提升。

### Insight 5：Probing 先于比较

Pairwise tournament 里，先让 MLLM 给每个 video 写独立 critique，再做 comparison。Paper 说这避免了 model 同时做"分析 + 比较"的 cognitive overload。这个 insight 可能也适用于其他 LLM-as-judge 场景。

---

## 我自己觉得 paper 没讲透的地方

### 1. Selection 和 Critique 的 criteria 不一致
Selection 用 5 个 criteria（$\mathcal{M}_{user}^S$），Critique 用 14 个（$\mathcal{M}_{user}^C$）。这意味着 optimize 的方向可能和 evaluation 方向 misalign。比如 critique 找到 camera focus 问题，但 selection 不 weight camera focus——那 optimizer 改了也白改。Paper 没讨论这个。

### 2. Tournament 的随机 tie-break noise
Algorithm 2 L7 在 forward/swapped conflict 时"randomly assign"，这在小 sample size（30 videos）下会引入显著 variance。Paper 没分析这个 noise 对结果稳定性的影响。

### 3. Adversarial judge 在 multi-scene collapse 的原因
Ablation 显示 only Adversarial 在 multi-scene 停在 26.7%，但 paper 没深入分析 why。我猜是 over-critical 导致 prompt over-complication，最终 violate constraint。但这需要更多 analysis。

### 4. Multi-scene dataset 是 internal 的
161 prompts 来自 Google internal dataset，external researcher 无法复现这部分实验。这是个 reproducibility concern。

### 5. Cost 的实际可部署性
3.5M tokens + 140 videos per prompt，按 Gemini Flash 和 Veo 3 定价（估算），单 prompt 优化成本 $5-10。这在 high-value creative 场景（广告、电影预览）OK，但 consumer scale 不现实。Future work 需要 learned value function 替代 tournament。

### 6. Single-scene vs Multi-scene 的 mechanism 差异
Ablation 显示 PromptPlanner 对 single-scene 贡献 +10，对 multi-scene 只 +3.8。这暗示 single-scene 的 win 主要来自 structured prompt format 本身，multi-scene 的 win 主要来自 MMAC+DTPA 迭代。Paper 没把这个拆开分析——我觉得这是值得专门 section 讨论的。

---

## 和你工作的连接

### 1. Test-time scaling 的 modality-agnostic 本质
Snell et al. 2025 (https://openreview.net/forum?id=4FWAwZtd2n) 在 LLM reasoning 上证明 test-time compute 比 parameter scaling 高效。VISTA 在 video 上验证同样 work——Figure 6 显示 iteration 增加性能持续提升。

这指向 unified view：**只要 task 有可验证的 improvement signal，test-time scaling 就 work，与 modality 无关**。VISTA 的 MLLM-as-judge pairwise 就是这个 signal provider。

### 2. 两层 search 结构
VISTA 本质是 "search over prompts that are inputs to another model"——外层 search prompt space，内层 T2V model 生成 video。这种 nested search 结构和你 nanoGPT 强调的 "model 是 token sampler，外面是更高层 abstraction" 的直觉一致。

未来可能 generalizes 到更多层：search over planning strategies → search over prompts → search over model sampling params → model generates output。

### 3. Adversarial self-play 的普遍性
Normal + Adversarial Judge 本质是 self-play 的 critic side。一个 advocate 证明 output 好，另一个证明 output 差，meta 仲裁。这和 AlphaGo 的 policy/value 双角色、Debate AI (Irving 2018) 的设计、RLHF 中的 reward model 训练都同源。

VISTA 借用 Jury Decision Process (Klevorick 1979) 是个 elegant 类比，但底层是相同的 adversarial self-play intuition。这个 pattern 可能是 AGI-level reasoning 的关键 building block。

### 4. Eureka Labs / AI tutor 的连接
VISTA 的 multi-agent critique 架构可以平移到 education：
- Normal judge 评估 student understanding
- Adversarial judge 找 misconception
- Meta judge 综合
- DTPA 生成个性化 explanation

这是个 potential application direction，尤其在 student answer critique 上，adversarial probing 能 expose surface-correct-but-deeply-wrong 的 understanding。

---

## 一句话总结 intuition

**VISTA 把 human 视频制作的 implicit workflow 显式化为 multi-agent loop，每个 component 单独都不新颖（structured prompting、tournament、adversarial debate、CoT），但 composition + 适配 video 多模态特殊性的工程细节让它 work。核心 insight 是 SOTA model + naive prompting 还有 35% 的 headroom，通过 test-time scaling 在 prompt 空间 search 能释放这部分潜力。**

如果你想深挖某个具体 component，比如 MMAC 的 Meta judge prompt 设计细节（Appendix B.4 有完整 prompt），或者 PairwiseSelect 的 noise 分析，可以继续。Paper 的 Appendix B 是 prompt engineering 的金矿——每个 agent 的 system prompt 都写得很细，值得逐行分析。

---

# VISTA: Test-Time Self-Improving Video Generation Agent 深度讲解

Andrej，这篇来自 Google DeepMind 和 NUS 的工作（Do Xuan Long et al.）提出了一个相当精巧的 multi-agent 框架，核心 insight 在于：**在 T2V model 已经很强（Veo 3）的情况下，瓶颈从"模型能力"转移到了"prompt 表达"，于是 test-time scaling 在 prompt 空间而非参数空间展开**。这点和你之前在 LLM reasoning scaling 上的直觉是相通的——只是这里 search 的对象是 structured multi-modal prompts。

Project page: https://g-vista.github.io/
arXiv 链接（推测）：https://arxiv.org/abs/2506.0xxxx (需要在 arxiv 上确认)

---

## 1. 为什么 video prompt optimization 比文本/图像更难

Paper 在 Introduction 里点出一个被严重低估的 observation：现有的 test-time optimization 方法（如 Self-Refine Madaan 2023, APO Pryzant 2023, s1 Muennighof 2025, Maestro Wan 2025b）在文本和图像上奏效，但 video 有三个特殊性让它们 collapse：

- **多场景时间结构**：video unfold across scenes，prompt 必须是 temporal plan，flat string 描述会丢失时间因果。
- **多模态耦合**：visual / audio / context 必须 jointly optimize，单独优化 visual 会引入 audio regression（见 Table 2 中 VPO 在 multi-scene Δ=-16.3% 的 collapse）。
- **缺乏 ground-truth reference 的 evaluation**：IS/FID/CLIP-Score 单维度，VBench 没有 audio，TAVGBench 强依赖 embedding similarity。

VISTA 把这三个问题分别映射到三个 component：Step 1 (Structured Planning)、Step 3+4 (Joint MMAC + DTPA)、Step 2 (Pairwise Tournament)。

---

## 2. 整体架构：两阶段四步骤

### Algorithm 1 形式化

输入：
- User prompt $P$
- T2V 模型 $\text{T2V}(\cdot)$
- MLLM (Gemini 2.5 Flash)
- 迭代次数 $T$
- 可配置 criteria $\mathcal{M}_{user}^S$ (selection) 和 $\mathcal{M}_{user}^C$ (critique)
- Early-stop patience $m$

输出：$(V^*, P^*)$

**Initialization Phase**
- L1: $\mathcal{P} := \{P_1, ..., P_m, P\} \leftarrow \text{PromptPlanner}(P)$
- L2: $\mathcal{V} := \{V_1, ..., V_n\}$ where $V_i \leftarrow \text{T2V}(P_j)$
- L3: $(V^*, P^*) \leftarrow \text{PairwiseSelect}(\mathcal{V}, \mathcal{P}, P, \mathcal{M}_{user}^S)$

**Self-Improvement Phase** (for $t = 1$ to $T$)
- L5: $\mathcal{F}^t \leftarrow \text{MMAC}(V^*, P^*, P, \mathcal{M}_{user}^C)$
- L6: $\mathcal{P}^t := \{P_1^t, ..., P_m^t, P^*\} \leftarrow \text{PromptOptimizer}(P^*, P, \mathcal{F}^t)$
- L7: $\mathcal{V}^t := \{V_1^t, ..., V_n^t, V^*\}$
- L8: $(V^*, P^*) \leftarrow \text{PairwiseSelect}(\mathcal{V}^t, \mathcal{P}^t, P, \mathcal{M}_{user}^S)$
- L9-11: 若 $(V^*, P^*)$ 连续 $m$ 轮不变则 break

注意 L1 和 L6 都**保留原始 $P$ 和上一轮 champion $P^*$** 作为 residual candidate，这是关键的 exploration-exploitation 平衡——防止 planner/optimizer drift 导致 worse than direct prompting（这正是 prior method 在 single-scene 上失败的原因，见 Table 2(a) VSR Init Δ=6.0%）。

---

## 3. Step 1: Structured Video Prompt Planning 的细节

每个候选 prompt 被解析为时间序列：
$$P_i := [S_{i,1}, S_{i,2}, ...]$$

每个 scene $S_{i,j}$ 由 9 个 attribute 组成（这是相对 prior work 如 Google Cloud prompt guide 和 MovieGen 的关键升级）：

| Attribute | 例子 | 解决的问题 |
|---|---|---|
| Duration | "5.5 seconds" | 避免 T2V 自动分割导致 inconsistency |
| Scene Type | "interview" / "montage" | 给 model semantic prior |
| Characters | "a dog", "a flower" | 实体绑定 |
| Actions | "the flower is blooming" | motion 指定 |
| Dialogues | voiceover / on-screen text | audio-video alignment |
| Visual Environment | "tranquil canopy of boundless sky" | 背景一致性 |
| Camera | "close-up capturing subtle emotion" | cinematographic control |
| Sounds | "crashing waves in background" | ambient audio |
| Moods | "serene" | emotional tone |

**Planning Constraints**（默认开启，可配置）：
- Realism: 默认遵循真实物理，除非 prompt 明确说 animated
- Relevancy: 只包含 explicitly stated 或 implied 元素，避免 hallucinated invention
- Creativity: 鼓励 ambient sound 但避免过度场景切换

这些 constraint 的作用是防止 planner 自己 hallucinate content，是**fidelity-preserving 的关键**——这也是为什么 VISTA 在 Figure 14 里能"内容不漂移但质量提升"，而 VSR/ Rewrite 会 drift。

---

## 4. Step 2: Pairwise Tournament Selection —— 公式深挖

### 4.1 为什么不用 absolute scoring

Paper 引用 Lee et al. 2024 (RLAIF) 和 Liu et al. 2024c 的工作指出：scoring without ground truth 是 inherently subjective and unreliable。Pairwise comparison 更 align with human preference in RL。但 pairwise 直接做也有问题——于是引入 tournament + bidirectional swapping。

### 4.2 Tournament Algorithm (Algorithm 2)

```
输入：V = {V_1, ..., V_n}, P = {P_1, ..., P_n}, M_user^S
1. 生成 probing critiques Q := {Q_1, ..., Q_n} ← MLLM(V, P, M_user^S)
2. while |V| > 1:
   3. 分组为 pairs
   4. for each (V_i, V_j):
      5. (V_win^f, V_lose^f) ← MLLM(V_i, Q_i, V_j, Q_j, M_user^S)  # forward
      6. (V_win^s, V_lose^s) ← MLLM(V_j, Q_j, V_i, Q_i, M_user^S)  # swapped
      7. 若 forward == swapped，采纳；否则随机分配
      8. V.remove(V_lose); P.remove(P_lose)
9. return (V[0], P[0])
```

关键设计：**Probing critique 先于 comparison**。原因在 paper section 2.1 最后一段：让 model 一次性"分析 video + 比较 video"会 overload，critique 先作为 evidence，comparison 时引用，这种两步 decomposition 大幅提升 critical 评估质量。

### 4.3 Scoring 公式

$$s_i \leftarrow \frac{1}{k} \sum_{C \in \mathcal{M}_{user}^S} \left( \delta(C, V_i, V_j) - \lambda \cdot \mathbb{1}(C, V_i) \right)$$

$$s_j \leftarrow \frac{1}{k} \sum_{C \in \mathcal{M}_{user}^S} \left( 1 - \delta(C, V_i, V_j) - \lambda \cdot \mathbb{1}(C, V_j) \right)$$

变量解读：
- $C$：criterion，来自 $\mathcal{M}_{user}^S$，默认 5 个：{Visual Fidelity, Physical Commonsense, Text-Video Alignment, Audio-Video Alignment, Engagement}
- $\delta(C, V_i, V_j) \in \{0, 0.5, 1\}$：在 criterion $C$ 上 $V_i$ 对 $V_j$ 的结果，{Loss, Tie, Win}
- $\mathbb{1}(C, V) \in \{0, 1\}$：video $V$ 是否违反 $C$
- $\lambda$：violation penalty 系数（hardcoded 在 implementation 里）
- $k = |\mathcal{M}_{user}^S|$：criterions 数量

**Intuition**：$s_i$ 和 $s_j$ 是 zero-sum 但加了 violation penalty。如果一个 video win 但 violate 某些 criterion，它的 win 会被 $\lambda$ 抵消。这本质上是 constrained MCTS 的 utility function。

**Default penalty** 针对常见 T2V failure：sudden appearance/disappearance、unnatural speed、unnecessary text overlay、unexpected voiceover、过多 scene transitions。这恰好对应 Table 1 中 DP 失败但 VISTA 修复的 failure modes。

### 4.4 Bidirectional swapping 的理由

Zheng et al. 2024a 发现 LLM 在 multiple choice 上有 position bias——MLLM 看到 $(V_i, V_j)$ 倾向选前者。Swap 一次再比较，conflict 时 mark as tie 而非随机选择——这里 paper 的实现是"else assign randomly"，这是个小 caveat，可能引入 noise。

---

## 5. Step 3: Multi-Dimensional Multi-Agent Critiques (MMAC) —— 核心 contribution

### 5.1 三维分解

$\mathcal{D} = \{\text{Visual}, \text{Audio}, \text{Context}\}$，每个维度独立 evaluate。**Default metric configuration**（paper section 2.2）：

- **Visual**: Visual Fidelity, Motions and Dynamics, Temporal Consistency, Camera Focus, Visual Safety
- **Audio**: Audio Fidelity, Audio-Video Alignment, Audio Safety
- **Context**: Situational Appropriateness, Semantic Coherence, Text-Video Alignment, Physical Commonsense, Engagement, Video Format (Beginning, Ending, Transitions)

这些 metric 是从 Bansal 2024 (VideoPhysics), Cheng 2025a (MMAudio), Gao 2023, Liu 2024a (EvalCrafter) 中 strategic 筛选 refined 的。**关键 motivation**：现有 SOTA T2V 在大多数常规 metric 上已经饱和，必须用 failure-sensitive 的 fine-grained metrics 才能 differentiate。

### 5.2 Jury Decision Process 的借用

灵感来自 Klevorick and Rothschild 1979 的 Jury Decision Process——这是 law & economics 经典文献，paper 引用 https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1711&context=cowles-discussion-paper-series。核心 idea：jury 系统中既有 adversarial 又有 normal advocate，meta-judge 综合，能 expose 单一 perspective 漏掉的 flaw。

### 5.3 三 Judge 架构（公式 1）

对每个 dimension $D \in \mathcal{D}$：

$$\{C_D, S_D\} \leftarrow J_D(P, V^*, P^*) \quad \text{(Normal Judge)}$$
$$\{C_D^-, S_D^-\} \leftarrow J_D^-(P, V^*, P^*) \quad \text{(Adversarial Judge)}$$
$$\{C_D^*, S_D^*\} \leftarrow J_D^*(P, C_D, S_D, C_D^-, S_D^-) \quad \text{(Meta Judge)}$$

最终输出：
$$\mathcal{F} := \{C_D^*, S_D^* \mid D \in \mathcal{D}\}$$

变量：
- $C_D$：Normal Judge 的 critiques（文本）
- $S_D$：Normal Judge 的 scores（1-10 整数，按 Zheng et al. 2023 LLM-as-judge 的 scale 约定）
- 上标 $^-$：Adversarial 版本
- 上标 $^*$：Meta 综合

### 5.4 为什么只用 Normal Judge 会 collapse

Ablation (Table 3) 显示：
- Multi-scene only Normal Judge：iteration 5 跌到 **17.2%**（init 35.3%）
- Multi-scene only Adversarial Judge：stagnate 在 **18.8%** 多个 iterations

**Intuition**：Normal Judge 倾向给高分（Veo 3 已经很好，surface level critique 很难），但 multi-scene 复杂场景下，缺 negative probing 让 optimizer 看不到该修哪里；Adversarial alone 在 single-scene 强（35% → 42%）但 multi-scene 退步，因为 over-critical 导致 optimizer 过度修改 prompt 破坏 content。Meta judge 的作用是 balance——把"是否存在问题"和"问题严重程度"加权。

### 5.5 Critique 质量对比

Table 8 给了个 striking example：

**Prompt**: "A spaceship entering hyperdrive, stars streaking past as it accelerates."

**Self-Refine** 说："Overall, the generated video is highly successful..."

**VISTA (Motions and Dynamics)**: "the spaceship moves **vertically**, which conflicts with viewer expectations of horizontal acceleration. Additionally, the Negative Judge points out the **lack of micro-dynamics (e.g., rotational drift, buildup phases)** and **unrealistic exhaust behavior**, which diminish the believability of motion."

这正是 Veo 3 的典型 failure——surface 看起来很炫但 motion direction 是错的。Normal Judge 会 miss 这种；Adversarial Judge 的 probing question 设计专门针对"unnatural movement direction"。

---

## 6. Step 4: Deep Thinking Prompting Agent (DTPA) —— 六步 introspection

### 6.1 Why direct MLLM optimization fails

Section 4.3 提到：直接让 MLLM 改 prompt 会"overcomplicate prompts and interpret critiques shallowly"。这点和 self-refine 在 LLM 上的失败 mode 一致——模型把 critique 当成"必须改"的指令而非"诊断信息"。

### 6.2 六步 CoT

公式 (2): $\mathcal{M} := \{M_1, ...\} \leftarrow \text{DTPA}(P, P^*, \mathcal{F})$

注意 DTPA 输出的是 **modification actions**（list of strings），不是新 prompt。

公式 (3): $\mathcal{P} := \{P_1, ..., P_n, P^*\} \leftarrow \text{MLLM}(P, P^*, \mathcal{M})$

由另一个 MLLM call 根据 actions 采样 n 个新 prompt。

六步（每步至少 200 字 reasoning）：

1. **Review Issues**: identify all major issues with scores $\leq 8$，incorporate qualitative feedback。若无 major issue，skip 后续步骤——这是 early-exit。
2. **Define Objectives**: expected outcome（explainer/promotional/tutorial），success criteria，output format，constraints
3. **Identify Model Limitations**: 哪些 issue 是 model 固有局限（无法理解 context、特定 visual task、无法生成 audio）而非 prompt 问题
4. **Identify Prompt Issues**: vague terms、scope too broad、conflicting constraints（如 "short but detailed"）、prompt 过于复杂、missing information
5. **Propose Targeted Revisions**: 综合以上提出 modification list
6. **Revise Suggested Modifications**: 复查是否覆盖所有 major issues，否则 revise

### 6.3 Table 9 的 case study

Original prompt 部分内容：
```
'timestamp': '0-5.5', 'scene_type': 'Man asking and answering a trivia question outdoors.'
'timestamp': '5.5-8', 'scene_type': 'Outro screen with branding and call to action.'
```

VISTA suggested modifications（DTPA 输出）：
- "Update the scene's text overlays... text overlay should smoothly fade in/slide up from the bottom, be legible..."
- "Refine the 'sounds'... with dialogue free of noticeable wind noise. A subtle, consistent ambient street soundscape..."
- "Add a specific instruction for the transition between the first scene (timestamp '0-5.5') and the second scene (timestamp '5.5-8')..."

**注意这种 modification 的颗粒度**——它不是 rewrite，而是 surgical edit。这是 DTPA 相对 self-refine 的核心优势：preserve content，patch specific failure modes。

### 6.4 Ablation 验证 DTPA 的价值

Table 3 w/o DTPA：
- Single-scene: 35.5 → 37.8 (init→5)，仅 +2.3
- Multi-scene: 35.3 → 45.2，+9.9

而 full VISTA：
- Single: 35.5 → 45.9，+10.4
- Multi: 37.8 → 46.3，+8.5

有意思的是 multi-scene w/o DTPA 居然和 full 持平接近——说明在复杂场景里，critique quality 比 prompt revision strategy 更关键；但 single-scene 里 DTPA 的 surgical edit 提升明显（+10.4 vs +2.3）。

---

## 7. 实验数据深度解读

### 7.1 Baselines 设置

4 个 baseline（Section 4）：
1. **DP** (Direct Prompting): 直接用 user prompt
2. **VSR** (Visual Self-Refine, Madaan 2023): MLLM 评估后 iteratively refine
3. **Rewrite** (Google Cloud 2024): 按 Vertex AI guidelines rewrite prompt
4. **VPO** (Cheng 2025b): expand based on harmlessness/accuracy/helpfulness

公平性控制：Rewrite 和 VPO 单次运行，VISTA 跑 4 次 self-improvement iteration，所以引入 **VSR++** 和带 † 的 scaled baselines，通过 sampling 匹配 VISTA 总 video 数。

### 7.2 主实验结果（Table 2）

**Single-scene (MovieGenVideo 100 prompts)**：

VISTA vs DP at iteration 5: Win 45.9% / Tie 50.2% / Loss 13.9%，**Δ = +32.0%**

VISTA vs baselines (iter 5):
- vs VSR++: 30.3% Win / 57.9% Tie / 11.8% Loss, Δ=+18.5%
- vs Rewrite: 40.2% Win / 41.4% Tie / 18.4% Loss, Δ=+21.8%
- vs VPO: 35.7% Win / 45.7% Tie / 18.6% Loss, Δ=+17.1%

**Multi-scene (161 prompts)**：

VISTA vs DP at iteration 5: 46.3% / 47.2% / 13.4%, **Δ = +35.1%**

**关键观察**：baselines vs DP 结果 inconsistent——VPO 在 single-scene Δ=4.0%，在 multi-scene Δ=-16.3% (Rewrite)。这印证 paper 的 thesis：naive prompt optimization 在 multi-scene 上 backfire，因为它们没有 joint multi-modal optimization。

### 7.3 Conventional metrics（Table 5, 6）

| Method | Dynamic Degree (single) | Aesthetic Quality | CLIP-Score | Audio Noisiness (越低越好) |
|---|---|---|---|---|
| DP | 75.95 | 61.86 | 0.310 | 1.74 |
| VSR | 64.56 | 63.45 | 0.309 | 1.73 |
| Rewrite | 77.22 | 62.52 | 0.310 | 1.64 |
| VPO | 77.22 | 61.17 | 0.311 | 1.70 |
| **VISTA** | **89.87** | **64.53** | **0.358** | **1.88** |

Dynamic Degree 提升 +14 abs 是 striking 的——说明 VISTA 不只是让 video"更漂亮"，而是让 motion 更丰富真实。CLIP-Score 提升 +0.048（+15% relative）也说明 text-video alignment 实质性改善。

注：Audio Noisiness VISTA 反而略高（1.88 vs 1.74），但 Audio Discontinuity 改善（multi-scene 2.69 vs 2.62）。Paper 解释这是 trade-off：VISTA 鼓励 ambient sound 引入轻微 noise 但避免 discontinuity。

### 7.4 Human Evaluation（Figure 4, Table 7）

- **Win rate vs best baseline**: 66.4% (VISTA) vs 33.6%
- **Self-improvement score** (1-5): VISTA 3.78 vs VSR++ 3.33
- **Visual quality** (1-5): VISTA 3.77 vs DP 3.36
- **Audio quality** (1-5): VISTA 3.47 vs DP 3.21

5 个 annotator 的一致性高（Ann.1 76%, Ann.5 66%），验证 automatic eval 的 reliability。

### 7.5 Veo 2 实验（Table 4）—— 弱模型泛化

Veo 2 + VISTA：
- Single-scene: Init 15.0% → iter 5 23.8%, +8.8
- Multi-scene: Init 27.6% → iter 5 33.3%, +5.7

Gain 比 Veo 3 小（Veo 3 single-scene +10.4），paper 解释 "Veo 2 being less capable to fully leverage the details optimized by VISTA"。这和 LLM 上 weak base model + strong prompting 也受益但 gain 较小的现象一致。

### 7.6 Cost Analysis（Figure 5）

Per iteration:
- ~0.7M tokens
- ~28 videos
- Most tokens 在 tournament selection（each video input > 2K tokens）

Scaling trajectory：win rate 从 init 35% → 46.1%，token 和 video 用量线性增长时性能持续提升——证明 test-time scaling 在 video 上 work。

### 7.7 Ablation 汇总表（Table 3 重读）

| Configuration | Single Init→5 | Multi Init→5 | Insight |
|---|---|---|---|
| Full VISTA | 35.5→45.9 | 37.8→46.3 | - |
| w/o PromptPlanner | 25.2→35.1 | 34.0→38.8 | Planner 对 init 贡献最大（+10 single） |
| w/o PairwiseSelect | 24.5→33.3 | 27.9→33.8 | Unstable，后期跌 |
| only Adversarial | 35.0→42.0 | 35.3→26.7 | Single OK, Multi collapse |
| only Normal | 35.0→17.2 | 35.3→33.3 | Single collapse, Multi 平庸 |
| w/o DTPA | 35.0→37.8 | 35.3→45.2 | Single gain 砍半 |

**核心 takeaway**：四个 component 各有 distinct role，没有一个是 redundant 的。**Normal + Adversarial 的结合是最 critical 的**——两者 alone 都会 collapse。

---

## 8. 与相关工作的 positioning

### 8.1 Test-time scaling 谱系

- **Self-Refine** (Madaan 2023, NeurIPS): https://openreview.net/pdf?id=S37hOerQLB — 单 agent self-feedback，文本场景。VISTA 是它的 video analog 但解决多模态耦合。
- **s1: Simple test-time scaling** (Muennighof 2025): https://arxiv.org/pdf/2501.19393 — LLM reasoning scaling。VISTA 把这思路搬到 video。
- **Maestro** (Wan 2025b, Google): https://arxiv.org/pdf/2509.10704 — 同作者团队 image generation 的 multi-agent 前作。VISTA 是 video 版本，处理 temporal 维度。
- **APO** (Pryzant 2023, EMNLP): https://aclanthology.org/2023.emnlp-main.494/ — prompt optimization via "gradient descent"。VISTA 的 DTPA 可视为 APO 的 multi-modal 扩展。

### 8.2 Video evaluation 谱系

- **VBench** (Huang 2024, CVPR): https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench... — 8 个 visual metric，无 audio。
- **EvalCrafter** (Liu 2024a, CVPR): https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_EvalCrafter... — 多维但无 audio failure reasoning。
- **VideoScore** (He 2024, EMNLP): https://aclanthology.org/2024.emnlp-main.127/ — 多 agent eval 但 not failure-focused。
- **TAVGBench** (Mao 2024, ACM MM): https://openreview.net/forum?id=hCbSq4rpHq — audio-visual 但 rigid embedding similarity。
- **VideoPhysics** (Bansal 2024): https://arxiv.org/pdf/2406.03520 — physical commonsense，VISTA 借用其 metric。
- **Video-MME** (Fu 2025, CVPR): https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Video-MME... — 证明 MLLM 视频理解能力，支撑 MLLM-as-judge 合理性。

### 8.3 Video generation 谱系

- **Veo 3** (Google DeepMind 2025): https://deepmind.google/models/veo/ — VISTA 默认 backend，audio-video 联合生成 SOTA。
- **MovieGen** (Polyak 2025): https://arxiv.org/abs/2410.13720 — benchmark source + baseline model。
- **Wan** (Wan 2025a): https://arxiv.org/pdf/2503.20314 — open-source T2V。
- **Sora** (OpenAI 2024): https://openai.com/index/sora — 引用对比。
- **CogVideo** (Hong 2023, ICLR): https://openreview.net/forum?id=rB6TpjAuSRy — 早期 T2V。

### 8.4 Multi-agent 系统

- **FilmAgent** (Xu 2025): https://arxiv.org/pdf/2501.12909 — film automation multi-agent，但无 test-time optimization。
- **Mora** (Yuan 2024): https://arxiv.org/pdf/2403.13248 — video generation multi-agent。

### 8.5 Video optimization prior work

- **VideoAgent** (Soni 2024): https://arxiv.org/pdf/2410.10076 — refine plans but 需要 fine-tune generation model。
- **MotionPrompt** (Nam 2025, CVPR): https://openaccess.thecvf.com/content/CVPR2025/papers/Nam_Optical-Flow... — learns token embeddings，white-box。
- **RAPO** (Gao 2025a, CVPR): https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_The_Devil... — retrieval-augmented prompt optimization，需要 target prompts training。
- **VPO** (Cheng 2025b): https://arxiv.org/pdf/2503.20491 — harmlessness/accuracy/helpfulness alignment，not test-time。

VISTA 是**第一个 black-box test-time prompt optimization for video**。

---

## 9. Limitations 和我自己的 critique

Paper 自己列了三个 limitations：
1. Evaluation 依赖 MLLM 和 automatic metrics 有 systematic bias
2. Default metrics 可能不 generalize across cultures/styles
3. 依赖 strong MLLM 和 T2V

我补充几点 observation：

### 9.1 计算成本现实性
0.7M tokens × 5 iterations = 3.5M tokens per prompt，加上 ~140 videos。这对 production deployment 是 heavy 的。Gemini 2.5 Flash pricing 大约 $0.15/1M input token，单 prompt 优化成本 ~$1，加上 Veo 3 generation 成本（每 video 估 $0.05-0.1），单 prompt 总成本 ~$10。这在 high-value creative 场景可接受，但 consumer scale 难。

### 9.2 Early stop 没充分验证
Algorithm L9-11 的 early stop 用 patience $m$，但 paper 没有报告 early stop 触发率。从 Figure 6 的 20-iteration scaling 看，性能持续增长说明 early stop 触发率低——那它存在意义？这值得进一步分析。

### 9.3 Pairwise tournament 的 noise
Algorithm 2 L7 在 forward/swapped conflict 时"assign randomly"——这会引入随机性。论文没分析这个 random tie-break 对最终结果 variance 的贡献。在小 sample size（n=5 prompts × 3 variants × 2 videos = 30）下，tournament 结构对 noise 敏感。

### 9.4 Critique 和 selection criteria 解耦
$\mathcal{M}_{user}^S$ (selection) 和 $\mathcal{M}_{user}^C$ (critique) 默认配置不同——selection 5 个 criteria，critique 14 个。这意味着优化方向和评估方向可能 misalign。Paper 没讨论这种 misalign 的风险，例如 critique 找到 camera focus 问题但 selection 不权重 camera focus。

### 9.5 Adversarial judge 的 failure mode
Adversarial judge 在 multi-scene alone 时 collapse 到 18.8%。Paper 没分析为什么——我猜是 over-critical 导致 optimizer 把 prompt 越改越复杂，最终 violate Realism/Relevancy constraint。这暗示 Meta judge 的 design 极其重要，但 paper 没给 Meta judge 的 prompt 细节（只有 section B.4 给了 Meta judge 的 task description）。

### 9.6 Single-scene vs Multi-scene 的本质差异
Ablation 显示 PromptPlanner 对 single-scene Init 贡献 +10（35.5 vs 25.2）但对 multi-scene 只 +3.8（37.8 vs 34.0）。这暗示 single-scene 的 win 主要来自 structured prompt format 本身（Veo 3 对 structured prompt 更敏感），而 multi-scene 的 win 来自 MMAC+DTPA 的迭代。这值得专门拆开分析。

---

## 10. 与你的工作的潜在连接点

### 10.1 与 LLM101 / nanoGPT 直觉的呼应
VISTA 的本质是 **inference-time search over structured prompt space**，和 LLM 上的 best-of-N、beam search 思想相通，但 search space 是 multi-modal structured object。这种 "search over outputs that are themselves prompts to another model" 的两层 search 结构，和你在 nanoGPT 强调的 "model 是 token sampler，外面是更高层 abstraction" 的直觉一致。

### 10.2 与 test-time compute scaling 论文 (Snell 2025) 的连接
Snell et al. 2025 (ICLR, https://openreview.net/forum?id=4FWAwZtd2n) 显示 LLM 上 test-time compute 在 reasoning 任务上比 parameter scaling 更高效。VISTA Figure 6 显示 video 上同样 work——iteration 增加性能持续提升。这指向一个 unified view：**只要 task 有可验证的 improvement signal（这里通过 MLLM-as-judge pairwise），test-time scaling 就能 work**，和 modality 无关。

### 10.3 与 Eureka Labs / education 直觉
你多次强调 "AI tutor 在 education 上需要 test-time 适应学生"。VISTA 的 multi-agent critique + meta judge 架构可以平移到 education——Normal judge 评估 student understanding，Adversarial judge 找 misconception，Meta judge 综合，DTPA 改 explanation。这是个潜在 application direction。

### 10.4 与 self-play / AlphaGo 直觉
VISTA 的 Normal + Adversarial Judge 本质是 self-play 的 critic side——一个 advocate 试图证明 video 好，另一个试图证明 video 差，meta judge 仲裁。这和 AlphaGo 的 policy/value network 双角色，以及 Debate AI (Irving et al. 2018) 的 design philosophy 一致。Paper 引用 Klevorick 1979 Jury Decision Process 是一个 elegant 的法律系统类比，但底层是相同的 adversarial self-play intuition。

---

## 11. 可复现性 checklist

如果你要 reproduce 或扩展：

1. **MLLM**: Gemini 2.5 Flash (Gemini-2.5-flash-preview-05-20) — https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/
2. **T2V**: Veo 3 (Veo-3.0-generate-preview) — https://deepmind.google/models/veo/
3. **Benchmarks**:
   - Single-scene: MovieGenVideo (Polyak 2025) 100 random prompts — https://arxiv.org/abs/2410.13720
   - Multi-scene: 161 internal prompts（这部分不可复现，需自建）
4. **Iteration config**: 5 prompts × 3 variants × 2 videos = 30 videos per iteration，5 iterations total
5. **Eval MLLM**: Gemini 2.5 Flash + Qwen2.5-VL-32B-Instruct + Gemini 2.5 Pro 都验证过（Table 10）
6. **Metrics**: VBench (https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench...) + NISQA (https://www.isca-archive.org/interspeech_2021/mittag21_interspeech.pdf) + CLIP-Score

Prompt 模板都在 Appendix B，是相当 reproducible 的——除了 multi-scene internal dataset 之外。

---

## 12. 总结 intuition

VISTA 的 elegance 在于它**把 human 视频制作的 implicit process 显式化为 agent workflow**：
- Director 拆解剧本（PromptPlanner）
- Producer 选最佳 take（Pairwise Tournament）
- 三人评审团从对立角度批评（MMAC: Normal + Adversarial + Meta）
- 编剧 introspective 修订（DTPA）

每个 component 单独都不新颖——structured prompting、tournament selection、adversarial debate、CoT 都已存在。**Novelty 在 composition + 适配 video 多模态特殊性的工程细节**：保留 residual candidate 防 drift、probing critique 先于 comparison 降 overload、Normal+Adversarial balance 防 collapse、DTPA surgical edit 防过修改。

最终 60% pairwise win rate 和 66.4% human preference 证明 video test-time scaling 不是 trivially extrapolated from text/image，需要这种 carefully designed multi-agent decomposition。

潜在 follow-up direction：
1. 把 VISTA 用于 long video（>1min），需要 hierarchical scene planning
2. 加入 user preference model 让 $\mathcal{M}_{user}^S$ / $\mathcal{M}_{user}^C$ 个性化
3. Replace tournament with learned value function 降 token cost
4. 把 DTPA 和 reinforcement fine-tuning 结合——用 VISTA 的 critique 训 reward model

希望这个讲解能 build 你对 video test-time scaling 当前 frontier 的 intuition。如果你想深入某个具体 component（比如 MMAC 的 Meta judge prompt 设计，或 PairwiseSelect 的 noise analysis），可以继续深挖。
