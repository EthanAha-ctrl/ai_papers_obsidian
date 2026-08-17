---
source_pdf: WorldPrediction.pdf
paper_sha256: b3c51326a1c09c2dd8c24bfcd51fdffcaa497e2f9100d19c209531cf77720d66
processed_at: '2026-08-13T05:49:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 Paper

## 一句话概括

给模型看两张图——"之前"和"之后"，再给它几个candidate视频片段，问它：哪个视频里做的action能从"之前"变成"之后"？结果最好的模型只答对57%，人类100%。

## 为什么这件事重要

现在大家都在吹world model。Sora能生成视频，Cosmos能模拟物理，机器人能抓东西。但这些都是**low-level**的——帧级别的pixel prediction，或者固定频率的控制信号。

真正的人类活动是**high-level**的。你说"装宜家家具"，这中间有"拧螺丝"、"装桌腿"、"翻面板"这些动作，每个动作长度不一样，每个动作又包含一堆肌肉运动。这叫semantic abstraction + temporal abstraction。

之前的benchmark要么只测3-4步的短序列，要么只能eval video generation model，要么必须用text label。没有一个能公平比较所有类型world model的benchmark。

这篇paper就是填这个坑。

## 任务怎么设计的

### WorldPrediction-WM（World Modeling）

```
输入：
  - 图A：初始状态（比如生牛排放在盘子里）
  - 图B：最终状态（比如熟牛排放在烤架上）
  - 4个候选视频片段

输出：选哪个视频做的action能完成 A→B 的transition
```

### WorldPrediction-PP（Procedural Planning）

```
输入：
  - 图A：初始状态（食材散放）
  - 图B：最终状态（做完的三明治）
  - 4个candidate action序列（每个序列3-10个视频片段，顺序不同）

输出：选哪个序列的action顺序是对的
```

全是visual，没有text label。这是跟之前benchmark最大的区别。

## 最clever的设计：Action Equivalents

这是整篇paper我最喜欢的地方。

**问题**：如果candidate视频里的ground truth和初始/最终状态的图来自同一个视频，模型根本不需要理解action在干什么——它只要匹配background、lighting、camera angle这些low-level特征就行了。这是shortcut learning。

**解法**：用"action equivalents"替换ground truth action video。同一个动作类别（比如"切土豆"），从完全不同的视频里找——不同的厨房，不同的camera视角，甚至egocentric和exocentric视角互换。

```
原始ground truth：厨房A里切土豆（和状态图A、B来自同一段视频）
替换后：          厨房B里切土豆（完全不同的环境）
```

Distractors也从厨房B的环境里采样。这样模型就没法靠background走捷径了，必须真的理解"切土豆"这个动作的semantic content。

这个设计本质上是在task-irrelevant维度上inject noise，强迫模型只依赖task-relevant的causal features。

## 另一个clever设计：Observability Filtering

POMDP的核心假设是partial observability——你看到的不是真正的state。如果两张状态图变化太剧烈（比如camera突然切到完全不同的场景），根本无法infer causality。

用DINOv2算两张图在feature space的L2距离：
$$d = \|\phi(\mathcal{O}(s_{\text{init}})) - \phi(\mathcal{O}(s_{\text{final}}))\|_2$$

距离太大的直接丢掉。阈值WM用2.75，PP用10。粗糙但有效的heuristic——连续的real-world state不应该在feature space里突然跳变。

EgoExo4D还要额外过滤人背对camera、手被挡住的样本——这些case里关键信息根本看不到。

## 结果说了什么

| Model类型 | 最好成绩(WM) | 最好成绩(PP) | 人类 |
|-----------|-------------|-------------|------|
| VLMs | 57.0% (Qwen2.5-VL 72B) | 36.7% (Qwen2.5-VL 72B) | 100% |
| Socratic LLMs | 55.6% (Gemini-2.0) | 38.1% (Claude-3.5) | 100% |
| Video Diffusion | 30.5% (CogVideoX+DINOv2) | N/A | - |
| OEPP (专门训练的planner) | N/A | 36.8% (MLP) | - |

### 几个关键insights

**1. WM有scale effect，PP没有**

InternVL2.5从26B到38B，WM涨了20个百分点。但PP几乎不随scale变化。这说明single-step的perception能力能scale，但multi-step的causal chain reasoning不能靠堆参数解决。

**2. Video diffusion做得很烂**

CogVideoX只有30%，I2VGenXL只有26%。这很能说明问题。Pixel space generation根本不等于理解world dynamics。你能生成一帧看起来合理的画面，不代表你知道从state A到state B需要什么action。

用DINOv2 features替代RGB来做candidate selection，几乎没改善。说明问题不在最后一层metric，在于generation过程本身没有capture causal structure。

**3. Socratic LLMs出奇地好**

只用VLM生成caption，然后text-only LLM做reasoning，结果跟VLM直接做comparable。这说明当前阶段，visual grounding可能还不是bottleneck，reasoning capability才是。

但更值得深思的是：best WM model (Gemini) 不是best PP model (Claude)。WM更依赖perception quality，PP更依赖reasoning capability。

**4. OEPP的in-domain vs out-of-domain差距巨大**

专门训练的planner在COIN/CrossTask上能达到49.2%，但在其他数据集上只有29%。典型的overfitting。但给oracle caption（human annotation）时，性能跳到70.6%——说明planning算法本身没问题，bottleneck在perception。

## 理论框架为什么用POSMDP

POSMDP = Partially Observable Semi-MDP

- **Partially Observable**：对应图像和视频只提供partial view of true state，有occlusion、viewpoint限制
- **Semi-MDP**：对应high-level action跨越non-uniform duration，每个action包含多个low-level primitive——这就是Sutton的options framework

Tuple: $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{O} \rangle$

- $\mathcal{S}$：latent world states，真正的环境配置，无法直接access
- $\mathcal{A}$：high-level action vocabulary，每个action是abstract category
- $\mathcal{T}(s_{t+1} | s_t, a_t)$：true transition model，hidden的，world model要学的就是approximate它
- $\mathcal{O}$：observation model，把latent state映射成image，把action映射成video

这个formalism不是为了炫技，它直接guide了benchmark的设计：
- partial observability → observability filtering
- task-relevant vs task-irrelevant state components → action equivalents
- options framework → high-level action的temporal abstraction

## 我觉得这篇paper最核心的contribution

**Discriminative formulation + action equivalents 的组合**

这个设计让benchmark变得architecture-agnostic。不管你是：
- Predictive model（JEPA-style，latent space prediction）
- Generative model（diffusion，pixel space generation）
- VLM（直接encode visual observations）
- Socratic LLM（先caption再reasoning）
- 专门训练的planner（OEPP）

都可以在这个benchmark上公平比较。只要能输出一个score或选一个candidate就行。

之前的benchmark做不到这点——WorldScore只能eval video generation，PlanBench只能eval text-based planning。

## 对未来的implications

**1. Latent world model有大机会**

Video diffusion的30% vs VLM的57%直接证明LeCun一直说的：pixel space generation在task-irrelevant details上浪费capacity。JEPA-style的latent prediction可能更sample-efficient，也更aligned with真正的world understanding。

这个benchmark的discriminative formulation天然适合latent models——不需要generate pixel，只需要输出likelihood score。

**2. Hierarchical planning是必须的**

PP的scale insensitivity说明纯靠一个monolithic model做long-horizon planning不行。需要显式的hierarchical架构：
- Low-level：latent world model做single-step prediction
- High-level：LLM-based planner做long-horizon reasoning
- 中间：learned state abstraction做bridging

**3. Action equivalents应该被用作training signal**

不只是evaluation时防shortcut。训练时也应该用这个idea——同一个action的不同环境实例作为augmentation，强制模型学习action的invariance。

## 局限性

**1. Action equivalent的quality**

"相同textual label"作为equivalent的标准太粗。两个"cut potato"视频可能在cutting technique上差很多。这会引入label noise。

**2. 没有intermediate states**

PP task完全不提供中间状态，强制模型内部infer。这接近real world但让debug变难——不知道是single-step WM失败还是multi-step chaining失败。

**3. Plan length分布不均**

PP样本在length 3-10分布，但3-4占majority。模型可能在short plan上表现好掩盖了long-horizon reasoning的缺陷。

**4. Visual grounding的necessity存疑**

Socratic LLMs的competitive performance暗示：当前scale下，也许textual description就够用了。这对"我们需要真正visual world model"的thesis是counter-evidence。不过long term来看，embodied agent必须跨越这个gap。

---

**Bottom line**: 这篇paper给出了一个clean、architecture-agnostic、theoretically grounded的benchmark。结果显示当前frontier models在high-level world modeling上还很弱——最好的57% vs人类100%。Video diffusion的poor performance直接挑战了"generate video = understand world"的narrative。PP的scale insensitivity说明long-horizon reasoning需要不同的capability，不能靠堆参数解决。这个benchmark很可能成为world model研究的standard evaluation。

参考链接：
- Paper: https://arxiv.org/abs/2506.10975
- LeCun world model position paper: https://openreview.net/forum?id=BZ6aWDu1G
- Sutton options framework: https://arxiv.org/abs/1301.3754
- DINOv2: https://arxiv.org/abs/2304.07193
- Cosmos: https://arxiv.org/abs/2501.03575
- V-JEPA: https://arxiv.org/abs/2402.05792

---

# WorldPrediction Paper 深度解析

Andrej, 这篇 paper 挺有意思的，因为它直接挑战了当前 frontier models 在 high-level world modeling 上的能力边界。让我分层拆解。

## 1. Core Motivation 与 Problem Framing

当前 world model 的研究主要聚集在 low-level dynamics——robotics manipulation, autonomous driving, navigation——这些都涉及 fixed-frequency control 和精确的 physical dynamics。但 human skilled activities（cooking, furniture assembly, healthcare procedures）需要的是 **semantic abstraction** 和 **temporal abstraction**：单个 high-level action（如 "remove the battery"）跨越 non-uniform duration，并 encapsulate 多个 low-level motor primitives。

WorldPrediction 的核心 insight：现有 benchmark 要么 task-specific（PlanBench, COIN planning），要么 architecture-coupled（WorldScore 只能 eval video generation），要么依赖 text labels。这导致无法公平比较 predictive models（JEPA-style latent prediction）和 generative models（diffusion in pixel space），也无法评估真正从 observation 出发的 world modeling 能力。

Reference links:
- Yann LeCun's world model manifesto: https://openreview.net/pdf?id=BZ6aWDu1G
- JEPA (Joint Embedding Predictive Architecture): https://arxiv.org/abs/2301.08243
- PlanBench: https://arxiv.org/abs/2206.10498
- WorldScore: https://arxiv.org/abs/2504.00983

## 2. Theoretical Formulation: POSMDP

这个 benchmark 的理论基础是 **Partially Observable Semi-Markov Decision Process**，tuple 形式为 $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{O} \rangle$。

让我逐项拆解：

### 2.1 World States $\mathcal{S}$
$s \in \mathcal{S}$ 表示环境的完整 latent configuration。这里的关键区分：
- **Task-relevant components**：直接影响 action 的 causal outcomes
- **Task-irrelevant components**：background details，不影响 task

这个区分非常重要，因为它是后续 "action equivalents" 设计的理论依据——如果不区分 task-relevant 和 task-irrelevant，模型就会依赖 spurious correlations。

### 2.2 High-level Actions $\mathcal{A}$
$\mathcal{A} = A_1, A_2, \ldots, A_N$ 是 action category vocabulary。每个 $A_i$ 是 abstract category（如 "cut potato"），而 $a \in A_i$ 是在特定 context $s$ 下执行的具体 action instance。

这里借用了 Sutton 的 **options framework**（Semi-MDPs）：
- 每个 option = (policy over low-level primitives, termination condition, initiation set)
- 这就让 high-level action 有了 temporal abstraction 的 formal grounding

Reference: Sutton, Precup, Singh 1999 "Between MDPs and Semi-MDPs": https://arxiv.org/abs/1301.3754 (原 paper 在 Artificial Intelligence journal)

### 2.3 Transition Model $\mathcal{T}$
$$\mathcal{T}(s_{t+1} \mid s_t, a_t)$$

这是 hidden 的 true dynamics。World model 的目标就是学习一个 $\mathcal{W}$ 来近似 $\mathcal{T}$。在 real-world 场景中，$\mathcal{T}$ 不可直接 access，agents 必须 approximate it。这点和 LeCun 的 JEPA 哲学完全一致——predict in latent space, not pixel space。

### 2.4 Observation Model $\mathcal{O}$
$$\mathcal{O}(s_t) \to \text{image}, \quad \mathcal{O}(a_t) \to \text{video segment}$$

由于 occlusion, resolution, viewpoint constraints，observation 只提供 partial view。这直接 motivate 了 benchmark 中的 **observability filtering**。

### 2.5 Data-generative Process
$$[\mathcal{O}(s_0), \mathcal{O}(a_0), \mathcal{O}(s_1), \mathcal{O}(a_1), \ldots, \mathcal{O}(s_T)]$$

这个序列是 benchmark 中每个 sample 的结构基础。

## 3. Task Formulation: Discriminative 而非 Generative

### 3.1 WorldPrediction-WM
理论目标：
$$\mathcal{D}\left(\mathcal{W}(s_{t+1} \mid s_t, a_t) \parallel \mathcal{T}(s_{t+1} \mid s_t, a_t)\right) \tag{1}$$

其中 $\mathcal{D}$ 是 divergence metric，$\mathcal{W}$ 是 learned world model，$\mathcal{T}$ 是 true transition model。

经验近似（discriminative form）：
$$A^* \stackrel{?}{=} \arg\max_{A \in \mathcal{A}} \mathcal{W}(s_{t+1} \mid s_t, A) \tag{2}$$

- $A^*$：true action category
- $\mathcal{A}$：action vocabulary
- $\mathcal{W}$：learned transition model
- $s_t, s_{t+1}$：initial 和 final world states

这个 formulation 的精妙之处：它 probe 的是 $\mathcal{W}$ 对 $(s_t, a) \to s_{t+1}$ causal relationship 的 capture 程度，而非要求 model 精确 reconstruct 某个 specific plan。real-world activities 中多个 valid solutions 都能 achieve 同一 goal，discriminative setup 容纳了这种 intrinsic variability。

### 3.2 WorldPrediction-PP
$$\mathcal{P}^* \stackrel{?}{=} \arg\max_{\mathcal{P} \in \mathcal{A}^T} \mathcal{W}(s_{\text{final}} \mid s_{\text{init}}, \mathcal{P}) \tag{3}$$

- $\mathcal{P}^* = (a_1, \ldots, a_T)$：correct ordered action sequence
- $\mathcal{A}^T$：所有可能的 T-step 排列组合
- $T \in [3, 10]$：plan length

如果所有 intermediate states $(s_2, \ldots, s_{T-1})$ 都 known，PP 就 reduce 成 $T$ 个 successive WM steps。但 intermediate states unobserved，所以 model 必须 internally infer 它们——这强制 model reason about 整个 multi-step causal chain。

## 4. Action Equivalents 与 Shortcut Mitigation

这是 benchmark 设计中最 clever 的部分。

### 4.1 问题
如果 ground-truth action segment 和 state observations 共享相同的 camera viewpoint、background objects 等 task-irrelevant features，model 可以仅通过 low-level feature matching 来 identify correct action，而无需真正理解 action-state causality。

### 4.2 解决方案
对每个 high-level action category $A_i$，存在一组 observations 描绘它在 visually 不同 environments 或不同 viewpoints 下的执行。具体做法：

- **COIN, CrossTask, EPIC-KITCHENS-100, IKEA-ASM**：sharing 同 textual label 的 actions 构成 equivalents
- **EgoExo4D**：通过 consecutive timestamps 的 midpoints segmentation（discard <5s segments），select egocentric view for actions（clear hand movements）和 exocentric viewpoints for state observations（comprehensive scene coverage）

然后用 action equivalent 替换 ground-truth observation action，并 re-sample distractors from 同一 environment。

这个设计本质上是在 task-irrelevant dimensions 上 inject noise，强制 model 只能依赖 task-relevant causal features。这和 contrastive learning 中 hard negative mining 的哲学有共通之处。

## 5. Observability Filtering

### 5.1 Feature Distance Filtering
$$d = \|\phi(\mathcal{O}(s_{\text{init}})) - \phi(\mathcal{O}(s_{\text{final}}))\|_2$$

- $\phi(\cdot)$：pretrained vision encoder（DINOv2）
- $d$：两个 state observations 在 feature space 的 distance

Thresholds: WM=2.75, PP=10。超过 threshold 的 samples 被剔除，因为 scene 变化太 drastic，无法可靠 infer causal link。

### 5.2 Rationale
这基于 POMDP 的 assumption：consecutive observations of 同一 environment 不应该 appear uncorrelated if 它们 reflect smoothly evolving states。这个 filtering 等价于一个 coarse classifier，eliminate 大部分 bad state observations。

### 5.3 Occlusion Filtering
For EgoExo4D, 用 VLM prompt："Is the main person not showing their back and what they are doing with hands being clearly visible?" 来过滤 heavily obstructed samples。

Reference: DINOv2: https://arxiv.org/abs/2304.07193

## 6. Dataset Construction

### 6.1 Sources
| Dataset | Type | Domain Focus |
|---------|------|--------------|
| COIN | Instructional web videos | Cooking, household repairs |
| CrossTask | Instructional web videos | Everyday activities |
| EgoExo4D | Egocentric + multi-view exocentric | Cooking, healthcare |
| EPIC-KITCHENS-100 | Egocentric | Kitchen tasks |
| IKEA-ASM | Exocentric | Furniture assembly |

Dataset links:
- COIN: https://coin-dataset.github.io/
- CrossTask: https://github.com/DmZhukov/CrossTask
- EgoExo4D: https://egoexo4d-data.org/
- EPIC-KITCHENS-100: https://epic-kitchens.github.io/2022
- IKEA-ASM: https://github.com/StanfordVL/IKEAASM

### 6.2 Statistics
WorldPrediction-WM: 825 samples, 1800 unique actions, avg duration 10.02s
WorldPrediction-PP: 570 samples, 749 unique actions, avg duration 9.38s

### 6.3 Distractor Sampling
- **WM**：每个 correct action 配 3 个 distractors，共 4 个 candidates。Distractors 来自 same task context（同一 video）但 incompatible with observed state transition
- **PP**：distractors 通过 shuffle ground-truth action sequences 生成，preserving action-level plausibility 但 disrupting temporal correctness

## 7. Baseline Models 分析

### 7.1 VLMs
Qwen2.5-VL 和 InternVL2.5 系列。Prompt 结构包含 initial/final state images + candidate action video segments + textual instructions。

### 7.2 Socratic LLMs
两阶段架构：
1. VLM (Qwen2.5-VL 72B) 生成 textual descriptions
2. Text-only LLM 进行 reasoning

测试了 Llama-3.1, Qwen2.5, DeepSeek-R1, GPT-4o, Claude-3.5-Sonnet, Gemini-2.0。

### 7.3 Video Diffusion Models
I2VGenXL 和 CogVideoX-I2V。流程：
1. Initial state image $\mathcal{O}(s_t)$ 作为 conditioning
2. VLM 对每个 action candidate 做 captioning
3. 生成 video segments
4. 选择 last frame 与 $\mathcal{O}(s_{t+1})$ pixel-wise distance 最小的 candidate

Reference:
- I2VGenXL: https://arxiv.org/abs/2311.04145
- CogVideoX: https://arxiv.org/abs/2408.06072

### 7.4 OEPP Models
Open-Event Procedural Planning，使用 VideoCLIP embeddings。Planning model (MLP / Transformer / PDPP) 生成 T 个 text embeddings 对应 T 个 predicted actions，然后 select 与 candidate plans distance 最小的。

Reference:
- OEPP: https://arxiv.org/abs/2407.05119
- VideoCLIP: https://arxiv.org/abs/2107.06250
- PDPP: https://arxiv.org/abs/2303.07976

## 8. Experimental Results 深度解读

### 8.1 WorldPrediction-WM Results

| Model Family | Best Model | Accuracy |
|--------------|------------|----------|
| VLMs | Qwen2.5-VL (72B) | 57.0% |
| Socratic LLMs | Gemini-2.0 | 55.6% |
| Video Diffusion | CogVideoX + DINOv2 | 30.5% |
| Human | - | ~100% |

### 8.2 WorldPrediction-PP Results

| Model Family | Best Model | Accuracy |
|--------------|------------|----------|
| VLMs | Qwen2.5-VL (72B) | 36.7% |
| Socratic LLMs | Claude-3.5-sonnet | 38.1% |
| OEPP | MLP | 36.8% |
| Human | - | ~100% |

### 8.3 关键 Insights

**Insight 1: Scale Threshold for WM**
InternVL2.5 从 26B 到 38B 有 ~20% 的 jump（30.2% → 50.3%）；Qwen2.5-VL 从 3B 到 7B 有 ~24% 的 jump（21.6% → 45.5%）。这暗示 world modeling 需要某个 critical scale 才能 emerge。

**Insight 2: PP 没有 Scale Benefit**
Long-horizon procedural planning 没有显示显著的 model size benefit。这非常有意思——single-step world modeling 能力可以 scale，但 multi-step causal chain reasoning 似乎需要不同的 capability（可能是 explicit reasoning 或 latent state inference）。

**Insight 3: Socratic LLMs 的 Trade-off**
Socratic LLMs（用 Qwen2.5-VL 72B 做 captioning）的性能 comparable to VLMs。但 best WM model (Gemini-2.0, 55.6%) 不是 best PP model (Claude-3.5, 38.1%)。这暗示：
- WM 更多依赖 perception quality
- PP 更多依赖 reasoning capability
- Visual grounding 和 explicit reasoning 之间存在 trade-off

**Insight 4: Video Diffusion 的局限性**
CogVideoX 只有 30.1%，I2VGenXL 只有 26.1%。这表明 pixel-space generation 难以 capture detailed action-state causal relationships。DINOv2 features 替代 RGB 用于 candidate selection 也几乎没改善。这印证了 LeCun 的论点：generative models in pixel space 浪费 capacity 在 task-irrelevant details 上。

**Insight 5: OEPP 的 In-domain vs Out-of-domain**
PDPP 在 in-domain (COIN, CrossTask) 达到 49.2%，但 out-of-domain (EgoExo4D, EPIC-KITCHENS-100, IKEA-ASM) 只有 ~29%。这反映 supervised planning models 的 generalization limitation。但当提供 oracle captions（human annotations）时，性能提升到 70.6%——说明 bottleneck 在 perception 而非 planning algorithm 本身。

### 8.4 数据来源 Performance Breakdown

| Planner | COIN, CrossTask | EgoExo4D, E-100, IKEA-ASM | Overall |
|---------|-----------------|---------------------------|---------|
| Qwen2.5-VL (72B) | 37.6 | 35.0 | 36.1 |
| Llama-3.3 (70B) | 34.3 | 41.0 | 37.4 |
| OEPP-MLP | 42.3 | 26.5 | 36.8 |
| OEPP-Transformer | 48.3 | 29.5 | 34.2 |
| OEPP-PDPP | 49.2 | 29.4 | 34.4 |

OEPP 在 in-domain 显著领先（49.2% vs 37.6%），但 out-of-domain 大幅落后（29.4% vs 35.0%）。Socratic LLMs 表现最 balanced。

## 9. Human Evaluation Protocol

- 初始 1500 samples for each task
- 每个样本由 2 个 annotators 独立 solve
- Conservative filtering：只在两个 annotators 都答对时才保留
- Final：825 WM samples, 570 PP samples
- Inter-annotator agreement: WM=0.73, PP=0.65（substantial agreement）
- 34 annotators for WM (avg 88 samples each), 46 for PP (avg 65 samples each)
- Annotator workload: min 20, max 100 samples

## 10. 与 LeCun's Vision 的 Connection

这篇 paper 在多个层面与 LeCun 的 world model vision 呼应：

1. **Latent vs Pixel**: Video diffusion 的 poor performance (30.1%) vs VLMs 的 57.0% 支持 LeCun 的论点：generative models 在 pixel space 浪费 capacity 在 irrelevant details 上。JEPA-style latent prediction 可能更 sample-efficient。

2. **Predictive vs Generative**: Benchmark 的 discriminative formulation 允许公平比较 predictive models（如 V-JEPA, DINO-WM）和 generative models（如 Cosmos, Sora）。

3. **Hierarchical Abstraction**: POSMDP framework 中的 options 和 high-level actions 直接对应 LeCun 的 hierarchical world model 构想。

Reference:
- V-JEPA: https://arxiv.org/abs/2301.08243
- DINO-WM: https://arxiv.org/abs/2411.04983
- Cosmos: https://arxiv.org/abs/2501.03575
- LeCun's position paper: https://openreview.net/forum?id=BZ6aWDu1G

## 11. 局限性与 Open Questions

### 11.1 Action Equivalents 的 Quality
"相同 textual label" 作为 equivalence 的标准可能过粗。两个 "cut potato" 视频可能在 cutting technique、object state、tool usage 上有显著差异，这会引入 noise。

### 11.2 Distractor 的 Hardness
当前 distractors 来自 same video context（WM）或 shuffle（PP）。但没有 measure distractor 的 hardness level。Hard negatives（visually similar 但 causally different）可能更能 probe model 的真实理解。

### 11.3 PP 的 Plan Length Distribution
当前 PP 样本在 length 3-10 之间分布，但 length 3-4 占 majority。如果 model 在 short plans 上表现好，可能掩盖 long-horizon reasoning 的缺陷。

### 11.4 Intermediate States 的 Absence
PP task 完全不提供 intermediate states，强制 model 内部 infer。这接近 real-world scenarios，但也让 evaluation 变得 challenging——无法 diagnose 是 single-step WM 失败 还是 multi-step chaining 失败。

### 11.5 Cross-modal Grounding
Socratic LLMs 的 competitive performance 暗示 visual grounding 可能被 textual reasoning 部分 compensate。这引出 question：visual world modeling 是否真的需要，还是 textual description 足够？

## 12. 与 Related Work 的 Positioning

### 12.1 vs WorldScore (Duan et al., 2025)
WorldScore 专注 video generation quality。WorldPrediction 更广泛，可以 eval 任何 architecture。

### 12.2 vs PlanBench (Valmeekam et al., 2023)
PlanBench 是 text-based planning evaluation。WorldPrediction 是 observation-based，去除 text label 依赖。

### 12.3 vs Procedure Planning in Instructional Videos (Chang et al., 2020)
Chang 的 benchmark 限定 3-4 steps，依赖 text labels。WorldPrediction 扩展到 10 steps，label-free。

### 12.4 vs MMWorld (He et al., 2024)
MMWorld 是 VQA-style，要求 textual outputs。WorldPrediction 是 discriminative。

### 12.5 vs Object State Changes (Xue et al., 2024)
Xue 专注 object state transitions。WorldPrediction 拓展到 dynamic human behaviors in complex environments。

## 13. Technical Implementation Details 的一些 Hallucination 补充

基于 paper 描述，可以推断一些实现细节：

### 13.1 VLM Prompting 策略
Paper 没有给出具体 prompt template，但基于 "structured multimodal query" 的描述，可以推断类似：

```
Given the initial state image [IMAGE_1] and final state image [IMAGE_2], 
which of the following candidate action videos [VIDEO_A, VIDEO_B, VIDEO_C, VIDEO_D] 
correctly depicts the action that transitions the initial state to the final state?
Respond with only the letter (A, B, C, or D).
```

### 13.2 Video Diffusion 的 Captioning
Paper 提到用 VLM 对每个 action candidate 做 captioning。这里可能用的是 Qwen2.5-VL 72B，生成类似 "A person is flipping a steak on a grill" 的 description。

### 13.3 OEPP 的 Embedding Space
VideoCLIP embeddings 是 512-dim。Planning model 输出 T 个 512-dim embeddings，与 candidate plans 的 embeddings 计算 cosine distance。

## 14. 对 Future Research 的 Implications

### 14.1 Latent World Models 的机会
Video diffusion 的 poor performance (30%) vs VLMs 的 57% 强烈暗示 latent predictive models（如 V-JEPA2, DINO-WM 的 extension）有巨大机会。Discriminative evaluation 天然适合 latent models——只需输出 likelihood score。

### 14.2 Hierarchical Planning Architectures
PP 的 scale insensitivity 暗示需要 explicit hierarchical planning architectures。可能的 direction：
- Low-level: latent world model for single-step prediction
- High-level: LLM-based planner for long-horizon reasoning
- Bridging: learned state abstraction

### 14.3 Action Equivalents 作为 Training Signal
Action equivalents 不仅用于 evaluation，也可以作为 training 时的 augmentation——强制 model 学习 action invariance across environments。

### 14.4 Active World Modeling
当前 benchmark 是 passive evaluation。未来可以扩展为 interactive setting：model 可以 query specific viewpoints 或 zoom in on specific regions。

## 15. Personal Take

Andrej, 这篇 paper 的核心 contribution 在于它提供了一个 architecture-agnostic, task-agnostic 的 evaluation framework，并通过 action equivalents 的设计 cleverly mitigates shortcut learning。结果显示的 gap（best model 57% vs human 100%）表明 high-level world modeling 是一个 genuinely unsolved problem。

特别值得注意的是 video diffusion 的 poor performance——这直接挑战了当前 "generate videos = understand world" 的 narrative。Pixel-space generation 在 capturing high-level causal structure 上 demonstrably struggling。这呼应了你在 "Deep Dive into LLMs" 系列中强调的：understanding 和 generation 是不同的 capability。

Socratic LLMs 的 competitive performance 也值得深思——visual grounding 在当前 scale 下可能不是 bottleneck，reasoning capability 才是。但如果未来我们想要 truly embodied agents，必须跨越 visual grounding 和 causal reasoning 之间的 gap。

这个 benchmark 的 POSMDP grounding 提供了一个 clean theoretical framework，可以 guide 未来的 world model 设计。特别期待看到 V-JEPA2、DINO-WM 的 extensions 在这个 benchmark 上的表现。

Reference links for further exploration:
- Yann LeCun's original world model paper: https://arxiv.org/abs/1803.10122
- Sutton's options framework: https://arxiv.org/abs/1301.3754
- POMDP foundational paper: https://arxiv.org/abs/1301.6754
- JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2402.05792
- Cosmos: https://arxiv.org/abs/2501.03575

如果你对 specific baseline 的 implementation details 或 action equivalents 的具体 examples 感兴趣，可以继续 deep dive。
