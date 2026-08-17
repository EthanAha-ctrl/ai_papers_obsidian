---
source_pdf: Useful Memories Become Faulty When Continuously Updated by LLMs.pdf
paper_sha256: 16613d73b3dfe8de8dd73d42c4fb7b2e803b84a78d7ecf748c9e23a7e3b4aa92
processed_at: '2026-08-12T20:35:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇Paper

## 一句话总结

让LLM agent不断把过去经验"压缩"成memory bank这件事，越压缩越烂，烂到比完全没memory还差。

## 直觉比喻：传话游戏

想象你在玩传话游戏。第一个人说"我用柠檬汁和橄榄油调了沙拉酱"，传到第五个人变成"酸的东西配油可以当酱"，传到第二十个人变成"把互补的食材混在一起就好"。

每传一次，具体信息都掉一点，最后剩下一句**正确的废话**——对任何具体做菜都没帮助。

LLM agent的memory consolidation就是这个游戏。每次update memory，LLM rewrite一遍，信息loss一点，错误introduce一点，几十轮之后memory变成一堆vacuous abstraction。

## Paper发现了什么

### 发现1：记忆越用越差

用ScienceWorld举例（Figure 1a）：

```
Step 0:   no memory, score = baseline
Step 20:  memory peak, score最高
Step 100: memory退化，score低于baseline
```

WebShop更夸张（Figure 1b）：AWM从8个examples的0.64分，涨到128个examples时只剩0.20分。而no-memory baseline正好也是0.20——**积累越多经验，memory越没用**。

### 发现2：最强model也会退化

ARC-AGI上最clean的实验（Figure 2）：

- GPT-5.4在19个problems上，no memory时**100% solve**
- 拿ground-truth solutions做stream consolidation
- 同样这19个problems，consolidation之后**只剩52.6%**

Ground-truth都喂给它了，trajectory完全有用，唯一变量就是consolidation step本身。结果consolidation把model变蠢了。

### 发现3：原始记录比抽象总结更好用

Table 2最反直觉。把raw trajectory直接塞进context（episodic memory），效果跟ACE、AWM、Dynamic Cheatsheet这些精心设计的consolidator**打平甚至更好**。

你花了大力气抽象出来的"lessons"，效果就跟直接把原始记录丢进去一样。cognitive science里这叫instance theory——schema-like behavior可以直接从preserved instances中emerge，不需要单独store abstraction。

## 为什么会这样

### 原因1：每次rewrite都是lossy compression

Consolidation的数学本质：

$$M_t = f(M_{t-1}, B_t) + \epsilon_t$$

- $M_t$: 第 $t$ 步的memory state
- $f$: LLM consolidator的rewrite function
- $B_t$: 第 $t$ 步的new batch of trajectories
- $\epsilon_t$: 这一步引入的abstraction error

Static-All只做一次，error是 $\epsilon_{single}$。Stream做 $T$ 次，error是：

$$\sum_{i=1}^{T} \epsilon_i \cdot \prod_{j=i+1}^{T} \nabla f_j$$

后面那个Jacobian连乘 $\prod \nabla f_j$ 是error propagation的放大器。早期small errors成为later rewrites的context，compound成progressively distorted memories。

Figure 3直接证明：Stream比Static-All少17-38分，全部来自这个compounding effect。

### 原因2：分错组（Misgrouping）

 abstraction前要先segment——判断哪些episodes属于同一类。ARC-AGI Stream实验（Figure 7a）显示：

- Auto mode：71步之后model才能正确把6个problem family分开
- Force mode（强制每步consolidate）：routinely merge across class boundaries

Model其实**有能力**正确分类，但forced consolidation override了这个能力。每8个fresh examples就逼它abstract一次，它来不及segment就硬压缩，结果把不同family的episodes混在一起。

### 原因3：过度泛化（Overgeneralization）

每个abstraction pass都smooth和abstract existing entries——类似人类反复retelling故事导致的semanticization（Bartlett, 1932）。

具体例子，ALFWorld的Pick&Place vs Pick-Clean-Place（Figure 8c）：abstraction strip掉了lesson applies的preconditions，留下"reads broadly relevant但其实not"的内容。Solver看了这些overgeneralized lessons反而被mislead。

ScienceWorld 15-task switch（Figure 10）：
- Fresh（每个task单独distill）vs Cumulative（所有task一起consolidate）
- Cumulative落后Fresh **+203分**
- LLM judge发现Cumulative的overgeneralized memories是Fresh的**5倍**，garbage memories是**20倍**

### 原因4：过拟合（Overfit）

当input distribution变narrow时相反的failure。ARC-AGI上repeatedly feed同一family的tasks（Figure 11）：

- Exact repetitions: performance stable
- Same family的simple variations: **collapse**

Memory overfit到seen instances的surface regularities，而非underlying strategy。给narrow stream，abstraction loop会往observed instances的表面规律靠，而不是真正的problem-solving logic。

## 三个失败模式的Concrete Examples

### Vacuous abstraction（Section D.1）

GPT-5-mini在200-task pool后的memory entry：
> "Make a working copy of the input grid before mutating, perform all modifications on the copy, and return the copy to avoid mutating the original input."

这是**正确的Python编程idiom**，但跟6个ARC family的任何区分property都无关。对solver决策毫无帮助。

### Concrete-vocabulary erosion（Section D.2）

GPT-5-mini看"recolor the largest object"50次rewrite：

Round 1：`compute a global metric (e.g., max size)`
Round 50：`Find the maximum value of a derived per-object numeric attribute`

Round 1的"max size"是solver能计算的具体property。Round 50的"derived per-object numeric attribute"——哪个attribute？没说。**abstraction把具体feature abstract没了**。

### Foreign-family injection（Section D.3）

Consolidator把multiple families内容混进一个entry：
- Trigger引入shape-relation step（inside-frame task本不需要）
- Steps混入shape-signature lookup（group-by-shape family）+ color write into center（key-marker family）
- No family in the pool prescribes这composite

### Phantom strategies from failed attempts（Section D.4）

expose failed past attempts给consolidator，它从**failed code**中distilled patterns：
- "glyph-like objects"和"normalize the glyph"——6个family里都不存在
- Lineage traces to failed code fragment：`out[r][min(cols)] = 0`
- Consolidator读这fragment as intentional row-level edit rule
- Font-rendering metaphor是它produced的abstraction——**从failure中虚构出strategies**

### Single-strategy collapse（Section D.5）

19个tasks（6个family）revisited 10次后，memory只剩**1个entry**：
> "Extract connected objects, choose the largest as a frame, classify other objects by whether their bounding boxes lie strictly inside that frame, erase the frame and all outside objects, then hollow out each inside object..."

On held-out task whose true rule是"erase every object of one specific color"：
- With memory: empty grid, **0/10**
- Without memory: 8 lines of code, **10/10**

Memory把model变蠢了。

## ALFWorld的Erosion细节

### Stage 20（peak, 42 items）

Memory含diverse meta-strategies：
- Item 0: recognize task type early → prioritize fridge/microwave/sinkbasin/desklamp
- Item 3: **cooling vs heating asymmetry**（cooling/cleaning simpler than heating）
- Item 5: **shuttle default for two-object tasks**

### Stage 200（eroded, 38 items）

- 22/38 items mention desklamp（但look-at-in-light只占8/48 eval）
- 28/38 mention multi-object handling
- 33/38 mention systematic search
- **Items 3和5 byte-identical duplicates**
- Cooling-vs-heating asymmetry：**从所有item中消失**
- Shuttle default：concrete actionable form消失

### Collapse Event（Stage 168 → 169）

50 items（48,506 chars）→ 1 item（1,960 chars）

| Rollout | Stage 168 | Stage 169 | Δ |
|---------|-----------|-----------|---|
| Qwen3.5-4B | 35/48 | 29/48 | -6 |
| Qwen3.5-9B | 36/48 | 26/48 | -10 |
| Qwen3.5-27B | 37/48 | 24/48 | -13 |

**Stronger rollouts lose more**——因为它们能从structured memory中extract更多，merge成one item后损失更大。

## WebShop的Surgical Ablation

Section I.1的memory surgery特别直观。Consolidated memory有8个workflows，remove掉W8后：

| Rollout | Memory | wins/50 | mean reward |
|---------|--------|---------|-------------|
| GPT-5.4-mini | Full (8 workflows) | 7 | 0.23 |
| GPT-5.4-mini | **Minus W8** | **14** | **0.37** |
| GPT-5-mini | Full (8 workflows) | 18 | 0.49 |
| GPT-5-mini | **Minus W8** | **23** | **0.59** |

W8内容是"Search across pages when the first results do not match"，导致agent疯狂`click[Next >]`死循环。

行为数据（Table 13）：
- `click[Next >]` counts: 421 (full) vs 181 (minus W8), **2.3× ratio**
- Buy-Now episodes: 14/50 vs 21/50

**一个bad memory entry就显著hurt performance**，remove掉它就recover大部分loss。

## 解决方案：保留Episodes

### Section 5的核心实验

ARC-AGI Stream上三个observations：

**Observation 1**: Episodic store carries most of the gain（Figure 9a, 17）
- Abstract Only: ≈ no-memory baseline
- Episodic Only: ≈ Auto (full system)
- Auto vs Episodic Only: 只差几个points

**Observation 2**: Episodic Management Only matches or exceeds Auto
- 禁用abstraction，只允许retain/delete raw episodes
- 效果跟Auto打平甚至更好

**Observation 3**: Forced abstraction underperforms retained-episode policies（Figure 5）
- 400 training steps, Auto在两个backbone上都outperform Force

### Agent自己的选择

Figure 6, 7b：given option，agent快速saturate episodic buffer，keep abstract store sparse。Agent自己选了**episodic-first policy**。

### 核心Principle

Complementary Learning Systems (McClelland 1995, Sun 2023)：
- Fast episodic store和slow schema-forming store要**architecturally distinct**
- Fast learning不能overwrite slow one
- Consolidation要**gated by schema fit**，而非triggered on every event

Force regime collapse了这separation，recreate了dual-system design想prevent的interference conditions (McCloskey & Cohen, 1989; French, 1999)。

## 用一句话Build Intuition

**Current LLMs don't reliably decide when to abstract vs. preserve episodes.**

每次强制rewrite都是有损压缩，error compounding让memory drift away from underlying task structure。**Raw episodes contain signal solver可以直接exploit**，schema-like behavior可以从中emerge。Architecture应该keep episodic和abstraction roles distinct，gate consolidation explicitly，treat raw episodes as first-class evidence。

## 给Agent Memory设计者的建议

1. **不要update-after-every-interaction**：CLIN、AWM、ACE、Dynamic Cheatsheet都这么设计，treat each consolidation step as at worst neutral。Paper证明这assumption错了。

2. **Episodic baseline是diagnostic**：test你的consolidator之前，先test raw trajectory logs in context。如果你的consolidator打不过这baseline，它的"abstraction"没价值。

3. **Gate abstraction explicitly**：让agent choose when to consolidate，而非force every step。Auto mode和Episodic Management Only都match或exceed forced abstraction。

4. **Keep two stores distinct**：episodic buffer保留raw evidence，abstract store sparse。Don't collapse both into single rewrite loop。

5. **Long-term还是需要abstraction**：raw histories grow unboundedly，resist compositional reuse。但abstraction要selective, delayed, grounded in recoverable trajectories。

## Reference Links

- Complementary Learning Systems: https://doi.org/10.1037/0033-295X.102.3.419
- CLIN: https://arxiv.org/abs/2310.00575
- AWM: https://arxiv.org/abs/2409.07429
- Dynamic Cheatsheet: https://aclanthology.org/2026.eacl-long.333/
- ACE: https://arxiv.org/abs/2507.12643
- ConceptARC: https://arxiv.org/abs/2305.07141
- AppWorld: https://arxiv.org/abs/2407.18901
- WebShop: https://arxiv.org/abs/2207.01206
- Sun et al. 2023 (Nature Neuroscience): https://doi.org/10.1038/s41593-023-01382-9
- Bartlett 1932 (Remembering): 经典memory consolidation理论
- McCloskey & Cohen 1989: Catastrophic interference in connectionist networks
- Turpin et al. 2023: Unfaithful CoT explanations: https://arxiv.org/abs/2305.04388

---

# Paper深度解析: Useful Memories Become Faulty When Continuously Updated by LLMs

这篇paper来自UIUC和Tsinghua的合作工作，核心问题非常尖锐：**当LLM agents持续更新consolidated memory时，原本useful的experience反而会degrade成faulty memory**。让我系统讲解。

## 1. 核心问题与Motivation

### Background: 两种memory form

借鉴cognitive science的Complementary Learning Systems理论 (McClelland et al., 1995)：

- **Episodic traces**: 原始trajectory记录（what happened的raw log）
- **Consolidated abstractions**: 跨多个episode提炼出的reusable schema-like lessons

当前agentic memory systems（CLIN, AWM, ACE, Dynamic Cheatsheet等）都pursue consolidated form——LLM把过去trajectories重写成textual memory bank，并随new interactions持续更新。这承诺了**无需parameter update的self-improving agent**。

### 核心发现：Memory utility的非单调性

Paper最striking的发现用Figure 1展示：

```
Memory Utility(t) = f(consolidation_steps)
                    ↑ 先上升（early consolidation有用）
                    → plateau
                    ↓ 下降（可低于no-memory baseline）
```

具体地，在ScienceWorld上（Figure 1a）：score在step 20附近peak，到step 100时下降，某些memory size下低于no-memory baseline。WebShop上AWM从8 examples的0.64降到128 examples的0.20，而no-memory baseline就是0.20——**scaling memory最终erase了它自己的benefit**。

最clean的case在ARC-AGI（Figure 2）：GPT-5.4在19个problems上no-memory时100% solve；用ground-truth solutions做stream consolidation后，在**同样这些已解决的问题上54%失败**。这直接证明failure来自consolidation step，而non-noisy trajectories。

## 2. 实验设置详解

### 三个memory construction regimes

设trajectory pool为 $T = \{(\tau_1, y_1), \ldots, (\tau_N, y_N)\}$，其中 $\tau_i$ 是trajectory，$y_i$ 是label/solution。

**Static-All**:
$$M_0 = \text{Consolidate}(T)$$
一次性从整个pool构建memory $M_0$。

**Static-Group**（用于ALFWorld/ScienceWorld）:
$$M_0 = \bigcup_{k=1}^{K} \text{Consolidate}(T_k)$$
其中 $T_k$ 是task type $k$ 的subset，最后concatenate。这是offline construction的control。

**Stream**（模拟continual learning）:
$$M_t = \text{Update}(M_{t-1}, B_t)$$
其中 $B_t \subset T$ 是第 $t$ 个batch。这里 $M_t$ 是随时间累积更新的memory store。

### ARC-AGI Stream testbed

Paper自建的controlled testbed，三个关键属性：

**(i) Fully specified latent task taxonomy**：基于ConceptARC (Moskvichev et al., 2023)，定义6个family × 7个skill的正交axes：

- **Family**（selects which objects participate）：
  - `detect_color_property`（by color）
  - `detect_largest_objects_select`（by size extremum）
  - `detect_key_marker_rule`（global conditional cue，upper-left cell触发branch）
  - `group_by_shape_then_select`（cross-object relation）
  - `detect_inside_frame_relation`（spatial containment）
  - `compose_horizontal`（cross-grid structure）

- **Skill**（transformation applied to selected objects）：
  - `keep`, `border`, `recolor`, `translate`, `flip_horizontal`, `mark_center`, `hollow`

**(ii) Programmatic ground-truth**

**(iii) Exposed structured vocabulary of memory operations**：三个actions
- **Retain**: 保留raw episode到Episodic buffer
- **Delete**: 删除entry
- **Consolidate**: 将buffered episodes压缩到Abstract store

### 三种control loops

$$\text{Force}: \forall t, \text{action}_t = \text{Consolidate}; \text{Episodic buffer cleared each round}$$

$$\text{Auto}: \text{action}_t \in \{\text{Retain, Delete, Consolidate}\} \text{ (model chooses)}$$

$$\text{Episodic Management Only}: \text{action}_t \in \{\text{Retain, Delete}\}, \text{abstraction disabled}$$

两种trajectory regime：**GT**（stream ground-truth solutions）和**Running**（agent从自己attempts学习）。

## 3. 核心实验结果

### 3.1 Stream updates reveal cumulative effect of imperfect abstraction

**Finding 1: Abstracted memory utility是non-monotonic in updates**

形式化：定义 $U(M_t)$ 为memory $M_t$ 在held-out tasks上的utility。实验观察到：
$$U(M_0) < U(M_{t^*}) > U(M_T), \quad 0 < t^* < T$$
且有时 $U(M_T) < U(M_0) = \text{no-memory baseline}$。

**Finding 2: Strong memory不是fixed point**

用GPT-5.4做Static-Group consolidation得到strong starting memory $M_0^{strong}$，然后用GPT-5-nano继续batch updates：
$$M_0^{strong} \to M_1 \to \ldots \to M_T$$
在Qwen3.5-{27B, 9B, 4B}三个solver上evaluate，utility在所有三个上都decline，且可在consecutive evaluation steps之间drastically drop。

**Finding 3: Agent regress on tasks it has already solved**

ARC-AGI Stream最clean的实验：
- $\text{Acc}(\text{GPT-5.4}, \text{no memory}) = 100\%$ on 19-problem slice
- Streaming those problems through consolidation loop（ground-truth available each step）
- $\text{Acc}(\text{GPT-5.4}, \text{Stream } M_{10}) = 52.6\%$

**Finding 4: Consolidation procedure本身shapes decline rate**

对比Static vs Stream（Figure 3）：
- Static-Group > Static-All > Stream
- Stream loses 17–38 points vs whole-batch Pool

Heterogeneous batches加速decline（Figure 4）：mixed-task batches比single-task batches产生worse memory。

### 3.2 Episodic-only store应该是diagnostic baseline

Table 2的关键观察：在AppWorld、ALFWorld、WebShop上，episodic-only memory（append-only trajectory rollouts in context，无cross-trajectory abstraction）与lesson-style consolidators竞争力相当。

定义Trajectory Logs baseline：
$$M_{\text{episodic}} = \{\tau_1, \tau_2, \ldots, \tau_k\} \text{ (summarized rollouts as in-context exemplars)}$$

variants：
- `All`: 全pool
- `FS=64/32/16`: downsample到固定shots
- `Incr. Keep`: LLM-curated retain/delete (streamed)
- `One-Pass Keep`: LLM-curated retain/delete (one-pass)

Table 2中，红色(-)标记的cell表示：某个non-trajectory-log方法被同一row中至少一个trajectory-log baseline击败。大量consolidator cells被标记——**在多数情况下，abstracted memory无法significantly outperform direct in-context learning from trajectory logs**。

这呼应了cognitive science的instance theory (Hintzman, 1986; Brooks, 1978; Medin & Schafer, 1978; Logan, 1988)：**schema-like behavior可以从preserved instances的in-context learning中emerge**，而不需要separately stored abstraction。

## 4. Mitigation: 保留Episodes

### Section 5的核心实验

ARC-AGI Stream上三个observation：

**Observation 1: Episodic store carries most of the gain**

对比memory source ablations（Figure 9a, 17）：
- Abstract Only: 接近zero-shot baseline（不改善）
- Episodic Only: 几乎recover全部Auto gain
- Auto (both): Episodic Only + 几个extra points

形式化：
$$U(\text{Abstract Only}) \approx U(\text{NoMem})$$
$$U(\text{Episodic Only}) \approx U(\text{Auto})$$
$$U(\text{Auto}) - U(\text{Episodic Only}) \approx \text{small synergy}$$

**Observation 2: Episodic Management Only matches or exceeds Auto**

禁用abstraction，只允许retain/delete：
$$U(\text{Episodic Mgmt Only}) \geq U(\text{Auto})$$

**Observation 3: Forced abstraction underperforms retained-episode policies**

Figure 5：400 training steps的cumulative success，Auto在两个backbones上一般outperform Force。

### Agent自己的management decisions

Figure 6, 7b显示：given option，agent快速saturate episodic buffer（两种budget level都如此），keep abstract store sparse。这反映agent选择了**episodic-first policy**。

### Principle: Complementary Learning Systems

引用McClelland 1995, Kumaran 2016, Tse 2007, Sun 2023：
- Fast episodic store和slow schema-forming store应architecturally distinct
- Fast learning不应overwrite slow one
- Consolidation应gated by schema fit，而非triggered on every event

**Force regime** collapse了这个separation，recreate了dual-system design想prevent的interference conditions (McCloskey & Cohen, 1989; French, 1999)。

## 5. Failure Modes深度解析

### 5.1 Misgrouping before abstraction

**Correct segmentation是precondition** (Zacks et al., 2007)：agent不能从一组它未正确group的episodes中abstract出lesson。

**Capacity存在**：在Auto mode下，model eventually converges to clean episodic store covering all 6 problem types（Figure 6）。但takes time——只有after 71 steps (568 examples) does model admit all six problem types into buffer。

**Force regime破坏segmentation**：Figure 7a显示，在forced consolidation下，model routinely merges across class boundaries。Misclassification count（combine episodes from 2+ distinct problem classes的memory-management actions数量）在Force下显著高。

**Bottleneck**：non-recognizing class structure，而是preserving it through cross-episode abstraction——**forced act of abstraction overrides the segmentation the model is otherwise capable of**。

### 5.2 Interference from overgeneralized lessons

每个abstraction pass smooths和abstracts existing entries——**analogous to semanticization of episodic memories under repeated retelling** (Bartlett, 1932)。

当underlying chunks imprecisely bounded时，abstraction strips掉lesson applies的preconditions，留下reads broadly relevant但其实not的内容。

**Concrete example**: ALFWorld的Pick&Place vs Pick-Clean-Place（Figure 8c）：agent sees overgeneralized lessons并被misled而非aided。

**Task-level interference**: Figure 10的ScienceWorld 15-task switch sequence：
- **Fresh**: at each task $t$, 只在task $t$上distill memories
- **Cumulative**: 在tasks $1..t$上jointly consolidate

Cumulative ends **+203 points behind** Fresh。

LLM judge labeling发现：Cumulative累积overgeneralized memories at **~5× Fresh's rate**，garbage memories at **~20×**，两个gaps都monotonically widening across 15 tasks（Figure 18）。

### 5.3 Overfit to narrow streams

当input distribution narrows时，complementary failure mode emerges。

ARC-AGI实验：repeatedly feed model tasks drawn from single strategy family across consolidation cycles。形式化：设iteration count $r$，观察：
$$\text{Acc}(\text{exact repetitions}) \approx \text{stable}$$
$$\text{Acc}(\text{variations within same family}) \to \text{collapse as } r \to \infty$$

Figure 11展示这dynamic：**memory overfits to specific instances it has seen**，performance on exact repetitions稳定但on simple variations within same strategy family collapses。

**Mechanism**：abstraction loop，given narrow stream，abstracts toward **surface regularities of observed instances**而非underlying problem-solving strategy。

### 5.4 Case Study具体例证

#### Vacuous abstraction（Section D.1）

GPT-5-mini在200-task pool后，memory entry 0：
> "Make a working copy of the input grid (list of row lists) before mutating, perform all modifications on the copy, and return the copy to avoid mutating the original input."

**Why harmful**: defensive Python idiom shared by every nested-list program。Mentions no color, shape, or rule that 6 families distinguish themselves by。

#### Concrete-vocabulary erosion（Section D.2）

GPT-5-mini看一个training task("recolor the largest object") 50次rewrite：

**Round 1**: "compute a global metric (e.g., max size) → iterate elements and selectively apply targeted edits"

**Round 50**: "Find the maximum value of a derived per-object numeric attribute and apply a uniform transformation to every object whose attribute equals that maximum."

**Why harmful**: Round 1的"max size"names a property solver can compute in cells。Round 50 phrasing names no specific feature；entry不再records which attribute to maximize。

#### Foreign-family injection（Section D.3）

Strategy entry的trigger和steps混入multiple families内容：
- (1) Foreign-family extension to trigger：第二句introduces shape-relation step，inside-frame source task并不perform
- (2a-2b)Composite absent from every family：shape-signature lookup（group-by-shape family）+ color write into matched interior object's center（key-marker family），no family prescribes这composite

#### Phantom strategies from failed attempts（Section D.4）

expose strategy-update prompt to both successful和failed past attempts后，consolidator distilled recurring patterns from **failed code** into new entries。

具体："glyph-like" objects和"normalize the glyph"/"stroke-shift rule"匹配nothing in 6 families——**consolidator produced both**。Lineage traces to recurring fragment in failed solver code：`out[r][min(cols)] = 0`（"erase leftmost cell of this row"）。Strategy-update step读这fragment as evidence of intentional row-level edit rule；font-rendering metaphor是它produced的abstraction。

#### Single-strategy collapse on multi-family compression（Section D.5）

19 tasks revisited 10 times → memory contains **single entry**：
> "Extract connected objects, choose the largest as a frame, classify other objects by whether their bounding boxes lie strictly inside that frame, erase the frame and all outside objects, then hollow out each inside object in place by turning its interior to color 0 while preserving its boundary."

On held-out task whose true rule是"erase every object of one specific color"——unrelated to inside-frame pattern——model with memory producesempty grid（0/10），而same model without memory solves in 8 lines（10/10）。

## 6. Memory Surgery: WebShop W8 Ablation

Section I.1的surgical ablation很有意思。Consolidated memory含8个workflows W1-W8，**remove W8后performance提升**：

| Rollout | Memory | wins/50 | mean reward | mean steps |
|---------|--------|---------|-------------|------------|
| GPT-5.4-mini | Full (8 workflows) | 7/50 | 0.23 | 12.4 |
| GPT-5.4-mini | Minus W8 (7 workflows) | **14/50** | **0.37** | 11.4 |
| GPT-5-mini | Full (8 workflows) | 18/50 | 0.49 | 18.2 |
| GPT-5-mini | Minus W8 (7 workflows) | **23/50** | **0.59** | 15.8 |

**W8内容**: "Search across pages when the first results do not match"——倾向于`click[Next >]` dead loops。

**Behavioral evidence**（Table 13）：
- `click[Next >]` counts: 421 (full) vs 181 (minus W8)，**2.3× ratio**
- Buy-Now episodes: 14/50 (full) vs 21/50 (minus W8)
- 同样2.3× ratio in GPT-5-mini rollout

**这证明**：单个bad memory entry可以significantly hurt performance；memory surgery（remove W8）即可recover大部分loss。

## 7. ALFWorld Erosion Case Study

### Stage 20（peak, 42 items）

Memory含diverse meta-strategies：
- Item 0: recognize task type early（prioritize fridge for cooling, microwave for heating, sinkbasin for cleaning, desklamp for light）
- Item 1: systematic search by room type
- Item 3: **cooling vs heating asymmetry**（cooling/cleaning simpler than heating）
- Item 5: **shuttle default for two-object tasks**

### Stage 200（eroded, 38 items）

Collapse to small number of overlapping templates：
- 22/38 items mention desklamp
- 28/38 mention multi-object handling
- 33/38 mention systematic room-by-room search
- **Items 3 and 5 byte-identical duplicates**

**Lost meta-strategies**:
- Cooling-vs-heating asymmetry absent from every item at stage 200
- Shuttle default for two-object tasks absent in concrete actionable form

### Collapse Event（Section G.2）

Stage 168 → Stage 169：memory从50 items（48,506 chars）collapse到1 item（1,960 chars）。Manager merged 50 structured items into one numbered "unified loop"。

| Rollout policy | no memory | stage 168 (50 items) | stage 169 (1 item) | Δ |
|---------------|-----------|---------------------|--------------------|----|
| Qwen3.5-4B | 15/48 | 35/48 | 29/48 | -6 |
| Qwen3.5-9B | 15/48 | 36/48 | 26/48 | -10 |
| Qwen3.5-27B | 19/48 | 37/48 | 24/48 | -13 |

**Magnitude scales with rollout strength**：stronger rollouts extract more from structured 50-item memory，因此lose更多when merged into one。

## 8. Theoretical Framework: Metacognitive Control

### Schema formation的dual nature

Schema formation is **beneficial when new evidence fits existing structure** (Tse et al., 2007)，但**harmful when non-selective or poorly controlled**：
- Non-selective consolidation causes interference和loss of specificity (McClelland et al., 1995; Sun et al., 2023)

### Metacognitive control的role

**Whether consolidation helps depends on metacognitive control**——deciding:
1. Which experiences belong together（segmentation）
2. How abstractly to rewrite them（abstraction level）
3. When to preserve distinctions rather than collapse them（gating）

Human metacognition is itself imperfect (Flavell, 1979; Nelson & Narens, 1990; Koriat, 1997)。Whether LLMs reliably monitor self-generated abstractions是open question——chain-of-thought explanations已被shown to diverge from underlying computation (Turpin et al., 2023)。

**Agent-memory systems implement explicit analogue**：abstracting "lessons" reused on later tasks，同时putting same model in charge of both generating memory和monitoring its own abstraction。**Failure modes we observe are consistent with this control loop being unreliable in practice**。

## 9. 数学形式化：为什么Stream比Static差？

设单个consolidation step引入error $\epsilon_t$。Memory state：
$$M_t = f(M_{t-1}, B_t) + \epsilon_t$$

其中 $f$ 是consolidator的rewrite function，$B_t$ 是batch，$\epsilon_t$ 是step $t$ 的abstraction error。

**Static-All**: 
$$M^{\text{static}} = f(\emptyset, T) + \epsilon_{\text{single}}$$
单次error，无compounding。

**Stream**:
$$M_t = f(f(f(\ldots f(\emptyset, B_1) \ldots, B_{t-2}), B_{t-1}), B_t) + \sum_{i=1}^{t} \epsilon_i \cdot \prod_{j=i+1}^{t} \nabla f_j$$

**Key insight**: early abstractions anchor later rewrites，small errors in segmentation或abstraction compound into progressively distorted memories。Error propagation的Jacobian $\prod \nabla f_j$ 在长horizon下放大初始errors。

这解释了Figure 3的17-38 point loss：Stream的cumulative error远大于Static的single error。

## 10. 对当前Agentic Memory Systems的Implications

### Update-after-every-interaction designs的问题

Many recent systems—CLIN (Majumder et al., 2023), Agent Workflow Memory (Wang et al., 2024), Dynamic Cheatsheet (Suzgun et al., 2026), ACE (Zhang et al., 2025)—adopt update-after-every-interaction designs，treat each consolidation step as **at worst neutral**。

**Paper直接contradict这assumption**：每个consolidation step是有损rewrite，useful details被dropped，spurious rules被introduced，once-helpful abstractions drift away from underlying task structure。

### Episodic accumulation的limitation

Yet episodic accumulation cannot be long-term answer：
- As deployment continues，raw histories grow unboundedly
- Resist compositional reuse
- Compression和transfer ultimately require abstraction

**Long-term horizon看**：需要architectures that keep episodic和abstraction-forming roles distinct，rather than collapsing both into single rewrite loop。

## 11. Limitations

1. **Scope**: text-based agentic benchmarks（ALFWorld, ScienceWorld, WebShop, AppWorld, Mind2Web）+ controlled ARC-AGI Stream；embodied/multi-modal/tool-rich production settings未cover
2. **Memory type**: natural-language abstraction via contemporary LLMs（GPT-5.4 family, Qwen3.5 family）；parametric memory（weight updates）和structured non-textual representations out of scope
3. **Consolidator = solver = LLM**: faulty abstraction reflects current model capability，可能shift with stronger consolidators或consolidator-specific fine-tuning
4. **Statistical**: API-cost constraints下point estimates from small number of repeats per question，无formal error bars；通过multiple models, benchmarks, memory frameworks cross-check mitigation

## 12. Core Takeaways for Building Intuition

### Intuition 1: Abstraction is lossy compression，不是lossless distillation

每个consolidation step是rewrite operation，必然drops information。当useful details被dropped，spurious rules被introduced——memory drifts away from underlying task structure。

### Intuition 2: Errors compound in iterative rewriting

不像database的append-only log，consolidated memory的每一步都rewrites previous products。Small abstraction errors in early steps成为later rewrites的context，compound into progressively distorted memories。

### Intuition 3: Episodic evidence是first-class evidence

Raw trajectories含task-relevant signal solver可以直接exploit：observations, actions, intermediate failures, environmental feedback——都tied to concrete situation in which they occurred。Schema-like behavior可以从preserved instances的in-context learning中emerge。

### Intuition 4: Metacognitive control是bottleneck

问题non-recognizing class structure，而是preserving it through cross-episode abstraction。Current LLMs不reliably decide:
- Which episodes belong together
- Which distinctions should survive compression  
- When experience should remain episodic

### Intuition 5: Gate abstraction explicitly

Robust agent memory应该**treat raw episodes as first-class evidence**和**gate consolidation explicitly**，rather than firing it after every interaction。Auto mode和Episodic Management Only都match或exceed forced abstraction——pointing to **mandatory rewriting at every step，rather than abstraction itself，as the decisive failure mode**。

## 13. 未来方向

Paper结尾的vision：

> Reliable agentic memory will require LLMs that can consolidate **without overwriting the evidence they depend on**.

具体open problems：
1. **Selective abstraction**: 何时abstract，何时preserve episodes
2. **Gated consolidation**: schema fit-triggered rather than event-triggered
3. **Architectural separation**: episodic和schema-forming roles保持distinct
4. **Recoverable trajectories**: abstraction应grounded in recoverable raw episodes
5. **Better metacognition**: LLMs需要reliably monitor self-generated abstractions

## Reference Links

- Paper arXiv链接（推测）: https://arxiv.org/abs/2025.xxxxx（paper本身在UIUC完成，作者Dylan Zhang等）
- Complementary Learning Systems: https://doi.org/10.1037/0033-295X.102.3.419 (McClelland et al., 1995)
- CLIN: https://arxiv.org/abs/2310.00575 (Majumder et al., 2023)
- Agent Workflow Memory: https://arxiv.org/abs/2409.07429 (Wang et al., 2024)
- Dynamic Cheatsheet (EACL 2026): https://aclanthology.org/2026.eacl-long.333/
- ACE: https://arxiv.org/abs/2507.12643 (Zhang et al., 2025)
- A-Mem (NeurIPS 2025): https://openreview.net/forum?id=FiM0M8gcct
- ReasoningBank (ICLR 2026): https://openreview.net/forum?id=jL7fwchScm
- ConceptARC: https://arxiv.org/abs/2305.07141 (Moskvichev et al., 2023)
- Reflexion: https://arxiv.org/abs/2303.11366 (Shinn et al., 2023)
- ExpeL: https://doi.org/10.1609/aaai.v38i17.29936 (Zhao et al., 2024)
- Voyager: https://arxiv.org/abs/2305.16291 (Wang et al., 2023)
- AppWorld: https://arxiv.org/abs/2407.18901 (Trivedi et al., 2024)
- Mem0: https://arxiv.org/abs/2504.19413 (Chhikara et al., 2025)
- LangMem: https://langchain-ai.github.io/langmem/
- Letta (MemGPT): https://www.letta.com
- Generative Agents: https://doi.org/10.1145/3586183.3606763 (Park et al., 2023)
- WebShop: https://arxiv.org/abs/2207.01206 (Yao et al., 2022)
- Mind2Web: https://arxiv.org/abs/2306.06070 (Deng et al., 2023)
- Sun et al. 2023 (Nature Neuroscience): https://doi.org/10.1038/s41593-023-01382-9
- Tse et al. 2007 (Science): https://doi.org/10.1126/science.1135935

## Final Intuition Building

想象你是一个chef学习cooking。Episodic memory是每个recipe的完整step-by-step记录。Consolidated memory是你的"cooking principles"——比如"酸和油balance vinaigrette"。

**问题**：每次cook新dish，你rewrite你的"principles" book。某个recipe用lemon和olive oil，你note"acid + oil = dressing"。下个recipe是soy sauce和sesame oil（不是acid），你abstract成"liquid + oil = sauce"。再来个recipe是tomato和basil（neither liquid nor oil），你generalize成"complementary flavors"。

**20个recipes后**，你的"principles"book里只剩"combine complementary ingredients well"——vacuous abstraction，对任何具体cooking都unhelpful。

**而raw episodic memory**——每个recipe的完整记录——仍然useful：你可以直接reference specific recipes做similar dish。

这就是paper的core finding：**Current LLMs don't reliably decide when to abstract vs. preserve episodes**。Architecture应该keep both roles distinct，gate abstraction explicitly，treat raw episodes as first-class evidence。
