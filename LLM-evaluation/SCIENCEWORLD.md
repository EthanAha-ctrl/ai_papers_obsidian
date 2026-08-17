---
source_pdf: SCIENCEWORLD.pdf
paper_sha256: ab64687178194d265b665a7b020cefd4a7087319977ebd469d367e381588efbd
processed_at: '2026-08-12T04:12:11-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用大白话再讲一遍。

---

这篇 paper 就在问一件事：**现在的 AI 模型，到底是"真懂"科学知识，还是只是"背得多"？**

你看那些大模型做小学科学题，准确率都 90% 以上了，看着挺唬人。但作者怀疑——这可能就是见多了相似题目，模式匹配出来的，跟真正理解物理、化学、生物过程完全是两回事。

怎么验证？作者搭了个文字版的"虚拟实验室"叫 SCIENCEWORLD。里面有个虚拟的房子，有 kitchen、bathroom、greenhouse 这些房间，里面有 thermometer、stove、battery、light bulb、metal fork、seeds 这些东西。而且这些玩意儿是真的能动的——stove 能加热东西，battery 接上 wire 和 bulb 能让 bulb 亮，plant 浇水会长大，bee 会传粉让花变 fruit。

然后给 AI agent 一道题，比如"判断 metal fork 导不导电"。如果只是做选择题，模型秒答"导电"。但在这里，agent 得真的去把 fork 拿起来，找到 battery 和 wire，连成一个 circuit，接上 light bulb，看灯亮不亮，亮了就把 fork 放蓝 box，不亮放绿 box。全程 10 多步操作，一步错就全崩。

这就逼着 agent 把"知道 fork 是导体"这个 declarative knowledge 跟"怎么搭电路验证"这个 procedural knowledge 串起来用。光会背没用。

---

**30 个 task 涵盖小学 10 个 science topic**：

- 把 ice 放 stove 上融化 / 把 water 煮沸 / 把 water 放 freezer 结冰
- 用 thermometer 量温度、量 boiling point
- 搭 series circuit、判断 renewable vs non-renewable energy
- 测某个东西导不导电（known object vs unknown object 对比组）
- 找一个 living thing / non-living thing / plant / animal
- 种 seed 长出 plant，甚至要 release bee 传粉长出 fruit
- mix chemicals、mix paints 调出 secondary / tertiary 颜色
- 找寿命最长 / 最短的 animal
- 观察植物 / 动物的 life stages
- 在 inclined plane 上做 friction 实验
- 做 Mendelian genetics 实验，判断某个 trait 是 dominant 还是 recessive

每个 task 还有 10 到 1400 个 variation——换 substance、换起始位置、换房间里的物品摆放，总共 7200 个变体，防止 agent 死记硬背。

---

**测了 6 个 agent，结果很惨**：

最好的 DRRN 只有 1.5M 参数，平均得分 0.17（满分 1.0）。那些 11B 参数的大模型反而只有 0.08。

为什么小模型赢大模型？因为这里考的不是"谁见过更多科学文本"，而是"谁能在 environment 里一步步操作把任务做完"。DRRN 直接从 simulator 拿到当前所有 valid actions，学一个 ranker 挑最好的。它学的 representation 就是 SCIENCEWORLD 这个 environment 的状态空间，128 维够用了。

而 11B 的 T5 模型是 offline 学的——给它 20 多万条 oracle 示范轨迹，让它模仿。问题是 oracle 只给了 canonical solution（用 stove 煮水），但 test 时 stove 可能被 disable 了，要改用 campfire。BC 学没见过这个 fallback，直接懵。而且大模型 free-form 生成的 action 大量是 invalid 的，beam search 16 个候选里可能一个能用的都没有。

这就是 paper 的 headline finding：**1.5M 参数 online RL 100k 步 > 11B 参数 offline imitation 20 万条示范**。差了 7000 倍的参数量，小模型赢。

---

**几个特别扎心的观察**：

1. **连 navigation 都过不了**。Agent 经常在"去 kitchen 拿 fork"这步就卡住了，根本走不到需要 science knowledge 的步骤。所以 known vs unknown conductivity task 的分数几乎没区别——agent 还没 reach 到能用 prior knowledge 的地方。

2. **Inverse scaling**。770M 的 T5 得 0.15，11B 的 T5 得 0.08。参数大 14 倍，分数砍半。作者很谨慎地说"这可能只是 hyperparameter 没调好"，但这个趋势跟直觉相反——大模型在这类 task 上反而不行。

3. **最容易的 task 是"找个 non-living thing 放 box 里"**，random baseline 都能拿 0.63。最难的是"对某个 substance 做任意 state change"和"种 fruit"，几乎所有模型都是 0。区别在于前者 3 步搞定，后者要 50 步连贯操作。

---

**paper 真正想说的**：

科学 QA benchmark 上的 SOTA 分数，不代表模型懂科学。你要验证它懂不懂，把它扔进一个 grounded environment 里看它能不能把知识用起来。结果发现——当前模型远远做不到。

而且更反直觉的是，在这种需要 long-horizon 交互推理的任务上，小模型 + online RL 比 大模型 + offline imitation 更高效。这跟 QA 任务上"bigger is better"的规律完全相反。

---

**之后发生了什么**：

这篇是 2022 年的。后来 Reflexion、SwiftSage 这些工作用 GPT-4 当 planner + SCIENCEWORLD 当 executor，把分数推上去了。但本质上都是"LM 出主意 + symbolic environment 干活"的 hybrid 模式，纯 LM 自己还是不行。

这跟 Chollet 的 ARC Prize、你自己在 nanoGPT 系列里反复纠结的"LM 到底是 reasoning 还是 retrieval"是同一个大问题。SCIENCEWORLD 给了一个 quantifiable 的 testbed，让这个 debate 可以用数字说话。

---

一句话：**小学 5 年级学生能做的科学实验，2022 年最强的 AI agent 做不到，而且模型越大未必越好。这说明我们对"reasoning"的理解和评测方式都有根本性的 gap。**

---

# SCIENCEWORLD 深度解读

## 1. Core Thesis：Retrieval vs Reuse 的试金石

Karpathy 我猜你看到 paper 的第一直觉就是 "这是 Chollet ARC 的精神在 NLP 版本的延伸"。ARC 用 minimal symbolic priors 测试 abstract reasoning，SCIENCEWORLD 则用 grounded interactive text 测试 reusable science knowledge。两者都质疑同一个现象：**大模型表面 SOTA，但本质上在做 sophisticated pattern matching / retrieval**。

paper 的核心 hypothesis 是公式化的判断：
- 如果模型是 retrieving → 在 known object task (e.g., Task 3-3: 测 metal fork 是否导电) 上能利用 prior knowledge 跳过实验步骤
- 如果模型是 reasoning → 在 unknown object task (e.g., Task 3-4: 测 "unknown substance B" 是否导电) 上必须完成实验，应该有相似 performance
- 实验证据：agents 在 3-3 vs 3-4 上几乎没差异 (DRRN 0.07 vs 0.20，反而 unknown 更高)，说明 agents 还没达到能利用 retrieval 的程度，被 commonsense navigation 卡住，根本走不到能 retrieve 的步骤

这跟你在 "State of GPT" 演讲里说的 "LLM 是 lossy compression of internet" 异曲同工：QA 模型在 science 多选题上 90%+ 是 because 训练分布里见过太多相似 pattern，不是因为 model 学会了 reasoning。

**Project page**: https://scienceworld.github.io/
**Code**: https://github.com/allenai/ScienceWorld
**Paper**: https://arxiv.org/abs/2203.07540

---

## 2. Environment 设计：为什么 Text 而不是 3D

这里有个非常 deliberate 的 design choice。SCIENCEWORLD 不做 3D，因为：

1. **Abstraction 层级**: text environment 允许 high-level action (e.g., `connect battery anode to blue wire terminal 1`)，3D 则要 decompose 到 motor-control level (伸手、握住、对准、转动)。abstract action space 让 science reasoning 成为 bottleneck，而不是 motor skill。

2. **Long-horizon reasoning 突出**: 100 steps 上限的 episode 内，要让 agent 完成多步 plan (找 thermometer → 走到 kitchen → focus on object → use thermometer → 读 temperature → 放进 answer box)。Navigation 已经卡住大部分模型，连 reasoning 的边都摸不到。

3. **Interpretability**: text transcript 可以直接当 manner explanation。

### 2.1 Object Model (Appendix A.1)

Object 用 object tree 表示 (parent = container, children = contained)。每个 object 是一个 property set 的 collection：
- Material properties (thermal conductivity, melting/boiling point, combustion point)
- Life properties (life stage, needs, genotype/phenotype)
- Device properties (activatable, conditions)
- Electrical properties (terminals, polarized/unpolarized)

每个 object 能生成多个 referents (e.g., 固态 water 可以被叫作 `ice`, `solid water`, `substance`)。这是关键的 language grounding 问题——agent 不能假设唯一指称。

### 2.2 Action Space (Table 3)

25 个 actions，参数化后约 200k combinations/step。这是 sparse-reward RL 的 nightmare。对照：
- Atari: ~4-18 discrete actions
- Go: 361 legal moves
- StarCraft: ~10^26 possible actions per step
- SCIENCEWORLD: ~200k，但 meaningful subset 很小 (parser + valid action detection aid 是关键)

## 3. Simulation Engines 细节

这是 paper 最有意思的部分，决定了 benchmark 的 reasoning depth。

### 3.1 Thermodynamics

简化版 conductive heat model。每个 step：
$$\Delta Q_{i \to j} = k_{ij} \cdot (T_i - T_j)$$

其中 $k_{ij}$ 是 mediated by thermal conduction coefficient of materials。允许 metal pot (conductor) vs ceramic (insulator) 区分。

每个 material 有 phase transition points：melting point $T_m$、boiling point $T_b$、combustion point $T_c$。Object 跨越这些 threshold 触发 state change 或 combustion。Combustion 会让 object 最终变 ash，除非被 put out。

Convective heat：heat sources (stove, oven) 和 heat sinks (fridge, freezer) 提供 fixed heat flux。Room ambient 也会和 objects 交换。

**Karpathy 你应该会联想到 MuJoCo thermal dynamics 或 PhysBench 这类仿真器**——但 SCIENCEWORLD 是 symbolic discretized 版本，更适合 LM agent。

### 3.2 Electricity

Series circuit 严格建模：
- 每个 component 2 terminals (anode/cathode for polarized, terminal 1/2 for unpolarized)
- Connection 是显式 action：`connect battery anode to blue wire terminal 1`
- Non-electrical objects 也有 virtual unpolarized terminals → metal fork 可以替代 wire
- Light bulb 亮 → 完整 circuit + power source + conductor
- Solar panel 必须在 outside 才能产电

这就让 Task 3-3/3-4 (conductivity test) 可以通过 "build circuit with the unknown object + bulb + battery, 看灯亮不亮" 来 ground。

### 3.3 Reproduction and Genetics

Punnett square 决定 offspring genotype：
$$P(\text{genotype}) = \text{Punnett}(\text{parent}_1 \text{ alleles}, \text{parent}_2 \text{ alleles})$$

Phenotype 通过 dominant/recessive rules。这设计允许 Task 10-1 (known pea plant) vs 10-2 (unknown plant B with random dominant/recessive assignment) 的 controlled comparison——前者可 retrieve (white flower in pea 是 dominant)，后者必须 grow 两代 + count 才能判断。

### 3.4 Friction (Inclined Plane)

1D 仿真。位置更新：
$$v \propto \sin(\theta) - \mu \cos(\theta)$$
$$x_{t+1} = x_t + v \cdot \Delta t$$

其中 $\theta$ 是 angle，$\mu$ 是 friction coefficient of surface material。Agent 看不到 numeric value，只能观察到 "block 60% of the way down"。所以 agent 要做 model-based inference：从 block 速度反推 angle 或 $\mu$。这非常 Chollet-ARC-like：symbolic physics with hidden latent variables。

---

## 4. POMDP Formulation 细节

paper 给出标准的 POMDP tuple $\langle S, T, A, R, O, \Omega, \gamma \rangle$：

| 符号 | 含义 |
|---|---|
| $S$ | 全部 states（环境的完整内部 state，包括所有 object property） |
| $T$ | $T(s' \| s, a)$: 条件转移概率。SCIENCEWORLD 是 deterministic + auto-process (thermodynamics 自动 tick)，所以 $T$ 大部分是 0/1，但 stochasticity 来自 environment parametric variation |
| $A$ | text commands (discrete) |
| $R: S \times A \to \mathbb{R}$ | reward function，由 subgoals 决定 |
| $O$ | 所有 possible text observations |
| $\Omega: S \to O$ | observation 条件概率。环境给 agent 的 text 描述只 reveal visible objects + 当前 room + inventory，是 partial observation |
| $\gamma \in [0,1]$ | discount factor |

Agent 学 policy $\pi_\theta(o) \to a$，maximize：
$$\mathbb{E}\left[\sum_t \gamma^t R(s_t, a_t)\right]$$

注意这里 paper 用 $\pi_\theta(o)$，不写 $\pi_\theta(o, d)$，但 agent 实际拿到 task description $d$。这是 SCIENCEWORLD 的 multi-task 设定：一个 agent 要在 30 个 task 上 generalize。

### Reward Shaping (Appendix B.1)

每个 task 有：
- **Required goals**: method-agnostic (e.g., 物质 solid → liquid)
- **Optional subgoals**: 2-15 个，nudge 到 canonical solution (e.g., "stove turned on", "temperature increased by 10°C")
- Total score normalized to [0,1]

这个 reward shaping 设计很关键：sparse 0/1 reward 在 100-step horizon 上根本学不动，subgoal reward 给了 shape。但同时引入了 imitation 信号——agent 可能 overfit canonical solution，忘了 general solution (campfire 也能 boil water 但 subgoal 只 credit stove)。

---

## 5. Agents Architecture 深度对比

### 5.1 DRRN (He et al. 2016) — 1.5M params，冠军

Architecture:
```
encoder_obs(o_t, o_t^look, o_t^inv) → h_obs ∈ R^128
encoder_act(a) → h_act ∈ R^128
score = h_obs · h_act  (dot product)
Q(s, a) = score
```

- 用 unigram subword tokenizer (SentencePiece 风格, Kudo 2018)
- embedding=hidden=128
- lr=1e-4, memory=100k, priority fraction=0.5 (prioritized replay)
- 8 env threads × 100k steps = 800k environment interactions
- 依赖 **valid action detection aid**：simulator 提供 $A_t$ candidate list，DRRN 只需 re-rank

**为什么这么小能赢**：因为 SCIENCEWORLD 的 state 表示本质是 structured symbolic，128-dim 足以 capture "kitchen + fork + thermometer" 这类 local context。Pre-trained LM 11B 反而被 internet distribution 的 prior 拖累。

参考: https://github.com/microsoft/tdqn

### 5.2 KG-A2C (Ammanabrolu & Hausknecht 2020) — 5.5M params

Paper 改了两个东西让它 work：
1. 用 **heuristic regex extractor** 替代 OpenIE——parsing "look around" 文本得 (subject, relation, object) triples。这其实是 cheat：agent 直接拿到 ground truth KG。
2. 选 action **type** 而非 referent (e.g., "pick up apple" 而不是 "pick up apple #3")，再 random ground 到 visible referent。

Architecture:
```
Text → KG triples → GNN encoder → policy + value
```

lr=3e-3。Performance 0.11 < DRRN 的 0.17。**说明 KG prior 在这环境没帮上忙**——因为 SCIENCEWORLD 的状态变化太多由 simulation engine 决定，KG 抓不到 thermodynamics 这种 latent process。

### 5.3 CALM (Yao et al. 2020) — 131M (GPT-2 + 6.9M policy)

CALM = Contextual Action Language Model。Prompt：
```
[CLS] d [SEP] o_t [SEP] o_t^look [SEP] o_t^inv [SEP] o_{t-1} [SEP] a_{t-1} [SEP]
```

GPT-2 fine-tune on oracle trajectories，beam search 16 beams → 30 candidate actions → DRRN re-rank。

GPT-2 config: 12 layers, 768 hidden, 12 heads (small variant, 117M)。lr=2e-5, batch=12, 20 epochs。

Performance 0.05，最差。**诊断**：GPT-2 生成的 candidate actions 大部分 invalid 或 useless。Language modeling prior 在 interactive long-horizon 上失效。

### 5.4 Behavior Cloning (Torabi et al. 2018, via Ammanabrolu et al. 2021) — 11B (T5 + Macaw init)

T5 input: $(d, o_{t-1}, a_{t-1}, o_t)$, target: $a_t$。

训练数据：211,092 (s,a) pairs from oracle。
- Macaw-11B 初始化 (Tafjord & Clark 2021)
- v3-32 TPU pod, batch=16, 32-way model parallelism, 100k gradient steps

Test time: zero-shot inference online，beam=16, diversity penalty=50.0，取 beam 里第一个 valid action。

Performance 0.08。**关键失败模式**：generate invalid actions，导致 0 reward。这跟 Huang et al. 2022 (SayCan) 的 observation 一致——LM 作为 planner 需要一个 affordance filter。

### 5.5 Text Decision Transformer (Novel) — 11B (T5 + Macaw init)

paper 自创，inspired by Decision Transformer (Chen et al. 2021)。

Input: $(d, o_{t-1}, \hat{R}_{t-1}, a_{t-1}, o_t, \hat{R}_t)$，target: $a_t$。
$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$
Returns-to-go 给 sequence 一个 "target cumulative reward"，希望 model 学到 "given high return, generate action sequence"。

训练数据：224,902 examples。

Performance 0.08，跟 BC 一样差。**Decision Transformer 在 offline RL 上 work 的前提是 dataset 覆盖足够 state-action distribution**。SCIENCEWORLD oracle 是 hand-coded canonical solutions，覆盖太窄，TDT 学到的是 "模仿 oracle"，而不是 "high-return 路径"。

---

## 6. Table 2 仔细读

Average scores:
| Agent | Avg | Params |
|---|---|---|
| Random-Valid | 0.03 | - |
| DRRN | **0.17** | 1.5M |
| KG-A2C | 0.11 | 5.5M |
| CALM | 0.05 | 131M |
| BC (Macaw-11B) | 0.08 | 11B |
| TDT (Macaw-11B) | 0.08 | 11B |

### 几个关键 pattern

**Pattern 1: Easy commonsense 任务能学到**
- Task 4-2 "Find a non-living thing"：Random 0.63, DRRN 0.56, CALM 0.54。这个 task 是 navigation + pick-and-place，action 短，rewards dense。
- Task 7-1/7-2 "longest/shortest-lived animal"：DRRN 0.48/0.47。这种 task 是 retrieve knowledge → 放进 box，步骤短，能学。

**Pattern 2: 需要多步 reasoning + simulation engine 的 task 全崩**
- Task 1-1 to 1-4 "Changes of State": 几乎全 0。需要找 substance → 找 heat/cool source → activate → wait → observe state change。10+ steps。
- Task 5-1/5-2 "Grow plant/fruit": 几乎全 0。需要 seed → soil → pot → water → wait life stage → pollinator (for fruit)。50+ steps horizon。
- Task 8-1/8-2 "Life stages": 0.10 左右。需要 focus on object → 喂水 → 等待。
- Task 9-x "Forces": 全 0.13 DRRN。需要 inclined plane + stopwatch + 物理推理。

**Pattern 3: Retrieval vs Experiment 的对比失败**
- 3-3 (known conductivity) DRRN 0.07 vs 3-4 (unknown) DRRN 0.20
- 10-1 (known genetics) 0.19 vs 10-2 (unknown genetics) 0.17

paper 在 Section 5 说 "We do not yet observe this behavior"——agents 根本没 reach 到能利用 known 信息省步骤的程度。这是 negative result 但很有意义：agents 卡在 "找到 object + 建好 circuit" 这个 prerequisite，"metal fork 是 conductor" 这种 knowledge 都没用上。

---

## 7. Inverse Scaling 现象 (Table 5)

这个 sub-table 是隐藏 gem：

| Model | BC | TDT | Params |
|---|---|---|---|
| T5-Large | 0.15 | 0.13 | 770M |
| Macaw-Large | 0.17 | 0.15 | 770M |
| Macaw-11B | 0.08 | 0.08 | 11B |

14x 参数，performance 砍半。Paper 谨慎说 "suggestive of inverse scaling problem"，留 hyperparameter confound 作 future work。

**Karpathy 你应该会想到的 hypotheses**:

1. **Coverage hypothesis**: 大模型更容易 overfit narrow oracle distribution。11B 参数 + 211k examples 是 overparameterized regime，small model 反而能学到 "general action syntax + simple state mapping"。

2. **Prompt sensitivity hypothesis**: 大模型对 prompt format 极敏感。SCIENCEWORLD 的 prompt 是 (d, o, a) structured tuple，不是 LM 预训练自然分布。11B 模型的 prior 太强，反而跟 task format 冲突。

3. **Long-horizon credit assignment**: BC 的 loss 是 per-step cross-entropy，不区分 critical vs trivial action。11B 模型 capacity 大，会花 capacity 学 trivial step 的 surface form，但 critical step (e.g., connect bulb anode) 的 action distribution 是 long-tail，11B 反而 underfit 这个 tail。

4. **Macaw pretraining mismatch**: Macaw 是 multiple-choice QA 用的，输出是 "answer text"，不是 "action text"。770M 可能 capacity 不够 bias 太强；11B 的 Macaw prior 把 output 拉向 "答案"而非 "action"，导致 invalid action rate 高。

Inverse scaling 相关的 follow-ups：
- "Inverse Scaling: When Bigger Isn't Better" (McKenzie et al. 2022, https://arxiv.org/abs/2211.02011)
- "What Causes Inverse Scaling?" (Yang et al. 2023)

---

## 8. Why Online RL > Offline LM — Intuition Building

paper 的 headline finding：
> 1.5M DRRN, 100k steps online > 11B T5, 211k demonstrations offline

让我提供几个 intuition 角度：

### Intuition 1: Exploration vs Imitation 的 Horizon 问题

Oracle trajectory 是 deterministic canonical solution，长度 L。BC 学到的是 "given (state, history), reproduce canonical action"。但 SCIENCEWORLD 的 environment 有 parametric variation——比如 stove 可能 disabled，agent 要 fallback 到 campfire。

Offline LM 见过 stove solution，没见过 campfire fallback。Online RL 通过 exploration 会发现 fallback path。所以 RL agent 在 dev/test 上的 unseen variation 有 generalize 能力，BC 在 train variation 上拟合好，test variation 上崩。

### Intuition 2: State Representation 学习

DRRN 的 encoder 是 from-scratch 学 128-dim representation，专门 fit SCIENCEWORLD 的 observation distribution。T5/Macaw 是 11B internet distribution prior，在 fine-tune 时 encoder 大部分 capacity 在抵抗 internet prior，不是 adapt 到 SCIENCEWORLD observation 的 structured symbolic form。

这跟 you 在 "Micrograd" / "Building GPT" series 里讲的 "narrow model 学 narrow task 比 big model 学 narrow task 更 sample efficient" 一致。

### Intuition 3: Action Space Sparsity

SCIENCEWORLD 每步 200k possible action combos，valid subset ~10-100。BC 的 T5 是 free-form generation，90%+ 输出 invalid。即使有 "valid action aligner" filter，剩下的 valid actions 也大多是 useless (e.g., `look at painting` 在 boiling task 上)。

DRRN 直接对 valid action list 做 ranking，loss 完全用在 meaningful discrimination 上。

### Intuition 4: Reward Signal vs Cross-Entropy Signal

BC loss 是 per-token cross-entropy，对 action 文本 surface form 敏感。RL loss 是 return，对 action effect 敏感。SCIENCEWORLD 评估的是 effect (subgoal meet) 而非 surface form。所以 RL 的 objective 跟 eval metric aligned，BC 的 misaligned。

**Chollet 在 ARC Prize 论文里也提过类似观察**: symbolic reasoning task 上，纯 LM imitation 难以 generalize，需要 explicit program synthesis / RL-style search。

---

## 9. 跟同期/后续工作的关联

### 9.1 同期
- **ALFWorld** (Shridhar et al. 2020, https://arxiv.org/abs/2010.03768): Text+3D hybrid, 6 task types, 家务场景。SCIENCEWORLD 比 ALFWorld 更 wide (10 topics) 且 simulation engine 更深。
- **TextWorld** (Côté et al. 2018, https://arxiv.org/abs/1806.11532): SCIENCEWORLD 的前辈，linear-logic-based，不支持 autonomous thermodynamics。
- **Jericho** (Hausknecht et al. 2020): Z-machine games unified interface，被用作 SCIENCEWORLD 的 valid-action-detection 范式参考。

### 9.2 后续 LLM agent 工作
- **ReAct** (Yao et al. 2022, https://arxiv.org/abs/2210.03629): reasoning+acting 交叉，纯 LM agent。
- **Reflexion** (Shinn et al. 2023, https://arxiv.org/abs/2303.11366): self-reflection，让 LM agent 在 SCIENCEWORLD 这种 long-horizon 上 iterative 改进。Reflexion paper 实际上就在 SCIENCEWORLD 上做过实验，gpt-4 + Reflexion 把某些 task 推到 0.3+。
- **SwiftSage** (Lin et al. 2023, https://arxiv.org/abs/2305.17390): 又一个在 SCIENCEWORLD 上 push SOTA 的，用 LM + classic planner 混合。
- **SayCan** (Huang et al. 2022, https://arxiv.org/abs/2204.01691): affordance filter 概念，对应 paper 里 BC 的 valid action aligner。
- **Voyager** (Wang et al. 2023, https://arxiv.org/abs/2305.16291): Minecraft agent，跟 SCIENCEWORLD 同问题域但 open-world。

### 9.3 你大概感兴趣的
- **Chollet 的 ARC Prize v2** (https://arcprize.org/): 同样质疑 LLM 的 abstract reasoning。
- **Decision Transformer** (Chen et al. 2021, https://arxiv.org/abs/2106.01345): paper 的 TDT 灵感来源。DT 的核心 insight 是 RL = sequence modeling when you condition on return-to-go。
- **Trajectory Transformer** (Janner et al. 2021, https://arxiv.org/abs/2106.02038): DT 的姊妹工作，model (s,a,r) 整个 sequence。
- **Hoffmann et al. Chinchilla** (https://arxiv.org/abs/2203.15556): compute-optimal scaling，跟 inverse scaling 现象对读——大 model 需要 proportionally more data，否则 underfit。

---

## 10. Limitations & Threats to Validity

paper Section 5 自己列了：
1. **Valid action detection aid**：除了 KG-A2C 和 CALM，其他 agents 在 test time 都用了 simulator 提供的 valid action list。这等于作弊——真实 LM agent 必须 generate valid action。
2. **Description length 限制**：transformer sequence length 限制下，环境描述被迫压缩，丢失了 simulation depth 该有的 vividness。
3. **Canonical solution only oracle**：oracle 是 hand-coded canonical (stove 而非 campfire)，限制了 BC/TDT 学到的 solution diversity。
4. **Inverse scaling 没完全 rule out hyperparameter confound**：11B 用 32-way model parallelism，batch=16；770M 单卡 batch=16。Optimizer state、learning rate schedule 可能不一样。

我额外加几点你大概会想到的：
5. **No LLM prompting baseline**：2022 年 paper，没测 GPT-3.5 / GPT-4 prompting。后续 Reflexion/SwiftSage 补上了，但 paper 本身这个空缺很明显。
6. **No tool-use baseline**：paper 完全没考虑 "LM + valid-action-oracle + memory" 这种 tool-augmented 范式。后续 agent 框架 (ReAct, Toolformer) 都证明这是关键 missing piece。
7. **No curriculum baseline**：30 task 一起训，没研究 task 之间的 transfer。
8. **Reward shaping 可能 overfit canonical**：subgoal credit stove/campfire asymmetry 可能埋下了 inverse scaling 的种子——11B 模型 capacity 大，更愿意 fit subgoal pattern，反而失去 generality。
9. **Deterministic environment**: SCIENCEWORLD 大部分 transition 是 deterministic，只是初始化有 variation。这降低了 RL 的 exploration 复杂度，但也降低了跟真实 world model 的 gap。

---

## 11. 思想实验：如何让 11B T5 在 SCIENCEWORLD 上 work

Karpathy 你大概会想这种 intervention：

**方案 A: Hybrid DRRN+T5**
- T5 generate candidate action → valid filter → DRRN rank
- 实际上 CALM 就是这个，但 GPT-2 generator 太弱
- 换 11B Macaw → 应该好点？但 paper Table 5 显示 11B BC 0.08 vs 770M BC 0.17，说明 generator 大不一定好

**方案 B: Online RL on top of LM init**
- 用 T5 作 Q network，online RL fine-tune
- 但 11B model online RL compute 太贵，paper 直接回避了

**方案 C: Hierarchical LM**
- High-level: LM 输出 subgoal (e.g., "heat the water")
- Low-level: DRRN-style policy 执行
- SayCan/ReAct 框架的雏形

**方案 D: Program synthesis**
- LM 输出 executable plan (e.g., PDDL)，而非 step-by-step action
- 这跟 Chollet ARC 的 program induction 殊途同归
- 后续 Code-as-Policies (Liang et al. 2022, https://arxiv.org/abs/2209.07753) 走这路线

**方案 E: Reflexion**
- 每次失败后让 LM 写一段 verbal reflection，加进 context
- SwiftSage/Reflexion 都证明在 SCIENCEWORLD 上有效

---

## 12. 数据集统计 cheat sheet

- 40k 行 Scala codebase
- 10 locations
- 195 object types (23 animals, 11 plants, 25 substances, 13 electrical, 16 devices, 15 furniture, ...)
- 80 materials
- 25 high-level actions
- 200k action-object combinations per step
- 30 tasks across 10 topics
- 7200 total variations
- Split: 50% train / 25% dev / 25% test
- 30 hand-coded oracles
- 211,092 (s,a) pairs for BC; 224,902 for TDT

---

## 13. Compute 预算 (Table 4)

| Model | GPU/TPU | Runtime/run |
|---|---|---|
| DRRN | 4GB GPU | 12h |
| KG-A2C | 16GB GPU | 20h |
| CALM-GPT2 | 16GB GPU | 40h |
| T5 Pretrain | v3-32 TPU | 60h |
| BC-T5 | 3×48GB | 2h |
| TDT-T5 | 3×48GB | 2h |

Full benchmark ≈ runtime × 30 tasks × 5 seeds。粗算 DRRN 全跑 30×5×12 = 1800 GPU hours，11B 大概 30×5×60 = 9000 TPU hours。2022 年代算 reasonable。

---

## 14. 跟 "MuZero for Science" 的远期想象

最后一个 Karpathy 风格的 speculation：SCIENCEWORLD 是个 symbolic simulator。如果有个 agent 能在 SCIENCEWORLD 上做到 MuZero 的 self-play 水平——
- 学 world model $s_{t+1} = f(s_t, a_t)$，包括 thermodynamics、circuits、genetics
- 学 value function $V(s)$，能判断 "boil water 离我还几步"
- 学 policy $\pi(a|s)$，能 zero-shot generalization 到 unseen task variations
- MCTS-style planning at inference time

那这种 agent 真的具备 reusable scientific reasoning。后续 works 像 **DreamerV3** (Hafner et al. 2023, https://arxiv.org/abs/2306.09637) 在 Minecraft 上做到这种 level，但 Minecraft 是视觉的；SCIENCEWORLD 这种 text symbolic 上还没人做出来 MuZero-level agent。

这跟 **LeCun 的 JEPA** (https://openreview.net/forum?id=BZ5a1r-kVsf) 的 joint embedding predictive architecture 也对得上——SCIENCEWORLD 的 latent state 是离散 symbolic property set，JEPA 学这种 abstract representation 应该很自然。

---

## 15. 一句话总结

SCIENCEWORLD 提供了 quantitative evidence that **science QA SOTA 模型在做 retrieval，不在做 reusable reasoning**；并且 **online RL on symbolic interactive environments 比 offline LM imitation 更 sample/parameter efficient for long-horizon grounded reasoning**。这是 Bitter Lesson 在 science reasoning domain 的一个 microcosm——structured representation + search/RL > unstructured LM prior + imitation，至少在 2022 年的数据规模下成立。

后续 Reflexion / SwiftSage / GPT-4 series 把 SCIENCEWORLD performance 推上去了，但都是靠 LM 当 planner + symbolic env 当 executor 的 hybrid 模式，本质上依然没解决 "纯 LM 能不能 reusable reasoning" 这个 SCIENCEWORLD 提出的核心问题。这正是 Chollet ARC、Karpathy nanoGPT、LeCun JEPA 都在反复追问的同一个 meta-question。

---

## Web Reference 汇总

- Project: https://scienceworld.github.io/
- Code: https://github.com/allenai/ScienceWorld
- Paper: https://arxiv.org/abs/2203.07540
- DRRN repo: https://github.com/microsoft/tdqn
- Macaw: https://github.com/allenai/macaw
- Decision Transformer: https://arxiv.org/abs/2106.01345
- ALFWorld: https://arxiv.org/abs/2010.03768
- TextWorld: https://arxiv.org/abs/1806.11532
- Jericho: https://github.com/microsoft/jericho
- Reflexion (后续): https://arxiv.org/abs/2303.11366
- SwiftSage (后续): https://arxiv.org/abs/2305.17390
- SayCan: https://arxiv.org/abs/2204.01691
- ReAct: https://arxiv.org/abs/2210.03629
- Chinchilla scaling: https://arxiv.org/abs/2203.15556
- Inverse Scaling: https://arxiv.org/abs/2211.02011
- ARC Prize: https://arcprize.org/
- DreamerV3: https://arxiv.org/abs/2306.09637
- LeCun JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf
- Code as Policies: https://arxiv.org/abs/2209.07753
- Trajectory Transformer: https://arxiv.org/abs/2106.02038
- Voyager: https://arxiv.org/abs/2305.16291
- WorldTree (Jansen): https://aclanthology.org/LREC-2018.728/
- ARC dataset (Clark): https://arxiv.org/abs/1803.05457
