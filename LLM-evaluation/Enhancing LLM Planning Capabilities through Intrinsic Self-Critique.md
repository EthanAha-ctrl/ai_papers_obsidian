---
source_pdf: Enhancing LLM Planning Capabilities through Intrinsic Self-Critique.pdf
paper_sha256: afdb2ea6a75286fb5f5ef26f77298d647f77c7a4fa834016d1938db6fd24cadc
processed_at: '2026-08-04T04:35:34-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

---

## 一句话说清楚

这篇 paper 干的事就是: **让 LLM 自己当自己的 grader**。 它先写一个 plan, 然后自己检查这个 plan 对不对, 如果不对, 把错误找出来, 重写, 再检查, 循环往复, 直到自己满意为止。

就这么简单一件事。 但它跑出来的数字很漂亮 — Blocksworld 上从 50% 干到 89%, 几乎追上了用 ground-truth verifier 做反馈的 Oracle upper bound (91.5%)。

---

## 为什么这件事之前大家觉得做不成

两年前 Kambhampati 的 group (Valmeekam et al. 2023a) 发了一篇 paper, 标题基本就是 "LLMs cannot critique their own plans"。 他们发现: 你让 GPT-4 自己检查自己写的 plan, 它会把错的 plan 也说成对的 — false positive rate 高得离谱。 你让它 self-correct, 它反而越改越烂。

Huang et al. 2024 那篇 "Large Language Models Cannot Self-Correct Reasoning Yet" 也说了类似的话: intrinsic self-correction (不靠外部 oracle, 纯靠模型自己) 在 reasoning task 上基本不 work, 模型会越改越差, 或者把对的改错。

所以这个领域形成了一个 folklore: **self-critique 是个 illusion, 模型自己骂自己没用**。

Google DeepMind 这帮人想说的是: **你们 prompt 写得不对, 所以才不 work**。

---

## 他们到底改了什么

核心 insight 就一句话: **你让 LLM 做 critique, 不能给它一个模糊的指令 "please critique this plan", 你得给它一个明确的, structured procedure, 告诉它怎么一步一步 check**。

具体来说, self-critique prompt 长这样:

```
Given the domain definition: {完整的 PDDL domain}

For each action in the plan:
  1. Take the action and its preconditions from the domain definition
  2. Verify whether the preconditions are met
  3. Apply the action and provide the resulting state

The problem: {instance}
The proposed plan: {plan}

Verify each step. Do not skip steps. Conclude with either:
  "the plan is correct", "the plan is wrong", or "goal not reached"
```

就这个东西。 三条 instruction, 加上 domain definition。

ablation 的数字极其 informative:

| 你去掉什么 | Accuracy |
|---|---|
| 完整方法 | 84.6% |
| 去掉 self-consistency | 79.7% |
| 去掉 few-shot exemplar (0-shot critic) | 79.5% |
| 去掉 domain definition | 74.4% |
| **去掉三条 step instruction** | **64.0%** |
| **去掉 "verify each action" 这句话** | **57.5%** |

你看, 把 "verify each action" 这一句删掉, accuracy 直接掉回 baseline (57.1%)。 这说明什么? **LLM 不会自发地去做 step-by-step verification**, 你不明确告诉它 "for each action, check precondition, apply, output state", 它就给你一个 vibe-based 的判断 "looks reasonable to me", 然后 false positive 一堆。

这个 lesson 其实挺深: **LLM 有能力做 systematic 的 multi-step procedure, 但它不会自发启动这个 procedure, 你必须在 prompt 里显式 trigger**。

---

## 整个 loop 长什么样

```
Initialize: τ = empty context

for step in 0 to k=10:
    1. PlanGeneration: 
       prompt = domain + few-shot examples + τ (之前所有失败的 plan 和 critique)
       plan = LLM(prompt)
       
    2. SelfCritique:
       prompt = domain + problem + plan + "verify each step..."
       (optionally: sample N=5 次, majority vote)
       critique = LLM(prompt)
       
    3. If critique says "correct": break, return plan
       Else: τ = τ + (plan, critique)   # 把这次失败 append 到 context
             go to step 1
```

每轮循环包含两个 LLM call: 一个生成 plan, 一个 critique plan。 如果 critique 说 ok, 就 break。 如果说不对, 就把 (plan, critique) 这一对塞进 context, 下次生成 plan 的时候能看到 "上次我这么试, 错在哪了"。

---

## 为什么这件事能 work — 最深的 intuition

Planning 和 verification 在 computational complexity 上是两个完全不同量级的任务。

**Plan generation**: 从 initial state 找到 goal, 这是 search。 Blocksworld 是 PSPACE-complete (Bylander 1994)。 你要 forward search 一个 exponential 的 branching factor, 或者 backward search, 总之你要 explore。

**Plan verification**: 给你一个 plan, 让你 check 对不对。 你只需要从头到尾 apply 一遍 action, 每个 action 检查 precondition 满足不满足, 然后更新 state。 这是 polynomial 的, 线性于 plan length。

LLM 是一个 left-to-right token predictor, 它没有 backtracking, 没有真正的 search, 所以它在 generation 上弱 — 它只能一路走到底, 走错了就错了。

但 LLM 在 step-by-step 的 state simulation 上 OK, 因为这本质上是 chain-of-thought reasoning — 给定一个 state 和一个 action, 输出新 state, 这是 next-token prediction 能干的活, 只要你在 prompt 里把 action 的 precondition 和 effect 都写清楚。

所以这个 loop 利用的 asymmetry 就是: **让 LLM 做它擅长的事 (verification), 用 verification 的结果来引导它做不擅长的事 (search)**。

这和 alpha-beta pruning 的思想有点像: 你不直接 solve, 你 evaluate leaf, 然后 back up。 这里 LLM 不直接 search 出 correct plan, 它 generate 一个 candidate, evaluate, 如果错了就修正再 generate。

---

## 看一个实际的 trace 更直观

Paper 在 Appendix D 给了一个完整的 trace, 我把它简化讲:

**Problem**: 把几个 block 从一个 configuration 换到另一个。

**Iteration 1**: LLM 生成 plan:
```
(unstack b5 b2)
(put-down b5)
(unstack b2 b1)
(put-down b2)
(pick-up b3)
(stack b3 b2)
(pick-up b5)
(stack b5 b3)
(pick-up b2)        ← 这里 bug
(stack b2 b5)
...
```

**Self-critique**: LLM 一步一步 verify:
```
Step 9: (pick-up b2)
  Preconditions: (clear b2), (ontable b2), (handempty)
  - (clear b2): FALSE  ← 因为 step 6 把 b3 stack 到 b2 上了, b2 不 clear
  - (ontable b2): FALSE  ← 同理
  - (handempty): TRUE
  Precondition not met!
  
the plan is wrong because b2 is not clear in step 9.
```

LLM 自己发现自己的 bug 了。 然后这个 critique 被塞回 context。

**Iteration 2**: LLM 看到 "上次的 plan 在 step 9 错了, b2 不 clear", 重新生成:
```
(unstack b5 b2)
(put-down b5)
(unstack b2 b1)
(stack b2 b5)        ← 改了: 不 put-down, 直接 stack 到 b5
(pick-up b3)
(stack b3 b2)
(pick-up b1)        ← 又有新 bug: b1 在 b4 上, 不 ontable
(stack b1 b4)
```

Self-critique 又抓到: "step 7 pick-up b1, 但 b1 在 b4 上, 不 ontable, precondition 不满足"。

**Iteration 3**: 终于对了, LLM 学会先 unstack b1:
```
(unstack b5 b2)
(put-down b5)
(unstack b2 b1)
(stack b2 b5)
(pick-up b3)
(stack b3 b2)
(unstack b1 b4)    ← 先 unstack, 不是 pick-up
(stack b1 b4)
```

Self-critique 跑完所有 step, 输出 "the plan is correct", break。

你看, 这就是 trial-and-error, 但是在 LLM 自己的 token stream 里完成的, 没有外部 oracle, 没有人类介入。 LLM 自己当 explorer, 自己当 grader, 自己当 fixer。

---

## 数字到底多漂亮

### Blocksworld 3-5 (经典 benchmark)

| 方法 | Accuracy |
|---|---|
| Baseline (no critique) | 49.8% |
| + Self-Critique | 85.5% |
| + Self-Critique + Self-Consistency (5 votes) | **89.3%** |
| Oracle (用真实 PDDL validator 做反馈) | 91.5% |

Self-Critique + SC 离 Oracle 只差 2.2 个百分点。 也就是说 LLM 自我评估的能力, 加上 self-consistency 投票, 已经接近完美的 verifier 了。

### 跨数据集 (Table 2)

| Dataset | Baseline | + Self-Critique | Oracle |
|---|---|---|---|
| Logistics (easy) | 60.7 | **93.2** | 95.0 |
| Logistics Hard | 18.9 | 32.8 | 38.8 |
| Mini-Grid | 57.7 | 75.2 | 79.8 |
| Mini-Grid Hard | 39.7 | 43.5 | 52.3 |
| Blocksworld 3-7 | 57.2 | 79.5 | 92.7 |

Logistics easy 几乎解决了。 Hard 版本还有 gap, 主要因为 plan 长, 累积 verification error 大。

### 跨模型 (Table 5)

| Model | Baseline | + Self-Critique |
|---|---|---|
| Gemini 1.5 Pro | 49.8 | 85.5 |
| Claude 3.5 Sonnet | 68.0 | 89.5 |
| GPT-4o | 42.8 | 64.2 |
| Gemma 2-27B | - | "modest" |

注意 Gemma 2-27B 几乎不 work。 这暗示 **self-critique 是 emergent ability, 有个 model size 阈值**, 大概在 100B 以上才可靠。 小模型的 verification capability 不够, 它分不清 plan 对错, 自然无法自我修正。

---

## Self-Consistency 怎么用在这里

Wang et al. 2023 的 self-consistency 原本是: 让 LLM 生成 multiple chain-of-thought, 然后 majority vote 最终答案。

这里用法不一样: **在 critique 步骤用 self-consistency, 不在 plan generation 用**。

具体: 每次 critique, 让 LLM sample 5 次, 每次独立判断 plan 对不对, 然后 majority vote。 如果 5 次里 ≥3 次说 "wrong", 就判定 wrong, 继续 iterate。 否则 break。

为什么这么 work? 因为 critique 是 binary classification (correct vs wrong), vote 很自然。 而 plan generation 是 open-ended (有无数种 correct plan), vote 难。

复杂度:
- 没有 SC: $q = 2s$ (s 个 iteration, 每个 iteration 2 个 call — 1 generate + 1 critique)
- 有 SC, c 次 vote: $q = s + c \cdot s$ (critique 变成 c 个 call)
- 但 SC 的 c 个 call 可以并行, 不增加 latency

paper 里 $s = 10, c = 5$, 理论上限 60 个 call per problem, 但实际平均只有 14 个 call, 因为大多数 problem 在 1-2 iteration 就 break 了。

---

## Mystery Blocksworld — 一个好玩的 stress test

这 dataset 把 Blocksworld 的所有 predicate 和 action 名字都换掉, 换成完全无关的 deceptive 名字:

- `clear` → `province`
- `ontable` → `planet`
- `handempty` → `harmony`
- `holding` → `pain`
- `on` → `craves`
- `pick-up` → `attack`
- `put-down` → `succumb`
- `stack` → `overcome`
- `unstack` → `feast`

Domain definition 的 structure 完全一样, 就是名字换了。 这测试的是: **LLM 是在做纯 symbolic manipulation, 还是在利用 commonsense knowledge (比如 "block 要 clear 才能 pick up")?**

如果 LLM 依赖 commonsense, 换名字后应该大幅退化。 结果:

| 方法 | Accuracy |
|---|---|
| Baseline | 22.3% |
| + Self-Critique | 35.2% |
| + Self-Critique + SC | 37.8% |
| Oracle | 37.3% |

Baseline 确实掉了很多 (Blocksworld 原版 baseline 49.8%, Mystery 只有 22.3%), 说明 commonsense 贡献了一部分。 但 self-critique 的 relative gain 巨大 (22.3 → 37.8, 几乎翻倍)。 而且 **self-critique + SC (37.8) 略高于 Oracle (37.3)**, 虽然 confidence interval 重叠, 但这说明 SC 在这种 hard domain 上可能有额外的 exploration benefit, 不只是 noise reduction。

这是第一个在 Mystery Blocksworld 上达到 non-trivial accuracy 的工作。 之前 Stechly et al. 只有 4%。

---

## 整个方法的 limitation

我读完想到的几个问题:

**1. Context 累积爆炸**

每轮 iteration 都 `τ ← τ ⊕ plan ⊕ critique`, context 越来越长。 Blocksworld plan 短还好, Mini-grid 的 problem definition 本身就长, paper 在 Appendix 承认: "If a prompt exceeds this length limit at any step, we prematurely terminate the process"。 所以实际上 iteration 数被 context length 限制, 不是被 $k=10$ 限制。

**2. False Positive 没根除**

Figure 4 显示: accuracy 高, recall 高, precision 低。 意思是: **当 plan 错的时候, critique 大概率能抓出来 (high recall); 但当 plan 对的时候, critique 有时也说它错 (low precision)**。 这导致 loop 一直跑, 即使 plan 已经对了。

Self-consistency 缓解了这个问题 (5 次 vote 比单次更可靠), 但 Blocksworld 3-7 的 Oracle 是 92.7%, self-critique 只有 79.5%, 这 13% gap 几乎全是 false positive 导致的 (LLM 说 "wrong" 但其实 plan 对, 又瞎改一通)。

**3. 纯 inference-time, 没有 learning**

模型 weight 没动, 所有 "改进" 都是 in-context 的。 每个 new problem 都要重跑 loop, 计算成本高 (14-50 个 LLM call per problem)。 如果能把这些 (plan, critique) pair 蒸馏成 training data, finetune 一个 plan-verifier 出来, 可能更 efficient。

**4. 只测 correctness, 没测 optimality**

Paper 只衡量 "plan 能 reach goal 吗", 没衡量 plan length / step count。 在实际应用里 (机器人, logistics scheduling), 一个 correct but 冗长 plan 可能比 optimal plan 差很多。

**5. Self-critique 是 scale 的 emergent ability**

Gemma 2-27B 几乎不 work。 说明这能力有 model size 阈值。 小模型用不了这方法, 这对 open-source 社区是个坏消息。

---

## 这个工作在更大图景里的位置

### 和 Self-Refine (Madaan et al. 2023) 的关系

Self-Refine 是 similar idea, LLM 自己生成 feedback, 自己 refine。 但 Self-Refine 在 general task (sentiment, code, dialog) 上做, 没有 structured domain definition。 这篇 paper 的贡献是: **在 structured planning domain 上, structured prompt 让 self-refine 终于 work**。

### 和 Reflexion (Shinn et al. 2023) 的区别

Reflexion 用 verbal reinforcement, 把失败经验累积成 memory, 但它依赖 external evaluator (environment reward)。 这篇 paper: no external evaluator, intrinsic only。

### 和 Tree of Thoughts (Yao et al. 2023a) 的关系

ToT 显式做 tree search, branching + heuristic evaluation。 这篇 paper 是 linear search (single trajectory), 但用 self-critique 做 implicit pruning (如果 critique 说错, 就不继续这条 trajectory, 而是 revise)。

### 和 ReAct (Yao et al. 2023b) 的关系

ReAct 是 reasoning + acting 交错。 这篇 paper 是 planning + verification 交错, 结构类似, 但更 structured。

### 和 Chain-of-Thought (Wei et al. 2022) 的关系

这篇 paper 的 self-critique prompt 本质是 structured CoT — 每一步 verify precondition 就是 CoT 的 reasoning step。 但 CoT 是 forward reasoning, self-critique 是 verification, 是 CoT 的一个 specific application。

---

## 我觉得最深的 takeaway

**1. Verification ≪ Generation 在 complexity 上, LLM exploit 了这个 gap**

Plan generation 是 PSPACE-hard search, plan verification 是 polynomial simulation。 LLM 在 search 上弱, 但在 simulation 上 OK。 Self-critique 让 LLM 做 simulation 来 guide search。

**2. Prompt structure 决定 self-critique 能不能 work**

Generic "critique this" 失败, "for each action: 1. get preconditions, 2. check, 3. apply, 4. output state" 成功。 LLM 有能力做 systematic procedure, 但不会自发启动, 必须 prompt 显式 trigger。

**3. Self-consistency on critique 比 on generation 更 effective**

Critique 是 binary vote (correct/wrong), 容易 aggregate; generation 是 open-ended, 难 aggregate。 在 critique 上用 SC 把 accuracy 推到接近 oracle。

**4. Self-critique 是 scale 的 emergent ability**

27B 模型不 work, 100B+ 模型 work。 阈值在 capability 上, 不在 data 上。

**5. PDDL > Natural Language for planning**

结构化 representation 让 precondition explicit, verification 可靠。 NL 的模糊性让 self-critique 失败 (Table 6, NL baseline 18.5%, PDDL 40.3%, NL 加 critique 到 29.7%, PDDL 加 CoT 和 critique 到 65%)。

**6. 大部分 gain 在第一次 iteration**

Figure 2 显示 step 0 → step 1 一次性跳 20-30%, 之后缓慢爬升。 说明大多数错误是 obvious mistake (违反 precondition), 一次 critique 就抓出来。 后面 iteration 抓的是 subtle 错误, diminishing return。

---

## 我会怎么 extend

1. **Value function distillation**: 把 self-critique 的 output 蒸馏成 value head, finetune 到模型上, 替代 in-context loop。 把 inference-time 的 search cost 转成 training-time 的 one-time cost。

2. **MCTS integration**: self-critique 当 UCB 里的 value estimate, expansion 用 PlanGeneration, backup 用 critique confidence。 这样就能真的 explore 多条 trajectory, 而不是 linear search。

3. **Active context management**: τ 单调增长不可持续。 用 retrieval 选 relevant past failure 塞进 context, 而不是全 dump。

4. **Plan optimization, not just correctness**: 加 penalty for plan length, 让 self-critique 也 critique optimality, 不只 reach goal。

5. **Hierarchical planning**: high-level plan (粗粒度) + low-level refinement, self-critique 在两层分别 run。 这能 scale 到长 plan domain (Logistics Hard)。

6. **Cross-domain transfer learning**: 在 Blocksworld 上 bootstrap 出来的 self-critique prompt template, 是否 transfer 到 Logistics / Mystery / AutoPlanBench 的新 domain? Paper 用 zero-shot critic 部分回答了 (yes), 但更 systematic 的 transfer study 有价值。

---

## 总结一句话

这篇 paper 的 message 用最短的话说就是: **LLM 不是不能自我批评, 是你不会让它自我批评**。 给它一个 explicit 的, structure-respecting 的 verification procedure (每一步 check precondition, apply action, output state), 它就能做到接近 oracle 的自我评估。 这是 prompt engineering > capability assumption 的一个 clean demonstration, 也是 neural-symbolic 的一个 nice example — 把 symbolic planner 的 cycle (action → state transition → check goal) 嵌进 LLM 的 token stream 里。

paper 链接和相关 reference:
- 主 paper: Google DeepMind, Bohnet et al. 2024
- Valmeekam et al. 2023a: https://arxiv.org/abs/2310.08118 (LLM cannot self-critique — 被反驳的)
- Huang et al. 2024: https://arxiv.org/abs/2310.01798 (LLM cannot self-correct reasoning)
- Bohnet et al. 2024: https://arxiv.org/abs/2406.13094 (planning baseline)
- Stechly et al. 2024b: https://arxiv.org/abs/2402.08115 (self-verification limitations)
- Wang et al. 2023 (Self-Consistency): https://arxiv.org/abs/2203.11171
- Wei et al. 2022 (CoT): NeurIPS 2022
- Madaan et al. 2023 (Self-Refine): https://openreview.net/forum?id=S37h0erQLB
- Yao et al. 2023b (ReAct): https://openreview.net/forum?id=WE_vluYUL-X
- Singh et al. 2024 (self-training scaling): https://arxiv.org/abs/2312.06585
- Stein et al. 2024 (AutoPlanBench): https://arxiv.org/abs/2311.09830
- Agarwal et al. 2024 (Many-shot ICL): https://arxiv.org/abs/2404.11018
- Gemini 1.5: https://arxiv.org/abs/2403.05530
- Gemma 2: https://arxiv.org/abs/2408.00118

希望这个 version 更直觉。 本质上这篇 paper 就教会我们一件事: **别指望 LLM 自动做 systematic reasoning, 你得在 prompt 里把 procedure 写死, 它就会照着做, 而且做得很好**。

---

# Enhancing LLM Planning through Intrinsic Self-Critique — 深度解读

Andrej, 这篇 paper 很有意思, 因为它直接挑战了之前几篇 high-profile 工作的结论 (Huang et al. 2024, Valmeekam et al. 2023a), 那些工作声称 "LLMs cannot self-correct" 或 "LLMs cannot effectively critique their own plans"。 Google DeepMind 的这群人提出的反驳是: **self-critique 失败的原因是 prompt 设计错了**, 而不是 LLM 本身不具备这个能力。

让我把这篇 paper 拆开讲。

---

## 1. 核心问题与历史背景

Planning 是 classical AI 的地盘, 经典 planner 比如 Fast Downward, FF, LAMA 这些基于 PDDL (Planning Domain Definition Language, McDermott et al. 1998) 的 symbolic planner 在 Blocksworld 这种 toy domain 上几乎 100% solve。 LLM 进来后, Valmeekam et al. (2023b) 的 "On the planning abilities of large language models" 做了 systematic 的 critical investigation, 发现 GPT-4 在 Blocksworld 3-5 blocks 上只有 ~35-40% accuracy, 而且加 self-critique 反而变差 (因为 high false positive rate)。

后续 Bohnet et al. (2024) 的 "Exploring and benchmarking the planning capabilities of large language models" 用 many-shot in-context learning 把数字推到 ~57% (Blocksworld 3-7)。 这篇 paper 在这个 baseline 上加 self-critique, 推到 79.5%。

Reference:
- Valmeekam et al. 2023a: https://arxiv.org/abs/2310.08118
- Valmeekam et al. 2023b: NeurIPS 2023
- Valmeekam et al. 2023c: https://arxiv.org/abs/2302.06706
- Huang et al. 2024: https://arxiv.org/abs/2310.01798
- Bohnet et al. 2024: https://arxiv.org/abs/2406.13094
- Stechly et al. 2024b: https://arxiv.org/abs/2402.08115

---

## 2. 方法 — Algorithm 1 详解

### 2.1 Pseudocode

```
Input: Problem definition D, max iterations k,
       self-consistency samples N,
       LLM sampling distribution P(p, n) where
       p is context, n is number of samples
Output: Final plan

τ ← 0                                    # 初始化 context (累积的失败历史)
for step = 0 to k do
    plan ← PlanGeneration(D, τ)         # 步骤 i: 生成 plan
    critique ← SelfCritique(D, plan, N)  # 步骤 ii: 自我批评
    if critique deems plan correct then
        break                            # early stopping
    τ ← Revise-Prompt(τ, plan, critique) # 把 (失败plan, critique) append 到 context
return plan
```

### 2.2 三个核心 helper function

**PlanGeneration(D, τ)**:
```
prompt ← PlanPrompt(D, τ)   # PlanPrompt = domain definition + few-shot exemplars + 累积失败
return P(prompt, 1)          # 单次 sample
```

**SelfCritique(D, plan, N)**:
```
prompt ← CritiquePrompt(D, plan)
return Self-Consistency(P(prompt, N))  # N 次 sample, majority vote
```

**Revise-Prompt(τ, plan, critique)**:
```
return τ ⊕ plan ⊕ critique   # ⊕ 是字符串拼接, 把上一次的 plan 和它的 critique 都 append 进 context
```

### 2.3 Intuition: 为什么这个循环能 work

关键 insight 在于: **plan generation 和 plan verification 是两个不同 difficulty 的任务**。

Planning 是 forward search: 你需要从 initial state 推到 goal, 这是一个 PSPACE-hard 的搜索问题 (Blocksworld 是 PSPACE-complete, Bylander 1994)。 但是 verification 是 polynomial 的 — 你只需要按顺序 apply actions, 检查每个 precondition, 看最后是否 reach goal。

LLM 在 forward generation 上弱, 因为它本质上是一个 left-to-right 的 token predictor, 没有 backtracking, 没有 search。 但它在 step-by-step verification 上强, 因为这本质上是 chain-of-thought reasoning, 而且每一 step 都可以从 prompt 里的 domain definition 直接读出 precondition。

所以这个 loop 利用的 asymmetry 是: **让 LLM 做它擅长的事 (verification), 用 verification 结果来引导它做不擅长的事 (search)**。

### 2.4 为什么之前的 self-critique 失败了

Valmeekam et al. 2023a 用的是 generic self-critique prompt, 类似 "Please critique the following plan: ...". 这种 prompt 没有 domain definition, 没有 precondition 的明示, 没有 step-by-step 的 instruction。 结果 LLM 做的 "critique" 是基于 vibes 的, 比如 "this plan looks reasonable", 产生了大量 false positive (说错的 plan 是对的)。

这篇 paper 的核心贡献之一是: **self-critique prompt 必须包含 (1) 完整 domain definition 含 preconditions 和 effects, (2) "for each action: 1. Take action and preconditions, 2. Verify preconditions met, 3. Apply action and give resulting state" 的 explicit instruction**。

---

## 3. Self-Critique Prompt 设计 — 这是 paper 的灵魂

### 3.1 Zero-shot Self-Critique Prompt (Appendix A.2)

```
Given the domain definition: {domain_pddl}

So, for each action:
1. Take the action and its preconditions from the domain definition
   for the specific action.
2. Verify whether the preconditions are met for the action.
3. Apply the action and provide the resulting state.

The problem to solve: {instance}
The suggested solution: {plan}

Please carefully evaluate the plan. Verify each step as described above.
Do not stop until each action is verified; please *do not* omit steps.
Conclude with the assessment literally either with
'the plan is correct', 'the plan is wrong', or 'goal not reached'.
```

### 3.2 为什么这三条 instruction 关键

Ablation study (Table 4) 显示:

| Ablation | 11th step accuracy | Δ vs. full method |
|---|---|---|
| Full method (8-shot + SC 5 votes) | 84.6 ± 2.2 | baseline |
| No self-consistency, 8-shot critic | 79.7 ± 2.5 | -4.9 |
| 0-shot critic (no few-shot) | 79.5 ± 2.5 | -5.1 |
| No Domain Definition | 74.4 ± 2.7 | -10.2 |
| No 3-Step Instruction | 64.0 ± 3.1 | **-20.6** |
| No "verify each action" | 57.5 ± 3.1 | **-27.1** |

注意最后两行: 去掉 "verify each action" 这句 instruction, accuracy 从 84.6 掉到 57.5, 几乎回到了 baseline (57.1)。 这说明 **LLM 不会自发地去做 step-by-step verification**, 除非你显式地告诉它怎么做。

这是 prompt engineering 上一个很重要的 lesson: **当你想让 LLM 做一个 multi-step 的 systematic task, 你必须把 procedure 显式写出来, 而不是期望它 infer 出来**。

### 3.3 一个完整的 trace (Appendix D)

Paper 在 Appendix D 给了一个完整的 self-improvement trace, 极其有教育意义。 我摘关键部分:

**Iteration 1**: LLM 生成 plan
```
(unstack b5 b2)
(put-down b5)
(unstack b2 b1)
(put-down b2)
(pick-up b3)
(stack b3 b2)
(pick-up b5)
(stack b5 b3)
(pick-up b2)        # ← bug: b3 在 b2 上, b2 不 clear 也不 ontable
(stack b2 b5)
(pick-up b1)
(stack b1 b4)
```

**Self-Critique 输出** (LLM 一步一步 verify):
```
Step 9: (pick-up b2)
  action: pick-up ?ob
  preconditions: (and (clear ?ob) (ontable ?ob) (handempty))
  verification:
    (clear b2): FALSE (on b3 b2 from step 6)
    (ontable b2): FALSE (on b3 b2 from step 6)
    (handempty): TRUE
  preconditions are NOT met.
the plan is wrong because the preconditions for picking up b2 in step 9 are not met.
```

**Iteration 2**: LLM 看到 critique 后, 重新生成
```
(unstack b5 b2)
(put-down b5)
(unstack b2 b1)
(stack b2 b5)       # ← 不再 put-down b2, 而是直接 stack 到 b5 上
(pick-up b3)
(stack b3 b2)
(pick-up b1)        # ← 新 bug: b1 在 b4 上, 不 ontable
(stack b1 b4)
```

**Self-Critique 输出**:
```
Step 7: (pick-up b1)
  preconditions: (and (clear ?ob) (ontable ?ob) (handempty))
    (clear b1): TRUE
    (ontable b1): FALSE  (b1 is not on the table)
    (handempty): TRUE
  precondition not met: (ontable b1) is false.
the plan is wrong. the plan fails at step 7 because b1 is not on the table.
```

**Iteration 3**: 终于对了
```
(unstack b5 b2)
(put-down b5)
(unstack b2 b1)
(stack b2 b5)
(pick-up b3)
(stack b3 b2)
(unstack b1 b4)     # ← 先 unstack b1, 而不是 pick-up
(stack b1 b4)
```

这个 trace 完美展示了: **LLM 自己能发现错误, 但需要在 critique 的引导下, 经过多次 trial-and-error 才能修复**。

---

## 4. 复杂度分析 — Equation 1 & 2

### 4.1 没有 self-consistency

$$q = 2 \times s$$

- $q$: total number of LLM calls (upper bound)
- $s$: maximum number of self-critique iterations (paper 用 $s = 10$)
- 系数 $2$: 每个 iteration 包含 2 个 LLM call — 1 个 PlanGeneration + 1 个 SelfCritique

### 4.2 有 self-consistency

$$q = s + c \times s$$

- $c$: number of self-consistency samples per critique step (paper 用 $c = 5$)
- 第一项 $s$: PlanGeneration 仍然每个 iteration 1 次 call (单 sample)
- 第二项 $c \times s$: SelfCritique 每个 iteration 需要 $c$ 次 call (用于 majority voting)

注意 paper 特别指出: **self-consistency 不增加 latency**, 因为这 $c$ 次 critique call 可以并行执行。 这是个很实际的工程考量。

### 4.3 实际 call 数远低于 upper bound

Table 4 的 "LLM calls" 列:
- Full method (SC 5 votes): 14.0k calls (理论 upper bound = 5×10×1000 = 50k, 但实际 14k, 因为 early stopping)
- No SC, 8-shot critic: 6.3k (理论 20k)

也就是说 **~70% 的 problems 在 1-2 个 iteration 内就 self-deem correct 并 break**。

---

## 5. 实验 — 主结果

### 5.1 Table 1: Blocksworld 3-5

| Method | No-Critique | Critique | Critique+SC | Oracle |
|---|---|---|---|---|
| Stechly et al. 2024b (GPT-4) | 40 | 55 | - | 87 |
| **This work (Gemini 1.5 Pro)** | 49.8 ± 4.0 | 85.5 ± 2.8 | **89.3 ± 2.5** | 91.5 ± 2.3 |

注意几个观察:
1. **Self-Critique + Self-Consistency (89.3) 几乎追平了 Oracle (91.5)**。 Oracle 是用真正的 PDDL validator 做反馈的 upper bound。 这意味着 LLM 的自我评估能力, 配合 self-consistency 后, 已经接近完美 verifier。
2. Stechly et al. 用 GPT-4 + Natural Language 只有 40% baseline, 加 critique 到 55%。 这篇用 Gemini 1.5 Pro + PDDL 起点就是 49.8%, 加 critique 直接到 85.5%。 这个 gap 一部分来自模型, 一部分来自 representation (PDDL > NL, 见 Table 6)。

### 5.2 Table 2: 跨数据集

| Dataset | No-Critique | Self-Critique | Oracle |
|---|---|---|---|
| Logistics (easy) | 60.7 ± 3.9 | **93.2 ± 1.8** | 95.0 ± 1.7 |
| Logistics Hard | 18.9 ± 2.4 | 32.8 ± 2.9 | 38.8 ± 3.9 |
| Mini-Grid | 57.7 ± 3.9 | 75.2 ± 3.5 | 79.8 ± 3.2 |
| Mini-Grid Hard | 39.7 ± 4.0 | 43.5 ± 4.0 | 52.3 ± 4.0 |
| Blocksworld 3-7 | 57.2 ± 3.1 | 79.5 ± 2.5 | 92.7 ± 1.6 |

Intuition 解读:
- **Logistics easy 几乎解决了** (93.2% vs Oracle 95%, gap 只有 1.8%)。 Logistics 的 action space 比 Blocksworld 大 (有 truck, airplane, package, location 多种类型), 但 easy version 的 plan length 短, 所以 self-critique 容易。
- **Logistics Hard gap 大** (32.8 vs 38.8, gap 6%; 但 absolute 提升只有 13.9%)。 Hard 版有 8 个 package, plan 可能 30+ steps, LLM 在长 plan 上 verification 容易出错 (cumulative error)。
- **Blocksworld 3-7 vs Oracle gap 大** (79.5 vs 92.7, gap 13.2%)。 这说明在更复杂的问题上, self-critique 的 false negative (把对的 plan 说成错的) 开始显著, 导致 loop 一直跑下去直到 step 10 还没 break, 最后的 plan 未必最优。

### 5.3 Figure 2: Accuracy vs. Iteration 曲线

这条曲线很关键。 从 paper 描述看:
- Step 0 (baseline): ~50-60%
- Step 1 (第一次 critique + revise): 跳到 ~70-80% (一次性提升 20%+)
- Step 2-10: 缓慢爬升, 最后到 ~80-90%

**主要的 gain 来自第一次 iteration**。 这有重要含义: 大多数错误是 "obvious mistakes" (违反 precondition), 一次 critique 就能抓出来。 后面的 iteration 抓的是更 subtle 的错误 (goal 没完全达到, action 顺序问题), 这些更难, marginal return 递减。

### 5.4 Figure 3: Shots scaling

- Blocksworld: 0-shot → 40%, 8-shot → 50%, 16-shot → 57%, 32-shot → 60%, 64-shot → 62%
- Mini-Grid: 类似 Blocksworld, 持续上升
- Logistics: 2-shot 就饱和了 (~60%), 更多 shots 没用

Logistics 早饱和可能是因为 Logistics 的 action template 比 Blocksworld 复杂, few-shot exemplar 的 marginal information 递减快。

### 5.5 Table 5: 跨模型

| Model | No-Critique | Self-Critique |
|---|---|---|
| GPT-4o | 42.8 ± 3.9 | 64.2 ± 3.8 |
| Claude 3.5 Sonnet | 68.0 ± 3.7 | **89.5 ± 2.5** |
| Gemini 1.5 Pro | 49.8 ± 4.0 | 85.5 ± 2.8 |
| Gemma 2-27B | - | "modest improvement" |

几个 takeaways:
1. **Claude 3.5 Sonnet baseline 最高** (68%), 而且加 self-critique 后到 89.5%, 几乎和 Gemini 持平。 这暗示 Claude 在 systematic step-by-step reasoning 上特别强。
2. **GPT-4o 提升幅度小** (42.8 → 64.2, +21.4)。 GPT-4o baseline 低 + self-critique 效果一般, 可能是因为 GPT-4o 在长 context PDDL 上不如 Claude/Gemini。
3. **Gemma 2-27B 几乎没有提升**。 这是 paper 一个诚实的 negative result: 小模型不具备 self-critique 所需的 verification capability。 Self-critique 是 emergent ability of scale。

### 5.6 Table 6: Natural Language vs. PDDL

| Prompt | No-Critique | Self-Critique |
|---|---|---|
| NL, Formatted | 18.5 ± 3.1 | 19.2 ± 3.1 |
| NL, CoT + Formatted | 20.3 ± 3.2 | 29.7 ± 3.7 |
| PDDL, Formatted | 40.3 ± 3.9 | 47.3 ± 4.0 |
| PDDL, CoT + Formatted | 39.2 ± 4.0 | 65.0 ± 3.8 |

观察:
- **PDDL > NL** (40.3 vs 18.5 baseline)。 PDDL 的结构化 representation 让 LLM 更容易 ground。
- **CoT 在 PDDL + self-critique 上 gain 巨大** (47.3 → 65.0, +17.7)。 但在 NL 上 gain 小 (19.2 → 29.7, +10.5)。 这说明 CoT 和 self-critique 都依赖结构化的 precondition 检查, 而 NL 模糊性让 verification 失败。

---

## 6. Mystery Blocksworld — 一个有意思的 stress test

Mystery Blocksworld (Appendix E.1) 把 Blocksworld 的 predicates 和 actions obfuscate 成 deceptive 名字:
- `clear` → `province`
- `ontable` → `planet`
- `handempty` → `harmony`
- `holding` → `pain`
- `on` → `craves`
- `pick-up` → `attack`
- `put-down` → `succumb`
- `stack` → `overcome`
- `unstack` → `feast`

Domain definition 完全一样, 只是名字换了。 这测试的是: **LLM 是否在用 commonsense knowledge 而不是纯粹的 symbolic manipulation**。

结果 (Table 1):
- No-Critique: 22.3%
- Self-Critique: 35.2%
- Self-Critique + SC: 37.8%
- Oracle: 37.3%

注意 **Self-Critique + SC (37.8) 略高于 Oracle (37.3)**! 这个统计学上不显著 (置信区间重叠), 但说明 self-consistency 在这种 hard case 上不仅没害处, 还可能有 marginal benefit (可能因为 SC 的 diversity 探索了更多 plan space)。

paper 在 conclusion 里特别强调: **这是第一个在 Mystery Blocksworld 上达到 non-trivial accuracy 的工作** (之前 Stechly et al. 只有 4%)。

---

## 7. AutoPlanBench (Table 3) — 泛化性测试

AutoPlanBench (Stein et al. 2024) 包含 10 个新 domain (Goldminer, Rovers, Grid, Grippers, Satellite, Depot, Movie, Ferry, Vistall 等)。 这测试 method 是否 overfit 到 Blocksworld。

| Dataset | Act (golden feedback) | CoT 1-shot | Self-Critique |
|---|---|---|---|
| Goldminer | 30 | 20 | 32.3 ± 9.3 |
| Rovers | 50 | 10 | 7.6 ± 6.4 |
| Grid | 70 | 20 | 53.3 ± 7.6 |
| Grippers | 55 | 75 | 79.0 ± 6.4 |
| Satellite | 90 | 50 | 91.4 ± 4.6 |
| Depot | 20 | 15 | 20.9 ± 3.8 |
| Movie | 100 | 100 | 100.0 ± 0.0 |
| Ferry | 40 | 95 | 61.9 ± 6.0 |
| Vistall | 100 | 85 | 99.0 ± 1.9 |

观察:
- **Movie 和 Vistall 接近 100%** (trivial domain)
- **Satellite (91.4) 接近 Act baseline (90)**, Act 用的是 per-step golden feedback (相当于 cheating), self-critique 在没有 oracle 下达到 oracle 水平
- **Rovers 很差** (7.6 vs CoT 10), 这个 domain 复杂 (多 rover, 多 camera, 多 objective), 1-shot 不够
- **Ferry 退步** (61.9 vs CoT 95), 这个值得警惕 — 可能 self-critique 的 false negative 把对的 plan 给 reject 了

注意每个 domain 只有 21 instances, 置信区间很宽 (±9.3 这种), 所以这些数字要谨慎解读。

---

## 8. 局限性 — Paper 没明说但可以推断的

### 8.1 Context length 累积

每次 iteration 都 `τ ← τ ⊕ plan ⊕ critique`, context 单调增长。 Blocksworld 还好 (plan 短), 但 Mini-grid 的 problem definition 长, paper 在 Appendix B 提到: "If a prompt exceeds this length limit at any step, we prematurely terminate the process"。 这意味着在长 plan domain 上, iteration 数实际被 context length 限制, 而不是 $k = 10$。

### 8.2 False Positive 问题没根除

Figure 4 显示: **accuracy 高, recall 高, precision 低**。 也就是说 LLM 仍然倾向说 "plan is correct" 即使它不对。 Self-consistency 缓解了这个问题 (5 次 vote 比单次更准), 但没根除。 在 Blocksworld 3-7 上 Oracle 是 92.7%, self-critique 只有 79.5%, 这 13% gap 几乎全是 FP 导致的 (LLM 说对了, 但其实不对, 提前 break)。

### 8.3 没有 training, 纯 inference-time

这既是优点 (plug-and-play) 也是缺点 (没学到东西)。 所有 "改进" 都是 in-context 的, 模型 weight 没动。 这意味着每个新 problem 都要重跑整个 loop, 计算成本高。

### 8.4 Plan quality vs. Correctness

Paper 只衡量 "plan correct (reach goal)", 没衡量 plan length / optimality。 Oracle 的 plan 可能更短更优, self-critique 的 plan 可能 correct 但冗长。 在实际应用 (机器人, logistics) 中, optimality 可能比 correctness 更重要。

---

## 9. 和相关工作的关系网

### 9.1 Self-Refine (Madaan et al. 2023)
https://openreview.net/forum?id=S37h0erQLB
- 类似 idea: LLM 自己生成 feedback, 自己 refine
- 区别: Self-Refine 在 general task (sentiment, code, dialog) 上, 没有结构化 domain definition
- 这篇 paper 的贡献: 在 structured planning domain 上, 结构化 prompt 让 self-refine 终于 work

### 9.2 Reflexion (Shinn et al. 2023)
- 用 verbal reinforcement (自然语言 feedback) 累积到 memory
- 依赖 external evaluator (environment reward)
- 这篇 paper: no external evaluator, intrinsic only

### 9.3 Tree of Thoughts (Yao et al. 2023a)
- 显式 tree search, branching
- 需要 heuristic evaluation
- 这篇 paper: linear search (single trajectory), 但用 self-critique 做 implicit pruning

### 9.4 ReAct (Yao et al. 2023b)
https://openreview.net/forum?id=WE_vluYUL-X
- Reasoning + Acting 交错
- 这篇 paper: planning + verification 交错, 类似结构

### 9.5 Self-Consistency (Wang et al. 2023)
https://arxiv.org/abs/2203.11171
- Sample multiple CoT, majority vote
- 这篇 paper 把 self-consistency 用在 critique 而不是 plan generation 上, 这是 novel application

### 9.6 Chain-of-Thought (Wei et al. 2022)
- 这篇 paper 的 self-critique prompt 本质上是 structured CoT (每一步 verify precondition 就是 CoT 的 reasoning step)

### 9.7 Singh et al. 2024 — Beyond Human Data
https://arxiv.org/abs/2312.06585
- 用 self-training (filter + finetune) 扩展 reasoning
- 这篇 paper: inference-time, no finetuning, 互补方向

### 9.8 MCTS (Coulom 2007)
- Paper conclusion 提到可以 swap in-context learning 给 MCTS
- Intuition: self-critique 可以作为 MCTS 的 value function approximation

---

## 10. Intuition 总结 — 我从这篇 paper 学到什么

1. **Verification ≪ Generation in complexity, and LLM exploits this gap**. Plan generation 是 search (PSPACE-hard), plan verification 是 simulation (P). LLM 在 search 上弱, 但在 simulation (step-by-step apply rules) 上 OK。 Self-critique 本质是让 LLM 做 simulation 来 guide search。

2. **Prompt structure matters enormously for self-critique**. Generic "critique this" 失败; "for each action: 1. Get preconditions, 2. Check, 3. Apply, 4. Output state" 成功。 这说明 LLM 需要被显式告知 procedure, 不能 infer。

3. **Self-consistency on critique > self-consistency on generation**. 在 Blocksworld 上, 5-vote critique 把 accuracy 从 85.5 推到 89.3 (+3.8), 接近 oracle。 Critique 是 binary classification (correct/wrong), vote 效果好; generation 是 open-ended, vote 难。

4. **Self-critique 是 emergent ability of scale**. Gemma 2-27B 几乎不 work, Gemini 1.5 Pro / Claude 3.5 Sonnet work。 阈值可能在 100B+ params 附近。

5. **PDDL > Natural Language for planning**。 结构化 representation 让 verification 更可靠, 因为 precondition 是 explicit 的。 NL 的模糊性让 self-critique 失败 (Table 6)。

6. **False positive 是主要 failure mode**。 LLM 倾向说 "looks good"。 这和人类 review code 的 bias 类似 (你写的东西你自己看着觉得对)。 Self-consistency 是 mitigation。

7. **Most gain in 1 iteration**。 Figure 2 显示第一次 critique 抓 most obvious errors。 后面 iteration 是 diminishing return, 但 still positive (没恶化)。

---

## 11. 我会怎么 extend 这个工作

如果我来做 follow-up:

1. **Value function learning**: self-critique 的 output (correct/wrong + explanation) 可以蒸馏成一个 value head, finetune 到模型上, 替代 in-context loop。
2. **MCTS integration**: 用 self-critique 作为 UCB 中的 value estimate, expansion 时用 PlanGeneration, backup 时用 critique 的 confidence。
3. **Plan optimization, not just correctness**: 加一个 penalty for plan length, 让 self-critique 也 critique optimality。
4. **Cross-domain transfer**: 在 Blocksworld 上 bootstrap 出来的 self-critique prompt, 是否 transfer 到 Logistics / Mystery? Paper 用 zero-shot critic 已经部分回答了这个 (yes)。
5. **Active context management**: τ 单调增长不可持续。 用 retrieval 选 relevant past failures, 而不是全 dump 进 context。
6. **Hierarchical planning**: high-level plan (粗粒度) + low-level refinement (细粒度), self-critique 在两层分别 run。

---

## 12. References 汇总

- Paper 本身: Google DeepMind, Bohnet et al. 2024 (这篇)
- Valmeekam et al. 2023a: https://arxiv.org/abs/2310.08118 (LLM cannot self-critique plans — 被反驳的)
- Valmeekam et al. 2023b: NeurIPS 2023 (PlanBench)
- Valmeekam et al. 2023c: https://arxiv.org/abs/2302.06706 (Blocksworld 3-5 dataset)
- Huang et al. 2024: https://arxiv.org/abs/2310.01798 (LLM cannot self-correct reasoning)
- Bohnet et al. 2024: https://arxiv.org/abs/2406.13094 (planning benchmark, baseline)
- Stechly et al. 2024a: https://arxiv.org/abs/2405.04776 (CoT analysis in planning)
- Stechly et al. 2024b: https://arxiv.org/abs/2402.08115 (self-verification limitations)
- Stein et al. 2024: https://arxiv.org/abs/2311.09830 (AutoPlanBench)
- Madaan et al. 2023: https://openreview.net/forum?id=S37h0erQLB (Self-Refine)
- Shinn et al. 2023: Reflexion (NeurIPS 2023)
- Yao et al. 2023a: Tree of Thoughts
- Yao et al. 2023b: https://openreview.net/forum?id=WE_vluYUL-X (ReAct)
- Wang et al. 2023: https://arxiv.org/abs/2203.11171 (Self-Consistency)
- Wei et al. 2022: Chain-of-Thought (NeurIPS 2022)
- Singh et al. 2024: https://arxiv.org/abs/2312.06585 (self-training scaling)
- McDermott et al. 1998: PDDL technical report
- Coulom 2007: MCTS
- Agarwal et al. 2024: https://arxiv.org/abs/2404.11018 (Many-shot ICL)
- Gemini Team 2024: https://arxiv.org/abs/2403.05530 (Gemini 1.5)
- OpenAI GPT-4o 2024: GPT-4 technical report
- Anthropic 2023: Claude model card
- Gemma 2: https://arxiv.org/abs/2408.00118

---

这篇 paper 的核心 message 我觉得是: **"LLMs can't self-critique" 的结论是 premature 的, 它反映的是 prompt 设计的失败, 不是模型能力的缺失**。 一旦你给 LLM 一个 structure-respecting verification procedure, 它就能做 near-oracle 的自我评估。 这是一个 prompt engineering > capability assumption 的 lesson, 也是一个 "structure exploitation > raw intelligence" 的 lesson。

如果你 (Andrej) 在读这篇, 我猜你会同意: 这本质上是把 chain-of-thought 的 "show your work" 思想 push 到了 verification 上 — 不只是让 LLM "think step by step", 而是让它在每一步 explicit 地 enumerate preconditions, check them, apply effects, output state。 这是把 symbolic planner 的 cycle (action → state transition → check goal) 嵌进 LLM 的 token stream 里。 Neural-symbolic 的一个干净 example。
