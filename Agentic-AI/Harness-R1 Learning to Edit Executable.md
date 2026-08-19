---
source_pdf: Harness-R1 Learning to Edit Executable.pdf
paper_sha256: 606cdee79be0bf17313e811f920f364f3be1e729e50ce13f18f70b013e3552f3
processed_at: '2026-08-19T10:31:50-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用大白话再讲一遍。

## 1 这篇paper到底在干嘛

你有一个agent，比如 Qwen3.5-9B，你让它去跑 WebShop（网购）、ALFWorld（家务）、DBBench（数据库操作）这些任务。它会失败——比如在 WebShop 上找到商品但没选颜色就点了 Buy Now，在 ALFWorld 上拿起杯子但忘了放进微波炉加热，在 DBBench 上 SQL 查询写错列名。

传统的改进路径是 fine-tune 这个 model 本身。但这篇paper说：**model 先别动，我们来改 model 外面包着的那层 code**。

这层 code 叫 harness。它负责：给 model 拼接 context、把 model 的 action 转发给 environment、把 environment 的反馈传回给 model。你可以把它理解成 agent 的"操作系统"——model 是 CPU，harness 是 kernel + drivers + middleware。

paper 的核心 claim 是：**我们训练一个专门的 9B 小 model 来当"harness 工程师"，让它看 target agent 失败的 trajectory，然后写一个 code patch 去修复 harness，修完之后重新跑一遍，看 task success 有没有提升。有提升就 reward 这个工程师，没有就惩罚。用 RL（具体是 GRPO）来训练这个工程师。**

## 2 为什么要这么做——现有方法的痛点

痛点 1：**直接 prompt 一个大 model 来改 harness，不靠谱。** 你让 GLM-5.2 或 GPT-5.5 看 failure trace 然后写 patch，它们写出来的东西看起来很合理（syntax valid、逻辑通顺），但因为它们从不 re-run target 去验证，所以经常写出"看起来对但实际让 performance 变差"的 patch。Figure 1 里，Self-Refine 在三个 benchmark 上全部降低 reward，frontier model editor 也不稳定。

痛点 2：**现有的 harness optimization pipeline 都用 fixed proposer。** Meta-Harness、AutoHarness、AHE 这些系统让一个 coding agent 去 search/synthesize harness，但 proposer 本身的 weights 从来不从 editing outcome 里学习。HarnessX 甚至用了 GRPO，但训练的是 task model 不是 editor。

痛点 3：**prompt-level optimization（APE、OPRO、DSPy、EvoPrompt）的 edit space 太窄。** 它们只能改 prompt text 或 few-shot demos，无法表达 state tracking、conditional control flow、action veto 这种 imperative logic。

Harness-R1 的回答：**把 harness editing 变成一个 well-defined RL problem，让一个 dedicated 9B engineer 从 rerun outcome 里学习，而不是靠 scale 或 plausibility。**

## 3 方法怎么做的——failure 到 patch 到 rerun 的 loop

整个 pipeline 是一个 loop，分四步：

### 3.1 第一步：跑 baseline，挖 failure

冻结的 target agent A 在 batch B（n 个 task）上跑一遍，拿到 baseline trajectory $\tau_i^0$ 和 reward $R_i^0$。一个 deterministic extractor 把 failed episode 的关键信息（task constraints、action-observation 节选、outcome、必要 env state）压缩成一个 failure packet $s_B$。

### 3.2 第二步：engineer 生成 patch

Harness engineer $H_\theta$（一个 9B model）读 failure packet $s_B$，生成一个 executable overlay $P$。这个 overlay 是一个 JSON，里面包含若干 `add_code_hook` entries，每个 entry 指定一个 lifecycle 位置和对应的 Python function。

可以 hook 的位置有四个，覆盖了 agent-runtime interaction 的全部 I/O boundary：

- `on_init`：episode 开始前。setup 初始 context、tool hint。
- `make_pre_hint`：target 做决策之前。注入 state-conditioned message，告诉它"现在该做 X"。
- `on_before_action`：target propose 了 action 但还没 execute。可以 block、rewrite、或 force action。
- `on_post_step`：env feedback 之后。如果 trajectory stall 了，注入 recovery hint。

这些 hook 只触碰 frozen policy 的 input/output，不改 policy 本身。相当于在 model 外面套了一层"中间件"，拦截和加工它的输入输出。

### 3.3 第三步：rerun，算 reward

Patch 安装好后，同一个 frozen target 在同一个 batch B 上重新跑一遍（包括原本成功的 task 也要重跑，防止 regression）。拿到新的 reward $R_i^P$。

Reward 定义：

$$\Delta_B(P) = \frac{1}{n} \sum_{i=1}^{n} \left( R_i^P - R_i^0 \right)$$

- $B$：一个 batch 的 n 个 task
- $R_i^0$：patch 之前 target 在 task $i$ 上的 reward
- $R_i^P$：patch 之后 target 在 task $i$ 上的 reward
- $\Delta_B(P)$：整个 batch 上的平均 reward 变化

如果 patch invalid（parse 失败）、no-op（没产生实际干预）、incomplete（rerun 没跑完），reward 直接归零。

### 3.4 第四步：用 GRPO 更新 engineer

对每个 failure packet，engineer 采样 K=8 个 candidate patches。每个都 rerun 一次拿到 reward $r_k$。然后做 group-relative normalization：

$$\hat{A}_k = \frac{r_k - \mu_B}{\sigma_B}$$

- $r_k$：第 k 个 candidate patch 的 reward
- $\mu_B$：这 8 个 candidate 的平均 reward
- $\sigma_B$：这 8 个 candidate 的标准差
- $\hat{A}_k$：第 k 个 candidate 的 advantage

直觉：**baseline 是"同一个 failure packet 上当前 policy 的 local average"**，不是 global baseline。这样避免了不同 failure packet 之间 reward scale 差异（WebShop 是连续 reward 0-1，ALFWorld/DBBench 是 binary 0/1）导致的训练不稳定。

然后用 clipped surrogate 更新 engineer：

$$g_{k,t}(\theta) = \min\left\{ \rho_{k,t} \hat{A}_k, \ \text{clip}(\rho_{k,t}, 1-\epsilon_\ell, 1+\epsilon_h) \hat{A}_k \right\}$$

- $\rho_{k,t}(\theta)$：importance sampling ratio，new policy prob / old policy prob，用于 off-policy correction
- $\epsilon_\ell = 0.20$：lower clip bound
- $\epsilon_h = 0.28$：upper clip bound（asymmetric，允许正方向 update 幅度略大）
- $\hat{A}_k$：sequence-level advantage，被同一个 response 的所有 token share

注意几个关键设计：
- **只更新 engineer θ，target agent 全程 frozen**。这避免了 credit assignment 混乱。
- **No format-validity bonus, no explicit KL penalty**。因为 cold-start SFT 已经给了 strong prior。
- **Same-batch transductive objective**：用同一批 task before/after 比较，控制 task composition。但没有跨 batch 的 patch memory，也没有 instance-level iterative refinement——这是 explicit limitation。

## 4 训练的 two-stage

### 4.1 Stage 1: Cold-start SFT

先用 GPT-5.5 当 teacher，从 failure packet 生成 candidate patch。过滤条件：executable + complete rerun + non-regressive（$\Delta_B(P) \geq 0$）。最终 877 个 examples。

Loss 是标准 next-token prediction：

$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{\sum_{j=1}^{M} |y_j^T|} \sum_{j=1}^{M} \sum_{t=1}^{|y_j^T|} \log H_\theta(y_{j,t}^T \mid s_j, y_{j,<t}^T)$$

- $M$：SFT example 数（877）
- $|y_j^T|$：第 j 个 teacher response 的 token 长度
- $y_{j,t}^T$：第 j 个 response 的第 t 个 token
- $s_j$：第 j 个 failure packet
- 分母做 token-level normalization

目的：让 engineer 先学会"什么是 valid executable patch"这个 prior，否则 online RL 起步时几乎全部 reward=0。

### 4.2 Stage 2: Online GRPO

从 SFT checkpoint 开始，用 Algorithm 1 的 loop 训练：

```
repeat:
  sample update bundle U from cached RL records
  set θ_old = θ
  for each (B, s_B, R_B^0) in U:
    sample 8 candidates from H_{θ_old}(·|s_B)
    for each candidate:
      parse + validate → patch P_k
      install P_k, rerun frozen target on B, compute r_k
      if invalid/no-op/incomplete: r_k = 0
    normalize {r_k} into advantages {Â_k}
  update only θ via Eq.4
until budget exhausted
```

Base trajectories、rewards、failure packets 都预先 cache 好，online evaluation 只 rerun patched target。

## 5 实验结果的核心数字

### 5.1 主表（Table 1）

| Method | ALFWorld All | WebShop Succ. | DBBench Succ. | Avg. |
|--------|-------------|---------------|---------------|------|
| Qwen3.5-9B baseline | 40.6 | 31.2 | 61.0 | 44.3 |
| ReAct | 43.4 | 37.4 | 61.7 | 47.5 |
| Self-Refine | 39.0 | 29.0 | 57.3 | 41.8 |
| Reflection (success@2) | 59.2 | 43.6 | 64.7 | 55.8 |
| GLM-5.2 (frontier editor) | 45.0 | 36.0 | 65.3 | 48.8 |
| GPT-5.5 (frontier editor) | 43.2 | 36.6 | 64.0 | 47.9 |
| Supervised-only engineer | 39.4 | 38.6 | 61.3 | 46.4 |
| **Harness-R1** | **53.2** | **42.2** | **65.3** | **53.6** |
| Agent SFT | 71.2 | 42.6 | 63.7 | 59.2 |
| Agent SFT + Harness-R1 | **84.0** | 43.0 | 65.7 | **64.2** |

四个关键 take：

1. **Harness-R1 比 baseline +9.3pp**，三个 benchmark 全提升。
2. **9B trained engineer (53.6) > 397B prompted editor (48.8)**。Outcome-grounded training 打败 model scale。
3. **Online RL 比 SFT-only 多 7.2pp**。单纯 imitate GPT-5.5 的 patch 不够，必须从 rerun outcome 里 refine。
4. **Agent SFT + Harness-R1 还有 +5.0pp**。即使 model 本身 fine-tune 过了，harness editing 依然有 gain，说明 harness 和 model weight 是互补的 adaptation surface。

### 5.2 Cross-target generalization（Figure 3 + Table E.1）

训练时只见过 Qwen3.5-9B 这一个 target。trained engineer 应用到 20 个 unseen targets 上：

- 21 个 target 里 **56/63 target-benchmark 组合 improve**
- 只有 3 个 small regression（≤2.0pp），全在 WebShop
- 20 个 unseen targets 平均 **+7.06pp**
- 最强 gain：Qwen3-4B (+12.3), Gemma-3-4B (+12.0), Gemma-3-12B (+10.4)

注意：**这是 editing policy transfer，不是 patch transfer**。每个 unseen target 先跑出自己的 failure traces，engineer 读这些新 traces 生成 target-specific patch。Engineer 学到的是"看到某种 failure pattern 时如何诊断 + 如何写 executable patch + 如何让 patch non-regressive"这个 meta-skill，跨 target 可迁移。

### 5.3 Held-out task generalization（Figure 4a + Table F.1）

更严格：每个 engineer 只看 10 个 failure，生成 1 个 benchmark-specific patch，apply 到剩下 1270 个 held-out task。

| Engineer | Valid patches | Held-out Δ (1270 tasks) |
|----------|--------------|-------------------------|
| Harness-R1 | 9/9 | **+8.9 ± 1.5** |
| Qwen3.5-397B | 8/9 | -4.3 ± 2.5 |
| DeepSeek-V4-Pro | 6/9 | -0.4 ± 3.6 |

Harness-R1 在 3 个 seed 上都 positive（±1.5 spread），两个 frontier editor 都 straddle zero 甚至 negative。**把 few failures 转换成 broadly useful edit 这个能力，scale 本身给不了。**

### 5.4 Lifecycle position ablation（Figure 4b + Table G.1）

| Configuration | WebShop | ALFWorld | DBBench | Avg. |
|---------------|---------|---------|---------|------|
| Full patch | 41.6 | 52.1 | 65.4 | 53.1 |
| w/o episode init | 41.5 | 51.3 | 63.8 | 52.2 (-0.9) |
| w/o pre-decision | 41.5 | 50.4 | 65.6 | 52.5 (-0.6) |
| w/o pre-action | **31.5** | 50.7 | 65.4 | 49.2 (-3.9) |
| w/o post-feedback | 41.7 | **41.9** | 65.8 | 49.8 (-3.3) |
| No intervention | 31.8 | 40.7 | 60.1 | 44.2 |

Insight：**不同 environment 的 dominant intervention point 不同**。WebShop 靠 pre-action mediation（拦截 premature purchase），ALFWorld 靠 post-feedback recovery（stall 时注入 recovery hint）。不能简单加成一个 universal importance ranking。

## 6 Qualitative Cases 的直觉（Appendix H）

### 6.1 WebShop：narrow guard 胜过 large controller

Batch 008 的 recurring failure：找到对的商品但 Buy Now 前没选 required option。Harness-R1 的 patch 只做一件事——拦截 normalized "Buy Now" action，当 price 超 budget 或 required option 未选时 block，让 target 重新选。

意义：**effective harness edit 不需要替换 target 的 policy**。只在 unsafe action point 做一个 low-bandwidth intervention，保留 target 的 search behavior，只改变 final outcome。从 0.667 reward → 1.0 reward。

### 6.2 ALFWorld：closed-loop multi-hook policy

Batch 045 的 patch 用了 4 个 hook 形成闭环：
- `on_init`：initialize stage state，给出 find-take-transform-place ordering
- `on_post_step`：update "是否 held"、"transform 是否完成"、"destination 是否 reached"
- `make_pre_hint`：inject stage-specific hint
- `on_before_action`：block premature/wrong placement

Harness 实际上变成了一个 mini state machine，persistent 地 track task stage。但 two-object task 会出现 destination 和 placement guidance 的 alternation loop，说明 stage tracking 还能 imperfect。

### 6.3 Gemini-3.5-Flash 的 negative case

Gemini-3.5-Flash 在 ALFWorld 上把 success 从 41.6% 降到 35.4% (-6.2pp)。原因：它生成了 broad `on_before_action` rules，force actions from locally plausible stage estimate。在 two-object task 上，它 premature 地 place 第一个 object 而不是先 collect 两个。

核心反例：**plausible diagnosis 可以 compile 成 overly aggressive runtime behavior**。Frontier model 写出"看起来对"的 patch，但因为没有 rerun feedback，它不知道自己的 patch 破坏了原本正确的 decision。

### 6.4 DBBench：schema-aware format preservation

Task 要求改"Moosehead Grand Prix"的 length。Baseline 写"4hours"（小写）失败。GLM-5.2 patch 给 general backtick guidance 但还是写"4hours"。Harness-R1 patch **触发 schema recovery → inspect existing row → 写"4 Hours"匹配 stored convention → verify before commit**。

两个 engineer 都生成了 valid executable patch，但 Harness-R1 把 schema 和 row evidence 转换成 exact stored representation。这种 format-level细节只有 outcome-grounded training 才能学到。

## 7 为什么 outcome-grounded training 能 beat frontier model

我的核心 intuition：**frontier model 的 training distribution 里没有"edit-then-rerun-then-check"这个 loop**。它们生成 patch 时，prompt 里只有 failure traces，没有"this patch actually made it worse"的负反馈。所以它们优化的是"给一个 failure，生成一个看起来对的 patch"，而 Harness-R1 优化的是"给一个 failure，生成一个真的能提高 rerun reward 的 patch"。

这跟 CodeRL / RLF 里的现象一致：**execution feedback 是不可替代的 signal**。Text-form plausibility 是 proxy，realized outcome 是 ground truth。

类比：你不会用一个"看起来懂编译"的 LLM 去优化你的 code，你会用一个 profiling-driven optimizer。Harness-R1 就是给 agent harness 做了一个 profiling-driven optimizer，只不过 optimizer 本身是一个 9B 的 learned policy。

## 8 跟 Related Work 的差异化定位

### 8.1 LLM-Based Harness Evolution

[Meta-Harness](https://arxiv.org/abs/2603.28052)、[AutoHarness](https://arxiv.org/abs/2603.03329)、[AHE](https://arxiv.org/abs/2604.25850)、[Continual Harness](https://arxiv.org/abs/2605.09998)、[HarnessX](https://arxiv.org/abs/2606.14249)、[Self-Harness](https://arxiv.org/abs/2606.09498)、[HAS](https://arxiv.org/abs/2607.03935) 这些系统都有 agentic proposer，但 **proposer 本身是 fixed 的**。Outcome 只用来 select candidate、refine patch、做 regression testing，从不直接 update proposer weights。Harness-R1 是第一个把"editor weights"作为 RL 训练对象的。

### 8.2 Algorithmic Optimization of Harness Components

[APE](https://arxiv.org/abs/2211.01910)、[OPRO](https://arxiv.org/abs/2309.03409)、[ProTeGi](https://arxiv.org/abs/2305.03495)、[TextGrad](https://www.nature.com/articles/s41586-024-08561-z)、[EvoPrompt](https://arxiv.org/abs/2309.08532)、[Promptbreeder](https://arxiv.org/abs/2309.16797)、[GEPA](https://arxiv.org/abs/2507.19457)、[DSPy](https://arxiv.org/abs/2310.03714)、[MIPRO](https://arxiv.org/abs/2406.11695) 这些方法 optimize single artifact（prompt/demo/skill），用 fixed procedure search，不 post-train proposer weights from editing outcomes。Harness-R1 trains the editor as a learned policy。

### 8.3 Learned Harness Editors

[Learning to Self-Evolve](https://arxiv.org/abs/2603.18620)、[Skill-R1](https://arxiv.org/abs/2605.09359)、[CodeSkill](https://arxiv.org/abs/2605.25430)、[AutoFlow](https://arxiv.org/abs/2407.12821)、[Weak-for-Strong](https://arxiv.org/abs/2504.04785)、[FlowSteer](https://arxiv.org/abs/2602.01664)、[HarnessBridge](https://arxiv.org/abs/2606.12882)、[Yi & Song](https://arxiv.org/abs/2607.05458) 这些方法要么 edit space 太窄（只改 context field 或 skill bank），要么把 editing 和 task solving 耦合在同一个 actor 里。Harness-R1 **isolates harness editing as a learning problem in its own right**——failure-conditioned, lifecycle-wide, online RL, dedicated engineer, frozen target。

## 9 我的整体判断

这篇 paper 的真正贡献不在于"+9.3pp"这个数字，而在于**它把"harness engineering"这个原本是"prompting + manual design"的活动，reframe 成一个 well-posed RL problem**：

- **State space**：failure packet（structured representation of failed trajectories）
- **Action space**：executable patch，constrained 到 4 个 lifecycle hook 的 JSON
- **Reward**：same-batch rerun 的 Δ reward
- **Transition**：deterministic（install patch + rerun frozen target）

这是一个 completely well-defined MDP，且 reward grounded in execution。这跟 prompt engineering 的"vibes-based"优化有本质区别。

更深一层，这指向一个未来：**deployed agent system 的 improvement 不再是"重新 train model"这一条路，而是"model weights + harness code + harness editor"三个可优化 surface 的 co-evolution**。Harness-R1 给出了 harness editor 这个 surface 的 first principles formulation。

Limitations 也明显：
1. Same-batch transductive objective 有 overfitting risk
2. Binary success reward 在 ALFWorld/DBBench 上很 sparse
3. 4 个 hook 的 expressivity 可能不够表达 complex control flow
4. RL loop 里每个 candidate 要 rerun full batch，cost 很高

Future work 的 natural extension：**multi-round co-evolution**——alternating 更新 target 和 engineer，看 gain 是否 compound。这跟 [EvolveR](https://arxiv.org/abs/2510.16079)、[HAS](https://arxiv.org/abs/2607.03935)、[Self-Consolidation](https://arxiv.org/abs/2602.01966) 的 co-evolution 思路呼应。

## References

- Harness-R1 paper: https://github.com/DeepExperience/Harness-R1
- Harness-R1 models: https://huggingface.co/ShaoShuai0605/Harness-R1
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [WebShop](https://arxiv.org/abs/2207.01206)
- [ALFWorld](https://arxiv.org/abs/2010.03768)
- [AgentBench](https://arxiv.org/abs/2308.03688)
- [DSPy](https://arxiv.org/abs/2310.03714)
- [MIPRO](https://arxiv.org/abs/2406.11695)
- [APE](https://arxiv.org/abs/2211.01910)
- [OPRO](https://arxiv.org/abs/2309.03409)
- [EvoPrompt](https://arxiv.org/abs/2309.08532)
- [Promptbreeder](https://arxiv.org/abs/2309.16797)
- [GEPA](https://arxiv.org/abs/2507.19457)
- [TextGrad](https://www.nature.com/articles/s41586-024-08561-z)
- [Meta-Harness](https://arxiv.org/abs/2603.28052)
- [AutoHarness](https://arxiv.org/abs/2603.03329)
- [AHE](https://arxiv.org/abs/2604.25850)
- [HarnessX](https://arxiv.org/abs/2606.14249)
- [Continual Harness](https://arxiv.org/abs/2605.09998)
- [Self-Harness](https://arxiv.org/abs/2606.09498)
- [HAS](https://arxiv.org/abs/2607.03935)
- [HarnessBridge](https://arxiv.org/abs/2606.12882)
- [Looptool](https://arxiv.org/abs/2511.09148)
- [DeepAgent](https://arxiv.org/abs/2510.21618)
- [EvolveR](https://arxiv.org/abs/2510.16079)
- [Self-evolving agent survey](https://arxiv.org/abs/2507.21046)
- [AutoFlow](https://arxiv.org/abs/2407.12821)
- [Weak-for-Strong](https://arxiv.org/abs/2504.04785)
- [FlowSteer](https://arxiv.org/abs/2602.01664)
- [Skill-R1](https://arxiv.org/abs/2605.09359)
- [CodeSkill](https://arxiv.org/abs/2605.25430)
- [Self-Consolidation](https://arxiv.org/abs/2602.01966)
- [ProTeGi](https://arxiv.org/abs/2305.03495)
- [Learning to Self-Evolve](https://arxiv.org/abs/2603.18620)
- [Yi & Song offline RL for harness](https://arxiv.org/abs/2607.05458)
- [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)
- [GLM-5.2](https://z.ai/blog/glm-5.2)
- [Kimi K2.6](https://www.kimi.com/blog/kimi-k2-6)
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)
- [Gemini 3.5 Flash](https://deepmind.google/models/model-cards/gemini-3-5-flash/)
- [GPT-5.5](https://openai.com/index/introducing-gpt-5-5/)

---

# Harness-R1：把"harness engineering"本身变成一个可学习的RL能力

这篇paper要解决一个非常具体的问题：deployed agent会持续产生interaction trajectory，但除了更新model weights，我们能否用这些trajectory来**学习如何编辑surrounding runtime（harness）**，使frozen target agent表现更好？这是一个非常"Solar Levanthine-like"的问题——它把"agent improving agent"从model-weight层面拉到了code/harness层面。

## 1. 问题的根本张力

当前self-evolving agent的文献里，主要路径是更新actor的weights（SFT、RL、online learning）。Harness optimization（context construction、tool mediation、action validation、recovery logic）通常用**fixed proposer**完成——比如prompt GPT-5去synthesis一个harness，或者用 EvoPrompt 这样的evolutionary search。问题是：**proposer本身从来不从harness edit的outcome里学习**。它对每次edit只输出一个 plausible-looking patch，但从不rerun target来验证。结果是 Figure 1 显示的：GLM-5.2、DeepSeek-V4-Pro这些frontier model去做harness editing，在WebShop上甚至会降低reward。

Harness-R1的intuition是：**harness editing的质量只能由patched target的realized rerun outcome决定，而无法由text-form plausibility决定**。这就像编译器优化一样——一段代码看起来合理不等于它真的跑得快。所以必须有一个**grounded feedback path**：failure → executable edit → rerun → reward → update engineer。

## 2. 整体架构（Figure 2解析）

整个系统的数据流是 failure-to-edit-to-rerun loop：

```
Frozen target agent A 跑 batch B → baseline trajectories τ_i^0, rewards R_i^0
              ↓
   Deterministic extractor 抽取 failed episodes → compaction 成 failure packet s_B
              ↓
   Harness engineer H_θ 读 s_B，生成 batch-conditioned overlay P
              ↓
   P 被 parse + validate 成 4 个 lifecycle hooks
              ↓
   Patched frozen target 在相同 batch B 上 rerun → new rewards R_i^P
              ↓
   Δ_B(P) = mean_i (R_i^P - R_i^0) → 作为 reward 只更新 engineer θ
```

这里的关键约束：**target agent A 全程frozen，只有 engineer θ 在学习**。这避免了actor和editor耦合训练导致的credit assignment混乱。

### 2.1 四个 lifecycle hook 的设计哲学

这4个位置不是随便选的，而是覆盖了agent-runtime interaction的全部 I/O boundary：

| Hook | 触发时机 | 作用 | 直觉 |
|------|---------|------|------|
| `on_init` | 第一个target decision之前 | setup reusable task guidance / tool hint | 给target一个"任务开始时的checklist" |
| `make_pre_hint` | target decision之前 | 注入state-conditioned message | 告诉target "现在该做X了"，但不强制 |
| `on_before_action` | target propose action之后，env execute之前 | canonicalize / rewrite / veto | 像guardrail，挡掉明显错误的action |
| `on_post_step` | env feedback之后 | inject recovery hint when stalled | 在卡住时提供recovery建议 |

注意一个细节：这些hook **只触碰 frozen policy的输入和输出**，从不修改policy本身。这相当于在外科手术上把harness当成了一个"中间件层"，而不是替换actor。Appendix C给出了每个hook的return contract——比如 `block_and_prompt` 抑制pending action并让target重新选择，`rewrite_action` 直接替换pending action，`force_action` 选一个具体action。

这种设计有几个好处：
1. **Constraint surface有限**：避免unrestricted code edit破坏interface；
2. **Verifiable**：patch能被parse成结构化JSON，invalid直接reward=0；
3. **Composable**：4个hook可以cooperate（ALFWorld的case就同时用了4个）。

## 3. 方法的formulation：把harness editing cast成batch-conditioned transductive RL

### 3.1 Reward定义

定义batch B上的performance difference：

$$\Delta_B(P) = \frac{1}{n} \sum_{i=1}^{n} \left( R_i^P - R_i^0 \right)$$

变量含义：
- $B = \{x_i\}_{i=1}^n$：一个batch的n个tasks
- $R_i^0$：unmodified frozen target在task $x_i$上的baseline reward
- $R_i^P$：安装patch P之后的frozen target在相同task $x_i$上的reward
- $\Delta_B(P)$：patch P在整个batch上的**平均reward变化**

Engineer reward：

$$r(B, P) = \begin{cases} \Delta_B(P), & \text{if valid and complete} \\ 0, & \text{otherwise} \end{cases}$$

这里有两个微妙的design choice：
- **Same-batch transductive objective**：用同一批tasks before/after做对比，控制task composition。但这意味着没有跨batch的patch memory，也没有instance-level iterative refinement——这是一个explicit limitation。
- **Invalid直接归零**：parse失败、no-op、incomplete都reward=0。这避免了"看起来合理但跑不通"的patch被credit。

注意这个reward是 **non-differentiable** 且 **only observable after edit changes target behavior**——所以不能用policy gradient直接优化，必须用online RL（GRPO）。

### 3.2 Cold-start SFT（Eq. 2）

Cold-start的目的：让engineer先学会"什么是valid executable patch"这个prior，否则online RL会从几乎全部reward=0的状态起步。

用GPT-5.5作为teacher，从frozen target的failed trajectory生成candidate patches $y_j^T$。过滤条件：**executable + complete + same-batch rerun成功 + non-regressive**（即 $\Delta_B(P) \geq 0$）。最终得到877个SFT examples（381 WebShop + 248 ALFWorld + 248 DBBench）。

Loss就是标准teacher-forced next-token prediction：

$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{\sum_{j=1}^{M} |y_j^T|} \sum_{j=1}^{M} \sum_{t=1}^{|y_j^T|} \log H_\theta(y_{j,t}^T \mid s_j, y_{j,<t}^T)$$

变量：
- $M$：SFT example数（≈877）
- $|y_j^T|$：第j个teacher response的token长度
- $y_{j,t}^T$：第j个response的第t个token
- $s_j$：第j个failure packet
- $H_\theta$：engineer policy
- 分母 $\sum |y_j^T|$ 是为了做token-level normalization

### 3.3 Outcome-Grounded GRPO（Eq. 3, 4）

这是paper的核心。Algorithm 1的伪代码逻辑：

```
repeat:
  sample update bundle U ⊂ Q_RL
  set θ_old = θ
  for each (B, s_B, R_B^0) in U:
    sample {y_k}_{k=1}^K ~ H_{θ_old}(·|s_B)   # K=8 candidates
    for k in 1..K:
      parse+validate y_k → P_k
      install P_k, rerun frozen A on B, compute r_k via Eq.1
      if invalid/no-op/incomplete: r_k = 0
    normalize {r_k} → {Â_k} via Eq.3
  update only θ via Eq.4
until budget exhausted
```

#### Advantage normalization（Eq. 3）：

$$\hat{A}_k = \frac{r_k - \mu_B}{\sigma_B}$$

变量：
- $r_k = r(B, P_k)$：第k个candidate patch的reward
- $\mu_B, \sigma_B$：**同一个failure packet s_B生成的8个candidates的empirical mean和std**
- $\hat{A}_k$：第k个candidate的group-relative advantage

这里的intuition：**baseline不是global baseline，而是"针对同一failure packet的当前policy分布"的local baseline**。这避免了不同failure packet之间reward scale差异（WebShop是连续reward，ALFWorld/DBBench是binary success）导致的训练不稳定。这正是GRPO的核心思想，源自[DeepSeekMath](https://arxiv.org/abs/2402.03300)。

#### Clipped surrogate objective（Eq. 4）：

$$g_{k,t}(\theta) = \min\left\{ \rho_{k,t} \hat{A}_k, \ \text{clip}(\rho_{k,t}, 1-\epsilon_\ell, 1+\epsilon_h) \hat{A}_k \right\}$$

$$\mathcal{I}(\theta) = \mathbb{E}\left[ \frac{1}{K} \sum_{k=1}^{K} \frac{1}{T_k} \sum_{t=1}^{T_k} w_{k,t} \, g_{k,t}(\theta) \right]$$

变量含义：
- $y_k = (y_{k,1}, \ldots, y_{k,T_k})$：第k个engineer response，被parse成patch $P_k$
- $T_k$：第k个response的长度
- $\rho_{k,t}(\theta) = \frac{H_\theta(y_{k,t} \mid s_B, y_{k,<t})}{H_{\theta_{\text{old}}}(y_{k,t} \mid s_B, y_{k,<t})}$：importance sampling ratio，新policy概率/旧policy概率
- $\epsilon_\ell = 0.20, \epsilon_h = 0.28$：asymmetric clip bounds（注意upper bound是0.28，不是常见的0.2对称——这是为了稍微allow positive deviation）
- $w_{k,t} = \text{clip}(\ell_{k,t}^{\text{tr}} - \ell_{k,t}^{\text{ro}}, 0, 2)$：truncated importance weight
  - $\ell_{k,t}^{\text{tr}}$：training engine重新计算的old-policy token log prob
  - $\ell_{k,t}^{\text{ro}}$：rollout engine记录的old-policy token log prob
  - clip到[0, 2]是为了防止off-policy divergence
- $\mathcal{I}(\theta)$：token-averaged clipped surrogate，要maximize

几个细节：
- **Sequence-level advantage sharing**：$\hat{A}_k$ 是response-level的，被所有token share。这是GRPO的特点，区别于per-token reward。
- **No format-validity bonus, no explicit KL**：通常RLHF会加KL penalty to reference policy防止collapse，这里完全去掉。原因可能是cold-start SFT已经给了strong prior，不需要KL约束。
- **Asymmetric clipping** $\epsilon_\ell \neq \epsilon_h$ 是一个不太常见的tweak——可能是想允许"探索更好方向"略微多于"远离差方向"。

## 4. 实验数据深度解析

### 4.1 主表（Table 1）

| Method | ALFWorld All | WebShop Succ. | DBBench Succ. | Avg. |
|--------|-------------|---------------|---------------|------|
| Qwen3.5-9B baseline | 40.6 | 31.2 | 61.0 | 44.3 |
| ReAct | 43.4 | 37.4 | 61.7 | 47.5 |
| Self-Refine | 39.0 | 29.0 | 57.3 | 41.8 |
| Reflection (success@2) | 59.2 | 43.6 | 64.7 | 55.8 |
| GLM-5.2 (frontier editor) | 45.0 | 36.0 | 65.3 | 48.8 |
| GPT-5.5 (frontier editor) | 43.2 | 36.6 | 64.0 | 47.9 |
| Supervised-only engineer | 39.4 | 38.6 | 61.3 | 46.4 |
| **Harness-R1** | **53.2** | **42.2** | **65.3** | **53.6** |
| Agent SFT | 71.2 | 42.6 | 63.7 | 59.2 |
| Agent SFT + Harness-R1 | **84.0** | 43.0 | 65.7 | **64.2** |

关键观察：
1. **Harness-R1 (+9.3pp) > Supervised-only (+2.1pp)**：online RL比cold-start SFT多带来7.2pp。这说明单纯imitate GPT-5.5的patch不够，必须用realized outcome来refine。
2. **Harness-R1 (53.6) > 最强frontier editor GLM-5.2 (48.8)**：9B的trained engineer > 397B的prompted editor。这是paper最重要的claim——**outcome-grounded training > model scale**。
3. **Agent SFT + Harness-R1 (+5.0pp)**：即使在agent本身fine-tune之后，harness editing还有5pp gain。这说明harness和model weight是互补的两个adaptation surface。
4. **ALFWorld的gain最大**（40.6→53.2，+12.6pp）：因为ALFWorld的失败主要是state tracking和recovery，而harness hooks恰好可以maintain persistent state across turns。

### 4.2 Cross-target generalization（Figure 3 + Table E.1）

这是最impressive的部分。训练时只见过Qwen3.5-9B这一个target，但trained engineer应用到20个unseen targets上：

- 21个target里 **56/63 target-benchmark组合improve**
- 只有3个small regression (≤2.0pp)，全部在WebShop
- 20个unseen targets平均+7.06pp
- 最强gain：Qwen3-4B (+12.3), Gemma-3-4B (+12.0), Gemma-3-12B (+10.4)

注意：**这不是patch transfer，是editing policy transfer**。每个unseen target都先跑出自己的failure traces，然后engineer读这些新traces生成target-specific patch。这跟"训练一个universal harness"完全不同——是训练了一个"universal harness editor"。

直觉解释：engineer学到的不是"WebShop该怎么edit"，而是"看到WebShop-style failure时如何诊断+如何写executable patch+如何让patch非regressive"。这种meta-skill跨target transferable。

### 4.3 Held-out task generalization（Figure 4a + Table F.1）

更严格的测试：每个engineer只看10个failure，生成1个benchmark-specific patch，apply到剩下的1270个held-out tasks。

| Engineer | Valid patches | Held-out Δ (1270 tasks) |
|----------|--------------|-------------------------|
| Harness-R1 | 9/9 | **+8.9 ± 1.5** |
| Qwen3.5-397B | 8/9 | -4.3 ± 2.5 |
| DeepSeek-V4-Pro | 6/9 | -0.4 ± 3.6 |

Harness-R1在3个seed上都positive (±1.5 spread)，而两个frontier editor都straddle zero甚至negative。这进一步说明：**frontier model优化的是"plausibility"，scale本身不能confer "把few failures转换成broadly useful edit"的能力**。

### 4.4 Lifecycle position ablation（Figure 4b + Table G.1）

| Configuration | WebShop | ALFWorld | DBBench | Avg. |
|---------------|---------|---------|---------|------|
| Full patch | 41.6 | 52.1 | 65.4 | 53.1 |
| w/o episode init | 41.5 | 51.3 | 63.8 | 52.2 (-0.9) |
| w/o pre-decision | 41.5 | 50.4 | 65.6 | 52.5 (-0.6) |
| w/o pre-action | **31.5** | 50.7 | 65.4 | 49.2 (-3.9) |
| w/o post-feedback | 41.7 | **41.9** | 65.8 | 49.8 (-3.3) |
| No intervention | 31.8 | 40.7 | 60.1 | 44.2 |

两个insight：
1. **不同environment的dominant intervention point不同**：WebShop靠pre-action mediation（拦截premature purchase），ALFWorld靠post-feedback recovery（在stall时注入recovery hint）。
2. **不能简单相加**：因为patch可以coordinate多个position，单独的importance是conditional的。这跟MoE里的router importance类似——不能脱离具体配置谈"哪个expert最重要"。

## 5. Qualitative Cases的深度直觉（Appendix H）

### 5.1 WebShop case：narrow guard > large controller

Batch 008的recurring failure：找到对的产品但Buy Now之前没选required option。Harness-R1生成的patch只做一件事：拦截normalized "Buy Now" action，当price超budget或required option未选时block。

这case的意义：**effective harness edit不需要替换target的policy**。只需要在unsafe action point做一个low-bandwidth intervention，保留target的search behavior，只改变final outcome。从0.667 reward → 1.0 reward。

### 5.2 ALFWorld case：closed-loop multi-hook policy

Batch 045的patch用了4个hook形成closed loop：
- `on_init`：initialize stage state，给出find-take-transform-place ordering
- `on_post_step`：update "是否held", "transform是否完成", "destination是否reached"
- `make_pre_hint`：inject stage-specific hint
- `on_before_action`：block premature/wrong placement

这个case的impressive之处：**harness实际上变成了一个mini state machine**，persistent地track task stage，并在每个stage给出guidance。这跟Reflection这类纯text-based reflection有本质区别——state是structured的，不是latent的。

但也有失败：two-object task会出现destination和placement guidance的alternation loop。说明stage tracking还能imperfect。

### 5.3 Negative case：Gemini-3.5-Flash的overgeneralized intervention

Gemini-3.5-Flash在ALFWorld上把success从41.6%降到35.4% (-6.2pp)。原因是它生成了broad `on_before_action` rules，**force actions from locally plausible stage estimate**。在two-object task上，它premature地place第一个object而不是先collect两个。

这个case是paper的核心反例：**plausible diagnosis可以compile成overly aggressive runtime behavior**。frontier model可以写出"看起来对"的patch，但因为没有rerun feedback，它无法知道自己的patch破坏了原本正确的decision。

### 5.4 DBBench case：schema-aware format preservation

Batch 022的failure是multi-word identifier和exact mutation format。Task要求改"Moosehead Grand Prix"的length。Baseline写"4hours"（小写）失败。GLM-5.2 patch给general backtick guidance但还是写"4hours"。Harness-R1 patch **触发schema recovery → inspect existing row → 写"4 Hours"匹配stored convention → verify before commit**。

这是一个非常subtle的对比：**两个engineer都生成了valid executable patch，但Harness-R1的patch把schema和row evidence转换成exact stored representation**。这种format-level细节只有outcome-grounded training才能学到——text-form的plausibility无法区分"4hours"和"4 Hours"。

## 6. 与related work的差异化定位

paper的related work分了3个cluster：

### 6.1 LLM-Based Harness Evolution
- [Meta-Harness](https://arxiv.org/abs/2603.28052): coding agent search over prior candidates
- [AutoHarness](https://arxiv.org/abs/2603.03329): iteratively synthesize code harnesses from env-validity feedback
- [AHE](https://arxiv.org/abs/2604.25850): jointly evolve prompts, tools, middleware, skills, sub-agents, memory
- [Life-Harness / Continual Harness](https://arxiv.org/abs/2605.09998): lifecycle interventions
- [HarnessX](https://arxiv.org/abs/2606.14249): composable typed processors + AEGIS symbolic adaptation + cross-harness GRPO on **task model**

关键差异：**以上所有work，proposer都是fixed的**。HarnessX甚至用GRPO训练task model，但训练的是actor不是editor。Harness-R1是第一个把"editor weights"作为RL训练对象的。

### 6.2 Algorithmic Optimization of Harness Components
- [APE](https://arxiv.org/abs/2211.01910), [OPRO](https://arxiv.org/abs/2309.03409): LLM propose + search over instruction candidates
- [ProTeGi](https://arxiv.org/abs/2305.03495), [TextGrad](https://www.nature.com/articles/s41586-024-08561-z): natural-language "gradients" propagate to edit prompts/code
- [EvoPrompt](https://arxiv.org/abs/2309.08532), [Promptbreeder](https://arxiv.org/abs/2309.16797), [GEPA](https://arxiv.org/abs/2507.19457): population-based mutation/selection
- [DSPy](https://arxiv.org/abs/2310.03714), [MIPRO](https://arxiv.org/abs/2406.11695): pipeline compilers, search over instructions + bootstrapped demos

差异：这些方法optimize single artifact（prompt/demo/skill），用fixed procedure search，不post-train proposer weights from editing outcomes。Harness-R1 trains the editor as a learned policy。

### 6.3 Learned Harness Editors
- [Chen et al. 2026c](https://arxiv.org/abs/2603.18620), [Vishe et al. (Skill-R1)](https://arxiv.org/abs/2605.09359), [Li et al. (CodeSkill)](https://arxiv.org/abs/2605.25430): train editor over narrow target (context field / skill bank)
- [Li et al. (AutoFlow)](https://arxiv.org/abs/2407.12821), [Nie et al. (Weak-for-Strong)](https://arxiv.org/abs/2504.04785), [Zhang et al. (FlowSteer)](https://arxiv.org/abs/2602.01664): learn to generate task-solving workflows
- [Wang et al. (HarnessBridge)](https://arxiv.org/abs/2606.12882): instruction-tune observation/action projections (no RL, no persistent code patch)
- [Yi & Song](https://arxiv.org/abs/2607.05458): select among predefined structural actions under offline RL
- [Luo et al. (HAS)](https://arxiv.org/abs/2607.03935): fold harness edits into single task actor's action space

差异：Harness-R1 **isolates harness editing as a learning problem in its own right**——failure-conditioned, lifecycle-wide, online RL, dedicated engineer, frozen target。其他方法要么edit space太窄，要么把editing和task solving耦合在同一个actor里。

## 7. 我对这篇paper的intuition和critique

### 7.1 核心贡献的真正重要性

这篇paper的真正贡献不在于"+9.3pp"这个数字，而在于**它把"harness engineering"这个原本是"prompting + manual design"的活动，reframe成一个well-posed RL problem**。具体来说：

1. **State space**：failure packet s_B（structured representation of failed trajectories）
2. **Action space**：executable patch P，constrained到4个lifecycle hook的JSON
3. **Reward**：same-batch rerun的 Δ reward
4. **Transition**：deterministic（install patch + rerun frozen target）

这是一个**完全well-defined MDP**，且reward grounded in execution。这跟prompt engineering的"vibes-based"优化有本质区别。

### 7.2 为什么outcome-grounded training能beat frontier model

我猜测核心原因是：**frontier model的training distribution里没有"edit-then-rerun-then-check"这个loop**。它们生成patch时，prompt里只有failure traces，没有"this patch actually made it worse"的负反馈。所以它们优化的是"给一个failure，生成一个看起来对的patch"，而 Harness-R1优化的是"给一个failure，生成一个真的能提高rerun reward的patch"。

这跟 CodeRL / RLF 里的现象一致：**execution feedback是不可替代的signal**。Text-form plausibility是proxy，realized outcome是ground truth。

### 7.3 Failure-conditioned learning的深层意义

这个formulation让我想到 [Looptool](https://arxiv.org/abs/2511.09148) 和 [Self-Harness](https://arxiv.org/abs/2606.09498) 的工作——closing the data-training loop。Failure-conditioned learning意味着：

> **Agent不再从success里学，而是从failure的systematic pattern里学。**

这跟Reflection的区别：Reflection让actor自己reflect，actor的reflection能力受限于actor自己。Harness-R1让一个**separate engineer**从actor的failure里学，engineer可以observe到actorobserve不到的patterns（比如"actor在WebShop上系统性地premature purchase"）。

### 7.4 Co-evolution的未来方向

paper的limitation部分提到一个很exciting的方向：**multi-round co-evolution**，alternating更新target和engineer。这相当于：

```
Round 1: Train engineer E_1 on frozen target A_0 → A_0 improved to A_0'
Round 2: Fine-tune target A_0 → A_1 using A_0' generated successful trajectories
Round 3: Retrain engineer E_2 on frozen target A_1 → A_1 improved to A_1'
Round 4: ...
```

这种alternation是否能yield compounding gains？还是会saturate？这跟 [EvolveR](https://arxiv.org/abs/2510.16079)、[HAS](https://arxiv.org/abs/2607.03935)、[Self-Consolidation](https://arxiv.org/abs/2602.01966) 的co-evolution思路呼应，是一个非常有意思的方向。

### 7.5 我看到的potential issues

1. **Same-batch transductive objective的overfitting risk**：reward只看同一batch的Δ，没有held-out task的explicit optimization。虽然held-out test表现不错，但理论上可能overfit到training batch的failure distribution。Paper提到future work可以加held-out performance term。

2. **Reward sparsity in ALFWorld/DBBench**：binary success reward，加上 K=8 candidates的group normalization，如果8个candidates都fail（reward=0），advantage全部是0，这一步不update。这在training早期可能很常见。

3. **Patch interface的expressivity限制**：4个hook + structured JSON是一个相对约束的action space。如果一个failure需要更复杂的control flow（比如multi-step conditional branching across hooks），现在的interface可能不够。Paper里ALFWorld case用了4个hook cooperate，但two-object task还是失败——这可能就是expressivity ceiling。

4. **Cost of rerun in RL loop**：每个candidate要rerun frozen target on full batch B。如果batch有100个tasks，每个task平均20 turns，K=8 candidates，那一个update bundle就要跑 100×20×8 = 16000次target forward。虽然target是frozen可以cache KV，但env interaction的cost是real的。这可能限制了training scale。

### 7.6 跟更广义的"AI improving AI"议程的关系

最近一个很hot的agenda是 "AI improving AI" / "self-improving agents"。相关工作的lineage：

- **Weight-level self-improvement**: [Self-Rewarding LMs](https://arxiv.org/abs/2401.10020), [STaR](https://arxiv.org/abs/2203.14465), [EvolveR](https://arxiv.org/abs/2510.16079)
- **Context-level self-improvement**: [Reflexion](https://arxiv.org/abs/2303.11366), [ExpeL](https://arxiv.org/abs/2308.10144), [Self-Refine](https://arxiv.org/abs/2303.17651)
- **Harness-level self-improvement (this paper)**: 把context-level从"per-task reflection"扩展到"persistent executable patch across tasks"

Harness-R1的位置很特别：它不是改进actor的weights，也不是在context里加reflection，而是**修改actor和env之间的runtime layer**。这个layer在传统agent literature里是手工设计的（比如 ReAct 的prompt template, [Toolformer](https://arxiv.org/abs/2302.04761) 的tool use protocol, [Voyager](https://arxiv.org/abs/2305.16291) 的skill library）。Harness-R1说：**这个layer也可以是learned的**。

### 7.7 跟DSPy/MIPRO的关系和区别

[DSPy](https://arxiv.org/abs/2310.03714) 和 [MIPRO](https://arxiv.org/abs/2406.11695) 也optimize "harness components"（instructions + bootstrapped demos），但有几个关键区别：

| 维度 | DSPy/MIPRO | Harness-R1 |
|------|-----------|-----------|
| Edit target | prompt text + few-shot demos | executable code hooks |
| Search method | Bayesian optimization / random search | online GRPO |
| Editor | fixed LLM | learned 9B engineer |
| Feedback | task metric (one-shot) | same-batch rerun Δ |
| Edit space | declarative program structure | imperative runtime hooks |

最大区别：DSPy的output是declarative program，Harness-R1的output是imperative code。后者可以express state tracking, conditional control flow, action mediation——这些是prompt-level optimization无法表达的。

### 7.8 一个我特别感兴趣的细节：asymmetric clip bounds

paper里 $\epsilon_\ell = 0.20, \epsilon_h = 0.28$。这种asymmetric clip在GRPO文献里不常见。我的猜测是：

- Lower bound 0.20：当 $\rho < 0.8$ 时clip，防止"远离good response"的update幅度太大
- Upper bound 0.28：当 $\rho > 1.28$ 时clip，允许"靠近better response"的update幅度略大

这是一个mild exploration bonus的机制。可能是因为harness editing的action space很大，需要slight bias towards exploration。但paper没有详细解释这个choice，可能只是hyperparameter tuning结果。

### 7.9 对future work的speculation

结合paper的limitation和我的理解，几个可能的extension：

1. **Multi-round co-evolution**: 已经提到，最自然的next step。
2. **Hierarchical harness editing**: 现在每个hook是独立的function。可以想象 hierarchical patch，比如 `on_init` 里install一个sub-engineer 来动态生成 `make_pre_hint` 的content。
3. **Cross-benchmark patch transfer**: 现在每个benchmark训练一个engineer。能否训练一个universal engineer，看到任意benchmark的failure都能生成patch？这需要更abstract的failure representation。
4. **Safety-aware harness editing**: 现在的reward只有task success。可以加 safety constraints，比如 "patch不能force action that violates env policy"。
5. **Interpretability of learned patches**: paper的Appendix H做了qualitative analysis，但没有automated的方式去understand learned patches。可以想象用一个interpreter model来annotate learned patches，反推出"engineer学到了什么strategy"。

## 8. 总结性的intuition

如果让我用一句话总结Harness-R1的insight：

> **Harness是agent和env之间的runtime layer，这个layer的optimization应该被cast成一个well-posed RL problem，where reward是realized task outcome，action是executable patch，state是failure packet。Scale不能替代outcome-grounded training，因为plausibility ≠ usefulness。**

这跟 compiler optimization 的类比很贴切：你不会用一个"看起来懂编译"的LLM去优化你的code，你会用一个profiling-driven optimizer。Harness-R1就是给agent harness做了一个profiling-driven optimizer，只不过optimizer本身是一个9B的learned policy。

更深一层，这指向一个未来：**deployed agent system的improvement不再是"重新train model"这一条路，而是"model weights + harness code + harness editor"三个可优化surface的co-evolution**。Harness-R1给出了harness editor这个surface的first principles formulation。

## References

- Harness-R1 paper (this): https://arxiv.org/abs/2606.06324 (推测URL，paper text里没给)
- Harness-R1 code: https://github.com/DeepExperience/Harness-R1
- Harness-R1 models: https://huggingface.co/ShaoShuai0605/Harness-R1
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300): GRPO的原始方法
- [ReAct](https://arxiv.org/abs/2210.03629): Reasoning+Acting
- [Reflexion](https://arxiv.org/abs/2303.11366): Verbal reinforcement learning
- [Self-Refine](https://arxiv.org/abs/2303.17651): Iterative self-feedback
- [WebShop](https://arxiv.org/abs/2207.01206): Web navigation benchmark
- [ALFWorld](https://arxiv.org/abs/2010.03768): Text-based embodied environment
- [AgentBench](https://arxiv.org/abs/2308.03688): Multi-environment agent benchmark
- [DSPy](https://arxiv.org/abs/2310.03714): Declarative LM program compiler
- [MIPRO](https://arxiv.org/abs/2406.11695): Multi-prompt optimization
- [APE](https://arxiv.org/abs/2211.01910): Automatic prompt engineer
- [OPRO](https://arxiv.org/abs/2309.03409): LLMs as optimizers
- [EvoPrompt](https://arxiv.org/abs/2309.08532): Evolutionary prompt optimization
- [Promptbreeder](https://arxiv.org/abs/2309.16797): Self-referential prompt evolution
- [GEPA](https://arxiv.org/abs/2507.19457): Reflective prompt evolution
- [TextGrad](https://www.nature.com/articles/s41586-024-08561-z): Backprop through LM feedback
- [Meta-Harness](https://arxiv.org/abs/2603.28052): End-to-end harness optimization
- [AutoHarness](https://arxiv.org/abs/2603.03329): Auto-synthesizing code harness
- [AHE](https://arxiv.org/abs/2604.25850): Agentic harness engineering
- [HarnessX](https://arxiv.org/abs/2606.14249): Composable adaptive harness foundry
- [Continual Harness](https://arxiv.org/abs/2605.09998): Online adaptation for self-improving agents
- [Self-Harness](https://arxiv.org/abs/2606.09498): Self-improving harnesses
- [HAS](https://arxiv.org/abs/2607.03935): Harness-aware self-evolving
- [HarnessBridge](https://arxiv.org/abs/2606.12882): Learnable bidirectional controller
- [Looptool](https://arxiv.org/abs/2511.09148): Closing data-training loop for tool calls
- [DeepAgent](https://arxiv.org/abs/2510.21618): General reasoning agent
- [EvolveR](https://arxiv.org/abs/2510.16079): Experience-driven lifecycle self-evolution
- [Self-evolving agent survey](https://arxiv.org/abs/2507.21046)
- [AutoFlow](https://arxiv.org/abs/2407.12821): Automated workflow generation
- [Weak-for-Strong](https://arxiv.org/abs/2504.04785): Weak meta-agent for strong executors
- [FlowSteer](https://arxiv.org/abs/2602.01664): Agents designing agentic workflows
- [Skill-R1](https://arxiv.org/abs/2605.09359): Agent skill evolution via RL
- [CodeSkill](https://arxiv.org/abs/2605.25430): Self-evolving coding skills
- [Self-Consolidation](https://arxiv.org/abs/2602.01966): Self-evolving agents
- [ProTeGi](https://arxiv.org/abs/2305.03495): Automatic prompt optimization with gradient descent
- [Learning to Self-Evolve](https://arxiv.org/abs/2603.18620): Self-evolution framework
- [Yi & Song offline RL for harness](https://arxiv.org/abs/2607.05458)
- [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)
- [GLM-5.2](https://z.ai/blog/glm-5.2)
- [Kimi K2.6](https://www.kimi.com/blog/kimi-k2-6)
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)
- [Gemini 3.5 Flash](https://deepmind.google/models/model-cards/gemini-3-5-flash/)
- [GPT-5.5](https://openai.com/index/introducing-gpt-5-5/)

如果你想深入某个具体方面（比如GRPO的importance sampling derivation、ALFWorld的stage tracking细节、或者co-evolution的formulation），可以告诉我，我可以再展开。
