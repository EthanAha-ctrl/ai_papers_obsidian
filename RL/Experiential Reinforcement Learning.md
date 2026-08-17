---
source_pdf: Experiential Reinforcement Learning.pdf
paper_sha256: a77ca1f39925c40429544c991d92c9e4b464eb0eab952d88be6a81738cb90dfe
processed_at: '2026-08-04T06:13:56-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ERL 用人话说

## 一句话版本

你学骑自行车，摔了之后想想"刚才重心偏右了"，下次往左偏一点，成功后身体记住这个感觉，以后不用想就能骑稳——ERL 就是把这个过程塞进 RL 训练里。

## 问题在哪

Standard RLVR 像这样一个笨学生：

你在 Sokoban 推箱子，推了 8 步，环境只告诉你"0分"。你不知道哪步推错了，也不知道该往哪改。你只能下次随机再推，运气好撞对，运气不好继续 0 分。

Qwen3-4B 在 Sokoban 上跑 standard RLVR，最后只有 **6% 成功率**。因为在推箱子的游戏里，推错一步可能直接死锁（箱子推到墙角出不来），random exploration 击中正确路径的概率极低。

这就像蒙着眼睛在迷宫里走，撞墙了不知道是哪面墙，只能重新随机走。

## ERL 的解法

ERL 给 model 加了个"反思"环节：

1. **第一次试**：model 照常做，得到环境反馈"0分，箱子卡墙角了"
2. **写反思**：model 自己写一段话——"我把箱子推到左上角了，那边是墙，推不回来了。下次应该绕到箱子右边，往中间推"
3. **第二次试**：model 带着这段反思重新做，这次成功
4. **内化**：把"在反思帮助下做对的行为"distill 回 base policy，让 model 不用反思也能直接做对

结果 Sokoban 从 6% 涨到 87%。

## 为什么这个思路 work

### Reflection 是 local credit assignment

Standard RLVR 的问题：8 步 trajectory，最后 0 分，每一步都被同一个 0 分评估。你不知道是第 3 步推错了还是第 7 步推错了。这就是 sparse reward 下的 credit assignment 难题。

Reflection 做的事情：model 自己说"第 3 步把箱子推墙角是错的"。这是 **verbal credit assignment**，比 scalar reward propagation 精准得多。第二次试就直接避开这个错误。

### Internalization 是 amortized reasoning

问题来了：训练时有反思帮忙，部署时用户不会给你反思，怎么办？

解法是 distillation。你成功的那次行为（在反思引导下做的），被监督学习直接 distill 回"没有反思的 base policy"。

这就像你学骑自行车时，一开始要 conscious 想"重心往左"，学会后身体自动就这么做了，不用想了。**Reflection 是 training wheels，学会后可以拆掉**。

### Memory 是 prior accumulation

成功的反思会被存起来，下次类似情况可以复用。这就像你积累了"骑车经验库"：摔过几次后，你有一套"什么情况该怎么处理"的经验，不用每次都从头反思。

### Gating 是稳定性保障

只对失败的尝试做反思。如果第一次就成功了，不要再反思，因为：

1. **Reward hacking**：成功后再反思，model 会找 instance-specific 捷径——"这个 specific grid 应该走这条 specific 路径"。这种反思不 generalize，反而 hurt 泛化。
2. **Off-policy 失衡**：训练早期大多数都失败，如果成功的也做 retry，gradient 被 second attempt（off-policy）主导，on-policy 信号被稀释，训练不稳定。

只对失败反思，成功保持纯 on-policy，既稳定又高效。

## 实验数据讲了什么

| Task | RLVR | ERL | 提升 | 环境 characteristics |
|---|---|---|---|---|
| Sokoban (Qwen-4B) | 0.06 | 0.87 | **+81%** | sparse reward + long horizon + irreversible action |
| FrozenLake (Qwen-4B) | 0.86 | 0.94 | +8% | sparse reward + latent dynamics |
| HotpotQA (Qwen-4B) | 0.45 | 0.56 | +11% | dense F1 reward + tool use |

**关键 insight**：ERL 在 sparse reward + latent dynamics + long-horizon 的环境下提升最大。HotpotQA 上 reward 已经比较 dense（F1 partial reward），RLVR 已经能学到不少，reflection 的边际收益小。

Ablation 里还有个有趣发现：**去掉 reflection 的 degradation 远大于去掉 memory**。说明 within-episode 的 reflection correction 是主 driver，cross-episode 的 memory reuse 是次要的。

还有个 honest 的 negative result：Olmo3-7B 在 Sokoban 上，去掉 memory 反而更好（0.24 vs 0.20）。因为 Olmo 的 self-reflective 能力弱，memory 里积累了错误的反思，反而误导后续。这说明 ERL 的效果依赖 base model 的反思质量。

## 和其他方法的区别

### 和 Reflexion / Self-Refine 的区别

Reflexion 和 Self-Refine 是 **inference-time** reflection：每次都要反思，推理成本高。ERL 是 **training-time** reflection + internalization，推理时不需要反思。

这其实是 **amortized inference** 的思路：把 expensive 的 reasoning 集成到 cheap 的 policy 中，trade 训练成本换推理效率。

### 和 HER (Hindsight Experience Replay) 的类比

HER 在机器人 RL 中，goal 没达到 A 但达到了 B，就把 B relabel 成 goal，failed trajectory 变成 successful。ERL 做了类似的事，但用 textual reflection 替代 goal relabeling：failed first attempt 通过 reflection 提取 "what went wrong"，second attempt 在这个 knowledge 引导下 success，这个 success 被强化并 internalize。

### 和 AlphaGo 的类比

AlphaGo 的 pipeline：SL policy → RL policy → MCTS rollouts（inference-time expensive）→ 可以再 distill 回 policy。

ERL 结构很像：first attempt ~ SL policy，reflection + retry ~ MCTS rollout（improved play），internalization ~ distill rollout 回 policy。都是 amortized inference 的思路。

### 和 R1-style Long CoT 的关系

DeepSeek-R1 通过 RL 让 model 在 inference 时 generate 长 reasoning chain。ERL 在 training 时做 reasoning 然后 internalize。两者其实 complementary：

- R1：推理时长 CoT，训练时不 explicit 做 reflection
- ERL：训练时 reflection + internalize，推理时可以 reflection-free

可以想象 hybrid：ERL-style training + R1-style inference。或者 ERL 的 reflection 本身可以是 long CoT 形式。

### 和 Silver & Sutton "Era of Experience" 的呼应

Silver & Sutton 说下一个 scaling regime 来自 agent-generated experience data，而非 human static text。ERL 正是这个方向的 algorithmic instance：把 failures 转成 usable learning signal，让 corrective knowledge 累积，实现 continual improvement。

## 一个更深层的直觉

ERL 的本质是 **把 inference-time 的 reasoning cost amortize 到 training time**。

Standard RLVR：训练时只拿 scalar reward，inference 时（如果用 reflection）要 expensive reasoning。
ERL：训练时做 expensive reasoning（reflection + retry），通过 distillation 内化，inference 时 reflection-free。

这其实是 **compute allocation 的 rethinking**：与其让 model 在每个 inference 时都做 reasoning，不如在 training 时集中做 reasoning，然后 distill 到 policy。Trade 训练 FLOPs 换推理 latency。

从 variational inference 角度看，reflection 是 iterative inference，internalization 是 amortized inference network。ERL 先做 expensive iterative inference（reflection），再 train an inference network（distillation）一次性给出结果。

## 局限性

1. **Wall-clock time 更长**：虽然 compute budget 匹配（10 rollouts RLVR vs 4×2 ERL），但 ERL 的 two attempts 是 sequential 的，wall-clock time 必然更长。
2. **Memory 设计 simple**：当前是 plain text system prompt，simple overwrite。复杂任务可能需要 retrieval + structured update。
3. **依赖 base model 反思能力**：Olmo 在 Sokoban 上效果差且 memory 反而 hurt，说明 ERL 依赖 base model 的 self-reflective capability。
4. **只测了 short-horizon tasks**：step budget 8。真正 long-horizon tasks（web browsing, code generation with execution）上效果未知。
5. **Reflection interpretability 没分析**：没有展示 reflection 具体生成了什么，是否真的做了 sensible credit assignment。

## 我的直觉总结

ERL 的核心 insight 其实很 elegant：**feedback 不应该只是 scalar optimization signal，应该是触发 structured reasoning 的 catalyst**。

Standard RLVR 把 feedback 压缩成数字，丢掉了大量 information。ERL 把 feedback 转成 textual reflection，保留了 structured information，然后通过 retry + internalization 把这个 information 转成 behavioral change。

这让我想到人类学习的方式：我们不是只看分数（scalar reward），我们会分析错在哪（reflection），下次改正（retry），然后内化成技能（consolidation）。ERL 把这个 Kolb 的 experiential learning cycle 塞进了 RL。

从更宏观视角，ERL 是 "Era of Experience" 的一个 concrete instance：future AI systems 的核心能力是从自己的 experience 中持续学习，而非从 human static text 中模仿。ERL 提供了一个 mechanism：把 failures 转成 structured learning signal，让 corrective knowledge 累积，实现 continual improvement。

---

**References**:
- ERL Paper: [https://arxiv.org/abs/2602.03109](https://arxiv.org/abs/2602.03109)
- Reflexion: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- Self-Refine: [https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)
- STaR: [https://openreview.net/forum?id=3ELRdg2sgI](https://openreview.net/forum?id=3ELRdg2sgI)
- HER: [https://arxiv.org/abs/1707.01495](https://arxiv.org/abs/1707.01495)
- DeepSeek-R1: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
- On-Policy Distillation: [https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)
- Search-R1: [https://openreview.net/forum?id=Rwhi91ideu](https://openreview.net/forum?id=Rwhi91ideu)
- Silver & Sutton "Era of Experience": [https://api.semanticscholar.org/CorpusID:277919528](https://api.semanticscholar.org/CorpusID:277919528)
- Kolb Experiential Learning: [Kolb, 2014](https://www.pearson.com/en-us/subject-catalog/p/experiential-learning-experience-as-the-source-of-learning-and-development/P20000000D49)

---

# ERL: Experiential Reinforcement Learning 深度解析

## 1. 论文的核心叙事

这篇 paper 的 motivation 来源于一个很尖锐的观察：standard RLVR 把 environment feedback 压缩成 scalar reward signal，policy 必须通过 undirected exploration 隐式地 discover "怎么把 failure 转成 behavioral correction"。在 sparse reward + long-horizon + latent dynamics 的环境下，这种 implicit credit assignment sample inefficient 且 unstable。

ERL 的核心 insight 是借鉴 Kolb 的 experiential learning cycle（experience → reflection → conceptualization → experimentation），把一个 explicit 的 experience–reflection–consolidation loop 嵌入到 RL trajectory 内部。Model 先做 first attempt，观察 feedback，然后显式地 generate reflection（描述 how to improve），这个 reflection condition 一个 refined second attempt，最后通过 selective distillation 把 reflection-guided improvement 内化到 base policy，使得 inference 时不需要 reflection。

这构成了一个从 SFT（imitation）→ RLVR（scalar reward optimization）→ ERL（structured experiential revision）的 progression。

## 2. ERL 算法的技术细节

### 2.1 Core Loop

给定 input $x$，model $\pi_\theta$ 执行：

**First attempt**:
$$y^{(1)} \sim \pi_\theta(\cdot | x)$$

环境给出 textual feedback $f^{(1)}$ 和 scalar reward $r^{(1)}$。

**Gated reflection**（只当 $r^{(1)} < \tau$，paper 中 $\tau = 1$）:
$$\Delta \sim \pi_\theta(\cdot | x, y^{(1)}, f^{(1)}, r^{(1)}, m)$$

其中：
- $\Delta$ 是 reflection，即 model self-generated 的 textual correction guidance
- $m$ 是 cross-episode reflection memory，存储之前 successful 的 corrective patterns

**Second attempt**:
$$y^{(2)} \sim \pi_\theta(\cdot | x, \Delta)$$

得到 $(f^{(2)}, r^{(2)})$。Reflection 的 reward 定义为 $\tilde{r} := r^{(2)}$，即 reflection 的 reward 直接等于它引导的 second attempt 的 reward——这鼓励 model 生成能 actually 改善 downstream performance 的 reflection。

**Memory update**:
$$m \leftarrow \Delta \quad \text{if} \quad r^{(2)} > \tau$$

只有导致 improved outcome 的 reflection 才会被 stored 到 memory，避免污染。

### 2.2 Policy Gradient Loss

ERL 对 first attempt、reflection、second attempt 都做 RL update：

$$\mathcal{L}_{\text{policy}}(\theta) = -\mathbb{E}[A \log \pi_\theta(y | x, \cdot)]$$

变量含义：
- $\theta$：policy $\pi_\theta$ 的参数
- $y$：model 输出，可以是 $y^{(1)}$、$\Delta$、或 $y^{(2)}$
- $x$：input task
- $\cdot$：conditioning context，对应 Algorithm 1 中每个 step 的 input（first attempt 是 $x$；reflection 是 $x, y^{(1)}, f^{(1)}, r^{(1)}, m$；second attempt 是 $x, \Delta$）
- $A$：advantage estimate，由 GRPO 计算（基于 group-relative baseline）

这里关键的一点：reflection 本身被当作 policy 的一个 output 节点来优化，而不是只作为 second attempt 的 condition。Reflection 的 reward 是 $\tilde{r} = r^{(2)}$，所以 reflection 学到的是 "如何 critique 才能 lead to successful retry"。

### 2.3 Internalization via Selective Distillation

这是 ERL 最核心的 design point。问题在于：训练时有 reflection $\Delta$ 和 feedback $f$ 作为 condition，但 deployment 时没有这些。怎么把 reflection-guided improvement "内化"成 base policy 的能力？

Solution 是 selective distillation：

$$\mathcal{L}_{\text{distill}}(\theta) = -\mathbb{E}\left[\mathbb{I}(r^{(2)} > 0) \log \pi_\theta(y^{(2)} | x)\right]$$

变量含义：
- $\mathbb{I}(\cdot)$：indicator function，只在 $r^{(2)} > 0$ 时为 1
- $y^{(2)}$：reflection-guided 的 refined attempt
- $x$：原始 input（**去掉** reflection context $\Delta$）

这个 loss 训练 $\pi_\theta$ 从 $x$ alone 生成 $y^{(2)}$，即把 "在 reflection 帮助下产生的 good behavior" distill 到 "没有 reflection 的 base policy"。

这本质上是 **self-distillation**：teacher 是 $\pi_\theta(\cdot | x, \Delta)$（reflection-conditioned），student 是 $\pi_\theta(\cdot | x)$（reflection-free）。只 distill successful trajectories（$\mathbb{I}(r^{(2)} > 0)$），避免 distill bad behavior。

### 2.4 Gating Mechanism 的 Necessity

Algorithm 2 (full version) 加入了 gating：只在 $r^{(1)} < \tau$ 时触发 reflection。Paper 在 Appendix A 解释了为什么对所有 trajectory 都做 reflection 会有问题：

1. **Reward hacking**: Reflect on already-successful attempts 时，model 会生成 instance-specific shortcuts，对当前 sample 保证 success 但不 generalize 到 future episodes。
2. **Off-policy imbalance**: 训练早期 first-attempt reward 普遍低，optimization signal 被 second attempt 和 reflection 主导（这两者 inherently off-policy 相对于 base policy），weakens on-policy learning signal，destabilize policy。

Gating 确保 successful trajectories 保持 purely on-policy，reflection 只用于 corrective revision on failed attempts。这同时 align training with deployment（deployment 时本来就没有 reflection）。

### 2.5 On-Policy Distillation Generalization

Paper 在 Appendix A 提出了一个 generalization，用 on-policy reverse KL 替代 supervised distillation：

$$\mathcal{L}_{\text{OD}}(\theta) := \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{I}(r^{(2)} > 0) \mathbb{E}_{y \sim \pi_\theta(\cdot | x)}\left[\text{KL}(\pi_\theta(\cdot | x, \Delta) \| \pi_\theta(\cdot | x))\right]\right]$$

变量含义：
- 外层期望对 $x \sim \mathcal{D}$（数据分布）
- $\mathbb{I}(r^{(2)} > 0)$：只对 successful cases 做 distillation
- 内层期望对 $y \sim \pi_\theta(\cdot | x)$：从 **deployment policy** 采样，而非 teacher
- $\text{KL}(\pi_\theta(\cdot | x, \Delta) \| \pi_\theta(\cdot | x))$：从 contextual policy（teacher）到 deployment policy（student）的 KL divergence

注意这里是 **forward KL**（$p \| q$ 形式，采样自 $q$），是 **mean-seeking** 的，让 deployment policy 覆盖 contextual policy 的所有 mode。这符合 Agarwal et al. (2024) 的 on-policy distillation framework。

## 3. 实验设置与结果分析

### 3.1 Task Configuration

**FrozenLake**:
- $n \times n$ grid, $n \in [2, 9]$ uniform sample
- Frozen tile probability $p \in [0.6, 0.85)$
- Abstract symbols：A=agent, B=goal, C=hole, D=frozen tile（故意用 abstract 而非 semantic markers，减少 pretrained symbolic priors）
- Sparse reward: +1 goal, 0 otherwise
- Step budget = 8

**Sokoban**:
- $n \times n$ grid, $n \in [6, 8]$
- Single-box, single-goal layouts
- 最短 valid solution $\leq 8$ moves（BFS 验证 solvability）
- Symbols: A=agent, a=agent on box, b=box, B=box on goal, C=goal, E=wall, D=floor
- Step budget = 8

**HotpotQA**:
- Multi-hop QA with tool-augmented retrieval
- Tool: `local_search(query, top_k)`, 基于 FAISS + e5-base-v2 embeddings + Wikipedia corpus
- Up to 5 interaction turns
- Reward: 1.0 exact match, F1 if F1 ≥ 0.3, 0 otherwise

### 3.2 Compute Matching

Paper 很仔细地做了 compute budget matching：
- RLVR: 10 rollouts/prompt
- ERL: 4 rollouts/prompt per attempt（×2 attempts = 8）+ reflection overhead

这样 total generation compute 大致匹配。但要注意 ERL 的 wall-clock time 会更长（因为 sequential 的 two attempts + reflection）。

### 3.3 Main Results (Table 1)

| Task | RLVR | ERL | ERL w/o Mem | ERL w/o Refl |
|---|---|---|---|---|
| **Qwen3-4B** | | | | |
| FrozenLake | 0.86 | **0.94** | 0.86 (-0.08) | 0.60 (-0.34) |
| HotpotQA | 0.45 | **0.56** | 0.56 (0.00) | 0.48 (-0.08) |
| Sokoban | 0.06 | **0.87** | 0.87 (0.00) | 0.59 (-0.28) |
| **Olmo3-7B** | | | | |
| FrozenLake | 0.39 | **0.66** | 0.64 (-0.02) | 0.54 (-0.12) |
| HotpotQA | 0.47 | **0.50** | 0.47 (-0.03) | 0.46 (-0.04) |
| Sokoban | 0.04 | **0.20** | 0.24 (+0.04) | 0.06 (-0.14) |

关键观察：

1. **Sokoban 上 Qwen3-4B 提升 +0.81**（0.06 → 0.87）是 paper 的 headline result。Sokoban 需要 long-horizon planning + irreversible action consequences（推错 box 可能 deadlock），standard RLVR 在这种环境下 sample inefficient。ERL 的 reflection 让 model 在 first attempt 后 explicit reason about what went wrong，second attempt 直接 benefit。

2. **HotpotQA 提升最小**（+0.11 Qwen, +0.03 Olmo）。因为 HotpotQA 的 reward 更 dense（F1-based partial reward），interaction pattern 更 homogeneous（repeated tool invocation），RLVR 已经获得 informative gradients，所以 structured experiential revision 的 marginal benefit 较小。这印证了 paper 的 claim：**ERL 在 sparse reward + latent dynamics + long-horizon 的环境下最有价值**。

3. **Reflection > Memory**：w/o Refl 的 degradation 远大于 w/o Mem。这说明 reflection 的 within-episode correction 是主要 driver，memory 的 cross-episode reuse 是 secondary。

4. **Memory 的 caveat**：Olmo3-7B Sokoban 上 w/o Mem (0.24) > ERL (0.20)。当 model 的 self-reflective 能力有限（如 Olmo 在 Sokoban 上）时，persistent memory 可能 propagate early inaccurate reflections，造成 erroneous prior accumulation，使 recovery 更困难。这是一个很 honest 的 negative result。

### 3.4 Ablation: Reflection 的 Mechanistic Role

Figure 6 显示 training reward 的 pre-reflection vs post-reflection trajectories。Post-reflection consistently 高于 pre-reflection 和 RLVR，说明 reflection 在 within-episode 产生了 actionable correction，而非只是 long-horizon shaping。

## 4. 与相关工作的关系与联想

### 4.1 与 Inference-time Reflection Methods 的区别

**Reflexion (Shinn et al., 2023)** [https://arxiv.org/abs/2303.11366]: 推理时 verbal reflection + memory, 但需要 reflection at deployment。每次 episode 后 reflect, accumulate memory。

**Self-Refine (Madaan et al., 2023)** [https://arxiv.org/abs/2303.17651]: iterative self-feedback, 推理时。Model 自己 generate feedback 然后自我 refine。

**STaR (Zelikman et al., 2022)** [https://openreview.net/forum?id=3ELRdg2sgI]: bootstrap reasoning with reasoning, 用 rationalization 让 failed attempt 变成 training data。

ERL 的关键区别是 **internalization**：训练时用 reflection + retry，通过 distillation 把 improvement 内化到 base policy，推理时不需要 reflection。这解决了 Reflexion 类方法的 inference cost 问题（每次都要 reflect）。这其实呼应了 inference-time scaling vs training-time scaling 的 tradeoff：ERL 选择把 reasoning cost amortize 到 training time。

### 4.2 与 Hindsight Experience Replay (HER) 的类比

**HER (Andrychowicz et al., 2018)** [https://arxiv.org/abs/1707.01495]: 在 goal-conditioned RL 中，通过 relabel goals 让 failed trajectories 提供 informative updates。例如 robot 没达到 goal A 但达到了 point B，就把 B 当作 goal，这个 trajectory 就变成 successful。

ERL 做了类似的事情，但是用 **textual reflection** 替代 **goal relabeling**：failed first attempt 不直接 useful，但通过 reflection，model 提取 out "what went wrong"，然后 second attempt 在这个 knowledge 引导下 success，这个 success 被强化并 internalize。两者都在解决 sparse reward 下的 credit assignment，但 HER 在 continuous control 中 relabel goals，ERL 在 LM agent 中 verbal reflection。

### 4.3 与 Self-Distillation / RL via Self-Distillation 的关系

**Hubotter et al. (2026)** [https://arxiv.org/abs/2601.20802]: formalize RL via self-distillation，distill feedback-conditioned teacher policy 到 student policy。

**Song et al. (2026)** [https://arxiv.org/abs/2602.02482]: RL with text feedback，distill feedback-conditioned teacher。

**Agarwal et al. (2024)** [https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf]: On-policy distillation of language models，learning from self-generated mistakes。

ERL 与这些工作 aligned，但 key innovation 是 explicit self-reflection 作为 intermediate reasoning step，而 non-just feedback conditioning。Reflection 是 model-generated 的 text, 比 environment feedback 更 structured 且 task-relevant。

### 4.4 与 AlphaGo 的 Policy Network 蒸馏类比

AlphaGo 的 pipeline: SL policy network → RL policy network → value network → MCTS rollouts。Rollouts 在 inference 时提供 improved play，但 cost 高。

ERL 的结构很像：first attempt ~ supervised/SL policy，reflection + retry ~ MCTS rollout (improved play)，internalization ~ distill rollout-improved play 回 policy network。这种 amortized inference 的思路是一致的：把 expensive 的 reasoning 集成到 cheap 的 policy 中。

### 4.5 与 Silver & Sutton "Era of Experience" 的呼应

**Silver & Sutton (2025)** [https://api.semanticscholar.org/CorpusID:277919528]: 提出 AI 的 next scaling regime 来自 agent-generated experience data 而非 human static text。Continual, agent-generated data streams + long-horizon decision-making。

ERL 正是这个方向的 algorithmic instance：把 failures 转成 usable learning signal，而不依赖 rare successes。Cross-episode memory 让 corrective knowledge 累积，实现 continual improvement。

### 4.6 与 Search-R1 的关系

**Search-R1 (Jin et al., 2025)** [https://openreview.net/forum?id=Rwhi91ideu]: 训练 LLMs 用 RL reasoning + leverage search engines。ERL 的 HotpotQA 实验基于 Search-R1 setup，加入 reflection-consolidation loop。

### 4.7 与 R1-style Reasoning 的关系

**DeepSeek-R1 (Guo et al., 2025)** [https://arxiv.org/abs/2505.09388]: 通过 RL incentivize long CoT reasoning，model 在 inference 时 generate long reasoning chains。

ERL 与 R1 的相似点：都通过 RL 让 model 学会 reasoning。区别在于 ERL 更 explicit 地 decompose 成 reflection-retry-internalize，且 R1 的 reasoning 全部在 inference 时，ERL 把 reasoning 通过 distillation 内化到 policy（推理时可以 reflection-free）。

可以想象一个 hybrid：训练时 ERL-style reflection + internalization，inference 时 R1-style long CoT。两者其实可以是 complementary 的。

### 4.8 与 Decision Transformer 的类比

Decision Transformer 把 RL 转成 sequence modeling，conditioned on return-to-go $R$。ERL 也可以看作 sequence modeling：conditioned on reflection $\Delta$。但 ERL 的 $\Delta$ 是 model-generated text, 不是 pre-specified scalar。这其实是更 rich 的 conditioning，textual reflection 携带的信息远多于 scalar return-to-go。

### 4.9 与 Actor-Critic 的联系

从 RL 角度看，reflection $\Delta$ 实际上扮演了 critic 的角色，只不过 critic signal 是 textual 而不是 scalar value。这让我想到 **value function 的 sparse approximation**：textual reflection 是 rich but sparse（只在 failed attempts 后 generate），scalar reward 是 sparse but simple。ERL 把两者结合，reflection 提供了 local credit assignment，reward 提供了 global signal。

### 4.10 与 Self-Play 的类比

**Jiang et al. (2026)** [https://arxiv.org/abs/2602.03109]: Multi-turn multi-agent self-play RL for conversational social intelligence。

ERL 可以看作一种 implicit self-play：
- First attempt = "actor"
- Reflection = "critic"（self-generated critique）
- Second attempt = "improved actor"
- Internalization = "distillation back to actor"

这种 self-play 不需要 external opponent，model 与自己的 past failure 对弈。

### 4.11 与 Iterative Amortized Inference 的关系

从 variational inference 角度看，ERL 在训练时做了 amortized inference：把 reflection 的 computation amortize 到 base policy 通过 distillation。推理时，model 已经 internalize 了 reflection 的 reasoning，所以不需要 explicit reflection step。这和 amortized inference 的 induction-network 思路类似：先 expensive iterative inference，再 train an inference network 一次性给出结果。

## 5. 直觉构建：为什么 ERL Work

### 5.1 Exploration 效率

Standard RLVR 在 sparse reward 下需要大量 random exploration 才能偶尔 hit reward，然后 policy gradient 慢慢 propagate。在 Sokoban 这种 long-horizon + irreversible action 的环境下，random exploration hit success 的概率极低（Qwen3-4B 只有 0.06）。

ERL 的 reflection 把 exploration 从 "random trial" 变成 "informed retry"。First attempt 的 failure 提供 information，reflection 把这个 information verbalize 成 structured correction，second attempt 在这个 correction 引导下 explore 更 promising 的 region。这相当于 **bounce exploration off the failure** 而非 **ignore failure and try again randomly**。

### 5.2 Credit Assignment

Sparse reward 的核心难题是 credit assignment：哪个 action 导致 success/failure？Standard RLVR 通过 trajectory-level advantage propagation，每个 action 都被同一个 reward 评估，long-horizon 下 credit signal 被稀释。

ERL 的 reflection 实际上做了一次 **verbal credit assignment**：model 在 reflection 中 explicit 说 "the first attempt failed because I pushed the box into the corner, next time I should push from the other side"。这个 verbal analysis 已经做了 local credit assignment，second attempt 直接 benefit。

### 5.3 Behavioral Consolidation

Reflection 和 retry 在 within-episode 提供 improvement，但 deployment 时没有 reflection。Internalization 通过 distillation 把 "在 reflection 帮助下的 good behavior" distill 到 "没有 reflection 的 base policy"。

直觉上：reflection 是 training wheels，帮 model 学会 ride bicycle。一旦学会，training wheels 可以去掉。Distillation 就是 "去掉 training wheels" 的过程。

但要注意：distillation 只对 successful cases 做（$\mathbb{I}(r^{(2)} > 0)$），避免 distill bad behavior。这和 self-distillation 中 "只 distill correct examples" 的做法一致。

### 5.4 Memory 作为 Prior Accumulation

Cross-episode memory $m$ 让 successful reflections 跨 episode 累积。这相当于 model 在 training 过程中 build up 一个 corrective knowledge base，后续 reflection 可以 reuse 这个 prior。

直觉：reflection 像 hypothesis generation，memory 像 accumulated theory。每次 successful reflection 都 refine 这个 theory，让下次 reflection 更 efficient。

但 caveat 是 memory 可能 accumulate erroneous theories（如 Olmo Sokoban 的负 effect）。这和 offline RL 中的 distribution shift 类似：prior 来自 past data，如果 past data biased，prior 也 biased。

### 5.5 为什么 Gating Critical

如果对所有 trajectory（包括 successful）都 reflect + retry，会有两个问题：

1. **Reward hacking**: Successful trajectory 上 reflect，model 会找 instance-specific shortcut，比如 "在这个 specific grid layout 下应该走这条 specific path"。这种 reflection 不 generalize，反而 hurt generalization。

2. **Off-policy dominance**: 训练早期大多数 trajectory fail，大部分 gradient 来自 second attempt（off-policy 相对于 base policy $\pi_\theta(\cdot | x)$）。这 weakens on-policy signal，destabilize optimization。

Gating 确保 successful trajectories 保持 purely on-policy，reflection 只用于真正需要 correction 的 cases。这同时 improve stability 和 efficiency。

## 6. 批判性思考与潜在局限

### 6.1 Compute 虽然匹配，但 Wall-clock Time 更长

虽然 paper 声称 compute budget matching（10 rollouts RLVR vs 4×2 ERL），但 ERL 的 two attempts 是 sequential 的（second attempt 依赖 reflection，reflection 依赖 first attempt 的 feedback），wall-clock time 必然更长。在 latency-sensitive 的 training 场景下，这可能是 issue。

### 6.2 Memory 设计 Simple

当前 memory 是 plain text system prompt, simple overwrite。Paper 在 Appendix A 提到可以用 retrieval-and-update scheme，但没实现。Complex tasks 可能需要 selective retrieval + structured update，否则 memory 会 grow unbounded 且 noisy。

### 6.3 Reflection 质量依赖 Base Model

Olmo3-7B 在 Sokoban 上提升小（+0.16）且 w/o Mem 反而更好，说明 ERL 的效果依赖 base model 的 self-reflective capability。如果 base model 的 reflection 质量差，reflection 可能 mislead second attempt，且 memory accumulate bad reflections。

这让我想到 ERL 可能需要更强的 base model，或先用 SFT bootstrap reflection 能力（类似 STaR 的 rationalization）。

### 6.4 Generalization 到 More Diverse Tasks

Paper 只在 3 个 tasks 上测试，且都是相对 short-horizon（step budget 8）。真正 long-horizon tasks（如 web browsing, code generation with execution）上 ERL 的效果如何，memory 如何 scale，都是 open questions。

### 6.5 与 Long CoT 的关系未充分探讨

R1-style long CoT 在 inference 时做 reasoning，ERL 在 training 时做 reasoning 然后 internalize。两者其实可以是 complementary：ERL-style training + R1-style inference。Paper 没有探讨这个 hybrid 的可能性。

### 6.6 Reflection 的 Interpretability 没充分分析

Paper 没有展示 reflection 的具体内容，只是 quantitative evaluation。看 reflection 实际生成了什么，reflection 是否真的做了 sensible credit assignment，reflection 是否 generalize 到 unseen instances，都是值得 future work 探讨的方向。

## 7. 总结

ERL 是一个 elegant 的 paradigm，把 explicit reflection-consolidation loop 嵌入 RL training，解决了 sparse reward 下 implicit credit assignment 的 sample inefficiency 问题。Key insights:

1. **Reflection as credit assignment**: Verbal reflection 做 local credit assignment，比 scalar reward propagation 更 efficient。
2. **Internalization as amortization**: Distillation 把 reflection-guided improvement 内化到 base policy，inference 时 reflection-free。
3. **Memory as prior accumulation**: Cross-episode memory 让 corrective knowledge 累积，实现 continual improvement。
4. **Gating as stability mechanism**: 只对 failed trajectories reflect，避免 reward hacking 和 off-policy imbalance。

实验结果在 Sokoban 上 dramatic（+81%），在 HotpotQA 上 modest（+11%），印证了 paper 的 claim：**ERL 在 sparse reward + latent dynamics + long-horizon 的环境下最有价值**。

从更宏观的视角看，ERL 是 Silver & Sutton "Era of Experience" 的 algorithmic instance：把 failures 转成 usable learning signal，而非依赖 rare successes。这正是 future AI systems 的核心能力。

---

**References**:
- ERL Paper: [Experiential Reinforcement Learning](https://arxiv.org/abs/2602.03109) (本 paper)
- Reflexion: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- Self-Refine: [https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)
- STaR: [https://openreview.net/forum?id=3ELRdg2sgI](https://openreview.net/forum?id=3ELRdg2sgI)
- HER: [https://arxiv.org/abs/1707.01495](https://arxiv.org/abs/1707.01495)
- Search-R1: [https://openreview.net/forum?id=Rwhi91ideu](https://openreview.net/forum?id=Rwhi91ideu)
- DeepSeek-R1: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
- On-Policy Distillation: [https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)
- RL via Self-Distillation: [https://arxiv.org/abs/2601.20802](https://arxiv.org/abs/2601.20802)
- Silver & Sutton "Era of Experience": [https://api.semanticscholar.org/CorpusID:277919528](https://api.semanticscholar.org/CorpusID:277919528)
- Kolb Experiential Learning: [Kolb, 2014](https://www.pearson.com/en-us/subject-catalog/p/experiential-learning-experience-as-the-source-of-learning-and-development/P20000000D49)
- Jiang et al. Self-Play: [https://arxiv.org/abs/2602.03109](https://arxiv.org/abs/2602.03109)
