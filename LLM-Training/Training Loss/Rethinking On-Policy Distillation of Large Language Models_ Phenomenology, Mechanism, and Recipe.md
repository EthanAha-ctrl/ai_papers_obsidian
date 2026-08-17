---
source_pdf: Rethinking On-Policy Distillation of Large Language Models_ Phenomenology,
  Mechanism, and Recipe.pdf
paper_sha256: 7b43c679b5179aade605a2d9636c6a03e49234e0561bb5245e4de4ca6fcee53a
processed_at: '2026-08-11T23:27:13-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 喝咖啡聊 OPD

## 一句话版本

OPD 听起来很美——让 student model 自己 rollout，teacher model 在每一步都给 dense supervision，理论上比 SFT 和 outcome RL 都好。但这篇 paper 用一系列非常干净的实验告诉我们：**teacher 强不强根本不是重点，student 和 teacher 想 problem 的方式一不一致、teacher 有没有 student 没见过的新东西，才是 OPD 能不能跑起来的关键**。

---

## OPD 到底在干嘛？

先说清楚 OPD 的画面。普通的 distillation 是 teacher 写一堆 answer，student 拿来背。这有两个问题：(1) student 推理时候走的是自己的 trajectory，跟背的 trajectory 不一样，exposure bias；(2) teacher 写的东西可能 student 根本理解不了，背得很痛苦但没学到东西。

OPD 换个思路：让 student 自己 rollout，走自己的 trajectory，然后 teacher 在 student 走的每一步上说"这步我觉得应该这么走"。这听起来很优雅——supervision 完全 on student 自己的 distribution，而且每个 token 都有 reward 信号，比 outcome RL 的稀疏 reward dense 多了。

数学上就是最小化 reverse KL：

$$D_{\text{KL}}(\pi_\theta \| \pi_T)$$

reverse KL 是 mode-seeking 的，意思是 student 会在自己当前的 mode 附近找 teacher 最喜欢的位置，不会傻乎乎地往 teacher 的 mode 跳过去（那是 forward KL 的行为）。

---

## 第一个诡异现象：强老师反而教不好

这是一个非常 counterintuitive 的实验。作者拿了 R1-Distill-1.5B（一个 1.5B 的 RL-trained math model，叫它 JustRL-1.5B）当 student，然后让它退回去——distill 回它自己 pre-RL 的版本 R1-Distill-1.5B。

结果：JustRL-1.5B 几乎完全 regression 回 pre-RL 的水平，RL 学到的东西被 overwrite 掉了。

这还不够诡异。更诡异的是，换成 R1-Distill-7B（同 family 的 7B model，benchmark 分数甚至比 JustRL-1.5B 还高一点）当 teacher，distill 出来的 trajectory 跟用 1.5B 的 R1-Distill 当 teacher **几乎完全一样**——也是 regression。

一个 benchmark 上更强的 7B 老师，教出来的效果跟一个更弱的 1.5B 老师一模一样。这彻底打破了"teacher 越强 distillation 越好"的直觉。

**为什么？** 因为 OPD 是在 student 自己 visited states 上算 KL，teacher 在这些 state 上的 local distribution 才是真正有意义的。同 family 的 1.5B 和 7B 在这些 local state 上给出的分布几乎一致——它们都是在同一份 data 上训练出来的，只是参数量不同。scale up 提升的是 in-distribution 的 capacity，没有给 student 带来任何它没见过的新 knowledge。所以 OPD 在 reverse KL 的视角下，两个 teacher 是 distributionally indistinguishable 的。

---

## 两个 Governing Conditions

从这些现象，作者提炼出 OPD 成功的两个必要条件：

### 条件一：Thinking Pattern 一致

Student 和 teacher 想 problem 的方式要接近。如果 teacher 是 non-thinking model（直接出答案），student 是 thinking model（长链推理），两者的 top-k token 集合几乎不交，teacher 给的 per-token supervision 在 student 看来就是 noise。

实验：用 Qwen3-1.7B-Base 当 student，对比 Qwen3-4B (Non-thinking) 和 Qwen3-4B-Base-GRPO（base model 上做 RL，thinking pattern 接近 base student）两个 teacher。GRPO teacher 的 benchmark 分数跟 Non-thinking teacher 差不多，但 distill 效果明显更好。关键区别在于初始 overlap ratio——student 和 teacher 在 top-k token 上的重合度——GRPO teacher 一开始就高。

### 条件二：Teacher 要有 student 没见过的新东西

即使 thinking pattern 一致、teacher benchmark 更高，如果 teacher 只是同 family scale up 的版本，OPD 没用。

实验：R1-Distill-1.5B 当 student，
- R1-Distill-7B（同 pipeline，仅 scale up）：几乎无 improvement
- Skywork-OR1-Math-7B（在 R1-Distill-7B 基础上又做了 RL）：大幅 improvement

差别就是后者有 RL 带来的新 capability。Scale up 本身只是同分布上 capacity 增强，不是新 knowledge。

---

## 机制：Token 级的 Progressive Alignment

那 OPD 成功的时候到底在 token level 发生了什么？

对比 success run（JustRL-1.5B 当 teacher）和 fail run（R1-Distill-7B 当 teacher），student 都是 R1-Distill-1.5B：

**Success run 的 signature**：
- Overlap ratio 从 72% 涨到 91%，student 和 teacher 的 top-k token 集合越来越重合
- Overlap-token advantage 趋近 0，说明在重合的 token 上 confidence 也 match 了
- Entropy gap 收窄，confidence profile 对齐
- Gradient norm 一直 substantial，optimization 有信号

**Fail run 的 signature**：
- Overlap ratio 停滞
- Entropy gap 一直大
- Gradient norm 一开始就小，而且一直小

这里最关键的发现是：**overlap tokens 承载了 97%-99% 的 probability mass**。Student 和 teacher 的分布能量几乎全在那 k 个 token 上。如果两者 top-k 几乎不交，reverse KL 几乎没东西可优化——mass 要从 student 的 mode 跳到 teacher 的 mode，中间隔着一个 low-probability desert，gradient 信号极弱。

**Ablation 验证**：把 student top-k 拆成 overlap 和 non-overlap 两部分，只在 overlap 上算 loss，性能完全 recover。只在 non-overlap 上算 loss，性能差很多。说明 OPD 的 gradient signal 主要来自 overlap region。

**Self-reinforcing dynamic**：一旦某 token 进入 overlap 并被 teacher favor，reverse KL 会加重它的 mass，把 non-overlap token 挤出 student top-k。于是 overlap 扩大，更多 token 进入良性循环。但这有个临界点——初始 overlap 要够高才能 bootstrap 这个 cycle，否则就一直卡在 low-overlap 陷阱里。

---

## 两个实用 Recipe

当 thinking pattern mismatch 严重、初始 overlap 低的时候怎么救？

### Recipe 1: Off-Policy Cold Start

简单粗暴：先做一轮 SFT。用 teacher 在一批 prompts 上生成 rollout（off-policy），student 在这些 teacher rollout 上做 supervised fine-tuning。这把 student 的初始分布拉到 teacher 附近，初始 overlap ratio 直接拔高。然后再开始 OPD。

实验：Qwen3-1.7B-Base + Qwen3-4B (Non-thinking)，thinking pattern 严重 mismatch。纯 OPD 卡住，SFT cold start 后 OPD 跑得很顺。而且 ceiling 也更高——cold start 不只是 warmup，是把优化 trajectory 拉到 self-reinforcing 的良性循环里。

这个其实很有 intuition：SFT 是 forward KL 的近似，mean-seeking，整体拉近；OPD 是 reverse KL，mode-seeking，在已经近的距离上精修。两者 time scale 和几何性质互补。

### Recipe 2: Teacher-Aligned Prompts

让 OPD 训练用的 prompt 接近 teacher post-training 时见过的 prompt。

两个粒度：
- **Template alignment**：仅仅是 prompt 模板格式跟 teacher 训练时一致（比如用 `\boxed{}` 还是 `Answer: $Answer`），就有 measurable improvement
- **Content alignment**：用 teacher RL 训练集的 prompt（DAPO-Math-17K）vs 仅 in-domain 的 deduplicated subset（DeepMath），前者更好

但有个 catch：teacher-aligned prompts 会大幅降低 student entropy，可能 collapse 掉 exploration。建议 mix OOD prompts。

Intuition：prompt 决定 student 在 training 中 visited states 的分布。如果 prompts 与 teacher post-training 一致，teacher 在这些 state 上的分布更 sharp、更 well-defined，supervision 更有效。

---

## Dense Supervision 的代价

OPD 的卖点是 dense per-token reward。但 paper 最后揭示了一个 fundamental limitation。

### Trajectory Length 的 Sweet Spot

用不同 max response length 跑 OPD：0.5K, 1K, 3K, 7K, 10K, 15K。

- 0.5K, 1K：太短，supervision 不足
- 3K, 7K：sweet spot
- 10K, 15K：性能 plateau 或下降，training 不稳定

**Instability 起源**：15K 设置下，student entropy 的高位先出现在 trajectory 末尾，然后随训练**从后往前 propagate**。Teacher entropy 也是同样的趋势。

**为什么**：teacher 训练时见过的 prefix distribution 跟 student 长 trajectory 后段的 prefix distribution mismatch。当 student 走到 16K 位置时，prefix 已经 drift 出 teacher 的 familiar region，teacher 给的 per-token 分布接近 noise。

直接验证：截断 student rollout 在不同位置，让 teacher continue。Teacher 的 accuracy advantage 从 1K prefix 的 +0.37 单调衰减到 16K prefix 的 +0.02。到 16K 位置，teacher 在 student prefix 上几乎没 advantage 了。

这是一个 fundamental tension：dense supervision 的 reliability 跟 trajectory depth 反相关。短 trajectory 上 supervision 可靠但 reasoning depth 有限；长 trajectory 上能学深度 reasoning 但 supervision 不可靠。

### Global Reward 好 ≠ Local Gradient 好

Fail run 中（R1-Distill-7B 当 teacher），sequence mean reward 在 correct rollouts 上确实比 incorrect rollouts 高，AUROC 0.75，跟 success run 的 JustRL-1.5B（AUROC 0.73）差不多。Fail teacher 给的 global signal 质量不差。

那为什么 fail？作者 hypothesis：7B teacher 的 per-token advantage 虽然 magnitude 大，但在 sequence 内不同 position 上方向不一致，aggregate 成 gradient 时互相 cancel。JustRL-1.5B 因为 thinking pattern 兼容，advantage 集中在 coherent 子集上，gradient 方向一致。

这个 hypothesis 没有直接 verify，是 paper 留下的最大 open question。如果你能做 token-level gradient direction 的 PCA 或者 cosine similarity 分析，会是个很好的 follow-up。

---

## Top-k Size 不太重要

最后一个实用发现：Top-k 的 k 不太关键，只要不是 Top-1。

- Top-1：不稳定，overlap 增长突跳，entropy 和 gradient norm 有 spike。因为总是选 argmax，policy 一点小变化就 flip rank 1 token，reward 信号 unstable。
- Top-4, 16, 64：表现接近
- Sampled-token（只看 student 实际采样的那个 token）：性能其实很好，因为它按 student 分布采样，对 high-prob region 有 unbiased 覆盖

Thinking Machines Lab 的 OPD 复现工作用的就是 sampled-token，效果不错，跟这里的发现一致。

---

## 给你的 Intuition

1. **OPD 是 conditional mass transport**。它在 student 自己 visited states 上做 reverse-KL，不是 unconditional policy matching。Teacher 的 benchmark performance 与此无关——重要的是 teacher 在 student 走过的 state 上的 local distribution。

2. **Thinking pattern alignment 是 OPD 的 phase transition critical condition**。初始 overlap ratio 是临界变量，高于阈值就启动 self-reinforcing cycle，低于阈值就卡死。

3. **Scale up teacher 在 OPD 上收益甚微**。同 family 不同 scale 在 local state 上给出近乎相同 distribution。要引入 new knowledge，得做 post-training 或者换 data/recipe。

4. **Dense supervision 不是 free lunch**。Trajectory depth 上 reward reliability 衰减，long-horizon reasoning 是 OPD 的根本瓶颈。

5. **Cold start (SFT) + OPD 是互补的**。SFT 解决初始 alignment，OPD 在 alignment 基础上做精修。两者 time scale 和 KL 几何性质互补。

6. **OPD 学的是 thinking pattern**，甚至会 overwrite student 已有的 capability。这是双刃剑——alignment 强但可能丢东西。

7. **Reward 质量 globally 好 ≠ locally exploitable**。Fail teacher 给的 global signal 不差，但 local gradient 被 anisotropic 信号 cancel 掉。这指向一个深层问题：dense reward 的有效利用需要 gradient direction 的一致性，不只是 magnitude。

希望这个版本读起来不那么费劲。技术细节和公式在第一版回答里都有，这里就是给你一个聊天的版本。

---

# On-Policy Distillation 系统性分析: Phenomenology, Mechanism, Recipe

这篇 paper 是 Tsinghua NLP 组对 On-Policy Distillation (OPD) 的一个 deep dive。OPD 已经成为 Qwen3/MiMo/GLM-5 等 model post-training pipeline 的核心组件，但其 dynamics 一直是个黑盒。作者用大量 controlled experiments 揭示了几个 counterintuitive 的现象，特别是 stronger teacher **反而** fails to improve student 的 failure mode。

GitHub: https://github.com/thunlp/OPD

---

## 1. Big Picture: 这篇 paper 在回答什么问题

核心 question：**为什么 stronger teacher 不能保证 better distillation？**

直觉告诉我们，teacher 越强，distillation 效果越好。但作者在实验中反复观察到反例：一个 benchmark 分数明显更高的 teacher，distill 出来的 student 比从 weaker teacher distill 的 student 还差。这个 gap 是 OPD 与传统 off-policy KD 最大的区别——OPD 是一个 on-policy 的 trajectory-level reverse-KL 优化问题，teacher 的强弱并不直接决定 student-visited states 上的信号质量。

Paper 的三层结构：
- **Phenomenology (§3)**: OPD 成功/失败的 empirical conditions
- **Mechanism (§4)**: token-level 上 progressive alignment 的动力学
- **Recipe (§5)**: 两个 recovery 策略
- **Discussion (§6)**: dense supervision 的 hidden cost

---

## 2. Preliminaries: OPD 的数学形式

### 2.1 核心目标

OPD 的目标是最小化 student 在自己采样的 trajectory 上、与 teacher 的 reverse KL:

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_x, \hat{y} \sim \pi_\theta(\cdot|x)} \left[ \sum_{t=1}^{T} D_{\text{KL}}(p_t \| q_t) \right] \tag{2}$$

变量含义：
- $x$: input prompt，来自数据集 $\mathcal{D}_x$
- $\hat{y}$: student 自己采样得到的 rollout，$\hat{y} \sim \pi_\theta(\cdot|x)$
- $T$: rollout length
- $p_t(v) \triangleq \pi_\theta(v | x, \hat{y}_{<t})$: student 在 prefix $\hat{y}_{<t}$ 下的 next-token 分布
- $q_t(v) \triangleq \pi_T(v | x, \hat{y}_{<t})$: teacher 在**同一个 student-generated prefix** 下的 next-token 分布
- $D_{\text{KL}}(p_t \| q_t) = \sum_{v \in \mathcal{V}} p_t(v) \log \frac{p_t(v)}{q_t(v)}$: reverse KL（注意方向是 $\pi_\theta \| \pi_T$，这是 mode-seeking）

**关键 insight**: 这里 teacher 在 **student-visited states** $\hat{y}_{<t}$ 上 evaluate，不是 teacher 自己生成的 states。这就是 on-policy 的精髓，也解释了为什么 teacher 的 benchmark 分数不直接传递——teacher 在 unfamiliar prefix 上行为可能完全不同。

### 2.2 三种实现

(1) **Sampled-Token OPD** (Eq.3) - 最轻量、最常用：

$$\ell_t^{\text{sample}} = \log p_t(\hat{y}_t) - \log q_t(\hat{y}_t)$$

直觉：只看 student 实际采样的那个 token $\hat{y}_t$ 的 log-ratio。这是 $D_{\text{KL}}(p_t \| q_t)$ 的 unbiased single-sample estimator，因为：

$$\mathbb{E}_{\hat{y}_t \sim p_t}[\ell_t^{\text{sample}}] = \sum_v p_t(v) [\log p_t(v) - \log q_t(v)] = D_{\text{KL}}(p_t \| q_t)$$

Qwen3 / Thinking Machines Lab 的复现工作用的就是这种形式。Thinking Machines Lab 的 blog 是非常好的参考: https://thinkingmachines.ai/blog/on-policy-distillation

(2) **Full-Vocabulary OPD** (Eq.4): 对全部 vocab 算 KL，gradient 密度高，但内存开销 $O(BTM)$，$B$=batch, $T$=seq_len, $M=|\mathcal{V}|$。

(3) **Top-k OPD** (Eq.5): 中间方案。选 student top-k 集合 $S_t = \text{TopK}(p_t, k)$，在 $S_t$ 上 renormalize:

$$\bar{p}_t^{(S_t)}(v) = \frac{p_t(v) \mathbf{1}[v \in S_t]}{\sum_{u \in S_t} p_t(u)}, \quad \bar{q}_t^{(S_t)}(v) = \frac{q_t(v) \mathbf{1}[v \in S_t]}{\sum_{u \in S_t} q_t(u)}$$

然后最小化 $D_{\text{KL}}(\bar{p}_t^{(S_t)} \| \bar{q}_t^{(S_t)})$。丢弃了 $S_t$ 外的 probability mass，但减少了 teacher-query cost。

### 2.3 三个动态诊断指标

这三个指标是理解后面所有现象的核心工具。

**(1) Overlap Ratio (Eq.6)** - 衡量 student/teacher top-k 集合的 alignment：

$$\mathcal{M}_{\text{overlap}} = \mathbb{E}_t \left[ \frac{|S_t^{(p)} \cap S_t^{(q)}|}{k} \right]$$

- $S_t^{(p)} = \text{TopK}(p_t, k)$: student 的 top-k 集合
- $S_t^{(q)} = \text{TopK}(q_t, k)$: teacher 的 top-k 集合
- 值接近 1: student 找到了 teacher 的 support region
- 值低: mode mismatch，student 的概率 mass 分布在 teacher 不看好的 token 上

**(2) Overlap-Token Advantage (Eq.7)** - 衡量 overlap 区域内 confidence 是否匹配：

$$A_t(v) = \bar{p}_t(v) (\log \bar{q}_t(v) - \log \bar{p}_t(v))$$

$$\mathcal{M}_{\text{adv}} = \mathbb{E}_t \left[ \frac{1}{|S_t^{(p)} \cap S_t^{(q)}|} \sum_{v \in S_t^{(p)} \cap S_t^{(q)}} A_t(v) \right]$$

- 这是 importance-weighted 形式的 advantage
- 接近 0: alignment 好，student 在 overlap 内 confidence 与 teacher 匹配
- 大的负值: student 在 overlap 内 overconfident（$p_t$ 大但 $q_t$ 小）

**(3) Entropy Gap (Eq.8)** - 衡量 confidence profile 匹配：

$$\Delta H_t = |H(q_t) - H(p_t)|$$

- 大的 gap: confidence 不匹配
- 趋于 0: student 匹配了 teacher 的 uncertainty profile

---

## 3. Phenomenology: OPD 的两个 governing conditions

### 3.1 Thinking-Pattern Consistency

**Setup**: Qwen3-1.7B-Base 作 student，对比两个 teacher:
- Qwen3-4B (Non-thinking)
- Qwen3-4B-Base-GRPO（zero-RL 训出来的）

由于 student 也是 base model，hypothesis 是 GRPO teacher 的 thinking pattern 更接近 student。

**Results (Figure 2)**: GRPO teacher distill 效果更好，而且初始 overlap ratio 更高。虽然两个 teacher 的 benchmark 表现相近（Figure 3），但因为 thinking pattern 不一致，Non-thinking teacher 在 student-visited states 上给的 supervision 不对齐。

**Intuition**: OPD 是 reverse KL，它本质是 mode-seeking 的。如果 teacher 的 mode 在 student 当前分布之外，reverse KL 不会无差别地把 mass 搬过去，而是会在 student 自己的 mode 周围寻找最接近 teacher mode 的位置。如果两者 mode 距离太远，optimal solution 可能就是"维持原状"，结果就是 distillation 信号无效。

参考 reverse KL vs forward KL 的经典讨论：
- Forward KL $D_{\text{KL}}(\pi_T \| \pi_\theta)$: mean-seeking, 要求 student 覆盖 teacher 所有 mode
- Reverse KL $D_{\text{KL}}(\pi_\theta \| \pi_T)$: mode-seeking, student 倾向于 collapse 到 teacher 的单个 mode

MiniLLM (Gu et al., 2023) 最早在 LLM 上用 reverse KL，正是出于这个考虑: https://arxiv.org/abs/2306.08543

### 3.2 New Knowledge, Not Just Scale

**Setup**: 两个 model family 的对照实验

**DeepSeek family**:
- Student: R1-Distill-1.5B
- Teacher A (same pipeline): R1-Distill-7B
- Teacher B (post-trained): Skywork-OR1-Math-7B（对 R1-Distill-7B 做 RL）

**Qwen family**:
- Student: Qwen3-1.7B (Non-thinking)
- Teacher A: Qwen3-4B (Non-thinking)
- Teacher B: Qwen3-4B-Non-Thinking-RL-Math

**Results (Figure 4)**: 两边一致——post-trained teacher 给出大幅 gain，same-pipeline teacher 几乎没 improvement。Post-trained teacher 不仅 absolute 性能更高，**gap recovery rate** 也高得多：

$$\text{Gap Recovery} = \frac{\text{Acc}_{\text{after OPD}} - \text{Acc}_{\text{before OPD}}}{\text{Acc}_{\text{teacher}} - \text{Acc}_{\text{before OPD}}}$$

**Intuition**: same-pipeline 的两个 model（同一份 pretraining data + 同一份 SFT data + 同一份 RL data，只是 model size 不同）会收敛到相似 distribution。"scale up" 带来的性能提升主要是 in-distribution capacity 增强，没有引入新 knowledge。

### 3.3 Reverse Distillation: 最惊艳的实验

**Setup**: 
- JustRL-1.5B = 对 R1-Distill-1.5B 做 RL 得到，性能更高
- 反过来 distill：JustRL-1.5B 作 student，分别用 R1-Distill-1.5B（其 pre-RL 版本）和 R1-Distill-7B 作 teacher

**Results (Figure 5)** - 真的很惊艳:
- distilling 回 R1-Distill-1.5B: student 几乎完全 regression 到 pre-RL 性能
- distilling 从 R1-Distill-7B（benchmark 分数还略高于 JustRL-1.5B）: trajectory 与上面**几乎完全一样**，也是 regression 到同样水平

**这意味着什么**：
1. **OPD 学的是 thinking pattern**。它把 student 的 thinking pattern 拉向 teacher 的 thinking pattern，甚至会把 RL 学到的 enhancement 都 overwrite 掉。
2. **Benchmark 分数完全无法预测 OPD outcome**。R1-Distill-7B 分数更高，但效果和 1.5B 的 R1-Distill 一样。
3. **同 family 不同 scale 的 model 对 student 来说 distributionally indistinguishable**。因为 OPD 是 reverse KL，它只关心 student-visited states 上的 local target distribution，而两个 same-family 不同 scale 的 model 在这些 local states 上给出近乎相同的 distribution。

这是一个非常重要的 negative result: 简单 scale up teacher 不能让 OPD 走得更远。要引入 new knowledge，必须 post-training 或者改变训练 data/recipe。

---

## 4. Mechanism: Token-Level Progressive Alignment

### 4.1 Successful vs. Failing OPD 的对比

**Setup**: R1-Distill-1.5B 作 student
- Teacher A: JustRL-1.5B (success, gap recovery >80%)
- Teacher B: R1-Distill-7B (fail, 无 improvement)

**Dynamic 对比 (Figure 6)**:

| 指标 | Success (JustRL-1.5B) | Fail (R1-Distill-7B) |
|---|---|---|
| Overlap ratio | 72% → 91% 稳步上升 | 停滞 |
| Overlap-token advantage | 趋近 0 | 停滞在大负值 |
| Entropy gap | 收窄 | 持续大 |
| PG loss | 持续下降 | 一开始就小，几乎不变 |
| Gradient norm | 大且持续 | 始终小 |

**关键观察**: overlap tokens 承载了 **97%-99% 的 probability mass**（Appendix B.1）。这意味着 student 和 teacher 的 distribution 主要能量都在那 ~k 个 token 上。如果两个 model 的 top-k 集合几乎不交，那 reverse KL 几乎无信号可优化。

### 4.2 Overlap Sufficiency: 优化实验

最关键的 ablation (Figure 7)：把 student top-k 拆成两部分，分别训练：

- **Student Top-k**: 完整 student top-k 集合 $S_t^{(p)}$（基线）
- **Overlap Top-k**: 只优化 $S_t^{(p)} \cap S_t^{(q)}$（交集）
- **Non-Overlap Top-k**: 只优化 $S_t^{(p)} \triangle S_t^{(q)}$（对称差，即各自独有的部分）

**Results**:
- Overlap Top-k ≈ Student Top-k（性能完全恢复）
- Non-Overlap Top-k 显著差

**Self-reinforcing dynamic**: overlap 区域会随训练**扩大**——一旦某 token 进入 overlap 并被 teacher favor，reverse KL 更新会加重 mass 在它上面，把非 overlap token 挤出 student top-k。这是一个 virtuous cycle，但只在初始 overlap 满足某个阈值后才会启动。

**Intuition for build**: OPD 的本质是一个 **conditional mass transport** problem。在 student 自己生成的 state $\hat{y}_{<t}$ 上，student 当前的高概率 token 集合定义为 $S_t^{(p)}$，OPD 试图把 mass 重新分配使其接近 teacher 在 $S_t^{(q)}$ 上的分布。如果两个集合几乎不相交（low overlap），那么 student mass 要 "跳过去"，需要从近零概率 token 上抬起 mass，这个 gradient 信号会非常弱。但如果 student 高概率 token 已经基本在 teacher 的 top-k 里，OPD 只是在这个 region 内做 reweighting，gradient 一致且有效。

### 4.3 Gradient 视角的辅助证据 (Appendix B.2)

三个诊断指标一致支持上面的故事：

1. **PG Loss**: success run 一开始 loss 大，逐步下降；fail run 一开始就小且几乎不变。这看起来反常——fail run 的 loss 更小？但这恰好说明 fail run 一开始就处于 teacher-induced 信号很弱的状态，没什么可优化的。

2. **Gradient norm**: success run 持续有 substantial gradient；fail run 几乎一直小。这是最直接的证据。

3. **Extreme-advantage token 的概率差** $p_t(v) - q_t(v)$: success run 持续缩小最大的 disagreement；fail run 持续保持大的 disagreement。

---

## 5. Practical Recipe: 恢复 Failing OPD 的两个策略

### 5.1 Off-Policy Cold Start

**Setup**: Qwen3-1.7B-Base 作 student，Qwen3-4B (Non-thinking) 作 teacher（thinking pattern mismatch 严重）

**Two-stage 流程**:
1. **Stage 1 (Off-policy SFT)**: 用 Qwen3-4B 在 OpenThoughts3-1.2M math subset 上生成 200K responses，过滤 incomplete / degenerate，做 full-parameter SFT 得到 Qwen3-1.7B-SFT
2. **Stage 2 (OPD)**: 用 OpenThoughts 剩余 ~30K prompts 做 OPD

SFT 关键 hyperparameters (Table 3):
- Sequence length: 14,336
- Per-device batch: 8, gradient accumulation: 1
- Learning rate: $1 \times 10^{-5}$, cosine schedule, warmup 0.05
- BF16 precision

**Results (Figure 8)**: SFT-initialized OPD 一开始就有高 overlap ratio，trajectory 平滑；base-initialized 一开始低且不稳定，后期才慢慢 recover。**最终性能 ceiling 也更高**——cold start 不只是 warmup，而是 reset 了 OPD 优化 trajectory 的初始点，让它进入 self-reinforcing 的良性循环。

**为什么有效**: SFT 把 student 的初始分布拉到 teacher 高概率 region 附近，初始 overlap ratio 高，OPD 一开始就能利用 teacher 的 dense signal。

**Intuition**: 这其实是一个**two-time-scale optimization** 问题。SFT 是 forward KL（mean-seeking）的近似，它把 student mass 整体拉近 teacher 分布；OPD 是 reverse KL（mode-seeking），它在已经近的距离上做精修。两个 KL 的 geometric 性质互补。

参考: GKD (Agarwal et al., 2024) 的设计哲学就是 on-policy / off-policy 的 interpolation: https://arxiv.org/abs/2306.13649

### 5.2 Teacher-Aligned Prompts

**两个粒度的实验**：

**(1) Prompt Template Alignment (Figure 9)**:

Teacher: JustRL-1.5B, Student: R1-Distill-1.5B, 同一份数据 DAPO-Math-17K

| Original DAPO Template | Teacher-Aligned Template |
|---|---|
| "Solve the following math problem step by step. The last line... Answer: $Answer... Remember to put your answer..." | "{Question} Please reason step by step, and put your final answer within \boxed{}." |

仅仅切换 template 就提升三个 benchmark 性能，overlap ratio 一开始就更高。

**(2) Prompt Content Alignment (Figure 10)**:

Teacher: Qwen3-4B-Base-GRPO, Student: Qwen3-1.7B-Base, 对比:
- DAPO-Math-17K（teacher 的 RL 训练集，aligned）
- DeepMath subset（deduplicated against DAPO，仅 in-domain）

Teacher-aligned prompts 给更强 performance，但 overlap ratio **反而更低**——因为 student 把 mass 集中在**更少但更 strongly shared** 的 token 上，cumulative overlap mass 更高。

**Warning**: teacher-aligned prompts 会**大幅降低 student entropy**，可能导致 exploration 不足。建议 mixed with OOD prompts 保持 entropy。

**Intuition**: prompts 影响 student 在训练中 visited states 的分布。如果 prompts 与 teacher post-training 时见过的一致，teacher 在这些 state 上的 distribution 更 sharp 更 well-defined，给 student 的 supervision 更有效。

---

## 6. Discussion: Dense Supervision 的代价

### 6.1 Reward Quality vs. Trajectory Depth

**Setup**: R1-Distill-1.5B student vs JustRL-1.5B teacher, 6 个 max response length: 0.5K, 1K, 3K, 7K, 10K, 15K

**Results (Figure 11a)**:
- 0.5K, 1K: 太短，supervision 不足
- 3K, 7K: sweet spot，最好
- 10K, 15K: 性能 plateau 或下降

**Instability 起源 (Figure 13)**: 15K 设置下，student entropy 的高位首先出现在**响应末尾**，然后随训练 progress **从后向前** propagate。Teacher entropy 也呈现同样的 suffix-to-prefix 趋势 (Appendix D.1, Figure 23)。

**Teacher Continuation 实验 (Figure 11b)**: 
- 从 student-generated rollout 的不同位置截断，让 teacher continue
- Accuracy advantage 从 1K prefix 的 +0.37 单调下降到 16K prefix 的 +0.02

**Intuition**: teacher 训练时见过的 prefix distribution 与 student 长 trajectory 的 prefix distribution 随深度越来越 mismatch。当 student 走到位置 16K 时，prefix 已经 drift 出 teacher 的 familiar region，teacher 给出的 per-token distribution 接近 noise。这就是 OPD 在 long-horizon (extended CoT, agentic multi-turn) 上的根本瓶颈。

### 6.2 Globally Informative ≠ Locally Exploitable

**Setup**: R1-Distill-1.5B student, 对比 success (JustRL-1.5B) 和 fail (R1-Distill-7B) 两个 teacher

**Sequence Mean Reward** (per rollout):

$$\bar{r}(y) = \frac{1}{T} \sum_{t=1}^{T} \left[ \log \pi_T(y_t | x, y_{<t}) - \log \pi_\theta(y_t | x, y_{<t}) \right]$$

**Results (Figure 14)**: 两个 teacher 都给 correct rollouts 更高 reward，AUROC 分别 0.73 (JustRL) 和 0.75 (R1-Distill-7B)。**Failing teacher 的 global signal quality并不差**。

**Hypothesis - Anisotropic Gradient**: 7B teacher 的 per-token advantages 虽然大，但**在 sequence 内不同 position 上的方向不一致**。aggregate 成 gradient 时这些 heterogeneous 信号互相 cancel，effective gradient 小。JustRL-1.5B 因为 thinking pattern 兼容，advantage 集中在 coherent 子集上，gradient 方向一致。

这解释了 §4.1 的观察：fail run 中 per-token advantage 大但 gradient norm 小。

**重要 caveat**: 作者明确说没有直接 verify 这个 anisotropy hypothesis。如果有人能做一个 token-level gradient direction 的 PCA 或 cosine similarity 分析，会是非常好的 follow-up。

### 6.3 Top-k 大小的影响 (Figures 15, 16)

**Setup**: R1-Distill-1.5B student + JustRL-1.5B teacher, 对比 Top-k $\in \{1, 4, 16, 64\}$ 和 Sampled-Token

**Results**:
- Top-1: 不稳定，overlap 增长突跳，entropy 和 gradient norm 有 spikes。**总是选 argmax 会让 reward 集中在 single mode**，小 policy 变化就 flip rank 1，信号不稳定
- Top-4, 16, 64: 表现接近 sampled-token
- Sampled-token: 性能其实很好，因为它按 student 分布随机采样，对 high-prob region 有 unbiased 覆盖

**Takeaway**: Top-k size 不是关键设计点，只要避免 Top-1 这种 biased selection。这呼应了 Thinking Machines Lab 的实践——他们用 sampled-token 也取得不错效果。

---

## 7. 综合 Intuition 构建

把整个 paper 的 insight 浓缩成一句话：**OPD 在 student 自己 visited states 上做 reverse-KL mass transport，效果取决于 student 当前高概率 region 与 teacher 高概率 region 的初始 alignment 是否足以触发 self-reinforcing dynamic。**

几条延伸的直觉：

### 7.1 为什么 reverse KL 比 forward KL 适合 OPD

Forward KL $D_{\text{KL}}(\pi_T \| \pi_\theta)$ 是 mean-seeking，要求 student 覆盖 teacher 所有 mode。在 student 自己 visited state 上算 forward KL 时，teacher 在那些 state 上可能很 confident（接近 one-hot），student 一旦没把 mass 放在 teacher 的 argmax 上，KL 就巨大。这会让 student mass 容易 collapse 到 teacher argmax 上，丧失 entropy 和 exploration。

Reverse KL $D_{\text{KL}}(\pi_\theta \| \pi_T)$ 是 mode-seeking，student 会在自己 mode 附近找 teacher mode，保留自己的 entropy structure。这更适合 long-horizon generation。

### 7.2 Reverse KL 的 mode-seeking 解释

考虑一个简化 case：teacher 分布是 $\pi_T = (0.9, 0.1)$ over $\{A, B\}$，student 当前是 $\pi_\theta = (0.5, 0.5)$。

Reverse KL: $D_{\text{KL}}(\pi_\theta \| \pi_T) = 0.5 \log(0.5/0.9) + 0.5 \log(0.5/0.1) = 0.5 \times (-0.588) + 0.5 \times 1.609 = 0.51$

如果 student 改成 $(0.9, 0.1)$：$D_{\text{KL}} = 0.9 \log(1) + 0.1 \log(1) = 0$

如果 student 改成 $(1, 0)$（collapse 到 A）：$D_{\text{KL}} = 1 \log(1/0.9) + 0 = 0.105$——比 $(0.9, 0.1)$ 大，说明 reverse KL 不鼓励 collapse。

如果 student 改成 $(0, 1)$（collapse 到 B）：$D_{\text{KL}} = \infty$——惩罚严重，reverse KL 不会选 teacher 低概率的 mode。

所以 reverse KL 会把 student 推到 teacher 的高概率 mode（A），但会保留 distribution shape。这正是 OPD 想要的行为。

### 7.3 与 Outcome Reward RL 的关系

OPD 的 dense reward 是 $\log \pi_T(y_t | x, y_{<t}) - \log \pi_\theta(y_t | x, y_{<t})$（per token），可以看作一个 implicit reward。Yang et al. 2026b 把 OPD 形式化为 dense KL constrained RL: https://arxiv.org/abs/2602.12125

而 outcome-reward RL (GRPO, PPO) 只在 trajectory 末尾给稀疏 reward，需要靠 value function 估计 per-token advantage。OPD 把 per-token reward 直接给出来，denser supervision 在 short-horizon 上确实更 sample-efficient。但 paper §6 揭示了**denser supervision 在 long-horizon 上 reliability 衰减**——这构成 OPD 与 RL 各自的 sweet spot。

### 7.4 Self-Reinforcing Dynamic 的临界性

§4.2 揭示的 virtuous cycle 提示 OPD 有 critical initial overlap ratio 的相变点。低于阈值，OPD 信号弱，无法 bootstrap；高于阈值，self-reinforcing 启动，overlap 持续扩大。这非常像 GAN training 的 mode collapse、diffusion model 的 classifier-free guidance、contrastive learning 的 hard negative mining——这些都有类似的 critical initial condition。

这与 Busbridge et al. 2025 的 distillation scaling laws 呼应，他们发现了 U-shape regime: https://arxiv.org/abs/2502.08606

### 7.5 Self-Distillation 的延伸

Self-distillation（同 model 作 teacher 和 student，teacher 拿 privileged information）是 OPD 的特例，thinking-pattern consistency 自动满足。Paper §8 提到 self-distillation 的 knowledge novelty 来自 privileged information 而非不同 model。

相关 references:
- Shenfeld et al., "Self-distillation enables continual learning": https://arxiv.org/abs/2601.19897
- Hübotter et al., "Reinforcement learning via self-distillation": https://arxiv.org/abs/2601.20802
- Zhao et al., "Self-distilled reasoner": https://arxiv.org/abs/2601.18734

### 7.6 Capacity Gap 的 old problem 在 OPD 上的新形态

Cho & Hariharan 2019 在 CV 上证明 teacher 太强会 hurt distillation（capacity gap）: https://arxiv.org/abs/1910.01357
Mirzadeh et al. 2020 提出 teacher assistant 桥接 gap: https://arxiv.org/abs/1902.03393

但 OPD 上的 capacity gap 有新的形态：不是 capacity 大小问题，而是 **visited state distribution alignment** 问题。JustRL-1.5B 和 R1-Distill-7B capacity 差不多甚至 7B 略强，但前者 thinking pattern 与 R1-Distill-1.5B 一致，后者与 R1-Distill-1.5B 同 family 不同 scale。同 family 不同 scale 在 local state 上给出近乎相同 distribution——所以 scale up teacher 在 OPD 上收益甚微。

### 7.7 Long-Horizon Distillation 的 Open Problem

§6 揭示的 trajectory length sweet spot（3K-7K）是个 fundamental limitation。当前 RL 让 model 生成 10K-30K reasoning chain（DeepSeek-R1, Qwen3 都这样），OPD 在这个 length 上 supervision 不可靠。可能的解决方向：

1. **Hierarchical distillation**: 用 outcome reward 给 long-horizon guidance，用 OPD 在 short segments 内做 dense refinement
2. **Curriculum**: 渐进式增加 supervised horizon length
3. **Teacher-as-critic**: 让 teacher 在 student-generated long rollout 上做 segment-level critic 而非 token-level reward
4. **Anisotropic-aware optimizer**: §6.2 提到的 anisotropy hypothesis 如果成立，需要设计能利用 anisotropic 信号的 objective

---

## 8. 实验细节补充

### 8.1 默认训练配置 (Table 2)

- Training temperature: 1.0
- Global batch size: 64
- Rollout number: 4 (每个 prompt 采 4 个 rollout)
- LogProb top-K: 16
- Top-K strategy: Student Top-K
- Max prompt length: 1024
- Max response length: 7168
- Learning rate: $1 \times 10^{-6}$
- KL coefficient: 0.0（注意没有 KL regularization against reference）

### 8.2 Evaluation

- AIME 2024, AIME 2025, AMC 2023
- Sample 16 solutions per problem with temperature 0.7, top-p 0.95
- Max validation response length: 31,744
- Primary metric: avg@16

### 8.3 GRPO Teacher Training (Appendix A.1)

Qwen3-4B-Base-GRPO 的训练细节 (Table 1):
- GRPO on DAPO-Math-17K
- Rollout n=8
- Max response length: 7168
- Learning rate: $1 \times 10^{-6}$
- KL coefficient: 0.0
- Token-mean loss aggregation

### 8.4 JustRL-1.5B

JustRL 是这篇组之前的 work: https://arxiv.org/abs/2512.16649
"Scaling a 1.5B LLM with a simple RL recipe"——非常 simple 的 RL 把 1.5B 模型 push 到很高 math 性能。

---

## 9. 我的延伸思考

### 9.1 与 RLHF / DPO 的关系

OPD 与 RLHF 共享 on-policy rollout 的 paradigm，但 reward 来源不同：
- RLHF: reward model 给 outcome reward
- DPO: implicit reward via preference pairs
- OPD: teacher 的 per-token log-prob 给 dense reward

OPD 的 dense reward 在 short-horizon 上比 RLHF 更 sample-efficient，但 §6 揭示 long-horizon 上的 reward noise 比 outcome reward 严重。

混合 OPD + outcome RL 的方案可能是 sweet spot——比如 CRISPE (Sang et al., 2026): https://arxiv.org/abs/2603.05433

### 9.2 Token-Level Gradient Anisotropy 的更深入分析

§6.2 的 anisotropy hypothesis 没有直接 verify，这是 paper 最大的开放问题之一。可以做的实验：

1. 在 fail run 中，记录每个 token 位置的 (gradient direction, advantage magnitude)
2. 对 batch 内所有 token 的 gradient 做 PCA，看 effective rank
3. 计算 $\mathbb{E}_t[\cos(\nabla \log \pi_\theta(y_t), \nabla \log \pi_\theta(y_{t'}))]$ for $t \neq t'$
4. 对比 success 和 fail run 上的 effective rank / cosine distribution

如果 fail run 上 gradient 高度 anisotropic 但 advantage magnitude 大，就证实了 hypothesis。

### 9.3 OOD Prompts Mixing Ratio

§5.2 提到需要 mix OOD prompts 防止 entropy collapse，但没给具体 ratio。可以做的实验：
- 固定 teacher-aligned prompts，vary OOD prompts 比例 $\{0\%, 25\%, 50\%, 75\%\}$
- 监控 student entropy、overlap ratio、final performance
- 找 sweet spot

### 9.4 Curriculum 策略对 Long-Horizon 的可能方案

Paper §8 提到 long-horizon 的 curriculum，可以具体设计：
- 阶段 1：max response length = 3K，训练 N steps
- 阶段 2：扩展到 7K，从阶段 1 的 checkpoint 继续
- 阶段 3：扩展到 15K，但只在 trajectory 前 7K 上算 OPD loss，7K 之后用 outcome reward
- 这样让 model 先在 short horizon 上建立 reliable pattern，再逐步拓展

### 9.5 Model Family 与 Tokenizer 的影响

§8 future work 提到 cross-family distillation 的 confounding。Tokenizer 不同导致的 token boundary 不一致会破坏 OPD 的 alignment assumption——student token $y_t$ 在 teacher 那里可能对应不同 token boundary。这可能解释为什么同 family distill 通常更稳定。设计 tokenizer-aware OPD 是 open question。

### 9.6 Reward Sparsity 与 Long-Horizon 的本质矛盾

Dense supervision（per-token reward）的 density 与 reliability 之间存在 fundamental tension：
- Short trajectory: dense supervision 可靠，但 trajectory 本身短，能学的 reasoning depth 有限
- Long trajectory: 能学深度 reasoning，但 teacher 在 late tokens 上不可靠

这本质上是 reasoning depth 与 supervision reliability 的矛盾。Outcome reward RL 通过 sparse but reliable reward 反向 credit assignment 解决，但需要 value function approximation。OPD 通过 dense but potentially noisy reward 前向 update，避免 value function 但牺牲 long-horizon reliability。两条路都是局部最优，全图最优可能是 hybrid。

---

## 10. 总结

这篇 paper 的贡献可以总结为四点：

1. **诊断学**：识别出 OPD 失败的两个核心条件（thinking-pattern mismatch + lack of new knowledge），并用 reverse distillation 实验漂亮地验证。

2. **机制学**：揭示 OPD 的核心机制是 progressive alignment on high-probability tokens，且 overlap region 几乎承载全部 probability mass 和 gradient signal。

3. **方法论**：提出 off-policy cold start + teacher-aligned prompts 两个实用 recovery 策略。

4. **局限学**：揭示 OPD 的 trajectory length sweet spot（3K-7K），指出 dense supervision 在 long-horizon 上的 fundamental limitation。

**对你的 intuition 建设最有价值的部分**：
- OPD 是 conditional mass transport over student-visited states，不是 unconditional policy matching
- Benchmark performance ≠ OPD effectiveness——本质是 thinking pattern alignment + new knowledge 两个正交维度
- Self-reinforcing dynamic 是 OPD 成功的相变标志，初始 overlap 是关键 initial condition
- Dense supervision 的 "free lunch" 在 long-horizon 上有 cost

这篇 paper 是非常典型的 "do we understand our training pipeline" 类型工作，与 Anthropic、DeepMind 近年来的 interpretability-of-training-dynamics 风格一脉相承。

---

## References (Web Links)

- Paper code: https://github.com/thunlp/OPD
- Thinking Machines Lab OPD blog: https://thinkingmachines.ai/blog/on-policy-distillation
- MiniLLM (Gu et al., 2023): https://arxiv.org/abs/2306.08543
- GKD (Agarwal et al., 2024): https://arxiv.org/abs/2306.13649
- DeepSeek-R1 (Guo et al., 2025): https://arxiv.org/abs/2501.14448 (Nature paper: https://www.nature.com/articles/s41586-025-08865-3)
- GRPO (Shao et al., 2024, DeepSeekMath): https://arxiv.org/abs/2402.03300
- DAPO (Yu et al., 2025): https://arxiv.org/abs/2503.14476
- JustRL (He et al., 2025a): https://arxiv.org/abs/2512.16649
- Skywork-OR1 (He et al., 2025b): https://arxiv.org/abs/2505.22312
- DeepMath (He et al., 2025c): https://arxiv.org/abs/2504.11456
- Qwen3 (Yang et al., 2025): https://arxiv.org/abs/2505.09388
- GLM-5 (Zeng et al., 2026): https://arxiv.org/abs/2602.15763
- MiMo-v2 (Xiao et al., 2026): https://arxiv.org/abs/2601.02780
- Yang et al., 2026b (Learning beyond teacher): https://arxiv.org/abs/2602.12125
- Hinton et al., 2015 (original KD): https://arxiv.org/abs/1503.02531
- Kim & Rush, 2016 (seq-level KD): https://arxiv.org/abs/1606.07947
- Cho & Hariharan, 2019 (capacity gap): https://arxiv.org/abs/1910.01357
- Mirzadeh et al., 2020 (teacher assistant): https://arxiv.org/abs/1902.03393
- Busbridge et al., 2025 (distillation scaling laws): https://arxiv.org/abs/2502.08606
- Li et al., 2025 (small models from strong reasoners): https://aclanthology.org/2025.findings-acl.1427/
- Shenfeld et al., 2026 (self-distillation continual learning): https://arxiv.org/abs/2601.19897
- Hübotter et al., 2026 (RL via self-distillation): https://arxiv.org/abs/2601.20802
- Zhao et al., 2026b (self-distilled reasoner): https://arxiv.org/abs/2601.18734
- CRISPE (Sang et al., 2026): https://arxiv.org/abs/2603.05433
- OpenThoughts3: https://arxiv.org/abs/2506.04178
- MathArena (Balunović et al., 2025): https://arxiv.org/abs/2505.23281
- LLaMA-Factory: https://arxiv.org/abs/2403.13372
- Sentence-BERT (Reimers & Gurevych, 2019): https://arxiv.org/abs/1908.10084
