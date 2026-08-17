---
source_pdf: iGRPO Self-Feedback–Driven LLM Reasoning.pdf
paper_sha256: 64816e01828791bf222fbb89bb33f08a92cd6ae84673d1915f24f84d867560b9
processed_at: '2026-08-05T09:02:02-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# iGRPO 用人话说

## 一句话概括

**让 model 训练时先自己打草稿, 再看自己的草稿改一遍, 久了就学会了"自我修正"这个能力。**

推理的时候不需要再打草稿了, model 已经把"改稿子"的本领内化了, 一次性就能写出好答案。

## 核心类比

你写代码的时候, 从来不是一次性写完就提交。你先写一版跑跑看, 发现哪里崩了改哪里, 几轮迭代之后才 stable。iGRPO 就是把这个"先写一版再改"的过程 explicit 编进 RL training loop。

或者类比数学家解题: Polya 的《How to Solve It》里反复讲, 先写一个 rough sketch, 再 polish。心理学家 Flower & Hayes 1981 年的写作认知理论也是说, 写作分 planning / drafting / revising 三阶段, single-pass 几乎不存在。

LLM RL 之前基本都 ignore 这件事。GRPO 就是从 prompt 出发 sample 一堆 completions, 用 group 内 normalized reward 算 advantage, 然后更新 policy。Single-shot, 不迭代。iGRPO 说: 别那么着急, 先 sample 一批 draft, 选最好的那个, 把它拼到 prompt 后面, 再 sample 一批 refinement, 在 refinement 上做 update。

## 为什么这个 trick 听起来这么简单却有用

几个 angles:

**Angle 1: 把 inference-time best-of-N 内化到 training**

Best-of-N 大家都熟悉: 推理时 sample N 个答案, 选 reward 最高的返回。AlphaCode 用这个上了 Codeforces。问题: 推理时 N 越大越慢, 用户等不了。

iGRPO 在 training 时就做 best-of-N 选 draft, 然后训 model "如何在 best draft 基础上 refine"。训完之后, model 推理时 single-shot 就能达到接近 best-of-N + refinement 的效果。

这跟 OpenAI o1 / DeepSeek-R1 的思路一脉相承: 把推理时 expensive 的 search / reflection 内化成 training 时学会的 internal capability。

**Angle 2: Implicit curriculum**

训练初期 model 弱, Stage 1 best draft 也很烂, Stage 2 主要在学"copy 一个烂答案并修一些 typo"。训练后期 model 强了, best draft 接近 optimum 但还有 small bug, Stage 2 学的是"找到并 fix small bug"。

不需要人设计 curriculum, curriculum 自动随 policy evolution 上升。这是 bootstrapping 的魔力 - policy 和 conditioning signal 共同进化。

**Angle 3: 信号放大**

论文里 Proposition 3.1 给了个简洁公式: 如果 single-pass 成功率 $p$, best-of-N 成功概率是 $1-(1-p)^N$。

举例: $p=0.1, N=8$, best draft 成功率 = $1-0.9^8 \approx 0.57$。$p$ 从 0.1 涨到 0.3, best draft 成功率从 0.57 涨到 0.94。

这意味着 policy 改进一点点, conditioning 质量指数级改进, 反过来又驱动 policy 改进。这是个 positive feedback loop, 也是 bootstrapping 的理论基础。

## 跟其他"自我改进"方法比

最近一堆 paper 都在搞 LLM 自我改进, 几个代表:

- **Self-Refine** (Madaan 2023): model 自己生成 critique, 自己根据 critique 改。问题是 critique 本身可能错, 而且 model 容易 collapse 成"夸自己"。
- **Reflexion** (Shinn 2023): 类似 Self-Refine, 用 verbal reinforcement, model 自己反思错误。
- **Self-Rewarding LM** (Yuan 2024): model 自己当 judge 给自己打分, 然后 DPO。
- **Self-Verification** (Zhang 2025a): model 自己 verify 答案, 用 verify score reweight samples。
- **Critique-GRPO** (Zhang 2025b): model 生成 critique-conditioned refinement, 把 initial answer 和 refinement 都放进 RL loop。

iGRPO 跟这些方法的本质区别: **不引入任何 auxiliary task。**

上面所有方法都要求 model 学一个新能力 - verify, critique, 或者 judge。这些新能力跟 outcome reward 间接耦合, 容易 reward hack。比如 Self-Verification 学着学着可能学到"对所有答案都打高分"这种 degenerate policy。

iGRPO 直接在 outcome reward 上做 conditioning, 不需要 model 学新 capability。Stage 1 选 best draft 用的还是同一个 rule-based reward, Stage 2 update 也是标准 GRPO。Training loop 跟 vanilla GRPO 一样 clean, 只多了一个"把 best draft 拼到 prompt 后面"的操作。

这点很重要: 简单性是 feature, 不只是 limitation。简单意味着容易复现、容易 scale、容易 stack 在其他方法上。

## Entropy collapse 这点最 interesting

Figure 3 的 entropy 曲线信息量很大。

Per-token policy entropy $\mathcal{H} = -\sum_w p_w \ln p_w$, 单位 nats。初始 2.45 nats (vocab size 决定的 uniform distribution 上界附近)。训练过程中:

- GRPO 在 10% training 时崩到 0.60 nats, 30% 时 0.42, 之后基本平了
- iGRPO 30% 时 0.48, 60% 时 0.46, 100% 时 0.44 - 比 GRPO 慢一拍, 但最终接近

**这告诉我们什么?** GRPO 容易"过早 mode collapse"。Policy 在 10% training 就 commit 到某个 mode, 这个 mode 可能是"开头对但末尾有 bug"的 trajectory, collapse 之后 escape 不出去。

iGRPO 因为 conditioning on best draft, 相当于给 policy 一个"scaffold" - 你已经走到这里了, 接下来 explore 怎么 finish。这让 mid-training exploration 时间更长, policy 有机会 explore 到更好的 mode。

Final entropy 接近这点很关键 - iGRPO 不是让 model 更 random, 而是让 model **探索更久再 commit**。这跟深度学习中"learning rate schedule"的作用类似: 慢慢退火比快速收敛更好。

我有个 hunch: 这跟 R1-zero training 时的 entropy collapse 现象有关系。DeepSeek-R1 报告里提到 zero RL 训练 entropy 会下降, 但 reasoning 长度会自发上升。可能 iGRPO 提供了一种"控制 entropy collapse"的机制, 跟 DAPO 的 dynamic sampling 思路殊途同归 - 都是防止 model 早期 collapse 到 suboptimal mode。

## Generalization 到 non-math 这点最 surprise

Figure 2 显示 OpenReasoning-Nemotron-7B + AceReason-Math 训练后, 不只 math benchmark 涨, GPQA (+1.84) 和 MMLU-Pro (+0.91) 也涨。

GPQA 是 graduate-level science, MMLU-Pro 是 amplified MMLU, 两者都跟 math 关系不大。这暗示 iGRPO 训出来的"refinement 能力"是 general reasoning skill, 不是 math-specific pattern。

直觉: 数学训练让 model 学到"先做一遍再 review 自己"这种 meta-reasoning habit, 这个 habit transfer 到 science reasoning 也 work。这跟 R1 / o1 训练后 generalization 到 coding, science 是同一个现象 - 训练 distribution 是 math, 但 model 学到的是 reasoning process 本身。

## Compute cost 几乎免费

这点很关键。论文 setup 是 matched rollout budget - 总采样 budget 不变, 只是把 GRPO 的 8 个 completion 重新分配成 Stage 1 用 4 个 + Stage 2 用 4 个。

实测 (Table S.3):
- Peak memory: 几乎相同 (54.93 GB vs 54.93 GB)
- Throughput: 慢 17% (0.34 vs 0.41 samples/s)
- Total GPU hours: 多 13% (94.1 vs 83.3)

13% wall-clock 增加, 换来 +3.96 个点 (Nemotron-H-8B) 或者 +1.58 个点 (DeepSeek-R1-Distill-Qwen-7B)。trade-off 极其划算。

## Wrapper 性质这点容易 miss

Table 2 显示 iGRPO 套在 DAPO 上 +1.19, 套在 GSPO 上 +1.11。这说明 iGRPO 不是"GRPO 的改进版", 而是一个 **reusable refinement wrapper**。

类比: residual connection 不是 belong to ResNet, 它是个 architectural primitive, 可以套在任何 network。iGRPO 类似 - 任何 group-based PPO variant 都能加这个 two-stage refinement layer。

这暗示 design space 还很大。可以想象:
- 三阶段 iGRPO: draft → refine → final polish
- 多 draft iGRPO: Stage 1 选 top-3 drafts, Stage 2 分别 condition
- Adversarial iGRPO: Stage 1 选 worst draft, 训 model 学"如何 fix 烂答案"
- Tree-search iGRPO: Stage 1 不 sample 独立 drafts, 而是 MCTS explore
- iGRPO + PRM: 把 binary outcome reward 换成 process reward, 每个 reasoning step 都有 signal

## 推理时不需要两阶段这点最 magic

训练时 two-stage, 推理时 single-shot。这个不对称很重要。

类比: 学自行车时用 training wheels, 学会之后去掉 wheels 直接骑。iGRPO 训练时用 self-feedback scaffold, 训完之后 scaffold 不需要了, model 内化了 refinement 能力。

这跟 inference-time 的 self-refine / Reflexion 完全不同。那些方法推理时还是要 explicit 多轮, user 等得起吗? iGRPO 训完是 zero-overhead 推理。

## 我的几个 hunch

**Hunch 1: Stage 1 / Stage 2 budget 分配是大 design space**

论文固定 $N+G=8$, 切成 $N=4, G=4$。但 $N=1, G=7$ 或 $N=7, G=1$ 会怎样?

$N=1$: Stage 1 几乎没用, 退化成 standard GRPO 但 prompt 里塞一个随机 draft。可能接近 GRPO 性能。

$N=7, G=1$: Stage 2 只有一个 sample, advantage 无法 group normalize (std=0 → advantage=0), 没 gradient signal。完全训不动。

中间应该有 sweet spot。直觉上 $N$ 应该比 $G$ 略大 - 探索多一点, refine budget 少一点反而够用, 因为 conditioning 已经提升 baseline 成功率。但需要实验验证。

**Hunch 2: best draft 可能不是最好选择**

Stage 1 选 argmax reward draft。但 argmax 容易 overfit verifier 的 quirks。如果 verifier 在某个 draft 上误判高分, Stage 2 conditioned on 它反而学坏。

Symmetric alternative: 选 median-reward draft。或者 reward-weighted average (但 LLM 不是 vector, 怎么 average? 可能用, 但 context 会很长)。

更激进的: 用 worst draft 作为 "what not to do" 的 negative prompt。这跟 SPAG (self-playing adversarial) 类似。

**Hunch 3: 这跟 AlphaProof 思路接近**

DeepMind 的 AlphaProof 在 Lean 上做 RL, 训 model 生成 proof 然后 verify, 失败的 proof 作为 feedback 修正。iGRPO 在 natural language 上做类似事, 只是 reward 是 answer correctness 不是 proof validity。

可以想象 iGRPO + formal verifier 的 hybrid - Stage 1 用 LLM 生成自然语言 draft, Stage 2 用 Lean / Coq verify refinement。这可能是数学推理的下一个大 step。

**Hunch 4: 这跟 Constitutional AI 有点像**

Anthropic 的 Constitutional AI 让 model 自己给反馈 (constitution-based critique), 然后用反馈做 RLHF。iGRPO 用 model 自己的 best draft 作为 implicit constitution, 然后训 model refine。

差别: Constitutional AI 的 constitution 是人工写的规则, iGRPO 的 "constitution" 是 model 自己生成的 best-performing example。后者更动态, 跟 policy 共同进化。

**Hunch 5: 可能跟 process reward model 互补**

PRM (Lightman et al. 2023, PRM800K) 给每个 reasoning step打分, 比 outcome-only reward 信息丰富。但 PRM 容易 reward hacking - model 学着写"看着对但实际错"的步骤骗 PRM。

iGRPO 用 outcome reward, 不容易被 step-level reward hack。可以想象一个 hybrid: Stage 1 用 PRM 选 best draft (更细粒度), Stage 2 用 outcome reward 做 final update。这样既享受 PRM 的细粒度, 又避免 PRM-only 的 reward hacking。

**Hunch 6: Entropy collapse 跟 length 增长的关系**

DeepSeek-R1 zero training 报告 reasoning length 自发增长。但 entropy 下降。这两个看似矛盾 - length 增长应该让 model explore 更多, entropy 应该高。

我的理解: length 增长是 model 学到"写更多 reasoning step", 但每一步的 entropy 可能下降。Model 学的是"按固定模板写长 reasoning", 不是"explore 不同 reasoning"。

iGRPO 延迟 entropy collapse 可能意味着 model 学到更 diverse 的 reasoning 模式, 不只是更长的模板。这个 hypothesis 需要 ablation reasoning trace diversity 验证。

## 跟 DeepSeek-R1 zero 的关系

R1 zero 训练显示 base model 直接 RL 就能涌现 reasoning capability, 不需要 SFT cold start。这是 R1 的大 contribution。

iGRPO 跟 R1 zero 正交: R1 zero 是"不需要 SFT", iGRPO 是"需要 self-feedback"。可以叠加: 在 base model 上直接用 iGRPO 训, 看会不会比 R1 zero 更早涌现 reasoning。

实际上 paper 里很多实验就是 base + iGRPO (Nemotron-H-8B-Base-8K + iGRPO), 取得 +15.39 个点 macro avg 提升 (29.65 → 45.04), 而 GRPO 只提升 +11.43 个点。iGRPO 在 zero-RL 场景下比 GRPO 更有效, 这暗示 self-feedback 对 reasoning emergence 有帮助。

## 跟 inference scaling laws 的关系

Snell et al. 2024 的 inference scaling paper [https://arxiv.org/abs/2408.03314] 分析了三种 inference-time scaling axis:
1. Best-of-N (parallel sampling)
2. Sequential refinement
3. Search (tree with value function)

iGRPO 把 axis 2 (sequential refinement) 内化到 training, axis 1 (best-of-N) 部分内化 (Stage 1 selection)。

剩下 axis 3 (search with value function) 还没被 training 内化。可以想象一个 "tree-search GRPO": Stage 1 做 MCTS-style tree expansion, Stage 2 在 leaf node 上做 refinement update。这是把 AlphaZero 思路引入 LLM RL, 跟 Sutton & Barto 经典 RL 框架更接近。

## 为什么我觉得这 paper 重要

不是因为 SOTA - SOTA 总会被刷掉。是因为它揭示一个 architectural insight: **LLM RL 的瓶颈不在 optimizer, 而在 training loop 的结构。**

最近一堆 paper 在改 GRPO 的 objective: Dr. GRPO 修 bias, DAPO 加 dynamic sampling, GSPO 改 sequence-level clipping。这些都是 optimizer-level 改进, 增益 marginal。

iGRPO 不改 optimizer, 改 training loop 结构 - 加一个 self-feedback stage。gain 比所有 optimizer 改进都大, 而且跟它们 orthogonal, 可以叠加。

这暗示 LLM RL 还有大量 architectural innovation 没被探索。我们能想象的:
- Multi-turn refinement training
- Tree-search training
- Adversarial self-play training
- Curriculum co-evolution with model

这些都跟 iGRPO 一样, 不改 optimizer, 改 loop 结构。可能下一个 reasoning breakthrough 就来自这种 structural 改动, 不是来自新 optimizer。

## 一句话总结

让 LLM 训练时学会改自己的草稿, 推理时就能一次写对。简单, 有效, 可叠加到任何现有 RL 方法上。

---

# iGRPO 论文深度解析

## 1. 高层直觉

这篇paper 由 NVIDIA 的 Ali Hatamizadeh 等人提出，核心想法出奇地简洁: 把 GRPO 的 single-shot optimization 改成 **two-stage loop**, 让 policy 在每个 optimization step 里先"打草稿", 再"改稿子"。

这其实在模仿人类解题的行为模式: Flower & Hayes (1981) 的写作认知理论, Polya (2014) 的《How to Solve It》, 都强调 iterative refinement 的重要性。LLM RL 之前基本没有把这件事 explicit 编码到 training loop 里, iGRPO 就是来填这个 gap 的。

关键 insight 是 **dynamic self-conditioning**: 模型自己生成的 best draft 作为下一次 generation 的 context, 这个 context 会随 policy 共同 evolution, 形成一个 **bootstrapped learning dynamic**。这与 in-context learning (ICL) 的 static demonstration 不同, 与 critique-style 方法也不同 - 它把"自我改进能力"作为一个 generalizable capability 直接训进 policy 里, 推理时纯 single-shot 调用。

参考相关工作的脉络:
- **STaR** (Zelikman et al., 2022) [https://arxiv.org/abs/2203.14465]: bootstrapping with verified rationale
- **Self-Refine** (Madaan et al., 2023) [https://arxiv.org/abs/2303.17651]: LLM 自己给自己 critique
- **Reflexion** (Shinn et al., 2023) [https://arxiv.org/abs/2303.11366]: verbal RL with self-reflection
- **Self-Rewarding LM** (Yuan et al., 2024) [https://arxiv.org/abs/2401.10020]: model as its own judge
- **Self-Consistency** (Wang et al., 2022) [https://arxiv.org/abs/2203.11171]: majority vote over samples

## 2. GRPO 背景回顾

理解 iGRPO 必须先吃透 GRPO。GRPO (Group Relative Policy Optimization) 是 DeepSeekMath (Shao et al., 2024) [https://arxiv.org/abs/2402.03300] 提出的算法, 是 PPO (Schulman et al., 2017) [https://arxiv.org/abs/1707.06347] 的 value-function-free 变体。

**为什么去掉 critic?** PPO 需要训练一个 value function $V_\psi(s)$ 来估计 advantage $A(s,a) = Q(s,a) - V(s)$, 这个 critic 对 7B/14B 这种规模来说本身又是 7B/14B, 显存翻倍、训练不稳定。GRPO 用了一个 trick: 对同一个 prompt $q$ sample $G$ 个 completions, 用 group 内的 reward 均值和方差来 normalized 出 advantage。

**GRPO 的采样:**

$$o_i \sim \pi_{\theta_{\text{old}}}(\cdot \mid q), \quad i = 1, \ldots, G$$

其中 $\pi_{\theta_{\text{old}}}$ 是当前 iteration 的 policy snapshot (PPO 标准做法, 重要性采样用), $G$ 是 group size (论文里通常 8 或 16)。

**Advantage 计算:**

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_1, \ldots, R_G\})}{\text{std}(\{R_1, \ldots, R_G\})}$$

变量含义:
- $R_i = R_\phi(o_i)$: reward function 对第 $i$ 个 completion 的打分, 论文里用 rule-based binary reward $R_\phi(o) = \mathbb{1}[\text{extract}(o) = a]$
- $\text{mean}(\cdot)$: group 内 reward 的样本均值
- $\text{std}(\cdot)$: group 内 reward 的样本标准差
- 下标 $i$: 第 $i$ 个 completion, 下标 $t$: 第 $i$ 个 completion 的第 $t$ 个 token
- 注意 $\hat{A}_{i,t} = \hat{A}_i$ (同一 completion 内所有 token 共享同一 advantage), 这反映 outcome-only reward 的性质 - 整条 trajectory 只有一个 scalar 信号

**为什么用 group normalization?** 这相当于把 baseline $b(q')$ 从 learned value function 换成 group mean, variance 从 $\text{std}$ 做 scaling 让 gradient 的 magnitude 比较稳定。这种 implicit baseline 是无偏的。

**Clipped surrogate objective:**

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta\hat{D}_{\text{KL}}^{(i,t)}\right]\right]$$

变量含义:
- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}$: per-token importance ratio, 衡量新 policy 相对 old policy 在该 token 上的概率比
- $\epsilon$: PPO clip 参数, 限制单次 update 不让 ratio 偏离 $[1-\epsilon, 1+\epsilon]$ 太远 (典型 0.2)
- $\beta$: KL divergence 正则系数, 论文里实验发现设为 0 即可 (Table S.4)
- $\hat{D}_{\text{KL}}^{(i,t)}$: per-token KL 估计器 (Schulman 2020 [http://joschu.net/blog/kl-approx.html])
- $|o_i|$: completion $o_i$ 的 token 长度, 外面的 $1/|o_i|$ 是 token averaging

**Schulman 的 KL 估计器:**

$$\hat{D}_{\text{KL}}^{(i,t)} = \frac{\pi_{\text{ref}}(o_{i,t}\mid \cdot)}{\pi_\theta(o_{i,t}\mid \cdot)} - \log\frac{\pi_{\text{ref}}(o_{i,t}\mid \cdot)}{\pi_\theta(o_{i,t}\mid \cdot)} - 1$$

这个形式是 $x - \log x - 1$ with $x = \pi_{\text{ref}}/\pi_\theta$, 在 $x=1$ 时为 0, 全局非负 (因为 $x - \log x \geq 1$ by AM-GM)。它对 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 是有偏估计, 但对 $D_{\text{KL}}(\pi_{\text{ref}} \| \pi_\theta)$ 的反向是无偏的 (Schulman 选这个方向是为了在 samples from $\pi_\theta$ 时无偏)。iGRPO 在 augmented prompt $q'$ 下计算这个 KL, 跟 GRPO 一模一样, 只是把 $q$ 换成了 $q'$。

## 3. iGRPO 核心架构

### 3.1 两阶段 pipeline

**Stage 1: Exploratory Draft Generation**

对每个 prompt $q$ 从 current policy snapshot $\pi_{\theta_{\text{old}}}$ 采 $N$ 个 drafts:

$$d_i \sim \pi_{\theta_{\text{old}}}(\cdot \mid q), \quad i = 1, \ldots, N$$

用 reward function $R_\phi$ 给每个 draft 打分, 选 reward 最高的作为 best draft:

$$\hat{d} = \arg\max_{i \in \{1,\ldots,N\}} R_\phi(d_i)$$

**Stage 2: Conditioned Refinement**

把 best draft 直接拼接到原 prompt 后面形成 augmented prompt:

$$q' = \text{Concat}(q, \hat{d})$$

然后在 $q'$ 上采 $G$ 个 refinements:

$$o_j \sim \pi_{\theta_{\text{old}}}(\cdot \mid q'), \quad j = 1, \ldots, G$$

后续 advantage 计算、clipped surrogate、KL penalty 都跟 GRPO 一模一样, 唯一的区别是 prompt 从 $q$ 变成了 $q'$。

**关键点: 只有 Stage 2 的 tokens 拿到 gradient update。** Stage 1 用来生成 context, 但 Stage 1 的 draft $\hat{d}$ 在当次 iteration 内部不参与反向传播, 既不 diff argmax, 也不 diff Stage 1 的 sampling。但是 Stage 1 影响 $q'$ 的分布, 跨 iteration 时分布会 shift - 这就是 dynamic self-conditioning。

### 3.2 完整 objective

$$\mathcal{J}_{\text{iGRPO}}(\theta) = \mathbb{E}\Big[q \sim P(Q), \underbrace{\{d_i\}_{i=1}^N \sim \pi_{\theta_{\text{old}}}(\cdot\mid q), \hat{d} = \arg\max_i R_\phi(d_i), q' = \text{Concat}(q, \hat{d})}_{\text{Stage 1}}, \underbrace{\{o_j\}_{j=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot\mid q')}_{\text{Stage 2}}\Big]$$
$$\times \frac{1}{G}\sum_{j=1}^G \frac{1}{|o_j|}\sum_{t=1}^{|o_j|}\left[\min\left(r_{j,t}(\theta)\hat{A}_j, \text{clip}(r_{j,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_j\right) - \beta\hat{D}_{\text{KL}}^{(j,t)}\right]$$

跟 GRPO objective (Eq.1) 的唯一结构差别: 所有 Stage 2 completions 都 conditioned on augmented prompt $q'$。Importance ratio 也基于 $q'$:

$$r_{j,t}(\theta) = \frac{\pi_\theta(o_{j,t}\mid q', o_{j,<t})}{\pi_{\theta_{\text{old}}}(o_{j,t}\mid q', o_{j,<t})}$$

KL penalty 也基于 $q'$:

$$\hat{D}_{\text{KL}}^{(j,t)} = \frac{\pi_{\text{ref}}(o_{j,t}\mid q', o_{j,<t})}{\pi_\theta(o_{j,t}\mid q', o_{j,<t})} - \log\frac{\pi_{\text{ref}}(o_{j,t}\mid q', o_{j,<t})}{\pi_\theta(o_{j,t}\mid q', o_{j,<t})} - 1$$

### 3.3 与 ICL 的对比

Static ICL 的 objective 是:

$$\mathcal{J}_{\text{ICL}}(\theta) = \mathbb{E}_{q\sim P(Q)}\left[\mathbb{E}_{o\sim\pi_\theta(\cdot\mid q, e)}[R_\phi(o)]\right]$$

其中 $e$ 是 **fixed demonstration** (跟 $\theta$ 无关)。问题: $e$ 不会随 policy 改进而进化, 容易 stuck 在某个 distribution 上。

iGRPO 改成 dynamic self-conditioning:

$$\mathcal{J}_{\text{iGRPO}}(\theta) = \mathbb{E}_{q\sim P(Q)}\left[\mathbb{E}_{o\sim\pi_\theta(\cdot\mid q'_\theta(q))}[R_\phi(o)]\right], \quad q'_\theta(q) = \text{Concat}(q, \hat{d}_\theta(q))$$

这里 $\hat{d}_\theta(q)$ 由 current policy $\pi_\theta$ 生成, 随 $\theta$ 变化而变化。这就是 **bootstrapped learning dynamic**: policy 变好 → draft 变好 → conditioning 变好 → policy 进一步变好。

### 3.4 推理时: 普通单 shot

重要细节: **训练时用 two-stage, 推理时 single-shot。** 推理时不需要再 generate draft, 不需要 self-feedback loop, 直接从原 prompt $q$ 出发 sample。这意味着 model 训练阶段学到了一种 **内化的 refinement 能力**, 而不是依赖 explicit prompt engineering。

这跟 STaR-style 方法一致: 训练时用 samples 当新的 training data, 推理时直接 sample 即可。也跟 OpenAI o1 的 internal reasoning 类似。

## 4. 理论分析: Bootstrapping Effect

**Proposition 3.1 (Progressive Conditioning Quality for Binary Rewards)**

设 reward 为 binary: $R_\phi(o) \in \{0, 1\}$。Stage 1 drafts $\{d_i\}_{i=1}^N$ 从 $\pi_\theta(\cdot\mid q)$ i.i.d. 采样。设 $V_\theta(q) = \mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}[R_\phi(o)]$ 为 expected reward (binary case 下就是 success probability $p_\theta(q) = \Pr[R_\phi(o) = 1]$)。

那么 best draft $\hat{d}_\theta(q) = \arg\max_i R_\phi(d_i)$ 的 expected reward 是:

$$\mathbb{E}[R_\phi(\hat{d}_\theta(q))] = 1 - (1 - V_\theta(q))^N$$

**Proof:** 因为 $R_\phi(\hat{d}_\theta(q)) = 1$ iff 至少一个 draft 拿到 reward 1。i.i.d. 下:

$$\Pr[\forall i, R_\phi(d_i) = 0] = (1 - V_\theta(q))^N$$

所以

$$\Pr[R_\phi(\hat{d}_\theta(q)) = 1] = 1 - (1 - V_\theta(q))^N$$

binary 情况下 expectation = probability, 证毕。函数 $f(x) = 1 - (1-x)^N$ 在 $x \in [0,1]$ 单调递增, 所以 $V_\theta$ 增大 → $\mathbb{E}[R_\phi(\hat{d}_\theta)]$ 增大。

**Intuition building:** 这个公式对应统计学里的 **order statistic**。Best-of-N 的期望质量 = $1 - (1-p)^N$ (binary case)。它正是 Best-of-N sampling 的成功概率公式, 也是 AlphaCode [https://arxiv.org/abs/2203.07880] cluster-and-vote 里的核心 quantity, 也是 Self-Consistency 在 majority vote 下的理论 success rate。

例子: 如果 $V_\theta = 0.1$ (单次成功率 10%), $N = 8$, best draft 的成功率 = $1 - 0.9^8 \approx 0.57$。如果 $V_\theta$ 训到 $0.3$, best draft 的成功率 = $1 - 0.7^8 \approx 0.94$。

这个 proposition 揭示 bootstrapping 的 **monotonic amplification**: policy 改进一点点, draft 质量指数级改进, 进而又驱动 policy 改进。但要注意: 这只是 conditioning 质量, 不保证 policy 真的能学到 refinement function。能不能 refine 仍取决于 policy capacity, training data distribution, reward shaping 等。

**这个分析的实际含义:** 训练初期 $\hat{d}$ 可能很弱, Stage 2 主要在学 "copy the draft"。训练后期 $\hat{d}$ 接近 optimum, Stage 2 学的是 "fix the small errors"。这是 **implicit curriculum**: 难度自动随 policy evolution 上升。

## 5. 实验结果解读

### 5.1 Controlled comparison (Table 1)

最重要的对比是 **matched rollout budget**: 对每个 prompt 总共采样 8 个 completions。GRPO 用全部 8 个做 update, iGRPO 用 4 个做 Stage 1 + 4 个做 Stage 2。这是 fair comparison 的关键 - 不是花更多 compute, 而是 redistribute compute。

Table 1 关键数字 (Pass@1, macro avg):

| Model | GRPO | Self-Verification | Critique-GRPO | iGRPO | Δ over GRPO |
|---|---|---|---|---|---|
| Nemotron-H-8B-Base | 41.08 | 42.86 | 43.39 | **45.04** | +3.96 |
| DeepSeek-R1-Distill-Qwen-7B | 68.29 | 69.08 | 69.14 | **69.87** | +1.58 |
| OpenMath-Nemotron-7B | 75.02 | - | - | **76.07** | +1.05 |
| DeepSeek-R1-Distill-Qwen-14B | 71.29 | - | - | **73.02** | +1.73 |
| OpenMath-Nemotron-14B | 76.73 | - | - | **78.00** | +1.27 |

**关键观察:**
1. **Base 越弱, gain 越大**: Nemotron-H-8B 是 generalist base, gain 最大 (+3.96); OpenMath-Nemotron-7B 已经 math-specialized, gain 最小 (+1.05)。
2. **Hard benchmarks gain 更大**: AIME25/AIME24 这种 long-horizon competition math 上 gain 最大。Nemotron-H-8B 上 AIME25 从 7.78 → 9.17, AMC 从 45.10 → 48.75。这符合 iGRPO 的机制设计 - 长链推理容易在末尾出小错, Stage 2 学的就是 fix these residual errors。
3. **优于 critique-style baselines**: Self-Verification 和 Critique-GRPO 都需要 model 学 verify / critique 的 auxiliary task, 跟 outcome reward 间接耦合。iGRPO 直接在 outcome reward 上优化 refinement, training loop 更 tight。

### 5.2 跟 Self-Verification / Critique-GRPO 的对比

Self-Verification (Zhang et al., 2025a) [https://arxiv.org/abs/2506.01369]: model 自己 verify 自己的答案, 用 verify score reweight samples。
Critique-GRPO (Zhang et al., 2025b) [https://arxiv.org/abs/2506.03106]: model 自己 critique 自己, 再 generate critique-conditioned refinement, 把 initial answer 和 refinement 都放进 RL loop。

这两个方法都要求 model 学一个 **auxiliary capability** (verify or critique), 而 verify/critique 本身跟 outcome reward 间接耦合, 容易学到 reward hacking。iGRPO 不需要 model 学新能力, 直接在 outcome reward 上 refine, training loop 跟 GRPO 一样 clean, 只多了一个 conditioning signal。

### 5.3 Scaling: OpenReasoning-Nemotron-7B + AceReason-Math

这是 paper 的 headline result。在更强的 base (OpenReasoning-Nemotron-7B, NVIDIA 2025 [https://huggingface.co/nvidia/OpenReasoning-Nemotron-7B]) 和更难的 data (AceReason-Math, Chen et al., 2025b [https://arxiv.org/abs/2505.16400]) 上, iGRPO 取得:
- **AIME24: 85.62%**
- **AIME25: 79.64%**

这两个数字是 SOTA (as of writing)。注意 generalization 到 non-math benchmarks:
- GPQA (+1.84)
- MMLU-Pro (+0.91)

这说明 iGRPO 学到的 refinement capability 不只是 math-specific, 而是 generalizable 的 reasoning skill。这跟 R1 / o1 类 RL 训练得到的"reasoning habit"一致 - 训练 distribution 是 math, 但 generalization 到 science/coding 类 reasoning。

## 6. Ablation Studies

### 6.1 iGRPO wrapper 可以套在其他 optimizer 上 (Table 2)

把 iGRPO 的两阶段套在 DAPO (Yu et al., 2025) [https://arxiv.org/abs/2503.14476] 和 GSPO (Zheng et al., 2025) [https://arxiv.org/abs/2507.18071] 上:

| Method | Base Avg | + iGRPO Avg | Δ |
|---|---|---|---|
| DAPO | 69.74 | 70.93 | +1.19 |
| GSPO | 69.20 | 70.31 | +1.11 |

两个 case 都 +1.1 到 +1.2 提升。这证明 iGRPO 的 gain 主要来自 **refinement interface**, 跟具体 base optimizer (GRPO vs DAPO vs GSPO) 正交。可以认为 iGRPO 是一个 **reusable refinement wrapper**, 适用于所有 group-based PPO variants。

### 6.2 Generative judge (Table 3)

把 rule-based binary reward 换成 GPT-5 judge (score 在 [0,1] 连续), 让 judge 给每个 completion 评 partial credit:

| Benchmark | Rule-based | GPT-5 Judge | Δ |
|---|---|---|---|
| AIME25 | 40.16 | 41.12 | +0.96 |
| AIME24 | 56.30 | 57.45 | +1.15 |
| MATH500 | 93.80 | 94.20 | +0.40 |
| AMC | 95.00 | 96.25 | +1.25 |
| GSM8K | 92.42 | 92.95 | +0.53 |
| Minerva | 41.54 | 42.88 | +1.34 |
| Average | 69.87 | 70.81 | +0.94 |

AIME/Minerva 这种 near-miss 多的 benchmark 上 partial credit 帮助最大。这说明 iGRPO 实际上是一个 **general scalar-reward-compatible refinement framework**, 不依赖 binary verifier。这对 process reward model (PRM) 集成有启发, 见 Lightman et al. (2023) [https://arxiv.org/abs/2305.20050] 的 PRM800K 工作。

### 6.3 Entropy analysis (Figure 3)

最 interesting 的 ablation。Per-token policy entropy:

$$\mathcal{H}(\pi_\theta(\cdot\mid h_t)) = -\sum_{w\in\mathcal{V}}\pi_\theta(w\mid h_t)\ln\pi_\theta(w\mid h_t)$$

变量含义:
- $h_t$: 解码第 $t$ 步的 context (包含 prompt + 之前 generated tokens)
- $\mathcal{V}$: vocabulary
- 用 log-softmax logits (base $e$) 计算, 单位 nats

训练曲线:
- 起点: 两者都 2.45 nats
- 10% training: GRPO 0.60, iGRPO 较高
- 30%: GRPO 0.42, iGRPO 0.48
- 60%: GRPO ~0.42, iGRPO 0.46
- 100%: 两者收敛到接近 (~0.42-0.44)

**关键 insight: iGRPO 延迟了 entropy collapse (mode collapse)。** Mid-training entropy 保持较高意味着 policy 仍在 explore, 没有过早 commit 到某个 single mode。这跟 Stage 2 的 conditioning 有关: 把 best draft 放在 context 里, 给 policy 提供了一个 strong scaffold, 但 scaffold 不是 ground truth, policy 仍需要 explore alternative continuations 才能 refine。GRPO 单 shot 时, policy 容易快速 mode collapse 到某个"看着对但有小错"的 mode, 然后困在那里。

这一点让我想到 DeepSeek-R1 zero training 时的 entropy 现象, 见 SimplerL-Zoo (Zeng et al., 2025) [https://arxiv.org/abs/2503.18892] 和 Understanding R1-zero-like training (Liu et al., 2025) [https://arxiv.org/abs/2503.20783] 关于 entropy collapse 的分析。Dr. GRPO (Liu et al., 2025) 也讨论过 length bias 导致的 entropy 问题。

**Final entropy 接近** 这点很重要: iGRPO 不是让 final model 更 random, 而是 mid-training 探索更久, 最后 converge 到差不多 sharp 的 distribution。Gain 来自 better exploration schedule, 不是来自 final randomness。

### 6.4 KL coefficient $\beta$ (Table S.4)

| $\beta$ | Score (%) |
|---|---|
| 0 | 69.87 |
| 0.0001 | 70.23 |
| 0.001 | 69.31 |
| 0.01 | 69.91 |

差距很小, $\beta=0.0001$ 略优, $\beta=0$ 简单 pipeline 就够用。这跟 DeepSeek-R1 报告的发现一致: zero RL 在 math reasoning 上 KL term 不重要, 因为 base model 已经 pretrained 很好, 没必要 anchor 到 reference。这与 RLHF 场景 (alignment) 不同, RLHF 里 KL 重要因为要避免 model 偏离 human intent 太远。

### 6.5 Number of completions (Table S.4)

| Total completions | Score (%) |
|---|---|
| 4 | 67.79 |
| 8 | 69.87 |
| 16 | 70.17 |
| 32 | 70.33 |

4 → 8 跳升, 之后 marginal。说明 8 是 sweet spot, 跟 GRPO 常用 8 / 16 一致。这一点跟 Best-of-N 的 scaling 一致, 见 Snell et al. (2024) 关于 inference-time scaling 的 paper [https://arxiv.org/abs/2408.03314]。

## 7. Computational Analysis

paper 强调 iGRPO **不增加主要 generation cost**:

- $C_{\text{gen}}$: 单个 completion 的 generation cost (prompt encoding + autoregressive decoding + reward eval)
- $G_{\text{GRPO}}$: GRPO 每个 prompt 的 completions 数量

Baseline GRPO cost per prompt:

$$C_{\text{GRPO}} \approx G_{\text{GRPO}} \cdot C_{\text{gen}}$$

iGRPO cost per prompt (Stage 1: $N$ drafts + Stage 2: $G$ refinements):

$$C_{\text{iGRPO}} \approx (N + G) \cdot C_{\text{gen}}, \quad \frac{C_{\text{iGRPO}}}{C_{\text{GRPO}}} = \frac{N+G}{G_{\text{GRPO}}}$$

实验中保持 $N + G = G_{\text{GRPO}}$, 比如 GRPO 用 16 个 completions, iGRPO 用 $N=8, G=8$, 总采样 budget 相同。

**实测 (Table S.3):**
- Peak memory: GRPO 54.93GB, iGRPO 54.93GB (基本无差异)
- Throughput: GRPO 0.41 samples/s, iGRPO 0.34 samples/s (慢 17%)
- Total GPU hours: GRPO 83.3, iGRPO 94.1 (慢 13%)

Throughput 慢是因为 Stage 1 / Stage 2 串行 decode。Memory 不变是因为 vLLM 那个节点做 generation, training 节点 batch size 不变。13% wall-clock 增加换 +3.96 个点 (Nemotron-H-8B), trade-off 非常划算。

## 8. Policy Gradient 推导

Appendix A 给了完整的 derivation。这里 walk through。

### 8.1 Two-stage 采样的 distribution

对每个 $q$:
1. Stage 1: $d_i \sim \pi_{\theta_{\text{old}}}(\cdot\mid q)$, $\hat{d} = \arg\max_i R_\phi(d_i)$, $q' = \text{Concat}(q, \hat{d})$
2. Stage 2: $o_j \sim \pi_{\theta_{\text{old}}}(\cdot\mid q')$, $j = 1, \ldots, G$

这定义了一个 implicit distribution over $q'$ induced by $\pi_{\theta_{\text{old}}}$ 和 argmax selection。**Single iteration 内不 diff Stage 1 和 argmax**, $q'$ 当作 sampled context 处理。**跨 iteration $q'$ 分布 shift**, 因为 $\pi_{\theta_{\text{old}}}$ 变了。

### 8.2 From expected reward to REINFORCE gradient

Fixed $q'$, conceptual objective:

$$\mathcal{J}(\theta \mid q') = \mathbb{E}_{o\sim\pi_\theta(\cdot\mid q')}[R_\phi(o)]$$

Score-function identity (REINFORCE, Williams 1992 [https://people.cs.umass.edu/~wallin/Courses/Summer2/Lectures/rl/reinforce.pdf]):

$$\nabla_\theta \mathcal{J}(\theta\mid q') = \mathbb{E}_{o\sim\pi_\theta(\cdot\mid q')}\left[R_\phi(o)\nabla_\theta\log\pi_\theta(o\mid q')\right] = \mathbb{E}\left[R_\phi(o)\sum_{t=1}^{|o|}\nabla_\theta\log\pi_\theta(o_t\mid q', o_{<t})\right]$$

第二个等式用 factorization $\pi_\theta(o\mid q') = \prod_t \pi_\theta(o_t\mid q', o_{<t})$, log 后变 sum。

### 8.3 加 baseline (advantage)

任意 baseline $b(q')$ (不依赖 sampled tokens) 都不影响 unbiasedness:

$$\nabla_\theta\mathcal{J}(\theta\mid q') = \mathbb{E}\left[(R_\phi(o) - b(q'))\sum_t \nabla_\theta\log\pi_\theta(o_t\mid q', o_{<t})\right]$$

GRPO/iGRPO 把 $b(q')$ 替换成 group mean $\bar{R}$, 顺手做 variance reduction, 归一化成 $\hat{A}_j$:

$$\hat{A}_j = \frac{R_j - \bar{R}}{s_R}, \quad R_j = R_\phi(o_j), \quad \bar{R} = \text{mean}(\{R_1,\ldots,R_G\}), \quad s_R = \text{std}(\{R_1,\ldots,R_G\})$$

如果 $s_R = 0$, set $\hat{A}_j = 0$ (group 内所有 completions reward 相同时没 gradient 信号)。

On-policy gradient estimator:

$$g_{\text{on-policy}}(\theta\mid q') = \frac{1}{G}\sum_{j=1}^G\frac{\hat{A}_j}{|o_j|}\sum_{t=1}^{|o_j|}\nabla_\theta\log\pi_\theta(o_{j,t}\mid q', o_{j,<t})$$

$1/|o_j|$ 是 token averaging 让不同长度 completion 贡献 comparable。

### 8.4 Off-policy: importance ratio + clipping

实际从 $\pi_{\theta_{\text{old}}}$ sample, 用 importance ratio 校正:

$$r_{j,t}(\theta) = \frac{\pi_\theta(o_{j,t}\mid q', o_{j,<t})}{\pi_{\theta_{\text{old}}}(o_{j,t}\mid q', o_{j,<t})}$$

PPO-clipped surrogate:

$$\mathcal{L}_{j,t}^{\text{clip}}(\theta) = \min\left(r_{j,t}(\theta)\hat{A}_j, \text{clip}(r_{j,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_j\right)$$

Indicator of unclipped branch:

$$\mathbb{I}_{j,t}(\theta) = \begin{cases}1 & \hat{A}_j \geq 0 \text{ and } r_{j,t}(\theta) \leq 1+\epsilon \\ 1 & \hat{A}_j < 0 \text{ and } r_{j,t}(\theta) \geq 1-\epsilon \\ 0 & \text{otherwise}\end{cases}$$

Gradient of clipped surrogate (利用 $\nabla_\theta r_{j,t}(\theta) = r_{j,t}(\theta)\nabla_\theta\log\pi_\theta(\cdot)$):

$$\nabla_\theta\mathcal{L}_{j,t}^{\text{clip}}(\theta) = \mathbb{I}_{j,t}(\theta)\hat{A}_j r_{j,t}(\theta)\nabla_\theta\log\pi_\theta(o_{j,t}\mid q', o_{j,<t})$$

被 clip 时 gradient 为 0 - 这就是 PPO 的 trust region mechanism, 防止单 step 走太远。

### 8.5 KL penalty 的 gradient

设 $\rho_{j,t}(\theta) = \pi_{\text{ref}}(o_{j,t}\mid\cdot)/\pi_\theta(o_{j,t}\mid\cdot)$, 则 $\hat{D}_{\text{KL}}^{(j,t)} = \rho_{j,t} - \log\rho_{j,t} - 1$, gradient:

$$\nabla_\theta\hat{D}_{\text{KL}}^{(j,t)} = -(\rho_{j,t}(\theta) - 1)\nabla_\theta\log\pi_\theta(o_{j,t}\mid q', o_{j,<t})$$

KL term 在 objective 里是 $-\beta\hat{D}_{\text{KL}}^{(j,t)}$, 所以贡献:

$$-\beta\nabla_\theta\hat{D}_{\text{KL}}^{(j,t)} = \beta(\rho_{j,t}(\theta) - 1)\nabla_\theta\log\pi_\theta(\cdot)$$

### 8.6 Final iGRPO gradient

把上面所有 pieces 合在一起:

$$\nabla_\theta\mathcal{J}_{\text{iGRPO}}(\theta) = \mathbb{E}[\cdots]\frac{1}{G}\sum_{j=1}^G\frac{1}{|o_j|}\sum_{t=1}^{|o_j|}\left[\mathbb{I}_{j,t}(\theta)\hat{A}_j r_{j,t}(\theta) + \beta(\rho_{j,t}(\theta) - 1)\right]\nabla_\theta\log\pi_\theta(o_{j,t}\mid q', o_{j,<t})$$

**Interpretation:**
- Stage 1 决定 $q'$ 的分布, 不直接贡献 gradient within iteration
- Stage 2 标准 PPO/GRPO-style token-level policy gradient
- 每 token 加权 $\hat{A}_j$ (shared across tokens in completion)
- 通过 importance ratio clipping stabilize
- KL penalty to $\pi_{\text{ref}}$ regularize

## 9. Intuition Building: 为什么 iGRPO 有效

### 9.1 从 RL 角度: Implicit curriculum

Proposition 3.1 给出 $\mathbb{E}[R_\phi(\hat{d})] = 1 - (1-V_\theta)^N$。训练初期 $V_\theta$ 小, $\hat{d}$ 弱, Stage 2 学的是"copy + minor fix"。训练后期 $V_\theta$ 大, $\hat{d}$ 接近 optimum, Stage 2 学的是"fix subtle errors"。这就是 automatic curriculum learning, 不需要人工设计 curriculum。

对比 STaR 的 explicit rationalization: STaR 也是用 model 自己的 generation 做 new training data, 但 STaR 是 SFT on filtered examples, 没有 RL 的 advantage 信号。iGRPO 是 RL, 用 advantage 信号驱动 refinement learning。

### 9.2 从 search 角度: Amplified best-of-N

Best-of-N 是 inference-time 常用技巧: sample N, return argmax reward。AlphaCode [https://arxiv.org/abs/2203.07880] 用 large N + filter, OpenAI 早期代码生成也用。Best-of-N 的 expected reward 上界是 $1-(1-p)^N$ (binary case), 跟 iGRPO Proposition 3.1 一致。

但 best-of-N 是 **test-time only**, 不改 training。iGRPO 把 best-of-N 的"找到 good draft"用进来, 再训 model "如何 refine good draft"。这就把 inference-time search 的 benefit 内化成 training-time capability。

### 9.3 从 credit assignment 角度: Better signal-to-noise

GRPO 在 binary reward 下, 假设 group 内 5 对 5 错, advantage 是 ±1, 信号强。但假设 group 全错 (hard problem), $s_R = 0$, advantage 全 0, 没 gradient signal。这是 GRPO 的 known issue, 也是 DAPO 引入 dynamic sampling 的动机 [https://arxiv.org/abs/2503.14476]。

iGRPO 在 Stage 1 找到 best draft 后, Stage 2 conditioned on 一个 "已经 partially solved" 的 context。这给 Stage 2 提供了 higher baseline success rate, 更可能在 Stage 2 拿到非零 advantage, 更稳定 gradient signal。这是 **implicit dynamic sampling** - DAPO 通过 filter all-correct / all-wrong groups, iGRPO 通过 Stage 1 conditioning 把 hard problem 变得更 tractable。

### 9.4 从 exploration 角度: 延迟 entropy collapse

Figure 3 显示 iGRPO mid-training entropy 更高。这跟"conditioning on best draft"有关。Best draft 是"已经走通一部分"的 trajectory, Stage 2 conditioned on it, 可以 explore "如何 finish the rest"。而 GRPO single-shot, 容易 stuck 在某个"开头对但末尾错"的 mode, entropy collapse 后再也 escape 不出去。

这跟 **reachable set** 的概念有关: 给定当前 policy, reachable solution space 有限。Conditioning on best draft 把 reachable set 扩到 "包含 best draft 开头的所有 continuation", 这是比 single-shot reachable set 更 rich 的 search space。

### 9.5 跟 inference-time vs training-time scaling 的关系

最近 Snell et al. (2024) [https://arxiv.org/abs/2408.03314] 详细分析了 LLM inference-time scaling。三种 axis: 
1. Best-of-N (parallel sampling)
2. Sequential refinement (multi-turn)
3. Search (tree search with value function)

iGRPO 可以理解为把 **sequential refinement (axis 2)** 从 inference-time 内化到 training-time。一旦训练完成, 推理时不需要 sequential refinement, single-shot 就能达到接近 best-of-N-with-refinement 的性能。

这跟 OpenAI o1 / DeepSeek-R1 的 internal reasoning 类似: 训练时让 model 学会 internal "thinking process", 推理时这个 process 已经 baked in, 不需要外部 prompting。

## 10. 局限与延伸思考

### 10.1 论文没充分讨论的问题

**只采样 N 个 drafts 取 argmax, 是 N=8 时 best-of-8。** Proposition 3.1 显示 $N$ 越大 conditioning 越好, 但实验固定 total budget=8, Stage 1 只能取 4 (8 / 2 split)。如果 split 成 $N=1, G=7$ 或者 $N=7, G=1$ 会怎样? 这是个未 ablate 的 axis。直觉上 Stage 1 探索越广, Stage 2 越稳, 但 Stage 2 太小 group 内 normalized advantage 不稳定。

**Stage 1 用了 best draft, 没用 worst draft。** 一个 symmetric variant 是用 worst draft 作为 "what not to do" 提示。或者 average top-k。或者 reward-weighted average。这些都没 explore。

**Draft selection 用 rule-based reward, 可能 overfit reward。** 如果 verifier 有 bug 或 edge case, $\hat{d}$ 可能是 misleading。Generative judge ablation (Table 3) 表明 partial credit 帮助, 但 judge model 本身的 reliability 没分析。

**Stage 2 conditioned on best draft 容易 collapse 成 "copy"。** 论文 prompt (Appendix C) 明确加了 instruction "Do not repeat the draft verbatim", 说明这个问题确实存在。这跟 constitutional AI / critique-style 方法的"avoid mode collapse"问题类似。

### 10.2 可能的 extension

**Multi-turn iGRPO**: Stage 2 的 refinement 再作为 Stage 3 的 draft, 递归 K 层。这是 test-time sequential refinement 的 training-time 版本。但是每多一层 budget 翻倍, 需要新的 budget 分配策略。

**Adversarial iGRPO**: Stage 1 找 worst draft, Stage 2 训练 model "如何纠正 worst case"。类似 SPAG (Cheng et al., 2024) [https://arxiv.org/abs/2411.00062], self-playing adversarial。可能对 robustness 有帮助。

**iGRPO + PRM**: 用 process reward model 给 reasoning step 打分, 替代 binary outcome reward。PRM800K [https://arxiv.org/abs/2305.20050] 数据集可以拿来训 PRM。但 PRM 的 reward hacking 是 known issue, 集成需要 care。

**iGRPO with tree search**: Stage 1 不只 sample N independent drafts, 而是 MCTS-style tree exploration。把 AlphaZero 思路引入 LLM 训练。这跟 Sutton & Barto 经典 RL 框架更接近。

**DPO-style iGRPO**: 把 Stage 1 best draft 和 Stage 2 worst refinement 作为 preference pair, 用 DPO (Rafailov et al., 2023) [https://arxiv.org/abs/2305.18290] 替代 PPO-style objective。可能更稳定, 不需要 importance ratio clipping。

**Connection to AlphaProof / AlphaProof-zero**: Google DeepMind 的 AlphaProof 用 Lean + RL, 把 search-based proving 内化。iGRPO 的思路跟 AlphaProof 的 "generate draft proof, refine" 接近, 但 iGRPO 在 natural language 上做。

**Connection to process supervision**: OmegaPRM (Wang et al., 2024) [https://arxiv.org/abs/2406.06569] 用 process supervision + tree search, 也可以跟 iGRPO 结合。

## 11. 总结: iGRPO 的核心 contribution

1. **方法论**: 一个简洁的 two-stage wrapper, 把 self-feedback 引入 GRPO without 引入 auxiliary task (跟 critique / verify 不同)。
2. **理论**: Proposition 3.1 解释 bootstrapping effect under binary reward, 形式化 "policy 改进 → draft 改进 → conditioning 改进" 的 dynamical system。
3. **实验**: 在 7B/8B/14B 三个 scale 上一致超过 GRPO + Self-Verification + Critique-GRPO。SOTA on AIME24/25 (85.62% / 79.64%) 用 OpenReasoning-Nemotron-7B + AceReason-Math。
4. **Generalization**: refinement capability 不只 math-specific, 在 GPQA 和 MMLU-Pro 上也 transfer。这说明 iGRPO 训出来的是 general reasoning refinement skill。
5. **Reusable wrapper**: 在 DAPO 和 GSPO 上都 work (+1.1 ~ +1.2), 说明 gain 来自 refinement interface, 跟 base optimizer 正交。
6. **Learning dynamics**: 延迟 entropy collapse, mid-training 探索更久, final convergence 接近 GRPO。Gain 来自 better exploration schedule, 不是 final randomness。
7. **Compute efficient**: matched budget 下 gain 几乎免费, 实测只多 13% wall-clock time, peak memory 完全相同。

paper 的最大启示: **LLM RL 不一定要发明新 optimizer, 把"self-feedback"显式编码到 training loop 里就能带来 significant gain。** 这是一个 architectural insight 而非 algorithmic breakthrough, 跟 R1-zero "不需要 SFT 也能 RL" 的 insight 类似 - 不复杂, 但有效, 且对 community 启发大。

对未来工作的方向: 把这个 wrapper 推广到 multi-turn, 推广到 PRM, 推广到 tree search, 推广到 non-math domains (code, science, agent), 都是 low-hanging fruit。iGRPO 本身是 simple, 但它打开了一个 design space。

---

**关键 references:**

- GRPO: Shao et al. (2024) [https://arxiv.org/abs/2402.03300]
- DeepSeek-R1: Guo et al. (2025) [https://arxiv.org/abs/2501.12948]
- PPO: Schulman et al. (2017) [https://arxiv.org/abs/1707.06347]
- DAPO: Yu et al. (2025) [https://arxiv.org/abs/2503.14476]
- GSPO: Zheng et al. (2025) [https://arxiv.org/abs/2507.18071]
- Dr. GRPO: Liu et al. (2025) [https://arxiv.org/abs/2503.20783]
- Self-Verification: Zhang et al. (2025) [https://arxiv.org/abs/2506.01369]
- Critique-GRPO: Zhang et al. (2025) [https://arxiv.org/abs/2506.03106]
- AceReason-Math: Chen et al. (2025) [https://arxiv.org/abs/2505.16400]
- OpenMathReasoning: Moshkov et al. (2025) [https://arxiv.org/abs/2504.16891]
- Self-Refine: Madaan et al. (2023) [https://arxiv.org/abs/2303.17651]
- Reflexion: Shinn et al. (2023) [https://arxiv.org/abs/2303.11366]
- STaR: Zelikman et al. (2022) [https://arxiv.org/abs/2203.14465]
- Self-Rewarding LM: Yuan et al. (2024) [https://arxiv.org/abs/2401.10020]
- Self-Consistency: Wang et al. (2022) [https://arxiv.org/abs/2203.11171]
- Inference-time scaling: Snell et al. (2024) [https://arxiv.org/abs/2408.03314]
- KL estimator: Schulman (2020) [http://joschu.net/blog/kl-approx.html]
- PRM800K: Lightman et al. (2023) [https://arxiv.org/abs/2305.20050]
- AlphaCode: Li et al. (2022) [https://arxiv.org/abs/2203.07880]
- DPO: Rafailov et al. (2023) [https://arxiv.org/abs/2305.18290]
- SimplerL-Zoo: Zeng et al. (2025) [https://arxiv.org/abs/2503.18892]
- SPAG: Cheng et al. (2024) [https://arxiv.org/abs/2411.00062]
- REINFORCE: Williams (1992) [https://people.cs.umass.edu/~wallin/Courses/Summer2/Lectures/rl/reinforce.pdf]
- OpenReasoning-Nemotron-7B: NVIDIA (2025) [https://huggingface.co/nvidia/OpenReasoning-Nemotron-7B]
