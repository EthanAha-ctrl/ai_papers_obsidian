---
source_pdf: JustRL Scaling a 1.5B LLM with a Simple RL Recipe.pdf
paper_sha256: 39c0a56e728bc1f1a2220ef50a241924ffc6d011394e2bd29b369a8059cec6af
processed_at: '2026-08-05T10:58:49-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 JustRL

---

## 这篇 paper 在说什么？

一句话：**大家都把 RL 搞得太复杂了，其实不用。**

---

## 背景：这个领域发生了什么

2025 年初 DeepSeek-R1 出来，证明了一个事：用 RL 训练 LLM 做 math reasoning，效果特别好。但 R1 是大模型，小模型（1.5B）怎么办？

社区的 default 路线是 **distillation**——让大模型当老师，小模型当学生，SFT 学老师的输出。简单、稳定、立刻见效。

但 distillation 有个天花板：**学生再怎么学，也超不过老师**。老师 plateau 了，学生也就到头了。

RL 能突破这个天花板。所以大家开始往小模型上堆 RL。

---

## 然后就开始"军备竞赛"了

过去 10 个月，一篇接一篇 paper，每篇都加新 trick：

- **DeepScaleR**：分 3 个 stage 训练，context length 从 8k 涨到 24k
- **FastCuRL**：分 5 个 stage，先压缩 CoT 再拉长，来回切
- **ProRL-V2**：分 9 个 stage，加 cosine length penalty，hyperparameter 动态调
- **BroRL**：每个题 rollout 512 次（别人才 8 次），暴力搜索
- **QuestA**：把题目用 partial CoT hints 增强，curriculum learning
- **POLARIS**：动态过滤数据 + adaptive temperature + test-time extrapolation

每篇 paper 都 report 一些 training instability：reward collapse、entropy drift、length explosion。然后 propose 自己的 trick 来解决。

**问题是：没人验证过这些 trick 到底有没有用。** 大家都是在已经很复杂的 baseline 上加 trick，然后 report "我比 baseline 好"。但你不知道是 trick 本身有用，还是 trick 刚好补偿了 baseline 里另一个 trick 引入的问题。

---

## JustRL 的核心 idea

**把这些 trick 全砍掉，看看会怎样。**

保留的东西少得可怜：

1. **GRPO 算法**（DeepSeek 的，用 group statistics 替代 critic network）
2. **Clip higher**（upper clip bound 用 1.28 而不是标准的 1.2，让 exploration 有更多 headroom）
3. **Rule-based verifier**（DAPO 的，检查答案对错，不需要 SymPy 这种 symbolic math library）
4. **固定 hyperparameter**，从头到尾不变
5. **Single-stage training**，不切 stage，不换 context length
6. **Standard data**（DAPO-Math-17k），不做 difficulty filtering
7. **简单 prompt**："Please reason step by step, and put your final answer within \boxed{}."

就这些。没了。

---

## 结果如何

### 性能

在 DeepSeek-R1-Distill-Qwen-1.5B 上跑 4380 步：

- 9 个 math benchmark 平均 **54.87%**
- 比 ProRL-V2（9-stage pipeline + 一堆 trick）高 1.8 个点
- 9 个 benchmark 里 6 个领先

在 OpenMath-Nemotron-1.5B 上跑 3440 步，**完全相同的 hyperparameter**：

- 平均 **64.32%**
- 比 QuestA（curriculum + question augmentation）高 0.5 个点
- 9 个 benchmark 里 5 个领先

### 计算量

- JustRL 用了 ProRL-V2 约 **一半**的 compute
- 用了 BroRL 约 **五分之一**的 compute

### Training 稳定性

最 striking 的部分。Training curve 三条线全是 smooth 的：

- **Entropy**：在 1.0-1.6 之间自然 oscillate，没有 collapse 也没有 drift
- **Mean reward**：从 -0.6 单调爬到 +0.4，没有 plateau，没有 drop
- **Response length**：从 ~8000 tokens 自然收敛到 4000-5000，**没有加任何 length penalty**

---

## 最精彩的部分：Ablation

他们加了两个 "standard trick"，**两个都让性能下降**。

### Trick 1：加 length penalty

DAPO 原来的做法——response 超过一定长度就扣 reward。

直觉上："防止模型话太多，应该有用吧？"

实际结果：AIME24 从 55% 跌到 50%。**Entropy 从 1.2-1.4 直接跌到 0.5-0.6**，exploration 被 penalty 压垮了。

模型学到的不是"更高效地推理"，而是"赶紧说短一点别被扣分"。Reasoning 没变好，exploration 先死了。

### Trick 2：在 length penalty 基础上，换 robust verifier

用 DeepScaleR 的 robust verifier（symbolic 方法减少 false negatives——本来对的答案被判错的情况）。

直觉上："减少误判，学习信号更干净，应该更好吧？"

实际结果：AIME24 进一步跌到 45%。

为什么？两个 hypothesis：

1. **Strict verifier 提供 richer gradient spectrum**。8 个 rollout 中 3 对 5 错 vs 6 对 2 错，advantage 的分布不同，learning signal 更丰富。Robust verifier 倾向于把答案都判对，group 内 variance 变小，GRPO 的 advantage signal 变弱。

2. **Strict verifier 逼模型 develop internal precision**。Strict verifier 要求精确格式（`\boxed{}` 内严格匹配），模型被迫自己学会精确计算。Robust verifier 帮模型纠正错误，模型就不需要自己学会精确——但 inference 时没有 verifier 帮忙，generalization 就差了。

---

## 为什么 simple recipe 能 work

### 1. GRPO 天然过滤了 trivial samples

8 个 rollout 全对 → advantage = 0，zero gradient
8 个 rollout 全错 → advantage = 0，zero gradient
只有 mixed outcome 的 prompt 贡献 gradient

这就是 **dynamic difficulty filtering 的 implicit 版本**，不需要显式做。

### 2. Clip higher 保持了 exploration

Upper bound 1.28（而非 1.2）让 "被高度 reinforce 的 reasoning pattern" 还能继续 increase probability。如果 clip 在 1.2 就截断，rare-but-correct long-chain reasoning 永远无法被充分 amplify，policy 逐渐 collapse 到 short responses。

### 3. Length 自然收敛

Long-but-wrong response 被 disadvantage = -1，而且因为长，per-token 累积 gradient 更强，被 push away 更狠。Long-but-correct 和 short-but-correct 的 advantage 一样，但 short 版本 per-token update 更集中。模型自己 discover "短而正确" 比 "长而正确" 更 efficient。

**这是 emergent behavior，不需要 explicit penalty 来 impose。**

### 4. Distilled backbone 提供强 starting point

KL regularization 的作用是防止 policy 离 reference model 太远。但 distilled model 本身已经是 strong policy，不需要 anchor。去掉 KL 反而让 RL 能继续 push 超过 distillation 上限。

---

## 这篇 paper 真正的 takeaway

不是 "simple is always better"。

是 **"先跑 simple baseline，确认它 fail 了，再加 complexity"**。

现在的问题不是大家加 trick——trick 可能在某些 setting 下真的有用。问题是大家**没跑 simple baseline 就直接堆 trick**，然后 report "我比复杂 baseline 好"。你不知道你是在 solve 真正的问题，还是在 compensate 前一个 trick 引入的问题。

JustRL 证明：在 1.5B math reasoning 这个 setting 下，simple recipe 跑 4000+ 步，training curve smooth，performance 超过 complex pipeline，compute 减半。那些"RL is unstable"的 motivation，可能本身就是 complex recipe 造成的 self-inflicted problem。

**类比 AlexNet 之前 vision 领域堆 hand-crafted features 的情况——scale + simple end-to-end approach 打败了 domain-specific engineering。JustRL 在 RLVR 上做了类似的事。**

Rich Sutton 的 Bitter Lesson 又一次应验：general methods that scale with computation beat domain-specific methods。

---

## 一句话总结

**在 RLVR for small LLMs 这个领域，大家花 10 个月堆出来的 tricks，可能大部分是在解决自己堆 trick 引入的问题。把 trick 砍掉，问题也消失了。**

---

# JustRL: A Minimalist Manifesto for RL at Scale

这篇 paper 的核心 thesis 非常 Karpathy 式:在大家都往 complexity 方向狂奔的时候,有人停下来问 "Is this complexity necessary?",然后用一个 minimal recipe 跑出 SOTA。这是 RLVR(RL with Verifiable Rewards)for small LLMs 领域的一记"减法"宣言。

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

---

## 1. Context: 这个领域为什么会变复杂?

2025 年初 DeepSeek-R1 释放的 RLVR signal 极强,但对于 small language models(SLMs,~1.5B),大家默认走 distillation 路线,因为 efficient, stable。然而 distillation 有一个 fundamental 上限:teacher model 的 capability bound。一旦 teacher plateau,再多数据再多 epoch 也无法突破。

RLVR 成为"突破 distillation 上限"的关键工具,但社区在过去 10 个月内像滚雪球一样累加 tricks。Table 1 展示的 evolution:

- **STILL-3**:Hyperparameter tuning 探索
- **DeepScaleR**:3-stage training,context length 8k→16k→24k,加 KL reset + dynamic sampling
- **FastCuRL**:5-stage,alternating CoT compression/extension
- **ProRL-V1/V2**:8-stage,加 scheduled length penalty + cosine length penalty
- **BroRL**:在 ProRL-V2 之后,rollout N 从 16 提到 512,exhaustive exploration
- **QuestA**:Curriculum + question augmentation(用 partial CoT hints)
- **POLARIS**:Dynamic dataset filtering + adaptive temperature + test-time extrapolation

每篇 paper 都 report 各种 instabilities(reward collapse, entropy drift, length explosion),然后 propose trick 解决。但没人 isolated test 这些 tricks 的边际收益。JustRL 提出的 hypothesis:**复杂度本身可能就是某些 instability 的 root cause**。

参考:
- DeepSeek-R1: https://arxiv.org/abs/2501.14342
- DeepScaleR: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2
- ProRL: https://arxiv.org/abs/2505.24864
- ProRL-V2: https://hijkzzz.notion.site/prorl-v2
- QuestA: https://arxiv.org/abs/2507.13266
- POLARIS: https://hkunlp.github.io/blog/2025/Polaris
- BroRL: https://arxiv.org/abs/2510.01180

---

## 2. JustRL Recipe 的核心:GRPO 数学详解

JustRL 用的是 GRPO(Group Relative Policy Optimization),DeepSeek 在 DeepSeekMath 中提出。它的关键 insight:**去掉 critic network,用 group statistics 作为 implicit baseline**。

### 2.1 从 PPO 到 GRPO 的动机

传统 PPO advantage 需要 critic V(s):

$$A(s,a) = R + \gamma V(s') - V(s)$$

在 LLM RLHF 场景下,training 一个 1.5B critic 是 expensive,unstable,critic 的 estimation error 会直接 inject noise 到 policy gradient。DeepSeek 的 insight 是:**对于 outcome-supervised rewards(只看最终答案对错),可以用 group-relative baseline 替代 critic**。

### 2.2 GRPO 公式

给定 prompt $q$,从 old policy $\pi_{\theta_{old}}$ 采样一组 responses $\{o_1, o_2, \ldots, o_G\}$,每个 response $o_i$ 获得 reward $r_i$(在 JustRL 里是 binary:0 或 1)。

**Advantage 计算**:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

变量含义:
- $A_i$:第 $i$ 个 response 的 group-normalized advantage
- $r_i$:第 $i$ 个 response 的 reward(binary,在 math reasoning 里是 "answer correct or not")
- $\text{mean}(\cdot)$:group 内 reward 均值,作为 implicit baseline
- $\text{std}(\cdot)$:group 内 reward 标准差,normalize scale
- $G$:group size,JustRL 用 $G=8$ rollouts

关键 intuition:如果 8 个 rollout 全对,所有 advantage = 0,这个 prompt 在这一步贡献 zero gradient。如果 4 个对 4 个错,正确的 advantage = +1,错误的 = -1,产生清晰 contrastive signal。**这天然实现了 dynamic difficulty filtering**:trivially easy(全对)和 trivially hard(全错)的 samples 自动被 zero out,不需要 explicit dynamic sampling。

**Policy gradient loss**:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q, \{o_i\}_{i=1}^G}\left[\frac{1}{G}\sum_{i=1}^G \min\left(\rho_i A_i,\, \text{clip}(\rho_i, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) A_i\right)\right]$$

变量含义:
- $\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}$:importance sampling ratio,new policy 相对 old policy 的 probability ratio
- $\epsilon_{\text{low}} = 0.2$:lower clip(对应 clip ratio 0.8)
- $\epsilon_{\text{high}} = 0.28$:upper clip(对应 clip ratio 1.28)— **这就是 "clip higher"**
- $\min(\cdot, \cdot)$:standard PPO pessimistic bound,取 lower of two terms 防止 over-optimization

**JustRL 显式舍弃的两项**:

$$\text{传统 GRPO} = \mathcal{L}_{\text{GRPO}} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) - \eta \cdot \mathbb{H}[\pi_\theta]$$

- $\beta \cdot \text{KL}$:KL regularization to reference model($\beta=0$ in JustRL)
- $\eta \cdot \mathbb{H}$:entropy bonus to encourage exploration($\eta=0$ in JustRL)

为什么 JustRL 敢舍弃 KL?因为 backbone 是 distilled model,本身就是 strong policy,KL anchor 没必要;DeepSeek-R1 paper 自己也发现 KL 会 limit upper bound。

为什么舍弃 entropy bonus?因为 **clip higher 已经隐式提供了 exploration mechanism**。

参考:
- DeepSeekMath (GRPO 原始 paper): https://arxiv.org/abs/2402.03300
- DAPO (clip higher 来源): https://arxiv.org/abs/2503.14476
- veRL framework: https://github.com/volcengine/verl

---

## 3. Clip Higher:Why 1.28 Instead of 1.2?

这是 JustRL 唯一保留的 trick,但理解它对 build intuition 至关重要。

### 3.1 标准 PPO clip 的问题

Standard PPO 用 $[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$,对称设计基于 trust region assumption。但在 long-horizon RLVR 中:

- 某个 critical reasoning step(比如发现一个 key algebraic identity)被 policy 高度 reinforce 时,需要 $\rho_i$ 显著超过 1.2 才能 update
- Standard clip 在 1.2 处 truncate gradient,导致 **policy 无法充分 reinforce 那些 rare-but-crorrect long-chain reasoning patterns**
- 长期来看,这造成 **entropy collapse**:policy 倾向于 sticking with already-high-probability short responses,因为无法 explore 更长的 CoT chains

### 3.2 DAPO 的不对称设计

$$\text{clip}(\rho_i, 0.8, 1.28)$$

- **Lower bound 0.8**:防止 probability collapse to 0(standard PPO 设计)
- **Upper bound 1.28**:给 "good action 的 probability increase" 留出 6.67% 额外 headroom

具体数字 $\epsilon_{\text{high}} = 0.28$ 的来源:DAPO 经验调出来的,允许 $\rho$ 上探到 1.28 而非 1.2,刚好足够让 long CoT exploration 的 signal 流过去,同时不至于 completely lose trust region property。

### 3.3 Intuition

把它想象成 RLHF 的 "information flow":

- Clip lower bound(0.8):**保护 rare exploration**,防止 policy 过快放弃 低概率但有价值的探索路径
- Clip upper bound(1.28 而非 1.2):**允许 strong reinforcement**,让 discovered reasoning patterns 能真的被 amplify 到 policy 中

这两个机制协同形成 exploration/exploitation 的"软平衡"。JustRL 的 ablation 证实了这一点:加入 length penalty 后,entropy 从 1.2-1.4 跌到 0.5-0.6,说明 clip higher 提供的 exploration 被 penalty 压垮了。

参考 DAPO paper Figure 4,有 entropy collapse 的可视化:https://arxiv.org/abs/2503.14476

---

## 4. 完整 Recipe 配置剖析

Table 2 列出的 hyperparameters:

| Hyperparameter | Value | Intuition |
|---|---|---|
| Advantage Estimator | GRPO | 去掉 critic,group-relative baseline |
| Use KL Loss | No | Distilled backbone 已经 strong,不需要 anchor |
| Use Entropy Regularization | No | Clip higher 已经隐式提供 exploration |
| Train Batch Size | 256 | 256 prompts × 8 rollouts = 2048 samples per step |
| Max Prompt Length | 1k | DAPO-Math-17k 的 prompt 都不长 |
| Max Response Length | 15k | 给 long CoT 留充足空间 |
| PPO Mini Batch Size | 64 | 4 mini-batches per PPO epoch |
| PPO Micro Batch Size/GPU | 1 | Memory conservative,32 GPU × 1 = 32 micro-batch |
| Clip Ratio Range | [0.8, 1.28] | DAPO 的 clip higher |
| Learning Rate | 1e-6 constant | 极小 LR,稳定 RL training |
| Temperature | 1.0 | Rollout 时不调温 |
| Rollout N | 8 | Group size for GRPO |
| Reward Function | DAPO verifier | Rule-based,无 SymPy overhead |

**关键设计决策的 intuition**:

1. **Single-stage**:固定 context length 16k。如果 multi-stage 是为了 "先学会短再学长",那 length 自然收敛(见 Section 4.3)说明模型自己会找到 right length。
2. **Fixed hyperparameters**:与 dynamic temperature scheduling 相反,固定 temperature 1.0 让 training dynamics 保持 stationary,容易分析。
3. **Max response length 15k vs 16k context**:留 1k 给 prompt + special tokens,精确的 budget 控制。
4. **Learning rate 1e-6 极小**:RL 阶段的 LR 通常比 SFT 小 1-2 个数量级,因为 RL 的 gradient noise 远大于 supervised gradient。

**Prompt suffix**(极简但关键):

```
Please reason step by step, and put your final answer within \boxed{}.
```

这个 prompt 设计确保:
- Model 输出 CoT(激活 reasoning mode)
- Final answer 在 `\boxed{}` 里,被 rule-based verifier 容易 parse
- 不需要 few-shot examples 或 chain-of-thought demos

---

## 5. 实验结果:Two Backbones, Same Recipe

### 5.1 JustRL-DeepSeek-1.5B

起点:DeepSeek-R1-Distill-Qwen-1.5B(backbone avg 37.65%)
训练:4,380 steps,32 × A800-80GB,~15 days

| Model | AIME24 | AIME25 | AMC23 | MATH | Minerva | Olympiad | HMMT | BRUMO | CMIMC | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Backbone | 29.90 | 22.40 | 63.82 | 84.90 | 34.65 | 45.95 | 13.44 | 30.94 | 12.89 | 37.65 |
| DeepScaleR | 40.21 | 28.65 | 73.83 | 89.30 | 39.34 | 52.79 | 18.96 | 40.00 | 21.00 | 44.88 |
| ProRL-V2 | 51.87 | 35.73 | 88.75 | 92.00 | 49.03 | 67.84 | 19.38 | 47.29 | 25.86 | 53.08 |
| **JustRL-DeepSeek** | **52.60** | **38.75** | **91.02** | 91.65 | **51.47** | **67.99** | **21.98** | **52.71** | 25.63 | **54.87** |

JustRL 在 9 个 benchmark 中 6 个领先。比 ProRL-V2(9-stage + dynamic hyperparams)**高 1.79 个点 avg**,且 compute 用量是 ProRL-V2 的一半左右。

Compute budget 对比(Table 4):

| Model | Steps | Batch | Rollout N | Context | Token Budget |
|---|---|---|---|---|---|
| DeepScaleR | 1,750 | 128 | 8 | 8k→16k→24k | 2.2×10⁸k |
| ProRL-V1 | 2,450 | 256 | 16→32→16 | 8k→16k | 2.1×10⁸k |
| ProRL-V2 | +1,000 | 256 | 16→32→16 | 8k→16k→8k | 2.8×10⁸k |
| BroRL | +191 | 128 | 512 | 16k | 6.8×10⁸k |
| **JustRL** | 4,380 | 256 | 8 | 16k | **1.4×10⁸k** |

JustRL 比 ProRL-V2 节省 ~50% tokens,BroRL 的 ~20%。BroRL 用 512 rollouts 做 exhaustive exploration,但 191 steps 就 plateau,说明 brute-force exploration 在 RLVR 中效率很低。

### 5.2 JustRL-Nemotron-1.5B

起点:OpenMath-Nemotron-1.5B(backbone avg 56.74%,本身就强)
训练:3,440 steps,**完全相同的 hyperparameters**

| Model | AIME24 | AIME25 | AMC23 | MATH | Minerva | Olympiad | HMMT | BRUMO | CMIMC | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Backbone | 58.75 | 48.44 | 90.55 | 92.40 | 26.93 | 71.70 | 30.10 | 61.67 | 30.08 | 56.74 |
| QuestA | 71.56 | 62.08 | 93.44 | 92.95 | 32.08 | 72.28 | 40.94 | 67.50 | 41.48 | 63.81 |
| **JustRL-Nemotron** | 69.69 | **62.92** | **96.02** | **94.15** | 30.24 | **76.59** | 40.63 | 66.88 | **41.72** | **64.32** |

**关键 takeaway**:同一套 hyperparameters 在两个不同的 backbone 上都 work,这强烈暗示 JustRL recipe 不是 overfitting 到某个特定 model 的 quirks。一个通用 recipe 能跨 model 转移,才是真正的"找到 fundamental mechanism"。

### 5.3 为什么 starting point 重要?

对比两个 backbone:
- DeepSeek-1.5B:37.65% → 54.87%,**gain 17.22 个点**
- Nemotron-1.5B:56.74% → 64.32%,**gain 7.58 个点**

这印证 paper Section 1 的论点:**distillation 的天花板决定 RL 能多有效的上限**。Nemotron 本身已经更强(可能是更好的 distillation),所以 RL 的边际收益更小,但 absolute performance 更高。

这给未来 research 一个 hint:**distillation 和 RL 应该被看作 continuum**,而非替代品。Strong distillation 给 RL 更好的 starting point,RL 持续 push distillation 上限。

参考:
- OpenMath-Nemotron: https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B
- DeepSeek-R1-Distill-Qwen-1.5B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- DAPO-Math-17k: https://huggingface.co/datasets/BytedanceResearch/DAPO-Math-17k

---

## 6. Training Dynamics:三条曲线的解读

Figure 2 跟踪三个关键 dynamics,这是 JustRL 最有说服力的证据。

### 6.1 Entropy Dynamics

$$H(\pi_\theta(\cdot \mid q)) = -\sum_o \pi_\theta(o \mid q) \log \pi_\theta(o \mid q)$$

JustRL 的 entropy 在 1.0-1.6 区间 oscillate,**没有 upward drift(exploration collapse)也没有 downward drift(premature convergence)**。

Intuition:
- Upward drift = policy becoming uniformly random,signal vanishing
- Downward drift = policy collapsing to deterministic,exploration dead
- 1.0-1.6 oscillation = healthy exploration/exploitation balance

为什么 oscillation 是 healthy 的?GRPO 在每个 batch 里 zero out "all correct / all wrong" groups,只在 mixed-outcome groups 上产生 gradient。这种 adaptive filtering 让 entropy 自然在某个 equilibrium 附近波动,不需要 external entropy regularization。

### 6.2 Mean Reward Dynamics

Reward 从 -0.6 单调爬升到 +0.4。这个起点是负值很关键——说明初始 policy 在 binary verifier 下正确率 < 50%(考虑到 binary reward,normalized 之后 mean 在零附近,所以 -0.6 意味着模型刚开始时很多 responses 错误,且错的样本被 disadvantage 强烈 push down)。

**没有 plateau,没有 collapse**。这意味着:
- ProRL-V2 报告的 "length drift" 问题在 JustRL 中根本没出现
- BroRL 报告的 "plateau after 3K steps" 问题在 JustRL 中也没出现
- 多个 paper 描述的 "KL divergence explosion" 问题(因为没用 KL)也没出现

### 6.3 Response Length Dynamics

模型 response length 从 ~8000 tokens 自然收敛到 4000-5000 tokens,**没有 explicit length penalty**。

为什么 length 会自然 decrease?Intuition:

考虑 GRPO 的 advantage signal 在做什么。对于同一个 prompt,8 个 rollouts 中:
- 短而正确的 response:reward = 1,advantage = +1
- 长而正确的 response:reward = 1,advantage = +1
- 错误的 response:reward = 0,advantage = -1

由于 advantage 不直接依赖 length,但 **policy gradient update 是 per-token 的**,在 length $L$ 的 response 上累积的 gradient magnitude 与 $L$ 成比例。Long-but-wrong responses 会获得更强的 negative gradient,推动 policy 远离这些长错误路径。Long-but-correct 与 short-but-correct 共享相同 advantage,但 short 版本在 per-token normalize 后 update signal 更集中。**结果:model 学到 "短而正确" 比 "长而正确" 更 efficient**,自然压缩 length。

这是 emergent behavior,而不是 imposed behavior。Explicit length penalty 是 "告诉模型该多长",JustRL 让模型自己 discover right length。

参考对 length penalty 批评的 paper:DLER(Doing Length Penalty Right): https://arxiv.org/abs/2510.15110

---

## 7. Ablation Studies:负面结果蕴含的深层 insight

这是 paper 最有价值的部分。JustRL 测试两个 "standard tricks",**两个都让 performance 下降**。

### 7.1 Ablation 1: Overlong Penalty

加入 DAPO 的 length penalty(对 response 最后 4k tokens 添加惩罚):

$$r_{\text{new}} = r_{\text{correct}} - \lambda \cdot \mathbb{1}[\text{len}(o) > L_{\text{threshold}}]$$

结果:AIME24 从 55% plateau 到 50%。
机制:Entropy 从 1.2-1.4 跌到 0.5-0.6,**exploration 被 penalty 直接压垮**。

Intuition 解释:
- Length penalty 在 reward function 上 create adversarial objective
- Model 学会 game the penalty(shorter responses get higher reward),而非学会 reasoning
- 一旦 policy collapse 到 short responses,exploration budget 消失,无法 discover 新的 reasoning patterns
- DAPO 在它们 setting 下 work,是因为它们的 starting policy 更需要被 "推短";JustRL 已经自然收敛到合理 length,penalty 变成多余压力

**这个 ablation 直接反驳了 "length penalty 是 RLVR 必备" 的 implicit 假设**。

### 7.2 Ablation 2: Overlong Penalty + Robust Verifier

进一步加入 DeepScaleR 的 robust verifier(用 symbolic methods 减少 false negatives):

结果:AIME24 进一步 plateau 到 45%。

这个结果反直觉。"减少 false negatives 应该提供 cleaner learning signal",对吧?但实际 performance 下降。

JustRL 给出两个 hypothesis:

**Hypothesis 1**:Stricter verifier 提供 richer gradient spectrum
- Strict verifier 产生更多 partial-credit scenarios(8 个 rollout 中 3 对 5 错)
- Robust verifier 倾向于 binary outcome(全对或大部分对),丢失 contrastive signal
- GRPO 的 advantage normalization 依赖 group variance,richer spectrum = higher variance = stronger gradient

**Hypothesis 2**:Strict verifier forces internal precision
- Strict verifier 要求精确 formatting(`\boxed{}` 内必须严格匹配)
- Model 被迫 develop internal computational precision 来满足 format
- Robust verifier externally corrects errors,model 不需要 develop internal precision
- Inference time 没有 verifier,model 的 internal precision 才是真正的 generalization 能力

**Intuition**:Verifier 不仅是 reward source,更是 **implicit curriculum**。Strict verifier 教 model "be precise",robust verifier 教 model "be approximately right"。前者 better for generalization,因为推理时没有 external verifier 来 correct mistakes。

这个 insight 让我联想到 Inverse Reinforcement Learning 的一个观察:sparsity of reward 是 feature,不是 bug。Sparse but strict reward 强迫 model 自己 build world model,而不是 exploit reward shaping。

### 7.3 Ablation 的方法论意义

更深层 takeaway:**"Standard tricks" 没有跨 context 的鲁棒性**。DAPO 的 length penalty 在 DAPO setting work,DeepScaleR 的 robust verifier 在 DeepScaleR setting work。但把它们叠加到一个 minimal recipe 上,反而 degrade performance。

这暗示领域内的 "best practices" 可能在 **互相补偿彼此引入的问题**。Multi-stage pipeline 引入 instability,加 dynamic hyperparameters 缓解 instability,dynamic hyperparameters 引入 entropy collapse,加 entropy regularization 缓解 collapse,entropy regularization 引入 exploration drift,加 reference reset 缓解 drift……最终你得到一个 8-stage pipeline with 12 tricks,每一个都在补偿前面 trick 引入的问题。JustRL 从 root 上不引入问题,就不需要这一连串补偿。

参考:
- "Tricks or traps" paper: https://arxiv.org/abs/2508.08221
- RL for LLMs 陷阱分析

---

## 8. 系统架构:veRL Framework 与工程细节

虽然 paper 没有详细展开 system architecture,但理解 veRL 对复现至关重要。

veRL(Volcano Engine Reinforcement Learning)是字节开源的 RLHF training framework,核心设计:

```
[Prompts] → [Actor(1.5B)] → [Responses] → [Rollout Buffer]
                                                    ↓
                                          [Verifier (Rule-based)]
                                                    ↓
                                              [Rewards]
                                                    ↓
                                          [GRPO Advantage Compute]
                                                    ↓
                                  [Actor Backprop + Update]
```

**HybridFlow architecture**(参考 https://github.com/volcengine/verl):

- Actor model 用 Megatron 或 FSDP 分布式训练
- Rollout 用 vLLM 做 efficient generation
- 显存上:Actor + Reference model + Rollout engine 共存
- JustRL 不用 reference model(无 KL),省一份 1.5B 参数显存
- 不用 critic(GRPO),再省一份

**JustRL 实际显存占用估算**(32 × A800-80GB):

- Actor(1.5B,FP16/BF16):~3GB weights + optimizer states(Adam)~12GB + gradients ~3GB = ~18GB
- Rollout(vLLM,KV cache):~20-40GB 取决于 batch
- Activation memory(max response 15k × micro-batch 1):需要 activation checkpointing

veRL 用 3D parallelism(data + tensor + pipeline),把 32 GPU 分配给 actor 和 rollout。这是为什么 micro batch size = 1 per GPU——长 context 下 activation memory 极大。

参考:
- veRL paper: https://arxiv.org/abs/2409.19256
- HybridFlow (EuroSys 2025): https://github.com/volcengine/verl

---

## 9. 我的 Intuition 与联想

### 9.1 与 AlexNet 时刻的类比

CNN 时代之前,vision 领域堆叠 features engineering(SIFT, HOG, SURF)。AlexNet 的 takeaway:**end-to-end learning at scale 让 hand-crafted features 变得多余**。

JustRL 之于 RLVR,有点类似 AlexNet 之于 vision:**at adequate scale, simple end-to-end approaches beat hand-crafted engineering**。Table 1 列出的 8 种 tricks,本质都是 "manually engineered RL stabilization",JustRL 证明这些 manual engineering 在 scale 足够时不需要。

### 9.2 与 "Bitter Lesson" 的呼应

Rich Sutton 的 "Bitter Lesson":general methods that scale with computation consistently beat domain-specific methods。

JustRL 是 Bitter Lesson 在 RLVR for LLMs 上的实证:
- Domain-specific methods:multi-stage pipeline,curriculum,length penalty,robust verifier
- General method that scales:GRPO + clip higher + adequate training steps + verifiable reward
- 后者 win

### 9.3 关于 "RL is unstable" 的 myth

JustRL 的 training curves 直接挑战这个 myth。在 distillation 起点足够好的情况下,RL 训练 4000+ steps **完全 stable**。这个 myth 可能源自 RLHF with RM(reward model)的早期经验——RM 本身有 reward hacking 问题,引入 instability。但 RLVR with rule-based verifier **本质不同**:

- RLHF with RM:reward 是 learned,可以被 exploit,reward hacking 频发
- RLVR with rule-based verifier:reward 是 deterministic function of output correctness,**无法被 hack**(除非 verifier 本身有 bug)

JustRL 选择 rule-based DAPO verifier 而非 learned verifier model,正是为了利用这种 verifiable property。CompassVerifier-3B 只在 evaluation 时辅助 reduce false negatives,不进 training loop。

### 9.4 关于 "RL 不会 generalize" 的 myth

JustRL 在 9 个 benchmark 上的 broad improvement(不只是 AIME24,也包括 AIME25, AMC23, HMMT, BRUMO, CMIMC 等未见过的竞赛题)反驳了 "RL overfits" 的指控。

关键:**RLVR 不学 task-specific patterns,学的是 reasoning procedure**。GRPO advantage 在 "对/错" 上 normalize,而正确的 reasoning procedure 可以 generalize 到新题。这与 SFT on problem-solution pairs 的本质不同——SFT 学 answer pattern,RL 学 reasoning pattern。

### 9.5 与 Scaling Laws 的关系

ProRL-V2 paper 自己提出 RL scaling laws:performance 改进 log-linearly with training steps。JustRL 的 4000+ steps training curve 显示 **smooth monotonic improvement**,符合这个 scaling law。

如果 JustRL 继续训练到 8000 steps,performance 可能继续 improve。**JustRL 的 simplicity 让 longer training 变得可行**——多 stage pipeline 在 stage transition 时需要 manual intervention,single-stage 可以让训练无限延续直到 saturate。

这给未来工作一个 hint:**RL scaling 的 bottleneck 可能不在 algorithm complexity,而在 recipe simplicity enables longer stable training**。

### 9.6 关于 Entropy 与 Exploration 的精细 control

JustRL 的 entropy 在 1.0-1.6 oscillate,这个具体数字 range 可以推算 exploration budget:
- Entropy 1.0 大约对应 effective branching factor $e^{1.0} \approx 2.7$,即每个 token 选择在 ~3 个合理选项间
- Entropy 1.6 对应 effective branching factor $e^{1.6} \approx 5.0$
- 这个 range 让模型既能 commit to high-confidence reasoning steps(entropy 低),又能 explore alternatives(entropy 高)

Clip higher 在这里的作用是**防止 entropy 过快 collapse**。如果 clip upper bound 是 1.2,任何一个"被高度 reinforce"的 reasoning pattern 的 probability 增长受限,但同时 lower bound 0.8 也限制了 "alternative patterns" 的快速放弃。**这种不对称 design 创造了 "forward but not collapse" 的 dynamic**。

### 9.7 关于 Reward Sparsity 与 Rich Learning Signal

Binary outcome reward 是 sparse 的——只有 final answer 对错。但 GRPO 的 group-relative advantage + clip higher 让这个 sparse signal 变成 effective learning signal:

- 8 rollouts 中 3 对 5 错,正确 5 个的 advantage 都是 +1.4,错误 3 个都是 -1.4
- 每个 token 上的 policy gradient $\nabla_\theta \log \pi_\theta(o_i|q) \cdot A_i$
- 长 correct response 的每一步被 +1.4 加权,长 wrong response 的每一步被 -1.4 加权
- 长 wrong response 因为 length 长,累积 gradient 更强,**被 push away 更狠**

这解释了 length 自然收敛——错的长 response 被强烈惩罚,对的长 response 和对的短 response 之间的差异被 clip higher 平衡。

---

## 10. Limitations 与未来方向

Paper 在 Limitations 部分诚实承认:

1. **Domain limitation**:只在 math reasoning 测试,generalization 到 code, general QA 未知
2. **Scale limitation**:只在 1.5B 测试,7B/13B/70B 的 dynamic 可能不同
3. **Component attribution**:无法 isolate 是 hyperparameters, verifier, 还是 data 的具体贡献
4. **Long-horizon**:4000+ steps 没看到 plateau,但更长的 horizon 是否会出现新 failure mode 未知

**我补充几个 open questions**:

1. **Code reasoning**:Code verification 也是 verifiable reward(unit tests pass/fail)。JustRL recipe 能否直接 transfer?Code RL 的探索空间可能更大(程序空间比数学答案空间更 sparse)。

2. **Multi-modal reasoning**:在 visual math(几何证明)上,verifier 设计是 bottleneck。JustRL 的 simplicity philosophy 是否适用?

3. **Adversarial reward shaping**:如果 verifier 有 bug,model 会 exploit。JustRL 依赖 rule-based verifier,这个 verifier 本身的 robustness 是 silent dependency。

4. **Compute floor**:32 × A800 × 15 days ≈ 12,000 GPU-hours,对 academic lab 是 substantial。是否能 further democratize?

5. **Going beyond distillation start**:如果 starting point 是 pre-trained-only model(无 distillation),JustRL recipe 是否还 work?Distillation 给的 strong starting point 是 hidden precondition。

---

## 11. 资源汇总

- **Paper**:HuggingFace collections: https://huggingface.co/collections/hbx/justrl
- **Code**:GitHub: https://github.com/thunlp/JustRL
- **Trained models**:JustRL-DeepSeek-1.5B, JustRL-Nemotron-1.5B
- **Framework**:veRL: https://github.com/volcengine/verl
- **Training data**:DAPO-Math-17k: https://huggingface.co/datasets/BytedanceResearch/DAPO-Math-17k
- **Backbones**:
  - DeepSeek-R1-Distill-Qwen-1.5B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - OpenMath-Nemotron-1.5B: https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B
- **Key references**:
  - DeepSeek-R1: https://arxiv.org/abs/2501.14342
  - DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
  - DAPO: https://arxiv.org/abs/2503.14476
  - DeepScaleR: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2
  - ProRL: https://arxiv.org/abs/2505.24864
  - QuestA: https://arxiv.org/abs/2507.13266
  - POLARIS: https://hkunlp.github.io/blog/2025/Polaris
  - BroRL: https://arxiv.org/abs/2510.01180
  - e3: https://arxiv.org/abs/2506.09026
  - CompassVerifier: (Liu et al. 2025c)
  - "Tricks or traps": https://arxiv.org/abs/2508.08221
  - DLER: https://arxiv.org/abs/2510.15110
  - MathArena (eval benchmarks): https://matharena.ai

---

## 总结:JustRL 给我的三句话 intuition

1. **Simplicity is a feature, not a constraint**:Minimal recipe 让 training dynamics 变得 analyzable,反而 enables longer stable training,compounding 出比 complex pipeline 更好的最终 performance。

2. **Most "RL instability" is self-inflicted**:Multi-stage pipeline,dynamic hyperparams,explicit length penalty 这些 tricks 引入了 instability,然后 propose 更多 tricks 解决。JustRL 从 root 上不引入,就不需要解决。

3. **Verifier design is silent curriculum**:Strict rule-based verifier 强迫 model develop internal precision,robust verifier externalize precision to verifier,inference 时失去 generalization。

JustRL 的方法论 takeaway:**Start simple, scale up, only add complexity when a simple baseline demonstrably fails**。这是 Bitter Lesson 在 RLVR 时代的具体体现,也是对当前 "complexity arms race" 的一记清醒剂。
