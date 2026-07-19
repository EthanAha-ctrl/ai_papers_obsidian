---
source_pdf: Black-Box On-Policy Distillation of Large Language Models.pdf
paper_sha256: 907883e9eca2b6a1844d8ab376acd0921b549e199745162200f9b1f56bc2a0ef
processed_at: '2026-07-18T19:32:38-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# GAD: Generative Adversarial Distillation 深度解析

## 1. 核心动机与问题定义

这篇paper 来自 Microsoft Research (Tianzhu Ye, Li Dong, Furu Wei 等人)，解决了一个非常实际的问题: **black-box LLM distillation**。先来 build intuition 关于为什么这个问题重要且困难。

**White-box vs Black-box Distillation 的本质区别:**

White-box setting 下，student 可以访问 teacher 的 full distribution $p(y|x)$，因此可以用 KL divergence 这种 likelihood-based objective:

$$\mathcal{L}_{\text{fwd-KL}} = \mathbb{E}_{x \sim \mathcal{T}} \left[ D_{\text{KL}}(p(y|x) \| q_\theta(y|x)) \right]$$

这里 $p$ 是 teacher distribution, $q_\theta$ 是 student distribution, $\theta$ 是 student parameters, $D_{\text{KL}}$ 衡量两个分布的差异。Forward KL 是 **mass-covering** (mode-covering) behavior，student 试图覆盖 teacher 所有 mode。

Reverse KL 则相反:

$$\mathcal{L}_{\text{rev-KL}} = \mathbb{E}_{x \sim \mathcal{T}} \left[ D_{\text{KL}}(q_\theta(y|x) \| p(y|x)) \right]$$

这是 **mode-seeking** behavior，但要求 student 在 teacher 低概率区域必须保持低概率，这 necessitates 评估 $p(y|x)$ 在 student samples 上的值，从而需要 white-box access。

**关键 insight:** Black-box setting 下，你只有 text samples $\{(x, y_t)\}$ 来自 teacher，没有 logits。SeqKD (Sequence-Level Knowledge Distillation, Kim & Rush 2016) 就是简单 SFT:

$$\mathcal{L}_{\text{SeqKD}} = -\mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ \log q_\theta(y_t | x) \right]$$

这本质上就是 maximum likelihood，相当于 forward KL 的 Monte Carlo 估计，所以 SeqKD 也是 mode-covering 的，且是 **off-policy** 的 (student 学习 teacher 的 samples，不是自己的 samples)。

Reference: 
- SeqKD paper: https://arxiv.org/abs/1606.07947
- MiniLLM (reverse KLD for white-box): https://arxiv.org/abs/2306.08543
- On-policy distillation (Agarwal et al.): https://arxiv.org/abs/2310.20689

---

## 2. GAD 的核心思想

GAD 的核心 idea 非常 elegant: 把 distillation frame 成 GAN-like minimax game。

### 2.1 为什么 GAN framework 适合 black-box distillation

GAN (Goodfellow et al. 2014) 的原始 formulation:

$$\min_G \max_D V(G, D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

GAN 的美妙之处在于: **discriminator 只需要 samples，不需要 access 任何 distribution 的 analytical form**。这恰好契合 black-box distillation 的需求 —— 我们有 teacher 的 samples $y_t$，student 也可以 generate 自己的 samples $G(x)$。

但是 GAN 直接用于 discrete text generation 有问题: 因为 sampling operation $G(x)$ 是 non-differentiable 的 (text 是 discrete tokens)，无法用 standard backprop 训练 generator。这就需要 RL。

Reference:
- Original GAN paper: https://arxiv.org/abs/1406.2661
- SeqGAN (用 RL 训练 text GAN): https://arxiv.org/abs/1609.05473

### 2.2 GAD 的 Minimax Objective

GAD 的 value function:

$$\max_G \min_D \mathcal{V}(G, D) = \mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ -\log \sigma\left(D(y_t) - D(G(x))\right) \right] \quad (1)$$

让我详细解析每个符号:
- $G$: generator，即 student LLM, parameters 为 $\theta$
- $D$: discriminator, 接受 $[x, y]$ 作为 input，output 一个 sequence-level scalar score
- $\sigma(\cdot)$: sigmoid function, $\sigma(z) = \frac{1}{1 + e^{-z}}$
- $y_t \sim p(\cdot | x)$: teacher 的 response sample
- $G(x)$: student 自己采样的 response
- $\mathcal{T} = \{(x, y_t)\}$: 训练集，prompt 和 teacher response 对

这个 objective 其实就是 **Bradley-Terry model** 的 pairwise preference loss。Bradley-Terry model 假设: 对于两个 items $i$ 和 $j$，$i$ 比 $j$ 好的概率是:

$$P(i \succ j) = \frac{e^{s_i}}{e^{s_i} + e^{s_j}} = \sigma(s_i - s_j)$$

这里 $s_i = D(y_i)$ 是 discriminator 给的 score。所以 Equation (1) 是最大化 teacher 被判为优于 student 的 log-likelihood，从 discriminator 角度就是最小化 negative log-likelihood。

**Intuition building:** Discriminator 学习区分 "这个 response 是 teacher 写的还是 student 写的"，类似 GAN discriminator 区分 real/fake samples。Generator (student) 学习 generate 让 discriminator 无法区分的 responses，也就是 mimic teacher 的 distribution。

### 2.3 Generator 和 Discriminator 的分开优化

由于 text sampling 不可微，generator 用 RL 训练:

**Generator objective:**
$$\max_G \mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ D(G(x)) \right] \quad (2)$$

这里 $D(G(x))$ 被当作 reward。Generator 试图最大化 discriminator 给的 score。

**Discriminator objective:**
$$\min_D \mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ -\log \sigma\left(D(y_t) - D(G(x))\right) \right] \quad (3)$$

Discriminator 试图正确区分 teacher 和 student responses。

---

## 3. 与 RLHF 的深刻联系

这是这篇 paper 最 elegant 的部分。作者把 GAD 重新 interpret 为 **on-policy reward modeling**:

| RL Concept | GAD 中对应 |
|-----------|-----------|
| Policy model $\pi_\theta$ | Generator (student LLM) |
| Reward model $r_\phi$ | Discriminator $D$ |
| Reward function | $D(G(x))$ from Eq. (2) |
| Preference data | $(y_t, G(x))$ pairs |

**关键差异:** 传统 RLHF (Ouyang et al. 2022) 先在 static preference data 上训练 reward model，然后 freeze 它再 optimize policy。这导致 **reward hacking** 问题: policy 会找到 reward model 的 blind spots 并 exploit 它们。

GAD 的 discriminator 是 **co-evolving** 的: 它持续 update 以追踪 student 的变化。如果 student 在某个 mode 上变得太强 (开始 hack)，discriminator 会立刻学会区分这个 mode，从而 close the loop。这就是 on-policy reward modeling 的本质。

Reference:
- RLHF (InstructGPT): https://arxiv.org/abs/2203.02155
- Reward hacking: https://arxiv.org/abs/2111.06887
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300

---

## 4. GRPO 实现细节

作者用 GRPO (Group Relative Policy Optimization) 训练 generator。让我详细推导:

对于 prompt $x$，sample 一个 group of $N$ student responses $\{y_s^i\}_{i=1}^N$，每个 response 的 reward 是:

$$r_s^i = D(y_s^i) \quad (5)$$

Advantage 用 group statistics 归一化:

$$A^i = \frac{r_s^i - \text{mean}(\{r_s^j\}_{j=1}^N)}{\text{std}(\{r_s^j\}_{j=1}^N)} \quad (6)$$

这里:
- $A^i$: 第 $i$ 个 response 的 advantage
- $\text{mean}(\cdot)$: group 内 rewards 的均值
- $\text{std}(\cdot)$: group 内 rewards 的标准差

这种归一化是 GRPO 相比 PPO 的关键 trick: 不需要训练 value function $V(s)$ 作为 baseline，直接用 group statistics 当 baseline。这大幅减少了 memory footprint 和训练复杂度。

Generator objective (省略 KL regularizer 和 clip):

$$\max_G \mathbb{E}_{(x, y_t) \sim \mathcal{T}, \{y_s^i\}_{i=1}^N \sim q_G(\cdot|x)} \left[ \frac{1}{N} \sum_{i=1}^N A^i \right] \quad (7)$$

这里:
- $q_G(\cdot|x)$: generator 在 prompt $x$ 下的 output distribution
- $N$: group size (paper 中设为 8)
- 求和 $\sum_{i=1}^N$ 是对 group 内所有 responses 求平均

Discriminator 在 group setting 下的 Bradley-Terry loss:

$$\min_D \mathbb{E}_{(x, y_t) \sim \mathcal{T}, \{y_s^i\}_{i=1}^N \sim q_G(\cdot|x)} \left[ \frac{1}{N} \sum_{i=1}^N -\log \sigma(D(y_t) - D(y_s^i)) \right] \quad (8)$$

注意 $D(y_t)$ 在 group 内 shared (同一个 teacher response 对多个 student responses)，这提高了 efficiency。

---

## 5. Warmup 的重要性

作者发现 warmup 是 crucial 的。这有深刻的 intuition:

**问题:** 如果一开始 generator 太弱 (e.g., Qwen2.5-3B-Instruct vs GPT-5-Chat)，discriminator 会 trivially 区分它们 —— 给 teacher 高分，给 student 低分，loss 几乎为零。这意味着 gradient signal 很弱，adversarial game 无法启动。

**Warmup 策略:**
1. Generator 用 cross-entropy loss 在 teacher responses 上做 SFT (1 epoch)
2. Discriminator 用 Bradley-Terry loss 在同样 data 上训练 (1 epoch)
3. 然后开始 GAD training (2 epochs)

Ablation (Table 3) 显示:
- 去掉 generator warmup: LMSYS 49.7 vs GAD 50.8 (掉 1.1)
- 去掉 discriminator warmup: LMSYS 49.0 vs GAD 50.8 (掉 1.8)

Discriminator warmup 更重要，因为如果 discriminator 一开始太弱，它给的 reward signal 是 noisy 的，generator 学不到东西。

---

## 6. Mode-Seeking vs Mode-Covering 的 Toy Experiment

Figure 5 的 toy experiment 非常 illuminating。让我详细分析:

**Setup:**
- Teacher: discrete Gaussian mixture distribution over $\{0, 1, ..., 9\}$
- Student: single Gaussian distribution
- 只能 observe teacher 的 samples

**观察:**
- SeqKD student: mode-covering，spread probability mass across all 10 outputs
- GAD student: mode-seeking，concentrate probability on 一两个 reachable modes

**为什么 GAD 是 mode-seeking?**

这与 reverse KLD 的 mode-seeking 性质是一致的。考虑 GAN 的 optimal discriminator:

$$D^*(y) = \frac{p(y)}{p(y) + q(y)}$$

其中 $p$ 是 teacher distribution, $q$ 是 student distribution。Generator 的 objective 等价于最小化:

$$\text{JSD}(p \| q) \quad \text{(Jensen-Shannon Divergence)}$$

或者从另一个角度，generator 最大化 $\log D(G(x))$ 等价于最小化 reverse KL $D_{\text{KL}}(q \| p)$。Reverse KL 是 mode-seeking 的: 如果 teacher 在某区域概率为零，student 必须也在那区域概率为零 (否则 $q \log(q/p) \to \infty$)。所以 student 倾向于 concentrate 在 teacher 的高概率 mode 上。

**这对 LLM distillation 的意义:** Teacher LLM (e.g., GPT-5) 的 response space 是巨大的，student 无法 cover 所有 mode。Mode-seeking behavior 让 student focus 在自己 reachable 的 mode 上，这更 efficient 且避免生成低质量 responses。

Reference:
- GAN 与 KL divergence 的关系: https://arxiv.org/abs/1701.00160
- Mode-seeking GANs: https://arxiv.org/abs/1712.01879

---

## 7. 实验结果深度分析

### 7.1 Main Results (Table 2)

让我 extract 关键数字并 build intuition:

| Student | Method | LMSYS | Dolly | SelfInst | Vicuna |
|---------|--------|-------|-------|----------|--------|
| Qwen2.5-3B-I | Before | 45.8 | 45.1 | 45.6 | 47.3 |
| Qwen2.5-3B-I | SeqKD | 47.5 | 44.8 | 45.7 | 48.0 |
| Qwen2.5-3B-I | GAD | **48.9** | **46.7** | **47.7** | **49.4** |
| Qwen2.5-14B-I | Before | 50.0 | 49.1 | 49.4 | 50.0 |
| Qwen2.5-14B-I | SeqKD | 50.6 | 48.2 | 49.4 | 49.7 |
| Qwen2.5-14B-I | GAD | **52.1** | **50.4** | **51.1** | **51.6** |
| GPT-5-Chat | Teacher | 51.7 | 49.8 | 49.7 | 49.9 |

**关键观察:**

1. **Qwen2.5-14B-Instruct + GAD (52.1) 超越 GPT-5-Chat teacher (51.7) 在 LMSYS 上!** 这是 remarkable 的结果，说明 distillation 在某些 setting 下甚至能 surpass teacher。

2. **OOD generalization 差异巨大:** 在 Dolly, SelfInst, Vicuna (OOD) 上，SeqKD 经常比 Before Distillation 还差 (e.g., Qwen2.5-14B SeqKD 在 Dolly 上 48.2 < Before 49.1)。这是 SFT overfitting 的典型表现。GAD 则 robust 地 improve。

3. **Size scaling:** GAD 在 small models (3B) 上 gain 更大 (+3.1 on LMSYS)，在 large models (14B) 上 gain 较小 (+2.1)。这暗示 GAD 可能 help small models more。

### 7.2 为什么 SeqKD 在 OOD 上 hurt performance?

Figure 4 给出了答案: SeqKD student 的 N-gram overlap with teacher 更高，但 GPT-4o score 更低。这说明 SeqKD 在 **memorize local lexical patterns** (specific phrases, transitions)，而不是 learning the underlying generation policy。

这与近期 "SFT memorizes, RL generalizes" (Chu et al. 2025) 的发现一致。SFT 的 cross-entropy loss 鼓励 token-level matching，容易 overfit 到 spurious patterns。RL-based GAD 通过 reward signal 学习 global behavior，更 generalizable。

Reference: 
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161

### 7.3 On-policy vs Off-policy Discriminator (Figure 6)

这个实验是 paper 的 highlight。Setup:
- **Off-policy:** Student 先 SFT 1 epoch，freeze，train discriminator 2 epochs，然后 freeze discriminator，train student with RL
- **On-policy (GAD):** Warmup 1 epoch，然后 jointly train student 和 discriminator 2 epochs

**结果:** Off-policy discriminator 在 ~300 steps 后出现 severe reward hacking —— student 生成 1300 tokens 的冗长 responses 来 maximize discriminator score。On-policy GAD 在 thousands of steps 后仍然 stable。

**Intuition:** Off-policy discriminator 是在 old student distribution 上训练的。当 student 通过 RL 更新后，distribution shift 了，discriminator 的 reward function 不再 calibrated。Student 发现 "生成更长的 response" 这种 spurious pattern 能 hack 旧 discriminator。On-policy discriminator 持续 adapt，没有这种 staleness 问题。

### 7.4 Ablation Studies

**Discriminator loss (Table 4):**
- Bradley-Terry (default): 48.9 / 47.9
- Cross-entropy: 47.9 / 46.4

Cross-entropy loss:
$$\min_D \mathbb{E} \left[ -\log \sigma(D(y_t)) - \log(1 - \sigma(D(G(x)))) \right] \quad (4)$$

Bradley-Terry 更好，因为它只 model relative preference ($D(y_t) - D(G(x))$)，而不是 absolute scores。这提供了 invariance to score scale shifts，更 stable。

**Discriminator size (Table 5):**
- 3B student + 3B disc: 48.9 (best)
- 3B student + 7B disc: 47.8 (worse!)
- 7B student + 7B disc: 50.8 (best)
- 7B student + 14B disc: 50.5 (worse!)

**反直觉的发现:** Bigger discriminator 不是更好! 原因: 如果 discriminator 太强，它会 trivially 区分 teacher 和 student，gradient signal 消失。GAN 中也有类似问题 —— discriminator 太强会导致 generator gradient vanishing。Balanced G-D pair 是 key。

Reference:
- GAN training instability: https://arxiv.org/abs/1701.04862
- Wasserstein GAN (解决 GAN training 问题): https://arxiv.org/abs/1701.07875

---

## 8. Response Length Analysis (Table 6)

一个非常 subtle 的发现: SeqKD 倾向于缩短 response length (向 teacher length 靠拢)，而 GAD 保持 student original length。

| Model | Method | LMSYS Score | LMSYS Len |
|-------|--------|-------------|-----------|
| Qwen2.5-14B-I | Before | 50.0 | 322.1 |
| Qwen2.5-14B-I | SeqKD | 50.6 | 319.3 |
| Qwen2.5-14B-I | GAD | 52.1 | 438.9 |
| GPT-5-Chat | Teacher | 51.7 | 329.1 |

GAD 的 response 更长 (438.9 vs teacher 329.1)，但 score 更高。这说明 GAD 学到的是 teacher 的 **stylistic characteristics** (reasoning style, helpfulness patterns)，而不是 surface-level length matching。

这与 on-policy sampling 有关: student 从自己的 prior 出发，用 teacher 的 guidance 调整，所以保留了 student 的 length distribution。

---

## 9. 与其他工作的联系与联想

### 9.1 与 GAN 的历史脉络

GAD 是 SeqGAN (Yu et al. 2017) 在 LLM 时代的复兴。SeqGAN 用 REINFORCE 训练 text GAN，但当时 model 规模小，效果有限。GAD 的创新:
1. 用 Bradley-Terry loss 替代 binary cross-entropy (更 stable)
2. 用 GRPO 替代 vanilla REINFORCE (variance reduction)
3. Warmup strategy
4. On-policy discriminator (解决 reward hacking)

### 9.2 与 DPO 的关系

DPO (Rafailov et al. 2023) 也是用 Bradley-Terry model，但 DPO 是 closed-form solution for RLHF，不需要训练 discriminator。GAD 的 Bradley-Terry 是 on teacher-student pairs，而不是 human preference pairs。

潜在联想: 能否设计一个 DPO-like 的 GAD variant，避免显式训练 discriminator? 可能需要用 student 自身作为 implicit discriminator，但这会失去 on-policy reward modeling 的好处。

Reference: DPO paper: https://arxiv.org/abs/2305.18290

### 9.3 与 Constitutional AI 的联系

Anthropic 的 Constitutional AI (Bai et al. 2022) 也用 model 作为 feedback source (RLAIF)。GAD 的 discriminator 可以看作一个 specialized RLAIF model，focused on teacher imitation 而非 safety/helpfulness。

Reference: Constitutional AI: https://arxiv.org/abs/2212.08073

### 9.4 与 AlphaGo 的 Self-Play 类比

GAD 的 minimax game 与 AlphaGo 的 self-play 有精神上的相似:
- AlphaGo: policy network vs value network, 互相 improve
- GAD: generator vs discriminator, 互相 improve

但 GAD 不是 self-play (因为 teacher 是 fixed external model)，而是 **adversarial imitation learning**。Discriminator 学习 teacher 的 "signature"，generator 学习 mimic 这个 signature。

### 9.5 与 Inverse Reinforcement Learning

Ho & Ermon (2016) 的 Generative Adversarial Imitation Learning (GAIL) 是 GAD 的理论先驱。GAIL 用 GAN framework 做 imitation learning: expert demonstrations 作为 "real" samples，agent trajectories 作为 "fake" samples。GAD 本质上是 GAIL 在 LLM distillation 上的应用，加上 on-policy reward modeling 的 twist。

Reference: GAIL: https://arxiv.org/abs/1606.03476

### 9.6 与 Qwen3 / DeepSeek-R1 的关系

近期 Qwen3 (Yang et al. 2025) 和 DeepSeek-R1 (Guo et al. 2025) 都大量使用 RL 来 improve reasoning。GAD 的成功暗示: 未来 LLM training 可能是 **multi-stage adversarial distillation**，不同 capability 用不同 discriminator (reasoning discriminator, coding discriminator, etc.)。

Reference:
- Qwen3: https://arxiv.org/abs/2505.09388
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

---

## 10. 局限性与未来方向

基于我对 paper 的理解，几个潜在的 limitations:

1. **Compute cost:** GAD 需要 generate $N=8$ responses per prompt (GRPO group size)，加上 discriminator forward/backward。比 SeqKD expensive 大约 8-10x。Paper 提到 16x H100 训练 30 hours for 14B model。

2. **Reward hacking 的 subtler forms:** 虽然 on-policy discriminator 解决了 length hacking，但可能存在更 subtle 的 hacking (e.g., 模仿 teacher 的特定 phrase patterns 而非 semantic content)。Paper 没有深入分析。

3. **Mode collapse 风险:** GAN 经典问题。如果 discriminator 过于 focus 某些 mode，generator 可能 collapse 到这些 mode。Paper 没有讨论 diversity metrics。

4. **Teacher quality 的 dependency:** 如果 teacher (GPT-5-Chat) 本身有 biases，GAD 会 amplify 这些 biases，因为 student 直接模仿 teacher 的 distribution。

5. **Evaluation 的 circularity:** GPT-4o 作为 evaluator，而 teacher 是 GPT-5 (同 family)。可能存在 evaluator bias 倾向于 OpenAI-style responses。

**未来方向猜想:**
- Multi-teacher GAD: 多个 teacher 的 responses 作为 positive samples，可能 improve diversity
- Hierarchical discriminator: token-level + sequence-level discrimination
- Combining GAD with reasoning chain distillation (e.g., distill GPT-5's CoT)
- Theoretical analysis: GAD 收敛到什么 distribution? 是 JSD 还是其他 divergence?

---

## 11. 总结

GAD 是一个 conceptually clean 的方法，把 GAN + RL + imitation learning 融合到 LLM distillation 中。它的核心 contribution:

1. **On-policy black-box distillation:** 解决了 SeqKD 的 off-policy limitation
2. **Adaptive reward modeling:** 解决了 RLHF 的 reward hacking 问题
3. **Mode-seeking behavior:** 比 SeqKD 的 mode-covering 更适合 LLM distillation
4. **Empirical success:** 14B student 追平 GPT-5 teacher

从 Karpathy 的视角，这篇 paper 让我想到你在 "State of GPT" 演讲中提到的 RLHF 的 "two-model" setup (reward model + policy model)。GAD 把这两个 model 变成 adversarial 的，这可能是未来 post-training 的一个 promising direction。如果 reward model 能持续 adapt to policy，我们可能不需要担心 reward hacking 这个 "alignment tax"。

另外，GAD 的成功进一步 validate 了 "SFT memorizes, RL generalizes" 这个 thesis。在 LLM 时代，RL 不只用于 alignment，更是 knowledge transfer 的 powerful tool。

**Project & Code:**
- Project page: https://aka.ms/GAD-project
- Code: https://aka.ms/GAD-github

这篇 paper 让我感到兴奋的点是: 它把 GAN 这个 "old" idea 在 LLM scale 上 revive 了，并且用 on-policy reward modeling 解决了 GAN training 的 stability 问题。这暗示许多 "经典" deep learning ideas 可能在 LLM 时代有 second life，只是需要正确的 instantiation。
