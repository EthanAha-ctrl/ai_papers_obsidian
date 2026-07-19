---
source_pdf: Black-Box On-Policy Distillation of.pdf
paper_sha256: 907883e9eca2b6a1844d8ab376acd0921b549e199745162200f9b1f56bc2a0ef
processed_at: '2026-07-18T19:38:46-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Black-Box On-Policy Distillation via GAD — 深度技术讲解

## 1. 问题动机与背景

这篇 paper 来自 Microsoft Research (Tianzhu Ye, Li Dong, Furu Wei 等)，核心解决一个问题：**当我们只能 access 闭源 teacher LLM (如 GPT-5) 的 text output，无法 access logits 或 hidden states 时，如何做有效的 knowledge distillation？**

### 1.1 三种 distillation 设定的对比

| 设定 | 可用信号 | 典型方法 |
|------|---------|---------|
| White-box | logits, hidden states, attention | Forward KLD, Reverse KLD, hidden state matching |
| Black-box + off-policy | teacher text (fixed dataset) | SeqKD (SFT on teacher responses) |
| Black-box + on-policy | teacher text + student 自采样 | **GAD (本文)** |

### 1.2 为什么 on-policy 重要？

White-box distillation 的 recent 研究（如 MiniLLM [GDWH24], On-Policy Distillation [LL25], Agarwal et al. [AVZ⁺24]）揭示了 on-policy learning 的关键优势：student 从自己的 generated responses 中学习，可以:
- 实现 **mode-seeking behavior** (reverse KLD)，避免覆盖 teacher 所有 modes
- 减少 **exposure bias** (teacher-forcing 训练导致 inference 时 distribution mismatch)
- 避免 SFT 的 memorization 问题

但这些方法依赖 teacher 的 logits 来计算 reverse KLD。在 black-box 设置下，student 自采样的 response 没有 probability-level supervision 信号来评估质量。GAD 用一个 **adaptive discriminator** 来提供这个 implicit feedback signal。

参考链接:
- MiniLLM: https://arxiv.org/abs/2306.08543
- On-Policy Distillation (Thinking Machines Lab): https://thinkingmachines.ai/blog/on-policy-distillation
- Agarwal et al. On-policy distillation: https://arxiv.org/abs/2306.13649

---

## 2. GAD 方法详解

### 2.1 整体框架

GAD 将 distillation 形式化为一个 **two-player minimax game**，类似 GAN:
- **Generator G** = student LLM, 接收 prompt $x$，生成 response $G(x)$
- **Discriminator D** = 一个 reward model, 接收 $(x, y)$，输出 sequence-level scalar score $D([x,y])$

### 2.2 核心目标函数

minimax value function:

$$
\max_G \min_D \mathcal{V}(G, D) = \mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ -\log \sigma\left( D(y_t) - D(G(x)) \right) \right]
$$

**变量解析**:
- $G$: generator (student LLM), 参数为 $\theta$
- $D$: discriminator, 参数为 $\phi$
- $x$: prompt, 从训练集 $\mathcal{T}$ 采样
- $y_t$: teacher 对 $x$ 的 response
- $G(x)$: student 对 $x$ 的 generated response
- $D(y_t)$: discriminator 对 teacher response 的 scalar score (省略 $x$ 简写)
- $D(G(x))$: discriminator 对 student response 的 scalar score
- $\sigma(\cdot)$: sigmoid 函数, $\sigma(z) = 1/(1+e^{-z})$
- $\mathcal{T} = \{(x, y_t)\}$: 训练集, 包含 prompt 和对应的 teacher response

**关键观察**: 这是 **Bradley-Terry pairwise preference loss**。它学的是相对评分 $D(y_t) - D(y_s)$, 而非绝对二分类标签。这与传统 GAN discriminator 用 binary cross-entropy 不同 (详见 ablation 4.3)。

### 2.3 Generator 的优化

$$
\max_G \mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ D(G(x)) \right]
$$

由于 $G(x)$ 中的 sampling 操作对 $\theta$ 不可微, 采用 **policy gradient**。本文用 GRPO (Group Relative Policy Optimization, from DeepSeekMath [SWZ⁺24])。

### 2.4 Discriminator 的优化

$$
\min_D \mathbb{E}_{(x, y_t) \sim \mathcal{T}} \left[ -\log \sigma\left( D(y_t) - D(G(x)) \right) \right]
$$

Discriminator 用 **Bradley-Terry loss** 学习 "teacher response 应该比 student response 得分高"。

### 2.5 Discriminator 架构

Discriminator 初始化时复用 generator 的参数, 加一个额外的 **scalar prediction head**:
- Input: $[x, y]$ 拼接
- Forward: 经过 LLM backbone 得到每个 token 的 hidden state
- Output: 取 **最后一个 token** 的 hidden state, 通过线性层映射到 scalar score

这种设计类似 RLHF 中的 reward model, 但关键差异在于: RLHF 的 reward model 是 frozen 的, 而 GAD 的 discriminator 是 **online updated**。

参考链接:
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- GAN original paper: https://arxiv.org/abs/1406.2661
- SeqGAN (early text GAN): https://arxiv.org/abs/1609.05473

---

## 3. GRPO 实现细节

### 3.1 GRPO 公式

对每个 prompt $x$, 采样一组 N 个 student responses $\{y_s^i\}_{i=1}^{N}$, 每个的 reward 为:

$$
r_s^i = D(y_s^i) \tag{5}
$$

Advantage 计算 (group-relative, **没有 value function**):

$$
A^i = \frac{r_s^i - \mathrm{mean}(\{r_s^j\}_{j=1}^{N})}{\mathrm{std}(\{r_s^j\}_{j=1}^{N})} \tag{6}
$$

**变量解析**:
- 上标 $i$: group 中第 $i$ 个 response 的索引
- 下标 $s$: 表示 student-generated
- $r_s^i$: 第 $i$ 个 student response 的 reward (来自 discriminator)
- $\mathrm{mean}(\cdot)$, $\mathrm{std}(\cdot)$: group 内的均值和标准差
- $A^i$: normalized advantage, 衡量该 response 相对于 group 平均的好坏程度

Generator 的最终目标:

$$
\max_G \mathbb{E}_{(x, y_t) \sim \mathcal{T}, \{y_s^i\}_{i=1}^{N} \sim q_G(\cdot|x)} \left[ \frac{1}{N} \sum_{i=1}^{N} A^i \right] \tag{7}
$$

(实际实现中还包含 KL regularizer 和 PPO clip operator, 论文中省略)

### 3.2 Discriminator 的 GRPO 适配

对每个 group 中的 $y_s^i$, 与同一个 $y_t$ 配对, 形成 N 个 preference pairs, 用 Bradley-Terry loss 训练:

$$
\min_D \mathbb{E}_{(x, y_t) \sim \mathcal{T}, \{y_s^i\}_{i=1}^{N} \sim q_G(\cdot|x)} \left[ \frac{1}{N} \sum_{i=1}^{N} -\log \sigma(D(y_t) - D(y_s^i)) \right] \tag{8}
$$

注意 $D(y_t)$ 在 group 内是 **共享的**, 即同一个 teacher response 被比较 N 次。

### 3.3 训练超参数

| 参数 | 值 |
|------|-----|
| Group size $N$ | 8 |
| KL weight $\beta$ | 0.001 |
| Learning rate (GAD with GPT-5 teacher) | 1e-6 (warmup + GAD) |
| Learning rate (GAD with Qwen2.5 teacher) | 5e-6 warmup, 1e-6 GAD |
| Batch size | 256 |
| PPO mini-batch size | 256 |
| Total optimization steps | ~2400 |
| Max context (prompt) | 2048 tokens |
| Max context (response) | 1536 tokens |
| Training temperature | 0.8 |
| Checkpoint frequency | Every 50 steps |
| 16x H100 GPUs, 14B model | ~30 hours |

---

## 4. 训练算法 (两阶段)

### 4.1 Warmup Stage (1 epoch)

**关键发现**: warmup 对最终性能至关重要, 论文做了 ablation (Table 3)。

- **Generator warmup**: 在 teacher responses $y_t$ 上做 cross-entropy SFT
  - 作用: 缩小 teacher-student 之间的 distributional gap, 否则 discriminator 在早期太容易区分, 信号退化
- **Discriminator warmup**: 用同样的数据, 用 Bradley-Terry loss (Equation 3) 训练
  - 作用: 让 discriminator 一开始就有基本的判别能力, 否则 generator 收到的 reward 信号是 noise

具体策略: 在 warmup 阶段, 先单独训 discriminator 10 步, 然后开始 joint training。

### 4.2 GAD Training Stage (2 epochs)

```
for each batch (x, y_t) ~ T:
    1. Sample student responses G(x)  # on-policy sampling
    2. Update generator G using D(G(x)) as reward (GRPO)
    3. Update discriminator D with Bradley-Terry loss on (y_t, G(x)) pairs
```

### 4.3 Warmup Ablation (Table 3, Qwen2.5-7B)

| 配置 | LMSYS | Others (avg) |
|------|-------|--------------|
| SeqKD baseline | 49.2 | 48.3 |
| **GAD (full)** | **50.8** | **50.0** |
| w/o Generator warmup | 49.7 | 49.7 |
| w/o Discriminator warmup | 49.0 | 47.7 |

去掉 discriminator warmup 影响更大, 因为没有有效的 reward signal, adversarial interaction 失效, generator 几乎无法超越 warmup 性能。

---

## 5. 实验结果

### 5.1 主实验 (Table 2, GPT-4o scores)

| Student Model | Method | LMSYS | Dolly | SelfInst | Vicuna |
|---------------|--------|-------|-------|----------|--------|
| GPT-5-Chat | Teacher | 51.7 | 49.8 | 49.7 | 49.9 |
| Qwen2.5-3B-I | Before | 45.8 | 45.1 | 45.6 | 47.3 |
|  | SeqKD | 47.5 | 44.8 | 45.7 | 48.0 |
|  | **GAD** | **48.9** | **46.7** | **47.7** | **49.4** |
| Qwen2.5-7B-I | Before | 48.7 | 47.6 | 48.3 | 49.1 |
|  | SeqKD | 49.2 | 47.2 | 48.3 | 49.5 |
|  | **GAD** | **50.8** | **48.5** | **50.1** | **51.4** |
| Qwen2.5-14B-I | Before | 50.0 | 49.1 | 49.4 | 50.0 |
|  | SeqKD | 50.6 | 48.2 | 49.4 | 49.7 |
|  | **GAD** | **52.1** | **50.4** | **51.1** | **51.6** |
| Llama-3.2-3B-I | Before | 44.0 | 45.8 | 47.0 | 46.9 |
|  | SeqKD | 47.6 | 47.0 | 47.1 | 48.1 |
|  | **GAD** | **48.1** | **48.5** | **49.1** | **48.9** |
| Llama-3.1-8B-I | Before | 46.9 | 46.6 | 48.4 | 47.9 |
|  | SeqKD | 49.7 | 47.7 | 48.7 | 48.7 |
|  | **GAD** | **50.3** | **48.8** | **49.5** | **50.2** |

**关键发现**:
- Qwen2.5-14B + GAD (52.1) **超过** GPT-5-Chat teacher (51.7) on LMSYS
- OOD datasets (Dolly, SelfInst, Vicuna) 上 GAD 提升尤其明显
- **SeqKD 在 OOD 上经常出现 negative transfer** (如 Qwen2.5-14B on Dolly: 49.1 → 48.2)

### 5.2 Scaling Pattern

从 Figure 1 可以看到一个有趣的 "size compensation" 现象:
- Qwen2.5-3B + GAD ≈ Qwen2.5-7B + SeqKD
- Qwen2.5-7B + GAD ≈ Qwen2.5-14B + SeqKD
- Qwen2.5-14B + GAD ≈ GPT-5-Chat teacher

即 GAD 大致能让 student "升一级"。

### 5.3 Response Length 分析 (Table 6)

| Model | Method | LMSYS Score | LMSYS Len | Dolly Score | Dolly Len |
|-------|--------|-------------|-----------|-------------|-----------|
| GPT-5-Chat | Teacher | 51.7 | 329.1 | 49.8 | 148.5 |
| Qwen2.5-7B | SeqKD | 49.2 | 320.2 | 47.2 | 152.3 |
| Qwen2.5-7B | GAD | 50.8 | 414.0 | 48.5 | 225.1 |

**重要观察**: SeqKD 倾向于 mimic teacher 的 length distribution (320.2 vs 329.1), 而 GAD 保持 student 原始的 length distribution (414.0)。这说明 GAD 学到的是 teacher 的 **global stylistic characteristics**, 而非 surface-level 模仿。这一点也回应了 "SFT memorizes, RL generalizes" [CZY⁺25] 的观点。

参考链接:
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- LMSYS-Chat-1M dataset: https://arxiv.org/abs/2309.11998

---

## 6. 深入分析

### 6.1 SeqKD 过拟合 Local Patterns (Figure 4)

论文用 **N-gram F1 overlap** 衡量 student 和 teacher 的 surface-level 相似度:

- SeqKD student: 高 N-gram overlap, 低 GPT-4o score
- GAD student: 低 N-gram overlap, 高 GPT-4o score

**Intuition**: SFT 的 maximum likelihood objective 会迫使 student 在 token 级别精确复现 teacher 的 lexical patterns, 这容易导致 memorization 而非真正的 capability acquisition。GAD 只通过 sequence-level reward signal 监督, 给 student 更多自由度去探索语义等价但表达不同的 response。

### 6.2 Mode-Seeking vs Mode-Covering (Figure 5, Toy Experiment)

这是一个非常 illuminating 的 toy experiment。设置:
- Teacher: 离散 Gaussian mixture, 10 个 categorical outputs $\{0, ..., 9\}$
- Student: 单个 Gaussian distribution
- 信号: 只能观察 teacher 的 output samples (black-box)

结果:
- **SeqKD (mode-covering)**: 把 probability mass 分散到所有 10 个 modes, 试图覆盖 teacher 的整个 support
- **GAD (mode-seeking)**: 集中 probability mass 到 reachable modes (最显著的几个)

**为什么 mode-seeking 对 LLM distillation 更好?** 因为小 model 的 capacity 有限, 试图覆盖 teacher 所有 modes 会导致每个 mode 都做不好。Mode-seeking 让 student 在它能做好的区域达到 teacher 水平, 这对应 forward KLD vs reverse KLD 的经典分析 (MiniLLM [GDWH24] 有详细论证)。

### 6.3 On-Policy vs Off-Policy Discriminator (Figure 6) — 最关键的实验

这是论文最有说服力的实验之一, 直接证明了 on-policy 的必要性。

**Off-policy 设置**:
1. Student 先 warmup 1 epoch (SeqKD)
2. Freeze student, 训 discriminator 2 epochs on student outputs
3. Freeze discriminator 作为 fixed reward model
4. 用这个 frozen reward model 训 student (Equation 7)

**结果**: Off-policy discriminator 在 ~300 training steps 后出现严重 **reward hacking**:
- Response length 飙升到 1300+ tokens
- 偏离 teacher pattern
- 性能下降

**On-policy (GAD)**: 数千 steps 稳定, 无 reward hacking 迹象。

**Intuition**: Fixed reward model 容易被 student 通过 out-of-distribution outputs "钻空子"。Discriminator 持续 co-evolve, 始终 "盯着" student 当前行为, 形成动态对抗 equilibrium。这与 Skalse et al. [SHKK22] 对 reward gaming 的理论分析一致。

参考链接:
- Reward gaming characterization: https://arxiv.org/abs/2201.03544
- Reward hacking in RLHF: https://openai.com/research/learning-from-human-preferences

### 6.4 Discriminator Loss Ablation (Table 4, Qwen2.5-3B)

| Discriminator Loss | LMSYS | Others |
|--------------------|-------|--------|
| SeqKD baseline | 47.5 | 46.2 |
| **Bradley-Terry (default)** | **48.9** | **47.9** |
| Cross-entropy (Eq. 4) | 47.9 | 46.4 |

Cross-entropy loss (传统 GAN discriminator 用的):

$$
\min_D \mathbb{E} \left[ -\log \sigma(D(y_t)) - \log(1 - \sigma(D(G(x)))) \right] \tag{4}
$$

**为什么 BT loss 更好?** BT loss 只学 **相对 ordering** ($D(y_t) > D(y_s)$), 不强制 absolute score 收敛到 0/1。这给 discriminator 更多 flexibility, 也更稳定 (避免 saturation 后 gradient 消失)。这也是 RLHF reward modeling 的标准实践。

### 6.5 Discriminator Size Ablation (Table 5)

| Gen Size | Disc Size | LMSYS | Others |
|----------|-----------|-------|--------|
| 3B | 3B (default) | **48.9** | **47.9** |
| 3B | 7B | 47.8 | 46.9 |
| 7B | 7B (default) | **50.8** | **50.0** |
| 7B | 14B | 50.5 | 49.9 |

**反直觉发现**: 把 discriminator 变大反而 hurt performance。这与 GAN 的经验一致 — discriminator 过强会导致 generator gradient signal 退化 (discriminator 总是 win, 没有学习信号)。**平衡的 generator-discriminator pair 至关重要**。

---

## 7. Tokenizer 不兼容场景 (Table 7)

这是一个实用的 extension。当 teacher (Qwen2.5-14B) 和 student (Llama-3.x) 的 tokenizer 不兼容时, 无法直接做 logits-level KLD alignment。GAD 因为只用 text outputs, 天然适用:

| Student | Method | LMSYS | Dolly | SelfInst | Vicuna |
|---------|--------|-------|-------|----------|--------|
| Llama-3.2-3B | SeqKD | 46.9 | 47.6 | 47.6 | 48.5 |
| Llama-3.2-3B | GAD | **47.5** | **47.7** | 47.3 | **49.0** |
| Llama-3.1-8B | SeqKD | 49.0 | 48.4 | 48.6 | 49.4 |
| Llama-3.1-8B | GAD | **49.6** | **49.9** | **50.5** | **49.7** |

---

## 8. 与相关工作的关系

### 8.1 GAN for Text Generation 历史
- **SeqGAN** [YZWY17]: 早期 text GAN, 用 policy gradient 训 generator
- **GAD 的差异**: (1) conditional generation 而非 unconditional; (2) Bradley-Terry 而非 BCE; (3) 现代 LLM scale; (4) distillation 语境, teacher 提供 reference distribution

### 8.2 与 RLHF 的对比 (Table 1)

| 维度 | RLHF | GAD |
|------|------|-----|
| Policy model | Student LLM | Generator (= student LLM) |
| Reward model | Trained on human preferences, then frozen | Discriminator, **online updated** |
| Reward signal | Scalar from frozen RM | $D(G(x))$ from co-evolving D |
| Preference data | Human-annotated pairs | Auto-constructed (teacher vs student) |
| Risk | Reward hacking from frozen RM | Mitigated via co-evolution |

### 8.3 与 DPO 的潜在关系
GAD 的 Bradley-Terry loss 与 DPO 形式上类似, 但 GAD 是 **online**: preference pairs 是 on-policy 生成的 (student 当前 samples vs teacher), 而非来自静态 dataset。可以视为一种 "online DPO with adversarial teacher"。

参考链接:
- DPO: https://arxiv.org/abs/2305.18290
- GAIL (Generative Adversarial Imitation Learning): https://arxiv.org/abs/1606.03476
- verl RL framework: https://arxiv.org/abs/2409.19256

---

## 9. 我的 Intuition 总结与开放问题

### 9.1 为什么 GAD work? 三个核心机制:

1. **On-policy exploration**: Student 从自己的 generation 中学习, 避免 exposure bias, 这是 SFT 的根本局限。
2. **Adaptive reward**: Discriminator co-evolve, 避免 frozen RM 的 reward hacking。这是一个 self-supervised 的 curriculum — 随着学生变强, 难度自动提升。
3. **Mode-seeking via relative scoring**: BT loss + group-relative advantage (GRPO) 只学相对 ordering, 给 student 语义层面的自由度, 而非强迫 token-level 模仿。

### 9.2 可能的局限和 open questions

1. **Compute cost**: 每个 step 需要 sample N=8 responses + forward discriminator, 比 SeqKD 贵 ~8-10x。30 GPU-hours for 14B 已经是显著开销。
2. **Teacher response 复用**: 同一个 $y_t$ 在 group 内被比较 N 次 (Equation 8), 可能存在效率问题 — 是否可以用多个 teacher samples?
3. **Discriminator collapse 风险**: 论文没讨论 discriminator 是否会最终完全 fail to distinguish (即 $D(y_t) \approx D(y_s)$)。在 GAN 中这是 mode collapse 的前兆。
4. **Reward hacking 的 long-term stability**: Figure 6 只展示了数千 steps, 但更长的训练 (如 10K+ steps) 是否依然稳定?
5. **与 reasoning tasks 的适配**: 论文聚焦 chat / instruction following。对于有 verifiable reward 的 reasoning tasks (math, code), GAD 的优势可能减弱, 因为 ground-truth reward 已经可用。
6. **Multi-turn 对话**: 论文用 single-turn 设定。Multi-turn 场景下, discriminator 如何处理对话历史?
7. **理论 convergence analysis**: 缺乏 minimax game 的理论分析, 何时能保证 Nash equilibrium? GAN 的理论是否能直接迁移?

### 9.3 更广泛的联想

- **Self-play 的视角**: GAD 可以视为 student 和一个 "teacher-imitating critic" 的 self-play。这与 AlphaGo 的 self-play、Constitutional AI 的 critique-revise 有思想上的呼应。
- **Active learning**: On-policy sampling 本质上是 student 在主动选择 "自己不确定的区域" 探索, discriminator 提供反馈。这与 active learning 和 curiosity-driven RL 有联系。
- **Model merging / weak-to-strong generalization**: 当 student 接近或超过 teacher (如 Qwen2.5-14B + GAD > GPT-5 on LMSYS), 是否涉及 weak-to-strong generalization 现象? (student 学到了 teacher 的 implicit capability, 甚至在 teacher 表现一般的区域超越 teacher)
- **Connection to Inverse RL**: Discriminator 学的是 "区分 expert (teacher) 和 policy (student)"，这是 IRL (GAIL [HE16]) 的经典框架。GAD 可以视为 **batch IRL with a powerful teacher demonstration**。
- **Constitutional AI 替代**: 如果 discriminator 替换为 LLM-as-judge (如 GPT-4o), 是否能达到类似效果? 这会损失 on-policy co-evolution 的好处, 但可能提供更丰富的 feedback signal (natural language critique)。

参考链接:
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Weak-to-strong generalization: https://arxiv.org/abs/2312.09390
- GAIL: https://arxiv.org/abs/1606.03476
- Curiosity-driven exploration: https://arxiv.org/abs/1705.05363

---

## 10. 实用启示 (For practitioners)

如果你考虑用 GAD:

1. **先做 SeqKD warmup**: 不要直接从 base instruct model 开始 adversarial training。1 epoch SFT 是必要的前置。
2. **Discriminator 初始化**: 用 generator 参数 + 新 head, 保持大小一致。不要用更大的 discriminator。
3. **用 BT loss 而非 BCE**: 这是 RLHF reward modeling 的标准选择, 在 GAD 中也验证更优。
4. **监控 response length**: 这是 reward hacking 的早期 signal。如果 length 异常增长, 说明 reward 和真实质量脱钩。
5. **Group size N=8 是个合理起点**: 更大的 N 更稳定但更贵; 更小的 N 噪声大。
6. **KL regularizer weight $\beta=0.001$**: 很小的 KL penalty 防止 student 漂移过远。
7. **用 verl 或类似 RL framework**: 论文提到 verl [SZY⁺24] 作为实现基础, 可以复用。

这篇 paper 我认为是一个 solid 的 contribution: 它 elegantly 把 GAN 的思想 + RLHF 的 reward modeling + on-policy distillation 的洞察融合在一起, 解决了一个实际重要的问题 (black-box distillation)。Qwen2.5-14B + GAD 在 LMSYS 上超越 GPT-5 teacher 这个结果, 如果可复现, 是一个值得关注的 milestone。
