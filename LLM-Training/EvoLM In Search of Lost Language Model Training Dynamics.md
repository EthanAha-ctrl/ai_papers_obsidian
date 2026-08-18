---
source_pdf: EvoLM In Search of Lost Language Model Training Dynamics.pdf
paper_sha256: 15e5ebea59606b5e2a23358a9f17eac0d0c92bd4052db269ac0a99c05e6bea61
processed_at: '2026-08-18T11:32:30-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EvoLM 用人话版：把4个训练阶段的关系讲清楚

## 这篇paper到底在干嘛

简单说：现在大家训LM都是 Pre-training → CPT → SFT → RL 四步流水线，但是每一步到底贡献了多少、互相怎么影响、什么时候开始边际收益递减，没人说得清。之前的研究要么只看 pre-training loss，要么拿别人家的 base model 直接做 post-training，变量完全没控制住。

EvoLM 团队从 scratch 训了 100+ 个 1B/4B 模型，所有 run 都跑完整 LR schedule，所有 stage 的 data/code/model 全开源。本质上是把 LM training 的全生命周期做了一个**控制变量法的系统性实验**。

链接：[EvoLM HuggingFace](https://huggingface.co/EvoLM) ，相关讨论参考 [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) 和 [Pythia suite](https://proceedings.mlr.press/v202/biderman23a.html)。

---

## 12个takeaway 用大白话翻译

### Takeaway 1-2：Pre-training 训太多反而有害

**人话**：Chinchilla 说 1B 模型看 20B tokens 就够了，但大家都在疯狂 over-train (TinyLlama 看 2T，Llama 3.2 1B 看 9T)。EvoLM 发现，pre-training 在 80x-160x Chinchilla ratio (1B模型看80B-160B tokens) 之后，upstream accuracy 基本不动了，downstream 甚至开始掉。

特别有意思的是 Table 1：**1B-320BT 反而比 4B-80BT 强** (相同 compute)。直觉是：4B 模型每 token 的 flops 是 1B 的 4 倍，相同 compute 下 4B 只能看 1/4 的 tokens，还没到 saturation regime，model capacity 没被激活。只有当 tokens 也进了 4B 的 saturation regime (160B)，4B 才突然"开窍"，ID Maj@16 从 13.2% 跳到 26.4%。

公式化思考：经典 scaling law
$$L(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + L_\infty$$

其中 $N$ = 参数量，$D$ = token 数，$A, B$ 是常数，$\alpha \approx 0.34$, $\beta \approx 0.28$ (Chinchilla 估计)，$L_\infty$ 是不可压缩 entropy 下界。

Compute $C \approx 6ND$，在 $C$ 固定时增大 $N$ 必然减小 $D$。当 $D/N < 20$ (under-trained) 时，$D^{-\beta}$ 这一项 dominate，所以 4B-80BT 反而比 1B-320BT 差。只有 $D/N$ 也进入 saturation regime，$N^{-\alpha}$ 才开始 dominate。这就是 "model capacity 要在 saturation regime 才能被激活" 的数学根源。

参考：[Overtrained models harder to fine-tune (Springer et al., 2025)](https://arxiv.org/abs/2503.19206) , [Gadre et al. dual-axis scaling](https://arxiv.org/abs/2403.08540)。

### Takeaway 3：CPT 必须 replay，5% 是甜区

**人话**：你拿一个 general base model 直接在 math corpus 上做 CPT，会 catastrophic forgetting——general 知识忘光，math 也没学透。EvoLM 试了不同 replay 比例：

| Config | GSM8K Pass@1 |
|---|---|
| No CPT | 6.04 |
| 50B FineMath only | 19.27 |
| 1.6B replay + 48.4B math (3%) | 16.21 |
| **8B replay + 42B math (16%)** | **21.01** |
| 16B replay + 34B math (32%) | 15.22 |

5% replay 最优。太多 replay 稀释 math learning，太少 retain 不住 general 知识。

直觉模型：CPT 的 loss 是
$$\mathcal{L}_{CPT} = (1-\lambda)\mathcal{L}_{math} + \lambda\mathcal{L}_{web}$$

$\lambda$ 是 replay ratio。$\lambda = 0$ 纯 math forgetting 严重，$\lambda = 1$ 完全没 math adaptation。$\lambda \approx 0.05$ 是 Pareto frontier 上的 sweet spot，少量 general data 就能 anchor 住 distribution，大部分 gradient 仍流向 math。

参考：[D-CPT Law (Que et al.)](https://arxiv.org/abs/2407.10140) , [Ibrahim et al. continual pretraining](https://arxiv.org/abs/2403.08763) , [Bethune et al. forgetting scaling](https://arxiv.org/abs/2502.06042)。

### Takeaway 4-6：CPT 是 Pre-training 到 RL 的"桥"，没它 RL 会崩

**人话**：最反直觉的发现——**没有 CPT 时，RL 反而比 SFT 还差**。

Figure 5 里，CPT=0 的 SFT model 有一定 math 能力，但加上 RL 后 Maj@16, RM@16, Pass@16 全部下降。只有当 CPT tokens > 0 之后，RL 才开始稳定超越 SFT。

为什么？RL 靠 base model 自己 rollout。如果 base model 没有 math prior，rollout 里几乎没正确答案，binary reward 几乎都是 0，policy gradient 信号退化为噪声。CPT 注入 math prior 等于提升 rollout 中正确解的密度，让 RL 信号变得 informative。

PPO 的 advantage 估计
$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，$r_t$ 是 reward，$\gamma$ 是 discount factor，$\lambda$ 是 GAE parameter，$V$ 是 value function。

如果 $r_t$ 几乎都为 0 (math prior 不足)，$V$ 也学到 0，advantage 变成纯噪声，policy 更新方向随机。CPT 给 $r_t$ 注入非零概率，整个 RL 信号链才 work。

**这个 take away 对实践非常重要**：你想做 RL，base model 必须先有足够的 domain prior。直接拿 general base model 上 RL，大概率 collapse。

### Takeaway 7-8：SFT 多了反而限制 RL

**人话**：SFT epochs 多了，ID accuracy 上升 (memorization)，但 OOD 在 2-4 epochs 后开始掉。而且 SFT 过头之后，RL 的 marginal gain 急剧缩小。

Figure 6 数据 (1B-160BT-8+42BT base, 100K SFT examples)：
- SFT epochs 1 → 8: ID Maj@16 持续上升，~8 epochs saturation
- SFT epochs 2-4: OOD peak
- SFT epochs > 4: OOD 下降

为什么 RL gain 缩小？SFT 是在 base model 的 conditional distribution 上做 KL projection
$$\theta^* = \arg\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}_{SFT}}\left[-\log p_\theta(y|x)\right]$$

过多 epochs 让 distribution collapse 到 training set 的 narrow mode，entropy 被 squeeze 掉。RL 依赖 $\pi_{ref}(y|x)$ 保留的 entropy 做 exploration，distribution 太 sharp 时 exploration 空间被压缩。

社区经验 "SFT 3 epochs" ([LIMA, Zhou et al. 2023](https://arxiv.org/abs/2305.11206)) 跟这个发现完全一致。

### Takeaway 9-10：RL sharpen confidence，不创造 reasoning

**人话**：这是全文最深刻的发现。RL 训多了，Pass@16 (至少一个对的概率) 在 4 epochs 后开始**下降**，但 Correct Ratio (在有正确解的 group 里答对的频率) 持续上升。

这两个指标组合的含义：
- Pass@16 下降 = 模型能解决的问题总数在减少
- Correct Ratio 上升 = 在能解决的问题里，模型更自信了

翻译成人话：**RL 不是让模型学会新的推理能力，是把模型已有的正确输出的概率 mass 集中起来**。模型本来"偶尔能蒙对"，RL 让它"稳定蒙对"，但本来完全不会的题，RL 也救不了。

这印证了几个 concurrent work：
- [Yue et al. 2025: Does RL really incentivize reasoning?](https://arxiv.org/abs/2504.13837)
- [Zhao et al. Echo Chamber: RL amplifies pretraining patterns](https://arxiv.org/abs/2504.07912)
- [Chu et al. SFT memorizes, RL generalizes](https://arxiv.org/abs/2501.17161)

一个 mental model：pre-training + CPT 决定 ceiling (能解决多少题)，RL 决定 floor (实际能稳定解决多少题)。RL 把 distribution 推向 ceiling，但不能突破 ceiling。

### Takeaway 11：SFT/RL 数据分配是 zero-sum

**人话**：total 100K post-training examples 的 budget 下：
- ID accuracy 随 SFT 比例上升，70K SFT + 30K RL plateau
- OOD accuracy 在 10K SFT + 90K RL peak

直觉：SFT 提供 format + template learning，ID 任务直接 helpful；RL 提供 exploration-based generalization，OOD 更 helpful。10K SFT 就够 scaffolding format，剩下 90K 给 RL 探索。

实践含义：你做 math RL，SFT 不需要太多，关键是 RL compute 给足。

### Takeaway 12：ORM score 是靠谱的 proxy，PPL 不是

**人话**：post-training 阶段，validation perplexity 跟 downstream accuracy 几乎没相关性 (Pearson r ≈ 0)。但 ORM score (用 8B reward model 给 generation 打分) 与 Maj@16 accuracy 的 Pearson r 在 0.62-0.84 之间。

为什么 PPL 失效？post-training 把模型从 maximum likelihood 推向了 maximum reward / instruction following，loss landscape 已经偏离了 next-token cross-entropy 的 axis。PPL 定义
$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i)\right)$$
衡量的是 likelihood，但 post-trained model 的输出 distribution 已经被 SFT 和 RL 扭曲，不再对应 ground truth LM likelihood。

ORM 与 downstream metric 同分布，所以 correlation 高。**实践含义：post-training 阶段不要用 validation loss 监控，要用 ORM 或 task-specific reward**。这也是为什么 OpenAI/Anthropic 的 RLHF pipeline 都用 reward model 做 eval，不用 PPL。

---

## Bonus：Intermediate Checkpoint 是个坑

Table 3 比较了"训 20B tokens 的独立 run" vs "训 160B tokens 中第 20B 处的 checkpoint"：

| Model | Upstream | Math L1 Pass@16 | Math L2 Pass@16 |
|---|---|---|---|
| 20BT full | 46.43 | 17.85 | 15.10 |
| 20BT int. | 46.07 | 11.44 | 12.64 |
| 40BT full | 49.38 | 17.96 | 14.88 |
| 40BT int. | 49.06 | 9.38 | 8.72 |

Upstream 几乎一样，但 downstream Pass@16 差距巨大 (17.96 vs 9.38)。

原因：cosine LR schedule 的 annealing 阶段会改变 weight landscape 的几何 (sharper minima)，intermediate checkpoint 没经历 annealing，处在 flatter region，downstream probing 无法 effectively trigger。

**对社区警示**：任何拿 intermediate checkpoint 做 post-training study 的结论都要打折看。这影响一大批 paper，包括很多 scaling law 工作用的是 mid-training checkpoint。

参考：[Power Scheduler (Shen et al.)](https://arxiv.org/abs/2408.13359) , [LR annealing scaling law (Tissue et al.)](https://arxiv.org/abs/2408.11029)。

---

## 给你 (Andrej) 的 mental model 总结

1. **Pre-training saturation 是 $N$ 和 $D$ 双轴函数**：compute 在两轴间 trade off，小 model + 多 tokens 在没到大 model saturation regime 前能反超大 model + 少 tokens。

2. **CPT 是 bridge layer，不是简单的 domain adaptation**：没 CPT，RL 会 collapse；CPT 注入的 prior 是 RL exploration 的种子。5% replay 是 Pareto sweet spot。

3. **SFT memorizes ID，RL explores OOD**：fixed budget 下 zero-sum。SFT 3 epochs 就够，剩下全给 RL。

4. **RL redistributes probability mass，不创造 capability**：Pass@16 下降 + Correct Ratio 上升是铁证。RL ceiling 由 pre-training + CPT 决定。

5. **Post-training 不要用 PPL 监控**：用 ORM。PPL 在 post-training 阶段 miscalibrated。

6. **Intermediate checkpoint 不能替代 full run**：LR annealing 改变 weight 几何，对 downstream probing 至关重要。

这篇 paper 真正的价值在于**控制变量法的系统性**——把 4 个 stage 的 scaling study 在同一 framework 下跑完，变量全部 control 住。这种 end-to-end controlled study 在 literature 里非常稀缺，比 GPT-4/Claude technical report 里孤立 ablation 的信息密度高得多。

主要参考：
- [EvoLM HuggingFace](https://huggingface.co/EvoLM)
- [Chinchilla (Hoffmann et al.)](https://arxiv.org/abs/2203.15556)
- [D-CPT Law](https://arxiv.org/abs/2407.10140)
- [Springer et al. overtrained harder to fine-tune](https://arxiv.org/abs/2503.19206)
- [Yue et al. RL reasoning question](https://arxiv.org/abs/2504.13837)
- [Echo Chamber (Zhao et al.)](https://arxiv.org/abs/2504.07912)
- [SFT memorizes, RL generalizes (Chu et al.)](https://arxiv.org/abs/2501.17161)
- [LIMA (Zhou et al.)](https://arxiv.org/abs/2305.11206)
- [Skywork-Reward](https://arxiv.org/abs/2410.18451)
- [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

---

# EvoLM: 深入解析 LM Training Dynamics 跨阶段研究

## 1. Paper 高层定位与动机

这篇 EvoLM 来自 Harvard / Stanford / EPFL / CMU 的合作团队，核心动机非常直接：现代 LM 训练已经被切成 Pre-training → Continued Pre-training (CPT) → SFT → RL 四个相互纠缠的 stage，但下游开发者很难判断每个 stage 的设计选择究竟带来多少 marginal gain。已有 scaling law 工作大多只覆盖 pre-training loss vs compute，而忽略了：

1. **Training-inference mismatch**: autoregressive generation 下的 next-token prediction loss 与最终 problem-solving accuracy 之间是非 smooth、非线性关系 (参考 [Schaeffer et al., 2023](https://arxiv.org/abs/2304.15004))。
2. **Checkpoint confounding**: 大量研究基于长训练 run 中切出的 intermediate checkpoint，而 LR schedule 还没 anne化完，这些 checkpoint 实际上是 under-trained 的，会严重误导结论。
3. **Opaque base models**: post-training 研究往往直接拿 off-the-shelf base model，无法 control model size、pre-training data composition、token budget 这些关键变量。

EvoLM 的应对方案是从头训练 100+ 个 1B / 4B 模型，所有 run 都跑完整 LR schedule，并且完整开源 data / code / model。这本质上是一个**controlled experiment suite**，类似 Pythia ([Biderman et al., 2023](https://proceedings.mlr.press/v202/biderman23a.html)) 的精神，但延伸到了完整 post-training 链路。

Model suite 链接: [https://huggingface.co/EvoLM](https://huggingface.co/EvoLM) ；代码库 [https://github.com/hsrambo/evolm](https://github.com/hsrambo/evolm) (注：具体仓库 URL 可能是 hallucinated，建议查官方 release)。

---

## 2. 实验设置：四个阶段的完整 pipeline

### 2.1 模型架构

基于 LLaMA-2 decoder-only transformer，三个 size：

| Model | Hidden $d$ | Intermediate $d_{ff}$ | Layers $L$ | Heads $H$ | KV Groups | Context |
|---|---|---|---|---|---|---|
| 0.5B | 1536 | 3216 | 20 | 32 | 4 (GQA) | 2048 |
| 1B | 2048 | 4896 | 22 | 32 | 4 | 2048 |
| 4B | 4096 | 7792 | 28 | 32 | 4 | 2048 |

这里使用 **GQA (Grouped Query Attention)**，4 个 query group 对应 32 个 head，即每 8 个 query head 共享一对 KV head。这个比例与 LLaMA-2 一致，目的是 reduce KV cache 与 attention 计算的 memory bandwidth 压力。Transformer 的核心 attention 计算可以写为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

其中 $Q \in \mathbb{R}^{S \times H \times d_h}$，$K, V \in \mathbb{R}^{S \times G \times d_h}$ ($G$ 是 KV group 数)，$d_h = d/H = 64$ (1B 模型)，$S$ 是 sequence length。GQA 通过 broadcast $K, V$ 到 $H/G = 8$ 个 query head 上，减少 KV cache 至 MHA 的 $G/H = 1/8$。

### 2.2 四阶段训练协议

模型命名约定 `1B-160BT-8+42BT-100Kep1-100Kep16`：

- `1B`: 1B 参数
- `160BT`: pre-training 在 FineWeb-Edu 上看 160B tokens
- `8+42BT`: CPT 阶段 8B general-domain replay + 42B domain-specific (FineMath)
- `100Kep1`: SFT 在 100K examples 上跑 1 epoch
- `100Kep16`: RL 在 100K examples 上跑 16 epochs

**Pre-training**: FineWeb-Edu ([Penedo et al., 2024](https://arxiv.org/abs/2501.13835))，~1.3T tokens 总池。Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) 推荐 ~20 tokens/parameter 为 compute-optimal。本文研究：
- Mild over-training: 1x–16x Chinchilla (e.g., 1B 模型看 20B–320B tokens)
- Excessive over-training: >16x Chinchilla (e.g., 1B 看 320B 是 320x)

**CPT**: FineMath ([SmolLM2, Allal et al.](https://arxiv.org/abs/2502.07298)) ~50B token 池，token budget 0–50B。关键 trick 是 **replay** —— 在 FineMath 中随机 interleave 一部分 FineWeb-Edu，以缓解 catastrophic forgetting (参考 [Ibrahim et al., 2024](https://arxiv.org/abs/2403.08763) 和 [D-CPT Law, Que et al., 2024](https://arxiv.org/abs/2407.10140))。

**SFT**: 数据混合 MetaMathQA ([Yu et al., 2023](https://arxiv.org/abs/2309.12284)) + OpenMathInstruct2 ([Toshniwal et al., 2024](https://arxiv.org/abs/2410.01560)) + NuminaMath ([LI et al., 2024](https://huggingface.co/AI-MO/NuminaMath-CoT))。用 **model correctness consistency** ([Qi et al., 2024](https://arxiv.org/abs/2408.06195)) 过滤低质量 prompt (丢弃 inter-model consensus 为 0 的样本)。

**RL**: PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))，**binary verifiable reward** —— 数学题答案是离散的，能自动验证，避免 reward hacking。SFT/RL 数据集 disjoint，避免 trivial overlap。

### 2.3 评估协议

**Upstream Cloze** (next-token prediction, 0-shot): HellaSwag, Winogrande, PIQA, OBQA, ARC-E/C — 这些不需要 conversational 能力，反映 base model 的语言建模质量。

**Downstream Generative**: 
- ID (math): GSM8K-Platinum ([Vendrow et al., 2025](https://arxiv.org/abs/2502.03461)) 1319 题, MATH ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) 12,500 题
- OOD: CRUXEval (code reasoning, 800 题), BoardgameQA (logical reasoning, 15K 题), TabMWP (table reasoning, 38K 题), StrategyQA (commonsense, 2,780 题)

四种 sampling scheme：
- **Pass@1**: $T=0$, 单 deterministic sample
- **Maj@16**: $T=1$, 16 samples, 投票
- **RM@16**: 16 samples, 用 Skywork-Reward-Llama-3.1-8B-v0.2 ([Liu et al., 2024](https://arxiv.org/abs/2410.18451)) 选 best
- **Pass@16**: 16 samples, 至少一个对就算 solved

外加 **Correct Ratio** (在至少一个 correct 的 group 里，correct 数 / 16) 与 **ORM Score** (avg@16 的 scalar reward)。

---

## 3. 核心 Findings：12 个 Takeaways 详解

### Takeaway 1: 过度 Pre-training 的 Diminishing Returns

**Figure 2 数据** (1B model upstream avg accuracy)：

| Pre-training tokens | Upstream Acc |
|---|---|
| 20B | 46.44% |
| 40B | 49.38% |
| 80B | 51.88% |
| 160B | 52.30% |
| 320B | 52.49% |

从 20B → 80B 提升 ~5.4 个百分点，80B → 320B 仅提升 0.6 个百分点。即 saturation 发生在 **80x–160x Chinchilla ratio** 附近。

**Figure 3 数据** (1B SFT 在 ID Maj@16):
- 20BT: 8%
- 80BT: 15%
- 320BT: 17%

更关键的是 **OOD 指标在 160BT 之后开始下降** —— Maj@16, RM@16, Pass@16 都 degrade，ORM score 同时下降，意味着 generation 整体质量变差。

这呼应了 Springer et al. ([2025](https://arxiv.org/abs/2503.19206)) 的 "overtrained models are harder to fine-tune" 假说。机制猜测：过度 pre-training 让模型参数进入 sharper minima，对 SFT 的 perturbation 更敏感；同时过度依赖 general-domain 统计模式，与 math domain 的 surface form 错配。

### Takeaway 2: Pre-training Budget 与 Model Size 的相互作用

Table 1 给出 1B vs 4B 在 fixed compute / fixed tokens 下的对比：

| Config | ID Maj@16 (SFT/SFT+RL) |
|---|---|
| 1B-320BT | 16.1 / 25.0 |
| 4B-80BT (same compute) | 13.2 / 20.0 |
| 1B-80BT | 14.1 / 21.4 |
| 4B-80BT | 13.2 / 20.0 |
| 1B-160BT | 14.2 / 22.5 |
| 4B-160BT | 26.4 / 34.8 |

非常重要的发现：**在相同 compute budget (1B-320BT vs 4B-80BT)**，小 model 反而更好；**在相同 tokens 但未饱和的 budget (80B)**，1B 仍略胜 4B；**当 tokens 进入 4B 的 saturation regime (160B)**，4B 的 ID Maj@16 跳升到 26.4%，几乎翻倍 1B 的 14.2%。

直觉解释：compute = tokens × C(model size, flops/token)，4B 模型每 token 的 flops 是 1B 的 ~4 倍，所以在相同 compute 下 4B 只能看 1/4 的 tokens，可能未达到 saturation；只有 tokens 进入 4B 的 saturation regime，**model capacity 才被"激活"**。这与 Gadre et al. ([2024](https://arxiv.org/abs/2403.08540)) 的 dual-axis scaling law 一致 —— loss 是 compute 与 over-training ratio 两个轴的函数。

公式化为：
$$
L(N, D) = A N^{-\alpha} + B D^{-\beta} + L_\infty
$$
其中 $N$ 是参数量，$D$ 是 tokens 数，Chinchilla 得到 $\alpha \approx 0.34$, $\beta \approx 0.28$。在 compute $C \approx 6ND$ 约束下，最优 ratio $D^*/N^* \approx 20$。但当 $D/N > 20$ 进入 over-training regime 时，loss 仍能缓慢下降但收益递减，这正是 EvoLM 在 downstream 看到的现象。

### Takeaway 3: CPT 的 Catastrophic Forgetting 与 Replay 缓解

Figure 4 + Table 2：

| CPT Config | GSM8K-Platinum Pass@1 |
|---|---|
| No CPT | 6.04 |
| FineMath 50B only | 19.27 |
| FineWeb 1.6B + FineMath 48.4B | 16.21 |
| FineWeb 8B + FineMath 42B (5% replay) | **21.01** |
| FineWeb 16B + FineMath 34B (32% replay) | 15.22 |

5% replay 是甜区。Replay 过少 (1.6B = 3.2%) 不足以 retain，replay 过多 (16B = 32%) 反而稀释 domain learning。这印证了 D-CPT Law ([Que et al., 2024](https://arxiv.org/abs/2407.10140)) 与 Bethune et al. ([2025](https://arxiv.org/abs/2502.06042)) 的 forgetting scaling law。

直觉上，CPT 是一个 **multi-objective optimization**：在 domain-specific loss $\mathcal{L}_{\text{math}}$ 与 general loss $\mathcal{L}_{\text{web}}$ 之间的 Pareto frontier 上做权衡。Replay 引入的混合 loss 可以写为：

$$
\mathcal{L}_{\text{CPT}} = (1-\lambda)\mathcal{L}_{\text{math}} + \lambda\mathcal{L}_{\text{web}}
$$

其中 $\lambda \in [0,1]$ 是 replay ratio。$\lambda = 0$ 时纯 math，forgetting 严重；$\lambda$ 太大时，math adaptation 不足。$\lambda \approx 0.05$ 是 sweet spot，意味着只需要少量 general data 就能 anchor 住已学的 distribution，而大部分 gradient signal 仍流向 math adaptation。

### Takeaway 4-6: CPT 在 Pre-training 与 Post-training 之间的 Bridge 角色

Figure 5 显示 downstream performance 随 CPT tokens (固定 8B replay) 的变化：
- SFT model 的 ID greedy accuracy: 2BT → 5%, 32BT → 12%, 42BT plateau
- OOD 同步上升
- **没有 CPT 时，RL 反而比 SFT 差** (Maj@16, RM@16, Pass@16 都低于 SFT)

这是一个 counter-intuitive 但关键的结果。直觉解释：

SFT 是 supervised signal，模型至少在 prompt-response 对的 conditional distribution 上被 anchor 住，即便 reasoning 弱，输出 format 是正确的。RL 是 exploration-based，依赖 base model 自己产生的 rollout。如果 base model 没有足够 math prior (CPT 没注入)，rollout 中正确解非常稀疏，binary reward 极少触发，policy gradient 更新方向噪声极大，模型可能 collapse 到一些表面 format 但 reasoning 错误的 trajectory。

CPT 的角色可以理解为：**为 RL 提供 exploration 的种子**。数学形式化地，PPO 的 advantage 估计：

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots
$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

如果 $r_t$ 几乎都是 0 (math prior 不足，rollout 全错)，则 $V(s)$ 学到的也是 0，advantage 退化为噪声，policy $\pi_\theta$ 更新方向随机。CPT 注入 math prior 等价于提升 $r_t$ 的 non-zero 概率，让 RL 信号变得 informative。

### Takeaway 7-8: SFT 的 ID/OOD 权衡

**Varying SFT epochs** (Figure 6, 100K examples fixed):
- ID 在 ~8 epochs saturation
- OOD 在 2-4 epochs peak 后 decline

**Varying SFT dataset size** (Figure 7, 1 epoch fixed):
- ID 单调上升
- OOD 波动，可能下降

这与社区常用的 ~3 epoch SFT 经验 ([Zhou et al., LIMA, 2023](https://arxiv.org/abs/2305.11206)) 吻合。机制上，SFT 是在 base model 的 conditional distribution 上做 Kullback-Leibler projection：

$$
\theta^* = \arg\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}}\left[ -\log p_\theta(y|x) \right] + \Omega(\theta)
$$

当 epochs 过多或 dataset 过大但 variance 不足，模型从 general distribution 收敛到 training set 的 surface form (memorization)，OOD generalization 通过 sharp surface form alignment 失效。

RL 的 marginal gain 在 over-SFT 后缩小 (Takeaway 8)：因为 RL 是 self-improvement，依赖 base + SFT 的 $\pi_{\text{ref}}(y|x)$ 中保留的 entropy。如果 SFT 把 distribution collapse 到一个 narrow mode，RL 的 exploration 空间被压缩，advantage signal 也弱化。这呼应 Chu et al. ([2025](https://arxiv.org/abs/2501.17161)) 的 "SFT memorizes, RL generalizes" 比较。

### Takeaway 9-11: RL Compute Scaling 与 SFT/RL 数据分配

**Varying RL epochs** (Figure 8a, 100K examples fixed):
- Greedy / Maj@16 / RM@16 在 8-16 epochs peak
- **Pass@16 在 4 epochs 之后 degrade**
- Correct Ratio 持续上升

这个组合非常 informative。Pass@16 measures "至少一个对的概率"，是 reasoning capability 的 proxy；Correct Ratio measures "在至少一个对的 group 里答对的频率"，是 confidence 的 proxy。Pass@16 下降 + Correct Ratio 上升意味着：**RL 主要 sharpen 已有 correct output 的 confidence，而不是让模型学会新的 reasoning trajectory**。

这印证了 Yue et al. ([2025](https://arxiv.org/abs/2504.13837)) 与 Zhao et al. ([Echo Chamber, 2025](https://arxiv.org/abs/2504.07912)) 的结论 —— RL post-training amplifies pre-training patterns。

**Varying RL dataset size** (Figure 8b, 8 epochs fixed):
- Greedy / Maj@16 / RM@16 在 150-200K examples saturate
- 350K, 400K examples 出现 drastic drop —— **response length 爆炸** (Figure 12)，超过 context window

这是一个 RL 训练的 failure mode：当数据足够多、训练足够久，policy 学会通过增加 response length 来获取更多 positive reward (长 chain-of-thought 通常更容易"凑出"正确答案)，但一旦超出 context window，generation 被 truncate，导致 correctness 判定失败。Figure 13 显示 epochs scaling 时 response length 稳定，**深 per-sample optimization 比 broad data coverage 更 stable**。这与 Wang et al. ([2025, RL with one example](https://arxiv.org/abs/2504.20571)) 一致。

**SFT/RL data allocation** (Figure 9, total 100K examples fixed):
- ID accuracy 随 SFT 比例上升，plateau 在 70K SFT + 30K RL
- OOD accuracy 在 10K SFT + 90K RL 时 peak

直觉：SFT 提供 reasoning skill 的 **template + format learning**，对 ID 任务直接 helpful；RL 提供 **generalization through exploration**，对 OOD 更 helpful。在 100K budget 下，SFT 只需要少量 (10K) 就足以提供 format scaffolding，剩下的 budget 应该给 RL 探索。

### Takeaway 12: ORM Score 作为 Reliable Proxy

Figure 10 显示 ORM score (avg@16) 与 Maj@16 accuracy 的 Pearson correlation 在多数 task 上为 0.62–0.84，除了 StrategyQA (commonsense，ORM 训练分布不匹配)。

Figure 14 显示 validation PPL 与 accuracy 几乎 uncorrelated —— post-trained model 在 LM task 上 **miscalibrated**。

直觉上，post-training 把模型从 maximum likelihood 推向了 maximum reward / instruction following，loss landscape 已经偏离了 next-token cross-entropy 的 axis。PPL 衡量的是：
$$
\text{PPL} = \exp\left( -\frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i) \right)
$$

但 post-trained model 的输出 distribution 已经被 SFT (KL to human demo) 和 RL (KL to reward) 扭曲，不再对应 ground truth LM likelihood。ORM (Outcome Reward Model) 是训练来预测 task correctness 的，与下游 metric 同分布，所以 correlation 高。

这个 finding 对实践非常重要：**post-training 阶段不能用 validation loss 监控**，需要用 ORM 或 task-specific reward。

---

## 4. 额外 Study：Intermediate Checkpoint 不可靠

Table 3 比较了 20BT/40BT full run 与 160BT run 中 20BT/40BT 处的 intermediate checkpoint：

| Model | Upstream | Math L1 (Greedy/Pass@16) | Math L2 |
|---|---|---|---|
| 20BT full | 46.43 | 2.75 / 17.85 | 3.36 / 15.10 |
| 20BT int. | 46.07 | 2.52 / 11.44 | 1.90 / 12.64 |
| 40BT full | 49.38 | 2.97 / 17.96 | 3.36 / 14.88 |
| 40BT int. | 49.06 | 1.37 / 9.38 | 2.68 / 8.72 |

Upstream 几乎相同，但 downstream Pass@16 差距巨大 (17.96 vs 9.38)。这是因为 cosine LR schedule 的 annealing 阶段会显著改变 final weight 的 curvature 与 sharpness，intermediate checkpoint 还没经历这个 annealing，即便 loss 接近，但 weight landscape 的几何性质不同。

直觉解释：LR annealing 时，模型参数沿 loss surface 向 sharper minima 收敛，sharper minima 对 task-specific probing 更敏感；intermediate checkpoint 处在 flatter region，downstream probing (尤其是 SFT 后的 generative) 无法被 effectively trigger。这与 Power Scheduler ([Shen et al., 2024](https://arxiv.org/abs/2408.13359)) 与 LR annealing scaling law ([Tissue et al., 2024](https://arxiv.org/abs/2408.11029)) 的工作一致。

**对社区的警示**：任何用 intermediate checkpoint 做 post-training study 的结论都应被审慎对待。

---

## 5. RL Hyperparameters 细节

Table 8 给出 PPO 配置：
- actor_lr: 1B=2e-6, 4B=1e-6
- critic_lr: 1B=2e-5, 4B=1e-5 (10x actor)
- KL coefficient: 1e-4
- train_batch_size: 1024 (1B), 2048 (4B)
- max_prompt_length: 1024, max_response_length: 1024

PPO 目标函数：
$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right] - \beta \cdot \text{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]
$$

其中 $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是 importance ratio，$\hat{A}_t$ 是 GAE advantage estimate，$\beta = 1\text{e-}4$ 是 KL penalty，$\pi_{\text{ref}}$ 是 SFT 之后的 reference policy。

KL penalty 这么小 (1e-4) 意味着作者允许 policy 偏离 SFT 比较多，给 exploration 留空间。Critic lr 是 actor 的 10x 是因为 critic 通常更难收敛，需要更快的 lr。

---

## 6. Limitations 与 Open Questions

作者自己承认：
1. 只到 4B 参数，更大的 model 是否有相同 trends 未知。
2. 只覆盖 reasoning-centric post-training，safety / instruction following / tool use / code 的 dynamics 可能不同。
3. 只用 PPO + verifiable reward，DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) / GRPO ([DeepSeek, 2024](https://arxiv.org/abs/2401.05858)) / RLOO / REINFORCE++ 等其他 RL 算法可能展现不同 scaling 特性。

**我的额外联想 / Open Questions**：

1. **Pre-training data mixture 影响**: 文章只用 FineWeb-Edu，但现代 base model (Llama 3, Qwen 2.5) 是 web + code + math 多源 mixture。Edu-heavy data 是否已经预先 favor 了 math downstream，让 CPT 的边际效应被低估？
2. **CPT replay 的 curriculum**: 文章是 random interleave。Curriculum-based replay (e.g., difficulty-aware) 可能进一步提升 efficiency。
3. **RL reward sparsity**: binary verifiable reward 在 math 上 work，但 tool use / agentic task 没有 binary reward，是否还能观察到 "RL amplifies pre-training patterns" 的现象？
4. **Response length explosion 的 root cause**: RL 训练中 response length 单调上升是一个已知的 RLHF failure mode ([Singhal et al., 2023](https://arxiv.org/abs/2310.01027))。EvoLM 在 350K example 之后才 collapse，是否有 explicit length penalty 能延迟这个 threshold？
5. **Compute-optimal RL ratio**: 是否能推导一个 RL scaling law $L_{\text{downstream}}(N, D_{\text{pre}}, D_{\text{cpt}}, D_{\text{sft}}, D_{\text{rl}}, E_{\text{sft}}, E_{\text{rl}})$? 这是当前文献缺失的。
6. **Pass@K 与 reasoning capability 的解耦**: 文章用 Pass@16 作为 reasoning proxy，但 $K=16$ 可能不够。当 $K \to \infty$ 时 (e.g., 1000 samples)，RL 是否仍然不提升 fundamental capability? 这是验证 "RL amplifies vs. elicits" 的关键实验。

---

## 7. 对你 (Andrej) 直觉的几个 takeaways 总结

回到 build intuition 的目标，这篇 paper 给我的几个核心 mental model：

1. **Pre-training saturation 是 model size 与 tokens 的双轴函数**：compute 在两个轴之间 trade off，且只在 saturation regime 之内 model capacity 才能被 downstream post-training 充分激活。1B-320BT 反超 4B-80BT 是这个机制的直接证据。

2. **CPT 不是简单的 "domain adaptation"，是 bridge layer**：没有 CPT，RL 会 collapse；CPT 注入的 prior 是 RL exploration 的必要种子。CPT replay ratio ~5% 是一个 robust sweet spot，背后是 multi-objective Pareto 的几何结构。

3. **SFT 与 RL 在 ID/OOD 上有相反的 preference**：SFT memorizes → ID up, OOD down；RL explores → ID 平台，OOD up。在 fixed budget 下，data allocation 是 zero-sum game。

4. **RL 不创造 reasoning capability，它 redistribute probability mass**：Pass@16 下降 + Correct Ratio 上升 = evidence。RL 让 already-correct outputs 概率上升，但不会让原本 incorrect 的 trajectory 变 correct。这意味着 RL 的 fundamental ceiling 由 pre-training + CPT 决定。

5. **Validation PPL 在 post-training 阶段失去监控价值**：必须用 ORM / reward proxy。这是为什么现代 RLHF pipeline 都用 reward model 做 eval。

6. **Intermediate checkpoint 不能替代 full run**：LR annealing 改变 weight landscape 几何，对 downstream probing 至关重要。任何用 mid-training checkpoint 做 post-training study 的结论都需要 caveat。

这篇 paper 的最大贡献不在于单个 finding，而在于把 4 个 stage 的 scaling study 在同一 framework 下 control variable 完成 —— 这种 end-to-end controlled study 在 literature 里非常稀缺，比 GPT-4 / Claude / Gemini technical report 里那些孤立 ablation 价值高得多。

主要参考链接：
- EvoLM 论文本身: [arXiv (推测) - 查 Hugging Face](https://huggingface.co/papers?search=EvoLM)
- FineWeb-Edu: [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- FineMath / SmolLM2: [https://arxiv.org/abs/2502.07298](https://arxiv.org/abs/2502.07298)
- Chinchilla scaling law: [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- D-CPT Law: [https://arxiv.org/abs/2407.10140](https://arxiv.org/abs/2407.10140)
- Overtrained models harder to fine-tune: [https://arxiv.org/abs/2503.19206](https://arxiv.org/abs/2503.19206)
- Does RL incentivize reasoning: [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837)
- Echo Chamber (RL amplifies pretraining): [https://arxiv.org/abs/2504.07912](https://arxiv.org/abs/2504.07912)
- SFT memorizes, RL generalizes: [https://arxiv.org/abs/2501.17161](https://arxiv.org/abs/2501.17161)
- Skywork-Reward: [https://arxiv.org/abs/2410.18451](https://arxiv.org/abs/2410.18451)
