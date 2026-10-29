---
source_pdf: Scaling Behaviors of LLM Reinforcement Learning Post-Training An Empirical
  Study in Mathematical Reasoning.pdf
paper_sha256: cab7b0093cb2a4a87d8f6541324d2bb7bd82acc270fe448e3516750fa5dac2f0
processed_at: '2026-08-12T03:20:10-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy，行，我把刚才那堆公式和表格扔掉，咱们直接像在 SHAlab 茶水间那样聊。

---

## 这篇paper到底在干嘛

一句话：**把 Kaplan 2020 那套 scaling law 的思路，搬到 RL post-training 上，用 Qwen2.5 全家桶跑了一遍，看看 power-law 还成不成立。**

结果是：基本成立，但有几个 twist。

---

## 三个真正有工程价值的发现

### 1. RL post-training 的 efficiency 会 saturate，32B 基本就到顶了

这个是最 actionable 的结论。

他们拟合出来 learning efficiency $k(N)$ 长这样：

$$k(N) = \frac{K_{\max}}{1 + N_0/N}$$

这是个 Michaelis-Menten 形式——酶动力学那个公式。意思是 efficiency 随 model size 增长，但有 asymptote。拟合出来的 $N_0$ 大约在 10-30B。

**翻译成人话**：从 0.5B 到 32B，RL 的 learning efficiency 涨得很快；从 32B 到 72B，基本上就 plateau 了，边际收益很小。

这个跟 pretraining 完全不一样。Pretraining 你 70B 到 700B 还能明显看到 loss 在降。但 RL post-training 的 gain 是被 reward signal 的 information content 顶住的——binary reward 每个 token 最多 1 bit，再大的 model 也榨不出更多 signal。

**工程意义**：做 RL post-training，72B 可能已经接近 sweet spot。再往上 scaling model，不如去改 reward design 或者加 test-time compute。

---

### 2. 32B 在固定 compute 下短期会赢 72B

看 Figure 1 的左半边，你会发现 32B 的曲线在 72B 上方。这是因为相同 FLOPs 预算下，32B 能跑更多 steps，而 RL 早期 steps 的 loss 下降特别快。到后期 72B 才 crossover 上去。

**这告诉你一件事**：如果你的 compute budget 不够大，别盲目用大模型做 RL。32B 可能是更 cost-effective 的选择。Chinchilla 当年在 pretraining 上也观察到类似现象，但 crossover 点不一样。

---

### 3. Data reuse 是 free lunch（τ ≤ 25）

这是我觉得最 surprising 的发现。

他们固定 total data volume，变化 reuse factor $\tau$：
- $\tau = 1$：50k unique samples，跑 1 epoch
- $\tau = 25$：2k unique samples，重复 25 次，total 还是 50k
- $\tau = 100$：500 unique samples，重复 100 次

**结果**：$\tau \leq 25$ 时 performance 几乎没差别。$\tau = 100$ 才 overfit。

这跟 pretraining 完全相反。Pretraining 上 Hernandez 2022 做过类似实验，重复 4 次就有 measurable harm。

**为什么 RL 这么 robust？** 我的 intuition：
- Policy 在变，所以同一 prompt 在不同 training stage 产生的 rollout 是不同的
- Gradient signal 来自 reward，不是 likelihood，stochasticity 更高
- Binary reward 的 effective information density 高，reuse 的边际伤害小

**工程意义**：如果你只有 5k 道高质量数学题，重复 25 次训成 125k，效果跟 125k unique 差不多。这跟 LIMR 那篇 "Less is more" 的结论互相印证。

---

## 两个值得警惕的发现

### 1. RL 在数学上做多了会损害 logical reasoning

Figure 8 的 Zebra Puzzle 曲线：14B 模型随着 RL training，logical reasoning performance **下降**。

这是个 negative transfer 现象。RL 把模型的 reasoning pattern sharpen 到数学题那种 "step-by-step derivation" 的模式上，cost 是其他 reasoning pattern 的 forgetting。

**这点 paper 里没深挖，但我觉得很重要**。它暗示 RL post-training 不是 "通用 reasoning ability boost"，而是 "specific reasoning pattern sharpening"。如果你想要 general reasoning model，可能需要 multi-domain RL 或者某种 regularization。

---

### 2. OOD transfer 基本为零

HumanEval (code)、SuperGPQA (STEM) 在整个 RL training 过程中 performance 几乎 flat。

这跟很多人的直觉相悖——大家以为数学 reasoning 能力会 transfer 到 code reasoning。但这篇 paper 用数据说：不会。RL post-training 是 highly specialized 的。

---

## 这套 scaling law 的形式

他们最终提的公式其实就是：

$$\log L = -k(N) \cdot \log X + E(N)$$

其中 $X$ 可以是 compute $C$ 或 data $D$，$k(N)$ 是上面那个 saturating 函数。

**说白了**：log loss 和 log resource 是线性的，斜率是 $k(N)$，而 $k(N)$ 随 model size 增长但会 saturate。

他们还证明 compute 和 data 两个维度共享同一个 $k(N)$——因为 $C = N \cdot D \cdot \phi$ 这个 linkage。这其实是说 RL 的 scaling law 比预想的简单，只有一个 free 维度。

---

## 我对这篇 paper 的几个疑虑

### Test loss 这个 metric 根本不 well-defined

他们用 $L = 1 - R/R_{\max}$。但 $R_{\max}$ 在 GSM8K 是 1319，在 AIME 是 30。AIME 上做对一道题 loss 降 0.033，统计 noise 极大。Power-law 拟合在 AIME 上根本不可信。

Paper 自己在 Discussion 里承认这点，说找不到像 Hilton 2023 那种 "intrinsic performance" 的 normalization。这其实暴露了 LLM RL scaling law 的根本难题：**没有 environment-independent 的 performance measure**。

### Curriculum 是个 confound

他们用 difficulty-sorted curriculum。这意味着 "early training dynamics" 包含 curriculum 效应。如果 random ordering，曲线形状会不一样。Paper 没做这个 ablation。

### GRPO specific

GRPO 的 group normalization 是个 strong inductive bias。PPO 用 critic 估计 advantage，critic 本身有 scaling behavior。所以这篇 paper 的结论换算法可能不成立。

### 72B 上限不够

Saturation 在 32B 开始，但实验最大只到 72B。要真的 confirm $K_{\max}$ 是 asymptote，需要跑 200B+ 的实验。DeepSeek、xAI、Anthropic 内部可能已经有这个数据，但没公开。

---

## 对未来的 intuition

这篇 paper 给我的几个 intuition：

**1. RL post-training 的 scaling 不是 "更大队列更好"**，而是 "32B 量级最 efficient"。这跟 pretraining 的 Chinchilla optimal 不一样——pretraining 的 compute-optimal 在 70B 量级，RL post-training 的 compute-optimal 可能更小。

**2. Reward signal 是 bottleneck**。Binary reward 的 information content 顶住了 $K_{\max}$。要突破，得改 reward——process reward、multi-dimensional reward、或者 environment-grounded reward。

**3. Data reuse 在 RL 阶段几乎免费**。这对工业界是好消息，因为高质量 reasoning data 真的很难搞。

**4. OOD transfer 很弱**。如果你想要 general reasoning model，单 domain RL 不够，得 multi-domain 或者用 agentic RL 让 environment 提供 diverse signal。

**5. Test-time compute 和 train-time compute 是正交的**。这篇 paper 只覆盖 train-time。o1/o3 那条路是 test-time scaling。两个维度加起来才是完整的 reasoning scaling law。

---

## 最终一句话

**这篇 paper 用 Qwen2.5 全家桶告诉你：RL post-training 也遵循 power-law，但 efficiency 在 32B 左右 saturate；数据 reuse 几乎免费；OOD transfer 基本为零；想继续提升 reasoning，得从 reward design 和 test-time compute 找突破口，光堆 model size 已经不够了。**

---

# 《Scaling Behaviors of LLM Reinforcement Learning Post-Training》深度解析

Karpathy 你好！这篇论文是 2025 年从 Shanghai AI Lab + Oxford 等机构出来的工作，它做的事情其实就是把你 2020 年在 OpenAI 做的 Kaplan scaling law 那套思路搬到 RL post-training 阶段，然后用 Qwen2.5 全家桶（0.5B → 72B）做了一次系统性的 empirical study。我觉得这篇 paper 真正有意思的不是它拟合出来的 power-law 漂亮不漂亮，而是它揭示了 RL post-training 和 pretraining 在 scaling dynamics 上的几个结构性差异。

---

## 1. 论文动机：为什么 pretraining 的 scaling law 不能直接套用

Pretraining 阶段的 scaling laws 现在已经非常成熟：
- Kaplan et al., 2020 (https://arxiv.org/abs/2001.08361)：首次提出 cross-entropy loss 对 N、D、C 遵循 smooth power-law
- Hoffmann et al., 2022 Chinchilla (https://arxiv.org/abs/2203.15556)：在 fixed compute 下，parameters 和 tokens 应该按比例同步增长
- Hilton et al., 2023 (https://arxiv.org/abs/2301.13442)：把 RL 在 CNN 上的 scaling 也套进 power-law 框架

但是 DeepSeek-R1 (https://arxiv.org/abs/2501.12948)、Kimi K1.5 (https://arxiv.org/abs/2501.12599) 这种 RL-based reasoning model 出现之后，有一个根本问题没人回答：

**Pretraining loss 是一个 well-defined 的 target distribution (next token prediction on web text)，但是 RL post-training 的 objective 是 reward maximization，loss surface 完全不一样。** 而且在 LLM 上 RL 没有 AlphaZero 那种 self-play environment，只能用 human-curated dataset 当 proxy，所以 test loss L = 1 - R/R_max 这种 metric 本身就和 pretraining 的 cross-entropy loss 性质完全不同。这篇 paper 就是想搞清楚：在这种 proxy metric 下，scaling 关系长什么样？

---

## 2. 实验设置的技术细节

### 2.1 Model family 选择

用 Qwen2.5 (https://arxiv.org/abs/2412.15115) 全家桶 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)，关键设计是 **shared architecture**，这样 parameter count N 就是唯一的 scaling 变量，不需要考虑 depth/width ratio 这种 confounder。这点和 Kaplan 当年的做法是一致的。

后面还验证了 Llama 3 (1B, 3B, 8B, 70B-Instruct) 来做 cross-architecture generalization，证明这不是 Qwen-specific 的现象。

### 2.2 RL Algorithm：GRPO

用的不是 PPO 而是 GRPO (Group Relative Policy Optimization)，来自 DeepSeekMath (https://arxiv.org/abs/2402.03300)。这个选择本身就很值得讨论：GRPO 是 actor-only 设计，没有 critic，靠 group 内 normalization 来估计 advantage。

GRPO 的 objective（论文公式 3）：

$$\mathcal{L}_{\mathrm{GRPO}} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left[ \rho(\theta) \hat{A}_{i,t}, \, \mathrm{clip}(\rho(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right] - \beta D_{\mathrm{KL}} \right\}$$

变量含义：
- $G$：group size，每个 prompt 采样的 response 数量（ablation 中测试了 G ∈ {4, 8, 16, 32}）
- $o_i$：第 i 个 sampled response，$|o_i|$ 是其 token 长度
- $\rho(\theta) = \pi_\theta(o_{i,t} | q, o_{i,<t}) / \pi_{\theta_{\mathrm{old}}}(o_{i,t} | q, o_{i,<t})$：importance sampling ratio，PPO 的标准操作
- $\hat{A}_{i,t}$：advantage estimate
- $\varepsilon$：clip ratio，论文里 high & low 都是 0.2
- $\beta$：KL coefficient = 0.001
- $D_{\mathrm{KL}}$：current policy vs. reference policy 的 KL divergence，防止 policy 偏离太远

Advantage 的计算（公式 4）：

$$\hat{A}_{i,t} = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$$

这里 $\mathbf{r} = \{r_1, r_2, ..., r_G\}$ 是同一 prompt 下 G 个 response 的 reward vector。这个 normalization 是 GRPO 的核心 trick——它让 advantage 在 group 内是 zero-mean、unit-variance，因此即使没有 critic 也能有稳定的 gradient signal。这个设计本质上等价于把 group 当成一个 batch 做 baseline subtraction。

### 2.3 Reward signal

Binary reward：答案正确给 1，错误给 0。这种 sparse reward 在数学推理上特别合适，因为数学题有 ground truth 可以 verify（rule-based verification）。这避免了 reward model 自己的 scaling 问题，让 RL 的 scaling 行为可以纯粹归因于 policy network 的 scaling。

### 2.4 Test loss 定义

论文反复强调：

$$L = 1 - \frac{R}{R_{\max}}$$

其中 $R$ 是 correct solutions 数量，$R_{\max}$ 是 total。这个定义直接对标 Kaplan 当年的 test loss，是为了让 power-law 拟合在 log-log 空间有意义。**这个 metric 的关键限制是它是 task-dependent 的**——在 GSM8K 上和 AIME 上，同样模型表现出来的 "test loss" 数字含义完全不同，所以 paper 在 Discussion 里专门承认了这一点。

### 2.5 训练数据

guru-RL-92k 数据集的数学子集，经过 deduplication + difficulty filtering。**重要细节**：按 Qwen2.5-7B-Instruct 的 pass rate 做了 difficulty sorting，做 curriculum learning。这一点对结果可能有 substantial 影响，因为 curriculum 会让早期 training step 的 loss 下降速率和 random ordering 完全不一样。

### 2.6 FLOPs 计算（公式 10、11）

$$C_{\mathrm{train}} = C_{\mathrm{fwd}} + C_{\mathrm{bwd}} \approx 2NT + 4NT = 6NT \text{ FLOPs}$$

- $N$：non-embedding parameter count
- $T$：processed tokens
- Forward pass：$C_{\mathrm{fwd}} \approx 2NT$（每个 token 一次 matmul）
- Backward pass：$C_{\mathrm{bwd}} \approx 4NT$（forward + backward for gradient w.r.t. activations and weights）

这个 6NT 是标准近似，和 Chinchilla 论文用的方法一致。注意 RL 还要 rollout sampling，rollout 的 FLOPs 怎么算 paper 没说得很细，但应该是算进 forward pass 里了。

---

## 3. 核心 Scaling Law 公式

### 3.1 主公式（公式 1）

$$\log L(N, X) = -k(N) \cdot \log X + E(N)$$

变量：
- $L(N, X)$：test loss
- $N$：model parameter count
- $X$：resource budget，可以是 compute $C$ 或 data $D$
- $k(N)$：learning efficiency coefficient，是 $N$ 的函数
- $E(N)$：intercept，对应 model 的 "intrinsic performance" 或者说 initial loss floor

这个公式的关键 insight 是 **log-log linear**——log loss 和 log resource 之间是线性的。换句话说，loss 对 resource 是 power-law：$L \propto X^{-k(N)}$。

这和 Kaplan 的 $L(N) = (N_c/N)^{\alpha_N}$ 形式上一脉相承，但有几个根本性差异：
1. Kaplan 的 $\alpha$ 是 constant，这里 $k(N)$ 是 $N$ 的函数
2. Kaplan 是 pretraining loss，这里是 RL test loss
3. Kaplan 的 power-law exponent 来自 data distribution 的 intrinsic complexity，这里 $k(N)$ 反映的是 RL 优化的 efficiency

### 3.2 Efficiency saturation（公式 2）

$$k(N) = \frac{K_{\max}}{1 + \frac{N_0}{N}}$$

变量：
- $K_{\max}$：asymptotic efficiency limit，即 $N \to \infty$ 时 $k(N)$ 的上界
- $N_0$：half-saturation scale，当 $N = N_0$ 时 $k(N) = K_{\max}/2$
- $N$：model size

这个形式非常有意思，它是 **Michaelis-Menten kinetic equation** 的形式！在酶动力学里 $v = V_{\max} \cdot [S] / (K_m + [S])$，这里 substrate concentration 对应 $N$，max rate 对应 $K_{\max}$，Michaelis constant 对应 $N_0$。这种 saturating form 在很多 capacity-limited 的系统中都出现。

这个 saturation 的物理直觉：**RL learning efficiency 不能无限增长，因为 efficiency 的瓶颈不是 model capacity 本身，而是 reward signal 携带的信息量**。Binary reward 每个 token 顶多 1 bit，再大的 model 也无法从 1 bit 里挤出更多 signal。所以 $K_{\max}$ 实际上反映的是 reward channel 的 information capacity。

### 3.3 Compute-Optimal Scaling（公式 6）

$$\log L(N, C) = -k_C(N) \cdot \log C + E_C(N)$$
$$k_C(N) = \frac{K_{C\max}}{1 + \frac{N_C}{N}}$$

- $k_C(N)$：compute learning efficiency
- $K_{C\max}$：compute efficiency limit
- $N_C$：compute saturation scale

### 3.4 Data-Optimal Scaling（公式 8）

$$\log L(N, D) = -k_D(N) \cdot \log D + E_D(N)$$
$$k_D(N) = \frac{K_{D\max}}{1 + \frac{N_D}{N}}$$

- $k_D(N)$：data learning efficiency
- $K_{D\max}$：data efficiency limit
- $N_D$：data saturation scale

### 3.5 Compute-Data 一致性（Appendix C.2）

论文证明了一个很漂亮的 claim：如果 $C = ND\phi$（其中 $\phi$ 是常数），那么：

$$k_C(N) = k_D(N) = k(N)$$
$$E_C(N) = E_D(N) + k(N) \ln(N\phi)$$

这意味着 compute 和 data 两个 scaling 维度本质上是同一个 law 的两个 face。这个证明的逻辑很直接：把 $C = ND\phi$ 代入 compute 公式，分解 $\ln(ND\phi) = \ln D + \ln(N\phi)$，然后比较系数就得到了 $k_C = k_D$。

**这个 unification 很重要**，因为它意味着 RL post-training 的 scaling law 实际上只有一个 free 维度，而 pretraining 通常需要分别讨论 compute-optimal 和 data-optimal。这可能是因为 RL 的 data 是 (prompt, response) pair，每个 sample 同时贡献了 gradient signal 和 implicit compute (rollout length)。

---

## 4. 拟合参数的数据表（Table 6）

让我把这个表仔细解析一下：

| Family | Config | Scenario | $K_{\max}$ | $N_0$ (B) | $R^2$ |
|---|---|---|---|---|---|
| Qwen2.5 | Base L(N,C) | Intra | 0.135 | 13.1 | 0.996 |
| Qwen2.5 | Base L(N,C) | Inter | 0.152 | 17.4 | 0.994 |
| Qwen2.5 | Base L(N,D) | Intra | 0.135 | 11.5 | 0.995 |
| Qwen2.5 | Base L(N,D) | Inter | 0.163 | 17.0 | 0.995 |
| Qwen2.5 | Instruct L(N,C) | Intra | 0.128 | 17.3 | 0.997 |
| Qwen2.5 | Instruct L(N,C) | Inter | 0.144 | 28.3 | 0.995 |
| Qwen2.5 | Instruct L(N,D) | Intra | 0.133 | 17.1 | 0.997 |
| Qwen2.5 | Instruct L(N,D) | Inter | 0.148 | 27.2 | 0.995 |
| Llama 3 | Instruct L(N,C) | Intra | 0.089 | 11.3 | 0.998 |
| Llama 3 | Instruct L(N,C) | Inter | 0.074 | 8.5 | 0.995 |
| Llama 3 | Instruct L(N,D) | Intra | 0.091 | 12.7 | 0.998 |
| Llama 3 | Instruct L(N,D) | Inter | 0.087 | 11.8 | 0.997 |

**几个关键 observations**：

1. **$K_{\max}$ 的量级**：Qwen2.5 大约 0.13-0.16，Llama 3 大约 0.07-0.09。这个值相当于 power-law exponent $L \propto X^{-k}$ 中的 $k$。对比 pretraining 通常 exponent 在 0.05-0.1 量级，这里 RL 的 exponent 反而更大，意味着 **RL post-training 在 log space 上的下降速率比 pretraining 还快**——这是可以理解的，因为 RL 是在 already-pretrained model 上做 refinement，loss surface 局部更陡。

2. **$N_0$ 在 10-30B 范围**：这意味着 saturation 在 10-30B 就开始显著。32B → 72B 之间的 efficiency 增益已经很小（不超过 $K_{\max}$ 的 10-15%）。这点对工业界很有指导意义：**做 RL post-training，72B 已经接近 sweet spot，再往上 scaling 可能效率提升有限**。

3. **Intra vs Inter 的差异**：Inter-model prediction 的 $K_{\max}$ 普遍比 Intra 略大（0.152 vs 0.135），$N_0$ 也偏大（17.4 vs 13.1）。这暗示 small models 看自己的 trajectory 时倾向于低估极限 efficiency，但用 small models 拟合大模型时会高估极限。这个 bias 方向对实际 extrapolation 是 conservative 的，是好的。

4. **Llama 3 的 $K_{\max}$ 比 Qwen2.5 小一半左右**：尽管 Llama 3 也符合 power-law，但 efficiency 极限明显低。这可能反映 Qwen2.5 在数学任务上 pretraining 更充分（Qwen2.5-Math 系列的存在说明这点），所以 base 的 reasoning capability 更强，RL 能在上面 "挖" 出更多的 gain。

---

## 5. 几个图的关键解读

### Figure 1: Compute Scaling 主图

四个 subplot 分别是 Base/Instruct × Inter/Intra prediction：
- **(a)(b) Inter-model**：用 0.5B-32B 拟合，外推预测 72B（虚线）。72B 的实际曲线和预测高度吻合，说明 small models 可以 predict large model 的 RL 效率。
- **(c)(d) Intra-model**：用早期 training steps 拟合，外推剩余 trajectory。这非常有实用价值，意味着可以 early-stopping 决策。

**Figure 1 里有一个很重要的细节**：在 compute 较小的区域，32B 实际上比 72B 表现更好（看 (a) 图的 left 部分）。这就是 paper 里强调的 **performance crossover** 现象。原因是 72B 在相同 FLOPs 下能跑的 steps 更少，而 early steps 的 loss 下降很快，所以小模型短期内 "free" 跑更多步会赢。但长期 72B 会 cross over。

这个 crossover 在 Chinchilla 的 compute-optimal 分析里也有类似现象，只是 pretraining 那边 crossover 发生在 70B vs 10B 这种比例，这里发生在 32B vs 72B，意味着 **RL post-training 的 optimal model size 在 fixed compute 下更倾向于中等规模**。

### Figure 4: Efficiency saturation

这个图是 paper 的核心发现之一。$k_C(N)$ 和 $k_D(N)$ 都是 N 的单调递增函数，但 32B 之后增长显著放缓。拟合曲线就是公式 2 的 Michaelis-Menten 形式。

### Figure 8: Domain transfer

这个图很有信息量：
- In-domain (GSM8K, MATH-500, AMC, AIME)：loss 随 training compute 单调下降，**包括更难的 AIME** 都有正向 transfer
- Out-of-domain:
  - HumanEval (code): 几乎 flat，没 transfer
  - SuperGPQA (STEM): 几乎 flat
  - Zebra Puzzle (logical reasoning): **大模型 (14B) 反而 degradation**

最后这点是 paper 里一个 interesting but under-explored 的发现：**在数学上做 RL 会让 logical reasoning 能力下降**。这暗示 RL post-training 会 sharpen 某种 reasoning pattern，cost 是其他 reasoning pattern 的 forgetting。这和 catastrophic forgetting 在 continual learning 里的现象是一脉相承的，但这里发生在 RL 阶段。

---

## 6. Data Reuse 实验（Section 3.4）

这个实验设计非常干净：

固定 total data volume $D_{\mathrm{total}}$，变化 reuse factor $\tau$：

$$D_{\mathrm{unique}} \times \tau = D_{\mathrm{total}}$$

比如 $D_{\mathrm{total}} = 50k$，$\tau = 25$ 意味着用 2k unique samples 重复 25 次。

**关键发现**：
- $\tau \leq 25$：performance 几乎没有显著 degradation
- $\tau = 100$：明显 overfitting

这个结论和 Hernandez et al., 2022 (https://arxiv.org/abs/2205.10487) 在 pretraining 上做的 data reuse 实验结论相反——pretraining 上 data reuse 比这要 fragile 得多，每个 token 重复 4 次以上就有 measurable harm。

**为什么 RL post-training 这么 robust to data reuse？** 我的 intuition 是：
1. RL 的 gradient signal 来自 reward，而 reward 对相同 prompt 在不同 policy state 下是不同的（policy 在变化，rollout 在变化），所以即使 prompt 重复，effective training signal 是 diverse 的
2. RL 优化的是 policy 而不是 likelihood，policy gradient 有 stochasticity 来自 sampling
3. RL 的 data efficiency 本身就高（binary reward 信息密度高），所以 reuse 的边际伤害小

这个发现对实际工程意义很大：**如果你只有 5k 道数学题，把它重复 25 次训成 125k sample 的 training set，效果几乎和 125k unique sample 一样**。

---

## 7. GRPO Rollout Ablation（Appendix B.2）

在 7B 模型上测试了 $G \in \{4, 8, 16, 32\}$：

- **Data-centric view**：G=32 在相同 unique sample 数量下 lowest loss。G 越大，advantage estimate 的 variance 越小，gradient 越准确。
- **Compute-centric view**：optimal G 不是固定的——低 compute budget 下 small G 更好（因为 same FLOPs 可以跑更多 steps），高 compute budget 下 large G 更好（variance reduction 主导）。

这个 ablation 隐含一个推论：**RL 的 "batch size" 在 advantage estimate 维度上也有 scaling law**，而 PPO 类算法用 critic 估计 advantage 时，critic 本身就限制了 advantage 的 quality，所以 PPO 的 scaling 行为可能和 GRPO 不同。

---

## 8. Loss Decomposition Model（Appendix D）

这是 paper 里最 theoretical 的部分，但作者把它放在 appendix 说明他们自己也不完全确信。公式 19：

$$L(N, D) = L_\infty + \left(\frac{N_0}{N}\right)^\alpha + \frac{\lambda(N)}{1 + \left(\frac{D}{D_0(N)}\right)^\beta}$$

各项含义：
- $L_\infty$：irreducible loss，无法消除的 loss floor（task intrinsic uncertainty）
- $(N_0/N)^\alpha$：model-limited loss，finite capacity 下的 asymptotic floor（对标 Kaplan 的 $\alpha_N$ exponent）
- $\lambda(N)$：learnable capacity，model size N 能从 post-training 中获得的最大 loss reduction
- $D_0(N)$：characteristic dataset scale，到达一半 learnable capacity 的 dataset size
- $\beta$：logistic exponent，控制 S-shape 的陡峭程度

这个模型的 learning progress term 是 **logistic function in log D**，也就是 S-curve。这是对主文 power-law 的一个 refinement——主文的 power-law 假设 $\log L$ vs $\log D$ 是全局线性，但实际上 RL learning curve 通常是 S-shape 的，前期 flat（warmup），中期快速下降，后期 plateau。这个 decomposition 试图捕捉这种 S-shape。

公式 20-22 给出了这个 decomposition 和主文 power-law 之间的联系：在 $D \approx D_0(N)$ 这个 sweet spot 附近取 effective slope，得到 $k_{\max}(N) = K_{\max}/(1 + S(N))$，其中 $K_{\max} = \beta/2$，$S(N) = 2(L_\infty + (N_0/N)^\alpha)/\lambda(N)$。这就是主文公式 2 的物理来源——**Michaelis-Menten 形式实际上是 logistic learning curve 在 mid-point 处的 tangent slope**。

这个联系我觉得很漂亮：它说明主文的简单 power-law 和 appendix 的精细 S-curve model 之间不是 contradiction，而是 coarse-grained vs fine-grained 的关系。

---

## 9. 与已有 scaling law 工作的关系

### 9.1 vs Kaplan (2020)

- Kaplan 假设 $\alpha$ 是 constant，这里 $k(N)$ 是 $N$ 的函数
- Kaplan 优化 next-token cross-entropy，这里优化 reward-based test loss
- Kaplan 的 power-law exponent 反映 data distribution complexity，这里 $k(N)$ 反映 RL optimization efficiency

### 9.2 vs Chinchilla (2022)

- Chinchilla 说 compute-optimal 是 N 和 D 同步 scale，这里发现 RL post-training 在 fixed compute 下，32B 已经接近 sweet spot
- Chinchilla 的最优 N/C 比例在 pretraining 是 ~20 tokens/parameter，这里没有给出对应的 ratio，但 paper 暗示 RL 的 optimal N 比 pretraining 偏小

### 9.3 vs Hilton et al. (2023)

Hilton (https://arxiv.org/abs/2301.13442) 在 CNN RL 上做 scaling law，提出 "intrinsic performance" 概念。这篇 paper 在 Discussion 里承认在 LLM 上找不到对应的 intrinsic performance measure，因为 reward 是 dataset-dependent 的。这其实暴露了一个深层问题：**LLM RL 的 scaling law 很难做 cross-task normalization**。

### 9.4 vs DeepSeek-R1 / Kimi K1.5

这两个工作都是 RL-based reasoning model 的代表作，但都没系统研究 scaling。DeepSeek-R1 (https://arxiv.org/abs/2501.12948) 强调 RL 可以 incentivize reasoning without SFT cold start，Kimi K1.5 (https://arxiv.org/abs/2501.12599) 强调 long context scaling 和 curriculum。这篇 paper 给他们的工作补上了 scaling analysis 的维度。

### 9.5 vs LIMR (Li et al., 2025b, https://arxiv.org/abs/2502.11886)

LIMR 主张 "Less is more for RL scaling"，这篇 paper 的 data reuse 实验从另一角度验证了这点——少量 unique data + 多次 reuse 就能接近 full data 的效果。

---

## 10. 我对这篇 paper 的几个保留意见

### 10.1 Test loss 作为 metric 的局限性

Paper 自己承认这点。$L = 1 - R/R_{\max}$ 在 GSM8K (1319 题) 上和 AIME (30 题) 上的统计 noise 完全不同。AIME 的 R_max=30，每多答对一题 loss 就降 0.033，所以 30 道题上的 power-law 拟合本质上信号噪声比很差。这个 issue 可能被 holdout 500 题部分缓解，但 fundamental 的 task difficulty dependence 没法去除。

### 10.2 Binary reward 的特殊性

Binary reward 在数学上可行是因为有 ground truth。但在 coding、tool use、open-ended reasoning 上 reward 通常是 continuous or multi-dimensional。这篇 paper 的 scaling law 在 continuous reward 下是否成立完全不清楚。

### 10.3 Curriculum learning 的 confound

Paper 用 difficulty-sorted curriculum，这意味着 power-law 拟合的 "early dynamics" 包含了 curriculum 的影响。如果用 random ordering，loss vs compute 的曲线形状可能完全不同（早期可能 flat，因为难题在前面）。Paper 没做这个 ablation。

### 10.4 72B 上限

Saturation 的 $N_0$ 在 10-30B 范围，但实验最大只到 72B。要真正确认 $K_{\max}$ 是 asymptote，需要至少 200B+ 的实验。100B 模型上 $k(N)$ 是否真的接近 $K_{\max}$ 仍是个 open question。Paper 在 Limitations 里承认这点。

### 10.5 GRPO 算法 specific

Paper 用 GRPO，但 PPO、RLOO、REINFORCE++ 这些算法的 scaling behavior 可能不同。GRPO 的 group normalization 是一个 inductive bias，它可能让 efficiency saturation 更明显。如果用 vanilla REINFORCE，可能 saturation 出现得更晚。

---

## 11. 联想：这个工作对未来 reasoning model 的意义

### 11.1 Compute allocation 策略

如果 $K_{\max}$ 是 efficiency limit，那意味着 RL post-training 的 marginal return 是 bounded 的。要做下一步 reasoning 提升的路径：
1. 改 reward signal（从 binary 到 process reward，参考 Lightman et al. PRM https://arxiv.org/abs/2305.20050）
2. 加 test-time compute（Snell et al. 2024 https://arxiv.org/abs/2408.03314）
3. 多 agent / tool use（Mai et al. 2025 https://arxiv.org/abs/2505.07773）

### 11.2 与 OpenAI o1 / o3 的关系

OpenAI o1/o3 强调 test-time compute scaling。这篇 paper 是 train-time compute scaling 的 study。两个方向正交，最终 reasoning model 的 total scaling 应该是 train-time + test-time 的 joint optimization。这篇 paper 提供了 train-time 那一半的 quantitative foundation。

### 11.3 与 agentic RL 的关系

Paper 在 Discussion 里点到了 agentic LLM 是 future direction。Agent RL 的 scaling law 会比纯 reasoning RL 复杂得多，因为：
- Environment interaction 引入了额外的 compute
- Reward 通常是 sparse + delayed
- Action space 是 structured (tool calls, code execution)

我猜测 agent RL 的 efficiency saturation 会比 reasoning RL 出现得更早，因为 environment complexity 会成为新瓶颈。

---

## 12. 一句话总结

**这篇 paper 用 Qwen2.5 全家桶把 RL post-training 的 scaling 行为拟合成了一个带 efficiency saturation 的 power-law，揭示了三个工业上有用的结论：(1) 32B 左右是 RL post-training 的 sweet spot，(2) 数据 reuse 是 free lunch，(3) early training dynamics 可以 predict 最终 performance。** 

但是它的 metric (binary reward based test loss) 和 algorithm (GRPO) 选择都偏 specific，generalization 到 continuous reward、PPO、agent RL 还有很长的路要走。

---

## References

- Kaplan et al., 2020: https://arxiv.org/abs/2001.08361
- Hoffmann et al., 2022 (Chinchilla): https://arxiv.org/abs/2203.15556
- Hilton et al., 2023: https://arxiv.org/abs/2301.13442
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kimi K1.5: https://arxiv.org/abs/2501.12599
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Qwen2.5 Technical Report: https://arxiv.org/abs/2412.15115
- VeRL Framework: https://arxiv.org/abs/2409.19256
- Hernandez et al., 2022 (data reuse): https://arxiv.org/abs/2205.10487
- Lightman et al., 2023 (PRM): https://arxiv.org/abs/2305.20050
- Snell et al., 2024 (test-time compute): https://arxiv.org/abs/2408.03314
- LIMR: https://arxiv.org/abs/2502.11886
- Wei et al., 2023 (CoT): https://arxiv.org/abs/2201.11903
- Muennighoff et al., 2023 (data-constrained): https://arxiv.org/abs/2305.16264
- Agent RL scaling: https://arxiv.org/abs/2505.07773
