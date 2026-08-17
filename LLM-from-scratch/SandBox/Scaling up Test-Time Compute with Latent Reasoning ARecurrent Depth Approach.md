---
source_pdf: Scaling up Test-Time Compute with Latent Reasoning ARecurrent Depth Approach.pdf
paper_sha256: 96f4543683a06a470ccb0b6b0db27a98f283fde9467d175e838f03a629a76c7b
processed_at: '2026-08-12T03:44:05-07:00'
target_folder: LLM-from-scratch/SandBox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲这篇paper

## 核心idea一句话版

现在的reasoning model (o1, R1)是让模型"说"出思考过程——一步步用文字verbalize到context里。这篇paper说：干嘛非要"说"出来？让模型在脑子里（latent space）反复转几圈，想明白了再开口就行。

## 打个比方

想象你问一个人一道数学题：

**CoT做法**：那个人拿出纸笔，一步一步写下来"先算3×4=12，再算12+5=17..."，你看完他的推导过程，最后得到答案。

**Recurrent depth做法**：那个人盯着题目看了一会儿，脑子里转了几圈（你看不见他在想什么），然后直接告诉你答案。你问他怎么想的，他说"我也说不清，就是想明白了"。

这篇paper就是让模型学会后者。

## 怎么实现的

模型结构切成三段：

1. **Prelude（序曲）**：把input token embedding到latent space
2. **Recurrent block（循环主体）**：同一个block反复跑r次，每次都把原始embedding重新注入，在latent state上反复refine
3. **Coda（尾声）**：把最终latent state decode成output token

关键点：**每次iteration都重新注入input embedding**。这就像你反复看题目，每看一遍都在脑子里refine你的理解，而不是看一眼就闭眼想。

## 训练时怎么搞

不能每次都跑固定r次，不然模型只学会在某个特定深度下工作。所以训练时**每个batch随机采一个r**——这次跑5次，下次跑30次，偶尔跑100次。

分布用log-normal Poisson：大部分时候跑少于平均值的次数，偶尔来个大的。这让模型学会"给我多少compute我都能用"。

还有一个trick：**truncated backprop**——只backprop最后8次iteration的gradient。不然跑100次的话memory爆了。

## 为什么这事难搞

paper里记载了三次run，前两次都失败了，很有意思：

**第一次失败**：模型所有token的hidden state collapse到一起了——不管输入什么，latent state都变成一样的。因为recurrence本身就在mix token之间的信息，没控制好就全部糊成一团。

**第二次失败**：看起来训练loss在降，但发现问题——模型学会了"无视"recurrent state，只靠当前embedding做预测。r=1和r=32的validation perplexity一模一样。模型走了捷径。

**第三次成功**：用sandwich norm（每层前后都加norm）+ 更低learning rate + 特殊initialization，终于让模型engagement with recurrent state。r越大效果越好。

**Intuition**：recurrence这个设计很容易让模型走捷径——要么全塌缩，要么直接ignore。需要architectural prior强制它认真对待recurrent state。

## 结果如何

3.5B参数的model，800B token训练，在math reasoning上非常强：

- GSM8K：34.80%（strict），吊打OLMo-7B的6%
- MATH：12.58%，也大幅领先
- Code：HumanEval 23.17%，击败所有general purpose开源model

但memorization-heavy的任务（比如需要记住事实的closed-book QA）相对弱，因为参数少记不住那么多facts。

## 最有意思的发现：latent space里发生了什么

paper Section 7做了latent trajectory可视化，发现了几种emergent pattern：

**1. Convergence（收敛）**：简单token的latent state转几圈就停在一个点不动了。就像想简单问题想一会儿就想通了。

**2. Orbits（轨道）**：数学题里的数字token，latent state在high-dimensional space里画圈圈。比如"Claire makes a 3 egg omelet"里的"3"，model在latent space里旋转这个数字。这可能是在做arithmetic。

**3. Sliders（滑块）**：某些key token的latent state沿一个方向持续drift。这可能是在"计数"——记录已经iterate了多少次。

**4. Path independence**：不管初始state从哪里出发，最终都converge到同一个fixed point或orbit。模型自己学到了稳定的dynamics。

## Zero-shot就有的bonus能力

因为architecture本身的特性，这个模型天然支持一些normally需要specialized training的能力：

**Adaptive compute**：不同token自动花不同iteration。简单token转2圈就停了，复杂token转20圈。exit criterion是连续两次iteration的KL divergence小于阈值。

**KV-cache sharing**：recurrent的不同iteration之间共享KV cache。设个budget k=16，第17次iteration覆盖第1次的cache。实验证明性能几乎不降。

**Continuous CoT**：下一个token不重新随机初始化latent state，而是用上一个token的final state做warm start。这样latent information能跨token传递。

**Self-speculative decoding**：用小r draft，大r verify。不需要external draft model。

## 我的take

这篇paper真正exciting的不是benchmark数字，而是它demonstrate了一个alternative axis of scaling：

- Scale parameters（LLaMA, GPT路线）
- Scale context / CoT tokens（o1, R1路线）  
- **Scale latent recurrence**（这篇paper的路线）

三条路线可能互补。Recurrent depth的优势是：不需要specialized CoT training data，内存footprint小，能capture non-verbalizable的reasoning（spatial thinking, physical intuition之类）。

劣势是：compute-heavy（same params下FLOPs高几倍），latent reasoning不human-readable（oversight困难），memorization capacity有限。

如果未来把recurrent depth + MoE + RL post-training结合起来，可能是个很有潜力的方向。现在的proof-of-concept已经展现出emergent的orbit dynamics和context-dependent compute allocation，这些都是真正reasoning capacity在latent space涌现的signs。

---

# Latent Reasoning via Recurrent Depth: 深度技术解析

这篇paper探讨了一个非常优雅的idea: 让模型在 **latent space** 中通过反复迭代一个recurrent block来scale test-time compute，完全绕开verbalized chain-of-thought。让我从architectural、training dynamics、emergent behaviors几个层面深入展开。

## 1. 核心Motivation与Paradigm定位

主流reasoning model(o1, R1)依赖 **token-level外显推理**——把中间step用自然语言verbalize到context window里。这有几个fundamental limitations:
- 需要专门的CoT training data
- 长context window内存开销大
- 单步投影到verbal token是information bottleneck (把高维latent thought压缩到discrete token)

Recurrent depth approach的核心insight: **reasoning本质上可以发生在high-dimensional continuous latent space**，通过同一个block的反复apply来实现任意深度的computation chain。这与human cognition有analogy——大脑在说出第一个word之前，neural firing patterns已经反复iterate了。

Reference: [Universal Transformers](https://arxiv.org/abs/1807.03819) 最早提出depth recurrence for transformer；[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) 把recurrence看作fixed-point iteration；[Schwarzschild et al. 2021](https://arxiv.org/abs/2110.10355) 的deep thinking工作证明了recurrence能学到generalizable algorithms。

## 2. Architecture Design的细节

### 2.1 Macroscopic structure: Prelude-Recurrent-Coda

Model被切成三段，公式定义如下:

$$
\mathbf{e} = P(\mathbf{x})
$$
$$
\mathbf{s}_0 \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_{n \cdot h})
$$
$$
\mathbf{s}_i = R(\mathbf{e}, \mathbf{s}_{i-1}) \quad \text{for} \quad i \in \{1, \dots, r\}
$$
$$
\mathbf{p} = C(\mathbf{s}_r)
$$

变量解释:
- $\mathbf{x} \in V^n$: input token sequence, $n$ 是sequence dimension
- $\mathbf{e} \in \mathbb{R}^{n \times h}$: embedded latent representation, $h$ 是hidden dim (5280 for large model)
- $\mathbf{s}_i \in \mathbb{R}^{n \times h}$: 第 $i$ 次recurrent iteration的latent state
- $r$: recurrence count, training时随机采样, test-time可任意大
- $P, R, C$: 分别是prelude, core, coda blocks

Large model shape: $(l_P, l_R, l_C) = (2, 4, 2)$, $h = 5280$, 55 heads × 96 dim, MLP inner dim 17920。

**关键设计决策**: 每次iteration都re-inject embedding $\mathbf{e}$ (via concatenation + adapter matrix $A: \mathbb{R}^{2h} \to \mathbb{R}^h$)。这与deep thinking literature一致 (Bansal et al. 2022)，类比gradient descent: 每一步都需要data $\mathbf{y}$，不能只在initialization提供。

如果只通过 $\mathbf{s}_0 = \mathbf{e}$ 一次性注入，recurrence会变得unstable——solution只依赖boundary condition，无法持续refine。

### 2.2 Sandwich Norm Layout (Microscopic)

每个layer内部使用"sandwich" normalization:

$$
\hat{\mathbf{x}}_l = n_2\left(\mathbf{x}_{l-1} + \text{Attn}(n_1(\mathbf{x}_{l-1}))\right)
$$
$$
\mathbf{x}_l = n_4\left(\hat{\mathbf{x}}_l + \text{MLP}(n_3(\hat{\mathbf{x}}_l))\right)
$$

这里有 **4个norm layers** $n_1, n_2, n_3, n_4$ (都是RMSNorm)。区别于标准pre-norm (只有 $n_1, n_3$) 或post-norm。这种"sandwich"是为了stabilize recurrence。

**Intuition**: 在recurrent setting下，每次iteration会继续mix token维度上的信息。如果没有outer norm ($n_2, n_4$)，magnitude会累积，导致token correlation collapse到1.0 (见paper Figure 5的Bad Run 1)。Sandwich norm强制每层输出magnitude bounded。

Reference: [Gemma 2](https://arxiv.org/abs/2408.00118) 使用类似sandwich策略；[CogView](https://arxiv.org/abs/2104.05858v2) 是最早探讨的。

### 2.3 Initialization细节 (Large scale才critical)

使用Takase et al. 2024的variance scheme:

$$
\sigma_h^2 = \frac{2}{5h}
$$

对outprojection layers:
$$
\sigma_{\text{out}}^2 = \frac{1}{5hl}
$$

其中 $l = l_P + \bar{r} \cdot l_R + l_C = 2 + 32 \times 4 + 2 = 132$ 是effective depth。

初始state $\mathbf{s}_0$ 也用truncated normal, variance $\sigma_s^2 = \frac{2}{5}$。

**关键观察**: outprojection layers init得很小，这避免了early training时residual stream被dominated by random outprojection noise (Goyal et al. 2018的insight)。在3.5B scale下, initialization choices变得critical, small scale下任何sensible init都work。

## 3. Training Objective: Randomized Unrolling

### 3.1 Log-normal Poisson sampling

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \in X} \mathbb{E}_{r \sim \Lambda} L(m_\theta(\mathbf{x}, r), \mathbf{x}')
$$

$\Lambda$ 是log-normal Poisson分布，sampling过程:

$$
\tau \sim \mathcal{N}(\log(\bar{r}) - \frac{1}{2}\sigma^2, \sigma)
$$
$$
r \sim \mathcal{P}(e^\tau) + 1
$$

变量:
- $\bar{r} = 32$: 目标mean recurrence
- $\sigma = \frac{1}{2}$: 标准差
- $\mathcal{P}$: Poisson distribution
- $\mathcal{N}$: Normal distribution

**为什么log-normal Poisson?** 这个分布有两个关键性质:
1. **众数小于均值** ($\bar{r}$), 大部分训练step用较少recurrence
2. **Heavy tail**: 偶尔出现远超 $\bar{r}$ 的大recurrence step

这让模型学会在不同compute budget下都能function, 同时偶尔见习longer unrolling, 为test-time extrapolation打下基础。

### 3.2 Truncated Backpropagation

只backprop through last $k=8$ iterations。Memory和backward compute independent of $r$。这类似于classical RNN的truncated BPTT (Williams & Peng 1990)，但这里recurrence在depth而非time。

**Subtle point**: Prelude block每次iteration都接收gradient (因为 $\mathbf{e}$ 在每步injection)。所以Prelude是fully trained，但recurrent block R只通过最后8次unrolling更新。

Reference: [Truncated BPTT](https://ieeexplore.ieee.org/document/55834) 是90年代RNN训练的标准技巧。

### 3.3 Training Stability Story (Figure 5非常重要)

paper记录了3次run的失败/成功过程:

**Bad Run 1** (orange): parameter-free RMSNorm, 无embedding scale $\gamma$, parameter-free adapter $A(\mathbf{s}, \mathbf{e}) = \mathbf{s} + \mathbf{e}$, LR $4 \times 10^{-4}$
- 现象: token correlation collapse到1.0 (middle plot)
- 机理: recurrence不断mix sequence维度，没有合适的norm/magnitude control导致所有token collapse到同一hidden state
- 结果: training loss stalled

**Bad Run 2** (green): 加入embedding scale, conventional pre-norm, learned adapter
- 现象: 前期token correlation spike到1.0但recover (150 steps后)
- 但val perplexity在 $r=1$ 和 $r=32$ 时一样
- 机理: 模型early on学会了ignore incoming state $\mathbf{s}$, 陷入local minimum
- 结果: 无法利用test-time compute

**Main Run** (blue): revert回sandwich block + drop peak LR到 $4 \times 10^{-5}$
- 永远没接近token correlation 1.0
- $r=32$ 比 $r=1$ 显著更好
- 750B tokens无interruption

**Intuition**: Recurrence是双刃剑。设计稍有不慎，模型就走捷径: "ignore recurrent state, 只用current embedding预测"。这有点类似skip-connection会undermine深度的问题。需要architectural prior + 低LR联合强制模型engagement with recurrent state。

## 4. Benchmark结果细节分析

### 4.1 Standard benchmarks (Table 1)

3.5B params + 800B tokens的Huginn-0125, 在r=32时:

| Benchmark | Huginn r=4 | Huginn r=32 | OLMo-7B (2.5T) |
|-----------|-----------|-------------|----------------|
| ARC-E | 49.07 | 69.91 | 68.81 |
| ARC-C | 27.99 | 38.23 | 40.27 |
| HellaSwag | 43.46 | 65.21 | 75.52 |
| MMLU | 23.39 | 31.38 | 28.39 |
| SciQ | 80.00 | 93.50 | 88.50 |

观察:
- r=4时大幅落后OLMo-7B (远少tokens)
- r=32时overall comparable to first-gen OLMo-7B
- Lags behind later OLMo (0724, OLMo-2)

### 4.2 Math benchmarks (Table 2) - 显著优势

GSM8K CoT (with system prompt, r=32): 34.80/42.08 strict/flexible
MMLU Math reasoning显著超过所有对比 (除OLMo-2)
对比OLMo-7B (2.5T tokens): 6.07/7.28 → Huginn 34.80/42.08

**Intuition**: Recurrence prior + math-heavy data mixture (Figure 4显示math+code占大头) 让模型形成了strong reasoning capability, 但memorization capacity受限于parameter count。

### 4.3 Recurrent vs Non-recurrent twin (Table 4)

这是最key的ablation。同样的3.5B architecture, 同样data, 但non-recurrent (r=1 fixed) trained 180B tokens:

| Model | Tokens | ARC-C | HellaSwag | GSM8K CoT |
|-------|--------|-------|-----------|-----------|
| Fixed-Depth Baseline | 0.18T | 26.96 | 37.34 | 1.82/2.20 |
| Ours, r=32 (early ckpt) | 0.18T | 29.18 | 48.80 | 9.02/10.24 |
| Ours, r=1 (early ckpt) | 0.18T | 23.72 | 29.19 | 0.00/0.15 |
| Ours, r=32 (final) | 0.8T | 38.23 | 65.21 | 34.80/42.08 |
| Ours, r=1 (final) | 0.8T | 24.06 | 29.34 | 0.00/0.00 |

**Critical observation**: 当r=1 evaluation时, 800B checkpoint和180B checkpoint的performance几乎相同。所有gain都encoded在recurrent block iterations中, prelude/coda层不存储reasoning capability。

### 4.4 Context-dependent compute (Figure 9)

ARC-C evaluation, 不同few-shot数量:
- 0-shot: 8-12 iterations saturate
- 1-shot: ~20 iterations
- 25-shot: ~32 iterations
- 50-shot: ~32 iterations

**Intuition**: Recurrence不仅是reasoning深度, 也是context整合能力。More context → more recurrence needed to extract information。

## 5. Zero-shot Inference Features (Section 6 - 亮点)

Recurrent architecture天然支持一些normally需要specialized training的能力:

### 5.1 Adaptive Compute via KL divergence exit

定义exit criterion: 连续两次iteration的KL divergence $< 5 \times 10^{-4}$:
$$
\text{KL}(p_i \| p_{i-1}) < 5 \times 10^{-4}
$$

不同MMLU category的平均exit step分布 (Figure 10):
- High school mathematics: ~5 steps (converge快)
- Moral scenarios: 多3.5 steps (复杂deliberation)
- Philosophy: bimodal distribution

**KV-cache处理**: early-exit会让不同token有不同depth的KV cache entries。Trick是attend to "last, deepest available KV state for each previous token"。因为所有recurrent KV entries由同一组K,V projection matrices产生, 它们"in match", attention仍然有效。

Reference: [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10076) Elbayad et al. 2019 提出类似idea。

### 5.2 KV-cache Sharing

设置KV-cache budget $k$, 在iteration $i$ 时读写cache entry $i \mod k$。例如 $k=16$: 第17次iteration覆盖第1次的cache entry。

这能reduce KV memory。MT-Bench实验 (Table 6): $k=4$ 时5.856, baseline 64-iter是5.693, **性能不降反升** (差异不statistically significant, 但非常remarkable)。

### 5.3 Continuous Chain-of-Thought (warm-start)

不重新sample $\mathbf{s}_0 \sim \mathcal{N}(0, \sigma^2 I)$, 而是warm-start with上一个token的final state $\mathbf{s}_r$:

$$
\mathbf{s}_0^{(t)} = \mathbf{s}_r^{(t-1)}
$$

这样latent information能跨token传递。减少平均收敛step 1-2步。在philosophy类问题尤其明显。

这与[Hao et al. 2024 (Coconut)](https://arxiv.org/abs/2412.06769)的continuous CoT想法类似，但Hao的工作是finetune fixed-depth transformer，这里是pretrain from scratch for recurrence。

### 5.4 Self-Speculative Decoding

不需要external draft model。用 $r=N$ (small) 生成draft tokens, 然后用 $r=M > N$ verify。States computed during drafting不浪费, verify时reuse。

Reference: [Speculative Decoding](https://arxiv.org/abs/2211.17192) Leviathan et al. 2023; [Medusa](https://arxiv.org/abs/2401.10774); [LayerSkip](https://arxiv.org/abs/2404.16710)。

## 6. Latent Space Mechanics (Section 7 - 真正的interpretability金矿)

这是paper最fascinating的部分。通过PCA visualize latent trajectories $\{\mathbf{s}_i\}_{i=1}^r$:

### 6.1 三种emergent patterns

**(1) Fixed-point convergence** (Figure 12 top row)
简单token (e.g., intermediate tokens in trivia question)的latent state直接converge到一个fixed point, trajectory单调decreasing distance to $\mathbf{s}^*$。

**(2) Orbits** (Figure 12 middle row)
对于math/arithmetic tokens, state在多个PCA plane pair上形成 **闭环orbit**。例如Figure 16显示model在"3"这个token上 (Claire makes a 3 egg omelet...), 在6个PCA方向上都rotate。

**Intuition**: 这可能类似[Nanda et al. 2022 (Grokking)](https://arxiv.org/abs/2301.05217)中发现的periodic patterns in arithmetic transformers, 但这里emergent in much broader context (math reasoning, planning tokens like "makes", "thinks")。

**(3) Sliders** (Figure 12 bottom row middle)
对某些key tokens (e.g. "wrong" in unsafe question), trajectory沿单一方向持续drift。这可能用于count iterations——类似[Graves 2017 Adaptive Computation Time](https://arxiv.org/abs/1603.08983)中的halting neuron。

### 6.2 Path Independence

重新initialize多个不同 $\mathbf{s}_0$, 模型converge到相同fixed point或orbit。这是[Anil et al. 2022](https://arxiv.org/abs/2207.04230)定义的path independence。

**Intuition**: 模型implicitly学到了contraction mapping或者类似gradient descent的稳定dynamics。这和Deep Equilibrium Models (Bai et al. 2019)的fixed point formulation直接相关，但这里emergent from training, 没有explicit equilibrium constraint。

### 6.3 Context-dependent convergence speed (Figure 11)

Heatmap visualization: 行是token sequence, 列是recurrent iterations, 颜色是 $||\mathbf{s}_i - \mathbf{s}^*||$。

观察: 
- 关键token (e.g. "wrong" in unsafe question) converge慢
- Identical tokens (e.g. "...")根据context有不同convergence
- Convergence不总是monotonic——有时先diverge再converge (orbit pattern)

**这是重要的interpretability信号**: Model不是uniformly分配compute, 而是基于context-dependent判断哪里需要更多deliberation。

## 7. Connection to broader literature

### 7.1 与Diffusion Models的关系 (Remark 3.1)

Authors试过更类似diffusion的update rule:
$$
\mathbf{s}_i = R(\mathbf{e}, \mathbf{s}_{i-1}) + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma_i I)
$$

发现noise injection没help。也试过step-conditioned:
$$
\mathbf{s}_i = R_i(\mathbf{e}, \mathbf{s}_{i-1})
$$

这interacts badly with path independence, 无法extrapolate到unseen $r$。

**Key difference**: Diffusion models (e.g. [Latent Diffusion](https://arxiv.org/abs/2112.10752))的迭代是per-sequence (固定step数), 这里是per-token可变step数。Diffusion训练surrogate objective (noise prediction), 这里是直接next-token prediction + truncated unrolling。

### 7.2 与Deep Equilibrium Models (DEQ)

[DEQ](https://arxiv.org/abs/1909.01377)直接solve $\mathbf{s}^* = R(\mathbf{e}, \mathbf{s}^*)$ via root-finding (e.g. Anderson acceleration)。Training objective是equilibrium的direct problem。

Huginn的差异: truncated unrolling而非fixed-point solver, training objective是直接optimize不同 $r$ 下的prediction loss。这让model能learn non-equilibrium trajectories (orbits, sliders), DEQ只能find fixed points。

### 7.3 与Energy-Based Models的connection

Recurrent depth可以理解为implicitly学习一个energy function $E(\mathbf{s}, \mathbf{e})$的gradient:
$$
\mathbf{s}_{i+1} = \mathbf{s}_i - \eta \nabla_{\mathbf{s}} E(\mathbf{s}_i, \mathbf{e})
$$

但paper没显式construct energy。这与[LeCun's JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf)和[Amos & Kolter OptNet](https://arxiv.org/abs/1703.03133)的implicit layer思想相关。

## 8. Limitations与Future Directions

### 8.1 Compute economics

Large model 3.5B params, 但effective FLOPs接近32B parameter transformer (at r=32)。能scale到50B-equivalent compute。这意味着same parameter count下, pretraining compute贵4-14倍。

**Trade-off**: Memory footprint小 (3.5B params), 通信开销小 (weight sharing), 但matrix multiply FLOPs高。适合slow interconnect clusters (像Frontier AMD MI250X)。

### 8.2 Training cost

800B tokens, 47,000 steps, 用4096 GPU (Frontier AMD cluster)。Hand-crafted distributed data parallel implementation绕开AMD interconnect issue。Achieve 41%-51% AFU (52-64 TFLOP/s per GPU)。

Reference: [Frontier supercomputer](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) at Oak Ridge National Lab.

### 8.3 Future: Recurrent MoE

Paper的Section 9建议: 把recurrent depth和MoE结合。MoE是parameter-heavy (多active experts), recurrent depth是compute-heavy (少params多FLOPs)。

Recurrent MoE: 同一expert可被多次激活, 然后switch to different expert。这接近[Tan et al. 2023 Sparse Universal Transformer](https://arxiv.org/abs/2310.07096)和[MoEUT (Csordás et al. 2024)](https://arxiv.org/abs/2405.15075)的设计。

### 8.4 Internalizing CoT

Future work提到: 用RL或finetune把verbalized CoT的reasoning internalize到recurrence里。[Deng et al. 2024](https://arxiv.org/abs/2405.14838)已经exploring这条line。

## 9. 我的interpretation与open questions

**Why this might work fundamentally**: 

Standard transformer的depth是离散选择。要让模型"think harder"就必须增加参数或增加context tokens。Recurrent depth让model学到了一个 **iterative operator** $R$, 它能在latent space中实现任意深度的computation chain。

更深层的connection: 这类似programming中的recursion vs iteration。Fixed-depth transformer是固定行数的imperative code, recurrent depth是while loop with conditional termination。前者capacity bound by syntactic complexity, 后者capacity bound by termination condition + per-iteration computation。

**几个我特别好奇的方向**:

1. **Per-token different recurrence**: 当前training用locked-step sampling (整个micro-batch同一r)。如果能per-token variable r, 是否能让model学到更精细的adaptive compute?

2. **Long-horizon coherence in latent CoT**: Continuous CoT (warm-start)能跨token传递latent state。这能否扩展到极长generation? 类似 [Quiet-STaR](https://arxiv.org/abs/2403.09629)但是latent版本?

3. **Connection to in-context learning**: Recurrence的"orbit" behavior可能是ICL的latent analog。固定depth transformer用attention retrieve examples, recurrent depth用iterative refinement over examples。这是complementary mechanisms还是overlapping?

4. **Reasoning vs Memorization trade-off**: Table 5显示recurrent model在open-book (提供fact)时大幅超越closed-book, 与standard model的gap更显著。这暗示recurrence prior导致memorization被sacrificed for reasoning。能否通过MoE-style memory modules补回memorization?

5. **RL on recurrence**: 能否用RL让model学会"何时停止recurse"? 类似[Process Reward Models](https://arxiv.org/abs/2306.17492)的idea, 但在latent recurrence上?

**Resources**:
- [HuggingFace Model](https://huggingface.co/tomg-group-umd/huginn-0125)
- [GitHub Repo](https://github.com/seal-rg/recurrentpretraining)
- [Universal Transformers paper](https://arxiv.org/abs/1807.03819)
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)
- [Coconut: Continuous CoT](https://arxiv.org/abs/2412.06769)
- [Schwarzschild Deep Thinking](https://arxiv.org/abs/2110.10355)
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.07024)
- [Path Independent Equilibrium Models](https://arxiv.org/abs/2207.04230)

这个approach的真正美感在于: 它把test-time scaling从discrete token generation维度解放出来, 引入了continuous latent computation维度。如果scaled up + combined with MoE + good post-training, 可能是LLM architecture的一个fundamental新axis。当前proof-of-concept已经展现出emergent orbit dynamics和context-dependent compute allocation, 这些都是signs of真正reasoning capacity的emergence。
