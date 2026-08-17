---
source_pdf: The Art of Scaling Reinforcement Learning.pdf
paper_sha256: ddb5eb8f43fa3b3dbb0f7fa14cdf16eb170cee919c0713c7e38aabfa7647c441
processed_at: '2026-08-12T13:55:25-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲 ScaleRL

## 一句话总结

把 pre-training 那套 "small run extrapolate 到 big run" 的 scaling law 思路搬到 RL post-training 上，发现 sigmoid 比 power-law 好拟合，然后用 400k GPU-hours 的 ablation 找出哪些 design choice 真正 shift ceiling、哪些只是调 efficiency，最后组合成 ScaleRL 这个 recipe，成功在 100k GPU-hours 上 extrapolate 预测成功。

## 为什么 pre-training 早就有 scaling law 而 RL 一直没有

Pre-training 的 loss 是 unbounded metric，perfect power-law fit，Kaplan 2020 那篇开始大家就知道用 1/10 compute 的小 run 能预测 10x compute 的大 run。

RL post-training 的 reward 是 [0,1] bounded metric，会 saturate。你硬上 power-law 拟合会发现预测 $A=1.0$（明显错，因为 metric bounded），换不同 fit regime 给出的 $A$ 从 0.65 跳到 0.74 - 完全没法预测。没有 predictable scaling framework 的后果就是：没法 cheaply 筛 RL 算法，每次想确认一个 idea 好不好都得跑大 scale，academia 玩不起。

## Sigmoidal 曲线：三个参数直觉

$$R_C - R_0 = \frac{A - R_0}{1 + (C_{mid}/C)^B}$$

想象成一条 S 形曲线，横轴 log compute，纵轴 reward：
- **$A$**：天花板，compute 无穷大时能爬到多高
- **$B$**：陡峭度，$B$ 大 = 曲线陡 = compute 效率高
- **$C_{mid}$**：曲线中点位置，$C_{mid}$ 小 = 早早就爬到一半，表示效率高

设计 algorithm 时优先级是：**先把 $A$ 抬上去，再 optimize $B$**。Figure 13b 那个例子很直观 - 一个效率低但 $A$ 高的 method 最终会超过效率高但 $A$ 低的 method，这是 bitter lesson 在 RL 的体现。

## 关键 design choices 的直觉

### PipelineRL vs PPO-off-policy
PPO-off-policy 是"batch 模式" - generator 生成一整批 rollout，trainer 吃完整批再 update。staleness 由 $k$ 控制，$k=8$ 意味着每批 rollout 被用 8 次 update。

PipelineRL 是"streaming 模式" - generator 一边生成一边吐 rollout 给 trainer，trainer 一更新就立刻 push 新 weight 给 generator，generator 用 stale KV cache 但 updated weights 继续 generate。这等价于 staleness 永远在 0~8 之间动态变化，平均下来比 PPO-off-policy-8 更接近 on-policy。

这个 choice 罕见地同时 shift $A$ 和 $B$ - 大多数 choice 只动其中一个。

### CISPO vs DAPO vs GSPO
DAPO 把 importance ratio $\rho$ 直接乘进 surrogate loss，$\rho$ 一大就 gradient explosion，对 $\epsilon_{max}$ 极度敏感（Figure 19a 显示 $\epsilon_{max}$ 从 0.2 调到 0.28 会让 $A$ 漂移）。

CISPO 的本质是把 $\rho$ 降级成一个 stop-gradient 的 mask：
$$\mathbf{sg}(\min(\rho, \epsilon)) \cdot \hat{A} \cdot \log \pi_{train}$$
$\rho$ 决定该 token 的 gradient 是否被 truncate，但 $\rho$ 自己不传 gradient。真正的 gradient 信号来自 $\log \pi_{train}$（vanilla policy gradient）。这相当于"只 truncate 不放大"，hyperparameter 怎么调都稳。在 Figure 5a 里 CISPO 的 $A=0.61$，DAPO 只有 $0.52$。

GSPO 是 sequence-level IS ratio，长 sequence 会 compound overflow。性能接近 CISPO 但在 Scout MoE 上不稳，作者最后选 CISPO。

### FP32 at LM head
Generator 跑 inference kernel（vLLM 那种），trainer 跑 training kernel（FSDP），两套代码数值上有 tiny mismatch。这个 mismatch 在 hidden states 里不明显，但经过 LM head 的 softmax 后，logits 一除就把 $\rho$ 推偏。$\rho$ 偏离 1 并非 policy 真的变了，而是 numerical noise。

RL 对 $\rho$ 敏感到病态 - 训练会把这种 noise 当真实 policy change 来拟合。FP32 在 LM head 上消除这个 noise，$A$ 从 0.52 跳到 0.61。单看公式没任何 deep math，就是 numerical hygiene，但效果巨大。

### Loss aggregation: prompt-level 最好
Sample-level average 是先每个 completion 内部 token 平均，再 sample 之间平均。短 completion 内部平均后值稳定，会 dominate loss。Reasoning 任务里长 completion 往往是 hard problem，被边缘化 hurt performance。

Prompt-level average 就是 batch 里所有 token 直接平均，每个 token 等权重，长 completion 不被惩罚。

### Zero-variance filtering
Batch 里某些 prompt 的 16 个 generation 全对或全错，std=0，advantage=0，gradient=0。把这些 prompt 直接 drop 掉（不 resample 补充），剩下的都是"model 在这个 prompt 上有点会又有点不会"的有信号 prompt。Effective batch 变小但 gradient 质量高，$A$ 提升。

### No-Positive-Resampling
观察：prompt 一旦 pass_rate ≥ 0.9 就一直是简单的（policy 已经学会）。这种 prompt 后续 epoch 还采样就是浪费 compute。直接从 future epoch 永久移除。简单但有效，$A$ 提升。

## ScaleRL 的 leave-one-out 验证

ScaleRL 是上述所有 best choice 组合。LOO 实验从 ScaleRL 出发，每次 revert 一个 component 看 $A$ 变化。

发现：**几乎每个 component 的移除都不显著影响 $A$**（都在 ±0.02 误差内）。但是当你把 $A$ 固定下来重新拟合，看 efficiency $B$，每个 component 移除都让 $B$ 下降。

作者的解读：ScaleRL 的 strength 来自 cumulative effect，components 之间互相 compensate。FP32 在 ScaleRL 内部看起来 redundant，但放在 GRPO 上立刻 $A$ 飙升；CISPO 在 ScaleRL 内部看起来和 DAPO 差不多，但 CISPO 对 $\epsilon_{max}$ robust，DAPO 调一下 $\epsilon_{max}$ 整个 $A$ 都漂移。

换句话说，每个 component 都解决某类 instability source，组合起来对各种"未知 training regime"robust。即使当前 setup 看起来不需要，换到 MoE、换到 multi-task、换到 longer context 时可能就关键了。

## Sigmoidal 和 Power-law 在 high-compute regime 等价

把 sigmoid 在 $C \gg C_{mid}$ 处 Taylor 展开：
$$R_C \approx A - \frac{(A-R_0)C_{mid}^B}{C^B} = A - \frac{D}{C^B}$$
就是 power-law。所以 sigmoid 是 power-law 的"全局化版本" - 低 compute 不发散，高 compute 和 power-law 重合。这解释了为什么 pre-training 用 power-law work（一直在 high-compute regime），而 RL 需要从 low-compute extrapolate 到 high-compute 必须用 sigmoid。

## 几个反直觉的发现

1. **Generation length 14k → 32k：early 慢，但 $A$ 更高**（0.610 → 0.645）。Long context 是 ceiling-raising knob 不是 efficiency trade-off。Model 需要先学会"怎么用更多 token budget"。

2. **Batch 768 → 2048：early 停滞在 downstream，但 final performance 更高**。大 batch 等效低 noise gradient，需要更多 compute 启动但天花板高。

3. **Generations per prompt 8/16/24/32 在 fixed batch 下基本无差**。这个维度是 second-order choice，别浪费时间 tune。

4. **Entropy 不是 exploration 的 reliable proxy**。Batch 768 vs 2048 有 identical entropy trajectory 但 performance 差很大。Cui et al. 2025 那套用 entropy 当 exploration metric 的做法这 paper 直接打脸。

5. **Truncation rate > 10% 是 instability warning**。ScaleRL 能稳到 100k GPU-hours，核心原因之一是 truncation 一直 <5%。Model scale 越大 truncation 越低（Scout MoE <1%），大模型 instruction-following 强，interruption signal 更有效。

## Takeaway

**对 algorithm researcher**：用 sigmoidal fit + early extrapolate 来 cheaply 筛 scalable idea，不用每次都跑 100k hours。跑前 8k hours fit 一下就能预测 16k hours 的 performance。

**对 practitioner**：用 ScaleRL recipe（PipelineRL-8 + CISPO + FP32 head + prompt-level loss + batch-norm advantage + zero-variance filter + no-positive-resample + interruptions）。这个组合 stable scalable，在 8B/Scout MoE/math+code/long context 都验证过 predictable。

**对 scaling law researcher**：下一步是 derive compute-optimal allocation law for RL - generation length vs batch size vs epochs vs model size 的最优分配。这 paper 提供 framework 但没 solve 这个 allocation problem，类似 Chinchilla 之前的 Kaplan - 只给出 scaling 形式没给出最优 allocation。

代码在 https://www.devvrit.com/scalerl_curve_fitting 开源，可以拿去 fit 自己的 RL runs。

---

# The Art of Scaling RL Compute for LLMs - 深度技术解读

Andrej, 这篇paper本质上是在做RL scaling这件事的"Chinchilla时刻" - 把pre-training scaling law的predictability方法论搬到RL post-training。它由Meta和UT Austin的团队完成，消耗400k+ GPU-hours(Nvidia GB200)，提出ScaleRL recipe并验证可以predictable scaling到100k GPU-hours。

---

## 1. 核心问题与动机

pre-training时代我们已经有了power-law scaling laws (Kaplan et al. 2020 https://arxiv.org/abs/2001.08361, Hoffmann et al. 2022 https://arxiv.org/abs/2203.15556)，可以从small-compute runs extrapol预测large-compute performance。但RL post-training领域没有这种principled framework，新算法 (DAPO, GSPO, CISPO等)都是isolated empirical studies，缺乏预测scalability的方法。

DeepSeek-R1-Zero用了100k H800 GPU-hours (pre-training compute的3.75%)，o1→o3的RL compute增加10x+，Grok-3→Grok-4也是类似leap (xAI https://x.ai/news/grok-4)。compute成本爆炸，没有scaling methodology让academia几乎无法参与frontier RL研究。这paper就是要把RL scaling从"art"变成"science"。

---

## 2. Sigmoidal Scaling Law - 公式与变量解析

### 2.1 核心公式 (Equation 1)

$$R_C - R_0 = \frac{A - R_0}{1 + (C_{mid}/C)^B}$$

变量含义：
- $R_C$: 在compute budget $C$ (GPU-hours)下，model在i.i.d. validation set上的expected reward (这里用mean@16 pass rate)
- $R_0$: training开始时的初始reward (compute极小时的baseline performance)
- $A \in [0, 1]$: **asymptotic reward** - compute无穷大时能达到的performance ceiling
- $B > 0$: **scaling exponent** - 决定compute efficiency，控制曲线陡峭度
- $C_{mid}$: **midpoint compute** - 达到一半gain $(A-R_0)/2$ 所需的compute，小值意味着更快接近asymptote
- $C$: 训练compute budget (通常以GPU-hours计)

### 2.2 为什么用sigmoidal而不是power-law?

pre-training用power-law $R_C = A - D/C^B$ (在$C \geq C_0$ regime)。但作者发现RL用sigmoidal更好，三个理由：

1. **Bounded metrics**: accuracy/reward在$[0,1]$内，sigmoid natural captures saturation (Ruan et al. 2024 https://arxiv.org/abs/2405.10938; Srivastava et al. 2022 https://arxiv.org/abs/2206.04615)
2. **数据稀疏**: RL training只跑~75 eval points (每100 steps一次)，舍弃early points损失太多。Power-law必须舍弃$C < C_0$部分
3. **Extrapolation robustness**: power-law fit在(1.5k, 50k) GPU-hours regime预测$A=1.0$ (明显错误，实际~0.65)；sigmoidal预测$A=0.645$且在不同fitting regime下稳定(0-100k: A=0.655, 0-50k: A=0.645, 5k-50k: A=0.645)

**直觉联系**: sigmoidal在high-compute regime ($C \gg C_{mid}$) Taylor展开正好退化成power-law形式：

$$R_C \approx R_0 + (A - R_0)(1 - C_{mid}^B/C^B) = A - \frac{(A-R_0)C_{mid}^B}{C^B} = A - \frac{D}{C^B}$$

其中 $D = (A-R_0)C_{mid}^B$。这意味着sigmoidal是power-law的"全局化"版本 - 在低compute regime不发散，在高compute regime与power-law一致。

### 2.3 拟合方法
grid search over $A \in \{0.450, 0.455, ..., 0.800\}$ 和 $C_{mid} \in [100, 40000]$，对每个$(A, C_{mid})$用scipy curve_fit拟合$B$。所有runs从前~1.5k GPU-hours后开始fitting (相当于1 epoch后)，因为早期有rapid linear增长不符合sigmoidal。

误差margin: 跑3次独立ScaleRL (batch 768, gen length 14k)，A的variance为$\pm 0.015$，作者采用$\pm 0.02$作为误差margin。代码开源于 https://www.devvrit.com/scalerl_curve_fitting。

### 2.4 参数interpretation (Figure 3, 12, 13)

- **$B$ 和 $C_{mid}$ 影响 efficiency** (曲线多快爬升)
- **$A$ 影响 ceiling** (最终能达到多高)
- 设计时优先考虑**raise $A$**，然后才optimize $B$或$C_{mid}$。这点在Figure 13b中很清楚：一个inefficient但高$A$的方法最终会超过efficient但低$A$的方法 - 这是"bitter lesson"的体现。

---

## 3. RL Setup与Base Algorithm

### 3.1 Generator-Trainer Split Architecture

80 GB200 GPU分成：
- **64 generators**: 用optimized inference kernels高throughput生成rollouts，参数$\pi_{gen}^{\theta_{old}}$
- **16 trainers**: FSDP backend做policy update，参数$\pi_{train}^{\theta}$

两套模型同一参数但分布在不同GPU上，这是asynchronous RL的核心设计，generator和trainer通过参数broadcast同步。

### 3.2 Training Regimen

- Sequence length 16,384 tokens = 12,288 thinking + 2,048 solution + 2,048 prompt
- 用``special tokens包裹thinking
- Batch size 768 = 48 prompts × 16 generations per prompt
- Dataset: Polaris-53K (https://hkunlp.github.io/blog/2025/Polaris) 数学题
- 1,000 prompts held out做validation，每100 step用16 generations per prompt测mean@16

### 3.3 Base RL Algorithm (GRPO-like)

对每个prompt $x$，old policy生成$G$个completions $\{y_i\}_{i=1}^G$，每个reward $r_i$。

**Advantage computation**:
$$\hat{A}_i = r_i - \text{mean}(\{r_j\}_{j=1}^G), \quad \hat{A}_i^G = \hat{A}_i / (\text{std}(\{r_j\}_{j=1}^G) + \epsilon)$$

- $\hat{A}_i$: raw advantage (centered by group mean)
- $\hat{A}_i^G$: group-normalized advantage (除以group std)
- $G$: group size (这里=16)
- $\epsilon$: 小正数防除零

**Token-level importance sampling ratio**:
$$\rho_{i,t}(\theta) = \frac{\pi_{train}^{\theta}(y_{i,t}|x, y_{i,<t})}{\pi_{gen}^{\theta_{old}}(y_{i,t}|x, y_{i,<t})}$$

- $y_{i,t}$: 第$i$个completion的第$t$个token
- $y_{i,<t}$: 第$i$个completion前$t-1$个tokens (context)
- 分子: new policy给该token的概率
- 分母: old (generation) policy给该token的概率

**Asymmetric clipping** (DAPO风格):
$$\text{clip}_{\text{asym}}(\rho, \epsilon^-, \epsilon^+) = \text{clip}(\rho, 1-\epsilon^-, 1+\epsilon^+)$$

- $\epsilon^-$: lower clip threshold (typically 0.2)
- $\epsilon^+$: upper clip threshold (DAPO建议0.28，做"clip-higher"防entropy collapse)

**Surrogate objective** (Equation 3):
$$\mathcal{L}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min(\rho_{i,t}\hat{A}_i^G, \text{clip}_{\text{asym}}(\rho_{i,t})\hat{A}_i^G)\right]$$

- $|y_i|$: 第$i$个completion的长度
- 外层$1/G$: sample-level averaging
- 内层$1/|y_i|$: token-level averaging within sample
- $\min$: PPO-style clip防step过大

注意这里**没有KL regularization** - 大scale training reports (Magistral, MiniMax-M1)都不用KL，这点和早期RLHF不同。

### 3.4 Length Control via Interruptions

为防止generation长度爆炸，用forced interruptions: 在[10k, 12k]随机位置插入end-of-thinking phrase `"Okay, time is up. Let me stop thinking and formulate a final answer now. "`，强迫model结束thinking给出answer。这比length penalty更直接，作者在Section 4证明interruptions > length penalty。

---

## 4. 关键Design Choices的Ablation

### 4.1 Asynchronous RL Setup (Section 3.1)

两种off-policy范式：

**PPO-off-policy-k** (Qwen3, ProRL用):
- 一次性生成$B$个prompts的rollouts
- 分成$k$个mini-batch做$k$次update，每个mini-batch $B̂ = B/k$ prompts
- 实验中$B̂=48$，$k \in \{1, 8\}$，所以$B = 48k$

**PipelineRL-k** (Magistral用, Piche et al. 2025 https://huggingface.co/blog/ServiceNow/pipelinerl):
- Generators持续streaming生成rollouts
- 一旦trainer完成一个update，新参数立即push回generator
- Generator继续生成with **updated weights但stale KV cache** from old policy
- 一旦一个batch完成，立即传给trainer
- $k$: trainer最多领先generator $k$步

**结果** (Figure 4):
- PipelineRL: 同样$A$但$B$大幅提升 → compute efficiency高
- PipelineRL-8 vs PipelineRL-4: 类似性能，选8
- 为什么？PipelineRL更接近on-policy，staleness更少

**直觉**: PPO-off-policy的trainer完全依赖stale rollouts (整个batch)，而PipelineRL是streaming - generation和training overlap，feedback loop紧得多。这个选择甚至能shift asymptote $A$，是少数能改$A$的design choice之一。

### 4.2 Loss Type (Section 3.2)

比较三个loss:

**DAPO** (token-level IS, asymmetric clip):
就是上面base algorithm的形式。

**GSPO** (Group Sequence Policy Optimization, Zheng et al. 2025 https://arxiv.org/abs/2507.18071):
序列级IS ratio:
$$\rho_i(\theta) = \frac{\pi_{train}(y_i|x, \theta)}{\pi_{gen}(y_i|x, \theta_{old})} = \prod_{t=1}^{|y_i|} \rho_{i,t}(\theta)$$

整个completion的概率比，而非token-level。这意味着长sequence的IS ratio会compound exponentially (长序列ρ容易overflow/underflow)。但sequence-level更principled，因为reward也是sequence-level给的。

**CISPO** (Clipped IS + vanilla PG, MiniMax-M1 https://arxiv.org/abs/2506.13585, 基于Ionides 2008 truncated importance sampling):
$$\mathcal{L}_{\text{CISPO}}(\theta) = \mathbb{E}\left[\frac{1}{T}\sum_{i=1}^G \sum_{t=1}^{|y_i|} \mathbf{sg}(\min(\rho_{i,t}, \epsilon_{max}))\hat{A}_i \log \pi_{train}(y_{i,t}|x, y_{i<t}, \theta)\right]$$

- $T = \sum_i |y_i|$: batch中总token数
- $\mathbf{sg}$: stop-gradient (clip只做mask，不传gradient)
- $\epsilon_{max}$: upper clip threshold (typically 4-5，无lower clip)
- $\hat{A}_i$: 未经group std归一化的advantage
- 关键: ρ不做乘法进入gradient，只做stop-gradient mask

**直觉差异**:
- DAPO/GSPO: ρ直接进入surrogate objective参与gradient
- CISPO: ρ只做truncation mask (stop-gradient)，真正的gradient信号来自$\log \pi_{train}$ (vanilla policy gradient)

这意味着CISPO更稳定 - 即使ρ很大也不会放大gradient，只是简单地drop这些samples。

**结果** (Figure 5a):
- DAPO: $A \approx 0.52$
- GSPO: $A \approx 0.60$
- CISPO: $A \approx 0.61$, prolonged near-linear reward increase

CISPO + GSPO大幅超过DAPO。CISPO略胜GSPO。

### 4.3 FP32 Precision at LM Head (Section 3.2)

由He & Thinking Machines Lab (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 提出，MiniMax-M1首次应用。

**问题**: generator用inference kernels (e.g. vLLM)，trainer用training kernels (e.g. FSDP)，两套代码在numerical precision上微小差异。这个差异在LM head (final logits projection)被放大，因为softmax会magnify small differences → IS ratio $\rho_{i,t}$不稳定。

**Fix**: 在generator和trainer的LM head都用FP32计算logits。

**结果** (Figure 5b): $A$从0.52 → 0.61，巨大提升。

**直觉**: RL训练对IS ratio极度敏感。如果$\pi_{gen}$和$\pi_{train}$在数值上不一致，ρ偏离1不是因为policy真的变化了，而是因为kernel numerical noise - 这会导致spurious gradient signal。FP32 fix消除这个noise，让ρ真实反映policy change。

### 4.4 Loss Aggregation (Appendix A.9, Figure 14a)

三种aggregation:

**Sample average** (GRPO):
$$\mathcal{L} = \frac{1}{G}\sum_i \frac{1}{|y_i|}\sum_t \ell_{i,t}$$
每个sample先内部token average，再sample间average。短sequence被相对over-weighted (因为内部平均后值更稳定)。

**Prompt average** (DAPO):
$$\mathcal{L} = \frac{1}{\sum_i |y_i|}\sum_i \sum_t \ell_{i,t} = \frac{1}{T}\sum_i \sum_t \ell_{i,t}$$
每个prompt内的token等权重 (不管哪个sample)。等价于token-level average within prompt。

**Token average**:
$$\mathcal{L} = \frac{1}{T}\sum_i \sum_t \ell_{i,t}$$
所有token直接平均。

**结果**: Prompt average最优。

**直觉**: Sample average惩罚长completion (内部平均后短completion主导)。Prompt average让每个prompt的每个token等权重，避免长sequence被边缘化。在reasoning任务中长sequence通常包含更复杂的reasoning，over-penalizing它们会hurt performance。

### 4.5 Advantage Normalization (Appendix A.9, Figure 14b)

三种normalization:

**Prompt-level** (GRPO):
$$\hat{A}_i^G = \hat{A}_i / (\text{std}(\{r_j\}_{j=1}^G) + \epsilon)$$
每个prompt的G个generations内部std归一化。

**Batch-level** (Hu et al. 2025, Rastogi et al. 2025):
$$\hat{A}_i^{\text{batch}} = \hat{A}_i / \hat{A}_{\text{std}}$$
其中$\hat{A}_{\text{std}}$是整个batch所有$\hat{A}_i$的std。

**No normalization** (Dr. GRPO, Liu et al. 2025 https://arxiv.org/abs/2503.20783):
$$\hat{A}_i^{\text{raw}} = r_i - \text{mean}(\{r_j\})$$
只用mean-centering，不做variance scaling。

**结果**: 三者性能类似，batch-level略微好且理论更sound (避免不同prompt间scale不一致)。

**直觉**: Prompt-level normalization的问题在于不同prompt的"难度"不同 - 简单prompt所有generations都对(reward都=1，std=0 → 触发zero-variance filtering)，难prompt所有都错(reward都=0，同样std=0)。Batch-level把所有prompt的variance信息整合，提供更稳定的scale。

### 4.6 Zero-Variance Filtering (Figure 6a)

**观察**: batch中一些prompt的$G$个generations reward全相同 (全对或全错)，这些prompt的std=0 → advantage=0 → gradient=0。

**比较**:
- Default: 保留这些prompt (浪费batch capacity但不影响gradient)
- Effective batch: 过滤掉这些prompt，只对non-zero variance prompt算loss

**注意**: 这和DAPO的**dynamic sampling**不同。Dynamic sampling会resample更多prompt直到batch填满，effective batch只是drop不resample。

**结果**: Effective batch (zero-variance filtering)的$A$更高。

**直觉**: 过滤后剩下的都是有"learning signal"的prompt (有对有错，model能区分)。Effective batch size虽然变小但gradient quality高。

### 4.7 No-Positive-Resampling (Adaptive Prompt Filtering, Figure 6b)

来自An et al. 2025 (Polaris)的观察: **一旦一个prompt对model变得太简单，它就一直是简单的**。如果pass_rate $\geq 0.9$，从future epoch永久移除该prompt。

**结果**: 提升$A$和scalability。

**直觉**: 类似curriculum learning但更aggressive。这些"easy"prompt不再提供有用gradient signal (因为zero-variance概率高)，但消耗compute。永久移除让compute集中在model还学不会的prompt上。

---

## 5. ScaleRL Recipe

### 5.1 Final Recipe组合

基于以上ablations，组合成ScaleRL:

$$\mathcal{L}_{\text{ScaleRL}}(\theta) = \mathbb{E}_{x \sim D_g}\left[\frac{1}{\sum_g |y_g|}\sum_{i=1}^G \sum_{t=1}^{|y_i|} \mathbf{sg}(\min(\rho_{i,t}, \epsilon))\hat{A}_i^{\text{norm}} \log \pi_{train}^{\theta}(y_{i,t})\right]$$

with conditions:
$$\hat{A}_i^{\text{norm}} = \hat{A}_i / \hat{A}_{\text{std}}, \quad 0 < \text{mean}(\{r_j\}) < 1, \quad \text{pass\_rate}(x) < 0.9$$

完整components:
1. **PipelineRL-8** (asynchronous off-policy)
2. **Forced length interruptions** (token 10k-12k插入end-of-thinking)
3. **CISPO loss** (truncated IS + vanilla PG)
4. **Prompt-level loss averaging**
5. **Batch-level advantage normalization**
6. **FP32 precision at LM head**
7. **Zero-variance filtering**
8. **No-Positive-Resampling**

### 5.2 Leave-One-Out (LOO) Ablations

为了验证组合中每个component都贡献，跑LOO: 从ScaleRL出发，每次revert一个component到baseline。每个run 16k GPU-hours，用前8k fit extrapolate到16k。

**Figure 7结果**: 大部分LOO变体达到similar $A$ (在$\pm 0.02$误差内)，所以作者用power-law form凸显efficiency差异:

$$\mathcal{F}(R_c) = C^B, \quad \mathcal{F}(R_c) = \frac{C_{mid}^B}{(A-R_0)/(R_c-R_0) - 1}$$

plot $\log \mathcal{F}(R_c)$ vs $\log C$，slope直接是$B$。ScaleRL $B$最高。

**关键发现**: 在ScaleRL内部，单个component的remove对$A$影响小 (因为其他component compensate)，但对efficiency ($B$)有明显影响。

### 5.3 重要的"为什么这些choice还是值得保留"

即使单个component看似redundant in ScaleRL组合，作者argue应该保留，因为:

1. **FP32**: 在GRPO/DAPO上提供大gain，在Scout MoE上提供scaling boost (Figure 8b)。受益不局限于ScaleRL。
2. **CISPO**: 对$\epsilon_{max}$超参数robust (Section A.17.2)。DAPO的$\epsilon_{max}$变化会fundamentally shift $A$ (Figure 19a)，CISPO在{4, 5, 8}范围内稳定。

**General principle**: 单个choice提供stability/robustness，跨model/setup generalize。

---

## 6. Multi-Axis Scaling验证

### 6.1 Model Scale: Scout MoE (Llama-4 17B×16)

50k GPU-hours训练，fit前16k extrapolate到45k (Figure 1)。

**结果**:
- 同样predictable scaling
- $A = 0.710$ (vs 8B的0.645) - 更大模型更高ceiling
- 1/6 compute就超过8B的最终performance

**直觉**: 更大模型reasoning能力更强，相同RL compute能unlock更多capability。这点和pre-training scaling一致。

### 6.2 Generation Length: 14k → 32k (Figure 9)

- Early: 低$B$, 高$C_{mid}$ (效率低，慢启动)
- Late: $A$从0.610 → 0.645
- Long context是**ceiling-raising knob**而非efficiency trade-off

**Table 1数据**:
| Run | $C_{mid}$ | $B$ | $A$ |
|---|---|---|---|
| ScaleRL (14k) | 2542 | 1.92 | 0.610 |
| ScaleRL-32k | 11272 | 1.89 | 0.645 |

$C_{mid}$从2542涨到11272 - 需要4x compute才达到half gain。但$A$涨了0.035。

**直觉**: 长context允许更complex的reasoning chain。短context下model被迫早terminate thinking，complex problem解不出。长context一开始慢是因为需要先learn"如何使用更多token budget"。

### 6.3 Global Batch Size: 768 → 2048 (Figure 10)

- Small batch: early stage快，但downstream benchmark停滞
- Large batch: 慢启动，但$A$更高，downstream transfer更好

**Table 1**:
| Run | $C_{mid}$ | $B$ | $A$ |
|---|---|---|---|
| ScaleRL-bs512 | 2818 | 1.77 | 0.605 |
| ScaleRL-bs768 | 2542 | 1.92 | 0.610 |
| ScaleRL-bs2048 | 10909 | 1.70 | 0.645 |

2048的$C_{mid}$是768的4x多 - 需要更多compute达到half gain，但$A$更高。在100k run中，batch 2048既稳定又extrapolate到100k点。

**直觉**: 大batch类似pre-training中的large batch regime (Chinchilla) - gradient noise小，更新方向更稳定，但每step compute多。Small batch有high variance，可能在local minimum附近震荡。

### 6.4 Generations per Prompt (Figure 17)

Fixed total batch下，sweep generations per prompt: {8, 16, 24, 32}，调整prompt数量保持batch总size。

**结果**: $A$, $B$, $C_{mid}$基本不变。在moderate batch size下是second-order choice。

**Table 1**:
| Run | $C_{mid}$ | $B$ | $A$ |
|---|---|---|---|
| ScaleRL-8gen | 3054 | 2.44 | 0.585 |
| ScaleRL-16gen | 2542 | 1.92 | 0.610 |
| ScaleRL-24gen | 2936 | 2.07 | 0.595 |
| ScaleRL-32gen | 4242 | 1.65 | 0.595 |

$A$基本一样 (0.585-0.610在误差内)。但$B$有差异 - 16gen最高效率。可能太大batch下，更多generations per prompt反而引入更多noise。

### 6.5 Multi-task: Math + Code (Figure 11)

Joint training math (Polaris-53K) + code (Deepcoder, Luo et al. 2025 https://www.together.ai/blog/deepcoder)。

**Table 1**:
| Run | $C_{mid}$ | $B$ | $A$ |
|---|---|---|---|
| ScaleRL math+code, math | 2896 | 2.05 | 0.595 |
| ScaleRL math+code, code | 1675 | 1.09 | 0.615 |

每个domain独立sigmoidal scaling，extrapolation都predictable。Code的$B$低(1.09)说明code task收敛慢 - code generation需要更多diversity。

---

## 7. 与现有Recipes对比 (Figure 2)

对比四个popular recipes:
- **DeepSeek (GRPO)**: GRPO loss, sample-average, PPO-off-policy-8. 6k GPU-hours后unstable (truncations爆炸)
- **Qwen-2.5 (DAPO)**: DAPO loss ($\epsilon_{min}=0.2, \epsilon_{max}=0.26$), prompt-average, PPO-off-policy-8
- **Magistral** (https://arxiv.org/abs/2506.10910): 类似DAPO但用PipelineRL
- **MiniMax-M1**: CISPO + FP32 + PPO-off-policy + prompt-average

ScaleRL $A=0.61$最高。MiniMax次之 (因为也用CISPO+FP32)。

**直觉**: ScaleRL的advantage来自combination - 单看每个component都不是新发明，但组合并validate后形成best practice。

---

## 8. Stability Insights (Section A.15)

### 8.1 Truncation Rate作为instability signal

- 8B模型batch 768: truncation >10-15%时training destabilize
- ScaleRL 8B: 90%+ time truncation <5%
- ScaleRL batch 2048: 略高，偶尔~7% (因为generation更长)
- ScaleRL 32k context: spike到4%后迅速降到<2%
- Scout MoE: 90%+ time <1%，最大<2% (大模型instruction-following更强)

**实用建议**: 监控truncation rate，>10%是instability warning signal。Generation length budget要足够，且model要能follow interruption signal。

### 8.2 Entropy作为predictor的失败 (Section A.12)

Cui et al. 2025 (https://arxiv.org/abs/2505.22617) 提出entropy作为exploration的proxy。但作者发现:

- Batch 768 vs 2048: entropy轨迹几乎相同 (Figure 16)
- 但batch 2048的downstream performance远超768

**结论**: Entropy不是reliable predictor。Larger batch per step exploration少 (per sample)，但aggregate over more samples，最终performance更好。这意味着不能简单靠"维持高entropy"来ensure exploration。

---

## 9. 与ProRL和LitePPO对比 (Section 6)

**ProRL** (Liu et al. 2025a https://arxiv.org/abs/2505.24864):
- 1.5B模型，~2000 steps，64 batch size，16k GPU-hours
- 用KL-regularization, policy resetting, entropy controls来maintain stability
- ScaleRL compute是ProRL的6x

**LitePPO** (Liu et al. 2025c https://arxiv.org/abs/2508.08221):
- Qwen-3 4B/8B上系统ablate design choices
- Focus on comparative empirical findings而非scaling behavior
- 提出minimalist组合胜过GRPO/DAPO

**ScaleRL的差异化**:
1. 第一个develop & validate compute-performance framework with predictive fits
2. Much larger compute (6x ProRL)
3. 第一个scale to 100k GPU-hours without stability issues
4. 用in-distribution validation而非downstream做scaling analysis (类似pre-training practice)

---

## 10. Methodology贡献与Future Work

### 10.1 Methodology要点

1. **Sigmoidal fits**: 用$R_C - R_0 = (A-R_0)/(1+(C_{mid}/C)^B)$，跳过前1.5k GPU-hours (1 epoch后)
2. **Grid search fit**: $A \in [0.45, 0.80]$, $C_{mid} \in [100, 40000]$
3. **Error margin**: 跑3次独立run，A的variance ±0.02
4. **Validation而非downstream**: 在held-out prompts上测mean@16 pass rate，每100步一次
5. **Forward + LOO ablations**: Forward ablation逐个加component，LOO从最终recipe逐个减

### 10.2 最重要的design decisions (Section 7)

按importance排序:
1. **Off-policy algorithm** (PipelineRL > PPO-off-policy): 影响both $A$和$B$
2. **Loss function** (CISPO > GSPO > DAPO): 主要影响$A$
3. **Model precision** (FP32 at LM head): 影响数值稳定性，shift $A$
4. 其他 (loss aggregation, advantage normalization, batch size, generation length): 主要影响$B$或stability

### 10.3 Asymptote vs Efficiency权衡

Forward ablation (从baseline加): 优先选higher $A$
LOO ablation (从ScaleRL减): 单个component对$A$影响小，主要影响$B$

**解读**: ScaleRL的strength来自cumulative effect - 单个component的remove被其他component compensate，但efficiency下降。这说明components之间有synergy。

### 10.4 Generalization观察

虽然主metric是in-distribution validation，但观察到correlation with downstream (AIME-24, MATH-500, LiveCodeBench)。特别help generalization的choice:
- Larger batch size (Figure 10, A.14)
- Reducing truncations (A.15)
- Longer generation lengths (Figure 9)
- Larger model scale (Figure 1)

### 10.5 Future work方向

作者明确提出:
1. RL scaling laws across pre-training compute, model size, RL training data (类似Chinchilla但for RL)
2. Other reward types: structured rewards (Setlur et al. 2024 https://arxiv.org/abs/2410.08140), generative verifiers (Zhang et al. 2025a https://arxiv.org/abs/2408.15240)
3. Other post-training regimes: multi-turn RL, agentic interaction, long-form reasoning

---

## 11. 关键Intuition总结

1. **Bitter Lesson体现在RL**: Figure 2中ScaleRL早期不如DAPO (DAPO快启动)，但extrapolate后ScaleRL的$A$更高。小compute ablation可能mislead，必须用scaling framework预测large compute。

2. **多数design choice只调efficiency不调ceiling**: Loss aggregation, advantage normalization, curriculum等主要影响$B$。真正影响$A$的只有off-policy algorithm, loss type, precision, batch size, generation length这几个。

3. **PipelineRL的"streaming"优势**: 不是更efficient的off-policy，是更接近on-policy。staleness通过KV cache reuse+continuous weight update被minimize。

4. **CISPO的稳定性来自stop-gradient mask**: ρ不直接乘进gradient，只做truncation。这避免了DAPO中ρ过大导致的gradient explosion，且让训练对$\epsilon_{max}$超参数robust。

5. **FP32 fix反映RL的敏感性**: pre-training对numerical precision相对tolerant (loss smooth)，但RL的IS ratio是ratio - 分子分母的small noise会被division放大。LM head的softmax特别容易magnify。

6. **Sigmoidal vs Power-law的本质区别**: Power-law假设unbounded gain (但没有$C_0$ threshold)，sigmoidal有natural bound。RL的accuracy metric是bounded的，sigmoid更natural。在high-compute regime它们重合。

7. **Truncation rate作为instability proxy**: Generation长度爆炸是RL的常见pathology。ScaleRL通过interruption+longer context+larger model控制truncation，这是stability的关键。

8. **Multi-task保持predictable per-domain scaling**: 不同task有不同$B$但都follow sigmoid。这暗示scaling framework可以generalize到multi-task setup。

9. **Validation而非downstream做scaling analysis**: Downstream benchmark太noisy (Agarwal et al. 2021 https://arxiv.org/abs/2106.06449, Madaan et al. 2024 https://arxiv.org/abs/2406.10229)，in-distribution validation提供clean signal。这是从pre-training借来的methodology。

10. **Entropy作为proxy的失败**: 维持high entropy $\neq$ good exploration。Batch size 768 vs 2048有identical entropy trajectories但performance差很多。这质疑了Cui et al. 2025等用entropy作为exploration metric的做法。

---

## 12. 联想到的相关工作与延伸思考

- **Pre-training scaling laws** (Kaplan 2020, Hoffmann 2022, Porian 2025 https://arxiv.org/abs/2406.19146): 这paper直接borrowing方法论，特别是"跳过低compute regime"的做法。Chinchilla的compute-optimal allocation思路未来可能应用到RL - 比如最优分配compute在generation vs training。

- **OpenAI o1/o3** (https://openai.com/index/introducing-o3-and-o4-mini/): 报告说o1→o3 RL compute 10x增长，但没公开scaling law。这paper的methodology可能help understand这种scaling的内在dynamics。

- **DeepSeek-R1** (https://arxiv.org/abs/2501.12948): 第一个公开R1-Zero training recipe的paper，100k H800 hours。这paper的ScaleRL在该compute scale上验证predictable。

- **Kimi K1.5** (https://arxiv.org/abs/2501.12599): Length penalty是他们用的，paper比较interruption vs length penalty发现interruption更好。

- **Dr. GRPO** (Liu et al. 2025b https://arxiv.org/abs/2503.20783): 提出"no advantage normalization"，paper比较发现和batch-level normalization差不多。

- **DeepCoder** (https://www.together.ai/blog/deepcoder): Code RL dataset，paper用它在multi-task setup中。

- **Thinking Machines Lab nondeterminism blog** (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): FP32 fix的来源。这blog详细讲inference和training numerical mismatch问题。

- **Truncated Importance Sampling** (Ionides 2008): CISPO的理论基础，1980s的classic statistics technique被重新应用到deep RL。

- **Asynchronous RL** (Noukhovitch et al. 2024 https://arxiv.org/abs/2410.18252): Asynchronous RLHF的earlier work，PipelineRL是其streaming variant。

- **Scalable Overseas-Free (SILVER) 类工作**: RL without human labels的趋势，verifiable rewards让RL scale without reward model成为可能。

- **Open Problem - Compute-optimal RL**: Pre-training有Chinchilla (model size vs data的最优分配)，RL还没有"generation length vs batch size vs epochs"的最优分配law。这paper提供framework但没solve。

- **Connection to test-time compute**: Generation length scaling (Figure 9)本质上和test-time compute scaling (Snell et al. 2024)相关。这paper显示在训练时increasing context能raise ceiling，这和test-time scaling law的observation一致。

- **Connection to Grok-4/o3**: 报告说这些frontier model RL compute爆炸增长。这paper的scaling framework可能help理解这种growth的predictability。

希望这些技术细节和intuition能帮Andrej build mental model。如果需要深入某个具体component (比如CISPO的gradient flow分析、PipelineRL的staleness数学分析、sigmoidal fit的numerical procedure)，可以继续展开。
