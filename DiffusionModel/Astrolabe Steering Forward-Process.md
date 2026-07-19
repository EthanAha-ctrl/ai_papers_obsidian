---
source_pdf: Astrolabe Steering Forward-Process.pdf
paper_sha256: 335ad25fd61757cbe67d7560475df15414046011a2079ccb7076542ea815bd28
processed_at: '2026-07-18T09:36:53-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Astrolabe: Forward-Process RL for Distilled AR Video Models 深度解析

## 1. 问题定位:为什么需要新框架

Distilled AR video models(如Self-Forcing、Causal Forcing、LongLive)通过DMD(distribution matching distillation)将bidirectional diffusion teacher蒸馏成streaming student,实现KV-cache-based的实时生成。但蒸馏只保证student mimics teacher分布,**没有任何机制对齐human preference**,导致artifacts、motion不自然。

现有RL路径各有硬伤:
- **Reward-weighted distillation** (Reward Forcing):只能shift分布向high-reward,缺乏exploration,无法penalize suboptimal samples
- **Reverse-process RL** (Dance-GRPO, Flow-GRPO):需要估计sampling trajectory上的log-probability,这couple到具体solver,还要store所有intermediate states,memory/compute overhead直接抵消了streaming的效率优势

Astrolabe的核心insight:**绕过reverse process,直接在forward process上做contrastive policy improvement**。

参考链接:
- DiffusionNFT原paper: https://arxiv.org/abs/2509.16117
- Dance-GRPO: https://arxiv.org/abs/2505.07818
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Causal Forcing: https://arxiv.org/abs/2602.02214

---

## 2. Forward-Process RL的数学核心

### 2.1 Flow Matching背景

AR video model将joint分布factorize:
$$p(x_{1:N}) = \prod_{i=1}^{N} p(x_i | x_{<i})$$

每个conditional用flow matching建模,概率路径:
$$x_i^t = (1-t)x_i + t\epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0,I), \quad t\in[0,1]$$

变量含义:
- $x_i$:第$i$个clip的clean sample
- $\epsilon_i$:Gaussian noise
- $t$:time embedding,从0(clean)到1(noise)
- $v_\theta$:model预测的velocity field,即$\frac{dx^t}{dt}$的方向

### 2.2 Implicit Positive/Negative Policies(核心trick)

给定生成样本$x$和normalized reward $\tilde{r}\in[0,1]$,构造两个implicit policies:

$$v^+ = (1-\beta)v_{\theta_{old}} + \beta v_\theta \quad (Eq.1)$$
$$v^- = (1+\beta)v_{\theta_{old}} - \beta v_\theta$$

变量含义:
- $v_\theta$:**当前policy**(被训练)的velocity prediction
- $v_{\theta_{old}}$:**behavior/old policy**的velocity prediction(rollout时用的那个)
- $\beta$:interpolation strength,控制contrastive guidance的强度
- $v^+$:positive implicit policy—在old和current之间插值,代表"向current靠拢"
- $v^-$:negative implicit policy—在old基础上**远离**current方向

**Intuition**: 这本质是DPO思想在continuous diffusion上的forward-process版本。$v^+$代表"如果当前policy是对的,velocity应该这样";$v^-$代表"如果当前policy是错的,velocity应该那样"。通过contrast这两个implicit policy,隐式定义了improvement direction,无需显式估计$\log p_\theta(x)$。

### 2.3 Policy Loss

$$\mathcal{L}_{policy} = \tilde{r}\|v^+ - v_{target}\|_2^2 + (1-\tilde{r})\|v^- - v_{target}\|_2^2 \quad (Eq.2)$$

变量含义:
- $\tilde{r}$:normalized reward ∈[0,1]
- $v_{target}$:forward velocity target,基于noised sample $x^t$构造

**Intuition**: 
- 当$\tilde{r}\to 1$(好样本):loss拉$v^+$靠近target → 当前policy被rewarded,向current方向shift
- 当$\tilde{r}\to 0$(坏样本):loss拉$v^-$靠近target → 当前policy被penalized,远离current方向

这比reward-weighted regression强在:**坏样本不是被忽略,而是被主动推开**。

### 2.4 Theorem 1: 严格Improvement Guarantee

论文appendix B.1的Theorem 1给出local improvement的理论保证。关键定义:

$$\alpha \triangleq \alpha(x_n^t, \mathcal{C}_n) = \mathbb{E}_{\pi_{old}}[\tilde{r} | x_n^t, \mathcal{C}_n]$$

这是posterior positive probability—给定noised state和context,该样本是positive的期望概率。

Implicit distributions:
$$\pi^+(x_n|\mathcal{C}_n) = \frac{\tilde{r}\cdot\pi_{old}}{\mathbb{E}[\tilde{r}]}, \quad \pi^-(x_n|\mathcal{C}_n) = \frac{(1-\tilde{r})\cdot\pi_{old}}{\mathbb{E}[1-\tilde{r}]}$$

最优velocity field closed-form解:
$$v_\theta^* = v_{old} + \frac{1-\alpha(1+\beta)}{(1+\beta)(1-\alpha)}(v^+ - v_{old}) \quad (Eq.7)$$

**关键regime**: 当$\alpha(1+\beta) < 1$时,系数为正,shift方向严格对齐$v^+ - v_{old}$,即**严格向高advantage region移动**。

**退化case**: $\beta\to 0$时,$v^+\to v_{old}$,shift退化,变成standard reward-weighted regression。这解释了为什么需要negative repulsion。

推导关键步骤(Bayes decomposition):
$$\pi_{old}(x_n^0|x_n^t, \mathcal{C}_n) = \alpha\pi^+ + (1-\alpha)\pi^-$$

由total expectation:
$$v_{old} = \alpha v^+ + (1-\alpha)v^-$$

设$\delta^+ = v^+ - v_{old}$,则$v^- - v_{old} = -\frac{\alpha}{1-\alpha}\delta^+$。代入loss并对$u = v_\theta - v_{old}$求导置零,得到closed form。

---

## 3. Memory-Efficient Streaming Rollout

### 3.1 Rolling KV Cache with Frame Sinks

长视频生成中,naive cache所有历史frames的KV会导致memory线性增长。Astrolabe的方案:

$$\mathcal{C}_n = \text{Frame Sink}(S) \cup \text{Rolling Window}(L)$$

- **Frame Sink**: $S=3$帧永久保留,anchor global semantic context,防long-range drift(借鉴StreamingLLM的attention sink思想)
- **Rolling Window**: $L=21$(Self-Forcing)或$L=15$(LongLive)最近帧,提供fine-grained local conditioning

由于$S$和$L$都是固定hyperparameter,**peak KV memory与video length N无关**,实现真正的streaming。

### 3.2 Clip-level Group-wise Sampling

标准GRPO需要G条independent trajectories,对长视频而言开销巨大。Astrolabe改为:

- Autoregressively生成历史**一次**,freeze KV cache作为shared prefix
- 在第$n$步,基于frozen context并行decode G=24个candidate clips:

$$x_n^{(i)} \sim \pi_\theta(\cdot|\mathcal{C}_n, c), \quad i\in\{1,\dots,G\} \quad (Eq.3)$$

额外开销只在local chunk,而非全序列。

---

## 4. Advantage计算与Normalization

Group-wise mean-centering:
$$A^{(i)} = R(x_n^{(i)}, c) - \frac{1}{G}\sum_{j=1}^{G}R(x_n^{(j)}, c) \quad (Eq.4)$$

变量:
- $A^{(i)}$:第$i$个candidate的advantage
- $R$:composite reward(后面讲)
- $G$:group size = 24

Normalization到$[0,1]$:
$$\tilde{r}_i = \text{clip}(A^{(i)}/A_{max})/2 + 0.5$$

这样positive/negative样本在loss中分别承担不同角色。

---

## 5. Streaming Long Tuning

Distilled AR模型有**train-short/test-long mismatch**:训练5秒,测试30秒+。长程error accumulation不可避免。

Astrolabe的解法—**detached historical context**:

1. Forward pass累积KV cache到target step $n$
2. 到达active training window $x_n$时,**显式detach**$x_{<n}$的KV cache from computation graph
3. 这个detached cache作为historical context,模拟inference时progressively noisy的条件
4. Gradient只在active window内backprop

**Intuition**: 这相当于一种"truncated BPTT"—前向时模型看到真实的(long, noisy)context,但梯度只局部更新。这既解决distribution shift,又bound住memory。Table 4(a)的ablation显示:

| Config | HPSv3↑ | MQ↑ | Mem↓(GB) |
|--------|--------|-----|----------|
| Seq + Full BP | OOM | OOM | >140 |
| Seq + Detach | 10.21 | 1.72 | 96.4 |
| Clip + Full BP | 10.58 | 1.76 | 112.3 |
| **Clip + Detach** | **10.84** | **1.80** | **54.3** |

Clip+Detach相比Clip+Full BP内存减半,质量反而更好(因为能在更长context上训练)。

---

## 6. Multi-Reward Formulation

单一reward必然hacking。Astrolabe用3-axis composite:

### 6.1 Visual Quality (VQ)
$$R_{VQ} = \text{mean HPSv3 over top 30\% frames}$$

只用top 30%而非全部—避免transient motion blur拖累overall aesthetic评估。

### 6.2 Motion Quality (MQ)
$$R_{MQ} = \text{VideoAlign on grayscale input}$$

去掉颜色,force metric专注motion dynamics而非texture。

### 6.3 Text Alignment (TA)
$$R_{TA} = \text{VideoAlign on RGB}$$

### Ablation (Table 4c):

| Reward | HPSv3↑ | MQ↑ | TA↑ | VB↑ |
|--------|--------|-----|-----|-----|
| VQ only | 10.92 | 1.58 | -0.075 | 83.21 |
| MQ only | 9.31 | 1.82 | -0.058 | 83.67 |
| TA only | 9.42 | 1.62 | 0.082 | 84.25 |
| **All (Ours)** | **10.84** | **1.80** | **0.065** | **84.46** |

VQ-only导致MQ崩塌到1.58(static frames)—典型reward hacking。Multi-reward平衡各维度。

---

## 7. Uncertainty-Aware Selective KL

### 7.1 Reward Hacking Detection via Rank Disagreement

核心idea:如果primary reward model给某样本异常高分,但auxiliary reward models不买账,那就是hacking。

$$\Delta_{rank}^{(i)} = rank_p^{(i)} - \frac{1}{M-1}\sum_{m\neq p} rank_m^{(i)}$$

变量:
- $rank_p^{(i)}$:第$i$个样本在primary reward下的rank
- $rank_m^{(i)}$:第$i$个样本在auxiliary reward $m$下的rank
- $M$:reward model总数
- $\Delta_{rank}^{(i)}$:rank discrepancy

高正值 → primary reward与ensemble不一致 → 可能hacking。

Mask:
$$\mathcal{M}^{(i)} = \mathbb{1}[\Delta_{rank}^{(i)} > \tau]$$

其中$\tau$是$(1-\rho)$-th percentile of positive discrepancies,$\rho$是risk ratio。

### 7.2 Selective KL Penalty

$$\mathcal{L}_{KL} = \frac{1}{|\mathcal{M}|}\sum_{i:\mathcal{M}^{(i)}=1}\|v_\theta^{(i)} - v_{\theta_{ref}}^{(i)}\|^2$$

总loss:
$$\mathcal{L} = \mathcal{L}_{policy} + \lambda_{KL}\mathcal{L}_{KL}$$

**只在high-uncertainty samples上施加KL**,clean samples自由探索。Table 4(b)的ablation:

| Strategy | HPSv3↑ | MQ↑ | TA↑ |
|----------|--------|-----|-----|
| No KL | 10.67 | 1.74 | -0.068 |
| Uniform ($\lambda=1e^{-4}$) | 10.52 | 1.71 | 0.012 |
| Uniform ($\lambda=5e^{-4}$) | 10.28 | 1.68 | 0.028 |
| **Sel. + EMA** | **10.84** | **1.80** | **0.065** |

Uniform KL过度constrain,反而损害性能;Selective KL + EMA达到最佳。

### 7.3 Dynamic Reference Update

- $\theta_{old}$:EMA update,decay rate $\gamma=0.9$
- $\theta_{ref}$:当$\mathcal{L}_{KL} > \tau_{KL}$或epochs达到$K_{max}$时reset到当前$\theta$

这避免online RL中distribution shift导致的reference过时问题。

### 7.4 Theorem 2: Performance Lower Bound

$$\mathbb{E}_{\pi_\theta}[R^*] \geq \mathbb{E}_{\pi_\theta}[\hat{R}] - \epsilon_{safe} - \pi_\theta(\mathcal{U})(\epsilon_{risk} + 2R_{max}\sqrt{\mathcal{L}_{KL}}) \quad (Eq.13)$$

变量:
- $R^*$:true reward
- $\hat{R}$:proxy reward
- $\mathcal{U}$:high-uncertainty region
- $\epsilon_{safe}$:safe region的reward估计误差(small)
- $\epsilon_{risk}$:risk region的reward估计误差(large)
- $R_{max}$:reward上界
- $\mathcal{L}_{KL}$:selective KL loss

**Key insight**: Unlike global KL penalty, Astrolabe的selective penalty只通过$\pi_\theta(\mathcal{U})\cdot 2R_{max}\sqrt{\mathcal{L}_{KL}}$项起作用。Safe region完全自由探索,trust region只在policy drift进risk region时激活。证明用了TV dual representation + Pinsker's inequality。

---

## 8. 关键工程细节:去除Adaptive Weighting

DiffusionNFT原版用self-normalized $x_0$ denominator做adaptive loss weighting。Astrolabe**移除**了这个,原因:

在distilled AR setting(T=4步),discretization gap很大,这个dynamic denominator变得volatile,导致predicted $x_0$ norm在50步后爆炸,reward急剧collapse。

Figure 8b显示:移除adaptive weighting后,$x_0$ norm bounded,reward steady monotonic improvement。这是从理论方法到distilled AR场景的关键adaptation。

---

## 9. 实验结果

### 9.1 Short-Video (Table 1)

| Method | Total↑ | Quality↑ | HPSv3↑ | MQ↑ | Throughput↑ |
|--------|--------|----------|--------|-----|-------------|
| Self-Forcing | 83.74 | 84.48 | 80.77 | 9.36 | 17.0 |
| **+Ours** | 83.79 | 84.51 | 80.92 | **10.72** | 17.0 |
| LongLive | 83.22 | 83.68 | 81.37 | 9.38 | 20.7 |
| **+Ours** | **84.93** | **85.83** | 81.36 | **11.03** | 20.7 |
| Causal Forcing | 84.04 | 84.59 | 81.84 | 9.48 | 17.0 |
| **+Ours** | 84.46 | 85.15 | 81.72 | **10.84** | 17.0 |

**MQ提升最显著**(从~9.4到~11),且throughput完全保持—证明RL alignment不牺牲inference speed。

### 9.2 Long-Video (Table 2)

VBench-Long 30秒视频:

| Method | Total↑ | Quality↑ | HPSv3↑ | MQ↑ |
|--------|--------|----------|--------|-----|
| Self-Forcing | 81.59 | 83.82 | 9.12 | 1.61 |
| **+Ours** | 82.03 | 84.36 | 10.38 | 1.72 |
| Causal Forcing | 82.87 | 84.36 | 9.28 | 1.65 |
| **+Ours** | 84.24 | 86.18 | 10.52 | 1.74 |

**Short-video上做的alignment有效extrapolate到long video**—这是streaming training scheme的功劳。

### 9.3 Multi-Prompt (Table 3)

60秒multi-prompt视频,CLIP score per 10s interval:

LongLive+Ours在0-60s全interval都优于baseline,证明multi-prompt场景下的long-range coherence提升。

---

## 10. Intuition Building:整个框架的设计哲学

**为什么Forward-Process RL适合distilled AR?**

Distilled AR模型的核心特征:
1. Few-step (T=4) 生成 → reverse trajectory很短,但每步discretization gap大
2. KV-cache streaming → 不能store中间states
3. 已经是post-distillation → 不应再unroll reverse process

Forward-process RL恰好fit这些约束:
- 只需clean sample $x$ + reward → 不需trajectory
- Loss直接在velocity prediction上 → 不couple solver
- LoRA-based → 可以share frozen base model between $\theta$和$\theta_{old}$

**为什么Selective KL而不是Uniform KL?**

Uniform KL的fundamental问题:它对所有样本施加相同的exploration constraint,但reward model在不同样本上的confidence不同。对于high-confidence positive样本,施加KL就是浪费exploration budget;对于uncertain样本,确实需要constraint防hacking。Selective KL通过rank disagreement自动识别这两种情况。

**为什么Clip-level而非Sequence-level rollout?**

Sequence-level rollout有两个bottleneck:
1. **Temporal credit assignment**: global reward无法定位局部degradation
2. **Memory**: G条independent long trajectories的KV cache

Clip-level rollout + shared prefix把G的开销从$O(G\cdot N)$降到$O(N + G\cdot \text{clip\_len})$,且credit assignment局部化。

---

## 11. 局限性与未来方向

Paper自承两个limitation:

1. **Reward Model依赖**: Astrolabe无法超越reward model的能力。现有VideoAlign等对long-horizon physics、causality、multi-entity interaction评估能力有限。

2. **Base Architecture ceiling**: RL只能shift分布,不能instantiate base model不具备的能力。如果distillation过于aggressive丢失了某些capacity,RL无法恢复。

潜在extension方向:
- Physics-aware reward models for long-form video
- Multi-agent reward ensemble for better uncertainty estimation
- Hierarchical RL with both clip-level and sequence-level rewards
- Integration with test-time guidance (classifier-free guidance的RL版本)

---

## 12. 与相关工作定位

| Method | Trajectory Storage | Solver Coupled |适用于Distilled AR | Long Video |
|--------|--------------------|----------------|-------------------|------------|
| Reward Forcing | No | No | ✓(但无exploration) | Limited |
| Dance-GRPO | Yes | Yes | ✗ | ✗ |
| Flow-GRPO | Yes | Yes | ✗ | ✗ |
| DiffusionNFT | No | No | Partially | ✗ |
| WorldCompass | No | No | ✓(但heavy teacher) | ✓ |
| **Astrolabe** | **No** | **No** | **✓** | **✓** |

Astrolabe是第一个把forward-process RL **完整适配**到distilled AR streaming video的工作,关键贡献:
1. 去除adaptive weighting(适配大discretization gap)
2. Streaming rollout with rolling KV cache(适配streaming architecture)
3. Selective KL(适配online RL的distribution shift)

---

## 参考链接汇总

- Astrolabe paper本身(从附件内容看,应该是2026年的paper,引用了Causal Forcing arXiv:2602.02214)
- DiffusionNFT: https://arxiv.org/abs/2509.16117
- Dance-GRPO: https://arxiv.org/abs/2505.07818
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Causal Forcing: https://arxiv.org/abs/2602.02214
- LongLive: https://arxiv.org/abs/2509.22622
- Infinity-RoPE: https://arxiv.org/abs/2511.20649
- DMD (Distribution Matching Distillation): https://arxiv.org/abs/2311.18828
- CausVid: https://arxiv.org/abs/2504.04885
- DiffusionDPO: https://arxiv.org/abs/2311.12908
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- HPSv3: https://arxiv.org/abs/2503.01305
- VideoAlign: https://arxiv.org/abs/2505.01864
- VBench: https://arxiv.org/abs/2311.13582
- Gardo (selective KL inspiration): https://arxiv.org/abs/2512.24138
- WorldCompass: https://arxiv.org/abs/2602.09022
- VidProM dataset: https://arxiv.org/abs/2412.12622
- Wan2.1: https://arxiv.org/abs/2503.20314

这篇paper的核心贡献是把forward-process RL从"理论可行"推到"engineering practical for distilled streaming video",每个component(rolling KV、selective KL、去除adaptive weighting)都是针对具体engineering bottleneck的精准手术。整体设计非常clean,理论(Theorem 1, 2)和工程(ablation table)配合紧密。
