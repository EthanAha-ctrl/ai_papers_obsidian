---
source_pdf: One Token Per Frame Reconsidering Visual Bandwidth in World Models for
  VLA Policy.pdf
paper_sha256: ce0d3abb81a58a7698e90f96bba15c4a6652103bf9a1b3ded7b7bd5382be37df
processed_at: '2026-08-05T23:46:29-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 OneWM-VLA

## 一句话概括

这帮人发现：在 frozen VLA 上面挂 world model 的时候，**每帧视觉压成 1 个 token 就够了**，而且 token 越多反而越差。他们用 joint flow matching 把 latent 和 action 绑在一起 denoise，让 latent stream 成为 action 的 structural prior，而不是当个 side channel 挂着。

就这么个事，但背后的 intuition 挺深的。

---

## 核心矛盾：VLA 加 world model 为什么难

先说 background。现在的 VLA 模型（像 [π0](https://arxiv.org/abs/2410.24164)、[OpenVLA](https://arxiv.org/abs/2406.09246)）本质上是 reactive policy——看到画面就输出 action。短任务没问题，但长任务会 error accumulate：每步犯一点错，几十步之后就崩了。

解决方案就是加个 world model，让 policy 能"想象"未来会发生什么，相当于 implicit MPC。但问题来了：**world model 往哪 predict**？

- Pixel space（像 [WorldVLA](https://arxiv.org/abs/2506.21539)）：要生成未来视频帧，计算量随 horizon 爆炸，且大部分像素对 control 没用（背景光照纹理这些）
- Latent space（像 [Dreamer](https://arxiv.org/abs/1912.01603)）：在 learned representation 里 predict，但通常在自己 RL pipeline 里搞，没 transfer 到 frozen VLA 上

这篇 paper 选 latent space，但问了个更尖锐的问题：**per-frame latent 要多宽**？

---

## 最反直觉的发现：1 token per frame 最优

Tab. 4 是整篇 paper 最 striking 的 table：

| Tokens/View | Avg % | FPS |
|-------------|-------|-----|
| 256 (Full) | OOM | - |
| 12 | 20.54 | 0.13 |
| 6 | 33.85 | 0.56 |
| 3 | 41.86 | 1.21 |
| **1** | **53.13** | **4.81** |

**Token 从 1 加到 12，success rate 单调下降 32 个点**。这不是微调 noise，是真真实实的反直觉。

作者的 tentative explanation 是 implicit regularization——在 30K LoRA steps 的 budget 下，太多 token 会让每个 token 的 effective gradient signal 被稀释，反而学不好。

我的额外 intuition：VLA backbone（PaliGemma）已经做了重活，256 个 visual tokens 已经是 semantic level。再往 1 token 压，只是 distill 出 **control-sufficient statistic**。机器人操作中 task-relevant 信息本来就稀疏（gripper 位置、target 物体、障碍物），1 个 well-pooled token 够 encode "what's where and what to do next"。

而且 256 tokens × H=30 直接 OOM。1 token 让 world module 的 per-step cost 与 horizon 无关，这是 scalability 的关键。

---

## Adaptive Attention Pooling：怎么压出那 1 个 token

这里有个 design choice 要讲清楚。naive 的压缩（average pooling、CLS token）会丢信息。作者搞了个 multi-strategy adaptive pooling。

### 三种 scoring function

对每个 view $i$ 的 256 个 tokens $\{\mathbf{x}_i^{(n)}\}_{n=1}^{N}$，用三种互补的 scoring：

$$\phi_{\text{MAX}}(\mathbf{x}) = \max_d x^{(d)}, \quad \phi_{\text{SUM}}(\mathbf{x}) = \sum_{d=1}^D x^{(d)}, \quad \phi_{\text{LEARN}}(\mathbf{x}) = Q_\theta(\mathbf{x})$$

变量解释：
- $x^{(d)}$: token $\mathbf{x}$ 的第 $d$ 维 channel 值
- $D$: hidden dimension
- $Q_\theta$: view-specific 小 MLP

三种 score 的直觉：
- **MAX**: 抓 peak activation，类似 "哪里最 salient"
- **SUM**: 抓 total energy，类似 "overall 信号强度"
- **LEARN**: 学出来的 task-aware saliency

每种 score 在 token 维度做 softmax 得到 weights：

$$w_m^{(n)} = \frac{\exp(\phi_m(\mathbf{x}_i^{(n)})/\tau)}{\sum_{j=1}^N \exp(\phi_m(\mathbf{x}_i^{(j)})/\tau)}$$

- $w_m^{(n)}$: 第 $m$ 种策略对第 $n$ 个 token 的 attention weight
- $\tau$: temperature（实验里用 $\tau=0.1$，比较 sharp）

然后 aggregate：

$$\mathbf{p}_{\text{Max}} = \max_n(w_{\text{MAX}}^{(n)} \mathbf{x}_i^{(n)}), \quad \mathbf{p}_m = \sum_{n=1}^N w_m^{(n)} \mathbf{x}_i^{(n)}, \quad m \in \{\text{SUM, LEARN}\}$$

注意 MAX 策略用 element-wise max，另外两个用 weighted sum。得到 3 个 pooled tokens $\{\mathbf{p}_{\text{Max}}, \mathbf{p}_{\text{Sum}}, \mathbf{p}_{\text{Learn}}\}$，每个都是 $\mathbb{R}^D$。

### View-level fusion

3 个 pooled tokens 再用 learnable convex combination 融合：

$$\mathbf{Z}_i = \sum_{m \in \mathcal{M}} \beta_m \mathbf{p}_m, \quad \beta_m = \frac{\exp(\alpha_m/\tau)}{\sum_{m'} \exp(\alpha_{m'}/\tau)}$$

- $\alpha_m$: 可训练 scalar（每种策略一个）
- $\beta_m$: softmax-normalized fusion weight
- $\mathbf{Z}_i \in \mathbb{R}^{B \times T \times 1 \times D}$: 最终的 per-view per-frame world token

训练后 $\beta_{\text{LEARN}} \approx 0.48$–$0.57$, $\beta_{\text{MAX}} \approx 0.20$–$0.28$, $\beta_{\text{SUM}} \approx 0.20$–$0.28$。LEARN 主导但 MAX 和 SUM 仍贡献。

### 为什么不能 simpler？

Tab. 5 和 Tab. 6 的 ablation 很说明问题：

| Method | LIBERO Avg% |
|--------|-------------|
| Full adaptive | 93.3 |
| Static average pooling | 72.8 |
| No fusion logic | 47.6 |

| Branch config | MetaWorld Avg% |
|---------------|-----------------|
| LEARN+MAX+SUM | 61.30 |
| Only LEARN | 50.42 |
| Only MAX | 22.38 |
| Only SUM | 55.19 |

**Input-dependent attention 是关键**——static average pooling 直接掉 20 个点。说明压缩必须 instance-specific，不能 fixed。

Only MAX 在 Hard task 上直接 0%，因为 max pooling 太激进了，只保留最 salient 的 patch，丢了 spatial relation。SUM 反而更好（55.19%），因为它保留了 energy 的 spatial distribution。

---

## Joint Flow Matching：latent 和 action 怎么绑在一起

这是另一核心 design。先说 why——

### 为什么 separate decoder 不行

naive 做法：world model 有自己的 decoder predict future latent，action head 用 cross-attention 从 latent 读信息。问题：latent 的 supervision 是 reconstruction loss，优化方向不一定与 action relevant 方向一致。容易学到 "看起来对但 control 无关" 的 latent。

Tab. 8 量化了这个点：

| Method | MetaWorld Avg% (H=20) |
|--------|----------------------|
| Full joint | 58.09 |
| No latent branch | 43.04 |
| No latent loss (保留 input) | 21.47 |

第三个 setting 特别 informative——latent tokens 作为 input 存在但没有 supervision，比完全没有 latent branch 还差。说明 **latent supervision 是 coupling 的核心**，不是 added capacity。

### Joint flow matching 的数学

继承 [π0](https://arxiv.org/abs/2410.24164) 的 flow matching 框架，但同时 predict latent 和 action。

**Flow time schedule**（两个 branch 共享）：

$$x_t^a = t\epsilon_a + (1-t)a, \quad x_t^z = t\epsilon_z + (1-t)z, \quad t \sim \text{Beta}(1.5, 1)$$

- $a \in \mathbb{R}^{H \times D_a}$: 未来 action sequence（$H$ 是 horizon，$D_a$ 是 action 维度）
- $z \in \mathbb{R}^{H \times D_z}$: 未来 latent world tokens
- $\epsilon_a, \epsilon_z \sim \mathcal{N}(0, I)$: 两个 branch 各自的 Gaussian noise
- $t \in [0,1]$: flow time
- $\text{Beta}(1.5, 1)$: 时间采样分布，让 $t$ 偏向 0（data end），训练聚焦在 clean-ish 状态

Target velocity（constant 沿插值路径）：

$$u_t^a = \epsilon_a - a, \quad u_t^z = \epsilon_z - z$$

**Loss**（L1 距离，Tab. 11 ablation 确认 L1/L1 最优）：

$$\mathcal{L} = \lambda_a \mathbb{E}[\|v_\theta^a - u_t^a\|_1] + \sum_{i \in \{r, w_1, w_2\}} \lambda_i \mathbb{E}[\|v_\theta^z - u_t^z\|_1]$$

- $v_\theta^a, v_\theta^z$: network 预测的 velocity field（同一个 transformer 的不同 output position）
- $\lambda_a = 1.0$: action loss weight
- $\lambda_r = \lambda_{w_1} = \lambda_{w_2} = 0.1$: 每个 view 的 latent loss weight
- $r, w_1, w_2$: third-person view 和两个 wrist view

### "Joint" 的精髓

Transformer 输入是 interleaved sequence：
- 当前 tokens: $\{x_t^{z_r}, x_t^{z_{w_1}}, x_t^{z_{w_2}}, l_t, s_t\}$
- 未来 noisy queries: $\{x_{t+k}^{z_r}, x_{t+k}^{z_{w_1}}, x_{t+k}^{z_{w_2}}, x_{t+k}^a\}_{k=1}^h$

**$v_\theta^a$ 和 $v_\theta^z$ 在同一个 self-attention 里 co-evolve**。latent tokens 和 action tokens 互相 attend，所以 latent 必须能 "explain" action 的变化。如果 latent 学了 control-irrelevant 的内容，它对 action prediction 的 gradient 会被 push 掉。

推理时 10 ODE steps 联合 denoise，但 **只有 action stream 执行到 robot 上**，latent 是 internal auxiliary trajectory。

### L1 vs L2 的 intuition

Tab. 11:

| Action/Latent loss | LIBERO Avg% |
|-------------------|-------------|
| L2/L2 | 87.6 |
| L1/L2 | 90.2 |
| L1/L1 | 93.3 |

L1 比 L2 好 5.7 个点。Intuition：robot action 是 multi-modal 的（同一 observation 可能有多个 valid action），L2 对 outlier 敏感会 regression to mean，L1 更 robust 保留 multi-modal structure。这和 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 的选择一致。

### Beta(1.5, 1) 的直觉

$t \sim \text{Beta}(1.5, 1)$ 的 PDF 在 $t \to 1$ 处趋于 1.5，在 $t=0$ 处为 0。意味着训练时更多采样在 $t$ 接近 1（noise end）... 等等，让我重新想想。

公式 $x_t = t\epsilon + (1-t)a$，$t=0$ 是 pure data，$t=1$ 是 pure noise。Beta(1.5, 1) 的 mode 在 $t=1$ 附近，所以更多采样在 noise end，让 model 学好 "from noise to data" 的生成过程。

但 paper 原文说 "places more probability mass near $t=0$, biasing training toward the data end"——这里可能 paper 的 $t$ 定义和我不一致，或者 Beta(1.5, 1) 的参数顺序不同。不管怎样，核心 idea 是让训练 focus 在 data-rich 区域，因为 robot action 精度需求高。

---

## 实验结果的核心 takeaway

### 1. Long horizon 是 OneWM-VLA 的主场

MetaWorld MT50 (Tab. 1):

| $H$ | OneWM-VLA | π0 | π0.5 |
|-----|-----------|-----|------|
| 5 | 61.28 | 43.97 | 38.25 |
| 30 | 53.13 | 37.98 | 26.83 |

π0 从 H=5 到 H=30 掉 6 个点，π0.5 掉 11 个点，OneWM-VLA 只掉 8 个点且绝对值高。在 Very Hard tier H=30：OneWM-VLA 60% vs π0 4% vs π0.5 4%——15 倍差距。

LIBERO Long suite：OneWM-VLA 95.6% vs π0 85.2% vs π0.5 92.4%。其他三个 short suite 三者都 95%+ 饱和。

Real Piper Fold Cloth（长 horizon deformable）：OneWM-VLA 60% vs π0 20% vs π0.5 25%。3 倍提升。

**World model 的价值在 long horizon + 感知扰动 下最显著**。短任务 reactive policy 够用，长任务需要 "想象" 未来。

### 2. Real world robustness

Tab. 3 有 observation noise 的 setting（lighting shift + 位置扰动 + distractors）：

| Method | F.Cloth (noisy) |
|--------|-----------------|
| π0 | 0% |
| π0.5 | 10% |
| OneWM-VLA | 40% |

Baseline 直接崩了，OneWM-VLA 还能撑。这说明 latent world model 提供的 "imagined future" 对感知扰动有 robustness——即使当前 observation 有 noise，world model 可以基于 prior imagination 做 planning。

### 3. Semantic > Pixel compression

Tab. 7:

| Method | MetaWorld Avg% (H=30) |
|--------|----------------------|
| OneWM-VLA-pixel | 35.85 |
| OneWM-VLA (semantic) | 53.13 |

差 17 个点。Pixel compression 把 task-relevant 和 task-irrelevant 同等对待，保留 low-level noise。Semantic compression 在已与 policy 对齐的 feature space 里操作，能滤掉 noise。

### 4. Fisher ratio：77% 的 discriminative signal 保留

Appendix E 的 PCA 分析：

| | Fisher ratio $F$ |
|---|---|
| Before pooling (256 tokens) | 0.524 |
| After pooling (1 token) | 0.405 |

$$F = \text{tr}(S_B) / \text{tr}(S_W)$$

- $S_B$: between-class scatter matrix
- $S_W$: within-class scatter matrix

保留了 77% 的 class-level structure。丢的是 photometric detail，留的是 control-relevant semantic。

### 5. Inference horizon sweep 稳定

Tab. 13 (LIBERO-Long)：

| Train/Infer AH | Replan step | Success% |
|----------------|-------------|----------|
| 20/15 | 10 | 93.4 |
| 20/18 | 12 | 95.6 (peak) |
| 20/20 | 10 | 94.0 |

94-95.6% 的窄带波动。1 token per frame 让 per-step memory footprint 与 horizon 解耦，所以 horizon 扩展不爆内存。这是 scalability 的 key property。

---

## 我的几个延伸思考

### 1. 1 token 是否能 encode multi-object relation？

如果任务需要同时 manipulate 多个物体（比如 assembly），1 个 token 可能丢 object-object relation。这个 paper 的任务（MetaWorld, LIBERO, Fold Cloth）大多是 single-object focus。

可能的改进：**task-conditioned token budget**——简单任务 1 token，复杂任务自适应多 token。但 paper 的反直觉发现是 "更多 token 反而更差"，所以这可能不是单纯 budget 问题，而是 supervision signal dilution 问题。

### 2. Joint flow matching 和 model predictive control

OneWM-VLA 本质上是个 implicit MPC——world model 提供 imagined trajectory，policy 在 trajectory 上做 action generation。传统 MPC（如 [CEM](https://arxiv.org/abs/1904.05631)）需要 explicit model + optimization，这里全 learn 出来。

一个有趣方向：能不能用 latent rollout 做 **explicit planning**？比如生成多个 candidate latent trajectory，用 world model 的 confidence 做 selection。这接近 [PlaTe](https://arxiv.org/abs/2204.06236) 或 [TD-MPC](https://arxiv.org/abs/2203.05512) 的思想但 in latent space。

### 3. 与 token memory mechanism 的结合

作者在 Limitations 提到 "lightweight token-memory mechanisms" 是 future work。这让我想到 [Compressive Transformer](https://arxiv.org/abs/1911.05532) 和 [Memorizing Transformers](https://arxiv.org/abs/2203.08913)——把长序列压缩成少数 memory tokens。

OneWM-VLA 的 1 token per frame 是个 special case。更 general 的 design：每个 frame 贡献 1 个 token 到 **persistent memory**，world module 在这个 memory 上 rollout，而非重新 encode 每帧。这接近 [Episodic Memory](https://arxiv.org/abs/2104.02820) 思路。

### 4. RL fine-tuning 的挑战

[π_RL](https://arxiv.org/abs/2510.25889) 已经做了 flow-based VLA 的 online RL。OneWM-VLA 加了 latent branch，RL 的 credit assignment 会复杂——latent loss 是 unsupervised 的（predict future latent），action loss 是 supervised 的（imitation），RL 信号怎么传到 latent？

一个可能：latent branch 用 **self-predictive representation**（像 [BYOL](https://arxiv.org/abs/2006.07733) / [SimSiam](https://arxiv.org/abs/2011.10566)），action branch 用 RL。两个 branch 的 supervision 解耦，但在 forward pass 里仍 joint。

### 5. 为什么 L1 > L2 的更深原因

paper 给的 explanation 是 multi-modality。我有个额外 intuition：flow matching 的 velocity field $v_\theta$ 是 vector-valued，L2 loss $\|v - u\|^2$ 会惩罚 large deviation 很重，L1 $\|v - u\|_1$ 是 linear penalty。在 multi-modal distribution 下，target velocity $u = \epsilon - a$ 对不同 mode 的 $a$ 会变化很大，L2 会让 model 保守地 predict mean velocity，L1 允许 model 更 aggressive 地选某个 mode。

这和 [Score-based diffusion](https://arxiv.org/abs/2011.13456) 里 score function 的性质类似——L1 训练的 model 有 sharper distribution。

### 6. Beta(1.5, 1) schedule 的 alternative

如果用 [CosP](https://arxiv.org/abs/2305.13281) 或 [REPA](https://arxiv.org/abs/2410.06940) 的 schedule 会怎样？这些 schedule 设计 for image generation，robot action 的 schedule 可能需要不同——action 对 precision 更敏感，对 "global structure" 需求较弱。

一个实验设计：**task-adaptive schedule**——short horizon 任务用 uniform schedule（需要从 noise 完整生成），long horizon 用 data-biased schedule（需要 precise action refinement）。

### 7. 与 DreamVLA 的本质区别

[DreamVLA](https://arxiv.org/abs/2507.04447) 是最接近的工作，也做 latent world model for VLA。区别：
- DreamVLA 用 **separate decoder** for latent prediction，OneWM-VLA 用 **joint flow matching**
- DreamVLA 的 latent bandwidth 更宽，OneWM-VLA 证明 1 token 够用
- DreamVLA 的 supervision 是 reconstruction-style，OneWM-VLA 的 supervision 来自 action coupling

Tab. 2 显示 OneWM-VLA (98.1% LIBERO avg) 超过 DreamVLA (92.6%)，但这个对比可能 unfair（不同 backbone、不同 training data）。

### 8. 1 token per frame 的 information-theoretic view

从 [Information Bottleneck](https://arxiv.org/abs/1612.00436) 角度看，1 token per frame 是个 tight bottleneck $I(Z; X) \approx H(Z) \approx D$ bits（假设 D 维 token）。但 downstream task 只需要 $I(Z; A | X) = I(X; A)$ 的 task-relevant info。

如果 VLA backbone 已经把 $I(X; A)$ 大部分 encode 在 256 tokens 里，1 token bottleneck 的 job 是 compress 这 256 tokens 到 task-sufficient statistic。Fisher ratio 保留 77% 说明这个 compression 是 near-lossless for task purpose。

但有个 subtlety：**future latent prediction** 需要 encode temporal dynamics，不只是当前 frame 的 task info。1 token 能否 capture dynamics？实验说能，但这可能是 MetaWorld/LIBERO dynamics 简单的 artifact。更复杂的 dynamics（如 fluid、articulation）可能需要更多 token。

---

## Limitations 和 Potential

作者诚实承认：

1. **Single backbone**: 只在 π0 (2B) 上验证。更大 backbone 上 1 token 是否够用未知。我的猜测：backbone 越大，semantic encoding 越强，1 token 越可能够用（因为 single token 的 representation capacity 随 backbone $D$ 增长）。

2. **Fixed adaptation budget**: 30K LoRA steps。更长训练可能让大 token 数 viable（Tab. 4 的反转）。这指向 "compression vs capacity" 的 scaling law question——需要类似 [Chinchilla](https://arxiv.org/abs/2203.15556) 的 study for VLA adaptation。

3. **Perceptual complexity 中等**: MetaWorld/LIBERO/Piper 视觉不复杂。复杂 scene（kitchen with many objects, cluttered background）可能需要更多 token。

4. **Per-step efficiency**: 4.81 FPS 不够 real-time（30+ Hz）。但这个 number 是 batch=1 的单步 inference，实际 deployment 可以 chunk action（已实现，Tab. 12 的 replan step）。

---

## 我对这篇 paper 的整体评价

**真正贡献**不是 "1 token per frame" 这个具体 finding（这个可能 overfit to their setup），而是三个更深的东西：

1. **Design space calibration**: 在 frozen VLA + LoRA 这个 regime 下，systematic sweep 了 per-frame visual bandwidth，给了一个 calibrated design point。这种 careful ablation 本身有价值——大多数 paper 只给一个 point 不给 landscape。

2. **Bottleneck-rollout coupling principle**: bottleneck 让 joint objective tractable，joint objective 给 bottleneck control-relevant signal。这是个 general principle，可以 transfer 到其他 setting（不同 backbone、不同 task）。

3. **Conceptual reframe**: 把 "world model 在 VLA 中如何 parameterize" 从 "predict everything in pixel space" reframe 成 "predict minimal sufficient latent for action via joint generation"。这是个 modeling philosophy 贡献，类似 [JEPA](https://arxiv.org/abs/2301.08243) 对 representation learning 的 reframe。

**可能 impact**: 如果 1 token per frame 的结论 generalize（需要更多 backbone/task 验证），这会简化 future world-model-augmented VLA 设计。目前大家倾向于 "more is better"，这个 paper 说 "less is more under constrained budget"——和 [Chinchilla](https://arxiv.org/abs/2203.15556) 纠正 "bigger is better" 的精神类似。

**我的 open question**: 1 token 能否 encode **relational structure**？这篇 paper 的任务大多是 single-agent single-object manipulation。Multi-object interaction、tool use、bimanual manipulation 这些需要 object-object relation encoding的任务，1 token 是否够用？可能需要 object-centric decomposition（像 [SQA](https://arxiv.org/abs/2404.10370) 的 object slots），每个 object 一个 token，然后 world model 在 object slots 上 rollout。这是 [Object-centric world model](https://arxiv.org/abs/2305.08270) 的方向，和 OneWM-VLA 的 1-token principle 可能可以 combine。

---

## 参考资源

核心 paper:
- [π0 paper](https://arxiv.org/abs/2410.24164)
- [π0.5 paper](https://arxiv.org/abs/2504.16054)
- [Flow Matching](https://arxiv.org/abs/2210.02727)
- [LoRA](https://arxiv.org/abs/2106.09685)

相关 VLA:
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [CogAct](https://arxiv.org/abs/2411.19650)
- [CoT-VLA](https://arxiv.org/abs/2502.04353)

World models:
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [JEPA](https://arxiv.org/abs/2301.08243)
- [DreamVLA](https://arxiv.org/abs/2507.04447)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [DreamGen](https://arxiv.org/abs/2505.12705)

方法相关:
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Compressive Transformer](https://arxiv.org/abs/1911.05532)
- [Information Bottleneck](https://arxiv.org/abs/1612.00436)
- [BYOL](https://arxiv.org/abs/2006.07733)
- [Score-based diffusion](https://arxiv.org/abs/2011.13456)
- [Consistency Models](https://arxiv.org/abs/2303.01469)

Benchmark:
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [MetaWorld](https://arxiv.org/abs/2310.06031)

Scaling 和 theory:
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- [REPA](https://arxiv.org/abs/2410.06940)
- [Object-centric world models](https://arxiv.org/abs/2305.08270)

如果你想 deep dive 某个部分——比如 adaptive pooling 的 attention pattern 可视化、joint flow matching 的 ODE solver 选择、或者 1 token bottleneck 在 multi-object task 上的 failure mode——我可以再展开。这个 paper 的 design space 挺值得 explore 的。

---

# OneWM-VLA: 深入解析 Visual Bandwidth 与 World Model 的耦合设计

## 1. 这篇 paper 在回答什么问题？

这篇 paper 围绕一个很尖锐的 design question 展开：在 frozen VLA backbone 之上加 world module 的时候，**per-frame visual bandwidth 到底需要多大**？以及 **latent stream 应该如何与 action trajectory 耦合**？

作者给出的答案有点反直觉：在他们的实验设定下，**1 token per frame 就够了**，而且 token 数增加到 12 时 success rate 反而单调下降。耦合方式上，他们用 **joint flow matching** 把 latent 和 action 放在同一个 generator 里 co-evolve，避免 post-hoc auxiliary loss。

这个结论之所以有意思，是因为它 challenge 了 "world model 越大越好、视觉信息越丰富越好" 的默认直觉，并通过 systematic ablation 给出了一个 calibrated design point。

参考链接：
- π0 paper (Black et al.): https://arxiv.org/abs/2410.24164
- π0.5 paper (Physical Intelligence): https://arxiv.org/abs/2504.16054
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02727
- LoRA: https://arxiv.org/abs/2106.09685

---

## 2. 整体 architecture 直觉

OneWM-VLA 的核心是一个 **bottleneck–rollout coupling**。可以用两句话描述：

- **Bottleneck**：通过 Adaptive Attention Pooling，把每个 camera view 的 256 个 visual tokens 压成 **1 个 semantic token per frame**。这样 world-module 的 per-step token budget 与 horizon 无关 (horizon-invariant)，与视觉 resolution $N$ 解耦。

- **Rollout**：用 **joint flow matching** 把 future latent stream $\mathbf{z}$ 和 future action sequence $\mathbf{a}$ 放到同一个 vector field $v_\theta$ 里联合 denoise。latent 不再是 side product，而是 action trajectory 的 structural prior。

这个设计的关键 trick 在于：bottleneck 让 joint objective 在长 horizon 下 tractable（否则 256 tokens × H 会爆内存），而 joint objective 反过来给 bottleneck 一个 control-relevant 的监督信号（否则压出来的 token 可能丢了 task-relevant 信息）。两者互为支撑。

在推理时，所有模态都从 Gaussian noise 初始化并联合 denoise，但 **只有 action stream 被执行到 robot 上**，latent tokens 是 internal auxiliary trajectory，用于 structure action generation over horizon。

---

## 3. Adaptive Attention Pooling 的细节拆解

### 3.1 视觉编码

输入 $\mathbf{I}_i \in \mathbb{R}^{B \times T \times H \times W \times C}$，其中 $i \in \{r, w_1, w_2\}$ 分别是 third-person view 和两个 wrist view。通过 PaliGemma encoder $\mathcal{E}_\phi$：

$$\mathbf{X}_i = \mathcal{E}_\phi(\mathbf{I}_i) \in \mathbb{R}^{B \times T \times N \times D}$$

其中：
- $B$: batch size
- $T$: 时间步数 (frames)
- $N=256$: 每个 frame 的 visual token 数
- $D$: hidden dimension
- $\mathbf{x}_i^{(n)} \in \mathbb{R}^D$: 第 $i$ 个 view 的第 $n$ 个 token

### 3.2 Multi-strategy token pooling

作者的核心 insight 是：单一 pooling strategy 可能丢失某种 saliency。所以他们用了 **三种互补的 scoring functions**：

$$\phi_{\text{MAX}}(\mathbf{x}) = \max_d x^{(d)}, \quad \phi_{\text{SUM}}(\mathbf{x}) = \sum_{d=1}^D x^{(d)}, \quad \phi_{\text{LEARN}}(\mathbf{x}) = Q_\theta(\mathbf{x})$$

- $\phi_{\text{MAX}}$: 取 token 在 channel 维度上的最大响应，捕捉 peak activation（类似 max-pooling 的直觉）
- $\phi_{\text{SUM}}$: 取 token 在 channel 维度上的总和，捕捉 overall energy（类似 average pooling 的能量版本）
- $\phi_{\text{LEARN}}$: 一个 view-specific 的小 MLP $Q_\theta$，学习 task-aware saliency

每种策略在 token 维度上做 softmax (temperature $\tau$):

$$w_m^{(n)} = \frac{\exp(\phi_m(\mathbf{x}_i^{(n)})/\tau)}{\sum_{j=1}^N \exp(\phi_m(\mathbf{x}_i^{(j)})/\tau)}, \quad m \in \mathcal{M}=\{\text{MAX, SUM, LEARN}\}$$

然后 aggregation：

$$\mathbf{p}_{\text{Max}} = \max_n (w_{\text{MAX}}^{(n)} \mathbf{x}_i^{(n)}), \quad \mathbf{p}_m = \sum_{n=1}^N w_m^{(n)} \mathbf{x}_i^{(n)}, \quad m \in \{\text{SUM, LEARN}\}$$

这里有个细节：MAX 策略在 aggregation 时仍然用 max（element-wise max over token axis），而 SUM 和 LEARN 用 weighted sum。这样得到 3 个 pooled tokens $\{\mathbf{p}_m\}_{m \in \mathcal{M}} \subset \mathbb{R}^D$，每个对应一种 saliency 视角。

### 3.3 Adaptive view fusion

三个 pooled tokens 通过 learnable convex combination 融合成最终的 per-frame world token：

$$\mathbf{Z}_i = \sum_{m \in \mathcal{M}} \beta_m \mathbf{p}_m, \quad \beta_m = \frac{\exp(\alpha_m/\tau)}{\sum_{m'} \exp(\alpha_{m'}/\tau)}$$

- $\boldsymbol{\alpha} \in \mathbb{R}^{|\mathcal{M}|}$: 可训练 scalars
- $\beta_m$: softmax-normalized fusion weights
- $\mathbf{Z}_i \in \mathbb{R}^{B \times T \times 1 \times D}$: 单 view 单 frame 的 world token

实验里 training 后 fusion weights 收敛到 $\beta_{\text{LEARN}} \approx 0.48$–$0.57$, $\beta_{\text{MAX}} \approx 0.20$–$0.28$, $\beta_{\text{SUM}} \approx 0.20$–$0.28$。这说明 **LEARN 主导，但 MAX 和 SUM 仍然贡献**——这正是 Tab. 6 中 single-branch ablation 显示的（LEARN alone 50.42%, 三者合一 61.30%）。

### 3.4 为什么 adaptive design 重要？

Tab. 5 的 ablation 很关键：
- 完整 adaptive pooling: 93.3% (LIBERO Avg)
- Static average pooling: 72.8% (掉 20.5 pts, Long suite 掉得最多：41.4% vs 85.0%)
- No fusion logic: 47.6% (Long 掉到 4.4%)

**Input-dependent attention 是关键**——单纯的压缩本身不够，必须让 weights 随 instance 变化。这和 attention pooling 在 ViT 中的常见用法一致（[ViT attention pooling](https://arxiv.org/abs/2106.01549)），但作者额外强调 multi-strategy 互补的重要性。

---

## 4. Joint Flow Matching 的数学拆解

### 4.1 Flow Matching 背景

Flow matching 训练一个 continuous normalizing flow，回归 vector field $v_\theta$:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, \mathbf{x} \sim p_t}\left[\|\mathbf{v}_\theta(t, \mathbf{x}) - \mathbf{u}_t(\mathbf{x})\|^2\right]$$

- $t \in [0,1]$: flow time
- $p_t(\mathbf{x})$: probability path 连接 data distribution $p_0$ 和 prior $p_1$
- $\mathbf{v}_\theta(t, \mathbf{x})$: 网络预测的 velocity field
- $\mathbf{u}_t(\mathbf{x})$: target velocity field (一般 intractable)

因此用 Conditional Flow Matching (CFM)：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)}\left[\|v_\theta(t, x) - u_t(x|x_1)\|^2\right]$$

- $q(x_1)$: data distribution
- $p_t(x|x_1)$: Gaussian conditional paths
- $u_t(x|x_1)$: closed-form conditional fields

CFM 是 simulation-free 的，π0 backbone 就是继承这个 objective。

### 4.2 Joint probability path（关键设计）

OneWM-VLA 把 latent branch 和 action branch 都用 **optimal transport (OT) 直线插值**：

$$x_t^a = t\epsilon_a + (1-t)a, \quad x_t^z = t\epsilon_z + (1-t)z, \quad t \sim \text{Beta}(1.5, 1)$$

变量解释：
- $a \in \mathbb{R}^{H \times D_a}$: 未来 action sequence ($H$ 是 horizon, $D_a$ 是 action 维度)
- $z \in \mathbb{R}^{H \times D_z}$: 未来 latent world tokens
- $\epsilon_a, \epsilon_z \sim \mathcal{N}(0, I)$: 两个 branch 独立的 Gaussian noise
- $t \sim \text{Beta}(1.5, 1)$: 时间采样 schedule，让 $t$ 偏向 0 附近（data end），训练更聚焦在 action-relevant 区域
- $x_t^a, x_t^z$: 在 flow time $t$ 处的插值状态

Target velocity 是 constant：

$$u_t^a = \epsilon_a - a, \quad u_t^z = \epsilon_z - z$$

**两个 branch 共享同一个 flow time $t$ 和同一个 generator $v_\theta$**，这是 "joint" 的精髓。

### 4.3 Multi-objective flow-matching loss

Transformer 的输入是 interleaved sequence：
- 当前 tokens: $\{x_t^{z_r}, x_t^{z_{w_1}}, x_t^{z_{w_2}}, l_t, s_t\}$ (三个 view 的 latent, language $l$, robot state $s$)
- 未来 noisy queries: $\{x_{t+k}^{z_r}, x_{t+k}^{z_{w_1}}, x_{t+k}^{z_{w_2}}, x_{t+k}^a\}_{k=1}^h$

Loss:

$$\mathcal{L} = \lambda_a \mathbb{E}[\|v_\theta^a - u_t^a\|_1] + \sum_{i \in \{r, w_1, w_2\}} \lambda_i \mathbb{E}[\|v_\theta^z - u_t^z\|_1]$$

- $\lambda_a = 1.0$: action loss weight
- $\lambda_r = \lambda_{w_1} = \lambda_{w_2} = 0.1$: latent loss weights per view
- $\|\cdot\|_1$: L1 距离（Tab. 11 ablation 显示 L1/L1 比 L2/L2 高 5.7 pts，比 L1/L2 高 3.1 pts）

为什么 L1 比 L2 好？可能的 intuition：L1 对 outlier 更 robust，对于 multi-modal action distribution（机器人操作常有这种特性）更合适。这也呼应 [π0 paper](https://arxiv.org/abs/2410.24164) 中 flow matching 在 action 上的应用。

### 4.4 "Joint" 比 "Separate" 好在哪？

关键在于 **$v_\theta^z$ 和 $v_\theta^a$ 在 self-attention 中 co-evolve**，而不是 post-hoc 用 auxiliary loss 连接。Tab. 8 的 ablation 量化了这个点：

- OneWM-VLA (full joint): 58.09% (MetaWorld H=20)
- No latent branch (只生成 action): 43.04% (掉 15.05 pts, Very Hard 从 50% 掉到 10%)
- No latent loss (保留 latent input 但 $\mathcal{L}_{\text{latent}}=0$): 21.47% (掉 36.62 pts, Hard 从 60% 掉到 6%)

第二个 ablation 特别 informative：即使 latent tokens 作为 input 存在，没有 supervision 的话反而比完全没有 latent branch 还差。这说明 **latent supervision 是 coupling 的核心，而不仅是 added capacity**。

参数量上：OneWM-VLA 只在 π0 (2B) 上加 14.71M trainable params（projection layers + 3 个 fusion scalars），却超过 π0.5 (3B) 在所有三个 setting 上。这说明 gain 不能简单归因于 capacity。

---

## 5. 实验数据的深度解读

### 5.1 MetaWorld MT50 (Tab. 1)

四个 horizon $H \in \{5, 10, 25, 30\}$ 的对比：

| $H$ | OneWM-VLA | π0 | π0.5 | $\Delta_{\pi_0}$ | $\Delta_{\pi_0.5}$ |
|-----|-----------|-----|------|------------------|---------------------|
| 5 | 61.28 | 43.97 | 38.25 | +17.31 | +23.03 |
| 10 | 52.10 | 46.60 | 38.84 | +5.50 | +13.26 |
| 25 | 46.16 | 39.80 | 35.26 | +6.36 | +10.90 |
| 30 | 53.13 | 37.98 | 26.83 | +15.15 | +26.30 |

观察：
- π0 从 $H=5$ 到 $H=30$ 掉约 9 pts (43.97 → 37.98)
- π0.5 从 $H=5$ 到 $H=30$ 掉 11.42 pts (38.25 → 26.83)
- OneWM-VLA 从 $H=5$ 到 $H=30$ 掉 8.15 pts (61.28 → 53.13)

**OneWM-VLA 的 long-horizon 稳定性是其核心优势**。在 Very Hard tier 上 H=30 时，OneWM-VLA 60.0% vs π0 4.0% vs π0.5 4.0%——这是 15× 的差距，非常显著。

### 5.2 Per-frame Token Sweep (Tab. 4) - 最反直觉的结果

| Tokens/View | Avg % | FPS | Infer Tokens | Memory |
|-------------|-------|-----|--------------|--------|
| 256 (Full) | - | - | 15,390 | OOM |
| 12 | 20.54 | 0.13 | 750 | Stable |
| 6 | 33.85 | 0.56 | 390 | Stable |
| 3 | 41.86 | 1.21 | 210 | Stable |
| 1 | 53.13 | 4.81 | 90 | Stable |

**Success rate 随 token 数增加单调下降**。这是个非常 striking 的现象。作者给的 tentative reading：
- 小 latent 起到 implicit regularizer 作用
- 大 token 数在 30K steps 的 LoRA 训练 budget 下可能 underfit
- 更长训练或许能让大 token 数 viable (留作 future work)

我的额外 intuition：在 LoRA 设定下，新增 trainable params 有限，能拟合的额外容量也有限。如果 token 数过多，每个 token 的 effective gradient signal 被稀释，反而难以学到 control-relevant 的 representation。这和 [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) 中 "data/compute/token allocation" 的 trade-off 类似——只是这里是 latent dimension vs supervision signal 的分配。

### 5.3 Fisher Ratio 分析 (Appendix E)

$$F = \text{tr}(S_B)/\text{tr}(S_W)$$

- $S_B$: between-class scatter matrix
- $S_W$: within-class scatter matrix
- $S_B, S_W$ 都在 full feature vector 上算
- 每个 regime 把 $\text{tr}(S_W)$ normalize 到 1

结果：
- Before pooling (256 tokens, mean-aggregated): $F = 0.524$
- After pooling (1 token): $F = 0.405$
- 保留了约 77% 的 discriminative signal

**77% 的 class-level structure 在压缩后保留**，这解释了为什么 1 token 仍能 work——大多数 control-relevant 信息被保留，丢弃的主要是 photometric detail。

### 5.4 LIBERO (Tab. 2)

| Method | Spatial | Object | Goal | Long | Avg% |
|--------|---------|--------|------|------|------|
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| OneWM-VLA | 98.2 | 99.6 | 99.0 | 95.6 | 98.1 |

三个 short suite (Spatial/Object/Goal) 三者都在 95%+ 的 saturation 区间，差距不大。Long suite 是真正分水岭：OneWM-VLA 95.6% vs π0 85.2%（+10.4 pts）vs π0.5 92.4%（+3.2 pts）。

一个重要的 implementation detail (Tab. 12)：**所有 4 个 suite 用同一个 training checkpoint (train AH=20)**，只 tune inference AH 和 replan step。这表明 model 的 generalization 很强，不需要 per-task fine-tuning。

### 5.5 Real Piper Arm (Tab. 3)

| Method | P.Banana | F.Cloth | P.Drawer | Avg% |
|--------|----------|---------|----------|------|
| π0 (clean) | 100.0 | 20.0 | 30.0 | 50.0 |
| π0.5 (clean) | 100.0 | 25.0 | 50.0 | 58.3 |
| OneWM-VLA (clean) | 100.0 | 60.0 | 55.0 | 71.7 |

Fold Cloth 是最 demanding 的任务（cloth state 连续变化，long-horizon deformable manipulation）。OneWM-VLA 60% vs π0 20%——3 倍提升。在有 observation noise（lighting shift + 位置扰动 + distractors）时，OneWM-VLA 在 Fold Cloth 上 40% vs π0 0%——**baseline 完全崩了**，但 OneWM-VLA 还能撑住。

这印证了 paper 的核心 thesis：**world module 的价值在 long-horizon + 感知扰动 下最显著**。

### 5.6 Semantic vs Pixel Compression (Tab. 7)

| Method | Easy | Med. | Hard | V-Hard | Avg% |
|--------|------|------|------|--------|------|
| OneWM-VLA-pixel | 63.57 | 18.18 | 22.00 | 28.00 | 35.85 |
| OneWM-VLA (semantic) | 72.50 | 27.27 | 40.00 | 30.00 | 53.13 |

差距 +17.28 pts，Hard 上 40% vs 22%。这验证了 paper 的论点：**pixel-level compression 把 task-relevant 和 task-irrelevant patterns 同等对待，会保留 low-level noise**；而 semantic compression 在已经与 policy 对齐的 feature space 里操作，能滤掉噪声。

### 5.7 Inference-time Horizon Sweep (Tab. 13)

| Train/Infer | Replan step | Success% |
|-------------|-------------|----------|
| 20/15 (baseline) | 10 | 93.4 |
| 20/18 | 12 | 95.6 (peak) |
| 20/20 | 10 | 94.0 |

Success rate 在 94%–95.6% 的窄带内波动——这说明 latent predictive design 在 horizon 扩展时 per-step token budget 不增长，memory footprint 稳定。这是个很有意义的 scalability 性质。

---

## 6. 与相关工作的关系

### 6.1 Latent World Models 谱系

- **Dreamer** ([Hafner et al., 2019](https://arxiv.org/abs/1912.01603)): Recurrent state-space model，在低维 posterior 里 rollout。在 RL pipeline 内开发，没 transfer 到 frozen VLA adaptation。
- **JEPA** ([LeCun, 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)): 在 learned representation 里 predict，而非 pixel space。非生成式，用 contrastive/predictive loss。
- **DreamVLA** ([Zhang et al., 2025](https://arxiv.org/abs/2507.04447)): 与 OneWM-VLA 最接近的工作，但用的是 separate decoder 而非 joint flow matching。
- **WorldVLA** ([Cen et al., 2025](https://arxiv.org/abs/2506.21539)): Pixel-space autoregressive world model，per-step cost 随 horizon 增长。

OneWM-VLA 的差异化：**single flow-matching objective + 1 token/frame bottleneck**，专门为 frozen backbone + LoRA 设定设计。

### 6.2 VLA Backbone 谱系

- **OpenVLA** ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)): 7B, autoregressive action
- **π0** ([Black et al., 2026](https://arxiv.org/abs/2410.24164)): 2B, flow matching action
- **π0.5** ([Physical Intelligence, 2025](https://arxiv.org/abs/2504.16054)): 3B, open-world generalization
- **CogAct** ([Li et al., 2024](https://arxiv.org/abs/2411.19650)): VLA with diffusion action head
- **CoT-VLA** ([Zhao et al., 2025](https://arxiv.org/abs/2502.04353)): Visual chain-of-thought reasoning

OneWM-VLA 选 π0 是因为：(1) flow matching 天然支持 joint latent+action generation；(2) 2B 规模在 LoRA budget 下 tractable。

---

## 7. 我的几个 intuition 和延伸思考

### 7.1 为什么 1 token 就够？

这是 paper 最 striking 的点。我的 intuition：

1. **VLA backbone 已经做了重活**：PaliGemma encoder 已经把 224×224 图像压成 256 个 semantic tokens，这些 tokens 已经包含丰富的 task-relevant 信息。再压缩到 1 token 只是进一步 distill 出 "control-essential semantic"。

2. **Control-relevant 信息稀疏**：机器人操作中，task-relevant 的视觉信息（gripper position, object pose, target location）是高度 concentrated 的。1 个 well-pooled token 足以编码 "what's where and what to do next"。

3. **Joint flow matching 的 inductive bias**：latent 和 action 共享 generator，所以 latent 不需要独立 encode 所有信息——它只需要 encode 那些能 guide action 的信息。这是 "sufficient statistic for action" 的视角。

4. **Implicit regularization**：在 30K LoRA steps 下，更多 tokens 意味着更多参数要 fit，但 supervision signal 有限。1 token 的 bottleneck 强制 model 学到最 control-relevant 的 latent。

### 7.2 Joint vs Separate 的根本区别

**Separate decoder** 意味着 latent stream 和 action stream 是两个 head，通过 cross-attention 或 auxiliary loss 连接。问题：latent 优化方向不一定与 action 优化方向一致，容易学到 "video-realistic 但 control-irrelevant" 的 latent。

**Joint flow matching** 让两个 stream 在 self-attention 里 co-evolve，共享同一个 generator。这意味着 latent 必须能 "explain" action 的变化——如果 latent 学了 control-irrelevant 的内容，它对 action prediction 的 gradient 会被 push 掉。这是 "information bottleneck via shared computation" 的体现。

参考 [β-VAE](https://arxiv.org/abs/1606.05579) 的 disentanglement 视角：bottleneck + downstream task 一起监督，能学到 task-relevant 的 minimal sufficient statistic。

### 7.3 Long-horizon 为什么受益最大？

可以从 error accumulation 的角度理解：假设每步 action error 是 $\epsilon$，独立累加下 $H$ 步后 total error $\sim \epsilon\sqrt{H}$。World model 提供了一个 "imagined future" 让 policy 提前看到 consequence，相当于一个 implicit model predictive control (MPC)。

在短 horizon 下，这个 MPC 价值有限（receding horizon 不长）。但在 long horizon 下，每个 action 都需要考虑 future scene evolution，world model 的价值指数增长。这解释了为什么 OneWM-VLA 在 LIBERO-Long (H~18-20) 和 Fold Cloth (long-horizon deformable) 上优势最大。

### 7.4 与 Compressive Transformer / Memory 的关联

1 token per frame 的设计让人联想到 [Compressive Transformer](https://arxiv.org/abs/1911.05532) 和 [Set Transformer](https://arxiv.org/abs/1810.00825) 的思想：把长序列压缩成少数 "memory tokens"。OneWM-VLA 可以看作是 "memory = world model state" 的实现——每个 frame 贡献 1 个 token 到 rollout，整个 rollout 就是这 1×H×3 个 tokens 的 attention 序列。

未来一个自然方向：加入 [token memory mechanism](https://arxiv.org/abs/2402.17765) 让 long-horizon stability 和 per-step efficiency 同时改进（作者在 Limitations 里提到）。

### 7.5 L1 vs L2 loss 的 intuition

Tab. 11 显示 L1/L1 (action/latent) 比 L2/L2 高 5.7 pts。我的理解：

- Action distribution 是 multi-modal 的（同一 observation 可能有多个 valid action）
- L2 loss 对 outlier 敏感，会 regression to mean，丢失 multi-modality
- L1 loss 更 robust，保留 multi-modal structure
- 这和 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 用 flow matching + L1 的选择一致

### 7.6 Beta(1.5, 1) schedule 的含义

$t \sim \text{Beta}(1.5, 1)$ 让 $t$ 偏向 0（data end）。PDF 在 $t=0$ 处为 0，在 $t \to 1$ 处趋于 1.5。意味着训练时更多采样在 "接近 clean data" 的状态——这优先让 model 学好 "从 clean-ish 到 clean" 的精细 action generation，而 "from noise" 的粗粒度生成放在 background。

直觉：robot action 的精度需求远高于图像生成，所以让训练聚焦在 data-rich 区域。和 [Consistency Models](https://arxiv.org/abs/2303.01469) 中 "focus on data manifold" 的思想类似。

---

## 8. Limitations 和 Open Questions

作者诚实地列了几个 limitations：

1. **Single backbone**: 只在 π0 (2B) 上验证，更大 backbone (如 π0.5 3B, GR00T-N1) 上是否同样 work 未知。
2. **Fixed adaptation budget**: 30K LoRA steps，更长训练可能让大 token 数 viable (Tab. 4 的反转)。
3. **Perceptual complexity 中等**: MetaWorld/LIBERO/Piper 的视觉复杂度有限，更复杂的 scene（如 kitchen with many objects）是否仍只需 1 token 待验证。
4. **Per-step efficiency**: 现在是 4.81 FPS，对 real-time control (需要 30+ Hz) 还不够。

我自己的几个 open questions：
- **1 token 是否能 encode multi-object relation**？比如需要同时 manipulate 多个物体的任务，1 token 可能丢失 object-object relation。
- **Adaptive pooling 的 fusion weights 是 global 的**——如果不同 task 需要不同的 saliency view 怎么办？或许 task-conditioned fusion 能进一步提升。
- **Latent dimension $D_z$ 的选择**？paper 没讨论这个超参，可能用了 backbone default。
- **如何与 RL fine-tuning 结合**？[π_RL](https://arxiv.org/abs/2510.25889) 已经做了 flow-based VLA 的 online RL，OneWM-VLA 加 latent branch 后 RL 的 credit assignment 会更复杂。

---

## 9. 总结：这个 paper 的真正贡献

我的理解，这个 paper 的贡献可以分三层：

1. **Empirical finding**: 在 frozen VLA + LoRA 设定下，per-frame visual bandwidth 可以压到 1 token，且 monotonically better（在他们的 budget 下）。这是一个 calibrated design point，对未来 world-model-augmented VLA 工作有指导意义。

2. **Methodological design**: bottleneck-rollout coupling 是一个 general principle——bottleneck 让 joint objective tractable，joint objective 给 bottleneck control-relevant signal。这个 principle 可以 transfer 到其他 setting。

3. **Conceptual reframe**: 把 "world model 在 VLA 中如何 parameterize" 从 "predict everything in pixel space" 重新 frame 成 "predict minimal sufficient latent for action via joint generation"。这是一个 modeling philosophy 的贡献。

如果你（Andrej）希望深入某个部分，比如 adaptive pooling 的 attention pattern 分析、joint flow matching 的 ODE solver 选择、或者 long-horizon error accumulation 的理论分析，我可以进一步展开。这个 paper 的 design space 很值得 explore，特别是 "1 token per frame" 这个 counter-intuitive 的 finding，背后可能藏着关于 VLA 和 world model 耦合的更深层 principle。

参考资源：
- [π0 GitHub (if available)](https://github.com/Physical-Intelligence/openpi)
- [Flow Matching tutorial](https://arxiv.org/abs/2302.00482)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [MetaWorld benchmark](https://arxiv.org/abs/2310.06031)
- [Adaptive Attention Pooling in ViT](https://arxiv.org/abs/2106.01549)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [JEPA](https://arxiv.org/abs/2301.08243)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
