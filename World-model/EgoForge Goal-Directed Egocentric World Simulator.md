---
source_pdf: EgoForge Goal-Directed Egocentric World Simulator.pdf
paper_sha256: 611e22192e9692b34db8146a141e328ff949705e3b10b9c3382d365226567e57
processed_at: '2026-08-04T02:31:48-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EgoForge 人话讲解

## 一、这 paper 在干嘛

想象你戴着一副 smart glasses,我给你拍一张你眼前的照片,告诉你 "去打开冰箱拿牛奶倒进杯子里",再给你一张别人在同一个厨房拍的照片做参考。EgoForge 能 generate 出一段第一人称视频,模拟你完成这个 task 的整个过程——手伸出去、打开冰箱、拿出牛奶、倒进杯子,全程第一人称视角,物体不会变形,场景不会漂移。

**核心挑战**:egocentric video 非常难 generate,因为:

1. **第一人称视角动得快**——你转头看冰箱,视角就大角度变化,driving video 那种稳定前向视角的 assumption 全部失效
2. **手和物体频繁交互**——手会遮挡物体,物体会被打开/移动/变形,state 必须保持因果一致
3. **Goal-directed behavior 是 multi-step 的**——"open fridge AND pour milk" 这种 procedural task,单帧 prediction 捕捉不到 procedural dependency

现有方法要么需要 synchronized multi-view video streams(EgoWorld, Exo2Ego-V),要么需要 hand pose trajectories(Handi),要么需要 camera paths(EgoDreamer)。这些在真实 wearable 场景中很难获取。EgoForge 只要 **一张 ego image + 一条 text instruction + 一张 optional exo image**,就能 generate 出 10 秒的 goal-directed rollout。

---

## 二、Architecture 直觉

整个 system 分三块,我用做菜来类比:

### 2.1 Generator backbone: Wan2.2-5B DiT

这是 base model,一个 video diffusion transformer。在 latent space 操作,把 video encode 成 latent $\mathbf{z}_0$,然后加 noise 做 diffusion:

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

- $\bar{\alpha}_t = \prod_{s \leq t} \alpha_s$:cumulative noise schedule,t 越大 noise 越多
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:标准 Gaussian noise
- $t \sim \mathcal{U}(0,1)$:diffusion step

conditioning 通过 **channel concatenation** 注入:

$$
\tilde{\mathbf{z}}_t = \mathrm{Concat}(\mathbf{z}_t, \mathbf{f}_{\mathrm{ego}}, \mathbf{f}_{\mathcal{C}})
$$

- $\mathbf{f}_{\mathrm{ego}}$:DINOv3 提取的 ego image features(frozen)
- $\mathbf{f}_{\mathcal{C}}$:instruction text + exo image 融合后的 embedding

training objective 是 **velocity prediction**:

$$
\mathcal{L}_D = \mathbb{E}_{t, \mathbf{z}_t, \epsilon}\left[\|\epsilon - v_\theta(\tilde{\mathbf{z}}_t, t, \mathcal{C})\|_2^2\right]
$$

这里 $v_\theta$ 是 conditional velocity field,预测的是 "从 noisy latent 回到 clean latent 的方向",这是 flow-matching 的标准做法,比 noise prediction 数值上更稳定。

### 2.2 Geometry Weak Supervision (GWS): 给 DiT 装上 3D 眼睛

**问题**:diffusion model 只会 match pixel distribution,没有 3D consistency 的概念。生成 241 帧视频时,object 容易变形,viewpoint 容易 drift。

**解决方案**:找一个已经学好 3D geometry 的 teacher model(**VGGT**, Wang et al. 2025),把它的中间 features 作为 "geometry target",让 DiT 的中间 layers 去模仿。

具体做法:
- VGGT 第 $l$ 层 features: $\mathbf{g}_l \in \mathbb{R}^{N \times Q \times D_g}$
  - $N$:temporal tokens
  - $Q$:spatial tokens  
  - $D_g$:feature dimension
- DiT 第 $l$ 层 features: $\mathbf{h}_l \in \mathbb{R}^{N^l \times Q^l \times D_h}$
- 因为 resolution 不同,需要 learnable projection: $\mathbf{p}_l = \Pi_l(\mathbf{h}_l)$

**Angular alignment loss**(方向对齐):

$$
\mathcal{L}^{\mathrm{ang}} = -\frac{1}{LNQ}\sum_{l,n,q}\cos(\mathbf{g}_{l,n,q}, \mathbf{p}_{l,n,q})
$$

- $L$:选择的 layer 数
- $N, Q$:temporal 和 spatial token 数
- $l, n, q$:layer index, temporal index, spatial index

**Scale alignment loss**(幅度对齐,防止 collapse):

$$
\mathcal{L}^{\mathrm{sca}} = \frac{1}{LNQ}\sum_{l,n,q}\|\hat{\mathbf{g}}_{l,n,q} - \mathbf{g}_{l,n,q}\|_2^2
$$

- $\hat{\mathbf{g}}_l = \rho_l(\tilde{\mathbf{p}}_l)$:通过 learned linear head $\rho_l$ 得到 geometry prediction
- $\tilde{\mathbf{p}}_l = \mathbf{p}_l / (\|\mathbf{p}_l\|_2 + \varepsilon)$:normalize 后的 projected feature

总 loss:

$$
\mathcal{L}_G = \zeta_1 \mathcal{L}^{\mathrm{ang}} + \zeta_2 \mathcal{L}^{\mathrm{sca}}
$$

- $\zeta_1, \zeta_2$:balancing coefficients

**Intuition**:cosine loss 只管方向,容易让所有 feature collapse 到同一方向。scale loss 强制 magnitude 也要匹配,防止 trivial solution。这是 **REPA**(Yu et al. 2025) 和 **Geometry Forcing**(Wu et al. 2025) 的核心 idea——给 diffusion training 加一个 "你的中间 representations 也要能 reconstruct 3D geometry" 的 inductive bias。

### 2.3 VideoDifusionNFT: 最关键的创新

这是 paper 最大的 contribution。把 **DiffusionNFT**(Zheng et al. 2025) 从 image domain 扩展到 video domain,做 **trajectory-level reward-guided refinement**。

#### 为什么需要 trajectory-level

paper 在 Appendix A 给了三个原因,我用大白话解释:

**1. Reward sparsity & temporal credit assignment**

goal completion reward 只能在整个 trajectory 结束时评估——"put the can back" 成不成功要看最后一帧。如果 241 帧的 video 失败了,你不知道是第 50 帧的 denoising step 出问题,还是第 150 帧的 conditioning fusion 出问题。per-frame reward(比如 pixel-wise loss)捕捉不到这种 procedural, long-horizon objective。需要 RL 把 global success signal propagate 回整个 generative process。

**2. Preventing goals drift**

long video generation 容易 "忘记" initial conditioning。第 1 帧的 scene geometry,到第 200 帧可能已经完全 drift 了——object flickering, background mutating。standard diffusion objective 没有显式机制保证 frame 1 到 frame T 的 consistency。VideoDifusionNFT 作为 global supervisor,penalize drift 的 sequence。

**3. "Shortcut" problem in multimodal fusion**

多模态 conditioning 容易出现 cue dominance。model 可能发现:"我只模仿 exo image 的 visual style 就能骗过 training loss,根本不用管 text instruction 说什么"。结果是 visually plausible 但 semantically wrong 的 video。multi-dimensional reward + negative-aware finetuning 强制 model 真正 fuse 所有 input,disincentivize 这种 shortcut。

#### 数学框架详解

**Step 1: Reward normalization**

给定 supervised-finetuned policy $\pi^{\mathrm{old}}$,对每个 condition $c$,生成 $K$ 个 rollout candidates $\mathcal{X}_c = \{\mathbf{x}_{1:T}^{(k)}\}_{k=1}^K$。每个 candidate 有 reward $\mathcal{R}_{\mathrm{total}}^{(k)}(\mathbf{x}_{1:T}^{(k)}, c)$。

per-condition 的 expected reward:

$$
\mu_c = \frac{1}{K}\sum_{k=1}^K \mathcal{R}_{\mathrm{total}}^{(k)}(\mathbf{x}_{1:T}^{(k)}, c)
$$

normalized optimality probability:

$$
\tilde{\mathcal{R}}_{\mathrm{total}}^{(k)} = \frac{1}{2}\left[1 + \mathrm{clip}\left(\frac{\mathcal{R}_{\mathrm{total}}^{(k)} - \mu_c}{Z_c}, -1, 1\right)\right]
$$

- $Z_c > 0$:per-condition 的 normalization scale
- clip 到 $[-1,1]$ 然后 $\frac{1}{2}[1 + \cdot]$ 映射到 $[0,1]$
- 记 $r(\mathbf{x}^{(k)}, c) := \tilde{\mathcal{R}}_{\mathrm{total}}^{(k)}$,满足 $r \in [0,1]$

**为什么 per-condition normalization?** 不同 condition 的 reward scale 差异很大——简单任务 reward 普遍高,难任务普遍低。如果不 normalize,RL 会偏向 easy condition(挑软柿子捏)。per-condition z-score 把每个 condition 内部的 relative goodness 提取出来,RL 优化 "在每个 condition 下相对表现",和 absolute difficulty 无关。

**Step 2: 正负后验构造**

per-condition optimality mass:

$$
p_{\pi^{\mathrm{old}}}(o=1 \mid c) := \mathbb{E}_{\mathbf{x} \sim \pi^{\mathrm{old}}(\cdot|c)}[r(\mathbf{x}, c)]
$$

positive posterior(公式 6):

$$
\pi^+(\mathbf{x} \mid c) = \frac{r(\mathbf{x}, c)}{p_{\pi^{\mathrm{old}}}(o=1 \mid c) + \varepsilon} \pi^{\mathrm{old}}(\mathbf{x} \mid c)
$$

negative posterior(公式 7):

$$
\pi^-(\mathbf{x} \mid c) = \frac{1 - r(\mathbf{x}, c)}{1 - p_{\pi^{\mathrm{old}}}(o=1 \mid c) + \varepsilon} \pi^{\mathrm{old}}(\mathbf{x} \mid c)
$$

- $\varepsilon > 0$:避免除零

这是 importance reweighting:$\pi^+$ 把 high-reward 样本放大,low-reward 样本缩小;$\pi^-$ 反过来。可以验证 $\pi^+ \succ \pi^{\mathrm{old}} \succ \pi^-$ in expected reward(由 Cauchy-Schwarz 不等式):

$$
\mathbb{E}_{\pi^+}[r] = \frac{\mathbb{E}_{\pi^{\mathrm{old}}}[r^2]}{\mathbb{E}_{\pi^{\mathrm{old}}}[r]} \geq \mathbb{E}_{\pi^{\mathrm{old}}}[r]
$$

**Step 3: Conditional optimality at intermediate state**

定义:

$$
\alpha(\mathbf{z}_t, c) = \mathbb{E}[r(\mathbf{x}, c) \mid \mathbf{z}_t, c]
$$

- $\mathbf{z}_t$:intermediate forward state(noisy latent at step $t$)
- 这是 conditional expectation:"given we're at this latent state, expected final reward 是多少?"

improvement direction:

$$
\Delta(\mathbf{z}_t, c, t) = [1 - \alpha(\mathbf{z}_t, c)](v^{\mathrm{old}} - v^-) = \alpha(\mathbf{z}_t, c)(v^+ - v^{\mathrm{old}})
$$

这两个表达式等价(可从 posterior 关系推导)。物理意义:

- 当 $\alpha \to 1$(high expected reward):$\Delta \approx v^+ - v^{\mathrm{old}}$,向 positive 方向推
- 当 $\alpha \to 0$(low expected reward):$\Delta \approx v^{\mathrm{old}} - v^-$,远离 negative 方向
- $\alpha$ 起到 **adaptive guidance strength** 的作用——在 confident 区域轻推,在 unconfident 区域强推

guided target field:

$$
v^*(\mathbf{z}_t, c, t) = v^{\mathrm{old}}(\mathbf{z}_t, c, t) + \frac{1}{\beta}\Delta(\mathbf{z}_t, c, t)
$$

- $\beta > 0$:guidance strength。$\beta$ 小则 guidance 强,$\beta$ 大则弱

**Step 4: Negative-aware flow-matching loss**

定义 mixed velocity fields:

$$
v_\theta^+ = (1-\beta)v^{\mathrm{old}} + \beta v_\theta, \quad v_\theta^- = (1+\beta)v^{\mathrm{old}} - \beta v_\theta
$$

- $v_\theta^+$:positive interpolation,在 $v^{\mathrm{old}}$ 和 $v_\theta$ 之间
- $v_\theta^-$:negative extrapolation,在 $v^{\mathrm{old}}$ 反方向推 $v_\theta$

loss(公式 7):

$$
\mathcal{L}(\theta) = \mathbb{E}_{c, \mathbf{z}_t}\left[\rho \|v_\theta^+ - v^*\|_2^2 + (1-\rho)\|v_\theta^- - v^*\|_2^2\right]
$$

- $\rho \sim \mathrm{Ber}(\alpha(\mathbf{z}_t, c))$:Bernoulli 采样,probability 等于 conditional optimality $\alpha$

optimal solution(公式 8):

$$
v_{\theta^*} = v^{\mathrm{old}} + \frac{2r(\mathbf{x}, c) - 1}{\beta}(v^* - v^{\mathrm{old}})
$$

- $2r - 1 \in [-1, 1]$:signed optimality signal
- $r > 0.5$:朝 $v^*$ 方向移动
- $r < 0.5$:远离 $v^*$ 方向

#### 这套设计的 intuition

1. **不是单纯模仿 positive,而是 explicit repulsion from negative**。$v_\theta^-$ 项强制 $v_\theta$ 远离 low-reward 方向,即使 high-reward 区域也保持 negative pressure。这避免 standard RLHF 的 mode collapse 和 reward hacking。

2. **$\rho$ 的随机性引入 exploration**。在 high optimality state($\alpha$ 大)时更可能 sample positive term,在 low optimality state 更可能 sample negative term。这是一种 state-dependent exploration strategy。

3. **$v^*$ 已经包含了 $\alpha$-adaptive guidance**。loss 实际上在做:"student 要同时学会 positive 速度场和 negative 速度场,并且 negative term 的权重在低 optimality 区域更大"。

这和 **DPO**(Direct Preference Optimization)的 idea 很像——同时拉 positive 拒 negative,但 EgoForge 把它放到 continuous-time diffusion setting 下,且 guidance strength 是 state-dependent。

#### Reward 设计细节

paper 用 **VLM 作为 non-parametric evaluator**(看 Appendix B 的 prompt,就是给 GPT-4o 类模型视频和参考图,让它打分)。四个 reward,每个 0-2.0 分:

| Reward | Components | 作用 |
|--------|-----------|------|
| $\mathcal{R}_{\mathrm{goal}}$ | $R_{\mathrm{task}}$ + $R_{\mathrm{align}}$ | task completion + visual alignment with target |
| $\mathcal{R}_{\mathrm{env}}$ | $R_{\mathrm{consist}}$ + $R_{\mathrm{contam}}$ | environment consistency + freedom from target contamination |
| $\mathcal{R}_{\mathrm{temp}}$ | $R_{\mathrm{phys}}$ + $R_{\mathrm{logic}}$ | physics plausibility + causal logic |
| $\mathcal{R}_{\mathrm{per}}$ | $\mathcal{R}_{\mathrm{PSNR}}$ + $\mathcal{R}_{\mathrm{FVD}}$ + $\mathcal{R}_{\mathrm{LPIPS}}$ | pixel + distributional + perceptual fidelity |

**Contamination Score** 的设计特别有意思:如果 target image 在不同 environment,model 容易 "leak" target 的 environment 到 generated video。$\mathcal{R}_{\mathrm{contam}}$ 专门 penalize 这种 leakage,强制 model 只用 target 的 "action outcome",忽略 "environment context"。

---

## 三、X-Ego Benchmark

paper 自己 curate 了 X-Ego,因为现有 benchmark 缺少 fine-grained egocentric conditioning 评估。

### 数据源
- **Nymeria**(Ma et al., ECCV 2024):大规模 multimodal egocentric daily motion in the wild
  - Link: https://www.projectaria.com/nymeria/
  - arXiv: https://arxiv.org/abs/2406.00568
- **Ego-Exo4D**(Grauman et al., CVPR 2024):first-and-third-person skilled human activity
  - Link: https://ego-exo4d-data.org/
  - arXiv: https://arxiv.org/abs/2406.14395

### 构建 pipeline

1. 基于 action annotations 切分视频,统一为 **10 秒 clip**
2. 用 expert temporal-action summarizer 选 **stationary atomic actions**(限制 hand-on-object manipulations,排除 locomotion / speech / idle)
3. 用 multimodal LLM refine caption,生成 **4-sentence structured format**:
   - Sentence 1: Setup(hand 和 object 的 visual appearance)
   - Sentence 2: Action(fine-grained motion trajectory)
   - Sentence 3: Reaction(object 的 physical response)
   - Sentence 4: Outcome(final state)

### 规模
- **15,000 training samples**
- **100 held-out test samples**(覆盖所有 interaction category)

### 为什么不用现成 dataset

Ego4D / EPIC-KITCHENS 没有配 pair 的 exocentric view,且 caption 是 coarse 的。Ego-Exo4D 有 exo-ego pair 但是 caption 没有 fine-grained hand-object dynamics annotation。X-Ego 通过 caption refinement pipeline 把 coarse caption ground 到 visual evidence,得到 fine-grained description。

---

## 四、实验结果解读

### 4.1 主对比

| Model | DINO↑ | CLIP↑ | SSIM↑ | LPIPS↓ | FVD↓ | Flow MSE↓ | PSNR↑ |
|-------|-------|-------|-------|--------|------|-----------|-------|
| EgoDreamer | 42.35 | 25.40 | 0.58 | 0.35 | 580.45 | 8.15 | 15.20 |
| Handi | 31.12 | 18.25 | 0.42 | 0.52 | 912.30 | 14.50 | 12.85 |
| Cosmos | 49.42 | 29.77 | 0.70 | 0.26 | 448.12 | 6.40 | 18.73 |
| HunyuanVideo | 53.54 | 29.43 | 0.71 | 0.26 | 384.31 | 6.10 | 18.88 |
| WAN2.2 | 53.99 | 35.69 | 0.72 | 0.23 | 322.17 | 5.78 | 20.44 |
| **EgoForge** | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

相比最强 baseline WAN2.2:
- **DINO-Score**: +13.5%(semantic alignment)
- **CLIP-Score**: +10.1%(text-video alignment)
- **SSIM**: +9.7%(structural similarity)
- **PSNR**: +17.8%(reconstruction quality)
- **LPIPS**: -35%(perceptual error 下降)
- **FVD**: -43%(temporal coherence 大幅提升)
- **Flow MSE**: -51%(motion fidelity 大幅提升)

**FVD 和 Flow MSE 的巨大提升** 说明 EgoForge 在 **temporal modeling** 上的改进尤其明显,这正是 trajectory-level reward optimization 的 target。

### 4.2 Progressive enhancement 对比

| Model | DINO | CLIP | SSIM | LPIPS | FVD | Flow MSE | PSNR |
|-------|------|------|------|-------|-----|----------|------|
| WAN2.2+EV(naive exo view) | 52.91 | 35.11 | 0.71 | 0.27 | 352.41 | 6.25 | 20.05 |
| WAN2.2+TT(text-only adaptation) | 54.80 | 36.20 | 0.73 | 0.25 | 310.57 | 5.60 | 20.64 |
| WAN2.2+CI(他们的 conditioning + GWS) | 58.92 | 38.05 | 0.76 | 0.18 | 218.72 | 3.92 | 22.87 |
| **EgoForge(+ VideoDiffusionNFT)** | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

每个 component 的贡献:
- **naive visual augmentation(+EV)** 其实 hurt 了 performance,说明简单 concat 不 work
- **text-only domain adaptation(+TT)** 有一定提升
- **+CI(他们的 structured conditioning + Geometry Weak Supervision)**:FVD 从 310.57 降到 218.72(降 30%),Flow MSE 从 5.60 降到 3.92(降 30%)。这是 **geometry grounding 的主要贡献**
- **+VideoDiffusionNFT**:FVD 从 218.72 降到 182.25(再降 16.7%),DINO 从 58.92 升到 61.25。这是 **trajectory-level reward 的主要贡献**

### 4.3 Ablation 验证

| FT | GWS | VideoDiffusionNFT | DINO | CLIP | SSIM | LPIPS | FVD | Flow MSE | PSNR |
|----|-----|-------------------|------|------|------|-------|-----|----------|------|
| ✓ | ✗ | ✗ | 56.81 | 37.10 | 0.74 | 0.21 | 260.89 | 4.82 | 21.92 |
| ✓ | ✓ | ✗ | 58.92 | 38.05 | 0.76 | 0.18 | 218.72 | 3.92 | 22.87 |
| ✓ | ✓ | ✓ | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

- **GWS 贡献**:DINO +2.11,FVD -42.17(-16.2%),Flow MSE -0.90(-18.7%),LPIPS -0.03。GWS 主要改善 **spatial structure 和 temporal stability**
- **VideoDiffusionNFT 贡献**:DINO +2.33,FVD -36.47(-16.7%),Flow MSE -1.09(-27.8%),LPIPS -0.03。VideoDiffusionNFT 在 **temporal coherence** 上贡献最大,符合 trajectory-level optimization 的预期

### 4.4 Reward 组件 ablation

| Reward | DINO | CLIP | SSIM | LPIPS | FVD | Flow MSE | PSNR |
|--------|------|------|------|-------|-----|----------|------|
| $\mathcal{R}_{\mathrm{goal}}$ only | 59.62 | 38.49 | 0.78 | 0.16 | 205.96 | 3.48 | 23.48 |
| $\mathcal{R}_{\mathrm{env}}$ only | 60.67 | 39.05 | 0.78 | 0.16 | 200.49 | 3.43 | 23.60 |
| $\mathcal{R}_{\mathrm{temp}}$ only | 60.78 | 39.11 | 0.78 | 0.16 | 213.25 | 3.70 | 23.72 |
| $\mathcal{R}_{\mathrm{per}}$ only | 60.32 | 38.80 | 0.77 | 0.18 | 204.13 | 3.48 | 23.17 |
| **All** | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

分析:
- 移除 $\mathcal{R}_{\mathrm{per}}$:visual metrics(SSIM, PSNR, LPIPS)下降最明显。LPIPS 从 0.15 升到 0.18(+20%)
- 移除 $\mathcal{R}_{\mathrm{temp}}$:FVD 从 182.25 升到 213.25(+17%),Flow MSE 从 2.83 升到 3.70(+30.7%)。$\mathcal{R}_{\mathrm{temp}}$ 对 **temporal consistency** 最关键
- 移除 $\mathcal{R}_{\mathrm{goal}}$:DINO 和 CLIP 下降最明显(DINO -1.63,CLIP -0.81)。$\mathcal{R}_{\mathrm{goal}}$ 对 **semantic alignment** 最关键
- 移除 $\mathcal{R}_{\mathrm{env}}$:全面 moderate 下降,说明 env consistency 是 foundational

四个 reward 各司其职,组合起来才达到 best performance。验证了 multi-dimensional reward design 的必要性。

### 4.5 User Study

| Model | Q. | F. | M. | E. | A. |
|-------|----|----|----|----|----|
| Cosmos* | 3.29 | 2.54 | 3.07 | 2.47 | 2.19 |
| Hunyuan* | 3.46 | 2.86 | 3.72 | 3.16 | 3.08 |
| WAN2.2* | 3.22 | 3.48 | 3.82 | 4.07 | 3.15 |
| **EgoForge** | **4.58** | **4.71** | **4.25** | **4.48** | **4.75** |

- **Alignment 4.75**:语义对齐远超 baseline(WAN2.2 是 3.15),trajectory-level reward 显著改善 goal alignment
- **Fidelity 4.71**:object identity preservation 远超,geometry supervision + reward 让 object 不会变形
- WAN2.2 在 Environment(4.07)上其实已经不错,但 EgoForge 还是更高(4.48)

---

## 五、Implementation Details

### 训练 pipeline

**Stage 1: Denoising Fine-Tuning (FT)**
- 13,000 samples
- frozen: DINOv3 backbone + VGGT backbone
- trainable: DiT 主干(+GWS projection layers)
- LoRA rank 32
- Adam,lr 1e-4
- bf16 mixed precision
- 8 H100 GPUs
- batch size 1
- resolution 720p
- **241 frames per sequence @ 24 fps**(约 10 秒)
- 10 epochs,~108 hours

**Stage 2: VideoDifusionNFT**
- 2,000 samples(比 Stage 1 少很多,因为 RL finetuning 数据 efficient)
- frozen: 所有非 DiT 组件
- trainable: 只有 diffusion model
- 每个 sample 生成 **6 个 video variations** 获取 diverse trajectories 和 reward signals

### 241 frames 的设计

241 帧 @ 24fps = ~10 秒。这和 X-Ego 的 clip 长度(10 秒)对齐。10 秒是 stationary atomic action 的 typical duration。

---

## 六、Real-World Smart-Glasses 实验

paper 用 **DigiLens ARGO** smart glasses 做 OOD 测试,这是 egocentric world model 第一次在真实 wearable device 上验证。

### ARGO specs
- 48MP camera with autofocus
- Optical Image Stabilization (OIS) + Electronic Image Stabilization (EIS)
- 4×4 pixel binning + enhanced low-light
- 5-microphone beamforming array
- Qualcomm Snapdragon XR2 Platform

Link: https://www.digilens.com/argo

### 测试任务
1. "Pour into the cup...put the can back"
2. "Jump to the pool...arms forward"
3. "Take a marker...draw a circle"
4. "Take a bottle of water...on the box"

### 为什么重要

之前的 egocentric world models 都在 in-domain data 上测试,没在 real-world OOD setting 验证。EgoForge 在 ARGO 上能 transfer exocentric cues 并 follow high-level semantic intent,说明 model 学到的是 **generalizable goal-directed behavior**,dataset-specific pattern。

---

## 七、Intuition 总结

### 7.1 为什么 minimal input 能 work

EgoForge 不需要 dense supervision(camera trajectory, synchronized multi-view),因为它把 **goal-directed control** 和 **3D consistency** 内化到 model 的 training objective 里:
- Geometry Weak Supervision 让 DiT 学到 implicit 3D reasoning,不需要显式 camera pose
- VideoDifusionNFT 通过 reward 把 "task completion" 和 "physical consistency" 编码到 sampling policy 里,不需要 explicit action labels

### 7.2 为什么 trajectory-level 比 per-frame 更重要

egocentric video 的成功标准是 **procedural**(multi-step task 完成)和 **holistic**(整段 video 物理一致),per-frame reward 容易被 model 通过 single-frame hack 满足——每帧都很 realistic 但 motion 不连贯。trajectory-level reward 强制 model 在整个 rollout 上保持 goal alignment 和 physical consistency,这正是 VideoDifusionNFT 的设计哲学。

### 7.3 为什么 negative-aware finetuning 比 standard RLHF 好

standard RLHF 只 maximize positive reward,容易 mode collapse(总生成 high-reward 的 trivial pattern)和 reward hacking(满足 reward 但不真正完成任务)。negative-aware finetuning 同时:
- 拉向 positive posterior $\pi^+$
- 推离 negative posterior $\pi^-$

这相当于在 reward landscape 中 maintain diversity,避免 collapse。$\alpha(\mathbf{z}_t, c)$ 的 state-dependent guidance 让 high-confidence 区域轻推、low-confidence 区域强推,这是一种 **adaptive exploration-exploitation**。

### 7.4 Geometry supervision 作为 inductive bias

REPA 和 Geometry Forcing 的核心 insight:diffusion model 自己学 3D consistency 很难,把已经学好 geometry 的 teacher(VGGT)作为 representation target,diffusion training 会更 efficient,generation quality 更高。这是 **knowledge distillation**,但只 distill representation direction,不 distill final output。

---

## 八、对 Karpathy 直觉的连接

你之前在 tweet 和 lecture 中讲过 world model 的几个关键点,EgoForge 在这些点上都有 interesting 的 design choice:

1. **"World model 要 capture causal dynamics"**:EgoForge 用 trajectory-level reward $\mathcal{R}_{\mathrm{temp}}$ 专门 enforce causal logic 和 physics plausibility,per-frame L2 loss 做不到这点。

2. **"RL finetuning 是给 generative model 加 intent 的方式"**:VideoDifusionNFT 本质上是把 diffusion sampling 当作 policy,用 RL 优化 trajectory reward。这和 DPO / RLHF 在 LLM 上的思路一致,但在 continuous-time diffusion setting 下,且引入 negative-aware 双向优化。

3. **"Geometry / 3D consistency 是 video model 的 fundamental bottleneck"**:EgoForge 用 VGGT teacher 做 REPA-style alignment,这和 LeCun 的 JEPA 思路有 spirit 上的相似——不直接 predict pixel,align representation 到 geometry-aware 的 target。

4. **"Multi-modal conditioning 容易 cue dominance"**:你在 VLM 讨论中提到过这种 shortcut learning。EgoForge 的 multi-dimensional reward + negative pressure 显式 combat 这种 shortcut,强制 model 真正 fuse 多模态信号。

整体上 EgoForge 把 REPA、DiffusionNFT、VLM-as-judge 三个方向 merge 到 egocentric world simulation 这个具体问题上,工程执行度高,实验结果 convincing。最值得关注的创新是 **VideoDifusionNFT 的 negative-aware trajectory-level guidance**,这个 idea 在 principle 上可以 generalize 到其他 long-horizon generation 任务(robotics planning, embodied navigation 等)。

---

## 九、Reference Links

- **EgoForge project page**: https://plan-lab.github.io/egoforge
- **Wan2.2 (base generator)**: https://github.com/Wan-Video/Wan2.2 | arXiv: https://arxiv.org/abs/2503.20314
- **VGGT (geometry teacher)**: https://vgg-t.github.io/ | arXiv: https://arxiv.org/abs/2503.11651
- **REPA (Representation Alignment)**: https://sihyun.me/REPA/ | arXiv: https://arxiv.org/abs/2410.06940
- **Geometry Forcing**: arXiv: https://arxiv.org/abs/2507.07982
- **DiffusionNFT (image domain baseline)**: arXiv: https://arxiv.org/abs/2509.16117
- **Ego-Exo4D dataset**: https://ego-exo4d-data.org/ | arXiv: https://arxiv.org/abs/2406.14395
- **Nymeria dataset**: https://www.projectaria.com/nymeria/ | arXiv: https://arxiv.org/abs/2406.00568
- **DigiLens ARGO**: https://www.digilens.com/argo
- **HunyuanVideo**: https://github.com/Tencent/HunyuanVideo | arXiv: https://arxiv.org/abs/2412.03603
- **Cosmos (NVIDIA)**: https://github.com/nvidia-cosmos | arXiv: https://arxiv.org/abs/2501.03575
- **DINOv2 (used for DINO-Score)**: https://github.com/facebookresearch/dinov2
- **Flow Matching for Generative Modeling**: arXiv: https://arxiv.org/abs/2210.02747
- **EgoDreamer**: arXiv: https://arxiv.org/abs/2411.08380
- **Handi**: arXiv: https://arxiv.org/abs/2412.04189
- **EgoWorld**: arXiv: https://arxiv.org/abs/2506.17896
- **PlayerOne (related egocentric world simulator)**: NeurIPS 2025
- **CLIP-Score**: arXiv: https://arxiv.org/abs/2103.00020
- **LPIPS**: arXiv: https://arxiv.org/abs/1801.03924
- **FVD**: arXiv: https://arxiv.org/abs/1812.01758
- **DPO (Direct Preference Optimization)**: arXiv: https://arxiv.org/abs/2305.18290
- **JEPA (LeCun's Joint Embedding Predictive Architecture)**: https://ai.facebook.com/blog/yann-lecun-proposes-a-new-path-toward-autonomous-machine-intelligence/

---

## 十、Limitations 与 Potential Issues

paper 没有详细讨论 limitations,根据 method 可以推测:

1. **VLM 作为 reward evaluator 的成本**:每个 sample 要生成 6 个 video,每个 video 要 VLM 评估 4 个 dimension。Stage 2 training 成本高(虽然 paper 说只用 2000 samples)。

2. **VLM reward 的 noise**:VLM 评分有 stochasticity,可能引入 bias。paper 没有讨论 reward model 的 calibration。

3. **241 frames 的 horizon 限制**:10 秒的 rollout 对于 complex multi-step task(比如 "open fridge → take milk → pour → close fridge")可能不够长。

4. **Exocentric view 的 assumption**:虽然是 optional,但 Figure 6 显示有 exo view 时 quality 明显更好。在真实 wearable scenario 中,exo view 不一定 available。

5. **Real-world experiment 只有 qualitative**:paper 没有给出 ARGO 上的 quantitative metric,只有 visualization。

6. **Reward hacking 的风险**:虽然 negative-aware finetuning 缓解了 mode collapse,但 VLM-as-judge 仍可能被 adversarial generation 骗过——比如 generate 出 visually plausible 但 semantically wrong 的 video 恰好让 VLM 打高分。paper 没有讨论这种 robustness。

---

## 十一、个人 take

从工程角度,EgoForge 是一个相当 solid 的工作。它把三个 recent direction(REPA, DiffusionNFT, VLM-as-judge)merge 到 egocentric world simulation 这个具体问题上,每个 component 都有清晰的 ablation 验证,实验结果 convincing。

从 research 角度,**VideoDifusionNFT 的 negative-aware trajectory-level guidance** 是最有价值的 contribution。这个 idea 本质上是把 DPO 的 "同时拉 positive 拒 negative" 思路放到 continuous-time diffusion setting 下,且 guidance strength 是 state-dependent($\alpha(\mathbf{z}_t, c)$)。这个 framework 在 principle 上可以 generalize 到:
- **Robotics planning**:generate long-horizon manipulation trajectories,用 task success 作为 reward
- **Embodied navigation**:generate navigation rollouts,用 goal-reaching 作为 reward
- **Dialogue generation**:generate long conversations,用 user satisfaction 作为 trajectory-level reward
- **Code generation**:generate multi-step program traces,用 test pass 作为 trajectory reward

从 egocentric vision 角度,EgoForge 把 "goal-directed behavior" 从 "dense supervision" 中解放出来,用 minimal input(ego image + text + optional exo image)就能 generate plausible rollout。这对 real-world wearable application(AR glasses, robotics teleoperation 等)是重要的 step forward。

你 Karpathy 一直强调 "world model 要理解 causal dynamics 而非 appearance prediction",EgoForge 的 trajectory-level reward + causal logic penalty 正是朝这个方向走的具体实现。虽然距离真正的 "causal world model" 还有距离(比如 object permanence, physical interaction 的 explicit modeling 仍然依赖 VLM 的 implicit knowledge),但这个 direction 是 promising 的。

---

# EgoForge: Goal-Directed Egocentric World Simulator 深度讲解

## 一、Core Motivation 与 Problem Framing

EgoForge 想解决的问题是 **first-person goal-directed video simulation**。给定一张 egocentric image $\mathbf{x}_{1:k}$(通常是 1 帧)、一条 high-level instruction $y$、以及一张 optional 的 exocentric reference image $\mathbf{x}^{exo}$,生成一段能够体现 "用户执行指定任务" 的第一人称视频 rollout $\mathbf{x}_{k+1:T}$。

形式上建模为:

$$
p_\theta(\mathbf{x}_{k+1:T} \mid \mathbf{x}_{1:k}) = \prod_{t=k+1}^{T} p_\theta(\mathbf{x}_t \mid \mathbf{x}_{<t}, \mathcal{C})
$$

其中 conditioning context $\mathcal{C} = \{\mathbf{x}_{1:k}, y, \mathbf{x}^{exo}\}$。这里关键的一点是 **autoregressive over time but with global conditioning**,并且 **不需要 camera trajectory / pose / synchronized multi-view streams at inference**。

### 为什么 egocentric 难
1. **Rapid viewpoint changes**:第一人称视角下,头部和身体运动会引起大角度 view 变化,不像 driving video 有相对稳定的前向视角。
2. **Frequent hand-object interactions**:手会遮挡物体,物体状态会变化(打开、变形、位移),这种 state change 必须在 video 中保持因果一致。
3. **Goal-directed procedure depends on latent intent**:"open the fridge and pour milk" 这种 multi-step 行为,单帧 prediction 无法捕捉 procedural dependency,需要 trajectory-level reasoning。

### 现有方法的三个 fundamental limitations
1. **Dense supervision requirements**:EgoWorld、Exo2Ego-V、EgoX 需要 synchronized multi-view video streams;Handi 需要 hand pose trajectories;EgoDreamer 需要 camera paths。这些在 unconstrained wearable scenarios 中很难获得。
2. **Limited goal-directed control**:大多数 video diffusion model 用 short text prompt 或者 low-level action(joint controls)做 conditioning,无法表达 "open fridge → pour milk" 这种 multi-step semantic intent。
3. **Weak physical grounding**:video diffusion 优化 visual realism 但是缺乏 spatial coherence,3D-awareness 不足,导致 egocentric motion 不一致。

EgoForge 的策略:用 **geometry-aware DiT backbone** + **trajectory-level reward-guided refinement** 同时解决 (2) 和 (3),用 **minimal static inputs** 解决 (1)。

---

## 二、Architecture Overview

整个 pipeline 分为三个组件:

```
[ego img] + [exo img] + [instruction]
        │
        ├──► Conditioning Encoder (DINOv3 + CLIP-like text enc)
        │              │
        │              ▼
        │      fused embedding f_C
        │              │
        ▼              ▼
   noisy latent z_t  ────►  Wan2.2-5B DiT blocks
                                 │
                                 ├── Geometry Weak Supervision (VGGT teacher)
                                 │       ├── L_ang (cosine align)
                                 │       └── L_sca (scale align)
                                 │
                                 ▼
                          denoised latent → video rollout
                                 │
                                 ▼
                  VideoDiffusionNFT refinement
                  (trajectory reward: goal, env, temp, per)
```

### 2.1 Diffusion-Based Egocentric Generator

base model 是 **Wan2.2-5B**,一个 video DiT。在 latent space 操作,先 encode video 到 latent $\mathbf{z}_0 = \mathrm{Enc}(\mathbf{x}_{k+1:T})$。

采用 **variance-preserving flow-matching** (Lipman et al., 2022):

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

- $\bar{\alpha}_t = \prod_{s \leq t} \alpha_s$:cumulative noise schedule
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:standard Gaussian noise
- $t \sim \mathcal{U}(0,1)$:sampled diffusion step

reverse process:

$$
p_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathcal{C}) = \mathcal{N}(\mathbf{z}_{t-1}; \mu_\theta(\mathbf{z}_t, t, \mathcal{C}), \Sigma_t)
$$

- $\mu_\theta$:learned reverse step mean
- $\Sigma_t$:fixed variance from noise schedule

conditioning 通过 **adaptive normalization** 和 **cross-attention** 注入。具体做法是在 channel dimension 上 concat:

$$
\tilde{\mathbf{z}}_t = \mathrm{Concat}(\mathbf{z}_t, \mathbf{f}_{\mathrm{ego}}, \mathbf{f}_{\mathcal{C}})
$$

- $\mathbf{f}_{\mathrm{ego}}$:ego image 的 visual features(由 DINOv3 提取,frozen)
- $\mathbf{f}_{\mathcal{C}}$:instruction 和 exo image 融合后的 embedding

timestep $t$ 通过 learned time embedding $\gamma(t)$ 注入。

training objective 是 **velocity prediction loss**:

$$
\mathcal{L}_D = \mathbb{E}_{t, \mathbf{z}_t, \epsilon}\left[ \|\epsilon - v_\theta(\tilde{\mathbf{z}}_t, t, \mathcal{C})\|_2^2 \right]
$$

这里 $v_\theta$ 是 conditional velocity field。注意是 velocity 而不是 noise prediction,这是 flow-matching 的标准做法,数值上更稳定,sample efficiency 更好。

### 2.2 Geometry Weak Supervision (GWS)

这是借鉴 **REPA** (Yu et al., 2025) 和 **Geometry Forcing** (Wu et al., 2025) 的 idea:让 diffusion backbone 的中间 features **align 到一个已经学好 3D geometry representations 的 teacher**(这里是 **VGGT**,Wang et al., 2025)。

#### 为什么需要 geometry supervision
diffusion model 在 video generation 中容易丢失 3D consistency,因为它的 objective 是像素分布匹配,没有显式的 3D 信号。VGGT 是一个 visual geometry grounded transformer,能从单张或多张图片预测 camera pose、depth、point map 等多种 3D geometry 信号。把它的中间 features 当作 "geometry prior",通过 alignment 让 DiT 的中间 layers 隐式地 encode geometry 信息。

#### 公式细节

VGGT 在第 $l$ 层的 features: $\mathbf{g}_l \in \mathbb{R}^{N \times Q \times D_g}$

- $N$:temporal tokens 数量
- $Q$:spatial tokens 数量
- $D_g$:feature dimension
- $L$:选择的 layer 数

DiT 在第 $l$ 层的 hidden activations: $\mathbf{h}_l \in \mathbb{R}^{N \times Q^l \times D_h}$

由于两个 backbone 在不同 resolution 操作,引入 learnable projection operator:

$$
\Pi_l: \mathbb{R}^{N^l \times Q^l \times D_h} \to \mathbb{R}^{N \times Q \times D_g}
$$

通过 **spatiotemporal resampling** + **channel projection** 实现。projected feature:

$$
\mathbf{p}_l = \Pi_l(\mathbf{h}_l)
$$

**Angular alignment loss** (cosine):

$$
\mathcal{L}^{\mathrm{ang}} = -\frac{1}{LNQ} \sum_{l,n,q} \cos(\mathbf{g}_{l,n,q}, \mathbf{p}_{l,n,q})
$$

这只约束方向,不约束 magnitude,容易 collapse 到 trivial solution(比如所有 feature 都 normalize 到同一方向)。所以加 scale alignment:

先 normalize: $\tilde{\mathbf{p}}_l = \mathbf{p}_l / (\|\mathbf{p}_l\|_2 + \varepsilon)$

通过 learned linear head $\rho_l$ 得到 geometry prediction: $\hat{\mathbf{g}}_l = \rho_l(\tilde{\mathbf{p}}_l)$

**Scale alignment loss**:

$$
\mathcal{L}^{\mathrm{sca}} = \frac{1}{LNQ} \sum_{l,n,q} \|\hat{\mathbf{g}}_{l,n,q} - \mathbf{g}_{l,n,q}\|_2^2
$$

总的 geometry coordination objective:

$$
\mathcal{L}_G = \zeta_1 \mathcal{L}^{\mathrm{ang}} + \zeta_2 \mathcal{L}^{\mathrm{sca}}
$$

- $\zeta_1, \zeta_2$:balancing coefficients

**intuition**:这是 auxiliary representation alignment。VGGT 是 frozen teacher,DiT 是 student。student 在 denoising 的同时被 forced 去 produce features that are geometrically meaningful。这相当于给 diffusion training 加了一个 inductive bias:"你要 denoise,但你的中间 representations 也要能 reconstruct geometry"。结果就是生成的 video 在 3D consistency 上更强,object 不会变形,viewpoint 不会 drift。

---

## 三、VideoDifusionNFT:核心创新

这是 paper 的最大 contribution。它把 **DiffusionNFT** (Zheng et al., 2025) 从 image domain 扩展到 video domain,做一个 **trajectory-level reward-guided refinement**。

### 3.1 为什么需要 trajectory-level

paper 在 Appendix A 给了三个原因:

1. **Reward sparsity & temporal credit assignment**:goal completion reward 只能在整个 trajectory 结束时评估,无法 per-frame 评估。如果 241 帧的 video 失败,无法定位是哪个 denoising step 或哪个 conditioning fusion 出问题。需要 RL 把 global success signal propagate 回整个 generative process。

2. **Preventing goals drift**:long video generation 容易 "forget" initial conditioning。standard diffusion objective 没有显式机制保证 frame 1 到 frame T 的 scene geometry 或 background 保持一致。VideoDifusionNFT 作为 global supervisor,penalize drift 的 sequence。

3. **"Shortcut" problem in multimodal fusion**:多模态 conditioning 容易出现 cue dominance。比如 model 可能 overfit 到 visual input(产生 stylistically similar video 但 ignore textual procedural instructions)。multi-dimensional reward + negative-aware finetuning 强制 model 平衡所有 input,disincentivize shortcut solutions。

### 3.2 数学框架

#### Reward normalization

给定 supervised-finetuned policy $\pi^{\mathrm{old}}$,对每个 condition $c \in \mathcal{C}$,生成 $K$ 个 rollout candidates $\mathcal{X}_c = \{\mathbf{x}_{1:T}^{(k)}\}_{k=1}^K$。每个 candidate 有 reward $\mathcal{R}_{\mathrm{total}}^{(k)}(\mathbf{x}_{1:T}^{(k)}, c)$。

per-condition 的 expected reward 经验估计:

$$
\mu_c = \mathbb{E}_{\mathbf{x} \sim \pi^{\mathrm{old}}(\cdot|c)}[\mathcal{R}_{\mathrm{total}}(\mathbf{x}, c)] \approx \frac{1}{K}\sum_{k=1}^K \mathcal{R}_{\mathrm{total}}^{(k)}(\mathbf{x}_{1:T}^{(k)}, c)
$$

normalized optimality probability(公式 5):

$$
\tilde{\mathcal{R}}_{\mathrm{total}}^{(k)} = \frac{1}{2}\left[1 + \mathrm{clip}\left(\frac{\mathcal{R}_{\mathrm{total}}^{(k)} - \mu_c}{Z_c}, -1, 1\right)\right]
$$

- $Z_c > 0$:normalized local reward scale,保证 $\tilde{\mathcal{R}}_{\mathrm{total}}^{(k)} \in [0,1]$
- clip 到 $[-1, 1]$ 然后 $\frac{1}{2}[1 + \cdot]$ 映射到 $[0, 1]$
- 这相当于一个 **per-condition 的 z-score 然后 sigmoid-like squash**

记 $r(\mathbf{x}^{(k)}, c) := \tilde{\mathcal{R}}_{\mathrm{total}}^{(k)}$,满足 $r \in [0, 1]$。

**intuition**:为什么 per-condition normalization?因为不同 condition 的 reward scale 可能差异很大(简单任务 reward 普遍高,难任务普遍低)。如果不 normalize,RL 会偏向 easy condition。per-condition z-score 把每个 condition 内部的 relative goodness 提取出来,让 RL 优化 "在每个 condition 下相对表现"。

#### 正负后验构造

per-condition optimality mass:

$$
p_{\pi^{\mathrm{old}}}(o=1 \mid c) := \mathbb{E}_{\mathbf{x} \sim \pi^{\mathrm{old}}(\cdot|c)}[r(\mathbf{x}, c)]
$$

positive posterior(公式 6):

$$
\pi^+(\mathbf{x} \mid c) = \frac{r(\mathbf{x}, c)}{p_{\pi^{\mathrm{old}}}(o=1 \mid c) + \varepsilon} \pi^{\mathrm{old}}(\mathbf{x} \mid c)
$$

negative posterior(公式 7):

$$
\pi^-(\mathbf{x} \mid c) = \frac{1 - r(\mathbf{x}, c)}{1 - p_{\pi^{\mathrm{old}}}(o=1 \mid c) + \varepsilon} \pi^{\mathrm{old}}(\mathbf{x} \mid c)
$$

- $\varepsilon > 0$:避免除零
- 这相当于 importance reweighting:$\pi^+$ 把高 reward 的样本放大,低 reward 的样本缩小;$\pi^-$ 反过来

可以验证 $\pi^+ \succ \pi^{\mathrm{old}} \succ \pi^-$ in expected reward,因为:

$$
\mathbb{E}_{\pi^+}[r] = \frac{\mathbb{E}_{\pi^{\mathrm{old}}}[r^2]}{\mathbb{E}_{\pi^{\mathrm{old}}}[r]} \geq \mathbb{E}_{\pi^{\mathrm{old}}}[r]
$$

由 Cauchy-Schwarz / Jensen 不等式。

#### Conditional optimality at intermediate state

定义:

$$
\alpha(\mathbf{z}_t, c) = \mathbb{E}[r(\mathbf{x}, c) \mid \mathbf{z}_t, c]
$$

- $\mathbf{z}_t$:intermediate forward state(noisy latent at step $t$)
- 这是 conditional expectation,表示 "given we're at this latent state, what's the expected final reward?"

improvement direction:

$$
\Delta(\mathbf{z}_t, c, t) = [1 - \alpha(\mathbf{z}_t, c)](v^{\mathrm{old}} - v^-) = \alpha(\mathbf{z}_t, c)(v^+ - v^{\mathrm{old}})
$$

这两个表达式等价(可以从 posterior 关系推导)。它的物理意义:

- 当 $\alpha \to 1$(high expected reward):$\Delta \approx v^+ - v^{\mathrm{old}}$,向 positive 方向推
- 当 $\alpha \to 0$(low expected reward):$\Delta \approx v^{\mathrm{old}} - v^-$,远离 negative 方向
- $\alpha$ 起到 **adaptive guidance strength** 的作用

guided target field:

$$
v^*(\mathbf{z}_t, c, t) = v^{\mathrm{old}}(\mathbf{z}_t, c, t) + \frac{1}{\beta}\Delta(\mathbf{z}_t, c, t)
$$

- $\beta > 0$:guidance strength。$\beta$ 小则 guidance 强,$\beta$ 大则弱

#### Negative-aware flow-matching loss

定义 mixed velocity fields:

$$
v_\theta^+ = (1-\beta)v^{\mathrm{old}} + \beta v_\theta, \quad v_\theta^- = (1+\beta)v^{\mathrm{old}} - \beta v_\theta
$$

- $v_\theta^+$:positive interpolation,在 $v^{\mathrm{old}}$ 和 $v_\theta$ 之间拉
- $v_\theta^-$:negative extrapolation,在 $v^{\mathrm{old}}$ 反方向推 $v_\theta$

loss(公式 7):

$$
\mathcal{L}(\theta) = \mathbb{E}_{c, \mathbf{z}_t}\left[\rho \|v_\theta^+ - v^*\|_2^2 + (1-\rho)\|v_\theta^- - v^*\|_2^2\right]
$$

- $\rho \sim \mathrm{Ber}(\alpha(\mathbf{z}_t, c))$:Bernoulli 采样,probability 等于 conditional optimality

optimal solution(公式 8):

$$
v_{\theta^*} = v^{\mathrm{old}} + \frac{2r(\mathbf{x}, c) - 1}{\beta}(v^* - v^{\mathrm{old}})
$$

- $2r - 1 \in [-1, 1]$:signed optimality signal
- 当 $r > 0.5$:朝 $v^*$ 方向移动
- 当 $r < 0.5$:远离 $v^*$ 方向

**intuition**:这个 loss 设计的妙处在于:

1. **不是单纯模仿 positive,而是 explicit repulsion from negative**。$v_\theta^-$ 项强制 $v_\theta$ 远离 low-reward 的方向,即使在高 reward 区域也保持 negative pressure。这避免了 standard RLHF 中的 **mode collapse 和 reward hacking**。

2. **$\rho$ 的随机性引入 exploration**。在 high optimality state($\alpha$ 大)时更可能 sample positive term,在 low optimality state 更可能 sample negative term。这是一种 **state-dependent exploration strategy**。

3. **$v^*$ 已经包含了 $\alpha$-adaptive guidance**。所以 loss 实际上在做:"student 要同时学会 positive 速度场和 negative 速度场,并且 negative term 的权重在低 optimality 区域更大"。

这和 DPO 的想法很像(DPO 同时拉 positive 拒 negative),但在 diffusion continuous-time setting 下,且 guidance strength 是 state-dependent。

### 3.3 Reward 设计

paper 用 **VLM 作为 non-parametric evaluator**(看 Appendix B 的 prompt,就是给 GPT-4o 之类模型视频和参考图,让它打分)。四个 reward:

| Reward | Score | Components |
|--------|-------|-----------|
| $\mathcal{R}_{\mathrm{goal}}$ | 0-2.0 | $R_{\mathrm{task}}$ + $R_{\mathrm{align}}$(task completion + visual alignment with target) |
| $\mathcal{R}_{\mathrm{env}}$ | 0-2.0 | $R_{\mathrm{consist}}$ + $R_{\mathrm{contam}}$(environment consistency + freedom from target contamination) |
| $\mathcal{R}_{\mathrm{temp}}$ | 0-2.0 | $R_{\mathrm{phys}}$ + $R_{\mathrm{logic}}$(physics plausibility + causal logic) |
| $\mathcal{R}_{\mathrm{per}}$ | 0-2.0 | $\mathcal{R}_{\mathrm{PSNR}}$ + $\mathcal{R}_{\mathrm{FVD}}$ + $\mathcal{R}_{\mathrm{LPIPS}}$ |

每个 reward 是 composite 的,这样 VLM 可以分维度评估,降低 single-score 的 noise。

**Contamination Score** 的设计很有意思:如果 target image 在不同的 environment,model 容易 "leak" target 的 environment 到 generated video。$\mathcal{R}_{\mathrm{contam}}$ 专门 penalize 这种 leakage,强制 model 只用 target 的 "action outcome" 而非 "environment context"。

---

## 四、X-Ego Benchmark

paper 自己 curate 了 X-Ego,因为现有 benchmark 缺少 fine-grained egocentric conditioning 评估。

### 数据源
- **Nymeria** (Ma et al., ECCV 2024):大规模 multimodal egocentric daily motion in the wild
- **Ego-Exo4D** (Grauman et al., CVPR 2024):first-and-third-person skilled human activity

### 构建 pipeline
1. 基于 action annotations 切分视频,统一为 10s clip
2. 用 expert temporal-action summarizer 选 stationary atomic actions(限制 hand-on-object manipulations,排除 locomotion / speech / idle)
3. 用 multimodal LLM refine caption,生成 4-sentence structured format:
   - Sentence 1: Setup(hand 和 object 的 visual appearance)
   - Sentence 2: Action(fine-grained motion trajectory)
   - Sentence 3: Reaction(object 的 physical response)
   - Sentence 4: Outcome(final state)

### 规模
- 15,000 training samples
- 100 held-out test samples(覆盖所有 interaction category)

### 为什么不用现成 dataset
Ego4D / EPIC-KITCHENS 没有配 pair 的 exocentric view,且 caption 是 coarse 的。Ego-Exo4D 有 exo-ego pair 但是 caption 没有 fine-grained hand-object dynamics annotation。X-Ego 通过 caption refinement pipeline 把 coarse caption ground 到 visual evidence,得到 fine-grained description。

---

## 五、实验结果深度解读

### 5.1 主对比

| Model | DINO↑ | CLIP↑ | SSIM↑ | LPIPS↓ | FVD↓ | Flow MSE↓ | PSNR↑ |
|-------|-------|-------|-------|--------|------|-----------|-------|
| EgoDreamer | 42.35 | 25.40 | 0.58 | 0.35 | 580.45 | 8.15 | 15.20 |
| Handi | 31.12 | 18.25 | 0.42 | 0.52 | 912.30 | 14.50 | 12.85 |
| Cosmos | 49.42 | 29.77 | 0.70 | 0.26 | 448.12 | 6.40 | 18.73 |
| HunyuanVideo | 53.54 | 29.43 | 0.71 | 0.26 | 384.31 | 6.10 | 18.88 |
| WAN2.2 | 53.99 | 35.69 | 0.72 | 0.23 | 322.17 | 5.78 | 20.44 |
| **EgoForge** | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

相比最强 baseline WAN2.2:
- DINO-Score: +13.5%
- CLIP-Score: +10.1%
- SSIM: +9.7%
- PSNR: +17.8%
- LPIPS: -35%
- **FVD: -43%**(这是 temporal coherence 的 key metric)
- **Flow MSE: -51%**(motion fidelity 的 key metric)

FVD 和 Flow MSE 的巨大提升说明 EgoForge 在 **temporal modeling** 上的改进尤其明显,这正是 trajectory-level reward optimization 的 target。

### 5.2 Progressive enhancement 对比

| Model | DINO | CLIP | SSIM | LPIPS | FVD | Flow MSE | PSNR |
|-------|------|------|------|-------|-----|----------|------|
| WAN2.2+EV(naive exo view) | 52.91 | 35.11 | 0.71 | 0.27 | 352.41 | 6.25 | 20.05 |
| WAN2.2+TT(text-only adaptation) | 54.80 | 36.20 | 0.73 | 0.25 | 310.57 | 5.60 | 20.64 |
| WAN2.2+CI(他们的 conditioning + GWS) | 58.92 | 38.05 | 0.76 | 0.18 | 218.72 | 3.92 | 22.87 |
| **EgoForge(+ VideoDiffusionNFT)** | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

可以看到每个 component 的贡献:
- naive visual augmentation(+EV)其实 hurt 了 performance,说明简单 concat 不 work
- text-only domain adaptation(+TT)有一定提升
- **+CI(他们的 structured conditioning + Geometry Weak Supervision)**:FVD 从 310.57 降到 218.72(降 30%),Flow MSE 从 5.60 降到 3.92(降 30%)。这是 **geometry grounding 的主要贡献**
- **+VideoDiffusionNFT**:FVD 从 218.72 降到 182.25(再降 16.7%),DINO 从 58.92 升到 61.25。这是 **trajectory-level reward 的主要贡献**

### 5.3 Ablation 验证

| FT | GWS | VideoDiffusionNFT | DINO | CLIP | SSIM | LPIPS | FVD | Flow MSE | PSNR |
|----|-----|-------------------|------|------|------|-------|-----|----------|------|
| ✓ | ✗ | ✗ | 56.81 | 37.10 | 0.74 | 0.21 | 260.89 | 4.82 | 21.92 |
| ✓ | ✓ | ✗ | 58.92 | 38.05 | 0.76 | 0.18 | 218.72 | 3.92 | 22.87 |
| ✓ | ✓ | ✓ | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

- **GWS 贡献**:DINO +2.11,FVD -42.17(-16.2%),Flow MSE -0.90(-18.7%),LPIPS -0.03。GWS 主要改善 spatial structure 和 temporal stability。
- **VideoDiffusionNFT 贡献**:DINO +2.33,FVD -36.47(-16.7%),Flow MSE -1.09(-27.8%),LPIPS -0.03。VideoDiffusionNFT 在 **temporal coherence** 上贡献最大,这符合 trajectory-level optimization 的预期。

### 5.4 Reward 组件 ablation

| Reward | DINO | CLIP | SSIM | LPIPS | FVD | Flow MSE | PSNR |
|--------|------|------|------|-------|-----|----------|------|
| $\mathcal{R}_{\mathrm{goal}}$ only | 59.62 | 38.49 | 0.78 | 0.16 | 205.96 | 3.48 | 23.48 |
| $\mathcal{R}_{\mathrm{env}}$ only | 60.67 | 39.05 | 0.78 | 0.16 | 200.49 | 3.43 | 23.60 |
| $\mathcal{R}_{\mathrm{temp}}$ only | 60.78 | 39.11 | 0.78 | 0.16 | 213.25 | 3.70 | 23.72 |
| $\mathcal{R}_{\mathrm{per}}$ only | 60.32 | 38.80 | 0.77 | 0.18 | 204.13 | 3.48 | 23.17 |
| **All** | **61.25** | **39.30** | **0.79** | **0.15** | **182.25** | **2.83** | **24.08** |

分析:
- 移除 $\mathcal{R}_{\mathrm{per}}$:visual metrics(SSIM, PSNR, LPIPS)下降最明显。LPIPS 从 0.15 升到 0.18(+20%)
- 移除 $\mathcal{R}_{\mathrm{temp}}$:FVD 从 182.25 升到 213.25(+17%),Flow MSE 从 2.83 升到 3.70(+30.7%)。$\mathcal{R}_{\mathrm{temp}}$ 对 temporal consistency 最关键
- 移除 $\mathcal{R}_{\mathrm{goal}}$:DINO 和 CLIP 下降最明显(DINO -1.63,CLIP -0.81)。$\mathcal{R}_{\mathrm{goal}}$ 对 semantic alignment 最关键
- 移除 $\mathcal{R}_{\mathrm{env}}$:全面 moderate 下降,说明 env consistency 是 foundational

四个 reward 各司其职,组合起来才达到 best performance。这也验证了 multi-dimensional reward design 的必要性。

### 5.5 User Study

| Model | Q. | F. | M. | E. | A. |
|-------|----|----|----|----|----|
| Cosmos* | 3.29 | 2.54 | 3.07 | 2.47 | 2.19 |
| Hunyuan* | 3.46 | 2.86 | 3.72 | 3.16 | 3.08 |
| WAN2.2* | 3.22 | 3.48 | 3.82 | 4.07 | 3.15 |
| **EgoForge** | **4.58** | **4.71** | **4.25** | **4.48** | **4.75** |

- **Alignment 4.75**:语义对齐远超 baseline(WAN2.2 是 3.15),说明 trajectory-level reward 显著改善 goal alignment
- **Fidelity 4.71**:object identity preservation 远超,说明 geometry supervision + reward 让 object 不会变形
- WAN2.2 在 Environment(4.07)上其实已经不错,但 EgoForge 还是更高(4.48)。说明 EgoForge 不仅 motion 强,environment stability 也强

---

## 六、Implementation Details

### 训练 pipeline

**Stage 1: Denoising Fine-Tuning (FT)**
- 13,000 samples
- frozen: DINOv3 backbone + VGGT backbone
- trainable: DiT 主干(+GWS projection layers)
- LoRA rank 32
- Adam,lr 1e-4
- bf16 mixed precision
- 8 H100 GPUs
- batch size 1
- resolution 720p
- 241 frames per sequence @ 24 fps(约 10 秒)
- 10 epochs,~108 hours

**Stage 2: VideoDifusionNFT**
- 2,000 samples(比 Stage 1 少很多,因为 RL finetuning 数据 efficient)
- frozen: 所有非 DiT 组件
- trainable: 只有 diffusion model
- 每个 sample 生成 6 个 video variations 获取 diverse trajectories 和 reward signals

### 241 frames 的设计
241 帧 @ 24fps = ~10 秒。这和 X-Ego 的 clip 长度(10 秒)对齐。10 秒是 stationary atomic action 的 typical duration。

---

## 七、Real-World Smart-Glasses 实验

paper 用 **DigiLens ARGO** smart glasses 做 OOD 测试,这是 egocentric world model 第一次在真实 wearable device 上验证。

### ARGO specs
- 48MP camera with autofocus
- Optical Image Stabilization (OIS) + Electronic Image Stabilization (EIS)
- 4×4 pixel binning + enhanced low-light
- 5-microphone beamforming array
- Qualcomm Snapdragon XR2 Platform

### 测试任务
1. "Pour into the cup...put the can back"
2. "Jump to the pool...arms forward"
3. "Take a marker...draw a circle"
4. "Take a bottle of water...on the box"

### 为什么重要
之前的 egocentric world models 都在 in-domain data 上测试,没在 real-world OOD setting 验证。EgoForge 在 ARGO 上能 transfer exocentric cues 并 follow high-level semantic intent,说明 model 学到的是 **generalizable goal-directed behavior**,不是 dataset-specific pattern。

---

## 八、Intuition 总结

### 8.1 为什么 minimal input work
EgoForge 不需要 dense supervision(camera trajectory, synchronized multi-view),因为它把 **goal-directed control** 和 **3D consistency** 内化到 model 的 training objective 里:
- Geometry Weak Supervision 让 DiT 学到 implicit 3D reasoning,不需要显式 camera pose
- VideoDifusionNFT 通过 reward 把 "task completion" 和 "physical consistency" 编码到 sampling policy 里,不需要 explicit action labels

### 8.2 为什么 trajectory-level 比 per-frame 更重要
egocentric video 的成功标准是 **procedural**(multi-step task 完成)和 **holistic**(整段 video 物理一致),不是 per-frame。per-frame reward 容易被 model 通过 single-frame hack 满足(比如每帧都很 realistic 但 motion 不连贯)。trajectory-level reward 强制 model 在整个 rollout 上保持 goal alignment 和 physical consistency,这正是 VideoDifusionNFT 的设计哲学。

### 8.3 为什么 negative-aware finetuning 比 standard RLHF 好
standard RLHF 只 maximize positive reward,容易 mode collapse(总生成 high-reward 的 trivial pattern)和 reward hacking(满足 reward 但不真正完成任务)。negative-aware finetuning 同时:
- 拉向 positive posterior $\pi^+$
- 推离 negative posterior $\pi^-$

这相当于在 reward landscape 中 maintain diversity,避免 collapse。$\alpha(\mathbf{z}_t, c)$ 的 state-dependent guidance 让 high-confidence 区域轻推、low-confidence 区域强推,这是一种 **adaptive exploration-exploitation**。

### 8.4 Geometry supervision 作为 inductive bias
REPA 和 Geometry Forcing 的核心 insight:diffusion model 自己学 3D consistency 很难,但如果把已经学好 geometry 的 teacher(VGGT)作为 representation target,diffusion training 会更 efficient,generation quality 更高。这相当于 **knowledge distillation**,但只 distill representation direction,不 distill final output。

---

## 九、Limitations 与 Potential Issues

paper 没有详细讨论 limitations,但根据 method 可以推测:

1. **VLM 作为 reward evaluator 的成本**:每个 sample 要生成 6 个 video,每个 video 要 VLM 评估 4 个 dimension。Stage 2 training 成本高(虽然 paper 说只用 2000 samples)。
2. **VLM reward 的 noise**:VLM 评分有 stochasticity,可能引入 bias。paper 没有讨论 reward model 的 calibration。
3. **241 frames 的 horizon 限制**:10 秒的 rollout 对于 complex multi-step task(比如 "open fridge → take milk → pour → close fridge")可能不够长。
4. **Exocentric view 的 assumption**:虽然说是 optional,但 Figure 6 显示有 exo view 时 quality 明显更好。在真实 wearable scenario 中,exo view 不一定 available。
5. **Real-world experiment 只有 qualitative**:paper 没有给出 ARGO 上的 quantitative metric,只有 visualization。

---

## 十、Reference Links

- **EgoForge project page**: https://plan-lab.github.io/egoforge
- **Wan2.2 (base generator)**: https://github.com/Wan-Video/Wan2.2 | arXiv: https://arxiv.org/abs/2503.20314
- **VGGT (geometry teacher)**: https://vgg-t.github.io/ | arXiv: https://arxiv.org/abs/2503.11651
- **REPA (Representation Alignment)**: https://sihyun.me/REPA/ | arXiv: https://arxiv.org/abs/2410.06940
- **Geometry Forcing**: arXiv: https://arxiv.org/abs/2507.07982
- **DiffusionNFT (image domain baseline)**: arXiv: https://arxiv.org/abs/2509.16117
- **Ego-Exo4D dataset**: https://ego-exo4d-data.org/ | arXiv: https://arxiv.org/abs/2406.14395
- **Nymeria dataset**: https://www.projectaria.com/nymeria/ | arXiv: https://arxiv.org/abs/2406.00568
- **DigiLens ARGO**: https://www.digilens.com/argo
- **HunyuanVideo**: https://github.com/Tencent/HunyuanVideo | arXiv: https://arxiv.org/abs/2412.03603
- **Cosmos (NVIDIA)**: https://github.com/nvidia-cosmos | arXiv: https://arxiv.org/abs/2501.03575
- **DINOv2 (used for DINO-Score)**: https://github.com/facebookresearch/dinov2
- **Flow Matching for Generative Modeling**: arXiv: https://arxiv.org/abs/2210.02747
- **EgoDreamer**: arXiv: https://arxiv.org/abs/2411.08380
- **Handi**: arXiv: https://arxiv.org/abs/2412.04189
- **EgoWorld**: arXiv: https://arxiv.org/abs/2506.17896
- **PlayerOne (related egocentric world simulator)**: NeurIPS 2025
- **CLIP-Score**: arXiv: https://arxiv.org/abs/2103.00020
- **LPIPS**: arXiv: https://arxiv.org/abs/1801.03924
- **FVD**: arXiv: https://arxiv.org/abs/1812.01758

---

## 十一、对 Karpathy 直觉的连接

你之前在 tweet 和 lecture 中讲过 world model 的几个关键点,EgoForge 在这些点上都有 interesting 的 design choice:

1. **"World model 不只是 next-frame predictor,要 capture causal dynamics"**:EgoForge 用 trajectory-level reward $\mathcal{R}_{\mathrm{temp}}$ 专门 enforce causal logic 和 physics plausibility,而不是 per-frame L2 loss。这直接对应你说的 "causal dynamics" 而非 "appearance prediction"。

2. **"RL finetuning 是给 generative model 加 intent 的方式"**:VideoDifusionNFT 本质上是把 diffusion sampling 当作 policy,用 RL 优化 trajectory reward。这和 DPO / RLHF 在 LLM 上的思路一致,但在 continuous-time diffusion setting 下,且引入 negative-aware 双向优化。

3. **"Geometry / 3D consistency 是 video model 的 fundamental bottleneck"**:EgoForge 用 VGGT teacher 做 REPA-style alignment,这和 LeCun 的 JEPA 思路有 spirit 上的相似——不直接 predict pixel,而是 align representation 到 geometry-aware 的 target。

4. **"Multi-modal conditioning 容易 cue dominance"**:你在 VLM 讨论中提到过这种 shortcut learning。EgoForge 的 multi-dimensional reward + negative pressure 显式 combat 这种 shortcut,强制 model 真正 fuse 多模态信号。

整体上 EgoForge 是一个相当 solid 的工作,把 REPA、DiffusionNFT、VLM-as-judge 三个方向 merge 到 egocentric world simulation 这个具体问题上,工程执行度高,实验结果 convincing。最值得关注的创新是 VideoDifusionNFT 的 negative-aware trajectory-level guidance,这个 idea 在 principle 上可以 generalize 到其他 long-horizon generation 任务(robotics planning, embodied navigation 等)。
