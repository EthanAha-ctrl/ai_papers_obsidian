---
source_pdf: AffordanceVLA A Vision-Language-Action Model Empowering Action Generation
  through Affordance-Aware Understanding.pdf
paper_sha256: 916fbdd94134966c004287c4d287d42e27d3c4b8ac8c08b5ce5acf11d1d90df8
processed_at: '2026-07-18T03:33:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AffordanceVLA 深度解析

Andrej, 这篇 paper 我会按照 build intuition 的方式来讲——重点不是堆砌 method, 而是讲清楚 why each design choice works, 以及背后 representation learning 的深层逻辑。

Paper links:
- Project page: https://skywalker-yqz.github.io/AffordanceVLA/
- Code: https://github.com/Skywalker-yqz/AffordanceVLA/
- arXiv (π0 baseline): https://arxiv.org/abs/2410.24164
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- CALVIN benchmark: https://arxiv.org/abs/2112.03227
- MoT 原论文: https://arxiv.org/abs/2411.04996

---

## 1. 核心问题：Structural Mismatch

VLA 模型的一个根本性 tension: VLM 的预训练把 vision 和 language align 到一个 **semantic space**, 但是 robot action 本质上是 **3D physical space** 中的 representation。这两者之间存在 structural mismatch。

这个问题 Karpathy 你应该很熟悉——就像 LLM 的 token embedding space 和 visual token 的 embedding space 之间的 gap 一样, 只是这里更严重, 因为 action 不是离散的 semantic category, 而是连续的 6-DoF 控制信号。

paper 的核心 claim 非常 Karpathy-style: **"blindly scaling up data fails to maximize the intrinsic power within the datasets"**。换言之, representation 才是 ceiling, data 只是 floor。这点在 Section 4.2.4 的 data efficiency 实验里被 rigorous 证明了——40% data 配上 structured affordance representation 就 break 了 Pi0 用 100% data 的 ceiling。

参考 Karpathy 你之前的 "software 2.0" 思路, 这里其实是在说: dataset 不只是 (x, y) pairs, 而是 (x, structured_intermediate_y, y), 中间的 structured supervision 决定了 gradient 能流到 backbone 多深。

---

## 2. 为什么是 Affordance? Intuition Building

paper 选 affordance 作为 intermediate representation 不是 arbitrary 的。James Gibson 在 1977 年提出的 affordance theory (https://api.semanticscholar.org/CorpusID:60688620) 的核心 insight 是: 物体对 observer 隐含 "action possibilities"。Mug 的 handle "invites" grasping, 椅子 "invites" sitting。

这个 concept 之所以是 ideal bridge, 是因为它同时满足三个 property:

1. **Spatially grounded in vision**: affordance 锚在 image 的具体 region 上
2. **Semantically conditioned in language**: 同一个 mug, "pick" vs "pour" 产生不同 affordance
3. **Action-coupled**: affordance 直接 constrain 后续的 action

paper 里有一句话很关键: "Affordances serve as a perfect bridge, seamlessly coupling spatial grounding in vision, semantic conditioning in language, and execution guidance in action."

这个 framing 让我想到你 karpathy.ai 讲过的 "micrograd" 的 intuition——复杂的 gradient flow 背后是 simple but well-chosen 的 loss。Affordance 就是这个 well-chosen loss。

### 与其他 intermediate representation 的对比

paper 在 Section 2.1 系统对比了其他 paradigm:

- **Video prediction / visual foresight** (SuSIE https://arxiv.org/abs/2310.10639, GR-1 https://arxiv.org/abs/2312.13139, VPP https://arxiv.org/abs/2412.14803, CoT-VLA https://arxiv.org/abs/2503.22020): dense pixel-level target, 信息冗余, inference 慢
- **Textual chain-of-thought** (ThinkAct https://arxiv.org/abs/2507.16815): too coarse, 丢失 spatial precision
- **Keypose / pointflow** (https://arxiv.org/abs/2504.17784): 部分缓解, 但仍然是 sparse
- **Train-only structured supervision** (π0.5 https://arxiv.org/abs/2504.16054, π0.7 https://arxiv.org/abs/2604.15483): 用 discrete action tokens 或 bbox 作为 train-only anchor, 部署时不 decode

AffordanceVLA 的差异化在于: affordance 是 **intrinsic vision-language-action bridge**, 而 π0.5 的 bbox 只是 external anchor。这点在 Section 4.3.4 的 hypothesis 里被展开讨论——affordance training signal 离 VLM semantic space 更近, 因此 "anchors the backbone" 更自然。

---

## 3. Architecture: Mixture-of-Transformer (MoT)

这是 paper 最有意思的 design 之一。基于 Liang et al. 的 MoT (https://arxiv.org/abs/2411.04996), 但做了 task-specific 的改造。

### 3.1 三个 specialized experts

```
[Understanding Expert M_und] → [Affordance Generation Expert M_gen] → [Action Expert M_act]
                                  ↑                                         ↑
                              h_und (instruction-aware)             h_und + Â_t + s_t
```

- **Understanding Expert $\mathcal{M}_{und}$**: 接收 $O_t \in \mathbb{R}^{H \times W \times 3}$ (RGB observation) 和 instruction $l$, 输出 fused semantics $h_t^{und} = \mathcal{M}_{und}(O_t, l)$。注意 proprioceptive state $s_t$ bypass 这个 expert, 直接进入 $\mathcal{M}_{act}$——这是 $\pi_0$-style decoupling, 避免 low-level state 污染 high-level semantics。

- **Affordance Generation Expert $\mathcal{M}_{gen}$**: 这是一个 specialized visual planner, 输出 structured affordance tokens $\hat{A}_t = \mathcal{M}_{gen}(h_t^{und})$。这里关键 design 是把 $\hat{A}_t$ decompose 成三个 parallel sub-modules (Which/Where/How), 而不是 monolithic feature。

- **Action Expert $\mathcal{M}_{act}$**: 输入是 $(h_t^{und}, \hat{A}_t, s_t)$, 输出 action chunk $\hat{a}_{t:t+k}$。用 flow matching (Pi0-style) 做 generative decoding。

### 3.2 UAA Progressive Attention

这是 architecture 最关键的 detail:

- **Intra-expert**: bidirectional attention, 充分 contextual fusion
- **Inter-expert**: **strict causal / unidirectional**

具体地:
- $\mathcal{M}_{gen}$ 的 query 只能 attend 到 $\mathcal{M}_{und}$ 的 K/V: $\text{Attention}(Q_{gen}, K_{und}, V_{und})$
- $\mathcal{M}_{act}$ 能 attend 到 $\mathcal{M}_{und}$ 和 $\mathcal{M}_{gen}$ 两者

这个 design 防止 **action information leakage** 进入 affordance prediction stage, 保持 affordance feature 的 purity。

Karpathy 你想想——这其实就是 **information bottleneck** 的思想。如果 action gradient 能 flow 回 affordance head, 那么 affordance head 可能 "shortcut" 直接 memorize action, 失去作为 generalizable intermediate representation 的作用。Causal flow 强制 affordance 必须从 $h_t^{und}$ 推导, 而不是从 future action 偷看。

### 3.3 为什么不是 MoE?

MoT vs MoE 的区别值得 clarify: MoE 是 input-conditioned routing (每个 token 选 expert), MoT 是 modality/task-conditioned specialization (不同 input stream 进不同 expert)。这里 MoT 更合适, 因为 understanding / affordance / action 本质上是 **不同的 computation**, 不是可互换的 parallel pathways。

---

## 4. Affordance Knowledge Prediction: 三个 module 详解

这是 paper 的 technical core。让我把每个 module 的公式、变量含义、intuition 都拆开讲。

### 4.1 Which2Act: Object-Centric Grounding via Visual Latent Prediction

**目标**: 把 semantic intent (language) align 到 specific visual entity, 同时 suppress background distraction。

**方法**: 不直接预测 bbox (那样太 discrete), 而是预测 target object 的 visual latent。

具体 pipeline:
1. 从 ground truth bbox crop 出 target region
2. 用 frozen pre-trained encoder (Flux VAE, https://arxiv.org/abs/2506.15742) 提取 continuous latent $z_q \in \mathbb{R}^{C \times H \times W}$ (具体是 $16 \times 32 \times 32 = 16384$ floats)
3. Which2Act queries reconstruct $\hat{z}$
4. 用 MSE loss 训练:

$$\mathcal{L}_{\text{which}} = \frac{1}{C \cdot H \cdot W} \sum_{c, h, w} \left\| \hat{z}_{c, h, w} - z_{q, c, h, w} \right\|^2$$

**变量解释**:
- $C, H, W$: latent 的 channel / height / width 维度, 这里 $C=16, H=W=32$
- $\hat{z}_{c,h,w}$: 模型预测的 latent 在 position $(c,h,w)$ 的值
- $z_{q,c,h,w}$: Flux VAE 提取的 ground truth latent
- 求和 $\sum_{c,h,w}$: 遍历所有 latent 位置
- $\|\cdot\|^2$: L2 squared

**为什么用 latent reconstruction 而不是 bbox regression?** 这是 paper 的一个 clever design。Bbox regression 是 "where in image" 的几何问题, 但 latent reconstruction 是 "what is the visual content" 的 semantic 问题。后者 force model 真正理解 target object 的 visual appearance, 而不是只记住坐标。

**Design evolution (Section B.2 很重要)**: 早期版本用 VQ-VAE (discrete codebook), 但 codebook 强制 $\sqrt{N_{w2l}} \in \{1,2,3,...,16\}$, 必须选 $N_{w2l}=64$ tokens。但每个 token 的 projection target 只有 $D_{proj} = 16384/64 = 256$, 远小于 $\mathcal{M}_{gen}$ 的 hidden size $D_{gen}=1024$, 造成 **4× 信息压缩**。

切换到 Flux VAE 后, token count 自由, 选 $N_{w2l}=16$:

$$D_{proj} = \frac{16384}{N_{w2l}} = \frac{16384}{16} = 1024 = D_{gen}$$

这是 **equal-dimension nonlinear transform**, zero information compression/expansion, 最大化 per-token representational utility。

这个 insight 很 Karpathy——representation 的 bottleneck 不在 capacity, 而在 dimension alignment。

### 4.2 Where2Act: 2D Affordance Map via Cross-Attention Decoding

**目标**: 在 target object 上 pinpoint 具体交互 region (例如 mug handle 的哪个位置 grasp)。

**方法**: 把 1D query tokens "unfold" 成 2D spatial distribution。

具体 pipeline:
1. 用 spatial position embeddings 作为 queries
2. Lightweight Transformer decoder 做 cross-attention 提取 interaction cues
3. 输出 spatial logits $\hat{y} \in \mathbb{R}^{H_t \times W_t}$
4. 用 BCE loss 对齐 ground truth mask $M \in [0,1]^{H_t \times W_t}$:

$$\mathcal{L}_{\text{where}} = -\frac{1}{H_t W_t} \sum_{i=1}^{H_t W_t} \left[ M_i \log \sigma(\hat{y}_i) + (1-M_i) \log(1 - \sigma(\hat{y}_i)) \right]$$

**变量解释**:
- $H_t, W_t$: affordance map 的空间分辨率
- $i$: pixel index, 遍历 $H_t \times W_t$ 个 pixel
- $M_i \in [0,1]$: ground truth affordance mask 在 pixel $i$ 的值 (Gaussian heatmap, bounded by SAM segmentation)
- $\hat{y}_i$: model 预测的 logit
- $\sigma(\cdot)$: sigmoid 函数 $\sigma(x) = 1/(1+e^{-x})$
- $M_i \log \sigma(\hat{y}_i)$: positive class (interactive region) 的 cross-entropy
- $(1-M_i)\log(1-\sigma(\hat{y}_i))$: negative class (non-interactive) 的 cross-entropy

**Intuition**: 这个 "unfold" 操作很重要。1D token 是 compressed representation, 但 interaction region 本质是 2D spatial distribution。直接用 MLP regress 坐标会 lose spatial structure; 用 cross-attention with positional encoding queries 保留 spatial inductive bias。

**Ground truth 怎么来?** Section C.2 详细描述: 用 SAM (https://arxiv.org/abs/2304.02631) 在 bbox 内 segmentation, 然后在 affordance point 处生成 Gaussian heatmap, bounded by mask boundary。这样 ground truth 既是 spatially precise (Gaussian peak) 又是 semantically grounded (within object mask)。

### 4.3 How2Act: 3D Geometric Reasoning via Diffusion + Layout Regression

这是最 ambitious 的 module。Bifurcated design:

#### 4.3.1 Shape Generation Branch (Diffusion)

**目标**: 重建 target 的 3D voxel latent。

**方法**: Conditional diffusion process, iterative Transformer denoiser $\hat{\epsilon}_\theta$:

$$\mathcal{L}_{\text{shape}} = \mathbb{E}_{t \sim \mathcal{U}(0,T), \epsilon \sim \mathcal{N}(0, \mathbf{I})} \left[ \left\| \epsilon - \hat{\epsilon}_\theta(x_t, t, \bar{h}_{shape}) \right\|^2 \right]$$

**变量解释**:
- $t \sim \mathcal{U}(0,T)$: diffusion timestep, 从 uniform distribution 采样, $T$ 是 total timesteps
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: standard Gaussian noise, $\mathbf{I}$ 是 identity matrix
- $x_t$: noisy latent at timestep $t$ (forward process: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$)
- $\bar{h}_{shape}$: condition (来自 shape tokens 的 mean-pooled representation)
- $\hat{\epsilon}_\theta(x_t, t, \bar{h}_{shape})$: neural network 预测的 noise
- $\|\cdot\|^2$: L2 squared, 这是 standard DDPM noise prediction objective

#### 4.3.2 Layout Regression Branch (MLP)

**目标**: 回归 10-DoF spatial layout vector。

**方法**: MLP regressor, Smooth-L1 loss:

$$\mathcal{L}_{\text{layout}} = \frac{1}{10} \sum_{j=1}^{10} \text{SmoothL1}(\hat{y}_{layout}^{(j)}, y_{layout}^{(j)})$$

**变量解释**:
- $j \in \{1,...,10\}$: layout 维度 index
- $\hat{y}_{layout}^{(j)}$: 预测的第 $j$ 维
- $y_{layout}^{(j)}$: ground truth 的第 $j$ 维
- 10-DoF decomposition: rotation (4D, quaternion) + scale (3D) + translation (3D)
- SmoothL1: $\text{SmoothL1}(x) = 0.5x^2$ if $|x|<1$, else $|x|-0.5$。比 L2 对 outlier 更 robust, 比 L1 在 0 附近更 smooth

**为什么 How2Act 最 ambitious 但 ablation 影响最小?** Section 4.2.2 揭示: How2Act 在 LIBERO/CALVIN 这种 tabletop 2-finger setting 下 benefit modest, 但在 real-world 6-DoF interaction 上 pronounced。这暗示 3D reasoning 的 value 是 task-dependent 的——简单 task 可以从 2D affordance shortcut, 复杂 task 必须 understand 3D geometry。

#### 4.3.3 三个 module 的 synergy

Section 4.2.3 的 **Block-wise Tokens control** 是最 elegant 的 ablation。把 bidirectional attention 替换成 causal, 三个 head 变成 independent auxiliary tasks, supervision density 完全相同。结果: LIBERO 从 95.8% 跌到 90.3%, CALVIN 从 4.33 跌到 3.89。

**这个 ablation 直接 rule out "multi-task density" explanation**——gain 不来自更多 loss terms, 而来自 structured, jointly-refined representation。三个 head 之间有 implicit dependency: spatial localization 依赖 object grounding, 3D structure 依赖 2D interaction points。Shared attention 让它们 "promote each other"。

---

## 5. Training Strategy: 三阶段 Progressive Curriculum

### 5.1 Stage I: General Affordance Grounding Pre-training

**数据**: VQA datasets—AGD20K (https://arxiv.org/abs/2208.13196), RefSpatial (https://arxiv.org/abs/2506.04308), PRISM (https://arxiv.org/abs/2505.13441)

**Freezing**: Vision Encoder, Understanding Expert, Action Expert 都 frozen, 只训 Affordance Generation Expert + learnable queries + decoders

**Loss**:

$$\mathcal{L}_{\text{Stage1}} = \lambda_{\text{which}} \mathcal{L}_{\text{which}} + \lambda_{\text{where}} \mathcal{L}_{\text{where}} + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}} + \lambda_{\text{layout}} \mathcal{L}_{\text{layout}}$$

**Weight**: $\lambda_{\text{which}} = \lambda_{\text{where}} = \lambda_{\text{shape}} = 0.1, \lambda_{\text{layout}} = 0.04$

**Intuition**: 先让 affordance expert 学会 generic spatial/geometric reasoning, 不让它被 action loss 干扰。Frozen backbone 保证 pre-trained semantic priors 不被 wash out。

Layout weight 较小 (0.04 vs 0.1) 因为 10-DoF vector 信息密度低于 dense latent/heatmap, 过大 weight 会让 model over-optimize 简单 regression 而忽略 difficult shape generation。

### 5.2 Stage II: Affordance-Augmented Robotic Data Co-Training

**数据**: InternData-A1 (https://arxiv.org/abs/2511.16651), large-scale synthetic robotic data, 149K trajectories

**Unfreezing**: Understanding + Action Expert unfrozen, Vision Encoder 用 lower LR fine-tune

**Loss**:

$$\mathcal{L}_{\text{Stage2}} = \lambda_{\text{act}} \mathcal{L}_{\text{act}} + \lambda_{\text{afd}} \mathcal{L}_{\text{afd}}$$

其中 $\mathcal{L}_{\text{afd}}$ aggregate Stage I 的四个 affordance loss。$\lambda_{\text{act}}=1.0, \lambda_{\text{afd}}=0.5$。

**Intuition**: 这一阶段是 **bridge stage**——把 broad VLM 知识和 embodied control 在大规模数据上 align。Affordance weight (0.5) 显著高于 Stage III (0.15), 因为这一阶段 affordance supervision 是 primary signal, action 是 secondary。

### 5.3 Stage III: Target Task Post-Training

**数据**: LIBERO / CALVIN / DROID subset

**Loss**: 同 Stage II, 但 $\lambda_{\text{afd}}=0.15$

**Intuition**: 最后一阶段让 action loss 主导, affordance 退到 auxiliary anchor 角色。这个 annealing 很关键——前期 affordance 主导 build representation, 后期 action 主导 refine control。

### 5.4 设计哲学

paper Section 3.3.4 的 design philosophy 值得深读:

> "Rich annotations should encode not just what to do but how to do it; structured affordance supervision preserves the backbone's vision-language ability instead of eroding it under the action loss."

这个观点直接呼应 π0.5/π0.7 的 train-only intermediate supervision 思路 (https://arxiv.org/abs/2504.16054)。Karpathy 你应该会 appreciate 这个: 如果只有 action loss, gradient 直接 backprop 进 VLM, 但 action signal 既 not informative 又 not aligned with VLM semantics, 会 gradual erode instruction-following ability。Affordance loss 是 semantic-aligned supervision, anchor 住 backbone。

---

## 6. Data Augmentation Pipeline

这是 paper 的 hidden gem, 在 Section C.2 详细描述。Robotic datasets (LIBERO/CALVIN/DROID/A1) 都没有原生 affordance annotation, 必须自动生成。

### 6.1 Pipeline 概览

```
[Step 0] RexOmni fine-tune on PRISM
   ↓
[Step 1] Rule-based keyframe detection (6 rules)
   ↓
[Step 2a] Claude Opus 4.5: decompose long instruction → per-keyframe sub-instruction
   ↓
[Step 2b] Qwen3-VL: per-keyframe → detection category + affordance instruction
   ↓
[Step 3] RexOmni dual-mode: bbox + affordance point → SAM mask → heatmap
                                                       → SAM-3D → shape + layout tokens
   ↓
[Step 4] Quality verification (point-in-bbox + 100-round human audit)
```

### 6.2 Keyframe Detection (Step 1)

六种 keyframe type, 用 joint state 和 gripper state signal:

1. **Start**: trajectory 第一帧
2. **Pre-Action**: gripper state change 前 N=30 帧——这是最 informative 的 frame, 因为 robot 已经 oriented 但还未 contact
3. **Gripper**: gripper open/close flip 的 exact frame
4. **Stop**: 所有 joint velocity $\|\dot{q}_j\| < \epsilon=0.01$ 且 gripper state stable——sub-task transition
5. **Apex**: joint velocity norm local minimum, $\epsilon < \|\dot{q}\| < 5\epsilon$——motion direction reversal
6. **End**: trajectory end 前 M=25 帧

**Variable**: $\dot{q}_j$ 是第 $j$ 个 joint 的 velocity, $\|\dot{q}_j\|$ 是其 scalar norm。

### 6.3 Instruction Decomposition (Step 2a)

Long-horizon instruction (e.g., "Pick up the red cup and put it on the shelf") 跨多个 primitive interaction。用 Claude Opus 4.5 做 temporal decomposition, 每个 keyframe 得到一个 atomic sub-instruction。

关键 constraint: sub-instruction 必须 respect keyframe 的 semantic role (Pre-Action/Gripper/Stop 等)。

### 6.4 Per-Keyframe Annotation (Step 2b)

用 Qwen3-VL-235B (https://arxiv.org/abs/2505.09388) 对每个 keyframe image 输出:
1. **Detection category**: 最 specific 的 target part (e.g., "drawer handle" 而非 "drawer")
2. **Affordance instruction**: "Where to [verb] the [target] to [goal]?" 格式

这两个 expression 分别 drive Which2Act (what object) 和 Where2Act (where to interact) 的 supervision。

### 6.5 Quality Verification (Step 4)

两级 verification:
1. **Point-in-Bbox**: 100% 的 affordance points 必须落在对应 bbox 内
2. **Human audit**: 100 轮 random sampling, 每轮 30 samples, 必须 100% pass rate 才能进训练

这套 pipeline 产出超过 100K affordance annotations。Quality 是 paper 反复强调的——Section C.1 的 claim 是 "data quality as performance ceiling"。

---

## 7. 实验结果分析

### 7.1 LIBERO (Table 1)

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Pi0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| gr00t-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| F1-VLA | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| AffordanceVLA (w/o Stage II) | 88.5 | 91.7 | 91.3 | 73.3 | 86.2 |
| **AffordanceVLA (full)** | **98.6** | **98.4** | **96.2** | 89.8 | **95.8** |

**Key observations**:
- AffordanceVLA (full) 在 Spatial/Object/Goal 三个 suite 上 SOTA, Long suite 略低于 F1-VLA
- w/o Stage II 仍有 86.2% avg, 证明 MoT architecture 本身的 advantage
- Long suite 是 bottleneck (89.8%), 暗示 long-horizon task 需要 explicit memory mechanism

### 7.2 CALVIN ABC→D (Table 2)

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. Len |
|--------|-----|-----|-----|-----|-----|----------|
| Seer-Large | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP | 95.3 | 88.2 | 80.3 | 72.9 | 64.5 | 4.01 |
| Pi0 | 93.8 | 85.0 | 76.7 | 68.6 | 60.1 | 3.84 |
| AffordanceVLA (w/o Stage II) | 93.4 | 84.7 | 75.4 | 68.1 | 58.9 | 3.81 |
| **AffordanceVLA (full)** | **96.8** | **92.0** | **87.5** | **80.8** | **75.9** | **4.33** |

CALVIN ABC→D 是 zero-shot OOD protocol (train on ABC, test on D)。AffordanceVLA avg len 4.33 是 SOTA, 比 Seer-Large 高 0.05。

**Insight**: w/o Stage II 跌到 3.81, 证明 Stage II 的 large-scale affordance-augmented co-training 是 OOD generalization 的关键。Affordance supervision 让 model 学到 task-critical structure, 而非 superficial scene correlation。

### 7.3 Real-world (Table 5)

**Basic tasks**:
- Pi0: 70.8% avg
- AffordanceVLA: 88.3% avg (+17.5%)

**Complex tasks** (visual aliasing, identical observation, different instruction):
- Pi0: 44.8% avg
- AffordanceVLA: 82.9% avg (+38.1%)

**Drawer (pick vs close)**:
- Pi0: 46.7% / 40.0%
- AffordanceVLA: 86.7% / 100.0%

这个 gap 是 paper 最 striking 的结果。Identical visual input, 只差 language instruction, Pi0 表现差意味着 **instruction-following ability 被 eroded**——这是 Section 4.3.4 hypothesis 的直接证据。

**Pick all the rubbish (long-horizon)**:
- Pi0: 3rd execution 仅 6.7%, Empty Picks 33 次
- AffordanceVLA: 3rd execution 46.7%, Empty Picks 11 次

Affordance subgoal 提供 sequential visual target, compensate lack of explicit long-term planning。

---

## 8. Ablation 关键 Insights

### 8.1 Architecture vs Data (Table 3)

| Variant | LIBERO Avg | CALVIN Avg Len |
|---------|------------|----------------|
| No-Afd (Pi0 Arch, same data) | 92.4 | 3.93 |
| Frozen-Afd (Stage I frozen) | 67.1 | 2.83 |
| AffordanceVLA w/o Stage II | 86.2 | 3.81 |
| **AffordanceVLA (full)** | **95.8** | **4.33** |

三个 control 各 isolate 一个 factor:
- **No-Afd**: 同样 data, 去掉 affordance objective 和 expert → 只比 Pi0 高 marginal。证明 data alone 不够。
- **Frozen-Afd**: affordance expert 训完冻住, 当 external prior → catastrophic collapse (67.1%)。证明 affordance 必须 **co-optimize** with policy, 否则 representation mismatch 重现。
- **w/o Stage II**: 去掉 co-training → OOD 显著下降。

Frozen-Afd 这个 ablation 特别 informative——它 reproduce 了 prior work 的 external affordance pipeline 失败模式, 证明 internalization 是关键。

### 8.2 Affordance Representation (Table 3)

| Variant | LIBERO Avg | CALVIN Avg Len |
|---------|------------|----------------|
| w/o Which2Act | 94.6 | 4.20 |
| w/o Where2Act | 93.2 | 4.13 |
| w/o How2Act | 93.7 | 4.01 |
| **Full** | **95.8** | **4.33** |

去掉任一 module 都是 **graceful degradation**, 不是 catastrophic collapse。这证明三个 module 不是 chained pipeline (上游 error 会 propagate), 而是 jointly refined under shared representation。

### 8.3 Same-Density Control (Section 4.2.3)

Block-wise Tokens: 同样 loss, 同样 data, 只把 bidirectional attention 换成 causal → LIBERO 95.8→90.3, CALVIN 4.33→3.89。

**这是整个 paper 最 elegant 的 ablation**。Supervision density 完全相同, 唯一差别是 attention pattern。结果证明: gain 来自 **structured jointly-refined representation**, 不是 "更多 loss term"。

### 8.4 Data Efficiency (Figure 3)

**关键数据点**: 40% fine-tuning data 时, AffordanceVLA 已经达到 ~92% LIBERO, ~4.0 CALVIN avg len, **break 了 Pi0 100% data 的 ceiling**。

**Recovery trajectory 分析**:
- 10% data: Pi0 起点高 (pre-trained weights 直接 work), AffordanceVLA 起点低 (specialized pre-training 引入 temporary weight misalignment)
- 40% data: AffordanceVLA surge, break Pi0 ceiling
- 100% data: AffordanceVLA 持续领先

**No-Afd**: 恢复慢, 难超 Pi0 → robotic pre-training data alone 不够, 需要 affordance structure
**w/o Stage II**: shift 最严重, 恢复最慢 → Stage II 是 broad VLM 和 sample-efficient adaptation 的 indispensable bridge

---

## 9. Failure Mode & Hypothesis (Section 4.3.4)

paper 最 intellectually honest 的部分是 Section 4.3.4 的 hypothesis。Toaster task 失败模式:
- Pi0 在 button-pressing step 失败, 经常 close gripper 而非 extend press
- 即使 fine-tuned on real trajectories, Pi0 仍 driven by dominant action prior, 忽略 "press the button" instruction

**Hypothesis**: action-only VLA 中, low-level action loss 直接 backprop 进 VLM backbone, 这个 signal 既 insufficiently informative 又 misaligned with V-L semantics, gradually erode instruction-following ability。

π0.5/π0.7 用 train-only structured supervision (discrete action tokens / bbox) 作为 semantic anchor。AffordanceVLA conjecture: affordance 起 similar 但更 natural 的作用——它的 training signal 离 V-L semantic space 更近。

**重要 caveat**: paper 明确说这是 "interpretive hypothesis meant to guide intuition, not a claim backed by direct mechanistic evidence"。这种诚实是 rare 且值得 respect 的。

Karpathy 你应该会 appreciate 这个——很多 VLA paper 把所有 improvement 都归功于自己 method, 这篇 paper 明确区分 "what we proved" 和 "what we hypothesize"。

---

## 10. 与 Recent Strong VLA 的对比

### 10.1 vs π0 / π0.5 / π0.7

- **π0** (https://arxiv.org/abs/2410.24164): VLM + flow matching action expert, no intermediate representation
- **π0.5** (https://arxiv.org/abs/2504.16054): 引入 train-only structured supervision (discrete action tokens, bbox)
- **π0.7** (https://arxiv.org/abs/2604.15483): steerable generalist, emergent capabilities

AffordanceVLA 的差异化: affordance 是 **intrinsic V-L-A bridge**, 而 π0.5 的 intermediate supervision 是 external anchor。Affordance 同时是 spatially grounded + semantically conditioned + action-coupled, 而 bbox 只 spatially grounded。

### 10.2 vs Video Prediction VLAs (CoT-VLA, VPP, Seer, F1-VLA)

- **CoT-VLA** (https://arxiv.org/abs/2503.22020): visual chain-of-thought via future frame prediction
- **VPP** (https://arxiv.org/abs/2412.14803): video prediction policy
- **Seer** (https://arxiv.org/abs/2412.15109): predictive inverse dynamics
- **F1-VLA** (https://arxiv.org/abs/2509.06951): bridging understanding and generation

这些都是 dense pixel-level prediction, 信息冗余, inference 慢。AffordanceVLA 用 compact structured representation, 更 efficient。

### 10.3 vs SpatialVLA, CogACT, Spatial Forcing

- **SpatialVLA** (https://arxiv.org/abs/2501.15830): 空间 representation enhanced backbone
- **CogACT** (https://arxiv.org/abs/2411.19650): cognition + action synergizing
- **Spatial Forcing** (https://arxiv.org/abs/2510.12276): implicit spatial representation alignment

这些主要做 backbone-level spatial enhancement, AffordanceVLA 做 explicit structured intermediate supervision。

### 10.4 vs Prior Affordance Work (Robo-ABC, RAM, AffordDP, CoA-VLA)

- **Robo-ABC** (https://arxiv.org/abs/2401.07487): training-free affordance transfer via semantic correspondence
- **RAM** (https://arxiv.org/abs/2407.04689): retrieval-based affordance transfer
- **AffordDP** (https://arxiv.org/abs/2412.03142): transferable affordance for diffusion policy
- **CoA-VLA** (https://arxiv.org/abs/2412.20451): visual-textual chain-of-affordance

这些把 affordance 当 **external cue**, 配合 external grasp generator / motion planner, open-loop brittle。AffordanceVLA **internalize** affordance into VLA, jointly optimize with VLM backbone 和 action expert, closed-loop robust。

---

## 11. Inference Latency (Table 9)

| Component | Latency (ms) |
|-----------|--------------|
| Image preprocessing | ~6 |
| Image encoder (SigLIP) | ~22 |
| Understanding + Affordance Generation | ~52 |
| ×10 Action denoising (Euler flow matching) | ~92 |
| Misc | ~4 |
| **Total** | **~176** |

5.7 Hz 实时控制。Action denoising 占 52%——这是 flow matching 的 inherent cost。如果想 further 加速, 可以用 consistency model 或 fewer denoising steps, 但 risk action quality。

---

## 12. Limitations & Future Directions

paper 自己承认的 limitation:
1. **Long-horizon**: LIBERO-Long 89.8% (vs Spatial 98.6%)——explicit long-term memory 需要补充
2. **3D reasoning fidelity**: How2Act shape token 的 absolute accuracy 较低 (因为 voxel reconstruction from compressed token 困难)
3. **Bimanual / deformable**: 未涉及, 是 future work

我 (作为 reader) 看到的额外 limitation:
1. **Affordance annotation pipeline 依赖 closed-source LLMs** (Claude Opus 4.5, Qwen3-VL-235B)——reproducibility 风险
2. **RexOmni fine-tune on PRISM**: 引入 dataset bias, 可能 limit cross-domain generalization
3. **Real-world evaluation 规模有限**: 8 basic tasks + 4 complex tasks, 每 task 15 trials。需要更大 scale validation
4. **No comparison with π0.5 / π0.7**: 这两个是 direct competitor, 缺乏 head-to-head 比较

---

## 13. Karpathy-style 深度思考

### 13.1 Representation as Bottleneck

paper 反复强调 "VLA is fundamentally representation learning"。这个 framing 很 powerful, 但也值得 critical examine:

- **Pro**: 解释了为什么 data scaling alone 不够, 为什么 structured supervision 能 break ceiling
- **Con**: 可能 over-simplify——robot learning 还有 exploration, reward design, embodiment transfer 等问题, 这些不是 pure representation 能解决的

不过 paper 的 scope 是 imitation learning (offline trajectories), 所以 representation framing 在这个 scope 内 valid。

### 13.2 Information Bottleneck 的角色

UAA progressive attention 是 implicit information bottleneck。Action gradient 不能 flow 回 affordance head, 强制 affordance 必须从 $h^{und}$ 推导。

这个 design 让我想到 Karpathy 你在 nanoGPT 里讲过的 causal mask——同样是 restrict information flow 来 enforce 某种 structure。这里 causal flow 不是 temporal, 而是 **semantic hierarchy** (understanding → affordance → action)。

### 13.3 "Anchoring" Hypothesis 的 mechanistic 解释

Section 4.3.4 的 hypothesis 可以更 mechanistic 地思考:

VLM backbone 的 pre-trained representation 是 high-rank 的, encode 丰富 semantic structure。Action loss 的 gradient 是 low-rank 的 (只有 7-DoF signal), 直接 backprop 会 collapse representation 到 action-relevant subspace, erode 其他 semantic dimension。

Affordance loss 提供 higher-rank supervision (latent reconstruction 是 16384-dim, heatmap 是 $H_t \times W_t$-dim, shape latent 是 4096-dim), 这些 supervision anchor 住 representation 的更多 dimension, 防止 collapse。

这个解释可以 tested——可以做 representation similarity analysis (CKA, centered kernel alignment) 看 backbone feature 在 action-only vs action+affordance training 下的 rank 变化。

### 13.4 Curriculum 的 implicit curriculum

三阶段不只是 data curriculum, 也是 **task difficulty curriculum**:
- Stage I: static scene, single object, VQA-style (easy)
- Stage II: dynamic scene, multi-step, synthetic (medium)
- Stage III: target domain, real distribution (hard)

这个 difficulty escalation 让 model 先学 simple affordance grounding, 再学 action-coupled affordance, 最后 adapt to target domain。符合 curriculum learning 的经典 wisdom。

### 13.5 Affordance 的 "Category" vs "Instance" Tension

How2Act shape branch 用 mean-pooling over shape tokens, paper 承认这 limit instance-level 3D fidelity, 但 "suffices for encoding categorical shape priors"。

这是 affordance learning 的经典 tension: affordance 是 **category-level** (mug handle 都可 grasp) 还是 **instance-level** (这个 specific mug 的这个 specific handle)? Gibson 的 original theory 是 category-level, 但 robotic execution 需要 instance-level precision。

AffordanceVLA 的 design 妥协了——Which2Act + Where2Act 提供 instance-level grounding (specific bbox + specific heatmap), How2Act 提供 category-level shape prior。这个 decomposition 是 pragmatic 的, 但也暗示 future work 可能需要 instance-level 3D reasoning。

### 13.6 Train-only vs Inference-time Affordance

paper 在 main results 里 inference 时确实 decode affordance (用于 condition action), 但 Section 2.1 提到 π0.5/π0.7 用 train-only intermediate supervision (不 decode at inference)。

AffordanceVLA 选择 decode 的 trade-off:
- **Pro**: affordance 提供 explicit planning signal, 可能 improve interpretability 和 debug-ability
- **Con**: 增加 inference latency (Understanding+Affordance Generation 52ms)

Future work 可以 explore: 训练时用 affordance supervision, 推理时 distill 到 action-only model, 能否保留 representation benefit 而 eliminate latency cost?

### 13.7 与你的 "Micrograd" / "Software 2.0" 思路的连接

Karpathy 你应该会 appreciate 这个 paper 的 underlying philosophy: dataset 不是 (x, y) pairs, 而是 (x, structured_intermediate_y, y)。中间的 structured supervision 决定 gradient flow 的 quality, 进而决定 representation 的 quality。

这正是 Software 2.0 的核心——我们 design 的不是 algorithm, 而是 **loss function 和 data representation**。AffordanceVLA 的 contribution 是找到了一个 domain-specific (robotics) 的 well-chosen loss: structured affordance forecasting。

### 13.8 为什么 Affordance 是 "Natural" Bridge?

paper claim affordance 是 "natural" V-L-A bridge, 但没充分 explain 为什么。我 (作为 reader) 的 hypothesis:

Affordance 在 cognitive science 里就是 **perception-action coupling** 的 representation——看到 mug handle 就 "feel" grasping possibility。这种 coupling 是 embodied cognition 的核心。

VLM 的 pre-training align vision 和 language 在 semantic space, 但 semantic space 是 **disembodied** 的 (mug 的 concept 不包含 "how to grasp")。Affordance supervision 把 disembodied semantic 升级为 embodied semantic, 自然 bridge 到 action space。

这个 hypothesis 可以 tested: 用 fMRI 数据验证 human affordance processing 区域和 VLM feature space 的 alignment, 看 affordance-supervised VLA 是否更接近 human neural representation。

### 13.9 Token Allocation 的 dimension analysis

Section B.3 的 token allocation (AffordanceVLA-full: 64 Which + 64 Where + 256 Shape + 4 Layout = 388 total) 是精心 design 的:

- **Which2Act (64 tokens, 1024-dim each)**: 每 token 1024-dim = $D_{gen}$, equal-dimension transform
- **Where2Act (64 tokens)**: decoded to 2D heatmap via cross-attention, spatial resolution sufficient
- **How2Act Shape (256 tokens)**: mean-pooled to condition 3D diffusion, 256 提供 richer upstream representation
- **How2Act Layout (4 tokens)**: regress 10-DoF, 4 tokens × mean-pool 足够

这个 allocation 反映了 task difficulty——shape generation 最难, 分配最多 tokens; layout regression 最简单, 4 tokens 够用。是信息论意义上的 optimal allocation。

### 13.10 Inverse Dynamics 视角

paper 的 affordance prediction 可以 reframe 为 **inverse dynamics**: 给定当前 observation 和 future goal (affordance), predict action。这和 Seer (https://arxiv.org/abs/2412.15109) 的 predictive inverse dynamics 有 conceptual connection。

差别: Seer 预测 next-frame visual latent, AffordanceVLA 预测 structured affordance latent。后者更 compact, 更 task-relevant。

---

## 14. 总结

AffordanceVLA 的核心 contribution 是 **conceptual + technical**:

**Conceptual**: 把 affordance 作为 V-L-A 的 natural bridge, 用 structured forecasting (Which/Where/How) 作为 intermediate representation, 解决 VLM semantic space 和 embodied control 的 structural mismatch。

**Technical**:
1. MoT architecture with UAA progressive attention (prevent action leakage)
2. Three-stage training curriculum (progressive bridge from VQA to target task)
3. Automated affordance annotation pipeline (Claude + Qwen3-VL + RexOmni + SAM + SAM-3D)
4. Flux VAE for Which2Act (dimension-matched continuous latent)

**Empirical**:
- LIBERO 95.8% (SOTA on 3/4 suites)
- CALVIN 4.33 avg len (SOTA)
- Real-world +38.1% over Pi0 on complex tasks
- 40% data break Pi0's 100% data ceiling

**Open questions** (paper 自己 raise 或 我作为 reader raise):
1. Long-horizon 仍需 explicit memory
2. 3D reasoning fidelity 受 token bottleneck 限制
3. Hypothesis (affordance as semantic anchor) 缺 mechanistic evidence
4. 与 π0.5/π0.7 缺 head-to-head 比较
5. Affordance annotation pipeline 依赖 closed-source LLMs

---

## 15. 进一步阅读推荐

如果你想 deep dive, 我推荐这些 related work:

- **π0**: https://arxiv.org/abs/2410.24164 (VLA baseline, flow matching action expert)
- **π0.5**: https://arxiv.org/abs/2504.16054 (train-only structured supervision 的 parallel work)
- **MoT**: https://arxiv.org/abs/2411.04996 (architecture foundation)
- **OpenVLA**: https://arxiv.org/abs/2406.09246 (open-source VLA baseline)
- **SpatialVLA**: https://arxiv.org/abs/2501.15830 (spatial representation enhanced)
- **CoT-VLA**: https://arxiv.org/abs/2503.22020 (visual chain-of-thought)
- **F1-VLA**: https://arxiv.org/abs/2509.06951 (understanding-generation-action bridge)
- **Seer**: https://arxiv.org/abs/2412.15109 (predictive inverse dynamics)
- **InternData-A1**: https://arxiv.org/abs/2511.16651 (Stage II training data)
- **SAM-3D**: https://arxiv.org/abs/2511.16624 (3D shape annotation)
- **Flux**: https://arxiv.org/abs/2506.15742 (Which2Act VAE)
- **Qwen3**: https://arxiv.org/abs/2505.09388 (annotation pipeline VLM)
- **RexOmni**: https://arxiv.org/abs/2510.12798 (visual grounding model)
- **PRISM/GraspMolmo**: https://arxiv.org/abs/2505.13441 (affordance VQA data)
- **Gibson affordance**: https://api.semanticscholar.org/CorpusID:60688620 (theoretical foundation)
- **LIBERO**: https://arxiv.org/abs/2306.03310 (benchmark)
- **CALVIN**: https://arxiv.org/abs/2112.03227 (benchmark)
- **DROID**: https://arxiv.org/abs/2403.12945 (real-world dataset)
- **CoA-VLA**: https://arxiv.org/abs/2412.20451 (prior affordance VLA)
- **AffordDP**: https://arxiv.org/abs/2412.03142 (affordance diffusion policy)
- **RAM**: https://arxiv.org/abs/2407.04689 (retrieval-based affordance transfer)

---

这篇 paper 在 representation learning for robotics 方向树立了一个不错的 paradigm。Karpathy 你的 intuition 应该会 resonate with 它的 core message: **representation is the ceiling, data is the floor, structured supervision is the ladder**. 

如果你对某个具体 module 或 experiment 想更深入 discuss, 我可以进一步展开。特别 interested 的话, 可以聊聊:

1. UAA attention 的 information bottleneck 视角 vs standard causal masking 的区别
2. Flux VAE vs VQ-VAE 的 dimension analysis 是否能 generalize 到其他 modality
3. Affordance "anchoring" hypothesis 的 mechanistic testing 设计
4. Three-stage curriculum 和你的 nanoGPT training curriculum 的 conceptual connection
5. Token allocation 的信息论 optimal 分析

希望这个 review 对你有帮助!
