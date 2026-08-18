---
source_pdf: Beyond 3D VQAs Injecting 3D Spatial Priors into.pdf
paper_sha256: c7cb98ba9d05e1ebc41cc6b8a44818cac9062e16ac044c625af0f4a4f33ab92f
processed_at: '2026-08-18T02:42:47-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲GASP

Andrej，我换个画风，就当咱俩喝着咖啡聊这篇paper。

---

## 一句话说清楚

这篇paper就在讲一件事：**你想让VLM懂3D空间，别让它背VQA答案，要让它学会"同一个东西换个角度还是同一个东西"这件事**。

就这么简单。剩下所有的math、loss、ablation全是在服务这一句话。

---

## 为什么这件事值得做

现在主流的两条路都有坑：

**第一条路**：拿3D VQA数据集做SFT。问题是，model会learn到捷径。比如VSI-Bench里大部分房间都是20平米，model直接背这个数字就能拿高分。Appendix里那个实验特别狠 — 把平均room size作为text prior塞进prompt里，Object Abs Distance从0.14飙到0.61，比专门fine-tune的VLM-3R (0.49)还高。这就说明benchmark分数在某种意义上是"被hack"的，model根本没在reason，在猜。

**第二条路**：外面挂一个VGGT这种pre-trained 3D encoder。问题是又重又僵。你frozen权重，LLM得align自己的feature到这个外挂的rigid feature space里，alignment challenge特别大。

所以作者就跳出来说：**要不去教model一个fundamental的东西 — correspondence**。

---

## Correspondence是什么，为什么选它

Correspondence就是：frame t里这个像素点，在frame t+1里对应哪个像素点。

这件事为什么fundamental？因为如果你能跨视角track同一个点，你就懂了object constancy — 同一个3D entity从不同角度看依然是它自己。所有3D vision的task，从SLAM到multi-view stereo到optical flow，底层都依赖这个能力。

换句话说，correspondence是3D spatial reasoning的"原子"。你把这个原子学会了，上面的spatial reasoning自然就emerge出来。

---

## 怎么inject到LLM里

这里就是作者的小trick。在LLM的每一层后面挂一个小MLP head，叫correspondence head $\mathcal{H}_c$。这个head做的事就是把visual token的feature project到一个低维embedding space，在这个space里，同一个3D点跨frame的embedding应该接近，不同3D点的embedding应该远。

这个head用两个loss监督：

**Loss 1: Contrastive loss (L_corr)**

就是InfoNCE。anchor point在frame a的embedding，要和它在frame b的embedding接近，要和frame b里其他所有点的embedding远。非常标准的contrastive learning。

**Loss 2: Depth consistency loss (L_depth)**

这个是作者的创意点。光用contrastive有个问题：如果foreground和background各有一个texture相同的物体，contrastive loss会错配它们，因为visual embedding很像。

作者用depth来regularize这个事。具体做法很elegant — 复用contrastive的similarity scores做一个soft matching distribution，然后用soft-argmax算expected depth，再和ground-truth depth比。如果match错了（匹配到depth不一致的点），loss就大。

key insight: **depth不作为prediction target，作为discriminative regularizer**。作者明确说不训练depth prediction head，因为depth prediction是ill-posed且有scale ambiguity。用depth作为"哪些match是geometrically valid"的supervisor就好。

---

## 为什么挂在每一层

作者做了ablation（Table 4），挂在all layers最好。这个发现我特别appreciate：

- **Early layers**: 学low-level correspondence (edges, corners)
- **Middle layers**: object parts, boundaries
- **Deep layers**: semantic-geometric alignment

如果只在deep layer监督，shallow layer还在学view-dependent feature，整个representation就有bottleneck。Geometric consistency是hierarchical的，需要每一层都contribute。

这跟Zeiler & Fergus那种hierarchical feature learning的classic发现是parallel的 — 你不能用最后一层response去supervise整个model，得让每一层都学到它该学的东西。

---

## 结果怎么validate的

这里paper做得特别漂亮的地方 — 不只看downstream VQA accuracy，先probe internal representation。

三个metric：

1. **Layer-wise PCK**: 直接从attention的QK^T matrix里extract correspondence，看matching准不准。Baseline (Qwen2.5-VL, LLaVA-NeXT-Video) 峰值 < 5%。GASP训练后 > 70%。这是50倍提升。

2. **Confidence-accuracy correlation ρ**: model预测的confidence和实际对不对是否correlated。Baseline ρ ≈ -0.22（越confident越错！）。GASP ρ ≈ +0.62（well-calibrated）。

3. **Temporal robustness**: frame gap 24时baseline掉到 < 5%，GASP保持 > 85%。

这三个数字加在一起说的是：**baseline VLM的attention matrix本质上不encode geometric information，它们只encode semantic similarity**。GASP训练后attention space里有了geometric subspace。

---

## 训完之后呢

inference时，correspondence head直接discard。model就是个普通的VLM，没有额外的3D input，没有额外的latency。

但是attention的$W_Q$和$W_K$已经被geometric gradient reshape了。Appendix E里推导了gradient flow — $\mathcal{L}_{corr}$和$\mathcal{L}_{depth}$的gradient通过correspondence head backprop到attention weights，让$W_Q^T W_K$这个product里emerge出一个geometric alignment term $M_{geo}$，它和原本的semantic alignment term $M_{sem}$加在一起。

也就是说geometric prior被"baked into"了attention weights本身。这是为什么inference时auxiliary head可以discard — geometric reasoning capability已经internalized了。

---

## Downstream结果

最impressive的几个数字：

- **Camera Pose Estimation**: +18.2% (LLaVA), +18.7% (Qwen)
- **Object Counting on VSI-Bench**: +29.0% (LLaVA), +7.8% (Qwen) — 这个特别大
- **Multi-View on BLINK**: +15.0% (LLaVA), +11.9% (Qwen)

Object counting +29%的解释是：view-invariant feature帮model maintain object identity，不会因为视角变化就double-count或者miss object。

**General VQA基本不掉**：Video-MME +2.6, TempCompass +1.9。唯一掉的是NextQA -1.9。作者解释是capacity trade-off — action understanding更依赖object semantics和temporal dynamics，spatial specialization会hurt一点。这个解释我觉得reasonable。

---

## Fairness baseline的关键对照

这个对照必须highlight。作者用同样的DL3DV data，但把correspondence reformulate成VQA pair (e.g., "Which point in Image-2 corresponds to...")，做SFT。

结果：DL3DV VQA SFT居然让baseline degrade（Cam Pose 22.7%→19.8%，Object Count 23.5%→21.4%）。

这说明什么？**不是data exposure的事，是objective的事**。同样看correspondence data，用contrastive + depth consistency loss学，比把它formulate成QA pair好得多。这间接证明了paper的核心thesis — VQA-based supervision encourage memorization，geometric prior-based supervision encourage generalization。

---

## 几个我想push back的地方

1. **Pseudo ground-truth concern**: 用VGGT生成pseudo depth和correspondence，noise怎么propagate到GASP训练里？paper没分析。

2. **没有和RAFT/SuperPoint直接比PCK**: 如果RAFT能90% PCK而GASP只70%，那VLM internal representation还差很多。不过作者可能argue说，RAFT是specialized model，GASP是要general VLM也学会这件事，目标不同。

3. **+29% Object Counting的解释hand-wavy**: 是不是单纯因为temporal attention改善？需要更controlled的实验。

4. **$M_{geo}$ + $M_{sem}$ decomposition是conceptual的**: 没empirical verification。可以做个linear probe看QK space里是不是真的有geometric subspace。比如用probing classifier从QK matrix predict correspondence label，看能不能decode出geometric info。

5. **Outdoor / non-DL3DV generalization没显式测**: indoor场景训的，outdoor怎么样？

---

## 我为什么觉得这篇paper有意思

它和你的"software 2.0" philosophy特别契合 — **别hard-code geometric constraints，让model从fundamental signal里learn**。

也和你在nanoGPT里强调的"train on next token, learn structure of world"呼应 — 这里是"train on correspondence, learn geometric structure of world"。

更深一层，它说的其实是：**high-level capability不是直接supervise出来的，是从low-level principle emerge出来的**。你想让model懂spatial reasoning，不要直接教它spatial reasoning QA，要教它correspondence这个原子能力，spatial reasoning自然emerge。

这个intuition我觉得可以推广到很多地方。比如你想让model懂physics，不要直接教它Newton's law的QA，要教它"intervention → outcome"的correspondence。你想让model懂causality，不要直接教它counterfactual QA，要教它"do X vs do Y"的contrastive。

这是个deep idea，这篇paper只是一个instantiation。

---

References:
- Project page: https://danielchyeh.github.io/GASP/
- VGGT: https://arxiv.org/abs/2503.05534
- DL3DV: https://arxiv.org/abs/2311.14022
- VSI-Bench bias hacking paper (Brown et al.): https://arxiv.org/abs/2511.04655
- DiffTrack QK-matching metric: https://arxiv.org/abs/2506.17220
- RAFT (soft-argmax): https://arxiv.org/abs/2003.12039
- MASt3R (correspondence in 3D): https://arxiv.org/abs/2406.09656
- Dust3R: https://arxiv.org/abs/2312.14132
- RoPE: https://arxiv.org/abs/2104.09864
- All-Angles Bench: https://arxiv.org/abs/2504.15280
- VSI-Bench: https://arxiv.org/abs/2412.14171
- BLINK: https://arxiv.org/abs/2404.12330
- Zeiler & Fergus hierarchical features: https://arxiv.org/abs/1311.2901

---

# GASP: Geometric-Aware Spatial Priors 论文深度解读

## 论文核心idea

Andrej, 这篇paper的立意非常对你口味 - 它本质上是在说：**别再让VLM死记硬背3D VQA答案了，要让它学会世界本身的几何consistency**。作者Chun-Hsiao Yeh (FAIR Meta + UC Berkeley, Ma lab) 提出GASP，把correspondence + depth consistency这两个geometric prior直接inject到LLM transformer的所有layer里，训练完后丢弃auxiliary head，inference时就是个普通VLM。

Project page: https://danielchyeh.github.io/GASP/
arXiv (推测): search "Beyond 3D VQAs GASP" on arxiv.org
相关baseline: VGGT (https://arxiv.org/abs/2503.05534), Dust3R (https://arxiv.org/abs/2312.14132), MASt3R (https://arxiv.org/abs/2406.09656)

---

## 1. Motivation & Diagnostic Analysis

### 1.1 现有两条path的死结

**Path 1: 3D VQA SFT/RL fine-tuning**
- 在VSI-Bench、3D-LLM这类dataset上做SFT会让model学到superficial correlations
- Appendix D做了一个brilliant的bias hacking实验：把VSI-Bench里object/room size的均值作为textual prior喂给baseline VLM，Object Abs Distance从0.14 → 0.61 (+0.47)，居然超过专门fine-tune的VLM-3R (0.49)
- 这说明VSI-Bench本身有严重的non-visual shortcut，参考 [Brown et al., 2025] (https://arxiv.org/abs/2511.04655)
- Figure 4的generalization gap图很说明问题：SpaceR-7B在VSI-Bench上+14.2%，但在MMSI-Bench/STI-Bench/SpaceVista上掉-7.7%

**Path 2: 集成specialized 3D encoder (VGGT, point cloud, BEV)**
- 增加model size & inference latency
- 必须frozen，因为3D训练pipeline和VLM不兼容
- LLM要align自己的2D feature和这些rigid pre-computed 3D feature，alignment challenge巨大

### 1.2 核心hypothesis

> Robust spatial intelligence emerges from learning the fundamental perceptual signals of 3D geometry - specifically, **visual correspondence across changing viewpoints** = object constancy.

这非常接近你常讲的"model should learn the world model from prediction" - 这里是"model should learn geometric consistency from correspondence"。

### 1.3 Diagnostic: baseline VLM的QK-matching基本是0

这是paper最有洞察力的analysis。Section 3把self-attention的similarity matrix S = QK^T分解成4个quadrant：

$$S = QK^T = \binom{Q_V}{Q_L}\binom{K_V^T \quad K_L^T} = \binom{Q_V K_V^T \quad Q_V K_L^T}{Q_L K_V^T \quad Q_L K_L^T}$$

变量解释：
- $Q, K, V \in \mathbb{R}^{(N+M) \times d_k}$: 整个sequence的query/key/value
- $N$: visual token数, $M$: language token数, $d_k$: attention head dim
- $Q_V, K_V$: visual token的query/key
- $Q_L, K_L$: language token的query/key
- $Q_V K_V^T$: visual self-attention submatrix - **这个才是geometric reasoning的核心**

作者测了Qwen2.5-VL-7B和LLaVA-NeXT-Video-7B在DL3DV 200个held-out sequence上的PCK (Percentage of Correct Keypoints)，发现：
- Layer-wise peak PCK < 5% (Figure 3a, d)
- Confidence-accuracy Pearson correlation ρ ≈ -0.22 (Figure 3b, e) - 这是**systematic miscalibration的statistical signature**，model越confident越错
- Temporal robustness: 超过8帧gap掉到 < 5% (Figure 3c, f)

Reference for QK-matching这个metric: Emergent temporal correspondences from video diffusion transformers (https://arxiv.org/abs/2506.17220), DiffTrack

---

## 2. GASP架构详解

### 2.1 整体设计

```
Input video frames + text
        ↓
[Visual Encoder] (frozen-ish via LoRA)
        ↓ visual tokens V ∈ R^{N×d}
[LLM Transformer Layer 1]  ← H_c (correspondence head) → L_corr + L_depth
        ↓
[LLM Transformer Layer 2]  ← H_c
        ↓
   ...
[LLM Transformer Layer L] ← H_c
        ↓
LM head → L_LM
```

Key design choices:
1. **Correspondence head只在训练时存在**，inference时discard
2. **H_c是2-layer MLP**: $d \to 2 d_{emb}$ (GELU) $\to d_{emb}$
   - Qwen2.5-VL-7B: $d_h = 3584$, 28 layers
   - LLaVA-NeXT-Video-7B: $d_h = 4096$, 32 layers
3. **SVD initialization**: H_c的weights从同层pre-trained query projection matrix $W_Q$做SVD分解初始化，这样minimally disrupt pre-trained representations
4. **Deep supervision across ALL layers**: 这是关键 - 不是只在最后一层supervise

### 2.2 View-Invariant Visual Correspondence Loss (L_corr)

对anchor point $\mathbf{p}_i^a$ in source frame $a$，target frame $b$里的positive是$\mathbf{p}_i^b$，negatives是同frame其他所有点。用InfoNCE：

$$\mathcal{L}_i = -\log \frac{\exp(\langle \mathbf{e}_i^a, \mathbf{e}_i^b\rangle / \tau)}{\exp(\langle \mathbf{e}_i^a, \mathbf{e}_i^b\rangle / \tau) + \sum_{k \neq i} \exp(\langle \mathbf{e}_i^a, \mathbf{e}_k^b\rangle / \tau)}$$

变量解释：
- $\mathbf{e}_i^a = \mathcal{H}_c(\mathbf{v}_i^{a,(l)})$: anchor point $i$ 在frame $a$ 经过H_c后的embedding
- $\mathbf{e}_i^b$: 同一个3D点在frame $b$ 的corresponding embedding
- $\mathbf{e}_k^b$: frame $b$ 里其他点作为negative
- $\tau$: temperature hyperparameter
- $\langle \cdot, \cdot \rangle$: L2-normalized后的cosine similarity
- Negatives: 从除了anchor frame外的所有frame采样，maximize diversity

Why contrastive 而不是regression? 作者在paper里专门argue：
- 学习view-invariant embeddings而不是view-specific coordinates
- 自然scale到大量negatives
- 在高维feature space里coordinate regression poorly calibrated

这跟你之前讲的"contrastive learning gives more semantically meaningful features than classification"是一致的。

### 2.3 Depth-Aware 3D Consistency Loss (L_depth)

这部分是paper的另一个亮点 - **depth不是用来predict的，是用来regularize的**。

Step 1: 用contrastive loss里的similarity scores算soft matching distribution：

$$\mathbf{A}_{ij} = \frac{\exp(\langle \mathbf{e}_i^a, \mathbf{e}_j^b\rangle / \tau)}{\sum_{k=1}^{N_{cand}} \exp(\langle \mathbf{e}_i^a, \mathbf{e}_k^b\rangle / \tau)}$$

变量解释：
- $\mathbf{A}_{ij}$: anchor point $i$ 在frame $b$ 里匹配到candidate patch $j$的belief
- $N_{cand}$: frame $b$里candidate patch总数
- **复用Eq. 5的similarity computation** - 不重复计算，效率高

Step 2: Soft-Argmax算expected depth (参考RAFT, https://arxiv.org/abs/2003.12039)：

$$\hat{d}_i^b = \sum_{j=1}^{N_{cand}} \mathbf{A}_{ij} \cdot d_j^b$$

变量解释：
- $d_j^b$: candidate patch $j$ 在frame $b$的ground-truth depth (来自DL3DV)
- $\hat{d}_i^b$: 模型预测anchor point $i$在frame $b$对应位置的expected depth
- **关键**: 这个weighted sum是differentiable的w.r.t. correspondence embeddings

Step 3: Relative depth consistency loss:

$$\mathcal{L}_{depth} = \frac{1}{N_{valid}} \sum_{i \in valid} \frac{|d_i^b - \hat{d}_i^b|}{d_i^b + \hat{d}_i^b + \epsilon}$$

变量解释：
- $d_i^b$: ground-truth depth of point $i$ at its corresponding location $\mathbf{p}_i^b$ in frame $b$ (来自point correspondence annotation)
- $\hat{d}_i^b$: 模型predicted expected depth
- $\epsilon$: 防止divide-by-zero的小常数
- **Relative formulation = scale-invariant**: 处理不同scene depth range不需要per-scene normalization
- Summation over $valid$: 只对visibility和confidence够高的点做loss

**为什么这个loss是discriminative geometric regularizer而不是depth estimator?**

考虑这个场景：foreground和background各有一个texture完全相同的物体。Standard contrastive loss会因为visual embedding相似而错配。但因为depth不同 ($d_{fg} \neq d_{bg}$), L_depth会penalize这个match，force model学到context-aware representations。

更一般地：**visually similar patches at different depths**被force to have lower feature similarity。这解决了repetitive texture和foreground-background confusion。

### 2.4 Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{LM} + \lambda_c \mathcal{L}_{corr} + \lambda_d \mathcal{L}_{depth}$$

- $\lambda_c = 0.3$: contrastive weight
- $\lambda_d = 1.0$: depth weight
- $\mathcal{L}_{LM}$: 标准next-token prediction loss (LLaVA-Video-178K)
- 4x higher differential learning rate for $\mathcal{H}_c$ head

Training setup: 32 H200 GPUs, ~10 hours, LoRA rank 512 (LLaVA) or 128 (Qwen), AdamW + cosine LR schedule (peak 1e-4), gradient clip 1.0, bfloat16 mixed-precision + gradient checkpointing

---

## 3. Training Data Construction

### 3.1 Geometric supervision来源

DL3DV-10K (https://arxiv.org/abs/2311.14022) - large-scale 3D scene dataset，follow VGGT的annotation recipe:
1. 用COLMAP SfM重建的camera intrinsics $K \in \mathbb{R}^{3\times3}$和extrinsics $[R|t] \in \mathbb{R}^{3\times4}$
2. Query frame的depth map $\mathbf{D}_0 \in \mathbb{R}^{H\times W}$
3. Backproject pixel to 3D world: $\mathbf{p}_w = K^{-1} D_0(u,v) [u,v,1]^T$
4. Project to subsequent frames: $\mathbf{p}_i = K[R_i|t_i] \mathbf{p}_w$
5. Validate via depth consistency: $|D_{proj} - D_{map}| < 0.05 \times \min(D_{proj}, D_{map})$
6. Boundary margin of 4 pixels to avoid projection artifacts

### 3.2 Sequence sampling strategy

- Sample anchor frame $t_a$ from video $\mathcal{V} = \{I_t\}_{t=1}^{T_{max}}$
- Sample remaining $F-1$ frames from local window $[t_a - R, t_a + R]$ uniformly
- $F \in [8, 24]$ random, $R = 48$
- Result: ~1.75M sequences with rich motion parallax
- 8×8 (coarse) + 24×24 (fine) grids per sequence
- Top 50% tracks ranked by visibility duration
- Negative samples: random spatial perturbations within 50%

### 3.3 Joint training with LLaVA-Video-178K

Interleave geometric data with general video QA to prevent catastrophic forgetting。这个joint training的mix至关重要 - 没有general video QA，model会spatial特化但lose language capability。

---

## 4. Evaluation: Internal Geometric Representations

### 4.1 Three diagnostic metrics

这个evaluation framework值得单独讲 - 它直接probe VLM的internal representation quality，不是end-to-end VQA accuracy。

**Metric 1: Layer-wise PCK**

$$S_{ij}^{1,k} = \frac{\mathbf{f}_{i,Q}^1 \cdot (\mathbf{f}_{j,K}^k)^T}{\|\mathbf{f}_{i,Q}^1\|_2 \|\mathbf{f}_{j,K}^k\|_2}$$

$$\mathbf{p}_i^k = \operatorname*{argmax}_{\mathbf{p} \in \Omega_k} (S^{1,k}(\mathbf{p}_i^1, \mathbf{p}))$$

变量解释：
- $\mathbf{f}_{i,Q}^1 \in \mathbb{R}^d$: query descriptor of patch $i$ in frame 1, shape $HW \times d$
- $\mathbf{f}_{j,K}^k \in \mathbb{R}^d$: key descriptor of patch $j$ in frame $k$
- $S^{1,k} \in \mathbb{R}^{HW \times HW}$: pairwise cosine similarity matrix
- $\mathbf{p}_i^k$: predicted location of point $i$ in frame $k$ via argmax
- $\Omega_k$: spatial domain of feature grid in frame $k$
- $\delta = 2$ patches: PCK threshold
- $\mathcal{T}_i = \mathrm{Interpolate}(\mathrm{Concat}(\mathbf{p}_i^1, ..., \mathbf{p}_i^F))$: full predicted track

**Metric 2: Confidence-Accuracy Correlation (Pearson ρ)**

$$\rho_\ell = \frac{\sum_{i=1}^N (c_i - \bar{c})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^N (c_i - \bar{c})^2}\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2}}$$

- $c_i$: max attention probability (confidence) for prediction $i$
- $y_i \in \{0, 1\}$: PCK@2 correctness label
- $\rho > 0$: well-calibrated
- $\rho < 0$: positional bias - model confidently predicts incorrect matches

**Metric 3: Temporal Robustness**

$$Y(\Delta t) = \mathrm{PCK}(\Delta t) / \mathrm{PCK}(\Delta t = 1)$$

- $\Delta t \in \{1, 2, ..., 24\}$: temporal offset between matched frames
- Normalized to 1.0 at $\Delta t = 1$ for fair comparison of degradation rates

### 4.2 Diagnostic Results (Figure 3)

| Metric | LLaVA Baseline | LLaVA + GASP Full | Qwen Baseline | Qwen + GASP Full |
|---|---|---|---|---|
| Peak Layer PCK | < 5% | > 70% | < 5% | > 70% |
| ρ (calibration) | -0.22 | +0.62 | -0.22 | +0.62 |
| Temporal robustness @ Δt=24 | < 5% | > 85% | < 5% | > 85% |

Peak位置: middle-to-deep layers (LLaVA: 20-25, Qwen: 25-28)。这跟你之前观察到deep layers有semantic information、shallow layers有low-level features一致。

Full model (with L_depth) consistently outperform correspondence-only，证明depth supervision确实resolves了foreground-background confusion。

---

## 5. Downstream Spatial Reasoning Benchmarks

### 5.1 主要结果 (Table 1)

**All-Angles Bench** (multi-view understanding, https://arxiv.org/abs/2504.15280):
- Camera Pose Est: 34.1 → 52.8 (+18.7) for Qwen, 22.7 → 40.9 (+18.2) for LLaVA
- Relative Direction: 36.9 → 37.2 (+0.3) Qwen, 24.7 → 29.8 (+5.1) LLaVA

**VSI-Bench** (object permanence, https://arxiv.org/abs/2412.14171):
- Object Counting: 33.8 → 41.6 (+7.8) Qwen, 23.5 → 52.5 (+29.0) LLaVA - **巨大!**
- Route Planning: 26.8 → 30.4, 24.7 → 32.5

**BLINK spatial subset** (https://arxiv.org/abs/2404.12330):
- Multi-View: 41.5 → 53.4 (+11.9) Qwen, 42.1 → 57.1 (+15.0) LLaVA

**Comparison with specialized 3D VLMs:**
- VG-LLM, VLM-3R, AoTD这些fine-tuned on 3D VQA的model，在in-domain强但OOD差
- GASP没有任何3D VQA training data，但比它们更generalize

### 5.2 Fairness Baseline的关键对照

作者做了一个非常honest的对照：用同样的DL3DV point tracks，但reformulate成VQA pairs (e.g., "Which labeled point in Image-2 corresponds to..."), 用同样的data mix做SFT。

**结果**: DL3DV VQA SFT甚至让baseline degrade (Cam Pose 22.7%→19.8%, Object Count 23.5%→21.4% for LLaVA)。这证明gain来自GASP的geometric objective本身，不是data exposure。

### 5.3 General VQA Capability (Table 2, 3)

**CV-Bench** (https://arxiv.org/abs/2406.17760):
- Qwen2.5-VL-7B + GASP Full: Overall 79.8 (vs base 76.6)
- 3D-depth: 87.3 (+1.8), 3D-distance: 79.2 (+6.5)

**Video-MME w/o sub**: 60.6 → 63.2 (+2.6)
**Video-MME w/ sub**: 59.3 → 61.6 (+2.3)
**TempCompass**: 68.4 → 70.3 (+1.9)
**NextQA**: 76.6 → 74.7 (-1.9) - **唯一掉的地方**

这个NextQA的drop很有意思：作者hypothesize是capacity trade-off，geometric specialization以action-understanding为代价。Action task依赖object semantics + temporal dynamics多于spatial localization。

---

## 6. Ablation Studies (Table 4)

### 6.1 LoRA rank impact

| LoRA Rank | Avg PCK (LLaVA) | All-Angles | VSI | BLINK |
|---|---|---|---|---|
| 64 | - | 28.5 | - | - |
| 128 | 8.4 | 30.1 | - | 44.9 |
| 256 | 13.7 | 32.6 | 30.6 | 45.8 |
| 512 | 17.1 | 35.8 | 33.9 | 47.5 |
| 1024 | 26.2 | 38.1 | 37.1 | 51.0 |

| LoRA Rank | Avg PCK (Qwen) | All-Angles | VSI | BLINK |
|---|---|---|---|---|
| 64 | 18.2 | 38.5 | 37.3 | 70.2 |
| 128 | 26.7 | 43.4 | 36.9 | 74.3 |
| 256 | 28.8 | 41.8 | 35.5 | 73.5 |
| 512 | 31.2 | 40.2 | 34.1 | 72.4 |
| 1024 | 32.5 | 38.9 | 33.2 | 73.8 |

**关键发现**: Avg PCK随rank单调上升，但downstream benchmark peaks early (LLaVA: 512, Qwen: 128)。Higher PCK ≠ better spatial reasoning - 高rank会开始harm language capability。

这跟你之前说的"more capacity isn't always better - it can hurt generalization if not properly regularized"是一致的。

### 6.2 Layer injection strategy

| Layers | Avg PCK | All-Angles | VSI | BLINK |
|---|---|---|---|---|
| LLaVA Layer 10-18 | 21.7 | 34.8 | 35.9 | 47.7 |
| LLaVA Layer 18-25 | 25.1 | 37.5 | 35.2 | 49.5 |
| LLaVA Layer 25-32 | 25.8 | 39.1 | 36.5 | 49.3 |
| **All Layers (1-32)** | **26.2** | **38.1** | **37.1** | **51.0** |

| Layers | Avg PCK | All-Angles | VSI | BLINK |
|---|---|---|---|---|
| Qwen Layer 10-16 | 19.8 | 37.9 | 34.2 | 68.2 |
| Qwen Layer 16-22 | 23.3 | 38.8 | 35.5 | 71.1 |
| Qwen Layer 22-28 | 25.2 | 42.7 | 37.4 | 72.8 |
| **All Layers (1-28)** | **26.7** | **43.4** | **36.9** | **74.3** |

**关键发现**: All layers supervision最好。**Geometric consistency is fundamentally hierarchical**:
- Early layers: low-level visual features (edges, corners)
- Middle layers: object parts and boundaries
- Deep layers: semantic-geometric alignment

如果只在deep layers监督，shallow layers会继续学view-dependent features，造成representational bottleneck。

---

## 7. Gradient Analysis (Appendix E - 最重要的理论部分)

这部分是paper的核心理论贡献，详细推导了geometric supervision如何reshape QK space。

### 7.1 Gradient decomposition

$$\frac{\partial \mathcal{L}_{total}}{\partial \theta^{(l)}} = \underbrace{\frac{\partial \mathcal{L}_{LM}}{\partial \theta^{(l)}}}_{\text{Language Modeling}} + \underbrace{\lambda_c \frac{\partial \mathcal{L}_{corr}}{\partial \theta^{(l)}} + \lambda_d \frac{\partial \mathcal{L}_{depth}}{\partial \theta^{(l)}}}_{\text{Geometric Supervision}}$$

变量解释：
- $\theta^{(l)} = \{W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}\}$: layer $l$ self-attention weights
- 第一项: dense semantic gradient from next-token prediction
- 第二项: spatially localized geometric gradient

### 7.2 Chain rule through attention

$$\frac{\partial \mathcal{L}_{corr}}{\partial W_Q^{(l)}} = (X^{(l-1)})^T \cdot \left[\frac{1}{\sqrt{d_k}} K \cdot \nabla_A^{softmax} \cdot V \cdot \frac{\partial \mathcal{L}_{corr}}{\partial V^{(l)}}\right]$$

$$\frac{\partial \mathcal{L}_{corr}}{\partial W_K^{(l)}} = (X^{(l-1)})^T \cdot \left[\frac{1}{\sqrt{d_k}} Q^T \cdot \nabla_A^{softmax} \cdot V \cdot \frac{\partial \mathcal{L}_{corr}}{\partial V^{(l)}}\right]$$

其中 $\nabla_A^{softmax} = \mathrm{diag}(A)(I - \mathbf{1}A)$ 是softmax的Jacobian。

### 7.3 W_Q^T W_K的分解

$$W_Q^{T,(l)} W_K^{(l)} \approx M_{geo} + M_{sem}$$

$$M_{geo} = \sum_{t=1}^T \eta \lambda_c \left[\frac{\partial \mathcal{L}_{corr}}{\partial W_Q^{(l)}}\right]^T \left[\frac{\partial \mathcal{L}_{corr}}{\partial W_K^{(l)}}\right]$$

- $M_{geo}$: encodes geometric alignment (high values for corresponding 3D locations)
- $M_{sem}$: encodes semantic similarity (from $\mathcal{L}_{LM}$)
- 几何项来自累积的gradient outer product

### 7.4 Depth consistency gradient

$$\frac{\partial \mathcal{L}_{depth}}{\partial S_{ij}} = \mathcal{D}(d_i, d_j) \cdot A_{ij}(1 - A_{ij})$$

- $\mathcal{D}(d_i, d_j)$: depth discrepancy between points $i$ and $j$
- $A_{ij}(1 - A_{ij})$: **amplifies gradients for mid-confidence predictions** ($A_{ij} \approx 0.5$)
- 教model suppress geometrically invalid matches

### 7.5 Final effective attention

$$S_{ij}^{effective} = \frac{x_i^T W_Q^T W_K x_j}{\sqrt{d_k}} - \lambda_d \cdot \mathcal{D}(d_i, d_j) + \text{noise}$$

**Depth penalty被implicitly encoded进$W_Q^T W_K$**。这就是为什么inference时可以discard correspondence head - geometric prior已经被permanently baked into $\theta^{(l)}$。

---

## 8. 与RoPE的关系 (Appendix F)

这部分有意思 - paper专门argue GASP和positional embeddings是互补的，不是替代关系。

### 8.1 RoPE的局限

Standard ViTs和VLMs用absolute learnable PEs或RoPE (https://arxiv.org/abs/2104.09864)。Video Transformers扩展到3D-RoPE (Vivit, https://arxiv.org/abs/2106.10270; TimeSformer, https://arxiv.org/abs/2102.05063)。

但RoPE只提供static coordinate information ("this token is at location $(x,y)$")。Baseline models (Qwen2.5-VL, LLaVA-NeXT)已经用了advanced RoPE，但correspondence accuracy还是near-zero。

**经验证据**: RoPE提供坐标信息insufficient让model学到"Frame t里的object at $(x_1, y_1)$ 和 Frame t+1里的object at $(x_2, y_2)$ 是同一个entity"。

### 8.2 GASP的complementary role

| 维度 | RoPE | GASP |
|---|---|---|
| Signal level | Input-level | Interaction mechanism (QK^T) |
| Content-awareness | Content-agnostic (blank wall = complex face) | Content-aware (via L_corr) |
| 3D consistency | Explicit 3D input required | Implicit (baked into 2D weights) |
| Function | "Where" in image grid | "What" and "which" across spatio-temporal manifold |

GASP complements RoPE: RoPE告诉model token在image grid里的位置，GASP教model跨视角识别同一个entity。

---

## 9. Intuition Building: Why This Works

### 9.1 The Deep Learning lesson

这让我想到你之前在YouTube讲的"train a model to predict the next token, and it learns the structure of the world"。这篇paper本质上是这个philosophy的spatial reasoning版本：

> Train a model to predict correspondences across viewpoints, and it learns the 3D geometric structure of the world.

而不是: Train a model to answer "what's in front of the chair?" and it learns to memorize dataset-specific patterns.

### 9.2 Why Correspondence is the Right Auxiliary Task

Correspondence is **the most fundamental spatial perception task**:
- Optical flow, point tracking, SLAM, SfM, multi-view stereo都依赖它
- 它定义了object constancy - 同一个3D entity跨视角的identity preservation
- 它是view-invariant representation的天然训练信号

参考: 
- TAP-Vid (https://arxiv.org/abs/2106.02091)
- OmniMotion (https://arxiv.org/abs/2306.05722)  
- CoTracker (https://arxiv.org/abs/2303.12021)

### 9.3 Why Depth as Regularizer (not Predictor)

这里有个deep insight - 作者刻意avoid训练一个depth prediction head (像VGGT那样)。原因:
1. Depth prediction是ill-posed (scale ambiguity)
2. Depth prediction需要per-scene normalization
3. Depth prediction会competewith而不是complement correspondence

用depth作为**soft matching的discriminative regularizer**:
- Depth disagreement → penalize match → force embeddings to encode depth-aware features
- Depth agreement → no penalty → matches go through

这类似于metric learning里的hard negative mining，只不过这里的"hardness"来自depth discrepancy。

### 9.4 Why All Layers Matter

这个发现很重要 - geometric reasoning不是deep layer的专利。Hierarchical feature learning theory (Zeiler & Fergus, https://arxiv.org/abs/1311.2901)告诉我们:
- Layer 1-5: Gabor filters, edges
- Layer 6-10: textures, simple patterns  
- Layer 11-15: object parts
- Layer 16+: semantic categories

Spatial correspondence同样需要hierarchical processing:
- Low-level: corner matching (rotation/scale invariant local features, like SIFT)
- Mid-level: object part correspondence
- High-level: semantic-geometric alignment (e.g., "chair leg here = chair leg there")

只在deep layersupervise会让shallow layerscontinue学习view-dependent low-level features，整个representation就broken。

这跟你的"microscope into the model"philosophy一致 - probe intermediate representations, 不要只看final output。

---

## 10. Limitations & Future Work

作者自己承认:
1. **Pseudo ground-truth depth**: 依赖VGGT等model的pseudo-label，noise会propagate
2. **Modest trade-off on action-centric tasks**: NextQA掉1.9%
3. **No combination with VQA supervision**: 论文是"instead of"的关系，未来可以"complementary"

可能的extension:
1. **Combine with reinforcement learning**: 用RL refine correspondence learning (类似AlphaGo的self-play)
2. **Scale to larger architectures**: Qwen2.5-VL-72B, LLaMA-3.2-90B
3. **Multi-modal correspondence**: 不仅visual-visual，还可以text-visual correspondence (e.g., "this caption describes this region")
4. **Action-conditioned correspondence**: 在ego-centric video里学习first-person action对应的third-person action (参考 Viewpoint Rosetta Stone, https://arxiv.org/abs/2503.09612)
5. **Long-horizon tracking**: 当前F=8-24 frames，扩展到分钟级甚至小时级

---

## 11. 与你之前工作的联系

你之前讲过nanoGPT、micrograd、microscope into transformer layers等。这篇paper和你这些interests的intersection:

1. **Interpretability**: QK-matching analysis就是microscope into attention
2. **First principles**: 用correspondence而不是VQA，是从first principle重新思考spatial reasoning
3. **Karpathy's "software 2.0"**: GASP就是Software 2.0的spatial reasoning版本 - 学习geometric constraints而不是hand-code them
4. **Positional encoding debate**: 你之前讨论过RoPE vs absolute PE - 这篇paper提供了新的角度: RoPE提供"where", GASP提供"what/which"

References:
- nanoGPT: https://github.com/karpathy/nanoGPT  
- micrograd: https://github.com/karpathy/micrograd
- "Intro to LLMs" YouTube: 讨论了tokenization, attention, transformer架构

---

## 12. Critique / Open Questions

我可以提几个值得push back的地方：

1. **Pseudo ground-truth concern**: VGGT的pseudo-label可能有systematic bias (e.g., reflective surfaces, thin structures)。作者没分析这个noise对训练的影响。

2. **Comparison with simpler baselines**: 没有和简单的optical flow baseline (RAFT, https://arxiv.org/abs/2003.12039) 或者经典correspondence方法 (SuperPoint, https://arxiv.org/abs/1712.07634) 在PCK上直接比较。如果RAFT能达到90% PCK而GASP只有70%，那VLM的internal representation还是差很多。

3. **Object Counting +29%的surprising result**: 这个gain huge，但作者的解释 ("view-invariant features prevent double-counting")比较hand-wavy。需要更controlled的实验 - 是不是单纯因为temporal attention改善了？Multi-object tracking task的evaluation需要更细的analysis。

4. **Generalization to non-DL3DV scenes**: 在indoor场景(DL3DV主要是indoor)训练，在outdoor场景表现如何？paper没显式evaluate。

5. **Fairness baseline的strictness**: 把correspondence reformulate成VQA确实是一个baseline，但更好的baseline可能是: 用同样的correspondence loss但只在最后一层监督。这样能更干净地isolate "deep supervision across all layers"的贡献。

6. **Inference cost**: 训练时discard H_c，但inference时和baseline完全一样的cost吗？需要confirm。

7. **QK structure**: M_geo + M_sem的decomposition是conceptual的，没有empirical verification。可以做个linear probe看看QK space是否真的有geometric subspace。

---

## 13. Final Thoughts

这篇paper有非常清晰的research taste - 它distinguishes between:
- **Surface-level supervision** (VQA) → memorization
- **Fundamental geometric prior** (correspondence + depth) → generalization

而且它通过internal representation analysis (PCK, ρ, temporal robustness) 而不只是end-to-end metrics来论证。这种methodology值得推广。

我觉得最exciting的方向是: **把这种"supervise intermediate representations with structured signals"的paradigm扩展到其他modality**:
- Audio-visual correspondence supervision for audio VLMs
- Physics-aware supervision (e.g., object permanence, gravity) for embodied agents
- Causal supervision (e.g., intervention-outcome pairs) for reasoning VLMs

如果有任何地方需要更deep dive (e.g., 具体的PyTorch implementation of correspondence head, gradient derivation的某个step,或者和某个具体baseline的head-to-head comparison)，我可以继续展开。
