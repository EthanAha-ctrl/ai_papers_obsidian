---
source_pdf: ACE-Brain-0 Spatial Intelligence as a Shared Scaffold for Universal Embodiments.pdf
paper_sha256: 89db3051f4b84dd20d71f507bc70d931205a6776ef2d643e81b991ec4f57c835
processed_at: '2026-07-18T00:13:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ACE-Brain-0: Spatial Intelligence作为Universal Embodiments的Shared Scaffold

## 1. Core Insight: 为什么Spatial Intelligence可以作为Cross-Embodiment的共享基础

这篇paper的核心论点非常清楚: **autonomous driving, UAV, robotics这些heterogeneous embodiments在morphology上差异巨大, 但它们都需要建模3D空间——感知object layout、理解geometric relations、预测actions的spatial consequences**。这个common denominator让spatial modeling成为天然的domain-agnostic foundation。

这里有一个hierarchical认知的intuition值得思考:
- **Coarse level (VLN类任务)**: AD和UAV主要做spatial-aware planning, 重点是trajectory planning或behavior decision
- **Fine level (VLA类任务)**: Embodied interaction需要fine-grained execution, 涉及low-level kinematic control和precise object manipulation

这种coarse-to-fine的认知进阶刚好契合了SSR paradigm的stage设计。

参考文献关于neuroscience的部分很有意思(References [127, 128, 129, 130]):
- Hippocampal/entorhinal system存在internal GPS (grid cells, place cells) [Hafting et al. 2005](https://www.nature.com/articles/nature03721), [Fyhn et al. 2004](https://www.science.org/doi/10.1126/science.1099905)
- Primate recordings发现decision-related subspaces和stimulus-related subspaces几乎orthogonal [Tian et al. 2026](https://www.nature.com/articles/s41467-026-xxxxx), [Goudar et al. 2023](https://www.nature.com/articles/s41593-023-01324-8)

这给spatial scaffold的可复用性提供了生物学inspiration。

---

## 2. Architecture详解

### 2.1 Task Formulation

定义domain集合:

$$\mathcal{M} = \{m_{\mathrm{general}}, m_{\mathrm{embodied}}, m_{\mathrm{spatial}}, m_{\mathrm{driving}}, m_{\mathrm{aerial}}\}$$

每个domain $m_k$ 诱导一个task distribution $\mathcal{D}_{m_k}$, 产生训练样本 $(o, c, y)$:
- $o \in \mathcal{O}_{m_k}$: multimodal observations (images or video sequences)
- $c \in \mathcal{C}$: task conditioning (natural language instructions, queries, goals)
- $y \in \mathcal{V}_{m_k}$: target output (text responses, reasoning traces, action sequences, trajectories)

所有任务用统一conditional autoregressive formulation:

$$p_\theta(y \mid o, c)$$

其中 $\theta$ 是shared MLLM的参数。这个设计强制让所有embodiments共享一个representation backbone和thinking substrate。

### 2.2 Multimodal Architecture

ACE-Brain-0采用标准MLLM架构, 三个核心component:

1. **Vision Encoder** + **MLP Projector**: 处理single-view images, multi-view images, video
2. **Tokenizer**: 将natural language instructions转成text tokens
3. **ACE-Brain-0 LLM Decoder**: autoregressively生成output

Formal formulation:

$$p = \mathcal{F}_{\mathrm{dec}}\Big(t_N \mid \mathcal{F}_{\mathrm{proj}}\big(\mathcal{F}_{\mathrm{enc}}(o; \theta_{\mathrm{enc}}); \theta_{\mathrm{proj}}\big), \mathcal{F}_{\mathrm{tok}}(c), t_{0:N-1}; \theta_{\mathrm{dec}}\Big)$$

变量解析:
- $o \in \mathbb{R}^{T \times H \times W \times 3}$: visual observation, T是frame数 (T=1为single image, T>1为multi-view或video)
- $\mathcal{F}_{\mathrm{enc}}(\cdot; \theta_{\mathrm{enc}})$: Vision Encoder
- $\mathcal{F}_{\mathrm{proj}}(\cdot; \theta_{\mathrm{proj}})$: MLP Projector
- $\mathcal{F}_{\mathrm{tok}}(\cdot)$: Tokenizer
- $\mathcal{F}_{\mathrm{dec}}(\cdot; \theta_{\mathrm{dec}})$: LLM Decoder
- $t_i$: 第i个generated token
- $p \in \mathbb{R}^m$: 在vocabulary size $m$ 上的probability distribution

**Architecture intuition**: Visual tokens按domain分5类—General, Spatial, Driving, Aerial, Embodied。这并不是说有5个不同的encoder, 而是说同一个encoder处理后, tokens被语义上划分到这些domain。所有visual tokens和text tokens拼成一个unified sequence送入decoder, 让decoder joint attend到visual和textual信息。

### 2.3 Autoregressive Objective

Standard left-to-right autoregressive loss:

$$\mathcal{L}_{\mathrm{full}}(\theta) = -\sum_{i=1}^{L} w_i \log p_\theta(s_i \mid s_{<i})$$

变量:
- $s_i$: 第i个token in序列 $\mathbf{s} = (s_1, \ldots, s_L)$
- $w_i$: token $s_i$ 的loss weight
- $s_{<i} = (s_1, \ldots, s_{i-1})$: preceding context

实际中只对text tokens算loss, visual tokens只做conditioning:

$$\mathcal{L}_{\mathrm{Text}}(\theta) = -\sum_{i=1, s_i \in \mathrm{Text}}^{L} w_i \log p_\theta(s_i \mid s_{<i})$$

**关于 $w_i$ 的设计选择很关键**: naive的token averaging会让long response的gradient contribution过大, sample averaging会让short response过大。他们采用**square averaging**——平衡不同sequence length的gradient contribution。这个细节在MLLM training里经常被忽视, 但实际上对训练稳定性影响很大。

---

## 3. SSR Training Paradigm详解

### 3.1 Stage 1: Spatial Scaffold Training

这是整个paradigm的foundation。分两步:
1. 用Cambrain-737K general data [Tong et al. NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/xxx) 做instruction tuning得到 $\theta_{\mathrm{base}}$
2. 用大规模spatial data训练得到 $\theta_{\mathrm{spatial}}$

$\theta_{\mathrm{spatial}}$ 作为后续所有expert training的初始化点。

**这里的关键intuition**: spatial representation是"reusable structural prior"。一旦模型学会了3D空间理解(layout, relative pose, depth, topology), 这些representation在其他domain里就不需要重新学习, 只需要学习domain-specific的部分。

### 3.2 Stage 2: Supervised Specialized Expert Fine-Tuning

从 $\theta_{\mathrm{spatial}}$ 独立初始化3个expert:
- $\theta_{\mathrm{spatial}}$: 继续spatial cognition训练
- $\theta_{\mathrm{uav}}$: 从 $\theta_{\mathrm{spatial}}$ 初始化, 专攻UAV low-altitude sensing
- $\theta_{\mathrm{ad}}$: 从 $\theta_{\mathrm{spatial}}$ 初始化, 专攻autonomous driving

**为什么必须isolation training?** Appendix A.2给出了严格的gradient interference analysis。考虑一个joint risk:

$$R_w(\theta) := \sum_{j=1}^{K} w_j R_j(\theta)$$

One shared update:

$$\theta^+ = \theta - \eta \nabla R_w(\theta) = \theta - \eta \sum_{j=1}^{K} w_j g_j(\theta)$$

变量:
- $w \in \Delta_K$: K个domain的nonnegative weights
- $g_j(\theta) := \nabla_\theta R_j(\theta)$: domain $j$ 的risk gradient
- $\eta$: learning rate

**Theorem 1 (One-step interference bound)**: 在L-smoothness assumption下 (Eq 12), 对任意morphology $i$:

$$R_i(\theta^+) \leq R_i(\theta) - \eta\Big(w_i \|g_i(\theta)\|^2 + \sum_{j \neq i} w_j \langle g_i(\theta), g_j(\theta) \rangle\Big) + \frac{L\eta^2}{2}\Big\|\sum_{j=1}^{K} w_j g_j(\theta)\Big\|^2$$

变量:
- $L$: smoothness constant (Adam + gradient clipping + weight decay让loss landscape近似quadratic)
- 第一项 $w_i \|g_i(\theta)\|^2$: **beneficial self term** (gradient沿descent方向)
- $\sum_{j \neq i} w_j \langle g_i(\theta), g_j(\theta) \rangle$: **cross terms** (其他domain的gradient对domain $i$ 的影响)
- 最后一项: second-order的smoothness penalty

**核心insight**: 当cross terms持续为负时, shared update包含一个反 $R_i$ descent方向的component, 这会stall甚至reverse domain $i$ 的progress。这就是gradient interference的本质。Isolation training通过构造方式消除cross terms。

### 3.3 Stage 3: Across-Embodiment Reconcile Model Merging

这是paper最技术性的部分。他们采用optimization-based merging (WUDI - "Whoever Started the interference Should End It" [Cheng et al. ICML 2025](https://proceedings.mlr.press/v257/xxx))。

**Task vector定义**:

$$\tau_m := \theta_m - \theta_{\mathrm{base}}$$

这是fine-tuned expert和base model的参数差。layer-wise版本:

$$\tau_{m,l} := \theta_{m,l} - \theta_{\mathrm{base},l}, \quad l \in \{1, \ldots, L\}$$

**Merging optimization**: 初始化为所有expert的平均:

$$\theta_{\mathrm{merge}}^{(0)} = \frac{1}{K} \sum_{i=1}^{K} \theta_i$$

然后迭代优化:

$$\theta_{\mathrm{merge},l}^* = \arg\min_{\theta_{\mathrm{merge},l}} \sum_{i=1}^{K} \mathbb{E}_{\mathbf{x}_{i,l} \sim \mathcal{D}_{m_i,l}} \|\theta_{i,l} \mathbf{x}_{i,l} - \theta_{\mathrm{merge},l} \mathbf{x}_{i,l}\|_2^2$$

变量:
- $\theta_{i,l}$: 第i个expert在第l层的参数
- $\theta_{\mathrm{merge},l}$: merged model在第l层的参数 (待优化)
- $\mathbf{x}_{i,l} \sim \mathcal{D}_{m_i,l}$: 第i个domain在第l层的(近似)数据分布
- $\mathcal{D}_{m_i,l}$: 用task vector近似fine-tuning data的linear subspace

**关键的upper bound derivation** (Eq 8):

$$\mathbb{E}_{\mathbf{x}_{i,l} \sim \mathcal{D}_{m_i,l}} \|(\theta_{i,l} - \tau_{\mathrm{merge},l}) \mathbf{x}_{i,l}\|_2^2 \leq \omega_{i,l}^1 \cdot \|(\theta_{i,l} - \tau_{\mathrm{merge},l})(\tau_{i,l})^\top\|_F^2 + \omega_{i,l}^2 \cdot \|\theta_{i,l} - \tau_{\mathrm{merge},l}\|_F^2$$

变量:
- $\omega_{i,l}^1, \omega_{i,l}^2$: constants
- $\|\cdot\|_F$: Frobenius norm

将这个upper bound代入, 得到:

$$\theta_{\mathrm{merge},l}^* \approx \theta_{\mathrm{pre},l} + \arg\min_{\tau_{\mathrm{merge},l}} \sum_{i=1}^{K} \frac{1}{\|\tau_{i,l}\|_F^2} \|(\tau_{\mathrm{merge},l} - \tau_{i,l})\tau_{i}^\top\|_F^2$$

**Intuition**: 这个objective的本质是让merged task vector $\tau_{\mathrm{merge}}$ 在每个expert $i$ 的task vector $\tau_i$ 的"主方向"上, 尽量接近 $\tau_i$。$\frac{1}{\|\tau_{i,l}\|_F^2}$ 起normalization作用, 让不同magnitude的task vector贡献平衡。

**Implementation**: Adam optimizer, lr=1e-5, weight_decay=0, 1000 iterations, 用FusionBench framework [Tang et al. 2025](https://arxiv.org/abs/2406.03280)。

他们还对比了alternative merging methods:
- **TSVM** (Task Singular Vector Merging) [Gargiulo et al. CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_xxx)
- **Model Soups** (vanilla parameter averaging) [Wortsman et al. ICML 2022](https://proceedings.mlr.press/v162/wortsman22a.html)

### 3.4 Stage 4: Embodied SFT

merged model $\theta_{\mathrm{merged}}$ 进一步在embodied data上做SFT得到 $\theta_{\mathrm{embodied}}$。重点在embodied interaction, task planning, action prediction。

### 3.5 Stage 5: GRPO Reinforcement Learning

最后用GRPO [Shao et al. DeepSeekMath 2024](https://arxiv.org/abs/2402.03300) 进一步refine, 100k mixed data from spatial, ad, uav, embodied。

GRPO objective:

$$\mathcal{I}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{\{o_i\}_{i=1}^G \sim P(Q)(\cdot|q)} \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \Big[\min\Big(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t}, \exp\Big(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\Big)\hat{A}_{i,t}\Big)\Big]$$

变量:
- $q$: question
- $\{o_1, \ldots, o_G\}$: 从old policy $\pi_{\theta_{\mathrm{old}}}$ 采样的G个outputs
- $\pi_\theta$: 当前policy
- $\varepsilon$: clipping hyperparameter (类似PPO的clip)
- $\hat{A}_{i,t}$: group-relative advantage

**注意**: 他们省略了KL divergence penalty (与原GRPO不同), 因为empirically发现clipped surrogate objective alone就足够regularization。这与DeepSeek-R1的发现一致。

Advantage计算用outcome supervision:

$$\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$$

变量:
- $\mathbf{r} = \{r_1, \ldots, r_G\}$: 一组G个outputs的rewards
- $r_i$: 第i个output的reward
- $\tilde{r}_i$: group-normalized reward, uniform assigned到所有tokens

**Intuition**: GRPO用relative rewards (group内normalization)代替absolute rewards, 这样不需要learned value function, 简化了critic的训练。在embodied场景里, 多步planning任务本身reward稀疏, group-relative信号比absolute reward更稳定。

---

## 4. Experiments: 24 Benchmarks across 4 Domains

### 4.1 Spatial Intelligence (7 benchmarks)

| Benchmark | ACE-Brain-0 | Gemini-2.5-Pro | GPT-4o | Strongest baseline |
|-----------|------------|----------------|--------|---------------------|
| VSI | **63.3** | 47.8 | 43.6 | Vlaser-8B (60.3) |
| BLINK | **83.9** | 81.8 | 77.9 | InternVL3.5-8B (84.1) |
| SAT | **92.0** | 79.3 | 66.7 | MiMo-Embodied-7B (78.7) |
| MindCube | **82.1** | 57.6 | 46.1 | Vlaser-8B (34.6) |
| Multi3DRef | 59.6 | - | 8.1 | VeBrain-7B (67.8) |

特别值得注意的是**MindCube (82.1% vs Gemini-2.5-Pro 57.6%)**——这个benchmark要求reasoning over unobservable space, 是真正的spatial mental modeling测试。

### 4.2 Autonomous Driving (6 benchmarks)

| Benchmark | ACE-Brain-0 | Gemini-2.5-Pro | GPT-4o | MiMo-Embodied-7B |
|-----------|------------|----------------|--------|-------------------|
| MME-RealWorld | **71.2** | 67.0 | 58.0 | 60.3 |
| MAPLM | **77.8** | 26.1 | 26.6 | 74.5 |
| DriveAction | **81.3** | 73.5 | 72.5 | 81.0 |
| NuPlanQA | **91.7** | - | 81.5 | 73.7 |
| LingoQA | 65.8 | 64.1 | 56.0 | 69.9 |

MAPLM上77.8%远超所有baseline, NuPlanQA上91.7%说明kinematics理解和multi-view integration很强。

### 4.3 Low-Altitude UAV (5 benchmarks)

| Benchmark | ACE-Brain-0 | Qwen-VL-Max | GPT-4o | RoboBrain2.5-8B |
|-----------|------------|-------------|--------|------------------|
| UrbanVideo-Bench | **56.9** | 45.5 | 43.6 | 37.5 |
| AircopBench | **70.3** | 50.5 | 51.8 | 49.9 |
| AVI-Math | **35.0** | - | 33.5 | 26.1 |
| HRVQA | **61.2** | - | 36.9 | 19.2 |

AircopBench (70.3%)特别有意思, 它需要topology-aware spatial relations (crosswalk occupancy, lane-aware ordering)。

### 4.4 Embodied Egocentric (6 benchmarks)

| Benchmark | ACE-Brain-0 | GPT-4o | Qwen3-VL-8B | MiMo-Embodied-7B |
|-----------|------------|--------|--------------|-------------------|
| RoboVQA | **64.6** | 3.3 | 47.0 | 32.8 |
| OpenEQA | 70.0 | 56.4 | 67.1 | 74.1 |
| EmbSpatial | 77.3 | 71.9 | 78.5 | 76.2 |
| EgoPlan-Bench2 | **55.3** | 41.8 | 53.5 | 43.0 |
| EB-Habitat | **42.3** | 59.0 | 27.7 | 16.7 |

RoboVQA上64.6%几乎是所有baseline的2倍以上, 这很impressive。

---

## 5. Ablation Studies: 验证SSR的有效性

### 5.1 Spatial Knowledge作为Shared Scaffold

| Initialization Route | AD Avg. | ∆ | UAV Avg. | ∆ | Embodied Avg. | ∆ |
|----------------------|---------|---|-----------|---|----------------|---|
| Qwen3-VL-8B-Instruct ($\theta$) | 47.0 | - | 37.8 | - | 52.7 | - |
| AD Experts ($\theta \to \theta_A$) | 58.1 | +11.1 | - | - | - | - |
| UAV Experts ($\theta \to \theta_U$) | - | - | 48.8 | +11.0 | - | - |
| Embodied Experts ($\theta \to \theta_E$) | - | - | - | - | 50.8 | **-1.9** |
| **Spatial → AD Expert** ($\theta_S \to \theta_A$) | **72.6** | **+25.6** | - | - | - | - |
| **Spatial → UAV Expert** ($\theta_S \to \theta_U$) | - | - | **54.3** | **+16.5** | - | - |
| **Spatial → Embodied Expert** ($\theta_S \to \theta_E$) | - | - | - | - | **58.1** | **+5.4** |

**关键观察**:
1. 直接从base model训练domain expert, AD和UAV有moderate gains, 但**Embodied expert反而下降1.9%**。paper解释说embodied benchmark需要更fine-grained capability (fine-grained manipulation vs coarse-grained planning), 直接从general domain transfer困难
2. 从spatial scaffold初始化, 三个domain都有大幅提升: **AD +25.6, UAV +16.5, Embodied +5.4**。这是spatial scaffold作为transferable structural prior的强有力证据

### 5.2 Merging Methods对比

| Method | Spatial Avg. | AD Avg. | UAV Avg. |
|--------|--------------|---------|-----------|
| Qwen3-VL-8B-Instruct | 51.6 | 47.0 | 37.8 |
| Spatial ($\theta_S$) | 72.5 | - | - |
| Spatial→AD ($\theta_S \to \theta_A$) | - | 72.6 | - |
| Spatial→UAV ($\theta_S \to \theta_U$) | - | - | 54.3 |
| AVG Merging | 71.6 | 66.6 | 48.0 |
| TSVM Merging | 74.8 | 72.8 | 51.4 |
| **WUDI Merging** | **76.7** | **72.9** | **52.6** |

**关键insight**: WUDI不仅超过base model, 还超过了最强的individual specialist (e.g., 76.7% in Spatial vs 72.5% for spatial-only expert)。这说明merging产生了**super-additive composition effect**——不仅仅是parameter ensembling, 而是真正的knowledge composition。

### 5.3 SSR vs Joint vs Sequential Training

| Paradigm | Spatial Avg. | AD Avg. | UAV Avg. | Embodied Avg. |
|----------|--------------|---------|-----------|----------------|
| Joint Training | 68.0 (-4.5) | 65.3 (-7.3) | 45.7 (-8.6) | 56.8 (-1.3) |
| Sequential Training | 67.6 (-4.9) | 70.1 (-2.5) | 50.8 (-3.5) | 59.0 (+0.9) |
| SSR Training | 78.5 (+6.0) | 72.0 (-0.6) | 50.9 (-3.2) | 59.7 (+1.6) |
| **SSR w/ GRPO** | **79.1 (+6.6)** | **72.1 (-0.5)** | **54.1 (-0.2)** | **60.0 (+1.9)** |

**关键发现**:
1. **Joint training全面失败**: 混合training导致gradient interference, 所有domain都比isolation specialist差
2. **Sequential training有catastrophic forgetting**: 提升最终target domain (Embodied +0.9), 但sacrificing之前学到的capabilities
3. **SSR + GRPO**达到了最佳综合性能, 几乎preserving所有domain的能力同时boost Embodied

---

## 6. Theory: Spatial Scaffold作为Universal Bridge

### 6.1 Mathematical Framework

定义:
- $\mathcal{M} = \{m_1, \ldots, m_K\}$: morphologies集合
- $D_m$: 每个morphology $m$ 诱导的data distribution over tuples $(o, c, y)$
- $R_m(\theta) := \mathbb{E}_{(o,c,y) \sim D_m}[\ell_\theta(o,c,y)]$: morphology $m$ 上的expected risk
- $g_m(\theta) := \nabla_\theta R_m(\theta)$: risk的gradient

Per-sample loss (autoregressive):

$$\ell_\theta(o, c, y) := -\log p_\theta(y \mid o, c) = -\sum_{t=1}^T \log p_\theta(y_t \mid y_{<t}, o, c)$$

### 6.2 Spatial Scaffold Mechanism

**Core假设**: 存在一个shared latent spatial variable $g \in \mathcal{G}$ (capture 3D spatial relations: layout, relative pose, depth, topology), morphology-invariant; 和morphology-specific latent $a_m \in \mathcal{A}_m$ (capture sensor intrinsics, dynamics constraints, actuation semantics):

$$o = \Psi_m(g, a_m), \quad y = \Phi(g, c)$$

其中 $g \sim P_G$, $a_m \sim P_{A|m}$。

**Intuition**: observation channel $o$ 同时依赖geometry $g$ 和morphology-specific factors $a_m$, 但target output $y$ 只依赖geometry $g$ 和conditioning $c$。这意味着只要能从representation中恢复出 $g$, 就能transfer到任何morphology。

### 6.3 Assumption 1 (Recoverable Spatial Scaffold)

让 $\theta_{\mathrm{spatial}}$ 表示Scaffold stage后的参数, induced representation $z_{\mathrm{sp}} := h_{\theta_{\mathrm{spatial}}}(o, c)$。存在decoder $\mathrm{Dec}(\cdot)$ 和constant $\varepsilon_g \geq 0$:

$$\mathbb{E}\big[\|\mathrm{Dec}(z_{\mathrm{sp}}) - g\|\big] \leq \varepsilon_g$$

**这个assumption的实证支持**: Spatial Expert在spatial benchmarks上从baseline 51.6提升到72.5, 表明spatial representation确实可恢复。

### 6.4 Theorem 2 (Scaffold-to-Morphology Transfer Bound)

在Assumptions 1-4下, 对任意target morphology $m$:

$$R_m(\theta_{\mathrm{spatial}}) \leq R_{\mathrm{sp}}(\theta_{\mathrm{spatial}}) + C_m \delta_m + 2 L_g \varepsilon_g + \varepsilon_m$$

变量:
- $R_{\mathrm{sp}}(\theta)$: scaffold分布下的risk
- $C_m$: 比例于 $L_g$ 的constant
- $\delta_m$: geometric distribution shift ($\mathrm{Disc}(P_G^{(m)}, P_G^{(\mathrm{sp})}) \leq \delta_m$)
- $L_g$: Lipschitz constant for loss w.r.t. geometry
- $\varepsilon_g$: spatial scaffold的recoverability error
- $\varepsilon_m$: residual morphology-dependent effects (sensor artifacts, dynamics, actuation semantics等)

**这个bound告诉我们什么**:
1. **$L_g \varepsilon_g$**: 几何recoverability越差, target risk越高; spatial training越好, $\varepsilon_g$ 越小, transfer越好
2. **$C_m \delta_m$**: target morphology和scaffold distribution的geometric mismatch越大, transfer越差——这解释了为什么不同embodiment的transfer performance不对称
3. **$\varepsilon_m$**: 这部分是无法通过geometry transfer的residual, 这正是Stage 3 (Reconcile)和Stage 4 (Embodied SFT)要解决的

### 6.5 Proof intuition

证明通过三角不等式分解target risk:

1. $R_m(\theta_{\mathrm{spatial}})$ 先approximate成 morphology-conditional loss $\mathbb{E}[\ell(g,c;\theta) | g,c]$, 误差 $\varepsilon_m$
2. 用Lipschitz (Assumption 3) 把 $\ell(g,c)$ 和 $\ell(\hat{g},c)$ 联系起来, 误差 $L_g \varepsilon_g$
3. 用discrepancy (Assumption 4) 把target distribution下的期望转到scaffold distribution下的期望, 误差 $C_m^{(2)} \delta_m$
4. 再次用Lipschitz和recoverability做reverse step, 误差 $L_g \varepsilon_g$

---

## 7. Training Datasets

**General**: Cambrain-737K [基于LLaVA-665K augmented with OCR/chart]

**Spatial Intelligence**:
- VSI-590K [Yang et al. CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_and_CVPR_2025_paper.pdf): 590K QA pairs
- SAT [Ray et al. 2024](https://arxiv.org/abs/2412.07755): 175K QA pairs on ProcTHOR-10K
- VICA-322K [Feng 2025](https://arxiv.org/abs/2505.12312): video-based spatial
- GPT4Scene [Qi et al. 2025](https://arxiv.org/abs/2501.01428): 165K with 3D point clouds, BEV images
- Scene-30K [Huang et al. 2025](https://arxiv.org/abs/2507.23478): CoT dataset for 3D VLM reasoning
- VLM-3R [Fan et al. 2025](https://arxiv.org/abs/2505.20279): 包含VSI和VSTI两个子集
- EmbSpatial-Bench [Du et al. ACL 2024](https://aclanthology.org/2024.findings-acl.28/): 25K samples from Matterport3D
- MindCube [Yin et al. ICCV 2025](https://arxiv.org/abs/2508.13142): 10K SFT数据, 强调cross-view consistency
- SpaceR-151K [Ouyang et al. 2025](https://arxiv.org/abs/2504.01805): video spatial reasoning

**Autonomous Driving**:
- MAPLM [Cao et al. CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_MapLM_A_Real-World_Large-Scale_Vision-Language_Benchmark_for_Map_and_Traffic_CVPR_2024_paper.pdf)
- DriveAction [Hao et al. 2025](https://arxiv.org/abs/2506.05667)
- NuScenes-QA [Qian et al. AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28128)
- NuPlanQA [Park et al. 2025](https://arxiv.org/abs/2503.12772)
- LingoQA [Marcu et al. ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73223-0_15)

**Low-Altitude UAV**:
- HRVQA [Li et al. ISPRS 2024](https://www.sciencedirect.com/science/article/pii/S0924271624001861)
- AirSpatial-VQA [Zhou et al. 2025](https://www.sciencedirect.com/science/article/abs/pii/S092427162500xxx)
- Open3DVQA [Zhang et al. ACMMM 2025](https://dl.acm.org/doi/10.1145/3728343.3728xxx)
- AirCopBench [Zha et al. 2025](https://arxiv.org/abs/2511.11025)
- AVI-Math [Zhou et al. ISPRS 2025](https://www.sciencedirect.com/science/article/pii/S0924271625001452)
- CapERA [Bashmal et al. 2023](https://www.mdpi.com/2072-4292/15/8/2139)

**Embodied & Egocentric**:
- MuEP [Li et al. IJCAI 2024](https://www.ijcai.org/proceedings/2024/0017)
- OWMM-VLM Data [Chen et al. 2025](https://arxiv.org/abs/2506.04217)
- Eb-Alfred, Eb-Habitat [Shi et al. 2025; Szot et al. ICLR 2024](https://openreview.net/forum?id=5wpXJhlGiC)
- RoboVQA [Sermanet et al. ICRA 2024](https://ieeexplore.ieee.org/document/10610682)
- Robo2VLM [Chen et al. 2025](https://arxiv.org/abs/2505.15517)
- EgoPlan [Chen et al. 2023](https://arxiv.org/abs/2312.03022)
- EgoCOT [Mu et al. NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/xxx)

---

## 8. Architecture Intuition & 个人思考

### 8.1 为什么SSR优于Joint/Sequential Training

让我从一个更根本的视角理解SSR:

**Joint Training的根本问题** (从Appendix A.2的Theorem 1):
- 当 $\langle g_i(\theta), g_j(\theta) \rangle < 0$ (negative cross-gradient inner product), shared update会降低domain $i$ 的progress
- 这是为什么在Table 8中, joint training在所有domain都underperform specialist (-4.5, -7.3, -8.6, -1.3)

**Sequential Training的根本问题** (catastrophic forgetting):
- 在sequential training中, 后面的stage会overwrite之前学到的capabilities
- Table 8显示, Spatial Avg.从72.5降到67.6 (-4.9), AD Avg.从72.6降到70.1 (-2.5), UAV Avg.从54.3降到50.8 (-3.5)
- 只有最后的Embodied domain有提升 (+0.9)

**SSR的优雅之处**:
1. Spatial scaffold作为stable structural prior, 不会被后续training完全overwrite
2. Isolation training完全消除cross-domain gradient interference
3. Data-free model merging在parameter level synthesize knowledge, 不需要额外数据
4. Super-additive effect (WUDI Merging 76.7% > Spatial-only expert 72.5%)表明merging确实做了某种knowledge composition

### 8.2 关于Model Merging的Theoretical Basis

WUDI的核心思想是: **"Whoever Started the interference Should End It"**——每个task vector $\tau_i$ 是从base model出发的"movement", 在linear regime下, 两个task vector叠加可能导致interference, 因此需要优化merged task vector $\tau_{\mathrm{merge}}$ 让它在每个 $\tau_i$ 的主方向上都尽可能接近 $\tau_i$。

这个approach的优势:
1. **Data-free**: 不需要原始training data
2. **Per-layer optimization**: 在每个layer单独优化, 避免layer间interference
3. **Subspace approximation**: 用task vector近似fine-tuning data的linear subspace, computational efficient

### 8.3 未来的发展方向

paper结尾提到了三个axes:
1. **Spatially-grounded visuomotor policies**: 扩展到VLA model做closed-loop control
2. **Physics-aware continuous prediction**: 从discrete scene understanding到fine-grained physical world modeling
3. **Cross-Embodiment Continual Learning**: 推进SSR到lifelong, interference-free capability accumulation

特别感兴趣的是第3点——如果SSR能扩展到continual learning, 那意味着可以无限添加新embodiment而不忘记旧的。这本质上是把SSR变成一个incremental的knowledge integration framework。

### 8.4 一些可能值得深挖的细节

1. **Square averaging loss weighting**: paper里只说"adopt square averaging", 没有给出公式。这可能是指 $w_i = 1/\sqrt{L}$ 这种normalization? 需要查看code确认
2. **GRPO without KL penalty**: 这与DeepSeek-R1的做法一致, 但为什么会work得这么好? 可能因为group-relative advantage已经implicit地regularize了policy
3. **Spatial scaffold的representation analysis**: 假设1说spatial representation可恢复geometry $g$, 但paper没有做probe实验直接验证这一点。这是future work可以深挖的方向
4. **WUDI merging的1000 iterations是否收敛**: Table 1显示1000 steps, 但没有convergence analysis。是否对不同layer需要不同iterations?

---

## 9. 总结

ACE-Brain-0是一个**principled approach to cross-embodiment learning**, 核心贡献有三:

1. **Identify spatial intelligence as shared scaffold**: 这不是trivial observation, 而是经过严格empirical验证的(AD +25.6, UAV +16.5, Embodied +5.4的提升都来自spatial scaffold)

2. **SSR paradigm decouples shared structure from domain-specific specialization**: 通过Scaffold-Specialize-Reconcile三阶段, 同时解决gradient interference (joint training的问题)和catastrophic forgetting (sequential training的问题)

3. **24 benchmarks验证**: 在Spatial, AD, UAV, Embodied四个domain都达到competitive或SOTA performance, 显著超过general-purpose VLMs和domain-specific embodied brains

最让人impressed的是paper的理论分析(Appendix A.1-A.3): 
- Theorem 1从gradient inner product角度quantify gradient interference
- Theorem 2给出scaffold-to-morphology的transfer bound, $R_m \leq R_{\mathrm{sp}} + C_m \delta_m + 2 L_g \varepsilon_g + \varepsilon_m$

这个bound不仅解释了为什么spatial scaffold有效, 还解释了为什么不同embodiment的transfer performance不对称($\delta_m$不同), 以及为什么Reconcile和Embodied SFT stages是必要的(处理 $\varepsilon_m$)。

**Projects Page**: [https://ace-brain-team.github.io/ACE-Brain-0/](https://ace-brain-team.github.io/ACE-Brain-0/)  
**Code**: [https://github.com/ACE-BRAIN-Team/ACE-Brain-0](https://github.com/ACE-BRAIN-Team/ACE-Brain-0)  
**HuggingFace**: [https://huggingface.co/ACE-Brain/ACE-Brain-0-8B](https://huggingface.co/ACE-Brain/ACE-Brain-0-8B)
