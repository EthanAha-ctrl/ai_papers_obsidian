---
source_pdf: Beyond Language Modeling.pdf
paper_sha256: 440b629770a36bd5b9d02894a622f2f780af80ead2c0fc1aa4677b8873b5bdae
processed_at: '2026-07-18T16:46:16-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Beyond Language Modeling 深度解读

Andrej，这篇 paper 来自 FAIR/Meta 和 NYU 团队（Shengbang Tong、Saining Xie、Yann LeCun 等），核心目标是为 native multimodal pretraining 的 design space 提供 empirical clarity。我把它拆开来跟你聊聊。

Project page: https://beyond-llms.github.io/

## 1. 论文的核心定位

这篇文章不是又一个 "multimodal LLM" 论文，它的关键 motivation 是 **"from-scratch, controlled experiments"**。绝大多数现有多模态工作（Janus、BAGEL、Show-o、Emu3）都从 pretrained LLM 初始化，这会 confound 任何关于 multimodal training 本身的结论。作者用 Plato's Cave 的比喻开场——language 只是 reality 的 lossy compression，shadow 而非物体本身。Sutskever 在 2025 No Priors 也提到 text data 接近 exhaustion。

四个核心发现：
- **S1**: RAE (Representation Autoencoder) 单一 encoder 同时搞定 understanding 和 generation
- **S2**: Visual 和 language 数据 complementary，甚至 synergy
- **S3**: World modeling capability 从 general multimodal pretraining 自然 emerge
- **S4**: MoE 在 multimodal 场景下高效 scaling 并自然产生 modality specialization

---

## 2. 方法论：Transfusion + Flow Matching

### 2.1 训练目标

框架基于 Transfusion (Zhou et al., 2025a)，但对架构做了关键简化（去掉 U-Net，用 linear projection）。

**Language modeling loss**（标准 next-token cross-entropy）:

$$
\mathcal{L}_{\mathrm{LM}} = -\sum_{i=1}^{n} \log p_{\theta}(x_i \mid x_{<i})
$$

- $x_i$: 第 $i$ 个 text token
- $x_{<i}$: 前 $i-1$ 个 token 的 context
- $p_{\theta}$: 参数 $\theta$ 的 transformer 预测分布

**Flow matching loss**（image-wise）:

$$
\mathcal{L}_{\mathrm{flow}} = \mathbb{E}_{t, z_0, \epsilon}\Big[\| v_{\theta}(z_t, t, \cdot) - (z_0 - \epsilon) \|_2^2\Big]
$$

- $z_0$: clean latents for 一张 image 或 video frame（flatten 成 sequence）
- $\epsilon \sim \mathcal{N}(0, I)$: 标准 Gaussian noise
- $t \sim \mathcal{U}[0,1]$: flow matching 的时间步
- $z_t = (1-t)\epsilon + t z_0$: interpolated latent（从 noise 到 clean 的线性路径）
- $v_{\theta}(z_t, t, \text{context})$: model 预测的 velocity field
- 目标 $z_0 - \epsilon$ 就是直线方向

关键设计点：**每张 image/frame 采样一个独立 t**，整个 image 的所有 token 共享同一个 $t$。这跟 Diffusion Forcing (Chen et al., 2024a) 思路一致，让 training dynamics 更稳定。

**Joint loss**:

$$
\mathcal{L} = \lambda_{\mathrm{LM}}\mathcal{L}_{\mathrm{LM}} + \lambda_{\mathrm{flow}}\mathcal{L}_{\mathrm{flow}}
$$

默认 $\lambda_{\mathrm{LM}}=1.0$, $\lambda_{\mathrm{flow}}=3.0$（Transfusion 原文用 6.0，作者调低了不少）。

### 2.2 Hybrid attention masking

这是我觉得很有意思的一个设计点。用 FlexAttention 实现：
- **Text tokens**: standard causal mask
- **Visual tokens**: block-wise causal mask —— 同一帧内的 patches 之间 bidirectional，跨帧 causal

直觉上：单张 image 本身没有时序，bidirectional 更合理；但 video 序列必须保持 causal 以维持自回归性质。这个设计让任意 modality 组合都能在一个 sequence 里处理。

### 2.3 Inference 的 modal-switching

这个细节很关键。生成时：
1. Text mode 下正常 autoregressive 采样
2. 一旦采到 `<BOI>` token，暂停 text 生成
3. Append 一组 pure noise tokens（比如 256 个，对应 16×16 grid）
4. 用 25-step Euler sampler 做 flow matching denoise
5. Denoise 完成后 append `<EOI>`，继续 text 生成

这是一个真正 unified 的 token stream，vision 和 language 在同一个 sequence 里交替。

---

## 3. S1: RAE 统一 visual representation

### 3.1 历史问题

历史上：
- **Understanding**（VQA）喜欢用 semantic encoder：SigLIP、DINOv2
- **Generation** 偏好 VAE：SD-VAE、FLUX.1 VAE（低维 latent）

所以 Janus / BAGEL 用 **dual encoder**（SigLIP 做 understanding + VAE 做 generation），架构复杂。

### 3.2 RAE 的核心 insight

RAE (Representation Autoencoder, Zheng et al., 2026) 的发现：diffusion 可以在 **high-dimensional latent space** 里有效工作。这打破了"必须用 VAE 压到低维"的假设。

为什么 RAE work？我个人的 intuition：
- VAE 强制低维 bottleneck，会让 latent 失去 semantic richness
- Semantic encoder（如 SigLIP）的 feature 已经是高维、信息密集的
- Diffusion 不需要 latent 空间满足 VAE 的 KL 约束
- RAE decoder 学着把 semantic latent 还原到 pixel

实验数据（Figure 4）很有说服力：

| Encoder | Type | DPGBench | GenEval | VQA avg | Text PPL |
|---------|------|----------|---------|---------|----------|
| Raw Pixel | - | 0.40 | 0.40 | 0.55 | 14.9 |
| SD-VAE | VAE | 0.42 | 0.43 | 0.50 | 15.0 |
| FLUX.1 | VAE | 0.50 | 0.46 | 0.55 | 15.0 |
| DINOv2-L | Semantic | 0.48 | 0.45 | 0.60 | 15.0 |
| WebSSL-L | Semantic | 0.50 | 0.45 | 0.60 | 15.0 |
| **SigLIP 2** | **Semantic** | **0.57** | **0.50** | **0.62** | 15.0 |

SigLIP 2 在所有 vision 指标上碾压 VAE，**包括 generation**。这个结果其实挑战了领域里的传统认知。

### 3.3 细节 caveat

作者也诚实地提到：SigLIP 2 在 fine-grained pixel reconstruction 上仍不如 VAE（Figure 28a 显示 PSNR 从 layer 0 的 29.6dB 降到 layer 26 的 20.9dB）。这是 semantic 和 spatial fidelity 的 trade-off。但 RAE 的 decoder 学着弥补这个差距。

我猜未来的方向：**generation-aware semantic encoder** —— 一个在训练时就考虑 reconstruction loss 的 semantic encoder。

参考链接:
- RAE paper (ICLR 2026): https://openreview.net/forum?id=rAEhCBxpsj
- Tong et al. scaling T2I with RAE: https://arxiv.org/abs/2506.11871

---

## 4. S2: Data Synergy

### 4.1 Vision 不损害 language

这个发现很反直觉。常见 assumption 是"加 vision data 会 tax language"。实验显示（Figure 5）：

| Data Mixture | DCLM PPL | Notes PPL |
|--------------|----------|-----------|
| Text-only (520B) | baseline | baseline |
| Text + Video | **略好于** text-only | 略好 |
| Text + MetaCLIP | 略差 | 略差 |
| Text + Video + MetaCLIP + Action | 接近 text-only | 略差 |

注意 **Text + Video 比 text-only 还要好一点点**。这说明纯视频（无 caption）对 language modeling 是 neutral-positive 的。

### 4.2 I/T 数据的 distribution shift

MetaCLIP caption 与 DCLM 的 cosine distance（Table 1）：
- MetaCLIP: 0.196
- SSTK (Shutterstock): 0.215
- MetaCLIP Recaption (用 Qwen2.5-VL 重写): 0.286

Recaption 距离更大，所以 PPL 退化更严重。这印证了 author 的 hypothesis：**"modality tax" 的来源不是 vision 本身，而是 caption 分布的偏移**。

### 4.3 Decoupling data by objective

一个非常 actionable 的发现：不同 I/T 来源有不同的优势：
- **MetaCLIP** (web-crawled)：好 for I2T (understanding)
- **SSTK** (high-aesthetic)：好 for T2I (generation)
- **Recaption**：好 for VQA

最终策略：**decouple data by objective**。MetaCLIP 用于 image-to-text task，SSTK 用于 text-to-image task。

### 4.4 Cross-modal synergy

Figure 8 的实验最 striking：固定 vision budget，增加 text budget → generation quality (GenEval) **持续提升**。

Figure 9：20B VQA + 80B heterogeneous data（text / video / MetaCLIP 都试过）**全部超过** 100B VQA-only。这意味着：

> **"In-domain scaling" < "diverse pretraining"**

即使 unlabeled video（看似与 VQA 无关）也能提升 VQA performance。这一点跟你之前在 tweet 里讲的 "data diversity > data quantity" 一致。

参考链接:
- DCLM: https://www.datacomp.ai/dclm/
- MetaCLIP: https://ai.facebook.com/blog/meta-clip/

---

## 5. S3: World Modeling Emerges

### 5.1 NWM setup

作者采用 Navigation World Model (Bar et al., 2025) 的设置，但做了一个关键简化：**action 直接写成 text tokens**（比如 "0.5,0,0.3" 这样的数字串），不再用专门的 continuous action adapter。

任务格式：`State_t + Action_text → State_{t+1}`，即 `I + T → I` prediction。

训练数据：SCAND、RECON 等 egocentric robot navigation 数据。

### 5.2 两个关键发现

**Finding 1**: World modeling 依赖 general multimodal pretraining，不是 domain-specific data。

Figure 12 的对照实验（50B NWM + 50B multimodal vs 100B NWM-only）：

| Extra data | ATE | RPE |
|------------|-----|-----|
| 100B NWM only | baseline | baseline |
| + Text | ↓ | ↓ |
| + MetaCLIP | ↓↓ | ↓↓ |
| + Video | ↓↓↓ | ↓↓↓ |

纯 video 贡献最大。

**Finding 2**: 只需要 1% in-domain NWM data，性能就 saturate（Figure 13）。

这点跟 Hafner et al. 2025 的发现类似。我个人 intuition 是：world modeling 的核心能力（视觉一致性、物理常识、几何变换）已经在 video pretraining 中学会了，NWM-specific 数据只负责"format alignment"——告诉模型 "你现在的任务是预测下一帧，输入是这种格式"。

### 5.3 Free-form language as actions

Figure 14 的 qualitative 结果很 striking。给定 4 个 context frames，model 可以根据：
- WASD 控制信号（数值）
- **自然语言 prompt**："get out of the shadow!" "go on the road" "take big steps forward"

都能生成合理的下一帧。后者完全 OOD（训练时 NWM 用的是数值 action）。

这意味着：**language-conditioned navigation 是 zero-shot emergence**。这正是 LeCun 在 "A Path Towards Autonomous Machine Intelligence" 里描述的 vision。

参考链接:
- NWM paper: https://arxiv.org/abs/2401.06037
- LeCun position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 6. S4: MoE 设计

### 6.1 Modality-specific FFN 的基础

Figure 3 是 baseline：把 shared FFN 换成 modality-specific FFN（text-only FFN + vision-only FFN），unanimously 提升。注意这只增加参数不增加 FLOPs（每个 token 只激活一个 FFN）。

这跟 LMFusion (Shi et al., 2024) 和 MoMa (Lin et al., 2024) 的发现一致。

### 6.2 MoE Granularity

定义 granularity:

$$
G = \frac{4 d_{\mathrm{model}}}{d_{\mathrm{expert}}}
$$

- $d_{\mathrm{model}}$: transformer 的 hidden dimension（这里 2048）
- $d_{\mathrm{expert}}$: 单个 expert 的 hidden dimension
- $4 \times$ 是因为标准 FFN 有 4x expansion

G=1 → 16 个大 expert（d=8192），Top-1 routing
G=64 → 1024 个小 expert（d=128），Top-64 routing

实验结果（Figure 15）：
- G 从 1 → 16: 显著提升所有指标
- Vision 在 G=4 饱和
- Language 在 G=16 饱和
- 最终选 G=16

我的 intuition：language token 的"语义模式"更分散，需要更细粒度的 expert 分工；vision token 的 latent 已经很 compact，少量 expert 就够。

### 6.3 x-pred vs v-pred

非常 subtle 的发现：
- **RAE (SigLIP 2)**: x-prediction > v-prediction（在 generation 上）
- **VAE (FLUX.1)**: v-prediction > x-prediction

x-prediction 直接预测 $z_0$，v-prediction 预测 velocity $z_0 - \epsilon$。

直觉：x-pred 依赖 manifold assumption（$z_0$ 在一个低维流形上）。当 expert dimension 太小（G 大时），低维 VAE latent 可能突破不了 rank 瓶颈，x-pred 会 spike。Semantic latent 已经在高维流形上，反而能 leverage x-pred 的 simplicity。

参考：JiT (Li & He, 2025) https://arxiv.org/abs/2511.13720

### 6.4 Sparsity scaling

Figure 16 是 MoE 的核心 evidence。固定 active compute（16 active experts），把 total expert count 从 32 涨到 1008（active ratio 从 50% 降到 1.6%），**两个 modality 的 loss 都持续下降**。

| Total experts | Total params | Text PPL ↓ | Diff Loss ↓ | GenEval ↑ |
|---------------|--------------|-----------|-------------|-----------|
| 32 | 2.3B | 15.0 | 0.50 | 0.36 |
| 256 | 13.5B | 14.8 | 0.48 | 0.36 |
| 1008 | 51.5B | 14.7 | 0.47 | 0.37 |

注意 vision 在 VAE 下 saturate（Figure 17），在 RAE 下持续改善——又一个支持 RAE 的证据。

### 6.5 Per-modality shared expert

Table 2 比较：

| Configuration | DCLM PPL | GenEval |
|---------------|----------|---------|
| No shared | 14.802 | 0.360 |
| Global shared (1 expert always active) | 14.794 | 0.364 |
| **Per-modality shared** (1 text + 1 vision) | **14.785** | **0.367** |

每个 modality 有一个 always-active 的 "shared expert"。这跟 DeepSeekMoE 的 shared expert 思路类似，但分成 modality-specific。

参考: DeepSeekMoE https://arxiv.org/abs/2401.06066

---

## 7. Emergent Specialization

这是 paper 最 intellectually satisfying 的部分。

### 7.1 Modality specialization 自然形成

定义 Specialization Score:

$$
S_i = \frac{R_{\mathrm{text}}^{(i)} - R_{\mathrm{image}}^{(i)}}{R_{\mathrm{text}}^{(i)} + R_{\mathrm{image}}^{(i)}}
$$

- $R_m^{(i)} = C_m^{(i)} / (N_m \times k)$: expert $i$ 对 modality $m$ 的 selection rate
- $C_m^{(i)}$: expert $i$ 被 modality $m$ 的 token 选中次数
- $N_m$: modality $m$ 的总 token 数
- $k$: 每个 token 激活 expert 数

$S_i \in [-1, +1]$。+1 = 纯 text expert，-1 = 纯 vision expert。

实验观察（Figure 18）：
1. **Text expert 比 vision expert 多很多**，即使有 load-balancing loss 鼓励均匀使用
2. 早期层 text-heavy，后期层 vision/multimodal 增加
3. 形成 "separate-then-integrate" 的处理策略

这跟 Section 7 的 scaling law 完美吻合：**language 是 parameter-hungry，vision 是 data-hungry**。模型自动学到了这个不对称。

### 7.2 Timestep 不专业化

非常反直觉。假设：vision expert 可能按 diffusion timestep 分工（high-noise expert vs fine-grained expert）。

测量：Coefficient of Variation (CV) of expert selection rates across 10 timestep bins。结果 CV ≈ 0.15，几乎是均匀的。

我的解读：diffusion 的 timestep 是一个**正交维度**，跟"语义处理"是无关的。同一个 expert 在 t=0.1 和 t=0.9 做的事情本质相似——都是"给定 noisy latent，预测 velocity"。模型不需要为不同 t 准备不同 expert。

这跟 Wan (Wan et al., 2025) 显式分离 timestep expert 的做法形成对比。可能 Wan 是为了 efficiency，不代表 capability 上必要。

### 7.3 Generation 和 understanding 共享 expert

Figure 20：image-to-text (understanding) 和 text-to-image (generation) 的 expert selection rate 的 Pearson correlation ≥ 0.90 across all layers。

这意味着：**同一个 vision expert 处理理解任务的 image token 和生成任务的 noisy latent**。

这是 unified representation 的深层证据。模型自己学到了 "visual concept" 的统一表征，无论它是用来 caption 还是 denoise。

参考: Emergent Modality Specialization 相关研究 https://arxiv.org/abs/2510.08220

---

## 8. Scaling Laws: The Asymmetry

### 8.1 Dense models 的不对称

IsoFLOP 方法：在不同 compute budget 下，sweep model size N 和 token count D，拟合最优 $N_{\mathrm{opt}}(C)$ 和 $D_{\mathrm{opt}}(C)$。

Power law: $N_{\mathrm{opt}} \propto C^a$, $D_{\mathrm{opt}} \propto C^b$, 其中 $a + b = 1$。

| Modality | a (param) | b (data) | 含义 |
|----------|-----------|----------|------|
| Language | 0.47 | 0.53 | Chinchilla-like, balanced |
| Vision | 0.37 | 0.63 | **Data-hungry** |

Vision 的 data exponent 比 language 高 0.10。换成"数据/参数比"：

$$
D_{\mathrm{opt}} \propto N_{\mathrm{opt}}^{b/a}
$$

Vision 的 $b/a = 0.63/0.37 = 1.70$，Language 的 $b/a = 0.53/0.47 = 1.13$。

**vision 需要的 data/param ratio 随 model size 增长更快**。从 1B 到 100B：vision 相对 language 的 data 需求增加 14×；到 1T：增加 51×。

这是一个**致命的不对称**。意味着：
- 如果按 Chinchilla 分配 compute（balanced），vision 会被 under-train
- 如果按 vision 最优分配，language 会被严重 over-train（浪费 compute）

实际上现在大 LLM 都 overtrain（LLaMA-3 系列就是），但这只解决了 language 一边。

### 8.2 MoE 调和 asymmetry

MoE 的 sparsity ratio = 16（total experts / active experts = 16）下：

| Modality | a | b |
|----------|---|---|
| Language | 0.41 | 0.59 |
| Vision | 0.36 | 0.64 |

Gap 从 0.10 缩小到 0.05。

关键 insight：**MoE 让 language 变得更 data-hungry**（b 从 0.53 涨到 0.59），向 vision 靠拢。这是因为 MoE 提供了更多 capacity，让 language 可以吸收更多 data；vision 本来就是 data-bound，cap 在 encoder 上。

Figure 26 显示 MoE Multimodal 在 $10^{21}$ FLOPs 几乎匹配 unimodal baselines：
- DCLM PPL: 12.3 (multimodal) vs 12.0 (text-only)
- FID: 39.2 (multimodal) vs 39.8 (T2I-only)

**这就是 "harmonize scaling asymmetry" 的含义**。

### 8.3 我的解读

我觉得这是 paper 最深刻的发现之一。MoE 不只是 efficiency trick，它是 **multimodal coexistence 的结构性解决方案**。

想象一个 dense multimodal model 在 1T 参数：
- Language 那部分 "想要" 大约 0.5T 参数 × Chinchilla-optimal data
- Vision 那部分 "想要" 大约 0.5T 参数 × 14× Chinchilla-optimal data
- 两者不能同时满足

MoE 把"模型容量"和"激活计算"解耦。可以给 language 分配 80% 的 expert pool（parameter-hungry 的需求），给 vision 分配 20%（data-hungry 不需要那么多参数，需要数据）。同时 active compute 还是平衡的。

这跟 Sutton 的 "Bitter Lesson" 完美呼应：**不要手动设计 modality 分工，让学习决定**。

参考链接:
- Chinchilla: https://arxiv.org/abs/2203.15556
- Sutton Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

## 9. Stacking Design Choices (Section 6.3)

Figure 21 是一个 incremental ablation，从 Transfusion baseline 开始逐步叠加：

| Step | Modification | PPL ↓ | DPG ↑ |
|------|--------------|-------|-------|
| 1 | Transfusion baseline | 15.93 | 0.45 |
| 2 | + Modality-specific FFN | 15.13 | 0.47 |
| 3 | + SigLIP 2 (replacing VAE) | 15.06 | 0.57 |
| 4 | + MoE (replacing dense) | 12.49 | 0.63 |
| 5 | + x-pred (replacing v-pred) | - | 0.65 |

每一步都有显著提升，验证 design choices 互补。

Figure 22 的 WISE benchmark（测试 knowledge-informed generation）：
- Semantic encoder 比 VAE-based 高 **3-4×** across all knowledge categories
- MoE > MoT > dense（在固定 encoder 下）

这印证了 unified model 的核心 promise：**language 的 world knowledge 通过 unified training 真正 transfer 到 generation**。

---

## 10. Loss Centering (Appendix D.2)

这是一个被低估的小创新。固定 $\lambda_{LM}, \lambda_{flow}$ 在不同 encoder 下表现不一致，因为不同 encoder 的 loss magnitude 不同。

Loss centering 方案：

$$
c_{\mathrm{current}} = \alpha \mathcal{L}_{\mathrm{flow}} + (1-\alpha)\mathcal{L}_{\mathrm{LM}}
$$

- $\alpha \in [0,1]$: 控制对 vision vs text 的 emphasis

EMA update:

$$
c_{\mathrm{target}} \leftarrow \mu c_{\mathrm{target}} + (1-\mu) c_{\mathrm{current}}
$$

- $\mu$: momentum

Per-step weight:

$$
w_{\mathrm{flow}} = \frac{c_{\mathrm{target}}}{\mathcal{L}_{\mathrm{flow}}}, \quad w_{\mathrm{LM}} = \frac{c_{\mathrm{target}}}{\mathcal{L}_{\mathrm{LM}}}
$$

直觉：让两个 loss 的 effective magnitude 都等于 $c_{\mathrm{target}}$，自动平衡。

实验（Table 3）显示 DPG score 全面提升，代价是 PPL 略升。

这个 idea 跟 DINO 的 centering 机制同源，很 elegant。

---

## 11. 整体评价与联想

### 11.1 这篇 paper 的真正贡献

我读下来觉得这篇 paper 的价值在于**它系统地、controlled 地把 multimodal pretraining 的 design space 扫了一遍**。不是 SOTA chasing，是 science。

四个 finding 串起来形成一个 coherent story：

> Vision 和 language 是 complementary 的（S2），用一个统一 representation（S1）和合适的 architecture（MoE，S4），可以让两者在同一个 model 里 coexist，scaling 出来还 natural emerge world modeling（S3）。

### 11.2 跟你（Karpathy）的某些观点的关联

你在不同场合讲过几个观点，这篇 paper 都印证了：

1. **"Data diversity > data quantity"** —— Figure 9 的 20B VQA + 80B diverse > 100B VQA-only
2. **"Bitter lesson 的延伸"** —— MoE > 手工 MoT 分离
3. **"Raw pixels 是未来"** —— Figure 4 显示 raw pixel 在 VQA 上 competitive，作者明确说这是 promising direction

### 11.3 跟 JEPA / LeCun vision 的关联

LeCun 一直在推 JEPA (Joint Embedding Predictive Architecture) 作为 world model 的正确框架。这篇 paper 在某种意义上是 "Transfusion + RAE" 路线下对 world modeling 的探索。

但区别在于：JEPA 在 latent space 做 predictive，不 reconstruct pixel；这篇 paper 还是 reconstruct（通过 RAE decoder）。两路是否 converge 是开放问题。

参考:
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeCun 2022: https://openreview.net/pdf?id=BZ5a1r-kVsf

### 11.4 我对 future work 的猜想

1. **Interleaved data**：作者在 Limitations 里明确说没做 interleaved data。这是 web 的天然格式，可能带来更多 synergy。
2. **Multimodal RL**：post-training 阶段让 model 同时 generate 和 interpret visual latents，这是 closed-loop agent 的关键。
3. **Generation-aware semantic encoder**：现在的 SigLIP 是 understanding-tuned 的。如果能训练一个 encoder 同时 optimize semantic + reconstruction，RAE 路线会更完整。
4. **Video native encoder**：现在 video 是 frame-by-frame encode 的，没有 temporal pooling。一个真正 native 的 video encoder（带时序维度）会进一步释放 world modeling 潜力。
5. **Audio**：作者完全没提 audio。Audio 也是 first-class modality，跟 video 高度 complementary。

### 11.5 一些 caveat

- **Compute budget 有限**：最大到 ~$10^{21}$ FLOPs，1T tokens 级别。比真正 frontier model 小 2-3 个数量级。一些 trend 是否 extrapolate 还需验证。
- **VQA evaluation 用了 finetune**：1 epoch Cambrian-7M。所以 multimodal pretraining 的好处部分通过 finetune 体现。
- **Reconstruction quality 没完全解决**：SigLIP + RAE decoder 在 fine-grained pixel 上还是不如 VAE，作者承认。

---

## 12. 总结

这是一篇"科学性 > SOTA"的 paper。它告诉我们的核心 takeaway：

> **Multimodal pretraining 不需要"妥协"，需要"正确的架构 + 正确的 representation"。Vision 和 language 不是 trade-off，是 synergy。MoE 是让两者 coexist 的结构性解决方案。World modeling 不是单独训练出来的，是 unified pretraining 自然 emerge 的。**

跟你一直讲的 "let the gradient flow" 哲学一致：手动设计的 modality separation（dual encoder、MoT、fixed modality-specific expert）都会被 learned routing 击败。

希望对你 build intuition 有帮助！如果想深入聊某个具体 section，比如 flow matching 的数学细节、MoE routing 算法、或者 scaling law 的拟合方法，我可以继续展开。

相关参考链接汇总:
- Project page: https://beyond-llms.github.io/
- Transfusion: https://arxiv.org/abs/2408.11039
- RAE: https://arxiv.org/abs/2410.06944
- SigLIP 2: https://arxiv.org/abs/2502.14786
- Cambrian-1: https://arxiv.org/abs/2406.16860
- Chinchilla: https://arxiv.org/abs/2203.15556
- Plato's Republic (Project Gutenberg): https://www.gutenberg.org/ebooks/1497
- Sutton Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
