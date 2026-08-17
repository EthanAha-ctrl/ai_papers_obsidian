---
source_pdf: Repurposing Geometric Foundation Models for Multiview Diffusion.pdf
paper_sha256: c9515a19235ffe7b71265a2e93474a0b6804ae839572f8437cc1d818f4d8c8a0
processed_at: '2026-08-11T22:51:34-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GLD 用人话说

## 1. 一句话总结

**以前做3D视角生成, 大家都在用"2D图像专用"的VAE latent space硬塞3D几何信息进去; GLD直接把geometric foundation model (DA3)的内部feature space拿来当diffusion的latent space, 因为这个space本身就编码了3D结构, 模型不用再辛苦学geometry, 训练快4.4倍, 3D consistency大幅提升, 还能免费得到depth和camera pose。**

参考: https://cvlab-kaist.github.io/GLD

---

## 2. 背景故事: 这个领域之前在干什么

### 2.1 Single image generation的经验

Image generation领域已经走过三代latent space:
1. **Pixel space**: 直接在像素上diffusion, 太慢
2. **VAE latent space** [36]: Stable Diffusion的做法, 压缩图像到低维latent, 快很多
3. **Semantic feature space (RAE)** [56]: 用frozen DINOv2的feature当latent, 比VAE还快, 质量更好

RAE的核心发现: **frozen vision encoder的feature space本身就是好的生成空间**, 你只要train一个decoder把feature还原成RGB就行。

### 2.2 NVS (Novel View Synthesis)的困境

NVS要做的task: 给你几张source view, 生成新的target view, 要求多张target view之间geometrically consistent。

**但是**现有的NVS方法(CAT3D [12], MVGenMaster [4], Matrix3D [30])全都用VAE latent space, 这个space压根没编码3D信息。怎么办? 它们的解决方案是:
- 从外部inject geometry condition (depth map, warped RGB)
- 或者用T2I pretraining的strong prior来implicitly learn geometry

这就像你想做中餐, 但是厨房里只有西餐的锅具, 你得各种workaround。GLD的insight是: **为什么不直接用中餐的锅?**

### 2.3 Geometric Foundation Model的崛起

最近一年, DA3 [26], VGGT [47]这类geometric foundation model出现了。它们能feed-forward地从任意多张unposed image重建3D geometry, 预测depth和camera pose。内部架构关键点是**3D attention**: 让不同view的tokens能互相通信, 建立cross-view correspondence。

那这些model的internal feature space是什么样? Probe3D [10]等分析显示, 这些feature已经strongly encode了cross-view geometric correspondence。GLD的idea: **这个feature space就是NVS梦寐以求的latent space**。

---

## 3. GLD的核心思路: 三步走

### Step 1: 验证DA3 feature能重建高保真RGB

这是必要前提。如果DA3 feature连RGB都还原不了, 那当latent space也没用。作者train了一个ViT decoder $\mathcal{D}_{\text{rgb}}$, 用$\ell_1$ + LPIPS + GAN loss, 还加了level-wise dropout trick (训练时随机mask掉某些level, 强制robustness)。

**结果** (Table 1 & Table 7):
| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| VAE (Stable Diffusion) [36] | 34.53 | 0.939 | 0.028 |
| VAE (SDXL) [33] | 34.97 | 0.945 | 0.029 |
| RAE (DINO) [56] | 26.78 | 0.830 | 0.148 |
| **DA3 Decoder (Ours)** | **35.41** | **0.960** | **0.019** |

DA3 feature的reconstruction fidelity比SDXL VAE还好, 这一步通过。

### Step 2: 决定diffuse到哪个level

DA3有4个level的intermediate features ($l \in \{0,1,2,3\}$), 全部diffuse太贵。关键观察: **deeper level可以由shallower level通过frozen DA3 encoder层propagation得到**, 不用explicitly generate。

所以只需要选一个boundary level $k$, explicitly generate到$k$, 更深的level自动propagate。

**实验** (Table 2): 测试不同boundary
- $k=0$: 只生成Level 0 → PSNR 12.55
- **$k=1$: 生成Level 0和1 → PSNR 13.61 (最优)**
- $k=2$: 生成0,1,2 → PSNR 13.35 (反而下降)
- $k=3$: 全部生成 → PSNR 13.35

### Step 3: Cascaded生成Level 0

确定了boundary $k=1$后, Level 0怎么生成? 独立生成会导致Level 0和Level 1不一致。所以用cascaded: $\mathcal{M}_{1\to 0}$以生成的Level 1为condition生成Level 0。

训练trick: 给condition加noise (用noisy版GT $\mathbf{F}_1$), 让模型对inference时的imperfect latent更robust。

**结果** (Table 8):
| Method | PSNR↑ | ATE↓ |
|--------|-------|------|
| Independent | 18.81 | 0.197 |
| **Cascaded** | **19.00** | **0.182** |

---

## 4. 为什么Level 1是sweet spot? (核心intuition)

这个分析非常elegant。作者从两个维度评估每个level:

### 4.1 Geometric Correspondence (Table 5)

用PCK (Percentage of Correct Keypoints)在ScanNet上测量cross-view matching能力:
| Feature | $F_0$ | $F_1$ | $F_2$ | $F_3$ | DINOv2 |
|---------|-------|-------|-------|-------|--------|
| PCK ↑ | 22.25 | 35.98 | **40.70** | 20.98 | 31.64 |

- Level 2几何对应最强
- Level 0太浅, 还没建立几何理解
- DA3的Level 2甚至超过DINOv2

### 4.2 Photometric Reconstruction (Table 6)

用decoder单独从每个level重建RGB:
| Feature | $F_0$ | $F_1$ | $F_2$ | $F_3$ |
|---------|-------|-------|-------|-------|
| PSNR ↑ | **28.01** | 25.36 | 14.01 | 10.19 |
| LPIPS ↓ | 0.138 | 0.138 | 0.491 | 0.768 |

- Level 0/1保留丰富appearance (颜色、纹理)
- Level 2/3丢弃photometric detail, 颜色流失, 纹理平滑

### 4.3 Sweet Spot直觉

NVS需要两个能力:
1. **Cross-view consistency**: 需要geometry understanding
2. **High-fidelity RGB**: 需要photometric detail

| Level | Geometry | Photometry | 适合NVS? |
|-------|----------|------------|----------|
| 0 | 弱 (PCK 22) | 强 (PSNR 28) | ✗ 缺geometry |
| **1** | **中强 (PCK 36)** | **中强 (PSNR 25)** | **✓ Sweet spot** |
| 2 | 强 (PCK 41) | 弱 (PSNR 14) | ✗ 缺photometry |
| 3 | 弱 (PCK 21) | 极弱 (PSNR 10) | ✗ 都不行 |

Level 1刚好两头都够用。这就像Goldilocks principle: 不深不浅, 刚刚好。

---

## 5. 架构细节

### 5.1 Multi-view Diffusion Model (Fig. 8A)

用$\text{DiT}^{\text{DH}}$ [48]架构, decoupled设计:

**Condition Encoder**:
- Hidden dim $C_1 = 768$
- 28个DiT blocks
- 处理source-view condition和cross-view context

**Velocity Decoder**:
- Hidden dim $C_2 = 2048$
- Level-wise model $\mathcal{M}_l$: 6个blocks
- Cascaded model $\mathcal{M}_{1\to 0}$: 2个blocks (更轻量)
- 预测velocity field $\mathbf{u}_{t,l} \in \mathbb{R}^{V \times T \times C}$

输入是noisy latent $\mathbf{z}_t^l$和source-only condition $\mathbf{F}_l^{\text{src}}$的channel-wise concat。

### 5.2 关键技术细节

- **3D self-attention**: 替换standard self-attention, 让V个view的tokens互相attend
- **PRoPE** [25]: Positional Rotation Encoding, 编码camera geometry
- **Plücker coordinates**: 6D per-pixel ray embedding + 1D source/target indicator = 7D, linear projection到hidden dim
- **CFG**: 10% dropout on camera embeddings, inference CFG scale 1.5
- **Patch size**: DA3 latents用1 (保留token resolution), VAE latents用2 (因为空间分辨率更高)

### 5.3 为什么Source和Target要jointly generate?

这是很精妙的设计。DA3的3D attention会让source和target views的feature耦合 - 如果你只generate target feature, 它和source feature在encoder里就没法正常interact。

所以GLD同时生成source和target的features:
$$\tilde{\mathbf{F}}_l = [\tilde{\mathbf{F}}_l^{\text{src}}, \tilde{\mathbf{F}}_l^{\text{tgt}}] \in \mathbb{R}^{(N+M) \times T \times C}$$

虽然source views其实有GT, 但为了保持DA3 internal的coupling, 必须jointly generate。

---

## 6. 实验结果: 亮点解析

### 6.1 2D Quality (Table 3, Re10K)

| Method | From Scratch | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:---:|-------|-------|--------|
| MVGenMaster [4] | ✗ | 15.226 | 0.588 | 0.456 |
| Matrix3D [30] | ✗ | 14.490 | 0.580 | 0.448 |
| CAT3D$^\dagger$ [12] | ✗ | 13.350 | 0.527 | 0.561 |
| DINO [32] | ✓ | 15.638 | 0.601 | 0.448 |
| VAE [36] | ✓ | 15.656 | 0.606 | 0.456 |
| **GLD (Ours)** | **✓** | **16.362** | **0.630** | **0.431** |

**震撼点**: GLD从scratch训练, 超过了所有fine-tuned from T2I pretrained的方法。MVGenMaster、Matrix3D都用了Stable Diffusion级别的pretraining, 结果还不如GLD。

### 6.2 3D Consistency (Table 3, Re10K)

| Method | ATE↓ | RPE$_r$↓ | RPE$_t$↓ | Reproj↓ | MEt3R↓ |
|--------|------|----------|----------|---------|--------|
| MVGenMaster | 0.282 | 6.42 | 0.526 | 0.664 | 0.339 |
| Matrix3D | 0.413 | 8.93 | 0.638 | 0.666 | 0.344 |
| DINO | 0.345 | 15.59 | 0.719 | 0.721 | 0.319 |
| VAE | 0.278 | 8.68 | 0.552 | 0.681 | 0.375 |
| **GLD** | **0.211** | **7.07** | **0.444** | **0.673** | **0.328** |

- ATE (Absolute Trajectory Error): GLD 0.211, 比VAE baseline低2.8×
- RPE (Relative Pose Error): GLD比VAE低2.6×

**这个gap说明**: DA3 latent space让model自然生成geometrically consistent的view, 因为latent本身就编码了geometry。

### 6.3 Sparse View Advantage (Table 4)

| Setting | Method | PSNR↑ | ATE↓ |
|---------|--------|-------|------|
| N=1 (DL3DV) | DINO | 12.80 | 0.743 |
| N=1 | VAE | 12.47 | 0.880 |
| N=1 | **GLD** | 13.03 | **0.237** (3× better) |
| N=4 (DL3DV) | DINO | 15.44 | 0.274 |
| N=4 | VAE | 16.37 | 0.294 |
| N=4 | **GLD** | 17.09 | **0.143** (2× better) |

**Intuition**: Source view越少, visual cue越稀缺, geometric prior的相对价值越大。N=1时GLD的ATE比baseline低3×, N=4时低2×。这就像: 信息充足时大家都做得不错, 信息匮乏时strong prior的优势就显现出来。

### 6.4 4.4× Faster Convergence

Fig. 1(c)显示GLD比VAE/DINO快4.4×收敛。直觉: VAE/DINO latent没有geometric prior, model要从头learn cross-view correspondence; DA3 latent已经encoding了这种correspondence, model只需要learn generate, 不用learn geometry。

### 6.5 Zero-shot Geometry (§5.7, Table 9)

这是GLD的另一个bonus: 因为在DA3 latent space生成, 直接用frozen DA3 decoder就能预测depth和camera pose, 不需要额外训练。

**Depth Evaluation** (ETH3D):
| Method | AbsRel↓ | SqRel↓ | $\delta_1$↑ | RGB PSNR↑ |
|--------|---------|--------|-----------|-----------|
| Matrix3D [30] | 0.197 | 0.475 | 0.731 | 14.13 |
| **GLD (Ours)** | **0.160** | **0.410** | **0.800** | 14.80 |

GLD的depth比Matrix3D更准, RGB质量也更好。这是**geometry generation as free byproduct**。

### 6.6 VGGT Backbone验证 (Table 13, Appendix C.1)

为了证明方法不依赖DA3特定, 用VGGT [47]替代:

| Method | PSNR↑ | ATE↓ |
|--------|-------|------|
| DINO | 15.64 | 0.345 |
| VAE | 15.66 | 0.278 |
| GLD w/ VGGT | 16.17 | 0.216 |
| **GLD w/ DA3** | **16.36** | **0.211** |

VGGT版本也比VAE/DINO好很多, 证明**core hypothesis与具体backbone无关**。

---

## 7. 深度分析: 为什么GLD work?

### 7.1 Internal Correspondence Analysis (Appendix D.1)

作者follow CAMEO [22]的protocol, 测量diffusion model内部3D attention map的cross-view correspondence (Fig. 14):

**Findings**:
1. GLD (DA3 latent)在几乎所有layer都展现最强correspondence, 特别在decoder blocks中margin最大
2. Correspondence在condition encoder中几乎absent, 只在velocity decoder中emerge, peak在layers 31-32

**解释**:
- Condition encoder主要preserve per-view conditioning信息
- Cross-view geometric correspondence是在decoder通过3D attention建立的
- DA3 latent提供了更好的initialization, 让model更容易learn这种correspondence

这呼应CAMEO的核心发现: **更强的internal correspondence → 更高质量的multi-view generation**。

### 7.2 计算成本 (Appendix D.2, Table 15)

| Method | Sampling (s) | Decode (s) | Total |
|--------|--------------|------------|-------|
| DINO | 0.41 | 35.2 | ~35.6 |
| VAE | 0.50 | 28.0 | ~28.5 |
| GLD | - | 66.1 | 66.8 |

GLD更慢, 因为两阶段sampling。但breakdown显示propagation极快:

| Phase | Time (s) |
|-------|----------|
| Lv.1 sampling | 37.8 |
| Lv.1→Lv.2,3 propagation | **0.15** |
| Lv.1→Lv.0 sampling | 28.4 |
| RGB decoding | 0.43 |

Propagation只需0.15s, 几乎free, 验证了boundary design的efficiency。

---

## 8. 核心Insight总结

### 8.1 范式对比

| Paradigm | Latent Space | Geometry来源 | 需要T2I pretraining? |
|----------|-------------|--------------|---------------------|
| CAT3D, MVGenMaster | VAE | External (depth warping) | ✓ |
| RAE (single image) | DINO/SigLIP | None | - |
| **GLD** | **DA3 features** | **Intrinsic (latent本身)** | **✗** |

### 8.2 为什么这个idea之前没人做?

- Geometric foundation model (DA3, VGGT)都是最近才出现
- 大家习惯了VAE latent space, 没意识到task-specific latent space的重要性
- 从discriminative feature到generative latent的repurposing需要验证step

### 8.3 Limitations (Appendix D.3, Fig. 15)

- Severe occlusion时会hallucinate
- Extreme lighting变化时cross-view correspondence难建立
- Inference比VAE慢~2.4×
- Object-centric generalization略弱 (只在scene-level data训练)

---

## 9. 对未来的启示

1. **Task-specific latent space design**: 不一定非要VAE, 根据task选择合适的latent space可能更重要
2. **Foundation model reuse**: Discriminative model的feature可以repurpose为generative latent
3. **From-scratch can beat T2I pretraining**: 只要有right inductive bias, 不需要T2I prior也能达到SOTA
4. **Geometry-aware generation新范式**: 把geometry内置到representation中, 而不是external conditioning

---

## 10. 相关References

- **GLD Project**: https://cvlab-kaist.github.io/GLD
- **Depth Anything 3**: https://arxiv.org/abs/2511.10647
- **VGGT**: https://arxiv.org/abs/2403.05122
- **RAE (Representation Autoencoders)**: https://arxiv.org/abs/2510.11690
- **DiT**: https://arxiv.org/abs/2212.09748
- **CAT3D**: https://arxiv.org/abs/2405.10314
- **CAMEO**: https://arxiv.org/abs/2512.03045
- **DUSt3R**: https://arxiv.org/abs/2312.14132
- **Probe3D**: https://arxiv.org/abs/2312.16208
- **PRoPE (Plücker ray embeddings)**: https://arxiv.org/abs/2506.13785
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **Latent Diffusion (Stable Diffusion)**: https://arxiv.org/abs/2112.10752
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Mip-NeRF 360**: https://arxiv.org/abs/2111.12027
- **RealEstate10K**: https://arxiv.org/abs/1805.09817

---

**最后一句话总结**: GLD告诉我们, 对于geometry-aware task, 用geometry-aware latent space是natural choice; 与其费劲从外部inject geometry, 不如直接让geometry存在于representation本身。这个简单insight让from-scratch训练的model超越了用T2I pretraining的庞然大物, 在3D consistency上还大幅领先。Paper写得很清晰, 实验solid, 是latent space design领域的范式转移之作。

---

# GLD: Repurposing Geometric Foundation Models for Multi-view Diffusion

这篇paper来自KAIST AI、NYU和Intel Labs的团队, 核心idea非常优雅且重要。让我深入讲解。

## 1. 核心Intuition: 为什么这个问题重要?

当前NVS (Novel View Synthesis) 领域有一个根本性矛盾：

- **Single-image generation领域**已经证明latent space的选择至关重要(RAE [56]显示DINO feature比VAE latent更好)
- **NVS任务**本质上是geometry-aware generation, 需要cross-view consistency
- 但是现有NVS方法(MVGenMaster [4], CAT3D [12], Matrix3D [30])依然用view-independent的VAE [36] latent space

这就好比你要做3D重建, 但你的representation space压根没有编码3D信息, 模型只能从头learn geometry。这篇paper的insight在于: **geometric foundation model (DA3 [26], VGGT [47])的内部feature space已经天然编码了cross-view geometric correspondence**, 我们直接复用它作为diffusion的latent space。

参考链接：
- DA3 paper: https://arxiv.org/abs/2511.10647
- VGGT: https://arxiv.org/abs/2403.05122
- RAE: https://arxiv.org/abs/2510.11690
- Project page: https://cvlab-kaist.github.io/GLD

---

## 2. Preliminaries: 技术背景

### 2.1 Representation Autoencoder (RAE)

RAE [56]的核心idea是：用frozen vision encoder替代VAE的encoder, 只train一个decoder。给定单张图像 $\mathcal{T} \in \mathbb{R}^{H \times W \times 3}$：

$$\mathbf{F} = \mathcal{E}(\mathcal{T}) \in \mathbb{R}^{T \times C}$$

其中：
- $\mathcal{E}(\cdot)$: frozen encoder (如DINOv2)
- $T$: token sequence length
- $C$: channel dimension

生成时: $\tilde{\mathcal{T}} = \mathcal{D}(\tilde{\mathbf{F}})$, decoder从synthesized feature恢复RGB。

### 2.2 Geometric Foundation Models

DA3/VGGT这类model的架构是ViT encoder + DPT decoder。关键在于它有**3D attention**层处理multi-view输入：

$$\{\mathbf{F}_l\}_{l=0}^{L-1} = \mathcal{E}_{\text{geo}}(\mathbf{I})$$

其中：
- $\mathbf{I} \in \mathbb{R}^{V \times H \times W \times 3}$: $V$个views
- $\mathbf{F}_l \in \mathbb{R}^{V \times T \times C}$: level $l$的multi-view features, $L=4$
- 3D attention让不同view的tokens能交互, 编码geometric correspondence

几何预测：
$$\mathbf{G} = \mathcal{D}_{\text{geo}}(\{\mathbf{F}_l\}_{l=0}^{L-1})$$

$\mathbf{G}$包含depth maps和camera poses。

---

## 3. GLD Framework: 核心方法

### 3.1 整体架构 (Fig. 2)

GLD是一个三阶段pipeline：

```
Stage 1: Multi-view Diffusion (synthesize features up to boundary k=1)
  ↓ cascaded generation
Stage 2: Feature Propagation (derive deeper features through frozen DA3 encoder)
  ↓
Stage 3: RGB Decoding (map multi-level features to target views)
```

给定：
- $N$张source images $\mathbf{I}^{\text{src}} \in \mathbb{R}^{N \times H \times W \times 3}$
- Source poses $\mathbf{P}^{\text{src}}$
- $M$个target poses $\mathbf{P}^{\text{tgt}}$

生成target views $\tilde{\mathbf{I}}^{\text{tgt}} \in \mathbb{R}^{M \times H \times W \times 3}$。

**关键设计**: 因为DA3的3D attention会jointly encode source和target views, 所以feature是coupled的。GLD同时生成source和target的features:

$$\tilde{\mathbf{F}}_l = [\tilde{\mathbf{F}}_l^{\text{src}}, \tilde{\mathbf{F}}_l^{\text{tgt}}] \in \mathbb{R}^{(N+M) \times T \times C}$$

这里 $[\cdot, \cdot]$ 是view dimension的concatenation。

最终解码:
$$\tilde{\mathbf{I}}^{\text{tgt}} = \mathcal{D}_{\text{rgb}}(\{\tilde{\mathbf{F}}_l^{\text{tgt}}\}_{l=0}^{L-1})$$
$$\tilde{\mathbf{G}}^{\text{tgt}} = \mathcal{D}_{\text{geo}}(\{\tilde{\mathbf{F}}_l^{\text{tgt}}\}_{l=0}^{L-1})$$

注意第二个公式: **几何预测是free的byproduct**, 因为我们在DA3 feature space中生成, 直接用frozen DA3 decoder就能得到depth和camera poses, 不需要额外训练。

### 3.2 Validating Reconstruction Capability (§4.2)

在用DA3 feature做diffusion之前, 必须验证它能重建高保真RGB。作者train了一个ViT-based decoder $\mathcal{D}_{\text{rgb}}$, 关键trick是**level-wise dropout**: 训练时随机mask掉某些level, 强制decoder从partial input重建, 提高robustness。

**Table 1结果** (Re10K test set, 4000 samples):
| Metric | $\mathcal{D}_{\text{rgb}}$ |
|--------|---------------------------|
| PSNR ↑ | 35.41 |
| SSIM ↑ | 0.960 |
| LPIPS ↓ | 0.019 |

对比Table 7的baseline:
- VAE (Stable Diffusion) [36]: PSNR 34.53
- VAE (SDXL) [33]: PSNR 34.97  
- RAE (DINO) [56]: PSNR 26.78
- **DA3 Dec. (Ours): PSNR 35.41**

DA3 feature的reconstruction fidelity甚至超过了SDXL VAE, 这证明了geometric feature space足够expressive来支持高质量image synthesis。

### 3.3 Boundary Layer Selection (§4.3): 关键技术决策

这是paper中最精妙的部分。DA3有4个level的features ($l \in \{0,1,2,3\}$), 全部diffuse计算代价太高。作者观察到: **deeper features可以由shallower feature通过frozen encoder层propagation得到**, 所以只需要explicitly synthesize到某个boundary level $k$。

**实验设计**: 训练4个独立的diffusion model $\{\mathcal{M}_l\}_{l=0}^{3}$, 测试不同boundary $k \in \{0,1,2,3\}$:

**Table 2结果**:
| Boundary $k$ | Models | PSNR↑ | SSIM↑ | LPIPS↓ | AbsRel↓ | RMSE↓ | $\delta<1.25$↑ |
|---|---|---|---|---|---|---|---|
| $k=0$ | $\{\mathcal{M}_0\}$ | 12.55 | 0.323 | 0.579 | 0.267 | 0.400 | 0.641 |
| **$k=1$** | $\{\mathcal{M}_0, \mathcal{M}_1\}$ | **13.61** | **0.366** | **0.555** | **0.191** | **0.311** | **0.744** |
| $k=2$ | $\{\mathcal{M}_0,\mathcal{M}_1,\mathcal{M}_2\}$ | 13.35 | 0.355 | 0.566 | 0.254 | 0.393 | 0.659 |
| $k=3$ | all | 13.35 | 0.355 | 0.567 | 0.260 | 0.402 | 0.647 |

**$k=1$最优**, 这个结果很有意思 - 更深的level反而更差。作者在§5.5给出了非常清晰的分析。

### 3.4 为什么Level 1最优? (§5.5) - Build Intuition

作者从两个角度分析每个level:

**Geometric Correspondence** (Table 5, ScanNet上用PCK评估):
| Feature | $F_0$ | $F_1$ | $F_2$ | $F_3$ | DINOv2 |
|---------|-------|-------|-------|-------|--------|
| PCK ↑ | 22.25 | 35.98 | **40.70** | 20.98 | 31.64 |

- Level 2/3有强geometric correspondence (Level 2最高)
- Level 0太浅, 缺少cross-view matching能力
- **DA3的Level 2甚至超过了DINOv2**, 这很impressive

**Photometric Reconstruction** (Table 6):
| Feature | $F_0$ | $F_1$ | $F_2$ | $F_3$ |
|---------|-------|-------|-------|-------|
| PSNR ↑ | **28.01** | 25.36 | 14.01 | 10.19 |
| LPIPS ↓ | 0.138 | 0.138 | 0.491 | 0.768 |

- Level 0/1保留了rich appearance cues (颜色、纹理)
- Level 2/3丢弃了photometric details, 导致color loss和smoothed textures

**Intuition**: NVS需要同时满足两个要求:
1. Cross-view geometric consistency → 需要足够的geometric correspondence
2. High-fidelity RGB → 需要足够的photometric information

Level 1恰好是**sweet spot**: 它的geometric correspondence接近Level 2 (35.98 vs 40.70), 但photometric information远超deeper levels (PSNR 25.36 vs 14.01)。Level 0 photometric好但geometry弱, Level 2 geometry强但photometric差。

### 3.5 Cascaded Feature Generation (§4.4)

确定了boundary $k=1$后, 还需要生成Level 0。作者比较了两种方案:

**Independent**: $\mathcal{M}_0$和$\mathcal{M}_1$分别独立生成
**Cascaded**: 用$\mathcal{M}_{1\to 0}$, 以生成的Level 1 feature作为condition生成Level 0

**Table 8结果** (Re10K, N=4 source views):
| Method | PSNR↑ | SSIM↑ | LPIPS↓ | ATE↓ | RPE$_r$↓ | RPE$_t$↓ | Reproj↓ | MEt3R↓ |
|--------|-------|-------|--------|------|----------|----------|---------|--------|
| Independent | 18.81 | 0.692 | 0.335 | 0.197 | 7.179 | 0.430 | 0.666 | 0.335 |
| **Cascaded** | **19.00** | **0.695** | **0.327** | **0.182** | **6.694** | **0.397** | **0.652** | **0.326** |

Cascaded在所有metrics上都更好。关键trick: 训练时给$\mathcal{M}_{1\to 0}$的condition加noise (用noisy版本的GT $\mathbf{F}_1$), 这样inference时对imperfect latent更robust, 同时保证$\tilde{\mathbf{F}}_0$ anchored to $\tilde{\mathbf{F}}_1$。

---

## 4. 架构细节 (Appendix A.2)

### 4.1 Multi-view Diffusion Model

采用$\text{DiT}^{\text{DH}}$ [48]架构, decoupled成两部分:

**Condition Encoder** (Table 10):
- Hidden dim: $C_1 = 768$
- 28个DiT blocks
- 16 attention heads
- 处理source-target context

**Velocity Decoder**:
- Hidden dim: $C_2 = 2048$
- $\mathcal{M}_l$ (level-wise): 6个DiT blocks
- $\mathcal{M}_{1\to 0}$ (cascaded): 2个DiT blocks (更shallow)
- 预测velocity field $\mathbf{u}_{t,l} \in \mathbb{R}^{V \times T \times C}$

输入: noisy latent $\mathbf{z}_t^l$ 与source-only condition $\mathbf{F}_l^{\text{src}}$ channel-wise concat ($2C = 3072$)。

**关键设计细节**:
- Standard self-attention被替换为**3D self-attention** [12]实现cross-view interaction
- **PRoPE** [25] (Positional Rotation Encoding) 编码camera geometry
- Camera conditioning: per-pixel 6D **Plücker coordinates** + 1D source/target indicator → 7D embedding → linear projection
- DA3 latents用patch size 1, VAE latents用patch size 2 (因为VAE空间分辨率更高)
- CFG: 10% dropout on camera embeddings, CFG scale 1.5

### 4.2 RGB Decoder

ViT-based, patch size 14, 12 transformer layers, intermediate dim 3072, dropout 0.5。Loss是 $\ell_1$ + LPIPS + GAN的加权和。

---

## 5. 实验结果分析

### 5.1 主实验 (Table 3)

**2D Metrics** (Re10K):
| Method | From Scratch | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:---:|-------|-------|--------|
| MVGenMaster [4] | ✗ | 15.226 | 0.588 | 0.456 |
| Matrix3D [30] | ✗ | 14.490 | 0.580 | 0.448 |
| CAT3D$^\dagger$ [12] | ✗ | 13.350 | 0.527 | 0.561 |
| DINO [32] | ✓ | 15.638 | 0.601 | 0.448 |
| VAE [36] | ✓ | 15.656 | 0.606 | 0.456 |
| **GLD (Ours)** | **✓** | **16.362** | **0.630** | **0.431** |

**GLD从头训练, 超过了所有fine-tuned from T2I pretrained的方法!** 这非常impressive, 因为MVGenMaster、Matrix3D等都用了大规模text-to-image pretraining。

**3D Metrics** (Re10K, 更重要):
| Method | ATE↓ | RPE$_r$↓ | RPE$_t$↓ | Reproj↓ | MEt3R↓ |
|--------|------|----------|----------|---------|--------|
| MVGenMaster | 0.282 | 6.42 | 0.526 | 0.664 | 0.339 |
| Matrix3D | 0.413 | 8.93 | 0.638 | 0.666 | 0.344 |
| DINO | 0.345 | 15.59 | 0.719 | 0.721 | 0.319 |
| VAE | 0.278 | 8.68 | 0.552 | 0.681 | 0.375 |
| **GLD** | **0.211** | **7.07** | **0.444** | **0.673** | **0.328** |

GLD在ATE上达到0.211, 比VAE baseline降低2.8×, RPE降低2.6×。这证明了geometric latent space的几何consistency优势。

### 5.2 4.4× Faster Convergence

这是paper标题claim的核心优势之一。Fig. 1(c)显示GLD比VAE和DINO收敛快4.4×。直觉解释: VAE/DINO latent space没有geometric prior, diffusion model需要从头learn cross-view correspondence; 而DA3 latent space已经encoding了这种correspondence, model只需要learn如何generate, 不用learn如何建立geometry。

### 5.3 Sparse View Setting (Table 4)

作者测试了N=1和N=4 source views的情况。最interesting的发现:

**GLD在sparse view下优势更大**:
- DL3DV N=1: GLD的ATE比baseline低3×+
- DL3DV N=4: GLD的ATE比baseline低约2×

**Intuition**: Source view越少, visual cue越稀疏, geometric prior的价值就越大。DA3 feature space的geometric structure在信息匮乏时提供了关键grounding。

### 5.4 Geometry Evaluation (§5.7)

**Depth Evaluation** (Table 9, ETH3D):
| Method | AbsRel↓ | SqRel↓ | $\delta_1$↑ | RGB PSNR↑ |
|--------|---------|--------|-----------|-----------|
| Matrix3D [30] | 0.197 | 0.475 | 0.731 | 14.13 |
| **GLD (Ours)** | **0.160** | **0.410** | **0.800** | 14.80 |

GLD的depth accuracy超过Matrix3D, 且RGB质量也更好。这是因为在DA3 latent space中生成, 直接用frozen DA3 decoder就能得到geometrically consistent的depth - 这是**zero-shot geometry prediction as byproduct**。

### 5.5 Geometric Correspondence Analysis (Appendix D.1)

这是非常深刻的分析。作者follow CAMEO [22]的protocol, 测量diffusion model内部3D attention map的cross-view correspondence:

**Finding**: GLD (DA3 latent)在几乎所有layer都展现出最强的cross-view correspondence, 特别是在decoder blocks中margin最大。

**更深刻的观察**: Correspondence在condition encoder中几乎不存在, 只在velocity decoder中emerge并peak at intermediate blocks (layers 31-32)。这说明:
- Condition encoder主要preserve per-view conditioning信息
- Cross-view geometric correspondence是在decoder中通过3D attention建立的

这呼应了CAMEO的发现: **更强的internal correspondence → 更高质量的multi-view generation**。DA3 latent space提供了更好的initialization, 让diffusion model更容易learn这种correspondence。

参考CAMEO: https://arxiv.org/abs/2512.03045

### 5.6 VGGT Backbone验证 (Appendix C.1)

为了证明方法的generality, 作者用VGGT [47]替代DA3作为backbone:

**Table 13** (Re10K):
| Method | PSNR↑ | ATE↓ | RPE$_r$↓ |
|--------|-------|------|----------|
| DINO | 15.64 | 0.345 | 15.59 |
| VAE | 15.66 | 0.278 | 8.68 |
| GLD w/ VGGT | 16.17 | 0.216 | 7.17 |
| **GLD w/ DA3** | **16.36** | **0.211** | **7.07** |

VGGT版本也比VAE/DINO好很多, 证明**core hypothesis (geometric latent space更好) 与具体backbone无关**。DA3略好于VGGT, 但都显著优于non-geometric baselines。

---

## 6. Computational Cost (Appendix D.2)

**Table 15**: Inference latency对比
| Method | Sampling (s) | Decode (s) |
|--------|--------------|------------|
| DINO | 0.41 | 35.2 |
| VAE | 0.50 | 28.0 |
| GLD (ours) | - | 66.1 |

GLD更慢, 因为需要两个sampling stages。但breakdown显示propagation非常快:

**GLD Runtime Breakdown**:
| Phase | Time (s) |
|-------|----------|
| Lv.1 sampling | 37.8 |
| Lv.1 → Lv.2, Lv.3 propagation | 0.15 |
| Lv.1 → Lv.0 sampling | 28.4 |
| RGB decoding | 0.43 |
| **Total** | **66.8** |

Lv.1→Lv.2,3 propagation只需0.15s, 远快于independent generation, 验证了boundary design的efficiency。

---

## 7. 关键Insights总结

### 7.1 为什么这个方法work?

1. **Geometric prior is free**: DA3已经用海量数据训练过, feature space天然编码3D structure
2. **3D attention的coupling效应**: Source和target views在DA3内部通过3D attention交互, 生成时自然保持consistency
3. **Sweet spot level**: Level 1平衡了geometry (PCK 35.98) 和photometry (PSNR 25.36)
4. **Zero-shot geometry byproduct**: 生成feature后直接用DA3 decoder得到depth/pose, 无需额外监督

### 7.2 与现有范式的对比

| Paradigm | Latent Space | Geometry Source |
|----------|-------------|-----------------|
| 传统NVS (CAT3D, MVGenMaster) | VAE | External (depth warping) |
| RAE (single image) | DINO/SigLIP | None |
| **GLD (Ours)** | **DA3 features** | **Intrinsic (latent space本身)** |

GLD的范式转换在于: **不再从外部注入geometry, 而是让geometry存在于latent space中**。

### 7.3 对未来的启示

1. **Task-specific latent space design**: Paper最后呼吁"encourages further investigation into task-specific latent space design for geometry-aware generation"
2. **Foundation model reuse**: Geometric foundation model的discriminative feature可以repurpose为generative latent space
3. **From-scratch can compete with T2I pretraining**: 只要有right inductive bias, 不需要T2I prior也能达到SOTA

---

## 8. 可能的Limitations和Future Work

1. **Inference speed**: GLD比VAE慢~2.4×, 可以探索distillation或fewer sampling steps
2. **Register/camera tokens**: 作者在appendix提到omitted了这些tokens, 可能还有提升空间
3. **Failure cases** (Fig. 15): Severe occlusion和extreme lighting变化时仍会hallucinate
4. **Object-centric generalization**: GLD只在scene-level data上训练, 在Mip-NeRF 360 (object-centric)上略逊于fine-tuned方法

---

## 9. 相关Reference Links

- **Project page**: https://cvlab-kaist.github.io/GLD
- **Depth Anything 3**: https://arxiv.org/abs/2511.10647
- **VGGT**: https://arxiv.org/abs/2403.05122
- **RAE (Representation Autoencoders)**: https://arxiv.org/abs/2510.11690
- **DiT (Diffusion Transformers)**: https://arxiv.org/abs/2212.09748
- **CAT3D**: https://arxiv.org/abs/2405.10314
- **CAMEO**: https://arxiv.org/abs/2512.03045
- **DUSt3R**: https://arxiv.org/abs/2312.14132
- **Probe3D**: https://arxiv.org/abs/2312.16208
- **Plücker coordinates for cameras (PRoPE)**: https://arxiv.org/abs/2506.13785
- **Flow Matching**: https://arxiv.org/abs/2210.02747

---

这篇paper的beautiful之处在于它发现了一个简单但深刻的truth: **对于geometry-aware generation task, 用geometry-aware latent space是natural choice**。之前大家都忙着把geometry作为condition从外部注入, 而GLD直接让geometry存在于representation本身。这个paradigm shift让from-scratch训练的model能compete with T2I-pretrained behemoths, 在3D consistency上甚至大幅超越。
