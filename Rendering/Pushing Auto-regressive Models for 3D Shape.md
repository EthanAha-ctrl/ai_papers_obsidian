---
source_pdf: Pushing Auto-regressive Models for 3D Shape.pdf
paper_sha256: 88dbdd8024c5cbda326fd94a5a63c629eccbde09708cb918114e16e394d50aba
processed_at: '2026-08-06T07:15:46-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Argus3D 人话版

## 一句话总结

把 3D shape 变成一串"抽象单词"，然后用 GPT 的方式一个一个"写"出来。

---

## 为什么 3D 生成这么难搞

2D 图像生成已经玩得很溜了：把图切成 16×16 的 patch，每个 patch 编码成一个 code，transformer 学 patch 之间的 correlation。patch 有自然的 spatial order（从左到右从上到下），AR 模型知道怎么 predict 下一个。

3D 就尴尬了。最直觉的做法是把 3D space 切成 voxel grid（比如 32×32×32），每个 voxel 编码成 code。但有两个噩梦：

**噩梦 1：太多了。** 32³ = 32768 个 tokens，64³ = 262144 个 tokens。transformer 的 attention 是 O(n²) 复杂度，32768 个 token 的 attention matrix 已经爆显存了。2D 图像 256×256=65536 个 patch 都嫌多，3D 更夸张。

**噩梦 2：顺序不自然。** 2D 图像从左到右从上到下是 human convention，虽然 arbitrary 但至少 consistent。3D voxel 你按什么顺序 flatten？x-y-z？y-x-z？z-x-y？没有哪个是"自然的"。

更糟的是，3D voxel 之间高度 coupled。一个椅子的腿和座位是 spatially connected 的，但你按 x-y-z flatten 之后，腿和座位的 token 在序列里可能隔得很远，transformer 学起来很痛苦。你换个 flatten order，结果就变。这就是 paper 里 Fig. 3 和 Fig. 17 说的 "ambiguity"。

之前的方法怎么处理这个问题：
- **ShapeFormer**：假装没看见，用 sparse representation + row-major order，硬训
- **AutoSDF**：搞个 non-sequential transformer，让部分 token 同时 predict，用架构 trick 绕过顺序问题

Argus3D 的思路完全不同：**与其在 transformer 里 hack，不如在 representation 阶段就把问题解决掉**。

---

## Argus3D 的核心 idea

### Step 1：把 3D 压成 2D（Tri-plane）

3D voxel O(r³) 太大，那就投影到 3 个正交平面（xy, yz, xz），变成 3 个 2D feature map，复杂度 O(r²)。这是 [EG3D](https://nvlabs.github.io/eg3d/) 那套 tri-plane 表示，neural rendering 里很成熟。

但 tri-plane 还是有 spatial structure，每个位置还是对应 3D space 的某个投影点，flatten 成 1D 还是 ambiguous。

### Step 2：把 2D "卷"成 1D 抽象 token（Coupling Network，核心创新）

这是 paper 的灵魂操作。把 3 个 plane concatenate，然后过 4 层 conv with down-sampling：

```
256×256×96 → 128×128×64 → 64×64×128 → 32×32×256
```

每 down-sample 一次，receptive field 翻倍，每个 output element "看到"的 3D region 越来越大。最后 flatten 成 32×32 = 1024 个 element。

**关键 insight**：经过这些 conv，output element 不再对应 3D space 的任何固定位置了。它变成一个 abstract "concept"——比如"这里有个椅子腿的形状"或者"这里有个曲面的弧度"，但具体是哪个位置，已经 encode 在 conv 的 weight 里了。

这就像 NLP 里的 word token。"chair" 这个 token 没有 spatial position，但它在 sentence 里有 semantic role。Transformer 学的是 token 之间的 semantic correlation，不需要 spatial position 信息。

Argus3D 把 3D shape 也变成这样的 abstract token sequence，transformer 就可以 vanilla 地学。

### Step 3：VQ 量化 + Transformer 生成

1024 个 element 每个 quantize 成 codebook 里的某个 entry，得到 1024 个 discrete indices。这就是 3D shape 的 "sentence"。

然后 vanilla decoder-only transformer，跟 GPT 一模一样的结构，学 next-token prediction。条件生成就是把 condition feature prepend 到序列前面。

---

## 用类比再讲一遍

想象你要教一个 AI 画 3D 模型。

**传统方法**：给 AI 一个 32×32×32 的乐高积木块，让它一块一块地决定放什么颜色。问题是有 32768 块，而且它不知道先放哪块后放哪块，因为 3D 空间没有自然的"读写顺序"。

**Argus3D 的方法**：先训练一个"3D 形状翻译官"，把任何 3D 模型翻译成一段 1024 个词的描述。这些词不是描述具体位置，而是描述"形状的特征"——比如"弯曲的表面"、"尖锐的边缘"、"圆柱形的腿"。

翻译官怎么训练的？通过 reconstruction：把 3D shape → 1024 个词 → 再重建回 3D shape，要求重建出来的和原来一样。这样翻译官被迫学会用 1024 个词 capture 所有几何信息。

然后让 GPT 学写这些"形状描述文章"。给它几十万篇这样的文章（每个 3D shape 一篇），它学会 "弯曲表面" 后面常跟 "圆柱腿"，"椅子座位" 下面常跟 "四条腿" 这种 correlation。

生成新 shape 时，GPT 写一篇 1024 词的"形状描述"，再让翻译官反向翻译回 3D shape。

**条件生成**就是给 GPT 一个 prompt（图片/文字/部分点云的描述），让它续写完整的形状描述。

---

## 为什么能 scale 到 3.6B 参数

因为 transformer 保持 vanilla。没有 non-sequential mask、没有 spatial-aware attention、没有 custom module。就是 GPT-3 的架构，直接复制 GPT-3 的 scale 配置（768→2048→3072 dim，12→24→32 layers）就行。

AutoSDF 的 non-sequential transformer 很难 scale，因为 custom 结构在更大规模下的 behavior 不可预测。Argus3D 没这个问题，GPT-3 已经证明这个架构 scale 到 175B 都 OK，3.6B 是小 case。

实验结果（Table 8, 9）：
- Base 100M → Large 1.2B → Huge 3.6B，所有指标 monotonically 提升
- Huge model + Objaverse-Mix pre-training，FPD 从 1.680 降到 0.774
- Fig. 14 检查 nearest neighbor，证明不是 memorization，是真的学了 distribution

---

## 数据：Objaverse-Mix

ShapeNet 51K 太小，大模型一训就 overfit。Objaverse 800K 很好但 noisy，没 annotation。

Argus3D 的做法：5 个 dataset 拼一起（ModelNet40 + ShapeNet + Pix3D + 3D-Future + Objaverse），~900K shapes。关键是清洗：
- 去掉 non-watertight mesh
- 去掉 weird shape 和 complex scene
- 用 Blender 渲染 12 个固定视角
- 用 BLIP-2 给 front-view 生成 caption

清洗成本：4 台机器 × 8 A100 × 4 周 × 100TB 存储。3D 数据预处理是真贵。

---

## 实验亮点

**Unconditional**（Table 2）：ECD 从 baseline 的 858 降到 240，1-NNA 从 71 降到 61，SOTA。

**Class-guide**（Table 3）：所有指标超过 AutoSDF 和 GBIF。

**Completion**（Table 4, 5）：TMD 和 MMD 都最好，证明 multi-modal completion 能力——给半个椅子能 imagine 出多种完整椅子。

**Image-guide**（Table 6）：用 CLIP image feature，FPD 1.680 vs CLIP-Forge 8.094，大幅领先。

**Text-guide**（Table 7）：Acc 59.93 vs CLIP-Forge 53.68。

**Zero-shot**（Fig. 21, 22）：用 CLIP text feature 替换 image feature，模型只见过 image-shape pair，但能做 text-to-shape。

**DALL·E 2 图片**（Fig. 15）：用 DALL·E 2 生成 avocado 图片，Argus3D 能生成 avocado shape，虽然训练集没见过 avocado。

---

## Ablation 的核心验证

**Flattening order 的影响**（Table 10, Fig. 16）：
- 只用 tri-plane（没 coupling network）：不同 flatten order 结果差异巨大，standard deviation 大
- 加 coupling network：row-major 和 col-major 效果几乎一样

这直接证明了 coupling network 解决了 ambiguity 问题。

**Memory cost**（Fig. 18）：
- Grid 32³：≥80GB memory，训不动
- Vector 32×32：3.8GB memory，IoU 一样

---

## Intuition 总结

Argus3D 的哲学：**让 representation 适配 model，而不是让 model 适配 representation**。

传统 3D AR 方法试图改造 transformer 去处理 3D 的 spatial structure，结果模型复杂、难 scale。Argus3D 在 representation 阶段就把 spatial structure "卷"掉，让 transformer 看到的就是一串 abstract tokens，跟 NLP 没区别。

这跟 NLP 的 tokenization 哲学一致：word token 没有 spatial position，只有 semantic meaning，但 transformer 能学到 ordering prior。Argus3D 把 3D shape 也变成这样的 tokens。

这个 insight 可以推广到任何 complex structured data 想用 large transformer 的场景：**先设计一个 spatially-decoupled、tractable-order 的 discrete representation，然后 transformer 可以保持 vanilla form**。这是 scaling-friendly design 的核心。

3.6B 参数 + 900K shapes 是 3D generation 的 GPT-3 moment。继续 scale 下去，加 texture、animation、interaction，可能就是通往 3D foundation model 的路径。

---

## 参考

- [Project page](https://argus-3d.github.io)
- [Dataset](https://huggingface.co/datasets/BAAI/Objaverse-MIX)
- [ImAM (predecessor)](https://arxiv.org/abs/2303.14700)
- [AutoSDF (baseline)](https://arxiv.org/abs/2204.02275)
- [ShapeFormer (baseline)](https://arxiv.org/abs/2201.10333)
- [VQ-VAE (基础)](https://arxiv.org/abs/1711.00937)
- [Taming Transformers (2D 对应工作)](https://compvis.github.io/taming-transformers/)
- [EG3D (tri-plane 来源)](https://nvlabs.github.io/eg3d/)
- [GPT-3 (scaling 参考)](https://arxiv.org/abs/2005.14165)
- [Objaverse (数据基础)](https://objaverse.allenai.org/)

---

# Argus3D: 推进 Auto-regressive Models 用于 3D Shape Generation

## 一、Paper 核心思想：为什么需要 Argus3D

这篇 paper 的核心 motivation 来自一个 observation：**2D image generation 用 auto-regressive (AR) models 取得了巨大成功**（如 VQ-GAN, DALL-E, ImageGPT），但 3D shape generation 用 AR 一直效果不佳。作者认为问题不是 AR 模型本身不行，而是 **3D 表示方法导致 AR 难以训练**。

传统 AR 3D 方法（如 ShapeFormer, AutoSDF）将 3D shape 编码为 volumetric grid 的离散 codes（例如 32³ = 32768 个 codes），然后 flatten 成一维序列给 transformer 学。这有两个根本问题：

1. **Computational explosion**：从 O(r²)（2D image）变成 O(r³)（3D voxel），32³ 已经 32768 个 tokens，64³ 直接爆炸
2. **Order ambiguity**：3D voxel 高度 spatially coupled，强行按 x-y-z 或 y-x-z 等任意顺序 flatten 会产生 approximation error。Fig. 3 和 Fig. 17 直观展示了不同 flatten order 对 generation quality 的巨大影响

**Argus3D 的 key insight**：与其在第二阶段（transformer 训练）用 complex modules 或 non-sequential 设计去 hack 这个 ambiguity（像 AutoSDF 那样），不如在第一阶段（representation learning）就把 3D shape 编码成一个 **tractable order 的 1D latent vector**。

---

## 二、数据集 Objaverse-Mix：900K Objects 多模态标注

构建大规模 3D dataset 的难点：ShapeNet 只 51K shapes 容易 overfit，Objaverse 800K 但 noisy 且无 annotations。作者采取 **assembling 策略**：合并 5 个公开 dataset

| Dataset | Scale | 特点 |
|---------|-------|------|
| ModelNet40 | 9,843 | 40 CAD categories |
| ShapeNet | 51,300 | 55 categories |
| Pix3D | 395 | real-world images + shapes |
| 3D-Future | 16,563 | furniture specialized |
| Objaverse | 800,000+ | Sketchfab source |

**预处理 pipeline**：
- Normalize + rescale 所有 shapes 到统一标准
- 用 [Stutz & Geiger 工具](https://github.com/PrincetonLIP/AutoPBR) 生成 watertight meshes、depth maps
- **手动过滤 noisy samples**：weird shapes、complex scenes、non-watertight meshes（这一步对 convergence 至关重要）
- 用 Blender 渲染 12 个 fixed views（前/后/左/右/上/下 + 4 个 45°/135° polar angle 视角），分辨率 512×512
- 用 BLIP-2 对 front-view image 生成 text caption
- 资源消耗：4 台机器 × (64-core CPU + 8×A100) × 4 周 × ~100TB 存储

最终 ~900K objects，每个 object 都有：mesh、point cloud、voxel、occupancy、12 个 rendered images、text caption。

Dataset 链接：[BAAI/Objaverse-MIX on HuggingFace](https://huggingface.co/datasets/BAAI/Objaverse-MIX)

---

## 三、Argus3D 架构详解

### 3.1 两阶段 Pipeline 总览

**Stage 1: Auto-encoder** 学 discrete representation
**Stage 2: Vanilla decoder-only transformer** 学 joint distribution of discrete codes

图 4 展示了完整 pipeline。这是经典 VQ-VAE + Transformer 的两阶段 paradigm（如 [Taming Transformers](https://compvis.github.io/taming-transformers/)），但 Stage 1 的 representation design 是 Argus3D 的核心创新。

### 3.2 Stage 1: Improved Discrete Representation Learning

完整 pipeline（参考 Appendix Table 12 的 architecture details）：

**Step 1: PointNet encode**
- 输入：point cloud P ∈ ℝ^(n×3)，n=30,000
- 输出：point features ∈ ℝ^(n×32)

**Step 2: Tri-plane projection**
将每个 point 投影到三个 axis-aligned orthogonal planes（xy, yz, xz），分辨率 256×256。落同一 grid cell 的 point features 通过 mean-pooling 聚合：
- f^xy, f^yz, f^xz ∈ ℝ^(256×256×32)
- 这步把 O(r³) 复杂度降到 O(r²)，是 [EG3D](https://nvlabs.github.io/eg3d/) 风格的 tri-plane 表示

**Step 3: Coupling network（核心创新）**
Concatenate 三个 plane（在 channel 维度），通过 3 个 conv layers 耦合：

$$f = \tau(\mathcal{G}([f^{xy}; f^{yz}; f^{xz}]; \theta)) \in \mathbb{R}^{m \times d}$$

变量解释：
- [·; ·]：concatenation operation（沿 channel 维度拼接，得到 ℝ^(256×256×96)）
- G(·; θ)：3 个 conv layers，kernel size 3, stride 1, padding 1，channel 分别 96→96→32
- τ(·)：flatten operation，row-major order
- m：latent vector 长度
- d：feature dimension

**Step 4: Down-sampler**
4 个 conv layers，stride 2 逐级下采样：
- 256×256×32 → 128×128×64 → 64×64×128 → 32×32×256
- 这一步至关重要：通过 stacked convolutions 扩大 receptive field，让每个 latent element 不再对应 fixed 3D position，从而消除 spatial mapping，得到 tractable order

**Step 5: Squeezer + Quantizer**
- Squeezer：1×1 conv 把 channel 从 256 降到 4（low-dimensional codebook lookup trick，来自 [ViT-VQGAN](https://arxiv.org/abs/2110.04627)）
- Vector quantization：

$$\mathbf{z} = \mathcal{Q}(f) := \arg\min_{\mathbf{e}_i \in \mathbf{q}} ||f - \mathbf{e}_i||$$

- Codebook q ∈ ℝ^(m_codebook × 4)，m_codebook=4096（base）/8192（large/huge）
- 每个 latent element被替换为 codebook 中最近邻的 index

**Step 6: Decoder（对称结构）**
- Unsqueezer：4→256 channel
- 2D U-Net 补充 global context
- Upsampler：32×32 → 256×256
- Decoupler：3 个 conv layers 分离为 3 个 tri-plane
- 再一个 2D U-Net 平滑每个 plane
- Implicit function：5 层 FC residual blocks，输入 query point (x,y,z)，输出 occupancy probability

### 3.3 Loss Function

$$\mathcal{L}_{rec} = \mathcal{L}_{occ} + \mathcal{L}_{code}$$

**Occupancy BCE loss**:

$$\mathcal{L}_{occ} = -(\tilde{y}_o \cdot \log(y_o) + (1 - \tilde{y}_o) \cdot \log(1 - y_o))$$

- ỹ_o：ground-truth occupancy (0 or 1)
- y_o：predicted occupancy probability

**Codebook commitment loss**:

$$\mathcal{L}_{code} = \beta ||sg[f] - \mathbf{q}_{(z)}||_2^2 + ||f - sg[\mathbf{q}_{(z)}]||_2^2$$

- sg[·]：stop-gradient operation（防止 codebook 和 encoder 互相漂移）
- β=0.25（VQ-VAE 原始值），本文用 0.4
- 第一项把 codebook entry 拉向 encoder output
- 第二项把 encoder output 拉向 codebook entry（commitment）

### 3.4 Stage 2: Vanilla Transformer

**这是 Argus3D 设计哲学的精髓**：因为 Stage 1 已经产生了 compact、tractable order 的 1D discrete representation，所以 Stage 2 可以用最 simple 的 vanilla decoder-only transformer，不需要任何 trick。

**Unconditional generation**:

$$p(\mathbf{z}) = \prod_{i=1}^{m} p(\mathbf{z}_i | \mathbf{z}_{<i})$$

- z = {z_1, z_2, ..., z_m}：m 个离散 indices
- 每步预测下一个 index，条件于之前所有 index

**Conditional generation**:

$$p(\mathbf{z}) = \prod_{i=1}^{m} p(\mathbf{z}_i | \mathbf{c}, \mathbf{z}_{<i})$$

- c：condition feature vector
- 实现方式：把 c prepend 到 [SOS] token 前面，简单到极致

**Condition encoding**:
- Point cloud：用 Stage 1 的 auto-encoder encode 成 discrete representation，再过 embedding layer
- Category：learnable embedding layer
- Image：pre-trained CLIP ViT-B/32 提取 512-d feature，再 1 个 FC layer 升到 d
- Text：CLIP 或 BERT 编码

**Objective**: Negative log-likelihood

$$\mathcal{L}_{nll} = \mathbb{E}_{x \sim p(x)}[-\log p(\mathbf{z})]$$

**Inference**:
- 用 top-k sampling 逐个 sample index
- 拿到完整 index 序列后送入 Stage 1 decoder
- 在 128³ grid 上 query occupancy
- Marching Cubes 提取 mesh，threshold=0.2

---

## 四、Scaling Up：从 100M 到 3.6B Parameters

### 4.1 三个 scale 配置（Table 1）

| Size | d (dim) | Layers | Heads | Params |
|------|---------|--------|-------|--------|
| Base | 768 | 12 | 12 | 100M |
| Large | 2048 | 24 | 16 | 1,239M |
| Huge | 3072 | 32 | 24 | 3,670M |

遵循 [GPT-3](https://arxiv.org/abs/2005.14165) 协议设计。Stage 1 的 auto-encoder 参数在三个 scale 间共享，只有 Stage 2 transformer 变大。

### 4.2 Training details

- Stage 1：ShapeNet 上 600K iterations，Objaverse-Mix 上 1300K iterations，lr=1e-4，batch=16
- Stage 2 Base：lr=1e-4，batch=8，~600K iters，单卡
- Stage 2 Huge：lr=1e-5，batch=1，3500K iters，8 卡 A100
- Huge model 还需要 manual learning rate decay（3e-6 → 1e-6）

### 4.3 推理速度
- Base：单 shape ~14 秒；32 shapes 并行 ~3 分钟
- Huge：单 shape ~50 秒

---

## 五、实验结果分析

### 5.1 Unconditional Generation（Table 2）

在 ShapeNet 5 个类别（plane, car, chair, rifle, table）上评估。Base model 已经在所有指标上超过 IM-GAN, GBIF, PointFlow, ShapeGF, PVD：

- **ECD**（Edge Count Difference，越低越好）：Argus3D 平均 240 vs 次优 GBIF 858
- **1-NNA**（越接近 50% 越好，越低表示越像 ground truth 分布）：61.17 vs 次优 ShapeGF 62.58
- **MMD**（越低越好，fidelity）：2608 vs 次优 ShapeGF 2780
- **CovT**（带 threshold 的 coverage，作者改进的 metric）：50.98 vs 次优 ShapeGF 47.41

作者引入 **Coverage with threshold (CovT)** 解决 COV 不惩罚 outliers 的问题：只有 LFD 距离小于 threshold t（用所有 competitors 的平均 MMD 设定）才算 matched。Fig. 6 直观显示 CovT 能过滤掉 outlier 的 false positive matching。

### 5.2 Class-guide Generation（Table 3, Table 8）

vs GBIF 和 AutoSDF：Argus3D 在 5 个类别平均 1-NNA 66.37（次优 AutoSDF 77.01），ECD 672 vs 1764。Table 8 加入 3DILG 对比，并用 CD/EMD 双指标评估，Argus3D-H 在 AVG 上 1-NNA CD 57.08，超过所有 baseline。

### 5.3 Multi-modal Partial Point Completion（Table 4, 5）

vs cGAN, PVD, ShapeFormer, AutoSDF，两个设置：
- Perspective completion：随机 viewpoint 移除 25%-75% 最远点
- Bottom-half completion：移除上半部分

Argus3D 在 TMD（diversity）和 MMD（fidelity）上都最佳，UHD 也最佳或次优。这说明 model 真正学到了 multi-modal completion——给定 partial shape，可以 imagine 多种合理的完整 shape。

### 5.4 Image-guide Generation（Table 6）

vs AutoSDF 和 CLIP-Forge，用 CLIP image features 作 condition：
- TMD 4.274 vs AutoSDF 2.523
- FPD 1.680 vs CLIP-Forge 8.094
- MMD 1.590 vs CLIP-Forge 1.926

有趣的是作者也实验了 ViT32 patch embeddings 和 ResNet image features 作 condition，效果也不错，证明 architecture 对 condition form 的灵活性。

### 5.5 Text-guide Generation（Table 7）

在 Text2Shape 上 vs ITG, AutoSDF, CLIP-Forge：
- Descriptions：Acc 59.93 vs CLIP-Forge 53.68
- Prompts：Acc 60.87 vs CLIP-Forge 55.00

也测试了 sequence embedding（BERT, CLIP-seq）vs single embedding（CLIP）：sequence embedding 略好（Acc 60.68 vs 59.93），证明 model 可以利用 text 的 sequence nature。

### 5.6 Large Scale Experiments（Table 8, 9）

**Argus3D-H + Objaverse-Mix pre-training** 在 image-guide generation 上：
- TMD 5.136 vs Base 4.274
- FPD 0.774 vs Base 1.680
- MMD 1.338 vs Base 1.590

Fig. 14 检查 nearest neighbor in training set，证明 Huge model 学的是 distribution 而非 memorization。

Fig. 15 展示用 DALL·E 2 生成的 unseen images（如 avocado）作 condition，Argus3D 仍能生成 reasonable shapes，说明 generalization 强。

### 5.7 Ablation Studies（核心验证）

**Ablation 1: Flattening Order 的 Effect（Table 10, Fig. 16, 17）**

测试 3 种 tri-plane flattening order（Iter-A/B/C），以及加 coupling network 后的 row-major/col-major。

关键发现：
- Tri-plane representation 对 flattening order 极其敏感，不同类别需要不同 order（Plane 偏 Iter-A，Rifle 偏 Iter-B，对应 Fig. 17 中 xz-plane 对飞机信息更多，xy-plane 对步枪信息更多）
- 标准差大，证明 ambiguity 问题
- **加上 coupling network 后，row-major 和 col-major 效果几乎一样**，证明 coupling network 确实解决了 order ambiguity

**Ablation 2: Representation Capacity（Table 11）**

- Grid (32³)：Stage 1 IoU 88.87 但 Stage 2 无法训练（sequence length 32768 太长）
- Tri-plane (32×32×3)：Stage 1 IoU 87.81，Stage 2 能训但效果差（1-NNA 73.67）
- Vector (32×32=1024)：Stage 1 IoU 88.01（接近 Grid），Stage 2 1-NNA 59.95（大幅优于 Tri-Plane）
- Codebook entry 数量从 1024 → 4096 提升明显

**Ablation 3: Memory Cost（Fig. 18）**

- Grid Reso-32：≥80GB memory
- Vector Reso-32：3.8GB memory
- 重构质量几乎一样（IoU 88.01 vs 88.87），但内存节省 20 倍以上

---

## 六、关键 Insight 与 Reflection

### 6.1 为什么 coupling network 这么有效

这是 paper 最 deep 的设计。Tri-plane representation 虽然 efficient，但仍然有 spatial structure（每个 plane 上的 element 对应 3D space 的某个投影位置）。一旦你 serialize 成 1D，element 之间的 order 就和 3D structure 强 coupling，导致 AR 学习困难。

Coupling network（4 个 stacked conv with down-sampling）做了两件事：
1. **增大 receptive field**：让每个 output element 看到 large 3D region 的 information
2. **打破 spatial mapping**：经过多次 conv + down-sample + flatten，output element 不再对应 fixed 3D position，变成了 abstract "tokens"

这种 abstract tokens 类似 NLP 中的 words——没有 spatial position，但有 semantic meaning，正好适合 transformer 的 next-token prediction 范式。

### 6.2 与 AutoSDF, ShapeFormer 的本质区别

| 方法 | 处理 ambiguity 的方式 |
|------|---------------------|
| ShapeFormer | 用 sparse representation + row-major order，未解决 ambiguity |
| AutoSDF | 用 non-sequential transformer（部分 token 同时预测），用 architecture trick 绕过 ambiguity |
| **Argus3D** | 在 representation 阶段就消除 ambiguity，transformer 保持 vanilla |

Argus3D 的优雅之处：把难点从 transformer 移到 auto-encoder，auto-encoder 简单可优化，transformer 可任意 scale。这是 **scaling-friendly design**——保留 transformer 的 simplicity 才能扩到 3.6B 参数。

### 6.3 多模态 condition 的统一

因为 Stage 2 是 vanilla transformer，condition 只是 prefix tokens，所以 condition 可以是：
- 1D vector（category, CLIP image/text feature）
- 2D feature map（ResNet feature map, ViT patch tokens）
- Sequence（BERT text sequence）
- 3D discrete representation（partial point cloud，用 Stage 1 encode）

这种 generality 是 architecture simplicity 的副产物，而非刻意设计。

### 6.4 Limitations（paper 未充分讨论）

- **3.6B 参数训练成本高**：8 卡 A100 × 3500K iters，batch=1
- **Inference 慢**：Huge model 50s per shape
- **仍需 watertight mesh**：限制了数据规模
- **没有 texture**：只生成 geometry，texture 需要额外 model
- **Codebook size 限制**：8192 entries 可能不够 large dataset 的多样性

---

## 七、相关工作脉络

Argus3D 站在几个 key 工作上：

1. **VQ-VAE / VQ-GAN** ([Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937), [Taming Transformers](https://compvis.github.io/taming-transformers/), [ViT-VQGAN](https://arxiv.org/abs/2110.04627))：两阶段 discrete representation + transformer 的 paradigm
2. **AutoSDF** ([Mittal et al. CVPR 2022](https://arxiv.org/abs/2204.02275))：non-sequential AR for 3D，Argus3D 直接对比
3. **ShapeFormer** ([Yan et al. CVPR 2022](https://arxiv.org/abs/2201.10333))：sparse voxel AR，Argus3D 在 completion 任务上对比
4. **3DILG** ([Zhang et al. NeurIPS 2022](https://arxiv.org/abs/2209.04166))：irregular latent grids for 3D generation
5. **Objaverse** ([Deitke et al.](https://objaverse.allenai.org/))：800K 3D dataset，Argus3D 的数据基础
6. **EG3D** ([Chan et al.](https://nvlabs.github.io/eg3d/))：tri-plane 表示思想来源
7. **SDFusion** ([Cheng et al. CVPR 2023](https://arxiv.org/abs/2303.13349))：同期 diffusion-based 3D generation 工作

Argus3D 的 preliminary version 叫 **ImAM** ([Luo et al. 2023](https://arxiv.org/abs/2303.14700))，本 paper 相对 ImAM 的扩展：
1. 模型从 ~100M 扩到 3.6B
2. 数据从 ShapeNet 扩到 Objaverse-Mix 900K
3. 增加 CD/EMD 指标
4. 更多 ablation 和 zero-shot 实验

---

## 八、与 Diffusion Models 对比

Paper Section 2 末尾提到 diffusion 的 limitation：U-Net 架构限制 resolution 且 training cycle 长。但这是 2023 年初的 view，现在 [3DShape2VecSet](https://arxiv.org/abs/2301.11445), [LION](https://arxiv.org/abs/2210.06978), [Rodin](https://arxiv.org/abs/2212.06135) 等 diffusion 方法也用了 transformer-based denoiser，resolution 问题用 latent diffusion 解决。

Argus3D 相对 diffusion 的真正优势在 **scaling simplicity**：vanilla decoder transformer 直接 copy GPT-3 配置就能 scale 到 3.6B，diffusion model 每次架构变体都要重新设计 conditioning 机制。但 diffusion 在 sample quality 和 mode coverage 上有 intrinsic 优势，Argus3D paper 没有正面比较最新 diffusion baselines。

---

## 九、Future Directions（基于 paper 的开放问题）

1. **Native multi-modal pre-training**：当前分别训练 image/text/point cloud condition，能否像 [Gemini](https://arxiv.org/abs/2312.11805) 一样统一训练？
2. **Tokenization 改进**：coupling network 输出 32×32=1024 tokens 仍然偏多，[MAGVIT-v2](https://arxiv.org/abs/2310.05753) 的 LFQ (Lookup-Free Quantization) 可以进一步压缩
3. **3D native transformer**：当前用 1D latent vector 是为了适配 vanilla transformer，但 3D native attention（如 [DiT-3D](https://arxiv.org/abs/2304.01748)）可能更 efficient
4. **Texture + Geometry joint generation**：Fig. 15 显示需要外挂 texture model，应该 joint training
5. **Video/4D extension**：3D shape AR 可以自然扩展到 4D motion generation，时间维度可以 append 到 latent vector

---

## 十、复现参考

- Project page: [argus-3d.github.io](https://argus-3d.github.io)
- Dataset: [BAAI/Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX)
- Baselines: [AutoSDF](https://github.com/mint97/autosdf), [ShapeFormer](https://github.com/qianxinye/ShapeFormer), [3DILG](https://github.com/zhangqianhui/3dilg)
- 相关: [ImAM (predecessor)](https://arxiv.org/abs/2303.14700), [VQ-VAE](https://github.com/deepmind/deepmind-research/tree/master/vqvae), [Taming Transformers](https://github.com/CompVis/taming-transformers)

---

## 总结 Intuition

Argus3D 的核心 insight 是 **把 representation learning 当作 AR 模型的 bottleneck 来优化**。传统思路是在 transformer 里加复杂模块去处理 3D 的特殊性，Argus3D 反其道而行——把 3D shape 编码成一个 spatially-decoupled 的 1D abstract token 序列，让 transformer 可以保持 vanilla form，从而 scale-friendly。

这种思路可以推广到其他 domain：**任何 complex structured data 想用 large-scale transformer，都应该先设计一个 spatially-decoupled、tractable-order 的 discrete representation**。这是 NLP tokenization 思想在 3D 上的对应——word token 也没有 spatial position，只有 semantic meaning，但 transformer 可以学到 ordering prior。

3.6B 参数 + 900K objects 是 3D generation 的 GPT-3 moment。这条 scaling 路线继续走下去，加上 texture、animation、interaction，可能就是通往 3D foundation model 的可行路径。
