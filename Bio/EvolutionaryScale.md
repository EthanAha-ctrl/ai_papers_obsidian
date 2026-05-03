

我来搜索并解析这个网站的内容，了解这家公司的详情。










搜索引擎暂时没有返回结果，但根据我已有的知识，我可以非常详尽地为你介绍这家公司。

---

# EvolutionaryScale — 用 AI 解码生命的语言

**官网**: https://www.evolutionaryscale.ai/

---

## 一、公司概览

**EvolutionaryScale** 是一家专注于 **Biological Foundation Model**（生物学基础模型）的 AI 公司，其核心产品是 **ESM (Evolutionary Scale Modeling)** 系列 Protein Language Model。该公司从 **Meta AI (FAIR)** 的 Protein Science 团队 spin-off 而来，由 **Alexander Rives** 领导创立。

### 核心团队
- **Alexander Rives** (CEO/Co-founder)：原 Meta FAIR 的 Protein Language Model 研究负责人，ESM 系列论文的第一作者/通讯作者。
- **Tom Sercu** (Co-founder)：同样来自 Meta FAIR，深度参与 ESM-1b、ESM-2、ESMFold 的开发。
- 团队汇集了来自 Meta FAIR、DeepMind、各大生物学实验室的顶尖人才。

### 融资情况
- 2024年中完成了约 **$142M** 的种子轮融资，估值约 **$1B+**（unicorn 级别），投资方包括 **Nat Friedman & Daniel Gross**, **Lux Capital**, **Amazon**, **NVentures (NVIDIA)** 等。

---

## 二、核心技术：ESM 系列模型

### 2.1 从 ESM-1b 到 ESM-2 再到 ESM3 的演化

| 模型 | 年份 | 参数量 | 关键创新 |
|------|------|--------|----------|
| **ESM-1b** | 2020 | ~650M | 首次证明 Protein Sequence 上的 Masked Language Model 能学到 Contact Map |
| **ESM-2** | 2022 | 15B | Scale up，结构预测接近 AlphaFold2 |
| **ESMFold** | 2022 | ~15B | 端到端从 Sequence 预测 3D Structure，无需 MSA |
| **ESM3** | 2024 | **98B** | **Multimodal Generative Model**，同时处理 Sequence + Structure + Function |

### 2.2 ESM 的第一性原理：为什么 Language Model 能理解 Protein？

从第一性原理出发，理解的关键在于：

**Protein Sequence 就是一种"语言"**。

- 自然界有约 **20 种标准 Amino Acid**，每个用一个字母表示（A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P, S, T, W, Y, V）。
- 一个 Protein 就是一个由这些字母组成的"句子"，长度从几十到几千不等。
- **进化（Evolution）** 是这些句子的"作者"：经过 **~3.8 Billion 年** 的自然选择，存活下来的 Protein Sequence 编码了 **结构和功能的约束**。

这意味着：如果你训练一个 Language Model 在大量 Protein Sequence 上做 **Masked Language Modeling (MLM)**，模型必须隐式地学习到：
- **残基间的 Co-evolution 关系**（哪些位置共同变化 → Contact Map）
- **二级结构偏好**（α-helix, β-sheet 的氨基酸偏好）
- **功能约束**（活性位点的保守性）

### 2.3 ESM-2 的 Architecture 详解

ESM-2 采用标准 **Transformer Encoder** 架构：

```
Input: Protein Sequence  x = (x₁, x₂, ..., xₗ)
      其中 xᵢ ∈ {A, R, N, D, ..., V} (20 种 amino acid + special tokens)

Embedding: E(xᵢ) ∈ ℝᵈ
           d = model dimension (例如 ESM-2 15B 中 d = 5120)

Transformer Block (重复 N 层):
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization (Pre-LN)

Output: h = (h₁, h₂, ..., hₗ)  每个 token 的 representation ∈ ℝᵈ
```

**训练目标 (Masked Language Modeling)**：

$$\mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

其中：
- $\mathcal{M}$ = 被 mask 掉的 position 集合（通常随机 mask 15% 的 token）
- $x_{\backslash \mathcal{M}}$ = 除去 masked position 之后剩余的 sequence context
- $P(x_i | x_{\backslash \mathcal{M}})$ = 模型预测第 $i$ 个位置的 amino acid 的概率

**关键洞察**：Contact Map 可以从 **Attention Map** 中提取！

$$C_{ij} = \text{APC}\left(\frac{1}{H \cdot L} \sum_{h=1}^{H} \sum_{l=1}^{L} A_{ij}^{(h,l)}\right)$$

其中：
- $C_{ij}$ = 残基 $i$ 和 $j$ 是否在空间中接触的预测
- $A_{ij}^{(h,l)}$ = 第 $l$ 层、第 $h$ 个 head 的 attention weight
- $H$ = head 数量，$L$ = 层数
- $\text{APC}$ = **Average Product Correction**，去除 phylogenetic bias

### 2.4 ESMFold：端到端 Structure Prediction

**ESMFold** 的突破在于 **不需要 Multiple Sequence Alignment (MSA)**。

与 **AlphaFold2** 的对比：

| | AlphaFold2 | ESMFold |
|---|---|---|
| **Input** | Sequence + MSA + Templates | 仅 Sequence |
| **MSA 搜索时间** | 分钟~小时 | **0** |
| **推理时间** | ~分钟级 | **~秒级** |
| **精度 (GDT-TS)** | SOTA | 略低，但在 well-represented families 上接近 |

为什么不需要 MSA？因为 ESM-2 的 Transformer 已经通过在 **~2.5亿条 Protein Sequence** 上的预训练，将 co-evolution 信息 **内化（internalize）** 到了模型参数中。MSA 本质上是在推理时 retrieve co-evolution signal，而 ESM 在训练时就已经学会了。

---

## 三、ESM3：旗舰产品——Multimodal Generative Protein Model

### 3.1 革命性架构

**ESM3** 是一个 **98 Billion Parameter** 的 **All-to-All Generative Model**，同时在三个 modality 上进行 conditioning 和 generation：

```
三个 Modality:
┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│  Sequence    │   │  Structure   │   │  Function    │
│  (AA letters)│   │ (3D coords)  │   │ (GO terms,   │
│              │   │              │   │  keywords)   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                 ┌────────▼────────┐
                 │   ESM3 (98B)    │
                 │  Transformer    │
                 │  Encoder-Decoder│
                 └────────┬────────┘
                          │
       ┌──────────────────┼───────────────────┐
       │                  │                   │
┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐
│  Sequence    │   │  Structure   │   │  Function    │
│  Generation  │   │  Generation  │   │  Annotation  │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 3.2 Structure Tokenization (VQ-VAE)

ESM3 如何将 3D Structure 变成 discrete token？使用 **Vector Quantized Variational Autoencoder (VQ-VAE)**：

$$z_q = \arg\min_{e_k \in \mathcal{C}} \| z_e(x) - e_k \|_2$$

其中：
- $z_e(x)$ = Encoder 输出的 continuous latent representation（对每个残基的局部结构）
- $\mathcal{C} = \{e_1, e_2, ..., e_K\}$ = Codebook，包含 $K$ 个离散的 code vector
- $z_q$ = 量化后的离散 token

这样，一个 Protein 的 3D Structure 就被表示为一串 **Structure Token**，可以和 Sequence Token 一起喂给 Transformer。

### 3.3 训练方式：Masked Generative Modeling

ESM3 使用 **Masked Generative Pre-training**，统一处理三个 modality：

$$\mathcal{L} = - \mathbb{E}_{\mathbf{x}, \mathcal{M}} \left[ \sum_{i \in \mathcal{M}} \left( \lambda_{\text{seq}} \log P(x_i^{\text{seq}} | \mathbf{x}_{\backslash \mathcal{M}}) + \lambda_{\text{struct}} \log P(x_i^{\text{struct}} | \mathbf{x}_{\backslash \mathcal{M}}) + \lambda_{\text{func}} \log P(x_i^{\text{func}} | \mathbf{x}_{\backslash \mathcal{M}}) \right) \right]$$

其中：
- $x_i^{\text{seq}}$, $x_i^{\text{struct}}$, $x_i^{\text{func}}$ = 第 $i$ 个位置的 Sequence Token, Structure Token, Function Token
- $\lambda_{\text{seq}}$, $\lambda_{\text{struct}}$, $\lambda_{\text{func}}$ = 各 modality 的 loss weight
- $\mathcal{M}$ = 随机 masked 的位置和 modality（可以 mask 任意子集）
- $\mathbf{x}_{\backslash \mathcal{M}}$ = 未被 mask 的所有信息

**关键洞察**：通过随机 mask 不同 modality 的不同组合，模型学会了 **任意 modality 之间的 conditional generation**：
- 给 Structure → 生成 Sequence（**Inverse Folding**）
- 给 Sequence → 生成 Structure（**Folding**）
- 给 Function Description → 生成 Sequence + Structure（**Function-guided Design**）
- 给 partial Sequence + partial Structure → 补全（**Infilling/Scaffolding**）

### 3.4 里程碑成果：生成全新的 Fluorescent Protein (esmGFP)

ESM3 最震撼的成果是 **从头生成了一个功能性 Fluorescent Protein**，命名为 **esmGFP**：

- 与自然界已知的最近的 GFP 同源物的 Sequence Identity 仅 **~57%**
- 这意味着 ESM3 "跨越"了相当于 **>5亿年** 的进化距离
- 该蛋白在实验中被合成、表达，并 **确认发出了绿色荧光**
- 这是 AI **首次从头设计出一个功能性 Fluorescent Protein**

这从根本上证明了：**Generative AI 可以探索自然进化尚未到达的 Protein Space**。

---

## 四、与竞品的对比

| 维度 | **EvolutionaryScale (ESM3)** | **DeepMind (AlphaFold)** | **David Baker Lab (RFdiffusion)** | **Generate Biomedicines** |
|---|---|---|---|---|
| **核心方法** | Protein Language Model (Generative) | Structure Prediction | Diffusion-based Design | Diffusion + Language Model |
| **是否 Generative** | ✅ 原生 Generative | ❌ AlphaFold2 不是; AF3 部分是 | ✅ | ✅ |
| **Multimodal** | ✅ Seq + Struct + Function | ❌ 仅 Struct Prediction | ❌ 仅 Struct | 部分 |
| **参数量** | 98B | ~约数百M | ~数百M | 未公开 |
| **训练数据** | ~数十亿 Protein Sequences | PDB + UniProt | PDB | 多种 |
| **无需 MSA** | ✅ | ❌ (AF2 需要) | ✅ | ✅ |

---

## 五、商业模式与应用场景

### 5.1 API/Platform 模式
EvolutionaryScale 提供 **ESM API**，类似于 OpenAI 的模式，允许 Biotech/Pharma 公司调用：

- **Drug Discovery**: 设计新的 therapeutic protein, antibody, enzyme
- **Enzyme Engineering**: 设计工业用酶（洗涤剂、生物燃料、食品加工）
- **Synthetic Biology**: 设计新的 biosynthetic pathway
- **Agriculture**: 设计抗逆性更强的蛋白

### 5.2 合作伙伴
与多家 Pharmaceutical 和 Biotech 公司合作，具体未全部公开。

---

## 六、第一性原理总结：为什么这家公司重要？

1. **Biology 的 "GPT Moment"**: 就像 GPT 证明了 Language Model 可以理解和生成人类语言，ESM 证明了同样的范式可以理解和生成 **生命的语言（Protein Sequence）**。

2. **Scale Hypothesis 在 Biology 中成立**: ESM-1b (650M) → ESM-2 (15B) → ESM3 (98B)，每次 scaling up 都带来了显著的能力涌现（Emergent Capabilities）。这验证了 **Scaling Law** 在 Protein Space 中同样适用。

3. **统一模型 vs. 专用模型**: ESM3 用一个模型统一了 Folding, Inverse Folding, Function Prediction, De Novo Design，而之前这些都需要不同的专用模型。

4. **数据飞轮**: 自然界约有 **~10⁸ 已知 Protein Sequence**（UniProt），但 Protein Space 的理论大小是 **~20³⁰⁰ ≈ 10³⁹⁰**（一个 300 残基蛋白的可能组合）。ESM3 的 Generative 能力让我们可以开始探索这个天文数字般的空间。

---

## 参考链接
- 公司官网: https://www.evolutionaryscale.ai/
- ESM3 论文 (2024): https://www.evolutionaryscale.ai/papers/esm3-simulating-500-million-years-of-evolution-with-a-language-model
- ESM-2 / ESMFold 论文: https://www.science.org/doi/10.1126/science.ade2574
- ESM-1b 论文: https://www.pnas.org/doi/10.1073/pnas.2016239118
- GitHub (ESM Open Source): https://github.com/facebookresearch/esm
- TechCrunch 报道融资: https://techcrunch.com/2024/06/25/evolutionaryscale-a-meta-spinoff-raises-142m-for-protein-generating-ai/

---

**总结一句话**: EvolutionaryScale 正在做的是 **Protein 的 Foundation Model**——用 Transformer + Scaling 的思路，训练一个能理解并创造蛋白质的 AI，目标是成为 Biology 领域的 "OpenAI"。他们的 ESM3 已经用实验验证了 AI 可以设计出自然界不存在的、有功能的全新蛋白质，这是 **Computational Biology 的里程碑事件**。