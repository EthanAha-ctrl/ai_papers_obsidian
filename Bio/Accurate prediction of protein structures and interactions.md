---
source_pdf: Accurate prediction of protein structures and interactions.pdf
paper_sha256: 078e37f11365f4994811d11fd2b945818017815046a51524d81514ad6cf7ab51
processed_at: '2026-07-18T00:01:56-07:00'
target_folder: Bio
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# RoseTTAFold: Three-Track Neural Network for Protein Structure Prediction

## 1. Paper 背景与动机

这篇 paper 由 David Baker 实验室的 Minkyung Baek 等人于 2021 年 7 月发表在 Science 上 (DOI: [10.1126/science.abj8754](https://www.science.org/doi/10.1126/science.abj8754))，是直接回应 DeepMind AlphaFold2 在 CASP14 上碾压性表现的学术作品。CASP14 (Critical Assessment of protein Structure Prediction, 14th edition) 于 2020 年举办，AlphaFold2 的 GDT_TS 中位数达到 92.4，远超其他方法。

DeepMind 在 CASP14 公开但未完整发表时，Baker 实验室基于公开的 5 点关键思路进行了独立复现与改进：

| AlphaFold2 关键点 | 描述 |
|---|---|
| 1. Raw MSA input | 直接用 MSA 而非处理后的 inverse covariance matrix |
| 2. Attention vs 2D conv | 用 attention 替代 2D 卷积处理 residue-residue 远距离关系 |
| 3. Two-track architecture | 1D sequence track 与 2D distance map track 之间迭代 |
| 4. SE(3)-equivariant refinement | 直接 refine 原子坐标而非 distance map |
| 5. End-to-end learning | 从 3D 坐标 backprop 到 input |

Baker 团队的核心改进：**将 two-track 升级为 three-track**，加入一个并行运行的 3D coordinate track，让信息在 1D / 2D / 3D 三层之间双向流动。这是 RoseTTAFold 相对于 AlphaFold2 two-track 架构最本质的结构性差异。

---

## 2. Three-Track Architecture 深度解析

### 2.1 整体数据流

参考 [RoseTTAFold GitHub](https://github.com/RosettaCommons/RoseTTAFold) 与 [Baker Lab 主页](https://www.bakerlab.org/)：

```
Input MSA ──┐
            ↓
    ┌─────────────────────────────────────────┐
    │  1D track  (M_i ∈ R^{L×d})             │
    │   - axial attention along sequence     │
    │   - row-wise attention over MSA        │
    └────┬──────────────────┬─────────────────┘
         ↑ 2D→1D projection ↓ 1D→2D projection
    ┌────┴──────────────────┴─────────────────┐
    │  2D track  (P_ij ∈ R^{L×L×d})           │
    │   - pair representation                 │
    │   - predicts distogram + orientations   │
    └────┬──────────────────┬─────────────────┘
         ↑ 3D→2D projection ↓ 2D→3D projection
    ┌────┴──────────────────┴─────────────────┐
    │  3D track  (X_i ∈ R^{L×3×3} backbone)  │
    │   - SE(3)-equivariant attention         │
    │   - updates N, Cα, C coordinates        │
    └─────────────────────────────────────────┘
         ↓
    Final 3D coordinates
```

### 2.2 1D Track 数学形式

设 MSA 长度为 $L$，深度为 $N_{\text{seq}}$，嵌入维度 $d$。1D 单序列表示：

$$M_i^{(l+1)} = \text{LayerNorm}\Big( M_i^{(l)} + \text{AxialAttn}_{\text{row}}\big(M^{(l)}\big)_i + \text{AxialAttn}_{\text{col}}\big(M^{(l)}\big)_i + \text{Proj}_{2D \to 1D}\big(P^{(l)}\big)_i + \text{Proj}_{3D \to 1D}\big(X^{(l)}\big)_i \Big)$$

变量说明：
- $M_i^{(l)} \in \mathbb{R}^{d}$：第 $l$ 层、第 $i$ 个 residue 的 1D 表示
- $P_{ij}^{(l)} \in \mathbb{R}^{d}$：第 $l$ 层 residue pair $(i, j)$ 的 2D 表示
- $X_i^{(l)} \in \mathbb{R}^{3 \times 3}$：第 $i$ 个 residue 的 backbone 三个原子坐标 (N, Cα, C)
- $\text{Proj}_{2D \to 1D}$：通过 outer product mean 或 attention pooling 把 pair 信息聚合到 single
- $\text{Proj}_{3D \to 1D}$：从 3D 坐标通过 invariant point attention (IPA) 投影

**Axial Attention** ([Ho et al. 2019, arXiv:1912.12180](https://arxiv.org/abs/1912.12180)) 是对 MSA 这一 3D tensor $(N_{\text{seq}}, L, d)$ 分别沿两个轴做 attention：
- row attention：固定 residue position $i$，对所有 sequences 做 attention
- col attention：固定 sequence，对所有 positions 做 attention

这是 standard multi-head self-attention ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762))：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

其中 $d_k$ 是 key 的维度，用于 scaling 防止内积过大导致 softmax 饱和。

### 2.3 2D Track 数学形式

2D pair representation 更新：

$$P_{ij}^{(l+1)} = \text{LayerNorm}\Big( P_{ij}^{(l)} + \text{TriangleAttn}_{\text{out}}(P^{(l)})_{ij} + \text{TriangleAttn}_{\text{in}}(P^{(l)})_{ij} + \text{OuterProdMean}(M^{(l)})_{ij} + \text{Proj}_{3D \to 2D}(X^{(l)})_{ij} \Big)$$

**Triangle multiplication / attention** 是 AlphaFold2 引入的 key inductive bias：若 $i$ 与 $k$ 接近、$k$ 与 $j$ 接近，则 $i$ 与 $j$ 也可能接近（triangle inequality 的 soft 形式）。这对 3D 几何关系建模至关重要。

**Outer Product Mean** 把 1D 信息升级到 2D：

$$\text{OuterProdMean}(M)_{ij} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} M_{s,i} \otimes M_{s,j}$$

其中 $\mathcal{S}$ 是 MSA 中的 sequences 集合，$\otimes$ 是 outer product。这相当于一个 co-evolution 信号的 soft 提取，类似 direct coupling analysis (DCA) 的 idea，但可微分、可学习。

### 2.4 3D Track 数学形式

3D track 用 SE(3)-equivariant transformer ([Fuchs et al. 2020, arXiv:2006.10503](https://arxiv.org/abs/2006.10503))。

**SE(3) 等变性定义**：对任意刚体变换 $T = (R, t) \in SE(3)$（旋转 $R \in SO(3)$、平移 $t \in \mathbb{R}^3$）：

$$f\big(T \cdot X\big) = T \cdot f(X)$$

这意味着网络学到的特征不依赖输入坐标系的朝向，是 protein 建模的天然对称性。

RoseTTAFold 使用类似 IPA (Invariant Point Attention) 的机制：对每个 residue $i$，在其 backbone 局部坐标系中查询若干 query points $Q_i = \{q_{i,k}\}_{k=1}^{K}$，然后看其他 residues $j$ 的对应点：

$$\text{IPA}_i = \sum_j w_{ij} V_j, \quad w_{ij} = \text{softmax}_j\left( \frac{1}{\sqrt{K}} \sum_{k=1}^{K} (R_i q_{i,k} + t_i)^\top (R_j q_{j,k} + t_j) \right)$$

变量说明：
- $R_i \in SO(3)$：第 $i$ 个 residue 局部坐标系的旋转矩阵
- $t_i \in \mathbb{R}^3$：第 $i$ 个 residue 的平移（通常取 Cα 位置）
- $q_{i,k} \in \mathbb{R}^3$：第 $k$ 个 query point，在 residue $i$ 的局部坐标系中
- $K$：query point 数量（RoseTTAFold 用 8 或 16）
- $w_{ij}$：attention weight

### 2.5 Three-Track Information Flow 的 Intuition

AlphaFold2 的 two-track 设计中，1D 和 2D 信息迭代收敛后，才送入 structure module 生成 3D。1D/2D 之间互相 informed，但 3D 只接收信息不反馈。

RoseTTAFold 的三轨设计相当于在 inference 时让模型有"几何直觉"：当 2D distance map 说 "residue 50 和 residue 200 距离 5Å" 但 3D track 算出来的是 12Å，模型可以 early detect inconsistency 并修正 2D prediction。反之，3D structure 反过来约束 2D distogram 的物理合理性，避免几何上不可能的 contact map。

这种 cyclic feedback 的 inductive bias 类似 diffusion model 中 score function 与 sample 之间的 mutual refinement，也类似 GNN 中 message passing 的多 hop propagation。

---

## 3. 训练与推理细节

### 3.1 Discontinuous Crops Strategy

受 GPU memory 限制 (24GB)，无法直接 forward 整条长 protein。RoseTTAFold 用 **discontinuous crops**：随机选取两个不连续的 sequence segment，共 260 residues，作为一次 forward 的输入。

为什么 discontinuous 比 continuous 更好？作者在 fig. S4A 给出实验：discontinuous crops 让网络见到非邻接但空间上可能接近的 residue pairs，模拟了真实 protein 的 contact pattern。同时，从 MSA 中可以分别选每个 crop 对应的最 informative sequences (fig. S4B)，相当于 soft attention over MSA。

形式化地，crop $C = (s_1, e_1, s_2, e_2)$，其中 $s_1 < e_1 < s_2 < e_2$，长度 $e_1 - s_1 + e_2 - s_2 = 260$。在 MSA 中，对每个 crop 选择该 crop 区域内 Shannon entropy 最高的 top sequences。

### 3.2 Perceiver Cross-Attention for MSA Compression

进一步扩展 MSA 利用率时，作者实验了 [Perceiver architecture (Jaegle et al. arXiv:2103.03206)](https://arxiv.org/abs/2103.03206)。Perceiver 用 cross-attention 让小 seed MSA ($\le 100$) 从大 MSA ($\sim 10^4$) 中"汲取"信息：

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

其中 $Q$ 来自 seed MSA (small)，$K, V$ 来自 full MSA (large)。这避免了 self-attention on full MSA 的 $O(N^2)$ 复杂度，让 10000+ sequences 的信息也能进入网络。fig. S4D 显示 promising 但需要更多训练。

### 3.3 Two Output Pathways

| 路径 | 内存 | 速度 (RTX 2080) | Side chain | 精度 |
|---|---|---|---|---|
| pyRosetta | 8GB | 5 min + 1 hour (15 CPU) | ✓ | 较高 |
| End-to-end (SE(3)) | 24GB | ~10 min (for <400 res) | ✗ | 略低 |

pyRosetta 路径的 loss：

$$\mathcal{L}_{\text{pyRosetta}} = \mathcal{L}_{\text{distogram}}(P, P^*) + \mathcal{L}_{\text{orientation}}(O, O^*) + \mathcal{L}_{\text{FAPE}}(X, X^*) + \lambda \mathcal{L}_{\text{confidence}}(c, c^*)$$

End-to-end 路径直接通过 SE(3) layer 输出 backbone，损失用 FAPE (Frame Aligned Point Error，AlphaFold2 引入)：

$$\mathcal{L}_{\text{FAPE}} = \frac{1}{L^2} \sum_{i,j} \Big\| T_i^{-1} X_j - T_i^{*-1} X_j^* \Big\|_2$$

其中 $T_i = (R_i, t_i)$ 是 residue $i$ 的局部坐标系（通常由 N-Cα-C 三原子定义），$T_i^{*-1} X_j^*$ 是 ground truth 中 $j$ 在 $i$ 局部系下的坐标。FAPE 是 SE(3)-invariant 的关键 loss。

### 3.4 Confidence Estimation

用 [DeepAccNet (Hiranuma et al. 2021, Nat Commun)](https://doi.org/10.1038/s41467-021-21511-x) 预测 per-residue accuracy，输出 PAE (Predicted Aligned Error) 和 pLDDT。在 MR (molecular replacement) 任务中，高 confidence residues 给高权重，低 confidence residues 给低权重，这对 borderline MR 案例成功至关重要。

---

## 4. CASP14 与 CAMEO Benchmark 数据

### 4.1 CASP14 性能

来自 [CASP14 官方 z-scores](https://predictioncenter.org/casp14/zscores_final.cgi)：

| 方法 | 类型 | Average TM-score |
|---|---|---|
| AlphaFold2 | Human | ~0.93 |
| RoseTTAFold (pyRosetta) | Auto | ~0.83 |
| RoseTTAFold (end-to-end) | Auto | ~0.80 |
| Two-track attention model | Auto | ~0.72 |
| BAKER (trRosetta) | Human | ~0.70 |
| BAKER-ROSETTASERVER | Server | ~0.65 |
| Zhang-server | Server | ~0.65 |

注意 AlphaFold2 使用 multi-GPU days-long inference，而 RoseTTAFold 是 single-pass。这是工程与学术自由的 trade-off。

### 4.2 CAMEO Benchmark (May 15 – June 19, 2021)

[CAMEO 官网](https://cameo3d.org/) 上的 69 个 medium/hard targets 中，RoseTTAFold 超过 Robetta、IntFold6-TS、BestSingleTemplate、SWISS-MODEL。

### 4.3 MSA Depth 鲁棒性

fig. S2 显示 RoseTTAFold 与 AlphaFold2 类似，对 MSA depth 的依赖远低于 trRosetta。这是因为 attention 机制不仅依赖 co-evolution，还能从 single sequence pattern 与 template 中提取信号。

---

## 5. 实验结构学应用

### 5.1 Molecular Replacement (MR) 成功案例

四个长期无法解析的 X-ray 数据集，用 RoseTTAFold 模型成功 MR：

| Protein | 来源 | PDB template 情况 | 结果 |
|---|---|---|---|
| GLYAT | Bos taurus | distant homolog | Solved |
| Bacterial oxidoreductase | - | distant homolog | Solved |
| SLP (surface layer protein) | bacterial | N-term partial (38% seq), C-term 无 homolog | Solved |
| Lrbp | Phanerochaete chrysosporium | 无任何可检测 homolog | Solved |

SLP 案例：RoseTTAFold model 95 个 Cα 与 refined structure 在 3Å 内 (Cα-RMSD 0.98Å)；最近 PDB template 4l3a 仅 54 个 Cα 在 3Å 内 (Cα-RMSD 1.69Å)。trRosetta 模型完全失败。

MR 用 [Phaser (McCoy et al.)](https://doi.org/10.1107/S0021889807021206) 进行，依赖 log-likelihood gain (LLGI) 评估。

### 5.2 Cryo-EM: PI3Kγ Complex p101 GBD

p101 的 Gβγ binding domain (GBD)，167 residues，HHsearch top hit E-value = 40 (远超显著性阈值 1e-3)，只覆盖 14 residues。trRosetta 完全预测错 fold。RoseTTAFold 模型与最终 refined structure 的 Cα-RMSD = 3.0Å over beta-sheets，能直接 fit 进 cryo-EM 密度图。

PDB: [7MEZ](https://www.rcsb.org/structure/7MEZ)

---

## 6. 生物学功能洞察

### 6.1 TANGO2 (transport and Golgi organization protein 2)

疾病关联：代谢紊乱、enigmatic cardiomyopathy。
RoseTTAFold 模型显示 Ntn (N-terminal nucleophile) hydrolase fold。Ntn 超家族成员催化 C-N 键水解，提示 TANGO2 可能是 enzyme（之前认为是 transport protein）。

致病突变定位：
- R26K, R32Q, L50P：靠近 active site，影响催化
- G154R：hydrophobic core 中产生 steric clash

这与 [Lalani et al. 2016](https://doi.org/10.1016/j.ajhg.2015.12.008) 报道的 TANGO2 mutations 一致。

### 6.2 ADAM33 Prodomain

ADAM (A Disintegrin And Metalloprotease) 家族有 40+ 人源基因，与 cancer metastasis、inflammation 相关。其 metalloprotease domain 结构已知，但 prodomain fold 完全未知。

RoseTTAFold 预测 prodomain 为 **lipocalin-like β-barrel**。Lipocalin 超家族包括 metalloprotease inhibitors (MPIs)。prodomain 后的延伸段有一个 cysteine，符合 "cysteine switch" mechanism ([Van Wart & Birkedal-Hansen 1990](https://doi.org/10.1073/pnas.87.14.5578))：cysteine 配位 Zn²⁺ 抑制 metalloprotease 活性。

### 6.3 CERS1 (Ceramide synthase 1)

跨膜酶，参与 sphingolipid 代谢。RoseTTAFold 预测 residues 98-304 含 6 个 TMH，up-and-down 排列。中央 crevice 包含：
- H182, D213：催化必需 (experimental)
- S212：可能参与催化
- W298：保守
- H183Q：progressive myoclonus epilepsy 致病突变

这为 [Vanni et al. 2014, Ann Neurol](https://doi.org/10.1002/ana.24170) 报道的 H183Q 致病机制提供结构基础。

### 6.4 GPCR 全人类蛋白组预测

对所有 unknown structure 的人类 GPCR (G protein-coupled receptors) 生成 active 和 inactive state 模型。GPCR 是最大类药物靶点。Benchmark 在已知 GPCR 结构上显示 RoseTTAFold 优于 [GPCRdb](https://gpcrdb.org/) 和 [RosettaGPCR](https://doi.org/10.1371/journal.pcbi.1007597)。

模型下载：[http://files.ipd.uw.edu/pub/RoseTTAFold/all_human_GPCR_unknown_models.tar.gz](http://files.ipd.uw.edu/pub/RoseTTAFold/all_human_GPCR_unknown_models.tar.gz)

### 6.5 Human Disease-Related Proteins

预测了 693 个 domain。其中 1/3 模型 pLDDT > 0.8，对应 CASP14 平均 Cα-RMSD ≈ 2.6Å。下载：[http://files.ipd.uw.edu/pub/RoseTTAFold/human_prot.tar.gz](http://files.ipd.uw.edu/pub/RoseTTAFold/human_prot.tar.gz)

lDDT (local Distance Difference Test, [Mariani et al. 2013](https://doi.org/10.1093/bioinformatics/btt473)) 公式：

$$\text{lDDT} = \frac{1}{N} \sum_{i} \frac{1}{4} \sum_{t \in \{1, 2, 4, 8\}} \Theta(d_{ij}^{\text{model}}, d_{ij}^{\text{true}}, t)$$

其中 $\Theta$ 是阈值函数，对距离差 $|d^{\text{model}} - d^{\text{true}}| \le t$ 返回 1，否则 0。lDDT 是 local metric，不依赖 global superposition。

---

## 7. Protein-Protein Complex 直接预测

### 7.1 关键 Insight: Chain Break as Feature

end-to-end 版本最后 SE(3) layer 本就处理 discontinuous crops 的 chain break。这意味着网络见过 internal chain break。**因此只需 input 两条或多条 sequence，network 直接输出 multimer 的 backbone coordinates**，省去传统的 "monomer prediction + rigid docking" pipeline。

这是 RoseTTAFold 一个意外的副产品，类似于 in-context learning：训练分布中 chain break 的存在让 network 学会了 "handle multiple chains"。

### 7.2 Paired MSA

Complex prediction 用 paired MSA ([Cong et al. 2019, Science](https://doi.org/10.1126/science.365.6449.185))：从 genomes 中找到两个 subunit 在同一 operon / gene neighborhood 中的 co-occurrence，构造跨 subunit 的 co-evolution signal。

测试集包含 2-chain 和 3-chain complexes，TM-score > 0.8 视为成功。fig. S10 显示 paired MSA 中 sequences 越多，complex 精度越高，验证 cross-chain co-evolution 是关键信号。

### 7.3 IL-12R/IL-12 Complex Case

四链 human IL-12R/IL-12 complex（fig. 4C），用 RoseTTAFold 直接生成模型，fit 进入已发表 cryo-EM 密度图 [EMD-21645](https://www.ebi.ac.uk/emdb/EMD-21645)。

关键发现：IL-12p35 的 Y189 与 IL-12Rβ2 的 G115 相互作用，类比 IL-23p19 的 W156 与 IL-23R 的 G116。这为 selective IL-12 抑制剂设计提供 atomic-level 接口。参考 [Glassman et al. 2021, Cell](https://doi.org/10.1016/j.cell.2021.01.018)。

---

## 8. 与 AlphaFold2 的对比 Intuition

### 8.1 Architecture Difference

| 维度 | AlphaFold2 | RoseTTAFold |
|---|---|---|
| Track 数 | 2 (1D, 2D) → 3D module | 3 (1D, 2D, 3D) 并行 |
| 3D refinement | 单次 final module | 迭代式与 1D/2D 交互 |
| Recycling | 3 次循环 unrolled | 同样使用 recycling |
| Compute | multi-GPU, days | single GPU, minutes |
| Open source | 后续开源 | 立即开源 |

### 8.2 为什么 Three-Track 可能更好

直觉上，1D ↔ 2D ↔ 3D 的 cyclic information flow 类似 iterative message passing in loopy graphical models。每次更新让 representation 在三种 modality 间达成一致性 (consensus)。AlphaFold2 的 structure module 是 "1D/2D 收敛 → 投影到 3D"，单向流水。

但 RoseTTAFold 精度低于 AlphaFold2 的原因可能在于：
- 训练 crop 受限 (260 residues)
- 模型 capacity 较小（hardware 限制）
- 缺少 side chain level information
- End-to-end 训练时没有 fully incorporate side chains

### 8.3 Inference Cost Trade-off

AlphaFold2 用 multi-GPU days-long recycles 与 ensembling，是 "inference time compute scaling" 的早期案例（类似 test-time compute 之于 reasoning model）。RoseTTAFold 选择 single-pass，作为 public server 友好。后续 AlphaFold-Multimer 与 RoseTTAFold All-Atom (RFdiffusion) 都吸收了这些经验。

---

## 9. 后续影响与生态

RoseTTAFold 开启了 protein design 的新时代：

1. **RFdiffusion** ([Watson et al. 2023, Nature](https://doi.org/10.1038/s41586-023-06415-8))：基于 RoseTTAFold 的 diffusion framework，用于 de novo binder design。
2. **ProteinMPNN** ([Dauparas et al. 2022, Science](https://doi.org/10.1126/science.add2187))：sequence design，与 RoseTTAFold structure prediction 互补。
3. **RoseTTAFold All-Atom** ([Krishna et al. 2024, Science](https://doi.org/10.1126/science.adl2528))：扩展到核酸、小分子、共价修饰。
4. **AlphaFold-Multimer** ([Evans et al. 2022](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2))：回应 RoseTTAFold complex prediction。

代码与 weights: [https://github.com/RosettaCommons/RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold)
在线 server: [https://robetta.bakerlab.org](https://robetta.bakerlab.org)
Zenodo: [https://zenodo.org/record/5068265](https://zenodo.org/record/5068265)

---

## 10. 关键 Take-aways (intuition building)

1. **Multi-modal parallel tracks 优于 sequential pipeline**：1D/2D/3D 三种 representation 各有 inductive bias，让它们 cyclically 交互比 cascade 更能 forced consistency。

2. **Co-evolution signals 是核心 prior**：MSA 中的 covariation 提供了 distant residues 在 3D 中接近的弱监督，attention 是 efficient 提取器，替代了传统 DCA 的 Potts model。

3. **SE(3)-equivariance 必须被 enforced**：3D coordinate prediction 必须对刚体变换等变，否则网络会浪费 capacity 学习旋转不变性。SE(3)-Transformer / IPA / FAPE 三件套共同 enforce 这一点。

4. **Geometry aware loss (FAPE) > pure distance loss**：直接监督 3D 坐标在 local frame 下的相对位置，比监督 pairwise distance 更强。

5. **Discontinuous crops 是有效的 data augmentation**：模拟 contact map 的 sparse structure，同时让长 protein 训练在 memory limit 内。

6. **Inference time compute 是 hidden dimension**：AlphaFold2 用 multi-GPU days 换取 GDT 92+，RoseTTAFold 用 single-pass 换取 serviceability。这是 scaling law 之外的另一条 axis。

7. **Cross-modal chain break induction**：训练时 handle discontinuous segments 让 network 学会处理 multimer inputs，这是 emergent capability，类似 LLM 的 in-context learning。

8. **Confidence estimation 价值 huge**：DeepAccNet 提供 per-residue accuracy prediction，让下游 MR / cryo-EM 任务可以加权 trust，是 ML for structure 的 unsung hero。

---

## References (web links)

- 主 paper: [https://www.science.org/doi/10.1126/science.abj8754](https://www.science.org/doi/10.1126/science.abj8754)
- Code: [https://github.com/RosettaCommons/RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold)
- Baker Lab: [https://www.bakerlab.org/](https://www.bakerlab.org/)
- Robetta server: [https://robetta.bakerlab.org](https://robetta.bakerlab.org)
- CASP14: [https://predictioncenter.org/casp14/zscores_final.cgi](https://predictioncenter.org/casp14/zscores_final.cgi)
- CAMEO: [https://cameo3d.org/](https://cameo3d.org/)
- AlphaFold2 (Jumper et al. 2021, Nature): [https://doi.org/10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- trRosetta: [https://doi.org/10.1073/pnas.1914677117](https://doi.org/10.1073/pnas.1914677117)
- SE(3)-Transformers: [https://arxiv.org/abs/2006.10503](https://arxiv.org/abs/2006.10503)
- Perceiver: [https://arxiv.org/abs/2103.03206](https://arxiv.org/abs/2103.03206)
- DeepAccNet: [https://doi.org/10.1038/s41467-021-21511-x](https://doi.org/10.1038/s41467-021-21511-x)
- "Attention is all you need": [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Axial attention: [https://arxiv.org/abs/1912.12180](https://arxiv.org/abs/1912.12180)
- MSA Transformer: [https://doi.org/10.1101/2021.02.12.430858](https://doi.org/10.1101/2021.02.12.430858)
- Protein coevolution networks (Cong et al.): [https://doi.org/10.1126/science.365.6449.185](https://doi.org/10.1126/science.365.6449.185)
- IL-12 receptor paper: [https://doi.org/10.1016/j.cell.2021.01.018](https://doi.org/10.1016/j.cell.2021.01.018)
- GPCRdb: [https://gpcrdb.org/](https://gpcrdb.org/)
- HHsuite: [https://github.com/soedinglab/hh-suite](https://github.com/soedinglab/hh-suite)
- RFdiffusion: [https://doi.org/10.1038/s41586-023-06415-8](https://doi.org/10.1038/s41586-023-06415-8)
- ProteinMPNN: [https://doi.org/10.1126/science.add2187](https://doi.org/10.1126/science.add2187)
- RoseTTAFold All-Atom: [https://doi.org/10.1126/science.adl2528](https://doi.org/10.1126/science.adl2528)

---

简短总结：这篇 paper 的核心贡献是证明了 three-track cyclic architecture（1D-2D-3D 并行 + cross-track message passing）在 single-GPU single-pass inference budget 下可以接近 AlphaFold2 的精度，同时具备 immediate practical utility（MR、cryo-EM fitting、complex prediction、disease variant interpretation）。它的开源、可服务性、和 David Baker 实验室下游 design pipeline 的天然 coupling，使得它成为 AlphaFold2 之外的另一根支柱，直接催生了 RFdiffusion、ProteinMPNN、RF*AA 等一系列后续工作，构成了 modern protein design stack 的 backbone。
