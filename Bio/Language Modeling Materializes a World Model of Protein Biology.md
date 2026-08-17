---
source_pdf: Language Modeling Materializes a World Model of Protein Biology.pdf
paper_sha256: fbeb6e342ed9b1976bac4cf04de8e5adc54772283c78a92c825e3f89bba02409
processed_at: '2026-08-05T11:43:35-07:00'
target_folder: Bio
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇Paper

## 一句话总结

**你拿几十亿条蛋白质序列，训一个猜下一个氨基酸的transformer，结果它自己就把整个蛋白质生物学给学会了——结构、功能、催化机制、进化关系，全在它的latent space里。**

这不是夸张。这篇paper用实验证明了这个claim。

---

## 1. 为什么"猜氨基酸"能学会生物学？

### 1.1 Intuition: 进化已经帮你做好标注了

想想看，蛋白质序列是什么？它是进化这个"优化算法"跑了几十亿年输出的solution。每个position上的amino acid不是随便选的，是被约束的：

- 这个位置在alpha helix里 → 倾向Ala, Glu, Leu
- 这个位置是active site的catalytic residue → 高度保守，几乎是His/Ser/Asp
- 这个位置和远处某个Cys形成disulfide bond → 必须是Cys
- 这个位置在membrane里 → 必须hydrophobic
- 这个位置是binding interface → 和partner蛋白co-evolve

所以当你训练模型去预测masked position的amino acid时，模型**必须**学会所有这些约束。MLM objective看起来是"猜词"，实际上是在做"inverse biology"——从进化的输出反推背后的physics和chemistry。

这就是为什么paper标题叫"Language Modeling **Materializes** a World Model"——world model不是你explicitly设计的，是它从data里自己**materialize**出来的。

### 1.2 和NLP的类比

| NLP | Protein |
|---|---|
| Token = word | Token = amino acid (20种) |
| Vocab = 50k | Vocab = 20 (+special) |
| 语义、语法、语用 | 结构、功能、进化 |
| MLM (BERT) | MLM (ESMC) |
| Scaling law emergent abilities | Scaling law → contact precision log-linear |

区别在于：蛋白质的"语法"是物理定律（ folding thermodynamics），"语义"是biochemical function，这些都是objective ground truth，不像人类语言有ambiguous和cultural的东西。所以蛋白质LM可能比text LM更clean地展现emergent world model。

---

## 2. ESMC: 基础模型

### 2.1 规模

| Model | Params | Layers | dim_model | Training data |
|---|---|---|---|---|
| ESMC 300M | 333M | 30 | 960 | 2.8B sequences |
| ESMC 600M | 575M | 36 | 1152 | 2.8B sequences |
| ESMC 6B | 6.35B | 80 | 2560 | 2.8B sequences |

对比ESM2: 最多15B params但只有50M sequences。**ESMC的breakthrough是用2.8B sequences (metagenomic数据)，让data scale跟上model scale**。

### 2.2 为什么能continue scaling?

ESM2在650M→15B之间有diminishing returns。ESMC在6B仍然log-linear:

$$\text{P@L-LR} = 0.115 \cdot \log_{10}(\text{FLOPs}) - 1.98, \quad R^2 = 0.99$$

P@L-LR = top-L most confident contact predictions中正确的比例。

**人话**: 算力翻10倍，contact prediction精度涨0.115。看起来不多，但log-linear在deep learning里已经是holy grail了——意味着没有plateau，继续scale就继续improve。

为什么ESM2 plateau但ESMC没有？因为ESM2把50M sequences学完了，再scale params就是overfitting。ESMC有2.8B sequences，6B params还underfitting data。

**这是这篇paper最重要的empirical finding之一: data scale enables compute scale**。对整个AI for Science都有implication。

### 2.3 架构细节

标准transformer + 几个trick:

- **RoPE**: rotary position embedding, 让模型能处理varying length
- **SwiGLU FFN**: 比ReLU/GELU表现好
- **Pre-norm + residual scaling**: $1/\sqrt{n_{\text{layers}}}$ scaling让深层训练稳定
- **μP**: 在小模型tune hyperparameters, transfer到大模型，避免per-scale tuning
- **WSD schedule**: Warmup-Stable-Decay, stable阶段用constant LR, decay阶段finetune

训练分两阶段:
- Stage 1: context=512, 重metagenomic (65%), 学broad statistics
- Stage 2: context=2048, 重UniRef (63%), refine长序列

### 2.4 Layerwise organization

ESMC 6B的80层里:
- Layer 0-40: contact precision很低 (在学local features)
- Layer 40-80: contact precision急剧上升 (开始学global structure)
- Layer 50-60: function classification (EC number) peak
- Layer 79 (penultimate): contact precision peak

**人话**: 模型先学局部，再学全局。Function representation在中间层最rich，structure在最深层最rich。这和CNN里low-level edge→high-level object的hierarchy类似。

reference: [ESM2 paper](https://www.science.org/doi/10.1126/science.ade2574)

---

## 3. ESMFold2: 结构预测

### 3.1 核心idea

AlphaFold2的input是MSA (multiple sequence alignment)。ESMFold的input是**单个sequence**，用language model的representation代替MSA的evolutionary information。

为什么能work? 因为ESMC已经从2.8B sequences里学到了evolutionary statistics。当你给ESMFold2一个sequence，ESMC的representation已经encode了"这个protein家族通常长什么样、有哪些constraints"。

### 3.2 架构pipeline

```
Sequence 
  → ESMC 6B (frozen, 用所有80层的hidden states)
  → Project to pair representation z_lm (L×L×256)
  → Add to initial pair state
  → [Recurrent Pair Folding Layers] × T loops
     (48 layers per loop, triangle multiplication only)
  → Diffusion Module (12-layer transformer)
  → Atomic coordinates
  → Confidence Head (pLDDT, PAE, PDE)
```

### 3.3 关键创新: Stable Recurrent Folding

AlphaFold2的recycle是feed-forward: 跑一遍，output再喂回去，不backprop through recycle。

ESMFold2用**looped transformer with contractive map**:

$$\mathbf{z}_{t+1} = \text{PairFoldingLayers}(\bar{\mathbf{A}} \odot \mathbf{z}_t + \bar{\mathbf{B}} \cdot \text{LN}(\mathbf{u}_t))$$

- $\mathbf{z}_t$: pair representation at loop $t$
- $\bar{\mathbf{A}} = \exp(-\Delta \odot \mathbf{A})$: 对角元素在(0,1)的矩阵
- $\bar{\mathbf{B}} = \Delta \odot \mathbf{B}$: input mixing矩阵
- $\Delta, \mathbf{A}, \mathbf{B}$: learnable

**人话**: 把folding layers看作一个dynamical system。标准residual $\mathbf{z}_{t+1} = \mathbf{z}_t + R(\mathbf{z}_t)$ 是unstable的（特征值>1就explode）。通过$\bar{\mathbf{A}}$把特征值压到(0,1)，系统变成contractive的，可以loop很多次不explode。

训练时backprop through 2 loops, inference时可以loop到64次。**这是inference-time scaling的enabler**。

### 3.4 简化的Pair Folding Layer

每个layer只有三个操作:
```
z = z + TriMul_out(z)     # triangle multiplication outgoing
z = z + TriMul_in(z)      # triangle multiplication incoming  
z = z + PairTransition(z) # FFN on pair state
```

**没有attention！** Paper实验发现attention在pair state processing中unnecessary，triangle multiplication就够。这让每layer快很多。

为什么triangle multiplication重要? 它implement了geometric reasoning: 如果$i$近$j$, $j$近$k$, 则$i$近$k$有约束。这是triangle inequality的neural implementation。

### 3.5 Benchmark结果

| Task | ESMFold2 single-seq | ESMFold2 +MSA | AlphaFold3 +MSA |
|---|---|---|---|
| Antibody-Antigen (DockQ≥0.23) | 50% ± 2% | 53% ± 2% | 47% ± 2% |
| Protein-Protein (DockQ≥0.23) | 70% ± 1% | 76% ± 1% | 73% ± 1% |
| Protein-Ligand success | 66% ± 1% | - | - |
| Monomer LDDT | 84% | 89% | - |

ESMFold2在single-sequence模式下beat AlphaFold3在MSA模式下的antibody-antigen prediction！

ESMFold2-Fast (24 layers, no MSA): antibody-antigen 50% ± 2%，和full ESMFold2 (MSA) 一样。Latency只要9.4s (length 1024 on H100)。

### 3.6 Inference-time Scaling

ESMFold2可以用更多inference compute换更高accuracy:
- 1 seed: antibody-antigen 49%
- 1000 seeds: 65%

**人话**: 每次用不同random seed跑（不同init noise, dropout mask, MSA subsample），得到diverse predictions。用ipTM score做ranking选最好的。这和text LLM的best-of-N sampling一个道理。

### 3.7 为什么ESMFold2比AlphaFold3快?

1. **Triangle multiplication代替attention** in pair processing
2. **Truncated diffusion**: 100步→68步，跳过高noise regime
3. **Sliding window attention** with 3D RoPE 代替 explicit pair bias tensor
4. **Custom Triton kernels** fuse operations, 减少HBM round-trips

reference: [AlphaFold3 paper](https://www.nature.com/articles/s41586-024-07487-w)

---

## 4. Protein Design: 反转模型

### 4.1 核心idea

如果你有一个model $p(\text{structure} \mid \text{sequence})$，你可以**invert**它来找sequence使得predicted structure满足某些property。

具体到binder design:
- 目标: 找一个binder sequence $\mathbf{x}$, 使它和target $t$形成high-confidence complex
- 用ESMC作为"naturalness prior": $p(\mathbf{x})$ 评价sequence是否像天然蛋白
- 用ESMFold2作为"structure oracle": $p(s \mid \mathbf{x}, t)$ 评价complex是否plausible

联合优化:
$$p(\mathbf{x}, s) = p(s \mid \mathbf{x}, t) \cdot p(\mathbf{x})$$

### 4.2 怎么optimize?

Amino acid是discrete的，不能直接gradient descent。Trick是**continuous relaxation**:

1. 初始化 logits $\mathbf{x} \in \mathbb{R}^{L \times 20}$ (每个位置20个amino acid的logit)
2. $\text{soft\_binder} = \text{softmax}(\mathbf{x} / T)$ 得到continuous distribution
3. 把soft_binder当one-hot喂给ESMFold2 (它内部用GEMM能处理)
4. Forward pass得到distogram logits $\mathbf{D}$
5. 计算losses:
   - **IntraContactLoss**: binder内部应该有confident contacts (folds into globular)
   - **InterContactLoss**: binder和target之间应该有confident contacts (binds)
   - **GlobularityLoss**: binder的radius of gyration不能太大 (应该compact)
   - **ESMC pseudo-perplexity**: binder在ESMC看来应该plausible
6. Backprop到logits $\mathbf{x}$
7. **Temperature annealing**: $T_k = T_{\min} + (1-T_{\min}) \cdot \frac{1}{2}(1 + \cos(\pi k/K))$，从high T (exploratory)到low T (commit)

### 4.3 Contact Loss细节

Distogram $\mathbf{D}^{rc} \in \mathbb{R}^{N_r \times N_c \times 128}$, 128个distance bins。

```python
# Restrict to contact bins (d < d_c)
P = softmax(D - 1e7 * (d >= d_c))  # zero out non-contact bins
# Cross-entropy per pair
H = -sum_b P * log_softmax(D)
# For each residue i, take mean of k_con smallest H_ij
# (k_con most confident contacts, with sequence separation s_min)
r_i = mean(k_con smallest H_ij where |i-j| >= s_min)
loss = mean_i(r_i)
```

参数:
- Intra (binder内部): $k_{\text{con}}=2, s_{\text{min}}=9, d_c=14$Å
- Inter (binder-target): $k_{\text{con}}=1, s_{\text{min}}=0, d_c=22$Å

**人话**: 让模型对binder内部和binder-target之间的contact预测越confident越好。如果模型说"这俩residue肯定接触"，loss就低。

### 4.4 实验设计

5个therapeutic targets: PDGFRβ, EGFR, PD-L1, CD45, CTLA-4

两种modality:
- **Minibinders**: 全新design的小蛋白 (60-200 aa)
- **scFvs**: 单链抗体, framework固定, design CDRs

两档compute:
- Low: 15k-28k candidates, 500-1800 H100-hours
- High: 67k-117k candidates, 2.4k-7.7k H100-hours

### 4.5 实验结果 - 令人震惊

**Hit rates (high compute, stringent threshold)**:
- Minibinders: 36-88% (!!)
- scFvs: 15-29%

**人话**: 设计84个minibinder去target EGFR，实验室测出来有36-88%真的能binding。这命中率在protein design领域是惊人高的。传统方法hit rate经常是个位数。

**Affinities**: 68 pM 到 70 nM。68 pM是sub-nanomolar, therapeutic级别。

**Novelty**:
- 438个minibinder中只有5个在NCBI NR里有BLAST hit (E<10^-3), 且sequence identity < 34%
- 所有scFv的CDR离PDB最近的antibody至少10 edits

**人话**: 这些design不是copy已知蛋白，是模型generate的新solution。

**Specificity**: Anti-EGFR binder对HER3 (close homolog) 无binding。Anti-CTLA-4对CD28无binding。这说明模型understand了selective binding。

**Functional validation**: PD-L1 minibinder IC50 = 1.6 nM (vs atezolizumab scFv 2.6 nM)。这是在cell-based assay里测的，binder能relieve PD-1/PD-L1 mediated T cell suppression。

**Cryo-EM**: EGFR-minibinder complex在3.8Å分辨率解析。Predicted structure和experimental structure的RMSD = 1.204Å。**模型predict的结构和真实结构几乎一模一样**。

**Inference compute scaling**: high compute vs low compute, hit rate在9/10 target-modality pair上有提升。Minibinder平均53.8%→70.0%, scFv 12.1%→21.0%。

**人话**: 多花算力search, 实验结果更好。这和text LLM的inference scaling一个道理——trade compute for quality。

reference: [BindCraft paper](https://www.nature.com/articles/s41586-024-08453-8) (类似的design思路)

---

## 5. Sparse Autoencoders: 打开黑盒

### 5.1 为什么需要SAE?

ESMC的representation是2560维的dense vector。每个维度是polysemantic的（一个维度可能同时encode helix, charge, 和binding site）。

这就像Mikolov的word2vec: king - man + woman ≈ queen, 但你不知道哪个维度encode "royalty"。

SAE (Sparse Autoencoder) 把dense representation投影到higher-dimensional但sparse的空间，希望每个dimension变成monosemantic（一个feature = 一个concept）。

### 5.2 SAE怎么训?

```python
# Input: x ∈ R^2560 (ESMC hidden state)
# Encoder
f = ReLU(W_enc^T (x - b))  # f ∈ R^16384
# TopK sparsity: only keep K=64 highest activations
f_topk = top_k(f, 64)
# Decoder
x_hat = W_dec^T f_topk + b  # reconstruct
# Loss
L = ||x_hat - x||^2 / 2560 + (1/32) * L_aux
```

L_aux是dead feature resurrection: 连续5次forward都零activation的feature被标记dead, 用它们去reconstruct residual $r = x - \hat{x}$, 让它们重新收到gradient。

训练在8B tokens上, 每层都训了一个SAE。主要分析用layer 60的SAE, 16384 features, K=64。

### 5.3 发现的Feature Hierarchy

这是paper最beautiful的部分。SAE自动学出了一个**protein biology的hierarchical decomposition**:

**Level 1 - Residue identity (88 features)**
- F189: 检测Tryptophan
- F2319: 检测Threonine
- F12861: 检测aromatic residues

**Level 2 - Secondary structure (~2000 features)**
- F13801: generic α-helix (i, i+4 positions)
- 还有helix end-caps, β-strand edges, hydrogen bond periodicity等

**Level 3 - Tertiary motif (1554 features)**
- Disulfide bonds (两个Cys在空间接近)
- Salt bridges
- Metal coordination sites
- Helix packing (两个helix的特定residue接触)

**Level 4 - Domain/fold**
- β-propeller domain
- Helix-turn-helix DNA binding domain
- 这些feature在整个domain上activate

**Level 5 - Disorder (686 features)**
- Intrinsically disordered regions
- 低complexity stretches

**Level 6 - Biochemical microenvironment (1770 features)**
- Hydrophobic patches (binding interface)
- Acidic regions (electrostatic environment)
- Amphipathic helix packing interfaces

**Level 7 - Localization (2319 features)**
- Signal peptides (ER targeting)
- Membrane anchors
- Organelle localization signals

**Level 8 - Functional site (5382+ features)**
- Catalytic sites
- PTM sites (phosphorylation, cleavage)
- DNA binding, carbohydrate binding
- Molecular transporters

**人话**: 模型自己discover了蛋白质生物学的整个reductionist hierarchy，从单个amino acid → 二级结构 → 三级motif → domain → function。这和生物学家几十年研究出来的framework一致，但它是unsupervised从序列学出来的。

### 5.4 Nucleophilic Elbow: 杀手例子

Nucleophilic elbow是一个catalytic motif: sharp turn把nucleophilic residue (Ser/Cys)精确定位来attack substrate。这个motif在**25个不同的protein fold**里独立evolve。

**Feature F6716**:
- 在99个nucleophilic elbow enzymes中的75个上activate (76%)
- 跨25个不同folds
- 在17/21个non-elbow nucleophilic enzymes上**不**activate (81% specificity)

**为什么这是big deal?** 这个feature不是在识别某个fold或某个sequence。它在识别一个**biochemical concept**: "核酸催化 elbow这个structural/chemical arrangement"。

在latent space里, F6716是一个和fold **orthogonal**的direction。同一个fold里, 有elbow的activate F6716, 没有的不activate。不同fold里, 只要有elbow, F6716就fire。

这就像在text LM里发现一个feature专门检测"sarcasm", 跨越语言、文化、语境。

### 5.5 Kinase Compositional Grammar

Kinase的P-loop (phosphate-binding loop)由一组features组合表示:

| Feature | Specificity |
|---|---|
| F792, F10583 | Generic loops/coils |
| F1635, F3614, F10646 | Glycine-rich loops (跨家族) |
| F278, F1013 | Phosphate-binding (broad) |
| F119, F4266, F4787, F6171 | Kinase-specific P-loop |

**人话**: 模型用"通用loop" + "glycine-rich" + "phosphate-binding" + "kinase-specific"层层叠加来表示P-loop。这是compositional representation——complex concept由simpler concepts组合而成。

### 5.6 Src Kinase Fitness Landscape

108个features足够预测Src kinase的mutation effect (Spearman ρ = 0.74)。

最intriguing的发现: feature F10351 (peaks at I453) 被许多mutation zero out, 这些mutation在sequence上up to 79 residues away, 但在3D空间上中位距离只有11Å。

**人话**: 模型的feature在implicitly tracking 3D neighborhood。改一个远处的residue（sequence上远，空间上近），feature的activation就变了。这暗示模型学会了一些类似allosteric regulation的东西。

### 5.7 NMF Topics: 跨生命域的Functional Themes

用NMF在81个reference proteomes上分解出3000个"topics" (feature combinations):

- **Universal topics (1103)**: ATP synthase catalytic machinery, 跨Archaea/Bacteria/Eukarya
- **Lineage-specific**: Immunoglobulins (vertebrate-only adaptive immunity)
- **Cross-lineage**: β-barrel outer membrane proteins (bacterial signature in eukaryotic organelles, 反映endosymbiosis event)

**人话**: SAE features的组合模式能detect major evolutionary events。线粒体和叶绿体里的β-barrel蛋白是几十亿年前细菌被古菌吞进去的fossil record, 这在feature space里还能看到。

reference: [Sparse Autoencoders paper](https://transformercircuits.pub/2023/monosemantic-features/index.html)

---

## 6. ESM Atlas: 行星级Map

### 6.1 规模

- **6.8 billion** unique protein sequences (来自8个database)
- **1.1 billion** predicted structures (ESMFold2)
- **230 million** SAE-based clusters (≥5 members)
- **7.7 million** clusters with ≥50 members

对比: AlphaFold Database有~200M structures, 原ESM Atlas有~700M structures。这是目前最大的protein structure atlas。

### 6.2 SAE-based Clustering

用MinHash LSH在16k-dimensional SAE feature space上cluster:

1. 每个protein → max-pooled SAE feature vector (binary: 哪些features activate)
2. 过滤IPF < 0.87的non-discriminative features (去掉22%)
3. Compute MinHash sketch (256 hashes)
4. LSH banding (r=6, b=1000 bands) → candidate pairs
5. Verify candidates with sketch Jaccard ≥ 0.6
6. Greedy clustering

Runtime scaling: $9.836 \times 10^{-5} \cdot N^{0.730}$ — sublinear! 6.8B proteins in 2 days on 64-128 nodes.

### 6.3 发现的Biology

**Fanzor/TnpB Cluster**: 9个酵母Fanzor (eukaryotic RNA-guided endonuclease) 和prokaryotic TnpB在SAE space cluster在一起, 尽管sequence identity极低。这反映了shared function, ESMC "看到"了sequence alignment看不到的evolutionary connection。

**Cas12k inactive variant**: Type V-k CRISPR的Cas12k失去catalytic function。Feature F4804 (DDE triad acidic residues) 在active Cas12f和TnsB上activate, 在inactive Cas12k上不activate。

**人话**: SAE feature能detect catalytic state, 不仅仅是structural similarity。它知道"这个蛋白有catalytic machinery" vs "这个蛋白structural上类似但catalytically dead"。

**Dark clusters**: 315个clusters没有任何Pfam annotation, 但SAE Jaccard > 0.6 similarity to Cas12/TnpB。这是unknown biology connected to known biology through latent space。

**Biome enrichment**:
- Human gut: gram-positive secretion machinery
- Soil: actinobacteria, signaling motifs
- Marine: photosystems (algae, cyanobacteria)
- Halophilic: acidic regions (高盐环境需要acidic surface维持稳定性)

### 6.4 跨越Annotation的Functional Mapping

用SAE features map Gene Ontology biological processes:
- 4113 features在precision > 0.5时specific to 61个processes
- 即使precision > 0.9, 还有491 features for 49个processes

在atlas里, 这些features identify tens到hundreds of millions of proteins per process。Up to 70%的retrieved clusters没有Pfam annotation——**模型找到了sequence similarity和Pfam HMM找不到的functional connections**。

举例:
- F2383 (polysaccharide-binding cleft) 在unannotated metagenomic protein上activate
- 用Foldseek搜索, 找到PDB里的structural match (TM-score high, 但sequence identity只有10-22%)

**人话**: 模型能给完全uncharacterized的蛋白hypothesize功能, 这是sequence-based方法做不到的。

reference: [ESM Atlas](https://atlas.esm.co/)

---

## 7. 关键Insights

### 7.1 Data Scale Enables Compute Scale

ESM2在15B plateau因为只有50M sequences。ESMC在6B still log-linear因为有2.8B sequences。

**Generalizable insight**: 很多modeling plateau其实是data bottleneck, 不是architecture bottleneck。对AI for Science尤其重要: 你的domain data规模决定你能scale到什么程度。

### 7.2 Inference-Time Compute是Design Lever

ESMFold2: 1 seed → 1000 seeds, accuracy 49% → 65%
Design: low compute → high compute, hit rate 53.8% → 70.0%

**Generalizable insight**: Test-time compute scaling在science domain也work。不只是text LLM的"thinking", 结构预测和design也能benefit from更多sampling + ranking。

### 7.3 Language Modeling Internalizes Physics

MLM objective alone, no explicit supervision, 学到了:
- 3D structure (contact prediction)
- Function (EC classification)
- Catalytic mechanisms (nucleophilic elbow)
- Evolutionary relationships (Fanzor/TnpB)
- Allosteric effects (distant mutation sensitivity)

**Generalizable insight**: Self-supervised learning on natural data, 如果data足够大足够diverse, 能learn underlying generative process的structure。这支持"compression is intelligence"的hypothesis。

### 7.4 SAEs are Universal Interpretability Tool

SAE在text LLM上work, 在protein LLM上也work, 发现的features都是domain-relevant的monosemantic concepts。

**Generalizable insight**: Superposition和sparse coding可能是brain和neural network的universal principle。SAE给的interpretability不依赖modality。

### 7.5 Structure Prediction → Design Inner Loop

ESMFold2的fast inference让gradient-based design变得tractable。需要几千次forward+backward pass, 如果像AlphaFold3那样慢, design就不可行。

**Generalizable insight**: Forward model的efficiency直接决定inverse design的feasibility。Fast differentiable simulator是computational design的key enabler。

---

## 8. 和你熟悉的工作的联系

### 8.1 和Text LLM的类比

| Text LLM | Protein LM (ESMC) |
|---|---|
| Next token prediction | Masked token prediction (MLM) |
| Emergent abilities at scale | Contact precision log-linear with compute |
| SAE features = "sycophancy", "deception" | SAE features = "nucleophilic elbow", "P-loop" |
| Best-of-N sampling | Multi-seed folding + ipTM ranking |
| RLHF alignment | (目前没有, 但design可以看作某种alignment) |
| Scaling law (Chinchilla) | Scaling law (P@L vs FLOPs) |
| Chain-of-thought | Recurrent folding loops |

### 8.2 和Diffusion Models的联系

ESMFold2的diffusion module和image diffusion (DDPM, EDM) 几乎一样:
- Forward: add Gaussian noise to coordinates
- Backward: neural net denoise
- EDM-style denoiser: $\mathbf{x}_{\text{denoised}} = \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + t^2} \mathbf{x}_{\text{noisy}} + \frac{\sigma_{\text{data}} t}{\sqrt{\sigma_{\text{data}}^2 + t^2}} \mathbf{r}_{\text{update}}$

创新在conditioning: 用pair representation (来自ESMC + triangle updates) 作为condition, 而不是text embedding (像DALL-E) 或class label。

### 8.3 和AlphaFold系列的对比

| | AlphaFold2 | AlphaFold3 | ESMFold2 |
|---|---|---|---|
| Primary input | MSA | MSA | ESMC representation |
| Pair processing | Triangle attention | Triangle attention | Triangle multiplication (no attention) |
| Structure module | Invariant Point Attention | Diffusion | Diffusion (EDM-style) |
| Recycle | Feed-forward | Feed-forward | Contractive recurrent (looped) |
| Inference scaling | Fixed | Fixed | Multi-seed + loops |
| Speed | 慢 | 中 | 快 |

ESMFold2的bigger picture idea: 如果你的language model足够好, 你不需要MSA。Language model已经internalized了evolutionary information。

---

## 9. Limitations & Open Questions

1. **SAE dimension选择**: 16384是heuristic, optimal dimension和model scale的关系不清楚
2. **Dark clusters interpretation**: 2M+ clusters没Pfam annotation, SAE给hypothesis但需要实验validation
3. **CDR design**: scFv hit rate 15-29%比minibinder 36-88%低, CDR conformational flexibility是挑战
4. **Long-range allostery**: F10351等feature对distant mutation的sensitivity是intriguing, 但需要systematic study
5. **Diffusion efficiency**: 68步, 但更aggressive truncation或distillation可能进一步加速
6. **Safety**: Paper有evaluation on viral protein DMS, ESMC对viral fitness prediction不比existing tools好, 所以dual-use risk有限

---

## 10. My Take (作为ML researcher)

这篇paper让我最excited的几点:

1. **Data scale enabling compute scale** 这个finding太重要了。现在到处有人讨论"LLM scaling is slowing down", 这篇paper说: 不是scaling停了, 是你的data不够。换个domain有足够data, 继续scale。

2. **World model从self-supervised learning emerge**。这和Ilya Sutskever说的"compression is intelligence"完全一致。你compress 2.8B protein sequences, 你就compress了整个protein biology。

3. **SAE features对应reductionist biology**。这不是coincidence。如果biology本身是hierarchical compositional的, 而你的model学到了generative process的structure, 那model的internal representation也应该是hierarchical compositional的。SAE只是把这个本来就存在的structure暴露出来。

4. **Inference-time scaling在science domain也work**。Text LLM社区的test-time compute trend会spread到science。更多sampling + better ranking = better results, 不需要retrain model。

5. **Fast differentiable simulator enables inverse design**。ESMFold2快+可微分, 让gradient-based design变得practical。这和不同iablephysics simulator (如Brax, MuJoCo MJX) enable RL的narrative类似。

6. **Interpretability不只是debug工具, 是discovery工具**。SAE features不只是让我们"看懂"model, 它们generate biological hypotheses (nucleophilic elbow, Fanzor/TnpB connection, dark clusters)。

这篇paper是AI for Science的一个milestone。它证明了一个simple thesis: 在足够大的natural data上做self-supervised learning, 能materialize domain-specific world model, 这个world model能用于prediction, design, interpretability, 和discovery。

和text LLM不同的是, 这里的evaluation更objective (DockQ, RMSD, hit rate, cryo-EM), interpretability更grounded (SAE features对应biochemical concepts), application更immediate (drug design)。

我suspect这个paradigm会spread到其他science domain: chemistry (molecular LM), materials (crystal LM), genomics (DNA LM)。每个domain只要有足够natural data, 都能materialize自己的world model。

reference: [EvolutionaryScale](https://www.evolutionaryscale.ai/) | [ESM Atlas](https://atlas.esm.co/) | [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) | [Sparse Autoencoders](https://transformercircuits.pub/2023/monosemantic-features/index.html)

---

# Language Modeling Materializes a World Model of Protein Biology - 深度解析

## 1. 这篇paper在讲什么 - Big Picture

这是EvolutionaryScale (ESM团队) 2026年6月的bioRxiv preprint，是ESM系列的集大成之作。核心thesis非常深刻：**一个masked language model，通过预测蛋白质序列中被mask掉的amino acid，internalize了整个protein biology的world model**。

让我build intuition: 想象一下你有一个超大的数据库，包含地球上所有生命形式产生的几十亿个蛋白质序列。如果你训练一个transformer去预测序列中missing的残基，这个模型必须"理解"什么？它必须理解蛋白质的structure、function、evolutionary relationships、catalytic mechanisms、cellular localization——因为所有这些都constrain了哪些amino acid能出现在哪个位置。这就是这篇paper的insight：**language modeling objective本身就是sufficient的supervision，因为它force模型去学习进化背后的physics和chemistry**。

reference: [bioRxiv preprint](https://doi.org/10.64898/2026.06.03.729735)

---

## 2. ESM Cambrian (ESMC) - the Foundation

### 2.1 架构选择

ESMC是一个标准的transformer，但有几个细节值得注意：

| Hyperparameter | ESMC 300M | ESMC 600M | ESMC 6B |
|---|---|---|---|
| Layers ($n_{\text{layers}}$) | 30 | 36 | 80 |
| Model dim ($d_{\text{model}}$) | 960 | 1152 | 2560 |
| Heads ($n_{\text{heads}}$) | 15 | 18 | 40 |
| Head dim ($d_{\text{head}}$) | 64 | 64 | 64 |
| FFN dim | 2560 | 3072 | 6912 |

关键技术点：

1. **RoPE (Rotary Position Embeddings)**: 用rotary encoding而非absolute position，这让模型能generalize到varying sequence lengths
2. **Pre-norm architecture**: layer norm放在每个sub-layer之前
3. **SwiGLU activation**: FFN用SwiGLU而不是ReLU/GELU
4. **Bias-free**: 所有linear layer都没bias
5. **Residue scaling**: $\frac{1}{\sqrt{n_{\text{layers}}}}$ scaling factor应用在attention和FFN输出上，这个trick让深层transformer训练更稳定

为什么residue scaling有效？intuition是：在深层网络中，residual stream的variance会随着层数线性累积。如果你在每层加一个$O(1)$的perturbation，累积$n$层后总variance是$O(n)$，这会让下游head的训练困难。通过$1/\sqrt{n}$ scaling，你保持residual stream的variance大致constant。

### 2.2 Training objective: MLM

$$\mathcal{L} = \mathbb{E}_{x, M}\left[-\sum_{i \in M} \log p(x_i \mid x_{\backslash M})\right]$$

变量解释：
- $x$: 完整的蛋白质序列 (e.g., "MKLVL...")
- $M$: 被mask掉的positions集合 (随机采样15%)
- $x_i$: 位置$i$的真实amino acid
- $x_{\backslash M}$: 除了mask positions之外的所有positions
- $p(x_i \mid x_{\backslash M})$: model预测position $i$的amino acid分布

这个objective为什么能work？intuition：进化决定了每个position的amino acid。如果你想准确预测，你必须理解：
- 局部化学环境 (charge, hydrophobicity)
- 二级结构倾向性 (alpha helix偏好Ala, Glu, Leu)
- 长程contact约束 (disulfide bond需要两个Cys在空间上接近)
- functional约束 (active site residues高度保守)
- evolutionary信号 (同家族蛋白的co-evolution)

### 2.3 Scaling Law - 这里的关键发现

paper的核心empirical observation：

$$\text{P@L-LR} = 0.115 \times \log_{10}(\text{FLOPs}) - 1.98, \quad R^2 = 0.99$$

- P@L-LR: 用attention head预测long-range contacts的precision (top L most confident predictions中正确的比例)
- L: 蛋白质长度
- FLOPs: training compute

**关键发现**: ESMC在scaling上表现出log-linear关系，没有plateau。这和ESM2不同，ESM2在650M到15B之间有diminishing returns。

为什么ESMC能继续scale？因为ESM2只在50M sequences (UniRef50)上训练，而ESMC用了**2.8 billion sequences**，包括：
- UniRef 2023_02: 156M sequences
- JGI metagenomic: 2.029B sequences  
- MGnify 2023_02: 621M sequences

这给了一个重要insight: **data scale是continue scaling的关键enabler**。模型容量不是瓶颈，data才是。

### 2.4 Two-stage Training

| Stage | Steps | Context | Data mix |
|---|---|---|---|
| Stage 1 | 1M | 512 | JGI 54%, MGnify 11%, UniRef 36% |
| Stage 2 | 500K | 2048 | UniRef 63%, JGI 31%, MGnify 6% |

intuition: Stage 1学broad sequence statistics from metagenomic data (short context)，Stage 2 refine representations on curated UniRef data (long context)。

### 2.5 μP (Maximal Update Parameterization)

ESMC用μP来scale learning rate across model sizes：
- 在小proxy model (d=512, n=16)上tune hyperparameters
- Transfer to larger models时: LR scale inversely with width, square root of depth
- 这避免了per-model hyperparameter tuning

reference: [μP paper](https://arxiv.org/abs/2203.03466)

---

## 3. ESMFold2 - Structure Prediction

### 3.1 整体架构

ESMFold2的pipeline：

```
Sequence → ESMC (frozen) → representations from all layers
                            ↓
                         Project to pair state (z_lm)
                            ↓
       ┌─────────────────────────────────────┐
       │  Recurrent Pair Folding Layers      │
       │  (48 layers, looped T times)       │
       └─────────────────────────────────────┘
                            ↓
                    Diffusion Module
                    (12-layer transformer)
                            ↓
                    Atomic coordinates
                            ↓
                    Confidence Head
                    (pLDDT, PAE, PDE)
```

### 3.2 关键创新: Stable Recurrent Folding

传统AlphaFold2的recycle是feed-forward的，而ESMFold2用**looped transformer with contractive map**:

$$\mathbf{z}_{t+1} = \text{PairFoldingLayers}(\bar{\mathbf{A}} \odot \mathbf{z}_t + \bar{\mathbf{B}} \mathbf{LN}(\mathbf{u}_t))$$

变量解释：
- $\mathbf{z}_t$: pair representation at iteration $t$ (shape: $L \times L \times d_{\text{pair}}$)
- $\mathbf{u}_t$: input features (from ESMC + atom features + MSA)
- $\bar{\mathbf{A}} = \exp(-\Delta \odot \mathbf{A})$: Zero-Order-Hold discretized negative-diagonal matrix
- $\bar{\mathbf{B}} = \Delta \odot \mathbf{B}$: Euler discretized
- $\Delta, \mathbf{A}, \mathbf{B}$: learnable channel-wise parameters
- $\odot$: element-wise product
- $\mathbf{LN}$: layer norm

**Contractive map**: constrain $\bar{\mathbf{A}}$ to have eigenvalues in $(0, 1)$. 这prevent residual stream从unbounded growth across long recurrent unrolls。

intuition: 这相当于把pair folding layers看作一个**discretized linear dynamical system**。标准residual $\mathbf{z}_{t+1} = \mathbf{z}_t + R(\mathbf{z}_t)$ 是unstable的（特征值在单位圆外就会explode）。通过$\bar{\mathbf{A}}$把特征值限制在(0,1)，保证了stability。

### 3.3 简化的Pair Folding Layer

```python
def PairFoldingLayer(z, mask):
    z = z + Dropout(TriMul_out(z, mask))
    z = z + Dropout(TriMul_in(z, mask))
    z = z + PairTransition(z)
    return z
```

关键insight: **attention在pair state processing中是unnecessary的**！只用triangle multiplication就sufficient了。这大幅减少compute和memory。

为什么triangle multiplication重要？它maintain了geometric consistency: 如果residue $i$和residue $j$接近，residue $j$和residue $k$接近，那么$i$和$k$的距离有约束。Triangle updates implement了这个triangle inequality的reasoning。

reference: [Triangle Multiplication paper](https://arxiv.org/abs/2510.18870)

### 3.4 Diffusion Module

ESMFold2用diffusion来predict atomic coordinates:

$$\mathbf{x}_{\text{denoised}} = \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + t^2} \cdot \mathbf{x}_{\text{noisy}} + \frac{\sigma_{\text{data}} \cdot t}{\sqrt{\sigma_{\text{data}}^2 + t^2}} \cdot \mathbf{r}_{\text{update}}$$

变量：
- $\mathbf{x}_{\text{noisy}}$: noisy atomic positions
- $t$: noise level
- $\sigma_{\text{data}} = 16$: data scaling factor
- $\mathbf{r}_{\text{update}}$: network output (normalized)
- 训练时: $\sigma = \sigma_{\text{data}} \cdot \exp(\mu + \sigma_{\text{std}} \cdot z)$, $z \sim \mathcal{N}(0,1)$, $\mu = -1.2$, $\sigma_{\text{std}} = 1.5$

**Truncated diffusion**: 跳过high-noise regime ($\sigma \gg \sigma_{\text{data}}$)，因为这时signal-to-noise ratio可忽略，denoiser的prediction被scaling prefactor主导。$\sigma_{\max} = 256$ 让100步变成68步，没有性能损失。

### 3.5 Sliding Window Atom Attention

AlphaFold3用block local attention with explicit pair bias tensor (内存$O(L^2)$)。ESMFold2用**standard sliding-window attention (window=128)** with **3D RoPE**:

```python
def Build3DRoPE(r, u):
    # r: atom positions [B, A, 3]
    # u: residue space UIDs [B, A]
    for d in {x, y, z}:
        for k in range(n_spatial):  # n_spatial=2
            theta_d_k = r[:, :, d] / f_spatial^(k/n_spatial)  # f_spatial=20
    for k in range(n_uid):  # n_uid=10
        theta_uid_k = u / f_uid^(k/n_uid)  # f_uid=10000
    theta = concat(theta_x, theta_y, theta_z, theta_uid)
    return cos(theta), sin(theta)
```

intuition: 把atom的3D坐标和residue ID编码进attention的query/key。这让attention能"看到"空间邻近性而不需要explicit pair tensor。3D RoPE在$x, y, z$三个轴各用2个frequencies ($f_{\text{spatial}}^{0/2}, f_{\text{spatial}}^{1/2}$ = 1, 4.47)，再用10个frequencies编码residue ID ($f_{\text{uid}}^{k/10}$, $k=0...9$)。

### 3.6 Benchmark结果

| Benchmark | ESMFold2 (single seq) | ESMFold2 (MSA) | AlphaFold3 (MSA) |
|---|---|---|---|
| Antibody-Antigen DockQ≥0.23 | 50% ± 2% | 53% ± 2% | 47% ± 2% |
| Protein-Protein DockQ≥0.23 | 70% ± 1% | 76% ± 1% | 73% ± 1% |
| Protein-Ligand | 66% ± 1% | - | - |
| Monomer LDDT | 84% | 89% | - |

ESMFold2-Fast (24 layers, no MSA) 在antibody-antigen上达到50% ± 2%，和full ESMFold2 (with MSA) 相当！

### 3.7 Inference-Time Scaling

ESMFold2的inference-time scaling是这篇paper的重要发现：
- 1 seed: antibody-antigen 49%
- 1000 seeds: 65%

为什么？每次run用不同的random seed (init noise, dropout mask, MSA subsample)，这相当于diverse ensemble。通过ranking by ipTM，你能选出correct structures。

### 3.8 Latency

| Model | Length 1024 latency |
|---|---|
| ESMFold2-Fast | 9.4s |
| ESMFold2 | 15.8s |
| AlphaFold3 | ~20s |
| Chai-1 | 110s |

reference: [AlphaFold3 paper](https://www.nature.com/articles/s41586-024-07487-w)

---

## 4. Protein Design - 真正的杀手锏

### 4.1 设计objective

paper把design formulates成joint optimization:

$$p(\mathbf{x}, s) = p(s \mid \mathbf{x}, t) p(\mathbf{x})$$

- $\mathbf{x}$: candidate binder sequence
- $t$: target sequence (已知)
- $s$: structure of target-bound complex
- $p(\mathbf{x})$: ESMC提供的sequence prior (像"naturalness" prior)
- $p(s \mid \mathbf{x}, t)$: ESMFold2提供的conditional structural model

intuition: 我们要找一个序列$\mathbf{x}$，使得：
1. $\mathbf{x}$在ESMC看来是"plausible"的 (像天然蛋白质)
2. $\mathbf{x}$和target $t$形成的complex $s$在ESMFold2看来是high-confidence的

### 4.2 Gradient-based Search

Algorithm 11的核心:
```
Initialize x (logits over amino acids)
for k = 1 to 150:
    T_k = T_min + (1 - T_min) * 0.5 * (1 + cos(πk/K))  # cosine anneal
    α_k = α_max * T_k  # learning rate tracks temperature
    soft_binder = softmax(x / T_k)
    F ~ Uniform{F1, F2}  # random ESMFold2 replica
    soft_complex = [onehot(target); soft_binder]
    
    if T_k >= 0.05:
        D_k = F(soft_complex; loops=1, trunk only)
    else:
        (D_k, ipTM_k) = F(soft_complex; loops=1, steps=50, confidence)
    
    L_intra = IntraContactLoss(D_k)  # binder internal contacts
    L_inter = InterContactLoss(D_k)  # binder-target contacts
    L_glob = GlobularityLoss(D_k)   # binder should be globular
    L_struct = λ_intra * L_intra + λ_inter * L_inter + λ_glob * L_glob
    
    L_LM = MaskedPseudoPPL(ESMC, soft_binder)  # naturalness
    
    g = normalize(∇_x L_struct) + λ_LM * normalize(∇_x L_LM)
    x = x - α_k * g
```

关键设计:
1. **Continuous relaxation**: 把discrete amino acid选择relax成continuous softmax distribution over 20 amino acids
2. **Temperature annealing**: 从high T开始 (soft, exploratory)，逐渐降低到low T (接近one-hot, commit to choices)
3. **Straight-through estimator**: $\tilde{b} = \text{onehot}(\arg\max \text{soft\_binder}) + \text{soft\_binder} - \text{stop\_grad}(\text{soft\_binder})$ 让forward是discrete，backward是soft
4. **Gradient normalization**: $\hat{g} = \sqrt{n_{\text{mutable}}} \cdot (g \odot m) / \|g \odot m\|_F$ 保证different loss terms量级一致

### 4.3 Contact Loss

对distogram logits $\mathbf{D}^{rc} \in \mathbb{R}^{N_r \times N_c \times B}$:

```python
def ContactLoss(D, k_con, s_min, d_c):
    # Restrict softmax to contact bins (d < d_c)
    P = softmax(D - 10^7 * (d >= d_c))  # mask non-contact bins
    H = -sum_b P * log_softmax(D)  # binned cross-entropy
    # For each row i, take mean of k_con smallest values of H_ij
    # where |i - j| >= s_min
    r_i = mean of k_con smallest H_ij (j satisfies separation)
    return mean over i of r_i
```

变量：
- $k_{\text{con}}$: contacts per row (intra: 2, inter: 1)
- $s_{\text{min}}$: minimum sequence separation (intra: 9, inter: 0)
- $d_c$: contact cutoff (intra: 14Å, inter: 22Å)
- $B$: number of distogram bins (128)

### 4.4 Globularity Loss

```python
def GlobularityLoss(D_bb, L_binder):
    E_ij = sum_b softmax(D_bb_ij)_b * min(d_b, 27)^2  # expected squared distance
    R_g = sqrt(sum_{i>j} E_ij / L_binder^2)  # radius of gyration
    return ELU(R_g - 2.38 * L_binder^0.365)  # empirical packing radius
```

intuition: $R_g = 2.38 \cdot L^{0.365}$ 是empirical relation for globular proteins。如果predicted radius超过这个，penalize。

### 4.5 实验结果

**Targets**: PDGFRβ, EGFR, PD-L1, CD45, CTLA-4

**Hit rates (high compute)**:
- Minibinders: 36-88%
- scFvs: 15-29%

**Affinities**: 68 pM到70 nM (sub-nanomolar的都有)

**Novelty**: 
- Minibinders: 只有5/438在NR中有significant BLAST hit (E-value < 10^-3)，且sequence identity < 34%
- scFvs: 所有都至少10 edits away from nearest PDB antibody CDR

**Specificity**: Anti-EGFR binders对HER3无binding；Anti-CTLA-4对CD28无binding

**Functional activity**: PD-L1 minibinder IC50 = 1.6 nM (vs atezolizumab scFv 2.6 nM)

**Inference compute scaling**: minibinder hit rate 53.8% → 70.0%, scFv 12.1% → 21.0%

### 4.6 Cryo-EM Validation

EGFR-minibinder complex在3.8Å分辨率下解析。Predicted structure vs experimental: RMSD = 1.204Å (within experimental uncertainty)。这是非常强的validation！

---

## 5. Sparse Autoencoders - 解析Latent Space

### 5.1 SAE Architecture

```python
# Encoder
f = ReLU(W_enc^T (x - b))  # f ∈ R^d_SAE, d_SAE >> d_model

# Decoder
x_hat = W_dec^T f + b  # reconstruct x

# TopK sparsity
f_topk = top_k(f, K)  # only keep K highest activations

# Loss
L_recon = ||x_hat - x||^2 / d_model
L_aux = ||r_hat - r||^2 / d_model if |D| > 0 else 0
# where r = x - x_hat, r_hat from dead features
L = L_recon + (1/32) * L_aux
```

变量：
- $x \in \mathbb{R}^{d_{\text{model}}}$: hidden state from ESMC (d=2560 for 6B)
- $W_{\text{enc}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$: encoder weight
- $W_{\text{dec}} \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$: decoder weight  
- $d_{\text{SAE}}$: feature dimension (16,384 for main analysis)
- $K$: sparsity (64 for main analysis)
- $\mathcal{D}$: dead features (zero activation for 5+ forwards)
- $K_{\text{aux}} = \min(512, |\mathcal{D}|)$: auxiliary TopK

**Dead feature resurrection**: 当feature连续5次forward都零activation，把它标记为dead。然后用dead features去reconstruct residual $r = x - \hat{x}$，让它们重新收到gradient。

### 5.2 发现的Features Hierarchy

paper识别出8个complexity levels的features：

| Level | Category | # Features | Example |
|---|---|---|---|
| 1 | Residue identity | 88 | Trp, Thr, Cys; aromatic, negative |
| 2 | Secondary structure | ~2000 | α-helix, β-strand, helix caps, strand edges |
| 3 | Tertiary motif | 1554 | Disulfide bonds, salt bridges, metal coordination |
| 4 | Domain/fold | - | β-propeller, HTH DNA-binding |
| 5 | Disorder/low-complexity | 686 | IDR, low-complexity stretches |
| 6 | Biochemical microenvironment | 1770 | Hydrophobic patches, acidic regions |
| 7 | Localization/topology | 2319 | Signal peptides, membrane anchors |
| 8 | Functional site/region | 5382+ | Catalytic sites, PTM sites, binding sites |

### 5.3 Nucleophilic Elbow - 案例分析

这是paper最beautiful的例子。Nucleophilic elbow是一个catalytic motif: 一个sharp turn把nucleophilic residue (Ser/Cys)精确定位来attack substrate。这个motif在25个不同folds中独立evolve。

**Feature F6716**:
- 在99个nucleophilic elbow enzymes中的75个上activate (76%)
- Spanning 25 distinct folds
- 在17/21个non-elbow nucleophilic enzymes上不activate (specificity 81%)

为什么这是big deal？这feature不是在识别某个fold或某个sequence motif，它在识别一个**biochemical concept**：核酸催化 elbow 这个具体的structural/chemical arrangement。这在latent space中是一个**orthogonal direction to fold**。

### 5.4 Kinase Compositional Grammar

Kinase的核心catalytic machinery由多个features组合表示：

P-loop (phosphate-binding loop):
- F792, F10583: generic loops
- F1635, F3614, F10646: glycine-rich loops (universal)
- F278, F1013: phosphate-binding (broad)
- F119, F4266, F4787, F6171: kinase-specific P-loop

intuition: 同一个P-loop，由"generic loop" + "glycine-rich" + "phosphate-binding" + "kinase-specific"层层叠加表示。这反映了蛋白质的compositional structure。

### 5.5 Src Kinase Fitness Prediction

108个features足够预测Src kinase的local activity landscape (Spearman ρ = 0.74)。

最interesting的发现: **features可以respond to distant mutations**。例如F10351 (peaks at I453)被许多mutation zero out，这些mutation在sequence上up to 79 residues away，但在3D空间上中位Cα距离只有11Å。这暗示feature捕捉了3D structural neighborhood。

### 5.6 NMF Topics

用Bernoulli-Poisson NMF分解81个reference proteomes的feature activation:

$$\Lambda \approx W \cdot H + \mathbf{1}_N b^T$$

- $X_{ij} \sim \text{Bernoulli}(P_{ij})$, $P_{ij} = \mathbb{P}[Z_{ij} \geq 1]$
- $Z_{ij} \sim \text{Poisson}(\Lambda_{ij})$
- $W \in \mathbb{R}^{N \times T}$: protein-topic weights
- $H \in \mathbb{R}^{T \times M}$: topic-feature weights
- $b$: feature-specific biases
- $T = 3000$ topics

发现:
- 1103 topics跨所有三域universal (e.g., ATP synthase)
- Lineage-specific: immunoglobulins (vertebrate-specific)
- Cross-lineage: β-barrel outer membrane proteins (endosymbiosis signature)

---

## 6. ESM Atlas - 规模化

### 6.1 Numbers

- 6.8 billion unique protein sequences
- 1.1 billion predicted structures
- 230M clusters with ≥5 members
- 7.7M clusters with ≥50 members

### 6.2 SAE-based Clustering

用MinHash LSH在16k-dimensional SAE feature space上cluster:

```
Stage 1: Compute m-dimensional MinHash sketch
  - Filter features with IPF > 0.87 (remove 22% non-discriminative)
  - r=6 band width, b=1000 bands
Stage 2: Candidate pair generation + verification
  - For each band, group proteins by hash
  - Select representative per bucket
  - Verify with 256-dim sketch
Stage 3: Greedy clustering
  - Priority-based set cover
```

Jaccard threshold = 0.6 (Pfam same-family中88%超过这个)

Runtime scaling: $9.836 \times 10^{-5} \cdot N^{0.730}$, sublinear! 100M proteins in 1 hour, 6.8B in 2 days.

### 6.3 发现的Biology

**Fanzor/TnpB cluster**: 9个酵母Fanzor和TnpB在SAE space中cluster在一起，尽管sequence identity极低。这反映了它们的shared function (RNA-guided endonuclease)。

**Cas12k inactive variant**: Type V-k CRISPR系统的Cas12k失去了catalytic function。F4804 (DDE triad acidic residues)在active Cas12f和TnsB上activate，但在inactive Cas12k上不activate。这是feature在detect catalytic state，不仅仅是structural similarity。

**Dark clusters**: 315个clusters没有任何Pfam annotation，但Jaccard > 0.6 similarity to Cas12/TnpB。这是unknown biology connected to known biology through latent space。

### 6.4 Biome Enrichment

- Human gut: enriched for gram-positive secretion machinery
- Soil: enriched for actinobacteria, signaling motifs
- Marine: photosystems, lumenal domains (algae, cyanobacteria)
- Halophilic: haloarchaeal signatures, acidic regions (盐稳定性需要acidic surface)

---

## 7. 重要的Insights for ML Researchers

### 7.1 Data Scale Enables Compute Scaling

ESM2在15B params遇到plateau，因为只有50M sequences。ESMC用2.8B sequences，在6B仍然log-linear improvement。这暗示**很多model plateau其实是data bottleneck**。

### 7.2 Inference-Time Compute as a Design Lever

ESMFold2用recurrent loops和multi-seed sampling实现inference-time scaling。这和text LLM的"thinking" trend一致。Design quality也通过inference compute scaling (53.8% → 70.0% hit rate)。

### 7.3 Language Modeling Internalizes Physics

通过MLM objective alone，模型学到了：
- 三维结构 (contact precision作为proxy)
- 功能 (EC number classification)
- Catalytic mechanisms (nucleophilic elbow feature)
- Evolutionary relationships (Fanzor/TnpB clustering)

这和text LLM里观察到的emergent capabilities类似，但更striking，因为这里我们没有任何explicit supervision。

### 7.4 SAEs Provide Interpretability Across Modalities

SAEs在text LLM上work，在protein LLM上也work。这暗示**superposition和monosemantic features是universal phenomenon**，不是text-specific。

### 7.5 Structure Prediction可以作为Design的Inner Loop

ESMFold2的efficiency让gradient-based design变得tractable。design算法需要几千次forward+backward pass，所以fast structure model是enabler。

---

## 8. Limitations and Open Questions

1. **Diffusion module仍有改进空间**: 当前68 steps，但paper show这个就够了。是否更aggressive truncation或distillation能进一步加速？
2. **SAE dimension的选择**: 16,384是heuristic选择，optimal dimension和model scale的关系还没理论指导
3. **Dark clusters的interpretation**: 2M+ clusters没有Pfam annotation，SAE features能给出hypothesis但需要experimental validation
4. **Antibody CDR design**: 虽然scFv hit rate 15-29%不错，但相比minibinder 36-88%还是低。CDR的conformational flexibility是挑战
5. **Long-range allostery**: F10351等feature对distant mutations的sensitivity是intriguing，但需要更systematic study

---

## 9. 相关References

- [ESM2 (Lin et al. 2023, Science)](https://www.science.org/doi/10.1126/science.ade2574)
- [ESM3 (Hayes et al. 2025, Science)](https://www.science.org/doi/10.1126/science.ads0788)
- [AlphaFold2 (Jumper et al. 2021, Nature)](https://www.nature.com/articles/s41586-021-03819-2)
- [AlphaFold3 (Abramson et al. 2024, Nature)](https://www.nature.com/articles/s41586-024-07487-w)
- [Sparse Autoencoders (Bricken et al. 2023)](https://transformercircuits.pub/2023/monosemantic-features/index.html)
- [Scaling Monosemanticity (Templeton et al. 2024)](https://transformercircuits.pub/2024/scaling-monosemanticity/index.html)
- [InterPLM (Simon & Zou 2025, Nature Methods)](https://www.nature.com/articles/s41592-025-02671-9)
- [BindCraft (Pacesa et al. 2025, Nature)](https://www.nature.com/articles/s41586-024-08453-8)
- [RoPE (Su et al. 2021)](https://arxiv.org/abs/2104.09864)
- [FlashAttention (Dao 2024)](https://arxiv.org/abs/2307.08691)
- [μP (Yang et al. 2022)](https://arxiv.org/abs/2203.03466)
- [Looped Transformers (Prairie et al. 2026)](https://arxiv.org/abs/2604.12946)
- [Evolutionary Scale Blog](https://www.evolutionaryscale.ai/blog)

---

## 10. 总结

这篇paper的core message:

**Language modeling materializes a world model of protein biology within the model's representation space.**

通过一个seemingly simple的MLM objective，在足够大的data上训练足够大的model，你得到一个representation space，它:
1. recover了atomic-resolution structure prediction (ESMFold2)
2. enable了inverse design (high-affinity binders)
3. organize了biology的hierarchical structure (SAE features)
4. connected evolutionarily distant proteins (Fanzor/TnpB)
5. scale到planetary-size data (6.8B sequences)

这和text LLM的"emergent abilities"叙事类似，但更compelling，因为:
- Objective更simple (20 amino acids vs 50k tokens)
- Evaluation更objective (DockQ, RMSD vs BLEU, MMLU)
- Interpretability更grounded (SAE features对应biochemical concepts)
- Application更immediate (drug design, atlas building)

这是AI for Science的一个strong existence proof：general-purpose self-supervised learning on natural data can internalize the underlying science.
