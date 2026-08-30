---
source_pdf: D-SPIN-constructs-regulatory-network-models-from-s.pdf
paper_sha256: db30b086d7602ce9676561d4b76c9102988ae9c7b12d71da703d73d4dbae74eb
processed_at: '2026-08-18T04:30:35-07:00'
target_folder: Bio
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

D-SPIN 把 gene regulatory network 当成一块**永远不变的电路板**，每种 perturbation（drug、knockdown）只是给电路板的不同位置通电或断电。你观察够多次"通电后的输出"，就能反推电路板长什么样，还能预测没试过的通电组合会产生什么。

## 为什么之前的方法不行

想象你在 reverse engineer 一块 mystery circuit board。你能做的事：probe 每个引脚的电压，看它们怎么联动。

传统方法（GENIE3、GRNBoost2、PIDC）的思路是：**盯着引脚电压的 correlation**。如果两个引脚总是同涨同跌，就猜它们有连线。

问题在哪？电路板有 redundant inhibition。A3 抑制 B3，C3 也抑制 B3，两个 inhibition 叠加，B3 被压得死死的，看起来跟 A3 完全不相关。Correlation 为零，但 A3→B3 这条线真实存在。

这就像你想通过观察城市交通流量反推道路网络，但某条路永远堵车堵到没车流，你就以为这条路不存在。

**Perturbation 是破局关键**。你把 C3 这条路封掉，A3 对 B3 的抑制就显出来了，correlation 突然飙高。但每次 perturbation 只 reveal 一小块，你需要一个 global model 把所有 perturbation 的信息缝起来。传统方法没有这个 global model，它们只是把所有 perturbation 数据 pool 在一起跑 correlation，结果噪声越来越大。

D-SPIN 的核心 insight：**perturbation 不是噪声，是信号**。你需要一个数学框架显式建模"perturbation 如何作用于 network"，才能提取出 perturbation 携带的 network 信息。

参考 GENIE3 原始 paper: https://doi.org/10.1371/journal.pone.0012776
参考 BEELINE benchmarking: https://doi.org/10.1038/s41592-019-0690-6

---

## Spin Network：为什么是这块数学

D-SPIN 借用了统计物理里的 Ising model。这不是 fancy 包装，而是这个问题的**自然语言**。

想象每个 gene 是一个小磁针，可以指上（+1，activated）、指下（-1，inhibited）、或躺平（0，basal）。Gene 之间的 interaction 就是磁针之间的耦合——$J_{ij} > 0$ 让它们想对齐，$J_{ij} < 0$ 让它们想反向。

整个 cell population 是一堆磁针构型，按 Boltzmann 分布采样。低 energy 的构型出现概率高。Perturbation 就是外加磁场 $\mathbf{h}^{(n)}$，把某些磁针往上或往下拽，整个分布就 shift 了。

为什么这个 framing 好？

**第一，它天然是 generative**。你 infer 出 $\mathbf{J}$ 和 $\mathbf{h}^{(n)}$ 后，可以 sample 出 cell state distribution。GENIE3 做不到——它只给你一个 edge list，没法 simulate。这就像给你一张地图 vs 给你一个能生成城市的 simulator。

**第二，它是 maximum entropy model**。给定你观察到的 mean 和 pairwise correlation，spin network 是假设最少的 model。它不 invent 你没观察到的 high-order interaction。这跟 Occam's razor 对齐，也解释了为什么它 generalize 好——在 drug dataset 上用 50% data 训练，另外 50% 的 cell state distribution 还能 high fidelity 重建。

**第三，perturbation integration 有数学必然性**。Log-likelihood gradient 是 data correlation 减 model correlation。每个 perturbation condition 贡献一个 gradient term，所有 condition 一起 push $\mathbf{J}$ 往能 explain 所有 condition 的方向走。这不是 heuristic，是 maximum likelihood 的数学结构决定的。

参考 Jaynes 1957 maximum entropy: https://doi.org/10.1103/PhysRev.106.620
参考 Hopfield 1982 neural network as spin glass: https://doi.org/10.1073/pnas.79.8.2554

---

## $J$ 不变 vs $h$ 变：这个 factorization 是灵魂

D-SPIN 最关键的设计 choice：**network $\mathbf{J}$ 在所有 perturbation 下不变，只有 perturbation response $\mathbf{h}^{(n)}$ 随 condition 变**。

这听起来 restrictive，但它是让 model tractable 和 interpretable 的根基。

类比：你有一台钢琴（network），每个键按下去的声音是固定的。不同 perturbation 是不同的弹奏方式（手指力度 $\mathbf{h}$）。你听够多首曲子，就能反推钢琴的 internal structure（哪些键共鸣、哪些键互相 dampen）。

如果钢琴本身每首曲子都在变（epigenetic reorganization），你就无法 disentangle "钢琴结构" 和 "弹奏方式"。

这个 factorization 让 D-SPIN 能把 3136 个 perturbation condition 的信息**全部灌进一个统一的 $\mathbf{J}$**。每个 condition 单独看信息量很小（K562 Perturb-seq 平均每个 knockdown 只有 ~200 cells），但拼在一起就足够 constrain $\mathbf{J}$。

这也是为什么 D-SPIN 比 linear regression 强。LR 只看 perturbation → average expression 的 input-output map，忽略了 cell 之间的 covariation。在 perturbation 数量少时，LR accuracy 远低于 D-SPIN，因为 D-SPIN 额外利用了 gene-gene covariation 信息。当 perturbation 数量足够大（800+），LR 追上来了——说明信息源从 covariation 转移到了 average response。这个 transition 很漂亮。

---

## Perturbation 是信号不是噪声：一个 thought experiment

Paper 的 four-pathway toy model（Figure 1A）值得反复琢磨。

四个 pathway A-B-C-D。A3 抑制 B3，D3 抑制 C3，B3 和 C3 互相抑制。Wild-type 下 B3 被 A3 和 C3 双重压制，expression 几乎为零，跟 A3 的 correlation 接近零。

GENIE3 跑这个数据：A3-B3 edge 隐形。B3-C3 edge 隐形。D3-C3 edge 隐形。因为它只看 wild-type correlation。

D-SPIN 跑这个数据，加上 perturbation：knockdown C3，B3 就活了，A3-B3 的负 correlation 立刻显现。knockdown A3，B3 活了，B3-C3 的负 correlation 显现。D-SPIN 的 global model 把这些 conditional observation 缝成一个 coherent network，三条 hidden edge 全部恢复。

**这就是 perturbation integration 的 power**。每个 perturbation 像 flashlight 照亮 network 的一角，D-SPIN 把所有 flashlight 的 illumination 拼成完整地图。

这个 insight 对实验设计有直接 implication：combinatorial perturbation 比 single-gene perturbation 信息量更大。Paper 在 synthetic network 上验证，random combinatorial perturbation（每个 gene 随机 activate 或 inhibit）比 single-node perturbation 能更准确 reconstruct network。这是对未来 Perturb-seq 实验设计的 actionable 建议。

参考 compressed Perturb-seq (Yao et al. 2024): https://doi.org/10.1038/s41587-023-01964-9
参考 Pathway Sculptor combinatorial perturbation (Gu et al. 2025): https://doi.org/10.1101/2025.06.15.659618

---

## 三种 inference 算法的分工

D-SPIN 有三个 engine，对应不同 scale：

**Exact maximum likelihood**：暴力枚举所有 $3^M$ 个 state。$M=10$ 时 $3^{10} \approx 60000$，可行。用于 toy model 验证 theory。

**MCMC maximum likelihood**：Gibbs sampling 估计 model expectation。$M=30-50$ 可行。用于 program-level network（30 gene programs）。这个是 generative 的——你能 sample cell state distribution。

**Pseudolikelihood**：用 product of conditional distribution 代替 joint distribution，绕开 partition function $Z$ 的指数爆炸。$M=1000+$ 可行，millions of cells 也跑得动。这个能 infer directed network，但不是 generative（无法 sample joint distribution）。

这个分工很务实。Program-level 用 MCMC 保持 generative 能力，gene-level 用 pseudolikelihood 换 scalability。两者回答不同问题：program-level 回答 "pathway 之间怎么 coordinate"，gene-level 回答 "具体哪个 gene 是 hub、哪个 gene 是 drug target 的 effector"。

Pseudolikelihood 能 infer direction 是个 bonus。Conditional distribution $P(s_k | s_{\backslash k})$ 本质是 regression——用其他 gene 预测 gene $k$。如果 A 预测 B 比 B 预测 A 好，方向就是 A→B。但 paper 主要分析 undirected network，因为有 feedback loop 的 directed graphical model 无法定义 consistent stationary distribution（Judea Pearl 的经典结果）。Cell 里 feedback loop 太多了。

参考 Besag 1974 pseudolikelihood: https://doi.org/10.1111/j.2517-6161.1974.tb00999.x
参考 Pearl 1987 cyclic dependency inconsistency: https://doi.org/10.1016/0004-3702(87)90012-9

---

## K562 实验：D-SPIN 发现了 DE analysis 错过的东西

K562 是 chronic myelogenous leukemia cell line，能分化成 erythroid 或 myeloid。Replogle 2022 的 genome-wide Perturb-seq 跑了 9867 个 gene knockdown、200 万 cell。

传统 DE analysis 找 erythroid fate regulator：只找到 GATA1。Myeloid：一个都没找到。

D-SPIN 找到：
- Erythroid: KLF1、NFE2、GFI1B、GATA1
- Myeloid: SPI1 (PU.1)、MEF2C
- 双向 inhibitor: NPM1

为什么差距这么大？DE analysis 每次只看一个 perturbation，每个 knockdown 只有几百个 cell，noise 极大，effect 被淹没。D-SPIN 把 3136 个 perturbation 的信息全灌进一个 unified model，每个 perturbation 都贡献一点关于 $\mathbf{J}$ 的信息，拼起来就够 constrain 出准确 network。

NPM1 的发现特别精彩。BCR-ABL1 kinase 激活 NPM1，NPM1 把 KLF1 和 SPI1 从 nucleus 拽到 cytoplasm，让它们无法执行 fate control。所以 knockdown KLF1 或 SPI1 的 phenotype 很弱——它们本来就被 NPM1 隔离了。DE analysis 看 KLF1 knockdown 没效果，就以为 KLF1 不是 regulator。D-SPIN 通过 global network model 发现 NPM1 是 KLF1/SPI1 的 upstream inhibitor，才把这条 hidden regulatory axis 挖出来。

这种 posttranscriptional regulation（protein relocalization）是 motif-based method（SCENIC+、CellOracle）根本看不到的，因为它们只看 TF-promoter binding。D-SPIN 不依赖 prior knowledge，直接从 perturbation response data 推断，所以能发现 non-transcriptional regulation。

参考 Replogle 2022 Perturb-seq: https://doi.org/10.1016/j.cell.2022.05.013
参考 Zahran 2023 NPM1 relocalization: https://doi.org/10.1182/blood-2023-187016
参考 CellOracle: https://doi.org/10.1038/s41586-022-05688-9
参考 SCENIC+: https://doi.org/10.1038/s41592-023-01938-4

---

## 四种 Homeostatic Strategy：Cell 的 Distributed Control

K562 perturbation response 最 philosophically interesting 的发现：cell 用四种 global strategy 应对 perturbation。

Knockdown RNA polymerase subunit → cell upregulate metabolism、downregulate translation 和 degradation。Knockdown ribosomal subunit → cell downregulate translation、upregulate degradation。Knockdown translation initiation factor → cell upregulate translation、downregulate degradation。

注意 pattern：**compensatory function 跟被 knockdown 的 gene 没有直接关联**。RNA polymerase 坏了，cell 去 upregulate metabolism？这说明 cell 内部有 long-range feedback 连接 distinct cellular process。

更 striking 的是 ribosomal subunit vs translation initiation factor：都是 translation 相关 gene，但 knockdown 后触发相反 strategy。Ribosome 坏了 → downregulate translation、upregulate degradation（把坏掉的 protein 清掉）。Initiation factor 坏了 → upregulate translation（compensate 减少 translation）。

这暗示 cell 有非常精细的 sensing 和 control 机制，能区分 "哪个 sub-process 坏了" 并触发针对性 response。这跟工程里的 distributed control system 概念一样——没有 central controller，每个 module 自己 sense 和 respond，但通过 inter-module communication 实现 global coordination。

D-SPIN 的 value：它把 millions of cell、thousands of perturbation 组织成四个可解释的 strategy class，让你一眼看出 cell 的 control logic。UMAP 做不到这个——UMAP 只给你一团点云，不告诉你 "为什么"。

---

## Immune Cell Drug Screen：D-SPIN 作为 Drug Classification Tool

Paper 自己做的新实验：PBMC + anti-CD3/CD28 激活 T cell + 502 种 drug，1.5 million cell，28 种 cell state。

D-SPIN 把 502 种 drug 按 transcriptional effect 分成 7 类：strong inhibitor、weak inhibitor I/II、glucocorticoid、M1 macrophage inducer、epigenetic modifier、toxicant。

每类有 distinct 的 program-level signature。Strong inhibitor（dasatinib、tacrolimus）完全阻断激活。GC（halcinonide、dexamethasone）抑制激活但额外诱导 M2 macrophage。M1 inducer（TLR7/8 agonist）诱导 pathogen-responsive macrophage。Toxicant（bortezomib）强激活 stress response。

Gene-level network 更精细。657 个 regulatory gene（TF + kinase + phosphatase），D-SPIN 找到 41 个 hub，主要是 Src-family kinase（LYN、FYN、HCK、SYK、LCK、BTK、FGR）——这些是 TCR 下游最早激活的分子。

**关键对比**：在 immune activation context 下，TF motif prior 只 marginal 提升 accuracy。因为 signaling transduction 是 phosphorylation mediated，motif-based method 看不到。D-SPIN 直接从 perturbation response data 推断，能捕捉 kinase/phosphatase interaction。这是 D-SPIN 相对 SCENIC+/CellOracle 的根本优势。

Drug signature 也很 actionable：
- GC 独有：KLF9、TSC22D3、MAFB、DUSP1/2（M2 polarization + GC receptor）
- Src inhibitor 额外抑制：NFKBIA、IL1B、EGR2（inflammation）
- mTOR inhibitor 激活 EEF1A1（translation elongation factor，潜在 compensatory feedback）
- Topoisomerase inhibitor 激活 ATF3、MDM2（p53 pathway）

这些 signature 可以指导 drug combination design 和 biomarker 开发。

---

## Drug Combination：Additive Recruitment 是主旋律

这是 paper 最实用的部分。10 种 drug 的所有 pairwise combination，84% 是 additive 或 subadditive。

**Additive 意味着什么**？两个 drug 作用于相同 set of gene program，combination effect ≈ 单药 effect 之和。D-SPIN 的 response vector $\mathbf{h}$ 是线性叠加的，所以 additive 自然 emerge。

深入案例：GC halcinonide + Src inhibitor dasatinib。两个都是 anti-inflammatory drug，都抑制 activation、诱导 M2。但 intensity 不同：GC 强诱导 M2、弱抑制 activation；dasatinib 强抑制 activation、弱诱导 M2。

Combination 产生 hyper-suppressed M2 state——M2 program 被 super-activate，activation program 被 super-repress。这是 emergent combinatorial state，单药都产生不了。

**为什么是 additive 而非 synergistic**？Gene-level network 显示两个 drug 走 convergent 但 independent pathway：
- GC: TSC22D3、DUSP1、CEBPD、MAFB（GC receptor → M2）
- Dasatinib: IRF1 inhibition + CSF1R/ACP5 activation（IRF1 是 M1 controller，inhibit 后 bias 向 M2）

两条 pathway 独立调节同一个 M2 program，所以 effect 叠加。如果它们 share 下游 effector，就会 subadditive（saturation）。如果 cross-potentiate，就 synergistic。D-SPIN 的 network model 让你能区分这些机制。

参考 Geva-Zatorsky 2010 单蛋白 level additivity: https://doi.org/10.1016/j.cell.2010.02.011

---

## Dosage Interpolation：一个数据点解锁整个 Phase Diagram

最 magic 的结果。Dasatinib × halcinonide 的 5×6 dosage grid（30 个 combination condition）。

只用 single-drug dosage response：cosine similarity 0.72，很多 combination 预测很差（<0.5）。因为 single-drug data 无法 reveal drug-drug interaction parameter $\gamma$。

加 1 个 saturating dosage combination：cosine similarity 跳到 0.84。加 3 个：接近 full data 的 phase diagram。

模型是 additive sigmoid with multiplicative interaction：

$$h(c_1, c_2) = \text{sgm}_1(c_1) + \text{sgm}_2(c_2) + \gamma \cdot \text{sgm}_1(c_1) \cdot \text{sgm}_2(c_2)$$

Single-drug data fit 出 $\text{sgm}_1, \text{sgm}_2$（sigmoid 参数）。一个 combination data point fit 出 $\gamma$（interaction strength）。然后整个 dosage space 都能插值。

Phase diagram 显示 macrophage state 在 dosage space 上的 smooth transition：
- Low dasatinib + 增加 halcinonide：activated macrophage → M2
- High dasatinib + 增加 halcinonide：resting monocyte → inhibited monocyte → hyper-inhibited M2

**只有 combination 才出现的 emergent state（hyper-inhibited M2）**，single-drug data 无法预测，但加一个 combination point 就能识别。

这对 drug development 有直接 implication：你不需要跑完所有 dosage combination（$N \times M$ 个实验），只需要跑 single-drug dosage response + 极少数 combination，就能 map out 整个 combination landscape。这在实验成本上是数量级的节省。

---

## 为什么 Equilibrium Model 能描述 Non-Equilibrium Cell

Paper 末尾提了一个 deep question：cell 显然是 far-from-equilibrium dynamical system，为什么 equilibrium spin network 能 low-error 重建它的 state distribution？

可能的答案：cell 在某些 timescale 上可被 effective 视为 equilibrium system。Perturbation 把 cell 推到新的 basin，cell 在新 basin 内部 equilibrates（相对于 perturbation timescale）。这跟 Waddington epigenetic landscape 的 intuition 一致——cell rolling down landscape，在每个 attractor basin 内部近似 equilibrium。

这跟 machine learning 里的 energy-based model（EBM）思路相通。EBM 也是 equilibrium model，但能 model 非常复杂的 data distribution。D-SPIN 是 gene expression space 上的 EBM，perturbation 是 conditional input。

如果这个 equilibrium approximation 成立，它给了一个 simplifying principle：cell 的 transcriptional state distribution 可以被一个 energy function（pairwise interaction 足矣）capture，不需要 model full dynamics。这跟 statistical physics 里 mean-field theory 的成功类似——虽然真实 system 复杂，但 effective description 可以很简单。

参考 Teschendorff & Feinberg 2021 statistical mechanics meets single-cell: https://doi.org/10.1038/s41576-021-00341-z
参考 Lang et al. 2014 epigenetic landscape as spin glass: https://doi.org/10.1371/journal.pcbi.1003734

---

## 跟 Deep Learning 的关系

Paper 讨论了 D-SPIN vs scGPT / cell atlas foundation model。

两者目标类似：从 large data 学习 cell representation 并 generalize。但路径完全不同：

scGPT 用 transformer architecture，millions of parameter，feedforward，黑箱。它能 predict perturbation response，但不告诉你 "为什么"。你无法从 scGPT 的 attention weight 读出 gene-gene interaction 的物理意义。

D-SPIN 用 transparent graphical model，parameter 直接对应 biological interaction。你能指着 $J_{ij}$ 说 "这是 gene i 和 gene j 的 regulatory interaction"。你能指着 $\mathbf{h}^{(n)}$ 说 "这是 drug n 对 network 的作用 pattern"。

**Trade-off**：D-SPIN 的 expressiveness 受限于 pairwise interaction assumption。如果 biology 本质需要 high-order interaction（TF complex cooperativity、three-gene logic gate），D-SPIN 会 miss。Deep model 没 this limitation，但牺牲 interpretability。

我的 view：两者是 complementary。D-SPIN 适合 discovery phase——发现 hypothesis、identify key regulator、设计实验。Deep model 适合 scaling phase——当你有足够 data 且只关心 prediction accuracy 时。理想 pipeline 是 D-SPIN 先 build interpretable scaffold，deep model 再 refine 高维细节。

参考 scGPT: https://doi.org/10.1038/s41592-024-02201-0
参考 Heimberg et al. cell atlas foundation model: https://doi.org/10.1038/s41586-024-08411-y
参考 Bunne et al. 2024 Virtual Cell: https://doi.org/10.1016/j.cell.2024.11.015

---

## Limitations：诚实面对

1. **Pairwise only**。Real biology 有 high-order interaction。TF A 和 TF B 单独都没效果，合在一起激活 gene C——这种 AND gate D-SPIN 看不到。可以加 third-order term $J_{ijk} s_i s_j s_k$，但 parameter explosion。

2. **No dynamics**。D-SPIN 给 stationary distribution，不给 trajectory。你不知道 cell 从 state A 到 state B 走什么 path、花多长时间。这限制了 application 到 differentiation trajectory、reprogramming dynamics。

3. **$\mathbf{J}$ constant across conditions**。Differentiation、disease progression 中 epigenetic state变化会改变 network 本身。D-SPIN 假设 $\mathbf{J}$ 不变，适用于 perturbation response scenario，不适用于 long-term developmental process。

4. **Combination prediction from single perturbation is fundamentally hard**。D-SPIN 需要至少一个 combination data point 来 estimate interaction parameter $\gamma$。Ab initio 预测 $\gamma$ 需要分子层面的 protein-protein interaction 信息，这是 D-SPIN 当前 framework 不包含的。

---

## 我的 Takeaway

D-SPIN 让我想到你（Karpathy）常说的 "build intuition"。它的价值不只是 accuracy 数字，而是它给你的 mental model：

**Cell 是一块 circuit board，perturbation 是 probe，D-SPIN 是 oscilloscope**。

你 probe 够多次，就能 reverse engineer circuit board。而且这个 circuit board 是 generative 的——你能 simulate "如果我把这个 gene 拔掉会怎样"、"如果这两个 drug 一起上会怎样"。

最让我 excited 的不是单个 result，而是整个 framework 的 extensibility：
- 加 high-order term → capture TF cooperativity
- 加 time dimension → model dynamics
- 让 $\mathbf{J}$ condition-dependent → model epigenetic reorganization
- Multi-modal（加 protein、ATAC、spatial）→ richer state representation

D-SPIN 给了一个 solid mathematical skeleton，未来工作往上面挂 flesh 就行。这跟好的 architecture 设计一样——不是堆 feature，而是找到 right abstraction。

Tahoe-100M 那种 giga-scale perturbation atlas 是 D-SPIN 的天然 playground。一亿 cell、百万 perturbation，D-SPIN 的 parallel inference 能 handle。期待看到 D-SPIN 跑在那种数据上 reveal 什么 global principle。

参考 Tahoe-100M: https://doi.org/10.1101/2025.02.20.639398
D-SPIN GitHub: https://github.com/JialongJiang/DSPIN
Caltech DATA repository: https://doi.org/10.22002/2cjss-wgh69

---

# D-SPIN: 从 scRNA-seq Perturbation Data 构建 Gene Regulatory Network 的 Generative Model

## 1. Big Picture: 这篇 paper 在解决什么问题

Andrej 你好，这篇 paper 是 Caltech 的 Matt Thomson 实验室和 Rockefeller 的 Jialong Jiang 等人 2026 年发表在 Cell 上的工作。核心要解决的问题是：**如何从大规模 single-cell perturbation 数据（Perturb-seq、drug screen）中构建一个 mechanistically interpretable、generative 的 gene regulatory network (GRN) 模型**。

传统方法（GENIE3、GRNBoost2、PIDC、SCENIC+、CellOracle）存在几个根本性问题：
- 只用 stationary expression data，无法利用 perturbation 信息
- 不是 generative model（无法 simulate cell state distribution）
- 大多依赖 TF motif / ATAC-seq 信息，无法处理 posttranscriptional regulation、phosphorylation signaling
- 难以扩展到 millions of cells

D-SPIN 的核心洞察是：**perturbation 揭示了 hidden regulatory interactions**。在 wild-type 条件下，某些 gene-gene interaction 被其他 pathway 屏蔽（redundant inhibition），导致 correlation / mutual information 很低。Perturbation 打破这种 masking，让 hidden interaction 显现。但需要 global joint model 才能整合多个 perturbation 的信息。

参考链接：
- Paper: https://doi.org/10.1016/j.cell.2026.04.028
- GitHub: https://github.com/JialongJiang/DSPIN
- 原始 K562 Perturb-seq data: https://gwps.wi.mit.edu

---

## 2. 数学框架：Spin Network Model (Inverse Ising Problem)

### 2.1 模型定义

D-SPIN 借用统计物理中的 **Ising model / spin glass / Markov random field** 来建模 gene expression state。每个 gene（或 gene program）被 discretize 成 3 个状态：

$$s_i \in \{-1, 0, 1\}$$

这里 $s_i = -1$ 表示 inhibited，$s_i = 0$ 表示 unperturbed/basal，$s_i = 1$ 表示 activated。选择 3-state 而非 binary 的原因：
- $m=2$ 不足以描述 drug profiling 中观察到的不同 activation levels（比如 GC 和 strong inhibitor 都激活 M2 macrophage program，但程度不同）
- 3-state 让 self-interaction $J_{ii}$ 有更清晰的 biological interpretation：$J_{ii} > 0$ 类似 bistable switch，$J_{ii} < 0$ 类似 negative feedback
- 计算复杂度可控

完整的概率分布：

$$P(\mathbf{s}) = \frac{1}{Z} \exp\left[-E(\mathbf{s}; \mathbf{J}, \mathbf{h}^{(n)})\right]$$

$$E(\mathbf{s}; \mathbf{J}, \mathbf{h}^{(n)}) = -\sum_{i \leq j} J_{ij} s_i s_j - \sum_i h_i^{(n)} s_i$$

$$Z = \sum_{\mathbf{s}} \exp\left[-E(\mathbf{s}; \mathbf{J}, \mathbf{h}^{(n)})\right]$$

**变量含义**：
- $\mathbf{s} = (s_1, s_2, \ldots, s_M)$：M 个 gene/program 的 discretized state vector
- $\mathbf{J} \in \mathbb{R}^{M \times M}$：pairwise interaction matrix（symmetric for undirected network），$J_{ij} > 0$ 表示 co-activation，$J_{ij} < 0$ 表示 mutual inhibition
- $\mathbf{h}^{(n)} \in \mathbb{R}^M$：condition $n$ 的 perturbation response vector（external bias field），$h_i^{(n)} > 0$ 表示该 condition 倾向于 activate node $i$
- $Z$：partition function，归一化常数

**关键假设**：$\mathbf{J}$ 在所有 perturbation 下保持不变（unified network），只有 $\mathbf{h}^{(n)}$ 随 condition 变化。这是一个非常强的 assumption，类似于 Hopfield network 的 "associative memory" 概念——不同 perturbation 激活 network 中预先存在的不同 attractor states。

参考 Hopfield 1982: https://doi.org/10.1073/pnas.79.8.2554

### 2.2 为什么用 Spin Network / Maximum Entropy Model

从统计角度看，spin network 是给定 mean 和 pairwise cross-correlation 的 **maximum entropy model**（Jaynes 1957）。这意味着它对未测量的统计量做最少假设：

$$\max_{P} H(P) \quad \text{s.t.} \quad \langle s_i \rangle_P = \text{data mean}, \quad \langle s_i s_j \rangle_P = \text{data correlation}$$

解出来就是 Boltzmann distribution 形式。这个性质非常重要：模型不会 overfit 到没观察到的 high-order interaction。

从物理角度看，spin network 是 equilibrium model，定义了一个 **energy landscape**。Perturbation vector $\mathbf{h}^{(n)}$ "tilt" 这个 landscape，把 cell population 推向不同的 basin of attraction。这给出了一个优美的 visual metaphor：cell population 是 landscape 上的一群点，perturbation 改变 landscape 的形状，cell 跟着 reposition。

### 2.3 Inference: 梯度上升

通过最大化 log-likelihood：

$$\log \mathcal{L}(\mathbf{J}, \mathbf{h}^{(n)}) = \sum_n \sum_{c \in D_n} \log P(\mathbf{s}^{(c)} | \mathbf{J}, \mathbf{h}^{(n)})$$

梯度：

$$\frac{\partial \log \mathcal{L}}{\partial J_{ij}} = \frac{1}{N} \sum_n \left(\langle s_i s_j \rangle_{\text{Data}}^{(n)} - \langle s_i s_j \rangle_{\text{Model}}^{(n)}\right)$$

$$\frac{\partial \log \mathcal{L}}{\partial h_i^{(n)}} = \langle s_i \rangle_{\text{Data}}^{(n)} - \langle s_i \rangle_{\text{Model}}^{(n)}$$

非常 elegant：训练过程就是 **matching pairwise correlation 和 single-node mean** between data 和 model。这正是 Boltzmann machine learning 的经典 form。

**Convexity**：这个优化问题是 concave（唯一的 local max = global max），所以不需要 simulated annealing 之类的 trick。但 NP-hard 的原因是 partition function $Z$ 的计算和 model identifiability。

---

## 3. 三种 Inference 算法的 Trade-off

D-SPIN 实现了三种 inference 算法，对应不同的规模和需求：

### 3.1 Exact Maximum Likelihood（小网络，~10 nodes）

直接枚举所有 $3^M$ 个 state 计算 $Z$。$3^{10} \approx 6 \times 10^4$，可行。适用于 toy model（如 four-pathway model 验证）。

### 3.2 MCMC Maximum Likelihood（中等网络，30-50 nodes）

用 Gibbs sampling 估计 model expectation $\langle \cdot \rangle_{\text{Model}}$。每步采样一个 node $k$，按 conditional distribution 更新：

$$P(s_k | s_{\backslash k}, \mathbf{J}, \mathbf{h}) = \frac{\exp(s_k \theta_k + s_k^2 J_{kk})}{\exp(\theta_k + J_{kk}) + 1 + \exp(-\theta_k + J_{kk})}$$

其中 effective field：

$$\theta_k = h_k + \sum_{j \neq k} J_{jk} s_j$$

计算复杂度 $\Omega(M^5)$（主要是 mixing time 在 phase transition 附近指数增长）。适用于 program-level network（30 programs）。

### 3.3 Pseudolikelihood（大网络，1000+ nodes，millions of cells）

这是 D-SPIN scalability 的关键。Pseudolikelihood 用 product of conditional distributions 代替 joint distribution：

$$\text{Pseudo}P(\mathbf{s} | \mathbf{J}, \mathbf{h}) = \prod_k P(s_k | \mathbf{s}_{\backslash k}, \mathbf{J}, \mathbf{h})$$

这避免了 partition function $Z$ 的指数复杂度。梯度计算 $\mathcal{O}(M^2)$，可并行化。

**Bonus**：Pseudolikelihood 形式类似 regression，可以推断 **directed interaction**。如果 gene A 预测 gene B 比 B 预测 A 更好，则方向是 A → B。Directed gradient：

$$\frac{\partial \log \mathcal{L}_{\text{Pseudo}}}{\partial J_{ij, i \neq j}} = s_i s_j - s_i \frac{\exp(\theta_j + J_{ij}) - \exp(-\theta_j + J_{ij})}{\exp(\theta_j + J_{ij}) + 1 + \exp(-\theta_j + J_{ij})}$$

**为什么 paper 主要用 undirected network？** 因为 directed graphical model 在有 feedback loop 时无法定义 consistent stationary distribution（Judea Pearl 的经典结果）。Cell 中 feedback loop 很常见（如 PU.1-GATA1-GATA2-PU.1 in HSC），所以 undirected 更合适。

参考：
- Besag 1974 pseudolikelihood: https://doi.org/10.1111/j.2517-6161.1974.tb00999.x
- Ravikumar 2010 Ising model selection: https://doi.org/10.1214/09-AOS691
- Pearl 1987 cyclic dependency: https://doi.org/10.1016/0004-3702(87)90012-9

---

## 4. Perturbation Integration: 核心 Insight

### 4.1 Hidden Interaction 问题

Paper 用一个 four-pathway toy model 举例（Figure 1A）。Pathway A 和 D 分别 inhibit pathway B 和 C 中的 B3、C3。在 wild-type 下，B3 持续被 suppress，所以 A3-B3 的 correlation 几乎为零，mutual information 也很低。GENIE3、PIDC、GRNBoost2 都无法识别这种 masked interaction。

**Perturbation 揭示 hidden interaction**：当 perturbation shut down pathway C，A3 对 B3 的 inhibition 就显现出来，correlation 显著增强。但需要 global model 才能整合多个 perturbation 的信息——每次 perturbation 只 reveal 一部分 interaction。

### 4.2 D-SPIN 的优势

在 HSC synthetic network benchmarking 上：
- D-SPIN top-10 edge accuracy: 0.96
- PIDC/GRNBoost2/GENIE3: 0.77-0.83
- D-SPIN AUPRC: 0.87 vs others 0.72-0.82

在 directed network inference 上差距更大：
- D-SPIN AUPRC: 0.77
- Others: 0.47-0.57

在 large-scale synthetic network（125-1000 nodes，三种 topology：modular、Erdős-Rényi、scale-free）：
- 1000-node modular network: D-SPIN 0.913 vs best other 0.537
- 1000-node ER: 0.773 vs 0.505
- 1000-node scale-free: 0.721 vs 0.498

**关键 finding**：D-SPIN 的 accuracy 随 perturbation 数量增加而提升（0.415 → 0.930 with 800 perturbations）。而其他方法在 low perturbation number 时 accuracy 反而下降，因为它们无法区分 expression change 是来自 internal regulation 还是 external perturbation。

### 4.3 Computational Efficiency

D-SPIN 在大规模数据上比其他方法快几个数量级。256,000 cells on 2 CPU cores：D-SPIN 6 小时，其他方法一周内跑不完。原因是 pseudolikelihood 的 $\mathcal{O}(M^2)$ 复杂度 + 并行化（每个 condition 独立计算 gradient）。

---

## 5. 应用一：K562 Genome-wide Perturb-seq

### 5.1 Dataset

K562 是 chronic myelogenous leukemia cell line，有 erythroid/myeloid 双向分化潜能。Replogle et al. 2022 的 genome-wide Perturb-seq：
- 9,867 gene knockdown
- 2 million cells
- 经过 filtering：3,136 perturbations with >10 DEGs 和 >20 cells，0.6 million cells

### 5.2 与 ChIP-seq 的对应

D-SPIN 在 TFs+500/1000/1500 HVGs 三个 dataset 上都取得 best correspondence with ChIP-seq（early precision rate 11-15x over random）。加入 TF motif prior（来自 CellOracle base GRN）后 accuracy 进一步提升。值得注意的是 z-score / DE method 在 Perturb-seq 上表现差，因为 single-cell noise 大。

### 5.3 Program-level Network（30 programs）

用 oNMF（orthogonal non-negative matrix factorization）把 transcriptome coarse-grain 成 30 个 gene programs。oNMF 的优势：
- Non-negativity：避免 PCA 负 weight 的解释困难
- Orthogonality：program 不重叠，避免 confounding interaction

30 个 programs 包含 core cell biology（transcription、translation、RNA processing、mitosis）和 lineage-specific（erythroid HBG1/HBG2/HBZ/GYPA、myeloid phagosome ACTB/ARPC3、immune-response LAPTM5/RAC2）。

Network 用 Leiden community detection 分成 7 个 modules。负 interaction 主要出现在 mutually exclusive cell states 之间（如 P29 spindle microtubule vs P25 DNA replication，P4 erythroid vs P6 phagosome）。

### 5.4 识别 Erythroid/Myeloid Fate Regulators

传统 DE analysis 只找到 GATA1。D-SPIN 找到：
- Erythroid: KLF1, NFE2, GFI1B, GATA1
- Myeloid: SPI1 (PU.1), MEF2C
- 双向 inhibitor: NPM1

NPM1 的发现特别有意思：BCR-ABL1 kinase 激活 NPM1，把 KLF1 和 SPI1 从 nucleus 重定位到 cytoplasm，silencing 它们的 fate control 功能。这解释了为什么 KLF1/SPI1 的 knockdown phenotype 不明显——它们本来就被 NPM1 隔离在 cytoplasm 里。这种 posttranscriptional regulation 无法被 motif-based method 发现。

参考 Zahran 2023: https://doi.org/10.1182/blood-2023-187016

### 5.5 四种 Homeostatic Response Strategy

这是 paper 最 philosophically interesting 的部分。通过 coarse-graining（把 programs 分成 modules，把 perturbations 分成 strategies），发现 K562 在 perturbation 下用四种 global strategy 维持 homeostasis：

| Strategy | Upregulated Module | Downregulated Modules | 触发 perturbation |
|----------|-------------------|----------------------|------------------|
| Metabolism upregulation | Metabolism | Translation, Degradation | RNA polymerase, TFIID, Mediator |
| Transcription upregulation | Transcription | 其他 | mTOR components |
| Translation upregulation | Translation | Degradation | Translation initiation factors |
| Degradation upregulation | Degradation + Metabolism | Translation | Ribosomal subunits, rRNA |

**关键 insight**：compensatory function 通常 NOT 直接关联被 knockdown 的 gene。比如 knockdown RNA polymerase subunit → upregulate metabolism。这暗示 cell 内部有 long-range regulatory feedback 连接 distinct cellular processes。knockdown 翻译起始因子 → upregulate translation、downregulate degradation；knockdown ribosomal subunit → downregulate translation、upregulate degradation。同样是 translation 相关 gene，根据具体 sub-process 不同触发相反 strategy。

---

## 6. 应用二：Human Immune Cell Drug Response

### 6.1 实验设计

这是 paper 自己做的新实验：
- PBMCs from healthy donor，anti-CD3/CD28 antibody 激活 T cell
- 502 small molecule drugs（mTOR、MAPK、GC、JAK/STAT、HDAC inhibitors 等）
- 1.5 million filtered cells
- 1,200+ conditions
- 28 immune cell states（5 CD4 T、10 CD8 T、1 NK、4 B、8 myeloid）

Time-course 实验显示 T cell 2 小时内激活，myeloid cell 16 小时激活——说明 myeloid 激活是 T cell-driven 的 paracrine signaling。

### 6.2 Program-level Drug Network

30 gene programs，包括 P6 T cell、P15 B cell、P14 NK cell、P20 myeloid cell，以及 M1/M2 macrophage 等状态 program。Network model 只用 465 个 interaction parameter + 每 condition 30 个 response parameter，就能 high fidelity 重建 cell state distribution（92.4% samples cosine similarity >90%）。

### 6.3 七个 Drug Phenotypic Classes

| Class | 代表 drug | 转录组效应 |
|-------|----------|----------|
| Strong inhibitor | Dasatinib, tacrolimus, cyclosporine | 完全阻断激活，回到 resting-like state |
| Weak inhibitor I | Nilotinib | 轻微增加 resting state 比例 |
| Weak inhibitor II | Temsirolimus | 类似 weak I |
| Glucocorticoid (GC) | Halcinonide, budesonide, dexamethasone | 抑制激活 + 强诱导 M2 macrophage (CD163) |
| M1 macrophage inducer | Vesatolimod, resiquimod, motolimod (TLR7/8 agonist) | 诱导 pathogen-responsive M1 状态 |
| Epigenetic modifier | HDAC inhibitors (vorinostat) | 诱导 CD8 T epi. 状态（histone + TOP2A） |
| Toxicant | Bortezomib, panobinostat, 10-hydroxycamptothecin | 强激活 stress response P30 |

### 6.4 Gene-level Network: Signaling Hubs

657 regulatory genes（399 TFs、187 kinases、71 phosphatases）。D-SPIN 找到 41 个 hub gene，主要是 Src-family TKs：LYN、FYN、HCK、SYK、LCK、BTK、FGR。这些是 TCR 下游最早激活的分子。还包括多个 DUSP family（MAPK phosphatase）。

**重要**：在 immune activation context 下，TF motif prior 只 marginal 提升 accuracy。这证明 D-SPIN 作为 data-driven、perturbation-based method 的优势——signaling pathway（phosphorylation）是 motif-based method 无法捕捉的。

### 6.5 Gene-level Drug Signature

Hierarchical clustering 把 inhibitor 分成 strong/weak/GC 三大类，strong inhibitor 再细分（Src、JAK、calcineurin）。关键 signature gene：
- GC 独有：KLF9、TSC22D3、MAFB、NEAT1、DUSP1/2（M2 polarization + GC receptor signaling）
- Strong inhibitor：STAT1/3、JAK3、IRF1/4/7/9 repression
- Src inhibitor 额外：NFKBIA、IL1B、EGR2 repression
- mTOR inhibitor (rapamycin)：EEF1A1 activation（潜在 compensatory mechanism）
- Topoisomerase inhibitor：ATF3、MDM2、PHPT1（p53 pathway）

---

## 7. Drug Combination: Additive Recruitment of Gene Programs

### 7.1 实验设计

10 个 drugs 的所有 pairwise combination（Figure S7A）。发现 **84% 的 drug interaction 是 additive 或 subadditive**（在 gene program level）。其他类型：dominant、synergistic、antagonistic。

### 7.2 Halcinonide + Dasatinib 产生新 Macrophage State

GC halcinonide 和 Src inhibitor dasatinib 都是 anti-inflammatory drug，作用于相同 set of gene programs 但 intensity 不同：
- Halcinonide 弱抑制 macrophage activation (IDO1, CD40, SLAMF7)，强激活 M2 (CD163, MS4A6A, VSIG4)
- Dasatinib 强抑制 activation，弱激活 M2
- Combination: **additive recruitment** → hyper-suppressed M2 state

**关键 insight**：两个 drug 通过不同 gene regulator 达到相同 program 效果：
- GC: TSC22D3、DUSP1、CEBPD、MAFB（GC receptor signaling + M2 polarization）
- Src inhibitor: IRF1 inhibition + CSF1R、ACP5 activation（IRF1 是 M1 controller，inhibition → bias toward M2）

这解释了为什么 combination 是 additive 而非 synergistic——它们走 convergent 但 independent 的 pathway。这给 drug combination design 提供了 conceptual framework：可以通过 fine-tune dosage 操控 macrophage state spectrum。

参考 Geva-Zatorsky 2010 单蛋白 level additivity: https://doi.org/10.1016/j.cell.2010.02.011

### 7.3 Dosage Combination Interpolation

这是 D-SPIN generative power 最 impressive 的展示。**只用 single-drug dosage response + 一个 saturating dosage combination condition**，就能定量插值未观察到的 dosage combination。

模型：additive sigmoid with multiplicative interaction

$$h(c_1, c_2) = \text{sgm}_1(c_1) + \text{sgm}_2(c_2) + \gamma \cdot \text{sgm}_1(c_1) \cdot \text{sgm}_2(c_2)$$

其中 $c_1, c_2$ 是 log-dosage，$\text{sgm}_i$ 是 sigmoid curve，$\gamma$ 是 interaction strength。

具体形式：

$$h(c_1, c_2) = \alpha + \frac{\beta_1}{1+\exp[-\kappa_1(c_1 - c_1^*)]} + \frac{\beta_2}{1+\exp[-\kappa_2(c_2 - c_2^*)]} + \frac{\gamma}{(1+\exp[-\kappa_1(c_1-c_1^*)])(1+\exp[-\kappa_2(c_2-c_2^*)])}$$

**变量含义**：
- $\alpha$：baseline response
- $\beta_1, \beta_2$：两个单药的最大 effect size
- $\kappa_1, \kappa_2$：sigmoid 的 steepness
- $c_1^*, c_2^*$：EC50（半效浓度）
- $\gamma$：药物相互作用强度

**结果**：
- 0 个 combination condition: average cosine similarity 0.72（很多 combination <0.5）
- 1 个 combination condition: 0.84（大部分 >0.7）
- 3 个 combination condition: 接近 full data 的 phase diagram

### 7.4 Macrophage State Phase Diagram

D-SPIN 能画出 dasatinib × halcinonide dosage 2D phase diagram：
- Low dasatinib + 增加 halcinonide：activated macrophage → M2 macrophage
- High dasatinib + 增加 halcinonide：resting monocyte → inhibited monocyte → hyper-inhibited M2

只用 single-drug data 无法识别 hyper-inhibited M2 state（这是 combination 才出现的 emergent state），但加一个 combination condition 就能 qualitatively 恢复完整 phase diagram。这说明 **药物相互作用的 critical information 可以从极少数 combination experiment 提取**。

---

## 8. 与 Deep Learning / Foundation Model 的对比

Paper 末尾讨论了 D-SPIN 与 transformer-based cell model（如 scGPT、Clemberg et al. foundation model）的关系。两者都能从 large data 学习并 generalize，但：

- D-SPIN：透明 graphical model，parameter 直接对应 gene-gene interaction，可解释为 pathway/circuit
- Deep model：millions of parameter，feedforward 架构，无法 map 到 biochemical pathway

D-SPIN 不需要 ATAC-seq 等额外数据，特别适合 perturbation-based data日益增多（condition barcoding strategy）的时代。

参考：
- Bunne et al. 2024 Virtual Cell: https://doi.org/10.1016/j.cell.2024.11.015
- scGPT: https://doi.org/10.1038/s41592-024-02201-0
- Heimberg et al. cell atlas foundation model: https://doi.org/10.1038/s41586-024-08411-y

---

## 9. Statistical Physics 视角的 Deep Insight

Paper 提出一个深刻的问题：**为什么 equilibrium spin network model 能对 strongly non-equilibrium 的 cell 产生 low-error 重建？**

Equilibrium model 假设 system 在 energy landscape 上达到 Boltzmann distribution。但 cell 显然是 far-from-equilibrium 的 dynamical system。可能的解释：
- Cell 在某些 situation 下可被 effective 视为 equilibrium system driven through different states by perturbation bias
- 这是 simplifying principle，类似 Waddington epigenetic landscape
- 为更 global 的 gene regulation theory 提供路径

参考：
- Sokolik et al. 2015 TF competition: https://doi.org/10.1016/j.cels.2015.08.001
- Teschendorff & Feinberg 2021 statistical mechanics meets single-cell: https://doi.org/10.1038/s41576-021-00341-z

---

## 10. Limitations 和未来方向

1. **Only pairwise interaction**：没有 higher-order multi-body interaction（如 TF complex cooperativity）。可加入 third-order term $J_{ijk} s_i s_j s_k$。
2. **Equilibrium model, no dynamics**：无法模拟 trajectory，只能 stationary distribution。可扩展到 Glauber dynamics、Sompolinsky-Zippelius spin glass dynamics。
3. **J assumed constant across conditions**：在 differentiation、disease progression 中 epigenetic reorganization 会改变 network 本身。可让 $J$ 也 condition-dependent。
4. **Combination prediction from single perturbation only is hard**：ab initio 预测非-additive interaction 需要更详细的 molecular characterization。

---

## 11. 我的 Intuition 总结

D-SPIN 的 elegant 之处在于把 inverse Ising problem（统计物理经典问题）应用到 single-cell perturbation biology，并巧妙 factorize 成 $J$（不变 network）+ $\mathbf{h}^{(n)}$（condition-specific bias）。这个 factorization 让模型：
- 能整合 thousands of perturbation 的信息（每个 perturbation 都 leak 一点 network 信息）
- Generative（能 sample cell state distribution）
- Interpretable（$J_{ij}$ 直接读作 gene-gene interaction）
- Scalable（pseudolikelihood 到 1000 nodes, millions of cells）
- Predictive（插值未观察 condition）

最让我兴奋的是 drug combination dosage interpolation 的部分——**只用 1 个 combination data point 就能恢复完整 phase diagram**，这暗示 biological system 的 combinatorial perturbation response 有非常低 effective dimensionality，可以用 sparse parameterization 捕捉。这对 drug development 有直接实用价值。

从 build intuition 角度，D-SPIN 给出的 mental model 是：**cell population 是 energy landscape 上的一团 cloud，perturbation 通过 bias field $\mathbf{h}$ 倾斜 landscape，cloud 滚到新的 basin**。两个 drug combination 就是两个 bias field 叠加，如果它们走 independent pathway 就是 additive。这跟 Hopfield network 的 associative memory、Waddington epigenetic landscape、Sokolik-Thomson TF competition 都是一脉相承的 thinking。

期待看到 D-SPIN 应用到 Tahoe-100M 那种 giga-scale perturbation atlas 上，或者扩展到 multi-modal（加入 protein、ATAC、spatial）的版本。

参考：
- Tahoe-100M: https://doi.org/10.1101/2025.02.20.639398
- D-SPIN GitHub: https://github.com/JialongJiang/DSPIN
- Caltech DATA repository: https://doi.org/10.22002/2cjss-wgh69
