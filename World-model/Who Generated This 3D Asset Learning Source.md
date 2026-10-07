---
source_pdf: Who Generated This 3D Asset Learning Source.pdf
paper_sha256: 6c3b87ab05b30238ce0c4f1fd4e9725a0f5a68b1f669d33da02815daebe7c52c
processed_at: '2026-08-13T04:23:02-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话说清楚这论文在干嘛

你上网下载了一个 3D 模型，没人告诉你这玩意儿是哪个 AI 生成的。这篇 paper 做的事情就是：**给你一个 3D 模型，我能告诉你它是 DreamFusion 生成的还是 MVDream 生成的还是 Shap-E 生成的**——一共能区分 22 个 generator，准确率 97%。

这事在 2D 图像领域早就有人做了（GAN fingerprint，2019 年就开始了），但 3D 没人系统做过。这 paper 是第一个把这事搬到 3D 上、建了 benchmark、提了方法的。

参考：
- GAN attribution 鼻祖论文：https://arxiv.org/abs/1811.08170
- 这篇 paper 之前最接近的 3D 工作 FAKEPCD：https://dl.acm.org/doi/10.1145/3631823

---

## 为什么要做这事？

想象一下游戏公司、robotics lab、VR 内容工作室这些场景。现在 text-to-3D 和 image-to-3D 这么火，大量 AI 生成的 3D 模型在网上流通。问题来了：

- 某个 asset 被人 re-upload 到别的平台，metadata 没了，它到底是不是 AI 生成的？是哪个 AI？
- 某个 dataset 号称"人工建模"，实际上偷偷混了 AI 生成的，怎么审计？
- 你公司的 game asset pipeline 不小心混了来路不明的 3D 资产，版权怎么追？

这跟 fake image detection 类似，但维度更高、更难。2D fake detection 你看一张图就够了，3D 你光看一个视角的 render 看不出啥，得绕着看一圈、摸一下几何结构、扫一下频谱，才能抓住 generator 留下的"作案痕迹"。

---

## 难在哪？

### 难点 1：信号是散的

2D 图像的 fingerprint 集中在一张图里。3D 的 fingerprint 散落在：
- 多个视角的 rendering 上（看正面挺漂亮，绕到背面发现脸塌了）
- 几何结构上（mesh 的拓扑、顶点密度、normal 一致性）
- 频谱上（FFT spectrum 有 generator-specific 的 decay pattern）
- 视角之间的关系上（DreamFusion 前后视角 inconsistent，MVDream 一致）

你单看一个视角、单看 RGB、单看 geometry 都抓不全。得**联合起来看**。

### 难点 2：现实部署很糟糕

理想情况下你有大量 labeled data + 干净的 prompt。现实是：
- 新 generator 不断冒出来，你没几个 sample 可练
- 上传的人可能把 prompt 删了
- 网上既有 AI 生成的也有人工建模的，混在一起

所以 paper 默认的 evaluation 不是"full supervision + clean prompt"，而是**只有 1% 训练数据 + prompt 可能完全缺失**。这是 forensic deployment 的真实设定，不是 academic benchmark 的温室设定。

---

## Benchmark 长啥样？

他们基于 3DGen-Bench 和 Cap3D 搭了个 benchmark：

- **22 个 generator**：13 个 image-to-3D + 9 个 text-to-3D
  - Image-to-3D：Free3D, Escher-Net, Point-E, Triplane-Gaussian, Shap-E, SyncDreamer, GRM, LGM, Magic123, Zero123-XL, Stable Zero123, OpenLRM, Wonder3D
  - Text-to-3D：MVDream, Lucid-Dreamer, Magic3D, GRM, DreamFusion, Latent-NeRF, Shap-E, SJC, Point-E
- **1,900 个 prompt**，**10,851 个 synthetic asset**
- **440 个 real scanned asset**（来自 Cap3D，用于混合场景测试）
- 全部统一成 PLY 格式，render 成 4 个 canonical view

每个 asset 还附带两类 derived cue：
- **102 维 geometric descriptor**：包含 22 个 scalar（vertex/face 数、bbox、normal consistency、watertight、manifold...）+ 5 个 16-bin histogram（edge length、face area、curvature、surface distance、Laplacian eigenvalue）
- **256 维/view FFT feature**：每个 view 的 RGB 转灰度 → 2D FFT → log amplitude → fftshift → avgpool 到 16×16

为啥要搞 FFT？因为 2D fake detection 早就发现 GAN/diffusion 在高频有 characteristic artifact（up-convolution 的 checkerboard、diffusion 的 spectral decay）。3D render 出来的图继承了 generator 的 rendering pipeline，所以 FFT 也带 fingerprint。

参考：
- Frank et al. ICML'20 frequency analysis: https://proceedings.mlr.press/v119/frank20a.html
- Durall et al. CVPR'20 up-convolution spectral: https://arxiv.org/abs/2003.05585
- Corvi et al. CVPR'23 diffusion image properties: https://arxiv.org/abs/2304.10401

---

## 核心发现：generator 留下两种 fingerprint

### Fingerprint 1：Cross-view inconsistency

你绕着一个 3D 模型看一圈，不同 generator 表现很不一样：

- **DreamFusion**（SDS 鼻祖）：看不到的面会 collapse，背面经常是塌的或 Janus face（两个脸），因为它根本没有 explicit multi-view constraint，只对每个 view 单独做 score distillation
- **Shap-E**（feed-forward implicit）：不同 view 之间 shape 微 inconsistent
- **MVDream / LGM**：显式 multi-view 训练，绕一圈 geometry 稳定

所以你看单个 view 都"漂亮"，但绕一圈，DreamFusion 就露馅。这种 inconsistency 本身就是 fingerprint。

### Fingerprint 2：Structural artifacts

#### Geometry 层面

不同 generator 产生的 mesh 在拓扑、平滑度、watertight 性质上差异巨大：
- NeRF + marching cubes 提出来的 mesh：watertight、dense、 manifold
- 3DGS 提出来的 mesh：经常 non-manifold、有 dangling
- Point-E 直接生成点云转 mesh：topology 很乱

这 102 维 geometric descriptor 就是把这些差异量化。

#### Frequency 层面

每个 view 的 FFT spectrum 有 generator-specific 的能量分布。有的 generator 高频衰减快（smooth prior 重），有的衰减慢（细节多但 noisy）。

### 两种 fingerprint 是 complementary 的

Ablation 显示：
- 只用 rendering：54.52% accuracy
- 加 geometry + FFT：64.66%（+10.14%）
- 加显式 multi-view modeling：75.26%（再多 +10.60%）
- 两个都加 + hierarchical fusion：77.17%

**关键**：geometry 贡献 +10%，cross-view 贡献 +15%，两个加起来 +20.7%。这不是简单叠加，是 complementary 的——说明它们捕捉的是**不同类型的 fingerprint**，不冗余。

---

## 方法：Hierarchical Multi-view Multi-modal Transformer

架构分四步，我用大白话讲：

### Step 1：每个 view 单独 tokenize

对每个 view $i$（默认 4 个 view），你有：
- RGB rendering $r_i$
- Normal map $n_i$
- 102 维 geometry descriptor $s_i$
- 256 维 FFT feature $q_i$
- Optional metadata $m$（text 或 image prompt）

每个 modality 通过自己的 encoder 变成 token sequence：
- RGB / Normal / metadata → 用 **frozen pretrained vision-language encoder**（CLIP 类）—— 这些是 high-dim data，pretrained encoder 有 generic visual prior
- Geometry / FFT → 用 **lightweight learnable MLP** —— 这些是 low-dim hand-crafted feature，MLP project 就行

**Intuition**: 用 pretrained encoder 处理图像类输入是因为它们有 web-scale prior 可以 bootstrap；geometry/FFT 是 hand-crafted 数值，没有 pretrained encoder 可用，MLP 够了。这个混合策略在 multi-modal learning 里常见。

### Step 2：View 内 fusion

每个 view 内部，所有 modality token 喂进一个 **shared Transformer**：

$$h_i = \text{Transformer}_{\text{intra}}\left(\text{Concat}[\text{tok}(r_i), \text{tok}(n_i), \text{tok}(s_i), \text{tok}(q_i), \text{tok}(m)]\right)$$

注意"shared"——同一个 intra-view Transformer 跨所有 view 共享参数。这是关键：强制它学的是"如何 fuse modality"，而不是 view-specific pattern。这样 cross-view 的差异才会反映在 $h_i$ 之间的差异上。

### Step 3：Cross-view reasoning

把 $V$ 个 $h_i$ 喂进 **global Transformer**：

$$h_{\text{global}} = \text{Transformer}_{\text{cross}}\left(\text{Concat}[h_1, h_2, \ldots, h_V]\right)$$

这里 self-attention 让 model 显式计算 view $i$ 和 view $j$ 之间的关系。**这是 capture cross-view inconsistency 的核心**——不显式建模的话，model 看到的是"四个独立漂亮的图"，显式建模才能发现"view 1 和 view 3 的 normal 不一致"。

### Step 4：分类头

$$\hat{y} = \text{softmax}(W_{\text{cls}} h_{\text{global}} + b)$$

Cross-entropy loss，end-to-end 训练。

### 训练 trick：Metadata dropout

训练时以固定概率随机 drop 掉 metadata input。这让 model 学到："有 metadata 用它，没有就 fallback 到 structural/cross-view fingerprint"。这是为了 robust to deployment 时 prompt 缺失。

这思路来自 multi-modal learning 里的 modality dropout——train-time 让 model 不能依赖 single modality shortcut。

参考：
- Modality dropout (What Makes Training Multi-modal Networks Hard?): https://arxiv.org/abs/1905.12681

---

## 实验数据——讲讲几个有意思的点

### 1% data vs Full data

| Model | 1% Acc. | Full Acc. |
|---|---|---|
| GRID-CNN | 30.76 | 35.34 |
| GRID-MLP | 47.10 | 51.68 |
| GRID-TRANS | 54.52 | 92.93 |
| **Ours** | **77.17** | **97.22** |

**故事**：GRID-TRANS 是把 multi-view rendering 拼成 grid，用 standard Transformer 处理——full data 下能到 92.93%，说明 multi-view Transformer 即使不显式 cross-view，有足够监督也能学。但 1% data 下只有 54.52%，说明 implicit aggregation 在 few-shot 下 underfit。

Ours 在 1% 下 77.17%，**+22.65% 的 gap**——这是 paper 的核心 selling point：显式 structural prior + cross-view modeling 提供 strong inductive bias，让 few-shot 能 generalize。

### Per-generator 难度

**1% data 下最容易归因的**：
- L-Dreamer (Text): 100%
- Triplane-Gaussian (Image): 98.77%
- OpenLRM (Image): 96.43%
- Free3D (Image): 96.25%

**1% data 下最难的**：
- DreamFusion (Text): 6.41%（你没看错，6%）
- Shap-E-T (Text): 26.32%
- Wonder3D (Image): 45.68%
- Stable-Zero123 (Image): 45.00%

**为啥 DreamFusion 这么难？** 因为 SDS（Score Distillation Sampling）没有 explicit multi-view constraint，生成的 3D 几何乱、view 之间 inconsistent 严重、hidden surface collapse 严重。**但乱的方式是 DreamFusion 特有的**——所以 full data 下能提到 96.15%，因为 model 学会了"DreamFusion-style 的崩坏 pattern"。

这有个深层 insight：**generator 的 weakness 本身就是它的 fingerprint**。DreamFusion 难归因，不是因为它没留 fingerprint，而是因为它的 fingerprint 是"乱"本身，few-shot 下 model 还没学到"乱成什么样是 DreamFusion"。

### Robustness 实验——这是 paper 最 impressive 的部分

#### Text prompt 退化
- Full prompt: 68.67%
- 4 words: 67.92%（几乎没掉）
- 1 word: 63.55%
- Empty (test only): 63.25%
- Empty\* (train+test 都没 prompt): 60.69%

**Story**：即使训练和测试都没 prompt，还能 60.69%。这强烈说明 model 主要靠 structural + cross-view fingerprint，prompt 只是锦上添花。

#### Image prompt 加 Gaussian noise
- Clean: 82.49%
- σ=96（几乎看不清图）: 81.92%
- Empty\*: 78.72%

**Insane**——加 σ=96 的 noise 几乎不掉。这说明 image-to-3D attribution 根本不靠 prompt image，靠的是 generator 输出本身的 structural signature。

#### Image mask
- 5% mask: 82.77%
- 50% mask: 79.85%
- 90% mask: 74.01%

90% 都 mask 掉还能 74%——model 抓的是全局 structural pattern，不依赖 local visual cue。

#### 加入 real asset
- Synthetic only: 77.17%
- Mix real: 78.90%（反而提升）

加 real 反而提升 1.73%。**Intuition**：real asset 充当了"non-synthetic anchor"，让 model 更 confident 区分 synthetic 之间的差异。类似 contrastive 里加 hard negative 的效果。

---

## Confusion matrix 的有趣发现

错误不是随机的，是结构性的：

| GT → Pred | Confusion |
|---|---|
| Shap-E-T → Point-E-T | 0.46 |
| Stable-Zero123 → Zero123-XL | 0.29 |
| DreamFusion → EscherNet | 0.23 |
| GRM-T → GRM-I | 0.23 |

**解读**：
- Shap-E 和 Point-E 都是 OpenAI 出品的 feed-forward 3D generator，share 架构 + 训练数据 + representation，fingerprint 重叠合理
- Stable-Zero123 就是 Zero123-XL 的 fine-tune，同源
- GRM-T / GRM-I 是同一个 GRM 模型用于不同 task
- DreamFusion 和 EscherNet 都是 view-synthesis 路线，artifact pattern 部分重合

**深层 insight**：attribution error 本身揭示了 generator 之间的**architectural relationship**，类似 phylogenetic tree。Generator 的 fingerprint 按架构家族聚类，这个 finding 和 2D GAN attribution 的观察一致——GAN fingerprint 也 cluster by architecture。

---

## View 数量 ablation

| #Views | F1 |
|---|---|
| 1 | 72.6 |
| 4 | 74.8 |
| 5+ | saturate |

1→4 views 只涨 2.2 F1。但显式 cross-view modeling（vs implicit grid aggregation）涨 4.88 F1。

**结论**：**怎么 model 多个 view 比 view 数量本身重要**。View 数量很快 saturate，但 architecture 设计的 inductive bias 更值钱。这呼应 DeepSets / Set Transformer 的核心 insight——permutation invariance 应该被 architecture 编码。

参考：
- DeepSets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825

---

## 公式细节讲一下

### FFT feature 提取
$$q_i = \text{AvgPool}_{16\times 16}\left(\text{fftshift}\left(\log(1 + |\text{FFT}(\text{gray}(r_i))|)\right)\right) \in \mathbb{R}^{256}$$

- $\text{gray}(r_i)$: RGB 转灰度，去 color channel 干扰
- $\text{FFT}(\cdot)$: 2D Fourier Transform，分解 spatial frequency
- $|\cdot|$: 取 magnitude，丢掉 phase
- $\log(1 + \cdot)$: compress dynamic range——FFT 的 DC component（低频）通常是 pixel sum，数值巨大，log 压平
- $\text{fftshift}(\cdot)$: 把 DC 移到中心，高频在四周，方便 spatially-aligned pooling
- $\text{AvgPool}_{16\times 16}$: 压成 256 维 compact descriptor

### AdamW 更新
$$\theta_{t+1} = \theta_t - \eta \cdot \left(\frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t\right)$$

- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$: first moment estimate（β₁=0.9）
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$: second moment estimate（β₂=0.999）
- $\eta = 10^{-4}$: learning rate
- $\lambda = 10^{-2}$: weight decay
- $\epsilon = 10^{-8}$: numerical stability

paper 用 100 epoch、batch 32、cosine LR decay、H100 GPU。

参考：AdamW: https://arxiv.org/abs/1711.05101

---

## 失败案例的启示

Figure 5 给了 4 个 case，最后一个 failure：两个 generator 产生**都很 smooth + multi-view consistent** 的 3D，RGB、normal、geometry、FFT 都差不多——attribution 失败。

**启示**：当 generative 3D model 越来越好，traditional fingerprint（appearance + geometry + frequency）会失效。未来需要：
- Tracing optimization trajectory 的 artifact（SDS step count、training log）
- Internal representation probing（access latent code）
- Active watermarking——但违反 passive 假设
- 新 fingerprint：light transport、material response、subsurface scattering pattern

---

## 我（作为 reader）的几个批评

### 1. 22 个 generator future-proof 吗？

Paper 是 2026 年初，3D generation 在 fast iterate——2024-2025 出现大量 3DGS-native generator、video-to-3D、4D generator。22 个 cover 不了 wild deployment。Open-set（unknown class $u$）是 mitigation，但 paper 没充分 evaluate $u$ 的召回率。

### 2. Adversarial robustness 没测

攻击者可以 post-process 3D asset：re-mesh、smooth、decimate、re-texture、add noise——这些都能 attenuate fingerprint。Paper 完全没 evaluate robustness to adversarial post-processing。Forensic 工作这是 key concern。

### 3. Generator inheritance 问题

如果 generator B 是 generator A 的 fine-tune（如 Stable-Zero123 是 Zero123-XL 的 fine-tune），attribution 应该 return A 还是 B？Paper 没明确讨论这个哲学问题。Confusion matrix 显示它们确实混——但这是 bug 还是 feature？

### 4. View selection 的 SO(3) 问题

默认 4 个 canonical view。如果 attacker 选择 unusual view rendering 发布，model 还能 generalize 吗？这和 view-conditioned 3D recognition 的 rotation equivariance 问题相关。

### 5. Hand-crafted descriptor 的上限

102 维 hand-crafted geometry descriptor 在未来 generator 上可能不够 expressive。可以用 PointNet++ / DGCNN / MeshCNN 学习 end-to-end geometric fingerprint，让 model 自己学哪些 geometric feature 是 discriminative 的。

参考：
- MeshCNN: https://arxiv.org/abs/1905.02843
- DGCNN: https://arxiv.org/abs/1801.07829
- PointNet++: https://arxiv.org/abs/1706.02413

---

## 未来方向联想

### 1. Trajectory-based attribution
不只看 final asset，看 SDS optimization trajectory（如果能 access partial logs）。DreamFusion 的 SDS step count、SDS loss curve 可能是 strong fingerprint。

### 2. Cross-modal contrastive attribution
用 contrastive loss 让 same-generator assets 在 embedding space close、不同 generator 远——比 cross-entropy 更 data-efficient。这能直接 address 1% few-shot 问题。

### 3. Fingerprint 的 intrinsic structure
类似 2D 的 ManiFPT（Song et al. CVPR'24），定义和分析 fingerprint 的 intrinsic structure——哪些 dimension 对应 generator 的哪个组件？让 attribution 可解释。

参考：ManiFPT: https://arxiv.org/abs/2404.03476

### 4. 3D watermarking vs passive attribution 的 trade-off
行业朝 C2PA + content credentials 方向走（active defense）。Passive forensic 是 fallback——当 watermark 被 strip 时仍能 work。两者 complementary，paper 没讨论这个 ecosystem 角度。

参考：C2PA: https://c2pa.org/

---

## Intuition 总结（最后用 mental model 帮你记住）

把每个 generative 3D model 想成一个人的**嗓音**。每个人说话有：
- **口音**：geometry regularization 偏好（NeRF 提的 mesh watertight，Gaussian 提的 mesh 乱）
- **语速**：frequency spectrum decay 模式
- **咬字习惯**：cross-view consistency（DreamFusion 咬字不清，MVDream 字正腔圆）

Paper 做的是 **voice recognition**——不依赖 speaker 自己说自己是谁（metadata），纯靠声学特征。这就是为啥 metadata 全删掉还能 60%+ accuracy——嗓音本身够 unique。

DreamFusion 难归因，因为它"说话含糊"，但含糊的方式是 DreamFusion 特有的——一旦你听过足够多 DreamFusion 的录音（full data），你就能认出"DreamFusion 式含糊"。Few-shot 下你只听过 5 段，还没学会它的含糊 pattern，所以 6% accuracy。Full data 下你听过几百段，96% accuracy。

Generator 越来越好（说话越来越清晰），fingerprint 会越来越弱——这是 long-term challenge。未来 3D forensics 可能得往 trajectory-based、internal-representation-based、或者 active watermarking 方向走。

---

## 给你的 take-away

如果你（Andrej）想 build intuition，重点记三件事：

1. **3D attribution 比 2D 难在信号分散**——multi-view + geometry + frequency 三个 domain 的 fingerprint 必须 jointly model，single view 不够。Hierarchical Transformer（intra-view fusion + cross-view reasoning）是 natural architecture。

2. **Generator 的 weakness 就是 fingerprint**——DreamFusion 几何乱、Shap-E topology 不稳定、MVDream 太规整——这些"缺陷"在 classifier 眼里都是 feature。Generator 越完美，attribution 越难，这是个 arms race。

3. **Passive forensic 是 last-resort 设定**——假设 metadata 不可信、训练数据少、real/synthetic 混合。Paper 在这个设定下做到 77% (1% data) 到 97% (full data)，建立了 3D provenance 的 forensic foundation，类似 2D GAN attribution 在 2019-2020 为 deepfake detection 奠定的基础。

主要参考 link 一并整理：
- 3DGen-Bench: https://arxiv.org/abs/2503.21745
- Cap3D: https://arxiv.org/abs/2306.07279
- DreamFusion: https://arxiv.org/abs/2209.14988
- MVDream: https://arxiv.org/abs/2308.16512
- Shap-E: https://arxiv.org/abs/2305.02463
- Point-E: https://arxiv.org/abs/2212.08751
- GAN Attribution (Yu et al.): https://arxiv.org/abs/1811.08170
- Frequency analysis (Frank et al.): https://proceedings.mlr.press/v119/frank20a.html
- AdamW: https://arxiv.org/abs/1711.05101
- FAKEPCD: https://dl.acm.org/doi/10.1145/3631823
- C2PA: https://c2pa.org/
- DeepSets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825
- ManiFPT: https://arxiv.org/abs/2404.03476
- MeshCNN: https://arxiv.org/abs/1905.02843
- DGCNN: https://arxiv.org/abs/1801.07829
- Modality dropout: https://arxiv.org/abs/1905.12681

---

# "Who Generated This 3D Asset?" — 深度技术解析

## 1. Problem 的核心定位

这篇 paper 处理的是 **passive 3D source attribution** 问题。给定一个已经"流通"在 wild 中的 3D asset $A$（可能被人 re-upload、rename、strip metadata），目标是预测它来自哪个 generative model：

$$f_\theta(\mathcal{O}(A), m) \rightarrow y, \quad y \in \mathcal{Y}_{\text{syn}} = \{g_1, g_2, \ldots, g_{22}, u\}$$

变量含义：
- $A$：released 3D asset（可能是 mesh / implicit field / Gaussian / point cloud 等异构格式）
- $\mathcal{O}(A)$：observable cues（renderings + structural priors），**defender 唯一能看的输入**
- $m$：optional metadata（text prompt 或 reference image），可能缺失/损坏
- $g_i$：第 $i$ 个已知 generator，共 22 个
- $u$：unknown generator class（open-set 选项）
- $r$：real asset class（在 mixed real/synthetic 协议中扩展进来）

扩展后的 label space 写成：
$$\mathcal{Y}_{\text{mix}} = \{g_1, \ldots, g_{22}, u, r\}$$

**Intuition**: 这其实是把 2D GAN attribution (Yu et al. ICCV'19, "Attributing fake images to GANs") 的思路 lift 到 3D。但 3D 多了三个复杂性：(1) 表示异构（PLY / NeRF / 3DGS / point cloud），(2) 信号分散在多 view，(3) geometric domain 比 image domain 多一整套 structural signature。

参考文献：
- GAN fingerprints: https://arxiv.org/abs/1811.08170
- FakePCD (closest prior 3D attribution): https://dl.acm.org/doi/10.1145/3631823
- 3DGen-Bench (benchmark source): https://arxiv.org/abs/2503.21745
- Cap3D: https://arxiv.org/abs/2306.07279

---

## 2. 为什么 3D attribution 比 2D attribution 难？— 两个核心 challenge

### Challenge 1: Dispersed Attribution Signals

2D image attribution 通常一个 CNN 就能 pull fingerprint。但 3D asset 的 fingerprint 不是 concentrated 在 single view 上，而是分散在：

- **Multi-view appearance** $R = \{r_i\}_{i=1}^V$ — 每个 view 单独看，可能都是"漂亮图"，但 generator-specific artifact 在 view 之间才显形
- **Cross-view geometry consistency** $N = \{n_i\}_{i=1}^V$ — 同一个 surface 从不同角度看，generator 可能产生 inconsistent normal
- **Frequency domain** $Q = \{q_i\}_{i=1}^V$ — 每个 view 的 FFT spectrum 有 generator-specific decay pattern
- **Structural statistics** $S$ — mesh 级别的 topological / geometric scalar descriptors

**Intuition**: 2D attribution 类似"识别一个人的笔迹"，3D attribution 类似"识别一个人的雕塑风格" — 需要从多视角观察 + 摸表面 + 听敲击声，单一视角不够。

### Challenge 2: Realistic Deployment Constraints

| Protocol | Prompt | Real asset | #Data | Goal |
|---|---|---|---|---|
| Standard | Clean | ✗ | Full | Base |
| Few-shot | Clean | ✗ | 1% | Efficiency |
| Missing Prompt | Empty | ✗ | 1% | Prompt-free |
| Noisy Prompt | Corrupted | ✗ | 1% | Robustness |
| Real-Synthetic | Clean | ✓ | 1% | Deployment |

**Intuition**: 默认设定 **不是** "我有一个 big labeled training set + clean prompt"。Default 是 1% data（fewer than 5 samples per generator）+ prompt 可能完全是空。这把 paper 从 academic benchmark 推向 forensic deployment 设定 — 假设你只有 few-shot supervision，且不能依赖 metadata。

---

## 3. Benchmark 的构造细节

数据规模：
- **Synthetic subset**: 22 generators × 1,900 prompts → 10,851 assets
  - Image-to-3D: 13 models, 510 prompts, 6,361 assets
  - Text-to-3D: 9 models, 510 prompts, 4,050 assets
- **Real subset**: 880 prompts, 440 assets (来自 Cap3D，3 个 annotator cross-validation)

22 个 generator 覆盖了主要 paradigm：

| Task | Methods |
|---|---|
| Image-to-3D | Free3D, Escher-Net, Point-E, Triplane-Gaussian, Shap-E, SyncDreamer, GRM, LGM, Magic123, Zero123-XL, Stable Zero123, OpenLRM, Wonder3D |
| Text-to-3D | MVDream, Lucid-Dreamer, Magic3D, GRM, DreamFusion, Latent-NeRF, Shap-E, SJC, Point-E |

**Harmonization pipeline**:
1. 所有 asset 统一转成 **PLY** format（统一表示，避免 NeRF/3DGS/mesh 异构比较）
2. 每个资产 render 4 个 canonical views（$V=4$）
3. 每个资产额外计算 **102-dim geometric descriptor** + **256-dim/view FFT feature**

---

## 4. 关键 Empirical 发现：两种 Stable Fingerprints

Paper 通过 visual analysis 发现 generative 3D model 留下两类 stable、complementary 的 fingerprint：

### Fingerprint 1: Cross-view Inconsistency Pattern

不同 generator 的 multi-view rendering 之间存在 characteristic inconsistency：
- **Hidden-surface collapse**: DreamFusion 类方法，看不到的 surface 经常 collapse 或 Janus-face
- **View-dependent structural inconsistency**: Shap-E 这类 feed-forward 方法，不同 view 的 shape 可能微 inconsistent
- **Stable multi-view geometry**: MVDream / LGM 这类显式 multi-view 训练的方法，view consistency 强

**Intuition**: 你从单一 view 看 DreamFusion 和 MVDream 可能都"漂亮"，但绕一圈，DreamFusion 的 back face 会塌；MVDream 是真 multi-view consistent。这是 SDS-based 方法（score distillation sampling）的固有缺陷 — SDS 没有 explicit multi-view constraint。

### Fingerprint 2: Structural Artifacts

#### 2a. Geometric Statistics
每个 generator 在 102-dim geometric descriptor 空间有 distinct signature。这个 102-dim vector 拆解为：

$$\underbrace{22}_{\text{scalar}} + \underbrace{4 \times 16}_{\text{histograms}} + \underbrace{16}_{\text{Laplacian eigenvalues}} = 102$$

具体构成：

| Group | Items | Dim |
|---|---|---|
| count | vtx, face, v/f | 3 |
| bbox | bx, by, bz, bx/by, by/bz, bx/bz, bmax/bmin | 7 |
| topo+shape | area, vol, ncons, cc, wt, mani, wind, euler, nmf, bef, sip, sdiam | 12 |
| edge-hist | edge length distribution | 16 |
| face-hist | face area distribution | 16 |
| curv-hist | curvature distribution | 16 |
| spec | Laplacian eigenvalues | 16 |
| dist-hist | surface distance distribution | 16 |

**Intuition**: 
- `vtx / face / v/f` 反映 mesh density — feed-forward 方法（Point-E） vs optimization-based（DreamFusion）密度差很多
- `ncons` (normal consistency) 直接捕捉 generator 的 surface smoothness prior
- `wt / mani / wind` (watertight / manifold / winding) 捕捉 topology regularization — NeRF marching cubes 提取 vs Gaussian 的 mesh 提取 difference 巨大
- **Laplacian eigenvalues** 是 shape isometry invariant — 同一个 generator 生成的相似 shape，前 16 个 eigenvalues 分布会有 generator-specific bias
- **Histograms** 捕捉 distribution-level signature — 比 mean/std 信息更丰富

这个想法类似 Functional Maps (Ovsjanikov et al.) 中 Laplacian spectrum 作为 shape descriptor 的思路。

#### 2b. Frequency Domain

每个 view 的 RGB image 经过：

$$q_i = \text{AvgPool}_{16\times 16}\left(\text{fftshift}\left(\log(1 + |\text{FFT}(\text{gray}(r_i))|)\right)\right) \in \mathbb{R}^{256}$$

各步骤的 intuition：
1. **grayscale**：去掉 color channel 干扰，专注 spatial frequency
2. **2D FFT**：分解成 spatial frequency component
3. **$\log(1 + |FFT|)$**：compress dynamic range — FFT magnitude 通常 DC component 是 pixel sum，huge；log 压平
4. **fftshift**：把 DC（低频）移到中心，high frequency 在四周边缘，方便 spatially-aligned pooling
5. **avgpool to 16×16**：256-dim compact descriptor，捕捉 frequency energy distribution

**Intuition**: 这个 trace back 到 Frank et al. ICML'20 "Leveraging frequency analysis for deep fake image recognition" 和 Corvi et al. CVPR'23 "Intriguing properties of synthetic images"。GAN/Diffusion 生成的图在 high frequency 有 characteristic decay — up-convolution / upsampling 留下 checkerboard-like artifact。3D 这里，rendered view 继承了 generator 的 rendering pipeline artifact。

Per-view total: 256 dims，4 views = 1024 dims raw FFT feature。

参考文献：
- Frank et al. ICML'20: https://proceedings.mlr.press/v119/frank20a.html
- Corvi et al. CVPR'23: https://arxiv.org/abs/2304.10401
- Durall et al. CVPR'20 ("Watch your up-convolution"): https://arxiv.org/abs/2003.05585

---

## 5. Method: Hierarchical Multi-view Multi-modal Transformer

### 5.1 Architecture Walk-through

**Stage 1: Per-view Observable Cue Tokenization**

对每个 viewpoint $i \in \{1, \ldots, V\}$（默认 $V=4$）：

$$x_i = \{r_i, n_i, s_i, q_i\}$$

- $r_i \in \mathbb{R}^{H \times W \times 3}$: RGB rendering
- $n_i \in \mathbb{R}^{H \times W \times 3}$: normal map rendering
- $s_i \in \mathbb{R}^{102}$: geometric descriptor
- $q_i \in \mathbb{R}^{256}$: FFT feature

Optional metadata $m$（text or image prompt）。

每个 modality 通过 modality-specific encoder 得到 token sequence：
- RGB $r_i$ → **frozen pretrained vision-language encoder** (类似 CLIP/SigLIP) → patch tokens
- Normal $n_i$ → 同一个或类似的 frozen vision-language encoder → patch tokens
- Metadata $m$ → text/image branch of vision-language encoder
- Structural $s_i$ → **lightweight learnable MLP** → single token
- Frequency $q_i$ → **lightweight learnable MLP** → single token

**Intuition**: 为什么 RGB/normal/metadata 用 frozen pretrained encoder，而 structural/frequency 用 learnable MLP？
- RGB/normal/text 这些是 high-dimensional data，pretrained encoder 已经从 web-scale data 学到 generic visual representation，可以 bootstrap
- $s_i \in \mathbb{R}^{102}$ 和 $q_i \in \mathbb{R}^{256}$ 是 hand-crafted low-dimensional feature，用 MLP project 到 embedding space 就够了，没有 pretrained encoder 可用

类似想法在 Point-E / UViT / MultiMAE 都能看到：pretrained encoder 处理 high-dim modal，small MLP 处理 low-dim "structural" modal。

**Stage 2: Intra-view Multi-modal Fusion**

在每个 view 内部，所有 modality tokens 通过 **shared Transformer** 做 cross-attention / self-attention 融合：

$$h_i = \text{Transformer}_{\text{intra}}\left(\text{Concat}[\text{tok}(r_i), \text{tok}(n_i), \text{tok}(s_i), \text{tok}(q_i), \text{tok}(m)]\right)$$

输出 $h_i$ 是 viewpoint $i$ 的 fused representation。

**Intuition**: 这里 "shared" 是关键 — 同一个 intra-view Transformer 跨所有 view 共享参数，强制它学的不是 view-specific 特征，而是"如何 fuse modality"。这样 cross-view 的差异才会反映在 $h_i$ 的差异上，而不是 transformer 参数差异上。

**Stage 3: Cross-view Relationship Modeling**

把 $V$ 个 fused view representations 喂入 **global Transformer**：

$$h_{\text{global}} = \text{Transformer}_{\text{cross}}\left(\text{Concat}[h_1, h_2, \ldots, h_V]\right)$$

这里 cross-view self-attention 让 model 显式计算 $h_i$ 和 $h_j$ 之间的关系。这是 capture cross-view inconsistency 的核心 mechanism。

**Stage 4: Attribution Head**

$$\hat{y} = \text{softmax}\left(W_{\text{cls}} h_{\text{global}} + b\right)$$

训练 loss：

$$\mathcal{L} = -\sum_{k} y_k \log \hat{y}_k$$

标准 cross-entropy，end-to-end 训练。

### 5.2 训练 trick: Metadata Dropout

为了 robust to missing prompt，训练时以固定概率 $p$ 随机 drop metadata input。这让 model 学到 "metadata available 时用它，不可用时 fallback 到 structural/cross-view fingerprint"。

**Intuition**: 类似 DropConnect / DropToken，但 application 是 robustness to deployment-time missing information。这和 multi-modal learning 里的 modality dropout (e.g., What Makes Training Multi-modal Networks Hard? Wang et al. ICML'20) 思路一致 — train-time modality dropout 让 model 不依赖 single modality shortcut。

### 5.3 为什么是 hierarchical？— Ablation 揭示的关键

Ablation table 4 是理解 architecture 设计动机的核心：

| ID | Rendering | Geometry | MVC | Hierarchical | Acc. | F1 |
|---|---|---|---|---|---|---|
| 1 | ✓ | | | | 54.52 | 52.74 |
| 1' | ✓ | ✓ | | | 64.66 | 62.43 |
| 2 | ✓ | | ✓ | | 69.47 | 67.31 |
| 3 | ✓ | ✓ | ✓ | | 75.26 | 73.27 |
| 4 | ✓ | ✓ | ✓ | ✓ | 77.17 | 74.78 |

**Decomposition**:
- **Rendering only**: 54.52% — 22 类随机猜 ~4.5%，看起来 OK 但远不够
- **+ Geometry/Frequency**: +10.14% — structural cue 提供 strong complementary fingerprint，验证 fingerprint 2 的存在
- **+ Multi-view Consistency (MVC)**: +14.95% — 把 views 单独 tokenize（vs 拼成 grid）+ 显式 cross-view，验证 fingerprint 1 的存在
- **+ Hierarchical**: +1.91% — intra-view fusion 再加 cross-view，相对小但稳定提升

**最关键 insight**: MVC 单独贡献 +14.95%，Geometry 单独贡献 +10.14%，但两个加起来贡献 +20.74% — 不是 simple additive，而是 complementary。这说明 cross-view inconsistency 和 structural artifacts 是 **两种不同** fingerprint，不是冗余信号。

---

## 6. 主要实验结果

### 6.1 Few-shot vs Full Data (Table 2)

| Model | 1% Acc. | Full Acc. | 1% F1 | Full F1 |
|---|---|---|---|---|
| GRID-CNN | 30.76 | 35.34 | 26.70 | 27.84 |
| GRID-MLP | 47.10 | 51.68 | 43.22 | 49.04 |
| GRID-TRANS | 54.52 | 92.93 | 52.74 | 93.00 |
| **Ours** | **77.17** | **97.22** | **74.78** | **97.25** |

**Intuition**: 
- GRID-* baselines 是把 multi-view renderings + normals + metadata 拼成 grid，用 standard backbone。GRID-TRANS 在 full data 下达到 92.93% — 说明 multi-view Transformer 即使没有显式 cross-view modeling，full supervision 下也能学。但在 1% data 下只有 54.52%，说明 implicit aggregation 在 few-shot 下 underfit。
- Ours 在 1% data 下 77.17% — 显式 structural prior + cross-view modeling 提供 strong inductive bias，让 few-shot 能 generalize。
- 这个 +22.65% 的 few-shot gap 是 paper 卖点。

### 6.2 Per-source Analysis (Table 3, Table 10)

**1% Data 设定下，最容易归因**：
| Generator | Task | Acc. |
|---|---|---|
| L-Dreamer | Text | 100.00 |
| Triplane-Gaussian | Image | 98.77 |
| OpenLRM | Image | 96.43 |
| Free3D | Image | 96.25 |
| GRM-I | Image | 95.18 |

**1% Data 设定下，最难归因**：
| Generator | Task | Acc. |
|---|---|---|
| DreamFusion | Text | 6.41 |
| Shap-E-T | Text | 26.32 |
| Wonder3D | Image | 45.68 |
| Stable-Zero123 | Image | 45.00 |

**为什么 DreamFusion 这么难？**：DreamFusion 是 SDS (Score Distillation Sampling) 的鼻祖。SDS 没有 explicit multi-view constraint，只对每个 view 的 2D distribution 做 score matching，导致：
1. Hidden-surface collapse — 看不到的面 random / degenerate
2. Janus face — 多视角下出现多个 face
3. Geometry 噪声极大

但难归因的反面是：**这些 artifact 本身就是 fingerprint**。Full data 下 DreamFusion 提到 96.15% — 因为有足够监督，model 学会了 "DreamFusion-style 的崩坏 pattern"。

**Full Data 设定**：大多数 generator 达到 95%+，平均 97.22%。说明 22 个 generator 在足够监督下高度 separable。

### 6.3 Robust Attribution (Table 5, 11)

#### Text Prompt Degradation
| Setting | Prec. | Rec. | F1 | Acc. |
|---|---|---|---|---|
| Full prompt | 85.06 | 69.17 | 69.97 | 68.67 |
| Sparse (4 words) | 85.35 | 68.47 | 70.25 | 67.92 |
| Sparse (1 word) | 85.09 | 64.16 | 67.27 | 63.55 |
| Empty (test only) | 87.29 | 63.74 | 67.08 | 63.25 |
| Empty* (train+test) | 84.14 | 61.50 | 66.24 | 60.69 |

**Intuition**: 
- Full → 4 words: 几乎没掉（68.67 → 67.92）— 4 个词足够 disambiguate
- Full → empty (test): 掉 5.42% — 没 prompt 仍能保持 63.25%，说明 model 主要是 structural/cross-view fingerprint，prompt 是锦上添花
- Empty vs Empty*：差 2.56% — train-time metadata dropout 让 model 不依赖 prompt，但 metadata 仍提供 marginal signal

#### Image Prompt Degradation
| σ (Gaussian) | Acc. |
|---|---|
| Clean | 82.49 |
| σ=8 | 82.67 |
| σ=16 | 82.86 |
| σ=32 | 82.49 |
| σ=48 | 81.73 |
| σ=64 | 81.73 |
| σ=96 | 81.92 |
| Empty* | 78.72 |

**Insane robustness** — 加 σ=96 的 Gaussian noise（几乎看不清图）几乎不掉。这强烈暗示 image-to-3D attribution 主要靠 structural cue，prompt image 几乎不参与。

#### Masked Image Prompt
| r (mask ratio) | Acc. |
|---|---|
| 5% | 82.77 |
| 20% | 82.58 |
| 50% | 79.85 |
| 90% | 74.01 |

**Intuition**: 90% mask 仍 74.01% — 即使只能看到 10% 的 reference image，attribution 主要靠 generator 输出本身，不是 input prompt。

#### Real-Synthetic Mix
| Setting | Acc. |
|---|---|
| Synthetic only | 77.17 |
| w/ Real | 78.90 |

加入 real asset 反而 **提升** 1.73%。**Intuition**: real asset 给 model 提供了 "non-synthetic baseline"，让它更 confident synthetic 之间的 distinction。类似 contrastive learning 中加入 hard negative / anchor 的效果。

---

## 7. Confusion Matrix 分析 (Figure 8, Table 10)

Paper 给的 row-normalized confusion matrix 显示**结构性 confusions**（不是 random error）：

| GT → Pred | Confusion |
|---|---|
| Shap-E-T → Point-E-T | 0.46 |
| Stable-Zero123 → Zero123-XL | 0.29 |
| DreamFusion → EscherNet | 0.23 |
| GRM-T → GRM-I | 0.23 |

**Intuition**:
- Shap-E 和 Point-E 都是 OpenAI 的 feed-forward 3D generator，share architecture family + training data + representation，所以 fingerprint 重叠
- Stable-Zero123 是 Zero123-XL 的 fine-tuned 版本，同源
- GRM-T / GRM-I 是同一个 GRM 模型用于不同 task — 应该 share 大量 fingerprint
- DreamFusion → EscherNet：两者都有 multi-view 不稳定，但 EscherNet 也是 view-synthesis 方法，artifact pattern 部分重合

这说明 attribution error 本身揭示 generator 之间的**architectural relationship** — 类似 phylogenetic tree。这个 finding 和 2D GAN attribution (Yu et al. ICCV'19) 的 observation 一致：GAN fingerprints cluster by architecture family。

---

## 8. View Number Ablation (Figure 7)

| #Views | F1 |
|---|---|
| 1 | 72.6 |
| 2 | ~73.5 |
| 3 | ~74.2 |
| 4 | 74.8 |
| 5+ | saturate |

**关键 finding**: 1 → 4 views 提升 +2.2 F1，但 Table 4 显示显式 MVC 建模相对 implicit aggregation 提升 +4.88 F1。

**结论**: **How 多个 view 被 model 比 view 数量本身更重要**。从 1 view 到 4 view，gain 已经 saturate，但显式 cross-view attention 比 brute-force 加 view 更有效。这呼应了 set transformer / DeepSets 的核心 insight — permutation invariance 应该被 architecture 编码，不应让 model 隐式学。

参考文献:
- DeepSets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825

---

## 9. 失败案例（Figure 5）

Paper 给了 4 个案例（3 个成功 + 1 个 failure）：

**Case 3**: RGB + normal 几乎 indistinguishable，但 FFT + geometry statistic 差异大 → structural fingerprint 让 model 区分
**Failure case**: 两个 generator 都产生 smooth geometry + multi-view consistent — 当前 fingerprint (appearance + geometry + frequency) 不足够，需要新 fingerprint。

**Intuition**: 失败案例指向 future work — 当 generative 3D model 越来越好（smooth + multi-view consistent），traditional fingerprint 失效。下一步可能需要：
- Tracing 优化 trajectory 的 artifact（SDS step count）
- Internal representation probing（access latent code / NeRF density field）
- Generative watermarking（active defense）— 但这违反 paper 的 passive 假设

---

## 10. 与 Related Work 的关系

### 10.1 与 2D Attribution 的对比

2D GAN attribution (Yu et al. ICCV'19, Marra et al. MIPR'19) 的 key insight：GAN 的 generator 在 image 上留下 fingerprint，可以训练 classifier 区分。后续 work 扩展到 diffusion model (Corvi et al. ICASSP'23, Song et al. CVPR'24)。

2D 设定下，单个 image 已经有足够 fingerprint。3D 设定下，**signal 在 multi-view 上 dispersed**，需要 cross-view modeling。这是 paper 的核心 novelty。

### 10.2 与 Watermarking 的对比

3D watermarking (e.g., CopyrightShield, Yang et al. 2024) 主动在 generation 阶段 embed signal。Provenance metadata (C2PA) 在 metadata 层面 attestation。

Paper 的设定是 **passive** — 已经流通的 asset，没有 watermark，metadata 可能 stripped。这是 forensic investigation 的设定，不是 DRM 设定。

参考文献:
- CopyrightShield: https://arxiv.org/abs/2412.01528
- C2PA: https://c2pa.org/

### 10.3 与 Provenance Tracking 的对比

最近 dataset provenance 工作 (Longpre et al. Nature MI'24) 关注 dataset license/attribution audit，但不做 single-asset attribution。Paper 在 single-asset 粒度上做 attribution，更接近 forensic。

---

## 11. Implementation 细节

- **Optimizer**: AdamW, $\eta = 10^{-4}$, weight decay $10^{-2}$, cosine LR decay
- **Epochs**: 100
- **Batch size**: 32
- **Hardware**: NVIDIA H100
- **Augmentation**: random horizontal flip + color jitter on RGB renderings
- **Metadata dropout**: fixed probability during training

公式细节：
- AdamW update: $\theta_{t+1} = \theta_t - \eta \cdot (m_t / (\sqrt{v_t} + \epsilon) + \lambda \theta_t)$
  - $m_t$: first moment estimate (β₁=0.9 default)
  - $v_t$: second moment estimate (β₂=0.999 default)
  - $\lambda$: weight decay coefficient ($10^{-2}$)
  - $\epsilon$: numerical stability (1e-8)

参考文献:
- AdamW: https://arxiv.org/abs/1711.05101

---

## 12. 我的 Critique & Open Questions

### Strengths
1. **First benchmark** for passive 3D attribution with 22 generators — 之前的 FAKEPCD 只做 point cloud + few generators
2. **Strong empirical robustness** under deployment constraints — 1% data + missing prompt 仍 60%+
3. **Discovered two fingerprints** (cross-view + structural) with clear ablation evidence
4. **Confusion matrix 揭示 architectural relationship** — 学术上 cool

### Weaknesses / Open Questions
1. **22 generators 是否足够 future-proof？** 论文 2026 年初，generative 3D 已经 fast iterate（2024-2025 出现大量 3DGS-native generator, video-to-3D, 4D generator），22 个能否 cover wild deployment？Open-set (unknown generator class $u$) 是个 mitigation，但没充分 evaluate $u$ 的召回率
2. **Passive 设定 vs Modern generator 的 watermarking** — 行业正朝 C2PA + content credentials 方向走，passive forensic 是否 long-term viable？
3. **Cross-generator inheritance**: 如果 generator B 是 generator A 的 fine-tune（如 Stable-Zero123 是 Zero123-XL fine-tune），attribution 应该 return 哪个？是 base 还是 fine-tune？Paper 没明确讨论。
4. **Adversarial robustness**: 攻击者可以 post-process 3D asset（re-mesh, smooth, decimate, re-texture）来 attenuate fingerprint。Paper 没 evaluate robustness to such adversarial post-processing。这是 forensic 工作的 key concern。
5. **Real-asset label 扩展的边界**: 加 real class 反而提升 synthetic attribution (+1.73%)，这暗示 real class 起到 "anchor" 作用。但若 real 分布 OOD（如 medical scan vs artistic sculpt），效果如何？
6. **Render view selection**: 默认 4 个 canonical view。如果 attacker 选择 unusual view rendering 发布，model 还能 generalize 吗？这和 view-conditioned 3D recognition 的 SO(3) equivariance 问题相关。

### 联想到的 Future Direction
1. **Trajectory-based attribution**: 不只看 final asset，看 SDS optimization trajectory（如果能 access partial logs）。DreamFusion 的 SDS step count 可能是 strong fingerprint
2. **Geometric deep learning 替代 hand-crafted descriptor**: 102-dim hand-crafted descriptor 在未来 generator 上可能不够 expressive。可用 PointNet++ / DGCNN / MeshCNN 学习 end-to-end geometric fingerprint
3. **Cross-modal contrastive attribution**: 用 contrastive loss 让 same-generator assets 在 embedding space close，不同 generator 远 — 比 cross-entropy 更 data-efficient
4. **3D model fingerprint 的 interpretability**: 类似 2D 的 ManiFPT (Song et al. CVPR'24)，定义和分析 fingerprint 的 intrinsic structure — 哪些 dimension 对应 generator 的哪个组件？

参考文献:
- ManiFPT: https://arxiv.org/abs/2404.03476
- MeshCNN: https://arxiv.org/abs/1905.02843
- DGCNN: https://arxiv.org/abs/1801.07829

---

## 13. Intuition 总结

这篇 paper 的核心 thesis 可以浓缩成一段：

> Modern generative 3D model 不是 perfect。它们在 multi-view rendering、geometric statistics、frequency spectrum 上留下 system-level artifact，这些 artifact **stable、complementary、generator-specific**，足以训练 classifier 在 few-shot + degraded metadata 下做到 77-97% attribution accuracy。Cross-view inconsistency 是 SDS-based method 的固有 fingerprint（DreamFusion 难归因正是因为它最不一致，但 once 学到 "DreamFusion-style inconsistency" 反而高度 discriminative）。Structural artifact 是 representation-level fingerprint（NeRF marching cube vs 3DGS rasterization vs mesh 生成 topology 差异巨大）。

这建立了 3D content provenance 的 forensic foundation，类似 2D GAN attribution 早期工作（2019-2020）为 deepfake detection 奠定的基础。

主要参考 link：
- Paper (NTU): https://drive.google.com/file/d/1LkBM... (假设的，paper 未公开链接)
- 3DGen-Bench: https://arxiv.org/abs/2503.21745
- Cap3D: https://arxiv.org/abs/2306.07279
- Objaverse: https://objaverse.allenai.org/
- DreamFusion: https://arxiv.org/abs/2209.14988
- MVDream: https://arxiv.org/abs/2308.16512
- Shap-E: https://arxiv.org/abs/2305.02463
- Point-E: https://arxiv.org/abs/2212.08751
- GAN Attribution (Yu et al.): https://arxiv.org/abs/1811.08170
- Frequency analysis for deepfake (Frank et al.): https://proceedings.mlr.press/v119/frank20a.html
- AdamW: https://arxiv.org/abs/1711.05101

如果你（Andrej）想 build intuition 更深一层，我建议 mental model 是这样：把每个 generative 3D model 想成一个 "voice"。Voice 由口音（geometry regularization）、说话节奏（frequency decay）、咬字习惯（cross-view consistency）组成。Paper 做的是 voice recognition，不依赖 speaker 自己说自己是谁（metadata），而是声学特征本身。这个 metaphor 帮助理解为什么 metadata 完全去掉仍能 work — 声学特征足够 unique。
