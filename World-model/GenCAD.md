---
source_pdf: GenCAD.pdf
paper_sha256: a857572f9106ca220f7f024e136d8c43c1ab9c63df7086b2af12e45cff13cdd9
processed_at: '2026-08-04T13:20:30-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenCAD 用人话说

Andrej，我换个画风，用大白话讲一遍。

---

## 这篇 paper 到底干了啥

**一句话版本**：你给一张 CAD 图纸的照片，模型吐出一串可以编辑的 CAD 操作指令，把这串指令喂给商业 CAD 软件就能重建出 3D 实体模型。

**类比版本**：这就是 DALL-E 2 的 architecture，只不过把 "text → image" 换成了 "image → CAD program"。OpenAI 的那套 recipe（CLIP 对齐 + diffusion prior + decoder）原封不动搬过来，modality 换成 CAD command sequence。

---

## 为什么这事难

CAD 模型在工业里的标准格式叫 **B-rep**（boundary representation），它是一堆 surface / edge / vertex 的拓扑图。这玩意儿 neural network 直接吃不了——拓扑结构太乱，参数化几何太复杂。

而且 B-rep 只存最终结果，不存"这个零件是怎么一步步造出来的"。工程师真正想要的是 **design history**——先画个 sketch，再 extrude，再 cut 个孔——因为这样才 editable。

所以关键 design choice：**把 CAD 当成一段 program**。每个操作是一个 token，带 continuous parameters。整段 program 喂给 geometry kernel（OpenCascade）就能重建 3D solid。

这就把 CAD generation 变成了 **language modeling 问题**。

---

## CAD program 长什么样

一段 CAD program 大概是这种 sequence：

```
⟨SOL⟩ → Line(x1,y1) → Line(x2,y2) → Arc(x3,y3,α,f) 
       → Circle(cx,cy,r) → ⟨SOL⟩ → Extrude(plane_params, distance, bool_op) → ⟨EOS⟩
```

每个 command 编码成 17 维 vector：1 维 type + 16 维 parameters（不够的 padding）。

参数被 quantize 到 8-bit（256 levels），所以连续值也变离散 token 了，整个 sequence 就是纯 discrete token stream，可以用 next-token prediction 训练。

这和 ImageGPT / VQ-VAE 把 image patch 量化后做 autoregressive 是同一个套路。

---

## 四步 pipeline

### Step 1: CSR（Command Sequence Reconstruction）

训练一个 transformer encoder-decoder，输入 CAD command sequence，输出 reconstructed sequence。

**关键点**：用 **causal masking**（GPT-style），不是 BERT-style 的 bidirectional。

为什么？因为 CAD construction 本身就是 sequential 的——每条 line 的起点是上一条 line 的终点，extrude 的 plane 依赖之前 sketch 的 geometry。Causal mask 让 encoder 学到这种 order dependency。

实验结果（Table 1）：比 DeepCAD 的 bidirectional encoder 好一点，而且 **sequence 越长优势越大**（Figure 5）。直觉就是 GPT vs BERT 在 generation 任务上的差异。

这一步的产物是一个 frozen 的 **CAD encoder**（把 sequence 压成 256 维 latent）和 frozen 的 **CAD decoder**（把 latent 解回 sequence）。

### Step 2: CCIP（Contrastive CAD-Image Pre-training）

这就是 **CLIP 的 CAD 版**。

- CAD encoder：frozen（来自 Step 1）
- Image encoder：ResNet-18，把 448×448 grayscale CAD 渲染图压成 256 维 latent
- Loss：InfoNCE，一个 batch 里 B 个 CAD-image pairs，让正样本 cosine similarity 高，2(B-1) 个负样本低

数据增强：每个 CAD 模型渲染 5 个不同 scale 的版本，逼 model 学 scale-invariant 的几何特征。

**惊人结果**（Table 2）：用 image 从 2048 个 CAD 里 retrieval 正确的那个，GenCAD 60.77%，ImageNet pretrained ResNet 做 image-to-image search 只有 3.91%。**15× 提升**。

直觉：ImageNet features 只学到 appearance，没学到 CAD-specific 的几何结构。GenCAD 的 image latent 和 CAD program latent 在同一空间，所以 retrieval 准。

这个 retrieval 能力本身就是个 standalone product feature——工程师拍张照就能从公司 CAD 库里找到最像的 model。

### Step 3: CDP（CAD Diffusion Prior）

CCIP 让 image latent 和 CAD latent 对齐了，但是 **一张图对应多个合理 CAD program**（multi-modality）。如果直接 deterministic regression $\mathbf{z}_{\text{CAD}} = f(\mathbf{z}_{\text{image}})$，会 mode-collapse 到 average latent。

所以用 **latent diffusion** 学 $p(\mathbf{z}_{\text{CAD}} | \mathbf{z}_{\text{image}})$。

这完全是 DALL-E 2 的 prior——CLIP image embedding → diffusion prior → CLIP text embedding。这里是 image latent → diffusion → CAD latent。

**架构细节**：denoising model 用 **ResNet-MLP**（不是 U-Net），因为 diffusion 在 256 维 vector 上做，没有 spatial structure，U-Net 的 inductive bias 没用。ResNet-MLP 就是几个 `Linear → Norm → ReLU → Linear + residual` 的 block stack 起来。

Forward diffusion 标准 DDPM，500 timesteps。Reverse denoising 预测 noise，conditioning 通过 concat image latent 到 input。

### Step 4: CAD Decoder

直接用 Step 1 训好的 decoder，**frozen**。

CDP sample 出来的 $\mathbf{z}_{\text{CAD}}$ → constant embeddings broadcast 成 sequence → transformer decoder autoregressive 生成 command sequence → prepend ⟨SOL⟩ → 喂给 OpenCascade → B-rep 3D solid。

**为什么 frozen**：decoder 是最贵的部分（autoregressive transformer），换 conditioning modality 不用重训。这和 Stable Diffusion 用 frozen VAE decoder 是一个道理。

---

## 实验结果人话版

### Reconstruction（Table 1）

GenCAD 在 command accuracy / parameter accuracy / chamfer distance / invalid ratio 上都比 DeepCAD 好，但 margin 很小（99.51 vs 99.36 之类）。真正的优势在 **长 sequence**——越复杂越赢。

### Retrieval（Table 2）

15× better than image-to-image search。这是 paper 最强的 claim 之一。

### Generation（Table 3）

Conditional（image / sketch conditioning）> Unconditional > DeepCAD l-GAN。

具体数字：

| | COV ↑ | MMD ↓ | JSD ↓ |
|---|---|---|---|
| DeepCAD l-GAN | 78.13 | 1.45 | 3.76 |
| GenCAD unconditional | 78.27 | 1.44 | 3.94 |
| GenCAD-image | 81.37 | 1.38 | 3.49 |
| GenCAD-sketch | 82.59 | 1.33 | 3.53 |

Conditional 把 COV（diversity）从 78 拉到 82，MMD（fidelity）也降了。Conditioning 提供了 strong prior，让生成更聚焦在 test distribution 附近。

Sketch 比 image 略好——可能因为 Canny edge 更直接对应 CAD sketch 的 line/arc 结构，少了一层 photometric noise。

### FID（Figure 12）

Conditional diffusion 最低 → unconditional diffusion → l-GAN → deterministic prior 最高（mode collapse）。

Diffusion > Deterministic，因为 diffusion 能 sample 多个 modes，deterministic regression 只给 average。

---

## 最核心的 intuition

**GenCAD = unCLIP for CAD programs**。

整个 recipe 没有任何新东西：
1. Autoregressive transformer 学 sequence latent（= CLIP text encoder 的角色）
2. ResNet 学 image latent（= CLIP image encoder）
3. Contrastive loss 对齐（= CLIP）
4. Latent diffusion 学 conditional prior（= DALL-E 2 prior）
5. Frozen AR decoder 解码（= DALL-E 2 decoder，只不过 image 用 diffusion decoder，CAD 用 AR transformer decoder）

Novelty 不在 method，在 **application domain**。把这套成熟 recipe 搬到 CAD 上，证明它 transfer 得很好。15× retrieval improvement 和 conditional > unconditional 的实验结论都印证了。

---

## 局限性

Paper 自己承认的：
- CAD vocabulary 太小（只有 line/circle/arc + extrude，没有 fillet/revolve/mirror/loft/sweep）
- 3.32% 生成的 CAD invalid，没有 geometry kernel feedback loop
- Image 太干净（isometric, noise-free, grayscale），real-world photo 没测
- Dataset 偏简单（168k models，远不如 LAION-5B）

我额外想的：
- 没有 text conditioning（CLIP 有 text，这里丢了）
- Single-view only（工程师实际用三视图）
- 生成一次性，没有 human-in-the-loop refinement
- 256 维 latent 可能 bottleneck 复杂 CAD

---

## 可能的下一步

1. **Geometry kernel in the loop**：把 OpenCascade 当 verifier，invalid CAD 给 negative reward，RL fine-tune 或 rejection sampling
2. **Text-to-CAD**：加 text encoder，text-image-CAD 三元组 contrastive
3. **Multi-view input**：front/top/side 三视图，multi-view ViT encoder
4. **CAD-specific BPE**：学 sub-sequence 作为 compound token，加速长 sequence 生成
5. **Hierarchical diffusion**：先 diffusion topology 再 diffusion parameters
6. **Discrete diffusion on sequence**：跳过 latent bottleneck，直接在 command sequence 上做 discrete diffusion（VQ-Diffusion / D3PM 风格）
7. **Test-time scaling**：diffusion sample N 个 candidates，geometry kernel + chamfer distance 做 verifier，best-of-n

---

## 参考链接

- GenCAD OpenReview: https://openreview.net/forum?id=e817c1wEZ6
- DeepCAD: https://arxiv.org/abs/2105.01890
- ABC dataset: https://archive.nyu.edu/collections/4v6vz4q8/
- CLIP: https://arxiv.org/abs/2103.00020
- DDPM: https://arxiv.org/abs/2006.11239
- DALL-E 2 (unCLIP): https://cdn.openai.com/papers/dall-e-2.pdf
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- ResNet-MLP (tabular): https://arxiv.org/abs/2106.11959
- SkexGen: https://arxiv.org/abs/2207.04632
- BrepGen: https://arxiv.org/abs/2401.15563
- ContrastCAD: https://arxiv.org/abs/2404.01645
- ROCA: https://arxiv.org/abs/2204.06214
- DiffCAD: https://arxiv.org/abs/2401.09115

---

# GenCAD: Image-Conditioned CAD Generation 深度解析

Andrej, 这篇 paper 本质上是把 OpenAI 在 DALL-E / CLIP / latent diffusion 那套 recipe 搬到了 CAD 这个 engineering domain。我把核心思路、架构细节、公式和实验数据都展开讲，目标是让你 build intuition。

---

## 1. 核心问题与 motivation

### 1.1 为什么 CAD generation 难

CAD 模型的工业标准 representation 是 **B-rep (boundary representation)**——通过 parametric surfaces, edges, vertices 的拓扑关系编码 3D solid。B-rep 的复杂之处在于：

- topological graph 结构（face-edge-vertex 之间非线性 adjacency）
- 参数化几何（NURBS surfaces, parametric curves）
- design history 缺失（B-rep 只是最终结果，没有"怎么造的"信息）

对比 mesh / voxel / point cloud / implicit (NeRF, SDF)，B-rep 的优势在于：
- resolution-independent（不像 voxel 受 grid 限制）
- memory-efficient（不像 dense mesh）
- manufacturable & editable（工程师可以直接改参数）
- 可以通过 geometry kernel（OpenCascade, Parasolid）无缝转成 mesh / voxel / point cloud

但是 B-rep 不能直接喂给 neural network。所以核心 design choice 是：**把 CAD 当成 language**——一个 CAD 模型是一串 parametric command sequence，类似一段 program，每个 command 是一个 token with continuous parameters。

### 1.2 为什么是 image-conditioned

之前的 DeepCAD (Wu et al., 2021)、SkexGen (Xu et al., 2022)、BrepGen (Xu et al., 2024b) 都是 **unconditional generation**——sample 一个 latent 出来生成 random CAD。这对实际 design pipeline 几乎没用。工程师想要的是：**给我一张 sketch / 图像 / 截图，输出可编辑的 CAD program**。这就是 image-to-CAD 的核心价值。

类比 DALL-E 的 text-to-image，GenCAD 是 **image-to-CAD program**。

---

## 2. CAD 作为 language：tokenization 设计

### 2.1 Command vector 结构

每个 CAD command 编码为：

$$\mathbf{c}_i \in \mathbb{R}^{17}, \quad \mathbf{c}_i = (t_i, \mathbf{p}_i)$$

- $t_i \in \mathbb{R}$: command type（离散的，类似 NLP token id）
- $\mathbf{p}_i \in \mathbb{R}^{16}$: parameters（连续值，但 quantize 到 256 levels / 8-bit int）

为什么 16 维？因为不同 command 参数数量不同（line 2 个，circle 3 个，arc 4 个，extrude 10 个），通过 padding + masking 统一到固定维度，方便 batch 处理。这和 NLP 里面 pad sequence 是一个套路。

### 2.2 Vocabulary

| Token type | Parameters | 说明 |
|-----------|-----------|------|
| ⟨SOL⟩ | 0 | Start of loop（sketch loop 开始）|
| ⟨EOS⟩ | 0 | End of sequence |
| Line | $(x, y)$ | endpoint，起点由上一个 command 隐式给出 |
| Circle | $(x, y, r)$ | center + radius |
| Arc | $(x, y, \alpha, f)$ | endpoint + sweep angle + direction flag |
| Extrusion | $(\theta, \phi, \gamma, p_x, p_y, p_z, s, e_1, e_2, b_{op}, b_{side})$ | 10 params: sketch plane orientation (3 Euler angles) + origin (3) + scale (1) + extrude distances two-sided (2) + boolean op (join/cut/intersect/new) (1) + one/two-sided flag (1) |

注意 $N_c = 6$（command types 总数），$N_p = 16$（padded parameter dim）。Dataset 限制在 sketch + extrude，没有 fillet / chamfer / revolve / mirror，这是 DeepCAD dataset 的限制，也是 paper 的 limitation。

### 2.3 Sequential semantics

关键直觉：CAD sketch 是 **chain-coded** 的——每条 line 的起点 = 上一条 line 的终点。这和 NLP 里面 token 之间有强 dependency 是一致的，所以 autoregressive modeling 自然契合。Extrusion 之后，下一个 sketch 在新的 plane 上，plane 由 extrude command 的 $(\theta, \phi, \gamma, p_x, p_y, p_z)$ 定义。

类比：这就像 SVG path 的 `M L L A C Z` commands，只不过多了一维 extrude 操作。

参考：DeepCAD 论文 https://arxiv.org/abs/2105.01890 ，ABC dataset https://archive.nyu.edu/collections/4v6vz4q8/

---

## 3. 四步框架：CSR → CCIP → CDP → Decoder

整个 GenCAD 是个 pipeline，每个 module 单独训练，最终串起来做 inference。

### 3.1 Step 1: Command Sequence Reconstruction (CSR)

#### 架构
- Transformer encoder + decoder，causal masking（both encoder 和 decoder 都用 causal mask，这是和 DeepCAD 的关键区别——DeepCAD 用 non-causal / bidirectional encoder）
- 4 个 self-attention layers，8 attention heads
- Embedding layer 把 quantized 8-bit 参数映射到 $d$-dim continuous space
- Encoder 输出 $\mathbf{z}_{\text{CAD},t} \in \mathbb{R}^{d_z}$，$d_z = 256$
- Average pooling 得到 single latent $\mathbf{z}_{\text{CAD}} \in \mathbb{R}^{256}$
- Constant learned embeddings 把 single latent broadcast 回 sequence of latents
- Decoder 输出 reconstructed command sequence
- 最后 tanh activation 生成 latent

#### Loss function (Equation 1)

$$\mathcal{L} = \sum_{i=1}^{N_c} \ell(\hat{t}_i, t_i) + \beta \sum_{i=1}^{N_c} \sum_{j=1}^{N_p} \ell(\hat{\mathbf{p}}_{ij}, \mathbf{p}_{ij})$$

变量含义：
- $N_c = 6$: command types 数
- $N_p = 16$: parameters per command
- $t_i, \hat{t}_i$: ground truth 和 predicted command type（第 $i$ 个 command）
- $\mathbf{p}_{ij}, \hat{\mathbf{p}}_{ij}$: ground truth 和 predicted 第 $i$ 个 command 的第 $j$ 个 parameter
- $\ell(\cdot, \cdot)$: cross-entropy loss（注意 parameters 也被 quantize 到 256 levels，所以可以用 CE 而不是 MSE）
- $\beta$: 平衡 type loss 和 parameter loss 的权重

直觉：把 continuous parameters 离散化后，整个 CAD sequence 变成纯 discrete token sequence，可以完全用 next-token prediction 的方式训练。这和 ImageGPT / VQ-VAE 把 image patches 量化后做 AR 是一个思路。

#### 为什么 autoregressive 比 DeepCAD 的 bidirectional encoder 好

Table 1 显示 GenCAD 在所有指标上都比 DeepCAD 好（虽然 margin 不大）：

| Method | $\mu_{\text{cmd}}$ ↑ | $\mu_{\text{param}}$ ↑ | $\mu_{\text{CD}}$ ↓ | IR ↓ |
|--------|------|--------|------|------|
| DeepCAD | 99.36 | 97.59 | 0.783 | 3.44 |
| GenCAD | **99.51** | **97.78** | **0.762** | **3.32** |

更关键的在 Figure 5：随着 sequence length 增加，GenCAD 的 advantage 越来越明显。直觉是：**CAD 是 sequential by construction**（每个 command 依赖前一个 command 的输出 state），causal mask 让 encoder 学到这种 sequential dependency，而 bidirectional encoder 把 sequence 当 bag 看待，丢失了 order 信息。

类比：这就是 GPT vs BERT 在 generation 任务上的差异。BERT 适合 understanding，GPT 适合 generation。这里 CSR 是为了下游 generation 服务的，所以 AR 更合适。

参考：Transformer 原文 https://arxiv.org/abs/1706.03762 ，VQ-VAE https://arxiv.org/abs/1711.00937

### 3.2 Step 2: Contrastive CAD-Image Pre-training (CCIP)

#### 目标
学一个 joint embedding space，让 CAD command sequence 和对应的 rendered image 在 latent space 里对齐。这是 CLIP 的 CAD 版本。

#### 架构
- CAD encoder: frozen（来自 Step 1）
- Image encoder: ResNet-18（也试过 ResNet-35, ViT，发现 ResNet-18 够用了，见 Appendix Table 3）
- Image preprocessing: resize 到 $256 \times 256$，center crop，normalize $\mathcal{N}(0.5, 0.5)$
- ResNet-18 输出 $512 \times 8 \times 8$，linear projection 到 $d_z = 256$

#### Loss function (Equation 2)

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},j}) / \tau)}{\sum_{k=1}^{2B} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},k}) / \tau)}$$

$$\mathcal{L} = \frac{1}{2B} \sum_{k=1}^{B} [\ell(2k-1, 2k) + \ell(2k, 2k-1)]$$

变量含义：
- $B$: batch size
- $2B$: 一个 batch 里有 $B$ 个 CAD-image pairs，共 $2B$ 个 samples
- $\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},j}$: CAD latent 和 image latent
- $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T \mathbf{v} / \|\mathbf{u}\| \|\mathbf{v}\|$: cosine similarity
- $\tau$: temperature parameter（控制 similarity distribution 的 sharpness，越小越 sharp，模型越 confident）
- $\mathbb{1}_{[k \neq i]}$: indicator function，排除 self-similarity
- $2(B-1)$: negative pairs 数量

直觉：这就是 **InfoNCE loss**，和 SimCLR / CLIP 完全一样。Batch 越大 negative 越多，contrastive signal 越强。

#### 数据增强
每个 CAD 模型生成 5 个 scaled versions（x/y/z 轴不同 scale），这给 CCIP 提供了天然 augmentation——同一 CAD 的不同 scaled image 应该 map 到同一 CAD latent。这有点像 CLIP 里面对同一 image 做不同 crop。

#### 参数量
- ResNet-18 CCIP: 28.22M trainable parameters（CAD encoder frozen）

#### Retrieval 实验

Table 2 是惊艳的结果：

| Method | $R_{B=10}$ | $R_{B=128}$ | $R_{B=1024}$ | $R_{B=2048}$ |
|--------|-----------|------------|-------------|-------------|
| Random | 10.06 | N/A | N/A | N/A |
| ResNet-18 (ImageNet pretrained, image-to-image) | 77.70 | 19.26 | 5.21 | 3.91 |
| **GenCAD-image** | **98.49** | **91.41** | 70.28 | **60.77** |
| GenCAD-sketch | 98.36 | 87.5 | **70.67** | 60.77 |

直觉分析：
- Image-to-image search（用 ImageNet pretrained ResNet）在 small batch 上还行（77.7%），但 batch 大了就崩了——因为 ImageNet features 没学到 CAD-specific 的几何信息，只学到 appearance。
- GenCAD 学到的是 **CAD-aware image features**：image latent 和 CAD program latent 在同一空间，所以即使 2048 个 candidates 里也能 60% 准确找回。
- **15× better** than image-to-image（60.77 vs 3.91 at B=2048）。

这个 retrieval 能力本身就是一个 standalone product feature——工程师拍张照就能从公司 CAD 库里找到最像的 model。

参考：CLIP https://arxiv.org/abs/2103.00020 ，SimCLR https://arxiv.org/abs/2002.05709

### 3.3 Step 3: CAD Diffusion Prior (CDP)

#### 为什么需要 diffusion prior

CCIP 学到的 image latent $\mathbf{z}_{\text{image}}$ 和 CAD latent $\mathbf{z}_{\text{CAD}}$ 在同一空间，但是：
- $\mathbf{z}_{\text{image}} \to \mathbf{z}_{\text{CAD}}$ 不是 deterministic 的——一张图可能对应多个合理的 CAD program（multi-modality）
- Deterministic regression 会 mode-collapse，得到 average of modes（blurry image 的 latent 对应物）

所以需要一个 **generative prior** $p(\mathbf{z}_{\text{CAD}} | \mathbf{z}_{\text{image}})$。Paper 用 latent diffusion 来 model 这个 conditional distribution。

这完全是 **DALL-E 2 / unCLIP** 的架构：CLIP image embedding → diffusion prior → CLIP text embedding → decoder。只不过这里是 image → CAD latent。

#### Forward diffusion

标准 DDPM (Ho et al., 2020)：

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t} \mathbf{z}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

- $\mathbf{z}_0 = \mathbf{z}_{\text{CAD}}$: clean CAD latent
- $\mathbf{z}_t$: noised latent at timestep $t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$: cumulative noise schedule
- $T = 500$: total timesteps

#### Reverse denoising

Denoising model $\epsilon_\theta(\mathbf{z}_t, t, \mathbf{z}_{\text{image}})$ 预测 noise。

关键架构选择：**用 ResNet-MLP 而不是 U-Net**。因为 diffusion 是在 256-dim latent vector 上做的，不是在 image / spatial tensor 上，所以 U-Net 的 spatial inductive bias 没用，MLP-with-residual-blocks 更合适。

ResNet-MLP block 结构（来自 Gorishniy et al., 2021, https://arxiv.org/abs/2106.11959）：
```
Linear → BN/LinearNorm → ReLU → Linear → +residual → ...
```
若干个这样的 block stacked，最后 normalization + linear head。

Input 是 $[\mathbf{z}_t; \mathbf{z}_{\text{image}}]$ concat（可能还有 timestep embedding），先经过 projection layer 再进 ResNet-MLP。

#### 也试了 deterministic prior 作为 baseline
- 简单 ResNet-MLP，直接 regress $\mathbf{z}_{\text{CAD}} = f(\mathbf{z}_{\text{image}})$
- 结果（Figure 12 FID score）：deterministic prior 最差，因为它 mode-collapse

#### Training
- 500 timesteps (diffusion)
- 1M training steps
- lr = $1 \times 10^{-5}$, fixed
- Gradient accumulation every 2 steps
- Max gradient norm 1.0 (gradient clipping)

参考：DDPM https://arxiv.org/abs/2006.11239 ，DALL-E 2 https://cdn.openai.com/papers/dall-e-2.pdf ，latent diffusion https://arxiv.org/abs/2112.10752

### 3.4 Step 4: CAD Decoder

直接复用 Step 1 训好的 CSR decoder，**frozen**。
- Input: $\mathbf{z}_{\text{CAD}}$ from CDP
- Constant embeddings broadcast 到 sequence
- Transformer decoder autoregressive 生成 $\mathbf{c}_2, ..., \mathbf{c}_{N+1}$
- Prepend ⟨SOL⟩ 得到完整 sequence
- 最终 sequence 喂给 OpenCascade geometry kernel 生成 B-rep

这种 **frozen decoder** 的设计有 scaling benefit——decoder 是最贵的部分（autoregressive transformer），如果每次换 conditioning modality 都要重训 decoder 就太贵了。这和 Stable Diffusion 用 frozen VAE decoder 是一个思路。

---

## 4. Dataset

### 4.1 DeepCAD
- 来源：ABC dataset (1M CAD models from Onshape public repo)
- DeepCAD 过滤后：178,238 CAD designs，只用 sketch + extrude
- GenCAD 进一步过滤（用 OpenCascade 验证能 render 3D solid）：168,674 models
  - Train: 152,530
  - Val: 8,515
  - Test: 7,629

### 4.2 Image augmentation
每个 CAD 生成 5 个 scaled versions（x/y/z 轴独立 scale），渲染成 grayscale $1 \times 448 \times 448$ isometric view。
- Total images: 845,105
- Sketch dataset: Canny edge + Gaussian blur of images，845,105 sketches

直觉：scaling augmentation 让 model 学到 scale-invariant 的几何特征。Canny edge 模拟工程师手绘 sketch。

参考：DeepCAD https://deepcad.org/ ，ABC dataset https://archive.nyu.edu/collections/4v6vz4q8/

---

## 5. Evaluation metrics 详解

### 5.1 Reconstruction metrics

$$\mu_{\text{cmd}} = \frac{1}{|\mathcal{G}|} \sum_{k=1}^{|\mathcal{G}|} \frac{1}{N_c} \sum_{i=1}^{N_c} \mathbb{I}[t_i = \hat{t}_i]$$

$$\mu_{\text{param}} = \frac{1}{|\mathcal{G}|} \sum_{k=1}^{|\mathcal{G}|} \frac{1}{\sum_i \mathbb{I}[t_i = \hat{t}_i] N_p} \sum_i \sum_j |\mathbf{p}_{i,j} - \hat{\mathbf{p}}_{i,j}| < \eta \cdot \mathbb{I}[t_i = \hat{t}_i]$$

- $|\mathcal{G}|$: test set 大小
- $N_c$: command 数
- $\mathbb{I}[\cdot]$: indicator
- $\eta$: parameter tolerance threshold
- $\mu_{\text{param}}$ 只在 command type 正确时才计算 parameter accuracy（避免 type 错了还去惩罚 parameter）

$$\mu_{\text{CD}} = \frac{1}{|\mathcal{G}|} \sum_{k \in \mathcal{G}} \text{CD}(k, \mathcal{S})$$

Chamfer distance 在 2000-point point cloud 上算（B-rep → point cloud）。

### 5.2 Generation metrics (Achlioptas et al., 2018)

- **COV (Coverage)**: $\text{COV}(\mathcal{S}, \mathcal{G}) = \frac{|\{ \arg\min_{j} d(\mathbf{g}_j, \mathcal{S}) \mid \mathbf{g}_j \in \mathcal{G} \}|}{|\mathcal{G}|}$，衡量 generated set 覆盖 ground truth 的程度（diversity）
- **MMD (Minimum Matching Distance)**: $\text{MMD}(\mathcal{S}, \mathcal{G}) = \frac{1}{|\mathcal{G}|} \sum_i \min_j d(\mathbf{g}_i, \mathbf{s}_j)$，衡量 fidelity（generated 离 ground truth 多近）
- **JSD (Jensen-Shannon Divergence)**: 两个 point cloud 分布的 statistical distance

直觉：COV 高 = 多样性够，MMD 低 = 质量好，JSD 低 = 分布匹配。这三个有 trade-off——只生成少数 high-quality 样本 MMD 低但 COV 也低；乱生成 COV 高但 MMD 高。

### 5.3 FID for image-conditional generation

$$\text{FID} = \|\mu_S - \mu_\mathcal{G}\|_2^2 + \text{tr}\left(\Sigma_S + \Sigma_\mathcal{G} - 2(\Sigma_S \Sigma_\mathcal{G})^{1/2}\right)$$

- $\mathcal{N}(\mu_S, \Sigma_S)$: generated CAD latents 的 Gaussian
- $\mathcal{N}(\mu_\mathcal{G}, \Sigma_\mathcal{G})$: ground truth CAD latents 的 Gaussian
- 第一项: mean difference
- 第二项: covariance difference（Frechet distance）

FID 衡量 generated CAD 和 ground-truth-aligned CAD 在 latent space 的分布距离。

参考：Achlioptas et al. https://arxiv.org/abs/1707.02392 ，FID https://arxiv.org/abs/1706.08500

---

## 6. 实验结果深度分析

### 6.1 Unconditional generation (Table 3)

| Method | type | COV ↑ | MMD ↓ | JSD ↓ |
|--------|------|------|------|------|
| DeepCAD (l-GAN) | unconditional | 78.13 | 1.45 | 3.76 |
| SkexGen | unconditional | 78.17 | 1.55 | 4.89 |
| BrepGen | unconditional | 73.10 | **1.05** | **1.22** |
| ContrastCAD + RRE | unconditional | 78.93 | 1.44 | 3.67 |
| GenCAD | unconditional | 78.27 | 1.44 | 3.94 |
| **GenCAD-image** | conditional | 81.37 | 1.38 | 3.49 |
| **GenCAD-sketch** | conditional | **82.59** | **1.33** | 3.53 |

关键观察：
1. **Conditional > Unconditional**：image conditioning 把 COV 从 78.27 拉到 81.37，MMD 从 1.44 降到 1.38。这符合直觉——conditioning 提供了 strong prior，让生成更聚焦在 test distribution 附近。
2. **BrepGen 的 trade-off**：MMD/JSD 最低但 COV 也最低（73.10）。BrepGen 直接生成 B-rep，可能在 fidelity 上有优势但牺牲了 diversity——可能 mode collapse 到几个 typical shapes。
3. **GenCAD 在 COV/MMD/JSD 上 balanced**：没有极端优化单一指标。
4. **Sketch 比 image 略好**：可能因为 sketch（Canny edge）更直接对应 CAD sketch 的 line/arc 结构，少了一层 photometric noise。

### 6.2 FID score (Figure 12)

排序（低 to 高，低好）：
1. GenCAD-image (conditional diffusion) - 最低
2. GenCAD-sketch (conditional diffusion) - 略高（因为更多样的生成）
3. Unconditional diffusion
4. DeepCAD l-GAN
5. Deterministic prior - 最高（mode collapse）

直觉：
- **Diffusion prior > Deterministic prior**：deterministic regression 输出 average latent，远离任何 mode。Diffusion 能 sample 多个 modes，更接近真实分布。
- **Conditional > Unconditional**：conditioning 把生成限制在 test distribution 附近。
- **Sketch FID > Image FID**：看似矛盾（sketch COV 更高），但 FID 衡量分布对齐，sketch 模型生成更多样化，分布更"宽"，FID 自然高一些。这和 COV 高是一致的——diversity 高意味着分布 spread 大。

### 6.3 Diversity (Figure 11)

同一个 image input，sample 多个 CAD programs。Diffusion 的 stochasticity 让生成有 variation，但都在"合理"范围内。这是 deterministic prior 做不到的。

### 6.4 Editability (Figure 13)

输出是 CAD command sequence，可以导入 Onshape 等 commercial CAD 软件，工程师可以 edit specific feature。这是 mesh / voxel / point cloud generation 做不到的——那些 representation 是 "baked"，不能 parametrically edit。

---

## 7. Limitations

Paper 自己承认的：
1. **CAD vocabulary 有限**：只有 line/circle/arc + extrude，没有 fillet/chamfer/revolve/mirror/loft/sweep。工业级 CAD 远比这复杂。
2. **不保证 valid CAD**：约 3.32% 生成的 CAD invalid（IR metric）。没有 geometry kernel feedback loop。
3. **Image 简单**：isometric view, noise-free, grayscale。Real-world photo with cluttered background / occlusion / non-isometric view 没测。
4. **Dataset 偏简单**：DeepCAD 过滤后只剩相对简单的 mechanical parts。

我额外想到的：
- **No text conditioning**：CLIP 有 text，这里只有 image。如果能加 text（"a flange with 4 holes"）会更强。
- **Single-view only**：multi-view engineering drawing 是 CAD 的 native input format，paper 没利用。
- **No iterative refinement**：生成是一次性的，没有 human-in-the-loop refinement。
- **Diffusion 在 256-dim latent 上**：比直接在 sequence 上 diffusion 快，但 latent bottleneck 可能丢失细节。
- **CSR 的 improvement margin 小**：99.51 vs 99.36 在 $\mu_{\text{cmd}}$ 上，可能 statistical significance 不强。

---

## 8. 整体架构直觉总结

把 GenCAD 和你熟悉的 OpenAI stack 对比：

| OpenAI / Stability | GenCAD 对应 |
|---|---|
| Text encoder (CLIP text) | CAD encoder (CSR transformer) |
| Image encoder (CLIP image) | Image encoder (ResNet-18) |
| CLIP contrastive loss | CCIP contrastive loss |
| Diffusion prior (DALL-E 2) | CDP (CAD diffusion prior) |
| Diffusion decoder (DALL-E 2 / SD) | Frozen CSR decoder |
| Image (pixel space) | CAD command sequence |
| VAE latent | CAD latent $\mathbf{z}_{\text{CAD}} \in \mathbb{R}^{256}$ |

所以 GenCAD 本质上是 **unCLIP for CAD programs**。Recipe 完全一样，只是 modality 从 image 换成 CAD command sequence，decoder 从 diffusion decoder 换成 autoregressive transformer decoder（因为 CAD 是 sequence，不是 spatial grid）。

### 为什么这个 recipe work

1. **CAD 作为 language 的可学习性**：CAD command sequence 有明确的语法和语义，transformer 擅长 capture 这种 structure。
2. **Contrastive learning 对齐 modality**：CLIP 证明了 contrastive 能学到 strong cross-modal alignment，这里复用。
3. **Diffusion 处理 multi-modality**：image-to-CAD 是 one-to-many（一张图多个合理 CAD），diffusion 自然处理。
4. **Frozen decoder 的 scalability**：可以加更多 conditioning modality（text, multi-view）而不用重训 decoder。

### 为什么这个 recipe 可能不够

1. **CAD program 的 combinatorial complexity**：6 个 command types 看起来少，但 parameter space 连续，且 command 之间有强 dependency（extrude plane 依赖 sketch geometry）。Transformer 的 AR 在长 sequence 上 error accumulate。
2. **Geometry kernel 没在 loop 里**：生成的 CAD 可能 invalid，没有 verifier feedback。这和 LLM + tool use（如 Program-aided reasoning）的 gap 一样。
3. **Latent bottleneck**：256-dim latent 要 encode 整个 CAD program 的所有信息，对于复杂 CAD 可能不够。
4. **Dataset scale**：168k CAD models 远不如 LAION-5B 的 image-text pairs。CAD data 的 long tail（rare command types, complex topology）很难覆盖。

---

## 9. 相关工作与延伸阅读

### CAD as language
- **DeepCAD** (Wu et al., 2021): https://arxiv.org/abs/2105.01890 - 第一个大规模 CAD command sequence dataset + l-GAN
- **SkexGen** (Xu et al., 2022): https://arxiv.org/abs/2207.04632 - disentangled codebook for CAD construction sequences
- **BrepGen** (Xu et al., 2024b): https://arxiv.org/abs/2401.15563 - diffusion for B-rep directly
- **ContrastCAD** (Jung et al., 2024): https://arxiv.org/abs/2404.01645 - contrastive learning for CAD
- **CAD as language** (Ganin et al., 2021): https://arxiv.org/abs/2105.01890 - protocol buffers for CAD
- **SketchGen** (Para et al., 2021): https://arxiv.org/abs/2107.04632

### Datasets
- **ABC dataset** (Koch et al., 2019): https://archive.nyu.edu/collections/4v6vz4q8/ - 1M CAD models
- **Fusion 360 gallery** (Willis et al., 2021): https://github.com/AutodeskAILab/Fusion360GalleryDataset
- **MFCAD / MFCAD++**: machining feature recognition

### Image-to-CAD retrieval
- **ROCA** (Gümeli et al., 2022): https://arxiv.org/abs/2204.06214 - CAD retrieval + alignment from single image
- **DiffCAD** (Gao et al., 2024): https://arxiv.org/abs/2401.09115 - weakly-supervised CAD retrieval

### Foundation models (recipe 来源)
- **Transformer**: https://arxiv.org/abs/1706.03762
- **CLIP**: https://arxiv.org/abs/2103.00020
- **DDPM**: https://arxiv.org/abs/2006.11239
- **Latent Diffusion (Stable Diffusion)**: https://arxiv.org/abs/2112.10752
- **DALL-E 2 (unCLIP)**: https://cdn.openai.com/papers/dall-e-2.pdf
- **ResNet-MLP for tabular** (Gorishniy et al., 2021): https://arxiv.org/abs/2106.11959

### 3D generation (non-CAD)
- **Shape-E** (Jun & Nichol, 2023): https://arxiv.org/abs/2305.02463
- **InstantMesh** (Xu et al., 2024a): https://arxiv.org/abs/2404.07191
- **PointFlow / DPM** (Achlioptas et al., 2018): https://arxiv.org/abs/1707.02392

---

## 10. 可能的下一步联想

基于这个工作，几个我觉得 promising 的方向（hallucinate 一下）：

1. **RLHF-style refinement with geometry kernel as reward**：把 OpenCascade 当 verifier，invalid CAD 给 negative reward，fine-tune decoder with RL 或 best-of-n rejection sampling。类似 Lean theorem proving 里的 verifier-in-the-loop。

2. **Text-to-CAD**：加 text encoder，contrastive align text-CAD-image 三元组。Text 描述（"a bracket with two mounting holes and a rib"）+ image 共同 condition。这是 Text2CAD 的方向。

3. **Multi-view engineering drawing input**：工程师实际用 front/top/side 三视图。Encoder 改成 multi-view ViT，cross-view attention。

4. **CAD-specific tokenizer with BPE**：现在每个 command 是一个 token，可以用 byte-pair encoding 学 CAD command 的 sub-sequences 作为 compound tokens（比如 "line-line-line-extrude" 作为一个 macro token），加速 long sequence 生成。

5. **Hierarchical diffusion**：先 diffusion 出 sketch topology（粗粒度），再 diffusion 出 parameters（细粒度）。类似 cascaded diffusion in image generation。

6. **Diffusion in parameter space, not latent space**：直接在 command sequence 上做 discrete diffusion（类似 VQ-Diffusion / D3PM），避免 latent bottleneck。但 inference 慢。

7. **In-context learning for CAD editing**：把 CAD editing history 作为 context，让 model in-context 学习 edit pattern。CAD 版的 GPT-4 code editing。

8. **Test-time scaling**：diffusion sample 多个 candidates，用 geometry kernel + chamfer distance to input image 做 verifier，best-of-n selection。类似 code generation 的 majority voting。

9. **Neural geometry kernel**：现在的 OpenCascade 是 deterministic 的 rule-based kernel，可学习 kernel 可以 propagate gradients through CAD construction，enable end-to-end differentiable CAD optimization。

10. **Symmetry / topology priors**：CAD 有大量 rotational / mirror symmetry。在 latent space 加 symmetry-aware augmentation 或 equivariant encoder 可以大幅提升 sample efficiency。

---

## 11. 一句话总结

**GenCAD = unCLIP recipe applied to CAD programs**：autoregressive transformer 学 CAD sequence 的 latent（代替 CLIP text encoder），ResNet-18 学 image latent，contrastive loss 对齐两者，latent diffusion prior 学 image→CAD latent 的 conditional distribution，frozen AR decoder 把 latent 解码成 CAD command sequence，最后 OpenCascade 把 sequence 转 B-rep。Recipe 没有novelty，但 application domain（CAD）的工程化和 image-to-CAD retrieval / generation 的实际价值是真实的，15× retrieval improvement 和 conditional > unconditional 的实验结论也印证了 recipe 的 transferability。

希望这些细节帮你 build intuition, Andrej。如果你想深入某个 component（比如 ResNet-MLP denoising 的具体架构，或者 contrastive loss 的 batch size effect），告诉我。
