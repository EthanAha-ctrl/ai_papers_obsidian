---
source_pdf: GenCAD Image-Conditioned Computer-Aided Design Gen.pdf
paper_sha256: 94c6a3cd824175bdc9e98da8c9fd1b1a75c3ec15d8df0fc74ceef05127d79f75
processed_at: '2026-08-19T09:09:07-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenCAD 人话版

Andrej，我用最直白的方式再讲一遍，省去 paper 的学术腔，直接讲这玩意儿在干嘛、为啥这么干、效果咋样。

参考链接：
- Paper: https://arxiv.org/abs/2405.17176
- DeepCAD: https://arxiv.org/abs/2105.03031
- DALL-E 1: https://arxiv.org/abs/2102.12092
- CLIP: https://arxiv.org/abs/2103.00020
- DDPM: https://arxiv.org/abs/2006.11239
- LDM: https://arxiv.org/abs/2112.10752

---

## 这篇 paper 到底想解决啥问题

你给一张 CAD model 的图（或者 hand-drawn sketch），model 吐出一个 CAD program——就是一连串 parametric command，比如"画一条线从这到那""画个圆""extrude 多远"，能直接喂给 Onshape / OpenCascade 这种 geometry kernel 渲出 3D solid，而且**engineer 还能 edit**。

这个事为啥难：

第一，engineer 真正要的是 B-rep + design history。mesh、voxel、point cloud 都不行，因为没法 edit、没法 manufacture。一个 hole 是 drill 出来的还是 sketch 里就画好的？这个区别在 mesh 里完全丢了。

第二，B-rep 本身是个 graph（face-edge-vertex 邻接关系），neural network 不好直接吃。而且直接生成 B-rep 等于生成"结果"，丢了"过程"。

第三，以前有人做 CAD generation（DeepCAD、SkexGen、BrepGen），但都是 **unconditional**——随机生成，跟用户意图毫无关系。这就跟 2019 年的 GAN 生成随机人脸一样，玩具而已。

GenCAD 想做的就是 **image-conditioned CAD program generation**，这在该领域是头一遭。

---

## 把 CAD 当语言问题来处理

关键 insight：**CAD modeling 本质上是写代码**。

工程师在 Onshape 里点"画线""extrude""fillet"，每一步都是一条 command。把这些 command 串起来就是一个 sequence，喂给 geometry kernel 就能 reconstruct 出 3D solid。这跟 Python 代码 → interpreter 执行一模一样。

所以 paper 把每条 CAD command 编码成一个 17 维向量：

$$\mathbf{c}_i = (t_i, \mathbf{p}_i)$$

- $t_i$：command type，离散的，比如 Line=1, Arc=2, Circle=3, Extrude=4
- $\mathbf{p}_i$：参数向量 16 维，比如 Line 就是终点坐标 $(x,y)$，Circle 是圆心 $(x,y)$ + 半径 $r$，Extrude 有 sketch plane orientation $(\theta,\phi,\gamma)$、plane origin $(p_x,p_y,p_z)$、scale $s$、extrude distance $(e_1,e_2)$、boolean op $b$（join/cut/intersect）、unilateral flag $u$

参数有连续有离散，paper 把所有 parameter **quantize 到 8-bit (256 levels)**，统一成离散 token，跟 NLP 里的 word token 一样处理。

这样一个 CAD model 就是一个 sequence of tokens，平均长度 60 左右。CAD modeling 就变成了 language modeling。

---

## 整个 pipeline 四步走

我画个 flow：

```
CAD sequence ──► [Step 1: CSR] ──► z_CAD (256-dim latent)
                                         │
Image ──► [ResNet-18] ──► z_img ──► [Step 2: CCIP] ◄─ CLIP-style 对齐
                                         │
                                         ▼
                                  [Step 3: CDP diffusion prior]
                                         │ 给定 z_img 采样 z_CAD
                                         ▼
                                  [Step 4: frozen decoder]
                                         │
                                         ▼
                                  CAD command sequence
                                         │
                                         ▼
                                  geometry kernel ──► 3D B-rep
```

### Step 1: CSR (Command Sequence Reconstruction)

目的：学一个 transformer encoder-decoder，能 reconstruct CAD command sequence，中间逼出一个 256 维的 latent $\mathbf{z}_{\text{CAD}}$ 压缩住整个 sequence 的 design intent。

架构：
- Transformer encoder + decoder，各 4 层，8 attention heads
- Causal masking（autoregressive，跟 GPT 一样）
- Encoder 输出 per-timestep 的 latent，average pool 成单个 $\mathbf{z}_{\text{CAD}} \in \mathbb{R}^{256}$
- Constant embedding 把 $\mathbf{z}_{\text{CAD}}$ 投回 sequence，喂 decoder
- Decoder 输出 reconstruct 的 CAD command

Loss（Eq. 1）：

$$\mathcal{L} = \sum_{i=1}^{N_c} \ell(\hat{t}_i, t_i) + \beta \sum_{i=1}^{N_c} \sum_{j=1}^{N_p} \ell(\hat{\mathbf{p}}_{ij}, \mathbf{p}_{ij})$$

- $N_c=6$：command type 数
- $N_p=16$：每条 command 的 parameter slot 数
- $\ell$：cross-entropy（因为 parameter 被 quantize 成 256 levels，可视为分类）
- $\beta$：balance 权重

参数量才 6.72M，Adam lr=1e-3，2000 warmup，1000 epochs，batch 512。

**这一步跟 DeepCAD 的区别**：DeepCAD 的 encoder 是 bidirectional 的（没 causal mask），GenCAD 加了 causal mask。结果就是 GenCAD 的 latent 显式编码了"程序执行顺序"的信息，reconstruction 在长 sequence 上明显赢 DeepCAD（Figure 5）。

### Step 2: CCIP (Contrastive CAD-Image Pre-training)

目的：学一个 joint latent space，让 image 和对应 CAD 在 latent space 里对齐。冻结 Step 1 的 CAD encoder，加一个 ResNet-18 image encoder：

- Image resize 到 256×256，normalize $\mathcal{N}(0.5, 0.5)$
- ResNet-18 4 个 stage，输出 512×8×8，linear projection 到 256-dim
- 每 encoder block 加 dropout 0.1

Loss 是标准 SimCLR / CLIP 的 InfoNCE（Eq. 2）：

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},j})/\tau)}{\sum_{k=1}^{2B} \mathbb{1}_{[k\neq i]} \exp(\text{sim}(\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},k})/\tau)}$$

- $B$：batch size，256
- $\text{sim}$：cosine similarity
- $\tau$：temperature
- 正样本：配对的 (image, CAD)，负样本：batch 内其他所有组合

**Image augmentation 的关键**：每个 CAD 在 x/y/z 三个 axis 做不同 scale，生成 5 个 variant。这让 model 学到"同样的 CAD 在不同 scale 下 latent 应该接近"，等于自带 scale-invariance。

总参数 28.22M，Adam + weight decay，500 epochs。

### Step 3: CDP (CAD Diffusion Prior)

目的：给定 $\mathbf{z}_{\text{img}}$，采样出 $\mathbf{z}_{\text{CAD}}$。

为啥不直接 deterministic map $\mathbf{z}_{\text{img}} \to \mathbf{z}_{\text{CAD}}$？因为**同一张图可能对应多个合理的 CAD**——同一个 3D shape 可以用不同的 command sequence 构造。deterministic map 会把所有可能平均成一个 blur，跟 GAN 生成模糊人脸一个道理。

所以 paper 学分布 $p(\mathbf{z}_{\text{CAD}} | \mathbf{z}_{\text{img}})$，用 DDPM。

**Forward diffusion**：$\mathbf{z}_{\text{CAD}}$ 加 500 步 Gaussian 噪声
**Reverse denoising**：学 $\epsilon_\theta(\mathbf{z}_t, t, \mathbf{z}_{\text{img}})$ 预测噪声

输入构造：noised latent $\mathbf{z}_t$ 跟 condition $\mathbf{z}_{\text{img}}$ concatenate，过 projection，喂 denoising network。

**Denoising network = ResNet-MLP**（Gorishniy et al. 2021 的 tabular DDPM 同款）：
- 10 个 MLP-ResNet block
- 2048 维 projection
- dropout 0.1

为啥用 ResNet-MLP 不用 U-Net？因为 $\mathbf{z}_{\text{CAD}}$ 是 256 维 vector，没有 spatial structure，U-Net 的 conv assumption 不成立。

训练：lr=1e-5 固定，batch 2048，grad accumulation 2 step，1M timestep，500 diffusion step，grad clip 1.0。

这个思路直接来自 OpenAI 的 unCLIP / DALL-E 2——用 diffusion prior bridge 两个 modality 的 latent space。Karpathy 你应该熟。

### Step 4: CAD Decoder

直接复用 Step 1 训练好的 CSR decoder，冻结。把 Step 3 采样的 $\mathbf{z}_{\text{CAD}}$ 喂进去，autoregressive 出 command sequence。前面 prepend 一个 ⟨SOL⟩ token。

这步的好处：**不用重新训 decoder**，scalable 到大数据集。CSR decoder 已经学好怎么从 latent 解 command，prior 只负责生成正确的 latent distribution。

---

## 数据

基于 DeepCAD（源自 ABC dataset）：
- 168,674 个 CAD（经过 OpenCascade validity 过滤）
- 152,530 train / 8,515 val / 7,629 test
- 每 CAD 生成 5 个 scale variant 的 grayscale image（1×448×448）
- 总共 845,105 张 image
- Sketch 数据：image 做 Canny edge + Gaussian blur，也 845,105 张

---

## 实验结果

### 结果 1: CSR reconstruction（Table 1）

| Method | µ_cmd ↑ | µ_param ↑ | µ_CD ↓ | IR ↓ |
|---|---|---|---|---|
| DeepCAD | 99.36 | 97.59 | 0.783 | 3.44 |
| GenCAD | **99.51** | **97.78** | **0.762** | **3.32** |

- $\mu_{\text{cmd}}$：command type 准确率
- $\mu_{\text{param}}$：parameter 准确率
- $\mu_{\text{CD}}$：mean chamfer distance（2000 点 cloud）
- IR：invalid ratio（geometry kernel 渲染失败的）

GenCAD 全面小赢 DeepCAD。重点是 **sequence 越长优势越大**（Figure 5），因为 causal autoregressive 让 latent 编码更多 sequential info。

### 结果 2: Image-based CAD retrieval（Table 2）——最有亮点的

| Method | R@10 | R@128 | R@1024 | R@2048 |
|---|---|---|---|---|
| Random | 10.06 | N/A | N/A | N/A |
| ResNet-18 (image-to-image) | 77.70 | 19.26 | 5.21 | 3.91 |
| GenCAD-image | **98.49** | **91.41** | **70.28** | **60.77** |
| GenCAD-sketch | 98.36 | 87.5 | 70.67 | 60.77 |

R@2048 时 60.77% vs 3.91%，**15.6× 提升**。这就是 paper 标题里强调的卖点。说明 joint latent 学得非常对齐。

### 结果 3: Generation quality（Table 3）

| Method | COV ↑ | MMD ↓ | JSD ↓ |
|---|---|---|---|
| DeepCAD (l-GAN) | 78.13 | 1.45 | 3.76 |
| SkexGen | 78.17 | 1.55 | 4.89 |
| BrepGen | 73.10 | **1.05** | **1.22** |
| ContrastCAD | 78.93 | 1.44 | 3.67 |
| GenCAD (uncond) | 78.27 | 1.44 | 3.94 |
| GenCAD-image | 81.37 | 1.38 | 3.49 |
| GenCAD-sketch | **82.59** | **1.33** | 3.53 |

**Metric 直觉**：
- **COV (Coverage)**：generated set 覆盖 reference set 的 fraction，测多样性
- **MMD (Min Matching Distance)**：每个 reference 到最近 generated 的平均 chamfer，测保真度
- **JSD**：两个点云分布的统计距离

GenCAD conditional 在 COV 和 MMD 都 best。BrepGen 在 MMD/JSD 略好但 COV 73.10 很差——因为它直接生成 B-rep，保真度高但多样性低。GenCAD 牺牲一点保真换更多样 + 可 editability + 条件输入。

### 结果 4: FID alignment（Figure 12）

FID 公式：

$$FID = \|\mu_\mathcal{S} - \mu_\mathcal{G}\|_2^2 + \text{tr}\left(\Sigma_\mathcal{S} + \Sigma_\mathcal{G} - 2(\Sigma_\mathcal{S}\Sigma_\mathcal{G})^{1/2}\right)$$

排序：**image-conditional diffusion < sketch-conditional < unconditional diffusion < l-GAN < deterministic prior**。

deterministic prior 最差强有力证明 multimodal diffusion prior 必要——直接 map image→CAD latent 会丢失 multimodality。

---

## Ablation 简述

**CSR 层数**（Table 5）：4 层最佳。2 层 CD/IR 更好但 cmd/param 准确度差，6 层直接崩（数据不够撑）。选 4 层。

**Image encoder**（Table 6）：ResNet-18 最佳。ResNet-34 过拟合，ViT 在数据不够时不如 conv。

---

## 跟其他 image-to-3D 方法的对比（Figure 14）

跟 image-to-mesh 模型（PolyDiff、InstantMesh、Shap-E）定性对比：这些模型生成 mesh，不能 edit，捕捉不到 mechanical feature（hole、slot、pocket）。GenCAD 输出 command sequence，直接 import 到 Onshape 当 feature tree 编辑（Figure 13）。

---

## Limitations

1. **CAD vocabulary 太小**：只 Line/Arc/Circle + Extrude。工业 CAD 的 revolve、fillet、chamfer、mirror、loft、sweep 都没。这是最大短板。
2. **不保证 valid CAD**：IR=3.32% 还是会失败，没 geometry kernel verifier in-the-loop。
3. **Image 只测 isometric 渲染图**：real-world photo 没测。
4. 数据量 168k 在 LLM 时代算小，model 只能 scale 到 6.72M-28M 参数。

---

## 我的几个直觉判断

### 这架构哲学就是 DALL-E 1 / unCLIP 血脉

CSR = dVAE tokenizer + decoder，CCIP = CLIP joint embedding，CDP = diffusion prior bridge，decoder = decoder。完全照搬 DALL-E 1 → unCLIP → DALL-E 2 这条线。CAD domain 终于等到这套 pattern 被搬过来。

### 跟 LLM code generation 的类比特别 clean

CAD program 就是代码，geometry kernel 就是 interpreter。CSR 是 code autoencoder，CCIP 是 (image, code) contrastive，CDP 是 image-conditioned code prior。如果未来扩 vocabulary 到几十种 op，就是真正的 CAD code LLM。可以联想 code-Caption-Python 三段式。

### Verification feedback 是自然延伸

paper 提 limitation 没 verifier in-the-loop。一个明显方向是让 geometry kernel 当 critic 给 reward，做 rejection sampling 或 RL fine-tune decoder。Karpathy 你应该会想到这跟 LLM 里 execution-guided decoding / RL from execution 一个套路。

### Conditional > unconditional 这点很有意思

Table 3 显示 conditional COV 82.59 比 unconditional 78.27 还高。直觉上 condition 会限制多样性，但这里反而提升。我想是因为 image 既 constrain 又 hint——给 model 一个 "anchor" 帮它 navigate latent space 到更有意义的区域，避免 unconditional 在 mode 之间瞎转。

### ResNet-18 赢 ViT 在这数据规模下合理

168k CAD 太小，ViT 的 inductive bias 弱反而吃亏。如果 scale 到 10M+ CAD，ViT 可能反超。这是 data scale vs architecture inductive bias 的经典 trade-off。

### 真正的突破点在 retrieval

60.77% @ 2048 batch 的 retrieval 比 image-to-image 的 3.91% 提升 15×，这比 generation 部分更有实用价值。大型 CAD library（GrabCAD、TraceParts 都几百万 model）的 image-based retrieval 是工业界真痛点。joint latent 学得对齐到这程度，可以直接 deploy。

### 未来要 watch 的方向

1. CAD vocabulary 扩到几十种 op——需要 scale data 和 model
2. Real-world photo 输入——需要 viewpoint estimation + background removal 前置 stage
3. Geometry kernel verifier in-the-loop——rejection sampling / RL fine-tune
4. Multi-view + text 多模态条件——扩展 CCIP framework
5. 跟 DiffCAD / ROCA 这种 image-based CAD retrieval 弱监督方法正面比

---

## 一句话总结

GenCAD 把 image-to-CAD 这问题 properly 形式化成 language modeling + CLIP joint embedding + latent diffusion prior 三件套，是 DALL-E 1 在 CAD domain 的对应物。架构成熟度未必最高，但 problem framing 和 baseline 建立意义重大。retrieval 15× 提升是最 solid 的结果，conditional generation 的 COV/MMD 全面赢 unconditional baseline 也证明 image condition 对 CAD 生成有真实增益。最大短板是 CAD vocabulary 太窄，扩 vocabulary 是下一步必须做的。

---

# GenCAD 深度技术讲解

Andrej，这篇 paper 来自 MIT 的 Ferdous Alam 和 Faez Ahmed，发表在 2024 年左右。核心命题是：**把 image/sketch 输入转成可 edit、可 manufacture 的 parametric CAD command sequence**。读完之后我立刻想到的是 DALL-E 1 + unCLIP 的架构思路被迁移到了 CAD domain——contrastive pretraining + latent diffusion prior + autoregressive decoder 的三段式 pipeline。下面我从直觉到技术细节一层层拆。

参考链接：
- Paper PDF: https://arxiv.org/abs/2405.17176 (GenCAD)
- Project page (MIT Design Lab): https://MIT-Design-Intelligence.github.io/GenCAD/
- DeepCAD (Wu et al., 2021): https://arxiv.org/abs/2105.03031
- SkexGen (Xu et al., 2022): https://arxiv.org/abs/2207.04632
- BrepGen (Xu et al., 2024b): https://arxiv.org/abs/2401.15563
- ContrastCAD (Jung et al., 2024): https://arxiv.org/abs/2404.01645
- DALL-E 1 (Ramesh et al., 2021): https://arxiv.org/abs/2102.12092
- CLIP (Radford et al., 2021): https://arxiv.org/abs/2103.00020
- DDPM (Ho et al., 2020): https://arxiv.org/abs/2006.11239
- LDM / Stable Diffusion (Rombach et al., 2022): https://arxiv.org/abs/2112.10752

---

## 1. 动机与 problem framing 的直觉

工程 CAD 跟一般 3D shape generation (mesh, voxel, point cloud, NeRF/SDF) 最大的差别在于：**最终交付物是 B-rep + design history**。B-rep 是 industry standard（自 1980s 的 Weiler 提出的 radial edge structure 起），它由 parametric surface、edge、vertex 组成，不 resolution-dependent，可以无缝喂给 OpenCascade / Parasolid / ACIS 这些 geometry kernel。

但直接生成 B-rep 有两个 hard 问题：
1. **topology 复杂**：face-edge-vertex 的邻接关系是 graph，NN 不友好；
2. **丢失 design intent**：B-rep 是 "结果"，不带 "过程"。一个 fillet 是后来加的还是 sketch 里就画好的？extrude 是 join 还是 cut？这些信息只在 command sequence 里。

所以 GenCAD 选择生成 **CAD program**（command sequence），每条 command 比如 `Line(x,y)`、`Circle(x,y,r)`、`Arc(x,y,α,f)`、`Extrude(θ,φ,γ,px,py,pz,s,e1,e2,b,u)`。这相当于把 CAD modeling 当成一种 "language"，把 geometry kernel 当成 "interpreter"——非常 analogous to 代码生成 + 编译器执行。这点直觉上特别 clean。

> 重要 insight：CAD command 的 type 是 discrete token（像 NLP word），但 parameters 是 continuous（坐标、角度、长度）。作者把所有 parameter quantize 到 8-bit (256 levels)，统一成 discrete 输入，但 latent 学的是 continuous space。

---

## 2. CAD vocabulary 与 command 表示

每个 CAD command 被表示成一个 17 维向量：

$$\mathbf{c}_i = (t_i, \mathbf{p}_i), \quad \mathbf{c}_i \in \mathbb{R}^{17}$$

- $t_i \in \mathbb{R}$：command type（离散 id）
- $\mathbf{p}_i \in \mathbb{R}^{16}$：参数向量（mask 掉不用的 slot）

**Token 字典**（参见 Table 4 Appendix）：

| index | token | parameters (16-dim) |
|---|---|---|
| 0 | ⟨SOL⟩ | ∅ (start of loop) |
| 1 | Line | [x, y, □, ..., □] (2 used) |
| 2 | Arc | [x, y, α, f, □, ..., □] (4 used) |
| 3 | Circle | [x, y, □, □, r, □, ..., □] (3 used) |
| 4 | Extrude | [□,□,□,□,□, θ, φ, γ, p_x, p_y, p_z, s, e_1, e_2, b, u] (10 used) |
| 5 | ⟨EOS⟩ | ∅ |

**变量解释**：
- Line：终点坐标 $(x,y)$，起点由上一条 sketch command 决定（sequential 生成）
- Arc：终点 $(x,y)$、sweep angle $\alpha$、direction flag $f \in \{0,1\}$（顺/逆时针）
- Circle：圆心 $(x,y)$、半径 $r$
- Extrude：sketch plane orientation $(\theta,\phi,\gamma)$（欧拉角）、plane origin $(p_x,p_y,p_z)$、sketch scale $s$、两侧 extrude distance $(e_1,e_2)$、boolean op $b$（new/join/cut/intersect）、unilateral flag $u$

这种 formulation 跟 DeepCAD (Wu et al., 2021) 一致，DeepCAD 也是他们的 baseline。一个 sequence 长度典型在 60 左右（padded 到 $N$）。

---

## 3. GenCAD 四步框架（架构图解析）

参考 Figure 2，整个 pipeline 像这样：

```
CAD sequence ──► [CSR Encoder] ──► z_CAD ──────────┐
                                                    │
Image ──► [ResNet-18] ──► z_img ──► [CCIP] ◄─contrastive─┤
                                       │
                                       ▼
                                  [CDP diffusion prior]
                                       │ (sampled z_CAD given z_img)
                                       ▼
z_CAD ──► [CSR Decoder (frozen)] ──► CAD command sequence ──► geometry kernel ──► B-rep/mesh
```

四个 step：

### Step 1: Command Sequence Reconstruction (CSR) — 自监督学 latent
- Transformer encoder + decoder，causal masking，autoregressive
- 输入 CAD command sequence $\{\mathbf{c}_i\}_{i=1}^N$ → encoder 输出 $\mathbf{z}_{\text{CAD},t} \in \mathbb{R}^{d_z}$，每个 timestep 一个
- Average pooling 得到单一 latent $\mathbf{z}_{\text{CAD}} \in \mathbb{R}^{d_z}$，$d_z=256$
- Constant embeddings 把 $\mathbf{z}_{\text{CAD}}$ 投回 sequence，喂 decoder
- Decoder 输出 reconstruct CAD commands

**关键 loss**（Eq. 1）：

$$\mathcal{L} = \sum_{i=1}^{N_c} \ell(\hat{t}_i, t_i) + \beta \sum_{i=1}^{N_c} \sum_{j=1}^{N_p} \ell(\hat{\mathbf{p}}_{ij}, \mathbf{p}_{ij})$$

- $N_c=6$（最多 6 种 command type）
- $N_p=16$（每条 command 的参数数）
- $t_i, \hat{t}_i$：ground truth / predicted command type
- $\mathbf{p}_{ij}, \hat{\mathbf{p}}_{ij}$：第 $i$ 条 command 第 $j$ 个参数 ground truth / predicted
- $\ell(\cdot,\cdot)$：cross-entropy（因为 parameter quantize 成 256 levels，可视为分类问题）
- $\beta$：balance type loss 和 parameter loss 的权重

**架构超参**（Appendix E.1）：
- 4 层 encoder + 4 层 decoder transformer
- 8 attention heads
- feed-forward dim 512，dropout 0.1
- 6.72M trainable parameters
- Adam, lr=1e-3, 2000 warmup steps, 1000 epochs, batch size 512, grad clip 1.0

直觉：因为 autoregressive，每一步预测 next command 时都能看到之前的所有 command，因此 latent $\mathbf{z}_{\text{CAD}}$ 必须压缩住整个 sequence 的 design intent。这一点跟 DeepCAD 的 transformer autoencoder 的区别在于 DeepCAD encoder 是 bidirectional 的 non-causal，而 GenCAD 加了 causal mask。后者把 "程序生成顺序" 显式编码进了 latent。

### Step 2: Contrastive CAD-Image Pre-training (CCIP) — 学 joint latent space

冻结 CSR encoder，加一个 ResNet-18 image encoder：

- Image preprocessing：resize 到 256×256，center crop，normalize $\mathcal{N}(0.5, 0.5)$
- ResNet-18：4 个 stage，dim 分别 64/128/256/512，每 stage 2 个 conv block，输出 512×8×8
- Linear projection 到 $d_z=256$，得到 $\mathbf{z}_{\text{img}} \in \mathbb{R}^{d_z}$
- 在每个 encoder block 加 dropout 0.1

**Loss 是 InfoNCE**（Eq. 2，源自 SimCLR Chen et al. 2020）：

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},j})/\tau)}{\sum_{k=1}^{2B} \mathbb{1}_{[k\neq i]} \exp(\text{sim}(\mathbf{z}_{\text{CAD},i}, \mathbf{z}_{\text{image},k})/\tau)}$$

$$\mathcal{L} = \frac{1}{2B} \sum_{k=1}^{B} [\ell(2k-1, 2k) + \ell(2k, 2k-1)]$$

- $B$：batch size（256）
- $2B$：因为 batch 里有 $B$ 个 image 和 $B$ 个 CAD，共 $2B$ 个样本
- $\text{sim}(\mathbf{u},\mathbf{v}) = \mathbf{u}^T\mathbf{v}/(\|\mathbf{u}\|\|\mathbf{v}\|)$：cosine similarity
- $\tau$：temperature（这里没明说数值，常见取 0.07~0.5）
- $\mathbb{1}_{[k\neq i]}$：indicator 排除自己

直觉：这就是 CLIP 的 InfoNCE，把 (CAD, image) 对拉到一起，batch 内非对 sample 推开。这里 image augmentation（5 个 scale 变体）很重要——它让 model 学会 "同样的 CAD 在不同 scale 下 z_CAD 应该接近"，等于自带了 scale-invariance 的 augmentation。

总参数 28.22M。Adam + weight decay，lr=1e-3 ReduceLROnPlateau，500 epochs。

### Step 3: CAD Diffusion Prior (CDP) — 条件生成 z_CAD

这里 paper 借鉴了 OpenAI unCLIP / DALL-E 1 的 diffusion prior 思路——不直接学 image→z_CAD 的 deterministic map，而是学分布 $p(\mathbf{z}_{\text{CAD}} | \mathbf{z}_{\text{img}})$。因为同样的 image 可能对应多个合理的 CAD（multimodal），deterministic map 会把所有可能平均成一个 blur。

**Forward diffusion**：把 $\mathbf{z}_{\text{CAD}}$ 按 DDPM 加 Gaussian 噪声 500 步得到 $\mathbf{z}_t$。
**Reverse denoising**：学一个网络 $\epsilon_\theta(\mathbf{z}_t, t, \mathbf{z}_{\text{img}})$ 预测噪声。

输入构造：把 noised latent $\mathbf{z}_t$ 和 condition $\mathbf{z}_{\text{img}}$ **concatenate**，过一个 projection 层，然后喂 denoising network。

**Denoising network = ResNet-MLP**（Gorishniy et al., 2021，Tabular DDPM 同款）：

- 10 个 MLP-ResNet block
- 2048 维 projection layer
- dropout 0.1
- 用 ResNet-MLP 而非 U-Net，是因为数据是 1D latent vector，没有 spatial structure，U-Net 的 conv 不必要

训练：固定 lr=1e-5，batch size 2048，gradient accumulation 每 2 step，1M timestep，grad clip 1.0，500 diffusion step。

**对比 baseline：deterministic prior**（MLP 直接 $\mathbf{z}_{\text{img}} \to \mathbf{z}_{\text{CAD}}$），paper 也跑了，结果 FID 比 diffusion prior 差很多，印证了 multimodal 假设。

### Step 4: CAD Decoder — 把 z_CAD 解码成 command sequence

冻结 Step 1 训练的 CSR decoder，把 CDP 采样出的 $\mathbf{z}_{\text{CAD}}$ 喂进去，autoregressive 生成 $\mathbf{c}_2, \dots, \mathbf{c}_{N+1}$，前面 prepend 一个 ⟨SOL⟩。

这步的设计直觉：**避免重新训练 decoder**，复用 Step 1 的 decoder 能力，scalable 到大数据集。

---

## 4. 数据集

基于 **DeepCAD**（Wu et al., 2021）：
- 源自 ABC dataset（Koch et al., 2019，1M CAD）
- DeepCAD 通过 Onshape API 解析 design history，过滤只保留 sketch + extrude
- 过滤后 178,238 CAD designs
- GenCAD 进一步用 OpenCascade 过滤能否 valid 渲染 → 168,674 CAD

划分：152,530 train / 8,515 val / 7,629 test

**Image 增强**：每个 CAD 在 x/y/z 三个 axis 做不同 scale，生成 5 个 variant，最终 845,105 张 image，全部 grayscale 1×448×448。

**Sketch 数据**：image 上做 Canny edge + Gaussian blur，共 845,105 sketch。这让 model 同时学 isometric image 和 line-drawing sketch 两种 condition。

---

## 5. 实验结果

### 5.1 CSR（Table 1）

| Method | µ_cmd ↑ | µ_param ↑ | µ_CD ↓ | IR ↓ |
|---|---|---|---|---|
| DeepCAD | 99.36 | 97.59 | 0.783 | 3.44 |
| GenCAD | **99.51** | **97.78** | **0.762** | **3.32** |

- $\mu_{\text{cmd}}$：command type 预测准确率
- $\mu_{\text{param}}$：参数预测准确率（阈值 $\eta$ 内算对）
- $\mu_{\text{CD}}$：mean chamfer distance（2000 点 cloud）
- IR：invalid ratio（geometry kernel 渲染失败的）

Figure 5 显示：sequence length 越长，GenCAD 比 DeepCAD 优势越明显。直觉：causal autoregressive 让 latent 编码更多 sequential 信息。

### 5.2 Retrieval（Table 2，最有亮点的一个结果）

| Method | R@10 | R@128 | R@1024 | R@2048 |
|---|---|---|---|---|
| Random | 10.06 | N/A | N/A | N/A |
| ResNet-18 (image-to-image baseline) | 77.70 | 19.26 | 5.21 | 3.91 |
| GenCAD-image | **98.49** | **91.41** | **70.28** | **60.77** |
| GenCAD-sketch | 98.36 | 87.5 | **70.67** | 60.77 |

batch=2048 时 60.77% vs 3.91%，**约 15× 提升**——这是 paper 标题强调的关键卖点。

### 5.3 Unconditional generation（Table 3）

| Method | COV ↑ | MMD ↓ | JSD ↓ |
|---|---|---|---|
| DeepCAD (l-GAN) | 78.13 | 1.45 | 3.76 |
| SkexGen | 78.17 | 1.55 | 4.89 |
| BrepGen | 73.10 | **1.05** | **1.22** |
| ContrastCAD + RRE | 78.93 | 1.44 | 3.67 |
| GenCAD (uncond) | 78.27 | 1.44 | 3.94 |
| GenCAD-image (cond) | 81.37 | 1.38 | 3.49 |
| GenCAD-sketch (cond) | **82.59** | **1.33** | 3.53 |

**Metric 公式**（Appendix E.4）：

- **Coverage (COV)**：reference set $\mathcal{S}$ 中被 generated set $\mathcal{G}$ 至少 match 一个的 fraction

$$COV(\mathcal{S}, \mathcal{G}) = \frac{|\{\text{argmin}_{Y\in\mathcal{S}} d_{CD}(X,Y) | X \in \mathcal{G}\}|}{|\mathcal{S}|}$$

- **Minimum Matching Distance (MMD)**：每个 reference 形状到 nearest generated 的平均距离

$$MMD(\mathcal{S}, \mathcal{G}) = \frac{1}{|\mathcal{S}|} \sum_{Y\in\mathcal{S}} \min_{X\in\mathcal{G}} d_{CD}(X,Y)$$

- **Jensen-Shannon Divergence (JSD)**：两个点云 marginal 分布的统计距离

$$JSD(\mathcal{P}_\mathcal{S}, \mathcal{P}_\mathcal{G}) = \frac{1}{2} D_{KL}(\mathcal{P}_\mathcal{S} \| M) + \frac{1}{2} D_{KL}(\mathcal{P}_\mathcal{G} \| M), \quad M = \frac{\mathcal{P}_\mathcal{S}+\mathcal{P}_\mathcal{G}}{2}$$

直觉：COV 测多样性、MMD 测保真度、JSD 测分布距离。GenCAD conditional 在 COV 和 MMD 都 best，JSD 略输给 BrepGen——BrepGen 用 structured latent geometry 在 B-rep 上做 diffusion，保真度天生好但牺牲多样性。

### 5.4 FID 评估 alignment（Figure 12）

把 generated CAD 的 latent 和 test set 的 latent 看成两个高斯 $\mathcal{N}(\mu_\mathcal{S}, \Sigma_\mathcal{S})$ 和 $\mathcal{N}(\mu_\mathcal{G}, \Sigma_\mathcal{G})$，FID 公式：

$$FID = \|\mu_\mathcal{S} - \mu_\mathcal{G}\|_2^2 + \text{tr}\left(\Sigma_\mathcal{S} + \Sigma_\mathcal{G} - 2(\Sigma_\mathcal{S}\Sigma_\mathcal{G}))^{1/2}\right)$$

排序：image-conditional diffusion prior < sketch-conditional < unconditional diffusion < l-GAN < deterministic prior。deterministic 最差强有力地证明了 multimodal diffusion prior 的必要性。

### 5.5 Ablation（Table 5, Table 6）

**CSR 层数 ablation**：

| layers | µ_cmd | µ_param | µ_CD | IR |
|---|---|---|---|---|
| $n_{enc}=2, n_{dec}=2$ | 99.43 | 97.50 | **0.754** | **2.03** |
| $n_{enc}=4, n_{dec}=4$ | **99.51** | **97.78** | 0.762 | 3.32 |
| $n_{enc}=6, n_{dec}=6$ | 94.25 | 95.20 | 3.13 | 14.94 |

6 层直接崩溃——CAD 数据量不足以支撑过深 transformer。2 层 CD/IR 更好但 cmd/param 准确度差，最终选 4 层。

**Image encoder ablation**：

| encoder | R@10 | R@128 | R@1024 | R@2048 |
|---|---|---|---|---|
| ResNet-10 | 97.31 | 85.0 | 57.74 | 49.52 |
| ResNet-18 | **98.49** | **91.41** | **70.28** | **60.77** |
| ResNet-34 | 98.14 | 85.55 | 64.25 | 52.12 |
| ViT (6 layers, 16 heads, patch 32) | 96.62 | 85.23 | 62.43 | 53.87 |

ResNet-18 最佳，更大的 ResNet-34 反而过拟合，ViT 在数据量不够时不如 conv。

---

## 6. 与 image-to-mesh 模型的对比（Figure 14）

Paper 还跟现代 image-to-mesh 模型（PolyDiff, InstantMesh, Shap-E 等）做了定性对比。这些模型生成 mesh 但无法 edit，无法精确捕捉 mechanical feature（hole、slot、pocket）。GenCAD 输出的是 command sequence，可以直接 import 到 Onshape 编辑（Figure 13 展示了真实 Onshape 中的 feature tree）。

---

## 7. Limitations

1. CAD vocabulary 太小：只有 Line/Arc/Circle + Extrude。没有 revolve、fillet、chamfer、mirror、loft、sweep——这些才是工业 CAD 的核心。
2. 不保证 valid CAD：IR=3.32% 还是会失败，没有 geometry kernel 的 verifier in-the-loop（这点 Karpathy 你应该会想到 RL from verifier feedback 的思路）。
3. 输入图限制：isometric、noise-free。real-world photo 没测。
4. 没有跟最新 image-to-CAD 弱监督方法（DiffCAD, ROCA）做 retrieval 之外的任务比较。

---

## 8. 相关联想与延伸（build intuition）

### 8.1 架构哲学：DALL-E 1 / unCLIP bloodline
这个 pipeline 几乎是 DALL-E 1 (Ramesh et al., 2021) 的 CAD 版：
- DALL-E 1：dVAE tokenizer + CLIP text/image joint + diffusion prior + decoder
- GenCAD：CSR encoder/decoder + CCIP image/CAD joint + CDP diffusion prior + decoder

unCLIP (DALL-E 2) 进一步把这一思路推向 image generation。Karpathy 你应该很熟。这种 "learn joint multimodal latent + diffusion prior to bridge" 的设计在 2022-2024 已经成标配。CAD 是一个新的 high-value application domain。

### 8.2 跟 LLM code generation 的类比
CAD program 本质是 "代码"，geometry kernel 是 "解释器"。CSR = code autoencoder，CCIP = (image, code) contrastive，CDP = image-conditioned code prior，decoder = code completion。可以联想 code-Caption-Python 三段式。如果未来 CAD command vocabulary 扩到几十种 op（revolve, sweep, pattern, mirror...），就是真正的 CAD code LLM。

### 8.3 跟 LDM (Rombach 2022) 的关系
Stable Diffusion 也是 latent diffusion——但它的 latent 是 VAE 的 spatial latent map，U-Net 合适。GenCAD 的 latent 是 256-dim vector（average pooled），没有 spatial structure，所以用 ResNet-MLP 替代 U-Net——这是合理的 architecture matching。

### 8.4 跟 BrepGen / SkexGen 的差别
- SkexGen：autoregressive + disentangled codebook，关注 sketch + extrude 的 disentangled factorization
- BrepGen：在 B-rep 的 structured latent 上做 diffusion，直接出 B-rep，没有 design history
- GenCAD：image-conditioned，输出 command sequence，可 edit

GenCAD 跟 BrepGen 的 trade-off：BrepGen 在 MMD/JSD 上更好（保真），GenCAD 在 COV 上更好（多样性）+ 可 editability + 条件输入。这是设计哲学上的根本差异。

### 8.5 Verification feedback 的可能扩展
Paper 提到 limitation 是不保证 valid CAD。一个自然延伸是在 decoder 采样时引入 **geometry kernel feedback**，类似 LLM 中的 execution-guided decoding / RL from execution。可以做 rejection sampling、constraint decoding，或者直接 RL fine-tune 让 reward = valid CAD + chamfer distance + tool 路径合理性。

### 8.6 Real-world image 的挑战
Paper 只测了 isometric 渲染图。real-world photo-to-CAD 要解决：
- viewpoint estimation（从 single image 估 pose）
- background clutter
- occlusion
- multi-view 融合
这其实需要类似 NeRF / multi-view diffusion 的前置 stage 来 normalize 输入。可以联想 Shap-E, InstantMesh 这种 image-to-3D 前置 + GenCAD 后置 editability 的两阶段 pipeline。

### 8.7 Sketch token 与 sketch-based CAD 的经典线
CCIP 用 Canny sketch 当第二模态很聪明。这联系到 1990s 以来 Dori & Tombre, Nagasamy & Langrana 的 engineering drawing understanding 传统。现代 contrastive + diffusion 让这一老问题有了 data-driven 解法。

### 8.8 数据集规模的瓶颈
168k CAD sequence 在 LLM 时代算小数据。CSR 只有 6.72M 参数，是因为数据不够撑更大 model。如果 Onshape 公开更多 design history、或做 CAD synthesis augmentation（parametric perturbation 保持 topology），可以 scale 到 10M+ CAD，model 也能 scale 到 100M-1B，可能解锁 revolve/fillet/pattern 等更复杂 op 的生成。

### 8.9 与 Anthropic Constitutional AI / RLHF 的类比
未来 CAD 生成也可以走 "constitutional CAD" 路线：让 geometry kernel 当 critic，对生成的 CAD 打分（validity, manufacturability, editability），用 PPO/DPO fine-tune decoder。这是把 LLM alignment 那套搬到 CAD。

### 8.10 Potential multi-modal extension
既然 CCIP 学了 (image, CAD) joint，自然可以扩展到：
- (text description, CAD)：用 GPT 编码 text prompt
- (engineering drawing front-side-top view, CAD)：multi-view image encoder
- (point cloud scan, CAD)：reverse engineering 任务，3D scan → editable CAD

这是把 contrastive framework 推到 multi-modal CAD alignment 的入口。

---

## 9. 总结直觉

GenCAD 的核心贡献是**把 image-to-CAD 问题 properly 形式化**：CAD program as language + causal autoregressive latent + CLIP-style joint embedding + latent diffusion prior + frozen decoder。这五件套组合在 2024 年看是 well-established pattern，但应用在 CAD domain 是 first-of-its-kind。

最重要的实验数据点：
- Retrieval 60.77% @ batch 2048（vs image-to-image 3.91%，**15.6× 提升**）→ joint latent 学得非常对齐
- Conditional COV 82.59 vs unconditional 78.27 → conditional diversity 反而更高，因为 image 既约束又 hint
- FID 排序 diffusion < deterministic → multimodal prior 必要

未来要 watch 的方向：
1. 扩展 CAD vocabulary 到几十种 op
2. Real-world image robustness
3. Geometry kernel verifier in-the-loop
4. Scale up data and model
5. Multi-view + text 多模态

这 paper 在我看来是把 generative CAD 从 "unconditional toy" 推进到 "conditional useful" 的关键一步，跟 DALL-E 1 在 image generation 历史中的位置类似——架构成熟度未必最高，但 problem framing 和 baseline 建立意义重大。
