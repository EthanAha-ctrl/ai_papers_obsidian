---
source_pdf: TEXGen.pdf
paper_sha256: 343e34cea5fe0ebb0ecc94069db673b0db597d0bdd2ce7df69fdf68f59237f3d
processed_at: '2026-08-12T13:41:56-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TEXGen 用人话说

## 一句话概括

给一个3D裸模，告诉它"我要一个红色的青蛙"，10秒钟直接吐出一张完整的、多面连贯的、1024×1024的高分辨率贴图。 不用反复优化，不用拼拼凑凑，一次前向推理搞定。

## 它到底在解决什么痛点？

### 现在主流做法有多蠢

目前给3D模型画贴图，SOTA方法大概分两类：

**第一类：SDS类（Score Distillation Sampling）**
思路是"借用"2D diffusion model（比如Stable Diffusion）的prior，在3D表面上做test-time optimization。 每次迭代从某个角度看3D模型，让2D diffusion model告诉你"这个角度应该长什么样"，然后用梯度反传去调整3D纹理。 反复迭代几百到几千次。
- 痛点：**慢到令人发指**，单个模型可能要优化几十分钟
- 痛点：**Janus problem** —— 从前面看生成了眼睛，转到后面，2D model不知道"后面不该有眼睛"，又给你画了一双眼睛
- 痛点：**色彩不自然**，因为SDS的梯度场本身有偏置，出来的纹理总有种"floaty"的感觉

**第二类：Multi-view Inpainting（如TEXTure、Text2Tex）**
思路是"看一面画一面"。先从某个角度生成一张图，project到mesh上，然后转一个角度，用inpainting model把没画的部分补上，再project，再补，循环往复。
- 痛点：**全局不一致** —— 每一步只看到局部信息，没有全局的coherence，不同角度画的纹理风格可能打架
- 痛点：**还是慢** —— 每个模型都要跑这个pipeline
- 痛点：**Janus problem依然存在** —— 因为底层还是2D model，没有真正的3D awareness

### TEXGen 的思路完全不同

**直接训练一个3D-native的diffusion model，在UV texture space里学习纹理的分布。** 一次feed-forward，端到端出结果。

这就好比说：与其每次都借用别人的大脑（2D diffusion model）来费力地逆向工程3D纹理，不如直接训练一个专门理解3D纹理的大脑。

## UV Space 的核心矛盾

### 什么是 UV Map？

3D mesh的表面是一个2D manifold。 要给它画贴图，最标准的方法是UV mapping：把3D表面"剥皮展开"到一张2D图上，这张图就是UV texture map。 类似把地球仪的表面剥下来摊成一张世界地图。

### UV Map 天生有个致命问题：Fragmentation

剥皮的时候必须切割，一个完整的3D表面被切成多个"islands"（碎片），散落在UV map的不同位置。

这就导致一个严重的拓扑错位：

- **3D上相邻的，UV上可能相隔万里**：比如青蛙的肚子和腿在3D上挨着，但UV展开后可能一个在左上角，一个在右下角
- **UV上相邻的，3D上可能毫无关系**：两个island恰好被展开到相邻位置，但它们在3D上根本不挨着

**纯2D CNN的灾难**：标准2D convolution只能看到UV space上的局部邻域。 它完全不知道"左上角的肚子"和"右下角的腿"在3D上是一体的。 结果就是：每个island内部纹理画得还行，但islands之间风格打架、颜色不协调、seam（接缝）处出现明显的artifacts。

**纯3D Point Cloud的灾难**：在3D surface上直接做point cloud attention能解决全局一致性，但point cloud太稀疏了，无法承载高分辨率纹理细节。 而且3D KNN/attention的计算复杂度远高于2D conv，scaling up很困难。

### TEXGen 的解法：Hybrid 2D-3D Block

核心intuition：**2D space负责高频细节，3D space负责全局一致性，两者交替工作。**

每个block的工作流：

```
输入 UV feature
    ↓
[2D Conv] ← 在UV map上做卷积，提取局部高频细节，非常高效
    ↓
[Rasterization] ← 把UV feature "贴回" 3D surface，变成3D point feature
    ↓
[3D Point Block] ← 在3D space做attention，学习全局3D关系
    ↓
[Scatter back] ← 把3D feature "贴回" UV space
    ↓
[Fusion] ← 2D feature + 3D feature 融合
    ↓
输出 UV feature
```

这样2D conv负责"画好每个island的细节"，3D attention负责"确保所有islands风格统一、seam处连贯"。 各司其职，互不干扰。

## 3D Point Block 里的工程细节

3D point cloud上直接做global attention，token数量爆炸。 TEXGen用了三招解决scalability：

### 1. Grid Pooling 稀疏化

把dense point cloud划分到3D voxel grid里，每个grid内的points pool成一个token。 假设原本100万个points，grid pooling后可能只有几千个tokens。 大幅降低attention的token数量。

### 2. Serialized Attention

Point cloud是无序的，没有"局部window"的自然定义。 怎么做local attention？

用space-filling curve（空间填充曲线），比如Hilbert curve或Z-order curve。 这些曲线的神奇之处在于：**3D空间中物理位置接近的点，在1D序列中也彼此接近。**

所以做法是：
1. 用Hilbert curve把所有sparse tokens序列化成1D序列
2. 把序列切成固定大小的patches
3. 在每个patch内做self-attention

这样既保留了空间局部性，又把attention限制在局部patch内，计算可控。

### 3. sCPE (sparse Conditional Positional Encoding)

Attention本身是permutation-invariant的，需要positional encoding注入位置信息。

原始的xCPE直接在attention前做sparse convolution，但当transformer维度$d=2048$时，sparse conv的内存和计算开销巨大。

sCPE的trick：
1. 先用一个linear layer把channel从2048降到很小（比如64）
2. 在低维做sparse conv提取位置信息
3. 再用linear layer升回2048

**降维→卷积→升维**，经典bottleneck trick，大幅省时间。

## 训练的几个关键技术选择

### v-prediction + Zero-terminal SNR

传统diffusion用$\epsilon$-prediction：网络预测噪声。 但当noise schedule的终端（$t=1000$）不是纯噪声时，训练和推理之间存在gap —— 训练时$x_{1000}$还残留一点$x_0$的信号，但推理时你从纯噪声$\mathcal{N}(0,I)$开始。

Zero-terminal SNR强制$\bar{\alpha}_{1000}=0$，保证$x_{1000}=\epsilon$，消除这个gap。

在zero-terminal SNR下，v-prediction比$\epsilon$-prediction数值更稳定：
$$v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$$

网络预测$v_t$，而不是$\epsilon$或$x_0$。

### Dual Loss：Diffusion + Rendering

光在UV space做diffusion loss不够，因为UV space和实际渲染外观之间有gap（UV的seam在渲染时可能不明显，但某些UV artifacts在渲染时才暴露）。

所以额外加了一个rendering loss：
1. 从网络输出反推预测的$\hat{x}_0$
2. 从随机视角渲染$\hat{x}_0$得到$\hat{I}_i$
3. 和ground-truth rendering $I_i$算LPIPS

最终loss：$\mathcal{L} = \mathcal{L}_{\text{diff}} + 0.5 \cdot \mathcal{L}_{\text{render}}$

### CFG Scale 只需要 2-3

图像diffusion通常用CFG scale $w=7.5$。 但TEXGen实验发现最优值在$w=2.0\sim3.0$。

intuition：TEXGen的condition包含了精确的3D geometry + projected image pixels，这个condition比纯text prompt强烈得多。 过高的CFG会导致over-saturation和纹理pattern退化。

## 数据和规模

- 原始数据：Objaverse的800K meshes
- 清洗后：120,000 meshes for training，400 for evaluation
- 清洗过程：用xAtlas重新展开UV，确保每个mesh只有单一UV atlas；用Gemini生成caption
- 模型规模：700M参数
- 训练资源：32块A100，batch size 64，400K iterations

## 结果有多炸裂

| Method | FID↓ | KID↓ | Time |
|--------|------|------|------|
| TEXTure | 48.31 | 48.00 | 80s |
| Text2Tex | 49.85 | 47.38 | 344s |
| Paint3D | 43.55 | 25.73 | 95s |
| **TEXGen** | **34.53** | **11.94** | **10s** |

FID/KID全面碾压，速度快一个数量级。 而且因为是在3D数据上训练的feed-forward model，**天然避免Janus problem**。

## Ablation 证明 Hybrid Design 的必要性

| Model | FID↓ | KID↓ | 现象 |
|-------|------|------|------|
| A: Full Hybrid | 69.74 | 17.89 | 正常 |
| B: w/o 3D Point Block | 72.58 | 25.52 | islands之间风格打架，seam处有artifacts |
| C: w/o 2D UV Block | 94.22 | 159.94 | 崩溃，纹理模糊，细节全无 |

Model C的FID直接爆炸到94，说明**高分辨率细节完全靠2D conv撑着**，3D attention只能做全局reasoning，无法替代2D conv的细节提取能力。

## Zero-shot 泛化能力

因为训练时是用"partial texture（从单视图project过来的）+ 条件"去denoise出"full texture"，模型天然学会了"补全缺失纹理"的能力。

所以推理时可以直接做：

1. **Text-to-Texture**：只有文字？随便渲染一个depth map，用ControlNet生成一张单视图图，再喂给TEXGen
2. **Texture Inpainting**：用户给一张被mask掉一部分的贴图，直接填空（image embedding设为zero，因为训练时有20%概率drop image condition，模型对此鲁棒）
3. **Sparse-view Completion**：用户给2-3张图，都project到UV space融合，模型补全被遮挡区域

这些应用**完全不需要fine-tuning**，直接zero-shot就能用。

## 我的个人思考

这篇paper最让我impressed的是它的problem formulation。 它没有去改进SDS，没有去优化multi-view inpainting pipeline，而是直接问了一个更fundamental的问题：**为什么不直接在UV space训练一个large diffusion model？**

这个问题的答案在于engineering：UV space的fragmentation问题使得纯2D架构不可行，而纯3D架构又撑不住高分辨率。 Hybrid 2D-3D Block的设计精妙地破解了这个dilemma。

从scaling的角度看，这篇工作打开了一扇门。 700M参数已经不小了，但如果给更多数据、更大模型，feed-forward 3D texture generation的上限可能远不止于此。 而且，如果能扩展到PBR material maps（roughness、metallic、normal等），价值会更大。

参考链接：
- Project Page: https://cvmi-lab.github.io/TEXGen/
- Code: https://github.com/CVMI-Lab/TEXGen
- Paper: https://doi.org/10.1145/3687909

---

Hi Andrej, 很高兴能和你深入探讨这篇 paper。 TEXGen 这篇工作非常 impressive，它从 representation 和 architecture 的根本上重新思考了 3D texture generation 的 scaling 问题。目前的 3D texturing 领域，主流范式是利用预训练的 2D diffusion model 进行 test-time optimization（比如 SDS 系列），或者通过 multi-view inpainting 拼接纹理。这些方法虽然能利用 2D diffusion 的强大 prior，但不可避免地面临 3D inconsistency（Janus problem）、耗时过长以及 lighting artifacts 等问题。 TEXGen 摒弃了这种 test-time optimization 的路径，转而选择直接在 UV texture space 中训练一个 large feed-forward diffusion model，这是一个非常 bold 且 fundamental 的转变。

下面我从 intuition、 architecture、 formula 以及 experimental data 的角度为你详细拆解。

### 1. Representation 的核心 Conflict 与 Intuition

3D mesh surface 本质上是一个嵌入在 3D space 中的 2D manifold。为了能够用 2D CNN 等标准架构处理，图形学中常用 UV mapping 将 3D surface “展开”到 2D UV space 上。

这就引出了一个核心 conflict：
*   **2D UV Space 的优势**：它将 3D surface 信号规整化为了 dense 2D grid，非常适合 2D convolution 提取 high-frequency details，并且可以直接用 ground-truth texture map 做 end-to-end 的 supervision，这与 diffusion model 的训练范式完美兼容。
*   **2D UV Space 的劣势**：UV unwrapping 必然带来 fragmentation。3D surface 上原本连续的 regions 被切割成多个 islands。在 3D space 中 geodesic distance 很近的 points，在 UV map 上可能距离很远；反之，UV map 上相邻的 pixels，在 3D surface 上可能毫无关联。纯 2D CNN 无法跨越这些 seams 感知 global 3D structure，这就是导致 texture style 不一致和 seam artifacts 的根源。

**TEXGen 的 Intuition**：我们需要 2D space 的高分辨率局部特征提取能力，同时也需要 3D space 的 global structural reasoning 能力。因此，论文提出了 **Hybrid 2D-3D Block**，在 network 的每一个 stage 中，让 features 在 2D UV space 和 3D point cloud space 之间来回穿梭、交互。

### 2. Architecture 深度解析

整个 denoising network 基于 UNet 框架，包含 5 个 stages（4次下采样，4次上采样）。核心在于其 Hybrid 2D-3D Block 的设计。

#### 2.1 Hybrid 2D-3D Block 工作流
输入 feature $f_{\text{in}}$ 首先进入 UV Head：
1.  **2D Conv (UV Space)**: 在 UV map 上做标准 2D convolution。这一步极其高效，且能聚合 surface neighborhood 的 features，保留 high-resolution details。输出 $f_{\text{out}}^{\text{uv}}$。
2.  **Rasterization Remap**: 通过 rasterization，将 2D UV features $f_{\text{out}}^{\text{uv}}$ 重新映射回 3D space，赋值给 mesh surface 上的 points，得到 dense point features $f_{\text{in}}^{\text{point}}$。
3.  **3D Point Block (3D Space)**: 在 3D space 中进行 global reasoning。为了控制计算复杂度，这里用了 sparse features 和 serialized attention。输出 $f_{\text{out}}^{\text{point}}$。
4.  **Fusion**: 将 3D features scatter 回 UV space，并与之前的 UV features 融合。论文引入了一个 learned gated scale $\alpha^{\text{point}}$（由 condition embedding 预测得到）来控制 3D features 的注入比例：
    $$f_{\text{out}} = f_{\text{out}}^{\text{uv}} + \alpha^{\text{point}} \cdot f_{\text{out}}^{\text{point}}$$

#### 2.2 3D Point Block 内部的 Engineering 细节
在 3D space 中直接做 global attention 复杂度极高。TEXGen 采用了几项关键优化：

*   **Grid-Pooling**: 将 dense points 划分到 3D voxel grids 中，每个 grid pool 成一个 token，极大减少了 token 数量。
*   **Serialized Attention**: Point cloud 是无序的，直接做 local window attention 难以定义 window。论文利用 space-filling curves（如 Hilbert curve 或 Z-order curve）对 sparsified points 进行序列化。这种序列化保证了在 3D space 中物理位置接近的 points，在 1D 序列中也彼此接近。随后将序列切分成 patches 进行 self-attention 计算，兼顾了局部性和计算效率。
*   **sCPE (sparse Conditional Positional Encoding)**: 绝对坐标的 positional encoding 效果不如 conditional positional encoding。原始的 xCPE 直接在 attention 前做 sparse convolution，但当 transformer dimension $d=2048$ 时计算非常耗时。 TEXGen 提出了 sCPE：先用一个 linear layer 将 channel 维度压缩，做 sparse convolution 提取位置信息，再用 linear layer 升维回去。这是一个非常实用的 efficiency trick。

#### 2.3 Condition Modulation
网络接收 text prompt $T$、单视图 image $I$ 和 timestep $t$。 Text 和 image 分别通过 text encoder 和 CLIP image encoder 提取 global embeddings，结合 timestep embedding 生成 global condition embedding $c$。
类似 DiT 的设计，通过 MLPs 从 $c$ 学习 modulation vectors $\gamma$ 和 $\beta$，以及 gated scale $\alpha$：
$$f_{\text{mod}} = (1 + \gamma) \cdot f_{\text{in}} + \beta$$
$$f_{\text{fuse}} = \alpha \cdot f_{\text{out}} + f_{\text{skip}}$$
这种 AdaLN-style 的 condition injection 在 large scale diffusion model 中被证明非常有效。

### 3. Diffusion Loss 与 Formula 推导

TEXGen 采用了 v-prediction 范式，并结合了 zero-terminal SNR 和 multi-view rendering supervision。

**前向加噪过程** (公式 1)：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
*   $x_t$: timestep $t$ 时的 noised texture map。
*   $x_0$: 干净的 ground-truth texture map。
*   $\bar{\alpha}_t$: noise scheduler 的累积超参数。论文使用了 zero-terminal SNR 技术，使得当 $t=1000$ 时，$\bar{\alpha}_t = 0$。这意味着 $x_{1000} = \epsilon$，即纯高斯噪声，消除了训练和推理（从纯噪声开始）之间的 distribution gap。
*   $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声。

**v-prediction 与 Diffusion Loss** (公式 2)：
$$v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} x_0$$
$$\mathcal{L}_{\text{diff}} = \lambda_t || v_t - x_{\text{out}} ||_2^2$$
*   $v_t$: velocity target。网络预测的不是 noise $\epsilon$，而是 $v_t$。在 zero-terminal SNR 设置下，v-prediction 比预测 $\epsilon$ 或 $x_0$ 数值上更稳定。
*   $x_{\text{out}}$: 网络的输出。
*   $\lambda_t$: soft-min-SNR weight。它根据不同 timestep 的 Signal-to-Noise Ratio 动态调整 loss 权重，防止高 noise timestep 梯度被淹没。

**Rendering Loss** (公式 3 & 4)：
$$\mathcal{L}_{\text{render}} = \frac{1}{N} \sum \text{LPIPS}(\hat{I}_i, I_i)$$
$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{diff}} + \lambda_2 \mathcal{L}_{\text{render}}$$
*   $\hat{I}_i$: 从随机视角使用预测的 $\hat{x}_0$ 渲染出的 image。
*   $I_i$: 对应视角的 ground-truth image。
*   论文设置 $\lambda_1 = 1.0, \lambda_2 = 0.5$。这个 loss 提供了 rendering domain 的 perceptual supervision，弥合了 UV map space 和实际 visual appearance 之间的 gap。

### 4. Experiments 与 Data 分析

论文基于 Objaverse 数据集，清洗后保留了 120,000 个 meshes for training，400 for evaluation。模型参数量约 700M，在 32 A100 GPUs 上训练 400K iterations，batch size 64。

**Quantitative Results (Table 1)**:
TEXGen 在 FID (34.53) 和 KID (11.94) 上大幅超越了 TEXTure, Text2Tex 和 Paint3D。最关键的是 inference time：仅需 10 秒（30步 DDIM），而对比方法需要 80s 到 344s 的 test-time optimization。这证明了 feed-forward model 在 efficiency 上的绝对优势。

**Ablation: Hybrid Block (Table 3)**:
*   **Model A (Full Hybrid)**: FID 69.74, KID 17.89。
*   **Model B (w/o Point Block)**: FID 72.58, KID 25.52。没有了 3D interaction，texture 风格一致性下降，出现 seam artifacts。
*   **Model C (w/o UV Block)**: FID 94.22, KID 159.94。崩溃性下降。纯 3D point features 无法维持 high-resolution 2D texture 的细节，生成结果模糊。这证明了 high-resolution 2D convolution 是细节的来源，3D attention 只能做 global reasoning。

**Ablation: CFG Scale (Table 4)**:
传统 image diffusion 通常用 CFG scale $w=7.5$。但 TEXGen 的最优 CFG scale 在 $w=2.0 \sim 3.0$ 之间。 Intuition 在于：TEXGen 的 condition 包含了精确的 3D geometry 和 projected image pixels，这些 condition 本身已经非常强烈且信息丰富。过高的 CFG scale 会导致 over-saturation 和 texture pattern 的退化。

### 5. Zero-shot Applications

由于模型是在 partial texture (projected from single view) 和 full texture 之间做 conditional denoising，它天然具备了 inpainting 的 zero-shot 泛化能力。
*   **Text-to-Texture**: 随机渲染一个 depth map，用 ControlNet 生成 single view image，再输入给 TEXGen。
*   **Texture Inpainting**: 输入 masked partial texture，将 image embedding 设为 zero（训练时采用了 20% probability 的 conditional dropout，使模型鲁棒）。
*   **Sparse-view Completion**: 将多个 views 的 pixels 都 project 到 UV space 并融合，随机选一个 view 提取 image embedding。

### 总结与个人 Insight

TEXGen 展示了直接在 UV space 学习 texture distribution 的巨大潜力。它的核心贡献在于识别出了 UV representation 的内在矛盾，并用一个极其 elegant 的 Hybrid 2D-3D architecture 解决了它。通过将 2D CNN 的 local efficiency 与 3D attention 的 global reasoning 结合，它成功训练了一个 700M 参数的 feed-forward diffusion model，避免了 test-time optimization 的种种弊端。

从 future work 的角度看，目前 TEXGen 的 condition image 需要 pose-aligned 和 shape-aligned，限制了 “texture transfer” 的应用。如果能在 network 中引入 cross-attention 机制处理 arbitrary image features，或者扩展到 PBR material maps 的生成，将会有更大的想象力空间。

**Reference Links:**
*   Project Page: [https://cvmi-lab.github.io/TEXGen/](https://cvmi-lab.github.io/TEXGen/)
*   Code Repository: [https://github.com/CVMI-Lab/TEXGen](https://github.com/CVMI-Lab/TEXGen)
*   Objaverse Dataset: [https://objaverse.allenai.org/](https://objaverse.allenai.org/)
*   Related Paper (Paint3D): [https://arxiv.org/abs/2312.13913](https://arxiv.org/abs/2312.13913)
*   Related Paper (TEXTure): [https://research.nvidia.com/labs/toronto-ai/TEXTure/](https://research.nvidia.com/labs/toronto-ai/TEXTure/)
