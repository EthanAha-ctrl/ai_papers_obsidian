









好的，我已经收集了足够的信息。下面是对这个 project 的详细技术解析。

---

# `haofanwang/inswapper` — One-click Face Swapper and Restoration

📎 **项目地址**: https://github.com/haofanwang/inswapper

---

## 1. 项目概述

这个 project 是由 **Haofan Wang** 开发的一个 **one-click face swapping（一键换脸）和 face restoration（面部修复）** 的 pipeline。它高度依赖于 **InsightFace** 开源库，核心使用的是 `inswapper_128.onnx` 这个预训练的 face swap model，并且结合 **CodeFormer** 或 **GFPGAN** 进行后处理的 face restoration，从而获得高质量的换脸结果。

简单来说，这个 project 做了三件事：
1. **Face Detection & Alignment**（人脸检测与对齐）
2. **Face Swapping**（人脸身份替换）
3. **Face Restoration**（人脸超分辨率修复）

---

## 2. 完整 Pipeline 架构

```
Input Image (Target)  +  Source Face Image
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌─────────────────┐
│  Face Detection  │   │  Face Detection  │
│  (RetinaFace /   │   │  (RetinaFace /   │
│   SCRFD)         │   │   SCRFD)         │
└────────┬────────┘   └────────┬────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌─────────────────────┐
│  Face Alignment  │   │  Identity Embedding  │
│  (5-point        │   │  Extraction          │
│   Landmark)      │   │  (ArcFace R100)      │
└────────┬────────┘   └────────┬─────────────┘
         │                      │
         ▼                      ▼
┌──────────────────────────────────────────┐
│         INSwapper Model (ONNX)           │
│    inswapper_128.onnx                    │
│  ┌────────────┐   ┌──────────────────┐   │
│  │ Target Face│ + │ Source Identity   │   │
│  │ (128×128)  │   │ Embedding (512-d)│   │
│  └────────────┘   └──────────────────┘   │
│              ↓                            │
│      Swapped Face (128×128)              │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│    Face Restoration (Post-processing)    │
│    CodeFormer / GFPGAN v1.4              │
│    (Super-Resolution + Quality Enhance)  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│   Paste Back to Original Image           │
│   (Inverse Affine Transform + Blending)  │
└──────────────────────────────────────────┘
                   │
                   ▼
            Output Image
```

---

## 3. 核心组件的技术细节

### 3.1 Face Detection: SCRFD / RetinaFace

InsightFace 使用 **SCRFD (Sample and Computation Redistribution for Efficient Face Detection)** 作为 face detector。

- **Input**: 原始图像
- **Output**: Bounding boxes + 5-point facial landmarks（两个眼角、鼻尖、两个嘴角）
- SCRFD 是一种 anchor-free 的 face detection network，比 RetinaFace 更快更轻量

5-point landmarks 的作用是进行 **similarity transform（相似变换）**，将检测到的人脸 warp 到一个标准的 **112×112** 或 **128×128** 的 aligned face template 上。

变换矩阵 $M$ 的计算：

$$M = \arg\min_{M} \sum_{i=1}^{5} \|M \cdot p_i^{src} - p_i^{dst}\|^2$$

其中：
- $p_i^{src}$ 是检测到的第 $i$ 个 landmark 的坐标
- $p_i^{dst}$ 是标准 template 中第 $i$ 个 landmark 的目标坐标
- $M$ 是 2×3 的 affine transformation matrix

### 3.2 Identity Embedding Extraction: ArcFace

Source face 的 identity 信息是通过 **ArcFace** (Additive Angular Margin Loss) 模型提取的，通常使用 **ResNet-100** 作为 backbone。

**ArcFace Loss 公式**:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j=1, j \neq y_i}^{n} e^{s \cdot \cos(\theta_j)}}$$

其中：
- $N$ = batch size
- $s$ = feature scale（通常 = 64），用于放大 cosine similarity 的值使 softmax 更 sharp
- $\theta_{y_i}$ = 第 $i$ 个样本的 feature vector 与其 ground-truth class center 之间的夹角
- $m$ = additive angular margin（通常 = 0.5），人为加大同类样本到 decision boundary 的角距离，使学习更 discriminative
- $n$ = 总 class 数量（训练时的 identity 数量）
- $\cos(\theta_j)$ = feature 与第 $j$ 个 class center 的 cosine similarity

**提取过程**：
1. Source face → aligned to 112×112
2. 通过 ArcFace ResNet-100 → 得到 **512-dimensional identity embedding vector** $\mathbf{z}_{id} \in \mathbb{R}^{512}$
3. 这个 $\mathbf{z}_{id}$ 编码了 source 的 **identity 信息**（面部骨骼结构、五官比例等），但**不包含** expression、pose、lighting 等 attribute 信息

### 3.3 INSwapper 核心模型

`inswapper_128.onnx` 是 InsightFace 团队训练的核心换脸模型。虽然 InsightFace 团队没有公开完整的训练代码和论文细节，但从代码和相关讨论可以推断其架构类似于 **SimSwap** / **FaceShifter** 系列：

**核心思想**: **Identity-Attribute Disentanglement（身份-属性解耦）**

模型的目标是：
- 从 **source face** 提取 identity information
- 从 **target face** 保留 attribute information（expression、pose、lighting、background）
- 将两者融合生成 swapped face

**推测的架构**:

```
Source Identity Embedding (512-d)
         │
         ▼
┌─────────────────────────┐
│  Identity Injection      │
│  (AdaIN / Feature       │
│   Modulation)           │
│         ↑               │
│    ┌────┴────┐          │
│    │ Target  │          │
│    │ Face    │          │
│    │ Encoder │          │
│    │ (128×128)│         │
│    └─────────┘          │
│         │               │
│         ▼               │
│    Decoder              │
│    (128×128 output)     │
└─────────────────────────┘
```

**关键技术 — Adaptive Instance Normalization (AdaIN)**:

$$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$

其中：
- $x$ = target face 在 encoder 某层的 feature map
- $y$ = 由 source identity embedding 经过 MLP 映射得到的 style parameters
- $\mu(x)$, $\sigma(x)$ = $x$ 的 channel-wise mean 和 standard deviation
- $\mu(y)$, $\sigma(y)$ = 从 identity embedding 计算得到的 target style statistics

这样，**target face 的 spatial structure（attribute）被保留在 feature map 的空间布局中**，而 **source 的 identity 信息通过 normalization statistics 注入进来**。

**训练 Loss（推测）**:

$$L_{total} = \lambda_{id} L_{id} + \lambda_{rec} L_{rec} + \lambda_{attr} L_{attr} + \lambda_{adv} L_{adv}$$

其中：
- $L_{id}$ = **Identity Loss**: 使用预训练 ArcFace 计算 swapped face 和 source face 的 identity embedding 之间的 cosine distance
  $$L_{id} = 1 - \cos(\mathbf{z}_{id}^{swap}, \mathbf{z}_{id}^{src})$$
- $L_{rec}$ = **Reconstruction Loss**: 当 source = target 时，输出应等于输入（self-reconstruction）
  $$L_{rec} = \|I_{swap} - I_{target}\|_1$$
- $L_{attr}$ = **Attribute Loss**: 保留 target 的 expression、pose 等，通常通过 landmark loss 或 perceptual loss 实现
- $L_{adv}$ = **Adversarial Loss**: GAN discriminator loss，确保生成结果的真实感
- $\lambda_{id}, \lambda_{rec}, \lambda_{attr}, \lambda_{adv}$ 是各个 loss 的权重超参数

### 3.4 Face Restoration: CodeFormer / GFPGAN

由于 `inswapper_128.onnx` 输出的分辨率只有 **128×128**，直接 paste 回原图会有明显的模糊和 artifact。所以需要 face restoration model 进行增强。

#### GFPGAN v1.4

**GFPGAN (Generative Facial Prior GAN)** 利用了预训练的 **StyleGAN2** 作为 facial prior：

$$I_{restored} = G(F_{enc}(I_{degraded}), \{W_i^+\})$$

其中：
- $I_{degraded}$ = 降质的 swapped face
- $F_{enc}$ = degradation removal encoder
- $\{W_i^+\}$ = StyleGAN2 的 multi-resolution style codes
- $G$ = 基于 StyleGAN2 的 decoder

GFPGAN 的关键创新是 **Channel Split Spatial Feature Transform (CS-SFT)**：将 encoder 的 feature 和 StyleGAN2 的 feature 在各个 resolution layer 进行融合。

#### CodeFormer

**CodeFormer** 使用了 **Transformer + Codebook（VQ-VAE）** 的架构：

1. 将 degraded face 编码到 discrete codebook space
2. 用 Transformer 预测最优的 code sequence
3. 从 codebook 解码出 high-quality face

$$\hat{c} = \text{Transformer}(E(I_{degraded}))$$
$$I_{restored} = D(\hat{c})$$

其中：
- $E$ = Encoder
- $\hat{c}$ = 预测的 discrete code sequence
- $D$ = Decoder（从 codebook 重建）

CodeFormer 有一个 **fidelity weight** $w \in [0, 1]$：
- $w = 0$: 更偏向质量（更好看但可能偏离原始 identity）
- $w = 1$: 更偏向保真度（更像原始 swap 结果但质量可能较低）

### 3.5 Paste Back（回贴）

最后一步是将 restored face 贴回原图：

1. 使用之前 face alignment 时计算的 **inverse affine matrix** $M^{-1}$，将 restored face warp 回原图坐标系
2. 使用 **seamless cloning（泊松融合）** 或 **feather blending** 使边界过渡自然：

$$I_{output}(x, y) = \alpha(x, y) \cdot I_{swap}(x, y) + (1 - \alpha(x, y)) \cdot I_{target}(x, y)$$

其中 $\alpha$ 是一个 soft mask（通常由 convex hull of landmarks 生成，边缘 Gaussian blur 后得到）。

---

## 4. 代码调用方式

```python
import insightface
from insightface.app import FaceAnalysis

# 初始化 Face Analysis（包含 detection + recognition）
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载 INSwapper model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', ...)

# 检测 source 和 target 的人脸
source_faces = app.get(source_img)
target_faces = app.get(target_img)

# 执行 swap
result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)
```

---

## 5. 与相关工作的对比

| 方法 | 是否需要 subject-specific training | 分辨率 | Identity Preservation | Attribute Preservation |
|------|------|------|------|------|
| **DeepFakes (Autoencoder)** | 是（每对人需单独训练） | 64-256 | 中等 | 中等 |
| **SimSwap** | 否 (one-shot) | 224×224 | 高 | 高 |
| **FaceShifter** | 否 (one-shot) | 256×256 | 很高 | 很高 |
| **INSwapper (InsightFace)** | 否 (one-shot) | 128×128 (raw) → 高分辨率 (+ restoration) | 很高 | 很高 |
| **Roop / FaceFusion** | 否（调用 inswapper） | 同上 | 同上 | 同上 |

---

## 6. 第一性原理的直觉

### 为什么 face swap 可以做到 one-shot？

**核心 insight**: 人脸可以被分解为两个近似正交的子空间：

$$\mathbf{Face} = f(\mathbf{Identity}, \mathbf{Attributes})$$

其中：
- **Identity** ≈ 面部骨骼结构、五官形状比例、肤色 → 由 ArcFace embedding 捕获
- **Attributes** ≈ expression、pose、lighting、occlusion → 由 target face 的 spatial feature 捕获

只要有一个足够好的 **identity encoder**（ArcFace 在几百万张人脸上训练过，已经学到了非常 discriminative 的 identity representation），就可以用 **一张** source 图提取 identity，然后注入到 target 的 attribute space 中。

### 为什么需要 face restoration？

INSwapper 的 **bottleneck resolution 是 128×128**，这是 identity-attribute disentanglement 效果和计算效率之间的 trade-off。但 128×128 在高清场景中不够用。

**Face restoration 的本质是一个 conditional image super-resolution problem**：给定一张 low-quality face，利用 facial structure prior（如 StyleGAN2 学到的人脸分布或 VQ-VAE codebook 中存储的 high-quality face patches）来「幻觉」出缺失的高频细节（毛孔、睫毛、皮肤纹理等）。

---

## 7. 相关链接

- **项目主页**: https://github.com/haofanwang/inswapper
- **InsightFace 官方**: https://github.com/deepinsight/insightface
- **InsightFace INSwapper 示例**: https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md
- **ArcFace 论文**: https://arxiv.org/abs/1801.07698
- **GFPGAN**: https://github.com/TencentARC/GFPGAN
- **CodeFormer**: https://github.com/sczhou/CodeFormer
- **SimSwap 论文** (类似架构参考): https://arxiv.org/abs/2106.06340
- **FaceShifter 论文** (类似架构参考): https://arxiv.org/abs/1911.09073
- **SCRFD Face Detection**: https://arxiv.org/abs/2105.04714
- **InsightFace Blog**: https://www.insightface.ai/blog/the-evolution-of-neural-network-face-swapping-from-deepfakes-to-one-shot-innovation-with-insightface

---

**总结**: `haofanwang/inswapper` 本质上是一个将 InsightFace 的 face analysis + INSwapper face swap + CodeFormer/GFPGAN face restoration 三大模块 **打包成一键式使用** 的 wrapper project。它的核心技术贡献在于 **pipeline 的工程化整合**，而底层的 face swap model 和 face restoration model 来自 InsightFace 和 Tencent ARC 团队的研究成果。