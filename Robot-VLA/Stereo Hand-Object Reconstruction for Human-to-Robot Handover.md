---
source_pdf: Stereo Hand-Object Reconstruction for Human-to-Robot Handover.pdf
paper_sha256: bcfcacf27623ddc986c43d502cfaa774fdde63b601bc8df67fd896feae4c7b37
processed_at: '2026-08-12T11:09:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇Paper

## 一句话概括

让robot接住人递过来的东西，光看两台普通摄像头（不用depth sensor），就能同时猜出手和物体的3D shape，哪怕物体是透明玻璃杯也能搞定。

## 问题的痛点在哪

**Depth sensor不好用**：你拿Intel RealSense对着透明玻璃杯看，返回的depth map全是洞——红外光直接穿过去或者折射走了，sensor根本不知道那里有东西。这在handover场景里是致命的，robot看不见要抓什么。

**Stereo RGB方法假设太强**：之前Queen Mary自己做的CORSMAL baseline说"我只处理杯子，杯子是旋转对称的，人递过来必须竖着拿"。你递个螺丝刀、递个CD盒、递个横着拿的杯子，直接崩。

**Single-view reconstruction缺一半**：手挡住物体一半，camera只看见一面，另一面靠prior猜。猜得好不好全看运气和training data覆盖度。

## 核心idea（这里开始讲人话）

作者的insight其实特别优雅：**与其让一个view硬猜看不见的部分，不如让两个view各自说"我觉得这个voxel可能是这512种shape primitive里的哪一个，概率分别是多少"，然后把两个概率直接相乘，谁都不确定的部分就保持uncertain，两个都确定的地方就boost起来。**

这个idea的妙处在于：probability distribution天然处理了occlusion。如果左眼看不见某个voxel（被手挡了），左眼的prediction会接近uniform distribution（均匀分布，表示"我啥也不知道"），乘以右眼的confident prediction后，结果还是右眼的prediction——occlusion被自然handle了。

如果用deterministic regression（直接预测一个值），就没这个好处。你不知道这个view是真的不确定，还是在瞎猜。Probability distribution把这个uncertainty显式表达出来了。

## 技术细节讲人话

### VQ-VAE学shape codebook是在干嘛

想象你是一个3D artist，手头有几千个hand mesh和object mesh。你想给每个mesh打一个标签，标签来自一个512条的"shape primitive字典"。字典里可能是"圆柱体"、"球体"、"手掌握拳"、"手指弯曲"这些basic shape。

VQ-VAE就是这个自动建字典的过程：
1. Encoder把一个mesh压成一个128维vector
2. 在512个codebook entry里找最近的
3. Decoder从这个codebook entry还原出mesh
4. 反复训练，codebook就学到了"哪些shape primitive最常出现"

训练好之后，codebook就成了一个"shape语言"——任何hand或object都可以用这512个"词"的组合来表达。

### Image-to-shape encoder在干嘛

现在你有了"shape语言"，下一步是从image直接预测"这个voxel对应字典里哪个词"。

ResNet-18抽image feature → 每个voxel投影到image plane拿对应feature → 3D conv预测每个voxel在这512个词上的probability distribution。

为什么要加segmentation mask作为input？因为synthetic data的RGB和real RGB差别太大（光照、纹理、材质全不一样），但mask几乎没差别——一个杯子的轮廓在synthetic和real里看起来差不多。这是sim-to-real的关键。

### Stereo fusion的超简单公式

$$P = P_L \odot P_R$$

就这一个公式。$P_L$是左view的probability distribution（shape $D \times D \times D \times C$），$P_R$是右view的，相乘就是element-wise乘。

数学上这是Bayesian的conjunction（两个独立观察的联合probability），假设两个view的prediction独立、prior uniform。实践中效果出奇地好——stereo setting的object CD改进35 cm²，比single-view的改进9 cm²大了快4倍。

### Outlier removal这一步很关键

Reconstruct出来的pointcloud往两个view re-project，用mask一filter——re-project出来落在mask外面的点直接删掉。这一步看起来trivial，但效果巨大。Fig. 6里可以看到，没有这一步的话reconstruction会有很多"飘在空中"的noise点，加上之后干净很多。

## 实验数据讲人话

### Reconstruction quality

DexYCB上测试，StereoHO vs baseline：

| Setting | 对比 | Object CD改进 | Hand CD变化 |
|---------|------|---------------|-------------|
| Single-view | vs IHOI | +9.71 cm² (seen), +12.52 cm² (unseen) | -1.10 cm² (略差) |
| Stereo | vs SVHO | +35.43 cm² (seen), +32.27 cm² (unseen) | +0.51 cm² (略好) |

**为什么object改进大，hand改进小甚至略差？**

因为object形状diverse（瓶子、盒子、手机…），prior的学习帮助大；而hand其实就是MANO model那一种拓扑结构，IHOI这种conditioned on hand pose的方法更针对hand，所以hand reconstruction上StereoHO没占到便宜。

### Handover success（这是真正落地的指标）

CORSMAL containers（杯子、玻璃杯，可能装米）：

| Method | G (grasp) | D (delivery) |
|--------|-----------|--------------|
| CB [6] (专门为container设计) | 0.79 | 0.67 |
| DB (depth only) | 0.41 | 0.50 |
| ClearGrasp (depth completion) | 0.16 | 0.05 |
| **StereoHO** | 0.75 | 0.66 |

StereoHO和CB打平，但CB只能处理container，StereoHO啥都能处理。

Household objects（8种不同形状，含透明喷雾瓶、螺丝刀、布、CD盒）：

| Method | G_avg |
|--------|-------|
| DB | 0.06-0.50 |
| ClearGrasp | 0.00-0.16 |
| **StereoHO** | 0.83 |

StereoHO在透明物体上拿到1.00的grasp success——depth-based方法完全瞎了，StereoHO靠RGB照样能看。

## 这篇paper的真正贡献

1. **Probabilistic multi-view fusion**：用probability distribution over codebook作为fusion interface，element-wise multiplication就是Bayesian update的简化版。这个idea可以推广到任何multi-view reconstruction。

2. **Learned shape priors替代handcrafted priors**：不假设旋转对称、不假设upright，让network从data学shape distribution。Generalization强很多。

3. **Sim-to-real via domain-invariant input**：用mask不用RGB，简单但有效。

4. **完整deployable pipeline**：从detection到grasp到robot control，不是paper-only的demo。

## 个人intuition

这篇paper让我想到几个更深层的问题：

**为什么element-wise multiplication这么有效？** 它本质上假设两个view的prediction independent且calibrated。实际上两个view看同一个scene，prediction肯定有correlation（共享的occlusion、光照等）。但实验证明这个approximation够用。如果要用更principled的方法，比如conditioned fusion或者attention，可能overkill。

**VQ codebook size 512够吗？** ObMan只有8类object，512个codebook entry足够区分。但如果扩展到ShapeNet全量（55类、几十万model），codebook可能需要扩到几千。VQ-VAE的codebook collapse问题（只有少部分entry被使用）也是open problem。

**为什么不用更新的方法？** 比如：
- [DiffHOI](https://arxiv.org/abs/2303.12884) 用diffusion，能表达multi-modal distribution（同一view可能有多种合理completion）
- [HandNeRF](https://arxiv.org/abs/2310.18709) 用NeRF，表达力更强
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 训练快、render快

推测作者选VQ-VAE是因为：
1. Inference速度快（handover需要reactive）
2. Discrete distribution天然适合multi-view fusion
3. Codebook可以pretrain，image encoder部分轻量

Diffusion的inference慢（需要几十步denoise），NeRF需要per-scene优化，都不适合real-time handover。

**Failure mode的启示**：paper提到cracker box被从wider side grasp导致失败。这说明reconstruction quality好不代表grasp success高——还需要考虑gripper的physical constraint。一个改进方向是在grasp estimation时显式encode gripper width，而不只是filter hand collision。

## 最后的intuition

这篇paper让我最印象深刻的是：**probability is the universal interface for fusion**。

不管是multi-view fusion、multi-modal fusion（RGB + depth + tactile）、还是temporal fusion（frame-by-frame tracking），只要每个observation输出calibrated probability distribution，就能用乘法或Bayesian update融合。这个idea远超hand-object reconstruction的scope，是一个通用的perception design principle。

相比之下，deterministic regression输出一个point estimate，fusion时只能用average或者attention这种heuristic，缺乏principled的uncertainty reasoning。

这也是为什么最近probabilistic representation（VQ-VAE、diffusion、flow matching、EBM）在perception task上越来越流行——它们不只是"更generative"，而是提供了更好的**information fusion interface**。

## 参考链接

- [Paper PDF (arXiv版本应该会有)](https://arxiv.org/abs/2502.xxxxx)
- [CORSMAL benchmark](https://corsmal.eecs.qmul.ac.uk/)
- [ObMan dataset](https://hassony2.github.io/obman.html)
- [DexYCB dataset](https://dex-ycb.github.io/)
- [AutoSDF (VQ-VAE for shape)](https://arxiv.org/abs/2206.03529)
- [VQ-VAE原paper](https://arxiv.org/abs/1711.00937)
- [6-DoF GraspNet](https://github.com/NVlabs/6dof-graspnet)
- [FastSAM](https://arxiv.org/abs/2306.12156)
- [FrankMocap](https://github.com/facebookresearch/frankmocap)
- [DiffHOI (对比方法)](https://arxiv.org/abs/2303.12884)
- [HandNeRF (对比方法)](https://arxiv.org/abs/2310.18709)

---

# StereoHO: Stereo Hand-Object Reconstruction for Human-to-Robot Handover 详细解析

## Paper核心背景与Problem Definition

这篇paper来自Queen Mary University of London的Alessio Cavallaro实验室,一作Yik Lung Pang。核心问题是**human-to-robot handover**——即人类把物体递给robot,robot需要安全地接收物体并送到目标位置。这个task需要解决两个关键perception问题:

1. **Object shape reconstruction** — 用于grasp estimation
2. **Hand shape reconstruction** — 用于collision avoidance保障human safety

### 现有方法的痛点

**Depth sensor方法的局限**:
- 透明物体(如drinking glass)的depth sensing失效[1]
- Single camera存在occlusion和partial visibility问题
- Depth completion方法(如ClearGrasp)只能补全visible side的depth

**Stereo RGB方法的局限**(如CORSMAL baseline [6]):
- 假设object是rotationally symmetric(container-like)
- 假设object在handover过程中保持upright orientation
- 无法处理non-container物体

Reference: [CORSMAL benchmark paper](https://doi.org/10.1109/LRA.2020.2969184)

## 核心创新:StereoHO

StereoHO的核心insight是:用**learned 3D shape priors**替代handcrafted geometric priors,并通过**probability distribution over codebook**来quantify single-view prediction的uncertainty,然后通过element-wise multiplication进行probabilistic fusion。

关键设计选择:
- **T-SDF (Truncated Signed Distance Field)** 作为shape representation,而不是pointcloud。SDF是implicit representation,smooth且continuous,容易学习[2]
- **Vector Quantization**学习discrete shape codebooks,降低embedding dimensionality
- **SegMask**作为input,facilitate sim-to-real transfer(domain-invariant)

## 方法详解

### A. 整体架构

StereoHO的inference pipeline包含三个stage:
1. **Encoding**: 从stereo images预测probability distribution over shape codebook
2. **Aggregation**: 通过element-wise multiplication融合两个view的predictions
3. **Decoding**: SDF decoder从aggregated shape codes生成T-SDF,sampling得到pointcloud

### B. Discrete 3D Shape Embeddings学习(VQ-VAE部分)

这是paper最核心的技术创新。使用3D Patch-wise Encoding VAE with Vector Quantization学习shape codebook。

**Input**: T-SDF $s \in \mathbb{R}^{D \times D \times D}$,其中$D=128$

**Encoding过程**:
- 3D convolutional layers将local patches of T-SDF编码成continuous shape code $z_e \in \mathbb{R}^S$,其中$S=128$是embedding dimension
- Vector quantization将$z_e$映射到最近的codebook entry $e \in \mathbb{R}^S$
- Codebook $\mathcal{C} = \{e_c\}_{c=1}^C$,其中$C=512$

**Loss function**:

$$\mathcal{L}_{ae} = |s - \hat{s}| + ||sg[z_e] - e||_2^2 + \beta ||z_e - sg[e]||_2^2$$

变量解释:
- $s$: ground-truth T-SDF values
- $\hat{s}$: reconstructed T-SDF values
- $z_e$: encoder输出的continuous embedding
- $e$: 从codebook中quantized得到的embedding($e$是$z_e$在codebook中的最近邻)
- $sg[\cdot]$: stop gradient operator,阻止梯度流过
- $\beta = 1.0$: commitment loss的weight,控制encoder输出与codebook embedding的对齐程度

**三个loss term的intuition**:
1. $|s - \hat{s}|$: reconstruction loss,L1 norm保证SDF重建精度
2. $||sg[z_e] - e||_2^2$: codebook loss,让codebook embedding $e$ 靠近encoder输出 $z_e$(stop gradient on $z_e$)
3. $\beta ||z_e - sg[e]||_2^2$: commitment loss,让encoder输出 $z_e$ 承诺到一个codebook vector(stop gradient on $e$)

**VQ-VAE的tricky之处**: codebook是离散的,standard backprop无法通过argmin操作传递gradient。Stop gradient + commitment loss这个trick来自[van den Oord et al., Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)。直通估计让gradient直接从decoder传到encoder。

### C. Image-to-Shape Encoder训练

使用ResNet-18 (ImageNet pretrained)作为image encoder,3D conv作为prediction heads。

**输入**:
- Cropped RGB image centered on hand
- Hand segmentation mask $M_H$
- Object segmentation mask $M_O$

**输出**: 对每个voxel,预测codebook中各embedding的probability distribution $P_v \in [0,1]^{D \times D \times D \times C}$,其中$v \in \{L, R\}$是view index。

**3D grid定义**: 以wrist pose $T_{Hv}$为中心定义3D grid。每个voxel通过wrist pose投影到image space,与对应pixel位置的image embedding关联。

**Loss function**:

$$\mathcal{L}_{ce} = -\sum_c^C w_c \cdot p_{gt}(c) \cdot \log p(c)$$

变量解释:
- $C = 512$: codebook size
- $p_{gt}(c)$: ground-truth probability for index $c$
- $w_c$: class weight,empty space的index设为$w_c = 0.25$,其余设为$w_c = 0.75$
- $p(c)$: predicted probability

**Intuition**: 3D grid中大部分voxel是empty space,如果不加权,model会degenerate到只预测empty class。通过给empty space较低weight,迫使model关注hand和object的surface voxels。

### D. Stereo Aggregation

**核心公式**: $P = P_L \odot P_R$(element-wise multiplication)

这是paper的关键设计。两个view独立预测probability distribution后,通过element-wise multiplication得到fused distribution。

**Intuition**: 假设两个view的prediction独立且calibrated,multiplication相当于计算两个probability的conjunction。如果一个view对某个voxel的prediction uncertain(uniform distribution),它不会dominate融合结果;如果两个view都confident且consistent,fused probability会更高。

**Shape code selection**: 选择fused distribution中argmax对应的codebook embedding:

$$E \in \mathbb{R}^{D \times D \times D \times S}$$

其中每个voxel的embedding $E_{i,j,k} = \mathcal{C}[\text{argmax}_c P_{i,j,k,c}]$。

**Multiview consistency**: 通过将predicted pointcloud $\mathcal{P}$用wrist pose $T_{Hv}$ re-project回每个view,用segmentation mask $M_H, M_O$ filter outliers,得到最终pointcloud $\mathcal{P}'$。

### E. 完整Handover Pipeline

```
Input frame → Hand-Object Detection → Crop + Segment + Wrist Pose Estimation
            → StereoHO Reconstruction → Grasp Estimation → Robot Control
```

**各模块技术栈**:
1. **Detection**: [Understanding human hands in contact](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.pdf) - bounding box estimation
2. **Segmentation**: [FastSAM](https://arxiv.org/abs/2306.12156) - object mask
3. **Wrist pose + Hand mask**: [FrankMocap](https://arxiv.org/abs/2108.09149) - monocular 3D pose
4. **Grasp**: [6-DoF GraspNet](https://arxiv.org/abs/1905.10520) - variational grasp generation
5. **Robot**: UR5 + Robotiq 2F-85 gripper

**Triangulation validation**: object mask centroid在两个view间triangulate,reprojection error < 5 pixels才算valid。

**Quality monitoring**: 用convex hull的IoU与mask对比,只保留reconstruction质量提升的frame(IoU > IoU*)。

**Safety filtering**: 6-DoF GraspNet生成N=200个candidate grasps,每个grasp周围构造gripper size的3D bounding box,若hand pointcloud $\mathcal{P}_H'$进入该bbox则被filter掉。

## 实验结果详解

### Reconstruction Quality (DexYCB)

**Single-view setting** vs IHOI [3]:
- Object CD改进: seen category +9.71 cm², unseen category +12.52 cm²
- Hand CD略差: seen -1.10 cm², unseen -1.21 cm²

**Stereo setting** vs SVHO [4]:
- Object CD改进: seen +35.43 cm², unseen +32.27 cm² (大幅改进!)
- Hand CD略好: seen +0.51 cm², unseen +0.13 cm²

**关键观察**: stereo setting的object reconstruction改进远大于single-view,说明stereo fusion有效。

### Handover Performance (Table I)

**Containers (CORSMAL benchmark)**:
- StereoHO: G=0.75, D=0.66, δ=0.48, γ=0.12, μ=0.74
- CB [6]: G=0.79, D=0.67 (在container场景略优,因为专门设计)
- DB: G=0.41, D=0.50 (depth在透明物体上失效)
- ClearGrasp: G=0.16, D=0.05 (depth completion质量差)

**Household objects (8个不同形状物体)**:
- StereoHO: G_avg=0.83, D_avg=0.58
- 透明wine glass: G=1.00, D=0.91 (大幅领先!)
- 透明spray bottle: G=0.75, D=0.83
- 非透明cracker box: G=0.50, D=0.50

**关键发现**: 在透明物体上StereoHO全面碾压depth-based方法,这正是paper的motivation。

## 技术细节深挖与Intuition Building

### 为什么用T-SDF而不是pointcloud或voxel?

**T-SDF的优势**:
1. **Compact representation**: 128³ grid vs 几十万pointcloud
2. **Differentiable**: implicit function,可backprop
3. **Complete surface**: 不像pointcloud只表达visible部分
4. **Easy to learn**: smooth continuous surface,比binary voxel occupancy更容易优化

**Truncation的意义**: 只在surface附近±1cm范围保留SDF values,远离surface的voxel直接设为±τ。这样network focus在surface附近,减少compute和memory。

### 为什么用VQ-VAE而不是continuous VAE?

**Discrete codebook的好处**:
1. **Mode capturing**: codebook entries可以理解为"shape primitives"
2. **Generalization**: discrete space减少overfitting
3. **Stable training**: 避免posterior collapse问题(标准VAE的常见问题)
4. **Multi-view fusion**: probability distribution over codebook提供了天然的fusion interface

**AutoSDF**[5]是这个idea的origin,本文将其应用到hand-object reconstruction。

### Stereo Aggregation的Mathematical Interpretation

Element-wise multiplication $P = P_L \odot P_R$ 可以理解为:

假设两view的prediction独立,联合probability:
$$P(c | I_L, I_R) \propto P(c | I_L) \cdot P(c | I_R)$$

这是Bayesian inference的特殊形式(uniform prior)。当一个view occluded时,其prediction接近uniform,不会影响另一view的confident prediction。这是paper处理self-occlusion的核心mechanism。

### Sim-to-real Transfer Strategy

**关键设计**: 使用segmentation mask作为input,而不是raw RGB。Mask是domain-invariant的(synthetic和real的mask分布相似),raw RGB则存在domain gap。

**Training data**: ObMan [6] - 87,190 synthetic images,MANO hand model grasp ShapeNet objects (8 categories)。

**Testing data**: DexYCB [7] - real-world hand-object manipulation videos。

这个sim-to-real gap是hand-object reconstruction的核心挑战。StereoHO通过mask + learned shape priors partially address这个问题。

## 延伸联想与Open Problems

### 1. 与NeRF/SDF表示的对比

StereoHO用的是discrete T-SDF grid,而NeRF用continuous MLP。NeRF的优势是连续resolution,但需要dense views。StereoHO的sparse stereo setting使得NeRF不太适用,但[HandNeRF](https://arxiv.org/abs/2310.18709)已经尝试single-view NeRF。

**Potential direction**: 用[Instant-NGP](https://github.com/NVlabs/instant-ngp)或类似fast NeRF作为shape representation,可能比discrete T-SDF更有表达力。

### 2. Diffusion Models的Potential

最近[DiffHOI](https://arxiv.org/abs/2303.12884)等用diffusion做hand-object reconstruction。Diffusion可以更好地capture multi-modal distribution(同一view可能有多种合理的shape completion)。

StereoHO的VQ-VAE + cross-entropy是unimodal的,argmax selection丢失了uncertainty。Diffusion可以保留distribution。

### 3. Foundation Models Integration

**SAM segmentation**: paper用FastSAM,可以换成[SAM 2](https://arxiv.org/abs/2408.00714)获得更好的mask quality。

**Hand pose estimation**: FrankMocap可以换成[SMPLer-X](https://github.com/caizhongang/SMPLer-X)等更准确的whole-body pose estimator。

**3D priors**: 可以用[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)替代T-SDF,获得更photorealistic的reconstruction。

### 4. Temporal Consistency

Paper是frame-by-frame的,虽然有IoU-based quality monitoring,但没有显式temporal smoothing。可以加入:
- Kalman filter on shape code
- Recurrent state update (LSTM on shape embeddings)
- Optical flow warping + temporal loss

### 5. Active Perception

StereoHO是passive perception。可以扩展为active perception:
- Robot主动调整camera角度获得更好view
- Active next-best-view selection based on current reconstruction uncertainty

### 6. Manipulation Beyond Grasping

Paper只做grasp-based handover。可以扩展到:
- Dexterous manipulation (multi-finger)
- Tool use (reconstruct tool affordance)
- Bimanual handover (two-handed objects)

### 7. Physics-aware Reconstruction

T-SDF只表达geometry,不表达material properties。可以扩展到:
- Transparency estimation
- Mass estimation (paper中的μ metric只是间接评估)
- Deformable object reconstruction (cloth in Fig. 9)

### 8. Failure Mode Analysis

Paper提到的failure cases:
- Cracker box从wider side被grasp (gripper width mismatch)
- Hand occlusion by robot arm导致tracking loss

**Potential solutions**:
- Grasp feasibility check with gripper width constraint
- Multi-hypothesis tracking (类似MHT)
- Robot arm self-occlusion reasoning

## 与最新工作的对比

### vs [gSDF](https://arxiv.org/abs/2304.04205) (CVPR 2023)
gSDF用geometry-driven SDF,直接从image预测SDF。StereoHO的VQ-VAE更modular,但gSDF是end-to-end。

### vs [HandNeRF](https://arxiv.org/abs/2310.18709) (ICRA 2024)
HandNeRF用NeRF做single-view hand-object reconstruction。NeRF的表达力更强但训练慢。StereoHO的speed advantage明显。

### vs [DiffHOI](https://arxiv.org/abs/2303.12884) (ICCV 2023)
DiffHOI用diffusion model生成multi-view hand-object sequence。更 expressive但inference慢。StereoHO更适合real-time handover。

## Critical Review

### 优点
1. **Probabilistic formulation**: element-wise multiplication提供principled multi-view fusion
2. **Generalization**: 不依赖object-specific priors,可以处理diverse objects
3. **Real-world deployment**: 完整pipeline,不只是lab demo
4. **Transparent object support**: RGB-only避开了depth sensor痛点

### 限制
1. **Sim-to-real gap**: 训练只在ObMan synthetic,测试在DexYCB real
2. **Speed**: Table I的efficiency score只有0.12,远低于CB的0.78
3. **Hand reconstruction略差**: single-view setting比IHOI差1.1 cm²
4. **Static assumption**: 不处理deformable objects的articulated structure
5. **Codebook size**: C=512可能限制expressiveness,但对大多数object足够

## 个人Intuition与启发

1. **Probability fusion的power**: element-wise multiplication这个简单操作,在calibrated probability下效果惊人。这让我想到Bayesian model averaging的思想。

2. **VQ作为uncertainty quantification**: 用discrete distribution天然表达epistemic uncertainty。比起deterministic regression更principled。

3. **Domain-invariant input**: 用mask而非RGB作为input是sim-to-real的关键insight。可以推广到其他sim-to-real场景。

4. **Grasp-then-filter的safety paradigm**: 生成多个candidate grasps再用hand geometry filter,比直接预测safe grasp更robust。

5. **Wide-baseline stereo的sweet spot**: dense multi-view质量高但慢,single-view快但incomplete。Wide-baseline stereo是reactive scenario的合理trade-off。

## References

- [1] [ClearGrasp](https://arxiv.org/abs/1911.01491)
- [2] [Implicit Representation Learning](https://arxiv.org/abs/1812.02613)
- [3] [IHOI - CVPR 2022](https://arxiv.org/abs/2204.09312)
- [4] [SVHO - CVPRW 2024](https://openaccess.thecvf.com/CVPR2024)
- [5] [AutoSDF](https://arxiv.org/abs/2206.03529)
- [6] [ObMan dataset](https://arxiv.org/abs/1904.05730)
- [7] [DexYCB](https://arxiv.org/abs/2104.04703)
- [CORSMAL benchmark](https://corsmal.eecs.qmul.ac.uk/)
- [6-DoF GraspNet](https://github.com/NVlabs/6dof-graspnet)
- [VQ-VAE](https://arxiv.org/abs/1711.00937)

这篇paper的insight是probabilistic multi-view fusion + learned discrete shape priors,而不是handcrafted geometric assumptions。它为human-robot interaction的vision-based perception提供了一个principled且deployable的framework,特别是在透明物体和diverse object categories上展示了明显优势。后续可以探索diffusion-based shape priors、active perception以及temporal consistency来进一步提升performance。
