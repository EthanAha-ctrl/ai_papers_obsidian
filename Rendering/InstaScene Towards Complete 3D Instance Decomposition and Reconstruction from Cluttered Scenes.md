---
source_pdf: InstaScene Towards Complete 3D Instance Decomposition and Reconstruction
  from Cluttered Scenes.pdf
paper_sha256: c2b793e79beea77f183d5c2c9cd4e2e1889386331eebfcc3846f144c989598ad
processed_at: '2026-08-05T09:56:18-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# InstaScene 人话版

## 0. 一句话讲清楚这 paper 在干嘛

想象你走进一个乱糟糟的厨房，台面上堆满了碗碟、刀叉、调料瓶。你眼睛一扫，脑子里瞬间就能把每个东西"单独拎出来"——哪怕一个瓶子被另一个挡住一半，你也知道它是完整的瓶子，能伸手去拿。

InstaScene 就是想让 AI 也具备这个能力：**从一个 messy 的 3D scene 里，把任意 object 精确拆出来，同时把它被遮挡的部分给"脑补"完整，最后给你一个能放回原位的完整 3D model**。

---

## 1. 为什么这事难

之前的三条路各自有坑：

**第一条路：NeRF / 3DGS / 2DGS 这类 reconstruction**
它们把整个 scene 当成一坨整体来建模，渲染出来的图确实漂亮，但里面没有"object"的概念——你没法说"把那个椅子单独拿出来"。整个 scene 就是一团带颜色的点，椅子、桌子、地板在它眼里没区别。

**第二条路：LangSplat / LERF / OpenScene 这类 open-vocabulary understanding**
能让你用文字或点击来 query 物体，但是它做不到 amodal completion——一个被挡住一半的杯子，它只认得露出来的那一半，挡住的部分就丢了。你要拿这个杯子去做 manipulation，手里拿的是半个杯子。

**第三条路：ShapeNet prior / CAD alignment 这类 generative**
能从 partial observation 生成完整 shape，但问题是它们用的是 synthetic CAD dataset 训练的 prior，放到 real-world scene 里 domain gap 巨大。生成出来的椅子形状可能是"通用的椅子"，跟你 scene 里那把特定的椅子对不上——尺寸、纹理、风格全歪。你拿这个生成结果放回 scene，就像把一张宜家产品图 PS 进你家客厅，违和感爆表。

InstaScene 的核心 insight：**这三件事必须串起来做，而且必须用 scene 本身的 geometry 来约束 generative prior**。

---

## 2. 整个 pipeline 的画面感

输入：一段 posed RGB 视频（比如机器人拿着相机绕厨房走一圈）

```
Step 0: 用 2DGS 把 scene 重建出来
        → 你得到一坨带几何位置的 Gaussian 点云
        
Step 1: 跑 EntitySeg 给每帧生成 2D mask
        → 每帧上标出"这里有个瓶子，那里有个碗"
        
Step 2: 关键的分解阶段
        → 把这些 noisy 的 2D mask 变成干净稳定的 instance
        → 训练一个 16 维 feature field，让每个 Gaussian 知道自己属于谁
        
Step 3: 关键的补全阶段  
        → 对每个被分解出的 instance，用 generative prior 补它没看到的部分
        → 但补出来的东西必须跟 scene 里的真实 geometry 对齐
        
Done: 你可以从 scene 里 pick up 任意 object，得到一个完整的 3D model
```

---

## 3. 第一招：Scene Decomposition 怎么干

### 3.1 难点在哪

你拿到的是 EntitySeg 在每帧上跑的 2D mask，这些 mask 有三类毛病：

1. **Cross-view 不一致**：view A 里分出 5 个瓶子，view B 里 merge 成 3 个
2. **Under-segmentation**：相邻两个物体被打成一个 mask
3. **Floater**：背景里飘出来一些莫名其妙的 mask

如果直接用这些 mask 去监督一个 feature field（每个 Gaussian 挂一个 16 维 vector），那 feature 会非常糊——因为同一个物理 object 在不同 view 上被监督成不同的 label，feature 就学不清自己到底是谁。

### 3.2 Spatial Tracker: 把 2D mask "锚定"到 3D

InstaScene 的招是：**每个 2D mask，去 3D 里找一组 Gaussian 作为它的"代理"**。

具体怎么做：对帧 $I_i$ 里的第 $j$ 个 mask $m_{i,j}$，你用已知 pose 去 render 这一帧，看哪些 Gaussian 对 mask 区域的 pixel 贡献了 transmittance > 0.5。这些 Gaussian 就是这个 mask 的 spatial tracker $P_{i,j}$。

> 这里的 intuition：transmittance > 0.5 意味着这些 Gaussian 是"真正构成这个 mask 渲染的主力"，不是被前面挡住的幽灵点。把它们拎出来，就等于在 3D 空间里给这个 mask 钉了一颗锚。

### 3.3 View Consensus Rate: 跨帧 mask 配对

现在你有一堆 tracker，每个都是一组 3D 点。问题变成：哪些 tracker 属于同一个物理 object？

用一个叫 **view consensus rate** 的指标：

$$\mathcal{C}(P_{i,j}, P_{k,l}) = \frac{N_{contain}(P_{i,j}, P_{k,l})}{N_{vis}(P_{i,j}, P_{k,l})}$$

翻译成人话：
- $N_{vis}$: 两个 tracker 同时 visible 的帧数（30% 点参与对方帧的 rasterization 算 visible）
- $N_{contain}$: 两个 tracker 同时 contained 的帧数（80% 点落在对方 tracker 内算 contained）
- $\mathcal{C} > 0.9$ → 同一个 object

这个 threshold 配合（30% visible / 80% contained / 0.9 consensus）是 MaskClustering [Yan et al. CVPR 2024] 调出来的经验值，InstaScene 直接沿用。

**Under-segmentation 检测**: 如果 mask $m_{i,j}$ 的 tracker 同时与另一个帧 $I_k$ 里的多个 tracker 持续交集 → 说明这个 mask 把多个 object 打成一个了 → 直接 discard。

这一步做完，你得到：
- **干净的 2D cross-view mask 集合** $\mathcal{M}_n^{2d}$：每个 instance 一组跨 view 一致的 mask
- **3D global mask** $\mathcal{M}_n^{3d}$：把所有 tracker Gaussian 合并 + DBSCAN 去噪

### 3.4 Mutual Guidance: 为什么 2D 和 3D 监督都要

这里有个很 subtle 的点。

如果你只用 3D mask 做 supervision，DBSCAN 会误删一些 semantically meaningful 但 spatially 孤立的 Gaussian——比如物体边缘的稀疏点、被遮挡区域的零星点。你的 feature field 会变得"太干净"，反而漏掉一些本该属于这个 object 的点。

如果你只用 2D mask 做 supervision，cross-view noise 会污染 feature field，让它学得不分明。

InstaScene 的解法是**三路 contrastive loss 同时监督**：

$$\mathcal{L}_{\mathcal{F}} = \lambda_1 \mathcal{L}_{CF}(\mathbf{F}_i) + \lambda_2 \mathcal{L}_{CF}(\bar{\mathbf{F}}_i) + \lambda_3 \mathcal{L}_{CF}(\mathbf{f}_i^{3d})$$

- $\mathcal{L}_{CF}(\mathbf{F}_i)$: 单帧内 2D feature + filtered 2D mask → dense 监督，补 DBSCAN 删掉的点
- $\mathcal{L}_{CF}(\bar{\mathbf{F}}_i)$: 相邻帧 feature + cross-view clustered mask → 跨 view consistency 监督
- $\mathcal{L}_{CF}(\mathbf{f}_i^{3d})$: 3D Gaussian feature + 3D mask → robust 锚定，避免 2D noise 把 feature 带偏

这个 InfoNCE-style loss 的核心：

$$\mathcal{L}_{CF}(\mathcal{F}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{|\{f_i\}|}\log \frac{\exp(f_i^j \cdot \bar{f}_i / \phi_i)}{\sum_{k=1}^{N}\exp(f_i^j \cdot \bar{f}_k / \phi_k)}$$

- $f_i^j$: instance $i$ 的第 $j$ 个 feature
- $\bar{f}_i$: instance $i$ 的 mean feature（positive prototype）
- $\phi_i$: instance-specific temperature（与 instance 大小相关，防大 instance 主导）
- 分子拉近同 instance，分母推远跨 instance

> Intuition: 这就像在 feature space 里把每个 instance 捏成一团紧凑的"球"，不同 instance 的球互相分开。最后 inference 时用 cosine similarity > 0.9 就能干净切出 instance。

### 3.5 这招为什么有效

看 Table 2 ablation 就懂了：

| 配置 | Kitchen mIoU |
|---|---|
| 只用 raw noisy 2D mask | 71.2 |
| 只用 3D mask | 67.0 |
| + filtered 2D intra-view | 75.4 |
| + cross-view 2D mask | **77.3** |

注意一个反直觉的点：**raw noisy 2D mask 居然比 clean 3D mask 表现还略好**（71.2 vs 67.0）。原因是 3D mask 经过 DBSCAN 太激进了，删掉了一些有用的点。所以 mutual guidance 的价值就在于：3D 给你"骨架"，2D 给你"血肉"，缺一不可。

LERF-Mask 上 Table 1 对比：

| Method | Kitchen mIoU |
|---|---|
| LangSplat | 50.7 |
| GSGrouping | 43.1 |
| **Ours** | **77.3** |

LangSplat 用 CLIP feature，CLIP 对"一排相同的瓶子"会产生 feature collision——它分不清第几个瓶子是哪个。GSGrouping 用 video tracking，occlusion 下 tracking 会 drift。InstaScene 的 contrastive learning 是 instance-level discrimination，不存在 collision 问题。

---

## 4. 第二招：In-Situ Generation 怎么干

### 4.1 难点在哪

现在你从 scene 里拎出一个 instance（比如一个被挡住一半的婴儿车），想把它补全。

直接用 image-to-3D 模型（InstantMesh / SpaRP / MVDFusion）会出三个问题：

1. **Misalignment**: 生成出来的尺寸/姿态跟原 scene 对不上（Figure 4 b/c 里婴儿车的把手和座位都错位了）
2. **Domain gap**: generative prior 训练集是 synthetic/curated data，跟你 scene 里真实物体（皮革质感、特定花纹）对不上
3. **Floater**: 生成出来一堆游离的几何碎片

InstaScene 的招是 **"in-situ"**——"原地"补全。核心 idea: 用 scene 自身的 observation 当 leash，把 generative prior 牢牢拴在 real-world geometry 上。

### 4.2 Omni-Conditioned Diffusion: 多视角交替 conditioning

base model 是 MVDFusion [Hu et al. CVPR 2024]，本来是 single-view conditioned。InstaScene 改成：**每个 diffusion step 交替用不同 input view 做 condition，最后 average noise prediction**。

公式：

$$\bar{\epsilon}_\theta^n = \frac{1}{N_k}\sum_{k=1}^{N_k}\epsilon_\theta^n(x_t^n, y^k, \hat{\pi}_n^k)$$

- $x_t^n$: timestep $t$ 时 target view $n$ 的 noisy latent
- $y^k$: 第 $k$ 个 input view
- $\hat{\pi}_n^k$: input view $k$ 到 target view $n$ 的 relative pose
- $\epsilon_\theta^n$: conditioned noise predictor
- $\bar{\epsilon}_\theta^n$: 多 input view 的 noise prediction 平均

> Intuition: 单 view conditioning 会被那个 view 的视角 bias 带跑——比如 input 是正面照，生成的 back view 可能就偏离真实物体。多 view 交替 + 平均，相当于让模型"听取多方意见"，输出更 consistent。

### 4.3 Geometry-Aware Feature Warping: 把 observation 焊进 latent

但光交替 conditioning 还不够，因为 diffusion 仍可能"自由发挥"生成跟 observation 不一致的东西。

InstaScene 的核心 trick: **每个 diffusion step，把已知 input view 的 latent feature warp 到 target view，强行替换 visible 区域**。

流程：
1. Input view $y^k$ 的 latent $z^k_t$ 加 timestep $t$ 对应的 noise（让它的 noise level 跟 target $x_t^n$ 一致）
2. 用 2DGS render 出来的 depth $d^k$ 把 $z^k_t$ warp 到 target view $n$ 的 pixel 位置
3. 用 2DGS 的 surface normal 判断 warp 后的点是否 back-facing → discard
4. Front-facing 区域：用 warped feature 覆盖 $x_t^n$ 对应位置
5. Back-facing / 未被投影到的区域：保留 $x_t^n$ 的 noisy latent，让 diffusion 自己 denoise

> 这个 trick 的精髓：**visible 区域的 latent 被 observation "焊死"了**，diffusion 不能乱改。invisible 区域才让 generative prior 发挥。等于你在 latent space 上做了一个硬约束——"已知区域必须跟 observation 一致，未知区域你 generative prior 去脑补"。

这个跟 SDS / SJC 类方法的区别在于：那些方法是 soft guidance（通过 loss 引导），InstaScene 是 hard replacement（直接替换 latent）。hard replacement 不会让 diffusion 偏离 observation，但保留了 generative prior 在未知区域的 hallucination 能力。

### 4.4 Occlusion-Aware Viewpoint Selection: 哪些 view 当 input

Generic image-to-3D 默认用 elevated view 当 input。但 cluttered scene 里 elevated view 经常被其他物体挡住。

InstaScene 的策略：
1. 围绕 object 均匀采 16 个 viewpoints（参考 SyncDreamer [Liu et al. 2023]）
2. 对每个 viewpoint，render 它看到的 object，计算 camera 到 object 之间有多少 scene occlusion
3. 选 occlusion 最少 + object coverage 最大的几个 viewpoints 作为 input condition
4. 剩下的 viewpoints 视为 unseen，需要 generative prior 补
5. 用前面训好的 feature field render 2D instance mask，把这些 input view 的 background 滤掉

### 4.5 Joint Refinement: 把生成结果塞回 2DGS

最后一步: 把 source observations + generated views 一起拿来 fine-tune 这个 instance 的 2DGS。

- Source views: 锁住 visible 区域的 fidelity（确保跟原 scene 一致）
- Generated views: 补 unseen 区域的 geometry + appearance

这样得到的 2DGS 可以**直接放回原 scene**——这就是"in-situ"的含义。它不是独立的 3D model，而是 scene 里的一个 instance 被补全了。

### 4.6 这招为什么有效

Table 3 的数字最能说明问题：

| Method | PSNR (Known) | PSNR (Unknown) | CD | Vol IoU |
|---|---|---|---|---|
| 2DGS (原版) | 31.67 | 27.44 | 0.028 | 0.361 |
| MVDFusion | 17.19 | 17.46 | 0.081 | 0.531 |
| InstantMesh | 23.05 | 22.83 | 0.045 | 0.570 |
| SpaRP | 25.09 | 23.03 | 0.037 | 0.590 |
| **Ours** | **32.57** | **29.02** | **0.016** | **0.716** |

注意几个反直觉的点：

1. **Known view PSNR 32.57 比原版 2DGS 还略高** (31.67)。这意味着 generative refinement 不仅补全了 unseen，还**反过来改进了 visible 区域**——因为 generated views 给 known region 提供了更多 supervision 信号。

2. **CD (Chamfer Distance) 0.016 比原版 2DGS 的 0.028 还低**。原版 2DGS 在 unseen 区域是空洞或 floaters，CD 自然高。InstaScene 用 generative prior 补全后 geometry 更完整。

3. **所有 baseline（包括 multi-view 的 SpaRP）在 known views 上 fidelity 都很差**（PSNR 17-25）。为什么？因为它们是"从头生成"——拿 input view 当 condition，生成一个 3D model。这个 model 跟 scene 里的真实 object 是两回事。哪怕 input view 是 scene 的某个视角，生成结果也会偏离。InstaScene 的 "in-situ" 关键在于：**它不是"从头生成"，是"在已有 2DGS 上补全"**——source views 锚定 visible，generated views 补 unseen，joint fine-tune 出来的还是原来那个 2DGS，只是变完整了。

这个区别是本质的：generic image-to-3D 是 "synthesize from scratch"，in-situ generation 是 "complete from partial observation"。前者你拿到的是一个新 model，后者你拿到的是原 scene 的延伸。

---

## 5. 整个系统的 Intuition

用一句 Karpathy 风格的话总结：

**InstaScene 把 segmentation、reconstruction、generation 三件事捏成一个 loop——segmentation 产生的 spatial prior 直接 feed 给 generative module 当 condition，generative module 补全的 geometry 又能反过来让 instance boundary 更清晰。当前是 sequential，未来 end-to-end joint train 是显然的演进方向。**

整个 system 的核心 insight 可以拆成三条：

1. **2D mask noisy 但 dense，3D mask clean 但 sparse → contrastive learning 互相对齐**
   - 不是"用哪个"的问题，是"两个都用，互相互补"
   - 2D 补 3D 删多的点，3D 补 2D 跨 view 的 noise

2. **Generative prior 能补 occlusion 但不 align real scene → 用 geometry warping 把 visible 区域焊死在 latent 里**
   - Hard replacement > soft guidance
   - visible 区域 zero freedom，invisible 区域 full generative freedom

3. **Source observation 锚定 visible，generated views 补 unseen → joint fine-tune 一个 complete 2DGS**
   - 不是替换，是增广
   - 输出还是原来那个 2DGS，只是变完整了

---

## 6. 我看这 paper 的几个 take-away

### 6.1 Spatial Tracker 是个被低估的 trick

把 2D mask 通过 rasterization mechanism 反查到 3D Gaussian 集合，这个 idea 其实很通用。任何用 3DGS + 2D supervision 的工作都可以用这招做 cross-view consistent 监督。MaskClustering 提出了 view consensus rate，InstaScene 把它跟 contrastive learning 结合，形成一个可学习的 feature field。

未来 4DGS、dynamic scene、robot manipulation 都可以借鉴这个 pattern——用 rasterization 反查做 spatial-temporal tracker。

### 6.2 In-Situ Generation 是 image-to-3D 的新 paradigm

传统 image-to-3D 是 "input image → 3D model"，输入输出是两个东西。In-situ generation 是 "partial observation → complete observation"，输入输出是同一个东西的两种状态。

这个 paradigm shift 重要在哪里：它把 generative prior 从"创造工具"变成了"补全工具"。创造工具不受 observation 约束（可以 hallucinate），补全工具必须 align observation。这对 robotics 极其重要——robot 需要的是"现实世界里那个特定的椅子"，不是"一个通用椅子"。

类比: 这就像 image inpainting 跟 image generation 的区别。Image generation 是"给我一句话生成一张图"，image inpainting 是"给我这张图补全缺失区域"。前者无约束，后者必须保持已知区域一致。In-situ generation 是 3D 版的 inpainting，但用的是 generative prior 而不是 2D diffusion。

### 6.3 Feature Field 当 Segmentation Backbone

InstaScene 给每个 Gaussian 挂一个 16 维 embedding，用 contrastive learning 训练。这比 LangSplat 的 CLIP distillation 有几个优势：
- 16D vs CLIP 512D，存储和计算开销小
- Instance-level discrimination vs semantic-level，能区分相同物体
- 不依赖 CLIP 的 pretrain quality

这个 pattern 可以推广: 任何需要 fine-grained instance-level 3D perception 的任务，都可以用 "Gaussian + small embedding + contrastive learning" 这个组合。Feature 3DGS [Zhou et al. CVPR 2024]、OpenGaussian [Wu et al. 2024]、GARField [Kim et al. CVPR 2024] 都在这个方向上探索。

### 6.4 Limitation 其实是 opportunity

Paper 自己说不能处理 transparent / reflective / dynamic objects。这三类是 3DGS 本身的软肋——SH 表达不了镜面反射，alpha blending 假设 opaque surface，static 表示做不了 dynamic。

Future work 沿 4DGS [Wu et al. 2024] + PBR material [Verbin et al. 2022 Ref-NeRF] 这两条路推进几乎是必然的。另一个有意思的方向是 physics-aware generation——现在 in-situ generation 只补 geometry + appearance，如果还能补 physical material property（friction、mass、articulation），就真的能直接 feed 给 robot simulator 做 grasp planning 了。

### 6.5 跟 Robotics 的 gap

InstaScene 离 robotics 真正用上还有几个 gap：

1. **Speed**: diffusion + 2DGS fine-tune 估计分钟级 per instance。Robot 实时操作需要秒级。Future work 可能要用 LRM 这种 feed-forward 模型替代 diffusion。

2. **Pose requirement**: 需要 posed RGB sequence。Robot 上得有 SLAM / visual odometry。这个一般可以假设有。

3. **Cluttered scene 的极端 case**: 堆叠很深的物体（比如抽屉里塞满的东西）可能 visible region 太少，generative prior 也救不了。

4. **Articulated object**: 抽屉、门、可开合的物体，现在当 rigid body 处理。未来需要 part-level segmentation + articulation estimation。

5. **Interactive perception**: 人会主动凑近看被挡住的部分。InstaScene 是 passive——给什么 view 就用什么 view。Next step 是 active perception——robot 看到 occluded object，主动 move camera 去看被挡住的部分，然后触发 in-situ generation。

---

## 7. Related Work 速查表

### 3DGS Segmentation
- LangSplat: https://arxiv.org/abs/2406.09431
- GSGrouping: https://arxiv.org/abs/2404.18732
- Click-Gaussian: https://arxiv.org/abs/2404.05820
- OpenGaussian: https://arxiv.org/abs/2406.02058
- Feature 3DGS: https://arxiv.org/abs/2311.16596
- GARField: https://arxiv.org/abs/2404.11072
- OmniSeg3D: https://arxiv.org/abs/2311.15566
- MaskClustering: https://arxiv.org/abs/2404.01844
- SAI3D: https://arxiv.org/abs/2404.02170
- Open3DIS: https://arxiv.org/abs/2401.02708
- OVIR-3D: https://arxiv.org/abs/2310.10635

### Scene Understanding
- LERF: https://arxiv.org/abs/2303.09553
- OpenScene: https://arxiv.org/abs/2212.00676
- OpenMask3D: https://arxiv.org/abs/2306.13631
- Feature Splatting: https://arxiv.org/abs/2404.01223
- LEGaussians: https://arxiv.org/abs/2404.13684

### Gaussian Splatting
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 2DGS: https://arxiv.org/abs/2403.17888
- 4DGS: https://arxiv.org/abs/2403.11142

### 3D Generation
- MVDFusion: https://arxiv.org/abs/2311.16830
- SyncDreamer: https://arxiv.org/abs/2309.03453
- Zero-1-to-3: https://arxiv.org/abs/2303.11313
- Zero123++: https://arxiv.org/abs/2310.15110
- LRM: https://arxiv.org/abs/2311.04400
- InstantMesh: https://arxiv.org/abs/2404.07191
- SpaRP: https://arxiv.org/abs/2408.05705
- DreamFusion: https://arxiv.org/abs/2209.14988
- CLAY: https://arxiv.org/abs/2403.10482
- Coin3D: https://arxiv.org/abs/2405.05424

### Instance-aware Reconstruction
- RICO: https://arxiv.org/abs/2308.09325
- ObjectSDF++: https://arxiv.org/abs/2305.19074
- Object-NeRF: https://arxiv.org/abs/2109.07903
- LASA: https://arxiv.org/abs/2404.03813
- Part123: https://arxiv.org/abs/2401.12903

### 3D Inpainting
- DP-Recon: https://arxiv.org/abs/2503.07253
- O²-Recon: https://arxiv.org/abs/2308.09691
- Nerfiller: https://arxiv.org/abs/2311.01068
- Infusion: https://arxiv.org/abs/2404.11613
- SPIn-NeRF: https://arxiv.org/abs/2305.16825

### Robotics
- GraspSplats: https://arxiv.org/abs/2409.02084
- Splat-Mover: https://splatmover.github.io/

### Foundational
- EntitySeg: https://arxiv.org/abs/2211.05776
- SAM: https://arxiv.org/abs/2304.02643
- CLIP: https://arxiv.org/abs/2103.00020
- DINOv2: https://arxiv.org/abs/2304.07193
- DBSCAN: https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

### Datasets
- LERF-Mask: https://www.lerf.io/
- ZipNeRF: https://arxiv.org/abs/2304.06753
- Replica-CAD / Habitat 2.0: https://aihabitat.org/
- 3D-OVS: https://arxiv.org/abs/2305.15740
- ShapeNet: https://shapenet.org/
- 3D-FRONT: https://arxiv.org/abs/2111.13419
- COLMAP: https://arxiv.org/abs/1503.01832

### Project Page
- InstaScene: https://zju3dv.github.io/instascene/

---

# InstaScene: 详尽解析

## 1. Paper Overview 与 Motivation

InstaScene 是来自 Zhejiang University State Key Lab of CAD&CG 和 ByteDance 的合作工作，目标是解决一个对 robotics 至关重要的 perception 问题：**在 cluttered scene 中既要把 arbitrary instance 精确分解出来，又要保证每个 object 的 complete reconstruction，且 geometry 和 appearance 必须与 physical world 对齐**。

核心 motivation 可以用 Figure 1 一句话概括：human 走进一个 crowded kitchen，本能地能 pick up 某个 occluded utensil——这种 amodal completion 能力在 robotics 上一直是 open problem。三个现有 paradigm 各自 fail：

| Paradigm | Limitation |
|---|---|
| Generic reconstruction (NeRF, 3DGS, 2DGS) | scene 作为 undifferentiated whole，无法做 instance-level manipulation |
| Open-set scene understanding (LERF, OpenScene) | 能 query 但无法 amodal complete occluded instance |
| Category-specific generative (ShapeNet prior, CAD alignment) | domain gap 大，无法 generalize 到 real-world cluttered scene |

InstaScene 提出统一 pipeline，把 segmentation 和 generative completion 串成一个 system，关键 trick 是 segmentation 阶段产生的高质量 spatial prior 直接 feed 给后续 generative module 作为 conditioning。

Project page: https://zju3dv.github.io/instascene/

---

## 2. Preliminary: 2D Gaussian Splatting 增强

回顾下基础公式，因为后续的 spatial tracker 是基于 rasterization 机制设计的。

**2D Gaussian tangent plane formulation**:

$$P(u,v) = \mathbf{p}_k + \mathbf{s}_u \mathbf{t}_u u + \mathbf{s}_v \mathbf{t}_v v$$

- $\mathbf{p}_k \in \mathbb{R}^3$: $k$-th Gaussian 的 center 位置
- $\mathbf{s}_u, \mathbf{s}_v \in \mathbb{R}^+$: 两个 tangent direction 上的 scaling (radius)
- $\mathbf{t}_u, \mathbf{t}_v \in \mathbb{R}^3$: 单位 tangent vector，定义 Gaussian disk 的 orientation
- $(u,v) \in \mathbb{R}^2$: tangent plane 上的局部坐标

**2D Gaussian kernel**:

$$\mathcal{G}(\mathbf{u}) = \exp\left(-\frac{u^2+v^2}{2}\right)$$

注意相比 3DGS 用 anisotropic 3D ellipsoid，2DGS 把 volume collapse 成 oriented disk，几何质量更好（更适合 mesh extraction，这是后续 feature warping 用 surface normal 的前提）。

**Alpha blending rasterization**:

$$\mathbf{c}(\mathbf{x}) = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \mathcal{G}_i(\mathbf{u}(\mathbf{x})) \prod_{j=1}^{i-1}(1-\alpha_j \mathcal{G}_j(\mathbf{u}(\mathbf{x})))$$

- $\mathbf{x}$: pixel 射线
- $\mathbf{c}_i, \alpha_i$: $i$-th Gaussian 的 view-dependent color (SH) 和 opacity
- $\mathbf{u}(\mathbf{x})$: pixel $\mathbf{x}$ 投影到 Gaussian $i$ 的局部坐标
- 累积 transmittance $T_i = \prod_{j=1}^{i-1}(1-\alpha_j \mathcal{G}_j)$ 控制 depth-sorted blending

**Feature field 渲染**（本文的关键 augmentation）:

$$\mathbf{f}(\mathbf{x}) = \sum_{i=1}^{N} \mathbf{f}_i^{3d} \alpha_i \mathcal{G}_i(\mathbf{u}(\mathbf{x})) \prod_{j=1}^{i-1}(1-\alpha_j \mathcal{G}_j(\mathbf{u}(\mathbf{x})))$$

- $\mathbf{f}_i^{3d} \in \mathbb{R}^{16}$: 每个 Gaussian 上挂一个 16 维 randomly initialized embedding，作为 instance-aware feature
- $D=16$ 是经过 ablation 选的 trade-off
- 其他 attributes (position, scaling, rotation, opacity, SH) 全部 freeze，只训练 feature

这个设计的好处是 feature 跟 scene geometry 严格绑定，做后续 segmentation 时可以利用 rasterization 机制做 spatial tracking。

---

## 3. Spatial Contrastive Learning: 核心创新 1

### 3.1 问题分析

直接用 2D segmentation mask 监督 feature field 有问题（Figure 2a）：
- Cross-view inconsistency: SAM/EntitySeg 在 view A 上分出来 5 个瓶子，view B 上变成 3 个（merge 了）
- Under-segmentation: 相邻物体被打成一个 mask
- 这些 noise 直接通过 feature distillation 会让 3D feature 模糊

只用 3D mask（spatial tracker + DBSCAN）也有问题（Figure 2b）：
- DBSCAN 会误把 semantically meaningful 但 spatially isolated 的点当 outlier 删掉
- 边缘 Gaussians 经常被丢

Insight：**2D mask dense 但 noisy，3D mask sparse 但 robust → 互相对齐，mutual guidance**。

### 3.2 Mask Clustering with Spatial Gaussian Tracker

**核心 idea**: 不直接用 2D mask 做监督，而是把每个 2D mask "trace" 回 3D，找一组 Gaussians 作为这个 mask 的 spatial tracker，然后用 tracker 之间的 consensus 关系做 cross-view mask matching。

具体步骤：

1. 用 EntitySeg [Qi et al. 2022] 跑所有 frame，得到 class-agnostic 2D instance masks
2. 对 frame $I_i$ 中的 $j$-th mask $m_{i,j}$，做 forward render 得到 $\bar{I}_i$
3. 找出 transmittance $T > 0.5$ 的所有 Gaussians，定义为这个 mask 的 tracker $P_{i,j}$

> 这里 transmittance > 0.5 的含义：这些 Gaussians 对 mask 区域 pixel 贡献了至少一半的渲染权重，可以认为它们是 mask 的主要 contributors，而不是被前面 Gaussians 完全遮挡的"幽灵点"。

**View consensus rate**（关键 metric）:

$$\mathcal{C}(P_{i,j}, P_{k,l}) = \frac{N_{contain}(P_{i,j}, P_{k,l})}{N_{vis}(P_{i,j}, P_{k,l})}$$

- $N_{vis}(P_{i,j}, P_{k,l})$: 两个 tracker 都 visible 的帧数
- $N_{contain}$: 两个 tracker 同时 contained 的帧数
- **Visible 定义**: tracker 的 30% points contribute to $I_{i'}$ 的 rasterization
- **Contained 定义**: tracker 80% points 出现在另一个 tracker 内
- 阈值: $\mathcal{C} > 0.9$ → 两 mask 同属一个 instance

这个 30% / 80% / 0.9 的阈值组合是 MaskClustering [Yan et al. CVPR 2024] 提出，InstaScene 直接沿用。

**Under-segmentation 检测**: 如果 $m_{i,j}$ 的 tracker 同时与 frame $I_k$ 中多个 tracker $\{P_{k,j'}\}$ 交集，且这些交集在 $m_{i,j}$ 的所有 visible frames 上一直存在 → 判定 $m_{i,j}$ 是 under-segmentation → discard。

这个 trick 很关键：它把"被多个不同 instance 共享的 mask"识别出来，避免污染监督信号。

**3D mask 生成**: 同一 instance 的所有 tracker Gaussians 合并，DBSCAN [Ester et al. KDD 1996] 去 floaters，得到 robust 的 $\mathcal{M}_n^{3d}$。

### 3.3 Contrastive Learning Formulation

把 instance segmentation 看作 metric learning 问题：同 instance 的 feature 互相拉近，跨 instance 的 feature 推远。

**InfoNCE-style loss**:

$$\mathcal{L}_{CF}(\mathcal{F}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{|\{f_i\}|}\log \frac{\exp(f_i^j \cdot \bar{f}_i / \phi_i)}{\sum_{k=1}^{N}\exp(f_i^j \cdot \bar{f}_k / \phi_k)}$$

变量解析：
- $\mathcal{F} = \{f_i^j\}$: batch 内所有 features，按 instance 分组
- $N$: batch 内 instance 数
- $|\{f_i\}|$: 属于 instance $i$ 的 feature 数量
- $f_i^j$: instance $i$ 的第 $j$ 个 feature vector
- $\bar{f}_i = \frac{1}{|\{f_i\}|}\sum_j f_i^j$: instance $i$ 的 mean feature (作为 positive prototype)
- $\phi_i$: instance-specific temperature (与 instance 大小相关)
- 分子: anchor 与 positive prototype 的 similarity
- 分母: anchor 与所有 instance prototypes 的 similarity 之和

**Total feature loss**（三路监督）:

$$\mathcal{L}_{\mathcal{F}} = \lambda_1 \mathcal{L}_{CF}(\mathbf{F}_i) + \lambda_2 \mathcal{L}_{CF}(\bar{\mathbf{F}}_i) + \lambda_3 \mathcal{L}_{CF}(\mathbf{f}_i^{3d})$$

- $\mathbf{F}_i$: frame $i$ 上 single-view rendered 2D features with 2D masks（提供 dense 但 noisy 监督）
- $\bar{\mathbf{F}}_i = \{(\mathbf{F}_j, \mathcal{M}_i^{2d}) | j \in [i-k, i+k]\}$: 相邻 views 的 cross-view features，用 clustered masks $\mathcal{M}_i^{2d}$ 监督
- $\mathbf{f}_i^{3d}$: frame $i$ visible Gaussians 的 3D features，用 3D global mask $\mathcal{M}_i^{3d}$ 监督（提供 sparse 但 robust 监督）

这三路 supervision 形成了 mutual guidance：
- 2D supervision 补 DBSCAN 误删的 points
- 3D supervision 锚定 cross-view consistent identity

**Inference**: 用每个 instance 的 mean feature $\hat{f}_i^{3d}$ 作为 query，对所有 Gaussian feature 算 cosine similarity，threshold $\tau_{seg} = 0.9$ → 得到 instance mask。

### 3.4 实验结果（Scene Decomposition）

LERF-Mask Dataset 上的 mIoU (%)：

| Method | Figurines | Teatime | Kitchen | Average |
|---|---|---|---|---|
| LangSplat [Qin CVPR 2024] | 58.1 | 73.0 | 50.7 | 60.6 |
| GSGrouping [Ye ECCV 2025] | 59.0 | 72.3 | 43.1 | 58.1 |
| **Ours** | **85.7** | **93.7** | **77.3** | **85.6** |

在 Kitchen scene（最复杂）上提升最显著（+34 mIoU vs LangSplat）。原因是 LangSplat 用 CLIP feature，对重复物体（如一排相同的瓶子）会产生 feature collision。

**Ablation Study（Table 2）**：

| Configuration | Figurines | Teatime | Kitchen | Average |
|---|---|---|---|---|
| with $M_{noisy}^{2d}$ (raw noisy 2D masks) | 80.3 | 90.1 | 71.2 | 80.5 |
| with $M^{3d}$ (only 3D masks) | 81.5 | 88.5 | 67.0 | 79.0 |
| + $m_{filter}^{2d}$ (filtered 2D intra-view) | 83.9 | 91.4 | 75.4 | 83.6 |
| + $m_{cv}^{2d}$ (cross-view 2D masks) | **85.7** | **93.7** | **77.3** | **85.6** |

Intuition: raw 2D 已经能跑出 80+ mIoU，因为 LERF-Mask 的 scene 不算很 cluttered。但加 filtered 2D 和 cross-view 2D 都有稳步提升——证明 mutual guidance 在 noisy scene 下价值更大。

---

## 4. In-Situ Generation: 核心创新 2

### 4.1 问题定义

给定 partial reconstructed instance $\mathcal{T}$（已有 visible views $\{y^k\}$, depths $\{d^k\}$, viewpoints $\{\pi^k\}$），目标是预测 unseen views $\{x^n\}$，要求：
- Geometry 与 real scene 对齐（不能 hallucinate 出一个 generic chair 替换真椅子）
- Appearance 与 real scene 一致（皮革质感、花纹等）
- 未知区域合理 complete

公式化：

$$p(\{x^n\} | \{y^k, d^k, \hat{\pi}_n^k\})$$

- $\{x^n\}$: unseen views at viewpoints $\{\pi^n\}$
- $\hat{\pi}_n^k$: input view $k$ 与 target view $n$ 之间的 relative pose

### 4.2 Omni-Conditioned Diffusion

基于 MVDFusion [Hu et al. CVPR 2024]，但做了一个关键改造：**alternated view conditioning**。

原始 diffusion denoising step:

$$x_{t-1}^n = \frac{1}{\sqrt{\alpha_t}}x_t^n - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\bar{\epsilon}_\theta^n$$

$$\bar{\epsilon}_\theta^n = \frac{1}{N_k}\sum_{k=1}^{N_k}\epsilon_\theta^n(x_t^n, y^k, \hat{\pi}_n^k)$$

变量解析：
- $x_t^n$: timestep $t$ 上 target view $n$ 的 noisy latent
- $\alpha_t, \beta_t, \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$: 标准 DDIM/ddpm noise schedule
- $\epsilon_\theta^n(\cdot)$: noise predictor (conditioned on input view $y^k$ and relative pose $\hat{\pi}_n^k$)
- $\bar{\epsilon}_\theta^n$: 多个 input views 的 noise prediction 平均
- $N_k$: input views 数量

核心 idea: 每个 timestep 不固定用同一个 input view 做 condition，而是 sequential 切换不同 input views $y^k$，然后 average noise prediction。这避免了 single-view conditioning 的 bias（不同 input view 会引导出不同 result，平均后更稳定）。

### 4.3 Geometry-Aware Feature Warping

但 alternated view conditioning 还不够——generated views 与 observation 之间有 domain gap（Figure 9 显示会有 floaters）。

**Trick**: 在每个 diffusion iteration，把 input views 的 latent feature 加 time-dependent noise，然后用 rendering depth $d^k$ warp 到 target view $n$ 的 visible pixels。

具体：
1. Input view $y^k$ 的 latent $z^k_t$ 加 timestep $t$ 对应的 noise（与 $x_t^n$ 同 noise level）
2. 用 $d^k$ 和 pose $\hat{\pi}_n^k$ 做 warping，得到 $z^{k \to n}_t$ 在 target view 上的 projection
3. 用 2DGS 的 surface normal 判断 warp 后的点是否 back-facing → discard
4. Front-facing 区域：用 warped feature 替换/融合 $x_t^n$ 对应位置
5. Back-facing / occluded / 未投影到的区域：保留 $x_t^n$ 的 noisy latent，让 diffusion 自己 denoise

这个 warping 等价于在 feature space 上做几何约束——visible 部分必须与 observation 一致，invisible 部分由 generative prior 补全。

### 4.4 Occlusion-Aware Viewpoint Selection

Generic image-to-3D 默认用 elevated view 作 input。但 cluttered scene 中，object 经常被其他物体挡住，没有 clean view 可用。

InstaScene 的策略：
1. 在 object 周围均匀采样 16 viewpoints（参考 SyncDreamer [Liu et al. 2023]）
2. Render 每个 viewpoint，计算 object 与 camera 之间 scene occlusion
3. 选 occlusion 最少 + coverage 最大的 viewpoints 作为 input condition
4. 剩余 viewpoints 视为 unseen，需要 generative prior 补全
5. 用前面训练好的 feature field 渲染 2D instance mask，把 background 滤掉

最后一步 fine-tune instance 的 2DGS：**joint refinement with source observations + generated views**。Source views 锁定 visible 部分 fidelity，generated views 补全 unseen 部分。

### 4.5 实验结果（Instance Completion）

**Replica-CAD Dataset [Szot et al. NeurIPS 2021]** 上的量化（Table 3）：

| Method | PSNR↑ (Known) | PSNR↑ (Unknown) | SSIM↑ (Known) | SSIM↑ (Unknown) | LPIPS↓ (Known) | LPIPS↓ (Unknown) | CD↓ | F1↑ | Vol IoU↑ |
|---|---|---|---|---|---|---|---|---|---|
| 2DGS (Origin) | 31.67 | 27.44 | 0.976 | 0.918 | 0.034 | 0.093 | 0.028 | 0.734 | 0.361 |
| MVDFusion | 17.19 | 17.46 | 0.797 | 0.787 | 0.232 | 0.251 | 0.081 | 0.150 | 0.531 |
| InstantMesh | 23.05 | 22.83 | 0.853 | 0.862 | 0.129 | 0.139 | 0.045 | 0.382 | 0.570 |
| SpaRP | 25.09 | 23.03 | 0.881 | 0.868 | 0.112 | 0.129 | 0.037 | 0.406 | 0.590 |
| **Ours** | **32.57** | **29.02** | **0.979** | **0.944** | **0.028** | **0.066** | **0.016** | **0.767** | **0.716** |

观察：
- **Known views PSNR 32.57**：超过原始 2DGS（31.67）。说明 generative refinement 不仅补全 unseen，还**反向改进 known 区域**（generated views 提供更多 supervision）
- **Unknown views PSNR 29.02**：远超所有 baseline（best 23.03 SpaRP）
- **Chamfer Distance 0.016**：比 2DGS 的 0.028 还低一半，geometry alignment 极好
- **Volume IoU 0.716**：相比 single-view methods 的 0.5-0.59，大幅提升

直觉：**single-view 和 generic multi-view methods 在 known views 上 fidelity 也不行**——因为它们从头生成，对不上 real scene。InstaScene 是 "in-situ"——把 generated views 与 source views 联合 fine-tune 2DGS，known 区域用 source 锚定，所以 known PSNR/SSIM/LPIPS 与 origin 2DGS 持平甚至超越。

---

## 5. 整体 Pipeline 串联

```
Input: Posed RGB image sequence
   ↓
Stage 0: 2DGS reconstruction (Huang et al. SIGGRAPH 2024)
   → 得到每个 Gaussian 的位置/scale/rotation/opacity/SH
   ↓
Stage 1: EntitySeg 2D mask prediction (Qi et al. 2022)
   → 得到每帧的 class-agnostic 2D instance masks
   ↓
Stage 2a: Spatial Gaussian Tracker construction
   → 每个 mask 找 transmittance > 0.5 的 Gaussians 作为 tracker
   ↓
Stage 2b: View Consensus Rate clustering
   → C > 0.9 的 mask 合并；under-segmentation 检测并 discard
   ↓
Stage 2c: DBSCAN filtering → 3D global masks M^3d
   ↓
Stage 2d: Spatial Contrastive Learning
   → 训练 16D feature field，loss = λ1·L_CF(F_i) + λ2·L_CF(F̄_i) + λ3·L_CF(f_i^3d)
   ↓
Stage 2e: Segmentation by cosine similarity > τ_seg = 0.9
   → 得到每个 instance 的 Gaussian subset
   ↓
Stage 3a: Occlusion-aware viewpoint selection (16 viewpoints)
   → 选 occlusion 最少的几个 views 作为 input condition
   ↓
Stage 3b: Omni-conditioned diffusion with alternated views + geometry warping
   → 生成 unseen views
   ↓
Stage 3c: Joint fine-tune instance 2DGS with source + generated views
   → Complete instance reconstruction, "in-situ" 放回原 scene
```

---

## 6. 与相关工作的对比与联想

### 6.1 3D Segmentation with Gaussian Splatting

- **LangSplat** [Qin et al. CVPR 2024]：把 CLIP feature 蒸馏到 3DGS，做 open-vocabulary segmentation。Limitation: 重复物体（同一货架上的相同瓶子）CLIP feature 会 collide → 误判。InstaScene 的 contrastive learning 显式做 instance-level 区分，避开这个 issue。
  - Paper: https://arxiv.org/abs/2406.09431
  
- **GSGrouping** [Ye et al. ECCV 2025]：用 SAM + video tracking，把 2D mask track 到 3D。Limitation: video tracking 在 heavy occlusion 下会 drift，跨 mask ID 会乱。
  - Paper: https://arxiv.org/abs/2404.18732
  
- **Click-Gaussian** [Choi et al. ECCV 2025]：interactive segmentation，用户点一下输出 mask。不做 amodal completion。
  - Paper: https://arxiv.org/abs/2404.05820
  
- **OpenGaussian** [Wu et al. 2024]：point-level open-vocabulary，但 feature 是 per-point CLIP embedding，同样有 collision 问题。
  - Paper: https://arxiv.org/abs/2406.02058
  
- **Feature 3DGS** [Zhou et al. CVPR 2024]：把 CLIP/DINO feature 蒸馏进 3DGS。
  - Paper: https://arxiv.org/abs/2311.16596
  
- **GARField** [Kim et al. CVPR 2024]：hierarchical grouping，用 SAM mask 做 contrastive。也用 contrastive loss 但 supervision 是 raw SAM masks，没有 spatial tracker 做 cross-view consensus。
  - Paper: https://arxiv.org/abs/2404.11072
  
- **OmniSeg3D** [Ying et al. CVPR 2024]：hierarchical contrastive learning，无 mask 标注。但需要 dense samplable scene，cluttered scene 表现差。
  - Paper: https://arxiv.org/abs/2311.15566
  
- **MaskClustering** [Yan et al. CVPR 2024]：InstaScene 直接借鉴其 view consensus rate 公式。但 MaskClustering 只做 segmentation，不做 reconstruction completion。
  - Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Yan_MaskClustering_View_Consensus_Based_Mask_Graph_Clustering_for_Open-Vocabulary_3D_CVPR_2024_paper.html

- **SAI3D** [Yin et al. CVPR 2024]：memory-based mask propagation。
  - Paper: https://arxiv.org/abs/2404.02170
  
- **Open3DIS** [Nguyen et al. CVPR 2024]：2D mask guided 3D instance segmentation。
  - Paper: https://arxiv.org/abs/2401.02708

- **OVIR-3D** [Lu et al. CoRL 2023]：open-vocabulary 3D instance retrieval。
  - Paper: https://arxiv.org/abs/2310.10635

### 6.2 Open-Vocabulary Scene Understanding

- **LERF** [Kerr et al. ICCV 2023]：language embedded radiance field，CLIP feature 蒸馏进 NeRF。InstaScene 在其 dataset 上做 evaluation。
  - Paper: https://arxiv.org/abs/2303.09553
  
- **OpenScene** [Peng et al. CVPR 2023]：3D scene understanding with open vocabulary，点云级语义。
  - Paper: https://arxiv.org/abs/2212.00676
  
- **OpenMask3D** [Takmaz et al. 2023]：open-vocabulary 3D instance segmentation，基于 mask features。
  - Paper: https://arxiv.org/abs/2306.13631
  
- **Feature Splatting** [Qiu et al. 2024]：language-driven physics-based scene synthesis，把 feature 与 dynamics 结合。
  - Paper: https://arxiv.org/abs/2404.01223
  
- **LEGaussians** [Shi et al. CVPR 2024]：language embedded 3D Gaussians。
  - Paper: https://arxiv.org/abs/2404.13684

### 6.3 3D Reconstruction & Generation

- **3DGS** [Kerbl et al. SIGGRAPH 2023]：原始 3D Gaussian Splatting。
  - Paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
  
- **2DGS** [Huang et al. SIGGRAPH 2024]：2D oriented disks，InstaScene 的 reconstruction backbone。
  - Paper: https://arxiv.org/abs/2403.17888
  
- **NeRF** [Mildenhall et al. 2021]：原始 Neural Radiance Fields。
  - Paper: https://arxiv.org/abs/2003.08934
  
- **NeuS** [Wang et al. 2021]：volume rendering + SDF。
  - Paper: https://arxiv.org/abs/2106.10689
  
- **ZipNeRF** [Barron et al. ICCV 2023]：anti-aliased grid-based NeRF，InstaScene 在其 dataset 上做 eval。
  - Paper: https://arxiv.org/abs/2304.06753
  
- **MVDFusion** [Hu et al. CVPR 2024]：single-view 3D via depth-consistent multi-view generation，InstaScene 的 generative backbone。
  - Paper: https://arxiv.org/abs/2311.16830
  
- **SyncDreamer** [Liu et al. 2023]：multi-view consistent generation from single image。
  - Paper: https://arxiv.org/abs/2309.03453
  
- **Zero-1-to-3** [Liu et al. ICCV 2023]：zero-shot single image to 3D。
  - Paper: https://arxiv.org/abs/2303.11313
  
- **Zero123++** [Shi et al. 2023]：single image to consistent multi-view diffusion。
  - Paper: https://arxiv.org/abs/2310.15110
  
- **LRM** [Hong et al. 2023]：Large Reconstruction Model，单图到 3D mesh。
  - Paper: https://arxiv.org/abs/2311.04400
  
- **InstantMesh** [Xu et al. 2024]：single image to 3D mesh with sparse-view LRM。
  - Paper: https://arxiv.org/abs/2404.07191
  
- **SpaRP** [Xu et al. ECCV 2025]：fast 3D reconstruction from sparse views。
  - Paper: https://arxiv.org/abs/2408.05705
  
- **DreamFusion** [Poole et al. 2022]：text-to-3D via SDS。
  - Paper: https://arxiv.org/abs/2209.14988
  
- **CLAY** [Zhang et al. 2024]：controllable large-scale generative model for 3D assets。
  - Paper: https://arxiv.org/abs/2403.10482
  
- **Coin3D** [Dong et al. SIGGRAPH 2024]：interactive controllable 3D asset generation，作者团队之前的工作。
  - Paper: https://arxiv.org/abs/2405.05424

### 6.4 Instance-aware Reconstruction

- **RICO** [Li et al. ICCV 2023]：regularizing unobservable for indoor compositional reconstruction。
  - Paper: https://arxiv.org/abs/2308.09325
  
- **ObjectSDF++** [Wu et al. ICCV 2023]：object-compositional neural implicit surfaces。
  - Paper: https://arxiv.org/abs/2305.19074
  
- **Object-NeRF** [Yang et al. ICCV 2021]：object-compositional NeRF for editable scene rendering。
  - Paper: https://zju3dv.github.io/object_nerf/
  
- **OC-NeRF** [Wu et al. ECCV 2022]：object compositional neural implicit surfaces。
  - Paper: https://arxiv.org/abs/2207.07604
  
- **LASA** [Liu et al. CVPR 2024]：instance reconstruction from real scans with aligned shape annotation。
  - Paper: https://arxiv.org/abs/2404.03813
  
- **Part123** [Liu et al. SIGGRAPH 2024]：part-aware 3D reconstruction from single-view。
  - Paper: https://arxiv.org/abs/2401.12903

### 6.5 3D Inpainting & Amodal Completion

- **DP-Recon** [Ni et al. CVPR 2025]：concurrent work，decompositional neural scene reconstruction with generative diffusion prior。先做 geometry completion 再 refine texture（两阶段），InstaScene 是 single-step joint completion。
  - Paper: https://arxiv.org/abs/2503.07253
  
- **O²-Recon** [Hu et al. AAAI 2024]：occluded object reconstruction with 2D diffusion。
  - Paper: https://arxiv.org/abs/2308.09691
  
- **Nerfiller** [Weber et al. CVPR 2024]：scene-level 3D inpainting via 2D diffusion。
  - Paper: https://arxiv.org/abs/2311.01068
  
- **Infusion** [Liu et al. 2024]：3D Gaussians inpainting via depth completion。
  - Paper: https://arxiv.org/abs/2404.11613
  
- **SPIn-NeRF** [Mirzaei et al. CVPR 2023]：multi-view segmentation + perceptual inpainting in NeRF。
  - Paper: https://arxiv.org/abs/2305.16825
  
- **Repaint** [Lugmayr et al. CVPR 2022]：2D inpainting with DDPM。
  - Paper: https://arxiv.org/abs/2201.09865
  
- **LaMa** [Suvorov et al. WACV 2022]：large mask inpainting with Fourier convolutions。
  - Paper: https://arxiv.org/abs/2109.07161

### 6.6 Robotics & Scene Manipulation

- **GraspSplats** [Ji et al. 2024]：efficient manipulation with 3D feature splatting。
  - Paper: https://arxiv.org/abs/2409.02084
  
- **Splat-Mover** [Shorinwa et al. CoRL 2024]：multi-stage open-vocabulary robotic manipulation via editable Gaussian Splatting。
  - Paper: https://splatmover.github.io/

### 6.7 Dynamic & Urban Scene

- **Street Gaussians** [Yan et al. 2024]：dynamic urban scene modeling。
  - Paper: https://arxiv.org/abs/2401.01339
  
- **DrivingGaussian** [Zhou et al. CVPR 2025]：composite Gaussian Splatting for autonomous driving。
  - Paper: https://arxiv.org/abs/2406.01607

### 6.8 Foundational Models Used

- **EntitySeg** [Qi et al. 2022]：high-quality class-agnostic entity segmentation，InstaScene 用来生 2D masks。
  - Paper: https://arxiv.org/abs/2211.05776
  
- **SAM** [Kirillov et al. ICCV 2023]：Segment Anything Model。
  - Paper: https://arxiv.org/abs/2304.02643
  
- **DINOv2** [Oquab et al. 2023]：self-supervised vision transformer features。
  - Paper: https://arxiv.org/abs/2304.07193
  
- **CLIP** [Radford et al. ICML 2021]：contrastive language-image pretraining。
  - Paper: https://arxiv.org/abs/2103.00020
  
- **DBSCAN** [Ester et al. KDD 1996]：density-based clustering，用来去 floaters。
  - Paper: https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

### 6.9 Datasets

- **LERF-Mask** [Kerr et al. ICCV 2023]：LERF + 手工 instance mask 标注，3 个 scene。
  - https://www.lerf.io/
  
- **ZipNeRF Dataset** [Barron et al. ICCV 2023]：complex real-world scenes，用于 qualitative comparison。
  - https://github.com/google-research/multinerf
  
- **Replica-CAD** [Szot et al. NeurIPS 2021]：synthetic scenes with artist-created CAD objects，用于 instance completion quantitative。
  - https://aihabitat.org/
  
- **3D-OVS** [Liu et al. NeurIPS 2023]：weakly supervised 3D open-vocabulary segmentation。
  - Paper: https://arxiv.org/abs/2305.15740
  
- **ShapeNet** [Chang et al. 2015]：CAD model repository。
  - https://shapenet.org/
  
- **Scan2CAD** [Avetisyan et al. CVPR 2019]：CAD alignment in RGB-D scans。
  - Paper: https://arxiv.org/abs/1812.09102
  
- **3D-FRONT** [Fu et al. ICCV 2021]：3D furnished rooms。
  - Paper: https://arxiv.org/abs/2111.13419
  
- **COLMAP** [Schönberger et al. 2016]：SfM + MVS，用来得到 posed RGB sequence。
  - Paper: https://arxiv.org/abs/1503.01832

---

## 7. Critical Analysis: Strengths & Limitations

### 7.1 Strengths

1. **统一 framework**: segmentation 阶段输出的 spatial prior 直接被 generative stage 利用，不像传统 pipeline 各模块独立。
2. **Spatial contrastive learning mutual guidance**: 2D dense but noisy supervision 与 3D sparse but robust supervision 互补——这种 mutual refinement 是核心 insight。
3. **Geometry-aware feature warping**: 把 geometric prior 注入 diffusion latent，强制 visible 区域与 observation 一致，避免 generic generation 的 misalignment。这是 in-situ 与 generic image-to-3D 的本质区别。
4. **Joint fine-tune**: 不丢 source views，generated views 只补 unseen，所以 known PSNR 仍能保持 32.57（甚至略超 origin 2DGS 31.67）。

### 7.2 Limitations（paper 自己指出）

- 无法处理 dynamic objects（4D）
- 无法处理 transparent objects（玻璃杯，Gaussian 渲染假设 opaque surface）
- 无法处理 highly reflective objects（mirror、metal，view-dependent appearance 难以从 diffusion prior 重建）

### 7.3 潜在 future directions

- **4D extension**: 跟 4DGS [Wu et al. 2024] 这类工作结合，做动态 instance segmentation + amodal completion
- **Physics-based rendering**: 把 PBR material 替代 SH，让 transparent/reflective surfaces 可处理
- **End-to-end joint training**: 当前 segmentation 和 generation 是 sequential，如果能端到端 joint train，feature field 可以学到 generative-friendly representation
- **Multi-instance scene generation**: 当前每个 instance 独立 in-situ generation，如果 scene-level joint generation（如 DP-Recon 路线），可以保证 instance 之间 geometry consistency
- **Robotics integration**: 跟 GraspSplats / Splat-Mover 这类 manipulation work 结合，把 in-situ generation 的 complete geometry 作为 grasp planning input
- **Self-supervised contrastive temperature**: $\phi_i$ 当前是 instance-specific 但非 learnable，可以改为可学习的
- **Generative model backbone**: 当前用 MVDFusion，可以替换为 LRM、TripoSR、Direct3D 这类更 modern 的 LRM
- **Open-vocabulary feature**: 当前 16D feature 只能做 instance discrimination，如果加 CLIP feature distillation 可以做 open-vocabulary instance retrieval

---

## 8. 总结：Intuition for Andrej

InstaScene 的核心 intuition 三句话：

1. **2D segmentation priors noisy 但 dense；3D mask clean 但 sparse → 用 contrastive learning 互相对齐**
2. **Generic 3D generation 能 complete occlusion 但不 align real scene → 用 geometry warping 把 visible 区域"焊死"在 diffusion latent 里**
3. **Source observation 锚定 visible，generated views 补 unseen → joint fine-tune 一个 complete 2DGS**

整个 system 的关键是把 segmentation 和 generation 串成一个 loop——前者的 spatial prior 喂给后者，后者的 complete geometry 反过来可以让前者获得更清晰的 instance boundary。当前 paper 是 sequential，但 future work 朝端到端 joint train 演进几乎是必然。

从 deployment 角度看，这个 work 距离 robotics 真正用上还有 gap：
- 当前每个 instance 走完整 generative pipeline 很慢（diffusion + 2DGS fine-tune 估计分钟级）
- 需要 posed RGB sequence，对 robot 上 SLAM 有要求
- Cluttered kitchen 这种场景里的透明玻璃杯/反光金属餐具无法处理

但作为 path forward，把 segmentation + amodal completion + generation 统一进一个 3DGS-based system 是非常 solid 的 paradigm shift。类似工作的 next step 大概会沿这个方向继续：把 generative prior 越来越紧密地嵌入 reconstruction pipeline，而不是作为独立 inpainting step。

---

## References (key papers)

- InstaScene Project Page: https://zju3dv.github.io/instascene/
- 2DGS: https://arxiv.org/abs/2403.17888
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- MaskClustering: https://openaccess.thecvf.com/content/CVPR2024/html/Yan_MaskClustering_View_Consensus_Based_Mask_Graph_Clustering_for_Open-Vocabulary_3D_CVPR_2024_paper.html
- LangSplat: https://arxiv.org/abs/2406.09431
- GSGrouping: https://arxiv.org/abs/2404.18732
- LERF: https://arxiv.org/abs/2303.09553
- MVDFusion: https://arxiv.org/abs/2311.16830
- SyncDreamer: https://arxiv.org/abs/2309.03453
- EntitySeg: https://arxiv.org/abs/2211.05776
- SAM: https://arxiv.org/abs/2304.02643
- CLIP: https://arxiv.org/abs/2103.00020
- DINOv2: https://arxiv.org/abs/2304.07193
- DBSCAN: https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf
- Replica-CAD / Habitat 2.0: https://aihabitat.org/
- ZipNeRF: https://arxiv.org/abs/2304.06753
- InstantMesh: https://arxiv.org/abs/2404.07191
- SpaRP: https://arxiv.org/abs/2408.05705
- GARField: https://arxiv.org/abs/2404.11072
- Click-Gaussian: https://arxiv.org/abs/2404.05820
- OpenGaussian: https://arxiv.org/abs/2406.02058
- Feature 3DGS: https://arxiv.org/abs/2311.16596
- OmniSeg3D: https://arxiv.org/abs/2311.15566
- Open3DIS: https://arxiv.org/abs/2401.02708
- OVIR-3D: https://arxiv.org/abs/2310.10635
- OpenScene: https://arxiv.org/abs/2212.00676
- OpenMask3D: https://arxiv.org/abs/2306.13631
- Feature Splatting: https://arxiv.org/abs/2404.01223
- LEGaussians: https://arxiv.org/abs/2404.13684
- SAI3D: https://arxiv.org/abs/2404.02170
- Coin3D: https://arxiv.org/abs/2405.05424
- LRM: https://arxiv.org/abs/2311.04400
- Zero-1-to-3: https://arxiv.org/abs/2303.11313
- Zero123++: https://arxiv.org/abs/2310.15110
- DreamFusion: https://arxiv.org/abs/2209.14988
- CLAY: https://arxiv.org/abs/2403.10482
- NeuS: https://arxiv.org/abs/2106.10689
- NeRF: https://arxiv.org/abs/2003.08934
- DP-Recon: https://arxiv.org/abs/2503.07253
- O²-Recon: https://arxiv.org/abs/2308.09691
- Nerfiller: https://arxiv.org/abs/2311.01068
- Infusion: https://arxiv.org/abs/2404.11613
- SPIn-NeRF: https://arxiv.org/abs/2305.16825
- Repaint: https://arxiv.org/abs/2201.09865
- LaMa: https://arxiv.org/abs/2109.07161
- RICO: https://arxiv.org/abs/2308.09325
- ObjectSDF++: https://arxiv.org/abs/2305.19074
- Object-NeRF: https://zju3dv.github.io/object_nerf/
- LASA: https://arxiv.org/abs/2404.03813
- Part123: https://arxiv.org/abs/2401.12903
- GraspSplats: https://arxiv.org/abs/2409.02084
- Splat-Mover: https://splatmover.github.io/
- Street Gaussians: https://arxiv.org/abs/2401.01339
- DrivingGaussian: https://arxiv.org/abs/2406.01607
- ShapeNet: https://shapenet.org/
- Scan2CAD: https://arxiv.org/abs/1812.09102
- 3D-FRONT: https://arxiv.org/abs/2111.13419
- COLMAP: https://arxiv.org/abs/1503.01832
- 3D-OVS: https://arxiv.org/abs/2305.15740
