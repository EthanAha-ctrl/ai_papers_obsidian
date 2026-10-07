---
source_pdf: FE2E From Editor to Dense Geometry Estimator.pdf
paper_sha256: aad9ba3b5a69a256afa5aa3e61c5d8163532bb899487682b7afcb2fcc496e308
processed_at: '2026-08-04T08:13:13-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FE2E 论文用人话讲解

Karpathy 你好，让我用大白话再讲一遍这篇 paper 的核心思路，然后 still 保留技术细节。

## 一、这篇 paper 到底在说什么？

### 1.1 核心问题

假设你要训练一个 model，从单张 RGB image 预测 dense depth map（每个 pixel 的深度值）或者 surface normal map（每个 pixel 的法向量）。

**传统的 data-driven 路线**：收集 millions 张有 GT depth 的 image，train 一个 big model（比如 DepthAnything 用了 62.6M images）。这需要海量 data + 巨大 compute。

**Generative prior 路线**（Marigold, Lotus 等开创）：拿一个 pre-trained Stable Diffusion（T2I generator），用少量 data fine-tune。理由是 SD 在 LAION-5B 上训练过，已经 "understand the world"，depth 只是换个 output head。

### 1.2 FE2E 的 insight

这篇 paper 说：**等等，dense geometry prediction 是 image-to-image task，input 和 output 都是 image-shaped 的。我们为什么用 T2I generator 当 foundation？**

T2I generator 的 input 是 noise + text，output 是 image。它从来没见过 "image → image" 这种 mapping。

Image editor（如 Step1X-Edit, InstructPix2Pix）的 input 是 source image + instruction，output 是 edited image。它天然就是 I2I model，input 的 spatial structure 是 output 的强 prior。

**Hypothesis**：editor 应该是 dense geometry prediction 更好的 foundation，因为它 aligns with task 的 intrinsic structure。

---

## 二、为什么 editor 比 generator 好？

### 2.1 直觉解释

想象你要让一个 painter 画 depth map：
- **Generator painter**：平时画的是从文字描述生成新 image，他擅长从 nothing 创造 something
- **Editor painter**：平时做的是 "把这张图的人换到海边"，他擅长在 existing image structure 上做修改

Depth/normal estimation 其实是：看一张 image，然后在每个 pixel 上标注几何属性。这更像 editing task——输入 image 的 structure 是 output 的骨架。

### 2.2 实验证据

Authors 做了 controlled comparison：
- **FLUX**（T2I generator，DiT 架构）
- **Step1X-Edit**（image editor，基于 FLUX fine-tune，同样 DiT 架构）

把 FLUX 改成和 editor 一样的 input format（concat image + noise），然后用相同 data、相同 hyperparameter fine-tune 做 depth estimation。

**Figure 3 的 feature visualization**：
- Epoch 1 时，editor 的 intermediate features 已经 show 出 input image 的 geometric structure
- Generator 的 features 是混乱的、abstract 的
- Epoch 30 后，editor 是 "refinement"（从 good → better），generator 是 "reshaping"（from chaos → structure）

**Figure 4 的 training loss**：
- Editor 平滑收敛到 0.073
- Generator 卡在 0.08，有 oscillation

**Conclusion**：editor 的 inductive bias 给 fine-tune 提供了更好的 starting point，generator 需要从 scratch 学 I2I alignment，学不彻底。

---

## 三、三个 Technical Contribution

如果只是直接 fine-tune editor（叫 "DirectAdapt"），性能其实不行。因为 editor 原本是 generative task，和 deterministic depth prediction 有三个 mismatch，FE2E 针对 each 提出解决方案。

### 3.1 Mismatch 1: Stochastic vs Deterministic → Consistent Velocity

**问题**：Editor 用 flow matching loss，training 时从 random noise $\mathbf{z}_0 \sim \mathcal{N}(0, \mathbf{I})$ 出发，学一个 velocity field 把 noise transport 到 target latent $\mathbf{z}_1 = \mathcal{E}(\mathbf{y})$。Inference 时用 ODE solver 从 random noise 积分到 target。

但 depth prediction 是 deterministic task——给定 image 只有一个 GT depth。Random noise starting point + curved trajectory + discretized ODE solver = 累积 approximation error，对 high-precision depth 是灾难。

**FE2E 的 fix**：两步简化

**Step 1: Consistent Velocity**

原 flow matching loss：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{z}_1^y, \mathbf{z}_0^y} \| \mathbf{v} - f_\theta(\mathbf{z}^x, \mathbf{z}_t^y, t) \|^2$$

变量：
- $\mathbf{z}^x = \mathcal{E}(\mathbf{x})$：input image 的 VAE latent
- $\mathbf{z}_t^y = t\mathbf{z}_1^y + (1-t)\mathbf{z}_0^y$：flow path 上 t 时刻的 sample
- $f_\theta$：DiT，预测 velocity
- $\mathbf{v} = \mathbf{z}_1^y - \mathbf{z}_0^y$：target velocity

FE2E 改成：
$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_1^y, \mathbf{z}_0^y} \| \mathbf{v} - f_\theta(\mathbf{z}^x, \mathbf{z}_0^y) \|^2$$

注意 $f_\theta$ 的 input 去掉了 $\mathbf{z}_t^y$ 和 $t$——velocity 与时间无关，trajectory 是真正的直线。

**Step 2: Fixed Start**

既然 task 是 deterministic，random starting point 就是冗余的。设 $\mathbf{z}_0^y = \mathbf{0}$：

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_1^y} \| \mathbf{v} - f_\theta(\mathbf{z}^x) \|^2$$

**Inference 简化**：
$$\mathbf{z}_1^y = \mathbf{0} + \int_0^1 f_\theta(\mathbf{z}^x) dt = f_\theta(\mathbf{z}^x)$$

**这就是 single forward pass！** 模型直接 output target latent，再过 VAE decoder 拿 depth map。

这等价于把 generative flow matching 退化成 deterministic regression，但 training 时仍然用 flow matching loss（帮助 stable convergence），inference 时单步出结果。

参考：
- [Flow Matching paper](https://arxiv.org/abs/2210.02747)
- [MeanFlow (inspiration for consistent velocity)](https://arxiv.org/abs/2505.13481)

### 3.2 Mismatch 2: BF16 精度不够 → Logarithmic Quantization

**问题**：Step1X-Edit 是 BF16 训练的。BF16 在 [0.5, 1.0] 区间 worst-case precision 是 $1/256$。

对 RGB image（0-255 整数）完全够用，但对 depth 严重不够。

假设 depth range 是 [0, 80m]（Virtual KITTI dataset），要 normalize 到 [-1, 1] 让 VAE 接受：

**方案 A: Uniform quantization**
$$V = \frac{D}{40} - 1$$

$\Delta V = 1/256$ 对应 $\Delta D = 40 \times 1/256 \approx 0.16$ m

- 80m 处：error 16cm，AbsRel = 0.002（good）
- 0.1m 处：error 16cm，AbsRel = 1.6（catastrophic）

近距离完全崩坏。

**方案 B: Inverse quantization**（Marigold 用 disparity = 1/depth）

- 0.1m 处：error 0.2mm（excellent）
- 80m 处：error 125m（catastrophic）

远距离完全崩坏——39m 和 78m 的 disparity 差小于 quantization step，变得 indistinguishable。

**方案 C: Logarithmic quantization**（FE2E）

$D_{log} = \ln(D)$, 然后 normalize 到 [-1, 1]。

关键 insight：在 log space，quantization step 是恒定的 $\Delta D_{log} \approx 0.013$，对应 real depth error $\Delta D \approx D \cdot \Delta D_{log}$，所以 **AbsRel = $\Delta D / D \approx 0.013$ 全程恒定**。

- 80m 处：error 1.04m，AbsRel 0.013
- 0.1m 处：error 1.3mm，AbsRel 0.013

**完美 balanced**——近处远处相对误差一致。

完整公式（Eq. 7）：
$$\mathbf{y}_D = \left\langle \frac{D_{log} - D_{log,2}}{D_{log,98} - D_{log,2}} - 0.5 \right\rangle \times 2$$

变量：
- $D_{log} = \ln(D_{GT} + 10^{-6})$：log depth（$10^{-6}$ 防止 log(0)）
- $D_{log, i}$：第 i 百分位（用 2% 和 98% 过滤 outliers）
- $\langle \cdot \rangle$：BF16 truncation
- 减 0.5 乘 2：[0,1] → [-1,1]

**为什么不用 FP32？**
1. 训练成本高
2. **Prior 继承差**：Step1X-Edit 的 weights 是 BF16 训练的，强行 FP32 fine-tune 会让模型偏离原始 weight manifold，损失 prior
3. 无法 fine-tune BF16-only model

### 3.3 Mismatch 3: DiT 输出有浪费 → Cost-Free Joint Estimation

**问题**：Step1X-Edit 的 DiT 用 horizontal concat 注入 condition：

$$\mathbf{z}^{x+\Theta} = \text{concat}(\mathbf{z}^x, \mathbf{z}^\Theta) \in \mathbb{R}^{h \times 2w \times c}$$

左半边是 input image latent，右半边是 noise latent。DiT 处理后输出 same shape $[p_l, p_r]$。

**但原 editor 只监督 $p_r$（noise 区域）**，$p_l$ 计算了却丢弃——50% 计算浪费。

**FE2E 的 fix**：用 $p_l$ 监督另一个 task！

$$\mathcal{L}_{fm} = \mathbb{E}_{\mathbf{z}_1^y} \left( \| \mathbf{v}_D - p_l \|^2 + \| \mathbf{v}_N - p_r \|^2 \right)$$

- $p_l$ 监督 depth
- $p_r$ 监督 normal
- **零额外 cost**：训练 forward 一样，inference 也是 single forward pass 同时出 depth 和 normal

**为什么 joint training 有性能增益？** DiT 的 global attention 让 depth 和 normal features 在 attention layers 隐式 exchange information。Depth gradient 和 surface tangent 本来就几何耦合，joint supervision 帮助 model 在 challenging regions resolve ambiguity。

Figure 8 显示 joint training 在 flat butterfly structures 和 distant buildings 上有显著提升。

参考：
- [DiT architecture](https://arxiv.org/abs/2212.09748)
- [GeoWizard (previous joint depth+normal)](https://arxiv.org/abs/2403.12013)

---

## 四、实验结果到底有多强？

### 4.1 Zero-shot Depth（Table 2）

| Method | Training Data | ETH3D AbsRel | KITTI AbsRel | Avg Rank |
|--------|--------------|--------------|--------------|----------|
| DepthAnything V2 | 62.6M | 13.1 | 7.4 | 5.4 |
| DepthAnything V1 | 62.6M | 12.7 | 7.6 | 3.5 |
| Lotus-D | 59K | 6.1 | 8.1 | 3.7 |
| Marigold v1.1 | 77K | 7.0 | 11.0 | 2.0 |
| **FE2E** | **71K** | **3.8** | **6.6** | **1.4** |

**亮点**：
1. ETH3D 上 AbsRel 3.8，比第二名 Lotus-D 的 6.1 提升 35%——巨大 gap
2. 只用 71K data 击败用 62.6M 训练的 DepthAnything 系列
3. ETH3D 是各种 scene 混合（indoor/outdoor/various），说明 generalization 极强

### 4.2 Zero-shot Normal（Table 3）

| Method | ScanNet MeanErr | iBims-1 MeanErr | Sintel MeanErr |
|--------|----------------|-----------------|----------------|
| DSINE | 16.2 | 17.1 | 34.9 |
| Lotus-D | 14.7 | 17.1 | 32.3 |
| Marigold v1.1 | 14.5 | 16.3 | — |
| **FE2E** | **13.8** | **15.1** | **31.2** |

### 4.3 Ablation Study（Table 4）

逐项 ablation 在 KITTI/ETH3D AbsRel 上：

| ID | Setting | KITTI | ETH3D |
|----|---------|-------|-------|
| 1 | FLUX + DirectAdapt | 9.2 | 6.0 |
| 2 | Step1X-Edit + DirectAdapt | 9.5 | 5.6 |
| 3 | + Consistent Velocity | 8.8 | 5.0 |
| 4 | + Fixed Start | 8.6 | 4.8 |
| 5 | + Inverse Quant | 6.9 | 4.6 |
| 6 | + Log Quant | 6.8 | 4.5 |
| 7 | FLUX + all improvements | 7.1 | 3.9 |
| 8 | **FE2E (full)** | **6.6** | **3.8** |
| 9 | FLUX-Kontext (other editor) | 6.7 | 3.6 |

**关键观察**：
- ID2 → ID4：consistent velocity + fixed start 带来 ETH3D 5.6 → 4.8
- ID4 → ID6：log quant 带来 ETH3D 4.8 → 4.5
- ID6 → ID8：joint estimation 额外提升 ETH3D 4.5 → 3.8
- ID7 (FLUX + all) vs ID8 (Step1X + all)：3.9 vs 3.8，editor 优势在加满所有 improvement 后依然存在
- **ID9 FLUX-Kontext 比 Step1X-Edit 还略好**，证明 paradigm 对其他 editor 可扩展

### 4.4 与 Unified Models 对比（Table 6）

| Method | Training Data | ETH3D AbsRel | Avg Rank |
|--------|--------------|--------------|----------|
| Qwen-Image | billions | 6.6 | 2.6 |
| DINOv3 | 1.7B | 5.4 | 1.8 |
| **FE2E** | 71K | **3.8** | **1.6** |

即使 DINOv3 用 1.7B images 训练，FE2E 用 71K 还是赢——**editor prior > data scaling**。

---

## 五、Implementation 细节

### 5.1 Training Setup

- **Backbone**：Step1X-Edit v1.0（DiT-based，based on FLUX）
- **Frozen**：所有参数除 DiT module
- **LoRA**：rank=64, scale $\alpha=32$（参数 efficient fine-tune）
- **Epochs**：30
- **Optimizer**：AdamW, lr=$10^{-4}$
- **Hardware**：单卡 RTX 4090 可（with gradient checkpointing），实际用 NVIDIA H20，1.5 天训完
- **Auxiliary dispersion loss**（Appendix A.1）：

$$\mathcal{L}_{disp} = \log \mathbb{E}_{i,j} \left[ \exp(-\|\eta_i - \eta_j\|_2^2 / \tau) \right]$$

来自 [Diffuse and Disperse](https://arxiv.org/abs/2506.20701)，鼓励 batch 内不同 sample 的 features 在 hidden space 中 spread out，提升 representation 能力。Total loss = $\mathcal{L}_{fm} + 0.5 \mathcal{L}_{disp}$。

### 5.2 Training Data

- **Hypersim**（indoor photorealistic）：51k images @ 1024×768，过滤 invalid pixel > 1%
- **Virtual KITTI**（outdoor synthetic driving）：20k images @ 1216×352，4 个 scenarios，max depth 80m
- **Sampling ratio**：90% Hypersim + 10% Virtual KITTI（follow Marigold）

### 5.3 Inference Cost（Table 7）

| Method | MACs | RunTime | ETH3D AbsRel |
|--------|------|---------|--------------|
| Marigold | 133T | 9.67s | 6.5 |
| Lotus-D | 2.65T | 212ms | 6.1 |
| Qwen-Image | 2.13P | 63.4s | 6.6 |
| DINOv3 | 14.5T | 632ms | 5.4 |
| **FE2E** | **28.9T** | **1.78s** | **3.8** |

FE2E 比 Lotus-D 慢 8 倍但性能好 60%；比 Qwen-Image 快 35 倍且性能好 75%。

---

## 六、Intuition Building

### 6.1 这篇 paper 的 mental model

**传统 view**：dense prediction = task-specific model + big data

**FE2E view**：dense prediction = task-aligned foundation + minimal data + surgical adaptation

关键不是 model 多大或 data 多多，关键是 foundation 的 inductive bias 是否 match task 的 structure。

### 6.2 三个 trick 为什么 work

1. **Consistent Velocity + Fixed Start**：把 stochastic flow matching 退化成 deterministic regression，eliminate ODE solver error，同时保留 editor 的 prior。这其实是 consistency model 的精神，但通过 deterministic task 的特殊性实现了更彻底的简化。

2. **Logarithmic Quantization**：人 perceive depth 本来就是 log scale（Weber-Fechner law）。Log space 中相对误差恒定对应人眼的 depth discrimination 能力（远处分辨几米，近处分辨几毫米）。这个 trick 可以 borrow 到任何 BF16 model 做 dense depth 的场景。

3. **Cost-Free Joint Estimation**：DiT 的 concat input design 本身就有 50% 算力浪费，repurpose 成 joint task supervision 是 "free lunch"。而且 depth 和 normal 在几何上本来就耦合（normal 是 depth gradient 的方向），joint supervision 让 attention 机制 help 两个 task 互相 resolve ambiguity。

### 6.3 为什么这个 paradigm 重要

1. **Data efficiency**：71K vs 62.6M，三个数量级的差距。这 matter 在很多 domain（medical imaging, scientific imaging）data 稀缺的场景。

2. **Prior inheritance**：editor 在海量 I2I data 上训练过，已经 "理解" image structure。Fine-tune 只需 refine 而不是 reshape，更 data efficient。

3. **Generalization**：FE2E 在 ETH3D（各种 scene）上 generalization 最好，说明 editor prior 比 task-specific data 更 robust。

### 6.4 Potential Applications

1. **其他 dense prediction task**：optical flow, disparity, semantic segmentation 都是 I2I task，可以试 editor prior
2. **Data-scarce domain**：medical depth, scientific imaging data 少，editor prior route 更可行
3. **Real-time AR/VR**：single forward pass 的 inference 适合 latency-sensitive 场景

### 6.5 Open Questions

1. **Editor 选择的影响**：Step1X-Edit vs FLUX-Kontext vs 其他 editor，prior 质量如何量化？
2. **Scaling behavior**：如果真的 scale up data 到 millions，editor prior 的 advantage 会保持还是消失？
3. **Task 范围**：editor prior 对哪些 I2I task 有效，对哪些无效？比如 image restoration 这种 ill-posed task 是否也适合？

---

## 七、参考资源

### Core papers
- [FE2E project page (待更新)](https://github.com/) 
- [Step1X-Edit](https://arxiv.org/abs/2504.17761)
- [FLUX.1 by Black Forest Labs](https://blackforestlabs.ai/)
- [FLUX-Kontext](https://arxiv.org/abs/2506.15742)

### Flow Matching & Diffusion
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [MeanFlow](https://arxiv.org/abs/2505.13481)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Diffusion Transformers (DiT)](https://arxiv.org/abs/2212.09748)

### Dense Geometry Estimation
- [Marigold](https://arxiv.org/abs/2312.02145)
- [Lotus](https://arxiv.org/abs/2409.18124)
- [DepthAnything V2](https://arxiv.org/abs/2406.09414)
- [GeoWizard](https://arxiv.org/abs/2403.12013)
- [Diffusion-E2E-FT](https://arxiv.org/abs/2412.07561)
- [GenPercept](https://arxiv.org/abs/2403.06090)
- [StableNormal](https://arxiv.org/abs/2406.16864)

### Datasets & Benchmarks
- [Hypersim](https://github.com/apple/ml-hypersim)
- [Virtual KITTI 2](https://arxiv.org/abs/2001.10773)
- [ETH3D](https://www.eth3d.net/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/)

### Technical Background
- [BFloat16 format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Diffuse and Disperse](https://arxiv.org/abs/2506.20701)
- [InstructPix2Pix](https://arxiv.org/abs/2211.09800)

---

希望这个用人话的版本帮你看清了 paper 的 forest，又不丢 trees。核心 story 就是：**用对 foundation (editor) + surgical adaptation (三个 trick) = 用 0.1% data 打败 data-driven SOTA**。三个 trick 中 log quantization 和 cost-free joint estimation 我觉得直接 borrow 到其他 dense prediction 工作也能 work。

---

# FE2E: From Editor to Dense Geometry Estimator 深度解析

Karpathy 你好，这篇 paper 非常有意思，它从一个很务实的角度切入——dense geometry prediction (depth, normal estimation) 本质上是 image-to-image (I2I) task，那 T2I generative model 凭什么是最好的 foundation？这个直觉很关键，下面我把每个技术细节都展开讲，帮你 build intuition。

## 一、核心 motivation：为什么 editor > generator？

### 1.1 任务对齐性

Dense geometry prediction 的输入是 image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$，输出是 pixel-aligned 的 depth/normal $\mathbf{y} \in \math 作者 R^{H \times W \times 3}$，**输入输出空间共享 spatial grid**，这是 I2I task 的本质特征。

- T2I generator (FLUX, SD)：从 noise + text condition → 全新 image，**输入 image 不参与 output 的 spatial alignment**
- Image editor (Step1X-Edit, InstructPix2Pix)：从 source image + edit instruction → modified image，**输入 image 的 spatial structure 是 output 的强 prior**

### 1.2 实验证据（Figure 3, 4）

Authors 做了一个非常 surgical 的对照实验：FLUX (generator) vs Step1X-Edit (editor)，两者都是 DiT 架构，author 把 FLUX 改成 same input structure (concat noise + RGB) 来消除 confounding factor。

**Feature evolution 对比（30 epochs fine-tune）**：
- Epoch 1（initial）：
  - Editor Block1 features 已经 align with input image 的 geometric structure
  - Generator Block1 features 是 abstract/unstructured
- Epoch 30：
  - Editor features 是 "refinement"——从 well-structured → clearer, task-oriented
  - Generator features 是 "reshaping"——从 chaotic → highly structured，有 qualitative leap
- Training loss convergence：
  - Editor 稳定收敛到 0.073
  - Generator 卡在 0.08 bottleneck，且有 oscillation

这个 insight 非常重要：**editor 的 inductive bias 提供了更好的 starting point，让 fine-tune 只需 refine，而 generator 需要从 scratch 学习 I2I alignment**。

参考链接：
- [Step1X-Edit paper](https://arxiv.org/abs/2504.17761)
- [FLUX.1 by Black Forest Labs](https://blackforestlabs.ai/)

---

## 二、技术贡献 1：Consistent Velocity Flow Matching

### 2.1 Flow Matching 背景

Flow Matching [Lipman et al., 2022] 是 rectified flow 的 generalization，核心是学一个 vector field $v_t(\mathbf{z})$ 把 prior $p_0 = \mathcal{N}(0, \mathbf{I})$ transport 到 data distribution $p_1$。

**Rectified Flow** 路径定义：
$$\mathbf{z}_t = t \mathbf{z}_1 + (1-t) \mathbf{z}_0, \quad t \in [0, 1]$$

变量含义：
- $\mathbf{z}_0 \sim \mathcal{N}(0, \mathbf{I})$：起点 noise
- $\mathbf{z}_1 = \mathcal{E}(\mathbf{y})$：target latent (VAE encoded depth/normal)
- $t$：时间变量，0 到 1
- 路径是 $\mathbf{z}_0$ 到 $\mathbf{z}_1$ 的**直线**

对应 velocity（路径对 t 的导数）：
$$\mathbf{v} = \frac{d\mathbf{z}_t}{dt} = \mathbf{z}_1 - \mathbf{z}_0$$

**条件 velocity** 是常数（与 t 无关），但 marginal velocity field（在所有 $\mathbf{z}_0, \mathbf{z}_1$ pair 上求期望）是非线性的，因为不同的 pair 起点终点都不同。

### 2.2 原始 EditorAdapt 的 Loss（Section 3.2）

把编辑任务的 source image $\mathbf{x}$ 作为 condition，EditorAdapt 的 flow matching loss：

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{z}_1^y, \mathbf{z}_0^y} \| \mathbf{v} - f_\theta(\mathbf{z}^x, \mathbf{z}_t^y, t) \|^2$$

变量：
- $\mathbf{z}^x = \mathcal{E}(\mathbf{x})$：input image 的 VAE latent
- $\mathbf{z}_t^y$：t 时刻的 flow path sample
- $f_\theta$：DiT backbone，预测 velocity
- $\mathbf{v} = \mathbf{z}_1^y - \mathbf{z}_0^y$：ground truth velocity

**推理**（Eq. 3）：通过 ODE solver 从 $\mathbf{z}_0^y \sim \mathcal{N}(0, \mathbf{I})$ 积分到 $\mathbf{z}_1^y$：
$$\hat{\mathbf{z}}_1^y = \mathbf{z}_0^y + \int_0^1 f_\theta(\mathbf{z}^x, \mathbf{z}_t^y, t) dt$$

### 2.3 问题：为什么这样不行？

MeanFlow [Geng et al., 2025] 的 insight：因为 $f_\theta$ 学的是 marginal velocity field（在所有 $\mathbf{z}_0^y$ 上期望），**全局瞬时 velocity 是 non-linear 的，trajectory 是弯曲的**（Figure 5b）。

离散数值 solver（如 Euler）approximate 弯曲 trajectory 时引入 approximation error，对于 dense geometry 这种 high precision task 是致命的。

### 2.4 FE2E 的 Reformulation

**第一步：Consistent Velocity**——让 velocity 的 magnitude 也与 t 无关：

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_1^y, \mathbf{z}_0^y} \| \mathbf{v} - f_\theta(\mathbf{z}^x, \mathbf{z}_0^y) \|^2$$

注意 $f_\theta$ 的输入**去掉了 $\mathbf{z}_t^y$ 和 $t$**，velocity 完全由 $\mathbf{z}^x$ 和起点 $\mathbf{z}_0^y$ 决定。这意味着模型预测的 velocity 是常数，不随 t 变化，trajectory 是真正的直线。

**第二步：Fixed Departure**——对于 deterministic task，random starting point 是冗余的：

$$\mathbf{z}_0^y = \mathbf{0}$$

把对 $\mathbf{z}_0^y \sim \mathcal{N}(0, \mathbf{I})$ 的 expectation 简化为固定 $\mathbf{0}$，loss 进一步：

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_1^y} \| \mathbf{v} - f_\theta(\mathbf{z}^x) \|^2$$

**推理**（Eq. 6）退化成 single forward pass：
$$\mathbf{z}_1^y = \mathbf{0} + \int_0^1 f_\theta(\mathbf{z}^x) dt = (1-0) f_\theta(\mathbf{z}^x) = f_\theta(\mathbf{z}^x)$$

**这是一个非常 elegant 的简化**：本质上把 generative flow matching 退化成了一个 deterministic regression。但保留了 editor backbone 的 prior，并且 training 时仍然在 latent space 用 flow matching loss（稳定收敛）。

参考：
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Mean Flows for One-Step Generative Modeling](https://arxiv.org/abs/2505.13481)

---

## 三、技术贡献 2：Logarithmic Quantization

### 3.1 BF16 精度问题

这是 paper 中最容易被忽视但极其重要的细节。Step1X-Edit (基于 FLUX) 用 BF16 训练，BF16 的 normalized value 表示：

$$V = (-1)^S \times 2^{(E-127)} \times (1.F)_2$$

变量：
- $S$：sign bit (1 bit)
- $E$：exponent (8 bits, bias 127)
- $F$：fraction (7 bits)

在 [-1, 1] 区间内（VAE 强制输入范围），worst-case precision 发生在 $[0.5, 1.0]$：
$$\Delta V = 2^{126-127} \times 2^{-7} = 1/256 \approx 0.0039$$

对于 RGB (0-255) 这 1/256 精度完全够用，因为 RGB 离散化本来就是 0-255 整数。

### 3.2 三种 Quantization 方案对比（Table 1）

以 Virtual KITTI 数据集 depth range [0, 80m] 为例：

**方案 (a) Uniform Quantization**：
$$V = 2 \times \frac{D - 0}{80 - 0} - 1 = \frac{D}{40} - 1$$

BF16 量化步长 $\Delta V = 1/256$ 对应真实 depth error：
$$\Delta D = 40 \times \Delta V \approx 0.156 \text{m}$$

这是**恒定的绝对误差**：
- 80m 处：error = 16cm, AbsRel = 0.002（excellent）
- 0.1m 处：error = 16cm, AbsRel = 1.600（catastrophic）

近距离完全崩坏。

**方案 (b) Inverse Quantization**（Marigold 等用 disparity $P = 1/D$）：
- $P$ range: $[1/80, 1/0.1] = [0.0125, 10]$
- $\Delta P \approx 0.0195$（恒定）
- Depth error: $\Delta D \approx D^2 \Delta P$

- 0.1m：error = 0.2mm, AbsRel = 0.002（excellent）
- 80m：error = 125m, AbsRel = 1.563（catastrophic）

远距离完全崩坏——**39m 和 78m 的 disparity 差（0.0128）小于 quantization step（0.0195），变得 indistinguishable**。

**方案 (c) Logarithmic Quantization**（FE2E 采用）：
- $D_{log} = \ln(D)$, range $[\ln(0.1), \ln(80)] \approx [-2.30, 4.38]$
- $\Delta D_{log} \approx 0.013$（恒定）
- Depth error: $\Delta D \approx D \cdot \Delta D_{log}$
- AbsRel = $\Delta D / D \approx 0.013$（**全程恒定**）

- 80m：error = 1.04m, AbsRel = 0.013
- 0.1m：error = 1.3mm, AbsRel = 0.013

**这是最 balanced 的方案**，log space 中相对误差恒定，符合人对 depth 的 perception（Weber-Fechner law 启发）。

### 3.3 完整 normalization 公式（Eq. 7）

$$\mathbf{y}_D = \left\langle \frac{D_{log} - D_{log,2}}{D_{log,98} - D_{log,2}} - 0.5 \right\rangle \times 2$$

变量：
- $D_{log} = \ln(D_{GT} + 1\times10^{-6})$：log depth
- $D_{log, i}$：第 i 百分位的 $D_{log}$ 值（用 2% 和 98% percentile 是为了 robustness，过滤 outliers）
- $\langle \cdot \rangle$：BF16 precision truncation operator
- $-0.5$ 再 $\times 2$：把 $[0, 1]$ 映射到 $[-1, 1]$（VAE 输入范围）

**为什么不用 FP32？** 因为：
1. 训练成本高
2. **prior 继承差**——Step1X-Edit 是 BF16 训练的，强行 FP32 fine-tune 会让模型偏离原始 weight manifold
3. 限制 BF16-only 模型的能力发挥

参考：
- [Marigold normalization](https://arxiv.org/abs/2312.02145)
- [BFloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

---

## 四、技术贡献 3：Cost-Free Joint Estimation

### 4.1 DiT 输入结构

Step1X-Edit 等 DiT-based editor 用 **horizontal concatenation** 注入 condition：

$$\mathbf{z}^{x+\Theta} = \text{concat}(\mathbf{z}^x, \mathbf{z}^\Theta) \in \mathbb{R}^{h \times 2w \times c}$$

变量：
- $\mathbf{z}^x$：input image latent（左半边）
- $\mathbf{z}^\Theta$：noise latent（右半边）
- $h, w, c$：latent spatial 维度 + channels

### 4.2 浪费的算力

DiT 处理后输出 shape 与输入相同：
$$f_\theta(\mathbf{z}^{x+\Theta}) = [p_l, p_r] \in \mathbb{R}^{h \times 2w \times c}$$

- $p_l \in \mathbb{R}^{h \times w \times c}$：左半边，对应 $\mathbf{z}^x$ 区域
- $p_r \in \mathbb{R}^{h \times w \times c}$：右半边，对应 $\mathbf{z}^\Theta$ 区域

**原 editor 只监督 $p_r$**（noise 区域），$p_l$ 被计算后丢弃——**50% 计算浪费**。

### 4.3 FE2E 的 repurpose

利用这个 "free" 的 $p_l$ 区域做第二个 task：

$$\mathcal{L}_{fm} = \mathbb{E}_{\mathbf{z}_1^y} \left( \| \mathbf{v}_D - p_l \|^2 + \| \mathbf{v}_N - p_r \|^2 \right)$$

- $\mathbf{v}_D$：depth 的 velocity target
- $\mathbf{v}_N$：normal 的 velocity target
- $p_l$ 监督 depth，$p_r$ 监督 normal

**零额外 cost**：训练时 forward pass 一样，loss 项加一个；推理时一次 forward 同时出 depth 和 normal。

### 4.4 为什么 joint training 有性能增益？

Authors 假设是 DiT 的 global attention 允许 depth 和 normal features 在 attention layers 隐式 exchange information。深度和法向本来就有几何耦合（depth gradient 对应 surface tangent），joint supervision 帮助 model 在 challenging regions（如 flat surfaces, distant buildings）resolve ambiguity。

Figure 8 的 qualitative 比较显示 joint training 在 butterfly structures 和 distant buildings 上有显著改善。

参考：
- [GeoWizard (joint depth+normal)](https://arxiv.org/abs/2403.12013)
- [Diffusion Transformers (DiT)](https://arxiv.org/abs/2212.09748)

---

## 五、架构图解析（Figure 2）

整个 pipeline：

```
┌─────────────────────────────────────────────────────────┐
│  Training                                                  │
│                                                           │
│  x (image) ──┐                                            │
│              ├─→ VAE Encoder E(·) ──→ z^x                │
│  GT depth d ─┤                                            │
│              ├─→ Log Quantize ─→ VAE E(·) ─→ z_1^y(D)    │
│  GT normal n ┘                                            │
│                                                           │
│  z_0^y = 0 (fixed)                                        │
│                                                           │
│  z^x, z_0^y ──→ concat ──→ DiT f_θ                        │
│                              │                            │
│                          [p_l, p_r]                       │
│                              │                            │
│            ┌─────────────────┴─────────────────┐          │
│            ↓                                   ↓          │
│       Loss on p_l                          Loss on p_r    │
│       (depth velocity)                  (normal velocity) │
│            ‖v_D - p_l‖²                  ‖v_N - p_r‖²    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Inference (single forward pass)                          │
│                                                           │
│  x ──→ VAE E(·) ──→ z^x                                   │
│                                                           │
│  z^x, z_0^y=0 ──→ concat ──→ DiT f_θ ──→ [p_l, p_r]      │
│                                                    │      │
│                              ┌─────────────────────┘      │
│                              ↓                            │
│              p_l (depth latent) → VAE D(·) → depth ẑ_D    │
│              p_r (normal latent) → VAE D(·) → normal ẑ_N  │
└─────────────────────────────────────────────────────────┘
```

关键点：
1. **Training 时**：用 flow matching loss 在 latent space 监督，velocity target 是 $\mathbf{z}_1^y - \mathbf{z}_0^y = \mathbf{z}_1^y - 0 = \mathbf{z}_1^y$
2. **Inference 时**：根据 Eq. 6，$\mathbf{z}_1^y = f_\theta(\mathbf{z}^x)$，单次 forward 直接拿到 latent，再过 VAE decoder

---

## 六、实验结果深度分析

### 6.1 Zero-shot Depth Estimation（Table 2）

FE2E 在 5 个 benchmark 上 average rank = 1.4（最低）：

| Method | Training Data | ETH3D AbsRel | KITTI AbsRel | Avg Rank |
|--------|--------------|--------------|--------------|----------|
| DepthAnything V2 | 62.6M | 13.1 | 7.4 | 5.4 |
| DepthAnything V1 | 62.6M | 12.7 | 7.6 | 3.5 |
| Lotus-D | 59K | 6.1 | 8.1 | 3.7 |
| Marigold v1.1 | 77K | 7.0 | 11.0 | 2.0 |
| Diffusion-E2E-FT | 74K | 6.4 | 9.6 | 4.6 |
| **FE2E** | **71K** | **3.8** | **6.6** | **1.4** |

**ETH3D 上 35% AbsRel 提升（vs second best 5.9 Lotus-D → 3.8）**，这是巨大 gap。ETH3D 是各种 scene types 混合（indoor/outdoor/various），说明 FE2E 的 generalization 能力极强。

**仅用 71K 训练数据就超越用 62.6M 训练的 DepthAnything 系列**——这验证了 paper 的核心 thesis：foundation prior 比 data scaling 更重要。

### 6.2 Zero-shot Normal Estimation（Table 3）

FE2E average rank = 1.6：

| Method | ScanNet MeanErr | iBims-1 MeanErr | Sintel MeanErr |
|--------|----------------|-----------------|----------------|
| DSINE | 16.2 | 17.1 | 34.9 |
| Lotus-D | 14.7 | 17.1 | 32.3 |
| Diffusion-E2E-FT | 14.7 | 16.1 | 33.5 |
| Marigold v1.1 | 14.5 | 16.3 | — |
| **FE2E** | **13.8** | **15.1** | **31.2** |

### 6.3 Ablation Study（Table 4）

逐项 ablation，KITTI/ETH3D AbsRel：

| ID | Setting | KITTI | ETH3D |
|----|---------|-------|-------|
| 1 | FLUX + DirectAdapt | 9.2 | 6.0 |
| 2 | Step1X-Edit + DirectAdapt | 9.5 | 5.6 |
| 3 | + Consistent Velocity | 8.8 | 5.0 |
| 4 | + Fixed Start | 8.6 | 4.8 |
| 5 | + Inverse Quant | 6.9 | 4.6 |
| 6 | + Log Quant | 6.8 | 4.5 |
| 7 | FLUX + all improvements | 7.1 | 3.9 |
| 8 | **FE2E (full)** | **6.6** | **3.8** |
| 9 | FLUX-Kontext (extension) | 6.7 | 3.6 |

**关键 observation**：
- ID1 vs ID2：editor vs generator 直接对比，editor 略好（ETH3D 5.6 vs 6.0）
- ID2 → ID4：consistent velocity + fixed start 带来 ETH3D 从 5.6 → 4.8（14% 提升）
- ID4 → ID6：logarithmic quantization 带来 ETH3D 从 4.8 → 4.5（quantization 影响最大）
- ID6 → ID8：joint estimation 额外提升 ETH3D 4.5 → 3.8（16%）
- ID7 (FLUX + all improvements) vs ID8 (Step1X-Edit + all)：ETH3D 3.9 vs 3.8，editor 优势在加满所有 improvement 后依然存在但缩小——说明 editor 的优势主要在 prior 层面
- ID9 FLUX-Kontext（concurrent editor）甚至比 Step1X-Edit 略好，证明**这个 paradigm 对其他 editor 也可扩展**

### 6.4 与 Concurrent Unified Models 对比（Table 6）

| Method | Training Data | ETH3D AbsRel | Avg Rank |
|--------|--------------|--------------|----------|
| Qwen-Image | billions | 6.6 | 2.6 |
| DINOv3 | 1.7B | 5.4 | 1.8 |
| **FE2E** | 71K | **3.8** | **1.6** |

即便 DINOv3 用了 1.7B 图片训练，FE2E 用 71K 还是赢了——这进一步强化了 "editor prior > data scaling" 的 thesis。

---

## 七、Implementation Details

### 7.1 Training Setup

- **Backbone**：Step1X-Edit v1.0 (DiT-based)
- **Frozen**：所有参数除 DiT module
- **LoRA**：rank=64, scale $\alpha=32$
- **Epochs**：30
- **Optimizer**：AdamW, lr=$1 \times 10^{-4}$
- **GPU**：单卡 RTX 4090 可（with gradient checkpointing），实际用 NVIDIA H20，1.5 天训完
- **Auxiliary dispersion loss**（Appendix A.1）：

$$\mathcal{L}_{disp} = \log \mathbb{E}_{i,j} \left[ \exp(-\|\eta_i - \eta_j\|_2^2 / \tau) \right]$$

- $\eta_i, \eta_j$：batch 中第 i, j 个 sample 在第 9 个 block 的 output feature
- $\tau = 1$：temperature
- $\lambda = 0.5$：loss weight
- 来自 [Diffuse and Disperse](https://arxiv.org/abs/2506.20701)，鼓励 feature 在 hidden space 中 spread out

### 7.2 Training Data

- **Hypersim**（indoor）：51k images @ 1024×768，过滤掉 invalid pixel > 1% 的 sample
- **Virtual KITTI**（outdoor）：20k images @ 1216×352，4 个 driving scenarios，max depth 80m
- **Sampling ratio**：Hypersim 90% + Virtual KITTI 10%（follow Marigold）

### 7.3 Inference Cost（Table 7）

| Method | MACs | RunTime | ETH3D AbsRel |
|--------|------|---------|--------------|
| Marigold | 133T | 9.67s | 6.5 |
| Lotus-D | 2.65T | 212ms | 6.1 |
| Qwen-Image | 2.13P | 63.4s | 6.6 |
| DINOv3 | 14.5T | 632ms | 5.4 |
| **FE2E** | **28.9T** | **1.78s** | **3.8** |

FE2E 比 Lotus-D 慢 8 倍但性能更好；比 Qwen-Image 快 35 倍且性能更好。这是合理的 trade-off。

---

## 八、Intuition 总结

### 8.1 这篇 paper 的真正 contribution

1. **Conceptual**：把 dense geometry prediction 从 "T2I prior fine-tune" paradigm 转向 "I2I editor prior fine-tune" paradigm
2. **Technical**：解决 editor 适配 deterministic task 的三个 mismatch：
   - Stochastic ↔ Deterministic → Consistent Velocity + Fixed Start
   - BF16 precision ↔ High precision demand → Logarithmic Quantization
   - Single task ↔ Multi-task efficiency → Cost-Free Joint Estimation
3. **Empirical**：用 0.1% 的 data 击败 data-driven SOTA，证明 prior inheritance 的威力

### 8.2 与其他工作的关系

- vs **Marigold**：Marigold 用 SD prior，FE2E 用 editor prior；Marigold 需多步 denoising，FE2E 单步
- vs **Lotus**：Lotus 也在 SD 上做 end-to-end simplification，FE2E 把这思路迁移到 editor + DiT 上
- vs **DepthAnything**：DepthAnything 走 data scaling 路线，FE2E 走 prior inheritance 路线
- vs **GeoWizard**：GeoWizard 用 cross-attention 做 joint depth+normal，FE2E 利用 DiT global attention 的 free region，无需额外模块

### 8.3 Limitations & Future Work（Appendix G）

- **Computational load**：28.9T MACs 比 Lotus-D 重 10 倍
- **Foundation diversification**：作者承认应测试更多 editor 模型（FLUX-Kontext 已验证）
- **Data scaling**：当前只用了 71K，scaling up 可能进一步提升性能

### 8.4 一些值得深究的点

1. **为什么 editor prior 这么强？** Editor 训练时本身就接触了大量 I2I pair（image + edited image），学到的 feature 已经是 image-aligned representation；而 T2I generator 只见过 noise→image，对 image→image 的 mapping 没有 inductive bias。
2. **Log quantization 的通用性**：这个 trick 实际上适用于任何 BF16-based model 做 depth estimation 的场景，值得在其他 dense prediction task (如 optical flow, disparity) 上验证。
3. **Consistent Velocity 的理论联系**：这其实和 consistency models [Song et al.] 有精神上的相似——都是把 multi-step 生成简化成 single-step，但 consistency model 是通过 self-consistency constraint，FE2E 是直接利用 deterministic nature 把 velocity 设为 t-independent。

---

## 九、相关参考链接

- [FE2E arXiv paper](https://arxiv.org/abs/2507.07415)（注：实际 paper ID 需查证，附件未明示）
- [Step1X-Edit](https://arxiv.org/abs/2504.17761)
- [FLUX.1](https://blackforestlabs.ai/)
- [FLUX-Kontext](https://arxiv.org/abs/2506.15742)
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [Marigold](https://arxiv.org/abs/2312.02145)
- [Lotus](https://arxiv.org/abs/2409.18124)
- [DepthAnything V2](https://arxiv.org/abs/2406.09414)
- [Diffusion Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [MeanFlow](https://arxiv.org/abs/2505.13481)
- [Hypersim dataset](https://github.com/apple/ml-hypersim)
- [Virtual KITTI 2](https://arxiv.org/abs/2001.10773)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Diffuse and Disperse](https://arxiv.org/abs/2506.20701)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [ETH3D benchmark](https://www.eth3d.net/)
- [KITTI benchmark](http://www.cvlibs.net/datasets/kitti/)

---

希望这个 detailed breakdown 帮你 build 了 intuition，Karpathy。这篇 paper 真正 contribution 不在于某个 SOTA number，而在于它重新 frame 了 dense prediction 的 foundation 选择问题——**用 image-to-image prior 取代 text-to-image prior 是 natural fit**。技术上三个 contribution 都很 surgical，但 logarithmic quantization 和 cost-free joint estimation 这两个 trick 我觉得可以直接 borrow 到其他 dense prediction 工作中。
