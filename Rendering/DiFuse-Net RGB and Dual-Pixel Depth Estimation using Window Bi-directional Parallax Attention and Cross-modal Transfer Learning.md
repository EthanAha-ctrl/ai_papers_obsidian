---
source_pdf: DiFuse-Net RGB and Dual-Pixel Depth Estimation using Window Bi-directional
  Parallax Attention and Cross-modal Transfer Learning.pdf
paper_sha256: ff0c85b1bde3cc2cf017deed6ddc16066237cd290dd2280b5076baf224d8ee57
processed_at: '2026-08-03T21:59:30-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiFuse-Net 人话版

---

## 一句话先说清楚

手机里那个帮你拍照自动对焦的 dual-pixel 传感器，其实偷偷藏了一对"迷你 stereo camera"。这篇 paper 就是教神经网络怎么从这对迷你 stereo 里抠出 depth 来。难点就一个：这两个 mini view 隔得太近，视差小到只有几个 pixel，常规 stereo 方法根本抓不到。

---

## Dual-pixel 到底是个啥

你手机摄像头每个像素其实不是一个光电二极管，是两个并排放的。两个各看半个 aperture 的光。

打个比方：你用两只眼睛看世界，但把两只眼睛贴到几乎挨在一起，近到只有 1 毫米。这样你还能感知到 depth 吗？勉强能，但信号非常非常微弱。

手机做自动对焦就是靠这个：物体离焦平面越远，两个半像素看到的像就错开越多，这个"错开量"就是 disparity。对焦对准了，disparity 就归零。

所以 DP sensor 本质就是 **baseline 极小的 stereo**，附赠在你已经有的主摄里，不要额外硬件、不要额外电、不要第二个镜头。Apple、Samsung、Google 的旗舰手机全都有。

---

## 这事为什么难

DP disparity 在手机上通常只有 **-8 到 +8 pixel** 这么窄的范围。你拿经典的 stereo matching 方法（建 cost volume，枚举所有可能视差）来做，就像用大锤子砸钉子——大部分算力都浪费在"不可能出现"的 disparity 上。

更麻烦的是无纹理区域。一面白墙，左右两个 DP view 看起来一模一样，你根本看不出哪里对哪里。这时候 DP 就废了。

但 RGB 图像还有用啊——白墙虽然没纹理，但我知道"这是墙"，墙大概在那个 distance。所以 paper 的核心 idea 就出来了：

**RGB 管全局语义，DP 管局部视差，两条路分开跑，再 fuse 到一起。**

---

## 架构三句话

1. **RGB encoder**：拿 EfficientNet-Lite3 当 backbone，把图压到 1/64 大小，提取"这是客厅、那是沙发、远处是窗户"这种 global context。
2. **DP encoder**：两个浅层 siamese branch 分别处理 left DP 和 right DP，每个只下采样 2 倍，保留细节。关键点——**不能再深**，paper ablation 显示 3 层就开始掉点，5 层直接塌。因为 DP 信号本来就是亚像素级的高频信息，下采样狠一点就洗没了。
3. **WBiPAM**：在 DP 两个 branch 之间做 windowed bidirectional cross-attention，学"left 的这个 pixel 应该对应 right 的哪个 pixel"。然后把 RGB 和 DP 两路 feature 在前两个 stage fuse，丢给 U-Net decoder 出深度图。

---

## WBiPAM 怎么工作的，用大白话

想象你拿着 left DP 图的一个小 patch，要在 right DP 图里找"同一个点跑到哪去了"。

**第一步：开个小窗**  
因为 DP 视差最多 ±8 pixel，没必要在全图找。只在当前 pixel 周围开一个 $k \times 1$ 的小窗（沿 epipolar 方向），在这个窗里找对应。这跟 Swin Transformer 把图切成小窗做 attention 是一个套路，只不过这里切窗的动机是**物理 prior**——disparity 不可能跨整张图。

**第二步：算 attention**  
公式就一行：
$$\mathcal{A}_{lr} = \mathrm{softmax}(Q K^T)$$

变量含义：
- $Q = W_q \cdot F_l$：left feature 经过一个 1×1 conv 投影出来的 query
- $K = W_k \cdot F_r$：right feature 投影出来的 key
- $W_q, W_k \in \mathbb{R}^{C \times C}$：可学习投影矩阵
- $Q, K \in \mathbb{R}^{P \times k \times C}$，$P$ 是 window 数量合并进 batch，$k$ 是窗内 token 数
- $Q K^T \in \mathbb{R}^{P \times k \times k}$：每个窗内 $k \times k$ 的 matching score
- softmax 沿 key 维归一化，让每行和为 1

**第三步：把 attention 用回去**  
$$F_l^a = \mathcal{A}_{lr} \cdot F_l$$

$F_l^a[p, i, c] = \sum_j \mathcal{A}_{lr}[p, i, j] \cdot F_l[p, j, c]$

意思就是：left 的第 $i$ 个位置，按 attention 权重把窗内所有 left feature 加权求和，重新组合一下。attention 峰值偏移多少，就暗示 disparity 是多少。

**第四步：对称做一遍**  
right 也当 reference 重做一次，attention matrix 直接用 $\mathcal{A}_{rl} = \mathcal{A}_{lr}^T$ 复用（stereo 对称性）。然后 left 和 right 两路输出各自跟原 feature concat 一下过个 conv，输出 $F_l'$ 和 $F_r'$。

**为什么双向这么重要**：ablation 里单向版本性能从 0.0799 塌到 0.1454，几乎退化到 DPNet 水平。单向 attention 只让一边迁就另一边，correspondence 学不稳。双向等于左右互相校验，类似 stereo matching 里经典的 left-right consistency check，但是 soft 版本。

---

## Fusion 怎么做

RGB feature $F_i$ 跟 DP 两路 $F_l', F_r'$ 在 channel 维 concat，过 conv 算出一个 $A_f \in \mathbb{R}^{H_f \times W_f \times 3}$ 的 spatial weight map。三个 channel 分别是三个 modality 的权重。

然后：
$$F_{ilr} = A_f^{(0)} \cdot F_l' + A_f^{(1)} \cdot F_r' + A_f^{(2)} \cdot F_i$$

最后过 conv 出 $F_{ilr}'$。

直觉上：在 textureless 区域，网络学会给 RGB 通道大权重；在有纹理、DP 可信的区域，给 DP 通道大权重。这是 feature-wise recalibration，比 pixel-wise（太细、学不稳）和 channel-wise（太粗、丢空间信息）都好。Ablation 数字支持这一点。

---

## CmTL：跨模态迁移学习，解决数据少的问题

RGB-DP-D 数据集就 Google 那一份，2506 张 unique scene。RGB-D 数据集倒是一堆（KITTI、NYUv2、MegaDepth、Hypersim、Virtual KITTI 2、DIODE...）。

DiFuse-Net 的 modality-decoupled 设计正好让 RGB encoder 可以独立训练。于是三阶段：

| Stage | 训啥 | 用啥数据 |
|---|---|---|
| 1 | DP encoder + decoder | RGB-DP-D（小） |
| 2 | RGB encoder + decoder | 大型 RGB-D 数据集 |
| 3 | 整个网络 end-to-end | RGB-DP-D，把 stage 1、2 的权重当 init |

Stage 2 相当于让 RGB encoder 见过几百万张"场景 + depth"的对应关系，学到"客厅沙发大概多远、户外路面大概怎么延伸"这种 prior。Stage 3 再让 DP 信号去 fine-tune 这些 prior，把几何精度提上来。

效果：CmTL 让 1-SRCC 从 0.0833 降到 0.0799，约 4% 相对提升。不算巨大，但是 free lunch——反正 RGB-D 数据集都是公开的。

Loss 是：
$$\mathcal{L} = \mathrm{MAE}(d, \hat{d}) + 30 \cdot \mathrm{Grad}(d, \hat{d})$$

- $d$：GT inverse depth（affine invariant）
- $\hat{d}$：预测
- $\mathrm{MAE}$：scale-invariant mean absolute error
- $\mathrm{Grad}$：image gradient 域 MAE，惩罚边缘模糊
- $\lambda = 30$：edge sharpness 权重给得很重，paper 强调 sharp boundaries

---

## DCDP 数据集：怎么搞出高质量 GT

Google DP 数据集的 GT 是用 5 台 Pixel 3 组成 rig，multi-view stereo 算出来的。问题：稀疏、有孔洞、边界糊。

DiFuse 这帮人的做法：
1. **两台 Samsung S23 Ultra 对称放**，用前置摄像头（前置 baseline 短，2.5cm，更接近 DP 物理尺度）
2. **每次拍摄 session 前都重新标定**（手机镜头有 OIS、会温漂，不能一次标定终身用），用 30-40 张 checkerboard
3. 用 **RAFT-Stereo** 类 AI stereo 算法在 FlyingThings3D 上预训练，额外 augment 轻微 vertical distortion 让它对 rectification 残余误差鲁棒
4. 把 disparity 从 rectified plane **投影回 original plane**——因为去 rectify DP 图像会破坏 DP 物理信号，宁可反过来动 GT
5. **40-pixel border crop** 去掉投影边缘 artifact
6. **人工画 mask** 标出 GT 不对的区域，训练时 mask 掉

最后 5000 train + 700 test，叫 DCDP。看 Fig. 12 视觉对比：DCDP 训出来的模型边界 sharp、thin structure 保留好；Google DP 训的有"depth leakage"——物体边界处深度漏到背景里。

---

## 实验数字讲人话

主表（Google DP test）：

| Model | 1-SRCC | AIWE1 | params |
|---|---|---|---|
| DPNet (baseline) | 0.1520 | 0.0181 | 小 |
| Baseline (deeper DPNet) | 0.0927 | 0.0142 | ~10M |
| Stereo Baseline (RAFT-stereo 替 WBiPAM) | 0.0911 | 0.0137 | ~10M |
| **DiFuse-Net** | **0.0799** | **0.0128** | **9.9M** |

跟 MiDaS v3.1 BEiTL-512（345M params）比：
- MiDaS：1-SRCC = 0.0971
- DiFuse-Net：1-SRCC = 0.0799，**9.9M 打败 345M**

这数字很说明问题——纯 monocular 再大也没用，DP 那一点点几何 hint 价值连城。

---

## Ablation 关键发现，用大白话

1. **WBiPAM 拿掉**：0.0799 → 0.0865，掉 8%。Disentangle 架构本身就有价值，但 attention 还是必要的。
2. **Window 拿掉**（全图 attention）：0.0912，掉 14%。物理 prior 太重要——DP disparity 不可能跨整图，强行让远距离像素参与匹配只会引入噪声。
3. **单向 attention**：0.1454，掉 80%，几乎退回 DPNet 水平。这是最 dramatic 的发现，说明双向 cross-attention 是 WBiPAM 的真正灵魂。
4. **Fusion 策略**：feature-wise > channel-wise > pixel-wise。太细学不稳，太粗丢空间，feature-wise 刚刚好。
5. **DP encoder depth**：2 层最优。1 层不够 expressive，3 层开始掉，5 层掉到 0.0950。DP 信号太娇贵，下采样狠了就洗没了。
6. **CmTL**：+4% 相对提升。Free lunch。

---

## 这篇 paper 的真正 insight

DP sensor 给你一个"几乎 monocular 但有一点点 stereo hint"的输入。这点 hint 太弱，网络自学学不会，必须用架构把几何 prior 烧进去——具体说就是 windowed bidirectional cross-attention，限制搜索范围在物理可行的局部窗内，左右互相校验。

同时 RGB 通路单独 pretrain 在海量 RGB-D 上，让网络先学会"世界长什么样"，再用 DP 信号去 fine-tune 几何细节。

整个故事其实是个 bias-variance trade-off 的精致处理：DP 信号 bias 高（物理 prior 强）但 variance 大（noise 高），用 attention + window 去稳住 variance；RGB 信号 variance 小（数据多）但 bias 弱（无几何），用 pretrain + 大数据去稳住 bias。两条腿走路，9.9M 打 345M。

---

## 联想与开放问题

### 1. DP 物理本质是 Circle of Confusion
薄透镜公式：$\frac{1}{f} = \frac{1}{z} + \frac{1}{z'}$，CoC 直径 $c = A \cdot |1 - z'/z_{\text{focus}}|$。DP disparity 实际是 CoC 半径的离散采样。DiFuse 没显式建模这个物理，是纯 data-driven 学的。潜在改进：把 CoC monotonicity 作为 attention prior 加进去，可能在小数据上更稳。

### 2. Window size $k$ 怎么选
paper 没明说。从 disparity ≤ 8 pixel + 2× downsampling 推断 $k$ 在 7-8 之间。$k$ 太小漏掉真实 correspondence，太大退化为 "No Window" 引入噪声。这是个 bias-variance trade-off 的经典案例。

### 3. Vertical vs Horizontal disparity
手机 DP 物理上是 vertical disparity（sensor 设计如此），paper 沿用 DSLR 习惯叫 left/right，实际操作可能 transpose 一下。paper 没明说，算个 implementation detail 的 grey zone。

### 4. CmTL 的更广应用
"Modality-specific encoder + shared decoder" 这个 pattern 可以套到任何"小数据 modality + 大数据相关 modality"的场景：
- RGB + event camera depth
- RGB + polarization depth  
- RGB + NIR depth
- RGB + thermal depth

只要架构 decoupled，就能 cross-modal pretrain。

### 5. 蒸馏到纯 conv
WBiPAM 的 attention 在 inference 时是 $k \times k$ 矩阵乘，对手机 NPU 不一定友好。一个 follow-up：把训好的 WBiPAM 蒸馏成 depthwise separable conv，去掉 softmax，end-to-end 再 fine-tune。可能牺牲一点精度换大幅加速。

### 6. Manual masking 不可扩展
5700 张人工画 mask 是诚意满满但不可复制。可以改用 RAFT-Stereo 自带的 confidence map（或 MC dropout 估 uncertainty）自动识别低质量 GT 区域，公式类似 $\text{mask}(p) = \mathbb{1}[\text{conf}(p) > \tau]$。

### 7. 跟 LynVar Dense depth 的关系
Samsung 内部很可能在为下一代 Galaxy 的 portrait mode / AR depth 用这个。DP 已经是手机标配，DiFuse 这种 10M 级模型完全可部署。下一步可能是 on-device real-time version + NeRF / 3D Gaussian Splatting 集成。

---

## 一句话最终版

DP 是手机相机送你的"迷你 stereo"，但信号弱得像耳语。DiFuse-Net 教网络怎么听这个耳语：用 windowed bidirectional attention 把 geometric prior 烧进架构，用 RGB pretrain 把 semantic prior 烧进 backbone，然后让两者在 feature space 互相 recalibrate。结果就是 9.9M 模型在 narrow-band DP 信号上打败 345M 的 monocular SOTA，顺便贡献了第一个高质量 RGB-DP-D 数据集。

---

参考链接：
- Paper PDF: https://arxiv.org/abs/2409.08340
- DPNet (Garg ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/papers/Garg_Learning_Single_Camera_Depth_Estimation_Using_Dual-Pixels_ICCV_2019_paper.pdf
- Du²Net (Zhang ECCV 2020): https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf
- RAFT-Stereo: https://arxiv.org/abs/2106.13866
- MiDaS: https://arxiv.org/abs/1907.01341
- Swin Transformer: https://arxiv.org/abs/2103.14030
- Parallax Attention TPAMI: https://ieeexplore.ieee.org/document/9370715

---

# DiFuse-Net 深度讲解

Andrej，这篇 paper 来自 Samsung Research 团队（Kunal Swami 等），发表于 IEEE robotics 类会议（2024），核心是用 **smartphone dual-pixel (DP) sensor** 做 **RGB-DP depth estimation**。下面我从动机、架构、公式、训练策略、数据集、实验逐一拆解，并穿插一些延伸联想来 build your intuition。

---

## 1. 核心动机：为什么用 DP sensor 做 depth

Dual-pixel 是现代手机相机 CMOS 的"半像素分光"机制：每个像素由两个光电二极管（photodiode）组成，从同一 micro-lens 接收光线，两路信号分别对应"左 half-aperture"和"右 half-aperture"的视角。本质上是 **baseline 极小（~1mm 量级）、位于同一个 micro-lens 后方的半孔径 stereo**。

DP 的两大物理特性：
1. **defocus disparity**：离焦量越大，左右两子图在该像素处的视差就越大——这是 DP 自动对焦的工作原理，也是这篇 paper 想利用的信号。
2. **极小 disparity 范围**：智能手机光圈小（f/1.8 左右），实际 DP disparity 通常在 **−8 到 +8 pixels** 范围内，远小于 DSLR 的 DP（DSLR 光圈大、disparity 能到几十像素）。这意味着直接套用 stereo matching 网络（cost volume）会"过宽匹配"，浪费算力且容易在 textureless 区域失败。

这篇 paper 的"病理观察"是：DP 信号极弱、又高度局部化，而 RGB 全局语义信息正好能补上 DP 失效的无纹理区域。这就引出了 **"disentangle then fuse"** 这个核心设计哲学。

参考链接：
- Garg et al. ICCV 2019 (Google DP, DPNet): https://research.google/pubs/learning-single-camera-depth-estimation-using-dual-pixels/
- Wadhwa et al. SIGGRAPH 2018 (synthetic DoF): https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/

---

## 2. 整体架构：DiFuse-Net

输入：
- RGB image $I \in \mathbb{R}^{H \times W \times 3}$
- DP pair $I_l, I_r \in \mathbb{R}^{H \times W \times 1}$（注意在 phone 上 DP 物理上是 vertical disparity，但 paper 沿用 DSLR 习惯叫 left/right；而且只用 green channel——因为 DP sensor 在 Bayer 上只对 G 通道做了双光电二极管布局，节省成本）

输出：relative inverse depth $\hat{d} \in \mathbb{R}^{H \times W \times 1}$

架构由四大块构成（Fig. 3）：

### 2.1 RGB Encoder
- Backbone: **EfficientNet-Lite3**，ImageNet pretrained
- 下采样到 1/32，再加一个 inverted residual block 下采样到 1/64，扩大 receptive field
- 目的：提供 global scene context，弥补 DP 在 textureless 区域失效的问题

### 2.2 DP Encoder (Siamese, two-block)
- 两个共享权重的浅层 branch 分别处理 $I_l$ 和 $I_r$
- 每个分支两个 inverted residual block，每 block 下采样 2×
- 第一 block 输出 $F_l, F_r \in \mathbb{R}^{H/2 \times W/2 \times C}$，第二 block 输出 $F_l, F_r \in \mathbb{R}^{H/4 \times W/4 \times C}$
- **关键 ablation 结论**：DP encoder 不能再深！3/4/5 层都会掉点（Table II）。原因：DP disparity 信号本身就是亚像素级、空间高频、信息量低，进一步下采样会把它"洗掉"。这是 paper 里很有价值的一个工程发现。

### 2.3 WBiPAM (核心创新)
后面单独详述。

### 2.4 Fusion Module
- 只在 RGB encoder 的前两个 stage 融合 DP 特征（因为 DP encoder 只有两层）
- 做法：channel-wise concat $F_l', F_r', F_i$ → conv → 生成 spatial weight $A_f \in \mathbb{R}^{H_f \times W_f \times 3}$（每 channel 是一个 modality 的 weight）→ weighted sum → conv → $F_{ilr}'$
- **ablation**：feature-wise fusion 比 pixel-wise 和 channel-wise 都更好（Table II: 0.0128 vs 0.0138 vs 0.0129 in AIWE1）。Intuition：pixel-wise 太精细，无法捕捉 modality 的整体可靠性；channel-wise 太粗，丢掉了 spatial 中哪些区域 DP 可信 / 哪些区域 RGB 可信的信息。

### 2.5 Decoder
- U-Net 风格，逐级 upsample，PReLU，最终 sigmoid
- Skip connection 来自 RGB encoder（不是 DP encoder）
- Deep supervision：每个 decoder block 都输出中间深度图，参与 loss
- 这种设计让 RGB 通路成为"主干"，DP 通路成为"语义增强"

---

## 3. WBiPAM：公式与 intuition

这是 paper 的灵魂。我们逐公式拆解。

### 3.1 窗口分割（Window Partition）
输入 $F_l, F_r \in \mathbb{R}^{H_f \times W_f \times C}$。考虑到 DP disparity 只在 $[-8, +8]$ pixels 范围内，paper 沿 epipolar line 取一个 $k \times 1$ 的矩形窗（在 feature map 上）。

reshape 操作：把 $F_l, F_r$ 划分成 $k \times k$ 的 non-overlapping window，每个 window 重组为 $\mathbb{R}^{P \times k \times C}$，其中

$$P = \frac{H_f}{k} \times \frac{W_f}{k} \times k$$

被合并进 batch dim。这一步实际是 **Swin Transformer 的 window partition 思路** 嫁接进 stereo correspondence 问题。

**为什么 window 重要**（ablation "No Window WBiPAM": 0.0912 vs full 0.0799，差 14%）：
- 全图 attention 会让远离当前像素的位置也参与匹配，引入大量 noise（DP disparity 不可能跨越整张图）
- Window 把 correspondence search 限制在物理可行的局部范围，attention score 在 $k \times k$ 内学习"哪个像素对应我"
- 计算量从 $O((H_f W_f)^2)$ 降到 $O(H_f W_f \cdot k^2)$，对手机端部署友好

### 3.2 Parallax Attention
记 $F_l$ 为 reference，用两个可学习投影 $W_q, W_k$ 生成 query 和 key：

$$Q = W_q \cdot F_l \quad (1a)$$
$$K = W_k \cdot F_r \quad (1b)$$

- $W_q, W_k \in \mathbb{R}^{C \times C}$ 是 1×1 conv 的权重矩阵（linear projection）
- $Q, K \in \mathbb{R}^{P \times k \times C}$，每个 query/key 都是 $k$ 个 token，每个 token $C$ 维

Cross-attention score：

$$\mathcal{A}_{lr} = \mathrm{softmax}(Q K^T) \quad (1c)$$

- $K^T$ 把 $K$ 转置为 $\mathbb{R}^{P \times C \times k}$
- $QK^T \in \mathbb{R}^{P \times k \times k}$：每个 window 内 $k \times k$ 的 attention matrix
- softmax 沿 key 维度归一化，让每行和为 1
- $\mathcal{A}_{lr}[p, i, j]$ 含义：第 $p$ 个 window 里，left DP feature 的第 $i$ 个位置，与 right DP feature 第 $j$ 个位置的匹配概率

把 attention 应用回 $F_l$：

$$F_l^a = \mathcal{A}_{lr} \cdot F_l \quad (1d)$$

- 这里 $F_l \in \mathbb{R}^{P \times k \times C}$，矩阵乘法沿 $k$ 维
- $F_l^a[p, i, c] = \sum_j \mathcal{A}_{lr}[p, i, j] \cdot F_l[p, j, c]$
- **关键 insight**：这里 modulated 的不是 $F_r$ 而是 $F_l$ 自己！这意味着 $\mathcal{A}_{lr}$ 实际作为"对 left feature 的位置 reweighting"，把 right 当作"哪些 left 位置应该被强调"的查询依据。这和 stereo 里的 disparity estimation 是一致的——attention 峰值的偏移就编码了视差。

### 3.3 Residual + Bi-directional
把 $F_l$ 和 $F_l^a$ 在 channel 维 concat，再过一个 conv block，得到 $F_l'$。

对称地，用 $F_r$ 作为 reference，相同流程得到 $F_r'$：

$$\mathcal{A}_{rl} = \mathcal{A}_{lr}^T \quad (2a)$$
$$F_r^a = \mathcal{A}_{rl} \cdot F_r \quad (2b)$$

- $\mathcal{A}_{rl}$ 直接复用 $\mathcal{A}_{lr}$ 的转置（correspondence symmetry）
- Bi-directional 是核心：ablation 中 "Unidirectional WBiPAM" (只用 $F_l$) 性能塌到 0.1454，比 full 版本 0.0799 差近 2 倍。说明双向 attention 让两个 view 互相校准，correspondence learning 才能稳定。

### 3.4 最后 window merge
把 $\mathbb{R}^{P \times k \times C}$ 重组回 $\mathbb{R}^{H_f \times W_f \times C}$，送入 fusion module。

### 3.5 Intuition 总结
WBiPAM 本质是：
- **用 attention 替代 cost volume**：cost volume 在 4D 空间枚举所有 disparity，对 DP 的窄范围场景是浪费；attention 在 $k \times k$ 小窗内学软 correspondence，参数效率高
- **用 window 替代全图 attention**：物理先验——DP disparity ≤ 8 pixels，超出 $k$ 的匹配无意义
- **双向 cross-attention**：左右 feature 互相做 query/key，类似 stereo matching 中"left-right consistency check"的软版本

这跟 Swin Transformer 的 window attention、Parallax Attention (Wang et al. TPAMI 2022, ref [24]) 一脉相承，但加上了 stereo 对称性约束。

参考：
- Parallax attention TPAMI 2022: https://ieeexplore.ieee.org/document/9370715
- Swin Transformer: https://arxiv.org/abs/2103.14030

---

## 4. CmTL：三阶段 Cross-modal Transfer Learning

数据困境：
- RGB-D 数据集海量（KITTI, NYUv2, MegaDepth, Hypersim, Virtual KITTI 2, DIODE）
- RGB-DP-D 数据集只有 Google 那一份（12,530 train，但其中实际只 2506 张 unique，因为 5-camera rig）
- DiFuse-Net 的 modality-decoupled 设计让两个 encoder 可以独立训练，这是 CmTL 成立的前提

三阶段：

| Stage | 训练模块 | 数据 | 初始化 |
|---|---|---|---|
| 1 | DP Encoder + Decoder | RGB-DP-D (small) | random |
| 2 | RGB Encoder + Decoder | large RGB-D | ImageNet pretrained backbone |
| 3 | 整个 DiFuse-Net end-to-end | RGB-DP-D | stage1 + stage2 权重，Fusion 和 Decoder random init |

Loss：

$$\mathcal{L} = \mathrm{MAE}(d, \hat{d}) + \lambda \cdot \mathrm{Grad}(d, \hat{d}), \quad \lambda = 30$$

- $d$：GT inverse depth（affine invariant）
- $\hat{d}$：predicted affine invariant depth
- MAE：scale-invariant mean absolute error（来自 MiDaS 范式）
- Grad：image-gradient domain MAE，保留边缘
- $\lambda = 30$：gradient 项权重很大，paper 强调 sharp edges

CmTL 效果（Table II）：DiFuse-Net w/o CmTL = 0.0833 → DiFuse-Net = 0.0799 (1-SRCC)，相对提升 ~4%。

**Intuition**：这其实是把 RGB-D datasets（场景多样性）作为 RGB encoder 的 prior 注入，避免在小规模 RGB-DP-D 上 overfit。这跟 MiDaS 的 "mixing datasets for zero-shot" 思想一致，但保留了 DP 模态的 specialization。

---

## 5. DCDP 数据集：硬件 + GT 生成

这是 paper 的另一大贡献。问题：Google DP 数据集 GT 来自 5-camera multi-view stereo，sparse 且 noisy（Fig. 2 一眼能看到边界糊、孔洞多）。

### 5.1 硬件
- 两台 Samsung Galaxy S23 Ultra，对称放置
- 用 **front cameras**（不是主摄），baseline 2.5cm
- 金属支架 + rigid holder
- 用 S-pen + USB-C camera switch 同步触发

为什么用 front camera：减小 baseline 和 occlusion，更接近 DP 的物理尺度。

### 5.2 标定与 rectification
- 每次 capture session 之前都用 30-40 张 checkerboard 重新标定
- 每个 session 拍 120-150 张
- 标定/rectify 在 half-res 做，映射回 full-res
- 这是因为手机镜头是非刚性的（OIS、温漂），不能"一次标定终身用"

### 5.3 GT depth 生成
- 用 RAFT-Stereo 类的 AI stereo disparity estimation（ref [30]）
- 在 synthetic dataset（FlyingThings3D, ref [27]）上训练，并 augment slight vertical distortion → 鲁棒于 rectification 残余误差 (<3 pixels)
- 把 disparity 从 rectified plane 投影回 original plane（避免 rectify DP images 而破坏 DP 物理信号）
- 40-pixel border crop 去掉投影边缘 artifact

### 5.4 人工质量管控
作者手动画 binary mask 标出 GT disparity 错误区域，训练时 mask 掉不参与 loss。这是工业级数据集的诚意——5700 张总样本 + 人工 annotation，质量优于 Google 的自动 MVS。

数据集规模：5000 train + 700 test，叫 DCDP (Dual-Camera Dual-Pixel)。

---

## 6. 实验数据解读

### 6.1 Google DP dataset 上主结果（Table I）

| Model | 1-SRCC ↓ | AIWE1 ↓ | AIWE2 ↓ |
|---|---|---|---|
| DPNet | 0.1520 | 0.0181 | 0.0268 |
| Baseline (deeper DPNet, 同参数量) | 0.0927 | 0.0142 | 0.0218 |
| Stereo Baseline (RAFT-stereo + 同 backbone) | 0.0911 | 0.0137 | 0.0216 |
| **DiFuse-Net** | **0.0799** | **0.0128** | **0.0202** |

注意：
- Baseline 提升明显（0.1520 → 0.0927），说明 DPNet 原架构欠拟合，部分功劳来自 capacity
- Stereo Baseline 略优于 Baseline，说明 cost volume 对 DP 也有用，但不如 attention 适合窄视差
- DiFuse-Net 在所有指标上最佳，且没有增加参数量（同等参数对比）

### 6.2 Ablation（Table II）关键发现
- **WBiPAM 三种 ablation**：
  - No WBiPAM: 0.0865（仍然优于 Baseline，说明 disentangled 设计本身有价值）
  - No Window: 0.0912（window 物理 prior 至关重要）
  - Unidirectional: 0.1454（双向 attention 决定生死，性能塌到 DPNet 水平）
- **Fusion**：feature-wise 0.0799 > channel-wise 0.0855 > pixel-wise 0.0919
- **DP encoder depth**：1 layer 0.0859, 2 layer 0.0799, 3 layer 0.0868, 4 layer 0.0889, 5 layer 0.0950——非单调，2 层最优。证明"浅 but sharp"对 DP 信号是 fundamental 选择
- **CmTL**：+0.0034 绝对增益

### 6.3 与 SOTA monocular 对比（Table III）

| Model | params | 1-SRCC ↓ | AIWE1 ↓ | AIWE2 ↓ |
|---|---|---|---|---|
| MiDaS v3.1 BEiTL-512 | 345M | 0.0971 | 0.0168 | 0.0267 |
| ZoeDepth (ZoeD-M12-N, NYU) | ~300M+ | 0.1168 | 0.0272 | 0.0379 |
| **DiFuse-Net** | **9.9M** | **0.0799** | **0.0128** | **0.0202** |

9.9M params 打败 345M MiDaS——这是 DP 模态给的"几何 hint"远胜纯 monocular prior 的强证据。ZoeDepth 在 Google DP test 上表现差，因为 metric head 假设 indoor domain。

### 6.4 DCDP 上 benchmark（Table IV）

| Model | 1-SRCC ↓ | AIWE1 ↓ | AIWE2 ↓ |
|---|---|---|---|
| DPNet | 0.1522 | 0.0087 | 0.0128 |
| Baseline | 0.0928 | 0.0068 | 0.0104 |
| Stereo Baseline | 0.0912 | 0.0066 | 0.0098 |
| **DiFuse-Net** | **0.0878** | **0.0062** | **0.0092** |

注意 DCDP 上 AIWE 绝对值都比 Google DP 上小一倍多——这恰恰说明 DCDP 的 GT 更稠密、更尖锐，模型学到的 depth 也更精细。Fig. 12 视觉对比明显看到 Google DP 训练的模型有 "depth leakage"（物体边界处深度泄漏到背景），而 DCDP 训练的模型边界 sharp。

---

## 7. Intuition 延伸与联想

### 7.1 DP vs ToF vs Stereo 的物理谱
- **Stereo**：baseline 大，disparity 大，几何强，但需要两摄像头、calibration、遮挡问题严重
- **ToF**：主动式，depth 直接测量，但 power 大、室外受干扰、近距离精度差
- **DP**：baseline 极小，disparity 极小，相当于"almost monocular + tiny stereo hint"。它的优势是**单 camera、零额外硬件、零额外 power、与主摄光路完全共视**。劣势是 disparity 信噪比低，需要 strong prior。

这其实是个非常优美的"传感器–算法 co-design"问题：DP 把 stereo 的几何信号"压缩"到 sub-pixel 量级，要求算法必须**用 attention 而非 cost volume** 来提取。

### 7.2 与 Defocus Deblur 的关系
ref [14] Abuolaim & Brown 用 DP 做 defocus deblurring；ref [18] Pan et al. CVPR 2021 同时做 depth + deblur。DP 信号的物理本质是 **circle of confusion**：离焦圆的半径与 depth 偏离焦平面的距离平方相关（thin lens equation: $\frac{1}{f} = \frac{1}{z} + \frac{1}{z'}$，CoC 直径 $c = A \cdot |1 - z'/z_{focus}|$）。所以 DP disparity 实际是 CoC 半径的离散化测量。

DiFuse-Net 没显式建模 CoC，而是用 attention 隐式学习。一个潜在改进方向是：把 optical physics 作为 attention prior（比如知道 disparity 应该随 depth 单调变化）注入到 $\mathcal{A}_{lr}$ 上。

### 7.3 与 Swin/Parallax Attention 的演化链
- **ViT** (Dosovitskiy 2020): global self-attention
- **Swin** (Liu 2021): window attention + shift → 降低 O(N²) 复杂度
- **Parallax Attention** (Wang TPAMI 2022): 在 stereo 上沿 epipolar line 做 1D attention
- **WBiPAM** (this paper): 在 parallax attention 上加 window partition + bidirectional，专门为 DP 窄视差定制

演化逻辑很清晰：global → windowed → parallax-constrained → windowed parallax。每一步都引入更具体的物理/几何 prior。

### 7.4 Window size $k$ 的选择
paper 没明说 $k$ 取多少，但从 disparity ≤ 8 pixels + 2× downsampling 推断，$k$ 应该在 7-8 范围（覆盖 ±4 pixel feature-level disparity）。如果 $k$ 太小，会丢失实际存在的 correspondence；太大则退化为 "No Window" 情形，引入噪声。这其实是个 bias-variance trade-off：$k$ 小 → bias 大、variance 小；$k$ 大反之。

### 7.5 CmTL 的更广泛意义
这其实暗示一个更宏大的 principle：**当目标 modality 数据稀缺、但相关 modality 数据丰富时，modality-decoupled architecture 让 cross-modal pretraining 成为可能**。同样的思路可以迁移到：
- RGB + event camera depth estimation（event 数据少，RGB-D 多）
- RGB + polarimetric depth（偏振数据少）
- RGB + NIR (near-infrared) depth

只要架构里能找到 "modality-specific encoder + shared decoder"，就能套这个 CmTL 框架。

### 7.6 局限与潜在改进
1. **GT 投影误差**：把 disparity 从 rectified 投回 original plane 用"反转 distortion 系数符号"近似，paper 自己承认不精确。40-pixel crop 是个工程 patch。可以用更准确的 inverse mapping（比如精确的 cv2.undistort + forward warp）
2. **Manual masking**：5700 张人工标注 mask 不可扩展。可以用 confidence-based uncertainty estimation（如 MC dropout on RAFT-Stereo）自动识别低质量区域
3. **DP encoder 限于 2 层**：意味着 DP 特征空间分辨率不能太低。如果未来手机 DP sensor 改进、disparity 信号更强，可以加深 DP encoder
4. **Vertical disparity**：智能手机 DP 物理上是 vertical disparity，paper 没显式讨论 attention 是不是改成了沿 vertical axis。可能需要 transpose 后处理。这是 paper 没说清的一个 implementation 细节。

### 7.7 与 LEAS / Mobile 架构的关联
EfficientNet-Lite3 + MobileNetV2 inverted residual + U-Net decoder → 这是典型的 mobile-first 架构。9.9M params 在 RTX 3090 上训练，但部署目标是手机。对比 MiDaS 345M、ZoeDepth 数亿参数，DiFuse-Net 在 edge 部署上有明显优势。一个有趣的 follow-up：把 WBiPAM 蒸馏成纯 conv 形式（无 attention），用蒸馏在像素端 inference 进一步提速。

---

## 8. 关键 references 与延伸阅读

- **Garg et al. ICCV 2019** (DPNet, 基础工作): https://openaccess.thecvf.com/content_ICCV_2019/papers/Garg_Learning_Single_Camera_Depth_Estimation_Using_Dual-Pixels_ICCV_2019_paper.pdf
- **Wadhwa et al. SIGGRAPH 2018** (synthetic DoF on Pixel phone): https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/
- **Zhang et al. ECCV 2020** (Du²Net, dual-cam + DP): https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf
- **Pan et al. CVPR 2021** (DP depth + deblur): https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Dual_Pixel_Exploration_Simultaneous_Depth_Estimation_and_Image_Restoration_CVPR_2021_paper.pdf
- **Punnappurath et al. ICCP 2020** (modeling defocus-disparity): https://imaginglab.ca/docs/Punnappurath20-ICCP.pdf
- **Wang et al. TPAMI 2022** (parallax attention): https://ieeexplore.ieee.org/document/9370715
- **Liu et al. Swin Transformer IJCV/CVPR 2021**: https://arxiv.org/abs/2103.14030
- **RAFT-Stereo (Lipson et al. 3DV 2021)**: https://arxiv.org/abs/2106.13866
- **MiDaS (Ranftl et al. TPAMI 2021)**: https://arxiv.org/abs/1907.01341
- **ZoeDepth (Bhat et al. 2023)**: https://arxiv.org/abs/2302.12288
- **EfficientNet (Tan & Le ICML 2019)**: https://arxiv.org/abs/1905.11946
- **MobileNetV2 (Sandler et al. CVPR 2018)**: https://arxiv.org/abs/1801.04381

---

## 9. 一句话总结

DiFuse-Net 的核心 insight 是：**DP 信号太弱不能让网络"自学"如何提取，必须用 attention-based correspondence matching 把几何 prior 烧进架构；同时 RGB 通路单独预训练以利用海量 RGB-D 数据**。WBiPAM + CmTL 这两条腿，让 9.9M 模型打败 345M monocular SOTA，也产出了第一个高质量 RGB-DP-D 数据集。是 mobile depth estimation 一个非常扎实的工程贡献，物理 prior 和 attention 设计融合得相当干净。
