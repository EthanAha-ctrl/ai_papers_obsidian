---
source_pdf: FoundationStereo Zero-Shot Stereo Matching.pdf
paper_sha256: 24891ba61e728544529c95609869964a9fb603d6cfca869cda6c4c9dc9912ba9
processed_at: '2026-08-19T08:12:43-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FoundationStereo

好，我尽量像跟人聊天那样讲，但关键技术点还是会保留公式和术语，这样你既能 build intuition 又能 drill down。

---

## 这篇 paper 在干嘛

一句话：**stereo matching 领域终于有了自己的 SAM**。

你看 2D vision 这几年——SAM 切一切一切、DINOv2 做 feature、DepthAnything 做深度——都是 download 一个权重，zero-shot 跑在任何图上都 work，不用 fine-tune。但 stereo matching 这个任务一直没这种东西。你想要一个 stereo 模型 work，就得拿目标域数据 fine-tune 一下，不然准确度垮得厉害。

FoundationStereo 想做的就是：**下载一个 checkpoint，扔到任何 stereo pair 上，就给你靠谱的 disparity map**，不管你是室内、室外、驾驶、机器人、还是自己拿手机拍的两张图。

结果数字很猛：Middlebury 这个 benchmark 上，之前 SOTA BP-2 是 7.5 左右，他们做到 **1.1**。这是个数量级的跃迁，不是那种 7.5 → 7.2 的 incremental bump。

paper project page: <https://nvlabs.github.io/FoundationStereo/>

---

## 为什么之前 stereo 一直做不出 foundation model

两个老大难问题：

### 问题一：数据不够

你看其他 vision 任务的大模型，背后都是海量数据。SAM 是 11M 图像、DINOv2 是 142M、DepthAnythingV2 是 1.5M+ unlabeled 真实图。stereo 呢？主流训练集 Scene Flow 才 **40K pair**，还都是 Blender 渲的，realism 一般。后面 CREStereo 200K、TartanAir 306K 算大了，但跟 2D foundation model 的数据规模比还是差一截。

而且 stereo 数据有个天然麻烦——**real-world dense GT 很难搞**。KITTI 用 LiDAR 扫，稀疏；Middlebury 用 structured light，只能室内小场景；DrivingStereo 180K 也是稀疏 GT。所以大家都退回去用合成数据，但合成数据又有 sim-to-real gap。

### 问题二：网络架构的 inductive bias 不对

stereo 这几年主要两条技术路线：

**A. Cost volume + 3D CNN 路线**（PSMNet、GwcNet、IGEV）：先构造一个 4D cost volume $\mathbf{V} \in \mathbb{R}^{C \times D \times H \times W}$，$D$ 是 disparity 维度，然后用 3D conv 做 hourglass filtering。优点是精度高、初始 disparity 预测好；缺点是 3D conv 显存爆炸，高分辨率图直接跑不动。

**B. RAFT-Stereo iterative refinement 路线**：跳过 4D volume，只建一个 all-pairs correlation volume，然后反复用 GRU refine $d_k$。优点是显存友好、disparity range 可变；缺点是循环更新慢、没有 long-range context、容易陷 local optimum。

两条路线各打五十大板，谁也没法 dominate。而且都用 Scene Flow 训，数据规模天花板就那么高。

之前确实有人试过 zero-shot generalization（DSMNet、Mask-CFNet、GraftNet、HVT-RAFT、Former-RAFT-DAM），但都在 Scene Flow 上训，数字也不够亮。所以 stereo 一直没出现那种"我下个 checkpoint 就能用"的 foundation model。

---

## FoundationStereo 的三大杀器

这篇 paper 的核心就三件事，缺一不可。

### 杀器一：造大数据集 FSD

用 NVIDIA Omniverse 渲了 **1M stereo pair**，分辨率 1280×720，迄今最大（TartanAir 才 306K，Scene Flow 才 40K）。

关键设计有几个：

**(1) Domain randomization 做得狠**

- baseline 随机
- focal length 随机
- camera pose 随机
- lighting 类型（global illumination / directed sky rays / baked / dynamic light spheres）+ 颜色 + 强度 + 方向随机
- 5000+ 个 3D object asset，分组成 furniture / vehicles / robots / walls / plants / forklifts / digital humans / distractors，每组单独的 randomization range
- 12 个大场景（factory、hospital、office、grocery store、warehouse 等）

**varying baseline + varying focal length** 这一点特别重要。之前几乎所有合成 stereo 数据集都是固定 stereo 几何——固定 baseline、固定 focal length。模型学到的是"这个相机配置下 disparity 长什么样"，换相机配置就废。FSD 让模型在训练时就见过各种几何，所以 generalize 出去不会傻。

**(2) 高 photorealism**

用 RTX path-tracing，每像素 32-128 samples。不是 Blender 那种快但糙的渲染。看 Table 1，FSD 的 rendering realism 是 High，跟 Unreal Engine 渲的 IRS / TartanAir / FallingThings 一个档次，远超 Scene Flow 的 Low。

**(3) Iterative Self-Curation**

这一招很聪明。Domain randomization 必然会造出一些 **ambiguous samples**——比如纯色背景、过度反射、重复纹理——这些样本对学习是噪音，但生成时没法预过滤（你不知道哪些会 ambiguous）。

他们的做法：

1. 拿一版训好的 FoundationStereo 在 FSD 上跑
2. 找出 BP-2 > 60% 的样本（disparity 误差 > 2 pixels 的像素占比超过 60%）
3. 把这些样本**替换**成新生成的样本（保持总量不变）
4. 重复两次

效果：Middlebury BP-2 从 1.27 → 1.15（Table 8）。

直觉：这是 self-distillation 思路的反向应用——不是找 hard sample 重点学，是找生成器自己也搞不定的样本扔掉。因为 stereo 的 ambiguity 是物理可定义的（BP-2 大就是 ambiguous），所以比 LLM 那种 preference learning 更 grounded。

参考：
- Domain randomization (Tobin et al.): <https://arxiv.org/abs/1703.06907>
- NVIDIA Omniverse: <https://www.nvidia.com/en-us/omniverse/>

---

### 杀器二：STA — 白嫖 DepthAnythingV2 的 prior

这个 trick 我最喜欢，**简单到出奇但 work**。

问题背景：FSD 虽然大，但还是合成数据，跟真实图有 gap。怎么补？**借 DepthAnythingV2 的力量**——这个模型在 1.5M 真实 unlabeled 图像上训过，见过真实世界的纹理、光照、材质分布，远超任何合成数据。

但怎么借？他们试了三种 design（Fig. 3）：

**(a) Naive**：直接用 frozen DepthAnythingV2 的 feature pyramid，不用 CNN。BP-2 = 6.48，烂。

**(b) ViT-Adapter style**：CNN 和 ViT 双向 exchange feature（参考 ViT-Adapter）。BP-2 = 2.22，还行。

**(c) Side-tuning**：CNN 正常提特征；同时把 image 喂 DepthAnythingV2，**取 final output head 之前那一层 feature**，用 4×4 stride 4 conv 下采样到 1/4 scale，**concat** 到 CNN 的 1/4 feature 上。BP-2 = 1.97，最好。

(c) 赢得意外地明显。而且 ViT 必须冻结——unfreeze 会让 BP-2 从 1.97 掉到 3.94（Table 5）。

**为什么 (c) 这么 simple 却最强？** paper 没给严格解释，我自己猜：

DepthAnythingV2 在 final hidden representation 已经是 "ready to decode depth" 的状态——所有 metric、scale、geometric reasoning 都发生在那一层之前。ViT-Adapter 的双向 exchange 引入了 interference：adapter 试图同时教 ViT 学 stereo 和被 ViT 教 monocular prior，gradient signal 不干净。Side-tuning 是单向的"借力"，不动 ViT，所以 prior 不被污染。

这跟 Side-Tuning 那篇 ECCV 2020 的观察一脉相承——主网络 frozen，旁路加 additive adapter，比 fine-tune 整个 backbone 更稳、更 sample efficient。

**形式化**：

输入 $I_l, I_r \in \mathbb{R}^{H \times W \times 3}$，CNN（EdgeNeXt-S）输出多级金字塔：

$$
f_l^{(i)}, f_r^{(i)} \in \mathbb{R}^{C_i \times \frac{H}{i} \times \frac{W}{i}}, \quad i \in \{4, 8, 16, 32\}
$$

变量含义：
- $i$: downsample factor（4 = 1/4 resolution, ..., 32 = 1/32）
- $C_i$: 该 level 的 channel 数
- 左右 weight shared

DepthAnythingV2 frozen。输入前先把 image resize 到能被 14 整除（因为 ViT patch size = 14）。

STA 还用于 context feature 提取，CNN 换成 residual block + down-sampling 序列，输出 $f_c^{(i)} \in \mathbb{R}^{C_i \times \frac{H}{i} \times \frac{W}{i}}, i \in \{4,8,16\}$。这些 context feature 用来初始化 GRU hidden state 和每步输入，**带着 monocular prior 引导 iterative refinement**。

**为什么 DINOv2 不如 DepthAnythingV2？** Table 5：DINOv2-L 给 BP-2 = 2.46，DepthAnythingV2-S 2.22、B 2.11、L 1.97。paper 解释：DINOv2 是 contrastive self-supervised，对 dense pixel-level metric 不够锐；DepthAnythingV2 直接 supervised 在大规模 relative depth 上，更接近 stereo 需要的几何 prior。

参考：
- Side-Tuning paper: <https://arxiv.org/abs/1912.13500>
- ViT-Adapter: <https://arxiv.org/abs/2205.08534>
- DepthAnythingV2: <https://arxiv.org/abs/2406.09418>
- Probing 3D awareness of foundation models: <https://arxiv.org/abs/2406.09114>

---

### 杀器三：AHCF — 把 cost volume filtering 做对

前面 STA 给了好 feature，下一步是构造 cost volume 并 filter 它。这一块 paper 提了两个子模块：APC + DT。

#### Hybrid Cost Volume Construction

给定 1/4 scale 的 unary feature $f_l^{(4)}, f_r^{(4)}$，cost volume $\mathbf{V_C} \in \mathbb{R}^{C \times \frac{D}{4} \times \frac{H}{4} \times \frac{W}{4}}$ 由两部分组合：

$$
\mathbf{V}_{gwc}(g, d, h, w) = \langle \hat{f}_{l,g}^{(4)}(h, w), \hat{f}_{r,g}^{(4)}(h, w - d) \rangle
$$

$$
\mathbf{V}_{cat}(d, h, w) = [\text{Conv}(f_l^{(4)})(h, w), \text{Conv}(f_r^{(4)})(h, w - d)]
$$

$$
\mathbf{V_C}(d, h, w) = [\mathbf{V}_{gwc}(d, h, w), \mathbf{V}_{cat}(d, h, w)] \tag{1}
$$

变量：
- $g \in \{1,...,G\}$, $G=8$ group index（参考 GwcNet，features 均分 8 组做 group-wise correlation）
- $d \in \{1,...,D/4\}$ disparity index（feature 已经 1/4，所以 disparity 也 /4）
- $h, w$ 空间坐标
- $\hat{f}$: $L_2$ normalized feature
- $\langle \cdot, \cdot \rangle$: dot product
- $\text{Conv}$: 1×1 conv，把 unary feature 降到 14 通道省显存（左右 weight shared）

**关键直觉**：纯 correlation 只给"匹配相似度"，丢了 unary identity 信息；纯 concat 浪费 channel 且没归一化。Hybrid 是事实标准。更重要的——**concat 部分把 STA 提取的 monocular prior 保留下来了**。这是后面 zero-shot generalize 的关键，因为 correlation 在 reflective / textureless 区域会失败，但 monocular prior 还能 anchor 一个 prior disparity。

参考 GwcNet: <https://arxiv.org/abs/1901.02746>

#### APC (Axial-Planar Convolution)

问题：3D conv 的 3×3×3 kernel 在 disparity 大时 receptive field 太小，但 5×5×5 显存爆掉（80GB A100 都 hold 不住）。

解法：把一个 3D conv 拆成两个：

1. **Spatial part**: kernel size $K_s \times K_s \times 1$（只在 H, W 上做，disparity 维不动）
2. **Disparity part**: kernel size $1 \times 1 \times K_d$（只在 disparity 维上做）

每个 part 后接 BatchNorm + ReLU。

直觉：3D 版的 separable conv（参考 Xception），但**只在 spatial 和 disparity 两轴分离，channel 不分 group**——保留 representation power。Disparity 维特别处理是因为它语义不同：disparity 维编码的是"在候选 disparity 集合上的概率分布"，spatial 维只是相邻像素 smoothing。

实验扫 kernel size（Table 6）：

| Kernel (spatial, disp) | BP-2 |
|---|---|
| (3,3,1), (1,1,5) | 2.10 |
| (3,3,1), (1,1,9) | 2.06 |
| (3,3,1), (1,1,13) | 2.01 |
| **(3,3,1), (1,1,17)** | **1.97** |
| (3,3,1), (1,1,21) | 1.98 |
| (7,7,1), (1,1,17) | 1.99 |

$K_d$ 从 5 涨到 17 持续提升，到 17 饱和。Spatial kernel 用 3 比 7 还略好，更省显存。

参考 Xception: <https://arxiv.org/abs/1610.02357>

#### DT (Disparity Transformer)

这一块有个**反直觉的发现**。

流程：

$$
\mathbf{Q_0} = \text{PE}\left(\mathbf{R}\left(\text{Conv}_{4 \times 4 \times 4}(\mathbf{V_C})\right)\right) \in \mathbb{R}^{(\frac{H}{16} \times \frac{W}{16}) \times C \times \frac{D}{16}}
$$

变量：
- $\text{Conv}_{4 \times 4 \times 4}$: stride 4 的 3D conv，把 cost volume 降为 1/16 scale
- $\mathbf{R}(\cdot)$: reshape，把每个 spatial location $(h, w)$ 转成一个 batch 元素，sequence length = $\frac{D}{16}$
- $\text{PE}$: position encoding（cosine）
- 输出形状：batch = $\frac{H}{16} \times \frac{W}{16}$，token seq = $\frac{D}{16}$，channel = $C$

然后 4 层 Transformer encoder：

$$
\text{head}_i = \text{FlashAttention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)
$$

$$
\mathbf{Q_1} = \text{Norm}(\text{MultiHead}(\mathbf{Q_0}, \mathbf{Q_0}, \mathbf{Q_0}) + \mathbf{Q_0})
$$

$$
\mathbf{Q_2} = \text{Norm}(\text{FFN}(\mathbf{Q_1}) + \mathbf{Q_1})
$$

- $h = 4$ heads
- FlashAttention: 为了 memory efficiency

最后 DT 输出 trilinear upsample 回 $\mathbf{V_C}$ 尺寸，跟 APC hourglass 输出 **element-wise sum**。

**关键 ablation（Table 6）**：

| Variation | BP-2 |
|---|---|
| Full volume attention | 2.25 |
| **Disparity-only attention** | **1.97** |
| RoPE position encoding | 2.19 |
| **Cosine position encoding** | **1.97** |
| Pre-hourglass placement | 2.06 |
| Post-hourglass | 2.20 |
| **Parallel (with hourglass)** | **1.97** |

最反直觉的：**full volume attention 比 disparity-only attention 差**（2.25 vs 1.97）。理论上 full 应该有更大 receptive field，但实际更差。

paper 假设：4D cost volume 空间太大，attention 学不稳；disparity 维已经提供足够 context。

我的解读：disparity 维才是 stereo cost volume 真正"做选择"的轴——soft-argmin 就是对这一维做加权平均。Spatial 维的 smoothing APC 已经 cover 了。所以 attention 的归纳偏置放在 disparity 维更对路。这跟 ViT 在 dense prediction 任务里效果不如 CNN 的现象一脉相承——transformer 不是万能的，要放在它最擅长的地方。

参考：
- FlashAttention: <https://arxiv.org/abs/2205.14135>
- Attention Is All You Need: <https://arxiv.org/abs/1706.03762>
- RoPE: <https://arxiv.org/abs/2104.09864>

#### Initial Disparity Prediction

$$
d_0 = \sum_{d=0}^{\frac{D}{4}-1} d \cdot \text{Softmax}(\mathbf{V_C'})(d) \tag{2}
$$

- $d_0$: 初始 disparity（在 1/4 scale）
- $\mathbf{V_C'}$: filtered cost volume（APC + DT 输出 sum 后）
- Softmax 沿 disparity 维
- 经典的 **soft-argmin**（GC-Net, Kendall et al. ICCV 2017），可微的 argmin 近似

参考 GC-Net: <https://arxiv.org/abs/1703.04409>

---

## Iterative Refinement — RAFT-Stereo 风格的 GRU

有了 $d_0$，再用 GRU 迭代精修。这部分主要借鉴 RAFT / IGEV。

每步 $k$：

**Lookup**：

$$
\mathbf{V}_{corr}(w', h, w) = \langle f_l^{(4)}(h, w), f_r^{(4)}(h, w') \rangle \tag{3}
$$

$$
\mathbf{F_V}(h, w) = [\mathbf{V_C'}(d_k, h, w), \mathbf{V}_{corr}(w - d_k, h, w)] \tag{4}
$$

- $\mathbf{V}_{corr} \in \mathbb{R}^{\frac{W}{4} \times \frac{H}{4} \times \frac{W}{4}}$: 完整 4D all-pairs correlation volume（RAFT-Stereo 风格）
- $\mathbf{F_V}$: 当前 $d_k$ 处从两个 volume 分别 lookup，concat
  - $\mathbf{V_C'}(d_k, h, w)$: filtered hybrid cost volume 在 $d_k$ 处切片（2D feature map）
  - $\mathbf{V}_{corr}(w - d_k, h, w)$: correlation volume 在 $w - d_k$ 处切片

直觉：hybrid cost volume 给"过滤后的语义+几何 prior"，correlation volume 给"原始匹配相似度分布"。前者信息丰富但 coarse，后者 fine 但 noisy。互补。

**GRU update**：

输入向量：

$$
x_k = [\text{Conv}_v(\mathbf{F_V}), \text{Conv}_d(d_k), d_k, c]
$$

- $\text{Conv}_v$: volume feature 降维
- $\text{Conv}_d(d_k)$: 当前 disparity 自己过 conv（让网络知道"现在估计到哪了"）
- $d_k$: 直接拼进来
- $c = \text{ReLU}(f_c)$: context feature（**带 STA 的 monocular prior**）

GRU 三件套：

$$
z_k = \sigma(\text{Conv}_z([h_{k-1}, x_k])) \quad \text{(update gate)}
$$

$$
r_k = \sigma(\text{Conv}_r([h_{k-1}, x_k])) \quad \text{(reset gate)}
$$

$$
\hat{h}_k = \tanh(\text{Conv}_h([r_k \odot h_{k-1}, x_k])) \quad \text{(candidate)}
$$

$$
h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \hat{h}_k \quad \text{(hidden state update)}
$$

$$
d_{k+1} = d_k + \text{Conv}_\Delta(h_k)
$$

- $\odot$: element-wise product (Hadamard)
- $\sigma$: sigmoid
- $\text{Conv}_\Delta$: 输出 single-channel residual $\Delta d$

还有几个细节：

- 3 levels GRU（1/4, 1/8, 1/16），coarse-to-fine
- 初始 hidden state：$h_0^{(i)} = \tanh(f_c^{(i)}), i \in \{4,8,16\}$
- 每 level 用 **Selective-Stereo**（CVPR 2024）的 attention-based selection mechanism 来 capture 不同频率信息
- 最后 $d_k$ 用 **convex upsampling**（RAFT 原作方法）上采样到 full resolution

参考：
- RAFT: <https://arxiv.org/abs/2003.12039>
- Selective-Stereo: <https://arxiv.org/abs/2407.08856>

---

## Loss Function

$$
\mathcal{L} = |d_0 - \bar{d}|_{smooth} + \sum_{k=1}^{K} \gamma^{K-k} \| d_k - \bar{d} \|_1 \tag{11}
$$

- $\bar{d}$: ground-truth disparity
- $|\cdot|_{smooth}$: smooth $L_1$ loss（Huber loss，对 outlier 鲁棒）
- $\gamma = 0.9$: exponential weighting，**后期 iteration 权重大**（$\gamma^{K-k}$ 随 $k$ 增加而增加）
- $K = 22$ training, 32 inference

直觉：早期 iteration 允许大错（$d_0$ 用 smooth L1 是因为初始估计必然粗），后期必须精（指数加权拉高后期 supervision 权重）。这种 curriculum-style supervision 来自 RAFT 系列，已经是 stereo 领域标配。

---

## 结果数字看几个关键点

### Zero-shot benchmark (Table 2)

| Method | Mid. BP-2 | ETH3D BP-1 | KITTI-12 D1 | KITTI-15 D1 |
|---|---|---|---|---|
| IGEV++ (prev SOTA) | 7.8 | 4.1 | 5.1 | 5.9 |
| Selective-IGEV* | 7.5 | 3.4 | 3.2 | 4.5 |
| **FoundationStereo** | **1.1** | **0.5** | **2.3** | **2.8** |

Middlebury BP-2 从 7.5 → 1.1，**15% 的 prior SOTA error**。这是数量级跃迁。

### ETH3D leaderboard (Table 4)

zero-shot inference 的 EPE = 0.13，跟所有 in-domain finetune 的 SOTA 持平甚至更好。fine-tune 后 rank 1st。

### Ablation 模块组合 (Table 7 left)

| STA | APC | DT | BP-2 |
|---|---|---|---|
| ✗ | ✗ | ✗ | 2.48 |
| ✓ | ✗ | ✗ | 2.21 |
| ✓ | ✓ | ✗ | 2.16 |
| ✓ | ✗ | ✓ | 2.05 |
| ✓ | ✓ | ✓ | **1.97** |

STA 单模块就 bring −0.27（最大单模块收益），APC + DT 一起 bring −0.24。**STA 是核心，验证了"复用 monocular prior"是整个方法的关键**。

### FSD 数据集 ablation (Table 7 right)

| With FSD? | BP-2 |
|---|---|
| ✗ | 2.34 |
| ✓ | 1.15 |

加 FSD 后 BP-2 几乎砍半。Table 9 还显示 FSD 对其他方法（IGEV, Selective-IGEV）也有提升，说明 FSD 本身就是 valuable 资源（虽然没公开）。

### Translucent objects (Booster dataset, Appendix 9)

| Method | BP-1 | BP-2 | EPE |
|---|---|---|---|
| Selective-IGEV | 23.8 | 15.0 | 6.6 |
| IGEV | 30.8 | 22.3 | 22.7 |
| **FoundationStereo** | **19.0** | **9.6** | **2.2** |

EPE 6.6 → 2.2，对透明/反射物体效果显著好。这正是 monocular prior + hybrid volume 联合作用的直接体现——correlation 在透明物体上完全失败，但 monocular prior 还能 anchor 一个 prior depth。

### 速度 / 显存 (Appendix 10)

Full resolution Middlebury:
- FoundationStereo: BP-2 = 4.8, 18.5 GB peak, 8.14 s
- Selective-IGEV: BP-2 = 12.9, 6.9 GB, 2.52 s
- IGEV: BP-2 = 13.1, 6.3 GB, 2.06 s

精度 2.5× 于 baseline，但显存 / 速度都 ~3-4× 更差。paper 自己承认这是 limitation，下一步做 distillation / pruning。

参考 EfficientSAM 思路: <https://arxiv.org/abs/2312.07279>

---

## 我自己的看法

几个值得思考的点：

**1. STA (c) 为什么 work 得意外地好？** paper 没给严格解释，我猜：DepthAnythingV2 在 final hidden representation 已经是 "ready to decode depth" 的状态，所有 metric、scale、geometric reasoning 都发生在那一层之前。ViT-Adapter 的双向 exchange 引入 interference——adapter 试图同时教 ViT 学 stereo 和被 ViT 教 monocular prior，gradient signal 不干净。Side-tuning 是单向的"借力"，不动 ViT，所以 prior 不被污染。这跟 Side-Tuning 那篇 ECCV 2020 的核心观察一脉相承。

**2. 为什么 disparity-only attention 比 full volume attention 好？** 这是 paper 最深的 finding。传统 wisdom 是 receptive field 越大越好。但在 4D cost volume 上，spatial smoothing 是 APC（CNN）的强项，disparity 维的"概率分布 reasoning"才是 transformer 擅长的——因为 disparity 维上要做的是 softmax-like selection（注意力），这恰好是 self-attention 的归纳偏置。Spatial 维用 attention 反而 over-parameterize、train 不稳。

**3. Self-curation 跟 DPO / RLAIF 之类的"数据自清洗"思路是同一精神**——用 model 自己做 judge 来清洗训练数据。Stereo 里的"ambiguity"是物理可定义的（BP-2 大就 ambiguous），所以比 LLM 的 preference learning 更 grounded。

**4. FSD 的 baseline / focal length randomization** 看起来小但极为关键。这相当于让模型学到"几何 invariant representation"——不依赖固定 stereo 几何先验。这是 cross-domain generalization 的硬要求。

**5. Limitation 是 transparent objects**。FSD 的 transparent assets 有限，Booster 虽然领先，但 BP-1 还有 19%。透明物体是 stereo 的"暗物质"——correlation 完全失败，monocular prior 也只能给一个 prior depth。这块要靠更多 transparent assets + 物理建模（折射/反射）。

---

## 对 stereo 领域的 bigger picture

FoundationStereo 给 stereo 领域确立了几条新 paradigm：

1. **Data scale matters**：1M pairs + path-tracing realism + domain randomization = 必要条件
2. **Borrow from 2D foundation models**：monocular prior via side-tuning > 从头训 stereo backbone
3. **Hybrid cost volume + DT on disparity axis** 比 pure 3D CNN 或 pure transformer 都强
4. **Cost filtering + iterative refinement 合流**：IGEV 走了一半，FoundationStereo 走完了另一半（加 APC + DT）

后面如果要 follow up，可以想：
- 把 STA 替换成更强 foundation model（Depth Pro, Metric3D v2）
- 把 DT 扩展到 video stereo（temporal axis）
- 用 FSD + self-curation 思路做 multi-view stereo foundation（DUSt3R / MASt3R 路线）

参考：
- DUSt3R: <https://arxiv.org/abs/2312.14132>
- MASt3R: <https://arxiv.org/abs/2406.09756>
- Depth Pro: <https://arxiv.org/abs/2410.02073>
- Metric3D v2: <https://arxiv.org/abs/2404.15506>
- FoundationPose: <https://arxiv.org/abs/2403.07966>

---

整体来说，FoundationStereo 做了一件 stereo 领域 overdue 的事——**把 foundation model 的 scaling law recipe 真的搬到 stereo 上**，并且展示了 monocular foundation model 的 prior 可以通过 side-tuning 这种简单方式被 stereo pipeline 利用。Table 2 的数字说明，这不再是 incremental improvement，而是 paradigm shift。

如果硬要类比，这就是 stereo 领域的 "ViT moment"——之前大家都觉得 stereo 是个特殊任务，需要特殊网络架构特殊训练 trick，结果发现只要你 scale up 数据 + borrow 现有 foundation model 的 prior + 把架构 inductive bias 放对地方，zero-shot generalization 自然就来了。

---

# FoundationStereo 深度技术解析

Andrej，这篇paper很有意思，它做了 stereo matching 领域长久以来缺的一件事：**做一个真正能 zero-shot generalize 的 foundation model**，跟 2D vision foundation models (SAM, DINOv2, DepthAnything) 的成功范式对齐。下面我把每个组件讲透，让你 build 起直觉。

---

## 1. 大背景与动机：为什么 stereo matching 一直没"foundation model 化"

主流 stereo matching 路线分两派：

**(A) Cost volume aggregation 路线**（PSMNet, GwcNet, CFNet, ACVNet, IGEV）：构造 4D cost volume $\mathbf{V} \in \mathbb{R}^{C \times D \times H \times W}$（其中 $D$ 是 disparity 维度），用 3D CNN 做 hourglass filtering。优点：精度高、初始 disparity 好；缺点：3D conv 显存吃紧、不容易 scale。

**(B) Iterative refinement 路线**（RAFT-Stereo, CREStereo, IGEV++）：跳过 4D volume construction，只建一个 all-pairs correlation volume，然后用 GRU 反复 refine $d_k$。优点：disparity range 可变、显存友好；缺点：循环更新慢、缺 long-range context、容易陷入 local optimum。

之前所有 zero-shot 尝试（DSMNet, Mask-CFNet, GraftNet, HVT-RAFT, Former-RAFT-DAM）几乎都只在 Scene Flow 上训练（40K pair），跟 foundation model 的 scaling law 完全不沾边。FoundationStereo 想做的事很清楚：**把两条路线合并 + 大数据 + 复用 monocular foundation model 的 prior，做出 stereo 领域的 SAM 等价物**。

> Intuition：stereo 难做 foundation model 的根源是 stereo pair 数据少 + sim-to-real gap 大。前者靠合成大数据，后者靠"借力" DepthAnythingV2 在互联网真实图像上学到的 monocular geometric prior。这是一个非常聪明的"知识蒸馏式"思路——monocular 模型见过的真实世界纹理/光照分布远超任何 stereo 合成数据。

参考链接：
- Paper project page: <https://nvlabs.github.io/FoundationStereo/>
- DepthAnythingV2: <https://depth-anything-v2.github.io/>
- DINOv2: <https://dinov2.github.io/>
- RAFT-Stereo (3DV 2021): <https://arxiv.org/abs/2106.13837>
- IGEV-Stereo: <https://arxiv.org/abs/2303.14006>

---

## 2. 数据集 FSD (FoundationStereo Dataset)

数据是 foundation model 的命根子。他们用 **NVIDIA Omniverse** 渲染了 **1M stereo pairs**，这是迄今为止最大的 stereo 合成数据集（之前最大是 TartanAir 的 306K，Scene Flow 才 40K）。

### 2.1 关键参数（来自 Sec. 3.5 与 Appendix 11）

| 属性 | FSD | Scene Flow | TartanAir | Spring |
|---|---|---|---|---|
| Stereo Pairs | 1000K | 40K | 306K† | 6K |
| Resolution | 1280×720 | 960×540 | 640×480 | 1920×1080 |
| Sim | Omniverse | Blender | Unreal | Blender |
| Reflections | ✓ | ✗ | ✗ | ✗ |
| Camera params | Varying baseline/focal | Constant | Constant | Constant baseline |
| Layout Realism | High | Low | High | High |

注意 **"Varying baseline + varying focal length"** 这一点非常关键——大多数合成数据集都是固定 stereo 几何，导致模型学到的是固定 prior，无法 generalize 到任意相机配置。

### 2.2 Domain Randomization 策略

- **Baseline / focal length / camera pose**：随机
- **Lighting**：global illumination、directed sky rays、baked lights、动态 light spheres，颜色/强度/方向都随机
- **Object assets**：5000+ 个，分组成 furniture / open containers / vehicles / robots / walls / stairs / plants / forklifts / animated digital humans / distractors，每组单独的 randomization range
- **Scene models**：12 个大场景（factory、hospital、wood attic、office、grocery store、warehouse 等）
- **Layout 风格**：chaotic（flying objects + skybox + 平面）和 realistic（碰撞物理 + 自然光照）混合，参考 Tobin et al. 的 domain randomization 思路

### 2.3 Iterative Self-Curation（自我清洗 pipeline）

合成数据随机化必然产生 **ambiguous samples**（重复纹理、过度反射、纯色）——这些会 confuse 学习过程。

策略：
1. 训练初始版 FoundationStereo on FSD
2. 用它在 FSD 上 evaluate，找出 BP-2 > 60% 的样本（即 disparity 误差 > 2 pixels 的像素占比超过 60%）
3. 把这些样本**替换**成重新生成的新样本（不是简单删，是替换以保持数据规模）
4. 重复训练 → curation 两次

> Intuition：这是一种 self-distillation 式的 hard sample mining 反向版——不是"找难样本重点学"，而是"找生成器自己也搞不定的样本扔掉"。从实验看 Table 8，curation 让 Middlebury BP-2 从 1.27 → 1.15（约 10% 提升）。

参考：
- Domain randomization (Tobin et al. 2017): <https://arxiv.org/abs/1703.06907>
- NVIDIA Omniverse: <https://www.nvidia.com/en-us/omniverse/>

---

## 3. 网络架构总览

整体流水线（Fig. 2）：

```
I_l, I_r
   │
   ▼
[STA: EdgeNeXt-S + frozen DepthAnythingV2] ──► f_l^(4..32), f_r^(4..32)
   │
   ▼
[Hybrid Cost Volume Construction] (GWC + concat) ──► V_C ∈ R^(C × D/4 × H/4 × W/4)
   │
   ├─► [APC Hourglass 3D filter] ──► V_C' (spatial + disparity filtered)
   │                                     │
   ├─► [DT: 3D conv 4³ stride 4 → reshape → 4×Transformer] (disparity-only self-attn)
   │                                     │
   └───────────── trilinear upsample ────┘ (sum)
   │
   ▼
[soft-argmin] ──► d_0 (initial disparity @ 1/4 scale)
   │
   ▼
[3-level ConvGRU iterative refinement (22 iter train, 32 iter test)]
   │   - lookup from V_C' and correlation volume at current d_k
   │   - context feature c (with STA features) → hidden state init
   ▼
[Convex upsampling] ──► d_final @ full res
```

下面逐个剖析。

---

## 4. STA (Side-Tuning Adapter) — 复用 monocular prior 的关键 trick

这是论文我最喜欢的设计，**简单到出奇但 work**。

### 4.1 三种 design choice（Fig. 3 left）

**(a) Naive use**：直接拿 frozen DepthAnythingV2 的 DPT head 输出 feature pyramid，不用 CNN。

**(b) ViT-Adapter style**：CNN 和 ViT 之间双向 feature exchange（参考 ViT-Adapter for dense predictions, ICLR 2023）。

**(c) Side-tuning**：CNN 正常提特征；同时把 image 喂 DepthAnythingV2，**取其 final output head 之前的 feature**，用 4×4 stride 4 conv 下采样到 1/4 scale，**concat** 到 CNN 的 1/4 feature 上。

实验（Table 5）：

| Variation | BP-2 |
|---|---|
| STA (a) | 6.48 |
| STA (b) | 2.22 |
| **STA (c)** | **1.97** |
| Unfreeze ViT | 3.94 |
| Freeze ViT | 1.97 |

**(c) 完胜 (b)**，且 **freeze 是必须的**——unfreeze 会破坏 monocular prior。

> Intuition：这非常符合 side-tuning（Zhang et al. ECCV 2020）的核心观察——主网络 frozen，旁路加 additive 适配器，比 fine-tune 整个 backbone 更稳、更 sample efficient。但这里更深的一层是：DepthAnythingV2 的 **final hidden feature**（softmax 回归 depth 前那一层）已经编码了 high-resolution、metric-aware、fine-grained 的几何 + 语义 prior。把它直接 concat 进来，让 stereo cost volume 在 group-wise correlation 之外还能"看到"monocular 模型对场景的理解。

### 4.2 为什么 DINOv2 不如 DepthAnythingV2？

Table 5 row 1-4：DINOv2-L 给 BP-2 = 2.46，DepthAnythingV2-S 2.22，B 2.11，L 1.97。

paper 解释："DINOv2 less task-relevant and limited resolution to reason high-precision pixel-level correspondence"——DINOv2 是 contrastive self-supervised，对 dense pixel-level metric 还是不够锐；DepthAnythingV2 直接 supervised 在大规模 relative depth 上，更接近 stereo 需要的几何 prior。

参考：
- Side-tuning paper: <https://arxiv.org/abs/1912.13500>
- ViT-Adapter: <https://arxiv.org/abs/2205.08534>
- DepthAnythingV2: <https://arxiv.org/abs/2406.09418>
- Probing 3D awareness of foundation models: <https://arxiv.org/abs/2406.09114>

### 4.3 STA 形式定义

输入图像 $I_l, I_r \in \mathbb{R}^{H \times W \times 3}$，CNN（EdgeNeXt-S）输出多级金字塔：

$$
f_l^{(i)}, f_r^{(i)} \in \mathbb{R}^{C_i \times \frac{H}{i} \times \frac{W}{i}}, \quad i \in \{4, 8, 16, 32\}
$$

变量含义：
- $i$: downsample factor（4=1/4 resolution, ... 32=1/32）
- $C_i$: channel at level $i$
- 左右 weight shared

DepthAnythingV2 frozen，输入前先 resize 到能被 14 整除（因为 ViT 的 patch size = 14）。

STA 也用于 context feature 提取，但 CNN backbone 换成 residual block + down-sampling 序列，输出 $f_c^{(i)} \in \mathbb{R}^{C_i \times \frac{H}{i} \times \frac{W}{i}}, i \in \{4,8,16\}$，用于初始化 GRU hidden state 和每步输入。

---

## 5. AHCF (Attentive Hybrid Cost Filtering)

这是 paper 在 architecture 层面真正的核心创新，分两块：**Hybrid Cost Volume** + **APC** + **DT**。

### 5.1 Hybrid Cost Volume Construction (Eq. 1)

给定 1/4 scale 的 unary features $f_l^{(4)}, f_r^{(4)}$，cost volume $\mathbf{V_C} \in \mathbb{R}^{C \times \frac{D}{4} \times \frac{H}{4} \times \frac{W}{4}}$ 由两部分组合：

$$
\mathbf{V}_{gwc}(g, d, h, w) = \langle \hat{f}_{l,g}^{(4)}(h, w), \hat{f}_{r,g}^{(4)}(h, w - d) \rangle
$$

$$
\mathbf{V}_{cat}(d, h, w) = [\text{Conv}(f_l^{(4)})(h, w), \text{Conv}(f_r^{(4)})(h, w - d)]
$$

$$
\mathbf{V_C}(d, h, w) = [\mathbf{V}_{gwc}(d, h, w), \mathbf{V}_{cat}(d, h, w)] \tag{1}
$$

变量与上下标：
- $g \in \{1,...,G\}$, $G=8$ 是 group index（features 均分 8 组做 group-wise correlation，参考 GwcNet CVPR 2019）
- $d \in \{1,...,D/4\}$ 是 disparity index（注意这里 disparity 维度被 /4，因为 feature 已经 1/4 了，但下面说 $D$ 是原始 image disparity range）
- $h, w$: 空间坐标
- $\hat{f}$: $L_2$ normalized feature（更好 training stability）
- $\langle \cdot, \cdot \rangle$: dot product
- $[\cdot, \cdot]$: channel 维 concat
- $\text{Conv}$: 1×1 conv，**把 unary feature 降到 14 通道**以省显存（左右 weight shared）

> Intuition：纯 correlation 只给"匹配相似度"，丢失了 unary identity 信息；纯 concat 浪费 channel 数且不归一化。Hybrid 是 PSMNet/GwcNet 之后的事实标准。这里有个细节——**concat 部分把 STA 提取的 monocular prior 保留下来**了！这是关键，因为 correlation 在 reflective/textureless 区域会失败，但 monocular prior 还能 anchor 一个 prior disparity。

参考：
- GwcNet: <https://arxiv.org/abs/1901.02746>
- PSMNet: <https://arxiv.org/abs/1803.08669>

### 5.2 APC (Axial-Planar Convolution)

**问题**：3D conv 的 3×3×3 kernel 在 disparity 大时（高分辨率 image disparity 能到 400+）receptive field 太小，但 5×5×5 显存爆掉（80GB A100 都 hold 不住）。

**解法**：把一个 3D conv 拆成两个：

1. **Spatial part**: kernel size $K_s \times K_s \times 1$（只在 H, W 上做，disparity 维不动）
2. **Disparity part**: kernel size $1 \times 1 \times K_d$（只在 disparity 维上做）

每个 part 后接 BatchNorm + ReLU。

> Intuition：这就是 3D 版的 separable conv（参考 Xception, Chollet CVPR 2017），但**只在 spatial 和 disparity 两轴分离，channel 不分 group**——保留 representation power。Disparity 维特别处理是因为它的语义不同：disparity 维编码的是"在候选 disparity 集合上的概率分布"，spatial 维只是相邻像素 smoothing。

实验（Table 6 right, row 10-15）扫 kernel size：

| Kernel (spatial, disp) | BP-2 |
|---|---|
| (3,3,1), (1,1,5) | 2.10 |
| (3,3,1), (1,1,9) | 2.06 |
| (3,3,1), (1,1,13) | 2.01 |
| **(3,3,1), (1,1,17)** | **1.97** |
| (3,3,1), (1,1,21) | 1.98 |
| (7,7,1), (1,1,17) | 1.99 |

$K_d$ 从 5 涨到 17 持续提升，到 17 饱和。Spatial kernel 用 3 比 7 还略好（1.97 vs 1.99），更省显存。

参考：
- Xception / separable conv: <https://arxiv.org/abs/1610.02357>

### 5.3 DT (Disparity Transformer) — 在 disparity 维上做 self-attention

这是 paper 最 tricky 的设计选择之一。

**流程**（Eq. 2 区域）：

$$
\mathbf{Q_0} = \text{PE}\left(\mathbf{R}\left(\text{Conv}_{4 \times 4 \times 4}(\mathbf{V_C})\right)\right) \in \mathbb{R}^{(\frac{H}{16} \times \frac{W}{16}) \times C \times \frac{D}{16}}
$$

变量与上下标：
- $\text{Conv}_{4 \times 4 \times 4}$: stride 4 的 3D conv，把 cost volume 降为 1/16 scale
- $\mathbf{R}(\cdot)$: reshape，把每个 spatial location $(h, w)$ 转成一个 batch 元素，sequence length 就是 $\frac{D}{16}$
- $\text{PE}$: position encoding (cosine，见 ablation)
- 输出形状：batch = $\frac{H}{16} \times \frac{W}{16}$，token seq = $\frac{D}{16}$，channel = $C$

然后 4 层 Transformer encoder：

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = [\text{head}_1, \dots, \text{head}_h] \mathbf{W}_O
$$

$$
\text{head}_i = \text{FlashAttention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)
$$

$$
\mathbf{Q_1} = \text{Norm}(\text{MultiHead}(\mathbf{Q_0}, \mathbf{Q_0}, \mathbf{Q_0}) + \mathbf{Q_0}) \quad \text{(self-attention + residual)}
$$

$$
\mathbf{Q_2} = \text{Norm}(\text{FFN}(\mathbf{Q_1}) + \mathbf{Q_1}) \quad \text{(FFN + residual)}
$$

- $h = 4$ heads
- $\mathbf{W}_O \in \mathbb{R}^{C \times C}$: 输出 linear projection
- FlashAttention: 为了 memory efficiency

最后 DT 输出 trilinear upsample 回 $\mathbf{V_C}$ 的尺寸，再与 APC hourglass 输出 **sum**（element-wise add）。

### 5.4 DT 的关键 ablation（Table 6 left）

| Variation | BP-2 |
|---|---|
| RoPE position encoding | 2.19 |
| **Cosine position encoding** | **1.97** |
| Transformer at 1/32 scale | 2.06 |
| **Transformer at 1/16 scale** | **1.97** |
| Full volume attention | 2.25 |
| **Disparity-only attention** | **1.97** |
| Pre-hourglass placement | 2.06 |
| Post-hourglass | 2.20 |
| **Parallel (with hourglass)** | **1.97** |

**最反直觉的两点**：

1. **Full volume attention 比 disparity-only attention 差**（2.25 vs 1.97）。理论上 full 应该有更大 receptive field，但实际更差。作者假设：4D cost volume 空间太大，attention 学不稳；disparity 维已经提供足够 context。
   
   > 我的解读：disparity 维才是 stereo cost volume 真正"做选择"的轴——soft-argmin 就是对这一维做加权平均。Spatial 维的 smoothing APC 已经 cover 了。所以 attention 的"归纳偏置"放在 disparity 维更对路。这跟 ViT 在 dense prediction 任务里效果不如 CNN 的现象一脉相承——transformer 不是万能的，要放在 it 最擅长的地方。

2. **Cosine PE > RoPE**。RoPE 理论上更 adaptive，但这里 disparity size 是 constant（4D cost volume 的 D 固定），cosine 反而足够稳定。

参考：
- FlashAttention: <https://arxiv.org/abs/2205.14135>
- Attention Is All You Need: <https://arxiv.org/abs/1706.03762>
- RoPE: <https://arxiv.org/abs/2104.09864>

### 5.5 Initial Disparity Prediction (Eq. 2)

$$
d_0 = \sum_{d=0}^{\frac{D}{4}-1} d \cdot \text{Softmax}(\mathbf{V_C'})(d) \tag{2}
$$

- $d_0$: 初始 disparity（在 1/4 scale）
- $\mathbf{V_C'}$: filtered cost volume（APC + DT 输出 sum 后）
- Softmax 沿 disparity 维
- 这是经典的 **soft-argmin**（Kendall et al. ICCV 2017），可微的 argmin 近似

参考：
- End-to-end learning of geometry and context (GC-Net): <https://arxiv.org/abs/1703.04409>

---

## 6. Iterative Refinement — RAFT-Stereo 风格的 GRU 更新

有了初始 $d_0$，再用 GRU 迭代精修。这一部分主要借鉴 RAFT / IGEV 的设计。

### 6.1 多个 volume lookup

每步 $k$ 计算：

$$
\mathbf{V}_{corr}(w', h, w) = \langle f_l^{(4)}(h, w), f_r^{(4)}(h, w') \rangle \tag{3}
$$

$$
\mathbf{F_V}(h, w) = [\mathbf{V_C'}(d_k, h, w), \mathbf{V}_{corr}(w - d_k, h, w)] \tag{4}
$$

- $\mathbf{V}_{corr} \in \mathbb{R}^{\frac{W}{4} \times \frac{H}{4} \times \frac{W}{4}}$: 完整 4D all-pairs correlation volume（参考 RAFT-Stereo）
- $\mathbf{F_V}$: 当前 $d_k$ 处从两个 volume 分别 lookup，concat 起来
  - $\mathbf{V_C'}(d_k, h, w)$: 从 filtered hybrid cost volume 在 $d_k$ 处取切片（a 2D feature map）
  - $\mathbf{V}_{corr}(w - d_k, h, w)$: 从 correlation volume 在 $w - d_k$ 处取切片

> Intuition：hybrid cost volume 给"过滤后的语义+几何 prior"，correlation volume 给"原始匹配相似度分布"。两者互补——前者信息丰富但 coarse，后者 fine 但 noisy。

### 6.2 ConvGRU update（Eq. 5-9，标号在 paper 里有点错位）

输入向量 $x_k$：

$$
x_k = [\text{Conv}_v(\mathbf{F_V}), \text{Conv}_d(d_k), d_k, c]
$$

- $\text{Conv}_v$: 把 lookup 出的 volume feature 降维
- $\text{Conv}_d(d_k)$: 当前 disparity 自己经过 conv（让网络知道"现在估计到哪了"）
- $d_k$: 直接拼进来
- $c = \text{ReLU}(f_c)$: context feature（带 STA 的 monocular prior！）

GRU 三件套（reset / update / candidate）：

$$
z_k = \sigma(\text{Conv}_z([h_{k-1}, x_k])) \tag{5} \quad \text{(update gate)}
$$

$$
r_k = \sigma(\text{Conv}_r([h_{k-1}, x_k])) \tag{6} \quad \text{(reset gate)}
$$

$$
\hat{h}_k = \tanh(\text{Conv}_h([r_k \odot h_{k-1}, x_k])) \tag{7} \quad \text{(candidate)}
$$

$$
h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \hat{h}_k \tag{8} \quad \text{(hidden state update)}
$$

$$
d_{k+1} = d_k + \text{Conv}_\Delta(h_k) \tag{9}
$$

- $\odot$: element-wise product (Hadamard)
- $\sigma$: sigmoid
- $\text{Conv}_\Delta$: 输出 single-channel residual $\Delta d$，加到 $d_k$ 上得到 $d_{k+1}$

注意 paper 里 Eq. 10 是空的，估计是 typo，本应是 convex upsampling 的公式。

### 6.3 多 level GRU + attention selection

- 3 levels：1/4, 1/8, 1/16 同时 update hidden state（coarse-to-fine）
- 初始 hidden state：$h_0^{(i)} = \tanh(f_c^{(i)}), i \in \{4,8,16\}$
- 每 level 用 **Selective-Stereo** (CVPR 2024) 的 attention-based selection mechanism 来 capture 不同频率信息
- 最后 $d_k$ 用 **convex upsampling**（RAFT 原作方法）上采样到 full resolution

参考：
- RAFT (original optical flow): <https://arxiv.org/abs/2003.12039>
- Selective-Stereo: <https://arxiv.org/abs/2407.08856>

---

## 7. Loss Function (Eq. 11)

$$
\mathcal{L} = |d_0 - \bar{d}|_{smooth} + \sum_{k=1}^{K} \gamma^{K-k} \| d_k - \bar{d} \|_1 \tag{11}
$$

- $\bar{d}$: ground-truth disparity
- $|\cdot|_{smooth}$: smooth $L_1$ loss（Huber loss，对 outlier 更鲁棒，过渡参数通常是 1.0）
- $\gamma = 0.9$: exponential weighting，**后期 iteration 权重大**（$\gamma^{K-k}$ 随 $k$ 增加而增加）
- $K = 22$ during training, 32 during inference

> Intuition：早期 iteration 允许大错（$d_0$ 用 smooth L1 是因为初始估计必然粗），后期 iteration 必须精（指数加权拉高后期 supervision 权重）。这种 curriculum-style supervision 来自 RAFT 系列，已经是 stereo 领域标配。

---

## 8. 实验结果深度解读

### 8.1 Zero-shot benchmark comparison (Table 2)

| Method | Mid. BP-2 | ETH3D BP-1 | KITTI-12 D1 | KITTI-15 D1 |
|---|---|---|---|---|
| IGEV++ (SOTA) | 7.8 | 4.1 | 5.1 | 5.9 |
| NMRF | 7.5 | 3.8 | 4.2 | 5.1 |
| Selective-IGEV* | 7.5 | 3.4 | 3.2 | 4.5 |
| **FoundationStereo** | **1.1** | **0.5** | **2.3** | **2.8** |

Middlebury BP-2 **从 7.5 → 1.1**，相当于 prior SOTA 的 **15% error**！这是数量级跃迁。即使只用 Scene Flow 训练（Table 2 第一行 FoundationStereo (Scene Flow)），也已经 5.5 / 1.8 / 3.2 / 4.9 全面超越所有 baseline。

### 8.2 ETH3D Leaderboard (Table 4)

| Method | Zero-shot? | BP-0.5 | BP-1.0 | EPE |
|---|---|---|---|---|
| Selective-IGEV (finetuned) | ✗ | 3.06 | 1.23 | 0.12 |
| CroCo-Stereo (finetuned) | ✗ | 3.27 | 0.99 | 0.14 |
| IGEV-Stereo (finetuned) | ✗ | 3.52 | 1.12 | 0.14 |
| CREStereo (finetuned) | ✗ | 3.58 | 0.98 | 0.13 |
| **FoundationStereo (finetuned)** | ✗ | **1.26** | **0.26** | **0.09** |
| FoundationStereo (zero-shot) | ✓ | 2.31 | 1.52 | 0.13 |

**两种亮点**：
1. finetune 后 BP-1.0 从 0.98 → 0.26，**减少 73%**，rank 1st on ETH3D leaderboard
2. **zero-shot 推理 EPE = 0.13，跟所有 in-domain finetune 的 SOTA 持平甚至更好**——这恰恰是 foundation model 该有的特性

参考：
- ETH3D leaderboard: <https://www.eth3d.net/low_res_two_view_stereo>
- Middlebury leaderboard: <https://vision.middlebury.edu/stereo/eval3/>

### 8.3 Scene Flow in-domain (Table 3)

| Method | EPE |
|---|---|
| Selective-IGEV | 0.44 |
| NMRF | 0.41 |
| **FoundationStereo** | **0.34** |

即 in-domain 也是 SOTA，说明 architecture 本身就 strong，不只是数据帮忙。

### 8.4 Ablation: 模块组合 (Table 7 left)

| STA | APC | DT | BP-2 |
|---|---|---|---|
| ✗ | ✗ | ✗ | 2.48 |
| ✓ | ✗ | ✗ | 2.21 |
| ✓ | ✓ | ✗ | 2.16 |
| ✓ | ✗ | ✓ | 2.05 |
| ✓ | ✓ | ✓ | **1.97** |

每个模块都 bring 价值：
- STA: −0.27 (最大单模块收益！)
- APC + DT 一起: −0.24（在已有 STA 基础上）

**STA 收益最大，验证了"复用 monocular prior"是整个方法的核心**。

### 8.5 Ablation: FSD 数据集 (Table 7 right)

| With FSD? | BP-2 |
|---|---|
| ✗ | 2.34 |
| ✓ | **1.15** |

数据集加 FSD 后 BP-2 几乎砍半！Table 9 还显示 FSD 对其他方法（IGEV, Selective-IGEV）也有提升，说明 FSD 本身是 valuable 的资源（虽然目前没公开）。

### 8.6 Booster (translucent / specular) (Appendix 9)

| Method | BP-1 | BP-2 | EPE |
|---|---|---|---|
| Selective-IGEV | 23.8 | 15.0 | 6.6 |
| IGEV | 30.8 | 22.3 | 22.7 |
| **FoundationStereo** | **19.0** | **9.6** | **2.2** |

EPE 6.6 → 2.2，对透明/反射物体效果显著好，这是 monocular prior + hybrid volume 联合作用的直接体现。

### 8.7 Speed / Memory (Appendix 10)

Full resolution Middlebury:
- FoundationStereo: BP-2 = 4.8, 18.5 GB peak, 8.14 s
- Selective-IGEV: BP-2 = 12.9, 6.9 GB, 2.52 s
- IGEV: BP-2 = 13.1, 6.3 GB, 2.06 s

精度 2.5× 于 baseline，但显存 / 速度都 ~3-4× 更差。Paper 自己承认这是 limitation，下一步做 distillation / pruning。

参考：
- 0.1% data makes segment anything slim (EfficientSAM): <https://arxiv.org/abs/2312.07279>

---

## 9. 几个值得思考的点

1. **STA (c) 为什么 work 得意外地好？** paper 没给出严格解释，但我猜：DepthAnythingV2 在 final hidden representation 已经是 "ready to decode depth" 的状态，所有 metric / scale / geometric reasoning 都已经发生在那一层之前。ViT-Adapter 的双向 exchange 反而引入 interference——adapter 试图同时教 ViT 学 stereo 和被 ViT 教 monocular prior，gradient signal 不干净。Side-tuning 是单向的"借力"，不动 ViT，所以 prior 不被污染。

2. **为什么 disparity-only attention 比 full volume attention 好？** 这是 paper 最深的一个 finding。传统 wisdom 是 receptive field 越大越好。但在 4D cost volume 上，spatial smoothing 是 APC（CNN）的强项，disparity 维的"概率分布 reasoning"才是 transformer 擅长的——因为 disparity 维上要做的是 softmax-like selection（注意力），这恰好是 self-attention 的 inductive bias。Spatial 维用 attention 反而 over-parameterize、train 不稳。

3. **Self-curation 跟 DPO / RLAIF 之类的"数据自清洗"思路是同一精神**——用 model 自己做 judge 来清洗训练数据。Stereo 里的"ambiguity"是物理可定义的（BP-2 大就 ambiguous），所以比 LLM 的 preference learning 更 grounded。

4. **FSD 的 baseline / focal length randomization** 看起来小但极为关键。这相当于让模型学到 **"几何 invariant" representation**——不依赖固定的 stereo 几何先验。这是 cross-domain generalization 的硬要求。

5. **Limitation 是 transparent objects**。FSD 的 transparent assets 有限，Booster 数据集虽然 zero-shot 已经领先，但 BP-1 还有 19%。透明物体是 stereo 的"暗物质"——correlation 完全失败，monocular prior 也只能给一个 prior depth。这块要靠更多 transparent assets + 物理建模（折射 / 反射）。

---

## 10. 对 stereo 领域的 bigger picture

FoundationStereo 给 stereo 领域确立了几条新 paradigm：

1. **Data scale matters**：1M pairs + path-tracing realism + domain randomization = 必要条件
2. **Borrow from 2D foundation models**：monocular prior via side-tuning > 从头训 stereo backbone
3. **Hybrid cost volume (concat + gwc) + DT on disparity axis** 比 pure 3D CNN 或 pure transformer 都强
4. **Cost filtering + iterative refinement 合流**：IGEV 走了一半，FoundationStereo 走完了另一半（加 APC + DT）

后面如果要 follow up，可以想：
- 把 STA 替换成 更强 foundation model（Depth Pro, Metric3D v2）
- 把 DT 扩展到 video stereo（temporal axis）
- 用 FSD + self-curation 思路做 multi-view stereo foundation（DUSt3R / MASt3R 路线的 stereo 数据增强）

参考链接汇总：
- FoundationStereo project: <https://nvlabs.github.io/FoundationStereo/>
- DUSt3R: <https://arxiv.org/abs/2312.14132>
- MASt3R: <https://arxiv.org/abs/2406.09756>
- FoundationPose: <https://arxiv.org/abs/2403.07966>
- Depth Pro (metric monocular): <https://arxiv.org/abs/2410.02073>
- Metric3D v2: <https://arxiv.org/abs/2404.15506>
- MVSAnywhere (zero-shot MVS): <https://arxiv.org/abs/2505.01842>

---

整体来说，FoundationStereo 做了一件 stereo 领域 overdue 的事——**把 foundation model 的 scaling law recipe 真的搬到 stereo 上**，并且展示了 monocular foundation model 的 prior 可以通过 side-tuning 这种简单方式被 stereo pipeline 利用。Table 2 的数字说明，这不再是 incremental improvement，而是 paradigm shift。后续工作大概会沿这条"大数据 + 借力 monocular FM + hybrid filtering"路线继续推下去，等下一个公开数据集和模型 checkpoint 出来后会有更广的 follow-up。
