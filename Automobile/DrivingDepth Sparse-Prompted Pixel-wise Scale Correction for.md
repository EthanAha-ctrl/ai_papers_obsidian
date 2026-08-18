---
source_pdf: DrivingDepth Sparse-Prompted Pixel-wise Scale Correction for.pdf
paper_sha256: 0080b0b1efc3968289b44b618ca1a5eef661b6cd00b05ddf7dea4e8d524549c4
processed_at: '2026-08-18T07:05:48-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DrivingDepth

## 一句话说清楚

> **Foundation model 画出来的 depth 长得对，但是尺寸不准。LiDAR 量的尺寸准，但是只有稀疏几个点。与其从头重新画，不如拿 LiDAR 当尺子，只调一下 foundation 画的图的"缩放比例"——per pixel 地调。**

就这么回事。整个 paper 就是围绕这一句话展开的。

---

## 1. 问题是什么——用一个生活类比

想象你拿手机拍了一张风景照，然后让一个很会画画的人（foundation model）凭记忆画一张风景素描。这个画手有几个特点：

- 画得**很像**，房子的轮廓、树的形状、远山的层次都对（dense visual geometry 好）；
- 但是他不知道**实际距离**——他可能把 10 米外的房子画得像 5 米，把 50 米外的山画得像 20 米（metric scale 不对）；
- 你手头有一个 laser rangefinder（LiDAR），但只能测零星几个点的距离，而且测的时候手抖，有些点还对歪了。

现在你要怎么把这张素描变成一张**既准确又真实**的工程图？

### 笨办法（MapAnything / PriorDA 的路线）

把素描扔了，拿 rangefinder 测的那几个点当 ground truth，**从头画一张新的**。结果就是：rangefinder 测到的地方画得准，没测到的地方全靠脑补——于是在墙上突然出现一个洞、在连续的路面上莫名其妙裂开一条缝。因为你是从头画的，foundation 那个画手积累的"什么 surface 应该连续"的经验全丢了。

> 这就是 paper Figure 3 里 PriorDA 的惨状：AbsRel 数字很好看（8.25），但 EdgeCR 跌到 2.524——depth map 上全是 hallucinated 的 discontinuities，和 RGB 对不上。

### 另一个笨办法

拿一个全局缩放因子去乘整张图。但问题是：远处误差大、近处误差小，不同材质误差不一样。一个全局因子调好了远处，近处就歪了。

### DrivingDepth 的办法

**素描不扔，画手不换。** 在素描上盖一层透明纸，每个像素位置写一个"缩放倍数"——这个倍数在哪大、在哪小，由 rangefinder 的那几个点来 anchor，中间没测到的地方就 smooth 地插值过去。而且这个缩放倍数的初始值到处都是 1.0，意思是"一开始完全不动素描"，然后慢慢学。

> 这就是 "minimal-intervention principle"——**假设 foundation 是对的，只做最小修正。**

---

## 2. 为什么这个思路 Work——Intuition 拆解

### 2.1 Foundation model 已经干完了最难的事

DA3（DepthAnything3）在几百万张多视角图片上预训练过，它已经学会了：

- 哪里是 edge、哪里是 flat surface；
- relative depth 谁远谁近；
- 不同 object 的 surface 怎么连续。

这些能力是 **最难学的部分**，需要海量数据和算力。而 metric scale（"这堵墙到底是 5 米还是 8 米"）只是在这之上的一个 *scalar labeling*——相对 depth 已经对了，缺的只是一个倍数。

这就像一个人已经学会了画人体解剖图，骨骼肌肉比例都对，只是不知道这个人实际身高 1.7 米还是 1.9 米。你给他一个尺子量一下身高，他就能把整张图按比例缩放。**不需要重新学画人体。**

> 参考：[Depth Anything V2 的 affine-invariant 训练](https://arxiv.org/abs/2406.09414) 就是这个思路的源头——它故意丢弃 scale/shift，只学 relative depth，反而获得了最强的 generalization。

### 2.2 Sparse LiDAR 够用，因为只需要 anchor scale

如果要从头画 dense depth，10% 的 LiDAR 点根本不够——大部分区域没信号，model 只能瞎猜。

但如果只是 anchor 一个 per-pixel scale map，这个 map 本身是 *smooth* 的（相邻像素的 scale 差不多），几个点就能 anchor 出一块区域的 scale。

数学上：scale map $\mathbf{S}_{\text{pix}}$ 是一个 low-frequency 信号（空间上变化缓慢），而 dense depth 是 high-frequency + low-frequency 混合。要 reconstruct 一个 low-frequency 信号，需要的采样点远少于 high-frequency 信号——这是 Nyquist 采样定理的直觉。

> 参考：[Depth Completion 综述](https://arxiv.org/abs/2203.01258) 里传统方法也利用了这个 smoothness 假设，但 DrivingDepth 把它和 foundation prior 结合，效果质变。

### 2.3 By construction 比 by loss 更强

很多方法试图在 loss 里加 regularization 来"鼓励"保留 foundation geometry，比如：

$$\mathcal{L}_{\text{preserve}} = \|\hat{\mathbf{D}} - \mathbf{D}^{\text{prior}}\|_2$$

但问题是：当 LiDAR supervision 很强时，model 会 trade off——牺牲一点 preserve loss 来换 depth loss，最终还是 override 了 prior。

DrivingDepth 的做法更狠：**backbone 直接 frozen**，不让你改。scale head 初始化为 1.0，意味着训练第一步的输出 *精确等于* DA3 的输出。新增的 branch 全部 zero-init——LiDAR 信息在训练开始时 *完全进不来*。

这意味着 geometry preservation 不是靠 loss 鼓励的，而是靠 architecture *物理上保证* 的。model *想* override prior 都做不到，因为它能动的只有 scale 这一个 degree of freedom。

> 这种思路在 RL 里有 [Residual Policy Learning](https://arxiv.org/abs/1812.06298)——freeze base policy, learn small residual。在 NLP 里有 [LoRA](https://arxiv.org/abs/2106.09685)——freeze base model, learn low-rank delta。哲学一致。

---

## 3. 架构里几个有意思的 Trick，用人话讲

### 3.1 多分辨率 LiDAR Prompt——"涂抹"效应

driving LiDAR 投影到 image 上极稀疏——1920×1080 的图可能只有几千个点。在 decoder 的深层 feature map 上（比如 32×18），可能一个 valid 点都没有。

解决办法：把 sparse depth **先 downsample 到多个分辨率**，再 upsample 回 native。downsample 时 sparse 点被"涂抹"开——原本一个孤立的点，在低分辨率上变成一小块覆盖区域。这样低分辨率 decoder stage 也能看到有效信号。

类比：你在一张大白纸上撒了几粒沙子（sparse LiDAR）。直接看几乎看不到。但你拿粗砂纸磨一下（downsample），沙子被磨成一片薄薄的粉末，整张纸都有淡淡的标记了。

> 公式 Eq (4)：
> $$\mathbf{P}^m = \text{cat}_l([\hat{\mathbf{D}}_l^{\text{sp}}, \hat{\mathbf{M}}_l^{\text{sp}}])$$
> - $l$: 分辨率 level 索引（比如 level 0 = 1/1, level 1 = 1/4, level 2 = 1/8）
> - $\hat{\mathbf{D}}_l^{\text{sp}}$: 先下采样到 level $l$，再上采样回 native resolution 的 sparse depth
> - $\hat{\mathbf{M}}_l^{\text{sp}}$: 对应的 validity mask，也做同样的下采样再上采样
> - $\text{cat}_l$: 沿 channel 维度拼所有 level
>
> 消融实验 Table 5 显示：去掉 multi-resolution，AbsRel 从 11.19 跌到 13.13——掉了 2 个点，几乎回到不用 LiDAR 的水平。证明这个 "涂抹" 是核心 trick。

### 3.2 Constrained Attention——只让该说话的人说话

surround-view 有 6 个相机（前、前左、前右、后左、后右、后）。在 multi-view attention 时，如果让所有相机互相 attend，会出现荒谬的连接：前摄像头和后摄像头看到的场景完全不相交，强行 attend 只会引入噪声。

DrivingDepth 只允许两种 connection：

1. **同一时刻的相邻相机**（front + front-left，它们有重叠的视野）；
2. **同一相机跨不同帧**（同一相机在 ego motion 下看到相同场景）。

这就像开会时：只让坐在相邻位置的人讨论（视野重叠），不让前面的人和后面的人瞎搭话（毫无共同话题）。

> 公式 Eq (3) 后面的 constrained self-attention：
> $$\mathbf{X}_\ell^d = \text{Attn}_{\text{constrained}}(\tilde{\mathbf{X}}_\ell)$$
> constrained = attention mask 只允许 adjacent view at same timestamp + same view across frames。
>
> 消融 Table 5 (b)：换成 vanilla cross-view attention（所有相机互相 attend），AbsRel 从 11.19 跌到 11.38。数字上差不大，但 paper 强调定性上 vanilla 带来 geometric noise。

### 3.3 Confidence Map——让 model 自己判断 LiDAR 哪里靠谱

LiDAR 投影到 image 上有各种 noise 来源：

- calibration drift（标定漂移）；
- temporal offset（LiDAR 扫描和 camera 曝影有时间差，dynamic object 会错位）；
- surface reflection（玻璃、镜面反射导致测距错误）；
- distant points（远处 LiDAR 点投影误差大）。

与其 hand-craft 规则去过滤（比如"距离 > 50m 的点不要"），不如让 model 学一个 confidence map $\mathbf{C} \in (0,1)$：

- 在 reliable 的位置（比如直接打到实心墙面上的点），$C_p \to 1$；
- 在 unreliable 的位置（玻璃、远处、dynamic object edge），$C_p \to 0$。

然后 loss 里用 $C_p$ 加权：

$$\mathcal{L}_{\text{conf}} = \frac{1}{|\mathcal{V}|} \sum_p \bigl(C_p |\hat{D}_p - D_p^{\text{sp}}| - \lambda_c \log C_p\bigr)$$

- 第一项 $C_p |\hat{D}_p - D_p^{\text{sp}}|$：confidence 高的地方，depth error 权重大；
- 第二项 $-\lambda_c \log C_p$：防止 model 把所有 $C_p$ 都降到 0 来"作弊"（因为如果 $C=0$，第一项就消失了）。这个 $-\log C$ 像 entropy regularization，鼓励 confidence 不要太低。

直觉：model 有两个选择来减少某个 noisy LiDAR 点的 loss——要么 hard fit 它（depth 预测强行歪过去，但破坏 surface），要么把它的 confidence 降下来（承认这个点不可信，忽略它）。第二个选择代价更小（只付一个 $-\lambda_c \log C$ 的代价），所以 model 自然倾向于"不信坏点"。

> 参考：[DUSt3R](https://arxiv.org/abs/2312.14132) 首先在 3D vision 里用了这种 confidence-weighted regression，[MetricAnything](https://arxiv.org/abs/2601.22054) 借鉴并加了 80% filtering。DrivingDepth 把两者都用上了。

### 3.4 Scale Parameterization——为什么用 exp(sigmoid)

$$\mathbf{S}_{\text{pix}} = \exp\bigl(\alpha \cdot (2\sigma(\mathbf{x}) - 1)\bigr)$$

- $\mathbf{x}$: decoder 输出的 raw logits；
- $\sigma(\cdot)$: sigmoid，$\sigma(\mathbf{x}) \in (0,1)$；
- $2\sigma(\mathbf{x}) - 1 \in (-1, 1)$: 拉到对称区间；
- $\alpha$: learnable，初始 0.5；
- $\exp(\cdot)$: 确保正数。

三个好处：

**1. 初始为 1**：训练开始时 $\mathbf{x} \approx 0$，$\sigma(0) = 0.5$，$2 \times 0.5 - 1 = 0$，$\exp(0) = 1$。所以 $\mathbf{S}_{\text{pix}} = 1$，输出 = DA3 prior。Model 从"不动"开始，慢慢学 correction。

**2. 对称性**：在 log space 里，$\log \mathbf{S}_{\text{pix}} = \alpha(2\sigma - 1) \in (-\alpha, \alpha)$。乘 2 和乘 0.5 在 log space 里分别是 $+\log 2$ 和 $-\log 2$——对称。如果直接用 linear parameterization（$S = 1 + x$），model 学"放大"和"缩小"的难度不对称。

**3. Bounded**：因为有 $\alpha$ 和 sigmoid，scale 不会 explode。即使 model 想在某些位置大幅修正，最多到 $\exp(\alpha)$。这防止了 outlier LiDAR 点把 scale 拉到极端值。

类比：这就像调音量旋钮——不是线性转（0 到无穷），而是对称的"减一点 / 加一点"，而且有最大最小限制。初始位置在正中间（scale=1）。

> 参考：类似 parameterization 在 [NeRF 的 color](https://arxiv.org/abs/2003.08934) 里用过（用 sigmoid 限制范围），在 [Gaussian Splatting 的 opacity](https://arxiv.org/abs/2308.14737) 里也用过（用 sigmoid + exp）。都是"用 bounded function parameterize 物理量"的思路。

### 3.5 Surface-Normal Regularization——最巧妙的 loss

$$\mathcal{L}_{\text{norm}} = \frac{1}{|\mathcal{M}|} \sum_p \bigl(1 - \langle \mathcal{N}(\hat{\mathbf{D}}^c, \mathbf{K})_p, \mathcal{N}(\mathbf{D}^{\text{prior}}, \mathbf{K})_p \rangle\bigr)$$

- $\mathcal{N}(\mathbf{D}, \mathbf{K})$: 把 depth map 用 intrinsics $\mathbf{K}$ unproject 成 3D 点云，再用 finite difference 算每个 pixel 的 surface normal（单位向量）；
- $\langle \cdot, \cdot \rangle$: 两个 unit normal 的点积（= cosine similarity）；
- $\mathcal{M}$: valid pixels，排除天空。

这个 loss 的意思：**corrected depth 的 surface 朝向，要和 prior depth 的 surface 朝向一致。**

为什么这个比直接约束 depth 差值更好？

直觉：depth 差值约束 $\|\hat{\mathbf{D}} - \mathbf{D}^{\text{prior}}\|$ 只看 *绝对值*，允许 model 在 LiDAR 点位置 local spike（一个小尖刺），只要 spike 不太大。但 normal 是 *一阶 spatial derivative*——如果你在 flat wall 上突然凸起一个点，wall 的 normal 在那个位置会突变（从指向你变成指向斜方），立即被 normal loss 抓住。

类比：depth loss 像检查"海拔高度"，normal loss 像检查"坡度"。一个孤立的尖刺海拔可能不高（depth loss 抓不到），但坡度极陡（normal loss 立即惩罚）。

消融 Table 6 显示：

| $\lambda_{\text{norm}}$ | AbsRel↓ | EdgeCR↑ | 现象 |
|---|---|---|---|
| 2 | 10.16 | 5.667 | LiDAR coverage 边界出现 horizontal line（surface 断裂） |
| 5 | 11.19 | 5.741 | sweet spot |
| 10 | 11.79 | 5.776 | 过度约束，scale correction 被压抑 |

$\lambda_{\text{norm}}$ 就是一个 **连续旋钮**，直接控制 geometry-scale conflict 的 trade-off。扭到一边 metric 准但 surface 烂，扭到另一边 surface 好但 metric 差。Paper 选 5 作为平衡点。

> 参考：[Metric3D v2](https://arxiv.org/abs/2406.18112) 也用 surface normal 作为 auxiliary supervision，但 DrivingDepth 用法相反——不是用 normal 来 *监督* depth，而是用 normal 来 *约束* depth correction 不要破坏 prior。

---

## 4. 实验结果的人话解读

### 4.1 主结果（Table 2）的核心 story

| 方法 | AbsRel↓ | EdgeCR↑ | 一句话评价 |
|------|---------|---------|-----------|
| DA3（无 LiDAR） | 15.88 | 7.745 | 画得好看，尺寸瞎猜 |
| PriorDA（fit LiDAR） | 8.25 | 2.524 | 尺寸准，画烂了 |
| MapAnything（从头画） | 11.99 | 1.914 | 尺寸还行，画也烂了 |
| **DrivingDepth** | **11.19** | **5.741** | 尺寸准，画也好看 |

最关键的是 **AbsIn02**（0.2m 误差内像素比例）：

- MapAnything: 50.43
- DrivingDepth: **61.14**

差 11 个点。AbsIn02 直接对应 occupancy grid 的精度需求——一个 voxel 20cm，你的 depth 误差必须在 20cm 内才有用。这 11 个点的差距意味着 DrivingDepth 重建出来的 3D 世界里，有 11% 更多的像素落在了正确的 voxel 里。

### 4.2 10% LiDAR 的鲁棒性（Table 3）——最 punch 的结果

nuScenes 上：

- MapAnything 用 100% LiDAR：AbsIn02 = 50.43
- **DrivingDepth 用 10% LiDAR：AbsIn02 = 54.66**

**DrivingDepth 用 1/10 的 LiDAR，比 MapAnything 用满血 LiDAR 还好。**

这说明什么？说明 MapAnything 的 model 大部分 capacity 都浪费在"从头画 depth"上了，只有一小部分 capacity 真正在利用 LiDAR 信号。而 DrivingDepth 的 model 因为有 foundation prior 兜底，所有 capacity 都集中在"如何用 LiDAR 调 scale"这一件事上——所以 sparse 信号也能榨干用。

更极端的是 DDAD 数据集上：MapAnything 在 100% LiDAR 下 AbsRel = 09.56，在 10% 下 = 09.91——**LiDAR 变多反而变差了**。因为 DDAD 的 LiDAR 虽然密但也更 noisy，MapAnything rigidly fit 这些 noisy 点，越 fit 越烂。DrivingDepth 的 confidence mechanism 自动 downweight 坏点，所以不受影响。

> 这呼应了一个 deep learning 的经典 insight：[Rethinking the Value of Network Weights (Frankle & Carbin)](https://arxiv.org/abs/1803.03635)——不是参数越多越好，是"对的参数"越多越好。DrivingDepth 用对的方法（residual correction），用更少信号（10% LiDAR），效果反而更好。

### 4.3 单帧也能 work（Table 4）

1 frame（无 multi-view fusion）：AbsRel = 11.86
4 frames：AbsRel = 11.19

差 0.67 个点。但 AbsIn02 从 49.76 跳到 61.14——多 frame 主要提升 *精细结构*，不提升 *scale accuracy*。

这进一步证明：**pixel-wise scale correction 本身就是核心机制**，multi-view 只是锦上添花。即使你只有单张图 + sparse LiDAR，这个方法依然有效。

---

## 5. 这篇 Paper 的 Design Philosophy

### 5.1 Minimal Intervention——foundation model 时代的正确姿势

| 时代 | 做法 | 哲学 |
|------|------|------|
| Pre-foundation | 从头训 depth model | "什么都要学" |
| Foundation era (naive) | Foundation + end-to-end finetune | "Foundation 给初始，我来改进" |
| Foundation era (wise) | **Frozen foundation + residual correction** | "Foundation 已经对了，我只做最小修正" |

DrivingDepth 属于第三种。这种 philosophy 的核心假设是：**foundation model 的 prior 足够强，大部分情况下是对的；少数不对的地方，用 lightweight correction 修补即可。**

这个假设在 2025-2026 年越来越成立——DA3、VGGT、Metric3D v2 这些 model 在大量数据上预训练，geometric prior 已经非常 robust。再 end-to-end retrain 反而容易 overfit 到特定 dataset，破坏 generalization。

> 类比：
> - [ControlNet](https://arxiv.org/abs/2302.05543) 在 Stable Diffusion 上加 conditional control——frozen base + trainable copy；
> - [IP-Adapter](https://arxiv.org/abs/2308.06721) 加 image prompt——frozen base + lightweight adapter；
> - [LoRA](https://arxiv.org/abs/2106.09685) 在 LLM 上加 task adaptation——frozen base + low-rank delta。
>
> 都是同一个 philosophy：**don't fix what ain't broken, just add what's missing.**

### 5.2 Geometry-Scale Conflict 作为 Problem Reformulation

paper 最有价值的 contribution 可能不是某个 module，而是 **把问题重新表述** 了：

- 原始问题："如何用 sparse LiDAR 生成 dense metric depth"——一个 underdetermined problem，有无数解，model 容易 overfit；
- 重新表述后："给定 dense relative depth prior，如何学 per-pixel scale map"——一个 well-posed problem，因为 scale map 是 low-frequency smooth signal，sparse anchor 足以确定。

这种 reformulation 让问题的 *难度* 大幅下降：从"生成 high-frequency dense depth"降到"生成 low-frequency smooth scale map"。后者的 sample complexity 远低于前者——所以 10% LiDAR 就够用了。

> 类比：这就像从"预测完整图像"降到"预测图像的全局色调"。前者需要理解所有 visual structure，后者只需要一个 scalar。把 hard problem decompose 成 easy problem + frozen prior，是 foundation model 时代的关键 skill。
>
> 参考：[NeRF 到 Gaussian Splatting 的演进](https://arxiv.org/abs/2308.14737) 也有类似 philosophy shift——从"学一个 implicit field"到"直接优化 explicit representation"，problem formulation 变简单了。

### 5.3 By Construction > By Loss

paper 反复强调一个点：geometry preservation 不是靠 loss 鼓励的，是靠 architecture 保证的。

- backbone frozen → geometry 不可能被 overwrite；
- scale init = 1 → 训练起点 = prior；
- zero-init injection → LiDAR 信号训练初始时进不来；
- exp parameterization → scale bounded，不会 extreme；
- normal regularization → 即使 scale 偏了，surface orientation 约束住。

每一层都是 *defense in depth*。即使 loss 完全不加 normal regularization，backbone frozen + scale init=1 也保证了 geometry 不会崩——只是 metric 可能差一点。这就是 "by construction" 的力量。

> 类比：这就像写代码时，与其靠 runtime check 防止 bug（by loss），不如靠 type system 让 bug 编译不过（by construction）。[Rust 的 ownership system](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) 就是 by construction 的 memory safety——编译时保证，不需要 runtime GC。
>
> 在 ML 里，[Normalizing Flows](https://arxiv.org/abs/1505.05770) 的 invertibility 也是 by construction——架构设计保证可逆，不需要 loss 来鼓励。

---

## 6. 联想：这个思路还能用在哪

### 6.1 Medical Imaging：CT/MRI + sparse annotation

场景：你有一个 foundation model 能生成 dense organ segmentation，但只有医生标注的几个点。与其从头训 segmentation，不如 frozen foundation + 学一个 per-pixel correction map。

> 类似工作：[SAM (Segment Anything)](https://arxiv.org/abs/2304.02643) + sparse prompt，但 SAM 是从头 predict mask。DrivingDepth 式的 "frozen + residual" 可能更鲁棒。

### 6.2 Video Generation：text-to-video foundation + sparse control

场景：text-to-video foundation model 生成质量很高，但用户想控制某个物体的运动轨迹。与其 retrain model，不如 frozen foundation + 学一个 per-frame motion correction。

> 类似思路：[ControlVideo](https://arxiv.org/abs/2305.17098) 但它是加 conditional branch，不是 residual correction。DrivingDepth 式的 residual 可能更轻量。

### 6.3 LLM + fact-checking

场景：LLM 生成了一段文字，你有一个知识库验证了其中几个 fact。与其 retrain LLM，不如 frozen LLM + 学一个 per-token "correction weight"——在 fact 错误的位置 weight 大，其他位置 weight 小。

> 这其实就是 [in-context editing](https://arxiv.org/abs/2305.12740) 的思路，但 DrivingDepth 的 per-pixel scale 可以类比为 per-token confidence correction。

### 6.4 Robotics：imitation learning + tactile feedback

场景：你有一个 imitation learning policy 从视觉生成动作，但有些细节（如抓取力度）需要 tactile sensor 校正。Frozen visual policy + 学一个 residual force correction——这比 end-to-end retrain 更 sample efficient。

> 参考：[Residual Policy Learning](https://arxiv.org/abs/1812.06298) 已经在做这个，DrivingDepth 的 confidence weighting 可以加进来——让 model 自动判断 tactile 信号哪里可靠。

### 6.5 Music Generation：score + audio alignment

场景：symbolic music score 是"骨架"（像 foundation prior），audio recording 是"实际表现"（像 LiDAR measurement）。Frozen score + 学一个 per-note timing/velocity correction——比从头生成 audio 更 controllable。

> 参考：[NES-Music](https://arxiv.org/abs/1703.11447) 类似思路，但不是 residual correction。DrivingDepth 的 framework 可能能直接套用。

---

## 7. 有什么不足 / 可以挑刺的地方

### 7.1 只能修 scale，不能修 structure

paper 自己承认：如果 foundation 在玻璃/透明表面上 geometry 本身就错了，scale correction 救不回来——因为乘以 scale 还是错的形状。

可能的改进：detect prior 不可靠的区域（比如用 confidence map），在那些区域 *selective unfreeze* backbone，让它重新学。但这会破坏 "by construction" 的保证，需要 trade-off。

> 参考：[Mamba 的 selective scan](https://arxiv.org/abs/2312.00752) 也是"selective"思路——不是所有位置都同等对待，根据 input 决定哪里要重点处理。

### 7.2 Global scale $s_g$ 是 closed-form，不参与 backprop

这意味着 model 学的 $\mathbf{S}_{\text{pix}}$ 只关心 *shape*（哪里 scale 大、哪里小），不关心 *magnitude*（整体多大）。magnitude 完全由 ROE 算法决定。

问题：如果 LiDAR 全是 noise（极端情况），ROE 算出的 $s_g$ 也会错，但 model 没有机制去修正它——因为 $s_g$ detached。

可能的改进：让 $s_g$ 可学但加 strong regularization，或者用一个 *learnable* scale head 预测 global scale，和 ROE 做加权平均。

> 参考：[Metric3D](https://arxiv.org/abs/2307.10984) 用 canonical camera space 把 metric scale 变成可学的，DrivingDepth 可以借鉴。

### 7.3 Dynamic object 没有专门处理

paper 的 constrained attention 假设"同一相机跨帧看到相同 scene"——但这只在 static scene 成立。如果有车开过，同一像素位置在不同帧看到不同 object，cross-frame attention 会 confused。

paper 没有显式处理 dynamic object（Figure 4 的 green crop 里 moving truck 有些 distortion）。可能的改进：引入 optical flow / object detection mask，在 dynamic region 禁用 cross-frame attention。

> 参考：[DGGT](https://arxiv.org/abs/2603.04535) 和 [StreetForward](https://arxiv.org/abs/2603.19552) 专门处理 dynamic driving scene，DrivingDepth 可以借鉴它们的 dynamic masking。

### 7.4 只测了 nuScenes 和 DDAD

两个都是 driving dataset，camera 配置固定（6 相机 surround-view）。在其他配置（比如单目前视、或者 3 相机）下是否同样有效，没有验证。

而且 nuScenes 的 LiDAR 是 32 线，比较稀疏。如果是 128 线 LiDAR（更密），multi-resolution prompt 的增益可能下降——因为 native resolution 已经够密了。

> 参考：[KITTI](https://www.cvlibs.net/datasets/kitti/) 和 [Waymo Open Dataset](https://waymo.com/open/) 有不同 LiDAR 配置，可以做更多验证。

### 7.5 和 Gaussian Splatting 路线的对比缺失

paper 比的都是 depth estimation 方法，但没有和 [3D Gaussian Splatting](https://arxiv.org/abs/2308.14737) 路线比（如 [DrivingForward](https://arxiv.org/abs/2412.07834), [VGD](https://arxiv.org/abs/2510.19578)）。GS 路线不直接输出 depth map，但输出的 3D representation 可以 render 出 depth——而且 boundary consistency 天然好（因为 Gaussian 是 explicit representation）。

如果 DrivingDepth 的 point cloud 和 GS 的渲染结果比，谁更 sharp？这是个 open question。

---

## 8. 最核心的 Takeaway

如果只能记住一句话：

> **当 foundation model 已经足够好时，正确的做法不是 replace it，而是 calibrate it——用最少的 learnable parameter，做最小的修正，by construction 保留 prior 的优点。**

这句话不仅适用于 depth estimation，适用于整个 foundation model 时代的技术选择。

DrivingDepth 就是把这个 philosophy 落地到 driving depth 上：

- Frozen DA3 backbone（保留 dense visual geometry）；
- Lightweight scale head（只学 per-pixel scale）；
- Multi-resolution LiDAR prompt（处理 sparsity）；
- Confidence weighting（处理 noise）；
- Surface-normal regularization（处理 surface continuity）；
- Constrained attention（处理 co-visibility）。

每一个 component 都在服务这一个 philosophy。没有花哨的新 module，只有 *对问题本质的清晰理解* + *对 prior 的尊重*。

> 这让我想起 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)（Rich Sutton）的反面——Sutton 说"利用 computation 的方法最终会赢过利用 human knowledge 的方法"。但 DrivingDepth 似乎是个 counter-example：它 *利用* foundation model 的 human-like geometric knowledge，只做 minimal computation correction。也许 truth 是：**当 computation 已经训出了 foundation model，我们要做的不是再堆 computation，而是 smartly 利用 foundation 的 knowledge。**
>
> 或者说：The Bitter Lesson 适用于 *训练 foundation*，但不适用于 *使用 foundation*。使用 foundation 时，smart inductive bias（如 minimal intervention）反而是对的。

---

## 9. 一句话总结

**DrivingDepth = Frozen DA3 + Per-pixel Scale Head + Sparse LiDAR Prompt + Confidence + Normal Regularization。**

核心 insight：**foundation model 画得对，只是尺寸不对；LiDAR 知道尺寸，只是点太少。那就别重画，只调缩放比例。**

结果：用 10% LiDAR 打赢别人 100% LiDAR，EdgeCR 高 3 倍，训练成本 1/4。干净利落。

---

# DrivingDepth: Sparse-Prompted Pixel-wise Scale Correction 深度解读

## 1. 问题动机：Geometry-Scale Conflict 的本质

这篇 paper 抓住的是 autonomous driving depth estimation 中一个长期被忽视的**结构性矛盾**，作者称之为 **geometry–scale conflict**：

- **Dense visual geometry** 来自 camera + foundation model（DepthAnything3、VGGT、DUSt3R 等），pixel-aligned、edge-sharp，但是 *没有可靠 metric scale*；
- **Metric scale** 来自 LiDAR projection，物理上可信，但是 *sparse, noisy, misaligned with image structures*。

关键点在于这两个 property 来源 *orthogonal*——任何一个 sensor modality 单独都无法满足 downstream task（3D reconstruction / occupancy prediction）的全部需求：dense + pixel-aligned + metric-consistent。

Naive combination 的失败模式（这是 intuition building 的关键）：

| 路线 | 失败模式 |
|------|----------|
| 直接 fit sparse LiDAR（PriorDA, MapAnything） | 在 LiDAR 点位置产生 *phantom discontinuities*，在 visually continuous surfaces 上出现 holes / broken structures |
| 单纯 foundation model | 没有 metric scale，distant structure 与 LiDAR 产生 *layering*（图 4 中 DA3 的剥离现象） |
| Global post-hoc alignment（如 ROE scale） | 只能纠正 average scale，无法处理 spatially varying metric error（不同 surface、不同 depth range 误差不同） |

> 参考：[Depth Anything V2](https://arxiv.org/abs/2406.09414), [VGGT](https://arxiv.org/abs/2503.11651), [DUSt3R](https://arxiv.org/abs/2312.14132), [MapAnything](https://mapanything.github.io/)

---

## 2. Key Insight：Frozen Prior + Residual Scale Correction

这是整篇 paper 最 deep 的 insight，需要仔细展开：

> Foundation model（DA3）**已经** capture 了 geometrically coherent *relative* depth；**不需要再学 surface structure**，只需要学一个 *per-pixel scale factor*，将 relative geometry mapping 到 metric coordinates。

这意味着两件事：

### 2.1 数学上的 decomposition

记 frozen foundation model 的输出为 relative depth $\mathbf{D}^{\text{prior}}$，则 ground-truth metric depth $\mathbf{D}^{\text{gt}}$ 可以近似为：

$$\mathbf{D}^{\text{gt}} \approx \mathbf{D}^{\text{prior}} \odot \mathbf{S}^{\text{true}}_{\text{pix}}$$

其中 $\mathbf{S}^{\text{true}}_{\text{pix}}$ 是 ground-truth 的 per-pixel scale map。这个分解之所以可行，是因为 DA3 输出本身就在一个 *affine-invariant* 的 equivalence class 中——它 capture 了 relative depth structure，但是 *scale* 和 *shift* 是 ambiguous 的。DrivingDepth 假定 shift 可以被 absorb 进 scale（在远距离场景 shift 项相对可忽略），只学 scale 这一个 degree of freedom。

### 2.2 为什么这是 minimal intervention

如果直接 regenerate depth（像 MapAnything 那样），model 在 LiDAR sparse / noisy 的地方可以 *freely deviate*，于是破坏了 foundation 的 coherent geometry。DrivingDepth 通过**架构设计**（而非仅靠 loss）保证几何先验 *by construction* 被保留：

- backbone 完全 frozen；
- scale head 初始化为 $\mathbf{S}_{\text{pix}} = \mathbf{1}$，意味着训练起点 = DA3 输出本身；
- 新增 branch 都用 zero-init convolution / neutral init，progressive learning。

> 参考：[PromptDA](https://arxiv.org/abs/2503.19013), [PriorDA / Depth Anything with Any Prior](https://arxiv.org/abs/2506.06720)

---

## 3. Method 详细架构解析

### 3.1 Overall Pipeline

输入：surround-view images $\mathcal{I} = \{\mathbf{I}_{t,v}\}$，$T$ 帧 × $V$ 相机（nuScenes 是 $T=4, V=6$），以及 camera intrinsics/extrinsics $\pi$。

Pipeline：
```
images + π ─► [Frozen DA3] ─► D_prior (dense relative depth)
                            │
sparse LiDAR ──► P^L ──┐    │
                       ▼    ▼
              [Geometry-Preserving Feature Adapter]
                       │
                       ▼  depth-aware features {F_ℓ}
              [Sparse-Aware Pixel-Scale Head]
                       │
                       ▼  S_pix (per-pixel scale) + C (confidence)
              D^c = D_prior ⊙ S_pix
                       │
              s_g = ROE(D^c, D^sp)   ← 不进入 backprop
                       │
                       ▼
              D̂^c = D^c · s_g  (final metric depth)
```

### 3.2 核心公式逐项解析

#### Eq (1): Scale Decomposition

$$\mathbf{D}^c = \mathbf{D}^{\text{prior}} \odot \mathbf{S}_{\text{pix}}, \qquad \hat{\mathbf{D}}^c = \mathbf{D}^c \cdot s_g$$

- $\mathbf{D}^c$: **locally-corrected depth**，per-pixel scale 已经校正但还在 relative space；
- $\mathbf{D}^{\text{prior}}$: frozen DA3 输出，shape $[B, T, V, H, W]$；
- $\odot$: element-wise (Hadamard) product；
- $\mathbf{S}_{\text{pix}}$: per-pixel multiplicative scale，shape 同 $\mathbf{D}^{\text{prior}}$；
- $s_g$: **clip-level** scalar（一个 clip 中所有 frame、所有 view 共享），保证 cross-view/cross-frame metric consistency；
- $\hat{\mathbf{D}}^c$: 最终 metric depth输出。

**Intuition**: 分解成 *local per-pixel correction* + *global scale*。Local 项处理 spatially varying error（近处 vs 远处、不同 material），global 项确保 clip 整体 scale 一致——这对 surround-view reconstruction 至关重要，否则 6 个相机拼出来的点云会撕裂。

#### Eq (2): Global Scale via ROE

$$s_g = \text{ROE}(\mathbf{D}^c, \mathbf{D}^{\text{sp}})$$

- ROE = **Robust Outlier Elimination** alignment（参考文献 [29]，即 MoGe 的 robust scale alignment 方法）；
- $\mathbf{D}^{\text{sp}}$: projected sparse LiDAR depth；
- 关键：**$s_g$ detached from gradient**。这意味着 model 直接学的只是 $\mathbf{S}_{\text{pix}}$，$s_g$ 是 inference 时通过最小二乘 / median-of-ratios 计算的 closed-form scale。

为什么 detach？因为如果把 $s_g$ 也放进 backprop，model 会 trivially 让 $s_g \to \infty$ 来 minimize depth loss；且 $s_g$ 依赖 sparse 点，gradient 会很 noisy。Detach 之后，$\mathbf{S}_{\text{pix}}$ 学到的是 *shape*（哪里 scale 偏大、哪里偏小），$s_g$ 学到的是 *magnitude*——这种 disentanglement 非常 elegant。

#### Eq (3): Cross-Attention 注入 sparse depth

$$\tilde{\mathbf{X}}_\ell = \text{Attn}_{\text{cross}}(Q=\mathbf{X}_\ell, K=V=\mathbf{Z}^d)$$

- $\mathbf{X}_\ell \in \mathbb{R}^{B \times TV \times N \times C}$: 第 $\ell$ 层 image tokens。$B$=batch, $T$=frames, $V$=views, $N$=patches per view, $C$=channels；
- $\mathbf{Z}^d = E_d(\mathbf{P}^m)$: sparse-depth tokens，由 lightweight CNN patch-embedding 编码；
- $Q, K, V$ 是 attention 的 standard query/key/value；
- 关键设计：$K=V$ 是 sparse depth tokens，$Q$ 是 image tokens。即 image tokens **从** sparse depth **取信息**，而不会把 image 信息反向写回 sparse depth——保护 prior 不被 overwrite。

#### Constrained Cross-View/Frame Propagation

之后还有一个 *constrained self-attention*：

$$\mathbf{X}_\ell^d = \text{Attn}_{\text{constrained}}(\tilde{\mathbf{X}}_\ell)$$

constrained 指的是 *attention mask* 只允许两类 connection：

1. **同一 timestamp 的相邻 cameras**（如 front + front-left，shared field of view）；
2. **同一 camera 跨所有 frames**（ego motion 下观察到相同 scene）。

非相邻 view（如 front 与 rear）不允许 attend，因为它们看到 disjoint regions of the world——强行建立 attention 会引入 geometric noise。最终 attention connectivity 是一个 sparse $T \times V$ pattern，aligned with physical co-visibility。

最终 feature：$\mathbf{F}_\ell = \text{cat}(\mathbf{X}_\ell, \mathbf{X}_\ell^d)$——**concatenation 而不是 replacement**，下游 head 同时看到原始 prior feature 和 LiDAR-conditioned feature。

#### Eq (4-5): Multi-Resolution LiDAR Prompt

$$\mathbf{P}^m = \text{cat}_l\big([\hat{\mathbf{D}}_l^{\text{sp}}, \hat{\mathbf{M}}_l^{\text{sp}}]\big)$$
$$\mathbf{P}^L = \text{cat}(\mathbf{P}^m, \mathbf{D}^{\text{prior}})$$

- $l$: resolution level 索引；
- $\hat{\mathbf{D}}_l^{\text{sp}}$: 把 sparse depth *先 downsample 到 level $l$*，再 *upsample 回 native resolution*。这样在低分辨率 level 上，sparse 点被"涂抹"开，prompt density 显著增加——decoder 浅层 feature map 分辨率低，单分辨率 sparse LiDAR 在那里几乎没有 valid prompt；
- $\hat{\mathbf{M}}_l^{\text{sp}}$: 对应 validity mask；
- $\mathbf{P}^L$: 完整 LiDAR prompt，concat 了 $\mathbf{D}^{\text{prior}}$，让 head 能直接 *比较* sparse LiDAR 与已有 depth estimate，做 refinement。

#### Eq (6): Scale Head Output

$$[\mathbf{S}_{\text{pix}}, \mathbf{C}] = H_{\text{scale}}(\{\mathbf{F}_\ell\}, \mathbf{P}^L)$$

- $H_{\text{scale}}$: PromptDA-style DPT decoder，但 repurposed 为预测 *scale* 而非 dense depth；
- $\mathbf{C}$: confidence map，用于 loss 中 downweight unreliable LiDAR projections；
- *zero-initialized convolutions* 注入 $\mathbf{P}^L$：保证训练初始时 LiDAR cues *完全不贡献*，模型从 unmodified backbone features 出发。

#### Eq (7): Scale Parameterization（关键 trick）

$$\mathbf{S}_{\text{pix}} = \exp\bigl(\alpha \cdot (2\sigma(\mathbf{x}) - 1)\bigr)$$

- $\mathbf{x}$: raw scale logits from decoder；
- $\sigma(\cdot)$: sigmoid function，$\sigma(\mathbf{x}) \in (0,1)$；
- $2\sigma(\mathbf{x})-1 \in (-1, 1)$: 把 sigmoid 拉伸到对称区间；
- $\alpha$: learnable coefficient，初始值 $0.5$；
- $\exp(\cdot)$: 确保最终 scale 始终为正。

**为什么这么设计——intuition**：

- **对称性 in log space**: 乘法 correction 在 linear space 不对称（乘 2 vs 乘 0.5 在 log space 距离不同）。取 $\log \mathbf{S}_{\text{pix}} = \alpha(2\sigma-1) \in (-\alpha, \alpha)$，对称且 bounded；
- **初始为 1**: $\sigma(0)=0.5 \Rightarrow 2\sigma-1 = 0 \Rightarrow \exp(0) = 1$，即训练起点 model 完全等于 DA3 输出。这对应 paper 标题里的 "minimal-intervention principle"；
- **Bounded correction**: 因为 $\alpha$ 可学且初始小，correction 是 progressive 的，不会一开始就大跨度破坏 prior。

### 3.3 Training Loss 全面拆解

总目标：

$$\mathcal{L} = \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{norm}} \mathcal{L}_{\text{norm}} + \lambda_{\text{scale}} \mathcal{L}_{\text{scale}}$$

三个 loss term 分别对应：metric anchoring、geometry preservation、scale smoothness。这种 *loss-level mediation* 是 paper 的另一个核心 contribution——直接把 geometry-scale conflict 表达成可微的 objective。

#### Eq (9): Multi-Resolution Sparse Depth Alignment

$$\mathcal{L}_{\text{sp}} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|\mathcal{V}_k'|} \sum_{p \in \mathcal{V}_k'} \bigl| \hat{D}_p^{c,(k)} - D_p^{\text{sp},(k)} \bigr|$$

- $K$: supervision resolution 数量；
- $k$: 每个 resolution level 索引；
- $p$: pixel index；
- $\hat{D}_p^{c,(k)}$: 预测 depth 在 level $k$ 的值；
- $D_p^{\text{sp},(k)}$: level $k$ 下的 sparse LiDAR depth；
- $\mathcal{V}_k$: level $k$ 所有 valid LiDAR pixels；
- $\mathcal{V}_k' \subseteq \mathcal{V}_k$: 只保留 per-pixel error 最小 80%（参考 MetricAnything [22]），过滤 misaligned projection。

**Intuition**: 多分辨率 supervision 解决 sparse LiDAR 在低分辨率 feature map 上的 *under-coverage* 问题；80% filtering 处理 LiDAR projection 本身的 noise（calibration drift, temporal offset, reflection）。

#### Eq (10): Confidence-Weighted Loss

$$\mathcal{L}_{\text{conf}} = \frac{1}{|\mathcal{V}|} \sum_{p \in \mathcal{V}} \bigl(C_p |\hat{D}_p^c - D_p^{\text{sp}}| - \lambda_c \log C_p\bigr)$$

- $\mathcal{V}$: native resolution valid LiDAR pixels；
- $C_p \in (0,1)$: 预测 confidence at pixel $p$；
- $\lambda_c$: confidence regularization 系数，防止 trivial all-zero $C$ solution。

这个 form 来自 DUSt3R [31] 的 confidence-weighted regression。直觉：让 model 学一个 *置信度*——当 LiDAR 投影到玻璃、镜面、远景等不可靠位置时，model 可以主动 *降低* $C_p$，于是那个像素对 loss 贡献小。$-\lambda_c \log C_p$ 是 entropy-like 项，防止 model 把所有 $C$ 都降到 0 来"作弊"最小化第一项。

#### Eq (11): Surface-Normal Regularization（geometry preservation 的关键）

$$\mathcal{L}_{\text{norm}} = \frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} \bigl(1 - \langle \mathcal{N}(\hat{\mathbf{D}}^c, \mathbf{K})_p, \mathcal{N}(\mathbf{D}^{\text{prior}}, \mathbf{K})_p \rangle\bigr)$$

- $\mathbf{K}$: camera intrinsics matrix（$3 \times 3$）；
- $\mathcal{N}(\mathbf{D}, \mathbf{K})$: 把 depth map unproject 成 3D 点云，再用 finite difference 计算每个 pixel 的 surface normal；
- $\langle \cdot, \cdot \rangle$: 单位 normal 之间的 inner product（cosine similarity）；
- $\mathcal{M}$: valid pixels excluding sky。

**这是 paper 最 elegant 的 regularization**：直接约束 *corrected depth 的 normal* 与 *prior depth 的 normal* 一致。这比直接 L2 约束 depth 差值更强，因为 normal 是 *一阶 spatial derivative*，捕捉的是 local surface orientation。如果只在 LiDAR 点位置 over-fit，neighboring pixels 会出现 *spike*，normal 会突变，立即被这个 loss 惩罚。

消融（Table 6）显示 $\lambda_{\text{norm}}=2$ 时在 LiDAR coverage 边界出现 *horizontal line*——即 surface 在 unsupervised 区域被破坏；$\lambda_{\text{norm}}=10$ 时过度约束，AbsRel 退化 0.6 个点。$\lambda_{\text{norm}}=5$ 是 sweet spot。

#### Eq (12): Scale Smoothness Regularization

$$\mathcal{L}_{\text{scale}} = \frac{1}{|\mathcal{M}|} \sum_{p} \bigl( |O_p| + |\partial_x O_p| + |\partial_y O_p| \bigr)$$

- $\mathbf{O} = \log \mathbf{S}_{\text{pix}}$: log-space scale map；
- $O_p$: pixel $p$ 处的 log-scale；
- $\partial_x, \partial_y$: horizontal / vertical finite differences。

两项惩罚：(1) $|O_p|$ 惩罚 deviation from unity（鼓励 scale 接近 1，即少改动 prior）；(2) $|\partial_x O_p| + |\partial_y O_p|$ 惩罚 scale 在 spatial 上 *rough*——鼓励 scale 在 sparse anchors 之间 smoothly 变化。

> 参考：[Metric3D v2](https://arxiv.org/abs/2312.06591), [Surface normal in Metric3D](https://arxiv.org/abs/2406.18112)

---

## 4. Architecture 模块间数据流总览

| 模块 | 输入 | 输出 | 可学参数？ |
|------|------|------|-----------|
| DA3 backbone (frozen) | $\mathcal{I}, \pi$ | $\mathbf{D}^{\text{prior}}$, $\{\mathbf{X}_\ell\}$ | 否 |
| Sparse-depth tokenizer | $\mathbf{P}^m$ | $\mathbf{Z}^d$ | 是 (lightweight CNN) |
| Cross-Attention | $\mathbf{X}_\ell, \mathbf{Z}^d$ | $\tilde{\mathbf{X}}_\ell$ | 是 |
| Constrained Self-Attn | $\tilde{\mathbf{X}}_\ell$ | $\mathbf{X}_\ell^d$ | 是 |
| Feature concat | $\mathbf{X}_\ell, \mathbf{X}_\ell^d$ | $\mathbf{F}_\ell$ | 否 |
| Scale Head $H_{\text{scale}}$ | $\{\mathbf{F}_\ell\}, \mathbf{P}^L$ | $[\mathbf{S}_{\text{pix}}, \mathbf{C}]$ | 是 (zero-init) |
| ROE Aligner | $\mathbf{D}^c, \mathbf{D}^{\text{sp}}$ | $s_g$ | 否 (closed-form) |

整个 learnable 部分（adapter + scale head）非常 lightweight，能在 *single 8-GPU node* 训 25 epochs / 4 days，远低于 end-to-end sparse-prompt baselines 的 dense depth retraining 成本。

---

## 5. 实验数据全面解读

### 5.1 主结果 Table 2: nuScenes 4F×6V

| Method | Setting | AbsRel↓ | δ₁↑ | δ₀.₅↑ | AbsIn02↑ | EdgeCR↑ | RevEdgeCR↑ |
|--------|---------|---------|------|--------|----------|---------|------------|
| Depth Pro | Mono, no LiDAR | 16.86 | 76.47 | 62.90 | 29.93 | 5.404 | 2.381 |
| MOGE-2 | Mono, no LiDAR | 13.78 | 83.64 | 72.25 | 37.97 | 5.226 | 2.393 |
| PromptDA | Mono, +LiDAR | 45.55 | 29.56 | 15.78 | 4.56 | 1.531 | 1.385 |
| PriorDA | Mono, +LiDAR | 08.25 | 90.87 | 77.88 | 30.62 | 2.524 | 2.069 |
| DA3 | 4F×6V, no LiDAR | 15.88 | 84.48 | 72.22 | 30.68 | 7.745 | 2.597 |
| MapAnything | 4F×6V, +LiDAR | 11.99 | 92.10 | 86.88 | 50.43 | 1.914 | 1.561 |
| **DrivingDepth** | 4F×6V, +LiDAR | **11.19** | 89.96 | 85.22 | **61.14** | **5.741** | **2.273** |

关键观察：

1. **PriorDA 的 trade-off 灾难**: AbsRel 最低（8.25），但 EdgeCR 跌到 2.524——完美印证"naive fit LiDAR 破坏 image structure"的论断。它的 $\delta_1$ 高达 90.87，但 AbsIn02 仅 30.62，说明整体 depth error 小但是在 boundary / structure 上完全跑偏。

2. **MapAnything 的 boundary 失守**: AbsRel 11.99 看着不错，但 EdgeCR 只有 1.914——比 DA3（7.745）跌了 75%。MapAnything "regenerate from scratch" 的代价就是 RGB-depth alignment 大幅退化。

3. **DrivingDepth 的双赢**: AbsRel 11.19（比 MapAnything 略好），EdgeCR 5.741（比 MapAnything 高 3 倍）。**AbsIn02 高达 61.14，比 MapAnything 高 10+ points**——这说明 DrivingDepth 在 *0.2m 精度内* 的像素比例大幅领先，这是 occupancy grid 这种下游任务最关心的指标。

4. **PromptDA 失败的启示**: AbsRel 45.55，几乎不可用。原因：PromptDA 假设 dense prompt，driving LiDAR 太 sparse 了，model 完全没法学。这印证 paper 选择 multi-resolution prompt 的必要性。

### 5.2 Sparse Robustness Table 3（10% vs 100% LiDAR）

nuScenes 上：
- MapAnything @100%: AbsIn02 = 50.43, EdgeCR = 1.914
- MapAnything @10%: AbsIn02 = 35.11（跌 15+）
- DrivingDepth @10%: AbsIn02 = 54.66（仍超 MapAnything @100%）
- DrivingDepth @10% EdgeCR = 5.757

**这是 paper 最强的论据之一**：DrivingDepth 在只用 10% LiDAR 点的情况下，metric accuracy 仍然超过 MapAnything 用 100% LiDAR 的表现。说明：
- Multi-resolution prompt encoding 把 sparse 信号放大有效；
- Confidence-weighted loss 自动 downweight 噪声点；
- 整体 design philosophy "LiDAR as prompt not dense supervision" 在 sparse regime 下鲁棒性远胜 regenerate-from-scratch 方法。

DDAD 上更夸张：MapAnything 在 100% LiDAR 下 AbsRel 反而比 10% 时更差（09.56 vs 09.91）——*dense noisy points 反而 hurt MapAnything*，因为它倾向于 rigidly fit。这进一步验证了 paper 的核心论点。

### 5.3 Input Context Table 4

| Frames | Interval | AbsRel↓ | δ₁↑ | AbsIn02↑ |
|--------|----------|---------|------|----------|
| 4 | 1 | 11.19 | 89.96 | 61.14 |
| 6 | 1 | 11.27 | 90.02 | 61.02 |
| 4 | 2 | 11.37 | 89.93 | 60.21 |
| 1 | – | 11.86 | 88.56 | 49.76 |

关键发现：

1. **Single-frame 仍然有效**: 1 frame 时 AbsRel = 11.86，仅比 4-frame 差 0.67 个点。证明 *pixel-wise scale correction 本身就是有效机制*，不依赖 multi-view fusion；

2. **Multi-frame 的价值在 AbsIn02**: 单 frame AbsIn02 = 49.76，4-frame 跳到 61.14（+11.38）——多 frame modeling 在 *strict 0.2m 评估* 下显著提升，但在 relaxed metric（AbsRel）下增益小。说明 cross-frame information 主要修正 *fine-grained 结构*，不是 scale；

3. **Robustness to temporal spacing**: interval=1 vs 2 几乎无差（11.19 vs 11.37），说明 model 不依赖严格时序相邻。

### 5.4 Component Ablation Table 5

| Variant | Mul-Res | GPFA | AbsRel↓ | δ₁↑ | EdgeCR↑ |
|---------|---------|------|---------|------|---------|
| Full | √ | √ | 11.19 | 89.96 | 5.741 |
| w/ CVA (vanilla attn) | √ | √ | 11.38 | 89.56 | 5.771 |
| w/o GPFA | √ | × | 12.49 | 88.49 | 5.825 |
| w/o Mul-Res | × | √ | 13.13 | 87.80 | 5.787 |
| w/o LiDAR | – | × | 13.39 | 87.46 | 5.776 |

关键 takeaway：

1. **w/o LiDAR**: AbsRel = 13.39——这是 *frozen DA3 + lightweight finetune* 的 baseline。和有 LiDAR 的 11.19 差 2.2 个点，是 LiDAR prompt 的纯贡献；

2. **w/o Mul-Res** 跌 2 个点：single-resolution sparse LiDAR 在低分辨率 decoder feature 上几乎没有信号，multi-resolution 是 *dense enough to be useful* 的关键；

3. **w/o GPFA** 跌 1.3 个点：feature-level injection 比 output-level only 更有效，因为 backbone intermediate layers 能融合 sparse depth 与 dense visual geometry；

4. **vanilla CVA vs constrained**: 差 0.19 个点——constrained attention 比 vanilla 略好，但更重要的是 paper 指出 vanilla "blindly establishes correlations between all camera pairs, bringing geometric noise"—— qualitative 上更重要；

5. **EdgeCR 几乎不变（5.75–5.83）**: 这是 *by construction* 的体现——backbone frozen + scale init = 1，所以 boundary consistency 永远 close to DA3 prior，与 metric calibration 模块无关。这是 paper architecture design 的核心保证。

### 5.5 Surface-Normal Weight Trade-off Table 6

| $\lambda_{\text{norm}}$ | AbsRel↓ | δ₁↑ | AbsIn02↑ | EdgeCR↑ |
|---|---------|------|----------|---------|
| 2 | 10.16 | 91.38 | 62.64 | 5.667 |
| 5 | 11.19 | 89.96 | 61.14 | 5.741 |
| 10 | 11.79 | 89.27 | 58.63 | 5.776 |

非常 *smooth* 的 trade-off：
- $\lambda_{\text{norm}}=2$: AbsRel 最优（10.16）但 EdgeCR 最低（5.667）——LiDAR fitting 激进，boundary 出现 artifact（图 5 显示 horizontal line at LiDAR coverage boundary）；
- $\lambda_{\text{norm}}=10$: AbsRel 退化到 11.79，但 EdgeCR 最高——surface 被过度约束；
- $\lambda_{\text{norm}}=5$: sweet spot。

这 *直接* 把 geometry-scale conflict 表达为一个 *continuous knob*——loss weight 是 conflict 的可调参数，而不是 binary 的设计选择。

---

## 6. 评估 Metric 解读

paper 引入三个 non-standard metric，专门 measure geometry-scale conflict 的不同侧面：

| Metric | Formula | 含义 |
|--------|---------|------|
| AbsIn02↑ | $\frac{1}{N}\sum \mathbf{1}[|\hat{d}_i - d_i| < 0.2\text{m}]$ | 0.2m 内像素比例，反映 occupancy grid 所需精度 |
| EdgeCR↑ | $\frac{\text{mean}_{\mathcal{E}_I}(G_D)}{\text{mean}_{\bar{\mathcal{E}}_I}(G_D)+\epsilon}$ | RGB edge 处 depth gradient 强度 vs 其他区域——forward 检查 RGB edge 是否产生 depth change |
| RevEdgeCR↑ | $\frac{\text{mean}_{\mathcal{E}_D}^*(G_I)}{\text{mean}_{\bar{\mathcal{E}}_D}^*(G_I)+\epsilon}$ | depth edge 处 RGB gradient 强度——reverse 检查 depth edge 是否有 image structure 支撑 |

- $G_I = |\nabla I|$: RGB image 的 Sobel gradient；
- $G_D = |\nabla \hat{D}|$: depth map 的 Sobel gradient；
- $\mathcal{E}_I = \text{Top}_{10\%}(G_I)$: top 10% RGB edge pixels；
- $\mathcal{E}_D = \text{Top}_{10\%}(G_D)$: top 10% depth edge pixels；
- $\epsilon$: 防止除零的小常数；
- $\text{mean}^*$: 排除对方 edge 区域后的 mean（避免 circularly 重叠）。

这两个 metric *双向* 测量 image-depth boundary alignment——传统 metric（如 AbsRel）只 measure metric accuracy，忽略 boundary 一致性，所以 MapAnything 看着 AbsRel 不错但实际点云拼接出来 boundary 撕裂。DrivingDepth 引入这两个 metric 的动机就是 *把 geometry 量化*，否则 community 会继续 ignore geometry-scale conflict。

---

## 7. Intuition 总结：为什么这个方法 Work

把 paper 的核心 intuition 压成几句话：

1. **Foundation model 已经 capture 了相对几何**：DA3 在大量 multi-view 数据上预训练，已经知道 surface continuity、edge structure、relative depth——这部分 *不用再学*；
2. **Metric scale 是 spatially varying 的低自由度信号**：相对 depth 已经对，剩下的只是 per-pixel "stretch factor"——这个 map 比 dense depth 简单得多，可以用 sparse LiDAR 直接 anchor；
3. **By construction 优于 by loss**：与其在 loss 里加 regularization 防止 prior 被 overwrite，不如 *frozen backbone + neutral init* 让 overwrite 在数学上不可能；
4. **Sparse 数据需要 multi-resolution 处理**：driving LiDAR 在 native resolution 太稀疏，必须 downsample 让低分辨率 decoder stage 也能看到有效 prompt；
5. **Confidence 学习比硬过滤好**：与其 hand-craft filter noisy LiDAR 点，不如让 model 学一个 $C_p$，end-to-end 决定哪些点可信。

这五点合起来，*把 geometry-scale conflict 从一个 architectural dilemma 变成可微 optimization problem*。

---

## 8. Limitation 与扩展方向

paper 自己提到的 limitation：如果 foundation model 几何本身 wrong（reflective / transparent surface），scale correction 无法 recover missing structure——因为 frozen backbone 没给正确 prior，乘以 scale 还是错的。

可能的 extension 方向（paper 未明说但可联想）：

1. **选择性 unfreeze**: 对 transparent / reflective 区域，detect prior 不可靠，selectively fine-tune backbone 局部；
2. **Dynamic object handling**: moving truck / dynamic vehicle 场景里（图 4 green crop），跨 frame attention 会 confused。可以引入 flow / object mask 分支；
3. **Metric learning inside scale head**: 现在 $s_g$ 是 ROE 算的 closed-form，不参与 backprop。可以设计一个 *可学 but constrained* 的 $s_g$，让它参与 backprop 但加 strong regularization；
4. **Multi-foundation fusion**: paper 只用 DA3。可以 ensemble DA3 + VGGT + Metric3D v2，让 prior 更 robust；
5. **Uncertainty quantification**: 现在 confidence $C$ 只用于 loss，inference 时可以输出 *depth uncertainty map*，给下游 planner 用；
6. **Long-range scale drift**: $s_g$ 是 clip-level shared，但长 sequence 上会 drift。可以加 recurrent state 让 $s_g$ 跨 clip 平滑。

---

## 9. 个人感想：Design Philosophy

这个 paper 给我的最大启发是 *minimal-intervention principle* 在 deep model design 中的应用。绝大部分 sparse-prompt / multi-modal fusion 方法默认 "regenerate everything end-to-end"，理由是 "let model learn the optimal fusion"。但这种方法在 sparse / noisy prompt 下失败，因为 model 没有 enough signal to override foundation prior，于是只能 *partially* override，产生 inconsistent hybrid。

DrivingDepth 反其道而行：*assume foundation is right*, 只学一个最小 correction。这种 *bias toward prior* 是一种 strong inductive bias，特别适合 sparse supervision regime。可以类比 Bayesian inference 中 *strong prior + sparse likelihood* 的情形——posterior 几乎等于 prior 加小修正。

这种思路在 RL 里有类似 Imitation Learning + small residual policy（如 ResNet policy, Residual Policy Learning）；在 NLP 里类似 LoRA / Adapter 的 *minimal update*；在 robotics 里类似 *operating space control* on top of a nominal controller。

paper 把这个 idea 用在 depth estimation 上，并且通过 specific architecture choice（frozen backbone + zero-init injection + unit-init scale + log-space parameterization）*guarantee* minimal intervention by construction——这是 engineering 上很 elegant 的地方。

> 相关参考：
> - [Residual Policy Learning](https://arxiv.org/abs/1812.06298)  
> - [LoRA](https://arxiv.org/abs/2106.09685)  
> - [Adapter Tuning](https://arxiv.org/abs/1902.00751)

---

## 10. 相关联想：与最近工作的位置

DrivingDepth 在 2025–2026 这波 *feed-forward 3D reconstruction* 工作中的定位：

| 工作 | 范式 | 与 DrivingDepth 关系 |
|------|------|---------------------|
| [DUSt3R / MASt3R](https://arxiv.org/abs/2312.14132) | pointmap regression, 2-view | DrivingDepth 借鉴其 confidence-weighted loss |
| [VGGT](https://arxiv.org/abs/2503.11651) | unified multi-view transformer | 与 DrivingDepth 用 multi-view，但无 metric |
| [CUT3R](https://arxiv.org/abs/2505.12232) | continuous 3D perception with state | 处理时序，DrivingDepth 用 constrained cross-frame attention |
| [π³](https://arxiv.org/abs/2507.13347) | permutation-equivariant geometry | 处理任意 view 顺序，DrivingDepth 假设固定 surround-view |
| [DepthAnything3](https://arxiv.org/abs/2506.19257) | visual space from any views | DrivingDepth 的 foundation base |
| [MapAnything](https://mapanything.github.io/) | universal feedforward metric 3D | DrivingDepth 的直接 baseline / 对手 |
| [PromptDA](https://arxiv.org/abs/2503.19013) | prompting depth anything | DrivingDepth 借鉴 DPT head 结构 |
| [PriorDA](https://arxiv.org/abs/2506.06720) | depth anything with any prior | mono baseline，naive fit |
| [MetricAnything](https://arxiv.org/abs/2601.22054) | scaling metric depth pretraining | DrivingDepth 借鉴 80% filtering |
| [DVGT](https://arxiv.org/abs/2503.11060) | driving visual geometry transformer | driving-specific 几何 transformer |
| [DGGT](https://arxiv.org/abs/2603.04535) | feedforward 4D dynamic driving | 处理动态场景，DrivingDepth 静态 |
| [DrivingForward](https://arxiv.org/abs/2412.07834) | feedforward 3DGS surround-view | Gaussian 表示 vs point cloud |
| [VGD](https://arxiv.org/abs/2510.19578) | visual geometry gaussian | 类似 DrivingForward |
| [StreetForward](https://arxiv.org/abs/2603.19552) | feedforward causal dynamic street | causal attention 处理动态 |
| [DynamicVGGT](https://arxiv.org/abs/2603.08254) | dynamic point maps 4D | 4D extension of VGGT |
| [Dist-4D](https://arxiv.org/abs/2411.16079) | disentangled spatiotemporal diffusion | generative 路线 |
| [InfiniDepth](https://arxiv.org/abs/2503.17764) | arbitrary resolution depth | 用 neural implicit field |
| [FastVGGT](https://arxiv.org/abs/2509.02560) | training-free VGGT acceleration | inference 加速 |
| [SCAL3R](https://arxiv.org/abs/2604.08542) | scalable test-time training | test-time adaptation 路线 |
| [TTT3R](https://arxiv.org/abs/2509.26645) | test-time training for 3D recon | 类似 SCAL3R |
| [UniDepth](https://arxiv.org/abs/2403.18931) | universal monocular metric | metric foundation 但 mono |
| [CompletionFormer](https://arxiv.org/abs/2305.07366) | depth completion with conv+ViT | traditional depth completion 路线 |
| [M²Depth](https://arxiv.org/abs/2403.17649) | self-supervised 2-frame multi-cam | self-supervised 路线 |
| [SurroundDepth](https://arxiv.org/abs/2210.07906) | entangling surround views | surround-view self-supervised |
| [Metric3D v2](https://arxiv.org/abs/2406.18112) | metric + normal foundation | metric + normal 联合预训练 |

DrivingDepth 的 *独特生态位*：在 foundation model 已经 strong 的前提下，做一个 *minimal correction module* 把 metric scale 加进去，且不破坏 foundation geometry。这是介于 "纯 foundation" 与 "end-to-end fusion" 之间的 *第三条路*。

> 参考：
> - DrivingDepth 项目页（如已开源）：可搜索 "DrivingDepth Baidu"  
> - [DepthAnything3 官方仓库](https://github.com/DepthAnything/Depth-Anything-V2)（DA3 是 V3）  
> - [nuScenes 数据集](https://www.nuscenes.org/)  
> - [DDAD 数据集](https://github.com/TRAILab/DDAD)  
> - [MoGe / MOGE-2](https://arxiv.org/abs/2410.24120)  
> - [PromptDA](https://github.com/DepthAnything/PromptDA)  
> - [VGGT](https://vgg-t.github.io/)  
> - [DUSt3R](https://dust3r.europe.naverlabs.com/)  
> - [MapAnything](https://mapanything.github.io/)

---

## 11. 总结一句话

DrivingDepth 把 autonomous driving 的 depth estimation 问题重新表述为：*given a frozen geometric prior, learn only a per-pixel multiplicative scale map from sparse LiDAR prompts*。通过 *frozen backbone + neutral init + multi-resolution prompt + confidence weighting + surface-normal regularization* 这套组合拳，让 foundation model 的 dense visual geometry *by construction* 被保留，sparse LiDAR 只 anchor metric scale——在一个 8-GPU 节点上以 1/4 的训练成本跑出比 MapAnything 更好的 metric accuracy + 3 倍的 EdgeCR，且在 10% LiDAR density 下仍 robust。

这 paper 的核心贡献 *不是一个新 module*，而是一种 **problem reformulation**：把 geometry-scale conflict 从 "dilemma" 变成 "decomposition"，剩下的就是工程实现。这种 *minimal-intervention* 的设计哲学在 foundation model 时代会越来越重要——当 foundation 已经足够好，我们的工作不再是 replace it，而是 *calibrate* it。
