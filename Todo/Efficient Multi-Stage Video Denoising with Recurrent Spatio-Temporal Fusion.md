---
source_pdf: Efficient Multi-Stage Video Denoising with Recurrent Spatio-Temporal Fusion.pdf
paper_sha256: 5fe7c9e271868cd6b692967e72531f10199bb48240ea641a77bdc439f3b075cb
processed_at: '2026-08-04T01:48:32-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EMVD 用人话讲

好嘞,咱们把前面那堆 LaTeX 和图表先放一边,我用大白话从头捋一遍。这篇 paper 我觉得是 CVPR 2021 里最值得反复琢磨的一个 — 不是因为它有多 fancy,而是因为它把 "怎么用 architecture 表达 task 的内部分解" 这件事做得很干净。

paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf
arXiv mirror: https://arxiv.org/abs/2103.05732

---

## 1. 这 paper 一句话讲清楚

**手机拍夜景视频噪声大,以前 SOTA 方法需要 2000 GFLOPs 才能去干净,他们用 5 GFLOPs 就跑出差不多甚至更好的效果,还能在 Huawei P40 Pro 上 30fps 实时跑 720p。**

就是这么个事儿。下面拆开看怎么做到的。

---

## 2. 视频 denoise 难在哪

你拍夜景,传感器收到的光子数少,信噪比很差。噪声模型大概长这样:

$$z_t(x) = y_t(x) + \eta_t(x), \quad \eta_t \sim \mathcal{N}(0, \sigma_t^2(y_t))$$

- $z_t$: 你拿到的 noisy frame
- $y_t$: 你想要的 clean frame(地面真值,未知)
- $\eta_t$: 噪声
- $\sigma_t^2(y_t) = a_t \cdot y_t + b_t$: 噪声方差是信号的函数,亮的地方噪声大(shot noise),暗的地方也有底噪(read noise)

参考 noise model: https://ieeexplore.ieee.org/document/4491068 (Foi 2008,TPAMI)

视频 denoise 跟单帧 denoise 的差别在于:视频有时间维度,相邻几帧画面几乎一样,**多帧平均一下,噪声方差除以 $\sqrt{N}$**。这是物理定律送你的免费午餐。

但这里有两个坑:
1. **运动**: 如果画面里有东西在动,你直接平均就会 ghost。要先对齐,或者只平均静止部分。
2. **盲 denoise 性能差**: 如果 denoiser 不知道当前噪声水平,性能比 non-blind 差很多(后面 ablation 会看到差 1.24dB)。

EMVD 就是围绕这两个坑设计的。

---

## 3. 三个 stage,各干一件事

EMVD 把 video denoising 拆成三个 sub-task,每个用一个小网络干:

### Stage 1: Fusion — 跨帧平均去噪

这个 stage 干的事:**把过去所有帧跟当前帧做个加权平均,权重告诉它哪里静止哪里动**。

公式:
$$\bar{y}_t(x) = \bar{y}_{t-1}(x) \cdot \bar{\gamma}_{t-1}(x) + z_t(x) \cdot \gamma_t(x)$$

- $\bar{y}_t$: 当前时刻的 fused output(recurrent state,一直往前滚)
- $\bar{y}_{t-1}$: 上一时刻的 fused output
- $z_t$: 当前 noisy frame
- $\gamma_t \in [0,1]$: 当前帧的权重
- $\bar{\gamma}_{t-1} = 1 - \gamma_t$: 上一帧的权重(凸组合约束)

这玩意儿其实就是一个 **per-pixel 的 EMA(Exponential Moving Average),但权重是 spatially adaptive 的**。

权重怎么来? 一个小 CNN (FCNN) 预测:

$$\{\gamma_t, \bar{\gamma}_{t-1}\} = \text{FCNN}\Big(|z_{LL|t} - \bar{y}_{LL|t-1}|, \hat{\sigma}_t^2\Big)$$

- 输入 1: 当前 noisy low-pass subband $z_{LL|t}$ 跟前一 fused low-pass $\bar{y}_{LL|t-1}$ 的 **绝对差**。这就是个 motion indicator — 差小就是静止,差大就是动。
- 输入 2: 噪声方差 $\hat{\sigma}_t^2$
- 输出: 通过 sigmoid 保证凸组合

paper Figure 3 里可视化了这个 $\gamma_t$ — 静止区域蓝色(两边都给 0.5),运动区域红色(主要给当前帧)。这玩意儿完全是 emergent 的,loss 里没显式监督它"区分动静"。

参考 kernel prediction networks 思路: https://arxiv.org/abs/1712.02358 (Mildenhall et al.)

### Stage 2: Denoising — 空间去噪

Fusion 把噪声干掉一大半,但没干净(运动区域没法平均)。所以再过一个小 CNN:

$$\tilde{y}_t = \text{DCNN}\Big(\bar{y}_t, z_{LL|t}, \bar{\sigma}_t^2\Big)$$

- $\bar{y}_t$: fused image(还残留噪声)
- $z_{LL|t}$: 当前 noisy frame 的 low-pass(原始未污染信息,作为细节 reference)
- $\bar{\sigma}_t^2$: **fused image 的噪声方差**(精确递推出来,见下一节)

**为什么 input 要塞 $z_{LL|t}$**: ablation 显示去掉这个 input PSNR 掉 0.76dB。因为 fusion 可能把细节模糊掉,DCNN 需要原始 noisy 信号作为"细节在哪里"的提示。

### Stage 3: Refinement — 把高频捞回来

DCNN 去噪完,细节通常被过度平滑。Refinement 做一个自适应混合:

$$\hat{y}_t(x) = \bar{y}_t(x) \cdot \bar{\omega}_t(x) + \tilde{y}_t(x) \cdot \tilde{\omega}_t(x)$$

- $\bar{y}_t$: fused image(细节全但还有噪声)
- $\tilde{y}_t$: denoised image(干净但糊)
- $\omega_t$: 也是 sigmoid 保证凸组合
- $\hat{y}_t$: 最终输出

paper Figure 4 可视化 $\omega_t$: 边缘/纹理区域大量给 $\bar{y}_t$,平坦区域大量给 $\tilde{y}_t$。也是 emergent 的。

**这 stage 跟 fusion 公式同构,但语义完全不同**:fusion 是"跨帧信任多少",refinement 是"帧内高频信任多少"。架构层面共用一个数学模板,任务层面完全不同。

---

## 4. Variance Propagation — 我觉得这是全 paper 最骚的操作

OK 现在说说我最喜欢的部分。

DCNN 怎么知道当前 fused image 还剩多少噪声?**因为 fusion 是线性的,所以方差可以 closed-form 递推**:

$$\bar{\sigma}_t^2 = \bar{\gamma}_{t-1}^2 \cdot \sigma_{t-1}^2(\bar{y}_{LL|t-1}) + \gamma_t^2 \cdot \sigma_t^2(z_{LL|t})$$

- 第一项: 上一帧 fused 的方差,衰减 $\bar{\gamma}_{t-1}^2$
- 第二项: 当前 raw 帧的方差
- 协方差项为 0(假设时序噪声独立)

直觉推导:如果 $\text{Var}[X] = \sigma_X^2$, $\text{Var}[Y] = \sigma_Y^2$,那 $\text{Var}[aX+bY] = a^2\sigma_X^2 + b^2\sigma_Y^2 + 2ab\text{Cov}(X,Y)$。最后那项 Cov 假设是 0,剩下两项就是上面公式。

**这玩意儿随时间单调非增**(因为 $\gamma \leq 1$),数学上保证 fusion 越融合越干净。

为什么这个 **值 1.24 dB**?看 ablation:

| 配置 | PSNR |
|---|---|
| 完整 EMVD (with variance input) | 42.63 |
| 去掉所有 variance input (blind) | 41.39 |

差 1.24 dB,这在 raw denoise 里是巨大差距。**raw domain denoising 中,blind vs non-blind 是性能鸿沟**。EMVD 通过 closed-form propagation 把这个 advantage 免费拿到了 — 不用学,不用估计,数学直接给你。

参考 Kalman filter 类似的 covariance propagation: https://en.wikipedia.org/wiki/Kalman_filter

**我自己的直觉**:这玩意儿跟 Kalman filter 的精神是相通的 — Kalman 也是线性 + 高斯假设下 covariance 闭式传播。EMVD 在 video denoising 上做了一个类似的事情,只不过不是预测状态,而是预测噪声水平。这种 "把不确定性显式 propagate 给下游" 的思路在 visual computing 里其实被低估了 — 现在大家一股脑搞 end-to-end black box,把不确定性信息藏在了 latent 里。

---

## 5. Transform Domain — 顺便砍 4 倍算力

EMVD 不在 pixel domain 操作,先做两个 invertible transform。

### Color transform $\mathcal{T}_c$

Bayer raw 有 4 个通道 ($R, G_1, G_2, B$)。$G_1$ 跟 $G_2$ 几乎一样,$R$ 跟 $B$ 也有强相关。1×1 conv 学一个 $4 \times 4$ 矩阵,把它们变成 luminance + chrominance。

Invertibility 通过 loss 强制:
$$\mathcal{L}_c = \|M \cdot M' - I_4\|_F^2$$

- $M$: forward color matrix
- $M'$: inverse color matrix(不 share weight,biorthogonal)
- $I_4$: 4×4 单位矩阵

### Frequency transform $\mathcal{T}_f$

学一个类似 Haar wavelet 的东西,把 spatial 砍成 4 个 subband (LL, LH, HL, HH),分辨率砍半。

- Forward: strided conv, kernel 从 1D wavelet filter $\psi$ 初始化
- Inverse: transposed conv, kernel 从 reconstruction filter $\phi$ 初始化
- Invertibility: $\mathcal{L}_f = \|\psi \cdot \phi^\top - I_2\|_F^2$

**为什么这步能省算力**:
1. 分辨率砍半 → 每层 conv FLOPs 砍 4×
2. 能量集中到 LL,high-freq subband 信号稀疏,网络不用学怎么区分高低频
3. 多尺度递归(用 3 个 scale)进一步压缩感受野需求

**为什么这步还能涨精度**: decorrelate 之后,denoiser 不用浪费容量去学习通道间冗余。Ablation 显示去掉 color transform PSNR 掉 0.27dB,用 pixel shuffle 替代 frequency transform 掉 0.28dB。

参考 wavelet CNN 思路: https://arxiv.org/abs/1805.07071 (Liu et al. CVPRW 2018)

---

## 6. Recurrent vs Multi-frame — 一笔关键账

EMVD 是 recurrent (用前一帧 fused 状态),不是 multi-frame (堆 5/7 帧一次性算)。

**Recurrent 的好处**:
- FLOPs 跟历史长度无关,不管累积了多少帧,计算量恒定
- 长序列 PSNR 单调上升(paper Figure 7)
- 内存友好,只需要存一个 fused state

**Multi-frame 的好处**:
- 能看 future frames,在 occlusion / 大运动场景短暂占优
- 训练简单(不用 BPTT)

paper Figure 7 数据特别有意思:在 SRVD MOT17-01 序列上,前 24 帧 EMVD 的 ΔPSNR 一直爬升,比 RViDeNet / FastDVDnet / EDVR 都高。frame 25-30 之间有动态物体,这三个 multi-frame 方法短暂反超(因为它们能看 future),但 EMVD 几帧就 recover 回来。

这个 trade-off 在 **real-time 部署场景下根本没得选** — 你不可能等 future frames。所以 EMVD 的 recurrent 设计在 mobile 上是必选项。

参考 recurrent VSR 思路: https://arxiv.org/abs/1711.06005 (Sajjadi et al. CVPR 2018)

---

## 7. 数据说话:跟 SOTA 怎么 trade-off

看 Table 3 (CRVD dataset):

| Model | GFLOPs | PSNR raw |
|---|---|---|
| EDVR | 3088.98 | 44.71 |
| RViDeNet | 1965.04 | 44.08 |
| FastDVDnet | 664.99 | 44.30 |
| **EMVD (79)** | **79.52** | **44.05** |
| FastDVDnet† | 22.16 | 42.25 |
| **EMVD (5)** | **5.38** | **42.63** |

关键观察:
- 79 GFLOPs 的 EMVD 跟 1965 GFLOPs 的 RViDeNet 差距只有 **0.03 dB**(算力比 1:25)
- 5.38 GFLOPs 的 EMVD 比 22.16 GFLOPs 的 FastDVDnet† 高 **0.38 dB**,算力还少 4×

Table 4 实测在 Huawei P40 Pro 上 720p 跑:

| Model | GFLOPs | Time (ms) | DDR (MB) | PSNR |
|---|---|---|---|---|
| FastDVDnet† | 22.16 | 177 | 724 | 37.43 |
| EMVD | 5.38 | **36** | **112** | **38.27** |

36 ms 一帧,27.8 fps,real-time 拿下。比 FastDVDnet† 快 4.9×,内存少 6.5×,精度还高 0.84 dB。

**这种 1-2 个数量级的效率提升,不是靠把网络砍小得到的,是靠重新设计架构把 inductive bias 焊死在 pipeline 里得到的**。

---

## 8. 我自己的 take:这 paper 为什么重要

### 8.1 Inductive bias is compute

你看 EMVD 三 stage:
- Fusion: 线性 + 凸组合 → closed-form variance
- Denoising: 非线性 CNN,但 input 拿到精确 noise level
- Refinement: 线性混合 → 把高频捞回来

**每个 stage 都对应 video denoising 的一个明确的子问题**。如果你不用这种分解,让一个 monolithic CNN 从头学,它得自己学:(1) 区分动静;(2) 估计噪声水平;(3) 保留高频细节。每多学一件事,就要多花算力。

EMVD 把这三件事的 inductive bias 全部硬编码到 architecture 里,网络只学"具体权重怎么定"。这就是为什么 5 GFLOPs 能跑赢 1965 GFLOPs — 算力省下来不是靠 trick,是靠"任务结构已知,不要让网络重新 discover"。

参考我喜欢的另一篇类似思路: https://arxiv.org/abs/1609.05158 (DnCNN,把 residual learning 作为 inductive bias)

### 8.2 显式 uncertainty propagation 被严重低估

EMVD 用 closed-form 把方差传给下游,这件事让我想起 Kalman filter。在 modern deep learning 时代,大家喜欢端到端,uncertainty 都藏在 latent 里。但在 sensor pipeline、ISP、robotics 这种需要 reliable、可解释、可调试的场景,显式 uncertainty 还是非常有价值的。

我自己的联想:
- **Diffusion models 的 $\sigma_t$ schedule**: Diffusion 里每步知道噪声水平是关键的。EMVD 的思路跟 diffusion 在精神上有点像 — 都是"显式 track 噪声水平"。
- **NeRF 的不确定性**: NeRF 现在一般用 density 表达 occupancy,但 density 不是 uncertainty。如果用 EMVD 这种 closed-form covariance propagation,可能能做更可靠的 few-shot NeRF。
- **Self-supervised learning**:Noise2Noise 类思路(Mapping 不确定性)其实跟 EMVD 的 fusion 是一脉相承的。

参考 Noise2Noise: https://arxiv.org/abs/1803.04189

### 8.3 Recurrent + Multi-scale 在 mobile 上是黄金组合

mobile NPU 有几个特点:(1) memory bandwidth 比 compute 更稀缺;(2) 不喜欢大 kernel;(3) 不喜欢 dynamic shape。

EMVD 的设计正好踩在这些点上:
- recurrent 只存一个 state,内存 footprint 极小
- 多尺度下分辨率砍半,大感受野用低 scale 实现
- 1×1 fusion kernel 在 NPU 上很友好

这给我一个直觉:**在 mobile 部署场景,architecture 设计要顺着硬件走,逆着走是死路**。RViDeNet 那种 7-frame 堆叠 + 大网络在 server 上跑得欢,搬到手机就死。

参考 mobile AI benchmark: https://ai-benchmark.com

### 8.4 Limitations 和 follow-up 方向

我看完 paper 想了几个可能的发展方向:

1. **Motion 大时 EMVD 还是会吃亏**: Table 2 显示 kernel prediction 收益有限,说明 implicit motion compensation 有天花板。结合 deformable conv 做 explicit alignment 可能是 follow-up。
2. **Future frame 利用**: 现在 EMVD 是 causal 的。offline 场景下能不能做个 bidirectional EMVD?
3. **Transform 的可解释性**: paper 没可视化 $M$ 学到了什么 color basis。如果能像 wavelet 那样有明确语义,可能能用于 compression / HDR pipeline 之类下游任务。
4. **跟 diffusion 结合**: Diffusion-based denoising 在 raw domain 已经开始兴起(比如 https://arxiv.org/abs/2306.02985 这类)。EMVD 的 closed-form variance propagation 能不能作为 diffusion 的 conditioning?
5. **Self-supervised 训练**: 现在 EMVD 需要 ground-truth clean video (CRVD dataset)。能不能用 Noise2Noise 思路做 self-supervised 训练?这点对部署很重要,因为真实场景下没有 GT。
6. **Transformer 替代**: 三个 CNN 都很小。如果换成 linear attention 之类的轻量 transformer,能不能进一步提精度?

---

## 9. 总结

人话版:

**EMVD 就是把 video denoising 这个 task 的内部结构看透了,然后用 architecture 把每个子问题对应的小网络焊在一起,顺便用 wavelet domain 把算力砍 4 倍,用 closed-form variance propagation 让 denoiser 知道当前噪声水平,最后用 recurrent 设计让信息能跨任意长序列累积。这三招加一起,5 GFLOPs 干掉了 2000 GFLOPs 的 SOTA。**

这 paper 给我的最大启示:**架构层面的 inductive bias 比 network 容量更值钱**。同样的算力,你让它学有结构的东西,它就能学得好;你让它从头 discover 结构,算力就浪费在 re-learning 已知规律上了。

这个道理跟 Karpathy 你自己常说的 "surgical architecture design" 思路是一致的 — micrograd、nanoGPT 那种 "把 task structure 写进代码" 的精神,在 EMVD 这里得到了 video denoising 上的体现。

---

## 10. 一些可能你想继续挖的 reference

- paper 主页: https://openaccess.thecvf.com/content/CVPR2021/html/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.html
- CRVD dataset: https://github.com/CS-GiXun/Raw-Video-Denoising
- FastDVDnet code: https://github.com/m-tassano/fastdvdnet
- EDVR code: https://github.com/xinntao/EDVR
- VBM4D (Maggioni 之前的工作): https://ieeexplore.ieee.org/document/6205214
- KPN (kernel prediction networks): https://arxiv.org/abs/1712.02358
- DnCNN: https://arxiv.org/abs/1608.03981
- Noise2Noise: https://arxiv.org/abs/1803.04189
- Practical Deep RAW Denoising on Mobile (ECCV 2020): https://arxiv.org/abs/2007.05589
- Recurrent VSR: https://arxiv.org/abs/1711.06005
- Poisson-Gaussian noise model: https://ieeexplore.ieee.org/document/4491068
- Wavelet CNN: https://arxiv.org/abs/1805.07071
- Deformable conv (DCN): https://arxiv.org/abs/1703.06211
- AI benchmark for mobile: https://ai-benchmark.com
- MindSpore (Huawei 框架): https://www.mindspore.cn
- Burst denoising (Deep Burst Denoising): https://arxiv.org/abs/1712.05790
- Cycle-consistency 思路: https://arxiv.org/abs/1703.10593 (CycleGAN,精神上同源)
- Modern diffusion denoising survey: https://arxiv.org/abs/2306.02985

---

如果你想让我再深入聊某一块,比如:
- 跟 Kalman filter 的精确对应关系
- 跟 diffusion model 的精神相似性
- 怎么把这套 variance propagation 思路用到 NeRF / ISP pipeline
- 为什么 mobile NPU 上 recurrent 设计天然有优势
- 跟你的 nanoGPT / micrograda d哲学的对照

随便点,我可以接着展开。

---

# EMVD (Efficient Multi-Stage Video Denoising with Recurrent Spatio-Temporal Fusion) 详解

这是一篇 CVPR 2021 的工作,来自 Huawei Noah's Ark Lab。核心 trade-off：**用 multi-stage 的 inductive bias 换 computational complexity**。在 5.38 GFLOPs 这个 budget 下,跑赢了 1965 GFLOPs 的 RViDeNet,且能在 Huawei P40 Pro 上 30fps 实时 720p denoise。

paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf
arXiv: https://arxiv.org/abs/2103.05732
CRVD dataset (Huanjing Yue et al.): https://github.com/CS-GiXun/Raw-Video-Denoising
FastDVDnet: https://github.com/m-tassano/fastdvdnet
EDVR: https://github.com/xinntao/EDVR

---

## 1. Motivation: 为什么不用一个大 network end-to-end?

Video denoising 的 SOTA 普遍是 multi-frame CNN (RViDeNet, FastDVDnet, EDVR),它们直接把 5 个或 7 个 frame 堆成 tensor 一次性 feed 进 network,用 attention 或 3D conv 做 spatio-temporal aggregation。问题是这种 naive 路线下,FLOPs 随 frame 数 linear 增长,real-time 在 mobile 上不可能。

EMVD 的核心 idea:**视频去噪这个 task 实际上有清晰的内部分解**,可以拆成三个数学上 well-defined 的 sub-task,每个用一个小网络干,而且每个 sub-task 都有一个 closed-form 的统计 interpretation:

- **Fusion**: 用过去所有 frames 平均把噪声干掉 (recursive temporal averaging)
- **Denoising**: 把 fusion 没干干净的 residual noise 干掉 (spatial CNN)
- **Refinement**: 把 denoising 过度平滑掉的高频细节从 fused image 里 restore 回来 (adaptive blending)

这种 pipeline 的好处:每个 stage 的容量都很小,但加在一起却能把 1000 GFLOPs 量级的 multi-frame SOTA 压到 5 GFLOPs 量级。

---

## 2. Observation Model

$$z_t(x) = y_t(x) + \eta_t(x), \quad \eta_t \sim \mathcal{N}(0, \sigma_t^2(y_t))$$

- $t \in T \subset \mathbb{N}$: 时间 frame index
- $x \in X \subset \mathbb{N}^2$: 空间 pixel 坐标
- $z \in \mathbb{R}^{H \times W \times C}$: 观测到的 noisy raw video,packed Bayer form,$C=4$ (R, G1, G2, B)
- $y$: 真实 noise-free 数据(要估计的)
- $\eta_t$: heteroskedastic Gaussian noise

$$\sigma_t^2(y_t) = a_t \cdot y_t + b_t$$

- $a_t$: shot noise 系数(signal-dependent,跟 photon count 有关)
- $b_t$: read noise 系数(signal-independent,跟 sensor 电路有关)

这是 Poisson-Gaussian 噪声模型的标准简化形式。$a_t, b_t$ 跟 ISO 设置有关,假定已知(可以从 sensor calibration 拿到)。

**Intuition**: 这个 model 是关键,因为它告诉我们噪声方差 $\sigma_t^2$ 是关于信号 $y_t$ 的 affine function。后面 fusion 的 closed-form variance propagation 全靠这个。

参考 Poisson-Gaussian modeling: Foi et al. 2008 https://ieeexplore.ieee.org/document/4491068

---

## 3. Learnable Invertible Transforms (核心创新之一)

EMVD 不在 pixel domain 操作,而是先做两个可学的线性变换:**Color transform** 和 **Frequency transform**。这两个都是 invertible 的,所以 forward 和 inverse 都能学,通过 loss 强制它们确实是 invert 的。

### 3.1 Color Transform $\mathcal{T}_c$

实现是 point-wise conv (1×1 conv),kernel 是一个 $C \times C$ 的矩阵 $M$。它把 4 个 Bayer 通道 ($R, G_1, G_2, B$) decorrelate 成 luminance-chrominance 表征(类似 YUV,但在 raw domain)。

- Forward: $M \in \mathbb{R}^{C \times C}$,初始化为正交矩阵(用 [6] 的方法)
- Inverse: $M' \in \mathbb{R}^{C \times C}$,初始化为 $M^{\top}$,但训练时 **不 share weights**(biorthogonal,更多自由度)

Invertibility 通过 loss 项强制:
$$\mathcal{L}_c = \|M \cdot M' - I_C\|_F^2$$

- $M \cdot M'$: 矩阵乘
- $I_C$: $C \times C$ 单位矩阵
- $\|\cdot\|_F$: Frobenius norm

**Intuition**: 为什么这一步能省算力又增精度?Bayer 的 4 个通道彼此相关性极强(G1 和 G2 几乎一样),不 decorrelate 直接喂 CNN 是浪费容量。decorrelate 之后能量集中在 luminance 上,chrominance 噪声大但人眼不敏感。

### 3.2 Frequency Transform $\mathcal{T}_f$

受 biorthogonal wavelet 启发,把 spatial domain 变成 4 个 half-resolution subband:
$$\mathcal{T}_f: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^{H/2 \times W/2 \times 4C}$$

4 个 subband 是 $\{LL, LH, HL, HH\}$ — low-pass 和三个 high-pass。

- Forward: strided conv,stride=2,4 个 $n \times n$ kernels,kernel 初始化为 $\psi \in \mathbb{R}^{2 \times n}$ 的 outer product
- $\psi$: 1-D wavelet 分解 filter,$n$ 是 filter 长度(Haar 是 $n=2$)
- Inverse: transposed conv,kernel 从重建 filter $\phi \in \mathbb{R}^{2 \times n}$ 初始化

Invertibility loss:
$$\mathcal{L}_f = \|\psi \cdot \phi^{\intercal} - I_2\|_F^2$$

- $I_2$: $2 \times 2$ 单位矩阵
- 关键: **学的是 1-D filter representation,不是 2-D conv kernel**(参数更少,且能约束 invertibility)

**Intuition**: 
1. Spatial 分辨率砍半 → FLOPs 降 4 倍(对每层 conv)
2. 多尺度分解让后续 network 不用自己学 "什么是 low-frequency vs high-frequency" 的 decomposition,这部分归纳偏置免费送了
3. 可以递归地在 $LL$ 上再 apply $\mathcal{T}_f$ 形成多尺度金字塔(paper 里用 3 个 scale)

**关键 trade-off**: 这里学的是 "类似 wavelet 但更自由" 的 filter,可能不满足传统 wavelet 性质(比如 vanishing moments),但 invertibility loss 保证它能完整地 encode/decode 信息。

### 3.3 为什么两个 transform 一起用?

$\mathcal{T}_c \circ \mathcal{T}_f$ 仍是线性可逆。组合起来同时:
- 把图像能量集中到少数系数(后续 network 容量要求降低)
- 分辨率减半(FLOPs 减小)
- 把"信号相关"和"噪声相关"的能量分开

---

## 4. Fusion Stage (递归融合)

这是 EMVD 的 temporal backbone。

$$\bar{y}_t(x) = \bar{y}_{t-1}(x) \cdot \bar{\gamma}_{t-1}(x) + z_t(x) \cdot \gamma_t(x)$$

- $\bar{y}_t$: 当前时刻 fused output
- $\bar{y}_{t-1}$: 前一时刻 fused output(recurrent state)
- $z_t$: 当前 noisy frame
- $\gamma_t \in \mathbb{R}^{H/2 \times W/2 \times 1}$: 当前 frame 的权重(单一通道,broadcast 到 4C 通道)
- $\bar{\gamma}_{t-1}$: 前一 fused 的权重
- 凸性约束: $\bar{\gamma}_{t-1}(x) + \gamma_t(x) = 1$,通过 sigmoid 保证

初始条件 $\bar{y}_0 \equiv z_0$。

**Weights 通过一个小网络 FCNN 预测**:
$$\{\gamma_t, \bar{\gamma}_{t-1}\} = \text{FCNN}\Big(|z_{LL|t} - \bar{y}_{LL|t-1}|, \hat{\sigma}_t^2\Big)$$

- $|z_{LL|t} - \bar{y}_{LL|t-1}|$: 当前 noisy low-pass 和前一 fused low-pass 的绝对差(这是一个 motion indicator)
- $\hat{\sigma}_t^2 = \sigma_t^2(z_{LL|t})$: 用 $z_{LL}$ 近似 $y_t$ 算出的噪声方差
- FCNN 输出两个 map(给当前和前一帧),通过 sigmoid

**Intuition**: 这是个 kernel-prediction network 的特例($1\times 1$ kernel)。如果当前 frame 跟前一 fused 一样(static 场景),绝对差小,FCNN 给两边都 ~0.5 权重,做时间平均。如果差别大(motion / occlusion),FCNN 主要 trust 当前 frame。Figure 3 可视化展示了这一点 — red 区域是 dynamic,blue 是 static。

### 4.1 为什么用 fused 而不是用 final output $\hat{y}_{t-1}$ 做 recurrence?

Table 1a 的 ablation:用 $\hat{y}_{t-1}$ 替代 $\bar{y}_{t-1}$ → PSNR 掉 0.25dB。

**Intuition**: 
1. Denoising 是非线性的,递归方差不再有 closed-form,DCNN 不知道当前 frame 该 trust 多少
2. $\hat{y}_{t-1}$ 已经被 denoise 过,高频细节可能被平滑掉,后续 fusion 拿不回来这些信息

### 4.2 Fusion kernel 的推广 (公式 7)

$$\bar{y}_t(x) = \bar{y}_{t-1}(x) \circledast \bar{k}_{t-1}(x) + z_t(x) \circledast k(x)$$

- $\circledast$: 卷积
- $\bar{k}_{t-1}$: 作用在 previous fused 上的 spatially adaptive kernel,大小 $\bar{p} \times \bar{p}$
- $k$: 作用在当前 noisy frame 上的 kernel,大小 $p \times p$
- 凸性通过 softmax 保证

Table 2 ablation 显示:在 previous frame 上用 $3 \times 3$ kernel + 当前 frame 用 $1 \times 1$ kernel,PSNR 只多 0.1dB 但 GFLOPs 增加 1.4。**motion compensation 主要需要在前一帧那侧做**(因为当前 frame 没有 alignment 问题),不过性价比不高,所以 baseline 用 $1 \times 1$。

### 4.3 多尺度融合的递归细节

paper 里用 3 个 scale。低 scale 的 fusion 权重会上采样 + concat 到 (6) 的输入,作为 guidance。因为低 scale 的运动估计更可靠(感受野相对更大)。

---

## 5. Recursive Variance Propagation (这是 paper 最关键的创新)

这一节我认为是整个方法的灵魂。

fusion 是线性组合,所以 fused image 的方差也能 closed-form 递归:

$$\bar{\sigma}_t^2 \equiv \sigma_t^2(\bar{y}_{LL|t}) = \bar{\gamma}_{t-1}^2 \cdot \sigma_{t-1}^2(\bar{y}_{LL|t-1}) + \gamma_t^2 \cdot \sigma_t^2(z_{LL|t})$$

- 第一项: 前一帧 fused 方差,衰减系数 $\bar{\gamma}_{t-1}^2$
- 第二项: 当前帧 raw 方差
- 协方差项 = 0 (假定噪声时序独立)
- 初始条件 $\sigma_t^2(\bar{y}_{LL|0}) \equiv \sigma_t^2(z_{LL|0})$

**为什么这很重要**: 
1. 因为 $\gamma_t(x) \leq 1$ 对所有 $x, t$ 都成立,$\bar{\sigma}_t^2$ 随时间 **单调非增**(non-strictly decreasing)。这数学上证实了 "fusion 越融合噪声越小" 这个直觉。
2. 给 DCNN 一个 **精确的、时变的、per-pixel 的 noise variance map**,而不是用一个常数或 heuristics 估计。
3. Table 1a 最后一行显示,把所有 variance input 去掉(blind formulation),PSNR 暴跌 1.24dB。这告诉我:在 raw domain denoise,知道噪声水平是非盲性能和盲性能差距巨大的根本原因。

**Intuition**: 传统 video denoise 在 pixel domain 操作,fusion 后的方差要么近似 (用 $\sqrt{N}$ 假设),要么直接忽略。EMVD 因为在 transform domain + linear fusion + heteroskedastic noise model 都对得上,所以方差可以精确 track 到每一个 pixel 的当前噪声水平。

参考 variance propagation: $\text{Var}[aX + bY] = a^2\text{Var}[X] + b^2\text{Var}[Y] + 2ab\text{Cov}[X, Y]$

---

## 6. Denoising Stage

$$\tilde{y}_t = \text{DCNN}\Big(\bar{y}_t, z_{LL|t}, \bar{\sigma}_t^2\Big)$$

- $\tilde{y}_t$: denoise 后的 image
- $\bar{y}_t$: fused image(含残余噪声)
- $z_{LL|t}$: 当前 noisy frame 的 low-pass(原始未污染信息)
- $\bar{\sigma}_t^2$: fused image 的噪声方差(来自公式 9)

**为什么 input 要塞 $z_{LL|t}$**: Table 1a 显示去掉它 PSNR 掉 0.76dB。Denoise 网络需要"未经 fusion 的原始信息"作为细节 reference,因为 fusion 可能因为 motion 把细节模糊掉。

**多尺度**: 低 scale 估计结果 concat 到当前 scale input。

**容量分配**: Table 1b 显示 DCNN 分配最大容量最划算(denoise 是最难的 stage)。

---

## 7. Refinement Stage

Denoising 会过度平滑,特别是在 SNR 差的地方。Refinement 把 detailed-but-noisy 的 fused image $\bar{y}_t$ 和 noise-free-but-oversmoothed 的 denoised image $\tilde{y}_t$ 自适应混合:

$$\hat{y}_t(x) = \bar{y}_t(x) \cdot \bar{\omega}_t(x) + \tilde{y}_t(x) \cdot \tilde{\omega}_t(x)$$

- $\hat{y}_t$: 最终输出
- $\bar{\omega}_t + \tilde{\omega}_t = 1$ (convex)
- 权重由 RCNN 预测:
$$\{\tilde{\omega}_t(x), \bar{\omega}_t\} = \text{RCNN}\Big(\tilde{y}_t, \bar{y}_t, \bar{\sigma}_t^2\Big)$$

**Intuition**: Figure 4 显示 $\omega_t$ 在高频区域(边缘、纹理)把权重大量给 fused image(因为 denoised 把这里 smooth 掉了),在 flat 区域把权重给 denoised image(因为 fused 还残留噪声)。这个 **完全无监督地 emergent 出来** — loss 只在 final output 上有 $\mathcal{L}_r$,RCNN 自己学到了"哪里该 trust denoised 哪里该 trust fused"。

**Refinement 的妙处**: 它跟 fusion 公式同构,但语义完全不同:
- Fusion $\gamma$: 跨帧 trust 多少(时间一致性)
- Refinement $\omega$: 同一帧内 trust high-freq 多少(空间细节)

---

## 8. 训练 Loss

$$\mathcal{L} = \mathcal{L}_r + \mathcal{L}_c + \mathcal{L}_f$$

$$\mathcal{L}_r = \frac{1}{n} \sum_{t=1}^{n} \|\hat{y}_t - y_t\|_1$$

- $\mathcal{L}_r$: L1 reconstruction loss(对每个 frame)
- $\mathcal{L}_c$: color transform invertibility
- $\mathcal{L}_f$: frequency transform invertibility

**关键**: 整个 pipeline 只在 final output $\hat{y}_t$ 上监督。三个网络 (FCNN/DCNN/RCNN) 完全靠 backprop through time (BPTT) 自然分工,无额外 supervision。

训练 setting:
- $n=25$ patches (因为 recurrent,需要 long unrolling)
- patch size 128×128
- Adam, lr=1e-4, batch size=16
- 300k iterations on CRVD+SRVD → 300k fine-tune on CRVD only

---

## 9. Baseline architecture (yellow row in Table 1)

- 3 个 CNN,每个:2 conv (3×3, 16 filters) + ReLU + 1 output conv
- Color transform: orthonormal 4×4
- Frequency transform: Haar init, 3 scales
- **5.38 GFLOPs**

具体 capacity (Table 1b yellow row):
- FCNN: 2 conv / 16 filters
- DCNN: 2 conv / 16 filters  
- RCNN: 2 conv / 16 filters

---

## 10. Ablation 关键 takeaway

Table 1a (network structure):
| 修改 | PSNR 变化 |
|---|---|
| 用 $\hat{y}_{t-1}$ 替代 $\bar{y}_{t-1}$ 做 recurrence | -0.25 dB |
| 去掉 DCNN 的 $z_{LL|t}$ input | -0.76 dB |
| 去掉 color transform | -0.27 dB |
| 用 pixel shuffle 替代 frequency transform | -0.28 dB |
| 去掉 refinement stage | -0.17 dB |
| 去掉 variance input (blind) | **-1.24 dB** |
| 去掉 fusion + refinement | -1.28 dB |

**最重要的 insight**: variance input 价值 1.24dB — raw domain denoising 中,blind vs non-blind 的 gap 是巨大的。EMVD 通过 closed-form variance propagation 把这个 advantage "免费"拿到了。

Table 2 (fusion kernel size):
- $1\times 1$ baseline: 2542.86 GFLOPs / 44.51 dB
- $\bar{p}=3, p=1$: 2544.25 GFLOPs / 44.58 dB (+0.07dB, 几乎免费)

---

## 11. 实验结果

Table 3 (CRVD dataset):

| Model | GFLOPs | PSNR (raw) / SSIM | PSNR (sRGB) / SSIM |
|---|---|---|---|
| EDVR | 3088.98 | 44.71 / 0.9902 | 40.89 / 0.9838 |
| RViDeNet | 1965.04 | 44.08 / 0.9881 | 40.03 / 0.9802 |
| FastDVDnet | 664.99 | 44.30 / 0.9891 | 39.91 / 0.9812 |
| **EMVD (79.52)** | 79.52 | 44.05 / 0.9890 | 39.53 / 0.9796 |
| FastDVDnet† | 22.16 | 42.25 / 0.9806 | 37.43 / 0.9693 |
| **EMVD (5.38)** | 5.38 | **42.63 / 0.9851** | **38.27 / 0.9722** |
| VBM4D | — | — | 35.20 / 0.9577 |

**关键观察**:
1. 在 5.38 GFLOPs 下,EMVD 比 FastDVDnet† 多 **0.84dB**(算力还少 4×)
2. 79.52 GFLOPs 的 EMVD 跟 1965 GFLOPs 的 RViDeNet 差距只有 0.03dB,算力比 1:25
3. Figure 7 显示 EMVD 在时间轴上 PSNR 单调上升,而 multi-frame 方法达到 plateau(因为它们 frame 数固定)。这证实 recurrent 比 multi-frame 在长序列上有信息累积优势
4. dynamic scene (frame 25-30) 时 multi-frame 方法有短暂优势(它们能看 future frames),但 EMVD 几帧内就 recover

Table 4 (on-chip profiling on Huawei P40 Pro):

| Model | GFLOPs | Time (ms) | DDR (MB) | PSNR |
|---|---|---|---|---|
| FastDVDnet† | 22.16 | 177 | 724 | 37.43 |
| **EMVD** | 5.38 | **36** | **112** | **38.27** |

- 36ms per 720p frame → 27.8 fps (real-time)
- 比 FastDVDnet† 快 4.9×,内存少 6.5×,精度还高 0.84dB

---

## 12. 为什么这个 work 重要 — build your intuition

### 12.1 Multi-stage vs monolithic

EMVD 用了 3 个 stage,每个 stage 容量极小。这暗示一个深层规律:**对有明确子结构的任务,把 inductive bias 编码到 pipeline 里,比让大 network 从头学要高效得多**。video denoising 的"显式分解"是:
1. 时间平均(线性,可分析方差)
2. 空间去噪(非线性,DCNN 干)
3. 高频 refinement(线性混合,RCNN 干)

每一个 sub-task 用最适合它的工具,而不是用一个万能 black box。

### 12.2 Transform domain + linear fusion = closed-form variance

这是数学上最优雅的点。如果在 pixel domain 做 fusion,然后做非线性 denoise,variance 没法 propagate。EMVD 把这三件事对齐了:
- transform 是线性的
- fusion 是线性的  
- 所以方差 closed-form 传播

下游 DCNN 因此能拿到精确的 per-pixel noise level。这就是为什么 variance input 值 1.24dB。

### 12.3 Recurrent vs multi-frame trade-off

multi-frame (RViDeNet, FastDVDnet, EDVR) 优势:能看 future frames,在 dynamic scene 短暂占优。
recurrent (EMVD) 优势:长序列信息累积,PSNR 单调上升,FLOPs 与历史长度无关。

EMVD 的选择是 recurrent + 可学 fusion weights,这是它在 mobile 上能跑的关键。

### 12.4 没有显式监督

整个 pipeline 只有一个 $\mathcal{L}_r$ 在 final output 上。三个网络 FCNN/DCNN/RCNN 的分工完全靠 gradient 自然 emergent。这意味着 architecture 的 inductive bias 足够强,网络能自己 discover 正确的 behavior — fusion 学到区分 static/dynamic,refinement 学到区分 high-freq/low-freq。

### 12.5 跟其他方法的对比 intuition

- **VBM4D** [28]: classical,non-local block matching,无监督,但 PSNR 低且慢
- **RViDeNet** [42]: multi-frame (7 frames),large capacity,无 motion compensation,直接堆 frames 喂大网络
- **FastDVDnet** [36]: multi-frame (5 frames),用 U-Net 做 spatial denoise,无 explicit motion
- **EDVR** [38]: multi-frame + deformable conv 做 alignment,SOTA 但极重 (3088 GFLOPs)
- **EMVD**: recurrent + multi-stage + transform domain + closed-form variance,极致高效

### 12.6 我会关心的几个 limitation / follow-up 方向

1. **Motion 补偿**: EMVD 主要靠 fusion weights 做 implicit motion compensation,大运动场景下应该不如 explicit alignment (deformable conv)。Table 2 显示 kernel prediction 收益有限。
2. **future frame**: recurrent 不能看 future,real-time 场景下确实如此,但 offline processing 下 multi-frame 仍有优势
3. **长时间遗忘**: recurrent 在极长序列下,early frame 信息会被 $\gamma$ 衰减掉。是否需要 keyframe 机制?
4. **Color transform 学到了什么**: paper 没可视化 $M$ 学到的 basis,值得分析
5. **跟 Nerf/implicit 的关联**: 这种 closed-form variance propagation 思路,可以用到 NeRF 的不确定性估计上
6. **Diffusion models**: 现在 diffusion-based denoising (Noise2Noise 类) 在 raw domain 也有进展,EMVD 的 multi-stage 思路能否结合?

---

## 13. 参考 / 资源

- Paper PDF (CVPR 2021): https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf
- CVPR 2021 listing: https://openaccess.thecvf.com/content/CVPR2021/html/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.html
- arXiv: https://arxiv.org/abs/2103.05732
- CRVD dataset (Huanjing Yue et al.): https://github.com/CS-GiXun/Raw-Video-Denoising
- FastDVDnet (Tassano et al.): https://github.com/m-tassano/fastdvdnet
- EDVR (Wang et al.): https://github.com/xinntao/EDVR
- VBM4D (Maggioni et al. 2012): https://ieeexplore.ieee.org/document/6205214
- Kernel Prediction Networks (Mildenhall et al.): https://arxiv.org/abs/1712.02358
- Poisson-Gaussian noise model (Foi et al. 2008): https://ieeexplore.ieee.org/document/4491068
- Practical Deep RAW Denoising on Mobile (Wang et al. ECCV 2020): https://arxiv.org/abs/2007.05589
- MindSpore: https://www.mindspore.cn
- AI Benchmark tool (Ignatov et al.): http://ai-benchmark.com

---

总结一句:EMVD 的精髓是 **"用 architecture 表达 task 的内部分解,用 closed-form 数学把 uncertainty 显式 propagate 给下游,用 invertible transform 把空间冗余抽走"**。这三件事一起,把 1965 GFLOPs 的 SOTA 压到 5 GFLOPs 还能保持竞争力。这是一个非常 elegant 的 example,值得反复琢磨 — 它告诉你"inductive bias is compute"在 video restoration 这个 task 上到底能值多少 dB。
