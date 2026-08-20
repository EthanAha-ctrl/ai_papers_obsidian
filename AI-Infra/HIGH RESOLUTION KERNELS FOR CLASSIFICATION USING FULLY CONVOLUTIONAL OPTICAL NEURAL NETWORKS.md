---
source_pdf: HIGH RESOLUTION KERNELS FOR CLASSIFICATION USING FULLY CONVOLUTIONAL OPTICAL
  NEURAL NETWORKS.pdf
paper_sha256: 5f19ab4a4f479dcdd4e54de4dbd5b4d17114fd0a54c441d4ad7b09b5b2538b0b
processed_at: '2026-08-04T23:46:02-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

传统 CNN 长成漏斗形（32→16→8→4→flatten→FC）是为了讨好 GPU，因为 GPU 上算大 kernel 很贵、算小图很快.
但到了光学 4f 系统上，规则全变了——**光做 Fourier transform 是免费的，分辨率大小无所谓，真正的瓶颈是"光和电之间来回切换的次数"**。所以漏斗形反而浪费，桶形（保持高分辨率 + 大 kernel + 少 channel）才是光学友好的。

FatNet 就是按这个新规则把 ResNet-18 改造成桶形，conv 操作数降 8.2 倍，代价是 CIFAR-100 上掉 6% accuracy。

---

## 1. 先讲清楚 4f 系统到底在干啥

### 1.1 物理装置

就是一个老式光学实验装置，1966 年 Weaver 和 Goodman 就玩过了：

```
[Laser] --f--> [Lens 1] --f--> [SLM/相位mask] --f--> [Lens 2] --f--> [Camera]
input          Fourier         Fourier plane           inverse FT      output
              transform         (kernel乘法)             back to space
```

四个 $f$ 是焦距，光从输入到输出总共走 $4f$，所以叫 4f correlator。

### 1.2 为什么这玩意儿能做卷积

回忆 convolution theorem：

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}
$$

左边是空间域的卷积，右边是频域的逐点相乘。**凸透镜天然在 focal plane 上做 Fourier transform**——这是光学几百年来都知道的事实。所以：

1. Lens 1 在 Fourier plane 投出输入的频谱
2. SLM 在 Fourier plane 上把频谱乘上 kernel 的频谱（逐点乘，光速完成）
3. Lens 2 做 inverse Fourier，相机拍到空间域的卷积结果

整个过程光只走 $4f$ 距离，**与输入分辨率无关**。这就是光学加速的核心：32×32 和 256×256 的图，光都一样快地算完。

### 1.3 与电子 FFT 的对比

电子 FFT 算 2D Fourier 是 $O(n^2 \log n)$，分辨率翻倍计算量翻四倍多。

光学呢？光走 $2f$ 距离用的时间是 $\frac{2f}{c}$，$c$ 是光速，几纳秒，**和 $n$ 无关**。

所以电子计算里我们拼命减小分辨率（pooling、stride），光学计算里这个优化方向完全失效。

### 1.4 真正的瓶颈

paper 里反复强调，光学 4f 的瓶颈是：

- **SLM（spatial light modulator）和相机的 frame rate**：现代约 2 MHz
- **optics ↔ electronics 之间的转换**：每次"光进光出"之间，SLM 要刷新、相机要读出，这都是毫秒级

所以最优策略是：**每次光学 inference 处理尽量多的信息，减少调用次数**。

---

## 2. 传统 CNN 为什么对光学不友好

### 2.1 ResNet-18 在 CIFAR-100 上的形状

```
Input 32×32×3
  → Conv 3×3, 64 channels, 32×32
  → Pool, 16×16×64
  → Conv 3×3, 64 channels
  → Pool, 8×8×128
  → Conv 3×3, 128 channels
  → Pool, 4×4×256
  → Conv 3×3, 256 channels
  → Pool, 2×2×512
  → Flatten, 2048
  → FC → 100 classes
```

漏斗形！空间分辨率从 32 降到 2，channel 从 3 涨到 512。这是 GPU 时代的标准做法。

### 2.2 在光学上跑这套有什么问题

每个 conv 层都要：
1. 把上层的 output 从 electronics 转成 SLM 上的光信号
2. SLM 刷新，光跑一遍 4f
3. 相机读出，转回 electronics
4. （ResNet 还有 skip connection，再加一次加法）

ResNet-18 有 ~20 个 conv 层，就要做 ~40 次光-电转换。每次都几毫秒。**光是物理 setup 就要秒级**。

而且空间分辨率越小（深层），4f 系统的高分辨率能力越浪费——4K 相机只用 2×2 个像素，剩下 8M 像素都闲着。

### 2.3 还有 stride 的问题

4f 系统物理上做不了 strided convolution。光传播是连续的，没有"跳着采样"这回事。所以 ResNet 里那些 `stride=2` 全得改成普通 conv + pooling。

### 2.4 还有负数的问题

相机只能读 intensity = $|amplitude|^2$，物理上读不出负值。workaround 叫 **pseudo-negativity**：把每个 kernel 拆成正/负两份，跑两次，相减得到真实结果。这又把 conv 次数翻倍。

所以传统 CNN 在光学上的劣势被多重叠加：
- 层数多 → 转换次数多
- 小分辨率 → 浪费光学 parallelism
- stride → 做不了
- 负值 → conv 翻倍

---

## 3. FatNet 的核心思路：桶形 + 大 kernel

### 3.1 一句话

**别再做 cone-shape 了，保持空间分辨率，少用 channel，用大 kernel 补偿信息容量。**

### 3.2 转换的四条规则

paper 给的规则其实很简单，目标是 fair comparison：

| Rule | 说明 | 为什么 |
|------|------|--------|
| 1 | 层数不变 | 保持非线性数量 |
| 2 | 浅层不动，直到 feature map 像素数 ≤ 类别数 | 浅层本来分辨率就高，没必要动 |
| 3 | 每层输出总像素数（$C \times H \times W$）保持不变 | 信息容量守恒 |
| 4 | 每层参数量保持一致 | 公平比较，控制过拟合 |

具体怎么算？看一个例子：

原始 ResNet-18 中间某层：`128 → 128, k=3×3`，feature map 8×8
- 总像素 = $128 \times 8 \times 8 = 8192$
- 参数 = $128 \times 128 \times 3 \times 3 = 147456$

FatNet 要让 feature map 保持 10×10（不 pool 下去），那么：
- 由 rule 3：$C_{new} \times 10 \times 10 = 8192$，解出 $C_{new} \approx 82$
- 由 rule 4：$82 \times 82 \times K_{new}^2 = 147456$，解出 $K_{new} \approx 5$
- 所以 FatNet 这层是：`82 → 82, k=5×5`

深层更夸张。原始 `512 → 512, k=3×3`，feature map 4×4：
- 总像素 = $512 \times 4 \times 4 = 8192$
- FatNet 保持 10×10 → $C_{new} = 82$
- 但等等，深层 feature map 在原 ResNet 里是 2×2，对应 $C \times 2 \times 2 = 2048$，所以 FatNet 这里 $C_{new} = 21$
- 参数匹配：$21 \times 21 \times K^2 = 2359296$ → $K \approx 73$

**73×73 的 kernel！** 在 10×10 的 feature map 上滑，same padding 下中心 10×10 真正训练，外圈基本是浪费。paper 自己也承认这里效率低，但权衡下来还是比加 channel 更好（因为 channel 多要光学并行做多次，而 kernel 大只是 SLM 上一个静态 mask）。

### 3.3 最后输出层

CIFAR-100 有 100 类，FatNet 输出是 `21 → 1, k=49×49`，输出 10×10×1，flatten 成 100 维向量。**全卷积，无 FC layer**。

这是关键设计：FC 层在光学上没法直接做（需要 dense connection，4f 是 conv 专用），全卷积就能整套在 4f 上跑。

### 3.4 架构对比图

```
ResNet-18 (cone):
  32×32, 64ch
    ↓ pool
  16×16, 64ch
    ↓ pool
  8×8, 128ch
    ↓ pool
  4×4, 256ch
    ↓ pool
  2×2, 512ch
    ↓ flatten
  FC → 100

FatNet (barrel):
  32×32, 64ch      (浅层不动)
    ↓ pool
  16×16, 64ch      (浅层不动)
    ↓
  10×10, 82ch      (开始 barrel)
    ↓ (无 pool)
  10×10, 82ch
    ↓
  10×10, 41ch
    ↓
  10×10, 41ch
    ↓
  10×10, 21ch
    ↓
  10×10, 1ch → flatten → 100
```

桶形！中间一直保持 10×10，channel 缓慢减少，kernel 越来越大。

---

## 4. 4f 系统的 simulator 怎么建

paper 用 PyTorch 写了个 `OptConv2d` layer，用 **Angular Spectrum of Plane Waves (ASPW)** 模拟光传播。

### 4.1 核心公式

给定入射波前 $U_1(x, y)$，传播距离 $z$ 后：

$$
U_2(x, y) = \mathcal{F}^{-1}\left[\mathcal{F}[U_1(x, y)] \cdot H(f_x, f_y)\right] \tag{2}
$$

- $U_1, U_2$：复振幅分布（complex amplitude）
- $\mathcal{F}, \mathcal{F}^{-1}$：2D Fourier / inverse Fourier
- $H(f_x, f_y)$：transfer function，描述每个空间频率 $(f_x, f_y)$ 经过距离 $z$ 的相位变化

物理直觉：把入射波分解成不同方向的平面波（Fourier），每个平面波走 $z$ 距离后相位变了多少由 $H$ 决定，再合成回空间域。

### 4.2 Transfer function 的具体形式

Fresnel 近似下的 free-space transfer function：

$$
H_F(f_x, f_y) = \exp\left[jkz - j\pi\lambda z (f_x^2 + f_y^2)\right] \tag{3}
$$

- $k = \frac{2\pi}{\lambda}$：波数
- $\lambda$：光波长（paper 用 532 nm 绿光）
- $z$：传播距离
- $f_x, f_y$：空间频率
- 第一项 $jkz$：全局相位（与频率无关，可以丢掉不影响结果）
- 第二项 $-j\pi\lambda z (f_x^2 + f_y^2)$：二次相位，类似 chirp。高频分量（大 $f_x, f_y$）相位变化快，低频慢。这就是衍射的本质——不同空间频率的角谱以不同角度传播，走 $z$ 距离后累积不同相位。

### 4.3 透镜的 transmittance

薄透镜近似：

$$
t_A(x, y) = P(x, y) \exp\left[-j\frac{k}{2f}(x^2 + y^2)\right] \tag{4}
$$

- $P(x, y)$：pupil function，孔径内取 1，外取 0
- $f$：焦距（paper 用 10 mm）
- $\exp[-j\frac{k}{2f}(x^2+y^2)]$：抛物相位，把平面波聚焦成球面波。**这个抛物相位乘在空间域上，等价于在频域做 Fourier transform**——这就是透镜做 FT 的数学本质。

### 4.4 像素尺度与传播距离的关系

$$
z = \frac{N(\Delta x)^2}{\lambda} \tag{5}
$$

- $N$：一维像素数
- $\Delta x$：像素物理尺寸（pixel pitch）
- $\lambda$：波长

这个公式来自 Nyquist 采样和 Fresnel 衍射的自洽条件。**直觉**：像素越大（采样越粗，能表示的空间频率越低），需要走更远才能让 Fresnel 衍射的相位累积完整一个周期。

paper 故意选 $\Delta x$ 使得 $z = f$，这样每个 focal distance 只需一次 iteration。否则要分多步传播，计算量翻倍。

### 4.5 训练成本

- 标准 FatNet（PyTorch Conv2d）：epoch 15 秒
- 光学 simulator FatNet：epoch **67 分钟**

慢 268 倍！因为 ASPW 把光传播加入了 computation graph，每步传播都要存梯度。这就是为什么 optical simulator 只跑了部分实验。

---

## 5. Batch Tiling：4f 系统怎么真正快起来

### 5.1 问题

4f 系统的 frame rate ~2 MHz，单次 inference 要约 0.5 μs。但单张 32×32 图只用了 4K 相机的极小一部分像素。剩下 8M 像素闲着。

### 5.2 解决方案

把同一 batch 的多张图 tile 到一个大 input block 里，kernel pad 到同尺寸，**一次光学 inference 同时算整个 batch 的 conv**。

### 5.3 公式

$$
n = \left\lfloor \frac{R}{M + N - 1} \right\rfloor^2 \tag{6}
$$

- $R$：4f 系统分辨率（4K = 3840）
- $M \times M$：单张 input 尺寸（需 padding）
- $N \times N$：kernel 尺寸
- $M + N - 1$：linear conv 输出尺寸（valid convolution），所以 input 和 kernel 都 pad 到这个大小
- $\lfloor \cdot \rfloor$：floor function
- **平方**：因为 2D 排布

### 5.4 举例

$R = 3840$, $M = 32$, $N = 3$：

$$
n = \left\lfloor \frac{3840}{32 + 3 - 1} \right\rfloor^2 = \left\lfloor \frac{3840}{34} \right\rfloor^2 = 112^2 = 12544
$$

一次光学 inference 可以同时算 12544 张图的 conv！这就是 batch 3136 的来源（paper 用 3136 作为 4K 完全利用的合理 batch size）。

### 5.5 为什么传统 CNN 用不了这个

ResNet 深层 feature map 是 2×2 或 4×4，batch tiling 后 $n$ 还是大，但每层都要做 optics-electronics 转换。**层数才是瓶颈，不是 batch size**。FatNet 通过减少层数（8.2× conv ops reduction）让 batch tiling 的收益不被转换成本吃掉。

---

## 6. 实验结果

### 6.1 Table 2: Accuracy vs Conv Ops

| Architecture | Test Accuracy | Conv Ops | Ratio |
|---|---|---|---|
| ResNet-18 (GPU) | 66 ± 1.4% | 1,220,800 | 1.0 |
| FatNet (GPU) | 60 ± 1.4% | 148,637 | 0.12 |
| FatNet (Optical sim) | 60% | 148,637 | 0.12 |

**关键观察**：
- Conv ops 减少 8.2 倍
- Accuracy 损失 6 个百分点
- Optical simulator 与 GPU 训练的 FatNet accuracy 一致（60% = 60%），说明 ASPW 物理近似足够好，没引入额外误差

### 6.2 Table 3: Inference Time（秒/样本）

| Architecture | Batch 64 | Batch 3136 |
|---|---|---|
| ResNet-18 (GPU) | 1.350e-4 | 1.167e-4 |
| FatNet (GPU) | 4.565e-4 | 7.942e-4 |
| ResNet-18 (Optics) | 3.815e-2 | 7.786e-4 |
| **FatNet (Optics)** | 4.645e-3 | **9.479e-5** |

**故事全在这张表里**：

1. **GPU 上 FatNet 比 ResNet 慢**（batch 64: 4.565e-4 vs 1.350e-4，慢 3.4 倍）。大 kernel 在 GPU 上是计算密集，GPU 擅长的小 kernel batch parallelism 用不上。**这反向证明 FatNet 是为 optics 量身定做**。

2. **Optics 上 batch 64 时 ResNet-18 极慢**（3.815e-2，比 GPU 慢 280 倍！）。因为 batch 太小，4f 系统的高分辨率 parallelism 全浪费，每层都要光学转换，又慢又没用上像素。

3. **Batch 3136 时 FatNet Optics 是全场最快**（9.479e-5），比 ResNet-18 GPU（1.167e-4）还快 1.2 倍，比 ResNet-18 Optics（7.786e-4）快 8 倍。**这就是 batch tiling + FatNet 高分辨率策略的胜利**。

4. ResNet-18 在 Optics 上即使 batch 3136 也只有 7.786e-4，因为它 cone-shape 导致深层 feature map 太小（4×4、2×2），batch tiling 的 $n$ 在小 $M$ 时受 padding 限制（要 pad 到 $M+N-1$），而且每层都要 optics-electronics 转换。

### 6.3 Figure 4 训练曲线

- 三个网络 train accuracy 都到 99%，ResNet-18 收敛快
- Validation accuracy：ResNet-18 ~66%，FatNet 和 optical sim ~57-58%
- Test accuracy 60%（比 val 高，因为 test 没用 augmentation，train/val 用了 horizontal flip + random crop，更难）

---

## 7. 物理限制（paper 提到但没解决）

### 7.1 非负约束

Camera 只能读 intensity = $|amplitude|^2$，物理上读不出负值。**Pseudo-negativity** workaround：每个 kernel 拆正/负两份，跑两次，相减：

$$
y = y^+ - y^-
$$

这意味着实际 conv 调用次数翻倍，部分抵消了 FatNet 节省的 8.2×。

### 7.2 180° 旋转

几何光学导致 4f 输出相对输入旋转 180°。CNN 不在乎，因为 conv 是 translation-equivariant 学出来的，旋转后的 feature map 后续层照样能提取特征。但要注意架构感知这一点。

### 7.3 No stride

4f 物理上没法做 strided conv。paper 用普通 conv + pooling 替代。但这丢失了 stride 在 CNN 里的 multi-scale 信息聚合作用，FatNet 完全靠大 kernel 模拟类似 receptive field。

### 7.4 Alignment

Free-space optics（vs silicon photonics）对物理 alignment 极其敏感，任何 lens 偏移都导致输出错乱。paper simulator 没建模这个，实际部署需要 optical cage systems 固定。

### 7.5 Quantization & Noise

没建模。但作者认为 noise 可作 regularization（类似 [mixup](https://arxiv.org/abs/1710.09412) 或 dropout），低 bit 量化可能不影响 accuracy 太多。

---

## 8. 我的判断

### 8.1 核心贡献

paper 最大的价值是**点破一个被忽视的事实**：CNN 的 cone-shape 不是深度学习的固有最优，而是电子计算成本模型下的局部最优。一旦换硬件，cost function 变了，架构最优解也变。这是 hardware-aware NAS 的手工版本。

### 8.2 可质疑的点

1. **6% accuracy 损失非平凡**。CIFAR-100 上掉 6 个点很大。73×73 kernel on 10×10 feature map 的有效参数利用率极低（中心 10×10 真正训练，外圈浪费），paper 自己也承认。
2. **Batch 3136 不现实**。SGD 在 batch > 1024 时常有 generalization gap（见 [Goyal et al., 2017](https://arxiv.org/abs/1706.02677)），而且实际部署中很少有这么大的 batch。
3. **CIFAR-100 32×32 太小**，论证光学"高分辨率优势"需要更大输入（ImageNet 224×224）才能真正体现。paper 自己承认 GPU memory 限制没跑 ImageNet。
4. **ResNet-18 backbone 是 2016 的老架构**，现代 EfficientNet / ConvNeXt 的 channel-scaling 平衡点和 FatNet 的设计哲学可能冲突。
5. **没考虑 alignment / quantization / noise**。真实光学系统的 noise floor 和 dynamic range 限制可能让 60% 进一步降。

### 8.3 对未来工作的启发

- 跟 [Diffractive D²NN](https://www.science.org/doi/10.1126/science.aat8084) 这类 all-optical network 互补：D²NN 是层数全光学，FatNet 是架构全卷积化
- 与 [Channel Tiling (Li et al.)](https://arxiv.org/abs/2001.06912) 互补：channel tiling 适合多 channel 网络，FatNet 走 batch tiling 路线
- 类似 spirit 的工作：[Large Kernel Matters (GCN)](https://openaccess.thecvf.com/content_cvpr_2017/html/Peng_Large_Kernel_Matters_CVPR_2017_paper.html) 已经在 segmentation 上证明大 kernel 有用，FatNet 把这个 insight 推到极限
- 如果把这个 "optics-friendly cost metric" 嵌入 [DARTS](https://arxiv.org/abs/1806.09055) / differentiable NAS，应该能搜出比手工 FatNet 更好的架构

---

## 9. Reference & 延伸阅读

主 paper：
- FatNet paper: [arXiv 2210.14134](https://arxiv.org/abs/2210.14134)
- FatNet code: [github.com/riadibadulla/simulator](https://github.com/riadibadulla/simulator)

核心相关：
- 4f channel tiling (Li et al.): [arxiv.org/abs/2001.06912](https://arxiv.org/abs/2001.06912)
- Hybrid optical-electronic CNN (Chang et al. 2018): [nature.com/articles/s41598-018-30619-y](https://www.nature.com/articles/s41598-018-30619-y)
- Diffractive D²NN (Ozcan group, Science 2018): [science.org/doi/10.1126/science.aat8084](https://www.science.org/doi/10.1126/science.aat8084)
- Massively parallel amplitude-only Fourier NN (Miscuglio, Optica 2020): [optica.org/abstract.cfm?uri=optica-7-12-1812](https://opg.optica.org/optica/abstract.cfm?uri=optica-7-12-1812)
- Silicon photonics MZI NN (Shen et al., Nature Photonics 2017): [nature.com/articles/nphoton.2017.93](https://www.nature.com/articles/nphoton.2017.93)

背景：
- ResNet original: [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- Large Kernel Matters (GCN): [arxiv.org/abs/1703.02719](https://arxiv.org/abs/1703.02719)
- Large batch SGD generalization gap: [arxiv.org/abs/1706.02677](https://arxiv.org/abs/1706.02677)
- DARTS differentiable NAS: [arxiv.org/abs/1806.09055](https://arxiv.org/abs/1806.09055)
- Fourier optics textbook (Voelz): [spiedigitallibrary.org](https://spiedigitallibrary.org/ebooks/Computational-Fourier-Optics-A-MATLAB-Tutorial/eISBN-9780819482047)
- CIFAR-100 dataset: [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- Optical NN survey: [ieeexplore.ieee.org/document/9023086](https://ieeexplore.ieee.org/document/9023086)

---

## 10. 最终直觉

FatNet 揭示的事实：**我们习惯的 CNN 形状（漏斗）是电子计算时代塑造的偏见**。换硬件，形状就要变。光学时代，桶形（保持分辨率、大 kernel、少 channel）反而合理。

paper 用 8.2× conv reduction 换 6% accuracy 损失，在 optical deployment scenario 下是值得的 trade-off。随着 SLM/camera frame rate 改善（现在 2 MHz，未来可能 GHz），这个 trade-off 会越来越 favorable。等光学硬件成熟，我们可能需要重新审视"3×3 kernel 是默认最优"这个被 VGG 固化下来的设计哲学。

对 Andrej 你来说，这 paper 最有意思的点可能是：**它本质上是在做 hardware-aware NAS 的手工版本**，cost metric 不是 FLOPs / params，而是 "optics-electronics conversion count + resolution utilization"。把这个 metric 自动化、嵌入 differentiable NAS，应该是个很有意思的方向。

---

# FatNet: 为 Free-Space Optical Neural Network 重新设计 CNN 架构

这篇 paper 的核心 insight 非常有意思: **传统 CNN 的 cone-shape (漏斗形) 架构是为电子计算优化的产物, 一旦换到 4f free-space optical system, 这种架构反而是 suboptimal 的**. 作者提出 FatNet, 一种 barrel-shape (桶形) 全卷积架构, 用大 kernel + 高分辨率 feature map + 少 channel 来匹配光学系统的物理特性.

---

## 1. 核心动机: 为什么 GPU-favorable 的架构在 optics 上是 wasteful 的

### 1.1 电子计算的成本模型

在 CPU/GPU 上, 一层 conv 的成本大致正比于:

$$
\text{FLOPs} \propto C_{out} \cdot C_{in} \cdot K^2 \cdot H_{out} \cdot W_{out}
$$

- $C_{out}$: 输出通道数
- $C_{in}$: 输入通道数  
- $K$: kernel size
- $H_{out}, W_{out}$: 输出空间尺寸

所以电子计算里, 减小 $K$ (从 7×7 → 3×3, VGG/ResNet 的设计哲学) 和减小 $H_{out}, W_{out}$ (通过 stride/pooling) 都能 linear 降低 FLOPs. 这就是为什么 ResNet 是 cone-shaped: 32×32 → 16×16 → 8×8 → 4×4 → flatten → FC.

### 1.2 4f Optical System 的成本模型

4f system 完全不同, 它的成本几乎完全由 **optics ↔ electronics 转换次数** 决定, 而单次 inference 的"工作量"由 modulator + camera 的 frame rate 决定:

- 光通过透镜做 2D Fourier transform 是 **at the speed of light**, 复杂度 $O(n^2)$ 但时间是常数 (光只走两个 focal distance)
- 相比之下电子 FFT 是 $O(n^2 \log n)$
- 关键: **分辨率 $n$ 的变化对光的传播时间无影响** — 一个 32×32 image 和一个 256×256 image, 光都只走 $2f$ 距离
- 真正的 bottleneck 是 SLM (spatial light modulator) 和相机的切换速度, 现代 4f 系统约 2 MHz frame rate

所以光学加速的 mantra 应该是: **每次 optical inference 完成 as much work as possible**, 而不是 "保持小 kernel". 传统 CNN 反其道而行之 — 用 3×3 kernel + 高 channel + stride pooling, 每次光学 inference 处理的信息量很小, 还要做很多次 optics-electronics 转换.

### 1.3 Pseudo-negativity 限制

光学 detector 只能读 amplitude 的平方 (intensity), 无法直接表示负值. 一种绕法叫 pseudo-negativity: 把每个 kernel 拆成正/负两份, conv 跑两次, 最后相减:

$$
y = y^+ - y^-
$$

这意味着光学系统对 conv 的"调用次数"本就翻倍, 所以**减少 conv 层数**收益更大.

---

## 2. 4f 系统的物理与数学

### 2.1 系统组成

```
[Laser] →f→ [Lens1] →f→ [Fourier Plane: SLM/kernel] →f→ [Lens2] →f→ [Camera]
```

光从 input 到 output 总共走 $4f$, 因此叫 4f correlator.

### 2.2 Convolution Theorem 的光学实现

Convolution theorem:

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}
$$

其中 $\mathcal{F}$ 是 2D Fourier transform. 4f 系统直接物理实现这个等式:
1. Lens1 在 Fourier 平面成像输入信号的 Fourier 变换
2. 在 Fourier 平面用 SLM/phase mask 做 element-wise 乘 kernel 的频谱
3. Lens2 做 inverse Fourier, 在相机平面得到空间域的卷积结果

### 2.3 Angular Spectrum Method (ASPW) — 用于 simulator

paper 中 simulator 用 ASPW 模拟光传播. 给定入射波前 $U_1(x, y)$, 传播后的波前:

$$
U_2(x, y) = \mathcal{F}^{-1}\left[\mathcal{F}[U_1(x, y)] \cdot H(f_x, f_y)\right] \tag{2}
$$

变量含义:
- $U_1(x, y), U_2(x, y)$: 复振幅在 $(x, y)$ 平面的分布
- $\mathcal{F}, \mathcal{F}^{-1}$: 2D Fourier / inverse Fourier transform
- $H(f_x, f_y)$: free-space 的 transmittance (transfer function), 描述每个空间频率分量 $(f_x, f_y)$ 经过距离 $z$ 传播后的相位变化

### 2.4 Fresnel Diffraction Transfer Function

$$
H_F(f_x, f_y) = \exp\left[jkz - j\pi\lambda z (f_x^2 + f_y^2)\right] \tag{3}
$$

- $k = \frac{2\pi}{\lambda}$: 波数 (wave number), $\lambda$ 是波长
- $z$: 光传播距离
- $\lambda$: 光波长 (paper 中设 532 nm, 绿光)
- $f_x, f_y$: 空间频率 (cycles per unit length)
- 第一项 $jkz$: 全局相位偏移 (与频率无关)
- 第二项 $-j\pi\lambda z (f_x^2 + f_y^2)$: 二次相位 (chirp), 这就是 Fresnel 衍射的核心, 高频分量相位变化更快

### 2.5 透镜的 transmittance

凸透镜在薄透镜近似下:

$$
t_A(x, y) = P(x, y) \exp\left[-j\frac{k}{2f}(x^2 + y^2)\right] \tag{4}
$$

- $P(x, y)$: pupil function, 透镜孔径内取 1, 外取 0
- $f$: 焦距 (paper 中设 10 mm)
- $\exp[-j\frac{k}{2f}(x^2+y^2)]$: 抛物相位, 把平面波聚焦成球面波, 等价于做 Fourier transform

### 2.6 传播距离与像素尺度的关系

$$
z = \frac{N(\Delta x)^2}{\lambda} \tag{5}
$$

- $\Delta x$: 像素物理尺寸 (pixel pitch)
- $N$: 一维像素数
- $\lambda$: 波长

这个公式的直觉: 像素越大 (低空间频率采样), 需要传播更远才能完成一个完整的 Fresnel 衍射周期. paper 选 $\Delta x$ 使 $z$ 正好等于 $f$, 这样每个 focal distance 只需一次 propagation iteration.

### 2.7 Batch Tiling — 充分利用 4K 分辨率

paper 关键的 parallelism 技巧. 把同一个 batch 的多个 input tile 到一个大的 input block 上, kernel padding 到同尺寸, 一次 optical inference 同时算出多个样本的 conv:

$$
n = \left\lfloor \frac{R}{M + N - 1} \right\rfloor^2 \tag{6}
$$

- $R$: 4f system 总分辨率 (如 4K = 3840)
- $M \times M$: 单个 input 尺寸 (需要先 padding)
- $N \times N$: kernel 尺寸
- $M + N - 1$: linear conv 输出尺寸, 所以 input 和 kernel 都需 pad 到这个大小
- $\lfloor \cdot \rfloor$: floor function
- 平方因为 2D 排布

举例: $R = 3840$, $M = 32$, $N = 3$, 则 $n = \lfloor 3840/34 \rfloor^2 = 112^2 \approx 12544$ 个样本可以一次 inference 同时算. 这就是为什么 paper 说 batch 3136 时 FatNet 在 optics 上速度反超 GPU.

---

## 3. FatNet 的构造规则

把任意 cone-shaped classifier 转成 FatNet 的四条规则:

| Rule | 内容 | 直觉 |
|------|------|------|
| 1 | 保持层数不变 | 保持非线性激活数量, 保留 depth 的表达能力 |
| 2 | 浅层不动, 直到 feature map 像素数 ≤ 类别数 | 早期层分辨率还没降下来, 没必要动 |
| 3 | 每层输出总像素数 (channels × H × W) 保持不变 | 控制信息瓶颈, 与原网络 capacity 对齐 |
| 4 | 每层可训练参数数保持一致 | 公平比较, 控制过拟合风险 |

### Table 1 详细解读 — ResNet-18 → FatNet

让我详细看一层: 第二层 (原 ResNet-18 中 `128 → 128, k=3×3`):

- 原始: 输入 128 channels, 输出 128 channels, 3×3 kernel
  - weights = $128 \times 128 \times 3 \times 3 = 147456$
  - feature pixels = $128 \times 8 \times 8 = 8192$ (CIFAR-100 在第 2 层空间 8×8)

- FatNet 对应: `82 → 82, k=5×5`
  - channels 降到 82 (因为 feature map 保持 10×10 而不是 8×8, pixel 数比原来多, channel 比例降低)
  - pixel 数 = $82 \times 10 \times 10 = 8200$ ≈ 8192 (rule 3 保持)
  - weights = $82 \times 82 \times 5 \times 5 = 168100$ — 略多于原值, paper 提到这里有 violation 但为了不 underfit

更深层 (例如 `512 → 512, k=3×3` → `21 → 21, k=73×73`):
- 原始 weights = $512 \times 512 \times 9 = 2359296$
- FatNet weights = $21 \times 21 \times 73 \times 73 = 2336409$ — 几乎完全保持
- 注意 kernel 73×73 远大于 feature map 10×10! 但 same padding 让有效 kernel 中心 10×10 起作用, 外圈被 padding "屏蔽"

### 最后一层处理

paper 提到: kernel 大于 input 时, outer regions 不训练 (因为 same padding + small input). 所以最后层 `21 → 1, k=49×49` 而不是 73×73, 通过增加 channel (rule 3 违反) 来保持参数量.

输出层: 1 个 channel, 10×10 spatial, flatten 成 100 维向量对应 CIFAR-100 的 100 个类别. 全卷积, 无 FC layer — 这就是与 4f 系统 100% 兼容的关键.

---

## 4. ResNet-18 Backbone 的修改

paper 没用原版 ResNet-18, 改了几处:

1. **移除 stride**: 4f 系统做不了 strided convolution (光学上没有简单的 stride 操作), 所以用普通 conv + 2×2 MaxPooling 替代
2. **跳过第二个 non-residual conv**: 适配 CIFAR-100 的 32×32 小图
3. 第一个 conv 后用 2×2 MaxPool 代替 stride=2

公式 (1) 是标准 ResNet block:

$$
y = F(x, \{W_i\}) + x
$$

- $x$: residual block 输入
- $y$: 输出  
- $F(x, \{W_i\})$: 由权重集合 $\{W_i\}$ 参数化的残差映射 (通常 2 个 conv + ReLU)
- $+x$: skip connection, identity mapping

这条 skip 是 4f 实现时的头疼点 — 光学系统做加法需要额外光学元件 (beam splitter + 额外路径), 但 paper 假设 hybrid electronic-optical, 加法可以在 electronics 侧做.

---

## 5. 实验结果与解读

### Table 2: Accuracy vs Conv Ops

| Architecture | Test Accuracy | Conv Ops | Ratio |
|---|---|---|---|
| ResNet-18 (GPU) | 66 ± 1.4% | 1,220,800 | 1.0 |
| FatNet (GPU) | 60 ± 1.4% | 148,637 | 0.12 |
| FatNet (Optical sim) | 60% | 148,637 | 0.12 |

- Accuracy 降 6 个百分点 (66% → 60%)
- Conv ops 减少 **8.2 倍** (1,220,800 → 148,637)
- 光学 simulator 与 GPU 训练的 FatNet 一致 (60% = 60%), 证明 simulator 的物理近似足够好

### Table 3: Inference Time (秒/样本)

| Architecture | Batch 64 | Batch 3136 |
|---|---|---|
| ResNet-18 (GPU) | 1.350e-4 | 1.167e-4 |
| FatNet (GPU) | 4.565e-4 | 7.942e-4 |
| ResNet-18 (Optics) | 3.815e-2 | 7.786e-4 |
| **FatNet (Optics)** | 4.645e-3 | **9.479e-5** |

几个关键 observation:

1. **GPU 上 FatNet 反而比 ResNet-18 慢** (4.565e-4 vs 1.350e-4 at batch 64). 因为大 kernel 在 GPU 上是计算密集型, 没法用 GPU 擅长的小 kernel batch parallelism. 这反向证明 FatNet 是为 optics 量身定做.

2. **Optics 上 batch 64 时 ResNet-18 极慢** (3.815e-2, 比 GPU 慢 280 倍!). 因为小 batch 浪费 4f 系统的高分辨率 parallelism, 大量像素空闲.

3. **Batch 3136 时 FatNet Optics 是全场最快** (9.479e-5), 比 ResNet-18 GPU (1.167e-4) 还快 1.2 倍, 比 ResNet-18 Optics (7.786e-4) 快 8 倍. 这就是 batch tiling + FatNet 高分辨率策略的胜利.

4. ResNet-18 在 Optics 上即使 batch 3136 也只有 7.786e-4, 因为它 cone-shape 导致深层 feature map 太小 (4×4), batch tiling 的 $n = \lfloor R/(M+N-1) \rfloor^2$ 在小 $M$ 时反而受限, 而且每层都要 optics-electronics 转换.

### Figure 4 训练曲线

- 三种网络 train accuracy 都到 99%, 但 ResNet-18 收敛更快
- Validation accuracy 上 ResNet-18 ~66%, FatNet 和 optical sim ~57-58% (test 60%, 因为 train/val 用了 augmentation 而 test 没用, gap 来自 augmentation difficulty)

### 训练成本

- 标准 FatNet epoch: 15 秒 (PyTorch Conv2d)
- Optical simulation FatNet epoch: **67 分钟** — 慢 268 倍, 因为 ASPW 把光传播加入了 computation graph, 每个传播步骤都要存梯度. 这就是为什么 optical sim 只跑了部分实验.

---

## 6. 4f 系统的几个物理限制 (paper 没完全解决)

### 6.1 非负约束
Camera 读 intensity = amplitude², 没法直接得到负值. Pseudo-negativity 让 conv 数量翻倍, 部分 offset 了 FatNet 节省的 conv ops.

### 6.2 180° 旋转
几何光学导致 4f 输出相对输入旋转 180°. CNN 不在乎 (因为 conv 是 translation/rotation-equivariant 学习出来的), 但需要架构感知.

### 6.3 No stride
4f 没法原生做 strided conv. paper 用 pooling 替代. 这其实是个有 signal 含义的 limitation: stride 在 CNN 里承担了 multi-scale 信息聚合的作用, FatNet 完全靠大 kernel 模拟类似 receptive field.

### 6.4 Alignment
Free-space optics (vs silicon photonics) 对物理 alignment 敏感, 任何 lens 偏移都会导致输出错乱. paper simulator 没建模这个.

### 6.5 Quantization & Noise
没建模. 但作者认为 noise 可作 regularization (类似 [mixup](https://arxiv.org/abs/1710.09412) 或 dropout 的效果), 量化低 bit 可能不影响 accuracy 太多.

---

## 7. 我对这篇 paper 的整体判断

### 优点
- **核心 idea 反直觉但合理**: 摆脱 "小 kernel + 多 channel" 的电子时代习惯, 重新思考 optical 加速下的架构搜索空间. 这正是硬件-aware neural architecture 的精髓.
- **公平比较**: 保持参数量、层数、激活数对齐, 隔离架构 shape 这个变量
- **提供 simulator 开源** ([GitHub](https://github.com/riadibadulla/simulator)), 可复现

### 弱点 / 可质疑点
1. **6% accuracy 损失非平凡**. 在 CIFAR-100 这种 100 类任务上掉 6 个点是很大的退化. 大 kernel 是否真的需要那么大? 73×73 kernel on 10×10 feature map 的 effective 参数利用率极低 (中心 ~10×10 真正训练), 这与 paper 自己说"outer regions 不训练"矛盾.
2. **Batch 3136 不现实**. 实际部署中很少有这么大的 batch, 而且 SGD 在 batch > 1024 时常有 generalization gap (见 [Goyal et al., 2017](https://arxiv.org/abs/1706.02677))
3. **没考虑 alignment / quantization / noise**. 真实光学系统的 noise floor 和 dynamic range 限制可能让 60% 进一步降.
4. **CIFAR-100 32×32 太小**, 论证光学"高分辨率优势"需要更大输入 (ImageNet 224×224) 才能体现 — paper 自己承认 GPU memory 限制没跑 ImageNet
5. **ResNet-18 backbone 已经是 2016 的老架构**, 现代 EfficientNet / ConvNeXt 的 channel-scaling 平衡点和 FatNet 的设计哲学可能有冲突

### 对未来工作的启发
- 跟 [Diffractive D²NN](https://www.science.org/doi/10.1126/science.aat8084) 这类 all-optical network 互补: D²NN 是层数全光学, FatNet 是架构全卷积化
- 与 [Channel Tiling (Li et al.)](https://arxiv.org/abs/2001.06912) 互补: channel tiling 适合多 channel 网络, FatNet 走 batch tiling 路线
- 类似 spirit 的工作: [Large Kernel Matters (GCN)](https://openaccess.thecvf.com/content_cvpr_2017/html/Peng_Large_Kernel_Matters_CVPR_2017_paper.html) 已经在 segmentation 上证明大 kernel 有用, FatNet 把这个 insight 推到极限

---

## 8. 参考 & 延伸阅读

主 paper:
- FatNet (this paper): arXiv 2210.14134 — [link](https://arxiv.org/abs/2210.14134)
- Code: https://github.com/riadibadulla/simulator

相关工作:
- 4f channel tiling (Li et al.): https://arxiv.org/abs/2001.06912
- Hybrid optical-electronic CNN (Chang et al. 2018, Scientific Reports): https://www.nature.com/articles/s41598-018-30619-y
- Diffractive D²NN (Ozcan group, Science 2018): https://www.science.org/doi/10.1126/science.aat8084
- Massively parallel amplitude-only Fourier NN (Miscuglio, Optica 2020): https://opg.optica.org/optica/abstract.cfm?uri=optica-7-12-1812
- Silicon photonics MZI NN (Shen et al., Nature Photonics 2017): https://www.nature.com/articles/nphoton.2017.93
- ResNet original: https://arxiv.org/abs/1512.03385
- Large Kernel Matters (GCN): https://arxiv.org/abs/1703.02719
- Fourier optics textbook (Voelz): https://spiedigitallibrary.org/ebooks/Computational-Fourier-Optics-A-MATLAB-Tutorial/eISBN-9780819482047
- CIFAR-100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Accompanying research on ONNs survey: https://ieeexplore.ieee.org/document/9023086

---

## 9. 总结一句话直觉

FatNet 揭示一个被忽视的事实: **CNN 的 cone-shape 不是深度学习的固有最优, 而是电子计算成本模型下的局部最优**. 一旦把成本函数换成 optical (resolution-free, conversion-bottlenecked), 架构最优解就 barrel 化 — 大 kernel, 高分辨率 feature map, 少 channel, 全卷积无 FC. paper 用 8.2× conv reduction 换 6% accuracy 损失, 在 optical deployment scenario 下是值得的 trade-off, 而且随着 SLM/camera frame rate 改善, 这个 trade-off 会越来越 favorable.

对 Andrej 你来说, 这篇 paper 可能最有共鸣的点在于: 它本质是在做 **hardware-aware NAS** (neural architecture search) 的手工版本 — 但用的 cost metric 不是 FLOPs / params, 而是 "optics-electronics conversion count + resolution utilization". 未来如果把这个 metric 嵌入 DARTS / differentiable NAS, 应该能搜出比手工 FatNet 更好的架构.
