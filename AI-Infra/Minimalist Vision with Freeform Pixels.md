---
source_pdf: Minimalist Vision with Freeform Pixels.pdf
paper_sha256: 78a10fe0db6c7c103ea6e1ce66a8401493141261f88c94701921ea6040c38aa5
processed_at: '2026-08-05T18:33:20-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直白的话来讲，这篇 paper 做了一件极其优雅的事情：**它把神经网络的 first layer 剥离出来，用塑料片和感光元件在物理世界上 3D 打印出来了。**

我们平时搞 deep learning，input 都是一张分辨率为 $H \times W$ 的 image。这篇 paper 的作者 (Nayar 组) 觉得，对于很多 "lightweight vision" 的任务（比如数数房间里有几个人、测测交通车速、看看灯关没关），用几百万个 pixel 去拍一张完整的图简直蠢透了。

所以，他们搞了个 "freeform pixel"。传统的 pixel 就是正方形，捕捉一小块面积的光。freeform pixel 的形状是任意学出来的，本质就是一个贴在光敏二极管 前面的 mask (遮罩)。

### 物理上的直觉

想象你在一个窗户上贴一张黑色的剪纸，剪纸上有几个形状奇怪的小洞。外面的风景透过小洞，在屋里的墙上形成光斑。如果你只测量这个光斑的总亮度，你得到的是什么？

就是外面风景和剪纸形状的乘积的积分。

paper 里的公式 (1) 就是干这个的：

$$ p = \iint_{x,y} I(x,y) M(x,y) dx dy $$

- $I(x,y)$: 外面的 scene 投影到 mask 平面上的光场。
- $M(x,y)$: 你贴的剪纸的透射率，$0$ 到 $1$ 之间。
- $p$: 光敏二极管 读到的数字。

你看这个公式，这不就是 fully-connected layer 里的 dot product 吗？$\mathbf{p} = \mathbf{W}^T \mathbf{I}$。剪纸的形状就是 weights $\mathbf{W}$。8 个 freeform pixel 就是 8 个 output neurons。物理光学直接帮你把第一层矩阵乘法在光速下、零能耗地算完了。

### 硬件的现实毒打 (Sensor Model)

如果你只拿公式 (1) 去训练，学出来的剪纸贴到硬件上肯定废了。因为真实的物理世界很恶劣。作者在 paper 里展现了极强的工程能力，他们把所有物理限制都塞进了 forward pass 里。

公式 (2) 考虑了光敏二极管的面积 (会 blur 掉图像) 和它的角度响应 (边缘光线会衰减)：

$$ p_d = \iint_{x,y} (I(x,y) * b(x,y)) M(x,y) d(x,y) dx dy $$

- $b(x,y)$: active area 造成的 blur kernel。
- $d(x,y)$: directional response，类似于 vignetting effect。

公式 (3) 和 (4) 更绝，考虑了放大器增益、读出噪声和饱和。特别注意的是饱和处理：

$$ p_f = \begin{cases} p_n & p_n \leq p_{max} \\ \alpha (p_n - p_{max}) + p_{max} & p_n > p_{max} \end{cases} $$

如果光太强，探测器饱和了，直接 clip 掉的话梯度就变成 0，网络就学不动了。所以作者在这里用了一个类似 Leaky ReLU 的 trick，在饱和区域保留一个极小的斜率 $\alpha$，让 gradient 能继续 backpropagate。这就叫 hardware-aware training。如果不这么干，训出来的 mask 贴到硬件上，RMSE 直接从 0.93 飙到 2.28。

### 整个 pipeline 怎么跑通的

1. 架一个普通的 high-res camera 拍一段视频。
2. 把视频喂进一个包含 sensor model 的 differentiable network 里。第一层 weights 就是 $M(x,y)$。
3. 开始训练，loss 是 task loss (比如数人头数)。
4. 梯度不仅更新后面的 inference network，还回传更新 first layer 的 $M_t(x,y)$。这里用 sigmoid 把无界的 $M_t$ 压到 $(0,1)$ 区间内。
5. 训练结束后，把学出来的 first layer weights 用普通的 inkjet printer 打印在透明塑料片上。
6. 把塑料片插到 24 个光敏二极管前面。
7. 把这 24 个二极管读出的数字直接喂给后面那半截训好的 inference network。

任务完成。你得到了一个只有 8 个 pixel 的 camera，性能相当于 64x64 (4096 个 pixel) 的传统 camera。这是 2048 倍的 compression。

### 为什么这事极其 cool (我的视角)

我天天讲 Software 2.0，讲 everything is learned。这篇 paper 是 Software 2.0 把触角伸进物理硬件的极致体现。

通常我们做 model compression 是在数字层砍权重。这篇 paper 直接在物理层砍 measurements。为什么能砍？因为 task 的 intrinsic dimensionality 极低。数房间里几个人，根本不需要知道脸长啥样，只需要测一个空间内的总体光强变化模式。传统 camera 强行采集了千万个 pixel 的冗余信息，然后再用 CNN 费劲巴力地 throw away 信息。这纯属脱裤子放屁。Nayar 的做法是：**在光子撞上传感器的那一刻，就把无关信息物理过滤掉。**

而且这个架构直接白嫖了两个极其诱人的 emergent property：

1. **Privacy Preservation (隐私保护)**:
   它只有 8 个数字输出。这 8 个数字是为 "数人头" 优化的线性投影。作者试过用这 8 个数字去恢复人脸身份 (CelebA dataset)，准确率只有 2.0% (瞎猜都有 1%)。Information bottleneck 直接在物理层面卡死了，你永远无法从 8 个为 counting 优化的标量里重建出一张高清人脸。这对监控摄像头普及的隐私争议是完美的解药。

2. **Self-Powered (自供电)**:
   普通图像传感器功耗几百毫瓦，因为要 readout 几百万个 pixel 并做 ADC 转换。这玩意儿只有 24 个 pixel，功耗极低。作者在 camera 侧面贴了 4 块小小的太阳能板，在 600 lux 的普通室内光线下，靠环境光收集的能量就足够驱动 30 FPS 的 readout 和 Bluetooth Low Energy (BLE) 无线传输。没有电池，没有电线，贴在墙上能一直跑。

### 细节联想与扩展

顺着你刚才让我多联想的指示，我展开说几个点：

**关于 Sim-to-Real Gap 的处理**
作者在 Supplementary C 里提到，仿真训出来的 mask 拿到真实硬件上会有 mismatch (塑料打印机的灰度只能打 0.01 到 0.67，几何对齐也有误差)。他们的解决办法非常务实：freeze 住打印出来的物理 mask，然后在真实硬件上收集数据，只 retrain 后面的 digital inference network。这把 "光学层" 和 "数字层" 的学习解耦了，非常像我在 Tesla 搞 autonomous driving 时的 sensor calibration 逻辑——感知网络要 robust 到能容忍前端物理的轻微 mismatch。

**关于 Temporal 维度**
在 Traffic monitoring 实验里，单帧测不出速度。作者把 1 秒内 30 帧的 8 个 pixel 值 stack 起来，做个 forward differencing，扔进 MLP 里去估车速 (RMSE 2.30 mph)。这说明 freeform pixel 在时间维度上也是 usable 的。如果未来把打印的塑料片换成 Spatial Light Modulator (SLM, 比如小 LCD)，mask 就能随时间动态变化。这相当于在光速上做 time-multiplexing，8 个物理 pixel 就能时分成几百个 virtual pixel。这绝对是下一步的引爆点。

**与 Diffractive Optics 的对比**
UCLA 的 Aydogan Ozcan 组搞了很多 diffractive deep neural networks (D2NN)，用光波相位穿过几层 3D 打印的塑料板来算 conv。那是真正的光学计算。这篇 paper 相对简单，只做了 linear projection，用空间振幅 mask。但它的巧妙之处在于目标函数不同：D2NN 是为了替代整个网络做 inference，这篇 paper 是为了极小化 measurement 数量以换取 privacy 和 power efficiency。目标决定了架构的取舍。

### References

- 原文 PDF: https://www.cs.columbia.edu/CAVE/publications/ (找 Klotz & Nayar 最近的 paper)
- 前作 The Minimalist Camera (BMVC 2018): https://bmva-archive.org.uk/bmvc/2018/contents/papers/0129.pdf
- 前作 Self-Powered Camera (ICCP 2015): https://www.cs.columbia.edu/CAVE/publications/pdfs/Nayar_ICCP15.pdf
- Deep Optics 综述: https://www.nature.com/articles/s41586-020-03093-1
- Diffractive Neural Networks (Science 2018): https://www.science.org/doi/10.1126/science.aat8084
- Software 2.0 essay (我自己写的): https://karpathy.medium.com/software-2-0-a6eb4a0f4ad1

---

# Minimalist Vision with Freeform Pixels 深度讲解

## 1. 核心直觉：把光学当成可学习的网络第一层

这篇 paper 来自 Columbia University 的 Jeremy Klotz 与 Shree K. Nayar (Nayar 是计算摄影领域的传奇人物, cybernetics / BRDF / catadioptric 等等都有他的贡献). 核心想法可以用一句话概括: **如果任务只需要"轻量级"视觉推理 (lightweight vision), 我们完全可以放弃传统百万像素的 grid, 转而让光学硬件本身学习出几个任意形状的 "freeform pixels", 直接把 task-relevant 信息压缩到 handful 个 measurements 里**.

这里的关键 conceptual move 在于: 把 camera 的光学层和后面的 inference network 视作同一个端到端可微网络. 第一层 weights 的物理对应就是 optical mask 的 transmittance function $M(x,y)$, 训练完之后直接 print 出来贴到 photodetector 前面. 整个 pipeline 就成了:

```
Scene → [Mask M(x,y), 学出来的] → Photodetector → ADC → [Inference Net] → Task output
        └─── 第一层, 物理实现 ──┘                              └── 数字层 ──┘
```

这与 deep optics (Sitzmann et al. 2018, Tseng et al. 2021 Nature Comm) 的思想同源, 但目标截然不同: deep optics 做的是用学习改进 image quality 或 task performance, 这里做的是 **极小化 measurement 数量**, 同时附带两个 emergent property — privacy preservation 与 self-sustainability.

参考链接:
- Nayar 的 Columbia CAVE 实验室: https://www.cs.columbia.edu/CAVE/
- Deep optics survey: https://www.nature.com/articles/s41586-020-03093-1
- 论文 PDF (作者主页): https://www.cs.columbia.edu/~nayar/papers/
- 先驱 paper "The Minimalist Camera" (Pooj et al. BMVC 2018): https://bmva-archive.org.uk/bmvc/2018/contents/papers/0129.pdf

---

## 2. Freeform Pixel 的数学

### 2.1 最朴素的形式 — 公式 (1)

设 $I(x,y)$ 为 3D 场景以 detector 中心为投影中心投影到 mask 平面后的 2D irradiance 分布. $M(x,y) \in [0,1]$ 为 mask 的 transmittance. 那么 detector 测量值为:

$$
p = \iint_{x,y} I(x,y) \, M(x,y) \, dx \, dy \tag{1}
$$

变量含义:
- $I(x,y)$: scene irradiance at mask plane, 即"如果没 mask, 该位置接收到的光强". 它本质上是 3D scene 的一个 perspective projection, 投影中心是 detector 的位置.
- $M(x,y)$: 透射率函数, 取值 $[0,1]$, 1 表示完全透光, 0 表示完全挡住.
- $p$: 最终落到 detector 上的光能量 (积分后的标量).

直觉: 每个 freeform pixel 就是对 scene 做一次 **加权积分** (inner product). 如果把 $I$ 拉平成长向量 $\mathbf{I} \in \mathbb{R}^N$, 把 $M$ 拉成 $\mathbf{m} \in \mathbb{R}^N$, 那么 $p = \mathbf{m}^\top \mathbf{I}$. 

这就是为什么一束 freeform pixels 可以完全等价于一个 fully-connected layer (无 bias). 一组 $K$ 个 freeform pixels 就对应一个矩阵 $\mathbf{W} \in \mathbb{R}^{K \times N}$, 它把"完整图像"$\mathbf{I}$ 投影成 $K$ 维 measurement 向量. 训练目标就是同时学 $\mathbf{W}$ (mask 物理实现) 与后续 inference 网络.

### 2.2 Square pixel 只是 freeform pixel 的特例

传统 square pixel 的 mask 是一个 box function:

$$
M_{\text{square}}(x,y) = \mathbf{1}_{[x_0, x_0+\Delta]\times[y_0, y_0+\Delta]}(x,y)
$$

所以 baseline camera (传统 grid) 可以看作 "mask 固定为 box 函数的 minimalist camera 的退化形式". 这给了一个非常干净的对比实验: 固定 mask 为 box, 让 detector 数量从小到大变化, 与 learned freeform mask 比较.

### 2.3 Sigmoid parameterization — 让 mask 可微且 bounded

由于 mask 必须 $\in [0,1]$, 但网络 trainable parameter 希望无界 (Adam 之类). 引入:

$$
M(x,y) = \sigma\big( M_t(x,y) \big), \quad \sigma \text{ 是 sigmoid}
$$

$M_t(x,y) \in \mathbb{R}$ 是真正的 trainable tensor, 经过 sigmoid 之后被压缩到 (0,1). 训练梯度可以无障碍地回流到 $M_t$. 在实际硬件中, 由于 inkjet 打印的 transparency 只能实现 $M \in [0.01, 0.67]$, 训练时作者把 mask 值 rescale 到这个区间, 让 sim-to-real gap 尽量小.

---

## 3. Sensor Model — 为什么不能省略

这部分是 paper 的关键工程贡献. 如果只把公式 (1) 放进网络训练, 学出来的 mask 在真实硬件上几乎不可用. 原因是真实 detector 有方向性响应、有 active area、有噪声、有 saturation. 必须把所有这些物理效应都嵌进 forward pass, 才能让学出来的 mask "知道"自己将要面对的硬件现实.

### 3.1 Optics — 公式 (2)

引入两个效应:

1. **Detector directional response** $d(x,y)$: 入射光角度 $\theta$ 越大衰减越多, 行为类似 vignetting. $d$ 是关于方向的函数, 但可以重写为 mask 平面上 (x,y) 的函数, 因为每个 (x,y) 对应一个固定的入射角.
2. **Active area blurring** $b(x,y)$: detector 不是点, 是有面积的, 所以 scene irradiance $I$ 先被一个 kernel $b$ blur 一下, 等价于 $I * b$.

加上之后:

$$
p_d = \iint_{x,y} \big( I(x,y) * b(x,y) \big) \, M(x,y) \, d(x,y) \, dx \, dy \tag{2}
$$

变量:
- $b(x,y)$: blur kernel, 宽度等于 detector active area 物理宽度. Hamamatsu S9119-01 的 active area 是 $0.88 \times 0.88$ mm², 大约对应到 mask 平面上一个 small Gaussian-like patch.
- $d(x,y)$: directional attenuation, 中心为 1, 边缘衰减. 这在 FOV = $70° \times 70°$ 时不可忽略.

这两个效应都是关于 (x,y) 的已知函数 (从 detector datasheet 测量), 在 forward pass 中作为 fixed multiplier/filter 出现. 训练时 $I$ 已知 (从 training camera 拿到), $b, d$ 已知 (从 datasheet), 唯一的 trainable 参数是 $M_t$ (经过 sigmoid 后变成 $M$).

### 3.2 Detector — 公式 (3) 和 (4)

$$
p_n = G \, p_d + n_r + n_q \tag{3}
$$

- $G$: transimpedance gain, 实验中是 $10^7$ V/A, 把 photocurrent 转成 voltage.
- $n_r \sim \mathcal{N}(0, \sigma_r^2)$: read noise, Gaussian, $\sigma_r = 400\,\mu V$.
- $n_q \sim \mathcal{U}(0, p_{lsb})$: quantization noise, uniform, $p_{lsb}$ 是 ADC 的 least significant bit 对应的电压.

这两种噪声放在 forward pass 里, 训练时每个 batch 重新采样, 让网络对噪声鲁棒. 这非常重要 — 否则学出来的 mask 可能"开得很小", 只让极少光通过, 实际硬件上一加噪声 SNR 就崩.

**Saturation 的 clipping trick — 公式 (4)**:

$$
p_f = \begin{cases} 
p_n & p_n \leq p_{max} \\ 
\alpha (p_n - p_{max}) + p_{max} & p_n > p_{max} 
\end{cases} \tag{4}
$$

- $p_{max}$: detector saturation threshold (实验中 3.2 V, 16-bit ADC).
- $\alpha$: 一个小的正数 (类似 leaky ReLU 的负斜率), 用来避免 saturation 区域的 zero gradient.

为什么必须这样写? 如果直接 hard-clip (即 $\min(p_n, p_{max})$), 在 saturation 区域 $\partial p_f / \partial M_t = 0$, gradient 直接死掉, mask 无法继续优化. 用 leaky 形式后, 即使饱和也有微小梯度回流, 训练能继续. 这是一个非常 deep-learning-flavored 的工程细节, 但在 hardware-aware training 里极其常见 (类似量化网络里的 straight-through estimator).

### 3.3 为什么 sensor model 不能省 — Supplementary 的对照实验

Supplementary A 给了 quantitative evidence:
- 不加 sensor model 训练出的 4 个 freeform pixels, 后期 frozen mask 并 retrain inference net, RMSE = 2.28
- 加 sensor model 训练出的 4 个 freeform pixels, 同样 frozen mask + retrain, RMSE = 0.93

差距 2.45x, 充分说明 sensor model 决定了 sim-to-real transfer 的成败. 这和 sim-to-real 在 robotics 里的教训一致 — 真实物理过程必须出现在 forward pass 里, 否则学到的 policy 在部署时会崩.

---

## 4. 网络架构 — end-to-end 可微 pipeline

整个 minimalist vision system 可以画成:

```
                      ┌──────────── 物理层 ────────────┐    ┌──── 数字层 ────┐
Training Camera Image → I(x,y)                       →    → Inference Net → Task output
                        │                                 │
                        ├ * b(x,y) [blur]                 │  2 hidden layers
                        ├ × M(x,y) = σ(M_t) [learnable]  │  each 128 units wide
                        ├ × d(x,y) [directional]          │  LeakyReLU activation
                        ├ ∫∫ dx dy [spatial integration]  │
                        ├ × G [gain]                      │
                        ├ + n_r, n_q [noise]              │
                        └ clip with leaky slope α         │
                                                          │
                        K 个 freeform pixels 输出 K 维向量 →
```

注意这里有几个关键的 "shape-preserving" 操作:
- $I \in \mathbb{R}^{H \times W}$ 是 training camera 给出的高分辨率图像
- $M \in \mathbb{R}^{K \times H \times W}$ — K 个 mask, 每个 mask 与 I 同尺寸
- 第一层输出 $\mathbf{p} \in \mathbb{R}^K$ — K 个标量 measurement
- 后续是普通 MLP, 输入 $\mathbf{p}$, 输出 task label

loss 函数视任务而定:
- People counting: MSE between predicted count and ground truth
- Door state / zone occupancy: cross-entropy
- Lighting state: cross-entropy per light
- Traffic speed: MSE in mph

初始化: $M_t \sim \mathcal{U}(0.08, 0.12)$ — 注意这里直接初始化的是 sigmoid 之前的 $M_t$, 这样 $M = \sigma(M_t)$ 大约在 $\sigma(0.08) \approx 0.52$ 到 $\sigma(0.12) \approx 0.53$, 即初始 mask 几乎是 ~52% 透射率的均匀灰片. 这给了网络一个"什么都不做"的起点, 然后慢慢分化出 task-relevant 的 spatial pattern.

Optimizer: Adam, learning rate 通过 grid search. 数据集大小: toy example 用 1M 训练 + 100K val + 250K test, workspace monitoring 用 40 分钟视频 (68K 帧), traffic 用 166 分钟 (24K clips).

---

## 5. Toy Example — Counting Patches

这是验证 freeform pixel 信息容量的最干净的 synthetic experiment.

### 5.1 数据生成

- 图像里随机放置 0~10 个 patches
- 每个 patch: random position, random brightness, random size (within a range)
- 允许 partial overlap (occlusion simulation)
- 乘以 smooth sinusoidal illumination (random 参数) 来模拟 local illumination variation
- 1,000,000 训练 / 100,000 验证 / 250,000 测试

这个 setup 覆盖了 real-world counting task 的几个关键 nuisance: occlusion, illumination variation, position variation. 如果 freeform pixels 在这种条件下工作良好, 就说明它们确实在抓 task-relevant 信息, 不是 overfit 到某个 specific configuration.

### 5.2 结果 — Fig 4(c)

| Camera | Pixels | RMSE (counting 0-10) |
|--------|--------|----------------------|
| MinCam | 4 freeform | 0.71 |
| Baseline | 32×32 = 1024 square | ~0.71 |

256× reduction. 也就是说, 4 个学出来的 freeform pixels 在 counting task 上等价于 1024 个 square pixels.

直觉解释: counting 的本质是检测 "有几个亮度 blob", 这只依赖于图像的某些 aggregate statistics (例如总能量、二阶矩), 与具体位置关系不大. Square pixel 由于其 rigid 的空间划分, 必须用很多格子才能把 "number of blobs" 这个统计量恢复出来. 而 freeform pixel 可以学成不同尺度的 Gaussian-like kernel, 类似 multi-scale blob detector, 直接把"几个 blob"这种信息编码到几个标量里.

理论上, counting 与 image moment 有强关联. 一个各向同性 Gaussian mask 大致测量 $\int I \cdot \exp(-r^2/2\sigma^2)$, 类似 "weighted blob count". 多个不同尺度的 Gaussian 组合, 可以恢复出 multi-scale structure, 这与传统 pyramid + blob detection 思路一致. 论文里的 Fig 4(b) 显示学到的 mask 确实呈现 blob-like 形态, 印证这个直觉.

---

## 6. Workspace Monitoring — 真实场景第一个实验

### 6.1 Setup

场景: 室内 workspace, 人们进出, 走动, 占据不同 zones. 任务有 3 个:
1. **People counting**: 房间里 0-8 人
2. **Zone occupancy**: 4 个 zones 各自是否被占据 (binary)
3. **Door state**: 门是否打开

捕获 1 小时视频, 分割: 40 min training / 10 min val / 10 min test.

### 6.2 结果 — Fig 7

| System | Pixels | RMSE (people count) |
|--------|--------|---------------------|
| MinCam | 2 freeform | 0.68 |
| Baseline | 64×64 = 4096 | ~0.68 |
| MinCam | 8 freeform (hardware) | 1.10 (prototype real measurement) |
| MinCam | 16 freeform | ~0.55 (simulated) |

**2 freeform pixels ≈ 64×64 baseline → 2048× reduction**. 这是非常惊人的数字. 直觉: counting 0-8 人本质上是 ordinal regression, 可以由两个 well-designed 的 spatial kernel 捕捉 — 例如一个 wide FOV kernel 测总光量 (人多了反射光多), 一个 anti-symmetric kernel 测变化模式. 训练学到的 mask (Fig 7c) 确实呈现这种 complex 多瓣形态.

### 6.3 隐私测试 — 极强的副产物

用 16 个为 counting 训练的 freeform pixels, 再 fine-tune 一层 inference net 做人脸识别. 数据集: CelebA subset, 100 人, 2751 张图. 人脸 scaled 到 cover 整个 FOV, 加 small noise + random gain.

结果: **2.0% recognition rate** (chance level = 1%). 与 SOTA 人脸识别 > 98% 相比, 直接掉到 chance level. 

直觉解释: 这 16 个 measurements 是 counting-relevant 的 linear projections of the image, 16 个数字根本不足以重建 face identity (face identification 通常需要几百维 feature). 这是个 information-theoretic 的 argument — 16 个标量无法保留 100 个身份的判别信息, 除非这 16 个 projection 恰好对人脸判别有利, 而训练目标恰恰是 counting, 所以学到的 projections 是 face-identity-orthogonal 的方向.

这个 property 极其重要: 不需要专门设计 privacy-preserving mechanism, 极小 measurement 数本身就构成 information bottleneck. 这与 differential privacy 的 spirit 一致, 但实现路径完全不同 — 这里是通过物理层 measurement 极小化达成的.

参考: CelebA dataset https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

---

## 7. Room Lighting Estimation — 间接推断

### 7.1 Setup

场景: 3 个 floor lamps + 2 个 overhead light banks. 5 个 light 都对 camera 不可见 (不在 FOV 内). 必须从 scene shading 推断每个 light 的 on/off 状态.

这是一个非常 interesting 的 setup, 因为信息路径是: light → scene surface shading → camera. 这本质上是一个 inverse rendering 问题, 但被压缩成 5-bit classification.

Ground truth 用 fisheye camera (直接看到所有 light) 标注. 训练数据 30 分钟.

### 7.2 结果 — Fig 8

| System | Pixels | Accuracy |
|--------|--------|----------|
| MinCam | 16 freeform (sim) | ~94% |
| Baseline | 12×12 = 144 | ~94% |
| MinCam | 8 freeform (hardware) | 94.0% |

Fig 8(c) 展示了训练过程中 2 个 freeform pixel 的演化: 从 uniform noise 初始化 (灰片), 逐渐分化成有 spatial structure 的 mask. 这是 SGD 在高维 parameter space 里寻找 task-relevant spatial pattern 的可视化.

直觉: 每个 light 在场景表面投射不同的 shading pattern. 例如 floor lamp 在地面附近创造强 gradient, overhead light 创造更均匀的 diffuse illumination. Freeform pixel 学到的就是这些 shading signature 的 matched filter — 一个 pixel 学一个 light 的"特征 mask", 它对某个 light 开启时场景 shading pattern 的响应最强. 这与 Wiener filter / matched filter theory 完全一致.

### 7.3 为什么 8 pixels 够用

5 个 lights, 每个二元状态, 总共 $2^5 = 32$ 个 possible configurations. 8 个 binary measurements 理论上可以编码 256 个状态, 信息容量足够. 但 measurements 是 continuous 的 (受噪声影响), 加上 people 走动会引入干扰, 所以需要 8 而不是 5. 这里有个 implicit redundancy 用于 denoise.

---

## 8. Traffic Monitoring — 时间维度的利用

### 8.1 Setup

任务: 估计两个方向 (left + right) 的平均 traffic speed, 单位 mph.

**Key difference**: 这个任务需要 **temporal history**. 单帧不足以推断速度, 必须观察一段时间内的 measurement 变化. 

输入: 1 秒的 measurement stack (30 帧 × 8 pixels = 240 维). Forward differencing 应用在时间维度, 再送入 inference net. 

Ground truth: YOLOv8 (Ultralytics) 检测 + 跟踪车辆, 计算 ground truth speed.

数据: 全天 8 小时视频, 随机抽 5 分钟 clip 用于 val/test, 其余用于 train.

### 8.2 结果 — Fig 9

| System | Pixels | RMSE (mph) |
|--------|--------|-----------|
| MinCam | 16 freeform (sim) | ~2.0 |
| Baseline | 8×14 = 112 | ~2.0 |
| MinCam | 8 freeform (hardware) | 2.30 |

直觉: traffic speed 的视觉线索是 "average motion magnitude". 对一个静态 camera, 车辆以速度 $v$ 行驶时, image plane 上的 motion 大致是 $\dot{x} \propto v/z$ (perspective), 但 averaged over time 后, 这个量纲被 absorb 到一个 scalar 里. Freeform pixel 学到的就是空间上对 motion 敏感的 projections — 例如 anti-symmetric double-lobe mask 类似 time-derivative filter, 加上 forward differencing 在时间上, 整个系统等价于一个 spatio-temporal gradient detector.

这个实验说明 freeform pixel 不局限于 spatial task, 通过 stack 时间维度, 也能处理 motion 相关任务. 这给未来扩展留了空间: 如果 mask 本身可以是 SLM (spatial light modulator) 电控可变, 那么时间维度也可以编码到 mask 本身, 形成 spatio-temporal freeform pixel.

参考: Ultralytics YOLOv8 https://github.com/ultralytics/ultralytics

---

## 9. 硬件原型 — Table 1 与 Section 5

### 9.1 物理构成

| Component | 数量 | 说明 |
|-----------|------|------|
| Photodiode Hamamatsu S9119-01 | 24 | 0.88×0.88 mm² active area |
| Transimpedance amp TLV521DCKR | 24 | Gain $10^7$ V/A |
| Multiplexer ADG732BCPZ | 1 | 32-to-1 选通 |
| MCU STM32WB5MMG | 1 | BLE 无线 |
| Solar panel PowerFilm MP3-37 | 4 | 一面一个 |
| Supercapacitor | 8 × 11mF = 88mF | 储能 |
| Training camera Basler daA1920-160uc | 1 | 仅用于训练 |
| Training lens Edmund 3mm f/2.5 | 1 | wide FOV |
| IR filter Schott KG3 | 1 | 阻挡 NIR |

### 9.2 光学几何

- Mask size: $16 \times 16$ mm²
- Mask-to-detector distance: 11.4 mm
- FOV per freeform pixel: $70° \times 70°$
- 24 个 mask 全部 print 在一张 transparency film 上, 用 inkjet 打印
- Mask 可以 slide-in/slide-out 替换, 便于不同任务快速切换

### 9.3 制造约束

Inkjet 打印 transparency 的 transmittance 范围有限, 实测 $M \in [0.01, 0.67]$. 训练时把 sigmoid 输出 rescale 到这个 range. 这是 sim-to-real gap 的一个主要来源, 也是为什么训完 inference net 后还要在 hardware 上 retrain.

### 9.4 Calibration gap 处理 — Supplementary C

实际部署时, 模拟 measurement 与真实 measurement 之间有偏差, 来自:
- Sensor model 与硬件不完全 match
- Radiometric 校准误差
- Geometric misalignment (mask 与 detector 相对位置)

解决方案: 
1. 训练阶段: 用 training camera 视频 → 学 mask
2. 部署阶段: 用 hardware 测量 + 同步 training camera 帧 → 在 hardware 上 retrain inference net (mask frozen)

这是一个非常实用的 two-stage 流程, 把 "学光学" 与 "校准硬件" 分开, 避免每次硬件微调都重新学 mask.

---

## 10. Self-Powered Mode — Fig 6

这是 paper 最 striking 的 demo 之一.

### 10.1 能量分析

传统 image sensor (例如手机相机): 100-300 mW. 主因是像素多, readout 与 A/D 转换能量与像素数 linear.

Minimalist camera: 24 pixels, readout 与 BLE transmission 总功耗低到 solar panel 在 600 lux indoor 光照下就够用.

### 10.2 实测

- 4 个 PowerFilm MP3-37 solar panel, 每侧一个
- 88 mF supercapacitor 缓冲能量
- 600 lux indoor 环境 (典型办公室照明)
- 30 FPS readout + BLE transmission
- 完全无外部电源, 无电池

Supercapacitor 用来 smooth 光照波动 — 例如有人遮挡 panel, 短时间内仍能维持工作. 这是 energy harvesting 系统的标准做法.

### 10.3 与之前 self-powered camera 的对比

Nayar et al. 2015 ICCP 的 self-powered camera 用 30×40 = 1200 pixels, harvested energy 只够 readout, 不够 wireless transmission. 这里 24 pixels, 测量少 50×, readout + transmission 全都能搞定. 这是 minimalist vision 在 energy 上的 quantitative win.

参考: Nayar 2015 ICCP https://www.cs.columbia.edu/CAVE/publications/pdfs/Nayar_ICCP15.pdf

---

## 11. 局限与未来方向

### 11.1 当前局限

1. **Mask 是 binary-ish 的静态 print**: 实测 transmittance $\in [0.01, 0.67]$, 不能实现 phase modulation, 也不能动态变化. 这限制了 mask 的表达力.
2. **Linear projection only**: 公式 (2) 是纯 linear. 无法实现 optical convolution (那需要 lens).
3. **Sim-to-real gap 仍存在**: 必须 two-stage retrain. 这增加了部署成本.
4. **任务复杂度有上限**: paper 明确指出, 对 fine-grained 任务 (optical flow, face ID), freeform pixel 数量会逼近传统 camera, minimalist 优势消失.
5. **Temporal 维度处理粗糙**: traffic 任务用 1 秒 stack, forward differencing + MLP. 没有用更 sophisticated 的 temporal model (e.g. RNN, TCN).

### 11.2 未来方向 (Section 7)

1. **SLM mask**: 用 liquid crystal SLM 替代 print transparency, mask 可电控可变. 这开启两个能力:
   - 时间复用: 不同时刻不同 mask, 等价于用 1 个 pixel 时间多路复用实现多 virtual pixels
   - 任务自适应: 不同 task 切换 mask
2. **Lens-augmented freeform pixel**: 加 lens 后, 每个 pixel 可以做 optical convolution with learned kernel. 这把第一层 conv 也搬进光学, 类似 Chang et al. 2018 / Lin et al. 2018 Science 的 diffractive optics 思路, 但用于 minimalist objective.
3. **更复杂 optical mapping**: 例如 metasurface 实现 arbitrary linear operator, 把 freeform pixel 推广到 freeform linear operator.

参考: Diffractive deep neural network https://www.science.org/doi/10.1126/science.aat8084

---

## 12. 更深层的直觉与连接

### 12.1 与 Compressive Sensing 的关系

公式 (1) $p = \langle I, M \rangle$ 在形式上与 compressive sensing 的 measurement 完全一致. Duarte et al. 2008 single-pixel camera 用 thousands of such measurements 重建 image. 这里关键区别在于: minimalist camera **不做重建**, 直接 task inference. 跳过 reconstruction 是巨大节省 — 因为 reconstruction 需要 measurement 数与 image sparsity 和 desired resolution 挂钩, 而 task inference 只需要 measurement 数与 task complexity 挂钩, 后者通常远小于前者.

这给了一个 general principle: **intermediate reconstruction 是 overkill, 直接 task-relevant projection 才是 information-theoretically optimal**.

参考: Single pixel camera https://ieeexplore.ieee.org/document/4472239

### 12.2 与 Information Bottleneck 的关系

Minimalist camera 实现了一个 hardware-level information bottleneck: measurement 数 K 是个 explicit hyperparameter, 直接限制信息流过第一层的 bit rate. 训练在 K 很小的约束下找最优 projection, 自然学到的就是 task-relevant 的低维 manifold.

这与 Tishby 的 information bottleneck theory 一致: $\min_{p(t|x)} I(X;T) - \beta I(T;Y)$, 这里 $T$ 是 measurements, $X$ 是 image, $Y$ 是 label. K 限制相当于对 $I(X;T)$ 加上限. 在 K 极小时, 网络被迫只保留 task-relevant 信息, 自动丢弃 privacy-relevant 信息 (例如 face identity). 这就是 privacy preservation 的 information-theoretic 解释.

参考: Information bottleneck theory https://arxiv.org/abs/physics/0004057

### 12.3 与 Edge Sensing / 软硬件协同设计

这 paper 在更广 context 里属于 "软硬件协同的视觉前端" trend:
- Deep optics (Sitzmann, Tseng, Metzler): 学习光学提升图像质量
- Diffractive neural networks (Ozcan group): 整个网络在光学里
- Optical conv first layer (ASP Vision, Chang 2018): 第一层 conv 在光学
- CANOPIC, Pittaluga Koppal: 用光学保护隐私
- FlatCam, PhlatCam, DiffuserCam: lensless camera, 但目标是 reconstruction

Minimalist vision 在这个谱系里独树一帜: 它把 measurement 数本身当作优化目标, 并以 privacy 和 self-power 为驱动 application. 这是一个非常 "task-first" 的视角, 与传统 "image-first" 视角形成对比.

### 12.4 与我的 (Karpathy) 一直思考的 topic 的连接

我 (Karpathy) 在很多场合讲过 "software 2.0", "the entire stack is learned". 这 paper 是 software 2.0 的极致推论: 连 sensor 都被吸收进 learned system, 第一层 weights 物理实现成 optical mask. 训练 loop 直接对物理硬件做优化, 这是真正意义上的 "differentiable physics-in-the-loop". 

这跟 Tesla 的 photon-to-controls 端到端架构 (我熟悉的领域) 在精神上完全一致 — 都在试图打通从光子到 action 的全链路. 区别在于 Tesla 用大量 pixel 拥抱 rich task, minimal vision 用极小 pixel 服务 lightweight task. 两者结合 (在不同 task 上动态切换 measurement budget) 会是个有趣的 future.

另一个连接点: 极小 measurement 数下网络仍能工作, 暗示 lightweight task 的 intrinsic dimensionality 极低. 这与 manifold hypothesis 一致 — 自然图像虽然 pixel space 是高维, 但 task-relevant manifold 通常极低维. Minimalist camera 用 handful 个 linear projections 就把 task manifold "钉住"了, 这是 manifold 低维的 direct empirical evidence.

参考: Software 2.0 essay https://karpathy.medium.com/software-2-0-a6eb4a0f4ad1

---

## 13. 总结 — Take-aways

1. **Freeform pixel = learned linear projection 物理实现**. 公式 (1) 是核心, 公式 (2)-(4) 把真实硬件物理塞进 forward pass, 是 sim-to-real 的关键.
2. **Lightweight task 的 measurement 复杂度远低于 image 复杂度**. 实验 evidence: counting 2048× reduction, lighting 18× reduction, traffic 14× reduction.
3. **Privacy 是 emergent property**: 16 个 counting-optimized pixels 做 face ID 只有 2% 准确率, 几乎 chance level. 信息瓶颈天然保护隐私, 不需要专门 design.
4. **Self-power 是物理 free lunch**: 24 pixels readout + BLE 在 600 lux 下完全自给, 因为能耗与像素数 linear.
5. **端到端 differentiable 是核心 enabler**: 把 camera 当第一层, 训练完 print 出来, 这是 deep optics 思想在 minimalist goal 上的应用.
6. **适用边界**: fine-grained task (optical flow, face ID) 时 advantage 消失, minimalist vision 不是万能 camera, 是 lightweight vision 的专用工具.

---

## 14. 关键参考链接汇总

| 主题 | 链接 |
|------|------|
| Paper 作者主页 | https://www.cs.columbia.edu/CAVE/ |
| Nayar self-powered camera 2015 | https://www.cs.columbia.edu/CAVE/publications/pdfs/Nayar_ICCP15.pdf |
| Pooj et al. Minimalist Camera BMVC 2018 | https://bmva-archive.org.uk/bmvc/2018/contents/papers/0129.pdf |
| Deep optics survey (Nature) | https://www.nature.com/articles/s41586-020-03093-1 |
| Sitzmann deep optics | https://dl.acm.org/doi/10.1145/3197517.3201333 |
| Tseng neural nano-optics | https://www.nature.com/articles/s41467-021-26443-0 |
| Diffractive DNN (Lin et al. Science) | https://www.science.org/doi/10.1126/science.aat8084 |
| Single pixel camera (Duarte) | https://ieeexplore.ieee.org/document/4472239 |
| FlatCam | https://ieeexplore.ieee.org/document/7880889 |
| PhlatCam | https://ieeexplore.ieee.org/document/9005778 |
| DiffuserCam | https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-5-1-1&id=381709 |
| CelebA dataset | http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| Ultralytics YOLOv8 | https://github.com/ultralytics/ultralytics |
| Hamamatsu S9119-01 datasheet | https://www.hamamatsu.com/eu/en/product/optical-sensors/photodiodes/si-photodiodes/S9119-01.html |
| Information bottleneck theory | https://arxiv.org/abs/physics/0004057 |
| Software 2.0 (Karpathy) | https://karpathy.medium.com/software-2-0-a6eb4a0f4ad1 |
| Pittaluga & Koppal privacy optics | https://openaccess.thecvf.com/content_cvpr_2015/papers/Pittaluga_Privacy_Preserving_Optics_for_2015_CVPR_paper.pdf |

---

这篇 paper 在我 (Karpathy) 看来最 worth pondering 的地方不是它的硬件 demo, 而是它对"什么算 vision" 的重新定义: 传统 vision 默认 camera 给一张完整 image, 然后所有方法在 image 上做. 这 paper 挑战了这层默认, 把 sensor 本身变成 task-driven learned module, 让 measurement 数与 task 复杂度匹配, 而非与 image resolution 匹配. 这种 "match the bandwidth to the task" 的思想, 在 edge AI, IoT sensing, autonomous system 会有长远影响.
