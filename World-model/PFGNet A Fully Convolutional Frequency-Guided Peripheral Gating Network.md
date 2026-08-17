---
source_pdf: PFGNet A Fully Convolutional Frequency-Guided Peripheral Gating Network.pdf
paper_sha256: 66c20401750d0a96bdc56fc1cd972eeb37f47edf392328580037c976a22c0b47
processed_at: '2026-08-06T03:04:10-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 PFGNet

## 这篇论文在干啥

你有一段视频的前几帧，想预测接下来几帧长啥样。比如看交通监控前4分钟，猜后面4分钟车流怎么走。这事儿叫 **spatiotemporal prediction**。

以前大家两条路走：

**第一条路：RNN 路线**。给模型一个"记忆本"，一帧一帧往后算。好处是记得久，坏处是慢，得一帧一帧来，没法并行。ConvLSTM、PredRNN 都是这个路子。

**第二条路：纯 conv 路线**。把所有帧堆一起，一把过，快是快但每个像素看的"视野范围"固定死了。SimVP、TAU 走这路。

PFGNet 想说的是：**纯 conv 也能又快又准，前提是让每个像素自己决定看多大范围**。

## 核心灵感：抄人眼的设计

你眼睛里有个东西叫 **center-surround receptive field**。简单说就是：视网膜对"中心看清楚 + 周围看模糊"这种对比特别敏感。

为啥这样？因为大脑懒。大片相同颜色（背景）不用细看，突然变的地方（边缘、运动）才重要。center-surround 这个结构天生就是**只看变化，跳过无聊区域**。

数学上这叫 **Difference of Gaussians (DoG)**，是个 **band-pass filter**——放行中间频率，干掉低频（大片相同背景）和高频（噪点）。

PFGNet 的 idea 就是：**把这个人眼机制搬到 CNN 里，而且让每个像素根据自己周围是啥内容，自动调节这个 filter**。

## 怎么做的

### 1. 先看每个像素"是什么货色"

每个像素，先拿三个固定探测器扫一下：
- **梯度**（Sobel）：这里边沿强不强
- **曲率**（Laplacian）：这里弯不弯
- **局部方差**：这里纹理多不多

三个结果拼一起，就是每个像素的"**频率身份证**"——告诉网络这地方是边缘还是平地。

### 2. 根据身份证决定用多大的"镜头"

有一组不同大小的 kernel：9×9、15×15、31×31。31×31 能看到很远的 context，9×9 只看近邻。

那个"频率身份证"过一个小 MLP，吐出三个数字，softmax 一下变成"选哪个 scale"的概率。

**平滑背景的像素** → 偏好大 kernel（看远一点找结构）
**运动边缘的像素** → 偏好小 kernel（盯细节）

这是 pixel-wise 的 conditional computation，每个像素自己挑工具。

### 3. 关键 trick：大kernel减小kernel = band-pass

光选 kernel 大小不够。PFGNet 最有意思的操作是：

**每个 scale 的 response = 大kernel卷积 − tanh(β) × 小kernel卷积**

直觉：
- 大kernel像低通，保留低频（慢变化）
- 小kernel像相对高通，保留高频（快变化）
- 大减小 = **中间频段保留，两边都干掉**

这就是 DoG 的精神。$\beta$ 是可学习的，每个 channel 一个，能正能负——意味着这个"减法"可以是"增强"也可以是"抑制"，网络自己学。

为啥用 tanh 不用 sigmoid？因为 sigmoid 只能压（0到1），tanh 能正能负，feature 有正有负，要双向调。这个 ablation 证明 tanh 确实好。

### 4. 拼起来

最后每个 pixel 的输出是三个 scale 的 (大−β·小) response 加权平均，权重就是那个 softmax 出来的 $\alpha$。

**效果**：每个像素得到一个**根据内容自动调好的 band-pass filter**，重点放在 motion-relevant 频段，把无关背景和噪点都压掉。

### 5. 省钱的工程 trick

31×31 的 depthwise conv 听起来很贵。但可以拆：

$$k \times k \rightarrow 1 \times k + k \times 1$$

横一刀，竖一刀，cost 从 $k^2$ 降到 $2k$。对 $k=31$，**15 倍** 省。这就是为啥能用 31 大kernel还只有 1.9M params。

## 为啥这事儿 work

### 频域直觉

低频 = 大片相同 = 没信息 = 压掉
高频 = 噪点 = 没用 = 压掉
中频 = 边缘 + 运动 + 纹理 = 你要预测的东西 = 放大

人眼 early visual cortex 就是这么干的。PFGNet 让 CNN 也能学到这事儿，且 spatially adaptive。

### 理论支撑

论文给了两个 theorem：

**Theorem 1（存在性）**：如果大kernel频率响应 slow decay、小kernel fast decay，它们的差（带合适 $\beta$）一定存在一个环形通带——也就是说 band-pass 性质有保证。

**Theorem 2（最优性）**：存在某个 $\beta^*$ 让 SNR 最大化，且这个最优 $\beta^*$ 让 composite filter 严格优于大kernel alone。证明用 intermediate value theorem + 极值定理。

**Lemma 1**：更进一步，存在非零 $\hat{\beta}$ 使 SNR($\hat{\beta}$) > SNR(0)。说明加 center suppression 这事儿不浪费，数学上一定比"光用大kernel"好。

## 实验结果

### TaxiBJ（交通流预测）

PFGNet: **1.9M params, 0.6G FLOPs, MSE 0.2881**（最好）

对比：
- TAU: 9.6M params, 2.5G FLOPs, MSE 0.3108
- SimVP: 13.8M, 3.6G, MSE 0.3282
- VMRNN: 2.6M, 0.9G, MSE 0.2887（差不多但用 RNN）

**用 1/5 的参数、1/4 的 FLOPs 打赢 TAU**。

### KTH（人体动作预测）

SSIM 是最好的（结构相似度），但 PSNR 略低于 SwinLSTM。

论文解释：KTH 背景是大片高对比度静态区域。PSNR 对绝对亮度误差敏感，但 PFGNet 是 band-pass，**故意不在乎 DC**，所以背景亮度可能漂移，但**运动边界更清楚**。

这是 band-pass filter 的 inherent trade-off：要 sharp edge 就得牺牲绝对亮度精度。

### Human3.6M（256×256 人体动作）

7.3M params, 58.3G FLOPs，SSIM 第二，MAE 第二。

TAU 用 5x params、3x FLOPs 才 marginal 更好。Recurrent 模型如 MIM 用 1051G FLOPs（18 倍）才打平 SSIM。

## Ablation 里的关键发现

1. **去掉 MSInit**：性能大幅退化。给后续 gating 提供分化好的多尺度 feature 是必要的预热。

2. **softmax vs mean fusion**：softmax (pixel-wise 自适应) 比 mean (固定权重) 好。每个像素挑 scale 这事儿 work。

3. **单一大 kernel**：k=31 单用最低 MSE 且最高 SSIM，但 multi-scale 整体最优。**大kernel强但不够**，要 scale 互补。

4. **$\beta$ 可学习 vs 固定**：固定 $\beta = 0$、$\pm 1$ 都比可学习差很多。**spatial adaptivity 是灵魂**。

5. **tanh vs sigmoid**：tanh 略好。双向调节比单向压制强。

6. **三个频率 cue 缺一不可**：gradient、Laplacian、variance 互补，去掉任何一个都退化。

## 可视化的"实锤"

Figure 14 在 Human3.6M 上解剖了一个学好的 PFG block：

- **空间结构**：学到的 effective kernel 自动出现 center-surround sign contrast 的环状结构，**emergent 不是手写**
- **频谱分析**：log power ratio 显示低频被压、中高频被抬、最高频趋平（不放大 noise）。**band-pass 行为被实证**
- **kernel 选择 map**：动态区域选大kernel，静态背景选小kernel。**conditional computation 在干活**

## 跟其他工作的关系

- **PeLK**：也搞 peripheral convolution，但是 uniform kernel，没 pixel-wise adaptivity
- **Per-ViT**：ViT 里加 center-surround bias，但 PFGNet 在 conv 里更适合 dense prediction
- **Octave Conv / AFNO / DC-Former**：在频域显式做，要 FFT/DCT 变换开销。PFGNet 完全 spatial domain 隐式做
- **ConvNeXt**：PFGNet 借了 GLU + LayerScale + GRN 这套稳定训练的配方，但把 depthwise conv 推到 31×31 separable + center suppression
- **RepLKNet / UniRepLKNet / SLaK**：大kernel路线先驱，但都是 uniform kernel。PFGNet 是 **adaptive large kernel**

## 我的几点吐槽和思考

### 担忧

**1. SimVP-style 假设**：把时间打包进 channel，假设帧间无强因果依赖。对 long horizon（KTH 10→40）可能吃亏，PSNR 没拿 SOTA 是信号。

**2. 固定 frequency cue**：Sobel、Laplacian 是 hand-crafted。能不能 end-to-end 学这些探测器？虽然省参数但限制表达。

**3. DC suppression 副作用**：band-pass 不在乎绝对亮度，对 photometric fidelity 要求高的场景（比如医疗影像）可能不合适。

**4. scale 集合 $\{9, 15, 31\}$ 没解释**：为啥这三个？不同任务最优 scale 集合可能不同，没 ablation。

**5. softmax 不 sparse**：三个 scale 都算，只加权融合。如果 top-k sparse gating 能省更多 compute，但可能 hurt gradient flow。

**6. separable 近似误差**：$k \times k \approx 1 \times k + k \times 1$ 不能精确 capture 对角方向。UniRepLKNet 用 dilation 缓解，PFGNet 没处理。

### 觉得有意思的

**1. 生物 + 信号处理 + CNN 三合一**：DoG → band-pass → learnable center suppression 这条链路非常 elegant。每一步都有理论支撑（theorem 1/2）和实证（visualization）。

**2. Conditional compute 的 lightweight 版**：不需要 MoE 的 routing 复杂度，用 frequency cues 当 router，pixel-wise 自适应。这个 idea 可以推广到其他 dense prediction task。

**3. 工程实现真的省**：1.9M params on TaxiBJ，5x smaller than TAU 还更准。这是 benchmark 上的真实 gain，不是 cherry pick。

**4. Ablation 做得扎实**：每个 design choice 都验证了，连 $\beta$ 的分布都可视化了，发现 Moving MNIST 上用 $\pm 1$ 对称分布，TaxiBJ 上接近 0——和 SNR 理论预测一致。

### 可探索方向

- 让 frequency cue 可学（不要固定 Sobel）
- cross-temporal frequency cue（看 motion 频率不只看空间频率）
- orientation selectivity（加方向维，模仿 V1 Gabor）
- sparse gating（top-k alpha 省更多 FLOPs）
- 3D band-pass（spatiotemporal volume 上做，处理时间频率）
- 跟 Mamba SSM 结合（selective state space 思路类似）
- physics-aware band-pass（weather nowcasting 中物理定律有频域 signature，可注入 inductive bias）

## 一句话总结

**用大kernel减去小kernel，得到一个 band-pass filter；用频率探测器告诉每个像素该用多大的 band-pass；整个网络是堆叠的"自适应中频提取器"，专攻运动和结构信息，背景和噪点都不要——这就是 PFGNet**。

人眼这么干了几亿年，CNN 现在也学会这么干了，而且还让每个像素自己挑工具，比人眼还灵活。

---

# PFGNet 深度解读：从生物视觉到 Band-Pass Filter 的 ConvNet

## 1. Big Picture：这篇 paper 在说什么

STPL (Spatiotemporal Predictive Learning) 这个 task 本质上是给过去帧预测未来帧。传统路线分成两派：

- **Recurrent-based** (ConvLSTM, PredRNN, SwinLSTM, VMRNN)：temporal memory 强，但 autoregressive inference 慢，parallelism 差
- **Recurrent-free** (SimVP, TAU, STLight)：纯 conv 或 attention，parallelism 好，但 receptive field 固定，无法 spatially adapt 到不同 motion pattern

PFGNet 想同时解决两件事：**用纯 conv 达到 SOTA**，同时 **让 receptive field 像生物视觉系统那样 pixel-wise 自适应**。

核心 insight 是把 **center-surround antagonistic receptive field** (Hubel & Wiesel 1962, Kuffler 1953) 这种生物机制，用 **大kernel减去小kernel** 这个简单操作数学化实现，并且用 **local frequency cues** 来 gating 选择哪个 scale 起作用。

[Hubel & Wiesel 1962 receptive fields](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/)
[Marr & Hildreth 1980 edge detection](https://royalsocietypublishing.org/doi/10.1098/rspb.1980.0020)

---

## 2. 生物学直觉：Center-Surround 与 DoG

视网膜 ganglion cells 的 receptive field 是经典的 **ON-center OFF-surround** 或反过来。这种拮抗结构数学上用 **Difference of Gaussians (DoG)** 建模：

$$\text{DoG}(r) = A_1 e^{-r^2 / 2\sigma_1^2} - A_2 e^{-r^2 / 2\sigma_2^2}$$

其中 $\sigma_1 < \sigma_2$，第一个 Gaussian 是窄的中心响应，第二个是宽的 surround 响应。

频域上，DoG 是一个 **band-pass filter**：放大 mid-frequency (edges, textures)，抑制 DC (uniform background) 和 high-frequency noise。

PFGNet 的核心 idea：把这个机制搬到 CNN 里。大kernel卷积 $H_L$ (低通，宽 receptive field) 减去小kernel卷积 $H_S$ (相对高通) 乘以系数 $\beta$，就形成一个 learnable ring-shaped band-pass filter。

[DoG in visual system - Turner et al 2018 eLife](https://elifesciences.org/articles/38841)

---

## 3. 数学核心：为什么 $H_L - \beta H_S$ 是 Band-Pass

### 3.1 空间域 vs 频域直觉

考虑一个 large kernel $H_L$，它的 frequency response 在 DC (零频) 处接近 1，然后 **slow decay**——这意味着它保留低频、抑制高频，本质是 **low-pass**。

考虑一个 small kernel $H_S$，它的 frequency response 在 DC 处也接近 1，但 **fast decay**——保留更多高频成分。

**关键 intuition**：两个 filter 都让 DC 通过，但 $H_L$ 衰减慢，$H_S$ 衰减快。在 mid-frequency 区间，$H_L > H_S$。它们的差 $H_L - \beta H_S$：

- **在 DC 附近**：$H_L \approx H_S \approx 1$，差值接近 $1 - \beta$（被抑制）
- **在 mid-frequency**：$H_L$ 还大，$H_S$ 已经衰减，差值为正（放大）
- **在 high-frequency**：两个都衰减到接近 0，差值接近 0（抑制）

这就是 **ring-shaped band-pass filter** 的本质。

### 3.2 形式化 (Theorem 1 - Weak Existence)

论文 Theorem 1 给出弱存在性定理。设 $H_1, H_2: [0, \pi] \to \mathbb{R}$ 是连续 radial frequency response。定义：

$$f(r) = H_1(r) - \beta H_2(r), \quad \beta \in (-1, 1)$$

变量说明：
- $H_1$：大kernel的radial frequency response (radial 表示沿径向频率 $r = \|\omega\|$ 取值)
- $H_2$：小kernel的radial frequency response
- $\beta$：可学习 suppression coefficient，用 tanh bound 在 (-1, 1)
- $r$：radial frequency，$r \in [0, \pi]$ 是 normalized Nyquist range

如果存在 $0 \leq c < a < b \leq \pi$ 使得：
- $f(c) \leq 0$ (low frequency 处 small kernel dominates, 差值非正)
- $f(a) > 0$ (mid frequency 处 large kernel retains more energy, 差值为正)
- $f(b) \leq 0$ (high frequency 处两个都衰减, 差值非正)

那么存在 $r_1, r_2$ 使得 $r_1 < a < r_2$，且 $f(r) > 0$ for $r \in (r_1, r_2)$，形成 ring-shaped pass band $\{\omega : r_1 < \|\omega\| < r_2\}$。

**证明思路**：连续函数 + intermediate value theorem。集合 $S_1 = \{r \in [c,a] : f(r) \leq 0\}$ 非空闭集，取 $r_1 = \max S_1$，则 $f(r_1) = 0$ 且 $f(r) > 0$ for $r \in (r_1, a]$。同理对 $S_2 = \{r \in [a,b] : f(r) \leq 0\}$ 取 min 得到 $r_2$。

### 3.3 SNR 最优 $\beta^*$ (Theorem 2)

设 input signal spectral power $P_S(\omega) \geq 0$，additive white noise $P_N(\omega) = \sigma_N^2$。Composite filter $H_\beta = H_L - \beta H_S$。SNR 定义：

$$\text{SNR}(\beta) = \frac{\int |H_\beta(\omega)|^2 P_S(\omega) d\omega}{\int |H_\beta(\omega)|^2 \sigma_N^2 d\omega}$$

变量说明：
- $H_\beta(\omega)$：composite filter 在频率 $\omega$ 的复响应
- $P_S(\omega)$：signal 在频率 $\omega$ 的 power spectral density
- $\sigma_N^2$：white noise constant power
- $|H_\beta|^2$：filter power response
- 分子：filter 输出中的 signal energy
- 分母：filter 输出中的 noise energy

定义 signal energy $N(\beta)$ 和 noise energy $D(\beta)$：

$$N(\beta) = \int |H_L - \beta H_S|^2 P_S d\omega = A - 2\beta B + \beta^2 C$$

$$D(\beta) = \sigma_N^2 \int |H_L - \beta H_S|^2 d\omega = \sigma_N^2 (\tilde{A} - 2\beta \tilde{B} + \beta^2 \tilde{C})$$

其中：
- $A = \int |H_L|^2 P_S d\omega$ (大kernel alone 的 signal energy)
- $B = \int \text{Re}(H_L \overline{H_S}) P_S d\omega$ (cross term，$\overline{H_S}$ 是 $H_S$ 共轭)
- $C = \int |H_S|^2 P_S d\omega$ (小kernel alone 的 signal energy)
- $\tilde{A}, \tilde{B}, \tilde{C}$ 对应 noise 版本 (即 $P_S$ 替换为 1)

**极限分析**：当 $\beta \to \pm\infty$：

$$\lim_{\beta \to \pm\infty} \text{SNR}(\beta) = \frac{C}{\sigma_N^2 \tilde{C}} =: L$$

是一个有限正常数。SNR($\beta$) 连续可微，在 $\pm\infty$ 处都趋于 $L$。如果存在 $\beta_0$ 使 SNR($\beta_0$) > L，则 SNR 在有限点达到 global maximum $\beta^*$。Fermat 定理给 $\frac{d}{d\beta}\text{SNR}(\beta^*) = 0$。

**Lemma 1**：进一步证明存在 $\hat{\beta} \neq 0$ 使 SNR($\hat{\beta}$) > SNR(0)，即 composite filter 严格优于大kernel alone。证明看 numerator 差值：

$$\Delta(\beta) = N(\beta)\tilde{A} - A \frac{D(\beta)}{\sigma_N^2} = -2\beta(B\tilde{A} - A\tilde{B}) + \beta^2(C\tilde{A} - A\tilde{C})$$

当 $B\tilde{A} \neq A\tilde{B}$（即 $H_L$ 不是 SNR stationary point），线性项 dominant，存在小 $\hat{\beta}$ 使 $\Delta(\hat{\beta}) > 0$，从而 SNR($\hat{\beta}$) > SNR(0)。

**Intuition 总结**：center suppression 不是浪费，而是把大kernel的 response 在 mid-band 抬起来，同时压住 DC 的 redundant background，达成比单纯大kernel更高的 SNR。

[Difference of Gaussians on Wikipedia](https://en.wikipedia.org/wiki/Difference_of_Gaussians)

---

## 4. 架构详解：从 Input 到 Output

### 4.1 整体 Pipeline (SimVP-style)

输入序列 $\{\mathbf{I}_t \in \mathbb{R}^{C_{\text{in}} \times H \times W}\}_{t=1}^{T_{\text{in}}}$：

1. **Shared spatial encoder**：每帧独立过 encoder，得 $\mathbf{F}_t = \text{Enc}(\mathbf{I}_t) \in \mathbb{R}^{C \times H' \times W'}$
2. **Temporal packing**：沿 channel 维 stack，得 $\mathbf{Z} \in \mathbb{R}^{C' \times H' \times W'}$，其中 $C' = T_{\text{in}} \cdot C$。这一步是 SimVP 的核心 trick——把 temporal 信息塞进 channel，让后续 conv 同时处理时空。
3. **MSInit** (Multi-Scale Initialization)：先用小kernel生成多尺度 initial features
4. **$N_t$ 个 PFG block 堆叠**：核心模块，做 frequency-guided gating
5. **Temporal unpacking**：把 channel 拆回 temporal，得 $\{\mathbf{F}'_t\}_{t=1}^{T_{\text{out}}}$
6. **Decoder**：对称上采样恢复分辨率，输出 $\{\mathbf{O}_t\}_{t=1}^{T_{\text{out}}}$

$N_s$ 是 encoder/decoder 的下采样 block 数，$N_t$ 是中间 PFG block 数。

### 4.2 MSInit (Multi-Scale Initialization)

公式 (2)：

$$T_m(\mathbf{Z}) = \mathbf{v}_m * (\mathbf{h}_m * \mathbf{Z}) + d_m(\mathbf{Z}) + \mathbf{Z}$$

变量说明：
- $m \in \{1, \ldots, M\}$：scale index，论文用 $k_m \in \{3, 5, 7\}$，对应 $M=3$
- $\mathbf{h}_m \in \mathbb{R}^{1 \times k_m}$：horizontal 1D conv kernel
- $\mathbf{v}_m \in \mathbb{R}^{k_m \times 1}$：vertical 1D conv kernel
- $*$：卷积
- $d_m(\cdot)$：$3 \times 3$ depthwise conv branch，增强 mid-frequency sensitivity
- $\mathbf{Z}$：identity skip，保 gradient flow

为什么用 separable 1D？因为 $k_m \times k_m$ 的 2D conv 成本是 $O(k_m^2)$，分解后 $1 \times k_m$ + $k_m \times 1$ 成本 $O(2k_m)$，对 $k=7$ 减少 ~2.4x。

公式 (3) 把每个 scale 的输出过 $1 \times 1$ conv 投影到 $C'/M$ 通道，然后 concat 回 $C'$：

$$\mathbf{X} = \text{Concat}(\{\text{Conv}_{1\times 1}(T_m(\mathbf{Z}))\}_{m=1}^M) \in \mathbb{R}^{C' \times H' \times W'}$$

**Intuition**：MSInit 给后续 PFG block 提供已经分化好的 multi-scale features。如果没有这个 initialization，PFG 的 gating 就缺乏可选择的素材，要么只在小kernel feature 上工作（缺 global context），要么触发全 scale 计算（贵）。

### 4.3 PFG Block 核心详解

这是 paper 的灵魂。分四步：

#### Step 1: Frequency Descriptor Extraction (公式 4)

从输入 $\mathbf{X} \in \mathbb{R}^{C' \times H' \times W'}$ 提取 3 个 spectral cues：

$$\mathbf{f}_1 = \sqrt{(\mathbf{G}_x * \mathbf{X})^2 + (\mathbf{G}_y * \mathbf{X})^2}$$

$$\mathbf{f}_2 = |\mathbf{L} * \mathbf{X}|$$

$$\mathbf{f}_3 = \mathbb{E}_{3\times 3}[\mathbf{X}^2] - \mathbb{E}_{3\times 3}[\mathbf{X}]^2$$

变量说明：
- $\mathbf{G}_x, \mathbf{G}_y = \mathbf{G}_x^\top$：Sobel edge detector，$3 \times 3$ fixed depthwise kernel
- $\mathbf{L}$：Laplacian curvature detector，如 $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$
- $\mathbb{E}_{3\times 3}[\cdot]$：$3 \times 3$ average pooling (padding=1, stride=1)，估计 local mean
- $\mathbf{f}_1$：gradient magnitude (一阶导，edge strength)
- $\mathbf{f}_2$：Laplacian magnitude (二阶导，curvature)
- $\mathbf{f}_3$：local variance (texture energy)

每个 cue 先 depthwise conv 然后沿 channel average，得到 single-channel map。三个 concat 成 $\mathbf{F} \in \mathbb{R}^{3 \times H' \times W'}$。

**Intuition**：这三个 cue 是经典 image processing 的频率分析工具。Gradient 是 first-order edge detector (mid-frequency)，Laplacian 是 second-order (更 sensitive to high-frequency)，local variance 是 texture energy (mid-frequency 集中区域)。组合起来 cover 不同频段，提供 pixel-wise "这里有什么 frequency 内容" 的 signature。

#### Step 2: Gating Logits (公式 5, 6)

$$\mathbf{Z}_g = \mathbf{W}_g * \mathbf{F} + \mathbf{b}_g \in \mathbb{R}^{K \times H' \times W'}$$

$$\alpha_k(h,w) = \frac{\exp(Z_{g,k}(h,w))}{\sum_{j=1}^K \exp(Z_{g,j}(h,w))}, \quad k \in \mathcal{K}$$

变量说明：
- $\mathbf{W}_g$：$1 \times 1$ conv learnable weight，shape $\mathbb{R}^{K \times 3}$（输入3 channel，输出 K channel）
- $\mathbf{b}_g$：bias，shape $\mathbb{R}^K$
- $K = |\mathcal{K}|$：scale 数量，论文 $\mathcal{K} = \{9, 15, 31\}$，$K=3$
- $\alpha_k(h,w)$：在 pixel $(h,w)$ 选择 scale $k$ 的 softmax 权重

每个 pixel 三个 scale 的权重加起来 = 1，形成 pixel-wise adaptive 选择。

**Intuition**：这是 **conditional computation** 的 soft 版本。Texture-rich pixel 可能偏好小 kernel（fine detail），smooth pixel 偏好大kernel（global context）。Softmax 让网络可以混用，不强制 hard 选择。

#### Step 3: Peripheral Response with Center Suppression (公式 7, 8)

对每个 scale $k \in \mathcal{K} = \{9, 15, 31\}$：

$$\mathbf{P}_k = \mathbf{v}_k * (\mathbf{h}_k * \mathbf{X})$$

$$\mathbf{Y}_k = \mathbf{P}_k - \tanh(\beta_k) \odot (\mathbf{C} * \mathbf{X})$$

变量说明：
- $\mathbf{h}_k \in \mathbb{R}^{1 \times k}$：horizontal depthwise conv kernel，scale $k$
- $\mathbf{v}_k \in \mathbb{R}^{k \times 1}$：vertical depthwise conv kernel，scale $k$
- $\mathbf{P}_k$：large kernel 的 peripheral response（被 separable decompose 近似）
- $\mathbf{C}$：$3 \times 3$ depthwise center kernel
- $\beta_k \in \mathbb{R}^{C'}$：可学习 channel-wise suppression coefficient，broadcast 到 spatial
- $\tanh(\beta_k) \in (-1, 1)^{C'}$：bounded，允许 bidirectional (enhance or suppress)
- $\odot$：element-wise multiplication
- $\mathbf{Y}_k$：center-surround antagonistic response

**为什么用 tanh 而不是 sigmoid**：sigmoid 输出 $(0, 1)$ 只能 suppress。tanh 输出 $(-1, 1)$ 允许 **enhance center** (negative $\beta$) 或 **suppress center** (positive $\beta$)。Feature map 有正有负，需要双向调节。Ablation Table 7 证实 tanh 优于 sigmoid。

**为什么 $\beta$ 是 channel-wise 不是 pixel-wise**：channel-wise $\beta$ 共享 spatial，参数量小 ($C'$ 个 scalar)，依赖 softmax gating 在 spatial 维度做 adaptivity。如果 $\beta$ 也 pixel-wise 会 over-parameterized。

#### Step 4: Gated Fusion (公式 9)

$$\text{PFG}(\mathbf{X}) = \sum_{k \in \mathcal{K}} \alpha_k \odot \mathbf{Y}_k$$

每个 pixel 把三个 scale 的 antagonistic response 用 $\alpha_k$ 加权求和。

**最终效果**：每个 pixel 得到一个 **spatially-adaptive, frequency-selective band-pass filter**，pass-band 位置由 local frequency content 决定，filter 形状由 learnable $\beta$ 决定。

### 4.4 GLU-style Channel Mixing (公式 11)

PFG 之后接 GLU [Dauphin 2017] 做 channel mixing：

$$\mathbf{Z}_c = \text{PW}_{E \to C'}(\sigma(\mathbf{U}) \odot \text{DW}_{3\times 3}(\mathbf{V}))$$

变量说明：
- $\text{PW}_{E \to C'}$：point-wise $1 \times 1$ conv 从 $E$ channel 到 $C'$ channel
- $E = 4C'$：expansion ratio $r = 4$，先把 $C'$ expand 到 $2E$（因为要 split 成 U 和 V）
- $\mathbf{U}, \mathbf{V}$：从 $2E$ channel 均匀 split 出的两个 $E$-channel tensor
- $\sigma$：sigmoid
- $\text{DW}_{3\times 3}$：$3 \times 3$ depthwise conv
- $\odot$：element-wise

GLU 让 channel 维度也有 gating 机制，配合 spatial gating 形成双路 adaptivity。

之后接 GRN normalization [ConvNeXt V2, Woo 2023] 和 LayerScale [CaiT, Touvron 2021] 稳定训练。

[GLU original paper](https://arxiv.org/abs/1612.08083)
[ConvNeXt V2 GRN](https://arxiv.org/abs/2306.00937)

### 4.5 Computational Efficiency 分析

每个 $k \times k$ 2D conv 分解成 $1 \times k$ horizontal + $k \times 1$ vertical：

- 原始 depthwise 2D conv per channel：$k^2$ MACs per pixel
- Separable：$k + k = 2k$ MACs per pixel
- Reduction factor: $k^2 / 2k = k/2$

对 $k = 31$：$31^2 / 62 = 15.5\times$ 减少。这就是为什么 PFGNet 能用 $k=31$ 的大kernel还保持 1.9M params / 0.6G FLOPs on TaxiBJ。

类似 trick 在 RepLKNet [Ding 2022], UniRepLKNet [Ding 2024], SLaK [Liu 2023] 都有，但 PFGNet 的不同在于 **配合 center suppression 让大kernel变成 band-pass**，而不仅是为了大 receptive field。

[RepLKNet](https://arxiv.org/abs/2203.06717)
[UniRepLKNet](https://arxiv.org/abs/2311.15599)
[SLaK](https://arxiv.org/abs/2207.13592)

---

## 5. 实验结果深度解读

### 5.1 TaxiBJ (Table 3)

PFGNet: 1.9M params, 0.6G FLOPs, MSE 0.2881 (SOTA), MAE 14.75, SSIM 0.9857

对比：
- VMRNN (recurrent): 2.6M, 0.9G, MSE 0.2887 (相近)
- SwinLSTM: 2.9M, 1.3G, MSE 0.3026
- TAU: 9.6M, 2.5G, MSE 0.3108
- SimVP: 13.8M, 3.6G, MSE 0.3282

**Insight**：PFGNet 用比 TAU 少 5x 的 params、4x 的 FLOPs 达到更好 MSE。证明 frequency-guided gating 比 generic attention 更 sample-efficient。

### 5.2 KTH (Table 4)

10→20 frames: SSIM 0.911 (SOTA), PSNR 34.10
10→40 frames: SSIM 0.891 (SOTA), PSNR 32.64

**Insight**：PFGNet 在 SSIM 上是 SOTA 但 PSNR 略低于 SwinLSTM。论文解释 KTH 高对比度静态背景 dominate pixel-wise error，PSNR 惩罚 minor intensity shift in large uniform regions。PFGNet **故意 de-emphasize photometric fidelity** 防止 limb collapse 和 boundary diffusion——这是 band-pass filter 的副作用，但换来更好的 perceptual quality。

这个 trade-off 很有意思：**band-pass filter 本质上 de-emphasize DC**，所以均匀背景区域的绝对亮度可能漂移，但 motion boundary 更锐利。

### 5.3 Human3.6M (Table 5)

7.3M params, 58.3G FLOPs, MAE 1392.4 (2nd), SSIM 0.9838 (2nd)

TAU 5.2x params、3.1x FLOPs 仅 marginally 更好。Recurrent 模型如 MIM 用 1051G FLOPs（18x）才打平 SSIM。

### 5.4 Ablation 关键发现

#### Table 6: 宏观结构

- 去掉 MSInit (model1): MSE 0.3119 → 显著退化
- Mean fusion 替代 softmax (model2): MSE 0.3033，softmax 0.2881
- 5×5 center vs 3×3 center: 3×3 略好，但应该 align 主kernel scale
- **单 scale 实验 (Figure 8a)**：k=31 单独最低 MSE 且最高 SSIM，但 multi-scale 整体最好

**Insight**：单一大kernel很强，但缺 fine texture 处理。Multi-scale 让 network 在 boundary 和 interior 选不同 scale。

#### Table 7: 细节机制

- $\beta = 0$ (no suppression, model3): MSE 0.2993 vs PFGNet 0.2881，证明 center suppression 重要
- Fixed $\beta = -1$ 或 $\beta = 1$：MSE 0.3209 / 0.3286，远差于 learnable $\beta$。**spatial adaptivity 是关键**
- tanh vs sigmoid gate：tanh 略好 (MSE 27.61 vs 28.04 on Moving MNIST 100 epoch)
- 三个 frequency cue 缺任何一个都退化，三者组合最优

#### Table 9, 10: $N_t$ ablation

TaxiBJ: $N_t = 8$ 最优，$N_t = 10$ 略退化（mild overfitting）。KTH 10→20: $N_t \geq 6$ 边际收益递减。

#### Figure 10: tanh($\beta$) 分布

- Moving MNIST: tanh($\beta$) 对称分布于 0，明显 mass 在 $\pm 1$——简单 digit motion 既用 enhance 也用 suppress
- TaxiBJ: tanh($\beta$) 均值近 0，大kernel branch variance 更小——balanced mild modulation 保留 traffic flow pattern

这与 Theorem 2 的 SNR 分析一致：learnable $\beta$ 在 sign 和 magnitude 上 adapt 到 local spectral statistics。

### 5.5 Figure 14: Mechanistic Visualization (Human3.6M)

最 revealing 的可视化：

**Left - Spatial antagonistic structure**：median effective kernel (K=31 branch) 展现 center-surround sign contrast，**emergent** 而非 hand-coded。形状类比 retinal ganglion cell DoG。

**Middle - Adaptive spectral re-weighting**：plot radial log-power ratio $\Delta \log P(r) = \log P_{\text{full}} - \log P_{\beta=0}$：
- Low frequency 处能量减弱
- Mid-to-high 频段能量增强
- Highest frequency 趋平（noise 不过度放大）

这直接 visualize 了 band-pass filter 行为。

**Right - Spatially adaptive kernel selection**：argmax $\alpha$ map 显示 dynamic region 和 motion boundary 偏好大kernel，smooth static background 偏好小kernel。这是 conditional computation 的 visual proof。

---

## 6. 我的思考与 Critical Analysis

### 6.1 亮点

1. **Theory-meets-biology-meets-implementation**：DoG → band-pass filter → learnable center suppression 这条链路很 elegant。Theorem 1 和 Theorem 2 把直觉 formalize，给出存在性和最优性证明。

2. **Conditional computation without MoE complexity**：用 frequency cues 做 pixel-wise gating 是一种 lightweight conditional compute。比 MixNet, CondConv, MoE 简单很多。

3. **Efficiency genuinely strong**：1.9M params on TaxiBJ vs TAU 9.6M (5x smaller) while better MSE。这个不是 marketing number，是 OpenSTL benchmark 上的真实 gain。

4. **Ablation 做得相当 thorough**：从宏观到微观，每个 design choice 都被验证。

### 6.2 可能的问题与 Limitations

1. **TST (Temporal Shift Module) 风险**：SimVP-style 把 temporal 打包进 channel 假设 frames 之间无强 causal dependence。对 long-horizon (>40 frames) 可能退化。论文没在 KTH 10→40 给 PSNR SOTA。

2. **Frequency cues 的固定性**：Sobel、Laplacian、local variance 是 hand-crafted。能不能让 network 学到这些 detector？虽然 fixed 降参数，但 limit expression power。可能可以 end-to-end learn 这些 cues。

3. **Band-pass filter 的 DC suppression 副作用**：在 KTH 上 PSNR 略低（34.10 vs SwinLSTM 34.34），因为 band-pass de-emphasize DC，导致 background intensity 可能漂移。对 photometric fidelity 要求高的 application 可能不理想。

4. **Scale 集合 $\{9, 15, 31\}$ 的选择**：为什么这三个？没有 ablation 不同 $\mathcal{K}$ 组合。可能不同 task 最优 scale 集合不同。

5. **Gating 是 softmax 不是 sparse**：每个 pixel 三个 scale 都计算，只是加权融合。如果用 Gumbel-softmax 或 top-k sparse gating 可以省更多 compute。但 sparse 路径可能 hurt gradient flow。

6. **Separable conv 的近似误差**：$k \times k$ 用 $1 \times k + k \times 1$ 近似有 angle 衰减问题（不能精确 capture 对角方向）。UniRepLKNet 用 dilation、sparse kernel 等缓解，PFGNet 没讨论这个。

### 6.3 与其他工作的联系

#### vs PeLK [Chen 2024]
PeLK 用 peripheral convolution + parameter sharing 把 kernel 推到 100×100+。但 PeLK 是 uniform kernel，没有 pixel-wise adaptivity。PFGNet 借鉴了 "peripheral" 命名但加了 frequency gating。

[PeLK paper](https://arxiv.org/abs/2403.07589)

#### vs Per-ViT [Min 2022]
Per-ViT 在 ViT 中加 eccentricity-aware positional encoding 模拟 center-surround attention bias。PFGNet 在 conv 中实现类似 idea，更适合 STPL 的 dense prediction。

[Per-ViT](https://arxiv.org/abs/2205.12964)

#### vs Octave Conv [Chen 2019]
Octave Conv 把 feature 分 high/low frequency group，low frequency 在 lower resolution 计算。PFGNet 不显式分 group，而是用大kernel implicit 处理 frequency。

[Octave Conv](https://arxiv.org/abs/1904.05049)

#### vs AFNO [Guibas 2021]
AFNO 在 Fourier space 做 token mixing with learnable filtering。PFGNet 完全 spatial domain，avoid FFT overhead。但 AFNO 的 frequency selectivity 更 explicit。

[AFNO](https://arxiv.org/abs/2110.02958)

#### vs DC-Former [Li 2023]
DC-Transformer 直接在 DCT domain 工作，做 frequency component selection。需要 explicit DCT transform。

[DC-Former](https://arxiv.org/abs/2302.01044)

#### vs ConvNeXt [Liu 2022]
ConvNeXt 用 $7 \times 7$ depthwise conv + GLU + LayerScale + GRN。PFGNet 借鉴了 GLU/LayerScale/GRN 但把 depthwise conv 推到 31×31 separable + center suppression。

[ConvNeXt](https://arxiv.org/abs/2201.03545)

#### vs MogaNet [Li 2024]
MogaNet 用 multi-order gated aggregation。PFGNet 与之相比在 Human3.6M 上 7.3M vs 8.6M params，相近精度但更省。

[MogaNet](https://arxiv.org/abs/2211.03297)

#### vs STMFANet [Jin 2020]
STMFANet 也是 spatiotemporal multi-frequency analysis，但用 explicit FFT-based filtering。PFGNet 是 spatial-domain implicit。

[STMFANet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Exploring_Spatial-Temporal_Multi-Frequency_Analysis_for_High-Fidelity_and_Temporal-Consistency_Video_CVPR_2020_paper)

### 6.4 与生物视觉的更深联系

论文引了 Andy Clark "Surfing Uncertainty" 的 predictive processing 框架 [Clark 2015]。Predictive coding theory 认为大脑是 hierarchical prediction machine，每一层 predict 下一层 input，error signal 反向 propagate。

[Andy Clark - Surfing Uncertainty](https://global.oup.com/academic/product/surfing-uncertainty-9780190214783)

PFGNet 的 center-surround 在某种意义上 mimic **feed-forward error computation**：center 是 prediction (local context)，surround 是 actual (broader context)，差值是 residual error。但 PFGNet 不做 explicit prediction error，而是 band-pass filtering。

更深层的联系是 **V1 simple cells**：被建模为 Gabor filters，band-pass oriented。PFGNet 学到的是 isotropic band-pass（没有 orientation selectivity）。如果能加 orientation dimension 可能更接近 V1。

### 6.5 公式化的 Intuition: 一句话总结

把 conv 大kernel看成 low-pass，小kernel看成 relatively high-pass，**差值是 band-pass**。用 frequency cues 告诉每个 pixel 该用多大 receptive field 的 band-pass。整个网络是 stack of adaptive band-pass filters，本质上在做 **spatially-adaptive spectral filtering for spatiotemporal prediction**。

### 6.6 工程实现细节 (paper 没明说但可推断)

- Depthwise separable 1D conv：$\mathbf{h}_k$ 和 $\mathbf{v}_k$ 是 depthwise，所以参数量是 $C' \cdot k$，不是 $C'^2 \cdot k$。这是关键。
- $\beta_k$ 是 channel-wise scalar，每个 scale 每个 channel 一个，参数量 $K \cdot C'$ 个 scalar，几乎 0 cost。
- Gating 网络 $\mathbf{W}_g$ 是 $1 \times 1$ conv：参数 $K \cdot 3 = 9$ 个 weight + $K = 3$ bias，极轻。
- 整个 PFG block 主要参数在 large depthwise kernels 和 GLU 的 pointwise conv 上。

### 6.7 Future Direction 联想

1. **Dynamic $\mathcal{K}$**：不同 layer 用不同 scale 集合，early layer 小kernel fine detail，late layer 大kernel global context
2. **Learnable frequency cues**：让 network 自己学 detector 而不是固定 Sobel/Laplacian
3. **Cross-frame frequency gating**：现在 frequency cue 只看当前 frame 的 feature，可以 cross-temporal 看 motion frequency
4. **Combine with SSM**：VMRNN 已经把 Mamba 引入 recurrent。PFGNet 的 gating 和 Mamba 的 selective SSM 思路类似，可能 unify
5. **3D extension**：spatiotemporal volume 上做 3D band-pass filter，处理 video 中 time frequency
6. **Sparse gating**：top-k sparse alpha 让某些 scale 不计算，省更多 FLOPs
7. **Orientation selectivity**：加 Gabor-like oriented filter 学习 V1-style 多方向 band-pass
8. **Physics-aware band-pass**：在 weather nowcasting 中，physical law 在 frequency domain 有 specific signature，可以 inject as inductive bias

### 6.8 Implementation Note 推测

代码在 https://github.com/fhjdqaq/PFGNet。从 description 推测：

```python
class PFGBLock(nn.Module):
    def __init__(self, C, scales=[9, 15, 31]):
        # MSInit: 3 branches, each 1xk + kx1 + 3x3 dw + identity
        # frequency cues: fixed Sobel, Laplacian, avg pool
        # gating: 1x1 conv from 3 to K channels
        # for each scale: separable large kernel + center suppress with tanh(beta)
        # GLU channel mixing
        # GRN + LayerScale
```

如果想 run 一个 minimal version，关键是 fixed Sobel/Laplacian depthwise filter（用 `nn.Conv2d` with `groups=C` and `bias=False`，weight 用 Sobel/Laplacian init 并 freeze）。

---

## 7. 总结

PFGNet 是一个 **theoretically grounded, biologically inspired, engineering efficient** 的工作。它把 biological center-surround、signal processing band-pass filter、CNN large kernel、conditional computation 四个 idea 用一个 elegant formulation 串起来：

$$\text{PFG}(\mathbf{X}) = \sum_k \text{softmax}(\mathbf{W}_g * \text{FreqCues}(\mathbf{X}))_k \odot \left[\text{SepLargeConv}_k(\mathbf{X}) - \tanh(\beta_k) \cdot \text{SmallConv}(\mathbf{X})\right]$$

每个符号都有 biologically-plausible 解释，每个 design choice 都有 ablation 支撑，最终在 OpenSTL benchmark 上达到 SOTA 或 near-SOTA with 显著更少的 params/FLOPs。

这篇 paper 对 STPL 社区最大的贡献可能是 **证明 pure conv + 显式 frequency inductive bias 能 beat recurrent + attention**，给后续工作提供了一条 alternative design space：**spatially-adaptive spectral filtering**。

---

## References

- [PFGNet GitHub](https://github.com/fhjdqaq/PFGNet)
- [SimVP - simpler yet better video prediction](https://arxiv.org/abs/2211.12509)
- [TAU - Temporal Attention Unit](https://arxiv.org/abs/2306.12026)
- [OpenSTL Benchmark](https://arxiv.org/abs/2310.16056)
- [Hubel & Wiesel 1962](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/)
- [Marr & Hildreth 1980](https://royalsocietypublishing.org/doi/10.1098/rspb.1980.0020)
- [Andy Clark - Surfing Uncertainty](https://global.oup.com/academic/product/surfing-uncertainty-9780190214783)
- [DoG Wikipedia](https://en.wikipedia.org/wiki/Difference_of_Gaussians)
- [RepLKNet](https://arxiv.org/abs/2203.06717)
- [UniRepLKNet](https://arxiv.org/abs/2311.15599)
- [SLaK](https://arxiv.org/abs/2207.13592)
- [PeLK](https://arxiv.org/abs/2403.07589)
- [Per-ViT](https://arxiv.org/abs/2205.12964)
- [ConvNeXt V2 GRN](https://arxiv.org/abs/2306.00937)
- [GLU original](https://arxiv.org/abs/1612.08083)
- [Octave Conv](https://arxiv.org/abs/1904.05049)
- [AFNO](https://arxiv.org/abs/2110.02958)
- [MogaNet](https://arxiv.org/abs/2211.03297)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [VMRNN](https://arxiv.org/abs/2403.18495)
- [SwinLSTM](https://arxiv.org/abs/2307.16518)
- [PredRNN](https://arxiv.org/abs/1706.09185)
