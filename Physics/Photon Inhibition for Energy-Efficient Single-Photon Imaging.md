---
source_pdf: Photon Inhibition for Energy-Efficient Single-Photon Imaging.pdf
paper_sha256: d8ab331c24f73508f736404fafb351b0480c08d0447e71cfd467ff0b2a28cab6
processed_at: '2026-08-06T03:09:03-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在讲啥

Andrej，换个画风，像咱在 whiteboard 前面聊天那样讲。

---

## 故事的开头：SPAD 很猛，但有个尴尬

SPAD camera 能数单个光子，这就很离谱——传统 CMOS sensor 要攒好几百个光子才出一个像样的 pixel 值，SPAD 一个光子就响一下。

听起来很爽对吧？但有个 catch：**每次响一下都要烧几 pJ 的电**。因为 SPAD 是靠 avalanche（雪崩）放大一个光子的信号成可测的电流脉冲，这个雪崩过程本身吃电。

这就导致一个很反直觉的事：**光越亮，电费越贵**。

你想想普通相机，拍太阳和拍黑夜，功耗差不多，300 mW 左右。但 SPAD 拍太阳的时候，每个 pixel 每秒被几百万个光子砸，每个光子都触发一次雪崩——几瓦电就没了。paper 里估了一下，把现在的 megapixel SPAD extrapolate 到几十 megapixel，亮光下要好几瓦，远超普通 CMOS 的 300 mW。

**这就是为啥 SPAD 一直没法做高分辨率**。不是 pixel 做不小，是做小了也扛不住亮光的电费。

---

## 核心想法：不是所有光子都值得数

作者问了一个我觉得很优雅的问题：

> 已经亮到快饱和的 pixel，再数一个光子能给你多少 information？几乎为零。
> 一片均匀亮度的区域，中间那个 pixel 的光子，用周围 8 个 pixel 的值就能猜出来。
> edge 上的光子呢？值老钱了。

所以——**干嘛均匀地数所有光子？** 关掉那些"不值得数"的 pixel 不就行了。

关掉的意思是真的在电子层面把 SPAD 的 bias voltage 拉低，让它对光子无响应。雪崩根本不发生，电就从源头省了。

这跟 event camera 的"只在有变化的时候报"哲学很像，但 layer 更低——event camera 省的是 readout / data transfer，这里省的是 detection / avalanche 本身。

---

## 灵感来源：你的 retina 早就这么干了

人眼 retina 里有个机制叫 **lateral inhibition**——一个 photoreceptor 被光激活后，会让旁边的邻居变得 *less sensitive*。这是 1953 年 Barlow 发现的，后来 Franke 2017 在 Nature 上又确认了一遍。

效果是：retina 在送信号给大脑之前，已经在做 "edge enhancement" 和 "redundancy reduction"。一片均匀亮度的区域，retinal ganglion cell 几乎不报，因为邻居互相抑制掉了；只有 edge 上有信号。

**Photon inhibition 就是把这个机制搬到 SPAD 上。**

---

## 怎么决定关哪个 pixel？一个小卷积

policy 长这样：每个 pixel 维护一个 score，score 是用一个小 kernel（3×3 空间 × 4 帧）卷积过去几帧的 binary detection 结果算出来的。score 超过阈值 η，就关掉这个 pixel 接下来 τ_H 帧。

具体 kernel 举例，他们试了几个：

**Center + Ring**：中间 pixel 权重 8，周围 8 个邻居各 1。这就是个 center-surround 结构，模仿 retina。检测"这个 pixel 比邻居亮多少"。
```
1 1 1
1 8 1
1 1 1
```

**Laplacian**：中间 -8，周围 +1。检测 spatial variation，平的地方 score 接近 0，edge 上 score 大。
```
1 1 1
1 -8 1
1 1 1
```

**为什么所有系数都是 power of 2？** 因为最终要做在 in-pixel hardware 里。乘 8 就是左移 3 bit，乘 -8 就是左移 + 翻符号位。比真 multiplier 便宜几个数量级。这是给"future tape-out"留路的。

---

## 一个很优雅的小 trick：ternary 编码

pixel 的输出是 binary 的——0 或 1。但如果直接拿 0/1 去卷积，你分不清两种情况：
1. pixel 开着，但没光子来（结果是 0）
2. pixel 被关掉了（也是 0）

第一种是 *evidence of absence*——"我盯着看了，没东西"，这是 information！
第二种是 *absence of evidence*——"我压根没看"，没 information。

所以作者用了个小 trick：把 binary 结果编码成 ternary：
- 检测到光子 → +1
- pixel 开着但没光子 → -1
- pixel 被关了 → 0

具体就是 `(2F - 1) × M`，F 是 detection，M 是 enable mask。

这样卷积的时候，"没光子但 pixel 开着"会贡献负分，让 score 变低，倾向于继续开（因为这里 flux 低，值得继续盯）。而"pixel 关了"贡献 0，不影响 score。

**这个细节我挺喜欢的**，因为它把 information-theoretic 的直觉塞进了一个 tiny arithmetic trick。

---

## 两个效率指标：到底省了什么

paper 提了两个 metrics，我觉得这个 framing 是它最 conceptual 的贡献。

**SNR_H / D²**（detection efficiency）：你花一个 detection（一个 avalanche）能买多少 signal-to-noise。
- 暗的地方：每个光子都 informative，efficiency ≈ 1
- 亮的地方：pixel 快饱和了，再来光子没用，efficiency → 0

**SNR_H / W²**（measurement efficiency）：你花一个 measurement window（一次 readout / 一次计数器 tick）能买多少 SNR。
- 在 H ≈ 1.6（Y ≈ 0.8）时 peak——这是 quanta image sensor 的经典 sweet spot

第一个 metric 对应 avalanche energy，第二个对应 readout + compute + latency。**任何 inhibition policy 都应该在这两条曲线上找平衡**。

bright pixel 在第一条曲线上是浪费的（detection efficiency 低），dark pixel 在第二条曲线上是浪费的（measurement efficiency 低）。所以 policy 要 *把测量 budget 倾斜给中间亮度的 pixel，bright 的直接关，dark 的少测*。

---

## Oracle 告诉你理论上限在哪

如果 image 已知（oracle），怎么分配 measurement 数 W_i 来 minimize image MSE？

推出来是：

$$W_i \propto \sqrt{1 - Y_i}$$

Y 是 binary rate，Y→1（饱和）时 allocation → 0，Y→0 时 allocation 最大。

**直觉**：把测量预算倾斜给 dim pixel。bright pixel 不需要更多测量了，再测也是饱和。

实际 policy 没法做到 oracle（不知道 ground truth），但实验显示 handcrafted policy 已经能逼近 oracle 的 ~13% detection reduction。

---

## Edge Detection 的专门 policy

这是我觉得最 "task-aware" 的部分。作者专门为 edge detection 设计了一个 Boolean 组合：

- 算两个 score：S1 = Laplacian（spatial variation），S2 = average（local brightness）
- 抑制条件：(spatial flat ∧ 不太暗) ∨ (很亮)
- 翻译：平的地方 + 中等亮度 → 关（没 edge，没信息）；很亮的地方 → 关（饱和了）

这样 edge 和 dark region 都保留，中间的平坦亮区砍掉。结果：edge detection F-score 持平的情况下，detections 减少 30%。

**这就是 "which photons for *this task*" 的具体体现**——你不在乎 reconstruction quality，只在乎 edge，所以 policy 长得不一样。

---

## Video 怎么办？Saturation Look-ahead

静态场景可以长时间 hold-off，但 video 里 hold-off 太长会 motion blur。

作者用 exposure bracketing：一串不同 exposure time 的帧，T = {1, 1, 2, 3, 5, 8, 13, 21}（Fibonacci）。短 exposure 先测，如果短 exposure 已经检测到 ≥2 个光子，那长 exposure 几乎肯定饱和，直接关掉。

这就像个 early-warning：用 cheap 的短曝光预判，决定要不要做 expensive 的长曝光。

---

## 真硬件实验：SwissSPAD2

他们拿 SwissSPAD2（512×256，97,700 FPS）拍了 58 万帧，光照从 <1 lux 到 >4000 lux 跨 7 stops。

对比两种 inhibition：
1. **Sub-sampling 10×**：固定丢 9/10 帧（flux-agnostic）
2. **Saturation look-ahead**：flux-adaptive

结果：
- **亮光**：两者都 OK，都能砍 90%+ 光子
- **暗光**：sub-sampling 翻车——家具、人形都没了；look-ahead 还保留 detail
- **累积 detections**：look-ahead 比 sub-sampling 更少

**为什么 look-ahead 赢？** 因为它是 adaptive 的——亮的地方砍得多，暗的地方砍得少。sub-sampling 一视同仁，暗的地方本来光子就少，再丢 90% 就废了。

然后他们把 burst reconstruction 的结果喂给 **YOLOv8**，**95% 光子被抑制**的情况下 object detection 仍然成功。

这个数字我觉得是 paper 最 punchy 的 result。说明 high-level vision task 对 photon-level noise 的 robustness 远超 SSIM 这种 pixel-level metric——你不需要一张好看的图，你需要一张 *task-sufficient* 的图。

---

## 硬件可行性：能做在 pixel 里吗？

back-of-envelope 估算：
- Avalanche energy：~11.6 pJ/detection
- In-pixel compute：~729 nW/pixel（参考 UltraPhase）
- Break-even：能砍 ≥ 62,845 detections/sec/pixel 就划算
- 例子：90,000 → 25,000 det/sec（230 lux → 30 lux），仍能成像

in-pixel 需要的东西：
- 2-bit register ×4（temporal shift register）
- Adder + shift left（算 score）
- Comparator（跟 η 比）
- 5-bit counter（数 τ_H）
- 10-bit counter（数 detections）
- 5-bit counter（数 inhibition starts）

全都能塞进 macropixel compute block（4×4 block 共享 spatial kernel 的 adder，per-pixel 只存自己的 temporal history）。

而且 saturation look-ahead 更猛：Fibonacci bracketing + look-ahead 之后，detection sequence 只有 **15 种可能组合**（vs 192 种无 inhibition），可以直接编码成 histogram index readout，bandwidth 也省。

---

## 这事让我想到啥

我觉得这个工作本质上是 "attention / dynamic routing / early exit" 哲学落到 physical sensing layer：

- **Transformer attention**：soft gating，决定哪个 token 值得算
- **Token pruning**：hard gating，决定哪个 token 直接丢
- **Photon inhibition**：hard gating，决定哪个 pixel 的光子直接不数

都是同一个 idea——"don't be uniform, allocate resource where it matters"——只是 layer 不同。

它跟 **Bayesian experimental design / active perception** 也一脉相承：你观察 history，决定下一步在哪里 sample。Adaptive gating for SPAD 3D imaging（Po et al. 2022 [46]）是 active illumination 端的类似工作。

它跟 **importance sampling** 的数学结构也很像——oracle allocation $W_i \propto \sqrt{1-Y_i}$ 跟 optimal proposal $q^*(x) \propto |f(x)|p(x)$ 都是"minimize estimator variance via non-uniform allocation"。

---

## 我觉得最值得 follow 的 future direction

paper 自己提的 + 我加的：

1. **Learnable inhibition policy**：现在 kernel 是 handcrafted（P_cr, P_L 都是人工调的）。如果做个 differentiable SPAD simulator，用 RL 或 differentiable programming 训 kernel K_s, K_t, threshold η, hold-off τ_H，可能能突破 handcrafted 上限。这跟 software 2.0 思路完全一致。

2. **End-to-end task-aware**：下游 network 已知的话，inhibition policy 可以 take network gradient w.r.t. input photon count 来决定哪里 sample 更 valuable。这跟 Bayesian experimental design 接近。

3. **Information-theoretic objective**：suppl 提了用 entropy 替代 SNR。我觉得这个方向更有前途——直接 maximize mutual information I(measurement; scene) under energy constraint，比 minimize MSE 更 generalize 到 arbitrary task。

4. **Color SPAD**：现在都是 monochrome。如果 color SPAD，inhibition 可以考虑 spectral——saturated red channel 的光子可以关，blue channel 继续。

5. **Active illumination coupling**：passive 是一面。如果 active（LiDAR、structured light），可以 co-design illumination pattern + inhibition pattern，双向 adaptive。

---

## 一句话总结

> SPAD 的功耗随光变亮爆炸，是因为它无脑数所有光子。但大部分光子（饱和区的、平坦区的）不值得数。在 pixel 电子层面关掉那些"不值"的 pixel，电就从源头省了。用一个 3×3×4 的小卷积（系数全是 2 的幂，in-pixel 可实现）就能决定关哪个，结果能砍 90%+ 光子还能跑 YOLOv8。

---

## 相关链接

- **项目主页 + 代码**：https://wisionlab.com/project/inhibition
- **Quanta Burst Photography**（video reconstruction 用的）：https://dl.acm.org/doi/10.1145/3386569.3392470
- **What Does a One-Bit Quanta Image Sensor Offer?**（SNR_H/W peak 的来源）：https://ieeexplore.ieee.org/document/9922519
- **Exposure-Referred SNR**（SNR_H 的系统推导）：https://ieeexplore.ieee.org/document/9835049
- **Computational Imaging SPAD Cameras**（in-pixel compute hardware 参考，Ardelean PhD thesis）：https://infoscience.epfl.ch/record/305697
- **Holistically-Nested Edge Detection**（edge detection baseline）：https://arxiv.org/abs/1504.06375
- **Fibonacci Exposure Bracketing**：https://www.cs.columbia.edu/CAVE/projects/FHDR/
- **Inhibition decorrelates visual feature representations in the inner retina**（retina inspiration）：https://www.nature.com/articles/nature21394
- **Beyond Binomial and Negative Binomial**（Bernoulli estimation 的 optimal stopping）：https://ieeexplore.ieee.org/document/8715191

---

这事我觉得 cool 在哪：它把"attention is all you need"那种 *resource allocation* 的哲学，push 到了 *physical sensing layer*。以前大家觉得 sensor 是 fixed 的，processing 是 flexible 的；这篇说 sensor 也可以 be flexible，而且 flexible 的方式可以 task-aware。

最大 open question 是能不能 learn 这个 policy 而不是 handcraft 它。如果有人做出 differentiable SPAD simulator + end-to-end trained inhibition kernel，那就是 software 2.0 在 computational imaging 里的落地。

---

# Photon Inhibition for Energy-Efficient Single-Photon Imaging 深度讲解

Andrej，这篇 paper 我觉得非常对你的胃口——它把 SPAD camera 的硬件特性、信号统计、in-pixel compute、以及 computer vision task goal 串成一个统一的 computational imaging framework。下面我把关键 intuition、数学、architectural detail、experimental data 都拆开来讲，并尽量补充我自己的相关联想。

## 1. 问题：SPAD 的 flux-dependent power 是个 fundamental bottleneck

传统 CMOS Image Sensor (CIS) 的 readout power 基本和 scene brightness 解耦——一颗 16 MP 的 IMX206 大约 300 mW，无论你拍 moonless night 还是 sunny noon。但 SPAD 不一样：每个 photon-induced avalanche 都要 burn 几 pJ 到几十 pJ 的 energy，这 energy 直接正比于 photon arrival rate。

paper 里 Fig 1(a) 给的估算非常 striking：把一颗最近的 megapixel SPAD（Ota et al. 2022 ISSCC, 1 Mpixel backside-illuminated charge-focusing SPAD）的 per-avalanche energy extrapolate 到 10's of megapixel format，bright light 下要消耗 **数瓦**，远超 CIS 的 300 mW。

这意味着：SPAD 的 spatial resolution 不能光靠 pixel shrink 来 scale。再小 pixel 虽然 avalanche energy 低（Morimoto 的 scaling law [42]），但只要 flux 上去，total power 还是爆。所以需要一种从根本上 **decouple detection energy from photon flux** 的方法。

## 2. Key Insight：从 retina 偷来的 lateral inhibition

paper 提到 inspiration 来自 human visual system 的 retinal pre-processing：retina 的 inhibitory interneurons 在 small spatio-temporal neighborhood 上 aggregate photon information，然后让邻近 photoreceptor 变得 *less sensitive*。这是 Barlow 1953 [3] 和 Diamond 2017 [13]、Franke 2017 [17] 的工作。

这其实就是一个 "which photons are worth detecting?" 的问题。在 information-theoretic sense，并不是所有 photons 同等 informative：
- 已经 saturated 的 pixel 再多收 photon 不增加 information（Y → 1）
- Spatially flat region 的 photons 用 spatially-adjacent measurement 就能推断
- Edge 上的 photon 比平地上的 photon 信息量大得多

这点跟 **event camera** 哲学类似，但 event camera 是在 readout / data transfer layer 做 reduction，photon inhibition 是直接在 **detection layer**——电子层面 disable SPAD pixel，avalanche 根本不发生，power 从源头就省了。

## 3. Observation Model：把 SPAD 当 Bernoulli 通道来建模

### 3.1 基本 Poisson + Bernoulli 模型

设 photon flux 是 φ（已含 PDP，即 effective flux），exposure time T，则 exposure

$$H := \phi T$$

Photon conversion count K ~ Poisson(H)，即 $\mathsf{P}(K=k;H) = \frac{H^k e^{-H}}{k!}$。这里 **H 是平均 photon arrival per exposure window**（无量纲），**k 是离散 photon 数**。

SPAD 在每个 binary exposure window 只能记录 0 或 1（≥1 photon），所以是 Bernoulli trial。Detection probability

$$Y := 1 - \mathsf{P}(K=0;H) = 1 - e^{-H}$$

这里 **Y 是 binary rate**，也就是"一个 exposure window 里至少检测到一个 photon 的概率"，取值 [0, 1]。

W 个 measurement window 累积后，total detections

$$D := \left[\sum_{n=1}^{W} B_n\right] \sim \mathsf{Binomial}(W, 1-e^{-H})$$

其中 **B_n 是第 n 个 window 的 Bernoulli outcome (0/1)**，**W 是 measurement 总数**，**D 是 detection 总数**。

估计：

$$\widehat{Y} = \frac{D}{W}, \quad \widehat{H} = -\ln(1-\widehat{Y})$$

这个 $-\ln(1-\widehat{Y})$ 是 key——它是从 binary observation 反推 continuous exposure 的 MLE，是 quanta image sensor 的核心变换。当 Y → 1 时 $\widehat{H}$ 爆炸，这就是 SPAD 的 **soft saturation**：bright light 下 estimation variance 灾难性放大。

### 3.2 加入 inhibition 后

inhibition pattern M 是 binary state，M_n = 1 表示第 n 个 measurement enabled。

$$W_{inh.} := \sum_{n=1}^{W} M_n \leq W$$

$$D_{inh.} := \left[\sum_{n=1}^{W} M_n B_n\right] \sim \mathsf{Binomial}(W_{inh.}, 1-e^{-H})$$

关键点：因为 Poisson arrival 的 **memoryless property**，disabled 期间的 photons 不影响 enabled 期间的 statistics。这是一个很重要的 assumption——它要求 inhibition 的 transition 跟 clocked recharge 同步，并且 SPAD 的 dead-time 远小于 clock period。

## 4. Energy-Aware Performance Metrics：从 SNR_H 到 efficiency

这是 paper 我觉得最有 conceptual contribution 的地方。他们没直接用 SNR_H（exposure-referred SNR），而是提出两个 *效率* metrics。

### 4.1 经典 SNR_H

$$\mathsf{SNR}_H = \frac{H}{\sqrt{\mathbb{E}[(\widehat{H}-H)^2]}} = H\sqrt{\frac{W}{e^{H}-1}}$$

变量解释：**分子 H** 是真实 exposure，**分母** 是估计 $\widehat{H}$ 的 RMSE。化简后依赖 H 和 W。

Behavior：
- H << 1: SNR_H ≈ H·sqrt(W)，纯 shot noise limit
- H ≈ 1.6: SNR_H 达到 peak
- H >> 1: SNR_H 下降，因为 $e^H - 1$ 项 dominant——这就是 soft saturation

### 4.2 Detection Efficiency SNR_H/D^2

$$\mathsf{SNR}_{H/D}^2 := \frac{\mathsf{SNR}_H^2}{\mathbb{E}[D]} = \frac{H^2 e^{-H}}{(1-e^{-H})^2}$$

这里 **E[D] = W(1-e^{-H})** 是 expected detections，正比于 avalanche energy。把 SNR^2 除以 E[D] 就是"per unit avalanche energy 能买多少 information"。

Behavior（Fig 3a 红线）：
- H << 1: $\mathsf{SNR}_{H/D}^2 \approx 1$（上界）——dark regime 每个 photon 都 informative
- H ≈ 0.5 开始 degrade——saturation 开始 bite
- H → ∞: → 0——bright photon 完全 uninformative

**Intuition**：这个 metric 告诉你 bright pixel 的 photon 是浪费 energy 的。

### 4.3 Measurement Efficiency SNR_H/W^2

$$\mathsf{SNR}_{H/W}^2 := \frac{\mathsf{SNR}_H^2}{W} = \frac{H^2 e^{-H}}{1-e^{-H}}$$

这里 **W** 是 measurement 总数，正比于 readout energy、in-pixel counter depth、latency。

Peak 在 **H = 1.59, Y = 0.80**（Chan 2022 [9]）。这就是为什么很多 QIS/SPAD 设计选这个 operating point。

**Intuition**：这两个 metrics 给出两个 constraint axis——detection energy（avalanche）和 measurement energy（readout/compute/latency）。理想 inhibition policy 要同时 track 两条曲线。

## 5. Spatio-Temporal Inhibition Policies：在 in-pixel 实现的 lightweight filter

### 5.1 Calculation-based inhibition (Fig 2)

核心公式：

$$S(i,j,t) = K * \left[(2F(i,j,t)-1) \cdot M(i,j,t)\right]$$

变量解析：
- **(i,j)** 是 pixel 坐标，**t** 是 frame index
- **F(i,j,t)** 是 binary photon cube：M=1 且检测到 photon 时为 1，否则 0
- **2F - 1** 把 {0,1} 映射到 {-1, +1}
- **乘 M(i,j,t)** 把 disabled 期间的 entry 设为 0——这是 ternary 编码：detection=+1，no detection but enabled=-1，disabled=0
- **K** 是 spatio-temporal kernel，dimension L×H×T，可分离为 $K = K_s \otimes K_t$
- **\*** 是 3D convolution
- **S** 是 inhibition score

Decision rule：若 S(i,j,t) > η，则 disable 接下来 τ_H 帧：

$$M(i,j,t') = 0 \quad \text{for } \{t' | t+1 \le t' \le t+1+\tau_H\}$$

**τ_H** 是 hold-off time（以 frame 为单位），**η** 是 inhibition threshold。

### 5.2 为什么用 ternary 编码

这个 (2F-1)·M 的 ternary 编码我觉得很 elegant：
- 把"我看到 0 个 photon 但 pixel 是 active" 这种 *evidence of absence* 信号化成 -1
- 把"pixel 被关掉了" 信号化成 0（neutral，不参与 score）
- 这样 spatio-temporal kernel 能 distinguish "low flux region" vs "inhibited region"

如果直接用 F·M（binary），你就丢失了"看到 0 也是 information"的事实，会导致 inhibition 在低 flux 区误启动。

### 5.3 具体 Policies (Suppl. S3.3)

paper 测了 4 种 spatial kernel + temporal kernel = [1,1,1,1] 组合：

**P_cr (Center + Ring):** 突出中心像素
$$K_s = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 8 & 1 \\ 1 & 1 & 1 \end{bmatrix}, \quad \eta=12, \tau_H=32$$

**P_L (Laplacian):** 检测 spatial variation
$$K_s = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}, \quad \eta=24, \tau_H=4$$

**P_avg (Average):** 纯 spatio-temporal average
$$K_s = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}, \quad \eta=6, \tau_H=32$$

**P_s (Single-pixel):** 不看 neighbor
$$K_s = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \quad \eta=2, \tau_H=32$$

注意所有系数都是 power of 2，因为最终要做 in-pixel hardware——乘 8 就是 shift left 3 bits，乘 -8 就是 shift + sign flip，比真实 multiplier 便宜几个数量级。

### 5.4 Edge-Enhancement Policy (Sec 6.2)

这是为 edge detection task 专门设计的：

$$\text{Inhibit if } \left((\eta_1 < S_1 < \eta_2) \wedge (S_2 > \eta_3)\right) \vee (S_2 > \eta_4)$$

其中：
- **S_1** = 3×3 Laplacian score
- **S_2** = 3×3 average score
- **η_1 = -12, η_2 = 12**：检测 spatial variation 小的区域
- **η_3 = 4**：要求 neighborhood 不太暗（dim region 别 inhibit，省 energy 无意义）
- **η_4 = 16**：bright pixel 直接 inhibit

这个 Boolean 组合的逻辑：
1. "spatial flat" ∧ "neighborhood 不太暗" → inhibit（flat mid-bright region 的 photon 不 informative）
2. OR "very bright" → inhibit（saturated region 的 photon 不 informative）

保留了 edges 和 dark regions。这是个非常 task-aware 的 design。

### 5.5 Saturation Look-Ahead (Fig 4)

这是给 dynamic scene / video 用的 policy，避免长时间 hold-off 导致 motion blur。

Design：用 exposure bracketing 序列，T_1 < T_2 < T_3，每个 cycle 用同样 exposure。如果 cycle i 累积 detections 超过 threshold d_i，下一个 cycle 直接 inhibit pixel。

公式上（suppl S3.5）：
$$Y_1 = 1 - e^{\frac{T_1}{T_2}\log(1-Y_2)} = 1 - e^{0.2 \cdot \log(1-0.99)} \approx 0.60 = 6/10$$

意思：如果短 exposure T_1 已经达到 60% detection rate，那么长 exposure T_2 = 5·T_1 几乎肯定 saturate（99%），所以提前关掉。

**Intuition**：这相当于一个 early-warning 系统——用 cheap 的短 exposure 测量来 decide 是否值得做 expensive 的长 exposure。

paper 用 Fibonacci bracketing：**T = {1, 1, 2, 3, 5, 8, 13, 21}**（参考 Gupta 2013 [23]）。Fibonacci 选的好处是每个 exposure 都是前两个的和，in hardware 容易生成，且 log-spaced 覆盖 dynamic range。Look-ahead 阈值 **{2, 1, 1, 1, 1, 1}** 对应 6 个 unique exposure length 的 transition。

## 6. Oracle Analysis：最优 measurement allocation 长什么样

Suppl S4.1 给了一个 very instructive 的分析。如果 image 是已知的（oracle），怎么分配 W_i 来 minimize image MSE？

Per-pixel variance:
$$\sigma_{Y_i}^2 = \frac{1}{W_i} Y_i (1-Y_i)$$

约束总 detections：$\sum_i W_i Y_i = D_T$

Lagrange multiplier 解得：

$$W_i^* = \frac{D_T}{\sqrt{Y_i}} \cdot \frac{\sqrt{Y_i(1-Y_i)}}{\sum_j Y_j \sqrt{1-Y_j}} \propto \sqrt{1-Y_i}$$

**Intuition**：测量数 ∝ sqrt(1-Y)。当 Y→1（saturated）时 allocation → 0，当 Y→0 时 allocation 最大。这个 allocation 说"把 measurement budget 倾斜给 dim pixel"。

推广到 general loss $\mathcal{L}_{im.} = \sum_i \frac{E_i}{W_i}$：

$$W_i^{opt.} \propto \sqrt{\frac{E_i}{Y_i}}$$

Suppl Table 3 给了 4 个 metric 的最优 allocation，我整理一下：

| Metric | $E_i$ | $W_i^{opt.}$ | $\mathbb{E}[D_i]^{opt.}$ |
|---|---|---|---|
| Binomial MSE | $Y_i(1-Y_i)$ | $\propto \sqrt{1-Y_i}$ | $\propto Y_i\sqrt{1-Y_i}$ |
| Exposure-referred MSE | $\frac{Y_i(1-Y_i)}{(1-Y_i)^2} = \frac{Y_i}{1-Y_i}$ | $\propto \frac{1}{\sqrt{1-Y_i}}$ | $\propto \frac{Y_i}{\sqrt{1-Y_i}}$ |
| Relative exp-referred MSE (= SNR_H^2) | $\frac{Y_i(1-Y_i)}{H_i^2(1-Y_i)^2} = \frac{1}{H_i^2(1-Y_i)}$ | $\propto \frac{1}{H_i\sqrt{1-Y_i}} = \frac{SNR_{H/D,i}}{SNR_{H/W,i}}$ | $\propto \frac{1}{SNR_{H/D,i}}$ |
| SNR_H/W-tracker (k=2) | $\frac{1}{SNR_{H/W,i}^2} \cdot \frac{SNR_{H/D,i}^2 \cdot SNR_{H/W,i}^2}{Y_i} = \frac{SNR_{H/D,i}^2}{Y_i}$ | $\propto \frac{SNR_{H/W,i}}{Y_i} \propto \frac{1}{SNR_{H/D,i}}$ | $\propto SNR_{H/W,i}$ |

观察：
- Binomial MSE：倾向 dim pixel（√(1-Y) weighting）
- Exposure-referred MSE：倾向 bright pixel（1/√(1-Y)）——这是个 surprising result，因为它在 radiance domain 衡量 error，bright pixel error 放大
- Relative exp-referred MSE：peaky allocation，同时 dim 和 bright 都受 penalty，倾向于 mid-tone
- SNR_H/W-tracker：正好对应 saturation look-ahead policy 的 behavior

paper 自己说 saturation look-ahead 这个"loss function"看起来 counter-intuitive（sum of SNRs），但实际效果是 favoring dim pixel，跟 binomial MSE 一致。这是个 retrospective rationalization。

**我的联想**：这跟 importance sampling、optimal proposal distribution 哲学很像——你想 sample from 一个让 estimator variance 最小的分布，而不是 uniform 分布。也跟 neural network 里 *learnable* routing / mixture-of-experts 的 expert selection 类似。

## 7. Experimental Results：硬件实测 + 仿真

### 7.1 Image Reconstruction (Fig 5)

- 数据集：BSDS500 20 张图
- Exposure bracketing: {0.1, 1.0, 10.0} ppp
- 每 exposure 1000 binary frames
- HDR reconstruction：SNR^2 weighting [21]
- Metric: SSIM

Result（exposure bracket）：
- SSIM = 0.7：detections 减少 **42%** 平均
- SSIM = 0.8：仍有显著 reduction

Result（single exposure, 1.0 ppp，更难）：
- SSIM = 0.7：detections 减少 **14%**
- 这比 oracle 上限（suppl Table 2 显示 ~13% SSIM 0.7）还要好一点——意味着实际 policy 接近 oracle performance

### 7.2 Edge Detection (Fig 6)

- 数据集：BSDS500 with ground truth boundary
- Pipeline: binary rate image → HED [58] → structured edge toolbox [14] eval
- Metric: OIS F-score

Result（at low photon counts）：
- Edge-enhancing policy 减少 detections **30%** at equal F-score
- F-score plateau ~0.813（at high photon count），与原 image 持平

### 7.3 YOLOv8 Object Detection (Fig 1c,d)

- Real SwissSPAD2 data
- Quanta burst photography reconstruction [37] 后喂 YOLOv8
- **95% photons inhibited**，object detection 仍成功

这是非常 impressive 的 number。说明 high-level vision task 对 photon-level noise 的 robustness 远超过低-level image quality metric（如 SSIM）。这跟 image classification 不需要 pixel-perfect reconstruction 的 classic insight 一致。

### 7.4 SwissSPAD2 Video Sequence (Fig 7)

硬件：SwissSPAD2 [54]，512×256，97,700 FPS，>580,000 binary frames，光照从 <1 lux 到 >4,000 lux（7 stops）。

三个对比：
1. No inhibition
2. Sub-sampling 10×（固定丢 9/10 frames）
3. Saturation look-ahead + Fibonacci bracketing

Keyframes 处理：12,000 binary frames per keyframe → quanta burst photography reconstruction。

Result：
- **Bright light**：所有 policy 都 OK，saturation look-ahead 可以 >90% inhibition
- **Mid/low light**：sub-sampling 失败（家具轮廓、人形丢失），saturation look-ahead 仍保留 detail
- **Cumulative detections**：look-ahead 比 sub-sampling 少——因为它 bright light 更 aggressive，dim light 保守

**Intuition**：sub-sampling 是 *flux-agnostic*，所有场景一视同仁；look-ahead 是 *flux-adaptive*，在需要 photon 的地方多收，不需要的地方少收。这本质上是个 closed-loop control 问题。

## 8. Hardware Implementation Estimate (Suppl S2, S7)

这是 paper 里很务实的部分。他们没自己 tape out，但给了 back-of-envelope：

**Energy balance**：
- Avalanche energy: ~11.6 pJ/detection（来自 [48]）
- Computation: ~729 nW/pixel（UltraPhase [2] 估的，但他们的 inhibition 应该更轻）
- Break-even: 如果能 inhibit ≥ 62,845 detections/sec/pixel 就划算
- 例子：90,000 det/sec（~230 lux）→ 25,000 det/sec（~30 lux），仍能成像

**In-pixel circuitry（Suppl Table 5）**：
- SPAD control: PMOS + OR gate（per-pixel）
- Inhibition score: 2-bit register ×4（temporal shift register，per-pixel）+ adder + shift left（per-macropixel）
- Inhibition control: comparator + 5-bit counter for τ_H（per-pixel）
- Measurement results: 10-bit detection counter + 5-bit inhibition-start counter（per-pixel）

Critical observation：所有 multiplication 都是 power-of-2，全是 bit shift。Spatial kernel 共享给 macropixel（4×4 block，参考 [2]）。Temporal kernel T=4 意味着只需 8 个 2-bit register 存 signed detection history。

**Saturation look-ahead 的 memory footprint** 更小：Fibonacci bracketing [1,1,2,3,5,8,13,21] 配合 look-ahead policy 后，只有 **15 个 unique (B_T, M_T) 组合**可能（vs. 192 个无 inhibition 时）。这意味着可以把 detection sequence 编码成 histogram index 直接 readout，进一步省 bandwidth。

## 9. 与其他工作的关系 & 我的联想

### 9.1 直接相关

- **Quanta Image Sensor (QIS)** by Fossum [16, 35]：jots 不用 avalanche，所以没 flux-dependent power 问题。paper scope 限定在 SPAD 但 QIS 也是 single-photon binary sensor，philosophy 上共通。
- **Quanta Burst Photography** [37]：paper 的 video 实验直接用这个做 reconstruction。
- **High Flux Passive Imaging** [30]：先期工作，描述 SPAD 在 bright light 的 saturation 问题。
- **Inter-Photon Imaging** [29]：用 inter-photon timing 做 imaging。

### 9.2 Broader 联想

**1. Event Camera Philosophy**：event camera 是 "only transmit change"，photon inhibition 是 "only detect informative photon"。但 inhibition 在更底层（detection layer vs. readout layer），所以省的是 avalanche energy 而不只是 data transfer。

**2. Active Inference / Bayesian Sensing**：这个 paper 的 inhibition policy 本质上是个 closed-loop policy：观察 history → 决定下一 action（enable/disable pixel）。跟 Kaushik Bhojan 的 Texas Instruments work、Adaptive Gating for SPAD 3D imaging [46] 一脉相承。Active perception 文献里这个 idea 很成熟。

**3. Attention Mechanism**：transformer attention 是 soft gating（softmax over tokens），photon inhibition 是 hard gating（binary enable/disable）。两者都在做 "where to allocate compute / sensing resource"。Vision Transformer 的 patch selection、token pruning 是 conceptual cousin。

**4. Sparse Coding / Compressive Sensing**：Compressive sensing 在 acquisition time 做线性 projection；photon inhibition 在 acquisition time 做 data-dependent thinning。后者是 non-linear、adaptive 的，前者是 linear、fixed 的。

**5. Retina 的 Lateral Inhibition**：经典工作 Barlow 1953 [3]、Franke 2017 [17]——retinal ganglion cell 的 receptive field 中心兴奋 + 周边抑制。paper 的 P_cr kernel（center ×8 + ring ×1）就是 center-surround 结构，但用于 inhibition 而非 firing。

**6. neuromorphic computing**：在 in-pixel 做 compute 是 neuromorphic 哲学（sensor + processor co-design）。IBM TrueNorth、Intel Loihi 是纯 compute 端，SPAD with in-pixel inhibition 是 sensor-compute co-design。

**7. Importance Sampling & Optimal Proposal**：oracle analysis 里的 $W_i \propto \sqrt{1-Y_i}$ 跟 importance sampling 的 optimal proposal $q^*(x) \propto |f(x)|p(x)$ 数学结构类似——minimize estimator variance via non-uniform allocation。

**8. Task-Aware Sensing**：edge-enhancement policy 是为 HED 边缘检测专门设计的。这预示着 future work 可能是 end-to-end learnable inhibition policy，用 differentiable simulation 训出 task-specific kernel。Paper 明确提了这个 open direction。

## 10. Limitations & Future Directions（paper 自己提的 + 我加的）

**Paper 提到的**：
- Noise model 简化，没考虑 pixel sensitivity variation、crosstalk、afterpulsing
- 数据依赖 stopping 的 unbiased estimator [26] 没用
- Implementation cost 没考虑 readout energy 和 compute energy 的 holistic model
- 没 generalize 到 arbitrary vision task，每个 task 要单独 design policy

**我觉得还可以做的**：
1. **Learnable inhibition policy**：用 differentiable SPAD simulator + reinforcement learning，learn kernel K_s, K_t, threshold η, hold-off τ_H end-to-end for downstream task loss。可能能突破 handcrafted policy 的上限。
2. **Multi-task inhibition**：一个 policy 同时 support reconstruction + detection + segmentation，类似 multi-task learning。
3. **Inhibition + Neural Network Co-design**：如果下游 network 是 known 的，inhibition policy 可以 take network 的 gradient w.r.t. input photon count 来决定哪里 sample 更 valuable。这跟 Bayesian experimental design 接近。
4. **Cross-pixel inhibition with global context**：现在 policy 都是 local 3×3 kernel，理论上 global context 能进一步优化（e.g., sky region 整片 inhibit）。但 in-pixel 实现挑战很大。可能 hierarchical design：local inhibition + occasional global update。
5. **Causal bias correction**：data-dependent stopping 引入 bias，suppl 提到 Haldane 1945 的 unbiased estimator [26] 没用，但其实可以 integrate。这对 quantitative imaging（如 FLIM、LiDAR）很关键。
6. **Color SPAD**：现在都是 monochrome。如果 color SPAD array，inhibition 可以 take spectral information into account，比如 saturated red channel 的 photon 可以 inhibit blue channel pixel 来做 chromatic inference。
7. **Active illumination coupling**：paper 是 passive imaging。如果 active illumination（LiDAR、structured light），可以 co-design illumination pattern + inhibition pattern——双向 adaptive sensing。
8. **Information-theoretic analysis**：suppl 提了 entropy 替代 SNR 的可能 [22]。我觉得这个方向更有前途，因为 entropy 直接 measure information content，跟 downstream task 更 align。一个 ideal inhibition policy 应该 maximize mutual information I(inhibited measurement; scene) under energy constraint。

## 11. 我觉得 paper 最大的 conceptual contribution

不是某个具体 policy（P_cr, P_L 这些都是 handcrafted heuristic），而是它**提出了一个问题**：

> "Which photons should be detected?"

之前整个 SPAD / QIS 文献都默认 "detect all photons, then process"。这个 paper 第一次 system-level 提出可以在 detection 阶段就做 selection，并且 selection policy 可以 be tuned for downstream task。这跟 deep learning 里 "early exit"、"dynamic routing"、"conditional computation" 哲学同源，但落地到 physical sensing layer。

第二个 contribution 是 **energy-aware performance metric framework**。SNR_H/D 和 SNR_H/W 这两个 efficiency metric 应该会成为 SPAD 文献的标准 tool——任何 future SPAD 算法都可以在这两个 axis 上 plot。

## 12. 一些值得 follow up 的 references

我把 paper 里我特别推荐 follow up 的列出来：

1. **Quanta Burst Photography** (Ma et al. SIGGRAPH 2020) - https://dl.acm.org/doi/10.1145/3386569.3392470 - paper 的 video 实验直接用这个做 reconstruction
2. **What Does a One-Bit Quanta Image Sensor Offer?** (Chan 2022) https://ieeexplore.ieee.org/document/9922519 - SNR_H/W peak at H=1.59 的来源
3. **Exposure-Referred SNR for Digital Image Sensors** (Gnanasambandam & Chan 2022) https://ieeexplore.ieee.org/document/9835049 - SNR_H metric 的系统推导
4. **Computational Imaging SPAD Cameras** (Ardelean PhD thesis, EPFL 2023) https://infoscience.epfl.ch/record/305697 - in-pixel compute 的 hardware 参考
5. **Holistically-Nested Edge Detection** (Xie & Tu ICCV 2015) https://arxiv.org/abs/1504.06375 - edge detection baseline
6. **Fibonacci Exposure Bracketing** (Gupta et al. ICCV 2013) https://www.cs.columbia.edu/CAVE/projects/FHDR/ - bracketing 设计 motivation
7. **Inhibition decorrelates visual feature representations in the inner retina** (Franke et al. Nature 2017) https://www.nature.com/articles/nature21394 - retina inspiration
8. **Beyond Binomial and Negative Binomial** (Medin et al. 2019) https://ieeexplore.ieee.org/document/8715191 - Bernoulli parameter estimation 的 optimal stopping theory

**项目页面**：https://wisionlab.com/project/inhibition - 有代码和 supplementary video

## 13. 总结：核心 takeaways

1. **SPAD 的 flux-dependent power 是 fundamental bottleneck**，限制 spatial resolution scale up
2. **Photon inhibition = data-dependent pixel disable**，受 retina lateral inhibition 启发
3. **两个 efficiency metrics**：SNR_H/D^2（detection efficiency）和 SNR_H/W^2（measurement efficiency），分别 capture avalanche energy 和 readout energy
4. **Lightweight in-pixel policy**：3×3 spatial × 4 temporal kernel，所有系数 power-of-2，ternary 编码 (2F-1)·M 保留 evidence of absence
5. **Task-specific policies**：P_cr 给 image reconstruction，Laplacian+avg Boolean 给 edge detection，saturation look-ahead 给 video
6. **Real hardware 实验**：SwissSPAD2 上 95% inhibition 仍能跑 YOLOv8 object detection
7. **Oracle 上限**：W_i ∝ √(1-Y_i) 是 MSE-optimal allocation，proposed policy 接近这个上限
8. **硬件可行性**：break-even ~62,845 inhibitions/sec/pixel，远低于 typical photon rate

最大的 future direction 是 **learnable, task-aware inhibition policy**，用 differentiable SPAD simulator + end-to-end training，可能突破 handcrafted policy 的 ceiling。这跟你一直倡导的 "software 2.0" 思路完全契合。

---

我希望这个讲解 build up 了 intuition。如果想深入任何一个 policy 的推导细节、具体 hardware implementation 数字、或者 oracle allocation 的泛化，我可以继续展开。
