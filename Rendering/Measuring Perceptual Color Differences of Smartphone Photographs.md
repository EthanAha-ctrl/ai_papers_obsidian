---
source_pdf: Measuring Perceptual Color Differences of Smartphone Photographs.pdf
paper_sha256: e45fa9481fa2f33888b453434c6b821ff12d1a350caf2e15eefb04b3d2483022
processed_at: '2026-08-05T17:07:39-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

好，我换一种方式讲，先从大图景开始，再慢慢 zoom in。

---

## 一句话说清楚这篇 paper 在干嘛

**手机拍照时代，怎么衡量两张照片"颜色差多少"这件事，其实没人好好做过。这群人就做了一件事：搞了个大 dataset，然后训了个 tiny network 来当 color difference meter。**

---

## 为什么这件事值得做？

你想想，你拿 iPhone 和 Samsung 拍同一个夜景，出来的照片 structural detail 差不多——大楼轮廓、窗户位置都一样——但 color 完全不同。一台偏暖偏绿，一台偏冷偏蓝。这种 color difference 怎么 measure？

传统 color science 有个叫 CIEDE2000 的公式，1976 年那代人搞的。但问题是，**这个公式是在 uniform color patch 上 calibrate 的**——就是放两块纯色色块在灰色背景上，问人 "这两块差多少"。这跟手机照片完全是两个世界：

- 手机照片有 spatial structure，人看 color 是在 context 里看的，不是逐 pixel 比的
- 手机 ISP 产生的 color distortion 是 highly nonlinear 的，patch experiment 根本 cover 不到
- 之前有人试过在 natural image 上测 CD，但 dataset 都很小，几十张图，distortion 也都是 linear transform，toy experiment 性质

所以现状就是：**color science 社区有一堆精心设计的公式，computer vision 社区有一堆 deep learning metric，但没人知道它们在真实手机照片上到底好不好使。**

---

## 他们干了什么？

两件事：

### 第一件：搞了个 SPCD dataset

30,000 对 image pair，每对都有 human rating 的 color difference score。怎么搞的：

**图片来源**——四种 realistic distortion：

1. **六台旗舰手机拍同一场景**：iPhone 12 Pro, HUAWEI Mate40 Pro, OnePlus 7 Pro, Samsung S21, OPPO Find X3, Xiaomi 11。同一场景六张图，color 天然不同。这部分 10,005 对，而且故意保留轻微 misalignment（因为真实场景中手机之间 framing 总会差一点）

2. **Photoshop 模拟 ISP 操作**：调 white balance、color correction、tone mapping、gamma，模拟 ISP pipeline 里的 color manipulation

3. **iPhone 内置 filter**：vivid, dramatic, mono 等 9 个 filter，模拟用户 post-processing

4. **错误的 color profile**：sRGB 图在 DCI-P3 显示器上看，或者反过来。这是 color management 出错时常见的 distortion

**Human rating 怎么搞的**——这是最 labor-intensive 的部分：

- 两台 EIZO CG319X 专业级显示器，HDR + wide gamut，全面 colorimetric characterization
- 完全 dark room，viewing distance 1m 固定
- 20 个 subject（10 男 10 女），Ishihara test 筛过 color vision
- 用 gray scale method：给 5 个 reference gray pair（对应 ΔE ≈ 0, 1.7, 3.4, 6.8, 13.6），subject 拿 slider 评每对图相当于哪个 gray pair level
- 每人每天 300 对，每 30 分钟休息，整个实验跑 4 个月
- 总共 600,000 个 rating

然后 raw gray scale grade 通过一个 exponential function 转成 ΔE 单位：

$$\Delta V = 1.6036 \cdot \exp(0.5391 \cdot G) - 1.2943$$

$G$ 是 gray scale grade（0-4），$\Delta V$ 是 perceptual color difference。这个 mapping 在 reference gray pair 上验证过，max error < 0.37 ΔE，远低于 JND。

### 第二件：训了个 CD-Net

CD-Net 的核心 idea 非常 clean，我分三步讲：

**Step 1: Coordinate transform**

传统 CD 公式第一步都是 color space transform（RGB → CIELAB 之类）。CD-Net 用一个小 CNN 来做这件事，但有几个 key design：

- Front-end 用两个 kernel size 的 conv：1×1（pixel-wise，类似 CIELAB 的 pointwise transform）和 11×11（patch-wise，capture spatial context）
- Concatenate 之后接几层 1×1 conv + Leaky ReLU
- **No downsampling**——保持 spatial resolution，为了能出 local CD map
- **No bias term**——保证 scaling invariance：$f(\alpha x) = \alpha f(x)$

最终把 RGB 的 3 维 transform 成 12 维 feature。整个 network 14,464 params，tiny。

**Step 2: Mahalanobis distance**

传统公式第二步是 distance calculation。CIEDE2000 有一堆 weighting factor 和 hue rotation correction。CD-Net 用一个 learnable 的 Mahalanobis distance：

$$\Delta E = \sqrt{(f(x) - f(y))^T \cdot S^{-1} \cdot (f(x) - f(y))}$$

$S$ 是一个 12×12 的 positive semi-definite matrix，通过 Cholesky decomposition parameterize（$S = LL^T$，$L$ 是 lower triangular）。只有 78 个 params。

**直觉**：$S$ 的对角元素就是 per-channel weighting（对应 CIE94 的 $S_L, S_C, S_H$），非对角元素是 channel 间 cross-coupling（对应 CIEDE2000 的 hue rotation term $T$）。所以 Mahalanobis 自动 generalize 了传统公式的 correction 机制。

**Step 3: Global average**

所有 spatial position 的 local CD 简单平均，得到 global CD score。没有 fancy weighting。

**Training**：MSE loss 对 ground-truth ΔV，Adam optimizer，70/10/20 split，content independent，重复 10 次取 mean。

---

## 结果怎么样？

### Quantitative：跟 33 个 baseline 比

Table 3 是 main result。几个 take-away：

**CIE 公式族**（CIELAB, CIE94, CIEDE2000, CIECAM02, CIECAM16）在 SPCD 上 STRESS 都在 31-35 之间，correlation 0.7 左右。**这些几十年来 color science 的 golden formula，在手机照片上只有 mediocre performance。** 说明 patch 上 calibrate 的东西 generalize 不好。

**S-CIELAB**（加 spatial filter 的 CIELAB）反而比 CIEDE2000 还差一点。说明 naive spatial filtering 不够。

**IQA models**（SSIM, LPIPS, DISTS）很烂。STRESS 40-66，correlation 0.3-0.4。因为 IQA 是 measure distortion 导致的 quality loss，但 SPCD 里的 color change 不一定是 quality degradation（比如 iPhone filter 出来的 vivid warm 图，color 跟原图差很多，但 quality 没变差）。

**Deep CNN**（VGG 14.7M, ResNet-18 11.2M, UNet 31M）STRESS 18-21，correlation 0.84-0.89。ResNet-18 最好。**但 CD-Net 只有 14K params 就达到 STRESS 21.4, correlation 0.84，跟 14M-31M 的 CNN 持平。**

### Qualitative：local CD map

STRESS 数字接近的 method，CD map 质量天差地别：

- CIEDE2000：noise 多，因为 pixel-wise
- FLIP：过度强调 edge，sky 和 building 边界一条亮线
- ResNet-18：CD map noisy
- UNet：ringing artifact
- **CD-Net**：smooth transition，edge 处合理过渡，最像 human perception

这是 paper 一个重要 argument：**光看 global correlation number 不够，要看 local CD map 才知道 method 是不是真学了 perceptual rule，还是 overfit 到 global score。**

### Generalization

**TID2013 color subset**（没见过的 distortion type）：CD-Net 比 pixel-wise metric 好，但比 LPIPS 和 FLIP 差。因为 LPIPS/FLIP 在 ImageNet pretrain 时可能见过类似 color appearance。

**COM dataset**（homogeneous color patch，从没见过）：CD-Net STRESS 38.9，CIELAB 45.2，CIEDE2000 29.0。**CD-Net 比 CIELAB 好！** 一个在 natural image 上 train 的 12 维 learned space，在 patch data 上比几十年手工设计的 CIELAB 还 uniform。这很 surprising。

### Proper metric 验证

一个 metric 要满足四个性质：non-negativity, symmetry, identity of indiscernibles, triangle inequality。

前两个自动满足。后两个做 empirical test：

**Reference image recovery**：给一张 reference image $x$，从 noise 或 tone-altered image $y$ 出发，optimize $\min_y \Delta E(x, y)$，看能不能 recover 出 $x$。

- CD-Net：成功 recover，最终 CD 远低于 JND
- VGG, ResNet-18, UNet, CAN：全部 fail，recover 出完全不同的图，artifact 严重

**Triangle inequality**：测了 ~2M triplet，只有 VGG 在 13 个上 violate，CD-Net 零 violation。

**结论**：CD-Net empirical 上是 proper metric，deep CNN 不是。**这解释了为什么 deep CNN 的 STRESS 数字虽然好，但在 perceptual optimization 里不好使——它们的 feature space geometry 是 messed up 的。**

---

## 为什么 CD-Net 能 work？我的理解

### 1. Inductive bias 对了

CD-Net 把 domain knowledge encode 进 architecture：
- Two-step structure（transform + distance）= 传统 CD formula 的结构
- Multi-scale filter bank = spatial context 的 necessity
- Mahalanobis = 传统 weighting/rotation correction 的 generalization
- No bias + scaling invariance = color perception 的 scale invariance

剩下的 freedom 让 network 从 data 学。这比 throw 一个 ResNet 进去让它自己 figure out 要好。

### 2. 小 model 避免 overfit

14K params vs 14M params。Deep CNN 能 fit training data 更好，但学到的 feature space geometry 不好——reference image recovery fail 就是证据。CD-Net 被迫学更 simple 的 transform，反而 generalize 更好。

这跟 [Belkin et al. 的 double descent](https://www.pnas.org/doi/10.1073/pnas.1903070116) 有点像——但这里是 "small model + right bias" sweet spot，不是 "interpolate regime"。

### 3. Dataset realism

SPCD 的四类 distortion 都是真实手机摄影会遇到的。比之前的 linear color transform 或小 dataset 真实得多。Train on realistic data，test on realistic data，自然 work。

---

## 跟你平时关注的东西的 connection

### LPIPS 的对比

LPIPS [Zhang et al. 2018](https://arxiv.org/abs/1801.03924) 也是 learn perceptual metric，但用 AlexNet/VGG backbone + ImageNet pretrain + 大量 pairwise preference data。CD-Net 用 14K params + 30K pairs 从 scratch train。

两条路线的 trade-off 很明显：
- LPIPS：general-purpose，across domain 都还行，但 feature space geometry 不好，perceptual optimization 不好使
- CD-Net：domain-specific（color difference of smartphone photos），但 metric property 好，perceptual optimization 好

这其实是一个 universal 的 tension：**用大 model 学 general representation，还是用小 model + right bias 学 task-specific representation。** CD-Net 是后者的一个 clean 例子。

### Perceptual loss 的角度

如果你在训练 ISP pipeline 或 image generation model，需要一个 perceptual loss 来 optimize color reproduction，你会选什么？

- L1/L2 in RGB：太简单，perceptually irrelevant
- LPIPS：popular，但 [Ding et al. 2021](https://arxiv.org/abs/2004.07728) 和这篇 paper 都显示 LPIPS 对 color 不敏感，feature space geometry 也不好
- CIEDE2000：perceptually meaningful，但对 natural image generalize 不好
- CD-Net：perceptually meaningful + generalize 好 + proper metric + lightweight

**CD-Net 这种 proper metric + lightweight + domain-specific 的组合，其实非常适合做 perceptual loss。** Reference image recovery 实验就是 mini version 的 "用 CD-Net 做 perceptual optimization"。

### Diffusion model 的 connection

现在 diffusion model 生成图像，怎么 evaluate perceptual quality？FID？IS？Inception score？这些都是 distribution-level metric，per-image quality 不好 measure。

CD-Net 这种 lightweight proper metric 如果能 extend 到 "generated image vs. real image" 的 color difference，可能比 FID 更 interpretable。当然需要新的 dataset。

### Metric learning 的角度

CD-Net 本质是 learn a Mahalanobis metric in a transformed feature space。这跟 [deep metric learning](https://arxiv.org/abs/1905.01169) 文献思路相通，但 CD-Net 把 domain prior encode 进 architecture（multi-scale filter, scaling invariance, two-step structure），而不是纯 black-box learn。这种 "architecture as inductive bias" 的思路在 metric learning 里 under-explored。

---

## 我觉得 paper 的 limitation

1. **Subject pool 20 人，很可能都是中国人**。Color preference 是 culturally conditioned 的——[Luo et al. 的 cross-cultural color emotion study](https://onlinelibrary.wiley.com/doi/abs/10.1002/col.20091) 显示不同文化对 color 有不同 preference。SPCD 的 rating 可能 not globally representative。

2. **Display 是 LCD，没 test on OLED**。现在主流旗舰手机都是 OLED，color gamut、contrast、viewing angle 都不同。SPCD 的 rating 在 LCD 上收集，transfer 到 OLED 上的 perceptual CD 可能要重新 calibrate。

3. **Mahalanobis 是 global matrix**。不同 image region（sky vs. skin vs. grass）可能需要不同 metric。现在所有 pixel 共享一个 $S$ matrix，可能 suboptimal。可以考虑 spatially-adaptive Mahalanobis。

4. **没有 dense supervision**。Local CD map 是 emergent property，没有 ground truth。如果有一部分 pair 带 pixel-level CD annotation，CD map 可能更好。但 pixel-level annotation 极其 expensive。

5. **没 test on video**。Video CD 要考虑 temporal flicker 和 color consistency，这跟 single image 完全不同。SPCD 是纯 image dataset。

6. **HDR 场景没深入**。虽然用了 DCI-P3 和 Rec.2020，但没真正 HDR content。HDR 的 perceptual CD 可能需要 [Jzazbz](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-13-15131) 这种 HDR-tailored space 的 learned version。

---

## 总结：给你的 takeaway

如果只记一句话：**在 smartphone photography 时代，perceptual color difference 需要重新 define，而 lightweight network + right inductive bias + realistic dataset 是比 big CNN 或传统公式都好的 recipe。**

如果记三句话：
1. Traditional color difference formulas（CIEDE2000 等）在手机照片上 generalize 不好，因为它们是在 homogeneous patch 上 calibrate 的
2. Deep CNN 虽然 correlation 数字好，但 feature space geometry messed up，perceptual optimization 不好使，local CD map 质量烂
3. CD-Net 用 14K params + multi-scale filter + Mahalanobis distance + proper metric structure，达到 deep CNN 的 performance，同时 generalize 好、CD map 漂亮、是 proper metric

如果记一个 intuition：**Color perception 在 natural image 上是 spatially contextual 的、nonlinear 的、跟 image quality 不同 concept 的。Measure 它需要 spatial context + learnable perceptual space + metric structure guarantee。Lightweight + right bias > big + wrong bias。**

如果你后续想 chat 关于这个 paper 跟 perceptual loss for ISP optimization、或者跟 diffusion model evaluation 的 connection，我都很感兴趣。

---

# Measuring Perceptual Color Differences of Smartphone Photographs 深度讲解

你好 Andrej，这篇 paper 我读了之后觉得很有意思——它其实是把 color science 这个传统学科和 modern deep learning 做了一次很 lightweight 但又 principled 的结合。让我从 motivation、dataset、method、experiment 四个 angle 给你 build intuition。

---

## 1. Motivation: 为什么需要新的 color difference (CD) measure?

### 1.1 传统的 color difference 公式来自 homogeneous patches

CIE 在 1976 年推出了 CIELAB 和 CIELUV 两个 "perceptually uniform" color space，希望 Euclidean distance $\Delta E_{ab}^*$ 能直接对应人类感知到的颜色差异。但很快人们发现 CIELAB 并不真正 uniform，于是有了 CIE94、CIEDE2000 等非欧式修正公式。

**关键问题**：这些公式都是在 **homogeneous color patches**（4° 和 10° 视场、灰色背景）上 calibrate 的，比如 CIEDE2000 用的是 COM dataset（BFD-P、Leeds、Witt、RIT-DuPont 四个 patch dataset 加权组合）。

你可以想象一下，这些 patch 的 viewing condition 跟 smartphone photograph 完全不一样：
- patch 是均匀色块，没有 spatial context
- 自然图像中，人看 color sensation 时会 attend 到 homogeneous textured region，并在 larger spatial context 下比较 [Zhang & Wandell, S-CIELAB, 1997](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291520-6327%28199702%2940%3A1%3C61%3A%3AAID-JEDM6%3E3.0.CO%3B2-7)
- 现代 smartphone 的 ISP pipeline (demosaicing, AWB, color correction, tone mapping, gamma) 都是 highly nonlinear 的，会产生 patch 数据里根本不存在的 color distortion pattern

### 1.2 为什么以前没有大 scale 的 natural image CD dataset?

之前 CIE TC 8-02 委员会做过一些尝试，但 dataset 都很小，而且 distortion 都是 linear/quasi-linear color transform（比如在 CIELAB 维度上做 power function）。这显然 oversimplify 了 smartphone photography 的真实情况。

Liu et al. 2013 收集了 100 images (5 reference × 20 alternations)，已经是当时最大的之一了。Stokes 1991 只用了 6 reference images。Song & Luo 2000 只有 4 reference images。

### 1.3 Smartphone photography 的特殊性

看 Fig.1 你就能直观感受到：六台旗舰手机拍同一夜景 scene，structural detail 都差不多（ISP 已经很厉害了），但 color appearance 差异巨大。这正是这篇 paper 想要 measure 的东西。

所以 paper 的核心 thesis 是：**在 smartphone photography 时代，我们需要重新审视 CD measurement，既需要 realistic 的 dataset，也需要能 generalize 的 method。**

---

## 2. SPCD Dataset: 构造细节

### 2.1 四类 CD 来源

这是这篇 paper 最有价值的 contribution 之一。他们设计了四类 realistic 的 color distortion：

**Type I: Same scene, different smartphones**
- 6 台旗舰机: Apple iPhone 12 Pro, HUAWEI Mate40 Pro, OnePlus 7 Pro, Samsung S21 Ultra, OPPO Find X3 Pro, Xiaomi 11 Ultra
- 667 scenes × 6 phones = 4,002 images
- 因为不同手机的 camera system 和 ISP 都是 proprietary，所以 color appearance 天然不同
- 用 feature-based affine registration 对齐，但**故意保留轻微 misregistration**，用来测试 CD measure 的 robustness

**Type II: Photoshop 模拟 ISP function**
- 对同一张图用 Photoshop 调整 white balance, color correction, tone mapping, gamma correction 四个子模块的对应参数
- 这相当于模拟 ISP 的某些操作

**Type III: iPhone built-in filters**
- 9 个 filter: vivid, vivid warm, vivid cool, dramatic, dramatic warm, dramatic cool, mono, silver tone, noir
- 这模拟用户 post-processing 的 artistic style

**Type IV: Incorrect ICC color profiles**
- sRGB image 在 DCI-P3 gamut 显示器上显示，反之亦然
- 这是 color management system failure 的常见原因
- 参考 [DCI-P3 spec](https://developer.apple.com/documentation) 和 [Rec.2020](https://www.itu.int/rec/R-REC-BT.2020)

### 2.2 Dataset 统计

- 总共 15,335 images，从 1,000 distinct scenes 派生
- 30,000 image pairs（10,005 non-perfectly aligned Type I pairs + 19,995 perfectly aligned Type II/III/IV pairs）
- 所有图像 resize & crop 到 1024×1024，uncompressed 存储
- Content diversity: animal, plant, human, food, landscape, cityscape
- 背景、光照、天气、camera mode 都有 coverage

Fig.4 用 colorfulness vs. brightness 和 colorfulness vs. contrast 的 convex hull 显示了 SPCD 的分布广度，跟以往小 dataset 对比明显。

### 2.3 Psychophysical experiment: 600,000 ratings 怎么来的

这部分非常 rigorous，我详细讲：

**环境**：完全 dark 的室内办公室，no illumination, little reflection。

**显示设备**：两台 EIZO CG319X 31.1" LCD
- 分辨率 4096×2160
- 最大对比度 1500:1
- HDR + wide color gamut (WCG)
- Peak white 设为 100 cd/m²，D65 白点

**Display characterization**：他们做了非常彻底的 display calibration，包括：
- Temporal stability（冷启动后短期/中期 luminance 稳定性）
- Spatial independence（背景对中心灰块的影响）
- Spatial uniformity（不同位置 peak white 差异）
- Channel independence（RGB 三通道的 additivity 和 interactivity）
- Chromaticity constancy
- Color gamut

用 CIE 推荐的 GOG (gamma-offset-gain) display model，结合 JETI Specbos 1211uv tele-spectroradiometer 测量。最终在 Macbeth ColorChecker Chart 上验证，performance 达到 0.56 $\Delta E_{ab}^*$，非常适合 color vision 实验。

**Rating method**: Gray scale method（源自 textile 工业的 color fastness 评估）
- 5 个 grayscale sample pair 作为 reference，对应 grade level G ∈ {0, 1, 2, 3, 4}
- 推荐的 $\Delta E_{ab}^*$ 值是 0, 1.7, 3.4, 6.8, 13.6
- 实测值 (Table 1): 0.00, 1.83, 3.59, 6.45, 12.66

**Subjects**: 10 male + 10 female，Ishihara color vision test 筛查，viewing distance 1m 固定，每 30 分钟休息，每人每天 300 pairs，整个实验持续 4 个月。

### 2.4 Raw scores → perceptual CDs

这一步很关键。raw score 是 grayscale grade $G$，要转换成 $\Delta E_{ab}^*$ 单位。他们 fit 了一个 exponential function:

$$\Delta V = 1.6036 \cdot \exp(0.5391 \cdot G) - 1.2943$$

变量含义：
- $\Delta V$: predicted perceptual color difference (in $\Delta E_{ab}^*$ unit)
- $G$: grayscale grade level ∈ {0, 1, 2, 3, 4}
- 1.6036, 0.5391, 1.2943: fitted parameters

形式是 $a \exp(bG) + c$，general form 来自 ISO 105-A02 standard。

Table 1 显示 measured vs. predicted 的 max error 小于 0.37 $\Delta E_{ab}^*$，远低于 JND（大约 2.3），所以 conversion 很合理。

### 2.5 Outlier detection & reliability

- Outlier: 超过 3 std 的 rating
- Subject valid if outlier rate ≤ 5%
- 实际 outlier 只占 1.09%，所有 subject 都 valid

Reliability 验证 (Table 2)：把 20 subjects 随机 split 成两个 equal-size subgroup，计算它们的 mean CDs 之间的 STRESS、SRCC、PLCC。重复 100 次：
- STRESS median 18.75
- SRCC median 0.866
- PLCC median 0.869

这种 inter-subgroup consistency 说明数据 quality 很高。Fig.6 显示 30,000 CDs 的 histogram 是 unimodal，mode 在 3.5 附近。

---

## 3. CD-Net: 架构和 formulation

### 3.1 Design philosophy

CD-Net 的核心 insight 是：**传统 CD measure 都是 two-step——coordinate transform + distance calculation**。CD-Net 用 lightweight DNN 实现 coordinate transform，用 learnable Mahalanobis distance 实现 distance calculation，用 global average pooling 聚合 local CD 成 global score。

这样设计的好处：
1. 概念上 generalize 了 CIELAB（coordinate transform）+ CIEDE2000（weighted distance）
2. 参数少（14,464 + 78 = 14,542），不易 overfit
3. 在 transformed space 天然是 proper metric
4. 在 RGB space empirical 验证也满足 metric 性质

### 3.2 Coordinate transform: multi-scale filter bank

给定 RGB image $x \in \mathbb{R}^{H \times W \times 3}$，其中 $H, W$ 是 spatial height 和 width。

Front-end filter bank 用 $T$ 个不同 kernel size 的 convolution 做 multi-scale processing:

$$z_{p,q,r}^{(t)} = \sum_{(i,j) \in \mathcal{N}_t} \sum_{k=1}^{3} w_{i,j,k,r} \cdot x_{p+i, q+j, k}$$

变量和上下标：
- $z_{p,q,r}^{(t)}$: 第 $t$ 个 convolution 在位置 $(p,q)$、channel $r$ 上的输出
- $(p,q)$: spatial center location
- $r$: output channel index
- $\mathcal{N}_t$: 第 $t$ 个 convolution 的 neighboring grid（kernel size 决定）
- $(i,j)$: 相对 center 的 offset
- $k$: input channel index（1 到 3，对应 RGB）
- $w_{i,j,k,r}$: learnable convolution weight

然后 multi-scale response concatenation:

$$z = \text{concat}(z^{(1)}, z^{(2)}, \dots, z^{(T)})$$

后面接几层 $1 \times 1$ convolution + Leaky ReLU (negative slope $10^{-2}$)。**关键：no spatial downsampling**，所以 $f_\theta(x) \in \mathbb{R}^{H \times W \times C}$ 和 input 同 spatial size，目的是 preserve local detail for local CD map。

**重要 design choice: 去掉 bias term** 来 enforce scaling invariance: $f_\theta(\alpha x) = \alpha f_\theta(x)$。这跟 [Mohan et al., ICLR 2019](https://arxiv.org/abs/1906.07462) 的 bias-free CNN 思想类似，保证 coordinate transform 是 homogeneous function。

### 3.3 具体网络 specification

Fig.7 下半部分给出了 layer-by-layer 细节：

| Layer | Filter size | In channel | Out channel | Params |
|-------|-------------|------------|-------------|--------|
| Conv1 | 1×1 | 3 | 64 | 192 |
| Conv2 | 11×11 | 3 | 64 | 23,104 |
| Concat | - | - | 128 | - |
| Conv3 | 1×1 | 128 | 64 | 8,192 |
| LReLU | - | - | - | - |
| Conv4 | 1×1 | 64 | 32 | 2,048 |
| LReLU | - | - | - | - |
| Conv5 | 1×1 | 32 | 12 | 384 |
| LReLU | - | - | - | - |

Total: 14,464 params in $f_\theta$。

为什么 $T=2$ 且 kernel size 是 1 和 11？
- $1 \times 1$: pixel-wise processing，类似于 CIELAB 的 pointwise color space transform
- $11 \times 11$: patch-wise processing，capture spatial context，类似于 S-CIELAB 的 spatial low-pass filter
- 最后 channel $C=12$，这是 transformed feature space 维度

### 3.4 CD calculation: learnable Mahalanobis distance

在 transformed space，pixel-wise CD 用 Mahalanobis distance:

$$\Delta E(x_{ij}, y_{ij}) = \sqrt{(f(x)_{ij} - f(y)_{ij})^T \cdot S^{-1} \cdot (f(x)_{ij} - f(y)_{ij})}$$

变量和上下标：
- $\Delta E(x_{ij}, y_{ij})$: 位置 $(i,j)$ 处的 local color difference
- $f(x)_{ij}, f(y)_{ij}$: image $x$ 和 $y$ 在位置 $(i,j)$ 处的 transformed feature vectors（12 维）
- $S \in \mathbb{S}_+^C$: symmetric positive semi-definite 的 $C \times C$ matrix（12×12）
- $S^{-1}$: $S$ 的 inverse
- $(\cdot)^T$: transpose

$S$ 通过 Cholesky decomposition parameterize: $S = L L^T$，其中 $L$ 是 lower triangular matrix 且 diagonal entries 非负。这样自动保证 $S$ 是 PSD。$L$ 有 $C(C+1)/2 = 78$ 个参数。

**为什么 Mahalanobis？**
- 对角元素对应 per-channel scaling（generalize CIE94 的 weighting factors $S_L, S_C, S_H$）
- 非对角元素对应 channel 间的 cross-coupling（generalize CIEDE2000 的 hue rotation term $T$）
- 这跟 [Imai et al. 2001](https://onlinelibrary.wiley.com/doi/10.1117/1.1348339) 的 Mahalanobis color difference metric 思想一致

### 3.5 Global aggregation

$$\Delta E(x, y) = \frac{1}{HW} \sum_{i,j} \Delta E(x_{ij}, y_{ij})$$

- $H, W$: image spatial dimensions
- 求和 over all spatial positions
- 简单 mean aggregation

**Intuition**: 这里没用 weighted sum（比如 salience-based 或 ROI-based），保持简单。后续可以加 weighting，但 paper 想保持 lightweight 和 generalizable。

### 3.6 Training

Loss function (Eq.6):

$$\ell = \frac{1}{|B|} \sum_{i=1}^{|B|} \|\Delta E^{(i)} - \Delta V^{(i)}\|_2^2$$

- $|B|$: mini-batch size = 8
- $\Delta E^{(i)}$: CD-Net predicted CD for pair $i$
- $\Delta V^{(i)}$: ground-truth perceptual CD for pair $i$
- MSE loss

Optimizer: Adam, lr = $10^{-3}$, decay factor 2 每 50 epochs, 共 100 epochs。

Data split: 70% train / 10% val / 20% test, **content independent**（同一 scene 的所有 pair 不能跨 split）。Training 时 crop 到 768×768，testing 时 keep original size。

整个 procedure 重复 10 次，report mean。

---

## 4. Evaluation metrics

### 4.1 STRESS (Standardized Residual Sum of Squares)

$$\text{STRESS} = 100 \sqrt{\frac{\sum_{i=1}^{M} (\Delta E_i - F \cdot \Delta V_i)^2}{F^2 \sum_{i=1}^{M} \Delta V_i^2}}$$

$$F = \frac{\sum_{i=1}^{M} \Delta E_i^2}{\sum_{i=1}^{M} \Delta E_i \cdot \Delta V_i}$$

- $M$: test pair 数量
- $\Delta E_i$: model predicted CD
- $\Delta V_i$: ground-truth perceptual CD
- $F$: scale correction factor（消除 model 和 human 之间的 scale 差异）

STRESS 范围 0-100，越小越好。这是 [Garcia et al. 2007](https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-24-7-1823) 提出的，是 color science 社区的 standard metric。

### 4.2 PLCC (Pearson Linear Correlation Coefficient)

计算 PLCC 前先 fit 一个 4-parameter logistic (Eq.10) 来 linearize:

$$\Delta \hat{E} = \frac{\eta_1 - \eta_2}{1 + \exp(-(\Delta E - \eta_3)/|\eta_4|)} + \eta_2$$

- $\eta_1, \eta_2$: upper and lower asymptote
- $\eta_3$: mid-point
- $\eta_4: slope at mid-point$

然后 PLCC (Eq.9):

$$\text{PLCC} = \frac{\sum_{i=1}^{M} (\Delta E_i - \Delta \bar{E})(\Delta V_i - \Delta \bar{V})}{\sqrt{\sum_{i=1}^{M} (\Delta E_i - \Delta \bar{E})^2} \sqrt{\sum_{i=1}^{M} (\Delta V_i - \Delta \bar{V})^2}}$$

- $\Delta \bar{E}, \Delta \bar{V}$: mean values
- 范围 [-1, 1]，越大越好，measure prediction linearity

### 4.3 SRCC (Spearman Rank Correlation Coefficient)

$$\text{SRCC} = 1 - \frac{6 \sum_{i=1}^{M} d_i^2}{M(M^2-1)}$$

- $d_i$: 第 $i$ pair 在 $\Delta E$ 和 $\Delta V$ 中的 rank 差
- Measure prediction monotonicity

---

## 5. Main results: 33 baselines 大对比

Table 3 把 33 个 method 分成 5 类：

### 5.1 CIE-recommended 公式（homogeneous patches）

- CIELAB: STRESS 31.87, PLCC 0.716, SRCC 0.666 (all pairs)
- CIE94: STRESS 34.33
- CIEDE2000: STRESS 31.44, PLCC 0.726, SRCC 0.686
- CIECAM02: STRESS 33.40
- CIECAM16: STRESS 32.14
- Jzazbz: STRESS 32.76

**Observation**: CIEDE2000 最好，但 STRESS 仍然 31+，correlation 0.7 左右，说明这些公式对 smartphone 照片 generalize 不好。

### 5.2 Spatial extensions of CIELAB

- S-CIELAB: STRESS 32.78（反而比 CIEDE2000 差）
- Ouni08: STRESS 31.44（跟 CIEDE2000 持平）

**Surprising observation**: 加 spatial filtering 没明显 improvement。说明 naive spatial filtering 不够。

### 5.3 General-purpose image quality models

- SSIM: STRESS 48.03, PLCC 0.309（很差）
- VSI: STRESS 36.48
- PieAPP: STRESS 41.38
- LPIPS: STRESS 66.59（很差）
- DISTS: STRESS 52.41

**Why IQA models 表现差？** 因为 SPCD 里的 color alteration 不一定是 "visual distortion" 导致 quality degradation。比如 iPhone filter 出来的 vivid warm 跟原图 color 差异大，但 image quality 并没有变差。IQA models 学的是 distortion-quality 关系，不直接对应 CD。

### 5.4 JND measures

- Chou07: STRESS 49.55
- Lissner12: STRESS 41.45
- Butteraugli: STRESS 54.74

JND measures 设计来 measure threshold-level visibility，但 SPCD 里很多 suprathreshold CD，所以 fail。

### 5.5 Deep CNN backbones (作为 reference)

- VGG (14.7M params): STRESS 20.91
- ResNet-18 (11.2M params): STRESS 18.57, PLCC 0.876, SRCC 0.889 ← top performer
- UNet (31M params): STRESS 21.07
- CAN (37.6K params): STRESS 21.15

### 5.6 CD-Net

- STRESS 21.43, PLCC 0.846, SRCC 0.842 (all pairs)
- 14,542 params

**Key takeaway**: CD-Net 跟 14M-31M params 的 deep CNN 性能相当甚至略差一点（ResNet-18 最好），但 CD-Net 只有 14K params！而且后面的 generalization 和 metric property 实验说明 deep CNN 严重 overfit。

### 5.7 Perfectly aligned vs. non-perfectly aligned

Table 3 分了两列。几乎所有 method 在 perfectly aligned pairs 上都比 non-perfectly aligned 上好。这说明**轻微 misregistration 是 CD measure 的 killer**。CD-Net 在 perfectly aligned 上 STRESS 21.43，non-perfectly aligned 上差不多——说明 11×11 kernel 提供了很好的 misregistration robustness。

---

## 6. Qualitative: local CD maps

Fig.9 展示了几个 method 的 local CD map。这是 build intuition 的好地方：

- **CIEDE2000**: 噪声多，因为是 pixel-wise 比较，没有 spatial smoothness
- **S-CIELAB**: 跟 CIEDE2000 差不多，spatial filtering 没明显改善
- **Lee05**: 16×16 non-overlapping block，blocky artifact，sky 里 capture 不到 local variation
- **FLIP**: 过度强调 salient edges（sky 和 building 边界）—— FLIP 本来就是为 rendering 评估设计的，对 edge 敏感
- **ResNet-18**: CD map 看起来 noisy
- **UNet**: ringing artifacts
- **CAN**: 也有问题
- **CD-Net**: smooth transitions across strong edges，符合 human perception

这是 paper 的一个重要 argument：**STRESS 数字接近的 method，CD map 质量 can differ dramatically**。Deep CNN 虽然 number 好，但 map 烂，说明 overfit 到 global score 而不是真正学了 perceptual rule。

---

## 7. Ablation studies

### 7.1 Front-end filter bank (Table 4)

| Setting | STRESS | PLCC | SRCC |
|---------|--------|------|------|
| Only 1×1 | 24.76 | 0.785 | 0.769 |
| Only 11×11 | 21.93 | 0.838 | 0.838 |
| 1×1 + 11×11 (default) | 21.43 | 0.846 | 0.842 |

**Intuition**: 
- 1×1 only ≈ pixel-wise，类似 CIELAB，最差
- 11×11 only ≈ patch-wise，好很多，说明 spatial context 重要
- 两者 concatenate 最好，因为 pixel-wise 和 patch-wise 信息 complementary

### 7.2 Last conv channel $C$ (Table 5)

| $C$ | STRESS | PLCC | SRCC |
|-----|--------|------|------|
| 3 | 22.29 | 0.843 | 0.843 |
| 6 | 22.16 | 0.842 | 0.841 |
| 9 | 21.50 | 0.844 | 0.842 |
| 12 | 21.43 | 0.846 | 0.842 |
| 15 | 21.28 | 0.845 | 0.842 |
| 18 | 21.26 | 0.845 | 0.842 |

**Intuition**: 性能对 $C$ 不敏感。$C=3$（跟 RGB 同维）已经不错，$C \geq 12$ 基本 saturate。说明 transformed space 维度不需要太高就能 capture perceptual uniformity。

### 7.3 Image size discrepancy (Table 6)

Training 和 testing 用不同 size 的对比。Key finding:
- 测试 size 越大，性能略降（misalignment 在 high res 更明显）
- Training size 越大，性能越好
- 推荐 train 和 test 都用 high res

---

## 8. Generalization experiments

### 8.1 Unseen alternations (Table 7, TID2013 color subset)

测试 3 种 SPCD 没见过的 distortion: quantization noise, color quantization with dither, chromatic aberration。

CD-Net: STRESS 15.47, PLCC 0.814, SRCC 0.813

**比 pixel-wise CD metric 和 spatial extension 好，但比 LPIPS (14.16) 和 FLIP (12.61) 差**。Paper 解释这是因为 LPIPS 和 FLIP 在 ImageNet 上 pretrain 时可能见过类似 color appearance。这暴露了 CD-Net 的一个 limitation: training data 多样性不够。

### 8.2 Homogeneous color patch data (Table 8, COM dataset)

这是最 striking 的 generalization test：CD-Net 在 natural image 上 train，从没见过 patch data，然后在 COM 上 test。

| Method | COM STRESS | COM PLCC |
|--------|-----------|----------|
| CIELAB | 45.20 | 0.693 |
| CIEDE2000 | 28.98 | 0.862 |
| CD-Net | 38.87 | 0.786 |

**CD-Net 比 CIELAB 好**，说明 CD-Net 学到的 12 维 feature space 比 CIELAB 更 perceptually uniform——这是 surprising 的发现，因为 CIELAB 是几十年 color science 的结晶。

当然 CIEDE2000 最好，因为它就是 fit COM dataset 出来的。

**重要 insight**: 12 维 learned feature space > 3 维 CIELAB，证明从 natural image 直接 learn perceptual space 是 promising direction。

### 8.3 Empirical verification as proper metric

这是 paper 的一个数学严谨性 contribution。一个 proper metric 要满足四个性质：
1. Non-negativity: $\Delta E(x, y) \geq 0$
2. Symmetry: $\Delta E(x, y) = \Delta E(y, x)$
3. Identity of indiscernibles: $\Delta E(x, y) = 0 \iff x = y$
4. Triangle inequality: $\Delta E(x, y) + \Delta E(y, z) \geq \Delta E(x, z)$

性质 1 和 2 from Mahalanobis definition 自动满足。性质 3 和 4 需要 empirical 验证，因为 $f_\theta$ 的 invertibility 是 open problem。

**Identity of indiscernibles (reference image recovery, Fig.10)**:

给定 reference $x$ 和 initial $y$（Gaussian noise 或 tone-altered），solve:

$$y^* = \arg\min_y \Delta E(x, y)$$

- CD-Net: 从 noise 和 tone-altered 都能 recover 出接近 $x$ 的图像，最终 CD 远低于 JND
- VGG, ResNet-18, UNet, CAN: 都 fail，recovery 出完全不同的图像，artifact 严重

**Intuition**: deep CNN 的 feature space 有很多 local minima 和 flat regions，优化时容易掉进去。CD-Net 的 lightweight 结构 + Mahalanobis distance 更 well-behaved。

**Triangle inequality**:

测试了约 2 million image triplets。只有 VGG 在 13 个 triplet 上 violate。CD-Net 和其他三个 CNN-based model 都没 counterexample。所以 CD-Net empirical 上是 proper metric。

---

## 9. 我的整体 assessment 和联想

### 9.1 这篇 paper 真正的 contribution

1. **SPCD dataset**：30,000 pairs, 600,000 ratings, 4 months experiment, dual EIZO display with full colorimetric characterization。这是 color science 和 computer vision 之间很久以来缺少的 bridge。Dataset 已经 [public](https://github.com/hellooks/CDNet)。

2. **CD-Net 的 design philosophy**：用 lightweight DNN 实现 coordinate transform，但保留传统 CD measure 的 two-step 结构（transform + distance）。这比直接 throw 一个 ResNet 进去更 principled，也更 interpretable。

3. **Empirical metric verification**：reference image recovery 这个 test 很 elegant，能同时验证 identity of indiscernibles 和 capture overfitting。

### 9.2 跟你的工作可能相关的联想

**跟 LPIPS 的对比**：LPIPS [Zhang et al. 2018](https://arxiv.org/abs/1801.03924) 也是 learn perceptual metric，但用 deep network (AlexNet/VGG) 和大量 ImageNet pretraining。CD-Net 用 14K params + 30K pairs 就达到 competitive performance。这有点像 "small data + right inductive bias" vs "big data + big model" 的对比。

**跟 S-CIELAB 的关系**：S-CIELAB 加 spatial low-pass filter。CD-Net 用 11×11 learnable conv，相当于 learnable version 的 S-CIELAB filter。Ablation table 4 也印证了 spatial filtering 重要。

**跟 modern diffusion-based image generation 的联系**：现在 diffusion model 生成的图像，perceptual quality 评估仍然很 hard。CD-Net 这种 lightweight、proper metric 的方法可能可以用于 diffusion training 的 perceptual loss。Reference image recovery 实验其实就有点像 reverse optimization。

**HDR 和 wide gamut 场景**：paper 用了 DCI-P3 和 Rec.2020，但没深入 test HDR。未来工作可以把 SPCD 扩展到 HDR scene，对应 [Jzazbz](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-13-15131) 这种 HDR-tailored color space 的 learned version。

**跟 computational photography 的联系**：[Mobile computational photography: A tour](https://www.annualreviews.org/doi/10.1146/annurev-vision-091521-043055) 这篇 review 也是 same vibe。CD measure 可以作为 ISP pipeline 优化的 loss function。这也是 paper 在 introduction 提到的 motivation。

**Self-supervised learning 可能性**：CD-Net 是 supervised on 30K pairs。但 color difference 本质是 pairwise comparison，可能可以用 contrastive learning 思路 self-supervised pretrain on unlabeled smartphone photos。

**Metric learning 的角度**：CD-Net 本质是 learn a Mahalanobis metric in a transformed feature space。这跟 deep metric learning 文献（[Kaya & Bilge, 2019](https://arxiv.org/abs/1905.01169) review）思路相通，但 paper 把 prior knowledge (color space structure, multi-scale processing) encode 进 architecture，是 useful inductive bias。

### 9.3 Limitations

Paper 自己也提到了几个：
1. 没用 dense supervision（local CD map 的 ground truth）来训练，所以 local CD map 是 emergent property
2. TID2013 generalization 不如 LPIPS/FLIP，说明训练 distribution 不够广
3. Affine registration 只能处理 global geometric transform，对 local misalignment 不 robust

我觉得还有几个 implicit limitations:
1. Subject pool 只有 20 人，cultural background 单一（很可能都是中国 subjects），color preference 是 culturally conditioned 的
2. Display 是 LCD，没 test on OLED（现在主流旗舰手机都是 OLED）
3. 没考虑 viewing condition variation（户外强光下手机屏幕的 color perception 跟 dark room 完全不同）
4. Mahalanobis 是 global matrix，没考虑 local adaptivity（不同 image region 可能需要不同 metric）

### 9.4 一些可以继续探索的方向

1. **Lightweight + inductive bias + small data** 这个 recipe 可以 apply 到其他 perceptual measurement task，比如 tone mapping quality, HDR naturalness, etc.
2. **CD-Net 作 perceptual loss**：在 ISP pipeline optimization, image generation, color transfer 里用 CD-Net 作 loss
3. **Active learning**: 让 CD-Net 主动选 informative pair 来 query human rating，可能减少 human experiment cost
4. **Cross-device generalization**: train 在一种 display，test 在另一种 display
5. **Temporal extension**: video CD measure，考虑 temporal flicker 和 color consistency

### 9.5 一些可以参考的 related resources

- [S-CIELAB original paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291520-6327%28199702%2940%3A1%3C61%3A%3AAID-JEDM6%3E3.0.CO%3B2-7)
- [CIEDE2000 paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/1520-6378(200108)26:5%3C340::AID-COL6%3E3.0.CO;2-7)
- [LPIPS paper](https://arxiv.org/abs/1801.03924)
- [FLIP paper (Andersson et al. 2020)](https://research.nvidia.com/publication/2020-07_FLIP)
- [DISTS paper](https://arxiv.org/abs/2004.07728)
- [Mobile computational photography review](https://www.annualreviews.org/doi/10.1146/annurev-vision-091521-043055)
- [Jzazbz HDR color space](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-13-15131)
- [TID2013 dataset](https://www.sciencedirect.com/science/article/pii/S0923596514001474)
- [Kede Ma's homepage](https://www.cse.ust.hk/~kede/) - senior author
- [CDNet github repo](https://github.com/hellooks/CDNet)
- [DXOMARK camera test protocol](https://www.dxomark.com/cameras-testing-protocol/)
- [ISO 105-A02 gray scale standard](https://www.iso.org/standard/3787.html)
- [ITU-R BT.500 subjective assessment methodology](https://www.itu.int/rec/R-REC-BT.500)
- [Bias-free CNN denoising (Mohan et al.)](https://arxiv.org/abs/1906.07462) - scaling invariance 灵感来源

---

## 10. 总结：build your intuition

如果让我用一句话总结这篇 paper 的 insight 给你：

**Color perception 在 natural image 上跟 homogeneous patch 完全不同，需要 spatial context + learnable perceptual space + proper metric structure 来 measure，而 lightweight network + right inductive bias 可以达到甚至超过 heavyweight CNN，同时还能 generalize 和 behave as proper mathematical metric。**

更具体的 intuition：
1. **Spatial context is critical**: 1×1 convolution 远不如 1×1 + 11×11。Human perceive color 在 context 里。
2. **Higher-dim learned space > 3-dim CIELAB**: 12 维 learned feature space 在 patch data 上比 CIELAB 还好，说明 perceptual uniformity 可以直接 from data 学。
3. **Mahalanobis > Euclidean**: 学习一个 PSD matrix 比 fix Euclidean distance 灵活，且能 generalize CIE94/CIEDE2000 的 weighting 和 hue rotation。
4. **Lightweight > heavyweight for proper metric**: 14K params 的 CD-Net 满足 metric 性质，14M params 的 VGG 不满足。Reference image recovery 是 beautiful test。
5. **Realistic distortion matters**: SPCD 的 4 类 distortion 都是 smartphone photography 真实场景，比之前 linear color transform 真实得多。

希望这个讲解对你的 intuition building 有帮助。CD-Net 这种 design pattern——**先 encode domain knowledge 到 architecture，再用小 model 学剩余的 freedom——其实在 modern deep learning 时代是一个被 under-appreciated 的方向，跟 general-purpose large model 形成有趣对照。**

如果你之后想讨论这个 paper 跟 modern diffusion-based image generation 的 perceptual loss、或者跟 contrastive learning 的 metric learning 路线的 connection，可以继续聊。
