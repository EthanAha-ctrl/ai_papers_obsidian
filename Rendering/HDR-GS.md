---
source_pdf: HDR-GS.pdf
paper_sha256: 687dd3852b44de808b7a70dace9d6ecf9702b042c62cc5ab480b644409287ef1
processed_at: '2026-08-04T23:36:03-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HDR-GS 人话版：用大白话讲清楚这篇 paper

## 一、这 paper 在搞什么？

想象你拿手机拍窗外的风景。窗户里阳光灿烂，房间角落暗得看不清。你拍一张照片——要么窗户白花花一片过曝，要么房间角落黑乎乎一片欠曝。因为 camera sensor 的 dynamic range 太窄，记不下这么大的明暗差距。

**HDR 的梦想**：把不同曝光时间拍的几张照片融合起来，还原出真实世界那从极暗到极亮的所有细节。这事儿在 2D 摄影 (手机 HDR 模式) 已经做烂了。问题升级到 3D：给你一堆不同视角、不同曝光的照片，想从任意新视角渲染出 HDR 画面——这就叫 **HDR Novel View Synthesis**。

以前的人 (HDR-NeRF) 用 NeRF 做这事儿，train 9 小时，渲染一张图要 8 秒。完全没法实时用。这篇 paper 说：我用 3D Gaussian Splatting 改造一下，train 只要 34 分钟，推理快 1000 倍，质量还更好。

---

## 二、为什么 NeRF 慢得让人抓狂

NeRF 的思路是：在 3D 空间里，每个位置都 query 一个 MLP，问它"这个点的 density 是多少？color 是什么？"。

渲染一个 pixel，要从 camera 发射一条 ray，沿 ray 采样 100 个 3D 点，每个点都跑一遍 MLP。一张 400×400 的图有 16 万 pixel，每个 pixel 100 个采样点，总共要 query MLP 1600 万次。MLP forward 本身就慢，乘上这个数量级，慢得让人想砸键盘。

参考 NeRF 原文 https://arxiv.org/abs/2003.08934 理解这个 bottleneck。

3DGS 的思路完全不同：直接在 3D 空间撒一堆 Gaussian point (每个是个小椭球，带 color 和 opacity)，然后用 GPU 的 tile-based rasterization 一次性把所有 Gaussian 投影到屏幕上做 alpha blending。这是图形学的 standard pipeline，GPU 硬件极其擅长，跑 100+ fps 轻轻松松。

3DGS 原文在 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 三、直接用 3DGS 搞 HDR 会撞三个墙

**墙 1：color 被压死了**

原始 3DGS 的 color 公式长这样：
$$c = \text{sigmoid}(\text{SH}(\mathbf{d}, \mathbf{k}))$$

sigmoid 把输出死死压在 [0, 1]，再乘 255 就是 [0, 255]。HDR 的物理意义是 radiance 可以从 $10^{-3}$ 到 $10^6$，跨越 9 个数量级。sigmoid 直接把这事儿物理上 impossible 了。

**墙 2：不同曝光训练会崩**

假设同一个 3D 点，在 view A 下你拍了短曝光 (暗)，在 view B 下你拍了长曝光 (亮)。3DGS 的 SH coefficients 只能建模 view-dependent color (view direction 变，color 变)，完全不知道"曝光"这个概念。它拼命想用 SH 同时拟合暗的和亮的，结果学出个四不像，color distortion + blur 全来了。

更糟的是 3DGS 的 adaptive density control 会发现"哎这个区域 color variance 好大，得 split 更多 Gaussian 去拟合"，于是 Gaussian 数量爆炸，train 越来越慢，还越来越烂。

**墙 3：推理时调不了曝光**

3DGS 输出的 image 是固定 appearance，没有 exposure time 这个 input。你训练完想渲染一个不同曝光的画面？做不到。但 AR/VR/film 里用户就是要动态调光照氛围的。

---

## 四、HDR-GS 的三个 trick

Paper 的核心就是三个 trick，每个对应解决一个墙。

### Trick 1：用 exp 替代 sigmoid 释放 HDR range

公式 (7)：
$$c_i^h(\mathbf{d}, \mathbf{k}) = \exp\left(\sum_{l=0}^{L} \sum_{m=-l}^{l} k_l^m Y_l^m(\theta, \phi)\right)$$

逐项说：
- $\mathbf{d} = (\theta, \phi)$ 是 view direction，球坐标两个角度
- $k_l^m$ 是 SH coefficients，$l$ 是 degree (0 到 L)，$m$ 是 order (-l 到 l)，每对 $(l, m)$ 对应一个 basis function，每个 coefficient 是 RGB 三通道的 vector
- $Y_l^m(\theta, \phi)$ 是 real spherical harmonic basis，定义在 unit sphere 上的函数
- $L$ 是 SH degree，通常取 3，总共 $(L+1)^2 = 16$ 个 coefficients

把 sigmoid 换成 exp，输出范围从 [0, 1] 变成 $[0, +\infty)$，物理上对应 radiance 非负且无上限，这就是 HDR。

为什么用 exp 还有个好处：保证非负。物理上 radiance 不可能是负数，exp 天然满足。NeRF 里 density 也用 $\exp(\cdot)$ 保证非负，套路一样。

Spherical harmonics 背景知识在 https://en.wikipedia.org/wiki/Spherical_harmonics

### Trick 2：Log-domain Tone Mapping 解决训练崩溃

这是整篇 paper 最精妙的地方。

物理上，LDR pixel value 来自 HDR radiance 乘以 exposure time 再经过 camera response function：
$$c_i^l = f_{TM}(c_i^h \cdot \Delta t)$$

- $c_i^l$ 是 LDR color (你训练时看到的 image pixel)
- $c_i^h$ 是 HDR color (你想恢复的物理量)
- $\Delta t$ 是 exposure time (从照片 metadata 读)
- $f_{TM}$ 是 Camera Response Function (CRF)，把 sensor exposure 映射到 pixel value

Naive 做法：直接用 MLP 学 $f_{TM}$。但会崩，因为 $c_i^h \cdot \Delta t$ 这个乘积数值范围乱跳——$c_i^h$ 可能是 50000，$\Delta t$ 可能是 0.001，乘积是 50；下一秒 $c_i^h$ 还是 50000 但 $\Delta t$ 变成 32，乘积变成 160 万。MLP 输入跨度太大，sigmoid 输出要么全 saturate 到 0 要么全 saturate 到 1，gradient 消失，学不动。

**Paper 的解法**：套用 Debevec & Malik 1997 的经典 HDR calibration trick，把整个关系转到 log domain。

先对物理公式两边取逆再取 log：
$$\log f_{TM}^{-1}(c_i^l) = \log c_i^h + \log \Delta t$$

- $f_{TM}^{-1}$ 是 CRF 的逆函数 (从 LDR 反推 sensor exposure)
- $\log$ 是 natural log，base $e = 2.71828...$

关键：**乘法变成加法了**。$c_i^h \cdot \Delta t$ 的 log 是 $\log c_i^h + \log \Delta t$。

再定义 $g_\theta(x) \triangleq (\log f_{TM}^{-1})^{-1}(x)$，即对数域逆 CRF 的逆函数 (听着绕，其实就是把 log-domain 的 CRF 用 MLP 参数化)。最终公式 (6)：
$$c_i^l = g_\theta(\log c_i^h + \log \Delta t)$$

现在 MLP 的 input 是 $\log c_i^h + \log \Delta t$，是个加法。$\log c_i^h$ 把 HDR 压缩到比如 [-10, 10] 范围，$\log \Delta t$ 是个常数 shift (每张图固定)。MLP 学的是 $g_\theta$ 的 shape，即 CRF curve 本身，跟 exposure time 解耦了。换不同 exposure 只是 input 加个常数，MLP 自动泛化。

消融实验 (Table 3b) 显示：log domain 比 linear domain 高 **12.13 dB** (HDR PSNR)，这数据说明 trick 的价值有多大。

MLP 结构很简单 (Figure 3b)：对 RGB 三通道各一个独立 MLP，每个 MLP 是 FC → ReLU → FC → Sigmoid。三通道独立是因为 sensor 对红绿蓝三种波长的响应曲线不同，物理上就该分开建模。

Debevec 原文 https://www.pauldebevec.com/Research/HDR/

### Trick 3：两路 Parallel Rasterization

公式 (9)：
$$I^h = F_{HDR}(\mathbf{M}_{int}, \mathbf{M}_{ext}, \{\mu_i, \Sigma_i, \alpha_i, c_i^h\}_{i=1}^{N_p})$$
$$I^l(\Delta t) = F_{LDR}(\mathbf{M}_{int}, \mathbf{M}_{ext}, \{\mu_i, \Sigma_i, \alpha_i, c_i^l(\Delta t)\}_{i=1}^{N_p})$$

- $I^h, I^l$ 是渲染出的 HDR 和 LDR image，shape 是 $H \times W \times 3$
- $\mathbf{M}_{int} \in \mathbb{R}^{3 \times 4}$ 是 camera intrinsic (焦距主点)
- $\mathbf{M}_{ext} \in \mathbb{R}^{4 \times 4}$ 是 camera extrinsic (旋转平移)
- $\mu_i, \Sigma_i, \alpha_i$ 是 Gaussian 的位置、协方差、opacity
- $N_p$ 是 Gaussian 数量

核心 design：**HDR 和 LDR 共享同一套几何** (同一组 $\mu, \Sigma, \alpha$)，只是 color 走两条路。HDR color 走 SH 那条，LDR color 走 tone-mapper 那条。两路 color 各自进一个 rasterization pipeline，渲染出 HDR image 和 LDR image。

这个 design 的物理 intuition 很强：同一个 3D 点的几何位置不随曝光变，只有 sensor 测量值变。所以 geometry 共享是对的，color 分开是对的。

具体 blending 公式 (13)：
$$I^h(p) = \sum_{j \in \mathcal{N}} c_j^h \sigma_j \prod_{k=1}^{j-1}(1 - \sigma_k)$$

- $p$ 是某个 pixel
- $\mathcal{N}$ 是覆盖这个 pixel 的 Gaussian 集合 (按深度排序)
- $c_j^h$ 是第 $j$ 个 Gaussian 的 HDR color
- $\sigma_j = \alpha_j \cdot P(\mathbf{x}_j | \mu_j, \Sigma_j)$ 是 effective opacity，等于 Gaussian 的 opacity 乘以它在该 pixel 位置的概率密度
- $\prod_{k=1}^{j-1}(1 - \sigma_k)$ 是 transmittance，光穿过前面 Gaussian 后剩余比例

这是 front-to-back alpha blending，标准图形学操作，只不过用两路 color 各算一次。

---

## 五、SfM Recalibration 这事儿看着小其实关键

HDR-NeRF 数据集给的是 NDC (Normalized Device Coordinate) 系下的 camera pose。NDC 把 3D 坐标压到 [-1, 1]。

NeRF 用 MLP 表示，对 coordinate scale 不敏感 (MLP 学个 mapping 而已)。但 3DGS 是显式 3D 表示，Gaussian 的 $\mu, \Sigma$ 直接在这个坐标系里。压到 [-1, 1] 意味着整个场景塞进一个单位立方体里，Gaussian 体积太小，scene 表示 capacity 严重不足，细节全丢了。

Paper 用 COLMAP 的 SfM 重新校准 camera 参数，公式 (14)：
$$\mathbf{M}_{int}, \{\mathbf{M}_{ext}^j\}_{j=1}^{N_v}, N_p, \{\mu_i\}_{i=1}^{N_p} = F_{SfM}(\{\hat{I}_j^l(t_s)\}_{j=1}^{N_v})$$

- $N_v$ 是 viewpoint 数
- $\hat{I}_j^l(t_s)$ 是第 $j$ 个视角在 exposure $t_s$ 下的 LDR image
- 输出 intrinsic、所有 extrinsic、SfM 3D points (给 Gaussian 初始化)

关键细节：SfM 必须用**同一曝光时间** $t_s$ 的所有 view。因为 SfM 靠 feature detection & matching，不同曝光下 feature appearance 差异大，matching 会失败。

消融实验 (Table 3d) 试了 5 个曝光时间做 SfM，$t_4 = 8$ 秒效果最好。intuition：太短曝光 (0.125s) 暗部全黑没 feature，太长曝光 (32s) 亮部 saturate 没 feature，中等偏长曝光 (8s) feature 最丰富。

消融 (Table 3a) 显示这个 recalibration 贡献 +2.27 dB，SfM points 初始化又贡献 +4.84 dB，加起来 +7.11 dB，相当可观。

COLMAP 在 https://colmap.github.io/

---

## 六、Loss Function 的设计 logic

公式 (17) 总 loss：
$$\mathcal{L} = \mathcal{L}_p + \gamma \cdot \mathcal{L}_c$$

- $\gamma$ 是 hyperparameter，合成场景 0.6，真实场景 0

**Photometric Loss $\mathcal{L}_p$** (公式 15)：
$$\mathcal{L}_p = \sum_{j=1}^{B} [\mathcal{L}_1(I_j^l, \hat{I}_j^l) + \lambda \cdot \mathcal{L}_{D-SSIM}(I_j^l, \hat{I}_j^l)]$$

- $B$ 是 batch size
- $\lambda$ 平衡 L1 和 D-SSIM
- 只在 LDR image 上算，因为 ground truth 就是 LDR image
- L1 对 outlier robust，D-SSIM 抓 structural 信息

**HDR Constraint Loss $\mathcal{L}_c$** (公式 16) 用 µ-law companding：
$$\mathcal{L}_c = \sum_{j=1}^{B} \left\| \frac{\log(1 + \mu \cdot \text{norm}(I_j^h))}{\log(1 + \mu)} - \frac{\log(1 + \mu \cdot \text{norm}(\hat{I}_j^h))}{\log(1 + \mu)} \right\|_2^2$$

- $\mu$ 是 compression amount，通常很大 (比如 5000)
- $\text{norm}(\cdot)$ 是 min-max normalization 到 [0, 1]
- $I_j^h, \hat{I}_j^h$ 是 rendered 和 ground truth HDR

为什么不能直接在 HDR domain 算 L2？因为 HDR 数值跨度太大，bright region 数值是 dark region 的几万倍，L2 loss 被 bright region 主导，dark region 几乎没 gradient，学不动。

µ-law 是 telecommunication 里的老 trick (电话音频编码用过)，$\log(1 + \mu x) / \log(1 + \mu)$ 把 wide range 压缩到 narrow range，让 bright 和 dark 在 loss 里权重平衡。合成场景有 GT HDR 所以用，真实场景没 GT HDR 所以 $\gamma = 0$。

µ-law 背景在 https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

---

## 七、实验数字告诉你什么

### 合成场景 (Table 1)

| Method | Train (min) | Infer (fps) | LDR-OE PSNR | LDR-NE PSNR | HDR PSNR |
|---|---|---|---|---|---|
| NeRF | 405 | 0.190 | 13.97 | 14.51 | - |
| 3DGS | 38 | 121 | 19.46 | 18.97 | - |
| NeRF-W | 437 | 0.178 | 29.83 | 29.22 | - |
| HDR-NeRF | 542 | 0.122 | 39.07 | 37.53 | 36.40 |
| **HDR-GS** | **34** | **126** | **41.10** | 36.33 | **38.31** |

看几个对比：

**vs HDR-NeRF**：
- 训练时间 542 → 34 分钟，只要 **6.3%**
- 推理速度 0.122 → 126 fps，**1033× 加速**
- LDR-OE +2.03 dB，HDR +1.91 dB
- LDR-NE 略低 1.20 dB (NE = novel exposure，训练没见过的曝光。MLP tone-mapper 在 unseen exposure 上泛化稍弱是可接受的)

**vs 原始 3DGS**：
- LDR-OE 高 **21.64 dB**！这数字夸张到说明 3DGS 直接做 HDR 根本不 work
- 推理还稍微快一点 (126 vs 121 fps)。Paper 解释：3DGS 在多曝光训练下崩溃，盲目 split Gaussian 拟合 exposure variance，Gaussian 数量爆炸反而拖慢；HDR-GS 正确建模 exposure，Gaussian 数量稳定

### 真实场景 (Table 2)

| Method | LDR-OE PSNR | LDR-NE PSNR |
|---|---|---|
| HDR-NeRF | 31.63 | 31.43 |
| **HDR-GS** | **35.47** | **31.66** |

真实场景 LDR-OE +3.84 dB，比合成场景 (+2.03) 提升更大。说明泛化能力强，不是只在 synthetic data 上过拟合。

### 关键消融 (Table 3a)

| 加什么 | LDR-OE PSNR | 增量 |
|---|---|---|
| Baseline (3DGS + NDC) | 12.35 | - |
| + Camera Recalibration | 14.62 | +2.27 |
| + SfM Points | 19.46 | +4.84 |
| + DDR Model | 41.10 | +21.64 |

DDR Model 贡献 +21.64 dB，是绝对主力。Camera recalibration 和 SfM points 加起来 +7.11 dB，也不小。三个 trick 缺一不可。

---

## 八、我的 intuition 和联想

### 1. Log-domain 是关键 insight

这个 paper 最值得记住的 insight：**当物理关系是乘法且数值跨度大时，转 log domain 让 MLP 学加法**。这个 trick 不只适用于 HDR，任何 multiplicative relationship 都适用：
- Optics: $E = L \cdot \Delta t$
- Acoustics: intensity 跨多个数量级
- 金融: 复利
- 任何 power law: $y = x^k$ 转 log 变 $\log y = k \log x$

NeRF 里的 density 用 $\exp(\cdot)$ 也是类似思路 (保证非负 + 释放数值范围)。HDR-GS 的 color 用 $\exp(\text{SH})$ 是同源思想。

### 2. Disentanglement 通过 input design 实现

HDR-GS 没有显式 disentanglement loss，但通过 input design ($\log c^h + \log \Delta t$) 让 MLP 物理上不可能不 disentangle。这比加 contrastive loss 之类的软约束强得多。

NeRF-W 用 per-image appearance embedding 做 disentanglement，但 embedding 是 latent 无物理意义。HDR-GS 的 exposure time 是 scalar 有明确物理含义 (从 metadata 读)，可解释性好得多。NeRF-W 在 https://nerf-w.github.io/

### 3. Shared Geometry 是 physics-motivated

两路 rasterization 共享 $\mu, \Sigma, \alpha$ 这事儿物理上极正确：3D 点的几何不随 exposure 变，只有 sensor measurement 变。

这个 idea 可以推广到其他 appearance variation 场景：
- 不同光照条件下，geometry 不变，material 变
- 不同 weather 下，geometry 不变，atmospheric effect 变
- 不同 time of day，geometry 不变，sun direction 变

GS-IR [Liang et al. 2023] 做逆渲染也是这个思路 (geometry 和 material 分离)。https://arxiv.org/abs/2311.16473

### 4. SfM Recalibration 是被低估的 contribution

很多人忽略这点，但实际做 3DGS 项目时这是常见的坑：NeRF 时代的 dataset (NDC 坐标系) 直接拿来用 3DGS 不行，必须 recalibrate。

这个经验可以推广：任何 NeRF → 3DGS 的迁移工作，第一步都是检查 coordinate system 和 SfM initialization。HDR-GS 这篇 paper 实际上为后续 3DGS-based HDR 研究铺了 data foundation。

### 5. 跟 4DGS / 动态场景的结合

HDR-GS 假设场景静态。结合 4D Gaussian Splatting [Wu et al. 2023] 处理动态场景，每个 Gaussian 加 time dimension，可以做 video HDR NVS。这是个很自然的 next step。

4DGS 在 https://github.com/hustvl/4DGaussians

### 6. 跟 SLAM 的结合

Gaussian-SLAM [Matsuki et al. 2023] 用 3DGS 做 SLAM 的 map representation。结合 HDR-GS 可以做 HDR-aware SLAM，在极端光照环境下 (比如隧道进出口) 也能稳定重建。这对 autonomous driving 很有价值。

Gaussian-SLAM 在 https://arxiv.org/abs/2312.06741

### 7. Mobile Deployment 的挑战

Paper 自己提的 limitation：3DGS memory usage 大 (每个 Gaussian 存位置、协方差、opacity、SH coefficients、MLP params)。百万级 Gaussian 时 VRAM 消耗显著，mobile 部署困难。

可能解决方向：
- MobileNeRF [Chen et al. CVPR 2023] 用 polygon rasterization 替代 alpha blending，思路可以借鉴
- Compression：SH coefficients 可以量化或低秩近似
- Pruning：基于 contribution 的 Gaussian pruning

MobileNeRF 在 https://github.com/google-research/mobilenerf

### 8. Sparse-view 场景

Paper 用 18 个 view 训练。如果只有 3-5 个 view 怎么办？SparseNeRF [Wang et al. ICCV 2023] 用 depth ranking distillation 解决 few-shot NVS。结合 HDR-GS 可以做 sparse-view HDR NVS，对实际拍摄很有用 (拍 HDR 多曝光本来就麻烦，view 越少越好)。

SparseNeRF 在 https://github.com/WangGWHU/SparseNeRF

### 9. 生成式 HDR

DreamGaussian [Tang et al. 2023] 做 text-to-3D 生成。结合 HDR-GS 可以做 text-to-HDR-3D，生成在极端光照下也好看的 3D 资产。对 game 和 film 行业很有价值。

DreamGaussian 在 https://github.com/dreamgaussian/dreamgaussian

### 10. Tone Mapping 方向的混淆要注意

HDR 领域有两种 tone mapping，方向相反：
- **Forward tone mapping (CRF)**：从 HDR radiance 到 LDR pixel value，是 sensor 的物理过程。HDR-GS 学的就是这个
- **Backward tone mapping (Display)**：从 HDR image 到 LDR display，因为大多数显示器只能显示 LDR。Reinhard 的经典 tone mapping 是这个方向

Reinhard tone mapping 在 https://www.cs.tut.fi/~gerk/publ/reinhard02-tmo.pdf

Paper 里用 Photomatix Pro 可视化 HDR 时用的是 backward tone mapping。两个方向别搞混。

---

## 九、一句话人话总结

**把"exposure = radiance × 时间"这个乘法物理关系，两边取 log 变成加法，让 neural network 只需要学一个稳定的 camera response curve shape，不用每次 exposure 变就重新学；同时把 3D Gaussian 的 color 输出从 sigmoid 换成 exp 释放出 HDR 的数值范围，几何和颜色两路并行渲染，就实现了又快又好的 3D HDR 渲染**。

整个 paper 的核心就这一句话。其他都是工程细节和实验支撑。这个 log-domain 加 exp 的 idea 在物理建模的神经网络里有普适价值，值得记到 toolbox 里。

---

# HDR-GS：基于 Gaussian Splatting 的高动态范围新视角合成深度讲解

## 一、问题定位与 motivation

这篇 paper 要解决的 core problem 是 **3D HDR Novel View Synthesis** 的效率瓶颈。让我先从物理直觉出发构建你的 intuition。

### 1.1 LDR vs HDR 的本质差异

普通 camera sensor 的 dynamic range 限制在 [0, 255]（8-bit），原因在于 CMOS/CCD sensor 在每个 pixel 上累积的电荷量有上限。当场景中同时存在极暗区域（如阴影下的细节）和极亮区域（如直射光源、高光反射）时，sensor 会发生 saturation 或 under-exposure，导致信息丢失。

HDR imaging 的物理本质是：**场景的 scene radiance** $L$（单位 $W \cdot sr^{-1} \cdot m^{-2}$）理论上跨越多个数量级（从 $10^{-3}$ 到 $10^{6}$），而 sensor 经过 exposure time $\Delta t$ 后测量到的是 **sensor exposure** $E = L \cdot \Delta t$，再经过 Camera Response Function (CRF) 非线性映射到 LDR pixel value $\in [0, 255]$。

数学关系可以写成：
$$I_{LDR} = f_{CRF}(L \cdot \Delta t)$$

其中 $f_{CRF}$ 是单调递增但非线性（通常近似 sigmoid 或 gamma 函数）的相机响应函数。

### 1.2 现有方法的瓶颈

**NeRF-based 方法**（如 HDR-NeRF [Huang et al., CVPR 2022]）的 bottleneck 在于 ray-tracing scheme：

对每个 pixel 发射一条 ray，沿 ray 采样 $N$ 个 3D points，每个 point 都要 query 一个 MLP 计算 density $\sigma$ 和 color $c$，最后通过 volume rendering 积分。复杂度约 $O(H \times W \times N \times \text{MLP forward})$。

HDR-NeRF 的具体数字：
- 训练时间：**542 分钟**
- 推理速度：**0.122 fps**（渲染一张 400×400 图像需要 8.2 秒）

这种速度根本无法用于 AR/VR、游戏、autonomous driving 等实时场景。

**3DGS 的优势**：把 scene 表示成显式的 Gaussian point cloud，通过 tile-based parallel rasterization 在 GPU 上做 alpha blending，避免逐 ray 采样。原始 3DGS 在 LDR NVS 上能达到 121 fps。

### 1.3 直接用 3DGS 做 HDR 的三个难题

Paper 中明确指出三个 issues，这是理解整个方法设计的钥匙：

**Issue 1：Dynamic Range 限制**
原始 3DGS 的 color 直接用 SH 系数输出，经过 sigmoid 归一化到 [0, 1]，再缩放到 [0, 255]。这意味着输出被严格 cap 在 8-bit 范围，丢失了 HDR 物理意义。

**Issue 2：不同曝光训练不收敛**
当用不同曝光时间的 LDR 图像训练 3DGS 时，同一个 3D point 在不同 view 下被监督成不同的 color。但 SH 是 view-dependent 的，它假设 color 只随 view direction 变化，不随 exposure 变化。这导致 SH 无法拟合，引发严重的 color distortion 和 blur。

**Issue 3：无法控制 exposure**
3DGS 输出的 image 是固定的，没有 exposure time 这个 input，无法在推理时渲染不同曝光的 LDR view。这在 AR/VR/film 场景中是致命的——用户需要根据情绪、氛围调节光照。

---

## 二、核心方法：HDR-GS 架构解析

### 2.1 整体 Pipeline

参照 Figure 3，整个 pipeline 分三部分：

**(a) SfM Initialization & Camera Recalibration**
使用 COLMAP 的 SfM 算法 [Schönberger & Frahm, CVPR 2016] 重新计算 camera intrinsics、extrinsics，并产生初始 3D point cloud 作为 Gaussian 初始化。

**(b) Dual Dynamic Range (DDR) Gaussian Model**
每个 3D Gaussian 同时携带 HDR color（由 SH 建模）和 LDR color（由 MLP-based tone-mapper 从 HDR color + exposure time 生成）。

**(c) Parallel Differentiable Rasterization (PDR)**
HDR 和 LDR 两路 color 分别走两个独立的 rasterization pipeline，渲染出 HDR image 和可调曝光的 LDR image。

### 2.2 DDR Gaussian Point Cloud 形式化定义

公式 (1) 给出了整个 scene 表示：
$$\mathcal{G} = \{G_i(\mu_i, \Sigma_i, \alpha_i, c_i^l, c_i^h, \Delta t, \theta) \mid i = 1, 2, ..., N_p\}$$

逐项解释：
- $\mu_i \in \mathbb{R}^3$：第 $i$ 个 Gaussian 的中心位置
- $\Sigma_i \in \mathbb{R}^{3 \times 3}$：协方差矩阵，刻画 Gaussian 的形状（椭球）
- $\alpha_i \in \mathbb{R}$：opacity（不透明度），经过 sigmoid 后 ∈ [0, 1]
- $c_i^l \in \mathbb{R}^3$：LDR RGB color（由 tone-mapper 计算）
- $c_i^h \in \mathbb{R}^3$：HDR RGB color（由 SH 计算）
- $\Delta t \in \mathbb{R}$：exposure time（从图像 metadata 读取，每张训练图有一个）
- $\theta$：tone-mapper MLP 的参数（全局共享）
- $N_p$：3D Gaussian 的数量（动态变化，通过 splitting/pruning 控制）

公式 (2) 协方差分解：
$$\Sigma_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^\top \mathbf{R}_i^\top$$

这里 $\mathbf{R}_i \in \mathbb{R}^{3}$ 实际上是用四元数表示的 rotation，$\mathbf{S}_i \in \mathbb{R}^3$ 是 scaling vector。这种分解保证 $\Sigma_i$ 是 positive semi-definite，可学习且数值稳定。这种设计直接来自原始 3DGS [Kerbl et al., SIGGRAPH 2023]。

### 2.3 Tone Mapping：从 HDR 到 LDR 的核心创新

这是 paper 最精妙的部分，也是与 HDR-NeRF 的关键区别。

#### 2.3.1 直接建模的失败

最 naive 的方式是直接学一个 MLP 模拟 CRF：
$$c_i^l = f_{TM}(c_i^h \cdot \Delta t) \quad \text{(公式 3)}$$

直觉上：HDR color 乘以 exposure time 得到 sensor exposure，再过 CRF 得到 LDR pixel value。

但这样训练会失败，原因有二：
1. **Numerical overflow/underflow**：$c_i^h$ 是 HDR，可能非常大或非常小，乘以 $\Delta t$ 后范围更不可控，导致 sigmoid/sigmoid-like 输出 saturate，gradient 消失
2. **Input nonlinearity**：MLP 输入是 $c_i^h \cdot \Delta t$，这是个 multiplication，导致 input signal 在不同 exposure 下高度 nonlinear 和 discontinuous，MLP 难以拟合

#### 2.3.2 对数域变换（Debevec-Malik 思路）

Paper 借鉴了 Debevec & Malik 的经典 HDR calibration method [SIGGRAPH 1997] 的核心 trick：把 CRF 转换到 logarithmic domain。

从公式 (3) 反推：
$$\log f_{TM}^{-1}(c_i^l) = \log c_i^h + \log \Delta t \quad \text{(公式 4)}$$

这里 $f_{TM}^{-1}$ 是 CRF 的逆函数（从 LDR pixel value 反推 sensor exposure）。两边取对数后，**乘法变成加法**。

继续变形：
$$c_i^l = (\log f_{TM}^{-1})^{-1}(\log c_i^h + \log \Delta t) \quad \text{(公式 5)}$$

定义 $g_\theta(x) \triangleq (\log f_{TM}^{-1})^{-1}(x)$，即对数域逆 CRF 的逆，最终得到：
$$c_i^l = g_\theta(\log c_i^h + \log \Delta t) \quad \text{(公式 6)}$$

**Intuition**：现在 MLP 的 input 是 $\log c_i^h + \log \Delta t$，这是一个**加性组合**，数值范围被压缩到对数尺度（比如 $[-10, 10]$），MLP 容易拟合，gradient 稳定。同时，不同 exposure time 只是 input 的一个 constant shift，MLP 学到的是 shape of CRF，而非每个 exposure 的独立映射。

#### 2.3.3 SH 建模 HDR color

HDR color 必须保证非负（物理意义上 radiance 非负），所以不能直接用 SH 输出。

公式 (7)：
$$c_i^h(\mathbf{d}, \mathbf{k}) = \exp\left(\sum_{l=0}^{L} \sum_{m=-l}^{l} k_l^m Y_l^m(\theta, \phi)\right)$$

逐项解释：
- $\mathbf{d} = (\theta, \phi)$：view direction，球坐标
- $\mathbf{k} = \{k_l^m \mid 0 \leq l \leq L, -l \leq m \leq l\} \in \mathbb{R}^{(L+1)^2 \times 3}$：SH 系数，每个 $k_l^m \in \mathbb{R}^3$ 对应 RGB 三通道独立系数
- $L$：SH 的 degree（通常 3，对应 16 个 coefficients）
- $Y_l^m: \mathbb{S}^2 \to \mathbb{R}$：real spherical harmonic basis function，定义在 unit sphere 上
- $\exp(\cdot)$：保证输出非负，且把 SH 的 linear output 映射到 HDR 的 wide dynamic range

注意这里和原始 3DGS 的关键区别：原始 3DGS 用 $c = \text{sigmoid}(\text{SH output})$ 把 color 压缩到 [0, 1]；HDR-GS 用 $\exp(\text{SH output})$ 把 color 释放到 $[0, +\infty)$。

#### 2.3.4 完整 LDR color 公式

把公式 (7) 代入 (6)：
$$c_i^l(\mathbf{d}, \mathbf{k}, \Delta t) = g_\theta\left(\sum_{l=0}^{L} \sum_{m=-l}^{l} k_l^m Y_l^m(\theta, \phi) + \log \Delta t + b\right) \quad \text{(公式 8)}$$

新增的 $b \in \mathbb{R}$ 是一个 constant bias，用来 fine-tune SH function 的 offset，让 SH coefficients 更好拟合数据。

**Tone-mapper MLP 结构**（参照 Figure 3b）：
- 输入：$\log c_i^h + \log \Delta t$（标量，per channel）
- 对 RGB 三通道用三个**独立**的 MLP（因为 R/G/B 三个 channel 的 CRF 不同，物理上 sensor 对不同波长的响应不同）
- 每个 MLP：FC → ReLU → FC → Sigmoid
- 输出：LDR pixel value ∈ [0, 1]

三个 MLP 独立这点很重要——它建模了 sensor 的 spectral response 差异，符合 Debevec-Malik 经典 CRF 校准中 R/G/B 分通道处理的传统。

### 2.4 Parallel Differentiable Rasterization (PDR)

#### 2.4.1 两路渲染

公式 (9)：
$$I^h = F_{HDR}(\mathbf{M}_{int}, \mathbf{M}_{ext}, \{\mu_i, \Sigma_i, \alpha_i, c_i^h\}_{i=1}^{N_p})$$
$$I^l(\Delta t) = F_{LDR}(\mathbf{M}_{int}, \mathbf{M}_{ext}, \{\mu_i, \Sigma_i, \alpha_i, c_i^l(\Delta t)\}_{i=1}^{N_p})$$

变量解释：
- $I^h, I^l \in \mathbb{R}^{H \times W \times 3}$：渲染的 HDR/LDR image
- $\mathbf{M}_{int} \in \mathbb{R}^{3 \times 4}$：camera intrinsic matrix（焦距、主点等）
- $\mathbf{M}_{ext} \in \mathbb{R}^{4 \times 4}$：camera extrinsic matrix（rotation + translation）

两路 rasterization 共享几何参数（$\mu, \Sigma, \alpha$），只是 color 不同。这意味着 **Gaussian point cloud 的几何是统一的，HDR 和 LDR 共享同一个 3D 结构**，只差 color representation。

#### 2.4.2 Splatting 过程

公式 (10) Gaussian 在 3D 空间的概率密度：
$$P(\mathbf{x} | \mu_i, \Sigma_i) = \exp\left(-\frac{1}{2}(\mathbf{x} - \mu_i)^\top \Sigma_i^{-1} (\mathbf{x} - \mu_i)\right)$$

这是标准 multivariate Gaussian（未归一化，峰值是 1 而非 $\frac{1}{(2\pi)^{3/2}|\Sigma|^{1/2}}$）。

公式 (11) 投影：
$$\tilde{\mathbf{v}}_i = \mathbf{M}_{ext} \tilde{\mu}_i, \quad \tilde{\mathbf{u}}_i = \mathbf{M}_{int} \tilde{\mathbf{v}}_i$$

- $\mathbf{v}_i \in \mathbb{R}^3$：camera coordinate 下的 3D 位置
- $\mathbf{u}_i \in \mathbb{R}^2$：image plane 上的 2D 位置
- $\tilde{\cdot}$ 表示 homogeneous coordinate（加一维 1）

公式 (12) 协方差投影（EWA splatting [Zwicker et al. 2001] 的 standard formulation）：
$$\Sigma_i' = \mathbf{J}_i \mathbf{W}_i \Sigma_i \mathbf{W}_i^\top \mathbf{J}_i^\top$$

- $\mathbf{J}_i \in \mathbb{R}^{3 \times 3}$：projective transformation 的 Jacobian（affine approximation）
- $\mathbf{W}_i \in \mathbb{R}^{3 \times 3}$：viewing transformation，取 $\mathbf{M}_{ext}$ 的前 3 行 3 列

最终 2D 协方差 $\Sigma_i''$ 通过 skip 第三行第三列得到。

#### 2.4.3 Tile-based Alpha Blending

公式 (13) 是核心 blending 公式：
$$I^h(p) = \sum_{j \in \mathcal{N}} c_j^h \sigma_j \prod_{k=1}^{j-1}(1 - \sigma_k)$$
$$I^l(p | \Delta t) = \sum_{j \in \mathcal{N}} c_j^l(\Delta t) \sigma_j \prod_{k=1}^{j-1}(1 - \sigma_k)$$

变量解释：
- $\mathcal{N}$：覆盖 pixel $p$ 的 ordered Gaussian 集合（按深度排序）
- $\sigma_j = \alpha_j \cdot P(\mathbf{x}_j | \mu_j, \Sigma_j)$：effective opacity，即 Gaussian opacity 乘以 Gaussian 在该 pixel 处的概率密度
- $\mathbf{x}_j$：第 $j$ 个 Gaussian 在 ray 上的 intersection point
- $\prod_{k=1}^{j-1}(1 - \sigma_k)$：前面所有 Gaussian 的 transmittance，表示光线穿过前面 Gaussian 后剩余的能量比例

这是 front-to-back volumetric alpha blending，和原始 3DGS 完全一致，但用 HDR/LDR 两路 color。

**Intuition**：blending 时 HDR color 通过 opacity 加权平均，得到该 pixel 的 HDR radiance；LDR color 同理，但因为每个 Gaussian 的 $c_j^l$ 已经包含 exposure time 的作用，所以 LDR image 是可调曝光的。

### 2.5 SfM Recalibration：解决 Data Foundation 问题

这部分容易被忽略，但实际是 paper 的一个重要 contribution。

#### 2.5.1 NDC 不适用的原因

HDR-NeRF 数据集 [Huang et al. CVPR 2022] 只提供 NDC (Normalized Device Coordinate) 系下的 camera pose。NDC 把 3D 坐标归一化到 [-1, 1] 或 [0, 1]。

NDC 适合 NeRF 是因为 NeRF 用 MLP 隐式表示，对 coordinate scale 不敏感（MLP 可以学到任意 scale 的 mapping）。但 3DGS 是显式 3D 表示，对 coordinate scale 敏感：
1. Gaussian 的 $\mu, \Sigma$ 在 NDC 下被压缩到 [-1, 1]，导致 Gaussian 体积小，scene 表示 capacity 不足
2. NDC 是 screen space coordinate，丢失了真实 3D 几何关系，影响 projection

#### 2.5.2 SfM Recalibration

公式 (14)：
$$\mathbf{M}_{int}, \{\mathbf{M}_{ext}^j\}_{j=1}^{N_v}, N_p, \{\mu_i\}_{i=1}^{N_p} = F_{SfM}(\{\hat{I}_j^l(t_s)\}_{j=1}^{N_v})$$

- $N_v$：viewpoints 数量
- $\hat{I}_j^l(t_s)$：第 $j$ 个 viewpoint 在 exposure time $t_s$ 下的 LDR image
- 输出：intrinsic matrix、所有 viewpoint 的 extrinsic、SfM 3D points（用于 Gaussian 初始化）

**关键 trick**：用同一 exposure time $t_s$ 的所有 view 做 SfM。因为 SfM 依赖 feature detection & matching，不同曝光下 feature 的 appearance 差异大，matching 会失败。

消融实验（Table 3d）显示 $t_s = t_4 = 8$ 秒效果最好，因为：
- $t_4$ 是中等偏长的曝光，dark 和 bright region 都有 reasonable visibility
- 太短曝光（$t_1 = 0.125s$）暗部全黑，feature 缺失
- 太长曝光（$t_5 = 32s$）亮部 saturate，feature 也丢失

### 2.6 Loss Function

#### 2.6.1 Photometric Loss

公式 (15)：
$$\mathcal{L}_p = \sum_{j=1}^{B} \left[ \mathcal{L}_1(I_j^l(\Delta t_j), \hat{I}_j^l(\Delta t_j)) + \lambda \cdot \mathcal{L}_{D-SSIM}(I_j^l(\Delta t_j), \hat{I}_j^l(\Delta t_j)) \right]$$

- $B$：batch size
- $\lambda$：balance L1 和 D-SSIM 的 hyperparameter
- L1 loss 对 outlier robust
- D-SSIM 强调 structural similarity，对 perceptual quality 友好

注意这里只在 LDR image 上做 photometric loss（因为 ground truth 是 LDR image）。

#### 2.6.2 HDR Constraint Loss（µ-law）

公式 (16)：
$$\mathcal{L}_c = \sum_{j=1}^{B} \left\| \frac{\log(1 + \mu \cdot \text{norm}(I_j^h))}{\log(1 + \mu)} - \frac{\log(1 + \mu \cdot \text{norm}(\hat{I}_j^h))}{\log(1 + \mu)} \right\|_2^2$$

- $\mu$：compression amount（通常 $\mu = 5000$ 或类似大值）
- $\text{norm}(\cdot)$：min-max normalization 到 [0, 1]
- $I_j^h, \hat{I}_j^h$：rendered 和 ground truth HDR image

µ-law companding 是 telecommunication 中的 classic trick（类似 µ-law PCM audio encoding），把 wide dynamic range 信号压缩到窄 range，让 L2 loss 在 HDR domain 上有意义。

**Intuition**：直接在 HDR domain 算 L2 loss 会被 bright region 主导（数值大），dark region 几乎无 gradient。µ-law 用 $\log(1 + \mu x) / \log(1 + \mu)$ 压缩 dynamic range，让 bright 和 dark region 在 loss 中权重平衡。

#### 2.6.3 总损失

公式 (17)：
$$\mathcal{L} = \mathcal{L}_p + \gamma \cdot \mathcal{L}_c$$

- $\gamma$：控制 HDR constraint 权重
- 合成场景 $\gamma = 0.6$（有 GT HDR）
- 真实场景 $\gamma = 0$（无 GT HDR，只用 LDR supervision）

---

## 三、实验结果深度分析

### 3.1 Quantitative Results（合成场景，Table 1）

| Method | Training (min) | Inference (fps) | LDR-OE PSNR↑ | LDR-NE PSNR↑ | HDR PSNR↑ |
|---|---|---|---|---|---|
| NeRF | 405 | 0.190 | 13.97 | 14.51 | - |
| 3DGS | 38 | 121 | 19.46 | 18.97 | - |
| NeRF-W | 437 | 0.178 | 29.83 | 29.22 | - |
| HDR-NeRF | 542 | 0.122 | 39.07 | 37.53 | 36.40 |
| **HDR-GS** | **34** | **126** | **41.10** | 36.33 | **38.31** |

关键观察：
1. **训练时间**：HDR-GS (34 min) vs HDR-NeRF (542 min)，只需 **6.3%** 的训练时间
2. **推理速度**：HDR-GS (126 fps) vs HDR-NeRF (0.122 fps)，**1033× 加速**
3. **LDR-OE PSNR**：+2.03 dB vs HDR-NeRF（OE = observed exposure，训练时见过的 exposure）
4. **LDR-NE PSNR**：-1.20 dB vs HDR-NeRF（NE = novel exposure，训练时没见过的 exposure）。HDR-GS 略低，可能因为 MLP-based tone-mapper 在 unseen exposure 上泛化稍弱
5. **HDR PSNR**：+1.91 dB
6. **vs 3DGS**：HDR-GS 在 LDR-OE 上高 21.64 dB！这说明 DDR model 是核心

**为什么 HDR-GS 比 3DGS 稍快（126 vs 121 fps）？** Paper 给的解释很有意思：3DGS 在不同曝光训练时不收敛，会盲目 split Gaussian point cloud 试图拟合 exposure variance，导致 Gaussian 数量膨胀；HDR-GS 通过 DDR 正确建模 exposure，Gaussian 数量稳定，反而更快。

### 3.2 Quantitative Results（真实场景，Table 2）

| Method | LDR-OE PSNR↑ | LDR-NE PSNR↑ |
|---|---|---|
| HDR-NeRF | 31.63 | 31.43 |
| **HDR-GS** | **35.47** | **31.66** |

真实场景下 HDR-GS 在 LDR-OE 上 +3.84 dB，提升比合成场景更大，说明泛化能力强。

### 3.3 Ablation Study 深度解析

#### 3.3.1 Break-down Ablation（Table 3a）

| Component | LDR-OE PSNR |
|---|---|
| Baseline (3DGS + NDC) | 12.35 |
| + Camera Recalibration | 14.62 (+2.27) |
| + SfM Points | 19.46 (+4.84) |
| + DDR Model | 41.10 (+21.64) |

- **Camera Recalibration**：+2.27 dB，从 NDC 解放到真实 3D coordinate
- **SfM Points**：+4.84 dB，提供好的 Gaussian 初始化，避免 3DGS 过拟合
- **DDR Model**：+21.64 dB，这是最大贡献。DDR 让模型能处理 exposure variance，避免 3DGS 在多曝光训练下的崩溃

#### 3.3.2 CRF Domain（Table 3b）

| Domain | HDR PSNR | LDR-OE PSNR |
|---|---|---|
| Linear | 26.18 | 29.53 |
| Logarithmic | 38.31 | 41.10 |

**对数域 vs 线性域**：+12.13 dB（HDR）、+11.57 dB（LDR-OE）。这是巨大提升，验证了 paper 的核心 hypothesis：log domain 让 MLP 训练稳定，避免 numerical overflow 和 input nonlinearity。

#### 3.3.3 Training Exposure Times（Table 3c）

| Exposure Set | HDR PSNR |
|---|---|
| {t3} | 22.86（fail to reconstruct HDR） |
| {t1, t5} | 32.06 |
| {t1, t3, t5} | 38.31 |
| {t1, t2, t3, t4, t5} | 38.50 |

- 单 exposure 无法恢复 CRF（Debevec-Malik 理论要求至少 2 个 exposure）
- 3 个 exposure 已经接近 saturation 性能，再加 2 个只 +0.19 dB
- **实用 conclusion**：3 个 exposure 是性价比最高的选择

#### 3.3.4 Recalibration Exposure Time（Table 3d）

| $t_s$ | HDR PSNR |
|---|---|
| $t_1 = 0.125s$ | 36.88 |
| $t_2 = 0.25s$ | 37.90 |
| $t_3 = 2s$ | 38.16 |
| $t_4 = 8s$ | **38.31** |
| $t_5 = 32s$ | 38.05 |

最优是 $t_4 = 8s$（中等偏长曝光），验证了 intuition：太短曝光暗部 feature 缺失，太长曝光亮部 saturate，中等偏长曝光 feature 最丰富。

---

## 四、Intuition 总结与相关联想

### 4.1 核心创新点 Intuition

**1. SH + exp 保证 HDR 非负**
原始 3DGS 的 sigmoid 把 color 限制在 [0, 1]，HDR-GS 用 exp 释放到 $[0, +\infty)$。这和 NeRF 中 density 用 $\exp(\cdot)$ 保证非负的 trick 一脉相承。

**2. Log-domain Tone Mapping**
这个 trick 的本质是：把 multiplicative physical relationship（$E = L \cdot \Delta t$）转换成 additive log-domain relationship（$\log E = \log L + \log \Delta t$）。加法对神经网络友好，因为：
- Numerical stability：log 把 wide range 压缩到窄 range
- Linearity：加法 shift 是 MLP 容易拟合的 input transformation
- Disentanglement：HDR color 和 exposure time 在 input 上独立相加，MLP 只需学 CRF shape

**3. Shared Geometry, Separate Color**
HDR 和 LDR 共享 $\mu, \Sigma, \alpha$，只有 color 不同。这是 physics-motivated design：同一个 3D point 的 geometry 不随 exposure 变化，只有 sensor 测量值变化。

### 4.2 与其他方法的联想

**1. 与 HDR-NeRF 的关系**
HDR-NeRF 也用 log-domain CRF，但用 MLP 表示 radiance field。HDR-GS 用 SH 表示 HDR color，更 efficient，因为 SH 是 explicit basis function，不需要 per-point MLP query。

**2. 与 NeRF-W 的关系**
NeRF-W [Martin-Brualla et al., CVPR 2021] 处理 unconstrained photo collection 中的 appearance variation，用 per-image embedding 做 conditional radiance field。HDR-GS 的 exposure time $\Delta t$ 类似 NeRF-W 的 appearance embedding，但是**物理上有明确意义**（exposure time 是 scalar，可读自 metadata），而 NeRF-W 的 embedding 是 latent，无物理含义。

**3. 与 3DGS 后续工作的关系**
- GS-IR [Liang et al. 2023]：3DGS for inverse rendering，也涉及 material/lighting 分解
- PhysGaussian [Xie et al. 2023]：physics-integrated 3DGS，关注 dynamics
- HDR-GS 是 3DGS 在 computational photography 方向的早期探索

**4. 与 Tone Mapping 经典方法的关系**
- Reinhard's global tone mapping: $L_d = \frac{L}{1 + L}$
- Reinhard's local tone mapping: 基于 dodging-and-burning
- Photomatix Pro (paper 中用于 HDR visualization): commercial HDR tone mapping software
- HDR-GS 学到的是 sensor CRF（从 HDR radiance 到 LDR pixel value），这是 forward tone mapping，与 Reinhard 等 backward tone mapping（从 HDR display 到 LDR display）方向相反

### 4.3 Limitation 与未来方向

Paper 提到的主要 limitation 是 3DGS 的 memory usage 大（每个 Gaussian 要存 $\mu, \Sigma, \alpha, \text{SH coefficients}, \text{MLP params}$，百万级 Gaussian 时 VRAM 消耗显著）。

未来可能的扩展：
1. **Video HDR-GS**：结合 4DGS [Wu et al. 2023] 处理动态场景的 HDR NVS
2. **HDR-SLAM**：结合 Gaussian-SLAM [Matsuki et al. 2023] 做实时 HDR SLAM
3. **Sparse-view HDR**：结合 SparseNeRF [Wang et al. ICCV 2023] 的 few-shot 技术
4. **HDR Generation**：结合 DreamGaussian [Tang et al. 2023] 做 text-to-3D HDR content creation
5. **Medical HDR**：结合 R2-Gaussian [Zha et al. NeurIPS 2024] 在 X-ray/CT 等医学影像中做 HDR 重建
6. **Mobile deployment**：结合 MobileNeRF [Chen et al. CVPR 2023] 的 polygon rasterization 思路降低 memory

---

## 五、关键 References

1. **3D Gaussian Splatting** [Kerbl et al., SIGGRAPH 2023]
   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

2. **HDR-NeRF** [Huang et al., CVPR 2022]
   https://github.com/cszhilu1998/HDR-NeRF

3. **Debevec-Malik HDR Calibration** [SIGGRAPH 1997]
   https://www.pauldebevec.com/Research/HDR/

4. **COLMAP SfM** [Schönberger & Frahm, CVPR 2016]
   https://colmap.github.io/

5. **NeRF** [Mildenhall et al., ECCV 2020]
   https://www.matthewtancik.com/nerf

6. **NeRF-W** [Martin-Brualla et al., CVPR 2021]
   https://nerf-w.github.io/

7. **EWA Volume Splatting** [Zwicker et al., 2001]
   https://www.cs.umd.edu/~zwicker/publications/EWAVolSplatting-VIS2001.pdf

8. **HDR-GS Official Code**
   https://github.com/caiyuanhao1998/HDR-GS

9. **Photomatix Pro** (HDR tone mapping software)
   https://www.hdrsoft.com/

10. **PyTorch**
    https://pytorch.org/

11. **Adam Optimizer** [Kingma & Ba, ICLR 2015]
    https://arxiv.org/abs/1412.6980

12. **SSIM** [Wang et al., TIP 2004]
    https://www.cns.nyu.edu/~lcv/ssim/

13. **LPIPS** [Zhang et al., CVPR 2018]
    https://github.com/richzhang/PerceptualSimilarity

14. **Blender** (synthetic scene generation)
    https://www.blender.org/

15. **µ-law Companding** (Wikipedia)
    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

16. **Spherical Harmonics** (Wikipedia)
    https://en.wikipedia.org/wiki/Spherical_harmonics

17. **Camera Response Function** (computer vision intro)
    https://en.wikipedia.org/wiki/Camera_response_function

18. **Real-time HDR Imaging** [Kalantari et al., ACM ToG 2017]
    https://cseweb.ucsd.edu/~nkalanant/research.html

19. **Deep HDR Imaging Survey**
    https://arxiv.org/abs/2103.11604

20. **4D Gaussian Splatting** [Wu et al. 2023]
    https://github.com/hustvl/4DGaussians

---

## 六、一句话总结

HDR-GS 的核心 insight 是：**把物理上的 multiplicative exposure model ($E = L \cdot \Delta t$) 通过 log-domain 变换转成 additive model ($\log E = \log L + \log \Delta t$)，让 SH-based HDR color 和 exposure time 在 MLP tone-mapper 输入端独立相加，从而用 3DGS 的 explicit representation 高效实现 HDR NVS，同时获得 1000× 推理加速和更好的重建质量**。

这个工作在 NeRF → 3DGS 的 paradigm shift 中填补了 HDR imaging 这个重要 niche，其 log-domain CRF 建模、shared-geometry dual-color rasterization、SfM recalibration 三个 contribution 都具有方法学价值，可以推广到其他涉及 physics-aware appearance modeling 的 3D vision 任务。
