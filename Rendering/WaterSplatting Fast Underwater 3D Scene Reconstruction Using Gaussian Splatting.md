---
source_pdf: WaterSplatting Fast Underwater 3D Scene Reconstruction Using Gaussian
  Splatting.pdf
paper_sha256: 883a90755e6d86dce3f75f684d8a741b2b317e37b59cd93294218a9b3dd3d713
processed_at: '2026-08-13T03:46:30-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,Andrej,我用大白话跟你聊聊这篇paper在干嘛。

## 一句话说清这paper在干嘛

**在水下拍的照片又蓝又雾,想重建出清晰的3D场景,而且要实时跑。**

传统做法要么太慢(NeRF系,训练10小时,渲染0.07 FPS),要么压根不认水下场景(3DGS只会画硬表面,水这种"半透明雾"它不会画,硬画就糊成一坨)。这帮人说:那我拆开——硬的东西用3DGS画,水用一个小网络+指数衰减单独算,两套拼一起,又快又好。

---

## 问题在哪:水下照片为什么难重建?

你想象一张水下照片,它实际是两层东西叠加:

```
你看到的 = 真实物体被水衰减后的残影 + 水自己散射进来的蓝绿背景光
```

- **物体那部分**:光从物体表面出发,穿过水,被水吸收+散射,越远越暗、越偏蓝绿。这是"direct component"。
- **水自己那部分**:环境光被水里的粒子散射,有一小部分恰好朝你相机飞回来,形成那层蓝绿色的"雾"。这是"backscatter"。

这两层用的衰减系数还**不一样**——因为光路不同:
- direct 路径主要被absorption吃掉,衰减系数叫 $\beta^D$
- backscatter 路径是"粒子把你方向的光弹回来",衰减系数叫 $\beta^B$

这俩一般不相等,用一个数会算错颜色。这是Akkaynak 2018那篇的核心insight,WaterSplatting直接沿用。

---

## 为什么3DGS自己搞不定水下

3DGS的本质是:拿一堆3D高斯球,投影到屏幕上,alpha blend叠加。

它的优化机制有两个杀手:
1. **opacity低的会被prune掉**——水下那层"雾"本质是低密度体积,3DGS一看opacity低就剪了,结果雾没了。
2. **如果不剪,3DGS会拿一堆muddy的cloud-like高斯去拟合雾**,产生一堆floater,新视角下就是一坨artifact。

一句话:3DGS是给"硬表面"设计的,水是"软体积",硬塞进去就崩。

---

## 为什么NeRF系慢

SeaThru-NeRF的做法是two fields:一个geometry field画物体,一个medium field画水。理论上对,但:
- NeRF本质是沿ray采样几百个点,每个点过MLP,慢。
- proposal sampler加速后又破坏了medium的学习能力。
- 训练10小时,渲染0.07 FPS,实用不了。

---

## WaterSplatting的核心trick

直觉是这样的:**沿一条ray,光线经历的"遮挡"有两种本质不同的东西**:

1. **碰到硬物体**:这是个离散事件,光线要么被挡要么没被挡。3DGS的alpha product天然适合表达这个。
2. **穿过水**:这是个连续过程,光被指数衰减。解析公式 $\exp(-\sigma \cdot s)$ 天然适合表达这个。

这俩在物理上独立——"有没有被物体挡"和"被水衰减多少"是两件不相关的事。所以transmittance可以因式分解:

$$T(s) = T^{obj}(s) \times T^{med}(s)$$

- $T^{obj}$:前面所有高斯的 $(1-\alpha)$ 乘积,离散的
- $T^{med}$:$\exp(-\sigma \cdot s)$,连续的指数衰减

**这就是整个paper的灵魂**:把一个连续体积渲染积分,拆成"高斯段贡献 + 高斯之间水段的贡献",交替累加。

---

## 具体怎么算每个pixel的颜色

沿ray走,假设有N个高斯挡在路上,深度依次是 $s_1, s_2, ..., s_N$:

```
相机(深度0)
  │
  │← 这段是水,贡献 c_med * [1 - exp(-σ_bs·s_1)] * T_1^obj
  │
高斯1(深度s_1)
  │← 贡献 c_1 * α_1 * exp(-σ_attn·s_1) * T_1^obj
  │
  │← 这段是水,贡献 c_med * [exp(-σ_bs·s_1) - exp(-σ_bs·s_2)] * T_2^obj
  │
高斯2(深度s_2)
  │← 贡献 c_2 * α_2 * exp(-σ_attn·s_2) * T_2^obj
  │
  ...
  │
高斯N(深度s_N)
  │← 贡献 c_N * α_N * exp(-σ_attn·s_N) * T_N^obj
  │
  │← 后面全是水到无穷远,贡献 c_med * exp(-σ_bs·s_N) * T_N^obj (这就是backscatter背景)
  ∞
```

几个细节:
- **物体贡献用 $\sigma^{attn}$ 衰减**:光从物体出发到相机,被水吸收。
- **水段贡献用 $\sigma^{bs}$ 衰减**:水自己散射的光,衰减规律不同。
- **最后一段无穷远**:如果光线穿过了所有高斯,你看到的就是纯水背景色,这就是 $B^\infty$ 那项。
- $T_i^{obj}$ 是前面所有高斯累积的"没被挡住"的概率,水段不改变这个值(水不"挡"光,只衰减)。

---

## Medium怎么参数化

$\sigma^{attn}, \sigma^{bs}, c^{med}$ 不是全局常数,是per-pixel query一个小MLP:
- 输入:ray direction(经过spherical harmonic encoding)
- 输出:3个数($c^{med}$用sigmoid,$\sigma$用softplus保证正)
- MLP很小:2层,128 hidden units

为什么只用ray direction不用3D position?因为水下介质在一个scene内大致随方向变化(向上看光多,向下看光少),空间变化相对弱。这是简化,也是局限——真实水下分层(温跃层、浊度层)它建模不了。

每次3DGS做densification/pruning后,这个medium MLP的Adam动量被reset,避免优化历史污染。

---

## Loss那块在干嘛

水下照片HDR特性明显:暗区多,人眼对暗区敏感。普通L1/L2在暗区权重低,优化会忽略暗区细节。

**NeRF in the Dark**那篇提了个trick:给暗区更高权重

$$w_{i,j} = \frac{1}{\hat{y}_{i,j} + \epsilon}$$

像素越暗,$\hat{y}$ 越小,$w$ 越大,loss被放大。stop-gradient保证权重不参与反传(否则会出NaN)。

WaterSplatting把这个weight同时乘到预测和GT上,做两件事:
1. **Reg-L2**:pixel level,逼medium的smoothness
2. **Reg-DSSIM**:patch level,逼结构相似

为什么3DGS必须用D-SSIM而NeRF不用?因为3DGS每个高斯是独立参数,没有shared MLP的内在耦合。暗区如果只用pixel loss,高斯们会各自为政搞出一堆孤立floater去拟合单个暗像素。D-SSIM在patch级别引入结构约束,逼相邻高斯形成连贯结构。这是3DGS和NeRF在loss设计上一个微妙但关键的区别。

---

## 实验结果说人话

SeaThru-NeRF数据集(4个真实水下场景):

| 方法 | 平均PSNR | FPS | 训练时间 |
|---|---|---|---|
| SeaThru-NeRF | 26.84 | 0.07 | 10小时 |
| ZipNeRF | 28.77 | 0.9 | 6小时 |
| 3DGS(原版) | 25.50 | 412 | 17分钟 |
| **WaterSplatting** | **29.69** | **41.8** | **9.4分钟** |

翻译:
- 比SeaThru-NeRF质量高约3dB,速度快600倍
- 比ZipNeRF质量高约1dB,训练快40倍,渲染快50倍
- 比原版3DGS质量高约4dB(因为3DGS在水下崩了),速度降一些(因为多了medium计算)但仍是实时

去水(restoration)实验(合成雾场景):
- easy fog:他们15.70 vs SeaThru-NeRF 13.11
- hard fog:他们14.06 vs 10.76,差距更大

意思是雾越重,他们方法的优势越明显——因为SeaThru-NeRF的geometry field会"贪心"地把雾也学进surface,产生wave-like artifact,而WaterSplatting的geometry和medium解耦干净。

---

## 局限性,人也话说

1. **远处分不清是水还是背景物体**:远处transmittance接近0,信号弱,无论3DGS还是NeRF都难。比如远处水面和远处的水雾看起来一样。
2. **要已知camera pose**:水下SfM本来就难(dome port能避免折射问题,但flat port有折射,feature也少)。
3. **训练视角没覆盖的区域**:3DGS会产生floater,SeaThru-NeRF会blur,都烂,烂的方式不同。
4. **颜色恢复本质ill-posed**:观测到 $I = O \cdot e^{-\beta z}$,$O$和$\beta$在同一像素耦合,没先验就分不开。远处物体的"真实颜色"其实恢复不准。

---

## 我读完的intuition

这个paper的优雅之处在于:**它没有发明新的物理模型,也没有改3DGS的底层 rasterizer,而是做了一个数学上的transmittance因式分解,让两种本质不同的物理过程(离散遮挡 vs 连续衰减)各用各擅长的表达方式**。

- 离散遮挡 → 3DGS的alpha product
- 连续衰减 → 解析的指数函数
- 两者相乘 → 完整的体积渲染

这种"让representation匹配物理过程本质"的思路,其实和NeRF当初用volume rendering表达emission-absorption模型是一脉相承的哲学——只不过NeRF用MLP表达一切,WaterSplatting让explicit primitive和continuous field分工。

如果你想做extension,我猜几个有意思的方向:
- **time-varying medium**:海流、浮游生物随时间变,加个temporal embedding到medium MLP
- **3D-varying medium**:把medium MLP输入改成3D position(不只ray direction),建模温跃层
- **active illumination**:ROV有探照灯,光路模型要加一条light source到物体到相机的双段衰减
- **polarization**:偏振信息能更好估计backscatter,塞进medium MLP当input
- **SLAM集成**:41.8 FPS + explicit representation天生适合做underwater SLAM,接SplaTAM那套框架

paper本身是solid的工程+物理建模工作,quality提升不大(1dB量级),但**efficiency提升是数量级的**,这是它的实际价值——让水下3D重建从"论文里能跑"变成"机器人上能跑"。

项目页:https://water-splatting.github.io

---

# WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting 深度解析

## 1. Paper Overview 与核心动机

这篇paper来自 University of Zurich / ETH Zurich / CTU in Prague 团队(作者 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig, Jonas Kulhanek),核心目标是解决 underwater 3D scene reconstruction 中一个根本性的 representation bottleneck:

- **NeRF-based methods**(如 SeaThru-NeRF [Levy et al. CVPR 2023])是 fully volumetric,理论上可以同时建模 geometry 与 medium(water),但 proposal sampler 加速后 medium 学习能力被破坏,且 training/rendering 慢(SeaThru-NeRF training 10 小时,FPS 仅 0.07)。
- **3DGS**(Kerbl et al. 2023)是 explicit representation,real-time 渲染、易编辑,但只渲染 surface geometry,对半透明 medium(水雾、散射)无能为力 —— 因为 3DGS 在 optimization 时会 prune 低 opacity 的 Gaussians,导致 medium 部分无法被 dense 的 muddy cloud-like primitives 拟合,反而产生 artifact。

WaterSplatting 的核心 insight 是:**保留 3DGS 的 explicit geometry 表示,在 alpha compositing 的两个相邻 Gaussian 之间"插入" medium 的 volumetric 积分 contribution**。即把 medium 作为一个"per-pixel queried" 的 volumetric field,与 3DGS 的 explicit primitives 交错渲染。

Web: https://water-splatting.github.io

---

## 2. Preliminaries: 3DGS 渲染方程回顾(公式 1-5)

### 公式 (1):3D Gaussian 定义

$$
G_i(p) = e^{-\frac{1}{2}(p - \mu_i)^T (\Sigma_i)^{-1} (p - \mu_i)}
$$

- $G_i(p)$:第 $i$ 个 Gaussian primitive 在 3D 点 $p$ 处的概率密度值。
- $\mu_i$:该 Gaussian 的中心位置(mean),是一个 3D 向量,可学习。
- $\Sigma_i$:3×3 的 covariance matrix,可学习,实际优化时分解为 $\Sigma_i = R_i S_i S_i^T R_i^T$(rotation $R_i$ 用 quaternion 表示,scale $S_i$ 是 3D 向量),保证 positive semi-definite。
- 上标 $T$:转置;上标 $-1$:矩阵求逆。

### 公式 (2)-(4):从 3D 投影到 2D

$$
\hat{\Sigma}_i = (JW\Sigma_i W^T J^T)_{1:2,1:2}, \quad \hat{\mu}_i = (P\mu_i)_{1:2}
$$

$$
s_i = (P\mu_i)_3
$$

$$
\hat{G}_i(p) = e^{-\frac{1}{2}(p - \hat{\mu}_i)^T (\hat{\Sigma}_i)^{-1} (p - \hat{\mu}_i)}
$$

- $W$:viewing transformation(world to camera)。
- $J$:Jacobian of affine approximation of projective transformation(camera to screen 透视投影的局部线性化)。
- $P$:projection matrix。
- $\hat{\Sigma}_i$:projected 2D covariance(取前 2 行 2 列)。
- $\hat{\mu}_i$:projected 2D mean。
- $s_i$:该 Gaussian 在 camera space 的 z-depth(第 3 分量),用于后续 depth sorting 与 medium 衰减。
- 下标 $1:2, 1:2$:取前两行两列;$_3$:取第 3 分量;$_{1:2}$:取前 2 分量。

### 公式 (5):Alpha Blending

$$
C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = \sigma(o_i) \cdot \hat{G}_i(p)
$$

- $C$:pixel 最终颜色。
- $c_i$:第 $i$ 个 Gaussian 在 view direction 下经由 SH(spherical harmonics)计算得到的颜色。
- $\alpha_i$:第 $i$ 个 Gaussian 在 pixel $p$ 处的 effective opacity。
- $\sigma(\cdot)$:Sigmoid 函数,把 raw opacity $o_i$ 压缩到 $(0,1)$。
- $\prod_{j=1}^{i-1}(1-\alpha_j)$:transmittance,即前面 $i-1$ 个 Gaussian 累积的"未遮挡率"。
- $N$:参与 alpha blending 的 Gaussian 数(经过 3-sigma truncation 与 16×16 patch culling)。

这里的 $\prod_{j=1}^{i-1}(1-\alpha_j)$ 即对应后续公式中的 $T_i^{obj}$。

---

## 3. 水下 Image Formation Model(公式 6)

$$
I = \underbrace{O \cdot e^{-\beta^D(\mathbf{v}_D) \cdot z}}_{\text{Direct Image}} + \underbrace{B^\infty \cdot (1 - e^{-\beta^B(\mathbf{v}_B) \cdot z})}_{\text{Backscatter}}
$$

- $I$:传感器测量到的 underwater image。
- $O$:假设无 medium 时直接看到的 clear scene radiance。
- $z$:depth(沿 ray 距离 camera 的距离)。
- $B^\infty$:backscatter 在无穷远处的渐近颜色(整个水体被环境光充分散射后的"背景色")。
- $\beta^D$:direct component 的 attenuation coefficient(直接光被水吸收+散射衰减)。
- $\beta^B$:backscatter component 的 attenuation coefficient(背景散射光被衰减到达 camera)。
- $\mathbf{v}_D$:direct component 的 dependency vector,包含 $z$、reflectance、ambient light、water scattering、attenuation coefficient 等。
- $\mathbf{v}_B$:backscatter component 的 dependency vector,包含 ambient light、scattering properties、backscatter coefficient、attenuation coefficient 等。

**Intuition**:水下图像本质是"被衰减的真实物体 + 被衰减的环境散射"。两者使用不同的 attenuation 是因为 wide-band color channel 下 direct 路径与 backscatter 路径的有效衰减不同。这是 Akkaynak & Treibitz [CVPR 2018] 的 revised underwater image formation model,与经典 Koschmieder 模型区别在于分离了两套 $\beta$。

---

## 4. Splatting with Medium:核心方法(公式 7-18)

### 公式 (7)-(8):连续形式 volumetric 渲染

$$
C(r) = \int_0^{\infty} T(s) \big(\sigma^{obj}(s) c^{obj}(s) + \sigma^{med}(s) c^{med}(s)\big) ds
$$

$$
T(s) = \exp\left(-\int_0^{s} (\sigma^{obj}(s') + \sigma^{med}(s')) ds'\right)
$$

- $r(s) = o + ds$:camera ray,$o$ 是 camera origin,$d$ 是 ray direction。
- $T(s)$:transmittance,从 camera 到深度 $s$ 之间未被任何东西遮挡的概率。
- $\sigma^{obj}(s)$:object(几何表面)的体积密度,在 3DGS 中由 Gaussian primitives 离散化。
- $\sigma^{med}(s)$:medium(water)的体积密度。
- $c^{obj}(s), c^{med}(s)$:object 和 medium 的颜色。
- $ds$:沿 ray 的积分微元。

**关键 insight**:这里把"object emission" 与 "medium emission" 显式分离,允许后续用 3DGS 处理前者、用 neural field 处理后者。

### 公式 (9)-(10):Transmittance 分解

$$
T_i(s) = T_i^{obj} T^{med}(s), \quad T_i^{obj} = \prod_{j=1}^{i-1}(1-\alpha_j)
$$

$$
T^{med}(s) = \exp(-\sigma^{med} \cdot s)
$$

- $T_i(s)$:在第 $i$ 个 Gaussian 之前(即 $s \in [s_{i-1}, s_i]$ 区间)的总 transmittance。
- $T_i^{obj}$:由前面 $i-1$ 个 Gaussian 累积的"object transmittance"(来自公式 5 的累积项)。
- $T^{med}(s)$:由 medium 衰减产生的 transmittance,假设 $\sigma^{med}$ per-ray constant(每条 ray 一个值,每个 color channel 独立),所以 $-\int_0^s \sigma^{med} ds' = -\sigma^{med} s$。

**Intuition**:这是 dual representation 的核心 —— 3DGS 用离散的 alpha product 表达几何遮挡,medium 用连续的指数衰减表达体积衰减,二者相乘。

### 公式 (11)-(14):离散化渲染

$$
C(r) = \sum_{i=1}^{N} C_i^{obj}(r) + \sum_{i=1}^{N} C_i^{med}(r)
$$

$$
C_i^{obj}(r) = T_i^{obj} \alpha_i c_i \exp(-\sigma^{med} s_i)
$$

$$
C_i^{med}(r) = T_i^{obj} c^{med} \big[\exp(-\sigma^{med} s_{i-1}) - \exp(-\sigma^{med} s_i)\big]
$$

$$
C_\infty^{med}(r) = T_N^{obj} c^{med} \exp(-\sigma^{med} s_N)
$$

- $C_i^{obj}(r)$:第 $i$ 个 Gaussian 对 pixel 颜色的直接贡献 —— 它自己的颜色 $c_i$ 被 alpha blending 权重 $\alpha_i$ 加权,再被前面 Gaussian 累积遮挡 $T_i^{obj}$ 衰减,再被 medium 从 camera 到 $s_i$ 距离的衰减 $\exp(-\sigma^{med} s_i)$ 进一步衰减。
- $C_i^{med}(r)$:第 $(i-1)$-th Gaussian 到第 $i$-th Gaussian 之间的 medium segment 贡献。其积分是 $T_i^{obj} \int_{s_{i-1}}^{s_i} \exp(-\sigma^{med} s) \sigma^{med} c^{med} ds = T_i^{obj} c^{med} [\exp(-\sigma^{med} s_{i-1}) - \exp(-\sigma^{med} s_i)]$,这是 $\exp$ 的原函数差。
- $C_\infty^{med}(r)$:从最后一个 Gaussian 到无穷远的 medium 贡献,本质上就是"如果光线穿过了所有 Gaussian,看到的就是 $T_N^{obj}$ 衰减后的 backscatter 背景色"。

### 公式 (15)-(18):最终渲染(分离 attenuation 与 backscatter)

$$
C(r) = \sum_{i=1}^{N} C_i^{obj}(r) + \sum_{i=1}^{N} C_i^{med}(r) + C_\infty^{med}(r)
$$

$$
C_i^{obj}(r) = T_i^{obj} \alpha_i c_i \exp(-\sigma^{attn} s_i)
$$

$$
C_i^{med}(r) = T_i^{obj} c^{med} \big[\exp(-\sigma^{bs} s_{i-1}) - \exp(-\sigma^{bs} s_i)\big]
$$

$$
C_\infty^{med}(r) = T_N^{obj} c^{med} \exp(-\sigma^{bs} s_N)
$$

- $\sigma^{attn}$:object light 受 medium 的 attenuation 系数(对应公式 6 的 $\beta^D$)。
- $\sigma^{bs}$:backscatter 受 medium 衰减的系数(对应公式 6 的 $\beta^B$)。
- $s_0 = 0$:camera 位置作为起点。

**关键设计**:为什么用 $\sigma^{attn}$ 和 $\sigma^{bs}$ 两套参数,而非一个 $\sigma^{med}$?因为 wide-band color channel 下,direct light(从物体到 camera)经历的有效衰减与 backscatter light(从光源被 medium 散射回 camera)的有效衰减不同,前者更受 absorption 主导,后者更受 scattering 主导。这是 Akkaynak 模型与物理光学的核心。

### Medium Encoding 架构

Medium 的 $\sigma^{attn}, \sigma^{bs}, c^{med}$ 由一个小 MLP 输出:
- Input:pixel 的 ray direction(或位置)经 spherical harmonic encoding [Ref-NeRF, Verbin et al. CVPR 2022] 编码后的特征。
- Architecture:2 linear layers × 128 hidden units,Sigmoid 中间激活。
- Output heads:
  - $c^{med}$:Sigmoid 激活(取值 $(0,1)^3$)。
  - $\sigma^{attn}$ 和 $\sigma^{bs}$:Softplus 激活(保证正值)。
- 在每次 3DGS 的 densification 和 pruning 后,Adam optimizer 的 medium encoding moving averages 被 reset,保证后续迭代独立性。

**Intuition**:medium 是 spatially-varying 的(per-pixel query 一次),允许不同 ray 方向有不同的水属性(例如上下光强不同)。这与 SeaThru-NeRF 的两 field 思路一致,但更轻量。

---

## 5. Loss Function Alignment(公式 19-22)

### 公式 (19):NeRF in the Dark 的 Regularized L2

$$
\mathcal{L}_{\text{Reg-}L_2} = \big((sg(\hat{y}) + \epsilon)^{-1} \odot (\hat{y} - y)\big)^2
$$

- $\hat{y}$:rendered image。
- $y$:ground truth。
- $\odot$:Hadamard(element-wise)积。
- $sg(\cdot)$:stop-gradient 算子,反向传播时梯度为 0。
- $\epsilon$:小常数(防除 0)。

**Intuition**:这是 Mildenhall et al. [NeRF in the Dark, CVPR 2021] 提出,目的是给 dark 区域(低 $\hat{y}$ 值)更高权重,匹配人眼对 dark region 高 dynamic range 的敏感性。$(\hat{y} + \epsilon)^{-1}$ 在 dark pixel 处值大,放大 dark region 误差。

### 公式 (20)-(21):Regularized L1 与 Regularized D-SSIM

$$
\mathcal{L}_{\text{Reg-}L_1} = |W \odot (\hat{y} - y)|, \quad w_{i,j} = (sg(\hat{y}_{i,j}) + \epsilon)^{-1}
$$

$$
\mathcal{L}_{\text{Reg-DSSIM}} = \mathcal{L}_{DSSIM}(W \odot y, W \odot \hat{y})
$$

- $W = \{w_{i,j}\}$:per-pixel weight,$w_{i,j}$ 与 dark region 成反比。
- $\mathcal{L}_{DSSIM}$:D-SSIM loss(标准结构相似性损失)。
- $(i,j)$:pixel 坐标。

**Intuition**:WaterSplatting 把 $W$ 同时作用在 $\hat{y}$ 与 $y$ 上做 D-SSIM,而 SeaThru-NeRF 直接用 Reg-L2 on pixel level。作者强调 3DGS 的 discrete primitive 性质需要 structural regularization(D-SSIM)来保证 independently optimized Gaussians 之间的 perceptual consistency,而 NeRF 因为是 shared parameter volumetric representation,pixel-level regularized loss 已经够用。这是 3DGS 与 NeRF 在 loss 设计上一个微妙但重要的差异。

### 公式 (22):Final Loss

$$
\mathcal{L}_{Reg} = (1 - \lambda) \mathcal{L}_{\text{Reg-}L_2} + \lambda \mathcal{L}_{\text{Reg-DSSIM}}
$$

- $\lambda$:loss 平衡权重。

最终选择 Reg-L2 + Reg-DSSIM 组合,而非 Reg-L1,因为 medium 是 smooth volumetric,Reg-L2 更适合 smoothness。

---

## 6. 实验结果深度分析

### Dataset

- **SeaThru-NeRF Dataset**(Levy et al. 2023):
  - IUI3 Red Sea, Curaçao, Japanese Gardens Red Sea, Panama 共 4 个 underwater scene。
  - 训练/验证:25/4, 17/3, 17/3, 15/3。
  - 拍摄设备:Nikon D850 SLR + Nauticam underwater casing + dome port(防 refraction)。
  - 分辨率:约 900×1400。
  - Pre-processing:white balancing(0.5% clipping per channel 去噪)+ COLMAP [Schönberger & Frahm, CVPR 2016] 求 pose 与 undistortion。

- **Simulated Dataset**:用 Mip-NeRF 360 的 Garden scene + 合成 fog:
  - Easy foggy:$\beta^D = [0.6,0.6,0.6], \beta^B = [0.6,0.6,0.6], B^\infty = [0.5,0.5,0.5]$。
  - Hard foggy:$\beta^D = [0.8,0.8,0.8], \beta^B = [0.6,0.6,0.6], B^\infty = [0.5,0.5,0.5]$。

### Table 1 主实验结果(SeaThru-NeRF Dataset)

| Method | IUI3 PSNR | Curaçao PSNR | J.G. PSNR | Panama PSNR | Avg. PSNR | FPS | Train Time |
|---|---|---|---|---|---|---|---|
| SeaThru-NeRF | 27.31 | 29.31 | 22.12 | 28.61 | 26.84 | 0.07 | 10h |
| SeaThru-NeRF-NS | 27.65 | 30.61 | 23.54 | 31.86 | 28.42 | 0.9 | 2h |
| ZipNeRF | 29.35 | 29.93 | 23.45 | 32.34 | 28.77 | 0.9 | 6h |
| 3D Gauss. | 22.98 | 28.31 | 21.49 | 29.20 | 25.50 | 412.1 | 17.4min |
| **Ours** | **29.39** | **32.67** | **25.20** | 31.49 | **29.69** | **41.8** | **9.4min** |

**Intuition 解读**:
1. **Ours 在 IUI3、Curaçao、J.G. 三场景 PSNR 第一**,Panama 略输 ZipNeRF(因为 Panama medium 极少且各 ray 方向均匀,ZipNeRF 的强 anti-aliasing grid 优势显现,但 ZipNeRF 训练 6h、FPS 0.9,工程上不可用)。
2. **FPS 41.8** vs SeaThru-NeRF 的 0.07(约 600× 加速),相对 3DGS 的 412.1 仍有差距(因为要 query medium MLP 与计算 exp 衰减),但已达 real-time 标准。
3. **Training 9.4min** 是所有方法中最快,SeaThru-NeRF-NS 是 2h(约 12× 加速)。
4. **3DGS 在水下场景严重失效**(IUI3 PSNR 仅 22.98,比 Ours 低 6.4 dB),证明 medium 建模必要。

### Table 2 Restoration(去 medium)性能

| Method | Foggy-Easy PSNR | Foggy-Hard PSNR |
|---|---|---|
| SeaThru-NeRF-NS | 13.11 | 10.76 |
| **Ours** | **15.70** | **14.06** |

**Intuition**:Ours 在 hard foggy 上比 SeaThru-NeRF 高 3.3 dB,差距比 easy foggy 上更大,说明在 medium 严重时 Ours 优势更明显(因为 SeaThru-NeRF 的 geometry field 容易把 medium 拟合进 surface,造成 wave-like artifact)。

### Table 3 Ablation Study

关键发现:
1. $\mathcal{L}_1 + \mathcal{L}_{DSSIM}$(vanilla 3DGS loss):PSNR 29.219。
2. $\mathcal{L}_{\text{Reg-}L_2} + \mathcal{L}_{\text{Reg-DSSIM}}$(Ours):PSNR 29.687(最佳)。
3. $\mathcal{L}_{\text{Reg-}L_1} + \mathcal{L}_{\text{Reg-DSSIM}}$:29.603(略低于 Reg-L2 版本)。
4. **没有 Reg-DSSIM 时**,$\mathcal{L}_{\text{Reg-}L_2}$ 单独不能 train 3DGS,验证作者的论点:3DGS discrete nature 需要 structural regularization。
5. **w/o Medium** 配置 PSNR 仅 29.353,但 underwater 数据 medium 建模理论上必要 —— 这里数值看似接近是因为 PSNR 在浅水区对 medium 不敏感,但 visual quality(Fig. 3, 4)显示无 medium 会出现明显 artifact。

---

## 7. Limitations 与失败模式

1. **远 distance medium vs background object 难区分**(Fig. 7):远处的 medium 和"看起来像 medium 的 background object"(如远处水面)在 low opacity 区域都难以判别,3DGS 的 prune 机制会把它们都剪掉或都保留。SeaThru-NeRF 也有同样问题。

2. **依赖已知 camera pose**:underwater SfM/SLAM 困难(dome port refraction、feature 少、medium distortion),实际部署需先解决 pose estimation。

3. **Insufficient observation 区域 artifact**(Fig. 8):训练 view 未覆盖的区域,3DGS 会产生 floater / 伪 Gaussian。SeaThru-NeRF 在这些区域是 blur + distortion,本质都是欠拟合但表现不同。

4. **Color restoration 不精确**:在 medium 影响下,object color 与 attenuation $\sigma^{attn}$ 在 optimization 中 entangled,深层物体的真实 color 无法被精确 disentangle(尤其 background-like object)。这物理上是个 ill-posed problem —— 一个观测到的颜色 $I = O \cdot e^{-\beta z}$,$O$ 和 $\beta$ 在同一像素上耦合,需要 spatial smoothness prior 才能解。

---

## 8. 与相关工作的联系与对比

### NeRF 系列
- **NeRF** [Mildenhall et al. 2020]:原始 volumetric radiance field,理论上能建模 medium,但 proposal sampler(Mip-NeRF/ZipNeRF)打破这种能力。https://arxiv.org/abs/2003.08934
- **SeaThru-NeRF** [Levy et al. CVPR 2023]:two fields(geometry + medium),用 image formation model 分离 direct/backscatter。https://arxiv.org/abs/2304.06604
- **Mip-NeRF / Mip-NeRF 360 / ZipNeRF** [Barron et al.]:anti-aliasing 与 grid-based,适合 air scene 但 underwater 退化。
- **NeRF in the Dark** [Mildenhall et al. CVPR 2021]:HDR noisy raw image 渲染,Reg-L2 loss 来源。https://arxiv.org/abs/2111.13679
- **ScatterNeRF** [Ramazzina et al. 2023]:fog 场景的 physically-based inverse neural rendering。https://arxiv.org/abs/2306.15435
- **WaterNeRF** [Sethuraman et al. 2023]:separately estimate medium 参数。https://arxiv.org/abs/2302.10891

### 3DGS 系列
- **3DGS** [Kerbl et al. ACM TOG 2023]:原始 Gaussian splatting,real-time,explicit。https://repo.sam.lgbt/extras/3dgaussian.git
- **Mip-Splatting** [Yu et al. 2023]:3DGS 的 anti-aliasing 版本。https://arxiv.org/abs/2311.16493
- **AbsGS** [Ye et al. 2024]:densification 改进,恢复 fine detail。https://arxiv.org/abs/2404.10484
- **GOF** [Yu et al. 2024]:Gaussian Opacity Fields,surface reconstruction。
- **SplaTAM / SGS-SLAM**:3DGS-based SLAM。

### Underwater Vision
- **SeaThru** [Akkaynak & Treibitz CVPR 2019]:image-level 去水算法。https://arxiv.org/abs/1904.02153
- **Revised Underwater Image Formation Model** [Akkaynak & Treibitz CVPR 2018]:公式 6 的来源,区分 $\beta^D$ 和 $\beta^B$。https://arxiv.org/abs/1709.07262

### Depth Estimation
- **Depth Anything V2** [Yang et al. 2024]:用于生成 Fig. 4, 5 的 GT depth map(因为 SeaThru-NeRF dataset 无 dense GT depth)。https://arxiv.org/abs/2406.09414

---

## 9. Intuition 构建:为什么这个方法 Work

### 9.1 Explicit Geometry + Volumetric Medium 的分工直觉

3DGS 擅长表达 sharp surface(高 $\sigma$ 集中在小范围),但 medium 需要的是 spatially smooth、低密度的 volume —— 用 Gaussian 拟合 medium 会要么 prune 掉(低 opacity),要么变成 muddy cloud artifact。把 medium 剥离到 small MLP + exponential decay 后,3DGS 专注 surface,medium 由 $\exp(-\sigma s)$ 这种解析形式保证 smoothness。

### 9.2 Transmittance 因式分解的几何意义

$T_i(s) = T_i^{obj} T^{med}(s)$ 这个分解的物理含义是:**object 的遮挡是离散事件**(光线遇到 opaque surface 就停),**medium 的衰减是连续过程**(沿 ray 指数衰减)。两者在概率上独立,所以可以相乘。这等价于说:"光线能否到达 depth $s$ 取决于两件事 —— 有没有被 surface 挡住,有被 medium 衰减多少"。

### 9.3 Two $\sigma$ 的物理直觉

direct light(从物体到 camera)与 backscatter light(从环境光被 medium 散射回 camera)的 path 不同:
- Direct 路径:光从物体表面出发,$\lambda$ 路径经过 medium 衰减到达 camera。Attenuation 主要由 absorption + forward scattering 主导,有效 $\beta^D$ 较小。
- Backscatter 路径:光从环境进入 medium,被各向异性散射,只有"恰好朝向 camera"的部分被接收。有效 $\beta^B$ 由后向 scattering coefficient 主导,通常与 $\beta^D$ 不同。

如果用一个 $\sigma$,会强制假设两者相等,导致颜色 reconstruction 失真 —— 这是 SeaThru-NeRF 与 WaterSplatting 共享的关键物理 insight。

### 9.4 为什么 Reg-DSSIM 必要

3DGS 的每个 Gaussian 是 independent 参数,不像 NeRF 是 shared MLP。在 dark region,如果只用 pixel-level Reg-L2,Gaussian 之间没有 structural 耦合,容易产生 disconnected floater 拟合 dark pixel。D-SSIM 在 patch level 引入 structural correlation,迫使相邻 Gaussian 在 dark region 共同形成 coherent structure。这是 3DGS 优化 geometry 时与 NeRF 的本质区别。

### 9.5 离散-连续交错的 alpha compositing 直观图

沿一条 ray,渲染顺序是:
```
[medium attenuation from 0 to s_1]
[Gaussian 1: contributes c_1 * α_1 * exp(-σ_attn * s_1)]
[medium segment from s_1 to s_2: contributes c_med * (exp(-σ_bs*s_1) - exp(-σ_bs*s_2)) * T_1^obj]
[Gaussian 2: contributes c_2 * α_2 * T_1^obj * exp(-σ_attn * s_2)]
...
[after last Gaussian: c_med * exp(-σ_bs * s_N) * T_N^obj = backscatter background]
```

注意 $T_i^{obj}$ 在 medium segment 上保持为 $T_i^{obj}$(因为 medium 不"遮挡",只衰减),但 medium 衰减 $\exp(-\sigma s)$ 在每个 Gaussian 出现的"瞬间"用 $s_i$ 算,在 segment 上用 $[s_{i-1}, s_i]$ 积分。

---

## 10. 可能的 Extension 与联想

1. **Dynamic underwater scene**:当前方法假设 static scene + static medium。真实 underwater 有 moving particles、海流导致 medium 时变。可考虑 4D Gaussian Splatting [deformable] 加 time-varying $\sigma^{med}(t)$。

2. **Fog / Smoke / Haze 场景**:paper 在 simulated fog 上验证,实际应用可推到 driving scene 的 fog removal 或 smoke scene 重建,结合 [ScatterNeRF] 的物理参数化。

3. **SLAM 集成**:水下 ROV/SLAM 需要实时 dense reconstruction + pose estimation。WaterSplatting 的 41.8 FPS 与 explicit representation 适合 [SplaTAM] 框架改造为 underwater SLAM。

4. **Generative underwater scene**:结合 [LucidDreamer] / [GaussianDreamer] 等 3DGS generative 方法,可生成 underwater scene 用于 simulation/data augmentation。

5. **Refraction 建模**:dome port 避免 refraction,但 flat port 水下相机有明显 refraction。可结合 [NeRF with flat port refraction] 工作进一步扩展。

6. **Polarization cue**:水下 polarization 信息能更准确估计 backscatter,可将 polarization 作为 medium MLP 的额外 input,改善 $\sigma^{bs}$ 估计。

7. **Active illumination**:ROV 通常有 headlight,active light 的衰减与 ambient 不同,可引入 active vs ambient 双 path 的 image formation model。

---

## 11. 一些 Critique 与 Open Question

1. **PSNR 提升幅度**:Ours 29.69 vs ZipNeRF 28.77,提升 ~1 dB,但 ZipNeRF 训练 6h,Ours 9.4min。这本质是 efficiency-quality trade-off 的胜利,而非 quality 突破。

2. **medium MLP 的 spatial generalization**:per-pixel query 只用 ray direction 作为 input(加 SH encoding),没有 explicit 3D position。意味着 medium 只随 viewing direction 变化,不能 spatially vary(例如水面附近 vs 深处 water 性质不同)。这是局限 —— 真实 underwater medium 是 3D-varying 的,SeaThru-NeRF 的 medium field 是 3D 的。

3. **depth supervision 缺失**:SeaThru-NeRF dataset 无 dense depth GT,paper 用 Depth Anything V2 生成伪 GT。这引入 supervision noise,可能解释为什么 depth map visual 看似不错但定量比较缺失。

4. **训练 reset 策略**:每次 densification/pruning 都 reset Adam moving average of medium encoding,这是 heuristic,可能 sub-optimal。为什么 medium encoding 必须独立于 densification?

5. **Ablation 缺 medium ablation 的 visual**:w/o Medium 配置 PSNR 仅降 0.3 dB(29.353 vs 29.687),但 visual 应该差异显著。建议补充 visual ablation。

6. **Failure mode 远距离 medium**:作者在 Fig. 7 admit 远距离 medium 与 background 难区分。这本质是因为远距离 transmittance $T^{med}$ 接近 0,signal 弱,任何 method 都难。但可考虑 depth-aware prior 或 temporal consistency 来缓解。

---

## 12. 关键 References Web Links

- **WaterSplatting Project**: https://water-splatting.github.io
- **3DGS (Kerbl et al.)**: https://repo.sam.lgbt/extras/3dgaussian.git ,https://inria.github.io/gaussian-splatting/
- **SeaThru-NeRF (Levy et al.)**: https://arxiv.org/abs/2304.06604 , https://deborahLevy96.github.io/SeaThru-NeRF/
- **SeaThru (Akkaynak & Treibitz)**: https://arxiv.org/abs/1904.02153 , https://www.deryaakaynak.com/seathru
- **Revised Underwater Image Formation Model**: https://arxiv.org/abs/1709.07262
- **NeRF (Mildenhall et al.)**: https://arxiv.org/abs/2003.08934 , https://www.matthewtancik.com/nerf
- **NeRF in the Dark**: https://arxiv.org/abs/2111.13679
- **Mip-NeRF 360**: https://arxiv.org/abs/2111.12055
- **ZipNeRF**: https://arxiv.org/abs/2304.06485
- **Ref-NeRF**: https://arxiv.org/abs/2112.02524
- **Mip-Splatting**: https://arxiv.org/abs/2311.16493
- **AbsGS**: https://arxiv.org/abs/2404.10484
- **GOF**: https://arxiv.org/abs/2404.07206
- **SplaTAM**: https://arxiv.org/abs/2312.02126
- **SGS-SLAM**: https://arxiv.org/abs/2402.02430
- **Langsplat**: https://arxiv.org/abs/2312.16084
- **VastGaussian**: https://arxiv.org/abs/2402.17427
- **LucidDreamer**: https://arxiv.org/abs/2311.13336
- **ScatterNeRF**: https://arxiv.org/abs/2306.15435
- **WaterNeRF**: https://arxiv.org/abs/2302.10891
- **Depth Anything V2**: https://arxiv.org/abs/2406.09414
- **COLMAP**: https://colmap.github.io/
- **NeRFStudio**: https://docs.nerf.studio/
- **Tetra-NeRF**: https://arxiv.org/abs/2304.09987
- **Point-NeRF**: https://arxiv.org/abs/2108.03064
- **TensorRF**: https://arxiv.org/abs/2204.04609
- **InstantNGP**: https://arxiv.org/abs/2201.05989
- **Plenoxels**: https://arxiv.org/abs/2112.05131
- **PlenOctrees**: https://arxiv.org/abs/2103.02450

---

## 总结

WaterSplatting 的核心贡献是把 3DGS 的 discrete alpha compositing 与 medium 的 continuous exponential decay 通过 transmittance 因式分解优雅地融合 —— $T_i(s) = T_i^{obj} T^{med}(s)$ 让 object 与 medium 各司其职。配合 revised underwater image formation model 中的 dual $\sigma$(attenuation vs backscatter),加上 NeRF in the Dark 启发的 HDR-aware regularized loss,实现了 SeaThru-NeRF 同等 quality 的同时 training time 缩短 ~60×、FPS 提升 ~600×,达到 real-time。方法是水下场景重建与介质解耦重建的一个 elegant 工程化解决方案,但 medium spatial variation 有限、远距离 medium 分离、color disentanglement ill-posed 等问题仍是未来工作方向。
