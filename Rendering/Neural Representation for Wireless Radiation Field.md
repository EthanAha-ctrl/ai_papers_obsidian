---
source_pdf: Neural Representation for Wireless Radiation Field.pdf
paper_sha256: d213b3d61a3d4e705b5f47f943b564ca3aa0f78170af3f883d6c24e701e34b98
processed_at: '2026-08-05T22:22:01-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 WRF-GS / WRF-GS+

## 一句话总结

**手机信号在房间里怎么传播的，本质上就像一堆"虚拟小灯泡"在天花板墙壁地板上发光，你在某个位置收到的信号，就是这堆小灯泡光线叠加的结果。这篇paper用3D Gaussian Splatting把房间里的"虚拟小灯泡"学出来，然后就能秒级预测任意位置的信号长啥样。**

---

## 1. 为什么这个问题难

你拿着手机在屋里走，信号强度忽强忽弱，为什么？因为电磁波从TX出发，会撞墙反射、绕过桌角衍射、被书架散射，最后一堆"分身"从四面八方到达你手机天线。每个分身走了不同路径，amplitude衰减不同、phase旋转不同，最后在你天线处complex number叠加。

数学上就是 Eqn.(2):
$$y = Ae^{j\varphi} \sum_{l=0}^{L-1} \Delta A_l e^{j\Delta\varphi_l}$$

- $y$: 收到的信号（复数）
- $Ae^{j\varphi}$: TX发出的原始信号（amplitude $A$ + phase $\varphi$）
- $L$: multipath的路径数量
- $\Delta A_l$: 第 $l$ 条路径的amplitude衰减
- $\Delta\varphi_l$: 第 $l$ 条路径的phase旋转

问题是你不知道 $L$ 是多少、每条路径的 $\Delta A_l, \Delta\varphi_l$ 是多少，环境一变全变。

传统解法：
- **Probabilistic model**（log-distance path loss）: 只用距离猜信号强度，粗得要命，完全不知道信号从哪个方向来
- **Ray tracing**: 用LiDAR扫房间3D点云，然后发射光线模拟反射衍射散射。准，但需要知道墙壁材料反射系数（LiDAR给不了），AND算一个case要7秒
- **NeRF²**: 用NeRF学一个隐式WRF，准但渲染慢（0.2s/sample），cloud gaming要求<20ms，digital twin要求ms级，都扛不住

所以核心痛点：**又快又准又少sample**，三个全要。

---

## 2. 核心insight: Gaussian = Virtual TX

这篇paper最漂亮的idea：

3D-GS在光学里用一堆3D Gaussian来表征场景，每个Gaussian有position、color、opacity。WRF-GS说：**在RF domain里，每个Gaussian就是一个virtual TX（虚拟发射器）**。

为什么这个mapping natural？因为multipath的每条路径，都可以看成是从某个"虚拟源"发出的一条LoS射线。墙壁反射点 = 虚拟TX，桌角衍射点 = 虚拟TX，散射点 = 虚拟TX。你在RX收到的信号 = 所有这些virtual TX发出来的信号叠加。

然后3D-GS的α-blending公式:
$$C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

- $C$: 像素颜色
- $c_i$: 第 $i$ 个Gaussian的color
- $\alpha_i$: 第 $i$ 个Gaussian的opacity
- $\prod_{j<i}(1-\alpha_j)$: 前面所有Gaussian的累积透过率

这个公式本质就是"沿射线按深度排序，逐个叠加贡献"。RF domain里也是一样：沿射线方向，每个virtual TX的信号经过前面所有Gaussian的attenuation后到达RX，叠加。

所以RF版的公式变成：
$$S_i(\mathbf{x}) = \left(\prod_{j=0}^{i-1}\delta(\mathbf{x}_j)\right) S(\mathbf{x}_i)$$
$$R_k = \sum_{i=1}^{N} S_i(\mathbf{x})$$

- $S(\mathbf{x}_i)$: 第 $i$ 个virtual TX发出的信号（复数，有amplitude + phase）
- $\delta(\mathbf{x}_j)$: 第 $j$ 个Gaussian位置的attenuation（复数，表示信号经过这里衰减多少）
- $R_k$: 某个方向上RX收到的总信号

完美对应，物理意义清晰。

参考: [3D Gaussian Splatting原paper](https://repo.samgraph.org/gs/)

---

## 3. 三个必须解决的工程问题

光有idea不够，从光学搬到RF有三个硬核engineering challenge。

### Challenge A: 光只看亮度，RF要看amplitude + phase

光学相机像素记录RGB intensity，实数。RF天线收到的信号是 $s = Ae^{j\varphi}$，复数。如果只建模amplitude丢了phase，multipath的interference（建设性/破坏性叠加）全没了。

**WRF-GS的解法**: Scenario Representation Network输出复数。具体实现上，用Euler公式 $e^{j\varphi} = \cos\varphi + j\sin\varphi$ 把复数拆成real + imaginary两路，CUDA kernel并行算。

为什么不用amplitude + phase两路？因为phase有 $2\pi$ wrap-around问题，做loss的时候会有discontinuity，gradient不稳定。real/imaginary都是smooth连续函数，好优化。

### Challenge B: 相机投影到平面，天线投影到半球

相机是pinhole model，把3D点投影到2D flat image plane。但RX天线阵列接收的是hemisphere方向的信号，最终spatial spectrum是 $360^\circ \times 90^\circ$ 的矩阵（azimuth × elevation）。

**WRF-GS的解法**: 用Mercator projection做桥梁。流程：

1. 天线坐标系 $\mathbf{t} = [t_x, t_y, t_z]^T$ → 球坐标：
$$\Omega_{lon} = \arctan2(t_y, t_x), \quad \Omega_{lat} = \arcsin(t_z / t_r)$$
   - $t_r = \sqrt{t_x^2 + t_y^2 + t_z^2}$: Gaussian中心到天线原点的距离
   - $\Omega_{lon} \in [-\pi, \pi)$: azimuth
   - $\Omega_{lat} \in [0, \pi/2)$: elevation，只看上半球

2. 球坐标 → uniform坐标 $[s_x, s_y] = [\Omega_{lon}/\pi, \Omega_{lat}]$，归一化到 $[-1,1) \times [0,1)$

3. uniform坐标 → pixel坐标 $[p_x, p_y] = [(s_x+1) \cdot W/2, s_y \cdot H]$，$W=360, H=90$

这样就把hemisphere摊平成cylinder再展开成flat plane，可以直接复用3D-GS的tile-based CUDA rasterization。

### Challenge C: 光渲染RGB，RF渲染功率谱

光学splatting输出RGB color，RF要输出spatial power spectrum $\mathbf{P}(\alpha, \beta)$，是信号幅度的平方。

**WRF-GS的解法**: 先用Eqn.(12)(13)算出每个angle的复数信号 $R_k$，然后取模平方 $|R_k|^2$ 就是power。天线阵列的spatial spectrum公式 Eqn.(3):
$$P(\alpha, \beta) = \left|\frac{1}{K}\sum_{m,n} e^{j(\Delta\hat{\theta}_{m,n} - \Delta\theta_{m,n})}\right|^2$$

- $K$: 天线数量（比如 $4\times4=16$）
- $\Delta\hat{\theta}_{m,n}$: 第 $(m,n)$ 天线实测phase
- $\Delta\theta_{m,n}$: 来自 $(\alpha, \beta)$ 方向信号在第 $(m,n)$ 天线的理论phase差
- 对所有方向扫一遍，得到 $360 \times 90$ 的功率矩阵

---

## 4. WRF-GS的网络架构

### Scenario Representation Network（Fig.5）

两个MLP串联：

**MLP1**: 输入3D点位置 $\mathbf{x}$，输出attenuation $\delta(\mathbf{x})$ + feature vector。8层FC，128维，ReLU。学的是**环境本身的attenuation特性**，与TX位置无关。墙壁、桌子、玻璃窗的反射/吸收特性都编码在这里。

**MLP2**: 输入MLP1的feature + TX位置 $P_{TX}$，输出signal $S(\mathbf{x})$。2层FC（128→64），ReLU。学的是**TX在某个位置时，各virtual TX的信号特征**。

为什么这样分？因为环境的attenuation是静态的（桌子不会跑），但signal分布随TX位置剧烈变化（TX移动一米，multipath全变）。分开学，负担小，sample efficiency高。

加上positional encoding（$L=9$）让MLP能学高频细节，loss用L1 + SSIM（$\eta=0.2$）。

### 整体pipeline

```
Random 3D points → MLP1(学环境attenuation) 
                  → MLP2(加TX位置，学signal) 
                  → 得到每个Gaussian的(δ, S)
                  → Mercator投影到2D plane
                  → Electromagnetic splatting(复数α-blending)
                  → Spatial spectrum P(α,β)
                  → 与ground truth比loss
```

参考: [DeepSDF架构灵感](https://arxiv.org/abs/1901.05103), [NeRF positional encoding](https://arxiv.org/abs/2003.08934)

---

## 5. WRF-GS+的两个改进

WRF-GS已经不错，但有两个问题：
1. Static Gaussian无法capture高频信号变化（multipath引起的小尺度fading）
2. 网络要explicit预测 $\delta(\mathbf{x})$ + $S(\mathbf{x})$，参数冗余

### 改进1: Deformable 3D Gaussians

把Gaussian属性拆成static + dynamic两部分：

**Static part**（只由Gaussian中心位置决定）: signal strength、rotation、scaling。对应large-scale fading（path loss、墙壁穿透损耗），TX动一下变化不大。

**Dynamic part**（由deformation network预测）: $\Delta_{sig}, \Delta_{rot}, \Delta_{scal}$。对应small-scale fading（multipath interference），TX动一下剧烈变化。

Deformation network $D_\Theta$:
$$D_\Theta: (G(\mathbf{x}), P_{TX}) \Rightarrow (\Delta_{sig}(\mathbf{x}), \Delta_{rot}(\mathbf{x}), \Delta_{scal}(\mathbf{x}))$$

8层FC + 256维 + ReLU + skip connection（第4层输出和输入concat进第5层，NeRF经典trick防gradient vanishing）。

**直觉**: 这就像把信号分解成"大尺度趋势 + 小尺度扰动"。趋势用静态参数编码（少参数），扰动用deformation network学（有inductive bias，sample efficient）。通信理论里large-scale/small-scale fading本来就是两个不同物理机制，这个分解是物理motivated的。

参考: [Deformable NeRF](https://deformable-cnerf.github.io/), [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/)

### 改进2: 用opacity直接当attenuation权重

WRF-GS+干脆把explicit的 $\delta(\mathbf{x})$ attribute扔了，直接用3D Gaussian自带的opacity当attenuation权重。新的splatting公式：

$$R_k = \sum_{i=1}^{N} (S(\mathbf{x}_i) + \Delta_{sig}(\mathbf{x}_i)) \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

- $S(\mathbf{x}_i)$: 静态signal
- $\Delta_{sig}(\mathbf{x}_i)$: 动态signal offset
- $\alpha_i$: opacity，表示该位置环境的"density"（墙壁density高=信号被挡多，空旷density低=信号透过得好）
- $\prod_{j<i}(1-\alpha_j)$: 前面Gaussian的累积透过率

**直觉**: opacity在光学里是"被挡住的概率"，在RF里对应"信号被环境吸收/scatter掉的energy比例"。虽然从complex attenuation简化成scalar opacity损失了一些表达力，但参数少了、训练快了，AND配合deformable Gaussian的capacity提升，总效果反而更好。

---

## 6. 实验结果一句话版

### Spatial Spectrum Synthesis（实验室，915MHz，4×4天线阵列）

| Method | Median SSIM | 渲染时间 |
|--------|------------|---------|
| WRF-GS+ | **0.90** | 0.008s |
| WRF-GS | 0.82 | 0.005s |
| NeRF² | 0.78 | 0.2s |
| Ray Tracing | 0.38 | 7s |

WRF-GS+比NeRF²准15% AND快25倍。Training data越少WRF-GS+优势越大（sample efficiency高，因为static/dynamic分解的inductive bias强）。

### RSSI Prediction（BLE dataset，15,000 ft²，21个RX）

| Method | Median Error (dB) |
|--------|-------------------|
| WRF-GS+ | **2.4** |
| NeRF² | 3.1 |
| MRI | 8.3 |

WRF-GS+比NeRF²好0.7 dB。注意这里RX是单天线，output从 $360\times90$ matrix变成scalar，但框架照样work，因为static/dynamic decomposition的inductive bias不依赖output形式。

### Downlink CSI Prediction（Argos dataset，104天线，52 subcarriers）

| Method | Median CEA (dB) |
|--------|----------------|
| WRF-GS+ | **23.91** |
| NeRF² | 20.55 |
| FIRE | 15.29 |
| R2F2 | 8.57 |

WRF-GS+比NeRF²高3.36 dB。这里有个巧妙的reformulation: uplink CSI当"input view"，downlink CSI当"target view"，channel prediction问题变成view synthesis问题。因为uplink和downlink经历同一物理环境，uplink CSI可以唯一确定location，通过3D-GS学到的environment representation render出downlink CSI。

参考: [NeRF²](https://dl.acm.org/doi/10.1145/3570361), [FIRE](https://dl.acm.org/doi/10.1145/3447993), [R2F2](https://dl.acm.org/doi/10.1145/2973750.2973760)

---

## 7. 为什么这事儿重要

### 7.1 RF + Vision的cross-pollination

RF和光学都是EM wave，但两个community长期各自发展。这篇paper通过3D-GS这个桥梁，把vision的explicit scene representation技术成功迁移到wireless。这打开了一扇门：vision领域的各种trick（multi-view fusion、dynamic scene、level-of-detail、inverse rendering）都可以试试搬到RF。

未来方向：
- **Dynamic scene**: 人走动引起的channel变化 → Dynamic 3D Gaussians for RF
- **Multi-RX协作**: 多天线阵列协同建模 → Multi-view 3D-GS
- **室外大场景**: 城市级channel modeling → Level-of-detail 3D-GS
- **毫米波/THz**: 高频段multipath更复杂 → Frequency-dependent deformation
- **Inverse rendering**: 从channel measurement反推环境material property → RF inverse rendering

参考: [RFCanvas (SenSys 2024)](https://dl.acm.org/doi/10.1145/3666059), [Winert (ICLR 2023)](https://arxiv.org/abs/2301.10136)

### 7.2 6G + Digital Twin

6G的一个big idea是environment-aware communication：用AI学environment-channel mapping，而非依赖physical model。WRF-GS/WRF-GS+的优势：
- **Low sample complexity**: 几百到几千samples就能学，工程师不用满屋子扫
- **Fast rendering**: ms级latency，满足digital twin实时要求
- **High fidelity**: SSIM 0.9+，CEA 23+ dB
- **Explicit representation**: Gaussian可视化，能看见"虚拟TX"分布在哪，可解释性强，方便debug AND 给通信工程师intuition

参考: [6G roadmap](https://arxiv.org/abs/1905.05138), [Environment-aware communication tutorial](https://arxiv.org/abs/2404.13085)

---

## 8. 我的几点takeaway

### 8.1 Explicit > Implicit

NeRF用MLP隐式编码场景，3D-GS用explicit Gaussian。在RF domain，explicit representation的优势更明显：
- 渲染快（不用沿ray采样点过MLP）
- 可解释（Gaussian = virtual TX，物理意义清晰）
- 可编辑（可以手动加/删/改Gaussian来模拟加个桌子、拆堵墙的效果）

这个trend在vision领域已经发生（3D-GS爆发），RF领域应该会跟上。

### 8.2 Static/Dynamic Decomposition是通用trick

WRF-GS+的static/dynamic分解本质是"大尺度趋势 + 小尺度扰动"的物理先验。这个trick在很多领域都有效：
- 信号处理: trend + seasonal decomposition
- 机器人: global plan + local correction
- Graphics: static scene + dynamic actor

在RF domain对应large-scale fading + small-scale fading，非常natural。这个思路可以推广到其他neural channel modeling工作。

### 8.3 View Synthesis范式统一

这篇paper最深的insight是把channel prediction reformulate成view synthesis：
- Spatial spectrum synthesis = 从TX位置"看"RF场
- CSI prediction = 从uplink CSI"看"downlink CSI

一旦问题变成view synthesis，整个NeRF/3D-GS toolbox都可以用上。这个范式统一的力量很大，未来可能有"GS for everything EM"的trend。

---

## 9. 几个值得吐槽的limitation

### 9.1 Mercator projection在high latitude失真

Mercator在 $\beta \to 90^\circ$（正上方）会无限拉伸，Gaussian在头顶会被严重变形。Paper说"只关心discrete angle所以忽略"，但如果TX在天花板正上方，这个distortion可能导致coverage不准。更好的选择可能是[gnomonic projection](https://en.wikipedia.org/wiki/Gnomonic_projection)或者cube map（6个面各用一次透视投影）。

参考: [OmniGS处理半球投影](https://arxiv.org/abs/2404.03202)

### 9.2 Scalar opacity代替complex attenuation

WRF-GS+用scalar opacity近似complex attenuation，丢了phase information。在弱multipath场景没问题，但在强multipath（比如金属工厂、电梯井）phase interference可能是dominant effect，这个简化可能break。需要更多实验验证。

### 9.3 Static environment assumption

Paper假设环境静止，moving small obstacle用Kalman filter。但人体对915MHz/2.4GHz影响巨大（人体含水率高，是强scatterer），办公室场景人走来走去，这个假设可能不成立。Dynamic scene extension是必须的。

### 9.4 Single frequency

只test了单频点（915MHz、BLE 2.4GHz、Argos某band）。Wideband channel（比如OFDM 100MHz带宽）不同子载波的multipath profile不同，可能需要frequency-dependent deformation network。

---

## 10. 最终直觉

如果让我给一个analogy：

**WRF-GS就是给房间里的每个反射/衍射/散射点放一个"虚拟小灯泡"，用3D Gaussian描述它的位置、亮度、影响范围。手机在不同位置收到的信号，就是这些小灯泡光线叠加的结果。训练就是调这些小灯泡的参数，让预测的信号谱和实测一致。渲染就是把这些小灯泡投影到天线半球上，用α-blending叠加，毫秒级出结果。**

WRF-GS+更进一步：小灯泡分"固定底座"（环境决定的static part）和"可调灯头"（TX位置决定的dynamic part），底座不动只调灯头，sample efficiency高、高频变化capture得好。

代码: [WRF-GS+ GitHub](https://github.com/wenchaozheng/WRF-GSplus)

这个工作的deeper significance: **它证明了vision领域的explicit neural scene representation技术可以直接迁移到RF domain，而且比RF领域自己的方法（ray tracing、NeRF²）都好。** 这只是一个开始，未来几年可能会看到一堆"GS for RF"的变体工作，就像2020年NeRF之后涌现一堆"NeRF for X"一样。

参考链接合集:
- [3D Gaussian Splatting (SIGGRAPH 2023)](https://repo.samgraph.org/gs/)
- [NeRF² (MobiCom 2023)](https://dl.acm.org/doi/10.1145/3570361)
- [NeWRF (ICML 2024)](https://arxiv.org/abs/2402.16120)
- [Winert (ICLR 2023)](https://arxiv.org/abs/2301.10136)
- [Argos channel dataset](https://dl.acm.org/doi/10.1145/2973750.2973754)
- [Deformable NeRF](https://deformable-cnerf.github.io/)
- [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/)
- [SSIM](https://ieeexplore.ieee.org/document/1284395)
- [6G roadmap](https://arxiv.org/abs/1905.05138)
- [Environment-aware communication tutorial](https://arxiv.org/abs/2404.13085)
- [RFCanvas (SenSys 2024)](https://dl.acm.org/doi/10.1145/3666059)
- [WRF-GS+ code](https://github.com/wenchaozheng/WRF-GSplus)

---

# WRF-GS / WRF-GS+ 论文深度讲解

## 1. 核心idea的intuition

这篇paper的key insight非常elegant: **3D-GS中的Gaussian primitive在optical domain表征场景中的particle, 当迁移到RF domain时, 每一个Gaussian primitive就成为一个virtual TX**, 把multipath propagation中那些复杂的reflection/diffraction/scattering路径, 看作是从各个virtual TX发出的"伪LoS信号"在RX处的superposition。

这个mapping很natural, 因为Eqn.(2)中 $y = Ae^{j\varphi}\sum_{l=0}^{L-1}\Delta A_l e^{j\Delta\varphi_l}$ 本质上是 $L$ 条path的线性叠加, AND每条path都可以interpret为从某个virtual source发出的一条LoS射线。3D-GS的α-blending formula $C = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)$ 恰好也是沿ray的有序叠加, 所以可以一一对应。

参考: [3D Gaussian Splatting原paper](https://repo.samgraph.org/gs/), [NeRF²](https://dl.acm.org/doi/10.1145/3570361), [NeWRF](https://arxiv.org/abs/2402.16120)

---

## 2. 三个核心challenge和WRF-GS的解决方案

### Challenge 1: Amplitude + Phase vs. Intensity

Optical 3D-GS只渲染intensity (color), 但RF signal是complex-valued $s = Ae^{j\varphi}$, 必须同时建模amplitude和phase。

**Solution**: Scenario Representation Network输出complex值, $\delta(\mathbf{x}) = \Delta A(\mathbf{x})e^{j\Delta\psi(\mathbf{x})}$ 和 $S(\mathbf{x}) = A(\mathbf{x})e^{j\psi(\mathbf{x})}$。Implementation上, 不拆成amplitude/phase, 而是用Euler formula拆成real + imaginary两路, 在CUDA kernel里并行算。这一点很关键, 因为 $e^{j\varphi}$ 的phase wrap问题会被avoid掉, 同时CUDA kernel可以直接复用原3D-GS的rasterization实现。

### Challenge 2: Camera Model vs. Antenna Model

Camera是pinhole/fisheye投影到一个flat 2D image plane, 但RX antenna array接收hemispherical方向信号, AND最终spatial spectrum是 $360^\circ \times 90^\circ$ 的矩阵 $\mathbf{P}$ (Eqn.(4))。

**Solution**: 用 **Mercator projection**做中间媒介, 流程是:
1. Antenna coordinate $\mathbf{t} = [t_x, t_y, t_z]^T$ → spherical coordinates $\Omega_{lon}, \Omega_{lat}$:
$$\begin{bmatrix}\Omega_{lon}\\ \Omega_{lat}\end{bmatrix} = \begin{bmatrix}\arctan2(t_y, t_x)\\ \arcsin(t_z/t_r)\end{bmatrix}, \quad t_r = \sqrt{t_x^2+t_y^2+t_z^2}$$
   - $\Omega_{lon} \in [-\pi, \pi)$ 是azimuth, $\Omega_{lat} \in [0, \pi/2)$ 是elevation (只考虑upper hemisphere)
   - $\arctan2$ 是4-quadrant inverse tangent, 处理全 $360^\circ$ 范围

2. Spherical → uniform coordinate:
$$[s_x, s_y] = [\Omega_{lon}/\pi, \Omega_{lat}]$$
   归一化到 $[-1, 1) \times [0, 1)$

3. Uniform → pixel coordinate:
$$[p_x, p_y] = [(s_x+1) \cdot W/2, \; s_y \cdot H]$$
   其中 $W=360, H=90$ 对应1度angular resolution。

**Intuition**: Mercator projection的本质是把sphere摊平成cylinder, 这样Gaussian primitive的2D projection就可以在flat tile上做rasterization, 完美复用3D-GS的CUDA pipeline。但Mercator在high latitude有distortion, paper说因为只关心discrete angle的signal, 忽略distortion即可。

### Challenge 3: Splatting from Color to Signal Power

Optical splatting渲染RGB color, 但RF domain要渲染spatial power spectrum $\mathbf{P}(\alpha, \beta) = |\cdot|^2$ (Eqn.(3))。

**Solution**: Electromagnetic splatting, 关键formula:
$$S_i(\mathbf{x}) = \left(\prod_{j=0}^{i-1}\delta(\mathbf{x}_j)\right) S(\mathbf{x}_i) \tag{12}$$
$$R_k = \sum_{i=1}^{N} S_i(\mathbf{x}) \tag{13}$$

变量含义:
- $S(\mathbf{x}_i)$: 第 $i$ 个Gaussian (virtual TX)的complex signal
- $\delta(\mathbf{x}_j)$: 第 $j$ 个Gaussian的complex attenuation, 对应optical中的opacity
- $i-1$ 的product: 沿ray的累积attenuation, 等价于optical中的 $\prod_{j<i}(1-\alpha_j)$
- $R_k$: 像素 $k$ (即某个angle)上的总received signal

这里有个subtle点: optical中opacity是 $[0,1]$ 之间的实数, 表示"被遮挡的概率"; 而RF中 $\delta$ 是complex number, 表示signal经过该Gaussian位置的attenuation (amplitude shrinkage + phase rotation)。最后 $|R_k|^2$ 就是spatial spectrum在那个angle的power。

---

## 3. WRF-GS架构详解

### 3.1 Scenario Representation Network (Fig.5)

两个MLP串联:
- **MLP1**: 8层FC, 128 channels, ReLU activation。Input是3D point position $\mathbf{x}$, Output是attenuation $\delta(\mathbf{x})$ AND一个feature vector。这个MLP学的是**location-dependent environmental information**, 与TX位置无关。
- **MLP2**: 2层FC (128→64), ReLU。Input是MLP1的feature vector AND $P_{TX}$, Output是signal $S(\mathbf{x})$。这个MLP学的是**TX-dependent signal characteristics**。

整体映射: $F_\Theta: (G(\mathbf{x}), P_{TX}) \Rightarrow (\delta(\mathbf{x}), S(\mathbf{x}))$

**Intuition**: 这个decomposition对应于通信理论中的small-scale fading (随TX变化剧烈) AND large-scale fading (随environment结构稳定)。MLP1学环境(静态), MLP2学TX-specific propagation(动态)。但WRF-GS这个design有redundancy, 因为 $\delta(\mathbf{x})$ AND $S(\mathbf{x})$ 都需要explicitly预测, 参数量大。这成为WRF-GS+改进的motivation。

### 3.2 Position Encoding

用NeRF经典的positional encoding:
$$\gamma(\mathbf{t}) = (\sin(\pi\mathbf{t}), \cos(\pi\mathbf{t}), \ldots, \sin(2^L\pi\mathbf{t}), \cos(2^L\pi\mathbf{t}))$$

- $\mathbf{t}$: 3D坐标
- $L$: encoding order, 取 $L=9$
- 这是高频信号capture的关键, 否则MLP只能学到smooth low-frequency结构

### 3.3 Loss Function

$$\mathcal{L} = (1-\eta)|I_{gt} - I_{pred}| + \eta(1 - \xi(I_{gt}, I_{pred}))$$

- $\eta = 0.2$: weighting factor
- $\xi(\cdot, \cdot)$: SSIM (Structural Similarity Index Measure)
- L1 loss占80%, SSIM loss占20%

**Intuition**: 用图像质量评估的SSIM来评估spatial spectrum, 因为spatial spectrum本质上就是一个image, 用SSIM可以capture结构相似性(peak位置、能量分布形状), 比纯MSE更好。这个insight从NeRF²继承过来。

参考: [SSIM paper](https://ieeexplore.ieee.org/document/1284395)

---

## 4. WRF-GS+的两个核心改进

### 4.1 Deformable 3D Gaussians

WRF-GS的limitation: static 3D Gaussians无法capture high-frequency signal variation (由multipath引起的小尺度fading)。

**Solution**: 把Gaussian属性分解为static + dynamic:
- **Static components** (由Gaussian中心位置唯一决定): signal strength (原color attribute), rotation, scaling → 表征large-scale fading
- **Dynamic components** (由deformation network预测): $\Delta_{sig}, \Delta_{rot}, \Delta_{scal}$ → 表征small-scale fading

Deformation network $D_\Theta$:
$$D_\Theta: (G(\mathbf{x}), P_{TX}) \Rightarrow (\Delta_{sig}(\mathbf{x}), \Delta_{rot}(\mathbf{x}), \Delta_{scal}(\mathbf{x})) \tag{14}$$

Network architecture (Fig.9):
- 8层FC, 256 hidden dim, ReLU → 256-dim feature vector
- 类似NeRF的skip connection: 第4层output和input concat后feed进第5层
- 最后3个FC (no activation)分别output 3个offset

**Intuition**: 这借鉴了[Deformable NeRF](https://deformable-cnerf.github.io/)和[Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/)的思想, 用一个deformation field来表征时变signal。在RF场景下, "time-varying"对应TX移动引起的multipath变化, AND这个deformation只影响small-scale部分, large-scale部分保持static, 这样网络学习负担更小, sample efficiency更高。

### 4.2 α-blending替代explicit attenuation

WRF-GS+放弃了explicit的attenuation attribute $\delta(\mathbf{x})$, 而是直接复用3D Gaussian的**opacity**作为attenuation weight。

新的splatting formula:
$$R_k = \sum_{i=1}^{N} (S(\mathbf{x}_i) + \Delta_{sig}(\mathbf{x}_i)) \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \tag{15}$$

变量含义:
- $S(\mathbf{x}_i)$: 静态signal
- $\Delta_{sig}(\mathbf{x}_i)$: 动态signal offset
- $\alpha_i$: 第 $i$ 个Gaussian的opacity, 表征该位置的"环境density"
- $\prod_{j<i}(1-\alpha_j)$: 前面Gaussian的累积阻挡

**Intuition**: 这把RF attenuation完全统一到optical的α-blending framework里。Opacity在optical中是"被挡住的概率", 在RF中对应"信号被environment吸收/scatter掉的energy比例"。物理意义有subtle差异: optical的opacity是[0,1]的概率, RF的attenuation是complex amplitude; WRF-GS+做了simplification, 用opacity标量来近似complex attenuation, 但配合α-blending的累积乘积效果, 可以学到一个等效的attenuation profile。这个简化让模型参数减少, 训练更快, 同时因为deformable Gaussians的引入, 模型capacity反而增加, 所以accuracy更高。

---

## 5. 实验数据详解

### 5.1 Spatial Spectrum Synthesis (Lab Environment, 915MHz, 4×4 antenna array)

**SSIM CDF (Fig.10)**:

| Method | Median SSIM | 90th percentile |
|--------|------------|----------------|
| WRF-GS+ | **0.90** | **0.95** |
| WRF-GS | 0.82 | 0.88 |
| NeRF² | 0.78 | 0.86 |
| VAE | 0.70 | 0.82 |
| DCGAN | 0.56 | 0.69 |
| Ray Tracing | 0.38 | 0.61 |

**Intuition**: WRF-GS+比WRF-GS提升8% median SSIM, 主要来自deformable Gaussians对high-frequency multipath variation的建模能力。NeRF²差是因为voxel sampling rate的trade-off — high resolution需要海量training data AND computation, low resolution丢失WRF细节。Ray tracing差是因为需要environment material properties (reflection coefficient等), 而LiDAR point cloud只有geometry没有material。

### 5.2 Sample Efficiency (Fig.11)

在不同training dataset size (200-4800 samples)下:
- WRF-GS+比WRF-GS高3%-13% median SSIM
- WRF-GS+比NeRF²高11%-19% median SSIM
- **Training data越少, gap越大** — WRF-GS+的sample efficiency优势更显著

**Intuition**: 这是因为static/dynamic decomposition引入了强inductive bias — environment的static部分只需要少量data就能学, dynamic部分通过deformation network参数化, 也比从零学整个WRF sample-efficient得多。

### 5.3 Rendering Time (Fig.12)

| Method | Time per sample |
|--------|----------------|
| WRF-GS | 0.005s |
| WRF-GS+ | 0.008s |
| VAE | 0.05s |
| DCGAN | 0.09s |
| NeRF² | 0.2s |
| Ray Tracing | 7s |

WRF-GS+虽然比WRF-GS慢3ms (deformation network overhead), 但比NeRF²快25倍, 比ray tracing快875倍。这对应cloud gaming <20ms AND digital twin <ms级的latency要求。

---

## 6. Case Study I: RSSI Prediction

### Setup
- Dataset: public BLE dataset (NeRF² paper提供)
- Scene: 15,000 ft², 21个RX, 1个mobile TX
- 6,000 measurements, 2,000 valid per RX, 1,600 train + 400 test
- RX是single omnidirectional antenna (不是array), 输出是scalar power而非 $90\times360$ matrix
- Invalid measurements (TX太远): RSSI = -100 dBm

### Results (Fig.13)

| Method | Median Error (dB) | 10th pct | 90th pct |
|--------|-------------------|----------|----------|
| WRF-GS+ | **2.4** | 1.1 | 4.6 |
| WRF-GS | 2.9 | - | - |
| NeRF² | 3.1 | - | - |
| MRI | 8.3 | - | - |

WRF-GS+比NeRF²好0.7 dB, 比MRI好5.9 dB。MRI基于log-distance path-loss model, 是parametric方法, 无法capture multipath结构, 所以差是预期的。

**Intuition**: WRF-GS+的static/dynamic decomposition在single antenna场景下也有效, 因为它本质上capture的是signal的large-scale (TX-RX距离相关) AND small-scale (multipath)两个分量, 即使output从matrix变成scalar, decomposition的inductive bias仍然helpful。

参考: [MRI paper](https://ieeexplore.ieee.org/document/6835080)

---

## 7. Case Study II: Downlink CSI Prediction

### Problem Setup
- FDD MIMO系统中, uplink和downlink用不同frequency band, 没有reciprocity
- 目标: 从uplink CSI预测downlink CSI, 避免大量feedback overhead
- Dataset: Argos channel dataset, 104-antenna BS, 52 subcarriers (前26 uplink, 后26 downlink)

### Adaptation

WRF-GS+把Eqn.(14)改成:
$$D_\Theta: (G(\mathbf{x}), I_u(\mathbf{x})) \Rightarrow (\delta_{sig}(\mathbf{x}), \delta_r(\mathbf{x}), \delta_s(\mathbf{x})) \tag{19}$$

- Input: 从 $P_{TX}$ 改成uplink CSI $I_u(\mathbf{x})$
- Output: downlink CSI (而非spatial spectrum)

**Intuition**: 这里有个idea值得强调, 即uplink CSI是location的"fingerprint", AND uplink/downlink经历同一物理环境, 所以uplink CSI可以唯一确定一个location, AND通过3D-GS学到的environment representation, 可以render出downlink CSI。这个pipeline其实把"channel prediction"问题reformulate成了"view synthesis"问题 — uplink CSI是"input view", downlink CSI是"target view"。

### Results (Fig.15)

Metric: CEA (Channel Estimation Accuracy) in dB:
$$\text{CEA} = -10\log_{10}\left(\frac{\|C_{pred} - C_{GT}\|^2}{\|C_{GT}\|^2}\right) \tag{20}$$

| Method | Median CEA (dB) | 10th pct | 90th pct |
|--------|----------------|----------|----------|
| WRF-GS+ | **23.91** | 20.85 | 25.52 |
| WRF-GS | 22.98 | 20.82 | 24.33 |
| NeRF² | 20.55 | - | - |
| FIRE | 15.29 | - | - |
| R2F2 | 8.57 | - | - |
| OptML | 8.47 | - | - |

WRF-GS+比NeRF²高3.36 dB, 比FIRE高8.62 dB, 比R2F2/OptML高15+ dB。

**Intuition**: R2F2/OptML这类方法基于parametric multipath model (估计path parameters), 但实际environment的multipath远比parametric model能描述的复杂。FIRE用VAE学latent distribution, 但VAE的latent space没有显式的3D structure inductive bias。NeRF²和WRF-GS系列利用了3D scene structure作为prior, 所以显著better。WRF-GS+比NeRF²好的原因: explicit 3D Gaussian + deformable + α-blending比implicit NeRF的voxel sampling更高效AND表达力更强。

参考: [FIRE](https://dl.acm.org/doi/10.1145/3447993), [R2F2](https://dl.acm.org/doi/10.1145/2973750.2973760), [OptML](https://dl.acm.org/doi/10.1145/3349623.3349628)

---

## 8. 几个值得深挖的技术细节

### 8.1 Complex Number在CUDA中的处理

Paper提到用Euler formula把complex拆成real + imaginary两路, 而非amplitude + phase。这一点engineering上很关键:
- Phase有 $2\pi$ wrap-around问题, 用phase直接做loss会有discontinuity
- Real/Imaginary都是连续smooth function, gradient descent稳定
- CUDA kernel可以直接处理两个channel, 复用3D-GS的rasterization代码

### 8.2 为什么3D-GS比NeRF²快这么多

NeRF²需要沿每个pixel ray做voxel sampling, 渲染 $360 \times 90$ spatial spectrum需要cast $32400$ 条ray, 每条ray采样 $N$ 个点, 每个点过MLP, 计算量是 $O(W \cdot H \cdot N \cdot \text{MLP cost})$。

3D-GS是explicit representation, 每个Gaussian直接project到image plane, 只需要sort + α-blend, 计算量是 $O(\text{Num\_Gaussians} \cdot \text{tile\_size}^2)$, AND有高度优化的CUDA implementation。所以快1-2个数量级是符合预期的。

参考: [3D-GS CUDA implementation](https://github.com/graphdeco-inria/gaussian-splatting)

### 8.3 Random Point Initialization vs LiDAR

WRF-GS paper的之前版本[1]用LiDAR point cloud初始化3D Gaussians, 但这个版本改成random initialization + adaptive density control。这个change的影响:
- 优点: 不需要multimodal input, LiDAR quality不再影响结果
- 缺点: 训练时间可能更长, 因为需要从random start学geometry
- Adaptive density control: 训练中根据gradient大小split/clone/prune Gaussians, 自动学习Gaussian的spatial distribution

参考: [Adaptive density control details](https://repo.samgraph.org/gs/)

### 8.4 Deformation Network的skip connection

Fig.9提到"concatenate feature vector from 4th layer with input AND feed into 5th layer", 这是NeRF的经典trick, 目的是让deep layer也能access low-frequency input information, 避免gradient vanishing AND保留fine detail。

---

## 9. 与相关工作的broader context

### 9.1 Neural Channel Modeling的evolution

1. **Probabilistic models**: log-distance path loss, Okumura-Hata — 只给RSSI, 没有spatial structure
2. **Deterministic models**: Ray tracing — 精确但需要material info AND expensive
3. **Neural models (NeRF-based)**: NeRF², NeWRF, Winert — implicit WRF, 慢
4. **Neural models (3D-GS-based)**: WRF-GS/WRF-GS+ — explicit WRF, 快AND准

### 9.2 RF + Vision的cross-pollination

这个paper是一个很好的example, 展示了computer vision技术如何迁移到wireless communications:
- Optical: camera → image
- RF: antenna array → spatial spectrum
- Optical: NeRF → radiance field
- RF: NeRF² → wireless radiation field (WRF)
- Optical: 3D-GS → explicit Gaussian scene
- RF: WRF-GS → explicit Gaussian virtual TXs

未来可能的extension:
- **Dynamic scene**: 人体移动引起的channel变化, 对应Dynamic 3D Gaussians
- **Multi-RX**: 多个RX协作, 类似multi-view 3D-GS
- **Large-scale outdoor**: 城市级channel modeling, 需要level-of-detail
- **毫米波/THz**: 高频段multipath更复杂, 可能需要frequency-dependent deformation

参考: [Winert](https://arxiv.org/abs/2301.10136), [NeRF²](https://arxiv.org/abs/2308.07657), [RFCanvas](https://dl.acm.org/doi/10.1145/3666059)

### 9.3 6G AND Environment-Aware Communication

这个paper的big picture是6G的"environment-aware communication" — 用AI学environment-channel mapping, 而非依赖physical model。WRF-GS/WRF-GS+的优势:
- **Low sample complexity**: 几百到几千samples就能学
- **Fast rendering**: ms级latency, 适合digital twin AND real-time application
- **High fidelity**: SSIM 0.9+, CEA 23+ dB
- **Explicit representation**: Gaussian可视化, 可解释性强

参考: [6G roadmap](https://arxiv.org/abs/1905.05138), [Environment-aware communication tutorial](https://arxiv.org/abs/2404.13085)

---

## 10. 几点critique AND open questions

### 10.1 Mercator projection的distortion

Mercator projection在high latitude ($\beta \to 90^\circ$)有严重distortion, 即Gaussian在high elevation会被拉伸。Paper说"ignore distortion because we only care about discrete angles", 但这可能在high elevation场景下导致Gaussian coverage不准。一个可能的improvement: 用[gnomonic projection](https://en.wikipedia.org/wiki/Gnomonic_projection)或者cube map projection。

### 10.2 Complex attenuation的简化

WRF-GS+用scalar opacity替代complex attenuation $\delta(\mathbf{x})$, 这是一个simplification。在强multipath场景下, complex attenuation的phase information可能重要。不过实验结果显示WRF-GS+仍然比WRF-GS好, 说明这个简化在test的场景下是reasonable的。但在更复杂的场景(如大量metal reflector)可能需要重新评估。

### 10.3 Static environment assumption

Paper assume environment stationary, moving small obstacle用Kalman filter处理。但实际场景中人体移动对RF影响很大, 这个assumption可能在office/home场景下不成立。如何model dynamic environment是open question。

### 10.4 Single frequency limitation

Paper只test了915MHz (lab) AND BLE (2.4GHz) AND Argos (specific band)。Wideband channel可能需要frequency-dependent deformation network, 这是natural的extension方向。

---

## 11. 总结

这篇paper的核心贡献是:
1. **第一次把3D-GS迁移到RF domain**, 建立了Gaussian primitive ↔ virtual TX的mapping
2. **设计了三个适配模块**: scenario representation network (complex output), Mercator projection (hemisphere to flat), electromagnetic splatting (complex α-blending)
3. **WRF-GS+的deformable Gaussian + α-blending**, 在accuracy AND efficiency上都超越NeRF-based方法
4. **两个practical case study**, 验证了method在RSSI AND CSI prediction上的effectiveness

这个工作的deeper意义在于: **RF AND optical都是EM wave, 但长期以来两个community各自发展, paper通过3D-GS这个桥梁, 展示了vision技术对wireless的巨大潜力**, 这只是一个开始, 未来可能看到一个"NeRF/3D-GS for everything EM"的trend。

代码: [WRF-GS+ GitHub](https://github.com/wenchaozheng/WRF-GSplus)

参考链接:
- [3D Gaussian Splatting](https://repo.samgraph.org/gs/)
- [NeRF² (MobiCom 2023)](https://dl.acm.org/doi/10.1145/3570361.3570369)
- [NeWRF (ICML 2024)](https://arxiv.org/abs/2402.16120)
- [Winert (ICLR 2023)](https://arxiv.org/abs/2301.10136)
- [Argos channel dataset](https://dl.acm.org/doi/10.1145/2973750.2973754)
- [DeepSDF](https://arxiv.org/abs/1901.05103)
- [Deformable NeRF](https://deformable-cnerf.github.io/)
- [Dynamic 3D Gaussians](https://dynamic3dgaussians.github.io/)
