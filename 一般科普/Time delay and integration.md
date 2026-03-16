# Time Delay and Integration (TDI) 详解

## 一、基本概念

**Time Delay and Integration (TDI)** 是一种**前向运动补偿** 技术，专门用于在低光照条件下捕获运动物体的图像。它是一种特殊的**线扫描** 成像模式，通过将多个线性阵列并排放置，并同步电荷转移与物体运动，实现对同一目标的多重曝光叠加。

### 核心思想

TDI 的本质是**构造性干涉**——将多次独立观测的结果叠加，从而在不牺牲空间分辨率的情况下提高信噪比。这与天文观测中的长时间曝光、斑点成像 和自适应光学 的原理相通。

---

## 二、工作原理与架构

### 2.1 与传统 CCD 的对比

| 类型 | 结构 | 工作方式 | 特点 |
|------|------|----------|------|
| **Staring Array (凝视阵列)** | 2D 面阵 | 整个传感器同时曝光，然后逐行读出 | 类似普通相机，适合静态场景 |
| **Scanning Array (扫描阵列)** | 单线或少数几线 CCD | 依靠机械扫描，逐线成像 | 结构简单，如扫描仪、卫星遥感 |
| **TDI Array (TDI 阵列)** | 多线并行排列 | 电荷转移与物体运动同步 | 低光照下高灵敏度 |

### 2.2 TDI 工作流程

```
物体运动方向 →  →  →  →  →  →
    
    时间 t1:    [Line 1] [Line 2] [Line 3] ... [Line N]
                 ↓ 曝光
    时间 t2:    [Line 1] [Line 2] [Line 3] ... [Line N]
                 电荷转移 →  ↓ 曝光
    时间 t3:    [Line 1] [Line 2] [Line 3] ... [Line N]
                 电荷转移 →  电荷转移 →  ↓ 曝光
    
    最终: 电荷从 Line 1 逐级传递到 Line N，完成 N 次曝光叠加
```

**关键同步条件**：
- 电荷转移速度 = 物体在成像面上的移动速度
- 转移节拍必须精确匹配物体的运动

---

## 三、数学模型与公式推导

### 3.1 信噪比提升

对于 N 级 TDI，信噪比的提升遵循以下关系：

$$\text{SNR}_{\text{TDI}} = \sqrt{N} \cdot \text{SNR}_{\text{single}}$$

其中：
- $\text{SNR}_{\text{TDI}}$ = TDI 模式下的信噪比
- $\text{SNR}_{\text{single}}$ = 单线扫描的信噪比
- $N$ = TDI 级数

**推导**：
假设单次曝光的信号为 $S$，噪声为 $\sigma$（包含光子散粒噪声、读出噪声等），则：

$$\text{SNR}_{\text{single}} = \frac{S}{\sigma}$$

对于 N 次独立曝光叠加：
- 信号累加：$S_{\text{total}} = N \cdot S$
- 噪声累加（假设不相关）：$\sigma_{\text{total}} = \sqrt{N} \cdot \sigma$

因此：
$$\text{SNR}_{\text{TDI}} = \frac{N \cdot S}{\sqrt{N} \cdot \sigma} = \sqrt{N} \cdot \frac{S}{\sigma} = \sqrt{N} \cdot \text{SNR}_{\text{single}}$$

### 3.2 等效曝光时间

$$t_{\text{eq}} = N \cdot t_{\text{line}}$$

其中：
- $t_{\text{eq}}$ = 等效曝光时间
- $t_{\text{line}}$ = 单线曝光时间
- $N$ = TDI 级数

### 3.3 运动同步精度要求

电荷转移时间 $t_{\text{transfer}}$ 必须满足：

$$t_{\text{transfer}} = \frac{d}{v_{\text{image}}}$$

其中：
- $d$ = 相邻 CCD 线之间的间距
- $v_{\text{image}}$ = 物体像在传感器上的移动速度

如果同步误差超过一定阈值，会导致 **MTF (Modulation Transfer Function)** 下降：

$$\text{MTF}_{\text{motion}} = \frac{\sin(\pi \cdot f \cdot \Delta x)}{\pi \cdot f \cdot \Delta x}$$

其中：
- $f$ = 空间频率
- $\Delta x$ = 运动模糊量

---

## 四、技术架构详解

### 4.1 CCD 结构

```
┌─────────────────────────────────────────┐
│  TDI CCD 芯片结构示意                    │
├─────────────────────────────────────────┤
│                                         │
│   ┌─────┬─────┬─────┬─────┬─────┐      │
│   │ L1  │ L2  │ L3  │ ... │ Ln  │ ← 多条并行CCD线      │
│   ├─────┼─────┼─────┼─────┼─────┤      │
│   │ ↓   │ ↓   │ ↓   │     │ ↓   │      │
│   │ P1  │ P2  │ P3  │ ... │ Pn  │ ← 像素              │
│   │ ↓   │ ↓   │ ↓   │     │ ↓   │      │
│   │ ... │ ... │ ... │     │ ... │      │
│   └─────┴─────┴─────┴─────┴─────┘      │
│          ↓ 电荷转移方向                  │
│   ┌───────────────────────┐            │
│   │   读出寄存器          │            │
│   └───────────────────────┘            │
│          ↓                              │
│   ┌───────────────────────┐            │
│   │   输出放大器          │            │
│   └───────────────────────┘            │
│                                         │
└─────────────────────────────────────────┘
```

### 4.2 电荷转移机制

CCD 的电荷转移基于 **MOS 电容结构**：

```
┌──────────────────────────────────┐
│  栅极            │
├──────────────────────────────────┤
│  绝缘层 (SiO₂)                   │
├──────────────────────────────────┤
│  耗尽区                │
│  ┌──────────────────────────┐   │
│  │  电荷包     │   │
│  └──────────────────────────┘   │
├──────────────────────────────────┤
│  衬底                 │
└──────────────────────────────────┘
```

通过控制栅极电压，可以改变势阱深度，驱动电荷包定向移动：

$$\phi_{\text{well}}(x) = -\frac{q \cdot N_A}{2 \epsilon_s} \cdot (x - x_0)^2 + V_g$$

其中：
- $\phi_{\text{well}}$ = 势阱电势
- $q$ = 电子电荷
- $N_A$ = 受主掺杂浓度
- $\epsilon_s$ = 硅介电常数
- $V_g$ = 栅极电压

---

## 五、TDI 的分类

### 5.1 模拟 TDI (Analog TDI)

- 电荷在 CCD 内部直接累加
- 优点：读出噪声低，只需一次读出
- 缺点：灵活性差，需要专门的 TDI CCD 芯片

### 5.2 数字 TDI (dTDI)

- 使用软件算法对多次曝光进行叠加
- 优点：不依赖特定传感器类型，可使用标准 CMOS 或 CCD
- 缺点：多次读出引入额外噪声

**dTDI 算法流程**：

```python
# 简化的 dTDI 算法示意
def digital_tdi(image_sequence, motion_vectors):
    """
    image_sequence: 连续帧图像序列
    motion_vectors: 每帧的运动矢量
    """
    accumulated_image = zeros(image_shape)
    
    for i, (frame, motion) in enumerate(zip(image_sequence, motion_vectors)):
        # 根据运动矢量对齐图像
        aligned_frame = shift(frame, -motion * i)
        # 累加
        accumulated_image += aligned_frame
    
    return accumulated_image
```

---

## 六、关键技术参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| **TDI 级数** | 2, 4, 8, 16, 32, 64, 96, 128 | 级数越高，SNR 提升越大，但对同步要求也越高 |
| **线间距** | 7-14 μm | 决定空间分辨率和运动同步精度 |
| **像素尺寸** | 5-10 μm × 5-10 μm | 影响灵敏度和分辨率 |
| **电荷转移效率** | > 0.99999 | 决定信号损失 |
| **读出噪声** | < 5 e⁻ | 低噪声对 TDI 至关重要 |
| **暗电流** | < 10 pA/cm² | 长时间积分的关键指标 |

### 典型 TDI CCD 性能对比

| 型号 | 级数 | 像素尺寸 | 线数 | 响应率 | 应用领域 |
|------|------|----------|------|--------|----------|
| Teledyne DALSA FT18 | 128 | 7 μm | 1024 | 10 V/μJ/cm² | 工业检测 |
| Hamamatsu S11071 | 96 | 14 μm | 2048 | 14 V/μJ/cm² | 医疗成像 |
| e2v CCD204 | 16 | 12 μm | 4096 | - | 天文观测 |
| ON Semiconductor KAI-29050 | - | 5.5 μm | 2900 万像素 | - | 航空侦察 |

---

## 七、应用领域详解

### 7.1 天文学

**应用场景**：
- 掠射角 X 射线望远镜
- 空间望远镜 (如 Chandra X-ray Observatory)
- 地基望远镜的跟踪成像

**技术挑战**：
- 高能光子对传感器的辐射损伤
- 需要辐射硬化 设计
- 低温工作 (cryogenic operation)

**X 射线 TDI 的特殊考虑**：
$$E_{\text{photon}} = \frac{hc}{\lambda}$$

对于 X 射线 (0.1-10 keV)：
- 单个光子可产生多个电子-空穴对
- 需要防止事件串扰

### 7.2 医学成像

**血管造影**：
- 要求高帧率 + 高灵敏度
- TDI 可在低剂量下获得清晰图像

**乳腺摄影**：
- 剂量敏感型应用
- TDI 有助于降低患者辐射剂量

### 7.3 工业检测

**应用**：
- 印刷电路板检测
- 晶圆缺陷检测
- 快速流水线检测

**优势**：
- 高吞吐量
- 高分辨率
- 低光照适应性强

### 7.4 航空侦察

**特点**：
- 平台高速运动
- 需要前向运动补偿
- 远距离低光照成像

**典型参数**：
- 飞行速度：100-300 m/s
- 高度：1-20 km
- 地面分辨率：0.1-1 m

---

## 八、TDI 的优缺点

### 优点

1. **高灵敏度**：SNR 提升 $\sqrt{N}$ 倍
2. **高分辨率**：线扫描方式可超越像素限制
3. **低噪声**：等效于长时间曝光但无运动模糊
4. **适合连续运动物体**：天然适合扫描成像

### 缺点

1. **同步要求高**：需要精确匹配物体运动
2. **灵活性受限**：主要适用于单向运动场景
3. **复杂度高**：需要精密的时序控制
4. **成本较高**：专用 TDI CCD 芯片昂贵

---

## 九、TDI 与其他技术的对比

| 技术 | 原理 | SNR 提升 | 适用场景 |
|------|------|----------|----------|
| **TDI** | 多线电荷同步累加 | $\sqrt{N}$ | 连续运动物体扫描 |
| **长时间曝光** | 单次长时间积分 | $\sqrt{t}$ | 静态场景 |
| **帧叠加** | 多帧图像平均 | $\sqrt{N}$ | 可对齐的静态场景 |
| **斑点成像** | 短曝光序列重建 | 取决于算法 | 天文成像，消除大气扰动 |
| **自适应光学** | 实时波前校正 | 显著提升 | 地基望远镜 |

---

## 十、实际设计考虑

### 10.1 同步误差分析

同步误差来源：
1. **速度误差**：$v_{\text{actual}} \neq v_{\text{assumed}}$
2. **方向误差**：运动方向与 CCD 线不完全平行
3. **振动**：平台抖动引入随机误差

速度误差导致的 MTF 损失：

$$\text{MTF}_{\text{velocity}}(f) = \frac{\sin\left(\pi \cdot f \cdot N \cdot d \cdot \frac{\Delta v}{v}\right)}{\pi \cdot f \cdot N \cdot d \cdot \frac{\Delta v}{v}}$$

其中：
- $\Delta v$ = 速度误差
- $v$ = 标称速度

### 10.2 光学系统匹配

TDI 系统的光学设计需要考虑：

$$\text{景深} = \frac{2 \cdot N \cdot f^2}{d^2}$$

其中：
- $N$ = 光学 F 数
- $f$ = 焦距
- $d$ = 允许的模糊圆直径

---

## 十一、参考资源

### 学术文献

1. **原始论文与综述**：
   - Janesick, J. R. (2001). *Scientific Charge-Coupled Devices*. SPIE Press. [Link](https://spiedigitallibrary.org/ebooks/PM/Scientific-Charge-Coupled-Devices/eISBN/9780819478356)
   - Holst, G. C. (1998). *CCD Arrays, Cameras, and Displays*. SPIE Press.

2. **TDI 技术论文**：
   - Lareau, A. M. (1993). "Time Delay and Integration: A New Sensor for Reconnaissance". *SPIE Proceedings*. [Link](https://www.spiedigitallibrary.org/conference-proceedings-of-spie)
   - Toshniwal, R. H. (2017). "Time Delay Integration (TDI) Cameras: A Review". *IEEE Sensors Journal*.

3. **天文应用**：
   - Jorden, P. R. et al. (2004). "TDI operation of CCDs for astronomy". *SPIE Proceedings*. [Link](https://www.spiedigitallibrary.org/conference-proceedings-of-spie)

### 制造商资源

- **Teledyne DALSA**: [TDI Line Scan Cameras](https://www.teledynedalsa.com/en/products/imaging/cameras/linescan/)
- **Hamamatsu**: [TDI CCD Sensors](https://www.hamamatsu.com/eu/en/product/type/S11071-01/index.html)
- **e2v**: [Space Imaging Solutions](https://www.teledyne-e2v.com/products/space/)
- **ON Semiconductor**: [CCD Image Sensors](https://www.onsemi.com/products/image-sensors)

### 教程与应用笔记

- Teledyne DALSA: *TDI Imaging: Understanding Time Delay and Integration* [PDF](https://www.teledynedalsa.com/en/learn/knowledge-center/)
- Hamamatsu: *Technical Note on TDI Operation*
- EDN Network: [Understanding TDI for Machine Vision](https://www.edn.com/)

### 相关 Wikipedia 页面

- [Time delay and integration](https://en.wikipedia.org/wiki/Time_delay_and_integration)
- [CCD](https://en.wikipedia.org/wiki/Charge-coupled_device)
- [Push broom scanner](https://en.wikipedia.org/wiki/Push_broom_scanner)
- [Speckle imaging](https://en.wikipedia.org/wiki/Speckle_imaging)

---

## 十二、总结

TDI 是一种精妙的成像技术，通过**时空同步的电荷累加**，在不牺牲空间分辨率的前提下显著提升灵敏度。其核心在于：

1. **物理原理**：利用 CCD 的电荷转移特性，实现对运动目标的连续跟踪成像
2. **数学基础**：SNR 按 $\sqrt{N}$ 提升，等效于延长曝光时间 N 倍
3. **技术实现**：需要精确的机械-电子同步，包括速度匹配和方向对准
4. **应用场景**：从天文观测到医学成像，从工业检测到航空侦察

TDI 的发展趋势包括：
- 更高集成度 (更多级数)
- 更高帧率
- CMOS TDI 的发展 (降低成本)
- 智能 TDI 算法 (自适应同步)