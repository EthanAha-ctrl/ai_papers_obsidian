# Michelson Interferometer 详解

## 一、第一性原理出发：什么是干涉？

要理解 **Michelson Interferometer**，我们必须首先回到物理学的第一性原理——**wave 的 superposition principle（叠加原理）**。

### 1.1 Wave 的本质

Light 本质上是 **electromagnetic wave**。任何 wave 都可以用三个基本量描述：

| 物理量 | 符号 | 物理意义 |
|--------|------|----------|
| Amplitude（振幅） | $A$ | wave 的强度 |
| Frequency（频率） | $f$ 或 $\nu$ | 单位时间内振动次数 |
| Phase（相位） | $\phi$ | wave 在某时刻的振动状态 |

一个单色平面波可以表示为：

$$E(x,t) = A \cdot e^{i(kx - \omega t + \phi_0)}$$

其中：
- $E$：electric field（电场强度）
- $A$：amplitude
- $k = \frac{2\pi}{\lambda}$：wave number（波数），$\lambda$ 是 wavelength
- $\omega = 2\pi f$：angular frequency（角频率）
- $\phi_0$：initial phase（初相位）

### 1.2 Superposition Principle（叠加原理）

当两束 coherent（相干）的 light beam 在空间中相遇时，它们会 **叠加**：

$$E_{total} = E_1 + E_2$$

关键点在于：我们观察到的 **intensity（光强）** 正比于电场振幅的平方：

$$I \propto |E_{total}|^2 = |E_1 + E_2|^2$$

如果两束光来自同一光源（coherent），且存在 **phase difference（相位差）** $\Delta\phi$，则：

$$I = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos(\Delta\phi)$$

这就是 **interference term（干涉项）** 的来源！

---

## 二、Michelson Interferometer 的核心架构

### 2.1 基本结构图解

```
                    Mirror M1（固定或可移动）
                         ↑
                         |
                         |  Beam 2（反射）
                         |
Light Source → [Beam Splitter BS] → Detector/Screen
                    ↓
                    |
                    |  Beam 1（透射）
                    ↓
               Mirror M2（固定参考镜）
```

### 2.2 四个核心组件

| 组件 | 英文名称 | 功能 |
|------|----------|------|
| 光源 | Light Source | 提供 coherent light |
| 分束器 | Beam Splitter (BS) | 将光分成两束（transmission 和 reflection） |
| 反射镜 | Mirrors (M1, M2) | 将光反射回 BS |
| 探测器 | Detector/Screen | 观察干涉 pattern |

### 2.3 光路详解

1. **Light Source** 发出的光到达 **Beam Splitter**（通常是一个 half-silvered mirror，50% 反射，50% 透射）
2. 光被分成两束：
   - **Beam 1**：透射后向 **Mirror M2** 传播，反射后返回 BS
   - **Beam 2**：反射后向 **Mirror M1** 传播，反射后返回 BS
3. 两束光在 BS 处重新 **recombine（重组）**，产生 **interference**
4. 干涉结果由 **Detector** 记录

---

## 三、数学推导：光程差与干涉条件

### 3.1 Optical Path Difference（光程差）

假设：
- M1 和 M2 到 BS 的几何距离分别为 $d_1$ 和 $d_2$
- 两束光都往返，所以光程分别为 $2d_1$ 和 $2d_2$

**Optical Path Difference（OPD）**：

$$\Delta L = 2(d_1 - d_2) = 2\Delta d$$

这里 $\Delta d = d_1 - d_2$ 是两臂的长度差。

### 3.2 Phase Difference（相位差）

光程差 $\Delta L$ 导致相位差：

$$\Delta\phi = \frac{2\pi}{\lambda} \cdot \Delta L = \frac{4\pi \Delta d}{\lambda}$$

### 3.3 干涉条件

| 干涉类型 | 条件 | 物理意义 |
|----------|------|----------|
| **Constructive Interference（相长干涉）** | $\Delta L = m\lambda$，$m = 0, 1, 2, ...$ | 光程差为波长整数倍，两波峰叠加 |
| **Destructive Interference（相消干涉）** | $\Delta L = (m + \frac{1}{2})\lambda$ | 光程差为半波长奇数倍，波峰遇波谷 |

**Intensity 公式**（假设两束光强度相等 $I_1 = I_2 = I_0$）：

$$I = 4I_0 \cos^2\left(\frac{\Delta\phi}{2}\right) = 4I_0 \cos^2\left(\frac{2\pi \Delta d}{\lambda}\right)$$

### 3.4 条纹可见度

定义 **Visibility（可见度）**：

$$V = \frac{I_{max} - I_{min}}{I_{max} + I_{min}}$$

对于理想情况，$V = 1$。实际情况中，由于：
- 光源的非单色性
- 两束光强度不等
- 环境振动

可见度会下降。

---

## 四、Interference Fringes（干涉条纹）的形态

### 4.1 两种典型情况

#### 情况一：M1 严格垂直于入射光

此时产生 **circular fringes（圆环形条纹）**，称为 **fringes of equal inclination（等倾干涉条纹）**。

原因：不同倾角入射的光，光程差不同。

光程差公式（考虑入射角 $\theta$）：

$$\Delta L = 2d\cos\theta$$

其中 $d$ 是两镜的等效距离差。

#### 情况二：M1 稍微倾斜

此时产生 **localized fringes（定域条纹）**，呈 **直线条纹**，称为 **fringes of equal thickness（等厚干涉条纹）**。

类比：Newton's rings、空气劈尖干涉。

### 4.2 条纹移动与镜面位移

当移动 Mirror M1 距离 $\Delta d$ 时，条纹会移动。关系为：

$$\Delta d = N \cdot \frac{\lambda}{2}$$

其中 $N$ 是条纹移动数目。

**这是一个极其精确的测量原理！** 可以用来测量：
- 微小位移（精度可达 $\lambda/100$ 或更高）
- 波长
- 折射率

---

## 五、历史意义：Michelson-Morley Experiment

### 5.1 背景：Luminiferous Aether（以太）假说

19世纪物理学家认为，light 作为 wave，必须在某种 **medium（介质）** 中传播，这种假想介质称为 **Aether**。

### 5.2 实验设计思想

如果 Aether 存在，地球在 Aether 中运动会产生 **Aether wind（以太风）**，类似于骑车时感觉到的风。

Michelson 和 Morley 的设计：
- 将 Michelson Interferometer 的两臂分别平行和垂直于地球运动方向
- 旋转整个装置，观察条纹是否移动

### 5.3 预期结果 vs 实际结果

| 项目 | 预期（Aether 存在） | 实际观测 |
|------|---------------------|----------|
| 条纹移动 | 明显移动 | 无移动！ |

### 5.4 历史意义

这个 **null result（零结果）** 直接否定了 Aether 理论，为 Einstein 的 **Special Relativity（狭义相对论）** 奠定了实验基础。

> "The most famous failed experiment in physics history."

---

## 六、现代应用详解

### 6.1 LIGO（Laser Interferometer Gravitational-Wave Observatory）

**人类历史上最精密的测量仪器！**

| 参数 | 数值 | 物理意义 |
|------|------|----------|
| 臂长 | 4 km | 增大光程差灵敏度 |
| 灵敏度 | $10^{-18}$ m | 小于质子直径的千分之一！ |
| 光源 | Nd:YAG Laser（1064 nm） | 高功率、高稳定性 |

**工作原理**：
- Gravitational wave 经过时，会拉伸一个方向的空间，压缩另一个方向
- 导致两臂长度发生微小变化
- Michelson Interferometer 检测这个变化

**2015年首次探测到 gravitational wave**（来自双黑洞合并），获得 2017 年 Nobel Prize in Physics。

### 6.2 Fourier Transform Spectroscopy（傅里叶变换光谱学）

利用 Michelson Interferometer 可以测量光源的 **spectrum（光谱）**。

**原理**：
1. 扫描 Mirror M1 的位置 $d$
2. 记录 intensity $I(d)$ 作为位置的函数（称为 **interferogram**）
3. 对 interferogram 进行 Fourier Transform：

$$S(\nu) = \int_{-\infty}^{+\infty} I(d) \cdot e^{-i2\pi\nu d} \, dd$$

得到光谱 $S(\nu)$。

### 6.3 Optical Coherence Tomography（OCT）

应用于医学成像（如眼底检查）。

**原理**：利用 **low-coherence light**（宽带光源），只有当两臂光程差在 **coherence length** 以内时才能观察到干涉。

通过扫描参考镜位置，可以探测样品不同深度的反射信号，实现三维成像。

### 6.4 精密测量应用

| 应用领域 | 测量量 | 精度 |
|----------|--------|------|
| 计量学 | 长度标准 | 亚波长级 |
| 光学检测 | 表面平整度 | $\lambda/100$ |
| 气体检测 | 折射率 | $10^{-6}$ RIU |
| 温度传感 | 热膨胀 | 微米级 |

---

## 七、深入技术细节

### 7.1 Coherence Length 和 Coherence Time

实际光源不是严格单色的，具有一定的 **bandwidth（带宽）** $\Delta\lambda$。

**Coherence length**：

$$L_c = \frac{\lambda^2}{\Delta\lambda}$$

只有当光程差小于 $L_c$ 时，才能观察到清晰的干涉条纹。

**Coherence time**：

$$\tau_c = \frac{L_c}{c} = \frac{1}{\Delta\nu}$$

### 7.2 补偿板的作用

在实际装置中，通常加入 **compensating plate（补偿板）**：

原因：
- 光在 Beam Splitter 的玻璃中传播
- Beam 1 和 Beam 2 穿过玻璃的次数不同
- 补偿板使两束光的光程完全相等

### 7.3 偏振效应

如果使用偏振光，需要考虑：
- Beam Splitter 对 s-polarization 和 p-polarization 的反射/透射率不同
- 可能导致两束光振幅不等，降低 fringe visibility

解决方案：使用 **polarizer** 或特殊的 **non-polarizing beam splitter**。

---

## 八、实验数据示例

### 8.1 典型 Michelson Interferometer 参数

| 参数 | 典型值 |
|------|--------|
| 光源波长 $\lambda$ | 632.8 nm（He-Ne laser） |
| 镜面移动范围 | 0-10 cm |
| 条纹计数精度 | 1/100 fringe |
| 测量精度 | ~3 nm |

### 8.2 条纹移动数据表（示例）

| Mirror 位移 $\Delta d$ (μm) | 理论条纹移动数 $N$ | 实测条纹移动数 |
|-----------------------------|---------------------|----------------|
| 10 | 31.6 | 31.5 ± 0.1 |
| 20 | 63.2 | 63.1 ± 0.1 |
| 50 | 158 | 157.8 ± 0.2 |
| 100 | 316 | 315.5 ± 0.5 |

---

## 九、第一性原理视角的 Intuition Building

### 9.1 核心直觉

**Michelson Interferometer 本质上是一个"光程差放大器"和"相位比较器"。**

从第一性原理出发：

1. **Wave 的唯一性**：一个 wave 由 amplitude、frequency、phase 完全确定
2. **Interference 的本质**：phase difference 决定叠加结果
3. **Measurement 的本质**：将不可直接观测的量（如位移、折射率变化）转化为可观测的 intensity 变化

### 9.2 类比理解

把 light beam 想象成两个人在跑步：
- 两人从同一起点出发（coherent source）
- 跑不同的路径（two arms）
- 在终点会合
- 如果两人跑的距离相同，同时到达→**constructive**
- 如果距离差半个波长→一个在波峰，一个在波谷→**destructive**

### 9.3 为什么 Michelson Interferometer 如此精密？

关键在于：
$$\Delta d = N \cdot \frac{\lambda}{2}$$

由于 $\lambda \approx 500$ nm，每移动 250 nm 就产生一个完整的 fringe 移动！

如果我们能分辨 1/100 fringe，精度就是 **2.5 nm**！

---

## 十、常见误区与澄清

### 10.1 "两束光必须来自同一光源"

**正确理解**：两束光必须 **spatially coherent** 和 **temporally coherent**。

- 两个独立的 laser 不能产生稳定的干涉（phase 随机变化太快）
- 但可以用一个 laser 分束后进行干涉

### 10.2 "光程差必须小于波长"

**错误！**

光程差可以远大于波长，只要小于 **coherence length** 即可。

### 10.3 "条纹一定是圆形"

**错误！**

条纹形状取决于镜面配置：
- 等倾干涉→圆环
- 等厚干涉→直线
- 一般情况→混合形态

---

## 参考资源

### 教科书与文献
1. Hecht, E. (2017). *Optics* (5th ed.). Pearson. [经典光学教材]
2. Born, M., & Wolf, E. (1999). *Principles of Optics* (7th ed.). Cambridge University Press. [光学"圣经"]
3. Michelson, A. A., & Morley, E. W. (1887). "On the Relative Motion of the Earth and the Luminiferous Ether". *American Journal of Science*, 34, 333-345. [原始论文]

### 在线资源
- LIGO 官网: https://www.ligo.caltech.edu/
- HyperPhysics (Georgia State University): http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/michel.html
- MIT OpenCourseWare - Optics: https://ocw.mit.edu/courses/physics/
- Nobel Prize - Michelson-Morley Experiment: https://www.nobelprize.org/prizes/physics/1907/michelson/biographical/

### 视频资源
- Veritasium - Michelson-Morley Experiment: https://www.youtube.com/watch?v=j-3azMffnOw
- Sixty Symbols - Interferometers: https://www.youtube.com/watch?v=vLpmvh8-Q_I

---

## 总结

**Michelson Interferometer 的本质**：

$$\boxed{\text{Phase Comparison} \rightarrow \text{Path Difference} \rightarrow \text{Interference Pattern}}$$

它将 **微观的 phase difference** 转化为 **宏观的 intensity variation**，是物理学中最优雅、最精密的测量工具之一。

从否定 Aether 到探测 gravitational wave，这个装置见证了物理学一个多世纪的发展，完美诠释了 **"simple but profound"** 的科学美学。