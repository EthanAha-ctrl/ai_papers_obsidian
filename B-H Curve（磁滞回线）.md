## B-H Curve（磁滞回线）详解

### 一、基本定义

B-H Curve，也称为**Magnetic Hysteresis Loop**或**Magnetization Curve**，是描述**ferromagnetic material**（铁磁材料）中**magnetic flux density B**（磁通密度，单位：Tesla）与**magnetic field strength H**（磁场强度，单位：A/m）之间非线性关系的曲线。核心关系式为：

$$B = \mu_0(H + M)$$

其中：
- $\mu_0 = 4\pi \times 10^{-7}$ H/m 是**vacuum permeability**（真空磁导率）
- $M$ 是**magnetization**（磁化强度，单位：A/m），表示单位体积内的**magnetic moment**

---

### 二、曲线的几何特征与关键参数

#### 2.1 Initial Magnetization Curve（初始磁化曲线）
当材料从**demagnetized state**（H = M = 0）开始施加**increasing H field**（递增磁场）时，M follows the **initial magnetization curve**。曲线特征：
- **Rapid increase**（急剧上升）：初期磁化强度快速增加
- **Asymptotic approach**（渐近逼近）：最终接近**magnetic saturation**（磁饱和）

#### 2.2 Hysteresis Loop（磁滞回线）的关键参数

```
        Bsat
         ^
         |    ↗︎  (saturation point)
         |   /|
         |  / |
         | /  |
   Br    |/   |
---------+----|-----------> H
        /|    |   Hc
       / |    |
      /  |    |
     /   |    |
         |    |
        -Bsat
```

| 参数 | 符号 | 物理意义 | 公式/定义 |
|------|------|----------|-----------|
| **Saturation Flux Density** | $B_{sat}$ | 磁饱和时的最大磁通密度 | $B_{sat} = \mu_0(H_{sat} + M_{sat})$ |
| **Remanence** / **Retentivity** | $B_r$ 或 $M_r$ | H = 0时残留的磁化 | 外加磁场为0时的B值 |
| **Coercivity** | $H_c$ | 使B = 0所需反向磁场 | 磁滞回线与H轴的交点 |
| **Maximum Energy Product** | $(BH)_{max}$ | 磁体储能能力的指标 | $\max[B(H) \cdot H]$ |

---

### 三、物理机制：Domain Theory（磁畴理论）

#### 3.1 Magnetic Domains（磁畴）
在**ferromagnetic material**内部，**atomic magnetic dipoles**（原子磁偶极子）自发排列形成**magnetic domains**：
- **Single-domain magnets**（单畴磁体）：小型磁体，磁化通过**rotation**（旋转）响应
- **Multi-domain magnets**（多畴磁体）：大型磁体，分为多个磁畴，通过**domain wall motion**（畴壁运动）响应

#### 3.2 Domain Walls（畴壁）
- **Domain wall**是磁化方向旋转的过渡区域
- 当H变化时，**domain wall motion**改变各磁畴的相对大小
- **Nucleation**（成核）和**denucleation**（去成核）过程改变磁畴数量

#### 3.3 Barkhausen Jumps（巴克豪森跳跃）
在magnetization curve上观察到的**random jumps**（随机跳跃），源于：
- **Crystallographic defects**（晶体缺陷）
- **Dislocations**（位错）
- 畴壁克服能垒的突然运动

---

### 四、数学模型

#### 4.1 Preisach Model
最著名的**empirical hysteresis model**（经验磁滞模型）：
- 基于大量**elementary hysteresis operators**（基本磁滞算子）
- 能够准确模拟磁滞回线
- 局限：**scalar model**（标量模型），缺乏热力学基础

#### 4.2 Jiles-Atherton Model
广泛应用于工业的模型：
- 基于**domain wall pinning**（畴壁钉扎）概念
- 描述**reversible**（可逆）和**irreversible**（不可逆）磁化过程
- 参数包括：$M_s$（饱和磁化强度）、$a$（形状参数）、$\alpha$（平均场系数）、$k$（钉扎强度）、$c$（可逆系数）

#### 4.3 VINCH Model（Vectorial Incremental Nonconservative Consistent Hysteresis）
较新且**thermodynamically consistent**（热力学一致）的模型：
- 基于Landau-Lifshitz-Gilbert方程
- **Variational formulation**：所有内变量来自thermodynamic potential的最小化
- 提供**stored magnetic energy**和**dissipated energy**的实时信息

#### 4.4 Stoner-Wohlfarth Model
**Physical model**（物理模型）解释磁滞：
- 基于**magnetic anisotropy**（磁各向异性）
- 每个**crystalline grain**（晶粒）有**easy axis**（易轴）和**hard axis**（难轴）
- 解释**coherent rotation**（一致旋转）机制

---

### 五、测量方法

#### 5.1 Open-Circuit Measurement（开路测量）
- 使用**vibrating-sample magnetometer (VSM)**
- 样品悬浮在自由空间
- 需要修正**demagnetizing field**（退磁场）
- 内部H与applied H不同

**Demagnetizing factor修正**：
$$H_{internal} = H_{applied} - N \cdot M$$

其中$N$是**demagnetizing factor**，取决于样品几何形状。

#### 5.2 Closed-Circuit Measurement（闭路测量）
- 使用**hysteresis graph**
- 样品flat faces直接与electromagnet的pole faces接触
- **Demagnetizing field被消除**
- $H_{internal} = H_{applied}$

#### 5.3 测量原理
施加变化的H field → 测量induced **electromotive force (EMF)** → 通过**Faraday's Law**计算B：

$$\varepsilon = -\frac{d\Phi}{dt} = -N \cdot A \cdot \frac{dB}{dt}$$

其中：
- $\varepsilon$：感应电动势
- $\Phi$：磁通量
- $N$：coil匝数
- $A$：sample截面积

---

### 六、材料分类与应用

#### 6.1 Hard Magnets（硬磁材料）
**特征**：
- **High coercivity**（高矫顽力，$H_c > 10^4$ A/m）
- **High remanence**
- **High energy product** $(BH)_{max}$

**应用**：
- **Permanent magnets**（永磁体）：NdFeB、SmCo、Ferrite
- **Magnetic recording**（磁记录）：hard disks, magnetic tape
- **Credit cards**（信用卡）：magnetic stripe
- **Motors and generators**（电机和发电机）

| 材料 | $H_c$ (kA/m) | $B_r$ (T) | $(BH)_{max}$ (kJ/m³) |
|------|-------------|-----------|-------------------|
| NdFeB | 800-2000 | 1.0-1.4 | 200-450 |
| SmCo | 500-1500 | 0.8-1.1 | 120-200 |
| Ferrite | 150-300 | 0.2-0.4 | 20-40 |

#### 6.2 Soft Magnets（软磁材料）
**特征**：
- **Low coercivity**（低矫顽力，$H_c < 10^2$ A/m）
- **High permeability**
- **Low hysteresis loss**

**应用**：
- **Transformer cores**（变压器铁芯）：silicon steel, amorphous metal
- **Electromagnets**（电磁铁）：soft iron, permalloy
- **Inductors**（电感器）
- **Magnetic shielding**（磁屏蔽）

| 材料 | $H_c$ (A/m) | $\mu_r$ | Loss (W/kg @ 1T, 50Hz) |
|------|-------------|---------|------------------------|
| Silicon steel | 10-50 | 5000-10000 | 0.5-1.0 |
| Permalloy | 0.1-1 | 20000-100000 | <0.1 |
| Ferrite | 5-20 | 200-5000 | 0.01-0.1 |

#### 6.3 Hysteresis Loss（磁滞损耗）
每个磁滞循环的能量损耗等于loop的**面积**：

$$W_{hysteresis} = \oint B \, dH$$

对于sinusoidal excitation：
$$P_{hysteresis} = k_h \cdot f \cdot B_m^n$$

其中：
- $k_h$：材料常数
- $f$：频率
- $B_m$：磁通密度峰值
- $n$：Steinmetz指数，通常1.6-2.0

---

### 七、特殊现象与扩展

#### 7.1 Minor Loops（小环）
当H field变化不超过full loop范围时形成的**partial hysteresis loops**：
- 在AC磁化中尤为重要
- **Minor loop area**用于计算partial loss

#### 7.2 Differential Permeability（微分磁导率）
$$\mu_{diff} = \frac{dB}{dH}$$

在磁滞回线上逐点变化的斜率，表征材料对小幅磁场变化的响应。

#### 7.3 Temperature Dependence（温度依赖性）
- **Curie temperature**（居里温度）$T_c$：低于此温度为ferromagnetic，高于此温度为**paramagnetic**
- 在$T_c$附近，coercivity和remanence急剧下降

#### 7.4 Frequency Dependence（频率依赖性）
高频下出现**dynamic hysteresis**，包含：
- **Eddy current loss**（涡流损耗）
- **Excess loss**（异常损耗）

总损耗公式：
$$P_{total} = P_{hysteresis} + P_{eddy} + P_{excess}$$

---

### 八、实验数据示例

典型的**silicon steel transformer core**的B-H curve数据：

| H (A/m) | B (T) | $\mu_r$ |
|---------|-------|---------|
| 0 | 0.0 | - |
| 10 | 0.1 | 7958 |
| 50 | 0.5 | 7958 |
| 100 | 1.0 | 7958 |
| 200 | 1.3 | 5173 |
| 500 | 1.5 | 2387 |
| 1000 | 1.6 | 1273 |
| 2000 | 1.7 | 677 |
| 5000 | 1.8 | 287 |

---

### 九、参考链接

1. [Magnetic Hysteresis - Wikipedia](https://en.wikipedia.org/wiki/Magnetic_hysteresis)
2. [B-H Curve and Magnetic Saturation - HyperPhysics](http://hyperphysics.phy-astr.gsu.edu/hbase/Solids/hyst.html)
3. [Ferromagnetism - Wikipedia](https://en.wikipedia.org/wiki/Ferromagnetism)
4. [Hysteresis Models - ScienceDirect](https://www.sciencedirect.com/topics/engineering/hysteresis-models)
5. [Jiles-Atherton Model - IEEE Magnetics Society](https://ieeexplore.ieee.org/document/5310638)