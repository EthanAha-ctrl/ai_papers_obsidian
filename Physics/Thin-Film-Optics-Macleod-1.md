---
source_pdf: Thin-Film-Optics-Macleod-1.pdf
paper_sha256: 69e880a2665563d30f37ed1fa4715bf9e8517502e45702ace2b1f6ea2e759497
processed_at: '2026-08-12T15:20:20-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Thin-Film Optics 人话版 - 给 Karpathy 的 Intuition

好, 我把刚才的技术展开重新用"人话"讲一遍. 核心思路: 丢掉公式, 抓住物理图像.

---

## 这篇文档到底在讲什么

Angus Macleod 这个人, 薄膜光学领域的 godfather, 写过 *Thin-Film Optical Filters* 这本圣经. 这篇是他一个 lecture 的讲稿, 覆盖了 thin-film optics 从最基本原理到最前沿应用的整个 spectrum.

一句话总结: **光学系统里几乎每个 surface 都有 coating, coating 通过 interference 这个物理现象, 让你 engineer 反射和透射的特性**.

---

## Light 是什么 - 为什么我们只 care electric field

先 from first principles. Maxwell 告诉我们 light 是 electromagnetic wave, 同时有 E 场和 B 场. 但 Macleod 开篇就说一句话: "only the electric field has any significant interaction with materials".

**人话**: 想象 light 撞到一块 glass, 实际上是 light 里的 electrons 在驱动 material 里的 electrons 振荡. 驱动力来自 Lorentz force $\mathbf{F} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})$. 问题: 在原子里的 bound electron, 速度 $v$ 特别小 (相对于 $c$), 所以 magnetic 那一项 $q\mathbf{v}\times\mathbf{B}$ 几乎可以忽略. **只剩下 E 场干活**.

这就是为什么整个 thin-film optics 我们只 track 一个 scalar 量 $E(z,t)$, 加上 polarization 信息. Maxwell 方程退化成一个简单的 Helmholtz equation, 一维问题. **世界简化了**.

**Linearity** 是第二个关键. Material 对光场的响应是线性的 (在 low intensity 下), 所以 superposition 成立. 任何 pulse 都能拆成 monochromatic components, 每个 component 独立穿过 coating, 最后 superpose 回来. **设计变成 frequency-by-frequency 的事**, 一次算一个 wavelength.

**Vacuum wavelength convention**: 因为光进不同 medium 速度不同, 实际 wavelength 会变 ($\lambda_{eff} = \lambda_0/n$). 但为了大家能对齐讨论, 整个领域约定用 vacuum wavelength $\lambda$ 来标记光. 像 "550nm 的光" 永远指 vacuum 里 550nm, 不管它在 glass 里变成 362nm.

---

## Interference - 这门学问的灵魂

文档里反复出现 interference, 这是 thin-film optics 的核心 engine.

**人话图像**: 想象一个 wave 碰到 film 的 front surface, 一部分 reflect, 一部分进 film. 进去的那部分又碰到 back surface, 一部分 reflect 回来, 一部分 transmit 出去. 反射回来的那部分又碰到 front surface...如此循环.

每一个 round trip, wave 累积一个 phase shift $\delta = 2\pi n_f d / \lambda$ (单程) 或 $2\delta = 4\pi n_f d / \lambda$ (round trip). 同时每次 reflection 在某些 interface 会 pick up 一个 $\pi$ phase shift (从 low index 到 high index 反射时).

所有这些 partial waves 最后 superpose 在 front surface, 决定 total reflectance. **phase 对得上就 constructive, 对不上就 destructive**. thickness $d$ 是 knob, 你 turn 这个 knob, reflectance 跟着变.

---

## Half-wave 和 Quarter-wave - 两个极端

文档给两个公式:

**Half-wave** ($d = \lambda/4n_f$, round trip $= \lambda$): 
$$R_{HW} = \frac{(n_0 - n_{sub})^2}{(n_0 + n_{sub})^2}$$

**Quarter-wave** ($d = \lambda/4n_f$, round trip $= \lambda/2$): 
$$R_{QW} = \frac{(n_0 - n_f^2/n_{sub})^2}{(n_0 + n_f^2/n_{sub})^2}$$

变量: $n_0$ = incident medium index (air=1), $n_{sub}$ = substrate index (glass=1.52), $n_f$ = film index, $R$ = reflectance.

### Half-wave 的直觉

Half-wave layer 的 round trip phase $= 2\pi \equiv 0$. 所有 partial beam 回到 front surface 时 phase 关系跟"无限薄膜"完全一样. **所以 half-wave layer 对 reflectance 没影响, 就像它不存在**. 文档叫它 "absentee layer".

这是一个 super useful design primitive: 你想"加一层" 但又不想影响某个波长, 就用那个波长的 half-wave.

### Quarter-wave 的直觉 - Admittance Transform

这才是 magic 所在. Quarter-wave round trip $= \pi$, $\cos\delta = 0$, $\sin\delta = 1$. 这让薄膜的 characteristic matrix 变成一个简单的 swap matrix:

$$M_{QW} = \begin{pmatrix} 0 & i/n_f \\ in_f & 0 \end{pmatrix}$$

这个 matrix 作用于 substrate admittance $Y_{sub} = n_{sub}$, 输出 input admittance:
$$Y_{input} = \frac{n_f^2}{n_{sub}}$$

**Quarter-wave layer 把 substrate 的 admittance $n_{sub}$ 变成 $n_f^2/n_{sub}$**. 这是一个 nonlinear transform. 如果你能选 $n_f$ 让 $n_f^2/n_{sub} = n_0$ (incident medium), 那 reflectance $R = 0$ - **完全 antireflection**.

对 air/glass, 理想 $n_f = \sqrt{1 \times 1.52} \approx 1.23$. 但最低的实用 material 是 **MgF2** ($n=1.38$), 比理想高. 所以单层 MgF2 QW 不能让 $R=0$, 但能把 bare glass 的 4.3% reflectance 降到 ~1.3%. 文档说 "MgF2 is the best we have" 就是这个意思.

这就是整个 antireflection coating 行业的基础. 多层 coating 用多个 QW 或 fractional QW 来逼近 broadband $R \approx 0$.

参考: [Essential Macleod software](http://www.thinfilmcenter.com/essential.htm), [Admittance transformer concept](https://www.rp-photonics.com/optical_admittance.html)

---

## Quarterwave Stack - 1D Photonic Crystal

文档的核心 building block 是 **(HL)^N** 结构: alternating high-index 和 low-index quarter-wave layers.

### 为什么这个结构 reflect 得这么强

**人话推导**: 在 design wavelength $\lambda_0$, 每层是 QW. 考虑从每个 interface 反射的 beam:

- Air/H interface: $r_{0H} = (n_0 - n_H)/(n_0 + n_H)$, **negative** (因为 $n_H > n_0$)
- H/L interface: $r_{HL} = (n_H - n_L)/(n_H + n_L)$, **positive**
- L/H interface: $r_{LH} = -r_{HL}$, **negative**
- ...如此交替

每次 round trip 经过两层 (HL), phase 累积 $2 \times \pi = 2\pi \equiv 0$. 但 reflection coefficient 的 sign 在 H→L 和 L→H 之间 **flip**. 

所以: H/L interface 贡献 $r_{HL} \cdot e^{i \cdot 0} = r_{HL}$ (positive)
L/H interface 贡献 $-r_{HL} \cdot e^{i \cdot 0} = -r_{HL}$ (但 wait, 还要算到这一层的 round trip phase)

让我重新理清: 第 $j$ 个 interface 贡献的 reflection amplitude 是 $r_j \cdot \prod e^{i\delta_k}$, 其中 $\delta_k$ 是光从 top surface 一路 round trip 到这个 interface 再回来的 phase.

对 (HL)^N, 中间每个 interface 的累积 phase 都对齐成 0 mod $2\pi$, 加上 reflection sign flip, 所有 contributions **同号相加**. 层数越多, 总 reflectance 越接近 1. 这是 **constructive interference 在所有 interface 同时发生**.

这就是 1D photonic crystal 的 bandgap. 物理上等价于: 这个 wavelength 的光"无法 propagate" 进 crystal, 被全部 reflect 回来.

### Stop Band 宽度

High reflectance zone 不是无限宽, 它有 bandwidth:
$$\frac{\Delta\lambda}{\lambda_0} = \frac{2}{\pi}\sin^{-1}\left(\frac{n_H - n_L}{n_H + n_L}\right)$$

$n_H/n_L$ 比越大, band 越宽. 这就是文档说 "higher the ratio of high to low index the broader is the region" 的精确表达.

实例: Ta2O5/SiO2 ($n_H=2.1, n_L=1.46$), stop band $\approx 17\%$. 用在 visible 的 mirror 设计 $\lambda_0 = 550$nm, stop band 覆盖 ~500-600nm, 显然不够覆盖整个 visible 400-700nm. 需要 **staggered thickness design** (不同 $\lambda_0$ 的 stack 串联).

参考: [Distributed Bragg reflector](https://en.wikipedia.org/wiki/Distributed_Bragg_reflector), [Photonic crystal Wikipedia](https://en.wikipedia.org/wiki/Photonic_crystal)

---

## Cavity Structure - Fabry-Perot Filter

### Single Cavity

两个 high reflector 中间夹一个 spacer. 这就是 **Fabry-Perot etalon**.

透射率公式 (Airy formula):
$$T = \frac{(1-R_m)^2}{(1-R_m)^2 + 4R_m \sin^2(\delta/2)}$$

- $R_m$: 单个 mirror 的 reflectance (假设两侧 mirror 一样)
- $\delta = 4\pi n_c d_c / \lambda$: round-trip phase in cavity

当 $\delta = 2m\pi$ ($m$ 整数), $\sin^2 = 0$, $T = 1$ 完美透射. 这些是 **resonance condition**.

$R_m$ 越接近 1, peak 越窄. Finesse $\mathcal{F} = 4R_m/(1-R_m)^2$ 量化这个 sharpness.

### 多 Cavity 串联

Single cavity 的问题: passband 是 Lorentzian, 翼很长, rolloff 慢, 旁瓣大.

解法: **串联多个 cavity**. 文档图示的是 3-cavity 设计:
```
Air | [Reflector] [Spacer] [Reflector] [Spacer] [Reflector] [Spacer] [Reflector] | Glass
```

每个 cavity 是一个 resonance, 多个 resonance 耦合后:
1. Passband 顶部 flatten (ripple 减少)
2. 两侧 rolloff 变陡

但代价是 out-of-band ripple 增加. 文档图示 "two-layer match next to air has reduced the ripple" - 这是用 **matching layers** 在 air interface 做 impedance match, 把 ripple 压下去.

### Telecom Filter 的疯狂设计

文档给的这个:
```
Air | 1.267L 0.338H (HL)^7 H (HL)^15 H (HLLH(HLL)^15 HHLHH)^2 (HL)^15 H (HL)^7 H | Glass
```

158 层! 仔细看结构:
- 中心是 `(HLLH(HLL)^15 HHLHH)^2` - 这是 2 个高 order cavity
- 每个 cavity 周围有 (HL)^15 强 mirror 和 (HL)^7 弱 mirror
- 这叫 **apodization** - 不同 cavity 周围 mirror 强度不同, 优化 passband shape 和 stopband rejection

目的: telecom DWDM (Dense Wavelength Division Multiplexing) 需要通道间距 100 GHz (~0.8nm at 1550nm), passband FWHM 要窄到 ~50 GHz, 同时 stopband isolation 要 >40 dB. 这种 filter 用在 add/drop multiplexer 里, 158 层才能达到 spec.

参考: [DWDM filter design](https://www.rp-photonics.com/dwdm_filters.html), [Multi-cavity filters](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=905)

---

## Surface Plasmon Resonance - 用 evanescent wave 传感

这部分文档讲得很 magic, 让我把机制说清楚.

### Kretschmann Configuration

Setup: high-index glass prism / ~50nm silver film / water (analyte). 光从 prism 一侧入射, 角度大于 critical angle, 发生 total internal reflection. 但 silver film 太薄, evanescent field 能 leak 出来到 silver/water interface.

### 物理: Phase Matching

Evanescent wave 的 in-plane wavevector:
$$k_x = n_{prism} \cdot \frac{\omega}{c} \cdot \sin\theta$$

Silver/water interface 支持 **surface plasmon polariton (SPP)**, 一个 bound surface wave, 其 wavevector:
$$k_{sp} = \frac{\omega}{c} \sqrt{\frac{\epsilon_m \epsilon_d}{\epsilon_m + \epsilon_d}}$$

- $\epsilon_m$: silver 的 complex dielectric function (Drude model)
- $\epsilon_d$: water 的 dielectric constant (~1.77 at visible)

当 $k_x = \text{Re}(k_{sp})$, phase matching, 光耦合进 SPP, 反射率急剧下降到接近 0. 这是文档说的 "very narrow resonance-like feature in angle of incidence".

只有 **p-polarization** 能激发 SPP (因为 SPP 需要 E 场 perpendicular component, s-polarization 没有).

### 为什么能做 sensor

SPP 的 field 在 silver/water interface 外侧指数衰减, penetration depth $\sim \lambda/2\pi \approx 100$nm. 所以 surface 上 100nm 范围内的任何 dielectric 变化都会 perturb SPP 的 dispersion, 进而 move resonance angle.

文档说 "1nm dielectric of index 2.00 on outer silver surface" 引起 perceptible shift. 实际数据: 1nm 层 ($n=2.0$) 大约引起 $0.01-0.1$ degree shift, 现代 SPR 仪器能测到 $10^{-5}$ degree, 对应 sub-ng/mL 的 protein detection.

### Bio应用

文档提到 *E. coli* detection. 流程:
1. Silver/gold surface 修饰 antibodies (receptors)
2. 流 sample 过 surface, target pathogen bind 到 antibody
3. Bind 增加 surface mass, 改变 local dielectric, move resonance angle
4. Real-time 监测 angle shift, 反推 binding kinetics

商业系统: Biacore (Cytiva), SPR sensitivity 到 $\sim 1$ pg/mm² surface mass density, 能测 affinity $K_D$ 从 mM 到 pM. 这是 thin-film optics 在生命科学的最成功应用.

参考: [Homola SPR review Chemical Reviews 2008](https://pubs.acs.org/doi/10.1021/cr068107d), [Biacore technology](https://www.cytivalifesciences.com/en/products/biacore)

---

## Ultrafast Optics - Chirped Mirror 的核心

这部分是文档最 tricky 的, 我慢慢讲.

### Pulse = Envelope × Carrier

一个 short pulse 的 electric field:
$$E(z,t) = \mathcal{E}(z,t) \cdot e^{i(\omega_0 t - k_0 z)}$$

- $\mathcal{E}(z,t)$: 缓变的 envelope
- $\omega_0$: carrier angular frequency
- $k_0 = n(\omega_0) \omega_0/c$: carrier wavevector

**Envelope 以 group velocity $v_g = d\omega/dk$ 移动, carrier 以 phase velocity $v_p = \omega_0/k_0 = c/n(\omega_0)$ 移动**. 两者不同因为 material 有 dispersion $n(\omega)$ 非常数.

### Group Delay 和它的导数

文档定义:
- **Group Delay (GD)**: $\tau_g = -d\varphi/d\omega$ (units: time) - pulse 何时到达
- **Group Delay Dispersion (GDD)**: $D_2 = -d^2\varphi/d\omega^2$ (units: time²) - chirp
- **Third Order Dispersion (TOD)**: $D_3 = -d^3\varphi/d\omega^3$ (units: time³) - asymmetric distortion

$\varphi$ 是 coating 引入的 phase. 负号是 convention (phase 通常写成 $e^{-i\omega t + i\varphi}$, 所以 phase 增加对应时间延迟).

### Chirp 的图像

Positive GDD 意味着: 低频 (long wavelength) 比 高频 (short wavelength) 先到. Pulse 变成 frequency sweep - "red-to-blue chirp".

Glass 的 normal dispersion: $n$ 随 $\lambda$ 增大而减小, 所以长波长跑得快, 先到. **Glass 给 pulse positive GDD**. Femtosecond pulse 穿过 1cm glass 大约累加 ~1000 fs² GDD, 把 10fs pulse 拉宽到 50fs.

### Chirped Mirror 怎么 compensate

要 undo positive GDD, 需要 **negative GDD** - 让长波长后到, 短波长先到.

**Chirped mirror 的 trick**: 让不同 wavelength 在 coating 内穿透到不同 depth 再 reflect 回来. 如果短波长穿透更深 (反射得晚), 长 wavelength 穿透更浅 (反射得早), 就实现 negative GDD.

文档图示的 67-layer 设计: "base quarterwave stack reflector with a phase correction section in front that takes the form of a variable depth reflector". 

具体做法: perturb layer thicknesses, 让 Bragg condition $\lambda = 2nd$ 在不同 depth 对应不同 $\lambda$. 这叫 **chirped Bragg mirror**.

目标: 在某个 bandwidth 内, 让 $\varphi(\omega)$ 是 quadratic in $\omega$, 二阶导数 = 目标 GDD (-500 fs²).

### 为什么不能让 pulse "提前到达"

文档说 "coating design techniques simply fail to yield a suitable result" 对 "pulse arrives sooner before reaching mirror". 

这是 **causality** 的体现. Kramers-Kronig relation 连接 phase 和 amplitude: 你想要 anomalous dispersion (negative GD), 必然伴随 absorption peak. 在透明 coating 里没法实现 negative absolute GD, 只能实现 negative GDD (二阶导数). 这就是为什么文档强调 "within limits, these quantities are susceptible to design".

### 实际应用

Ti:Sapphire laser cavity 里有 1-2 个 chirped mirror, 每次 bounce 给 ~-100 fs² GDD, compensate gain medium 和 crystal 的 positive GDD. 让 oscillator 维持 sub-10fs pulse.

更高级: **double-chirped mirror** (Kärtner, Matuschek), 把 coupling 也 chirp, 优化 GDD 和 impedance matching 同时. 这是 ultrafast laser 的 enabling technology.

参考: [Szipöcs chirped mirror original paper](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-13-2710), [Kärtner double-chirped mirror](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-22-11-816)

---

## Tilt 效应 - 倾斜就蓝移

### 为什么

Phase thickness 在 angle $\theta$ 下:
$$\delta(\theta) = \frac{2\pi n d \cos\theta_f}{\lambda}$$

$\theta_f$ 是 film 内部 angle (Snell: $n_0 \sin\theta_0 = n_f \sin\theta_f$).

QW condition $\delta = \pi/2$ 对应:
$$\lambda_{eff}(\theta) = 4 n_f d_f \cos\theta_f$$

$\theta_0$ 增加 → $\cos\theta_f$ 减小 → $\lambda_{eff}$ 减小. **蓝移**.

Small angle 估计: $\Delta\lambda/\lambda \approx -\theta_0^2/(2 n_0^2)$. 对 air ($n_0=1$), 10度倾角大约蓝移 1.5%.

### Polarization Splitting

s-polarization 的 effective admittance: $y_s = n\cos\theta$
p-polarization 的 effective admittance: $y_p = n/\cos\theta$

s 和 p 看到的是不同的薄膜. 特征曲线 split. 这是文档说 "p-characteristic becomes weaker, s becomes stronger" 的根源. Brewster angle 时 p 反射完全消失.

### Iridescence 应用

文档最后的 anti-counterfeit 例子:
- Cr (2.1nm) - 半透明 absorber
- MgF2 (500nm) - dielectric spacer
- Al (thick) - back reflector

这其实是一个 **absorbing etalon**: light 穿过 Cr, 在 MgF2 里 round trip, 再穿 Cr 出来. Cr 的 absorption 让 color 更 saturated. 不同 angle → 不同 path length → 不同 interference color.

US currency 用这种 color-shifting ink, 100 美元上的 "100" 字样, 从正面看绿色, 倾斜看黑色. 复印机做不到这个效果, 所以 anti-forgery 有效.

参考: [Optically variable ink Wikipedia](https://en.wikipedia.org/wiki/Optically_variable_ink), [Thin film tilt effects](https://www.thinfilmcenter.com/)

---

## 制造工艺 - 为什么 5nm 误差要命

### Thermal Evaporation 的问题

热蒸发: 加热 material 到 boil, vapor 凝结到 substrate. 粒子能量 ~0.1 eV, 到 substrate 后 mobility 低, 排不紧. 形成 **columnar microstructure**, 中间有 voids. 

密度只有 bulk 的 80-95%. 后果:
1. Refractive index 偏低且不稳
2. **Moisture absorption**: 水 ($n=1.33$) 慢慢渗进 voids, $n_{film}$ 上升, 光谱漂移. 文档没明说但这是 thin-film engineer 日常 nightmare
3. Temperature-dependent shift

### 现代工艺: Ion-Assisted 和 Sputtering

**Ion-Assisted Deposition (IAD)**: 蒸发同时用 ion beam (~100 eV O2+ 或 Ar+) bombard growing film. 把 atoms 撞紧, 提高 packing density.

**Magnetron Sputtering** (Bühler Helios 那个): 用 ~10-100 eV Ar+ sputter target material 到 substrate. 粒子能量更高, film density 接近 bulk (99%+), moisture-insensitive, repeatable.

代价: deposition rate 低, machine 昂贵, 但 yield 高. telecom filter 158 层用 sputtering 做, 因为 thermal evaporation 的 5nm 误差累积下来完全破坏 passband shape.

### 监控: Optical Monitoring

Gemini mirror 要 uniformity <1nm. telecom filter 158 层要 total thickness error <0.1%. 这是用 **in-situ optical monitoring** - 蒸发过程中实时测 reflectance/transmittance, 当 layer 到达 target thickness 自动 shutter 切断. 加上 planetary rotation 让 uniformity 跨整个 substrate.

Macleod 的 Essential Macleod 软件做 design + monitoring strategy 优化. 这种 software-driven process control 是现代 thin-film manufacturing 的核心.

参考: [Ion-assisted deposition](https://www.sciencedirect.com/science/article/pii/S0040609015002373), [Thornton structure zone model](https://en.wikipedia.org/wiki/Thornton_diagram)

---

## 几个 Bonus 联想

### Low-E Glass - 你家窗户上的 thin-film

**D/M/D 结构**: Glass / SnO2 (40nm) / Ag (12nm) / SnO2 (40nm).

Visible region: D/M/D 是 Ag 的 antireflection coating - 让 visible 透过 (T > 85%).
IR region: QW condition 不满足, Ag 的 intrinsic high reflectance 主导 - 反射 IR (热).

结果: 冬天反射室内 IR 回室内保温, 夏天反射室外 IR 隔热. 全球年需求 >10^9 m², 这是 thin-film optics 在 energy efficiency 的最大应用.

### Halogen Lamp 的 IR Reflector

文档说 "100-layer" all-dielectric IR reflector, 折射 IR 回 filament. 必须用 **refractory dielectric** (TiO2, HfO2, Al2O3 - 高 melting point) 因为 lamp envelope 温度 ~400°C. silver-based coating 撑不住. 这是 thin-film 让 halogen lamp 效率翻倍的 trick.

### Oil on Water - Thin-Film Optics 的 everyday showcase

油膜在水上的彩色是 thin-film interference in reflection:
$$I_{refl}(\lambda) = |r_{air/oil} + r_{oil/water} \cdot e^{i \cdot 4\pi n_{oil} d / \lambda}|^2$$

对 D65 daylight 积分得到 RGB. 文档图示把计算颜色叠在实际 oil slick 照片上, "agreement is close" - 这是 thin-film optics 作为 mature theory 的 showcase, 物理上没什么 unknown.

参考: [Low-emissivity glass Wikipedia](https://en.wikipedia.org/wiki/Low-emissivity_glass), [Thin film color simulation](http://www.thinfilmcenter.com/)

---

## The Big Picture

把整个文档的 intuition 收敛成 5 句话:

1. **Admittance 是 king**: 每层 film 是 2x2 matrix, 把 back-side admittance transform 成 front-side admittance. Quarter-wave 是最简单的 transform, 把 $Y$ 变成 $Y_f^2/Y$. 设计 coating 就是 chain 这些 transform, 让 input admittance 接近 incident medium admittance (for AR) 或远离它 (for high reflection).

2. **Phase is everything**: Transparent dielectric regime, 所有特性都是 phase interference. Reflectance 是 phase 的 sinusoidal function. Pulse shaping 是 phase derivative 的 engineering. Cavity filter 的 passband 是 resonance condition 决定的 phase matching.

3. **Materials constrain design**: $n_H/n_L$ ratio 决定 stop band width. $n_{min}$ (MgF2 = 1.38) 决定 AR coating 能多好. 你不能任意选 index, 所以 design 是 constrained optimization.

4. **Causality constrains phase**: Kramers-Kronig relation 意味着 reflectance 和 phase 不能独立 design. "Pulse 提前到达"违反 causality, 所以 design fail. Negative GDD 可行, negative GD 不可行.

5. **Manufacturing 是 real bottleneck**: Design 出来 158 层, 制造要每层 <0.1% 误差. 这需要 sputtering + optical monitoring + planetary tooling + yield optimization. **Theory is mature, engineering is hard**.

Macleod 最后说: "Wherever there is optics there will be an optical coating." 这话精确得可怕. 从 telescope mirror 到你家窗户, 从 fiber telecom 到 LIGO gravitational wave detector, 从 AR/VR headset 到 quantum computing cavity - 都是 thin-film optics 在 enable. 

**这是 optics 里最老又最新的分支**: 19世纪 Rayleigh, Fabry, Perot 奠基, 21世纪还在 chirped mirror for attosecond physics, metasurface coating, EUV lithography pellicle 等前沿驱动新工艺.

主要参考资源:
- [Thin Film Center (Macleod 公司)](http://www.thinfilmcenter.com/)
- [Essential Macleod software](http://www.thinfilmcenter.com/essential.htm)
- [Macleod's textbook Thin-Film Optical Filters](https://www.routledge.com/Thin-Film-Optical-Filters-Fourth-Edition/Macleod/p/book/9781420073014)
- [Society of Vacuum Coaters](https://www.svc.org/)
- [Homola SPR review](https://pubs.acs.org/doi/10.1021/cr068107d)
- [Chirped mirror original paper](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-13-2710)
- [Biacore SPR technology](https://www.cytivalifesciences.com/en/products/biacore)
- [Optically variable ink Wikipedia](https://en.wikipedia.org/wiki/Optically_variable_ink)

---

# Thin-Film Optics 详解 - 为 Karpathy 构建 Intuition

这篇文档是 Angus Macleod (Thin Film Center Inc, Tucson) 关于薄膜光学的演讲讲稿. Macleod 是这个领域的权威, 他的书 *Thin-Film Optical Filters* 是圣经级参考. 我会从物理直觉出发, 把文档中的核心概念展开, 补充背后的数学.

---

## 1. 光作为 Electromagnetic Wave 的基本直觉

文档开篇说: "At very high frequencies, only the electric field has any significant interaction with materials". 这个说法需要更深的解释.

光与物质相互作用的 coupling strength 取决于 Lorentz force:
$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

对于束缚电子 (binding 在原子中), 其速度 $v$ 远小于光速 $c$. 因为 $|\mathbf{B}| = |\mathbf{E}|/c$ (在平面波中), magnetic force 相对 electric force 的比值约为 $v/c \sim \alpha \approx 1/137$ (fine structure constant 量级). 所以 optical regime 下, magnetic interaction 可以忽略, 我们只需要追踪 $\mathbf{E}$ 场.

这给薄膜光学一个巨大简化: 我们只关心 **scalar wave** $E(z,t)$, 配合 **polarization** (E 场方向). Maxwell 方程在 monochromatic 假设下退化为 Helmholtz equation:
$$\nabla^2 E + k^2 n^2(\mathbf{r}) E = 0, \quad k = 2\pi/\lambda = \omega/c$$

**Linearity** 的关键推论是 superposition + Fourier decomposition. 任何 waveform 都能拆成 harmonic components, 每个 component 独立穿过涂层, 最后 superpose. 这让设计变成 frequency-by-frequency 的 problem. 文档里特别强调 "we usually assume a continuous spectrum of equal energy components" - 这是 standard design assumption, 用 uniform illumination 评估.

**Vacuum wavelength convention**: 实际介质中 $\lambda_{eff} = \lambda_0/n$. 文档说用 vacuum wavelength $\lambda$ 标记, 实际波长 $\lambda/n$. 这样做让所有 discussion 有统一 reference, 不随介质变化.

---

## 2. Interference in Thin Film 的核心公式 - 深入推导

文档给出两个关键 formula:

**Half-wave film:**
$$R_{HW} = \frac{(n_0 - n_{sub})^2}{(n_0 + n_{sub})^2}$$

**Quarter-wave film:**
$$R_{QW} = \frac{(n_0 - n_f^2/n_{sub})^2}{(n_0 + n_f^2/n_{sub})^2}$$

变量说明:
- $n_0$: incident medium 的 refractive index (air ≈ 1.0)
- $n_{sub}$: substrate 的 refractive index (glass ≈ 1.52)
- $n_f$: thin film 的 refractive index
- $R$: reflectance (intensity reflection coefficient 的平方)

### 物理直觉: Admittance Transformation

这两个公式背后的核心是 **admittance** 概念. 在薄膜光学中, 我们追踪的不是 impedance (像传输线), 而是 **modified admittance** $Y = \eta_0 \cdot n$ (normal incidence, $\eta_0 = \sqrt{\mu_0/\epsilon_0}$ 是 vacuum admittance).

对于 thickness $d$ 的单层 film, **characteristic matrix** 是:
$$M = \begin{pmatrix} \cos\delta & \frac{i}{y}\sin\delta \\ iy\sin\delta & \cos\delta \end{pmatrix}$$

其中:
- $\delta = 2\pi n_f d / \lambda$ 是 phase thickness (round-trip half)
- $y = n_f$ (normal incidence 时的 admittance)

整个 stack 的 input admittance 通过 matrix multiplication 得到. 然后用 Fresnel-like formula 计算 reflectance:
$$R = \left|\frac{Y_0 - Y_{input}}{Y_0 + Y_{input}}\right|^2$$

### Quarter-wave 的特殊情况

当 $\delta = \pi/2$ (quarter-wave, round trip = $\pi$), $\cos\delta = 0$, $\sin\delta = 1$:
$$M_{QW} = \begin{pmatrix} 0 & i/n_f \\ in_f & 0 \end{pmatrix}$$

对 substrate admittance $Y_{sub} = n_{sub}$ 作用:
$$Y_{input} = \frac{i \cdot n_f \cdot n_{sub} \cdot (i/n_f)}{...} = \frac{n_f^2}{n_{sub}}$$

(更精确地: $Y_{input} = (M_{11} Y_{sub} + M_{12}) / ...$, 但 QW 时简化为 $n_f^2/Y_{sub}$)

所以 quarter-wave 把 substrate admittance $n_{sub}$ **transform 成** $n_f^2/n_{sub}$. 这就是公式中 $n_f^2/n_{sub}$ 的来源.

### Half-wave 的特殊情况

当 $\delta = \pi$ (half-wave, round trip = $2\pi$), $\cos\delta = -1$, $\sin\delta = 0$:
$$M_{HW} = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix} = -I$$

这个 matrix 是 identity (modulo sign), 所以 film 完全透明 - admittance 不变, reflectance 等于 bare substrate. 这就是 "half-wave layer is absent" 原理 (absentee layer).

### Antireflection Design Intuition

要让 $R = 0$, 需要 $Y_{input} = Y_0 = n_0$. 对 quarter-wave film:
$$\frac{n_f^2}{n_{sub}} = n_0 \implies n_f = \sqrt{n_0 \cdot n_{sub}}$$

对 air ($n_0=1$) / glass ($n_{sub}=1.52$) 系统, 理想 $n_f = \sqrt{1.52} \approx 1.23$. **MgF2** ($n_f=1.38$) 是常用最低折射率材料, 比 ideal 稍高, 不能完全零反射, 但把 ~4.3% bare glass reflectance 降到 ~1.3%. 这就是为什么文档说 "ideally we would like a material of still lower index but MgF2 is the best we have".

参考链接: [Macleod's book reference](https://www.crcpress.com/Thin-Film-Optical-Filters/Macleod/p/book/9781420073014), [TFM Admittance Diagram](http://www.thinfilmcenter.com/)

---

## 3. Quarterwave Stack - Bragg Reflector 的物理

文档说 HLHL...HL 结构 "all the multiple beams emerging from the top surface to form the reflected beam are exactly in phase". 让我精确化这个 statement.

考虑 (HL)^N 结构, normal incidence. 在 design wavelength $\lambda_0$, 每层 QW. 反射贡献来自每个 interface:

- Air/H interface: reflection coefficient $r_{0H} = (n_0 - n_H)/(n_0 + n_H)$ (negative, since $n_H > n_0$)
- H/L interface: $r_{HL} = (n_H - n_L)/(n_H + n_L)$ (positive)
- L/H interface: $r_{LH} = (n_L - n_H)/(n_L + n_H) = -r_{HL}$ (negative)
- ...
- 最后一个 interface 到 substrate

关键: 当光从 QW layer 的 far side 反射回来时, 产生 round-trip phase $2\delta = \pi$. 但 reflection coefficient 的符号在 H→L 和 L→H 之间 **反转**. 

第 $j$ 个 interface 贡献的 effective reflection: $r_j \cdot e^{i \cdot \text{(phase accumulated)}}$

经过两层 (HL) 的 round trip 是 $2 \times \pi = 2\pi$, 等于 $0$ phase shift mod $2\pi$. 加上 reflection sign flip, 总相位贡献 from consecutive interfaces 是 **in phase**.

所以所有 reflection contributions **相长叠加**, total reflectance 接近 1. 层数越多, reflectance 越接近 1. 这是 1D photonic crystal 的 bandgap.

### Stop Band 宽度

High reflectance zone 的相对半宽:
$$\frac{\Delta\lambda}{\lambda_0} = \frac{2}{\pi}\sin^{-1}\left(\frac{n_H - n_L}{n_H + n_L}\right)$$

对比度 $n_H/n_L$ 越大, stop band 越宽. 这是文档说 "higher the ratio of high to low index the broader" 的精确表达.

对 Ta2O5/SiO2 ($n_H=2.1$, $n_L=1.46$): $\Delta\lambda/\lambda_0 \approx 0.17$ (约 17%).
对 ZnS/MgF2 ($n_H=2.35$, $n_L=1.38$): $\Delta\lambda/\lambda_0 \approx 0.23$.

参考: [Quarter-wave stack Wikipedia](https://en.wikipedia.org/wiki/Bragg_mirror), [Distributed Bragg reflector](https://www.rp-photonics.com/bragg_mirrors.html)

---

## 4. Cavity Structure (Fabry-Perot Filter) 的设计哲学

文档给出的设计 notation:
```
Air | HLHLHLH LLLL HLHLHLH L HLHLHLH LLLL HLHLHLH | Glass
       [Reflector]  [Spacer]  [Reflector]   [Reflector]  [Spacer]  [Reflector]
```

这是 **multiple-cavity Fabry-Perot** 结构. 每个 HLHLHLH 是 ~99% reflectance 的 Bragg mirror, 中间的 LLLL (4 quarter-waves = 1 full wave, 也叫 2L 在 some notation) 是 spacer layer.

### Single Cavity 的 Airy Formula

透射率:
$$T = \frac{(1-R_m)^2}{(1-R_m)^2 + 4R_m \sin^2(\delta/2)}$$

或写成 FWHM 形式:
$$T = \frac{1}{1 + \mathcal{F}\sin^2(\delta/2)}, \quad \mathcal{F} = \frac{4R_m}{(1-R_m)^2}$$

- $R_m$: mirror 的 reflectance (假设两个 mirror 相同)
- $\delta = 4\pi n_c d_c \cos\theta_c / \lambda$: round-trip phase in cavity
- $\mathcal{F}$: **coefficient of finesse**, 决定 peak 的 sharpness

FWHM 与 spacer thickness 关系:
$$\text{FWHM} \propto \frac{1}{\mathcal{F}} \cdot \frac{\lambda_0}{\text{order}}$$

更高 order (更厚 spacer) → 更窄 FWHM. 但文档强调: "Adjusting number of halfwaves in the cavities and number of layers in the reflectors gives fine-coarse control". 

### 多 Cavity 的优势

Single cavity filter 的 passband 是 Lorentzian, 翼很长 (slow rolloff). 多个 cavity 串联后, passband 顶部变平, 两侧更陡. 但代价是 ripple 增加. 文档图示的"two-layer match next to air" 就是用来 flatten ripple 的 **matching layers**.

这是经典的 **Chebyshev / Butterworth filter design** 思想在 optical domain 的应用. 文档中的 telecom filter:
```
Air | 1.267L 0.338H (HL)^7 H (HL)^15 H (HLLH(HLL)^15 HHLHH)^2 (HL)^15 H (HL)^7 H | Glass
```

158 层! 不是简单的 FP, 而是 **multi-cavity with apodization** - 每个 cavity 周围的 Bragg mirror 层数不同 (15, 15, 15), 内层 mirror 更强, 外层 (7, 7) 较弱, 这样 ripple 控制和 steepness 一起优化.

参考: [Thin-film narrowband filters](https://www.rp-photonics.com/narrowband_optical_filters.html)

---

## 5. Surface Plasmon Resonance - Kretschmann 配置

文档提到 SPR 用于生物传感. 这是 thin-film optics 中最美的应用之一.

### 物理机制

Kretschmann 配置: high-index prism / thin metal (~50nm Ag or Au) / analyte (water).

Total internal reflection 在 prism/metal interface 产生 **evanescent wave**, 其 in-plane wavevector:
$$k_x = n_{prism} \cdot k_0 \cdot \sin\theta$$

Metal/dielectric interface 支持 **surface plasmon polariton (SPP)**, 其 dispersion:
$$k_{sp} = k_0 \sqrt{\frac{\epsilon_m \epsilon_d}{\epsilon_m + \epsilon_d}}$$

- $\epsilon_m$: metal 的 complex dielectric function (Drude model: $\epsilon_m(\omega) = 1 - \omega_p^2/\omega^2$)
- $\epsilon_d$: dielectric (analyte) 的 dielectric constant

当 $k_x = \text{Re}(k_{sp})$ 时, p-polarized light 耦合进 SPP, 反射率急剧下降 - 这就是文档说的 "very narrow resonance-like feature". s-polarization 不能激发 SPP (因为 SPP 需要 E 场垂直于 surface 的分量).

### Sensitivity

SPP 的 field 在 metal 外侧指数衰减, penetration depth ~$\lambda/2\pi \approx 100$nm. 所以只有 surface 附近 100nm 范围内的变化能 perturb resonance.

文档说 "1nm of dielectric material of index 2.00 to the outer silver surface results in a perceptible perturbation" - 这是 angle shift 大约 $10^{-3}$ degrees 量级, 现代 SPR 仪器能测到 $10^{-5}$ degrees, 检测限达 $\sim 1$ pg/mm² surface mass density.

公式: 表面 mass 质量变化 $\Delta m$ (in pg/mm²) 与 angle shift $\Delta\theta$ (in degrees) 大约 $\Delta m \approx 1.5 \times 10^6 \times \Delta\theta$ (for water analyte at 633nm).

参考: [Homola SPR review](https://pubs.acs.org/doi/10.1021/cr068107d), [Biacore technology](https://www.sartorius.com/en/products/protein-analysis/protein-interaction-analysis)

---

## 6. Ultrafast Optics 与 Chirped Mirror

这部分文档讲得特别精炼, 让我展开.

### Group Velocity vs Phase Velocity

对于 pulse:
$$E(z,t) = \mathcal{E}(z,t) \cdot e^{i(\omega_0 t - k_0 z)}$$

- $\mathcal{E}(z,t)$: envelope function (slowly varying)
- $\omega_0$: carrier angular frequency
- $k_0 = n(\omega_0)\omega_0/c$: carrier wavevector

**Phase velocity** $v_p = \omega_0/k_0 = c/n(\omega_0)$ - 单个 oscillation 的速度.

**Group velocity** $v_g = d\omega/dk|_{\omega_0}$ - envelope 的速度.

由于 dispersion $n(\omega)$ 非常数, $v_g \neq v_p$.

### Group Delay Dispersion

文档的公式:
- Group Delay: $\tau_g = -d\varphi/d\omega$ (units: time)
- GDD: $D_2 = -d^2\varphi/d\omega^2$ (units: time²)
- TOD: $D_3 = -d^3\varphi/d\omega^3$ (units: time³)

负号约定: 让 phase $\varphi$ 对 reflected pulse 在 $\omega_0$ 附近展开:
$$\varphi(\omega) = \varphi_0 + \varphi_1(\omega-\omega_0) + \frac{1}{2}\varphi_2(\omega-\omega_0)^2 + \frac{1}{6}\varphi_3(\omega-\omega_0)^3 + ...$$

- $\varphi_1 = d\varphi/d\omega$: GD, 决定 pulse 何时到达
- $\varphi_2 = d^2\varphi/d\omega^2$: GDD, 决定 **chirp** (frequency sweep)
- $\varphi_3$: TOD, 决定 **asymmetric distortion**

### Chirp 的直觉

Positive GDD: 长波长 (低频) 先到, 短波长 (高频) 后到 ("positive chirp" or "red-to-blue chirp").
Normal dispersion of glass: positive GDD. 所以 pulse 通过 glass 后变宽并 chirp.

要 compensate, 需要 **negative GDD**. 两种方法:
1. **Prism pair** (geometric dispersion)
2. **Chirped mirror** (文档说的)

### Chirped Mirror 设计

文档图示的 67-layer 设计: "base quarterwave stack reflector with a phase correction section in front that takes the form of a variable depth reflector".

原理: 不同频率的光在 coating 内穿透到不同 depth. 长 wavelength 穿透更深 (因为 Bragg condition: $\lambda = 2nd$, $\lambda$ 大对应 $d$ 大). 所以长 wavelength 反射得晚 - 这产生 positive GDD.

要实现 negative GDD, 需要 **reverse** penetration depth. 方法是 perturb layer thicknesses, 让短 wavelength 穿透更深. 文档图示的"variable depth reflector"就是 thickness-modulated QW stack.

Negative GDD = -500 fs² over bandwidth. 6 fs pulse @ 800nm 的 Ti:Sapphire oscillator需要 ~-100 fs² per bounce, 所以 cavity 里 1-2 个 chirped mirror 就够.

公式化的设计目标:
$$\varphi(\omega) = \text{linear in } \omega \text{ with slope } \tau_g, \text{ plus } D_2(\omega-\omega_0)^2/2$$

可写的 specification, 但 Macleod 说 "coating design techniques simply fail to yield a suitable result" 对于 "pulse arrives sooner before reaching mirror" - 这违反 causality (Kramers-Kronig constraint).

参考: [Chirped mirror review (Szipöcs)](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-13-2710), [Ultrafast optics textbook](https://www.springer.com/gp/book/9783540730839)

---

## 7. Tilt Sensitivity - Polarization Splitting

文档说 "characteristics shift to shorter wavelengths" 和 "s-characteristic becomes stronger, p becomes weaker".

### Wavelength Shift

Phase thickness 变成:
$$\delta(\theta) = \frac{2\pi n d \cos\theta_f}{\lambda}$$

其中 $\theta_f$ 是 film 内 angle (Snell: $n_0 \sin\theta_0 = n_f \sin\theta_f$).

QW condition $\delta = \pi/2$ 对应:
$$\lambda_{eff}(\theta) = 4 n_f d_f \cos\theta_f$$

随 $\theta_0$ 增加, $\cos\theta_f$ 减小, $\lambda_{eff}$ 减小 → **蓝移**.

粗略估计: small angle, $\Delta\lambda/\lambda \approx -\theta_0^2/(2 n_0^2)$.

### Polarization Splitting

对 s-polarization, effective admittance: $y_s = n\cos\theta$
对 p-polarization, effective admittance: $y_p = n/\cos\theta$

所以 s 和 p 看到"不同的薄膜" (相同 physical thickness, 不同 optical admittance). 特征曲线分裂.

特别地, **Brewster angle** 时 p-polarization 的 reflection 消失, 这是文档说 p-characteristic "weaker" 的根源.

### Iridescence 应用

文档给出的 decorative coating:
- Chromium (2.1nm) - 半透明 absorber
- MgF2 (500nm) - dielectric spacer
- Aluminum (thick) - back reflector

这是一个 **etalon with absorption**. 不同 angle 不同 path length → 不同 interference color. 这种"颜色随角度变化"的特性 dye 做不到, 所以 anti-counterfeit 用 (US currency 上的 color-shifting ink).

参考: [Optical coating tilt effects](https://www.thinfilmcenter.com/), [Color-shifting ink Wikipedia](https://en.wikipedia.org/wiki/Optically_variable_ink)

---

## 8. 制造工艺 - 深层直觉

文档展示了 Optorun thermal evaporation, Shincron RAS (ion-assisted), Bühler Helios (magnetron sputtering).

### Thermal Evaporation 的限制

热蒸发粒子能量 ~0.1 eV. 到达 substrate 后, mobility 低, 形成 **columnar microstructure** (Thornton diagram 中的 Zone 1 结构). 这种结构有 **voids**, density 只有 bulk 的 80-95%. 

后果:
1. Refractive index 偏低, 不可预测
2. **Moisture absorption**: 水 ($n=1.33$) 进入 voids, $n_{film}$ 上升, **spectral shift** (文档没提, 但实际中 ~5-20 nm 蓝移到正常)
3. 环境稳定性差

### Ion-Assisted Deposition (IAD) 和 Sputtering

文档的 RAS 用 **radical source** (oxygen plasma, ~100eV 离子辅助) 提高 packing density. Magnetron sputtering (Bühler Helios) 用 ~10-100 eV Ar+ bombard, 达到接近 bulk density.

**Sputtering 的优势**: 稳定, repeatable, density 高 (99%+ bulk), moisture-insensitive. 但 deposition rate 低, machine 复杂昂贵.

### Process Monitoring

文档说 Gemini mirror uniformity <1nm. 这是 in-situ optical monitoring (or quartz crystal monitor) + planetary rotation 实现的. 对 telecom filter (158 层, 通道间距 100 GHz), 总厚度误差要 < 0.1% 来保证 passband edge 位置. 这需要 **optical monitoring with error compensation** (Macleod 的 Essential Macleod 软件做这个).

参考: [Ion-assisted deposition review](https://www.sciencedirect.com/science/article/pii/S0040609015002373), [Thornton structure zone model](https://en.wikipedia.org/wiki/Thornton_diagram)

---

## 9. 几个补充联想 (Bonus)

### Low-E Glass 的原理
文档说 "correctly chosen high index thin film can reduce reflectance of thin silver virtually to zero". 

这是经典的 **D/M/D 三层** (dielectric/metal/dielectric) 结构:
- Glass / SnO2 (~40nm) / Ag (~12nm) / SnO2 (~40nm)
- Visible region: D/M/D 是 **antireflection coating for Ag** - 让 visible 透过
- IR region: QW condition 不满足, Ag 的 intrinsic reflectance 主导 - 反射 IR (热)

让建筑冬天保温 (反射室内 IR 回室内), 夏天隔热 (反射室外 IR). 全球年需求 >10^9 m².

### Energy-Efficient Halogen Lamp
100 层 all-dielectric IR reflector, 折射 IR 回 filament. Ta2O5/SiO2 stack 能耐受 ~400°C. 文档说的 "refractory dielectric materials" - 候选包括 TiO2, HfO2, Al2O3 (高 melting point).

### Oil on Water Color Simulation
文档图示把 0-500nm 楔形油膜颜色计算叠在实际照片上. 

油膜颜色来自 thin-film interference in reflection:
$$I_{refl}(\lambda) = |r_{air/oil} + r_{oil/water} \cdot e^{i \cdot 4\pi n_{oil} d / \lambda}|^2$$

对 D65 daylight source 积分得到 RGB. Macleod 提到"agreement is close" - 因为 thin film optics 是 *mature theory* with negligible unknown physics.

---

## 整体 Takeaway

薄膜光学的核心 intuition:

1. **Admittance transformation**: 每层 film 是 matrix, 把 back-side admittance 转换成 front-side admittance. Quarter-wave 是"unitary transform"把 $Y$ 变成 $Y_f^2/Y$.

2. **Phase is everything**: 在 transparent dielectric regime, 一切都是 phase engineering. Reflectance/transmittance = phase interference. Pulse shaping = phase derivative engineering.

3. **Materials constrain design**: MgF2 1.38 是最低 index. Ta2O5 2.1, TiO2 2.3, SiO2 1.46 - 这些"标准"材料的选择决定 $n_H/n_L$ ratio, 进而决定 stop band 宽度, AR bandwidth, 等等. 

4. **Causality constrains phase**: Kramers-Kronig 关系意味着 reflectance 和 phase 不能独立 design. Macleod 提到的 "pulse arrives sooner before reaching mirror" 的失败是 KK 的体现.

5. **Manufacturing defines real performance**: 5nm layer 误差可能完全破坏 telecom filter 158 层设计. 实际 coating engineering = design + process control + monitoring + yield optimization.

薄膜光学是 *one of the oldest and newest branches of optics* - 19 世纪 Rayleigh, Fabry, Perot 奠基, 今天仍然在 quantum computing mirrors, LIGO, EUV lithography, AR/VR coatings 等前沿驱动新材料和新工艺.

主要参考资源:
- [Thin Film Center (Macleod 公司)](http://www.thinfilmcenter.com/)
- [Essential Macleod 软件](http://www.thinfilmcenter.com/essential.htm)
- [Macleod's textbook](https://www.routledge.com/Thin-Film-Optical-Filters-Fourth-Edition/Macleod/p/book/9781420073014)
- [Optical Society of America - Applied Optics](https://www.osapublishing.org/ao/)
- [SVC (Society of Vacuum Coaters)](https://www.svc.org/)
- [Surface Plasmon Resonance - Homola 2008 review](https://pubs.acs.org/doi/10.1021/cr068107d)
- [Chirped mirror original paper](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-13-2710)
- [Kramers-Kronig relations in optics](https://en.wikipedia.org/wiki/Kramers%E2%80%93Kronig_relations)
