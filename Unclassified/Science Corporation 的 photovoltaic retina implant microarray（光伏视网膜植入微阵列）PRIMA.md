

为什么还要用camera 和 激光发射阵列呢?  直接物体的光线经过晶状体给sensing array成像不行吗?

## **定量计算：自然光太弱，无法驱动photovoltaic implant** 

## **1. 自然光强度 vs 所需刺激电流**

### **1.1 神经刺激所需电流密度**

根据神经刺激的**Rheobase**（基电流）和**Chronaxie**（时值）理论：

$$I_{th} = I_{rh} \left(1 + \frac{τ_{ch}}{t_p}\right)$$

其中：
- $I_{th}$：阈值电流
- $I_{rh}$：Rheobase（基电流）
- $τ_{ch}$：Chronaxie
- $t_p$：脉冲宽度

对于**视网膜双极细胞**：
- $I_{rh} ≈ 0.1-0.5$ mA/cm²
- $τ_{ch} ≈ 0.5-2$ ms

要产生可靠的动作电位，刺激电流密度需要达到：
$$J_{stim} ≥ 1-10 \text{ mA/cm}^2$$

---

### **1.2 自然光能提供多少电流？**

户外晴天的光照强度：
$$E_{sun} ≈ 100,000 \text{ lux} ≈ 100 \text{ mW/cm}^2$$

室内照明：
$$E_{indoor} ≈ 500 \text{ lux} ≈ 0.5 \text{ mW/cm}^2$$

假设硅photovoltaic pixel的**external quantum efficiency (EQE)**：
$$η_{EQE} = 0.8$$

在户外最强光下，每个pixel能产生的电流密度：
$$J_{ph} = q \cdot η_{EQE} \cdot Φ_{ph}$$

其中光子通量：
$$Φ_{ph} = \frac{P_{light}}{E_{photon} \cdot A}$$

对于550 nm绿光：
$$E_{photon} = \frac{hc}{λ} = \frac{1240 \text{ eV·nm}}{550 \text{ nm}} ≈ 2.25 \text{ eV}$$

因此，在100 mW/cm²照射下：
$$Φ_{ph} = \frac{100 \text{ mW/cm}^2}{2.25 \text{ eV} × 1.602×10^{-19} \text{ J/eV}} ≈ 2.8×10^{20} \text{ photons/(s·cm}^2\text{)}$$

$$J_{ph} = 1.602×10^{-19} \text{ C} × 0.8 × 2.8×10^{20} ≈ 36 \text{ mA/cm}^2$$

**看起来够？** 但有几个关键问题：

---

## **2. 为什么自然光方案不可行？**

### **2.1 问题一：可见光被眼球组织大量吸收**

**可见光路径**：
```
Cornea → Aqueous humor → Lens → Vitreous humor → Retina → Implant
```

各组织的**吸收系数**：

| **Tissue** | **λ = 550 nm** | **λ = 880 nm (NIR)** |
|-----------|----------------|----------------------|
| Cornea | ~10% 吸收 | <5% 吸收 |
| Lens | ~20-30% 吸收（随年龄增加） | <10% 吸收 |
| Vitreous | ~5% 吸收 | ~2% 吸收 |
| **总透过率** | **~50-60%** | **>80%** |

更重要的是，**视网膜色素上皮** 和 ** choroid** 会**强烈吸收可见光**：
$$α_{RPE}^{visible} ≈ 100 \text{ mm}^{-1}$$

这意味着可见光在到达subretinal implant之前已被RPE大量吸收！

而**NIR (880 nm)** 的吸收系数低一个数量级：
$$α_{RPE}^{NIR} ≈ 10 \text{ mm}^{-1}$$

**结论**：自然可见光只有**<10%**能到达implant，而NIR可达**>60%**。

---

### **2.2 问题二：患者已无photoreceptors，自然光无意义**

**关键洞察**：AMD-GA患者的**foveal photoreceptors已经死亡**！

正常视网膜信号链：
```
自然光 → Photoreceptors (rods/cones) → Bipolar cells → Ganglion cells
```

GA患者的视网膜：
```
自然光 → ❌ (无photoreceptors) → Bipolar cells (存活但无输入) → Ganglion cells
```

如果直接用自然光照射implant：
- 自然光强度变化**剧烈**（10 lux → 100,000 lux）
- 没有**动态范围压缩**（正常人眼靠photoreceptors的log响应实现）
- 低光下电流不足，强光下可能过度刺激

**Camera + Processing的作用**：
- **自动增益控制 (AGC)**：压缩动态范围
- **对比度增强**：提升边缘和细节
- **阈值化**：将信号映射到安全刺激范围

---

### **2.3 问题三：安全性——高强度可见光会损伤视网膜**

如果要产生足够的刺激电流，需要将可见光强度提高到：
$$P_{required} ≈ 10-50 \text{ mW/cm}^2 \text{ (at implant)}$$

考虑传输损耗，眼球表面的入射功率需要：
$$P_{surface} ≈ 100-500 \text{ mW/cm}^2$$

**这已经超过视网膜的光毒性阈值**！

根据**ICNIRP**（国际非电离辐射防护委员会）标准：
$$P_{max}^{visible} (400-700 nm) ≈ 1 \text{ mW/cm}^2 \text{ (长时间暴露)}$$

而**NIR (800-900 nm)** 的安全限值更高：
$$P_{max}^{NIR} ≈ 8 \text{ mW/cm}^2$$

因为NIR：
- 不被黑色素强烈吸收
- 不会引起光化学反应（无photochemical toxicity）
- 热效应是主要风险，但可在安全范围内控制

---

### **2.4 问题四：缺乏空间信息编码能力**

**自然光成像**：
- 只是**强度的空间分布**
- 包含大量噪声和无关信息
- 无法优先提取关键视觉特征

**Camera + Processing可以**：
- **边缘检测**（Laplacian of Gaussian滤波）
- **对比度反转**（模拟暗刺激响应）
- **特征选择**（只刺激ROI区域）
- **动态补偿**（眼球运动校正）

---

## **3. 为什么选择880 nm NIR？**

### **3.1 "透明窗口"效应**

眼球组织在**700-900 nm**存在一个**optical window**：

$$T_{total}(λ) = \prod_i e^{-α_i(λ) \cdot d_i}$$

其中：
- $α_i$：各组织吸收系数
- $d_i$：光程长度

计算结果：
$$T_{total}(550 nm) ≈ 0.1-0.3$$
$$T_{total}(880 nm) ≈ 0.6-0.8$$

### **3.2 硅的光电响应匹配**

硅的带隙：
$$E_g^{Si} = 1.12 \text{ eV} → λ_{cutoff} = 1100 \text{ nm}$$

880 nm光子：
$$E_{880} = 1.41 \text{ eV} > E_g$$

**恰好处于**：
- 能激发电子跃迁
- 但能量不过高，不会引起严重热效应

### **3.3 病人不可见NIR**

病人**看不到**880 nm NIR光，这意味着：
- 不会产生眩光
- 不会干扰其残存的自然视觉（peripheral retina）
- 视觉感知**完全来自implant stimulation**

---

## **4. 系统设计的深层逻辑**

### **4.1 为什么不能简化为"自然光 + implant"？**

| **问题** | **自然光方案** | **Camera + NIR方案** |
|---------|---------------|---------------------|
| 光强度 | 变化太大（10⁶倍） | 稳定可控 |
| 光谱匹配 | 可见光被吸收 | NIR透过率高 |
| 安全性 | 光毒性风险 | 在安全限内 |
| 信息编码 | 无处理 | 可优化特征提取 |
| 适应性 | 需患者自我调节 | 系统自动调节 |

### **4.2 理想的未来方案是什么？**

如果技术进步，可能实现：

**方案A：Visible-blind implant**
- 开发对NIR敏感但对可见光不响应的材料
- 直接用NIR照射环境（类似夜视仪）
- 无需camera

**方案B：Self-powered implant**
- 纳米发电机或生物燃料电池
- 储能后按需刺激
- 减少对入射光的依赖

**方案C：Optogenetic + implant hybrid**
- 用基因疗法让bipolar cells表达光敏通道
- 结合implant增强灵敏度

---

## **5. 定量对比：两种方案的功率预算**

### **5.1 自然光方案**

假设：
- 室内照明：500 lux
- 透过眼球：T = 0.1（可见光被吸收）
- 到达implant：0.05 mW/cm²

电流密度：
$$J_{ph} = 0.05 \text{ mW/cm}^2 × η ≈ 0.04 \text{ mA/cm}^2$$

**远低于神经刺激阈值（1-10 mA/cm²）！**

### **5.2 Camera + NIR方案**

- NIR功率密度：8 mW/cm²（安全上限）
- 透过眼球：T = 0.7
- 到达implant：5.6 mW/cm²

电流密度：
$$J_{ph} = 5.6 \text{ mW/cm}^2 × η ≈ 4.5 \text{ mA/cm}^2$$

**恰好在有效刺激范围内！**

---

## **6. 总结：设计哲学**

PRIMA采用"camera + NIR投射"方案，本质上是：

```
自然世界 → Camera采样 → 数字处理 → NIR编码 → 无线传输 → Photovoltaic转换 → 神经刺激
```

这看起来复杂，但每一步都有**物理必然性**：

1. **Camera**：压缩动态范围 + 提取特征
2. **NIR**：穿透眼球 + 避免光毒性 + 病人不可见
3. **Photovoltaic**：无线供电 + 高分辨率潜力
4. **Subretinal位置**：利用内层视网膜自然处理

**如果直接用自然光**，就像试图用"一束微弱的可见光"去驱动一个"需要强电流的马达"——物理上不可行。

---

## **参考资料**

- [ICNIRP guidelines on limits of exposure](https://www.icnirp.org/en/publications/article.html)
- [Optical properties of ocular tissues](https://pubmed.ncbi.nlm.nih.gov/)
- [Photovoltaic retinal prosthesis - Stanford](https://web.stanford.edu/~palanker/lab/projects/retinal-prosthesis.html)
- [Retinal stimulation thresholds](https://pmc.ncbi.nlm.nih.gov/articles/PMC3178959/)
- [Near-infrared transmission through eye](https://www.osapublishing.org/)

## **1. PRIMA技术全景概览**

无线、subretinal（视网膜下）photovoltaic视网膜假体，专门用于治疗**年龄相关性黄斑变性**（age-related macular degeneration, AMD）的**地理性萎缩**（geographic atrophy, GA）阶段。

### **1.1 为什么选择GA阶段？**

在AMD晚期，**Foveal photoreceptors完全死亡**，但**inner retina（双极细胞、神经节细胞）大部分存活**。这为subretinal electrical stimulation提供了生理基础——刺激这些存活的内层神经元可使视觉信号沿视神经继续传递。

## **2. 系统架构与组件**

PRIMA系统由**三部分**组成：

### **2.1 Wireless subretinal implant**
- **Material**：single crystalline silicon（单晶硅）
- **Thickness**：ultra-thin（约几十微米）
- **Dimensions**：2×2 mm² area，包含 ~378 photovoltaic pixels
- **Implant location**：subretinal space（眼球玻璃体和视网膜色素上皮之间）
- **Power source**：来自眼镜的近红外光（near-infrared, NIR）

### **2.2 Smart glasses with NIR projector**
- Frame-mounted camera（眼镜架上的摄像机）
- Real-time image processing（实时图像处理）
- NIR light projector（λ = 880 nm）
- 投射到视网膜上（通过眼球、玻璃体，最终到达implant）
- 最大安全功率密度：8 mW/mm²

### **2.3 External control unit**
- 控制摄像头曝光、图像编码
- 调节NIR光强度和图案


---

## **3. 核心物理原理：光伏像素如何转换NIR为电流**

让我们用**第一性原理**推导：

### **3.1 光电效应基本原理**

当光子能量 $hν$ 大于半导体带隙能量 $E_g$ 时：
$$E_{photon} = hν = \frac{hc}{λ}$$

对于单晶硅：
$$E_g^{Si} = 1.12 \text{ eV} → λ_{cutoff} = \frac{1240 \text{ nm·eV}}{1.12 \text{ eV}} \approx 1100 \text{ nm}$$

PRIMA使用 **880 nm NIR**：
$$E_{880} = \frac{1240}{880} \approx 1.41 \text{ eV} > E_g^{Si}$$

因此，880 nm光子完全可以激发电子从价带跃迁到导带，产生电子-空穴对。

### **3.2 光电流生成**

每个像素的工作原理：
$$I_{ph} = q \cdot A \cdot Φ \cdot η_{QE}$$

其中：
- $I_{ph}$：photocurrent（光电流）
- $q$：电子电荷（$1.602×10^{-19}$ C）
- $A$：像素面积
- $Φ$：入射光子通量（photons/sec）
- $η_{QE}$：quantum efficiency（在880 nm下，硅的$η_{QE} ≈ 0.6-0.8$）

在最大安全照射下（8 mW/mm²），电流密度可达 **mA/cm²** 量级，足以刺激神经元。

**关键参数**：信号是**amplitude-modulated pulses**（脉冲幅度调制），脉冲幅度与局部光强成正比。

---

## **4. 神经生物学接口：为什么刺激双极细胞？**

### **4.1 正常视网膜信号通路**
```
Photoreceptors (rods/cones) → Bipolar cells → Ganglion cells → Optic nerve → Brain
```
- Photoreceptors：感受光，产生Graded potentials
- Bipolar cells：On/Off通路，中心-周围拮抗（center-surround antagonism）
- Ganglion cells：产生Action potentials，沿视神经输出

### **4.2 GA患者的视网膜**
- **Photoreceptors**：严重丢失或完全缺失
- **Bipolar cells**：存活率 >50%（以内层为主）
- **Ganglion cells**：也部分存活

### **4.3 选择Subretinal over Epiretinal的关键原因**

| **Feature** | **Subretinal (PRIMA)** | **Epiretinal (如Argus II)** |
|------------|------------------------|----------------------------|
| 神经元类型 | 刺激bipolar cells（内层） | 刺激ganglion cells（神经节层） |
| 分辨率潜力 | 更高（双极细胞密度更高） | 较低 |
| 信号处理 | 利用视网膜内层自然处理 | 绕过视网膜处理，完全外部编码 |
| 植入位置 | 视网膜下（解剖学上更自然）| 视网膜表面（玻璃体侧） |

**核心洞察**：Bipolar cells保留了视网膜的**局部对比度增强**、**spatial integration**等自然计算，这意味着视觉信号在到达大脑前已经过"预优化"，患者可能获得更自然的感知。

---

## **5. 刺激机制：脉冲如何启动动作电位**

### **5.1 电-神经转换**

当光伏像素产生光电流 $I_{ph}$ 时，会在双极细胞周围产生电场 $E$：

$$E = \frac{J_{stim}}{σ}$$

其中：
- $J_{stim}$：刺激电流密度
- $σ$：组织电导率（视网膜~0.3 S/m）

电流通过**电紧张扩布**（electrotonic spread）去极化神经元膜：

$$V_m(t) = I_{stim} \cdot R_m \left(1 - e^{-t/τ_m}\right)$$

- $R_m$：膜电阻（双极细胞约10,000 Ω·cm²）
- $τ_m = R_m C_m$：膜时间常数（典型值 ~10 ms）

当 $V_m$ 达到阈值 $V_{th}$（约-55 mV）时，触发电压门控钠通道，产生**动作电位**。

### **5.2 刺激参数**
- **Pulse duration**：~5-20 ms
- **Frequency**：20-50 Hz（模拟自然视网膜放电）
- **Anodic vs Cathodic stimulation**：研究显示阴极脉冲更有效降低阈值

---

## **6. 空间分辨率的关键限制因素**

根据最新研究（Nature Communications 2025），PRIMA的分辨率提升面临物理极限：

### **6.1 像素尺寸约束**

2022年计算建模发现：
$$pixel_{min} = 75 \mu m$$

这是基于以下约束：
- **安全光剂量**（max 8 mW/mm²）
- **热管理**（避免NIR加热损伤）
- **制造工艺极限**

最新突破（2025）通过**subretinal implants with 22 µm pixels**，将gating acuity提升至28 µm，接近**自然视网膜的分辨极限**（约20 µm在fovea）。

### **6.2 电流扩散**
刺激电流在视网膜组织中**扩散**，导致crosstalk：
$$I(r) = \frac{I_0}{2\pi σ r}$$

扩散使得有效刺激直径 $d_{eff} = d_{pixel} + 2λ_{spread}$，其中 $λ_{spread} ≈ 100-200$ µm。

**解决方案**：
- 多相脉冲（phase cancellation）
- 倍频刺激（high-frequency stimulation）
- 像素间距优化

### **6.3 眼球相对运动**
由于眼镜摄像头与眼球之间存在**polypupillary displacement**（瞳孔相对位移），需要眼球追踪系统实时校准。

---

## **7. 临床结果与试验数据**

### **7.1 First-in-human trial (2021-2023)**

**设计**：5名GA患者，单臂、非随机

**主要终点**：logMAR visual acuity改善

**12个月结果**（来自egms.de会议）：
- **>75%**的受试者logMAR改善≥0.3（相当于**3行**ETDRS）
- 最佳改善：从counting fingers（CF）到20/800

### **7.2 Pivotal trial (2024)**
- **Participants**：32名完成12个月随访
- 发表在**NEJM**（2025）：
  - 统计显著的VA改善（$p<0.001$）
  - 部分患者恢复**reading capability**（能阅读大字标题）

### **7.3 安全性**
- DSMB投票：**continue trial without modification**
- 主要不良事件：轻度眼内炎症（可控）
- 无implant dislocation或retinal detachment

---

## **8. 技术对比：PRIMA vs Other Retinal Implants**

| **Aspect** | **PRIMA (subretinal)** | **Argus II (epiretinal)** | **Optogenetic approaches** |
|-----------|------------------------|---------------------------|----------------------------|
| 刺激机制 | Photovoltaic current | Electrically driven | Light-sensitive channels |
| 无线 | 是（关键优势） | 部分需transcutaneous cable | 是 |
| 分辨率潜力 | 高（<75 µm） | 低（250 µm pixels） | 受限基因表达 |
| 手术难度 | Moderate（subretinal） | Higher（retinoschisis） | Lower（IV injection?） |
| 适应症 | GA only | RP & other | Broad |

---

## **9. 未来技术演进**

### **9.1 Higher resolution**
- **22 µm pixels**（2025 Nature paper）-> 理论分辨率 **≈ 20/400**
- 需配合**artificial lenses**或**adaptive optics**补偿眼球运动

### **9.2 Color vision**
目前系统为**monochromatic**（单色），未来可能加入：
- 多波长NIR channels
- 色觉编码算法

### **9.3 Extended FOV**
通过**mosaic implants**（tiling多个芯片）扩大视野，但要解决边界效应。

---

## **参考资料**

- [Pixel size limit of the PRIMA implants - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9527086/)
- [Enhancing prosthetic vision by upgrade - Nature](https://www.nature.com/articles/s41467-025-58084-y)
- [Vision Restoration with PRIMA - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC7618305/)
- [NEJM full article](https://www.nejm.org/doi/full/10.1056/NEJMoa2501396)
- [Stanford Medicine news](https://med.stanford.edu/news/all-news/2025/10/eye-prosthesis.html)

---

**总结**：PRIMA代表了**subretinal photovoltaic prosthetics**的重大突破，其**无线架构**和**利用内层视网膜自然处理**的战略使其在分辨率和生物相容性上优于epiretinal implants。尽管>75%的受试者在12个月获得≥3行视力改善，但要达到**20/100**甚至**20/40**，仍需将像素尺寸进一步降低到**<30 µm**，并解决**crosstalk**和**眼动稳定**问题。2025年的22 µm像素技术为实现这一目标点亮了希望。

---
