













# Substrate：用 X-ray lithography 颠覆半导体制造的美国 Foundry 创业公司

---

## 一、公司概述

**Substrate** 是一家总部位于 San Francisco 的半导体 foundry 创业公司，由 **James Proud**（Peter Thiel Fellow, 2011 届）于 **2022 年** 创立。公司的使命是 **"Returning the United States to dominance in semiconductor production"**——让美国重新主导半导体制造。

> 🔗 官网：https://substrate.com/
> 🔗 About 页面：https://substrate.com/about
> 🔗 Purpose 页面：https://substrate.com/our-purpose

---

## 二、核心技术创新：X-ray Lithography（XRL）

### 2.1 传统路径 vs Substrate 路径

| 参数 | EUV（ASML 路径） | X-ray Lithography（Substrate 路径） |
|---|---|---|
| **光源波长 λ** | 13.5 nm | < 10 nm（soft X-ray ~0.1–10 nm） |
| **光源产生方式** | CO₂ laser + Sn droplet plasma | **Compact particle accelerator**（小型粒子加速器） |
| **单台设备成本** | ~$300M+ | 声称为 EUV 的 **1/10** |
| **分辨率目标** | 5 nm node → 2 nm（需 multi-patterning） | **直接实现 2 nm-class** patterning |
| **光强** | ~EUV 功率有限 | 声称比太阳亮 **数十亿倍** |
| **Patterning 步骤** | 多次曝光（multi-patterning） | 更少步骤（波长更短 → 单次可达到更精细 pattern） |

### 2.2 关键物理原理

Lithography 的分辨率受 **Rayleigh 判据** 约束：

$$R = k_1 \cdot \frac{\lambda}{NA}$$

其中：
- $R$ = 最小可分辨特征尺寸
- $k_1$ = process factor（工艺系数，通常 0.25–0.5）
- $\lambda$ = 光源波长
- $NA$ = Numerical Aperture（数值孔径）

**核心 insight：** 缩短 $\lambda$ 是提升分辨率最直接的物理路径。EUV 的 $\lambda = 13.5$ nm，而 X-ray 的 $\lambda$ 可以是 **1 nm 以下**——这意味着即使 NA 相同、$k_1$ 相同，理论分辨率也可以提升 **一个数量级以上**。

Substrate 的核心主张就是：**与其在 EUV 波长上通过增加 NA 和 multi-patterning 来挤牙膏式地提升分辨率，不如直接用更短的 X-ray 波长来一步到位。**

### 2.3 Particle Accelerator 如何产生 X-ray

Substrate 使用 **compact particle accelerator** 作为 X-ray 源，其工作原理基于 **Synchrotron Radiation（同步辐射）**：

1. **Electron injection**：电子枪产生高能电子束
2. **Acceleration**：电子在 accelerator 结构中被加速到 **接近光速**（relativistic speed）
3. **Magnetic bending**：电子经过弯曲磁铁（bending magnets）或 undulators/wigglers 时，做圆周/振荡运动
4. **Synchrotron radiation emission**：根据经典电动力学，**做加速运动的带电粒子会辐射电磁波**。当电子速度接近光速时，辐射的功率和频率大幅提升（Larmor 公式的 relativistic 推广）：

$$P = \frac{q^2 a^2}{6\pi \epsilon_0 c^3} \gamma^4$$

其中：
- $P$ = 辐射功率
- $q$ = 电子电荷
- $a$ = 加速度
- $\epsilon_0$ = 真空介电常数
- $c$ = 光速
- $\gamma$ = Lorentz factor = $\frac{1}{\sqrt{1 - v^2/c^2}}$

5. **X-ray 输出**：当 $\gamma$ 足够大时（高能电子），辐射的峰值频率落在 X-ray 波段，产生 **数十亿倍太阳亮度** 的相干/准相干 X-ray 束

Substrate 的创新在于将传统只有国家实验室才有的 **大型同步辐射光源**（如 Diamond Light Source、SSRF 等）**miniaturize** 成可以放进 foundry 的紧凑型 accelerator 系统。

---

## 三、商业模式：垂直整合 Foundry

Substrate 不同于 ASML（只卖 lithography 设备给 TSMC/Samsung/Intel），而是采取 **fully vertically integrated model**：

```
┌──────────────────────────────────────────────────────────┐
│                    Substrate Vertical Stack               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: Particle Accelerator Design & Manufacturing    │
│           ↕ (产生 X-ray 光源)                             │
│  Layer 2: Lithography System Integration                  │
│           ↕ (将 X-ray 光源集成到 lithography 工具)          │
│  Layer 3: Fabrication Facility (Fab) Operation            │
│           ↕ (自建产线，直接代工芯片)                         │
│  Layer 4: Wafer Processing & Output                       │
│           ↕ (向客户交付完成的 wafer/chip)                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

这意味着 Substrate **既做 ASML 的活（造 lithography 设备），又做 TSMC 的活（运营 foundry 代工芯片）**。这是一个极其野心勃勃的模式——但也意味着它不需要依赖任何现有 lithography 供应商。

---

## 四、融资与估值

| 轮次 | 金额 | 领投方 | 估值 |
|---|---|---|---|
| 最新轮（2025年10月） | **$100M+** | General Catalyst, Founders Fund, In-Q-Tel（CIA 旗下基金）, Long Journey, Valor Equity Partners | **>$1B** |

> 🔗 General Catalyst 投资博客：https://www.generalcatalyst.com/stories/our-investment-in-substrate
> 🔗 Long Journey Ventures：https://www.longjourney.vc/news/substrate-second-believer

---

## 五、为什么这件事重要？（第一性原理分析）

### 5.1 当前半导体制造的瓶颈

现代先进芯片制造的核心瓶颈是 **lithography**。ASML 的 EUV 设备：

- 每台售价 **$300M+**
- 全球只有 ASML 一家能造
- 依赖极复杂的供应链（Zeiss 光学、Trumpf 激光等）
- 输出功率受限于 Sn plasma 源的亮度上限
- 需要 **multi-patterning** 才能达到 2nm node → 更多步骤 → 更多缺陷 → 更低 yield

### 5.2 Substrate 的第一性原理突破

从第一性原理出发，lithography 的核心需求是：

> **用更短的波长 → 更高的分辨率 → 更少的 patterning 步骤 → 更低的成本**

EUV 走到了 13.5 nm 波长就卡住了（NA 0.55 High-NA 是最后的挣扎），而 X-ray 天然在 **0.1–10 nm** 波段。问题从来不是 "X-ray 不能做 lithography"（学术界 1980 年代就验证过了），而是 **"如何以合理成本和 throughput 产生足够强的 X-ray 源"**。

Substrate 的答案就是 **compact particle accelerator**——将同步辐射光源从国家级设施缩小到工厂级别。

### 5.3 潜在风险与挑战

| 挑战 | 描述 |
|---|---|
| **Mask 制造** | X-ray 的 mask 技术难度极高（absorber 需要 ~1μm 厚度的 Ta/W 等高 Z 材料） |
| **Throughput** | 同步辐射光源的功率是否足以支撑 foundry 级别的 wafer/hour 产出？ |
| **Resist 材料** | X-ray 对 resist 的要求与 EUV 完全不同，需要新型 chemically amplified resist |
| **历史先例** | X-ray lithography 在 1990 年代曾被视为下一代技术但最终被放弃，主要因为 mask 和 alignment 问题 |
| **竞争** | ASML 也不会坐视不管，High-NA EUV 已经在推进 |
| **Yield** | Foundry 的核心是 yield，新技术需要数年时间证明 |

---

## 六、创始人背景

**James Proud**：
- 2011 年 **Peter Thiel Fellow**（Thiel Fellowship 给年轻人 $100K 让他们辍学创业）
- 20 岁时已卖掉第一家公司
- 之前做过 **Glow**（智能家居公司，曾在 Kickstarter 上筹到 $2.5M）
- 被 Forbes 报道过：https://www.forbes.com/profile/james-proud/

---

## 七、总结直觉

Substrate 的 story 可以用一个直觉来概括：

> **"EUV 是在 13.5 nm 波长上用尽一切工程手段挤分辨率；Substrate 则是换一个数量级的波长（X-ray），从物理层面直接绕过整个 EUV 的复杂度 mountain。"**

它本质上是一个 **物理层降维打击** 的策略——但这个 "降维" 能否在工程上实现（compact accelerator 的亮度/稳定性、mask、resist、alignment、throughput），是整个赌局的核心。

---

**参考链接：**
- 官网：https://substrate.com/
- Our Purpose：https://substrate.com/our-purpose
- General Catalyst 投资：https://www.generalcatalyst.com/stories/our-investment-in-substrate
- Tom's Hardware 报道：https://www.tomshardware.com/tech-industry/semiconductors/american-startup-substrate-promises-2nm-class-chipmaking-with-particle-accelerators-at-a-tenth-of-the-cost-of-euv-x-ray-lithography-system-has-potential-to-surpass-asmls-euv-scanners
- Medium 深度分析：https://medium.com/@cognidownunder/substrates-x-ray-bet-against-asml-and-tsmc-b900883cb3cc
- Sacra 数据：https://sacra.com/c/substrate/
- NextBigFuture：https://nextbigfuture.substack.com/p/substrate-inversion-semi-xlight-and
- Stratechery 采访（付费）：https://stratechery.com/2025/an-interview-with-substrate-ceo-james-proud-about-building-a-disruptive-foundry-in-america/