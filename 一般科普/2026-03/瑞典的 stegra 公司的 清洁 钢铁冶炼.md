# Stegra 公司清洁钢铁冶炼技术详解

---

## 1. 公司背景概述

**Stegra**（原名 H2 Green Steel）是一家瑞典清洁技术公司，成立于 2020 年，总部位于 Stockholm。该公司致力于通过 **green hydrogen-based Direct Reduced Iron (DRI)** 技术实现钢铁生产的深度 decarbonization。

> 官网：https://www.stegra.com/

---

## 2. 传统钢铁冶炼 vs Stegra 清洁技术

### 2.1 传统 Blast Furnace (BF) 工艺

传统钢铁生产依赖 **Blast Furnace-Basic Oxygen Furnace (BF-BOF)** 路线：

**核心化学反应：**

$$\text{Fe}_2\text{O}_3 + 3\text{CO} \rightarrow 2\text{Fe} + 3\text{CO}_2$$

**问题：**
- 每生产 **1 tonne crude steel** 排放约 **1.8-2.2 tonnes CO₂**
- 全球钢铁行业贡献约 **7-9%** 的 global CO₂ emissions

### 2.2 Stegra 的 Green DRI 技术路线

Stegra 采用 **Hydrogen-Based Direct Reduced Iron (H₂-DRI)** 配合 **Electric Arc Furnace (EAF)**：

**核心还原反应：**

$$\text{Fe}_2\text{O}_3 + 3\text{H}_2 \rightarrow 2\text{Fe} + 3\text{H}_2\text{O}$$

**关键区别：**
- **Reducing agent** 从 carbon monoxide (CO) 替换为 **green hydrogen (H₂)**
- **By-product** 从 CO₂ 变为 **water vapor (H₂O)**
- 理论上可实现 **>95% carbon emission reduction**

---

## 3. Stegra 技术架构详解

### 3.1 整体工艺流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Stegra Green Steel Production Flow                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐   │
│   │  Renew-  │───▶│ Electrolyzer │───▶│   Green     │───▶│    DRI        │   │
│   │  able    │    │   (PEM/SOEC) │    │  Hydrogen   │    │   Shaft       │   │
│   │  Energy  │    │              │    │   (H₂)      │    │   Furnace     │   │
│   │ (Wind/   │    └──────────────┘    └─────────────┘    │               │   │
│   │  Solar/  │                                       │   │  Fe₂O₃ + 3H₂  │   │
│   │  Hydro)  │                                       │   │     ↓         │   │
│   └──────────┘                                       │   │  2Fe + 3H₂O   │   │
│                                                      │   └───────┬───────┘   │
│                                                      │           │           │
│   ┌──────────┐                                       │           ▼           │
│   │  Iron    │───────────────────────────────────────┼──────▶ Sponge Iron   │
│   │  Ore     │                                       │     (DRI, ~90% Fe)   │
│   │ Pellets  │                                       │           │           │
│   └──────────┘                                       │           ▼           │
│                                                      │   ┌───────────────┐   │
│   ┌──────────┐                                       │   │   Electric    │   │
│   │  Green   │───────────────────────────────────────┘   │   Arc         │   │
│   │  Electr. │                                           │   Furnace     │   │
│   │          │───────────────────────────────────────▶   │   (EAF)       │   │
│   └──────────┘                                           └───────┬───────┘   │
│                                                                  │           │
│                                                                  ▼           │
│                                                          ┌───────────────┐   │
│                                                          │  Green Steel  │   │
│                                                          │  (<0.1 t CO₂/ │   │
│                                                          │   t steel)    │   │
│                                                          └───────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键子系统详解

---

#### 3.2.1 Green Hydrogen Production

**技术选择：PEM Electrolyzer (Proton Exchange Membrane)**

**电化学反应公式：**

**阳极:**
$$2\text{H}_2\text{O} \rightarrow \text{O}_2 + 4\text{H}^+ + 4e^-$$

**阴极:**
$$4\text{H}^+ + 4e^- \rightarrow 2\text{H}_2$$

**总反应:**
$$2\text{H}_2\text{O} \xrightarrow{\text{electricity}} 2\text{H}_2 + \text{O}_2$$

**关键参数：**

| Parameter | Value | Unit |
|-----------|-------|------|
| **Efficiency (LHV)** | 65-75 | % |
| **Electricity consumption** | 50-55 | kWh/kg H₂ |
| **H₂ purity** | >99.999 | % |
| **Operating pressure** | 30-80 | bar |
| **Operating temperature** | 50-80 | °C |

**变量说明：**
- $\text{H}^+$：proton（氢离子）
- $e^-$：electron（电子）
- LHV：Lower Heating Value（低热值）

---

#### 3.2.2 DRI Shaft Furnace 核心设计

**Stegra 采用的 Shaft Furnace 类型：MIDREX 或 Energiron 改进型**

**反应区分层结构：**

```
┌────────────────────────────────────────────────┐
│           DRI Shaft Furnace Structure           │
├────────────────────────────────────────────────┤
│                                                 │
│   ┌─────────────────────────────────────────┐  │
│   │         Feed Zone (Top)                  │  │
│   │   Iron ore pellets entry                 │  │
│   │   Temperature: ~200°C                    │  │
│   └─────────────────────────────────────────┘  │
│                      ▼                         │
│   ┌─────────────────────────────────────────┐  │
│   │      Preheating Zone                     │  │
│   │   - Remove moisture                      │  │
│   │   - Pre-heat to 600-800°C                │  │
│   └─────────────────────────────────────────┘  │
│                      ▼                         │
│   ┌─────────────────────────────────────────┐  │
│   │      Reduction Zone (Critical)           │  │
│   │   Main H₂ injection point                │  │
│   │   Temperature: 800-1000°C                │  │
│   │   Residence time: 4-6 hours              │  │
│   │                                          │  │
│   │   Fe₂O₃ → Fe₃O₄ → FeO → Fe              │  │
│   │   (Hematite→Magnetite→Wüstite→Iron)      │  │
│   └─────────────────────────────────────────┘  │
│                      ▼                         │
│   ┌─────────────────────────────────────────┐  │
│   │      Cooling Zone (Bottom)               │  │
│   │   - Cool to <100°C                       │  │
│   │   - Passivation to prevent re-oxidation  │  │
│   └─────────────────────────────────────────┘  │
│                      ▼                         │
│              DRI Product Output                │
│           (Sponge Iron, 90-95% metallization)  │
└────────────────────────────────────────────────┘
```

**分步还原反应机理：**

**Step 1: Hematite → Magnetite**
$$3\text{Fe}_2\text{O}_3 + \text{H}_2 \rightarrow 2\text{Fe}_3\text{O}_4 + \text{H}_2\text{O}$$

**Step 2: Magnetite → Wüstite**
$$\text{Fe}_3\text{O}_4 + \text{H}_2 \rightarrow 3\text{FeO} + \text{H}_2\text{O}$$

**Step 3: Wüstite → Metallic Iron**
$$\text{FeO} + \text{H}_2 \rightarrow \text{Fe} + \text{H}_2\text{O}$$

**Thermodynamics 分析：**

**Gibbs Free Energy 变化：**
$$\Delta G = \Delta H - T\Delta S$$

对于反应 $\text{FeO} + \text{H}_2 \rightarrow \text{Fe} + \text{H}_2\text{O}$：

$$\Delta G^\circ_{1000K} \approx -15 \text{ kJ/mol}$$

**说明：**
- $\Delta G$：Gibbs free energy change（吉布斯自由能变化）
- $\Delta H$：enthalpy change（焓变）
- $T$：temperature in Kelvin（开尔文温度）
- $\Delta S$：entropy change（熵变）
- 负值表示反应在 1000K 下 thermodynamically favorable

---

#### 3.2.3 Electric Arc Furnace (EAF)

**Stegra EAF 设计特点：**

**电弧加热原理：**
$$Q = P \cdot t = I^2 \cdot R \cdot t = V \cdot I \cdot t$$

**能量平衡方程：**
$$Q_{\text{input}} = Q_{\text{steel}} + Q_{\text{slag}} + Q_{\text{loss}} + Q_{\text{off-gas}}$$

**关键参数表：**

| Parameter | Conventional EAF | Stegra EAF |
|-----------|------------------|------------|
| **Tap-to-tap time** | 45-60 min | 35-50 min |
| **Electricity consumption** | 380-450 kWh/t | 350-420 kWh/t |
| **Electrode consumption** | 1.5-2.5 kg/t | <1.5 kg/t |
| **DRI input ratio** | 0-30% | 80-100% |
| **Carbon input** | High | Minimal |

**变量说明：**
- $Q$：heat energy（热能）
- $P$：power（功率）
- $I$：current（电流）
- $R$：resistance（电阻）
- $V$：voltage（电压）
- $t$：time（时间）

---

## 4. Stegra Boden 工厂项目详情

### 4.1 项目规格

**Location:** Boden, Norrbotten, Sweden

**设计产能数据表：**

| Phase | Capacity | Timeline | CAPEX |
|-------|----------|----------|-------|
| **Phase 1** | 2.5 Mt DRI + 1.5 Mt steel | 2025-2026 | ~€3.5 billion |
| **Phase 2** | 5.0 Mt DRI + 2.5 Mt steel | 2027-2030 | Additional €2-3 billion |

### 4.2 能源供应架构

**Boden 地区 renewable energy 资源：**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stegra Energy Supply System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ Hydropower   │         │ Wind Power   │                      │
│  │ (Lule River) │         │ (Onshore)    │                      │
│  │              │         │              │                      │
│  │ Capacity:    │         │ Capacity:    │                      │
│  │ ~2.5 GW      │         │ ~1.5 GW      │                      │
│  │ CF: ~45%     │         │ CF: ~35%     │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         └────────────┬───────────┘                               │
│                      ▼                                            │
│         ┌────────────────────────┐                               │
│         │   Grid Connection      │                               │
│         │   (Regional Grid)      │                               │
│         └────────────┬───────────┘                               │
│                      ▼                                            │
│         ┌────────────────────────┐                               │
│         │   PPA Agreements       │                               │
│         │   (Power Purchase      │                               │
│         │    Agreements)         │                               │
│         └────────────┬───────────┘                               │
│                      ▼                                            │
│         ┌────────────────────────┐                               │
│         │  Stegra Plant          │                               │
│         │  Total Demand:         │                               │
│         │  ~700 MW electrolyzer  │                               │
│         │  ~300 MW EAF/others    │                               │
│         └────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

**变量说明：**
- CF: Capacity Factor（容量因子）
- PPA: Power Purchase Agreement（购电协议）

---

## 5. 碳足迹与生命周期评估

### 5.1 Carbon Intensity 对比

**数据对比表：**

| Production Route | CO₂ Emissions (kg CO₂/t steel) |
|------------------|-------------------------------|
| **BF-BOF (Traditional)** | 1800-2200 |
| **BF-BOF with CCUS** | 800-1200 |
| **Natural Gas DRI-EAF** | 900-1200 |
| **Stegra H₂-DRI-EAF** | <50 (target: <25) |

### 5.2 Life Cycle Assessment (LCA) 边界

**Cradle-to-Gate Emissions 计算框架：**

$$\text{Total CO}_2 = \text{CO}_{2,\text{scope1}} + \text{CO}_{2,\text{scope2}} + \text{CO}_{2,\text{scope3,\text{upstream}}}$$

**Scope 分类：**

| Scope | Source | Stegra Status |
|-------|--------|---------------|
| **Scope 1** | Direct emissions from process | Near zero (H₂-based) |
| **Scope 2** | Indirect from electricity | Near zero (100% renewable PPA) |
| **Scope 3** | Value chain upstream/downstream | Focus area (ore mining, transport) |

---

## 6. 技术挑战与创新解决方案

### 6.1 核心技术挑战

**挑战 1: Hydrogen Storage and Handling**

**问题描述：**
- DRI 工艺需要 **continuous** H₂ supply
- H₂ 具有低密度、易泄漏特性
- Storage 需要 **high pressure** 或 **cryogenic** 条件

**Stegra 解决方案：**

**Storage 计算公式：**

**压缩氢气储能密度：**
$$E_{\text{density}} = \frac{n \cdot \text{LHV}_{\text{H}_2}}{V} = \frac{P}{RT} \cdot \text{LHV}_{\text{H}_2}$$

**变量说明：**
- $n$：moles of H₂（氢气摩尔数）
- $\text{LHV}_{\text{H}_2}$：Lower Heating Value of H₂ = 120 MJ/kg（氢气低热值）
- $V$：storage volume（储存体积）
- $P$：pressure（压力）
- $R$：gas constant = 8.314 J/(mol·K)（气体常数）
- $T$：temperature（温度）

---

**挑战 2: DRI Metallization Rate**

**目标：** Achieve >94% metallization

**Metallization Rate 定义：**
$$\text{Metallization} = \frac{\text{Fe}_{\text{metallic}}}{\text{Fe}_{\text{total}}} \times 100\%$$

**影响因素分析：**

| Factor | Impact | Optimization |
|--------|--------|--------------|
| **Temperature** | Higher T → faster reduction, but risk of sticking | Optimal: 850-950°C |
| **H₂/Fe ratio** | Excess H₂ drives equilibrium | Stoichiometric + 20% |
| **Pellet size** | Smaller → better gas penetration | 10-15 mm optimal |
| **Residence time** | Longer → higher metallization | 4-6 hours |

---

**挑战 3: DRI Re-oxidation Prevention**

**问题：**
- Hot DRI 在接触空气时容易 **re-oxidize**
- Re-oxidation 放热，可能导致 **self-ignition**

**Stegra 解决方案：**

**方案 A: Hot Briquetted Iron (HBI)**
$$\text{DRI} \xrightarrow{\text{briquetting}} \text{HBI}$$

**HBI 参数：**
- Density: >5.0 g/cm³
- Size: 90×60×30 mm
- Porosity: <20%

**方案 B: Inertization**
- 使用 **nitrogen (N₂)** 或 **CO₂** 氛围保护
- 添加 **passivation coating**

---

## 7. 经济性分析

### 7.1 Cost Breakdown Structure

**Levelized Cost of Steel (LCOS) 计算：**

$$\text{LCOS} = \frac{\sum_{t=1}^{n} \frac{C_t}{(1+r)^t}}{\sum_{t=1}^{n} \frac{Q_t}{(1+r)^t}}$$

**变量说明：**
- $C_t$：total cost in year $t$（第t年总成本）
- $Q_t$：steel production in year $t$（第t年钢铁产量）
- $r$：discount rate（折现率）
- $n$：project lifetime (years)（项目寿命）

### 7.2 成本结构对比

**数据表：**

| Cost Component | BF-BOF (%) | Stegra H₂-DRI-EAF (%) |
|----------------|------------|----------------------|
| **Iron ore** | 25-30 | 20-25 |
| **Energy (coal)** | 20-25 | - |
| **Energy (H₂)** | - | 35-45 |
| **Electricity** | 5-8 | 15-20 |
| **Capital** | 15-20 | 25-30 |
| **Labor & O&M** | 10-15 | 10-15 |

### 7.3 Green Premium 分析

**Green Steel Price Premium:**

| Market Segment | Premium Range | Timeline |
|----------------|---------------|----------|
| **Automotive OEM** | €50-150/t | 2025-2030 |
| **Construction** | €30-80/t | 2027-2035 |
| **Consumer goods** | €100-200/t | 2025-2030 |

---

## 8. 竞争格局与行业对标

### 8.1 主要竞争者对比

| Company | Country | Technology | Target Capacity | Status |
|---------|---------|------------|-----------------|--------|
| **Stegra** | Sweden | H₂-DRI-EAF | 5 Mt/y | Construction |
| **HYBRIT (SSAB)** | Sweden | H₂-DRI-EAF | 1.3 Mt/y | Pilot→Demo |
| **Salzgitter** | Germany | H₂-DRI-EAF | 1.9 Mt/y | Planning |
| **ArcelorMittal** | Luxembourg | H₂-DRI-EAF | 2.1 Mt/y | Multi-site |
| **POSCO** | South Korea | FINEX+H₂ | 2.0 Mt/y | R&D |
| **Baowu** | China | H₂-DRI-EAF | 1.0 Mt/y | Pilot |

### 8.2 HYBRIT vs Stegra 技术对比

**HYBRIT (SSAB, LKAB, Vattenfall) 项目特点：**

| Aspect | HYBRIT | Stegra |
|--------|--------|--------|
| **Technology origin** | Incremental from existing SSAB | Greenfield design |
| **Timeline** | Pilot 2021, Demo 2026 | Commercial 2026 |
| **Scale** | Conservative, phased | Aggressive, large-scale |
| **H₂ source** | Vattenfall integrated | Market + own production |
| **Target market** | SSAB internal steel | Open market supply |

---

## 9. 政策与监管环境

### 9.1 EU 政策支持

**关键政策框架：**

1. **EU Emissions Trading System (ETS)**
   - Carbon price: €80-100/t CO₂ (2024-2025)
   - Free allowances phase-out by 2034

2. **Carbon Border Adjustment Mechanism (CBAM)**
   - 对进口钢铁征收 carbon border tax
   - 保护 EU green steel 竞争力

3. **EU Hydrogen Strategy**
   - Target: 10 Mt domestic green H₂ by 2030
   - Funding: €470 billion investment needed

### 9.2 瑞典国家支持

**Reindustrialisation 支持：**
- **Industriklivet** program: €500 million for industrial transformation
- **Regional development funds:** Norrbotten priority region
- **Grid connection subsidies:** For renewable energy projects

---

## 10. 技术路线图与里程碑

### 10.1 Stegra 发展时间线

```
2020 ───────────────────────────────────────────────────────────── 2026
  │                                                                 │
  │  2020: Company founded                                          │
  │         │                                                        │
  │         ▼                                                        │
  │  2021: Feasibility studies, site selection (Boden)              │
  │         │                                                        │
  │         ▼                                                        │
  │  2022: Financing secured (€1.5B equity), EPC contracts          │
  │         │                                                        │
  │         ▼                                                        │
  │  2023: Construction start, infrastructure development           │
  │         │                                                        │
  │         ▼                                                        │
  │  2024: Electrolyzer installation, H₂ plant commissioning        │
  │         │                                                        │
  │         ▼                                                        │
  │  2025: Cold commissioning, pilot production                     │
  │         │                                                        │
  │         ▼                                                        │
  │  2026: Commercial production start (Phase 1)                    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### 10.2 技术成熟度评估

**Technology Readiness Level (TRL):**

| Component | TRL (2024) | Target TRL |
|-----------|-----------|------------|
| **PEM Electrolyzer** | 9 | 9 |
| **H₂-DRI Shaft Furnace** | 7-8 | 9 |
| **EAF with 100% DRI** | 8 | 9 |
| **Integrated system** | 6-7 | 9 |

**TRL 定义：**
- TRL 1-3: Basic research
- TRL 4-6: Technology development
- TRL 7-8: System demonstration
- TRL 9: Full commercial deployment

---

## 11. 风险与不确定性分析

### 11.1 技术风险

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Electrolyzer degradation** | Medium | High | Redundant capacity, preventive maintenance |
| **DRI quality variability** | Medium | Medium | Process control, AI optimization |
| **H₂ supply interruption** | Low | Critical | On-site buffer storage, backup supply |

### 11.2 市场风险

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Green premium erosion** | Medium | High | Long-term offtake contracts |
| **Renewable electricity price volatility** | High | High | PPAs, on-site generation |
| **Competitor scaling faster** | Medium | Medium | First-mover advantage, brand value |

### 11.3 First Principles 风险分析

**从热力学第一性原理分析：**

**最低理论能耗：**

对于 H₂ 还原 Fe₂O₃：

$$\Delta H_{\text{reaction}} = \sum \Delta H_f(\text{products}) - \sum \Delta H_f(\text{reactants})$$

$$\Delta H_{298K} = 3 \times (-241.8) - 1 \times (-824.2) = 101.6 \text{ kJ/mol Fe}_2\text{O}_3$$

**理论最低 H₂ 消耗：**
- 1 mole Fe₂O₃ 需要 3 moles H₂
- 160 g Fe₂O₃ 产生 112 g Fe
- 因此：$\frac{3 \times 2}{112} = 53.6 \text{ g H}_2/\text{kg Fe}$

**实际消耗（考虑效率）：**
- 工业实际：**55-65 kg H₂/t steel**
- Stegra 目标：**<55 kg H₂/t steel**

---

## 12. 未来展望与技术演进

### 12.1 技术迭代方向

**Generation 1 → Generation 2 改进：**

| Improvement Area | Gen 1 (2025) | Gen 2 (2030+) |
|------------------|--------------|---------------|
| **Electrolyzer efficiency** | 65-70% LHV | 75-80% LHV |
| **H₂ consumption** | 55-60 kg/t | <50 kg/t |
| **DRI metallization** | 92-94% | >96% |
| **Carbon capture integration** | None | optional BECCS |

### 12.2 产业链延伸

**Stegra 潜在 downstream 整合：**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stegra Value Chain Extension                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Upstream               Core               Downstream           │
│  ─────────             ───────             ──────────            │
│                                                                  │
│  Iron Ore Mining ───▶ H₂-DRI-EAF ───▶ Hot Rolled Coil           │
│         │                    │              │                    │
│         ▼                    ▼              ▼                    │
│  Pelletizing          Steelmaking      Automotive OEM           │
│         │                    │              │                    │
│         ▼                    ▼              ▼                    │
│  Logistics            Continuous        Construction            │
│         │             Casting              │                     │
│         ▼                    │              ▼                     │
│  Renewable Energy            ▼           Appliances             │
│         │              Finishing Lines                          │
│         ▼                    │                                   │
│  Electrolyzer H₂              ▼                                   │
│                          Coating/                               │
│                          Galvanizing                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. 参考资源与链接

### 13.1 官方资源

1. **Stegra Official Website**
   https://www.stegra.com/

2. **Stegra Press Releases**
   https://www.stegra.com/newsroom

3. **HYBRIT Project (Comparison)**
   https://www.hybritdevelopment.se/

### 13.2 技术参考文献

4. **IEA - Iron and Steel Technology Roadmap**
   https://www.iea.org/reports/iron-and-steel-technology-roadmap

5. **World Steel Association - Steel Statistical Yearbook**
   https://www.worldsteel.org/steel-by-topic/statistics.html

6. **European Commission - EU ETS**
   https://ec.europa.eu/clima/eu-action/eu-emissions-trading-system-eu-ets_en

7. **Boden Municipality - Stegra Project**
   https://www.boden.se/naringsliv/stora-investeringar/stegra/

### 13.3 学术论文

8. **Vogl et al. (2018) - "Assessment of hydrogen direct reduction for fossil-free steelmaking"**
   Journal of Cleaner Production
   https://doi.org/10.1016/j.jclepro.2018.02.279

9. **Fischedick et al. (2014) - "Techno-economic evaluation of innovative steel production technologies"**
   Journal of Cleaner Production
   https://doi.org/10.1016/j.jclepro.2014.04.075

10. **Toktarova et al. (2020) - "Iron and steel industry emissions analysis"**
    Journal of Industrial Ecology
    https://doi.org/10.1111/jiec.13029

---

## 14. 总结：Stegra 技术的核心洞察

**从 First Principles 理解 Stegra 的创新本质：**

### 14.1 根本问题重定义

传统钢铁冶炼的本质问题是：
$$\text{Fe}_2\text{O}_3 + \text{C} \rightarrow \text{Fe} + \text{CO}_2$$

**问题根源：** Carbon 作为 reducing agent 必然产生 CO₂

### 14.2 Stegra 的解法

**替换 fundamental agent：**
$$\text{Reducing agent: C} \rightarrow \text{H}_2$$

$$\text{Fe}_2\text{O}_3 + \text{H}_2 \rightarrow \text{Fe} + \text{H}_2\text{O}$$

**这个替换是本质性的，而非边际改进**

### 14.3 系统级挑战

替换 reducing agent 不是单一技术点，而是整个 system redesign：

1. **Energy system:** Coal → Renewable electricity → Hydrogen
2. **Process equipment:** Blast Furnace → DRI Shaft
3. **Melting technology:** BOF → EAF
4. **Material handling:** Molten iron → Solid DRI
5. **Infrastructure:** Coal logistics → H₂ infrastructure

### 14.4 Stegra 的战略定位

- **Greenfield advantage:** 无 legacy asset burden
- **Location choice:** Norrbotten = renewable energy + iron ore
- **Scale ambition:** World's first large-scale commercial H₂-DRI
- **Timing:** First-mover in emerging green steel market

---

**结论：** Stegra 代表了钢铁行业 decarbonization 的 most ambitious commercial effort。其成功与否将取决于技术 execution、成本控制、以及市场对 green premium 的接受度。从 first principles 分析，H₂-DRI 路线是实现 steel decarbonization 的 most promising path，而 Stegra 正处于这一转型的前沿。