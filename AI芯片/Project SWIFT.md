# Project SWIFT：IBM 的「一日晶圆厂」——半导体制造自动化的先驱

这篇文章由 Jesse Aronstein（Project SWIFT 的设备组经理）亲自撰写，发表于 IEEE Spectrum，讲述了 1970-1975 年间 IBM 内部一个几乎被遗忘的项目——**Project SWIFT**，它实现了从裸晶圆到成品 IC 仅需约 **20 小时**的惊人壮举，而当时行业平均需要一个月以上。即使以今天的标准衡量，SWIFT 的 per-layer 处理时间（5 小时）仍远快于现代 fab 的行业均值（36 小时/层）。

---

## 一、背景与动机：为什么需要「一天造芯片」？

### 1.1 1970 年的半导体制造困境

1970 年代初，IC 制造（如 RAM 芯片）是典型的**停停走走**式生产：

- 晶圆在几十个手动工位之间排队等待
- **Raw process time**（实际加工时间）< 48 小时
- **Total turnaround time** > 一个月
- 绝大部分时间花在**等待**上

这本质上是经典的**排队论**问题。晶圆在系统中经历的延迟可以粗略建模为：

$$W_q \approx \frac{\lambda \cdot \bar{S}^2}{2(1-\rho)}$$

其中 $W_q$ 是平均排队等待时间，$\lambda$ 是到达率，$\bar{S}^2$ 是服务时间方差，$\rho = \lambda / \mu$ 是系统利用率。手动操作导致 $\bar{S}^2$ 极大，排队时间远超加工时间。

### 1.2 Bill Harding 的六大核心论断

Harding 在 1970 年的 MR（Manufacturing Research）部门会议上提出了六条纲领：

| # | 论断 | 当时行业常态 | 今日验证 |
|---|------|-------------|---------|
| 1 | **VLSI 将基于 FET 技术** | 双极型晶体管（bipolar）占主导 | ✅ CMOS（FET 的变体）是绝对主流 |
| 2 | **零缺陷高良率是首要目标** | 良率通常很低，靠筛选 | ✅ 现代 fab 良率管理是核心 |
| 3 | **制造应全自动化** | 手动操作为主 | ✅ 高度自动化 |
| 4 | **单 wafer 逐片处理最优** | Batch 处理为主 | ✅ 单片处理已成为关键步骤标准 |
| 5 | **短 turnaround 时间带来重大优势** | 未被重视 | ✅ 快速原型迭代是竞争优势 |
| 6 | **产能通过复制产线扩展** | 单条大产线 | ✅ "fab cloning" 概念 |

这六条中，**第 4 条（单 wafer 处理）** 和 **第 5 条（短 turnaround）** 是最激进的——它们直接否定了当时 batch 处理的范式。

---

## 二、SWIFT 的系统架构：从流水线到「Sector + Taxi」

### 2.1 架构演进

最初的概念是**线性流水线**：

```
Step 1 → Step 2 → Step 3 → ... → Step N
```

但三个因素打破了线性模型：

1. **设备维护与故障**：需要 **buffer** 在工位间暂存晶圆
2. **光刻步进器**：所有 pattern exposure 必须在同一台 stepper 上完成（避免 machine-to-machine variation）
3. **光刻次数**：RAM-II 需要 4 次光刻（3 次器件 + 1 次金属布线），意味着晶圆必须 4 次回到同一台 stepper

### 2.2 最终架构：Five Sectors + Monorail Taxi

最终架构被分为 **5 个 sector**，每个 sector 是一个汽车大小的密封箱：

```
                    ┌──────────────┐
                    │   Stepper    │
                    │ (10:1 / 1:1) │
                    └──────┬───────┘
                           │
              Taxi ────────┤
         (Monorail)        │
      ┌────────────────────┘
      │
Sector 1 → Sector 2 → Sector 3 → Sector 4 → Sector 5
(Litho     (Litho     (Litho     (Litho     (Final
 prep)      prep)      prep)      prep)      metal)
```

**关键设计逻辑**：

- 晶圆进入 sector 时：photoresist 已被曝光，等待 develop
- 晶圆离开 sector 时：已涂覆新的 photoresist，等待下次曝光
- Taxi 在 sector 之间运输单个晶圆，确保晶圆每次回到同一台 stepper

每个 sector 内部包含的模块：
- **Wet-chemistry module**：清洗、显影、去胶、刻蚀
- **Miniature furnaces**：热氧化、退火等
- **Photoresist application module**：旋涂光刻胶
- **Taxi pickup port**：等待出租车

### 2.3 控制系统：三层层级架构

```
Level 3: IBM 1800 Computer (ECS - Execution Control System)
         ├── 全线管理、记录、taxi 调度、process monitoring
         │
Level 2: Sector Controllers (5 个)
         ├── 每个 sector 一个，管理 sector 内晶圆物流
         ├── 向 Level 3 报告数据
         │
Level 1: Module-level Controllers
         ├── 各处理模块和 wafer-handling 子系统独立控制
         ├── 支持独立设置和维护
```

**ECS 的创新之处**：
- 每个 wafer 有唯一序列号，全程追踪
- 存储并监控每个 wafer 的工艺参数
- 实时检测 out-of-spec 情况并快速响应
- 使用 **punch cards 和 tape cartridges** 作为存储介质（当时的技术条件）

> 这本质上是现代 **MES（Manufacturing Execution System）** 的雏形。

---

## 三、关键工程创新

### 3.1 工艺时间压缩：从 48 小时到 15 小时

Kleinfelder 的工艺组分析了每一步的必要性：

- **消除不必要的步骤**：例如，如果晶圆快速从一个步骤进入下一个步骤，某些化学清洗可以省略（因为晶圆还没有足够时间被污染）
- **加速某些步骤**：通过调整参数缩短处理时间

$$t_{raw} = \sum_{i} t_{process,i}^{optimized} < 15 \text{ hrs}$$

而传统线：

$$t_{total} = \sum_{i} (t_{process,i} + t_{wait,i}) \approx 720+ \text{ hrs (30 days)}$$

核心洞察：**等待本身制造了需要额外步骤的条件**（如等待导致的污染需要额外清洗），形成恶性循环。消除等待 → 消除额外步骤 → 进一步缩短时间。

### 3.2 光刻：10:1 Reduction Stepper

East Fishkill 光刻组开发的 **10:1 缩比步进投影系统**：

- 非接触式（避免接触式掩模的缺陷转移问题）
- 10:1 缩放比 → 掩模上颗粒的"阴影"缩小 10 倍
- 更高的光学分辨率
- 更长的掩模寿命

代价：**速度慢** → 需要多台 stepper → 但每片晶圆必须回同一台 → 引入 taxi 调度

最终实际生产中，大部分芯片是用 **1:1 contact mask machine** 生产的，因为 throughput 更高。

### 3.3 机械设计哲学：可靠 > 优雅

#### Geneva Drive（日内瓦机构）

SWIFT 大量使用 **Geneva drive**（源自钟表工业）来实现离散、精确、自锁定的运动：

- 每转动输入轴一圈 → 输出端精确前进一步
- 运动终点自动锁定，无需额外制动
- 适合需要**精确定位**且**可靠性优先**的场景
- 代价：看起来运动是"分段式"的，不够流畅

$$\theta_{output} = f(\theta_{input})$$

Geneva drive 的运动特性：在每一步的起始和终止点角速度为零（**dwell**），中间有加速减速段，天然适合间歇运动。

#### 同步电机驱动 Spin Coater

传统旋涂机需要精确控制转速，而"转速错误"是常见 reject 原因。SWIFT 的解决方案：

- 使用 **synchronous AC motor** 直接锁定在 3,600 rpm
- 由 60 Hz 交流电源同步（类似电唱机转盘原理）
- **消除了4个独立的转速控制器**
- 薄膜厚度通过调节其他变量（温度 $T$、粘度 $\mu$、旋涂时间 $t$）控制

旋涂薄膜厚度近似公式：

$$h \approx \left(\frac{3\mu V_0}{2\pi \rho \omega^2 R^2}\right)^{1/3} \cdot t^{-1/3}$$

其中 $\omega$ 固定为 $2\pi \times 3600/60$ rad/s，只调 $\mu$、$T$、$t$。

#### Bernoulli Wafer Handler

利用气流在晶圆上方产生的 **Bernoulli 效应**：

- 气流速度 $v$ 增大 → 压力 $P$ 降低（Bernoulli 方程）
- 晶圆被"负压"吸起，但无物理接触

$$P + \frac{1}{2}\rho v^2 = \text{const}$$

这是现代 **Bernoulli gripper** 的前身，至今仍广泛用于 300mm 晶圆搬运。

---

## 四、实验结果与数据

### 4.1 关键性能指标

| 指标 | SWIFT 实测 | 现代 fab 参考 |
|------|-----------|-------------|
| **平均 turnaround** | ~20 小时 | 数周 |
| **Raw process time** | 14 小时 | - |
| **Per-layer 处理时间** | 5 小时 | 19 小时（最快）/ 36 小时（平均） |
| **Wafer throughput** | 58 片/天（83% 设计产能） | - |
| **良率** | 等于 East Fishkill 常规线的最高良率 | - |
| **最长连续运行** | 12 天 | - |
| **总产出** | 600 片产品级 wafer，17,000 颗 RAM-II FET | - |

### 4.2 五次连续运行

1974 年中至 1975 年初，共进行 5 次 continuous-operation run：

- 每次运行后分析结果并实施改进
- 逐步提升 throughput 和 yield
- 训练了来自 IBM 全球各点的 135 名技术人员、工程师和管理者

---

## 五、SWIFT 的遗产：从 QTAT 到现代 Fab

### 5.1 直接传承路线

```
Project SWIFT (1970-1975)
    ↓ (renamed)
FMS Feasibility Line
    ↓ (FS project canceled 1975)
    ↓ (部分设备)
QTAT Line (Quick Turn Around Time) — IBM 内部展示线
    ↓ (理念扩散)
现代自动化 Fab
```

FS（Future System）项目被取消后，FMS 变得多余。但 SWIFT 的设备和理念被移植到 East Fishkill 的 **QTAT 线**，这条线比 SWIFT 更为人所知。

### 5.2 SWIFT 的创新 vs. 今日 Fab 对比

| SWIFT 创新 (1970s) | 今日 Fab 对应 |
|---|---|
| 全自动化 wafer processing | ✅ 现代 fab 全自动化 |
| 计算机控制 (IBM 1800 ECS) | ✅ MES + APC (Advanced Process Control) |
| Monorail taxi 晶圆运输 | ✅ OHV (Overhead Hoist Vehicle) / AGV |
| Bernoulli wafer handler | ✅ 标准 300mm wafer handling |
| Stepper lithography (10:1) | ✅ Scanner/Stepper (如今是 EUV) |
| Real-time process control | ✅ FDC (Fault Detection & Classification) |
| 薄膜形成后立即涂胶 | ✅ "coat-after-dep" 减少污染 |
| Single-wafer processing | ✅ 关键步骤单片处理 |

---

## 六、第一性原理分析：为什么 SWIFT 这么快？

### 6.1 时间分解

用第一性原理分析 turnaround 时间：

$$T_{turnaround} = T_{process} + T_{wait} + T_{queue} + T_{transport} + T_{rework}$$

传统 fab 的主要时间消耗：

$$T_{wait} \gg T_{process}$$

$$T_{queue} \propto \frac{\sigma^2}{2(1-\rho)} \cdot \lambda$$

SWIFT 的压缩策略：

| 项 | 传统 | SWIFT | 压缩方式 |
|----|------|-------|---------|
| $T_{process}$ | 48 hrs | 14 hrs | 工艺优化 + 消除不必要步骤 |
| $T_{wait}$ | ~672 hrs | ~0 | 自动化连续流 |
| $T_{queue}$ | 大 | ~0 | 单片处理 + buffer |
| $T_{transport}$ | 大（人工搬运） | 小 | 自动化 taxi |
| $T_{rework}$ | 有 | 极少 | 实时 process control |

### 6.2 正反馈循环

SWIFT 发现了一个关键的**正反馈循环**：

```
快速流转 → 晶圆暴露时间短 → 污染少
    ↓                           ↓
不需要额外清洗 ← ← ← ← ← ← ← 
    ↓
工艺步骤减少
    ↓
Raw process time 进一步缩短
    ↓
Turnaround 更快
    ↓
循环继续
```

这是突破性的洞察：**速度本身就是质量保证**，而非速度与质量的 trade-off。

---

## 七、Harding 的管理哲学与组织创新

### 7.1 领导风格

- **二战老兵**，曾在巴顿将军第三集团军服役，三次负伤
- 粗犷但有效——不是典型的 IBM 风格
- 善于利用 **staff meeting 作为 presentation 的排练场**（观察反应，调整论点）
- 成功为项目争取了全程资金（约 3 年）

### 7.2 组织策略

- 从公司各处**招募顶尖人才**（如 Bevan Wu、Sam Campbell 的整个部门从 Endicott 迁至 East Fishkill）
- **保护团队免受干扰**，确保专注
- 在政治上做出妥协（如使用 Burlington 的 air-track、采用 IBM System/7），以换取组织支持

### 7.3 "The 51st Dragon" 的隐喻

Harding 在 staff meeting 朗读 Heywood Broun 的短篇小说《The 51st Dragon》，强调**命名/口号的力量**——"SWIFT" 这个名字本身就激励团队相信不可能的任务是可以完成的。

---

## 八、SWIFT 的局限与遗憾

1. **设备可靠性**（Achilles' heel）：定制设备故障率高
2. **对准仍需人工**：lithographic alignment 依赖操作员
3. **架构不统一**：被迫使用两种控制系统（4 个 custom controller + 1 个 System/7），增加维护难度
4. **Burlington air-track 的强制采用**：引入了晶圆污染和可靠性问题
5. **项目寿命太短**：FS 项目取消后，SWIFT 未能规模化

---

## 九、为什么今天的 Fab 没有 SWIFT 那么快？

这是最令人深思的问题。现代 fab per-layer 时间是 SWIFT 的 **4-7 倍**，原因包括：

| 因素 | 影响 |
|------|------|
| **晶圆尺寸** | 300mm vs. 1.25 inch（面积 ~57 倍），工艺时间成比例增加 |
| **层数** | 现代先进 IC 有 50+ 层，SWIFT 的 RAM-II 仅 4 层 |
| **工艺复杂度** | CMP、EUV lithography、3D NAND 的堆叠等，每步时间更长 |
| **检测需求** | 现代对 defect 的容忍度极低（nm 级），需要大量检测步骤 |
| **Batch vs. Single** | 某些步骤（如炉管氧化）仍用 batch 处理，增加排队 |

但即使考虑这些因素，**per-layer 5 小时 vs. 19-36 小时** 的差距仍说明现代 fab 还有大量时间花在等待和排队上。SWIFT 的核心洞察——**连续流单片处理 + 消除等待**——仍未被现代 fab 充分实现。

---

## 十、总结

Project SWIFT 是半导体制造史上的一个里程碑，其核心贡献可以总结为：

> **Bill Harding 在 1970 年就预见了半导体制造的全部未来：FET 主导、全自动化、单片处理、快速 turnaround、良率优先、复制扩展。他的团队在 3 年内证明了这一切可行。**

正如 Aronstein 在文末所写：

> *"William E. Harding is clearly the father of the modern, automated, billion-dollar fab."*

---

**参考链接：**
- IEEE Spectrum 原文: [Project SWIFT: The IBM Wafer Fab That Could Make ICs in a Day](https://spectrum.ieee.org/project-swift-ibm-wafer-fab)
- Computer History Museum - IBM Semiconductor Collection: [www.computerhistory.org](https://www.computerhistory.org)
- Heywood Broun, "The 51st Dragon": [Project Gutenberg](https://www.gutenberg.org/)
- Geneva Drive 原理: [Wikipedia - Geneva Drive](https://en.wikipedia.org/wiki/Geneva_drive)
- Bernoulli Gripper: [Wikipedia - Bernoulli Gripper](https://en.wikipedia.org/wiki/Bernoulli_grip)
- Semiconductor Manufacturing Equipment: [SEMI Standards](https://www.semi.org/en/products/standards)