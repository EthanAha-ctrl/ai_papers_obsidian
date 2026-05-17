**Planckian.co** 是一家总部位于意大利 Pisa 的**量子技术公司**，专注于开发**超导量子计算机架构**。其核心成就是设计了一种解决量子芯片"布线瓶颈"（wiring problem）的新架构蓝图——即如何将大量控制线连接到超导量子比特上而不引入过多热噪声和信号串扰。


|项目|内容|
|---|---|
|**公司名**|Planckian|
|**官网**|[planckian.co](https://www.planckian.co/)|
|**所在地**|Pisa, Italy (意大利比萨，托斯卡纳大区)|
|**行业**|Quantum Computing / Nanotechnology Research|
|**员工规模**|~9 人 (小型 Deep Tech Startup)|
|**核心技术**|Superconducting Quantum Processor Architecture|
|**LinkedIn**|[Planckian LinkedIn](https://www.linkedin.com/company/planckian)|

Planckian 正在构建**下一代超导量子处理器 (next-generation superconducting quantum processor)**，其核心创新是解决量子计算中一个被长期忽视却至关重要的工程瓶颈——**布线问题 (Wiring Problem)**。

---

## 🔬 他们到底在解决什么问题？

### The Wiring Problem — 量子计算的 "最后一公里"

在超导量子计算中，每一个 qubit 需要一组控制线 (control wires) 用于：

- **Readout**：读取 qubit 状态
- **Gate operations**：执行量子门操作（如 X, Y, Z rotation gates）
- **Flux tuning**：调节 qubit 频率

传统的布线模型是 **1:1 的**：一个 qubit 需要对应 2–3 条控制线。当我们从当前工业界的 ~1000 qubits 扩展到 ~10,000+ qubits 时，布线数量线性增长：

Nwires≈α⋅NqubitsNwires​≈α⋅Nqubits​

其中 α∈[2,3]α∈[2,3] 是每个 qubit 需要的线数。

这导致的直接后果是：

1. **Heat load**：大量线缆从室温 (300K) 穿入 dilution refrigerator 的 mK 区域，引入巨大热负荷
2. **Space constraint**：冰箱内部物理空间有限，无法容纳海量同轴线缆
3. **Signal crosstalk**：线缆之间电磁串扰，降低 gate fidelity
4. **Manufacturing complexity**：布线制造和组装成本指数级上升

> 这就是 Planckian 所称的 **"Critical Wiring Problem"** — 它不是理论物理问题，而是一个工程实现问题，但正是这种工程瓶颈决定了量子计算能否真正 scale up。

---

## ⚡ Planckian 的解决方案：Decoupled Control Architecture

Planckian 的核心创新是一种**解耦控制架构 (Decoupled Control Architecture)**：

> **"By decoupling control and performance, it accelerates the development of practical quantum computers on the most advanced qubit platform."**  
> — Planckian LinkedIn

### 第一性原理解读

传统架构中，qubit 的**控制能力 (controllability)** 和**性能 (performance)** 是耦合的——为了获得高性能（高 T1T1​ time, 高 gate fidelity），你需要精细的控制线；而精细的控制线本身限制了扩展性。

Planckian 的思路是将这两者解耦：

Performance=f(qubit quality, materials, fabrication)⊥Controllability=g(architecture, wiring scheme)Performance=f(qubit quality, materials, fabrication)⊥Controllability=g(architecture, wiring scheme)

即：

- **Performance** 由 qubit 材料、fabrication 工艺决定（他们使用成熟的 transmon qubit 平台）
- **Controllability** 由架构层面的设计决定（他们的创新点）

这意味着你可以**在不牺牲 qubit 性能的前提下**，大幅减少物理布线需求。

### 技术细节推测

基于公开信息和领域知识，Planckian 的方案可能涉及以下技术路线：

1. **Frequency multiplexed readout**：多个 qubit 共享同一条 readout 线，通过不同谐振频率区分
2. **Global control + local addressing**：全局微波脉冲做单 qubit gate，仅 multi-qubit gate 需要局部寻址
3. **3D integration / flip-chip**：将控制电子学集成到芯片的另一面，减少外部线缆
4. **Cryogenic CMOS / SFQ control**：在 4K 或更低温度放置控制芯片，减少穿越温度梯度的线数

---

## 📰 重要里程碑

### 2024年12月17日 — 发布超导量子芯片蓝图

Planckian 于 2024 年 12 月 17 日发布了一项重要公告：

> _"Pisa, Italy – December 17, 2024 – Quantum technology company Planckian today announced the release of a **blueprint for a superconducting quantum chip** that addresses the critical wiring problem in quantum computing."_

来源：

- [EIN Presswire / Fox8 报道](https://fox8.com/business/press-releases/ein-presswire/769551054/planckian-develops-new-superconducting-quantum-computer-architecture-to-solve-critical-wiring-problem/)
- [InsideHPC 报道](https://insidehpc.com/2024/12/planckian-announces-architecture-addressing-quantum-wiring-problems/)

同时，一篇 MSN 报道标题为 **"Superconducting quantum processor performs well with significantly less wiring"**，很可能与 Planckian 的工作直接相关。

---

## 🔋 Quantum Battery — 另一个研究方向

根据 VentureRadar 的信息：

> _"Planckian is a company dedicated to the development of the technological concept of the **quantum battery**. In its architecture, quantum effects can be..."_

**Quantum Battery** 是量子热力学 (Quantum Thermodynamics) 的一个前沿方向，其核心思想是：

### 基本原理

在经典电池中，能量存储密度受限于：

ρE=EstoredVρE​=VEstored​​

而在量子电池中，利用**量子纠缠 (quantum entanglement)** 和 **quantum coherence**，可以实现：

ρEquantum>ρEclassicalρEquantum​>ρEclassical​

特别地，**charging speed** 可以通过纠缠获得超线性加速：

τcharge∝N−β,β>1τcharge​∝N−β,β>1

其中 NN 是电池中 entangled units 的数量，ββ 是加速指数。经典电池 β=1β=1（线性），而量子电池可以实现 β>1β>1 的**量子加速充电效应 (Quantum Speedup in Charging)**。

这是 Planckian 在 Nanotechnology Research 方面的另一个潜在研究方向。

---

## 🌐 竞争格局

|公司|方向|与 Planckian 的异同|
|---|---|---|
|**Google Quantum AI**|Superconducting qubits (Sycamore)|同为超导路线，但 Google 用传统布线架构|
|**IBM Quantum**|Superconducting qubits (Eagle, Heron)|IBM 也在探索 multiplexed readout，但未解耦控制|
|**IQM** (Finland)|Superconducting qubits|欧洲同行，但更偏整体系统集成|
|**SEEQC**|Superconducting + cryogenic CMOS|同样在 4K 层集成控制芯片，路线更接近|
|**Oxford Quantum Circuits**|3D coaxmon architecture|用 3D 腔体减少布线，思路不同|

Planckian 的独特定位在于：**专注于布线架构这一 "picks and shovels" 层面的基础创新**，而非试图与巨头在整体量子计算机上竞争。

---

## 💡 为什么这个方向重要？— 从第一性原理出发

量子计算的实用化有三大障碍：

Practical QC=Qubit Quality⏟T1,T2,fidelity×Qubit Quantity⏟scale to 103-106×Connectivity⏟wiring, couplingPractical QC=T1​,T2​,fidelityQubit Quality​​×scale to 103-106Qubit Quantity​​×wiring, couplingConnectivity​​

目前工业界的进展：

- ✅ Qubit Quality：transmon qubit 已经可以达到 T1>100μsT1​>100μs，2-qubit gate fidelity > 99.9%
- ⚠️ Qubit Quantity：Google Sycamore 72 qubits, IBM Heron 1000+ qubits，但布线已经逼近极限
- ❌ Connectivity：布线问题随着 qubit 数量非线性恶化

Planckian 攻克的是第二个和第三个维度之间的桥梁——**如何在增加 qubit 数量的同时保持合理的布线复杂度**。这决定了从 "lab demo" 到 "practical machine" 的飞跃能否实现。

---

## 📊 总结

|维度|内容|
|---|---|
|**一句话总结**|Planckian 通过解耦控制和性能的超导量子处理器架构，解决量子计算扩展中的布线瓶颈|
|**技术路线**|Superconducting qubits (transmon) + Decoupled control architecture|
|**核心价值主张**|显著减少布线需求，使大规模超导量子处理器成为可能|
|**阶段**|早期 — 已发布芯片蓝图 (blueprint)，尚在研发阶段|
|**地理位置优势**|意大利比萨，靠近 Scuola Normale Superiore 和 NEST Lab 等量子研究重镇|

---

### 参考链接

1. [Planckian 官网](https://www.planckian.co/)
2. [Planckian LinkedIn](https://www.linkedin.com/company/planckian)
3. [Planckian Crunchbase](https://www.crunchbase.com/organization/planckian)
4. [EIN Presswire: Planckian 发布超导量子架构 (2024.12.17)](https://fox8.com/business/press-releases/ein-presswire/769551054/planckian-develops-new-superconducting-quantum-computer-architecture-to-solve-critical-wiring-problem/)
5. [InsideHPC: Planckian Architecture 报道](https://insidehpc.com/2024/12/planckian-announces-architecture-addressing-quantum-wiring-problems/)
6. [Planckian VentureRadar Profile](https://www.ventureradar.com/organization/Planckian/fab3481db-b648-40de-9fd1-3e380aa25b05)
7. [Planckian PitchBook Profile](https://pitchbook.com/profiles/company/530059-69)