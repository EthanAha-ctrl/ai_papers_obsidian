---
source_pdf: Programmable optoelectronic Ising machine for.pdf
paper_sha256: 9038ddba147f36f1a79a5cf80d83f98fdc66810fe8d2ced8304fab08bafe5110
processed_at: '2026-08-06T06:39:45-07:00'
target_folder: 2026-01
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

---

## 1. 这帮人在干嘛

想象你有一大堆选择要同时决定, 每个选择互相牵扯, 牵扯方式千奇百怪, 你要找一组让所有牵扯最"舒服"的组合。比如北京西二环堵车了, 160 组车每组分 3 条路, 你要告诉每组车走哪条, 让整个片区最不堵。这就是 combinatorial optimization problem (COP), 大部分是 NP-hard, 传统 CPU 算到天荒地老。

Ising machine 的 idea: 把每个选择变成一个"陀螺"(spin), 上旋或下旋代表选 A 还是选 B。陀螺之间用弹簧连起来, 弹簧的拉力代表问题的牵扯关系。然后你把这个系统一松手, 物理规律会让所有陀螺自然转到能量最低的姿态, 那个姿态就是最优解。

物理并行, 一瞬间的事, 不用 CPU 一个一个试。

---

## 2. Ising model 到底在说什么

公式 (1):

$$
H = -\sum_{1 \le i < j < N} J_{ij} \sigma_i \sigma_j
$$

人话翻译:
- $\sigma_i$ 是第 $i$ 个陀螺, 值 $+1$ 或 $-1$, 就是"你选 A 还是选 B"
- $J_{ij}$ 是陀螺 $i$ 和陀螺 $j$ 之间的弹簧, 正的拉力表示"希望同向转", 负的表示"希望反向转"
- $H$ 是整个系统的能量, 物理学铁律: 系统总是往能量低的地方滚
- 找到最低 $H$ 的陀螺组合, 就是最优解

为什么这对 COP 有用? 因为 [Lucas 2014](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full) 证明几乎所有 NP 问题(MAX-CUT, TSP, SAT, graph coloring, 走迷宫...)都能写成这个形式。你只要会解 Ising, 就会解一大堆 NP 问题。

MAX-CUT 的映射公式 (2):

$$
\text{cut value} = -\frac{1}{2}\Big(\sum_{1\le i<j<N} J_{ij} + H\Big)
$$

人话: 把图切成两半, 切掉的边权重之和最大, 等价于让 Ising 能量最低。$J_{ij}$ 求和是个常数, 所以最大化 cut value 等价于最小化 $H$。

---

## 3. 为什么用 oscillator 当陀螺

这是整个领域的核心 intuition, paper 里没细讲但很关键。

一个 parametric oscillator(pump 给它能量, 它自己振荡)在 threshold 附近有个奇妙特性: 它起振时, phase 会随机锁在 $0$ 或 $\pi$(两个对称稳态), 50/50 概率。这就是天然的 $\sigma = \pm 1$ binary spin。

多个这样的 oscillator 互相注入 coupling, 系统会依据 **minimum power dissipation principle**([Onsager 1931](https://journals.aps.org/pr/abstract/10.1103/PhysRev.37.405); [Vadlamani PNAS 2020](https://www.pnas.org/doi/10.1073/pnas.2015052117); [Leleu PRE 2017](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.022118))自然 relax 到能量最低态。物理直觉: 系统总想找最省力的振荡模式, 而最省力模式恰好对应 Ising ground state。

跟 simulated annealing 的对比:
- SA: 在 digital computer 上, 每次 flip 一个 spin, 算 $\Delta H$, 用 Metropolis criterion 决定接不接受。串行, 有 thermal fluctuation 帮助跳出 local minima
- Oscillator Ising machine: 所有 spin 同时连续演化, coupling 信号并行注入, 在 threshold 附近 noise 让系统自然"游走"在 energy landscape 上, collective bifurcation 帮它跳出 local minima

物理 parallel + 物理 escape mechanism, 这是 speedup 的根源。具体理论见 [Wang & Roychowdhury UCNC 2019](https://link.springer.com/chapter/10.1007/978-3-030-19311-9_13) 和 [Mohseni Nat. Rev. Phys. 2022](https://www.nature.com/articles/s42254-022-00440-y)。

---

## 4. 这篇 paper 到底新在哪

作者团队 2022 年已经做过一版 OEPO Ising machine([Cen et al. Light Sci. Appl. 11, 333](https://www.nature.com/articles/s41377-022-00946-8)), 但有几个硬伤:

### 4.1 老版本的痛

coupling 用 wavelength-division multiplexed optical delay-line network 实现, 也就是不同波长的光走不同长度的光纤, 形成 spin-spin 连接。问题:
- **Connectivity 受限**: 物理上能拉的 delay line 数量有限, 不能做到任意 $J_{ij}$
- **Precision 低**: optical delay line 强度控制精度差, $J_{ij}$ 只能取几个离散值
- **Problem-specific**: 换一个问题得换硬件

所以老版本只能解 toy MAX-CUT($J_{ij} \in \{-1, 0, 1\}$), 玩玩可以, real-world 没法用。

### 4.2 新版本的关键决策: 把 coupling 搬到 microwave domain

核心 idea: **optical 只负责产生 spin, microwave + FPGA 负责算 coupling**。

为什么这么搞:
- Optical domain 难做高精度: 16-bit 权重要超贵的光调制阵列, 串扰难管
- Microwave domain 天然适合: 频率 GHz 级, 商用 16-bit ADC/DAC 一抓一大把, FPGA 是 universal 的
- FPGA 做矩阵-vector 乘, 任意 $J_{ij}$ 都能算, 数字精度无上限

代价: 牺牲 optical 的纯模拟并行, 变成"光学产生 spin + 电子学算耦合"的 hybrid。但换来 **arbitrary coupling + 16-bit precision + programmability**, 这是 real-world COP 的硬门槛。

### 4.3 四个条件同时满足

这是 paper 真正的 contribution。前人的 Ising machine 多半卡在 1-2 个条件上:

| 系统 | Scalable | Programmable | Stable | High precision | Room temp |
|---|---|---|---|---|---|
| **OEIM (本工作)** | 4096 | 任意 J, 16-bit | 1.1h avg | 16-bit | 室温 |
| CIM ([Honjo 2021](https://www.science.org/doi/10.1126/sciadv.abh0952)) | 100k | 任意 J | 需锁相 | 高 | 室温但精密 |
| D-Wave 2000Q | 5k | limited topology | mK 级 | 有限 | mK |
| ROSC ([Moy 2022](https://www.nature.com/articles/s41928-022-00746-2)) | 1968 | on-chip fixed | 室温 | 低 | 室温 |
| SPIM ([Wang 2025](https://www.nature.com/articles/s42005-025-01467-y)) | large | low-rank 约束 | 室温 | 几 bit | 室温 |
| SBM ([Tatsumura 2021](https://www.nature.com/articles/s41928-021-00560-y)) | large | 任意 | 室温 | digital | 室温 |

OEIM 是第一个同时打到 scalable + programmable + stable + high-precision + room temp 五个 checkbox 的。Real-world COP 需要这五个全满足, 少一个都不行:
- 不 scalable → real-world 问题动辄几千变量
- 不 programmable → 换问题换硬件
- 不 stable → 跑到一半失稳, 解不可信
- 不 high precision → real-world 的 $J_{ij}$ 动态范围跨 16 个数量级(Fig. 5a, b), 几 bit 完全不够
- 不 room temp → 运维成本爆炸, 没法部署

---

## 5. 硬件怎么搭

### 5.1 主回路 (Fig. 1b)

```
1550nm pump laser 
    ↓
MZM (50 MHz pulse 调制)
    ↓
16 km SMF (单模光纤, 84μs loop delay)
    ↓
SOA (semiconductor optical amplifier, 提供 gain)
    ↓
PD (photodetector, 光转电)
    ↓
EA (electrical amplifier)
    ↓
BPF (bandpass filter, 滤出有用信号)
    ↓
Mixer + 20 GHz LO (注入 local oscillator, 产生 degenerate parametric oscillation)
    ↓
回到 MZM, 形成 closed loop
```

关键数字:
- Loop delay: 84 μs(光在 16 km 光纤里跑一圈的时间)
- Pulse repetition rate: 50 MHz, 即每 20 ns 一个 pulse
- 84 μs / 20 ns = **4200 个 OEPO pulse 同时存在于 loop 里**
- 每个 pulse 的 phase 锁在 0 或 π(相对 LO), 就是 Ising spin

### 5.2 Feedback loop

```
Optical splitter (从主回路分一部分光出来)
    ↓
PD (光转电)
    ↓
Mixer (解调 binary phase, 得到 spin 信号)
    ↓
ADC 16-bit (采样量化)
    ↓
FPGA (存 J 矩阵, 做矩阵-vector 乘: feedback = J × spin_vector)
    ↓
DAC 16-bit (输出 feedback 信号)
    ↓
上变频到 10 GHz carrier
    ↓
Microwave coupler (注入回主回路)
```

### 5.3 Time-multiplexing 规模化的直觉

这是 CIM 的经典套路([Marandi Nat. Photonics 2014](https://www.nature.com/articles/nphoton.2014.249); [Inagaki Science 2016](https://www.science.org/doi/10.1126/science.aah4243)): 一个 optical loop 里塞几千个 pulse, 每个 pulse 是一个 spin, 共享同一套硬件。N 个 spin 的 circulation time:

$$
T_{\text{circ}} = N \times T_{\text{pulse\_interval}} = N \times 20\text{ns}
$$

- 4096-spin MAX-CUT: $4096 \times 20\text{ns} = 81.92\text{μs} \approx 84\text{μs}$(匹配 loop delay)
- 485-spin traffic: $485 \times 20\text{ns} = 9.7\text{μs}$, 279 circulations ≈ 2.71 ms ✓

---

## 6. MAX-CUT 实验讲什么

### 6.1 I4096 problem

4096 nodes, fully connected, 8,386,560 edges, 权重 $J_{ij} \in \{-1, +1\}$。目标值用 SG3 (Sahni-Gonzales 3) 算 = 90,984([arxiv:2312.10895](https://arxiv.org/abs/2312.10895))。SA 跑 100 次, 两种 mode:
- Speed mode: 快但粗糙
- Accuracy mode: 慢但精确

### 6.2 核心结果 (Fig. 2, Table 1)

| | OEIM | SA speed mode | SA accuracy 40ms | SA accuracy 2s |
|---|---|---|---|---|
| Time-to-target | 1.97 ms (23 circ) | 19.26 ms | — | — |
| Best cut value | 98,236 | 93,475 | 97,564 | 99,129 |
| Mean cut value | 96,794 | — | 95,731 | 97,943 |

几个直觉:
- **40ms 内 OEIM 完胜**: best 高 672, mean 高 1063
- **给 SA 2 秒才能反超**: best 99,129 > 98,236, mean 97,943 > 96,794
- **OEIM 快 10 倍到 target**: 1.97ms vs 19.26ms

### 6.3 Density 鲁棒性 (Fig. 2c)

这是 OEIM 最 striking 的优势, 也是 paper 真正的 selling point。4096-node graph, 密度 1%, 10%, 50%, 100%, 100 次 run 看 time-to-target:

- **1% density**: SA 反而比 OEIM 快(sparse graph 上 SA 串行 flip 很快)
- **100% density**: SA time-to-target 大幅增加, variance 爆炸; OEIM 几乎不变

直觉解释:
- SA 每次 flip 一个 spin, 要更新 $O(N \times \text{density})$ 个邻居状态。dense graph 上单个 epoch 慢得要死
- OEIM 每 circulation 用 FPGA 矩阵乘一次性算完 feedback, **不管 density 多大, 单 circulation 时间固定**。物理并行天然对 dense graph 友好

这条结论对 real-world 很重要: real-world 问题几乎都是 dense graph(交通网络、社交网络、金融网络...), SA 在这里会退化得很厉害, OEIM 不会。

### 6.4 Evolution dynamics (Fig. 2a)

三个阶段:
1. **Circulations 1-23**: noise 长出来, 快速 bifurcation, cut value 急升
2. **Circulations 23-40**: 多数 spin amplitude 饱和
3. **Circulations 40+**: 少数 spin 继续 flip, 微调找更低 energy

跟 SA 的 monotonic descent 完全不同: OEIM 是 **collective bifurcation**, 像模拟退火的 parallel 版, 同时所有 spin 演化。这是 [simulated bifurcation algorithm](https://www.science.org/doi/10.1126/sciadv.aav2372) 的物理实现。

---

## 7. Stability 实验

### 7.1 怎么测

停 feedback, 让 4200 spins 随机锁定, 每 15 秒采集一次 spin 序列, 算和初始序列的 Pearson correlation $r$:
- $r > 0.95$ → 视为系统稳定([Altman & Krzywinski Nat. Methods 2015](https://www.nature.com/articles/nmeth.3697))

### 7.2 结果 (Fig. 3)

- 43 次测量: 平均 4250s (1.1h), 最佳 19785s (5.5h)
- 失稳机制: temperature drift → fiber length 变 → pulse timing 漂移 → spin 状态乱掉

### 7.3 为什么这个数字重要

CIM 的 DOPO 需要 phase-locked pump laser, 微小扰动就失稳。D-Wave 要 mK 级温度。OEIM 在室温跑 1.1h, 已经够实用。加上 temperature feedback control(16 km 光纤外面裹温控), 能推到 5.5h。这是 deployment 的硬指标: 你不能让交通管理系统每 10 秒重启一次。

---

## 8. Traffic Optimization — 真实世界 demo

### 8.1 问题设置

- **区域**: 北京西二环, West Railway Station → Gulou Avenue, ~50 km²
- **数据**: T-Drive dataset([Yuan SIGSPATIAL 2010](https://dl.acm.org/doi/10.1145/1869790.1869807); [KDD 2011](https://dl.acm.org/doi/10.1145/2020408.2020462)), 真实出租车 GPS
- **车辆**: 1200 辆 → 160 groups(每组 3 条 alternative routes), 1500 baseline 车不参与 diversion
- **模型**: User Equilibrium (UE)([Wardrop 1952](https://www.icevirtuallibrary.com/doi/10.1680/ipeds.1952.11362)), 交通领域经典

UE 模型目标函数可以 cast 成 Ising energy, spin 集对应 diversion 方案。

### 8.2 Spin 编码

- 每组车 3 条路 → 3 个 spin, spin-up = 选这条
- 约束: 每组只能选 1 条 → 需要 auxiliary spin
- 5 个 auxiliary spin 把带 external field 的 Ising 转成 zero-field 形式(OEIM 只能解 zero-field)
- 总 spin 数: $160 \times 3 + 5 = 485$

### 8.3 为什么这问题能验证 high-precision 的必要性

**$J_{ij}$ 动态范围跨 16 个数量级**(Fig. 5a, 5b heatmap)。交通网络里, 主干道车流密集(影响大), 支路车流稀疏(影响小), $J_{ij}$ 自然有巨大动态范围。之前的 optical delay-line 方案精度只有几 bit, 根本无法表达这种 $J$ 矩阵, 解出来全是 garbage。这就是 paper 强调 16-bit ADC/DAC + FPGA 的原因。

### 8.4 结果 (Fig. 5c, 5d, 6a)

| | OEIM | SA (自研) | PySA (NASA) |
|---|---|---|---|
| 平均时间 | 2.5 ms | 6.49 s | 也被击败 |
| 平均 Ising energy | −1,052,495 | −1,041,229 | — |
| 最佳时间 | 2.71 ms (279 circ) | — | — |
| 最佳 energy | −1,054,280 | −1,047,440 | — |

Reference Ising energy (IA method, 140s) = −1,054,280, OEIM 达到了这个值, SA 没达到。

直觉:
- **速度差 3 个数量级**: 2.5 ms vs 6.49 s。OEIM 每 circulation 9.7μs, 279 圈 = 2.71ms; SA 在 dense traffic graph 上每次 epoch 几十 ms
- **解质量更好**: OEIM 平均 energy 更低, 说明 diversion 方案更优

### 8.5 Evolution dynamics (Fig. 5c, 5d)

Traffic 问题的 evolution 比 MAX-CUT 复杂:
1. **Circulations 1-30**: 多数 spin 从 noise 长出来, energy 暂时上升(因为还没满足约束)
2. **Circulations 30-70**: 大规模 bifurcation, energy 急降, 一度低于 reference(因为约束还没完全 enforce)
3. **Circulations 70-279**: 少数 spin 受约束影响缓慢 flip
4. **Circulation 279**: 最后一个 spin flip → 达到 reference energy

这个三阶段 dynamics 是 Ising machine 解带约束问题的典型 pattern。约束通过 $J_{ij}$ 的特殊结构隐式 encode, spin 在演化过程中自然 satisfy 约束。

### 8.6 热力图对比 (Fig. 6b-d)

- 优化前: 西二环主路严重拥堵(红), 支路也堵(黄)
- OEIM 优化后(2.7 ms): 红区大幅缩小, 多数黄变绿, 拥堵明显缓解
- SA 优化后(6.46 s): 效果类似, 但慢了 2400 倍

---

## 9. Limitation 和下一步

### 9.1 Pump gain 没做 schedule

paper 在 Discussion 承认, 当前用固定 pump gain。理论([McMahon Science 2016](https://www.science.org/doi/10.1126/science.aah5178))表明:
- **强 gain** → 快收敛, 但探索空间窄, 易卡 local minima
- **弱 gain (near threshold)** → 慢, 但探索广, 更可能找到更好解

这跟 SA 的 temperature schedule 一模一样。下一步应该做 dynamic pump gain tuning, 类似 SA 的 cooling schedule。

### 9.2 FPGA 传输 bottleneck

J 矩阵传输 ~60s, 这对"实时 traffic management"是致命伤。4096×4096 dense J 是 268 MB, gigabit 传要分钟级。改进:
- HBM-based FPGA
- 增量更新 J
- 多 chip 互联(参考 SBM [Tatsumura 2021](https://www.nature.com/articles/s41928-021-00560-y))

### 9.3 规模化路径

- **增 repetition rate**: 50 MHz → GHz 级。但 GHz pulse 在 16 km 光纤里 dispersion 累积严重, 需要 dispersion compensation
- **延长 fiber**: 长 loop → 更多 pulse → 更多 spin, 但 circulation time 同步增加, latency 抵消速度优势
- 这俩要联合优化

### 9.4 高阶相互作用

很多 NP 问题(SAT, 3-SAT)需要 3-spin 或 higher-order interaction, 标准 Ising 只 pair interaction。要么用 auxiliary spin 嵌入([Lucas 2014](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full)), 要么 FPGA 改做高阶 tensor。CMOS 路线已有 [Su IEEE SSL 2023](https://ieeexplore.ieee.org/document/10248025) 探索 3-body, 可以借鉴。

### 9.5 神经网络潜力

paper 最后提到, high precision 让 OEIM 可能克服 Ising machine 与复杂神经网络架构的兼容性限制。指 Hopfield network, Boltzmann machine 这类 energy-based model。参考 [Shen Nat. Photonics 2017](https://www.nature.com/articles/nphoton.2017.93) 的 photonic ResNet。

---

## 10. 最直觉的总结

这篇 paper 做了三件大事:

**第一, 把 coupling 从 optical 搬到 microwave + digital**。这个决策牺牲了 optical 的纯模拟并行, 但换来 arbitrary + programmable + 16-bit precision。这是 real-world COP 的硬门槛, 前 optical 路线都过不了。

**第二, 用 OEPO + time-multiplexing 实现大规模**。4096 spin 在室温稳定 1.1h, 比 CIM 稳定(不用锁相), 比 D-Wave 便宜(室温), 比 ROSC scalable(光纤 loop 能塞几千 pulse)。

**第三, 拿真实 traffic 数据验证**。北京西二环, 1200 辆车, 50 km², T-Drive dataset。OEIM 2.7 ms 解出比 SA 6.49 s 更好的方案, 三个数量级 speedup。这不是 toy problem, 是能直接 deploy 的 demo。

核心 insight: **real-world COP 需要 scalable + programmable + stable + high-precision + room-temp 五个条件同时满足, 缺一不可**。前人的 Ising machine 多半卡在 1-2 个上。OEIM 用 hybrid(optical spin + microwave coupling + FPGA digital feedback)这个组合, 第一次打到五个 checkbox 全绿。这是它能解 real-world 问题的根本原因。

Traffic optimization 选得很巧: 问题真实(GPS 数据), 但 spin 数(485)能在当前硬件跑, $J_{ij}$ 动态范围(16 个数量级)恰好验证 16-bit precision 的必要性。三个数量级 speedup 主要来自 circulation time(9.7μs)远短于 SA 在 dense graph 上的 epoch 时间(几十 ms), 加上物理 collective bifurcation 的 escape local minima 能力。

---

## 关键 references

- paper 原文: [https://doi.org/10.1038/s41377-025-02100-9](https://doi.org/10.1038/s41377-025-02100-9)
- 前作 OEPO Ising: [Cen et al. Light Sci. Appl. 11, 333 (2022)](https://www.nature.com/articles/s41377-022-00946-8)
- OEPO 原始: [Hao et al. Light Sci. Appl. 9, 102 (2020)](https://www.nature.com/articles/s41377-020-0326-6)
- Ising formulation of NP: [Lucas Front. Phys. 2, 5 (2014)](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full)
- CIM review: [Mohseni Nat. Rev. Phys. 4, 363 (2022)](https://www.nature.com/articles/s42254-022-00440-y)
- 100k-spin CIM: [Honjo Sci. Adv. 7, eabh0952 (2021)](https://www.science.org/doi/10.1126/sciadv.abh0952)
- Time-multiplexed OPO CIM: [Marandi Nat. Photonics 8, 937 (2014)](https://www.nature.com/articles/nphoton.2014.249)
- 2000-spin CIM: [Inagaki Science 354, 603 (2016)](https://www.science.org/doi/10.1126/science.aah4243)
- 100-spin programmable CIM: [McMahon Science 354, 614 (2016)](https://www.science.org/doi/10.1126/science.aah5178)
- Quantum annealer vs CIM: [Hamerly Sci. Adv. 5, eaau0823 (2019)](https://www.science.org/doi/10.1126/sciadv.aau0823)
- D-Wave 2000Q benchmark: [Willsch QIP 21, 141 (2022)](https://link.springer.com/article/10.1007/s11128-022-03424-x)
- ROSC 1968-node: [Moy Nat. Electron. 5, 310 (2022)](https://www.nature.com/articles/s41928-022-00746-2)
- Coupled oscillator Ising chip: [Cılasun Nat. Electron. 8, 537 (2025)](https://www.nature.com/articles/s41928-025-01369-3)
- Memristor crossbar Ising: [Jiang Nat. Commun. 14, 5927 (2023)](https://www.nature.com/articles/s41467-023-42126-6)
- Simulated bifurcation: [Tatsumura Nat. Electron. 4, 208 (2021)](https://www.nature.com/articles/s41928-021-00560-y)
- SPIM low-rank: [Wang Commun. Phys. 8, 86 (2025)](https://www.nature.com/articles/s42005-025-01467-y)
- 3-body CMOS Ising: [Su IEEE SSL 6, 221 (2023)](https://ieeexplore.ieee.org/document/10248025)
- OIM theory: [Wang & Roychowdhury UCNC 2019](https://link.springer.com/chapter/10.1007/978-3-030-19311-9_13)
- Minimum power dissipation: [Onsager Phys. Rev. 37, 405 (1931)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.37.405)
- Lagrange multiplier physics: [Vadlamani PNAS 117, 26639 (2020)](https://www.pnas.org/doi/10.1073/pnas.2015052117)
- Dynamical phase transition: [Leleu PRE 95, 022118 (2017)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.022118)
- T-Drive dataset: [Yuan SIGSPATIAL 2010](https://dl.acm.org/doi/10.1145/1869790.1869807); [KDD 2011](https://dl.acm.org/doi/10.1145/2020408.2020462)
- User Equilibrium traffic model: [Wardrop 1952](https://www.icevirtuallibrary.com/doi/10.1680/ipeds.1952.11362)
- PySA: [github.com/nasa/PySA](https://github.com/nasa/PySA)
- SG3 MAX-CUT target: [arxiv:2312.10895](https://arxiv.org/abs/2312.10895)
- Pearson correlation: [Altman & Krzywinski Nat. Methods 12, 899 (2015)](https://www.nature.com/articles/nmeth.3697)
- Photonic neural network: [Shen Nat. Photonics 11, 441 (2017)](https://www.nature.com/articles/nphoton.2017.93)

---

# Programmable Optoelectronic Ising Machine (OEIM) — 深度技术解析

## 1. 这篇paper一句话总结

来自中科院半导体所 Ming Li 团队的工作, 把基于 optoelectronic parametric oscillator (OEPO) 的 Ising machine 规模推到 **4096 spins**, 室温长期稳定运行(平均 4250s, 最佳 5.5h), 并第一次用真实 traffic 数据(T-Drive dataset, 北京西二环)做了 real-world COP 的求解, 比 simulated annealing 快三个数量级。

published on *Light: Science & Applications*, online 2026-01-01, DOI: [10.1038/s41377-025-02100-9](https://doi.org/10.1038/s41377-025-02100-9)

---

## 2. Ising machine 的物理直觉 — 为什么要用 oscillator 算 COP

### 2.1 Ising Hamiltonian 与 COP 的映射

Ising model 的 energy(无外场):

$$
H = -\sum_{1 \le i < j < N} J_{ij} \sigma_i \sigma_j \quad (1)
$$

| 符号 | 含义 |
|---|---|
| $\sigma_i \in \{-1, +1\}$ | 第 $i$ 个 Ising spin, 二值变量 |
| $J_{ij}$ | spin $i$ 与 spin $j$ 之间的 coupling, 实数 |
| $N$ | spin 总数 |
| $H$ | Ising energy, 越低越接近 ground state |

直觉: 当 $J_{ij} > 0$ 时, 系统偏好 $\sigma_i = \sigma_j$(ferromagnetic, 平行能量低); 当 $J_{ij} < 0$ 时偏好反平行(antiferromagnetic)。Lucas 2014([Nat. Front. Phys.](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full))证明大量 NP problem(MAX-CUT, graph coloring, SAT, TSP…)都能 cast 成这个形式。求 $H$ 的 minimum 即求原问题的 optimum。

MAX CUT 与 Ising 的等价映射:

$$
\text{cut value} = -\frac{1}{2}\Big(\sum_{1\le i<j<N} J_{ij} + H\Big) \quad (2)
$$

最小化 $H$ ↔ 最大化 cut value, 共享同一个 spin 配置。

### 2.2 为什么 oscillator 是天然的 Ising solver

这是整篇 paper 的核心 intuition。一个 parametric oscillator 在 threshold 附近, 它的 steady state phase 会 **bifurcate 成两个对称稳态**($0$ 或 $\pi$, 相对 LO 的相位)。这两个稳态就是天然的 $\sigma = \pm 1$。多个 oscillator 互相注入 coupling 信号, 系统会依据 **minimum power dissipation principle**(Onsager [Phys. Rev. 37, 405 (1931)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.37.405); Leleu et al. [Phys. Rev. E 95, 022118 (2017)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.022118); Vadlamani et al. [PNAS 117, 26639 (2020)](https://www.pnas.org/doi/10.1073/pnas.2015052117))自然 relax 到能量最低态。

关键物理直觉: **oscillator 是模拟系统, 它的弛豫过程是连续动力学, 不像离散算法那样一个一个 flip spin**。当系统接近 threshold, 注入的 noise 让它能"游走"在能量 landscape 上, 遇到 saddle 时容易被耦合项 push 出去, 类似 simulated annealing 的 thermal fluctuation, 但完全是物理 parallel 的。这一点在 Wang & Roychowdhury 的 OIM paper([UCNC 2019](https://link.springer.com/chapter/10.1007/978-3-030-19311-9_13))和 Mohseni et al. 的 review([Nat. Rev. Phys. 4, 363 (2022)](https://www.nature.com/articles/s42254-022-00440-y))里都有详细论述。

---

## 3. OEIM 的硬件架构 — 把 coupling 从 optical 搬到 microwave

### 3.1 与上一代 OEPO Ising machine 的区别

作者团队 2022 年的 Light: Sci. Appl. paper([Cen et al., 11, 333 (2022)](https://www.nature.com/articles/s41377-022-00946-8))已经在 OEPO 上实现过 Ising machine, 但 coupling 用 **wavelength-division multiplexed optical delay-line network**, 固定 pattern, connectivity 受限, bit resolution 也低。这次的飞跃是: **coupling 整个搬到 microwave domain, 用 FPGA + DAC/ADC 做任意 programmable 反馈**。

为什么 optical domain 的耦合做不好:
- 不同 wavelength 的 delay line 互相串扰难管理
- 高分辨率(16-bit)权重在 optical domain 实现需要昂贵的光调制阵列
- Connectivity 受物理 channel 数量限制

Microwave domain 的优势:
- 频率低(GHz vs THz), 容易用 FPGA 数字处理
- 商用 16-bit ADC/DAC 直接支持高分辨率权重
- FPGA 是 universal 的, 任意 $J_{ij}$ 都能算

### 3.2 实验装置拆解(Fig. 1b)

主回路: 1550nm pump laser → MZM(50 MHz pulse modulation) → 16 km SMF → SOA → PD → EA → BPF → mixer → back to MZM。
- **16 km SMF**: 单模光纤, 提供 84 μs 的 loop delay
- **84 μs × 50 MHz = 4200 个 OEPO pulse**, 每个 pulse 20 ns 间隔
- **Mixer + 20 GHz LO**: 起到 CIM 里 nonlinear crystal 的作用, 实现 degenerate parametric oscillation
- **SOA**: 提供 gain, 控制 threshold
- **MZM**: 把 electrical 信号 modulate 回 optical domain

Feedback loop: Optical splitter 取出脉冲能量 → PD 转 electrical → mixer 解调 binary phase → ADC (16-bit) → **FPGA (存 J 矩阵, 做矩阵-vector乘)** → DAC (16-bit) → 上变频到 10 GHz carrier → microwave coupler 注入 cavity。

### 3.3 Time-multiplexing 规模化直觉

这是整个方案能 scale 到几千 spin 的关键。一个 OPO/OEPO 腔里同时存在数千个 pulse, 每个 pulse 在腔里 round-trip 时都被 feedback loop 读一次、写一次。物理上, **N 个 pulse 在 loop 里就是 N 个独立的 spin, 共享一套硬件**。这正是 CIM 时间多路复用的思路([Marandi et al., Nat. Photonics 8, 937 (2014)](https://www.nature.com/articles/nphoton.2014.249); [Inagaki et al., Science 354, 603 (2016)](https://www.science.org/doi/10.1126/science.aah4243); [Honjo et al., Sci. Adv. 7, eabh0952 (2021)](https://www.science.org/doi/10.1126/sciadv.abh0952))。

对应时间:
- 单个 spin pulse 间隔 = 20 ns
- $N$ 个 spin 的 circulation time = $N \times 20\text{ns}$
- I4096 问题: $4096 \times 20\text{ns} = 81.92\text{μs} \approx 84\text{μs}$ (和 loop delay 匹配)
- Traffic 问题(485 spin): $485 \times 20\text{ns} \approx 9.7\text{μs}$, 所以 279 circulations ≈ 2.71 ms ✓

---

## 4. 实验结果细节

### 4.1 MAX CUT I4096 主结果

| 指标 | OEIM | SA (speed mode) | SA (accuracy, 40ms) | SA (accuracy, 2s) |
|---|---|---|---|---|
| Time-to-target | 1.97 ms (23 circ.) | 19.26 ms | — | — |
| Best cut value | 96,386 (40ms run) | 93,475 | 97,564 | 99,129 |
| Mean cut value | 96,794 | — | 95,731 | 97,943 |

- SG3 (Sahni-Gonzales 3) target = 90,984(参考 [arxiv:2312.10895](https://arxiv.org/abs/2312.10895))
- **OEIM 在 40 ms 内 best 98,236 / mean 96,794, 都超过 SA 同时间窗口**
- 给 SA 跑 2 秒才能在 best 上超过 OEIM, mean 上反超 1149

### 4.2 Graph density 鲁棒性(Fig. 2c)

这是 OEIM 最 striking 的优势:
- 在 1% density 下, SA 反而比 OEIM 快
- 但 density 升到 100%, SA 的 time-to-target 大幅增加且 variance 爆炸, OEIM 几乎不变

直觉解释: SA 在 dense graph 上每次 flip spin 需要更新 $O(N)$ 个邻居状态, 计算量随 density 线性增长; OEIM 是物理并行, 每 circulation 用 FPGA 矩阵乘一次性算完 feedback, **不管 density 多大, 单 circulation 时间固定**。

### 4.3 Stability 实验

停 feedback, 让 4200 spins 随机锁定, 监测 Pearson correlation $r$ between 当前序列与初始序列:
- $r > 0.95$ 视为稳定([Altman & Krzywinski, Nat. Methods 12, 899 (2015)](https://www.nature.com/articles/nmeth.3697))
- 43 次测量: 平均 4250s (1.1h), 最佳 19785s (5.5h)
- 失稳机制: temperature drift → fiber length 变 → pulse timing 漂移

对比: D-Wave quantum annealer 需要 mK 级温度, CIM 用 DOPO 需要锁相 pump laser, OEIM 在室温能稳定数小时, 这是巨大工程优势。

### 4.4 Traffic Optimization — 真实世界验证

**问题设置**:
- 区域: 北京西二环, West Railway Station → Gulou Avenue, ~50 km²
- 数据: T-Drive dataset([Yuan et al. SIGSPATIAL 2010](https://dl.acm.org/doi/10.1145/1869790.1869807); [KDD 2011](https://dl.acm.org/doi/10.1145/2020408.2020462))
- 1200 vehicles → 160 groups(每组 3 条 alternative routes)
- 1500 baseline vehicles(不参与 diversion)
- 模型: User Equilibrium (UE)([Wardrop, Proc. Inst. Civ. Eng. 1952](https://www.icevirtuallibrary.com/doi/10.1680/ipeds.1952.11362))
- Spin 数: $160 \times 3 + 5 = 485$ (5 个 auxiliary spin 把带 external field 的 Ising 转成 zero-field 形式)

**精度要求**: $J_{ij}$ 元素 dynamic range 跨 **16 个数量级**(Fig. 5a, 5b heatmap), 所以必须用 16-bit ADC/DAC + FPGA。这是为什么之前的 optical delay-line 方案完全做不了 real-world 问题。

**结果对比**:
- Reference (IA method, 140 s): Ising energy = −1,054,280
- OEIM: 2.71 ms (279 circ.), 20 次平均 2.5 ms, mean energy −1,052,495
- 自研 SA: mean 6.49 s, mean energy −1,041,229
- PySA (NASA, [github.com/nasa/PySA](https://github.com/nasa/PySA)): 也被 OEIM 击败
- **速度差 3 个数量级 + 解质量更好**

Evolution dynamics(Fig. 5c, 5d):
- 前 30 circulations: 多数 spin 从 noise 长出来, energy 暂时上升
- 30–70 circulations: 大规模 bifurcation, energy 急降
- 70–279 circulations: 最后少数 spin 受 constraint 影响, 缓慢 flip
- 279 circulations: 最后一个 spin flip → 达到 reference

这个分阶段 dynamics 跟 SA 完全不同: SA 是 monotonic descent(带 thermal fluctuation), OEIM 是 **collective bifurcation**, 物理上更接近 simulated bifurcation algorithm([Goto et al., Sci. Adv. 2019](https://www.science.org/doi/10.1126/sciadv.aav2372); [Tatsumura et al., Nat. Electron. 4, 208 (2021)](https://www.nature.com/articles/s41928-021-00560-y))。

---

## 5. 跟其他 Ising machine 横向对比

| 系统 | 原理 | 规模 | 工作条件 | Programmability |
|---|---|---|---|---|
| **OEIM (本工作)** | OEPO + FPGA feedback | 4096 | 室温, 1.1h 稳定 | 任意 J, 16-bit |
| CIM (Honjo 2021, [Sci. Adv. 7](https://www.science.org/doi/10.1126/sciadv.abh0952)) | DOPO + FPGA | 100,000 | 锁相 laser, 需精密 phase control | 任意 J |
| CIM (Takesue 2025, [Sci. Adv. 11](https://www.science.org/doi/10.1126/sciadv.ads7223)) | DOPO | large | 同上 | 任意 J |
| SPIM (Wang 2025, [Commun. Phys. 8](https://www.nature.com/articles/s42005-025-01467-y)) | Spatial light modulator | large | 室温 | 受 low-rank/circulant 约束 |
| D-Wave 2000Q | Superconducting QA | ~5000 | mK 级 | limited topology |
| ROSC (Moy 2022, [Nat. Electron. 5](https://www.nature.com/articles/s41928-022-00746-2)) | Coupled ring oscillator on chip | 1968 | 室温 | on-chip fixed |
| SBM (Tatsumura 2021) | Simulated bifurcation FPGA/GPU | large | 室温 | 任意, 但是 digital |
| Memristor crossbar (Jiang 2023, [Nat. Commun. 14](https://www.nature.com/articles/s41467-023-42126-6)) | Analog memristor | mid | 室温 | limited precision |

OEIM 的 sweet spot: 比 CIM 便宜稳定(室温, 不用锁相), 比 D-Wave 通用(arbitrary coupling), 比 SPIM 精度高(16-bit vs optical SLM 几 bit), 比纯 digital SBM 物理并行度高。代价是 circulation time 长(84μs for 4096), 单次 latency 比 CMOS chip 大。

---

## 6. 几个值得深挖的 limitation 和 follow-up 方向

### 6.1 Pump gain 控制

paper 在 Discussion 里坦承: 没用 refined pump gain schedule。理论表明:
- 强 gain → 快收敛, 但探索空间窄, 容易卡 local minima
- 弱 gain(near threshold) → 慢, 但探索广, 找到更好解的概率高

这跟 CIM 里 [McMahon et al., Science 354, 614 (2016)](https://www.science.org/doi/10.1126/science.aah5178) 的 pump rate schedule 思路一样, 是下一步必须做的。可以想象成 simulated annealing 里的 temperature schedule, gain 类比 temperature。

### 6.2 FPGA 传输 bottleneck

J 矩阵传输时间 ~60s, 完全压住了"实时 traffic management"的 promise。一个 4096×4096 的 dense J 是 134M 元素, 16-bit 是 268 MB, FPGA 用 gigabit 传要分钟级。改进路径:
- 用 HBM-based FPGA
- 增量更新 J 矩阵
- 多 chip 互联(SBM 用的 multi-chip 架构)

### 6.3 规模化路线图

paper 提两条路:
1. 增 repetition rate(50 MHz → 几 GHz), 单 loop 容纳更多 pulse
2. 延长 fiber loop, 但 circulation time 同步增长

第二条有问题: 长 loop 会让单次 computation latency 增加, 抵消速度优势。repetition rate 是主路。但 GHz 级 pulse 会带来 dispersion 问题(16km fiber 累计 dispersion 很大), 需要 dispersion compensation。

### 6.4 多体相互作用

很多 NP 问题(如 SAT, 3-SAT)需要 3-spin 或 higher-order interaction, 标准 Ising Hamiltonian(1)只有 pair interaction。要么用 auxiliary spin 嵌入([Lucas 2014](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full)), 要么 FPGA 改成做高阶 tensor contraction。CMOS 路线已经有 [Su et al. 2023](https://ieeexplore.ieee.org/document/10248025) 探索 3-body Ising, OEIM 完全可以借鉴。

### 6.5 神经网络应用潜力

paper 在 Discussion 里提到, 高 precision 让 OEIM 有可能"克服 Ising machine 与复杂 neural network 架构的兼容性限制"。这指 Hopfield network、Boltzmann machine 之类能量模型训练。可以参考 [Shen et al., Nat. Photonics 11, 441 (2017)](https://www.nature.com/articles/nphoton.2017.93) 的 photonic ResNet 思路。

---

## 7. 给直觉的总结

OEIM 的真正贡献不在"做出又一个 Ising machine", 而在于三个工程决策的组合效应:

1. **把 OEPO 作为 spin carrier** — 比 DOPO 稳定(室温, 长 wavelength 相位漂移小), 比 electronic oscillator 频率高(能 time-multiplex 大量 spin)
2. **把 coupling 全搬到 microwave + digital domain** — 牺牲 optical 的并行度, 换来 16-bit 任意 $J_{ij}$, 这是 real-world COP 的硬门槛
3. **用 FPGA 做 universal feedback** — 任意问题无需改硬件, 跟 SBM 一样 programmable, 但是物理 bifurcation 而不是数字迭代

把这三点合起来, paper 第一次让 Ising machine 同时满足 **scalable + programmable + stable + high-precision**, 这四个条件的交集是 real-world COP 的最小要求。前人的工作多半卡在其中一两个条件上:
- CIM: scalable + programmable, 但 stability 要求苛刻
- D-Wave: scalable + 算得上 stable, 但 topology 限制 + 极低温
- ROSC: stable + room temp, 但 on-chip 难 scale 到几千
- SPIM: scalable, 但 precision 不够做 real-world

Traffic optimization 这个 demo 选得很巧: 问题不是 toy problem(真实 GPS 数据, 50 km², 160 个 group), 但 spin 数(485)又能在当前硬件跑, $J_{ij}$ dynamic range 又恰好能验证 16-bit precision 的必要性。三个数量级的 speedup 主要是 circulation time(9.7μs)远短于 SA 在 dense graph 上的 epoch 时间(几十 ms), 加上物理 bifurcation 的 collective escape local minima 能力。

---

## 关键 reference 链接

- paper 原文: [https://doi.org/10.1038/s41377-025-02100-9](https://doi.org/10.1038/s41377-025-02100-9)
- 前作 OEPO Ising: [Cen et al. Light Sci. Appl. 11, 333 (2022)](https://www.nature.com/articles/s41377-022-00946-8)
- OEPO 原始 paper: [Hao et al. Light Sci. Appl. 9, 102 (2020)](https://www.nature.com/articles/s41377-020-0326-6)
- Ising formulation of NP problems: [Lucas, Front. Phys. 2, 5 (2014)](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full)
- CIM review: [Mohseni et al. Nat. Rev. Phys. 4, 363 (2022)](https://www.nature.com/articles/s42254-022-00440-y)
- 100,000-spin CIM: [Honjo et al. Sci. Adv. 7, eabh0952 (2021)](https://www.science.org/doi/10.1126/sciadv.abh0952)
- Time-multiplexed OPO CIM: [Marandi et al. Nat. Photonics 8, 937 (2014)](https://www.nature.com/articles/nphoton.2014.249)
- 2000-spin CIM: [Inagaki et al. Science 354, 603 (2016)](https://www.science.org/doi/10.1126/science.aah4243)
- 100-spin fully programmable CIM: [McMahon et al. Science 354, 614 (2016)](https://www.science.org/doi/10.1126/science.aah5178)
- Quantum annealer benchmark vs CIM: [Hamerly et al. Sci. Adv. 5, eaau0823 (2019)](https://www.science.org/doi/10.1126/sciadv.aau0823)
- D-Wave 2000Q: [Willsch et al. Quantum Inf. Process. 21, 141 (2022)](https://link.springer.com/article/10.1007/s11128-022-03424-x)
- ROSC 1968-node: [Moy et al. Nat. Electron. 5, 310 (2022)](https://www.nature.com/articles/s41928-022-00746-2)
- Coupled oscillator Ising chip: [Cılasun et al. Nat. Electron. 8, 537 (2025)](https://www.nature.com/articles/s41928-025-01369-3)
- Memristor crossbar Ising: [Jiang et al. Nat. Commun. 14, 5927 (2023)](https://www.nature.com/articles/s41467-023-42126-6)
- Simulated bifurcation: [Goto et al.; Tatsumura et al. Nat. Electron. 4, 208 (2021)](https://www.nature.com/articles/s41928-021-00560-y)
- SPIM low-rank: [Wang et al. Commun. Phys. 8, 86 (2025)](https://www.nature.com/articles/s42005-025-01467-y)
- 3-body CMOS Ising: [Su et al. IEEE SSL 6, 221 (2023)](https://ieeexplore.ieee.org/document/10248025)
- OIM theory: [Wang & Roychowdhury, UCNC 2019](https://link.springer.com/chapter/10.1007/978-3-030-19311-9_13)
- Minimum power dissipation: [Onsager Phys. Rev. 37, 405 (1931)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.37.405)
- Lagrange multiplier physics: [Vadlamani et al. PNAS 117, 26639 (2020)](https://www.pnas.org/doi/10.1073/pnas.2015052117)
- Dynamical phase transition: [Leleu et al. Phys. Rev. E 95, 022118 (2017)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.022118)
- T-Drive dataset: [Yuan et al. SIGSPATIAL 2010](https://dl.acm.org/doi/10.1145/1869790.1869807)
- User Equilibrium traffic model: [Wardrop 1952](https://www.icevirtuallibrary.com/doi/10.1680/ipeds.1952.11362)
- PySA: [github.com/nasa/PySA](https://github.com/nasa/PySA)
- SG3 MAX CUT target: [arxiv:2312.10895](https://arxiv.org/abs/2312.10895)
