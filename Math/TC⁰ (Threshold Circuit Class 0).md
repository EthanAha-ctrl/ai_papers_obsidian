




# TC⁰ (Threshold Circuit Class 0) 深度解析

## 一、核心定义：从第一性原理出发

### 1.1 什么是 TC⁰？

**TC⁰** 全称 **Threshold Circuit class 0**，是计算复杂性理论和电路复杂性理论中的一个核心复杂度类。它是 TC 类层级中的第一个（也是最基础的）类。

**一句话概括**：TC⁰ 是由 **常数深度** 和 **多项式规模** 的、仅含 AND、OR、NOT 和 MAJ（Majority）门的 **无界扇入** 布尔电路族所判定的所有语言的集合。

### 1.2 形式化定义

一个 **布尔电路族** 是一列布尔电路 $C_1, C_2, C_3, \ldots$，由布尔函数的前馈网络组成。一个二元语言 $L \in \{0,1\}^*$ 属于 TC⁰ 类，当且仅当存在一个布尔电路族 $C_1, C_2, C_3, \ldots$，满足：

| 条件 | 形式化表述 | 含义 |
|------|-----------|------|
| **规模约束** | 存在多项式函数 $p(n)$ | 电路的门数量不超过 $p(n)$ |
| **深度约束** | 存在常数 $d$ | 电路的层数不超过 $d$（不随 $n$ 增长）|
| **门类型** | AND、OR、NOT、MAJ | 无界扇入，即每个门的输入数可以任意多 |
| **正确性** | $\forall x \in \{0,1\}^n: x \in L \iff C_n(x) = 1$ | 电路正确判定语言 |

> **关键直觉**：深度是**常数**（与输入长度无关），而规模是**多项式**的。这意味着电路虽然"浅"，但每层可以"宽"——每个门可以接收任意多的输入。这正是 TC⁰ 的精髓：**通过宽度而非深度来换取计算能力**。

---

## 二、Threshold Gate（阈值门）的等价刻画

### 2.1 从 MAJ 门到 Threshold Gate

TC⁰ 的定义中，MAJ 门和 Threshold Gate 是等价的。这是一个深刻的事实：

**MAJ 门**：一个 $k$-输入的 MAJ 门输出 1 当且仅当超过一半的输入为 1。

**Threshold Gate（阈值门/人工神经元）**：一个 $k$-输入的阈值门由权重列表 $w_1, \ldots, w_k$ 和一个阈值 $\theta$ 定义。对二元输入 $x_1, \ldots, x_k$，输出为：

$$\text{output} = \begin{cases} +1 & \text{若 } \sum_{i=1}^{k} w_i x_i > \theta \\ -1 & \text{若 } \sum_{i=1}^{k} w_i x_i \leq \theta \end{cases}$$

其中：
- $w_i$：第 $i$ 个输入的**权重**（整数），决定该输入的重要性
- $x_i$：第 $i$ 个二元输入（0 或 1）
- $\theta$：**阈值**（整数），决定输出的切换点
- $k$：门的扇入数

### 2.2 等价转换定理

**定理**：MAJ 门和多项式有界权重的阈值门在以下意义上等价——

> 任何多项式规模、深度 $d$ 的阈值电路（权重和阈值为整数且多项式有界），可以被多项式规模、深度 $d+1$ 的 MAJ 电路一致地模拟。

具体而言：

1. **MAJ 门是阈值门的特例**：MAJ 门即所有权重为 1、阈值为 $\lceil k/2 \rceil$ 的阈值门。

2. **权重可被"复制输入"模拟**：权重 $w_i$ 可以通过将第 $i$ 个输入复制 $|w_i|$ 次来模拟（当权重为多项式有界时，复制次数仍是多项式的）。

3. **阈值可被常量输入模拟**：阈值 $\theta$ 可以通过添加 $\theta$ 个常量 True 输入（或 $|\theta|$ 个常量 False 输入）来模拟。

4. **显式算法**：给定一个 $n$-输入的任意整数权重和阈值的阈值门，可以构造一个深度为 2 的、使用 $\text{poly}(n)$ 个 AND、OR、NOT 和 MAJ 门的电路来模拟。

### 2.3 等价的算术刻画

在算术电路复杂性理论中，TC⁰ 有一个优美的等价刻画：

$$\text{TC}^0 = \left\{ \text{sign} \circ f_n \;\middle|\; f_n: \{0,1\}^n \to \mathbb{Z} \text{ 由常数深度、多项式规模、无界扇入的} + \text{ 和 } \times \text{ 门算术电路计算，常量来自 } \{-1, 0, +1\} \right\}$$

其中：
- $\text{sign}(\cdot)$：符号函数，将整数映射到 $\{-1, +1\}$
- $f_n$：从 $\{0,1\}^n$ 到 $\mathbb{Z}$ 的函数
- 算术电路使用 $+$（加法）和 $\times$（乘法）门
- 常量只允许 $\{-1, 0, +1\}$

> **直觉**：TC⁰ 可以看作是"先做多项式级规模的整数算术运算（常数深度），再取符号"。这把电路模型和算术模型联系了起来。

---

## 三、TC⁰ 能计算什么？——重要问题

TC⁰ 包含许多**看起来需要大量计算**的问题，这令人惊讶，因为常数深度电路似乎"太浅"了：

| 问题 | 描述 | 参考 |
|------|------|------|
| **排序** | 将 $n$ 个 $n$-位整数排序 | — |
| **乘法** | 两个 $n$-位整数相乘 | — |
| **整数除法** | 给定 $n$-位整数 $x, y$，计算 $\lfloor x/y \rfloor$ | [1] |
| **Dyck 语言识别** | 多种括号类型的匹配识别 | — |
| **幂运算** | 给定 $n$-位整数 $x$ 和 $O(\ln n)$-位整数 $k$，计算 $x^k$ | [10] |
| **迭代乘法** | 将 $n$ 个 $n$-位整数相乘 | [10] |

> **惊人之处**：整数乘法和除法居然可以在**常数深度**内完成！这依赖于无界扇入的阈值门可以"并行地"处理大量输入。

---

## 四、复杂度类之间的关系

### 4.1 已知的包含关系

$$\text{AC}^0 \subsetneq \text{AC}^0[p] \subsetneq \text{TC}^0 \subseteq \text{NC}^1$$

| 复杂度类 | 定义 | 与 TC⁰ 的关系 |
|----------|------|--------------|
| **AC⁰** | 常数深度、多项式规模、无界扇入 AND/OR/NOT 电路 | $\text{AC}^0 \subsetneq \text{TC}^0$（严格包含） |
| **AC⁰[p]** | AC⁰ 加上模 $p$ 门 | $\text{AC}^0[p] \subsetneq \text{TC}^0$（严格包含） |
| **NC¹** | $O(\log n)$ 深度、多项式规模、有界扇入（扇入 2）电路 | $\text{TC}^0 \subseteq \text{NC}^1$（是否严格？未知！） |

### 4.2 核心开放问题

**$\text{TC}^0 \stackrel{?}{=} \text{NC}^1$**

这是电路复杂性理论中**最重要的开放问题之一**。

更令人震惊的是，**甚至连 $\text{TC}^0 \subsetneq \text{P/poly}$ 是否成立都是开放的**！原因是：

- 根据 **Natural Proof** 理论（Razborov & Rudich, 1997），在假设 TC⁰ 中存在密码学安全的伪随机数生成器的条件下，不存在自然证明可以分离 TC⁰ 和 P/poly。
- 这类伪随机数生成器已经在**分解 Blum 整数是困难的**这一广泛认可的假设下被显式构造出来。

> **第一性原理思考**：为什么证明 $\text{TC}^0 \neq \text{NC}^1$ 这么难？因为 TC⁰ 电路是非一致的——对每个输入长度 $n$，可以有一个完全不同的电路 $C_n$。不存在"统一的算法"来生成这些电路，除非穷举所有 $2^{\text{poly}(n)}$ 个可能的电路再逐一验证。

### 4.3 已知的部分结果

**如果 $\text{TC}^0 = \text{NC}^1$，则**：对任意 $\epsilon > 0$，存在门数为 $O(n^{1+\epsilon})$ 的 TC⁰ 电路族解决 Boolean Formula Evaluation 问题。因此，**任何超线性下界**就足以证明 $\text{TC}^0 \neq \text{NC}^1$。

**NEXP 与 TC⁰ 的关系**：$\text{NEXP} \nsubseteq \text{ACC}^0$ 仅在 2011 年被证明（Williams, 2011）。但 $\text{NEXP} \subseteq \text{TC}^0$ 是否成立仍是开放的。注意：非一致的 TC⁰ 和 ACC⁰ 可以计算非 Turing 可计算的函数，所以 $\text{TC}^0 \nsubseteq \text{NEXP}$ 实际上成立！

---

## 五、TC⁰ 的精细结构——深度层级

### 5.1 深度层级

TC⁰ 可以进一步细分为按深度划分的层级：

$$\text{TC}_1^0 \subset \text{TC}_2^0 \subset \cdots \subset \text{TC}^0 = \bigcup_{d=1}^{\infty} \text{TC}_d^0$$

其中 $\text{TC}_d^0$ 表示深度至多为 $d$ 的阈值电路族所判定的语言类。

### 5.2 MAJ 门 vs 阈值门的深度差异

记号：
- $\widehat{\text{LT}}_d := \text{TC}_d^0$（多项式有界权重和阈值，深度 $d$）
- $\text{LT}_d$：权重和阈值**无界**（"大权重阈值电路"，深度 $d$）

关键包含关系：

$$\text{TC}_d^0 \subset \text{LT}_d \subset \text{TC}_{d+1}^0$$

且已证明 $\text{TC}_2^0 \subsetneq \text{LT}_2$（严格包含）。

### 5.3 IP 函数的分离结果

**Boolean 内积函数** $\text{IP}_n$ 定义为：

$$\text{IP}_n(x_1, \ldots, x_n, x_1', \ldots, x_n') = \bigoplus_{i=1}^{n} \text{AND}(x_i, x_i')$$

即两个 $n$-位向量的内积模 2。

**分离定理**：
- $\text{IP}_n$ 可以用 3 层 $O(n)$ 个门的 MAJ 电路计算
- 但 $\text{IP}_n$ **不能**用 2 层多项式个门的阈值电路计算

更精确的下界：

| 条件 | 下界 |
|------|------|
| 底层门权重为多项式 | 需要 $\geq 2^{(1/2 - \epsilon)n}$ 个门 |
| 顶层门权重 $\leq 2^{o(n^{1/3})}$ | 顶层扇入 $\geq 2^{\Omega(n^{1/3})}$ |
| 底层门权重 $\leq 2^{n/3}$ | 顶层扇入 $\geq 2^{\Omega(n)}$ |

### 5.4 权重上界

对于 $n$ 变量的阈值函数，令 $W(n)$ 为使得所有实阈值函数都能用整数权重 $|w_i| \leq W$ 实现的最小 $W$。已知：

$$\frac{1}{2}n\log n - 2n + o(n) \leq \log_2 W(n) \leq \frac{1}{2}n\log n - n + o(n)$$

其中 $n$ 为输入变量数。这意味着**权重可以指数级增长**，但在 TC⁰ 的定义中我们只要求权重为多项式有界。

---

## 六、Uniform TC⁰

### 6.1 DLOGTIME-uniform TC⁰ = FOM

**DLOGTIME-uniform TC⁰** 也被称为 **FOM**（First-Order logic with Majority quantifiers），因为它等价于带 Majority 量词的一阶逻辑。

**Majority 量词** $M$：给定一个恰好含一个自由变量 $x$ 的公式 $\phi(x)$，量词表达式 $Mx\,\phi(x)$ 为真当且仅当 $\phi(x_i)$ 对超过一半的 $i \in \{1, \ldots, n\}$ 为真。

### 6.2 Functional 版本

Uniform TC⁰ 的函数版本等价于以下函数集的投影闭合：

$$\{n+m,\; n \dot{-} m,\; n \wedge m,\; \lfloor n/m \rfloor,\; 2^{\lfloor \log_2 n \rfloor^2}\}$$

或等价地：

$$\{n+m,\; n \dot{-} m,\; n \wedge m,\; \lfloor n/m \rfloor,\; n^{\lfloor \log_2 m \rfloor}\}$$

其中：
- $n \dot{-} m = \max(0, n-m)$：截断减法
- $n \wedge m$：按位 AND

### 6.3 已知分离

$$\text{uniform TC}^0 \subsetneq \text{PP}$$

且 0-1 矩阵的 permanent 不在 uniform TC⁰ 中。

---

## 七、概率版本：RTC⁰

### 7.1 定义

类似 P 与 BPP 的关系，TC⁰ 有其概率版本 **RTC⁰**（Randomized TC⁰）。

电路 $C_n$ 接受两类输入：
- **确定性输入** $x_1, \ldots, x_n$
- **随机输入** $y_1, \ldots, y_m$，其中 $m = \text{poly}(n)$，随机输入均匀采样自 $\{0,1\}^m$

判定条件：
- 若 $x \in L$：$\Pr[C_n(x,y) = +1] \geq \frac{1}{2} + \frac{1}{\text{poly}(n)}$
- 若 $x \notin L$：$\Pr[C_n(x,y) = +1] \leq \frac{1}{2} - \frac{1}{\text{poly}(n)}$

### 7.2 BPTC⁰ 版本

有时也称为 **BPTC⁰**，与 BPP 类比：
- 若 $x \in L$：$\Pr[C_n(x,y) = +1] \geq \frac{2}{3}$
- 若 $x \notin L$：$\Pr[C_n(x,y) = +1] \leq \frac{1}{3}$

**通过多次采样取 MAJ**，任何 $d$-层 RTC⁰ 电路可以转化为 $(d+1)$-层 BPTC⁰ 电路。

### 7.3 层级关系

$$\text{TC}_1^0 \subset \text{RTC}_1^0 \subset \text{TC}_2^0 \subset \text{RTC}_2^0 \subset \cdots \subset \text{TC}^0 = \text{RTC}^0$$

**已知分离**：
- $\text{RTC}_1^0 \subsetneq \text{TC}_2^0$：PARITY 函数在 $\text{TC}_2^0$ 中但不在 $\text{RTC}_1^0$ 中
- $\text{TC}_2^0 \subsetneq \text{RTC}_2^0$：IP 函数在 $\text{RTC}_2^0$ 中但不在 $\text{TC}_2^0$ 中

且有 $\text{RTC}^0/\text{poly} = \text{TC}^0/\text{poly}$。

---

## 八、与神经网络的联系

TC⁰ 最初就是为了建模**有界深度神经网络**的计算复杂性而提出的（参考 [2]）。

### 8.1 Sigmoid 激活不增加计算能力

**定理**：允许 sigmoid 激活函数 $\sigma$ 不增加 TC⁰ 的计算能力：

$$\text{TC}_d^0 = \text{TC}_d^0(\sigma), \quad \forall d \geq 1$$

（假设权重为多项式有界）

> **直觉**：一个 sigmoid 神经元可以被阈值门加上少量额外深度来模拟。这是因为 sigmoid 函数是单调的，其"决策边界"本质上是一个超平面——与阈值门的线性分割一致。

### 8.2 Boltzmann 机与 RTC⁰

前馈 Boltzmann 机被建模为带**有界不可靠阈值单元**的 RTC⁰ 电路——每个阈值单元可以以有界概率 $\epsilon < 1/2$ 独立随机地输出错误结果。

---

## 九、开放问题总结

| 开放问题 | 状态 |
|----------|------|
| $\text{TC}^0 \stackrel{?}{=} \text{NC}^1$ | 核心开放问题 |
| $\text{TC}^0 \subsetneq \text{P/poly}$? | 开放（自然证明障碍） |
| $\text{NEXP} \subseteq \text{TC}^0$? | 开放 |
| $\text{TC}^0 = \text{TC}_3^0$?（层级是否坍塌到深度 3？）| 开放 |
| $\text{LT}_2$ 是否有指数下界？| 开放 |
| 深度层级有多少层？| 开放 |

---

## 十、直观总结图

```
AC⁰ ⊊ AC⁰[p] ⊊ TC⁰ ⊆ NC¹
                          ↑
                    核心开放问题:
                    TC⁰ = NC¹ ?

TC⁰ 的深度层级:
TC₁⁰ ⊂ TC₂⁰ ⊂ TC₃⁰ ⊂ ... ⊂ TC⁰ = ⋃ TC_d⁰

概率扩展:
TC₁⁰ ⊂ RTC₁⁰ ⊂ TC₂⁰ ⊂ RTC₂⁰ ⊂ ... ⊂ TC⁰ = RTC⁰

大权重扩展:
TC_d⁰ ⊂ LT_d ⊂ TC_{d+1}⁰
TC₂⁰ ⊊ LT₂  (已知严格)
```

---

## 参考

- [1] Hesse, Allender, Barrington — *Uniform constant-depth threshold circuits for division and iterated multiplication*
- [2] Hajnal, Maass, Pudlák, Szegedy, Turán — *Threshold circuits of bounded depth*
- [3] Agrawal, Allender, Rudich — *Reductions in circuit complexity*
- [4] Arora, Barak — *Computational Complexity: A Modern Approach*
- [5] Naor, Reingold — *Number-theoretic constructions of efficient pseudo-random functions*
- [6] Impagliazzo, Wigderson — *P=BPP if E requires exponential circuits*
- [7] Williams — *Non-uniform ACC circuit lower bounds* (2011)
- [8] Barrington, Immerman, Straubing — *Bounded-depth circuits and first-order logic with majority quantifiers*
- [9] Barrington, Immerman, Straubing — 同上
- [10] Hesse, Allender, Barrington — 同上
- Wikipedia: [TC0](https://en.wikipedia.org/wiki/TC0)
- Arora & Barak: *Computational Complexity: A Modern Approach*, Cambridge University Press
- Vollmer: *Introduction to Circuit Complexity*, Springer