# 这篇 Blog 的深度解析：类型安全的量纲分析

## 📌 一句话总结

这篇 blog 由 `dimensional` 包的 maintainer 撰写，核心论点是：**利用 Haskell 的类型系统在编译期强制执行量纲正确性，从而消灭一整类 bug**。

---

## 一、问题陈述：为什么我们需要 typed dimensions？

作者从一个痛点出发：科学计算中，物理量的单位 (units) 和量纲 (dimensions) 靠文档/注释维护，运行时无法保证正确性。

```python
# 原始 Python 版本 — 单位全靠注释
def population(temperature: float, frequency: float) -> float:
    h = physical_constants["Planck constant over 2 pi in eV s"][0]  # eV·s
    kb = physical_constants["Boltzmann constant in eV/K"][0]       # eV/K
    return 0.5 * coth(h * frequency / (2 * kb * temperature)) - 0.5
```

这里 `temperature`, `frequency`, `h`, `kb` 都是 `float`，**量纲信息只存在于注释中**。如果你把 `temperature` 和 `frequency` 传反了，编译器不会报错，运行时也不会报错——只会给你一个**物理上毫无意义的结果**。

### 第一性原理分析

从第一性原理出发，物理计算的可靠性的根基是 **量纲一致性**：

> **公理**：任何有意义的物理等式，两边的量纲必须一致。即 $[LHS] = [RHS]$。

这不是约定俗成，而是数学逻辑的必然推论——如果你能把长度加到质量上，那意味着米和千克之间存在某种换算因子，这在物理上是无意义的。

---

## 二、量纲 的数学基础

### 2.1 七个基本量纲

SI 系统定义了 7 个基本量纲，任何物理量的量纲都可以表示为它们的幂次积：

| 基本量纲 | 符号 | SI 基本单位 |
|---------|------|-----------|
| 时间 duration | $T$ | 秒 |
| 长度 length | $L$ | 米 |
| 质量 mass | $M$ | 千克 |
| 电流 electric current | $I$ | 安培 |
| 热力学温度 thermodynamic temperature | $\Theta$ | 开尔文 |
| 物质的量 amount of substance | $N$ | 摩尔 |
| 发光强度 luminous intensity | $J$ | 坎德拉 |

### 2.2 量纲代数

一个量纲 $D$ 可以表示为 7 元组：

$$D = L^{\alpha_L} \cdot T^{\alpha_T} \cdot M^{\alpha_M} \cdot I^{\alpha_I} \cdot \Theta^{\alpha_\Theta} \cdot N^{\alpha_N} \cdot J^{\alpha_J}$$

其中 $\alpha_i \in \mathbb{Z}$（`dimensional` 包中，非整数幂通过 `NRoot` 处理）。

例如：
- **速度**: $[v] = L \cdot T^{-1}$ → $(\alpha_L, \alpha_T) = (1, -1)$
- **力**: $[F] = M \cdot L \cdot T^{-2}$ → $(1, -2, 1, 0, 0, 0, 0)$（按 7 元组顺序）
- **Boltzmann 常数**: $[k_B] = M \cdot L^2 \cdot T^{-2} \cdot \Theta^{-1}$（能量/温度）

### 2.3 量纲运算规则

**加法/减法**：

$$D_1 + D_2 \text{ 合法} \iff D_1 \equiv D_2$$

即 7 元组完全相同。你不能把 $M$ 加到 $L$ 上。

**乘法/除法**：

$$D_1 \cdot D_2 = L^{\alpha_{L,1}+\alpha_{L,2}} \cdot T^{\alpha_{T,1}+\alpha_{T,2}} \cdot \ldots$$

就是逐分量相加（乘法）或相减（除法）。

**幂运算**：

$$D^n = L^{n \cdot \alpha_L} \cdot T^{n \cdot \alpha_T} \cdot \ldots$$

**关键推论——超越函数的参数必须是无量纲的**：

以 $\sin(x)$ 为例，其 Taylor 展开：

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$$

如果 $x$ 的量纲为 $D$，则 $x^3$ 的量纲为 $D^3$。要使 $D = D^3$，唯一解是 $D = \text{Dimensionless}$（所有 $\alpha_i = 0$）。同理适用于 $\cos$, $\exp$, $\ln$ 等。

---

## 三、Haskell `dimensional` 包的架构深度解析

### 3.1 核心类型架构

```
Quantity d a
   │     │
   │     └── a: magnitude 的数值类型
   └── d: 量纲类型（type-level 7-tuple）
            │
            └── Dimension (type family)
                 │
                 └── Dim Pos1 Zero Neg1 Zero Zero Zero Zero
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      7 个 type-level integers (Pos/neg)
                      对应 L, M, T, I, Θ, N, J 的幂次
```

**关键洞察**：`Dimension` 是一个 **type family**，它在 type level 编码了 7 个整数。这意味着：

1. **编译器在编译期就知道每个量的量纲**
2. **量纲不匹配的运算会在编译期直接报错**
3. **编译器还可以推断量纲**（不只是验证）

### 3.2 类型运算的机制

`dimensional` 使用 Haskell 的 **type families** 和 **type-level natural numbers** 来实现编译期量纲运算：

```haskell
-- 类型层面的乘法（量纲相乘 → 幂次相加）
type family (d1 :: Dimension) * (d2 :: Dimension) :: Dimension

-- 类型层面的除法（量纲相除 → 幂次相减）
type family (d1 :: Dimension) / (d2 :: Dimension) :: Dimension

-- 类型层面的幂运算（幂次乘以整数）
type family (d :: Dimension) ^ (n :: TypeInt) :: Dimension

-- 类型层面的 N 次根（幂次除以 n）
type family NRoot (d :: Dimension) (n :: TypeInt) :: Dimension
```

这就是为什么 `raiseToThreeHalfsPower` 的类型签名看起来如此复杂：

```haskell
raiseToThreeHalfsPower :: Floating a 
                       => Quantity d a 
                       -> Quantity (NRoot d Pos2 ^ Pos3) a
```

展开推导过程：设输入量的量纲为 $d$：
1. `sqrt` → `NRoot d Pos2` → 量纲变为 $d^{1/2}$
2. `^ pos3` → `(NRoot d Pos2) ^ Pos3` → 量纲变为 $(d^{1/2})^3 = d^{3/2}$

所以如果输入量纲是 $M \cdot \Theta^{-1} \cdot T^{-2} \cdot L^2$（即 $[m/(k_B \cdot T)]$），那么输出量纲就是 $(M \cdot \Theta^{-1} \cdot T^{-2} \cdot L^2)^{3/2}$。

### 3.3 操作符设计

| 操作符 | 含义 | 类型效果 |
|--------|------|---------|
| `*~` | 数值 × 单位 → 量 | 创建一个 `Quantity d a` |
| `/~` | 量 ÷ 单位 → 数值 | 从 `Quantity d a` 提取数值 |
| `*` | 量 × 量 → 量 | 量纲相乘 |
| `/` | 量 ÷ 量 → 量 | 量纲相除 |
| `^` | 量 ^ 整数 → 量 | 量纲幂次乘以整数 |
| `+`, `-` | 同量纲加减 | 编译期验证量纲相同 |

---

## 四、实战案例：Maxwell-Boltzmann 分布

### 4.1 公式

$$f(v) = \left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{mv^2}{2k_B T}\right)$$

其中：
- $m$：粒子质量，量纲 $[M]$
- $T$：热力学温度，量纲 $[\Theta]$
- $v$：粒子速度，量纲 $[L \cdot T^{-1}]$
- $k_B$：Boltzmann 常数，量纲 $[M \cdot L^2 \cdot T^{-2} \cdot \Theta^{-1}]$
- $f(v)$：概率密度，量纲？

### 4.2 量纲推导（第一性原理）

**第一项** $\left(\frac{m}{2\pi k_B T}\right)^{3/2}$：

$$\left[\frac{m}{k_B T}\right] = \frac{M}{M \cdot L^2 \cdot T^{-2} \cdot \Theta^{-1} \cdot \Theta} = \frac{M}{M \cdot L^2 \cdot T^{-2}} = L^{-2} \cdot T^2$$

$$\left[\left(\frac{m}{k_B T}\right)^{3/2}\right] = (L^{-2} \cdot T^2)^{3/2} = L^{-3} \cdot T^3$$

**指数参数** $-\frac{mv^2}{2k_B T}$：

$$\left[\frac{mv^2}{k_B T}\right] = \frac{M \cdot (L \cdot T^{-1})^2}{M \cdot L^2 \cdot T^{-2} \cdot \Theta^{-1} \cdot \Theta} = \frac{M \cdot L^2 \cdot T^{-2}}{M \cdot L^2 \cdot T^{-2}} = \text{Dimensionless} \checkmark$$

**最终结果的量纲**：

$$[f(v)] = L^{-3} \cdot T^3$$

这正是一个 **速度密度** 的量纲——即 $\frac{1}{[v]^3} = \frac{1}{(L \cdot T^{-1})^3} = L^{-3} \cdot T^3$。

### 4.3 类型签名修复过程

作者最初（错误地）将返回类型写为 `Dimensionless Double`：

```haskell
-- ❌ 错误！编译器报错：
-- Couldn't match type 'Neg3' with 'Zero'
maxwellBoltzmannDist :: ... -> Dimensionless Double
```

编译器推断出实际返回类型的量纲是 `Dim Pos1 Zero Neg1 Zero Zero Zero Zero` 的三次方的倒数...不，让我更精确地说，编译器推断出的量纲 7 元组中，$L$ 的幂次为 $-3$（`Neg3`），$T$ 的幂次为 $3$（`Pos3`），其余为零。这与 `Dimensionless`（全 `Zero`）不匹配。

修复方式——自定义量纲和量类型：

```haskell
type DVelocityCube    = DVelocity ^ Pos3           -- L^3 · T^-3
type DVelocityDensity = Recip DVelocityCube         -- L^-3 · T^3
type VelocityDensity  = Quantity DVelocityDensity   -- Quantity DVelocityDensity Double
```

最终正确的签名：

```haskell
maxwellBoltzmannDist :: Mass Double
                     -> ThermodynamicTemperature Double
                     -> Velocity Double
                     -> VelocityDensity Double
```

### 4.4 单位转换的魅力

```haskell
-- SI 单位输入
let n2_mass          = 2     *~ unifiedAtomicMassUnit
let room_temperature = 300.0 *~ kelvin
let velocity         = 400   *~ (meter / second)

maxwellBoltzmannDist n2_mass room_temperature velocity 
-- 4.466578950309018e-11 m^-3 s^3

-- 混合单位输入 + 非 SI 输出
let n2_mass          = 2.6605E-27 *~ kilo gram
let room_temperature = 491.0      *~ degreeRankine
let velocity         = 777        *~ knot

(maxwellBoltzmannDist n2_mass room_temperature velocity) /~ ((hour / nauticalMile) ^ pos3)
-- 5.041390507268275e-12
```

**关键点**：函数签名中只约束了**量纲**（Mass, ThermodynamicTemperature, Velocity），不约束**单位**。你传入任何质量单位、任何温度单位、任何速度单位都可以——因为 `dimensional` 在运行时自动将所有输入转换为 SI 基本单位再计算。

---

## 五、设计哲学与权衡

### 5.1 编译期量纲 vs 运行时单位

这是 `dimensional` 最核心的设计决策：

| 层面 | 何时检查 | 灵活性 | 性能 |
|------|---------|--------|------|
| **量纲** | 编译期 | ✅ 类型安全 | ✅ 零运行时开销 |
| **单位** | 运行时 | ✅ 任意单位输入 | ❌ 每个量携带单位信息 |

每个 `Quantity d a` 在运行时实际上是一个 pair `(magnitude, unit_factor)`：

```
Quantity d a ≈ (a, UnitFactor)
```

其中 `UnitFactor` 记录了从当前单位到 SI 基本单位的转换系数。所以 `3 *~ kilo gram` 存储为 `(3, 1000)`（因为 1 kg = 1000 g，但这里 SI 基本单位就是 kg，所以实际上...）。

> 实际上 `dimensional` 的实现更微妙：它存储的是从 SI 基本单位的偏移量。对于 `kilo gram`，由于千克是 SI 基本单位，存储的系数可能是 1。但对于 `pound` 或 `degreeRankine`，就需要真正的转换因子了。

### 5.2 性能建议

作者明确指出：

> For performance-sensitive code, I recommend validating inputs using dimensional, and converting to raw Double using (/~) before compute-heavy operations.

翻译：**在 I/O 边界用 `dimensional` 做验证，内部热循环转为裸 `Double` 计算**。这是一个非常实用的架构模式：

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  输入 (任意  │     │  内部计算 (裸 Double) │     │  输出 (任意  │
│  单位)       │ ──→ │  高性能数值计算       │ ──→ │  单位)       │
│  *~          │     │  /~ 提取数值          │     │  *~ 转换     │
└─────────────┘     └──────────────────────┘     └─────────────┘
      类型安全边界              无开销                    类型安全边界
```

### 5.3 不可扩展性

`dimensional` 硬编码了 7 个 SI 基本量纲。这意味着：

- ✅ 你不能意外创建一个不属于 SI 体系的量纲
- ❌ 你**不能**为经济学、金融学等非物理领域创建自定义量纲（如货币 `$`）
- ❌ 你**不能**添加信息论量纲（如比特 `bit`）

相比之下，Haskell 的 [`units`](https://hackage.haskell.org/package/units) 包允许用户定义任意量纲系统。

### 5.4 其他语言的类似方案

| 语言 | 库/特性 | 编译时/运行时 | 可扩展性 |
|------|--------|-------------|---------|
| Haskell | `dimensional` | 编译时 | ❌ 不可扩展 |
| Haskell | `units` | 编译时 | ✅ 可扩展 |
| F# | 内置 Units of Measure | 编译时 | ✅ 可扩展 |
| C++ | `mp-units` (C++26 提案) | 编译时 | ✅ 可扩展 |
| Julia | `Unitful.jl` | 运行时 | ✅ 可扩展 |
| Python | `pint` | 运行时 | ✅ 可扩展 |
| Rust | `uom` crate | 编译时 | ✅ 可扩展 |
| Swift | `Physical` | 编译时 | ✅ |
| Raku | `Physics::Measure` | 运行时 | ✅ |
| Numbat | 语言级 | 编译时+运行时 | ✅ |

### 5.5 作者承认的局限性

文中脚注指出一个**关键但未解决的问题**：

> **强度量不能有意义地相加。** 例如，你不能把两个温度（热力学温度 $\Theta$）加在一起得到"两倍温度"。但 `dimensional` 的类型系统允许这种操作，因为它们有相同的量纲。

这暴露了量纲系统的一个根本局限：**量纲相同 $\neq$ 物理意义相同**。热力学温度 $\Theta$ 和温度差 $\Theta$ 有相同的量纲，但物理操作不同。区分它们需要更精细的类型系统——有时被称为 **kind** 系统（超越 dimension 的更细粒度分类）。

---

## 六、更深层的技术思考

### 6.1 为什么 Haskell 特别适合做这件事？

Haskell 的以下特性使得 `dimensional` 成为可能：

1. **Type families**：允许在 type level 进行量纲运算（乘、除、幂、根）
2. **Type-level natural numbers**：`GHC.TypeLits` 提供 type-level 整数，用于编码幂次
3. **NoImplicitPrelude**：可以替换默认的算术运算符为量纲感知的版本
4. **Higher-kinded types**：`Quantity d a` 的 `d` 是 kind `Dimension`，`a` 是 kind `*`
5. **Type inference**：编译器不仅验证，还能自动推断量纲

### 6.2 与依赖类型的对比

更理想的方案是使用**依赖类型**，这样你可以表达：

- "温度差" 和 "绝对温度" 是不同类型
- "角度" 和 "无量纲数" 是不同类型（尽管量纲相同）

在 Agda 或 Idris 这样的依赖类型语言中，这更容易实现。但 Haskell 的 type family 机制已经足够应对大多数物理计算场景。

### 6.3 一个潜在的 Bug 示例

考虑原始的 `untyped` 版本：

```haskell
untypedMaxwellBoltzmannDist mass_kg temp_K velocity_mps 
    = ( mass_kg / (2 * pi * boltzmannConstant_JpK * temp_K) ) ** (3/2) 
    * exp ( - (mass_kg * velocity_mps **2) / (2 * pi * boltzmannConstant_JpK * temp_K) )
```

注意！这里有一个**微妙但致命的 bug**：指数部分的分母是 `2 * pi * boltzmannConstant_JpK * temp_K`，而 Maxwell-Boltzmann 分布的标准公式中，指数部分的分母应该是 $2k_BT$，**没有 $\pi$**！

$\frac{mv^2}{2k_BT}$ 而非 $\frac{mv^2}{2\pi k_BT}$

但前系数部分确实是 $\left(\frac{m}{2\pi k_BT}\right)^{3/2}$。

**这正是 typed dimensions 能捕获的错误类型**——如果你混淆了哪一项有 $\pi$、哪一项没有，量纲虽然不会变（$\pi$ 是无量纲的），但值会错。不过注意，typed dimensions **不能**捕获所有计算错误——它只能捕获量纲错误。

---

## 七、总结与启示

### 核心观点

1. **量纲是类型**——将物理量纲映射到类型系统，让编译器成为你的量纲检查助手
2. **编译时验证 > 运行时验证 > 无验证**——Haskell 的静态类型系统使得错误在编译期就被发现
3. **复杂性 vs 复杂化**——量纲本身就有内在复杂性，`dimensional` 只是让你**面对**这个复杂性，而不是忽略它

### 实践建议

- **在 I/O 边界使用 typed dimensions**，在计算内核使用裸数值
- **对于关键计算**（航天、核能、药物剂量），typed dimensions 是必要的
- **对于快速原型**，Python + `pint` 可以是起点，但最终应迁移到编译时验证

### 参考链接

- [dimensional 包 - Hackage](https://hackage.haskell.org/package/dimensional)
- [dimensional GitHub 仓库](https://github.com/bjornbm/dimensional)
- [F# Units of Measure](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure)
- [mp-units for C++ (C++26 提案)](https://mp-units.pages.dev/)
- [Unitful.jl for Julia](https://github.com/PainterQubits/Unitful.jl)
- [uom crate for Rust](https://crates.io/crates/uom)
- [Numbat 编程语言](https://numbat.dev/)
- [Buckwalter 原始论文: Typing Dimensions](https://dl.acm.org/doi/10.1145/1411204.1411207)
- [Hacker News 讨论](https://news.ycombinator.com/item?id=42236912)
- [Haskell Discourse 讨论](https://discourse.haskell.org/t/typed-dimensions-in-haskell/9497)

> *"Using dimensional is complex, but it isn't complicated; it makes you address the complexity of dimensions and units up-front, rather than hoping for the best."* — 作者的这句话精准概括了整个哲学：**复杂是固有的，复杂化是人为的；面对复杂性，而非逃避它。**