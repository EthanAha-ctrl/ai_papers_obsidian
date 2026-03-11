# Generalized Linear Models (GLM) 深度技术解析

## 一、GLM 的核心概念与直觉理解

GLM（Generalized Linear Models，广义线性模型）是统计学和机器学习中的一个**统一框架**，它将线性回归、Logistic回归、Poisson回归等多种模型纳入同一数学体系。让我从直觉上解释其核心思想：

### 1.1 基本直觉

**传统线性回归的局限性**：经典线性回归假设响应变量服从正态分布，且与预测变量呈线性关系。但在实际问题中：
- 计数数据（如交通事故次数）服从 Poisson 分布
- 二分类数据（如成功/失败）服从 Bernoulli 分布
- 等待时间数据可能服从指数分布

**GLM的突破**：GLM通过**链接函数**将非正态分布的数据映射到线性空间，从而扩展了线性回归的适用范围。

## 二、GLM 的数学框架（三要素架构）

GLM 由三个核心组件构成，我将其称为"GLM三位一体"：

```
┌─────────────────────────────────────────────────────────┐
│                    GLM 架构图                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   X₁, X₂, ..., Xp   →   线性预测器 η = β₀ + β₁X₁ + ... + βpXp   │
│      (预测变量)               (系统组件 Systematic Component)    │
│                            ↓                               │
│                     链接函数 g()                          │
│                            ↓                               │
│                     随机组件 Y                             │
│                  (来自指数族分布)                           │
│                                                         │
│   Y ~ 服从指数族分布，E[Y] = μ                           │
│   η = g(μ)  ← 链接函数将均值映射到线性空间               │
├─────────────────────────────────────────────────────────┤
```

### 2.1 系统组件

**线性预测器**：
```
η = β₀ + Σⱼ βⱼXⱼ
```
- η（eta）：线性预测器，取值在 ℝ（实数域）
- β₀：截距项
- βⱼ：第 j 个回归系数
- Xⱼ：第 j 个预测变量
- j = 1, 2, ..., p（p 是预测变量数量）

### 2.2 随机组件

**指数族分布**：响应变量 Y 服从指数族分布，概率密度函数为：
```
f(y|θ, φ) = exp{[yθ - b(θ)]/a(φ) + c(y, φ)}
```
其中：
- θ：自然参数
- φ：离散参数
- a(·), b(·), c(·)：已知函数
- b(θ)：累积量函数，决定了分布类型

**常见指数族分布**：

| 分布类型 | 符号 | 参数 | 典型应用 |
|---------|------|------|---------|
| 正态分布 | N(μ, σ²) | μ: 均值, σ²: 方差 | 连续响应 |
| Poisson分布 | Poisson(μ) | μ: 均值=方差 | 计数数据 |
| Bernoulli分布 | Bernoulli(μ) | μ: 成功概率 | 二分类 |
| Binomial分布 | Binomial(n, μ) | n: 试验次数, μ: 成功概率 | 成功计数 |
| Gamma分布 | Gamma(α, β) | α: 形状参数, β: 率参数 | 正连续值 |

### 2.3 链接函数

**链接函数定义**：
```
η = g(μ)  或  μ = g⁻¹(η)
```
- g(·)：链接函数，单调可导
- μ = E[Y]：响应变量的期望
- η：线性预测器

**正则链接函数**（Canonical Link）是使 θ = η 的链接函数：

| 分布类型 | 正则链接函数 | 公式 | 解释 |
|---------|------------|------|------|
| Normal | Identity | g(μ) = μ | η = μ，直接映射 |
| Poisson | Log | g(μ) = ln(μ) | η = ln(μ)，对数链接 |
| Bernoulli/Binomial | Logit | g(μ) = ln(μ/(1-μ)) | η = ln(odds)，logit链接 |
| Gamma | Inverse | g(μ) = 1/μ | η = 1/μ，倒数链接 |

## 三、各类GLM的详细技术解析

### 3.1 Classical Linear Regression（经典线性回归）

**数学模型**：
```
Y = β₀ + Σⱼ βⱼXⱼ + ε
ε ~ N(0, σ²)
```

**GLM视角**：
- 分布：Y ~ N(μ, σ²)
- 链接函数：g(μ) = μ（恒等链接）
- 线性预测器：η = β₀ + Σⱼ βⱼXⱼ
- 参数估计：**最小二乘法 (OLS)**
```
β̂ = (XᵀX)⁻¹XᵀY
```
  其中 X 是设计矩阵

**特点**：
- 优点：可解释性强，数学基础完善
- 局限：要求 Y 连续、正态分布、等方差
- 适用：房价预测、温度预测等

### 3.2 Poisson Regression（泊松回归）

**数学模型**：
```
Y ~ Poisson(μ)
ln(μ) = η = β₀ + Σⱼ βⱼXⱼ
```
即：
```
μ = exp(β₀ + Σⱼ βⱼXⱼ)
```

**参数估计**：**最大似然估计 (MLE)**
对数似然函数：
```
L(β) = Σᵢ [Yᵢ(β₀ + Σⱼ βⱼXᵢⱼ) - exp(β₀ + Σⱼ βⱼXᵢⱼ) - ln(Yᵢ!)]
```
通过数值优化（如 Newton-Raphson）求解：
```
∂L/∂β = 0
```

**特点**：
- 要求：E[Y] = Var(Y)（均值等于方差）
- 局限：若存在过度离散（Var(Y) > E[Y]），需用负二项回归
- 适用：交通事故次数、网页点击量、电话呼叫次数

**过度离散检验**：
```
Pearson χ² = Σᵢ (Yᵢ - μ̂ᵢ)²/μ̂ᵢ
若 χ²/df >> 1，则存在过度离散
```

### 3.3 Negative Binomial Regression（负二项回归）

**数学模型**：
```
Y ~ NB(r, p)
E[Y] = μ = r(1-p)/p
Var(Y) = μ + μ²/r
```

**GLM视角**：
- 分布：负二项分布
- 链接函数：g(μ) = ln(μ)
- 额外参数：r（离散参数，控制过度离散程度）

**特点**：
- 当 r → ∞ 时，收敛到 Poisson
- Var(Y) = μ + μ²/r > μ，允许过度离散
- 适用：保险索赔次数、医院急诊就诊人数

### 3.4 Logistic Regression（逻辑回归/Logit模型）

**数学模型**：
```
Y ~ Bernoulli(μ)
logit(μ) = ln(μ/(1-μ)) = η = β₀ + Σⱼ βⱼXⱼ
```

**概率表达式**：
```
P(Y=1|X) = μ = 1/(1 + exp[-(β₀ + Σⱼ βⱼXⱼ)])
P(Y=0|X) = 1 - μ = 1/(1 + exp[β₀ + Σⱼ βⱼXⱼ])
```

**Odds解释**：
```
Odds = P(Y=1)/P(Y=0) = exp(β₀ + Σⱼ βⱼXⱼ)
```
- Odds 比率（Odds Ratio）：Xⱼ 每增加 1 个单位，Odds 乘以 exp(βⱼ)

**对数似然函数**：
```
L(β) = Σᵢ [Yᵢ ln(μᵢ) + (1-Yᵢ) ln(1-μᵢ)]
```

**梯度上升更新规则**：
```
β^(new) = β^(old) + α·Xᵀ(Y - μ̂)
```
其中 α 是学习率

**特点**：
- 输出：概率 ∈ [0, 1]
- 可解释：系数可转换为 Odds Ratio
- 适用：客户流失预测、疾病诊断、垃圾邮件分类

### 3.5 其他GLM变体

**Probit回归**：
```
Φ⁻¹(μ) = η  （Φ 是标准正态CDF）
```

**Gamma回归**（生存分析）：
```
Y ~ Gamma(α, β)
g(μ) = 1/μ 或 ln(μ)
适用：生存时间、设备寿命
```

## 四、GLM 参数估计的统一方法

### 4.1 Iteratively Reweighted Least Squares (IRLS)

**算法步骤**：

```
初始化: β^(0), μ^(0), 设定收敛阈值 ε
重复:
    1. 计算线性预测器: η^(t) = Xβ^(t)
    2. 应用链接函数逆: μ^(t) = g⁻¹(η^(t))
    3. 计算权重: W^(t) = diag[wᵢ^(t)], wᵢ = [μᵢ^(t)]²/V(μᵢ^(t))
    4. 计算工作响应: Z^(t) = η^(t) + (Y - μ^(t))·g'(μ^(t))
    5. 更新系数: β^(t+1) = (XᵀW^(t)X)⁻¹XᵀW^(t)Z^(t)
    6. 检查收敛: ||β^(t+1) - β^(t)|| < ε
```

其中 V(μ) 是方差函数，不同分布取不同值：
- 正态分布：V(μ) = σ²
- Poisson：V(μ) = μ
- Bernoulli：V(μ) = μ(1-μ)

### 4.2 信息矩阵与标准误

**Fisher信息矩阵**：
```
I(β) = XᵀWX
其中 W = diag[wᵢ], wᵢ = [g'(μᵢ)]² / V(μᵢ)
```

**系数标准误**：
```
SE(β̂ⱼ) = √[Var(β̂ⱼ)] = √[I(β̂)⁻¹ⱼⱼ]
```

**Wald检验**：
```
z = β̂ⱼ / SE(β̂ⱼ)
p-value = 2·Φ(-|z|)
```

## 五、模型评估与诊断

### 5.1 偏差

**定义**：
```
D = 2·[L(Y, Y) - L(Y, μ̂)]
```
- L(Y, Y)：饱和模型的对数似然
- L(Y, μ̂)：拟合模型的对数似然
- D 越小，模型拟合越好

**各类分布的偏差**：

| 分布 | 偏差公式 |
|------|---------|
| 正态 | D = Σᵢ (Yᵢ - μ̂ᵢ)² |
| Poisson | D = 2·Σᵢ [Yᵢ ln(Yᵢ/μ̂ᵢ) - (Yᵢ - μ̂ᵢ)] |
| Binomial | D = 2·Σᵢ [Yᵢ ln(Yᵢ/μ̂ᵢ) + (nᵢ-Yᵢ) ln((nᵢ-Yᵢ)/(nᵢ-μ̂ᵢ))] |

### 5.2 AIC与BIC

**AIC（Akaike Information Criterion）**：
```
AIC = -2·L(θ̂) + 2·k
```

**BIC（Bayesian Information Criterion）**：
```
BIC = -2·L(θ̂) + k·ln(n)
```
- k：参数数量
- n：样本量
- 越小越好

### 5.3 拟合优度检验

**Pearson残差**：
```
rᵢ = (Yᵢ - μ̂ᵢ) / √V(μ̂ᵢ)
```

**偏差残差**：
```
rᵢ(D) = sign(Yᵢ - μ̂ᵢ) · √dᵢ
其中 dᵢ 是第 i 个观测的偏差贡献
```

**Hosmer-Lemeshow检验**（Logistic回归专用）：
```
χ²_HL = Σₖ (O₁ₖ - E₁ₖ)² / [E₁ₖ(1 - E₁ₖ/nₖ)]
其中 O₁ₖ、E₁ₖ 分别是第 k 组的观测和期望成功数
```

## 六、GLM的扩展与应用

### 6.1 正则化GLM

**Lasso（L1正则化）**：
```
minβ: -L(β) + λ Σⱼ |βⱼ|
```

**Ridge（L2正则化）**：
```
minβ: -L(β) + λ Σⱼ βⱼ²
```

**Elastic Net**：
```
minβ: -L(β) + λ[α Σⱼ |βⱼ| + (1-α) Σⱼ βⱼ²]
```

### 6.2 广义加性模型

**扩展线性预测器**：
```
η = β₀ + Σⱼ fⱼ(Xⱼ)
其中 fⱼ(·) 是平滑函数
```

### 6.3 应用场景汇总

| 场景 | GLM类型 | 链接函数 | 示例 |
|------|---------|---------|------|
| 连续值预测 | Linear | Identity | 房价预测 |
| 计数预测 | Poisson/NB | Log | 交通事故次数 |
| 二分类预测 | Logistic | Logit | 客户流失 |
| 生存分析 | Gamma | Inverse | 设备寿命 |
| 比例预测 | Binomial | Logit | 点击率 |

## 七、GLM的理论基础：指数族分布

### 7.1 指数族的标准形式

```
f(y|θ, φ) = exp{[yθ - b(θ)]/a(φ) + c(y, φ)}
```

**关键性质**：

1. **均值**：
```
E[Y] = μ = b'(θ)
```

2. **方差**：
```
Var(Y) = b''(θ)·a(φ) = V(μ)·a(φ)
```

3. **正则链接**：
```
θ = g(μ) （自然参数等于线性预测器）
```

### 7.2 常见分布的指数族表示

| 分布 | θ | b(θ) | a(φ) | V(μ) |
|------|-----------|----------|------|------|
| N(μ, σ²) | μ | θ²/2 | σ² | 1 |
| Poisson(μ) | ln(μ) | exp(θ) | 1 | μ |
| Bernoulli(μ) | ln(μ/(1-μ)) | ln(1+exp(θ)) | 1 | μ(1-μ) |
| Gamma | -1/μ | -ln(-θ) | 1/α | μ² |

## 八、GLM的重要假设与检验

### 8.1 核心假设

1. **独立性**：观测值相互独立
2. **分布正确性**：Y 服从指定指数族分布
3. **链接函数正确性**：g(μ) 与 X 呈线性关系
4. **无多重共线性**：预测变量之间无严重线性关系

### 8.2 共线性诊断

**VIF（方差膨胀因子）**：
```
VIFⱼ = 1 / (1 - Rⱼ²)
其中 Rⱼ² 是用其他变量预测 Xⱼ 的 R²
```
- VIF > 10：严重共线性
- VIF > 5：需注意

### 8.3 残差分析

**QQ图**：检验正态性（对正态GLM）

**残差vs拟合值图**：检验方差齐性

**Cook距离**：识别强影响点
```
Dᵢ = Σⱼ (β̂ⱼ - β̂ⱼ(-i))² / (p·MSE)
```

## 九、GLM在现代机器学习中的地位

GLM虽然传统，但仍然是：
- **基准模型**：作为复杂模型的比较基准
- **可解释模型**：在需要解释性的场景（医疗、金融）不可替代
- **特征工程的灵感来源**：启发了许多现代算法
- **AutoML的核心**：H2O、AutoSklearn等都包含GLM

## 参考文献

1. **McCullagh, P., & Nelder, J. A. (1989).** *Generalized Linear Models* (2nd ed.). Chapman and Hall. [链接](https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606)

2. **Nelder, J. A., & Wedderburn, R. W. M. (1972).** Generalized linear models. *Journal of the Royal Statistical Society: Series A*, 135(3), 370-384. [链接](https://rss.onlinelibrary.wiley.com/doi/10.2307/2344614)

3. **Hardin, J., & Hilbe, J. (2018).** *Generalized Linear Models and Extensions* (4th ed.). Stata Press. [链接](https://www.stata-press.com/books/generalized-linear-models-and-extensions/)

4. **Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013).** *Applied Logistic Regression* (3rd ed.). Wiley. [链接](https://www.wiley.com/en-us/Applied+Logistic+Regression%2C+3rd+Edition-p-9780470582473)

5. **Wood, S. N. (2017).** *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC. [链接](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R/Wood/p/book/9781498728331)

6. **H2O.ai Official Documentation**. [链接](https://docs.h2o.ai/)

7. **Friedman, J., Hastie, T., & Tibshirani, R. (2010).** Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [链接](https://www.jstatsoft.org/article/view/v033i01)

---

**总结**：GLM通过统一的数学框架，将线性思维扩展到非正态数据。掌握GLM不仅能解决实际问题，还能为理解更复杂的机器学习模型奠定坚实基础。