### 1.1 什么是最优传输（Optimal Transport, OT）？

最优传输是一个将一个概率分布转换为另一个概率分布的数学框架。核心思想是：给定两个数据集X和Y，我们需要找到一个**最优匹配**（matching），即一个双射映射，将X中的每个点xᵢ映射到Y中的某个点yⱼ。

### 1.2 两个核心目标函数

文章定义了两个关键的目标函数来评估匹配的质量：

**线性目标函数（E₁）：**
```
E₁(σ) = (1/n) Σᵢ₌₁ⁿ c(xᵢ, y_{σᵢ})
```
其中：
- **σ** 是一个排列（permutation），表示点之间的映射关系
- **c: X × Y → ℝ** 是**ground cost function**，衡量点对之间的成本
- **xᵢ** 是源空间X中的第i个点
- **y_{σᵢ}** 是目标空间Y中与xᵢ匹配的点

**二次目标函数（E₂）- Gromov-Wasserstein形式：**
```
E₂(σ) = (1/n²) Σᵢ,ᵢ'₌₁ⁿ Δ(c_X(xᵢ, x_{i'}), c_Y(y_{σᵢ}, y_{σ_{i'}}))
```
其中：
- **c_X: X × X → ℝ** 和 **c_Y: Y × Y → ℝ** 分别是X和Y空间内的cost函数
- **Δ: ℝ × ℝ → ℝ** 是**discrepancy function**（差异函数），用于比较两个数值之间的差异
- 这个目标函数鼓励找到**近似等距**（near-isometry）的映射

### 1.3 放松约束：从排列到双随机矩阵

由于直接在排列空间上优化是NP-hard问题，文章介绍了**Birkhoff polytope放松**：

将排列σ表示为n×n的**二元矩阵**，每行每列只有一个1。放松后变为**双随机矩阵**（bistochastic matrices）：

```
U(a,b) = {P ∈ ℝⁿˣᵐ⁺ | P1ₘ = a, Pᵀ1ₙ = b}
```

其中：
- **P** 是**transportation matrix**（传输矩阵）
- **a ∈ ℝⁿ** 和 **b ∈ ℝᵐ** 是边缘概率向量（marginal probability vectors）
- **1ₘ** 和 **1ₙ** 是全1向量
- 条件 aᵀ1ₙ = bᵀ1ₘ = 1 保证概率归一化

## 二、核心问题形式化

### 2.1 线性传输问题（Kantorovich问题）

**离散形式：**
```
L_c(μ, ν) = min_{P∈U(a,b)} Σᵢ,ⱼ c(xᵢ, yⱼ) P_{ij}
```

**连续形式：**
```
L_c(μ, ν) = min_{π∈Π(μ,ν)} ∫_{X×Y} c(x,y) dπ(x,y)
```

其中：
```
Π(μ,ν) = {π ∈ P(X×Y) | ∀A⊂X,B⊂Y: π(A×Y) = μ(A), π(X×B) = ν(B)}
```
- **π** 是**coupling**（耦合），即联合分布
- **μ** 和 **ν** 是两个概率测度

### 2.2 二次传输问题（Gromov-Wasserstein）

**离散形式：**
```
Q_{c_X,c_Y}(μ, ν) = min_{P∈U(a,b)} Σ_{i,i',j,j'} Δ(c_X(xᵢ, x_{i'}), c_Y(yⱼ, y_{j'})) P_{ij} P_{i'j'}
```

**连续形式：**
```
Q_{c_X,c_Y}(μ, ν) = min_{π∈Π(μ,ν)} ∫_{X²} ∫_{Y²} Δ(c_X(x, x'), c_Y(y, y')) dπ(x,y) dπ(x',y')
```

## 三、主要挑战

文章指出了解决OT问题的三大核心挑战：

### 3.1 可扩展性挑战

- 网络流求解器的最坏情况复杂度是**超立方级**：**O(nm(n+m)log(n+m))**
- 二次问题（问题2）是**NP-hard**的（特别是当Δ(u,v) = (u-v)²时）
- 通常需要迭代线性化来求解，每次都要付出三次方代价

### 3.2 维度灾难

在高维空间中：
- 使用经验分布 ̂μₙ 和 ̂νₙ 来逼近连续分布
- 计算L_c(̂μₙ, ̂νₙ) 会**过度拟合**样本，浪费计算资源
- Fournier和Guillin (2015) 的理论结果表明：精确计算会受到维度灾难的严重影响

### 3.3 最优解的可微性问题

- 虽然L_c和Q的值可以使用**envelope定理**（Danskin定理）来微分
- 但最优传输矩阵**P*** 的变化**不够平滑**
- 例如：雅可比矩阵 **J_{xᵢ}P*** 几乎处处为0（因为微小的改变通常不会影响最优匹配）

## 四、OTT-JAX的解决方案

### 4.1 熵正则化（Entropic Regularization）

**Cuturi (2013)**提出的熵正则化方法能有效解决上述三个问题：

正则化目标函数：
```
L_{c,ε}(μ, ν) = min_{P∈U(a,b)} Σ_{i,j} c(xᵢ, yⱼ) P_{ij} + ε H(P)
```

其中：
- **ε > 0** 是正则化参数
- **H(P) = Σ_{i,j} P_{ij} log P_{ij}** 是熵项

**优势：**
1. **统计性质改善**（Genevay et al., 2019; Mena & Niles-Weed, 2019）
2. **可微性提升**：可以通过**unrolling**（Adams & Zemel, 2011; Bonneel et al., 2016）或**implicit differentiation**（Luise et al., 2018; Cuturi et al., 2020）实现

### 4.2 低秩Sinkhorn方法（Low-Rank Sinkhorn）

**Scetbon et al. (2021a)**提出的低秩约束方法：

```
P* ≈ Q Rᵀ，其中 Q ∈ ℝⁿˣʳ，R ∈ ℝᵐˣʳ
```

其中 **r ≪ min(n,m)** 是低秩约束的秩。

**优势：**
- 大幅降低计算复杂度
- 也可用于二次问题（Scetbon et al., 2021b）

### 4.3 JAX框架的优势

OTT-JAX充分利用了JAX的特性：

| JAX特性 | 在OTT中的应用 |
|--------|--------------|
| **自动微分** | 反向模式微分，支持自定义梯度 |
| **向量化** | 批量处理多个OT问题 |
| **JIT编译** | just-in-time编译，加速执行 |
| **加速器支持** | 支持GPU/TPU加速 |

## 五、实现架构

### 5.1 核心模块结构

```
OTT-JAX
├── geometry/       # 几何结构模块
│   └── Geometry    # 封装cost矩阵的数学属性
├── core/           # 核心算法模块
│   ├── problems.py / quadproblems.py    # 问题定义
│   ├── sinkhorn.py / sinkhornlr.py     # Sinkhorn求解器
│   ├── gromovwasserstein.py            # Gromov-Wasserstein求解器
│   ├── discretebarycenter.py           # 重心计算
│   └── icnn.py                         # Input-Convex Neural Networks
└── tools/          # 工具模块
    ├── softsort.py  # Soft-sorting
    └── gaussian_mixtures/  # 高斯混合模型OT
```

### 5.2 Geometry类的设计

Geometry类封装了cost矩阵的数学属性，**避免显式存储**：

**例子1：点云几何**
- 当cost是**欧氏距离平方**时：cost矩阵C的秩最多为 **d+2**
- 不需要存储所有n×m的距离

**例子2：网格几何**
- 当支持是d个单变量离散化的笛卡尔积时
- 理论上cost矩阵C的大小为 **nᵈ × nᵈ**
- 但应用C或其kernel e^(-C/ε)可以在 **O(d n^{d+1})** 操作中完成

### 5.3 典型使用流程

```python
# 1. 创建几何对象
geom = PointCloud(x, y, cost=...)

# 2. 定义OT问题
prob = LinearProblem(geom, a, b)

# 3. 求解
out = Sinkhorn()(prob)           # 标准Sinkhorn
# 或
out = LRSinkhorn(rank=r)(prob)   # 低秩Sinkhorn

# 4. 获取结果
transport_matrix = out.matrix
```

## 六、高级应用

### 6.1 Wasserstein重心（Barycenters）

计算多个分布的**平均分布**：

```
μ̄ = argmin_{μ} Σ_{k=1}^K w_k L_c(μ, μ_k)
```

其中：
- **μ_k** 是第k个输入分布
- **w_k** 是权重（Σw_k = 1）

**应用场景：**
- 图像平均
- 人群数据分析（如Voxel-Based Morphometry）

### 6.2 Gromov-Wasserstein

匹配不同空间中的结构：

- **左图**：二维螺旋
- **右图**：三维瑞士卷
- 相同颜色的点相互匹配

### 6.3 Soft-sorting

可微的排序和分位数：

- 使用OT来解决排序问题
- 参数**ε**控制平滑程度（可微性）
- 应用示例：soft-sort array [1.0, 5.0, 4.0, 8.0, 12.0]

### 6.4 Input-Convex Neural Networks (ICNN)

- 用于估计**凸映射**
- 参考：Amos et al., 2017

## 七、参考链接

1. **OTT-JAX GitHub仓库**：https://github.com/ott-jax/ott
2. **Cuturi (2013) - Sinkhorn distances**：https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html
3. **Scetbon et al. (2021a) - Low-rank Sinkhorn**：https://proceedings.mlr.press/v139/scetbon21a.html
4. **Mémoli (2011) - Gromov-Wasserstein**：https://link.springer.com/article/10.1007/s10208-011-9093-5
5. **Peyré & Cuturi - Computational Optimal Transport Book**：https://optimaltransport.github.io/book/

## 八、总结

OTT-JAX的核心贡献在于：

1. **统一框架**：在一个工具箱中集成多种OT算法
2. **可微分**：通过JAX实现端到端的梯度计算
3. **高效**：利用熵正则化和低秩近似处理大规模问题
4. **灵活**：支持点云、直方图、网格等多种数据结构

这使得研究者可以更轻松地将OT技术应用于各种机器学习任务中。