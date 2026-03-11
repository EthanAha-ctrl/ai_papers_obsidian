
- 句子：**“i love programming”**（3 个 token）
- 假设：**d_model = 4**
- 假设：这是 **Self-Attention** 中 **Query / Key** 的 Position Encoding

# 1. 为什么需要 RoPE


传统 **Absolute Position Encoding**（如 Sinusoidal PE）的问题在于：


- 它是**加法注入**：

$$
X_{token} + PE_{position}
$$
- Attention 计算的是 **dot-product**，位置关系并不会自然体现在

$$
Q_i \cdot K_j
$$

而 **RoPE 的核心目标**是：



让 **relative position** 直接体现在


$$
Q_i \cdot K_j
$$




# 2. RoPE 的核心思想（一句话）



**把 Position 信息编码为对 Query / Key 向量的旋转（Rotation）**



即：


- 不加 Position vector
- 而是对 embedding 的 **二维子空间做旋转**


# 3. d_model = 4 的结构划分


RoPE 的前提：


- **d_model 必须是偶数**
- 每 **2 个维度** 组成一个 **2D rotation plane**

因此：


| Index | 维度 |
| ---- | ---- |
| 0 | x₀ |
| 1 | x₁ |
| 2 | x₂ |
| 3 | x₃ |


拆成两个子空间：


- Plane 1：`(x₀, x₁)`
- Plane 2：`(x₂, x₃)`


# 4. Rotation 的数学定义


对 position = *p*，在第 *k* 个 plane 上：


### 4.1 角度定义


RoPE 使用 frequency 递减的角度：


$$
\theta_{p,k} = p \cdot \omega_k
$$


其中：


$$
\omega_k = 10000^{-2k / d_{model}}
$$


对于 **d_model = 4**：


- k = 0 → plane 1
- k = 1 → plane 2

所以：


$$
\omega_0 = 10000^{0} = 1
$$


$$
\omega_1 = 10000^{-1/2} = 0.01
$$



### 4.2 Rotation Matrix


对任意 2D vector：


$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$



# 5. 示例句子与 position 编号


Sentence：


```css
i        love        programming

```

Position index（从 0 开始）：


| Token | Position p |
| ---- | ---- |
| i | 0 |
| love | 1 |
| programming | 2 |



# 6. 假设原始 embedding（示例）


设 token embedding（来自 Embedding Layer）为：


```ini
i            = [1.0, 0.0, 1.0, 0.0]
love         = [0.0, 1.0, 1.0, 0.0]
programming  = [1.0, 1.0, 0.0, 1.0]

```

这些是 **未加 Position 的向量**。



# 7. 对每个 token 应用 RoPE


## 7.1 Token: i（p = 0）


### Plane 1（k = 0）


$$
\theta = 0 \cdot 1 = 0
$$


Rotation 不变：


```
(x0, x1) = (1, 0)

```

### Plane 2（k = 1）


$$
\theta = 0 \cdot 0.01 = 0
$$


```
(x2, x3) = (1, 0)

```

### RoPE 后：


```vbnet
i' = [1, 0, 1, 0]

```


## 7.2 Token: love（p = 1）


### Plane 1


$$
\theta = 1 \cdot 1 = 1
$$


$$
\cos 1 \approx 0.5403,\quad \sin 1 \approx 0.8415
$$


原始：


```
(x0, x1) = (0, 1)

```

旋转后：

x0′x1′
$$
\begin{aligned}
x_0' &= 0 \cdot \cos 1 - 1 \cdot \sin 1 = -0.8415 \\
x_1' &= 0 \cdot \sin 1 + 1 \cdot \cos 1 = 0.5403
\end{aligned}
$$



### Plane 2


$$
\theta = 1 \cdot 0.01 = 0.01
$$


$$
\cos 0.01 \approx 0.99995,\quad \sin 0.01 \approx 0.01
$$


原始：


```
(x2, x3) = (1, 0)

```

旋转后：


```vbnet
x2' ≈ 0.99995
x3' ≈ 0.01

```


### RoPE 后：


```vbnet
love' ≈ [-0.8415, 0.5403, 0.99995, 0.01]

```


## 7.3 Token: programming（p = 2）


### Plane 1


$$
\theta = 2
$$


$$
\cos 2 \approx -0.4161,\quad \sin 2 \approx 0.9093
$$


原始：


```
(x0, x1) = (1, 1)

```

旋转后：

x0′x1′
$$
\begin{aligned}
x_0' &= 1 \cdot \cos 2 - 1 \cdot \sin 2 = -1.3254 \\
x_1' &= 1 \cdot \sin 2 + 1 \cdot \cos 2 = 0.4932
\end{aligned}
$$



### Plane 2


$$
\theta = 2 \cdot 0.01 = 0.02
$$


$$
\cos 0.02 \approx 0.9998,\quad \sin 0.02 \approx 0.02
$$


原始：


```
(x2, x3) = (0, 1)

```

旋转后：


```vbnet
x2' ≈ -0.02
x3' ≈ 0.9998

```


### RoPE 后：


```vbnet
programming' ≈ [-1.3254, 0.4932, -0.02, 0.9998]

```


# 8. RoPE 如何编码 relative position（关键结论）


在 Attention 中：


$$
Q_i \cdot K_j
$$


经过 RoPE 后：


$$
\text{RoPE}(Q_i) \cdot \text{RoPE}(K_j)
=
Q_i \cdot R_{(j-i)} K_j
$$


也就是说：



**dot-product 显式依赖于 position difference (j - i)**



这是 RoPE 相对于 **Sinusoidal PE** 的本质优势。



# 9. 总结（工程视角）


| 维度 | RoPE 特性 |
| ---- | ---- |
| 注入方式 | Multiplicative（Rotation） |
| 位置类型 | Relative Position |
| Attention 兼容性 | 原生支持 |
| 长序列 extrapolation | 优秀 |
| d_model 要求 | 必须为偶数 |



如果你愿意，下一步我可以：


- 推导 **complex number 视角的 RoPE**
- 对比 **RoPE vs ALiBi vs Absolute PE**
- 从 **源码（PyTorch / LLaMA）** 层面拆解实现
- 画出 **Attention 几何解释图**

你可以直接指定。

这里需要**明确纠正一个事实性错误**，而且这个纠正非常重要。

> **《Attention Is All You Need》（2017）中根本没有 RoPE。**

你现在的表述在时间线与方法归属上是**不成立的**。下面我用**严格的文献与技术事实**来澄清，并解释为什么会产生这种混淆。
