# Softmax 函数的梯度推导

## 一、Softmax 函数定义

对于输入向量 **z** = [z₁, z₂, ..., z_K]ᵀ ∈ ℝ^K，Softmax 函数定义为：

$$s_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

其中：
- e^{z_i} 是第 i 个元素的指数
- 分母是所有元素指数的和（归一化因子）

---

## 二、梯度推导：∂s_i/∂z_j

Softmax 函数的梯度是一个 **Jacobian 矩阵**，其元素为 ∂s_i/∂z_j。

### **情况一：当 i = j 时**

$$s_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = \frac{e^{z_i}}{D}$$

其中 D = Σ_{j=1}^K e^{z_j}

使用商的导数法则：

$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \cdot D - e^{z_i} \cdot e^{z_i}}{D^2} = \frac{e^{z_i}(D - e^{z_i})}{D^2}$$

整理得到：

$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i}}{D} \cdot \frac{D - e^{z_i}}{D} = s_i \left(1 - \frac{e^{z_i}}{D}\right) = s_i(1 - s_i)$$

### **情况二：当 i ≠ j 时**

$$\frac{\partial s_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}\right)$$

此时 e^{z_i} 视为常数（对 z_j 求导）：

$$\frac{\partial s_i}{\partial z_j} = e^{z_i} \cdot \frac{\partial}{\partial z_j}\left(\frac{1}{D}\right) = e^{z_i} \cdot \left(-\frac{1}{D^2} \cdot e^{z_j}\right) = -\frac{e^{z_i} \cdot e^{z_j}}{D^2}$$

整理得到：

$$\frac{\partial s_i}{\partial z_j} = -\frac{e^{z_i}}{D} \cdot \frac{e^{z_j}}{D} = -s_i \cdot s_j$$

---

## 三、梯度矩阵（Jacobian）的完整表示

将两种情况合并，可以用 **Kronecker delta** 符号 δ_{ij}（当 i=j 时为1，否则为0）表示：

$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$

或者写成矩阵形式：

$$\nabla_z \text{softmax}(z) = \text{diag}(s) - s \cdot s^T$$

其中：
- s = [s₁, s₂, ..., s_K]ᵀ 是 softmax 输出向量
- diag(s) 是以 s 为对角线的对角矩阵
- s · sᵀ 是外积矩阵

### **具体矩阵形式示例（K=3）**

$$\nabla_z \text{softmax}(z) = \begin{bmatrix} s_1(1-s_1) & -s_1s_2 & -s_1s_3 \\ -s_2s_1 & s_2(1-s_2) & -s_2s_3 \\ -s_3s_1 & -s_3s_2 & s_3(1-s_3) \end{bmatrix}$$

---

## 四、在 Cross-Entropy Loss 中的梯度

### **Cross-Entropy Loss 定义**

$$L = -\sum_{i=1}^{K} y_i \log(s_i)$$

其中 y 是 one-hot 编码的真实标签。

### **梯度推导**

$$\frac{\partial L}{\partial z_j} = \sum_{i=1}^{K} \frac{\partial L}{\partial s_i} \cdot \frac{\partial s_i}{\partial z_j}$$

其中：
$$\frac{\partial L}{\partial s_i} = -\frac{y_i}{s_i}$$

所以：

$$\frac{\partial L}{\partial z_j} = \sum_{i=1}^{K} \left(-\frac{y_i}{s_i}\right) \cdot s_i(\delta_{ij} - s_j)$$

$$= \sum_{i=1}^{K} -y_i(\delta_{ij} - s_j) = -\sum_{i=1}^{K} y_i\delta_{ij} + \sum_{i=1}^{K} y_i s_j$$

由于 y 是 one-hot 向量，Σ y_i = 1，且只有一个 y_c = 1（c 是正确类别）：

$$\frac{\partial L}{\partial z_j} = -y_j + s_j = s_j - y_j$$

写成向量形式：

$$\nabla_z L = s - y$$

这是一个非常简洁优雅的结果！

---

## 五、数值稳定性实现

### **Log-Softmax**

为了数值稳定性，常使用 log-softmax：

$$\log s_i = z_i - \log\left(\sum_{j=1}^{K} e^{z_j}\right)$$

### **实现技巧（减去最大值）**

$$\text{softmax}(z)_i = \frac{e^{z_i - z_{\max}}}{\sum_{j=1}^{K} e^{z_j - z_{\max}}}$$

其中 z_max = max(z₁, ..., z_K)，这样可以防止指数溢出。

---

## 六、PyTorch 实现示例

```python
import torch
import torch.nn.functional as F

# Softmax 函数
z = torch.tensor([2.0, 1.0, 0.1])
s = F.softmax(z, dim=0)
print(f"Softmax output: {s}")

# 梯度计算
z.requires_grad_(True)
s = F.softmax(z, dim=0)
loss = -torch.log(s[0])  # 假设第一个类别是正确的
loss.backward()
print(f"Gradient: {z.grad}")  # 应该等于 s - y

# 使用 log_softmax 的更稳定版本
log_s = F.log_softmax(z, dim=0)
loss = -log_s[0]
loss.backward()
```

---

## 七、应用场景扩展

1. **Multi-class Classification**: Softmax + Cross-Entropy 是标准组合
2. **Attention Mechanism**: Attention 权重常使用 softmax 归一化
3. **Policy Gradient (RL)**: 策略网络的输出概率分布
4. **Gumbel-Softmax**: 可微分的离散采样技巧

---

## 八、参考资源

1. **Stanford CS231n Notes** - Softmax derivation:
   https://cs231n.github.io/linear-classify/#softmax

2. **Deep Learning Book (Goodfellow et al.)**:
   https://www.deeplearningbook.org/contents/optimization.html

3. **The Matrix Calculus You Need For Deep Learning**:
   https://explained.ai/matrix-calculus/

4. **Softmax Derivation - Stack Exchange**:
   https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function

5. **PyTorch Softmax Documentation**:
   https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html

6. **Why Softmax is Called Softmax**:
   https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/