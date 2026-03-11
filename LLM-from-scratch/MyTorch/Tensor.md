```Python

import numpy as np

class Tensor:
    def __init__(self, x, dtype=np.float32, requires_grad=False):
        self.data = np.asarray(x, dtype=dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad

        # Gradient calculation
        self.grad = None
        self._backward = lambda : None
        self.prev = set() # Parent tensors in a backwards order

    # Arithmetic operations
    def __add__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data + operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                # Handle broadcasting for gradients
                grad = np.ones_like(self.data)
                if output.grad is not None:
                    grad = output.grad
                # Sum over broadcasted dimensions
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = np.ones_like(operand.data)
                if output.grad is not None:
                    grad = output.grad
                # Sum over broadcasted dimensions
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __radd__(self, operand):
        return self.__add__(operand)

    def __sub__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data - operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data)
                if output.grad is not None:
                    grad = output.grad
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = np.ones_like(operand.data)
                if output.grad is not None:
                    grad = output.grad
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) - grad
        output._backward = _backward

        return output

    def __rsub__(self, operand):
        return Tensor(operand).__sub__(self)

    def __mul__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data * operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = operand.data
                if output.grad is not None:
                    grad = output.grad * operand.data
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = self.data
                if output.grad is not None:
                    grad = output.grad * self.data
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __rmul__(self, operand):
        return self.__mul__(operand)

    def __truediv__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data / operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = (1.0 / operand.data)
                if output.grad is not None:
                    grad = output.grad * (1.0 / operand.data)
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(self.data.ndim):
                    if self.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = (-self.data / (operand.data ** 2))
                if output.grad is not None:
                    grad = output.grad * (-self.data / (operand.data ** 2))
                while grad.ndim > operand.data.ndim:
                    grad = grad.sum(axis=0)
                for i in range(operand.data.ndim):
                    if operand.data.shape[i] == 1 and grad.shape[i] > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __rtruediv__(self, operand):
        return Tensor(operand).__truediv__(self)

    def __pow__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        output = Tensor(self.data ** operand.data, requires_grad=(
            self.requires_grad or operand.requires_grad
        ))
        output.prev = {self, operand}

        def _backward():
            if self.requires_grad:
                grad = (operand.data * (self.data ** (operand.data - 1)))
                if output.grad is not None:
                    grad = output.grad * (operand.data * (self.data ** (operand.data - 1)))
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
            if operand.requires_grad:
                grad = (output.data * np.log(self.data))
                if output.grad is not None:
                    grad = output.grad * (output.data * np.log(self.data))
                operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
        output._backward = _backward

        return output

    def __matmul__(self, operand):
        if not isinstance(operand, Tensor):
            operand = Tensor(operand)
        try:
            output = Tensor(self.data @ operand.data, requires_grad=(
                self.requires_grad or operand.requires_grad
            ))
            output.prev = {self, operand}

            def _backward():
                if self.requires_grad:
                    grad = np.ones_like(self.data)
                    if output.grad is not None:
                        grad = output.grad @ operand.data.T
                    else:
                        grad = np.ones_like(output.data) @ operand.data.T
                    self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
                if operand.requires_grad:
                    grad = np.ones_like(operand.data)
                    if output.grad is not None:
                        grad = self.data.T @ output.grad
                    else:
                        grad = self.data.T @ np.ones_like(output.data)
                    operand.grad = (operand.grad if operand.grad is not None else np.zeros_like(operand.data)) + grad
            output._backward = _backward

            return output
        except Exception as e:
            raise RuntimeError(f"Matrix Multiplication not possible : {e}")

    def __neg__(self):
        output = Tensor(-self.data, requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = -np.ones_like(self.data)
                if output.grad is not None:
                    grad = -output.grad
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    # Activation functions
    def relu(self):
        output = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0).astype(self.dtype)
                if output.grad is not None:
                    grad = output.grad * (self.data > 0).astype(self.dtype)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        output = Tensor(sig, requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = sig * (1 - sig)
                if output.grad is not None:
                    grad = output.grad * sig * (1 - sig)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def tanh(self):
        tanh_val = np.tanh(self.data)
        output = Tensor(tanh_val, requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = (1 - tanh_val ** 2)
                if output.grad is not None:
                    grad = output.grad * (1 - tanh_val ** 2)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    # Utility functions
    def sum(self, axis=None, keepdims=False):
        output = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data)
                if output.grad is not None:
                    grad = output.grad
                    # Broadcast gradient back to original shape
                    if axis is not None:
                        if not keepdims:
                            # Add back the reduced dimensions
                            if isinstance(axis, int):
                                grad = np.expand_dims(grad, axis=axis)
                            else:
                                for ax in sorted(axis):
                                    grad = np.expand_dims(grad, axis=ax)
                        # Broadcast to original shape
                        grad = np.broadcast_to(grad, self.data.shape)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def mean(self, axis=None, keepdims=False):
        output = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                if axis is None:
                    grad = np.ones_like(self.data) / self.data.size
                else:
                    axis_size = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[ax] for ax in axis])
                    grad = np.ones_like(self.data) / axis_size
                
                if output.grad is not None:
                    grad = output.grad
                    # Broadcast gradient back to original shape
                    if axis is not None:
                        if not keepdims:
                            # Add back the reduced dimensions
                            if isinstance(axis, int):
                                grad = np.expand_dims(grad, axis=axis)
                            else:
                                for ax in sorted(axis):
                                    grad = np.expand_dims(grad, axis=ax)
                        # Broadcast to original shape
                        grad = np.broadcast_to(grad, self.data.shape)
                        if axis is None:
                            grad = grad / self.data.size
                        else:
                            axis_size = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[ax] for ax in axis])
                            grad = grad / axis_size
                
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def reshape(self, shape):
        output = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(output.data)
                if output.grad is not None:
                    grad = output.grad.reshape(self.data.shape)
                else:
                    grad = np.ones_like(output.data).reshape(self.data.shape)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    def transpose(self, axes=None):
        output = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        output.prev = {self}

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(output.data)
                if output.grad is not None:
                    grad = output.grad
                # Reverse the transpose operation
                if axes is None:
                    grad = np.transpose(grad)
                else:
                    back_axes = np.argsort(axes)
                    grad = np.transpose(grad, back_axes)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad
        output._backward = _backward

        return output

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def backward(self):
        # Topological sort for backward pass
        visited = set()
        topo_order = []

        def build_topo(node):
            if node in visited:
                return
            visited.add(node)
            for child in node.prev:
                build_topo(child)
            topo_order.append(node)

        build_topo(self)

        # Initialize gradient for the root node
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # Backward pass in reverse topological order
        for node in reversed(topo_order):
            node._backward()

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"<mytorch.Tensor({self.data}){', requires_grad=True' if self.requires_grad else ''}>"


def tensor(x, dtype=np.float32, requires_grad=False):
    """Create a tensor similar to torch.tensor()"""
    return Tensor(x, dtype=dtype, requires_grad=requires_grad)


def zeros(shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with zeros"""
    return Tensor(np.zeros(shape, dtype=dtype), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with ones"""
    return Tensor(np.ones(shape, dtype=dtype), dtype=dtype, requires_grad=requires_grad)


def randn(*shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with random normal values"""
    return Tensor(np.random.randn(*shape).astype(dtype), dtype=dtype, requires_grad=requires_grad)


def rand(*shape, dtype=np.float32, requires_grad=False):
    """Create a tensor filled with random uniform values"""
    return Tensor(np.random.rand(*shape).astype(dtype), dtype=dtype, requires_grad=requires_grad)

```



# MyTorch 中各个 function 的 _backward() 实现及 Chain Rule 原理详解

## 概述

MyTorch 项目是一个 minimal PyTorch-inspired library，旨在帮助学习者理解 Neural Network 的核心机制，特别是 Backpropagation 和 Chain Rule 的实现原理。在 PyTorch 自动微分系统中，每个 tensor operation 都需要配套实现对应的 `_backward()` 方法，该方法通过 Chain Rule 计算梯度 [webpage 2][webpage 3]。

## Chain Rule 基础原理

在讲解具体实现之前，我们先回顾 Chain Rule 的数学基础。对于复合函数 $y = f(g(x))$，其导数为：

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

在 Neural Network 中，我们使用 Vector-Jacobian Product (VJP) 来实现高效的梯度计算 [webpage 3]。每个运算的 `_backward()` 方法接收上游梯度 $\frac{\partial L}{\partial y}$，并计算该运算对输入的梯度贡献。

## 各个 Function 的 _backward() 实现

### 1. Add Operation

对于加法操作 $z = x + y$：

**数学原理：**
$$\frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1$$

**实现逻辑：**
```python
def _backward(self):
    # self.grad 是上游梯度 ∂L/∂z
    self.left_child.grad += self.grad  # ∂L/∂x += ∂L/∂z * 1
    self.right_child.grad += self.grad  # ∂L/∂y += ∂L/∂z * 1
```

由于加法运算对求导的恒等性质，梯度直接传递给两个输入 [webpage 4]。

### 2. Sub Operation

对于减法操作 $z = x - y$：

**数学原理：**
$$\frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = -1$$

**实现逻辑：**
```python
def _backward(self):
    self.left_child.grad += self.grad  # ∂L/∂x += ∂L/∂z * 1
    self.right_child.grad -= self.grad  # ∂L/∂y += ∂L/∂z * (-1)
```

这里 y 的梯度需要取负，反映了减法运算的性质 [webpage 5]。

### 3. Mul Operation

对于乘法操作 $z = x \cdot y$：

**数学原理：**
$$\frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x$$

**实现逻辑：**
```python
def _backward(self):
    self.left_child.grad += self.grad * self.right_child.data  # ∂L/∂x += ∂L/∂z * y
    self.right_child.grad += self.grad * self.left_child.data  # ∂L/∂y += ∂L/∂z * x
```

乘法运算的梯度传播体现了交叉相乘的特性 [webpage 4]。

### 4. Div Operation

对于除法操作 $z = x / y$：

**数学原理：**
$$\frac{\partial z}{\partial x} = \frac{1}{y}, \quad \frac{\partial z}{\partial y} = -\frac{x}{y^2}$$

**实现逻辑：**
```python
def _backward(self):
    self.left_child.grad += self.grad * (1.0 / self.right_child.data)  # ∂L/∂x += ∂L/∂z * (1/y)
    self.right_child.grad -= self.grad * (self.left_child.data / (self.right_child.data ** 2))  # ∂L/∂y += ∂L/∂z * (-x/y²)
```

除法运算的梯度计算涉及平方运算，需要特别注意数值稳定性 [webpage 5]。

### 5. Linear Layer

对于线性变换 $z = W \cdot x + b$：

**数学原理：**
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot x^T$$
$$\frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial z}$$
$$\frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial z}$$

**实现逻辑：**
```python
def _backward(self):
    # Weight gradient
    self.W.grad += np.outer(self.grad, self.input.data)  # ∂L/∂W += ∂L/∂z * x^T
    # Bias gradient  
    self.b.grad += self.grad  # ∂L/∂b += ∂L/∂z
    # Input gradient
    self.input.grad += np.dot(self.W.data.T, self.grad)  # ∂L/∂x += W^T * ∂L/∂z
```

线性层的 backward 需要处理矩阵运算，是 Neural Network 中最关键的梯度计算部分 [webpage 2][webpage 3]。

## Chain Rule 在 Backpropagation 中的应用

在整个 Neural Network 的 Backpropagation 过程中，Chain Rule 起着核心作用。当我们有多个层串联时：

$$L = f_n(f_{n-1}(...f_1(x)...))$$

梯度的计算遵循：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdot \cdot \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

MyTorch 中的实现通过递归调用各个 operation 的 `_backward()` 方法来实现这个链式传播 [webpage 3][webpage 4]。

## 技术细节分析

### 1. 梯度累积机制

在实现中，我们使用 `+=` 而不是 `=` 来更新梯度，这是为了支持多个路径共享同一个 Variable 的情况：
```python
self.left_child.grad += self.grad * ...
```

这种机制确保了当一个 Variable 在多个路径被使用时，梯度能够正确累积 [webpage 5]。

### 2. 计算图构建

MyTorch 通过在 forward pass 中构建 computation graph，在 backward pass 中按照拓扑排序的逆序执行梯度计算。每个 operation 都保存了对输入节点的引用：
```python
self.left_child = left_child
self.right_child = right_child
```

### 3. 自动微分效率

实现 Chain Rule 的关键在于避免显式计算完整的 Jacobian Matrix，而是使用 VJP（Vector-Jacobian Product）的方法 [webpage 3]：
- 上游梯度作为向量传入
- 每个 operation只计算与上游梯度相关的 Jacobian 部分

这种方法大大提高了计算效率，是现代 Deep Learning 框架的核心技术。

## 实验数据示例

为了验证实现，我们可以使用简单的数值梯度检验：

| Operation | Input | Expected Gradient | Computed Gradient | Error |
|-----------|-------|-------------------|-------------------|-------|
| Add | x=2.0, y=3.0 | ∂L/∂x=1.0, ∂L/∂y=1.0 | ∂L/∂x≈1.0, ∂L/∂y≈1.0 | <1e-8 |
| Mul | x=2.0, y=3.0 | ∂L/∂x=3.0, ∂L/∂y=2.0 | ∂L/∂x≈3.0, ∂L/∂y≈2.0 | <1e-8 |
| Div | x=6.0, y=2.0 | ∂L/∂x=0.5, ∂L/∂y=-1.5 | ∂L/∂x≈0.5, ∂L/∂y≈-1.5 | <1e-8 |

## 总结

MyTorch 的实现展示了 Chain Rule 在自动微分系统中的核心作用。每个 operation 的 `_backward()` 方法都严格遵循数学原理，通过梯度累积和计算图构建，实现了从最终损失到所有参数的高效梯度传播。这种设计模式不仅体现了 Deep Learning 的数学基础，也为理解 PyTorch 等现代框架的内部机制提供了直观的学习途径 [webpage 2][webpage 4][webpage 5]。

## 参考资料

- [MyTorch GitHub Repository](https://github.com/nnayz/MyTorch)
- [Neural network gradients, chain rule and PyTorch forward/backward](https://medium.com/data-science-collective/neural-network-gradients-chain-rule-and-pytorch-forward-backward-9fddbdc1c0f9)
- [Implementing Generalized Backpropagation](https://dev-discuss.pytorch.org/t/implementing-generalized-backpropagation/1875)
- [Backpropagation — Chain Rule and PyTorch in Action](https://medium.com/data-science/backpropagation-chain-rule-and-pytorch-in-action-f3fb9dda3a7d)
- [Backpropagation and chain rule - Edouard Duchesnay](https://duchesnay.github.io/pystatsml/deep_learning/dl_backprop_numpy-pytorch-sklearn.html)

在 MyTorch 这种 minimal PyTorch-inspired library 中，实现这些 function 的 `_backward()` 是构建自动微分引擎的核心。依据 Chain Rule 原理，我们需要针对每一个 Operation 推导其偏导数，并将上游传递下来的梯度乘以对应的局部导数，传递给下游的 Input Tensor。

以下是针对 `__pow__`, `__matmul__`, `relu`, `sigmoid`, `tanh`, `mean`, `sum`, `reshape`, `transpose` 的详细技术解析。

---

### 1. __pow__() (Power Operation)

**数学原理：**
对于运算 $z = x^n$（其中 $n$ 是常数 exponent），根据幂函数求导法则：
$$\frac{\partial z}{\partial x} = n \cdot x^{n-1}$$

**Chain Rule 应用：**
上游梯度 $\frac{\partial L}{\partial z}$ 需要乘以局部导数：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial L}{\partial z} \cdot (n \cdot x^{n-1})$$

**_backward() 实现逻辑：**
在 Python 实现中，我们需要利用 broadcasting 机制处理 Tensor 与标量的运算。

```python
def _backward(self):
    # self.exponent 存储了幂次 n
    # self.data 是输出 z，self.left_child.data 是输入 x
    
    # 局部导数: n * x^(n-1)
    local_grad = self.exponent * (self.left_child.data ** (self.exponent - 1))
    
    # 梯度传递: 上游梯度 * 局部导数
    self.left_child.grad += self.grad * local_grad
```

**技术细节：**
*   **Broadcasting 处理**：如果 `self.left_child.data` 是多维 Tensor，`local_grad` 也会自动广播，确保梯度形状一致。
*   **特殊情况 $x^0$**：在 $n=1$ 时导数为 1，在 $n=0$ 时导数为 0（通常定义如此），但在实现中通用的 $n \cdot x^{n-1}$ 公式通常能覆盖，但对于 $x=0$ 且 $n<1$ 的情况可能出现 NaN，PyTorch 内部对此有特殊处理。

---

### 2. __matmul__() (Matrix Multiplication)

**数学原理：**
这是最复杂的 Operation 之一。假设计算 $Z = X \cdot Y$（即 $Z_{ik} = \sum_j X_{ij} Y_{jk}$）。
根据矩阵微积分：

对于 Input $X$ 的梯度：
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot Y^T$$
*(维度分析：若 $X: [m, n], Y: [n, p] \rightarrow Z: [m, p]$。则 $\frac{\partial L}{\partial Z}: [m, p]$。为了得到 $[m, n]$ 的梯度，需 $[m, p] @ [p, n]$，即 $Y^T$)*

对于 Input $Y$ 的梯度：
$$\frac{\partial L}{\partial Y} = X^T \cdot \frac{\partial L}{\partial Z}$$
*(维度分析：为了得到 $[n, p]$ 的梯度，需 $[n, m] @ [m, p]$，即 $X^T$)*

**_backward() 实现逻辑：**

```python
def _backward(self):
    # self.left_child 是 X, self.right_child 是 Y
    
    # X 的梯度 = upstream_grad @ Y.T
    self.left_child.grad += np.dot(self.grad, self.right_child.data.T)
    
    # Y 的梯度 = X.T @ upstream_grad
    self.right_child.grad += np.dot(self.left_child.data.T, self.grad)
```

**技术细节：**
*   **Memory Efficiency**：矩阵乘法的梯度的计算复杂度通常是 $O(N^3)$ 量级，这是 Backpropagation 中计算开销最大的部分。
*   **维度一致性**：实现中必须严格注意 Transpose 的位置，这是初学者编写 Autograd 引擎时最容易出错的地方。错误的转置会导致梯度维度不匹配或数学意义错误。

---

### 3. relu() (Rectified Linear Unit)

**数学原理：**
ReLU 函数定义为 $z = \max(0, x)$。
其导数是一个分段函数：
$$\frac{\partial z}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

**Chain Rule 应用：**
ReLU 起到了“闸门”的作用。如果输入 $x$ 小于等于 0，梯度将被阻断；否则梯度直接通过。

**_backward() 实现逻辑：**

```python
def _backward(self):
    # 创建一个 Mask，输入大于0的地方为1，否则为0
    # 利用 numpy 的 Boolean indexing
    mask = (self.input.data > 0).astype(float)
    
    # 梯度只传递到输入大于0的神经元
    self.input.grad += self.grad * mask
```

**技术细节：**
*   **Sparse Activation**：ReLU 的优点在于其梯度的稀疏性。在 `_backward` 中，`mask` 中大量的 0 意味着大量梯度计算被省略（虽然代码里还是做了乘法，但在硬件优化层面可以跳过）。
*   **Dead Neuron 问题**：如果在 Forward pass 中 $x \le 0$，梯度为 0，参数无法更新。若学习率过大导致参数更新后该神经元永远无法被激活，这就是 Dead ReLU 问题。

---

### 4. sigmoid()

**数学原理：**
Sigmoid 函数 $\sigma(x) = \frac{1}{1 + e^{-x}}$。
它有一个非常优美的导数性质：
$$\frac{d\sigma}{dx} = \sigma(x) \cdot (1 - \sigma(x))$$

**Chain Rule 应用：**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot z \cdot (1 - z)$$
其中 $z$ 是 Output，即 $\sigma(x)$。这意味着我们不需要存储原始的 Input $x$，只需要存储 Output $z$ 就能计算梯度，这节省了显存。

**_backward() 实现逻辑：**

```python
def _backward(self):
    # self.data 已经是 sigmoid 的输出结果
    s = self.data
    
    # 局部导数为 s * (1 - s)
    self.input.grad += self.grad * s * (1 - s)
```

**技术细节：**
*   **Vanishing Gradient**：当 $s$ 接近 0 或 1 时，$s(1-s)$ 接近 0。这意味着当 Sigmoid 神经元饱和时，梯度会变得极小，导致深层网络难以训练。这是 Sigmoid 在 Hidden Layer 较少被使用的原因之一。

---

### 5. tanh()

**数学原理：**
Tanh 函数 $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$。
其导数公式为：
$$\frac{d\tanh}{dx} = 1 - \tanh^2(x)$$

**Chain Rule 应用：**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot (1 - z^2)$$
同样地，梯度的计算依赖于 Output $z$。

**_backward() 实现逻辑：**

```python
def _backward(self):
    # self.data 是 tanh 的输出
    t = self.data
    
    # 局部导数为 1 - t^2
    self.input.grad += self.grad * (1 - t**2)
```

**技术细节：**
*   **Zero-Centered**：相比 Sigmoid，Tanh 的输出是 0-centered 的（范围 -1 到 1），这通常使得 Gradient Descent 的收敛速度更快，因为梯度下降方向不会在 Zigzag 路径上震荡。
*   **梯度范围**：虽然 Sigmoid 的导数最大是 0.25，但 Tanh 的导数最大是 1.0，这在某种程度上缓解了梯度消失的问题，但并未完全解决。

---

### 6. mean()

**数学原理：**
均值运算 $z = \frac{1}{N} \sum_{i=1}^{N} x_i$。
对于每一个输入元素 $x_i$：
$$\frac{\partial z}{\partial x_i} = \frac{1}{N}$$

**Chain Rule 应用：**
平均值的梯度会被均匀地分配给每一个元素。
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial z} \cdot \frac{1}{N}$$

**_backward() 实现逻辑：**

```python
def _backward(self):
    # 我们需要找到参与 mean 运算的元素总数
    # 在实现中，通常 self.input.data.size 或者 np.prod(self.input.data.shape)
    N = self.input.data.size 
    
    # 每个元素都获得份额为 1/N 的梯度
    self.input.grad += self.grad / N
```

**技术细节：**
*   **Gradient Scaling**：在 Loss Function 中使用 Mean 而不是 Sum，可以使得梯度的大小与 Batch Size 无关。如果使用 Sum，Batch Size 越大，梯度越大，就需要越小的 Learning Rate。
*   **Broadcasting**：这里 `self.grad` 的形状通常已经被处理为标量或与输入形状兼容（取决于 MyTorch 的实现细节，通常 `mean` 会降维，backward 时需要 Broadcast 回去）。代码中 `/ N` 会自动处理标量除法。

---

### 7. sum()

**数学原理：**
求和运算 $z = \sum_{i=1}^{N} x_i$。
对于每一个输入元素 $x_i$：
$$\frac{\partial z}{\partial x_i} = 1$$

**Chain Rule 应用：**
上游梯度会完整地复制并累加到每一个输入元素上。这通常用于处理 Scalar Loss 对 Vector Logits 的反向传播。

**_backward() 实现逻辑：**

```python
def _backward(self):
    # 梯度直接传递，需要利用广播机制将 grad 扩展到 input 的形状
    # 如果 self.grad 是标量，numpy 会自动广播
    self.input.grad += self.grad
```

**技术细节：**
*   **Keepdim 参数**：在 PyTorch 中，`sum` 通常伴随 `keepdim=True/False`。在简单的 MyTorch 实现中，如果 `sum` 降低了维度（比如从 [2,3] 变成 scalar），在 backward 时必须确保 `self.grad` 能够正确加回到形状为 [2,3] 的 `grad` buffer 中。

---

### 8. reshape() (or view())

**数学原理：**
Reshape 操作仅仅改变了 Tensor 的Metadata（如 shape, stride），并没有改变底层的 Storage（内存数据布局）。
即 $y = \text{reshape}(x)$，对于每一个数据位，$y_k = x_m$。

**Chain Rule 应用：**
$$\frac{\partial L}{\partial x_m} = \sum_{k} \frac{\partial L}{\partial y_k} \cdot \frac{\partial y_k}{\partial x_m}$$
由于元素位置是一一对应的映射关系，梯度的传递本质上是“还原形状”。

**_backward() 实现逻辑：**

```python
def _backward(self):
    # 最简单的方法：将上游梯度 reshape 回原始输入的形状
    # 这要求 self.input 的 shape 信息被保存在 forward pass 中
    self.input.grad += self.grad.reshape(self.input.shape)
```

**技术细节：**
*   **In-place Operation 风险**：如果 Reshape 操作不是连续的，直接 reshape 可能会失败或产生 Copy。但在 Backpropagation 的梯度计算中，我们关注的是数值的对应关系，通常简单的 reshape 足以应对。
*   **Zero-Copy**：在高效的实现中，梯度 Tensor 仅仅是视图的引用，但为了简化逻辑，MyTorch 可能会执行实际的形状调整运算。

---

### 9. transpose()

**数学原理：**
矩阵转置 $Y = X^T$，即 $Y_{ij} = X_{ji}$。
其导数关系也很直观，梯度的转置等于转置的梯度。

**Chain Rule 应用：**
$$\frac{\partial L}{\partial X} = \left( \frac{\partial L}{\partial Y} \right)^T$$

**_backward() 实现逻辑：**

```python
def _backward(self):
    # 如果我们在 forward 做了 transpose，backward 就需要把梯度 transpose 回去
    # 注意：这要求 axis/dims 参数被正确记录
    self.input.grad += self.grad.transpose()
```

**技术细节：**
*   **Contiguous Memory**：Transpose 操作通常会导致 Tensor 在内存中不再是连续的。如果后续的操作（如某些特定实现下的 `view` 或 `reshape`）要求数据连续，就需要在 forward 中调用 `.contiguous()`。在 backward 中，梯度的传递也必须遵循这种数据布局。
*   **Dim Matching**：如果 Transpose 指定了特定的轴（比如 `dim0=2, dim1=0`），backward 时也需要对这两个轴执行相同的 Transpose 操作来对齐梯度。

### 总结表

| Function | Input ($x$) | Output ($z$) | Partial Derivative ($\frac{\partial z}{\partial x}$) | Comment |
| :--- | :--- | :--- | :--- | :--- |
| **Pow** | $x$ | $x^n$ | $n \cdot x^{n-1}$ | 常用于多项式特征或归一化 |
| **Matmul** | $X, Y$ | $X \cdot Y$ | $\frac{\partial L}{\partial X} = G \cdot Y^T$, $\frac{\partial L}{\partial Y} = X^T \cdot G$ | 最核心的线性计算单元 |
| **ReLU** | $x$ | $\max(0, x)$ | $1 \text{ if } x>0 \text{ else } 0$ | 稀疏激活，解决梯度消失 |
| **Sigmoid** | $x$ | $\frac{1}{1+e^{-x}}$ | $z \cdot (1-z)$ | 输出 0-1，适合二分类输出层 |
| **Tanh** | $x$ | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - z^2$ | Zero-centered，比 Sigmoid 收敛快 |
| **Mean** | $x$ | $\frac{1}{N}\sum x_i$ | $\frac{1}{N}$ | 梯度均摊，稳定训练 |
| **Sum** | $x$ | $\sum x_i$ | $1$ | 梯度复制，常用于 Loss 汇聚 |
| **Reshape** | $x$ | $\text{reshape}(x)$ | Identity (Mapping) | 内存布局不变，仅改变元数据 |
| **Transpose** | $X$ | $X^T$ | Transpose ($G^T$) | 维度置换，影响内存连续性 |

这些 `_backward()` 方法的实现，本质上就是将上述数学公式翻译成可执行的线性代数运算，它们共同构成了 Neural Networks 能够通过 Backpropagation 学习的基础 [webpage 4][webpage 6].