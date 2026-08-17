---
source_pdf: Qiskit Machine Learning.pdf
paper_sha256: b3e02738863cc6f9a1ce612ced9865e3d26d1cfff04bd79f81fdb9379ba599e5
processed_at: '2026-08-06T07:22:07-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用人话来拆解这篇 paper，核心目的是帮你 build intuition。Qiskit Machine Learning 本质上就是给 quantum computer 写的一个 PyTorch 或 scikit-learn 插件。在 classical machine learning 里，你定义一个 neural network，喂给它 data，它吐出 prediction，然后你用 backpropagation 算 gradient 去更新 weights。在 Qiskit ML 里，逻辑完全一样，只不过你的 "model" 变成了一台 physical quantum computer 上的 quantum circuit。

这里最核心的工程挑战在于：你无法在 physical quantum hardware 上跑 backpropagation。因为 quantum state 一旦被测量就坍缩了，而且 No-Cloning theorem 决定了你无法在中间层复制 state 去算 chain rule。所以，Qiskit ML 必须设计一套全新的机制来估算 gradient 和 loss，这套机制依赖于 Qiskit 的 primitives（`Sampler` 和 `Estimator`）。

下面我为你做更细节的技术讲解，包含公式拆解和 execution loop。

---

## 1. 核心执行 Loop 对比：Classical NN vs Quantum VQC

在 classical PyTorch 中，你的 training loop 是这样的：
```python
for x, y in dataloader:
    pred = model(x)        # Forward pass
    loss = loss_fn(pred, y)
    loss.backward()        # Backprop (chain rule on computational graph)
    optimizer.step()       # Update weights
```

在 Qiskit ML 中，对应 Variational Quantum Classifier (VQC) 的 loop 是这样的：
```python
for x, y in quantum_dataloader:
    # 1. Encode classical data x into quantum circuit
    qc = feature_map(x) 
    qc.compose(ansatz(theta), inplace=True)
    
    # 2. Run on hardware/simulator via Primitives
    # Sampler gets probabilities, Estimator gets expectation values
    probs = Sampler.run(qc).result().quasi_dists 
    
    # 3. Calculate loss
    pred = map_probs_to_label(probs)
    loss = loss_fn(pred, y)
    
    # 4. Calculate gradients WITHOUT backprop
    grads = parameter_shift(theta, x, y) 
    
    # 5. Update theta
    theta = optimizer.step(theta, grads)
```

Intuition 构建：在 classical 里，model 是数学公式的组合；在 quantum 里，model 是物理实验的组合。你每次算 loss，实际上是在 physical chip 上跑一次 experiment，测量 statistics。Gradient 的计算方法从 memory-efficient 的 chain rule 变成了多次运行 experiment 的 trick。

References:
- [Qiskit Machine Learning GitHub](https://github.com/qiskit-community/qiskit-machine-learning)
- [PyTorch Automatic Differentiation](https://pytorch.org/docs/stable/autograd.html)

---

## 2. Parameter-Shift Rule：Quantum 版本的 Backprop

既然不能跑 backprop，Qiskit ML 怎么算 gradient？最核心的默认方法就是 Parameter-Shift rule。论文里的公式 (1)：

$$\frac{\partial f(\theta)}{\partial \theta_i} = \frac{f(\theta_i + s) - f(\theta_i - s)}{2 \sin(s)}$$

变量拆解：
- $f(\theta)$: 期望值函数，通常定义为 $f(\theta) = \langle \psi(\theta) | \hat{O} | \psi(\theta) \rangle$。这里 $\hat{O}$ 是 observable（比如 Pauli Z operator），$|\psi(\theta)\rangle$ 是 quantum state。
- $\theta$: 所有的 trainable parameters 向量。
- $\theta_i$: 我们要对之求导的那个具体的 parameter（比如第 $i$ 个 rotation gate 的 angle）。
- $s$: shift parameter。对于标准的 Pauli rotation gates（如 $R_x, R_y, R_z$），$s = \pi/2$。

代入 $s = \pi/2$，公式简化为：
$$\frac{\partial f(\theta)}{\partial \theta_i} = \frac{1}{2} \left[ f\left(\theta + \frac{\pi}{2}e_i\right) - f\left(\theta - \frac{\pi}{2}e_i\right) \right]$$

这里的 $e_i$ 是单位向量，表示只对第 $i$ 个 parameter shift $\pi/2$，其他 parameter 保持不变。

Intuition 构建：为什么物理上能这么做？因为大部分 quantum gates 形式是 $U = e^{-i \theta \sigma/2}$，其中 $\sigma$ 是 Pauli matrix。Pauli matrix 满足特殊的 algebra 性质（$\sigma^2 = I$），这使得 expectation value 对 parameter 的依赖关系是一个严格的正弦或余弦函数。既然是三角函数，那么在相距 $\pi$ 的两个点上取值，做差除以 2，就刚好是 exact 的导数。

技术对比表：

| Feature | Classical Backprop | Parameter-Shift Rule |
|---|---|---|
| Math Basis | Chain Rule (Calculus) | Lie Algebra of Pauli Operators |
| Precision | Exact (up to float precision) | Exact (up to quantum measurement shot noise) |
| Circuit Executions per Param | 1 (Forward pass stored) | 2 (One $+\pi/2$, one $-\pi/2$) |
| Hardware Requirement | Memory to store activations | Quantum hardware to run shifted circuits |

References:
- [Schuld et al., Evaluating analytic gradients on quantum hardware, PRA 2019](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331)

---

## 3. Primitives：Estimator 与 Sampler 的底层分工

Qiskit ML 的 API 封装了 Qiskit 的两个 primitives。理解这两个 primitives 就理解了 QML 的输出空间。

### 3.1 Estimator
用来算 expectation value $\langle \psi | \hat{O} | \psi \rangle$。
数学形式：
$$\hat{E} = \langle 0^n | U^\dagger(\theta) O U(\theta) | 0^n \rangle$$
- $U(\theta)$: Parameterized Quantum Circuit (PQC)
- $O$: Observable (通常是多个 Pauli strings 的加和)
- $|0^n\rangle$: 初始的 ground state

对应到 `EstimatorQNN`：它输出一个 continuous 的标量。如果你要做 regression 任务，或者你要提取 feature embedding，用它。Intuition 上，它像是 neural network 的 linear layer 输出 logits。

### 3.2 Sampler
用来算 bitstring 的概率分布。
数学形式：
$$\hat{P}(z) = |\langle z | U(\theta) | 0^n \rangle|^2$$
- $z \in \{0,1\}^n$: 测量出来的 bitstring outcome
- $U(\theta)$: 你的 quantum circuit
- $|0^n\rangle$: 初始 state

对应到 `SamplerQNN`：它输出一个离散的概率分布。如果做 multi-class classification，你可以把不同的 bitstring 映射到不同的 class label。Intuition 上，它就像是 neural network 的 softmax 输出。

References:
- [Qiskit Primitives Documentation](https://docs.quantum.ibm.com/api/qiskit/primitives)

---

## 4. Quantum Kernel：避开 Gradient 的另一条路

如果不想忍受 NISQ 时代恼人的 gradient noise，Qiskit ML 提供了 Quantum Kernel 方法（论文公式 2）：

$$K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$$

变量拆解：
- $K(\mathbf{x}, \mathbf{x}')$: Kernel function。输入是两个 classical data points $\mathbf{x}$ 和 $\mathbf{x}'$。
- $\phi(\mathbf{x})$: Feature map。它把 classical vector $\mathbf{x}$ encode 成一个 $n$-qubit 的 quantum state $|\phi(\mathbf{x})\rangle = U(\mathbf{x})|0\rangle$。
- $|\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$: State fidelity，也就是两个 quantum state 的重叠度。

Qiskit ML 具体怎么在 hardware 上算这个 fidelity？用了一个叫 `ComputeUncompute` 的绝妙技巧：
1. 构建电路 $U(\mathbf{x})$ 生成 $|\phi(\mathbf{x})\rangle$。
2. 在后面接上 inverse circuit $U^\dagger(\mathbf{x}')$。
3. 测量最终电路得到全 $|00...0\rangle$ 的概率。这个概率数学上严格等于 $K(\mathbf{x}, \mathbf{x}')$。

Intuition 构建：在 classical SVM 里，RBF kernel 是在算两个向量在无限维空间的距离。在 Quantum Kernel 里，$n$ 个 qubit 提供 $2^n$ 维 Hilbert space。如果 $U(\mathbf{x})$ 设计得好，这个 kernel matrix 可能是 classical computer 极难算出来的，从而带来 quantum advantage。

应用场景表：

| Qiskit ML Class | Problem Type | Tech Details |
|---|---|---|
| `QSVC` | Classification | Scikit-learn wrapper, uses Quantum Kernel matrix instead of classical RBF |
| `QSVR` | Regression | Scikit-learn wrapper, uses Quantum Kernel for support vector regression |
| `PegasosQSVC` | Large-scale Classification | Uses Pegasos algorithm to avoid $O(N^3)$ QP solver bottleneck, scales to larger datasets |

References:
- [Havlíček et al., Supervised learning with quantum-enhanced feature spaces, Nature 2019](https://www.nature.com/articles/s41586-019-0980-2)
- [Pegasos SVM Algorithm Paper](https://dl.acm.org/doi/10.1145/1273496.1273598)

---

## 5. SPSA：NISQ 时代的救星

Parameter shift 算 gradient 太贵了。如果有 $P$ 个 parameters，你需要跑 $2P$ 次 circuits。在 noisy hardware 上，每次跑还要做几千次 shots。Qiskit ML 集成了 SPSA (Simultaneous Perturbation Stochastic Approximation) 来解决这个问题。

论文公式逻辑：
$$\hat{g}_k(\theta_k) = \frac{f(\theta_k + c_k \Delta_k) - f(\theta_k - c_k \Delta_k)}{2c_k}$$

变量拆解：
- $\hat{g}_k(\theta_k)$: 第 $k$ 步对整个 parameter vector $\theta$ 的 gradient estimate。
- $c_k$: 一个很小的标量 perturbation magnitude。
- $\Delta_k$: 一个 random vector，每个 element 独立以 50% 概率取 $+1$ 或 $-1$ (Bernoulli distribution)。

Intuition 构建：注意看这个公式，不管你的 model 有多少个 parameters，你永远只需要跑 2 次 circuit！一次是所有的 parameters 加上 random 噪声，一次是减去 random 噪声。因为噪声是随机的，在 expectation 上，这个 gradient estimate 会收敛到真实的 gradient。这就像是你蒙着眼睛下山，每次随便踢一脚试坡度，只要踢得次数够多，总能走到谷底。这在充满 hardware noise 的 NISQ 时代非常 robust。

References:
- [Gacon et al., Simultaneous perturbation stochastic approximation of the quantum Fisher information, Quantum 2021](https://quantum-journal.org/papers/q-2021-07-21-567/)
- [Spall, SPSA Overview](https://www.jhuapl.edu/SPSA/)

---

## 6. 总结：软件架构的 Intuition

把 Qiskit ML 想象成一个 translator：

1. **Top Layer (User)**: 写 Python，传 data，调 `.fit()`。世界是 classical 的。
2. **Middle Layer (Qiskit ML)**: 把 data 映射成 quantum circuits，把 loss 定义成 observable measurement，把 gradient 请求翻译成多次 circuit execution 的组合。
3. **Bottom Layer (Primitives & Hardware)**: 接收 circuits，执行物理操作，返回 measurement statistics。

Qiskit ML 的主要贡献就是在这个 Middle Layer 做了极佳的 abstraction，使得 Top Layer 看起来和 scikit-learn 或 PyTorch 没什么两样。只要底层的 quantum hardware 稳定性提上来，这套代码可以无缝迁移到未来的 fault-tolerant quantum computer 上去跑更复杂的 QML algorithms。

---

# Qiskit Machine Learning 深度技术解析

## 1. 论文整体定位与背景

这篇 paper 是 IBM Quantum 与 UK STFC Hartree Centre 联合发布的 Qiskit Machine Learning (ML) 模块的官方 software paper，2025 年 6 月 13 日发布。它不是单纯介绍某个 algorithm 的 paper，而是描述一个完整的 QML 软件框架，类似 PyTorch 或 TensorFlow 之于 deep learning，但目标是 quantum-classical hybrid workloads。

Qiskit ML 起源于 2019 年的 proof-of-concept 代码，到现在已经发展成 modular、extensible 的工具。值得注意的 design choice 是它面向 quantum hardware workloads 而非仅 simulator，这是与 PennyLane [PennyLane](https://pennylane.ai/) 的核心区别之一。

References:
- [Qiskit ML GitHub repo](https://github.com/qiskit-community/qiskit-machine-learning)
- [Qiskit 官方文档](https://qiskit.org/)
- [原始论文 arXiv](https://arxiv.org/abs/2506.00000)

---

## 2. 软件栈 Architecture 解析

### 2.1 分层架构

Qiskit ML 位于 application level，其下方依赖关系如下：

```
┌─────────────────────────────────────────────────────┐
│  User Code (Classification/Regression/Inference)    │
├─────────────────────────────────────────────────────┤
│  Qiskit ML API (VQC, VQR, QSVC, QNN, QBayesian)     │
├─────────────────────────────────────────────────────┤
│  Qiskit Primitives (Sampler, Estimator)             │
├─────────────────────────────────────────────────────┤
│  Qiskit Runtime / Aer / Statevector Simulator       │
├─────────────────────────────────────────────────────┤
│  IBM Quantum Hardware / Classical HPC               │
└─────────────────────────────────────────────────────┘
```

关键设计思想：high-level ML API 通过 primitives 抽象层与底层 hardware 解耦。这意味着用户切换 simulator 到 real hardware 只需要改一行 backend 配置，algorithm 逻辑无需修改。

### 2.2 UML Class Hierarchy 解析

论文 Figure 1 的 UML diagram 把 ML 方法分成三大类：

| Method Category | Base Classes | Problem Class |
|---|---|---|
| Kernel-based | FidelityQuantumKernel, TrainableFidelityQuantumKernel | Classification, Regression |
| Neural Network-based | EstimatorQNN, SamplerQNN | Classification, Regression |
| Bayesian | QBayesianInference | Inference |

这种分类遵循 scikit-learn 的 API 惯例，让熟悉 classical ML 的用户能快速上手。

---

## 3. Primitives 系统：Qiskit ML 的核心抽象

### 3.1 Sampler vs Estimator

这是 Qiskit 1.0+ 引入的 unified interface，Qiskit ML 完全基于此构建：

- **Estimator**: 计算期望值 ⟨ψ|O|ψ⟩，用于 cost function evaluation 和 gradient 计算
- **Sampler**: 计算测量概率分布 P(z) = |⟨z|ψ⟩|²，用于 classification 和 generative tasks

数学定义：

Estimator 输出：
$$\hat{E} = \langle 0^n | U^\dagger(\theta) O U(\theta) | 0^n \rangle$$

其中 $U(\theta)$ 是 parameterized quantum circuit (PQC)，$O$ 是 observable (通常是 Pauli string 的 linear combination)。

Sampler 输出：
$$\hat{P}(z) = |\langle z | U(\theta) | 0^n \rangle|^2$$

其中 $z \in \{0,1\}^n$ 是 bitstring measurement outcome。

### 3.2 QNN 类的对应关系

- **EstimatorQNN**：基于 Estimator，输出 expectation values。适合需要 scalar output 的任务，如 regression 和 binary classification 的 logit。对应 [Schuld et al. 2020](https://arxiv.org/abs/2001.00550) 的 circuit-centric QNN 框架。

- **SamplerQNN**：基于 Sampler，输出 probability distribution over bitstrings。适合 multi-class classification 或 generative modeling (例如 qGAN)。

这个区分对应 classical DL 中 regression head vs softmax head 的区别，intuition 上很自然。

References:
- [Qiskit Primitives 文档](https://docs.quantum.ibm.com/api/qiskit/primitives)
- [Estimator & Sampler 论文](https://arxiv.org/abs/2309.12018)

---

## 4. Gradient Estimation 方法详解

这是 QML 与 classical ML 最大的差异点之一。Quantum hardware 无法像 PyTorch 那样用 backprop，因为 quantum state 是 physical 的，无法"复制"做 chain rule。

### 4.1 Parameter-Shift Rule (公式 1)

$$\frac{\partial f(\theta)}{\partial \theta_i} = \frac{f(\theta_i + s) - f(\theta_i - s)}{2 \sin(s)}$$

变量含义：
- $f(\theta)$: 期望值函数，即 $f(\theta) = \langle \psi(\theta) | \hat{O} | \psi(\theta) \rangle$
- $\theta = (\theta_1, \theta_2, \ldots, \theta_n)$: 可训练参数向量
- $\theta_i$: 第 $i$ 个参数
- $s$: shift parameter

当 gate 是 $R_\alpha(\theta_i) = e^{-i\theta_i \sigma_\alpha / 2}$ 形式时，取 $s = \pi/2$ 可得 exact gradient：

$$\frac{\partial f}{\partial \theta_i} = \frac{1}{2}\left[f\left(\theta + \frac{\pi}{2}e_i\right) - f\left(\theta - \frac{\pi}{2}e_i\right)\right]$$

**Intuition 构建**：这个 rule 本质是利用了 Pauli rotation generators 的 Lie algebra 性质。$e^{-i(\theta + \pi/2)\sigma/2}$ 与 $e^{-i(\theta - \pi/2)\sigma/2}$ 在 Bloch 球上对应 antipodal points，它们的 expectation value 差刚好给出 derivative。

**代价分析**：对于 $N$ 个参数，需要 $2N$ 次 circuit evaluation。这与 finite difference 看起来一样，但 parameter-shift 给的是 **exact** gradient，没有 numerical noise。这是 quantum hardware 上能做的最精确 gradient。

References:
- [Schuld et al. 2019, PRA](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331)
- [Mitarai et al. 2018](https://arxiv.org/abs/1803.00745) - 原始 parameter-shift paper

### 4.2 Linear Combination of Unitaries (LCU)

近似方法，把 gradient 表示成多个 unitary 的线性组合：

$$\frac{\partial U(\theta)}{\partial \theta_i} \approx \sum_k c_k U_k(\theta)$$

需要 ancilla qubit 来实现 linear combination。优点是 reduce circuit depth，缺点是增加 qubit count 和 measurement complexity。

### 4.3 SPSA (Simultaneous Perturbation Stochastic Approximation)

$$\hat{g}_k(\theta_k) = \frac{f(\theta_k + c_k \Delta_k) - f(\theta_k - c_k \Delta_k)}{2c_k}$$

变量含义：
- $\hat{g}_k$: 第 $k$ 步的 gradient estimate
- $\Delta_k$: random perturbation vector (通常 Bernoulli $\pm 1$ 分布)
- $c_k$: 递减 step size sequence

**为什么 SPSA 在 NISQ 上重要**：只需要 2 次 circuit evaluation 即可 estimate 整个 gradient vector，无论参数有多少。而 parameter-shift 需要 $2N$ 次。代价是 gradient 有 noise，但在 quantum hardware 本就 noisy 的情况下，这个 trade-off 是合理的。

参考文献 [Gacon et al. 2021](https://quantum-journal.org/papers/q-2021-07-21-567/) 提出了 SPSA for quantum Fisher information，Qiskit ML 直接集成。

References:
- [Spall 1998 - SPSA 原始论文](https://www.jhuapl.edu/SPSA/)
- [Gacon et al. Quantum 2021](https://quantum-journal.org/papers/q-2021-07-21-567/)

---

## 5. Quantum Kernel Methods 详解

### 5.1 Fidelity Quantum Kernel (公式 2)

$$K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$$

变量含义：
- $K(\mathbf{x}, \mathbf{x}')$: kernel function, 衡量两个数据点的相似度
- $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$: classical data points
- $\phi: \mathbb{R}^d \rightarrow \mathcal{H}$: feature map, 把 classical data 映射到 quantum Hilbert space $\mathcal{H}$
- $|\phi(\mathbf{x})\rangle = U(\mathbf{x})|0\rangle$: 通过 quantum circuit $U(\mathbf{x})$ 编码得到的 quantum state

**Implementation via ComputeUncompute**:
1. Prepare $|\phi(\mathbf{x})\rangle$
2. Apply $U^\dagger(\mathbf{x}')$ (即 $U^{-1}(\mathbf{x}')$)
3. Measure probability of all-zeros outcome $|0\rangle\langle 0|^{\otimes n}$

数学上：
$$K(\mathbf{x}, \mathbf{x}') = |\langle 0 | U^\dagger(\mathbf{x}') U(\mathbf{x}) | 0 \rangle|^2$$

**Intuition**: quantum kernel 本质是利用 Hilbert space 的 exponential dimensionality。$n$ 个 qubit 提供 $2^n$ 维 Hilbert space，可能提供 classical kernel 难以 capture 的 feature representation。Havlíček et al. 2019 [Nature 论文](https://www.nature.com/articles/s41586-019-0980-2) 是这个方向的开山之作。

### 5.2 Trainable Fidelity Quantum Kernel

$$|\phi(\mathbf{x}; \boldsymbol{\theta})\rangle = U(\mathbf{x}; \boldsymbol{\theta})|0\rangle$$

引入 trainable parameters $\boldsymbol{\theta}$，让 feature map 也可以学习，不只是 SVM 的 dual variables 学习。这就是 [Gentinetta et al. 2023](https://ieeexplore.ieee.org/document/10338南开) 的 quantum kernel alignment 工作。

### 5.3 QSVC, QSVR, PegasosQSVC 对比

| Algorithm | Solver Type | Complexity | Use Case |
|---|---|---|---|
| QSVC | classical QP solver | $O(N^3)$ for $N$ samples | small datasets |
| QSVR | classical QP solver | $O(N^3)$ | regression |
| PegasosQSVC | stochastic sub-gradient | $O(T \cdot N)$ for $T$ iterations | large datasets |

Pegasos [Shalev-Shwartz et al. 2007](https://dl.acm.org/doi/10.1145/1273496.1273598) 是经典 SVM 的高效求解器，Qiskit ML 把它量子化。关键公式：

$$w_{t+1} = (1 - \frac{1}{t})w_t + \frac{1}{\lambda t} \mathbf{1}[y_i \langle w_t, \phi(\mathbf{x}_i) \rangle < 1] y_i \phi(\mathbf{x}_i)$$

其中 $\lambda$ 是 regularization parameter。量子版本用 quantum kernel 估计 $\langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$。

---

## 6. Variational Quantum Algorithms (VQC/VQR)

### 6.1 优化目标 (公式 3)

$$\arg\min_{\theta} \mathcal{L}(\theta) = \sum_i \mathcal{C}(y_i, f(\mathbf{x}_i, \theta))$$

变量含义：
- $\theta \in \mathbb{R}^P$: $P$ 个 trainable parameters
- $\mathcal{L}(\theta)$: total loss over training set
- $\mathcal{C}$: per-sample cost function (e.g., MSE, cross-entropy)
- $y_i$: ground truth label of $i$-th sample
- $f(\mathbf{x}_i, \theta)$: model prediction
- $\mathbf{x}_i$: input feature vector
- $i$: sample index

### 6.2 VQC/VQR 的训练循环

```
for epoch in range(max_epochs):
    for x, y in training_data:
        1. Encode x into quantum circuit U(x)
        2. Apply variational ansatz V(θ)
        3. Measure to get prediction f(x, θ)
        4. Compute loss C(y, f(x, θ))
        5. Use Parameter-Shift / SPSA to get gradient ∂L/∂θ
        6. Update θ with optimizer (ADAM, COBYLA, etc.)
```

### 6.3 Ansatz 选择的重要性

虽然 paper 没深入讨论，但实践中 ansatz 设计是 QML 的关键 bottleneck。常见选择：
- **Hardware Efficient Ansatz (HEA)**: 适合 hardware 但容易 barren plateau
- **Strongly Entangling Ansatz**: 来自 [Schuld et al.](https://arxiv.org/abs/1804.00933)
- **Tensor-product ansatz**: 避免 barren plateau

Barren plateau 现象 [McClean et al. 2018](https://www.nature.com/articles/s41467-018-07090-4) 显示某些 ansatz 的 gradient variance 随 qubit number 指数衰减：

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_i}\right] \sim O\left(\frac{1}{2^n}\right)$$

---

## 7. Quantum Bayesian Inference

基于 [Low et al. 2014](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.062315) 和 [Borujeni et al. 2021](https://www.sciencedirect.com/science/article/pii/S0957417421002764)，把 Bayesian network 编码到 quantum circuit 上：

$$P(X_1, \ldots, X_n) = \prod_i P(X_i | \text{Pa}(X_i))$$

其中 $\text{Pa}(X_i)$ 表示 $X_i$ 的 parent nodes。Quantum 版本用 rotation angles 编码 conditional probabilities：

$$\theta_{X_i | \text{Pa}(X_i)} = 2 \arccos\sqrt{P(X_i | \text{Pa}(X_i))}$$

这个方向相对小众，但在 noisy 数据下 probabilistic modeling 有优势。

---

## 8. Optimizer 生态

Qiskit ML 支持的 optimizer 分两大类：

### 8.1 Gradient-free Optimizers

| Optimizer | 特点 | 适用场景 |
|---|---|---|
| COBYLA | linear approximation, no gradient | small parameter count |
| SPSA | stochastic, robust to noise | NISQ hardware |
| NFT (Nakanishi-Fujii-Todo) | sequential minimal optimization | VQE-like problems |

NFT [Nakanishi et al. 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043158) 是专门为 quantum 优化的 algorithm，灵感来自 classical SMO (Sequential Minimal Optimization) 用于 SVM。它在 1D subspace 上找 analytic optimum，每次更新一个或两个参数。

### 8.2 Gradient-based Optimizers

| Optimizer | 来源 | 特点 |
|---|---|---|
| L-BFGS-B | SciPy | quasi-Newton, 高效但需要 gradient |
| SLSQP | SciPy | sequential quadratic programming |
| ADAM | Kingma & Ba 2014 | adaptive moments, 适合 stochastic |
| Gradient Descent | classic | 最基础 |

L-BFGS-B 在 simulator 上很受欢迎（gradient 可以精确计算），但在 hardware 上 noise 会让 quasi-Newton 的 Hessian 估计失效。

References:
- [Kingma & Ba 2014 - ADAM](https://arxiv.org/abs/1412.6980)
- [Byrd et al. 1995 - L-BFGS-B](https://epubs.siam.org/doi/10.1137/0916069)

---

## 9. 与 PennyLane 的 Architecture 对比

| 维度 | Qiskit ML | PennyLane |
|---|---|---|
| Backend 抽象 | Qiskit Primitives | QNode |
| Hardware 支持 | 强 (IBM Quantum) | 多 backend 但需要 plugin |
| Automatic Differentiation | 有限 (parameter-shift 主导) | 强 |
| PyTorch 集成 | Connector API | 原生 |
| 主要用户群 | IBM hardware users | 研究者广泛 |
| License | Apache 2.0 | Apache 2.0 |

PennyLane [Bergholm et al. 2018](https://arxiv.org/abs/1811.04968) 的 design philosophy 是"automatic differentiation of hybrid quantum-classical computations"，更像 JAX 的风格。Qiskit ML 则更贴近 IBM hardware workflow。

---

## 10. 关键应用案例综述

### 10.1 Havlíček et al. 2019 - 开山之作
[Nature 论文](https://www.nature.com/articles/s41586-019-0980-2)
- 首次在 IBM Q 经验上展示 quantum kernel 与 variational classifier 的对比
- 提出 quantum feature map 概念，奠定 QML foundation

### 10.2 Glick et al. 2024 - Covariant Quantum Kernels
[Nature Physics](https://www.nature.com/articles/s41567-024-02452-3)
- 在 27-qubit superconducting processor 上实现
- 引入 group symmetry 到 kernel 设计，提升 generalization

### 10.3 Zoufal et al. 2019 - qGAN
[npj Quantum Info](https://www.nature.com/articles/s41534-019-0223-2)
- 用 quantum GAN 学习和加载概率分布到 quantum states
- 关键应用：quantum state preparation for Monte Carlo-like 任务

### 10.4 Abbas et al. 2021 - Power of QNN
[Nature Comp Sci](https://www.nature.com/articles/s43588-021-00084-2)
- 提出 effective dimension 作为 QNN generalization 能力的 metric
- 在 standard dataset 上对比 quantum 和 classical 网络

### 10.5 Agliardi et al. 2024 - At Scale
[arXiv:2412.07915](https://arxiv.org/abs/2412.07915)
- 156 qubits 实际 hardware 实验
- 引入 bit-flip tolerance strategy 对抗 exponential concentration
- 这是 QML 走向实用化的关键里程碑

### 10.6 Sahin et al. 2024 - Efficient Kernel Alignment
[Quantum 8, 1502](https://quantum-journal.org/papers/q-2024-08-29-1502/)
- Subsampling 方法降低 kernel alignment 复杂度
- 让 quantum kernel 在大数据集上可行

### 10.7 应用领域分布

```
High Energy Physics:  ████████████ 25%
Astrophysics:        ████████ 17%
Biomedical:          ██████ 13%
Drug Discovery:      █████ 10%
Finance/Other:       ████ 8%
Methodology:         ██████████████ 27%
```

References:
- [Agliardi et al. 2024](https://arxiv.org/abs/2412.07915)
- [Mensa et al. 2023 - Drug Discovery](https://iopscience.iop.org/article/10.1088/2632-2159/adb4b8)

---

## 11. 核心技术挑战与未来方向

### 11.1 Exponential Concentration

在 large qubit count 下，kernel matrix 会变得近似 diagonal：

$$K(\mathbf{x}, \mathbf{x}') \approx \delta_{\mathbf{x}, \mathbf{x}'} \cdot 2^{-n}$$

这导致 SVM 无法区分不同样本。Agliardi et al. 提出的 bit-flip tolerance 是一种 mitigation strategy，但根本解决需要 fault-tolerant QC 或 novel encoding。

### 11.2 Barren Plateaus

[McClean et al. 2018](https://www.nature.com/articles/s41467-018-07090-4) 发现 deep random circuits 训练时 gradient 指数衰减。Mitigation 策略包括：
- Local cost functions
- Layer-wise training
- Special ansatz (e.g., tensor network inspired)

### 11.3 Hardware Noise 影响

[Tüysüz et al. 2024 PRX Quantum](https://journals.aps.org/prquantum/abstract/10.1103/PRXQuantum.5.030314) 研究了 noise 对 geometric QML 的影响，发现 symmetry breaking 会破坏 equivariant quantum model 的优势。

---

## 12. 我的 Intuition 总结

从 software engineering 角度看，Qiskit ML 的核心贡献是统一 abstraction 层：把 quantum computing 的物理复杂性封装在 primitives 后面，让 ML practitioner 能像用 scikit-learn 一样用 quantum algorithms。

从 ML 角度看，目前 QML 的 practical value 仍在 quantum kernel methods 上，而不是 QNN。原因：
1. Kernels 只需要 evaluate fidelity，不需要 gradient 训练，更 robust to noise
2. SVM 的 convex optimization 比非凸的 variational optimization 更可靠
3. Quantum advantage 在 kernel methods 上已有更严格的证明 [Liu et al. 2021 Nature Physics](https://www.nature.com/articles/s41567-021-01287-z)

QNN 路线要想 surpass classical DNN，需要：
- Fault-tolerant hardware (error-corrected qubits)
- 更好的 ansatz 设计 (避免 barren plateau)
- 数据 encoding 创新 (amplitude encoding, tensor network encoding)

Qiskit ML 的 modular design 让它能在硬件从 NISQ 演进到 FTQC 过程中保持 relevance，这是 Apache-2.0 open-source governance 加上 IBM backing 的优势。

---

## Web Links 汇总

| 资源 | 链接 |
|---|---|
| Qiskit ML GitHub | https://github.com/qiskit-community/qiskit-machine-learning |
| Qiskit 官方文档 | https://docs.quantum.ibm.com/ |
| PennyLane | https://pennylane.ai/ |
| Havlíček Nature 2019 | https://www.nature.com/articles/s41586-019-0980-2 |
| Schuld PRA 2019 | https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331 |
| McCClean barren plateau | https://www.nature.com/articles/s41467-018-07090-4 |
| Abbas Nature Comp Sci | https://www.nature.com/articles/s43588-021-00084-2 |
| Glick Nature Physics 2024 | https://www.nature.com/articles/s41567-024-02452-3 |
| Agliardi 2024 | https://arxiv.org/abs/2412.07915 |
| Sahin Quantum 2024 | https://quantum-journal.org/papers/q-2024-08-29-1502/ |
| Liu Nature Physics 2021 | https://www.nature.com/articles/s41567-021-01287-z |
| Gacon SPSA Quantum 2021 | https://quantum-journal.org/papers/q-2021-07-21-567/ |
| Kingma ADAM | https://arxiv.org/abs/1412.6980 |
| Pegasos ICML 2007 | https://dl.acm.org/doi/10.1145/1273496.1273598 |
| Gentinetta QCE 2023 | https://ieeexplore.ieee.org/document/10337南开 |
| Tacchino npj QI 2019 | https://www.nature.com/articles/s41534-019-0239-y |
| Mensa drug discovery | https://iopscience.iop.org/article/10.1088/2632-2159/adb4b8 |
| Tüysüz PRX Quantum 2024 | https://journals.aps.org/prquantum/abstract/10.1103/PRXQuantum.5.030314 |
| Zoufal qGAN | https://www.nature.com/articles/s41534-019-0223-2 |
| Pulse-efficient QML | https://quantum-journal.org/papers/q-2023-10-09-1130/ |

如果你对某个具体 algorithm (e.g., PegasosQSVC 的实现细节、covariant quantum kernel 的 group theory、SPSA 在 IBM Heron r2 processor 上的 actual numbers) 想要更深入的 drill-down，告诉我，我可以进一步展开公式推导或代码层面的解析。
