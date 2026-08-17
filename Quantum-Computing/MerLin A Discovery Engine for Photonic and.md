---
source_pdf: MerLin A Discovery Engine for Photonic and.pdf
paper_sha256: 6edb75c4c53c97ad1f05cb14323df99eff10a4e5ad8db7d3b4f2aaef04b8343e
processed_at: '2026-08-05T17:41:09-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MerLin 用人话讲

## 一句话

Quandela 这帮人做了个开源工具叫 MerLin，让你能像写 PyTorch 一样训 photonic quantum circuit，顺手把 photonic QML 圈里过去几年发的 18 篇代表 paper 全复现了一遍，目的是让这个领域终于能像 normal ML 那样搞 benchmarking。

Repo: https://github.com/merlinquantum/merlin

---

## 它到底在干啥

先说背景。photonic quantum computing 用光子跑量子计算，光子有 bosonic 的特性，进 beam splitter 之后会发生 interference，输出概率由一个叫 **matrix permanent** 的东西决定（permanent 就是行列式把所有负号变正号）。

这套数学很漂亮，但软件上一直很乱：每家研究组都用自己临时搭的代码栈，互相不通。你想把 A 组的 kernel method 跟 B 组的 reservoir computing 拼起来比较？得重写一遍。

MerLin 干的事就是把 Quandela 之前那个叫 **Perceval** 的 photonic simulator（https://perceval.quandela.net/）再包一层，让它原生支持 PyTorch autograd。意思是：

- 你定义一个 photonic circuit（就是一串 beam splitter + phase shifter）
- 光子数 $n$ 和 mode 数 $m$ 定下来
- classical data $x$ 通过 phase shifter encode 进去
- 输出态用 **SLOS** 算法 exact 算出来（不是采样，是把整个 quantum state vector 算出来）
- 然后 `loss.backward()` 就能跑，`optimizer.step()` 就能训

这就把 photonic QML 从"做个实验发个 paper"变成"训个模型做个 ablation"。

---

## 为什么 photonic 特别（这部分最值得 build intuition）

### Fourier 直觉

Schuld 2021（https://arxiv.org/abs/2008.08605）发现一件很妙的事：你用 angle encoding 把数据 $x$ 塞进 VQC（就是 $S(x) = e^{ix\hat{n}}$ 这种 phase shifter），整个 VQC 的输出 observable expectation **天然就是一个 truncated Fourier series**：

$$f(x) = \sum_\omega c_\omega e^{i\omega x}$$

- $x$ 是 classical input
- $\omega$ 是频率，集合 $\Omega$ 由电路结构决定
- $c_\omega$ 是系数，由 trainable 参数决定

Gate-based VQC 里，你想让 Fourier series 能拟合更高频率的函数，得加 layer —— 跟 classical NN 加深度才能学到高频一样。

但 photonic 平台有个 killer feature：**你多发一个光子，frequency spectrum 就线性变大**，不用加 layer 不用加深 circuit。Gan 2022（https://link.springer.com/article/10.1140/epjqt/s40507-022-00121-w）把这个 generalization 到 photonic 上了，MerLin Figure 2b 就是复现这个 —— 3 modes + 3 photons 拟合 degree-3 Fourier series，完美。

直觉就是：**photon number 在 photonic VQC 里扮演了 "frequency budget" 的角色**，跟 gate-based 里的 depth 是两个不同的 scaling axis。这是 photonic 的本质 affordance。

### Fock space 维度

$n$ 个光子分到 $m$ 个 mode 上，能形成多少种 occupation pattern？答案是 stars-and-bars 组合数：

$$N_{\text{Fock}} = \binom{m+n-1}{n}$$

这就是你的 feature space 维度。所以 photonic QML 的 expressivity 跟 $n$ 和 $m$ 都挂钩，跟 gate-based 里 $2^{\text{qubits}}$ 完全是两套 scaling 逻辑。

---

## SLOS 是什么

SLOS = Strong Linear Optical Simulation，论文 https://arxiv.org/abs/2301.09594。

"Strong" 是相对于 "weak" 来说的：
- weak simulation = 你给我电路，我给你采样输出（像 boson sampling 那样）
- strong simulation = 你给我电路，我把整个 quantum state vector 算出来

weak 的问题在于训练时 gradient 是从采样估的，shot noise 极大。strong 直接给你 exact amplitude，autodiff 能给你 exact gradient。这是 MerLin 能进 PyTorch 的根本。

SLOS 的 trick 是把 circuit 拆成 layer-by-layer 的 transition graph，1-photon → 2-photon → ... → n-photon 递推。**这个 graph 的结构只依赖 input state 和 circuit topology，跟具体 unitary entries 无关**，所以 MerLin 初始化时预编译一次 sparse graph，训练时每个 forward 只重算 unitary-dependent 的系数。时间空间复杂度都是 $O(n \binom{m+n-1}{n})$。

实际上能跑 $n \lesssim 20$ photons，再大内存爆了。

---

## QuantumLayer 的设计

四个正交的轴，工程上很干净：

1. **Measurement strategy**: 你要输出什么 —— 全概率分布？per-mode 光子数期望？还是测量前的复数 amplitude？
2. **Computation space**: 用 full Fock space 还是某种 encoded subspace（dual-rail、QLOQ）
3. **Detector model**: PNR（能数有几个光子）还是 threshold（只能测有无光子，硬件常用）
4. **Data encoding**: angle encoding 还是 amplitude encoding

这四件事解耦开，做 ablation 就很舒服。

---

## 18 篇复现说了啥

我挑几条最有信号的：

### 表达力
Gan [14] 复现成功：photon 数加 → expressivity 加。证实 Fourier spectrum intuition。

### Photonic kernel
Yin et al. [15]（https://www.nature.com/articles/s41566-025-01677-y）：accuracy 跟 training set size 和 **geometric difference** 一起涨。geometric difference 是经典 kernel alignment theory 的概念（Huang 2021, https://arxiv.org/abs/2105.02276）—— photonic kernel 的 advantage 其实接到了经典 kernel 理论上，未必是 quantum speedup，而是 different inductive bias。

### Photonic QCNN
Monbroussou [19]：MerLin 通过 hyperparameter sweep 把 BAS 数据集从 92.7% 提到 98.2%，MNIST 0-vs-1 从 93.1% 提到 98.8%。**比原 paper 还高**。这展示了 MerLin "建立 stronger baseline" 的方法论 —— 很多人 paper 报的数其实没好好调参。

### Reservoir computing
Sakurai [17]：accuracy 跟 Fock space size scale。这是 "Fock space 维度 = feature 维度" 直觉的实证。

### 量子 LLM fine-tuning（重要 negative result）
Kim [39] 原 paper 报 quantum 模型有 3.14% accuracy gain。MerLin 复现发现 quantum 和 classical 都到 ~89%，**没有 clear separation**。这跟 Bowles [21]（https://arxiv.org/abs/2403.07059）的警告一致 —— binary classification 这种 task 太简单，根本看不出 quantum utility。这个 negative result 比 100 个 positive toy result 有价值。

### Adversarial robustness
Lu [40]：**amplitude encoding 对扰动极脆弱**，angle encoding 显著 robust。MNIST 上 98% clean accuracy，但 BIM attack 下只剩 15%，adversarial training 后恢复到 95%。所以 encoding strategy 在 adversarial 场景是 critical design choice。

### Photonic QGAN
Sedrakyan [37]：MerLin 把训练时间砍到 1/15，SSIM 指标一样。原因是原 paper 用 SPSA（zeroth-order optimizer，适合硬件），MerLin 直接用 PyTorch SGD + autodiff 必然快。

### HQNN parameter efficiency
Kashif [38]：VQC 在 5-60 feature dim 下用比 classical NN 少的参数达 ≥90% accuracy on noisy spiral。但 [32] 里又说 classical 在 parameter-efficiency 上 sometimes 仍胜。Mixed signal —— 没有压倒性结论。

---

## 我对这 paper 的吐槽

**Scalability 是最大短板**。$n \lesssim 20$ photons 对 toy dataset 够，但真要 competitive benchmark 还差太远。Moon / BAS / MNIST 0-vs-1 都是 toy。

**Hardware gradient 不清楚**。MerlinProcessor 接 Quandela QPU（Belenos / Ascella, https://cloud.quandela.com/），但硬件只能采样不能 strong simulate。那训练时 gradient 怎么算？REINFORCE？parameter-shift？paper 没讲透。

**18 篇里 negative result 太少**。只 [39] 一篇 Q-LLM 是 negative。应该 systematic 把 claim 有 advantage 的 paper 都试 reproduce，看哪些垮了 —— 这才是 benchmark-driven 该有的样子。

**没跟 DeepQuantum 直接比**。DeepQuantum（https://arxiv.org/abs/2512.18995）自称比 PennyLane 快一个量级，MerLin 没做 head-to-head，weak。

**Tensor network cross-pollination 没提**。SLOS 的 layer-by-layer construction 跟 MPS / TEBD 的 sweep 结构上很像，理论上可以用 tensor network compression 把 SLOS push 到更大 $n$。Paper 完全没这个方向，很可惜。

---

## 给你的 intuition

把 MerLin 想成 "photonic QML 的 PyTorch + OpenML 合体"：你定义 photonic circuit 当作一个 layer，光子进去、概率分布出来，autodiff 跑通，optimizer 调参，跟训 ResNet 一样。**photon number 在这里替代了 circuit depth 成为 frequency-spectrum 的 degree of freedom** —— 这是 photonic 相对 gate-based 的核心 affordance。

但真问题在于：这套 framework 现在能问的问题规模还太小。要让 photonic QML 从 "I can fit Moon dataset" 变成 "I can beat some classical baseline on a real task"，得先把 simulator scalability 推到 $n \sim 50$，或者 hardware gradient estimation 这条路走通。MerLin 把 infrastructure 搭好了，但 killer app 还没出现。

主要 link 汇总：
- https://github.com/merlinquantum/merlin
- https://github.com/merlinquantum/reproducedpapers
- https://perceval.quandela.net/
- https://arxiv.org/abs/2301.09594 (SLOS)
- https://arxiv.org/abs/2008.08605 (Schuld Fourier)
- https://arxiv.org/abs/2403.07059 (Bowles benchmarking)
- https://www.nature.com/articles/s41566-025-01677-y (photonic kernel)

---

# MerLin: Photonic QML Discovery Engine — 技术深读

## 1. Paper 一句话定位

Quandela 团队开源了 **MerLin**，这是一个把 **photonic linear-optical circuits** 当作 native primitive 嵌入 PyTorch / scikit-learn 的 discovery engine，并且配套复现了 **18 篇** photonic / hybrid QML 的代表性工作作为可重用 baseline。核心 motivation 是：photonic QML 的 literature 长期碎片化、缺少 reproducible benchmarking、各家 ad-hoc 软件栈互不通约，MerLin 想成为 photonic 领域的 "OpenML + PyTorch + hardware connector"。

- Repo: https://github.com/merlinquantum/merlin
- Reproduced papers: https://github.com/merlinquantum/reproducedpapers
- Perceval (底层 simulator): https://perceval.quandela.net/
- SLOS paper (Heurtel et al. 2023): https://arxiv.org/abs/2301.09594

---

## 2. Linear Optics 的数学 ground truth

要把 photonic QML 的 intuition 建起来，先得把 Fock space 这套语言钉死。

### 2.1 Fock state 与 unitary

- **n photons**, **m modes**。输入 Fock state
  $$|\mathbf{s}\rangle = |s_1, \ldots, s_m\rangle, \quad s_i \in \{0,1,\ldots,n\}, \quad \sum_{i=1}^m s_i = n$$
  其中 $s_i$ 是第 $i$ 个 optical mode 上 photon 占据数。这是bosonic 的 indistinguishable-particle 描述，与 qubit 的 $|0\rangle/|1\rangle$ register 完全不同。
- 整个 interferometer 由 **beam splitters** 和 **phase shifters** 构成，等价于一个 $m \times m$ 的 **unitary matrix** $U$。这是 Reck / Clements decomposition 的核心：任何 $U(m)$ 都可以分解成 $O(m^2)$ 个 beam splitter + phase shifter 的层叠。
- Lossless evolution 下 photon 总数守恒，输出态仍是 $\sum_i s'_i = n$ 的 Fock superposition。

### 2.2 Fock space 维度

$$N_{\text{Fock}} = \binom{m+n-1}{n}$$

这是 **stars-and-bars** 组合数：把 $n$ 个 indistinguishable photon 分到 $m$ 个 distinguishable mode 上。这个量级很关键 — 它决定了 strong simulation 的成本，也决定了 model 的 expressivity。

直觉上：
- 固定 $n$，$m$ 增大 → $N_{\text{Fock}}$ polynomial 增长 ($\sim m^n/n!$)
- 固定 $m$，$n$ 增大 → $N_{\text{Fock}}$ polynomial 增长 ($\sim n^{m-1}/(m-1)!$)
- 但 matrix permanent 的计算是 #P-hard，经典 #P-hard 区域是 $n \sim m$ 时

### 2.3 输出概率 = matrix permanent

输出构型 $\mathbf{s}'$ 的概率正比于 $U$ 的某个 submatrix 的 **permanent**（permanent 是 determinant 把所有 sign 换成正号）。permanent 的 #P-hardness 是 boson sampling 量子优势的 complexity-theoretic 基础 (Aaronson-Arkhipov 2011, https://arxiv.org/abs/1011.3245)。

---

## 3. SLOS — Strong Linear Optical Simulation

MerLin 的核心 numerical engine 是 **SLOS** (Strong Linear Optical Simulation, Heurtel et al. 2023)。这里的 "strong" vs "weak" simulation 是量子计算复杂性里的术语：
- **Weak simulation**: 抽样输出分布 (像 boson sampling 那样)
- **Strong simulation**: 计算 *整个* 量子态向量 / 全部 output probabilities

SLOS 的 complexity:
- Time: $O\!\left(n \binom{m+n-1}{n}\right)$
- Space: $O\!\left(n \binom{m+n-1}{n}\right)$

这比朴素的 "对每个输出构型各算一次 permanent" ($O(N_{\text{Fock}} \cdot \text{poly}(n))$ per permanent) 要快得多，因为 SLOS 复用中间结果。

### 3.1 SLOS 的构造性算法 intuition

SLOS 把 circuit 拆成 layer-by-layer 的 transition graph：
1. 从 1-photon intermediate state 开始
2. 递推构造 $k$-photon state, $k = 2, \ldots, n$
3. 每层的 $k \to k{+}1$ transition rule **只依赖 circuit topology + input state**，**与 $U$ 的具体 entries 无关**
4. 因此 MerLin 在 init 时 **预编译 sparse transition graph** 一次（用 sparse index mapping），训练时每个 forward pass 只重算 unitary-dependent coefficients

这是把 "graph structure reuse" 和 "parameter recompute" 解耦的经典 trick — 类似于 PyTorch 中把 static graph 编译成 TorchScript，只让 dynamic weights 在 forward 时流动。Practical 上能模拟 $n \lesssim 20$ photons。

### 3.2 为什么 SLOS 对 QML 重要

QML 训练需要 **gradients**。Weak simulation (采样) 给出的 gradient 估计噪声极大 (shot noise 在 $\sim 1/\sqrt{N_{\text{shots}}}$)。Strong simulation 直接给出 exact amplitudes → **exact analytic gradient via autodiff**。这是 MerLin 能 plug 进 PyTorch `loss.backward()` 的根本原因。

参考 PyTorch autograd: https://pytorch.org/docs/stable/autograd.html

---

## 4. QuantumLayer 抽象

`QuantumLayer` 是 `torch.nn.Module`，四个正交 concept：

| Concept | 作用 | Examples |
|---|---|---|
| **Measurement strategy** | 决定 layer 输出形式 | `probs` (full probability dist), per-mode photon-number expectation, complex amplitudes pre-measurement |
| **Computation space** | Hilbert subspace 选择 | `FOCK` (full), dual-rail, QLOQ qubit encoding [25] |
| **Detector model** | photonic state → classical outcome | PNR (photon-number-resolving), threshold (有/无 photon) |
| **Data encoding** | classical $x$ → quantum state | angle encoding, amplitude encoding |

这四个 axis 的解耦是工程上很漂亮的设计 — 让 circuit evolution / state representation / measurement semantics 各自独立 ablation。

### 4.1 Code Block 1 解析

```python
builder = ML.CircuitBuilder(n_modes=3)
builder.add_entangling_layer(trainable=True, name="W1")  # U(3) 通用 interferometer
builder.add_angle_encoding(modes=[0, 1], name="x_")      # 数据进来当 phase
builder.add_entangling_layer(trainable=True, name="W2")  # 第二个 U(3)
layer = ML.QuantumLayer(builder=builder, input_state=[1,1,1], ...)
```

构造的就是 $U(x,\boldsymbol{\theta}) = W^{(2)}(\boldsymbol{\theta}_2) S(x) W^{(1)}(\boldsymbol{\theta}_1)$。后面 PyTorch loop 完全 standard — `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` — 这是 MerLin 的 sell point：photonic 进来，PyTorch 出去。

---

## 5. Data Encoding — Photonic QML 的 bottleneck

### 5.1 Angle encoding (Schuld-style Fourier decomposition)

Schuld et al. 2021 (https://arxiv.org/abs/2008.08605) 证明：VQC 在 angle encoding 下自然 realize 一个 **truncated Fourier series**：
$$f^{(n)}(x, \boldsymbol{\theta}, \boldsymbol{\lambda}) = \sum_{\omega \in \Omega_n} c_\omega(\boldsymbol{\theta}, \boldsymbol{\lambda})\, e^{i\omega x}$$

变量含义：
- $x$: classical scalar input，被 encode 成 phase shifter $S(x) = e^{ix\hat{n}_1}$
- $\boldsymbol{\theta}$: trainable interferometer 参数（beam splitter reflectivities + phase shifter phases）
- $\boldsymbol{\lambda}$: measurement observable 参数
- $\Omega_n$: accessible frequency spectrum — **photonic 平台的 killer feature**
- $c_\omega(\boldsymbol{\theta}, \boldsymbol{\lambda})$: 由 trainable circuit + observable 决定的 Fourier 系数

**关键 intuition**: $\Omega_n$ 的大小 **线性随 photon number $n$ 增长**，*不需要* extra encoding gates 或 deeper circuit。这是 photonic 相对 gate-based VQC 的本质优势 — 在 gate-based 里要扩展 frequency spectrum 得加 layer；在 photonic 里多发一个 photon 就行。

具体地，对 $n$-photon input state $|1,1,\ldots,1\rangle$ 在第 1 mode 上加 $S(x) = e^{ix\hat{n}_1}$，输出的某种 observable expectation 展开成 Fourier series 时，频率集合 $\Omega_n \subseteq \{-n, -n{+}1, \ldots, n{-}1, n\}$，所以 frequency count 是 $2n{+}1$。

Gan et al. 2022 (https://link.springer.com/article/10.1140/epjqt/s40507-022-00121-w) 在 photonic 上 generalization 了这件事，MerLin 用 Figure 2b 复现：3 modes + up to 3 photons 拟合 degree-3 Fourier series。Figure 2c 在 Moon dataset 上 binary classification 验证 even 这种简单 3-mode VQC 已经有足够 nonlinearity。

### 5.2 Amplitude encoding

$$|\mathbf{x}\rangle = \sum_{i=0}^{N-1} x_i |i\rangle, \quad \|\mathbf{x}\|^2 = 1$$

变量：$\mathbf{x} \in \mathbb{C}^N$ 是 classical data vector（要先 normalize），$\{|i\rangle\}$ 是 Fock basis。这是 *directly* 把 vector 写进 quantum state 的 amplitude，不是经过参数化 circuit。

适用场景：上游已经有 photonic circuit 产出某个 StateVector，把它接下去。Raw classical features 走这条路通常不太适合（因为 normalization 破坏 scale information + adversarial robustness 差，见 §7）。

---

## 6. Hardware-aware design 的几个 key abstraction

### 6.1 MerlinProcessor

```python
proc = ML.MerlinProcessor(rp, microbatch_size=32, timeout=3600.0)
y = proc.forward(layer, X, nsample=5000)
```

封装了：
- **latency** (cloud QPU call)
- **shot-based sampling** (硬件不能 strong simulate，只能 sample)
- **limited parallelism** (QPU 一次能跑的 circuit 数有限)

参考 Quandela 的 cloud QPU (Belenos / Ascella): https://cloud.quandela.com/

### 6.2 Quantum Bridge — qubit ↔ photon

```python
bridge = ML.QuantumBridge(n_photons=2, n_modes=4, qubit_groups=[1,1],
                          computation_space=ML.ComputationSpace.UNBUNCHED)
model = torch.nn.Sequential(qubit_circuit, bridge, merlin_layer)
```

Qubit 平台是 $2^n$ dim computational basis，photonic 是 Fock space。Bridge 提供不同iable interface。**Dual-rail encoding** 是最简单 mapping：1 qubit = 2 modes + 1 photon，$|0\rangle \equiv |1,0\rangle$, $|1\rangle \equiv |0,1\rangle$。QLOQ scheme [25] (https://arxiv.org/abs/2407.18006) 是更一般的 qubit-qudit mapping，能减少 photonic resource。

### 6.3 Quantum memristor (neuromorphic)

Photonic memristor = Mach-Zehnder interferometer + feedback loop，internal phase 根据 detection statistics 更新 → 引入 memory + nonlinearity，对 time-series prediction 有用。这是 MerLin 模块化设计的 demo：emerging paradigm 可以 plug-in。

参考 quantum memristor (https://arxiv.org/abs/2504.xxxx) — paper ref [27]。

---

## 7. 18 篇复现的关键 takeaways (Table I 拆解)

我挑几条最有 signal 的：

### 7.1 Expressivity (Gan et al. [14])
**Confirmed**: photon number $n$ 增加 → expressivity 增加。这是 §5.1 Fourier spectrum intuition 的实证。

### 7.2 Photonic kernel (Yin et al. [15], https://www.nature.com/articles/s41566-025-01677-y)
**Confirmed**: accuracy 随 training-set size 和 geometric difference 提升。Geometric difference (Huang et al. 2021, https://arxiv.org/abs/2105.02276) 是衡量两个 kernel 在 RKHS 中差异的指标 — 这条 link 很重要，因为它把 photonic kernel 的 advantage story 接到经典 kernel alignment 理论上。

### 7.3 Photonic QCNN with adaptive state injection (Monbroussou et al. [19])
**Improved**: BAS dataset $92.7\pm2.1\% \to 98.2\pm2.2\%$；MNIST 0-vs-1 $93.1\pm3.6\% \to 98.8\pm1.0\%$。Adaptive state injection 是 pooling 的 photonic 实现 — 保持 photon number 同时 reduce effective Fock-space dimension。MerLin 用 density-matrix backend 实现，加上 hyperparameter sweep 后超越原 paper。这是 MerLin "建立 stronger baseline" 哲学的典范。

### 7.4 Reservoir computing (Sakurai et al. [17], https://opg.optica.org/quantum/abstract.cfm?uri=quantum-3-3-238)
**Confirmed**: accuracy 随 Fock space size scale。Reservoir = fixed untrained photonic circuit + linear probe。验证了 "Fock space 维度 = feature space 维度" 的直觉。

### 7.5 Quantum LLM fine-tuning (Kim et al. [39])
**Failed to reproduce**: 原 paper 报 3.14% accuracy gain，MerLin 复现到 ~89% accuracy 但 quantum 和 classical 没有 clear separation。这是 **negative result** 很重要 — 跟 Bowles et al. [21] (https://arxiv.org/abs/2403.07059) 的 "binary classification 太简单不足以证 quantum utility" 警告一致。

### 7.6 Adversarial robustness (Lu et al. [40], https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033212)
**Confirmed**: amplitude encoding 对 $\epsilon$ 扰动极脆弱，angle encoding 显著 robust。MNIST 上 ~98% clean accuracy, 15% BIM adversarial accuracy at $\epsilon=0.1$, 95% after adversarial training。**Encoding strategy 是 adversarial setting 的 critical design choice** — 这条对安全相关的 hybrid QML 部署很关键。

### 7.7 Photonic QGAN (Sedrakyan et al. [37])
**Speed**: MerLin 把训练时间 reduce 到 1/15，SSIM 相同。Switch from SPSA to PyTorch SGD 是关键 — SPSA 是 parameter-shift-friendly 的 zeroth-order optimizer，但在 strong simulation 下用 autodiff + Adam / SGD 必然快得多。

### 7.8 HQNN parameter efficiency (Kashif et al. [38], https://arxiv.org/abs/2412.04991)
**Confirmed**: VQC 在 5-60 feature dim 下用 fewer parameters than classical NN 达到 ≥90% accuracy on noisy spiral。但 §7.4 也说 classical 在 parameter-efficiency 上 sometimes 仍胜出。Mixed signal。

---

## 8. MerLin 在 software landscape 中的位置

| Framework | Paradigm | Differentiable | Hardware |
|---|---|---|---|
| Qiskit [5] | gate-based superconducting | partial (via Qiskit-Torch-Module) | IBM Q |
| Cirq [6] | gate-based | partial | Google |
| Pulser [7] | neutral-atom | partial | Pasqal |
| PennyLane [10] | multi-backend | yes | many |
| Strawberry Fields [8] | photonic CV | yes | Xanadu |
| Piquasso [9] | photonic CV+DV | yes | — |
| Perceval [2] | photonic DV | partial | Quandela |
| DeepQuantum [1] | qubit + photonic + MBQC | yes (PyTorch) | — |
| **MerLin** | **photonic DV native + SLOS** | **yes (PyTorch native)** | **Quandela QPU** |

MerLin 的 niche：**SLOS 优化 + PyTorch-native autodiff + hardware-aware co-design**。DeepQuantum 是最近的 concurrent work，也做 PyTorch + photonic，但 MerLin 的 strong benchmarking focus (18 篇复现) 是 unique selling point。

---

## 9. broader intuition / 联想

### 9.1 与 classical DL 的对应

- **Fock space** ≈ exponentially-large feature space，photonic circuit 是 implicit feature map $\phi(x)$
- **Beam splitter + phase shifter** ≈ trainable linear layer (但 unitary constrained)
- **Measurement** ≈ nonlinearity / readout head
- **Photon number $n$** ≈ "circuit depth" in frequency domain (Schuld Fourier view)
- **Reservoir computing** ≈ 不 train backbone，train linear probe — 完全是 classical reservoir computing (Jaeger 2001, Maass 2002) 的 photonic 版

### 9.2 与 neural tangent kernel / double descent 的潜在 link

Schuld 的 Fourier decomposition 和 NTK 的 kernel regime 有结构性相似 — 都是 "linear in feature, nonlinear in input"。Photonic kernel 的 geometric difference analysis [15] 实际上接的就是 kernel alignment theory。这暗示 photonic QML 的 advantage 未必在 "quantum speedup"，可能在 "different inductive bias"。这个 framing 在 Bowles et al. [21] 里也强调。

### 9.3 与 TensorNetwork / MPS 的 link

Strong simulation of linear optics 实际上等价于一个特殊结构的 tensor network contraction。SLOS 的 layer-by-layer construction 跟 MPS / TEBD 的 sweep 类似。这 link 值得探索 — 可能用 tensor network compression 把 SLOS push 到更大 $n$。

参考 tensor network methods: https://tensornetwork.org/

### 9.4 与 equivariant ML 的 link

Linear optics 的 $U(m)$ 是 $SU(m)$ 子群，对应 a specific symmetry。如果 data 本身有 $SU(m)$ 对称性 (e.g. 某些高能物理问题)，photonic VQC 是天然 equivariant model。这个方向 MerLin 的 `ComputationSpace` + `MeasurementStrategy` 抽象很适合探索。

参考 Equivariant ML (Cohen et al.): https://arxiv.org/abs/1602.07576

### 9.5 与 photonic LLM 的 link

最近有不少 photonic accelerator for Transformer / LLM 的工作 (e.g. Lightmatter, Luminous Computing)。MerLin 的 [39] 复现 (Quantum LLM fine-tuning) 虽然没 reproduce 出 advantage，但 *infrastructure* 已经存在 — 等真正有 photonic QPU 规模起来，可以做 photonic LoRA / photonic attention。这是 long shot 但 high upside 方向。

---

## 10. Paper 的 limitation / 我的 critique

1. **Strong simulation 只到 $n \lesssim 20$** — 对真 QML application 还太小。Figure 2c Moon dataset 是 toy。需要 push 到 $n \sim 50$ 才有 industrial signal。
2. **18 篇复现里很多是 toy dataset** (Moon, BAS, MNIST 0-vs-1)。真正 competitive benchmark 缺。
3. **Hardware execution 仍是 sample-based**，gradient 估计 noise 没有充分讨论。MerlinProcessor 的 `nsample=5000` 在训练 loop 里怎么跟 `loss.backward()` 接？是 REINFORCE-style 还是 parameter-shift？Paper 没说清。
4. **Negative result ([39] Q-LLM) 只复现一篇** — 应该 batch 多几篇 negative result 来建立 "QML advantage 的边界在哪" 的 map。
5. **没有跟 DeepQuantum 直接 benchmarking**。DeepQuantum 自称比 PennyLane 快一个量级，MerLin 应该 head-to-head。

---

## 11. 给 Karpathy 的一句 intuition summary

MerLin 把 photonic linear-optical circuit 的 **Fock-space strong simulation** 编译进 PyTorch autograd graph，使得 photonic QML 可以像 classical DL 一样 `loss.backward()` + `optimizer.step()`。**Photon number $n$ 替代了 circuit depth 成为 frequency-spectrum 的 degree of freedom**，这是 photonic 相对 gate-based VQC 的本质 affordance。18 篇复现 + hardware connector 让它成为 photonic QML 第一个真正 benchmark-driven 的 discovery engine，但 current scalability ($n \lesssim 20$) 限制了现在能问的问题的规模。

主要 reference links:
- MerLin repo: https://github.com/merlinquantum/merlin
- Reproductions: https://github.com/merlinquantum/reproducedpapers
- Perceval: https://perceval.quandela.net/
- SLOS: https://arxiv.org/abs/2301.09594
- Schuld Fourier: https://arxiv.org/abs/2008.08605
- Gan Fock expressivity: https://link.springer.com/article/10.1140/epjqt/s40507-022-00121-w
- Bowles benchmarking: https://arxiv.org/abs/2403.07059
- Yin photonic kernel: https://www.nature.com/articles/s41566-025-01677-y
- QLOQ: https://arxiv.org/abs/2407.18006
- Photonic QCNN: ref [19] / [43] (https://iopscience.iop.org/article/10.1088/2058-9565/ad8c5d)
