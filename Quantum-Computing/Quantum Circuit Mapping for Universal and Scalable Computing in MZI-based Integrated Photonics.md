---
source_pdf: Quantum Circuit Mapping for Universal and Scalable Computing in MZI-based
  Integrated Photonics.pdf
paper_sha256: 5b084a6cf45c39355d186d7628799555eec4fd7bb8a05948ded7b149754c8601
processed_at: '2026-08-06T07:51:43-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话概括

**在光子芯片上做量子计算，最大的麻烦不是物理原理，而是"布线问题"——这篇 paper 提出一套规则化的波导排布方案，让量子门能像乐高积木一样拼起来。**

---

## 1. 故事背景：为什么要在光子上做量子计算

Quantum computing 有很多平台候选——superconducting qubit（IBM, Google）、trapped ion（IonQ）、photon 等。每种平台都在满足 DiVincenzo criteria（[参考](https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E)）上有 trade-off。

**Photon-based QC 的卖点**：
- Room temperature 工作，不需要 dilution refrigerator（superconducting 要 mK 温度）
- Decoherence 极小，photon 几乎不跟环境耦合
- Integrated photonics 工艺成熟（silicon photonics 已经在 datacenter 用了）
- 玻色子天然适合 quantum interference

**麻烦**：photon 之间几乎不相互作用，而 two-qubit gate 需要相互作用。Superconducting qubit 用 capacitive coupling，trapped ion 用 Coulomb interaction，photon 用什么呢？

---

## 2. Path Encoding：把 qubit 编码在"哪条路"上

最直观的 photonic qubit 编码：一个 photon 在两条 waveguide 里走，左波导代表 $|0\rangle$，右波导代表 $|1\rangle$。

$$|0\rangle \equiv \hat{a}_{w_0}^{\dagger}|\Omega\rangle, \quad |1\rangle \equiv \hat{a}_{w_1}^{\dagger}|\Omega\rangle$$

**人话**：qubit 的值 = photon 在哪根波导里。这就是 **dual-rail encoding**。

- $\hat{a}_{w_0}^{\dagger}$：在波导 $w_0$ 上"放"一个 photon 的算符
- $|\Omega\rangle$：vacuum state（所有波导都空着）
- 下标 $w_0, w_1$：波导编号

**为什么这样编码好**？因为 photon loss 直接表现为"某个 qubit 的两条波导都没 photon"，detector 一看就知道——这是 **heralded error**，可以 post-select 掉。Superconducting qubit 的 error 是 silent 的，需要 quantum error correction 来 detect。

**代价**：一个 qubit 要占 2 根波导，n 个 qubit 要 2n 根。

---

## 3. MZI：万能单 qubit 门

Mach-Zehnder interferometer（MZI）是 photonic 电路的基本单元，结构是 **两个 beam splitter + 中间的 phase shifter**：

```
波导1 ─┬──BS──┬──PS──┬──BS──┬──输出1
        │      │      │      │
波导2 ─┴──BS──┴──PS──┴──BS──┴──输出2
```

数学上，MZI 等价于一个 **tunable beam splitter**：

$$U_{\mathrm{MZI}} = ie^{i(\theta_1+\theta_2+2\phi_2)/2} \begin{pmatrix} e^{i(\phi_1-\phi_2)}\sin\frac{\theta_1-\theta_2}{2} & \cos\frac{\theta_1-\theta_2}{2} \\ e^{i(\phi_1-\phi_2)}\cos\frac{\theta_1-\theta_2}{2} & -\sin\frac{\theta_1-\theta_2}{2} \end{pmatrix}$$

**变量解释**：
- $\theta_1, \theta_2$：MZI 内部两臂上的 phase shifter（可以理解为加热波导改变折射率）
- $\phi_1, \phi_2$：输入端的 phase shifter
- 只有 **差值** $\theta_1-\theta_2$ 和 $\phi_1-\phi_2$ 重要，因为 global phase 不可观测

**Intuition**：调两个 phase，就能让 MZI 在 0:100 到 50:50 之间任意 split。设成特定值就能实现 Hadamard、Pauli-X、Pauli-Y、Pauli-Z、T gate 等所有单 qubit 门。Table 1 给出了对应关系，比如 Hadamard 就是 $(\theta_1-\theta_2, \phi_1-\phi_2) = (\pi/2, 0)$。

**所以单 qubit 门在 photonic 上是 "免费" 的**——随便调 phase 就行，确定性的，100% success。

---

## 4. 两 qubit 门的噩梦

这才是真正的难点。CZ gate 需要的作用是：

$$|00\rangle \to |00\rangle, \quad |01\rangle \to |01\rangle, \quad |10\rangle \to |10\rangle, \quad |11\rangle \to -|11\rangle$$

只在两个 qubit 都是 $|1\rangle$ 时加个负号。这需要 **两个 photon 之间发生某种相互作用**。

### 4.1 为什么线性光学不够

Appendix A 证明：4 根波导（2 个 dual-rail）的 MZI 网络，无论怎么调 phase，都实现不了 CZ。

**人话**：线性光学只能对单个 photon 做 mode 之间的 linear transformation。两个 photon 在线性网络里只通过 **Hong-Ou-Mandel interference**（[参考](https://doi.org/10.1103/PhysRevLett.59.2044)）相互作用——两个 photon 同时到达 50:50 BS，会 "bunch" 到同一输出。这种 interaction 太弱，不足以实现 controlled-phase。

### 4.2 Post-Selection 救场

Hofmann-Takeuchi 方案（[参考](https://doi.org/10.1103/PhysRevA.66.024308)）的 trick：**加 2 根 auxiliary 波导，用 1/3 透射率的 BS，然后 post-select 只看特定输出**。

1/3-BS 的矩阵：

$$R_{1/3} = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & \sqrt{2} \\ -\sqrt{2} & -1 \end{pmatrix}$$

**为什么是 1/3**？经过优化推导，这个透射率让 6 波导网络能在 4 维 computational subspace 上实现 CZ，同时最大化 success probability。

完整的 6×6 CZ 矩阵（Equation 14）：

$$\overline{\mathbf{CZ}}_{\mathrm{ps}} = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & \sqrt{2} & 0 & 0 & 0 & 0 \\ \sqrt{2} & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & -1 & -\sqrt{2} & 0 & 0 \\ 0 & 0 & \sqrt{2} & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 & -1 & -\sqrt{2} \\ 0 & 0 & 0 & 0 & \sqrt{2} & -1 \end{pmatrix}$$

**矩阵结构**：3 个 2×2 block-diagonal，每个 block 是 1/3-BS。注意上划线 $\overline{\cdot}$ 表示 **non-regular labeling**——波导顺序是 $(w_A^0, w_0^0, w_1^0, w_0^1, w_1^1, w_A^1)$，auxiliary 在最外面。

### 4.3 Post-Selection 物理含义

输出 state 是：

$$|\Psi\rangle_{\mathrm{out}} = -\frac{1}{3}(\alpha_0\alpha_1|00\rangle + \alpha_0\beta_1|01\rangle + \beta_0\alpha_1|10\rangle - \beta_0\beta_1|11\rangle) + \text{垃圾项}$$

**垃圾项**包括：
- 1 个 photon 在 auxiliary + 1 个在 dual-rail
- 2 个 photon 在 auxiliary
- 2 个 photon 在同一 dual-rail（qubit 结构被破坏）

Post-selection 就是 **在 detector 上只保留每个 dual-rail 各有 1 个 photon 的事件**。Success probability = $|1/3|^2 = 1/9$。

**人话**：用 9 次实验，平均只有 1 次成功。但这一次成功的输出确实是正确的 CZ 操作。

---

## 5. 核心 Problem：Non-Regular Structure

CZ 的波导排布是 $(w_A^0, w_0^0, w_1^0, w_0^1, w_1^1, w_A^1)$——auxiliary 波导在最外面，**夹在两个 qubit 之间**。

**问题**：
1. 这个结构不 regular——每个 qubit 的 auxiliary 位置不固定
2. **不能 cascade**：做完一个 CZ，photon 可能在 auxiliary 上，下一个 CZ 就乱了
3. 只能做 nearest-neighbor qubit 之间的 CZ，non-neighbor qubit 没法连

这就是 scalability 的障碍。Large-scale QC 需要任意两个 qubit 都能交互，需要能把 gate 串起来用。

---

## 6. 论文的三个核心 Trick

### Trick 1：Three-Waveguide Qubit（Regular Labeling）

把每个 qubit 从 2 波导扩展到 3 波导：$(w_A, w_0, w_1)$。

**Regular 排布**：n 个 qubit 就是 3n 根波导，按 $(w_A^0, w_0^0, w_1^0, w_A^1, w_0^1, w_1^1, \dots)$ 顺序排列。每个 qubit 的 auxiliary 永远在固定位置（最上面那根）。

**为什么这样好**？
- 直接 fit 进 Clements/Reck universal MZI mesh
- 任意 qubit pair 之间都能用相同的 layout pattern 做 gate
- 类似 FPGA 的 regularity——同一套 routing 资源可用于任意位置

**代价**：波导数从 2n 增加到 3n，多 50% hardware。但这是值得的。

### Trick 2：Optical SWAP Gate

Regular labeling 下，auxiliary 永远在 qubit 的"上面"。但 CZ 需要 auxiliary 夹在两个 qubit 中间。怎么办？

**Solution**：做 CZ 之前，用 MZI 把 target qubit 的波导 cyclic permutation 一下，把 auxiliary 移到需要的位置；做完 CZ 再 permute 回来。

**Cyclic permutation 矩阵**（Equation 27）：

$$\mathrm{SWAP}_1^A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$$

作用是 $(w_A, w_0, w_1) \to (w_1, w_A, w_0)$，把 auxiliary 从顶移到底。

**实现**：两个 MZI 设成 Pauli-X 配置（cross state），就能实现这个 permutation。

更厉害的是 **qubit-level SWAP**（Equation 31）：交换两个 qubit 的位置，不动 auxiliary。用 5 层 MZI、10 个 Pauli-X 单元实现。这样就能把 non-neighbor qubit 交换到 neighbor 位置做 gate，做完再 swap 回来。

**关键 insight**：photonic 上的 SWAP 是 **deterministic** 的（MZI cross state 100% success），这跟 superconducting qubit 用 3 个 CNOT 做 SWAP 完全不同——光子芯片上 SWAP 是 "免费" 的。这是 photonic 平台的独特优势。

### Trick 3：Truncation Trick（Enable Cascade）

Post-selected CZ 不能直接 cascade 的原因：做完第一个 CZ，auxiliary 上可能有 photon，第二个 CZ 就被污染。

**Truncation trick**：在每个 CZ 的输出端，**物理上把 auxiliary 波导截断**（不连到下一级）。这样 auxiliary 上的 photon 直接 loss 掉，下一个 CZ 的 input 保证 auxiliary 是空的。

数学上用 projection operator（Equation 20）：

$$\hat{\tilde{P}}_{\mathrm{aux}} = \mathrm{diag}(0, 1, 1, 0, 1, 1)$$

这个对角矩阵把 auxiliary 对应的位置设为 0，强制丢弃 auxiliary 上的 photon。

**为什么合法**？原本 post-selection 也要丢 auxiliary 上的 photon，truncation 只是硬件化这件事——把 post-selection 提前到 circuit 中间执行。

**限制**：truncation 只允许 CZ 在 **共享一个 qubit 的不同 pair** 上 cascade（如 $(q_0, q_1)$ 然后 $(q_1, q_2)$）。**同一 pair 上不能 cascade 两次**，因为同一 dual-rail 上的双光子事件（$|2,0\rangle$ 或 $|0,2\rangle$）truncation 处理不了。

---

## 7. Compressed CZ：节省 MZI 资源

观察 Fig. 8(a)：SWAP 的一部分 MZI 和 CZ 的一部分 MZI 紧邻，可以合并。

具体来说，CZ 底部的 $R_{1/3}'$ 两侧是 Pauli-X MZI（来自 SWAP），根据 $X \cdot R \cdot X = R^\dagger$，可以合并成单个 MZI：

$$R_{1/3}^\dagger = X \cdot R_{1/3} \cdot X = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & -\sqrt{2} \\ \sqrt{2} & -1 \end{pmatrix}$$

对应的 phase 配置：$\theta_1-\theta_2 = -2\arcsin\frac{1}{\sqrt{3}}$，$\phi_1-\phi_2 = 0$。

**效果**：Regular-labeled CZ 从 6 层 MZI 压缩到 4 层。在 photonic 电路里，每少一层 MZI 就少一倍 insertion loss，这是实际工程中非常重要的优化。

---

## 8. 实例：Bell State 和 GHZ State

### Bell State（2 qubit）

电路：$H$ on $q_0$，然后 CNOT on $(q_0, q_1)$。

$$|00\rangle \xrightarrow{H\otimes I} \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) \xrightarrow{\mathrm{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

在 photonic 上实现：$H$ 是单个 MZI（设成 Hadamard 配置），CNOT = $H \cdot \mathrm{CZ} \cdot H$，所以总共 1 个 regular-labeled CZ。

**Success probability** = $1/9$（一个 CZ gate）。

### GHZ State（3 qubit）

电路：$H$ on $q_0$，CNOT on $(q_0, q_1)$，CNOT on $(q_1, q_2)$。

$$|000\rangle \xrightarrow{H} \frac{1}{\sqrt{2}}(|000\rangle + |100\rangle) \xrightarrow{\mathrm{CNOT}_{01}} \frac{1}{\sqrt{2}}(|000\rangle + |110\rangle) \xrightarrow{\mathrm{CNOT}_{12}} \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

需要 2 个 CZ，中间用 truncation trick 清空 auxiliary。

**Success probability** = $(1/9)^2 = 1/81$。

### SWAP-Enabled GHZ

如果要做 CNOT on $(q_0, q_2)$（non-neighbor），需要先 SWAP 把 $q_2$ 换到 $q_1$ 旁边，做完 CNOT 再 swap 回来。这演示了 optical SWAP 的实际用途。

### n-Qubit GHZ Scaling

n-qubit GHZ 需要 $n-1$ 个 CNOT，success probability $9^{1-n}$。

| n | Success Probability |
|---|---|
| 2 | 1/9 ≈ 11% |
| 3 | 1/81 ≈ 1.2% |
| 4 | 1/729 ≈ 0.14% |
| 5 | 1/6561 ≈ 0.015% |
| 10 | 1/9^9 ≈ 2.6×10⁻⁹ |

**人话**：$n > 5$ 基本就不 practical 了。这就是 gate-based LOQC 的根本限制——success probability 指数衰减。

---

## 9. 为什么这 Paper 重要

### 9.1 解决了 Layout 问题

之前 post-selected CZ 的 non-regular structure 是 LOQC scalability 的 "软肋"——即使物理上能做 CZ，芯片布局也无法 scale。这篇 paper 用 3 个 trick（regular labeling + optical SWAP + truncation）解决了 layout 问题。

### 9.2 建立 Photonic Compilation Framework

有了 regular structure，就可以像 classical digital circuit 一样做 **photonic circuit compilation**：
1. High-level quantum algorithm（Qiskit/Cirq）
2. Decompose 成 H, CNOT, T gate set
3. Map 到 regular-labeled MZI mesh（用 SWAP 做 routing）
4. Insert truncation between CZ gates
5. Compile 成 MZI phase 配置

这是从 "物理实现" 到 "工程系统" 的关键一步。

### 9.3 Honest Limitation

作者诚实地承认：success probability 的指数衰减是根本问题，gate-based LOQC 在 $n > 5$ 时 impractical。Large-scale QC 更可能走 **fusion-based measurement-based QC**（[参考](https://doi.org/10.1038/s41467-023-36406-y)）路线。

但这篇 paper 的价值在于：**定义了 gate-based LOQC 的 scalability 边界，并给出了边界内的最优解决方案**。

---

## 10. 与其他平台对比

| 维度 | Superconducting | Trapped Ion | Photonic (本 paper) |
|------|----------------|-------------|---------------------|
| Qubit 类型 | Localized | Localized | Path-encoded (flying) |
| Two-qubit gate | Capacitive coupler | Coulomb interaction | Post-selected CZ |
| Success rate | ~99% | ~99% | 1/9 |
| SWAP cost | 3 CNOT | 3 CNOT | Free (optical) |
| Decoherence | μs | s | Negligible |
| Temperature | mK | Room | Room |
| Scalability 瓶颈 | Cryo wiring | Trap control | Probabilistic gate |

**Photonic 的独特优势**：SWAP 是 deterministic 的，而且 decoherence 几乎为零。**独特劣势**：gate 是 probabilistic 的，需要大量 post-selection。

---

## 11. Open Problems

1. **Success probability 指数衰减**：$9^{1-n}$ scaling 让 large-scale impractical。需要 fusion gate 或 quantum error correction 来 boost。
2. **同一 pair 不能 cascade 两次 CZ**：限制了能实现的 quantum circuit 类别。
3. **Loss 累积**：每个 MZI 约 0.1-0.3 dB loss（[参考](https://doi.org/10.1038/s41566-019-0524-2)），deep circuit 总 loss 太高。
4. **Phase stability**：大量 MZI 需要长期 phase stability，thermo-optic crosstalk 是工程挑战。
5. **Detector efficiency**：post-selection 要求高 efficiency detector，SNSPD 约是 90%+，但仍有提升空间。

---

## 12. 我的 Intuition Takeaway

读完这篇 paper，我建立的 intuition 是：

1. **Photonic QC 的瓶颈不在物理，而在架构**。Linear optics + post-selection 能实现 universal QC，但 layout 必须规则化才能 scale。

2. **Regularity 比 success probability 更基础**。即使 success rate 低，只要 layout regular，就能用 error correction 或 fusion 补救；如果 layout 不 regular，连尝试 scale 都做不到。

3. **SWAP 是 photonic 的杀手锏**。Deterministic SWAP 让 non-neighbor qubit 交互成为可能，这是 superconducting 和 trapped ion 平台做不到的（它们的 SWAP 也要用 probabilistic gate）。

4. **Post-selection 是双刃剑**。它让 linear optics 能实现 non-linear gate，但也引入 probabilistic nature，导致 success rate 指数衰减。这是 LOQC 的根本 tension。

5. **Truncation trick 是巧妙的工程优化**。把 post-selection 硬件化到 circuit 中间，既节省 detector 资源，又 enable 了 cascade。这种 "把 quantum operation 偷偷 classical 化" 的思路在 photonic 工程里很常见。

6. **未来方向是 hybrid**。纯 gate-based LOQC 难以 scale，纯 MBQC 又需要大量 resource state。Hybrid approach（gate-based 生成 small cluster，fusion 连接成 large cluster）可能才是实际路线。

---

## References

- DiVincenzo criteria: [https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E](https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E)
- Reck scheme: [https://doi.org/10.1103/PhysRevLett.73.58](https://doi.org/10.1103/PhysRevLett.73.58)
- Clements scheme: [https://doi.org/10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460)
- KLM protocol: [https://doi.org/10.1038/35001024](https://doi.org/10.1038/35001024)
- Hofmann-Takeuchi CZ: [https://doi.org/10.1103/PhysRevA.66.024308](https://doi.org/10.1103/PhysRevA.66.024308)
- Hong-Ou-Mandel: [https://doi.org/10.1103/PhysRevLett.59.2044](https://doi.org/10.1103/PhysRevLett.59.2044)
- Fusion-based QC: [https://doi.org/10.1038/s41467-023-36406-y](https://doi.org/10.1038/s41467-023-36406-y)
- Integrated photonic review: [https://doi.org/10.1038/s41566-019-0524-2](https://doi.org/10.1038/s41566-019-0524-2)
- Surface codes: [https://doi.org/10.1103/PhysRevA.86.032324](https://doi.org/10.1103/PhysRevA.86.032324)
- Silicon photonics large-scale: [https://doi.org/10.1038/nphoton.2016.178](https://doi.org/10.1038/nphoton.2016.178)
- Rudolph "Why silicon photonics": [https://doi.org/10.1063/1.4976737](https://doi.org/10.1063/1.4976737)
- Boson sampling: [https://doi.org/10.1145/1806689.1806748](https://doi.org/10.1145/1806689.1806748)

---

# Quantum Circuit Mapping for MZI-based Integrated Photonics 详解

## 1. 论文核心问题与动机

这篇 paper 处理的是 **linear optical quantum computing (LOQC)** 在 integrated photonics 平台上的 scalability 问题。核心矛盾在于：**post-selected CZ gate** 是目前 resource-efficient 的 universal two-qubit gate，however 它的 waveguide 排列是非 regular 的，导致无法 cascade，阻碍 large-scale QC 的实现。

DiVincenzo criteria（参考 [https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E](https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E)）规定了实现 quantum computer 的五个条件，其中 **scalability with a regular qubit structure** 是关键一条。在 superconducting qubit、trapped ion 等平台，qubit 是 localized 的，连接图通过物理 coupler 决定；在 photonic 平台，qubit 是 path-encoded 的，本质上是 waveguide 几何决定的，所以 "regular structure" 直接转化为 waveguide layout 的几何约束。

## 2. Path Encoding 与 Bosonic Qubit

### 2.1 Dual-rail 表示

Equation (1) 定义了 path-encoded qubit：

$$|0\rangle \equiv \hat{a}_{w_0}^{\dagger}|\Omega\rangle = |1,0\rangle_{(w_0,w_1)}, \quad |1\rangle \equiv \hat{a}_{w_1}^{\dagger}|\Omega\rangle = |0,1\rangle_{(w_0,w_1)}$$

**变量说明**：
- $\hat{a}_{w_i}^{\dagger}$：在 waveguide $w_i$ 上创建一个 photon 的 creation operator；下标 $w_i$ 标识 photon 所在 spatial mode
- $|\Omega\rangle$：vacuum state，所有 mode 都没有 photon
- $|1,0\rangle$：bold number 表示 **occupation number**，即 waveguide $w_0$ 有 1 photon、$w_1$ 有 0 photon
- sans-serif $|0\rangle, |1\rangle$：computational basis，区别于 occupation number

**为什么选 path encoding 而不是 polarization**？Polarization 在 integrated photonics 里很难保持（chip 上的 waveguide birefringence 难以完全控制），而 path encoding 直接用 waveguide 几何定义，与 fabrication 兼容性更好。Time-bin encoding 也有用，但需要 active temporal modulation，不适合 gate-based 通用 QC。

### 2.2 双 rail 的局限性

n qubit 需要 2n waveguides 与 n 个 photon。**qubit structure preservation** 是 post-selection 的核心约束：每个 dual-rail pair 必须有且仅有一个 photon。Loss 破坏这个 structure（photon 数减少），双光子事件破坏这个 structure（同一 dual-rail 出现两个 photon）——这两种情况都被 post-selection 丢弃。

**Intuition**：dual-rail encoding 把 quantum information 编码在 "which path" 的 freedom 上，photon loss 直接表现为 "no photon in this qubit"，是 heralded error，可被 detector 检测。这是 LOQC 相对其他平台的优势——error 主要是 loss，而不是 decoherence。

## 3. MZI 作为 Single-Qubit Universal Gate

### 3.1 Beam Splitter 与 Phase Shifter

Equation (6) 给出 ideal BS：

$$\mathbf{BS} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & \pm i \\ \pm i & 1 \end{pmatrix}$$

**符号约定**：$+i$ 对应 MMI（multimode interferometer，基于 self-imaging），$-i$ 对应 directional coupler（基于 evanescent wave）。这两类器件物理实现不同，但数学上等价（差一个 global phase）。

Equation (7) 的 Phase Shifter：

$$\mathrm{PS}(\theta) = \begin{pmatrix} e^{i\theta_1} & 0 \\ 0 & e^{i\theta_2} \end{pmatrix}$$

其中 $\theta = (\theta_1, \theta_2)$ 是两条 waveguide 上独立施加的相位。物理上通过 thermo-optic effect（加热 waveguide 改变 $n_{\mathrm{eff}}$）或 electro-optic effect 实现。**只有 relative phase $\theta_1 - \theta_2$ 可观测**，global phase 不可区分。

### 3.2 MZI 的完整矩阵

Equation (8) 是核心公式：

$$U_{\mathrm{MZI}}(\theta) = \mathbf{BS} \cdot \mathrm{PS}(\theta) \cdot \mathbf{BS} \cdot \mathrm{PS}(\phi)$$

展开后：

$$U_{\mathrm{MZI}} = ie^{i(\theta_1+\theta_2+2\phi_2)/2} \begin{pmatrix} e^{i(\phi_1-\phi_2)}\sin\frac{\theta_1-\theta_2}{2} & \cos\frac{\theta_1-\theta_2}{2} \\ e^{i(\phi_1-\phi_2)}\cos\frac{\theta_1-\theta_2}{2} & -\sin\frac{\theta_1-\theta_2}{2} \end{pmatrix}$$

**上标下标意义**：
- $\theta_1, \theta_2$：MZI 内部两个 arm 上 PS 引入的相位
- $\phi_1, \phi_2$：MZI 输入端两个 PS 引入的相位
- 所有 cross-term 都依赖 $\theta_1 - \theta_2$ 和 $\phi_1 - \phi_2$，这就是为什么 Table 1 只列出这两个 difference

**关键观察**：MZI 等价于一个 tunable beam splitter，transmittance $|t|^2 = \sin^2\frac{\theta_1-\theta_2}{2}$，reflectance $|r|^2 = \cos^2\frac{\theta_1-\theta_2}{2}$。所以 MZI 可以实现 0:100 到 50:50 之间任意比例的 BS。

### 3.3 Single-Qubit Gate 实现

Table 1 列出常见量子门对应的 phase 配置。比如 Hadamard：$(\theta_1-\theta_2, \phi_1-\phi_2) = (\pi/2, 0)$。

$R_z$ 旋转只需 MZI 内部 phase；$R_x, R_y$ 需要在 MZI 输出端再加一对 PS（Equation 9 的 extended MZI）。

**Intuition**：MZI 实现 SU(2) 的任意 element。Bloch sphere 上任意单 qubit rotation 都可以分解为 $R_z(\alpha) R_y(\beta) R_z(\gamma)$，需要 3 个独立参数。标准 MZI 提供 2 个独立 phase（$\theta_1-\theta_2$ 和 $\phi_1-\phi_2$），加上输出端 PS 的 1 个 relative phase，共 3 个参数，足够覆盖 SU(2)。

## 4. Post-Selected CZ Gate：从 Linear 到 Non-Linear

### 4.1 为什么 4×4 MZI 网络无法实现 CZ

Appendix A 给出一个重要的 no-go result。给定 4×4 unitary $U_4$，对 input state 施加 $U_4^{-1}$，要求 output 满足 CZ 的形式（Equation 45），求解约束得到 Equation (46) 的矩阵结构：

$$U_4^{-1} = \begin{pmatrix} \gamma_{11} & 0 & 0 & 0 \\ 0 & -\gamma_{11} & \gamma_{23} & 0 \\ 0 & \frac{2c}{\gamma_{23}} & \frac{c}{\gamma_{11}} & 0 \\ 0 & 0 & 0 & \frac{c}{\gamma_{11}} \end{pmatrix}$$

要求 unitarity $U_4 U_4^{-1} = \mathbf{1}$，约束无解。

**Intuition**：4×4 linear network 只能做 4 维 Hilbert 空间的 unitary，single-photon 在 4 个 mode 间的线性变换。但 CZ 是关于 two-photon 的非 trivial joint transformation（要求 $|11\rangle \to -|11\rangle$ 而其他 basis 不变），这种 controlled operation 需要 two-photon interference。Photon 在线性网络里只通过 Hong-Ou-Mandel interference（[https://doi.org/10.1103/PhysRevLett.59.2044](https://doi.org/10.1103/PhysRevLett.59.2044)）相互作用，4 个 mode 不足以产生需要的 non-linearity。

### 4.2 6-Waveguide Post-Selected CZ

Hofmann-Takeuchi 方案（[https://doi.org/10.1103/PhysRevA.66.024308](https://doi.org/10.1103/PhysRevA.66.024308)）使用 6 个 waveguide：4 个 dual-rail 加 2 个 auxiliary。核心是 1/3-BS，其矩阵如 Equation (12)：

$$R_{1/3} = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & \sqrt{2} \\ -\sqrt{2} & -1 \end{pmatrix}, \quad R'_{1/3} = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & \sqrt{2} \\ \sqrt{2} & 1 \end{pmatrix}$$

对应的 phase：$\theta_1 - \theta_2 = 2\arcsin\frac{1}{\sqrt{3}}$（transmittance $|t|^2 = 1/3$），$\phi_1 - \phi_2 = \pi$ 或 $0$。

**为什么是 1/3**？这是 Hofmann-Takeuchi 经过优化得到的最优 transmittance，使 6-mode 网络能够实现 CZ 同时 success probability 最大化。Success probability 是 $1/9$（Equation 19），由 amplitude coefficient $-1/3$ 平方得到。

### 4.3 CZ 的完整矩阵

Equation (14)：

$$\overline{\mathbf{CZ}}_{\mathrm{ps}} = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & \sqrt{2} & 0 & 0 & 0 & 0 \\ \sqrt{2} & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & -1 & -\sqrt{2} & 0 & 0 \\ 0 & 0 & \sqrt{2} & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 & -1 & -\sqrt{2} \\ 0 & 0 & 0 & 0 & \sqrt{2} & -1 \end{pmatrix}$$

**结构理解**：
- 矩阵是 block-diagonal 的，3 个 2×2 block
- 每个 block 是 $R_{1/3}$ 或 $R'_{1/3}$ 的形式
- 上面的 bar 表示 **non-regular labeling**：waveguide 顺序是 $(w_A^0, w_0^0, w_1^0, w_0^1, w_1^1, w_A^1)$

**关键**：auxiliary waveguide 夹在 dual-rail pair 之间。这就是 non-regular structure 的根源——auxiliary waveguide 不在每个 qubit 的固定位置。

### 4.4 Post-Selection 的物理含义

Equation (16) 展开 output state：

$$|\Psi\rangle_{\mathrm{out}} = -\frac{1}{3}(\alpha_0\alpha_1 \hat{a}_{w_0^0}^{\dagger}\hat{a}_{w_0^1}^{\dagger} + \alpha_0\beta_1 \hat{a}_{w_0^0}^{\dagger}\hat{a}_{w_1^1}^{\dagger} + \beta_0\alpha_1 \hat{a}_{w_1^0}^{\dagger}\hat{a}_{w_0^1}^{\dagger} - \beta_0\beta_1 \hat{a}_{w_1^0}^{\dagger}\hat{a}_{w_1^1}^{\dagger})|\Omega\rangle + \dots$$

省略号包含 4 类 unwanted events：
1. 1 photon in $w_A^0$ + 1 photon in some dual-rail
2. 1 photon in $w_A^1$ + 1 photon in some dual-rail
3. 2 photons in auxiliary waveguides
4. 2 photons in one dual-rail

**Intuition**：post-selection 等价于在 36 维 Fock space 上投影到 4 维 computational subspace。$|1/3|^2 = 1/9$ 的 success rate 是 projection 的几何效果。

## 5. Regular Labeling 与 Three-Waveguide Qubit

### 5.1 三波导结构

论文的核心设计 choice（Section 2.1, Fig. 1）：每个 qubit 由 3 个 waveguide 组成，$(w_A, w_0, w_1)$，其中 $w_A$ 是 auxiliary。n 个 qubit 需要 3n waveguides。

**Regular labeling**（Section 3.1.1）：waveguide 顺序为 $(w_A^0, w_0^0, w_1^0, w_A^1, w_0^1, w_1^1, \dots)$。每个 qubit 占据连续 3 个 waveguide，auxiliary 永远在固定位置（最上方）。

**为什么这样设计**？这直接对应 Clements/Reck scheme 的几何（[https://doi.org/10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460), [https://doi.org/10.1103/PhysRevLett.73.58](https://doi.org/10.1103/PhysRevLett.73.58)）。Regular labeling 让任意 qubit 之间的 MZI 都能用相同 layout pattern 实现，类似 FPGA 中 LUT 的 regularity。

### 5.2 三波导的代价

代价是 resource 增加 50%（从 2n 到 3n waveguides）。但是这个 overhead 是值得的：
1. 任意 qubit pair 都能通过 optical SWAP 调整到相邻位置
2. Auxiliary waveguide 的固定位置让 CZ 能 regular 地实现
3. Post-selection 中 auxiliary 的位置是确定的

## 6. Optical SWAP Gate

### 6.1 Pauli-X 实现 SWAP

Equation (21)：

$$(\hat{a}_{w_0}^{\dagger}, \hat{a}_{w_1}^{\dagger}) \to \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} \hat{a}_{w_0}^{\dagger} \\ \hat{a}_{w_1}^{\dagger} \end{pmatrix}$$

输出 $(\hat{a}_{w_1}^{\dagger}, \hat{a}_{w_0}^{\dagger})$，即两个 mode 的 photon 被交换。MZI 配置为 Pauli-X（Table 1: $(\theta_1-\theta_2, \phi_1-\phi_2) = (0,0)$），等价于 cross state。

**物理直觉**：Pauli-X 的 MZI 等价于 waveguide crossing，但是 reconfigurable——可以 electrically switch between cross and bar state。

### 6.2 SWAP' Operation

Equation (24)：

$$\mathrm{SWAP}_2' = X_4^{(2,3)} \cdot X_4^{(1,2)} \cdot X_4^{(3,4)} \cdot X_4^{(2,3)} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

**变量说明**：$X_m^{(k,k+1)}$ 表示在 $m \times m$ 网络中第 $k$ 和 $k+1$ 个 waveguide 上施加 MZI（Pauli-X 配置）。

**矩阵结构**：这是 4 个 waveguide 的两两交换，等价于 block 交换 $|q_0\rangle \leftrightarrow |q_1\rangle$。用 4 个 MZI 实现，3 层深度。

**为什么需要 3 层而不是 1 层**？因为相邻 MZI 不能同时作用于共享 waveguide（photon 同时经过两个 MZI 会产生 unwanted interference），所以需要分时执行。这是 photonic network 的 fundamental 约束。

### 6.3 SWAP_1^A：交换 Ancilla 和 Dual-Rail

Equation (27)：

$$\mathrm{SWAP}_1^A = X_3^{(0,1)} \cdot X_3^{(A,0)} = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$$

这是 cyclic permutation：$w_A \to w_1, w_0 \to w_A, w_1 \to w_0$。把 auxiliary 从顶部移到底部，使得 CZ 的非 regular 结构能与 regular labeling 对接。

**关键用途**：CZ 需要 auxiliary 在 dual-rail 外侧（夹在两个 qubit 之间）。Regular labeling 中 auxiliary 在固定位置。SWAP_1^A 在 target qubit 上做 cyclic permutation，临时把 auxiliary 移到合适位置，CZ 执行后再 SWAP 回来。

### 6.4 Qubit SWAP（不动 Ancilla）

Equation (31) 的优化版本：

$$\mathrm{SWAP}_2 = \mathbb{X}_1 \cdot \mathbb{X}_2 \cdot \mathbb{X}_1 \cdot \mathbb{X}_2 \cdot \mathbb{X}_1$$

其中 $\mathbb{X}_1 = X_6^{(3,4)} \cdot X_6^{(5,6)}$，$\mathbb{X}_2 = X_6^{(2,3)} \cdot X_6^{(4,5)}$。

**关键观察**：auxiliary waveguide $w_A^0, w_A^1$ 不参与 SWAP，因为 post-selection 反正会丢弃 auxiliary 上的 photon。这节省了 MZI 资源，从 13 个减到 10 个（Section 3.2.1）。

**量子 SWAP 的 composition**：SWAP = CNOT · CNOT · CNOT，但在 LOQC 中直接用 optical SWAP 更高效，避免 probabilistic gate 的多次级联。

## 7. Regular-Labeled CZ 与压缩

### 7.1 Regular-Labeled CZ 构造

Equation (33)：

$$\mathbf{1} \otimes (\mathrm{SWAP}_1^A)^T \cdot \overline{\mathbf{CZ}}_{\mathrm{ps}} \cdot \mathbf{1} \otimes \mathrm{SWAP}_1^A$$

即：先对 target qubit 做 SWAP_1^A（regular → non-regular layout），执行 CZ，再做 SWAP_1^A 反向（non-regular → regular layout）。

### 7.2 压缩 CZ（Equation 34）

$$R_{1/3}^\dagger = X \cdot R_{1/3} \cdot X = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 & -\sqrt{2} \\ \sqrt{2} & -1 \end{pmatrix}$$

**为什么压缩**：注意到 Fig. 8(a) 中 CZ 底部的 $R_{1/3}'$ 两侧紧邻 Pauli-X MZI（来自 SWAP_1^A）。这两个 X MZI 可以和 $R_{1/3}'$ 合并成一个 MZI，因为 $X \cdot R \cdot X = R^\dagger$ 仍是单个 MZI 可实现的 unitary。

**压缩后的 regular-labeled CZ**（Equation 36）：

$$\mathrm{CZ}_{\mathrm{ps}} = X_6^{(w_A^1, w_0^1)} \cdot \overline{\mathbf{CZ}}_{\mathrm{ps}} \cdot X_6^{(w_A^1, w_0^1)}$$

只用 4 个 1/3-BS + 2 个 Pauli-X MZI（被压缩），共 4 层 MZI。

**Intuition**：这是 photonic circuit compilation 的核心思想——相邻的 MZI 可以合并如果它们的 phase 满足合成关系。类似 classical logic synthesis 中的 gate merging。

## 8. Truncation Trick 与 Cascade

### 8.1 Cascade 问题

Post-selected CZ 不能直接 cascade。如果对 $(q_0, q_1)$ 做完 CZ 后，photon 可能仍在 $w_A^0$ 或 $w_A^1$。下一个 CZ 在 $(q_1, q_2)$ 上执行时，$w_A^1$ 上的 photon 会干扰新的 CZ 干涉。

**数学原因**：CZ 的 output 是 superposition，包含 $w_A^1$ 上有 photon 的 component（虽然 post-selection 丢弃）。下一个 CZ 的 linear transformation 会作用于这些 component，产生与 correct state 混叠的 unwanted contribution。

### 8.2 Truncation Trick

Equation (20)：

$$\hat{\tilde{P}}_{\mathrm{aux}} = \mathrm{diag}(0, 1, 1, 0, 1, 1)$$

这是 non-unitary projection operator，丢弃 auxiliary waveguide 上的所有 photon。物理实现：直接断开 auxiliary waveguide，让 photon 逃逸（被 loss 掉）。

**为什么这样做合法**？原本 post-selection 也要丢弃 auxiliary 上的 photon，truncation 只是把这件事硬件化了——在 CZ 输出端就把 auxiliary 截断，使得下一个 CZ 的 input 一定是 auxiliary 空。

**Cascade 公式**（Equation 62）：

$$\mathrm{CZ}_{\mathrm{ps}}^{(2,3)} \cdot \hat{P}_{\mathrm{aux}} \cdot \mathrm{CZ}_{\mathrm{ps}}^{(1,2)}$$

中间插入 $\hat{P}_{\mathrm{aux}}$ 强制清空所有 auxiliary。

**Important caveat**（Section 5）：truncation 只允许 CZ 在 **共享一个 qubit 的不同 pair** 上 cascade（如 $(q_0,q_1)$ 然后 $(q_1,q_2)$），不允许在同一 pair 上 cascade 两次。这是因为同一 dual-rail 上的双光子事件（如 $|2,0\rangle$ 在 $(w_0^0, w_1^0)$）无法被 truncation 处理，会污染第二次 CZ 的结果。

## 9. Bell State 生成（Section 4）

### 9.1 电路与映射

Bell state 电路（Fig. 9a）：$H$ on $q_0$，然后 CNOT on $(q_0, q_1)$。

Equation (39)：

$$|00\rangle \xrightarrow{H \otimes \mathbf{1}} \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) \xrightarrow{\mathrm{CNOT}_{\mathrm{ps}}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

**Success probability**：$1/9$，因为只有一个 CZ gate。

### 9.2 MZI Network Embedding

Fig. 9(b) 和 (c) 分别给出 Clements 和 Reck scheme 中的 embedding。

**Clements scheme**：MZI 排列成矩形 mesh，深度 $m$，每层最多 $m/2$ 个 MZI。优点是 optical depth 浅，loss 低。

**Reck scheme**：MZI 排列成三角 mesh，深度 $2m-3$。优点是结构直观，便于理解 signal propagation。

**Embedding 的选择**：CNOT 的对称结构更适合 Clements（5 种 embedding 方式），Reck 只有 1 种。

## 10. GHZ State 生成（Section 4）

### 10.1 三 qubit GHZ 电路

Equation (40)：

$$U_{\mathrm{GHZ}} = \hat{P}_{\mathrm{aux}} \cdot (\mathbf{1} \otimes \mathrm{CNOT}_{\mathrm{ps}}^{(2,3)}) \cdot \hat{P}_{\mathrm{aux}} \cdot (\mathrm{CNOT}_{\mathrm{ps}}^{(1,2)} \otimes \mathbf{1}) \cdot (H \otimes \mathbf{1} \otimes \mathbf{1})$$

**Success probability**：$1/81 = (1/9)^2$，因为两个 CZ gate。

**Truncation 位置**：只在 $w_A^1$ 上需要 truncation（Fig. 10），因为只有 $w_A^1$ 是两个 CZ 共享的 auxiliary。论文中为了通用性在所有 CZ 后都加 truncation。

### 10.2 SWAP-Enabled GHZ（Fig. 11）

$$U_{\mathrm{GHZ}}^{(\mathrm{swap})} = \hat{P}_{\mathrm{aux}} \cdot (\mathrm{CNOT}^{(1,2)} \otimes \mathbf{1}) \cdot (\mathbf{1} \otimes \mathrm{SWAP}_2^{(2,3)}) \cdot \hat{P}_{\mathrm{aux}} \cdot (\mathrm{CNOT}^{(1,2)} \otimes \mathbf{1}) \cdot (H \otimes \mathbf{1} \otimes \mathbf{1})$$

这里先做 CNOT on $(q_0, q_1)$，然后 SWAP $(q_1, q_2)$，再做 CNOT on $(q_0, q_1)$（现在的 $q_1$ 是原来的 $q_2$），最后再 SWAP 回来。这等价于 CNOT on $(q_0, q_2)$。

**为什么需要 SWAP**：直接做 CNOT on $(q_0, q_2)$ 需要 6+6 个 waveguide（中间隔着 $q_1$），无法 fit 在 9 个 waveguide 的 network 中。SWAP 把 $q_2$ 移到 $q_1$ 旁边，做完 CNOT 再移回。

### 10.3 n-Qubit GHZ Scaling

Equation (41) 的 n-qubit GHZ 需要 $n-1$ 个 CNOT，success probability $9^{1-n}$。指数衰减意味着这种方法在 $n > 5$ 时 success rate 已经低于 $10^{-4}$，不再 practical。

**这是 paper 的 honest limitation**：作者明确指出 measurement-based approach（fusion-based QC, [https://doi.org/10.1038/s41467-023-36406-y](https://doi.org/10.1038/s41467-023-36406-y)）在这种规模上更优。

## 11. 与其他方案的比较

### 11.1 KLM Protocol（[https://doi.org/10.1038/35001024](https://doi.org/10.1038/35001024)）

KLM 用 teleportation-based gate，每个 gate 用大量 ancilla photon 和 detector，success probability 接近 1（with enough ancilla）。Resource 远大于 post-selected CZ，但 scalability 更好。

### 11.2 Fusion-Based QC（[https://doi.org/10.1038/s41467-023-36406-y](https://doi.org/10.1038/s41467-023-36406-y)）

MBQC with cluster state，每次 fusion gate 是 probabilistic 但通过 resource state injection 可以 boost success rate。Silicon photonics 平台已经 demonstrate 大规模 fusion（如 [https://doi.org/10.1038/s41566-019-0524-2](https://doi.org/10.1038/s41566-019-0524-2) review）。

### 11.3 Deterministic Photonic Gate

通过 cavity QED 或 Rydberg blockade 可以实现 deterministic two-photon gate，但技术成熟度远不如 linear optical。近期 Xanadu（[https://xanadu.ai](https://xanadu.ai)）和 PsiQuantum 都在 fusion-based 方向上推进。

## 12. Paper 的 Position 与 Open Problems

这篇 paper 的贡献是在 **gate-based LOQC framework 内** 提供 scalable architecture 的设计方案。它不解决 success probability 指数衰减的问题，但解决了 geometric/layout 的 regularity 问题。Regular labeling + optical SWAP + truncation trick 这三个 primitive 组合起来，让 photonic circuit 能像传统 digital circuit 一样 regular 地 layout 和 route。

**Open problems**（Section 5）：
1. $1/9$ success rate 的 exponential scaling
2. 不能在同一 qubit pair 上 cascade 两次 CZ（即使有 truncation）
3. Loss 累积：每个 MZI 约 0.1-0.3 dB loss（[https://doi.org/10.1038/s41566-019-0524-2](https://doi.org/10.1038/s41566-019-0524-2)），deep circuit 总 loss 太高
4. Phase stability：大量 MZI 需要长期 phase stability，thermo-optic crosstalk 是工程挑战

**Future direction**：作者暗示融合 measurement-based 和 gate-based 的 hybrid approach 可能是出路——gate-based 用于 small cluster，fusion 用于 large scale entanglement。

## 13. Architecture Diagram 解析（Fig. 4）

Fig. 4(a) 显示 6×6 Clements（上）和 Reck（下）scheme。每个小方格是一个 MZI，连接相邻两个 waveguide。

- **Clements**：MZI 排列成 6 列 × 3 行的矩形（深度 6），每列有 3 个 MZI 交替作用于 waveguide pair (1,2), (3,4), (5,6) 或 (2,3), (4,5)。深度浅，但每层 waveguide 间都要 cross。
- **Reck**：MZI 排列成三角，深度 11（$2 \times 6 - 1$）。结构递归，每个 MZI 处理一个自由度。

Fig. 4(b) 显示 3n × 3n regular labeling。每个 qubit 占据连续 3 个 waveguide，auxiliary 在最上。Universal MZI network 通过 Reck/Clements scheme 覆盖整个 3n modes。

## 14. 总结

这篇 paper 的核心 intuition 是：**在 LOQC 中，"regularity" 是 scalability 的关键，比 success probability 更基础**。即使 success rate 仍指数衰减，只要 layout regular，就能利用 photonic compiler 自动 map arbitrary quantum circuit 到 physical device。Regular labeling（3 waveguide per qubit）+ optical SWAP（reconfigurable routing）+ truncation trick（enabling cascade）这三个 primitive 构成了 gate-based LOQC 的 scalable substrate。

虽然最终 large-scale QC 可能要靠 fusion-based 或 hybrid approach，但这篇 paper 提供了理解 "为什么 LOQC 在 gate-based framework 内难以直接 scale" 的清晰图景，以及"如果要做，需要哪些 geometric 上的 design choice"。

**Key takeaway for building intuition**：
- Linear optical gate 本质是 mode permutation + phase shift
- Two-qubit non-linearity 来自 post-selection，不是真正的非线性
- Regularity 在 photonic 比 superconducting 更重要，因为 photonic 没有 localized qubit 概念
- Cascade 是 probabilistic gate 的根本难题，truncation 是部分解法，但只对 share-one-qubit 的情况有效
- SWAP gate 在 photonic 中是 "free"（deterministic），这是 photonic 相对其他平台的独特优势

**References**：
- DiVincenzo criteria: [https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E](https://doi.org/10.1002/1521-3978(200009)48:9<11<771::AID-PROP771>3.0.CO;2-E)
- Reck scheme: [https://doi.org/10.1103/PhysRevLett.73.58](https://doi.org/10.1103/PhysRevLett.73.58)
- Clements scheme: [https://doi.org/10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460)
- KLM protocol: [https://doi.org/10.1038/35001024](https://doi.org/10.1038/35001024)
- Hofmann-Takeuchi CZ: [https://doi.org/10.1103/PhysRevA.66.024308](https://doi.org/10.1103/PhysRevA.66.024308)
- Hong-Ou-Mandel: [https://doi.org/10.1103/PhysRevLett.59.2044](https://doi.org/10.1103/PhysRevLett.59.2044)
- Surface codes: [https://doi.org/10.1103/PhysRevA.86.032324](https://doi.org/10.1103/PhysRevA.86.032324)
- Fusion-based QC: [https://doi.org/10.1038/s41467-023-36406-y](https://doi.org/10.1038/s41467-023-36406-y)
- Integrated photonic review: [https://doi.org/10.1038/s41566-019-0524-2](https://doi.org/10.1038/s41566-019-0524-2)
- Silicon photonics large-scale: [https://doi.org/10.1038/nphoton.2016.178](https://doi.org/10.1038/nphoton.2016.178)
- Boson sampling: [https://doi.org/10.1145/1806689.1806748](https://doi.org/10.1145/1806689.1806748)
- Why silicon photonics for QC (Rudolph): [https://doi.org/10.1063/1.4976737](https://doi.org/10.1063/1.4976737)
