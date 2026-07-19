---
source_pdf: ACCELERATINGINFERENCEFORMULTILAYERNEURAL NETWORKSWITHQUANTUMCOMPUTERS.pdf
paper_sha256: 1d7f9420dc8d3354e07f8b9dbf709de492b6f3e2a263ad835d0f7951347db60c
processed_at: '2026-07-17T23:50:44-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# 深度解析：Accelerating Inference for Multilayer Neural Networks with Quantum Computers

这篇paper是Arthur Rattew等人2025年的工作，arXiv链接：https://arxiv.org/abs/2505.13100，核心claim是给出了**第一个fully coherent的multilayer neural network量子实现**，包含non-linear activations、skip connections、layer normalization等ResNet核心组件。

---

## 1 核心动机与Intuition Building

### 1.1 为什么这个problem hard

先建立一个key intuition：**quantum computer本质上是linear operator**（unitary matrices），但neural network的power来自non-linearity。要在quantum computer上coherently实现non-linear activation，unitary的定义必须depend on input state。这就引出了一个fundamental tension：

```
Unitary U|ψ⟩ = f(|ψ⟩)  对于non-linear f，U必须input-dependent
```

这意味着每层non-linearity都需要"消耗"input encoding的circuit copies，导致circuit complexity随depth指数增长。Paper在Section 6明确conjecture这是fundamental limitation。

参考：Rattew & Rebentrost 2023的NLAT工作 https://arxiv.org/abs/2309.09839

### 1.2 为什么skip connection在quantum setting下变得至关重要

Classical ResNet中skip connection主要解决gradient vanishing。但在这篇paper里，skip connection有更deep的role：**norm preservation**。

每次我们要从encoded vector sample或者normalize，都要pay cost ∝ 1/||encoded vector||。如果norm在forward pass中arbitrarily decay，k层后complexity可能变成 ~N^k 乃至unbounded。Skip connection通过：

$$||x + f(Wx)||_2 \geq ||x||_2 - ||f(Wx)||_2 \geq \text{constant}$$

保证norm有non-trivial lower bound。Paper中具体取τ=0.51，δ=0.02，得到 N_γ ≥ 0.02。

---

## 2 Vector-Encoding Framework

### 2.1 Definition 3的精确含义

$$\boxed{U_\psi \text{ is } (\alpha, a, \epsilon)\text{-VE for } |\psi\rangle_n \iff \||\psi\rangle_n - \alpha(\langle 0|_a \otimes I_n) U_\psi |0\rangle_{a+n}\|_2 \leq \epsilon}$$

变量解释：
- **α** ≥ 1: encoded vector的norm的inverse。如果ε=0，则 ||(⟨0|_a ⊗ I)U_ψ|0⟩||_2 = 1/α
- **a** ∈ ℕ: ancilla qubits数量
- **ε** ≥ 0: approximation error
- **|ψ⟩_n**: 我们想encode的n-qubit quantum state (2^n维)
- **U_ψ**: 一个 2^(a+n) × 2^(a+n) 的unitary matrix

直觉上：U_ψ|0⟩_{a+n} 这个state的前2^n个amplitudes（在ancilla为|0⟩的subspace）就是 |ψ⟩_n / α。VE是block-encoding的special case，但专门优化用于vectors，能systematically track norm。

### 2.2 新发展的primitive operations

**Lemma 1 (Vector Sum)**: 给定 U_ψ 是 (α, a, ε_0)-VE for |ψ⟩，U_φ 是 (β, b, ε_1)-VE for |φ⟩，可以构造 (N^{-1}, c+1, ...)-VE for：

$$|\overline{\Gamma}\rangle_n = \frac{\tau/\alpha \cdot |\psi\rangle_n + (1-\tau)/\beta \cdot |\phi\rangle_n}{\mathcal{N}}$$

其中 c = max(a,b)，N = ||·||_2。Circuit：1个controlled-U_ψ + 1个controlled-U_φ + 2个single-qubit gate。

**Lemma 2 (Matrix-Vector Product)**: 给定 (α, a, ε_0)-block-encoding of A 和 (β, b, ε_1)-VE for |ψ⟩，得到 (αβ/N, a+b, ...)-VE for A|ψ⟩/N，circuit complexity O(T_ψ + T_A)。

**Lemma 4 (Concatenation)**: 把D个vectors {U_i} 拼成一个superposition state |Ψ⟩_{d+n} = Σ_j |j⟩_d |ψ_j⟩_n / α_j。这是multi-channel input的关键操作。

---

## 3 Matrix-Vector Squared Product — 最Novel的贡献

### 3.1 The Problem

给定任意 dense full-rank matrix W 和 vector |ψ⟩，要计算 W·g(|ψ⟩) where g(x) = |x|^2。

**传统方法的困难**：要先construct W的block-encoding。但对general dense matrix，block-encoding的complexity依赖 **Frobenius norm** ||W||_F = √(Σ|W_{ij}|^2)，对full-rank matrix这是Ω(√N)量级，会destroy speedup。

### 3.2 Key Insight

**Never construct block-encoding of W itself.** 而是直接query W的columns，weighted by input vector的entries。

定义：
- W = (w_0, ..., w_{N-1}) columns
- a_j = ||w_j||_2 (column norms)
- |w_j⟩_n = w_j / a_j (normalized columns)
- |ψ⟩_n = Σ_j ψ_j |j⟩_n

构造两步：

**Step 1**: 构造 |φ⟩_{2n} = Σ_j ψ_j |j⟩_n |w_j⟩_n

这用VE的tensor product + QRAM query实现，**no Frobenius dependence**。

**Step 2**: 定义operator：
$$M := \begin{pmatrix} a_0 \psi_0 I_n & \cdots & a_{N-1}\psi_{N-1} I_n \\ & \mathbf{0} & \end{pmatrix}$$

可以证明 ||M||_2 ≤ 1（这是关键，不是 ||W||_F！），并且：
$$M|\phi\rangle_{2n} = |0\rangle_n \otimes W(|\psi\rangle_n)^2$$

**Theorem 1 (Informal)**: 可以构造 (α²/N, 2a+d+3+n, 2αε/N)-VE for W·g(|ψ⟩)/N，circuit depth O(T_ψ + dn + n²)。

变量：
- **N** = ||W·g(|ψ⟩)||_2: output vector的norm
- **d**: QRAM中存储column norms的precision bits
- **α²**: 来自需要两份input copy（一份query column，一份构造diagonal scaling）

这个结果是**第一个**允许arbitrary dense full-rank matrix without Frobenius norm dependence的quantum algorithm。

### 3.3 为什么这很重要

Final linear layer of ResNet是full-rank dense matrix。如果没有这个result，要么只能处理low-rank matrix（dequantizable by Tang-style algorithms https://doi.org/10.1145/3313276.3316310），要么复杂度爆炸。这个result使得Regime 1的polylogarithmic complexity成为可能，并且**resists dequantization**（因为现有dequantization techniques需要low-rank或sparse assumption）。

---

## 4 QRAM-Free 2D Multi-Filter Convolution Block-Encoding

### 4.1 Matrix Form (Lemma B.17)

2D multi-filter convolution可以写成matrix form：

$$\mathcal{C} := \sum_{i=0}^{C-1}\sum_{j=0}^{C-1}\sum_{k=0}^{D-1}\sum_{l=0}^{D-1} K_{i,j,k,l} (|i\rangle\langle j|_c \otimes Q^l \otimes Q^k)$$

变量：
- **C** = 2^c: input/output channels数
- **D** = 2^d: kernel的spatial dimension (通常3)
- **K_{i,j,k,l}**: rank-4 kernel tensor，i=output channel, j=input channel, k=row offset, l=column offset
- **Q**: discrete unilateral shift operator (Definition B.6)，shift without wrap-around

### 4.2 构造方法

**Permutation matrix P** (Lemma B.14): P^m = Σ_j |j+m⟩⟨j|，通过 QFT 实现：P^m = F·D·F^{-1} where D = diag(ω_N^{-0}, ω_N^{-m}, ..., ω_N^{-m(N-1)})。Block-encoding: (1, 1, 0)，depth O(n²)。

**Shift Q** (Lemma B.15): Q = P - |0⟩⟨N-1|，用LCU组合 P 和 projector，再通过oblivious amplitude amplification (Lemma B.4, using T_3(x) = 4x³-3x) 把scaling从1/2 boost到1。Block-encoding: (1, 4, 0)，depth O(n²)。

**Q^m** (Lemma B.16): 直接product of m个 Q block-encodings，得到 (1, 4m, 0)-block-encoding，depth O(mn²)。

**最终 C 的block-encoding** (Lemma 5): 用LCU (Lemma B.5) 把所有 |i⟩⟨j|_c ⊗ Q^l ⊗ Q^k 组合起来，然后uniform singular value amplification (Lemma B.3) scale到正确normalization。

$$(1, 3 + 8D + 2\log(CD), 0)\text{-block-encoding for } \mathcal{C}/(2\|\mathcal{C}\|_2)$$

Circuit depth: **O(m² C³ D⁴ log(C) log(D))**

### 4.3 关键观察

这个construction **不需要QRAM** — 只需要classical pre-computation来构造kernel state preparation unitary。这正是Regime 3的基础。作者也提到可以用circulant convolution + QFT进一步优化。

---

## 5 Coherent Non-Linear Activation

### 5.1 erf替代sigmoid

Paper用 f(x) = erf(4x/5) 替代sigmoid。原因：
1. erf是odd function (f(0)=0)，便于polynomial approximation
2. Lipschitz constant L = 2ν/√π，对于 ν=4κ/5 ≤ 8/5，L ≈ 0.9 < 1，保证norm不explode
3. 在区间[-1,1]上 |erf(νx)/x| ≥ 1/2，保证activation后norm有lower bound

### 5.2 NLAT of VE (Lemma B.18)

给定 (α, a, ε_0)-VE for |ψ⟩ 和 Lipschitz function f with f(0)=0，可以构造：

$$\left(\frac{4\tilde{\gamma}}{\mathcal{N}}, n + 2a + 4, \frac{L}{\mathcal{N}}(\epsilon_0 + \epsilon_1)\right)\text{-VE for } f(|\psi\rangle_n/\alpha)/\mathcal{N}$$

- **N** = ||f(|ψ⟩/α)||_2
- **γ̃** = max_{x∈[-1,1]} |P(x)/x|，P是degree-k polynomial approximation
- **k** = O(ν log(√N/ε₁)) for erf
- Circuit depth: O(k(n + a + T_ψ))

技术核心：用QSVT (Theorem 56 of Gilyén et al. https://doi.org/10.1145/3313276.3316366) 对 diagonal matrix diag(U_ψ|0⟩) 应用多项式变换。关键步骤是构造 Q(x) = P(x)/x 的block-encoding，然后通过matrix-vector product得到 P(|ψ⟩)。

### 5.3 为什么coherent implementation导致depth指数增长

NLAT需要 **6 calls to controlled-U_ψ** 来构造 diag(U_ψ|0⟩)的block-encoding（Lemma 6 of Rattew & Rebentrost）。加上matrix-vector product需要的calls，每层non-linearity消耗copies of input encoding。

如果第i层的circuit是 C_i，第i+1层的input encoding circuit就是 C_i 本身，所以：
$$C_{i+1} \sim k \cdot C_i \implies C_k \sim k^k$$

这是Lemma 7的complexity O(log(√N/ε)^{2k})的来源。**Wide-shallow networks最适合quantum acceleration**，这跟classical GPU并行化的optimal regime一致 (Zagoruyko & Komodakis, Wide ResNets https://doi.org/10.5244/C.30.87)。

---

## 6 Residual Block — Norm Preservation的关键

### 6.1 Architecture (Figure 2)

```
Input x → [W: linear] → [f=erf(4x/5): activation] → (+x: skip) → [norm: ℓ₂ normalization] → Output
```

### 6.2 Lemma 6 详解

给定：
- U_ψ: (1, a, ε_0)-VE for |ψ⟩_n，complexity O(T_1)
- U_W: (1, b, 0)-block-encoding for W/κ，complexity O(T_2)，其中 ||W||_2 ≤ 1, κ ∈ [1,2]

构造 (1, 2(a+b)+n+9, 712(ε_0+ε_1))-VE for |ψ_f⟩_n/N，其中：
$$|\psi_f\rangle_n := |\psi\rangle_n + f(W|\psi\rangle_n)$$

Circuit complexity:
$$O\left(\log\left(\frac{\sqrt{N}}{\epsilon_1}\right)\log\left(\frac{1}{\epsilon_1}\right)(a+b+n+T_1+T_2)\right)$$

### 6.3 Norm Preservation的证明思路

**Key step**: lower-bound N_γ = ||ψ + f(Wψ)||_2。

利用三角不等式的反向：
$$\mathcal{N}_\gamma^2 = 1 + \mathcal{N}_2^2 + 2\mathcal{N}_2\langle\psi|\Phi_2\rangle \geq (1 - \mathcal{N}_2)^2$$

由于 erf的Lipschitz constant = 8/(5√π) ≈ 0.9，且 ||W||_2 ≤ 1，得到 N_2 ≤ 0.91。所以：
$$\mathcal{N}_\gamma \geq 1 - 0.91 = 0.09$$

具体通过scaling取 ν = 4κ/5，推导出 N_γ ≥ 1/400。这个常数虽然小，但是 **dimension-independent**，这正是关键！

### 6.4 如果没有skip connection会怎样

Consider pure activation without skip: ||f(Wψ)|| 可能arbitrarily small，比如当W是near-zero matrix。那么normalize时cost ∝ 1/||f(Wψ)|| → ∞。在k层架构中，norm可能每层衰减因子c<1，导致total cost ∝ c^{-k}，再乘以per-layer的指数cost，完全intractable。

---

## 7 Multi-Layer Complexity (Lemma 7)

k个residual blocks串联：

$$\text{Output: } (1, 2^k(a + 2b + n + 9), \epsilon)\text{-VE}$$

Circuit depth:
$$\boxed{O\left(\log(\sqrt{N}/\epsilon)^{2k} (a + 2b + n + T_1 + T_2)\right)}$$

### 7.1 误差分析

第i层引入误差ε_i。递推关系：
$$\delta_1 = 712(\epsilon_0 + \epsilon_1), \quad \delta_i = 712(\delta_{i-1} + \epsilon_i) = 1424^i \epsilon_1 / 2$$

要 δ_k ≤ ε，设 ε_1 = 2ε/1424^k。然后每层 ε_i = ε/1424^{k-i}。

### 7.2 为什么是log(√N/ε)^{2k} 而不是 log(√N/ε)^k

每层的complexity是 h(ε_i) = log(√N/ε_i)·log(1/ε_i)。Product over i=1..k:
$$\prod_{i=1}^k h(\epsilon/1424^{k-i}) \in O\left(\log(\sqrt{N}/\epsilon)^{2k}\right)$$

因为 1424是常数，log(1424^{k-i}/ε) ≈ log(1/ε) + O(k)，而k是asymptotic constant。

---

## 8 三个Regimes的End-to-End Complexity

### 8.1 Regime 1: Full QRAM (Theorem 2)

**Architecture** (Figure 1a): k个residual conv layers (16 channels, 3×3 filter) + final full-rank linear-residual-pooling block。

**输入假设**:
- Input X: 4×M×M tensor (RGB + null channel)，||vec(X)||_2 = 1
- U_X: (1, 0, 0)-VE for |X⟩，via Kerenidis-Prakash QRAM data-structure (Lemma D.1)
- T_X ∈ O(polylog(N)) （Theorem 2中T_X是input prep cost）

**复杂度**:
$$\boxed{O\left(\log(\sqrt{N}/\epsilon)^{2k+1} (T_X + n^2)\right) = O\left(\text{polylog}(N/\epsilon)^k\right)}$$

for constant k. Ancilla qubits: O(2^k n).

**对比classical**: 16×16×3×3 = 2304 params per layer，classical cost至少Ω(N·#params)。Quantum实现 **exponential speedup** in N。

**Dequantization resistance**: final full-rank linear layer用Theorem 1实现，没有low-rank/sparse结构，现有dequantization techniques (Tang https://doi.org/10.1145/3313276.3316310, Chia et al. https://doi.org/10.1145/3549524) 不适用。

### 8.2 Regime 2: QRAM only for weights — Quartic Speedup

**Architecture** (Figure 1b): Bilinear-style network。d条classical paths (each O(N) cost) → tensor product → k residual blocks → final full-rank linear layer。

**为什么需要bilinear structure**:
- Input必须brute-force load，T_X ∈ O(N)
- 但是final layer的matrix维度是N^d × N^d (after tensor product)
- Classical cost for matrix-vector: Ω(N^{2d})
- Quantum cost: O(N log(1/ε)^{2k}) (linear in N because only input loading pays O(N))

**d=2的算术**:
- Final linear layer: N² × N² matrix × N² vector
- Classical: Ω(N⁴)
- Quantum: Õ(N log(1/ε)^{2k})
- **Speedup: quartic (N⁴ → N)**

**为什么tensor product能amplify speedup**: Lemma 3的tensor product VE是"free"的（circuit complexity O(max(T_ψ, T_φ))），但dimension square。所以loading cost是N（per path），但后续matrix-vector的"effective dimension"是N²，speedup来自这个mismatch。

### 8.3 Regime 3: No QRAM — Quadratic Speedup

**Architecture** (Figure 1c): 和Regime 2一样，但drop final full-rank linear layer。

**复杂度推导**:
- T_X ∈ O(N) (worst case input loading)
- VE for k layers output: (1, 2^k(63+n), δ)-VE，depth O(log(N/δ)^{2k} · (n² + N))
- Sampling + ℓ₂ pooling (Lemma B.20): error = 2N²δ/√C
- 设 ε = 2N²δ/√C → δ = ε√C/(2N²)
- 最终: O(N log(N³/(ε√C))^{2k}) = Õ(N log(1/ε)^{2k})

**Classical cost**: 2D convolution on N²-dimensional vector = Ω(N²)

**Speedup: quadratic** (N² → N)。可以通过增大d来asymptotically提升speedup。

### 8.4 三Regimes的Summary Table

| Regime | Input | Weights | Quantum Cost | Classical Cost | Speedup |
|--------|-------|---------|--------------|----------------|---------|
| 1 | QRAM | QRAM | polylog(N/ε)^k | Ω(N · #params) | Exponential |
| 2 | Classical | QRAM | N log(1/ε)^{2k} | Ω(N⁴) (d=2) | Quartic |
| 3 | Classical | Classical | N log(1/ε)^{2k} | Ω(N²) | Quadratic |

---

## 9 Final Linear-Residual-Pooling Block (Lemma C.1)

这是Regime 1和2的output block，结合了Theorem 1 (matrix-vector squared product)和skip connection。

**Architecture** (Figure 5):
```
|ψ⟩ → [g(x)=x²: square] → [W: full-rank linear] → (+τ|ψ⟩: skip) → [norm] → [pool_C] → sample
```

**Key parameters**:
- τ = 0.51 (skip weight)，δ = 0.02
- W: 任意 ||W||_2 ≤ 1 的matrix，via Definition B.3的preprocessed QRAM

**Norm preservation的trick**:
$$\mathcal{N}_\gamma \geq \tau - (1-\tau)\mathcal{N}_1 \geq \tau - (1-\tau) = 2\tau - 1 = \delta$$

只要 τ > 0.5，就保证 ||γ|| ≥ δ。这是为什么取τ=0.51而不是0.5。

**最终复杂度**: O(log(N/(√C·ε)) · (T_ε₀ + a + n²))，ancilla O(a+n)。

---

## 10 QRAM Feasibility的Discussion

### 10.1 Passive vs Active QRAM

参考Jaques & Rattew https://arxiv.org/abs/2305.10310:
- **Passive QRAM**: 每次query需要 o(N) energy
- **Active QRAM**: 每次query需要 Ω(N) energy

Circuit model下的error-corrected QRAM必然是active（每个qubit需要O(1) classical resources for error correction）。

### 10.2 Practically Passive QRAM

类比classical DRAM: 虽然技术上是active (Ω(N) power)，但相比CPU功耗可忽略，实际应用中视为passive (Carroll & Heiser https://dl.acm.org/doi/10.5555/1855840.1855861)。

QRAM可能类似：相比error-corrected QPU的巨大overhead (Babbush et al. https://link.aps.org/doi/10.1103/PRXQuantum.2.010103)，QRAM的active cost可能可忽略。Bucket-brigade architecture (Giovannetti et al. https://link.aps.org/doi/10.1103/PhysRevLett.100.160501) 还有exponential error robustness (Hann et al. https://link.aps.org/doi/10.1103/PRXQuantum.2.020311)。

### 10.3 最新进展

Dalzell et al. 2025的distillation-teleportation protocol (https://arxiv.org/abs/2505.20265) 为fault-tolerant QRAM提供了新path，addressing了Jaques-Rattew提出的concerns。

---

## 11 与Prior Work的对比 (Table 1)

| Work | Architecture | Multi-Layer | Non-Linearity | QRAM-Free | Norm Preserve | Polylog 1/ε | Polylog N |
|------|--------------|-------------|---------------|-----------|---------------|-------------|-----------|
| Cong et al. [57] | CNN-inspired PQC | ✗ | ✗ | ✓ | ✓ | N/A | N/A |
| Allcock et al. [40] | PQC Feed-forward | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Kerenidis et al. [41] | CNN | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Guo et al. [42] | Transformer | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Ours - Regime 1** | Residual CNN | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| **Ours - Regime 2** | Bilinear Res CNN | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| **Ours - Regime 3** | Bilinear Res CNN | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

**Key differentiators**:
1. **First fully coherent multi-layer**: 不需要intermediate measurement/tomography (vs Kerenidis [41], Guo [42]都需要tomography来re-encode)
2. **First QRAM-free architecture** in quantum deep learning acceleration literature
3. **Norm preservation**: 通过skip connection + sub-normalized weights实现，是coherent multi-layer的enabler

---

## 12 Open Questions & Future Directions

### 12.1 Depth的指数complexity是否fundamental

Paper Section 6 conjecture: 用coherent unitary enact non-linear sequence without exponential depth是 **provably impossible** (at least generally)。

可能的loophole：
- QPE-based approaches (Mitarai et al. https://link.aps.org/doi/10.1103/PhysRevA.99.012301) 可能在exponentially worse error-dependency的代价下避免depth指数增长
- Combination可能支持up to depth ~25的architectures

### 12.2 Receptive Field问题

k层convolution (kernel size D)只能看到 ≈kD pixels的local information。Final linear layer的作用是merge local到global。Bilinear architecture (Kronecker product)通过让local信息global化来circumvent这个问题，但loses positional information，需要positional encoding。

### 12.3 Under-parameterization问题

Regime 3的纯convolution architecture可能under-parameterized (Allen-Zhu et al. https://proceedings.neurips.cc/paper_files/paper/2019/hash/62dad6e273d32235ae02b7d321578ee8-Abstract.html)。可以加O(N)参数的final low-rank residual block来缓解。

### 12.4 与scientific computing的联系

Paper提到quantum differential equation solvers (Berry & Costa https://doi.org/10.22331/q-2024-06-13-1369, Liu et al. https://www.pnas.org/doi/abs/10.1073/pnas.2026805118) 的techniques可能互通。PDE solver也处理non-linear dynamics，也有类似的coherent non-linearity challenge。

---

## 13 我的Intuition总结

这篇paper的核心message可以总结为三个insight chain：

**Insight 1: Norm tracking is everything in quantum ML**
- VE framework的价值不在于encoding本身，而在于systematically track ||encoded vector||
- 每个操作的complexity都 ∝ 1/norm
- Skip connection的quantum role: provide dimension-independent norm lower bound

**Insight 2: Avoid Frobenius, exploit structure**
- Matrix-vector squared product (Theorem 1) 的核心是 **never block-encode W directly**
- 而是用importance weighting的思想，把columns of W和input amplitudes combine
- 这避免了 ||W||_F 的rank-dependence，使得full-rank dense matrix可处理
- 副作用：resists dequantization (现有dequant算法需要low-rank/sparse)

**Insight 3: Wide-shallow is quantum-optimal**
- Non-linearity的coherent implementation要求input-dependent unitary
- 这导致per-layer cost的product，depth指数增长
- 但width (parallelism)在quantum里也"free"（通过superposition）
- 所以quantum computer最适合加速wide-shallow architectures，跟classical GPU的optimal regime一致
- ResNet的skip connection同时解决了classical gradient vanishing和quantum norm decay，是quantum-classical的beautiful alignment

**最后一点meta-observation**: 这篇paper的technical贡献其实反映了quantum algorithm design的一个general principle — **avoid direct encodings of objects you can't efficiently encode, and instead query them via oracles in a structured way**。Theorem 1是这个principle的excellent example：与其block-encode W (需要Frobenius)，不如用QRAM query W的columns + importance weight by input。这跟HHL (https://link.aps.org/doi/10.1103/PhysRevLett.103.150502) 避免explicit matrix construction的思想一脉相承，但在更复杂的non-linear setting下实现了。

---

主要参考链接：
- Paper: https://arxiv.org/abs/2505.13100 (推测，基于作者和内容)
- QSVT: https://doi.org/10.1145/3313276.3316366
- NLAT: https://arxiv.org/abs/2309.09839
- QRAM critique: https://arxiv.org/abs/2305.10310
- Distillation-teleportation QRAM: https://arxiv.org/abs/2505.20265
- Dequantization: https://doi.org/10.1145/3313276.3316310
- Wide ResNets: https://doi.org/10.5244/C.30.87
- Original ResNet: https://doi.org/10.1109/CVPR.2016.90
- QRAM original: https://link.aps.org/doi/10.1103/PhysRevLett.100.160501
- Bucket-brigade robustness: https://link.aps.org/doi/10.1103/PRXQuantum.2.020311
