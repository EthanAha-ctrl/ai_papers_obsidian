---
source_pdf: A FULL STACK FRAMEWORK FOR HIGH PERFORMANCE.pdf
paper_sha256: d3150bf5d538d8ebdaf4213f83fc1ecc46e9cf348541eb672546c8969e87d872
processed_at: '2026-07-17T19:52:34-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# HPE HPC-QC Full Stack 框架深度解析

这篇 paper 来自 HPE 团队, 提出了一个完整的 hybrid HPC-QC (High Performance Computing - Quantum Computing) 全栈框架, 旨在将 QPU (Quantum Processing Unit) 作为 accelerator 集成到现有 HPC 基础设施中, 与 GPU, FPGA 并列。让我逐层深入讲解。

---

## 1. 核心问题与设计哲学

### 1.1 核心挑战

HPC-QC integration 面临三大维度挑战:

1. **Compatibility (兼容性)**: 现有 quantum SDK 多采用 Python-based DSL (Domain-Specific Language), 而 HPC 应用主要是 C/C++/Fortran。需要在语言层面桥接
2. **Performance (性能)**: 随 circuit size 增长 (~100 qubits+), declarative framework 出现 compilation bottleneck 与 classical-quantum latency
3. **Scalability (可扩展性)**: NISQ (Noisy Intermediate-Scale Quantum) era 设备 qubit 数有限, 需要 partition + distribute

### 1.2 三层架构

![](https://qir-alliance.org/assets/images/qir-alliance_org.png)

整个 stack 分为三层 extension (对应 Figure 1):

| 层次 | 组件 | 主要功能 |
|------|------|---------|
| 上层 | Quantum Interface Library | 高层 API 调用 quantum kernel |
| 中层 | Adaptive Circuit Knitting (ACK) Hypervisor | 电路 partition 与 workload distribution |
| 下层 | Quantum Compiler Extension | LLVM-based 代码生成与 QIR 消费 |

参考架构灵感: [Mohseni et al., "How to build a quantum supercomputer", arXiv:2411.10406](https://arxiv.org/abs/2411.10406)

---

## 2. Quantum Interface Library 深度解析

### 2.1 设计核心

该库用 **C 语言**实现 (而非 C++), 这样可以:
- 提供低层、高效、稳定的 ABI (Application Binary Interface)
- 同时被 C/C++ 和 Fortran 调用 (通过 `ISO_C_BINDING`)
- Dynamic linking 允许 quantum SDKs 独立更新, 无需 recompile

工作流程 (Figure 2):
```
HPC App (C/C++/Fortran) 
   ↓ quantum API call
Quantum Interface Library (C)
   ↓ routing + data marshaling
Vendor Quantum SDK (Python DSL, e.g., Classiq)
   ↓
OpenQASM 2/3
   ↓
Gate-based QPU or Simulator (cloud/on-prem)
```

### 2.2 Hybrid MPI 执行模型

这是该 paper 最关键的工程创新之一 (Figure 3):

```
┌─────────────────────┐    ┌─────────────────────┐
│  Classical MPI rank │    │  Quantum MPI rank   │
│  - HPC computation  │◄──►│  - Circuit synthesis │
│  - CSML libraries   │MPI │  - Execution         │
│  - Setup/invoke Q   │msg│  - Result return     │
└─────────────────────┘    └─────────────────────┘
        ↑                          ↑
   异步执行                  offload 到 simulator/QPU
```

优势:
- **并行效率**: quantum 处理可异步 offload
- **Scalability**: classical 与 quantum 进程独立 scale 到多个 CPU/GPU/QPU 节点
- **Modularity**: 调试方便, backend 可热插拔

### 2.3 SLURM Job 拆分策略

论文另一个创新 (Figure 4): 将 monolithic hybrid job 拆分为 basic building blocks, 通过 SLURM dependencies 链接。

参考: [Esposito & Haus, "Slurm heterogeneous jobs for hybrid classical-quantum workflows", arXiv:2506.03846](https://arxiv.org/abs/2506.03846)

原始 monolithic job 中, quantum 资源在 classical 长时间运行期间会 idle。拆分后:
```
Job_i_1 (classical) ──► Job_i_2 (quantum) ──► Job_i_3 (classical post-process)
                          ↑
                   QPU 只在此阶段占用
```

---

## 3. Case A: HHL Algorithm 求解线性方程组

### 3.1 数学公式详解

**问题**: 求解 $A\vec{x} = \vec{b}$, 其中 $A$ 为 $N \times N$ matrix, $\vec{b}$ 为 size $N = 2^n$ 的向量。

**量子化**: 将 $\vec{b}$ 编码为归一化 quantum state $|b\rangle \in \mathbb{C}^N$, 目标制备:
$$|x\rangle \propto A^{-1}|b\rangle$$

变量解释:
- $A^{-1}$: matrix $A$ 的逆
- $|b\rangle$: 通过 amplitude encoding 将 classical vector $\vec{b}$ 编码的 quantum state
- $|x\rangle$: 与 classical 解 $\vec{x}$ 成比例的 quantum state
- $N = 2^n$: $n$ 为 qubit 数, $N$ 为 Hilbert space 维度

### 3.2 HHL 算法 4 步

**Step 1**: State preparation of $\vec{b}$
$$|b\rangle = \sum_{i=0}^{N-1} b_i |i\rangle$$

**Step 2**: QPE (Quantum Phase Estimation) for unitary $U = e^{2\pi i A}$
$$U|u_j\rangle = e^{2\pi i \lambda_j}|u_j\rangle$$

其中 $\lambda_j$ 是 $A$ 的 eigenvalue, $|u_j\rangle$ 是对应 eigenvector。QPE 将 $\lambda_j$ 编码到 ancilla register (size $m$) 的相位上:
$$\sum_j \beta_j |u_j\rangle \otimes |\lambda_j\rangle_{\text{phase reg}}$$

**Step 3**: Eigenvalue inversion
通过 controlled rotation 将 $|\lambda_j\rangle$ 映射为 $1/\lambda_j$ 的 amplitude:
$$\sum_j \beta_j |u_j\rangle \otimes (c_j|1/\lambda_j\rangle + \sqrt{1-c_j^2}|0\rangle)$$

**Step 4**: Inverse QPE
应用 $\text{QPE}^\dagger$ 解开 phase register, 得到:
$$|x\rangle \propto \sum_j \frac{\beta_j}{\lambda_j}|u_j\rangle = A^{-1}|b\rangle$$

### 3.3 Pauli String 分解

预处理将 $A$ 分解为 Pauli strings 之和:
$$A = \sum_k c_k P_k$$
其中 $P_k \in \{I, X, Y, Z\}^{\otimes n}$, $c_k$ 为复系数。

论文中 4×4 矩阵的 10 个 Pauli terms (Figure 5):
| Pauli String | 系数 |
|-------------|------|
| $II$ | $0.408$ |
| $IZ$ | $-0.052$ |
| $IX$ | $-0.03$ |
| $ZI$ | $-0.017$ |
| $ZZ$ | $-0.057$ |
| $ZX$ | $0.02$ |
| $XI$ | $-0.025$ |
| $XZ$ | $0.045$ |
| $XX$ | $-0.16$ |
| $YY$ | $-0.06$ |

每个矩阵 entry 独立贡献 Pauli expansion (naive decomposition)。

### 3.4 实验结果

- 硬件: HPE Cray EX system, 2 nodes, 每节点 2× AMD EPYC 7763 (Milan) CPU
- 规模: 测试到 64×64 matrix
- 精度: HHL 解与 BLAS 解相对误差 **1.8%**
- Backend: Qiskit Aer statevector simulator
- SDK: Classiq

参考: [HHL original paper, Phys. Rev. Lett. 103, 150502](https://doi.org/10.1103/PhysRevLett.103.150502); [Classiq platform](https://platform.classiq.io/); [Qiskit Aer](https://www.ibm.com/quantum/qiskit)

---

## 4. Case B: QAOA 求解 MaxCut

### 4.1 QAOA Ansatz 公式

$$|\psi(\gamma, \beta)\rangle = \prod_{j=1}^{p} e^{-i\beta_j H_M} e^{-i\gamma_j H_C} |+\rangle^{\otimes n}$$

变量逐项解释:
- $|\psi(\gamma, \beta)\rangle$: 经过 $p$ 层 (depth $p$) 的 QAOA ansatz state
- $\prod_{j=1}^{p}$: 第 $j$ 层从 1 到 $p$ 顺序应用 (注意: $j=1$ 最先作用还是最后作用取决于 convention)
- $\gamma = (\gamma_1, \ldots, \gamma_p)$: cost Hamiltonian 的 variational parameters, 下标 $j$ 表示第 $j$ 层
- $\beta = (\beta_1, \ldots, \beta_p)$: mixer Hamiltonian 的 variational parameters
- $H_C$: cost Hamiltonian, 编码优化问题。对 MaxCut:
$$H_C = \sum_{(i,j) \in E} \frac{1}{2}(I - Z_i Z_j)$$
  其中 $E$ 是图的 edge set, $Z_i$ 是作用在 qubit $i$ 上的 Pauli-Z
- $H_M = \sum_{i=1}^{n} X_i$: mixer Hamiltonian, 是所有 Pauli-X 的和, $X_i$ 作用在 qubit $i$
- $|+\rangle^{\otimes n} = (|0\rangle + |1\rangle)^{\otimes n}/\sqrt{2^n}$: 初始等叠加态, $n$ 为 qubit 数
- 上标 $\otimes n$: tensor product 重复 $n$ 次

### 4.2 优化目标

$$\max_{\gamma, \beta} \langle \psi(\gamma, \beta) | H_C | \psi(\gamma, \beta) \rangle$$

这是 expectation value 最大化, 通过 classical optimizer (如 COBYLA, Nelder-Mead, SPSA) 迭代更新 $(\gamma, \beta)$。

### 4.3 QAOA² (QAOA-in-QAOA) 分治策略

对于 500-10000 节点的大图, 单个 QAOA circuit 不可行。QAOA² 采用:

```
大图 G ──► 图分割 (classical) ──► 子图 G1, G2, ..., Gk
                                      ↓
                              并行 QAOA on 各子图
                                      ↓
                              classical rank 0 汇总
                                      ↓
                              community-level MaxCut merging
```

每个 quantum process 独立模拟一个 QPU, MPI 通信异步回传结果。

参考: [Zhou et al., "QAOA-in-QAOA", Phys. Rev. Applied 19, 024027](https://doi.org/10.1103/PhysRevApplied.19.024027)

### 4.4 实验结果

- 规模: 节点数 up to 2500
- 资源: up to 512 CPU nodes on HPE Cray EX
- Benchmark 对比: Goemans-Williamson (GW), Greedy, Random
- 结果: Greedy 和 GW 在小图上仍强; QAOA 在更大 structured graphs 上表现更好
- ISC24 live demo: 20-qubit IQM quantum device accessed from LUMI (Finland)

参考: [QAOA original, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028); [LUMI supercomputer](https://www.lumi-supercomputer.eu/); [IQM quantum computers](https://www.meetiqm.com/)

---

## 5. Adaptive Circuit Knitting (ACK) Hypervisor

这是该 paper **最具技术深度的部分**, 也是 HPE 与 NVIDIA 合作的核心创新。

### 5.1 传统 Circuit Knitting 的局限

[Reference: Peng et al., PRL 125, 150504 (2020)](https://doi.org/10.1103/PhysRevLett.125.150504)

传统 circuit knitting 将大 circuit 切割为 sub-circuits, 独立执行, 然后 classical post-processing 重组 observables。但**采样开销指数增长**:

$$\text{Sampling overhead} \sim O(\kappa^{2k})$$

其中:
- $\kappa$: 与 cut 处 entanglement 相关的 contraction factor, $\kappa = \sum_i \sqrt{p_i^2 + q_i^2}$, $p_i, q_i$ 是 cut gate 的 Schmidt 系数
- $k$: cut 数量

当 cut 处 entanglement 高时, $\kappa$ 大, 开销爆炸。

### 5.2 ACK 核心思想

ACK 通过 **tensor network (TN)**, 特别是 **Matrix Product State (MPS)**, 分析 quantum state 的 entanglement 结构, 选择**最小 entanglement entropy** 的 cut 点。

MPS 表示:
$$|\psi\rangle = \sum_{i_1, \ldots, i_n} A^{i_1} A^{i_2} \cdots A^{i_n} |i_1 i_2 \cdots i_n\rangle$$

其中 $A^{i_k}$ 是 site $k$ 的 local tensor, bond dimension $\chi$ 反映 entanglement:
$$S_{\text{ent}} \leq \log_2 \chi$$

Cut 在 $\chi$ 小的位置 (entanglement 弱), 可大幅降低 $\kappa$。

### 5.3 双层循环架构 (Figure 7 top)

```
┌────────────────────────────────────┐
│   Outer Loop: 切割策略 refinement   │
│   - 基于 entanglement minimization  │
│   - 更新 cut locations              │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│   Inner Loop: sub-circuits 并行优化 │
│   - 每个子电路独立执行             │
│   - TN-inspired 结构               │
└────────────────────────────────────┘
```

### 5.4 应用: 1D Ising Spin Chain

Hamiltonian:
$$H = -J \sum_{i} Z_i Z_{i+1} - h_x \sum_i X_i - h_z \sum_i Z_i$$

- $J$: nearest-neighbor Ising coupling
- $h_x$: transverse field (沿 X)
- $h_z$: longitudinal field (沿 Z)
- $Z_i, X_i$: Pauli operators at site $i$

强无序 (strongly disordered) 情形会展现 **many-body localization (MBL)**, 经典模拟困难。

### 5.5 实验数据

| 规模 | 平台 | 资源 |
|------|------|------|
| 32 qubits | single node | 4× A100 GPU (40GB HBM2) |
| 40 qubits | Perlmutter (NERSC) | 256 nodes × 4 A100 = 1024 GPUs |

- SDK: [NVIDIA CUDA-Q](https://developer.nvidia.com/cuda-q) with GPU-accelerated simulator backends (statevector, density matrix, tensor networks)
- 物理后端支持: Quantinuum, IonQ, IQM, OQC
- **采样开销降低**: 相比 baseline load-balanced cuts, **多数 case 10-100x**, 个别 case **>1000x**

参考: [Mohseni, "How to build a distributed quantum computer: ACK", NVIDIA GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-dd73669/); [Perlmutter at NERSC](https://www.nersc.gov/systems/perlmutter/)

---

## 6. Quantum Compiler Extension

### 6.1 设计三原则

**原则 1: Integrated Quantum Co-Processor Programming Model**
QPU 作为 accelerator, 与 CPU/GPU 并列。编译时即生成 hybrid binary, offload quantum kernel 到 QPU。

**原则 2: Frontend Agnosticism via QIR**
[QIR (Quantum Intermediate Representation)](https://qir-alliance.org/) 由 Microsoft 主导, 基于 LLVM IR, 是连接多种 quantum 语言与 hardware backend 的通用接口。

支持 frontends:
- CUDA-Q (NVQ++)
- Q#
- OpenQASM-based toolchains

**原则 3: Hardware-Retargetable Code Generation**
通过 Cray LLVM 编译框架 consume LLVM IR + QIR, 生成不同 target architecture 的 assembly, 无需 nonstandard extensions。

### 6.2 QIR 代码解析 (Figure 8)

```llvm
# QIR types and intrinsics
%Array = type opaque           ; quantum array (e.g., qubit register)
%Qubit = type opaque           ; single qubit
%Result = type opaque          ; measurement result (0/1)

declare %Array* @__quantum__rt__qubit_allocate_array(i64)
declare void @__quantum__qis__h(%Qubit*)
declare void @__quantum__qis__x__ctl(%Array*, %Qubit*)
declare %Result* @__quantum__qis__mz(%Qubit*)
```

命名规则解读:
- `__quantum__rt__`: runtime 层 (memory management)
- `__quantum__qis__`: quantum instruction set (gate operations)
- `h`: Hadamard gate
- `x__ctl`: controlled-X (CNOT)
- `mz`: measurement in Z basis

GHZ kernel 流程:
```
entry:
  %qvec = qubit_allocate_array(n_qubits)   ; 分配 n 个 qubit
  %q0 = qvec[0]                            ; 取第 0 个
  h(%q0)                                   ; Hadamard on q0
  br label %loop                           ; 进入循环

loop:
  %i = phi [0, entry], [%next_i, loop_body]  ; 循环计数器
  %cond = %i < (n_qubits - 1)                ; 终止条件
  br %cond, loop_body, measure

loop_body:
  ; CNOT qvec[i] -> qvec[i+1]
  ; 建立 GHZ 纠缠链
```

### 6.3 实验演示

- 程序: C++ hybrid application 嵌入 CUDA-Q quantum kernel
- 任务: 30-qubit GHZ state 创建与测量
$$|\text{GHZ}\rangle = \frac{|0\rangle^{\otimes 30} + |1\rangle^{\otimes 30}}{\sqrt{2}}$$
- 编译: Cray LLVM codegen, lower QIR 为 binary object
- 链接: CUDA-Q runtime libraries
- 执行: single A100 GPU

参考: [GHZ state creation, npj Quantum Inf 7:24](https://doi.org/10.1038/s41534-021-00364-8); [LLVM-based quantum compiler, Quantum Sci. Technol. 5(3)](https://doi.org/10.1088/2058-9565/ab8c2c); [Chong et al., Nature 549:180](https://doi.org/10.1038/nature23459)

---

## 7. 关键技术联想与扩展

### 7.1 与 HPC 生态的深度整合

该 framework 的核心 insight: **不要从零造 quantum software stack, 而是嵌入成熟 HPC toolchain**。

涉及到的 HPC 组件:
- **MPI** (Message Passing Interface): [MPICH](https://www.mpich.org/), Cray MPICH
- **SLURM** workload manager: [SchedMD](https://slurm.schedmd.com/)
- **Cray Scientific and Math Libraries (CSML)**: [CSML docs](https://cpe.ext.hpe.com/docs/24.07/csml/index.html)
- **Cray LLVM-based Compilation Environment (CPE)**

### 7.2 与其他 Quantum-HPC 项目的对比

- [QICK (Quantum Instrumentation Control Kit)](https://github.com/openquantumhardware/qick) — 偏硬件控制
- [QIR Alliance](https://qir-alliance.org/) — IR 标准
- [NVIDIA CUDA-Q](https://developer.nvidia.com/cuda-q) — GPU-accelerated simulation
- [Classiq](https://platform.classiq.io/) — high-level synthesis
- [IBM Qiskit](https://www.ibm.com/quantum/qiskit) — SDK 与 runtime

### 7.3 Circuit Knitting 与 Tensor Network 深层联系

ACK 方法本质是**动态 tensor network slicing**:

- MPS 的 bond dimension $\chi$ 是 entanglement 的几何度量
- cut 在 $\chi$ 小处, 类似 SVD 截断低奇异值
- 与 [DMRG (Density Matrix Renormalization Group)](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group) 的 truncation 哲学相通
- 与 [PEPS (Projected Entangled Pair States)](https://arxiv.org/abs/cond-mat/0404466) 的 2D 扩展类似

### 7.4 HHL 实际性能的局限

HHL 的 "exponential speedup" 有重要 caveat:
1. Matrix $A$ 必须 sparse 或有 efficient quantum oracle
2. 输出 $|x\rangle$ 是 quantum state, 无法直接读出所有 entries
3. Condition number $\kappa(A)$ 影响 depth
4. 实际与 [VQLS (Variational Quantum Linear Solver)](https://arxiv.org/abs/1909.05820) 比较, 后者更适合 NISQ

### 7.5 QAOA 性能理论

QAOA 在 $p \to \infty$ 时趋近 QAOA-optimal:
$$\lim_{p \to \infty} \max_\gamma \langle H_C \rangle = \text{optimal}$$

对于 MaxCut on 3-regular graph, $p=1$ QAOA 的 approximation ratio $\geq 0.693$ (vs GW 的 0.878)。但 $p$ 增大带来 circuit depth 与 noise。

参考: [Farhi et al. original QAOA](https://doi.org/10.48550/arXiv.1411.4028); [Goemans-Williamson algorithm](https://doi.org/10.1145/227683.227684)

### 7.6 NISQ 时代的工程现实

[Preskill, "Quantum computing in the NISQ era and beyond"](https://doi.org/10.22331/q-2018-08-06-79) 提出的 NISQ 概念, 决定了当前 framework 的设计:

- 无 full error correction
- ~100-1000 qubits
- 高 noise rate
- 短 circuit depth
- Variational algorithms 主导 (QAOA, VQE)

ACK 与 QAOA² 都是 NISQ-aware 设计。

### 7.7 异构 QPU 编排挑战

论文提到但未深入的 critical issue:

| QPU modality | Clock (Hz) | Connectivity | Noise | Long-range |
|-------------|-----------|--------------|-------|------------|
| Superconducting | ~GHz | 限制 | 中 | 弱 |
| Trapped ion | ~MHz | 全连接 | 低 | 强 |
| Neutral atom | ~kHz-Hz | 可重构 | 低 | 中 |
| Photonic | ~THz | 弱 | 低 | 弱 (除非 measurement-based) |

异构 QPU 需要:
- 异步 scheduling
- 不同 QEC (Quantum Error Correction) codes
- 不同 real-time decoding 速率

参考: [Krantz et al., "A quantum engineer's guide to superconducting qubits", Appl. Phys. Rev. 6, 021318](https://doi.org/10.1063/1.5089550); [Akhtar et al., "quantum matter-link between ion-trap modules", Nat. Commun. 14:531](https://doi.org/10.1038/s41467-022-35285-3)

---

## 8. 我的 Intuition 总结

这篇 paper 的核心 intuition 是: **quantum computing 不会独立存在, 它将作为 HPC 生态中的一种 specialized accelerator**。

类比历史:
- 1990s: CPU-only
- 2000s: CPU + GPU (CUDA 革命)
- 2010s: CPU + GPU + FPGA + TPU
- 2020s+: CPU + GPU + FPGA + **QPU**

每一代 accelerator 集成都需要:
1. **Programming model** (论文: quantum interface library, 类比 CUDA runtime)
2. **Workload distribution** (论文: ACK hypervisor, 类比数据并行 / 任务并行)
3. **Compiler IR** (论文: QIR on LLVM, 类比 NVVM IR / SPIR-V)

ACK 的 tensor network-guided cutting 是真正 novel 的部分: 它把 quantum entanglement 的物理结构直接用作 partitioning 的指导信号, 类似把 distributed computing 的 graph partitioning 替换为 entanglement-aware partitioning。

**未来的关键开放问题**:
1. Heterogeneous QPU 间的 quantum interconnect 何时实用化
2. Real-time QEC decoding 的 HPC 集成
3. Quantum memory 的 persistency 与 classical cache hierarchy 的整合
4. LLVM-based QIR 是否会成为 de facto standard (vs Qiskit transpiler, Cirq, etc.)
5. Circuit knitting 的经典 post-processing 在百万 qubit 时代是否仍 feasible

完整技术报告与 follow-up: [Mohseni et al., "How to build a quantum supercomputer: Scaling from hundreds to millions of qubits", arXiv:2411.10406](https://arxiv.org/abs/2411.10406)

---

## 参考链接汇总

**Core references:**
- [HHL Algorithm - PRL 103, 150502](https://doi.org/10.1103/PhysRevLett.103.150502)
- [QAOA - arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028)
- [QAOA² - Phys. Rev. Applied 19, 024027](https://doi.org/10.1103/PhysRevApplied.19.024027)
- [Circuit Knitting - PRL 125, 150504](https://doi.org/10.1103/PhysRevLett.125.150504)
- [NISQ era - Quantum 2, 79](https://doi.org/10.22331/q-2018-08-06-79)

**Frameworks & SDKs:**
- [NVIDIA CUDA-Q](https://developer.nvidia.com/cuda-q)
- [Classiq Platform](https://platform.classiq.io/)
- [IBM Qiskit](https://www.ibm.com/quantum/qiskit)
- [QIR Alliance](https://qir-alliance.org/)
- [OpenQASM](https://openqasm.com/)

**HPC infrastructure:**
- [HPE Cray CSML](https://cpe.ext.hpe.com/docs/24.07/csml/index.html)
- [SLURM](https://slurm.schedmd.com/)
- [LUMI Supercomputer](https://www.lumi-supercomputer.eu/)
- [Perlmutter at NERSC](https://www.nersc.gov/systems/perlmutter/)

**Hardware partners:**
- [IQM Quantum](https://www.meetiqm.com/)
- [Quantinuum](https://www.quantinuum.com/)
- [IonQ](https://ionq.com/)
- [OQC (Oxford Quantum Circuits)](https://oxfordquantumcircuits.com/)

**Follow-up works:**
- [Mohseni et al., arXiv:2411.10406 - Scaling to millions of qubits](https://arxiv.org/abs/2411.10406)
- [Esposito & Haus, arXiv:2506.03846 - Slurm heterogeneous jobs](https://arxiv.org/abs/2506.03846)
- [Mohseni, NVIDIA GTC 2025 - ACK session](https://www.nvidia.com/en-us/on-demand/session/gtc25-dd73669/)
