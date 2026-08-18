---
source_pdf: Fast and Parallel High-Rate STAR Architecture for Megaquop Quantum Simulation.pdf
paper_sha256: 967f2ec971e1e8b8f65d867ff54c755c07246d71b54b6e79dd462d112c560c5a
processed_at: '2026-08-18T12:26:45-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这 paper 到底在干嘛

## 一句话

这 paper 说的是：**别用 surface code 跑量子模拟了，我们造了个新的 code，叫 bicycle chain code，码率高 5 倍多，跑一样的模拟只要两千多个物理 qubit、两百秒一 shot，比 surface code 方案省 5.5 倍 qubit**。

## 背景：为什么是 quantum simulation 先跑出来

Quantum computer 要实用，得先挑一个"精度要求不高但量子优势明显"的任务。Quantum simulation of many-body system 就是这个 sweet spot。原因很简单：你模拟一个 $8\times 8$ 的 lattice model，Trotter 出来的 circuit 大概是 megaquop 规模（$\sim 10^6$ 个逻辑 gate），每个逻辑 gate 的 error rate 只要 $\sim 10^{-6}$ 就够 beat classical。对比一下，跑 Shor 算法 factor RSA-2048 要 error rate $\sim 10^{-9}$ 甚至 $10^{-15}$，差了三个数量级。

所以 quantum simulation 是"早期 fault-tolerant"的天然 candidate：你不需要 fully fault-tolerant 的大 code distance（$d \sim 30$），$d \sim 6$ 就够了。

## 传统 FTQC 的三层 tax

传统路线是"先选一个通用 code（surface code），再在上面跑 universal gate set（Clifford + T），T gate 用 magic state distillation 造"。这路线能跑任何 algorithm，但三层 overhead 叠在一起很贵：

1. **Encoding overhead**：surface code 一个 logical qubit 用 $d^2$ 个 physical qubit。$d=9$ 就是 81 个，$d=15$ 就是 225 个。Encoding rate $1/d^2$，低得可怜。
2. **Gate overhead**：逻辑 gate 的 fault-tolerant 实现要额外 space-time。Lattice surgery 要 $O(d^3)$，transversal 好一点 $O(d^2)$。
3. **Synthesis overhead**：Trotterized simulation 里大量小角度 $R_Z(\theta)$ rotation，$\theta \sim 10^{-2}\text{-}10^{-3}$ rad。每个 rotation 要拆成几十个 T gate 来 synthesize，因为 T gate 是 Clifford+T 的 non-Clifford component。

第三层尤其 wasteful：你的 simulation 根本不需要精确的 T gate，它需要的是一个小角度 rotation，你硬要把它拆成 T gate 再合成回来，中间损耗巨大。

## STAR 的核心 idea：直接做小角度 rotation magic state

STAR（Space-Time efficient Analog Rotation）的 insight 很简单：**别 synthesize 了，直接 prepare**。

你想要 $R_Z(\theta)$ gate，它对应的 magic state 是 $|m_\theta\rangle = \cos(\theta/2)|+\rangle - i\sin(\theta/2)|-\rangle$。STAR 直接在 code block 里 prepare 这个 state（通过一个叫 TMR 的 protocol + post-selection），然后 teleport 到 computation 里实现 rotation。

核心公式是 logical error rate：

$$\varepsilon \sim \alpha \, p \, |\theta|$$

- $\alpha \sim 3$：protocol-dependent 常数
- $p$：physical error rate（$\sim 10^{-3}$）
- $\theta$：rotation angle（$\sim 10^{-3}$ rad）

所以 $\varepsilon \sim 3 \times 10^{-3} \times 10^{-3} = 3 \times 10^{-6}$ per gate。Megaquop 够用。**synthesis overhead 整个砍掉了**。

原始 STAR paper：[Akahoshi et al., PRX Quantum 5, 010337 (2024)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010337)

## 这篇 paper 要解决的问题

STAR 好是好，但之前都基于 surface code，encoding rate $O(1/d^2)$ 还是太低。理论上高码率 qLDPC code（encoding rate 可以到 $1/7$ 甚至更高）能大幅省 qubit。但把 STAR 搬到 qLDPC code 上有两个 hard problem：

**Problem 1：Gate matching**。qLDPC code 的 native logical gate 要 match 你 algorithm 需要的 gate。你不能随便选一个 qLDPC code，指望它恰好支持你 simulation 需要的 Clifford gate set。

**Problem 2：Parallel injection**。qLDPC code 一个 block 编码 $k$ 个 logical qubit（$k \sim 8\text{-}16$）。如果你串行 inject magic state，其他 $k-1$ 个 logical qubit 闲置，而且 post-selection rate 随 $k$ 指数衰减：$p_{\text{success}}^k$，$p \sim 0.4$ 的话 $k=8$ 就只剩 $0.4^8 \sim 6 \times 10^{-4}$，完全不可行。

这篇 paper 用 **symmetry-driven co-design** 同时解决这两个 problem。核心 idea：target Hamiltonian 的 translation symmetry 像一条主线，贯穿 algorithm、QEC code、hardware 三层，每一层都用这个 symmetry 降低复杂性。

## 三层 Co-design：一个 symmetry 统治一切

### Layer 1：Algorithm — Translation symmetry 把 gate set 砍到 4 个

考虑 $L \times L$ lattice 上的 spin Hamiltonian（Eq. 1）：

$$H = \sum_{\langle i,j\rangle}(J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j) + \sum_i h_i X_i$$

$\langle i,j\rangle$ 遍历最近邻 bond，$J_x, J_y, J_z$ 是 coupling。TFIM 是 $J_x=J_y=0$ 特例，Heisenberg 是 $J_x=J_y=J_z$ 特例。

这 Hamiltonian 有 $\mathbb{Z}_L \times \mathbb{Z}_L$ translation symmetry。把 lattice 按 coordinate parity 分成四个 sublattice（叫 plaquettization），一个 Trotter step 就变成 global rotation layer。然后几个 algebraic trick：

1. XX/YY layer 通过 global H 和 global S 共轭成 ZZ layer。
2. Inter-plaquette ZZ 通过 cyclic shift 变成 intra-plaquette ZZ。
3. Intra-plaquette ZZ 通过 CNOT ladder 变成 global CNOT + parallel $R_Z$。

最终你只需要 4 个 native logical gadget：

1. **Global transversal CNOT**（block 间）
2. **Global transversal H 和 S**（block 内）
3. **Parallel $R_Z$**（所有 logical qubit 同时）
4. **1D cyclic shift**（automorphism）

关键 insight：**gate 结构只由 Hamiltonian connectivity 决定，跟 interaction strength 无关**。所以同一个架构跑 TFIM、XXZ、Heisenberg、甚至 Fermi-Hubbard（通过 fermion-to-qubit encoding）都行。

### Layer 2：Code — Bicycle chain code 原生支持这 4 个 gadget

要找一个 qLDPC code 原生支持上面 4 个 gate，需要满足几个 property。我列个表：

| 需要的 gate | Code property | 为什么 |
|------------|--------------|--------|
| Transversal CNOT | CSS code | $X$ 和 $Z$ parity-check 分开，CNOT 保留 stabilizer |
| Transversal H | Self-dual（$H_X = H_Z$） | H 交换 $X$ 和 $Z$，self-dual 让 stabilizer 不变 |
| Transversal S | Self-dual + doubly-even | S 让 $X \to Y$，doubly-even（weight $\equiv 0 \bmod 4$）保证 stabilizer 不变 |
| Parallel $R_Z$ | Disjoint logical basis | 不同 logical qubit 物理 support 不重叠，能同时操作 |
| 1D cyclic shift | Translation-invariant stabilizer | 平移 stabilizer pattern 不变 |

Bicycle chain code 是 self-dual bivariate bicycle (BB) code 的一个可调族，恰好满足所有这些。参数：

$$[[n = 2\ell m,\; k = 2\ell,\; d \lesssim m]]$$

- $n$：物理 qubit 数
- $k$：逻辑 qubit 数
- $d$：code distance
- $\ell$：torus 的 length（chain length），决定 $k$
- $m$：torus 的 circumference，限制 $d$

**关键 decoupling**：$d$ 由 $m$ 决定，$k$ 由 $\ell$ 决定，两者独立。所以你想适配更大 lattice，只需加长 chain（增大 $\ell$），不改变 $d$ 和 encoding rate $k/n = 1/m$。

主力 code instance：$a(x,y) = 1 + y^3 + xy^2 + xy^4$，$m=7$，给 $[[14\ell, 2\ell, 6]]$ 族。$\ell=3$ 时就是 $[[42, 6, 6]]$ code（之前 Xu et al. 发现的）。$\ell=4$ 是 $[[56, 8, 6]]$，$\ell=5$ 是 $[[70, 10, 6]]$。encoding rate 固定 $1/7$，对比 surface code $d=9$ 的 $1/81$，高 11.6 倍。

Disjoint logical basis 选奇数 weight 的 length-$\ell$ column（torus slice）。$2\ell$ 个 slice 各自 disjoint，支持 fully parallel STAR injection。

参考 [Bravyi et al., Nature 2024](https://www.nature.com/articles/s41586-024-07107-7)、[Xu et al., arXiv:2510.06159](https://arxiv.org/abs/2510.06159)、[Liang & Chen, arXiv:2510.05211](https://arxiv.org/abs/2510.05211)。

### Layer 3：Hardware — Neutral atom AOD 天然 fit 这个 symmetry

Neutral atom platform 用 AOD（acousto-optic deflector）控制 atom 位置。AOD 能 global shift 整个 array，这恰好 match translation-invariant code 的 cyclic shift operation。

具体 mapping：

| Logical operation | Hardware primitive |
|--------------------|-------------------|
| Global transversal H/S | Global laser pulse，不 shuttling |
| Global transversal CNOT | AOD align 两个 block 的 data array，parallel entangling gate |
| Cyclic shift automorphism | AOD pick up array，translate 一个 lattice spacing，wraparound，deposit。1-2 步 |
| Syndrome extraction | Ancilla array 按 monomial cyclic shift，shuttle + entangle |
| Parallel STAR injection | TMR 的 CNOT ladder 是 column 内 local shuttle + entangle，selective AOD addressing 允许不同 column 同时操作 |

**所有关键 logical operation 都映射到少数几个 global、parallel 的 AOD primitive**，没有复杂 shuttling sequence 或 mid-circuit measurement。这跟 superconducting qubit 的 fixed connectivity 形成对比——后者做 cyclic shift 要一长串 SWAP gate。

参考 [Bluvstein et al., Nature 2023](https://www.nature.com/articles/s41586-023-06927-3)、[Evered et al., arXiv:2604.25987](https://arxiv.org/abs/2604.25987)。

## Parallel STAR Injection：为什么 $\log_2 k$ 是 magic number

这是这篇 paper 最精妙的技术贡献。

### TMR Protocol：怎么 prepare 小角度 magic state

目标：prepare $|m_\theta\rangle_L = R_{z,L}(\theta)|+\rangle_L$。

TMR（Transversal Multi-Rotation）思路：把 logical $Z$ operator $\hat{Z}_L$ 的物理 support 分成 $M$ 个 disjoint subset $\{c_j\}$：

$$\hat{Z}_L = \prod_{j=1}^M Z_{c_j}, \quad Z_{c_j} \equiv \prod_{i \in c_j} Z_i$$

每个 subset 上作用 joint rotation $e^{-i\theta^* Z_{c_j}/2}$，symmetric angle $\theta^*$。Partition 约束：没有任何 $Z_c$ 或它们 product 是 stabilizer。

作用在 $|+\rangle_L$ 上，每个 factor 展开 $e^{-i\theta^* Z_c/2} = \cos(\theta^*/2) - i\sin(\theta^*/2) Z_c$。Full product 是 $2^M$ 项求和。关键观察：任何 proper nonempty $Z_c$ product 跟至少一个 stabilizer 反对易，所以 carry 非平凡 syndrome。只有 identity term 和 full product $\prod_c Z_c = \hat{Z}_L$ 是 syndrome-free。

Post-select trivial syndrome 后留下（unnormalized）：

$$\cos^M(\theta^*/2)|+\rangle_L + (-i)^M \sin^M(\theta^*/2)|-\rangle_L \tag{C3}$$

对比 target $R_{z,L}(\theta)|+\rangle_L = \cos(\theta/2)|+\rangle_L - i\sin(\theta/2)|-\rangle_L$，定出 angle relation：

$$|\tan(\theta/2)| = \tan^M(\theta^*/2), \quad \theta^*/2 \approx (\theta/2)^{1/M} \text{（小角度近似）} \tag{C4}$$

- $\theta$：目标 logical rotation angle
- $\theta^*$：每个 subset 上的 physical rotation angle
- $M$：partition 数

$M$ 越大，$\theta^*$ 越接近 $\theta$（每个 physical rotation 越大），acceptance 越低，但 infidelity 的 $\theta$-scaling 越好。

TMR 原始 paper：[Toshio et al., arXiv:2408.14848](https://arxiv.org/abs/2408.14848)

### Infidelity 和 Acceptance

**Infidelity**（Eq. C8）：

$$\varepsilon(\theta, f) = C(M) \, p \, (1-f) \, \theta^{2(1-1/M)}$$

- $C(M)$：prefactor，$C(M) \simeq 0.061 \cdot M^{2.44}$（$M \geq 2$）
- $p$：physical error rate
- $f$：heralded atom loss fraction
- $(1-f)$：surviving Pauli fraction（loss 被 post-select 掉）
- 指数 $2(1-1/M)$：$M=1$ 是 flat floor（$\theta^0$），$M=3$ 是 $\theta^{4/3}$，$M \to \infty$ 是 $\theta^2$

具体数值（$p=10^{-3}$）：

| $M$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----|---|---|---|---|---|---|---|
| $C(M)$ | 0.47 | 0.50 | 0.68 | 1.66 | 2.95 | 5.28 | 6.79 |

$M=1$ 是 flat floor $\varepsilon \approx 4.7 \times 10^{-4}(1-f)$，angle-independent。$M=3$ 在小角度（$\theta/\pi \leq 1/8$）最优。Crossover 在 $\theta/\pi \approx 0.24$（$p=10^{-3}$）。

**Ideal acceptance**（Eq. C5）：

$$p_{\text{TMR}}(\theta; M) = [\sin^{2/M}(\theta/2) + \cos^{2/M}(\theta/2)]^{-M} \simeq 1 - M(\theta/2)^{2/M}$$

### RUS：为什么 $\log_2 k$ 是 magic

TMR prepare 的 resource state 通过 teleportation inject。Teleportation 的 transversal $Z$ 测量给每个 logical qubit 一个 outcome $m_j \in \{0,1\}$：

- $m_j = 0$（prob 1/2）：正确 sign $\theta_j$，成功。
- $m_j = 1$（prob 1/2）：错误 sign $-\theta_j$，re-teleport at $2\theta_j$。

重复给 RUS ladder $\theta_j \to 2\theta_j \to 4\theta_j \to \cdots$。每个 logical 独立跑。

$k$ 个 parallel ladder，最慢的那个决定 block-level depth。第 $r$ 轮单个 ladder 存活概率 $2^{-r}$，$k$ 个中至少一个存活的概率 $1-(1-2^{-r})^k$。期望最大深度（Eq. C2）：

$$\bar{R}_{\max}(k) = \sum_{r=0}^{\infty}[1-(1-2^{-r})^k] \xrightarrow{k\to\infty} \log_2 k + \frac{\gamma}{\ln 2} + \frac{1}{2} + o(1)$$

$\gamma \approx 0.577$ 是 Euler-Mascheroni 常数。$k=8$ 时 $\bar{R}_{\max}(8) \approx 4.42$，约两倍 per-logical mean $\mathbb{E}[R_j]=2$，远不是 8 倍。

所以 per-logical RUS cost 是 $\log_2(k)/k$ injections per rotation。$k=8$ 时 $4.42/8 \approx 0.55$，对比 surface code STAR 的 $\sim 2$ injections per rotation，**省了 3.6 倍**。

**直觉**：大部分 logical 早期成功（第 1 轮就有一半成功），只有最 unluckiest 的 tail 拖累。但 tail 几何衰减 $2^{-r}$，所以即使 $k$ 很大，最慢那个也就 $\log_2 k$ 轮。这跟 coupon collector problem 有类似 flavor，但分布是 geometric 而非 uniform。

### Post-Selection 的 Factorization

Success rate 分解（Eq. C6-C7）：

$$s_{\text{total}} = p_{\text{init}} \cdot p_{\text{TMR}}(\theta; M)^N \cdot p_{\text{loss}}$$

- $p_{\text{TMR}}(\theta; M)^N$：$N$ 个 rotated logical 的 ideal TMR acceptance，每个独立。
- $p_{\text{init}} \approx \exp[-\gamma N_{\text{loc}}(\ell,m)(1-f)p]$：block 初始化到 $|+\rangle_L^{\otimes k}$ 的 survival，**shared across all $k$ logical**。
- $p_{\text{loss}} = \exp[-f N_{\text{loc}}(\ell,m) p]$：heralded loss survival。

$\gamma \approx 0.93$ 是触发 state-prep detector 的 Pauli event fraction。$N_{\text{loc}}(\ell,m) = 32\ell m + 13$ 是 per-shot noise-location count。

**核心 insight**：$p_{\text{init}}$ 是 block-level cost，amortize 到所有 $k$ 个 logical 上。加 logical qubit 只通过 $p_{\text{TMR}}^N$ 增加 acceptance cost，不增加 $p_{\text{init}}$。这就是 parallel injection 的威力：你 prepare 一次 block，rotate 所有 logical，init cost 只付一次。

数值：$k=8$ bicycle chain 的 $p_{\text{init}} \approx 0.43$，surface code $d=9$ 的 $p_{\text{init}} \approx 0.29$。Bicycle chain 更高，因为虽然 noise location 多，但只需 2 轮 SE（surface code 要 3 轮）。

## Even-Weight Syndrome：为什么 bicycle chain 只要 2 轮 SE

这是 bicycle chain 相对 surface code 的一个结构性优势，我详细讲，因为它很精妙。

TMR 后的非平凡 syndrome 来自 $Z$ 作用在 logical representative 的 subset 上。这些 syndrome 的 weight 取决于 code 结构。

**Surface code**：logical $Z$ 是一根 string，终点在 boundary，被**单个** X-stabilizer 检查 → weight-1 syndrome。单个 measurement error 可以把 weight-1 翻成 weight-0（trivial），隐藏非平凡 syndrome。所以需要第二轮确认。Surface code TMR factory 要 3 轮 SE：1 轮 Z-only（rotation 前，project 回 code space）+ 2 轮 full（rotation 后，post-select）。

**Bicycle chain**：每个 data qubit 被 **4 个** X-stabilizer 检查（even number）。任何 $Z$-type operator 作用的 syndrome weight 都是 even。所以不可能 weight-1，单个 measurement error 无法隐藏非平凡 syndrome。Bicycle chain TMR factory 只需 2 轮 SE：1 轮 Z-only + 1 轮 full。

数学上：bicycle chain 是 torus（closed manifold），没有 boundary。每个 data qubit 参与 even number of X-checks。这来自 self-dual BB code 的拓扑性质。Surface code 有 boundary，logical string 有 endpoint，endpoint 被 single stabilizer 检查。

**实际影响**：bicycle chain SE cycle depth = 8（weight-8 stabilizer），surface code = 4（weight-4）。虽然每轮深 2 倍，但只要 2 轮而非 3 轮。TMR factory 总 SE depth：bicycle chain $8\times 2=16$，surface code $4\times 3=12$。差距不大。但 $p_{\text{init}}$ 更高，因为更少 noise location 被 post-select 掉。

这让我想到 algebraic topology 里的 boundary operator $\partial$。Surface code 的 logical operator是 1-chain with boundary（$\partial \neq 0$），bicycle chain 的是 1-cycle（$\partial = 0$，closed）。Closed 的好处是没有"端点"可以被单个 fault 隐藏。

## Atom Loss：免费 lunch

Heralded erasure model：fraction $f$ 的 physical error 是 atom loss，被 herald（检测到）并 post-select 掉。剩余 $(1-f)$ 是 Pauli error。

**对 infidelity 的影响**：undetectable logical fault 需要一组 $\bar{Z}$ error 恰好 match TMR partition。Heralded loss 不产生这种 pattern，被 post-select 掉，只有剩余 Pauli fraction $(1-f)$ 能 seed logical fault。所以：

$$\varepsilon \propto (1-f)$$

数值验证：$f \in [0, 0.9]$ 范围内 $(1-f)$ collapse 精确成立，不只是 small-$f$ 近似。

**对 acceptance 的影响**：大部分 Pauli event（$\gamma \approx 0.93$）已经触发 state-prep detector 被丢弃。转成 loss 只是额外丢弃 $\approx (1-\gamma) \approx 0.07$ 的部分。所以：

$$s_{\text{total}}(f) \approx s_{\text{total}}(0) \cdot \exp[-f(1-\gamma) N_{\text{loc}} p]$$

Acceptance 几乎不变。$f=0.7$ 时 infidelity 降 $0.3$ 倍，acceptance 只微降。

**免费 lunch**：loss 把 infidelity 降 3.3 倍，acceptance 几乎不降。当前 hardware 的 loss fraction 和这个一致（[Evered et al., arXiv:2604.25987](https://arxiv.org/abs/2604.25987)）。所以加 atom loss detection 几乎是免费的 fidelity boost。

## 数值结果：到底省了多少

用 Stim（Clifford）和 Clift（non-Clifford TMR）做 circuit-level simulation，depolarizing noise $p=10^{-3}$。Decoder 用 relay-BP 和 MLE（Gurobi mixed-integer programming）。

### Per-Gadget LER

Memory 和 transversal CNOT LER vs $k=2\ell$（Fig. 4a）：

- 小 $\ell$ 时 LER 大，$\ell \gtrsim 4$ 后 plateau。
- 加 logical qubit（fixed $d=6$, fixed rate $1/7$）不进一步提高 performance。
- MLE decoder 比 relay-BP 好 1-2 个数量级。
- $k=8$ 时 memory LER $\sim 10^{-7}$ per round per logical。

**关键对比**：bicycle chain $d=6$ 的 Clifford LER 和 surface code $d=9$ 相当。所以 $d=6$ 高码率 code 的纠错能力和 $d=9$ surface code 一样，但 encoding rate 高 11.6 倍。

### TMR 和 RUS Performance

**Teleported $R_Z$ LER vs $\theta$**（Fig. 4b）：

$$\varepsilon_{R_Z} \approx \alpha \, p \, (1-f) \, \theta$$

- bicycle chain $\alpha_{\text{BB}} \approx 3.1$
- surface code $\alpha_{\text{surf}} \approx 3.8$

小角度 regime bicycle chain 略好，因为 RUS 的 large-angle terminus $R_Z(\pi) = Z$ 是 free Pauli-frame update，而 surface code 的 $R_Z(\pi/2) = S$ 是 transversal Clifford（有 logical error $\sim 10^{-6}$）。

**Expected SE cycles per $R_Z$**（Fig. 4c）：

小角度 regime（$\theta \sim 10^{-3}\text{-}10^{-1}$），parallel bicycle STAR 的 cycle count 比 surface code $d=9$ 低 $\sim 10\times$，比 serial high-rate injection 低 $\sim 5\times$。

两机制驱动 gain：
1. $d=6$ vs $d=9$ 更少 noise location。
2. Parallel injection amortize $p_{\text{init}}$ across all $k$ logical，per-logical cost $\log_2(k)/k$。

### End-to-End Resource

**TFIM**（$8 \times 8$, $T^* = 2.0(zJ)^{-1}$, $z=4$, $J=g=1$, $L=8$）：

| | Bicycle chain STAR ($f=0$) | Bicycle chain STAR ($f=0.7$) | Surface STAR ($d=9$) |
|---|---|---|---|
| Physical qubits | 1,904 - 3,584 | 2,240 | $\sim 5.5\times$ more |
| Per-shot runtime | 8 - 116 s | $\sim 200$ s | comparable |
| Gate error | 0.43 | - | 0.46 |
| $T^*$ reach | $2.0(zJ)^{-1}$ | $8(zJ)^{-1}$（$\sim 2\times 10^6$ T equiv） | $2.0(zJ)^{-1}$ |

**Fermi-Hubbard**（$8 \times 8$, $T^* = 2.0(zt)^{-1}$, $U=4, t=1, L=8$）：

| | Bicycle chain STAR ($f=0$) | Bicycle chain STAR ($f=0.7$) | Surface STAR ($d=9$) |
|---|---|---|---|
| Physical qubits | 5,488 - 10,752 | $\sim 6,300$ | $\sim 5.7\times$ more |
| Per-shot runtime | 18 s - 10.2 min | $\sim 200$ s | comparable |
| Gate error | 0.65 | - | 0.84 |
| $T^*$ reach | $2.0(zt)^{-1}$ | $4(zt)^{-1}$（$\sim 5\times 10^6$ T equiv） | $2.0(zt)^{-1}$ |

FH 的 gate error 更高（0.65 vs 0.43），因为 Trotter step 更 Clifford-heavy（$\sim 6$ transversal Clifford per rotation vs TFIM 的 $\sim 2$）。

## 六种架构对比

Table IV 的数据（$p_{\text{phys}} = 10^{-3}$）：

| Architecture | Code | Enc. Rate | Cycle | Clifford | Rotation | Space | Time |
|---|---|---|---|---|---|---|---|
| **Bicycle chain STAR** (this work) | $[[14\ell, 2\ell, 6]]$ | 1/7 | 2.0 ms | Global transversal | Direct analog + parallel TMR | Data encoding | Analog $R_Z$ |
| Surface STAR (Ismail et al.) | $[[d^2, 1, d]]$, $d=9$ | 1/81 | 1.0 ms | Transversal | Direct analog + TMR | Surface patches | Analog $R_Z$ |
| Surface transv. + cult. (Zhou et al.) | $[[d^2, 1, d]]$ | $1/d^2$ | 1.0 ms | Transversal | Clifford+T + T cult. | Surface patches | T cultivation |
| Pinnacle (Webster et al.) | GB codes | 3%-26.7% | 1.5 ms | Frame tracking | Clifford+T + cult.+distill. | Magic engine ($\sim 4,410$) | T magic |
| Surface surgery (Beverland et al.) | $[[d^2, 1, d]]$ | $1/d^2$ | 1.0 ms | Code surgery | Clifford+T + 15-to-1 | Data + factories | 15-to-1 distill. |
| Extractor (Khan et al.) | $[[288, 12, 18]]$ | 1/24 | 1.5 ms | Code surgery | Clifford+T + T cult. | Data encoding | Serial floor |

**Space**（TFIM）：bicycle chain 1,904-3,584；surface STAR $\sim 5.5\times$ more；Beverland $\sim 15\text{-}75\times$ more；Webster/Khan $\sim 2.8\text{-}13\times$ more。

**Time**：bicycle chain 比 code-surgery 架构快 $100\text{-}1000\times$，比 Zhou et al. transversal surface 快 $>10\times$，跟 surface STAR 差不多。

**Gate error**（TFIM/FH）：bicycle chain 0.43/0.65；surface STAR 0.46/0.84；Zhou 0.32/0.31；Webster 0.16/0.09；Beverland 0.64/0.35；Khan 0.05/0.09。

Webster 和 Khan 的 error 更低因为用更大 distance（$d=16, 18$），而且 code instance 选择有限，降到更小 distance task 就不可行。STAR-only 架构 error 受限于 rotation resource state fidelity，没有 synthesis error 可以降。

**核心 take-away**：bicycle chain STAR 在 reduced space 下就已经比 fully fault-tolerant 架构快 $100\text{-}1000\times$。高码率 encoding 的 advantage 被 end-to-end 保留：qubit saving 没有 through slower operation、complex surgery、serialized magic 退回去。

## Fermi-Hubbard 怎么编译

FH 通过 Derby-Klassen compact encoding 映射到 spin Hamiltonian。核心 trick：在 checkerboard 奇数 face 上放 ancilla qubit，吸收 fermionic exchange statistics，让所有 encoded hopping operator 保持 weight-3 和 geometrically local。

**Encoding overhead**：$L \times L$ lattice 用 $L^2$ vertex + $\frac{1}{2}L^2$ face qubit = $1.5L^2$ per spin species。Spinful FH 用两层（$\uparrow, \downarrow$），$\sim 3L^2$ qubit。

**Hopping term**（Eq. A2-A3）：

$$H_V = -\frac{t}{2}(X_R X_B X_G + Y_R X_B Y_G)$$

$$H_H = -\frac{t}{2}(X_R Y_B X_G + Y_R Y_B Y_G)$$

$R, G$ 是 bond 两端 vertex qubit，$B$ 是中间 face qubit。Vertical 和 horizontal bond 只差 face qubit 上的 Pauli（$X$ vs $Y$）。

**On-site Coulomb**（Eq. A4）：

$$H_I = \frac{U}{4}(Z_\uparrow Z_\downarrow - Z_\uparrow - Z_\downarrow)$$

实际用 chemical-potential-shifted 形式 $H_I = \frac{U}{4}Z_\uparrow Z_\downarrow$，去掉两个 linear $Z$ term（只 shift energy 常数），砍掉 $2/3$ on-site rotation。

**编译**：weight-3 Pauli rotation 通过 single-qubit Clifford 共轭成 $Z_R Z_B Z_G$ rotation，再用 CNOT ladder 把 parity 算到一个 qubit 上，做 $R_Z$，uncompute。Weight-2 和 weight-1 同理。encoded graph 保持 translation-invariant，$\mathbb{Z}_L \times \mathbb{Z}_L$ 还是 shift automorphism，match 到 bicycle chain。FH 和 spin model 落在同样 compilable class 里。

参考 [Derby & Klassen, PRB 2021](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.035118)、[Jafarizadeh et al., arXiv:2408.14543](https://arxiv.org/abs/2408.14543)。

## Trotter Error Bound

用 Childs-Su spectral-norm bound。

**TFIM 二阶**（Eq. D3）：

$$\mathcal{E}_2 = n_{\text{steps}}\left[\frac{dt^3}{12} \cdot 4\delta^2 N_s |g| J^2 + \frac{dt^3}{24} \cdot 16 N_b g^2 |J|\right]$$

$n_{\text{steps}} = \lceil T/dt \rceil$，$\delta=4$ 是 lattice degree，$N_s$ 是 site 数，$N_b$ 是 bond 数。第一项 $[A,[A,B]]$ 类，第二项 $[B,[B,A]]$ 类。

**FH 二阶**（Eq. D8-D10）：

$$\mathcal{E}_2 = n_{\text{steps}} \cdot dt^3 (W_{\text{SO2}} + W_{\text{hop}})$$

$$W_{\text{SO2}} = \frac{Ut^2}{6}N_s(\sqrt{5}+8)\delta_{s,2} + \frac{U^2}{24}\|H_h\|\delta_{s,2}$$

$$W_{\text{hop}} = \frac{1}{12}\left\|\sum_{a,b,c \in \mathcal{L}_{\text{hop}}}[[H_c, H_b], H_a]\right\|$$

$\delta_{s,2}$ 在 spinful case ($s=2$) 给 1。$\|H_h\| = \|R\|_1$ 是 single-particle hopping matrix 的 trace norm。$W_{\text{hop}}$ 是 free-fermion single-particle norm，保留 layer sum 内 cancellation。

参考 [Childs et al., PRX 11, 011020 (2021)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011020)。

## Rotation Synthesis（T-based 架构）

T-based 架构把 $R_Z(\theta)$ 拆成 Clifford+T sequence。T-count per rotation（Eq. D13）：

$$n_{T,\text{rot}} = \left\lceil -0.53 \log_2\left(\frac{\epsilon_{\text{syn}}}{M_R}\right) + 5.3 \right\rceil$$

- $\epsilon_{\text{syn}}$：synthesis error budget
- $M_R$：rotation 总数
- Total T-count：$n_T = M_R \cdot n_{T,\text{rot}}$

不同架构 $\epsilon_{\text{syn}}$ 分配不同：
- Beverland et al.：$\epsilon_{\text{syn}} = \epsilon_{RZ}/3$
- Webster et al.：$\epsilon_{\text{syn}} = \epsilon_{\text{gate}}/3$（Clifford frame-track at zero error）
- Zhou et al.：Beverland model at $\epsilon_{RZ}$

TFIM 三者都 round 到 $\sim 1.6 \times 10^6$ T state。FH：Webster $\sim 8.0 \times 10^5$ vs Zhou/Beverland $\sim 9.6 \times 10^5$。STAR 架构直接 inject，没有 synthesis term。

## Limitation 和未来

### Rotation Error Ceiling

$R_Z$ logical error $\sim \alpha p |\theta|$ 限制 total simulatable rotation budget $\sim 1/(\alpha p)$，对 $N$ logical 是 $\sim 1/(\alpha p N)$。Fig. 6a 里 $T^*$ 随 tolerated gate error 趋近 1 而 saturate。要突破得改进 STAR resource state 或 hybridize with discrete magic（[Gidney et al., arXiv:2409.17595](https://arxiv.org/abs/2409.17595)；[Sahay et al., arXiv:2509.05212](https://arxiv.org/abs/2509.05212)）。

Hybrid 策略：arithmetic-heavy 部分用 conventional discrete magic，small-angle rotation 用 STAR。包括 improved angle-dependent synthesis（[Kliuchnikov et al., arXiv:2203.10064](https://arxiv.org/abs/2203.10064)）和 quasi-probability approach 的小角度 residual（[Bothe et al., arXiv:2605.31544](https://arxiv.org/abs/2605.31544)）。

### Noise Model

数值用 simple depolarizing。真实 hardware 有 structured noise bias（[Roffe et al., Quantum 7, 1005 (2023)](https://quantum-journal.org/papers/q-2023-02-05-1005/)）。Atom loss 只用 heralded-erasure 近似，detailed SE under loss 还没做（[Baranes et al., arXiv:2502.20558](https://arxiv.org/abs/2502.20558)；[Liu et al., arXiv:2603.04156](https://arxiv.org/abs/2603.04156)）。MLE decoder 不 scale 到 real-time，需要 ML decoder（[Ataides et al., arXiv:2509.11370](https://arxiv.org/abs/2509.11370)）。

### Replica Parallelism

Bicycle chain 的 left/right 两半可以跑两个独立 simulation instance，double throughput。更一般地，ultra-high-rate code（[Kasai, arXiv:2601.08824](https://arxiv.org/abs/2601.08824)；[Zhao et al., arXiv:2604.16209](https://arxiv.org/abs/2604.16209)）的 logical qubit 分成 $N_{\text{sub}}$ 个 internally cyclic subset，跑 $N_{\text{sub}}$ 个 replica，amortize SE 和 global transversal gate。挑战是 dense packing 可能 preclude disjoint physical representative，需要 batched injection 或额外 angle factory。

## 我的 Intuition 和联想

### Co-design 像 ASIC vs CPU

传统 FTQC 是"通用 CPU"——能跑任何 algorithm，overhead 巨大。STAR 架构是"quantum simulation ASIC"——针对 Trotterized local lattice simulation 优化，牺牲 universality 换 efficiency。Bicycle chain code 是这个 ASIC 的"定制指令集"：native gate 恰好是 simulation 需要的 gate set。

Symmetry-driven co-design 的哲学很优雅：translation symmetry $\mathbb{Z}_\ell$ 像 leitmotif，algorithm 层决定 gate 结构，code 层决定 code family，hardware 层决定 AOD move。三层用同一个 symmetry 串联，每层复杂性都降下来。这让我想到编译器优化里的 loop unrolling 和 vectorization：你识别出 loop 的 stride pattern，然后让 hardware 的 SIMD unit 直接执行。

### Disjoint Logical Basis 和 SIMD

Disjoint logical basis 让 $k$ 个 logical qubit 物理上不重叠，可以同时 TMR inject。这很像 GPU 的 SIMD：同样指令（TMR rotation）同时作用在不同 data lane（不同 logical qubit）。Bicycle chain 的 torus 结构天然给一套 disjoint column 作为 logical representative，比 surface code 的 string-like logical operator 优雅得多。Surface code 的 logical string 有 endpoint，endpoint 是"瓶颈"，而 bicycle chain 的 logical column 是 closed 的，没有 endpoint。

### $\log_2 k$ 和 Extreme Value Theory

RUS 的 $\log_2 k$ scaling 本质是 extreme value 问题。每个 logical 独立成功概率 $1/2$，$k$ 个并行最慢那个落在 tail $2^{-r} \sim 1/k$ 处，所以 $r \sim \log_2 k$。这跟 coupon collector 有类似 flavor，但分布是 geometric 而非 uniform。Inclusion-exclusion 精确求和给出 $\gamma/\ln 2$ 常数项，$\gamma$（Euler-Mascheroni）的出现跟 harmonic series 有关。

直觉上：大部分 logical 早期成功（第 1 轮一半成功），只有最 unluckiest 的 tail 拖累。但 tail 几何衰减，所以即使 $k$ 很大，最慢那个也就 $\log_2 k$ 轮。这就像 parallel computing 里的 Amdahl's law 的反向：串行部分决定 speedup 上限，这里 tail 决定 depth 下限，但 tail 几何衰减所以 limit 很温和。

### Even-Weight Syndrome 和 Homology

Bicycle chain 只要 2 轮 SE 而 surface code 要 3 轮，差别来自拓扑。Surface code 有 boundary，logical string 是 1-chain with boundary（$\partial \neq 0$），endpoint 被 single stabilizer 检查。Bicycle chain 是 torus，logical operator 是 1-cycle（$\partial = 0$，closed），没有 endpoint。

这让我想到 algebraic topology：closed manifold 的 cycle 没有 boundary，所以没有"端点"可以被单个 fault 隐藏。Surface code 的 planar 版本有 boundary，所以 logical operator 有 endpoint。Bicycle chain 的 torus 拓扑天然 immune to 这种 single-fault 隐藏。这和 topological order 的保护机制有深层联系。

### Neutral Atom 和 TPU

Neutral atom 的 AOD 能 global shift 整个 array，这天然 match translation-invariant code 的 cyclic shift。这让我想到 TPU 的 systolic array天然 fit 矩阵乘的 data flow。Hardware-code co-design 让 logical operation 映射到少数几个 global AOD primitive，没有复杂 shuttling sequence。Superconducting qubit 的 fixed connectivity 做 cyclic shift 要一长串 SWAP gate，完全没这个优势。

### $d=6$ 够用的深层原因

Megaquop 只需要 LER $\sim 10^{-6}$。Bicycle chain $d=6$ 给 memory LER $\sim 10^{-7}$ per round（MLE decoder），TMR LER $\sim \alpha p \theta \sim 3 \times 10^{-3} \times 10^{-3} = 3 \times 10^{-6}$。两者都够。这跟 fully fault-tolerant 需要 $d \sim 15\text{-}30$ 形成对比。

核心原因：quantum simulation 精度要求低（$\sim 10^{-6}$ vs arithmetic $\sim 10^{-12}$），而且 STAR 砍掉 synthesis overhead，所以 $d$ 可以小很多。这就像 mixed precision training：你不需要全 FP64，FP16/BF16 够用，所以 model size 和 training cost 大幅下降。STAR 是 quantum 版的"mixed precision"：小角度 rotation 用 analog magic（低精度但便宜），Clifford 用 transversal gate（高精度）。

### Rotation Error Ceiling 和 Irreducible Loss Floor

STAR 的 $R_Z$ error $\sim \alpha p |\theta|$ 像 deep learning 里的 irreducible loss floor：你可以加 qubit 或 factory，但 rotation fidelity 受限于 physical gate fidelity $p$ 和 TMR protocol 的 $\alpha$。要突破需要更好 magic state preparation 或 hybridize with discrete magic。这跟 model scaling 的 bottleneck 类似：data quality、optimizer、initialization 有 fundamental limit，光加参数不够。

### 未来的联想

Paper 提到 ultra-high-rate code 的 replica parallelism。如果有一个 $k/n \sim 1/2$ 的 code，logical qubit 分成 $N_{\text{sub}}$ 个 internally cyclic subset，每个跑一个 replica，throughput 翻 $N_{\text{sub}}$ 倍。挑战是 STAR injection 要 disjoint physical representative，dense packing 可能 break 这个。可能解法是 batched injection（分批 inject，每批 disjoint）或 hybrid（小角度 STAR，大角度 discrete magic factory）。

还可以扩展到 VQE 或 variational algorithm。这些也大量用小角度 $R_Z$，结构上是 Trotterized evolution 的变体。如果 ansatz 有 translation symmetry（lattice Hamiltonian 的 variational ansatz），同样 co-design 可以 apply。

更远一点，这套 framework 的 end-to-end resource estimate 方法论值得借鉴：从 Trotter error bound 出发，选 operating point，decompose circuit 成 architecture-specific primitive，用 circuit-level simulation 拟合 per-gadget error model，组合成 physical qubit count 和 runtime 的 Pareto frontier。这种 top-down from algorithm + bottom-up from hardware 的 approach 可以用到其他 quantum architecture 的 resource estimation。

## 总结

这篇 paper 的核心贡献：**第一个 fully evaluated 的高码率 STAR 架构**。Symmetry-driven co-design 把 translation symmetry 贯穿 algorithm、QEC code、hardware 三层。Bicycle chain code 提供 tunable 高码率 code family（$k/n=1/7$ vs surface code $1/81$），原生支持 global transversal Clifford + cyclic shift + parallel STAR injection。End-to-end simulation 验证 $8 \times 8$ TFIM 只需 2,240 qubit / $\sim 200$ s，FH 只需 $\sim 6,300$ qubit / $\sim 200$ s，相比 surface code baseline 有 $\sim 5.5\times$ space reduction，相比 code-surgery 架构有 $100\text{-}1000\times$ speedup。

四个核心技术 insight：
1. **Disjoint logical basis** 让 parallel STAR injection 把 per-logical RUS cost 从 $\sim 2$ 降到 $\log_2(k)/k$。
2. **Even-weight syndrome**（torus 拓扑）让 TMR factory 只要 2 轮 SE（surface code 要 3 轮）。
3. **Translation symmetry** 把所有 logical operation 映射到少数几个 global AOD primitive。
4. **Heralded atom loss** 把 infidelity 降 $(1-f)$ 倍，acceptance 几乎不变（免费 lunch）。

这让我对 quantum simulation 的 early fault-tolerant 路线有了清晰 picture：不用等 fully fault-tolerant 的 $d \sim 30$ surface code，几千 qubit + 几百秒就能跑 megaquop-scale simulation。这可能是量子计算第一个真正有 practical advantage 的 application。Co-design 的哲学——针对特定 workload 优化整个 stack——可能比追求 universal fault-tolerance 更早 deliver quantum advantage。

---

# Fast and Parallel High-Rate STAR Architecture 深度讲解

## 1. Big Picture：这篇 paper 在解决什么问题

Fault-tolerant quantum computing 的传统路线是"code-driven"：先选一个通用 QEC code（典型是 surface code），然后在上面实现 universal gate set（Clifford + T），用 magic state distillation 或 cultivation 供应 T gate。这条路线有三层 overhead 叠加：

1. **Encoding overhead**：surface code 每个逻辑 qubit 用 $d^2$ 个物理 qubit（$d \sim 10\text{-}30$），encoding rate $O(1/d^2)$ 极低。
2. **Gate overhead**：逻辑 gate 的 fault-tolerant 实现有额外 space-time cost（lattice surgery 要 $O(d^3)$，transversal 降到 $O(d^2)$）。
3. **Synthesis overhead**：Trotterized Hamiltonian simulation 里大量小角度 $R_Z$ rotation（$\theta \sim 10^{-2}\text{-}10^{-3}$ rad），每个要拆成几十个 T gate。

对 megaquop 规模（$\sim 10^6$ 逻辑 gate）的 quantum simulation，三层叠加太贵。STAR（Space-Time efficient Analog Rotation）架构的 insight 是：quantum simulation 需要的精度不高（logical error rate $\sim 10^{-6}$ 就够，远低于 arithmetic/chemistry 的 $10^{-9}\text{-}10^{-15}$），所以可以 trade generality for efficiency。STAR 直接在 code block 内制备小角度 rotation magic state $|\theta\rangle \propto |0\rangle + e^{i\theta}|1\rangle$，通过 RUS teleportation 实现 $R_Z(\theta)$，逻辑错误率 $\sim p|\theta|$，对 $p \sim 10^{-3}$、$\theta \sim 10^{-3}$ 给出 $\sim 10^{-6}$ per gate，砍掉了 synthesis overhead。Transversal STAR 进一步把 Clifford backbone 换成 transversal gate，把 Clifford overhead 从 $O(d^3)$ 降到 $O(d^2)$。

但现有 STAR 实现都基于 surface code，encoding rate $O(1/d^2)$ 仍然很贵。高码率 qLDPC code 理论上 encoding rate 高得多，但要把 STAR 扩展到 qLDPC code 有两个核心挑战：

- **Gate matching**：qLDPC code 的 native logical gate 要 match 目标 algorithm 的结构。
- **Parallel injection**：naive 串行 magic-state injection 在 $[[n,k,d]]$ block 里让其他 $k-1$ 个逻辑 qubit 闲置，且 post-selection rate 随 $k$ 指数衰减。

这篇 paper 的核心贡献就是通过 **symmetry-driven co-design** 同时解决这两个挑战，把 translation symmetry 贯穿 algorithm、QEC code、hardware 三层，实现一个 maximally parallel 的高码率 STAR 架构。

相关 reference：
- STAR 原始架构：[Akahoshi et al., PRX Quantum 5, 010337 (2024)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010337)
- Transversal STAR：[Ismail et al., arXiv:2509.18294](https://arxiv.org/abs/2509.18294)
- TMR protocol：[Toshio et al., arXiv:2408.14848](https://arxiv.org/abs/2408.14848)

---

## 2. 三层 Co-design 的核心逻辑

这篇 paper 的核心方法论是 **symmetry-driven co-design**：target Hamiltonian 的 translation symmetry $\mathbb{Z}_\ell$ 像一条主线，贯穿 algorithm gadget、QEC code、hardware implementation 三个层面。每一层都利用这个 symmetry 把复杂性降下来。

### 2.1 Algorithm 层：Translation Symmetry 把 Trotter step 降成 4 个 global gadget

考虑 $L \times L$ square lattice 上的 spin Hamiltonian（Eq. 1）：

$$H = \sum_{\langle i,j\rangle}(J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j) + \sum_i h_i X_i$$

这里 $\langle i,j\rangle$ 遍历最近邻 bond，$J_x, J_y, J_z$ 是 coupling，$h_i$ 是 transverse field。TFIM 是 $J_x=J_y=0$ 的特例，Heisenberg 是 $J_x=J_y=J_z$ 的特例。

**Plaquettization**：把 lattice 按坐标奇偶分成四个 disjoint 的 $L/2 \times L/2$ sublattice。最近邻 bond 或者是 inter-plaquette（连接不同 sublattice），或者是 intra-plaquette（同一 sublattice 内部）。一个 Trotter step 可以组织成 global XX/YY/ZZ rotation layer + single-site X rotation。

关键编译步骤：
1. **XX/YY → ZZ**：通过 global H（$X \to Z$）和 global S（$Y \to Z$）共轭，把 XX 和 YY layer 转成 ZZ layer。
2. **Inter-plaquette ZZ → intra-plaquette ZZ**：通过在 sublattice 的两个方向做 cyclic shift，把 inter-plaquette bond 映射到 intra-plaquette bond。
3. **Intra-plaquette ZZ → global CNOT + $R_Z$**：用经典恒等式（Eq. 3）：

$$e^{-i\frac{\theta}{2}Z_A Z_B} = \text{CNOT}_{AB}[I_A \otimes R_{Z_B}(\theta)]\text{CNOT}_{AB}$$

这里 $A$、$B$ 标记两个 qubit set，$\theta$ 是 rotation angle，CNOT 把 $Z_B$ 的 parity 计算到 $A$ 上，做完 $R_Z$ 再 uncompute。

把每一行 sublattice encode 到一个 code block，shift 的另一个方向就对应 code block 之间的 shifting。这样 automorphism 简化成 1D chain 上的 cyclic shift。最终需要的 native logical gadget 只有四个：

1. **Global CNOT**（block 间 transversal）
2. **Global H 和 S**（block 内 transversal）
3. **Parallel $R_Z$**（所有 logical qubit 同时）
4. **1D chain 上的 cyclic shift**（automorphism）

核心 insight 是：需要的 global Clifford 结构 **只由 Hamiltonian 的 connectivity 决定**，和 interaction strength 无关。这意味着同一个架构可以跑 TFIM、XXZ、Heisenberg，甚至 Fermi-Hubbard（通过 fermion-to-qubit encoding）。

### 2.2 Code 层：Bicycle Chain Code 原生实现这 4 个 gadget

要在 qLDPC code 上原生实现上面四个 gadget，需要满足几个 code property：

- **Transversal CNOT**：任何 CSS code 都有（$X$ 和 $Z$ parity-check matrix 分开）。
- **Transversal H**：需要 self-dual code（$H_X = H_Z$）。
- **Transversal S**：需要 self-dual + doubly-even（所有 stabilizer weight $\equiv 0 \pmod{4}$）。
- **Parallel $R_Z$**：需要 disjoint logical basis（每个 logical qubit 的物理 support 不重叠），才能同时做 TMR injection。
- **1D cyclic shift**：需要 translation-invariant code，stabilizer pattern 在某个方向平移不变。

Bicycle chain code 是 self-dual bivariate bicycle (BB) code 的一个可调族，满足所有这些要求。参数：

$$[[n = 2\ell m,\; k = 2\ell,\; d \lesssim m]]$$

- $n$：物理 qubit 数
- $k$：逻辑 qubit 数
- $d$：code distance
- $\ell$：torus 的 length（chain length），决定 $k$
- $m$：torus 的 circumference，限制 $d$

关键 decoupling：$\ell$ 和 $m$ 独立可调。chain 加长（加 logical qubit 适配更大 lattice）只增大 $\ell$，不改变 $d$ 和 encoding rate $k/n = 1/m$。

Disjoint logical basis 选为奇数 weight 的 length-$\ell$ column（torus slice）。$2\ell$ 个 slice 各自 disjoint，支持 fully parallel STAR injection。translation by two slices 生成 $\mathbb{Z}_\ell$ cyclic shift automorphism。

这篇用的主力 code instance：$a(x,y) = 1 + y^3 + xy^2 + xy^4$，$m=7$，给 $[[14\ell, 2\ell, 6]]$ 族。$\ell=3$ 时就是 Xu et al. 的 $[[42, 6, 6]]$ code。

相关 reference：
- Bravyi et al. BB code（Nature 2024）：[https://www.nature.com/articles/s41586-024-07107-7](https://www.nature.com/articles/s41586-024-07107-7)
- Xu et al. batched high-rate logical operations：[arXiv:2510.06159](https://arxiv.org/abs/2510.06159)
- Liang & Chen self-dual BB codes：[arXiv:2510.05211](https://arxiv.org/abs/2510.05211)

### 2.3 Hardware 层：Neutral-atom AOD 把 symmetry 变成物理 move

在 neutral-atom platform 上，同一个 translation symmetry 编译成 AOD（acousto-optic deflector）的 cyclic-shift move。bicycle chain code 的 left 和 right half 各排成 $\ell \times m$ 的 atom array，syndrome extraction 用两个额外的 $\ell \times m$ ancilla array（X-check 和 Z-check）。

- **Global transversal H/S**：single-qubit gate，global pulse，不需要 shuttling。
- **Global transversal CNOT**：AOD shuttling 把两个 block 的对应 data array align 到 entangling zone，然后 parallel physical entangling gate。
- **Cyclic shift automorphism**：AOD pick up data array，translate 一个 lattice spacing（periodic wraparound），deposit back。一个 AOD array 要两步，两个 AOD array 一步。
- **Syndrome extraction**：ancilla array 按 monomial cyclic shift，然后 shuttling + entangle with data。cyclic shift 用同一个 AOD primitive。
- **Parallel STAR injection**：TMR 的 $Z \otimes \cdots \otimes Z$ rotation 拆成 CNOT ladder + single-qubit $R_Z$，CNOT 对应 column 内 local shuttling + entanglement。selective AOD addressing 允许不同 logical representative 同时操作。

核心 hardware insight 是：**所有关键 logical operation 都映射到少数几个 global、parallel 的 AOD primitive**，没有复杂的 shuttling sequence 或 mid-circuit measurement feedback。

相关 reference：
- Bluvstein et al. neutral atom logical processor：[Nature 2023](https://www.nature.com/articles/s41586-023-06927-3)
- Evered et al. high-fidelity entangling gates：[arXiv:2604.25987](https://arxiv.org/abs/2604.25987)

---

## 3. Bicycle Chain Code 的构造细节

这部分我详细讲 code 的数学构造，因为它是整个架构的基石。

### 3.1 Group Algebra 和 Parity-Check Matrix

在 group algebra $\mathbb{Z}_2[\mathbb{Z}_\ell \times \mathbb{Z}_m]$ 里工作，即 bivariate polynomial modulo $x^\ell - 1$ 和 $y^m - 1$。取两个 low-weight polynomial $a(x,y)$ 和 $b(x,y)$，parity-check matrix：

$$H_X = [A \mid B], \quad H_Z = [B^\top \mid A^\top]$$

$A$、$B$ 分别是 $a$、$b$ 的 block-circulant matrix。$n = 2\ell m$ 个 data qubit 分成 left half $L$（前 $\ell m$ 个）和 right half $R$（后 $\ell m$ 个）。每个 qubit 由 torus cell $(i,j)$ 索引（$i \in \mathbb{Z}_\ell$，$j \in \mathbb{Z}_m$）。monomial $x^p y^q$ 作用是 cyclic shift $(i,j) \mapsto (i+p, j+q)$。

### 3.2 Self-dual 条件

取 self-dual instance $b = a^\dagger$，其中 adjoint 把 $x^i y^j \mapsto x^{-i} y^{-j}$。等价地 $B = A^\top$，所以 $H_X = H_Z$。这给 transversal H。

这篇用的 polynomial：

$$a(x,y) = 1 + y^3 + xy^2 + xy^4$$

weight-4。所以 $b(x,y) = 1 + y^{-3} + x^{-1}y^{-2} + x^{-1}y^{-4}$，也 weight-4。每个 stabilizer generator 的 weight = wt$(a)$ + wt$(b)$ = 8。因为 $8 \equiv 0 \pmod{4}$，code 是 **doubly-even**，给 transversal S。

### 3.3 Code Parameter 的 Decoupling

- $d$ 由 $a$ 和 circumference $m$ 决定。
- $k = 2\ell$ 由 chain length $\ell$ 决定。
- $n = 2\ell m$。

所以 chain 加长只增大 $\ell$，不改变 $d$ 和 encoding rate $k/n = 1/m$。这让你可以适配不同大小的 simulation lattice 而不牺牲纠错能力。

### 3.4 Disjoint Logical Basis

在 bicycle chain code 里，一套完整的 disjoint logical representative 可以简单选为奇数 weight 的 length-$\ell$ column。这些 column 在 torus 上是 slice，$2\ell$ 个 slice 各自 disjoint（物理 support 不重叠）。这是 parallel STAR injection 的前提：不同 logical qubit 的物理 qubit 完全不重叠，可以同时操作。

### 3.5 Code Instance

| $a(x,y)$ | $m$ | $n$ | $k$ | $d$ | $k/n$ |
|-----------|-----|------|------|-----|-------|
| $1+y^3+xy^2+xy^4$ | 7 | $14\ell$ | $2\ell$ | 6 | 1/7 |

$\ell=3$：$[[42, 6, 6]]$；$\ell=4$：$[[56, 8, 6]]$；$\ell=5$：$[[70, 10, 6]]$；等等。所有 instance 共享同一个 polynomial，只是 chain 加长。

---

## 4. Parallel STAR Injection 的数学

这是这篇 paper 最核心的技术贡献，我详细讲。

### 4.1 TMR（Transversal Multi-Rotation）Protocol

目标：制备 logical resource state $|m_\theta\rangle_L = R_{z,L}(\theta)|+\rangle_L = \cos(\theta/2)|+\rangle_L - i\sin(\theta/2)|-\rangle_L$。

TMR 的思路是把 $\hat{Z}_L$ 的物理 support 分成 $M$ 个 disjoint subset $\{c_j\}$：

$$\hat{Z}_L = \prod_{j=1}^M Z_{c_j}, \quad Z_{c_j} \equiv \prod_{i \in c_j} Z_i$$

每个 subset 上作用 joint rotation $e^{-i\theta^* Z_{c_j}/2}$，symmetric angle $\theta^*$。partition 的约束：没有任何 $Z_c$ 或它们的 product 是 stabilizer。

作用在 $|+\rangle_L$ 上，每个 factor 展开：

$$e^{-i\theta^* Z_c/2} = \cos(\theta^*/2) - i\sin(\theta^*/2) Z_c$$

full product 是 $2^M$ 项的和。关键 observation：任何 proper、nonempty 的 $Z_c$ product 和至少一个 stabilizer 反对易，所以 carry 非平凡 syndrome。只有 identity term 和 full product $\prod_c Z_c = \hat{Z}_L$ 是 syndrome-free。

Post-select trivial syndrome 后，留下（unnormalized）：

$$\cos^M(\theta^*/2)|+\rangle_L + (-i)^M \sin^M(\theta^*/2)|-\rangle_L \tag{C3}$$

对比 target $R_{z,L}(\theta)|+\rangle_L$，得到 angle relation（Eq. C4）：

$$|\tan(\theta/2)| = \tan^M(\theta^*/2), \quad \theta^*/2 \approx (\theta/2)^{1/M} \text{（小角度）}$$

这里 $\theta$ 是目标 logical rotation angle，$\theta^*$ 是每个 subset 上的 physical rotation angle，$M$ 是 partition 数。$M$ 越大，$\theta^*$ 越接近 $\theta$（每个 physical rotation 越大），但 acceptance 下降。

### 4.2 TMR Acceptance 和 Infidelity

**Ideal acceptance**（Eq. C5）：

$$p_{\text{TMR}}(\theta; M) = [\sin^{2/M}(\theta/2) + \cos^{2/M}(\theta/2)]^{-M} \simeq 1 - M(\theta/2)^{2/M} + O(\theta^4)$$

**Infidelity**（Eq. C8）：

$$\varepsilon(\theta, f) = C(M) \, p \, (1-f) \, \theta^{2(1-1/M)}$$

- $C(M)$：prefactor，随 $M$ 幂律增长 $C(M) \simeq A M^B$，$A=0.061$，$B=2.44$。
- $p$：physical error rate。
- $f$：heralded atom loss fraction。
- $(1-f)$：surviving Pauli fraction。
- 指数 $2(1-1/M)$：$M=1$ 时 $\theta^0$（flat floor），$M=3$ 时 $\theta^{4/3}$，$M\to\infty$ 时 $\theta^2$。

$M=1$ 是 flat floor $\varepsilon = C(1)p(1-f) \approx 4.7 \times 10^{-4}(1-f)$（at $p=10^{-3}$），angle-independent。$M=3$ 在小角度 regime（$\theta/\pi \leq 1/8$）最优。crossover 在 $\theta/\pi \approx 0.24$（at $p=10^{-3}$）。

### 4.3 RUS（Repeat-Until-Success）的 $\log_2 k$ Scaling

TMR 制备的 resource state 通过 teleportation 注入。teleportation 的 transversal $Z$ 测量给每个 logical qubit 一个 outcome $m_j \in \{0,1\}$：

- $m_j = 0$（概率 1/2）：正确 sign $\theta_j$，成功。
- $m_j = 1$（概率 1/2）：错误 sign $-\theta_j$，需要 re-teleport at $2\theta_j$。

重复给 RUS ladder $\theta_j \to 2\theta_j \to 4\theta_j \to \cdots$。每个 logical 独立运行。

$k$ 个 parallel ladder，最慢的那个决定 block-level depth。第 $r$ 轮单个 ladder 存活概率 $2^{-r}$，$k$ 个中至少一个存活的概率 $1-(1-2^{-r})^k$。期望最大深度（Eq. C2）：

$$\bar{R}_{\max}(k) = \sum_{r=0}^{\infty}[1-(1-2^{-r})^k] = \sum_{j=1}^k \binom{k}{j}\frac{(-1)^{j+1}}{1-2^{-j}} \xrightarrow{k\to\infty} \log_2 k + \frac{\gamma}{\ln 2} + \frac{1}{2} + o(1)$$

$\gamma \approx 0.577$ 是 Euler-Mascheroni 常数。$k=8$ 时 $\bar{R}_{\max}(8) \approx 4.42$，约两倍 per-logical mean $\mathbb{E}[R_j]=2$，远不是 8 倍。这归功于 geometric suppression：大部分 logical 早期成功，只有最慢的 tail 拖累。

所以 per-logical RUS cost 是 $\log_2(k)/k$ injections per rotation，对比 surface code STAR 的 $\sim 2$ injections per rotation。$k=8$ 时 $4.42/8 \approx 0.55$，已经低于 surface code。

### 4.4 Post-Selection Rate 的 Factorization

Success rate 分解成三个 factor（Eq. C6-C7）：

$$s_{\text{total}} = p_{\text{init}} \cdot p_{\text{TMR}}(\theta; M)^N \cdot p_{\text{loss}}$$

- $p_{\text{TMR}}(\theta; M)^N$：$N$ 个 rotated logical qubit 的 ideal TMR acceptance，每个独立。$N \leq k$。
- $p_{\text{init}} \approx \exp[-\gamma N_{\text{loc}}(\ell,m)(1-f)p]$：block 初始化到 $|+\rangle_L^{\otimes k}$ 的 survival，**shared across all $k$ logical**。$\gamma \approx 0.93$ 是触发 state-prep detector 的 Pauli event fraction。$N_{\text{loc}}(\ell,m) = 32\ell m + 13$ 是 per-shot noise-location count。
- $p_{\text{loss}} = \exp[-f N_{\text{loc}}(\ell,m) p]$：heralded loss survival。

关键 insight：$p_{\text{init}}$ 是 block-level cost，amortize 到所有 $k$ 个 logical 上。加 logical qubit 只通过 $p_{\text{TMR}}^N$ 增加 acceptance cost，不增加 $p_{\text{init}}$。

数值验证：$k=8$（$\ell=4, m=7$）的 bicycle chain $p_{\text{init}} \approx 0.43$，对比 surface code $d=9$ 的 $p_{\text{init}} \approx 0.29$。bicycle chain 更高，因为虽然 noise location 多，但只需要 2 轮 SE（surface code 要 3 轮）。

### 4.5 Atom Loss 的影响

Heralded erasure model：fraction $f$ 的 physical error 是 loss，被 herald 并 post-select 掉（整个 shot 丢弃）。剩余 $(1-f)$ 是 Pauli error。

对 infidelity 的影响：undetectable logical fault 需要一组 $\bar{Z}$ error 恰好 match TMR partition。heralded loss 不产生这种 pattern，被 post-select 掉，只有剩余 Pauli fraction $(1-f)$ 能 seed logical fault。所以：

$$\varepsilon \propto (1-f)$$

数值验证：$f \in [0, 0.9]$ 范围内，$(1-f)$ collapse 精确成立（不只是 small-$f$ 近似）。

对 acceptance 的影响：大部分 Pauli event（$\gamma \approx 0.93$）已经触发 state-prep detector 被丢弃。转成 loss 只是额外丢弃 $\approx (1-\gamma) \approx 0.07$ 的部分。所以：

$$s_{\text{total}}(f) \approx s_{\text{total}}(0) \cdot \exp[-f(1-\gamma) N_{\text{loc}} p]$$

acceptance 几乎不变。$f=0.7$ 时 infidelity 降 $0.3$ 倍，acceptance 只微降。当前 hardware 的 loss fraction 和这个一致。

---

## 5. Syndrome Extraction 的细节和结构性优势

### 5.1 Weight-8 Stabilizer 和 Depth-8 Schedule

每个 stabilizer generator weight-8（4 个在 $L$，4 个在 $R$）。每个 ancilla 执行 8 个 CNOT，最短 cycle depth = 8。用 Bravyi et al. 的搜索方法论枚举所有 candidate CNOT ordering，用 symbolic propagation over $\mathbb{Z}_2$ 验证每个 ancilla 干净测量目标 stabilizer，然后按 circuit-level fault distance ranking。

对 $d=6$ code：212 个 schedule 通过 correctness filter，其中 192 个 $d_{\text{fault}}=4$，20 个 $d_{\text{fault}}=5$。用 $d_{\text{fault}}=5$ 的 schedule（$d-1$）。因为 grammar 只依赖 check weight（=4）不依赖具体 monomial，同一个 schedule 适用于所有 $\ell$ 和 $m$。

Schedule（Table II）分 8 轮，每轮一个 CNOT layer，X-check 和 Z-check 在 opposite data half 上同时操作：

| Round | X-check (half, monomial) | Z-check (half, monomial) |
|-------|--------------------------|--------------------------|
| 1 | $L, A_1$ | $R, A_4^\top$ |
| 2 | $R, B_1$ | $L, B_4^\top$ |
| 3 | $L, A_2$ | $R, A_3^\top$ |
| 4 | $R, B_2$ | $L, B_2^\top$ |
| 5 | $R, B_3$ | $L, B_3^\top$ |
| 6 | $L, A_3$ | $R, A_2^\top$ |
| 7 | $R, B_4$ | $L, B_1^\top$ |
| 8 | $L, A_4$ | $R, A_1^\top$ |

$A_1 = xy^4, A_2 = xy^2, A_3 = y^3, A_4 = 1$。$B_i$ 是 $a^\dagger$ 的对应 monomial。

### 5.2 为什么 Bicycle Chain 只需 2 轮 SE 而 Surface Code 要 3 轮

这是 bicycle chain 相对 surface code 的一个结构性优势。

TMR 后的非平凡 syndrome 来自 $Z$ 作用在 logical representative 的 subset 上（由 TMR partition 选择）。这些 syndrome 的 weight 取决于 code 结构。

**Surface code**：logical $Z$ 是一根 string，终点在 boundary，被**单个** X-stabilizer 检查 → weight-1 syndrome → 单个 measurement error 可以隐藏它 → 需要第二轮确认。所以 surface code TMR factory 要 3 轮 SE：1 轮 Z-only（rotation 前，project 回 code space）+ 2 轮 full（rotation 后，post-select）。

**Bicycle chain**：每个 data qubit 被 **4 个** X-stabilizer 检查（偶数）。任何 subset 的 syndrome weight 都是偶数 → 不可能 weight-1 → 单个 measurement error 无法隐藏 → 不需要第二轮确认。所以 bicycle chain TMR factory 只需 2 轮 SE：1 轮 Z-only + 1 轮 full。

数学上：bicycle chain 的每个 data qubit 参与 even number of X-checks，所以任何 $Z$-type operator 作用的 syndrome 都是 even weight。这是 self-dual BB code 的拓扑性质（没有 boundary）。

实际影响：bicycle chain 的 SE cycle depth = 8（weight-8），surface code = 4（weight-4）。虽然每轮深 2 倍，但只需要 2 轮而非 3 轮，所以 TMR factory 的总 SE depth 是 $8 \times 2 = 16$ vs $4 \times 3 = 12$，bicycle chain 稍长但差距不大。更重要的是 $p_{\text{init}}$ 更高（更少 noise location 被 post-select 掉）。

---

## 6. 数值结果

### 6.1 Per-Gadget LER

用 Stim（Clifford）和 Clift（non-Clifford TMR）做 circuit-level simulation，depolarizing noise $p=10^{-3}$。Decoder 用 relay-BP 和 MLE（mixed-integer programming via Gurobi）。

**Memory 和 transversal CNOT LER vs $k = 2\ell$**（Fig. 4a）：

- 小 $\ell$ 时 LER 大，$\ell \gtrsim 4$ 后 plateau。
- 加 logical qubit（fixed $d=6$, fixed encoding rate $1/7$）不进一步提高 performance。
- MLE decoder 比 relay-BP 好 1-2 个数量级。
- $k=8$（$\ell=4$）时 memory LER $\sim 10^{-7}$ per round per logical，transversal CNOT LER 同量级。

**关键对比**：bicycle chain $d=6$ 的 Clifford LER 和 surface code $d=9$ 相当。但注意 bicycle chain 用 MLE decoder（near-optimal），surface code baseline 用 MWPF decoder（suboptimal），$d=6$ vs $d=9$ 的 gap 是否 persist 在 optimal decoding 下还待定。

### 6.2 TMR Infidelity 和 Acceptance

**Infidelity vs $\theta$**（Fig. 4b, Fig. 12, Fig. 18a）：

- $M=1$：flat floor $\varepsilon \approx 4.7 \times 10^{-4}(1-f)$，angle-independent。
- $M=3$：$\varepsilon = 0.68 \, p \, (1-f) \, \theta^{4/3}$，小角度 regime 最优。
- bicycle chain $d=6$ 和 surface code $d=9$ 的 per-injection infidelity **几乎相同**（$C(3) = 0.68$ vs $0.79$），说明 TMR infidelity 主要由 angle 和 $M$ 决定，不依赖 code。

**Teleported $R_Z$ LER vs $\theta$**（Fig. 4b）：

- 小角度线性 $\varepsilon = \alpha \, p \, (1-f) \, \theta$，$\alpha_{\text{BB}} \approx 3.1$，$\alpha_{\text{surf}} \approx 3.8$。
- bicycle chain 略好，因为 RUS 的 large-angle terminus（$R_Z(\pi) = Z$，free Pauli-frame update）比 surface code 的 terminus（$R_Z(\pi/2) = S$，transversal Clifford，有 logical error $\sim 10^{-6}$）便宜。

**Expected SE cycles per $R_Z$**（Fig. 4c）：

- 小角度 regime（$\theta \sim 10^{-3}\text{-}10^{-1}$），parallel bicycle STAR 的 cycle count 比 surface code $d=9$ 低 $\sim 10\times$，比 serial high-rate injection 低 $\sim 5\times$。
- 两机制驱动 gain：(1) $d=6$ vs $d=9$ 更少 noise location，(2) parallel injection amortize $p_{\text{init}}$ across all $k$ logicals，per-logical cost $\log_2(k)/k$。

### 6.3 End-to-End Resource Estimate

**TFIM**（$8 \times 8$, $T^* = 2.0(zJ)^{-1}$, $z=4$, $J=g=1$, $L=8$）：
- Trotter step $\delta t^* = 0.0274(zJ)^{-1}$，73 steps。
- T-gate equivalent $\sim 2.1 \times 10^5$。
- Bicycle chain STAR（$f=0$）：1,904-3,584 qubits，8-116 s/shot，gate error 0.43。
- Bicycle chain STAR（$f=0.7$）：2,240 qubits，$\sim 200$ s/shot，$T^* \approx 8(zJ)^{-1}$（$\sim 2 \times 10^6$ T equivalent）。
- Surface code transversal STAR（$d=9$）：5.5$\times$ more qubits，comparable speed，gate error 0.46。

**Fermi-Hubbard**（$8 \times 8$, $T^* = 2.0(zt)^{-1}$, $z=4$, $U=4, t=1, L=8$）：
- Trotter step $\delta t^* = 0.0230(zt)^{-1}$，87 steps。
- T-gate equivalent $\sim 1.4 \times 10^6$。
- Bicycle chain STAR（$f=0$）：5,488-10,752 qubits，18 s-10.2 min/shot，gate error 0.65。
- Bicycle chain STAR（$f=0.7$）：$\sim 6,300$ qubits，$\sim 200$ s/shot，$T^* \approx 4(zt)^{-1}$（$\sim 5 \times 10^6$ T equivalent）。
- Surface code transversal STAR（$d=9$）：5.7$\times$ more qubits，gate error 0.84。

FH 的 gate error 更高（0.65 vs 0.43），因为 FH 的 Trotter step 更 Clifford-heavy（$\sim 6$ transversal Clifford per rotation vs TFIM 的 $\sim 2$）。

---

## 7. 六种架构对比

Table IV 把六种架构按 host code、Clifford backbone、rotation strategy、dominant cost 分类。我用表格形式重新组织关键数据（$p_{\text{phys}} = 10^{-3}$）：

| Architecture | Code | Enc. Rate | Cycle | Clifford | Rotation | Space Bottleneck | Time Bottleneck |
|-------------|------|-----------|-------|----------|----------|-----------------|-----------------|
| **Bicycle chain STAR** (this work) | $[[14\ell, 2\ell, 6]]$ | 1/7 | 2.0 ms | Global transversal | Direct analog + parallel TMR | Data encoding (smallest) | Analog $R_Z$ |
| Surface STAR (Ismail et al.) | $[[d^2, 1, d]]$, $d=9$ | 1/81 | 1.0 ms | Transversal | Direct analog + TMR | Surface data patches | Analog $R_Z$ |
| Surface transv. + cult. (Zhou et al.) | $[[d^2, 1, d]]$ | $1/d^2$ | 1.0 ms | Transversal | Clifford+T synthesis + T cult. | Surface patches | T cultivation |
| Pinnacle (Webster et al.) | GB codes | 3%-26.7% | 1.5 ms | Frame tracking (PBC) | Clifford+T + cult.+distill. | Magic engine ($\sim 4,410$) | T magic |
| Surface surgery (Beverland et al.) | $[[d^2, 1, d]]$ | $1/d^2$ | 1.0 ms | Code surgery | Clifford+T + 15-to-1 distill. | Data + distill. factories | 15-to-1 distill. |
| Extractor (Khan et al.) | $[[288, 12, 18]]$ | 1/24 | 1.5 ms | Code surgery | Clifford+T + T cult. | Data encoding | Serial rotation floor |

**Space comparison**（TFIM，$f=0$）：
- Bicycle chain STAR：1,904-3,584 qubits
- Surface code STAR：$\sim 5.5\times$ more
- Beverland et al.：15-75$\times$ more
- Webster et al. (Pinnacle)：$\sim 2.8\text{-}13\times$ more
- Khan et al.：$\sim 2.8\text{-}13\times$ more

**Time comparison**：
- Bicycle chain STAR vs code-surgery architectures：100-1000$\times$ speedup
- Bicycle chain STAR vs surface transversal (Zhou et al.)：$>10\times$ speedup
- Bicycle chain STAR vs surface STAR：comparable speed

**Gate error comparison**（TFIM / FH）：
- Bicycle chain STAR：0.43 / 0.65
- Surface STAR (Ismail et al.)：0.46 / 0.84
- Zhou et al.：0.32 / 0.31
- Webster et al.：0.16 / 0.09
- Beverland et al.：0.64 / 0.35
- Khan et al.：0.05 / 0.09

注意 Webster et al. 和 Khan et al. 的 gate error 更低，因为它们用更大 distance 的 code（$d=16$ 和 $d=18$），而且 code instance 选择有限——降到更小 distance 会让 task 不可行。STAR-only 架构的 error 受限于 rotation resource state fidelity，没有 synthesis error 可以降。

核心 take-away：**bicycle chain STAR 在 reduced space 下就已经比 fully fault-tolerant 架构快 100-1000$\times$**，而且这个 speedup 在 reduced space 下就实现了。高码率 encoding 的 advantage 被 end-to-end 保留：qubit saving 没有 through slower logical operation、complex surgery、serialized magic preparation 退回去。

相关 reference：
- Beverland et al.：[arXiv:2211.07629](https://arxiv.org/abs/2211.07629)
- Webster et al. Pinnacle：[arXiv:2602.11457](https://arxiv.org/abs/2602.11457)
- Khan et al. extractor：[arXiv:2604.19735](https://arxiv.org/abs/2604.19735)
- Zhou et al. low-overhead transversal：[Nature 2025](https://www.nature.com/articles/s41586-025-69619-3)

---

## 8. Fermi-Hubbard 的编译

FH 通过 Derby-Klassen compact encoding 映射到 spin Hamiltonian。这个 encoding 的核心是在 checkerboard 的奇数 face 上放 ancilla qubit，吸收 fermionic exchange statistics，让所有 encoded hopping operator 保持 weight-3 和 geometrically local。

**Encoding overhead**：$L \times L$ lattice 用 $L^2$ vertex + $\frac{1}{2}L^2$ face qubit = $1.5L^2$ qubit per spin species。spinful FH 用两层（$\uparrow, \downarrow$），$\sim 3L^2$ qubit。

**Hopping term**（Eq. A2-A3）：

$$H_V = -\frac{t}{2}(X_R X_B X_G + Y_R X_B Y_G) \tag{A2}$$

$$H_H = -\frac{t}{2}(X_R Y_B X_G + Y_R Y_B Y_G) \tag{A3}$$

$R, G$ 是 bond 两端的 vertex qubit，$B$ 是中间的 face qubit。vertical bond（$V$）和 horizontal bond（$H$）只差 face qubit 上的 Pauli（$X$ vs $Y$）。

**On-site Coulomb**（Eq. A4）：

$$H_I = \frac{U}{4}(Z_\uparrow Z_\downarrow - Z_\uparrow - Z_\downarrow) \tag{A4}$$

实际实现时用 chemical-potential-shifted 形式 $H_I = \frac{U}{4}Z_\uparrow Z_\downarrow$，去掉两个 linear $Z$ term（只 shift energy 常数），砍掉 $2/3$ 的 on-site rotation。

**编译**：weight-3 Pauli rotation $e^{-i\frac{\theta}{2}P_1 P_2 P_3}$ 通过 single-qubit Clifford 共轭成 $Z_R Z_B Z_G$ rotation（$H$ for $X$, $SH$ for $Y$），再用 CNOT ladder 把 parity 算到一个 qubit 上，做 $R_Z$，uncompute。weight-2 $Z_\uparrow Z_\downarrow$ 是两-body rotation，weight-1 $Z$ 是 single-logical $R_Z$。

encoded graph 保持 translation-invariant：vertex 和 face qubit 各自 periodic tile torus，$\mathbb{Z}_L \times \mathbb{Z}_L$ 还是 shift automorphism，match 到 bicycle chain shift。一行 vertex qubit（加 face qubit）assign 到一个 code block 的 disjoint logical qubit，每个 weight-3 bond rotation 要么在 block 内（intra-row + face），要么跨两个相邻 block（inter-row），都能用 native transversal CNOT + shift 到达。

相对 spin case 的结构变化：constant 1.5$\times$ qubit overhead，weight-3 而非 weight-2 rotation。都不需要新 logical gadget，所以 FH 落在和 spin model 同样的 compilable class 里。

相关 reference：
- Derby-Klassen encoding：[PRB 104, 035118 (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.035118)
- Jafarizadeh et al. FH recipe：[arXiv:2408.14543](https://arxiv.org/abs/2408.14543)
- Campbell early FT Hubbard：[QST 7, 015007 (2022)](https://iopscience.iop.org/article/10.1088/2058-9565/ac3ca6)

---

## 9. Trotter Error Bound

用 Childs-Su spectral-norm bound。

### TFIM

$H = A + B$，$A = -g\sum_v X_v$，$B = -J\sum_{\langle u,v\rangle} Z_u Z_v$。2D square lattice，periodic boundary，degree $\delta = 4$，$N_s$ sites，$N_b$ bonds。

**一阶**（Lie-Trotter）：

$$\mathcal{E}_1 = n_{\text{steps}} \frac{dt^2}{2} \cdot 4 N_b |g| |J| \tag{D2}$$

**二阶**（symmetric Strang）：

$$\mathcal{E}_2 = n_{\text{steps}}\left[\frac{dt^3}{12} \cdot 4\delta^2 N_s |g| J^2 + \frac{dt^3}{24} \cdot 16 N_b g^2 |J|\right] \tag{D3}$$

$n_{\text{steps}} = \lceil T/dt \rceil$。第一项是 $[A,[A,B]]$ 类 commutator，第二项是 $[B,[B,A]]$ 类。

### Fermi-Hubbard

五层 $\{V_0, V_1, H_0, H_1, I\}$。一阶 bound：

$$\mathcal{E}_1 = n_{\text{steps}} \frac{dt^2}{2}(6s N_s t^2 + 4N_s |tU| \delta_{s,2}) \tag{D6}$$

$s$ 是 spin species数，$\delta_{s,2}$ 在 spinful case ($s=2$) 给 1。

二阶用 Campbell split-operator bound，interaction bracket hopping：

$$U_{\text{step}} = e^{-iH_I dt/2} S_2^{\text{hop}}(dt) e^{-iH_I dt/2} \tag{D7}$$

Error 分成 hopping-interaction contribution $W_{\text{SO2}}$ 和 pure-hopping residual $W_{\text{hop}}$：

$$\mathcal{E}_2 = n_{\text{steps}} \cdot dt^3 (W_{\text{SO2}} + W_{\text{hop}}) \tag{D8}$$

$W_{\text{hop}}$ 是 free-fermion single-particle norm，保留 layer sum 内的 cancellation。

相关 reference：
- Childs-Su Trotter error：[PRX 11, 011020 (2021)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011020)

---

## 10. Rotation Synthesis Cost（T-based 架构）

T-based 架构（Beverland et al.、Webster et al.、Zhou et al.）把每个 $R_Z(\theta)$ 拆成 Clifford+T sequence。T-count per rotation（Eq. D13）：

$$n_{T,\text{rot}} = \left\lceil -0.53 \log_2\left(\frac{\epsilon_{\text{syn}}}{M_R}\right) + 5.3 \right\rceil$$

$\epsilon_{\text{syn}}$ 是 synthesis error budget，$M_R$ 是 rotation 总数。total T-count $n_T = M_R \cdot n_{T,\text{rot}}$。

不同架构的 $\epsilon_{\text{syn}}$ 分配不同：
- Beverland et al.：$\epsilon_{\text{syn}} = \epsilon_{RZ}/3$（余下 2/3 给 distillation + data storage）。
- Webster et al.：$\epsilon_{\text{syn}} = \epsilon_{\text{gate}}/3$（用 full gate budget，因为 Clifford frame-track at zero error）。
- Zhou et al.：Beverland model at $\epsilon_{RZ}$ budget。

对 TFIM，三者都 round 到 $\sim 1.6 \times 10^6$ T state。对 FH，Webster et al. $\sim 8.0 \times 10^5$ vs Zhou/Beverland $\sim 9.6 \times 10^5$。

STAR 架构直接 inject analog resource state，没有 synthesis term。

---

## 11. Limitation 和 Future Direction

### 11.1 Noise Model

所有数值用 simple circuit-level depolarizing noise。真实 hardware 可能有 structured noise bias（XZZX surface code 那种），可以被 tailored decoding 利用。Atom loss 只用 heralded-erasure 近似，detailed syndrome extraction under loss 还没做。MLE decoder 不 scale 到 real-time large circuit，需要 ML decoder（[Ataides et al., arXiv:2509.11370](https://arxiv.org/abs/2509.11370)）。

### 11.2 Platform Generalizability

架构不限于 neutral atom。任何有 long-range connectivity 的 platform（trapped ion、spin qubit）都能 host 同样的 logical structure。

### 11.3 Algorithm Generalizability

可以扩展到：
- 超出 native lattice 的 Hamiltonian（embed into lattice grid，selectively control term strength）。
- Randomized / sampling-based time evolution（[Childs et al., Quantum 3, 182 (2019)](https://quantum-journal.org/papers/q-2019-08-05-182/)；[Campbell, PRL 123, 070503 (2019)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.070503)）。
- Real-time response measurement。
- Phase estimation / Hadamard-test（controlled $e^{-iHt}$，用 GHZ register keep operation structural）。

### 11.4 Rotation Error Ceiling

$R_Z$ 的 logical error $\sim \alpha p |\theta|$ 限制了 total simulatable rotation budget $\sim 1/(\alpha p)$，对 $N$ logical qubits 是 $\sim 1/(\alpha p N)$。Fig. 6a 里 $T^*$ 随 tolerated gate error 趋近 1 而 saturate。突破要改进 STAR resource state 或 hybridize with discrete magic（[Gidney et al., arXiv:2409.17595](https://arxiv.org/abs/2409.17595)；[Sahay et al., arXiv:2509.05212](https://arxiv.org/abs/2509.05212)）。

Hybrid 策略：arithmetic-heavy 部分（first quantization chemistry）用 conventional discrete magic，small-angle rotation 用 STAR。包括 improved angle-dependent synthesis（[Kliuchnikov et al., arXiv:2203.10064](https://arxiv.org/abs/2203.10064)）和 quasi-probability approach 的小角度 residual（[Bothe et al., arXiv:2605.31544](https://arxiv.org/abs/2605.31544)）。

### 11.5 Ultra-High-Rate Code 和 Replica Parallelism

Bicycle chain code 的 left/right 两半可以跑两个独立 simulation instance，double sampling throughput。更一般地，ultra-high-rate code（[Kasai, arXiv:2601.08824](https://arxiv.org/abs/2601.08824)；[Zhao et al., arXiv:2604.16209](https://arxiv.org/abs/2604.16209)）的 logical qubit 分解成 $N_{\text{sub}}$ 个 internally cyclic subset，可以跑 $N_{\text{sub}}$ 个 replica，amortize syndrome extraction 和 global transversal gate。挑战是 STAR injection：denser logical packing 可能 preclude fully disjoint physical representative，需要 batched injection 或额外 angle factory。

---

## 12. 我的 Intuition 和联想

### 12.1 Co-design 的类比

这篇 paper 的 co-design 哲学让我想到 ASIC vs CPU 的关系。传统 FTQC 是"通用 CPU"——能跑任何 quantum algorithm，但 overhead 巨大。STAR 架构是"quantum simulation ASIC"——针对 Trotterized local lattice simulation 这个特定 workload 优化，牺牲 universality 换 efficiency。bicycle chain code 就是这个 ASIC 的"定制指令集"：它的 native gate（global transversal CNOT/H/S + cyclic shift + parallel $R_Z$）恰好是 simulation 需要的 gate set。

这种 symmetry-driven co-design 的思路非常优雅。translation symmetry $\mathbb{Z}_\ell$ 像一个 leitmotif：algorithm 层它决定 gate 结构，code 层它决定 code family（translation-invariant stabilizer），hardware 层它决定 AOD move。三层用同一个 symmetry 串联起来，每一层的复杂性都因为这个 symmetry 降下来。

### 12.2 Disjoint Logical Basis 和 SIMD

Disjoint logical basis 让 $k$ 个 logical qubit 在物理上不重叠，可以同时做 TMR injection。这很像 GPU 里的 SIMD：同样指令（TMR rotation）同时作用在不同 data lane（不同 logical qubit）上。关键 trick 是 bicycle chain code 的 torus 结构天然给了一套 disjoint column 作为 logical representative，这比 surface code 的 string-like logical operator 优雅得多。

### 12.3 RUS 的 $\log_2 k$ 和 Coupon Collector

RUS 的 $\log_2 k$ scaling 本质上是一个 extreme value 问题。每个 logical 独立成功概率 $1/2$，$k$ 个并行的最慢那个落在 tail $2^{-r} \sim 1/k$ 处，所以 $r \sim \log_2 k$。这和 coupon collector problem 有类似的 flavor，但分布不同（geometric vs. uniform）。inclusion-exclusion 的精确求和给出 $\gamma/\ln 2$ 的常数项，这个 $\gamma$（Euler-Mascheroni）的出现和 harmonic series 有关。

### 12.4 Even-Weight Syndrome 的拓扑起源

Bicycle chain 只需 2 轮 SE 而 surface code 要 3 轮，这个差别来自 code 的拓扑结构。Surface code 有 boundary，logical string 终止在 boundary 上，被单个 stabilizer 检查 → weight-1 syndrome。Bicycle chain 是 torus（closed manifold），没有 boundary，每个 qubit 被 even number of stabilizer 检查 → 任何 $Z$-type operator 的 syndrome 都是 even weight。这让我想到 homology 里的 boundary operator：surface code 的 logical operator 是 1-chain with boundary，bicycle chain 的是 1-cycle（closed）。closed 的好处是没有"端点"可以被单个 fault 隐藏。

### 12.5 和 deep learning 的类比

这篇 paper 的 resource estimation pipeline 让我想到 deep learning 里的 model scaling law。Trotter error bound $\sim dt^3$ 类似 loss 的 power-law scaling，gate error budget $\epsilon_{\text{gate}}$ 类似 target loss，physical qubit count 和 runtime类似 model size 和 training time。Fig. 5 的 space-time Pareto frontier 就像 compute-optimal scaling 的 frontier：你可以 trade space（qubit）换时间（factory 数），最优 operating point在 curve 的拐点。

STAR 的 rotation error ceiling $\sim \alpha p |\theta|$ 类似 model 的 irreducible loss floor：你可以加更多 qubit 或 factory，但 rotation fidelity 受限于 physical gate fidelity $p$ 和 TMR protocol 的 $\alpha$。要突破需要更好的 magic state preparation（类似更好的 optimizer）或 hybridize with discrete magic（类似混合 precision训练）。

### 12.6 为什么 $d=6$ 够用

Megaquop regime 只需要 logical error rate $\sim 10^{-6}$ per gate。bicycle chain $d=6$ 给出 memory LER $\sim 10^{-7}$ per round（MLE decoder），TMR LER $\sim \alpha p \theta \sim 3 \times 10^{-3} \times 10^{-3} = 3 \times 10^{-6}$。两者都够用。这和 fully fault-tolerant 架构需要 $d \sim 15\text{-}30$ 形成鲜明对比。核心原因是 quantum simulation 的精度要求低（$\sim 10^{-6}$ vs arithmetic的 $\sim 10^{-12}$），而且 STAR 砍掉了 synthesis overhead，所以 $d$ 可以小很多。

### 12.7 Neutral Atom 的天然 fit

Bicycle chain code 的 translation-invariant stabilizer pattern 天然 fit neutral atom 的 AOD control。AOD 可以 global shift 整个 array，这正好 match cyclic shift automorphism 和 syndrome extraction 的 ancilla shift。这比 superconducting qubit 的 fixed connectivity 要好得多——后者做 cyclic shift 要一长串 SWAP gate。Neutral atom 的 long-range interaction（Rydberg）也支持 transversal CNOT 的 parallel entangling gate。

这种 hardware-code co-design 让我想到 TPU 和 matrix multiplication 的关系：TPU 的 systolic array 天然 fit 矩阵乘的 data flow，neutral atom 的 AOD 天然 fit translation-invariant code 的 operation。

### 12.8 未来方向的联想

Paper 提到 ultra-high-rate code 的 replica parallelism。如果有一个 $k/n \sim 1/2$ 的 code，logical qubit 分成 $N_{\text{sub}}$ 个 internally cyclic subset，每个 subset 跑一个 replica，那 throughput 可以再翻 $N_{\text{sub}}$ 倍。挑战是 STAR injection 要 disjoint physical representative，dense packing 可能 break 这个。可能的解法是 batched injection（分批 inject，每批 disjoint）或 hybrid（小角度用 STAR，大角度用 discrete magic factory）。

另一个联想是把这个 framework 扩展到 VQE 或 variational algorithm。这些 algorithm 也大量用小角度 $R_Z$ rotation，而且结构上也是 Trotterized evolution 的变体。如果目标 ansatz 有 translation symmetry（比如 lattice Hamiltonian 的 variational ansatz），同样的 co-design 可以 apply。

最后，这篇 paper 的 end-to-end resource estimate 方法论值得学习：从 Trotter error bound 出发，选 operating point，decompose circuit 成 architecture-specific primitive，用 circuit-level simulation 拟合 per-gadget error model，最后组合成 physical qubit count 和 runtime 的 Pareto frontier。这种"top-down from algorithm + bottom-up from hardware"的 approach 可以借鉴到其他 quantum architecture 的 resource estimation。

---

## 总结

这篇 paper 的核心贡献是**第一个 fully evaluated 的高码率 STAR 架构**，通过 symmetry-driven co-design 把 translation symmetry 贯穿 algorithm、QEC code、hardware 三层。Bicycle chain code 提供了 tunable 的高码率 code family（$k/n = 1/7$，vs surface code 的 $1/d^2$），原生支持 global transversal Clifford + cyclic shift + parallel STAR injection。End-to-end simulation 验证了 $8 \times 8$ TFIM 只需 2,240 qubit / $\sim 200$ s，FH 只需 $\sim 6,300$ qubit / $\sim 200$ s，相比 surface code baseline 有 $\sim 5.5\times$ space reduction，相比 code-surgery 架构有 $100\text{-}1000\times$ speedup。这给 early fault-tolerant era 的 quantum simulation 提供了一条 low-resource pathway。

核心技术 insight：
1. Disjoint logical basis 让 parallel STAR injection 把 per-logical RUS cost 从 $\sim 2$ 降到 $\log_2(k)/k$。
2. Self-dual BB code 的 even-weight syndrome 让 TMR factory 只需 2 轮 SE（surface code 要 3 轮）。
3. Translation symmetry 把所有 logical operation 映射到少数几个 global AOD primitive。
4. Heralded atom loss 把 infidelity 降 $(1-f)$ 倍，acceptance 几乎不变。

这让我对 quantum simulation 的 early fault-tolerant 路线有了更清晰的 picture：不需要等 fully fault-tolerant 的 $d \sim 30$ surface code，几千 qubit + 几百秒就能跑 megaquop-scale simulation，这可能是量子计算第一个真正有 practical advantage 的 application。
