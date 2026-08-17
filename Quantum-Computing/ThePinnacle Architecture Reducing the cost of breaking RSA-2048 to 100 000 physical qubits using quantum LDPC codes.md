---
source_pdf: ThePinnacle Architecture Reducing the cost of breaking RSA-2048 to 100
  000 physical qubits using quantum LDPC codes.pdf
paper_sha256: 23e7a3246a4e6886e0f85ab37db01a5538c4fe6339fd3c71f0c89d77783527ca
processed_at: '2026-08-12T15:16:23-07:00'
target_folder: Quantum-Computing
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Paper

Andrej，我换个角度，像跟朋友聊天一样讲一遍。

参考链接：
- 原文: https://arxiv.org/abs/2505.15917
- Gidney 2025 (前作): https://arxiv.org/abs/2505.15917
- IBM Tour de gross: https://arxiv.org/abs/2506.03094
- Litinski surgery: https://arxiv.org/abs/1808.02892
- Webster gadgets: https://arxiv.org/abs/2511.15989
- Mohseni scaling roadmap: https://arxiv.org/abs/2411.10406
- Gidney cultivation: https://arxiv.org/abs/2409.17595
- Sahay fold-transversal: https://arxiv.org/abs/2509.05212

---

## 一句话总结

之前大家以为破 RSA-2048 至少要 100 万个 physical qubits，这篇 paper 说用 QLDPC codes 加一些新招，10 万就够了，而且能跑一个月跑完。

---

## 为什么之前要 100 万

量子计算机有 noise，必须用 error correction。Surface code 是目前最成熟的方案，但它的 "encoding rate" 很糟糕：要保护 1 个 logical qubit，得用 2d² 个 physical qubits。对 d=24（需要这个 distance 来对抗 p=10⁻³ 的 noise），就是 ~1151 physical qubits 保护 1 个 logical qubit。

RSA-2048 估计需要 ~2000 个 logical qubits 来跑 Shor 算法。2000 × 1151 ≈ 230 万？不对，因为 Gidney 用了一些优化把 logical qubit 数压下来，最后大约 2 万 logical qubits 跑算法、加上 magic state distillation factory 之类，total 物理量子比特 ~100 万。

总之 surface code 太"费料"了。每多保护一个 logical qubit，就要堆上千个 physical qubits。

---

## QLDPC 的承诺

QLDPC codes (Quantum Low-Density Parity-Check) 是另一类 error-correcting codes。关键区别：surface code 只能用 nearest-neighbor interaction（只能跟隔壁 qubit 说话），QLDPC 放宽这个限制，允许"远距离"交互（但仍然是有限范围）。

放宽 connectivity 之后，一个 code block 可以 encode 多个 logical qubits。Paper 用的 generalized bicycle codes，d=24 的版本 [[510, 16, 24]]，510 个 physical qubits 编码 16 个 logical qubits。每 logical qubit 只花 32 个 physical qubits，surface code 是它的 35 倍贵。

所以理论上，如果能把 QLDPC 用起来做 fault-tolerant quantum computing，physical qubit 数应该能降一个数量级。

---

## 之前 QLDPC 卡在哪

QLDPC codes 在 "memory" 任务上早就 beat surface code 了：把数据存进去、保持 coherence，没问题。但要做 quantum computation——也就是在 logical qubits 上做 gate——非常麻烦。

Surface code 用 lattice surgery 做 logical gate：把两个 code block "粘"在一起测量一个 joint Pauli operator，就能做 CNOT 或者别的 gate。简单粗暴。

QLDPC 一个 block 有 k 个 logical qubits（比如 16 个），但你想测量任意一个 logical Pauli operator（比如 X̄₁ X̄₃ Z̄₇）时，没有简单办法。IBM 的 Tour de gross 用 bivariate bicycle codes + 一些 surgery gadget，但只支持一个 subset of Pauli measurements。这意味着做某些 gate 要花多轮 surgery，time overhead 上去了。

所以之前 QLDPC 架构的 trade-off：空间省了，时间贵了。总 spacetime overhead 没明显胜出。

---

## Pinnacle Architecture 怎么破的

这篇 paper 的关键就是同时解决三个问题：

### Trick 1: Generalized surgery gadgets

这其实来自作者自己的前作 (Ref [11], Webster et al.)。核心：给 QLDPC code block 加一个 "gadget system"——一群额外的 physical qubits 配置好，使得测量任意 logical Pauli operator 都等价于在 gadget 上做一组 parity check measurements。

具体怎么搞：选 4 个 "seed operators"——2 个 X-type, 2 个 Z-type，分别对应 L 和 R sector。这 4 个 seed 的 cyclic shifts（利用 GB code 的循环对称性）能 span 出所有 logical Pauli operators。所以只需要构造 4 个 gadget，加上 cyclic shift 机制，就能测任意 Pauli。

每个 processing block 用 4 个 gadget + 4 个 bridge（3 个内部连接 + 1 个连到下一个 block）。Physical qubit 数从纯 code block 的 510（d=24）涨到 1620——大约 3 倍，但还是比 surface code 便宜 10 倍多。

任意 logical Pauli product measurement 在 1 个 logical cycle（= d+2 code cycles）内完成。Time overhead 没有增加。

### Trick 2: Magic Engine（这个是真创新）

传统 magic state distillation 是 "batch model"：蒸馏一批 |T̄⟩ states，存起来，慢慢用。这需要专门的 distillation factory，spacetime overhead 巨大。

Pinnacle 的 magic engine 是 "pipeline model"：一个 code block 分两个 sector (L 和 R)。
- Odd cycle: L sector 做 15-to-1 distillation 蒸一个新 |T̄⟩; R sector 把上一个 cycle 蒸好的 |T̄⟩ 注入到 processing unit
- Even cycle: 角色互换

每个 logical cycle 产出 + 消费一个 |T̄⟩。Distillation 的 latency 完全 hide 在 computation 后面。

这个 trick 利用了 QLDPC 一个 block 有多个 logical qubits 的事实。Surface code 一个 block 只编码 1 个 logical qubit，玩不了这种 sector 分工。这是 QLDPC 架构的 "free lunch"——k 越大越省。

每个 processing unit 配一个 magic engine。Magic engine 的 physical qubit count：~2128（p=10⁻⁴ 时用 colour code injection）或 ~8694（p=10⁻³ 时用 surface code cultivation，更复杂）。

Reject rate 处理：distillation 有可能失败（post-selection reject），p_r ≈ 0.15%（p=10⁻⁴）或 6%（p=10⁻³）。Reject 时下一 cycle idle 重试。平均每个 T gate 花 1/(1-p_r) 个 logical cycle。

### Trick 3: Clifford Frame Cleaning

Pauli-based computation 框架（Bravyi-Smith-Smolin）：任意 Clifford+T 电路可以转成一系列 Pauli measurements，每个 T gate 配一个 |T̄⟩。Clifford gate 全部 commute 到末尾 absorb 进 final measurement。

这个框架对单个 processing unit 很好——每个 logical cycle 做 1 个 Pauli measurement。但多个 processing unit 想并行就麻烦了。

问题：两个 unit 间有 CNOT gate，这个 CNOT 在 Pauli-based 框架里会"进入 Clifford frame"（一个待处理的累积 Clifford 算符）。后续 commute 时，一个 unit 上的 logical measurement 的 support 会蔓延到另一个 unit。结果：两个 unit 上的 T gate 必须串行做。

Tour de gross 的解法（Extractors, Ref [34]）：物理实现这个 CNOT，让它从 frame 里出去。但每个 inter-unit CNOT 都要花 logical cycle，如果电路有大量 inter-unit gate，cost 高。

Pinnacle 的解法：先让 CNOT 留在 frame 里（不物理实现），但每当想恢复并行时，花 4κ 个 Pauli π/4 rotation "clean" frame（κ = 一个 unit 的 logical qubit 数）。Clean 完之后两 unit 再次独立。

Intuition: 像是"累积账单"。Inter-unit CNOT 进 frame 等于记账。想并行就 "结账"——花 4κ 个 measurement 把账清空。对长 circuit，这个 amortized cost 很低。

Memory port 更便宜：port 上的 fan-out CNOT 的 control 在 port 上，commute 时只产生 Z-type Pauli on port，clean 只要 2w（w = window size）。

---

## 结果有多牛

Paper 给了两个 benchmark:

### Fermi-Hubbard

模拟一个 16×16 格点的 Hubbard model，精度 0.5%。Pinnacle 需要 62k physical qubits (p=10⁻³) 或 22k (p=10⁻⁴)。Surface code 方案需要 940k / 200k。一个数量级提升。

Runtime：1.6–3.6 分钟 (t_c=1µs) 或 1.1–2.5 天 (t_c=1ms)。

### RSA-2048

p=10⁻³, t_c=1µs（superconducting 假设）：
- 1 month: 98k qubits
- 1 week: 151k qubits
- 1 day: 471k qubits
- 10 hours: 1M qubits

对比 Gidney 2025：1M qubits 跑 5 天。

p=10⁻⁴, t_c=1ms（trapped ion 假设）：
- 1 month: 3.1M qubits
- 1 week: 14M qubits

对比 Beverland 2022 (Ref [58])：8.6M qubits 跑 3 年。

p=10⁻³, t_c=1ms（neutral atom 假设）：
- 19M qubits 跑 21 天

对比 Zhou et al. 2025 (Ref [16]) 用 transversal gates：5.6 天。这个对比 Pinnacle 略输，因为 transversal gates 把 logical cycle time 压到了 code cycle time（无 d_t 倍数），而 Pinnacle 还在 d_t = 26。作者说未来用 fast surgery (Ref [59]) 可能反超。

---

## 关键数字直觉

为什么是 ~10 万而不是 50 万？

RSA 算法需要 ρ 个 working register 并行（ρ 是优化变量，paper 没明确给最优值，但从结果反推 ρ 大约几十个）。每个 working register 大约 2 个 processing block，每个 block 1620 physical qubits (d=24)。

ρ × 2 × 1620 = 几十万，加上 ρ 个 magic engine (~8694 each) = 又几十万... 这就超 100k 了。

所以 ρ 应该比较小，可能 ~20-30 个。具体数字在优化空间里。但 trend 明显：通过 sharing input register + memory + parallelism，把 ρ 的 cost 压下来。

公式的核心（公式 15）：
$$N = \left\lceil \frac{\rho}{\lceil m/w_1 \rceil} \right\rceil m + \rho N_w$$

变量：
- ρ：并行 working register 数
- m：input register 大小（~2000+ for RSA-2048）
- w_1 = k/2：lookup window size
- N_w：单 working register 大小（~30-50）

关键 insight：input register 可以被 ρ 个 working register 共享，只要 ρ ≤ ⌈m/w_1⌉ ≈ 200+。所以 input cost 只算 1 份 m ≈ 2000，working cost 算 ρ 份 N_w ≈ 30。总 logical qubit ≈ 2000 + 30ρ。

ρ=30 时：2000 + 900 = 2900 logical qubits。每个 logical qubit ~100 physical qubits (含 processing block + magic engine overhead amortized) → ~300k physical qubits. 嗯，跟 Table VI 的 ~100k 还有差距，说明 ρ 可能更小，或者 magic engine overhead 被 ρ 个 processing unit 共享一些（其实每个 unit 配一个 magic engine，所以是 ρ 个 magic engine，不省）。

具体 ρ 在最优化里被 code distance、f、ℓ、s 等参数联合优化，不是单一值。可能 ρ~10-20 量级。总之优化器找到了一个甜蜜点。

---

## 但有几个 caveat

1. **Decoder 没解决**: Paper 用 most-likely error decoding，是 mixed integer program，理论上 optimal 但实际跑很慢。Real-time decoder for QLDPC 是 open problem。如果 decoder 跑不过 code cycle time，整个 system 慢下来。

2. **Code parameters 是猜的**: GB code family 的 [[2(2^m-1), 2m, m+(m-4)²]] 没 rigorous 证明。前几个 (m=4,5,6,7,8) numerical verified，但更大 code 依赖外推。

3. **Connectivity 要求**: QLDPC 需要 "quasi-local" interaction——比 nearest-neighbor 远，但仍然 bounded range。Superconducting 需要长程 coupler 或 shuttle。Trapped ion (all-to-all) 和 neutral atom (reconfigurable) 更适合。

4. **Cultivation success rate**: p=10⁻³ 时依赖 fold-transversal cultivation，success 2/3，5 次尝试后 94% 总 success。这是 Sahay 2025 的新技术，还需实验验证。

5. **跟 IBM Tour de gross 的 head-to-head**: 两者用相似 code families（bivariate vs generalized bicycle），关键差别在 gadget 能力。Pinnacle 的 generalized gadget 支持 arbitrary Pauli measurement，Tour de gross 只支持 subset。但 Tour de gross 已经有 IBM 的 hardware roadmap 支持。Pinnacle 目前是 Iceberg Quantum 一家之言。

6. **跟 neutral atom transversal 方案的对比**: Zhou et al. (Ref [16]) 用 transversal gates 把 logical cycle time 压到 code cycle time，runtime 上赢。Pinnacle 的 d_t=26 是个 penalty。作者提到 fast surgery (Ref [59]) 可能反超，但 future work。

---

## 给 Karpathy 的 ML 类比

你说"build intuition"，我尝试用 ML 类比帮你 build:

### 类比 1: Surface code vs QLDPC = Thick model vs Mixture of Experts

Surface code 像 BERT-style model：每层都是 dense 的，简单但 parameters 多。
QLDPC code 像 Mixture of Experts：每个 token 只激活部分 experts，parameters 利用率高，但 routing 复杂。

Surface code "routing 简单" 是因为只有 1 个 logical qubit per block，surgery 直接。QLDPC "routing 复杂" 是因为 k 个 logical qubit per block，surgery 要选 operator。

### 类比 2: Magic Engine = Pipeline Parallel Training

Pipeline parallel 把 forward pass 和 backward pass overlap 在不同 GPU stage 上。Magic engine 把 distillation (produce) 和 injection (consume) overlap 在不同 sector 上。

Reject rate 类似 pipeline bubble——不完美 overlap 导致 idle cycle。p_r=6% 类似 6% 的 bubble overhead。

### 类比 3: Clifford Frame Cleaning = Gradient Checkpointing

Gradient checkpointing: 不存所有中间 activation，需要时 recompute。Trade memory for compute.

Clifford frame cleaning: 不物理执行 inter-unit CNOT (省 time)，但想并行时花 4κ measurements "recompute" 出 frame 分解。Trade time for parallelism.

4κ 是 recompute cost，类似 recompute forward pass。当 circuit 长且 inter-unit gate 不频繁时，cleaning cost amortized 得很便宜。

### 类比 4: Code Distance Scaling = Model Scaling Laws

Logical error rate (公式 5):
$$p_L = A \left(\frac{p}{B}\right)^{d/2 + C}$$

类似 scaling law:
$$L = A \cdot (N/N_0)^{\alpha}$$

Code distance d 像 model size，physical error rate p 像 data quality。Threshold B 像 "critical batch size"——超过它 scaling 变好。指数 d/2+C 像 power law exponent。

在 ML 里你 fit scaling law 然后 extrapolate。这里作者也 fit A, B, C 然后外推到更大 d（d=24 是外推出来的，只 numerical 验证到 d=10）。

### 类比 5: Pauli-based Computation = Compiler Optimization

Pauli-based compilation: 把 Clifford gates 全部 defer 到末尾，运行时只执行 Pauli measurements + T injections. 类似 JIT compiler 把 computation graph 重排优化。

Clifford frame 是 "lazy execution"——不立即做 Clifford，记下来等需要时再处理。Cleaning 是 "flush"——把累积的 Clifford 操作 settle 掉。

### 类比 6: Modularity = Data Parallel Training

Pinnacle 架构的 processing units 是 modular 的，可以并行跑独立 task（比如 RSA factoring 里 ρ 个 working register 并行）。类似 data parallel：每个 worker 跑独立 batch，最后 reduce。

Memory 的 read-only parallel access 像 parameter server——多个 worker 共享同一份 weights，read 不冲突。

---

## 真正的 Big Picture

这篇 paper 的意义不在于 100k 这个数字本身，而在于它证明 QLDPC 架构已经从"理论可能"变成了"工程可行"。三件事同时成立:
1. QLDPC 的空间优势 (vs surface code)
2. QLDPC 的时间不亏 (vs Tour de gross)
3. Modular, hardware-compatible (quasi-local connectivity)

如果 Iceberg Quantum 的 simulator 结果可信，且 decoder 问题能解决，那 100k qubits 是一个 hardware engineering 问题，不再是 architecture 问题。

Mohseni et al. (Ref [9]) 的 roadmap 预测 100k qubit device 在 2030s 中期可能出现，1M qubit 要等到 2040s。如果 Pinnacle 成立，RSA-2048 的威胁 timeline 提前 5-10 年。

对 ML 的影响：用 fault-tolerant quantum computer 跑 quantum chemistry (Fermi-Hubbard 之类的) 在 22k qubits 就能做。Quantum ML algorithms (HHL, quantum SVM) 也能用这个架构 estimate resource。

总结一句：surface code 时代可能要结束了，QLDPC 时代可能要开始了。但需要 decoder 突破 + code parameter 证明 + cultivation 实验验证，这三个 milestone 还没到。

更多 reading:
- https://arxiv.org/abs/2308.07915 (Bravyi et al. high-threshold QLDPC)
- https://arxiv.org/abs/2407.18393 (Improved QLDPC surgery)
- https://arxiv.org/abs/2410.02753 (Homological measurement)
- https://arxiv.org/abs/2502.19406 (GB codes single-shot decoding)
- https://arxiv.org/abs/2503.10390 (Extractors, IBM Pauli-based QLDPC)
- https://arxiv.org/abs/2510.04521 (Fast surgery, 同作者)
- https://arxiv.org/abs/2503.05003 (Parallel logical measurements)

---

# Pinnacle Architecture 深度解析

Andrej，这篇 paper 是 Iceberg Quantum 团队的工作，核心 claim 相当大胆：用 QLDPC codes 把 RSA-2048 factoring 的 physical qubit 成本从 Gidney 2025 年的 ~1 million 降到 <100,000。我下面从 architecture、code family、magic engine、Clifford frame cleaning、RSA compilation 几个层面逐步拆解，build 一下你的 intuition。

参考链接：
- arXiv: https://arxiv.org/abs/2505.15917 (Gidney, "How to factor 2048 bit RSA integers with less than a million noisy qubits")
- arXiv: https://arxiv.org/abs/2506.03094 (Tour de gross, bivariate bicycle codes, IBM)
- arXiv: https://arxiv.org/abs/2511.15989 (Webster et al., low-overhead gadgets for QLDPC)
- arXiv: https://arxiv.org/abs/2103.06309 (Breuckmann & Eberhardt, QLDPC review)
- arXiv: https://arxiv.org/abs/1808.02892 (Litinski, "Game of surface codes")
- arXiv: https://arxiv.org/abs/1905.06903 (Litinski, magic state distillation)
- arXiv: https://arxiv.org/abs/2503.10390 (Extractors, QLDPC Pauli-based computation)
- arXiv: https://arxiv.org/abs/2510.04521 (Fast surgery for QLDPC)
- arXiv: https://arxiv.org/abs/2509.05212 (Fold-transversal surface code cultivation)

---

## 1. 整体定位 & 核心洞察

Surface code 的 fundamental 问题在于 encoding rate 太低：[[d², 1, d]] code 用 ~2d² physical qubits 才编码 1 个 logical qubit。Gidney 2025 的 surface code 估算需要 ~10⁶ physical qubits。QLDPC codes 放宽 nearest-neighbour connectivity 约束后，可以做到 [[n, k, d]] with k ~ Θ(n/log n) 或更高，于是 physical qubits per logical qubit 从 d² 量级降到 ~50–100 量级。

但 QLDPC 一直面临一个 fundamental tension：高 encoding rate 往往伴随高 time overhead，因为 multi-logical-qubit code block 上做任意 logical Pauli measurement 需要复杂的 surgery。IBM 的 Tour de gross（bivariate bicycle codes）用 modular 的方式做 surgery，但只支持 subset of logical Pauli measurements，导致 time overhead 增加。Pinnacle 的关键 insight：用 generalized surgery gadgets（Ref [11]，同作者的前序工作）使得任意 logical Pauli product measurement 都可以在单个 logical cycle 内完成，从而在保持 QLDPC 空间优势的同时避免 time penalty。

---

## 2. Architecture 三大模块

### 2.1 Processing Unit

一个 processing unit 包含 β 个 processing block 排成 line，相邻 block 用 bridge 连接。每个 processing block = QLDPC code block + gadget system + bridges。

Physical qubit accounting（公式 3）：
$$n_{pb} = n_{cb} + 4 n_g + 4 n_b$$

变量含义：
- $n_{cb} = 2n$：code block physical qubits（n 个 data qubits + n 个 check qubits）
- $n_g$：单个 gadget 的 physical qubits
- $n_b$：单个 bridge 的 physical qubits
- 4 个 gadgets 对应 4 个 seed operators（每个 logical sector 1 个 X-type + 1 个 Z-type seed）
- 4 个 bridges：3 个用于 block 内部 gadget 互连，1 个用于连接到下一个 block

对 d=16 GB code：n_pb = 508 + 4×57 + 4×31 = 860
对 d=24 GB code：n_pb = 1020 + 4×99 + 4×51 = 1620

每个 processing block 编码 k 个 logical qubits，β 个 block 编码 κ = βk 个 logical qubits。任意 logical Pauli product operator on 这 κ 个 logical qubits 都可以在一个 logical cycle 内测量。

关键点：connectivity 限制在单个 processing block 的尺度（约 √n_pb in 2D），constant in logical qubit number。这是 scalability 的核心保证。

### 2.2 Magic Engine

这是 paper 的真正创新点。传统 magic state distillation 是 batch model：distill 一批 |T̄⟩ states，存起来，慢慢用。Pinnacle 改成 pipeline model：每个 logical cycle 同时 distill 一个新 state + inject 上一个 cycle 蒸馏好的 state。

构造：把 QLDPC code block 的 logical qubits 分成 L 和 R 两个 sectors（各 k/2 个 logical qubits）。
- Odd logical cycle：L sector 做 15-to-1 distillation，R sector 通过 joint measurement 把上一个 cycle 蒸馏好的 |T̄⟩ 注入到 processing unit
- Even logical cycle：L 和 R 角色互换

Magic engine 的 physical qubit accounting（公式 4）：
$$n_{me} = n_{cb} + 16 n_g + 60(n_a + d_a - 1) + n_\alpha$$

变量含义：
- $n_{cb}$：GB code block 本身
- $16 n_g$：16 个 gadgets（因为 15-to-1 distillation 需要并行 15 个 injection + 4 个 post-selection measurement，总共 16 个并行 logical measurement）
- $60(n_a + d_a - 1) = 2 \times 15 \times (2n_a - 1)$：15 对 ancilla codes，每对是两个 [[n_a, 1, d_a]] codes（一个持 |T⟩，一个做 auto-correction）
- $n_\alpha$：magic state cultivation 额外 ancilla

Reject rate 估算：
- p=10⁻⁴：用 colour code post-selected injection，p_r ≈ 15 p_in = 0.15%
- p=10⁻³：用 surface code fold-transversal cultivation，cultivation 本身 success rate 2/3，5 次尝试后 success prob (1 - 1/3⁵)¹⁵ ≈ 94%，所以 p_r ≈ 6%

Magic state output fidelity：p_T ≈ 35 p_in³（15-to-1 distillation 的标准结果），target p_T ≈ 10⁻¹¹ for RSA。

Pipeline 的本质：把 magic state distillation 的 latency 完全 hide 在 computation 后面。每个 processing unit 配一个 magic engine，每个 logical cycle 持续提供 1 个 high-fidelity |T̄⟩。这意味着 T count τ 直接对应 τ 个 logical cycles（modulo reject rate）。

### 2.3 Memory（optional）

Memory 用纯 code block（不带 gadget），通过 port 被 processing unit 访问。Window size w = k/2，每个 port 对应一个 Z-type gadget + 一个 bridge。

Memory code blocks 排成 loop，通过 cyclic shift 实现访问任意 window。每个 logical cycle 最多 shift 一个位置，所以 ν 个 code block 需要 ν 个 logical cycles 让每个 processing unit 访问全部 memory。

Memory 的 physical qubit accounting：
- d=16: 508ν + 88ρ（ν 个 memory block + ρ 个 ports）
- d=24: 1020ν + 150ρ

关键点：多个 processing unit 可以 parallel read-only 访问 memory，因为 fan-out 操作（CNOT with control on memory, target on processing unit）commute。

---

## 3. Generalised Bicycle Codes 细节

GB codes 由 lift l ∈ ℕ 和 sets A, B ⊆ ℤ_l 定义。Parity check operators（公式 1, 2）：
$$S_{X,j} = \prod_{a \in A} X_{(j+a),L} \prod_{b \in B} X_{(j+b),R}$$
$$S_{Z,j} = \prod_{a \in A} Z_{(j-a),R} \prod_{b \in B} Z_{(j-b),L}$$

变量含义：
- $l$：lift parameter，code 有 2l 个 physical qubits（L 和 R sectors 各 l 个）
- $j \in \mathbb{Z}_l$：parity check 的 index
- $A, B \subseteq \mathbb{Z}_l$：定义 parity check 的 support sets
- 第一个下标 $(j+a)$：qubit 在 sector 内的位置（cyclic）
- 第二个下标 $L$ 或 $R$：sector label
- Parity check weight = |A| + |B|

这个构造有 cyclic shift automorphism：把所有 physical qubits 平移 σ 个位置，parity check group 保持不变。这是 gadget 设计的关键——seed operator 的 cyclic shift 可以覆盖所有 logical operators。

具体使用的 code family（Table I）：
| [[n,k,d]] | d_t | l | A | B | n_cb | n_g | n_b | n_pb |
|----------|-----|---|---|---|------|-----|-----|------|
| [30,8,4] | 6 | 15 | {0,6,13} | {0,1,4} | 60 | 13 | 7 | 140 |
| [62,10,6] | 8 | 31 | {0,6,15} | {0,5,7} | 124 | 19 | 11 | 244 |
| [126,12,10] | 12 | 63 | {0,4,37} | {0,29,49} | 252 | 31 | 19 | 452 |
| [254,14,16] | 18 | 127 | {0,32,100} | {0,28,49} | 508 | 57 | 31 | 860 |
| [510,16,24] | 26 | 255 | {0,39,55} | {0,70,127} | 1020 | 99 | 51 | 1620 |

Conjectured parameters: [[2(2^m - 1), 2m, m + (m-4)²]]。注意 d 的 scaling 是 m + (m-4)² ≈ m²，而 k = 2m 是 log(n)。Encoding rate k/n ≈ m/(2^(m-1))，随 m 指数下降——这是 QLDPC 的典型 trade-off，distance 增长比 k 快得多。

d_t = d + 2：empirically 发现多做 2 轮 syndrome extraction 能改善 logical error rate。Logical cycle time t_l = d_t × t_c。

---

## 4. Logical Error Rate 拟合

Ansatz（公式 5, 6）：
$$p_{L,cb}(p, d) = A \left(\frac{p}{B}\right)^{d/2 + C}$$
$$p_L(p, k, d) = \frac{A}{k} \left(\frac{p}{B}\right)^{d/2 + C}$$

变量含义：
- $p$：physical error rate（circuit-level depolarising）
- $d$：code distance
- $A, B, C$：fit parameters（distance-independent，允许外推）
- $p_{L,cb}$：所有 k 个 logical observables 在 d_t 轮后的总失败率
- $p_L$：per logical qubit per logical cycle 的失败率

Fitted parameters（Table II）：
| Experiment | A | B | C |
|-----------|---|---|---|
| Memory | 5.9 | 0.0179 | 0.50 |
| Log. Meas. | 6.2 | 0.0158 | 0.47 |

B ≈ 0.016–0.018 是 threshold，~1.6%。这比 surface code 的 ~1% threshold 高一些。指数 d/2 + C 而不是 d，说明 error correction 在 sub-threshold regime 的 scaling 不完全是 distance-limited，C ≈ 0.5 给了一个常数修正。

Table III 给出关键数字：
| p | d=4 | d=6 | d=10 | d=16 | d=24 |
|---|-----|-----|------|------|------|
| 10⁻³ | 8×10⁻⁴ | 4×10⁻⁵ | 1×10⁻⁷ | 3×10⁻¹¹ | 4×10⁻¹⁶ |
| 10⁻⁴ | 3×10⁻⁶ | 1×10⁻⁸ | 5×10⁻¹³ | 1×10⁻¹⁹ | 1×10⁻²⁸ |

RSA-2048 需要 p_L ≲ 10⁻¹⁴，所以 p=10⁻³ 选 d=24，p=10⁻⁴ 选 d=16。

Decoder 选择：most-likely error decoding（mixed integer program，optimal solution）。作者明确说 real-time fast decoder 是 future work。这是一个 caveat——实际 runtime 可能受 decoder 速度限制。

---

## 5. Pauli-Based Computation & Clifford Frame Cleaning

### 5.1 Pauli-Based Computation

Bravyi-Smith-Smolin 的框架（Ref [13]）。任意 κ-qubit circuit with T count τ 和 o 个 intermediate measurements 可以用 τ + κ + o 个 Pauli measurements 实现，加上 τ 个 |T̄⟩ states。

Compilation：把每个 T gate 替换成 magic state injection circuit（一个 Pauli measurement），所有 Clifford gates 通过 conjugation commuted 到 circuit 末尾，absorb 进 final measurement。因为 Clifford map Pauli to Pauli under conjugation，结果还是 Pauli measurements。

这是为什么 Pinnacle 的 logical cycle count scales with T count，而 logical qubit count scales with κ。

### 5.2 Clifford Frame Cleaning（核心创新）

Pauli-based computation 的 parallelism 问题：当两个 processing unit 之间有 entangling Clifford gate（比如 CNOT），commute Clifford 到末尾时，一个 unit 上的 logical measurement 的 support 会 spread 到另一个 unit。具体说，一个 |T̄⟩ injection 对应的 logical measurement 会 straddle 两个 unit。每个 unit 一个 logical cycle 只能做一次 logical measurement，所以两个 unit 上的 T gate 无法并行。

Tour de gross 的解法（Ref [34] Extractors）：物理实现 CNOT 把两个 unit 真正 entangle，这样 CNOT 不进入 Clifford frame。但代价是每个 inter-unit CNOT 都要花时间，如果 circuit 一部分 highly parallelisable 另一部分不是，parallelisable 部分的 cost 会 scale with 非并行部分的 CNOT 数。

Pinnacle 的解法：Clifford frame cleaning。

定义：clean Clifford frame C of K'（K' ⊂ K）意味着物理执行一个 Clifford U 使得 CU 在 K̄' 上 trivially act。

Lemma 1：clean w 个 qubits 最多需要 4w 个 Pauli π/4 rotations（每个 rotation = 1 个 logical Pauli measurement with |0̄⟩ ancilla）。
Lemma 2：对于 memory port（Clifford 只 act as CNOT control on port），只需要 2w 个 rotations。

Proof 思路（Appendix A）：用 symplectic representation。n-qubit Pauli 对应 v ∈ ℤ₂^{2n}，Clifford 对应 2n×2n symplectic matrix M。Pauli π/4 rotation R_{π/4}(P_u) 对应 E_u(v) = v + ⟨u,v⟩u。目标是通过 4w 个这样的 elementary operation 把 M 变成只在后 n-w 个 qubit 上有 support 的形式。归纳法：每步用 4 个 rotation 把第 k 行映射到 e_k（标准基），同时保持 symplectic structure。

Operation framework：
1. 两个 processing unit 初始独立并行
2. 电路某点有 inter-unit CNOT → Clifford frame entangling → 两 unit join，之后只能串行
3. 任何后续时刻，可以用 ≤ 4κ 次 logical measurement clean frame → 两 unit separate → 恢复并行

对 memory access：fan-out CNOT 的 control 在 port 上，commute Clifford frame 只产生 Z-type measurement on port。所以 clean memory port 只要 2w 次（w = port window size = k/2）。

直觉：Clifford frame cleaning 像是 "undo entanglement in the Clifford frame"。如果你不想物理做 CNOT，就让 CNOT 留在 frame 里，等需要并行的时候花 4κ 的 cost 把它 "踢出去"。这个 4κ 是 amortized cost，对长 circuit 来说很便宜。

---

## 6. RSA-2048 Factoring 算法

### 6.1 Gidney Algorithm 概述

Gidney 2025 的算法（Ref [8]）基于 Ekerå-Håstad 和 Chevignard 等的技术。用 residue number system：把 mod N_RSA 的算术替换成 mod p_i（p_i 是小 prime，size polylog in N_RSA）的算术。Working register 从 Θ(log N_RSA) 降到 Θ(log log N_RSA)。

代价：|P| 个 prime 串行处理，|P| ≈ 2.1×10⁴ for RSA-2048。

### 6.2 并行化（Pinnacle 的第二个创新）

把 working register 复制 ρ 份，P 分成 ρ 个 subset，每个 subset 在一个 register 上独立跑。最后用 parallel reduction（binary tree of additions）合并 accumulators。

时间成本（公式 14）：
$$\mathcal{T} = \frac{\lceil |P|/\rho \rceil}{|P|} \mathcal{T}_G + O(\log \rho)$$

Logical qubit 数（公式 15）：
$$N = \left\lceil \frac{\rho}{\lceil m/w_1 \rceil} \right\rceil m + \rho N_w$$

变量：
- $m = \Theta(\log N_{RSA})$：input register size
- $N_w = \Theta(\log \log N_{RSA})$：working register size
- $w_1 = k/2$：loop 1 window size（fixed）
- $\rho$：parallelisation factor

关键 insight：input register 可以被多个 working register 共享（lookup 操作只有 control 在 input register，commute）。当 ρ ≤ ⌈m/w_1⌉，只需要 1 个 input register copy。对 ρ ≲ 200，N ≈ m + ρ N_w，远小于 ρ(m + N_w)。

每个 working register 的 logical qubit 数（公式 16）：
$$\kappa = f + 2\ell + \text{len}(m) + 2\max(f, \ell + \text{len}(m)) + 1$$

变量：
- $f$：truncated accumulator length
- $\ell$：residue prime bit length
- $\text{len}(m) = \lfloor \log_2 m \rfloor + 1$：m 的 bit length
- 2 个 ancilla sub-register（size max(f, ℓ+len(m))）+ 1 个 ℓ-size ancilla + 1 个 Toffoli ancilla

### 6.3 Physical qubit accounting

Working registers（公式 17）：
$$n_w = \rho \left( n_{pb} \left\lceil \frac{\kappa(f, \ell, m)}{k} \right\rceil + n_{me} \right)$$

Memory + ports（公式 18）：
$$n_m = 2n \left\lceil \frac{\rho}{\lceil m/w_1 \rceil} \right\rceil \left\lceil \frac{m}{k} \right\rceil + \rho(n_g + n_b)$$

Total: n_total = n_w + n_m

### 6.4 Runtime

每 prime 的 logical cycle 数 Σ = Table V 各行 Logical Cycles 列求和。

Total logical cycles（公式 19, 20）：
$$\mathcal{T}' = \lceil |P|/\rho \rceil \Sigma + v + 6(f-1)\lceil \log_2 \rho \rceil$$
$$\mathcal{T} = \left(\frac{2}{3}(1-p_r)^{-1} + \frac{1}{3}\right) \mathcal{T}'$$

变量：
- $v = \lceil m/w_1 \rceil (6(2^{w_1} - w_1 + \ell + \text{len}(m) - 2) + 2w_1)$：loop 1 uncompute 的 logical cycles
- $6(f-1)\lceil \log_2 \rho \rceil$：parallel reduction 的 addition 成本
- 系数 2/3：约 2/3 的 logical cycles 需要 T state
- $(1-p_r)^{-1}$：magic engine reject 导致的 slowdown

Shots 数（公式 21, 22）：
$$\sigma = \frac{s+1}{0.99 p_S (1 - 2 N_{RSA} \sqrt{(s+2)/(2^{f+1} s w_1)})}$$
$$p_S = (1-p_L)^{N\mathcal{T}} (1-p_T)^\tau$$

变量：
- $s$：Ekerå-Håstad parameter
- $p_S$：单 shot 无 logical error 的概率
- $\tau$：T count per shot（公式 23）

Total runtime: t_total = σ × t_l × T = σ × d_t × t_c × T

---

## 7. 实验结果分析

### 7.1 Fermi-Hubbard（Table IV）

Hamiltonian（公式 7）：
$$H = \sum_{\langle i,j \rangle} \sum_\sigma (a^\dagger_{i,\sigma} a_{j,\sigma} + h.c.) + u \sum_i \hat{n}_{i,\uparrow} \hat{n}_{i,\downarrow}$$

Logical qubit 数（公式 8）：N = 2L² + 2

T count（公式 9, 10）：
$$\mathcal{T} = 6.203 \sqrt{\frac{W}{(\varepsilon(1-x))^3}} \times (N_R(1.15 \log_2(\cdots) + 11.2) + N_T)$$

变量：
- $N_T = 12L^2$：T gates per Trotter step
- $N_R = 4L^2$：arbitrary Z rotations per Trotter step
- $\varepsilon = 0.005 E_0$：相对误差 0.5%
- $W$：Trotter error bound
- $x$：error budget split

L=16, u=4：62k qubits at p=10⁻³，22k at p=10⁻⁴。对比 Ref [12] Kivlichan et al. 的 940k / 200k。Runtime 1.6–3.6 min（t_c=1µs）或 1.1–2.5 days（t_c=1ms）。

### 7.2 RSA-2048（Table VI，核心结果）

| Code Cycle | p | 1 year | 1 month | 1 week | 1 day |
|-----------|---|--------|---------|--------|-------|
| 1µs | 10⁻³ | 97k | 98k | 151k | 471k |
| 1µs | 10⁻⁴ | 53k | 53k | 65k | 149k |
| 10µs | 10⁻³ | 97k | 223k | 624k | 3.8M |
| 100µs | 10⁻³ | 198k | 1.3M | 5.4M | 41M |
| 1ms | 10⁻³ | 1.1M | 13M | 58M | - |
| 1ms | 10⁻⁴ | 307k | 3.1M | 14M | 128M |

关键数字：p=10⁻³, t_c=1µs, 1 month → 98k qubits。比 Gidney 2025 的 ~1M 少一个数量级。

Trapped ion（p=10⁻⁴, t_c=1ms）：53k qubits 最少，1M qubits 3 个月。对比 Beverland et al. (Ref [58]) 的 8.6M qubits 3 年。

Neutral atom（p=10⁻³, t_c=1ms）：19M qubits 21 天，对比 Zhou et al. (Ref [16]) 的 5.6 天（用 transversal gates + algorithmic fault tolerance 把 logical cycle time 压到 t_c）。

---

## 8. 架构图解析（Fig 1）

Fig 1a：1 个 processing unit + ~100k qubits，1 month factoring at p=10⁻³, t_c=1µs。
Fig 1b：81 个 processing units + ~1M qubits，3 months factoring at p=10⁻⁴, t_c=1ms。

Spacetime trade-off 通过调节 ρ（并行度）和 code distance 实现。更多 processing unit → 更快 runtime → 更多 qubits。

---

## 9. Intuition Building

### 9.1 为什么 QLDPC 比 surface code 省这么多

Surface code：1 logical qubit 需要 2d²-1 physical qubits。对 d=24（p=10⁻³），需要 ~1151 physical qubits per logical qubit。

GB code d=24：510 physical qubits / 16 logical qubits = 31.875 per logical qubit（pure code block）。加上 gadget 和 bridge：1620 / 16 ≈ 101 per logical qubit（processing block）。还是 surface code 的 ~1/11。

但这只是空间。关键在于 time overhead 没有爆炸：
- Surface code surgery：每个 logical measurement 1 个 logical cycle = d 个 code cycles
- Pinnacle surgery：每个 logical measurement 1 个 logical cycle = d+2 个 code cycles
- 两者基本持平

IBM Tour de gross 的 time overhead 高是因为只支持 subset of Pauli measurements，需要多轮组合。Pinnacle 用 generalized surgery（Ref [11]）一次性测量任意 Pauli，保持 d_t = d+2。

### 9.2 Magic Engine 的妙处

传统 surface code magic state distillation 需要专门 distillation factory，spacetime overhead 巨大。Litinski 2019 优化后仍然显著。

Pinnacle magic engine 的 trick：利用 QLDPC code 有 k 个 logical qubits 的事实，把 distillation 和 injection 放在同一个 code block 的两个 sector 上。L sector 蒸馏时 R sector 注入，下一 cycle 互换。这相当于把 distillation 的 latency 完全 overlap 进 computation 里。

Reject rate 处理：p_r = 6%（p=10⁻³ 时），意味着每个 T gate 平均花 1/(1-0.06) ≈ 1.06 logical cycles。这是公式 20 里 (1-p_r)⁻¹ 的来源。

### 9.3 Clifford Frame Cleaning 的几何直觉

Clifford frame 是 "尚未物理执行的 Clifford gates" 的累积。当两个 unit 间有 CNOT，这个 CNOT 进 frame，导致 frame 不再 tensor product 分解。这时两个 unit 上的 logical measurement 互相纠缠，必须串行。

Cleaning 就是花 4κ 个 π/4 rotation 把 frame 重新分解成 tensor product。π/4 rotation 用 Pauli measurement 实现，所以 4κ 个 logical cycles。

什么时候 clean？Compile 时优化决定。Circuit 一段高度并行 + 一段串行时，可以在两段交界处 clean，让并行段并行跑。

Memory port 只需 2w 的原因：fan-out CNOT 的 control 在 port 上，CNOT commute with Z on control，所以 frame 在 port 上只 act trivially 或 as CNOT control。这种 restricted frame form（公式 A8）clean 起来便宜一半。

### 9.4 RSA 并行化的 non-trivial 之处

朴素的并行化：复制 working register ρ 份，cost ×ρ。但 input register 是 bottleneck。

Gidney algorithm 的 input register 只通过 lookup 被访问，lookup gate 是 control 在 input、target 在 working。所有 control 在同一 register 上的 CNOT 互相 commute。所以 ρ 个 working register 可以 share 1 个 input register，只要 ρ ≤ ⌈m/w_1⌉（pipelining window access）。

对 m ≈ 4096（RSA-2048 的 2m logical qubits for spin up/down of L² sites...等等不对，RSA 里 m ≈ 2048+overhead），w_1 = k/2 = 7（d=16）或 8（d=24）。⌈m/w_1⌉ ≈ 256 或更大。所以 ρ ≲ 200 时只需 1 个 input register copy，N ≈ m + ρ N_w ≈ 2048 + ρ × ~20，远小于 ρ × 2048。

这就是 Fig 7 显示的 spacetime savings 来源：ρ 增大时 time 降 1/ρ，但 space 只增加 N_w ≈ 20 per unit。

---

## 10. Caveats & Open Questions

1. **Decoder**：用了 most-likely error decoding（MIP，optimal but slow）。Real-time fast decoder for QLDPC 是 open problem。Belief propagation 有 error floor 问题（Ref [51]）。这可能成为实际 runtime bottleneck。

2. **Code parameters 是 conjectured**：[[2(2^m-1), 2m, m+(m-4)²]] 没有 rigorous proof。Table I 的前几个 code 是 numerically verified，但 d=24 的 510 code 依赖外推。

3. **Cultivation success rate**：p=10⁻³ 时依赖 fold-transversal cultivation（Ref [46]），success rate 2/3，需要 5 次尝试，total success 94%。如果 cultivation 失败，magic engine reject。

4. **Connectivity requirement**：QLDPC 需要 non-local interaction within code block scale。Trapped ion（all-to-all）和 neutral atom（reconfigurable）OK，superconducting 需要 long-range coupler 或 shuttle。

5. **Fast surgery**：Paper 末尾提到 fast surgery（Ref [59]）可能再降一个数量级 runtime。如果 logical cycle time 能压到 code cycle time（像 Zhou et al. 的 transversal approach），runtime 大幅改善。

6. **与 IBM Tour de gross 的比较**：两者都用 bivariate bicycle codes（IBM）或 generalized bicycle codes（Pinnacle）。关键差异在 gadget 系统——Pinnacle 支持 arbitrary Pauli measurement，Tour de gross 只支持 subset。这导致 Pinnacle 的 time overhead 更低。

7. **Logical qubit 数 vs Physical qubit 数**：RSA-2048 用 d=24 GB code，每 processing block 16 logical qubits / 1620 physical qubits。κ ≈ 30+ per working register，所以 2 个 processing block per working register。ρ=200 个 working register → 400 processing block → 648k physical qubits（仅 processing）+ 200 magic engine（~1.7M）... 等等这与 98k 不符，说明 ρ 优化后远小于 200。需要看具体 ρ 选择。

实际 98k qubits 对应 ρ ≈ 几十个 working register。具体 ρ 在 paper 里是优化变量，没有明确给出最优 ρ 值。但 Table VI 的 trend 说明 ρ 不大（否则 qubit 数会高很多）。

---

## 11. 与你 (Karpathy) 可能相关的联想

这篇 paper 的思维方式和你 NN training 的 mindset 有几个 analogies：

1. **Spacetime trade-off like batch size tuning**：ρ 像 batch size，增大 ρ → runtime 降但 qubit 升。最优点取决于 hardware 参数（t_c, p）。这跟选 batch size 平衡 throughput 和 memory 类似。

2. **Modularity like pipeline parallel**：Magic engine 的 pipeline 模型跟 pipeline parallel training 完全同构——distillation 是 forward pass，injection 是 backward pass，两者 overlap 在不同 "stage"。

3. **Clifford frame cleaning like gradient checkpointing**：Cleaning 付出 4κ 代价换并行，类似 gradient checkpointing 付 recomputation 代价换 memory。都是 amortized cost 换 primary resource。

4. **LDPC code family scaling like model scaling**：Code family parameterized by m，k=2m, d≈m²。Scaling law 形式（公式 5）像 power law scaling，fit A, B, C 然后外推。

5. **Gadget system like residual connections**：Gadget 让 code block "augment" 出 arbitrary Pauli measurement 能力，类似 residual block 给 NN 加 expressivity without losing 特性。

---

## 12. 总结

Pinnacle Architecture 的核心贡献是三件事的 synthesis：
1. QLDPC codes（GB code family）的低空间开销
2. Generalized surgery gadgets 的任意 Pauli measurement 能力（避免 time overhead）
3. Magic engine + Clifford frame cleaning 的 pipeline + flexible parallelism

合起来把 RSA-2048 从 ~1M qubits 降到 ~100k qubits。作者明确说这是 "beginning of the story"——更高 rate 的 QLDPC codes（比如 bivariate bicycle 的 IBM 工作，或 Tanner codes, quantum Tanner codes）可能进一步降低。

对 cryptography 的影响：如果 hardware progress 按 Mohseni et al.（Ref [9]）的 scaling roadmap 走，100k qubit device 可能在 2030s 中后期出现，比 1M qubit device 早 5-10 年。RSA-2048 的 deprecation timeline 可能需要重新评估。

更多相关 reading：
- https://arxiv.org/abs/2308.07915 (IBM high-threshold QLDPC memory)
- https://arxiv.org/abs/2407.18393 (Improved QLDPC surgery)
- https://arxiv.org/abs/2410.02753 (Homological measurement)
- https://arxiv.org/abs/2409.17595 (Magic state cultivation, Gidney)
- https://arxiv.org/abs/2502.19406 (GB codes single-shot decoding)
- https://arxiv.org/abs/2411.10406 (Mohseni et al. quantum supercomputer scaling)
