---
source_pdf: The Deep Learning Compiler A Comprehensive Survey.pdf
paper_sha256: 9943e4946520978f19d8b772e57541dc8ed3c2745047894c15b91cd7763f5b33
processed_at: '2026-08-12T14:12:11-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 DL Compiler 综述

好, 我换一种讲法. 之前的版本太像在复述 paper, 这次我把它当跟一个聪明的朋友喝咖啡时聊这件事 — 你读完之后脑子里应该有个清晰的 mental model, 知道 DL compiler 在干嘛、为什么这么设计、哪些设计是历史包袱、哪些是真正聪明的.

---

## 一句话概括

DL compiler 干的事情就是把 PyTorch / TF 写出来的 model, 编译成能在 NVIDIA GPU、Intel CPU、Google TPU、Apple ANE、AWS Inferentia 这些乱七八糟硬件上跑得快的代码. 听起来像传统 compiler, 但中间多塞了两层专门为 DL 设计的 IR, 因为 LLVM IR 太底层, 丢失了 DL 的语义信息, 而 framework 自带的 graph IR 又太高层, 没法表达硬件细节.

---

## 为什么不能直接用 LLVM

你想想, 你写一个 ResNet, 在 PyTorch 里是 `nn.Conv2d` 一堆堆拼起来的. PyTorch 内部把它变成一个 computation graph, 每个节点是一个 op. 如果你想把这个 graph 编译成 GPU 代码, 最朴素的做法是给每个 op 写一个 CUDA kernel, 然后 framework 在 runtime 调用这些 kernel.

这就是 2017 年之前大家干的事情. cuDNN 就是 NVIDIA 给你写好的 kernel 库, 你调函数就行. 但很快大家发现几个问题:

**问题 1**: kernel 库跟不上 model 发展. 你今天发明了一个新的 activation function `SwiGLU`, cuDNN 里没有, 你要么自己写 CUDA kernel, 要么 fallback 到慢的实现. 库永远比 model 慢半年.

**问题 2**: 跨 framework 不通用. 你在 TF 里写好的 model, 想在 PyTorch 里跑, 或者反过来, 得重写. ONNX 解决了一部分, 但 ONNX 只是 graph 格式, 不解决 "怎么高效跑" 的问题.

**问题 3**: 跨硬件不通用. 你给 NVIDIA GPU 写的 kernel, 跑不了 TPU, 跑不了 Apple ANE, 跑不了 AWS Inferentia. 每个新硬件出来都要重新写一遍 kernel.

**问题 4**: 融合优化做不了. `conv + bn + relu` 这三个 op 如果分别调三个 kernel, 中间结果要写回 global memory 再读出来, 浪费带宽. 但如果你能把它融合成一个 kernel, 中间结果留在 register / shared memory 里, 快 10 倍. 库是黑盒, 你没法跨库融合.

所以大家就想: 要不搞个 compiler, 让 compiler 自动干这些事. 这就是 DL compiler 的起源.

---

## DL compiler 的骨架

所有 DL compiler 都长一个样, 大概是这样:

```
PyTorch / TF / JAX 写的 model
       │
       ▼
  ┌─────────────────┐
  │  Frontend       │  把 model 翻译成 graph IR
  │  (graph IR)     │  做硬件无关优化 (fusion, CSE, layout)
  └─────────────────┘
       │
       ▼  lowering
  ┌─────────────────┐
  │  Low-level IR   │  把 graph 展开成嵌套循环 + 内存访问
  │  (loop IR)      │  做硬件相关优化 (tile, vectorize, tensorize)
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐
  │  Codegen        │  lower 到 LLVM IR 或直接生成 PTX / 机器码
  │                 │  JIT 或 AOT
  └─────────────────┘
       │
       ▼
  GPU / CPU / TPU / NPU 的机器码
```

核心 insight 是中间那两层 IR:

- **High-level IR (graph IR)**: 保留 DL 语义. 节点是 op (conv, relu, matmul), 边是 tensor. 在这层你能做 fusion, 因为还能看出 `conv` 后面接 `relu` 是 element-wise, 可以合并.
- **Low-level IR (loop IR)**: 接近硬件. 已经变成嵌套 for 循环了, 你能控制 tile size, unroll factor, 哪个维做并行. 在这层你能做 tiling, vectorization, 把循环映射到 GPU thread block.

为什么分两层? 因为有些优化必须在高语义层做 (fusion 需要知道 op 边界), 有些必须在低层做 (tiling 需要看 loop nest). 你把这两层混在一起, 要么高层丢失语义, 要么低层看不清硬件.

---

## High-level IR: 怎么表示一个 DL model

这块 paper 讲了几个设计选择, 我挑有意思的说.

### DAG vs Let-binding

最自然的表示是 DAG (有向无环图): 节点是 op, 边是 tensor. ResNet 就是个 DAG, conv 接 bn 接 relu 接 pool, 一路向下.

DAG 的问题: 当你开始做 automatic differentiation 或者处理 control flow (RNN, if/else) 时, 变量的 scope 不清楚. 比如你这个变量是 "现在算" 还是 "之后需要时再算"? DAG 没法表达这个.

解决方案是从 functional programming 借来的 let-binding:

$$\texttt{let } x = e_1 \texttt{ in } e_2$$

意思是 "在 scope $e_2$ 里, $x$ 等于 $e_1$ 的值". 变量有明确 scope, 编译器维护一个 variable map, 任何地方引用 $x$ 都查 map.

TVM 的 Relay IR 同时用两种: 简单情况用 DAG, 复杂情况 (control flow, AD) 用 let-binding. 这是个工程妥协, 既保持简单又留了 escape hatch.

### 怎么写 tensor 计算

三种风格, 对应三个 compiler:

**Function-based (XLA, Glow)**: 给你一堆预定义 op, 你调用就行. 简单, 但加新 op 要改 compiler.

**Lambda expression (TVM)**: 你写 output element 怎么算, compiler 决定怎么 schedule. 比如:
$$O[i, j] = \sum_k A[i, k] \cdot B[k, j]$$
你只写右边那个公式, loop 怎么排, 怎么 tile, 怎么并行, 全由 compiler 决定. 这就是 Halide 哲学: **algorithm 和 schedule 分离**.

**Einstein notation (TC)**: 更简洁, 重复出现的 index 自动 sum:
```
C(i, j) += A(i, k) * B(k, j)
```
不用写 sum, $k$ 出现两次就自动 contraction. 灵感来自 numpy.einsum. 好处是 reduction 顺序自由, 并行化容易.

### Dynamic shape

2020 年大部分 compiler 只能处理 static shape, 你得提前知道 input 是 `[1, 3, 224, 224]`. 但 NLP 模型 input length 变, batch size 变, 这个限制很烦.

TVM 用 `Any` 表示未知维度: `Tensor[(Any, 3), float32]`. XLA 用 `None`: `tf.placeholder("float", [None, 3])`. 但真正支持 dynamic shape 还需要 runtime bound inference + 动态内存分配, 这块 TVM 后来搞了 Relax (2022), 专门解决这个问题.

### Data layout

一个 tensor 在内存里怎么摆? NCHW (batch, channel, height, width) 还是 NHWC (batch, height, width, channel)? 给定 logical index $(n, c, h, w)$, 物理 offset 是:

$$\text{offset}(n, c, h, w) = \sum_d \text{stride}_L[d] \cdot i_d$$

NCHW: stride = $(C \cdot H \cdot W, H \cdot W, W, 1)$, 沿 $w$ 连续.
NHWC: stride = $(H \cdot W \cdot C, 1, W \cdot C, C)$, 沿 $c$ 连续.

不同硬件偏好不同. NVIDIA tensor core 在 NHWC 下 channel 维可以直接喂到 matrix core 的 contracting dimension, 性能更好. 但 depthwise conv 在 NCHW 下更自然, 因为 channel 独立.

更高级的 layout 是 **NCHW[x]c**: 把 channel $C$ 拆成 $(C/x, x)$, $x$ 匹配 vector register 宽度. AVX-512 FP32 时 $x=16$, INT8 时 $x=32$. 这样向量化效率最高. TVM 的 auto-tuning 自动找最优 $x$.

---

## Low-level IR: 怎么把 graph 变成循环

graph IR 是 "做什么", low-level IR 是 "怎么做". 三大流派.

### Halide-based (TVM)

Halide 原本是给 image processing 设计的, 但 DL 的计算模式 (大量规整的 nested loop) 跟 image processing 很像, 所以 TVM 直接拿来用.

核心思想: **computation 和 schedule 分离**. 你写 algorithm, compiler 试各种 schedule, 选最快的.

Schedule 是一组指令, 比如:
- `tile(x, y, 16, 16)`: 把循环拆成 tile
- `vectorize(xi, 8)`: 内层向量化
- `parallel(yo)`: 外层并行
- `compute_at(blur_x, y)`: 把一个 stage 的计算嵌入到另一个 stage 的循环里 (sliding window)

同一个 algorithm, 不同 schedule, 性能差 10-100x. 这就是为什么 auto-tuning 这么重要.

Halide 的限制: 循环边界必须是 axis-aligned box, 处理不了复杂 shape (L 形, 三角形). 对 DL 够用, 但 sparse tensor 就不行.

### Polyhedral (TC, PlaidML)

更数学化的方法. 每个 statement 的 iteration space 是一个整数多面体:

$$\mathcal{D} = \{ \mathbf{i} \in \mathbb{Z}^d \mid A\mathbf{i} + \mathbf{b} \geq \mathbf{0} \}$$

其中 $\mathbf{i}$ 是 iteration vector (loop indices), $A$ 是约束矩阵, $\mathbf{b}$ 是偏移.

然后做 affine transformation $T: \mathbf{i} \mapsto T\mathbf{i} + \mathbf{t}$ 重排循环, 实现 fusion, tiling, parallelization. 理论上很强, 可以处理任意 shape, 但:
- 难以参数化 schedule (search space 是离散 affine 变换集合)
- 工具链依赖 isl, Omega 这些库
- sparse tensor 的 `a[b[i]]` 这种非线性下标破坏 affine 假设

TC (Tensor Comprehension) 把 Halide computation 和 polyhedral schedule 混用, 各取所长. PlaidML 的 Stripe IR 用 nested polyhedral block, 把 polyhedron nesting 对应到 memory hierarchy (register → L1 → L2 → HBM).

### MLIR (现代主流)

MLIR 不是单一 IR, 是个 framework. 你可以定义自己的 dialect (一组 op), 不同 dialect 共存, 通过 conversion pattern 互相 lower.

现在 (2024-2026) 的 DL compiler 生态基本都迁移到 MLIR 上了:
- **StableHLO** (OpenXLA): 标准化的 graph IR, XLA, PyTorch, JAX 都往这靠
- **Torch-MLIR**: PyTorch → MLIR (Linalg dialect)
- **IREE**: MLIR → Vulkan / CUDA / CPU, 端到端 compiler
- **TVM Relax**: 基于 MLIR, 专门处理 dynamic shape
- **Modular Mojo**: MLIR 当 language backend, Python-like syntax

MLIR 的 dialect 系统让 "统一优化" 这件事第一次变得可能 — 你可以在 linalg dialect 做通用 fusion, 然后 lower 到 affine dialect 做 polyhedral 优化, 再 lower 到 LLVM dialect 做代码生成.

---

## Frontend optimizations: 在 graph 上做什么

graph IR 拿到后, 做硬件无关优化. 分三层.

### Node-level

- **Nop elimination**: 只有一个 input 的 sum 节点删掉, padding=0 的 pad 节点删掉
- **Zero-dim-tensor elimination**: 输入是 zero-dim 的 op 删掉. 比如 $A$ 是 zero-dim, $B$ 是 const, $A + B$ 直接换成 $B$

### Block-level

**Algebraic simplification**: 常见的 $x + 0 = x$, $x \cdot 1 = x$, constant folding ($2 + 3 \to 5$). DL 特有的: 两个 transpose 然后 GEMM, $A^T B^T = (BA)^T$, 省两次 transpose 改一次.

**Operator fusion**: 核心优化. TVM 把 op 分 4 类:
- **Injective** (1-to-1): ReLU, add, sigmoid, 总可融合
- **Reduction** (n-to-1): sum, max, mean, 可融合到下游 injective
- **Complex-out-fusible**: conv + bias + relu 这种, 数据流局部可融合
- **Opaque**: sort, topk, 不可融合

融合的好处:
- 消除 intermediate allocation (中间 tensor 不用写回 memory)
- 减少 kernel launch overhead (GPU 上一个 kernel launch 5-10 μs, 融合三个 op 省两次 launch)
- 合并 loop nest, 提升 locality

**Operator sinking**: 把 transpose 推到 ReLU / batch norm 后面, 让相似 op 靠近, 创造融合机会.

### Dataflow-level

**CSE** (Common Subexpression Elimination): 同一表达式只算一次. DL 里很常见, batch norm 的 mean / var 在 forward / backward 都用.

**DCE** (Dead Code Elimination): 输出没被用到的 op 删掉.

**Static memory planning**: 分析 tensor lifetime, 重用 buffer. 两种方式:
- **In-place sharing**: op 的 input/output 共享内存. 比如 ReLU 可以 in-place, $y = \max(x, 0)$, 直接覆盖 $x$
- **Standard memory sharing**: 不同 op 的 buffer 在时间上不重叠时共用

数学上, 设 buffer $B_i$ 的 lifetime 是 $[t_i^s, t_i^e)$. In-place 约束: $B_i$ 和 $B_j$ 是 producer-consumer. Standard sharing 约束:
$$[t_i^s, t_i^e) \cap [t_j^s, t_j^e) = \emptyset$$

然后做 graph coloring, 最小化 peak memory. 在 edge device (memory 受限) 上特别重要.

**Layout transformation**: 找最优 layout, 在 graph 中插 transform node. 不立即做, 留到 backend 合并.

---

## Backend optimizations: 在 loop IR 上做什么

### 硬件特定优化

**Hardware intrinsic mapping**: 识别 IR pattern, 替换成硬件指令. 比如 NVIDIA tensor core 的 mma 指令, 你在 IR 里看到 $16 \times 16$ FP16 GEMM tile, 就替换成 `mma.sync` PTX 指令. TVM 的 tensorization 就是干这个, 用户声明 intrinsic behavior + lowering rule, compiler 自动匹配.

**Memory allocation / fetching**: GPU 上 shared memory (低延迟, 小) vs global memory (高延迟, 大). TVM 用 memory scope tag 标记 stage, 自动生成 cooperative fetching + memory barrier.

公式: thread block $B$ 的 $T$ 个 thread 协作加载 tile:
$$X_{\text{shared}}[\text{tid} \cdot k : (\text{tid}+1) \cdot k] = X_{\text{global}}[\text{offset}_B + \text{tid} \cdot k : \text{offset}_B + (\text{tid}+1) \cdot k]$$

**Memory latency hiding**: GPU 靠 warp context switching 硬件隐藏. TPU 这种 DAE (decoupled access-execute) 架构需要 compiler 调度. TVM virtual threading 让你写虚拟并行线程, compiler 把它们 interleave 成一个 instruction stream, 插 memory barrier.

**Loop optimizations**:
- **Tiling**: $i \in [0, N)$ 拆成 $ii \in [0, N)$ 和 $i' \in [0, T)$, $i = ii \cdot T + i'$. $T$ 匹配 cache line
- **Loop fusion**: 同 boundary 的 loop 合并, 共享 iteration
- **Sliding window**: on-the-fly 计算 + reuse, 直到不再需要
- **Loop reordering**: 重排 nesting, 优化 spatial locality
- **Loop unrolling**: 展开 $n$ 次, 增加 ILP

**Parallelization**: SIMD + multi-thread. Halide 的 `parallel(dim)` 把维度并行化, GPU 上用 `block` / `thread` annotation 映射.

### Auto-tuning

为什么需要? 因为优化参数空间巨大. tile size 是 8, 16, 32, 64? unroll factor 是 1, 2, 4, 8? 这些组合是指数级, 人手工调一周也调不完.

四个组件:

**Parameterization**: 把 schedule 参数化. 包括 data shape, hardware 特性 (shared memory size, register file), 优化选项 (tile size, unroll factor).

**Cost model**: 预测每个 config 的 latency.
1. **Black-box** (TC): 只看 final latency, 简单但搜索效率低
2. **ML-based** (TVM, XLA): GBDT 或 FNN 预测. TVM 用 GBDT:
   $$\hat{y}(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x})$$
   其中 $f_k$ 是第 $k$ 棵回归树, $\mathbf{x}$ 是 config 特征 (tile sizes, loop order 等)
3. **Pre-defined**: 解析模型, 快但新硬件需重写

**Searching technique**:
- **Genetic Algorithm** (TC): crossover + mutation + selection
- **Simulated Annealing** (TVM): 接受 worse solution 的概率 $P(\Delta E) = e^{-\Delta E / T}$, $T$ 降温, 避免局部最优
- **Reinforcement Learning** (Chameleon): 把 schedule 选择建模成 MDP

**Acceleration**:
- Parallelization: 多 CPU thread 编译 + 多 GPU 评估
- Config reuse: 编译 cache / log file

TVM AutoTVM 后来被 Ansor (2020) 取代: Ansor 用 program synthesis 自动生成 sketch + annotation, 不需要人工写 schedule template. 再之后 MetaSchedule (TVM 0.10+) 用 unified tuning API.

### Kernel libraries

cuDNN (NVIDIA), DNNL (Intel), MIOpen (AMD), 以及 ASIC 自家的库. DL compiler 可以 fallback 到这些库. 但代价:
- 必须先做 layout / fusion style 转换, 可能破坏最优 control flow
- 库是黑盒, 不能跨 op 融合
- 库永远比 model 慢半年

TVM / nGraph / TC 支持 fallback, 但默认走自己 codegen. Glow / XLA 深度依赖 cuDNN / DNNL.

---

## 五大 compiler 对比

| | TVM | nGraph | TC | Glow | XLA |
|---|---|---|---|---|---|
| 谁搞的 | Apache | Intel | Meta | Meta | Google |
| 怎么写 | Lambda expr | Tensor expr | Einstein | Layer | TF interface |
| ONNX | built-in | pkg | ✗ | built-in | pkg |
| 训练 | developing | only NNP-T | single kernel AD | limited | TF support |
| 量化 | int8/fp16 | int8 | ✗ | int8 | int8/int16 |
| High-IR | Relay | nGraph IR | TC IR | own | HLO |
| Low-IR | Halide | None | Polyhedral | own | HLO |
| Dynamic shape | Any | PartialShape | ✗ | ✗ | None |
| Auto-tuning | AutoTVM | ✗ | GA | ✗ | FNN |
| Kernel libs | MKL/cuDNN | DNNL/cuDNN | ✗ | ✗ | cuDNN/TRT |
| 设备 | CPU/GPU/FPGA/VTA | CPU/Intel GPU | NVIDIA GPU | CPU/GPU | CPU/GPU/TPU |

直觉读法:
- **TVM** 生态最广, 最 academic, 偏 "compiler first"
- **XLA** 嵌在 TF 里, 偏 "framework first", TPU 专属
- **nGraph** 是 Intel 阵营, 重库
- **TC** 是 kernel DSL, 不算 full compiler, 没成气候
- **Glow** 早死了, Meta 2020 后停止投入

---

## Evaluation 关键发现

paper 在 V100, 2080Ti, Broadwell, Skylake 上跑 19 个 ONNX 模型:

1. **TVM 调优 vs 不调优**: CPU 上差不多, GPU 上 41.26× 平均加速. GPU schedule space 大, 不调就是垃圾
2. **Glow CPU 拉胯**: 不支持 thread parallelism, 多核 CPU 全废
3. **XLA end-to-end overhead 来自 TF runtime**: 单 batch 测延迟被 TF runtime 吃掉, batch 大了 throughput 才正常
4. **nGraph 在 CPU 上很好**: DNNL 的 JIT 在 Intel CPU 上做了非常细的 instruction-level 优化, nGraph 只是 fallback
5. **TC 性能不稳定**: autotuning cache 依赖初始 kernel, 后续层形状不同就要重调

per-layer 对比显示:
- conv 层 TVM 在 GPU 上最稳, nGraph 在 CPU 上最好
- XLA 把 conv 和其他 op cluster 在一起, 单独测不到 conv 时间
- Glow 在 1×1 conv 和 depthwise conv 上明显差

---

## 这篇 paper 的 future directions, 2024-2026 现状

paper 列了 8 个方向, 对照现在:

**Dynamic shape**: TVM Relax (2022) 解决, PyTorch 2.0 TorchDynamo + TorchInductor 解决 Python-side 动态, JAX + XLA 走 pmap/pjit 路线.

**Advanced auto-tuning**: Ansor / MetaSchedule 全自动, 不依赖 template. Triton autotune, OpenAI 的 kernel autotune 都是演进.

**Polyhedral model**: 仍小众, 主要在 PlaidML, AKG (Ascend). sparse tensor + polyhedral 还没成熟. LLM 的 sparse attention, MoE 反而让 polyhedral 复活了一点.

**Subgraph partitioning**: PyTorch fx partition, ONNX Runtime ORT optimize, Apple CoreML subgraph split 到 ANE. heterogeneous execution 阿里 MNN, 小米 MACE 都做.

**Quantization**: LLM 的 INT4 / FP8 / NVFP4, 用硬件 native dtype. torchao, TensorRT-LLM 把 quant 推到 tensor core 的 INT8 / FP8 native 路径.

**Unified optimizations (MLIR)**: StableHLO 成 standard graph IR. MLIR 生态: Torch-MLIR, IREE, Modular Mojo 把 MLIR 当 language backend.

**Differentiable programming**: Julia Zygote.jl 成熟, Enzyme 做 source-to-source AD on LLVM. JAX jit(grad()) 是 tracing-based AD + XLA lowering 典范. production 还以 PyTorch autograd (runtime) 为主, compile-time AD 还没大规模替代.

**Privacy protection**: split inference + noise injection 在联邦学习里有, production DL compiler 还没把这块做进 pass. 同态加密 + compiler 是早期实验.

**Training support**: XLA 训练 OK (TPU + JAX). TVM Relay training 仍 partial. PyTorch Inductor 训练 + 推理都覆盖, 是目前最全的.

---

## 给 Karpathy 的 intuition

你那个 micrograd 用 100 行 Python 实现了 reverse-mode AD + tiny autograd, 是 DL framework 的最简内核. 这篇 paper 讲的是 framework 之上的下一层 — framework (PyTorch autograd / TF) 解决 "怎么算 gradient", DL compiler 解决 "怎么把 forward+backward graph 高效部署到硬件".

完整 mental model:

```
User code (micrograd / nanoGPT / PyTorch eager)
       │
   Tracing / symbolic evaluation
       ▼
Computation graph (PyTorch fx, TF Graph, JAX jaxpr)    ← framework IR
       │
   frontend lower
       ▼
Graph IR (Relay, StableHLO, HLO)                       ← DL compiler high-IR
       │
   frontend opt (fusion, layout, CSE, DCE, mem plan)
       ▼
Lowering
       │
       ▼
Tensor / Loop IR (TensorIR, Stripe, Halide IR)        ← DL compiler low-IR
       │
   backend opt (tile, vectorize, tensorize, par)
   + auto-tuning (AutoTVM / Ansor / MetaSchedule / Triton autotune)
       ▼
LLVM IR / Triton IR / MLIR LLVM dialect
       │
       ▼
Hardware (PTX, SASS, x86-64, ARM64, NPU firmware)
```

每一层 abstraction 都用对应的 IR, 越往下越 hardware-specific, 越往上越 semantic-rich.

为什么 PyTorch 2.0 用 TorchInductor 而不是直接 LLVM? 因为 DL compiler 在中间多塞了 graph IR + tensor IR 两层, 这两层让 fusion + autotune + tensorization 有了 first-class 表达. 直接 LLVM 那一层做这些会非常痛苦, 因为 LLVM IR 太底层, op 边界消失, 你看不出 `conv` 后面接 `relu` 是 element-wise, 融合就无从下手.

Triton 让程序员手写中间层 (像 CUDA 但带 autotune, 而且有 Python-like syntax). Mojo 把 MLIR 直接当 language backend. 这些都是同一个 abstraction hierarchy 的不同切片.

paper 的最大贡献在 2019 那个时间点把整个 landscape 用清晰 framework 描述出来, 后续演化路线都能从这 8 个 future direction 里找到影子. 今天看仍然是一本好 "DL compiler 入门地图", 虽然 Glow 死了, nGraph 在 Intel 重心下降, TC 没成气候, 但 abstraction skeleton 没变.

参考链接:
- Paper: https://arxiv.org/abs/2002.10979
- TVM: https://tvm.apache.org/
- MLIR: https://mlir.llvm.org/
- StableHLO: https://github.com/openxla/stablehlo
- TorchInductor: https://dev-discuss.pytorch.org/t/what-is-torchinductor/951
- Triton: https://github.com/openai/triton
- TVM Relax: https://arxiv.org/abs/2305.16091
- Ansor: https://arxiv.org/abs/2006.06762
- IREE: https://github.com/openxla/iree
- Modular Mojo: https://www.modular.com/mojo
- Original Halide paper: https://people.csail.mit.edu/jrk/halide-pldi13/halide_pldi13.pdf

---

# The Deep Learning Compiler: A Comprehensive Survey — 深度解读

Andrej, 这篇由北航 + 清华团队 2019/2020 年发布的综述, 是把 DL compiler 这个领域从 "工业级拼装" 提炼成 "可教学的 mental model" 的一次相当扎实的尝试. 我接下来按 paper 的骨架展开, 同时把每个组件背后的 intuition、对应公式、以及与现在 (2024-2026) 的演化 (TorchInductor, Relax, StableHLO, Triton, MLIR, Mojo) 串起来.

参考链接:
- Paper (arXiv): https://arxiv.org/abs/2002.10979
- TVM: https://tvm.apache.org/
- MLIR: https://mlir.llvm.org/
- StableHLO: https://github.com/openxla/stablehlo
- TorchInductor (PyTorch 2.0): https://dev-discuss.pytorch.org/t/what-is-torchinductor/951
- Triton: https://github.com/openai/triton

---

## 1. 为什么要单独抽出一个 "DL compiler"

把背景先说清楚, 后面的设计 trade-off 才有意义.

DL 的部署压力来自两端:
- **Frontend diversity**: TF, PyTorch, MXNet, CNTK, Paddle, JAX, Flax… 每家 graph IR 都不同, 一个模型要跑在多平台上就得反复重写 kernel.
- **Backend diversity**: NVIDIA tensor core, Intel AMX, Google TPU MXU (systolic array), Apple ANE, AWS Inferentia, 阿里 Hanguang, Cambricon MLU, Graphcore IPU, AMD MI 系列矩阵核, 甚至 mobile NPU.

这两个轴 orthogonal 之后, 设计空间是 $N_{\text{frameworks}} \times N_{\text{hw}}$ 个 kernel 实现路径. DL compiler 的目标是把这个问题降到 $N_{\text{frameworks}} + N_{\text{hw}}$ 的复杂度 — 把 framework-specific graph 翻译成 hardware-agnostic intermediate representation, 再 lower 到 hardware-specific code.

直觉上这就是 LLVM 的 multi-level IR 哲学在 DL 领域的延伸. LLVM 通过 Clang IR → LLVM IR → SelectionDAG → MachineIR → MC layer 把 C++ 编译到任意 backend, DL compiler 通过 Graph IR → Tensor IR → Loop IR → LLVM IR 把 ResNet 编译到任意加速器. 主要的 "新东西" 集中在两层 IR: 高层的 graph abstraction 和中间的 schedule IR.

---

## 2. Common design architecture (Figure 2 解析)

paper Figure 2 把通用架构画成这种 pipeline:

```
DL model (TF/PyTorch/MXNet/ONNX)
        │
   [Frontend]
        │  transform + format conversion
        ▼
  ┌─────────────────────────────┐
  │  High-level IR (Graph IR)     │   hardware-independent
  │  - Relay / HLO / Glow graph  │   semantics: op, dataflow, shapes
  └─────────────────────────────┘
        │
   frontend optimizations
   - node-level: nop elim, zero-dim elim
   - block-level: algebraic simpl, op fusion, op sinking
   - dataflow-level: CSE, DCE, static memory planning, layout transform
        │
        │  lowering
        ▼
  ┌─────────────────────────────┐
  │  Low-level IR (Loop / Stmt IR)│   hardware-specific
  │  - Halide IR / Polyhedral /  │   exposes loop nest, memory scope
  │    Stripe / HLO-low           │
  └─────────────────────────────┘
        │
   backend optimizations
   - hardware intrinsic mapping
   - memory alloc / fetching / latency hiding
   - loop fusion, tiling, sliding window, reorder, unroll
   - parallelization (SIMD, GPU thread/block)
   - auto-tuning (AutoTVM, TC-GA, XLA-FNN)
   - kernel library fallback (cuDNN, DNNL, MIOpen)
        │
        ▼  JIT / AOT
  LLVM IR → codegen → CUDA / NVPTX / x86-64 / ARM / HLS / TPU instruction
```

关键 insight: **graph IR 保留 DL 语义边界, low-level IR 保留硬件物理边界**. 如果只有 graph IR, hardware-specific 优化空间打不开; 如果只有 low-level IR, global fusion 和 layout 一致性难做. 多级 IR 既是 abstraction ladder, 也是 optimization 不同阶段的最适表达载体.

---

## 3. High-level IR (graph IR)

### 3.1 两种 representation

**DAG-based IR**
最经典形式: 节点 = atomic op (conv, pool, add), 边 = tensor. 与传统 compiler 的 DDG (data dependence graph) 类似, 但去掉了 cycle (无 control flow back edge, 在 basic DL 模型里足够). Glow, nGraph, XLA 早期, TC 都是这类.

DAG 的缺点: 当同一个 sub-expression 在不同 control flow path 上被引用时, 它的 evaluation scope 不明确 (recursive descent 搜图时, "现在算 / 之后再算" 不清楚). 这就引出 **let-binding**.

**Let-binding-based IR**
来自 Lisp / Scheme / F# / JS. 形式:

$$\texttt{let } x = e_1 \texttt{ in } e_2$$

变量 $x$ 在表达式 $e_1$ 处被绑定, scope 严格限制在 $e_2$. 编译器维护一个 variable map, 任何引用 $x$ 的位置都从 map 查到其 evaluation, 而非依赖图的拓扑.

Relay IR (TVM) 同时采用两种, 兼顾 DAG 的简洁和 let-binding 的 scope 清晰. 这样做的好处:
- AD (automatic differentiation) 实现 source-to-source 变换时, scope 决定 gradient 传播路径
- Mutation / control flow 安全 (任意 control flow 都可以拆成 recursion + pattern matching, 这一点在 Relay 的设计文档里有专门讨论)

### 3.2 Tensor computation 的三种写法

**Function-based**: 把每个 op 当一个 primitive function, 用户调用即可. XLA HLO 是典型: HloModule → HloComputation → HloInstruction 三层. 简单, 但 extensibility 弱 (新 op 需要写 lowering rule).

**Lambda expression** (TVM Tensor Expression): 形式
$$O[i, j] = \lambda (i, j) . \sum_{k} A[i, k] \cdot B[k, j]$$
其中 $i, j, k$ 是 index variable, $A, B$ 是 input tensor. 用户只写 "怎么算这个 output element", schedule 由编译器决定. 这是 Halide 思想在高层的映射 — **algorithm/schedule 分离**从最底层就渗到顶层.

**Einstein notation** (Tensor Comprehension): 形式
```
def matmul(float(N, K) A, float(K, M) B) -> (C) {
    C(i, j) += A(i, k) * B(k, j)
}
```
规则: 出现一次的 index 是 free index (出现在 output), 出现两次的 index 是 contraction index (隐式 sum). 要求算子可结合可交换, 这样 reduction 可以任意顺序执行, 自动 parallelization / reassociation 变得简单. 灵感来自 numpy.einsum.

### 3.3 Data representation

**Placeholder**: 在 symbolic programming 里 (Lisp, TF), 一个有 shape 但没有具体 value 的变量. 编译时只是个 symbol, 运行时填充. 让用户先定义计算图再喂数据.

**Unknown / Dynamic shape**:
- TVM: `Tensor[(Any, 3), float32]` — `Any` 表示该维度运行时决定
- XLA: `tf.placeholder("float", [None, 3])`
- nGraph: `PartialShape` 类

paper 指出当时 (2020) 的痛点: 大部分 compiler 只能处理 static shape, 真正 dynamic 需要的是运行时 bound inference + memory validity 检查. 这正是 TVM 后来发展出 Relax (2022-) 的原因, 把 dynamic shape 当一等公民: https://tvm.apache.org/docs/arch/relax.html

**Data layout**: 把 logical index 映射到 memory index. 涉及 dimension order (NCHW vs NHWC), tiling, padding, striding.

公式: 给定 logical index $(n, c, h, w)$ 和 layout $L$, 物理 offset:
$$\text{offset}(n, c, h, w) = \sum_{d \in \{n, c, h, w\}} \text{stride}_L[d] \cdot i_d$$
其中 $\text{stride}_L[d]$ 是 layout $L$ 下 dimension $d$ 的 stride. 比如 NCHW: $\text{stride} = (C \cdot H \cdot W, H \cdot W, W, 1)$; NHWC: $\text{stride} = (H \cdot W \cdot C, 1, W \cdot C, C)$.

不同硬件最优 layout 不同: NVIDIA cuDNN 在 NHWC 上 conv 通常更快 (V100 之后), 而 NCHW 在某些 RNN op 上更自然. TPU / 自研 ASIC 经常用 NCHW[x]c 这种 blocked layout 直接喂给 systolic array.

**Bound inference**: 给定 placeholder shape, 推断所有 iterator 的范围. TVM 把 iterator 抽象成 DAG of iterators, 节点是 iterator, hyper-edge 是 relation (split / fuse / rebase). Root iterator 由 placeholder shape 决定, 其他递归推断.

例子 (split): 设原 iterator $i \in [0, N)$, split factor $f$, 得到:
$$i_{\text{outer}} \in \left[0, \left\lceil \frac{N}{f} \right\rceil\right), \quad i_{\text{inner}} \in [0, f)$$
关系: $i = i_{\text{outer}} \cdot f + i_{\text{inner}}$.

### 3.4 Operators

**Broadcast**: 把 shape 不一致的 tensor 复制到兼容 shape 后做 element-wise op. XLA / Relay 直接支持, 否则用户得手动 tile.

**Control flow**: RNN, RL, 条件分支必须有 control flow. 两条路:
- Relay: $\texttt{if}$ + recursive function (任意 control flow = recursion + pattern match, 借鉴 functional programming)
- XLA: `while` / `conditional` HLO op (把 control flow 编码成 op)

**Derivative**: AD 需要 high-level IR 暴露每个 op 的 gradient. 比如 conv2d 的 gradient 需要 conv2d_backward_input / conv2d_backward_weight. TVM 逐步推进到所有 algebraic + NN op 都有 gradient op; PlaidML 直接自动生成 derivative op.

**Customized operator**: 用户写新 op 的工程量决定 extensibility. Glow 要求用户写 logic + node encapsulation + lowering + IR generation + instruction generation; TVM 只要求用户写 computation + schedule, 大幅降低门槛. 这点直接影响 TVM 在工业界的成功 — 自研 ASIC 团队 (比如 Tesla, Amazon) 可以方便地加新 op 并 mapping 到自家 intrinsic.

---

## 4. Low-level IR

low-level IR 是把 "图" 变成 "嵌套循环 + 内存访问" 的关键 step. 三大流派:

### 4.1 Halide-based IR

Halide (Ragan-Kelley et al. PLDI 2013) 最早是为 image processing pipeline 设计. 哲学: **separation of computation and schedule**.

Computation 用纯函数表达:
$$\texttt{blur}_x(x, y) = (I(x-1, y) + I(x, y) + I(x+1, y)) / 3$$
$$\texttt{blur}_y(x, y) = (\texttt{blur}_x(x, y-1) + \texttt{blur}_x(x, y) + \texttt{blur}_x(x, y+1)) / 3$$

Schedule 是 orthogonal 的指令:
- `compute_at(blur_x, y)`: blur_x 在 y 循环内计算, 减少 re-fetch
- `vectorize(x, 8)`: 把 x 维向量化成 8-wide SIMD
- `parallel(y)`: y 维并行
- `tile(x, y, xo, xi, yo, yi, 16, 16)`: tiling

公式上, schedule $S$ 把 computation $F$ 重写成 loop nest $L(S, F)$. 同一个 $F$, 不同 $S$ 产生不同的 $L$, 性能可以差 10x-100x.

TVM 改造 Halide IR 成独立 symbolic IR (TVM 0.6 起, 后来演化成 Relax 的 TensorIR):
- 去掉 LLVM 强依赖, 重构 module 结构
- 用 SSA (static single-assignment) 保证每个 variable 只在一个位置 define
- 加 runtime dispatch 机制, custom op 注册方便

Halide 的限制: memory reference 和 loop nest 的边界必须是 axis-aligned box, 表达不了复杂 pattern (三角、L 形). 对 DL 这种规整 workload 够用, 但越界 / sparse / non-rectangular stencil 就棘手.

### 4.2 Polyhedral-based IR

polyhedral model 把每个 statement 的 iteration space 表达成整数多面体:
$$\mathcal{D} = \{ \mathbf{i} \in \mathbb{Z}^d \mid A \mathbf{i} + \mathbf{b} \geq \mathbf{0} \}$$

其中 $\mathbf{i} = (i_1, i_2, \ldots, i_d)$ 是 iteration vector ($d$ = loop nest depth), $A \in \mathbb{Z}^{m \times d}$ 是约束矩阵, $\mathbf{b} \in \mathbb{Z}^m$ 是偏移, $m$ 是约束数.

affine transformation $T$ 把 $\mathcal{D}$ 映射到新多面体 $\mathcal{D}'$:
$$T: \mathbf{i} \mapsto T \mathbf{i} + \mathbf{t}$$
然后 codegen 用 scanning loop 重写. 经典变换: fusion, tiling, skewing, parallelization, interchange, strip-mining.

理论上, polyhedral 可以做 global loop optimization, 处理任意 shape (L 形, 三角形). 但:
- 难以参数化 schedule (search space 是离散 affine transformation 集合, 难用 ML cost model 直接预测)
- 工具链依赖 isl / Omega / PIP / Polylib / PPL
- sparse tensor 的非线性下标 `a[b[i]]` 破坏 affine 假设

**TC** (Vasilache et al. 2018) 把 Halide computation 和 polyhedral schedule 混用: computation 用 Halide IR, loop structure 用 polyhedral. 用 domain node 定义 index range, band node 决定 iteration order, filter node 表示 statement instance, extension node 描述 memory movement.

**PlaidML** 的 Stripe IR 用 nested polyhedral block, 把 polyhedron nesting 对应到 memory hierarchy (register → L1 → L2 → HBM), tag 机制把 hardware info 注入而不改 kernel 结构. 这跟 TVM 后来的 TensorIR + meta-schedule 思路很像.

### 4.3 其他独特 IR

**Glow**: instruction-based. 显式声明 buffer lifetime, 用 `@in`, `@out`, `@inout` 标记 operand 访问模式, 帮助 memory pass 做 alias 分析.

**MLIR**: 不是单一 IR, 是一个 framework. 提供 dialect 系统: 每个 dialect 是一组 op, 可以共存 (TensorFlow dialect, XLA HLO dialect, linalg dialect, affine dialect, LLVM dialect…), 之间通过 conversion pattern 互相 lower. 这给整个 DL compiler 生态带来 unification 的可能 — TVM 的 Relax, Google IREE, StableHLO, Torch-MLIR 都基于 MLIR. Reference: https://mlir.llvm.org/docs/Tutorials/understanding-the-ir-structure/

**XLA HLO**: 同一个 IR 跨 high/low 两层 (HLO 既描述 graph 级 fusion, 也描述 hardware-specific lowering). 紧凑, 但 abstraction 层次不够干净.

### 4.4 Code generation

几乎所有 DL compiler 最后都 lower 到 LLVM IR, 借 LLVM 的成熟 backend (NVPTX, AMDGPU, x86, ARM, RISC-V). 例外是 TPU / Inferentia / Cambricon 这种自带 instruction set 的 ASIC — compiler 直接生成它们的 binary.

JIT vs AOT:
- JIT (TC, nGraph, TVM 默认): 运行时编译, 能用 runtime profile 优化, 但启动慢
- AOT (XLA, Glow, TVM experimental): 离线编译成 binary, 启动快, 可以 cross-compile, 适合 edge / 嵌入式

---

## 5. Frontend optimizations

frontend 在 graph IR 上做 hardware-independent 的 pass. 分三层:

### 5.1 Node-level

**Nop elimination**: 删除没实际作用的 op. 例如只有一个 input 的 sum 节点, 或 padding=0 的 pad 节点.

**Zero-dim-tensor elimination**: 删除输入是 zero-dim tensor 的 op. 例:
- $A$ 是 zero-dim, $B$ 是 const: $A + B$ 替换成 $B$
- $C$ 的 shape 是 $\{0, 2, 3\}$: $C$ 无元素, $\arg\min / \arg\max(C)$ 可删

### 5.2 Block-level

**Algebraic simplification**: 三类
1. Algebraic identity (e.g., $x + 0 = x$, $x \cdot 1 = x$)
2. Strength reduction (e.g., $x \cdot 2^k$ → $x \ll k$, $x^2$ → $x \cdot x$)
3. Constant folding ($2 + 3 \to 5$)

DL 特有 case: 两个 transpose 然后做 GEMM $A^T B^T = (BA)^T$, 省两次 transpose 改成一次; 连续 transpose 合并成一次; reduce-mean 替换成 avg-pool (Glow 优化).

**Operator fusion**: DL compiler 的核心优化. TVM 把 op 分 4 类:
- **Injective** (1-to-1, 如 element-wise add, ReLU, sigmoid): 总可融合
- **Reduction** (n-to-1, 如 sum, max, mean): 可融合到下游 injective
- **Complex-out-fusible** (如 conv + bias + ReLU 这种数据流局部): 复杂但可融合
- **Opaque** (sort, topk, shuffle): 不可融合

融合规则 (递归表):
| Producer \ Consumer | Injective | Reduction | Complex-out | Opaque |
|---|---|---|---|---|
| Injective | ✓ | ✓ | ✓ | ✗ |
| Reduction | ✗ | ✗ | ✗ | ✗ |
| Complex-out | ✓ (pointwise 后继) | ✗ | ✗ | ✗ |
| Opaque | ✗ | ✗ | ✗ | ✗ |

收益:
- 消除 intermediate allocation
- 减少 kernel launch / synchronization overhead (GPU 上一个 kernel launch 5-10 μs)
- 合并 loop nest, 提升 locality

后续工作 (FusionStitching, https://arxiv.org/abs/1911.11576) 把 fusion 扩展到 broadcast / reduction 混合 block.

**Operator sinking**: 把 transpose 推到 batch norm / ReLU / sigmoid / channel shuffle 后面, 让相似 op 靠近, 创造 algebraic simpl 机会.

### 5.3 Dataflow-level

**CSE** (Common Subexpression Elimination): 同一表达式只算一次. DL 里很常见, 比如 batch norm 的 mean / var 在 forward / backward 都用.

**DCE** (Dead Code Elimination): 输出 / 副作用没被用到的 op. DL 里常由其他 pass 产生.

**Static memory planning**: 离线分析 lifetime, 重用 buffer. 两种方式:
- **In-place sharing**: op 的 input/output 共享内存, 只在计算前 alloc 一次. 例: ReLU 可以 in-place
- **Standard memory sharing**: 不同 op 的 buffer 在时间上不重叠时共用

数学上, 设 $B_i$ 是 op $i$ 的输出 buffer, $t_i^s, t_i^e$ 是 lifetime start/end. In-place 约束: $B_i$ 与 $B_j$ 在 graph 上是 producer-consumer. Standard sharing 约束:
$$[t_i^s, t_i^e) \cap [t_j^s, t_j^e) = \emptyset$$
然后求解 graph coloring 类问题, 最小化 peak memory. 这块在 edge 部署 (memory 受限) 上特别重要, Reference: Ahn et al. "Ordering Chaos" (https://arxiv.org/abs/2003.02369).

**Layout transformation**: 找最优 tensor layout, 在 graph 中插 transform node. 实际 transform 不立即做, 留到 backend evaluation 时合并.

公式 (NHWC vs NCHW 影响卷积): 给定 conv $O[n, c_o, h, w] = \sum_{c_i, kh, kw} I[n, c_i, h+kh, w+kw] \cdot W[c_o, c_i, kh, kw]$:
- NCHW: $I$ stride $(C H W, H W, W, 1)$, 主访内沿 $w$ 连续
- NHWC: $I$ stride $(H W C, 1, W C, C)$, 主访内沿 $c$ 连续

GPU tensor core 在 NHWC 下 channel 维可以直接喂到 matrix core 的 contracting dimension, 性能更好. 但某些算子 (depthwise conv) NCHW 更自然, 因为 channel 维做独立 reduction.

更高级 layout: **NCHW[x]c** (Liu et al. ATC 2019, https://www.usenix.org/conference/atc19/presentation/liu-yizhi): 把 channel $C$ 拆成 $(C/x, x)$, $x$ 由 auto-tuning 决定, 匹配 CPU vector register width (AVX-512: x=16 for fp32, x=32 for int8). 公式上, 重写 conv:
$$O[n, c_o / x, h, w, x] = \sum_{c_i, kh, kw} I[n, c_i, h+kh, w+kw] \cdot W[c_o / x, x, c_i, kh, kw]$$

---

## 6. Backend optimizations

### 6.1 Hardware-specific

**Hardware intrinsic mapping**: 把一组 IR pattern 映射到 hand-tuned micro-kernel 或硬件指令. TVM 的 tensorization — 用户声明 intrinsic 的 behavior + lowering rule, 编译器自动匹配 IR pattern 替换.

例: tensor core 的 GEMM $D = A \cdot B + C$, 其中 $A \in \mathbb{R}^{16 \times 16}, B \in \mathbb{R}^{16 \times 8}$ (FP16). 在 IR 里识别到这种 tile pattern 就替换成 `mma.sync` PTX 指令.

**Memory allocation and fetching**: GPU 上 shared memory (low latency, small) vs global memory (high latency, large). TVM 用 memory scope schedule primitive 给 stage 打 tag (shared / thread-local), 自动生成 cooperative data fetching + memory barrier.

公式 (cooperative fetching): 一个 thread block $B$ 的 $T$ 个 thread 协作加载 tile $X_{B}$ from global to shared:
$$X_{B, \text{shared}}[\text{tid} \cdot k : (\text{tid}+1) \cdot k] = X_{\text{global}}[\text{offset}_B + \text{tid} \cdot k : \text{offset}_B + (\text{tid}+1) \cdot k]$$
其中 $\text{tid}$ 是 thread id, $k$ 是每 thread 加载的 element 数.

**Memory latency hiding**: GPU 靠 warp context switching 硬件隐藏; TPU 这种 DAE (decoupled access-execute) 架构需要 compiler 调度. TVM virtual threading 让用户写虚拟并行线程, 编译器把它们 interleave 成一个 instruction stream, 插 memory barrier.

**Loop oriented optimizations**:
- **Loop fusion**: 把同 boundary 的多个 loop 合并, 共享 iteration + 提升 locality
- **Sliding window**: 计算 on-the-fly + store for reuse, 直到不再需要 (Halide 的 `compute_at` 就是 sliding window 的一种 schedule)
- **Tiling**: 把 loop $i \in [0, N)$ 拆成外层 $ii \in [0, N), i = ii \cdot T + i'$ ($i' \in [0, T)$), 让 $T$ 大小匹配 cache 行
- **Loop reordering**: 重排 nesting order, 优化 spatial locality. 需注意 data dependence 不能被违反
- **Loop unrolling**: 展开 $n$ 次循环, 增加 ILP

**Parallelization**: SIMD + multi-thread. Halide `parallel(dim)` 把 dim 维并行化; GPU 上用 `block` / `thread` annotation 映射. Stripe 的 nested polyhedral model 在 tiling / striding 各级检测 hierarchy parallelism.

Glow 把 vectorization 完全交给 LLVM auto-vectorizer; TVM / TC 更倾向手工 + auto-tuning 联合, 因为可以注入 DL domain knowledge (知道 conv 的 contraction 维适合做 SIMD contracting).

### 6.2 Auto-tuning

四个组件:

**Parameterization**:
- Data parameter: input shape
- Target parameter: 硬件特性 (shared memory size, register file, cache line)
- Optimization option: schedule 选择 + 参数 (tile size, unroll factor)

**Cost model**:
1. Black-box (TC): 只看 final latency, 简单但搜索效率低
2. ML-based (TVM, XLA): GBDT (TVM) 或 FNN (XLA) 预测每个 config 的 latency
3. Pre-defined: 解析模型, 快但新硬件 / 新 op 需重写

GBDT 形式: $\hat{y}(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x})$, 其中 $f_k$ 是第 $k$ 棵回归树, $\mathbf{x}$ 是 config 特征向量 (tile sizes, loop order, vectorization width 等). 训练数据是实际测量的 (config, latency) 对.

**Searching technique**:
1. Initialization: 随机 / 已知好配置 / 用户指定
2. Genetic Algorithm (TC): crossover + mutation + selection, 适合离散 search
3. Simulated Annealing (TVM): 接受 worse solution 的概率 $P(\Delta E) = e^{-\Delta E / T}$, $T$ 降温, 避免局部最优
4. Reinforcement Learning (Chameleon, https://arxiv.org/abs/2001.08743): 把 schedule 选择建模成 MDP

**Acceleration**:
- Parallelization: 多 CPU thread 编译 + 多 GPU 评估 (TC)
- Config reuse: 编译 cache / log file, 跳过已测 config

TVM AutoTVM 后被 Ansor (Auto-scheduler, OSDI 2020, https://arxiv.org/abs/2006.06762) 取代: Ansor 用 program synthesis 自动生成 sketch + annotation, 不需要人工写 schedule template. 再之后 MetaSchedule (TVM 0.10+) 进一步用 MLIR-style unified tuning API: https://github.com/apache/tvm/tree/main/src/meta_schedule

### 6.3 Optimized kernel libraries

cuDNN (NVIDIA), DNNL (Intel), MIOpen (AMD), 以及 ASIC 自家的库 (Cambricon, Graphcore). DL compiler 可以在 codegen 阶段生成对它们的调用.

代价:
- 必须先做 layout / fusion style 转换, 可能破坏最优 control flow
- library 是黑盒, 不能跨 op 融合 (所以 kernel-lib 路线和 fusion 路线常有冲突)
- 库总比模型发展慢半拍

TVM / nGraph / TC 都支持 fallback 到库, 但默认走自己的 codegen. Glow / XLA 则深度依赖 cuDNN / DNNL.

---

## 7. 五大 DL compiler 对比 (paper Table 1)

| 维度 | TVM | nGraph | TC | Glow | XLA |
|---|---|---|---|---|---|
| Developer | Apache | Intel | Meta | Meta | Google |
| Programming | Python/C++ Lambda | Python/C++ Tensor expr | Python/C++ Einstein | Python/C++ Layer | TF interface |
| ONNX | built-in from_onnx | ngraph-onnx pkg | ✗ | ONNXModelLoader | tensorflow-onnx |
| Framework | TF/PyTorch/MXNet/Caffe2/CoreML/DarkNet | TF/Paddle (*-bridge) | PyTorch+DLPack | PyTorch/Caffe2/TF Lite | TF |
| Training | developing | only NNP-T | auto-diff single kernel | limited | TF support |
| Quant | int8/fp16 | int8 (+train) | ✗ | int8 | int8/int16 |
| High-IR | Relay | nGraph IR | TC IR | own | HLO |
| Low-IR | Halide | None | Polyhedral | own | HLO |
| Dynamic shape | Any | PartialShape | ✗ | ✗ | None |
| Auto-tuning | AutoTVM/Ansor | ✗ (call libs) | GA | ✗ | FNN (conv/gemm) |
| Kernel libs | MKL/cuDNN/cuBLAS | Eigen/DNNL/cuDNN | ✗ | ✗ | Eigen/MKL/cuDNN/TRT |
| Compilation | JIT + AOT(exp) | JIT | JIT | JIT + AOT | JIT + AOT |
| Devices | CPU/GPU/ARM/FPGA/VTA | CPU/Intel GPU/NNP/+PlaidML | NVIDIA GPU | CPU/GPU | CPU/GPU/TPU |

一个直觉的读法:
- TVM 是生态最广 + 最 academic 的, 偏 "compiler first"
- XLA 最 embedded in framework, 偏 "framework first"
- nGraph 是 Intel 阵营的, 重库 + IR 兼容
- TC 是 kernel-DSL 风格, 不算 full compiler
- Glow 早死了 (Meta 2020 年后停止投入), 思想被 TorchInductor 部分吸收

---

## 8. Evaluation insights

paper 在 V100 + 2080Ti + Broadwell + Skylake 上跑了 19 个 ONNX 模型 (ResNet, DenseNet, VGG, MobileNet, MNASNet). 关键发现:

1. **TVM tuned vs untuned**: CPU 上差不多 (经典 schedule 默认就够好), GPU 上 41.26× 平均加速 — 说明 GPU 的 schedule space 大, 不调就是垃圾
2. **Glow CPU 拉胯**: 不支持 thread parallelism, 多核 CPU 全废
3. **XLA end-to-end overhead 来自 TF runtime**: 单 batch 测延迟被 TF runtime 吃掉, batch 大了 throughput 才正常. 但 latency 敏感场景 XLA 不友好
4. **nGraph 在 CPU 上很好**: 因为 DNNL (MKL-DNN) 的 JIT 在 Intel CPU 上做了非常细的 instruction-level 优化, nGraph 只是 fallback 给 DNNL
5. **TC 性能不稳定**: autotuning cache 依赖初始 kernel, 后续层形状不同就要重调, 工程量大

per-layer 对比 (Figure 6-9) 显示:
- conv 层 TVM 在 GPU 上最稳, nGraph 在 CPU 上最好 (库给力)
- XLA 把 conv 和其他 op cluster 在一起, 单独测不到 conv 时间 (Table 3)
- Glow 在 1×1 conv 和 depthwise conv 上明显差 (融合策略不当)

paper 提到的一个 open challenge: frontend / backend 优化紧耦合, 难独立评估. 他们后来做了 model fragment benchmarking, 把目标 op + 必要 context 抽出来再喂给 compiler, 但这块到今天还是 unsolved (TorchBench / MLPerf 仍有这个问题).

---

## 9. Future directions (paper Section 7) 的现状对照

paper 列了 8 个方向, 我对照 2024-2026 的现状:

**Dynamic shape + pre/post-processing**:
- 2022 TVM Relax 把 dynamic shape 当一等公民 (https://arxiv.org/abs/2305.16091)
- PyTorch 2.0 TorchDynamo 解决 "Python-side 动态", TorchInductor 把 graph lower 到 Triton + C++
- JAX + XLA 走另一条路: jit(pmap(pjit(...))) 层层 lowering
- pre/post-processing: 现在 OpenXLA / IREE 把整条 pipeline (含 numpy / pillow 等 host ops) 也纳入

**Advanced auto-tuning**:
- Ansor / MetaSchedule: 全 sketch + annotation 自动 search, 不依赖 template
- Facebook AITemplate, NVIDIA's cuOPT, OpenAI's Triton autotune 都是演进
- 多目标 (memory, energy): 演化成 device-aware cost model

**Polyhedral model**:
- 仍然小众, 主要在 PlaidML, AKG (Ascend), TC 早期
- sparse tensor + polyhedral 的结合还没成熟
- 最近 LLM 推理 (sparse attention, MoE) 反而让 polyhedral 复活了一点

**Subgraph partitioning**:
- PyTorch 的 `fx.GraphModule` partition, ONNX Runtime 的 ORT optimize, Apple CoreML 的 subgraph split 到 ANE
- heterogeneous execution: 阿里 MNN, 小米 MACE 都做

**Quantization**:
- Relay 量化 rewriting 是早期思路
- 今天主流: LLM 的 INT4 / FP8 / NVFP4, 用硬件 native dtype 而不是模拟
- torchao, lm-head's quantize.py, TensorRT-LLM 把 quant 推到 tensor core 的 INT8 / FP8 native 路径

**Unified optimizations (MLIR)**:
- StableHLO (OpenXLA): https://github.com/openxla/stablehlo 已经成 standard graph IR
- MLIR 生态: Torch-MLIR (PyTorch → Linalg), IREE (MLIR → Vulkan/SPIR-V/CUDA)
- Modular Mojo: MLIR + Python-like syntax, 把 compiler 跟语言融合

**Differentiable programming**:
- Julia's Zygote.jl 已经成熟, Enzyme 做 source-to-source AD on LLVM
- JAX 的 jit(grad(...)) 是 tracing-based AD + XLA lowering 的典范
- 但 production 的 AD 仍以 PyTorch autograd (runtime) 为主, compile-time AD 还没大规模替代

**Privacy protection**:
- split inference + noise injection 在联邦学习里有, 但 production DL compiler 还没把这块做进 pass
- 同态加密 (CKKS) + compiler 的方向有 OpenFHE + MLIR 实验, 还早期

**Training support**:
- XLA 训练 OK (TPU + JAX 全家桶)
- TVM Relay training 仍 partial
- PyTorch Inductor 训练 + 推理都覆盖, 是目前最全的

---

## 10. 给 Karpathy 的 extra context

你这个 micrograd (https://github.com/karpathy/micrograd) 用 100 行 Python 实现了 reverse-mode AD + tiny autograd, 是 DL framework 的最简内核. paper 这套 DL compiler 是站在 framework 之上的下一层 — framework (PyTorch autograd / TF) 解决 "怎么算 gradient", DL compiler 解决 "怎么把 forward+backward graph 高效部署到硬件".

一个完整 mental model 的 hierarchy:
```
User code (micrograd / nanoGPT / PyTorch eager)
        │
   Tracing / symbolic evaluation
        ▼
Computation graph (PyTorch fx, TF Graph, JAX jaxpr)   <- framework IR
        │
   frontend lower
        ▼
Graph IR (Relay, StableHLO, HLO)                      <- DL compiler high-IR
        │
   frontend opt (fusion, layout, CSE, DCE, mem plan)
        ▼
Lowering
        │
   ▼
Tensor / Loop IR (TensorIR, Stripe, Halide IR)        <- DL compiler low-IR
        │
   backend opt (tile, vectorize, tensorize, par)
   + auto-tuning (AutoTVM/Ansor/MetaSchedule/Triton autotune)
        ▼
LLVM IR / Triton IR / MLIR LLVM dialect
        │
   ▼
Hardware (PTX, SASS, x86-64, ARM64, NPU firmware)
```

每一层 abstraction 都用对应的 IR, 越往下越 hardware-specific, 越往上越 semantic-rich. AutoTVM / Ansor / MetaSchedule 在中间两层做 search; Triton 让程序员手写中间层 (像 CUDA 但带 autotune); Mojo 把 MLIR 直接当 language backend.

paper 的最大贡献在 2019 那个时间点 (PyTorch 2.0 还没出, MLIR 刚开始, TVM 还是 0.6) 把整个 landscape 用一个清晰 framework 描述出来, 后续几个大的演化路线都能从这 8 个 future direction 里找到影子. 今天看仍然是一本好 "DL compiler 入门地图", 虽然具体细节 (Glow 死了, nGraph 在 Intel 重心下降, TC 没成气候) 已经过时, 但 abstraction skeleton 没变.

如果你要给学生讲 "为什么 PyTorch 2.0 用 TorchInductor 而不是直接 LLVM", 这篇 paper 提供了最完整的语言 — 因为 DL compiler 在中间多塞了 graph IR + tensor IR 两层, 这两层让 fusion + autotune + tensorization 有了 first-class 表达. 直接 LLVM 那一层做这些会非常痛苦, 因为 LLVM IR 太底层, op 边界消失.

参考链接补充:
- TVM's Relax (dynamic shape IR): https://arxiv.org/abs/2305.16091
- Ansor (auto-scheduler): https://arxiv.org/abs/2006.06762
- StableHLO: https://openxla.github.io/stablehlo/
- IREE: https://github.com/openxla/iree
- TorchInductor design: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-cpu-gpu-support-for-multiple-backends/1474
- Triton tutorial: https://triton-lang.org/getting-started/tutorials/01-vector-add.html
- MLIR Dialect guide: https://mlir.llvm.org/getting_started/
- Chris Lattner on MLIR & Mojo: https://www.modular.com/mojo
- Original Halide paper: https://people.csail.mit.edu/jrk/halide-pldi13/halide_pldi13.pdf
