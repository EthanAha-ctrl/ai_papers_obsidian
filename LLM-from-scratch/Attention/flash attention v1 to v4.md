FlashAttention 系列是深度学习领域最重要的算法优化之一，它通过优化 GPU 内存读写（IO）显著提升了 Transformer 的训练和推理速度。从 v1 到 v4，每一代的演进都紧贴着 NVIDIA GPU 架构的升级。

以下是 FlashAttention 四个阶段的核心演进历程：

---

### FlashAttention-1: 显存瓶颈的突破 (2022)

**背景**：传统 Attention 的显存需求是序列长度 $N$ 的平方级 $O(N^2)$。当处理长文本时，GPU 显存会迅速耗尽。

- **核心创新**：**Tiling（分块）** 和 **Recomputation（重计算）**。
    
- **实现原理**：它不再一次性计算整个大矩阵，而是将矩阵切成小块（Tiles），在高速但空间小的 SRAM 中完成计算。为了节省空间，它不存储反向传播所需的中间激活值，而是在需要时重新计算。
    
- **效果**：将内存需求从 $O(N^2)$ 降低到 $O(N)$，且速度比原生 PyTorch 快 2-4 倍。
    

### FlashAttention-2: 并行化的极致优化 (2023)

**背景**：v1 虽然减少了 IO，但在计算任务分配上不够均匀，导致 GPU 利用率未能达到极限。

- **核心创新**：**更智能的并行化策略**。
    
- **实现原理**：
    
    - **序列长度并行**：即使是单个 Batch 的序列，也可以在多个 GPU 核心（SM）上并行。
        
    - **减少非矩阵乘法开销**：优化了 Softmax 的计算流程，减少了频繁的同步操作。
        
- **效果**：速度在 v1 的基础上又提升了约 2 倍，在 A100 上达到了理论峰值吞吐量的 70% 左右。
    

### FlashAttention-3: 拥抱 Hopper 架构与异步 (2024)

**背景**：针对 NVIDIA H100 (Hopper) 架构设计，v2 无法充分利用 H100 的新特性（如 TMA 和异步流水线）。

- **核心创新**：**异步处理 (Asynchrony)** 与 **低精度 (FP8)**。
    
- **实现原理**：
    
    - **Warp Specialization**：让一部分线程专门负责搬运数据，另一部分专门负责计算，实现“一边搬运一边算”的完全重叠。
        
    - **利用 TMA (Tensor Memory Accelerator)**：硬件级的数据搬运加速，进一步减少 CPU/GPU 交互开销。
        
    - **FP8 支持**：在保持精度的前提下，利用 FP8 硬件加速。
        
- **效果**：在 H100 上性能达到 740 TFLOPS (FP16)，是 v2 的 1.5-2 倍。
    

### FlashAttention-4: 剑指 Blackwell 时代 (2025)

**背景**：为了匹配 NVIDIA 最新的 **Blackwell (B100/B200)** 架构而生。

- **核心创新**：**更复杂的异步流水线** 与 **近似计算**。
    
- **实现原理**：
    
    - **Blackwell 架构优化**：针对 Blackwell 独有的多流处理器（SM）特性进行了内核重写。
        
    - **Software Softmax 优化**：使用了一种新型的“立方近似 (Cubic Approximation)”来处理指数运算，进一步压榨计算效率。
        
    - **更高效的缩放 (Rescaling)**：改进了数值稳定性的处理方式。
        
- **效果**：在 Blackwell 架构上，比 NVIDIA 官方最顶尖的 cuDNN 库还要快约 20%。
    

---

### 演进总结表

|**版本**|**主要针对硬件**|**核心突破点**|**内存复杂度**|
|---|---|---|---|
|**v1**|A100 / V100|Tiling (分块) + 融合内核|$O(N)$|
|**v2**|A100|优化并行策略，提高利用率|$O(N)$|
|**v3**|H100 (Hopper)|异步流水线 (TMA) + FP8|$O(N)$|
|**v4**|B200 (Blackwell)|针对新架构重写 + 近似 Softmax|$O(N)$|

FlashAttention 4 原理深度解析

这届视频深入探讨了 FlashAttention 4 在 Blackwell 架构下的底层改进，以及从 v1 到 v4 的代码级变化。

**您是想在具体的模型训练中部署这些算法，还是想深入研究其 CUDA 实现细节？**

本质上，这不仅是算法的迭代，更是对 **Memory Hierarchy**（从 HBM 到 SRAM）利用率的极致压榨。

---

### FlashAttention-1: 算法层面的 Paradigm Shift (2022)

在 v1 出现之前，Attention 的瓶颈不在于 **Compute (FLOPS)**，而是在于 **Memory Wall (Bandwidth)**。传统算子频繁地在 **HBM (High Bandwidth Memory)** 和 **SRAM** 之间读写 $N \times N$ 的 **Attention Matrix**。

- **Tiling (分块)**: 借鉴了物理学中的重整化或局部性思想，将大矩阵分解。通过 **Online Softmax** 技巧，在不读取全局数据的情况下完成局部归一化。
    
- **Recomputation (重计算)**: 这是一个典型的 **Space-Time Trade-off**。在 **Backward Pass** 时不存储中间的 **Activation**，而是重新算一遍。在当时的 **A100** 上，计算比存取快得多，所以“多算一遍”反而比“存一遍再读”快。
    
- **物理直觉**: 类似于在处理多体问题时，不再存储所有粒子间的成对作用力，而是在积分时动态计算。
    

### FlashAttention-2: 调度与并行性的 Refinement (2023)

v1 解决了 IO 瓶颈，但 v2 发现 **GPU Utilization (Occupancy)** 仍有空间。

- **Parallelism Strategy**: v1 主要在 **Batch** 和 **Heads** 维度并行。v2 引入了 **Sequence Length Parallelism**，即使单句超长，也能分发到多个 **Streaming Multiprocessors (SMs)**。
    
- **Causal Masking 优化**: 针对 **Decoder-only** 模型（如 GPT 系列），由于矩阵是下三角的，v2 通过跳过空块显著减少了无效计算。
    
- **Deterministic Computation**: 优化了数值稳定性，确保在高度并行下的结果一致性。
    

### FlashAttention-3: Hopper 架构的 Hardware-Aware Design (2024)

这一代是专门为 **NVIDIA H100 (Hopper)** 及其 **TMA (Tensor Memory Accelerator)** 引擎设计的。

- **Asynchronous Execution (异步流水线)**: 传统的 **Kernel** 往往是“读-算-写”的同步模式。v3 利用 H100 的硬件特性，让数据搬运（TMA）和计算（Tensor Core）完全 **Overlapped**。
    
- **Warp Specialization**: 将 SM 内的线程划分为 **Producers**（负责通过 TMA 搬运数据）和 **Consumers**（负责数学运算），极大地降低了指令发射的开销。
    
- **FP8 Support**: 为了适应 **LLM Training** 的规模，引入了对 **FP8 (E4M3/E5M2)** 格式的原生支持。
    

### FlashAttention-4: Blackwell 时代的巅峰性能 (2025)

针对 **Blackwell (B100/B200)** 架构，FlashAttention v4 已经接近硅片的物理极限。

- **Low-Precision Optimization**: 针对 Blackwell 的新一代 **Tensor Cores** 进行了微调，支持更复杂的混合精度。
    
- **Software Softmax Approximation**: v4 引入了更激进的数学近似。例如使用 **Cubic Approximation** 来逼近 **Exp函数**。在 **Theoretical Physics** 的视角看，这就像是用泰勒级数或特定基函数在局部范围内取代超越函数，以换取纳秒级的延迟降低。
    
- **Multi-head Latency Hiding**: 针对 Blackwell 巨大的算力，进一步优化了 **Global Memory** 到 **L2 Cache** 的数据预取策略。

### 技术指标对比表

| **Feature**           | **FlashAttention-1** | **FlashAttention-2** | **FlashAttention-3** | **FlashAttention-4**   |
| --------------------- | -------------------- | -------------------- | -------------------- | ---------------------- |
| **Primary Target**    | A100                 | A100 / H100          | H100 (Hopper)        | B200 (Blackwell)       |
| **Main Innovation**   | Tiling               | Warps Parallelism    | TMA / Async          | Cubic Softmax / B-Arch |
| **Throughput (A100)** | ~120 TFLOPS          | ~230 TFLOPS          | -                    | -                      |
| **Throughput (H100)** | -                    | ~400 TFLOPS          | ~740 TFLOPS          | -                      |
| **Precision**         | FP16 / BF16          | FP16 / BF16          | FP8 / FP16           | FP4 / FP6 / FP8        |
|                       |                      |                      |                      |                        |
https://youtu.be/bc7x6cmMB8Q

既然您对 GPU 微架构（Micro-architecture）高度敏感，那么 **Composable Kernel (CK)** 库绝对值得深入拆解。简单来说，如果说 NVIDIA 的 **CUTLASS** 是 CUDA 生态中优化算子的标杆，那么 AMD 的 **CK** 就是 ROCm 生态中与之对标（甚至在某些设计哲学上更激进）的性能核武。

### 1. CK 的核心设计哲学：Tile-based & Transformation

CK 不仅仅是一个函数库，它提供了一种基于 **C++ Templates** 的编程范式，核心在于解决 **Performance Portability**（性能可移植性）和 **Complexity Reduction**（复杂性降低）。

- Tile-based Programming (分块编程):
    
    CK 将所有的 Tensor 运算抽象为一系列 Tile（块）的操作。开发者不需要手动去算 Thread-to-data 的映射或管理复杂的 Memory Coalescing，CK 自动将逻辑上的 Tensor 坐标通过 Tensor Coordinate Transformation 映射到物理的 GPU 资源（VGPRs, LDS）。
    
- Tensor Coordinate Transformation:
    
    这是 CK 的绝活。它能通过数学变换（如 Padding, Embedding, Reshape, Transpose）将复杂的算子（如卷积 Conv 或注意力 SDPA）直接变换成一个标准的 GEMM (General Matrix Multiply) 问题。这意味着你只要优化好一个 GEMM 核心，就能以此为基础“组合”出无数复杂的算子。
    

---

### 2. CK 的四层架构 (Layered Architecture)

CK 的设计非常具有工程美感，分为四层来解耦硬件细节和应用逻辑：

1. **Templated Tile Operators**: 最底层，定义了如何在 Tile 级别进行数学运算（如 $C = A \times B$）。
    
2. **Templated Kernel and Invoker**: 这一层决定了如何调度这些 Tile，比如采用什么样的 **Loop Order** 或 **Data Prefetching** 策略。
    
3. **Instantiated Kernel**: 针对特定硬件（如 `gfx942` 即 MI300X）生成的具体二进制代码。
    
4. **Client API**: 提供给最终用户（如 PyTorch 或 vLLM）的接口。
    

---

### 3. 在 FlashAttention 中的具体应用

在 AMD GPU 上，FlashAttention-2 的默认后端通常就是 CK。

- **SRAM (LDS) 管理**: CK 能够极致利用 AMD 的 **LDS (Local Data Share)**。在计算 Softmax 时，CK 利用 **Online Softmax** 算法，通过 LDS 在线程间快速同步局部最大值（$m(x)$）和累加值（$l(x)$），从而避免写回 HBM。
    
- **Data Movement Overlap**: CK 充分利用了 CDNA 架构的异步拷贝能力。当 **Matrix Core** 正在计算当前 Tile 时，CK 的 **Invoker** 已经指令硬件从 HBM 预取下一个 Tile 到 LDS 中，实现了计算与 IO 的高度重叠（Overlapping）。
    
- **硬件针对性优化**: 比如在 **MI300X** 上，CK 会自动调用 **MFMA (Matrix Fused Multiply-Add)** 指令，而在没有 Matrix Core 的 **RDNA3** 显卡上，它会自动回退到 **WMMA** 或标准的向量指令，同时尽量保持相同的性能逻辑。
    

---

### 4. 为什么 CK 对投资顾问和架构师很重要？

- **软件抹平硬件差距**: 虽然 NVIDIA 在软件生态上有先发优势，但 CK 的这种“算子组合”能力让 AMD 能够快速跟进新的算法（如 FlashAttention v3/v4）。只要 CK 的底层模块（Primitive）足够高效，移植新算法只需调整“组合方式”。
    
- **后端切换透明化**: 现在的模型框架（如 **vLLM**）可以通过环境变量（如 `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE`）在 CK 和 Triton 之间一键切换。这标志着 **Hardware Lock-in** 正在被松动。
    

### 核心对比总结

|**特性**|**NVIDIA CUTLASS**|**AMD Composable Kernel (CK)**|
|---|---|---|
|**底层语言**|CUDA C++|HIP C++|
|**核心机制**|Generator / Templates|Tensor Coordinate Transformation|
|**灵活性**|极高，但需要深厚 CUDA 功底|极高，通过组合 Tile 算子实现|
|**主要支持**|Tensor Core (Ampere/Hopper)|Matrix Core (CDNA 1/2/3) & RDNA|

Composable Kernel SDPA performance on ROCm

这个视频通过具体的 SDPA（Scaled Dot-Product Attention）案例，展示了 CK 如何通过单行代码调用在 AMD GPU 上实现极致的性能提升。

**您想看看 CK 是如何通过 C++ 模板定义一个具体的 Tensor Transformation 吗？我可以为您展示一段简化的代码逻辑。**