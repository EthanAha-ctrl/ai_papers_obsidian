
















# NVIDIA/Fuser (nvFuser) —— 一个 NVIDIA GPU 的 Fusion Code Generator

## 一、项目定位与核心概念

**nvFuser** 是一个面向 **NVIDIA GPU** 的 **Deep Learning Fusion Compiler**，其核心使命是：

> **自动将多个细粒度 GPU kernel 融合（fuse）成少量粗粒度 kernel**，从而消除冗余的 global memory 读写、kernel launch overhead，提升端到端模型训练/推理速度。

用第一性原理来理解：GPU 程序的性能瓶颈本质上是 **memory bandwidth** 和 **kernel launch latency**。假设一个模型有 $N$ 个独立的 pointwise/reduction 算子：

- **不融合**：每个算子一次 global memory load + store，共 $2N$ 次 memory round-trip；每次 kernel launch 有 $\sim\!5\text{-}10\,\mu s$ 的 overhead。
- **融合后**：中间 tensor 直接留在 register/shared memory，只有首尾各一次 global memory 访问，共 $2$ 次；launch overhead 也从 $N$ 次降到 $1$ 次。

$$\text{Speedup}_{\text{ideal}} \approx \frac{N \cdot (t_{\text{launch}} + 2 \cdot t_{\text{mem}})}{t_{\text{launch}} + 2 \cdot t_{\text{mem}}}$$

其中 $t_{\text{launch}}$ 是单次 kernel launch 时间，$t_{\text{mem}}$ 是单次 global memory load/store 时间。当 $N$ 很大且 $t_{\text{mem}}$ 为主时，speedup 可达 $N$ 倍。

**GitHub Repo**: [https://github.com/NVIDIA/Fuser](https://github.com/NVIDIA/Fuser)

---

## 二、项目历史与演化

| 阶段 | 时间 | 关键事件 |
|------|------|----------|
| **origin** | ~2020-2021 | nvFuser 最初作为 PyTorch 内部的 `torch/csrc/jit/codegen/cuda` 存在，是 TorchScript 的 Profiling Graph Executor 的后端 |
| **PyTorch 集成** | PyTorch 1.12 (2022) | 成为 TorchScript 的 **默认** fusion 系统，替代了旧的 NNC (Naive Codegen) |
| **独立开源** | 2023 | 从 PyTorch repo 独立出来，移至 [NVIDIA/Fuser](https://github.com/NVIDIA/Fuser)，支持 standalone Python API |
| **Thunder 集成** | 2024 | Lightning-AI 发布 [Thunder](https://github.com/Lightning-AI/lightning-thunder)，将 nvFuser 作为其核心 codegen 后端 |
| **Multi-device** | 2024-2025 | 引入 Host IR、MultiDeviceExecutor，支持 multi-GPU 通信-计算重叠 |

---

## 三、架构全景（Architecture Diagram）

```
┌─────────────────────────────────────────────────────────────────┐
│                     User / Framework Layer                       │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │ PyTorch      │   │ Lightning Thunder│   │ Standalone API  │  │
│  │ TorchScript  │   │ source-to-source │   │ FusionDefinition│  │
│  └──────┬───────┘   └────────┬─────────┘   └────────┬────────┘  │
│         │                    │                      │            │
│         ▼                    ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Fusion Definition → Fusion IR (IR Builder)      ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                Segmentation (Segmenter / Segmenter Logic)    ││
│  │   Decides fusion boundary: which ops → which fused kernel     ││
│  └──────────┬────────────────────────┬────────────────────────┘│
│             │ (seg 1)                 │ (seg 2)   ...            │
│             ▼                         ▼                          │
│  ┌──────────────────┐    ┌──────────────────────┐               │
│  │   Scheduler 1    │    │   Scheduler 2        │   ...         │
│  │ (Pointwise /     │    │ (Reduction /          │               │
│  │  Transpose /     │    │  Persistent /         │               │
│  │  Matmul / ...)   │    │  InnerOuter / ...)    │               │
│  └────────┬─────────┘    └──────────┬───────────┘               │
│           │                        │                             │
│           ▼                        ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Lowering: Fusion IR → Kernel IR (Loop Nest)     ││
│  │   compute_at, parallelization, vectorization,                ││
│  │   unrolling, caching, circular buffering                      ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           Code Generation: Kernel IR → CUDA C++ Code        ││
│  │   GpuLower → codegen::CodeWriteCudaKernel                    ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           JIT Compilation: nvrtc → CUDA Cubin               ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           FusionExecutor: Launch Kernel on GPU              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、核心 IR 体系（Fusion IR）

nvFuser 的 IR 体系是其最核心的设计之一，采用 **多层 IR** 分层抽象：

### 4.1 基础 IR 节点

| IR Node | 说明 |
|---------|------|
| **`IterDomain`** | 表示一个迭代维度。关键属性包括：`start`, `extent`, `stop`（定义迭代范围）；`iter_type`（`Iteration` / `Reduction` / `Broadcast` / `GatherScatter` / `Stride`）；`is_persistent_dimension`；`parallel_type`（`ThreadX`, `ThreadY`, `ThreadZ`, `Vectorize`, `Unroll`, `Bulk`, ...）|
| **`TensorDomain`** | 由一组 `IterDomain` 组成，描述一个 tensor 的逻辑/计算/分配/loop 域。一个 `TensorView` 可能有多个 `TensorDomain`：`logical_domain`（用户视角），`compute_domain`（compute_at 变换后），`loop_domain`（最终循环结构），`allocation_domain`（物理存储布局） |
| **`TensorView`** | 对 tensor 的一个"视图"，是 IR 中最核心的节点。持有 `TensorDomain`、`dtype`、`memory_type`（`Register`, `Shared`, `Global`）。所有算子操作生成新的 `TensorView` |
| **`Val`** | 所有值的基类（scalar / tensor），是 IR DAG 的节点 |
| **`Expr`** | 所有表达式的基类，如 `UnaryOp`, `BinaryOp`, `TernaryOp`, `ReductionOp`, `BroadcastOp`, `WelfordOp`, `LoadStoreOp`, `IndexSelectOp` 等 |

### 4.2 TensorView 的多域变换

这是 nvFuser 最精妙的设计之一。一个 `TensorView` 在整个 compilation pipeline 中经历一系列域变换：

```
Logical Domain  ──[compute_at]──►  Compute Domain ──[parallelize]──►  Loop Domain
     │                                                                   │
     │                                                                   │
     └──[set_allocation_domain]──────────────────────────────►  Allocation Domain
```

**`compute_at`**：决定一个 tensor 在哪个 loop level 计算其值。语义上等价于 "将此 tensor 的计算 inline 到目标 tensor 的循环结构中"。例如：

```
T2 = T1 * T0   // T2 is computed at T3's position
T3 = relu(T2)
```

若 `T2->computeAt(T3, -1)`，则 T2 的计算被推入 T3 的 innermost loop，T2 的值留在 register 中，无需写回 global memory。

### 4.3 IterDomain 的 Parallel Type

`IterDomain` 上的 `ParallelType` 枚举直接映射到 GPU 的并行机制：

| ParallelType | 映射 |
|-------------|------|
| `ThreadX` | CUDA thread blockIdx.x / threadIdx.x |
| `ThreadY` | CUDA thread blockIdx.y / threadIdx.y |
| `ThreadZ` | CUDA thread blockIdx.z / threadIdx.z |
| `Vectorize` | 向量化加载（float4, float8, etc.）|
| `Unroll` | 循环展开 |
| `Mma` | Tensor Core MMA (Matrix Multiply-Accumulate) 操作 |
| `Bulk` | 异步拷贝（cp.async）|
| `Serial` | 普通 CPU-like 顺序循环 |

---

## 五、Compilation Pipeline 详解

### 5.1 Fusion Definition（定义阶段）

用户通过以下两种方式定义 fusion：

#### (a) Standalone Python API（`FusionDefinition`）

```python
from nvfuser import FusionDefinition, DataType
import torch

def my_fusion(fd: FusionDefinition):
    # 定义输入
    T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], 
                          dtype=DataType.Float, is_cpu=False)
    S0 = fd.define_scalar(1.0, dtype=DataType.Float)
    # 算子
    T1 = fd.ops.mul(T0, S0)       # T1 = T0 * 1.0
    T2 = fd.ops.add(T1, T0)       # T2 = T1 + T0 (residual)
    T3 = fd.ops.relu(T2)          # T3 = relu(T2)
    # 定义输出
    fd.add_output(T3)

with FusionDefinition() as fd:
    my_fusion(fd)

inputs = [torch.randn(1024, 1024, device='cuda')]
outputs = fd.execute(inputs)
```

#### (b) 通过 Thunder / TorchScript 自动生成

Thunder 框架将 PyTorch 程序 trace 成一系列 primitive operations，然后传给 nvFuser 的 Python API 或 C++ API 自动构建 Fusion IR。

### 5.2 Segmentation（分段阶段）

**Segmenter** 决定哪些算子应该被融合到同一个 kernel 中，哪些需要分割。这是一个 **NP-hard** 的决策问题，nvFuser 采用 **启发式规则** + **代价模型** 的混合方法：

**Segmentation 的核心约束**：
1. **Scheduler 兼容性**：同一个 fusion segment 内的所有算子必须能被同一种 scheduler 处理
2. **数据依赖**：如果有 reduction → pointwise → reduction 的链路，中间的 pointwise 只能跟随其中一个 reduction scheduler
3. **资源约束**：register pressure、shared memory 限制
4. **收益预估**：fusion 的收益是否大于分割的代价

**Segmentation 算法概述**：

```
1. 构建 fusion DAG (有向无环图)
2. 识别 DAG 中的 "分割点" (segmentation barrier):
   - 不同 scheduler 类型之间的边界
   - 形状不兼容的边界
3. 使用 BFS/DFS 将 DAG 分割为多个 "segment"
4. 对每个 segment，尝试匹配最优 scheduler
5. 若某 segment 匹配失败，则进一步细分
```

### 5.3 Scheduling（调度阶段）

nvFuser 的 scheduler 是一组 **预定义的调度模板**，每种模板针对特定类型的 computation pattern 进行了极致优化：

| Scheduler | 适用场景 | 关键优化 |
|-----------|----------|----------|
| **Pointwise** | 纯 element-wise 操作（relu, add, mul, ...） | Vectorized load/store, unroll, cache broadcast input at register |
| **Reduction (Inner)** | 内层维度 reduce（如 `sum(dim=-1)`） | Block-level reduction，warp shuffle, register accumulation |
| **Reduction (Outer)** | 外层维度 reduce（如 `sum(dim=0)`） | Grid-level reduction, multi-block cooperative |
| **Persistent** | 连续多次 reduction（如 LayerNorm, Softmax） | 中间结果 persist 在 register/shared memory |
| **Transpose** | 需要 dimension transpose 的 pattern | Shared memory transpose + vectorized load, 避免 bank conflict |
| **Matmul** | 矩阵乘法 (`GEMM/MMA`) | Tensor Core MMA, double buffering, cp.async |
| **Normalization** | LayerNorm / BatchNorm / RMSNorm | Welford online algorithm, register tiling |

**Scheduler 选择逻辑**（伪代码）：

```cpp
SchedulerType pickScheduler(FusionSegment* seg) {
    auto ops = seg->ops();
    bool has_reduction = any_of(ops, isReduction);
    bool has_matmul = any_of(ops, isMatmul);
    bool has_transpose = any_of(ops, needsTranspose);
    
    if (has_matmul)           return SchedulerType::Matmul;
    if (has_transpose)        return SchedulerType::Transpose;
    if (isPersistent(ops))    return SchedulerType::Persistent;
    if (has_reduction) {
        if (isInnerReduction(ops))  return SchedulerType::ReductionInner;
        else                         return SchedulerType::ReductionOuter;
    }
    return SchedulerType::Pointwise;
}
```

### 5.4 Lowering（IR 下降阶段）

**Lowering** 是将高层 Fusion IR 转换为底层 Kernel IR（loop nest）的过程。关键步骤：

1. **`compute_at` propagation**：通过 `compute_at` 关系确定每个 tensor 的计算位置
2. **`IterDomain` graph 构建**：建立 `IterDomain` 之间的映射关系（如 `SWizzle`, `Reshape`, `IndexSelect` 等变换）
3. **Loop nest 生成**：基于 `IterDomain` graph 和 `compute_at` 关系，生成嵌套循环结构
4. **Parallelization mapping**：将 `IterDomain` 映射到 GPU 的并行维度（`threadIdx`, `blockIdx`, vectorize, unroll 等）
5. **Memory promotion**：决定哪些 tensor 放在 register、shared memory 还是 global memory
6. **Circular buffering**（又称 double buffering）：对于 matmul 等 compute-intensive pattern，使用 `cp.async` 异步加载 + double buffer 来 overlap 数据加载和计算

**Loop Nest 示例**（Pointwise scheduler 生成的伪代码）：

```
// Grid: blockIdx.x, blockIdx.y; Block: threadIdx.x
// T0[i, j] = input[i, j]
// T1[i, j] = T0[i, j] * alpha
// T2[i, j] = relu(T1[i, j])
// T3[i, j] = T2[i, j] + T0[i, j]  // residual

for (i : vectorized over blockIdx.x * blockDim.x + threadIdx.x) {
  for (j : unrolled over inner_dim / vector_size) {
    float4 vec = *(float4*)&T0[i][j * 4];  // vectorized load
    T1_local[j*4:j*4+4] = vec * alpha;
    T2_local[j*4:j*4+4] = relu(T1_local[j*4:j*4+4]);
    T3_local[j*4:j*4+4] = T2_local[j*4:j*4+4] + vec[j*4:j*4+4];
    *(float4*)&T3[i][j*4] = T3_local;  // vectorized store
  }
}
```

### 5.5 Code Generation（代码生成阶段）

nvFuser 不生成 PTX 或 LLVM IR，而是生成 **CUDA C++ 代码**，然后通过 **NVRTC** (NVIDIA Runtime Compiler) 编译成 cubin：

1. **`codegen::CodeWriteCudaKernel`**：遍历 Kernel IR，输出 CUDA C++ 代码字符串
2. **Register allocation hints**：通过 `__launch_bounds__` 控制 register 使用量
3. **内联 PTX intrinsics**：对于 `MMA` (Tensor Core)、`cp.async` 等操作，直接嵌入 PTX inline assembly

### 5.6 Execution（执行阶段）

**`FusionExecutor`** 负责：
1. 调用 NVRTC 编译 CUDA C++ → cubin
2. 管理 kernel cache（相同 fusion + 输入形状 → cache hit）
3. 计算 launch parameters（grid size, block size, shared memory size）
4. Launch kernel 并传递输入/输出指针

**`FusionExecutorCache`** 在更上层管理多个 `FusionExecutor`，实现：
- **Dynamic shape support**：当输入形状变化时，自动 recompile 或从 cache 中查找匹配的 kernel
- **Respecialization**：对于 symbolic dimension，首次遇到具体值时 specialize，后续同 shape 则 cache hit

---

## 六、Dynamic Shape 支持

nvFuser 的 dynamic shape 支持是其区别于很多传统 compiler 的关键特性：

**核心思想**：
- Fusion IR 中的 `IterDomain` 的 `extent` 可以是 **symbolic** 的（即不是编译时常量）
- Scheduler 根据 **shape 的比值关系**（如 `extent_A / extent_B = constant`）做决策，而非绝对值
- 编译后的 kernel 使用 runtime 传入的形状参数动态计算 loop bound
- 当 shape ratio 发生变化时，触发 **respecialization**，重新编译

**Dynamic Shape 公式**：

设 $e_i$ 为第 $i$ 个 `IterDomain` 的 extent，定义 **shape ratio vector**：

$$\mathbf{r} = \left(\frac{e_1}{e_0}, \frac{e_2}{e_0}, \ldots, \frac{e_n}{e_0}\right)$$

若 $\mathbf{r}$ 不变，则可以复用同一份编译结果；若 $\mathbf{r}$ 变化，则 respecialize。

---

## 七、Multi-Device 支持（Host IR）

nvFuser 正在积极开发 **multi-GPU** 支持：

- **Host IR**：描述多设备之间的通信和计算编排。Host IR 在 CPU 侧执行，调度 NCCL 通信和 compute kernel 的执行顺序
- **MultiDeviceExecutor**：管理跨设备的 tensor sharding、通信（allreduce, allgather, reduce-scatter）和计算
- **Communication-Compute Overlap**：利用 circular buffering 在 Host IR lowering 阶段实现 matmul 和 allreduce 的 overlap

**Host IR 结构示意**：

```
[Device 0]                [Device 1]
  Matmul(A0, B0)            Matmul(A1, B1)
      │                         │
  AllReduce(C0)  ←────────→  AllReduce(C1)
      │                         │
  Pointwise(C0, D0)         Pointwise(C1, D1)
```

---

## 八、性能数据

根据 PyTorch 官方博客 [Introducing nvFuser](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/) 的数据：

| Model (HuggingFace) | Baseline (Eager) | nvFuser Speedup |
|---------------------|------------------|------------------|
| BERT-large | 1.0x | ~1.5x |
| GPT-2 | 1.0x | ~1.4x |
| T5-base | 1.0x | ~1.3x |
| ViT-base | 1.0x | ~1.12x |

**加速来源分解**：
- **~60%** 来自减少 global memory 访问（fusion 消除中间 tensor 的读写）
- **~25%** 来自减少 kernel launch overhead
- **~15%** 来自 scheduler 的针对性优化（vectorization, shared memory, Tensor Core 等）

---

## 九、与生态系统的集成

| 生态系统 | 集成方式 |
|---------|----------|
| **PyTorch TorchScript** | 作为 Profiling Graph Executor 的后端，自 PyTorch 1.12 起默认启用 |
| **Lightning Thunder** | Thunder 是 Lightning-AI 的 source-to-source compiler，nvFuser 是其核心 codegen 后端。[Thunder](https://github.com/Lightning-AI/lightning-thunder) 将 PyTorch 程序 trace → primitive operations → nvFuser fusion |
| **Standalone Python** | 直接通过 `pip install nvfuser` 安装，使用 `FusionDefinition` API 手动定义 fusion |
| **PyPI** | `pip install nvfuser` 或 `pip install nvfuser-cu121` (针对特定 CUDA 版本) |

---

## 十、构建与安装

```bash
# 克隆仓库
git clone https://github.com/NVIDIA/Fuser.git
cd Fuser
git submodule update --init --recursive

# 安装 Python 依赖
pip install -r requirements.txt

# 构建 (需要 C++20 编译器: GCC >= 13.1 或 Clang >= 19)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install

# 或者直接安装预编译 wheel
pip install nvfuser
```

---

## 十一、关键技术细节补充

### 11.1 Welford Online Algorithm（用于 Normalization）

LayerNorm / BatchNorm 需要计算均值和方差。传统方法需要两次遍历，nvFuser 使用 **Welford online algorithm** 在一次遍历中同时计算：

$$\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}$$

$$M_{2,n} = M_{2,n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n)$$

$$\sigma_n^2 = \frac{M_{2,n}}{n}$$

其中：
- $\bar{x}_n$ = 前 $n$ 个元素的均值
- $M_{2,n}$ = 前 $n$ 个元素的二阶中心矩
- $\sigma_n^2$ = 方差

这避免了两次 global memory round-trip，是 Persistent scheduler 的核心。

### 11.2 Vectorization

Pointwise scheduler 对 innermost dimension 自动应用 vectorized load/store。原理：

若 innermost dimension 的 `extent` 为 $E$，`dtype` 为 `float32`（4 bytes），则：
- 若 $E$ 是 4 的倍数，使用 `float4`（128-bit）load
- 若 $E$ 是 8 的倍数，使用 `float8`（256-bit）load（需 compute capability >= 7.0）

$$\text{Memory throughput}_{\text{vectorized}} = \frac{V \cdot \text{sizeof}(T) \cdot \text{bandwidth}_{\text{peak}}}{\text{latency}_{\text{per\_transaction}}}$$

其中 $V$ 是 vectorization factor。

### 11.3 Tensor Core MMA (Matrix Multiply-Accumulate)

Matmul scheduler 生成 Tensor Core 指令。关键参数：

- **MMA tile**: $M_{\text{tile}} \times N_{\text{tile}} \times K_{\text{tile}}$，如 $16 \times 16 \times 16$ (Ampere) 或 $16 \times 8 \times 16$ (Turing)
- **Double buffering**: 将 `cp.async` 异步拷贝与 MMA 计算重叠
- **Swizzle pattern**: 在 shared memory 中重排数据布局，避免 bank conflict

$$\text{MMA throughput} = \frac{M_{\text{tile}} \times N_{\text{tile}} \times K_{\text{tile}}}{t_{\text{MMA}}}$$

---

## 十二、Debug 与 Profiling 工具

nvFuser 提供了一系列环境变量用于调试：

| 环境变量 | 作用 |
|---------|------|
| `NVFUSER_DUMP=python_definition` | 打印 FusionDefinition 的 Python 代码 |
| `NVFUSER_DUMP=cuda_kernel` | 打印生成的 CUDA C++ 代码 |
| `NVFUSER_DUMP=scheduler_params` | 打印 scheduler 的参数选择 |
| `NVFUSER_DUMP=fusion_ir` | 打印 Fusion IR |
| `NVFUSER_DUMP=segmenter_logging` | 打印 segmentation 过程 |
| `NVFUSER_DISABLE=parallel_compile` | 禁用并行编译（用于 debug）|

---

## 十三、总结：nvFuser 的核心设计哲学

1. **JIT Compilation**：运行时根据实际输入 shape 生成最优代码，避免 AOT 编译无法利用 runtime 信息的问题
2. **Multi-level IR**：从高层 Fusion IR → Kernel IR → CUDA C++ 的分层抽象，每层关注不同维度的优化
3. **Scheduler-based**：使用预定义的高质量调度模板而非通用 polyhedral 编译，在常见 deep learning pattern 上获得更稳定、更极致的性能
4. **Dynamic Shape First**：从 IR 设计时就支持 symbolic extent，而非将 dynamic shape 作为后加功能
5. **Ecosystem Integration**：既可作为 PyTorch 的隐形加速器，也可作为 Thunder 的核心后端，也可独立使用

---

## 参考链接

- **GitHub Repo**: [https://github.com/NVIDIA/Fuser](https://github.com/NVIDIA/Fuser)
- **PyTorch Blog - Introducing nvFuser**: [https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/)
- **PyPI Package**: [https://pypi.org/project/nvfuser/](https://pypi.org/project/nvfuser/)
- **Lightning Thunder**: [https://github.com/Lightning-AI/lightning-thunder](https://github.com/Lightning-AI/lightning-thunder)
- **Thunder Installation (includes nvFuser)**: [https://lightning-thunder.readthedocs.io/en/stable/fundamentals/installation.html](https://lightning-thunder.readthedocs.io/en/stable/fundamentals/installation.html)
- **PyTorch Dev Discuss - Tracing with Primitives**: [https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
- **Colab Tutorial**: [https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/3aee46f78e182e15b64326eddd31f2e2/nvfuser_intro_tutorial.ipynb](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/3aee46f78e182e15b64326eddd31f2e2/nvfuser_intro_tutorial.ipynb)
- **Multi-GPU Host IR Issue**: [https://github.com/NVIDIA/Fuser/issues/6003](https://github.com/NVIDIA/Fuser/issues/6003)