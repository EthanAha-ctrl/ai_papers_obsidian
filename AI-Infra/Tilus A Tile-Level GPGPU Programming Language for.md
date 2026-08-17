---
source_pdf: Tilus A Tile-Level GPGPU Programming Language for.pdf
paper_sha256: 0871ac7c3d2147336baea59763de3a3732a9343c103b4ebd7939a19c8b54c7c4
processed_at: '2026-08-12T16:12:33-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Tilus

## 故事的起点

LLM 太大了，大家想用低 bit（4bit、6bit 等）来省内存、加速。4bit 量化能省 4 倍内存带宽，但 accuracy 掉得厉害。5/6/7 bit 能保 accuracy，可是 GPU 根本没有高效 kernel 支持——因为 GPU 硬件和软件栈只优化了 2 的幂次 bit width（4、8、16、32）。这就是 Tilus 要解决的核心矛盾：**accuracy 想要 5/6/7 bit，硬件只喜欢 4/8 bit，中间有巨大的 gap**。

## 现有方案为什么不行

先看 Triton。Triton 的 philosophy 是"把一切抽象掉"——tile 是一等公民，你看不见 tile 里数据怎么分布在 32 个 thread 的 register 里。这听起来很美，但 low-precision 场景下会卡壳。

举个 uint4 的例子。你想 load 一块 uint4 weight 进 register 算 matmul，Triton 的流程是：
1. 从 global memory load uint32 到 shared memory
2. 从 shared memory load 到 register
3. 手动 unpack：把 uint32 拆成 8 个 uint4
4. 转成 float16
5. 但 tensor core 要的 layout 跟你 unpack 出来的不一样，还得绕回 shared memory 做 layout 转换

Step 5 是致命的——shared memory round-trip 开销巨大，而且 Triton 的 programming model 让你**根本没法绕过这一步**，因为它把 shared memory 和 register layout 都藏起来了。

再看 Ladder（基于 TVM）。它用 "type-level packing"：把 4 个 int6 打包成 3 个 uint8 存储。这招对 power-of-two 的 bit width 还行，但对 6 bit 这种就尴尬了——pack 不齐。而且它的 schedule primitive 写不出 software pipelining，batch size > 1 的 decode 阶段性能就很差。

Marlin 呢？hand-crafted 的 INT4 专用 kernel，性能极强，但只支持 INT4、不支持 Hopper GPU、代码几千行很难维护。

## Tilus 的核心 trick

Tilus 的核心 insight 就一句话：**把 layout 变成代数系统，变成一等公民**。

### Layout 是什么？

想象 32 个 thread，每人手里有几张牌（register 里的数据）。Layout 就是"第 t 个 thread 的第 i 张牌对应 tile 的哪个 (row, col) 位置"。Triton 把这个映射藏起来了，Tilus 把它显式暴露出来，而且用 Kronecker 积构造复合 layout。

为什么要 Kronecker 积？因为 GPU 硬件指令的 layout 本身就是分层的。比如 PTX 的 `mma.m16n8k16` 的 operand A 的 layout 是：

$$\text{local}(2,1) \otimes \text{spatial}(8,4) \otimes \text{local}(1,2)$$

从外到内三层：先分 2 个"行块"，每个行块再按 8×4 分布到 32 个 thread，每 thread 最后存 1×2 的两个 element。这种层级结构用 Kronecker 积表达最自然。

### Zero-cost Reinterpretation：关键魔法

这是 Tilus 最漂亮的地方。回到 int6 的例子：

- 你要 load 一块 `i6[16, 8]`，每 thread 存 4 个 int6 = 24 bits
- 24 bits 也可以看成 3 个 uint8 = 24 bits
- **因为每 thread 的 bit 数完全相同，你可以零成本地把 register 里的数据"假装"成另一种类型**

具体流程（Figure 2c）：
1. 在 global memory 里先把 weight tile 的 layout 转成 `u8[BK*BN*6/8]` 连续存储（这是个 preprocessing，离线做一次）
2. Load 的时候按 uint8 来 load，用 `cp.async.v4` 这种硬件友好的 vectorized instruction
3. Load 到 register 后，用 `View` 指令零成本 reinterpet 成 `i6[16, 8]`，layout 变成 tensor core 要的那个
4. Cast 成 float16，喂给 tensor core

**全程不需要 shared memory round-trip**。这就是 Figure 1 里三个 pipeline 的本质区别：Triton 要绕 shared memory 转 layout，Ladder 没有 pipelining，Tilus 直接在 register 里 reinterpret 完事。

为什么 Triton 做不到这一步？因为它看不见 layout，**没法验证"这两个 tensor 的 bit 分布是否兼容"**。Tilus 的代数 layout 系统能做这个验证，这是 zero-cost reinterpretation 的前提。

### 暴露 memory hierarchy

Tilus 的另一个关键选择是暴露 GPU 的三层 memory（global、shared、register）。这听起来像把复杂性丢给程序员，但实际上：

- 不暴露 shared memory → 没法做 software pipelining（pipeline 需要在 shared memory 里维护多个 buffer 来 overlap load 和 compute）
- 不暴露 register layout → 没法做 register-level reinterpretation

Tilus 用 "thread-block-level programming model"（SIMB）来 mitigate 复杂性——每个 instruction 是整个 thread block 的操作，不是单个 thread 的操作。所以虽然你能精细控制 memory，但心智负担还是比 CUDA 的 SIMT 小很多。

## 性能数据说了什么

vs cuBLAS FP16 baseline：

| 对手 | Speedup | 备注 |
|---|---|---|
| Triton | 1.75× | Triton 的 layout 转换太贵 |
| Ladder | 2.61× | Ladder 没 pipelining + type packing 限制 |
| QuantLLM | 1.29× | QuantLLM 只支持 FP5/FP6 |
| Marlin | 1.03× | 几乎打平，Marlin 是 INT4 hand-crafted 极限 |

Marlin 那个 1.03× 最值得琢磨。Marlin 是 hand-tuned 的 INT4 专用 kernel，理论上已经是 near-optimal。Tilus 能打平甚至略微胜出，原因可能是：

1. Tilus 能 auto-tune tile size，Marlin 是 hardcoded 的
2. Software pipelining 在 Tilus 里是一等公民
3. 同一个 program template 能适应不同 batch size，Marlin 的 pipeline depth 是固定的

但更重要的不是这 1.03×——**而是 Tilus 用同一个 parameterized template 覆盖了 uint1 到 float8 的全 spectrum**。传统方法是每种 bit width 都要 hand-craft 一个 kernel，Tilus 用一套代码 + auto-tune 解决所有。这才是真正的 system-level 胜利。

## 一个 broader 的 intuition

Tilus 的哲学跟 Triton 完全相反。Triton 说"把硬件复杂性藏起来，让程序员开心"。Tilus 说"把硬件暴露出来，让程序员能榨干性能"。这两种哲学在不同场景下都对——对于标准 FP16/INT8 matmul，Triton 已经够好了；对于 low-precision LLM serving 这种 latency-critical 且 hardware-antagonistic 的场景，必须暴露细节才能赢。

从长期看，Tilus 的 VM-based architecture 比 hand-crafted kernel 更有生命力。新 GPU 架构（Blackwell 的 FP4 tensor core、TMA、warp group MMA）只需要加新的 VM instruction 就能支持，不用重写整个 kernel。Compiler-based approach 的核心价值就在这里——**一次抽象，多架构复用**。

我自己看完这篇 paper 的感受：layout algebra 这个 idea 其实不新（Halide、TVM、Graphene 都碰过），但 Tilus 把它做对了——用 Kronecker 积 + unified representation 让 layout 既能表达硬件指令的层级结构，又能做 compositional analysis。这个"做对"的功夫，可能比 idea 本身更值钱。

---

# Tilus: Tile-Level GPGPU Programming Language 详解

## 1. 核心动机与问题定位

LLM serving的核心瓶颈在quantized matmul (e.g., A16W4 quantization可将DRAM bandwidth consumption降至1/4)。但是当前GPU hardware和software stack过度偏向power-of-two bit widths (4, 8, 16, 32)，对于5-bit, 6-bit, 7-bit等"硬件不友好"的bit width缺乏efficient kernel支持。这是一个严重的accuracy-efficiency tradeoff gap——4-bit quantization (e.g., QuaRot [1], GPTQ [2], AWQ [3], QuIP# [4]) 都存在non-negligible accuracy degradation，而5/6/7-bit quantization (eXmY format [5]) 虽然能preserve accuracy，却缺乏GPU kernel支持。

现有approaches的几个具体限制：

- **Triton** [6]: 抽象了tensor layout，不暴露GPU memory hierarchy (shared memory, registers)。处理uint4需要手动从uint32 unpack，并且layout conversion要经过shared memory round-trip（Figure 1a的Step 4），这是一个主要的性能bottleneck。
- **Ladder** [7] (基于TVM [8]): 用type-level packing，把low-precision type打包成storage type，所以只能处理power-of-two bit width。此外primitive-style scheduling无法表达software pipelining。
- **Marlin** [9]: 4-bit signed integer专用，不支持Hopper architecture (H100)。
- **QuantLLM** [10]: 只支持floating-point 5/6-bit，不支持sub-channel quantization granularity。

Tilus的目标是构建一个**single program template**能通过parameterize tile sizes支持uint1→uint8, int2→int8, float3→float8 (任意exponent/mantissa分布) 的全谱quantized matmul。

## 2. 系统架构总览

Tilus的设计建立在四个key ideas上：

### 2.1 GPGPU Virtual Machine
整个Tilus program是一个abstract GPGPU VM的instruction sequence。VM有完整的instruction set (Table 1)。这个abstraction layer的核心价值是decouple programming model from specific GPU architecture features——添加新架构支持只需要增加新的VM instruction。

### 2.2 Thread-Block-Level Programming Model (SIMB)
区别于PTX/CUDA的SIMT (Single-Instruction-Multiple-Thread)，Tilus采用SIMB (Single-Instruction-Multiple-Block)。每个instruction operation在thread block级别定义。这跟Triton的tile-level abstraction类似，但关键区别是Tilus显式暴露hierarchical memory space (global, shared, register)。

### 2.3 Algebraic Layout System
这是Tilus的核心创新。需要解释一个具体的layout形式化问题：tensor的elements如何分布在warp的32个threads的registers中。

### 2.4 Native Low-Precision Data Type Support
支持1-8 bit arbitrary precision，包括signed integer, unsigned integer, floating-point (任意exponent/mantissa ratio)。

## 3. Algebraic Layout System 深度解析

### 3.1 Layout的形式化定义

一个distributed layout被定义为一个function $f$，它把(thread index $t$, local element index $i$) 映射到tensor的logical index $(r, c)$：

$$f(t, i) = (r, c)$$

例如，PTX中的 `mma.m16n8k8.f32.f16.f16.f32` instruction的operand A layout (Figure 3)：

$$f(t, i) = (t/4 + i/2 \times 8, \quad t\%4 \times 2 + i\%2)$$

其中：
- $t \in [0, 31]$: warp内的thread index
- $i \in [0, 3]$: 每个thread内的local element index（即每个thread存储4个elements）
- $r \in [0, 15]$: tensor的row index (16行)
- $c \in [0, 7]$: tensor的column index (8列)
- 总共16×8=128个elements分布在32个threads上，每thread 4个elements

这个layout function刻画了"thread $t$的local element $i$对应tensor的哪个$(r, c)$位置"。

### 3.2 Primitive Layouts

定义两类基础layout：

**Local Layout** `local(n1, n2)`：把所有$n_1 \times n_2$个elements放在一个thread内
$$f(t, i) = (i/n_2, \quad i\%n_2)$$
注意这里 $t$ 实际上不参与映射，因为所有elements都在一个thread里。

**Spatial Layout** `spatial(n1, n2)`：把$n_1 \times n_2$个elements分布在$n_1 n_2$个threads上，每thread一个element
$$f(t, i) = (t/n_2, \quad t\%n_2)$$
这里 $i$ 恒为0（每thread一个element）。

### 3.3 Kronecker Product of Layouts

这是layout algebra的核心操作。给定两个layout $f$ 和 $g$，它们的Kronecker product $h = f \otimes g$ 定义为：

$$h(t, i) = f(t/T_g, \quad i/N_g) \odot S_g + g(t \% T_g, \quad i \% N_g)$$

变量含义：
- $T_g$: layout $g$ 使用的thread数量
- $N_g$: layout $g$ 中每个thread存储的local element数量
- $S_g$: layout $g$ 的shape（$n_1 \times n_2$形式的tuple）
- $\odot$: element-wise product
- $t/T_g$ 和 $i/N_g$: 对应"外层"的thread和local index
- $t\%T_g$ 和 $i\%N_g$: 对应"内层"的thread和local index

**Intuition**: 把 $f$ 的每个element当作一个"macro-pixel"，这个macro-pixel内部是 $g$ 的layout。这是**层级化分布**的代数表达。

### 3.4 例子：mma.m16n8k16 的Operand A Layout

PTX的`mma.m16n8k16` instruction的operand A (shape 16×8, 分布在32 threads，每thread 4 elements) 的layout可以表示为：

$$\text{local}(2, 1) \otimes \text{spatial}(8, 4) \otimes \text{local}(1, 2)$$

拆解理解：
- 最外层 `local(2, 1)`: 把16行分成8组，每组2行（一个thread处理2个不同的"行块"）
- 中间 `spatial(8, 4)`: 32 threads按8×4的grid分布（每thread group共享同一对行块）
- 最内层 `local(1, 2)`: 每个thread的4个elements在column方向上排列2个

这个表达式的associativity使得layout可以分层级理解和操作。注意Kronecker product不满足commutativity：$f \otimes g \neq g \otimes f$。

### 3.5 Unified Layout Representation

统一的四元组表示 $(S, M, S_{spatial}, S_{local})$：
- **shape** $S$: register tensor的shape
- **mode_shape** $M$: 将每个dimension split成sub-dimensions后的shape
- **spatial_modes**: 分配给spatial threads的sub-dimensions
- **local_modes**: 分配给每个thread的local storage的sub-dimensions

例如Figure 6的layout：register tensor shape $[4, 8]$，mode_shape $[2, 2, 2, 2, 2, 2]$（即把每个dim分成3个sub-dims），spatial_modes选择某些sub-dims作为thread index。

给定一个logical index $[i, j]$，三步映射：
1. **Split** (unravel): $i \to [i_0, i_1, i_2]$, $j \to [j_0, j_1, j_2]$
2. **Distribute**: 选出某些sub-dims作为thread index ($[i_2, j_1]$)，其余作为local index ($[i_0, j_0, i_1, j_2]$)
3. **Merge** (ravel): 把multi-dim index线性化为linear thread index和linear local index

**重要性质**: 这个表示在Kronecker product下是闭合的，即两个这种形式的layout的product仍然是这种形式。这是layout algebra可分析性的基础。

## 4. Thread-Block-Level Programming Model

### 4.1 Type System
- **Scalar variables**: int32, float16等
- **Pointer variables**: memory address
- **Tensor variables**: 多维数组，包含4个属性：
  - dtype: element type
  - shape: tensor shape
  - memory scope: global / shared / register
  - layout: element到linear memory的映射

### 4.2 Program Structure (Figure 7)
```
program name <grid_shape>(parameters):
    body  # statements: if-else, for-loop, while-loop, instructions
```

保留high-level control flow statements (不lower到jump instructions)是为了human readability。

### 4.3 Instruction Set (Table 1)
关键指令类别：

**Indexing**: `BlockIndices()` - 获取当前thread block在grid中的indices

**Tensor Creation**:
- `AllocateGlobal(dtype, shape, [layout])`: 在global memory分配
- `AllocateShared(dtype, shape, [layout])`: 在shared memory分配
- `AllocateRegister(dtype, shape, [layout])`: 在register分配
- `ViewGlobal(ptr, [dtype], shape, [layout])`: 创建global memory的tensor view

**Tensor Transferring**:
- `LoadGlobal`, `LoadShared`, `StoreGlobal`, `StoreShared`
- `CopyAsync(shared_tensor, global_tensor)`: 异步copy
- `CopyAsyncCommitGroup()`, `CopyAsyncWaitGroup(n)`: 异步copy的commit和wait (用于software pipelining)

**Register Tensor Computation**:
- Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Neg`
- `Cast(a, dtype)`: 类型转换，保持layout
- `View(a, [dtype], [layout])`: **零成本** reinterpretation！这是Tilus的核心优势之一
- `Dot(a, b, c)`: $d = \text{dot}(a, b) + c$ (使用tensor core)

**关键洞察**: `View` instruction可以在register level重新解释tensor的dtype和layout，**without任何数据movement**。这只有在algebraic layout system精确刻画了bit分布后才能实现。

### 4.4 SIMB Execution Semantics
不同warps可以同时执行不同instructions。但如果有两条instruction access同一个shared/global memory region并且第二条依赖第一条的完成，必须插入`Synchronize()`。

## 5. Low-Precision Data Type Support

### 5.1 Compact Storage基本机制
现代处理器最小寻址单元是1 byte (8 bits)。低于8 bit的data需要compact storage (Figure 8)：
- 多个low-precision values紧凑打包在bytes中
- 单个value可能跨两个bytes (e.g., uint4的element b[1]在byte 0的高4 bits + byte 1的低4 bits)

**Load操作** (Figure 8b)：
1. Bitwise AND提取相关bits
2. Bitwise SHIFT调整位置
3. 如果跨bytes，用bitwise OR合并两部分

**Store操作** (Figure 8c)：
1. Bitwise mask清空目标bit位置
2. Bitwise OR插入新value，保留其他bits

这是**fallback机制**，效率不高，主要用于correctness。

### 5.2 Efficient Weight Loading (Section 7.2)

核心insight: 通过layout transformation在global memory中pre-arrange weight，让loading能用hardware-friendly instruction。

**Formalization**: 给定一个tensor，每thread $n$ bits，$T$ threads，reinterpret为：
$$\text{dtype: uint8, layout: local}(n_2) \otimes \text{spatial}(T) \otimes \text{local}(n_1)$$

其中：
- $n_1 = \gcd(n, 16)$: 每个thread内的byte数（用16是因为GPU memory transaction通常以16-byte为单位）
- $n_2 = n / \gcd(n_1, 16)$: 这个公式有点tricky，需要仔细分析

**例子** (Figure 9, int6 weight): 
- 原始tensor: `i6[K, N]`
- Tile化: 分成`[BK, BN]`大小的tiles
- Reinterpretation: `i6[BK, BN]` → `u8[BK * BN * 6 / 8]` (因为6 bits × BK × BN elements / 8 bits per byte = 字节数)
- Contiguous存储: 每个tile连续存储在global memory中

**关键效果**: 这样loading时可以用vectorized instructions (e.g., `cp.async.v4`, `ldg128`)，避免non-coalesced memory access。

### 5.3 Register Tensor Reinterpretation (Figure 2c)

回到int6的例子 (Line 11 in Figure 2a)：

**为什么reinterpretation是valid的？**
- 原始: `u8[16, 8]` with layout `local(3).spatial(32)` → 每thread 3 bytes = 24 bits
- 目标: `i6[16, 8]` with layout `local(2, 1).column_spatial(4, 8).local(2, 1)` → 每thread 4 × 6 = 24 bits

**关键**: 两种表示下，每thread的bit数相同 (24 bits)，所以可以零成本reinterpret！

这个 reinterpretation 是 Triton 完全做不到的——因为 Triton abstract掉了layout，无法验证两个 tensor 的 bit 分布是否兼容。

### 5.4 Efficient Vectorized Casting
从int6 cast到float16时，Tilus使用CUDA-specific instructions：
- **PRMT** (permute bytes in 32-bit register): 重新排列bytes
- **LOP3** (arbitrary logical operation on 3 inputs): 任意3-input的boolean function
- Bitwise instructions

所有操作在register内完成，**不需要thread间通信**。这是关键的性能优势——避免了shared memory round-trip。

## 6. Implementation细节

### 6.1 系统组成
35K lines Python + C++，包含5个component：
1. **DSL in Python**: 用户interface
2. **IR**: VM的intermediate representation
3. **Optimization passes**: 消除冗余、简化算术表达式
4. **Code generator**: 翻译IR到Hidet IR [11]
5. **Runtime system**: 管理动态loaded binaries

### 6.2 Compilation Pipeline (4 Steps)

**Step 1: Global and Shared Memory Planning**
- 计算shared memory需求
- Map shared tensor到kernel的shared memory region
- 管理global memory workspace (通过`AllocateGlobal` instruction)

**Step 2: Code Emitting for Each Instruction**
- 用Hidet IR表示低级GPU code
- **Instruction selection**: 
  - 默认: `lds` PTX instruction (shared → register)
  - 如果layout兼容`spatial(8, 4).repeat(1, 4)`: 用更高效的`ldmatrix` PTX instruction
- **Automatic vectorization**:
  - `cp.async.v4`: 异步copy 16 bytes
  - `lds128`: shared → register 128 bits
  - `ldg128`: global → register 128 bits

**Step 3: Lowering Low-Precision Data Types**
- 应用Section 7.1的rules
- 把所有low-precision operation transform成hardware-friendly type上的operation
- 大多数情况下只有vectorized type casting (e.g., int6 → float16)需要应用
- Memory loading已经被replaced为standard type loading (通过layout reinterpretation)

**Step 4: Loading by Runtime System**
- 动态loaded binary
- Runtime state:
  - Workspace memory (on-demand allocated)
  - Execution context (CUDA stream)
  - Kernels cached (avoid recompilation)

## 7. Experimental Evaluation深度分析

### 7.1 实验配置
- **Models**: Gemma-2-9B [12], QWen2.5-32B [13], Llama-3.3-70B [14]
- **Stages**: prefill (2048 tokens), decode (1 token, 16 tokens)
- **Baselines**: cuBLAS [15], Triton [6], Ladder [7], QuantLLM [10], Marlin [9]
- **Hardware**: NVIDIA L40S (48 GiB, Ada Lovelace), A100 (Ampere), H100 (Hopper)
- **Software**: vLLM v0.5.3, Triton v3.1.0, bitblas v0.0.1.dev15, Marlin v0.1.1
- **Protocol**: operator 50次运行，model 10次运行，取median latency，每次清L2 cache

### 7.2 Operator-Level Performance (Figure 10)

测试的data types: uint8, f6e3m2, int4, uint4, uint2, uint1

vs cuBLAS FP16 baseline的speedup:
- **vs Triton**: 1.75× average
- **vs Ladder**: 2.61× average  
- **vs QuantLLM**: 1.29× (QuantLLM只支持FP5/FP6)
- **vs Marlin**: 1.03× (Marlin只支持INT4)

**为什么Tilus能赢Marlin?** Marlin是hand-crafted INT4专用kernel，理论上已经接近optimal。Tilus能略微胜出的原因可能是：
1. 单一program template能auto-tune tile sizes
2. Software pipelining在Tilus中是first-class support
3. Register-level reinterpretation避免了shared memory round-trip

### 7.3 Full Spectrum Data Type Support (Figure 11)

测试matrix: rows = type kind (uint, int, float), columns = bit width (1-8)
- Unsigned integers: uint1-uint8
- Signed integers: int2-int8
- Floating-point: e4m3, e3m3, e3m2, e2m2, e2m1, e1m1 (float3-float8)

实验设置: BS=16, K=8192, N=57344 (对应Llama-70B的维度)

**关键数字**:
- ~200 configurations per operator
- ~1 minute compile time per operator
- float16 activation (也支持bfloat16, int8)

**Insight**: 这个figure展示了Tilus的"广谱"优势——传统approaches每个data type需要单独hand-craft kernel，而Tilus用**同一个parameterized program template**覆盖所有data types。

### 7.4 End-to-End Performance (Figure 12)

测试场景: prefill (2048 tokens), decode (1 token, 16 tokens)

**Decode stage (BS=1)**: 主要瓶颈是weight loading (memory bandwidth bound)
- Tilus的优势来自efficient weight loading pipeline (Figure 1c)

**Decode stage (BS=16)**: bottleneck转移到compute
- Tilus的优势来自software pipelining + k-dimension parallelization
- Ladder在这场景下suboptimal，因为它没有实现这两个optimization

**Prefill stage (2048 tokens)**: computation-bound
- 所有方法都把quantized weights decode到float16，用标准f16×f16 matmul
- Tilus的优势在于efficient quantized weight layout handling，使得decoding overhead最小

### 7.5 Hardware Ablation (Figure 13)

测试QWen2.5-30B在A100 (Ampere), L40S (Ada), H100 (Hopper)上的表现。

**关键观察**:
- H100上Ladder无法生成valid kernel (CUDA error "illegal instruction encountered")——这显示了Ladder的TVM-based approach对新architecture的适应性差
- L40S上vLLM (float16) OOM (48 GiB DRAM不够)——quantization是必须的
- Tilus在所有hardware configuration上都outperform Ladder

### 7.6 Batch Size Scaling (Figure 14)

测试BS: decode (1, 4, 8, 16), prefill (4096, 8192, 12288)
Model: Llama-3.3-70B
Data types: float6_e3m2, uint4
Dimensions: K=8192, N=57344

**观察**: Tilus在所有batch size下都outperform baselines。这很重要因为LLM serving的batch size会动态变化（continuous batching [16]）。

## 8. Related Work联想

### 8.1 Tile-Oriented Compilers的演进
- **Graphene** [17]: 最早引入tile IR with layout representation，但用strides representation，不如Tilus的algebraic system expressive
- **Hidet** [18]: Tilus的backend，task-mapping programming paradigm，但没有low-precision support
- **Triton** [6]: 最popular的tile-based compiler，但是layout abstraction使得它无法实现register-level reinterpretation

### 8.2 Layout System的对比
- **PartIR** [19]: Google的工作，引入tile-level layout system，但abstraction level比Tilus高，不涉及thread-level distribution
- **ExTensor** [20], **FuseMax** [21]: 用einsum定义computation，implicitly决定memory access pattern，但不specify thread-level distribution

### 8.3 任意bit-width quantization
- **APNN-TC** [22]: 支持arbitrary precision，但只限integer quantization，性能suboptimal
- **Any-Precision LLM** [23]: 部署多个不同size的LLM，但不是focus on kernel efficiency
- **ABQ-LLM** [24]: Arbitrary-bit quantized inference acceleration

### 8.4 Microscaling Data Types
**Microscaling formats (MX)** [25] 是AMD/Intel/NVIDIA等联合提出的standard，例如MXFP4 (一个shared exponent + 4-bit mantissa per element)。Tilus理论上可以通过新的lookup instruction支持——这是一个有价值的extension方向。

### 8.5 Codebook Quantization
**LCQ** [26] (Low-rank Codebook Quantization) 这种codebook-based quantization方法，Tilus可以通过添加新的lookup instruction支持。这比传统的bit-packing quantization更接近vector quantization。

## 9. 我的Intuition Building

### 9.1 为什么Algebraic Layout System是key?

Triton的核心限制不是性能问题，而是**abstraction level**问题。Triton abstract掉layout后，开发者完全无法控制register-level的bit分布。这意味着：
- 无法验证两个tensor的bit分布是否compatible for reinterpretation
- 无法直接emit `ldmatrix` instruction (需要特定layout)
- 无法实现register-level的layout转换 (只能通过shared memory round-trip)

Tilus的algebraic layout system把layout变成**first-class citizen**，并且通过Kronecker product提供了**compositional**的layout构造方式。这使得layout analysis成为可能——只有当你能分析两个tensor的bit分布是否compatible时，才能做zero-cost reinterpretation。

### 9.2 为什么Thread-Block-Level > Thread-Level?

CUDA的SIMT model要求开发者thinking at thread level。但是GPGPU programming的mental model实际上是"32个threads协作处理一个tile"。Triton和Tilus都把这个mental model变成了一等公民。

Tilus更进一步：暴露**hierarchical memory** (global, shared, register)。这听起来像是把复杂性抛给开发者，但实际上：
- 在low-precision scenario下，**memory layout决定了performance**
- 如果不暴露shared memory，就无法实现software pipelining (因为pipeline需要overlap global memory load和compute，这需要在shared memory中维持multiple buffers)
- 如果不暴露register layout，就无法做register-level reinterpretation

### 9.3 为什么1-8 bit是合理的range?

- 1 bit: extreme quantization (e.g., binary weights in XNOR-Net style)
- 2-3 bit: aggressive quantization research (e.g., QuIP [27]探索2-bit quantization)
- 4 bit: 当前production standard (Marlin, AWQ, GPTQ)
- 5-7 bit: accuracy preservation zone (eXmY [5]研究表明5/6/7-bit能preserve accuracy)
- 8 bit: standard int8 (cuBLAS, Triton原生支持)

>8 bit的需求实际上很少，因为GPU hardware对8-bit及以上有native support。

### 9.4 Marlin Comparison的微妙之处

Marlin (1.03× speedup，几乎平手) 的对比很微妙。Marlin是hand-crafted INT4的**optimal**实现，理论上compiler-generated code很难超过hand-tuned code。Tilus能打平甚至略微胜出的原因：

1. **Software pipelining**: Marlin的实现细节决定了它的pipeline stages，Tilus通过program template可以auto-tune pipeline depth
2. **Tile size flexibility**: Marlin的tile size是hardcoded的，Tilus可以针对不同batch size选择不同tile size
3. **Maintainability**: Marlin的代码非常复杂 (e.g., ~2000 lines CUDA)，修改难度大；Tilus的~50 lines program template更容易adapt新architecture

### 9.5 联想到CUDA 12.x的进展

CUDA 12.x引入了几个相关feature：
- **TMA (Tensor Memory Accelerator)** in Hopper: 类似CopyAsync但更efficient
- **Warp Group MMA**: 新的tensor core programming model
- **Distributed Shared Memory**: 跨SM的shared memory access

Tilus的VM-based design理论上可以extension支持这些feature——只需要添加新的VM instruction。这是Tilus相比hand-crafted kernel的长期优势。

### 9.6 与TVM/Ladder的哲学对比

TVM/Ladder的哲学是**schedule-oriented**: 把computation和schedule分离，用declarative scheduling primitives描述optimization。

Tilus的哲学是**procedure-oriented**: 用imperative programming model直接描述computation和memory management。

这两种approach的根本区别：
- Schedule-oriented (Ladder): 适合auto-tuning和search space exploration，但表达能力受限于primitives
- Procedure-oriented (Tilus): 表达能力强，可以描述任意的memory access pattern，但auto-tuning更难

Tilus通过"single template + parameterized tile sizes"的方式来mitigate auto-tuning的难度——这实际上是一个pragmatic的折中。

## 10. Open Questions & Future Directions

1. **Auto-tuning space**: Tilus的~200 configurations per operator是否足够？是否能用RL或Bayesian optimization自动探索更大空间？

2. **Fusion support**: 当前Tilus主要focus on单个matmul operator。能否支持operator fusion (e.g., matmul + activation + residual)?这需要更复杂的shared memory management。

3. **Dynamic shapes**: LLM serving中sequence length是dynamic的。Tilus的grid shape支持runtime expressions，但tile size selection如何适应dynamic shape？

4. **Multi-GPU support**: 当前Tilus是single-GPU。能否extension到multi-GPU scenario (e.g., tensor parallelism)?

5. **Non-NVIDIA platforms**: Tilus的backend是Hidet IR → CUDA C。能否port到AMD ROCm/HIP [28]或Intel SYCL?

6. **Sparse computation**: Tilus目前focus on dense matmul。能否extension到sparse matrix (e.g., 2:4 structured sparsity)?

7. **Speculative decoding**: LLM serving的speculative decoding需要quick verification of draft tokens。Tilus的efficient low-precision kernel能否加速verification phase?

## 11. 参考资源

- [Tilus GitHub Repository](https://github.com/NVIDIA/tilus) - 论文中的开源链接
- [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/10.1145/3315508.3329973) - Triton原始论文
- [Ladder: Enabling Efficient Low-Precision Deep Learning Computing](https://www.usenix.org/conference/osdi24/presentation/wang-lei) - Ladder OSDI'24 paper
- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference](https://arxiv.org/abs/2408.11743) - Marlin论文
- [QuantLLM: Accelerating the Serving of LLMs via FP6-Centric Co-Design](https://www.usenix.org/conference/atc24/presentation/xia) - QuantLLM USENIX ATC'24
- [Hidet: Task-Mapping Programming Paradigm](https://dl.acm.org/doi/10.1145/3575693.3575702) - Tilus的backend
- [TVM: An Automated End-to-End Optimizing Compiler](https://dl.acm.org/doi/10.5555/3291178.3291201) - Ladder的基础
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) - PTX指令集参考
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - CUDA编程指南
- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180) - vLLM论文
- [eXmY: Arbitrary Bit Precision Quantization](https://arxiv.org/abs/2405.13938) - 任意精度quantization研究
- [QuaRot: Outlier-Free 4-Bit Inference](https://openreview.net/forum?id=dfqsW38v1X) - 4-bit quantization state-of-the-art
- [QuIP#: Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) - 2-bit quantization with codebooks
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) - 4-bit weight quantization
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) - 4-bit post-training quantization
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) - MX data types spec
- [PartIR: Composing SPMD Partitioning Strategies](https://dl.acm.org/doi/10.1145/3669940.3707284) - PartIR layout system
- [Graphene: An IR for Optimized Tensor Computations](https://dl.acm.org/doi/10.1145/3582016.3582018) - 早期tile IR with layout
- [Stream-K: Work-Centric Parallel Decomposition](https://dl.acm.org/doi/10.1145/3572848.3577479) - K-dimension parallelization
- [AlCoP: Automatic Load-Compute Pipelining](https://proceedings.mlsys.org/paper/2023/hash/680-694) - Software pipelining研究

---

## 总结

Tilus的核心contribution是把**layout变成algebraic first-class citizen**，使得register-level reinterpretation成为可能。这个看似简单的abstraction choice，使得一个**single program template**能cover从uint1到float8的全spectrum quantized matmul，并且performance能match甚至beat hand-crafted kernels。

从system design的角度，Tilus做了一个有争议但pragmatic的选择：暴露GPU memory hierarchy的复杂性，换取low-precision kernel的极致performance。这跟Triton的"abstract away everything"哲学形成鲜明对比。这个选择的代价是programming complexity增加，但对于LLM serving这种latency-critical scenario，这个tradeoff是值得的。

长期来看，Tilus的VM-based architecture可能比hand-crafted kernel更有生命力——新GPU architecture (e.g., Blackwell的FP4 tensor core) 可以通过添加新VM instruction支持，而不需要rewrite整个kernel。这是compiler-based approach的核心价值。
