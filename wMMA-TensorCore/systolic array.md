https://github.com/PSCLab-ASU/Systolic-CNN
https://github.com/sun-jay/FPGA-Hardware-NN-Accelerator
https://github.com/bsc-loca/sauria
https://github.com/spcl/gemm_hls

下面用中文解释 **什么是 systolic array**（有节奏阵列），其原理、特点、优势、适用场景等。

---

## 什么是 **systolic array**

**systolic array** 是一种并行计算硬件架构，由大量 **Processing Element (PE)** 以规则网格连接组成。数据像心脏跳动一样按节奏（systolic）在这些 **PE** 之间流动、计算、传递，从而实现高吞吐量的矩阵运算。

## 核心思想

- 通过 **local data movement**（局部数据移动）替代全局访问。
    
- 把大规模计算分解为许多小的、重复的 **PE** 执行单元。
    
- 按照固定的数据流模式跨 **PE** 传递数据。
    

这使得计算和数据传输的节奏高度一致（systolic），类似心脏脉动。

---

## 结构示意

```
Input A → PE → PE → PE → … → Output A
           ↓    ↓    ↓
Input B → PE → PE → PE → … → Output B
```

- 每个 **PE** 接收来自上游、左侧的数据。
    
- 执行简单乘加（MAC）或加法等操作。
    
- 把结果传给下游、右侧的 **PE**。
    
- 数据同步流过整个阵列。
    

---

## 关键特点

### 1. **数据流驱动（dataflow）**

数据不是被 **pull**（按需取）进来计算，而是**push**（按固定节奏流过阵列）驱动计算。

### 2. **局部通信**

PE 只和相邻的 PE 交换数据，避免了对远端存储器的频繁访问，从而降低了功耗和延迟。

### 3. **高吞吐量**

当阵列规模较大时，可实现非常高的并行计算能力，尤其适合矩阵乘法、卷积等计算密集型任务。

### 4. **可预测延迟**

因数据流动是固定模式，延迟可预估，有利于硬件时序分析。

---

## 原理说明

以 **矩阵乘法 C = A×B** 为例：

- 把矩阵 A 的行和矩阵 B 的列分别送入阵列。
    
- 在不同时间步，PE 接收 A 的一个元素和 B 的一个元素。
    
- 每个 PE 累加部分乘积到自己的 **accumulator**。
    
- 最终输出对应的 C 元素。
    

这样 **每个 PE** 贡献一个结果，并且所有乘加都在阵列内部完成。

---

## 优势

优点包括：

- **高性能：** 并行执行多个乘加单元。
    
- **低功耗：** 局部通信替代远端加载。
    
- **可扩展性：** 阵列规模可调整以平衡资源和性能。
    
- **专用加速：** 对于矩阵运算、卷积等非常高效。
    

---

## 局限与挑战

欠缺灵活性：

- 仅适合结构规律、可数据流表达的运算。
    
- 对控制逻辑要求较高。
    
- 对外存带宽仍有压力（尤其是输入/输出数据流）。
    

---

## 与你提到的项目关系

|项目仓库|关联内容|
|---|---|
|`PSCLab-ASU/Systolic-CNN`|基于 **systolic array** 实现卷积神经网络加速|
|`sun-jay/FPGA-Hardware-NN-Accelerator`|FPGA 上的神经网络硬件加速器，常用 systolic 或类似结构|
|`bsc-loca/sauria`|通用加速架构，可能包含数据流阵列思想|
|`spcl/gemm_hls`|用于生成高效矩阵乘法（GEMM）的 HLS 库，底层数据流/阵列技术|

---

## 什么时候用 **systolic array**

适合以下场景：

- **矩阵乘法（GEMM）**
    
- **卷积神经网络（CNN）**
    
- **信号处理**
    
- **线性代数密集计算**
    

总结来说，**systolic array 是一种以数据流和局部通信为核心的高效并行硬件架构**，在加速矩阵运算和神经网络推理中非常常见。

下面针对你提出的 **三个具体问题**，从 **hardware micro-architecture / dataflow / memory hierarchy** 角度逐条回答，以 **systolic array** 的典型实现为参考（FPGA / ASIC 均适用）。

---

## 1. **PE 和 PE 之间是否有 stall signal？后面的 PE 怎么知道前面的 PE 计算好了？**

### 简短结论

**经典 systolic array 设计中：**

- **通常没有显式 stall / ready-valid handshake**
    
- **PE 之间依靠全局时钟 + 固定 pipeline latency**
    
- 数据是否“计算完成”是 **implicit 的（时间决定）**
    

---

### 详细说明

#### （1）时间驱动，而不是事件驱动

systolic array 是 **time-scheduled dataflow architecture**：

- 每个 **PE** 的行为在 **cycle level** 是静态已知的
    
- 数据在 **每一个 cycle** 沿固定方向流动
    
- 计算完成的“通知”不是 signal，而是 **时序本身**
    

即：

> 后面的 PE **不需要知道** 前面的 PE “算完了”，  
> 因为 **设计时就保证在某个 cycle 数据一定是 valid 的**

---

#### （2）典型 PE pipeline

一个常见 PE 包含：

- input register（A / B）
    
- multiplier
    
- adder + accumulator
    
- output register
    

假设：

- MAC latency = 1 cycle
    
- 数据每 cycle 推进一次
    

那么：

- cycle t：PE(i,j) 接收 A(t), B(t)
    
- cycle t+1：PE(i,j+1) / PE(i+1,j) 接收到结果
    

**不需要 stall signal**。

---

#### （3）什么时候会有 stall / backpressure？

在以下“非理想”或“增强型”设计中，可能存在 stall：

|场景|是否有 stall|
|---|---|
|纯 2D systolic GEMM|通常没有|
|可变 shape / sparsity|可能有|
|shared PE / time-multiplex|可能有|
|NoC-like systolic|有 ready/valid|

但注意：  
**一旦引入 ready/valid，就开始偏离“纯 systolic”设计哲学**。

---

## 2. **intermediate data 存在哪里？SRAM 吗？地址如何划分？software 还是 hardware？**

### 简短结论

- **绝大多数 intermediate data 不进 SRAM**
    
- **intermediate data 存在 PE 内部 register / accumulator**
    
- SRAM 主要用于：
    
    - input tile buffer
        
    - weight buffer
        
    - output tile buffer
        
- 地址划分是 **software compiler 决定，hardware 固化执行**
    

---

### 分层来看 memory hierarchy

```
DRAM
 ↓
Global SRAM (tile buffer)
 ↓
PE local register / accumulator
```

---

### （1）intermediate data 在哪里？

|数据类型|存储位置|
|---|---|
|partial sum (psum)|**PE local accumulator register**|
|A/B stream|PE input register|
|final output tile|SRAM / DRAM|

**关键点：**

> systolic array 的核心优势之一  
> 就是 **psum 不回写 SRAM**

---

### （2）SRAM 如何使用？

SRAM 通常被组织为：

- input activation buffer
    
- weight buffer
    
- output buffer
    

特点：

- 大块
    
- 顺序访问
    
- banked / multi-port
    

---

### （3）SRAM 地址由谁决定？

#### 在现代 accelerator 中：

|层级|职责|
|---|---|
|software compiler|tile size / mapping / loop order|
|hardware controller|address generation FSM|
|PE|完全 unaware（只看到 stream）|

也就是说：

- **software** 决定：
    
    - 如何 tiling
        
    - 哪个 tile 进 systolic array
        
- **hardware**：
    
    - 根据预定义 address pattern 生成地址
        
- **PE**：
    
    - 只接收 stream，不知道 address
        

这是 **software-defined, hardware-executed** 的典型范式。

---

## 3. **相比 non-systolic（如 von Neumann + register file），好在哪里？SRAM 利用率更高吗？**

这是一个**非常关键的问题**。

### 简短结论

systolic array 的优势 **不在 SRAM 容量利用率本身**，而在于：

- **SRAM access efficiency**
    
- **data reuse distance**
    
- **energy per MAC**
    

---

### 对比分析

#### （1）von Neumann / SIMD + register file

特点：

- central register file
    
- broadcast / read/write 多端口
    
- psum 反复 load/store
    

问题：

- register file 面积大
    
- multi-port SRAM 功耗极高
    
- psum traffic 爆炸
    

---

#### （2）systolic array

特点：

- PE local accumulator
    
- neighbor-to-neighbor forwarding
    
- psum 永不出阵列
    

---

### SRAM 利用率的真实对比

|指标|von Neumann|systolic|
|---|---|---|
|SRAM 容量利用率|相近|相近|
|SRAM **访问次数**|高|**极低**|
|psum SRAM traffic|高|**接近 0**|
|energy / MAC|高|**低一个数量级**|

**结论：**

> systolic array 不是“更省 SRAM 空间”，  
> 而是 **更少用 SRAM**

---

### 本质优势总结

systolic array 真正赢在：

1. **temporal locality → spatial locality**
    
2. **register > SRAM > DRAM 的能耗差异**
    
3. **把 memory hierarchy 压缩到 PE 内**
    

Google TPU 的经典结论是：

> Moving data costs orders of magnitude more energy than computing.

systolic array 正是为此而生。

---

## 总结一句话

- **Q1：** 没有 stall，靠 cycle-level schedule
    
- **Q2：** intermediate data 在 PE register，SRAM 由 compiler 决定、hardware 执行
    
- **Q3：** 不是 SRAM 用得更“满”，而是 **SRAM 用得更“少、更聪明”**
    

如果你愿意，下一步可以：

- 画一个 **cycle-accurate systolic GEMM timing diagram**
    
- 对比 **output-stationary / weight-stationary / row-stationary**
    
- 结合你给的某个 repo 具体拆一段 RTL / HLS 代码


---

### 核心点

1. **数据尽量“就地”累积**
    
    - 每个 PE 内部有 **local accumulator register**
        
    - partial sum (psum) 从不写回 SRAM，直接在 PE 内累加
        
    - 因此消除了大量 psum 的 SRAM 访问
        
2. **neighbor-to-neighbor forwarding**
    
    - A / B 输入在 PE 间以 **shift register / pipeline register** 形式流动
        
    - 数据在 PE 之间传递，而不是回到全局 SRAM
        
    - 这让数据“离 ALU 更近”，降低延迟和能耗
        
3. **减少 SRAM access = 节省能耗**
    
    - SRAM 或 DRAM 的访问能耗远大于 ALU register
        
    - systolic array 利用寄存器（靠近 ALU）存储临时数据，从而提高 **energy efficiency**
        

---

### 简化理解

可以把 systolic array 想象成：

```
ALU + register = 小型 local memory
```

- 数据在 **小型 local memory** 中流动和累积
    
- **全局 SRAM 只用于输入 tile 和输出 tile**
    
- 因此访问次数极少 → 能效高
    

---

如果你需要，我可以画一个 **PE 内部寄存器 + SRAM 数据流示意图**，直观展示 systolic array 是如何“用 register 替代 SRAM”的。

你希望我画吗？

完全可以这么理解，把 **systolic array 当作以 PE 为元素的宏观 pipeline（macro pipeline）** 是一个非常精确的比喻。可以从几个维度来看：

---

## 1. Pipeline 视角

|层级|类比|
|---|---|
|**单个 PE 内部**|Micro-pipeline：输入寄存器 → ALU → accumulator → 输出寄存器|
|**整个 array**|Macro-pipeline：数据在多个 PE 之间流动，每个 cycle 前进一格，类似流水线推进|
|**数据流**|stage-by-stage（类似 instruction pipeline 的 stage），每个 stage 做一部分计算|

特点：

- 每个 PE 是一个 **pipeline stage**
    
- 每个 cycle，整个 array 同时计算不同的 “stage 的数据”
    
- **后续 PE 不用 stall**，只要按照固定 cycle 排布数据就行
    

---

## 2. Pipeline 优势

1. **高吞吐量**
    
    - 只要 array 被填满，每个 cycle 就有一个 output 出来
        
    - 类似 instruction pipeline 的 throughput 优化
        
2. **数据局部化**
    
    - psum 保存在 PE 内部 register，不回 SRAM
        
    - 只在 macro-pipeline 前端/后端与 SRAM 交互
        
3. **可预测延迟**
    
    - 输出延迟 = array 深度 × cycle 时间
        
    - 类似 pipeline 的 latency 公式
        

---

## 3. 典型类比

- **传统 pipeline（CPU ALU pipeline）**：instruction 在每个 stage 流动
    
- **systolic array**：matrix element / partial sum 在每个 PE 流动
    
- **区别**：
    
    - CPU pipeline stage 更复杂（控制 + forwarding + hazard）
        
    - Systolic array stage 更“纯粹”，只做固定计算，控制极少
        

---

如果要画一张示意图，可以这样想：

```
Time → 
PE0 --- PE1 --- PE2 --- PE3
 A*B    +累加   +累加   输出
```

- 每个 PE 是 pipeline stage
    
- 数据像波浪一样在 macro pipeline 中流动
    

---

所以，你理解成 **“PE 级的 macro pipeline”** 非常贴切，而且抓住了 **systolic array 核心设计哲学**：

> 数据尽量在 PE 内部累积 + 顺序流动 → 高吞吐 + 低能耗

---

如果你想，我可以画一个 **macro pipeline timing diagram**，把 **寄存器流 + psum 累积 + SRAM 访问** 一起展示，直观感受 systolic array 的 pipeline 特性。

你希望我画吗？