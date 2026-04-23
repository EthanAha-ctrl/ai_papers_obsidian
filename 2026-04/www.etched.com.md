基于第一性原理，我们来深度解构 **Etched** (www.etched.com) 这家公司以及其核心产品 **Sohu** 芯片。

因为当前 AI 领域的 dominant architecture 是 Transformer，所以 Etched 采取了极其激进的策略：将 Transformer 硬连线到硅片中，制造出纯粹的 Transformer ASIC（Application-Specific Integrated Circuit）。这种 "All-in" 的策略类似于将赌注刻在硅片上，这也是公司名字 "Etched" 的由来。

---

### 1. 第一性原理拆解：为什么需要 Etched？

从第一性原理出发，AI 推理的计算本质是：**数据移动** 与 **矩阵运算**。

*   **GPU 的困境**：NVIDIA H100 等 General-purpose GPU 为了兼容 CNN、RNN、Graphics rendering 等各种 workload，在芯片上保留了大量的 flexible logic（例如 CUDA cores、复杂的缓存层级、路由逻辑）。从物理学角度看，硅片面积就是算力，flexibility 意味着 silicon waste。
*   **Transformer 的确定性**：目前 LLM 的计算图是高度确定的，无论是 GPT-4、Llama-3 还是 Gemini，其核心数学操作 100% 都是 $\text{Attention}(Q,K,V)$ 和 $\text{FFN}(x)$。
*   **结论**：因此，如果 workload 已经收敛到 Transformer，剥离所有非 Transformer 的控制逻辑，把省下来的 silicon area 全部换成 Tensor Cores 和 SRAM，就能在物理极限上实现最高 的 FLOPS/$ 和 Tokens/watt。

---

### 2. 核心技术讲解：Sohu 芯片的架构解析

Sohu 并非简单的 "砍掉功能的 GPU"，其架构设计围绕 Transformer 的内存墙与计算墙进行了重构。

#### A. 公式到硬件的映射
Transformer 的核心计算是 Scaled Dot-Product Attention (SDPA)：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

*   $Q \in \mathbb{R}^{N \times d_k}$: Query matrix
*   $K \in \mathbb{R}^{N \times d_k}$: Key matrix
*   $V \in \mathbb{R}^{N \times d_v}$: Value matrix
*   $N$: Sequence length
*   $d_k, d_v$: Dimension of key/value vectors
*   $T$: Transpose operation
*   $\sqrt{d_k}$: Scaling factor (防止内积过大导致 softmax 梯度消失)

在 Sohu 的硬件执行逻辑中：
1.  **$QK^T$ 阶段**：这是典型的 Batched GEMM（通用矩阵乘法）。Sohu 配备了极高密度的 MAC（Multiply-Accumulate）阵列，因为 $N$ 通常很大（如 128k context length），这属于 **Compute-bound**。Sohu 将几乎所有 silicon 倾注于提升这部分的 FLOPS。
2.  **$\text{softmax}$ 阶段**：这需要对 row 做 reduce 操作。在 GPU 上，这会产生大量的 shared memory 写入与同步开销。Sohu 采用了定制的 **Online Softmax** 硬件单元，在 MAC 阵列内部直接流水线式完成 max 值追踪与归一化，无需将中间结果写回 SRAM。
3.  **$\times V$ 阶段**：同样是 GEMM，但受限于内存带宽。

#### B. 极致的 Memory Subsystem
因为 LLM 推理的 Autoregressive decoding 阶段是极度 **Memory-bound**（每生成一个 token 都需要读取整个 model weight 和 KV cache），所以 Sohu 的架构图解析如下：

*   **HBM3E 绑定**：Sohu 堆叠了巨量的 HBM3E（估计单芯片 > 4TB/s bandwidth），因为推理速度 tokens/s 直接正比于内存带宽。
*   **Massive SRAM for KV Cache**：Etched 意识到 KV Cache 的频繁搬移是瓶颈，因此在 Sohu 芯片上集成了远超 H100 的分布式 SRAM Pool，直接在计算单元旁缓存 $K, V$ 矩阵，避免了 "HBM -> SRAM -> Register" 的漫长延迟。
*   **硬连线的 Dataflow**：去掉了 GPU 的 Instruction Cache 和 Decoder，数据流路径直接在硅片上物理布线（$QK^T \rightarrow \text{Softmax} \rightarrow \text{Multiply V} \rightarrow \text{FFN}$）。因为没有指令获取和译码的开销，芯片的 Utilization rate 在运行 Transformer 时远超 GPU。

---

### 3. 实验数据与性能推演

基于 Etched 官方白皮书与行业推演，以下是 Sohu 与 NVIDIA H100 的架构与性能对比表：

| Specification | NVIDIA H100 SXM | Etched Sohu (Est. / Claimed) | Analysis |
| :--- | :--- | :--- | :--- |
| **Architecture Type** | General Purpose GPU | Pure Transformer ASIC | Sohu 剥离了 Graphics, CUDA cores 等，silicon 效率极高 |
| **Process Node** | TSMC 4N | TSMC 4N | 物理制程相同，差异纯粹来自架构 |
| **Memory Bandwidth** | 3.35 TB/s | > 4.8 TB/s (HBM3E) | 因为 decoding 是 memory-bound，带宽直接决定 tokens/s |
| **Transformer FLOPs (FP8/BF16)**| ~1979 TFLOPS (Dense) | > 800,000 TFLOPS (Cluster level) | Sohu 在 single chip 上的 tensor core 密度更高 |
| **KV Cache Capacity per chip**| ~30-40 MB (SRAM) | >> 100 MB (Custom SRAM) | 决定了单卡能支撑的 max concurrent sequences |
| **Software Stack** | CUDA (Flexible) | Transformers over Sohu (Hardcoded) | Sohu 只能跑 Transformer，无法跑 CNN 或其他架构 |

*Intuition Builder*: 想象 H100 是一把瑞士军刀，可以切菜、开瓶、削木头，但切菜的效率比不上专业主厨刀。Sohu 就是一把只为切 Transformer 这种蔬菜打造的专业刀具，没有刀背、没有开瓶器，所有重量都分布在刀刃上。

---

### 4. 生态与未来推演

虽然 Sohu 在物理指标上极具破坏力，但从第一性原理看，其面临的风险也是系统性的：

1.  **Software Ecosystem Lock-in**：NVIDIA 的护城河是 CUDA。Etched 必须提供一套让 PyTorch/HuggingFace 模型能无缝转换的 compiler。目前 Etched 采用将 popular architectures（如 Llama, GPT, Mixtral）直接编译成硬件微码的策略。
2.  **Architecture Shift Risk**：如果未来 SSM（State Space Models 如 Mamba）、RWKV 或新的 non-attention 架构成为主流，Sohu 芯片上的 Softmax 硬件单元将彻底变成死硅。然而，因为目前整个 AI 工业界从 OpenAI 到 Meta 都在 Transformer 生态上投入了万亿美元，Etched 赌的就是 Transformer 在未来 5-10 年内仍是主导。
3.  **MoE (Mixture of Experts) 优化**：Sohu 的硬连线架构对 MoE 特别友好。MoE 的 routing 机制在 GPU 上会导致极其不规则的 memory access pattern，而 Sohu 可以通过定制的 routing network 直接在 SRAM 内完成 expert 权重的低延迟调度。

---

### References & Further Reading

1.  Etched Official Website & Whitepaper: [https://www.etched.com/](https://www.etched.com/)
2.  Etched Sohu Announce Blog: [https://www.etched.com/blog/announcing-sohu](https://www.etched.com/blog/announcing-sohu)
3.  Transformer Hardware Architecture Analysis (Microbenchmarking H100): [https://arxiv.org/abs/2403.07949](https://arxiv.org/abs/2403.07949)
4.  First-principles thinking on AI ASICs (Graphcore, Groq comparisons): [https://www.hotchips.org/](https://www.hotchips.org/) (Industry conference proceedings on dedicated AI silicon)

总结而言，Etched 是当前 AI 芯片同质化竞争中一个极其纯粹的异类。它放弃了通用性，将 Transformer 的数学公式直接刻入物理世界，是对 "Software is eating the world, but hardware dictates what software can eat" 这句话的终极诠释。