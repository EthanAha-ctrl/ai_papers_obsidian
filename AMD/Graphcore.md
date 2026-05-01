**2. Massively Parallel Processing**
- GC200 IPU 拥有 **1,472 processing tiles**
- 每个 tile 包含：
  - **Independent Cores**：可自主执行 computation
  - **SRAM**：每个 tile 拥有 624KB 的 **In-Processor Memory** (总计 900+ MB on-chip SRAM)

$$ \text{Total On-Chip Memory per IPU} = 1472 \text{ tiles} \times 624 \text{KB/tile} \approx 900 \text{ MB} $$

**3. All-to-All Interconnect**
IPU 内部采用 **fully connected communication fabric**，任意 tile 之间可以以 **GB/s** 级别的 bandwidth 直接通信，避免传统显存访问的 **bottleneck**

**4. 3D Wafer-on-Wafer (WoW) Stacking [https://www.nextplatform.com/2022/03/03/graphcore-goes-3d-with-ai-chips-architects-10-exaflops-ultra-intelligent-machine/](https://www.nextplatform.com/2022/03/03/graphcore-goes-3d-with-ai-chips-architects-10-exaflops-ultra-intelligent-machine/)**

第三代 **Bow IPU** 与 **TSMC** 合作，采用 **Wafer-on-Wafer** 3D 堆叠技术：
- 将两个硅晶圆垂直堆叠
- 通过 **Through-Silicon Vias (TSVs)** 实现垂直 electrical connections
- 相比传统 2.5D **CoWoS (Chip-on-Wafer-on-Substrate)** 封装：
  - 更低的 **power consumption**
  - 更高的 **bandwidth density** (40% improvement)
  - 更低的 **latency**

### IPU Product Line

| Generation | Chip | Process | Cores | Compute | Memory | Technology |
|------------|------|---------|-------|---------|--------|------------|
| 1st Gen | GC2 | 16nm | - | - | - | Baseline |
| 2nd Gen | GC200 | 7nm | 1,472 | ~250 TOPS | 900MB SRAM | Colossus MK2 |
| 3rd Gen | Bow GC200 | 7nm WoW | 1,472 | ~350 TOPS (40%↑ perf) | 900MB SRAM | Wafer-on-Wafer 3D |

$$ \text{FP16 Performance}_{\text{Bow}} = 350 \text{ TFLOPS} \quad \text{(estimated)} $$

---

## IPU vs GPU 架构差异对比 [https://www.eetasia.com/graphcore-ipu-vs-nvidia-gpus-how-theyre-different/](https://www.eetasia.com/graphcore-ipu-vs-nvidia-gpus-how-theyre-different/)

| Feature | IPU (Graphcore) | GPU (NVIDIA) |
|---------|-----------------|--------------|
| **Parallelism Model** | MIMD (Independent threads) | SIMT (Warps, lock-step) |
| **Memory Architecture** | Distributed on-chip SRAM (~1GB) | Centralized HBM/VRAM (40-80GB) |
| **Access Pattern** | Local SRAM preferred, low latency | High bandwidth to off-chip DRAM |
| **Best Workload** | Sparse graphs, dynamic shapes, small batch | Dense matrices, large batch, CNN |
| **Sparsity Support** | Native hardware support | Software emulated/limited |
| **Compile-time Graph** | Static graph compilation (Poplar) | Runtime JIT (CUDA Graphs) |

### 性能比较数据 (基于 Graphcore IPU-M2000 与 NVIDIA DGX-A100) [https://www.electronicsweekly.com/news/business/graphcore-benchmarks-outperform-nvidia-2020-12/](https://www.electronicsweekly.com/news/business/graphcore-benchmarks-outperform-nvidia-2020-12/):

| Model/Framework | Graphcore IPU-M2000 | NVIDIA A100 | Speedup |
|---------------|--------------------|-------------|---------|
| EfficientNet-B0 PyTorch | 2,043 images/s | 1,044 images/s | **1.96x** |
| BERT-Large Phase 1 | 3,402 sequences/s | 2,520 seq/s | **1.35x** |
| ResNet-50 PopART | 21,000+ images/s | 19,000+ images/s | **1.1x** |
| EfficientNet-B4 PyTorch | 391 images/s | 156 images/s | **2.5x** |

$$ \text{Speedup}_{\text{EfficientNet-B4}} = \frac{391}{156} \approx 2.5 \text{ times} $$

---

## 软件栈: Poplar SDK [https://www.graphpicore.ai/products/poplar](https://www.graphpicore.ai/products/poplar)

Poplar 是 Graphcore 的专用 software stack，与 IPU 硬件 **co-designed**：

### 四层架构

```
┌─────────────────────────────────────────────┐
│  Layer 4: ML Frameworks                     │
│  TensorFlow, PyTorch, JAX, PaddlePaddle    │
└──────────────────┬──────────────────────────┘
                   │ Graph Lowering
┌──────────────────▼──────────────────────────┐
│  Layer 3: Poplar ML Framework Integration   │
│  Poplar-TensorFlow, PopTorch, JAX-IPU,     │
│  PopART (ONNX runtime)                      │
└──────────────────┬──────────────────────────┘
                   │ Graph Compilation
┌──────────────────▼──────────────────────────┐
│  Layer 2: Poplar C++ API                    │
│  Direct hardware control, custom kernels    │
│  Tile mapping, communication primitives     │
└──────────────────┬──────────────────────────┘
                   │ Instruction Generation
┌──────────────────▼──────────────────────────┐
│  Layer 1: IPU Hardware                      │
│  1,472 tiles, 900MB SRAM, Exchange fabric  │
└─────────────────────────────────────────────┘
```

### PopART (Poplar Advanced Runtime) [https://docs.eidf.ac.uk/services/graphcore/training/L4_other_frameworks/](https://docs.eidf.ac.uk/services/graphcore/training/L4_other_frameworks/)

支持 **ONNX** (Open Neural Network Exchange) 格式的 import 与 execution：

$$ \text{Model} \xrightarrow{\text{ONNX Export}} \text{ONNX Graph} \xrightarrow{\text{PopART}} \text{Optimized IPU Binary} $$

**Execution Modes**:
- **Inference Mode**: 纯 forward pass，用于 production deployment
- **Evaluation Mode**: forward + metric computation，无 gradient
- **Training Mode**: 完整 forward + backward + optimizer step

---

## 技术专长: Sparse Computation

Graphcore 的一个核心优势是 **sparse computation**。 [https://www.graphcore.ai/posts/graphcore-and-aleph-alpha-demonstrate-80-sparsified-ai-model](https://www.graphcore.ai/posts/graphcore-and-aleph-alpha-demonstrate-80-sparsified-ai-model)

### 与 Aleph Alpha 的合作案例

将 **13 billion parameters** 的 chatbot model **sparsified** 到仅 **2.6 billion parameters** (80% sparsity)，而保持 accuracy：

| Model | Original Parameters | Sparsified | Reduction | Hardware Efficiency |
|-------|---------------------|------------|-----------|-------------------|
| Aleph Alpha Chatbot | 13B | 2.6B (80% sparsity) | 5x | IPU native support |

**Sparse Matrix Representation**:
- IPU 的 local SRAM 结构与 fine-grained parallelism 特别适合 **sparse matrix multiplication**
- Graph 结构中的 **adjacency matrix** 通常是稀疏的，IPU 的 **MIMD** 架构可以跳过 zero entries，而 **GPU** 仍需进行无效 computation

---

## 商业模式与竞争格局

### 产品形态

1. **IPU-M2000** [https://www.electronicsweekly.com/news/business/graphcore-benchmarks-outperform-nvidia-2020-12/](https://www.electronicsweekly.com/news/business/graphcore-benchmarks-outperform-nvidia-2020-12/)
   - 4x GC200 IPU 芯片
   - 1 petaFLOPS of AI compute (FP16)
   - 高达 3.6GB 的 In-Processor Memory
   - 支持 up to 8 cards interconnected

2. **IPU-POD64**
   - 64 个 IPU 大规模集群
   - 16 petaFLOPS AI compute
   - 用于 data center scale AI training

3. **Bow Pod** (第三代)
   - 采用 **Wafer-on-Wafer** 3D 堆叠技术
   - 40% performance improvement over IPU-M2000
   - 350+ TFLOPS per chip (estimated)

### 与 NVIDIA 的竞争 [https://www.eetasia.com/graphcore-ipu-vs-nvidia-gpus-how-theyre-different/](https://www.eetasia.com/graphcore-ipu-vs-nvidia-gpus-how-theyre-different/)

| Aspect | Graphcore IPU | NVIDIA GPU (CUDA) |
|--------|--------------|-------------------|
| **Programming Model** | Graph-based, static compilation | CUDA kernels, runtime JIT |
| **Memory Model** | Distributed SRAM, explicit management | Unified memory, caching |
| **Best Use Cases** | NLP, Graph Neural Networks, Sparse models | CNN, Computer vision, Dense linear algebra |
| **Software Maturity** | Emerging (Poplar, PopTorch) | Mature (CUDA, cuDNN, TensorRT) |
| **Market Share** | Niche, cloud partnerships | Dominant (>80% AI training market) |

### 为什么 Graphcore 选择不同的架构？

**核心洞察**: AI computation 的未来是 **sparse, dynamic, 和 graph-structured**，而非密集矩阵操作。

**GPU 的局限**:
- **Von Neumann bottleneck**: GPU 采用 **external HBM (High Bandwidth Memory)**，data 需通过 memory bus 传输，latency 高
- **SIMT divergence**: 当 parallel threads 走不同 code path 时，GPU 需 serial execution (branch divergence penalty)
- **Sparse inefficiency**: sparse matrix 在 GPU 上需转换为 dense 格式或使用复杂 indexing，效率低

**IPU 的优势**:
- **Data locality**: 900MB 分布式 on-chip SRAM，避免频繁 memory access
- **MIMD flexibility**: 每个 tile 独立 execution，无 divergence penalty
- **Graph native**: Graph computation (如 GNN) 可直接 map 到 hardware tiles，每个 node 对应一个 tile

---

## 最新动态 (2024-2025)

### SoftBank 收购后的战略转变 [https://techcrunch.com/2024/07/11/softbank-acquires-uk-ai-chipmaker-graphcore/](https://techcrunch.com/2024/07/11/softbank-acquires-uk-ai-chipmaker-graphcore/)

被 SoftBank 收购后，Graphcore 获得了资金支持来扩大生产和市场 reach：

1. **India Expansion** [https://www.graphcore.ai/posts/graphcore-to-invest-1bn-in-india-creating-500-semiconductor-jobs](https://www.graphcore.ai/posts/graphcore-to-invest-1bn-in-india-creating-500-semiconductor-jobs):
   - 在 Bengaluru 建立 AI Engineering Campus
   - 10 年投资 £1 billion ($10亿英镑 ≈ $13亿美元)
   - 创造 500 个 semiconductor 工作岗位
   - 战略意义：利用 India 的工程人才，同时接近 Asia-Pacific 市场

2. **Europe Research Project** [https://www.graphpcore.ai/posts/graphcore-advances-sparse-compute-in-new-eu-research-project](https://www.graphpcore.ai/posts/graphcore-advances-sparse-compute-in-new-eu-research-project):
   - 参与 EU-funded research projects
   - 专门研究 **sparse computation**
   - 为欧洲 supercomputing framework 做贡献

---

## 技术总结与前景展望

### 为什么 Graphcore 在 AI chip 领域独特？

**1. Architecture Philosophy 的根本差异**:
- **GPU** (NVIDIA): 从 graphics rendering 进化而来，擅长 **dense SIMD** operations
- **IPU** (Graphcore): 从 graph theory 和 sparse computation 出发，专为 **dynamic AI workloads** 设计

**2. Memory Wall 的解决方案**:
传统架构的 **Memory Wall** 问题：
$$ \text{Compute Performance} \propto \sqrt{\text{Memory Bandwidth}} $$

传统解决方法是增加 **HBM (High Bandwidth Memory)**：
- NVIDIA A100: 40-80GB HBM2 with 1.6-2.0 TB/s bandwidth
- 但 data movement 仍是 bottleneck，尤其对小模型

Graphcore 的解决方法是 **In-Processor Memory**:
- 900MB distributed SRAM (每个 tile 624KB)
- 数据直接存储在 compute unit 旁边
- 避免了 von Neumann bottleneck
- 特别适合 **Model Parallelism**，将大型模型 distributed 在多个 IPU 上，每个部分 reside in local SRAM

**3. Sparse Computation Acceleration** [https://www.graphipicore.ai/posts/graphcore-and-aleph-alpha-demonstrate-80-sparsified-ai-model](https://www.graphcore.ai/posts/graphcore-and-aleph-alpha-demonstrate-80-sparsified-ai-model)

稀疏矩阵加速的计算优势：
$$ \text{Computation}_{\text{dense}} = O(n^3) $$
$$ \text{Computation}_{\text{sparse}} = O(\text{nnz}) \quad \text{where } \text{nnz} \ll n^3 $$

实际案例：Aleph Alpha 的 13B 模型被 80% sparsified 后仅需 2.6B 参数，推理速度显著提升而 accuracy 保持。

### 挑战与机遇

**挑战**:
1. **Software Ecosystem**: CUDA 有 15+ 年积累，Poplar 相对较新
2. **Market Adoption**: 大型云厂商 (AWS, Azure, Google Cloud) 主要提供 NVIDIA GPU，Graphcore 需自建数据中心或依赖合作伙伴 (如 Cirrascale)
3. **Price Competition**: NVIDIA 的 scale economies 使 GPU 价格持续下降

**机遇**:
1. **Transformer 与 LLM**: 现代 LLM (Large Language Models) 的计算 pattern 越来越 dynamic (variable sequence lengths, attention graphs)，IPU 的 flexibility 有优势。
2. **Graph Neural Networks (GNN)**: 如 molecular simulation, social network analysis, recommendation systems，这些 workloads 天然是 graph 结构，IPU 的 architecture 完美契合。
3. **Edge AI**: 900MB on-chip memory 允许 large models run without external DRAM，适合 power-constrained edge devices。
4. **SoftBank 生态**: 被 SoftBank 收购后，可与 ARM (同属 SoftBank) 的技术 synergize，进入 Japan 及 Asia 市场。

---

总结来说，**Graphcore** 代表了 AI computing 架构的另一种哲学：不是将 GPU 推向极致，而是针对 AI 的 **sparse, dynamic, graph-centric** 本质重新设计 hardware。虽然在 market share 上目前无法与 NVIDIA 抗衡，但其技术创新 (尤其是 3D stacking 和 sparse acceleration) 和在 India 的 strategic investment 预示着它在特定 AI  workloads (GNN, sparse LLM, edge inference) 中仍有重要地位。