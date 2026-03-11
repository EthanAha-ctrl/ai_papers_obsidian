## 一、vLLM 概述

vLLM 是一个**high-throughput** 和 **memory-efficient** 的 **LLM inference** 和 **serving engine**，由 **UC Berkeley**、**CMU**、**Stanford** 等机构的研究人员开发。该项目在 **SOSP 2023** 上发表，主要解决 **LLM serving** 过程中 **KV cache memory** 管理效率低下的问题。

**主要参考：**
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [PagedAttention Paper - arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- [vLLM Official Website](https://vllm.ai/)

## 二、核心技术：PagedAttention

### 2.1 PagedAttention 核心思想

**PagedAttention** 是受到 **OS virtual memory** 和 **paging** 技术启发的 **attention algorithm**。其核心思想包括：

1. **Block-based KV Cache Partitioning**
2. **Non-contiguous Memory Allocation**
3. **On-demand Memory Allocation/Deallocation**
4. **KV Cache Sharing** 机制

### 2.2 PagedAttention 架构详解

```
传统 KV Cache 管理:
┌─────────────────────────────────────┐
│  Request 1: [████████████████████]  │  连续内存块
│  Request 2: [████████████████████]  │  难以动态调整
│  Request 3: [████████████████████]  │  容易产生内存碎片
└─────────────────────────────────────┘

PagedAttention 管理:
┌─────────────────────────────────────┐
│  Block Pool (物理内存块池)          │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐ │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ │
│  └───┴───┴───┴───┴───┴───┴───┴───┘ │
│         ↓     ↓     ↓              │
│  Logical Block Table (逻辑块表)    │
│  ┌─────────────────────────────┐   │
│  │ Req1: [1→3→7]               │   │  非连续分配
│  │ Req2: [2→5→6]               │   │  灵活管理
│  │ Req3: [4→8]                 │   │  减少碎片
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 2.3 PagedAttention 技术细节

#### 2.3.1 Block Table 结构

每个 **sequence** 维护一个 **Block Table**，记录 **logical block index** 到 **physical block index** 的映射：

```python
# Pseudo-code for Block Table
class BlockTable:
    def __init__(self, num_blocks: int):
        # 映射: logical_block_index -> physical_block_index
        self.mapping: List[int] = [-1] * num_blocks
        
    def allocate(self, logical_idx: int, physical_idx: int):
        self.mapping[logical_idx] = physical_idx
        
    def deallocate(self, logical_idx: int):
        self.mapping[logical_idx] = -1
```

#### 2.3.2 Page Size 选择

**Page size** (block size) 的选择影响性能：

| Page Size (tokens) | 内存利用率 | 访问局部性 | 调度开销 | 适用场景 |
|-------------------|-----------|----------|---------|---------|
| 1 | 最高 | 最差 | 最高 | 长序列、高并发 |
| 16 | 中等 | 良好 | 中等 | 平衡场景 |
| 32 | 较低 | 最好 | 较低 | 短序列、低延迟 |

vLLM 默认使用 **16 tokens per block**。

#### 2.3.3 PagedAttention Kernel 实现

**PagedAttention** 通过修改 **attention kernel** 来支持 **paged KV cache**：

```python
# Traditional attention
# K_cache shape: [num_layers, num_heads, head_dim, seq_len]
attention(Q, K_cache[:, :, :, :current_len])

# Paged attention
# K_cache blocks: list of physical blocks
# block_table: logical -> physical mapping
def paged_attention(Q, K_blocks, V_blocks, block_table):
    # 1. 根据 block_table 获取所需的物理块
    required_blocks = [block_table[i] for i in logical_indices]
    
    # 2. 从物理块中 gather KV 数据
    K_paged = gather_blocks(K_blocks, required_blocks)
    V_paged = gather_blocks(V_blocks, required_blocks)
    
    # 3. 计算注意力
    return attention(Q, K_paged, V_paged)
```

## 三、其他核心技术

### 3.1 Continuous Batching (Iterative Batching)

**vLLM** 实现了 **continuous batching** 机制，允许动态添加和移除 requests：

```
Timeline 0:    [Req1━━━━━━━━━━━━━━] [Req2━━━━━━━━━━━━━━]
               └───────────────────┴───────────────────┘
                         Batch size = 2

Timeline 1:    [Req1━━━━━━━━━━] [Req2━━━━━━━━━━━━━━] [Req3━━━]
               └─────────────┴───────────────────┴───────┘
                          Batch size = 3

Timeline 2:    [Req1████████████] [Req2━━━━━━━━━━] [Req3━━━━━━━]
               └──────────────────┴───────────────┴───────────┘
                    (completed)         Batch size = 2
```

**Continuous Batching 算法流程：**

```python
def continuous_batching_scheduler():
    running_requests = []
    waiting_queue = []
    
    while True:
        # 1. 检查已完成的请求
        completed = [req for req in running_requests if req.finished]
        for req in completed:
            running_requests.remove(req)
            release_blocks(req.block_table)
        
        # 2. 从等待队列添加新请求
        while can_add_request(waiting_queue, running_requests):
            req = waiting_queue.pop(0)
            if can_allocate_blocks(req):
                allocate_blocks(req)
                running_requests.append(req)
        
        # 3. 执行推理
        if running_requests:
            execute_batch(running_requests)
```

### 3.2 KV Cache Sharing

**vLLM** 支持两种级别的 **KV cache sharing**：

#### 3.2.1 Intra-request Sharing

同一请求内共享 **common prefixes**：

```
System Prompt (shared)
    ↓
┌─────────────────────────────┐
│ Block 0 [████████████████]  │ ← 共享的 system prompt KV
├─────────────────────────────┤
│ Block 1 [████████████████]  │ ← Request A 的扩展
├─────────────────────────────┤
│ Block 2 [████████████████]  │ ← Request B 的扩展
└─────────────────────────────┘
```

#### 3.2.2 Inter-request Sharing

不同请求间共享 **identical prefixes**：

```python
# CUDAGraph Cache 实现原理
class PrefixCache:
    def __init__(self):
        self.cache = {}  # hash(prefix) -> KV blocks
        
    def get_or_compute(self, prefix_tokens):
        key = hash(tuple(prefix_tokens))
        if key not in self.cache:
            # 计算并缓存
            self.cache[key] = compute_kv_cache(prefix_tokens)
        return self.cache[key]
```

**Automatic Prefix Caching** 流程：

```
Input Request:
┌─────────────────────────────────────────────┐
│ System: "You are a helpful assistant."      │
│ User: "What is AI?"                         │
└─────────────────────────────────────────────┘
                    ↓
Prefix Matching:
┌─────────────────────────────────────────────┐
│ Cache Hit: [System: "You are a..."]        │
│ Cache Miss:  [User: "What is AI?"]          │
└─────────────────────────────────────────────┘
                    ↓
KV Cache Assembly:
┌─────────────────────────────────────────────┐
│ [Cached Block 0-3] + [New Block 4]          │
│   (System prompt)    (User query)           │
└─────────────────────────────────────────────┘
```

### 3.3 Speculative Decoding 支持

**vLLM** 支持 **MLP Speculator** 进行加速生成：

**Speculative Decoding 原理：**

```
Step 1: Small Model (Speculator) 生成候选 tokens
─────────────────────────────────────────────
Input:  [The capital of France is]
        ↓
Speculator: [Paris, ,, the, largest, city]
            └────────────────────────────┘
              5 个候选 tokens

Step 2: Large Model 验证候选 tokens
─────────────────────────────────────────────
Input:  [The capital of France is Paris]
        ↓
Verifier: ✓  ✓  ✓  ✗ (stop)
        
Result:  [The capital of France is Paris, ]
         └──────────────────────────────┘
           3 个 tokens 一次性生成
```

**Speculative Deciving 性能公式：**

```
Speedup ≈ 1 / (T_draft / N_draft + T_verify / N_verify)

其中:
T_draft = draft model inference time per token
T_verify = large model verification time per token  
N_draft = number of tokens drafted
N_verify = number of tokens verified/accepted
```

### 3.4 CUDA Graph Integration

**CUDA Graph** 通过减少 **kernel launch overhead** 提升性能：

```cpp
// CUDA Graph 创建流程
cudaGraph_t graph;
cudaGraphExec_t graphExec;

// 1. 创建 graph stream
cudaStream_t graphStream;
cudaStreamCreate(&graphStream);

// 2. 捕获 kernel 执行
cudaStreamBeginCapture(graphStream, cudaStreamCaptureModeGlobal);

// 执行 vLLM kernel 序列
paged_attention_kernel<<<...>>>(...);
mlp_kernel<<<...>>>(...);
layer_norm_kernel<<<...>>>(...);

cudaStreamEndCapture(graphStream, &graph);

// 3. 实例化 graph
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 4. 执行 graph (低开销)
cudaGraphLaunch(graphExec, stream);
```

**CUDA Graph 优化效果：**

| 操作类型 | 传统方式 | CUDA Graph | 提升 |
|---------|---------|-----------|------|
| Kernel Launch | ~5-10 μs | ~0.5-1 μs | 5-10x |
| 小 Batch 延迟 | 高 | 低 | 显著 |
| GPU 利用率 | 中等 | 高 | 10-20% |

## 四、vLLM 架构设计

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  OpenAI API │  │  LangChain  │  │ Custom App  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼──────────────────┘
          │                │                │
┌─────────┼────────────────┼────────────────┼──────────────────┐
│         │        API Server Layer (FastAPI)                   │
│  ┌──────▼──────┐  ┌─────────────────────────────┐             │
│  │  /generate  │  │     /v1/chat/completions    │             │
│  └──────┬──────┘  └────────────┬────────────────┘             │
└─────────┼─────────────────────┼───────────────────────────────┘
          │                     │
┌─────────▼─────────────────────▼───────────────────────────────┐
│                    LLM Engine (vLLM)                          │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Scheduler                                           │   │
│  │  - Continuous Batching                              │   │
│  │  - Request Priority Management                        │   │
│  │  - Block Manager                                     │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  KV Cache Manager                                     │   │
│  │  - Block Table Mapping                               │   │
│  │  - Prefix Caching                                    │   │
│  │  - Memory Pool Management                            │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Block Manager (GPU Memory)                           │   │
│  │  - Physical Block Allocation                          │   │
│  │  - Block Reuse                                       │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────┐
│                    Model Execution Layer                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PagedAttention Kernel                                 │   │
│  │  - Paged KV Cache Access                              │   │
│  │  - Flash Attention Integration                        │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Model Kernels (CUDA)                                  │   │
│  │  - Linear Layers                                      │   │
│  │  - Layer Norm                                         │   │
│  │  - Activation Functions                               │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼─────────────────────────────────────────────────────┐
│                      Hardware Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   NVIDIA    │  │     AMD     │  │     TPU     │             │
│  │     GPU     │  │     GPU     │  │   (Cloud)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 核心组件详解

#### 4.2.1 Scheduler

**Scheduler** 负责 **request scheduling** 和 **resource allocation**：

```python
class LLMScheduler:
    def __init__(self, block_manager, max_model_len):
        self.block_manager = block_manager
        self.waiting_queue = []
        self.running_requests = []
        self.max_model_len = max_model_len
        
    def schedule(self):
        """Main scheduling loop"""
        # 1. Preemption: swap out low-priority requests if needed
        self._preempt_requests()
        
        # 2. Promote: move waiting requests to running
        self._promote_requests()
        
        # 3. Schedule steps for running requests
        return self._create_schedule()
        
    def _promote_requests(self):
        """Promote requests from waiting to running"""
        for request in self.waiting_queue[:]:
            # Check if we can allocate required blocks
            required_blocks = request.num_required_blocks()
            if self.block_manager.can_allocate(required_blocks):
                self.block_manager.allocate(request, required_blocks)
                self.waiting_queue.remove(request)
                self.running_requests.append(request)
```

#### 4.2.2 BlockManager

**BlockManager** 管理 **physical memory blocks**：

```python
class BlockManager:
    def __init__(self, block_size, num_gpu_blocks):
        self.block_size = block_size  # tokens per block
        self.num_gpu_blocks = num_gpu_blocks
        self.free_blocks = list(range(num_gpu_blocks))
        self.block_tables = {}  # request_id -> BlockTable
        
    def allocate(self, request, num_blocks):
        """Allocate blocks for a request"""
        if len(self.free_blocks) < num_blocks:
            return False
        
        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        
        # Create block table for request
        block_table = BlockTable(num_blocks)
        for i, block_idx in enumerate(allocated):
            block_table.allocate(i, block_idx)
            
        self.block_tables[request.request_id] = block_table
        return True
        
    def free(self, request):
        """Free blocks for a request"""
        block_table = self.block_tables[request.request_id]
        for logical_idx in range(len(block_table.mapping)):
            physical_idx = block_table.mapping[logical_idx]
            if physical_idx != -1:
                self.free_blocks.append(physical_idx)
        
        del self.block_tables[request.request_id]
```

#### 4.2.3 CacheEngine

**CacheEngine** 处理 **KV cache operations**：

```python
class CacheEngine:
    def __init__(self, block_manager, model_config):
        self.block_manager = block_manager
        self.num_layers = model_config.num_hidden_layers
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.dtype = model_config.dtype
        
        # Allocate GPU cache
        self.gpu_cache = self._allocate_gpu_cache()
        
    def _allocate_gpu_cache(self):
        """Allocate GPU KV cache"""
        cache_size = (
            self.block_manager.num_gpu_blocks *
            self.num_layers *
            self.num_heads *
            self.head_dim *
            2  # K and V
        )
        return torch.empty(
            (self.num_layers, 2, 
             self.block_manager.num_gpu_blocks,
             self.num_heads, 
             self.block_manager.block_size,
             self.head_dim),
            dtype=self.dtype,
            device='cuda'
        )
    
    def get_cache_block(self, layer_idx, block_idx):
        """Get KV cache for a specific block"""
        return self.gpu_cache[layer_idx, :, block_idx, :, :, :]
```

## 五、性能对比与实验数据

### 5.1 与其他系统的性能对比

根据论文中的实验数据：

| System | Model | Throughput (tokens/s) | Latency (ms) | Memory Waste |
|--------|-------|---------------------|-------------|-------------|
| FasterTransformer | LLaMA-7B | 29.3 | 832 | 20-30% |
| Orca | LLaMA-7B | 31.2 | 785 | 15-25% |
| **vLLM** | **LLaMA-7B** | **89.7** | **275** | **<4%** |
| FasterTransformer | LLaMA-13B | 15.8 | 1546 | 25-35% |
| Orca | LLaMA-13B | 17.2 | 1421 | 20-30% |
| **vLLM** | **LLaMA-13B** | **47.3** | **518** | **<4%** |

### 5.2 不同 Sequence Length 下的性能

```
Throughput vs. Sequence Length (LLaMA-13B)
───────────────────────────────────────────────────────────────────────
                │
         2000   │                          ◯ vLLM
                │                      ◯
         1500   │                  ◯
Throughput      │              ◯
 (tokens/s)     │          ◯
         1000   │      ◯
                │  ◯ FasterTransformer
          500   │◯
                │◯─────────────────────────────────────────────────────
           0    └─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─
                  512  1024 1536 2048 2560 3072 3584 4096 4608 5120
                          Sequence Length
───────────────────────────────────────────────────────────────────────
```

**速度提升公式：**

```
Speedup = Throughput_vLLM / Throughput_baseline

对于 seq_len = 4096:
- vs FasterTransformer: 1450 / 220 ≈ 6.6x
- vs Orca: 1450 / 280 ≈ 5.2x
```

### 5.3 Batch Size 扩展能力

| Batch Size | FasterTransformer | Orca | vLLM |
|-----------|-------------------|------|------|
| 8 | 100% | 100% | 100% |
| 16 | 180% | 190% | 195% |
| 32 | 300% | 340% | 380% |
| 64 | OOM | 520% | 750% |
| 128 | OOM | OOM | 1420% |

### 5.4 内存利用率对比

```
Memory Utilization Over Time
───────────────────────────────────────────────────────────────────────
100% ┤  ██████████████████████████████████████████████████████████████
     │  ██                    vLLM: ~96% utilization
 80% ┤  ████
     │  ██████████████████████████████████████████████████████████████
 60% ┤  ████                    Orca: ~75% utilization
     │  ████████████████████████████████████████████████████████████
 40% ┤  ████████                FasterTransformer: ~60% utilization
     │  ████████████████████████████████████████████████████████████
 20% ┤  ████████████
     │  ████████████████████████████████████████████████████████████
  0% └─────────────────────────────────────────────────────────────
     0    2    4    6    8    10   12   14   16   18   20
                      Time (minutes)
───────────────────────────────────────────────────────────────────────
```

## 六、高级特性与优化

### 6.1 Quantization 支持

**vLLM** 支持多种 **quantization schemes**：

```python
# Supported quantization types
QUANTIZATION_TYPES = {
    "fp8": {
        "weight_dtype": torch.float8_e4m3fn,
        "activation_dtype": torch.float8_e4m3fn,
        "memory_saving": "2x"
    },
    "int8": {
        "weight_dtype": torch.int8,
        "activation_dtype": torch.float16,
        "memory_saving": "1.5x"
    },
    "int4": {
        "weight_dtype": torch.int4,
        "activation_dtype": torch.float16,
        "memory_saving": "3.5x"
    },
    "awq": {
        "type": "Activation-aware Weight Quantization",
        "precision": "4-bit",
        "memory_saving": "4x"
    },
    "gptq": {
        "type": "GPT Quantization",
        "precision": "4-bit", 
        "memory_saving": "4x"
    }
}

# Usage example
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="awq"  # 使用 AWQ 4-bit 量化
)
```

**Quantization 性能对比：**

| Quantization | Model Size | Latency | Throughput | Quality Loss |
|-------------|-----------|---------|-----------|--------------|
| FP16 | 70B (140GB) | baseline | baseline | 0% |
| FP8 | 70B (70GB) | +15% | +15% | <1% |
| INT8 | 70B (85GB) | +20% | +20% | <1% |
| AWQ-INT4 | 70B (40GB) | +35% | +35% | 1-2% |
| GPTQ-INT4 | 70B (40GB) | +35% | +35% | 1-2% |

### 6.2 LoRA 支持

**vLLM** 原生支持 **LoRA (Low-Rank Adaptation)**：

```python
# LoRA architecture
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, rank)
        )
        self.scaling = alpha / rank
        
    def forward(self, x):
        # Base path: x @ W^T
        base_output = F.linear(x, self.base_weight)
        
        # LoRA path: x @ A^T @ B^T * scaling
        lora_output = F.linear(
            F.linear(x, self.lora_A), 
            self.lora_B
        ) * self.scaling
        
        return base_output + lora_output

# Multi-LoRA serving
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=8,  # 支持8个不同LoRA
    max_lora_rank=64
)

# Apply different LoRA adapters to requests
outputs = llm.generate(
    prompts=[prompt1, prompt2],
    lora_request=[
        LoRARequest("adapter1", 1, "path/to/adapter1"),
        LoRARequest("adapter2", 2, "path/to/adapter2")
    ]
)
```

### 6.3 Distributed Inference

**Tensor Parallelism** 实现：

```python
# vLLM Tensor Parallelism 配置
TP_SIZE = 4  # 使用4张GPU进行张量并行

# Linear layer parallelization
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        # 每个GPU持有输出维度的一部分
        self.out_per_partition = out_features // world_size
        self.weight = nn.Parameter(
            torch.empty(self.out_per_partition, in_features)
        )
        self.rank = rank
        self.world_size = world_size
        
    def forward(self, x):
        # 计算本地部分
        output = F.linear(x, self.weight)
        # All-gather 合并各GPU的结果
        output = dist.all_gather(output, dim=-1)
        return output

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        # 每个GPU持有输入维度的一部分
        self.in_per_partition = in_features // world_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_partition)
        )
        self.rank = rank
        self.world_size = world_size
        
    def forward(self, x):
        # Split 输入到各GPU
        x = dist.scatter(x, dim=-1)
        # 计算本地部分
        output = F.linear(x, self.weight)
        # All-reduce 求和各GPU的结果
        output = dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output
```

### 6.4 Prefix Caching 详解

**Prefix Caching** 技术细节：

```python
class PrefixCacheManager:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.prefix_cache = {}  # hash -> cached blocks
        self.block_ref_count = defaultdict(int)
        
    def compute_prefix_hash(self, tokens):
        """计算prefix的哈希值"""
        # 使用 rolling hash 提高效率
        h = 0
        for token in tokens:
            h = (h * 31 + token) % (2**64)
        return h
    
    def get_cached_blocks(self, prefix_tokens):
        """获取缓存的blocks"""
        if not prefix_tokens:
            return []
            
        # 计算每个可能的prefix hash
        cached_blocks = []
        current_hash = 0
        
        for i, token in enumerate(prefix_tokens):
            current_hash = (current_hash * 31 + token) % (2**64)
            if current_hash in self.prefix_cache:
                cached_blocks.extend(
                    self.prefix_cache[current_hash]
                )
            else:
                break
                
        return cached_blocks
    
    def cache_blocks(self, tokens, blocks):
        """缓存blocks"""
        hash_value = self.compute_prefix_hash(tokens)
        self.prefix_cache[hash_value] = blocks
```

**Prefix Caching 效果：**

| 场景 | 无缓存 | Prefix Caching | 提升 |
|------|--------|---------------|------|
| System Prompt 复用 | 100% | 30% | 3.3x |
| Few-shot Examples | 100% | 25% | 4x |
| 对话历史重用 | 100% | 40% | 2.5x |
| 无重复输入 | 100% | 95% | 1.05x |

### 6.5 Ray Integration

**vLLM** 与 **Ray** 集成实现 **distributed serving**：

```python
from ray import serve

@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,  # 每个replica占用1张GPU
    },
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5
    }
)
class vLLMDeployment:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(
            model="meta-llama/Llama-2-70b-hf",
            tensor_parallel_size=1,  # 单卡
            gpu_memory_utilization=0.9
        )
        
    async def generate(self, prompt: str):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=256
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

# 部署
serve.run(vLLMDeployment.bind())
```

## 七、使用示例

### 7.1 基础使用

```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # 单GPU
    gpu_memory_utilization=0.9,  # GPU内存利用率
    max_model_len=4096,  # 最大序列长度
    block_size=16,  # KV cache block大小
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,  # 温度
    top_p=0.95,  # nucleus sampling
    top_k=50,  # top-k sampling
    max_tokens=256,  # 最大生成token数
    stop=["\n\n"],  # 停止条件
    presence_penalty=0.0,  # 存在惩罚
    frequency_penalty=0.0,  # 频率惩罚
    repetition_penalty=1.0,  # 重复惩罚
)

# 批量推理
prompts = [
    "Write a story about AI.",
    "Explain quantum computing.",
    "What is the meaning of life?",
]
outputs = llm.generate(prompts, sampling_params)

# 处理输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")
```

### 7.2 OpenAI Compatible API

```python
from vllm.entrypoints.openai.api_server import run_server
import uvicorn
from fastapi import FastAPI

app = FastAPI()

# 启动vLLM服务器
run_server(
    app=app,
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    host="0.0.0.0",
    port=8000,
)

# 客户端调用示例
import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-70b-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=128,
)
```

### 7.3 Prefix Caching 使用

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 启用prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
    max_model_len=4096,
)

# 第一次请求（需要计算system prompt）
system_prompt = "You are a helpful AI assistant."
user_prompt1 = "What is Python?"
outputs1 = llm.generate(
    prompts=[system_prompt + user_prompt1],
    sampling_params=SamplingParams(max_tokens=256)
)

# 第二次请求（重用cached system prompt）
user_prompt2 = "What is machine learning?"
outputs2 = llm.generate(
    prompts=[system_prompt + user_prompt2],
    sampling_params=SamplingParams(max_tokens=256)
)
# system prompt的KV cache会被复用，加速推理
```

## 八、最新发展与版本特性

### 8.1 vLLM 0.6.x 新特性

```
vLLM 版本特性对比
───────────────────────────────────────────────────────────────────────
特性               │ 0.5.x   │ 0.6.x   │ 0.7.x (preview)
──────────────────────────┼─────────┼─────────┼─────────────────────
PagedAttention      │    ✓    │    ✓    │        ✓ (优化版)
Continuous Batching │    ✓    │    ✓    │        ✓ (改进)
Prefix Caching     │    ✓    │    ✓    │  ✓ (Automatic)
Speculative Decoding│   Manual │    ✓    │        ✓ (MLP Spec.)
Multi-LoRA         │  Limited │    ✓    │        ✓ (动态加载)
Quantization       │   AWQ   │ AWQ/GPTQ │  AWQ/GPTQ/FB8
Distributed        │   TP    │  TP+DP  │   TP+DP+EP
Vision Models      │    ✗    │   LLaVA │   多模态支持
Long Context       │  32K    │   128K  │      1M+
───────────────────────────────────────────────────────────────────────
```

### 8.2 Long Context 支持

**vLLM** 支持 **long context** 扩展：

```python
# vLLM 0.7+ 支持1M+ context
llm = LLM(
    model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    max_model_len=1048576,  # 1M tokens
    enable_chunked_prefill=True,  # 分块预填充
    long_context_backend="flashinfer",  # 使用FlashInfer后端
)

# Chunked Prefill 原理
# ┌─────────────────────────────────────────────────────────────┐
# │ Long Prompt (1M tokens)                                       │
# │ ┌─────────────────────────────────────────────────────────┐ │
# │ │ Chunk 1 (16K) → Prefill → Decode →                      │ │
# │ │ Chunk 2 (16K) → Prefill → Decode →                      │ │
# │ │ Chunk 3 (16K) → Prefill → Decode →                      │ │
# │ │ ...                                                      │ │
# │ │ Chunk N → Prefill → Decode → Final Output               │ │
# │ └─────────────────────────────────────────────────────────┘ │
# └─────────────────────────────────────────────────────────────┘
```

### 8.3 FlashAttention 集成

**vLLM** 集成了 **FlashAttention-2** 和 **FlashInfer**：

```python
# FlashAttention 优化
import flash_attn

# PagedAttention + FlashAttention 混合
def hybrid_attention(Q, K, V, block_table, cache_blocks):
    # 对于连续内存部分使用 FlashAttention
    if is_contiguous(cache_blocks):
        return flash_attn.flash_attn_func(Q, K, V)
    # 对于非连续（paged）部分使用 PagedAttention
    else:
        return paged_attention_kernel(Q, K, V, block_table)

# 性能对比
# FlashAttention:  ~2x speedup vs naive attention
# FlashAttention-2: ~2x speedup vs FlashAttention
# Total: ~4x speedup vs baseline
```

### 8.4 Vision-Language Models

**vLLM** 支持多模态模型：

```python
from vllm import LLM
from vllm.assets.image import ImageAsset

# 支持的多模态模型
vision_models = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "microsoft/Phi-3-vision-128k-instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "nvidia/NVLM-D-72B"
]

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    max_model_len=4096,
)

# 多模态推理
image = ImageAsset("example.jpg").pil_image
outputs = llm.generate({
    "prompt": "Describe this image in detail.",
    "multi_modal_data": {"image": image}
})
```

## 九、性能调优指南

### 9.1 配置参数调优

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `gpu_memory_utilization` | 0.9 | 0.85-0.95 | GPU内存利用率 |
| `block_size` | 16 | 16-32 | KV cache block大小 |
| `max_num_batched_tokens` | - | 8192 | 最大batch tokens数 |
| `max_num_seqs` | 256 | 128-512 | 最大序列数 |
| `enable_prefix_caching` | False | True (对话场景) | 前缀缓存 |
| `enforce_eager` | False | False (调试时True) | 禁用CUDA Graph |

### 9.2 性能分析工具

```python
from vllm import LLM
import time

# 启用性能分析
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_profiling=True,
    profile_output_dir="./profile"
)

# 分析指标
metrics = {
    "time_to_first_token": [],  # TTFT
    "tokens_per_second": [],    # TPS
    "gpu_memory_usage": [],     # 内存使用
    "batch_size": [],           # 批次大小
}

# 运行benchmark
for batch_size in [1, 4, 16, 32, 64]:
    prompts = ["Hello!"] * batch_size
    
    start_time = time.time()
    outputs = llm.generate(prompts)
    end_time = time.time()
    
    # 收集指标
    tps = sum(len(o.outputs[0].text.split()) for o in outputs) / (end_time - start_time)
    metrics["tokens_per_second"].append(tps)
    metrics["batch_size"].append(batch_size)
```

### 9.3 常见问题解决

```python
# OOM 错误解决方案
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 增加TP
    gpu_memory_utilization=0.85,  # 降低GPU利用率
    swap_space=4,  # 启用CPU swap
    max_model_len=2048,  # 减少最大长度
)

# 高延迟问题解决方案
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    enable_chunked_prefill=True,  # 启用分块预填充
    max_num_batched_tokens=8192,  # 调整batch限制
    max_num_seqs=128,  # 减少最大序列数
)

# 内存碎片问题解决方案
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    block_size=32,  # 增大block大小
    enable_prefix_caching=True,  # 启用prefix caching
    preemption_mode="recompute",  # 使用recompute而非swap
)
```

## 十、生态系统与集成

### 10.1 框架集成

**vLLM** 可以与以下框架集成：

```python
# LangChain 集成
from langchain.llms import VLLM

llm = VLLM(
    model="meta-llama/Llama-2-7b-hf",
    trust_remote_code=True
)

result = llm("Explain quantum computing")

# LlamaIndex 集成
from llama_index.llms import Vllm
from llama_index import VectorStoreIndex

llm = Vllm(
    model="meta-llama/Llama-2-7b-hf",
    temperature=0.1
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)

# Transformers 集成
from transformers import pipeline
from vllm.transformers_utils import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer="meta-llama/Llama-2-7b-hf"
)
```

### 10.2 部署选项

```
vLLM 部署架构选项
───────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────┐
│ 选项1: 单机部署 (Single Node)                                       │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │
│ │  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │       │   │
│ │  │ TP Rank │  │ TP Rank │  │ TP Rank │  │ TP Rank │       │   │
│ │  │   0     │  │   1     │  │   2     │  │   3     │       │   │
│ │  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │   │
│ │              vLLM Engine (Tensor Parallelism)               │   │
│ └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 选项2: Ray 分布式部署 (Ray Distributed)                               │
│ ┌──────────────────────────┐  ┌──────────────────────────┐           │
│ │  Head Node               │  │  Worker Node 1          │           │
│ │  ┌────────────────────┐  │  │  ┌─────────┐  ┌───────┐ │           │
│ │  │  API Server        │  │  │  │ GPU 0   │  │ GPU 1 │ │           │
│ │  │  (FastAPI)         │  │  │  │ vLLM    │  │ vLLM  │ │           │
│ │  └────────────────────┘  │  │  └─────────┘  └───────┘ │           │
│ │  ┌────────────────────┐  │  │  ┌─────────┐  ┌───────┐ │           │
│ │  │  Ray Serve         │  │  │  │ GPU 2   │  │ GPU 3 │ │           │
│ │  └────────────────────┘  │  │  │ vLLM    │  │ vLLM  │ │           │
│ └──────────────────────────┘  └──────────────────────────┘           │
│         │ Ray GCS/Driver              │ Ray GCS/Driver            │
│ └─────────────────────────────────────┴──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 选项3: Kubernetes 部署 (K8s Deployment)                               │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │  ┌───────────────────────────────────────────────────────────┐  │  │
│ │  │  Kubernetes Cluster                                      │  │  │
│ │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │  │
│ │  │  │    Pod 1   │  │    Pod 2   │  │    Pod 3   │      │  │  │
│ │  │  │ GPU: 1     │  │ GPU: 2     │  │ GPU: 1     │      │  │  │
│ │  │  │ vLLM Instance│  │ vLLM Instance│  │ vLLM Instance│     │  │  │
│ │  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │  │
│ │  │         │                 │                 │           │  │  │
│ │  │  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐    │  │  │
│ │  │  │ Load Balancer│  │ Load Balancer│  │ Load Balancer│    │  │  │
│ │  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │  │
│ │  └───────────────────────────────────────────────────────────┘  │  │
│ └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 十一、总结

vLLM 通过其创新的 **PagedAttention** 技术彻底改变了 **LLM serving** 的内存管理方式，实现了：

1. **接近零的 KV Cache 内存浪费** (<4%)
2. **2-4倍于现有系统的吞吐量**
3. **灵活的 KV Cache 共享机制**
4. **支持各种高级特性** (Quantization, LoRA, Speculative Decoding)
5. **良好的生态系统集成**

**主要参考链接：**
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Benchmarks](https://docs.vllm.ai/en/latest/benchmarking/)
- [vLLM Community](https://docs.vllm.ai/en/latest/community/)