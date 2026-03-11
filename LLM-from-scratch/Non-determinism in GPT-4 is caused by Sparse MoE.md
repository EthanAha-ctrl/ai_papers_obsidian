 **Sherman Chann 的"Non-determinism in GPT-4 is caused by Sparse MoE"**（2023年8月5日）。这是解释 GPT-4 即使 temperature=0 仍然不确定性的关键文献。

## 核心问题：Sparse MoE 导致的 Batch 竞争

### 问题根源

Sparse Mixture of Experts 架构在 **capacity constraints** 下，tokens 会 **按照固定大小的组进行 routing**：

```
┌─────────────────────────────────────────┐
│  Batch Level Determinism Only           │
├─────────────────────────────────────────┤
│                                         │
│   Sequence A (tokens) ─┐              │
│                        ├─→ Expert Router → Expert Buffers │
│   Sequence B (tokens) ─┘              │
│                        │               │
│   Sequence C (tokens) ─┤               │
│                        └─ Competition for Expert Capacity  │
│                                         │
└─────────────────────────────────────────┘
```

**关键机制**：
- tokens 从 **不同 sequences** 在同一 batch 中会被 **grouped together**
- 这些 tokens 会 **竞争** expert buffers 中的 **available spots**
- 结果：**模型在 sequence-level 不再 deterministic，只在 batch-level 是 deterministic**
- 某些 input sequences 可能 **影响** 其他 inputs 的最终预测

### 实验证据

Sherman Chann 的实验结果（30次尝试，temperature=0）：

| Model | Unique Completions | 平均值 | 确定性 |
|-------|-------------------|--------|--------|
| gpt-4 | 12, 11, 12 | 11.67 | ❌ 严重不稳定 |
| gpt-3.5-turbo | 4, 4, 3 | 3.67 | ❌ 不稳定 |
| text-davinci-003 | 3, 2, 4 | 3.00 | ❌ 不稳定 |
| text-davinci-001 | 2, 2, 2 | 2.00 | ❌ 不稳定 |
| davinci-instruct-beta | 1, 1, 1 | 1 | ✅ deterministic |
| davinci | 1, 1, 1 | 1 | ✅ deterministic |

这说明：
- **MoE 模型（GPT-4）** 确实表现出了最严重的 non-determinism
- **非 MoE 模型（davinci 系列）** 是 fully deterministic 的

## 相关理论支撑

### Puigcerver et al. (2023) - "From Sparse to Soft Mixtures of Experts"

论文 Section 2.2 中的关键论述：

> **Under capacity constraints**, all Sparse MoE approaches route tokens in **groups of a fixed size** and enforce (or encourage) **balance within the group**. When groups contain tokens from **different sequences or inputs**, these tokens often **compete against each other** for available spots in expert buffers. **As a consequence, the model is no longer deterministic at the sequence-level, but only at the batch-level**, as some input sequences may affect the final prediction for other inputs.

## 其他影响因素（非主要）

### 1. Floating-Point Non-Associativity

```
(a + b) + c ≠ a + (b + c)  // 由于舍入误差
```

Python 示例：
```python
(1 + 1e16) - 1e16   # 0.0
1 + (1e16 - 1e16)   # 1.0
```

**影响机制**：
- GPU 并行计算中，浮点数加法顺序可能不同
- 不同执行顺序导致不同的舍入误差累积
- 误差可能改变 top-k logits 的排序

### 2. Non-deterministic GPU Operations

某些 GPU kernels 使用 atomic operations：
```python
# 并行 reduction 的伪代码
# 多个线程同时累加到同一个位置
atomic_add(buffer[thread_idx], value)  # 执行顺序不确定
```

## 可能的解决方案方向

虽然从现有信息中没有看到 Sherman Chann 明确提出的"固定输出方法"，但可以从相关研究推断几个方向：

### 1. **Per-Sequence Capacity Reservation**

为每个 sequence 分配独立的 expert capacity，避免跨序列竞争：

```
┌─────────────────────────────────────────┐
│  Per-Sequence Determinism               │
├─────────────────────────────────────────┤
│                                         │
│   Sequence A ─→ Expert Buffer A (dedicated) │
│   Sequence B ─→ Expert Buffer B (dedicated) │
│   Sequence C ─→ Expert Buffer C (dedicated) │
│                                         │
└─────────────────────────────────────────┘
```

### 2. **Soft MoE Routing**

使用 soft assignment 而不是 hard routing：

- 通过加权组合多个 experts 的输出
- 减少对 capacity constraints 的依赖
- 提高数值稳定性

### 3. **Deterministic Kernel Implementation**

如 llama.cpp 的 deterministic mode：
- 确保相同的输入产生 bit-identical 的输出
- 独立于 batch size 和 prompt chunking
- 参考：https://github.com/ggml-org/llama.cpp/pull/16016

### 4. **Fixed Batch Composition**

对于需要确定性的场景：
- 使用 **相同的 batch composition**（包括其他 sequences）
- 保持 **一致的 batch ordering**
- 确保 **expert capacity 分配一致**

## 相关参考资料

1. **Sherman Chann (2023)** - "Non-determinism in GPT-4 is caused by Sparse MoE"  
   https://152334h.github.io/blog/non-determinism-in-gpt-4/

2. **Puigcerver et al. (2023)** - "From Sparse to Soft Mixtures of Experts"  
   https://arxiv.org/html/2308.00951v2

3. **Martynas Šubonis (2025)** - "Zero Temperature Randomness in LLMs"  
   https://martynassubonis.substack.com/p/zero-temperature-randomness-in-llms

4. **Thinking Machines Lab** - "Defeating Nondeterminism in LLM Inference"  
   https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

5. **Atil et al. (2024)** - "Non-Determinism of 'Deterministic' LLM Settings"  
   https://arxiv.org/html/2408.04667v5

## 总结

您提到的 Sherman Chann 的论文正是揭示了 **Sparse MoE 架构中的 batch-level 竞争** 是 GPT-4 temperature=0 时仍然 non-deterministic 的主要原因。这个问题根源于 MoE 的 **per-sequence determinism 缺失**，而不是简单的浮点精度或并行执行问题。要实现确定性输出，需要从架构层面解决 token routing 的跨序列竞争问题。