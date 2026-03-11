
**n-gram model** 是一种经典的 **statistical language model**，主要用于 **natural language processing** 任务，例如 **language modeling**, **speech recognition**, **machine translation**。其核心思想是：**predict 一个 word 的 probability 时，仅依赖前面 n−1 个 word**，因此 是 **Markov assumption** 的一种具体实现。

---
## 基本定义

- **n-gram**：由连续的 **n 个 word** 组成的 **sequence**
    
    - **unigram**：n = 1
        
    - **bigram**：n = 2
        
    - **trigram**：n = 3
        

例如，给定 一个 sentence：

```
I love natural language processing
```

- **unigram**：`I`, `love`, `natural`, `language`, `processing`
    
- **bigram**：`I love`, `love natural`, `natural language`, `language processing`
    
- **trigram**：`I love natural`, `love natural language`, `natural language processing`
    

---

## Probability 建模思想

在 **language model** 中，目标 是 计算 一个 sentence 的 **joint probability**：

[  
P(w_1, w_2, ..., w_T)  
]

### Chain rule

理论上：  
[  
P(w_1, ..., w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, ..., w_{t-1})  
]

但是，**full history** 计算 复杂，因此 n-gram model 采用 **approximation**：

- **bigram model**：  
    [  
    P(w_t \mid w_{t-1})  
    ]
    
- **trigram model**：  
    [  
    P(w_t \mid w_{t-2}, w_{t-1})  
    ]
    

因此，**context window** 被 限制 为 固定 长度。

---

## Parameter 估计

**probability** 通常 通过 **maximum likelihood estimation (MLE)** 从 **training corpus** 中 计算：

[  
P(w_t \mid w_{t-n+1}^{t-1}) =  
\frac{\text{count}(w_{t-n+1}^{t})}{\text{count}(w_{t-n+1}^{t-1})}  
]

其中：

- **count** 来自 corpus 中 的 **frequency statistics**
    

---

## Data sparsity 问题

但是，随着 n 增大：

- **state space** 指数级 增长
    
- 大量 **n-gram** 在 corpus 中 **never appear**
    

因此，会 出现 **zero probability** 问题。

---

## Smoothing 技术

为了解决 **data sparsity**，通常 使用 **smoothing methods**：

- **Laplace smoothing (add-one)**
    
- **Good–Turing smoothing**
    
- **Kneser–Ney smoothing**
    
- **Backoff** 与 **interpolation**
    

例如，在 **backoff model** 中：

- 如果 **trigram** count 为 zero
    
- 那么 回退 到 **bigram**
    
- 如果 仍然 为 zero
    
- 再 回退 到 **unigram**
    

---

## 优点 与 局限

### 优点

- **conceptually simple**
    
- **easy to implement**
    
- 在 **small-scale system** 中 效果 稳定
    
- 训练 与 inference **computationally efficient**
    

### 局限

- **limited context modeling**
    
- **long-distance dependency** 无法 捕获
    
- 对 **large vocabulary** 不 友好
    
- 性能 明显 弱于 **neural language model**
    

---

## 与 Modern model 的 对比

|Aspect|n-gram model|neural language model|
|---|---|---|
|Context length|fixed|variable / long|
|Feature learning|manual|automatic|
|Data sparsity|severe|alleviated|
|Typical examples|Katz model|RNN, LSTM, Transformer|

因此，在 当前 **deep learning era**，n-gram model 更多 用于：

- **baseline system**
    
- **theoretical teaching**
    
- **resource-constrained environment**
    

---

## 总结

**n-gram model** 是 **language modeling** 的 基石 方法，通过 **local context approximation** 来 估计 **word probability**。虽然 在 现代 应用 中 已 被 **neural model** 超越，但是 其 **probabilistic formulation** 与 **smoothing theory** 仍然 对 理解 NLP 具有 重要 价值。

如果 你 希望，我 可以 进一步 解释 **smoothing algorithm**, **perplexity evaluation**, 或 **n-gram vs Transformer** 的 定量 对比。