



# 🧠 Transformer 作为计算机：深度技术解析

这两篇文章来自 **Percepta.ai** 团队，提出了一个革命性的研究方向：**将 Transformer 变成一台可以执行程序的计算机**。让我深入讲解这个工作。

---

## 一、动机与背景：为什么 LLM 不能计算？

### 1.1 核心矛盾

| 能力 | 表现 |
|------|------|
| **高级数学推理** | 可达到 IMO 金牌水平，甚至解决开放数学问题 |
| **基础计算** | 连简单的加法都会出错，无法独自解决简单的 Sudoku |

这个矛盾非常深刻：

> "Humans cannot fly. Building airplanes does not change that; it only means we built a machine that flies for us."

目前的 LLM 就像人类造飞机一样——它们可以使用工具来计算，但**计算能力仍然在模型之外**。

### 1.2 当前解决方案的局限性

```
┌─────────────────────────────────────────────────────────┐
│  Tool Use 的工作流程：                                   │
│                                                          │
│  LLM → "让我计算一下" → 生成 Python 代码                │
│                              ↓                          │
│                         外部解释器执行                   │
│                              ↓                          │
│                         返回结果 → LLM 继续              │
└─────────────────────────────────────────────────────────┘
```

**问题**：模型本身仍然"残废"，必须依赖外部系统。

---

## 二、核心贡献：在 Transformer 内部构建计算机

### 2.1 总体架构

```
┌────────────────────────────────────────────────────────────┐
│                    构建流程                                 │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   C 代码 → WebAssembly → Token 序列 → Transformer 执行    │
│                              ↓                             │
│                      执行跟踪          │
│                              ↓                             │
│                         输出结果                           │
│                                                            │
│   关键：每条指令映射到最多 5 个 token！                    │
└────────────────────────────────────────────────────────────┘
```

### 2.2 为什么是 WebAssembly？

**WebAssembly (WASM)** 的特点：
- 低级指令集，快速确定性执行
- C/C++ 可以编译到 WASM
- 设计简洁，适合嵌入

---

## 三、关键技术突破：指数级加速的 Attention

### 3.1 传统 Transformer 的结构缺陷

**问题**：标准自回归解码每一步都与整个历史交互。

```
传统解码复杂度：
- 第 t 步：需要与长度为 t 的前缀交互
- 每步工作：O(t)
- 生成 t 个 token 的总成本：O(t²)
```

**对比真实计算机**：
| 特性 | 真实计算机 | 传统 Transformer |
|------|------------|------------------|
| 状态更新 | 紧凑状态（寄存器、栈、内存）| 每步生成 token |
| 每步工作 | 近似常数 | 随序列长度线性增长 |

### 3.2 解决方案：2D Attention + HullKVCache

#### 核心思想

将 **attention head dimension 限制为 2**，使得 attention 查询变成**凸包查询**。

```
┌─────────────────────────────────────────────────────────────┐
│  2D Attention 的几何视角                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     Key k 映射到 2D 点：k ↦ (2k, -k²)                      │
│                                                             │
│     这些点位于开口向下的抛物线上！                          │
│                                                             │
│         k=0  k=1  k=2  k=3  k=4  k=5                       │
│          ●    ●    ●    ●    ●    ●                        │
│                    抛物线                                   │
│                                                             │
│     Query q = (i, 1) 作为方向向量                          │
│                                                             │
│     score(q, k) = 2ik - k² = -(k-i)² + i²                 │
│                                                             │
│     当 k = i 时，score 最大！                               │
└─────────────────────────────────────────────────────────────┘
```

#### 数学推导

**Score 函数**：

$$\text{score}(q, k) = q \cdot \text{key}(k) = (i, 1) \cdot (2k, -k^2) = 2ik - k^2$$

可以重写为：

$$\text{score}(q, k) = -(k-i)^2 + i^2$$

**关键性质**：
- 当 $k = i$ 时，score 达到最大值 $i^2$
- 对于任何 $k \neq i$，存在二次惩罚 $-(k-i)^2 < 0$

这实现了**精确的 key-value 查找**！

### 3.3 性能对比

| 缓存类型 | 每步复杂度 | 生成 41,709 tokens 耗时 |
|----------|------------|------------------------|
| **标准 KVCache** | O(t) 线性扫描 | **258.9 秒** |
| **HullKVCache** | O(log t) 凸包查询 | **1.3 秒** |

**加速比：约 200 倍！**

```
图表对比：

tokens generated
    ↑
40k ┼─────────────────────────────────●●●●● HullKVCache
    │
30k ┼─────────────────────────────────/
    │                                /
20k ┼────────────────────────────────/
    │                               /
10k ┼───────────────────────────────/
    │                              /
    ┼──────────────────────────────●●●●●●●●●●● KVCache
    │                             /
    └──────┬──────┬──────┬──────┬──────┬──────→ time (s)
           50    100    150    200    250
```

---

## 四、技术细节：ALM 与 CALM

### 4.1 Append-only Lookup Machine (ALM)

**抽象计算模型**，定义 Transformer 可实现的原始操作：

| 原语 | 数学表示 | Transformer 实现 |
|------|----------|------------------|
| **Read/Write** | $\text{write}_c(k, v)$, $\text{read}_c(q)$ | Attention |
| **Cumulative Sum** | $\text{cumsum}_c(v)$ | Attention + 乘法 |
| **Product** | $a \cdot b$ | ReGLU |
| **Conditional** | $\text{if } a \text{ then } b \text{ else } c$ | ReGLU |
| **Linear Combination** | $c_1 \cdot x + c_2 \cdot y$ | Residual stream |

### 4.2 精确内存查找：抛物线 Key 编码

```
Key 编码：k → (2k, -k²)

示例：查询 key = 3

k     key(k)        q·key    score
─────────────────────────────────────
0     (0, 0)        +0       0
1     (2, -1)       +5       5
2     (4, -4)       +8       8
3     (6, -9)       +9       ★ 最大！
4     (8, -16)      +8       8
5     (10, -25)     +5       5
6     (12, -36)     +0       0
7     (14, -49)     -7       -7
```

### 4.3 累积和的实现

当所有 position 使用相同的 attention key 时，head 返回**均匀平均值**：

$$\frac{1}{t+1} \sum_{i \leq t} v_i$$

乘以 $(t+1)$ 恢复精确累积和：

$$\sum_{i \leq t} v_i = (t+1) \cdot \frac{1}{t+1} \sum_{i \leq t} v_i$$

### 4.4 ReGLU 实现条件逻辑

**ReLU 的离散性质**（对于整数输入 $z$）：

$$\mathbf{1}_{[z \geq 0]} = \text{ReLU}(z+1) - \text{ReLU}(z)$$

**等式测试**：

$$\mathbf{1}_{[x = c]} = \mathbf{1}_{[x-c \geq 0]} - \mathbf{1}_{[x-c-1 \geq 0]}$$

**条件选择**：

$$\text{if } C \text{ then } u \text{ else } v = u \cdot \mathbf{1}_{[C]} + v \cdot (1 - \mathbf{1}_{[C]})$$

---

## 五、编译流程：从程序到权重

### 5.1 Gate Graph（门图）

CALM 程序编译为两种门：
- **LookUp gates**：Attention，实现精确 key-value 查找和累积和
- **ReGLU gates**：Feed-forward，实现阶跃函数、指示器、乘积、条件逻辑

```
┌────────────────────────────────────────────────────────┐
│           Gate Graph 示例               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  [LookUp gate] ──→ [ReGLU gate] ──→ [LookUp gate]     │
│        ↑                    │                ↓         │
│        └────────────────────┴────────────────┘        │
│                   residual stream                      │
│                                                        │
│  每个门对应一个确定性的数学操作                         │
└────────────────────────────────────────────────────────┘
```

### 5.2 调度问题

将 Gate Graph 嵌入有限层和有限宽度。

**每层的四个阶段**：
1. **Attention**：LookUp gates 从序列读取
2. **Materialization**：attention 输出写入 residual stream
3. **Feed-forward**：ReGLU gates 执行局部非线性逻辑
4. **Materialization**：feed-forward 输出写入 residual stream

### 5.3 Mixed-Integer Linear Program (MILP)

**决策变量**：将每个操作分配到特定层和阶段

**约束**：
1. **Precedence**：消费者必须在生产者之后调度
2. **Type compatibility**：LookUp 只能在 attention 阶段，ReGLU 只能在 feed-forward 阶段
3. **Co-location**：共享中间结果的操作约束到同一层

**目标**：最小化 peak utilization（同时存活值的数量），这直接决定 $d_{\text{model}}$

---

## 六、程序特化：将程序编译进权重

### 6.1 Universal Interpreter vs. Specialized Executor

| 类型 | 程序存储 | 指令获取方式 |
|------|----------|-------------|
| **Universal Interpreter** | Prompt 前缀 | 通过 attention 读取 |
| **Specialized Executor** | Feed-forward 权重 | ReGLU step-function 查找 |

### 6.2 特化的数学原理

对于有 $N$ 条指令的程序，构建 $2N$ 个共享 ReGLU neurons，计算程序计数器的阶跃函数：

$$\mathbf{1}_{[\text{cursor} \geq i]}, \quad 0 \leq i < N$$

每个获取的字段变成线性组合：

$$\text{fetched\_field} = c_0 + \sum_{i=1}^{N-1}(c_i - c_{i-1}) \cdot \mathbf{1}_{[\text{cursor} \geq i]}$$

其中 $c_i$ 是指令 $i$ 处该字段的值。

**效果**：程序现在存在于权重矩阵中，而不是输入序列中！

---

## 七、实现细节

### 7.1 模型配置

```python
class VanillaTransformer(nn.Module):
    def __init__(self, vocab, d_model=36, n_heads=18, n_layers=7, d_ffn=36):
        # d_model = 36, n_heads = 18 → 每个头 2D！
        self.tok = nn.Embedding(vocab, d_model)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
            for _ in range(n_layers)
        ])
        self.ff_in  = nn.ModuleList([nn.Linear(d_model, 2*d_ffn, bias=False)])
        self.ff_out = nn.ModuleList([nn.Linear(d_ffn, d_model, bias=False)])
        self.head = nn.Linear(d_model, vocab, bias=False)
```

**完全标准的 PyTorch Transformer！** 唯一特别的是权重。

### 7.2 执行示例

**计算 3 + 5**：

```
输入：
{
  i32.const 03 00 00 00
  i32.const 05 00 00 00
  i32.add   00 00 00 00
  output    00 00 00 00
}

执行跟踪：
03 00 00 00  commit(+1,sts=1,bt=0)   ← 栈增长
05 00 00 00  commit(+1,sts=1,bt=0)   ← 栈增长
08 00 00 00  commit(-1,sts=1,bt=0)   ← 加法，栈减少
out(08)
halt
```

---

## 八、演示案例

### 8.1 Min-Cost Perfect Matching（匈牙利算法）

**输入**：10×10 代价矩阵

**输出**：最优匹配

**性能**：
- 33,887 tok/s
- 19,798 tokens
- 7,367 lines/s

### 8.2 Arto Inkala Sudoku（世界最难 Sudoku）

**性能**：
- 31,148 tok/s
- 5,608,298 tokens
- 用时 < 3 分钟
- 32 次猜测，7 层深度

**关键洞察**：autoregressive 模型本身不是问题，问题是长执行跟踪的成本。我们的 fast attention path 解决了这个问题。

---

## 九、未来方向

### 9.1 更丰富的 Attention 机制

- **k-sparse softmax attention**：检索 top-k keys，复杂度 $O(k + \log n)$
- **3D attention**：通过 3D 凸包，但高维度效率下降

### 9.2 训练大规模 2D-head 模型

可能的应用模式：
1. **Fast path + Slow path**：快速执行器配对慢速通用模型
2. **Hybrid architecture**：单系统内 fast/slow 混合
3. **Speculative execution**：快速提出 token，常规 attention 模型验证

### 9.3 程序编译为权重

```
┌───────────────────────────────────────────────────────┐
│  传统训练：数据 → 梯度下降 → 权重                      │
│                                                        │
│  新范式：   程序 → 编译器 → 权重                       │
│                                                        │
│  意义：权重成为软件的部署目标！                        │
└───────────────────────────────────────────────────────┘
```

### 9.4 Growing AI Systems Like Software

未来 AI 系统可能像软件库一样演化：
- 积累模块、抽象、可重用组件
- 新的计算能力增量添加到模型内部执行引擎

---

## 十、第一性原理分析

### 10.1 为什么 2D Attention 如此重要？

**第一性原理**：凸包的性质

1. **凸包的定义**：包含所有点的最小凸集
2. **支持点查询**：给定方向，找凸包上最远的点
3. **复杂度**：使用适当数据结构（如凸包树），查询复杂度 $O(\log n)$

**关键洞察**：2D attention 将 attention 变成支持点查询！

### 10.2 为什么抛物线编码有效？

抛物线 $y = -x^2$ 的性质：
- **开口向下**：凸函数
- **任意方向**的点积最大值在凸包上

结合：
- Keys 落在抛物线上
- Query 作为方向向量
- 找最大点积 = 找凸包支持点

### 10.3 为什么 ALM 是 Turing Complete？

**五个原语足够表达任意整数计算**：
1. Read/Write：实现内存
2. Cumsum：实现计数器
3. Product：实现乘法
4. Conditional：实现分支
5. Linear combination：实现数据流

这与真实计算机的基本能力等价。

---

## 十一、潜在应用与影响

### 11.1 可靠计算

- **医疗决策**：需要精确计算
- **供应链优化**：需要长执行跟踪
- **金融建模**：需要可靠执行

### 11.2 可微分计算

执行跟踪是 forward pass 的一部分，可以传播梯度！

**这意味着**：
- 计算过程本身可训练
- 与外部工具有本质区别

### 11.3 形式验证

由于执行是确定性的、可追踪的，可以形式验证 Transformer 实现的逻辑。

---

## 十二、参考资料

1. **Min-cost perfect matching**: https://en.wikipedia.org/wiki/Assignment_problem
2. **IMO AI systems**: DeepMind AlphaGeometry, etc.
3. **Sudoku-Bench**: https://github.com/...
4. **Tool-integrated reasoning**: PAL, Code Interpreter, etc.
5. **Transformer universality**: "Transformers are universal computers"
6. **Training computational capabilities**: "Chain-of-thought elicits reasoning"
7. **KV caching**: "Efficient inference in transformers"
8. **Transformer decoding research**: FlashAttention, etc.
9. **WebAssembly**: https://webassembly.org/
10. **Neural Sudoku solvers**: Various papers
11. **Autoregressive limitations**: Various papers
12. **Fast KV cache**: FlashAttention, PagedAttention, etc.

**GitHub**: https://github.com/transformer-vm

---

## 总结

这项工作的核心贡献是：

1. **理论**：证明 Transformer 可以高效执行程序
2. **技术**：2D attention + HullKVCache 实现指数级加速
3. **系统**：ALM/CALM + MILP 编译器 + WASM 解释器
4. **愿景**：软件成为神经网络的一部分

**最终图景**：

> "In that world, software itself becomes part of the model."

这不仅仅是让 LLM "能计算"，而是让计算成为模型内在能力的一部分。模型不再只是协调计算，而是**本身就是一台计算机**。