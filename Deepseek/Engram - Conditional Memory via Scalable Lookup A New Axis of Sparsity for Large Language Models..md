
![[Pasted image 20260126144636.png]]
# Engram论文深度技术解析

## 一、核心问题背景和动机

### 1.1 稀疏性的双重属性

现代LLM中主要存在两种稀疏性：
- **稀疏计算**：Mixture-of-Experts(MoE)通过条件计算扩展容量
- **稀疏记忆**：Engram提出的条件记忆，用于静态知识的O(1)查找

### 1.2 语言建模的双重任务

语言建模包含两种异质子任务：
- **组合推理**：需要深度、动态的计算
- **知识检索**：本地、静态、高度模式化

当前Transformer缺乏原生知识查找原语，被迫通过计算模拟检索。

### 1.3 效率问题

解决常见多token实体需要消耗多个前序层，这本质上是在运行时重建静态查找表：

```
实体解析过程（以"Diana, Princess of Wales"为例）：
Layer 1-2:  → Wales
Layer 3:     → Europe的国家 → Wales
Layer 4:     → 女性主权者的头衔 → Princess of Wales (非特定)
Layer 5:     → 威尔士亲王妻子的头衔 → Princess of Wales (非特定)
Layer 6:     → Diana, Princess of Wales完整信息
```

## 二、Engram架构详细技术实现

### 2.1 整体架构

Engram是一个条件记忆模块，通过**检索**和**融合**两个阶段处理：

```
输入序列 X = (x₁, ..., x_T)
在层ℓ的隐藏状态 H^(ℓ) ∈ R^(T×d)

每个位置t的处理：
1. 检索阶段：提取并压缩后缀N-gram，确定性哈希检索静态嵌入
2. 融合阶段：上下文感知门控、卷积细化
```

### 2.2 Tokenizer压缩

#### 2.2.1 问题分析

标准子词分词器优先无损重建，常为语义等价词分配不同ID：
- "Apple" vs "␣apple"
- "Python"(语言) vs "Python"(蛇)

#### 2.2.2 技术实现

通过预计算的满射函数 `P: V → V'` 压缩词表：

```python
# 伪代码
def tokenizer_compression(raw_id, P):
    canonical_id = P(raw_id)
    # 基于NFKC归一化、小写化等文本等价性
    return canonical_id

# 效果：128k词表压缩23%
```

#### 2.2.3 N-gram构造

对于位置t的token，构造后缀N-gram：

```
g_(t,n) = (x'_(t-n+1), ..., x'_t)
```

### 2.3 多头哈希机制

#### 2.3.1 哈希函数设计

采用轻量级乘-XOR哈希：

```
z_(t,n,k) = φ_(n,k)(g_(t,n))
e_(t,n,k) = E_(n,k)[z_(t,n,k)]
```

其中：
- `φ_(n,k)`：确定性哈希函数
- `E_(n,k)`：大小为素数M_(n,k)的嵌入表
- `n`：N-gram阶数（2,3）
- `k`：哈希头索引（1到K）

#### 2.3.2 哈希冲突缓解

```python
# 多头哈希设计
def multi_head_hash(gram, num_heads=8):
    embeddings = []
    for head in range(num_heads):
        # 每个头使用不同的哈希种子
        hash_idx = multiplicative_xor_hash(gram, seed=head)
        embed = embedding_table[head][hash_idx]
        embeddings.append(embed)
    return concatenate(embeddings)
```

#### 2.3.3 嵌入向量化

最终记忆向量构建：

```
e_t = ||_(n=2)^N ||_(k=1)^K e_(t,n,k)
```

这个向量包含所有检索到的嵌入的拼接。

### 2.4 上下文感知门控机制

#### 2.4.1 设计原理

检索的嵌入 `e_t` 作为上下文无关先验，但可能存在哈希冲突和多义性噪声。通过上下文感知门控动态调制。

#### 2.4.2 核心计算

```python
# 伪代码实现
def context_aware_gating(h_t, e_t, W_K, W_V):
    # 当前隐藏状态h_t聚合全局上下文，作为动态Query
    # 检索记忆e_t作为Key和Value的源
    
    k_t = W_K @ e_t          # Key投影
    v_t = W_V @ e_t          # Value投影
    
    # RMSNorm应用
    h_t_norm = rms_norm(h_t)
    k_t_norm = rms_norm(k_t)
    
    # 标量门计算
    alpha_t = sigmoid(dot(h_t_norm, k_t_norm) / sqrt(d))
    
    # 门控输出
    v_tilde = alpha_t * v_t
    
    return v_tilde
```

#### 2.4.3 梯度稳定性

RMSNorm公式：

```
RMSNorm(x) = x / sqrt(mean(x²) + ε)
```

门控值的范围约束：
```
α_t = σ( RMSNorm(h_t)ᵀ · RMSNorm(k_t) / √d )
```

其中 `σ` 为sigmoid函数，确保门控值在(0,1)范围。

### 2.5 深度因果卷积

#### 2.5.1 扩张感受野

```python
def depthwise_causal_convolution(V_tilde, kernel_size=4, dilation=3):
    """
    V_tilde: 形状为[T × d]的序列
    kernel_size: 卷积核大小（设为4）
    dilation: 膨胀率，设为最大N-gram阶数
    """
    V_norm = rms_norm(V_tilde)
    conv_out = conv1d(V_norm, kernel_size=kernel_size, dilation=dilation)
    activated = silu(conv_out)
    final_output = activated + V_tilde  # 残差连接
    return final_output
```

#### 2.5.2 最终融合

```
Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ
H^(ℓ) ← H^(ℓ) + Y
```

### 2.6 多分支架构集成

#### 2.6.1 参数共享策略

对于M=4多分支架构：
- **共享组件**：
  - 单个稀疏嵌入表
  - Value投影矩阵 W_V
- **分支特定**：
  - M个不同的Key投影矩阵 {W^(m)_K}_(m=1)^M

#### 2.6.2 分支特定门控计算

```
α^(m)_t = σ( RMSNorm(h^(m)_t)ᵀ · RMSNorm(W^(m)_K e_t) / √d )
u^(m)_t = α^(m)_t · (W_V e_t)
```

#### 2.6.3 计算效率优化

```python
# 融合矩阵乘法的实现
class MultiBranchGating:
    def __init__(self, d_model, d_mem, num_branches=4):
        self.W_V = nn.Linear(d_mem, d_model)
        self.W_K_list = nn.ModuleList([
            nn.Linear(d_mem, d_model) 
            for _ in range(num_branches)
        ])
    
    def forward(self, h_branch_list, e_t):
        # 一次性计算所有分支
        v_shared = self.W_V(e_t)
        
        outputs = []
        for h_branch, W_K_branch in zip(h_branch_list, self.W_K_list):
            alpha = self.gate(h_branch, W_K_branch(e_t))
            u = alpha * v_shared
            outputs.append(u)
        return outputs
```

### 2.7 系统级设计

#### 2.7.1 训练阶段

```
模型并行策略：
- 嵌入表分片到可用GPU
- 使用All-to-All通信原语
- 前向传播收集活动行
- 反向传播分发梯度
```

#### 2.7.2 推理阶段

```
预取和重叠策略：
```
```python
class PrefetchManager:
    def __init__(self, host_memory, gpu_device):
        self.host_mem = host_memory
        self.gpu_device = gpu_device
        self.prefetch_queue = []
    
    def schedule_prefetch(self, sequence, engram_layer_idx):
        # 确定性预先计算哈希索引
        hash_indices = precompute_hashes(sequence)
        
        # 异步预取
        for idx in hash_indices:
            embedding = self.host_mem[idx]
            # 利用前置层计算时间覆盖通信延迟
            self.prefetch_queue.append(
                transfer_to_gpu(embedding, self.gpu_device)
            )
```

#### 2.7.3 多级缓存层次

利用N-gram的Zipfian分布：

```
访问频率分布：
- 20%的高频模式 → 80%的访问
- 80%的长尾罕见模式 → 20%的访问

缓存策略：
- HBM (GPU): 缓存高频嵌入（最快访问）
- Host DRAM: 存储中频嵌入
- NVMe SSD: 存储罕见模式的大容量存储
```

## 三、稀疏分配的U型缩放定律

### 3.1 问题的形式化定义

#### 3.1.1 参数指标定义

```
P_tot: 总可训练参数（排除词表嵌入和LM头）
P_act: 每token激活参数（决定训练FLOPs）
P_sparse = P_tot - P_act: 非激活参数预算
```

#### 3.1.2 分配比例定义

```
ρ ∈ [0,1]: 分配给MoE专家的非激活参数比例

P^(sparse)_MoE = ρ · P_sparse
P_Engram = (1-ρ) · P_sparse
```

#### 3.1.3 边界情况

```
ρ = 1: 纯MoE模型（所有非激活参数都是路由专家）
ρ < 1: 减少路由专家，重新分配参数给Engram嵌入槽
```

### 3.2 实验协议

#### 3.2.1 计算预算设置

```
小计算预算 (C=2×10^20 FLOPs):
- P_tot ≈ 5.7B
- P_act = 568M
- 基准(ρ=1)有106个专家

大计算预算 (C=6×10^20 FLOPs):
- P_tot ≈ 9.9B  
- P_act = 993M
- 基准(ρ=1)有99个专家
```

#### 3.2.2 稀疏比例约束

```
P_tot / P_act ≈ 10  跨两个设置保持恒定
```

### 3.3 U型缩放定律的发现

#### 3.3.1 验证损失曲线

```
实验结果 (10B模型):
- ρ = 100% (纯MoE): Loss = 1.7248
- ρ ≈ 80% (最优): Loss = 1.7109  (Δ = -0.0139)
- ρ → 0% (Engram主导): Loss上升

U型特征：
○ 最优点：ρ ≈ 75%-80%
○ 左侧上升(ρ→0): 失去条件计算能力
○ 右侧上升(ρ→100): 缺乏静态模式记忆
```

#### 3.3.2 稳定性分析

在不同计算规模下，最优ρ值保持稳定，表明稀疏分配偏好的一致性。

### 3.4 无限记忆缩放

#### 3.4.1 实验设定

```
固定MoE骨架:
- P_tot ≈ 3B
- P_act = 568M
- 训练: 100B tokens

Engram缩放: M从2.58×10^5到1.0×10^7嵌入槽
```

#### 3.4.2 幂律缩放

```
验证损失遵循严格幂律：
log(Validation Loss) = a · log(M) + b

表明Engram提供可预测的缩放控制旋钮：
更大的内存容量持续产生收益，无需额外计算
```

## 四、大规模预训练实验和结果

### 4.1 模型配置详情

#### 4.1.1 骨架架构

```
Transformer骨架 (30 blocks):
- Hidden Size: 2560
- Multi-head Latent Attention (MLA): 32 heads
- mHC expansion rate: 4
- Optimizer: Muon
```

#### 4.1.2 模型变体

```
Dense-4B:
- 总参数: 4.1B
- 结构: 标准密集FFN

MoE-27B:
- 总参数: 26.7B
- 结构: DeepSeekMoE
- 专家: 2共享 + 72路由 (top-k=6)

Engram-27B:
- 总参数: 26.7B
- 结构: 改进MoE-27B
- 专家: 2共享 + 55路由
- Engram参数: 5.7B (ρ=74.3%)
- Engram配置: Layers 2和15, N=3, K=8, d=1280
- Embedding优化: Adam (5×学习率, 0权重衰减)
- Conv初始化: 零(保持恒等映射)

Engram-40B:
- 总参数: 39.5B
- 结构: Engram-27B骨架
- Engram参数: 18.5B
```

### 4.2 训练数据

```
- Token数量: 262B tokens
- Tokenizer: DeepSeek-v3 (128k词表)
- 同样的数据课程和顺序
```

### 4.3 实验结果详解

#### 4.3.1 语言建模性能

```
Pile Test Loss:
- Dense-4B: 2.091
- MoE-27B: 1.960  (△ = -0.131)
- Engram-27B: 1.950  (△ = -0.010 vs MoE)
- Engram-40B: 1.942  (继续改善)

Validation Loss:
- Dense-4B: 1.768
- MoE-27B: 1.634
- Engram-27B: 1.622  (△ = -0.012 vs MoE)
- Engram-40B: 1.610  (△ = -0.024 vs MoE)
```

#### 4.3.2 知识与推理任务

```
MMLU (5-shot准确率):
- Dense-4B: 48.6%
- MoE-27B: 57.4%  (+8.8%)
- Engram-27B: 60.4%  (+3.0% vs MoE)
- Engram-40B: 60.6%

MMLU-Pro (5-shot准确率):
- Dense-4B: 21.1%
- MoE-27B: 28.3%
- Engram-27B: 30.1%  (+1.8% vs MoE)
- Engram-40B: 31.3%  (+3.0% vs MoE)

CMMLU (5-shot准确率):
- Dense-4B: 47.9%
- MoE-27B: 57.9%
- Engram-27B: 61.9%  (+4.0% vs MoE)
- Engram-40B: 63.4%  (+5.5% vs MoE)
```

#### 4.3.3 推理任务增强

```
BBH (3-shot EM):
- Dense-4B: 42.8%
- MoE-27B: 50.9%
- Engram-27B: 55.9%  (+5.0% vs MoE)
- Engram-40B: 57.5%

ARC-Challenge (25-shot准确率):
- Dense-4B: 59.3%
- MoE-27B: 70.1%
- Engram-27B: 73.8%  (+3.7% vs MoE)
- Engram-40B: 76.4%  (+6.3% vs MoE)

DROP (1-shot F1):
- Dense-4B: 41.6%
- MoE-27B: 55.7%
- Engram-27B: 59.0%  (+3.3% vs MoE)
- Engram-40B: 60.7%  (+5.0% vs MoE)
```

#### 4.3.4 代码与数学任务

```
HumanEval (0-shot Pass@1):
- Dense-4B: 26.8%
- MoE-27B: 37.8%
- Engram-27B: 40.8%  (+3.0% vs MoE)
- Engram-40B: 38.4%  (略微下降)

MATH (4-shot EM):
- Dense-4B: 15.2%
- MoE-27B: 28.3%
- Engram-27B: 30.7%  (+2.4% vs MoE)
- Engram-40B: 30.6%

GSM8K (8-shot EM):
- Dense-4B: 35.5%
- MoE-27B: 58.4%
- Engram-27B: 60.6%  (+2.2% vs MoE)
- Engram-40B: 62.6%
```

### 4.4 关键发现分析

#### 4.4.1 超越预期

```
预期: 知识密集型任务受益最大
实际: 推理任务改善更显著

知识任务增益: MMLU +3.0, CMMLU +4.0
推理任务增益: BBH +5.0, ARC-Challenge +3.7
代码任务增益: HumanEval +3.0
数学任务增益: MATH +2.4
```

#### 4.4.2 机制解释

引入专门的知识查找原语显著提高了表示效率，超越了单纯分配稀疏预算给条件计算的效果。

## 五、长文本训练能力

### 5.1 理论基础

#### 5.1.1 注意力容量释放

```
传统模型:
- Attention用于本地依赖建模 + 全局上下文处理

Engram模型:
- 本地依赖 → 静态查找（O(1)）
- Attention → 专注于全局上下文
```

### 5.2 长文本扩展实验

#### 5.2.1 训练配置

```
上下文扩展策略 (DeepSeek-V3):
- 超参数: s=10, α=1, β=32, f=0.707
- 训练长度: 32768 tokens
- 训练步骤: 5000步 (30B tokens高质量长文本数据)

YaRN扩展:
- 使用3种上下文窗口扩展方法
- 旋转位置编码的修改
```

#### 5.2.2 模型配置对照

```
比较配置:
1. MoE-27B (50k步) - 完全训练
2. Engram-27B (50k步) - 完全训练
3. Engram-27B (46k步) - 同等的验证损失控制
4. Engram-27B (41k步) - 早期停止（82%计算预算）
```

### 5.3 评估基准

#### 5.3.1 LongPPL评估集合

```
四个类别长文本:
1. 长篇书籍
2. 研究论文
3. 代码仓库
4. 长链思考轨迹
```

#### 5.3.2 RULER评估

```
14个子集聚合为8个类别:
1. 单键 Needle-in-a-Haystack (S)
2. 多键 NIAH (MK)
3. 多值 NIAH (MV)
4. 多查询 NIAH (MQ)
5. 多跳变量跟踪
6. 常见词提取 (CWE)
7. 频繁词提取 (FWE)
8. 问答 (QA)
```

### 5.4 实验结果

#### 5.4.1 LongPPL结果

```
32k上下文Perplexity:
- MoE-27B (50k, 1.63): 4.38
- Engram-27B (41k, 1.66): 4.37  (相当)
- Engram-27B (46k, 1.63): 4.19  (更好)
- Engram-27B (50k, 1.62): 4.14  (最佳)

关键发现: 长上下文性能与基础建模能力相关
```

#### 5.4.2 RULER详细结果

```
Multi-Query NIAH (最复杂的检索任务):
- MoE-27B: 84.2%
- Engram-27B (41k): 99.6%  (+15.4%)
- Engram-27B (46k): 97.6%  (+13.4%)
- Engram-27B (50k): 99.3%  (+15.1%)

变量跟踪:
- MoE-27B: 77.0%
- Engram-27B (46k): 87.2%  (+10.2%)

CWE (常见词提取):
- MoE-27B: 100.0%
- Engram-27B (46k): 100.0%  (保持)
```

### 5.5 关键结论

#### 5.5.1 临界控制必要性

```
假设: 长上下文性能由架构机制决定
现实: 与基础模型质量相关
结论: 需要控制基础模型损耗而非仅对齐训练步数
```

#### 5.5.2 架构优势

```
Iso-Loss设置 (46k vs Baseline 50k):
- 多查询NIAH: 97.0% vs 84.2%
- 变量跟踪: 87.2% vs 77.0%

Iso-FLOPs设置 (50k vs Baseline 50k):
- 全部RULER子集改善

极端设置 (82%计算):
- Engram-27B (41k) vs MoE-27B (50k)
- 同等或更优性能
```

## 六、机制分析

### 6.1 功能等效深度验证

#### 6.1.1 LogitLens分析

```
方法:
- 使用最终LM Head投影每个中间层的隐藏状态
- 计算中间输出和最终输出的KL散度
- 量化潜在表示"准备好预测"的程度

KL散度公式:
KL(P || Q) = Σ_x P(x) log(P(x) / Q(x))
```

#### 6.1.2 实验发现

```
对比三层模型:
- MoE-27B: KL散度下降缓慢
- Engram-27B: 前几个层的KL散度显著更低
- Engram-40B: 前期层收敛最快的表现

解释:
Engram加速预测收敛，早期层更快完成特征组合
```

#### 6.1.3 CKA分析

```
中心核对齐公式:

CKA(K, L) = HSIC(K, L) / √(HSIC(K, K) · HSIC(L, L))

其中:
K = X X^T (表示矩阵X的格拉姆矩阵)
L = Y Y^T
HSIC: Hilbert-Schmidt独立性准则
```

#### 6.1. 软对齐索引

```
对于每个Engram层j:
1. 计算与所有MoE层的CKA相似度
2. 选择top-k最高相似度
3. 计算加权质心:

a_j = (Σ_{i∈I_j} S_{i,j} · i) / (Σ_{i∈I_j} S_{i,j})

其中:
I_j = argtop_k_i(S_{i,j})
S_{i,j}: MoE层i和Engram层j的相似度
```

#### 6.1.5 深度等效结果

```
发现:
- Engram-27B的第5层表示
- 与MoE-27B的第12层表示最相似
- 系统性的对角线上移

结论:
Engram通过显式查找绕过早期特征组合
功能上等同于增加模型的有效深度
```

### 6.2 结构消融与层级敏感度

#### 6.2.1 基准配置

```
3B MoE骨干:
- 12层
- 激活参数: 0.56B
- 训练: 100B tokens
- Engram参数: 1.6B
- N-gram: {2,3}
- 插入层次: Layers 2和6
- 验证损失: 1.768 (vs 基础1.808)
```

#### 6.2.2 层级扫描结果

```
单层插入实验:
- Layer 1: 验证损失稍高于Layer 2
- Layer 2: 最佳性能 (1.770)
- Layer 3-12: 递减性能

双层插入:
- Layers 2和6: 最佳 (1.768)
- 原因: 平衡早期介入和后期丰富上下文门控

权衡分析:
○ 早期插入: 卸载本地模式重建
○ 深度插入: 更好的上下文感知门控
○ 双层插入: 结合两者的优势
```

#### 6.2.3 组件消融

```
关键组件排序 (按重要性降序):
1. 多分支集成
   - 移除后验证损失显著增加

2. 上下文感知门控
   - 语义对齐和噪声抑制

3. Tokenizer压缩
   - 提高语义密度

4. 深度因果卷积
   - 轻微性能下降

5. 4-gram扩展
   - 固定预算下略微次优
   - 高阶N-gram在大规模内存下可能更有利
```

### 6.3 敏感性分析

#### 6.3.1 后验消融方法

```
方法:
- 推理时完全抑制稀疏嵌入输出
- 保持骨干网络不变
- 评估性能保持比例
```

#### 6.3.2 功能二分性发现

```
事实知识任务 (极度依赖Engram):
- TriviaQA: 29% (下降71%)
- TriviaQA-ZH: 44%
- PopQA: 44%
- CMMLU: 72%
- MMLU: 75%

阅读理解任务 (骨干网络主导):
- C3: 93% (仅下降7%)
- RACE-Middle: 89%
- RACE-High: 84%
- DROP: 81%

分类:
1. 事实性知识: Engram作为主要参数知识存储
2. 上下文密集任务: 主要依赖骨干注意机制
```

### 6.4 系统效率验证

#### 6.4.1 确定性访问优势

```
确定性内存访问模式:
- 索引在令牌序列已知时即可确定
- 可在相应层执行前预先计算
```

#### 6.4.2 推理吞吐量实验

```
测试环境:
- 硬件: NVIDIA H800
- 工作负载: 512序列, 长度Uniform(100, 1024)
- Engram: 100B参数, 完全驻留在主机DRAM

4B骨架吞吐量:
- 基线: 9,031.62 tok/s
- +100B Engram (CPU卸载): 8,858.28 tok/s
- 下降: 2.8%

8B骨架吞吐量:
- 基线: 6,315.52 tok/s  
- +100B Engram (CPU卸载): 6,140.02 tok/s
- 下降: 2.8%
```

#### 6.4.3 通信效率

```
有效通信量计算:
- 每步与激活槽数量成比例
- 而非总嵌入表大小

利用层级设计:
- Zipfian分布缓存
- 高频项在HBM
- 长尾项在SSD
- 最小化有效延迟
```

## 七、门控机制可视化案例

### 7.1 门控激活模式

```
检测到的模式:
1. 多令牌命名实体
   - "Alexander the Great"
   - "Diana, Princess of Wales"
   - "the Milky Way"

2. 模式化短语
   - "By the way"
   - "Princess of Wales"

3. 跨语言泛化
   中文例子:
   - "四大发明" (Four Great Inventions)
   - "张仲景" (Zhang Zhongjing)
```

### 7.2 门控值范围

```
α_t ∈ [0, 1]
- 0: 完全抑制 (上下文矛盾)
- 1: 完全激活 (上下文一致)
- 中间值: 部分调制
```

## 八、技术创新点总结

### 8.1 核心创新

1. **条件记忆新稀疏轴**: 与MoE的条件计算互补
2. **N-gram现代化**: 哈希、门控、压缩等技术升级
3. **U型缩放定律**: 最稀疏分配的理论发现
4. **确定性访问**: 支持预取和计算重叠
5. **功能等效深度**: 验证的机制解释

### 8.2 与相关工作区别

```
对比OverEncoding:
- Engram: 严格的等参数和等FLOPs比较
- OverEncoding: 无法在稀疏MoE骨架上产生实质性改进

对比SCONE:
- Engram: 第一类建模原语
- SCONE: 推理聚焦，需要额外FLOPs
```

### 8.3 系统设计原则

```
算法系统协同设计:
1. 深度嵌入注入: 启用通信计算重叠
2. Zipfian分布利用: 最大化硬件层级效用
3. 整体方法: 支持大规模参数的近零延迟扩展
```

## 九、参考链接

**论文和代码:**
- Engram官方代码库: https://github.com/deepseek-ai/Engram
- DeepSeek-MoE论文: https://arxiv.org/abs/2401.04066
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3

**相关技术:**
- MoE架构: https://arxiv.org/abs/2101.03961
- Multi-head Latent Attention: DeepSeek技术报告
- Manifold-Constrained Hyper-Connections: Xie et al. (2025)
- YaRN位置编码: https://arxiv.org/abs/2309.00071

**评估基准:**
- MMLU: https://arxiv.org/abs/2009.03300
- BBH: https://arxiv.org/abs/2210.09261
- LongPPL: Fang et al.
- RULER: Hsieh et al.

**分析方法:**
- LogitLens: https://arxiv.org/abs/2304.14949
- CKA: https://arxiv.org/abs/1905.00414
- PatchScope: Ghandeharioun et al. (2024)

## 十、未来展望

### 10.1 直接扩展

```
1. 高阶N-gram探索
   - 4-gram和更大
   - 大规模内存预算下的效益

2. 更大的多分支架构
   - M > 4
   - 更细粒度的门控控制

3. 与检索集成
   - 组合参数和非参数记忆
   - 混合稀疏架构
```

### 10.2 应用扩展

```
1. 多模态应用
   - 图像和视频N-gram模式
   - 跨模态条件记忆

2. 代码生成优化
   - 代码特定模式查找
   - API调用模式识别

3. 推理加速
   - 特定任务模式库
   - 推理路径记忆
```

这篇论文通过引入条件记忆作为新的稀疏轴，与MoE的条件计算形成互补，实现了在等参数和等FLOPs条件下的性能提升。其技术实现的核心在于现代化的N-gram嵌入设计，包括哈希冲突缓解、上下文感知门控、多分支集成等创新，同时利用确定性访问实现高效的系统级设计。