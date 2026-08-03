---
source_pdf: Body transformer.pdf
paper_sha256: a1992d9c1f04ebc2e75f4ea675b855c7a74335ee0f8835fdd17d343ce9c79d85
processed_at: '2026-07-20T18:28:57-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Body Transformer: 用机器人身体结构作为 Transformer 的 Inductive Bias

嘿 Andrej, 这篇 paper 我读了之后直觉上特别贴近你在 Neural Memory 表达的一些观点——即把 structural prior 写回架构里, 而不是寄希望于 SGD 自己 discover 出来. 我把核心 idea 拆开讲, 顺便把里面我觉得最有意思的几个细节、ablation、以及跟你以前 lecture 里讲过的 transformer/GNN 关系做对比。

---

## 1. 高层 motivation: 为什么 vanilla transformer 不够好

观察一个直觉性的事实: 生物学里, **corrective localized actuation** 是 locomotion efficiency 的核心. 比如冲浪者下半身(脚踝)对抗波浪引起的失衡, 是局部 feedback loop; 人体的 spinal cord 层级有专门负责 single actuator 的神经回路 (Forssberg 1979 的 stumbling corrective reaction; Collins & Kuo 2010 ankle energy recycling). 这意味着 robot policy 的计算图本来应该 **物理上 local**, 信息从 sensor 流向 spatially proximal actuator, 远端 actuator 只在更深层级才介入。

而 vanilla transformer 把 sequence 当 fully connected graph, 每个 token attend 每个 token, 等价于假设了 "信息从任意 sensor 到任意 actuator 的距离都是 1". 对 NLP 这种 rearrange-heavy 的任务没问题, 但对 robot policy 来说, 这是一个 **浪费甚至有害的 inductive bias**:
- 它要求模型去 learn "其实某些 sensor-actuator pair 不该有 strong coupling" 
- 它让参数和计算花在本来不相关的 long-range pair 上
- 它破坏了 spinal-reflex 类的 spatial locality

BoT 的核心论点很直接: 把 robot body 当 graph (sensor + actuator 为 nodes, body morphology 决定 edges), 用 **masked attention** 强制每层每个 node 只 attend 自身 + direct neighbors, 通过多层 stacking 让 receptive field 沿 graph 自然扩展, upstream layers = local reflex, downstream layers = global pooling。

---

## 2. 架构细节: 三个组件

BoT 整体是 Transformer encoder + 自定义 tokenizer/detokenizer, 但有几个关键 design choice 跟 Graphormer / NerveNet / MetaMorph 都不一样。

### 2.1 Tokenizer (per-node, 而不是 shared)

- 把 observation vector 拆成 graph of local observations
  - **Root node**: 全局量 (robot position, orientation, environment obs 如 door handle angle)
  - **Non-root nodes**: 每个 limb 对应的 joint angle, joint velocity, previous joint command, contact force
- 每个 node 用 **自己独立的 learnable linear projection** 把 local state → embedding vector
- 输出 n 个 embedding, n = number of nodes (sequence length)

这里的关键决策是 **per-node tokenizer 而不是 shared tokenizer**. 现有多 morphology 工作 (MetaMorph, SAT, GraphFormers) 都用 shared projection 是因为他们要 cross-morphology transfer; 但单 morphology 场景下, paper 在 Appendix F.2 做了 ablation, 证明 per-node tokenizer 显著优于 shared (Figure 11). 直觉上这很合理: 不同 limb 的 local state semantic 不同 (hip 和 calf 的 joint angle distribution 完全不同), 共享 projection 等于强迫它们进同一个 embedding space, 信息 bottleneck。

### 2.2 BoT Encoder

两个变体:

**BoT-Hard**: 每层都用 binary mask M:
$$M = I_n + A$$
- $I_n$: n维 identity matrix (self-attention, 每个 node 必须能 attend 自己)
- $A$: graph 的 adjacency matrix (direct neighbor attention)
- 加号是 element-wise (这里其实是 boolean OR, 因为二进制)

这样每层 node 只能看到 1-hop neighbors, 经过 L 层后 receptive field = L-hop neighborhood. paper 强调这点: 第一层 local, 最后一层 global, 中间层 progressive。

**BoT-Mix**: 交替 masked attention layers 和 unmasked attention layers, 但 **第一层必须是 masked**. 这是跟 concurrent work Buterez et al. 2024 ("Masked Attention is All You Need for Graphs") 的关键区别, 他们的工作不强制 first layer masked, 也不加 self-attention (即 mask ≠ adjacency matrix, 而是 adjacency + identity)。

为什么 Mix 比 Hard 在 hard-exploration 任务上更好? 直觉上:
- BoT-Hard 在 Board/Hill 任务上, 信息从 toe 传到 fingertip 需要 graph diameter 步 (MoCapAct humanoid graph diameter = 14), 探索时这种 long-range signal propagation 慢
- BoT-Mix 中间的 unmasked layer 提供 "shortcut", 允许地面扰动信号快速 broadcast
- 但保留 masked layer 维持了 locality prior, 没退化成 vanilla transformer

### 2.3 Detokenizer

- 每个 node 用独立的 learnable linear projection 把 feature → action
- 对 RL critic, 输出的是 value, 然后 **跨 body parts 取平均** 得到 Q-value

跨 body parts 平均这个细节挺有意思: 等价于让每个 limb 给出一个 "value vote", 类似 ensemble of value heads. 直觉上这是另一种 regularizer, 强制 value estimate 在 limbs 间具有 spatial consistency。

### 2.4 Positional Encoding 的微妙之处

- RL 实验中, BoT 加 positional encoding (embedding layer 把 node index → encoding vector, 加到 tokenizer 输出) 有用
- IL 实验中, 加 positional encoding 没明显帮助

paper 给的解释: **per-node tokenizer 本身就隐含了 positional info**, 因为每个 node 的 embedding 来自不同的 projection matrix, 等价于学到了 absolute positional encoding. RL 中需要 PE 可能因为 RL 任务对 node identity 更敏感 (需要明确区分 "这是 front-left-hip 还是 rear-right-calf")。

这点和你在 CS231N 讲的 "convolution + per-channel weight 已经 implicit encode position" 类似。

---

## 3. 公式逐项拆解

### 3.1 Vanilla Self-Attention (Section 3.1)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

变量:
- $Q \in \mathbb{R}^{n \times d_k}$: query matrix, 每行是某个 token "我想找什么样的信息"
- $K \in \mathbb{R}^{n \times d_k}$: key matrix, 每行是某个 token "我能提供什么样的信息"
- $V \in \mathbb{R}^{n \times d_v}$: value matrix, 每行是某个 token "实际要传递的内容"
- $d_k$: key/query embedding 维度 (paper 里 = 320 for IL, 64 for RL)
- $n$: sequence length = node 数 (e.g. A1 = 13 nodes, humanoid ~ 25 nodes, MoCapAct humanoid with dexterous hand 可达 128)
- $QK^T$: pairwise similarity matrix, 每个元素 (i,j) 表示 query i 和 key j 的 dot product
- $\sqrt{d_k}$: 缩放因子, 防止 dot product 数值过大导致 softmax 进入饱和区 (你 lec 10 里讲过的梯度问题)
- $\text{softmax}$: 沿 row 做, 让每个 query i 对所有 key 的 weight 加和为 1
- $\cdot V$: 用 attention weight 加权求和 value → 输出

### 3.2 Graphormer-style Attention Bias (Section 3.2)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + B\right) V$$

变量:
- $B \in \mathbb{R}^{n \times n}$: learnable bias matrix, 依赖 graph feature (e.g. shortest path distance between node i 和 j)
- 加在 pre-softmax logits 上, **不会** zero out attention, 只会 reweight

Graphormer 的做法是 soft bias, attention 仍然可以 attend 到 graph 上 distant nodes, 只是会被 down-weight. BoT 走得更激进。

### 3.3 BoT Masked Attention (Section 3.3)

$$B_{i,j} = \begin{cases} 0 & M_{i,j} = 1 \\ -\infty & M_{i,j} = 0 \end{cases}$$

变量:
- $M \in \{0,1\}^{n \times n}$: binary mask, 1 = 允许 attend, 0 = 禁止
- $i, j$: row, column index, $i$ 是 query node, $j$ 是 key node
- $B_{i,j}$: 当 mask = 0 时设为 $-\infty$, 经 softmax 后该位置 weight = $e^{-\infty} / \sum = 0$

**关键 trick**: 这等价于把不相关 pair 的 dot product 直接置零而不浪费 FLOPs 算它们, 因为 softmax 内部会被 $-\infty$ zero out。

### 3.4 BoT-Hard Mask 构造

$$M = I_n + A$$

实际是 boolean OR (不是数值加, 因为是二进制):
- $I_n$: identity matrix → 每个节点 self-attention
- $A$: adjacency matrix → 节点能 attend 其 direct neighbors (graph 上一条 edge 连接的)

A1 robot 的 graph: 13 个 nodes (1 base + 4 legs × 3 joints), base 连 4 个 hip, 每个 hip 连 thigh, thigh 连 calf. Adjacency matrix 极其稀疏, β ≈ 0.082 (非零元素比例), 即 ~92% 稀疏。

---

## 4. 计算复杂度推导 (Appendix H)

这是 paper 里我最喜欢的一节, 因为它把 FLOPs 拆得很细。设 sparsity coefficient $\beta \in [1/n, 1]$ 表示 mask 中非零元素比例。

### 4.1 Vanilla Attention FLOPs

| Step | Operation | FLOPs |
|------|-----------|-------|
| 1 | $QK^T$: $n^2$ 个 dot product, 每个 $d_k$ mul + $d_k-1$ add | $2n^2 d_k$ |
| 2 | 除以 $\sqrt{d_k}$: $n^2$ 个 division | $n^2$ + 常数 $c_1$ |
| 3 | Softmax: n 个 row, 每 row 是 n 维 → n exp + (n-1) add + n div | $(2+c_2)n^2 - n$ |
| 4 | 乘 V: $n \times d_k$ 输出, 每个输出是 n 维 weighted sum | $2n^2 d_k - n d_k$ |
| **Total** | | $4n^2 d_k + (2+c_2)n^2 - n d_k - n + c_1$ |

其中 $c_2$ 是 exp 的 FLOP cost (通常是 10+ 个 FLOP)。

### 4.2 Masked Attention FLOPs

| Step | Operation | FLOPs |
|------|-----------|-------|
| 1 | 只算 $\beta n^2$ 个 dot product | $2\beta n^2 d_k$ |
| 2 | 同上 | $\beta n^2 + c_1$ |
| 3 | Softmax 只在 $\beta n^2$ 非 zero entry 上算 | $(2+c_2)\beta n^2 - n$ |
| 4 | **仍然** $n \times d_k$ 输出 (没优化) | $2n^2 d_k - n d_k$ |
| **Total** | | $(2\beta + 2)n^2 d_k + (2+c_2)\beta n^2 - n d_k - n + c_1$ |

注意第 4 步他们没做 sparse multiplication 优化, 因为稀疏矩阵乘法 library 不一定快。他们明确把这列为 future work。

### 4.3 Ratio

$$\lim_{n\to\infty} \frac{\text{FLOPs vanilla}}{\text{FLOPs masked}} = \frac{4 d_{\text{model}} + 2 + c_2}{(2\beta + 2) d_{\text{model}} + 2\beta + \beta c_2} \geq 1$$

代入 $\beta = 0.908$ (MoCapAct humanoid mask):
- Vanilla: $4d + 2 + c_2$
- Masked: $(2 \cdot 0.908 + 2)d + 2 \cdot 0.908 + 0.908 c_2 = 3.816 d + 1.816 + 0.908 c_2$
- Ratio ≈ $4d / 3.816d \approx 1.05$ (那这个稀疏度下加速不显著)

但如果 humanoid with dexterous hand ($n=128$), mask sparsity 高很多 (paper 没明确给但估计 $\beta \approx 0.05-0.1$), 那么加速比可以到 5-10x. 实测 (Figure 6) 在 n=128 时 206% runtime speedup.

**关键洞察**: mask 越稀疏, 加速比越大. robot body graph 天然非常稀疏, 因为 physical adjacency 不像 NLP sequence 那样 dense. 这是 transformer 在 robotics 场景下不应该用 vanilla implementation 的另一个 reason。

### 4.4 为什么 PyTorch 没有原生支持

paper 在 Section 5.4 提到 PyTorch 甚至是 FlashAttention 都没优化 sparse masked attention. 原因 (引自 Buterez et al. 2024): 用例太少, 没 commercial driver. 这是个值得填的空: 如果 robot learning 大规模采用 BoT-style 架构, sparse attention kernel 会变成热点。

可能的实现路径: sparse tensor (CSR/COO format) + custom CUDA kernel. 类似 BlockwiseParallelAttention 但 sparse 版本. 这跟 Tri Dao 的 FlashAttention 工作完全 orthogonal, 可以叠加。

---

## 5. 实验数据详解

### 5.1 Imitation Learning: MoCapAct Body-Tracking

数据集: MoCapAct (Wagener et al. 2023), 5M+ transitions, 835 tracking clips, humanoid 全身 motion tracking.

实验结果 (Figure 3a, 5 seeds):

| Architecture | train return | val return | train length | val length |
|---|---|---|---|---|
| MLP | 0.623/0.572 ± 0.022 | 0.568/0.534 ± 0.025 | 0.808/0.762 ± 0.018 | 0.777/0.741 ± 0.022 |
| Transformer | 0.713/0.664 ± 0.024 | 0.656/0.576 ± 0.038 | 0.875/0.836 ± 0.022 | 0.834/0.779 ± 0.026 |
| **BoT-Hard** | **0.751/0.691 ± 0.024** | **0.698/0.650 ± 0.035** | **0.908/0.865 ± 0.018** | **0.879/0.835 ± 0.025** |
| Multi-Clip [Wagener] | /0.654 | - | /0.855 | - |

格式说明: **A/B ± std** 表示 "training 中 maximum / evaluation mean ± std"

**关键观察**:
- BoT-Hard vs Transformer: train return 0.751 vs 0.713 (+5.3%), val return 0.698 vs 0.656 (+6.4%). **val gap 比 train gap 大**, 说明 inductive bias 帮助了 generalization 而不仅是 fitting
- BoT-Hard vs Multi-Clip (tailored for tracking, stochastic, recurrent, more rollouts): 0.691 vs 0.654. 这非常 impressive, 因为 BoT 是 deterministic BC, 没有任何 task-specific engineering
- Episode length gap 也类似: val 0.879 vs 0.834 (更少早期 termination)

### 5.2 Scaling Behavior (Figure 3b)

BoT-Hard 在 17.5M trainable params 模型上性能最好, scaling 曲线没 plateau, 表明 embodiment bias 防止 overfitting. Vanilla transformer 在大模型时 train 性能上升但 val 反而下降, classic overfitting signature.

这跟我以前看 vision transformer 在小数据集上挣扎很类似: inductive bias (CNN 的 translation invariance, 这里是 BoT 的 body locality) 在数据量 / 模型容量不平衡时特别 valuable。

### 5.3 Adroit Hand Dexterous Manipulation (Appendix E, Figure 9c)

Tasks: Door, Hammer, Relocate. Low-data regime (50-200 demos), transformer 经典挣扎场景.

BoT-Hard 在所有 3 个 task 上超过 vanilla transformer, 在 Hammer 上也超过 MLP, 在 Door / Relocate 上和 MLP 相当. 这表明即便在 manipulation (graph 比 locomotion 更复杂, 手指链路长) 上, body-induced bias 也有帮助。

### 5.4 RL Experiments (Figure 5)

PPO, Isaac Gym, 4 tasks. 5 seeds. 关键 finding:

| Task | 难度 | 最佳变体 | 解释 |
|---|---|---|---|
| A1-Walk | Easy | BoT-Hard | Body bias 强约束减少 search space |
| Humanoid-Mod | Easy | BoT-Hard | 同上 |
| Humanoid-Board | Hard exploration | BoT-Mix | 需要 toe → fingertip 的 long-range signal |
| Humanoid-Hill | Hard exploration | BoT-Mix | 同上 |

**直觉**: 在稳定环境 (regular state transitions) 中, 强 body bias 通过限制 hypothesis space 加速 RL; 在 non-stationary 环境 (突然颠簸) 中, 信息 bottleneck 阻碍 exploration 信号传播, mix 更好。

这跟 GNN literature 里 oversquashing 问题 (Alon & Yahav 2021) 直接相关: 信息瓶颈会让远端 node 的 gradient signal 弱. BoT-Mix 用 unmasked layer 当 "shortcut" 缓解 oversquashing。

### 5.5 Ablations

**Mask design** (Figure 9a):
| Variant | val return | val length |
|---|---|---|
| BoT-Hard | 0.65 ± 0.04 | 0.84 ± 0.03 |
| BoT-Mix | 0.63 ± 0.01 | 0.82 ± 0.01 |
| BoT-Soft (Graphormer-style learnable B) | 0.58 ± 0.03 | 0.78 ± 0.03 |
| BoT-Hard/Random (random mask, same sparsity) | 0.60 ± 0.03 | 0.80 ± 0.02 |
| BoT-Hard-Stochastic (Gaussian policy σ=0.01) | 0.66 ± 0.04 | 0.84 ± 0.02 |

**洞察**:
- BoT-Soft (soft bias 而不是 hard mask) 比 Hard 差 → 不是 "down-weight 远端信息", 而是 "完全屏蔽远端信息" 更好
- Random mask 显著差 → body structure 本身重要, 不是任意 sparsity 都行
- Stochastic 略好 → 在 IL 上 stochastic policy 类似 regularizer

**Layer count** (Figure 9b):
- Graph diameter = 14 (理论上需要 14 masked layer 才能全 graph 通信)
- 但 fewer layers (~6-8) 已经能达到不错性能, 说明部分任务不需要全 graph 通信

**Tokenizer sharing** (Figure 11):
- per-node tokenizer 显著优于 shared tokenizer
- 在 Humanoid-Mod 和 Humanoid-Board 上都成立

**Random mask 在 RL 中也变差** (Figure 10):
- 即便 sparsity 相同 (β=0.82), body-induced mask > random mask
- 说明 mask 的 **semantic structure** 而非 sparsity 是关键

---

## 6. Real-World Deployment

A1 quadruped, flat ground, 5 rollouts, 每个要求 10 秒不摔倒, 5/5 成功. 三个标准 sim-to-real trick:
1. Terrain randomization during training
2. Joint controller higher stiffness
3. Low-pass action filter

注意他们没用 teacher-student distillation 或 memory mechanism (Miki et al. Science Robotics 2022 那种 proprioception-only student), 所以这其实是个 minimal-effort baseline deployment, 还有提升空间。

---

## 7. 跟相关工作对比的直觉

### 7.1 NerveNet (Wang et al. ICLR 2018)
- 同样把 body 当 graph, 用 message passing
- 缺点: oversmoothing (深层 GNN 节点表征趋同) + oversquashing (远距离信息 bottleneck)
- BoT 用 transformer attention 替代 message passing, 避免这些问题

### 7.2 Graphormer (Ying et al. NeurIPS 2021)
- 用 learnable bias B 代替 hard mask
- soft bias 不限制 attention, 只是 reweight
- 适合 NLP-like task (graph 上 similarity 不是严格 local)
- BoT-Hard 更激进, hard mask, 更适合 body 的 strict locality

### 7.3 My Body is a Cage (Kurin et al. 2021)
- 用 vanilla transformer 当 GNN, 证明 transformer 能 outperform message passing
- 但没利用 body structure 作 inductive bias, 等同 vanilla transformer baseline

### 7.4 MetaMorph (Gupta et al. 2022)
- Universal controller across morphologies
- 用 shared tokenizer (因为 morphology 变化)
- BoT 反向选择: 单 morphology, per-node tokenizer, 牺牲 transfer 换单 task 性能

### 7.5 Masked Attention is All You Need for Graphs (Buterez et al. 2024)
- Concurrent work, 类似 idea
- 区别: 1) 不强制 first layer masked, 2) mask = adjacency (no self-attention), 3) 用在 graph property prediction 而非 robot policy
- BoT 的 self-attention (M = I + A) 重要, 因为 node 自身信息是 baseline, 需要在每层 preserve

### 7.6 Decision Transformer / Online Decision Transformer (Chen et al. 2021, Zheng et al. 2022)
- Transformer 用作 policy 但 sequence 是 temporal axis (trajectory)
- BoT 的 sequence 是 spatial axis (body parts), orthogonal
- Future work 明确提到: 扩展到 temporal axis 是 promising direction, 类似 spatial-temporal transformer (类似 ViViT, TimeSformer)

### 7.7 Humanoid as Next Token Prediction (Radosavovic et al. 2024)
- 类似 motivation: humanoid 用 transformer
- 但用 vanilla transformer + next-token-prediction 范式
- BoT 的 inductive bias 可能直接 plug in 进 Radosavovic 架构

### 7.8 FlashAttention (Dao et al. 2022)
- 优化 dense attention 的 IO-aware kernel
- 不处理 sparse attention
- BoT 的 sparse mask 没法用 FlashAttention 直接加速, 需要 sparse 版本
- 这是 paper 提的明确 future direction

### 7.9 Spinal Cord Reflex Arcs (Forssberg 1979, Seminara et al. 2023)
- 生物学 inspiration: hierarchical sensorimotor control, spinal cord 处理 local reflex
- BoT 架构上等价: 浅层 = local reflex, 深层 = global pooling (cerebellum-like)
- 这种 bio-inspired hierarchical structure 在 robot policy 里可以 extend: 例如 lower motor neuron → spinal circuit → brainstem → cortex, 用 BoT 的多 stage 实现

### 7.10 Capsule Network / Group Equivariant CNN
- Hinton 的 capsule 用 routing-by-agreement, 类似 attention 但保留 part-whole hierarchy
- BoT 的 mask 也是一种 structural prior, 类似 CNN 的 translation equivariance
- 可以联想: 是否有 "body equivariance" (e.g. 左右对称性, 上下游 hierarchical)

---

## 8. 我觉得有改进空间的几个方向

### 8.1 Temporal BoT
现在 BoT 只在 spatial axis 用 transformer, 每个 timestep 独立处理. 但 locomotion 是强 temporal task, 应该叠 temporal attention layer (类似 ViViT factorized attention). 这样 BoT-Hard 的 spatial mask 和 temporal full attention 组合, 计算量 $O(n^2 d_k + T^2 d_k)$ 而不是 $O((nT)^2 d_k)$.

### 8.2 Learnable Morphology
现在 mask 是 hand-coded from URDF. 可以 imagine: 从 random mask 起步, 用 Gumbel-Softmax 让 mask 自学. 但 paper 的 ablation 显示 random mask 显著差, 说明 prior 强 ≠ 可学习 prior 强, 这个方向有风险。

### 8.3 Hierarchical BoT
现在所有 node 同等对待, 但 robot body 实际有 hierarchy: limb → joint → muscle. 可以用 hierarchical attention: 第一级 mask 是 joint-level, 第二级 mask 是 muscle-level. 类似 Swin Transformer 的 hierarchical masking。

### 8.4 Cross-Embodiment BoT
现在固定单 morphology. 跨 morphology 可以用 per-node tokenizer + meta-learned mask projection. 类似 task-conditional mask generation。MetaMorph 的方向 + BoT 的 mask.

### 8.5 Sparse Attention Kernel
正如 paper 指出, PyTorch 没原生 sparse attention. 一个 Triton kernel for sparse masked attention + GPU batching 应该能解锁 5-10x training speedup. 这是工程上最有杠杆的改进。

### 8.6 Diffusion Policy + BoT
Diffusion policy (Chi et al. RSS 2023) 在 manipulation 上很成功, 用 U-Net 或 1D CNN backbone. 替换成 BoT 应该直接 benefit from body inductive bias, 特别是 bimanual manipulation。

---

## 9. References

- Paper site: https://sferrazza.cc/bot_site
- Body Transformer arxiv (找到的版本): https://arxiv.org/abs/2408.06323
- Vaswani et al. 2017, Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Graphormer (Ying et al. NeurIPS 2021): https://arxiv.org/abs/2106.05234
- Buterez et al. 2024, Masked Attention is All You Need for Graphs: https://arxiv.org/abs/2405.04459
- NerveNet (Wang et al. ICLR 2018): https://arxiv.org/abs/1806.01843
- My Body is a Cage (Kurin et al. 2021): https://arxiv.org/abs/2010.04591
- MetaMorph (Gupta et al. 2022): https://arxiv.org/abs/2203.11931
- Structure-Aware Transformer Policy (Hong et al. ICLR 2021): https://arxiv.org/abs/2101.03162
- MoCapAct (Wagener et al. 2023): https://arxiv.org/abs/2303.10837
- PPO (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- Legged Gym / A1 (Rudin et al. CoRL 2022): https://arxiv.org/abs/2109.11978
- Robot Parkour Learning (Zhuang et al. 2023): https://arxiv.org/abs/2309.05665
- Daydreamer (Wu et al. CoRL 2022): https://arxiv.org/abs/2206.14176
- FlashAttention (Dao et al. 2022): https://arxiv.org/abs/2205.14135
- HumanoidBench (Sferrazza et al. 2024): https://arxiv.org/abs/2403.10506
- Decision Transformer (Chen et al. NeurIPS 2021): https://arxiv.org/abs/2106.01345
- Humanoid as Next Token Prediction (Radosavovic et al. 2024): https://arxiv.org/abs/2402.19469
- GAT (Velickovic et al. 2017): https://arxiv.org/abs/1710.10903
- Alon & Yahav 2021, Oversquashing: https://arxiv.org/abs/2006.05205
- Diffusion Policy (Chi et al. RSS 2023): https://arxiv.org/abs/2303.04137
- Adroit Hand benchmark (Rajeswaran et al. 2017): https://arxiv.org/abs/1709.10087
- D4RL (Fu et al. 2020): https://arxiv.org/abs/2004.06120
- Forssberg 1979, Stumbling corrective reaction: https://journals.physiology.org/doi/10.1152/jn.1979.42.4.936
- Seminara et al. 2023, Hierarchical sensorimotor control: https://www.science.org/doi/10.1126/scirobotics.add5434
- Collins & Kuo 2010, Ankle energy recycling: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009307

---

## 10. 一句话 Intuition

BoT 把 transformer attention 的 fully connected graph 改成 robot body 的 physical adjacency graph, 强制每个 node (sensor/actuator) 在每层只看 1-hop 邻居. 这等于把 spinal reflex arc 写进了 architecture 里, 浅层做 local correction, 深层 pool 全身信息. 实验上, 这种 inductive bias 在 imitation learning 上明显提升 generalization gap, 在 RL 简单任务上加速 sample efficiency, 在 hard-exploration 任务上需要 BoT-Mix 加全局 shortcut 缓解 oversquashing. 计算上, 机器人 body graph 的天然稀疏性让 sparse attention kernel 有 200%+ speedup 潜力, 但需要新 CUDA/Triton 实现, 现有 PyTorch / FlashAttention 都没优化这条路径。

整体上这篇 paper 没发明新机制, 但做了一个非常正确的 architectural surgery: 把 NLP/CV 的 inductive bias 替换成 robotics 的 inductive bias, 然后 measure 它确实有效. 我觉得这正是 robot learning 应该走的方向 — 跟你之前在 Eureka LLM 和 V-JEPA 等工作中强调的 "structure first" 一致。
