---
source_pdf: Reasoning by Superposition.pdf
paper_sha256: b5c363ad9b312cff6d34c18cb5fd03cfa328c069df51bb71e7185ba5a9b9114b
processed_at: '2026-08-11T21:33:32-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Reasoning by Superposition

好，抛开公式，用大白话讲讲这篇 paper 在说什么。

---

## 故事的开头：一个古老的争论

你训了一个 LLM，让它做 reasoning。经典做法是 chain-of-thought (CoT) — 让它一步步输出中间 token，"先想 A，再想 B，所以答案是 C"。

然后有人问了：**为啥非得用文字想？能不能在脑子里想？**

COCONUT [Hao et al. 2024] 就试了一下：不输出 token，直接把 transformer 每一步的 hidden state 喂回去当下一步输入。叫 "continuous CoT" 或者 "latent reasoning"。

实验发现：在 graph reasoning 任务上，这个 latent 版本居然比 discrete token 版本强很多。但 **为啥强？没人说得清。**

这篇 paper 就是来填这个坑的。

参考: https://arxiv.org/abs/2412.06769

---

## 任务：走迷宫

简化一下，任务是这样的：

给你一个 directed graph（有向图），一堆 edge，一个起点 $r$，两个候选终点 $c_1, c_2$。问：$r$ 能走到 $c_1$ 还是 $c_2$？

这玩意儿叫 **graph reachability**。听起来简单，实际上它是很多 reasoning 任务的 worst case — 比如 knowledge graph 查询、planning、theorem proving 的某些 instance 都能 reduce 到这个。

---

## Discrete CoT 怎么做：DFS，一步一步走

如果用传统 discrete CoT，模型每一步必须 output 一个 token，表示 "我现在走到哪个 node 了"。

想象你在一个迷宫里，每一步只能站在一个地方。你从 $r$ 出发，走第一条路，撞墙了，backtrack，再试另一条。这是 **DFS (depth-first search)**。

最坏情况你要把所有 path 都试一遍，每条 path 长度最多 $n$（node 总数），path 数量可以到 $O(n)$，所以总共 $O(n^2)$ 步。

Merrill & Sabharwal 2023 严格证明了：constant-depth transformer + discrete CoT 确实需要 $\tilde{O}(n^2)$ 步才能解 directed reachability。

参考: https://arxiv.org/abs/2310.07923

---

## Continuous CoT 怎么做：BFS，一整层一起走

continuous CoT 的 magic 在于：**每一步的 hidden vector 不止表示一个 node，它同时表示一堆 node。**

作者管这叫 **superposition state**，借用量子力学的词儿。

具体来说，第 $c$ 步的 hidden vector $[t_c]$ 长这样：

$$[t_c] = \frac{1}{\sqrt{|\mathcal{V}_c|}} \sum_{v \in \mathcal{V}_c} u_v$$

人话翻译：**把所有 "从 $r$ 出发 $c$ 步以内能走到的 node" 的 embedding 加起来，归一化。**

- $\mathcal{V}_c$ = 从 $r$ 出发 $c$ 步内可达的 node 集合
- $u_v$ = node $v$ 的 token embedding
- $1/\sqrt{|\mathcal{V}_c|}$ = 归一化系数，让 vector 长度保持为 1

这就是一个 vector 编码了一整层 BFS frontier！

---

## 关键直觉：一个 vector 怎么编码多个 node？

这可能是最反直觉的部分。你问：一个 $d$ 维 vector 怎么同时表示 100 个 node？

答案：**靠 inner product 做 "membership test"**。

假设所有 node embedding 是 orthonormal（互相垂直），那么：

$$\langle [t_c], u_v \rangle = \begin{cases} \frac{1}{\sqrt{|\mathcal{V}_c|}} > 0 & \text{if } v \in \mathcal{V}_c \\ 0 & \text{otherwise} \end{cases}$$

也就是说，拿 $[t_c]$ 和某个 node embedding 做内积，非零就说明这个 node 在 frontier 里，零就说明不在。

**一次内积，同时检查了一个 node 是否在 frontier 里。** 这跟 quantum computing 里的 superposition 思路一样：state 在测量前是叠加的，测量时才 "回答" 具体问题。

---

## 那怎么从 $\mathcal{V}_c$ 扩展到 $\mathcal{V}_{c+1}$？BFS 的一步

现在 $[t_c]$ 表示了 "$c$ 步内可达的所有 node"。下一步要扩展：找到所有从 $\mathcal{V}_c$ 出发的 edge，把 target 加进来。

transformer 怎么做？用 attention。

**Query**：当前 thought $[t_c]$（在 content space）
**Key**：每个 edge 的 source node embedding（存在 edge token `<e>` 的 buffer 里）
**Value**：每个 edge 的 target node embedding

attention score 就是 query 和 key 的内积。由于前面说的 orthonormal 性质：

$$\text{attention}([t_c] \to \text{edge}_i) \propto \mathbb{1}\{s_i \in \mathcal{V}_c\}$$

**翻译成人话**：当前 frontier 的 superposition 和某个 edge 的 source 做内积，非零当且仅当这个 source 在当前 frontier 里。

于是 attention 自动 **同时** 聚合了所有 "从当前 frontier 出发的 edge" 的 target。一次 attention 操作 = 一次 BFS layer expansion。

这就解释了为啥 continuous 比 discrete 强：**discrete 一次只能选一个 edge，continuous 一次 attention 处理所有 edge。**

---

## 然后要 clean up：MLP 当过滤器

attention 之后，$[t_c]$ 里混了原来的 $\mathcal{V}_c$ 和新加的 target，但权重不均匀，还有 softmax 带来的 noise。

MLP 的作用是：
1. 把 vector 投影到 "每个 node 一个坐标" 的 space
2. 阈值化：小于 $\varepsilon$ 的当 noise 删掉
3. 投影回来
4. LayerNorm 归一化

公式上：
- $\mathbf{W}_1$ 把 vertex embedding rotate 到 standard basis
- $\sigma(x) = \mathbb{1}\{x \geq \varepsilon\}$ 是 hard threshold（实际用 GELU 近似）
- $\mathbf{W}_2$ rotate 回来

结果就是干净的 $\mathcal{V}_{c+1}$ superposition。

---

## 两层 transformer 够了

整理一下整个 construction：

**Layer 1（5 个 attention heads）**：每个 edge token `<e>` 从前面两个位置把 source 和 target 抄过来，存到自己的 buffer 里。这一层是 "预处理"，把 edge 信息准备好。

**Layer 2（1 个 attention head）**：当前 thought $[t_c]$ 用 attention 找到所有 "source 在 frontier 里" 的 edge，把 target 聚合过来，形成 $\mathcal{V}_{c+1}$ 的 superposition。

**MLP**：清理 noise，归一化。

**重复 $D$ 步**（$D$ = graph diameter），最终 $[t_C]$ 包含了所有从 $r$ 可达的 node。然后 `<A>` token 做最后一次 "测量"：看 $c_1$ 和 $c_2$ 哪个在 superposition 里，输出那个。

**这就是 Theorem 1：2-layer transformer + $D$ 步 continuous thought 解 directed reachability。**

---

## 为啥 discrete 做不到同样的事？

discrete CoT 每一步必须 output 一个 token，比如 "node_42"。这相当于把 superposition 坍缩到一个具体 node 上。

你想用 discrete 模拟 BFS？你得逐个 node 报数："node_1, node_5, node_12, ..."。一个 frontier 有 $O(n)$ 个 node，你要 $O(n)$ 步报完一层。$D$ 层就是 $O(nD) = O(n^2)$。

continuous 的优势：**一个 vector 一次性表示整层 frontier，不用逐个报数。** $D$ 步搞定。

这是 expressivity 上的根本差异，不仅仅是 "省点 token"。

---

## 实验部分：理论真的对得上吗？

作者训了一个 **2-layer GPT-2**（$d=768$, 8 heads），在 ProsQA 数据集上跑。

结果：

| 方法 | Layers | Accuracy |
|------|--------|----------|
| No CoT | 2 | ~50%（瞎猜） |
| Discrete CoT | 2 | ~75% |
| Discrete CoT | 12 | ~83% |
| **COCONUT (continuous)** | **2** | **~100%** |

2-layer continuous 碾压 12-layer discrete。expressivity gap 实锤了。

---

## 最酷的部分：mechanistic interpretability

作者不只是看 accuracy，还打开模型看了看里面到底在干嘛。

**Layer 1 attention**：确实每个 `<e>` token 都在 attend 自己的 source 和 target，跟理论 construction 一模一样。

**Layer 2 attention**：按 edge 类型分组统计 attention score —

- Not Reachable edges：~0.04（几乎没 attention）
- Reachable edges：~2.12（强 attention）
- Frontier edges（恰好 $i$ 步可达）：~2.12 → 1.00 → 0.67 → 0.61（随步数衰减，符合 BFS）
- Optimal edges（在最优路径上）：~2.54 → 1.72 → 1.67 → 2.23（持续高，因为 curriculum 训练 bias）

模型确实在做 parallel BFS。

**Continuous thought 的 inner product**：拿 $[t_c]$ 和每个 node embedding 算内积 —

- Reachable node 的 inner product 远高于 Not Reachable
- Frontier node 高于普通 Reachable
- Optimal node 最高

superposition state 真实存在，且自动涌现。

---

## 最 mysterious 的发现

作者还做了个对照实验 COCONUT-BFS：训练时不在 optimal path 上监督，而是从 frontier 里随机选一个 node 监督。

结果 COCONUT-BFS 也达到 ~100% accuracy，且 superposition 行为几乎一样。

**这说明：superposition BFS 不需要显式监督，它是 architecture + gradient descent 的 emergent property。**

你只告诉模型 "最终答案是 $c_{i^*}$"，模型自己学会了内部做 BFS。这跟你经常说的 "transformer 学到的 algorithm 比 surface supervision 更深" 完全一致。

这是 paper 留的最大 open problem：**为啥 gradient descent 会自发找到 BFS 而非 DFS 或别的？**

---

## Positional Encoding 的 trick

有个技术细节值得提一下：transformer 怎么知道 "看回 2 个位置"？

答案：利用 sinusoidal positional encoding 的 rotation 性质。

sinusoidal PE 里，position $i$ 和 position $i+\ell$ 的关系是一个固定 rotation $R^{(\ell)}$：

$$\bar{p}_{i+\ell} = R^{(\ell)} \bar{p}_i$$

因为 $\cos((i+\ell)\omega) = \cos(\ell\omega)\cos(i\omega) - \sin(\ell\omega)\sin(i\omega)$，这就是 2D rotation。

作者构造了一个 "attention chooser"：用 query 和 key 的 PE 部分做 rotation matching，让特定 token 自动 attend 到特定相对位置。这个 trick 也能扩展到 RoPE（Appendix B.6）。

参考: 
- Sinusoidal PE: https://arxiv.org/abs/1706.03762
- RoPE: https://arxiv.org/abs/2104.09864

---

## 跟其他东西的联系

### 跟 quantum computing 的类比

continuous thought = superposition state（测量前）
discrete token = collapsed state（测量后）
final prediction = measurement

这个类比其实挺深的。quantum computing 的优势就是 superposition + interference 做并行计算。continuous CoT 在某种意义上是 "classical superposition" — 用 high-dimensional vector 模拟。

### 跟 GNN 的等价性

每一步 continuous thought 做的事 = 一次 graph message passing。

$[t_c]$ 是当前 frontier 的 indicator vector（soft 版）。attention 相当于 adjacency matrix 乘以这个 vector：$A \cdot [t_c]$。这就是 GNN 的一层。

所以 **transformer + continuous CoT ≈ GNN with $D$ layers**。这给出了一个 unifying perspective。

参考: GNN message passing — https://arxiv.org/abs/1704.01212

### 跟你关注的方向

你最近讲 latent reasoning、讲 o1-style reasoning 的本质。这篇 paper 给出的启示：

1. **Latent reasoning 的优势是 real expressivity gain**，不是 "省 token" 这么简单。superposition 让一个 vector 编码整个 search frontier。

2. **Discrete CoT 可能是 reasoning 的 bottleneck**。强制每步 commit 到一个 token，相当于强制 DFS，丢失了 parallel exploration。

3. **未来的 reasoning model 可能是 hybrid**：部分 step 用 continuous thought 做 broad exploration，关键时刻用 discrete token 做 commit。

4. **Mechanistic interp 的方法论**：按 "Reachable / Frontier / Optimal" 分组分析 attention 和 representation，这套方法可以套到任何 search-based reasoning task 上。

参考你自己的工作：
- Grokked transformers as implicit reasoners: https://arxiv.org/abs/2405.15071
- Token assorted (hybrid latent + discrete): https://arxiv.org/abs/2502.03275

---

## 局限性（必须说）

1. **Construction 需要 orthonormal embedding**，即 $d_{TE} \geq n$。graph 大了 embedding 维度要线性增长，不 scale。

2. **只证明了 upper bound**，没证明 discrete CoT 一定需要 $\Omega(n^2)$。可能有更聪明的 discrete 算法。

3. **Hard threshold MLP** 是理想化构造，实际用 GELU，理论 gap 没完全 bridge。

4. **只在 graph reachability 上证明**，其他 reasoning task（math、code）是否也享受 superposition 优势？open。

5. **训练 dynamics 的 mystery**：为啥只监督 optimal path 就能涌现 BFS？paper 承认不知道，列为 future work。

---

## 一句话总结

**Continuous CoT 把 search frontier 编码成 vector superposition，一次 attention 等于一次 BFS layer expansion，所以 $D$ 步解 reachability；discrete CoT 每步必须 commit 到一个 node，只能串行 DFS，需要 $O(n^2)$ 步。**

这个 expressivity gap 是 fundamental 的，不是 "省几个 token" 的工程优化。

而且最漂亮的是：这个 superposition mechanism 是训练自动涌现的，不需要你显式告诉模型 "做 BFS"。gradient descent 自己找到了。

---

如果你想我再展开某部分 — 比如 RoPE 版的 attention chooser 怎么构造、训练 dynamics 为啥涌现 BFS、跟 spectral graph theory 的关系 — 告诉我。这 paper 里头藏的 mechanistic 细节比 main text 表面看到的深得多。

---

# Reasoning by Superposition: 深度讲解

Andrej, 这篇 paper 我觉得非常对胃口 — 它恰好打在你最近关心的几个点上：latent reasoning、CoT 的本质、expressivity 的理论边界、以及训练 dynamics 与理论 construction 的对齐。下面我把这篇 paper 拆开讲，重点是 build intuition。

---

## 1. 核心直觉：Superposition vs Collapse

这篇 paper 的核心 thesis 一句话就能说清楚：**continuous CoT 之所以比 discrete CoT 强，是因为 continuous thought vector 可以同时表示多个搜索状态（superposition state），而 discrete token 必须坍缩到一个状态上（collapsed state）。**

作者借用了 quantum mechanics 的类比 [Böhm, 2013]：
- **Superposition state**：$\lvert\psi\rangle = \sum_v \alpha_v \lvert v\rangle$，一个 vector 同时编码多个 vertex
- **Collapsed state**：测量后 $\lvert\psi\rangle \to \lvert v^*\rangle$，必须 commit 到某一个 vertex

在 graph reachability 上，这意味着：
- **Discrete CoT** ≈ DFS / greedy search：每一步必须选一个 token，错了就要 backtrack，复杂度 $O(n^2)$ steps
- **Continuous CoT** ≈ parallel BFS：每一步同时维护整个 frontier，所有可达 vertex 同时被激活，复杂度 $D$ steps（$D$ = graph diameter）

类比到 RL / search algorithm 上，这就像你经常说的 "tree of thoughts" 但用 vector 一次性把整个 frontier 编码进去，branching factor 不再是 sequential bottleneck。

参考：
- COCONUT 原始 paper: https://arxiv.org/abs/2412.06769
- Wei et al. CoT: https://arxiv.org/abs/2201.11903
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

---

## 2. Problem Setup: Directed Graph Reachability

### 2.1 任务定义

给定一个 directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
- $\mathcal{V} = \{v_1, ..., v_n\}$：n 个 vertex（每个对应 vocabulary 中一个 token）
- $\mathcal{E} = \{e_1, ..., e_m\}$：m 条 directed edge，$e_i = (s_i, t_i)$
- Root node $r$，两个 candidate destination $c_1, c_2$（保证恰有一个 reachable from $r$），目标预测 $c_{i^*}$

任务本质是 **transitive closure 的查询问题**，它在理论上包含了 Turing machine halting problem、knowledge graph traversal、planning 等许多 reasoning 任务的 worst case。

### 2.2 Prompt 格式

```
<s>  s_1 t_1 <e>  s_2 t_2 <e>  ...  s_m t_m <e>  <q>  c_1 c_2  <R>  r  [t_1] [t_2] ... [t_C]  <A>
```

- 每条 edge 用 3 个 token 编码：source $s_i$, target $t_i$, special `<e>` 标记
- Prompt 长度 $t_0 = 3m + 6$（BOS + 3m edges + `<q>` + $c_1$ + $c_2$ + `<R>` + $r$）
- $[t_c]$ 是第 c 个 continuous thought（d 维 vector）
- `<A>` 触发最终预测

### 2.3 为什么这个问题 representative

Graph reachability 是 **TC^0-complete 的 parallel problem** 的代表 [Merrill & Sabharwal 2023a]，constant-depth transformer 无法直接解决，需要 CoT 来 boost expressivity。Merrill & Sabharwal 证明了 constant-depth transformer + discrete CoT 需要 $\tilde{O}(n^2)$ steps 才能解决 directed reachability。这篇 paper 把这个 bound 砍到 $D$ steps（$D < n$，且经常 $D \ll n$），用的是 continuous CoT。

---

## 3. 关键 Building Block: Attention Chooser

### 3.1 想解决的问题

Transformer 是 sequence model，要让它做 graph reasoning，必须先解决一个 mechanistic 问题：**怎么让特定 token 的 attention 关注到特定相对位置？**

通常 attention 是 content-based 的（query 和 key 算 inner product），但 graph 算法需要 "看回 i-2 个位置那个 source node" 这种 position-based 的 hard-coded pattern。

### 3.2 构造思路

**Lemma 1 (Attention Chooser)**：对任意 token $\langle x\rangle$ 和相对位置 $\ell \geq 0$，存在 $K, Q \in \mathbb{R}^{(2d_{PE}) \times d}$ 使得：
- 如果 $h_i = u_{\langle x\rangle}$，则位置 $i$ 的 attention 全部集中在位置 $i - \ell$
- 否则 attention 集中在位置 1（BOS，即 attention sink）

核心 trick 是利用 sinusoidal PE 的 rotation 性质。对 sinusoidal PE：

$$\bar{p}_{i, 2j-1} = \cos(i \cdot M^{-2j/d_{PE}}), \quad \bar{p}_{i, 2j} = \sin(i \cdot M^{-2j/d_{PE}})$$

变量解释：
- $i$：position index（从 1 开始）
- $j \in [d_{PE}/2]$：PE 维度的 pair index（cos, sin 一对）
- $M$：base 常数，原 paper 取 $10^4$
- $d_{PE}$：positional encoding 维度
- $\bar{p}_{i, 2j-1}, \bar{p}_{i, 2j}$：position $i$ 的 PE vector 第 $2j-1$ 和 $2j$ 维

**关键性质（Lemma 4）**：存在 rotation matrix $R^{(\ell)}$ 使得 $\bar{p}_{i+\ell} = R^{(\ell)} \bar{p}_i$。这是因为每个 2D pair $(\cos, \sin)$ 实际上是一个 rotation，平移 $\ell$ 步就是再 rotate $\ell \cdot \omega^j$ 角度。

### 3.3 Query / Key 构造

$$\mathbf{Q} = \begin{bmatrix} \xi \cdot 0 & 0 & 0 & I_{d_{PE}} \\ \xi (\bar{p}_1 \otimes \tilde{u}_{\overline{x}})^\top & 0 & 0 & 0 \end{bmatrix}, \quad \mathbf{K} = \begin{bmatrix} 0 & 0 & 0 & \eta R^{(\ell)} \\ 0 & 0 & 0 & \eta I_{d_{PE}} \end{bmatrix}$$

变量解释：
- $\xi, \eta > 0$：scaling 超参（$\eta$ 大就 sharpening attention，$\xi$ 大就让 "non-$\langle x\rangle$" 走 sink 分支）
- $\tilde{u}_{\overline{x}} = \sum_{v \neq \langle x\rangle} \tilde{u}_v$：所有非 $\langle x\rangle$ token embedding 的 superposition
- $\bar{p}_1$：position 1（BOS）的 PE vector
- $R^{(\ell)}$：平移 $\ell$ 的 rotation matrix

得到：
- $\mathbf{q}_i = [\bar{p}_i; \xi \langle \tilde{u}_{\overline{x}}, \tilde{h}_i\rangle \bar{p}_1]$
- $\mathbf{k}_j = [\eta R^{(\ell)} \bar{p}_j; \eta \bar{p}_j] = [\eta \bar{p}_{j+\ell}; \eta \bar{p}_j]$

inner product：
$$\langle \mathbf{q}_i, \mathbf{k}_j \rangle = \eta \left(\langle \bar{p}_i, \bar{p}_{j+\ell}\rangle + \xi \langle \tilde{u}_{\overline{x}}, \tilde{h}_i\rangle \langle \bar{p}_1, \bar{p}_j\rangle\right)$$

**两个分支**：
- 若 $h_i = u_{\langle x\rangle}$：$\langle \tilde{u}_{\overline{x}}, \tilde{h}_i\rangle = 0$，第二项消失，attention 完全由 $\langle \bar{p}_i, \bar{p}_{j+\ell}\rangle$ 主导，max 在 $j = i - \ell$
- 若 $h_i \neq u_{\langle x\rangle}$：$\langle \tilde{u}_{\overline{x}}, \tilde{h}_i\rangle \geq 1$（由 orthonormal embedding 保证），第二项主导，max 在 $j = 1$（BOS）

这就是 "attention chooser"：根据当前 token content 选择相对位置 attend 到哪里。这个 mechanism 让 transformer 可以实现 "if 当前是 `<e>`，看回 2 个位置找 source node" 这种 hard-coded graph reasoning pattern。

参考：
- Vaswani et al. Attention Is All You Need: https://arxiv.org/abs/1706.03762
- RoPE paper: https://arxiv.org/abs/2104.09864

---

## 4. 核心 Lemma: Superposition State 的维护

这是整篇 paper 的核心。

### 4.1 形式化陈述

**Lemma 2**: 对每个 $c \geq 0$，存在 transformer 参数 $\theta$ 使得：

$$[t_c] = \frac{1}{\sqrt{|\mathcal{V}_c|}} \sum_{v \in \mathcal{V}_c} u_v$$

变量解释：
- $[t_c] \in \mathbb{R}^d$：第 c 步 continuous thought（d 维 vector）
- $\mathcal{V}_c$：从 root $r$ 出发 **c 步以内** 可达的 vertex 集合（c-step reachable frontier）
- $|\mathcal{V}_c|$：该集合大小
- $u_v$：vertex $v$ 的 token embedding
- $1/\sqrt{|\mathcal{V}_c|}$：L2 归一化系数，保证 $\|[t_c]\|_2 = 1$（利用 $U^\top U = I_V$ 的 orthonormality）

这就是 superposition state：一个 vector 同时编码了所有可达 vertex。在几何上，它是这些 vertex embedding 的 normalized sum，相当于在 $|\mathcal{V}_c|$ 维子空间里的 "中心向量"。

### 4.2 Inductive 构造：两层 Transformer 的分工

构造 proof 用 induction，每一步需要做两件事：
1. **取当前 frontier $\mathcal{V}_c$，找出所有从 $\mathcal{V}_c$ 出发的 edge，把 target 加进来形成 $\mathcal{V}_{c+1}$**（一次 BFS expansion）
2. **过滤掉 noise，重新归一化**

#### Layer 1: Edge context aggregation（5 个 attention heads）

第一层用 5 个 attention chooser heads：
- $h_0 = (\langle e\rangle, 2)$：当 token 是 `<e>` 时 attend 到 $i-2$（即 source $s_i$）
- $h_1 = (\langle e\rangle, 1)$：当 token 是 `<e>` 时 attend 到 $i-1$（即 target $t_i$）
- $h_2 = (\langle R\rangle, 2)$：当 token 是 `<R>` 时 attend 到 $c_1$ 
- $h_3 = (\langle R\rangle, 1)$：当 token 是 `<R>` 时 attend 到 $c_2$
- $h_4 = (\langle A\rangle, 1)$：当 token 是 `<A>` 时 attend 到最后一个 $[t_C]$

效果：每个 `<e>` token 的 representation 现在 buffer 里同时存了 source 和 target：
$$h_{\text{Idx}(\langle e\rangle, i)}^{(1)} = \frac{1}{\sqrt{3}}[\tilde{u}_{\langle e\rangle}^\top, \tilde{u}_{s_i}^\top, \tilde{u}_{t_i}^\top, 0]^\top$$

embedding 维度 $d = 3d_{TE} + d_{PE}$ 被切成四块：
- 前 $d_{TE}$ 维：content space（存原始 token identity）
- 中 $d_{TE}$ 维：buffer 1（存 source node）
- 后 $d_{TE}$ 维：buffer 2（存 target node）
- 最后 $d_{PE}$ 维：positional encoding

#### Layer 2: Frontier expansion（1 个 attention head）

第二层是关键。Query 来自当前 $[t_c]$（在 content space），Key 来自每个 `<e>` token 的 buffer 1（即 source node embedding）。

$$\langle q_{[t_c]}, k_{\langle e\rangle, i}\rangle = \left\langle \frac{1}{\sqrt{|\mathcal{V}_c|}} \sum_{v \in \mathcal{V}_c} \tilde{u}_v, \frac{\tau}{\sqrt{3}} \tilde{u}_{s_i}\right\rangle$$

变量解释：
- $q_{[t_c]}$：位置 $[t_c]$ 的 query vector，等于 content 部分
- $k_{\langle e\rangle, i}$：第 i 个 `<e>` token 的 key，等于 $\tau/\sqrt{3}$ 乘上 buffer 1（即 source embedding）
- $\tau > 0$：scaling 因子，控制 attention 的 sharpness

由于 token embedding orthonormal：
$$\langle q_{[t_c]}, k_{\langle e\rangle, i}\rangle = \frac{\tau}{\sqrt{3|\mathcal{V}_c|}} \cdot \mathbb{1}\{s_i \in \mathcal{V}_c\}$$

**直觉**：当前 frontier $\mathcal{V}_c$ 的 superposition 与某个 source $s_i$ 的 inner product，非零当且仅当 $s_i \in \mathcal{V}_c$。这就是 superposition state 的 magic — **一次 inner product 同时检测了所有 $|\mathcal{V}_c|$ 个 vertex**，并行而非串行。

Value 是 buffer 2（即 target $t_i$），output 写到 content space。结果：
$$h_{[t_c]}^{(1.5)} = \frac{1}{\sqrt{|\mathcal{V}_c|}} \sum_{v \in \mathcal{V}_c} \tilde{u}_v + \frac{1}{\sqrt{3}|\mathcal{T}_c|} \sum_{j: s_j \in \mathcal{V}_c} \tilde{u}_{t_j} + \text{noise}$$

其中 $\mathcal{T}_c = \{j : s_j \in \mathcal{V}_c\}$，第二项就是 $\Delta_c$（从 $\mathcal{V}_c$ 出发能一步到达的所有 vertex），加上原来的 $\mathcal{V}_c$ 就是 $\mathcal{V}_{c+1}$。

#### MLP: Filter & Equalize

经过 attention 后有两个问题：
1. Attention softmax 后所有 position 都有非零 weight，引入 noise
2. 不同 vertex 的 weight 不均匀（有的来自 $1/\sqrt{|\mathcal{V}_c|}$，有的来自 $1/(\sqrt{3}|\mathcal{T}_c|)$）

MLP 构造：
- $\mathbf{W}_1 = [\tilde{u}_1, ..., \tilde{u}_V]^\top$：把 vertex basis rotate 到 standard basis
- $\sigma(x) = \mathbb{1}\{x \geq \varepsilon\}$：coordinate-wise thresholding（hard filter）
- $\mathbf{W}_2 = \mathbf{W}_1^\top$：rotate 回来

效果：$\sigma(\mathbf{W}_1 h) = \sum_v \mathbb{1}\{\lambda_v \geq \varepsilon\} e_v$，把 superposition coefficient 阈值化。然后 $\mathbf{W}_2$ 把它转回 vertex embedding space，最后 LayerNorm 归一化得到 $\frac{1}{\sqrt{|\mathcal{V}_{c+1}|}} \sum_{v \in \mathcal{V}_{c+1}} u_v$。

变量解释：
- $\varepsilon$：threshold（取 $\frac{1}{4n}$，n 为 vertex 数），略大于 noise level $\frac{1}{16n}$，远小于真实 signal $>\frac{1}{2n}$
- $\lambda_v$：vertex $v$ 在当前 superposition 中的 coefficient

---

## 5. Main Theorem: D 步解决 Graph Reachability

**Theorem 1**：存在一个 2-layer transformer（参数 $O(|Voc|)$ 维），对任意 directed graph with $n_{max}$ vertices，对任意 $C >$ graph diameter $D$：

$$\widetilde{TF}_{\theta, C, W_O}(h_{[t_0]}) = c_{i^*}$$

也就是说 **$D$ 步 continuous thought 解决 reachability**。对比 Merrill & Sabharwal 的 $\tilde{O}(n^2)$ discrete CoT 步数上界。

**为什么是 D 而不是更小**？因为 BFS 一层一层扩展，最远 vertex 需要恰好 $D$ 步才能被纳入 superposition。每一步对应 BFS 的一层。

**为什么 discrete CoT 做不到**？discrete token 必须在每一步 commit 到一个具体 vertex，相当于 BFS 一次只 expand 一个 vertex，需要 $O(n)$ 步扩展完一层（一个 frontier），总共 $O(n \cdot D) = O(n^2)$（最坏情况 $D = n$）。Continuous CoT 利用 superposition 把同层 frontier 一次性编码，每个 step 处理一整层，所以只需要 $D$ 步。

参考：
- Merrill & Sabharwal 2023: https://arxiv.org/abs/2310.07923
- Li et al. (CoT inherently serial): https://arxiv.org/abs/2402.12875

---

## 6. Architecture 图解析（Figure 2 & 3）

### Figure 2: Embedding Space 与 Layer 1

```
Embedding layout (d = 3*d_TE + d_PE):
+---------+---------+---------+---------+
| content | buffer1 | buffer2 |   PE    |
+---------+---------+---------+---------+
   d_TE      d_TE     d_TE     d_PE

Layer 1 attention flow:
  <e> token (at pos i) → attends to:
    - pos i-2 (source s_i) → copied to buffer1
    - pos i-1 (target t_i) → copied to buffer2
  <R> token → attends to:
    - c_1 (at 3m+3) → buffer1
    - c_2 (at 3m+4) → buffer2  
  <A> token → attends to last [t_C] → buffer1 (保存 final superposition)
```

### Figure 3: Layer 2 Frontier Expansion

```
        Query (from content of [t_c])         Keys (buffer1 of each <e>)
        = sum_{v in V_c} u_v / sqrt(|V_c|)    = u_{s_i} / sqrt(3)
                        ↓
                Inner product = (τ/√(3|V_c|)) * 1{s_i ∈ V_c}
                        ↓
              Softmax → attention concentrated on edges from frontier
                        ↓
              Value (buffer2 of <e>) = u_{t_i}
                        ↓
        Output writes back to content: sum of u_{t_i} for matching edges
                        ↓
        Combined with residual: V_c + Delta_c = V_{c+1}
                        ↓
              MLP filters noise + LayerNorm → normalized superposition
```

---

## 7. 实验验证

### 7.1 Setup

- **Model**: GPT-2 style decoder, **2 layers**, $d_{model}=768$, 8 heads
- **Dataset**: ProsQA 子集 (3-4 hop reasoning)，统计：
  | Split | #Problems | $|V|$ | $|E|$ | Sol. Len |
  |-------|-----------|--------|--------|----------|
  | Train | 14785     | 22.8   | 36.5   | 3.5      |
  | Val   | 257       | 22.7   | 36.3   | 3.5      |
  | Test  | 419       | 22.7   | 36.0   | 3.5      |
- **Training**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight-decay $10^{-2}$), lr=$10^{-4}$
- **Curriculum**: Stage $i$ teaches model to use $i$ continuous thoughts before predicting $i$-th CoT node. 25 epochs/stage, 300 epochs total. Mixing previous stage data with prob 0.1.

### 7.2 主要结果（Figure 4）

| Method | Layers | Heads | Accuracy |
|--------|--------|-------|----------|
| No CoT | 2 | 8 | ~50% (random) |
| CoT | 2 | 8 | ~75% |
| CoT* | 12 | 12 | ~83% |
| **COCONUT** | **2** | **8** | **~100%** |

2-layer COCONUT 碾压 12-layer CoT。这是 expressivity 维度的明显 separation。

### 7.3 Mechanistic Interpretability

#### Layer 1 Attention 验证（Figure 5）

实测的 attention map 显示 `<e>` token 几乎 100% attention 到 source 和 target，符合 theoretical construction。这个 copying mechanism 是训练自动涌现的，没有显式监督。

#### Layer 2 Attention 分类（Table 1）

将 edges 分 4 类，统计 Layer 2 attention score（mean ± std）：

| Edge Group | Step 1 | Step 2 | Step 3 | Step 4 |
|------------|--------|--------|--------|--------|
| Not Reachable | 0.04±0.07 | 0.03±0.09 | 0.08±0.17 | 0.12±0.20 |
| Reachable | 2.12±1.07 | 0.71±0.92 | 0.38±0.72 | 0.29±0.66 |
| - Frontier | 2.12±1.07 | 1.00±0.96 | 0.67±0.87 | 0.61±0.95 |
| - Optimal | 2.54±1.03 | 1.72±1.13 | 1.67±1.20 | 2.23±1.35 |

分类定义：
- **Reachable**: source 在当前 reachable 集合内
- **Frontier** ⊂ Reachable: source 恰好 $i$ 步可达（在当前 frontier 上）
- **Optimal** ⊂ Frontier: source 在 optimal reasoning chain 上

观察：
1. Reachable edges 远高于 Not Reachable — 符合 superposition theory
2. Frontier > 普通 Reachable — 训练信号让模型偏向 frontier（新鲜节点）
3. Optimal > 普通 Frontier — curriculum 学习让模型偏向 optimal path（虽然 superposition 仍然包含所有 frontier）

#### Continuous Thought Representation（Figure 6）

计算 $[t_i]$ 与每个 vertex embedding $u_v$ 的 inner product，按节点分类画 histogram：

- **Reachable** nodes 的 inner product 远高于 **Not Reachable**
- **Frontier** nodes 的 inner product 高于普通 Reachable（验证 BFS-like expansion）
- **Optimal** nodes 的 inner product 最高（验证 curriculum supervision 偏向）

#### COCONUT-BFS 对照实验

去掉 curriculum 中的 optimal 偏向，改为从 frontier 中均匀随机采样 supervision。结果：
- COCONUT-BFS 也达到 near-perfect accuracy
- inner product 分布与原 COCONUT 类似（除了 Optimal 不再有偏向）

**关键发现**：superposition 的 BFS 行为不依赖 supervision 形式 — 即使只在 optimal path 上监督，模型仍然 internally 学到了 BFS-like parallel search。这是 emergent behavior，是 gradient descent + architecture inductive bias 的产物。

参考：
- ProsQA dataset (in COCONUT paper): https://arxiv.org/abs/2412.06769
- Mechanistic Interpretability: https://transformer-circuits.pub/

---

## 8. 联想与 Open Questions

### 8.1 与 Quantum-inspired Computing 的连接

Superposition 这个词用得有讲究。在 quantum computing 里：
- Quantum parallelism: 一次操作同时作用在所有 basis state 上
- Amplitude amplification: Grover 算法用 superposition + interference 加速搜索
- Measurement collapse: 测量后 superposition 坍缩到一个 state

这篇 paper 的 framework 完全可以套用：
- Continuous thought = pre-measurement state（amplitude 在各 vertex 上分布）
- Discrete token = post-measurement state（坍缩）
- Final `<A>` prediction = measurement（argmax over vocabulary）

这给出了一个潜在的研究方向：**CoT 的 amplitudes 能否用 interference 来增强 correct path，抑制 wrong path**？类似 quantum attention 的思路。我猜你可能已经在 latent reasoning 的方向上想到过类似的事情。

### 8.2 与 grokking 的联系

你写过的 "grokked transformers are implicit reasoners" [Wang et al. 2024a] 显示 grokking 后 transformer 学到了 in-context reasoning algorithm。这里 2-layer COCONUT 的训练过程也很类似 — 从 memorization 到 algorithmic reasoning 的 phase transition。COCONUT 的 superposition representation 可能就是 grokking 后的内部 circuit。

参考: https://arxiv.org/abs/2405.15071

### 8.3 与 Mega-Token / Latent Reasoning 的关系

COCONUT 是 "thinking in latent space" 的代表。最近的方向：
- Pause tokens [Goyal et al. 2023]: https://arxiv.org/abs/2310.02226
- Quiet-STaR: https://arxiv.org/abs/2403.09629
- Token Assorted [Su et al. 2025]: https://arxiv.org/abs/2502.03275
- Reflection / o1-style reasoning

这篇 paper 给出了这些方法的 **理论 backing**：latent space reasoning 的优势来自 superposition capacity，而非仅仅 "more compute per step"。

### 8.4 关于训练 Dynamics 的 Mystery

Section 5.4 的 COCONUT-BFS 实验暴露了一个理论 gap：**为什么只在 optimal path 上监督，模型能学会 BFS-like superposition？**

可能的解释方向：
1. **Gradient descent inductive bias**: 找到 optimal path 的最简单 circuit 是先做 BFS，再 select optimal
2. **Loss landscape**: superposition 是更 flat 的 minima，generalization 更好
3. ** lottery ticket hypothesis**: 网络初始化里就有 BFS subcircuit，gradient 只需 amplify

这是 paper 留的 future work，我觉得是最有意思的 open problem。

### 8.5 Position Encoding 的通用性

这篇 paper 一个 underappreciated 的贡献是 **construction 适用于 sinusoidal 和 RoPE 两种主流 PE**。之前的 expressivity 工作（Merrill, Feng 等）通常构造 problem-specific PE，这篇用 sinusoidal PE 的 rotation 性质 + RoPE 的 relative position 性质，构造了 attention chooser。

Appendix B.6 给出 RoPE 版本，trick 是用 $\bar{p}_{-T}^{\widetilde{TE}}$ (一个 "lookback" reference position) 来实现 sink 机制。这让 paper 的 construction 更接近真实 LLM (Llama 系列用 RoPE)。

### 8.6 Spectral Methods 的 Connection

Cohen et al. 2025 [https://arxiv.org/abs/2502.08794] 显示 2-layer transformer 可以用 **line graph 的 spectral decomposition** 解 shortest path。Superposition 也可以从 spectral 角度看：continuous thought 是 adjacency matrix 的 polynomial power 应用。

具体地，若 $A$ 是 adjacency matrix，则 $A^c \mathbf{e}_r$ 给出从 $r$ 出发 $c$ 步到达的 vertex superposition（未归一化）。Continuous CoT 实际上是在每一层做一次 $A \cdot$ (current superposition)，相当于一次 matrix-vector multiplication。

这给出一个统一视角：
- **Discrete CoT** = 串行 matrix-vector multiplication，每次只看一列
- **Continuous CoT** = 一次 matrix-vector multiplication 完整做

这与 graph neural network 的 message passing 机制完全等价。Transformer + continuous CoT ≈ GNN with $D$ layers。这个 connection 我觉得可以作为 unifying framework 的起点。

### 8.7 长上下文与 Buffer Space

Paper 的 construction 用了 $3d_{TE} + d_{PE}$ 的 embedding，分 content/buffer1/buffer2/PE 四块。这暗示一个 architecture design principle：**embedding space 应该有 explicit 的 working memory slots**。

实际 LLM 里这些 slot 是 implicit 的（vector 的不同 dimension 自动分化）。但 explicit 的 buffer space 设计可能让 latent reasoning 更稳定。联想到：
- Universal Transformer 的 adaptive computation
- RWKV 的 state vector
- Mamba 的 selective state space

这些都在做 "explicit working memory"，COCONUT 的 buffer space 给出了理论上的最优 allocation。

参考: 
- Universal Transformer: https://arxiv.org/abs/1807.03819
- Mamba: https://arxiv.org/abs/2312.00752

---

## 9. 局限与 Critical Thoughts

### 9.1 Construction 是非 constructive upper bound

Theorem 1 证明 **存在** 参数使 2-layer transformer 解决 reachability。但没证明 lower bound，即没证明 $D-1$ 步不够。也没有证明 discrete CoT 一定需要 $\Omega(n^2)$ — Merrill & Sabharwal 给的是 upper bound $\tilde{O}(n^2)$，可能存在更聪明的 discrete CoT 算法。

### 9.2 Superposition 容量有限制

Lemma 2 假设 token embedding $U$ orthonormal，意味着 $d_{TE} \geq |\mathcal{V}| = n$。当 graph 很大时，embedding 维度需要线性增长。实际 LLM 里 $d_{TE} \ll |Voc|$，superposition 是 approximate 的，noise 会累积。

这暗示 COCONUT 在 $n$ 较小时（如几千个 vertex）可能工作良好，但 scale 到 huge graph 时可能 break。是不是 RL 的 state space 太大就解释了 COCONUT 在某些 planning task 上的失败？

### 9.3 MLP 的 Hard Threshold 不现实

Construction 里 $\sigma(x) = \mathbb{1}\{x \geq \varepsilon\}$ 是 hard threshold。实际训练用的是 GELU/ReLU，是 soft 版本。Paper 在实验部分验证了 GELU 也能工作，但理论上没完全 bridge 这个 gap。这是个 open problem：soft nonlinearity 是否能 perfect 实现 superposition maintenance？

### 9.4 单 hop per step 的限制

Construction 里每个 continuous thought step 对应 BFS 的一层。但实际 reasoning 里我们可能想 "skip ahead" 或 "merge steps"。Paper 没讨论 multi-hop-per-step 的可能性。COCONUT 训练里 step 数是 hyper-parameter（paper 用 3-4 步），这个数字怎么选仍是个 art。

---

## 10. 对你（Andrej）研究方向的可能启发

考虑到你最近关注的 LLM reasoning、mechanistic interpretability、education（"Software 2.0"）、eureka labs 等方向，这篇 paper 可能给你的启发：

1. **Teaching latent reasoning**: COCONUT 的 superposition intuition 可以做成教学 demo — visualize continuous thought 与 graph frontier 的 inner product 演化，非常直观地展示 "parallel BFS in vector space"。

2. **Building reasoning models**: 当前主流 reasoning model (o1, R1, QwQ) 都是 discrete CoT。Continuous CoT 的 superposition 优势 + 训练 dynamics 的 emergent BFS 是 alternative path。如果你做新模型，可以考虑 hybrid (discrete + latent) 设计。

3. **Mechanistic studies**: Section 5 的 interpretability 方法（按 Reachable/Frontier/Optimal 分组分析 attention 和 inner product）是个好 template，可以套到其他 reasoning task 上。

4. **Theory-driven architecture**: 这篇 paper 给出了 "为什么 2-layer + continuous > 12-layer + discrete" 的 mechanistic 解释。这种理论 + 实验 + interp 的三件套可能是 reasoning research 的范式。

---

## 11. 总结

这篇 paper 的贡献浓缩成三点：

1. **Expressivity separation**: 2-layer transformer + $D$ 步 continuous CoT 解决 directed reachability，对比 discrete CoT 需要 $\tilde{O}(n^2)$ 步。$D$ 是 diameter，远小于 $n$。

2. **Mechanism**: continuous thought 是 superposition state，同时编码所有 c 步内可达 vertex，每步执行一次 parallel BFS expansion。Discrete token 是 collapsed state，必须串行搜索。

3. **Empirical validation**: 2-layer COCONUT 在 ProsQA 上达 near-perfect accuracy，远超 12-layer CoT。Mechanistic analysis 验证了 superposition 的存在 — Layer 1 复制 edge context，Layer 2 扩展 frontier，MLP 过滤 noise，且这一 circuit 自动涌现，无需显式监督。

---

## 12. 我觉得最值得深入想的问题

1. **Lower bound**: discrete CoT 是否真的需要 $\Omega(n^2)$ 步？Merrill & Sabharwal 给的 $\tilde{O}(n^2)$ 是 construction 上界，可能存在更聪明算法。如果 lower bound 也是 $\Omega(n^2)$，那 continuous CoT 的 $D$ 步才是真正的 exponential gap。

2. **Training dynamics theory**: 为什么只在 optimal path 监督就能学会 BFS-like superposition？这是 deep learning 里 "implicit curriculum" 的一个 clean 实例，可能有 generalized explanation。

3. **Beyond reachability**: 对 shortest path、connectivity、flow 等 graph 问题，continuous CoT 的步数上界是什么？Cohen et al. 用 spectral 方法做 shortest path，superposition 方法是否有不同 trade-off？

4. **Approximate superposition**: 当 $d_{TE} < n$ 时 superposition 是 approximate 的，如何分析 noise 累积？这关系到 COCONUT 在真实大 model 上的 scaling。

5. **Multi-modal superposition**: 当前 superposition 是 single-layer BFS。multi-modal distribution（不同 path 有不同 mode）能否用一个 vector 编码？这关系到 probabilistic reasoning 的 latent implementation。

---

如果你想我深入展开任何一部分（比如 Appendix B 的完整 proof、RoPE 版本的 attention chooser、spectral 视角、或者训练 dynamics 的 hypothesis），告诉我。这 paper 的 Appendix 实际上比 main text 信息量更大，特别是 Proposition B.1-B.4 的 explicit 参数构造，里面有很多可以聊的 mechanistic 细节。

参考链接汇总：
- 主 paper (Reasoning by Superposition): https://arxiv.org/abs/2412.06769 (注：实际是 COCONUT 的链接，这篇应该是其后续工作，arxiv 编号待查)
- COCONUT: https://arxiv.org/abs/2412.06769
- Merrill & Sabharwal CoT expressivity: https://arxiv.org/abs/2310.07923
- Merrill & Sabharwal log-depth: https://arxiv.org/abs/2503.03961
- Feng et al. (CoT mystery): https://arxiv.org/abs/2305.11854 (NeurIPS 2023)
- Wei et al. CoT: https://arxiv.org/abs/2201.11903
- Li et al. (inherently serial): https://arxiv.org/abs/2402.12875
- Cohen et al. spectral shortest path: https://arxiv.org/abs/2502.08794
- Wang et al. grokked implicit reasoners: https://arxiv.org/abs/2405.15071
- Vaswani et al. Attention: https://arxiv.org/abs/1706.03762
- Su et al. RoPE: https://arxiv.org/abs/2104.09864
- Goyal et al. Pause tokens: https://arxiv.org/abs/2310.02226
- Su et al. Token Assorted: https://arxiv.org/abs/2502.03275
- Pfau et al. Think dot by dot: https://arxiv.org/abs/2404.15758
- Sanford et al. transformer graph algorithms: NeurIPS 2024
- Universal Transformer: https://arxiv.org/abs/1807.03819
- Mamba: https://arxiv.org/abs/2312.00752
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
