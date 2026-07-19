---
source_pdf: BAB-ND.pdf
paper_sha256: c8a69ded33cf151e1fb24d7ebf0da8f11745f0adefdaaed6315042e6d9b88334
processed_at: '2026-07-18T13:13:17-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BaB-ND: Branch-and-Bound + Neural Dynamics 深度讲解

## 1. 这篇 paper 在解决什么 problem

核心痛点：**neural dynamics model (ND) 预测能力很强，但 planning 极其困难**。

给定一个学到的 dynamics model $f_{\text{dyn}}$ (一个 NN)，我们要解：

$$
\min_{\{u_t \in \mathcal{U}\}} \sum_{t=t_0}^{t_0+H} c(\hat{x}_t, u_t) \quad \text{s.t.} \quad \hat{x}_{t+1} = f_{\text{dyn}}(\hat{x}_t, u_t)
$$

变量解释：
- $t_0$: 当前时间步
- $H$: planning horizon (长程规划)
- $\hat{x}_t$: 时间步 $t$ 的 (predicted) state，其中 $\hat{x}_{t_0} = x_{t_0}$ 已知
- $u_t \in \mathcal{U} \subset \mathbb{R}^k$: 机器人 action，每个 time step 一个 $k$ 维 action
- $c(\cdot, \cdot)$: step cost
- $f_{\text{dyn}}$: pretrained NN，吃 (state, action) 吐 next state

把所有动力学约束递归地代入 objective，问题被压成一个纯约束优化问题：

$$
\min_{\mathbf{u} \in \mathcal{C}} f(\mathbf{u})
$$

其中：
- $\mathbf{u} = \{u_{t_0:t_0+H}\} \in \mathcal{C} \subset \mathbb{R}^d$ 是 flattened action sequence
- $d = k \cdot H$ 是总维度 (比如 $k=2, H=20$ 就是 $d=40$)
- $\mathcal{C}$ 是 box constraint $\{\mathbf{u} \mid \underline{\mathbf{u}} \leq \mathbf{u} \leq \overline{\mathbf{u}}\}$
- $f$ 是 scalar objective，**内含一个 H 层深度的 NN 复合** (因为 $f_{\text{dyn}}$ 被 unroll 了 $H$ 次)

**难点**：
- $f$ 严重 non-convex (NN 的 ReLU 嵌套)
- $H$ 大 → $d$ 大 → sampling 维度爆炸
- contact-rich 任务里 $c$ 经常 non-smooth (含 ReLU 障碍项)，feasible region non-convex

## 2. Existing 方法的困局

| 方法 | 优点 | 致命缺陷 |
|---|---|---|
| **CEM / MPPI** (sampling-based) | GPU 友好，flexible | high-dim 时 sample coverage 指数下降，无 systematic search |
| **Gradient Descent** | 简单 | stuck in local optima；non-smooth cost 处处不可导 |
| **MIP** (Liu et al. 2023b, NeurIPS) | 全局最优，sound | 要求 NN 用 sparse ReLU；难以 scale 到大 NN 和大 $H$，CPU 上跑 |
| **GCS** (Marcucci, Graesdal et al.) | contact-rich 强 | 不支持 NN dynamics |

**关键 observation**: 神经网络 verification 社区已经有一套非常成熟的 BaB + bound propagation 框架 (α,β-CROWN)，**它们解决的本质问题完全一样**：在非凸 NN 复合函数上做带约束的全局优化。区别只在于 verification 只需证 lower bound，而 planning 还要找 feasible solution。

## 3. BaB (Branch and Bound) 的核心思想 — 1D toy case

考虑 $f(\mathbf{u})$ 在 $\mathcal{C} = [-1, 1]$ 上找 $\mathbf{u}^*$ 使 $f$ 最小。

**四个步骤循环**：
1. **Sample (searching)**: 在 $\mathcal{C}$ 上撒点 (orange dots)，得到一个 upper bound $\overline{f}^*$ (因为任何采样点都 ≥ 真正最优 $f^*$)
2. **Branch**: 把 $\mathcal{C}$ 二分成 $\mathcal{C}_1, \mathcal{C}_2$
3. **Bound**: 在每个 subdomain 上估计 $f$ 的 **lower bound** $\underline{f}_{\mathcal{C}_i}^*$ (用线性函数 underestimate $f$)
4. **Prune**: 如果某 subdomain 的 $\underline{f}_{\mathcal{C}_j}^* > \overline{f}^*$，说明这里面找不到比当前 best 更好的解，直接砍掉

迭代下去 $\overline{f}^* \to f^*$ 收敛。**和 AlphaGo 的 MCTS、A* search 思想同源** — 都是 "tree + value/bound + prune"。

## 4. BaB-ND 的三大组件详解

### 4.1 Branching Heuristics — 选哪个 subdomain、怎么切

#### (a) 选 subdomain: `batch_pick_out(P, n)`

从候选 set $\mathbb{P}$ 里选 $n$ 个 subdomain 来 split。BaB-ND 提出 **exploitation + exploration 混合** 策略：

- **Exploitation**: 按 $\overline{f}_{\mathcal{C}_i}^*$ (upper bound) 升序排，取前 $n_1$ 个 — 已经找到好解的地方，继续挖
- **Exploration**: 从剩下 $N$ 个里 softmax 采样 $n - n_1$ 个，概率：

$$
p_i = \frac{\exp(-\underline{f}_{\mathcal{C}_i, \text{scaled}}^* / T)}{\sum_{j=1}^{N} \exp(-\underline{f}_{\mathcal{C}_j, \text{scaled}}^* / T)}
$$

变量解释：
- $\underline{f}_{\mathcal{C}_i, \text{scaled}}^*$: 对 $\underline{f}_{\mathcal{C}_i}^*$ 做 min-max normalization (数值稳定)
- $T$: temperature，小 $T$ 偏向 lower bound 小的 (更 promising)，大 $T$ 更 uniform
- 负号: lower bound 越小越有潜力

**为什么 verification 不需要这个**：verification 要 cover 所有 subdomain，顺序无所谓；planning 要快速找到好 feasible solution，必须 prioritize。

#### (b) 切 subdomain: `batch_split({C_i})`

对每个 box subdomain $\mathcal{C}_i = \{\mathbf{u} \mid \underline{\mathbf{u}}_j \leq \mathbf{u}_j \leq \overline{\mathbf{u}}_j, j=0,\ldots,d-1\}$，沿某维度 $j^*$ 二等分：

$$
\mathcal{C}_i^{\text{lo}} = \{\mathbf{u} \mid \underline{\mathbf{u}}_{j^*} \leq \mathbf{u}_{j^*} \leq \tfrac{\underline{\mathbf{u}}_{j^*} + \overline{\mathbf{u}}_{j^*}}{2}\}
$$
$$
\mathcal{C}_i^{\text{up}} = \{\mathbf{u} \mid \tfrac{\underline{\mathbf{u}}_{j^*} + \overline{\mathbf{u}}_{j^*}}{2} \leq \mathbf{u}_{j^*} \leq \overline{\mathbf{u}}_{j^*}\}
$$

**怎么选 $j^*$?** 朴素方法是选 range 最大的维度 $(\overline{\mathbf{u}}_j - \underline{\mathbf{u}}_j)$。BaB-ND 更聪明：

- 把 searching 过程中 **top $w\%$ 最好的 samples** 拿出来
- 对每个维度 $j$，统计有多少 top sample 落在左半边 ($n_j^{\text{lo}}$) 和右半边 ($n_j^{\text{up}}$)
- 计算 $|n_j^{\text{lo}} - n_j^{\text{up}}|$ — 分布越不平衡，说明这个维度对 objective 越敏感
- 排序指标：$(\overline{\mathbf{u}}_j - \underline{\mathbf{u}}_j) \cdot |n_j^{\text{lo}} - n_j^{\text{up}}|$，选最大那个维度切

**直觉**: range 大说明"这里很不确定"，sample 分布偏说明"好的解集中在某一侧"，两者结合能最大概率把好解和坏解分离到不同 subdomain，让 prune 发挥作用。

### 4.2 Bounding — 估计 lower bound (核心创新)

这是 paper 最技术性的部分，借鉴自 **CROWN** (Zhang et al. 2018, NeurIPS)。

#### 4.2.1 CROWN 背景

考虑一个 $L$ 层 MLP：
- $f(\mathbf{x}) := \mathbf{z}^{(L)}$
- $\mathbf{z}^{(i)} = \mathbf{W}^{(i)} \hat{\mathbf{z}}^{(i-1)} + \mathbf{b}^{(i)}$ (linear)
- $\hat{\mathbf{z}}^{(i)} = \sigma(\mathbf{z}^{(i)})$ (activation, 这里 ReLU)
- $\hat{\mathbf{z}}^{(0)} = \mathbf{x}$

变量：$\mathbf{W}^{(i)} \in \mathbb{R}^{d_i \times d_{i-1}}$ 是第 $i$ 层权重，$\mathbf{b}^{(i)} \in \mathbb{R}^{d_i}$ 是 bias，$\mathbf{z}^{(i)}$ 是 pre-activation，$\hat{\mathbf{z}}^{(i)}$ 是 post-activation。

**Lemma B.1 — ReLU 的 linear relaxation**: 给定 $\mathbf{l} \leq \mathbf{z} \leq \mathbf{u}$ (element-wise, 这是 pre-activation bound)，则 $\hat{\mathbf{z}} = \text{ReLU}(\mathbf{z})$ 可以被 linear bound 包围：

$$
\underline{\mathbf{D}} \mathbf{z} + \underline{\mathbf{b}} \leq \hat{\mathbf{z}} \leq \overline{\mathbf{D}} \mathbf{z} + \overline{\mathbf{b}}
$$

其中 $\underline{\mathbf{D}}, \overline{\mathbf{D}} \in \mathbb{R}^{d \times d}$ 是对角矩阵，第 $j$ 个对角元：

$$
\underline{\mathbf{D}}_{j,j} = \begin{cases} 1 & \text{if } \mathbf{l}_j \geq 0 \text{ (active)} \\ 0 & \text{if } \mathbf{u}_j \leq 0 \text{ (inactive)} \\ \alpha_j & \text{if } \mathbf{l}_j < 0 < \mathbf{u}_j \text{ (unstable)} \end{cases}
$$

$$
\overline{\mathbf{D}}_{j,j} = \begin{cases} 1 & \text{if } \mathbf{l}_j \geq 0 \\ 0 & \text{if } \mathbf{u}_j \leq 0 \\ \frac{\mathbf{u}_j}{\mathbf{u}_j - \mathbf{l}_j} & \text{if unstable} \end{cases}
$$

变量解释：
- Active: $\mathbf{l}_j \geq 0$，ReLU 在 $[\mathbf{l}_j, \mathbf{u}_j]$ 内是 identity，斜率 1，bias 0
- Inactive: $\mathbf{u}_j \leq 0$，ReLU 恒为 0
- Unstable: $\mathbf{l}_j < 0 < \mathbf{u}_j$，ReLU 跨过原点，要用直线 **上推/下托**
  - 下界斜率 $\alpha_j \in [0,1]$ 是 free parameter (优化时可以调，这是 α-CROWN 名字由来)
  - 上界斜率 $\frac{\mathbf{u}_j}{\mathbf{u}_j - \mathbf{l}_j}$ 是从 $(\mathbf{l}_j, 0)$ 到 $(\mathbf{u}_j, \mathbf{u}_j)$ 的弦

**Theorem B.2 — CROWN bound propagation**: 通过 backward 传播一个线性 bound：

$$
f(\mathbf{x}) \geq \widetilde{\underline{\mathbf{W}}}^{(1)} \mathbf{x} + \widetilde{\underline{\mathbf{b}}}^{(1)}
$$

其中 $\widetilde{\underline{\mathbf{W}}}^{(l)}, \widetilde{\underline{\mathbf{b}}}^{(l)}$ 递归定义 (从 $L$ 往 $1$ 传)：

$$
\widetilde{\underline{\mathbf{W}}}^{(l)} = \underline{\mathbf{A}}^{(l)} \mathbf{W}^{(l)}, \quad \widetilde{\underline{\mathbf{b}}}^{(l)} = \underline{\mathbf{A}}^{(l)} \mathbf{b}^{(l)} + \underline{\mathbf{d}}^{(l)}
$$

$$
\underline{\mathbf{A}}^{(L)} = \mathbf{I}, \quad \widetilde{\underline{\mathbf{b}}}^{(L)} = 0
$$

$$
\underline{\mathbf{A}}^{(l)} = (\widetilde{\underline{\mathbf{W}}}_{+}^{(l+1)}) \underline{\mathbf{D}}^{(l)} + (\widetilde{\underline{\mathbf{W}}}_{-}^{(l+1)}) \overline{\mathbf{D}}^{(l)}
$$

**直觉**: 当传到一个 ReLU 层，对于上一层权重矩阵的每个元素：
- 如果 $\widetilde{\underline{\mathbf{W}}}_{i,j}^{(l+1)} \geq 0$ (正系数)，要 lower bound 这个 ReLU 输出，就取它的下界 $\underline{\mathbf{D}} \mathbf{z} + \underline{\mathbf{b}}$
- 如果 $\widetilde{\underline{\mathbf{W}}}_{i,j}^{(l+1)} < 0$ (负系数)，要 lower bound 就用上界 $\overline{\mathbf{D}} \mathbf{z} + \overline{\mathbf{b}}$ (因为负×大=小)

下标 $\geq 0$ 表示 "取正元素，其余置零"，$<0$ 反之。

**Theorem B.3 — Concretization via Hölder's inequality**: 给定 $\mathcal{C} = \mathbb{B}_p(\mathbf{x}_0, \epsilon) = \{\mathbf{x} \mid \|\mathbf{x} - \mathbf{x}_0\|_p \leq \epsilon\}$，最终 lower bound：

$$
\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x}) \geq -\epsilon \|\widetilde{\underline{\mathbf{W}}}^{(1)}\|_q + \widetilde{\underline{\mathbf{W}}}^{(1)} \mathbf{x}_0 + \widetilde{\underline{\mathbf{b}}}^{(1)}
$$

其中 $\frac{1}{p} + \frac{1}{q} = 1$。这里用了 Hölder 不等式 $\|\mathbf{a}\mathbf{x}\| \leq \|\mathbf{a}\|_q \|\mathbf{x}\|_p$，把 $\max_{\|\lambda\|_p \leq 1} \widetilde{\underline{\mathbf{W}}}^{(1)} \lambda$ 算出来。

#### 4.2.2 直接用 CROWN 在 planning 上为什么不行

两个致命问题：

**问题 1 — Loose bound (vacuous)**: CROWN 每过一层 ReLU，linear relaxation 引入误差。planning 把 $f_{\text{dyn}}$ unroll $H$ 次 (e.g. $H=20$)，相当于一个超深 NN，误差累积到 lower bound 比任何 sample 的 upper bound 都大，**完全无法 prune**。

**问题 2 — Quadratic cost**: 原 CROWN 算 intermediate (pre-activation) bounds 时要 recursive 调用，每个中间层 bound 都得 propagate 回 input，时间复杂度 $O(L^2)$。$H$ 大时算不动。

#### 4.2.3 BaB-ND 的两个补救方案

**Approach 1 — Propagation early-stop**

不要 propagate 到 input！只 propagate 到中间某个 layer $v$ (称为 early-stop node)，然后用该 layer 的 empirical bound 做 concretize。

形式上，把 $f$ 解析成 computational graph $\mathcal{G} = (\mathbf{V}, \mathbf{E})$：
- 节点 $v \in \mathbf{V}$: 每个 math operation (ReLU, matmul, add, ...)
- 边 $e = (w, v) \in \mathbf{E}$: 数据流
- $\mathcal{T} = \{v \mid \text{In}(v) = \emptyset\}$: input nodes (action $\mathbf{u}$, constants, params)
- $o$: output node (scalar $f$)

Algorithm 3 (bound propagation with early-stop):
1. 从 output $o$ 用 BFS 反向走
2. 遇到 early-stop node $v \in \mathcal{S}$ 时 **不再向其 input 传播**
3. 在 $\mathcal{S} \cup \mathcal{T}$ 上用各自已知 bounds concretize

**直觉**: 与其把误差传 $L \cdot H$ 层得到 vacuous bound，不如只传几层得到一个**紧但可能不 sound** 的 bound。Planner 不需要 sound guarantee，只需要能 rank subdomain。

**Approach 2 — Search-integrated bounding**

中间 layer 的 pre-activation bounds 怎么来？原 CROWN 是 recursive 算。BaB-ND 直接**用 searching 过程中的 samples 来估**：

设有 $M$ 个 sample $\mathbf{u}^m$，layer $v$ 的输出函数 $\mathbf{g}_v(\mathbf{u})$，则：

$$
\mathbf{l}_v = \min_{m} \mathbf{g}_v(\mathbf{u}^m), \quad \mathbf{u}_v = \max_{m} \mathbf{g}_v(\mathbf{u}^m)
$$

这些是 **empirical bounds**，可能 underestimate 真实 range (因为 sample 不一定覆盖到极值)，但作为 guidance 够用了。

**关键 insight (要 build intuition)**: verification 要 soundness (必须 $f^* \geq \underline{f}^*$ provable)，planning 不要 — 只需要 **ranking 正确**：promising subdomain 的 lower bound 应该小于 unpromising 的。这种"松但快"的策略让 BaB-ND 跳出了 verification 的 strict 框架，可以 early-stop + empirical bound，复杂度从 $O(L^2)$ 降到接近 $O(L)$ 甚至 $O(1)$ (取决于 $\mathcal{S}$ 怎么选)。

### 4.3 Searching — 在 subdomain 里找 feasible solution

`batch_search(f, {C_i})` 用 CEM (cross-entropy method) 在每个 subdomain 里找最好的 $\tilde{\mathbf{u}}$。

返回 $\{(\overline{f}_{\mathcal{C}_i}^*, \tilde{\mathbf{u}}_{\mathcal{C}_i})\}$，其中：
$$
\overline{f}_{\mathcal{C}_i}^* := \min_{\mathbf{u}_k \in \mathcal{C}_i} f(\mathbf{u}_k)
$$

**关键设计**:
- searching 顺带 record 中间 layer outputs $\mathbf{g}_v(\mathbf{u}^m)$ → 喂给 Approach 2 算 empirical bounds
- subdomain 越来越小后，sample coverage 越来越好，CEM 容易找到 near-optimal
- 上一轮 subdomain 的 best solution $\tilde{\mathbf{u}}_{\mathcal{C}_i}$ 直接初始化下一轮 split 出的至少一个 subdomain 的 search

**这是 verification 没有的组件** — verification 只证 lower bound，不需要 feasible solution。Planning 必须输出可执行的 action sequence，所以 searching 是 first-class citizen。

## 5. Algorithm 1 完整流程

```
Input: f, C, n (batch size), terminate
1. {f̄*, ũ} ← batch_search(f, {C})       # 全局 search 找初始 upper bound
2. {f_*} ← batch_bound(f, {C})           # 全局 bound 找初始 lower bound
3. P ← {(C, f_*, f̄*, ũ)}                # 候选 subdomain 集合
4. while |P| > 0 and not terminate:
5.   {C_i} ← batch_pick_out(P, n)        # 选 n 个 promising subdomain
6.   {C_i^lo, C_i^up} ← batch_split({C_i})  # 二分
7.   {f̄_lo^*, f̄_up^*} ← batch_search(...)    # 新 subdomain search
8.   {f_lo_*, f_up_*} ← batch_bound(...)     # 新 subdomain bound
9.   if min(f̄_lo^*, f̄_up^*) < f̄*:         # 找到更好解
10.    更新全局 best f̄*, ũ
11.   P ← P ∪ Pruner(f̄*, 新 subdomains)  # lower bound > f̄* 的扔掉
12. return f̄*, ũ
```

整个流程在 GPU 上 batch 执行，多个 subdomain 同时 bound + search。

## 6. 与 NN Verification 的核心区别

| 维度 | NN Verification | BaB-ND (planning) |
|---|---|---|
| 目标 | 证 $\underline{f}^* \leq f^*$ (sound) | 找 $\tilde{\mathbf{u}}$ 使 $f(\tilde{\mathbf{u}}) \to f^*$ (feasible) |
| Lower bound 要求 | 严格 sound | 只要 ranking 准，可 unsound |
| Branching pick | 顺序无所谓，都要 cover | 必须 prioritize promising |
| Branching split | 改善 lower bound | 改善 feasible solution 发现 |
| Searching | 无 | 有，找 feasible solution |
| Bounding | 完整 CROWN propagation | early-stop + empirical bounds |

## 7. Experiments 详解

### 7.1 Synthetic function

$f(\mathbf{u}) = \sum_{i=0}^{d-1} 5\mathbf{u}_i^2 + \cos(50 \mathbf{u}_i)$, $\mathbf{u} \in [-1, 1]^d$.

每个维度 16 个 local optima，2 个 global，$f^* \approx -1.9803 d$。$d=100$ 时 BaB-ND 找到 98-100 维的全局最优，sampling 方法全陷局部最优。

这函数跟 Rastrigin function 是同类 — 经典测试全局优化算法的 hard case。

### 7.2 四个 manipulation tasks

| Task | Action DoF | State | Dynamics | 难点 |
|---|---|---|---|---|
| **Pushing w/ Obstacles** | 2D pusher $\Delta x, \Delta y$ | 4 keypoints on T-shape | 4-layer MLP [128,256,256,128] | non-convex feasible region (障碍) |
| **Object Merging** | 2D | 6 keypoints on 2 L-shapes | 同上 MLP | long-horizon, 多次 contact mode 切换 |
| **Rope Routing** | 3D gripper $\Delta x, \Delta y, \Delta z$ | 10 keypoints (FPS sampling) | 2-layer MLP [128, 128] | deformable, 不能贪心 |
| **Object Sorting** | 4 DoF (2D start + 2D movement) | object centers | GNN (DPI-Net 风格) | multi-object, 离散长程 action |

**Cost function 设计** (Eq. 29-32): 都是 $w_t \|x_t - x_{\text{target}}\| + \lambda \sum_o \text{ReLU}(s_o - \|p_t - p_o\|)$ 这种 — $w_t$ 随时间增长鼓励最后对齐，$\lambda$ 大惩罚碰撞。Obstacle 用 ReLU 形式表达，导致 cost landscape non-smooth。

### 7.3 Quantitative results

**Open-loop planning (Figure 6a)**: BaB-ND 在所有 task 上都低于 GD/MPPI/CEM (objective 越小越好)。Pushing w/ Obstacles 上差距尤其大，因为 sampling 在 non-convex region 里迷路。

**Closed-loop real-world (Figure 6b)**: 
- Pushing w/ Obstacles: BaB-ND 不仅避开障碍，还能完美对齐 target；MPPI/CEM 能避开但姿态差
- Rope Routing: 报 success rate (因为水平拉 rope 也能低 cost 但放不进 slot)
- Object Sorting: variance 显著低于 CEM

### 7.4 Scalability (Figure 7)

- **对比 MIP**: 36 个 (model size × horizon) 配置，MIP 只在 6 个 small 配置找到 optimal，3 个 sub-optimal，27 个 300s timeout。BaB-ND 能处理 **500K params + H=20**
- **Runtime breakdown**: model size 从 9K 涨到 500K (50x)，branching/bounding runtime 几乎不涨，只有 searching 涨 (因为 sample NN forward 是主要开销)

### 7.5 vs 传统 motion planning (Table 3)

对比 RRT 和 PRM (经典 sampling-based motion planning)。在 Pushing w/ Obstacles 上，RRT final step cost 10.65，PRM 1.68，BaB-ND 0.23。原因：RRT/PRM 是离散采样，难以覆盖高维连续空间，且容易卡障碍。

### 7.6 Ablation (Table 4)

- **Branching pick (a)**: 用 $\underline{f}^* + \overline{f}^*$ 双指标最好，纯用 $\overline{f}^*$ 缺 exploration，纯用 $\underline{f}^*$ 缺 exploitation
- **Branching split (b)**: range × bias 启发式最好，单用 range 或单用 bias 都差
- **Bounding (c)**: 完整 bounding 最优，把 lower bound 置 0 (退化为纯 CEM + tree) 显著变差，说明 bound 的 prune 和 prioritize 作用关键

### 7.7 高维 scaling (Table 8, 9)

synthetic function 从 $d=50$ 到 $d=300$：
- BaB-ND gap to optimal: 0.07 → 1.80 (基本线性增长)
- CEM gap: 5.16 → 92.43
- MPPI gap: 7.45 → 357.33
- Selected space size 一直保持 $10^{-3}$ 量级 → BaB 真的把搜索聚焦在极小区域

## 8. Real-world 部署

- **Pushing w/ Obstacles + Object Merging**: 两层 planning — BaB-ND 长程开环出 reference trajectory，MPPI 短程 closed-loop tracking (类似 MPC 思想，参考 [Camacho & Bordons](https://link.springer.com/book/10.1007/978-0-85729-398-5))
- **Rope Routing**: sim-to-real gap 小，直接开环执行长程 action sequence
- **Object Sorting**: 每步 observation 变化大，每步 replan

感知：4 个 RGB-D 相机，用 ICP (pushing/merging) 或 Grounding DINO + SAM + FPS (rope) 或 color filter + K-means (sorting) 提取 state。

## 9. 我的直觉与联想

### 9.1 为什么 BaB > 纯 sampling

本质上是 **search space partitioning** 把"高维采样覆盖不足"问题转化为"低维多次采样"问题。$d=40$ 直接 sample 难，但分成 $2^{20}$ 个 subdomain 后每个 subdomain 体积 $10^{-6}$，CEM 在小区域里几步就收敛。Pruning 又把无关 subdomain 砍掉，避免遍历 $2^{20}$ 全部。

**这和 AlphaGo 的 MCTS 思想完全一致**: tree + value function + selective expansion + UCB-like exploration。

### 9.2 为什么不直接 MIP

MIP solver (Gurobi) 把每个 ReLU 当 binary variable 编码，NN 越大 binary 越多，CPU branch-and-cut 跑不动。BaB-ND + CROWN 用 GPU batch propagation，把 ReLU 的 binary decision 替换成 linear relaxation + 启发式 branching，放弃了 soundness 换 scalability。这是 verification 社区多年沉淀的智慧。

### 9.3 Verification vs Planning 的哲学区别

Verification 是 **soundness-first**: 我宁可 bound 松点，但必须 provable correct (用于安全保证)。

Planning 是 **solution-first**: bound 只是 heuristic，能 rank subdomain 就行，最终看实际 feasible solution 的质量。这个观念转变让 BaB-ND 敢用 empirical bounds、early-stop、unsound linear relaxation — 在 verification 里这都是禁忌。

**这是一个非常 general 的 insight**: 当你把一个 algorithm 从 A 领域搬到 B 领域，先想清楚 A 的"硬约束"在 B 里是不是还硬。Verification 的 soundness 在 planning 里其实是软约束。

### 9.4 与其他相关工作的联想

- **Diffusion-based planning (Diffuser, Janner et al.)**: 也是长程 planning，但用 diffusion model 生成 trajectory，没有显式的 BaB。和 BaB-ND 是 complementary 方向 — 一个是 generative model，一个是 explicit optimization
- **Dreamer / MuZero**: latent imagination + actor-critic，端到端学习。BaB-ND 是 model-based planning 的 explicit optimization 路线，更可解释、可控
- **iLQR / DDP / trajectory optimization**: 都假设 dynamics 可微且 smooth，contact-rich 场景失效。BaB-ND 通过 NN learning 接触 + CROWN bound 处理 non-smoothness
- **AlphaBeta pruning in chess**: 同样的 prune logic — bound 太差就砍
- **kPAM (Manuelli et al.)**: 同样用 keypoint representation，BaB-ND 把它当 state encoding
- **GCS (Graph of Convex Sets, Marcucci et al.)**: 也是 contact-rich manipulation 的强方法，但需要显式 convex decomposition，不用 NN。BaB-ND 用 NN 学 dynamics 绕过这个要求

### 9.5 Limitation 与未来方向

作者自己列了三个：
1. **依赖 dynamics model 准确性**: model bias 会让 planner 找的解在真实世界不 work。可以加 robust optimization / tube MPC
2. **searching algorithm 选择影响 optimality**: 现在用 CEM，可以换成 PGD / MPPI / gradient-free methods
3. **branching heuristic 可能不通用**: 用 RL 学 branching policy 可能更好 (类似 AlphaZero 用 NN 学 tree policy)

我会补充：
- **早停位置 $\mathcal{S}$ 怎么自动选**: 现在 paper 没说清，可能经验设置。可以学
- **bound tightness 和 search budget 的 tradeoff**: 太紧的 bound 计算贵，太松的 bound prune 不掉。可能用 adaptive 深度
- **closed-loop replan 频率**: 现在 long-horizon 开环 + short-horizon MPC，能否做 full closed-loop BaB?

## 10. 参考链接

- **项目主页 (含视频)**: https://robopil.github.io/bab-nd/
- **α,β-CROWN ( inspiration 来源)**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **CROWN paper (Zhang et al. 2018)**: https://arxiv.org/abs/1811.00866
- **CROWN general activation (Zhang et al. NeurIPS 2018)**: https://arxiv.org/abs/1811.00866
- **MIP NN verification (Tjeng et al.)**: https://arxiv.org/abs/1711.07356
- **Liu et al. 2023b (MIP-based planning over sparse NN)**: https://openreview.net/forum?id=ymBG2xs9Zf
- **DPI-Net (Li et al. 2018, particle dynamics)**: https://arxiv.org/abs/1810.01566
- **Propagation Networks (Li et al. 2019)**: https://arxiv.org/abs/1905.04297
- **kPAM (Manuelli et al.)**: https://arxiv.org/abs/2009.05085
- **GCS for manipulation (Graesdal et al. 2024)**: https://arxiv.org/abs/2402.10312
- **CEM (Rubinstein & Kroese book)**: https://link.springer.com/book/10.1007/978-1-4757-4321-0
- **MPPI (Williams et al. 2017)**: https://arc.aiaa.org/doi/10.2514/1.G001770
- **Decentralized CEM (Zhang et al. 2022c)**: https://arxiv.org/abs/2212.08235
- **Pymunk simulator**: https://pymunk.org
- **NVIDIA FleX**: https://developer.nvidia.com/flex
- **Grounding DINO**: https://arxiv.org/abs/2303.05499
- **SAM (Segment Anything)**: https://arxiv.org/abs/2304.02643

---

**核心 takeaway**: 这篇 paper 的精髓是把 NN verification 社区多年沉淀的 BaB + CROWN 工具箱重新 purpose — 放弃 soundness 这个"硬约束"换取大规模 GPU-friendly planning。三个核心改动 (search-integrated bounding, propagation early-stop, sample-distribution-aware branching) 都是为了适配 planning 的"solution-finding"本质。这种"借鉴一个领域成熟工具到相邻领域，但仔细想清楚哪些假设可以松"的思路非常值得学习。
