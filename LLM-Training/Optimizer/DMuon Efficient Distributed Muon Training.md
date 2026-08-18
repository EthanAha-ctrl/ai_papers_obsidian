---
source_pdf: DMuon Efficient Distributed Muon Training.pdf
paper_sha256: 52cede28d3589e98c4159125002a8c515c8984701923022357a234b39b72cd13
processed_at: '2026-08-18T06:15:09-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，咱们把那些复杂的系统架构和数学公式先放一边，我用大白话给你讲讲这篇 paper 到底在折腾什么。

核心故事其实非常简单：**Muon 是个好东西，但它太慢了，DMuon 用工程手段把它加速到了和 AdamW 差不多的地步。**

---

### 1. 为什么 Muon 在多卡上慢得要命？

要理解 DMuon，你得先感受一下 Muon 在分布式训练里的“水土不服”。

AdamW 是一个“各扫门前雪”的 optimizer。每个 GPU 只需要管好自己那一小块 weight shard，根据本地的 gradient 算个 average，更新一下就完事了。GPU 和 GPU 之间在 optimizer step 里根本不需要说话。

Muon 不一样。Muon 的核心逻辑是 Newton-Schulz iteration，它要看**整个 weight matrix** 的样子。它需要对完整的 gradient matrix 算 SVD，拿到正交矩阵。

在多卡 FSDP 训练里，一个 weight matrix $W \in \mathbb{R}^{m \times n}$ 早被切成了 $D$ 块分给 $D$ 个 GPU。为了算 Muon，你得把所有切片收集到每个 GPU 上，拼成完整的 $W$，然后每个 GPU 都去跑一遍一模一样的 Newton-Schulz 算法，算完之后每个 GPU 再把属于自己的那一小块 update 切出来。

这叫什么？这叫**算力浪费+通信爆炸**。
GPU 越多，收集完整矩阵的通信量越大；而且本来 1 个 GPU 能算完的事，非要 $D$ 个 GPU 一起算 $D$ 遍。Paper 里测下来，这个 vanilla distributed Muon 的 optimizer step 时间，甚至比一次完整的 forward+backward 还要长。

### 2. DMuon 的三板斧：指派负责人 + 聪明传纸条 + 算法加速

DMuon 怎么解决这个问题的？靠非常精细的系统工程设计。

#### 第一招：指派负责人
既然 1 个 GPU 算一遍 Newton-Schulz 就够了，为什么要 $D$ 个 GPU 都算？
DMuon 给每个 weight matrix 指定了一个“owner GPU”。这个 owner 负责收集所有 gradient，算一遍 Muon update，然后把更新后的 weight 再发给大家。
这一下就把重复计算的 $D$ 倍开销砍掉了。

#### 第二招：聪明地传纸条
有了 owner，新的问题就来了：forward/backward 的时候，所有 GPU 都需要 weight，这就要从 owner 往外 broadcast；backward 算出的 gradient 又要 reduce 回 owner。通信变成了不对称的。
如果所有人同时找 owner 要数据，网络会堵死。
DMuon 用了一个基于 XOR 的小公式，把连续的 matrix 错开分配到了不同的 node 和不同的 inter-node columns 上。这样大家就不会在同一个网络通道上打架。
同时，它把通信和计算 overlap 起来。layer 1 在算 forward 的时候，layer 2 的 weight 已经在后台偷偷从 owner 那里拉过来了。

#### 第三招：把数学题算得更快
Owner 虽然只算一遍，但 Newton-Schulz 里的 $X X^\top$ 还是很慢。
DMuon 借鉴了 Gram Newton-Schulz 的思路。既然 $X X^\top$ 算出来是对称的，那干嘛要傻乎乎把整个矩阵全算一遍？只算一半（下三角），另一半直接复制就好了。
而且，对于很多小矩阵，单算一个根本喂不饱 GPU，DMuon 就把它们拼成一个大 batch 一起算。
最后，再用 TileLang 这种 DSL 去做 autotuning，针对不同的 matrix shape 找到最快的 kernel 执行配置存起来。

#### 第四招：绝对公平的分配
如果只按顺序轮流分，万一某个 owner 分到了几个超大矩阵，它就会变成“木桶的最短板”，其他人等它，optimizer step 还是慢。
DMuon 在初始化的时候，真的把每个 shape 的真实计算时间测出来，然后列出方程，扔给一个 Mixed-Integer Linear Programming (MILP) solver 去解。保证每个 owner 分配到的总计算时间几乎一模一样。

---

### 3. 细节直觉：拆解背后的数学与工程

为了 build your intuition，我再展开讲几个技术细节。

#### 3.1 XOR Layout 为什么能防拥堵？
公式是：
$$ \mathrm{gpu}(w) = w \bmod 8 $$
$$ \mathrm{node}(w) = (w \bmod 4) \oplus \left( \lfloor w/8 \rfloor \bmod 4 \right) $$
- $w$ 是 matrix 的逻辑序号。
- $\oplus$ 是 bitwise XOR（异或）。

直觉是什么？如果没有 XOR 那部分，前 8 个矩阵都在 node 0，接下来 8 个都在 node 1。这会导致 node 0 在忙的时候，node 1 闲着，而 node 0 忙完后，下一批又一起压到 node 1。
加上 XOR 逻辑后，每经过 8 个矩阵，owner 所在的 node 就会跳一下。结果就是，在任何一段往前看的时间窗里，需要通信的矩阵被均匀打散在所有的 node 和 inter-node columns 上。网络链路像多车道公路一样被充分利用，避免了单车道大堵车。

#### 3.2 Gram NS 省了多少算力？
原始 NS step:
$$ X_{i+1} = a X_i + b X_i X_i^\top X_i + c (X_i X_i^\top)^2 X_i $$
计算 $X_i X_i^\top$ 是 $O(m^2 n)$。如果 $m=2048, n=22016$，这个 GEMM 非常大。
Gram NS 把它变成了关于 $G_i = X_i X_i^\top$ 的递推：
$$ G_{i+1} = P_i G_i P_i $$
其中 $P_i = aI + bG_i + cG_i^2$。
因为 $G_i \in \mathbb{R}^{m \times m}$，现在所有的计算都在 $m \times m$ 的方阵里转圈圈。算力从 $O(m^2 n)$ 降到了 $O(m^3)$。当 $m \ll n$ 时，这是好几倍的提升。
再加上只算下三角的 SYRK kernel，又在 $O(m^3)$ 的基础上省了一半。

#### 3.3 测量式的 MILP Load Balancing
变量 $x_{s,b,r} \in \mathbb{Z}_{\geq 0}$ 表示把多少个 batch（batch size 为 $b$，shape 为 $s$）分给 owner $r$。
目标函数：
$$ \min \max_{r \in R} \sum_{s,b} c_{s,b} x_{s,b,r} $$
这里的 $c_{s,b}$ 是真实测出来的执行时间。直觉就是：我不信什么 FLOPs 估算，我只信秒表。哪个 owner 的总秒数最少，且没有木桶短板，我就用哪个方案。

---

### 4. 实验数据看什么？

Paper 跑了 Wall-OSS (VLA), Pi0 (VLA), Wall-WM (World Model), Qwen2.5-7B (LLM) 四个 workload。

最核心的数据在 Table 1：对比 AdamW 和 Vanilla Muon，DMuon 的 end-to-end step time 平均只比 AdamW 慢 2%。最狠的时候（比如 Qwen2.5-7B 256 卡），DMuon step time 是 2850ms，AdamW 是 2844ms，几乎一模一样。而 Vanilla Muon 需要 6219ms，直接慢了一倍多。

Table 2 的 breakdown 也很直观：
- 48% 的加速来自 Symmetric Gram kernel（数学层面的优化）
- 32% 来自 Owner scheduling（干掉了重复计算）
- 16% 来自 Auto-tuning & batching（榨干 GPU 利用率）

可以说，这是一篇把 systems engineering 做到极致的 paper。

---

### 5. 联想与延伸

1. **VLA 训练的特殊性**：Paper 专门提了 VLA 模型。VLA 的 temporal context 很短，forward/backward 很快。这意味着 optimizer step 的时间占比极高。如果用 AdamW，可能 optimizer 占 20%，但用 Vanilla Muon，optimizer 可能占到 60%。DMuon 对于 VLA 这种短 context 场景简直是刚需。
2. **Single-GPU 的局限**：Paper 老实承认，单卡上 DMuon 的收益没那么大，只有大概 2× optimizer step speedup。因为单卡没法享受 owner 分布式调度和通信 overlap 的红利。这也印证了这套系统是为大规模 FSDP/HSDP 量身定制的。
3. **Precision 的反直觉**：DMuon 在 NS iteration 里用 fp16 而非 bf16。因为 tensor core 上两者速度一样，但 fp16 多了 3 bits mantissa。只要 dynamic range 够，optimizer 内部计算用 fp16 精度更高，收敛更好。这是一个很有意思的工程 trick。
4. **Drop-in 替换的工程价值**：三行代码替换 AdamW，不需要改 PyTorch 底层。这才是系统能够被社区广泛采用的关键。像 Moonlight 和 Kimi-K2 都证明了 Muon 的算法优势，但大家不敢用是因为太贵。DMuon 把这个门槛踩平了。

参考链接：
- DMuon Code: https://github.com/X-Square-Robot/dmuon
- Muon Original Blog: https://kellerjordan.github.io/posts/muon/
- Gram Newton-Schulz: https://dao-lab.ai/blog/2026/gram-newton-schulz/
- Moonlight (Muon scaling proof): https://arxiv.org/abs/2502.16982
- Polar Express (Coefficients): https://arxiv.org/abs/2505.16932

---

# DMuon 深度解析：让 Muon 在分布式训练中达到 AdamW 级别的 overhead

Andrej，这篇 paper 我读下来感觉非常 systems-flavored，核心贡献是用一整套 co-design 把 Muon 的 distributed overhead 从 "2× forward+backward" 压到 "AdamW + 2%"。我会从 intuition 出发，把每个组件的数学、工程和数据拆开讲，最后给出我对这个方向的看法。

---

## 1. 问题的根源：Muon 违反了 element-wise contract

要 build intuition，先得理解为什么 distributed training stack 从一开始就假设 optimizer 是 element-wise 的。

AdamW 的更新规则对单个参数 $w_j$ 来说是：

$$w_{j,t+1} = w_{j,t} - \eta \cdot \frac{\hat{m}_{j,t}}{\sqrt{\hat{v}_{j,t}} + \epsilon}$$

关键性质：$w_j$ 的更新只依赖 $w_j$ 自己的历史 gradient 统计量 $(m_j, v_j)$，不依赖任何其他 $w_{j'}$。这意味着在 FSDP/ZeRO-3 下，每个 rank 只需要持有自己那一片 shard 的 optimizer state，local 算完 update 就完事，**optimizer step 是 embarrassingly parallel**。

Muon 打破这个 contract。对 weight matrix $W \in \mathbb{R}^{m \times n}$，更新规则是：

$$W_{t+1} = W_t - \eta \cdot \mathrm{NS}_k(M_t)$$

其中 $M_t$ 是 momentum-smoothed gradient，$\mathrm{NS}_k(\cdot)$ 是 $k$ 步 Newton-Schulz iteration，近似 matrix sign function（等价于 SVD $M_t = U\Sigma V^\top$ 的正交因子 $UV^\top$）。

每一步 NS 的形式：

$$X_{i+1} = a X_i + b X_i X_i^\top X_i + c (X_i X_i^\top)^2 X_i$$

- $X_i$: 第 $i$ 步迭代的 matrix，初始 $X_0 = M_t / \|M_t\|_F$
- $(a, b, c)$: 系数，选使得 $X_i$ 的 singular values 收敛到 1
- $X_i X_i^\top$ 和 $(X_i X_i^\top)^2$: 这些项耦合了**整个矩阵的所有行**

这就是 granularity mismatch：FSDP 把 $W$ 切成 shards 分给不同 rank，但 NS iteration 需要完整 $M_t$ 才能算 $X_i X_i^\top$。所以 vanilla distributed Muon（paper 叫 Muon-AG）只能：

1. **All-gather** 全 reduced gradient $M = \frac{1}{D}\sum_{r=1}^D g_r$ 到每个 rank
2. 每个 rank **redundantly** 跑 NS iteration
3. 每个 rank 取自己 shard 的 update

两个 cost：
- **Matrix materialization**: 每步每个 matrix 都要 all-gather full gradient，通信量随 model size 和 DP width 线性增长
- **Replicated orthogonalization**: $D$ 个 rank 跑同一个 NS，optimizer compute 乘以 $D$

paper 的测量：vanilla Muon 的 optimizer step 可以 rival 甚至 exceed forward+backward 的 wall-clock time。这对 VLA（vision-language-action）训练尤其致命，因为 VLA 的 temporal context 短，forward+backward 占 step time 比例小，optimizer overhead 难以 amortize。

参考链接：
- Muon 原始 blog: https://kellerjordan.github.io/posts/muon/
- Moonlight (Muon scales to 16B MoE): https://arxiv.org/abs/2502.16982
- Kimi-K2 (trillion-parameter Muon): https://arxiv.org/abs/2507.20534

---

## 2. DMuon 的核心 idea：Owner-centric execution

最直接的 fix：**不要让所有 rank 都跑 NS**。每个 matrix parameter 指定一个 owner rank，只有 owner 跑 Muon update。

这一步立刻消除了 replicated orthogonalization（$D\times$ 节省），但带来新的问题：

1. Forward pass 需要 $W$，但 $W$ 现在只在 owner 上 → 需要从 owner broadcast
2. Backward pass 产生 $g_r$，需要 reduce 到 owner → 需要路由 gradient
3. 不同 matrix 的 NS cost 不同（shape 异构）→ owner 之间 load imbalance
4. Owner 上的 NS 本身还是慢 → 需要更快的 kernel

DMuon 的设计就是把这四个问题一起解决。Algorithm 1 描述了一个完整的 training step：

```
Setup (once):
  1. (s*, r*) ← OwnerAssign({W^(p)})   # §3.4 MILP
  2. 在 owner (s_p*, r_p*) 上分配 W^(p) 和 M^(p)

Forward:
  3-6. for each layer ℓ: 等待 publish event, materialize {W^(p)} 到 packed buffer, forward(ℓ)

Backward:
  7-11. for each layer ℓ (reverse): materialize params, backward(ℓ), reduce gradient 到 owner

Optimizer (owner-only):
  12-15. for each owned W^(p): O^(p) ← GramNS_k(M^(p)); W^(p) ← W^(p) - η O^(p)
         非 matrix 参数走 host stack 的 sharded AdamW

Publish (async):
  17-19. for each layer: 从 owner publish {W^(p)}，记录 event
```

关键语义性质：**owner 收到的 averaged full-matrix gradient $\bar{g}^{(p)}$ 和 synchronous Muon reference 完全一致**，所以 DMuon 不改变 optimizer 的数学，只是改变执行方式。

---

## 3. 通信优化：Fine-grained layout + Forward/Backward overlap

这一节是 paper 里我觉得最精巧的部分。

### 3.1 问题的本质：collective contention

Naive 做法：owner 把 $W$ broadcast 给所有 consumer，gradient 从所有 rank reduce 到 owner。这本质是 1-to-all 和 all-to-1 的 collective。

在 multi-node 部署（比如 $4 \times 8$ mesh：4 nodes × 8 GPUs/node），如果连续的 matrix 都路由到同一个 inter-node column（比如都在 node 0 的 8 个 GPU 之间），那么这些 collective 会**争抢同一个 InfiniBand group**，导致 serialization。

### 3.2 XOR-based owner slot layout

paper 在 $4 \times 8$ mesh 上用一个 closed-form 规则分配 owner slot：

$$\mathrm{gpu}(w) = w \mod 8$$
$$\mathrm{node}(w) = (w \mod 4) \oplus \left(\left\lfloor \frac{w}{8} \right\rfloor \mod 4\right)$$

- $w$: matrix 在 communication schedule 里的 logical index
- $\mathrm{gpu}(w)$: owner 在 node 内的 GPU id (0-7)
- $\mathrm{node}(w)$: owner 所在的 node id (0-3)
- $\oplus$: bitwise XOR

**intuition**：
- $\mathrm{gpu}(w) = w \mod 8$ 把连续的 matrix 分散到 8 个 inter-node columns（每个 column 是一组跨 4 个 node 的同 GPU id）
- XOR term $\mathrm{node}(w)$ 让 owner node 在每 8 个 matrix 后 rotate，避免连续 matrix 都集中在同一个 node

效果：一个 lookahead window 内的 matrix publication 被分散到不同的 inter-node communication groups，可以 **concurrent** 而不互相干扰。

Figure 4 的例子：colors 表示 8 个连续 ID 的 block，可以看到同一行（intra-node group）和同一列（inter-node group）的 ownership 都被均匀打散。

### 3.3 Forward overlap：two-stage pipeline

Forward 的核心 insight：**parameter publication 不需要等到 layer 执行时才发起**。layer ℓ 在 compute 时，layer ℓ+1 的 parameter 已经可以开始 prepare。

两阶段：
1. **Inter-node stage**: 提前（lookahead window 内）把 owner-held weight 搬到每个 node 上对应的 GPU。由于 fine-grained layout，连续 weight 落在不同 inter-node columns，可以 concurrent。
2. **Intra-node stage**: 延迟到 weight 快被消费时，在 node 内 broadcast。

Pipeline 效果（Figure 5a）：
- layer ℓ 在 compute
- layer ℓ+1 在 intra-node broadcast（NVLink 内部，快）
- 某个 lookahead layer 在 inter-node broadcast（IB 跨 node）

三个任务用 training mesh 的不同部分，可以 overlap。

**Memory 优化**：materialized parameter 在 layer 执行完后立即释放，保留 FSDP-style 的 transient materialization lifecycle，peak memory 不会爆炸。这是关键——不是把所有 owner-held weight 都 resident 在每个 rank 上。

### 3.4 Backward overlap：reordered pipeline

Backward 每层有两个通信任务：
- **Before compute**: materialize parameter（broadcast from owner）
- **After compute**: reduce gradient to owner

跨层的关键 independence：materialize layer ℓ+1 的 parameter **不依赖** layer ℓ 的 gradient reduction 完成。

每层的逻辑阶段：

$$\mathrm{bcast}_{\mathrm{inter}} \to \mathrm{bcast}_{\mathrm{intra}} \to \mathrm{compute} \to \mathrm{reduce}_{\mathrm{intra}} \to \mathrm{reduce}_{\mathrm{inter}}$$

跨层 overlap（Figure 5b）：
- layer ℓ 的 gradient 在 reduce
- layer ℓ+1 的 parameter 在 broadcast（用不同 IB column，不干扰）

Contention 处理：intra-node broadcast 和 reduce 共享同一 NVLink fabric，所以 DMuon 把 **broadcast 排在 reduce 之前**避免干扰；inter-node 通信靠 fine-grained layout 分散到不同 column groups。

---

## 4. Gram Newton-Schulz：数学和 kernel

Owner-only 消除了 redundant compute，但 owner 上的 NS 本身还是慢。这里 paper 借鉴了 Zhang et al. 的 Gram NS formulation。

### 4.1 Gram space 的数学

原始 NS step：

$$X_{i+1} = a X_i + b X_i X_i^\top X_i + c (X_i X_i^\top)^2 X_i$$

可以重写为 $X_{i+1} = P_i X_i$，其中：

$$P_i := a I + b G_i + c G_i^2$$

- $G_i := X_i X_i^\top \in \mathbb{R}^{m \times m}$: Gram matrix
- $P_i$: $G_i$ 的多项式，因为 $G_i$ 对称所以 $P_i$ 也对称

Gram matrix 的 closed recurrence：

$$G_{i+1} = X_{i+1} X_{i+1}^\top = P_i X_i X_i^\top P_i^\top = P_i G_i P_i$$

（最后一步用 $P_i$ 对称，$P_i^\top = P_i$）

**复杂度对比**：
- 原始 NS 在 $m \times n$ space：每次 $X_i X_i^\top$ 是 $O(m^2 n)$，当 $m \ll n$（比如 $m=2048, n=22016$）时这是 dominant cost
- Gram NS 在 $m \times m$ space：recurrence $G_{i+1} = P_i G_i P_i$ 是 $O(m^3)$

当 $m < n$（LLM 里 common case，比如 hidden dim < vocab dim），complexity 从 $O(m^2 n)$ 降到 $O(m^3)$。对 $m=2048, n=22016$，这是 $\approx 10\times$ 的算术节省。

### 4.2 Symmetry-aware kernel

$G = X X^\top$ 对称，所以用 general GEMM 算所有 $m^2$ entries 是浪费。DMuon 用 **SYRK-style** execution path：

- Mainloop 只算 lower triangular portion of $G$
- Epilogue reconstruct upper triangle（对称复制）
- 几乎 halve dominant Gram update 的算术

进一步 fusion：Gram update 后的 elementwise operations（比如 $aI + bG + cG^2$ 里的加法）fuse 进同一个 epilogue，避免中间值写回 global memory 再 reload，省掉一次 kernel launch 和 memory round-trip。

### 4.3 Batched execution

不同 matrix 的 NS iteration **互相独立**（gradient 已经 reduce 到 owner 了）。这是 optimizer workload 和 forward/backward 的关键区别——forward/backward 有 inter-layer dependency，optimizer 没有。

利用方式：
- **Large matrices**（already saturate GPU）：走标准 path
- **Small matrices**（occupancy 不够）：grouped by shape，batched 一起跑一次 Gram NS iteration

Figure 7 的数据：small near-square weight（如 $1024 \times 1024$）在 batch=16 时 per-matrix 时间是 single-matrix 的 $\approx 1/3$（3× speedup）；large rectangular weight（如 $2048 \times 22016$）已经 saturate GPU，batch 收益小。

### 4.4 Autotuning

不同 shape 的最优 kernel schedule 差异很大（tile shape, software pipeline depth, warp scheduling, memory layout）。DMuon 用 tile-level DSL（TileLang 和 CUTE DSL）从 common template 生成一族 schedule variants，autotuner 在 target hardware 上 benchmark，选最快的，存 persistent kernel cache。

Workflow（Figure 6）：
1. 给定 Muon workload（shape）
2. Expand search space（tile shapes, block sizes, software-pipeline configs）
3. JIT specialize via tile-level DSL
4. Profile & rank on target hardware
5. Cache selected kernel

Cache key: problem shape + execution mode。Optimizer workload 的特点是 **same parameter shapes recur throughout training**，所以 tuning cost 只付一次，后续 step 直接 dispatch cached kernel。这和 TVM/Ansor 的思路一脉相承。

参考：
- Gram NS blog: https://dao-lab.ai/blog/2026/gram-newton-schulz/
- Gram NS code: https://github.com/Dao-AILab/gram-newton-schulz
- TileLang: https://arxiv.org/abs/2504.17577
- Polar Express (coefficient set): https://arxiv.org/abs/2505.16932

### 4.5 Precision choice

一个很 subtle 的点：DMuon 在 NS iteration 里用 **fp16 而非 bf16**。

- 两者在 tensor core 上 cost 相同
- fp16 比 bf16 多 3 bits mantissa（10 vs 7）
- NS iteration 只需要 preserve singular subspace，不需要 singular values 本身，所以 dynamic range 不是问题
- 但精度更高有助于 iteration 收敛质量

paper 实测 wgrad 的 dynamic range 在 fp16 表示范围内。整个 NS iteration 在 fp16（除 on-chip accumulation），orthogonalized update cast 到 fp32 应用到 fp32 master weights，再 cast 回 working dtype (bf16)。

---

## 5. Computation-aware load balancing：MILP formulation

这部分我觉得是 paper 里最 "systems" 的 contribution，因为它把一个看似简单的 assignment 问题 formalize 成 measured-cost optimization。

### 5.1 为什么 naive assignment 不行

- **Round-robin**: 忽略 matrix shape 差异，大矩阵集中到几个 rank 就成 straggler
- **LPT (Longest-Processing-Time)**: 用 analytical cost model（比如 FLOPs 或 parameter count），但 batching + autotuning 让真实 runtime 和 FLOPs 不成正比

### 5.2 Measured execution-cost model

初始化时（only once，因为 parameter shapes 固定）：
1. 按 shape 分组 parameters。$S$ = distinct shapes，$n_s$ = shape $s$ 的 parameter 数量
2. 对每个 shape $s$，evaluate candidate batch sizes $B_s$，benchmark owner-local Muon update
3. 得到 $c_{s,b}$: shape $s$ 用 batch size $b$ 处理一个 batch 的 measured time

$c_{s,b}$ 包含完整 execution path：batching behavior + kernel implementation + autotuned schedule。比 analytical estimate 准确得多。

### 5.3 MILP formulation

变量：$x_{s,b,r} \in \mathbb{Z}_{\geq 0}$，表示 shape $s$、batch size $b$、owner $r$ 的 batch 数量。

$$\min \quad \max_{r \in R} \sum_{s,b} c_{s,b} x_{s,b,r}$$

$$\text{s.t.} \quad \sum_{r,b} b \cdot x_{s,b,r} = n_s, \quad \forall s$$

$$x_{s,b,r} \in \mathbb{Z}_{\geq 0}$$

- 目标函数：最小化所有 owner rank 的最大 makespan（min-max，直接对应 critical path）
- 约束：每个 shape $s$ 的所有 parameter 必须被分配恰好一次（$b \cdot x$ 加起来等于 $n_s$，因为一个 batch size $b$ 的 batch 处理 $b$ 个 matrix）

用 SciPy 的开源 MILP solver 一次性解出。大规模时（decision variables 超过 threshold $S_{\mathrm{thr}}$）fallback 到 greedy search，保证 bounded initialization cost。

**intuition**：这不是简单的 "把大矩阵均匀分给 rank"，而是 "在 batch size 和 kernel schedule 的联合空间里，找到让所有 owner 的 measured wall-clock 尽量均衡的分配"。MILP 的好处是它同时决定了 *哪个 owner 处理哪些 matrix* 和 *用什么 batch size*。

---

## 6. 实验数据解读

### 6.1 End-to-end step time（Table 1）

四个 workload：Wall-OSS（VLA）、Pi0（VLA）、Wall-WM（world model）、Qwen2.5-7B（LLM）。

关键数据点：

| Model | GPUs | DMuon Step (ms) | Vanilla Step (ms) | AdamW Step (ms) | Δ_A |
|-------|------|-----------------|-------------------|-----------------|-----|
| Wall-OSS | 128 | 1437 | 2745 | 1412 | +1.8% |
| Wall-OSS | 256 | 1519 | 2857 | 1496 | +1.5% |
| Pi0 | 256 | 1648 | 2665 | 1637 | +1.2% |
| Wall-WM | 256 | 3011 | 9061 | 2915 | +3.3% |
| Qwen2.5-7B | 256 | 2850 | 6219 | 2844 | +0.2% |

观察：
1. DMuon 相比 AdamW 的 overhead 平均 +2%，最差 Wall-WM@256 是 +3.3%，最好 Qwen2.5@256 是 +0.2%
2. DMuon 相比 vanilla Muon-AG 的 speedup：step 1.48×–3.01×，optimizer 6.85×–163.00×
3. Optimizer speedup 随 GPU 数增长（Wall-OSS: 15.12× @8 → 109.44× @256），因为 vanilla 的 redundant compute 随 DP width 线性增长，DMuon 是 owner-only 固定 cost

### 6.2 为什么 Wall-WM 的 Δ_A 比 LLM 大？

Wall-WM 在 8 GPU 时 Δ_A = +17.6%，到 256 GPU 降到 +3.3%。paper 解释：small GPU count 时 Muon 允许更大 batch size（memory 省），所以 throughput 高；随 distributed width 增加 memory pressure 降低，batch-size advantage 消失，DMuon 收敛到 AdamW baseline。

这个观察的深层含义：**Muon 的 wall-clock 优势不完全来自 convergence efficiency，还来自 memory efficiency 允许的更大 batch**。在 memory-constrained regime（小 GPU 数），Muon 省的 optimizer state（只有 momentum，没有 AdamW 的 $v$）换成更大 batch，间接提速。

### 6.3 Component breakdown（Table 2）

在 Wall-OSS-0.5 @ 128 GPUs 上 ablation：

| Component | Share of speedup |
|-----------|------------------|
| Symmetric Gram kernel | 48% |
| Owner scheduling & load balancing | 32% |
| Auto-tuning & NS batching | 16% |

**intuition**：
- Symmetric Gram kernel 贡献最大（48%），因为它在每个 NS iteration 的 dominant product 上都省一半算术，而且 $k=5$ 次 iteration 都受益
- Owner scheduling 贡献 32%，因为消除了 $D\times$ redundant compute（$D=128$ 时这是巨大节省），load balancing 防止 straggler
- Auto-tuning + batching 贡献 16%，主要救 small matrix 的 occupancy 问题

---

## 7. 一些更细节的工程点

### 7.1 Tensor parallelism composition

TP shard 单个 weight matrix 跨 TP group。DMuon 用 **nested ownership**：
1. 先在 DP 维度用 MILP 分配 matrix 给 DP owner slot
2. 在 DP owner slot 内的 TP group 里，再指定一个 rank 作为 TP owner

TP owner 负责：
- Gather TP-sharded gradient slices
- Assemble full gradient
- 跑 Gram NS
- Re-partition orthogonalized matrix
- Scatter update slices 给 TP peers

之后 step 和 non-TP case 一样。TP handling 完全 confined 在 optimizer step，forward/backward/publish 都用 host stack 已有的 per-layer slices。

### 7.2 Non-owner placeholder

非 owner rank 上，原 parameter 被替换成 zero-size placeholder（同 dtype）。这保留了 module-graph traversal code（Apex, PEFT, gradient clipping libraries 走 `model.parameters()`），但 memory 几乎为零。

Owner 上分配 full-precision `_owned_data` tensor。Unshard 时 packed buffer 从 owner data 填充，暴露成 persistent `nn.Parameter`（storage 是 packed buffer），autograd 直接写 gradient 到 `.grad` field，避免中间 copy。

Tied parameters（比如 input embedding alias to output head）在每个 alias 都替换，没有 alias escape。

### 7.3 Polar Express coefficients

DMuon 默认用 Polar Express 的 coefficient set for $k=5$ NS steps。Polar Express 是 Amsel et al. 的工作，优化 per-step coefficients 使得固定 iteration count 下收敛最快。

paper 选 Polar Express 的理由：symmetric kernel 是 tune for Polar Express 后期 steps 产生的 matrix shapes。$(a,b,c)$-quintic coefficients 可通过 config 选择，但和 DMuon 的 systems contributions orthogonal。

---

## 8. 我的几点观察和联想

### 8.1 为什么不直接 distributed NS？

一个自然的 question：能不能不 gather full matrix，直接在 distributed representation 上做 NS？比如 distributed SVD 然后 reconstruct $UV^\top$？

理论上是可能的（比如用 QR decomposition 的 distributed 版本），但：
1. NS iteration 的收敛性依赖完整 $G_i = X_i X_i^\top$，sharded 计算 $G_i$ 需要跨 rank 的 reduction，通信量和 gather full matrix 相当
2. NS 的优势是简单（几次 matrix multiplication），换成 distributed SVD 会引入更复杂的通信 pattern 和数值稳定性问题

DMuon 的选择是：**承认 NS 需要完整 matrix，但通过 owner-only + overlap 把这个 cost 藏起来**。这是 systems 思路而非 algorithmic 思路。

### 8.2 和 Distributed Shampoo / Canzona 的对比

Distributed Shampoo（Shi et al. 2023）用 owner-compute/all-gather paradigm：Shampoo update 的 memory 和 computation 分配给 workers，search direction 在每步 all-gather。

Canzona（Wang et al. 2026）在 Megatron-style stack 上用 $k$-balanced DP partition + TP micro-group scheduling，Qwen3-32B @ 256 GPUs 报告 1.57× end-to-end speedup。

DMuon 的区别：
- 针对 PyTorch FSDP2/HSDP（ZeRO-3-style），不是 Megatron
- 用 fine-grained overlap runtime（scheduled broadcasts, two-stage reductions, async publish）而非简单的 owner-compute
- 通信和 computation 联合 optimize（MILP load balancing 包含 batching 和 kernel schedule）

### 8.3 fp16 vs bf16 的选择很有意思

这是一个工程上很 subtle 的决定。Muon community 一般用 bf16 因为 LLM 训练 default 是 bf16（dynamic range 大）。但 DMuon 在 NS iteration 里用 fp16，理由是：
1. Tensor core cost 相同
2. NS 只需要 singular subspace，dynamic range 不是瓶颈
3. fp16 多 3 bits mantissa → 更高精度

这暗示了一个更 general 的 principle：**optimizer 内部计算的 precision 需求和 forward/backward 不同**。Optimizer state 的 dynamic range 通常 well-behaved（gradient clipping + momentum smoothing），但精度对收敛质量敏感。fp16 在 optimizer internal 可能是 better default。

### 8.4 Limitations 和 future directions

paper 自己提的 limitation：single-GPU 训练 benefit 小（只有 ~2× optimizer step speedup，没有 distributed 的额外 benefit）。

我觉得还有几个值得探索的方向：
1. **Async Muon**: 当前 owner update 还是 synchronous（next step 等 publish 完成）。能不能把 owner update 延迟一步，用 stale momentum？这会改变 optimizer 数学，但可能 acceptable。
2. **Pipeline parallelism composition**: paper 主要讲 DP/TP，PP 的 interaction 没详细讨论。PP 下 owner assignment 需要考虑 PP stage boundary。
3. **Expert-parallel (MoE)**: MoE 的 expert weight 是 sparse-activated，owner assignment 需要考虑 expert 的 routing distribution。
4. **Overlap with activation checkpointing**: 重计算 backward 时，parameter materialization 的时机和 recompute 的 interleaving 值得 optimize。

### 8.5 这个工作对 Muon 普及的意义

Moonlight 和 Kimi-K2 证明了 Muon 在 scale 上的 convergence benefit，但 deployment cost 是 obstacle。DMuon 的贡献是把 deployment cost 压到 near-AdamW，让 Muon 成为 **drop-in replacement**。

从 Figure 1 看，三行代码就替换 AdamW：
```python
dm.dedicate_params(model, mesh)
optimizer = dmuon.Muon(model.parameters(), ...)
# 后续正常训练
```

这种 drop-in 的工程 ergonomics 对 adoption 很关键。Reference implementations 再好，如果 integration cost 高，community 不会用。

---

## 9. 总结

DMuon 的核心 thesis：**Muon 的 distributed overhead 不是 algorithmic 的，是 systems 的**。通过四层 co-design：

1. **Owner-centric execution** 消除 redundant NS compute
2. **Fine-grained communication layout** (XOR slot assignment + forward/backward overlap) 藏通信 cost
3. **Gram NS + symmetric kernel + autotuning** 压 owner 上的 NS cost
4. **MILP load balancing** 防 straggler

把 Muon 的 per-step overhead 从 "2× forward+backward" 压到 "AdamW + 2%"。

数学上等价（preserves exact Muon semantics），工程上 drop-in，实测在 4 个 production workload 上验证。这是 systems research 服务于 algorithmic progress 的好例子——Muon 的 convergence benefit 一直都在，DMuon 让它 practically usable。

代码：https://github.com/X-Square-Robot/dmuon

---

## References 我觉得值得 follow up 的

1. **Muon 原始 blog** (Keller Jordan): https://kellerjordan.github.io/posts/muon/
2. **Moonlight** (Muon scales to 16B MoE, 2× AdamW efficiency): https://arxiv.org/abs/2502.16982
3. **Kimi-K2** (trillion-parameter Muon): https://arxiv.org/abs/2507.20534
4. **Gram Newton-Schulz** (Zhang et al., Dao Lab): https://dao-lab.ai/blog/2026/gram-newton-schulz/
5. **Polar Express** (optimal NS coefficients): https://arxiv.org/abs/2505.16932
6. **TileLang** (tile-level DSL for kernel autotuning): https://arxiv.org/abs/2504.17577
7. **FSDP2 / DTensor** (PyTorch native sharding): https://arxiv.org/abs/2304.11277
8. **Distributed Shampoo** (owner-compute paradigm 的先驱): https://arxiv.org/abs/2309.06497
9. **Canzona** (Megatron-style distributed matrix optimizer): https://arxiv.org/abs/2602.06079
10. **Dion** (distributed orthonormalized updates, Microsoft): https://arxiv.org/abs/2504.05295

如果你想深入某个具体组件（比如 Gram NS 的 kernel 实现、MILP 的求解细节、或 forward/backward overlap 的具体 scheduling），我可以再展开。
