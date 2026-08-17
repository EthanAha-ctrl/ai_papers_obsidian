---
source_pdf: FlashAttention on a Napkin A Diagrammatic Approach to.pdf
paper_sha256: 03dce23b4e57a0070cbd4997986ff12a529fe3c3d9f3d49d1cb7578aca93be95
processed_at: '2026-08-04T08:47:27-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍

## 一句话

这 paper 想把 "怎么在 GPU 上写一个快的 attention kernel" 这件事, 从 "Tri Dao 花三年磨三版" 的手艺活儿, 变成 "画几张图、代几个公式就能算出来" 的工程活儿。工具是一套 string diagram, 论点是算出来的预测还真能对上 FlashAttention-3 的实测数。

## 为什么这件事有意义

GPU 优化的现状很尴尬。你写 PyTorch, XLA/TorchInductor 会帮你 fuse 一些 elementwise, 但遇到 FlashAttention 这种需要 tile + online softmax + tensor core fragment + async pipeline 一起考虑的, compiler 基本摆烂, 还得人手写 CUDA。Tri Dao 在 2022 写了 [FlashAttention](https://arxiv.org/abs/2205.14135), 2023 自己又重写一遍 [FlashAttention-2](https://openreview.net/forum?id=mZn2Xyh9Ec), 2024 交给 Jay Shah 等人用 Hopper 新特性再写 [FlashAttention-3](https://arxiv.org/abs/2407.08608)。三年三版, 每次都靠人脑重新推一遍。

而且更糟的是, 每次推出新版, 社区要花半年才能凑出"为什么 FA3 用 intra-warpgroup + 大 $s_x$ tile 而不是 inter-warpgroup + 小 tile"的 post-hoc 解释。Abbott 和 Zardini 这篇 paper 直接说: 这种"先 work 后解释"是 deep learning 的通病, 想换成"先预测后验证"。

## 三个核心 idea

### Idea 1: 把 algorithm 画成 wire diagram

一个 tensor $\mathbb{R}^{a\times b\times c}$ 画成三根叠着的 wire, 标 $a, b, c$。function 是 box, 横着排就是 sequential compose, 竖着排加 dashed line 就是 concat。这种 string diagram 在 category theory 圈子里是 standard, [Piedeleu & Zanasi 2023](https://arxiv.org/abs/2305.08728) 有 CS 入门。

关键动作叫 **weaving**: 给一个 function 加一个 axis, 把它 map 过去。SoftMax 沿每行做 = weave SoftMax over row axis。dot product weave over q 和 d 两边 = matrix multiplication。这跟 JAX 的 `vmap` 是一回事, 只是用图表示, 比 `einsum` 字符串清楚得多。

### Idea 2: 把 GPU hierarchy 上色

黑 = GMEM, 橙 = SMEM, 绿 = registers, 蓝 = tensor core fragment。颜色改变 = 一次 transfer。然后给每个 algorithm 定义两个数:

- $H_\ell$: level $\ell$ 上 transfer 的总量
- $M_\ell$: level $\ell$ 上任一时刻在场的最大数据量, 必须 $\le M_\ell^{\max}$ (硬件上限)

这两个数从图上直接读出来, 不用猜。

### Idea 3: 两条 rewrite rule 把 tiling 和 streaming 都机械化

**Group partition (tiling)**: 如果一个 algorithm 被 weave 在 axis $a$ 上, 就可以把 $a$ 切成 $a/g_a$ 批, 每批 $g_a$, 每批丢给一个 core。代价公式:

$$H_{\ell1} = abx\,(g_b^{-1} + g_a^{-1}) + aby$$

变量意思: $a, b$ 是 weaved 的两个 axis 的 full size, $x$ 是"被 broadcast 的 weight" 每个 slice 的大小, $y$ 是 output 每个 slice 的大小, $g_a, g_b$ 是 group size。

直觉: tile 越大越省 transfer, 但受 SRAM 大小限制。这点所有人都知道, 但 paper 把它写成图上一次 relabeling, 不用再脑补。

**Stream partition (online recomputation)**: 如果一个 function 是 polymorphic (任意 size 都能算) 而且有 accumulator 能合并 partial output, 就可以把 input chunk 成 $s_a$ 一份份喂进去, partial output 留在 chip 上。dot product 是 streamable (累加), SoftMax 也 streamable (维护 running max $\mu$ 和 sum $z$, 新 chunk 来了用 $\delta = e^{\mu'-\mu}$ rescale 老的 partial)。这就是 [FlashAttention Algorithm 1](https://arxiv.org/abs/2205.14135) 的 online softmax, 但 paper 让它变成一条可应用的 rewrite rule, 不再依赖"灵感"。

然后关键来了: **Theorem 1 (Fusion Theorem)** 说, streamable 的 algorithm 被 compose 或 weave 后还是 streamable。所以你只要找到一个 streamable kernel (比如 SoftMax-Contraction), 就能在它周围贴 QK matmul、再 weave over q 和 d, 自动得到 FlashAttention, 完全机械推导, 没有任何"灵机一动"。

## 一个公式就是全部

two-level 优化的 closed form:

$$H^*(\vec{a}, M) = \sum_t \alpha_t(\vec{a})\,M^{-\beta_t}$$

变量解释: $\vec{a}$ 是所有 axis size 的 vector, $M$ 是下层 memory 上限, $t$ 枚举不同项, $\alpha_t$ 是依赖 axis size 的系数, $\beta_t \ge 0$ 是"memory 敏感度指数"。

对 matmul: $\beta = 0.5$ → square tile 最优, $H \ge 2abc\,M^{-0.5} + ac$。

对 attention: $\beta = 1$ → KV 是 broadcast, $H \ge 2qd + 4xqd^2\,M^{-1}$。

$\beta$ 这个数字信息量极大:
- $\beta = 0.5$ 意味着 memory 加倍, 代价减到 $1/\sqrt{2}$
- $\beta = 1$ 意味着 memory 加倍, 代价直接减半

attention 对 memory 大小的敏感度远高于 matmul, 所以 SRAM 给 attention 用比给 matmul 用"收益更大"——这就解释了为什么 NVIDIA 一代代加 SMEM, FA 每一代都吃满。

而且 quantization (每 value 用 $q$ bytes) 代进去:

$$H^{*\,\text{Bytes}} \propto q^{1+\beta}$$

FP32→FP16 ($q$ 减半) 对 attention 加速 $\times 4 = 2^{1+1}$, 对 matmul 加速 $\times 2^{1.5} \approx 2.83$。这就解释了为什么 [FP8 FlashAttention-3](https://arxiv.org/abs/2407.08608) 的实测收益比"按 byte 减半算"大——broadcast 项的收益是 byte 节省的平方反比。

多级 hierarchy 加权:

$$H^* = \sum_\ell \dot{H}_\ell^{-1}\,H^*(\vec{a}, M_\ell)$$

变量: $\dot{H}_\ell^{-1}$ 是 level $\ell$ 的 per-value transfer cost, $M_\ell$ 是该 level 的 effective memory。一行公式把 L2 cache、SMEM、register、tensor core 全收进来。cross-transfer level (Hopper 的 thread block cluster 或 multi-GPU NVLink) 等效成插入一个新 level, effective memory 是 $N_c^{\max} M_c$ (所有 children 共享)。

## 把它当 hypothesis generator

paper Section 5 用八步法把 diagram refine 到 Hopper 上的可运行 kernel sketch:

1. 展开 streamable 算法成 loop pseudocode
2. 识别 subloop (哪些 op 可以再 split)
3. 给 tensor core 涂蓝色、register 涂绿色
4. 加 divisor 约束 ($w_q^{(128)}$ 是 warpgroup 倍数, $s_x^{(32)}$ 是 K-dim 倍数)
5. 列出所有 variables 的 memory footprint
6. 解 linear 约束 $N_{\max} = (M_{\max} - M_{TB})/M_{WG}$ 求最大 warpgroups
7. 算每个 op 的 ops/clock → 判断 compute bound 还是 bandwidth bound
8. 画 pipelining diagram, 让 barrier 都 wait on tensor core

第 7 步在 H100 SXM5 上算出: $g_q \ge 295$ 才不被带宽 limit, 理想 throughput 1.32 PFLOP/s。

然后 Section 6 拿这把尺子量 FlashAttention-3:

- FA3 FP16: 实测 740/989 TFLOP/s = 75%
- FA3 FP8: 实测 1.2/1.98 PFLOP/s = 60%

paper 算: 如果 SoftMax 操作有 66% overhead, 那 FP16 应该花 4/3 时间 (75%), FP8 应该花 5/3 时间 (60%)。**两个数都对上了**。

这是 paper 最强的一击——它不是事后解释, 而是事前预测。当然, 一个数据点对上可能是巧合, 但这就是科学方法: 提出可证伪假设, 跑实验看对不对。如果下一个 GPU (Blackwell) 上 FA4 的 utilization 也符合预测, 模型就更可信; 如果不符, 就回头改 $\beta$ 或 tensor core overhead 假设。

## paper 自己承认的盲点

tensor core fragment 的 thread-wise manipulation。CUTLASS 的 [CuTe layout](https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf) 让你能控制 tensor core 输出 fragment 怎么分布在 warp 内 32 个 thread 的 register 上, FA3 用这个避免一次 SMEM 中转。paper 的 diagram 把 tensor core memory 当成一个"incoherent 黑盒", 表达不了这种细节。补这块需要 **reindexing weaving**——对应 category theory 的 Yoneda natural transformation, Abbott 自己 2023 的 [Robust Diagrams](https://www.vtabbott.io/content/files/2023/11/Robust-Diagrams-for-Deep-Learning-Architectures.pdf) 已经埋了伏笔。

## 一点个人直觉

读完最大的 takeaway: **$\beta_t$ 是 paper 最有 productizable 价值的产出**。一个 algorithm 的 $\beta$ 直接告诉你:
- $\beta=0.5$: tile 越大越好, 受 memory 限制 → matmul
- $\beta=1$: broadcast 越少越好, 减少 head 数更划算 → attention, GQA, MQA
- $\beta=2$: 这种情况量化收益是 $2^{3} = 8$ 倍, 任何代价都值得做 quantization

如果 community 能给一组标准 microbenchmark 把常见 op 的 $\beta$ fit 出来, 就能直接用作 compiler 的 cost model prior, search space 缩到很小。Triton 现在的 autotune 之所以慢, 很大程度是 cost model 不行, 还在 grid search。paper 给的是一套可以替代的 analytical cost model, 虽然还差几步工程化。

更深一层: paper 暗示了一种 deep learning 优化的新风格——"diagram-first"开发。你先画图, 图告诉你哪里有 degree of freedom, 然后这些 degree of freedom 就是 compiler search 的合法区域, 性能模型告诉你哪一块值得花时间测。这跟 [FlashAttention-3](https://arxiv.org/abs/2407.08608) 的"读 NVIDIA manual, 凑三版 kernel, 跑 profile, 留最快的"风格是两个范式。

## 一句话再总结

**它把"写 GPU kernel"从一门手艺, 推向一门可以证伪的科学**。具体预测对不对还在其次, 关键是第一次给了一套"假设可写下来、实验可证伪"的框架, 让未来的 kernel 优化不再是 post-hoc rationalization on the test set of prior successes。

如果你想看 paper 真正 work 起来, 最 exciting 的下一步应该是有人拿这个框架去推 Blackwell 的 [TMEM](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief) 或 AMD [CDNA 3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) 的 attention kernel, 然后实测数和预测数对照——这就把 paper 的"scientific approach"从口号变成 practice 了。

---

# FlashAttention on a Napkin: 一篇用 Diagrammatic 方法重写 GPU 优化叙事的 paper

这篇 paper (Vincent Abbott @ UCL + Gioele Zardini @ MIT LIDS) 的核心 thesis 是: **GPU kernel 的 IO-aware 优化不应该靠三年三次迭代的"tinkering"硬磨出来, 而应当有一套 diagrammatic 形式化体系, 把"假设 → 性能预测"摆成可证伪的科学命题**。它把 Neural Circuit Diagrams 扩展成一种能同时表达 (a) tensor shapes, (b) function composition, (c) GPU hierarchy (DRAM/SMEM/RMEM/Tensor Cores) 上的数据分布和 (d) 资源消耗 的统一记号, 用 simple relabellings 把 streaming/tiling 的"调度"动作化成图上一次次的小改写, 然后顺势推出 closed-form performance model, 进一步推 quantization、multi-level cache、cross-transfer、再到 Hopper 上的 warpgroup pipelining。

下面我按"build intuition"的顺序展开, 不按 paper 章节顺序, 而是"为什么 → 记号 → 推导 → 性能模型 → Hopper 配置 → FlashAttention 反思 → 联想"。

---

## 1. 为什么 paper 价值不止于"再画一遍 FlashAttention"

paper 开头一句很锋利:

> "Automated compiled methods have consistently lagged behind. The current best technique for generating IO-aware algorithms that exploit hardware features remains slow, manual derivation."

这是在挑战 Triton、XLA、TorchInductor 这条"compiler 路线"的现实瓶颈: 你不能只是 fuse elementwise + matmul, 还要 manage tensor core fragmentation、async pipelining、barriers、swizzled layouts——这些目前还是 Tri Dao 这种人手写。Paper 的 meta-goal 是:

> "We aim to lay the groundwork for a scientific approach to GPU optimization where experiments can address clear hypotheses rather than post-hoc rationalizations."

这句话在 deep learning 语境里其实很扎心, 因为 DL kernel 优化长期是"先 work 再解释"的状态 (post-hoc rationalization on a "test set" of prior successes)。Abbott 自己 2023 的工作 [Robust Diagrams for Deep Learning Architectures](https://www.vtabbott.io/content/files/2023/11/Robust-Diagrams-for-Deep-Learning-Architectures.pdf) 给了 string-diagram 雏形, 这篇把它推到 **resource-aware + hardware-hierarchy-aware** 这一档。

同时 paper 与 **Andrea Zardini 的 categorical co-design 框架** 也有明显关联——Zardini 的 thesis [Co-design of Complex Systems](https://www.research-collection.ethz.ch/handle/20.500.11850/648075) 把 functionality/requirement 关系做成 compositional lattice, 这里想把 hardware capability / algorithm configuration / performance 拉进同一个 lattice, 最后给"硬件选择 + 算法配置"做组合推理。

---

## 2. Diagrammatic 记号: 把 tensor algebra 变成"电路图"

### 2.1 几何直觉

记号是交替的列: 一列是 **data type** (wires), 一列是 **function** (boxes), 像 bi-algebra 里的 string diagrams ([Piedeleu & Zanasi 2023](https://arxiv.org/abs/2305.08768) 给 CS 学的入门正好是这个味儿):

- 一个 tensor $\mathbb{R}^{a\times b\times c}$ → 三根叠放的 wires, 标 a, b, c
- 两个 tensor 组成的 tuple $\mathbb{R}^{a\times b\times c}\times \mathbb{R}^{d\times e\times c}$ → 用 dashed line 分隔
- **水平 compose** $F;G=G\circ F$ → sequential execution
- **垂直堆叠 + dashed line** $F\otimes G$ → concatenation, $(F\otimes G)(x\otimes y)=F(x)\otimes G(y)$
- **identity** → wire 直接穿过 function column

这套记号比 PyTorch eager 风格的代码更能让人看清"哪些 axis 在哪里 share / split", 因为代码里的 `einsum` 字符串经常把 axis 当成纯字母处理, 看不出 axis 是从哪里来的, 又会被谁 consume。

### 2.2 Weaving: paper 的核心动作

> A function can be **weaved**, which adds an axis to the outputs and some of the inputs. The function is mapped over this axis. When we weave the "item" $\mathbb{R}$ array represented by a thick dotted wire, we can remove it.

Weaving 本质就是 `vmap` (JAX 风格) 或者 `broadcast` + `map` 的几何化版本:

- SoftMax 沿行做: weave SoftMax over row axis
- dot product weave over q 和 d 两边 → matrix multiplication
- 注意 "weave 但 input 没被 weave 的那段 → 那段数据被 copy/broadcast 到每个 index"

这就是 paper Figure 5/6 想表达的——把"加法/mul/copy/split/join"全部还原成 primitive 的 weaving。这种 view 与 [Backprop as Functor (Fong/Spivak/Tuyéras 2019)](https://ieeexplore.ieee.org/document/8785665) 和 [Cruttwell et al. 2021](https://arxiv.org/abs/2103.01931) 里把 learning 看作 Cartesian/closed category 上的 parametrised morphism 是同一个 family, 这也是 paper 7 节末尾说"diagrams conform to a category-theoretic description"的伏笔。

### 2.3 Hierarchy 上色

GPU 的层级被抽象成 graph, level 之间用 pipe 连接, 见 Figure 7:

- 黑色: 高层 (GMEM/DRAM)
- 橙色: 中层 (SMEM, 一个 SM 的 shared memory)
- 绿色: 更低 (registers / tensor core fragments)
- 蓝: tensor core 段
- 颜色变化 = 一次 transfer

每个 algorithm 不仅是个 function $\text{in}\to\text{out}$, 它还有两个"resource signature":

- $H_\ell$ — level $\ell$ 上发生的总 transfer cost (load+save 的累加, 数值量纲是"values 数量")
- $M_\ell$ — level $\ell$ 上任一时刻需要的最大 memory usage, 必须满足 $M_\ell \le M_\ell^{\max}$ (硬件上限)

> Memory usage is lower bounded by the maximum size of data at a level for any column.

直觉: 图上每个 column 都对应"那一刻同时在场"的所有 colored array, 取最大值就是 $M_\ell$ 的下界。$H_\ell$ 是所有"颜色改变"事件的 size 求和。这是后面所有性能模型的"原子"。

---

## 3. 两大策略: Group Partition 和 Stream Partition

paper 把 FlashAttention-2/3 用到的两类 trick 抽成两条 rewrite rule, 这两条 rule 是后面所有推导的引擎。

### 3.1 Group partition (tiling)

如果一个 algorithm 被 weave 在 axis $a$ 上 (i.e., "对每个 a-slice 独立执行"), 那么 $a$ 可以被切成 $N_{\ell,g}=a/g_a$ 个 batch, 每批 $g_a$, 每个 batch 落到一个 lower-level core 上:

- $M_\ell$ 的计算用 $g_a$ (不是 $a$)
- $H_\ell = N_{\ell,g} \cdot H_{\ell,g}$
- **未被 weave 的 input** (比如 matmul 中 b 维) 会被 broadcast 到每个 core, 所以 transfer cost **按 $N_{\ell,g}$ 倍乘**

关键 insight:

> Smaller group sizes $g_a$ decrease memory usage but increase $N_{\ell,g}$, increasing $H_\ell$ if there is an unweaved input. To reduce total transfer costs, we must find the **maximum** $g_a$ value that does not exceed $M_\ell^{\max}$.

直觉就是: tile 越大 → 共享越多 → transfer 越省, 但受限于 SRAM 大小。这正是 [Gholami et al. AI and Memory Wall 2024](https://arxiv.org/abs/2403.14123) 提到的"AI 已经 passed memory wall" 量化背后的微观机制。

对多 axis 同时 group, 公式 Figure 12 给出 (我抄一遍, 标好变量):

$$H_{\ell 1} = abx\,(g_b^{-1} + g_a^{-1}) + aby$$

其中:
- $a, b$: 两个 weaved axis 的 full size
- $x$: "weight" input 的 size per slice (例如 matrix $A$ 的一列)
- $y$: output 的 size per slice
- $g_a, g_b$: group sizes
- $w$: 第三方 shared input 的 size per slice

注意 $abx(g_b^{-1}+g_a^{-1})$ 这项就是"两个 axis 各自被 broadcast 一遍"——这就是 tiling 的代价公式。

### 3.2 Stream partition (recomputation / online computation)

更精妙的是 stream partition。条件 (Figure 13):

- $F$ polymorphic 在 axis $a$ 上 (任意 size 都能算)
- 存在一个 **accumulator** $B$, 能把已经在 chip 上的 partial output 与新来的一块 input 合并

满足后, $a$ 可以递归切成 $a/s_a$ 份, 每份 $s_a$, 反复跑 $B$ 来 incrementally 更新 output。paper 把这种 axis 标记为 $s_b$ (stream batch size)。

两个性质 (Appendix A.1 Theorem 1, "Fusion Theorem"):

1. **Composition with E (weaved over the streamed axis) preserves streamability**: 因为 E 在 streamed axis 上是 elementwise/broadcast-like, 可以与 $B$ 拼成新 accumulator。
2. **Weaving over an extra axis preserves streamability**: 因为 map of composed = compose of maps。

这两条加起来意味着: 一旦你找到一个 streamable kernel, 你可以"在它周围贴"很多东西——比如在 SoftMax-Contraction 之前再 compose 一个 QK^T contraction (weaved by $s_x$), 然后 weave over q (queries) 和 d (head dim), 就自动得到了 attention 的 streamability。这是 paper Section 3.2 推 FlashAttention 的方式: **FlashAttention 不是手写出来的, 是从 SoftMax-Contraction 这个 seed kernel 通过两条 fusion rule 编出来的**。

这个视角与 Dao 2022 原论文 [FlashAttention](https://arxiv.org/abs/2205.14135) 的"online softmax + tiling"叙述比, 更强调**fusion 的代数可复合性**, 而非 "we cleverly notice that softmax can be computed in tiles"。这跟 [Ivanov et al. 2021 "Data movement is all you need"](https://arxiv.org/abs/2007.00072) 的"数据搬运视角"也合拍。

### 3.3 Contraction 的 accumulator

对向量点积 $v\cdot w = \sum_i v_i w_i$, 它是 streamable 因为:

$$v\cdot w = \sum_{i=0}^{s'-1} v_i w_i + \sum_{i=s'}^{n-1} v_i w_i$$

后半段就是 accumulator 的 input。所以 dot product 是 streamable kernel。

### 3.4 SoftMax-Contraction 的 accumulator

paper Table 4 给了三个 SoftMax 变体的伪代码 (Base / Auxiliary / Unscaled), 我挑 Auxiliary 的关键几行:

```
Initialize(): Return (-∞, 0, 0)
SoftMax_0(x⃗, (μ', δz', z')):
    μ ← max(β x_i, μ')
    s_i ← exp(β x_i − μ)
    δ ← exp(μ' − μ)
    z ← δ z' + Σ_i s_i
    y_i ← s_i / z
    Return y⃗, (μ, δz', z)
```

变量:
- $\mu$: 当前 running max
- $\mu'$: 前一批的 running max
- $\delta = e^{\mu'-\mu}$: 老概率要被 rescale 的因子 (这就是 online softmax 数值稳定的核心)
- $z'$: 前一批的 partial sum of exp
- $z$: 更新后的 sum

这就是 [FlashAttention Algorithm 1](https://arxiv.org/abs/2205.14135) 那个"两遍 SoftMax 用 max-rescale 拼起来"的代数身份, 但 paper 用 diagram + accumulator 把它"机械化"地接到了 contraction 后面。这一点很重要, 因为 paper 接下来要 weave 一个 QK^T contraction 在前面, 然后再 weave 在 q 和 d 上——所有这些步骤**不再需要新的数学灵感**, 只是 rule application。

---

## 4. 推导 Matrix Multiplication 和 Attention: 当 diagram 当 algebra

### 4.1 Matrix multiplication (Section 3.1)

从 dot product (streamable) 开始, weave 它在 a 和 c 两个 axis 上 → matmul。然后 group-partition over a 和 c, stream over b:

$$H = N_g H_g = \frac{ac}{g_a g_c}\,(g_a + g_a g_c) \cdot d \cdot (g_c + g_a s_b + s_b g_c)$$

化简 (Figure 18):

$$H = abc\,(g_c^{-1} + g_a^{-1}) + ac$$

受约束 $M \ge g_a s_b + s_b g_c$ (要装下 tile + accumulator 空间)。当 $g_a = g_c = g$ 时:

$$H \ge 2abc\,M^{-0.5} + ac$$

- $2abc M^{-0.5}$: tiling 代价, **$-0.5$ 次幂**说明 square tile 是最优
- $ac$: 输出写回成本 (与 tiling 无关, 只跟输出大小有关)

这给我们一个非常重要的 intuition: **指数 $\beta_t$ 直接告诉你 tile 的形状**。matmul 是 $-0.5$ → square; attention 是 $-1$ → 全 broadcast (KV 要 broadcast 到所有 query groups), 这就是 Section 4.1 的精辟总结。

#### Arithmetic intensity (Appendix B.1)

$$\frac{H}{K} = M^{-0.5} + (2b)^{-1} \quad\text{(transfers per FLOP)}$$

- $K = 2abc$: FLOPs
- 第一项来自 tiling 重复传输, 第二项来自输出 $ac$ 分摊到 $2abc$ FLOPs

H100 SXM5 的关键数字 (paper Appendix B.1 表):

- L2 带宽 3.35 TB/s, GMEM↔L2 等效 ~1.7e12 FP16 values/s
- SMEM 带宽 ~12 TB/s, 等效 6.0e12 FP16 values/s
- 绝对最小 L2 memory ≈ 681 KB, 绝对最小 b ≈ 295 values
- 对 SMEM 这一档, 绝对最小 b ≈ 82 values, 但实际 min b ≈ 151 values (考虑 SM 上 SMEM 分配)

这个 295 数值在 paper 后面变成"是否被带宽 bound"的临界值, 反复出现 (Section 5.7 又算了一次)。

### 4.2 Attention 的推导 (Section 3.2, Figure 21)

直接抄结果 (变量解释):

$$H = \frac{q}{g_q}\,(2 g_q d + 2 x d) = 2 q d + 2 x d q\,g_q^{-1}$$

约束 $M \ge 2 g_q d + 2 s_x d$

- $q$: sequence length (queries 数)
- $g_q$: query group size (一个 threadblock 处理多少 query)
- $d$: head dim
- $x$: KV 长度 (stream 的对象)
- $s_x$: KV stream batch size

最小化:

$$H \ge 2 q d + 4 x q d^{2} M^{-1}$$

注意 **$M^{-1}$ 而非 $M^{-0.5}$**——这就是 broadcast 的指纹: KV 被每个 query group 重新读一遍, memory 加倍不能让代价以 sqrt 衰减, 而是线性衰减。这点很关键, 因为它解释了:

- 为什么 attention 对 SRAM 大小极其敏感 (memory 加倍代价减半)
- 为什么 GQA/MQA 这类减少 KV head 数 (减少 broadcast 维度) 比单纯加 SRAM 更划算 ([Ainslie et al. GQA](https://arxiv.org/abs/2305.13245))
- 为什么 KV cache 在 long context 下是 bottleneck: $x d$ 这一项是 $O(x)$ 增长, 总 transfer 是 $O(x q)$

paper Figure 23 推 GQA 时也用同样手法, 但 GQA 把 query 一组 (g 个) 共享同一份 KV, 所以 broadcast 维度减少, transfer 类似 multi-head attention 但参数少——这就把 [GQA paper](https://arxiv.org/abs/2305.13245) 的实证结论"性能 ≈ MHA, 参数 ≈ MQA"用一张图还原了。

### 4.3 Multi-head attention (Figure 22)

$$H = 2 q h d + 2 x q h d\,(g_q g_h)^{-1} \ge 2 h q d + 4 h x q d^{2} M^{-1}$$

- $h$: head 数
- $g_h$: head group size

注意 $h$ 让 cost 线性增长, $d$ 让 cost 平方增长 (出现在 $d^2$ 项里)——这与 [Vaswani et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) 里"compute scales as $d \cdot h$ but memory scales as $d^2 \cdot h$"的直觉对得上, 但 paper 在 closed form 上给出了"为什么"。

---

## 5. 多层级性能模型: 一行公式把 hierarchy、量化、cross-transfer 全收进来

### 5.1 通用形式 (Equation 1, 2)

每个 two-level 优化给出:

$$H^*(\vec{a}, M) = \sum_t \alpha_t(\vec{a})\, M^{-\beta_t}$$

- $t$: 枚举项
- $\alpha_t(\vec a)$: 依赖 axis sizes 的系数
- $\beta_t \ge 0$: 敏感度指数, **越大表示越受益于 memory 扩大**

multi-level 总 weighted transfer cost:

$$H^* = \sum_\ell \dot{H}_\ell^{-1}\,H^*(\vec{a}, M_\ell) = \sum_t \alpha(\vec{a})\,\sum_\ell \dot{H}_\ell^{-1}\,M_\ell^{-\beta_t}$$

- $\dot{H}_\ell^{-1}$: level $\ell$ 的"per-value weighted transfer cost", 量纲是时间/字节之类
- $M_\ell$: 该 level 上分配到的 effective memory

直觉: 一个算法在不同 level 上分别按其 local optimization 跑, 总成本是各 level 的代价加权和。**这就把 GPU hierarchy 的"局部最优 → 全局可加"变成了可计算的命题**, 而不是模糊的"我们考虑了 L2、SMEM、registers"。

### 5.2 量化 (Equation 3)

每 value $q$ bytes:

$$M_\ell = \bar M_\ell^{\text{Bytes}} / q, \quad \dot H_\ell^{-1} = (\dot H_\ell^{\text{Bytes}}/q)^{-1}$$

代入:

$$H^{*\,\text{Bytes}} = \sum_t \alpha(\vec a)\,\sum_\ell (\dot H_\ell^{\text{Bytes}})^{-1}\,(M_\ell^{\text{Bytes}})^{-\beta_t}\, q^{1+\beta_t}$$

**关键 insight: 因为 $1+\beta \ge 1$, 总 transfer 对 quantization degree 是超线性的**。

- Attention ($\beta=1$): FP32→FP16 加速 ×4 ($=2^{1+1}$)
- 大 matmul ($\beta=0.5$): 加速 ×$2^{1.5}\approx 2.83$

这就解释了为什么 [GPTQ (Frantar et al.)](https://arxiv.org/abs/2210.17323) 和 [FP8 FlashAttention-3](https://arxiv.org/abs/2407.08608) 的实测收益远比"理论上按 byte 减半算"更大——因为 memory savings 直接降低了 broadcast 项, broadcast 项又跟 attention 的总成本成 $1$ 次比例, 所以收益是 byte 节省的平方根的反比再平方, 直观但 paper 第一次写成可证伪公式。

### 5.3 Intermediate caching (Section 4.4, Theorem 2)

把上一层 (比如 L2) 当 cache, 输出存在 lower level (SMEM/registers) 直到攒够再写回。等效于把 intermediate level 的 effective memory 换成 $M_{\ell2} N_{\ell2}^{\max}$:

$$H^*_{\ell1} = H^*(\vec a, M_{\ell2}\,N_{\ell2}^{\max})$$

- $N_{\ell2}^{\max}$: lower level 上能有多少 child 同时活
- 约束 $N_{\ell2}^{\max} \ge N_{g,\ell2}/N_{g,\ell1}$: lower level 数量限制

直觉: cache 这一层的"虚拟容量"就是下面所有 children 的总和, 而 cache 本身不挤占 SMEM 的 hardware 上限。这是 Hopper 上"用 SMEM 当 L2 写回 cache"的代数身份。

### 5.4 Cross-transfer level (Section 4.5, Theorem 3, Appendix A.2.2)

这一节最妙, 它对应 Hopper 的 thread block cluster (SM 间直接通信) 和 multi-GPU NVLink 拓扑。引入 cross-transfer level $x$ 介于 $h$ 和 $c$ 之间:

$$H^* \mapsto \ldots + (\dot H_{hc}^{-1} - \dot H_{xc}^{-1})\,H^*(\vec a, N_c^{\max} M_c) + \dot H_{xc}^{-1}\,H^*(\vec a, M_c) + \ldots$$

直觉: 一部分数据直接由 $h$ 发到每个 child (享受 $N_c^{\max} M_c$ 的"超大虚拟 memory"), 剩下的数据 child 之间 cross-transfer (按 $\dot H_{xc}^{-1}$ 算)。改写后, 这相当于插入一个新 level $x$, 它的 effective memory 是 $N_c^{\max} M_c$, 它的 transfer cost 是"差价" $\dot H_{hc}^{-1} - \dot H_{xc}^{-1}$。

paper Appendix A.2.3 给了 H800 (中国版 Hopper) 的实测数字 ([Luo et al. 2024](https://arxiv.org/abs/2402.13499)):

- cluster size N=2: 3.27 TB/s
- cluster size N=4: 2.65 TB/s
- 普通 GMEM↔SMEM: 2.04 TB/s

这给了 optimize $N$ 的明确公式:

$$\Delta H^* = \Delta \dot H^{-1}(N)\,\sum_t \alpha_t(\vec a)\, M_c^{-\beta_t}\,(1 - N^{-\beta_t})$$

cluster size 增大让 cross-transfer 折扣变大 (后面的 $1 - N^{-\beta_t}$ 增大), 但让 $\dot H_{xc}^{-1}$ 也增大 (带宽下降)——这是 **multi-GPU / cluster size 选择的 quantifiable tradeoff**, 比"经验上 N=2 比较好"强多了。

---

## 6. 从 diagram 到 Hopper 上的 pseudocode: 八步法

Section 5 是 paper 最 dense 也最 useful 的部分, 把抽象 two-level diagram 一步步 refine 到 Hopper 上的可运行 kernel sketch。八步 (Step 1–5 + 配置):

### Step 1 (Figure 29): Expand streamed algorithm 到 looped pseudocode

把 streamable kernel 写成 `init → loop(B) → finalize` 形式, 所有变量显式画出。对 attention, 用 QK matmul 当 E, SoftMax-Contraction accumulator 当 B, 再 weave over q 和 d。这就是 paper Figure 29, 一步到位。

### Step 2 (Figure 30): 识别 subloops

- Accumulator 内部本身是 streamable → 总能再开 subloop stream $s_a$ 切到 $u_a$
- matmul 可以沿任意 axis split → 累加 (linear subloop)
- dotted box 标记可以 split 的 op

这一步**暴露 algorithm 的自由度**: 哪些 axis size 是真正可配置的。和 [Stream-K work decomposition (Osama et al. 2023)](https://arxiv.org/abs/2301.03598) 的思路精神一致——尽量利用 split。

### Step 3 (Figure 31): Recoloring for tensor cores

- matmul 涂蓝 (tensor core op)
- elementwise / SoftMax 涂绿 (general-purpose register op)
- $g_q \to t_q$ (per-thread) 或 $w_q$ (per-warpgroup of 128 threads)
- 在 matmul 上叠 quantization tag (FP8 / FP16)
- 中间需要 SMEM pipe 把 coherent data fragment 给 tensor core

paper 这里点出一个**自己的方法局限**: 第二个 matmul 后面有个"scaling by $\delta$ weave over $g_q$" 操作, tensor core 上做这种 diagonal-scale 不在 standard paradigm 里, 所以 paper 的 sketch 用了 SMEM data 来做, 而 FlashAttention-3 用 thread-wise fragment manipulation 做了。这是 Section 6 比较 FlashAttention 时 paper 主动承认的盲点。

### Step 4 (Figure 32): Divisor constraints

- $w_q^{(128)}$ — warpgroup 必须是 128 倍数
- $s_x^{(32)}$ — tensor core K-dim 必须是 32 倍数
- $u_x^{(8)}$ — subloop tile 必须是 8 倍数
- $d^{(128)}$ — head dim 128
- $d'^{(16)}$ — output 分块 dim 16

superscript 表示"必须被整除"。这些约束直接来自 [NVIDIA H100 architecture guide](https://resources.nvidia.com/en-us-tensor-core) 和 [PTX ISA 8.5](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)。

### Step 5 (Figure 33): 识别所有 variables

每个 data column 列出, 看哪些同时在场, 然后预分配。比如 Table 1:

| Variable | Size | Q. | Level |
|---|---|---|---|
| Q | $w_q^{(128)}\times d^{(128)}$ | FP8 | SMEM |
| K | $s_x^{(32)} \times d^{(128)}$ | FP8 | SMEM |
| V | $s_x^{(32)} \times d^{(128)}$ | FP16 | SMEM |
| S (scores) | $w_q^{(128)} \times s_x^{(32)}$ | FP16 | Registers |
| ... | ... | ... | ... |

### Step 6 (Table 1, 2): Configuration table + max warpgroups

设 $w_q=128, t_q=1, s_x=u_x=64, d=128, d'=32, d''=8$:

- SMEM per TB: 48 KB
- SMEM per WG: 48 KB
- Register per WG: 74.75 KB
- $N_{\max}^{\text{SMEM}} = 3.7$, $N_{\max}^{\text{Reg}} = 3.4$
- 实际选 N=3 warpgroups per SM

这个表把"哪些配置可行"变成线性约束 $N_{\max} = (M_{\max} - M_{TB})/M_{WG}$, 跟硬件 spec sheet 一对就完事, 不用凑。

### Step 7 (Table 3): Throughput 分析

每个 op 算 ops/thread, ops/clock, clock/thread:

| Op | Pipeline | Ops/Th | Ops/Clk | Clk/Th |
|---|---|---|---|---|
| Q-K MatMul | Tensor | 16384 | 8192 | 2 |
| SoftMax exp | SFU | 65 | 16 | 4.06 |
| P-V MatMul | Tensor | 16384 | 4096 | 4 |
| FP16 accumulate | FP16 | 256 | 512 | 0.5 |

总 Clk/Th ≈ 6, 下界由 tensor core 决定。然后判断是否 bandwidth-bound:

$$k_{TC} g_q / f_K \ge H \cdot B / N_{SM}$$

H100 SXM5 上算出 $g_q \ge 295$ 才能不被带宽 limit (无 caching 假设下)。在 $N=3$ warpgroups、$g_q=384$ 时 compute time $1.26 \mu s$, transfer time $0.97 \mu s$, 所以 compute bound, 理想 throughput 1.32 PFLOP/s。

### Step 8 (Figure 34, 35): Pipelining diagram

最后画出"哪些 op 在哪个 clock cycle 段跑", 用宽度对应 Clk/Th, 用 hatched region 留 overhead buffer。Figure 35 是三 warpgroup、双 iteration overlapping 的最终 sketch:

- 第一 warpgroup: 等 tensor core
- 第二 warpgroup: 跑 SoftMax 在 SFU 上 (hatched 表示 50% overhead 空间)
- 第三 warpgroup: FP16 accumulate
- barriers 用 thick dotted lines, 都 wait on tensor core

这一步对应 [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608) 里的"async pipelining" 图示, 但 paper 把它从"看 diagram 凑"变成"算 Clk/Th 后必然得到的 layout"。

---

## 7. 与 FlashAttention-3 的对照: 可证伪的猜测

paper Section 6 是最 sharp 的部分, 它把自己的 model 当 hypothesis, 用 FlashAttention-3 的实测数据反推哪里假设错了。

### 7.1 FP16 FlashAttention-3

- FA3 实测: 740 TFLOP/s / 989 peak = 75%
- paper 的 inter-warpgroup 假设下应该能塞下 100% overhead 给 SoftMax (Figure 36 上半)
- 但 FA3 实际用 intra-warpgroup + $g_q=128$ 大 tile, SoftMax 没有 overhead 容纳空间

paper 的猜想: **如果 intra-warpgroup 真的更快, 那 toy model 里"tensor core overhead 可以忽略"的假设是错的**——尤其 small tile 的 tensor core overhead 可能远比想象大。这是 paper 自己点出的"待证伪"。

另一种假设: 假设 SoftMax 66% overhead 时:
- FP16 intra-warpgroup: 应该花 4/3 时间 → 75% utilization ✓ (匹配实测)
- FP8 inter-warpgroup: 应该花 5/3 时间 → 60% utilization ✓ (匹配实测)

这两个数对得上, 是 paper 模型的最强 evidence。

### 7.2 FP8 FlashAttention-3

- FA3 实测: 1.2 PFLOP/s / 1.98 peak = 60%
- paper 分析: FP8 bottleneck 是 SFU (SoftMax 的 exp), 与 tensor core ops 用一样多 clock
- 任何 overlapping 策略都被 SFU 限制
- Figure 37 显示 intra-warpgroup FP8 时 SFU 大部分时间 idle (至少 33% 浪费)

这就是 60% 的来源, **不是 small tile overhead, 是 SFU 与 tensor core 争抢 scheduling**。这个 hypothesis 可以被 microbenchmark 验证: 如果 FP8 attention 用 fused exp approximation (e.g. [FastExp tricks](https://arxiv.org/abs/2205.14135) 里提到的 polynomial approximation) 替代 SFU exp, 是否能把 60% 推高? 这是个很 actionable 的下一步。

### 7.3 paper 自己承认的盲点

> A major concern is our lack of utilization of tensor core-fragmented register-level operations.

也就是 thread-wise fragment manipulation (CUTLASS 的 [CuTe layouts](https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf)), 这是 [Colfax 的 FlashAttention-2 case study](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-flashattention.pdf) 里 Jay Shah 等人搞出来的, FA3 用得更狠。Paper 的 diagram 把 tensor core memory 当"incoherent", 不能表达 fragment 分布在 warp/thread 上的细节。要补这块需要引入 **reindexing weaves** ( Abbott 2023 已铺路 ), 对应 categorical Yoneda natural transformations。

---

## 8. 大量联想与未来工作 (Section 7 + paper 没明说的)

paper 末尾提到一堆方向, 我把每条都展开联想一下:

### 8.1 MoE

> Mixture-of-expert models use immense resources, making optimizations particularly impactful.

[Mixtral](https://arxiv.org/abs/2401.04088) 和 [Llama 3 herd](https://arxiv.org/abs/2407.21783) 的 expert routing 是个特殊的 diagram pattern: 一个 dispatch axis 把 token 分发到 expert sub-functions。用 weaving 视角, 这是 "weave + reindex + select"。MoE 的瓶颈不是 expert 内 matmul, 而是 **expert 间的 token redistribution (all-to-all)**, 这是 cross-transfer level 的典型 case, 直接套 Section 4.5 的 $\Delta H^* = \Delta \dot H^{-1}(N)\sum_t \alpha_t M_c^{-\beta_t}(1 - N^{-\beta_t})$ 公式就能算 expert 数 vs NVLink 带宽的最优 N。

### 8.2 Convolution / sliding window attention

> Convolution and sliding window attention use reindexings on weavings. These operations change how data is accessed without changing the data itself, and correspond to Yoneda natural transformations from category theory.

[Longformer (Beltagy et al.)](https://arxiv.org/abs/2004.05150) 这种 sliding window attention 是 attention + 一个 window reindex 的复合。diagram 上 sliding 等价于 "在 stream axis 上每次偏移一个固定 pattern 再 reduce", 这是典型的 [Leijns 上的 Yoneda 自然变换](https://arxiv.org/abs/2305.08768)。一旦 reindex weaving 进了记号, convolution / im2col / unfold 全部变成"换索引不换值"的同一类操作, 也许能解释为什么 [cuDNN winograd](https://docs.nvidia.com/deeplearning/cudnn/latest/) 这种"换 basis 计算 conv"在小 tile 上比 im2col 更快——它就是 reindex 把 reduce 的 $\beta_t$ 从 $0.5$ 改成更小。

### 8.3 Backprop

paper 没明说但提了 [Backprop as Functor (Fong et al. 2019)](https://ieeexplore.ieee.org/document/8785665) 的反向传播 categorical 形式化。Abbott 的 diagram 是 forward 的, 但同样有 reverse-mode automatic differentiation 的 diagram dual。如果加上, 那么 backward attention 也能用同样八步法推 Hopper 上的 kernel, 而且 forward/backward 共享 streaming 性质 (因为 reverse of streamable 仍 streamable, 这是 [Cruttwell et al. 2021](https://arxiv.org/abs/2103.01931) 的 reverse functor 性质)。这能解释为什么 FA2 backward 不需要重新设计, 同样 tiling 适用。

### 8.4 Categorical co-design

Zardini 自己的 thesis [Co-design of Complex Systems](https://www.research-collection.ethz.ch/handle/20.500.11850/648075) 把 functionality ↔ requirement 做成 profunctor, 现在加上 performance model 公式 $H^* = \sum_t \alpha_t \sum_\ell \dot H_\ell^{-1} M_\ell^{-\beta_t}$, 就能问: 给定 throughput target $T$ 和 energy budget $E$, 求 Pareto frontier 上 $(M_\ell, \dot H_\ell^{-1}, q, N_c)$ 配置。这等价于把硬件 spec、quantization、cluster size 当成 co-design 上的 4 个 profunctor 同时 optimize。

### 8.5 Multi-GPU / Blackwell

- Blackwell ([NVIDIA 2025 brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief)) 引入 tensor memory (TMEM) 和新的 5th-gen tensor core, 这是新的一个 level, 直接进 Section 4.5 cross-transfer 模型
- [AMD CDNA 3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) 的 MFMA 与 Hopper 的 wgmma 不同, 但 divisor constraints 类似, 同一套 Step 4 换参数即可
- multi-GPU NVLink: 等价 cross-transfer level, 直接套 Theorem 3

### 8.6 Streaming 与 Stream-K

[Stream-K (Osama et al.)](https://arxiv.org/abs/2301.03598) 把 matmul 的 K 维 split 到所有 SM, 避免"tile 完整 SM 才能开始"的浪费。这跟 paper Section 5.2 的 subloop 思路一致, 都是利用 linear accumulator (matmul 的 split 累加性)。如果 diagram 加上"accumulators-as-reductions", Stream-K 可以作为 diagram 的一个标准 rewriting rule, 不需要单独 kernel 库。

### 8.7 KV cache 与 quantization 的代数

paper 公式 $q^{1+\beta}$ 那一项对 KV cache compression 是直接 quantitative motivation:
- FP8 KV (q=1) vs FP16 KV (q=2): attention 加速 ×4
- INT4 KV (q=0.5): 理论 ×16 但受限于 attention 必须保留 logits 精度, 实际 ~×6 ([GPTQ](https://arxiv.org/abs/2210.17323) 实测)
- 这是 paper Figure 32 那种 quantization tag 的量化收益估计的"理论值", 可以拿来跟 [SmoothQuant (Xiao et al.)](https://arxiv.org/abs/2211.10438) 这类混合 quantization 做对比

### 8.8 Compiler 工程化

paper 的 configuration table 直接可以喂给 [Triton](https://dl.acm.org/doi/10.1145/3315508.3329973) / [CUTLASS](https://github.com/NVIDIA/cutlass) / [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 当 search space 的 prior。每个 axis size 的 divisor 约束就是 search space 的合法区域, $\beta_t$ 决定 sensitivity, 所以可以 enumerate 一小组 candidate 而不是 grid search。这是 diagram 工程化的最直接路径。

### 8.9 不同 SMEM 分配策略

H100 SMEM/RMEM 分配是可调的 (max 228 KB SMEM)。paper Table 2 算出 $N_{\max}^{\text{SMEM}}=3.7$, $N_{\max}^{\text{Reg}}=3.4$, 取 3 留 buffer。但 [Ootomo & Yokota 2023](https://arxiv.org/abs/2308.15152) 的工作显示: 减小 SMEM footprint 让更多 threadblock 并发反而比大 SMEM tile 快。paper 的 $\beta=1$ 公式给出明确判断标准——当 $g_q$ 已 ≥ 295 时, 不再 bandwidth bound, 此时减小 SMEM 让 $N$ 增加确实更快, 因为 compute bound 下 occupancy 决定一切。

---

## 9. 一些 critical 的小观察

### 9.1 paper 自己没有开源 code

paper 反复提"attached document"——它配了 Excel 风格 spreadsheet 算配置。这意味着 paper 的"理论"和"代码"之间还有一个 hand-rolled spreadsheet 的中间层, 离真正自动 compile 还远。但 paper 7 节说 categorical 形式化后能用 [Wilson 2023](https://eprints.soton.ac.uk/483757/) 这类 polynomial circuit 学习的 syntax → compile 自动化, 这是路线图。

### 9.2 $\beta_t$ 是不是真的可测?

paper 假设 $\beta_t \in \{0.5, 1\}$, 实际 hardware 上 L2 / SMEM / TMEM 的"等级"会让 $\beta$ 不同。能否设计 microbenchmark 直接 fit $\beta$? 比如:
- 固定 algorithm (matmul), 改 SMEM 大小 (用 `cudaFuncSetAttribute`)
- 测 H, fit $\beta$
- 若拟合 ≠ 0.5, 说明 L2 cache prefetch 在偷偷帮忙

这是 paper 风格的"做实验测假设"的最小可行实验, 应该很容易做。

### 9.3 与 [Luo et al. H800 benchmark](https://arxiv.org/abs/2402.13499) 的对齐

Luo 等人对 H800 (中国版 Hopper, 减少 NVLink 带宽) 的详细 benchmark 显示 thread block cluster 的 bandwidth 取决于 cluster size $N$, 这正是 paper Theorem 3 想建模的。用 Luo 的数据 + paper 的 $\Delta H^*$ 公式可以直接选最优 cluster size, 这是 paper 框架最 immediate 的工业可用 case。

### 9.4 Power / 能耗

paper Abstract 没强调但提到: "DRAM already accounts for 46% of total system power ([Ghose et al. 2018](https://arxiv.org/abs/1807.05102))". 这个数字今天更高。paper 的 $H^* = \sum_t \alpha_t \sum_\ell \dot H_\ell^{-1} M_\ell^{-\beta_t}$ 公式直接给一个能耗 proxy (transfer 量 × byte-per-joule), 加上 compute 的 FLOPs/W, 可以变成 energy-aware co-design 的 objective function。

---

## 10. 一句话总结

这篇 paper 把 GPU IO-aware 优化从"老中医把脉" 拉向 "实验生理学": 它给出一个 categorical string-diagram 记号 + 两条 fusion rule + 一个 $\sum_t \alpha_t M^{-\beta_t}$ closed-form, 然后用这把尺子量 FlashAttention-3 的 FP16 75%、FP8 60% 两个数, 发现"如果 SoftMax 有 66% overhead, 模型预测刚好对得上"——这是一种可以证伪的 claim, 而不是 post-hoc rationalization。它的盲点是 tensor core fragment 的 thread-wise manipulation (CuTe layout), 它的下一步是 reindex weaving + backprop + multi-GPU cluster + MoE routing + Blackwell TMEM, 每一个都已经有相应的 categorical 工具等在那里 (Yoneda, profunctor, functorial backprop, co-design)。

---

## 主要 reference 链接

- 本 paper: arxiv 暂未见, 作者主页 [Vincent Abbott](https://www.vtabbott.io/) / [Gioele Zardini @ MIT LIDS](https://lids.mit.edu/people/gioele-zardini)
- [FlashAttention (Dao 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao 2023)](https://openreview.net/forum?id=mZn2Xyh9Ec)
- [FlashAttention-3 (Shah et al. 2024)](https://arxiv.org/abs/2407.08608)
- [Robust Diagrams for Deep Learning Architectures (Abbott 2023)](https://www.vtabbott.io/content/files/2023/11/Robust-Diagrams-for-Deep-Learning-Architectures.pdf)
- [AI and the Memory Wall (Gholami et al. 2024)](https://arxiv.org/abs/2403.14123)
- [GQA (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245)
- [Attention is All You Need (Vaswani et al. 2017)](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- [Triton (Tillet et al. 2019)](https://dl.acm.org/doi/10.1145/3315508.3329973)
- [PyTorch (Paszke et al. 2019)](https://arxiv.org/abs/1912.01703)
- [GPTQ (Frantar et al. 2023)](https://arxiv.org/abs/2210.17323)
- [Longformer (Beltagy et al. 2020)](https://arxiv.org/abs/2004.05150)
- [Mixtral (Jiang et al. 2024)](https://arxiv.org/abs/2401.04088)
- [Llama 3 herd (Dubey et al. 2024)](https://arxiv.org/abs/2407.21783)
- [Stream-K (Osama et al. 2023)](https://arxiv.org/abs/2301.03598)
- [Benchmarking Hopper (Luo et al. 2024)](https://arxiv.org/abs/2402.13499)
- [Data movement is all you need (Ivanov et al. 2021)](https://arxiv.org/abs/2007.00072)
- [Reducing shared memory footprint (Ootomo & Yokota 2023)](https://arxiv.org/abs/2308.15152)
- [Colfax FlashAttention-2 case study (Bikshandi et al.)](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-flashattention.pdf)
- [CuTe Layouts (Shah 2024)](https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf)
- [NVIDIA H100 architecture](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA Blackwell brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief)
- [AMD CDNA 3 white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [PTX ISA 8.5](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)
- [DRAM power models (Ghose et al. 2018)](https://arxiv.org/abs/1807.05102)
- [String diagrams for CS (Piedeleu & Zanasi 2023)](https://arxiv.org/abs/2305.08728)
- [Backprop as functor (Fong et al. 2019)](https://ieeexplore.ieee.org/document/8785665)
- [Categorical foundations of gradient-based learning (Cruttwell et al. 2021)](https://arxiv.org/abs/2103.01931)
- [Co-design of Complex Systems (Zardini 2023)](https://www.research-collection.ethz.ch/handle/20.500.11850/648075)
- [Category-theoretic data structures for polynomial circuits (Wilson 2023)](https://eprints.soton.ac.uk/483757/)
- [Stable Diffusion 3 / Rectified flow (Esser et al. 2024)](https://arxiv.org/abs/2403.03206)
- [SDXL (Podell et al. 2023)](https://arxiv.org/abs/2307.01952)
- [Latent Diffusion (Rombach et al. 2022)](https://arxiv.org/abs/2112.10752)
- [DDPM (Ho et al. 2020)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
- [XLA (Sabne 2020)](http://research.google/pubs/xla-compiling-machine-learning-for-peak-performance/)
- [IO complexity of sorting (Aggarwal & Vitter 1988)](https://dl.acm.org/doi/10.1145/48529.48535)
- [Fast transformer decoding / MQA (Shazeer 2019)](https://arxiv.org/abs/1911.02150)
- [Markov categories and entropy (Perrone 2022)](https://arxiv.org/abs/2212.11719)

如果你想我进一步 drill down 某一块——比如把 Figure 21 那个 attention 推导一行行重写一遍, 或者用 paper 的公式去 fit 一组 Blackwell 的预期 throughput, 或者把 Theorem 3 cross-transfer 推导重写一遍——直接说, 我都能展开。
