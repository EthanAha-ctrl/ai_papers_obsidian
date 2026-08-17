---
source_pdf: Interval Tree Clocks A Logical Clock for Dynamic Systems.pdf
paper_sha256: af722364a4b4027fbe4c1d01cbab1665fc8bcfa0cadefd8fb56b4d1f5787b724
processed_at: '2026-08-05T10:20:41-07:00'
target_folder: Automata
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ITC 用人话讲

Andrej，我换一种讲法。之前那版是按论文章节走的，这版我按"想法是怎么长出来的"来讲，顺便多串一些联想。

---

## 0. 一个直觉性的开场

想象你在管一群仓库管理员，每个管理员要记账：自己经手过多少批货，也大概知道别的管理员经手过多少批。Vector Clock 就是这么个账本——一行一个管理员编号，旁边一个数字。

问题是：管理员会入职、会离职、会跳槽、会失联。Vector Clock 假设编号是固定座位号，人走了座位号还占着，新来的不能坐旧座位（怕账对不上），结果账本越来越长，还清不掉。

ITC 的做法是：**不要编号，发地皮**。把一块叫 $[0,1)$ 的"地皮"分给管理员们，每个人占一段。入职就切一块下来给他，离职就把他的地皮并回去。地皮可以无限二分（切到很细很细），也可以并回来。账本上记的不再是"X 编号经手了几批"，而是"在我管的这段地皮上发生过几次"。这就是全部 idea。

下面把这件事的每个环节拆开讲。

---

## 1. 为什么 classic vector clock 在 dynamic 系统下崩

Vector Clock [Fidge 1989, Mattern 1989] 的形式是：

$$
e = (c_1, c_2, \dots, c_n)
$$

- $e$：event component，一个长度 $n$ 的整数向量
- $c_k$：第 $k$ 个 participant 的本地 counter
- $n$：participant 总数，**必须预先知道**

每个 participant $p_k$ 跑 event 时：
$$
c_k := c_k + 1, \quad c_j := c_j \; (j \ne k)
$$

发消息就是把这个 vector 复制一份塞进 envelope；收消息就是 pointwise max 然后自己 +1。

这套机制在 $n$ 固定时完美。一旦 $n$ 会变，麻烦就来了：

**问题 A：新人入场怎么分配 id？**
- 如果用全局 counter 发号，需要一个 coordinator，违反 decentralized 原则
- 如果用 UUID / MAC，id 不可回收，vector 永远只增不减
- Bayou [18] 让每个 replica 自己造 id（用本地 counter + 时间戳），但同样不可回收

**问题 B：老人退场怎么回收 id？**
- 要安全回收，必须确认所有 active participant 都已经"看到"这个老人死了。否则你删掉他的 entry，但有个慢节点还拿着他三个月前的旧消息，下次 pointwise max 就会错乱
- 这就要求一个 **全局 GC 协议**，只要一个节点 unreachable 就 stale
- Dynamo [6] 在生产中干脆"激进裁剪"，砍掉老 entry，代价是偶尔有老 update 复活（resurgence）

**问题 C：作者自己 2002 年的 Version Stamps [2]** 试图做 localized retirement，理论可行但实际"结构增长失控" [3]——空间消耗随时间发散。

ITC 同时解决 A、B、C：**无全局 id 分配 / 局部创建 / 局部回收 / id 可复用 / 空间稳定**。参考综述 https://link.springer.com/article/10.1007/s00446-008-0068-3

---

## 2. 三个 core operations：把 distributed computation 代数化

作者把所有 logical clock 操作抽象成三个算子，作用于一个 stamp $(i, e)$：
- $i$：identity，"我是谁"
- $e$：event component，"我看到了哪些 event"

### fork

一个 participant 想生一个孩子。孩子继承父亲的全部 causal past（event component 不变），但要把 identity 一分为二：

$$
\text{fork}((i, e)) = ((i_1, e), (i_2, e))
$$

约束：
$$
i_1 \cdot i_2 = \mathbf{0} \quad \text{（不相交，不能两个人抢同一块地皮）}
$$
$$
i_1 + i_2 = i \quad \text{（守恒，父亲的地皮全分掉）}
$$

这就像分家产——父亲有 100 亩地，分给两个孩子各 50 亩，两个孩子的地不能重叠，加起来必须等于父亲原来的 100 亩。

### peek

fork 的退化版本，生成一个"匿名 stamp"：

$$
\text{peek}((i, e)) = ((\mathbf{0}, e), (i, e))
$$

匿名 stamp 的 $i = \mathbf{0}$，不能 event，只能携带 causal 信息。**Message 就是匿名 stamp**——一封信不需要"身份"，只需要告诉收信人"我看到的事件"。

### event

participant 干了一件事，要在自己的 $e$ 上记一笔：

$$
\text{event}((i, e)) = (i, e') \quad \text{s.t.} \quad e < e'
$$

严格 advance，且 $e'$ 恰好 cover $e$：不能跳过别人没看到的 event，也不能多记。

在 vector clock 里就是 $c_k := c_k + 1$。在 ITC 里，**自由度大得多**——你可以在 id 拥有的任意子区间上 +1，甚至可以同时 +1 多个区间。这个自由度后面用来压缩表示。

### join

两个 participant 合并（比如一个 replica 要下线，把它的状态并给另一个）：

$$
\text{join}((i_1, e_1), (i_2, e_2)) = (i_1 + i_2, \; e_1 \sqcup e_2)
$$

- id 是 **sum**：两个地皮合并
- event 是 **join**（pointwise max）：两个账本取并集

### 经典操作 = 复合

| 经典操作 | 复合 | 含义 |
|---|---|---|
| send | event ∘ peek | 自己 +1，然后复制一份 $e$ 进 message |
| receive | join ∘ event | 先把 message 的 $e$ 并进来，再自己 +1 |
| sync | join ∘ fork | 双向同步：先 join，再 fork 出两个新 stamp |
| spawn | fork | 父亲 fork 出孩子 |
| terminate | join | 孩子 join 回父亲 |

整个 distributed computation 的图论，被压缩成三个算子在一个 join semilattice 上的演算。这种 abstraction 后来被 CRDT 大量借鉴 https://link.springer.com/chapter/10.1007/978-3-642-25408-9_19

---

## 3. 从"离散 id"到"连续 interval"——ITC 的关键跳跃

### Function space framework

作者先升一层抽象：把 $i$ 和 $e$ 都看成从某个 domain $D$ 到 $\mathbb{N}$ 的函数。
- $i: D \to \{0, 1\}$ 是 **characteristic function**：$i(x) = 1$ 表示 $x$ 这个位置属于我
- $e: D \to \mathbb{N}$ 是 event function：$e(x)$ 表示"在位置 $x$ 上发生过几次 event"

**关键不变量**：所有 participant 的 id 函数两两不相交：
$$
\forall i_1 \ne i_2. \; i_1 \cdot i_2 = \mathbf{0}
$$

只要这个不变量成立，causality tracking 就是正确的。

### Classic mechanisms 选 $D = \{1, 2, \dots, n\}$

离散有限 domain，每个 participant 占一个点。一旦 $n$ 变了，整个 domain 要重构。

### ITC 选 $D = [0, 1)$

连续无限 domain，可以无限二分。一个 participant 占的是 $[0, 1)$ 上的一段区间（或几段）。这个 domain 永远是 $[0, 1)$，**不会因为 participant 数量变化而改变**。

这就是 ITC 的核心 insight：**用连续 domain 换取 id 管理的局部性**。要多少地皮就切多少，永远够用，永远不需要协调。

---

## 4. 用二叉树表示 $[0, 1)$ 上的函数

连续 domain 上的函数没法直接存，要有限编码。ITC 用二叉树。

### id tree

语法：
$$
i ::= 0 \mid 1 \mid (i_1, i_2)
$$

- $0$：空集（在 $[0,1)$ 上值为 0 的函数）
- $1$：全集 $[0,1)$（在 $[0,1)$ 上值为 1 的函数）
- $(i_1, i_2)$：左子树管 $[0, 1/2)$，右子树管 $[1/2, 1)$

语义函数 $\llbracket \cdot \rrbracket$：
$$
\begin{aligned}
\llbracket 0 \rrbracket(x) &= 0 \\
\llbracket 1 \rrbracket(x) &= 1 \quad \text{for } x \in [0,1) \\
\llbracket (i_1, i_2) \rrbracket(x) &= \begin{cases} \llbracket i_1 \rrbracket(2x) & x \in [0, 1/2) \\ \llbracket i_2 \rrbracket(2x - 1) & x \in [1/2, 1) \end{cases}
\end{aligned}
$$

变量解释：
- $x \in [0, 1)$：domain 中的位置
- $2x$：把 $[0, 1/2)$ "拉伸"回 $[0, 1)$ 给左子树看
- $2x - 1$：把 $[1/2, 1)$ 平移并拉伸回 $[0, 1)$ 给右子树看

举例，id 树 $(1, (0, 1))$ 长这样：

```
        (1, (0,1))
         /      \
        1      (0,1)
       [0,1/2)   /  \
                0    1
            [1/2,3/4) [3/4,1)
```

这个 participant 占据的区间是 $[0, 1/2) \cup [3/4, 1)$。每个叶子对应一个 dyadic interval（形如 $[k/2^d, (k+1)/2^d)$ 的区间），叶子为 1 表示"这块地皮归我"。

**联想**：这和 **buddy memory allocator** https://en.wikipedia.org/wiki/Buddy_memory_allocation 一模一样——buddy allocator 也是从 $[0, 1)$（或一段内存）开始，需要就二分，free 时如果兄弟正好互补就合并。ITC 的 fork/join 和 buddy 的 split/merge 是同构的。区别只是 buddy 切的是物理内存，ITC 切的是抽象的"identity 空间"。

### event tree

语法：
$$
e ::= n \mid (n, e_1, e_2)
$$

- $n$：常数函数，整个 $[0,1)$ 上值都是 $n$
- $(n, e_1, e_2)$：base value $n$ + 左右子树的 detail

语义：
$$
\begin{aligned}
\llbracket n \rrbracket(x) &= n \\
\llbracket (n, e_1, e_2) \rrbracket(x) &= n + \begin{cases} \llbracket e_1 \rrbracket(2x) & x \in [0, 1/2) \\ \llbracket e_2 \rrbracket(2x - 1) & x \in [1/2, 1) \end{cases}
\end{aligned}
$$

**这是 Haar wavelet 的多分辨率分析** https://en.wikipedia.org/wiki/Haar_wavelet。$n$ 是当前 scale 的 DC coefficient（低频、粗分辨率），$e_1, e_2$ 是更细 scale 的 detail（高频）。

为什么这种表示好？因为 event function 通常是 **分块常数** 的：某个 participant 在自己拥有的一段地皮上反复 +1，他这段的值高，别人那段低。Haar 表示让"两半共有的 base" 提到上层，"两半的差异"留在子树，从而自动压缩。

举个 event tree 例子 $(1, 2, (0, (1, 0, 2), 0))$：

```
         (1, 2, (0, (1,0,2), 0))
          /        |        \
         1         2     (0, (1,0,2), 0)
       base       left       right
       
       right 展开:
       (0, (1,0,2), 0)
        /     |     \
       0   (1,0,2)   0
            /  |  \
           1   0   2
```

读法：从 root 往下走，每下一层 base 累加。
- $[0, 1/2)$：root base $1$ + left 子树 $2$ = **3**
- $[1/2, 3/4)$：root base $1$ + right base $0$ + 左孙子 base $1$ = **2**
- $[3/4, 7/8)$：root base $1$ + right base $0$ + 右孙子左子 $0$ + 左孙子左子 $1$ = 等等，我重算一下

实际更清晰的算法是：函数在某点的值 = root 到该点对应的叶子路径上所有 base value 之和。比如 $x = 0.1$（落在 $[0, 1/2)$，左走），值 = $1$ (root) + $2$ (left 子树就是常数 2) = 3。$x = 0.6$（落在 $[1/2, 3/4)$，右走再左走），值 = $1$ + $0$ + $1$ = 2。$x = 0.9$（落在 $[7/8, 1)$，右右右走），值 = $1$ + $0$ + $0$ + $0$ + $2$ = 3。

---

## 5. Normalization —— "提取公因子"的魔法

同一函数有多种表示，比如 $\mathbf{1}$（在 $[0,1)$ 上恒为 1）可以写成：
$$
1 \equiv (1, 1) \equiv (1, (1, 1)) \equiv ((1, 1), 1) \equiv \dots
$$

ITC 要保持 **normal form**，让表示尽可能浅。

### id norm

$$
\text{norm}((0, 0)) = 0, \quad \text{norm}((1, 1)) = 1
$$

两个相同的子树就合并。这就是 buddy allocator 的 merge——左右兄弟都满了就并回父节点。

### event norm —— 提取公共 offset

定义 lift/sink 算子：
$$
\begin{aligned}
n \uparrow^m &= n + m \\
(n, e_1, e_2) \uparrow^m &= (n + m, e_1, e_2)
\end{aligned}
$$

$\uparrow^m$ 上标 $m$ 表示"给 base value 加 $m$"，整个函数向上平移 $m$。$\downarrow_m$ 下标 $m$ 同理向下平移。

event normalization：
$$
\text{norm}((n, e_1, e_2)) = (n + m, \; e_1 \downarrow_m, \; e_2 \downarrow_m)
$$
其中
$$
m = \min(\min(e_1), \min(e_2))
$$

**直觉**：左子树和右子树都有一个"公共底价 $m$"，把这个底价提到 root 的 base value 上，左右子树整体下沉 $m$。这保证了 normalized tree 至少有一个子树在某处取到 0（没有"白送的" 值了）。

类比 JPEG / Haar wavelet 中的 DC 系数提取——一个 8x8 block 如果所有像素都至少 128，先把 128 提为 DC，剩下的 AC detail 才稀疏。

参考：Haar wavelet 在 JPEG2000 中的应用 https://en.wikipedia.org/wiki/JPEG_2000

---

## 6. 三个操作的具体实现

### 6.1 Fork —— split id 树

$$
\text{fork}((i, e)) = ((i_1, e), (i_2, e)) \quad \text{where } (i_1, i_2) = \text{split}(i)
$$

split 递归定义：
$$
\begin{aligned}
\text{split}(0) &= (0, 0) \\
\text{split}(1) &= ((1, 0), (0, 1)) \\
\text{split}((0, i)) &= ((0, i_1), (0, i_2)) \quad \text{where } (i_1, i_2) = \text{split}(i) \\
\text{split}((i, 0)) &= ((i_1, 0), (i_2, 0)) \quad \text{where } (i_1, i_2) = \text{split}(i) \\
\text{split}((i_1, i_2)) &= ((i_1, 0), (0, i_2)) \quad \text{(两个子树都有时直接分)}
\end{aligned}
$$

**人话**：
- 空地皮分不出东西，给两边都空
- 整块 $[0,1)$ 分时，左孩子拿左半 $[0,1/2)$，右孩子拿右半 $[1/2,1)$
- 只有左半有地皮时，递归地分左半
- 只有一边有地皮时同理
- **两边都有地皮时最爽**——直接一个孩子拿一边，$O(1)$，不用递归

最后这个 case 是关键：在 fork 很多次之后，id 树一般都已经有两个孩子了，新 fork 几乎是常数时间。这就是 ITC 空间稳定的根源——**id 分配的 cost 摊在 fork 历史上，不在单次操作**。

### 6.2 Join —— sum id + max event

$$
\text{join}((i_1, e_1), (i_2, e_2)) = (\text{sum}(i_1, i_2), \text{join}(e_1, e_2))
$$

id 的 sum：
$$
\begin{aligned}
\text{sum}(0, i) &= i \\
\text{sum}(i, 0) &= i \\
\text{sum}((l_1, r_1), (l_2, r_2)) &= \text{norm}((\text{sum}(l_1, l_2), \text{sum}(r_1, r_2)))
\end{aligned}
$$

由于不变量保证两个 id 不相交，sum 不会产生值 2 的位置，结果仍是 characteristic function。如果两个子树 sum 后都变 1，norm 合并。

event 的 join（pointwise max）：
$$
\begin{aligned}
\text{join}(n_1, n_2) &= \max(n_1, n_2) \\
\text{join}(n_1, (n_2, l_2, r_2)) &= \text{join}((n_1, 0, 0), (n_2, l_2, r_2)) \\
\text{join}((n_1, l_1, r_1), (n_2, l_2, r_2)) &= \text{norm}((n_1, \text{join}(l_1, l_2 \uparrow^{n_2 - n_1}), \text{join}(r_1, r_2 \uparrow^{n_2 - n_1}))) \\
&\quad \text{if } n_1 \le n_2
\end{aligned}
$$

**关键技巧**：两个 tree 的 base value 不等时（$n_1 \le n_2$），第二个 tree 的子树要 lift $n_2 - n_1$。为什么？因为 $e_2$ 的子树存的是 **相对于 $n_2$ 的 detail**，要和 $e_1$ 的子树比较，得先统一坐标系——把 $e_2$ 的子树还原成相对于 $n_1$ 的值，差就是 $n_2 - n_1$。

最后 norm 一遍提取公因子，可能让整层合并回一个常数。

**人话**：join 两个账本时，如果两边都在某段地皮上记了值，取大值；如果两边有共同的"底价"，提到上层让树变浅。

### 6.3 Event —— fill + grow 的双策略

这是最精巧的部分。event 操作要在 $e$ 上加 $f \cdot i$（$f$ 任意正函数，$i$ 是我的 id），自由度很大。ITC 用这个自由度 **最小化 event tree 的增长**。

策略分两步：

$$
\text{event}(i, e) = \begin{cases}
(i, \text{fill}(i, e)) & \text{if fill 能简化树} \\
(i, e') & \text{否则用 grow 长一点}
\end{cases}
$$

#### Fill：试着"补齐"让树塌缩

fill 试图通过 inflate，让两个子树的 min 凑齐，从而 norm 能把整层合并。

几种典型情况：
- $\text{fill}(1, e) = \max(e)$：id 全覆盖时，把整个 $e$ 提到最大值，norm 成单常数。**树直接塌缩成一个整数**。
- $\text{fill}((1, i_r), (n, e_l, e_r))$：id 左半全覆盖、右半部分覆盖。先把左半 inflate 到 $\max(e_l)$，右半递归 fill，然后看能否 norm 合并。

**人话**：fill 像是"我有空闲地皮，能不能在上面加点东西让账本变整齐"。比如左半已经记到 5，右半只到 3，我就在右半 +2，让两边都是 5，然后这个 5 提到上层，整层塌缩。

#### Grow：必须增长时挑最便宜的位置

fill 失败就要真增长。grow 用 dynamic programming 选"最便宜"的增长位置：

$$
\begin{aligned}
\text{grow}(1, n) &= (n+1, 0) \\
\text{grow}(i, n) &= (e', c + N) \quad \text{where } (e', c) = \text{grow}(i, (n, 0, 0)) \\
\text{grow}((0, i_r), (n, e_l, e_r)) &= ((n, e_l, e_r'), c_r + 1) \\
\text{grow}((i_l, 0), (n, e_l, e_r)) &= ((n, e_l', e_r), c_l + 1) \\
\text{grow}((i_l, i_r), (n, e_l, e_r)) &= \text{选 } c_l \text{ 和 } c_r \text{ 中小的那边}
\end{aligned}
$$

变量含义：
- 返回 $(e', c)$：新 event tree + cost
- $c$：增长代价，越小越好
- $N$：一个大常数（大于最大可能树深），保证"increment 整数" 比"expand 整数为 tuple" 永远便宜

**优先级**：
1. 优先 increment 整数（用 $+N$ 让 expand 永远更贵）
2. 必须_expand_ 时优先浅层（每深一层 +1）
3. 同层任选

**人话**：账本上某个数 +1 是免费的（不动树结构），把一个数变成一对子树是昂贵的（树深一层）。所以 grow 极度偏好"只 +1"。

**关键 corollary**：当 $i = 1$（我独占整个 $[0,1)$），grow 永远走第一个 case，event tree 永远是单整数 $n$。这时 ITC 退化成 **Lamport clock**（单 participant 的 scalar clock）。

也就是说，ITC 把 Lamport clock、Vector clock、Version vector 都统一进来了：
- 1 个 participant：Lamport clock
- $n$ 个固定 participant：等价于 vector clock（每个 participant 占一个 dyadic interval，永远不 fork/join）
- dynamic participant：ITC 原生支持

---

## 7. 实验数据

论文 Figure 1 跑了 100 次、每次 25k~100k 迭代，统计平均 stamp size。我把数字整理成表：

| 场景 | 设置 | 128 participant 稳态 |
|---|---|---|
| Dynamic data causality | 每轮随机 fork + event + join（churn）| ~2900 bytes |
| Static process causality | 固定 participant，peek + join 传消息 | ~170 bytes |
| Classic VV + 128-bit UUID + 32-bit counter（mapping）| 对照 | 2560 bytes |
| Classic VV（紧凑下标）| 对照 | 512 bytes |

观察：

**Static 场景 ITC ~170 bytes** 比 VV 的 512 bytes 还小。因为消息是 anonymous stamp，id 不变；event tree 在 fill 优化下经常塌缩成单整数。这很反直觉——你以为连续表示会更大，实际上 Haar 表示 + normalization 比固定向量更紧凑。

**Dynamic 场景 ITC ~2900 bytes** 与 UUID-based VV 的 2560 bytes 相当。考虑到 ITC 不需要全局 id 分配、不需要 GC 协调、id 可复用，这个对比相当有利。Version Stamps [2] 在这种场景下增长失控，ITC 是第一个稳定的方案。

**增长曲线**：初始 fork 阶段增长，之后基本稳定（轻微 log）。这是 ITC 可实用的关键证据。

参考 Binary encoding 细节见论文 Appendix A https://haslab.uminho.pt/cbm/files/itc-paper.pdf

---

## 8. 一些重要联想

### 8.1 CRDT

ITC 的 fork-event-join 模型直接影响了 CRDT 理论。CRDT 的 state-based replication 也是 join semilattice 上的演算。Baquero 后续工作大量建立在 ITC 之上 https://link.springer.com/chapter/10.1007/978-3-642-25408-9_19

### 8.2 Dynamo 的 resurgence 问题

Dynamo [6] 在生产中因为 Version Vector 增长过快，激进裁剪老的 entry。但这导致一个 bug：被裁掉的 old update 偶尔会"复活"（resurgence），因为 GC 时不能保证所有 replica 都看到了。ITC 局部 retirement 不需要这种激进裁剪，从根本上避免了 resurgence。

参考 Dynamo 论文 https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf

### 8.3 Haar wavelet 与多分辨率分析

event tree 的 $(n, e_1, e_2) = n + e_1|_{\text{left}} + e_2|_{\text{right}}$ 就是 Haar wavelet 多分辨率分析。$n$ 是 scaling coefficient（低频），$e_1, e_2$ 是 wavelet coefficient（高频 detail）。Normalization 就是 DC 系数提取。这是 signal processing 的经典工具用在 distributed system 上。

参考 Haar wavelet https://en.wikipedia.org/wiki/Haar_wavelet

### 8.4 Buddy memory allocator

id tree 的 split/merge 和 buddy allocator 的 block split/coalesce 完全同构。Buddy allocator 用二分管理内存块，ITC 用二分管理 identity 区间。这是把 OS 的内存管理技术迁移到 distributed system 的 identity 管理上。

参考 https://en.wikipedia.org/wiki/Buddy_memory_allocation

### 8.5 Twitter Snowflake / UUID 的对比

Snowflake 用机器 id + 时间戳 + sequence；UUID v1 用 MAC + 时间 + 随机。都是"全局唯一不可回收"。ITC 提供"可回收的 id"——代价是参与者在 fork/join 时要维护 interval ownership。在需要 causal consistency 的场景下，ITC 把 id 分配和 causality tracking 统一了。

参考 Snowflake https://developer.twitter.com/en/docs/basics/developer-tools/overview/twitter-id

### 8.6 与 Roaring Bitmap 的类比

event tree 的"大部分叶子是 0 或小整数，少数 root 有大整数"分布，和 Roaring Bitmap 的统计假设类似。Roaring Bitmap 把 32-bit 整数分 high/low 两段，high 用 chunk 索引，low 用 bitmap 或 sorted array。ITC 的 binary encoding 类似——小整数用短编码，常见 case 用特殊 tag。

参考 Roaring Bitmap https://roaringbitmap.org/

### 8.7 与 Merkle Tree 的对比

Merkle Tree 也是二叉树 + 哈希，用来比对两个 replica 是否一致。ITC 用二叉树 + 计数器，比对两个 stamp 的因果关系。两者结构相似但目的不同：Merkle 是认证 + diff，ITC 是 ordering + merge。Git 内部用 Merkle tree 做 content addressing；如果 Git 要支持 fine-grained causal tracking 而不只是 commit DAG，可以考虑 ITC。

参考 Git 内部 https://git-scm.com/book/en/v2/Git-Internals-Git-Objects

### 8.8 与 Lamport logical clock 的统一

Lamport 1978 [11] 的 scalar clock 是 $n \to n+1$，单个 participant。ITC 在 $i = 1$ 时退化为它。Vector clock 是 $n$ 个 Lamport clock 的笛卡尔积。ITC 是"区间上连续的 Lamport clock"——每个 dyadic interval 上有一个独立 counter，但通过 Haar 表示共享了公共部分。

这种统一视角让我觉得 ITC 应该是 logical clock 设计空间里一个"natural point"，类似 SVM 之于 classification。

参考 Lamport 1978 https://lamport.org/pubs/time-clocks.pdf

### 8.9 后续工作

作者后续有改进 ITC event 操作的 paper https://haslab.uminho.pt/cbm/files/itc-improvements.pdf。Haskell 参考实现 https://github.com/VitorEnes/ITC。Baquero 在 CRDT 上的后续工作 https://haslab.uminho.pt/cbm

### 8.10 与 CRI 编程的联系

interval tree 这种 dyadic 区间结构在 competitive programming 里叫 CRI（Complete Range Interval），常用于 segment tree。ITC 本质上是一个 segment tree 上做 pointwise operation + lazy propagation 的分布式版本。

参考 segment tree https://cp-algorithms.com/data_structures/segment_tree.html

---

## 9. 一个完整例子串起来

从 seed stamp $(1, 0)$ 走一遍（论文 5.1 的 example）：

**Step 0**: 单个 participant，stamp $((1, 0))$
- id = $1$（独占 $[0,1)$）
- event = $0$（还没做事）

**Step 1**: fork 成两个
- split$(1) = ((1, 0), (0, 1))$
- 两个 stamp：$((1, 0), 0)$ 和 $((0, 1), 0)$
- 第一个占 $[0, 1/2)$，第二个占 $[1/2, 1)$

**Step 2**: 第一个 participant 做 1 次 event
- fill$(i, 0)$：id = $(1, 0)$ 左半全覆盖。fill $(1, 0)$ 在左半 $= \max(0) = 0$，没东西可 fill
- 改用 grow$(1, 0) = (0+1, 0) = (1, 0)$，cost 0
- 新 stamp：$((1, 0), 1)$
- event tree 就是常数 1，左半为 1 右半为 0

等等，这里 id = $(1, 0)$，不是 $1$。让我重新看 grow。grow$(1, n)$ 是 id 全为 1 时的 case。这里 id 是 $(1, 0)$，应该用 $\text{grow}((1, 0), (n, e_l, e_r))$ 的 case。

实际上 grow$(i, n)$ 当 $n$ 是常数时，会先 expand 成 $(n, 0, 0)$ 再处理。对 $((1, 0), 0)$：
- grow$((1, 0), (0, 0, 0))$：右半 id 是 0，左半 id 是 1
- 递归 grow$(1, 0)$ = $(1, 0)$，cost 0
- 返回 $((0, 1, 0), 0 + 1)$
- 但要 norm 一下，$m = \min(\min(1), \min(0)) = \min(1, 0) = 0$，norm 后还是 $(0, 1, 0)$

所以新 event tree 是 $(0, 1, 0)$：base 0，左半加 1，右半 0。函数在 $[0, 1/2)$ 上值为 1，$[1/2, 1)$ 上值为 0。

**Step 3**: 第一个 participant 再 fork
- split$((1, 0)) = ((1, 0), (0, 0)) = ((1, 0), 0)$
- 第一个孩子 $((1, 0), (0, 1, 0))$，第二个孩子 $(0, (0, 1, 0))$

第二个孩子是 anonymous stamp（id = 0），不能再 event，但它携带了"我父亲做过 1 次 event"的信息。如果作为 message 发出去，收信人能 join 进来。

**Step 4**: 三个 participant，两个 anonymous + 一个 original。Join 两个 anonymous 后再 fork...

这个例子继续下去会看到 join 时 norm 自动让树变浅，event 时 fill 让树塌缩。论文 5.1 的图示 https://haslab.uminho.pt/cbm/files/itc-paper.pdf 非常直观，建议对着图看。

---

## 10. 一句话总结

ITC 把 participant identity 从离散全局 id 替换为 $[0,1)$ 上的 dyadic interval（用二叉树表示），把 counter increment 替换为在 owned interval 上的 function inflation。配合 Haar wavelet 编码和 normalization，实现了一个无全局协调、id 可局部创建回收复用、空间自适应稳定的 logical clock。在 fork-event-join 代数框架下，它统一推广了 Lamport clock、Vector clock、Version vector 三种经典机制。

**核心 idea 用三个词概括**：Continuous identity + Haar representation + Localized retirement。

如果你要从这篇论文带走一句话，我推荐：

> "在 dynamic distributed system 里，id 管理和 causality tracking 应该是同一个机制，而不是两个独立的问题。"

这可能是 ITC 最深的设计哲学。Snowflake 之类的方案把 id 分配和 causality 分开，结果两者都难。ITC 把它们合并，结果两个问题一起解了。

参考后续阅读：
- 改进版 ITC: https://haslab.uminho.pt/cbm/files/itc-improvements.pdf
- CRDT 综述: https://link.springer.com/chapter/10.1007/978-3-642-25408-9_19
- Lamport 原始论文: https://lamport.org/pubs/time-clocks.pdf
- Haskell 实现: https://github.com/VitorEnes/ITC

---

# Interval Tree Clocks (ITC) 深度讲解

Andrej，这篇 2008 年 OPODIS 的论文我读完后觉得非常优雅。它的核心贡献是把 logical clock 从"离散全局 id + 整数 counter"的范式，迁到"连续区间 + 多分辨率函数"的范式。我尽量给你 build 出完整的 intuition。

---

## 1. 论文要解决的根本张力

经典 Vector Clocks [Fidge 1989, Mattern 1989] 和 Version Vectors [Parker 1983] 都假设：

- 参与者集合 **预先已知且固定**
- 每个 participant 拥有一个 **全局唯一 id**（通常是数组下标，或 MAC address / UUID）
- 每个 participant 的 counter 永远只能自己 +1

在 dynamic systems（P2P churn、mobile、elastic cloud）下这套假设崩塌，问题集中在 id 的 **生命周期管理**：

| 既有方案 | 缺陷 |
|---|---|
| Fidge [8] | 引入新 id 但从不 GC |
| Golden [9], Ratner [19] | 需要 GC，但 GC 需要 **所有** active entity 同意，一个 unreachable entity 就 stale |
| Bayou [18] | 抛弃 global id 假设但 id 仍不复用 |
| Landes Tree Clocks [12] | 只能 retire 直系祖先，限制太死 |
| Version Stamps [2]（作者自己 2002 的工作）| 理论上 localized，但实际结构增长失控 [3] |

ITC 同时满足四个性质：**无全局 id / 局部创建 / 局部回收 / id 可复用 / 增长可控**。这是第一次同时满足这些性质的实用机制。

参考：
- 原论文 PDF: https://haslab.uminho.pt/cbm/files/itc-paper.pdf
- Lamport 1978 原始论文: https://lamport.org/pubs/time-clocks.pdf
- Vector Clocks 综述: https://www.cs.rice.edu/~gw4314/teaching/comp520/papers/VectorClocks.pdf

---

## 2. Fork-Event-Join Model —— 把 causality 算子化

作者抽出三个 **core operations**，所有经典操作（send / receive / sync / spawn / terminate）都是这三者的复合。这个 abstraction 本身就是贡献，类似 Baez 把 distributed system 看成对称 monoidal category 的味道。

stamp 是一个 pair $(i, e)$：
- $i$：identity component（"我是谁"）
- $e$：event component（"我看到了哪些 events"）

### 三个核心算子

**fork** —— 克隆 causal past，分裂 identity：
$$
\text{fork}((i, e)) = ((i_1, e), (i_2, e)) \quad \text{s.t.} \quad i_1 \cdot i_2 = \mathbf{0}, \; i_1 + i_2 = i
$$
这里 $i_1, i_2$ 是新的 identity，对应 id 函数的 pointwise product 必须为 0（**不相交**），pointwise sum 必须等于原 id（**守恒**）。这就是一个资源分割。

**peek** —— fork 的退化版本，产生一个 anonymous stamp $(\mathbf{0}, e)$：
$$
\text{peek}((i, e)) = ((\mathbf{0}, e), (i, e))
$$
Anonymous stamp 不能再 event，只能携带 causal 信息。**Message 就是 anonymous stamp**。

**event** —— 在 e 上做严格 advance：
$$
\text{event}((i, e)) = (i, e') \quad \text{s.t.} \quad e < e', \; \forall x. \; e' \not\le x \land (x < e' \Rightarrow x \le e)
$$
即 $e'$ 恰好 cover $e$，不多不少。在 vector clocks 中表现为 $e'[i] := e[i] + 1$，其余不变。

**join** —— 合并两个 stamp，必须形成 join semilattice：
$$
\text{join}((i_1, e_1), (i_2, e_2)) = (i_3, e_1 \sqcup e_2)
$$
$e_1 \sqcup e_2$ 是 order-theoretic join（set union in causal histories，pointwise max in version vectors）。

### 经典操作 = 复合

- **send** = event ∘ peek（先 +1 自己的 counter，再把副本放进 message）
- **receive** = join ∘ event（先取 max，再 +1 自己）
- **sync** = join ∘ fork（双向同步：先 join 再 fork 出两个新副本）

整个 distributed computation 的语义被压缩成三个算子在一个 join semilattice 上的演算。这种 **algebraic reframing** 非常漂亮，CRDT 后来吸收了这个视角。参考：CRDT 综述 https://hal.inria.fr/inria-00629353/document

---

## 3. Function Space Framework —— 把 stamp 看成函数

这是从"具体机制"上升到"框架"的关键一步。

设 $i, e$ 都是从某个 domain $D$ 到 $\mathbb{N}$ 的函数：
- $i: D \to \{0, 1\}$ 是 **characteristic function**，标记 domain 中哪些元素可被这个 participant 用来 inflate
- $e: D \to \mathbb{N}$ 是 event function

定义 pointwise 运算：
$$
(f + g)(x) := f(x) + g(x), \quad (f \cdot g)(x) := f(x) \cdot g(x), \quad (f \sqcup g)(x) := f(x) \sqcup g(x)
$$
$\mathbf{0} := \lambda x.\, 0$ 是零函数。

### 关键不变量

最弱不变量（保证 causality 正确性）：
$$
\forall i. \; (i \cdot \bigcup_{i' \ne i} i') \ne i
$$
即每个 participant 必须独占至少一个 domain 元素，别人不能动这个元素。

实际采用更强的不变量（局部可维护）：
$$
\forall i_1 \ne i_2. \; i_1 \cdot i_2 = \mathbf{0}
$$
所有 participant 的 id 函数两两不相交（graphs 不重叠）。

### 在此框架下重定义算子

**Comparison**：
$$
(i_1, e_1) \le (i_2, e_2) \; := \; e_1 \le e_2
$$

**Join**：
$$
\text{join}((i_1, e_1), (i_2, e_2)) := (i_1 + i_2, \; e_1 \sqcup e_2)
$$
注意 id 是 **sum**（合并可用资源），event 是 **join**（取 causal max）。

**Fork**：任意满足 $i_1 + i_2 = i$ 且 $i_1 \cdot i_2 = \mathbf{0}$ 的分裂。

**Event**：
$$
\text{event}((i, e)) = (i, \; e + f \cdot i) \quad \text{for any } f \text{ s.t. } f \cdot i \ne \mathbf{0}
$$
这是关键的 **自由度**：可以选 $f$ 让 event tree 增长最少。经典 vector clock 强制 $f = \mathbf{1}$（在自己的 id slot 上 +1），而 ITC 可以在 id 覆盖的任意子区间上 inflate，从而选择"最便宜"的位置。

**Intuition**：这就像是把 "increment counter $i$" 推广为 "在 id 函数的支持集上做任意 non-negative 加法"。自由度用来 **minimize representation cost**。

---

## 4. ITC 的具体实例化 —— Interval Tree

### 关键迁移：离散 domain → 连续 domain $\mathbb{R}$

经典机制的 domain 是 $\{1, 2, \dots, n\}$（参与者编号），固定大小。ITC 的 domain 是 **连续区间 $[0, 1)$**，可以无限二分。

定义 unit pulse 函数：
$$
\mathbf{1}(x) := \begin{cases} 1 & x \in [0, 1) \\ 0 & \text{otherwise} \end{cases}
$$

### id tree 的语法和语义

语法：
$$
i ::= 0 \mid 1 \mid (i_1, i_2)
$$

语义函数 $\llbracket \cdot \rrbracket$ 把 tree 解释成 $[0,1) \to \{0,1\}$ 的函数：
$$
\begin{aligned}
\llbracket 0 \rrbracket &= \mathbf{0} \\
\llbracket 1 \rrbracket &= \mathbf{1} \\
\llbracket (i_1, i_2) \rrbracket(x) &= \llbracket i_1 \rrbracket(2x) + \llbracket i_2 \rrbracket(2x - 1)
\end{aligned}
$$

变量含义：
- $x \in [0, 1)$ 是连续 domain 中的位置
- $\llbracket i_1 \rrbracket(2x)$：把左子树的函数"压"到 $[0, 1/2)$ 上（输入 $x$ 映射到 $2x$）
- $\llbracket i_2 \rrbracket(2x-1)$：把右子树的函数"压"到 $[1/2, 1)$ 上

**这是 dyadic interval 上的 Haar-like 分解**。每个叶子对应一个 dyadic interval $[k/2^d, (k+1)/2^d)$，叶子为 1 表示这个 interval 属于该 participant。

例如 $(1, (0, 1))$ 表示：
- 左半 $[0, 1/2)$ 全取
- 右半 $[1/2, 1)$ 中再取右半，即 $[3/4, 1)$

合起来就是 $[0, 1/2) \cup [3/4, 1)$。

### event tree 的语法和语义

语法：
$$
e ::= n \mid (n, e_1, e_2)
$$
其中 $n \in \mathbb{N}_{\ge 0}$。

语义：
$$
\begin{aligned}
\llbracket n \rrbracket &= n \cdot \mathbf{1} \quad \text{（常数函数，整个 } [0,1) \text{ 上值为 } n\text{）} \\
\llbracket (n, e_1, e_2) \rrbracket(x) &= n \cdot \mathbf{1}(x) + \llbracket e_1 \rrbracket(2x) + \llbracket e_2 \rrbracket(2x - 1)
\end{aligned}
$$

**这是 Haar wavelet 的多分辨率表示**！
- $n$ 是当前 scale 上的 coarse value（base / DC 分量）
- $e_1, e_2$ 是下一级 scale 上的 detail（sub-band）

**核心直觉**：每个 participant 看到的"事件计数"被表达成一个 piecewise constant function on $[0,1)$。Haar 表示让"两半的公共 offset" 可以提到上层 base value，从而让表示 **自适应压缩**。如果某个 participant 的 event 在整个 $[0,1)$ 上都涨了 1，那只需要 root $n + 1$，不用动子树。

### 例子

事件树 $(1, 2, (0, (1, 0, 2), 0))$ 的解读（自顶向下）：
- root base $n = 1$：整个区间有 base 值 1
- 左子 $= 2$：左半 $[0, 1/2)$ 在 base 上额外加 2，所以左半值为 $1+2 = 3$
- 右子 $= (0, (1, 0, 2), 0)$：
  - 右半 base +0
  - 右半的左四分之一 $[1/2, 3/4)$：再 +1
  - 右半的右四分之一 $[3/4, 1)$：再 +(0, 0, 2) = 在最右八分之一 +2

最终函数在 $[0, 1/2)$ 上为 3，$[1/2, 3/4)$ 上为 2，$[3/4, 7/8)$ 上为 1，$[7/8, 1)$ 上为 3。

---

## 5. Normal Form —— 让表示保持紧凑

### id norm

$$
\text{norm}((0, 0)) = 0, \quad \text{norm}((1, 1)) = 1, \quad \text{norm}(i) = i
$$
即两个相同子树就合并。$1 \equiv (1, 1) \equiv (1, (1, 1)) \equiv \cdots$ 都表示整区间覆盖。

### event norm —— 关键的"提取公因子"

定义 lift / sink 算子（给 base value 加减常数）：
$$
\begin{aligned}
n \uparrow^m &= n + m \\
(n, e_1, e_2) \uparrow^m &= (n + m, e_1, e_2) \\
n \downarrow_m &= n - m \\
(n, e_1, e_2) \downarrow_m &= (n - m, e_1, e_2)
\end{aligned}
$$
符号解释：
- $\uparrow^m$：base value 加 $m$（向上平移整个函数 $m$）
- $\downarrow_m$：base value 减 $m$（向下平移）
- 上标 vs 下标只是符号习惯，无本质区别

event normalization：
$$
\text{norm}((n, e_1, e_2)) = (n + m, \; e_1 \downarrow_m, \; e_2 \downarrow_m)
$$
其中
$$
m = \min(\min(e_1), \min(e_2))
$$

$\min(e)$ 是函数 $\llbracket e \rrbracket$ 在 $[0,1)$ 上的最小值，递归定义：
$$
\begin{aligned}
\min(n) &= n \\
\min((n, e_1, e_2)) &= n + \min(\min(e_1), \min(e_2))
\end{aligned}
$$
对 normalized tree，由于"提取"已经做完，至少一个子树 min 为 0，简化为 $\min((n, e_1, e_2)) = n$。

**直觉**：当左子树和右子树都"至少有 $m$" 时，把这 $m$ 提取到 root 的 base value 上，子树值整体下降 $m$。这保证了 normalized tree 至少有一个子树在某处取到 0。这就是 Haar 表示的"DC 分量提取"。

类比：在图像压缩中，如果一个 block 的所有像素都至少是 128，先把 128 提出来作为 DC 系数，剩下的 detail 才能稀疏表示。

---

## 6. Operations 详解

### 6.1 Comparison (leq)

比较两个 stamp 等价于比较 event 函数 pointwise $\llbracket e_1 \rrbracket \le \llbracket e_2 \rrbracket$。递归定义：

$$
\begin{aligned}
\text{leq}(n_1, n_2) &= n_1 \le n_2 \\
\text{leq}(n_1, (n_2, l_2, r_2)) &= n_1 \le n_2 \\
\text{leq}((n_1, l_1, r_1), n_2) &= n_1 \le n_2 \land \text{leq}(l_1 \uparrow^{n_1}, n_2) \land \text{leq}(r_1 \uparrow^{n_1}, n_2) \\
\text{leq}((n_1, l_1, r_1), (n_2, l_2, r_2)) &= n_1 \le n_2 \land \text{leq}(l_1 \uparrow^{n_1}, l_2 \uparrow^{n_2}) \land \text{leq}(r_1 \uparrow^{n_1}, r_2 \uparrow^{n_2})
\end{aligned}
$$

**关键技巧**：比较子树前要先 lift base value。$l_1 \uparrow^{n_1}$ 表示"把 root 的 base 加到子树上去还原真实值"，因为 $\llbracket (n, e_1, e_2) \rrbracket = n + e_1$ on left half，子树本身存的只是相对值。

第二种 case（左边是 tree，右边是常数）需要把 tree 的两个子树都 lift 起来和常数比，等价于检查整个左半、右半都 $\le n_2$。

### 6.2 Fork (split)

$$
\text{fork}((i, e)) = ((i_1, e), (i_2, e)) \quad \text{where} \; (i_1, i_2) = \text{split}(i)
$$

split 的递归定义：
$$
\begin{aligned}
\text{split}(0) &= (0, 0) \\
\text{split}(1) &= ((1, 0), (0, 1)) \quad \text{— unit pulse 拆成左右两半} \\
\text{split}((0, i)) &= ((0, i_1), (0, i_2)) \quad \text{where} \; (i_1, i_2) = \text{split}(i) \\
\text{split}((i, 0)) &= ((i_1, 0), (i_2, 0)) \quad \text{where} \; (i_1, i_2) = \text{split}(i) \\
\text{split}((i_1, i_2)) &= ((i_1, 0), (0, i_2)) \quad \text{— 已有两棵子树直接分配}
\end{aligned}
$$

**直觉**：
- $1$ 是"完整区间"，二分为左半 $1$ 和右半 $1$，分别放进新 id 的左右
- 已经是 $(i_1, i_2)$ 形式时，分给两个 fork 一个最简单：左 fork 拿左子树，右 fork 拿右子树，无需深入
- 只有一边时，递归地拆那一边

**重要性质**：当 id 已经是 balanced 二叉形（fork 多次后），新的 fork 几乎是 $O(1)$ 操作（直接分两个子树）。这是 ITC 在 dynamic 场景下空间稳定的根源。

### 6.3 Join (sum + max)

$$
\text{join}((i_1, e_1), (i_2, e_2)) = (\text{sum}(i_1, i_2), \text{join}(e_1, e_2))
$$

id 的 sum（保持 normalized）：
$$
\begin{aligned}
\text{sum}(0, i) &= i \\
\text{sum}(i, 0) &= i \\
\text{sum}((l_1, r_1), (l_2, r_2)) &= \text{norm}((\text{sum}(l_1, l_2), \text{sum}(r_1, r_2)))
\end{aligned}
$$
两个 id 函数 pointwise 相加，因为不变量保证它们不相交，所以 $i_1 + i_2$ 不会产生 2，结果仍是 characteristic function。两个 $(1, 1)$ 子树会 norm 成 $1$。

event 的 join（pointwise max）：
$$
\begin{aligned}
\text{join}(n_1, n_2) &= \max(n_1, n_2) \\
\text{join}(n_1, (n_2, l_2, r_2)) &= \text{join}((n_1, 0, 0), (n_2, l_2, r_2)) \\
\text{join}((n_1, l_1, r_1), (n_2, l_2, r_2)) &= \text{norm}((n_1, \text{join}(l_1, l_2 \uparrow^{n_2 - n_1}), \text{join}(r_1, r_2 \uparrow^{n_2 - n_1}))) \quad \text{if } n_1 \le n_2
\end{aligned}
$$

**关键**：当两个 tree 的 base value 不等（$n_1 \le n_2$），先把 $e_2$ 的子树 lift $n_2 - n_1$，再和 $e_1$ 的子树 join。这是因为 $e_2$ 子树存的是相对于 $n_2$ 的值，要换成相对于 $n_1$ 的值，差值就是 $n_2 - n_1$。最后 norm 提取公因子。

这个 case analysis 用 $n_1 \le n_2$ 简化了对称情况。

### 6.4 Event (fill + grow) —— 最有意思的部分

event 操作有最大自由度：可以 inflate 任意 id 覆盖的子区间。ITC 用这个自由度来 **最小化 event tree 的增长**。

策略分两步：

$$
\text{event}(i, e) = \begin{cases}
(i, \text{fill}(i, e)) & \text{if } \text{fill}(i, e) \ne e \\
(i, e') & \text{otherwise, where } (e', c) = \text{grow}(i, e)
\end{cases}
$$

#### Fill：尽量简化

fill 尝试通过 inflate 让树能被 normalize 压缩。例如：
- 在 id 覆盖的子区间上 inflate，让两个子树最小值匹配，从而能提取公因子到上层
- 在 id 全覆盖时 $\text{fill}(1, e) = \max(e)$：把整个 event function 提升到最大值，整个区间一致，norm 成一个常数

fill 的递归定义：
$$
\begin{aligned}
\text{fill}(0, e) &= e \quad \text{— id 空集，啥也做不了} \\
\text{fill}(1, e) &= \max(e) \quad \text{— id 全集，把所有值提升到 max，norm 成单常数} \\
\text{fill}(i, n) &= n \quad \text{— event 已经是常数，不需要再 fill} \\
\text{fill}((1, i_r), (n, e_l, e_r)) &= \text{norm}((n, \max(\max(e_l), \min(e_r')), e_r')) \\
&\quad \text{where } e_r' = \text{fill}(i_r, e_r) \\
\text{fill}((i_l, 1), (n, e_l, e_r)) &= \text{norm}((n, e_l', \max(\max(e_r), \min(e_l')))) \\
&\quad \text{where } e_l' = \text{fill}(i_l, e_l) \\
\text{fill}((i_l, i_r), (n, e_l, e_r)) &= \text{norm}((n, \text{fill}(i_l, e_l), \text{fill}(i_r, e_r)))
\end{aligned}
$$

关键 case 解释：$(1, i_r)$ 表示 id 在左半全覆盖、右半部分覆盖。
- 把左半 inflate 到其 $\max(e_l)$（fill 全覆盖时简化为常数）
- 右半递归 fill
- 然后看左半的 max 和右半的 min 是否能凑齐：把左半整体拉到 $\max(\max(e_l), \min(e_r'))$，再 norm 一次

如果右半最小值能往上凑到左半最大值，整层就能被 norm 提取，从而降低树高。

#### Grow：必须增长时的最优决策

如果 fill 没法简化树，必须新增 inflation。Grow 用 **dynamic programming** 选择"最便宜"的位置：

$$
\begin{aligned}
\text{grow}(1, n) &= (n+1, 0) \\
\text{grow}(i, n) &= (e', c + N) \quad \text{where } (e', c) = \text{grow}(i, (n, 0, 0)) \\
\text{grow}((0, i_r), (n, e_l, e_r)) &= ((n, e_l, e_r'), c_r + 1) \quad \text{where } (e_r', c_r) = \text{grow}(i_r, e_r) \\
\text{grow}((i_l, 0), (n, e_l, e_r)) &= ((n, e_l', e_r), c_l + 1) \quad \text{where } (e_l', c_l) = \text{grow}(i_l, e_l) \\
\text{grow}((i_l, i_r), (n, e_l, e_r)) &= \begin{cases}
((n, e_l', e_r), c_l + 1) & \text{if } c_l < c_r \\
((n, e_l, e_r'), c_r + 1) & \text{if } c_l \ge r_r
\end{cases} \\
&\quad \text{where } (e_l', c_l) = \text{grow}(i_l, e_l), \; (e_r', c_r) = \text{grow}(i_r, e_r)
\end{aligned}
$$

变量含义：
- 返回值 $(e', c)$：新 event tree 和 cost
- $c$ 是增长代价（越小越好）
- $N$ 是个大常数（大于可能的最大树深），用来让 "把整数 expand 成 tuple" 的代价压倒一切 depth 累加

**优化目标（按优先级降序）**：
1. **优先 increment 整数** 而不是把整数 expand 成 tuple（用 $+N$ 实现）
2. 在必须 expand 时，**优先在浅层 expand**（每深一层 cost +1）
3. 在同层中，**任选一边**（按 $c_l < c_r$ 比较）

**Intuition**：整数 increment 是 $O(1)$ 操作且不增加树大小；而 expand 整数到 tuple 会让树多一层。所以 grow 极度偏好"只 increment 一个数"。这就是为什么 ITC 在 event 多次后很多分支仍是一个简单的整数。

例如，当 id = 1 时（参与者拥有整个区间），event 永远只是 $n \to n+1$，树就是个单整数——退化成 scalar clock。这同时统一了 Lamport clock（单 participant）、Vector Clock（固定 participant）、Version Vector 这些经典机制。

---

## 7. 实验：空间消耗

论文 Figure 1 给出了两种场景下 stamp size 的演化（binary encoding，见 Appendix A）：

| 场景 | 设置 | 128 个 participant 稳态 size |
|---|---|---|
| Dynamic data causality | 每轮随机 fork + event + join，churn | ~2900 bytes |
| Static process causality | 固定 participant，只 peek + join（消息传递）| ~170 bytes |
| 对照：传统 vector + 128-bit UUID + 32-bit counter（128 个 replica）| mapping 形式 | 2560 bytes |
| 对照：vector（紧凑下标）| vector 形式 | 512 bytes |

观察：
- **Static 场景下 ITC ~170 bytes** 远小于 vector 的 512 bytes。因为消息是 anonymous stamp，id 不变，event tree 在 fill 优化下经常塌缩成单整数。
- **Dynamic 场景下 ITC ~2900 bytes** 与 UUID-based vector 的 2560 bytes 相当。考虑到 ITC 不需要全局 id 分配、不需要 GC 协调、id 可复用，这是非常合理甚至惊人的结果。
- 增长曲线在初始迭代后基本稳定（带轻微 logarithmic 增长）。这是关键：与作者之前 Version Stamps 的失控增长形成对比。

### Binary Encoding 细节

Appendix A 的编码利用了 event tree 的统计特性（"root 附近少数大整数 + 叶子附近多数小整数"）：
- 整数用 **变长编码** `enc_n`：从 2 bits 起步，每不够 2 倍就再加 1 bit
- id tree 用 2-bit tag 区分 4 种 case：0, 1, 只有左子树, 只有右子树, 完整二叉
- event tree 用 1-bit 区分 "整数 vs tuple"，tuple 再用 2-bit tag 区分 base=0 或 base!=0、左空右空等常见 case

这是非常 Erlang 风格的二进制模式匹配（论文里直接用了 Erlang bit syntax）。参考 Erlang bit syntax: https://www.erlang.org/doc/programming_examples/bit_syntax

---

## 8. 与其他系统的关联

### 8.1 与 CRDT 的关系

ITC 的 fork-event-join 模型直接启发了后续 CRDT 的设计。CRDT 中的 join semilattice、state-based replication 都能看到这个框架的影子。Baquero 后续工作直接建立在 ITC 之上。

参考：https://link.springer.com/chapter/10.1007/978-3-642-25408-9_19

### 8.2 与 Dynamo 的对比

Dynamo 的 Version Vector 在生产中因为 VV 增长太快被 garbage collect 老的 entry，但这会导致 **老 update 复活**（resurgence）的 bug。ITC 因为局部 retirement 不需要这种激进裁剪。

参考 Dynamo 论文: https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf

### 8.3 Haar Wavelet 的类比

event tree 的递归结构 $(n, e_1, e_2) = n + e_1 \text{ on left} + e_2 \text{ on right}$ 就是 Haar wavelet 的多分辨率分析。base value $n$ 是 scaling coefficient（low frequency），子树是 wavelet coefficients（high frequency detail）。Normalization 等价于"如果 detail 的公共部分能提到 scaling coefficient，就提"。

参考 Haar wavelet: https://en.wikipedia.org/wiki/Haar_wavelet

### 8.4 与 Buddy Memory Allocation 的类比

id tree 的 split 操作和 buddy allocator 的 block splitting 一模一样：从 $[0, 1)$ 开始，需要就二分，需要更多就再二分。Join 时如果兄弟正好互补（norm 出父节点），就合并——这是 buddy allocator 的 merge。

参考 buddy allocator: https://en.wikipedia.org/wiki/Buddy_memory_allocation

### 8.5 与分布式 ID 分配（Snowflake, UUID）的对比

Twitter Snowflake 用机器 id + 时间戳；UUID 用 MAC + 时间 + 随机；都是 **全局唯一但不可回收** 的方案。ITC 提供了一种可回收的替代，代价是需要参与者在 fork/join 时维护 interval ownership。在 causal consistency 必须的场景下，ITC 把 id 分配和 causality tracking **统一** 了起来。

---

## 9. 我对这篇论文的评价

### 优点
1. **抽象层次恰当**：Fork-Event-Join model 把所有经典 clock 操作统一成三个算子，function space framework 给出充要条件，ITC 是一个具体 instance。这种 "framework + instance" 的写法非常有美感。
2. **核心技术发明极简洁**：连续 domain + Haar-like tree，两个 idea 加起来解决一切。Haar 表示让 normalization 自动压缩；id 用 interval 让 fork/join 自然守恒。
3. **实验诚实**：直接和传统 vector clock 比 byte 数，dynamic 场景承认和 UUID-vector 相当，static 场景展示出优势。

### 可以挑剔的地方
1. **event 操作的 fill / grow 实现非常复杂**，递归定义有 5+ case，工程实现容易出 bug。后续作者有论文改进 event operation。
2. **没有形式化 worst-case bound**：空间复杂度只有 simulation 数据，没有 $O(\log n)$ 之类的理论保证。
3. **未讨论并发 fork-event 的 race**：实现层面需要 mutex，但论文未涉及。
4. **没有大规模真实系统评测**，只有 simulation。后续 Microsoft 的 practices（如 CosmosDB）采用更简单的方案。

### 后续工作
- ITC 的 Haskell 参考实现: https://github.com/VitorEnes/ITC
- 改进版（更简单的 event）: https://haslab.uminho.pt/cbm/files/itc-improvements.pdf
- Baquero 在 CRDT 上的后续工作: https://haslab.uminho.pt/cbm

---

## 10. 一句话总结

ITC 把"participant identity" 从离散全局 id 替换为连续区间 $[0,1)$ 的 dyadic subintervals，把"counter increment" 替换为在 owned interval 上的 event function inflation。配合 Haar 多分辨率编码和 normalization，实现了一个 **无全局协调、id 可局部创建回收、空间自适应** 的 logical clock 机制。在 fork-event-join 的代数框架下，它统一并推广了 Lamport Clock、Vector Clock、Version Vector 三种经典机制，是 causality tracking 在 dynamic system 上的优雅解法。

---

如果你想深入，我建议下一步读：
1. Baquero 后续的 ITC 改进 paper（简化 event op）
2. CRDT 的 join semilattice 理论
3. Haar wavelet 的多分辨率分析理论
4. Microsoft Research Cambridge 的实践报告 https://www.microsoft.com/en-us/research/wp-content/uploads/2013/09/Lamport-Turing-Lecture.pdf
