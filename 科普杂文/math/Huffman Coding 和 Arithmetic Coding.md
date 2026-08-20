### 1. Huffman Coding：最优的前缀码

Huffman Coding 是一种基于 **Greedy Algorithm** 的无损压缩算法。它的核心直觉是：**出现频率高的 Symbol 使用短的 Code，出现频率低的 Symbol 使用长的 Code**。

#### 1.1 核心原理与直觉

Huffman Coding 本质上是在构建一棵 **Binary Tree**。它满足 **Prefix Property**（前缀性质），即没有任何一个 Code 是另一个 Code 的前缀。这允许我们在接收到 Bit stream 时能够即时解码，不需要分隔符。

**直觉构建:**
想象你在搭积木。
1.  找出当前频率最低的两个节点。
2.  把它们合并成一个新节点，新节点的频率是两者之和。
3.  重复上述过程，直到只剩下一个节点。

#### 1.2 算法步骤详解

假设我们有一个 Symbol 集合 $S = \{s_1, s_2, ..., s_n\}$，对应的概率为 $P = \{p_1, p_2, ..., p_n\}$。

1.  **初始化**: 将所有 Symbol 视为独立的 Leaf Node，构成一个 Forest。
2.  **循环**:
    *   从 Forest 中选取权重最小 的两个节点 $n_i$ 和 $n_j$。
    *   创建一个新的 Parent Node $n_k$，其权重 $w_k = w_i + w_j$。
    *   将 $n_i$ 设为 $n_k$ 的左孩子（分配 bit '0'），$n_j$ 设为右孩子（分配 bit '1'）。
    *   从 Forest 中移除 $n_i$ 和 $n_j$，加入 $n_k$。
3.  **终止**: 当 Forest 中只剩下一棵树时，构建完成。从 Root 到 Leaf 的路径即为该 Symbol 的 Code Word。

#### 1.3 数学公式与性能分析

**Entropy (香农熵):**
数据压缩的理论下限由 Shannon Entropy 定义：
$$H(X) = -\sum_{i=1}^{n} p_i \log_2 p_i$$
其中：
*   $X$: 随机变量，代表 Source Symbol。
*   $p_i$: 第 $i$ 个 Symbol 出现的概率。
*   $H(X)$: 平均每个 Symbol 所需的最小比特数。

**Huffman Coding 的平均码长:**
$$L_{avg} = \sum_{i=1}^{n} p_i \cdot l_i$$
其中：
*   $l_i$: 第 $i$ 个 Symbol 的码字长度。

**界限:**
Huffman Coding 是最优的前缀码，但它受限于 Integer Bits。
$$H(X) \le L_{avg} < H(X) + 1$$
这意味着 Huffman Coding 每个符号至少要花费 1 bit。当某个 Symbol 的概率极高（例如 $p = 0.99$）时，$H(X)$ 可能很小，但 Huffman 必须分配 1 bit，导致效率低下。

#### 1.4 技术变种

*   **Canonical Huffman Coding**: 不存储整棵树，而是存储每个 Code 的长度，解码时通过标准算法重建树结构。这大大减小了 Header 的大小。
*   **Adaptive Huffman Coding**: 允许在编码过程中动态更新频率树，适用于不知道先验概率或数据流变化的情况。

---

### 2. Arithmetic Coding：逼近熵极限的利器

Arithmetic Coding 解决了 Huffman Coding 必须使用整数比特的缺陷。它不再为每个 Symbol 分配一个单独的 Code Word，而是将整个 Message 编码为一个 **[0, 1) 之间的一个浮点数**。

#### 2.1 核心原理与直觉

**直觉构建:**
想象一根尺子，长度为 1。
1.  根据概率将尺子划分为若干段。
2.  读入第一个 Symbol，找到对应的段。
3.  将这个段“放大”到全长 1，重新按概率划分。
4.  读入下一个 Symbol，在新的细分段中继续寻找。
5.  最终，我们确定了一个极小的区间，只需输出该区间内的任意一个二进制小数即可。

#### 2.2 算法步骤详解

假设符号集 $S$，概率分布 $P$，累计概率分布函数 $CDF$。

1.  **初始化**: 设当前区间 $[Low, High) = [0, 1)$。
2.  **循环**:
    *   计算当前区间的范围 $Range = High - Low$。
    *   对于输入 Symbol $s_i$，计算新的区间：
        *   $New\_High = Low + Range \times CDF(s_i + 1)$
        *   $New\_Low = Low + Range \times CDF(s_i)$
        *   注意：$CDF(s_i)$ 是所有小于 $s_i$ 的符号概率之和。
    *   更新 $High$ 和 $Low$。
3.  **输出**: 编码结束后，输出一个足以唯一标识最终区间 $[Low, High)$ 的二进制小数。通常取 $Low$ 的二进制表示。

#### 2.3 数学公式与实现细节

**区间更新公式:**
设 $R_n$ 为第 $n$ 步时的区间大小，$L_n$ 为下界。
$$L_{n} = L_{n-1} + R_{n-1} \times CDF(s_n)$$
$$R_{n} = R_{n-1} \times p(s_n)$$

**变量解释:**
*   $CDF(s_n)$: 累积分布函数值，即符号 $s_n$ 的下界累积概率。
*   $p(s_n)$: 符号 $s_n$ 的概率。

**精度问题:**
实际计算机中无法表示无限精度的小数。因此实现中使用了 **Integer Arithmetic** 和 **Renormalization**。
*   当 $Low$ 和 $High$ 最高位相同时，将其移出区间并输出（因为该位已确定），并将剩余部分左移扩大精度。
*   如果区间太小且高位未收敛，使用 **E3 Mapping**（处理 $High < 0.5$ 且 $Low > 0.25$ 的情况，防止区间塌缩）。

#### 2.4 性能优势

Arithmetic Coding 的码长可以非常接近 Shannon Entropy：
$$L \approx H(X)$$
它允许一个 Symbol 只占用分数个比特（例如 0.3 bits）。对于高概率符号（如 $p=0.99$），Arithmetic Coding 能极大地节省空间。

---

### 3. Hoffman vs Arithmetic：深度对比与技术联想

| 特性 | Huffman Coding | Arithmetic Coding |
| :--- | :--- | :--- |
| **Granularity (粒度)** | Integer bits per symbol | Fractional bits per symbol |
| **Efficiency (效率)** | $H \le L < H+1$ | Very close to $H$ (especially for skewed distributions) |
| **Complexity (复杂度)** | Low (Tree lookup) | High (Interval multiplication, Renormalization) |
| **Patent (专利)** | Patent-free (Expired) | Historically patented (Most expired now) |
| **Adaptation (自适应)** | Adaptive variants exist | Naturally adaptive (update CDF on the fly) |

#### 3.1 为什么 Arithmetic 更强？——从 Kraft Inequality 角度看

Huffman Coding 之所以不够完美，是因为它受限于 **Kraft Inequality** 对于整数解的限制。
Kraft Inequality 指出，对于即时码，码长 $l_i$ 必须满足：
$$\sum_{i=1}^{n} 2^{-l_i} \le 1$$
Huffman 强制 $l_i$ 为整数，导致无法精确匹配 $p_i$。而 Arithmetic Coding 实际上打破了这种离散的限制，实现了连续的区间映射。

#### 3.2 技术联想与现代应用

1.  **Asymmetric Numeral Systems (ANS)**:
    *   这是现代压缩算法（如 Facebook 的 **Zstandard (zstd)**, Apple 的 **LZFSE**）的核心。
    *   **直觉**: 结合了 Arithmetic Coding 的效率和 Huffman Coding 的速度。它使用一个有限状态机来模拟区间编码，但在实现上通过查表操作，大大提升了 SIMD 指令的利用效率。
    *   它是 Arithmetic Coding 的离散化近似。

2.  **Context-Based Adaptive Binary Arithmetic Coding (CABAC)**:
    *   这是 H.264/AVC 和 HEVC (H.265) 视频编码标准中的核心 Entropy Coding 工具。
    *   **原理**: 它不仅使用 Arithmetic Coding，还结合了 Context Model。根据上下文动态更新概率估计（使用 **State Transition Table** 而非浮点运算）。Symbol 必须先 Binarize（二值化）成 Bin string。

3.  **Range Coding**:
    *   Arithmetic Coding 的一种实现方式，旨在避开早期的专利限制。广泛应用于 **xz**, **lzma**, **zlib** 等开源压缩库中。

4.  **JPEG vs JPEG 2000**:
    *   标准 JPEG 使用 Huffman Coding。
    *   JPEG 2000 使用 Arithmetic Coding (MQ-coder)。这提供了更好的压缩率，尤其是低比特率下的图像质量，但因为计算复杂度和专利问题，推广受阻。

#### 3.3 实验数据模拟

假设源数据概率分布极度不均：
Symbol A: 95% ($p_A = 0.95$)
Symbol B: 5% ($p_B = 0.05$)

**Entropy 计算:**
$$H = -0.95 \log_2(0.95) - 0.05 \log_2(0.05) \approx 0.286 \text{ bits/symbol}$$

**Huffman 结果:**
必须分配至少 1 bit 给 A，1 bit 给 B（或者构建扩展块）。
如果是简单编码：
$L_{avg} = 1 \text{ bit/symbol}$.
效率 $\eta = \frac{0.286}{1} = 28.6\%$。**极度浪费。**

**Arithmetic 结果:**
理论上可以逼近 0.286 bits/symbol。
假设编码 "AAA..."，区间会迅速收敛到接近 0 的一侧。
输出可能只需很少的比特位。
效率接近 100%。

---

### 4. 总结与 Web Links

Huffman Coding 以其简单和高效著称，适合通用场景和硬件实现。Arithmetic Coding 则在需要极致压缩率、尤其是概率分布极其倾斜的场景下（如视频编码 CABAC）展现统治力。随着专利过期，Arithmetic Coding 的变种如 ANS 和 Range Coding 正在成为现代高性能压缩库的标准配置。

**Reference Links:**

*   **Huffman Coding - Wikipedia**: [https://en.wikipedia.org/wiki/Huffman_coding](https://en.wikipedia.org/wiki/Huffman_coding)
*   **Arithmetic Coding - Wikipedia**: [https://en.wikipedia.org/wiki/Arithmetic_coding](https://en.wikipedia.org/wiki/Arithmetic_coding)
*   **Kraft Inequality**: [https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality](https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality)
*   **CABAC (Context-adaptive binary arithmetic coding)**: [https://en.wikipedia.org/wiki/Context-adaptive_binary_arithmetic_coding](https://en.wikipedia.org/wiki/Context-adaptive_binary_arithmetic_coding)
*   **Asymmetric Numeral Systems (ANS)**: [https://en.wikipedia.org/wiki/Asymmetric_numeral_systems](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems)
*   **Zstandard Compression**: [https://facebook.github.io/zstd/](https://facebook.github.io/zstd/)
*   **A Clear Explanation of Arithmetic Coding**: [https://www.arturocampos.com/cp_arith.html](https://www.arturocampos.com/cp_arith.html)
*   **Charles Bloom's Blog on Compression (Highly Technical)**: [http://cbloomrants.blogspot.com/](http://cbloomrants.blogspot.com/)