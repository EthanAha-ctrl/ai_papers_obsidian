---
source_pdf: Float Self-Tagging.pdf
paper_sha256: a71a593c58ae771a7916535282c90c586debad1369d0eccce229091dddddfa83
processed_at: '2026-08-04T09:20:06-07:00'
target_folder: Python
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Float Self-Tagging

好，我把前面那堆技术细节翻译成人话。

---

## 一句话概括

**float 这玩意儿在 dynamic language 里天生难搞，因为 IEEE754 把 64 bit 全占满了，没地方塞 type 信息。这帮人想了个绝招：把 float 的 bit 旋转一下，让某些 float 自己"碰巧"在 low bit 上长出正确的 type tag，就不用 heap allocate 了。**

---

## 问题是什么？

想象你在管理一个仓库。每个货物要贴个 label 标明类型——"这是 int"、"这是 string"、"这是 float"。

正常机器 word 是 64 bit，你可以留 low 3 bit 当 label，剩下 61 bit 存数据。small int 这么搞爽爆了：61 bit 够大，加减法直接算完修一下 label 就行。

**float 倒霉在哪**：IEEE754 double 把 64 bit 每一位都规定了用途——1 bit sign、11 bit exponent、52 bit mantissa，一行指令`addsd` 要求输入必须是完整 64 bit IEEE754 格式。你没法偷 3 bit 出来当 label，偷了 CPU 的 float 指令就不认了。

所以传统做法是：float 去仓库（heap）里租个格子存着，word 里只放 pointer + label。每次用 float 都得跑去仓库取，还老得打扫仓库（GC 压力）。

---

## 现有的两个妥协

**NaN-boxing**：IEEE754 的 NaN 值其实有 51 bit 是"废"的（标准规定 NaN 的 mantissa 不为 0 即可，高位用来区分 quiet/signaling，剩下随便填）。所以把 non-float object 塞进 NaN 的废 bit 里，float 自己保持原样。

代价：每次看一个 object 是什么类型，要先检查"是不是 NaN-boxed"，pointer dereference 也要先 mask 出来。所有 non-float object 都受拖累。

**NuN-boxing**：给所有 float 加个 offset，把它们推到 bit pattern 的中间区间，把 low 和 high 区间腾出来给 non-float object。

代价：float 每次用之前要减 offset，用完要加 offset。

---

## Self-tagging 的 insight

这是 paper 最漂亮的地方。

**任意一个可逆 bit 变换（比如 rotation），作用在 float 上，恰好有 1/8 的 float 会"碰巧"在 low 3 bit 等于你想要的 tag 值。**

道理很简单：3 bit 有 8 种组合，可逆变换是双射，所以每种组合各分到 1/8 的 float。

那剩下的 7/8 怎么办？heap allocate。

**但关键 insight 来了**：实际程序里 float 分布极不均匀。绝大多数 float 集中在 ±1.0 附近（对数尺度上），加上 ±0.0。极大的数（$10^{100}$）、极小的数（$10^{-100}$）几乎不出现。

所以只要你**选对变换**，让那 1/8 "碰巧命中 tag" 的 float 恰好是实际程序里最常出现的那些，你就赢了——实际命中率接近 100%，heap allocation 接近 0%。

这就像：你不是让所有货物都租仓库格子，而是发现"大部分货物都是那几种常见型号"，于是给这几种型号设计了特殊的"自带 label"包装，只有罕见型号才进仓库。

---

## 具体怎么选变换？

### 3-tag variant：最直观

IEEE754 double 的结构：
```
bit 63:      sign
bit 62-52:   exponent (11 bit)
bit 51-0:    mantissa (52 bit)
```

实际程序里最常出现的 float，它们 exponent 的高 3 bit 基本就是 `000`（±0.0 和极小数）、`011`（±1.0 附近）、`100`（稍大的数）这三种。

**变换**：把整个 64 bit 循环左移 4 位。这样原来 bit 62-60（exponent 高 3 bit）跑到了 bit 2-0（low 3 bit），正好当 tag 用。

```
Before rotate:
[ sign | exp_high3 | exp_low8 | mantissa52 ]
         ^^^^^^^^^
After rol 4:
[ exp_low8 | mantissa52 | sign | exp_high3 ]
                                    ^^^^^^^^^
                                    这就是 tag
```

如果 exponent 高 3 bit 是 `000`、`011`、`100` 之一，rotate 后 low 3 bit 就是这仨，匹配预设 tag → self-tagged，不用 heap allocate。

### 1-tag variant：更省 tag

只用一个 tag。给 float 的 high 6 bit（sign + exponent 高 5 bit）加个常数，再 rotate 5 位。

公式：
$$\text{encode}(n) = \left( n \oplus \left( (1 + 2 \times \text{tag}) \ll 58 \right) \right) \ll_{\text{rot}} 5$$

变量解释：
- $n$：float 的 64 bit 表示，当 unsigned int 看
- $\oplus$：mod $2^{64}$ 加法
- $\text{tag}$：你想用的 tag 值，比如 `000`
- $\ll 58$：把常数左移 58 位，放到 bit 63-58 的位置（就是 sign + exponent 高 5 bit）
- $\ll_{\text{rot}} 5$：循环左移 5 位

直觉：给 exponent 高 5 bit 加 1，让原来 `[01111, 10000]` 这个最频繁范围加完后中间 3 bit 变成 `000`，rotate 后正好命中 tag `000`。

### 2-tag variant：覆盖率翻倍

给 exponent 高 5 bit 加**偶数**（$2 \times \text{tag}$ 而不是 $1 + 2 \times \text{tag}$），中间 3 bit 会等于 $\text{tag}$ 或 $\text{tag}-1$ 两个值都算命中。用 2 个 tag 覆盖 1-tag 两倍的范围——恰好是整个 32-bit float 能表示的范围。

---

## 编码解码成本

**Boxing（f64 → object）3-tag variant 的 x86 汇编**：

```asm
mov rax, xmm0        ; float 搬到 int register
rol rax, 4           ; 循环左移 4 位
bt bx, ax            ; 检查 low 3 bit 是否在 tag set {0,3,4}
jnc heap_alloc_float ; 不在？走慢路径 allocate
```

Hot path 就 3 条 register-to-register 指令 + 1 条几乎总是 predict 对的 jump。非常便宜。

**Unboxing（object → f64）反过来**：

```asm
bt bx, ax
jc self_tagged
mov xmm0, [rax+offset]   ; heap allocated，读 memory
jmp done
self_tagged:
ror rax, 4               ; 循环右移 4 位（逆操作）
mov xmm0, rax
done:
```

---

## 实验结果的人话总结

作者在 Bigloo 和 Gambit 两个 Scheme 编译器上实现了各种 variant，跑 R7RS benchmarks，在 Intel、AMD、Apple M2、RISC-V 四种架构上测。

**核心发现**：

1. **Heap allocation 几乎清零**。Figure 6 显示 self-tagging 的 heap allocation 跟 NaN/NuN-boxing 一样接近 0%。

2. **没有 universal winner**。Float-heavy benchmark 上 NaN/NuN-boxing 通常更快（float 全 unboxed，无 encode/decode 成本）；non-float benchmark 上 self-tagging 通常更快（其他 object 不受拖累）。

3. **GC 类型很关键**。Gambit 用 copying GC + bump allocator，heap 压力小时 allocated float 居然不比 self-tagging 慢——bump allocation 几乎零成本。但 heap 超过 L2 cache size（~1 MB）后，GC 扫描成本主导，self-tagging 反超。Bigloo 用 mark-sweep GC，self-tagging 几乎一直赢，因为少 allocate float 直接减少 mark 工作量。

4. **一个重要的 negative result**：作者试了用 mantissa low bit 当 tag（encode/decode 零成本），结果某些 benchmark 反而变慢。原因：mantissa low bit 对 float 微小变化极度敏感，两个相邻 float 可能一个 self-tagged 一个 heap allocated，branch predictor 学不到规律，misprediction 飙升。

   **教训**：选变换时必须保证捕获的是**连续 float range**，让 "是不是 self-tagged" 这个 branch 有 locality。Exponent high bit 天然满足（连续数值 → 连续 exponent 区间），mantissa low bit 不满足。

---

## 为什么我觉得这 paper 好？

1. **Insight 简单且通用**："可逆变换 + 实际分布不均匀 = 免费午餐"这个思路不限于 float，任何"bit pattern 全占满但实际值分布不均"的场景都能用。

2. **Hybrid representation 是工程智慧**。不追求"所有 float 都 unboxed"的纯粹，接受少数 fallback。这让实现简单（不用重构整个 object system），且实际命中率够高。

3. **Negative result 诚实公布**。Section 4.6 的 mantissa low bit 实验是作者自己一开始的直觉，结果被 branch prediction 打脸，他们老老实实写出来了。这种诚实比纯 positive result 更有价值。

4. **不需要 static analysis**。Self-tagging 是 runtime encoding 层面的技巧，跟 type inference / escape analysis 正交，可以叠加使用。

5. **32-bit 友好**。NaN-boxing 在 32-bit 上基本废掉（NaN 只能放 22-bit pointer），self-tagging 在 32-bit 上照样工作。

---

## Reference

- [Paper PDF (OOPSLA'25)](https://doi.org/10.5281/zenodo.16356364)
- [IEEE 754-2019 Standard](https://doi.org/10.1109/IEEESTD.2019.8766229)
- [R7RS Benchmarks](https://github.com/ecraven/r7rs-benchmarks)
- [Bigloo](https://www-sop.inria.fr/indes/fp/Bigloo)
- [Gambit](https://gambitscheme.org)
- [ALP: float 分布 profiling (VLDB 2023)](https://doi.org/10.1145/3626717)
- [Daan Leijen - Tagged Integers hardware support](https://www.microsoft.com/en-us/research/publication/what-about-the-integer-numbers-fast-arithmetic-with-tagged-integers-a-plea-for-hardware-support/)
- [V8 Pointer Compression blog](https://v8.dev/blog/pointer-compression)
- [SpiderMonkey](https://spidermonkey.dev/)
- [JavaScriptCore / WebKit](https://webkit.org)
- [CPython PEP 703 (no-GIL)](https://peps.python.org/pep-0703)
- [Gudeman 1993 - Type info in dynamic languages](https://www.cs.arizona.edu/sites/default/files/TR93-27.pdf)

---

# Float Self-Tagging 论文详解

Andrej，这篇 paper 解决的是 dynamic language 里一个经典的老大难问题：**IEEE754 double 把 64 bit 全用光了，没地方塞 type tag**。作者 Olivier Melançon、Manuel Serrano、Marc Feeley 提出了一个叫 **self-tagging** 的新思路，核心 insight 非常 elegant，我来一层一层剥开讲。

---

## 1. 问题背景：为什么 float 这么难处理？

Dynamic language (Scheme, JavaScript, Python, Lua, Ruby...) 在 runtime 需要给每个 value 附带 type 信息。常用方案是 **tagged object**：在 64-bit machine word 的 low bits 留 3 bit 当 tag (8 个 tag 可用)，剩下的 bit 存 value 或 pointer。

三种 representation：

| Representation | Tag 含义 | Value 存哪 | Heap allocation? |
|---|---|---|---|
| **Tagged value** | type | inline in word | No |
| **Tagged pointer** | type | heap, pointer in word | Yes |
| **Generic pointer** | "其他" | heap, type 在 header 里 | Yes |

对 small integer 来说，用 tag `000` 做 tagged value，61 bit 存整数，加法可以直接用 machine instruction 加完再 fixup tag——非常爽。

**但 double 不行**。IEEE754 double-precision 结构：

```
[ sign(1) | exponent(11) | mantissa(52) ]
  bit 63    bit 62..52     bit 51..0
```

64 bit 全用光，CPU 的 float 指令 (x86 `addsd`, ARM `fadd`) 严格要求 64-bit IEEE754 输入，没法腾出 bit。

### 三种现有方案及其痛点

**1. Tagged pointers for floats**：每个 float 都 heap allocate。后果：
- 每次用 value 都要 dereference memory
- 大量 short-lived intermediate floats 给 GC 巨大压力
- JavaScript 这种 "float 是唯一 number type" 的语言尤其惨

用此方案的：V8, QuickJS, Hopc, Lua interpreter, CRuby, SBCL, Bigloo, Gambit, CPython (PEP 703 后)。

**2. NaN-boxing**：IEEE754 规定 exponent 全 1 且 mantissa ≠ 0 是 NaN。其中 quiet NaN 的最高 mantissa bit = 1，signaling NaN = 0。**negative quiet NaN** 区间 `0xfff8_0000_0000_0000` 到 `0xffff_ffff_ffff_ffff` 有 51 bit 空间没用。

把 non-float object 塞进这 51 bit（48-bit pointer + 3-bit tag）：
```
float:        bit pattern < 0xfff8_0000_0000_0000
non-float:    bit pattern >= 0xfff8_0000_0000_0000  (装在 NaN 里)
```

痛点：type check 和 dereference pointer 都要额外指令（要先 mask 高位看是不是 NaN-boxed，再 extract pointer）。32-bit 上直接废掉（32-bit float 的 NaN 只能放 22-bit pointer，4 MiB 地址空间）。

用此方案的：SpiderMonkey, tinylisp, LuaJIT, Zag。

**3. NuN-boxing (pointer-biased NaN-boxing)**：给所有 float 加 bias `0x0001_0000_0000_0000`，把 float 推到中间区间，low 和 high 16 bit 留给 tagged object。这样 pointer 的 tag 直接在 low bit，dereference 不要额外指令；但 float 用之前要减 bias，用完要加 bias。

用此方案的：JavaScriptCore (JSC)。

---

## 2. Self-Tagging 的核心 idea

这是 paper 的核心 insight，非常漂亮：

> 任意一个**可逆 bitwise transformation** $f$ 作用于 IEEE754 float 上，会有恰好 $1/8$ 的 float 被映射到 low 3 bits 等于某个特定 tag $T$ 的 bit pattern。这些 float 可以直接当 tagged value 用，不需要 heap allocation。

形式化一点：给定 tag $T \in \{0,\dots,7\}$ 和可逆变换 $f$，定义 self-tagged 集合：
$$S_T = \{ x \in \text{float64} : \text{low3}(f(x)) = T \}$$

那么 $|S_T| = 1/8 \cdot |\text{float64}|$。

**关键观察**：float 在实际程序里**分布极不均匀**。参考文献 [3, 9, 27, 43] 的 profiling 显示，real-world 程序里的 float 高度集中在 ±1.0 附近（log scale），加上 ±0.0。subnormal 极少出现，极大值极少出现。

所以**只要选对 transformation**，就能让 $S_T$ 覆盖实际程序里出现的大部分 float，剩下的少数 fallback 到 heap allocation。

这跟 NaN-boxing 的哲学完全不同：NaN-boxing 是"我保证所有 float 都 unboxed，代价是其他 object 受影响"；self-tagging 是"我保证其他 object 不受影响，代价是少数 float 还要 heap allocate"——但实际这"少数"在 profiling 下几乎是 0%。

---

## 3. Float 分布的实证 (Figure 5)

作者 instrument 了 R7RS benchmarks (Scheme 标准测试集) 来 profile float 分布。Table 按 11-bit exponent 的**高 5 bit** 分桶（32 个桶）。

几个观察：
- **`01111`** 桶（exponent 高 5 bit = 01111，即 exponent 在 0b01111xxxxxx 范围）覆盖 $1.1 \times 10^{-19}$ 到 $2$，几乎所有 benchmark 这里占比都最高（nucleic 89%, pnpoly 61%, simplex 63%, sum1 64%）
- **`10000`** 桶覆盖 $2$ 到 $3.7 \times 10^{19}$，也是高频（mbrot 63%, sumfp 70%, sum1 75%, ray 100%）
- **`00000`** 桶（含 ±0.0 和 subnormal）在某些 benchmark 占比高（fibfp 79%）
- 中间桶（极小或极大的 float）几乎都是 0%

也就是说：实际程序里出现的 float，**3 bit exponent 高位** 取值基本就是 `000`, `011`, `100` 三个值。

---

## 4. Self-Tagging 的几个 variant

### 4.1 3-tag variant (Figure 4)

最直观的方案：**4-bit left rotation**。

```
IEEE754 double:
[ s | e10 e9 e8 | e7...e0 | m51...m0 ]
       ^^^^^^^^^
       这 3 bit rotate 后变成 low 3 bit = tag

After rol 4:
[ e7...e0 | m51...m0 | s | e10 e9 e8 ]
                              ^^^^^^^^^
                              现在 low 3 bit 是 tag
```

用 tags `000`, `011`, `100`，覆盖：
- tag `000`：exponent 高 3 bit = `000`，即 ±0.0 和 subnormal 以及极小 normal ($\sim 10^{-251}$ 到 $\sim 10^{-289}$)
- tag `011`：exponent 高 3 bit = `011`，范围 $\sim 10^{-77}$ 到 $\sim 10^{38}$（含 ±1.0 附近）
- tag `100`：exponent 高 3 bit = `100`，范围 $\sim 10^{38}$ 到 $\sim 10^{77}$

加上对应负数范围。Figure 5 的绿/深蓝/浅蓝行。

### 4.2 4-tag variant

再加 tag `111`（exponent 高 3 bit = `111`，覆盖 ±Infinity 和 NaN）。Gambit 实现了这个。

### 4.3 2-tag with preallocated zeros (Bigloo)

把 tag `000` 让出来给其他 type 用，±0.0 用预分配的 tagged pointer 表示。Boxing 时先测 `== ±0.0`，是的话返回预分配对象。

痛点：**那个 `== 0` 的测试会引入 branch misprediction**（详见 Section 4.6，这是个 negative result）。

### 4.4 1-tag variant (Section 2.4)

只用一个 tag，但 transformation 更巧妙。

考虑 exponent 高 5 bit $e_{10\ldots 6}$。如果给 float 加 bias 让这 5 bit 变成 $e_{10\ldots 6} + 1$，那么中间 3 bit $e_{9\ldots 7}$ 在加 1 后的分布会发生变化。具体地，原来 $e_{10\ldots 6} = 01100$ 到 $10001$（即程序里最频繁的范围），加 1 后变成 $01101$ 到 $10010$，**中间 3 bit 都是 `100`** 或 `011`。

等等，让我重新读一下公式。Section 2.4 给的公式是：

$$\text{encode}(n) = \left( n \oplus \left( (1 + 2 \times \text{tag}) \ll 58 \right) \right) \ll_{\text{rot}} 5$$

变量含义：
- $n$：float 的 IEEE754 64-bit 表示，当作 unsigned integer
- $\oplus$：mod $2^{64}$ 加法
- $\text{tag}$：3-bit tag 值，比如 `000`
- $(1 + 2 \times \text{tag}) \ll 58$：把常数 $(1 + 2 \times \text{tag})$ 放到 high 6 bit 的位置（58 = 64 - 6，所以这个常数占据 bit 63..58）
- $\ll_{\text{rot}} 5$：5-bit 循环左移

为什么是 high 6 bit？因为 IEEE754 double 的 sign(1) + exponent(11) = 12 bit，高 6 bit 是 sign + exponent 高 5 bit。给这 6 bit 加上一个常数，等价于给 sign 和 exponent 高 5 bit 加常数。

加 `1` 到 high 6 bit 等价于给 exponent 高 5 bit 加 1（因为 sign bit 在最高位，加 1 不影响 sign，进位发生在 exponent 字段）。

然后 5-bit rotation 把原来 bit 62..58 (即 sign + exponent 高 5 bit) 旋转到 low 5 bit。low 3 bit 就是原来 exponent 的 bit 9, 8, 7。

实际效果：经过加 bias 再 rotate，那些原本 exponent 高 5 bit 在 $[01100, 10001]$ 范围的 float（最频繁的范围），加 1 后变成 $[01101, 10010]$，其中 bit 9..7 全是 `100` 或 `011`——具体看 Figure 5 第二列"5 most signif. expo. bits + 1"。

让 tag = `000`，覆盖的是加 1 后中间 3 bit = `000` 的范围，对应原 exponent 高 5 bit 是 `01111` 或 `10000`（即 Figure 5 里 bold pink 行），覆盖 $1.1 \times 10^{-19}$ 到 $3.7 \times 10^{19}$ 以及 Infinity/NaN。

### 4.5 2-tag variant (Section 2.5, Gambit 用的)

基于 1-tag 的扩展。如果加 $2 \times \text{tag}_1$ 而不是 $1 + 2 \times \text{tag}_1$，那么中间 3 bit 等于 $\text{tag}_1$ 或 $\text{tag}_1 - 1$ 的都算 self-tagged（因为加偶数不改变中间 3 bit 的奇偶性，但会让某些范围"分裂"成两个相邻 tag）。

效果：用 2 个 tag 覆盖的范围是 1-tag 的两倍，包括：
$$0 \dots 3.8 \times 10^{-270} \cup 5.9 \times 10^{-39} \dots 6.8 \times 10^{38} \cup 1.1 \times 10^{270} \dots \text{Infinity/NaN}$$

这是 IEEE754 32-bit float 范围的超集——任何能表示成 32-bit float 的值都能 self-tag。

**实现优势**：bias 可以在 rotation 之后加（因为 rotate 后 tag 在 low bit，bias 变成 small constant），当 $\text{tag}_1 = 000$ 时 bias 直接消失。这让指令比 1-tag variant 还少一条。

---

## 5. Implementation 细节 (Section 3)

这部分对 build intuition 很关键，看看实际汇编长什么样。

### 5.1 Single tag testing

测 low 3 bit 是否等于某个 tag：

```asm
and al, 7        ; al = low 8 bit of rax, mask 低 3 bit
cmp al, tag
jz tag_matches
```

问题：破坏了 `rax`。如果 `tag = 0`，可以用 `test`：

```asm
test al, 7       ; 不破坏 rax
jz tag_matches
```

对 `tag ≠ 0`，用 `lea` 技巧：把 `tag1` 映射到 0，然后用 `test`：

```asm
lea ebx, [eax + (8 - tag)]   ; ebx = eax + (8 - tag), 把 tag 变 0
test bl, 7
jz tag_matches
```

### 5.2 Multiple tag testing

用 x86 的 `bt` (bit test) 指令。比如测 tags {0, 3, 4}：

```asm
mov bx, 0x1919     ; 0x19 = 0b00011001, bit 0/3/4 = 1
bt bx, ax          ; 取 bx 的第 (ax mod 16) bit 放进 carry flag
jc tag_matches
```

`0x1919` 是把 `0x19` 重复两次填满 16-bit register，因为 `bt` 用 index mod 16。

ARM A64 没有 `bt`，用 shift：

```asm
lslv w3, w2, w1     ; w3 = w2 << w1
cmp w3, 0
bmi matching_tag    ; 检查 sign bit
```

`w2` 预载 `0x98989898`（bit-reversed tag set，让 tag 0 对齐到 sign bit）。

**关键点**：check 成本跟 tag 数量无关——加一个 tag 到 set 里只是改 `mov` 的常数，不增加指令。这对 hybrid representation（self-tagged + heap allocated float 都用 tagged pointer）很有用：把 heap float 的 tag 也加进 set，一次 check 就知道是不是 float。

### 5.3 Boxing: f64 → object (3-tag variant)

```asm
mov rax, xmm0        ; rax ← xmm0 (从 float reg 搬到 int reg)
rol rax, 4           ; 4-bit 循环左移
bt bx, ax            ; bx = 0x1919 预载
jnc heap_alloc_float ; tag 不在 {0,3,4}? 走慢路径
done:
```

Hot path：3 条 register-to-register 指令 + 1 条 easily predictable conditional jump。

### 5.4 Boxing: 1-tag variant

```asm
mov rax, xmm0
add rax, rbx         ; rbx = 1 << 58 预载
rol rax, 5
test al, 7
jnz heap_alloc_float
```

4 条指令。多一条 `add`，因为 bias 是 large constant 要用 register。

### 5.5 Boxing: 2-tag variant (tag1=011)

```asm
mov rax, xmm0
rol rax, 5
add al, 3            ; small constant, 只改 low 8 bit
bt bx, ax            ; bx = 0x0c0c 预载 (tags 010, 011)
jnc heap_alloc_float
```

注意 `add al, 3` 用的是 byte register，指令编码短。当 `tag1 = 000` 时这条 `add` 直接删掉，跟 3-tag variant 一样快（3 条指令）。

### 5.6 Unboxing: object → f64 (3-tag variant)

```asm
bt bx, ax
jc self_tagged
mov xmm0, [rax + offset]    ; heap allocated, 从 memory 读
jmp done
self_tagged:
ror rax, 4                  ; 4-bit 循环右移 (rol 的逆)
mov xmm0, rax
done:
```

### 5.7 C 实现 (Section 3.5)

两个 trick：

**(1) 用 union 做 f64 ↔ i64 转换**：

```c
union di { double d; int64_t i; };
```

**(2) 用 shifts 模拟 rotation**（GCC/clang 会识别并合成一条 `rol`/`ror`）：

```c
#define ROTL(n, s) ((int64_t)(((uint64_t)n << s) | ((uint64_t)n >> (64 - s))))
```

**(3) 用 bitwise trick 模拟 `bt`**：

```c
#define TAG_SET ((1<<0)|(1<<3)|(1<<4))  // 0x19

inline bool has_tag_0_or_3_or_4(int64_t n) {
    return (((uint32_t)1 << (n & 31)) & (~(uint32_t)0/0xff * TAG_SET)) != 0;
}
```

变量解释：
- `n & 31`：取 low 5 bit 作为 shift count（mod 32，因为 `uint32_t` 的 shift 是 mod 32）
- `(uint32_t)1 << (n & 31)`：构造一个只有第 `(n mod 32)` bit 为 1 的 mask
- `~(uint32_t)0 / 0xff`：等于 `0x01010101`，把 1 byte 复制 4 次填满 32 bit
- `~(uint32_t)0 / 0xff * TAG_SET`：等于 `0x19191919`，把 tag set 复制 4 次
- `&` 之后非零 ⇔ tag set 的第 `(n mod 8)` bit 是 1 ⇔ `n` 的 low 3 bit 在 tag set 里

GCC 13.2+ 和 clang 18.1+ 会把这个优化成 `bt` 指令。

---

## 6. 实验结果 (Section 4)

### 6.1 Setup

- **Compilers**：Bigloo (commit 5b1118) 和 Gambit v4.9.7 (commit 768900)，两个独立开发的 Scheme 编译器
- **Machines**：Intel Xeon W-2245, AMD Ryzen 7955WX, Apple M2 Max, RISC-V SiFive u74-mc
- **Benchmarks**：R7RS benchmarks 里的 macro-benchmarks 和 float-intensive benchmarks（fibfp, fft, mbrot, nucleic, pnpoly, ray, simplex, sum1, sumfp）
- **Repetitions**：每个配置跑 10 次，每次至少 5 秒，配对计算 relative time

### 6.2 Memory profiling (Figure 6)

Self-tagging 几乎消除所有 float heap allocation。除少数 benchmark（如 ray 在某些 variant 下还有微量 allocation），都跟 NaN/NuN-boxing 一样接近 0%。

### 6.3 Execution time (Figures 7-9)

Figure 9 的 summary table 是关键。每行是一个 encoding，每列是 (compiler, microarchitecture, benchmark type)，数值是相对 NuN-boxing 的 geometric mean execution time（< 1.0 = 比 NuN-boxing 快）。

几个观察：

**Bigloo non-float benchmarks**：
- 1-tag: 0.97-1.01× （基本持平或略快）
- 3-tag: 1.00-1.02×
- NaN-boxing: 1.00-1.06× （略慢，因为 non-float object 受影响）

**Bigloo float benchmarks**：
- 1-tag: 1.20-1.28× （慢 20-28%）
- 3-tag: 1.09-1.19×
- NaN-boxing: 0.84-0.90× （最快，因为 float 全 unboxed 且无 encode/decode）

**Gambit non-float benchmarks**：
- 1-tag: 0.87-0.95× （明显快！）
- 4-tag: 0.89-0.97×
- 这说明 Gambit 的 NuN-boxing baseline 对 non-float 有显著 overhead

**Gambit float benchmarks**：
- 4-tag: 0.90-1.04× （跟 NuN-boxing 持平或略快）
- 1-tag: 0.94-1.06×

**结论**：没有 universal winner。Float-heavy 程序 NaN/NuN-boxing 通常赢；non-float 程序 self-tagging 通常赢。Self-tagging 在 non-float 上的优势来自"不影响其他 object 编码"。

### 6.4 GC 压力的影响 (Section 4.5, Figure 12)

这是 paper 里一个重要的 nuanced finding。Gambit 用 stop-and-copy GC + bump allocator，Bigloo 用 Boehm mark-and-sweep。

实验：在 benchmark 开始前 preallocate 一个大 vector，size 从 $10^4$ 到 $10^8$ fields（80 KB 到 800 MB），制造 heap 压力。

结果（Figure 12, AMD machine）：
- **Heap 几乎空**时：Gambit 的 bump allocator 让 allocated floats 有时**比 self-tagging 还快**——因为 bump allocation 几乎零成本，而 self-tagging 要付 encode/decode
- **Heap > 1 MB**（约 L2 cache size per core）后：GC 扫描成本开始主导，self-tagging 反超

对 Bigloo（Boehm GC），self-tagging 几乎在所有 heap size 下都更快，因为 Boehm GC 的 mark-sweep 对 live object 数量敏感，少 allocate float 直接减少 mark 工作量。

**Insight**：self-tagging 的收益跟 GC 的性质强相关。Copying GC + bump allocator 的系统在 heap 压力小时收益不明显；mark-sweep GC 的系统收益直接。

### 6.5 Branch prediction 的 negative result (Section 4.6)

作者尝试了一个直觉上很 attractive 的 variant：**用 mantissa 的 low bits 做 tag**。

比如 reserve tag `000` 给 self-tagged float，那么所有 low 3 bit mantissa = `000` 的 float 都能 self-tag。这个 variant **encode/decode 零成本**（self-tagged float 的 bit pattern 跟 IEEE754 完全一样）。

用 tags `000` 和 `100`，覆盖 1/4 的所有 float（包括所有 $|n| \leq 2^{51}$ 的整数 float）。

**结果（Figure 13, 14）**：
- `fibfp` 和 `sumfp`（大量整数 float）确实变快了
- 但 `mbrot` 和 `nucleic` **变慢了**，尽管 allocate 的 float 少了 1/4

为什么？Figure 14 显示 branch misprediction 飙升。原因是：mantissa low bit 对 float 值的微小变化极度敏感。两个相邻的 float 值（比如 mandelbrot 迭代中的中间结果）可能一个 self-tagged、一个 heap allocated，pattern 完全随机，branch predictor 学不到。

**教训**：self-tagging transformation 必须捕获**连续的 float range**，让"是不是 self-tagged"这个 branch 在程序运行时表现出 temporal/spatial locality，branch predictor 才能工作。Exponent high bits 天然满足这个（一个连续数值范围对应一个连续 exponent 区间）；mantissa low bits 不满足。

这也解释了为什么 2-tag with preallocated zeros 的 ±0.0 测试会拖慢——那个 `== 0` 的 branch 在某些 benchmark 里也会 mispredict。

---

## 7. 32-bit 适配 (Section 5)

32-bit 上 popular 是 2-bit tag（4 个 tag）。IEEE754 single-precision float：1 sign + 8 exponent + 23 mantissa。

1-tag 公式：
$$\text{encode}(n) = \left( n \oplus \left( (1 + 2 \times \text{tag}) \ll 27 \right) \right) \ll_{\text{rot}} 4$$

变量：
- $n$：32-bit float 表示
- $\oplus$：mod $2^{32}$ 加法
- $(1 + 2 \times \text{tag}) \ll 27$：常数放到 high 5 bit (bit 31..27 = sign + exponent 高 4 bit)
- $\ll_{\text{rot}} 4$：4-bit 循环左移，把 high 4 bit 移到 low 4 bit，low 2 bit 就是 tag

用 tag = `00` 覆盖：
$$0.0 \dots 3.9 \times 10^{-34} \cup 3.1 \times 10^{-5} \dots 1.3 \times 10^{5} \cup 1.0 \times 10^{34} \dots \text{Infinity/NaN}$$

这是 32-bit float 里最常用的范围。

2-tag variant (tags `00` 和 `11`) 覆盖更广：
$$0.0 \dots 2.5 \times 10^{-29} \cup 4.7 \times 10^{-10} \dots 8.6 \times 10^{9} \cup 1.6 \times 10^{29} \dots \text{Infinity/NaN}$$

剩两个 tag：一个给 small integer ($-2^{29}$ 到 $2^{29}-1$)，一个给 heap object。

NaN-boxing 在 32-bit 上基本废掉（只能 22-bit pointer），所以 self-tagging 在 32-bit 上是特别有吸引力的方案。

---

## 8. 跟现有实现的比较

Paper 提到几个已有的"类 self-tagging"实现：

- **CRuby** 和 **OpenSmalltalk**：用接近 1-tag variant 的 encoding，但**排除 ±0.0**，要 special-case 处理
- **MonNom calculus** (paper [32])：用 2-bit tag 的 2-tag variant，也排除 ±0.0

这些实现因为要 special-case ±0.0，encode/decode 指令序列复杂，难以 inline，且有 branch misprediction 风险（正是 Section 4.6 说的）。Self-tagging 的 1-tag 和 2-tag variant 通过巧妙选择 bias **包含** ±0.0，避免了这个问题。

---

## 9. 我的 takeaways

给你几个我觉得最值得 internalize 的点：

1. **Bit-level invertibility + 实际分布的不均匀性 = 免费午餐**。任意可逆变换都能 cover 1/8 的 float，但选对变换 + 利用实际分布能 cover 接近 100% 的实际出现的 float。这是个非常通用的思路，不限于 float。

2. **Branch predictability 是第一公民**。Section 4.6 的 negative result 极其重要：一个看起来"更优"的 variant（零成本 encode/decode）实际更慢，因为破坏了 branch locality。做 system 优化时，"指令少"不等于"快"，branch pattern 的可预测性往往更重要。

3. **GC 性质决定收益**。Bump allocator + copying GC 的系统在 heap 压力小时对 allocation 不敏感；mark-sweep GC 的系统对 live object 数量敏感，self-tagging 收益直接。这提醒我们：优化要 profile 整个 system，不是 isolated micro-benchmark。

4. **Hybrid representation 是 pragmatic 的胜利**。Self-tagging 不追求"所有 float 都 unboxed"的纯粹性，接受少数 float fallback 到 heap。这种"覆盖 90%+ 就够"的工程思维让实现简化（不用像 NaN-boxing 那样重构整个 object encoding）。

5. **跟 static analysis 正交**。Self-tagging 不需要 type inference，可以直接 drop in 到任何 tagged object system。跟 escape analysis / storage use analysis 这些技术叠加使用效果更好。

---

## 参考链接

- [IEEE 754-2019 Standard](https://doi.org/10.1109/IEEESTD.2019.8766229)
- [R7RS Benchmarks](https://github.com/ecraven/r7rs-benchmarks)
- [Bigloo Scheme compiler](https://www-sop.inria.fr/indes/fp/Bigloo)
- [Gambit Scheme compiler](https://gambitscheme.org)
- [V8 Pointer Compression](https://v8.dev/blog/pointer-compression)
- [SpiderMonkey](https://spidermonkey.dev/)
- [JavaScriptCore / WebKit](https://webkit.org)
- [LuaJIT](https://luajit.org)
- [CPython PEP 703 (removing GIL)](https://peps.python.org/pep-0703)
- [ALP: Adaptive Lossless floating-Point Compression (float 分布 profiling)](https://doi.org/10.1145/3626717)
- [Paper artifact (Zenodo)](https://doi.org/10.5281/zenodo.16356364)
- [Daan Leijen - Tagged Integers hardware support](https://www.microsoft.com/en-us/research/publication/what-about-the-integer-numbers-fast-arithmetic-with-tagged-integers-a-plea-for-hardware-support/)
- [Gudeman 1993 - Representing Type Information in Dynamically Typed Languages](https://www.cs.arizona.edu/sites/default/files/TR93-27.pdf)

如果你想进一步 build intuition，我建议去看 Figure 5 那个 profiling table，自己拿几个 benchmark 的 float 值算算 exponent 高位，看看为什么 `011` 和 `100` 这两个 tag 能 cover 这么多。那个 table 是整个 paper 的 empirical foundation，理解了它，所有 variant 设计的选择就都 make sense 了。
