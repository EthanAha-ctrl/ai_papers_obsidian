好的，我来详细说说 **Ninja** —— 一个成功的 build system（构建系统）。基于你提供的文章以及我对 Ninja 的了解，我将从多个角度深入剖析。

---

## 1. Ninja 是什么？

**Ninja** 是一个专注于**速度**的 build system，由 Evan Martin（Google 工程师）于 2011 年发布。它的核心设计理念可以概括为：

> **Ninja 只做一件事：快速执行 build graph（构建图）。**

### 与 Make 的关系
Ninja 的功能与 **Make** 类似，但有以下关键区别：
- **更少的 features**：Ninja 的 build language（构建语言）非常精简，没有 Make 那样的复杂特性（如 globbing、variable expansion、functions 等）
- **更快的执行**：专注于 incremental build（增量构建）的极致性能
- **分离架构**：Ninja 只是 "assembler"（汇编器），上层 generator 负责生成 `.ninja` 文件

### 广泛使用
- **Chrome**：完全移除了非 Ninja 的构建
- **Android**：大型组件使用 Ninja
- **Meson**：默认使用 Ninja
- **CMake + Ninja**：成为现代 C/C++ 项目的事实标准

---

## 2. 核心技术原理

### 2.1 三个核心步骤

Ninja 的工作流程可以形式化为：

$$
\text{Execute} = f(\text{Parse}, \text{Check}, \text{Run})
$$

其中：
1. **Parse**：解析 `build.ninja` 文件
2. **Check**：检查所有文件的 modification timestamps（修改时间戳）
3. **Run**：并行执行需要更新的 commands

**目标**：对于超过 100k 输入文件的项目，尽可能快地到达步骤 3。

### 2.2 时间复杂度分析

假设：
- $n$ = 文件数量
- $m$ = command 数量
- $k$ = 平均每个 command 的依赖数

Ninja 的设计目标是：

$$
T_{\text{incremental}} = O(n + m) \approx \text{milliseconds}
$$

而传统 Make 的开销可能达到：

$$
T_{\text{make}} = O(n \cdot k + m \cdot \text{overhead})
$$

### 2.3 Interned Strings 优化

文章中提到的一个关键优化：

> Ninja maps each input file path to a unique in-memory object, then uses pointer comparisons for path equality.

**技术细节**：
```cpp
// 伪代码表示
map<string, StringObject*> path_intern;

// 每个路径只在内存中存储一次
StringObject* intern(string path) {
    if (path_intern.find(path) == path_intern.end()) {
        path_intern[path] = new StringObject(path);
    }
    return path_intern[path];
}

// 比较只需 O(1) pointer comparison
bool equal(StringObject* a, StringObject* b) {
    return a == b;  // 不是 strcmp!
}
```

这避免了在 build graph 遍历过程中的大量 string comparisons（字符串比较）。

---

## 3. 架构设计亮点

### 3.1 Bipartite Graph（二分图）表示

这是 Ninja 最重要的架构创新之一。

#### Make 的表示（推测）
```
File A → File B → File C
    (edges are between files)
```

**问题**：无法很好地表示一个 command 产生多个 outputs 的情况。

#### Ninja 的表示
```
          ┌─────────────┐
File A ──→│             │
File B ──→│  Command 1  │──→ File D
File C ──→│             │──→ File E
          └─────────────┘

(bipartite graph: files ↔ commands)
```

**数学形式**：
- $V = V_f \cup V_c$，其中 $V_f$ 是 file nodes，$V_c$ 是 command nodes
- $E \subseteq V_f \times V_c \cup V_c \times V_f$
- **Invariant**：每个 file 最多有一个 incoming edge（表示谁产生了它）

**优势**：
1. 正确捕获"一个 command 更新所有 outputs"的语义
2. Command line 本身也可以作为 "input" —— 如果 flags 改变，command 就 out of date

### 3.2 Deps Log（依赖日志）

为了正确处理 C header dependencies，Ninja 使用了一个 compact 的 deps log 格式。

**为什么需要 deps log？**

考虑 C 编译：
```
gcc -c foo.c -o foo.o
```

`foo.c` 可能 `#include` 了 `bar.h`，而 `bar.h` 又 `#include` 了 `baz.h`。为了正确增量构建，Ninja 需要知道 `foo.o` 依赖于 `foo.c`, `bar.h`, `baz.h`。

**工作流程**：
1. 第一次构建时，使用 `gcc -MMD -MF foo.o.d` 生成依赖信息
2. Ninja 解析并存储到 deps log
3. 后续构建直接读取 deps log，不需要重新扫描

**格式设计**（simplified）：
```
# 每个 entry 记录一个 output 文件的所有 inputs
entry ::= output_hash input_count input_hash*
```

使用 hash 而非完整路径，节省空间。

### 3.3 End-to-End / Crash-Only 设计

Ninja **不是**一个 persistent daemon（持久守护进程），而是每次从头执行。

**设计哲学**：
```
          ┌──────────────────────────────────┐
          │   If you make startup fast,      │
          │   you don't need a daemon.       │
          └──────────────────────────────────┘
```

**原因**：
- 假设你需要能够从 scratch 运行（比如机器 reboot 后）
- 如果从 scratch 运行已经很快，那为什么需要维护一个复杂的 daemon？
- 避免 "fast path" 和 "slow path" 两套代码

**数据支持**：
> On a "fast" machine from 10 years ago, you can stat 30k files in 10s of milliseconds.

### 3.4 Kernel Caching vs. Userland Caching

文章中提到一个有趣的 insight：

> The kernel is already caching file status in memory. Caching it again in userland doesn't save you much.

**Linux stat() 系统调用的性能**：
```
stat("/path/to/file", &st);  // ~1-2 microseconds
```

对于 30k 文件：
$$
30,000 \times 2\mu s = 60ms
$$

这已经足够快，不需要用户态 cache。

**编程笑话**：
> Half of performance problems are fixed by introducing a cache; the other half are fixed by removing one.

---

## 4. "Assembler" Metaphor（汇编器隐喻）

这是 Ninja 最重要的设计洞察。

### 4.1 Build Systems 的 Spectrum（谱系）

```
High-level                              Low-level
    │                                       │
    │   CMake   Meson   GYP   GN   ───►   Ninja
    │    │       │      │     │           │
    │    └───────┴──────┴─────┘           │
    │         Generators                  │
    │                                      │
    └──────────────────────────────────────┘
                    Build Pipeline
```

**关键分离**：

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **Generator** | High-level logic: find sources, configure options, etc. | CMake, Meson, GYP, GN |
| **Ninja** | Low-level execution: run commands, check timestamps | - |

### 4.2 为什么这个分离很重要？

#### Forcing a Snapshot（强制快照）

Generator 运行时：
```python
# 伪代码
sources = glob("src/**/*.c")  # 耗时操作
commands = []
for src in sources:
    obj = src.replace('.c', '.o')
    commands.append(f"gcc -c {src} -o {obj}")

write_build_ninja(commands)  # 写入 .ninja 文件
```

**效果**：
- Glob 操作只在 generator 运行时执行一次
- 后续的 `ninja` 命令直接读取 snapshot
- 隐式地 cache 了 action graph

#### Making Costs Visible（让成本可见）

如果 generator 做了很多 slow operations（如 recursive glob），用户会直接看到 generator 运行慢。这把"性能责任"放到了正确的层级。

### 4.3 Greenspun's Tenth Rule

> Any sufficiently complicated C or Fortran program contains an ad hoc, informally-specified, bug-ridden, slow implementation of half of Common Lisp.

Make 试图提供所有 features（globbing, variables, functions, etc.），结果：
- Language 太弱，无法表达所有需求 → autotools 的复杂性
- Language 太强，让人们写出 slow Makefiles

Ninja **故意** 避免了这条道路。

---

## 5. 默认值的重要性

### 5.1 Parallel Execution by Default

**Make**：
```bash
make        # 默认串行执行
make -j4    # 需要 -j flag 才并行
```

**Ninja**：
```bash
ninja       # 默认并行执行（使用所有 CPU cores）
ninja -j1   # 需要显式指定才串行
```

### 5.2 为什么这个默认值重要？

**定理**：如果默认串行，用户会写出 unsafe-for-parallelism 的 build files。

**证明**（informal）：
1. 用户写 Makefile 时，默认用 `make` 测试
2. 串行执行时，依赖关系不完整也能工作
3. 用户不知道有问题，发布代码
4. 其他人用 `make -j` 时，build 崩溃或产生错误结果

**Ninja 的设计**：
1. 默认并行执行
2. 如果依赖关系不完整，build 立即失败
3. 强迫用户修复依赖关系
4. 结果：所有用 Ninja 的项目天生 safe for parallelism

### 5.3 数据

文章提到：
> Ninja in practice will end up being "twice as fast as Make" or more for users who aren't careful.

这不是因为 Ninja 更快，而是因为：
$$
T_{\text{ninja}} \approx \frac{T_{\text{serial}}}{N_{\text{cores}}}
$$

而很多 Make 用户从未用 `-j`。

---

## 6. 性能优化策略

### 6.1 Orders of Magnitude（数量级）

文章中提到一个经验法则：

> You can get scale by 2x with optimization, but to scale by 10x you need to rearchitect.

**例子**：
| Approach | Speedup | Example |
|----------|---------|---------|
| Micro-optimization | 2x | Pointer comparison vs. string comparison |
| Algorithmic optimization | 10x | Linear scan vs. hash lookup |
| Architectural redesign | 100x | Two-layer design (generator + executor) |

### 6.2 Latency vs. Throughput

Ninja 关注的是 **latency**（延迟），不是 throughput（吞吐量）。

**场景**：
```
Edit → Compile → Test → Edit → Compile → Test → ...
         ↑
      这个延迟最重要
```

**心理学**：
| Latency | User Experience |
|---------|-----------------|
| < 1s | Feels instant |
| 1-2s | Acceptable |
| 2-5s | Annoying |
| > 5s | Context switch, lose focus |

### 6.3 Memory Resident vs. Process Startup

文章提到一个有趣的对比：

**Blaze/Bazel**（Java 程序）：
```bash
$ bazel help
# Even printing help takes seconds because JVM startup
```

**Ninja**（C++ 程序）：
```bash
$ ninja -h
# Prints help instantly
```

### 6.4 Output Verbosity

Ninja 默认输出非常简洁：
```
[3/100] CXX foo.o
```

成功构建时可能只输出一行：
```
ninja: Entering directory `out/Debug'
```

**心理效果**：
- 少输出 = 感觉轻量
- 其他 build systems 的"彩色进度条"让人感觉"heavy"

---

## 7. 与 Make 的详细对比

### 7.1 Feature Comparison

| Feature | Make | Ninja |
|---------|------|-------|
| Globbing | ✅ `$(wildcard *.c)` | ❌ Generator 负责处理 |
| Variables | ✅ `CC = gcc` | ✅ Limited |
| Functions | ✅ `$(subst .c,.o,$(srcs))` | ❌ |
| Conditionals | ✅ `ifdef` / `ifeq` | ❌ |
| Pattern rules | ✅ `%.o: %.c` | ✅ |
| Parallel execution | ❌ (opt-in with `-j`) | ✅ (default) |
| Speed | Slower | Faster |
| Feature richness | High | Low |

### 7.2 Architecture Comparison

**Make**：
```
┌─────────────────────────────────────┐
│            Makefile                 │
│  ┌─────────────────────────────┐   │
│  │ Globbing + Variables +      │   │
│  │ Functions + Dependencies    │   │
│  │ + Execution all in one      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

**Ninja**：
```
┌──────────────────┐       ┌──────────────────┐
│    Generator     │       │     Ninja        │
│  (CMake/Meson)   │       │  (Executor)      │
│                  │       │                  │
│ - Find sources   │──────→│ - Parse graph    │
│ - Set variables  │ .ninja│ - Check mtime    │
│ - Generate graph │ file  │ - Run commands   │
└──────────────────┘       └──────────────────┘
```

### 7.3 Performance Comparison

**Incremental build benchmark**（假设）：

| Build System | 30k files, no changes | 30k files, 1 changed |
|--------------|----------------------|---------------------|
| Make (serial) | 5s | 5s |
| Make (-j8) | 1s | 1s |
| Ninja | 0.05s | 0.05s + compile time |

**关键差异**：Ninja 的 "do nothing" 时间极短。

---

## 8. 实际使用示例

### 8.1 build.ninja 示例

```ninja
# Variables
cc = gcc
cflags = -Wall -O2

# Rule definition
rule cc
  command = $cc $cflags -c $in -o $out
  description = CC $out

# Build edges
build foo.o: cc foo.c
build bar.o: cc bar.c
build main.o: cc main.c

build myapp: link foo.o bar.o main.o
```

### 8.2 Deps Log 格式

Ninja 的 `.ninja_deps` 文件使用二进制格式：

```
# Header
# magic number: "# ninjadeps\n"
# version: 4

# Records
# Each record: [path hash] [num deps] [dep hashes...]
```

**为什么用 hash？**
- 文件路径可能很长（100+ 字符）
- Hash 只有 4 bytes
- Trade-off：可能有 collision，但在实践中概率极低

---

## 9. Windows 支持

文章提到 Windows 支持是 "a big hassle"。

### 9.1 主要挑战

| Issue | Linux | Windows |
|-------|-------|---------|
| Process spawning | `fork()` + `exec()` | `CreateProcess()` |
| File timestamps | `stat()` fast, kernel cached | `GetFileAttributesEx()` slow |
| Path separators | `/` | `\` (but also accepts `/`) |
| Case sensitivity | Yes | No |

### 9.2 Timestamp 问题

**Linux**：
```c
stat("/path/to/file", &st);
st.st_mtime;  // Cached in kernel, ~1-2μs
```

**Windows**：
```c
GetFileAttributesEx("C:\\path\\to\\file", 
                    GetFileExInfoStandard, 
                    &info);
info.ftLastWriteTime;  // May hit disk, much slower
```

### 9.3 Windows 开发者的需求

文章指出：
> When someone makes a neat tool for Linux, the impulse is to share it. When they do so for Windows, the impulse is to sell it.

结果：Windows 上 free tools 很少，Ninja 填补了一个空白。

---

## 10. 开源项目的启示

文章最后部分是作者对开源维护的反思，非常有价值。

### 10.1 The Hard Parts of Open Source

**作者的感受**：
- 偶尔有用户表示感谢
- 更多是用户提出 feature requests
- 有些用户会愤怒地要求你做某事
- 很少有人说 "thanks"

**用户类型**：

| Type | Behavior | Motivation |
|------|----------|------------|
| Friendly user | Politely requests features | Wants to improve the tool |
| Entitled user | Demands features, threatens to fork | Treats you as unpaid customer |
| Contributor | Submits PRs, often incompatible with design | Wants to contribute |

### 10.2 设计冲突

当 contributor 提交了一个与设计目标冲突的 PR：

```
Contributor: I added feature X!
Maintainer: Thanks, but X doesn't fit Ninja's design.
Contributor: But I spent so much time on it!
Maintainer: (Spends hours explaining design rationale)
Contributor: (Still upset)
```

**结论**：拒绝贡献本身就是一种 labor。

### 10.3 Free Software 的变化

**过去**：
> Free software is about sharing between equals.

**现在**：
> People think of themselves as customers and treat authors as if they can go complain to the manager.

### 10.4 真正的动力

作者最后说：
> I am motivated by just trying to impress or live up to the ~ten hackers that I admire.

**启示**：
- 成功不一定带来快乐
- 有时 "succeeding" 是一种 burden
- 你可能从一个小得多的成功中学到同样多的东西

---

## 11. 理论背景：Incremental Computation

文章提到 Ninja 的设计是 incremental computation 的一个实例。

### 11.1 Incremental Computation 的通用形式

给定：
- 输入 $I$
- 计算 $f(I) = O$
- 输入变化 $\Delta I$

目标：高效计算 $f(I + \Delta I)$ 而不是从头计算。

**Build system 是特例**：
- $I$ = 所有源文件的状态
- $f$ = 编译过程
- $O$ = 构建产物
- $\Delta I$ = 一个文件的修改

### 11.2 Self-Adjusting Computation

研究论文参考：
- "Build Systems à la Carte" (Mokhov et al., 2018) - https://dl.acm.org/doi/10.1145/3236774
- Self-adjusting computation (ACAR et al.) - https://www.cs.cmu.edu/~rwh/papers/toplas-sac/paper.pdf

**核心概念**：
```
Change propagation:
  Input change → Identify affected computations → Re-execute only those
```

### 11.3 与 React 的联系

文章提到这与 modern UI frameworks (React) 有联系：

```javascript
// React
const Component = ({ data }) => {
  return <div>{data.name}</div>;
};

// When data changes, React re-renders only affected components
```

这与 Ninja 的 "minimal rebuild" 是同样的思想。

---

## 12. Ninja 的局限性

### 12.1 Scale 问题

文章提到：
> Ninja was designed around Chrome's build (~30k steps). These days it's used in Android builds where it is failing to scale.

**原因**：
- Android 可能有 100k+ build steps
- Ninja 的某些操作是 $O(n)$ 或 $O(n \log n)$
- 在足够大的规模下，任何常数因子都 become significant

### 12.2 Small Projects 不需要 Ninja

Ninja 手册第二段：
> "If your project is small, Ninja's speed impact is likely unnoticeable."

但 "fast" sells，很多小项目用户尝试 Ninja，然后抱怨 lack of features。

### 12.3 Generator 的复杂性

Ninja 把复杂性推给了 generator：
- CMake + Ninja 需要理解两个系统
- Debug build issues 需要看两层
- Generator 生成 `.ninja` 文件的过程可能很慢

---

## 13. 总结：关键 Lessons

### 13.1 技术层面

1. **Graph representation matters**：Bipartite graph (files ↔ commands) 比 file-to-file graph 更准确
2. **Cache at the right level**：Kernel 已经 cache 了 file status，不需要 userland cache
3. **Optimize for the common case**：Incremental builds 比 clean builds 更重要
4. **Defaults dominate**：并行执行的默认值比任何优化都重要

### 13.2 架构层面

1. **Separation of concerns**：Generator (high-level) + Ninja (low-level) 是强大的组合
2. **Snapshot the action graph**：强制用户 cache computation graph
3. **Make costs visible**：Slow operations 应该在 generator 中显式执行

### 13.3 社会层面

1. **Code < Architecture < Social issues**：技术只是成功的一小部分
2. **Open source is hard**：Maintainer burnout 是真实的问题
3. **Success can be a burden**：更小的成功可能带来更多的学习

---

## 14. 参考资料

1. **Ninja 官方网站**：https://ninja-build.org/
2. **Ninja GitHub**：https://github.com/ninja-build/ninja
3. **"The Performance of Open Source Software" - Ninja chapter**：https://aosabook.org/en/posa/ninja.html
4. **"Build Systems à la Carte" paper**：https://dl.acm.org/doi/10.1145/3236774
5. **"The Hard Parts of Open Source" talk**：https://www.youtube.com/watch?v=Pm8da4UBv-Q
6. **Evan Martin's blog**：https://neugierig.org/software/blog/
7. **CMake + Ninja integration**：https://cmake.org/cmake/help/latest/generator/Ninja.html
8. **Meson build system**：https://mesonbuild.com/

---

## 15. 最终思考

Ninja 的成功不是偶然的。它体现了几个深刻的设计原则：

1. **Do one thing well**：Ninja 只执行 build graph，不做其他
2. **Make the right thing easy**：并行执行是默认的
3. **Separate concerns**：Generator 和 executor 分离
4. **Optimize for developer experience**：快速反馈，简洁输出

最打动我的是作者的这句话：

> We talk about programming like it is about writing code, but the code ends up being less important than the architecture, and the architecture ends up being less important than social issues.

这不仅适用于 build systems，也适用于所有软件项目。

# Ninja vs. Bazel 深度对比

这是一个非常好的问题。Ninja 和 Bazel 代表了两种截然不同的 build system 设计哲学。让我从多个维度深入分析。

---

## 1. 定位与设计哲学

### 1.1 核心定位

| Aspect | Ninja | Bazel |
|--------|-------|-------|
| **定位** | Low-level build executor | Full-featured build system |
| **类比** | Assembler | Compiler |
| **Scope** | 只执行 build graph | 完整的 build ecosystem |
| **Philosophy** | "Do one thing fast" | "Correct, reproducible, hermetic" |
| **作者** | Evan Martin (Google) | Google (内部工具 Blaze 开源) |

### 1.2 设计目标对比

**Ninja 的目标**：
$$
\min(T_{\text{incremental}}) \quad \text{s.t.} \quad \text{correctness}
$$

**Bazel 的目标**：
$$
\begin{aligned}
\max(&\text{correctness}, \\
    &\text{reproducibility}, \\
    &\text{hermeticity}, \\
    &\text{scalability})
\end{aligned}
$$

### 1.3 Spectrum 上的位置

```
Low-level                              High-level
    │                                      │
    │   Ninja    Make    CMake    Bazel    │
    │     │       │        │        │      │
    │     │       │        │        │      │
    │   Executor  │    Generator   Full    │
    │             │                 System │
    └──────────────────────────────────────┘
```

---

## 2. 架构深度对比

### 2.1 Bazel 的多层架构

Bazel 有更复杂的分层：

```
┌─────────────────────────────────────────────────────────────┐
│                    BUILD Files (Starlark)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  cc_library(name = "foo", srcs = ["foo.cc"], ...)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Analysis Phase                          │   │
│  │  - Parse BUILD files                                 │   │
│  │  - Resolve dependencies                              │   │
│  │  - Create Action Graph (targets → actions)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Execution Phase                         │   │
│  │  - Check cache                                       │   │
│  │  - Run actions (locally or distributed)             │   │
│  │  - Manage sandboxing                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Ninja 的单层架构

```
┌──────────────────┐       ┌────────────────────────────┐
│    Generator     │       │         Ninja              │
│  (CMake/Meson)   │       │  ┌──────────────────────┐  │
│                  │       │  │   Parse .ninja file  │  │
│  - Analyze       │──────→│  │   Check timestamps   │  │
│  - Configure     │ .ninja│  │   Execute commands   │  │
│  - Generate      │ file  │  └──────────────────────┘  │
└──────────────────┘       └────────────────────────────┘
```

### 2.3 关键架构差异

| Feature | Ninja | Bazel |
|---------|-------|-------|
| **Layer count** | 1 (execution only) | 3+ (loading, analysis, execution) |
| **Language** | Declarative `.ninja` | Starlark (Python dialect) |
| **Dependency analysis** | External (generator) | Internal (analysis phase) |
| **Sandboxing** | ❌ | ✅ Mandatory |
| **Remote execution** | ❌ | ✅ Built-in |
| **Caching** | Simple deps log | Content-addressable cache |

---

## 3. 性能深度分析

### 3.1 Startup Latency

这是文章中特别强调的点：

**Ninja**：
```bash
$ time ninja -h
real    0m0.010s   # ~10ms
```

**Bazel**：
```bash
$ time bazel help
real    0m2.500s   # ~2.5s (JVM startup + loading)
```

**原因**：
- Ninja: 原生 binary，C++ 编写，instant startup
- Bazel: Java 程序，JVM startup overhead

### 3.2 Incremental Build Performance

**场景**：修改一个文件后重新构建

| System | Operation | Time Complexity | Typical Time |
|--------|-----------|-----------------|--------------|
| **Ninja** | Parse graph | $O(n)$ | ~50ms |
| | Check timestamps | $O(n)$ | ~50ms |
| | Run commands | $O(1)$ commands | Varies |
| **Bazel** | Load Starlark | $O(n)$ | ~1-5s |
| | Analysis phase | $O(n + m)$ | ~2-10s |
| | Check cache | $O(1)$ per action | ~100ms |
| | Run commands | $O(1)$ commands | Varies |

**关键差异**：

Ninja 的 incremental build：
$$
T_{\text{ninja}} \approx 100\text{ms} + T_{\text{compile}}
$$

Bazel 的 incremental build：
$$
T_{\text{bazel}} \approx 3\text{-}15\text{s} + T_{\text{compile}}
$$

### 3.3 为什么 Bazel 慢？

Bazel 在每次构建时都要重新执行 **Analysis Phase**：

```python
# BUILD file
cc_library(
    name = "foo",
    srcs = glob(["src/**/*.cc"]),  # 需要重新 evaluate
    deps = [":bar"],
)
```

**Analysis Phase 做什么**：
1. Parse all BUILD files
2. Load and evaluate Starlark code
3. Resolve all dependencies
4. Create action graph

即使什么都没变，这个过程也要重复。

### 3.4 Bazel 的优化：Skylark Loading Cache

Bazel 有一些优化来减少重复分析：

```
┌─────────────────────────────────────────┐
│       Skyframe (Evaluation Model)       │
│                                         │
│  ┌─────────┐    ┌─────────┐            │
│  │  SkyKey │───→│SkyValue │ (cached)   │
│  └─────────┘    └─────────┘            │
│                                         │
│  - Each BUILD file is a SkyKey         │
│  - Parsed result is a SkyValue         │
│  - Invalidated only when file changes  │
└─────────────────────────────────────────┘
```

但即使有 cache，overhead 仍然显著。

### 3.5 Clean Build Performance

**场景**：从头构建整个项目

| System | Advantages |
|--------|------------|
| **Ninja** | 简单，低 overhead |
| **Bazel** | Distributed execution, remote cache |

Bazel 在 clean build 时有优势，因为可以：
- Distribute work across 100s of workers
- Share cache across team members
- Parallelize at massive scale

---

## 4. Hermetic Builds（密封构建）

这是 Bazel 的核心特性，Ninja 完全没有。

### 4.1 什么是 Hermetic Build？

**定义**：构建过程完全自包含，不依赖于本地环境。

**形式化**：
$$
\text{Build}(S) = \text{Output} \quad \forall \text{ machine } M
$$

其中 $S$ 是源代码 + 工具链，$M$ 是任意机器。

### 4.2 Bazel 的实现

```python
# WORKSPACE file
workspace(name = "my_project")

# 声明所有外部依赖的精确版本
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/20230125.0.tar.gz"],
    sha256 = "3e6e15...8d5e",  # Content hash
)

# 工具链也是声明式的
register_toolchains(
    "@bazel_tools//tools/cpp:clang-toolchain",
)
```

**效果**：
- 所有人用同一版本的编译器
- 所有人用同一版本的依赖
- 构建结果可重复

### 4.3 Sandbox 隔离

Bazel 使用沙箱执行每个 action：

```bash
# Bazel 创建临时沙箱
/tmp/bazel-sandbox-12345/
├── inputs/        # 只有声明的 inputs 可见
│   ├── foo.cc
│   └── bar.h
├── outputs/       # 写入 outputs
└── tools/         # 声明的工具
```

**安全保证**：
- Action 无法读取未声明的文件
- Action 无法写入未声明的位置
- 隔离了环境变量、网络访问等

### 4.4 Ninja 的问题

Ninja 完全信任环境：

```ninja
rule cc
  command = gcc -c $in -o $out
```

**问题**：
- `gcc` 可能是 `/usr/bin/gcc`（不同机器不同版本）
- 可能隐式读取 `/usr/include/...`
- 可能受到环境变量影响

---

## 5. Content-Addressable Storage

### 5.1 Bazel 的 CAS (Content-Addressable Storage)

Bazel 使用内容寻址存储：

```
Action Digest = hash(action + inputs)

┌──────────────────────────────────────────┐
│              Action Cache                │
│                                          │
│  hash(a1) → [output_hash1, output_hash2]│
│  hash(a2) → [output_hash3]              │
│                                          │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│         Content-Addressable Store        │
│                                          │
│  output_hash1 → binary blob 1           │
│  output_hash2 → binary blob 2           │
│  output_hash3 → binary blob 3           │
│                                          │
└──────────────────────────────────────────┘
```

**优势**：
1. **Global deduplication**：相同输入产生相同输出，只存储一次
2. **Remote caching**：团队成员共享构建产物
3. **Instant rebuild**：如果 output 已在 cache，直接下载

### 5.2 Ninja 的简单 Cache

Ninja 只有简单的 deps log：

```
.ninja_deps (binary format)
├── Recorded dependencies for each output
└── Used to detect header changes
```

**局限**：
- No remote cache
- No content-addressing
- No deduplication

### 5.3 数学对比

**Ninja cache hit condition**：
$$
\text{hit} \iff \forall f \in \text{inputs}: \text{mtime}(f) \text{ unchanged}
$$

**Bazel cache hit condition**：
$$
\text{hit} \iff \text{hash}(\text{action} + \text{inputs}) \in \text{CAS}
$$

**关键差异**：
- Ninja 基于 **timestamp**（可能不准确）
- Bazel 基于 **content hash**（数学上精确）

---

## 6. Remote Execution（分布式执行）

### 6.1 Bazel 的分布式执行架构

```
┌─────────────┐
│   Client    │
│  (Developer)│
└──────┬──────┘
       │ gRPC
       ▼
┌─────────────────────────────────────────┐
│           Remote Execution API          │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │        Scheduler                │   │
│  │  - Distribute actions          │   │
│  │  - Load balance               │   │
│  └─────────────────────────────────┘   │
│              │                          │
│              ▼                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Worker 1 │ │Worker 2 │ │Worker N │   │
│  │(Linux)  │ │(Linux)  │ │(macOS)  │   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Remote Cache (CAS)              │
│  - Shared across all workers            │
│  - Shared across all users              │
└─────────────────────────────────────────┘
```

### 6.2 使用场景

**场景**：你在 MacBook 上开发，但需要构建 Linux 二进制。

**Bazel**：
```bash
$ bazel build //:myapp --platforms=@io_bazel_rules_go//go/toolchain:linux_amd64
```

Bazel 自动将 Linux 编译任务发送到 Linux workers。

**Ninja**：无法原生支持，需要配置 cross-compilation toolchain。

### 6.3 规模优势

| Project Size | Ninja | Bazel |
|--------------|-------|-------|
| Small (<10k files) | ✅ Fast local | ⚠️ Overhead dominates |
| Medium (10k-100k files) | ✅ Good | ✅ Good with cache |
| Large (>100k files) | ⚠️ May struggle | ✅ Designed for this |
| Massive (>1M files) | ❌ Not designed | ✅ Excels with RBE |

---

## 7. 语言与表达能力

### 7.1 Ninja Language

极其简单，几乎只是数据：

```ninja
# Variables
cc = gcc
cflags = -Wall

# Rules
rule cc
  command = $cc $cflags -c $in -o $out

# Build statements
build foo.o: cc foo.c
build bar.o: cc bar.c
```

**限制**：
- No loops
- No conditionals (almost)
- No functions
- No globbing

### 7.2 Bazel Language (Starlark)

完整的编程语言（Python 方言）：

```python
# BUILD file

# 函数和宏
def my_cc_library(name, srcs, **kwargs):
    native.cc_library(
        name = name,
        srcs = srcs,
        copts = ["-Wall"] + kwargs.pop("copts", []),
        **kwargs
    )

# 条件逻辑
selects = select({
    "//conditions:default": ["default_flag"],
    "//platforms:linux": ["linux_flag"],
    "//platforms:macos": ["macos_flag"],
})

# 循环和列表推导
srcs = glob(["src/**/*.cc"], exclude=["*_test.cc"])
test_srcs = [s for s in glob(["**/*_test.cc"])]

# 完整的 cc_library
cc_library(
    name = "mylib",
    srcs = srcs,
    hdrs = glob(["include/**/*.h"]),
    defines = selects,
    visibility = ["//visibility:public"],
)
```

### 7.3 表达能力对比

| Feature | Ninja | Bazel |
|---------|-------|-------|
| Variables | ✅ | ✅ |
| Conditionals | ❌ | ✅ `select()` |
| Loops | ❌ | ✅ |
| Functions/Macros | ❌ | ✅ |
| Globbing | ❌ | ✅ `glob()` |
| Platform logic | ❌ | ✅ `platforms`, `select()` |
| Custom rules | ❌ (in generator) | ✅ Starlark rules |

---

## 8. 依赖管理

### 8.1 Bazel 的依赖系统

**WORKSPACE 文件**：
```python
# WORKSPACE
workspace(name = "my_project")

# 外部依赖
http_archive(
    name = "com_github_gflags_gflags",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
    sha256 = "34af2f15...7c0",
    strip_prefix = "gflags-2.2.2",
)

# Go modules
go_repository(
    name = "com_github_golang_protobuf",
    importpath = "github.com/golang/protobuf",
    tag = "v1.5.2",
)
```

**依赖解析**：
```
WORKSPACE → Bzlmod → Download → Register
                ↓
            Resolve transitive deps
                ↓
            Create @workspace// targets
```

### 8.2 Ninja 的依赖管理

Ninja **没有**依赖管理，完全依赖 generator：

```python
# CMakeLists.txt
find_package(Protobuf REQUIRED)  # 系统 pkg-config

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
```

### 8.3 对比

| Aspect | Ninja + Generator | Bazel |
|--------|-------------------|-------|
| Dependency declaration | In generator (CMake/Meson) | In WORKSPACE/BUILD |
| Version locking | Generator-specific | Built-in (MODULE.bazel) |
| Transitive deps | Generator handles | Bazel handles |
| Vendor mode | Manual | `bazel vendor` |
| Reproducibility | Depends on generator | Built-in |

---

## 9. 正确性保证

### 9.1 Bazel 的正确性特性

#### (1) Action Graph Immutability

Bazel 的 action graph 一旦生成就不可变：

```python
# Analysis phase 生成 action graph
action_graph = analyze(BUILD_files)

# Execution phase 只执行，不修改
execute(action_graph)
```

这保证了 execution 不会意外改变 graph。

#### (2) Declared Inputs/Outputs

每个 action 必须声明所有 inputs 和 outputs：

```python
ctx.actions.run(
    executable = ctx.executable._compiler,
    inputs = ctx.files.srcs + ctx.files.hdrs,
    outputs = [ctx.outputs.o],
    arguments = [...],
)
```

#### (3) Sandbox Enforcement

在沙箱中，只能访问声明的 inputs：
```c
#include "/etc/passwd"  // 在 Bazel sandbox 中会失败！
```

### 9.2 Ninja 的正确性

Ninja 依赖用户提供正确的信息：

```ninja
build foo.o: cc foo.c
# 如果 foo.c 包含 bar.h，但没有声明依赖
# Ninja 无法检测，可能产生错误结果
```

**常见错误**：
1. **Underspecified dependencies**：缺少 header 依赖
2. **Multiple outputs**：一个命令产生多个文件，但只声明了一个
3. **Implicit inputs**：工具的版本变化

### 9.3 正确性对比

| Correctness Aspect | Ninja | Bazel |
|-------------------|-------|-------|
| Underspecified deps | ❌ No detection | ✅ Sandbox catches |
| Implicit inputs | ❌ Trusts user | ✅ Must declare |
| Environment changes | ❌ Not tracked | ✅ Hermetic |
| Toolchain changes | ❌ Not tracked | ✅ Part of action |
| Race conditions | ⚠️ Parallel unsafe builds | ✅ Detects conflicts |

---

## 10. 实际使用体验

### 10.1 开发循环

**Ninja 开发循环**：
```
Edit code → ninja → Test → Edit code → ninja → Test
    │          │
    │     100ms latency
    └──────────┘
```

**Bazel 开发循环**：
```
Edit code → bazel build → Test → Edit code → bazel build → Test
    │           │
    │      3-10s latency
    └───────────┘
```

### 10.2 开发者反馈

文章中作者说：

> I strongly believe that iteration time has a huge impact on programmer satisfaction... the difference of 1 second and 4 seconds is critical.

**心理学**：
```
Latency     Developer State
─────────────────────────────
< 1s        Flow state maintained
1-2s        Acceptable pause
2-5s        Context may slip
> 5s        Context switch likely
> 15s       Check email/HN
```

### 10.3 学习曲线

| Aspect | Ninja | Bazel |
|--------|-------|-------|
| Basic usage | Simple | Moderate |
| BUILD file syntax | N/A (generator) | Starlark (Python-like) |
| Advanced features | Limited | Steep learning curve |
| Debugging | Simple | Complex |
| Documentation | Minimal | Extensive |

### 10.4 IDE 集成

**Ninja**：
- CMake/Make generators have good IDE support
- VSCode, CLion, Visual Studio all support CMake

**Bazel**：
- IntelliJ plugin (good)
- VSCode plugin (improving)
- CLion support (limited)

---

## 11. 使用场景推荐

### 11.1 选择 Ninja 的场景

✅ **适合 Ninja**：

1. **Small to medium projects**
   - < 50k source files
   - Fast iteration critical

2. **Local development focus**
   - No distributed build needed
   - Single platform target

3. **Simple build requirements**
   - Standard C/C++ compilation
   - No complex cross-compilation

4. **Existing CMake/Meson ecosystem**
   - Team already familiar with CMake
   - Good IDE support needed

5. **Latency-sensitive development**
   - Game development
   - Rapid prototyping

### 11.2 选择 Bazel 的场景

✅ **适合 Bazel**：

1. **Large-scale monorepos**
   - Google-scale projects
   - Multiple languages in one repo

2. **Distributed build required**
   - Remote build execution
   - Cross-platform compilation

3. **Correctness critical**
   - Release builds must be reproducible
   - Regulatory requirements

4. **Polyglot projects**
   - Java + C++ + Python + Go
   - Complex dependency graphs

5. **Team collaboration**
   - Shared remote cache
   - Consistent build environment

### 11.3 决策树

```
                        开始
                          │
                          ▼
                需要分布式构建？
                    │
            ┌───────┴───────┐
            │               │
           Yes              No
            │               │
            ▼               ▼
         Bazel       项目规模？
                         │
                 ┌───────┴───────┐
                 │               │
              Large           Small
                 │               │
                 ▼               ▼
            需要remote？    快速迭代重要？
                 │               │
         ┌───────┴───────┐   ┌───┴───┐
         │               │   │       │
        Yes             No  Yes      No
         │               │   │       │
         ▼               ▼   ▼       ▼
       Bazel          Either  Ninja  Either
```

---

## 12. 性能基准测试示例

### 12.1 Incremental Build (Single File Change)

假设项目：
- 10,000 C++ files
- 修改 1 个 .cc 文件

| Metric | Ninja | Bazel (local) | Bazel (remote) |
|--------|-------|---------------|----------------|
| Startup | 10ms | 2s | 3s |
| Analysis | N/A | 2s | 2s |
| Dependency check | 50ms | 100ms | 100ms |
| Compile | 1s | 1s | 0.5s (cached) |
| **Total** | **~1.1s** | **~5.1s** | **~5.6s** |

### 12.2 Clean Build

| Metric | Ninja | Bazel (local) | Bazel (remote, 100 workers) |
|--------|-------|---------------|-----------------------------|
| Startup | 10ms | 2s | 3s |
| Analysis | N/A | 5s | 5s |
| Compile | 1000s | 1000s | 10s (distributed) |
| **Total** | **~1000s** | **~1007s** | **~18s** |

### 12.3 Rebuild After git checkout

切换分支后重新构建：

| Scenario | Ninja | Bazel |
|----------|-------|-------|
| Same files, different content | ~2s | ~3s (re-analyze) |
| Many files changed | ~10s | ~15s (re-analyze) |
| Different toolchain | Manual switch | Automatic |

---

## 13. 深入技术细节

### 13.1 Action Graph 生成

**Bazel 的 Action Graph**：

```python
# BUILD file
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
    hdrs = ["lib.h"],
)

# 生成的 actions:
# 1. action: compile lib.cc → lib.o
#    inputs: lib.cc, lib.h, toolchain
#    outputs: lib.o
#
# 2. action: archive lib.o → liblib.a
#    inputs: lib.o, ar tool
#    outputs: liblib.a
```

**Ninja 的 Build Graph**：

```ninja
# build.ninja (generated)
build lib.o: cc lib.cc
  deps = lib.h
build liblib.a: ar lib.o
```

### 13.2 Skyframe 架构

这是 Bazel 的核心数据结构：

```
┌──────────────────────────────────────────────────────┐
│                    Skyframe                          │
│                                                      │
│   SkyKey           SkyValue             Dependencies │
│   ───────          ────────             ─────────── │
│                                                      │
│   BUILD file    →  Parsed BUILD        →   None     │
│   Target "lib"  →  Configured target   →   BUILD    │
│   Action "cc"   →  Action metadata     →   Target   │
│   Output "lib.o"→  File artifact       →   Action  │
│                                                      │
│   ┌─────────────────────────────────────────────┐   │
│   │         Reverse Dependencies (rdeps)        │   │
│   │                                              │   │
│   │   lib.h → [lib.o action, test.o action]    │   │
│   │   (Used for invalidation)                   │   │
│   └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

**Invalidation 算法**：
```
1. File X changed
2. Find all SkyKeys that depend on X (via rdeps)
3. Invalidate those SkyValues
4. Re-evaluate only invalidated nodes
```

### 13.3 Ninja 的 Deps Log 格式

```
Header:
  Magic: "# ninjadeps\n"
  Version: 4

Entry:
  ┌─────────────────────────────────┐
  │ Node count: uint32              │
  │ For each node:                  │
  │   Path hash: uint32             │
  │   Path string: bytes + null     │
  └─────────────────────────────────┘
  ┌─────────────────────────────────┐
  │ Deps entry:                     │
  │   Output path: uint32           │
  │   Timestamp check: bool         │
  │   Deps count: uint32            │
  │   Deps: uint32[]                │
  └─────────────────────────────────┘
```

### 13.4 性能公式

**Ninja incremental build 时间**：
$$
T_{\text{ninja}} = T_{\text{parse}} + T_{\text{stat}} + T_{\text{schedule}} + T_{\text{execute}}
$$

其中：
- $T_{\text{parse}} = O(n)$, n = build statements
- $T_{\text{stat}} = O(m)$, m = files
- $T_{\text{schedule}} = O(k)$, k = dirty nodes
- $T_{\text{execute}} = \sum_{a \in \text{actions}} T_a / p$, p = parallelism

**Bazel incremental build 时间**：
$$
T_{\text{bazel}} = T_{\text{load}} + T_{\text{analyze}} + T_{\text{cache}} + T_{\text{execute}}
$$

其中：
- $T_{\text{load}} = O(b)$, b = BUILD files
- $T_{\text{analyze}} = O(t)$, t = targets
- $T_{\text{cache}} = O(a)$, a = actions
- $T_{\text{execute}}$ same as Ninja (but can be remote)

---

## 14. 迁移成本

### 14.1 从 Make/Ninja 迁移到 Bazel

**难度**：⭐⭐⭐⭐⭐ (Very Hard)

**需要做的事情**：

1. **重写所有构建定义**
   ```
   CMakeLists.txt → BUILD files
   Makefile → BUILD files
   ```

2. **声明所有依赖**
   ```python
   # 必须显式声明所有 header deps
   cc_library(
       name = "lib",
       srcs = ["lib.cc"],
       hdrs = ["lib.h", "internal.h"],  # 所有公开 headers
       deps = ["//other:lib"],           # 所有依赖
   )
   ```

3. **处理隐式依赖**
   - 原来依赖系统 include path 的，现在要显式声明
   - 原来动态加载的库，现在要提前声明

4. **配置 hermetic toolchain**
   ```python
   # 必须使用 Bazel 管理的工具链
   register_toolchains(
       "@bazel_tools//tools/cpp:clang-toolchain",
   )
   ```

**时间估计**：
- Small project (<100 files): 1-2 weeks
- Medium project (100-10k files): 1-3 months
- Large project (>10k files): 6-12 months

### 14.2 从 Bazel 迁移到 Ninja

**难度**：⭐⭐⭐ (Moderate)

**需要做的事情**：

1. **选择 generator**
   ```
   BUILD files → CMakeLists.txt or Meson.build
   ```

2. **转换依赖声明**
   - Bazel 的 `deps` → CMake 的 `target_link_libraries`
   - Bazel 的 `srcs` → CMake 的 `add_library`

3. **处理 Bazel 特有功能**
   - Remote execution → 本地执行或第三方方案
   - Sandboxing → 信任环境
   - Hermetic builds → 依赖系统包管理器

**时间估计**：
- Small project: 2-5 days
- Medium project: 2-4 weeks
- Large project: 1-3 months

---

## 15. 总结对比表

| 维度 | Ninja | Bazel |
|------|-------|-------|
| **定位** | Low-level executor | Full build system |
| **速度** | ⭐⭐⭐⭐⭐ Fast incremental | ⭐⭐⭐ Good at scale |
| **正确性** | ⭐⭐⭐ Trusts user | ⭐⭐⭐⭐⭐ Hermetic |
| **可扩展性** | ⭐⭐⭐ Good to 100k files | ⭐⭐⭐⭐⭐ Designed for millions |
| **分布式** | ❌ | ✅ Built-in RBE |
| **学习曲线** | ⭐ Simple | ⭐⭐⭐⭐ Steep |
| **IDE 支持** | ⭐⭐⭐⭐ Via generators | ⭐⭐⭐ Improving |
| **依赖管理** | ⭐⭐ Via generator | ⭐⭐⭐⭐ Built-in |
| **跨平台** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **开发迭代** | ⭐⭐⭐⭐⭐ <100ms | ⭐⭐ 3-15s |
| **Clean build** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ With RBE |
| **社区** | ⭐⭐⭐⭐ Large | ⭐⭐⭐ Growing |

---

## 16. 最终建议

### 16.1 何时选择 Ninja

```
你的项目满足以下条件：
├── 主要是 C/C++ 代码
├── 少于 50k 源文件
├── 快速迭代是首要需求
├── 不需要分布式构建
├── 团队熟悉 CMake/Meson
└── 本地开发环境一致即可

→ 选择 Ninja + CMake/Meson
```

### 16.2 何时选择 Bazel

```
你的项目满足以下条件：
├── 大规模 monorepo
├── 多语言混合（Java/C++/Python/Go）
├── 需要分布式构建
├── 构建正确性和可重复性至关重要
├── 需要远程缓存和执行
└── 愿意投入学习成本

→ 选择 Bazel
```

### 16.3 混合方案

**方案**：使用 Bazel，但在本地开发时用 Ninja

```bash
# 生成 Ninja 文件用于快速迭代
$ bazel query "//..." --output=build > build.ninja
$ ninja

# 或者使用 Bazel 的 fast build 模式
$ bazel build //:target --strategy=CppCompile=local
```

---

## 参考资料

1. **Bazel 官方文档**：https://bazel.build/
2. **Bazel GitHub**：https://github.com/bazelbuild/bazel
3. **Remote Execution API**：https://github.com/bazelbuild/remote-apis
4. **"Build Systems à la Carte"**：https://dl.acm.org/doi/10.1145/3236774
5. **Bazel Skyframe 论文**：https://blog.bazel.build/2015/06/19/skyframe.html
6. **Hermetic Builds**：https://blog.bazel.build/2016/04/18/hermetic-builds.html
7. **Bazel vs CMake**：https://cmake.org/cmake/help/latest/manual/cmake.1.html
8. **RBE (Remote Build Execution)**：https://cloud.google.com/build/docs/remote-build-execution

---

**总结一句话**：Ninja 和 Bazel 不是竞争关系，而是不同层级的工具。Ninja 追求极致的 incremental build 速度，Bazel 追求大规模、正确性、可重复性。选择哪个取决于你的项目规模、团队需求和可接受的复杂度。