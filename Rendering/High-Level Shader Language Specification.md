---
source_pdf: High-Level Shader Language Specification.pdf
paper_sha256: 4d7c93fa7ecf0d5b41f31251f6c9f68fdda644f5c9485a7000a9060f17a66169
processed_at: '2026-08-04T23:48:44-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HLSL Spec 人话版

好, 我把这份 spec 拆解成几个"人话"角度讲。

## 一句话总结

这份 spec 干的事情就是: **HLSL 这个语言二十多年没正儿八经的文档, 大家都是看 FXC/DXC 两个 compiler 的行为来猜规则, 现在终于有人把它写下来了。**

你把它想象成 C 语言在 1989 年之前的状态——大家都在用, 但没人说得清到底什么是合法的什么是非法的。

---

## 1. 为什么 HLSL 长得像 C++ 又不是 C++

你打开一个 HLSL 文件看, 语法跟 C++ 几乎一模一样, 有 `struct`, 有 template, 有 overload。所以很多人第一反应是"HLSL 是 C++ 子集"。

**这个直觉是错的**, 原因有几点:

- HLSL **没有** `goto`, 没有 lambda, 没有 destructor
- HLSL **有** vector/matrix 作为一等公民, C++ 没有
- HLSL 的 overload resolution ranking 跟 C++ 不一样, 它加了一大堆关于 vector truncation, scalar splat 的 rank
- HLSL 有 swizzle 语法 `v.xyz`, `v.xzyw`, C++ 完全没这东西

所以 spec 里很明确地说: "HLSL is neither a subset nor a superset of C or C++"。

这个"长得像但其实不是"的状态其实是因为 DXC 这个 compiler 是基于 Clang 改的, Clang 是 C++ compiler, 所以 HLSL 顺手继承了一堆 C++ 行为。这份 spec 很大程度上是在 formalize Clang 的行为。

---

## 2. SPMD 是什么, 跟 SIMD/SIMT 有什么区别

这是理解整份 spec 的关键。

GPU 硬件本质上都是 SIMD——一条指令同时处理多个数据。但程序怎么写, 有不同的抽象层次:

- **SIMD**: 程序员手动写 vector 操作, 比如 `float4 a + float4 b`, 一次处理 4 个。
- **SIMT** (NVIDIA 叫法): 程序员写"单 thread"代码, 硬件把 32 个 thread 捆成 warp 一起跑, 看起来像 SIMD 但程序员感觉自己在写串行代码。
- **SPMD** (HLSL 的叫法): 跟 SIMT 本质差不多, 一个 program 在多个 data 上同时跑。

HLSL 选 SPMD 这个词, 是因为它不想绑死硬件实现。Wave 这个概念对应 NVIDIA warp, AMD wavefront, Metal SIMD-group, Vulkan subgroup——硬件 Wave size 可能是 32, 64, 8, 16, 都可以。

**关键的 hierarchy**:

```
Dispatch (整个 GPU 任务, 类似 CUDA grid)
  -> Thread Group (类似 CUDA block)
     -> Wave (硬件一次跑的一组 lane, 类似 warp)
        -> Lane (一个"thread", 处理一个 data element)
           -> Quad (4 个相邻 lane, 给 pixel shader 算 ddx/ddy 用的)
```

**最容易被忽略的一点**: Thread Group size 不一定是 Wave size 的整数倍。如果 Thread Group 大小是 64, Wave size 是 32, 没问题。但如果 Thread Group 是 50, Wave size 是 32, 那么会有 inactive lane——硬件 lane 存在但不参与计算。

---

## 3. Lane 的四种状态——这个非常 practical

spec 定义 lane 有四种状态:

- **active**: 正常的, 参与计算, 写 output
- **helper**: pixel shader 里算 `ddx`/`ddy` 需要的 2x2 pixel quad 中的"陪跑" lane。它会执行整个 shader, 但 output 被丢弃
- **inactive**: 硬件 lane 存在但没活干, 比如 Wave 没填满
- **predicated off**: control flow 分叉, 比如一个 `if` 把一半 lane 关了

**helper lane 这事很 practical**: 假设你画一个三角形, 只覆盖了 2x2 pixel block 里的 1 个 pixel, GPU 仍然要让 4 个 pixel 都跑 shader 来算 derivative。那 3 个"陪跑"的 pixel 就是 helper lane。

这意味着你在 shader 里写的任何"看起来只对当前 pixel 有用"的代码, helper lane 也会跑。如果你在 helper lane 路径上做了很重的计算, 性能就浪费了。所以有些优化技巧是先判断 `IsHelperLane()` 再决定要不要做重活。

---

## 4. Type System 的核心思想: Scalarization

这是 HLSL 设计上最优雅的一点。

形式化定义是 scalarized representation $SR(T)$:

- $SR(T[n]) = SR(T_0), SR(T_1), \ldots, SR(T_{n-1})$
- $SR(\text{vector}\langle T, n\rangle) = T, T, \ldots, T$ (n 个 T)
- $SR(\text{matrix}\langle T, n, m\rangle) = T, T, \ldots, T$ (n×m 个 T)
- $SR(\text{class } T) = SR(T::\text{base}), SR(m_0), \ldots, SR(m_k)$, 其中 $m_i$ 是 non-static member

**人话**: 把任何类型递归拆成一串 scalar, 如果两个类型拆出来的 scalar 序列一样, 它们就是 "scalar-layout-compatible"。

**为什么这事重要**: 因为这意味着 `float4` 跟 `struct { float x,y,z,w; }` 在底层完全一样, 都是 4 个 float。所以你可以把 `float4` 赋值给那个 struct, 反过来也行。

这是 HLSL "一切皆 scalar bag" 哲学的形式化。它让你写 shader 的时候不用太纠结数据怎么 pack, compiler 帮你处理。

---

## 5. Intangible Types: Resource 是 Handle 不是 Value

`Buffer<T>`, `Texture2D<T>`, `SamplerState` 这些 resource type 在 spec 里叫 "intangible types"。

"intangible" 的意思是: 编译器不知道它占多大内存, 没有 object representation。

**人话**: Resource 不是数据, 它是 handle。你不能把一个 `Buffer<float>` 当普通变量复制——你只是"持有"一个 binding 到 GPU register 的引用。

限制:
- 只能声明在 thread address space (不能放 `groupshared` 或 cbuffer)
- 不能作为 entry function 的参数/返回值
- 不能 load/store 到其他 address space

这跟 CUDA 里 `__constant__` 变量不能当 kernel 参数是一回事——resource 是 binding, 不是 value。

---

## 6. Conversion System: 为什么要搞这么复杂

HLSL 的 conversion 比 C++ 复杂得多, 因为它要处理 vector/matrix 这些 C++ 没有的类型。

**核心 4 步 conversion sequence**:

1. lvalue-to-rvalue 或 array-to-pointer (0 或 1 次)
2. integral/float/bool 转换 或 derived-to-base 或 flat conversion (0 或 1 次)
3. splat 或 truncation (0 或 1 次)
4. qualification conversion (0 或 1 次)

关键的 vector-specific 转换:

**Splat** (标量扩展到 vector):
$$T \to \text{vector}\langle T, n\rangle$$
比如 `float x = 1.0f; float4 v = x;` 会把 $x$ 复制 4 份变成 $[x, x, x, x]$。

**Truncation** (vector 缩短):
$$\text{vector}\langle T, m\rangle \to \text{vector}\langle T, n\rangle, \text{iff } n < m$$
比如 `float4 a = ...; float2 b = a;` 取前 2 个 element, 丢掉后 2 个。

**Component-wise** (vector 元素逐个转换):
$$\text{vector}\langle T, n\rangle \to \text{vector}\langle U, n\rangle$$
比如 `int4` 转 `float4`, 每个 int 元素独立转成 float。

**Conversion rank** 是为了 overload resolution 设计的, 从好到坏排 9 级:

| Rank | 名字 | 含义 |
|------|------|------|
| 1 | Exact Match | 完全匹配, 无转换 |
| 2 | Extension | splat + exact match |
| 3 | Promotion | 比如int 到 int64, 无损扩展 |
| 4 | Promotion Extension | splat + promotion |
| 5 | Conversion | 比如int 到 float, 有损 |
| 6 | Conversion Extension | splat + conversion |
| 7 | Truncation | vector 缩短 |
| 8 | Promotion Truncation | promotion + truncation |
| 9 | Conversion Truncation | conversion + truncation |

**人话 ranking 逻辑**:
- 扩展比缩短安全, 所以 splat (扩展) 比 truncation (缩短) 好
- 无损比有损好, 所以 promotion 比 conversion 好
- Truncation 主动丢信息, 所以排最后

比如你调 `f(float4)` vs `f(float2)`, 传 `float4`:
- `f(float4)`: Exact Match, rank 1
- `f(float2)`: Truncation, rank 7
- 选 `f(float4)`, 没歧义

但传 `float`:
- `f(float4)`: Extension, rank 2
- `f(float2)`: Extension, rank 2
- Ambiguous! 编译器报错

这设计背后的直觉是: **HLSL 宁可让你明确写, 也不愿意猜测你到底想 splat 到哪个长度的 vector**。

---

## 7. Swizzle 的设计哲学

Swizzle 是 HLSL 最舒服的特性之一。语法上分 vector 和 matrix 两套。

**Vector swizzle**:
- `v.xyzw` 或 `v.rgba`, 不能混用
- `v.x` (取一个, 返回 scalar)
- `v.xyz` (取三个, 返回 float3)
- `v.xzyw` (重排, 返回 float4)
- `v.xx` (重复, 返回 float2, 但只能读不能写)

**Matrix swizzle** 有两套 indexing:
- `_m00`, `_m01`, ..., `_m33`: 0-indexed (memory-style)
- `_11`, `_12`, ..., `_44`: 1-indexed (math-style)

注意 1-indexed 中 `_23` 表示 row 2, col 3, 不是 row 3, col 2。

**关键 lvalue 规则**: swizzle 表达式是 lvalue 当且仅当原始表达式是 lvalue **且** swizzle sequence 没有重复 component。

```hlsl
float4 v = float4(1,2,3,4);
v.xy = float2(5,6);    // OK
v.xx = float2(5,6);    // 编译错误! xx 有重复, 是 prvalue 不能赋值
```

**为什么**: `v.xx = float2(a, b)` 语义上模糊——是把 $a$ 写到 $v.x$ 还是 $b$?两个都要写的话哪个胜出?HLSL 直接禁止, 避免歧义。

---

## 8. Buffer 类型怎么选

HLSL 有好几种 buffer, 人话区分:

**Typed Buffer** (`Buffer<T>`):
- 最大 16 byte per element
- Load 时有 format conversion, 比如底层是 `R32_UINT`, 你用 `Buffer<float4>` 读取, 会做 reinterpret
- 适合标准 graphics 格式

**Structured Buffer** (`StructuredBuffer<T>`):
- 任意 struct 类型
- 无 format conversion, 直接按 bit reinterpret
- 适合自定义数据结构

**Byte Address Buffer** (`ByteAddressBuffer`):
- 按 byte offset 访问, 必须是 4 的倍数
- `Load`, `Load2`, `Load3`, `Load4` 返回 uint, uint2, uint3, uint4
- 模板版 `Load<T>(offset)` 可以返回任意类型
- 最灵活但最底层

**Append/Consume Structured Buffer**:
- 有 hidden counter, Append 加到末尾, Consume 从末尾取
- 类似 stack 语义

**RWStructuredBuffer** 也有 hidden counter, 通过 `IncrementCounter`/`DecrementCounter` 访问, 适合做 allocation 之类的事。

---

## 9. Constant Buffer 的 Packing 规则

这是 HLSL 最坑的部分之一, 因为它有一套自己的 packing 规则。

核心模型: **constant buffer 看成 16-byte row 数组**, 每个 row 是 4 个 32-bit 元素。

Packing 规则简化版:

1. Scalar 和 vector 尝试 pack 进当前 row 的剩余空间
2. 装不下就开新 row
3. Array element 和 struct 总是开新 row, 每个 element 16-byte 对齐
4. Matrix 每个 row (row-major) 或每个 column (column-major) 开新 row

**例子**:

```hlsl
cbuffer CB {
    float  a;     // row 0, element x
    float3 b;     // row 0, element yzw (跟 a 挤在一起!)
    float2 c;     // row 1, element xy (row 0 满了)
    float  d;     // row 1, element z
    float2 g[2];  // row 2, xy + row 3, xy (array element 每个 16-byte 对齐!)
};
```

注意 `float2 g[2]` 两个 element 各占 16 byte, 中间有 padding。这是 array 的特殊规则, 每个 element 都要 16-byte 对齐。

**`packoffset` 可以手动指定位置**:

```hlsl
cbuffer CB {
    float2 Pos : packoffset(c1.z);  // row 1, element z, byte offset 24
};
```

**Default constant buffer** ($Globals): 全局变量自动进这个 buffer, 除非标了 `static`, `groupshared`, 或是 resource。

这个 packing 规则跟 OpenGL 的 std140 类似但更 aggressive——HLSL 会尽量 pack, std140 更保守。

---

## 10. Resource Binding 的 Register 和 Space

DX12 时代 resource binding 有两个维度: register type 和 space。

**Register types**:

- `t/T`: SRV (read-only texture/buffer), "texture" 缩写
- `u/U`: UAV (writable buffer), "UAV" 缩写
- `b/B`: Constant Buffer
- `s/S`: Sampler
- `c/C`: Default constant buffer layout 标记

**Space**: 不同 space 是独立 namespace, 可以重用 register 号。

```hlsl
Buffer<float> buf1 : register(t0, space0);
Buffer<float> buf2 : register(t0, space1);  // 不冲突, 不同 space
RWBuffer<float> buf3 : register(u0, space0); // 不冲突, 不同 type
```

**Aggregate binding**:

```hlsl
Buffer<float> fbuf[3] : register(t0, space0);  // 占 t0, t1, t2
struct {
    RWBuffer<int> rwbuf;
    Buffer buf;
} sbufs : register(t15) : register(u0);  // t15 和 u0, 不同 type 各占一个
```

这个设计跟 Vulkan 的 descriptor set 几乎一一对应, space0 对应 set=0, register(t0) 对应 binding=0。

---

## 11. `out`/`inout` 参数和 cxvalue

这是 HLSL 独有的设计。

C++ 的 reference (`int&`) 在 HLSL 里没有。HLSL 用 `out` 和 `inout` 关键字:

```hlsl
void Foo(out int x) { x = 42; }
void Bar(inout int x) { x = x * 2; }

int a;
Foo(a);  // a 变成 42
Bar(a);  // a 变成 84
```

**cxvalue** 是 spec 新引入的术语来 formalize 这个行为:

- **xvalue** (expiring value): C++ 里的概念, 生命周期快结束的对象
- **cxvalue** (casted expiring value): HLSL 独有, xvalue 但过期时会把值写回绑定的 lvalue

调用 `Foo(out int x)` 时:
1. 调用前: 创建一个 cxvalue, 不初始化
2. 函数内: cxvalue 作为参数, 函数给它赋值
3. 函数返回: cxvalue 过期, 触发"写回"语义, 把值赋给 argument lvalue

调用 `Bar(inout int x)` 时:
1. 调用前: 创建一个 cxvalue, 从 argument lvalue copy-initialize
2. 函数内: cxvalue 作为参数, 函数读写它
3. 函数返回: cxvalue 过期, 写回 argument lvalue

**Inverted ICS**: 因为 cxvalue 要写回, 所以需要双向转换都成立。argument 到 parameter 要能转, parameter 到 argument 也要能转。

---

## 12. Overload Resolution 的 HLSL 特殊性

除了上面说的 conversion rank, HLSL overload 还有几个特殊点:

**参数修饰符影响 overload**:

```hlsl
void H(int);          // (1)
void H(inout int);    // OK, overload 允许 (in vs inout)
void I(int);
void I(out int);      // OK, in vs out 允许
void J(out int);
void J(inout int);    // 错误, out vs inout 不允许
```

**`in` 是默认的**, 所以 `void H(int)` 和 `void H(in int)` 是同一个函数, 不能 overload。

**Entry function 限制**: 一个 overload set 只能有一个 entry function, 用 `[shader("vertex")]` 等 attribute 标记的。

**类型等价**: `int4` 和 `vector<int, 4>` 是同一个类型, 不能互相 overload。

---

## 13. Spec 还缺什么

从目录看, 很多章节是空的:

- §10 Classes 只讲了 static member 和 conversion, 缺完整 class semantics
- §11 Templates 只有 instantiation 和 partial ordering, 缺 specialization 规则
- §13 Runtime 完全 TODO

这意味着这份 spec 是 "working draft", 还有大量工作要做。

**Aspirational 部分**: spec 说有些地方会 aspirational, 跟两个实现都不同。这说明 spec 不只是描述现状, 也在引导未来行为。

---

## 14. 对比其他 GPU 语言

**CUDA C++**: 跟 HLSL 最像, 但:
- CUDA 是 SIMT, HLSL 是 SPMD (本质上差不多)
- CUDA 没有 swizzle 一等公民
- CUDA 的 vector 是 library 实现, HLSL 是语言原生
- CUDA 用 `__shared__`, `__constant__`, HLSL 用 `groupshared`, `cbuffer`

**Metal MSL**: C++14 子集 + address space qualifier, 没有 swizzle 一等公民, 没有 vector type 作为语言原生概念。

**WGSL**: 吸取了 HLSL 教训, 没有 implicit conversion, 强制 explicit binding, 更严格更安全但更繁琐。

---

## 15. 我觉得最 elegant 的设计

**Scalarization 哲学**: 把所有类型最终看作 scalar 序列, 简化类型系统。

**SPMD 抽象 + Wave intrinsics 分层**: 大多数代码用 lane 抽象写, 性能关键部分用 wave intrinsics 手动优化。这是现代 GPU 编程的精髓。

**Conversion rank 系统**: 虽然复杂但逻辑自洽, 优先级清晰——扩展比缩短安全, 无损比有损好。

---

## 16. 我觉得最痛苦的设计

**Constant buffer packing**: 那套 16-byte row packing 规则太复杂, 容易出错。手动算 offset 很痛苦, 所以有了 `packoffset`。

**Conversion rank 9 级**: 太多层级, 写 generic 代码时容易踩坑。比如传 `float` 给 `f(float4)` 和 `f(float2)` 会 ambiguous, 需要显式 cast。

**`out`/`inout` 不能跟 C++ reference 互转**: 让 HLSL 跟 C++ 互操作时很别扭。

---

## 参考链接

- [DirectX Shader Compiler (DXC) GitHub](https://github.com/microsoft/DirectXShaderCompiler)
- [HLSL 官方文档](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl)
- [DirectX 12 Specs](https://microsoft.github.io/DirectX-Specs/)
- [DXIL Spec](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst)
- [SPIR-V Spec](https://www.khronos.org/registry/SPIR-V/)
- [HLSL Packing Rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules)
- [HLSL Swizzle](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math)
- [Wave Intrinsics SM 6.0](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [Vulkan GLSL vs HLSL (glslang)](https://github.com/KhronosGroup/glslang)
- [HLSL Atomic Operations](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/interlockedadd)

---

# HLSL Specification 深度解读

这是一份具有里程碑意义的文档——HLSL 在诞生二十余年后终于有了第一份正式的语言规范。让我系统性地梳理这份 spec 的核心内容，并补充大量相关联想和技术细节。

## 1. 历史背景与文档定位

### 1.1 这份 spec 的历史意义

HLSL 自 DirectX 9 时代（2002 年左右）就存在了，但其行为一直由两个 reference implementation 隐式定义：
- **FXC** (Legacy DirectX Shader Compiler)：老代码库，从 DX9 起使用
- **DXC** (DirectX Shader Compiler)：基于 Clang/LLVM，DX12 起的主力

DXC 基于 Clang 这点极其重要——它意味着 HLSL 的很多 corner case 行为继承自 Clang 的 C++ 实现，而非任何 formal 定义。这份 spec 偏向 DXC 行为，但在极少数情况下会 aspirational（与两个实现都不同）。

参考链接：
- [DirectX Shader Compiler (DXC) GitHub](https://github.com/microsoft/DirectXShaderCompiler)
- [HLSL 官方文档](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl)

### 1.2 与 C/C++ 的关系

spec 明确说 HLSL "draws heavy inspiration from ISO C standard (2011) and ISO C++ standard (2011)"，但又说 "HLSL is neither a subset nor a superset of C or C++"。

这点很关键——很多人误以为 HLSL 是 C++ 子集，实际上：
- HLSL 没有 `goto`、`lambda`、`destructor`
- HLSL 有 vector/matrix 一等公民、swizzle、resource types 等 C++ 没有的概念
- HLSL 的 overload resolution ranking 与 C++ 不同（加入了 vector truncation/splat 的 rank）

## 2. SPMD 执行模型——HLSL 的灵魂

### 2.1 术语层次结构

HLSL 用一套与 CUDA/Metal 略不同的术语：

```
Dispatch (类似 CUDA grid)
  └── Thread Group (类似 CUDA block / Metal threadgroup)
        └── Wave (类似 CUDA warp / Metal SIMD-group / Vulkan subgroup)
              └── Lane (类似 CUDA thread)
                    └── Quad (4 个相邻 lane, 用于 pixel shader derivatives)
```

关键insight：**Wave 是硬件相关的 power-of-2 size**，而 Thread Group 是编程模型概念，独立于硬件 Wave size。当 Thread Group size 不是 Wave size 的整数倍时，剩余的 lane 是 inactive。

### 2.2 Lane States——一个常被忽略的关键概念

spec 定义了四种 lane state：

| State | 何时出现 | 是否执行 | 是否写 output |
|-------|---------|---------|-------------|
| **active** | 正常 launch 条件 | Yes | Yes |
| **helper** | pixel shader 中为 derivative 计算 2x2 quad | Yes | No (output discarded) |
| **inactive** | 输入不足以填满 Wave / 减少 register pressure | No | No |
| **predicated off** | control flow divergence | 临时 No | No |

**Helper lane 的存在原因**：GPU 计算 `ddx`/`ddy`（屏幕空间导数）需要 2x2 pixel quad。如果 quad 中只有 1 个 pixel 实际需要渲染，其他 3 个仍然要执行整个 shader，但 output 丢弃。这对性能优化很重要——避免在 helper lane 路径上做重计算。

参考：[GPU Pro 中关于 helper lane 的讨论](https://www.realtimerendering.com/)

### 2.3 SPMD vs SIMT vs SIMD

spec 明确区分：
- **SIMD** (Single Instruction Multiple Data)：一条指令处理多个数据元素
- **SIMT** (Single Instruction Multiple Thread)：NVIDIA 的 warp 模型，threads lockstep 执行
- **SPMD** (Single Program Multiple Data)：编程模型，一个 program 在多个数据元素上执行

HLSL 是 SPMD 编程模型，硬件可能用 SIMD 或 SIMT 实现。Wave operations（如 `WaveActiveSum`）要求 lane 同步，这是编程契约。

## 3. Memory Model

### 3.1 四个 Address Space

| Memory Space | 关键字 | 可见性 | 类比 CUDA |
|-------------|-------|--------|----------|
| **Thread** | (默认) | 单 lane | local memory |
| **Thread Group** | `groupshared` | Thread Group 内 | shared memory |
| **Device** | (resources) | 全 device | global memory |
| **Constant** | `cbuffer` | 全 device, 只读 | constant memory |

### 3.2 历史 alignment 演进

- **SM 5.0 及之前**：128-bit slot aligned (4×32-bit vector 优化)
- **SM 5.1**：32-bit multiple
- **SM 6.0**：16-bit multiple (支持 half 精度高效访问)

这反映了 GPU 从早期 fixed-function 4-component vector 硬件到现代 scalar 架构的演进。早期 GPU（如 ATI R300, NV30）的 register file 和 ALU 真的是 4-component SIMD，现在则是 scalar SIMD。

参考：[DirectX 12 spec](https://microsoft.github.io/DirectX-Specs/)

## 4. Type System 与 Scalarization

### 4.1 Arithmetic Types

HLSL 的类型层次比 C++ 丰富：

**Integer types**:
- Standard: `int16_t`, `int32_t`, `int64_t` (signed); `uint16_t`, `uint32_t`, `uint64_t` (unsigned)
- Aliases: `int` = `int32_t`, `uint` = `uint32_t`
- Minimum precision: `min16int`, `min16uint` (object rep 是 int/uint, 但值表示最少 16 bit)

**Floating point types**:
- Standard: `half` (16 or 32 bit, impl-defined), `float` (32 bit, IEEE 754), `double` (64 bit, IEEE 754)
- Minimum precision: `min16float` (object rep 是 float, 值表示最少 16 bit)

`min16*` 类型的设计目的：让 compiler 可以在支持 16-bit 的硬件上用更窄的运算，但保持 32-bit 硬件的兼容性。

### 4.2 Scalarized Type Compatibility——核心概念

这是 HLSL 类型系统最优雅的部分。定义 scalarized representation $SR(T)$：

$$SR(T[n]) = SR(T_0), SR(T_1), \ldots, SR(T_{n-1})$$

其中 $T_i$ 表示 array $T$ 的第 $i$ 个 element。

$$SR(\text{vector}\langle T, n\rangle) = T_0, T_1, \ldots, T_{n-1}$$

$$SR(\text{matrix}\langle T, n, m\rangle) = T_0, T_1, \ldots, T_{n \times m - 1}$$

$$SR(T_{\text{class}}) = SR(T::\text{base}), SR(T::0), \ldots, SR(T::k)$$

其中 $T::\text{base}$ 是 base class（若有），$T::i$ 是第 $i$ 个 non-static member。

两个类型 $\text{cv}_1 T_1$ 和 $\text{cv}_2 T_2$ 是 **scalar-layout-compatible** 当且仅当 $T_1 = T_2$ 或 $SR(T_1) = SR(T_2)$（序列相同）。

**Intuition**：这解释了为什么 `float4` 可以赋值给 `struct { float x, y, z, w; }`——它们的 scalarized representation 都是 `float, float, float, float`。这是 HLSL "everything is a bag of scalars" 哲学的形式化。

### 4.3 Intangible Types——Resource 是 Handle 不是 Value

spec 第 12 节定义的 intangible types 是没有定义 object representation 的类型，size 在 compile time 未知。典型例子：`Buffer<T>`, `Texture2D<T>`, `SamplerState` 等 resource types。

使用限制：
- 只能在 Thread address space 声明（不能放 `groupshared` 或 cbuffer）
- 不能 load/store 到其他 address space
- 不能是 program/external linkage 函数的参数或返回类型

**Intuition**：Resources 本质上是"opaque handle"，编译器和运行时知道如何绑定到硬件 register，但你不能把它当普通数据 copy。这类似 CUDA 中 `__constant__` 变量不能作为 kernel 参数传递的 design。

## 5. Conversion System——HLSL 最复杂的部分

### 5.1 Standard Conversion Sequence 的 4 步

```
Step 1: Lvalue-to-rvalue OR Array-to-pointer (0 or 1)
Step 2: Integral/Float/Bool conversion OR derived-to-base OR flat conversion (0 or 1)
Step 3: Splat OR Truncation (0 or 1)
Step 4: Qualification conversion (0 or 1)
```

### 5.2 Vector/Matrix 特有转换

**Vector splat** (§4.10): 
$$T \to \text{vector}\langle T, n\rangle$$
标量 $v$ 复制到 vector 每个元素：$[v, v, \ldots, v]$

**Matrix splat** (§4.11):
$$T \to \text{matrix}\langle T, n, m\rangle$$
类似地，标量填充整个 matrix。

**Vector truncation** (§4.13):
$$\text{vector}\langle T, x\rangle \to \text{vector}\langle T, y\rangle \text{ iff } y < x$$
保留 $[0..y)$，丢弃 $[y..x)$。

**Matrix truncation**：每个维度独立 truncation。
$$\text{matrix}\langle T, x, y\rangle \to \text{matrix}\langle T, z, w\rangle \text{ iff } x \geq z \text{ and } y \geq w$$
保留 vectors $[0..z)$，每个 vector 保留 elements $[0..w)$。

**Component-wise conversion** (§4.14): 
$$\text{vector}\langle T, n\rangle \to \text{vector}\langle V, n\rangle$$
逐元素 $T \to V$ 转换。

### 5.3 Conversion Rank——Overload Resolution 的核心

HLSL 定义了 9 个 rank 级别（从好到差）：

| Rank | 描述 |
|------|------|
| 1. Exact Match | 无转换 |
| 2. Extension | scalar splat + exact match |
| 3. Promotion | integral/float promotion |
| 4. Promotion Extension | scalar splat + promotion |
| 5. Conversion | integral/float conversion |
| 6. Conversion Extension | scalar splat + conversion |
| 7. Truncation | vector/matrix truncation |
| 8. Promotion Truncation | promotion + truncation |
| 9. Conversion Truncation | conversion + truncation |

**Intuition**：这个 ranking 体现了 HLSL 的设计哲学：
- Splat 比 truncation 好（扩展比丢失信息安全）
- Promotion 比 conversion 好（无损比有损好）
- Truncation 是最差的（主动丢弃信息）

例如调用 `f(float4)` vs `f(float2)`，传 `float4`：
- `f(float4)`: Exact Match (rank 1)
- `f(float2)`: Truncation (rank 7) → 选 `f(float4)`

传 `float`：
- `f(float4)`: Extension (rank 2)
- `f(float2)`: Extension (rank 2) → ambiguous! 

## 6. Swizzle 语义

### 6.1 Vector Swizzle

两种命名：`xyzw` 或 `rgba`，不能混用。映射：

| Element Index | xyzw | rgba |
|--------------|------|------|
| 0 | x | r |
| 1 | y | g |
| 2 | z | b |
| 3 | w | a |

### 6.2 Matrix Swizzle——两种 indexing 风格

- `_m00`, `_m01`, ..., `_m33`：0-indexed (memory-style)
- `_11`, `_12`, ..., `_44`：1-indexed (math-style)

注意 1-indexed 中 `_23` 表示 row 2, col 3（数学惯例）。

### 6.3 Swizzle 作为 Lvalue 的规则

> If the postfix expression before the dot is an lvalue and the swizzle-component-sequence contains no repeated components, the swizzle expression is an lvalue; otherwise it is a prvalue.

这意味着：
```hlsl
float4 v = float4(1,2,3,4);
v.xy = float2(5,6);    // OK: lvalue, xy 无重复
v.xx = float2(5,6);    // Illegal: prvalue, 不能赋值
v.xz = v.yw;           // OK: 两侧都无重复
```

**Intuition**：重复 component 的 swizzle（如 `.xx`）读取时是清晰的（重复值），但写入时语义模糊——`v.xx = float2(a, b)` 应该把 $a$ 还是 $b$ 写到 $v.x$？HLSL 选择禁止这种情况。

参考：[HLSL Swizzle 文档](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math)

## 7. Resource Types

### 7.1 Buffer 类型层次

```
Typed Buffers (有 format conversion)
├── Buffer<T> (read-only)
└── RWBuffer<T> (writable)

Raw Buffers (无 conversion)
├── ByteAddressBuffer / RWByteAddressBuffer (按 byte offset)
└── StructuredBuffer<T> / RWStructuredBuffer<T> (按 struct)
    ├── AppendStructuredBuffer<T>
    └── ConsumeStructuredBuffer<T>
```

### 7.2 Typed vs Raw Buffer 的关键区别

**Typed Buffer** (`Buffer<T>`): 
- 元素最大 16 bytes
- Load 时进行 format conversion（如 `Buffer<float4>` 从 `R32_UINT` 读取会 reinterpret）
- 通过 `operator[]` 或 `Load()` 访问

**Structured Buffer** (`StructuredBuffer<T>`):
- 任意结构类型
- 无 format conversion，直接 reinterpret bits
- 适合自定义 struct

**Byte Address Buffer**:
- 按 byte offset 访问（offset 必须 4-byte 对齐）
- 有 `Load`, `Load2`, `Load3`, `Load4` 返回 `uint`, `uint2` 等
- 模板版 `Load<T>(offset)` 返回任意类型

### 7.3 Atomic Operations

`RWByteAddressBuffer` 有完整的 atomic 操作套件：

**32-bit integer atomics**:
- `InterlockedAdd`, `InterlockedAnd`, `InterlockedOr`, `InterlockedXor`
- `InterlockedMin`, `InterlockedMax` (signed/unsigned 都有)
- `InterlockedExchange`, `InterlockedCompareStore`, `InterlockedCompareExchange`

**64-bit integer atomics** (SM 6.0+):
- 同上的 64-bit 版本

**32-bit float atomics** (有趣！):
- `InterlockedExchangeFloat`
- `InterlockedCompareStoreFloatBitwise`
- `InterlockedCompareExchangeFloatBitwise`

**Float bitwise compare 的陷阱**：spec 明确警告——这种比较是 bitwise 比较，不像标准浮点比较。NaN 的不同 bit 表示、`+0` vs `-0` 等都不会匹配。这在 IEEE 754 浮点语义下是 dangerous 的，但在某些 lock-free 算法中很有用。

参考：[HLSL Atomic Operations](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/interlockedadd)

### 7.4 Hidden Counter

`RWStructuredBuffer` 和 `AppendStructuredBuffer` 有 hidden counter：
```hlsl
uint IncrementCounter();  // 返回 pre-increment 值
uint DecrementCounter();  // 返回 post-decrement 值
```

这类似于 CUDA 的 atomic counter，但由 runtime 管理。Append/Consume buffer 基于 counter 实现 stack-like 语义。

## 8. Constant Buffer Layout——最复杂的规则

### 8.1 16-byte Row 模型

Constant buffer 视为 16-byte row 数组（4×32-bit vector per row）。Packing 规则：

1. **Scalar/vector/single-row column-major matrix**: 尝试 pack 到当前 row 剩余空间
2. **如果跨 row boundary**: align 到下一个 16-byte row
3. **Array/struct**: 总是 16-byte row aligned
4. **Multi-row matrix (column-major)**: 每个 column 对齐到新 row
5. **Row-major matrix**: 每个 row 对齐到新 row

### 8.2 具体例子

```hlsl
cbuffer CB {
    float  a;     // offset 0, row 0, x
    float3 b;     // offset 4, row 0, yzw (fits!)
    float2 c;     // offset 16, row 1, xy (new row, 因为 row 0 已满)
    float  d;     // offset 24, row 1, z
    float  e;     // offset 28, row 1, w
    float  f;     // offset 32, row 2, x
    float2 g[2];  // offset 48, row 3, xy + row 4, xy (array elements 16-byte aligned!)
};
```

注意 `float2 g[2]` 每个 array element 都 16-byte aligned，所以是 row 3 xy + row 4 xy，中间有 padding。

### 8.3 packoffset Annotation

显式指定位置：
```hlsl
cbuffer MyConstants {
    float2 Pos : packoffset(c1.z);  // byte offset 24 (row 1, element z)
};
```

格式：`packoffset(c<row>.<element>)`，element 可以是 `x/y/z/w` 或 `r/g/b/a`。

### 8.4 Default Constant Buffer ($Globals)

全局变量隐式加入 `$Globals` constant buffer（除非 `static`, `groupshared`, 或 resource）：
```hlsl
float4 CameraPos : register(c2);  // 等价于 packoffset(c2)
```

### 8.5 与 std140 的对比

这与 OpenGL 的 `std140` layout 类似但更严格——HLSL 的 packing 更 aggressive（尽量 pack），而 std140 更保守（每个 struct member 16-byte aligned）。

参考：
- [HLSL Constant Buffer Packing](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules)
- [OpenGL std140 layout](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt)

## 9. Resource Binding

### 9.1 Register Types

| Register Type | 用途 | 缩写来源推测 |
|--------------|------|------------|
| t/T | Read-only SRV (Shader Resource View) | "texture" |
| u/U | Writable UAV (Unordered Access View) | "UAV" |
| b/B | Constant Buffer | "buffer" |
| s/S | Sampler | "sampler" |
| c/C | Default constant buffer layout | "constant" |

### 9.2 Register Space

```hlsl
Buffer<float> buf1 : register(t0, space0);
Buffer<float> buf2 : register(t0, space1);  // 不同 space, 不冲突
RWBuffer<float> buf3 : register(u0, space0); // 不同 type, 不冲突
```

每个 space 内每种 register type 有独立 namespace。这解决了 DX11 时代 register 数量上限的问题——DX12 可以有大量 space。

### 9.3 Aggregate Resource Binding

```hlsl
// 占 t0, t1, t2
Buffer<float> fbuf[3] : register(t0, space0);

// 占 t3-t14 (4*3=12 个)
Buffer<int4> ibuf[4][3] : register(t3, space0);

// struct, 不同 register type 独立占用
struct {
    RWBuffer<int> rwbuf;
    Buffer buf;
} sbufs : register(t15) : register(u0);
```

### 9.4 与 Vulkan Descriptor Set 的对应

DX12 的 register space 概念与 Vulkan 的 descriptor set 几乎一一对应：
- `space0` ↔ `set=0`
- `register(t0)` ↔ `binding=0` (combined image sampler / uniform texel buffer)
- `register(u0)` ↔ `binding=0` (storage image / storage buffer)
- `register(b0)` ↔ `binding=0` (uniform buffer)

参考：[DX12 vs Vulkan binding comparison](https://software.intel.com/content/www/us/en/develop/articles/api-without-secrets-introduction-to-vulkan-part-2.html)

## 10. Overload Resolution 的 HLSL 特殊性

### 10.1 参数修饰符导致的 overload

HLSL 有 `in`/`out`/`inout` 参数修饰符，这些影响 overload：

```hlsl
void H(int);          // (1)
void H(in int);       // OK: 与 (1) 相同 (in 是默认)
void H(inout int);    // OK: overload 允许 (in vs inout)
void I(int);          // 
void I(out int);      // OK: in vs out 允许
void J(out int);
void J(inout int);    // Illegal: out vs inout 不允许
```

### 10.2 cxvalue——HLSL 独有的 value category

spec 定义了 5 种 value category：
1. **lvalue**: 函数或对象
2. **rvalue**: 临时对象
3. **xvalue** (expiring value): 生命周期结束的对象
4. **cxvalue** (casted expiring value): xvalue，过期时将值写回绑定的 lvalue
5. **glvalue**: lvalue/xvalue/cxvalue 的统称
6. **prvalue**: 非 xvalue 的 rvalue

**cxvalue 是 HLSL 独有的**！它用于 `out`/`inout` 参数的实现：
- 调用前：从 argument lvalue copy-initialize 一个 cxvalue
- 函数内：cxvalue 作为参数
- 函数返回后：cxvalue 过期时 assignment 回 argument lvalue

这类似 C++ 的 `mutable reference` 但更明确——它是一个有"写回契约"的临时对象。

### 10.3 Inverted Implicit Conversion Sequence

对于 cxvalue 参数，还需要 **inverted ICS**——把 parameter type 转回 argument type 用于写回。这要求双向转换都成立。

## 11. 与其他 GPU 语言的对比

### 11.1 CUDA C++ vs HLSL

| 特性 | CUDA C++ | HLSL |
|------|----------|------|
| 编程模型 | SIMT | SPMD |
| Thread 分组 | warp | Wave |
| Block | block | Thread Group |
| Grid | grid | Dispatch |
| Shared memory | `__shared__` | `groupshared` |
| Vector types | `float4` (library) | `float4` (一等公民) |
| Swizzle | 无原生支持 | 原生支持 |
| Resource binding | 显式 API | `register()` attribute |
| Constant memory | `__constant__` | `cbuffer` |
| Derivative 函数 | 无 | `ddx`/`ddy` (需要 Quad) |

### 11.2 SPIR-V / Vulkan GLSL

Vulkan 的 GLSL → SPIR-V 与 HLSL → DXIL 有趣的对比：
- HLSL 的 `vector<T,N>` 在 SPIR-V 中是 `OpTypeVector`
- HLSL 的 swizzle 在 SPIR-V 中是 `OpVectorShuffle`
- HLSL 的 cbuffer 在 Vulkan 中是 uniform buffer + std140 layout

DXC 实际上有 SPIR-V 后端（[DXC SPIR-V support](https://github.com/microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst)），让 HLSL 可以编译到 Vulkan。

### 11.3 Metal Shading Language

Metal MSL 是 C++14 子集，设计哲学与 HLSL 不同：
- MSL 用 C++ 原生语法 + address space qualifier (`device`, `constant`, `thread`, `threadgroup`)
- HLSL 用特殊 syntax (`cbuffer`, `register()`, resource types)
- MSL 没有 swizzle 一等公民（虽然 `vec<T,N>` 有类似功能）

### 11.4 WGSL (WebGPU)

WGSL 是更新的设计，吸取了 HLSL 的教训：
- 没有 implicit conversion（更严格）
- 没有 swizzle lvalue 写入（更安全）
- 显式 binding (`@group`, `@binding`) 类似 Vulkan

## 12. 一些值得深挖的细节

### 12.1 Vector Literals (§2.9.4)

spec 定义了独特的 vector literal 语法：
```
vector-literal: integer-literal . scalar-element-sequence
                floating-literal . scalar-element-sequence
scalar-element-sequence: x | r | xx | rr | xxx | rrr | ...
```

例如 `1.0.x` 等价于 `float4(1.0, 1.0, 1.0, 1.0)`（vector splat）。这是个语法糖，但语法上 `.` 后跟 swizzle sequence 被特殊处理。

注意 spec 提到：`pp-number` token 在 `.` 后跟 `scalar-element-sequence` 时要被截断，这是 lexer 层面的特殊处理。

### 12.2 Preprocessing Number 的特殊性

spec §2.8 说 pp-number token 不能以 `.` 结尾如果后面是 `scalar-element-sequence`。这是为了支持 `1.0.x` 而不与浮点 `1.0.` 冲突的 lexer 特殊规则。

### 12.3 Attributes vs C++ Attributes

HLSL 的 `[unroll]`, `[loop]`, `[shader("vertex")]` 是 attributes，语法类似 C++17 attributes 但语义不同：
- `[unroll(n)]`: 提示 compiler unroll loop n 次
- `[loop]`: 提示不要 unroll
- `[shader("compute")]`: 标记 entry point 及其 stage

这些是 compiler hints，类似 `#pragma unroll` 但更结构化。

### 12.4 export 关键字

spec 的 `export` 关键字定义 program linkage（可跨 translation unit 或被 runtime 引用）。这让人联想到 C++ 的 `export` template（C++98 的失败 feature）和 DXC 的 library feature（SM 6.6+）。

DX12 的 shader library 允许把多个 shader 编译到一个 library，按需 link。这改变了传统 HLSL 一个文件一个 entry point 的模式。

参考：[DX12 Shader Libraries](https://microsoft.github.io/DirectX-Specs/d3d/ShaderCompiler.html)

## 13. 这份 spec 的局限与未来方向

### 13.1 缺失的内容

从目录看，许多章节还是 stub：
- §10 Classes 只有静态成员和转换，没有完整的 class semantics
- §11 Templates 只有 instantiation 和 partial ordering
- §13 Runtime 完全空白

这些是 spec 未来需要补充的方向。

### 13.2 Aspirational 部分

spec 说 "In very rare instances this spec will be aspirational, and may diverge from both reference implementation behaviors"。这意味着 spec 不只是描述现状，也在引导未来。比如 cxvalue 概念可能是为了更清晰地定义 `out`/`inout` 语义而引入的新术语。

### 13.3 与 LLVM/SPIR-V 的关系

DXC 基于 Clang/LLVM，生成 DXIL（DirectX Intermediate Language，LLVM IR 的子集 + extensions）。这份 spec 实际上是在 formalize LLVM-Clang 行为的 HLSL 子集。

SPIR-V 时代让 HLSL 也能编译到 Vulkan，这促使 HLSL 行为需要更清晰——因为 SPIR-V consumer（Vulkan driver）可能不假设 DXC 的具体行为。

参考：
- [DXIL Specification](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst)
- [SPIR-V Specification](https://www.khronos.org/registry/SPIR-V/)

## 14. 我的思考与延伸

### 14.1 为什么这份 spec 重要

历史上很多语言都是先有实现后有 spec（C 就是这样），但 spec 的存在让第三方实现成为可能。HLSL 一直被 FXC/DXC 双头垄断，社区无法做自己的 HLSL compiler 因为没有 spec。这份文档让 Rust-style 的 alternative HLSL compiler 成为可能。

### 14.2 SPMD 模型的优雅

HLSL 的 SPMD 模型让程序员写"单 lane"代码，compiler 负责映射到 SIMD/SIMT 硬件。这与 CUDA 的 SIMT 类似，但 HLSL 的 Wave intrinsics（`WaveActiveSum`, `WaveMultiPrefixOp` 等）暴露了 Wave 级别的 cooperation，让程序员可以手动优化。

这种分层模型（Lane 抽象 + Wave intrinsics）是现代 GPU 编程的精髓——大多数代码用 Lane 抽象写，性能关键部分用 Wave intrinsics 优化。

### 14.3 Scalarization 的哲学

`scalar-layout-compatible` 的概念实际上在说：HLSL 的类型在内存中只是 scalar 序列，vector/matrix/struct 只是 syntactic sugar。这与 LLVM IR 的 scalarizing 倾向一致。

这也解释了为什么 HLSL 可以如此自由地把 `float4` 当 `struct { float x,y,z,w; }` 用——底层是相同的 4 个 float。

### 14.4 对未来 GPU 编程语言的启示

WGSL 已经吸取了 HLSL 的教训（去掉 implicit conversion，强制 explicit binding）。未来 GPU 语言可能：
- 更严格的 type system（避免 HLSL 的 conversion rank 复杂性）
- 更显式的 parallelism（避免 Lane/Wave 模糊）
- 更好的 ergonomics（HLSL 的 `cbuffer` layout 规则太复杂）

但 HLSL 的设计也证明了"实用主义"的力量——很多复杂规则是为了映射硬件特性，不是美学选择。

---

总结：这份 HLSL spec 是 GPU 编程语言规范化的里程碑。它揭示了 HLSL 二十年来隐含的设计决策：SPMD 模型、scalarization 哲学、resource 作为 handle、conversion rank 系统。理解这些不仅有助于写好 HLSL，也有助于理解 GPU 编程的本质——在抽象与硬件之间寻找平衡。

**参考链接汇总**:
- [DirectX Shader Compiler GitHub](https://github.com/microsoft/DirectXShaderCompiler)
- [HLSL 官方文档](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl)
- [DirectX 12 Specs](https://microsoft.github.io/DirectX-Specs/)
- [DXIL Spec](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst)
- [SPIR-V Spec](https://www.khronos.org/registry/SPIR-V/)
- [Vulkan GLSL vs HLSL](https://github.com/KhronosGroup/glslang)
- [HLSL Packing Rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules)
- [HLSL Swizzle](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math)
- [Wave Intrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
