# OpenExec：OpenUSD 的通用计算框架

## 一、核心定义

**OpenExec** 是 Pixar 正在为 **OpenUSD**（Universal Scene Description）添加的**通用计算框架**，用于在 USD 场景中**表达和求值计算行为**。它源自 Pixar 内部 DCC 应用 **Presto** 中的 **Presto Execution System**，目标是将该系统开源并重新设计其与 USD 的接口。

> **一句话概括**：OpenExec 让 USD 从"静态场景描述"进化为"可计算的场景描述"。

---

## 二、为什么需要 OpenExec？—— 从第一性原理出发

### 2.1 核心矛盾

USD 的现有设计存在一个根本性矛盾：

| 维度 | 现状 | 需求 |
|------|------|------|
| **值获取** | `UsdAttribute::Get()` 返回 authored 值，固定开销 | 需要计算得到的值（如 bounding box、world transform），开销可变 |
| **计算** | 散落在各个 Schema 的 C++ 方法中（如 `UsdGeomBoundable::ComputeExtent()`） | 需要统一的计算框架，支持自定义、组合、缓存 |
| **缓存** | 无统一缓存/失效机制 | 计算结果需要缓存，且输入变化时需要自动失效 |
| **依赖追踪** | 无 | 需要"值 X 是否依赖时间？"这样的 debug 能力 |
| **可组合性** | 无 | 需要"搭建积木"式的计算组合（尤其角色 rigging） |

### 2.2 第一性原理推导

从**数据流**（dataflow）的角度重新审视场景计算：

1. **一切计算都是值的变换**：输入值 → 函数 → 输出值
2. **变换之间存在依赖关系**：形成有向无环图（DAG）
3. **DAG 可以被调度、并行化、缓存**：这就是数据流网络的核心优势
4. **缓存需要失效机制**：输入变化 → 沿 DAG 反向传播失效 → 按需重新计算

OpenExec 就是将这四个基本洞察工程化为一个完整系统。

---

## 三、OpenExec 是什么

### 3.1 核心能力

```
┌─────────────────────────────────────────────────────┐
│                   OpenExec 框架                      │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Named       │  │ Dataflow     │  │ Caching &  │ │
│  │ Computations│→ │ Network      │→ │ Invalidation│ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│         │                │                │         │
│         ▼                ▼                ▼         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Computation │  │ Multi-thread │  │ Dependency │ │
│  │ Callbacks   │  │ Evaluation   │  │ Tracking   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

- **Named Computations**：与 USD 场景对象（prim、property）关联的命名计算
- **Dataflow Network**：底层维护的向量化数据流网络（节点 = 计算任务，边 = 数据连接）
- **Fast Multi-threaded Evaluation Engine**：快速多线程求值引擎
- **Automatic Caching & Invalidation**：自动缓存计算结果，输入变化时自动失效

### 3.2 在架构中的位置

```
┌──────────────┐
│   Imaging    │   ← 消费计算结果
└──────┬───────┘
       │
┌──────▼───────┐
│  OpenExec    │   ← 观察 composed USD stage，提供计算值
└──────┬───────┘
       │
┌──────▼───────┐
│    USD       │   ← composed stage（只读观察）
└──────────────┘
```

OpenExec **观察** composed USD stage，**不修改** stage 的拓扑（不能增删 prim 或 property）。它坐在 USD 和 Imaging 之间。

---

## 四、OpenExec 不是什么

| 误解 | 真相 |
|------|------|
| OpenExec 能修改 stage 拓扑 | ❌ 不能增删 prim/property，只能**观察** composed stage |
| OpenExec 是 rigging 系统 | ❌ 它是 rigging 系统的**基础设施**。Pixar 的 rigging 系统包含三层：① Execution System → ② Rigging Object Model → ③ Deformer/Rigging 库。OpenExec 只提供第①层 |
| OpenExec 是事件驱动系统 | ❌ 它是**值驱动**（value-driven）的。输入变化 → 失效标记 → 下次请求时才重新计算。没有"点击触发重算"的机制 |

**事件驱动的例子**（非 OpenExec 职责）：
- 点击车门 → 修改 `openness` 属性 → 这是**应用层**的责任
- `openness` 变化 → 自动失效车门 transform 的计算结果 → 下次请求时重新计算 → 这是 **OpenExec** 的责任

---

## 五、代码组织与库结构

OpenExec 位于 `pxr/exec/`，包含以下库：

```
pxr/exec/
├── vdf/          ← 向量化数据流底层库
├── ef/           ← 执行基础库
├── esf/          ← 场景抽象层
├── exec/         ← 高级执行库
├── esfUsd/       ← esf 的 USD 特化
├── execUsd/      ← exec 的 USD 特化（客户端主要接口）
└── execGeom/     ← 核心 Schema 的计算行为注册
```

### 5.1 各库详解

| 库 | 全称 | 职责 |
|----|------|------|
| **vdf** | Vectorized Dataflow | 最底层。实现数据流网络的核心概念：**节点**、**连接器**、**边**，以及底层求值引擎和调度器。"向量化"指网络能编码向量数据的拓扑，解锁额外的稀疏性和并行性 |
| **ef** | Execution Foundation | 在 vdf 之上构建高层概念：具体节点类型、具有高层执行意识的调度器 |
| **esf** | Execution Scene Foundation | 场景抽象层，模仿 USD 概念但不依赖具体 USD 类型。提供**日志**机制，将场景访问映射到数据流网络的编辑 |
| **exec** | Execution | 高级库，引入**编译**概念：将场景描述 + 计算定义 → 向量化数据流网络。维护网络并响应场景变化 |
| **esfUsd / execUsd** | USD 特化 | esf 和 exec 针对 USD 的实现。**execUsd 是客户端主要接口** |

### 5.2 依赖关系

```
pxr/base (arch, tf, trace, work...)
    ↓
pxr/usd/sdf
    ↓
pxr/usd/usd
    ↓
pxr/exec/vdf → ef → esf → exec
                         ↓
                    esfUsd → execUsd
                                ↓
                          execGeom (核心Schema计算)
```

---

## 六、核心概念详解

### 6.1 Named Computations（命名计算）

任何 USD 场景对象都可以是**计算提供者**（computation provider），发布任意数量的命名计算。

**计算来源**：

1. **Built-in computations**（内置计算）：每个 USD 对象自动发布的计算
   - 例如：提供对象 scene graph path 的计算
   - 例如：`UsdAttribute` 的 `computeValue` 内置计算

2. **Registered computations**（注册计算）：通过 Schema 注册的自定义计算
   - 例如：在 `UsdGeomBoundable` 上注册 `computeExtent` 计算

**引用方式**：通过 `ExecUsdValueKey` 引用，包含：
- 计算提供者路径
- 计算名称 token

### 6.2 Input Parameters（输入参数）

输入参数是计算获取数据的方式。每个输入参数编码三个信息：

```
Input Parameter = {
    Computation Provider Resolution,  // 在哪个场景对象上查找
    Computation Name,                 // 计算名称
    Result Type / Input Value Type    // 数据类型（强类型）
}
```

**Provider Resolution 的类型**：
- **Attribute Input**：解析到 prim 上的某个 `UsdAttribute`
- **Relationship-targeted Input**：解析到 `UsdRelationship` 目标指向的场景对象（0 个或多个）

**关键约束**：输入依赖是**静态的**（static with respect to evaluation）—— 在求值期间不能动态决定是否需要某个输入。引擎总是会满足所有声明的依赖，即使回调不消费该值。

### 6.3 Computation Callbacks（计算回调）

计算回调是执行计算工作的 C++ 函数对象。

**关键约束**：

| 约束 | 原因 |
|------|------|
| **无副作用**（side-effect free） | 输入相同 → 输出必须相同，这是缓存正确性的基础 |
| **无状态**（stateless） | 所有状态通过输入连接器传入 |
| **不修改全局状态** | 保证线程安全 |
| **输出值不可变** | 回调返回后输出值变为不可变，因为多个回调可能并行读取同一输入值 |

**回调形式要求**：
- 自由函数或静态函数
- 非捕获 lambda + `operator+()` 转为函数指针

```cpp
// 典型回调签名
static VtValue ComputeExtent(const VdfContext& ctx) {
    // 从 ctx 读取输入值
    // 执行计算
    // 将输出值写入 ctx
}
```

**求值顺序保证**：
- 输入依赖一定在依赖它的计算之前求值
- 除此之外，不保证任何求值顺序
- 回调将与其他回调、同一回调的其他实例**并行执行**

### 6.4 Computation Registration（计算注册）

OpenExec 扩展 USD 的 Schema 注册机制，提供**执行行为注册**能力：

```
USD Schema Registration (现有)
    +
Exec Registration (新增)
    ↓
Exec Registry (类比 UsdPrimRegistry)
    ↓
Exec Computation Definition (类比 UsdPrimDefinition)
    ↓
Dataflow Network Compilation
```

注册使用**基于 C++ 元编程的 DSL**（类比 `pybind11` 的风格）。

### 6.5 Dynamic Registration / Configuration（动态注册/配置）

虽然输入依赖是静态的，但有时不需要的依赖会损害性能。**动态执行注册**允许：
- 将特定元数据标记为**配置元数据**
- 系统监控这些属性的变化
- 属性变化时，重新执行注册回调中的部分代码
- 回调可以**取消注册**和**重新注册**具有不同输入依赖的计算

### 6.6 Actions and Execution Markers（动作与执行标记）

这是 OpenExec 中最强大的概念之一，尤其对角色 rigging 关键。

**Action（动作）**：
- 提供类似控制器的行为
- 通过 relationship target 指向场景中的属性
- 动作提供一个计算来**影响**被指向的属性值
- 动作看到的是"动作生效前"的属性值，可以修改或覆盖
- 多个动作可以指向同一属性，形成**动作栈**—— 按 namespace 逆序执行（deformation chain）

**Execution Marker（执行标记）**：
- 用于消歧查询的是哪个"版本的"属性值
- `at-final`：所有动作执行后的最终计算值
- 也可以指定特定动作后的值

**关键设计决策**：OpenExec 不允许计算**取代**属性的已著值。而是**分叉**属性值：
- `UsdAttribute::Get()` → 始终返回 authored 值（不变）
- `ExecUsdValueKey(computeValue)` → 返回计算值（有计算时），或回退到 authored 值（无计算时）

```
属性 A 的值空间:
┌───────────────────────────────┐
│ authored value (UsdAttribute::Get() 返回)  ← 永远不变
├───────────────────────────────┤
│ connections + combiners        │  ← computeExpression
├───────────────────────────────┤
│ action 1 (namespace 逆序)      │  ← 影响层
├───────────────────────────────┤
│ action 2                       │
├───────────────────────────────┤
│ ...                            │
├───────────────────────────────┤
│ at-final value                 │  ← 最终计算值
└───────────────────────────────┘
```

### 6.7 Connectable Behaviors（可连接行为）

属性连接 = 数据流：

```
Attribute A ──connect──→ Attribute B
     │                       │
     │  computeValue(A)      │  computeValue(B)
     │  = computeValue(B)    │
     ▼                       ▼
  请求 A 的计算值 → 返回 B 的计算值
```

### 6.8 Authored Combiners（著写组合器）

当一个属性同时有 authored 值和多个连接时的组合方式：

| 组合器 | 含义 | 公式 |
|--------|------|------|
| **Plus (+)** | 连接属性值求和 | $result = \sum_{i} computeValue(B_i)$ |
| **PlusEquals (+=)** | 求和 + 加上自身 authored 值 | $result = authored_A + \sum_{i} computeValue(B_i)$ |
| **Times (\*)** | 连接属性值求积 | $result = \prod_{i} computeValue(B_i)$ |
| **TimesEquals (\*=)** | 求积 + 乘以自身 authored 值 | $result = authored_A \times \prod_{i} computeValue(B_i)$ |

其中 $B_i$ 表示第 $i$ 个连接到的属性。

### 6.9 Expressions（表达式）

每个属性提供 `computeExpression` 内置计算，`computeValue` 内部依赖于 `computeExpression` 的结果。`computeExpression` 可以通过 exec 注册被覆盖，允许 Schema 作者自定义连接行为。

**求值顺序**：connectable behaviors（连接 + 组合器 + 表达式）→ actions

### 6.10 Relationship Forwarding（关系转发）

Relationship 不像 connection 那样隐含数据流，但 OpenExec 会自动解析链式 relationship target 到**终端对象**：

```
Prim A --relationship--> Prim B --relationship--> Prim C
                                                    │
                                            输入参数解析到 Prim C
```

这对建模接口（接口背后的计算通过 relationship 暴露）至关重要。

---

## 七、架构：三大子系统

```
┌──────────────────────────────────────────────────────┐
│                    Client Request                     │
│                    (值请求)                           │
└─────────────┬────────────────────────────────────────┘
              │
              ▼
┌─────────────────────┐
│   1. Compilation    │ ← 最昂贵，最不频繁
│   (编译)            │    仅在结构性变更后触发
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   2. Scheduling     │ ← 中等开销
│   (调度)            │    按请求集合缓存
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   3. Evaluation     │ ← 最快，最频繁
│   (求值)            │    每帧/每次值变化后
└─────────────────────┘
```

### 7.1 Compilation（编译）

将**计算定义** + **著写的 USD 场景** → **数据流网络**。

- **结构性变更**（如 reparenting prim）→ 受影响子网络**反编译** → 下次请求时**重新编译**
- **非结构性变更**（如属性值变化）→ 不触发编译，通过值失效处理

**设计洞察**：编译将高层"复杂指令集"（exec registration + 场景描述）映射到底层"精简指令集"（数据流网络），类似于编译器优化。

### 7.2 Scheduling（调度）

**为什么需要调度？** 大型图结构的指针追踪效率低，除非图以连续遍历顺序布局在内存中。

调度是**加速结构**：
- 保证每个数据流节点**最多被访问一次**
- 数据以**连续顺序**访问
- 按**请求对象**缓存（同一组请求可复用调度）

### 7.3 Evaluation（求值）

**Pull-based 求值模型**：

```
请求值
  │
  ▼
从叶子节点开始 → 沿数据依赖向上遍历
  │
  ▼
遇到缓存命中 or 根节点 → 回溯
  │
  ▼
从根节点开始，输入依赖已缓存的节点 → 调用回调
  │
  ▼
节点并行求值，仅在输入依赖上同步
```

**时间处理**：时间作为数据流网络的输入值，初始化一个特殊的根节点。对时间根节点的拓扑依赖表示计算是**时间依赖的**。设置新时间不需要拓扑编辑，只需值失效 + 重新初始化时间码。

### 7.4 Vectorized Dataflow（向量化数据流）

这是 OpenExec 的一个核心创新。

**传统数据流**：连接传递标量值。

**向量化数据流**：连接可以传递**向量数据**，且向量的拓扑被编码在网络拓扑中。

连接上附带 **mask**（RLE 编码的轻量位集），指示计算依赖向量的哪些元素：

```
Connection Mask: [1,0,0,1,1,0,...]  ← 表示依赖第 0, 3, 4 个元素
```

**优势**：
1. **内存连续性**：同类型数据（如多条短曲线）连续存储，无假依赖
2. **稀疏操作**：deformer 可限制在 weight 界定的 mesh 点子集上操作
3. **额外并行性**：引擎追踪每个向量元素的数据依赖，自动提取并行性

**代价**：向量拓扑在求值期间是**固定的**。例如角色变形不能改变 mesh 拓扑，除非退出快速路径。

---

## 八、Client API

### 8.1 执行系统对象

```cpp
// 构建执行系统实例，与特定 USD stage 绑定
ExecUsdSystem system(stage);
```

- 与 stage 共享引用
- 多个实例可以指向同一 stage，但缓存不共享
- 典型一对一关系

### 8.2 请求值的流程

```
1. 编译数据流网络（首次请求时编译，后续复用）
2. 构建调度（按请求集合缓存）
3. 执行求值（并行调用回调，缓存结果）
```

**关键性能原则**：批量请求！

| 模式 | 性能 |
|------|------|
| 每次请求一个 value key | ❌ 悲观 |
| 批量请求所有需要的 value keys | ✅ 最优（调度、求值开销均摊） |

### 8.3 API 层次

**Low-level API**：`ExecRequest`
- 鼓励批量模式
- 最低性能开销

**High-level API**：
- 更友好的接口
- 适合移植现有代码
- 牺牲少量性能换取易用性

### 8.4 失效通知

```cpp
// 构建请求时提供失效回调
auto request = system.MakeRequest(valueKeys, /*invalidationCallback=*/[](const InvalidatedSet& invalidated, const TimeRange& range) {
    // invalidated: 失效的 value key 索引集合
    // range: 失效的时间范围
    // 用途：标记客户端持有的缓存为脏
});
```

**规则**：
- 只通知**之前请求过且现在失效**的值
- 只在值**首次失效**时通知
- 客户端需重新请求才能收到后续失效通知
- **不能**在回调内调用 OpenExec（不支持同步重算）

---

## 九、新编译器架构

### 9.1 为什么不直接移植 Presto 编译器？

| 问题 | 说明 |
|------|------|
| **代码量大** | Presto 编译器功能多，无法部分移植，全量移植范围过大 |
| **概念不匹配** | Presto 有些概念在 USD 中不存在或已被替代 |
| **API 差异** | Presto 场景 API 与 USD 差异大，需要抽象层 |
| **架构缺陷 1** | 首次编译和重编译代码路径不同，维护负担 |
| **架构缺陷 2** | 原始架构未考虑并发访问，数据结构不利于多线程 |

### 9.2 新架构核心：EsfStage

```
┌──────────────────────────────────────┐
│            EsfStage                  │
│  (无状态、轻量级场景访问抽象层)       │
│                                      │
│  1. 记录场景查询的日志               │
│     → 自动关联到生成的数据流子网络   │
│     → 自动确定哪些场景变化           │
│       需要触发反编译/重编译          │
│                                      │
│  2. 提供统一的场景访问接口           │
│     → USD 和 Presto 场景均可使用     │
└──────────────────────────────────────┘
```

**设计策略**：不从头重写，而是**架构革新 + 复制现有算法行为**，确保两个编译器输出一致。

---

## 十、交付计划

### Phase 1 — 最小可行产品（2025 年 6 月 / SIGGRAPH 2025）

- 添加一组计算：解释 xform ops → 生成 world-space transform matrices
- 不替换 OpenUSD 现有的 transform 计算
- 沙盒发布，API 可能变化

### Phase 2 及以后

1. 增量交付新功能，达到（并超越）Presto Execution System 的对等性
2. 迭代性能和可扩展性改进
3. 利用计算框架为 USD 添加新功能，或在有明显优势处替换现有功能

### Late Phase（远期）

- **弃用并移除**现有的 Schema 计算方法（如 `UsdGeomBoundable::ComputeExtent()`）
- 替换为 OpenExec 计算
- 提供过渡工具和指南
- 在过渡期内同时维护旧方法和新计算

---

## 十一、总结：OpenExec 的核心洞察

从第一性原理总结 OpenExec 的设计哲学：

1. **计算即数据流**：所有计算统一建模为 DAG 上的节点，数据沿边流动
2. **静态依赖 + 动态失效**：依赖关系在编译时确定（静态），值的变化通过失效传播（动态）
3. **无状态回调 + 侵略性缓存**：回调的纯函数性质使缓存正确；缓存使重复求值几乎免费
4. **向量化拓扑**：将向量数据的拓扑编码进网络拓扑，解锁稀疏性和并行性
5. **按需计算**：pay-for-what-you-use，只编译和求值请求的计算及其依赖
6. **关注点分离**：
   - 场景描述（USD）↔ 计算行为
   - 编译（慢但少）↔ 调度（中）↔ 求值（快但频）
   - authored 值 ↔ computed 值（永不混淆）

---

**参考链接**：
- OpenUSD 官方文档: https://openusd.org/release/index.html
- OpenExec 提案原文（Pixar USD Working Group）: https://wiki.aswf.io/display/OSG/OpenExec
- ASWF USD Working Group 会议记录: https://wiki.aswf.io/display/OSG/USD+Working+Group
- Presto / OpenUSD 相关 SIGGRAPH 演讲: https://graphics.pixar.com/usd/release/index.html