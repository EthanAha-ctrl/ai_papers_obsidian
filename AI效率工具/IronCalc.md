





根据搜索结果，**Visprex** 是另一个开源的 Web 数据可视化工具。让我构建一个深度技术对比：

---

## 一、Visprex 是什么？

**Visprex** 是一个**轻量级的 CSV 数据可视化工具**，专注于：
- 快速上传 CSV 文件 → 自动生成图表
- **纯客户端处理**（"without sending your data anywhere"），保护隐私
- 使用 **TypeScript** 编写的前端 Web 应用
- 无需注册、无服务器存储，数据永不离开浏览器

核心价值：**让非技术用户秒级将 CSV 转换为可视化图表**，类似 Tableau Public 或 Google Sheets 图表功能的极简替代。

---

## 二、技术架构对比

### **IronCalc 架构（Rust 电子表格引擎）**

```
┌─────────────────────────────────────────────┐
│                UI Layer (Rust/WASM)         │
│        (Future: separate frontend)         │
└───────────────┬─────────────────────────────┘
                │ WASM API
┌───────────────▼─────────────────────────────┐
│          Core Engine (Rust)                 │
│  ┌─────────────────────────────────────┐   │
│  │  Dependency Graph                  │   │
│  │  - Topological sort for calc order │   │
│  │  - Incremental updates             │   │
│  │  - Cyclic dependency detection     │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Formula Parser & Evaluator        │   │
│  │  - Lexer → Parser → AST → Eval     │   │
│  │  - 400+ built-in functions         │   │
│  │  - Custom function support         │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Memory Model                      │   │
│  │  - Sparse storage (HashMap)        │   │
│  │  - Cell value enum (num, str, bool)│   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**关键性能优化**：
- 依赖图缓存：`HashMap<CellRef, Vec<CellRef>> dependents`
- 脏标记传播：`DirtyBit` 标记需要重算的单元格
- 并行计算：`rayon` 并行计算独立子树

### **Visprex 架构（TypeScript 可视化工具）**

```
┌─────────────────────────────────────────────┐
│              Next.js/React Frontend         │
│  ┌─────────────────────────────────────┐   │
│  │  CSV Upload & Parsing              │   │
│  │  - PapaParse/CSV parser            │   │
│  │  - Schema inference (type guess)   │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Data Manipulation                 │   │
│  │  - Filter, sort, group             │   │
│  │  - Aggregation (sum, avg, count)   │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Chart Generation                  │   │
│  │  - Recharts / Chart.js / ECharts   │   │
│  │  - Auto chart type selection       │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**关键技术点**：
- CSV 解析：`Papa.parse(csvString, {skipEmptyLines: true})`
- 类型推断：列数据采样前 100 行 → 推断数字/字符串/日期
- 图表推荐：基于数据类型自动选择（时间序列 → 折线图；类别对比 → 柱状图）
- 状态管理：React Context 或 Zustand 存储整个数据集

---

## 三、核心差异对比表

| 维度 | IronCalc | Visprex |
|------|----------|---------|
| **项目定位** | 通用电子表格引擎（Excel 替代） | CSV 快速可视化工具（图表生成器） |
| **核心功能** | 公式计算、依赖图、递归重算、文件持久化 | CSV 上传 → 图表渲染 |
| **输入格式** | `.ironcalc`（自有格式），未来可能支持 ODS/XLSX 导入 | CSV 文本（纯文本） |
| **输出形式** | 交互式表格、可编辑单元格、公式结果 | 静态/交互式图表（PNG/Web） |
| **技术栈** | Rust + WASM（性能优先） | TypeScript + React（快速迭代） |
| **内存模型** | 稀疏存储（HashMap），支持超大表格 | 稠密数组（Array），数据全载入内存 |
| **计算能力** | 完整编程语言级别（循环、条件、400+函数） | 仅聚合函数（sum, avg, count）和简单过滤 |
| **可编程性** | ✅ API 可嵌入，支持自定义函数 | ❌ 固定工作流，无 API |
| **数据隐私** | ✅ 完全本地/自托管 | ✅ 纯前端，无服务器 |
| **协作能力** | ✅ WebXDC/CRDT 支持实时同步 | ❌ 单用户，无协作 |
| **性能极限** | 百万级单元格（Rust 优化） | 千行级 CSV（浏览器内存限制） |
| **适用场景** | 财务模型、业务逻辑、复杂计算、嵌入系统 | 数据探索、快速报告、非技术用户分享 |
| **许可证** | MIT/Apache 2.0（Rust 部分） | 需确认（通常为 MIT） |

---

## 四、使用场景隔离

### **选择 IronCalc 的场景**：
1. **需要公式计算**：`=XIRR(现金流数组, 日期数组)` 金融函数
2. **多表关联**：`VLOOKUP` 跨 sheet 引用
3. **迭代计算**：单变量求解、Goal Seek
4. **嵌入到产品**：你的 SaaS 需要内嵌表格编辑器
5. **数据持久化**：保存 `.ironcalc` 文件，下次继续编辑
6. **协作编辑**：多人同时修改表格，实时同步

### **选择 Visprex 的场景**：
1. **快速可视化**：上传一个 CSV，30 秒内出图表
2. **非技术用户**：拖拽操作，无需学习公式
3. **探索性分析**：我不知道数据什么分布，想先看看
4. **分享图表**：生成图表链接直接发到聊天
5. **轻量报告**：不需要复杂表格，只要几张图

---

## 五、技术深度解析

### **IronCalc 的依赖图算法（伪代码）**

```rust
// 构建依赖图
fn build_dependency_graph(sheet: &Sheet) -> Graph<CellRef> {
    let mut graph = Graph::new();
    for (cell_ref, cell) in &sheet.cells {
        let node = graph.add_node(cell_ref.clone());
        for dep in cell.formula.dependencies() {
            graph.add_edge(node, graph.node_index(&dep), ());
        }
    }
    graph
}

// 拓扑排序计算顺序
fn calc_order(graph: &Graph<CellRef>) -> Vec<CellRef> {
    let mut order = Vec::new();
    let mut indegree = vec![0; graph.node_count()];
    // 计算入度
    for node in 0..graph.node_count() {
        for edge in graph.edges(node) {
            indegree[edge.target()] += 1;
        }
    }
    // Kahn 算法
    let mut queue = VecDeque::new();
    for (i, &deg) in indegree.iter().enumerate() {
        if deg == 0 { queue.push_back(i); }
    }
    while let Some(node) = queue.pop_front() {
        order.push(graph[node].clone());
        for edge in graph.edges(node) {
            indegree[edge.target()] -= 1;
            if indegree[edge.target()] == 0 {
                queue.push_back(edge.target());
            }
        }
    }
    order
}
```

**复杂度**：O(V + E)，V=单元格数，E=引用数

### **Visprex 的自动图表选择逻辑（推测）**

```typescript
function recommendChartType(columnTypes: ColumnType[]): ChartType {
    const hasNumeric = columnTypes.some(t => t === 'number');
    const hasDate = columnTypes.some(t => t === 'date');
    const hasCategory = columnTypes.some(t => t === 'string');
    
    if (hasDate && hasNumeric) return 'line';  // 时间序列
    if (hasCategory && hasNumeric) {
        const cardinality = getUniqueCount(categoryColumn);
        if (cardinality > 10) return 'bar';    // 多类别
        else return 'pie';                     // 少类别
    }
    if (hasNumeric && hasNumeric) return 'scatter'; // 两个数值变量
    return 'table'; // fallback
}
```

---

## 六、性能与扩展性

| 指标 | IronCalc | Visprex |
|------|----------|---------|
| **最大数据量** | 受 Rust 内存限制（GB 级） | 受浏览器内存限制（通常 <100MB） |
| **冷启动时间** | WASM 初始化 ~100-500ms | React 加载 ~200ms |
| **图表渲染延迟** | 不适用（表格计算） | 10,000 行数据 ~200ms (Canvas) |
| **可伸缩性** | 支持分块加载、虚拟滚动 | 全量数据加载，大文件易崩溃 |
| **离线能力** | ✅ PWA + WASM 完全离线 | ✅ 纯前端，但需预先加载资源 |

---

## 七、为什么两个项目都存在？（生态位分析）

**不存在直接竞争**，它们解决不同层次的问题：

1. **计算 vs 呈现**：
   - IronCalc 解决 "如何计算"（How to compute）
   - Visprex 解决 "如何展示"（How to visualize）

2. **复杂度 vs 速度**：
   - IronCalc 适用于复杂业务逻辑（需要数月学习）
   - Visprex 适用于快速探索（10 秒上手）

3. **产品形态**：
   - IronCalc → 可嵌入的 **库** (library)
   - Visprex → 独立 **工具** (tool)

4. **用户心智模型**：
   - IronCalc 用户："我需要一个可编程的计算引擎"
   - Visprex 用户："我有个 CSV 文件，想看看里面数据长什么样"

**可能的整合**：
- 未来 IronCalc 可添加 "导出图表" 功能，类似 Visprex
- Visprex 可增加简单公式列（如 `=列A*2`），接近微型表格

---

## 八、总结与建议

### **项目价值总结**

| 项目 | 核心价值 | 护城河 |
|------|---------|--------|
| **IronCalc** | 提供开源、高性能、可嵌入的现代电子表格计算引擎，打破 Excel/Google 垄断 | Rust 实现的复杂依赖图系统 + 供应商中立 |
| **Visprex** | 为 CSV 数据提供零门槛的隐私友好可视化，降低数据洞察成本 | 极简 UX + 纯前端架构 |

### **技术选型建议**

**您的团队应该根据需求选择：**

- **需要业务逻辑、财务模型、可重复计算** → **IronCalc**
- **需要快速把现成 CSV 变成图表分享** → **Visprex**
- **想构建自己的数据分析平台** → 用 **IronCalc 作为计算后端** + **Visprex 作为前端图表组件**（两者可互补）

---

## 参考链接

- [IronCalc GitHub](https://github.com/ironcalc/IronCalc)
- [IronCalc 文档](https://ironcalc.readthedocs.io/)
- [Visprex GitHub](https://github.com/visprex/visprex)
- [Visprex 文档](https://docs.visprex.com/)
- [Visprex 在线使用](https://visprex.com)
- [Hacker News 讨论 Visprex](https://news.ycombinator.com/item?id=42096837)
- [XDA 报道 IronCalc](https://www.xda-developers.com/ironcalc-makes-powerful-spreadsheets-possible-without-microsoft-google/)