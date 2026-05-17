**Document Intelligence（文档智能）** 
将非结构化文档内容（如 PDF、图片、Office 文件）转化为精确、可被 AI 系统直接使用的结构化数据。


- 创始人：**VikParuchuri**（GitHub: [VikParuchuri](https://github.com/VikParuchuri)）
- GitHub 组织：[datalab-to](https://github.com/datalab-to)

**开源工具** 和 **商业 API** 两大类：

#### 1. Marker — 文档转换引擎 ⭐

| 属性 | 说明 |
|---|---|
| 仓库 | [github.com/datalab-to/marker](https://github.com/datalab-to/marker) |
| PyPI | `marker-pdf`（最新版本 1.10.2） |
| 功能 | 将 PDF、图片、Office 文件转换为 **Markdown / JSON / HTML / Chunks** |
| 特色 | 支持**表格提取**（`TableConverter`）、高精度版面还原 |

**技术架构解析（第一性原理）：**

Marker 的工作流程可以分解为以下 pipeline：

```
Input Document (PDF/Image/Office)
       ↓
  ┌─────────────────────┐
  │   Surya OCR Engine   │  ← 文字识别 + 版面分析
  │  (Text Line Detection│
  │   + OCR + Layout     │
  │   + Reading Order)   │
  └────────┬────────────┘
           ↓
  ┌─────────────────────┐
  │   Layout Builder     │  ← 构建页面结构树
  │  (Block Hierarchy:   │
  │   Section > Para >   │
  │   Text/Figure/Table) │
  └────────┬────────────┘
           ↓
  ┌─────────────────────┐
  │   Format Renderer    │  ← 输出目标格式
  │  (Markdown / JSON /  │
  │   HTML / Chunks)     │
  └────────────────────┘
```

其核心公式可以抽象为：

$$\text{Output} = f_{\text{render}}\left( f_{\text{layout}}\left( f_{\text{OCR}}(D_{\text{input}}) \right) \right)$$

其中：
- $D_{\text{input}}$ = 输入文档（像素矩阵或 PDF 流）
- $f_{\text{OCR}}$ = OCR 识别函数，输出文本行 $\{(t_i, b_i, c_i)\}$，$t_i$ 为文本，$b_i$ 为 bounding box，$c_i$ 为置信度
- $f_{\text{layout}}$ = 版面分析函数，输出结构化块层级
- $f_{\text{render}}$ = 渲染函数，映射到目标格式

#### 2. Surya — 多语言 OCR 引擎

| 属性 | 说明 |
|---|---|
| 仓库 | [github.com/VikParuchuri/surya](https://github.com/VikParuchuri/surya) |
| 核心能力 | OCR（90+ 语言）、**Layout Analysis**、**Reading Order Detection**、**Table Recognition** |
| 模型架构 | 基于 **Transformer** 的视觉模型 |

**Surya 的多任务能力分解：**

Surya 并非单一 OCR 模型，而是一个 **multi-task document understanding** 框架，同时解决：

1. **Text Line Detection**：检测文档中每一行文字的 bounding box
   - 输入：文档图像 $I \in \mathbb{R}^{H \times W \times 3}$
   - 输出：文本行集合 $\{(x_{min}^{(i)}, y_{min}^{(i)}, x_{max}^{(i)}, y_{max}^{(i)})\}_{i=1}^{N}$
   
2. **OCR Recognition**：对每个检测到的文本行进行文字识别
   - 输入：裁剪后的文本行图像
   - 输出：识别字符串 + 置信度

3. **Layout Analysis**：识别文档区域的语义类别（标题、正文、图片、表格、页脚等）

4. **Reading Order**：确定多栏/复杂排版的阅读顺序
   - 这对报纸、学术论文等复杂排版至关重要

5. **Table Recognition**：识别表格的行/列结构

**与 Tesseract 的对比优势：**
传统 Tesseract 是基于 LSTM 的 OCR，而 Surya 基于 Transformer 架构，在多语言、复杂版面场景下有显著优势。Marker 在处理 PDF 时，即使文档已有 embedded text layer，Surya 仍会被调用进行版面分析和阅读顺序检测。

#### 3. Tabled — 表格提取专用库

| 属性 | 说明 |
|---|---|
| 仓库 | [github.com/VikParuchuri/tabled](https://github.com/VikParuchuri/tabled)（已归档） |
| 功能 | 专门用于检测和提取 PDF 中的表格，识别行列结构并格式化输出 |
| 依赖 | 使用 Surya 进行表格定位 |

---

### 💰 商业模式 — API 服务

Datalab 通过 **[datalab.to](https://www.datalab.to/)** 提供商业 API 服务：

- **文档地址**：[documentation.datalab.to](https://documentation.datalab.to/)
- **定价页**：[datalab.to/pricing](https://www.datalab.to/pricing)
- **计费模式**：基于 **credits** 的月度订阅制，每月支付固定费用获得 credits 余额，用完即止
- **API 功能**：
  - PDF → Markdown / JSON / HTML 转换
  - 文档 OCR 识别
  - 表格提取
  - 版面分析

---

### 📊 Benchmark 表现

根据 [DeepWiki 对 Marker 的 Benchmark 分析](https://deepwiki.com/datalab-to/marker/6.6-benchmarking-and-evaluation)：

| 能力维度 | Benchmark 指标 |
|---|---|
| 整体 PDF → Markdown 转换 | 与 Nougat、MinerU 等工具对比 |
| 表格提取 | 专门的表格准确率评测 |
| 多语言 OCR | 90+ 语言支持评测 |

Marker 在开源 PDF-to-Markdown 工具中被认为是 **精度最高** 的之一（[参考：2026 年对比评测](https://themenonlab.blog/blog/best-open-source-pdf-to-markdown-tools-2026)）。

---

### 🔄 在 AI Pipeline 中的定位

Datalab 的产品定位于 **AI 数据准备层**：

```
Raw Documents (PDF/Image/Office)
       ↓
  ┌──────────────────┐
  │   Datalab API    │  ← Marker + Surya
  │  (Document →     │
  │   Structured     │
  │   Markdown/JSON) │
  └────────┬─────────┘
           ↓
  Structured Text / Chunks
           ↓
  ┌──────────────────┐
  │   LLM Pipeline   │  ← RAG, Fine-tuning, Agent
  │  (GPT/Claude/    │
  │   Open models)   │
  └──────────────────┘
```

这正是 **RAG（Retrieval-Augmented Generation）** 系统中最关键的前置步骤——**文档解析质量直接决定了检索质量和最终生成质量**。

---

### 🔗 相关链接

| 资源 | 链接 |
|---|---|
| 官网 | https://www.datalab.to/ |
| API 文档 | https://documentation.datalab.to/ |
| Marker GitHub | https://github.com/datalab-to/marker |
| Surya GitHub | https://github.com/VikParuchuri/surya |
| Tabled GitHub | https://github.com/VikParuchuri/tabled |
| Marker PyPI | https://pypi.org/project/marker-pdf/ |
| 定价 | https://www.datalab.to/pricing |
| DeepWiki Benchmark | https://deepwiki.com/datalab-to/marker/6.6-benchmarking-and-evaluation |
| PDF-to-Markdown 对比 | https://themenonlab.blog/blog/best-open-source-pdf-to-markdown-tools-2026 |
