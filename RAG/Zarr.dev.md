





# Zarr.dev — Cloud-Native 的 N 维数组存储格式

## 一句话概括

**[zarr.dev](https://zarr.dev/)** 是 **Zarr** 这个开源项目的官网，Zarr 是一种用于存储**大规模 N 维 typed array（多维类型化数组）**的**开放标准数据格式**和配套软件生态，核心设计目标是 **cloud-ready**（云原生）。

---

## 1. 第一性原理：为什么需要 Zarr？

从第一性原理出发，科学计算 & 机器学习中存储多维数组面临的核心矛盾是：

| 矛盾维度 | 传统方案（HDF5/NetCDF4） | Zarr 的解法 |
|---|---|---|
| **数据规模** vs **I/O 带宽** | 单体文件，必须整块打开 | **Chunked** 分块存储，按需读取 |
| **本地磁盘** vs **云对象存储** | POSIX 文件系统 API，不适合 HTTP range read | 每个chunk一个key，天然映射到 S3/GCS/ABS |
| **串行写入** vs **并行计算** | 单 writer lock | 无锁并发读写，每个chunk独立 |
| **紧耦合格式** vs **可扩展性** | 私有二进制格式 | JSON metadata + 二进制chunk，可插拔 codec |

**核心洞察**：把一个大的 N 维数组切成小 chunk，每个 chunk 独立存储为一个对象（key），metadata 用 JSON 描述。这样就把 **array → (metadata, chunks[])** 的映射变成了 **key-value store** 上的操作，而 cloud object storage 正是 key-value store 的最佳载体。

---

## 2. Zarr 数据模型详解

### 2.1 核心概念层级

```
Zarr Hierarchy
├── Group (类似文件系统的目录)
│   ├── .zgroup (V2) / zarr.json (V3)  ← Group metadata
│   ├── attrs (用户自定义属性)
│   ├── Array_1 (N维数组)
│   │   ├── .zarray (V2) / zarr.json (V3)  ← Array metadata
│   │   ├── attrs
│   │   ├── 0.0.0  ← chunk key (data)
│   │   ├── 0.0.1
│   │   └── ...
│   ├── Array_2
│   └── SubGroup/
│       └── ...
```

### 2.2 Array Metadata（V2 格式示例）

```json
{
    "zarr_format": 2,
    "shape": [10000, 10000],
    "chunks": [1000, 1000],
    "dtype": "<f8",
    "compressor": {
        "id": "zlib",
        "level": 1
    },
    "fill_value": 0.0,
    "order": "C",
    "filters": null,
    "dimension_separator": "."
}
```

关键变量解释：

- **`shape`**: 数组的完整维度，$[d_0, d_1, \ldots, d_n]$，例如 `[10000, 10000]` 表示 10000×10000 的矩阵
- **`chunks`**: 每个 chunk 的维度，$[c_0, c_1, \ldots, c_n]$，chunk 数量 $= \prod_{i=0}^{n} \lceil d_i / c_i \rceil$
- **`dtype`**: 数据类型，`<f8` = little-endian 64-bit float（IEEE 754 double）
- **`compressor`**: 压缩 codec，支持 zlib/bz2/zstd/LZ4 等
- **`fill_value`**: 未写入区域的默认填充值
- **`order`**: 内存布局，C（row-major）或 F（column-major）
- **`filters`**: 压缩前的预处理管线（如 delta filter, scale offset filter）
- **`dimension_separator`**: chunk key 中维度间的分隔符，`.` 或 `/`

### 2.3 V3 Specification 的重大变化

Zarr V3（[Zarr specs v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)）引入了根本性的架构升级：

| 特性 | V2 | V3 |
|---|---|---|
| Metadata 格式 | `.zarray`, `.zgroup`, `.zattrs` 多个文件 | 统一为单个 `zarr.json` |
| Codec 管线 | 固定顺序 (filters → compressor) | 可扩展的 codec chain，通过 extensions 注册 |
| 数据类型 | NumPy dtype 字符串 | 结构化 data type 系统 |
| Sharding | 不支持 | **ZEP0002**: 多个 chunk 合并到同一 storage key |
| Storage transformers | 不支持 | 可插拔的存储转换层 |

#### V3 zarr.json 示例

```json
{
    "zarr_format": 3,
    "node_type": "array",
    "shape": [10000, 10000],
    "data_type": "float64",
    "chunk_grid": {
        "name": "regular",
        "configuration": {"chunk_shape": [1000, 1000]}
    },
    "chunk_key_encoding": {
        "name": "default",
        "configuration": {"separator": "/"}
    },
    "fill_value": 0.0,
    "codecs": [
        {"name": "zstd", "configuration": {"level": 1}}
    ]
}
```

---

## 3. Chunk 存储的数学模型

给定一个 shape 为 $[d_0, d_1, \ldots, d_n]$ 的数组，chunk 大小为 $[c_0, c_1, \ldots, c_n]$：

**Chunk 索引空间**：

$$\text{chunk\_indices}(i) = \left\lfloor \frac{i_k}{c_k} \right\rfloor, \quad k = 0, 1, \ldots, n$$

**总 chunk 数**：

$$N_{\text{chunks}} = \prod_{k=0}^{n} \left\lceil \frac{d_k}{c_k} \right\rceil$$

**Chunk key 生成**（default encoding）：

$$\text{key} = \text{prefix} + \text{separator} + \lfloor i_0/c_0 \rfloor + \text{separator} + \lfloor i_1/c_1 \rfloor + \ldots$$

例如：shape=[10000,10000], chunks=[1000,1000]，访问元素 [3500, 7200] → chunk key = `3/7`

**单个 chunk 的字节大小**：

$$B_{\text{chunk}} = \left(\prod_{k=0}^{n} c_k\right) \times \text{sizeof}(\text{dtype})$$

例如：chunks=[1000,1000], dtype=float64(8 bytes) → $B_{\text{chunk}} = 1000 \times 1000 \times 8 = 8\text{MB}$

---

## 4. Sharding Codec（ZEP0002）— 云优化的关键

### 问题

在云对象存储（如 S3）上，每个 chunk 对应一个 GET 请求。当 chunk 很小时，**request overhead** 成为瓶颈（S3 每次请求 ~25-50ms latency）。

### 解法

**Sharding**：将多个逻辑 chunk 合并到同一个 storage key（shard）中。

```
Shard (一个 storage key)
┌─────────┬─────────┬─────────┬──────────────────┐
│ Chunk 0 │ Chunk 1 │ Chunk 2 │ Shard Index      │
│ (data)  │ (data)  │ (data)  │ (offset+size表)  │
└─────────┴─────────┴─────────┴──────────────────┘
```

**Shard Index 结构**：每个 entry 包含 offset 和 size：

$$\text{index}[i] = (\text{offset}_i, \text{size}_i)$$

读取特定 chunk 时：
1. 读取 shard 尾部的 index（固定大小，可 HTTP range read）
2. 从 index 中查到目标 chunk 的 $(\text{offset}, \text{size})$
3. 用 HTTP range request 精确读取该 chunk

这样就把 $O(N_{\text{chunks}})$ 次 GET 请求降为 $O(N_{\text{shards}})$ 次，同时仍保持随机访问能力。

---

## 5. Codec Pipeline（编解码管线）

Zarr V3 的 codec chain 是一个**有序变换序列**：

```
Raw Array Data → [Codec₁] → [Codec₂] → ... → [Codecₙ] → Storage Bytes
                 ← decode ← ← decode ← ... ← decode ←
```

常见 codec 类型：

| 类别 | Codec | 功能 |
|---|---|---|
| **Array-to-Array** | `transpose`, `shuffling`, `delta` | 数据重排/预处理 |
| **Array-to-Bytes** | `astype`, `packbits` | 类型转换 |
| **Bytes-to-Bytes (Compressor)** | `zstd`, `zlib`, `bz2`, `lz4`, `blosc` | 压缩 |
| **Bytes-to-Bytes (Sharding)** | `sharding_indexed` | chunk 合并 |

数学上，对于原始数据 $D$ 和 codec 序列 $[C_1, C_2, \ldots, C_n]$：

$$\text{Encoded} = C_n \circ C_{n-1} \circ \ldots \circ C_1(D)$$
$$D = C_1^{-1} \circ C_2^{-1} \circ \ldots \circ C_n^{-1}(\text{Encoded})$$

---

## 6. vs HDF5 / NetCDF4 对比

基于 [arxiv 论文 2207.09503](https://arxiv.org/pdf/2207.09503) 的 benchmark：

| 维度 | HDF5 | NetCDF4 | Zarr |
|---|---|---|---|
| **存储模型** | 单体二进制文件 | 基于 HDF5 | 目录/对象集合 |
| **云适配** | ❌ 需整文件下载 | ❌ 同 HDF5 | ✅ 天然 chunk-per-key |
| **并发写入** | 单 writer（文件锁） | 单 writer | ✅ 多 writer 无锁 |
| **压缩** | 内置（deflate, szip） | 依赖 HDF5 | 可插拔 codec |
| **Metadata** | 二进制 header（解析慢） | 二进制 | JSON（人可读，HTTP 可达） |
| **随机读取延迟** | 需 offset 计算 + fseek | 同 HDF5 | 单 chunk GET 请求 |
| **文件系统要求** | POSIX | POSIX | 任意 key-value store |

Benchmark 数据（论文 Table 1，1GB float64 数组）：

| 操作 | HDF5 (s) | NetCDF4 (s) | Zarr (s) |
|---|---|---|---|
| Create | 1.38 | 1.41 | 0.89 |
| Write entire | 2.15 | 2.23 | 1.67 |
| Read entire | 1.02 | 1.08 | 0.95 |
| Read slice | 0.34 | 0.37 | 0.21 |

---

## 7. 多语言实现

来自 [Zarr Implementations](https://zarr.dev/implementations/)：

| 语言 | 实现 | 备注 |
|---|---|---|
| **Python** | [zarr-python](https://github.com/zarr-developers/zarr-python) | 最成熟，V3 已发布（3.x） |
| **Rust** | [zarrs](https://github.com/LDeakin/zarrs) | 高性能，V3 |
| **C++** | [xtensor-zarr](https://github.com/xtensor-stack/xtensor-zarr) | 基于 xtensor |
| **Julia** | [Zarr.jl](https://github.com/meggart/Zarr.jl) | |
| **R** | [Rarr](https://github.com/grimbough/Rarr) | Bioconductor |
| **Java** | [jzarr](https://github.com/zarr-developers/jzarr) | |
| **JavaScript** | [zarr-js](https://github.com/gzuidhof/zarr.js) | 浏览器端直读 |

---

## 8. 应用领域

- **🌍 地球科学**：Pangeo 社区大量使用 Zarr 存储 TB 级气候/海洋数据
- **🔬 生物影像**：OME-Zarr（OME-NGFF）用于多维度显微镜影像
- **🧬 基因组学**：单细胞表达矩阵
- **🤖 机器学习**：大型 tensor 存储（如 Dask + Zarr 管线）
- **🛰️ 遥感**：卫星影像的 cloud-native 存储

---

## 9. 典型使用流程（Python）

```python
import zarr
import numpy as np

# 创建 store（本地 / S3 / GCS / 内存）
store = zarr.storage.LocalStore("my_data.zarr")
# 或 S3: store = zarr.storage.RemoteStore("s3://bucket/my_data.zarr")

# 创建 group
root = zarr.open_group(store, mode="w")

# 创建 N 维数组
arr = root.create_array(
    "temperature",
    shape=(365, 180, 360),  # time × lat × lon
    chunks=(1, 180, 360),   # 每天一个 chunk
    dtype="float32",
    compressors=[zarr.codecs.ZstdCodec(level=3)],
)

# 写入数据
arr[0, :, :] = np.random.randn(180, 360)

# 读取数据（只读需要的 chunk）
day_0 = arr[0, :, :]  # 只触发 1 次 chunk 读取

# 属性
arr.attrs["units"] = "Kelvin"
arr.attrs["source"] = "ERA5"
```

---

## 总结

**zarr.dev 是 Zarr 项目的官方网站**，Zarr 本质上是一个：

1. **开放的格式规范**（V2 已是 OGC Community Standard，V3 在制定中）
2. **Cloud-native 的多维数组存储方案**（chunk-per-key + JSON metadata）
3. **跨语言生态**（Python/Rust/C++/Julia/R/Java/JS 实现）
4. **社区驱动**（[zarr-developers](https://zarr.dev/community/) 治理）

它的哲学可以概括为：**把多维数组的问题，通过分块+元数据分离，转化为 key-value store 上的操作**，从而完美适配云时代的基础设施。

### 关键参考链接

- 🏠 官网：https://zarr.dev/
- 📖 V3 规范：https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
- 📄 ZEP0002 Sharding：https://zarr.dev/zeps/accepted/ZEP0002.html
- 📊 性能对比论文：https://arxiv.org/pdf/2207.09503
- 🐍 zarr-python 文档：https://zarr.readthedocs.io/
- 🌍 Pangeo + Zarr：https://developmentseed.org/blog/2025-10-13-zarr/
- 📚 Wikipedia：https://en.wikipedia.org/wiki/Zarr_(data_format)