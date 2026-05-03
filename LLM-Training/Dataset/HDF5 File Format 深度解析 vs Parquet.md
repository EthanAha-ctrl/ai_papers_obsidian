# HDF5 File Format 深度解析 vs Parquet

## 一、HDF5 的第一性原理理解

### 1.1 核心设计哲学

HDF5 (Hierarchical Data Format version 5) 的本质是一个 **self-describing, portable, extensible scientific data format**。从第一性原理出发，它解决的核心问题是：

> **如何在 single file 中高效存储、组织、访问 heterogeneous multi-dimensional data？**

其设计基于三个 fundamental principles：

1. **Hierarchical Organization** - 类似 filesystem 的树状结构
2. **Self-Description** - metadata 与 data 共存
3. **I/O Efficiency** - 支持 partial read/write，无需加载整个 file

### 1.2 HDF5 File 的逻辑结构

```
HDF5 File (类似于 Unix filesystem)
│
├── Group (类似于 directory)
│   ├── Group
│   │   ├── Dataset (类似于 file，存储实际数据)
│   │   └── Dataset
│   └── Dataset
│
├── Group
│   └── Dataset
│
└── Dataset
```

**核心对象类型：**

| Object Type | Analogy | Function |
|-------------|---------|----------|
| **Group** | Directory/Folder | Container for organizing objects |
| **Dataset** | File | Multidimensional array of homogeneous data |
| **Attribute** | Metadata | Small metadata attached to Group/Dataset |
| **Link** | Symbolic link | Reference to another object |

### 1.3 HDF5 Physical Storage Architecture

HDF5 file 的物理结构：

```
┌─────────────────────────────────────────────────────────────┐
│                     HDF5 File Structure                      │
├─────────────────────────────────────────────────────────────┤
│  Superblock (File signature + root group pointer)           │
├─────────────────────────────────────────────────────────────┤
│  File Driver Layer (Virtual File Layer - VFL)               │
├─────────────────────────────────────────────────────────────┤
│  Metadata Cache (B-tree structures, object headers)         │
├─────────────────────────────────────────────────────────────┤
│  Data Object Layer                                           │
│  ├── Object Headers (attribute info, dataset properties)    │
│  ├── B-trees (indexing for chunked datasets)                │
│  └── Global Heap (shared objects)                           │
├─────────────────────────────────────────────────────────────┤
│  Raw Data Storage (contiguous or chunked)                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 Chunked Storage 机制（关键性能特性）

HDF5 支持 **chunked storage**，这是其 I/O efficiency 的核心：

```
Dataset: 1000 × 1000 × 100 float32 array

┌─────────────────────────────────────────┐
│  Logical View (User Perspective)        │
│  3D array [1000][1000][100]             │
└─────────────────────────────────────────┘
              ↓ Chunking
┌─────────────────────────────────────────┐
│  Physical Storage (Chunks on Disk)      │
│  Each chunk: 100×100×10 elements        │
│  Total chunks: 10×10×10 = 1000 chunks   │
└─────────────────────────────────────────┘
```

**Chunk 的数学表示：**

设 dataset 维度为 $D = (d_1, d_2, ..., d_n)$，chunk 大小为 $C = (c_1, c_2, ..., c_n)$

$$N_{chunks} = \prod_{i=1}^{n} \lceil \frac{d_i}{c_i} \rceil$$

其中：
- $N_{chunks}$ = chunk 总数
- $d_i$ = 第 $i$ 维的大小
- $c_i$ = 第 $i$ 维的 chunk 大小
- $\lceil \cdot \rceil$ = 向上取整函数

**Partial I/O 公式：**

当读取一个 hyperslab（超切片）$R = (r_{start}, r_{end})$ 时：

$$Data\_transferred = \sum_{chunks\_intersecting\_R} chunk\_size$$

这比读取整个 dataset 大幅减少 I/O。

### 1.5 Hyperslab Selection（超切片选择）

HDF5 支持 hyperslab selection，允许用户选择 dataset 的任意子区域：

```python
# 概念示意
dataset[10:100, 50:200, 5:15]  # 选择一个 hyperslab
```

Hyperslab 由四个参数定义（每维）：
- **Start**: 起始位置 $s_i$
- **Stride**: 步长 $t_i$  
- **Count**: 选取数量 $n_i$
- **Block**: 每个选取块的长度 $b_i$

选取的元素索引：

$$I = \{s_i + k \cdot t_i + m \mid 0 \leq k < n_i, 0 \leq m < b_i\}$$

## 二、Parquet File Format 深度解析

### 2.1 核心设计哲学

Parquet 是 **columnar storage format**，专为 **analytical workloads** 优化。其第一性原理：

> **如何最大化 OLAP (Online Analytical Processing) 查询的 read 性能和 storage efficiency？**

核心 principles：
1. **Columnar Storage** - 同列数据连续存储
2. **Predicate Pushdown** - 谓词下推，减少数据读取
3. **Compression Efficiency** - 同类数据压缩比更高
4. **Projection Pushdown** - 只读取需要的列

### 2.2 Parquet 物理结构

```
┌─────────────────────────────────────────────────────────────┐
│                    Parquet File Structure                    │
├─────────────────────────────────────────────────────────────┤
│  Magic Number "PAR1"                                         │
├─────────────────────────────────────────────────────────────┤
│  Row Group 1                                                 │
│  ├── Column Chunk (Column A)                                │
│  │   ├── Page 1 (Dictionary Page)                          │
│  │   ├── Page 2 (Data Page - RLE encoded)                  │
│  │   └── Page 3 (Data Page)                                │
│  ├── Column Chunk (Column B)                                │
│  └── Column Chunk (Column C)                                │
├─────────────────────────────────────────────────────────────┤
│  Row Group 2                                                 │
│  └── ...                                                     │
├─────────────────────────────────────────────────────────────┤
│  File Metadata (Schema, Row Group locations, Statistics)    │
├─────────────────────────────────────────────────────────────┤
│  Footer Length (4 bytes)                                     │
├─────────────────────────────────────────────────────────────┤
│  Magic Number "PAR1"                                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Columnar Storage 的数学优势

假设一个表有 $R$ 行 $C$ 列，每列平均大小为 $S$ bytes：

**Row-based storage (如 CSV)：**
$$Read\_cost_{row} = R \times C \times S$$

**Column-based storage (Parquet)：**
$$Read\_cost_{column} = R \times S$$

当查询只涉及 1 列时：

$$Compression\_ratio = \frac{Read\_cost_{row}}{Read\_cost_{column}} = C$$

### 2.4 Parquet Encoding & Compression

Parquet 支持多种 encoding：

| Encoding | 适用场景 | 原理 |
|----------|---------|------|
| **PLAIN** | 通用数据 | 直接存储原始值 |
| **DICTIONARY** | 低基数列 | 构建 dictionary，存储 index |
| **RLE** (Run-Length Encoding) | 重复值多 | 存储 |
| **DELTA_BINARY_PACKED** | 整数序列 | 存储 delta values |
| **BYTE_STREAM_SPLIT** | 浮点数 | 分离字节提高压缩 |

**Dictionary Encoding 示例：**

原始数据：`["apple", "banana", "apple", "apple", "banana"]`

Dictionary：`{0: "apple", 1: "banana"}`

Encoded：`[0, 1, 0, 0, 1]`

压缩比计算：

$$Compression\_ratio_{dict} = \frac{n \times avg\_string\_length}{dict\_size + n \times \lceil log_2(k) \rceil / 8}$$

其中 $n$ = 元素数量，$k$ = distinct values 数量

### 2.5 Predicate Pushdown 机制

Parquet 的 Statistics (min/max/null_count) 存储在每个 Page 和 Column Chunk 的 metadata 中：

```
Column Chunk Metadata:
├── total_size: 1024000
├── num_values: 100000
├── statistics:
│   ├── min: 42
│   ├── max: 9999
│   ├── null_count: 5
│   └── distinct_count: N/A
```

当执行查询 `WHERE column_x > 10000` 时：

若 $max = 9999 < 10000$，则 **整个 Column Chunk 可以跳过**！

## 三、HDF5 vs Parquet 详细对比

### 3.1 设计目标对比

| Aspect | HDF5 | Parquet |
|--------|------|---------|
| **Primary Use Case** | Scientific computing, HPC, Multi-dimensional arrays | Data analytics, OLAP, Data lakes |
| **Data Model** | Hierarchical (Groups + Datasets) | Tabular (Row Groups + Columns) |
| **Optimization Target** | Partial I/O on arrays | Columnar analytical queries |
| **Ecosystem** | Python (h5py), C/C++, Fortran, MATLAB | Spark, Pandas, DuckDB, Arrow |

### 3.2 数据模型对比

**HDF5 Data Model：**

```
Hierarchical Structure:
/
├── /experiment_1/
│   ├── /measurements/
│   │   ├── temperature: Dataset [1000, 500] float32
│   │   └── pressure: Dataset [1000, 500] float32
│   └── /metadata/
│       └── config: Attribute {key: value}
├── /experiment_2/
│   └── ...
└── /simulations/
    └── result: Dataset [100, 100, 1000] float64
```

**Parquet Data Model：**

```
Flat Tabular Structure:
Schema:
├── id: INT64
├── timestamp: TIMESTAMP
├── sensor_id: INT32
├── temperature: FLOAT
├── pressure: FLOAT
└── location: STRUCT<lat: FLOAT, lon: FLOAT>
```

### 3.3 性能对比实验

以下是一个典型 benchmark（模拟数据）：

**实验配置：**
- Dataset: 1亿行 × 100列 float32
- File size: ~40 GB (uncompressed)
- Task: 读取单列全部数据

| Metric | HDF5 | Parquet |
|--------|------|---------|
| **Read Time (single column)** | 8.2s | 1.1s |
| **Read Time (all columns)** | 12.5s | 35.8s |
| **File Size (snappy compressed)** | 42 GB | 15 GB |
| **Random Access Latency** | ~0.5ms | ~5ms |
| **Write Speed** | 150 MB/s | 80 MB/s |

**解释：**

- Parquet 在 single column read 场景下显著优于 HDF5（因为 columnar layout）
- HDF5 在全量读取和 random access 场景下更优
- Parquet 的 compression efficiency 更高

### 3.4 Compression 对比

**HDF5 Compression Filters：**

HDF5 使用 **filter pipeline**：

```
Data → Filter 1 → Filter 2 → ... → Filter N → Disk
```

常用 filters：
- **DEFLATE (gzip)**: 通用压缩
- **SZIP**: NASA 开发，适合科学数据
- **LZF**: 高速压缩
- **BZIP2**: 高压缩比
- **ZSTD**: 平衡速度和压缩比

**Parquet Compression：**

Parquet 的 compression 在 Page 级别应用：
- **SNAPPY**: 默认，高速
- **GZIP**: 高压缩比
- **LZ4**: 极速
- **ZSTD**: 平衡

### 3.5 并行处理能力

**HDF5 Parallel I/O：**

HDF5 支持 **MPI-IO**，适用于 HPC 场景：

```python
# Parallel HDF5 (概念示意)
import h5py
from mpi4py import MPI

# 每个 MPI rank 独立写入不同 region
with h5py.File('parallel.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
    dset = f.create_dataset('data', (1000, 1000), dtype='f')
    # 每个 rank 写入自己的 hyperslab
    dset[rank*100:(rank+1)*100, :] = local_data
```

**Parquet Parallel Processing：**

Parquet 设计上支持 distributed processing：

```python
# Spark 读取 Parquet (概念示意)
df = spark.read.parquet('s3://bucket/data/')
# 自动并行读取不同 Row Groups
result = df.filter(df.column_x > 100).select('column_y')
```

**Row Group 大小优化：**

最优 Row Group 大小 $RG_{opt}$：

$$RG_{opt} \approx \frac{HDFS\_block\_size}{compression\_ratio}$$

通常建议 128MB - 1GB per Row Group。

## 四、深入技术细节

### 4.1 HDF5 B-tree 索引结构

HDF5 使用 B-tree 索引 chunked datasets：

```
B-tree Structure for Chunked Dataset:
                    ┌─────────────────┐
                    │   Root Node     │
                    │  [chunk_addr]   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌──────▼──────┐ ┌───────▼───────┐
    │  Leaf Node 1  │ │ Leaf Node 2 │ │  Leaf Node 3  │
    │ [chunk_0_idx] │ │ [chunk_1]   │ │ [chunk_2]     │
    │ [chunk_0_ptr] │ │ [chunk_1]   │ │ [chunk_2]     │
    └───────────────┘ └─────────────┘ └───────────────┘
```

**Chunk 查找复杂度：**

对于 $N$ 个 chunks，B-tree 高度 $h$：

$$h = \lceil log_k(N) \rceil$$

其中 $k$ = B-tree 的 fanout（通常 > 100）

时间复杂度：$O(log N)$

### 4.2 Parquet Statistics 下推深度

Parquet 的 statistics 在多个层级：

```
File Level:
├── Row Group 1
│   ├── Column Chunk A
│   │   ├── Page 1 [min=1, max=100]
│   │   ├── Page 2 [min=101, max=200]
│   │   └── Page 3 [min=201, max=300]
│   └── Statistics: [min=1, max=300]
├── Row Group 2
│   └── Statistics: [min=301, max=600]
└── File Statistics: [min=1, max=600]
```

**Query: `WHERE A > 250`**

跳过的数据：
- Row Group 1: 部分（Page 1, 2 跳过）
- Row Group 2: 全部读取

### 4.3 HDF5 Datatype 系统

HDF5 支持丰富的 datatype：

**Atomic Types：**
- Integer: signed/unsigned, 1-8 bytes
- Float: IEEE 754, 4/8/16 bytes
- String: fixed/variable length
- Bitfield: arbitrary bit width
- Opaque: raw bytes
- Time: date/time representation

**Compound Types (类似于 struct)：**

```python
# HDF5 Compound Type 示例
dt = np.dtype([
    ('id', 'i4'),           # 4-byte integer
    ('name', 'S20'),        # 20-byte string
    ('value', 'f8'),        # 8-byte float
    ('nested', [
        ('x', 'f4'),
        ('y', 'f4')
    ])
])
```

### 4.4 Parquet Nested Data Support

Parquet 使用 **Dremel encoding** 处理 nested data：

**Schema:**
```
message Document {
  required int64 DocId;
  optional group Links {
    repeated int64 Backward;
    repeated int64 Forward;
  }
  optional group Name {
    repeated group Language {
      required string Code;
      optional string Country;
    }
  }
}
```

**Definition Level 和 Repetition Level：**

| Value | Definition Level | Repetition Level |
|-------|------------------|------------------|
| DocId=10 | 0 | 0 |
| Links.Forward=20 | 2 | 0 |
| Links.Forward=40 | 2 | 1 |
| Links.Backward=NULL | 1 | 0 |
| Name.Language.Code='en' | 4 | 0 |

- **Definition Level**: 定义值的嵌套深度（用于表示 NULL）
- **Repetition Level**: 定义值在哪个层级重复

## 五、适用场景深度分析

### 5.1 选择 HDF5 的场景

```
✓ Multi-dimensional array data (2D+ images, 3D+ volumes)
✓ High-performance computing (HPC) workloads
✓ Need for hierarchical organization
✓ Partial I/O on large arrays (read/write subsets)
✓ Complex metadata attached to datasets
✓ MPI parallel I/O requirements
✓ Legacy scientific code (Fortran, MATLAB)
✓ Streaming data acquisition
```

**典型应用：**
- Climate simulation output (NetCDF4 基于 HDF5)
- Genomics data (DNA sequences)
- Neutron scattering data
- Astronomy image cubes
- Medical imaging (DICOM subsets)
- Machine learning model weights storage

### 5.2 选择 Parquet 的场景

```
✓ Analytical queries on tabular data (OLAP)
✓ Column-level aggregations and filtering
✓ Data lake / Data warehouse architecture
✓ Big data processing (Spark, Presto, DuckDB)
✓ Time-series data with regular schema
✓ Cloud storage optimization (S3, GCS, ADLS)
✓ Arrow ecosystem integration
✓ Schema evolution requirements
```

**典型应用：**
- Clickstream analytics
- Financial transaction logs
- IoT sensor data (tabular)
- User behavior analytics
- Data warehouse fact tables
- Feature stores for ML

### 5.3 混合场景

某些场景可能需要两者结合：

```
Example: Satellite Data Pipeline

Raw Satellite Images (HDF5)
    ↓
    Processing
    ↓
Extracted Features Table (Parquet)
    ↓
    Analytics/ML
```

## 六、性能优化技术细节

### 6.1 HDF5 Chunk Size 优化

**最优 chunk 大小计算：**

$$Chunk_{opt} \approx \frac{Disk\_page\_size}{element\_size}$$

通常建议：
- Minimum: 10 KB - 100 KB
- Maximum: 1 MB - 4 MB
- Target: 100 KB - 1 MB

**公式：**

$$Read\_efficiency = \frac{Useful\_data}{Total\_data\_read} = \frac{Selection\_size}{Chunks\_intersected \times Chunk\_size}$$

### 6.2 HDF5 Compression Filter Pipeline

```python
# 优化 filter pipeline
with h5py.File('optimized.h5', 'w') as f:
    # 创建 dataset 时指定 filter pipeline
    dset = f.create_dataset(
        'data',
        shape=(10000, 10000),
        dtype='float32',
        chunks=(100, 100),
        compression='gzip',      # Filter 1: compression
        compression_opts=4,       # Compression level (1-9)
        shuffle=True,             # Filter 2: byte shuffle
        fletcher32=True           # Filter 3: checksum
    )
```

**Filter Pipeline 效果：**

```
原始数据: float32 array
    ↓ Shuffle Filter (reorder bytes)
    ↓ GZIP Compression
    ↓ Fletcher32 Checksum
磁盘数据
```

**Shuffle Filter 原理：**

对于 float32 数组 `[1.1, 1.2, 1.3, 1.4]`：

原始 bytes：
```
Byte 0: 0x3F 0x8C 0xCC 0xCD  (1.1)
Byte 1: 0x3F 0x99 0x99 0x9A  (1.2)
...
```

Shuffle 后（按 byte position 重组）：
```
Byte 0 position: 0x3F 0x3F 0x3F 0x3F ... (更易压缩!)
Byte 1 position: 0x8C 0x99 ...
...
```

### 6.3 Parquet 优化技术

**Dictionary Encoding 判断公式：**

当 cardinality $k$ 满足：

$$k < \frac{n \times element\_size}{element\_size + \lceil log_2(k) \rceil / 8}$$

时，Dictionary encoding 有益。

**Page Size 优化：**

$$Page_{opt} \approx \frac{L1\_cache\_size}{2}$$

通常 8 KB - 1 MB。

**Bloom Filter 加速：**

Parquet 支持 Bloom Filter 用于 point queries：

```python
# Spark 中启用 Bloom Filter
spark.conf.set("parquet.bloom.filter.enabled", "true")
spark.conf.set("parquet.bloom.filter.expected.ndv", "1000000")
```

Bloom Filter 误判率：

$$P_{false\_positive} \approx (1 - e^{-kn/m})^k$$

其中：
- $n$ = 元素数量
- $m$ = bit array 大小
- $k$ = hash 函数数量

## 七、Ecosystem & Tooling 对比

### 7.1 HDF5 Ecosystem

| Tool/Library | Language | Description |
|--------------|----------|-------------|
| **h5py** | Python | 最流行的 Python HDF5 接口 |
| **PyTables** | Python | 高级 HDF5 操作，支持 table |
| **HDF5 C Library** | C | Official reference implementation |
| **HDF5 Fortran** | Fortran | Fortran bindings |
| **NetCDF4** | Multiple | Climate data format (HDF5-based) |
| **MATLAB HDF5** | MATLAB | Built-in HDF5 support |
| **HDFView** | Java | GUI viewer for HDF5 files |
| **h5dump** | CLI | Command-line tool for inspection |

### 7.2 Parquet Ecosystem

| Tool/Library | Language | Description |
|--------------|----------|-------------|
| **Apache Spark** | Scala/Python | Native Parquet support |
| **PyArrow** | Python | Arrow + Parquet I/O |
| **DuckDB** | C++/Python | Embedded analytical DB |
| **Pandas** | Python | `read_parquet()`, `to_parquet()` |
| **Dask** | Python | Parallel Parquet processing |
| **Trino/Presto** | Java | Distributed SQL engine |
| **Apache Drill** | Java | Schema-free SQL engine |
| **parquet-tools** | CLI | Command-line inspection |

### 7.3 Interoperability

**HDF5 → Parquet 转换示例：**

```python
import h5py
import pandas as pd
import pyarrow.parquet as pq

# Read HDF5
with h5py.File('data.h5', 'r') as f:
    df = pd.DataFrame({
        'col1': f['dataset1'][:, 0],
        'col2': f['dataset1'][:, 1],
    })
    # Add metadata
    attrs = dict(f['dataset1'].attrs)

# Write Parquet
df.to_parquet('data.parquet', engine='pyarrow')

# Add custom metadata
parquet_file = pq.ParquetFile('data.parquet')
```

## 八、Reference Links

**HDF5 Official Resources:**
- HDF Group Official: https://www.hdfgroup.org/solutions/hdf5/
- HDF5 Documentation: https://docs.hdfgroup.org/hdf5/v1_14/
- HDF5 File Format Spec: https://support.hdfgroup.org/HDF5/doc/H5.format.html
- h5py Documentation: https://docs.h5py.org/en/stable/

**Parquet Official Resources:**
- Apache Parquet: https://parquet.apache.org/
- Parquet Format Spec: https://github.com/apache/parquet-format
- PyArrow Parquet: https://arrow.apache.org/docs/python/parquet.html
- Dremel Paper: https://research.google/pubs/pub36632/

**Comparative Analysis:**
- HDF5 vs Parquet Discussion: https://stackoverflow.com/questions/tagged/hdf5+parquet
- Scientific Data Formats: https://www.hdfgroup.org/why-hdf/

## 九、总结

| Aspect | HDF5 | Parquet |
|--------|------|---------|
| **Best for** | Multi-dimensional arrays, scientific computing | Tabular analytics, data lakes |
| **Structure** | Hierarchical (filesystem-like) | Flat columnar |
| **Read Pattern** | Random access, hyperslab | Column scan, predicate pushdown |
| **Compression** | Filter pipeline (flexible) | Page-level (integrated) |
| **Parallelism** | MPI-IO (HPC) | Distributed (Spark/Dask) |
| **Metadata** | Rich attributes on any object | Schema + statistics |
| **Learning Curve** | Steeper | Easier (SQL-like) |

**Intuitive Decision Framework:**

```
                    Data Structure?
                         │
            ┌────────────┴────────────┐
            │                         │
    Multi-dimensional          Tabular/Relational
            │                         │
            ▼                         ▼
        HDF5                     Parquet
            │                         │
            │                    Query Pattern?
            │                         │
            │              ┌──────────┴──────────┐
            │              │                     │
            │        Column-heavy          Row-heavy
            │              │                     │
            │              ▼                     ▼
            │         Parquet ✓            Consider CSV/Avro
            │
       Access Pattern?
            │
     ┌──────┴──────┐
     │             │
  Random I/O   Sequential
     │             │
     ▼             ▼
   HDF5 ✓      Both OK
```

选择的核心原则：**Match the format to your access pattern and data model, not the other way around.**