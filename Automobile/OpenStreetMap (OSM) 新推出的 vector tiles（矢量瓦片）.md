# OSM Vector Tiles Blog 深度解析

这篇博客由 **Mark Litwintschik** 撰写，主题是 **OpenStreetMap (OSM) 新推出的 vector tiles（矢量瓦片）** 的可视化与分析。核心叙事线是：OSM 从传统的 raster tiles（栅格瓦片）升级到 Mapbox Vector Tiles (MVT) 格式的矢量瓦片，作者一步步演示如何下载、解码、查询、转换这些数据。下面我从第一性原理出发，逐层拆解。

---

## 1. Raster Tiles vs. Vector Tiles：从第一性原理理解

### 1.1 Raster Tiles 的本质

传统 raster tile 本质上就是**预渲染的 PNG 图片**。每个 zoom level $z$ 下，世界地图被切割成 $2^z \times 2^z$ 个瓦片网格。每个瓦片是一个 256×256 像素（或 512×512）的**光栅图像**：

$$\text{Total Tiles}(z) = 4^z$$

| Zoom Level | 瓦片总数 | 覆盖范围 |
|-----------|---------|---------|
| 0 | 1 | 全球 |
| 10 | 1,048,576 | 区域级 |
| 14 | 268,435,456 | 街区级 |
| 18 | 68,719,476,736 | 建筑级 |

**Raster 的根本问题**：
- **样式锁定**（style lock-in）：渲染规则在服务端 bake-in，终端用户无法修改
- **分辨率固定**：放大后出现锯齿/模糊，因为像素是离散采样
- **数据不可查**：你拿到的是像素，不是结构化数据——无法查询某个 POI 的 `amenity` 或 `cuisine` 属性
- **多语言标签不可切换**：文字已 rasterize 进像素

### 1.2 Vector Tiles 的本质

MVT 格式的矢量瓦片是一种**Protocol Buffers 编码的二进制格式**（[Mapbox Vector Tile Specification v2.1](https://github.com/mapbox/vector-tile-spec/tree/master/2.1)）。核心思想是：

> **不在服务端渲染像素，而是传输几何与属性数据，让客户端决定如何渲染。**

每个 MVT 瓦片内部是一个类似 GeoJSON 的结构，但使用 **integer coordinates** 在一个 $extent \times extent$ 的网格空间内表示几何：

$$x_{\text{pixel}}, y_{\text{pixel}} \in [0, \text{extent}]$$

默认 $extent = 4096$（即 $2^{12}$）。这意味着每个瓦片内部有一个 4096×4096 的逻辑坐标系。

**MVT 的关键特性**：
- **样式可定制**：客户端通过 style sheet（如 MapLibre Style Spec）控制渲染
- **多语言标签**：属性中可含 `name`, `name_en`, `name_de`, `name_ar` 等
- **分辨率无关**：几何是矢量，缩放不失真
- **数据可查询**：每个 feature 都有结构化属性

---

## 2. 坐标变换的数学原理：Pixel → WGS84

博客中最有技术深度的部分之一是 `pixel2deg` 函数。让我从第一性原理推导：

### 2.1 Web Mercator Projection (EPSG:3857)

OSM 瓦片系统基于 **Spherical Mercator**（Web Mercator），其坐标变换公式如下：

**经度**（简单线性映射）：
$$\lambda = \frac{x_t + \frac{x_p}{E}}{2^z} \cdot 360° - 180°$$

其中：
- $x_t$ = tile 的 x 编号
- $x_p$ = 瓦片内的像素 x 坐标 $\in [0, E]$
- $E$ = extent（默认 4096）
- $z$ = zoom level

**纬度**（Mercator 逆变换，涉及反双曲函数）：
$$\phi = \arctan\left(\sinh\left(\pi \cdot \left(1 - \frac{2 \cdot y_t'}{2^z}\right)\right)\right)$$

其中 $y_t' = y_t + \frac{E - y_p}{E}$，这是因为 MVT 的 y 轴方向与地理纬度方向相反（MVT 的 y=0 在瓦片顶部，对应更高的纬度）。

博客中的 Python 实现正是这些公式的直接编码：

```python
n       = 2.0 ** zoom          # 2^z
xtile   = xtile + (xpixel / extent)  # 混合 tile 坐标与瓦片内坐标
ytile   = ytile + ((extent - ypixel) / extent)  # 注意 y 轴翻转
lon_deg = (xtile / n) * 360.0 - 180.0  # 经度线性映射
lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))  # Mercator 逆变换
lat_deg = math.degrees(lat_rad)
```

**注意**：Mercator 投影在极地区域有无穷畸变，因此纬度范围被裁剪到约 $\pm 85.05°$。

---

## 3. MVT 解码后的数据结构

使用 `mapbox_vector_tile` 库解码后，博客展示了 Dubai Burj Khalifa 区域 zoom 14 瓦片的 18 个 layer：

```
addresses, bridges, buildings, ferries, land, ocean,
pier_lines, pier_polygons, place_labels, pois,
public_transport, sites, street_labels, street_polygons,
streets, streets_labels_points, water_polygons,
water_polygons_labels
```

这个 layer 组织遵循 **Shortbread schema**（[OSM Shortbread](https://shortbread.tiles.openstreetmap.org/)），这是一个简化版的 OSM 数据模型，专为矢量瓦片设计。

### 3.1 POI 属性分析

`SUMMARIZE` 命令是 DuckDB 的数据探查利器，它对 `pois` layer 的 23 列进行了统计：

| 关键属性 | 非空率 | 近似唯一值 | 洞察 |
|---------|--------|-----------|------|
| `name` | 70.04% | 258 | 大多数 POI 有名字 |
| `amenity` | 68.14% | 27 | 27 种 amenity 类型 |
| `cuisine` | 14.14% | 39 | 只有 14% 的 POI 有菜系信息，但 67 条记录中有 39 种不同菜系（迪拜的多样性！） |
| `shop` | 16.24% | 19 | 商店类型 |
| `emergency` | 0.84% | 1 | 几乎全是 `fire_hydrant` |
| `geom` | 100% | 511 | 474 个 POI，但有 511 个近似唯一坐标（部分重合点） |

**null_percentage 是理解 OSM 数据质量的核心指标**——OSM 是 volunteered geographic information (VGI)，属性覆盖率高度不均匀。例如 `cuisine` 只有 14.14% 非空，意味着如果你想基于菜系做分析，数据稀疏性是主要挑战。

---

## 4. tiles2columns：从瓦片到分析就绪数据

这是博客的**最大贡献点**。作者构建了一个工具 [tiles2columns](https://github.com/marklit/tiles2columns)，将 MVT 瓦片转换为 **GeoPackage (.gpkg)** 或 **Parquet** 文件。

### 4.1 为什么这是重要的？

传统 OSM 数据消费路径是：

```
OSM Planet (PBF) → osm2pgsql → PostgreSQL/PostGIS → 分析
```

或者：

```
GeoFabrik PBF → osm_split → GeoPackage → 分析
```

但这些路径都有**数据延迟**（GeoFabrik 的 PBF 通常滞后数天）。

新路径是：

```
OSM Vector Tiles (MVT) → tiles2columns → GeoPackage/Parquet → 分析
```

**关键优势**：OSM 的 vector tiles 在数据被编辑后**很快更新**（分钟级而非天级），这意味着你可以获得近乎实时的 OSM 数据。

### 4.2 工作流程

```bash
python3 ~/tiles2columns/main.py bbox 55.2112 25.2745 55.34279 25.17104
```

这行命令做了以下事情：

1. **BBox → Tile List**：将 bounding box $(λ_{min}, φ_{max}, λ_{max}, φ_{min})$ 转换为覆盖该区域的 zoom 14 瓦片集合
2. **批量下载**：从 `https://vector.openstreetmap.org/shortbread_v1/{z}/{x}/{y}.mvt` 下载所有瓦片
3. **MVT 解码**：每个瓦片解码为 GeoJSON-like 结构
4. **属性提升**：MVT 中属性以 key-value pairs 存储（稀疏），工具将其转换为**列式存储**（columnar），每层一个文件
5. **写出 GeoPackage**：使用 OGC GeoPackage 标准（基于 SQLite），支持空间索引

### 4.3 输出文件的语义

```
6.8M streets.gpkg      ← 道路网络（线几何）
6.2M buildings.gpkg     ← 建筑轮廓（面几何）
1.4M pois.gpkg          ← 兴趣点（点几何）
1.4M street_labels.gpkg ← 街道标注
744K land.gpkg          ← 土地利用
372K sites.gpkg          ← 场地
248K addresses.gpkg     ← 地址点
...
```

**42 个瓦片 → ~18 MB 的结构化地理数据**。每个文件对应 MVT 中的一个 layer，直接可用作 GIS 分析的输入。

---

## 5. 技术栈全景图

博客涉及的技术栈值得梳理：

```
┌─────────────────────────────────────────┐
│           可视化层 (Visualization)        │
│  QGIS 3.40  │  Leafmap/MapLibre GL JS   │
├─────────────────────────────────────────┤
│           数据处理层 (Processing)          │
│  mapbox_vector_tile  │  morecantile     │
│  tiles2columns       │  jq              │
├─────────────────────────────────────────┤
│           存储层 (Storage)                │
│  GeoPackage (.gpkg)  │  Parquet          │
├─────────────────────────────────────────┤
│           查询层 (Query)                  │
│  DuckDB + Spatial/H3/JSON/Lindel        │
├─────────────────────────────────────────┤
│           数据源 (Source)                │
│  OSM Vector Tiles (MVT)                 │
│  https://vector.openstreetmap.org/      │
└─────────────────────────────────────────┘
```

### 5.1 DuckDB 生态

DuckDB 在这里是**轻量级空间分析引擎**，扩展组合很有讲究：

| 扩展 | 用途 |
|------|------|
| `spatial` | 空间函数（`ST_READ`, 几何操作） |
| `h3` | 离散全球网格系统，用于空间聚合 |
| `json` | 读写 JSON 文件 |
| `parquet` | 列式存储读写 |
| `lindel` | 空间填充曲线（Morton/Z-order curve），用于空间聚类索引 |

`lindel` 特别有趣——它让你可以用**一维排序**来近似**二维空间邻近**，这对于大数据集的空间查询优化非常重要：

$$\text{Z-order}(x, y) = \text{interleave\_bits}(x, y)$$

即把 $x$ 和 $y$ 的二进制位交错合并，使得空间上相邻的点在一维编码中也倾向于相邻。

### 5.2 morecantile

`morecantile` 是一个 Python 库，用于处理 **tile 网格系统**（TMS / TileMatrixSet）。博客中用它做 **BBox → Tile 坐标**的转换：

```bash
echo "[55.27, 25.2]" | morecantile tiles 14
[10707, 7006, 14]
```

这实际上是在求解：给定经纬度 $(λ, φ)$ 和 zoom level $z$，求 tile 坐标 $(x, y)$：

$$x = \lfloor \frac{λ + 180}{360} \cdot 2^z \rfloor$$

$$y = \lfloor \frac{1 - \ln(\tan(\phi) + \sec(\phi))}{2\pi} \cdot 2^z \rfloor$$

---

## 6. 关键洞察与延伸思考

### 6.1 矢量瓦片的压缩效率

博客提到一个 **114 KB 的 MVT 文件解码为 1.4 MB 的 JSON**。压缩比约为 **12:1**。这是因为：

- MVT 使用 Protocol Buffers（二进制编码，无冗余字段名）
- 几何使用 **delta encoding**：每个坐标存储与前一坐标的差值
- 属性 key/value 只在 layer 级别定义一次，feature 通过索引引用

### 6.2 Shortbread Schema 的设计哲学

OSM 的 [Shortbread](https://wiki.openstreetmap.org/wiki/Shortbread) 是一种**简化 schema**，与传统 OSM tag 体系相比：

| 特性 | OSM 原始 tags | Shortbread |
|------|-------------|-----------|
| 标签数量 | ~7,000+ keys | ~18 layers |
| 复杂度 | 任意嵌套 | 扁平化 |
| 目标 | 通用数据模型 | 专为渲染优化 |

这意味着 Shortbread **有意丢失了一些 OSM 的细粒度信息**，以换取瓦片的紧凑和渲染的高效。

### 6.3 实时数据管道的可能性

这篇博客隐含了一个极具潜力的架构：

```
OSM 编辑 → 分钟级同步 → MVT 瓦片更新 → tiles2columns 下载 
→ GeoPackage/Parquet → DuckDB 分析 → 仪表盘/报表
```

这比传统 GeoFabrik → PBF → osm2pgsql → PostgreSQL 管道快了**几个数量级**。

### 6.4 局限性

- **Zoom 14 的数据精度**：vector tiles 在不同 zoom level 会做 **generalization**（简化），zoom 14 的数据不是完整的 OSM 数据
- **属性稀疏**：如 `cuisine` 只有 14.14% 非空，不适合做全面的菜系分析
- **Shortbread 是子集**：不是所有 OSM tags 都被保留
- **QGIS 的 sprite 渲染 bug**：博客提到 QGIS 对 style sheet 中的图标渲染有问题

---

## 7. 参考资料

- [Mapbox Vector Tile Specification v2.1](https://github.com/mapbox/vector-tile-spec/tree/master/2.1)
- [OSM Shortbread Schema](https://wiki.openstreetmap.org/wiki/Shortbread)
- [OSM Vector Tiles Demo](https://shortbread.tiles.openstreetmap.org/)
- [tiles2columns - GitHub](https://github.com/marklit/tiles2columns)
- [morecantile - GitHub](https://github.com/developmentseed/morecantile)
- [DuckDB Spatial Extension](https://duckdb.org/docs/extensions/spatial)
- [DuckDB H3 Extension](https://duckdb.org/docs/extensions/h3)
- [QGIS Vector Tile Support](https://www.qgis.org/en/site/forusers/visualchangelog314/index.html#feature-qgis-now-supports-vector-tile-layers)
- [MapLibre Style Spec](https://maplibre.org/maplibre-style-spec/)
- [GeoPackage Standard (OGC)](https://www.ogc.org/standards/geopackage)
- [Web Mercator Projection - EPSG:3857](https://epsg.io/3857)
- [作者原始博客](https://marklit.github.io/)（Mark Litwintschik 的技术博客）

---

**总结**：这篇博客的实质是从**数据消费**的视角，展示了 OSM 从 raster 到 vector 的范式转换。Raster tiles 的世界是"服务端画好图，客户端只看"；Vector tiles 的世界是"服务端提供数据，客户端既看又查"。而 `tiles2columns` 工具更进一步，把"看和查"的能力延伸到了**离线分析**场景——你可以用 42 个 MVT 瓦片（几分钟下载）就获得一个结构化的 Dubai 地理数据集，而不需要下载数 GB 的 OSM Planet PBF 再花数小时导入 PostGIS。这是一个**数据获取民主化**的重要步骤。