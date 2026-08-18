---
source_pdf: Fast Tetrahedral Meshing in the Wild.pdf
paper_sha256: 94711fa5d370d382fe1997d8c8650b6d50406ce85477be5ca0c15d94904d2ded
processed_at: '2026-08-18T12:44:12-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 fTetWild

## 要解决的 problem

你手头有一个 3D model 的 surface mesh（一堆三角形拼成的外壳），比如从 Thingiverse 下载的、CT 扫描的、或者 CAD 导出来的。你想做物理仿真（受力变形、热传导、流体），但 simulation 软件要的是 **volumetric tetrahedral mesh**——内部也得填满小四面体。

麻烦在于，你的输入很脏。有 self-intersections、有 gaps、有零面积的 degenerate triangles、有 duplicate vertices、有 open boundaries。传统 mesher（TetGen、CGAL）假设输入是干净的 closed manifold，遇到脏数据直接 crash 或者产出 garbage。

## TetWild 的功劳

2018 年的 [TetWild](https://doi.org/10.1145/3197517.3201353) 把这个问题解决了。核心 idea 特别简单：**允许 output surface 偏离 input surface 一点点**，在一个叫 ϵ-envelope 的薄壳层里就行。这样自相交、小裂缝这类毛病，只要在 envelope 范围内，自动被"抹平"——反正允许偏离嘛。

可以想象成给 input 套了一层海绵套，output 只要待在这层套里，脏数据自动 heal。

但 TetWild 付了两个代价：

1. 为了数学上的 exactness，内部全程用 **rational number**（分数）做计算。Rational 很慢——分子分母会膨胀，加减乘除都比 float 慢一两个数量级，而且内存吃得多。
2. 理论上，最后把 rational 坐标 round 回 float 时，可能 mesh 就不 valid 了（虽然 10000 个 model 上从没发生过）。

## fTetWild 的 core insight

fTetWild 问了一个很好的 question：**能不能全程用 float，完全不要 rational？**

这听起来很难。Float 有 round-off error，会让拓扑判断翻转——比如判断一个点在三角形里面还是外面，算错一个 sign 就整个 mesh 翻转了，这就是几何处理里经典的 robustness nightmare。

fTetWild 的解法思路很清晰：**一次只插入一个 input triangle，插不好就 rollback。**

具体流程：

- 先用 Delaunay 生成一个 background tet mesh（不 conform input，只是个起点）
- 然后一个一个把 input triangle "塞"进去
- 塞的时候，**拓扑决策用 exact predicates**（[Shewchuk 那套](https://www.cs.cmu.edu/~quake/robust.html)）——判断 in/out、orientation、cut detection 必须精确
- **几何构造用 float**，允许 snapping：如果某个交点离一个现有 vertex 很近（距离 < δ），直接 snap 过去，不强求精确相交
- 如果某次 insertion 产生了 inverted 或 zero-volume 的 tet，就 rollback，标记这个 triangle "待会儿再试"
- 跑几轮 mesh improvement（让 tet 形状变好）后，重新 attempt 那些失败的 triangle

**Key insight：拓扑决策必须 exact（用 predicates），但几何构造可以 inexact（用 float + tolerance），因为 envelope 已经允许了 ϵ 的偏差。** 既然允许偏离，那 δ-snapping 内的误差都合法。这是一种 "exact topology + inexact geometry + tolerance buffer" 的 hybrid。

## 为什么快

- 去掉 rational，全程 float，快 10-100×
- Incremental insertion 可以 fail-fast rollback，避免 TetWild 那种 BSP 一次性切所有 plane 导致的 polyhedral mesh 爆炸
- Preprocessing 的 edge collapse 和 mesh smoothing 都并行化了（graph coloring 分 independent set）

结果：比 TetWild 快 7×（平均 49.8s vs 360s），和 TetGen 这种 Delaunay-based 的不 robust 方法差不多快（18.5s vs 22s，在公平子集上甚至更快）。

## 代价是什么

fTetWild 放弃了 TetWild 的理论保证："保证插入所有 input triangle"。如果某个 triangle 死活插不进去，fTetWild 就输出一个缺了它的 mesh（仍然 valid，仍然在 envelope 内）。

但实践上，在 [Thingi10k](https://github.com/alecjacobson/thingi10k) 的 10000 个真实 model 上，**从未发生**过 insertion 失败。所以这个 trade-off 在实用中基本是 free lunch——理论上的"可能失败"在实践中近乎不可能，因为 envelope 给了足够 slack，snapping 处理了 near-degenerate，re-attempt 机制让 mesh 改善后总能成功。

## 一个特别精巧的小细节

他们发现 AMIPS energy（衡量 tet 质量的 metric）在 float 下计算时，对**同一个 tet 的 4 个 vertex 不同排列顺序**会给出差 2 个数量级的不同结果！

Energy 越大这个差异越大。当 energy ~ 10⁸ 时，float 计算的 energy 在不同 permutation 下从 5e10 到 2e11 跳，而真正的值（rational 算）是 1.13e11。这意味着 gradient descent 用的 descent direction 可能是错的——以为在下降，实际在上升。后果是 algorithm 不断 refine 试图改善 quality，但 quality 指标本身就不靠谱，导致 over-refinement 灾难。

修复特别 elegant：当 energy > 10⁸ 时（rare case），把 energy 三次方（变成完全 rational 的表达式 $AMIPS^3 = (\text{tr}(J^T J))^3 / \det(J)^2$），用 rational 算，再 round 回 float 取 cube root。因为 det 平方了，符号不会翻转，且 rational 保证 permutation invariant。Rare case 用 rational，common case 用 float——这就是 hybrid arithmetic 的精髓，既快又准。

## 设计哲学的一句话总结

fTetWild = TetWild 的 robustness + Delaunay 的速度，靠的是 **"incremental insertion + snapping tolerance + rollback + hybrid arithmetic"** 这套组合拳。

核心哲学：**拓扑要 exact，几何可以 fuzzy，失败就 retry。** 这其实是 robust algorithm design 的通用范式——topological/qualitative decision 必须准，quantitative value 可以有 tolerance，failure 不 fatal 只要能 retry。

---

如果想再深入看技术细节，[fTetWild 的 code](https://github.com/wildmeshing/fTetWild) 是开源的，C++ 实现，依赖 [Geogram](http://alice.loria.fr/index.php/software/4-library/75-geogram.html) 的 exact predicates 和 [Eigen](https://eigen.tuxfamily.org/) 做线性代数。Paper 本身在 [ACM DL](https://doi.org/10.1145/3386569.3392369) 上。

---

# fTetWild: Fast Tetrahedral Meshing in the Wild 深度技术讲解

## 1. Paper 的核心定位与历史脉络

这篇 paper 是 NYU 几何处理组（Daniele Panozzo, Denis Zorin 等）对 2018 年 SIGGRAPH paper [TetWild](https://doi.org/10.1145/3197517.3201353) 的迭代升级。TetWild 本身已经是一个里程碑式的 robust tetrahedral meshing algorithm，能够处理 "in the wild" 的 imperfect triangle soups（带 self-intersections、gaps、degenerate faces 的输入），但有两个痛点：(1) 内部使用 rational arithmetic 导致速度慢；(2) 理论上不能保证 rounding 到 floating-point 后 mesh 仍然 valid。fTetWild 用一种全新的 incremental triangle insertion 思路同时解决了这两个问题。

相关的代码仓库：
- fTetWild 官方实现：https://github.com/wildmeshing/fTetWild
- TetWild 原始实现：https://github.com/wildmeshing/TetWild
- TriWild（2D 对应版本）：https://github.com/wildmeshing/TriWild
- Thingi10k dataset：https://github.com/alecjacobson/thingi10k
- Geogram（Lévy 提供 Delaunay 和 exact predicates）：http://alice.loria.fr/index.php/software/4-library/75-geogram.html

---

## 2. 问题背景：为什么 Tetrahedral Meshing 很难

### 2.1 应用场景

Tetrahedral mesh 是 FEM（finite element method）仿真、物理动画、medical imaging 的基础数据结构。给定一个 surface triangle mesh（物体表面），要生成填充其内部 volume 的 tetrahedra 集合，且要求：
- 每个 tetrahedron 有 positive volume（non-inverted）
- 没有 degenerate（sliver）elements
- Boundary faces conform 到 input surface
- Mesh quality 足够好（用于 FEM 数值稳定性）

### 2.2 "In the Wild" 的挑战

真实世界的 3D model（来自 CAD、扫描、3D printing repository 如 Thingiverse）通常有：
- Self-intersections（自相交）
- Gaps（裂缝、孔洞）
- Degenerate triangles（零面积或近零面积）
- Non-manifold edges/vertices
- Duplicate vertices
- Open boundaries（不闭合的表面）

传统方法（[TetGen](https://www.wias-berlin.de/software/tetgen/)、[CGAL](https://www.cgal.org/) 的 3D mesh generation）假设输入是 closed manifold non-self-intersecting mesh，遇到这些 imperfection 就会 crash 或产生 garbage output。TetWild/fTetWild 的核心洞察是：**允许 output surface 在 input surface 周围的 ϵ-envelope 内小幅偏离，从而自动 heal 这些 imperfections**。

---

## 3. fTetWild 算法架构总览

```
Input: triangle soup S, target edge length ℓ, envelope size ϵ
  │
  ▼
[Phase 1] Preprocessing
  - Merge vertices closer than ϵ_zero
  - Parallel edge collapse (2-coloring) staying in ϵ_prep = 0.8ϵ envelope
  │
  ▼
[Phase 2] Background Mesh + Incremental Triangle Insertion
  - Delaunay tetrahedralization of bounding box (expanded by 2ϵ)
  - For each input triangle T:
      * Find cut tetrahedra set T_I
      * Snapping-based plane-tet intersection (tolerance δ)
      * Table-based tetrahedron subdivision
      * If any sub-tet has volume < ϵ_zero^3: rollback, mark un-inserted
  │
  ▼
[Phase 3] Mesh Improvement (interleaved with re-insertion)
  - Every 3 iterations: retry inserting un-inserted triangles
  - Local ops: edge split, edge collapse, edge swap, vertex smoothing
  - Optimize conformal AMIPS energy (hybrid float/rational evaluation)
  - Parallel smoothing via graph coloring
  - Stop when max AMIPS < 10 or iterations = 80
  │
  ▼
[Phase 4] Filtering
  - Fast winding number classification (inside/outside)
  - OR: mesh arrangement for Boolean operations
  │
  ▼
Output: valid floating-point tetrahedral mesh
```

---

## 4. 核心创新：Incremental Triangle Insertion 详解

### 4.1 与 TetWild 的根本区别

**TetWild** 的做法：把所有 input triangles 转成 planes，用 [BSP (Binary Space Partitioning)](https://en.wikipedia.org/wiki/Binary_space_partitioning) 一次性切分 background mesh，生成 polyhedral mesh，所有计算用 **rational numbers** 保证 exactness，然后用 mesh improvement 慢慢把 rational coordinates round 回 floating-point。问题：rounding 可能失败（理论上），且 rational arithmetic 极慢（分子分母膨胀）。

**fTetWild** 的做法：始终保持 floating-point coordinates，一次只插入一个 triangle，如果某次 insertion 产生 inverted/degenerate tet 就 rollback，等 mesh quality 改善后再重试。**整个 pipeline 任何时刻都维持一个 valid 的 floating-point tetrahedral mesh**。

### 4.2 Single Triangle Insertion 的三步

#### Step 1: Finding Cut Tetrahedra

定义：triangle T "cuts" tetrahedron 𝒯 当且仅当：
- T 完全在 𝒯 内部，OR
- T 切过 𝒯 的至少一个 face（intersection 包含两者的 interior points）

初始化 T_I = {被 T 切割的 tetrahedra}，这个集合会在后续步骤中扩张。

判断使用 exact predicates（[Shewchuk 1997](https://www.cs.cmu.edu/~quake/robust.html) 的 adaptive precision floating-point predicates，[Lévy 2019](http://alice.loria.fr/index.php/software/4-library/75-geogram.html) 的 Geogram 实现也提供）加上 [Guigue-Devillers](https://doi.org/10.1080/10867651.2003.10487580) 的 triangle-triangle overlap test。Exact predicates 保证了 topological correctness，即使坐标是 floating-point。

#### Step 2: Plane-Tetrahedra Intersection with Snapping

这是 fTetWild 最精巧的部分。理想情况下（infinite precision），计算平面 P（T 所在平面）与 T_I 中所有 tetrahedra edges 的交点，得到平面 P 上的一个 polygonal mesh 𝓕，它 cover 了 T。但 floating-point 会引入 round-off，导致 degenerate/inverted tets。

**Snapping 策略**：引入 tolerance δ（第一遍 δ = max(ϵ_zero, 10^{-3}ϵ)，后续 δ = ϵ_zero）。对于 T_I 中距离 P 小于 δ 的 vertex v：

- **Case 1 (move v to P)**：如果移动 v 到 P 上不会 invert 任何 element，就移动 v。这相当于 deform 𝓕，保持它在 P 上。如果 v 是 𝓕 的 boundary vertex，需要先 expand T_I 加入 v 的 1-ring neighborhood。
  
- **Case 2 (snap intersection point to v)**：如果移动 v 会 invert，则保留 v 不动，把本该在 P 上的 intersection point "snap" 到 v（即 v 成为 𝓕 的一个 vertex，距离 P 最多 δ）。这允许 𝓕 微微偏离 P。

这保证了 floating-point 下也能 robust 地完成 insertion。算法 iterate 这 4 步直到收敛：
1. 找 T_I 中距离 P < δ 的 vertices，放入 𝒱_δ
2. 尝试 move 𝒱_δ 中的 vertices 到 P（不 invert 才 move）
3. 对 𝒱_δ 中每个 vertex，加入其 vertex-adjacent tetrahedra 到 T_I（如果被 P 切且 face 的投影与 T 相交）
4. Repeat 直到没有新 tet 加入

#### Step 3: Table-based Tetrahedron Subdivision

对于 T_I 中（以及其 neighbor 中）被 P 切到 edge 的 tetrahedra，根据 **edge-cut configuration** 查表 subdivision。

一个 tetrahedron 有 6 条 edges，每条 edge 可能被切或不切，所以理论上 2^6 = 64 种 configuration。但其中 23 种是 impossible 的：
- 6 条都被切（1 种）
- 5 条被切（6 种）
- 4 条被切且 3 条在同一 face（3 × 4 = 12 种）
- 3 条被切且都在同一 face（4 种）

剩下 41 种 realizable configurations 分成 7 个 symmetry classes（见 paper Figure 9）。其中 5 个 class 在 [Schweiger & Arridge 2016](https://doi.org/10.1002/nme.5271) 中出现过（平面切 tet），另外 2 个额外 class（Figure 9 的 (4) 和 (6)）专门处理 neighbor tet 只有部分 edge 被切的情况。

**索引方式**：
- **Primary index (I)**：6-bit binary string，指示哪 6 条 edge 被切
- **Secondary index (II)**：当一个 face 有 2 条 edge 被切时，有 2 种 triangulation 选择，secondary index 区分

**保证 topology 一致的规则**：对于 face [v₀, v₁, v₂] 有两个交点 p₁, p₂，选择包含 edge [p₂, v₁] 的 triangulation 如果 v₁ 的 global integer label > v₂ 的 label，否则选择包含 [p₁, v₂] 的 triangulation。这个简单的 deterministic rule 保证相邻 tet 共享 face 时 triangulation 一致，避免 T-junction。

### 4.3 Open-boundary Edge Preservation

对于 open boundary edge（只有 1 个 incident triangle，或多个 coplanar triangles 在同一侧），单纯插入 triangle 不能保证 edge 被 preserve（因为相邻 triangle 的 plane 才会切 𝓕）。fTetWild 的处理：
- 把 edge e 和 𝓕 投影到 best-fit plane P'
- 在 P' 上计算投影后 e 与 𝓕 face 的 2D intersection
- Lift intersection 点回 3D
- 用同样的 table-based subdivision 切分相关 tetrahedra

---

## 5. Mesh Improvement 与 AMIPS Energy 的陷阱

### 5.1 AMIPS Energy 公式

paper 用的是 [conformal AMIPS 3D energy](https://doi.org/10.1145/2983621)（Rabinovich et al. 2017）：

$$
\text{AMIPS} = \frac{\text{tr}(\mathbf{J}^T \mathbf{J})}{\det(\mathbf{J})^{2/3}}
$$

变量解释：
- $\mathbf{J}$ 是从 reference regular tetrahedron 到当前 tetrahedron 𝒯 的 affine transformation 的 Jacobian matrix（3×3）
- $\text{tr}(\mathbf{J}^T \mathbf{J})$ 是 Jacobian 的 Frobenius norm 的平方，衡量 shape distortion
- $\det(\mathbf{J})$ 是 volume ratio
- 指数 $2/3$ 使得 energy 对 uniform scaling invariant
- 最小值 = 3，对应 regular tetrahedron（完美形状）
- 值越大 quality 越差，∞ 对应 degenerate（det = 0）或 inverted（det < 0）

### 5.2 Floating-point Instability 的发现与修复

这是 paper 的一个 subtle 但重要的发现。AMIPS energy 理论上对 tetrahedron 的 vertex permutation 不变，但 floating-point 计算下却会变！paper Appendix B 给了一个具体例子：4 个 vertices 给出的 AMIPS 在不同 permutation 下：

| Permutation | AMIPS (float) | AMIPS³ (float) |
|---|---|---|
| 1234 | 5.03e10 | 9.40e25 |
| 2341 | 2.17e11 | 1.83e25 |
| 3412 | 8.87e10 | 1.01e26 |
| 4123 | 7.10e10 | 3.46e26 |
| (rational) | 1.13e11 | - |

差异达到 2 个数量级！这意味着 gradient descent 用的 descent direction 可能是错的——以为在下降 energy 实际在上升。

**fTetWild 的修复**：当 energy > 10⁸ 时，把 energy 提升到三次方（完全 rational），用 rational arithmetic 计算，再 round 到 double，最后取 cube root：

$$
\text{AMIPS}^3 = \frac{(\text{tr}(\mathbf{J}^T \mathbf{J}))^3}{\det(\mathbf{J})^2}
$$

这样 det(J) 的符号不会因为 round-off 而翻转（因为 squared），且 rational 保证 permutation invariance。由于 high-energy tet 占比很小，overall overhead 可忽略，但避免了 Figure 10 所示的 over-refinement 灾难。

### 5.3 Local Operations

四种 local ops（与 TetWild 相同）：
1. **Edge splitting**：长 edge 一分为二
2. **Edge collapsing**：短 edge 合并两端点
3. **Edge swapping**（2-3 flip, 3-2 flip, 4-4 flip 等）：改变 connectivity
4. **Vertex smoothing**：移动 vertex 位置优化 AMIPS

每个 op 都要 rollback 如果：
- 产生 inverted tet
- Tracked surface 离开 ϵ-envelope

**Parallel smoothing**：用 graph coloring（类似 Figure 4）把 vertices 分成 independent sets，每个 set 内部可以并行 smoothing。这是一个简单的 shared-memory parallelization。

### 5.4 与 un-inserted triangles 的 interleave

每 3 次 mesh improvement iteration 后，重新尝试插入之前失败的 triangles。原理：mesh quality 提升后，之前 fail 的 region 可能已经有了更规整的 tet，insertion 成功率提高。

---

## 6. Preprocessing 的并行化细节

### 6.1 Edge Collapse 的 2-Coloring

edge collapse 是 serial bottleneck（因为 envelope containment check 慢）。fTetWild 用一个简单的 2-coloring 策略：

1. 初始化所有 input triangles 为 white
2. Iterative：标记一个 edge 为 "parallel-independent" 如果其 vertex-adjacent triangles 全是 white；标记这些 triangles 为 black
3. 并行 collapse 所有 parallel-independent edges
4. Repeat 直到能 remove 的 vertices < 0.01%

Figure 4 的 2D 示意图：选一条 edge（红），它的 vertex-adjacent triangles 染黑，这些 triangles 的其他 edges 不能再选。最终剩下的红色 edges 互不影响，可并行 collapse。8 cores 下平均 4× speedup。

### 6.2 Envelope 构造与检查

使用 [Hu et al. 2017](https://doi.org/10.1109/TVCG.2016.2632720) 的 envelope 定义：对每个 input triangle 构造一个 offset volume，output mesh 的 tracked surface 必须在其中。检查方法是采样 input triangle 的点，验证它们在 slightly smaller envelope 内（预留 sampling error margin）。

---

## 7. Filtering 与 Boolean Operations

### 7.1 Winding Number Filtering

使用 [Barill et al. 2018 的 Fast Winding Number](https://doi.org/10.1145/3197517.3201397)（基于 [Taylor 2018 魔术级加速](https://doi.org/10.1145/3197517.3201397)）判断每个 tet centroid 在 input surface 内还是外，过滤掉外部 tet。

### 7.2 Mesh Arrangement for Boolean Operations

[Zhou et al. 2016 的 Mesh Arrangements](https://doi.org/10.1145/2897824.2925905) 用 rational arithmetic 计算 arrangement，对 non-PWN（Positive Winding Number）输入会失败。fTetWild 的扩展：
- 每个 input triangle 记录其 source（属于哪个 input soup）
- 计算每个 tet centroid 对每个 input soup 的 generalized winding number
- 根据 Boolean operation（union / intersection / difference）和 winding number 集合决定保留哪些 tet

优点：
1. 支持 non-PWN 输入（gaps、self-intersections 都 OK）
2. 输出是 tetrahedral mesh（不只 surface），可直接用于 FEM
3. Surface quality 高（因为 ϵ-envelope 允许 remeshing）

---

## 8. 实验数据详解

### 8.1 Thingi10k 上的 Success Rate（Table 2）

| Method | Success Rate | OOM | Time Exceeded | Avg Time (s) |
|---|---|---|---|---|
| CGAL | 79.00% | 0% | 21.00% | 11.7 |
| TetGen | 49.50% | 0.10% | 48.70% | 32.3 |
| TetWild | 99.89% | 0.05% | 0.11% | 360.0 |
| **fTetWild** | **99.97%** | 0.02% | 0.03% | **49.8** |

关键观察：
- fTetWild 在 success rate 上几乎完美，比 TetWild 略好
- 速度比 TetWild 快 7×（49.8s vs 360s）
- 比 TetGen 略慢但 robustness 完爆（49.50% → 99.97%）
- 比 CGAL 慢 4× 但 robustness 显著好（79% → 99.97%）

### 8.2 Reduced Thingi10k（4540 models，4 种方法都成功的子集）

| Method | Avg Time (s) |
|---|---|
| TetGen | 22 |
| **fTetWild** | **18.5** |
| CGAL | 95 |
| TetWild | 107 |

在这个公平子集上，fTetWild 实际上比 TetGen 还快！分布的尾部也最短——只有 4 个 model 需要 >16 min，而 TetGen 20 个、CGAL 122 个、TetWild 25 个。98.7% 的 model 在 2 min 内完成。

### 8.3 Mesh Quality 指标（5 种）

paper 用了 5 种 tetrahedron quality measure 对比 fTetWild 和 TetWild：

1. **AMIPS energy**：范围 [3, +∞)，optimal = 3
2. **Minimal dihedral angle**：范围 (0, 1.23]，optimal = 1.23（regular tet 的 dihedral angle = arccos(1/3) ≈ 70.53°，归一化到 1.23）
3. **Volume-to-edge ratio**：$6\sqrt{2}V / \ell_{\max}^3$，范围 (0, 1]，optimal = 1
   - $V$ = tet volume
   - $\ell_{\max}$ = longest edge length
4. **Aspect ratio**：$\sqrt{3/2} h_{\min} / \ell_{\max}$，范围 (0, 1]，optimal = 1
   - $h_{\min}$ = minimum height（tet 4 个 face 的高的最小值）
5. **Radius-to-edge ratio**：$2\sqrt{6} r_{\text{in}} / \ell_{\max}$，范围 (0, 1]，optimal = 1
   - $r_{\text{in}}$ = inscribed sphere radius

Figure 17 的 histogram 显示 fTetWild 与 TetWild 在所有 5 个 measure 上质量几乎相同——这是 expected 的，因为用了相同的 optimization framework。

### 8.4 Mesh Density（Figure 18）

fTetWild 和 TetWild 都生成 as-coarse-as-possible mesh（target edge length ℓ = d/20，d = bounding box diagonal）。TetGen 因为 exact preserve input surface，dense input 导致 dense output。CGAL 偶尔在 sharp feature 和 small artifact 处 over-refine。

### 8.5 极端案例

- **Figure 15**：一个 model 上 fTetWild 比 TetWild 快 17×
- **Figure 21**：Velo3D 工业 additive manufacturing 的 exhaust pipe（93M vertices, 31M faces），带 gyroid triply periodic minimal surface 结构。fTetWild 55 min 完成（envelope 减半则 122 min），TetWild 215 min。人工修复要 2 周！
- **Figure 22**：建筑应用，80999 个 self-intersecting faces 的 cylinder network，fTetWild 成功 mesh

---

## 9. Applications 详解

### 9.1 Mesh Repair

TetWild/fTetWild 都可以当 mesh repair 工具用：tetrahedralize 后提取 boundary。fTetWild 的优势是 mesh improvement 可随时 stop（因为始终 valid float mesh），不需要等 optimization 收敛。

对比 [MeshFix (Attene 2010)](https://doi.org/10.1007/s00371-010-0416-3)：MeshFix 快但 greedy 可能 delete 大片 mesh；fTetWild 慢但 controllable error 且保 detail。

**Non-manifold 修复**：boundary extraction 可能产生 non-manifold surface。fTetWild 用 [Attene et al. 2009](https://doi.org/10.1016/j.cagd.2009.06.002) 的算法：identify non-manifold edges → split → duplicate non-manifold vertices → 保证 manifold output（可能产生 coincident vertices）。

### 9.2 Boolean Operations

Figure 23-24 展示了在 non-manifold、self-intersecting、non-PWN 输入上的 union/difference/intersection。所有 operation 在 30s 左右完成，output max AMIPS ≈ 8。

对比：
- [CGAL Nef Polyhedra](https://doc.cgal.org/4.14/Manual/packages.html#PkgNef3)：exact 但要求 closed manifold
- [Zhou et al. 2016 Mesh Arrangements](https://doi.org/10.1145/2897824.2925905)：robust 但要求 PWN，且用 rational 可能 rounding 失败
- [Cork (Bernstein 2013)](https://github.com/gilbo/cork)：快但 non-robust
- fTetWild：robust + 支持 non-PWN + 输出 tet mesh + 高 surface quality

### 9.3 FEM Simulation

- Figure 25：non-linear elastic deformation（直接用 fTetWild 输出做 FEM）
- Figure 26：[Schneider et al. 2018 的 a priori p-refinement](https://doi.org/10.1145/3272127.3275067)——根据每个 tet 的 quality 决定 polynomial order，允许 early stop mesh optimization。Max energy ≤ 10 用 107s 产生 90438 tets；用 p-refinement 标准 69s 产生 41735 tets，max energy 32.4 但 simulation accuracy 等价。
- Figure 27：流体仿真，background mesh 用 Boolean difference 生成

---

## 10. 与 Concurrent/Successor Work 的关联

### 10.1 [Harmonic Triangulations (Alexa 2019)](https://doi.org/10.1145/3306346.3323022)

与 fTetWild concurrent 的工作。提出直接优化 tetrahedralization 的 Dirichlet energy，对 sliver elimination 有效。paper 在 conclusion 提到 comparative study 会很有趣，可能进一步加速。

### 10.2 [TriWild (Hu et al. 2019)](https://doi.org/10.1145/3204409)

2D 对应版本，处理 curve constraints。fTetWild 的 incremental insertion 思想理论上可移植到 2D 加速 TriWild。

### 10.3 后续相关

- [Polygon Mesh Processing](https://www.crcpress.com/Polygon-Mesh-Processing/Botsch-Kobbelt-Pauly-Alliez-Levy/p/book/9781568814261) book by Botsch et al.
- [geometry-central](https://geometry-central.net/) 和 [libigl](https://libigl.github.io/) 都集成相关几何处理
- 现代 robust geometry 的 foundation：[Shewchuk's predicates](https://www.cs.cmu.edu/~quake/robust.html)、[Geogram](http://alice.loria.fr/index.php/software/4-library/75-geogram.html)、[exact init](https://github.com/GeometryCollective/geometry-central)

---

## 11. Intuition Building：为什么 fTetWild 又快又 robust

### 11.1 速度来源

1. **去掉 rational arithmetic**：rational number 的分子分母会膨胀，加减乘除都慢 10-100×。fTetWild 几乎全用 float64，只在 AMIPS energy > 10⁸ 的 rare case 用 rational。
2. **Incremental 而非 batch**：BSP 一次性切所有 plane 会产生极复杂 polyhedral mesh，要全部 rational 化。Incremental 可以 fail-fast rollback，避免 bad configuration 累积。
3. **Parallelization**：preprocessing 2-coloring，smoothing graph coloring。8 cores 下 4× preprocessing speedup。
4. **Early termination**：mesh repair 用例不需要 high quality mesh，可以 energy < 2000 就 stop。

### 11.2 Robustness 来源

1. **Exact predicates**：topological decision（in/out、orientation、cut detection）用 Shewchuk/Geogram exact predicates，保证 topology 不会因 round-off 翻转。
2. **ϵ-envelope**：允许 boundary 微动，自动 heal self-intersection/gap。
3. **Snapping tolerance δ**：处理 floating-point 下 near-degenerate 的情况，避免插入 zero-volume tet。
4. **Rollback mechanism**：任何 op 产生 inverted/degenerate tet 就 rollback，保证 invariant（valid float mesh）始终成立。
5. **Re-attempt after improvement**：失败的 insertion 不是永久失败，mesh quality 提升后重试，实践证明总能成功。

### 11.3 Trade-off 的本质

fTetWild 放弃了 TetWild 的 "guaranteed insertion of all input triangles" 这个理论保证，换取了：
- 速度（7×）
- 理论上的 float output 保证

实践上从未观察到 insertion 失败（10000+ model 全部成功），所以这个 trade-off 在实用中基本是 free lunch。理论上的"可能失败"在实践中近乎不可能发生，因为：
- ϵ-envelope 给了足够的 slack
- Snapping 处理了 near-degenerate 情况
- Re-attempt 机制让 mesh 改善后能成功

### 11.4 为什么这个设计 elegant

经典的 robust geometry 难题：**如何用 inexact arithmetic 实现 robust algorithm**。[Kettner 1999](https://doi.org/10.1007/PL00009390) 的著名 example 展示了 naive float 会导致 orientation predicate 翻转。传统解法有两条路：
- Exact arithmetic（如 [CGAL](https://www.cgal.org/) 的 [Filtered_kernel](https://doc.cgal.org/4.14/Manual/packages.html#PkgKernel23)：先 float 快速，fail 时 fallback exact）
- Exact predicates + inexact constructions（Shewchuk 的方案：topology exact，coordinates float）

fTetWild 属于第二类，但更激进：不仅 topology 用 exact predicates，construction（snapping、subdivision）也精心设计成在 float 下 robust。秘诀是把 "exactness" 推给 envelope tolerance——既然允许 ϵ 偏差，那么 δ-snapping 内的误差都合法。这是一种 **geometric tolerance + exact predicates** 的 hybrid，既快又 robust。

### 11.5 与 epsilon-geometry / tolerance volume 的传统

这个思路可以追溯到：
- [Hoffmann 1989 "Geometric and Solid Modeling"](https://books.google.com/books?id=OFhvAAAAMAAJ) 的 epsilon geometry
- [Sugihara 1999](https://doi.org/10.1016/S0010-4485(99)00038-3) 的 topology-oriented approach
- [Fortune 1997](https://doi.org/10.1007/PL00009390) 的 robustness analysis
- [Attene 2017 的 hybrid kernel](https://link.springer.com/chapter/10.1007/978-3-642-32014-0_7)

fTetWild 把这条 line of work 应用到 tetrahedral meshing 的极端 case（最复杂的 3D meshing），并展示了 industrial-scale 可行性。

---

## 12. 局限与未来方向（paper 自述 + 我的推测）

paper 自己提到的：
1. Naive parallelization，未来可扩展到 distributed/HPC
2. Dynamic remeshing 复用现有 mesh
3. 2D 版本（TriWild 加速）
4. 用 Dirichlet energy（Alexa 2019）替代 AMIPS 可能更快

我额外推测的 interesting extensions：
- **Adaptive ϵ**：当前全局 ϵ，对 sharp feature 可以局部更紧
- **Anisotropic meshing**：当前 isotropic，FEM 仿真有时需要 boundary layer
- **Hex-dominant meshing**：tet 是 fallback，hex mesh 更优但更难。相关：[PolyCut (Zhang et al. 2017)](https://doi.org/10.1145/3072959.3073692)、[Gao et al. 2017](https://doi.org/10.1145/3130800.3130848)
- **Differentiable tetrahedralization**：让 meshing step 可微以用于 shape optimization。相关：[DiffSim](https://github.com/PhysicsAwareLearning/DiffSim) 等
- **Neural meshing**：用 learned sizing field 或 learned placement。相关：[Nervana Mesh](https://arxiv.org/abs/2010.06408) 思路

---

## 13. 实现细节的一些 gotchas

### 13.1 Tolerance 选取

- $\epsilon_{\text{zero}} = 10^{-8}$：distance 低于此视为 0，area 用 $\epsilon_{\text{zero}}^2$，volume 用 $\epsilon_{\text{zero}}^3$
- $\epsilon_{\text{prep}} = 0.8\epsilon$：preprocessing envelope 留 20% slack 给 snapping。范围 [0.7, 0.999] 都 OK，paper 选 0.8 是 conservative mid-point
- 第一遍 insertion $\delta = \max(\epsilon_{\text{zero}}, 10^{-3}\epsilon)$，后续 $\delta = \epsilon_{\text{zero}}$：第一遍宽松，refinement 后收紧
- Stopping criteria：max AMIPS < 10 或 80 iterations（与 TetWild 一致，公平对比）

### 13.2 Background Mesh 构造

Delaunay tetrahedralization on：
- Preprocessed input vertices
- 额外 uniform grid points（spacing = d/20，d = bounding box diagonal）
- 距离 input face < ϵ 的 grid point 跳过
- Bounding box 扩大 2ϵ

这个 background mesh 不 conform input，只是个起始 tetrahedral mesh，后续 insertion 会让 boundary conform。

### 13.3 7 个 Symmetry Class 的来源

41 个 realizable edge-cut configurations 用 tetrahedral symmetry group T_d（24 个 symmetries）可以归约到 7 个 classes。其中 5 个是平面完整切 tet 的情况（Schweiger & Arridge 已研究），另外 2 个（config (4) 和 (6)）是 fTetWild 新增的，处理 neighbor tet 只有部分 edge 被切（因为相邻 tet 已 subdivision 消化了部分 cut）。这是 fTetWild 的 table 必须 self-contained 而不能直接用 Schweiger table 的原因。

### 13.4 Secondary Index 选择的 tricky

当 face 有 2 条 edge 被切，有 2 种 triangulation（对角线选哪个）。如果两个相邻 tet 选不同 triangulation，就产生 non-conforming T-junction。fTetWild 的 deterministic rule：比较两个 endpoint vertices 的 global integer label，选包含更大 label vertex 的对角线。这个 rule 是 global 一致的，所以相邻 tet 必选同一个。

这个 rule 也自动排除了需要 internal vertex 的 2 个 configuration（Appendix C 的 Figure 28）——通过反证法证明 label 不等式矛盾。

---

## 14. 总结：fTetWild 的 design philosophy

fTetWild 体现了几何处理算法设计的一种成熟范式：
1. **Topological robustness**：exact predicates 保证 topology 决策正确
2. **Geometric tolerance**：ϵ-envelope 允许 controlled deviation，使 inexact construction 安全
3. **Invariant-driven**：始终维持 "valid float tet mesh" invariant，任何 violation 立即 rollback
4. **Lazy failure + retry**：失败的 op 不终止算法，等条件改善后重试
5. **Hybrid arithmetic**：默认 float64，rare critical path 用 rational

这种范式 vs TetWild 的 "全 rational" 范式：牺牲了理论上的 universal guarantee，换来了实用中的 efficiency 和 simplicity。在 10000 个 real-world model 上的 empirical success（99.97%）证明这个 trade-off 在实践中是无痛的。

这种思路在更广义的 robust algorithm design 中也适用——比如 [Shewchuk's Triangle](https://www.cs.cmu.edu/~quake/triangle.html)（2D Delaunay）、[Si's TetGen](https://www.wias-berlin.de/software/tetgen/)（3D Delaunay refinement）、[Marx et al. 2021 incremental mesh boolean](https://doi.org/10.1145/3450626.3459765) 等都用了类似 hybrid。

---

## 参考资源链接

### Primary Sources
- fTetWild paper (这篇): https://doi.org/10.1145/3386569.3392369
- TetWild (前身): https://doi.org/10.1145/3197517.3201353
- TriWild (2D 版): https://doi.org/10.1145/3204409
- Mesh Arrangements (Zhou et al.): https://doi.org/10.1145/2897824.2925905
- Fast Winding Number: https://doi.org/10.1145/3197517.3201397
- AMIPS: https://doi.org/10.1145/2983621
- Decoupling simulation accuracy from mesh quality: https://doi.org/10.1145/3272127.3275067

### Software
- fTetWild code: https://github.com/wildmeshing/fTetWild
- TetWild code: https://github.com/wildmeshing/TetWild
- Geogram: http://alice.loria.fr/index.php/software/4-library/75-geogram.html
- Shewchuk predicates: https://www.cs.cmu.edu/~quake/robust.html
- TetGen: https://www.wias-berlin.de/software/tetgen/
- CGAL: https://www.cgal.org/
- Eigen: https://eigen.tuxfamily.org/

### Datasets
- Thingi10k: https://github.com/alecjacobson/thingi10k
- Thingi10k website: https://tenmodel.dnsdynamic.net/

### Surveys & Background
- Delaunay Mesh Generation book: https://www.routledge.com/Delaunay-Mesh-Generation/Cheng-Dey-Shewchuk/p/book/9781584887300
- Shewchuk "What is a good linear element": https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
- Polygon Mesh Repairing survey: https://doi.org/10.1145/2431211.2431214
- Attene hybrid kernel: https://link.springer.com/chapter/10.1007/978-3-642-32014-0_7

### Concurrent/Related
- Harmonic Triangulations (Alexa): https://doi.org/10.1145/3306346.3323022
- Geogram tetrahedral meshing: http://alice.loria.fr/index.php/software/4-library/75-geogram.html

---

## 给 Karpathy 的额外 intuition notes

如果你要从 ML/Differentiable programming 角度思考这类 work，几个 angle 可能 useful：

1. **fTetWild 的 incremental insertion 像不像 autoregressive generation？** 一个一个 triangle "token" 插入，每次局部 subdivide 改变 connectivity，rollback 像 rejection sampling。理论上可以把它看作一个 non-differentiable 的 forward process，要让它 differentiable 需要 Gumbel-softmax 之类的 relaxation 来处理 combinatorial decision。

2. **Mesh quality → simulation accuracy 的 decoupling** (Schneider et al. 2018) 非常像 "importance sampling" 或 "residual learning" 的思想——不需要每个 element 都 perfect，只要 polynomial order 补偿 shape 即可。这与 neural ODE / learned integrator 的思路有 resonance。

3. **ϵ-envelope 的 "soft constraint" 思路** 与 ML 中的 soft constraint / barrier method / Lagrangian relaxation 是同一个 idea 在不同 domain 的 instance。可以联想 [PANOZZO 2014 frame field](https://doi.org/10.1145/2601097.2601101) 的 soft constraint optimization。

4. **Exact predicates + inexact constructions** 的 robust computing paradigm 与 ML 中的 "exact gradient + inexact forward" 训练法（如 straight-through estimator）有抽象的 parallel——都是 topological/qualitative decision 必须准，quantitative value 可以 fuzzy。

希望这个 deep dive 帮你 build 出 fTetWild 在 robust geometry processing 大图景中的 intuition。这个 work 是 geometry + numerics + systems engineering 的精致结合，是 "make robust algorithm practical" 的 textbook case study。
