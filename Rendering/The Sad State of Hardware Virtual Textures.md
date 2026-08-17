---
source_pdf: The Sad State of Hardware Virtual Textures.pdf
paper_sha256: ea34c43356cd5c6f95738fa394bf17e10b420fc69adede89d34fc4ce79329c12
processed_at: '2026-08-12T14:50:29-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 paper

## 一句话版本

GPU 本来该有个"虚拟内存"给 texture 用，硬件支持的，代码写起来跟普通 texture 一模一样，采样还几乎免费。结果呢，driver 把它做废了 — binding 一个 tile 慢到你没法实时用，同一张显卡换不同 driver 能差 300 倍。作者实测证明这件事可以做快（Mesa 开源驱动就做快了），只是 AMD 和 Nvidia 的官方驱动没去做。

---

## 为什么需要 virtual texture

想象你做一款开放世界游戏，整个 map 要一张 128k × 128k 的 texture 贴满。这张 texture 如果是普通的，RGBA8 一存就是 64GB。你的显卡 VRAM 可能才 12GB，根本装不下。

但你想想，玩家站在一个山头上看出去，真正看到的 texture 区域有多少？可能就几 MB。远处那些山的 texture 根本不需要高分辨率，被山挡住的那些 tile 更是压根没人去 sample。

这跟 OS 的 virtual memory 是一模一样的故事：进程觉得自己有 4GB 内存，实际物理 RAM 只有 512MB，但只要 working set 小，一切照常跑。

Virtual texture 干的就是这件事 — 把 texture 切成一块一块的 tile（类比 page），只把真正会被 sample 到的 tile 加载进 VRAM，其他的 tile 标记成 "not resident"。

Rage (2011, id Software) 就是这么干的，一张 128k² 的 MegaTexture 跑在 256MB VRAM 的 PS3 上。Carmack 当年推动 id Tech 5 的核心 idea 就是这个。

参考: https://doi.org/10.1145/2343483.2343488

---

## Software virtual texture 怎么做

你自己手动管理 indirection。两个组件：

1. **Atlas** — 一个普通大 texture，所有加载进来的 tile 都塞这里
2. **Tile map** — 一个 lookup table，记录 "virtual tile (u,v) 对应 atlas 里的哪个 physical tile"

Shader 里采样的时候：

```glsl
ivec2 virtualTile = ivec2(p / tileSize);          // 哪个 virtual tile
ivec2 physicalTile = texelFetch(tileMap, virtualTile, 0).xy;  // 查表
ivec2 atlasP = physicalTile * tileSize + p % tileSize;       // 在 atlas 里的位置
return texelFetch(atlas, atlasP, 0);              // 真正采样
```

这就是你自己手写一个 "texture MMU"。Tile map 就是你的 page table，atlas 就是你的 physical RAM。

**好处**：binding 一个 tile 就是在 tile map 里写一个数，纳秒级。
**坏处**：
- 采样多一次 indirection（先查表再采样），慢；
- 跨 tile 边界的 bilinear / mipmap / anisotropic filtering 全得自己处理，要么手动插值（慢），要么 stochastic 抖动（有 shimmer），要么给每个 tile 加 border texel（费内存，128×128 tile 加 16× aniso border 能浪费 56% 内存）；
- 代码又长又脏。

---

## Hardware virtual texture 本来该多好

2012 年左右，Nvidia Maxwell、AMD PRT、Intel 都加了 hardware sparse texture。想法很美：

- Tile map 由 driver 管，shader 里**完全看不到** indirection
- 采样代码跟普通 texture 一模一样，`texture(sampler, uv)` 完事
- Hardware 的 TMU（texture memory unit）内部自动处理跨 tile 的 bilinear / mipmap / anisotropic
- API 覆盖 OpenGL / Vulkan / D3D11/12 / Metal / CUDA

理想情况下，这就是 silver bullet — 你写 shader 的体验跟普通 texture 完全一样，driver 偷偷帮你搞 virtualization。

参考: https://www.ece.lsu.edu/gp/refs/GeForce-GTX-980-Whitepaper-FINAL.pdf

---

## Hardware virtual texture 现在有多惨

作者列了 7 条 limitation，按严重程度排：

### 1. Texture size 上限太小
Virtual texture 本来就是为了做超大 texture 的，结果 hardware VT 反而比普通 texture 还小：
- Nvidia 2D 上限 32k²，AMD 16k²，Intel 16k²
- 3D 更惨，Intel 只能到 2k³

32k² × 4B = 4GB，离 "giant" 还远着。这跟 virtual texture 的初衷直接打架。

### 2. **Tile binding 慢到离谱**（这是最致命的）

Binding 一个 tile 就是告诉 GPU "virtual tile (u,v) 现在映射到 atlas 位置 (x,y)"。Software 里就是往 storage buffer 写一个 vec2，纳秒级。

Hardware 里要走 API call → driver → kernel → GPU memory management。作者实测 binding 1024 个 tile 的时间：

| Driver | 时间 |
|--------|------|
| Software VT (所有 driver) | < 1ms |
| Intel ANV (Mesa 开源) | ~1-2ms |
| NVK (Mesa 开源 Nvidia) | ~1-2ms |
| AMD RADV (Mesa 开源 AMD) | 中等 |
| AMD Windows 官方 | 慢 |
| Nvidia Windows 官方 | 慢 |
| AMDVLK (AMD 官方 Linux) | ~100ms |
| Nvidia Linux 官方 | 极慢 |

**最快和最慢差 300 倍，同一张显卡，只是 driver 不同**。

100ms bind 1024 tiles 是什么概念？一帧才 16.6ms（60fps），你光 binding tiles 就花掉 6 帧的预算。游戏里完全没法用。

### 3. Tile status 查询信息太少
Shader 里只能查到 "footprint 内所有 tile 是不是都 bound"（一个 bool）。你没法知道具体哪个 tile 没 bound，没法查"某个 UV 的最低 bound mip 是几"。对于 voxel streaming 这种场景，你其实想知道很精细的 residency 信息，driver 不给你。

### 4. Anisotropic filtering 支持奇葩
- Nvidia: 16×（满血）
- Intel: 4×
- AMD: **0×（完全不支持）**

AMD 在 sparse texture 上完全不支持 anisotropic filtering。这意味着同样一个游戏，AMD 卡上要么画质差（关掉 aniso），要么不能用 hardware VT（回退到 software）。

### 5. 不能从 GPU 端改 binding
现代渲染管线越来越 GPU-driven — GPU 自己 cull、自己 LOD、自己 indirect draw。结果 tile binding 还要 CPU 通过 API call 改。这意味着每帧都要 CPU-GPU sync，完全破坏 GPU-driven 的初衷。

作者自己的前作 GigaVoxels DP（https://doi.org/10.1145/3675389）就展示了如果 GPU 能自己改 tile binding，可以做出多么强大的 voxel streaming 系统 — 但 hardware VT 现在根本不支持。

### 6. 没有低级 API
Driver 把 tile map 整个藏起来了。你只能用 driver 给你的 API，不能换数据结构，不能绕开 indirection。这正是 binding 慢的 root cause — driver 内部 tile map 是怎么实现的，你完全不知道，可能是很慢的数据结构 + 锁 + validation，你只能干瞪眼。

### 7. Fixed 64KB tile size
Vulkan 规定 sparse tile 永远是 64KB，不管你是 2D 还是 3D，不管你什么格式。好处是不同 format 的 tile 能共享 pool；坏处是 3D texture 的 tile 没法开成完美立方体（$\sqrt[3]{64\text{KB}}$ 不是 2 的整数次幂），tile 形状很别扭。

---

## Sampling 性能：hardware 是真的快

作者做了 ray marching 测试，穿过一个 512³ 的 Perlin noise volume，纯烤 texture 采样：

- Regular texture: 基准
- Hardware VT: **几乎跟 regular texture 一样快**
- Software VT: 大约是 hardware VT 的 50%

这说明 hardware VT 的 indirection 在 TMU 内部完成，走的是 texture L1 cache 同一条快路径；software VT 要先发一次 fetch 查 tile map，再发一次 fetch 采样 atlas，两次 TMU round-trip，慢一倍可以理解。

所以 hardware VT 在**采样端**是完美解决问题的。问题完全在 **binding 端**。

---

## 为什么 binding 这么慢？作者的解释

作者没做 root cause 分析（也没法做，driver 是闭源的），但推测：

1. **Driver 内部 tile map 数据结构可能很重** — 带锁、带 validation、带多级结构
2. **每个 binding call 可能触发 kernel syscall** — Linux 下 AMDGPU sparse binding 走 drm ioctl，Windows 下走 kernel-mode driver。进 kernel 就慢
3. **NVK 和 ANV 证明可以做快** — 这两个 Mesa 开源 driver 都接近 software VT 的速度，说明硅层面完全有能力，纯粹是 vendor 官方 driver 没去做优化

这就是 paper 标题 "Sad State" 的真正含义 — 硬件支持早就有了，采样也快，但官方 driver 不上心，把整个 feature 做成了鸡肋。

---

## 作者的诉求

作者在 conclusion 里喊话：

> Intel 和 NVK 证明 hardware VT binding 可以做快，希望我们只差一个 AMD 和 Nvidia driver update 就能让 hardware VT 在所有 GPU 上真正可用。

更深层地，作者在 Limitation #6 里隐含一个 API design 建议：**未来的 sparse texture API 应该把 tile map 暴露成 user-managed resource**。Sampler 函数接受 (tile index, local UV) 而不是 (virtual UV)，让用户自己管 indirection 但享受 hardware TMU 的 aniso/mipmap/cache 路径。这其实是 software VT 和 hardware VT 的混合 — 既有 control 又有 hardware acceleration。

这跟现代 graphics 的整体趋势一致：Vulkan、DX12 都在给 user 更多 low-level control，把 driver 做薄。Hardware VT 的当前 API 设计正好走反了 — driver 做太多，结果做成了瓶颈。

---

## 这个故事的大图景

用一句话类比：

**Hardware VT 就像是 OS 给你一个 virtual memory 系统，但改 page table 要走一个很慢的 syscall，慢到不如你自己手写一个 page table。** 

这跟 OS 设计完全相反 — OS 的 page table 改起来是 hardware-fast 的（TLB miss 是纳秒级），user-space 看不到但也不需要看到。GPU 这边反过来 — user 想自己管，driver 不让；driver 自己管，又管得很慢。

这背后其实是一个 API philosophy 的冲突：
- 旧时代（OpenGL 风格）：driver 帮你做一切，user 写代码简单
- 新时代（Vulkan / DX12 风格）：user 自己管一切，driver 做薄

Hardware VT 的 API 是旧时代设计的 — "shader 代码跟 regular texture 一样"是它的 selling point。但这个 selling point 恰恰让 advanced user 失去了所有 control，driver 内部又没把性能做好，结果两头不讨好。

作者的潜台词：把 tile map 暴露出来，让 user 自己管，driver 只负责 TMU 硬件路径 — 这才是对的 direction。

---

## 对你（Karpathy）可能感兴趣的连接点

1. **这跟 ML 里的 memory management 有结构相似性** — CUDA sparse memory / unified memory 也有类似问题，driver overhead vs user-managed。
2. **跟 GPU-driven rendering 的哲学冲突** — Nanite 之所以强大，是因为 GPU 自己做所有决策；VT 的 CPU-side binding 是反方向的。
3. **API design 的普遍教训** — "让 user 看不到复杂性" 和 "给 user 足够 control" 是永恒张力，hardware VT 选错了边。
4. **开源 driver 的价值** — NVK 和 ANV 能做快，证明社区驱动有时比 vendor 自己更上心，这跟 ML 开源生态的观察一致。

参考链接汇总：
- Paper: https://hal.science/hal-05138369
- GigaVoxels DP: https://doi.org/10.1145/3675389
- id Tech 5 MegaTexture course: https://doi.org/10.1145/2343483.2343488
- Vulkan sparse memory spec: https://docs.vulkan.org/spec/latest/chapters/sparsemem.html
- WebGPU sparse texture discussion: https://github.com/gpuweb/gpuweb/issues/455
- 测试代码: https://gitlab.com/rcmz/sparsetexturestests
- Nvidia Maxwell whitepaper: https://www.ece.lsu.edu/gp/refs/GeForce-GTX-980-Whitepaper-FINAL.pdf

---

# The Sad State of Hardware Virtual Textures — 一篇 GPU API "吐槽论文" 的深度剖析

Andrej，这篇 paper 是 INRIA Grenoble 的 Antoine Richermoz 和 Fabrice Neyret 在 2025 年 7 月挂在 HAL 上的一个 short paper (13 页，含大量 GLSL snippet)。表面看是一个 survey + benchmark，但本质是写给 GPU vendor 和 driver writer 的一封公开信，控诉 "hardware virtual textures" 这件事在 hardware 层做了十几年，却依然处于一个 unusable 的尴尬境地。Paper 的标题用了 "Sad State"，不是 "Limitations of"，语气是非常直接的。

参考链接：
- 论文原文: https://hal.science/hal-05138369
- GigaVoxels DP (作者前作): https://doi.org/10.1145/3675389
- id Tech 5 / MegaTexture (Sean Obert, JP van Waveren, Graham Sellers): https://doi.org/10.1145/2343483.2343488
- Vulkan sparse memory standard tile sizes: https://docs.vulkan.org/spec/latest/chapters/sparsemem.html#sparsememory-standard-shapes
- ARB_sparse_texture2: https://registry.khronos.org/OpenGL/extensions/ARB/ARB_sparse_texture2.txt
- HLSL tiled resources: https://learn.microsoft.com/en-us/windows/win32/direct3d11/hlsl-tiled-resources-exposure
- MSL sparse textures: https://developer.apple.com/documentation/metal/reading-and-writing-to-sparse-textures
- GPUinfo Vulkan database: https://vulkan.gpuinfo.org
- CUDA programming guide (sparse): https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
- WebGPU sparse texture discussion: https://github.com/gpuweb/gpuweb/issues/455
- Unreal Engine SVT: https://dev.epicgames.com/documentation/en-us/unreal-engine/streaming-virtual-texturing-in-unreal-engine
- Pema, reverse engineering mipmap selection: https://pema.dev/2025/05/09/mipmaps-too-much-detail
- Nvidia Maxwell GTX 980 whitepaper: https://www.ece.lsu.edu/gp/refs/GeForce-GTX-980-Whitepaper-FINAL.pdf

---

## 1. 核心问题：为什么 texture 需要 virtualization

Regular texture 的内存分配是 "all or nothing"。一个 128k×128k 的 RGBA8 texture 需要 64GB VRAM — 装不下、装下了也几乎全浪费，因为：

- **Partial visibility**：地形 texture 大量像素落在 view frustum 外，或被 occluder 挡住 (Figure 2 left, 区域 C 和 D)。
- **Partial emptiness**：3D voxel texture 在物体之间的空腔里全是空气，但仍然占内存 (Figure 2 right)。
- **Distance-based resolution**：远处只需要低 mip，高分辨率 tile 根本没人去 sample。

类比一下，这和 OS 的 demand paging 是同一个 motivation：进程的 virtual address space 远大于 physical RAM，但实际 working set 很小。Texture 也一样 — virtual texture space 可以远超 VRAM，只要保证 camera 真正 touch 的 tile 被驻留即可。

Rage (id Software, 2011) 是这个 idea 的标志性产品：一个 128k×128k 的 MegaTexture 覆盖整个 game world，运行在 256MB VRAM 的 PS3 上。这正是 John Carmack 推动 id Tech 5 的核心思路。但当时 hardware 还没有 sparse texture，所以 Rage 的 MegaTexture 是纯 software 实现的，用了大量 console-specific 优化。

---

## 2. Software Virtual Texture 的实现 — 两个组件

### 2.1 Architecture

```
   Virtual UV space                    Physical atlas
   ┌──────────┐                        ┌──────────────┐
   │ v0 │ v1 │                          │              │
   ├────┼────┤   tile map (lookup)      │ p0 │ p1 │ .. │
   │ v2 │ v3 │  ──────────────────►    ├────┴────┴────┤
   └──────────┘                        │ pn │ .. │ .. │
                                       └──────────────┘
   每个 virtual tile vi          每个 physical tile pi 在 atlas 里有一个
   映射到一个 physical tile      实际占用的内存块，按需 allocate
```

两个核心资源：
1. **Atlas texture** — 一个 regular texture，physical tile 都塞在这里面。
2. **Tile map** — 一个 lookup table，记录 virtual tile → physical tile 的映射。

### 2.2 Texel fetching (Listing 1)

```glsl
vec4 texelFetchVirtualTexture(ivec2 p) {
    ivec2 virtualTile = ivec2(p / tileSize);
    ivec2 physicalTile = texelFetch(tileMap, virtualTile, 0).xy;
    ivec2 atlasP = physicalTile * tileSize + p % tileSize;
    return texelFetch(atlas, atlasP, 0);
}
```

变量含义：
- `p` : 整数 texel 坐标，在 virtual texture space 里
- `tileSize` : tile 的边长 (texel 数)，例如 128
- `virtualTile = p / tileSize` : floor 除法得到 virtual tile index (2D)
- `physicalTile` : 从 tile map 查到的 physical tile 在 atlas 中的起点坐标
- `atlasP = physicalTile * tileSize + p % tileSize` : 把"在 virtual tile 内部的偏移" `p % tileSize` 加到 "atlas 里的 tile 起点" 上，得到 atlas 中的绝对坐标

加入 mip level 后，virtual tile 多了一个 `lod` 维度，tile map 变成 3D（u, v, lod）→ physical tile (Listing 2)。

### 2.3 Filtering 的真正难点

这是 software VT 最 messy 的部分。Bilinear filtering 在 tile 内部没问题，但跨 tile 边界时，virtual space 中相邻的 texel 在 atlas space 里几乎肯定不相邻 — 因为 atlas 是按需 pack 的，tile 之间的物理邻居关系由 tile map 决定，跟 virtual space 的拓扑完全无关。

Hardware bilinear 会拿 atlasUV 周围的 4 个 texel 做 `(1-α)(1-β)` 加权，结果采样到完全无关的 neighboring physical tile 的内容。

Paper 列出三种解决方案：

#### (a) Manual interpolation
在 tile 边界处手工 fetch 邻居 tile 的边界 texel 并自己做加权。Cost：edge 上 2× sample，corner 上 4× sample，3D corner 上 8× sample。

#### (b) Stochastic interpolation
按 bilinear coefficient 当概率，随机只采一个 touched tile，靠 TAA 时域收敛。Stochastic 的方差会变成 tile 边界上的 shimmer，但如果 TAA 已经在管线上，这几乎免费。Unreal Engine 走的就是这条路用于 mipmap filtering。

#### (c) Tile borders
在 atlas 中每个 physical tile 外面套一圈 border texel，把 virtual neighbor 的对应 texel 拷过来 (Figure 3)。这样 hardware bilinear 在 tile 边界处也能拿到正确的相邻值。是最常用方案，但有 memory overhead。

### 2.4 Border memory overhead 的公式

对一个 2D tile，virtual tile size 为 $N \times N$，单边 border 宽度为 $B$：
- physical tile 尺寸 = $(N + 2B) \times (N + 2B)$ (double border) 或 $(N + B) \times (N + B)$ (single border, 只在 +u +v 方向加 border)
- 每 tile overhead = physical 面积 − virtual 面积

对 double border:
$$\text{overhead}_{\text{per-tile}} = (N+2B)^2 - N^2 = 4NB + 4B^2$$
$$\text{overhead ratio} = \frac{4NB + 4B^2}{N^2} = \frac{4B}{N} + \frac{4B^2}{N^2}$$

变量含义：
- $N$ : virtual tile 的边长 (texel)，是设计参数，典型 32/64/128
- $B$ : border 宽度 (texel)。Bilinear 需要 $B \geq 1$，Mipmap border 需要 $B$ 让 tile 自己包含 mip 链，hardware anisotropic 需要 $B \geq A_{\max}$ (允许的最大 anisotropy，因为各向异性探针 footprint 在 worst case 沿一个方向延伸 $A_{\max}$ texel)

Paper 给的 128×128 tile 例子（从 table 推算）：
| Filtering        | Border B | Overhead ratio |
|------------------|----------|----------------|
| Bilinear (single)| 1        | ≈ 3.1%         |
| Bilinear (double)| 1        | ≈ 6.3%          |
| Mipmap border    | (含 mip)| ≈ 33%+         |
| Aniso 16× (double)| 16       | ≈ 56.3%         |

最后一项：$\frac{4 \cdot 16}{128} + \frac{4 \cdot 16^2}{128^2} = 0.5 + 0.0625 = 56.25\%$。这就是为什么 UE 默认 cap anisotropic 到 4× 并改用 stochastic mipmap。

### 2.5 Anisotropic + tile border 的额外坑 (Listing 5)

Hardware anisotropic filter 内部要用 screen-space gradient (`dFdx`, `dFdy`) 来决定 footprint 形状。如果直接喂 atlasUV，gradient 是 atlas space 的，跟 virtual space 完全脱钩 — 边界处会拿到错误的 anisotropy shape。

解决：用 virtual UV 的 gradient，乘以缩放系数 `gradScale = (virtualTextureSize >> lod) / textureSize(atlas, 0)`，再用 `textureGrad(atlas, atlasUV, gradX, gradY)` 显式喂进去。这是 GLSL 中很常见的"骗过 hardware 计算 gradient"的 trick。

```glsl
vec2 gradScale = vec2(virtualTextureSize >> lod) / textureSize(atlas, 0);
return textureGrad(atlas, atlasUV, dFdx(uv) * gradScale, dFdy(uv) * gradScale);
```

### 2.6 Data structures for tile map

| 实现                    | 优点                            | 缺点                                |
|------------------------|--------------------------------|------------------------------------|
| Direct table (storage buffer) | $O(1)$ 查找，实现最简单       | 内存 $O(V)$，V = virtual tile 总数 |
| Tree (quadtree / octree)     | 内存紧凑，支持 sparse        | 查找 $O(\log V)$，shader 里递归    |
| Hash table                  | 紧凑且 $O(1)$ amortized      | 哈希冲突、需要 good hash          |

Paper 用的是 direct table + storage buffer，对于 512³ volume 还能接受。

Storage image vs sampled texture 选择会影响 texture unit / data bus / cache 的竞争 — 因为 atlas 也用 texture unit，两者争带宽。texelFetch 在某些硬件上绕开 texture unit 走 raw memory path，能缓解这个问题，但 vendor specific。

---

## 3. Hardware Virtual Texture 的历史与现状

### 3.1 Timeline

- **2012** 左右，Nvidia Maxwell (GTX 980) 引入 hardware sparse texture，AMD 推出 Partially Resident Textures (PRT)，Intel 跟进。
- API 覆盖面很广：OpenGL (`ARB_sparse_texture2`)、Vulkan (sparse memory)、D3D11/12 (Tiled Resources)、Metal (sparse texture)、GNM (PS)、CUDA。
- WebGPU 还在讨论中 (https://github.com/gpuweb/gpuweb/issues/455)。

Hardware VT 的核心吸引力：tile map 由 driver 管理，shader 代码和 regular texture 完全一致，硬件自动处理跨 tile 的 bilinear/mipmap/aniso — 对 shader writer 透明。

### 3.2 七大限制（按严重程度排序）

#### Limitation #1: Texture size cap
| Vendor | 2D max       | 3D max     |
|--------|--------------|------------|
| Nvidia | 32k × 32k    | 16k × 16k × 16k |
| AMD    | 16k × 16k    | 8k × 8k × 8k    |
| Intel  | 16k × 16k    | 2k × 2k × 2k    |

这直接和 virtual texture 的初衷相悖 — VT 本来就是为了能做 "比 VRAM 大得多" 的 texture。32k² × 4B = 4GB，已经不算 "giant" 了。CUDA 在关掉 `textureGather` 后能到 128k × 64k，说明硬件其实有能力，driver 限制了。

#### Limitation #2: Tile binding 性能差（论文核心 benchmark 之一）
最高可达 100ms bind 1000 tiles，远超 real-time 预算 (16.6ms / frame)。**这个限制几乎是 hardware VT 在 game 中不可用的根本原因**。

#### Limitation #3: Tile status 信息少
Shader 只能查到 "footprint 内所有 tile 是否都 bound" 这个 bool。无法知道具体哪个 tile 没 bound，也无法 query "某个 UV 的最低 bound mip level"。后者对 GigaVoxels 那种 out-of-core voxel streaming 是核心需求。

#### Limitation #4: Anisotropic filtering 支持不均
| Vendor | Max aniso |
|--------|-----------|
| Nvidia | 16×       |
| Intel  | 4×        |
| AMD    | **0× (不支持)** |

AMD 完全不支持 aniso on sparse texture，这是非常奇怪的不对称。

#### Limitation #5: GPU-side tile binding 不支持
现代 rendering 越来越 GPU-driven (indirect draw, mesh shader, bindless)。Tile binding 只能从 CPU 通过 API call 改，意味着每帧都要 CPU-GPU sync，破坏 GPU-driven 的整个 pipeline。作者的前作 GigaVoxels DP (SIGGRAPH I3D 2024) 展示了 GPU-side binding 能做多么强大的事情，但目前的 hardware VT 没法支撑。

#### Limitation #6: 缺乏 low-level API
Driver 把 tile map 整个藏起来了。Advanced user 想用更高效的 tile map 数据结构、想绕开 driver 的 indirection overhead，做不到。这是 binding 性能不稳的 root cause — driver 内部用一个未公开的 tile map 数据结构，可能很慢。

#### Limitation #7: Fixed 64KB tile size
Vulkan standard tile sizes 设计成 "tile 总是 64KB"，不管 dimensionality 或 format。好处：不同 format 的 tile 可以共享 tile pool；坏处：3D texture 64KB 不能开成完美立方体 (因为 $\sqrt[3]{64\text{KB}}$ 不是 2 的整数次幂)，导致非对称 tile shape。

---

## 4. Performance benchmarks — paper 的实证部分

### 4.1 Setup

| Vendor | GPU         | Drivers tested                           |
|--------|-------------|-----------------------------------------|
| AMD    | RX 6800     | AMD Windows, AMDGPUPRO, AMDVLK, RADV    |
| Nvidia | RTX 4080    | Nvidia Windows, Nvidia Linux, NVK       |
| Intel  | ARC A770    | Intel Windows, ANV                      |

CPU: Ryzen 9 7900X, Arch Linux kernel 6.12 / Windows 11. Test code: https://gitlab.com/rcmz/sparsetexturestests

### 4.2 Sampling benchmark (Figure 5)

场景：ray marcher 穿过一个 512³ mipmapped virtual texture，里面塞 FBM Perlin noise (Figure 4)。Resolution 1024²，无 shading，pure 烤 texture sampling 吞吐。

软件 VT 配置：virtual tile 32×32×16，single border 1 texel (physical tile 33×33×17)，direct table + storage buffer，no atlas mipmap (manual mip interp)。

**结论**：
1. **Driver 之间的差异巨大** — 同一张卡，不同 driver 性能能差出几倍。这本身就说明问题在 software stack 而非 silicon。
2. **Hardware VT ≈ regular texture** on most drivers — 说明一旦 hardware path 走通，sampling 几乎免费。
3. **Software VT ≈ 50% of hardware VT** — indirection 和 manual mip 是主要 overhead。

直觉解读：hardware VT 的 indirection 在 TMU 内部完成，跟 texture L1 cache 走的是同一条 fast path；software VT 的 indirection 要先发一次 texture fetch 查 tile map，再发一次 fetch atlas，TMU 之间流水线起来后差距不会到 2× 但仍然显著。

### 4.3 Binding benchmark (Figure 6) — paper 最有杀伤力的图

测试：随机非重叠绑定 1024 个 tile，取平均时间。

```
                Tile binding time (ms, log scale)
Software VT     ─── 几乎所有 driver 一致，< 1ms
HW Intel ANV    ─── 接近 software，~1-2ms
HW NVK          ─── 接近 software，~1-2ms
HW AMD RADV     ─── 中等
HW AMD Windows  ─── 慢
HW Nvidia Win   ─── 慢
HW AMDGPUPRO    ─── 非常慢
HW AMDVLK       ─── 极慢，接近 100ms
HW Nvidia Linux ─── 极慢
HW Intel Win    ─── 极慢
```

**最快与最慢相差约 300×**。这是一张相同的 RX 6800 在不同 driver 下的差异 — 说明 binding 慢的根因不在 hardware，而在 driver software 的 tile map 数据结构和 kernel call 路径。

为什么 binding 慢？作者推测：
- Driver 内部 tile map 数据结构可能很重 (比如带 locking、validation)。
- 每个 binding call 可能 trigger syscall 进入 kernel mode 管理 GPU memory (Linux 下 AMDGPU 的 sparse binding 走 drm AMDGPU ioctl，Windows 下 GPU memory management 走 kernel driver)。
- NVK (Mesa 的 Nouveau-based Vulkan driver for Nvidia) 和 ANV (Mesa Intel Vulkan) 都接近 optimal，证明这件事可以做快 — 只是 vendor 官方 driver 没去做。

---

## 5. 我对这篇 paper 的几个观察

### 5.1 它是一个 API design critique 而非算法 paper
没有新 algorithm、没有新 data structure。它的贡献是：把 "为什么 hardware VT 没人用" 这个 industry folklore 整理成一份带 benchmark 的可引用文档。这种 paper 在 graphics 领域不多见，但价值很高 — 它给了 community 一个共同的 reference point。

### 5.2 跟 OS virtual memory 的对照
| OS VM                          | GPU Virtual Texture              |
|--------------------------------|----------------------------------|
| Page (4KB)                     | Tile (64KB standard)            |
| Page table (multi-level)       | Tile map (driver internal)       |
| TLB (MMU cached)              | TMU 内 indirection cache         |
| `mmap` / `mprotect` syscall    | `vkBindSparseMemory` API call    |
| Demand paging (page fault)    | Tile residency, sparse binding   |
| User-space page table mgmt? No | GPU-side tile binding? **No** ← 限制 |

最大的 design failure 在最后一行：OS 也不允许 user-space 改 page table（除非用 `userfaultfd` 或 huge page reservation），但 OS 的 page fault handler 是 hardware-fast 的；GPU sparse binding 反而要 CPU driver + kernel round-trip，性能反而比 software 自己管 tile map 还慢几个数量级。这是"过度封装"的典型 — driver 想做太多事，结果做成了 perf bottleneck。

### 5.3 "透明 API"的诅咒
Hardware VT 的 selling point 是 "shader 代码和 regular texture 一样"。但这恰恰意味着 advanced user 失去了所有 control：无法换 tile map 数据结构、无法 query 具体 tile status、无法 GPU-side 修改。这跟现代 graphics 的"low-level, explicit"趋势（Vulkan、DX12、Mesh Shader、Bindless）完全相反。Paper 在 Limitation #6 (low-level API) 里提出一个 future API design：sampler 函数接受 (tile indices list, local UV) 而非 (virtual UV)，把 indirection 完全交给 user。这跟 storage buffer + manual indirection 的 software VT 其实很像，但享受 hardware TMU 的 aniso / mipmap / cache 路径。这可能是真正的 sweet spot。

### 5.4 与 GPU-driven rendering 的张力
现代 AAA 渲染管线（例如 Nanite, GPU scene）的趋势是 GPU-side 决策：GPU 自己决定 LOD、自己选 mesh、自己 cull。Tile binding 还要 CPU 介入是显性的 bottleneck。Nanite 实际上避开了 hardware VT，用 virtual geometry 而非 virtual texture（虽然也用了 virtual texture for streaming). GigaVoxels DP (Richermoz & Neyret 2024, https://doi.org/10.1145/3675389) 展示了 GPU-side sparse binding 能让大型 voxel world 实现无 stall 的 streaming — 这是 hardware VT 当前做不到的。

### 5.5 测试的 limitations 我自己补充
- 只测 Vulkan，OpenGL / D3D12 / Metal 行为可能不同。
- 只测 desktop GPU，移动 GPU (Mali, Adreno) sparse texture 支持很有限（Apple silicon 上 Metal sparse texture 是另一套故事）。
- Binding 测试是 "cold bind" 还是 "warm bind"（同一 tile 反复 bind/unbind）？driver 可能有 cache。
- 没有测试 sparse binding 的 synchronization overhead（VkFence / timeline semaphore），实际管线中这部分 cost 不可忽略。

### 5.6 这个问题的历史类比
- 早期 CPU 有 software TLB miss handler，后来全部 hardware 化。GPU VT 似乎也该走这条路。
- 但 CPU page table 是 OS 标准化的（每个 OS 一个 page table format），GPU tile map 是 vendor 私有的。这就像 PCI 之前每个显卡 vendor 自己写 driver API 的时代。
- Vulkan 的 sparse binding 是 API 层标准化的，但 driver 内部 tile map 还是 vendor 私有。要彻底解决，需要 hardware-level standardization，类似 CPU MMU 的 page table entry format (x86 PTE)。

---

## 6. 推荐你 (Andrej) 联想的相关工作

如果你对这块感兴趣：

1. **GigaVoxels** (Crassin et al. 2009, https://maverick.inria.fr/Publications/2009/CNKED09/) — GPU sparse voxel rendering 的奠基工作，software VT for 3D。
2. **Clipmaps** (Tanner et al. 1998, https://www.microsoft.com/en-us/research/wp-content/uploads/2004/07/tr-2004-66.pdf) — terrain texture streaming 的早期工作，"nested ring" cache。
3. **id Tech 5 MegaTexture** (Sean Obert, JP van Waveren, Graham Sellers, SIGGRAPH 2012 course, https://doi.org/10.1145/2343483.2343488) — paper 引用 [Obert et al. 2012] 的原始来源，Carmack 推动。
4. **Sparse Voxel Octrees** (Jon Olick, SIGGRAPH 2009) — 另一种 sparse voxel 表达。
5. **GigaVoxels DP** (Richermoz & Neyret 2024) — 作者前作，GPU-side tile binding 的实现。
6. **Nanite** (Brian Karis, SIGGRAPH 2021) — Epic 的 virtualized geometry，可以看成 VT 的"几何版本"，关键是 GPU-side LOD selection 完全避开了 CPU API call。
7. **Pema blog on mipmap selection reverse engineering** (https://pema.dev/2025/05/09/mipmaps-too-much-detail) — paper 引用，讲 hardware `textureQueryLod` 的精确实现细节，对 software VT 计算 lod 很关键。

---

## 7. 一句话总结

Paper 用实测数据揭示了一个 graphics community 都隐约知道但没人写下来的事实：**GPU 的 hardware sparse texture API 在 sampling 上几乎免费、和 regular texture 一样快，但在 tile binding 这件事上比你自己写 software VT 慢 100× 到 300×，而 NVK 和 ANV 两个 Mesa 驱动证明这件事是可以做快的 — AMD 和 Nvidia 的官方驱动只是没去做**。这个状态是 hardware VT 普及的最大障碍，也是 GPU-driven rendering pipeline 整合 virtual texture 的核心 blocker。作者的呼吁是：vendor 把 tile map 暴露成 low-level resource，让 user 用 storage buffer + hardware TMU indirection 的混合方案，而不是把整件事塞进 driver black box。
