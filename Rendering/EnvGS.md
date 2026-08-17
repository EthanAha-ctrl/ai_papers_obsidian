---
source_pdf: EnvGS.pdf
paper_sha256: 7931b147b5d1f0518829240108de6914254a79119fe1e8d7a180087ac30ba769
processed_at: '2026-08-04T04:41:41-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 EnvGS

## 一、一句话讲清楚在干嘛

你想用一组照片重建一个 3D 场景，场景里有镜子、金属、湿润的地面这些会反射的东西。3DGS / 2DGS 重建出来的反射全是糊的，因为它们用 Spherical Harmonics 表达"从不同角度看同一个点的颜色变化"，SH 太低频了，搞不定高频的镜面反射。

EnvGS 的做法很简单粗暴：**在场景里再放一套 Gaussian primitives 当"环境"，渲染的时候对每个像素算一下它的反射方向，沿这个方向去 trace 这套"环境 Gaussian"，把 trace 到的颜色当成反射颜色**。

就这么个事。听起来很自然，但工程上极难，因为 ray tracing 在 3DGS 生态里从来不是件容易事。

参考：https://zju3dv.github.io/envgs

## 二、为什么之前的方法不行

### 2.1 原版 3DGS / 2DGS

每个 Gaussian 存一组 SH 系数，渲染时根据 view direction 算出颜色。SH 本质是对球面函数做低阶多项式展开，3 阶 SH 一共 16 个系数，你能编码的频率信息很有限。镜面反射是高频信号——你稍微转一点头，反射内容就剧烈变化——SH 根本表达不了，所以反射区域永远是糊的或者"鬼影"。

### 2.2 GaussianShader / 3DGS-DR

这俩很自然地想到：既然 SH 不行，我加一张 environment map（cubemap）嘛。environment map 就是把场景周围的光照信息存在六张贴图里，渲染时根据反射方向去查这张贴图，拿到反射颜色。

问题有两点：

**第一，environment map 假设所有光源都在无穷远**。你站在一辆车前面看车身的反射，反射到的可能是你身后的树、房子、天空——这些都在远处，environment map 没问题。但如果你看一个厨房的水龙头，水龙头反射到的是旁边的洗碗池、水槽边缘——离它只有几厘米。environment map 把这些近场物体"压扁"到无穷远，位置信息全丢了，近场反射必然错。

**第二，environment map 是低分辨率 2D 贴图**。典型 128 或 256 分辨率，高频细节根本存不下。而且 cubemap 是固定网格采样，没法根据场景复杂度自适应地往高频区域塞更多细节。

参考：
- GaussianShader: https://arxiv.org/abs/2311.17977
- 3DGS-DR: https://arxiv.org/abs/2405.12235

## 三、EnvGS 的核心 trick

**把"环境"从 2D 贴图换成 3D 空间里的一套 Gaussian primitives**。

这套 environment Gaussian 跟 base Gaussian 一样，每个都有 position、scale、rotation、opacity、SH coefficients。区别在于它们扮演的角色：

- base Gaussian：负责场景的几何（表面在哪里、normal 朝哪）+ 漫反射颜色；
- environment Gaussian：负责"环境光照"，被反射 ray 命中时给出反射颜色。

因为 environment Gaussian 在 3D 空间里有真实位置，近场物体（旁边几厘米的水槽）和远场物体（远处的天空）都被统一编码在同一个 3D 空间里，没有"近场丢失"问题。又因为 Gaussian 可以 densify、可以自适应调整 scale，高频反射细节能被精准捕捉——哪里反射复杂，optimization 自然往那里多塞 Gaussian。

这套思路很"物理"——反射在真实世界就是光线打到表面后弹到环境里某个物体上，那个物体的颜色就是这个像素的反射颜色。EnvGS 把这个过程显式模拟出来。

## 四、渲染流程拆解

对应 paper Figure 2，每帧渲染分三步：

### Step 1：Base pass（rasterization）

用标准 2DGS 的 rasterizer 把 base Gaussian 渲一遍。对每个像素得到：

- surface position $\mathbf{x}$：这条 camera ray 打到的表面点的 3D 坐标；
- normal $\mathbf{n}$：该点的表面法向；
- base color $\mathbf{c}_{base}$：漫反射部分的颜色（来自 SH）；
- blending weight $\beta$：这个像素的"反射强度"，由 base Gaussian 自己学的标量。

这些都是按 alpha-blending 累加的：

$$v = \sum_{i} v_i \alpha_i \prod_{j<i}(1-\alpha_j), \quad v \in \{\mathbf{x}, \mathbf{n}, \beta\}$$

变量解释：$v_i$ 是第 $i$ 个 Gaussian 的属性（position/normal/weight），$\alpha_i$ 是它的 alpha（opacity × Gaussian 响应），$\prod_{j<i}(1-\alpha_j)$ 是它前面所有 Gaussian 的累积透射率。这就是 NeRF/3DGS 的标准 volume rendering 公式。

### Step 2：Reflection pass（ray tracing）

对每个像素，算反射方向：

$$\mathbf{d}_{ref} = \mathbf{d}_{cam} - 2(\mathbf{d}_{cam} \cdot \mathbf{n})\mathbf{n}$$

这就是镜面反射公式，$\mathbf{d}_{cam}$ 是从相机指向表面的方向，$\mathbf{n}$ 是表面法向，$\mathbf{d}_{ref}$ 是反射后的方向。任何入门图形学教材都讲这个。

然后从 $\mathbf{x}$ 出发，沿 $\mathbf{d}_{ref}$ 方向射一条 ray，去 environment Gaussian 上做一次 volume rendering（跟 Step 1 一样的公式，只不过 ray 从反射方向射出去），得到反射颜色 $\mathbf{c}_{ref}$。

### Step 3：Blending

$$\mathbf{c} = (1-\beta)\mathbf{c}_{base} + \beta\mathbf{c}_{ref}$$

$\beta$ 大就是镜面区域（金属、玻璃），$\beta$ 小就是漫反射区域（木头、布料）。$\beta$ 是每个 base Gaussian 自己学的，相当于把"材质类型"软分配给每个 Gaussian。

## 五、工程上的难点：为什么 ray tracing 这一步这么难

### 5.1 为什么不能用 rasterizer

3DGS 的 rasterizer 是 tile-based：把每个 Gaussian 投影到屏幕上一组 tiles，每个 tile 内的像素遍历这组 Gaussian。这套设计假设所有 primitive 都被同一个相机的 view frustum 看到。

但 reflection pass 不一样——每个像素的反射方向 $\mathbf{d}_{ref}$ 都不一样，相当于每个像素是一个独立"虚拟相机"，朝向不同方向。一个 environment Gaussian 可能被像素 A 的反射 ray 看到，也可能被像素 B 的反射 ray 看到，但投影到屏幕上的位置完全没规律。rasterizer 的 tile 假设彻底失效。

### 5.2 为什么用 OptiX RT core

NVIDIA 的 RT core 硬件加速 ray-triangle intersection，专门干这个。EnvGS 用 OptiX 框架（NVIDIA 的 ray tracing engine），每个 2D Gaussian 被转成两个三角形塞进 BVH。

具体怎么转：在 Gaussian 的 local tangent plane 上取 $3\sigma$ 范围的四个角点 $(-3,-3), (3,-3), (3,3), (-3,3)$，通过变换矩阵 $\mathbf{H}$ 映射到世界坐标，连成两个三角形。$3\sigma$ 是因为 Gaussian 在 $3\sigma$ 外的值已经几乎为 0，再大没意义。

然后 BVH 是经典的加速结构，ray 先跟 BVH 的 bounding box 做"粗测"，命中后再跟里面的三角形做"细测"。RT core 把这部分硬件化，每秒能跑几亿次 ray-triangle test。

### 5.3 Chunk-based traversal

OptiX 提供 `raygen` 和 `anyhit` 两个 programmable entry point。EnvGS 借鉴 NVIDIA 的 3D Gaussian Ray Tracing paper 的设计：

- `raygen`：发射 ray，启动 BVH traversal；
- `anyhit`：每次 BVH 命中一个 triangle，把它塞进一个 sorted k-buffer（按深度排序），$k=16$；
- k-buffer 满了之后，回到 `raygen` 把这 16 个 Gaussian 按 volume rendering 公式积分；
- 然后继续 traversal 取下 16 个；
- 重复直到没有更多 intersection 或者累积透射率降到阈值以下。

$k=16$ 是 trade-off：太小 traversal 次数多，太大每次 sort 16 个 Gaussian 排序成本高。实测 16 最优。

参考：https://arxiv.org/abs/2406.11848 (3D Gaussian Ray Tracing)

### 5.4 Backward pass 的坑

经典 3DGS 的 backward 是把 forward 时存下的所有 intersection 按 back-to-front 顺序遍历算梯度。EnvGS 没法这么干——每条 ray 的 intersection 数量可能上百个，存全部 intersection 显存爆了。

EnvGS 的方案：**重新 cast ray，按 front-to-back 顺序边遍历边算梯度**。这要求计算对 ray origin 和 direction 的梯度，因为 reflection ray 的 origin 和 direction 都来自 base pass 的 $\mathbf{x}$ 和 $\mathbf{n}$，而 $\mathbf{x}, \mathbf{n}$ 又是 base Gaussian 参数的函数。

公式在 paper 附录 D：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{o}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} + \frac{\partial \mathcal{L}}{\partial t} \cdot \frac{-\mathbf{n}_i}{\mathbf{n}_i^\top \mathbf{d}}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{d}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} \cdot t_i + \frac{\partial \mathcal{L}}{\partial t} \cdot \frac{-\mathbf{n}_i^\top(\mathbf{v}_1 - \mathbf{o})}{\mathbf{n}_i^\top(\mathbf{n}_i^\top \mathbf{d})^2}$$

变量解释：
- $\mathbf{o}$：ray origin；
- $\mathbf{d}$：ray direction；
- $\mathbf{x}_i = \mathbf{o} + t_i\mathbf{d}$：交点位置；
- $t_i$：ray 到三角形的有向距离；
- $\mathbf{n}_i$：三角形法向（叉乘出来的，未归一化）；
- $\mathbf{v}_1$：三角形顶点之一。

第一个公式：对 origin 的梯度有两项，第一项来自 $\mathbf{x}_i$ 对 $\mathbf{o}$ 的直接偏导（就是 identity，因为 $\mathbf{x}_i = \mathbf{o} + t_i\mathbf{d}$），第二项来自 $t_i$ 依赖于 $\mathbf{o}$（因为 $t_i = \mathbf{n}_i^\top(\mathbf{v}_1 - \mathbf{o}) / \mathbf{n}_i^\top \mathbf{d}$）。

第二个公式：对 direction 的梯度也有两项，第一项来自 $\mathbf{x}_i$ 对 $\mathbf{d}$ 的偏导（乘上 $t_i$），第二项来自 $t_i$ 依赖于 $\mathbf{d}$（分母里 $\mathbf{n}_i^\top \mathbf{d}$ 求导产生平方项）。

这两个梯度让 environment Gaussian 渲染出的反射颜色误差能反传回 base Gaussian，驱动 base Gaussian 的几何参数更新。**这是 joint optimization 的数学基础**。ablation 里去掉 joint optimization，PSNR 从 24.617 掉到 24.034，几何质量显著恶化——证明这个梯度流至关重要。

## 六、为什么必须用两套 Gaussian（base + env）

paper 附录 B.3 试过用一套 Gaussian 同时干这两件事：先 trace camera ray 拿到 base color 和 normal，再 trace reflection ray 拿到 reflection color。结果训练发散。

直觉解释：同一套 Gaussian 既要被 camera ray 命中又要被 reflection ray 命中，两条 ray 的"诉求"互相冲突。camera ray 希望这个 Gaussian 长在表面正确位置，reflection ray 希望这个 Gaussian 长在反射看到的位置——这两个位置在物理上经常不一致（你在镜子里看到的东西在镜子后面，但镜子表面的 Gaussian 应该长在镜子前面）。一套 Gaussian 同时满足两个约束会发散。

两套分离就清爽：base Gaussian 只管表面几何 + 漫反射，env Gaussian 只管"被反射看到的环境"，role 不冲突。env Gaussian 在 3D 空间里的位置可以任意，不需要对应任何真实表面——它只是"光照探针"。

这跟传统 graphics 里"几何 vs 光照探针"分离的思路一致，只不过 EnvGS 把两者都用 Gaussian primitives 统一表达。

## 七、训练和初始化的小细节

1. **先训 base Gaussian 一段时间**（bootstrapping），让几何大致稳定。如果一开始就 joint train，env Gaussian 会被错误的几何带跑偏。
2. **env Gaussian 初始化**：场景 bounding box 切成 $32^3$ 个 sub-grid，每个 grid 随机采 5 个 Gaussian，总共约 16 万个。这样空间均匀分布，避免一开始就在某些区域"瞎"。
3. **Densification criterion 修改**：3DGS 用"投影到屏幕的 2D 梯度积累"判断哪里需要 densify。ray tracing 没有"投影到 2D"的概念，所以 EnvGS 直接用 3D 空间 position 梯度积累，而且每个积累梯度乘以 intersection depth 的一半——**防止远处 under-densification**。远处 Gaussian 在屏幕上投影小，2D 梯度自然小，但远处其实需要更多 Gaussian 才够分辨率，所以乘 depth 给远处"加权"。
4. **最终 env Gaussian 约 30 万个**，占 base Gaussian 的 15%，显存 70MB。Pruning 时保留 rendering weight top 63 万个。

## 八、Loss 设计的直觉

$$\mathcal{L} = \mathcal{L}_{rgb} + 0.04\mathcal{L}_{norm} + 0.01\mathcal{L}_{mono} + 0.01\mathcal{L}_{perc}$$

- $\mathcal{L}_{rgb}$：0.8 L1 + 0.2 D-SSIM，常规重建 loss；
- $\mathcal{L}_{norm}$：rendered normal vs depth-derived normal 一致性，来自 2DGS；
- $\mathcal{L}_{mono}$：用预训练 monocular normal estimator（StableNormal 或 Lotus）监督 rendered normal；
- $\mathcal{L}_{perc}$：VGG-16 perceptual loss。

**为什么需要 $\mathcal{L}_{mono}$**：在反射/折射歧义表面，depth-derived normal 会错——比如镜子反射出一个深处的物体，深度图梯度可能把 normal 算反。Monocular normal 是 2D image prior，不依赖几何，能在这些歧义场景给出更鲁棒的引导。ablation 里去掉这个 loss，PSNR 从 24.617 掉到 24.107，反射重建在歧义表面错位明显。

参考：
- StableNormal: https://arxiv.org/abs/2407.18740
- Lotus: https://arxiv.org/abs/2409.18124

## 九、实验数据的解读

### 9.1 主表（Table 1）

EnvGS PSNR 24.617，跟 SOTA non-real-time NeRF-Casting 的 24.670 几乎一样，但快 100 倍（26 FPS vs <0.1 FPS），训练时间 2.5h vs >47h。在 real-time method 里全面领先 3DGS-DR 1.095 PSNR。

### 9.2 反射区域单独看（Table 2）

这是最 key 的表——在反射 foreground 区域 EnvGS 33.295，3DGS-DR 31.814，提升 1.481；在 near-field 区域 EnvGS 46.392，3DGS-DR 44.007，提升 2.385。**near-field 提升 2.4 PSNR 正是 environment Gaussian 相对 environment map 的核心优势**——近场反射靠 3D 空间位置编码，env map 搞不定。

### 9.3 速度 ablation（Table 6）

- PyTorch 手算 ray-primitive intersection：FPS 1/6157（一帧 1.6 小时）
- 1×1 tile rasterizer：FPS 1/11902（一帧 3.3 小时）
- EnvGS OptiX：32 FPS
- 只对高 $\beta$ 像素 trace（80% weight filtering）：44 FPS，质量损失 0.187 PSNR

PyTorch 和 rasterizer 路线完全不可用，OptiX RT core 是 enabler。这个数字对比非常有说服力——硬件 ray tracing 加速是 method 成立的必要条件。

### 9.4 关键 ablation（Table 4）

- w/o joint optimization: -0.583 PSNR
- w/o mono normal: -0.510 PSNR
- w/ environment map（替换 Gaussian env）: -0.472 PSNR
- w/o LPIPS loss: -0.050 PSNR（perceptual loss 贡献很小）

joint optimization 和 mono normal 是质量关键，environment Gaussian 表示本身也贡献了 0.5 PSNR。

## 十、跟相关工作的横向对比

### 10.1 vs NeRF-Casting

NeRF-Casting 思路非常类似（沿 reflection ray march），但用 implicit MLP，每条 ray 要 query 30+ 次 MLP，<0.1 FPS。EnvGS 用 explicit Gaussian，BVH traversal + Gaussian 评估，FPS 提到 26。**质量相当，速度 250×**——这就是 explicit representation 的胜利。

参考：https://arxiv.org/abs/2405.14871

### 10.2 vs 3D Gaussian Ray Tracing (NVIDIA)

3D GRT 把整个场景用 ray tracing 渲染，主要解决 3DGS 在 hair、mesh 等不规则场景的 aliasing。EnvGS 借鉴了它的 BVH + chunk traversal 框架，但只对 environment Gaussian 走 ray tracing，base Gaussian 仍用 rasterizer——**hybrid 设计保留了 rasterize 的高 FPS**。可以认为 EnvGS = 2DGS rasterizer + 3D GRT ray tracer + reflection ray 设计。

参考：https://arxiv.org/abs/2406.11848

### 10.3 vs Ref-GS（concurrent）

Ref-GS 用 directional factorization 把 2D Gaussian 的 SH 分解成 directional terms，单套 Gaussian 表达 view-dependent。参数更省，但表达能力理论上限受 factorization 设计约束。EnvGS 用两套 Gaussian 显式分离，参数更多但 role 清晰、优化稳定。两条路线对应 reflection 建模的两种 prior：implicit (factorized) vs explicit (secondary rays)。

参考：https://arxiv.org/abs/2412.00905

## 十一、局限和我的思考

paper 明确说的局限：**透明 / 折射材料搞不定**——目前只沿反射方向 trace，折射需要 Snell's law + transmission ray。要做玻璃这种材质，得加一套 refraction ray，复杂度翻倍。

我的几个联想：

**Relighting 方向**：env Gaussian 本质是 scene-aware lighting probe，但当前 base Gaussian 的"材质属性"是耦合在 SH + blending weight 里的，没有显式 BRDF 分解。如果要做 relighting（换 env），需要把 base 的 roughness、metallic 显式参数化，这跟 GS3 (https://arxiv.org/abs/2410.11419) 的 triple Gaussian splatting 思路接近。

**Glossy reflection**：当前 $\mathbf{d}_{ref}$ 是完美镜面方向，glossy 表面（半镜面，比如磨砂金属）应该沿一个 cone 采样多条 reflection ray 然后 average。可以做 importance sampling，cone 宽度由 roughness 参数控制，但 FPS 会按采样数线性下降。

**Multi-bounce**：当前只一次 bounce，镜子反射镜子的场景搞不定。可以做 recursive tracing（depth=2 或 3），但要权衡 FPS——每多一次 bounce，ray 数量乘以平均 cone 采样数。

**Dynamic scene**：当前 env Gaussian 是静态 3D 结构。如果场景有动态光源或动态反射物体，env Gaussian 要随时间变形，需要 4D Gaussian (4DGS, https://arxiv.org/abs/2310.12196) 框架。

**导出 environment map**：训练好后把 env Gaussian 渲染到 cubemap，可以导出作为离线 lighting asset 给传统 graphics pipeline 用。这样 EnvGS 训练出的 scene 既能 real-time 渲染，也能"导出知识"给其他系统。

**与 residual SH 互补**：当前 blending 是 $(1-\beta)\mathbf{c}_{base} + \beta\mathbf{c}_{ref}$，env Gaussian 主抓 high-frequency，base SH 抓 low-frequency diffuse。可以加一个 residual term $\gamma \mathbf{c}_{sh}$ 让 SH 也承担一部分中频 view-dependent，降低 env Gaussian 负担——可能能让 env Gaussian 数量减少，FPS 提升。

## 十二、整体直觉

EnvGS 的核心 insight 用一句话讲：**反射就是二次光线问题，把环境当成显式 3D 几何，沿反射方向做 volume rendering**。

这个 idea 本身很物理、很自然，难的是工程——把 ray tracing 做成可微、做成 real-time、做成能 joint optimize 几何和光照。NVIDIA 的 RT core + OptiX 是 enabler，可微 ray tracing 的梯度推导是 mathematical foundation，两套 Gaussian 分离是 optimization stability 的关键，monocular normal supervision 是 reflection ambiguity 的解药。

它代表 3DGS 生态从纯 rasterization 走向 hybrid rasterize + ray trace 的趋势。后续 inverse rendering、relighting、dynamic extension、multi-bounce 都可以在这套框架上叠加。短期内 reflection rendering 的 SOTA 基本就是这条路了——除非有更激进的 implicit representation 速度突破。

参考汇总：
- https://zju3dv.github.io/envgs (项目主页)
- https://arxiv.org/abs/2403.17888 (2DGS)
- https://arxiv.org/abs/2308.14773 (3DGS)
- https://arxiv.org/abs/2311.17977 (GaussianShader)
- https://arxiv.org/abs/2405.12235 (3DGS-DR)
- https://arxiv.org/abs/2111.15443 (Ref-NeRF)
- https://arxiv.org/abs/2405.14871 (NeRF-Casting)
- https://arxiv.org/abs/2406.11848 (3D Gaussian Ray Tracing)
- https://arxiv.org/abs/2306.10073 (Zip-NeRF)
- https://arxiv.org/abs/2412.00905 (Ref-GS)
- https://arxiv.org/abs/2410.11419 (GS3)
- https://arxiv.org/abs/2310.12196 (4DGS)
- https://arxiv.org/abs/2407.18740 (StableNormal)
- https://arxiv.org/abs/2409.18124 (Lotus)
- https://research.nvidia.com/labs/rt/gaussian-tracing/ (NVIDIA Gaussian Tracing)

---

# EnvGS 详解：用 Environment Gaussian 建模 View-Dependent Appearance

## 一、动机：为什么需要这篇 paper

3DGS / 2DGS 用 Spherical Harmonics (SH) 表达 view-dependent appearance，SH 的表达容量受限于低阶球谐展开，无法捕捉 high-frequency specular reflections。GaussianShader 和 3DGS-DR 引入 environment map 作为远场光照假设，问题在于：

1. environment map 假设**远场照明** (distant lighting)，场景里离反射表面很近的物体无法在 environment map 里得到准确的位置编码，所以**近场反射 (near-field reflection)** 失真；
2. environment map 是一张低分辨率 cubemap（典型 128 或 256），频率带宽有限，捕捉不到高频反射细节。

EnvGS 的核心 insight：**用另一套 3D Gaussian primitives 显式地表示"环境"，把反射问题转化成在 3D 空间里沿 reflection ray 进行 volume rendering 的问题**。这样近场/远场统一在 3D 空间表达，且 Gaussian primitives 通过自身 position、scale、SH 参数可以承载高频信息。

参考：
- 项目主页：https://zju3dv.github.io/envgs
- 2DGS: https://arxiv.org/abs/2403.17888
- 3DGS: https://arxiv.org/abs/2308.14773
- GaussianShader: https://arxiv.org/abs/2311.17977
- 3DGS-DR: https://arxiv.org/abs/2405.12235
- Ref-NeRF: https://arxiv.org/abs/2111.15443
- NeRF-Casting: https://arxiv.org/abs/2405.14871
- 3D Gaussian Ray Tracing (Moenne-Loccoz et al.): https://arxiv.org/abs/2406.11848

## 二、整体架构（对应 Figure 2）

渲染管线分三步：

1. **Base pass (rasterization)**：用标准 2DGS rasterizer 渲染 base Gaussian $\mathbf{P}_{base}$，得到每像素的 surface position $\mathbf{x}$、normal $\mathbf{n}$、base color $\mathbf{c}_{base}$、blending weight $\beta$；
2. **Reflection pass (ray tracing)**：对每个像素，根据 $\mathbf{x}, \mathbf{n}$ 计算反射方向 $\mathbf{d}_{ref}$，再在 environment Gaussian $\mathbf{P}_{env}$ 上沿 $(\mathbf{x}, \mathbf{d}_{ref})$ 做一次 volume rendering，得到 $\mathbf{c}_{ref}$；
3. **Blending**：最终颜色 $\mathbf{c} = (1-\beta)\mathbf{c}_{base} + \beta \mathbf{c}_{ref}$。

关键设计是 base 和 environment 用**两套独立 Gaussian**——论文附录 B.3 说明：如果用同一套 Gaussian 既做 base 又做 reflection，优化会发散，因为 suboptimal geometry 会把 reflection ray 导向错误方向打到错误的 Gaussian，造成训练不稳定。直觉上这跟"shadow ray 反复自我干扰"是同一类问题。

## 三、数学细节

### 3.1 2D Gaussian 参数化 (Eq. 1)

$$\mathbf{H} = \begin{bmatrix} s_u \mathbf{t}_u & s_v \mathbf{t}_v & \mathbf{0} & \mathbf{p}_k \\ \mathbf{0}^\top & 0 & 0 & 1 \end{bmatrix}$$

- $\mathbf{p}_k \in \mathbb{R}^3$：Gaussian 中心在世界坐标的位置；
- $(\mathbf{t}_u, \mathbf{t}_v) \in \mathbb{R}^3 \times \mathbb{R}^3$：局部 tangent plane 的两个主方向（正交单位向量），定义 Gaussian 的朝向；
- $(s_u, s_v) \in \mathbb{R}^2$：沿 $\mathbf{t}_u, \mathbf{t}_v$ 方向的 scale（标准差级别）；
- $\mathbf{H} \in \mathbb{R}^{4 \times 4}$ 是把局部 $[u,v,0,1]$ 映射到世界坐标的 transform。

2D Gaussian 比起 3D Gaussian 的好处：normal 由 $\mathbf{t}_u \times \mathbf{t}_v$ 直接定义，几何上更贴近真实表面，对反射计算至关重要。

### 3.2 Volume rendering (Eq. 2)

$$\mathbf{c} = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i, \quad \alpha_i = \sigma_i \mathcal{G}_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

- $N$：沿 ray 排好序的 Gaussian 数量；
- $\alpha_i$：第 $i$ 个 Gaussian 在交点处的 alpha（opacity × Gaussian 响应）；
- $\sigma_i$：opacity (learned)；
- $\mathcal{G}_i$：2D Gaussian 函数值（在 tangent plane 上的标准高斯）；
- $T_i$：前 $i-1$ 个 Gaussian 的累积透射率；
- $\mathbf{c}_i$：第 $i$ 个 Gaussian 的 view-dependent color（通过 SH 计算）。

### 3.3 Base pass 渲染 (Eq. 3)

$$v = \sum_{i \in \mathcal{N}} v_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j), \quad v \in \{\mathbf{x}, \mathbf{n}, \beta\}$$

这里把公式 2 推广到任意属性 $v$——surface position $\mathbf{x}$、normal $\mathbf{n}$、blending weight $\beta$ 都按 alpha-blending 集成。$\beta$ 是为每个 base Gaussian 学习的标量参数，决定这个像素应该多信任 reflection 通道。

### 3.4 Reflection direction (Eq. 4)

$$\mathbf{d}_{ref} = \mathbf{d}_{cam} - 2(\mathbf{d}_{cam} \cdot \mathbf{n})\mathbf{n}$$

这是标准的镜面反射公式：把入射方向 $\mathbf{d}_{cam}$（从相机指向表面）沿 normal $\mathbf{n}$ 翻转。注意 $\mathbf{n}$ 是从 base pass 渲染出来的 per-pixel normal，不是 per-Gaussian normal——这保证了反射方向在像素级别平滑。

### 3.5 Final blending (Eq. 5)

$$\mathbf{c} = (1-\beta)\mathbf{c}_{base} + \beta \mathbf{c}_{ref}$$

直觉：$\beta$ 大表示该像素是镜面反射区域（金属、玻璃、湿润表面），$\beta$ 小表示是漫反射区域。$\beta$ 是 per-base-Gaussian 学习的，等价于把"材质类型"软分配给每个 Gaussian。

### 3.6 Gaussian 响应评估 (Eq. 6)

$$\mathcal{G}_i(\mathbf{u}_i) = \mathcal{G}_i(\mathbf{H}^{-1}\mathbf{x}_i)$$

把世界坐标交点 $\mathbf{x}_i$ 通过 $\mathbf{H}^{-1}$ 变回 Gaussian 的局部 $(u,v)$ 坐标，然后评估标准 2D Gaussian 值。这步在 ray tracing 里很关键——BVH 找到 ray-triangle intersection 只是几何相交，Gaussian 的"软"响应还需要在 tangent plane 上重新评估。

## 四、可微 Ray Tracing Renderer

这是这篇 paper 工程上最难的部分。**Rasterization 没法用来渲染 environment Gaussian**，因为每个像素的 reflection ray 方向都不一样，相当于每个像素是一个独立的虚拟相机方向。3DGS 的 tile-based rasterizer 假设所有 primitive 投影到同一组 tiles，这套假设对 reflection ray 完全失效。

### 4.1 把 2D Gaussian 转成两个三角形

为了利用 GPU RT core (OptiX)，需要把每个 Gaussian 用几何 primitive 表示塞进 BVH。做法：

1. 在 local tangent plane 上定义四个 bounding vertex：
$$\mathbf{V}_{local} = \{(\text{sgn}(r), \text{sgn}(r))\}, \quad r=3$$
   即 $(-3,-3), (3,-3), (3,3), (-3,3)$，覆盖 $3\sigma$ 范围（Gaussian 在 $3\sigma$ 外几乎为 0）；
2. 通过 $\mathbf{H}$ 变换到 world space 得到 $\mathbf{V}_{world}$；
3. 拆成两个三角形插入 BVH。

这跟 3D Gaussian Ray Tracing (Moenne-Loccoz et al.) 的思路一致，参考 https://arxiv.org/abs/2406.11848。

### 4.2 Chunk-based traversal

利用 OptiX 的 `raygen` 和 `anyhit` 入口点：

- `raygen`：发射 ray，启动 BVH traversal，集成一个 chunk，然后调用 `anyhit` 取下一个 chunk；
- `anyhit`：每次 BVH 命中一个 triangle，把它塞进 sorted k-buffer（按 depth 排序），$k=16$ 是 trade-off —— $k$ 太小 traverse 次数多，$k$ 太大每次 sort 成本高；
- chunk 满后回到 `raygen`，按 Eq. 2 集成 Gaussian 属性；
- 重复直到没有更多 intersection 或 accumulated transmittance 低于阈值。

### 4.3 Backward pass 的设计

经典 3DGS backward 是按 back-to-front 顺序遍历存好的 intersection list。EnvGS 没法这么做，因为每个 ray 的 intersection 数量极大，存所有 intersection 的内存吃不消。他们的方案：**重新 cast ray，按 front-to-back 顺序计算每步的梯度**。

更关键的是要计算 $\partial \mathcal{L}/\partial \mathbf{o}$ 和 $\partial \mathcal{L}/\partial \mathbf{d}$（对 ray origin 和 direction 的梯度），因为 reflection ray 的 origin $\mathbf{o}=\mathbf{x}$ 和 direction $\mathbf{d}=\mathbf{d}_{ref}$ 都依赖于 base Gaussian 的参数（$\mathbf{x}, \mathbf{n}$ 来自 base pass）。没有这两个梯度，base 和 environment 就不能 joint optimize——ablation Sec 5.4 证明 joint optimization 对几何精度至关重要。

### 4.4 梯度公式 (Eq. 12-14)

给定 ray $(\mathbf{o}, \mathbf{d})$ 和一个三角形顶点 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$：

$$t_i = \frac{\mathbf{n}_i^\top(\mathbf{v}_1 - \mathbf{o})}{\mathbf{n}_i^\top \mathbf{d}}, \quad \mathbf{n}_i = (\mathbf{v}_2-\mathbf{v}_1)\times(\mathbf{v}_3-\mathbf{v}_1)$$

- $t_i$：ray 到 triangle 的深度；
- $\mathbf{n}_i$：triangle 法向（非归一化，大小为面积两倍）。

对 origin 的梯度：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{o}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} + \frac{\partial \mathcal{L}}{\partial t} \cdot \frac{-\mathbf{n}_i}{\mathbf{n}_i^\top \mathbf{d}}$$

第一项来自 $\mathbf{x}_i = \mathbf{o} + t_i\mathbf{d}$ 对 $\mathbf{o}$ 的直接偏导（identity），第二项来自 $t_i$ 依赖于 $\mathbf{o}$。

对 direction 的梯度：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{d}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} \cdot t_i + \frac{\partial \mathcal{L}}{\partial t} \cdot \frac{-\mathbf{n}_i^\top(\mathbf{v}_1 - \mathbf{o})}{\mathbf{n}_i^\top(\mathbf{n}_i^\top \mathbf{d})^2}$$

第一项来自 $\mathbf{x}_i$ 对 $\mathbf{d}$ 的偏导（乘 $t_i$），第二项来自 $t_i$ 对 $\mathbf{d}$ 的偏导（分母 $\mathbf{n}_i^\top \mathbf{d}$ 的平方项来自 chain rule）。

这两个梯度沿 base pass 的 $\mathbf{x}, \mathbf{n}$ 反向传播，使得 environment Gaussian 的渲染误差可以驱动 base Gaussian 的几何参数更新。**这是 EnvGS 能 joint optimize 的数学基础**。

## 五、Loss 设计

$$\mathcal{L} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_{norm} + \lambda_2 \mathcal{L}_{mono} + \lambda_3 \mathcal{L}_{perc}$$

- $\mathcal{L}_{rgb}$：0.8 L1 + 0.2 D-SSIM；
- $\mathcal{L}_{norm}$（Eq. 7）：rendered normal $\mathbf{n}$ 与 depth-derived normal $\mathbf{N}_d$ 的一致性，来自 2DGS；
- $\mathcal{L}_{mono}$（Eq. 9）：用预训练 monocular normal estimator（StableNormal, Lotus 等）监督 rendered normal，对于反射/折射歧义表面至关重要；
- $\mathcal{L}_{perc}$：VGG-16 perceptual loss；
- $\lambda_1=0.04, \lambda_2=0.01, \lambda_3=0.01$。

$\mathcal{L}_{mono}$ 的设计动机：纯几何约束（normal consistency）在反射/折射歧义表面会失效——一个看起来像镜子的区域，深度图梯度给出的 normal 可能完全错（因为深度被反射方向污染）。Monocular normal 是 2D prior，能在这些歧义场景给出更鲁棒的引导。

参考：
- StableNormal: https://arxiv.org/abs/2407.18740
- Lotus: https://arxiv.org/abs/2409.18124

## 六、初始化与优化策略

1. **Bootstrap**：先用 SfM 点云只训 base Gaussian，等几何大致稳定；
2. **Environment Gaussian 初始化**：场景 bounding box $\mathbf{B}_{scene}$（取 SfM 点云 99.5% quantile）切成 $N^3=32^3$ sub-grid，每 grid 随机采 $K=5$ 个 Gaussian，总共约 $32^3 \times 5 \approx 163840$ 个；
3. **Joint optimization**：从这步开始 base 和 env 一起优化；
4. **Adaptive control**：沿用 3DGS 的 densification / pruning，加上 3DGS-DR 的 **normal propagation** 和 **color sabotage**；
5. **Densification criterion**：3DGS 用投影到 2D 屏幕的梯度积累做 densify，但 ray tracing 没"投影 2D 中心"概念，所以直接用 3D 空间位置梯度积累，每个积累梯度乘以 intersection depth 的一半——**防止远处区域 under-densification**（远处 Gaussian 屏幕投影小，2D 梯度小，3D 梯度也小，但不 densify 远处就分辨率不够）。

Environment Gaussian 最终约 300k 个，占 base Gaussian 的 15%，70MB 显存。Pruning 时保留 top 630k 个 env Gaussian（按 rendering weight 排序）。

## 七、实验结果分析

### 7.1 主表（Table 1, Ref-Real + NeRF-Casting）

| 类别 | Method | PSNR | SSIM | LPIPS | FPS | Train |
|---|---|---|---|---|---|---|
| Non-real-time | NeRF-Casting | 24.670 | 0.659 | 0.246 | <0.1 | >47h |
| Non-real-time | Zip-NeRF | 23.677 | 0.635 | 0.247 | <0.1 | >47h |
| Real-time | 3DGS | 23.700 | 0.641 | 0.262 | 182 | 0.6h |
| Real-time | 3DGS-DR | 23.522 | 0.640 | 0.274 | 134 | 1.0h |
| Real-time | **EnvGS** | 24.617 | 0.671 | 0.241 | 26 | 2.5h |

EnvGS 在所有 real-time method 中 PSNR 最高，与 SOTA non-real-time NeRF-Casting 仅差 0.053 PSNR，但快 100×。FPS 26 比 3DGS 的 182 慢，是因为每个像素要做一次 ray tracing——这跟 ablation Table 6 的 speed-up 策略相关。

### 7.2 反射区域单独评测（Table 2）

在反射 foreground 区域：EnvGS PSNR 33.295 vs 3DGS-DR 31.814（提升 1.481）；在 near-field 区域：46.392 vs 44.007（提升 2.385）。**这正是 environment Gaussian 相对 environment map 的核心优势所在**——近场反射靠 3D 空间显式位置编码。

### 7.3 速度 ablation（Table 6）

| 策略 | FPS |
|---|---|
| PyTorch 手算 ray-primitive | 1/6157.613 |
| 1×1-tile rasterizer | 1/11902.431 |
| EnvGS (full) | 32.259 |
| 80% weight filtering（只对高 β 追踪）| 44.215 |

PyTorch 和 1×1 rasterizer 单帧小时级，完全不可用。EnvGS 通过 OptiX RT core 达到 32 FPS。进一步 speed-up：只对 $\beta > threshold$ 的像素做 ray tracing（diffuse 区域不 trace），80% weight filtering 时 FPS 提到 44，质量损失 0.187 PSNR，可接受。

### 7.4 关键 ablation（Table 4）

- **w/o joint optimization**：PSNR 从 24.617 掉到 24.034（-0.583），几何质量显著下降——证明 base/env 必须联合优化；
- **w/o mono normal**：24.107，反射重建在歧义表面错位；
- **w/ environment map**：24.145，近场反射丢失，验证 Gaussian env 表示比 environment map 强；
- **w/o LPIPS loss**：24.567（仅降 0.05），perceptual loss 贡献很小；
- **w/o color sabotage / normal propagation**：分别 24.268 / 24.192，3DGS-DR 这两个技巧在 EnvGS 仍有效。

## 八、与相关工作的对比和我的理解

### 8.1 vs GaussianShader / 3DGS-DR

| 维度 | GaussianShader / 3DGS-DR | EnvGS |
|---|---|---|
| Environment 表示 | 1× environment map (cubemap) | 3D Gaussian primitives |
| 近场反射 | 不支持（distant lighting 假设）| 支持 |
| 高频反射 | 受 cubemap 分辨率限制 | Gaussian 可自适应 densify |
| 渲染方式 | rasterization + shading function | ray tracing |
| FPS | GaussianShader 27.9 / 3DGS-DR 133.6 | 26.2 |

3DGS-DR FPS 高很多是因为它只用 environment map 查表，没有二次 ray traversal。EnvGS 用 ray tracing 换来质量，FPS 26 在 RTX 4090 上仍算 real-time。

### 8.2 vs NeRF-Casting

NeRF-Casting 沿 reflection ray march，每个 sample query MLP，能在近场/远场都强，但 <0.1 FPS + 47h 训练。EnvGS 用 explicit Gaussian 替代 implicit MLP，把每像素 30+ MLP query 降到 BVH traversal + Gaussian 评估，FPS 提升 250×，训练时间降到 2.5h。

### 8.3 vs 3D Gaussian Ray Tracing (3D GRT, NVIDIA, SIGGRAPH Asia 2024)

3D GRT 把整个 3DGS 场景用 ray tracing 渲染，主要解决 3DGS 在不规则 mesh / hair 等场景的 aliasing 问题。EnvGS 借鉴它的 BVH + chunk traversal 框架，但**只对 environment Gaussian 走 ray tracing，base Gaussian 仍 rasterize**——这是 hybrid 设计，保留了 3DGS rasterize 的高 FPS 优势。可以认为 EnvGS = 2DGS (raster) + 3D GRT (ray trace) + reflection ray 设计。

参考：https://arxiv.org/abs/2406.11848

### 8.4 vs Ref-GS (concurrent work)

Ref-GS 用 directional factorization 把 2D Gaussian 的 SH 分解成 directional terms，单套 Gaussian 表达 view-dependent。EnvGS 选择用**两套 Gaussian 显式分离** base / reflection，更直接但参数量更大。两种思路对应 reflection 建模的两种 prior：implicit (factorized SH) vs explicit (secondary rays)。

参考：https://arxiv.org/abs/2412.00905

### 8.5 vs 3iGS

3iGS 用 tensorial factorization 表示 illumination field，限制在 bounded scene。EnvGS 在 unbounded scene 也能工作，因为它本质就是再放一套 Gaussian，不依赖 grid bounds。

## 九、Intuition 总结

1. **Reflection 本质是二次光线问题**——把 environment 当作显式 3D 几何，自然支持近场/远场统一；
2. **Gaussian primitives 比 environment map 表达能力强**——Gaussian 可以在空间任意位置 densify，environment map 固定在 cubemap 网格；
3. **Joint optimization 是 reflection accuracy 的关键**——不 joint optimize，reflection 误差无法回传修正几何，几何错则 reflection 错，恶性循环；
4. **Ray tracing + RT core 是把 reflection 搬到 explicit 3D 表示的工程 enabler**——没有 RT core，PyTorch ray tracing 单帧小时级，方法根本不可行；
5. **两套 Gaussian 分离 vs 一套统一**——一套统一会发散，因为同一 Gaussian 既被 camera ray 命中又被 reflection ray 命中，优化目标互相干扰；两套分离让 base 专注几何，env 专注光照，role 清晰。

## 十、局限与未来方向

论文明确指出：**透明 / 折射材料无法处理**——目前只沿反射方向 trace，折射需要额外的 Snell's law 计算 + transmission ray。未来可以引入 BTDF 类型的二次 ray，或者像 NeRF-Casting 那样对 refraction 也做 tracing。

另一潜在方向：dynamic scene。当前 EnvGS 假设静态场景，env Gaussian 是固定 3D 结构；如果场景里有动态光源/反射物体，env Gaussian 需要随时间变形，可能需要 4D Gaussian (参考 4DGS, https://arxiv.org/abs/2310.12196)。

还有 relighting：当前 env Gaussian 与 base Gaussian 是 joint optimized 的耦合表示，如果想做 relighting（替换 env），需要把 base 的"材质属性"（roughness, metallic）显式分离出来，这跟 GS3 (https://arxiv.org/abs/2410.11419) 的思路更接近。

## 十一、可能的扩展联想

- **作为 inverse rendering 的 backbone**：env Gaussian 本质是 scene-aware lighting probe，配合 BRDF 分解可以做 relighting；
- **与 residual SH 互补**：env Gaussian 主要捕捉 high-frequency，SH 保留 low-frequency diffuse，可以加一个 residual term $\mathbf{c} = (1-\beta)\mathbf{c}_{base} + \beta \mathbf{c}_{ref} + \gamma \mathbf{c}_{sh}$；
- **Glossy reflection 的 roughness 控制**：当前 $\mathbf{d}_{ref}$ 是完美镜面方向，glossy 表面应该沿 cone 采样多条 reflection ray 然后 average——可以做 importance sampling；
- **从 env Gaussian 反向提取 environment map**：训练好后把 env Gaussian 渲染到 cubemap 上，导出作为离线 lighting asset；
- **多层 bounce**：当前只一次 bounce，可以做 recursive tracing（限制 depth=2）捕捉 mirror reflecting mirror 的场景，但要权衡 FPS。

整体上 EnvGS 是一个工程与理论结合得很紧的工作：reflection 的物理直觉（secondary ray）+ 3D Gaussian 的显式表达优势 + OptiX RT core 的硬件加速 + 可微 ray tracing 的梯度推导，四者缺一不可。它代表 3DGS 生态从纯 rasterization 走向 hybrid rasterize + ray trace 的趋势，后续 inverse rendering、relighting、dynamic extension 都可以在这套框架上叠加。

参考汇总：
- https://zju3dv.github.io/envgs
- https://arxiv.org/abs/2403.17888 (2DGS)
- https://arxiv.org/abs/2308.14773 (3DGS)
- https://arxiv.org/abs/2311.17977 (GaussianShader)
- https://arxiv.org/abs/2405.12235 (3DGS-DR)
- https://arxiv.org/abs/2111.15443 (Ref-NeRF)
- https://arxiv.org/abs/2405.14871 (NeRF-Casting)
- https://arxiv.org/abs/2406.11848 (3D Gaussian Ray Tracing)
- https://arxiv.org/abs/2306.10073 (Zip-NeRF)
- https://arxiv.org/abs/2111.12077 (Mip-NeRF 360)
- https://arxiv.org/abs/2412.00905 (Ref-GS)
- https://arxiv.org/abs/2408.03753 (3iGS)
- https://arxiv.org/abs/2410.11419 (GS3)
- https://arxiv.org/abs/2310.12196 (4DGS)
- https://arxiv.org/abs/2407.18740 (StableNormal)
- https://arxiv.org/abs/2409.18124 (Lotus)
- https://research.nvidia.com/labs/rt/gaussian-tracing/ (OptiX)
