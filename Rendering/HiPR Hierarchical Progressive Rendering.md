---
source_pdf: HiPR Hierarchical Progressive Rendering.pdf
paper_sha256: 5b1b503a969797c5eb85f0b826672e32f1f9b41b7ff62426a585c2e1a2495989
processed_at: '2026-08-19T11:12:44-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HiPR 用人话讲

## 一句话

你在 3D 场景里改了一个茶壶的颜色，传统 path tracer 把整个 4K 画面从头重算一遍。HiPR 说：**只算看得见茶壶的地方，再算被茶壶光照影响的地方，按人眼多容易注意到的顺序排个队，一层层往外推**。最终所有像素都会算到，所以结果跟普通 path tracer 一样准，但你能立刻看到改动。

---

## 痛点：现在的 renderer 有多傻

你在 Blender / Omniverse / Unreal 里做 look dev。桌上一个玻璃杯，你把 tint 从绿色调到蓝色。按刷新。

整个 830 万像素，全部从头开始 path trace。每个像素重新累积几百个 sample。你干等两秒，看着整张图从雪花噪点慢慢收敛到清晰。

**但茶壶在画面里可能只占 5% 面积**。剩下 95%——背景墙、天花板、远处的树——radiance 几乎跟上一帧一模一样。Renderer 不管，全部重算。这就是浪费。

更聪明的方式：你只重算"看得见茶壶"的像素，然后算"被茶壶光照影响"的像素，再算"被那些像素光照影响"的像素——一层层往外。这就是 HiPR 在做的事。

---

## 核心直觉：light transport 是一张地铁图

把整个画面的 light transport 想成一张地铁线路图。每个 surface 是一个 station，光线是列车。改了茶壶 = 茶壶这个 station 突然换颜色。

"哪些 station 会受影响"？当然首先是直接看见茶壶的 camera ray（直接 visible 的 pixels）；然后是从茶壶反射出去打到的下一个 surface；再下一个 surface；一层层往外。这就是 BFS。

HiPR 做的就是：**从改动的 object 出发，沿 light path 做 BFS，把 BFS 的 frontier 投影回 framebuffer，按 priority 排队渲染**。

---

## 具体例子：你调玻璃杯材质

你在 look dev 里把玻璃杯 tint 从绿色调到蓝色。HiPR 怎么反应？

### Stage 1: Visibility Pass（几毫秒）

Camera 每像素打一根 primary ray，存一个 mini G-buffer：`{object_id, depth}`。看看哪些 tile 直接看见这个杯子。把这些 tile 标进 active set $A$。

这一步极快，因为只打一根 ray per pixel，不算 radiance，不算 BSDF，不算 shadow ray。一行 inline ray query 在 compute shader 里搞定。

### Stage 2: Render + Discover（同时干两件事）

**(a) 把 $A$ 里的 tile 先 path trace 出来** —— 你立刻看到杯子颜色变了。

**(b) 在 path tracing 过程中，每条 path 走到第 $k$ 个 bounce vertex $\mathbf{x}_k$ 时，顺手把这个 vertex 的世界坐标投影回 framebuffer**，看看它落在哪个 tile。如果通过 depth test（没被别的几何挡住），就 atomic-add 一个 weight 到那个 tile。

这一步是 HiPR 最聪明的地方：**它免费复用了 path tracer 本来就要走完的 path**。没有额外的 ray，没有额外的 BSDF 采样。只是每条 path 走到 secondary vertex 时多做一次矩阵乘法（投影回 screen）+ 一次 atomic add。

### Stage 3: Sort + Render（按 priority 队列）

把 Stage 2 discover 出来的所有 tile 按 weight 降序排，从高到低渲染。

剩下没被 discover 的 tile，**保留上一帧画面**。最后再补渲染，保证最终 unbiased。

整个过程像水波纹：改了杯子 → 立刻看到杯子本身 → 看到杯子周围桌面反射变了 → 看到墙上 caustic 焦散变了 → 最后才看到远处天花板受杯子间接光照的细微变化。

---

## Weight 公式直觉

$$\Delta W(\mathbf{x}_k) = \beta_k \cdot w_{\text{lobe}}(\ell_k) \cdot w_{\text{path}}(c_k) \cdot \frac{1}{1+k}$$

四项相乘，每一项问一个问题：

| 项 | 问的问题 | 直觉 |
|---|---|---|
| $\beta_k$ | "这条 path 走到这还剩多少能量？" | path throughput，path tracer 本来就要算，免费复用 |
| $w_{\text{lobe}}(\ell_k)$ | "这个 vertex 是 specular 还是 diffuse？" | specular 改了眼睛立刻看见，diffuse 改了眼睛要找半天 |
| $w_{\text{path}}(c_k)$ | "这条 path 是 direct / indirect / caustic？" | direct 优先，caustic 靠后（虽然好看，但通常不是 look dev 优先关注） |
| $\frac{1}{1+k}$ | "这是第几个 bounce？" | 越深的 bounce，priority 越低。$\beta_k$ 在 perfect specular chain 上不衰减，所以需要显式 depth penalty 压住 |

为什么 $\beta_k$ 不够，还要 $1/(1+k)$？因为 mirror chain 上 $\beta_k \approx 1$ 不衰减，第 10 个 bounce 的 vertex 权重还很大，但你眼睛根本看不出第 10 bounce 的改动。$1/(1+k)$ 是 perceptual reality check。

---

## 为什么这不是 cheating

有人会问：你只重算一部分像素，剩下用旧的，那不就 biased 了吗？

关键论证：**HiPR 不改 sample 怎么采，不改 BSDF pdf，不改 estimator 形式，只改"先算谁后算谁"**。所以每个 pixel 的 estimator $\hat{L}(p) = \frac{1}{N}\sum_i L_i(p)$ 仍然是 i.i.d. unbiased。最终 converged image 跟 vanilla path tracer bit-exact 一样。

它只在 transient 帧上有"混合"：一部分 pixel 是新 scene 的新 estimate，另一部分是上一帧的旧 estimate。但**最终**所有 pixel 都会用新 scene 重渲染，bias 消失。

类比：你去餐厅吃饭，菜是一道一道上的，不是 10 道菜一起煮好才端上来。最后你都吃到了，但体验好得多。

这点跟 ReSTIR 不同——ReSTIR 是 sample-level reuse，MIS weight 设计不对会有 persistent bias。HiPR 没这个问题，因为它复用的是 *调度顺序*，不是 *sample*。

---

## 跟别的招数有啥不一样

| 方法 | 干什么 | scene edit 后能用吗 |
|---|---|---|
| Adaptive sampling (Heitz 2021) | per-pixel variance 决定 sample 数 | 不能，它假设整张图都在渲染 |
| ReSTIR / ReSTIR GI | 复用上一帧 sample | scene 一改历史 sample 大量失效 |
| SVGF / OIDN | denoiser 滤噪 | 跟 HiPR 正交，可以一起用 |
| Streaming path tracing (Yuksel 2022) | streaming amortize | 部分适用 |
| Ulschmid 2025 | editing-aware priority for raster | HiPR 是它在 path tracing 的推广 |

HiPR 的独特位置：**第一个把 path tracing 的 scheduling 和 sampling 解耦的方法**。sampling 决定 accuracy，scheduling 决定 perceived responsiveness。

---

## 工程细节里我觉得真正聪明的地方

### 1. Inline ray query 而非 Vulkan RT pipeline

Standard Vulkan RT pipeline 走 raygen → traceKHR → SBT lookup → closest-hit shader，shader stage 来回切换有 overhead。HiPR 用 inline ray query，**所有逻辑在一个 compute shader megakernel 里同步执行**，no SBT、no stage switching。GPU SM 占用率高、warp divergence 低。

这跟 megakernel path tracer 的哲学一致：path 本来就是 data-parallel，不需要 shader-parallel。

参考：https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/rayQueryEXT.html

### 2. OpenPBR BSDF 让 lobe 权重天然对齐

OpenPBR（https://github.com/AcademySoftwareFoundation/OpenPBR）是 Adobe 主导的 PBR 标准，lobe 是 first-class concept——diffuse / glossy / specular / transmission 各自独立。HiPR 的 $w_{\text{lobe}}(\ell_k)$ 直接 map 到 OpenPBR 的 lobe 结构，不需要额外分类逻辑。技术栈选得对。

### 3. $\beta_k$ 是免费的

这是我最喜欢的一点。path tracer 本来就要算 throughput $\beta_k = \prod_i \frac{f_i |\cos\theta_i|}{p_i}$。HiPR 顺手把它当 priority proxy，**零额外计算**。相当于"这条路本来就要走，顺便记一笔账"。

---

## 它真正的局限

Paper 自己承认的：

1. **Camera 一动，整个画面都受 parallax 影响，HiPR 退化成"全部重算"**。未来 work 想做 per-object heuristic：远处 background 移动 < 1 pixel 就跳过。

2. **Weight 用 contribution，不是 edit magnitude**。改 1% 颜色 vs 全换 material，priority 一样。应该按 edit 大小加权。

3. **手工调 $w_{\text{lobe}}, w_{\text{path}}$ 太麻烦**。希望未来推导 perceptual salience formula。

我补充的：

4. **跟 denoiser 怎么交互没讲**。如果只有部分 tile 有新 sample，SVGF / ReBLUR 的 temporal history 在未更新 tile 上会"卡住"，可能产生 ghosting。这是个开放问题。

5. **Atomic contention 在 caustic-heavy 场景可能成为 bottleneck**。所有 path 都往 hot tile 写 weight，cache line 竞争。需要 warp-level reduction 缓解。

6. **跟 ReSTIR 的组合是机会点**。HiPR 决定 tile 渲染顺序，tile 内用 ReSTIR reuse 历史 sample（即使 scene 改变，unchanged region 的 sample 可被 reuse）。两者正交，理论上可以叠加。Paper 没提。

7. **Heckbert path class 在 mixed-lobe BSDF 上是模糊的**。半磨砂塑料的 vertex 同时采 diffuse 和 glossy lobe，那这个 vertex 算 direct 还是 indirect？Paper 没说，可能取 max 或按 lobe 权重加权。

---

## 一句话 mental model

**Progressive rendering 的 latency 问题，根源不在 sampling，在 schedule。Sampling 决定 accuracy，schedule 决定 perceived responsiveness。把这两者解耦后，你能在保持 unbiased 的同时获得交互感**。HiPR 就是把 light transport 的因果结构（BFS frontier）+ perceptual salience（lobe / path class 权重）打包成一个 priority queue，决定哪些 tile 先渲染。

---

## 参考

- HiPR paper: https://doi.org/10.1145/3799821.3820912
- Cem Yuksel 主页: https://www.cemyuksel.com/
- Ulschmid et al. 2025 "Automated Prioritization": https://link.springer.com/article/10.1007/s42979-025-03863-z
- ReSTIR (Bitterli 2020): https://research.nvidia.com/labs/rtr/applications/RESTIR/ReSTIR.pdf
- SVGF (Schied 2017): https://research.nvidia.com/labs/rtr/publication/schied-2017-svgf
- Streaming ray-traced GI (Yuksel 2022): https://www.cemyuksel.com/research/yuksel22streaming/
- Vulkan ray query spec: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/rayQueryEXT.html
- OpenPBR BSDF: https://github.com/AcademySoftwareFoundation/OpenPBR
- Heckbert 1990 notation: https://www.cs.cmu.edu/~ph/1025_fall/lec/hec1990a.pdf
- NVIDIA Real-Time Denoisers (ReBLUR): https://github.com/NVIDIAGameWorks/RayTracingDenoiser

---

如果你想要我针对某个具体应用场景（比如 virtual production 里 director 实时改灯、或者 game engine 里 destructible wall 触发 HiPR）画一个 ASCII pipeline diagram，或者展开 atomic contention 的 GPU memory hierarchy 分析，告诉我就行。

---

# HiPR: Hierarchical Progressive Rendering 深度解读

Andrej，这篇 paper 来自 University of Utah 的 Cem Yuksel 团队（Cem 在 hair modeling、mesh shader、streaming path tracing 上有大量 foundational 工作），发表于 SIGGRAPH 2026 Real-Time Live!。它把"interactive path tracing 在 scene editing 之后如何给出即时反馈"这个问题，从 **sampling problem** 重新frame 为 **scheduling problem**——这是这篇 paper 最值得 build intuition 的地方。

我分几个层次来讲：核心 idea 的 mental model → 三 stage pipeline 的工程细节 → 公式变量逐项拆解 → unbiased 性的证明式论证 → 与相关工作对比 → implementation tricks → limitations → 我对它真正贡献的批判性思考。

---

## 1. Mental Model：为什么这是个 scheduling problem

### 1.1 现有 progressive renderer 的盲点

传统的 progressive path tracer (OptiX, Mitsuba, pbrt-v4, Cycles, Omniverse RTX) 在用户编辑 scene 之后，要重算整个 framebuffer。常见的 latency amortization 手段：

- **Adaptive sampling** (Heitz et al. 2021 "Adjoint-driven Russian Roulette", Drossel et al., Appleseed's AAF)：基于 pixel-level variance/error 估计分配 samples，但 error metric 是 per-pixel 局部量，没法回答"这个 pixel 是否被 scene edit 触及"。
- **Spatiotemporal reuse** (ReSTIR [Bitterli 2020], ReSTIR GI [Ouyang 2021], SVGF [Schied 2017])：复用上一帧 / 邻域 samples 来降 variance，但 reuse 跨 frame 的前提是 scene 静态；一旦 scene mutation 发生，rejection rules 会丢弃大量历史 samples，等于失效。
- **Temporal accumulation (TAA-style)**：在编辑时会清空 accumulation buffer，等于从零开始。

这三类方法的共性：**它们 amortize sample cost across frames，但没有 amortize "scene edit 的影响范围"**。也就是说，它们无法识别 "哪些 pixel 揭示 scene change"。

### 1.2 HiPR 的 reframe

HiPR 的核心洞察：scene edit 在 light transport 上有一个**因果传播结构**——改动一个 object → 它直接 visible 的 pixels 受影响 → 这些 pixels 出射的 path 在第二、第三 bounce 打到的 surface 也受影响 → ... 这是一个 BFS-like 的扩散。把这个扩散结构显式建立出来，就能按"perceptual impact" 排序，让用户最先看见改动最显著的地方。

这是个 scheduling 问题，因为：
- per-sample estimator 仍然是 unbiased i.i.d. 路径采样的均值（HiPR 不动 sampler）
- 唯一改变的是 **tile 的渲染顺序**
- 同一 pixel 无论何时被渲染，期望都等于 ground truth

所以 scheduling 不会引入 bias，只会改变 *何时* 收敛。这是个关键安全性论证，后面 §4 展开。

参考：Ulschmid et al. 2025 "Automated Prioritization for Context-Aware Re-rendering" 是 HiPR 引用的相关工作，做的是 editing-driven priority；HiPR 把它扩展到 path-traced 全局光照的层级结构。链接：https://link.springer.com/article/10.1007/s42979-025-03863-z

---

## 2. 三 Stage Pipeline 详解

### Stage 1: Initial Visibility Pass

**输入**：scene change set $E$ = transform / geometry / material / emission 改变的元素集合。
**操作**：每 pixel center 打 1 根 primary ray，存一个 mini G-buffer：`{object_id, camera-space_depth}`。
**输出**：active tile set $A$ = $E$ 中元素直接 visible 的 tiles。

为什么是 single primary ray per pixel 而非 multi-sample？因为这一 stage 只为**发现** direct visibility，不需要 radiance。一行 compute shader inline ray query 即可，几毫秒完成。G-buffer 后续 stage 2 还要复用作 depth test 的 reference。

Intuition：这一 stage 等价于"改了哪个 object，把它的 silhouette 标出来"。

### Stage 2: Render Directly Visible Tiles + Discover Secondary Bounces

这是最有意思的 stage。它同时做两件事：

**(a) Path trace $A$ 中的 tiles** —— 给用户 immediate feedback。
**(b) 在每条 path 的每个 secondary vertex $x_k$ 上，把它 reproject 到 framebuffer，找到对应 tile，如果通过 depth test 就 atomic-increment tile weight** —— 建立 light transport hierarchy。

注意"secondary"这里指 bounce count $k \geq 1$，即从 camera 出发第 $k$ 次散射后的 vertex。Primary hit ($k=0$) 已经在 Stage 1 处理过。

Depth test 的作用：reproject 后的 $[u, v]$ 落在 tile $T$，但 $x_k$ 是否真的对 $T$ 中那个 pixel 有贡献？要看 $x_k$ 的 camera-space depth $z$ 与 G-buffer 中存储的 $z_{\text{stored}}$ 是否在 tolerance $\delta$ 之内。否则 $x_k$ 被 scene 中其他 geometry 遮挡，不 visible，不该标 tile。

> 一个细节没在 paper 明说但很重要：depth test 失败的 vertex **丢弃**，意味着 HiPR 只追踪 *visible secondary bounces*。这避免了 diffuse surface 后面看不见的 caustic 给 tile 加权——但同时也漏掉了 "未来如果 camera 转过去才显现" 的 transport。这是 design choice，未来 work §6 提到 "delta-aware" 时可能补这块。

### Stage 3: Sort and Render Tiles

- 把 Stage 2 *discovered* 的所有 tiles 用 parallel sort 按 $W(T)$ 降序排。
- Descending order 渲染。
- Undiscovered tiles 保留前一帧 radiance，直到 discovered tiles 渲完，再补渲染 undiscovered —— 保证最终 unbiased。

并行排序在 GPU 上一般用 merge-bitonic radix sort，$O(\tau^2 \log \tau)$ per tile grid，开销可忽略。reference implementation 没说具体 radix；Cem 之前在 *Streaming ray-traced global illumination* 里用过的 prefix-sum partition 也可行。

---

## 3. 公式逐项拆解

### 公式 (1): 投影

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} O_x + f \frac{x}{z} \\ O_y + f \frac{y}{z} \end{bmatrix}, \quad \text{with} \quad \begin{bmatrix} x \\ y \\ z \end{bmatrix} = M_{\text{cam}}^{-1} \mathbf{x}_k
$$

变量：
- $\mathbf{x}_k \in \mathbb{R}^3$：light path 上第 $k$ 次 bounce 的 vertex，**world space** 坐标。下标 $k$ 是 bounce count（0 = primary hit, 1 = first secondary, ...）。
- $M_{\text{cam}}^{-1}$：camera 的 **inverse** world transform，即 world → camera。注意命名：通常 $M_{\text{cam}}$ 是 camera → world，inverse 后才是 world → camera。
- $[x, y, z]^T$：$\mathbf{x}_k$ 在 camera space 的坐标。
- $O_x, O_y$：camera **principal point**（光心在 image plane 上的投影坐标，单位 pixel）。在理想对称相机里 $O_x = W/2, O_y = H/2$。
- $f$：focal length（单位 pixel）。假设 square pixels，所以不分 $f_x, f_y$。
- $d(\mathbf{x}_k) = z$：定义为 $\mathbf{x}_k$ 的 camera-space depth。**注意这是 perspective 的 $z$，不是 $1/z$ 的 depth buffer 形式**——后者才是 rasterization hardware depth。HiPR 这里直用线性 $z$，因为比较时只看是否在 $\delta$ 内，单调即可。

Intuition：这是 pinhole camera 标准透视除法 $u = O_x + f \cdot x/z$。把 path 上的世界点投影到屏幕，找它该贡献给哪个 tile。

为什么不用 rasterization hardware 光栅化做这件事？因为 path tracer 产生的 secondary vertex 不在 triangle 流里，是从 BSDF importance sampling 来的随机点，必须 CPU-side 或 compute shader-side 手动投影。Inline ray query 正好提供 hit 信息，省一次 dispatch。

### 公式 (2): Tile weight increment

$$
\Delta W(\mathbf{x}_k) = \beta_k \cdot w_{\text{lobe}}(\ell_k) \cdot w_{\text{path}}(c_k) \cdot \frac{1}{1+k}
$$

四项相乘，每项代表不同维度的"重要性"。

**$\beta_k$ — path throughput（自然项）**

按 path tracing 定义，$\beta_k = \prod_{i=0}^{k-1} \frac{f_i(\omega_i^{\text{in}}, \omega_i^{\text{out}}) |\cos\theta_i|}{p_i(\omega_i)}$，即前 $k$ 次 scatter 累积的 BSDF × cosine / pdf。$\beta_k$ 越大，从 camera 到 $\mathbf{x}_k$ 这条 prefix 越亮，则 $\mathbf{x}_k$ 的变化对最终 radiance 的影响也越大。这一项是 path tracer 本来就计算的，**零开销复用**。

**$w_{\text{lobe}}(\ell_k)$ — lobe priority（perceptual 用户调参项）**

$\ell_k$ 是 vertex $k$ 处实际采样的 BSDF lobe 类别：diffuse / glossy / specular / transmission。用户给每类一个权重，要求 $\sum_\ell w_{\text{lobe}}(\ell) = 1$。

直觉：specular 镜面反射改一下材质颜色，眼睛立刻看见；diffuse 的小改不易察觉。所以默认 order：specular > direct diffuse > indirect > caustic。

**$w_{\text{path}}(c_k)$ — path-class priority（perceptual 用户调参项）**

$c_k$ 是 path prefix 的 **Heckbert 类别**，$k$ 时已知（不需要继续 path）。Heckbert notation：

| Notation | 含义 |
|----------|------|
| L | Light |
| D | Diffuse bounce |
| S | Specular bounce |
| E | Eye (camera) |

HiPR 把 path class 分成：

- **Direct**: $LDE$（光→diffuse→eye）或 $LSE$（光→specular→eye）。一次 bounce 直接到光源。
- **Indirect**: $LD D^+ E$（diffuse→diffuse→...→eye）。多次 diffuse bounce。
- **Caustic**: $LS^+ DE$（光→specular→...→diffuse→eye）。specular 聚焦在 diffuse surface 上的焦散。

要求 $\sum_c w_{\text{path}}(c) = 1$。

Cast-shadow 的特殊处理：当 next-event estimation 的 shadow ray 被 changed element $\varepsilon \in E$ 阻挡，那个 shadowed pixel 所在 tile 接受 **occlusion priority**。这个 case 没在 vertex 上体现，因为 cast shadow 不在 path vertex 之间——shadow ray 是 connection ray，不产 vertex。

**$\frac{1}{1+k}$ — explicit depth penalty**

为什么已经有 $\beta_k$ 的 natural falloff 还要再加 $1/(1+k)$？因为 $\beta_k$ 在 specular chain 上可能不衰减（perfect mirror throughput $\approx 1$），那 $k$ 很大时权重还很大——但 deep bounce 的改动 perceptually 难看见。$1/(1+k)$ 显式压低 deep bounce 优先级，确保 deep caustic / complex specular chain 不会把 priority queue 顶在前面。

### 公式 (3): Atomic summation

$$
W(T) = \sum_{\mathbf{x}_k \in T} \Delta W(\mathbf{x}_k)
$$

每个 projected vertex 通过 depth test，就 atomic-add $\Delta W$ 到对应 tile。GPU 上用 `InterlockedAdd` (HLSL) / `atomicAdd` (GLSL) / `atomicAdd` (CUDA)。这意味着多条 path、多个线程同时发现同一 tile 也没事，权重累加。

原子操作的开销主要在 cache line contention。Tile 数量 $\approx WH/\tau^2$，远小于 pixel 数，但 contention 会集中在 hot tile（比如直接被 caustic 聚焦的 tile）。Cem 团队以前在 GPU hair rendering 上对 atomic contention 有过经验，可能用了 shared-memory staging buffer 减缓——paper 没说。

---

## 4. 为什么 unbiased：调度安全性论证

这是 paper §2.3 末段的关键一段。我把它拆成三个命题：

**命题 1**: 对每个 pixel $p$，最终 radiance estimator $\hat{L}(p) = \frac{1}{N}\sum_{i=1}^N L_i(p)$，其中 $L_i(p)$ 是 i.i.d. path sample 期望等于 ground truth $L(p)$。

**命题 2**: HiPR 只改变 *何时* 给 pixel $p$ 的 sample $L_i(p)$ 分配 compute 资源；不改 sample 分布、不改 BSDF sampling pdf、不改 estimator 形式。

**命题 3**: 渲染到无穷多 samples 时，每个 pixel $p$ 的 $\hat{L}(p) \to L(p)$ a.s.（强收敛），与调度顺序无关。

**结论**: HiPR 是 asymptotically unbiased。

唯一 subtlety 是：未渲染的 tiles 在某一时刻显示 *前一帧* radiance，那是上一帧的 unbiased estimate，所以中间帧的 estimate 是某个 subset of pixels 的新 estimate + 剩余 pixels 的旧 estimate 的混合。这混合在 transient 帧上是 biased（因为旧 estimate 对应旧 scene），但**最终**全部 tiles 都用新 scene 重渲染后，bias 消失。

这跟 ReSTIR 不同——ReSTIR 的 MIS weight 设计如果不对会引入 persistent bias。HiPR 没这个问题，因为它复用的是 *frame-level* 调度，不是 *sample-level* reuse。

---

## 5. 与相关工作的位置图

| 方法 | 性质 | 应对 scene edit? |
|------|------|------------------|
| Adaptive sampling (Heitz 2021, Veach-style) | per-pixel error estimator | 否 |
| ReSTIR / ReSTIR GI | spatiotemporal sample reuse | 部分适用，但 mutation 大时失效 |
| SVGF / ASVGF | spatiotemporal filtering | 否，filtering 不是 scheduling |
| Streaming path tracing (Yuksel 2022) | amortize cost via streaming | 部分适用 |
| Ulschmid et al. 2025 | editing-aware priority for raster | 是，HiPR 是它在 path tracing 的推广 |
| Foveated rendering (VR/AR) | gaze-driven sample density | 类似精神，但 per-frame 不是 edit-driven |
| Error-Driven Importance Sampling (Zimmer et al.) | variance-aware | 否 |
| Quasi-Monte Carlo path tracing (Keller) | low-discrepancy sequences | 否 |

HiPR 在这个图里的 unique 坐标：**第一个显式用 light transport 因果结构 + perceptual salience 调度的 path tracing 系统**。

链接：
- ReSTIR: https://research.nvidia.com/labs/rtr/applications/RESTIR/ReSTIR.pdf
- SVGF: https://research.nvidia.com/labs/rtr/publication/schied-2017-svgf
- Streaming path tracing (Yuksel 2022): https://www.cemyuksel.com/research/yuksel22streaming/
- Adaptive sampling Heitz 2021: https://eheitzresearch.wordpress.com/

---

## 6. Reference Implementation 的工程细节

### Inline Ray Tracing 而非 Vulkan RT Pipeline

Standard Vulkan ray tracing pipeline：
```
raygen shader → traceKHR → (hit/miss shader) → closest-hit
                ↕ (shader binding table, SBT)
```
SBT 把 BLAS (bottom-level AS) 与 shader stages 绑定，意味着每 geometry type 一组 shader record。dispatch 模型需要 chit/ahit/miss 三个 shader stage，overhead 主要在 GPU scheduler 的 shader-stage switching。

Inline ray query (`rayQueryEXT` 或 `rayQueryInitializeKHR`)：
```
compute shader { 
    rayQuery q;
    q.initialize(...);
    q.proceed();
    if (q.getIntersectionType() == CANDIDATE_HIT) ...
}
```
所有逻辑在 **单个 compute shader 内同步执行**，no shader stage switching、no SBT lookup。代价：失去 ray-tracing pipeline 的 traversal shader 自定义（例如 alpha-tested geometry 自定义 any-hit）；但 HiPR 用 OpenPBR 的 BSDF，alpha 通过 mask texture 在 compute shader 里手动处理。

为什么 megakernel path tracer 适合？因为 path 本身就是 *data-parallel* 而非 *shader-parallel*，每条 path 独立走它的 bounce 序列，不存在 "geometry A 走 shader X、geometry B 走 shader Y" 的并行调度需求。Inline ray query 让 GPU SM 占用率高、warp divergence 低（路径内的 divergence 不算）。

### Object ID → Material Binding

Material 参数按 GPU alignment requirement (16B/32B packing) 打包进 SSBO/UBO，按 object_id 索引。一行 `Material m = materials[object_id]` 即可解析 shading state。这种 binding 是 *manual* 的，省去 SBT 的 indirection。

### OpenPBR BSDF

OpenPBR 是 Adobe 主导的 PBR shading model 标准（https://github.com/AcademySoftwareFoundation/OpenPBR），physically grounded，industry-standard，是 MaterialX 的 shading model。HiPR 选它意味着 paper 的 perceptual lobe weights（diffuse/glossy/specular/transmission）能直接 map 到 OpenPBR lobe 结构——lobes 是 OpenPBR 的 first-class concept。

---

## 7. Limitations & Future Work（paper 自承 + 我的补充）

Paper 自己列出：
1. **Camera motion 全 reproject 问题**：现在 camera 一动，所有 object 都因 parallax 被 trigger。但其实远距离 object 移动 < 1 pixel，perceptually irrelevant。需要 per-object "this camera motion 是否 warrant fresh HiPR pass" 的 heuristic。
2. **Delta-aware weighting**：目前 weight 用的是 changed object's **contribution**，不是 edit 的 **magnitude**。改 5% 颜色 vs 全换 material 应该有不同 priority。
3. **Simulation-guided sampling**：利用 motion vectors / simulation data guide sample density。
4. **Perceptual salience formula**：手工调 $w_{\text{lobe}}, w_{\text{path}}$ 太麻烦，希望自动推导。

我补充几个：

5. **Denoiser 交互**：paper 没提 HiPR 与 spatiotemporal denoiser (SVGF, OIDN, ReBLUR) 的交互。如果只有部分 tiles 有新 samples，denoiser 的 temporal history 在未更新 tiles 上会"卡住"，可能产生 temporal lag 或 ghosting。ReBLUR 的 confidence-driven history rejection 应该能适配 HiPR。

6. **Tile 大小 trade-off 的 quantitative sweet spot**：paper 没给 $\tau$ 与 $\delta$ 的 ablation。我猜 $\tau = 16$ 或 32 是典型 GPU dispatch 友好且 depth test 精度可接受的甜蜜点。

7. **Atomic contention profiling**：hot tile 的 atomic contention 在 caustic-heavy 场景可能成为 bottleneck。可能需要 hierarchical atomic（per-warp sum → tile-level atomic）。

8. **Adaptive tile 拓扑**：固定 $\tau \times \tau$ tile 是均匀的；但 perceptually important 区域用小 tile，unimportant 区域用大 tile 可能更优。Hierarchical tile (coarse-to-fine) 与 HiPR 的 hierarchy 是正交维度。

9. **Spectral / polarized light transport**：HiPR 假设 RGB 三通道。在 spectral rendering (Brigade, Mitsuba 3 spectral) 中，path class 的 perceptual rank 应该被 spectral response 重新加权。

---

## 8. 我对真正贡献的批判性思考

### 8.1 它"重新定义问题"是否真的成立？

Cem 在 §5 写 "HiPR reframes progressive rendering as a scheduling problem"。这是 marketing-strong，但 technically 准确吗？

- ✅ 它确实把"先渲染谁"这个决策从 sampling domain 提到 scheduling domain。
- ✅ 它显式建模 light-path dependencies 形成层级——这是新东西。
- ⚠️ 它的 *priority weight* $\Delta W$ 仍然混合了 importance sampling 的 throughput $\beta_k$（sampling 内部量）和 perceptual weight（外部量）。如果 *纯* scheduling，可能应该完全用 perceptual-only metric（比如 pixel-level difference predictor），不该用 $\beta_k$。

所以更精确的描述：HiPR 是一个 **hybrid scheduling**，既包含 transport causality 也包含 importance。它的优雅在于 $\beta_k$ 是 free 的（path tracer 已经算），相当于免费的 proxy for "this vertex 的视觉显著性"。

### 8.2 它对 GPU 的实际友好度

- Stage 1: G-buffer 一次 raster path，极快。
- Stage 2: 普通 path trace $A$ + 顺便 atomic update tile weights。atomic 在中等 contention 下 OK，高 contention 下是 perf cliff。
- Stage 3: parallel sort + dispatch remaining tiles。GPU 友好。

真正的 bottleneck 我猜是 **Stage 2 末尾的 tile weight 集中阶段**——所有 threads 都在写一个有限 tile grid 的 atomic counters。如果 $|A|$ 大（很多 tiles 直接 visible），同时 secondary bounce 也多，atomic add 可能 stall。一种缓解：用 warp-level reduction 先把同 warp 内对同一 tile 的 $\Delta W$ 合并，再 atomic 一次。Paper 没提，但 NVIDIA's sort/scan primitives 都用这个 trick。

### 8.3 与 ReSTIR 的真正区别

ReSTIR 是 sample-level reuse，HiPR 是 tile-level scheduling。两者正交，理论上可以叠加：用 HiPR 决定 tile 渲染顺序，tile 内用 ReSTIR reuse 历史 samples（即使 scene 改变，部分 unchanged regions 的 samples 可被 reuse）。这会进一步降 cost。Paper 没提组合，是机会点。

### 8.4 为什么 paper 不需要重测 quality metrics

因为 HiPR 不改 final image——它只改 *transient* frames 的内容。最终 converged image 与 vanilla path tracer 完全 bit-exact。所以质量评估只有 *perceptual latency* 这一个轴。paper 给了 Figure 1 但没给 user study；未来 work 可以做 "用户感受到 change 的时间" 的 A/B test。

### 8.5 关于 path class 分类的边界 case

Heckbert notation 的 LDD+ vs LS+D 在 specular-diffuse 混合 BSDF（比如半磨砂塑料）的 vertex 上是模糊的——vertex 既采了 diffuse lobe 又采了 glossy lobe。OpenPBR 是 lobes-mixed BSDF，single vertex 可能同时贡献到 direct 和 indirect class。Paper 没说这种情况怎么计算 $w_{\text{path}}(c_k)$，可能取 max 或按 lobe 权重加权。这是 implementation detail 但 production 中重要。

---

## 9. 我会怎么 build intuition 的一句话总结

HiPR 的精髓在于：**light transport 在 path space 上是 DAG（有向无环图），scene edit 是这个 DAG 上的 perturbation source，HiPR 用 $\beta_k \cdot \text{perceptual weights} \cdot \text{depth penalty}$ 作为边的权重，做 BFS-style 的 priority queue propagation，用 reprojection + depth test 把 path-space DAG 折叠成 framebuffer-space DAG，从而把"哪个 tile 先渲染"变成一个 well-defined optimization 问题——而 sampler 完全不动，所以 final image 仍然 unbiased**。

这个 reframe 的核心直觉：**progressive rendering 的 latency 问题不是 sampling 的问题，是 schedule 的问题；sampling 决定 accuracy，schedule 决定 perceived responsiveness；两者解耦后，你能在保持 unbiased 的同时获得交互感**。

---

## 参考链接汇总

- Paper 主页 (SIGGRAPH RT Live '26): https://doi.org/10.1145/3799821.3820912
- Cem Yuksel 主页: https://www.cemyuksel.com/
- Ulschmid et al. 2025 "Automated Prioritization": https://link.springer.com/article/10.1007/s42979-025-03863-z
- ReSTIR (Bitterli 2020): https://research.nvidia.com/labs/rtr/applications/RESTIR/ReSTIR.pdf
- ReSTIR GI (Ouyang 2021): https://research.nvidia.com/labs/rtr/publication/ouyang2021restir-gi/
- SVGF (Schied 2017): https://research.nvidia.com/labs/rtr/publication/schied-2017-svgf
- Heitz 2021 adjoint-driven RR: https://eheitzresearch.wordpress.com/
- Streaming ray-traced GI (Yuksel): https://www.cemyuksel.com/research/yuksel22streaming/
- Vulkan ray query spec: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/rayQueryEXT.html
- OpenPBR BSDF: https://github.com/AcademySoftwareFoundation/OpenPBR
- Heckbert 1990 notation: https://www.cs.cmu.edu/~ph/1025_fall/lec/hec1990a.pdf
- NVIDIA Real-Time Denoisers (ReBLUR): https://github.com/NVIDIAGameWorks/RayTracingDenoiser

如果你希望我展开任何 specific 公式推导（比如把 $\beta_k$ 从 BSDF sampling pdf 形式显式展开）、画 pipeline 的 ASCII 架构图、或者对比具体 alternative scheduling policies（比如 pure-error-driven 或 foveated-only），告诉我，我可以再深入。
