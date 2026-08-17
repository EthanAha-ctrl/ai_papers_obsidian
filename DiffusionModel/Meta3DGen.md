---
source_pdf: Meta3DGen.pdf
paper_sha256: f6259469ecc7a840a2b3a170dc1ba904a2dd37e58b1040ca8b8d7167085e2e5a
processed_at: '2026-08-05T18:00:51-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hi Andrej, 咱们用最直白的话把 Meta 3D Gen 这篇 paper 拆解一下，帮你 build 起底层的 intuition。

这篇 paper 的核心可以总结为一句话：**借 2D 图片生成的巨大力，绕道去生成 3D 模型，并且通过极其精巧的工程拼接，解决了 3D 生成中普遍存在的“丑”和“慢”的问题。**

---

### 1. 核心痛点：为什么 3D 生成这么难搞？

3D 生成面临的根本困境是**数据匮乏**。Image generator 吸收了百亿级的图片数据，学会了什么是“真实”；3D 模型只见过几十万个 3D assets，缺乏全局常识。因此，直接从头训练一个 text-to-3D diffusion model 往往生成出非常粗糙的、缺乏细节的模型。

老的方法（如 DreamFusion 用的 SDS, Score Distillation Sampling）怎么做呢？随机生成一个 3D blob，渲染成 2D 图片，丢给 2D image generator 打分，再根据分数优化 3D blob。这方法极度耗时，每个 asset 需要几十分钟到一小时，并且容易产生 "Janus effect"（多脸怪物，因为 AI 从每个角度渲染都觉得自己看到的是正面）。Meta 3D Gen 完全抛弃了这种 per-asset optimization 的慢套路。

### 2. Three-Space Pipeline：三种空间的接力赛

为了解决数据少、分辨率低、渲染慢的问题，3DGen 把生成过程放在三个完全不同的 representation space 里来回倒腾。这是这篇 paper 最精妙的 architecture 设计。

#### Space 1: View Space (2D 图像空间)
**优势：** 细节极丰富，可以白嫖 Emu image generator 的海量先验知识。
**劣势：** 多视图之间几何不一致，画 4 个角度可能画成 4 个物种。

#### Space 2: Volumetric Space (3D 体积空间，如 SDF)
**优势：** 保证 3D 几何拓扑结构绝对正确，没有 Janus effect。
**劣势：** 数据分辨率受限，surface 和 texture 通常很糊。

#### Space 3: UV Space (2D 纹理空间)
**优势：** 可以直接画超高清 4K 纹理，细节拉满。
**劣势：** 严重依赖 UV unwrap 质量，在展开图的拼接缝隙处极易出现 seam 裂痕。

### 3. Pipeline 全流程直觉推演

3DGen 把这三个空间串成了一个 two-stage 的 feed-forward pipeline，全流程不到 1 分钟。

#### Stage I: AssetGen — 搞定大形和底色 (约 30 sec)

1.  **Text → Multi-View Images (View Space)**
    给定 text prompt $y$，多视图 diffusion 模型 $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$ 直接生成 $K$ 个视角的图片 $I_1, \ldots, I_K$。
    关键 trick 在于：生成的图片通道数 $C$ 不仅仅是 RGB。为了支持 PBR (Physically-Based Rendering)，模型同时输出 shaded appearance (带光照的渲染图) 和 albedo (无光照的底色图)。强制网络解耦光照信息，是后续材质推断成功的关键。
    $$y \xrightarrow{\Phi_{\mathrm{mv}}^{\mathrm{obj}}} \{I_1, \ldots, I_K\} \sim p(I_1, \ldots, I_K \mid y)$$
    *变量解释：* $y$ 是文本输入，$\Phi_{\mathrm{mv}}^{\mathrm{obj}}$ 是生成多视角的 diffusion network，$I_k \in \mathbb{R}^{H \times W \times C}$ 是第 $k$ 个视角的图像，$K$ 是视角数量。

2.  **Multi-View Images → 3D Mesh (Volumetric Space)**
    拿到 $K$ 张图后，重建网络 $\Phi_{\mathrm{rec}}^{\mathrm{obj}}$ 做 forward pass，直接输出 3D mesh $M = (V, F, U)$ 和初始纹理 $T$。
    这里用了 **SDF (Signed Distance Field)** 表示 3D shape。比起 occupancy field，SDF 提供更平滑的表面梯度，提取出来的 mesh 拓扑更干净。
    $$\{I_1, \ldots, I_K\} \xrightarrow{\Phi_{\mathrm{rec}}^{\mathrm{obj}}} (M, T)$$
    *变量解释：* $M = (V, F, U)$ 其中 $V \in \mathbb{R}^{|V| \times 3}$ 是顶点坐标集合 ($|V|$ 是顶点数，3 代表 $x,y,z$)，$F \in \{1, \ldots, |V|\}^{|F| \times 3}$ 是三角面片索引集合，$U \in [0,1]^{|V| \times 2}$ 是每个顶点对应的 UV 展开坐标。

3.  **Texture Reprojection & Fusion (UV Space)**
    把生成的视图 $I_k$ 投影回 mesh 的 UV 空间，得到部分纹理 $T_1, \ldots, T_K$。网络 $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 把这些碎片拼起来，填补看不见的区域，输出初始的完整纹理 $T^*$。

#### Stage II: TextureGen — 极致的高清纹理精修 (约 20 sec)

Stage I 给了正确的几何，但 texture 比较糊。Stage II 专门负责把纹理做到 4K 级别的生产可用质量。

1.  **Geometry-Conditioned Multi-View Generation (View Space)**
    重点来了，这里的 multi-view generator $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$ 把 Stage I 生成的 mesh $M$ 作为条件。因为几何已经固定，这就把 “凭空想象 3D 视图” 降级成了 “给 3D 模型上色画图”。条件越多，任务越简单，生成的 2D 图分辨率和一致性极高。
    $$y, M \xrightarrow{\Phi_{\mathrm{mv}}^{\mathrm{tex}}} \{I_1, \ldots, I_K\} \sim p(I_1, \ldots, I_K \mid y, M)$$

2.  **Reprojection & UV Fusion (UV Space)**
    把新的高清视图再投影到 UV 空间，通过 $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ 网络融合成一张完整的 4K 纹理图 $T$。

3.  **The "Aha" Engineering Trick: Hybrid UV Fusion**
    这里碰到了一个致命的 **Distribution Shift** 问题。
    TextureGen 里的 $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ 是用 3D 艺术家手工制作的、极其规整的 UV 展开图训练的。
    而 Stage I (AssetGen) 自动生成的 mesh，其 UV 是用 xatlas 这类算法自动展开的，切线乱七八糟，岛块极多。
    如果直接把自动 UV 喂给 TextureGen 的融合网络，接缝处直接裂开，全是 artifact。

    Meta 的解法非常务实：混着用！
    - 用 TextureGen 的 $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$ 生成高清图（它擅长画图）。
    - 用 TextureGen 的 $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ 做初步融合（它擅长拼图）。
    - **最后再丢回 AssetGen 的 $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 做一次 fix**。为什么？因为 AssetGen 的 $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 就是在这种乱七八糟的自动 UV 上训练出来的，它见怪不怪，专门负责把残余的 seam 补平。
    $$\{T_1, \ldots, T_K\}, T \xrightarrow{\Phi_{\mathrm{uv}}^{\mathrm{obj}}} T^*$$

---

### 4. PBR 是工业界的生死线

生产级别的 3D asset 必须支持重新打光。游戏引擎里的光照随时变化，如果贴图是把光影“烤死”在表面的 RGB 图，稍微换个光源就假得不行。

PBR 需要解耦出材质属性。3DGen 的输出 $T \in \mathbb{R}^{L \times L \times 5}$，这 5 个通道分别是：
- **Albedo** $\rho_d \in \mathbb{R}^3$: 物体本身的底色，不受光照影响。
- **Roughness** $r \in [0, 1]$: 表面微表面的粗糙度，决定高光是锐利的点还是模糊的光晕。
- **Metalness** $m \in \{0, 1\}$: 是否是金属，决定反射率和边缘的 Fresnel 效应。

Rendering equation 依赖于 Cook-Torrance BRDF 微表面模型：
$$f_r(\omega_i, \omega_o) = (1 - m) \cdot \frac{\rho_d}{\pi} + m \cdot \frac{D(\theta_h, r) G(\theta_i, \theta_o) F(\theta_h, \rho_d)}{4 (\cos\theta_i)(\cos\theta_o)}$$
*变量解释：*
- $\omega_i, \omega_o$: 光线入射方向和视线出射方向。
- $m, r, \rho_d$: 即上文提到的 metalness, roughness, albedo。
- $D(\theta_h, r)$: Normal Distribution Function (法线分布函数)，给定半程向量角 $\theta_h$ 和 roughness $r$，算出有多少微表面正好能镜面反射。常用 GGX 模型。
- $G(\theta_i, \theta_o)$: Geometry/Shadowing function，微表面之间互相遮挡的概率。
- $F(\theta_h, \rho_d)$: Fresnel term，根据视角决定反射率。非金属基础反射率约 4%，金属基础反射率就是其 albedo 颜色。

通过让 model 在 Stage I 输出 albedo 和 shaded map，网络被迫学习物理光影逻辑，从而让提取出的 PBR maps 质量远超那些直接猜 RGB 的方案。

---

### 5. 实验数据与 Intuition 验证

来看他们针对 professional 3D artists 的 A/B 测试数据表。这是一个极好的 intuition 验证：

<table>
  <tr>
    <th>Method</th>
    <th>Q0: Fidelity Win%</th>
    <th>Q1: Quality Win%</th>
    <th>Q2: Texture Win%</th>
    <th>Q3: Geometry Win%</th>
  </tr>
  <tr>
    <td>vs Rodin Gen-1</td>
    <td>68.0%</td>
    <td>59.8%</td>
    <td>69.1%</td>
    <td>56.7%</td>
  </tr>
  <tr>
    <td>vs Meshy v3</td>
    <td>60.0%</td>
    <td>65.3%</td>
    <td>53.7%</td>
    <td>66.3%</td>
  </tr>
  <tr>
    <td>vs Third-party T23D</td>
    <td>59.1%</td>
    <td>61.3%</td>
    <td>60.2%</td>
    <td>60.2%</td>
  </tr>
</table>

**Intuition 提取：**

1.  **Geometry 战场:** 对比 Rodin Gen-1 时，Geometry 胜率最低 (56.7%)。因为 Rodin 专门做 quad mesh 拓扑，几何非常规整。对比 Meshy v3 时，Geometry 胜率极高 (66.3%)，因为 Meshy 经常出现 Janus effect 和破损几何。3DGen 凭借 SDF 重建稳居中间。
2.  **Texture 战场:** 对比 Meshy v3 时，Texture 胜率只有 53.7%。为什么？因为 Meshy 的纹理非常锐利、鲜艳，外行看热闹觉得好看。这就引出了下一组对比。
3.  **Professional vs Amateur 偏好:** Paper 里提到，普通用户容易被高对比度、高饱和度的“视觉系”纹理骗过，即使存在严重的 UV seam 和 inpainting artifact 也不在乎。Professional 3D artists 则极其看重几何正确性和纹理的无缝一致性。所以 paper 强调 evaluation 必须用 professional artists，否则数据会被 "fake sharpness" 带偏。
4.  **Prompt 复杂度分析:** 在 Objects 和 Characters 上，各家打得有来有回。但到了 Compositions (角色+物体复杂场景) 这一项，3DGen 胜率飙到 73.9%，把 Third-party T23D (67.9%) 和 Meshy (63.5%) 远远甩开。这证明了 multi-view diffusion 在复杂空间关系建模上的鲁棒性。

---

### 6. 关于 Speed 和架构的终极思考

DreamFusion 时代的 SDS: 1.5 hours。
CSM Cube 2.0: 1 hour。
Meshy v3: 10 min。
**Meta 3DGen: 1 min。**

这 60 倍的速度提升完全归功于彻底的 feed-forward 架构。Pipeline 里的每一个组件 $\Phi_{\mathrm{mv}}^{\mathrm{obj}}, \Phi_{\mathrm{rec}}^{\mathrm{obj}}, \Phi_{\mathrm{mv}}^{\mathrm{tex}}, \Phi_{\mathrm{uv}}^{\mathrm{tex}}, \Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 都是一次性 forward pass 出结果，没有任何 test-time optimization。

**为什么 Meta 不做一个 End-to-End 模型？**
直接 `text -> 3D mesh with 4K PBR texture` 一把梭，听上去更优雅。实际上做不到。
1.  Shape 优化目标在 Volumetric space (SDF loss)，Texture 优化目标在 UV/View space (perceptual loss)。两者梯度尺度差异巨大，联合训练会互相干扰。
2.  端到端需要海量的 `(text, high-quality textured mesh)` pair 数据。现有的 synthetic 3D 数据集支撑不起这种暴力美学的训练。
3.  Modular 设计允许独立 scale 各个模块。比如 Emu 升级了，直接换新的 base generator 进去就能全系统升级。

Meta 3D Gen 本质上是一篇极佳的 system engineering paper。它告诉你：面对一个极度困难的跨模态生成任务，把问题拆解成多个表示空间，用各自的 prior 解决各自的子问题，再用工程 trick 桥接 distribution mismatch，往往比硬炼一个单一巨兽模型更有效。

**参考链接：**
- Meta 3D Gen (本篇): https://arxiv.org/abs/2407.01945
- Meta 3D AssetGen (Stage I 核心): https://arxiv.org/abs/2407.01944
- Meta 3D TextureGen (Stage II 核心): https://arxiv.org/abs/2407.01943
- DreamFusion (SDS 基线): https://arxiv.org/abs/2209.14988
- MVDream (Multi-view diffusion 先驱): https://arxiv.org/abs/2308.16552
- LRM (Large Reconstruction Model): https://arxiv.org/abs/2311.00891
- Cook-Torrance BRDF / PBR 理论: https://arxiv.org/abs/2106.12052 (NeuS 引用)

---

# Meta 3D Gen: Text-to-3D Pipeline 深度解读

## 1. Paper 整体定位

这是 Meta GenAI 团队 (Andrea Vedaldi, Natalia Neverova, Oran Gafni, David Novotny 等) 在 2024 年 7 月发布的技术报告,核心贡献是把两个先前独立开发的 foundation model **AssetGen** 和 **TextureGen** 拼接成一个 two-stage feed-forward pipeline,在 60 秒内完成 text-to-3D asset 生成,支持 PBR materials,且在 professional 3D artists 评估下整体胜率约 60-70%。

这篇 paper 本身的方法论创新其实不多 —— 它更像是一个 "engineering integration paper",真正的技术 depth 在引用的两篇 companion papers 里:
- **Meta 3D AssetGen** (Siddiqui et al., 2024): https://arxiv.org/abs/2407.01944
- **Meta 3D TextureGen** (Bensadoun et al., 2024): https://arxiv.org/abs/2407.01943

paper: https://arxiv.org/abs/2407.01945

---

## 2. Pipeline 架构解析

### 2.1 三种 3D 表示的融合

3DGen 的核心 insight 是同时利用三种 representation space,各自有优劣:

| Representation | 优点 | 缺点 |
|---|---|---|
| **View space** (multi-view images) | 高分辨率,detail 丰富,可继承 image generator 的 prior | 多视图之间不一致 |
| **Volumetric space** (SDF / voxel / NeRF-like) | 3D consistent,可提取 mesh | 训练数据稀缺,resolution 受限 |
| **UV space** (2D texture map) | 可以复用 2D diffusion,高分辨率 texture | 依赖 UV unwrap 质量,seams |

整合流程: text prompt → view space 生成 → volumetric 重建 → mesh + UV extraction → view space 再生成(条件于 geometry) → UV space fusion → 最终 texture。这种 "从 2D prior 起,绕道 3D,再回 2D" 的设计,本质上是借用 image diffusion 的强大 prior 来弥补 3D 数据稀缺。

### 2.2 Stage I: AssetGen

AssetGen 内部又是一个 multi-stage 过程:

**Step 1**: $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$ — multi-view diffusion
$$y \xrightarrow{\Phi_{\mathrm{mv}}^{\mathrm{obj}}} \{I_1, I_2, \ldots, I_K\} \sim p(I_1, \ldots, I_K \mid y)$$

- $y$: text prompt
- $I_k \in \mathbb{R}^{H \times W \times C}$: 第 $k$ 个 view,通道 $C$ 包含 shaded appearance + albedo(intrinsic image decomposition),这是 PBR 推断的关键 —— 让 model 同时输出 lit 和 unlit 版本,降低 material 推断的 ambiguity。
- $K$: 视图数量(通常是 4-8 个,继承 Zero123++/MVDream 的设计)
- 训练: 基于 Emu image generator 微调,用 multi-view + multi-channel supervision。

**Step 2**: $\Phi_{\mathrm{rec}}^{\mathrm{obj}}$ — deterministic reconstruction
$$\{I_1, \ldots, I_K\} \xrightarrow{\Phi_{\mathrm{rec}}^{\mathrm{obj}}} (M, T)$$

- 这个 network 是 deterministic 的(非 diffusion),与 $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$ 的 aleatoric 性质形成对比。
- 输出 mesh $M = (V, F, U)$:
  - $V \in \mathbb{R}^{|V| \times 3}$: vertices,$|V|$ 为顶点数,3 对应 $(x, y, z)$
  - $F \in \{1, \ldots, |V|\}^{|F| \times 3}$: triangular faces,$|F|$ 为面数,每行三个 index
  - $U \in [0, 1]^{|V| \times 2}$: UV coordinates,每个 vertex $v_i$ 映射到 $u_i = (u_i^{(s)}, u_i^{(t)})$
- 3D shape 表示用 **SDF (Signed Distance Field)** 而非 occupancy,这是 AssetGen 改进 DreamGaussian/Magic3D 的关键点 —— SDF 提供更平滑的 surface, 便于 mesh extraction (Marching Cubes / DMTet)。
- texture $T \in \mathbb{R}^{L \times L \times C_{\text{tex}}}$,$L$ 是 texture 分辨率,$C_{\text{tex}} \in \{3, 5\}$: 3 = RGB shaded(baked light);5 = albedo RGB + roughness + metalness(PBR)。

**Step 3**: Reprojection + fusion
$$T, T_1, \ldots, T_K \xrightarrow{\Phi_{\mathrm{uv}}^{\mathrm{obj}}} T^*$$

- 把 generated views $I_k$ reproject 到 UV space 得到 partial textures $T_k$,这些 $T_k$ 覆盖 UV map 的不同区域(只覆盖 visible 的部分)。
- $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 是 UV-space diffusion,负责 fill missing 区域 + enhance + fuse inconsistent overlap。
- 训练在 **auto-extracted UV maps**(automatic UV unwrap 算法输出)上,这是后面集成时一个关键 detail。

### 2.3 Stage II: TextureGen

给定 Stage I 的 mesh $M$,重新生成更高质量 texture:

**Step 1**: $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$ — geometry-conditioned multi-view diffusion
$$y, M \xrightarrow{\Phi_{\mathrm{mv}}^{\mathrm{tex}}} \{I_1, \ldots, I_K\} \sim p(I_1, \ldots, I_K \mid y, M)$$

- 关键 difference vs $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$: 多了 $M$ 作 conditioning。这把 "从 text 直接生成 3D-consistent views" 这个 ill-posed 问题降级为 "给定 shape,只生成 appearance",前者随 3D 数据稀缺而受限,后者可以用 image data 训练。
- 这就是为什么 Stage II 的 views 比 Stage I 的更 consistent,texture 更 sharp。

**Step 2**: Reprojection + UV fusion
$$y, T_1, \ldots, T_K \xrightarrow{\Phi_{\mathrm{uv}}^{\mathrm{tex}}} T$$

- $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ 训练在 **artist-created UV maps**(高质量 handcrafted unwrap)上。

**Step 3**: Super-resolution (optional)
$$T \xrightarrow{\Phi_{\mathrm{super}}^{\mathrm{tex}}} T_{\text{4K}}$$

### 2.4 集成的核心 trick: UV map mismatch 问题

直接把 AssetGen 的 mesh 喂给 TextureGen 的 $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ 不 work,因为:
- AssetGen 输出 **auto-extracted UV map**(通常用 xatlas / smart unwrap,patches 多、layout 不规整)
- TextureGen 的 $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ 训练在 **artist UV maps**(规整、对称、atlas 少)
- Distribution shift 导致 fusion network 在 auto-UV 上产生 seams、artifacts

3DGen 的解决方案是 **混用两个 UV fusion network**:

1. AssetGen $\Phi_{\mathrm{mv}}^{\mathrm{obj}} + \Phi_{\mathrm{rec}}^{\mathrm{obj}}$ → mesh $M$ + auto UV $U$
2. TextureGen $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$ (conditioned on $M$) → new views $\{I_k\}$
3. Reproject $\{I_k\}$ to UV space using $U$ → $\{T_k\}$
4. TextureGen $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ → consolidated $T$ (但这一步因为 UV mismatch 会有 residual seams)
5. **AssetGen $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$** takes $\{T_k\} + T$ → final $T^*$ (这一步 fix seams,因为它训练在 auto UV 上)

这个 "用 TextureGen 生成,用 AssetGen fuse" 的设计非常 engineering-flavored,反映了 distribution mismatch 的现实问题。

---

## 3. 实验: Prompt Fidelity 数据解读

### 3.1 404 prompts benchmark (继承自 DreamFusion)

划分:
- Objects: 156
- Characters: 106
- Compositions (角色 + 物体): 141

这个划分很关键 —— compositions 是最难的部分,因为涉及多个 semantic 实体的空间关系建模,这正是 single-stage multi-view diffusion 容易 collapse 的场景。

### 3.2 Prompt fidelity (Table 2)

| Method | Stage I | Stage II | Objects | Characters | Compositions |
|---|---|---|---|---|---|
| CSM Cube 2.0 | — | 69.1% | 84.0% | 87.8% | 54.6% |
| Tripo3D | — | 78.2% | 77.6% | 87.9% | 71.6% |
| Rodin Gen-1 | 59.9% | (single stage) | 66.7% | 70.1% | 48.8% |
| Meshy v3 | 60.6% | 76.0% | 97.2% | 83.2% | 63.5% |
| Third-party T23D | 73.5% | 79.7% | 95.0% | 89.7% | 67.9% |
| **Meta 3DGen** | **79.7%** | **81.7%** | 96.5% | 84.1% | **73.9%** |

关键 observations:
- 3DGen 在 compositions 上领先 (73.9% vs 第三方 T23D 的 67.9%)
- 在 simple objects 上 Meshy 略胜 (97.2% vs 96.5%),但 Meshy 在 compositions 上跌到 63.5%
- Rodin Gen-1 在 compositions 上只有 48.8%,且 7% 的 prompt 直接 fail(可能 mesh 算法在复杂几何上 diverge)
- Stage II 比 Stage I 平均提升 ~2%, 说明 texture refinement 对 prompt fidelity 也有贡献(texture 也承载 semantic information,比如 "gold llama" 的金色必须体现在 texture 上)

### 3.3 A/B 测试 (Table 3)

四类问题:
- **Q0 fidelity**: 哪个更符合 prompt
- **Q1 quality**: 哪个整体质量更好
- **Q2 texture**: 哪个 texture 更好
- **Q3 geometry**: 哪个 geometry 更正确

vs Rodin Gen-1 (professional artists):
- Q0: 68.0% win
- Q1: 59.8% win
- Q2: 69.1% win (Rodin 的 quad mesh 拓扑好,但 texture 是弱项)
- Q3: 56.7% win (Rodin 在 geometry 上较强,差距最小)

vs Meshy v3:
- Q2 texture: 53.7% win — 这个差距小,因为 Meshy 的 texture 风格非常 sharp/vivid
- Q3 geometry: 66.3% win — Meshy 经常有 Janus effect 和 inpainting artifacts

**关键 insight**: non-expert 用户和 professional artists 的偏好不同。Non-experts 喜欢 "sharp texture" 即使有 artifact,professionals 更看重 geometry 正确性。这说明 evaluation 3D generation 必须用专业 annotators,否则会被 "texture 锐度" 误导。

### 3.4 速度 (Table 1)

| Method | Stage I | Stage I+II |
|---|---|---|
| CSM Cube 2.0 | 15 min | 1 hour |
| Meshy v3 | 1 min | 10 min |
| Third-party T23D | 10 sec | 10 min |
| Rodin Gen-1 | — | 3 min |
| Tripo3D | 30 sec | 3 min |
| **Meta 3DGen** | **30 sec** | **1 min** |

3DGen 比 Meshy 快 10×,比 CSM 快 60×,且 Meshy/CSM 都不是 feed-forward(有 per-asset optimization)。

---

## 4. 关键直觉构建

### 4.1 为什么 SDS 路线被淘汰?

DreamFusion (Poole et al., 2023, https://arxiv.org/abs/2209.14988) 引入 Score Distillation Sampling:
$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon} \left[ w(t) (\epsilon_\phi(\alpha_t x_\theta + \sigma_t \epsilon; y, t) - \epsilon) \frac{\partial x_\theta}{\partial \theta} \right]$$

- $\theta$: 3D 表示参数(NeRF / Gaussians / mesh)
- $x_\theta$: 渲染图像
- $\epsilon_\phi$: frozen 2D diffusion model 的 noise predictor
- $\alpha_t, \sigma_t$: diffusion forward process 的 noise schedule 参数
- $w(t)$: 时间步权重

问题:
1. **慢**: 每个角度的 SDS gradient 都要 backprop 到 3D,通常要 30 min - 数小时
2. **Janus effect**: SDS 没有显式的 multi-view consistency constraint,diffusion model 倾向在每个 view 独立生成 "front face",导致 multi-face artifact
3. **过 smoothing**: 长时间 optimization 会 collapse 到 mean mode,失去 detail

3DGen 走的是 **feed-forward multi-view diffusion + reconstruction model** 路线,完全避开 per-asset optimization:
- Multi-view diffusion (MVDream, Zero123++) 强制 view consistency
- Large Reconstruction Model (LRM, Instant3D) 学到 image-to-3D prior,一次 forward 就出 mesh

### 4.2 LRM 路线的限制

LRM (Hong et al., 2024, https://arxiv.org/abs/2311.00891) 训练 single-image-to-3D reconstruction,泛化性好但有几个 bottleneck:
- **Texture quality**: 因为 reconstruction 在低分辨率 volumetric space 做,texture 被 downsampling,失去高频 detail
- **PBR 不可分解**: reconstruction 通常输出 RGB radiance(baked light),不能 relight
- **UV unwrap 问题**: 自动 unwrap 的 UV map 质量差,texture 在 seams 处有明显 artifacts

3DGen 解决这些问题的方法:
1. 用 view-space diffusion (高分辨率 2D prior) 补充 volumetric space 的低分辨率
2. Multi-channel 输出(shaded + albedo)强制 model 学 intrinsic decomposition
3. UV fusion network ($\Phi_{\mathrm{uv}}^{\mathrm{obj}}$) 专门 fix auto-UV 的 seams

### 4.3 PBR 物理意义

Physically-Based Rendering 需要分解 appearance 为:
- **Albedo** $\rho_d \in \mathbb{R}^3$: 表面 base color,与光照无关
- **Roughness** $r \in [0, 1]$: 微表面粗糙度,控制 specular 模糊度
- **Metalness** $m \in \{0, 1\}$: 是否金属(影响 Fresnel 反射行为)

Rendering equation (Cook-Torrance BRDF, 参考 https://arxiv.org/abs/2106.12052):
$$L_o(\omega_o) = \int_{\Omega} f_r(\omega_i, \omega_o) L_i(\omega_i) \cos\theta_i \, d\omega_i$$

$$f_r = k_d \cdot \frac{\rho_d}{\pi} + k_s \cdot \frac{D(\theta_h) G(\theta_i, \theta_o) F(\theta_h)}{4 (\cos\theta_i)(\cos\theta_o)}$$

- $\omega_i, \omega_o$: 入射、出射方向
- $k_d = (1 - m)$, $k_s = m$: metalness 控制 diffuse/specular 比例
- $D$: normal distribution function (GGX)
- $G$: geometry/shadowing function
- $F$: Fresnel term (Schlick's approximation: $F = F_0 + (1 - F_0)(1 - \cos\theta_h)^5$)
- $\theta_h$: half-vector 角度
- $F_0$: base reflectance,非金属约 0.04,金属 = albedo RGB

3DGen 让 model 输出 albedo + roughness + metalness 三个 map,可以 re-light 任意环境。这比 "baked texture" (仅 RGB shaded) 工业可用性高一个数量级。

### 4.4 Emu 作为 base generator

3DGen 的所有 diffusion component 都基于 Emu (Dai et al., 2023, https://arxiv.org/abs/2309.15807):
- Emu 是 Meta 内部 text-to-image 模型,性能对标 SDXL/MidJourney v5 时代
- 用 renders of synthetic 3D data (内部数据集) fine-tune 成 multi-view / UV-space generator
- 这种 "先训 T2I,再 fine-tune 到 3D" 的范式由 MVDream (https://arxiv.org/abs/2308.16552) 推广

### 4.5 数据稀缺问题

paper 提到一个关键 challenge: 3D 数据比 image/video 数据少 3-4 个数量级。这就是为什么所有现代 text-to-3D 方法都依赖 image prior,而非直接训 3D diffusion(像 Shape-E https://arxiv.org/abs/2212.08751 和 Point-E https://arxiv.org/abs/2212.08751 试过,但泛化差)。

3DGen 用的训练数据是 Meta 内部 synthetic 3D asset collection,具体大小没披露,但从同期的 Objectaverse (https://arxiv.org/abs/2212.00878) 和 Objaverse-XL (https://arxiv.org/abs/2307.05663) 规模看,大概是 100k-1M 资产量级。

---

## 5. 与同期工作的关联

### 5.1 Multi-view diffusion 家族

| Method | View 数 | Conditioning | 用途 |
|---|---|---|---|
| Zero123 (Liu et al., 2023) | 1 → 1 | single image | novel view synthesis |
| Zero123++ (Shi et al., 2023) | 1 → 6 | single image | sparse reconstruction |
| MVDream (Shi et al., 2024) | text → 4 | text | text-to-3D |
| ImageDream (Wang & Shi, 2024) | image → 4 | image | image-to-3D |
| Wonder3D (Long et al., 2023) | 1 → 6 + normal | image | color+normal joint |
| SyncDreamer (Liu et al., 2023) | 1 → 16 | image | 3D consistent views |
| **AssetGen $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$** | text → K | text + multi-channel | text-to-3D with PBR |
| **TextureGen $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$** | text+M → K | text + geometry | text-to-texture |

### 5.2 Reconstruction model 家族

| Method | Input | Representation | Backbone |
|---|---|---|---|
| 3D-R2N2 (Choy et al., 2016) | multi-view | voxel | 3D CNN |
| PixelNeRF (Yu et al., 2021) | multi-view | NeRF | ViT + cross-attn |
| LRM (Hong et al., 2024) | 1 image | triplane | large transformer |
| Instant3D (Li et al., 2024) | 4 views | triplane | LRM-style |
| TripoSR (Tochilkin et al., 2024) | 1 image | triplane | LRM 变体 |
| **AssetGen $\Phi_{\mathrm{rec}}^{\mathrm{obj}}$** | K multi-channel views | SDF + texture | transformer |

### 5.3 Texture generation 家族

| Method | 空间 | 方式 | 速度 |
|---|---|---|---|
| Text2Tex (Chen et al., 2023) | view (sequential) | inpainting + SDS | 慢 |
| Texturify (Siddiqui et al., 2022) | surface | GAN | 中 |
| Texture (Richardson et al., 2023) | view (sequential) | depth-conditioned | 慢 |
| TexFusion (Cao et al., 2023) | view + UV | alternating | 中 |
| Point-UV (Yu et al., 2023) | point cloud | diffusion | 中 |
| **TextureGen** | view + UV | two-stage diffusion | 快(feed-forward) |

### 5.4 SDS / VSD 路线 (反例)

- DreamFusion (https://arxiv.org/abs/2209.14988): SDS, ≈ 1.5 hour/asset
- Magic3D (https://arxiv.org/abs/2211.10440): coarse-to-fine SDS
- ProlificDreamer (https://arxiv.org/abs/2305.16213): VSD (Variational SDS)
- DreamGaussian (https://arxiv.org/abs/2309.16653): SDS + 3DGS, 数分钟
- HiFA (https://arxiv.org/abs/2305.18766): improved SDS guidance

3DGen 完全放弃了 SDS 路线,走 feed-forward,这是 industry 趋势(因为 SDS 在 production 中太慢)。

---

## 6. Failure Modes 分析 (Figure 8)

paper 中提到的典型 failure:
1. **Janus effect** (Meshy v3 常见): multi-view 不一致,生成多面孔
2. **Texture seams** (UV unwrap 边界处): 这是 auto-UV 的固有问题,3DGen 用 $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 缓解
3. **Inpainting artifacts** (Meshy): 在不可见区域 inpaint 失败
4. **Geometry collapse** (Rodin Gen-1, 7% failure): 复杂 mesh 提取失败
5. **Over-smoothing** (CSM): 长时间 optimization 导致 detail 丢失

3DGen 的主要 failure mode 在 paper 中没有详细列举,但从架构推断:
- Texture 在极端 thin structures(头发、毛皮)上可能 fail,因为 UV unwrap 在这些区域 degenerate
- PBR 在 transparent/translucent materials(glass, water)上 fail,因为 Cook-Torrance BRDF 假设 opaque surface
- Compositions 中物体之间空间关系仍可能错位,虽然胜率最高(73.9%)

---

## 7. 工程直觉

### 7.1 为什么 two-stage 而不是 end-to-end?

理论上 end-to-end 训练一个 "text → high-quality textured mesh" 应该更好,但实际有阻碍:
1. **3D 数据稀缺**: end-to-end 需要 paired (text, high-quality textured mesh) 数据,这种数据集极小(几千到几万)
2. **Multi-task 干扰**: shape 和 texture 在不同 representation space,共享 backbone 会互相干扰
3. **Modular 训练**: 可以独立调试、独立 scale,AssetGen 和 TextureGen 各自 100% + 100% 比 joint 80% + 80% 好

3DGen 的 two-stage 是 **decoupled training + coupled inference** 的典型 pattern,类似 Stable Diffusion + ControlNet 的设计哲学。

### 7.2 为什么 TextureGen 不直接接 artist mesh?

TextureGen 完全可以用于 artist-created mesh(retexturing use case,paper Figure 9-10 展示了):
- Artist mesh 有规整 UV map
- $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$ condition on $M$,生成 views
- Reproject 到 UV
- $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ fuse(因为 UV 是 artist 风格,match training distribution)

所以 3DGen 实际上支持两种 retexture:
- Artist mesh + new prompt → new texture (用 TextureGen 完整 pipeline,20 sec)
- Generated mesh + new prompt → new texture (需要混合 $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 来 fix UV mismatch)

### 7.3 inference time 分解

Stage I (30 sec):
- $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$ diffusion: ~10 sec (K views, 50 steps DDIM)
- $\Phi_{\mathrm{rec}}^{\mathrm{obj}}$ forward: ~5 sec (large transformer)
- Mesh extraction (Marching Cubes / DMTet): ~5 sec
- $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ fusion: ~10 sec

Stage II (20 sec):
- $\Phi_{\mathrm{mv}}^{\mathrm{tex}}$ diffusion: ~10 sec (geometry-conditioned, 同 50 steps)
- Reprojection: ~1 sec
- $\Phi_{\mathrm{uv}}^{\mathrm{tex}}$ fusion: ~10 sec

这是估算,基于 LRM (~5 sec) 和 SDXL (~10 sec for 50 steps) 的经验。paper 没披露具体 step 数和 model size。

---

## 8. 缺失的细节 (paper 的局限)

paper 作为 "integration report",隐藏了大量技术 detail:
1. **Model size**: 没说 $\Phi_{\mathrm{mv}}^{\mathrm{obj}}$, $\Phi_{\mathrm{rec}}^{\mathrm{obj}}$, $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ 各自参数量。Emu 的 base size 估计 1-3B params。
2. **Training data**: 没说用了多少 3D assets,以及 text prompt 来源。
3. **Multi-view 数量 K**: 没明确,推测 K=4 (沿 MVDream 惯例) 或 K=6 (沿 Zero123++ 惯例)。
4. **Diffusion schedule**: 没说 DDIM/EDM,step 数,classifier-free guidance weight。
5. **PBR supervision**: 没说 albedo/roughness/metalness ground truth 来源(synthetic render 的 BRDF 设置)。
6. **Ablation**: 缺乏 "去掉 Stage II 会怎样"、"去掉 $\Phi_{\mathrm{uv}}^{\mathrm{obj}}$ fix step 会怎样" 的系统 ablation。

要真正理解技术细节,需要读 AssetGen paper (https://arxiv.org/abs/2407.01944) 和 TextureGen paper (https://arxiv.org/abs/2407.01943)。

---

## 9. 个人 takeaway

1. **3D generation 的瓶颈在 representation,不在 model size**: 这篇 paper 没用更大的 transformer,主要创新在 representation 路由(view → volumetric → UV → view → UV 的循环)。
2. **Distribution mismatch 是 system integration 的核心 enemy**: UV map 例子特别经典 —— 两个 model 各自 SOTA,组合起来 fail,需要 hybrid fusion。这反映了 ML system 的真实复杂度。
3. **PBR 是 production 的硬门槛**: 不能 relight 的 3D asset 在游戏/电影 pipeline 几乎不可用,3DGen 把 PBR 当 first-class citizen 是正确的战略选择。
4. **Feed-forward 是 industry 共识**: SDS 路线在 paper 中已经基本被判定为 "too slow for production",所有 commercial baselines(Meshy, Rodin, Tripo3D)都在转向 feed-forward。
5. **Evaluation 必须用 domain experts**: non-expert 评估会被 texture 锐度 biased,professionals 更关注 geometry 正确性,这对未来 3D 评估 benchmark 设计有指导意义。

---

## 10. 延伸阅读

核心 paper:
- Meta 3D AssetGen: https://arxiv.org/abs/2407.01944
- Meta 3D TextureGen: https://arxiv.org/abs/2407.01943
- Meta 3D Gen (本篇): https://arxiv.org/abs/2407.01945

基础工作:
- DreamFusion (SDS): https://arxiv.org/abs/2209.14988
- Magic3D: https://arxiv.org/abs/2211.10440
- ProlificDreamer (VSD): https://arxiv.org/abs/2305.16213
- MVDream: https://arxiv.org/abs/2308.16552
- Zero123++: https://arxiv.org/abs/2310.15110
- Wonder3D: https://arxiv.org/abs/2310.15008
- LRM: https://arxiv.org/abs/2311.00891
- Instant3D: https://arxiv.org/abs/2311.00891 (实际 https://arxiv.org/abs/2401.00907)
- TripoSR: https://arxiv.org/abs/2403.02151
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14505
- NeuS: https://arxiv.org/abs/2106.10689
- Emu: https://arxiv.org/abs/2309.15807
- CAT3D: https://arxiv.org/abs/2405.04476
- IM-3D (ICML 2024): https://arxiv.org/abs/2403.00256

数据集:
- Objaverse: https://arxiv.org/abs/2212.00878
- Objaverse-XL: https://arxiv.org/abs/2307.05663

Industry baselines:
- CSM Cube 2.0: https://www.csm.ai
- Tripo3D: https://www.tripo3d.ai
- Rodin Gen-1: https://hyperhuman.deemos.com/rodin
- Meshy v3: https://www.meshy.ai

如果你想 deep dive 真正的技术细节,建议先读 AssetGen paper(它讲了 SDF reconstruction 和 multi-channel PBR diffusion 的训练细节),再读 TextureGen paper(讲 view-space + UV-space joint training),最后回来读 3DGen 这篇 integration report,会发现很多 "为什么这样集成" 的设计动机才说得通。
