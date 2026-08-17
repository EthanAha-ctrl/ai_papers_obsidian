---
source_pdf: SynCity.pdf
paper_sha256: ec397872690bf816f63729d5472ca363c9a1b2202a27f72149eeaa579c9778a2
processed_at: '2026-08-12T11:47:00-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,用大白话讲一遍。

## 一句话总结

作者发现一个事儿: TRELLIS 这个 3D generator 虽然是拿 object 数据训的, 但你喂给它一张 isometric view 的"小场景图"(比如一个 tile 里有个房子加几棵树), 它其实能 reconstruct 出来。于是整篇 paper 就围绕这个发现展开——把"生成一整个世界"拆成"生成一堆 tile 再拼起来"。

## 为什么这件事难

你想生成一个大 3D 世界让人走进去逛, 现有路线两条, 都有硬伤:

**路线 A: 拿 2D image generator 往外画, 再 reconstruct 成 3D。** Flux 这类 model 画功很好, 你给它 prompt 它能画出漂亮场景。但问题是画完之后你要把它变 3D, 得估深度、做 NeRF 或 3DGS。一步步往外扩的时候, 几何会 drift, 走几步就穿帮。所以这类方法最后给你一个"3D bubble", 你站在中间转头看 360 度还行, 往前走两步就不行了。World Labs 那家公司的 SOTA 也这样。

**路线 B: 直接训一个 3D scene generator。** BlockFusion、LT3SD 这种, 直接在 3D 数据上训 diffusion。好处是几何 coherent, 能生成大场景。坏处是你得有 3D scene 数据, 训出来又只能生成特定 domain (比如 BlockFusion 只会 city, LT3SD 只会 indoor room), 而且 mesh 没 texture, 样子很单调。它们没沾 2D image generator 的光, 所以 artistic quality 和 prompt 理解能力都差。

SynCity 想鱼和熊掌兼得: 用 3D generator (TRELLIS) 保证几何规整, 用 2D generator (Flux) 保证 artistic quality 和 prompt 理解, 再用 LLM (ChatGPT) 把 world-level 描述拆成 tile-level 描述。三个都是现成 model, 零训练。

## 怎么做的

想象你在玩 SimCity, 地图是个 grid, 一格一格地填。SynCity 就这个思路。

### Step 1: LLM 拆 prompt

你给 ChatGPT 一句"一个中世纪小镇, 有条河穿过", 它吐给你一个 JSON: 每个格子的 tile prompt + 一个 global style prompt。比如 tile (0,0) 是"古石桥跨过小溪", tile (1,0) 是"溪水流过苔藓河岸", global 是"medieval setting, isometric view, glowing lanterns, soft shading"。

### Step 2: Flux 画 tile 的 isometric 图

这是整个 pipeline 最 hacky 也最聪明的部分。

直接 prompt Flux "画一个 isometric 的 city tile", 它不听话, 视角乱飘。作者的 trick 是: 先准备一张灰色方块板子(isometric 视角画的), 再在板子上方画个 mask, 让 Flux 做 inpainting——"在这个框框里填内容"。Flux 就老老实实把东西画在板子上了。

这个板子 = geometric anchor, 强制 Flux 输出的 framing 稳定, 后面才好拼。

对于非第一个 tile, 旁边的 tile 已经生成好了, 就把已生成的 3D 世界部分 render 成 isometric 图当 context 塞给 Flux。这样新 tile 的 scale、风格、颜色和邻居一致。有个细节: 如果邻居太高挡住当前 tile, 先把高出来的部分切掉再 render context。

### Step 3: TRELLIS 把 tile 图变 3D

把 Flux 画的图里新 tile 那块抠出来(用 mask + rembg 去 background)。然后又一个 trick: 在 tile 下面贴一个稍微大一点的灰色 slab, 叫 "rebasing"。

为什么贴这个 slab? 因为 TRELLIS 生成 3D 的时候, geometry 经常溢出 base 边界, 你根本搞不清 tile 到底多大、ground 在哪。贴了 slab 之后:
- TRELLIS 会把 geometry 收敛在 slab 上面
- slab 是个 easy-to-detect 的 handle, 后续可以自动找 tile 边界
- 可以验证 reconstruction 质量(检查 slab 有没有被 faithfully 重建)

数据说话: 不贴 slab, base area 2271, squareness 0.92, completeness 0.73; 贴了, base area 4096 (=64² 满分辨率), squareness 1.00, completeness 1.00。差距巨大。

然后还有一些 post-processing: 因为 TRELLIS 生成的 tile 大小、朝向、ground height 都不固定, 得 crop、resize、reorient。朝向那个特别朴素——试 4 个 90 度旋转, 渲染出来和原图比 LPIPS, 哪个最像就用哪个。

### Step 4: 把 tile 拼起来 (3D Blending)

虽然前面步骤让 tile 大致对齐了, 直接 union 的话边界还是有缝。

作者的做法是在 TRELLIS 的 latent space 里做 partial denoising。具体说:

1. 先在 2D 把两个相邻 tile 并排 render, 用 Flux inpaint 中间那条缝, 得到一张"无缝过渡"的图当 conditioning。
2. 把两个 tile 的 latent $\gamma^1, \gamma^2$ 拼一起, 左半边来自 $\gamma^1$, 右半边来自 $\gamma^2$, 缝在中间 $x=R/2$。
3. 初始化一个噪声 latent, 开始 denoise。但只在中间那条缝的宽度 $r$ 范围内 denoise, 两侧保持 frozen(加噪后的原始 latent)。这样缝的地方被重新生成得平滑, 两侧 tile 内容不动。

还有一个 latent upsampling 的问题: 因为 rebasing 后 crop 过, 两个 tile 的 latent 分辨率可能不一样。直接 interpolate 会坏, 因为 TRELLIS 的 latent 是 sparse 的。作者的做法是 upsample occupancy volume, 然后重新 denoise latent, 同时用多个 view 的渲染做 conditioning, 每个 step 取所有 view denoise step 的平均。LPIPS 从 naive 的 0.59 降到 0.32。

## 效果

人评对比 BlockFusion (一个在 city 数据上训过的 baseline): overall win rate 90.9%, geometry 81.8%, exploration 90.9%, diversity 90.9%, realism 86.4%。BlockFusion 是专门训的, SynCity 是 training-free, 还全面碾压。

生成的世界可以走进去逛, 不只是个 bubble。加个 skybox 就是个可探索的小游戏关卡的感觉。

## 我的看法

这篇 paper 的气质很像那种 "hacker 用现有工具搭出超越专门系统的东西"。每个 trick 都对应一个具体的 failure mode: isometric framing 对应 Flux 视角乱飘, rebasing 对应 TRELLIS geometry 溢出, context rendering 对应 tile scale 不一致, latent partial denoise 对应 3DGS union 有缝。没有花哨的理论, 全是工程上的 grounded 决策。

代价也有: pipeline 很长, tile 是 atomic 的(跨 tile 的结构比如一条长河需要 Flux 和 TRELLIS 默契配合, 不总是成功), 继承了 Flux 和 TRELLIS 各自的 limitation。

但从研究方向上, 它 points to 一个有意思的路线: 与其苦哈哈训 3D scene generator, 不如把 2D generator 的 artistic intelligence 和 3D generator 的 geometric regularity 用 prompt engineering 编起来。等 3D generator 再迭代几代 (Hunyuan3D 2.0, TripoSG 这些), 这个 recipe 的效果只会更好。

---

# SynCity: Training-Free Generation of 3D Worlds 深度解析

## 一、核心 Intuition

这篇paper的核心 insight 可以这样理解: **TRELLIS 虽然 trained 在 object-centric data 上, 但如果你给它一个 isometric view 的 "tile" (一个局部区域, 包含 building/bridge/trees 等多个 object 的 composition), 它其实能 reconstruct 出相当复杂的 local 3D 结构**。作者把这个 property 利用起来, 把 "生成整个 world" 的问题 reduce 成 "autoregressive 生成一堆 tile, 再 stitch 起来" 的问题。整个 pipeline 完全 training-free / optimization-free, 全靠 prompt engineering 把 LLM (ChatGPT o3-mini-high)、2D generator (Flux ControlNet)、3D generator (TRELLIS) 三者串起来。

这个思路其实和 video game 的 tile-based world building 高度类似 (论文里也明确 mention 了), 不同的是这里用 generative model 替代了 procedural generation, 用 LLM 替代了 designer。

## 二、Pipeline 全景 (对应 Figure 2)

Pipeline 有四个 stage:

1. **Language prompting** (Sec 3.1): $p_0$ (world description) → $\{p_{xy}\}_{(x,y)\in\mathcal{T}} \cup \{p_\star\}$, 其中 $p_{xy}$ 是 tile-specific prompt, $p_\star$ 是 global style prompt, $\mathcal{T} = \{0,\dots,W-1\}\times\{0,\dots,H-1\}$ 是 tile grid。
2. **2D prompting** (Sec 3.2): $q = p_{xy} \cdot p_\star$ → Flux ControlNet inpainting → isometric image $I(x,y)$。
3. **3D prompting** (Sec 3.3): $I(x,y)$ → foreground extraction + rebasing → $J(x,y)$ → TRELLIS → 3D Gaussian Splats $G(x,y)$。
4. **3D blending** (Sec 3.4): 把相邻 tile 在 TRELLIS 的 latent space 里 stitch, decode 出来 merge。

## 三、2D Prompting 的关键设计 (Sec 3.2)

### 3.1 Isometric framing 的 trick

直接 prompt Flux "isometric view of a city tile" 是不稳定的 (Figure 4 右侧), viewpoint 和 framing 会 random。作者的 trick 是构造:

- **Base image $B$**: 一张灰色方形 slab, 从固定的 isometric vantage point render 出来。
- **Inpainting mask $M$**: 一个 binary mask, 覆盖 slab 上方的 cube 区域。

然后 $I(x,y) \sim \Phi_{2D}(q, B, M)$。这相当于把 Flux 的 inpainting 当成一个 "受约束的 image-to-image" 任务, 强制它把 tile 生成在一个固定的 geometric frame 里。

### 3.2 Context-aware generation

对于 $(x,y) \neq (0,0)$, 已经生成的 tile 要给当前 tile 提供 context。具体做法:

- 把已经生成的 3D world 部分 render 成 base image $B$ (用 isometric view)。
- mask $M$ 改成不覆盖 west side 已经生成的 tile (Figure 5)。
- **Trimming tall structures** (Figure 6): 在 render context 之前, 把 high 到会 occlude 当前 tile 的 geometry 切掉, 这样 context 不会挡住要生成的 tile。

这里有个 bootstrapping 的 edge case (Appendix A.2): 对于 $\mathcal{L} := \{(x,y) \in \mathcal{T}: x=0 \land y>0\}$ 这些 tile, 因为 build order 是 row-by-row, 它们 west 边没东西, 所以作者 temporarily 把 $(0,y-1)$ 复制一份放在 $(-1, y)$ 提供 context, inpainting 完再移除。这个 trick 让上下 scale 一致。

## 四、3D Prompting 的关键设计 (Sec 3.3)

### 4.1 Foreground extraction & Rebasing

从 $I(x,y)$ 提取新 tile 部分 (用 mask + rembg + alpha matting), 然后 narrow crop。关键 trick 是 **rebasing** (Figure 7): 在 tile 下面 2D compose 一个 slightly larger 的灰色 slab, 得到 $J(x,y)$。

为什么 rebasing 重要? 看 Figure 8 和 Table 2:
- **没 rebasing**: TRELLIS 生成的 geometry 会 extend 超出 base, tile 真实 extent 难以 detect; Base Area = 2271, Squareness = 0.92, Completeness = 0.73。
- **有 rebasing**: Base Area = 4096 (= 64², 满分辨率!), Squareness = 1.00, Completeness = 1.00。

这个 base 同时提供了三个功能: (1) 让 tile 几何包含在 ground 之上; (2) 提供一个 easy-to-detect 3D handle 用于后续 post-processing; (3) 验证 reconstruction 质量的 anchor。

### 4.2 3D Geometric Validation (Appendix A.3)

TRELLIS 第一阶段输出 occupancy volume $V \in \{0,1\}^{R \times R \times R}$, $R=64$。作者用两个 heuristic 验证:

**Squareness check**: 对当前 height $w$ 求 active voxels 的 bounding box $(u_{\min}, u_{\max}, v_{\min}, v_{\max})$, 定义 $\text{ext}_u = \max\{0, 1 + u_{\max} - u_{\min}\}$ (宽度), $\text{ext}_v$ 同理 (高度)。丢弃条件:
- $\text{ext}_u \cdot \text{ext}_v < (R/2)^2$ (面积太小)
- $\min\{\text{ext}_u, \text{ext}_v\} / \max\{\text{ext}_u, \text{ext}_v\} < \alpha$ ($\alpha=1$, 不够 square)

**Base faithfulness check**: 找到 base 最大的 height $w^* = \arg\max_w \text{ext}_u(w) \cdot \text{ext}_v(w)$, 然后构造 template $V_B$ (附录公式, 实际上是 base footprint 在 $w^*$ 处的 indicator)。如果 $(V \cdot V_B) / (V_B \cdot V_B) < \beta = 0.95$ 就 discard。

### 4.3 3DGS Post-processing (Appendix A.4)

- **3D cropping/resizing/centering**: 通过分析 xy footprint 找四个 axis-aligned cuts。找 left cut $x^*$ 的方法: 取 slice $V_x = \{(x', y', z') \in \mathbb{R}^3 : x-\delta \leq x' < x+\delta\}$, 算该 slice 内 Gaussians 的平均 color $c_x$, 计算 $d(x) = \|c_x - c_{x_{\min}}\|$, 取 $x^* = \min\{x : d(x) > \tau\}$, 即从 background color 过渡到 "有东西" 的第一个 slice。这一步本质上是用 color contrast 自动 detect tile 边界。
- **Surface level alignment**: 用 base 四角的平均高度作为 ground surface level, 解决 TRELLIS 把 object 垂直 center 导致不同 tile ground level 不一致的问题。
- **3D reorientation**: TRELLIS 生成时 axis 是 ambiguous (限 90 度旋转)。试 4 个 90 度旋转, 渲染 view, 和 $\tilde{I}(x,y)$ 比 LPIPS, 取最小的作为正确 orientation。

## 五、3D Blending 在 Latent Space (Sec 3.4) — 这是最 math-heavy 的部分

### 5.1 2D Blending (准备 conditioning)

把两个 3D tile 并排放, 渲染 frontal view, 用 Flux inpaint 中间区域 (Figure 2), 得到一张 well-blended image 作为 conditioning。

### 5.2 3D Latent Stitching — 核心公式

TRELLIS 的 latent 是 $\gamma \in \mathbb{R}^{D \times R \times R \times R}$, 即 $D$ 维特征在 $R \times R \times R$ 的 3D voxel grid 上 (第二阶段 $R=64$, 第一阶段 $R=16$)。

两个相邻 tile 的 latent $\gamma^1, \gamma^2$ 在 side $x = R/2$ 处 stitch:

$$
\gamma_{:,x,y,z} = \begin{cases} \gamma^1_{:,x+R/2,y,z}, & \text{if } x < R/2 \\ \gamma^2_{:,x-R/2,y,z}, & \text{if } x \geq R/2 \end{cases}
$$

变量解释:
- $:,x,y,z$: $:$ 是 feature channel 维度 (即 $D$ 维), $x, y, z$ 是 voxel grid 的空间坐标。
- $\gamma^1$ 占左半 (取它中心右半部分, 即 $x+R/2$ 偏移), $\gamma^2$ 占右半 (取它中心左半部分, $x-R/2$ 偏移)。这样两个 tile 各自的 "tile content" 被放到一起, 边界在 $x = R/2$ 处。

### 5.3 区域性 Denoising — 第二个核心公式

初始化 $\tilde{\gamma} \sim \mathcal{N}(0, I)$, 然后在每个 denoising step $t$:

$$
\tilde{\gamma}_{t+1,:,x,y,z} = \begin{cases} \Omega(\tilde{\gamma}_{t,:,x,y,z}), & \text{if } |x - R/2| \leq r \\ \gamma_{t+1,:,x,y,z}, & \text{otherwise} \end{cases}
$$

变量解释:
- $\Omega$: TRELLIS 的 latent denoiser。
- $r$: blend region 的半径, 满足 $r < R/2$。这是用户可调参数, 控制 "blend 多宽的边界区域"。
- $\gamma_t$: 把原始 $\gamma$ 加上对应 noise level $t$ 的噪声得到的版本, 是 "frozen" 的 conditioning。
- 只在中间 $|x - R/2| \leq r$ 区域 (即 boundary strip) 用 $\Omega$ denoise, 其余区域 frozen 在加了噪的 $\gamma$ 上。

这是一种 partial diffusion: 边界区域被重新 denoise (让它和 2D blended conditioning 一致), 而非边界区域保持原来 tile 的 latent。在实践中作者只对第二阶段 (resolution $R=64$) 做, 因为第一阶段 $R=16$ 太粗, $r$ 的选择没空间。

### 5.4 Latent Upsampling (Figure 9, Table 3)

问题: rebasing 之后 cropping 导致相邻 tile 的 latent 分辨率不同, 但 5.2 的 stitching 公式假设 $\gamma^1, \gamma^2 \in \mathbb{R}^{D \times R \times R \times R}$ 同分辨率。所以需要 upsample 回 $R$。

**Naive interpolation 不行** (Figure 9 上): 因为 TRELLIS latent 是 sparse 的, 加上 decoder 的 quirks, 直接 interpolate 会产生 artifact。

**作者的方法**:
1. 先 upsample occupancy volume $V$ 回原分辨率 $R \times R \times R$。
2. Denoise 一组新的 latent $\gamma$ 在 upsampled occupancy 上。
3. 为保留原始 tile 的细节, 渲染多个 view, 联合 condition: 每个 timestep 的 denoise step 取所有 view denoise step 的 **平均**。

Table 3 的 quantitative 对比 (200 个 tile, 每个 10 个 view):
| Method | LPIPS↓ | SSIM↑ | FID↓ | KID↓ |
|---|---|---|---|---|
| Naive upsampling | 0.5914 | 0.3093 | 200.5 | 0.243 |
| Ours (single frame) | 0.3517 | 0.5149 | 111.6 | 0.069 |
| Ours (multi frame) | **0.3212** | **0.5312** | **89.1** | **0.051** |

LPIPS 降到 0.32, FID 降到 89.1, 显著优于 naive。

## 六、实验结果

### 6.1 Human Preference (Table 1)

对比 BlockFusion (一个需要 domain-specific 3D training data 的 baseline), 22 个参与者:
| Overall | Geometry | Exploration | Diversity | Realism |
|---|---|---|---|---|
| 90.9% | 81.8% | 90.9% | 90.9% | 86.4% |

注意 BlockFusion 是 trained 在 city data 上的, 而 SynCity 是 training-free 用 general model, 仍然全面碾压。

### 6.2 Ablations

**Context importance** (Figure 10): 移除 2D context, 每个 tile 独立 sample, building scale 不一致。

**Rebasing** (Table 2): 前面讲过, Base Area 2271 vs 4096, Completeness 0.73 vs 1.00。

**3D Blending** (Figure 12): 不 blend 的话 tile 边界 discontinuity 明显。

**Non-iterative city building** (Appendix A.5, Figure 15/16): 试图一次性让 Flux + TRELLIS 生成大 scene, 即使 prompt 加 explicit layout (e.g., "house in bottom left, pharmacy in top right"), layout 指令基本被 ignore, 细节也很差。这是 tile-based approach 的 motivation。

## 七、个人 Intuition & 评价

我觉得这篇工作最 elegant 的地方是它把 "training-free" 这件事做到极致: 没有任何 fine-tuning, 没有 SDS-style optimization (像 DreamFusion), 完全靠 prompt engineering + 既有 model 的组合。代价是 pipeline 长, 而且 (limitations 里也提到) tile 之间是 "atomic" 的, 跨 tile 的结构 (比如一条河) 需要 Flux 和 TRELLIS "harmonious cooperation"。

几个我观察到的技术亮点:

1. **Rebasing 是 simplest-but-most-effective trick**: 在 2D 加一个 gray slab 当 base, 既给 TRELLIS 一个 geometric anchor, 又给 post-processing 一个 detectable handle, 还能 validate reconstruction quality。Table 2 的 Completeness 从 0.73 → 1.00 很说明问题。

2. **Latent-space blending 而非 3DGS-space blending**: 在 3DGS 空间直接 union 会有 visible seam, 在 latent space 用 partial denoising + multi-view conditioning 能让 boundary 平滑过渡。这个思路其实和 Instruct-NeRF2NeRF、View-Consistent Editing 一类工作类似, 都是 "在 latent / feature 空间 partial denoise"。

3. **Multi-view conditioning 用 average denoise step**: 这是个简单但 effective 的选择, 比 SJC-style SDS gradient 或 score averaging 都更直接。

可能的延伸方向 / 我会想的 follow-up:
- **Coarse-to-fine tile pyramid**: 先生成 low-res global layout, 再 refine 每个 tile, 解决跨 tile 结构 (河流、道路) 的一致性问题。
- **Tile shift & scale relaxation**: 论文 Future Work 提到, 允许 tile 不严格对齐 grid, 用 Poisson-like blending。
- **替换 TRELLIS 为 Hunyuan3D 2.0 / TripoSG**: 这些更新的 3D generator 可能 fidelity 更高。
- **加入 video / dynamic**: Genie 2 (论文里 [38] 提到) 是 world video model, 但有 semantic drift; SynCity 是 static 但 consistent。combine 二者做 navigable + dynamic 是 open direction。

## 八、Reference Links

- **TRELLIS** (核心 3D generator): https://github.com/microsoft/TRELLIS
- **Flux**: https://github.com/black-forest-labs/flux
- **Flux ControlNet Inpainting** (AlimamaCreative, 论文 [1]): https://github.com/alimama-creative/FLUX-Controlnet-Inpainting
- **BlockFusion** (baseline): https://github.com/Zenn-Here/BlockFusion
- **3D Gaussian Splatting**: https://github.com/graphdeco-inria/gaussian-splatting
- **DreamFusion**: https://dreamfusion3d.github.io/
- **TRELLIS arXiv**: https://arxiv.org/abs/2412.01506
- **SynCity ethics statement**: https://www.robots.ox.ac.uk/~vedaldi/research/union/ethics.html
- **Objaverse-XL** (TRELLIS training data): https://objaverse.xl
- **World Labs** (referenced 作为 3D bubble SOTA): https://www.worldlabs.ai/
- **rembg** (background removal): https://github.com/danielgatis/rembg
- **LPIPS** (perceptual loss): https://github.com/richzhang/PerceptualSimilarity

这篇 paper 整体给我的感觉是: 一个 "把现有 SOTA model 当 lego 积木拼起来" 的 craft, 工程量很大, 但每个 trick 都对应一个 concrete failure mode, 设计得很 grounded。这种 training-free 路线 vs end-to-end trained (BlockFusion, LT3SD) 的 trade-off, 在 3D world generation 这个领域应该会持续一段时间。
