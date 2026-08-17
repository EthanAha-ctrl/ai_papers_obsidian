---
source_pdf: SeeThrough3DOcclusionAware3DControlinText-to-ImageGeneration.pdf
paper_sha256: 2796b319cc1cbb95d43e2bf94d7ea8438d5e880ac8c52c9215d9d927d02e94d7
processed_at: '2026-08-12T04:33:18-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SeeThrough3D

## 一句话版本

让 FLUX 这种 text-to-image 模型听懂"把 dog 放在 car 后面，camera 从低角度拍"这种 3D 指令，关键 trick 是把 3D box 渲染成**半透明彩色立方体**当 condition 喂进去——半透明让被挡住的物体也能"被看见"，颜色让朝向能被读出来。

---

## 问题到底难在哪

你让 Stable Diffusion / FLUX 画"a dog behind a car"，它经常画成 dog 和 car 并排，或者 dog 骑在 car 顶上。因为 text prompt 是 1D 序列，没告诉模型"behind"具体是哪个 pixel 在哪个 pixel 后面。

之前的人尝试过两条路，都不行：

**路线一：depth map conditioning** (LooseControl [4])
把 3D box layout 投影成一张 depth map $D(u,v)$，每个 pixel 存最近表面的 z 值：
$$D(u,v) = \min_{k} z_k(u,v)$$
问题：被遮挡物体的 z 值**直接被覆盖掉了**。depth map 只有最前面那一层。模型根本不知道后面还有个 box，所以生成时后面物体凭空消失。

**路线二：2D layered** (LaRender [76], VODiff [37])
把场景拆成"dog 是一层，car 是一层"，按顺序叠起来。问题：这是 2D 思维，没有 perspective，"behind" 和 "on top" 分不清，camera viewpoint 更没法表达。

所以根本矛盾是：**你需要让模型"看见"被遮挡的物体，但生成时又要让它真的被遮挡**。这是 amodal completion 问题——人眼天然能做，但 diffusion model 缺个接口。

---

## OSCR 的核心 insight

作者想到一个特别朴素的办法：**把 box 渲染成半透明的**。

$$I(u,v) = \sum_{k} \alpha_k C_k(u,v) \prod_{j<k}(1-\alpha_j)$$

变量含义：
- $k$：沿着 ray $(u,v)$ 遇到的 box surface 编号（从近到远）
- $\alpha_k$：第 $k$ 个 surface 的不透明度（0 到 1 之间，OSCR 用半透明所以 $\alpha \approx 0.5$）
- $C_k(u,v)$：第 $k$ 个 surface 在该 pixel 的颜色
- $\prod_{j<k}(1-\alpha_j)$：前面所有层的透射率，表示光穿过前面层后的剩余强度

这样一来，被遮挡的 box 也能在 image 里"透"出来——颜色被衰减但可见。模型就有了"看穿"的视觉 cue。

但光半透明还不够——你还得告诉模型每个 box 朝哪。作者的另一招是 **face color-coding**：box 的 6 个面各染不同颜色（比如 front=红、back=绿、left=蓝...）。渲染时哪个面朝相机可见，那个颜色就出现在 image 里。模型读 RGB 就能反推出 orientation。

一张 RGB image 同时 encode 了：
- N 个物体的 3D 位置（box 画在哪）
- 每个物体的 3D 朝向（可见面的颜色组合）
- 物体间的遮挡关系（半透明透出的颜色）
- Camera viewpoint（渲染的 perspective 本身）

这是 OSCR (Occlusion-Aware 3D Scene Representation) 的全部精髓。没有什么黑科技，就是 graphics 101 + 一点巧思。

Project page: https://seethrough3d.github.io

---

## 架构上怎么把它接进 FLUX

FLUX [35] 是 mmDiT 架构，text tokens 和 image tokens 在一个 transformer 里互相 attend。作者把 OSCR 当作"另一种 image"塞进去：

```
token sequence = [z_OSCR ; p_text ; x_t]
```

- $z_\text{OSCR} = \text{VAE}(r)$，把渲染的 OSCR 图压成 latent tokens
- $p_\text{text}$：T5 编码的 prompt
- $x_t$：noisy image latent

**关键细节 1：位置编码共享**
$z_\text{OSCR}$ 用和 $x_t$ 一样的 2D positional encoding。意思是 OSCR 在 $(i,j)$ 位置的 token 直接告诉 image 在 $(i,j)$ 位置该长什么样——spatial correspondence 是天然的。

**关键细节 2：LoRA scale 0 trick**
只在 OSCR tokens 走的 Q/K/V projection 上加 LoRA (rank 128)，text 和 image tokens 的 LoRA scale 设为 0——等价于它们走原始 frozen weight，完全不受训练影响。这样 FLUX 原本的 prior（文字渲染、透明物体、人物互动）一点不退化。

$$W_Q' = W_Q + \Delta W_Q, \quad \Delta W_Q = A B^\top, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$$
- $W_Q$：原始 query projection，frozen
- $\Delta W_Q$：低秩更新，$r=128$
- 对 text/image tokens：实际用 $W_Q$（scale=0）
- 对 OSCR tokens：用 $W_Q + \Delta W_Q$

这个 trick 来自 OminiControl [61] 和 EasyControl [79]，是 DiT 时代 control 的新范式，比 ControlNet 优雅得多。

**关键细节 3：attention 拓扑约束**
- 允许 $x_t \to z_\text{OSCR}$：image 在生成时查询 layout
- 允许 $z_\text{OSCR} \leftrightarrow p_\text{text}$：OSCR 吸收 text 语义（binding 用）
- **Block $z_\text{OSCR} \to x_t$**：防止 condition 反过来"看" image，避免 condition leakage

---

## Object binding：让 box 和 text 对上号

OSCR 只告诉你"这有个 box"，不告诉你"这是 dog 还是 car"。如果不管这个，模型可能把 dog 画进 car 的 box 里。

naive 解法是给 box 染色编码类别（红色=dog，蓝色=car），但这会限制到固定 category set，丧失 generalization。

作者的解法用 attention mask：
1. 在 Blender 里渲染每个 box 的 **amodal segmentation mask** $s_i \in \{0,1\}^{H \times W}$（注意是 amodal——包括被遮挡部分的整体轮廓）
2. 解析 text prompt 找到每个 object 的 noun tokens $\mathbf{p}_i$（"a dog" 中的 "dog"）
3. 构造 mask $M$：OSCR token $z_j$ 如果落在 $s_i$ 内，它只能 attend $\mathbf{p}_i$ 的 tokens

数学上：
$$\text{Attn}(z_j) = \sum_k \frac{M_{j,k} \exp(\text{score}_{j,k})}{\sum_{k'} M_{j,k'} \exp(\text{score}_{j,k'})} V_k$$

变量：
- $z_j$：第 $j$ 个 OSCR token
- $M_{j,k}$：mask，1 表示 $z_j$ 允许 attend 第 $k$ 个 text token
- $\text{score}_{j,k}$：standard attention score
- $V_k$：第 $k$ 个 text token 的 value

这样 OSCR tokens 在 dog box 区域只读 "dog" 的语义，把语义注入到几何位置上。

---

## 最反直觉的发现：重叠区不 mix

当两个 box 在 image 上重叠时，intersection 区的 OSCR tokens 同时 attend 两个 object 的 noun tokens（Fig 5b 绿区）。直觉上应该 mix 出"carg"或"dogn"这种怪物。

但实验发现：**生成结果在 intersection 处有 sharp occlusion boundary，没有 attribute mixing**。

作者 visualize 了 image tokens 对 object tokens 的 attention（Fig 6c, d）：
- bicycle 轮辐之间的空隙，attention to "van" 仍然清晰可见
- 这意味着 FLUX 内部，object-specific features 本来就是 disentangled 的
- OSCR 只是激活"哪个位置有哪个 object"，FLUX 的 3D prior 自动把两者摆到正确深度

进一步分析（Appendix D, Fig 18）：spatial alignment 集中在 DiT 的 layer 8-25，且在 denoising step 5/25 才 emerge。早期 layer + 早期 step 是"layout planning"，后期是"rendering"。

**这点的哲学意义**：diffusion model 早就会画 occlusion，OSCR 只是个"接口"把它释放出来。这点和 El Banani 的 "Probing 3D awareness" [18]、Bhattad 的 "Generative models know things" [17] 是一条线的——foundation model 内部已经 3D-aware，缺的是暴露出来的 interface。

Probing 3D awareness: https://arxiv.org/abs/2404.06002
Generative models know things: https://arxiv.org/abs/2311.17137

---

## 数据：用 FLUX 给 SeeThrough3D 造数据

完全靠合成数据，但做得非常聪明：

**Step 1：Blender 渲染**
- 39 个 3D assets，手动对齐 canonical front 到 +Y
- 用 Gemini 2.5 Pro [11] 获取真实物体相对尺寸（jeep < elephant 这种 prior）
- 在 hemisphere 内随机放物体 + camera
- 渲染 paired (image, OSCR)

**Step 2：Visibility filtering**
定义 visibility ratio $x_i = v_i / a_i$：
- $v_i$：object $i$ visible pixel area
- $a_i$：object $i$ amodal pixel area
- 丢弃所有 $x_i > 0.7$ 的场景（occlusion 不够强）
- 丢弃任何 $x_i < 0.3$ 的场景（物体几乎看不见）

**Step 3：FLUX 自举增强**
Blender 渲染图背景太单调，过拟合风险高。但造 3D 场景又贵。所以：
1. Blender 渲染图 → Depth Anything [72] → depth map
2. depth map + diverse background prompt → FLUX.1-Depth-dev → realistic image
3. CLIP filtering：对每个 object crop 算 CLIP similarity，<0.25 丢掉
4. 保留 layout 一致 + appearance diverse 的 augmented image

最终 25K rendered + 25K augmented = 50K paired samples。

这是个 closed loop——用 FLUX 的能力给 FLUX+SeeThrough3D 造训练数据。漂亮的 bootstrapping。

**数据集统计**（Fig 14）：
- Visibility ratio 偏 low（filtering 生效，偏向 heavy occlusion）
- Orientation 均匀分布（无 bias）
- 2D bbox size 偏小（小物体能容多 object，强 occlusion）
- Camera elevation 偏低（低视角 → 强 occlusion）

---

## 训练 setup 速览

| 参数 | 值 |
|---|---|
| Base | FLUX.1-dev |
| LoRA rank | 128 |
| LoRA target | Q, K, V (all attention layers) |
| LoRA scale (text/image) | 0 |
| Optimizer | AdamW |
| LR | $10^{-4}$ |
| Steps | 30K |
| Batch size | 2 |
| Resolution | 25K @ 512², 5K @ 1024² |
| GPU | 2 × H100 |
| Time | ~9 hours |

Staged resolution 是个小 trick：低分辨率先学 macro layout，高分辨率再 polish 细节。

---

## 实验结果

### 主表（Table 1）

| Method | depth ord.↑ | obj.↑ | ang.↓ | text↑ | KID(×10⁻³)↓ |
|---|---|---|---|---|---|
| VODiff | 0.68 | 19.70 | 92.73 | 29.51 | 15.40 |
| LooseControl | 0.82 | 20.02 | 89.88 | 28.43 | 14.32 |
| Build-A-Scene | 0.89 | 21.0 | 91.62 | 28.05 | 20.12 |
| LaRender | 1.02 | 21.83 | 89.63 | 30.20 | 13.46 |
| **Ours** | **1.46** | **22.86** | **47.92** | **31.87** | **5.43** |

几个 takeaway：

1. **depth ord 翻倍**（1.46 vs 1.02）：半透明设计直接 translate 到遮挡顺序准确度。
2. **angular err 减半**（47.92 vs 89.88）：color-coding 让 orientation 在 image space 可读。
3. **KID 暴跌**（5.43 vs 13.46）：LoRA scale 0 保留 FLUX prior，没有 inversion artifact。
4. **Build-A-Scene KID 最差**（20.12）：inversion artifact + sequential generation 导致 scene 不 coherent。

### Ablation（Table 2）

| 去掉啥 | depth ord. | obj. | ang. | KID |
|---|---|---|---|---|
| w/o transparency | 1.20 | 21.67 | 46.15 | 5.90 |
| w/o color-coding | 1.36 | 22.23 | 88.77 | 5.93 |
| w/o binding | 0.98 | 20.45 | 57.44 | 6.35 |
| w/o hard data | 1.24 | 21.89 | 49.73 | 6.34 |
| Full | **1.46** | **22.86** | **47.92** | **5.43** |

读这张表的方式：
- **Translucency 是 occlusion 的核心**（去掉 depth ord 跌 17.8%）
- **Color-coding 是 orientation 的核心**（去掉 angular err 翻倍）
- **Binding 是 layout adherence 的核心**（去掉 depth ord 跌到最低 0.98）
- **Hard data filter 提升 occlusion**（去掉 depth ord 跌到 1.24）

**反直觉的一点**：去掉 transparency 后 angular err 反而**略微变好**（46.15 < 47.92）。原因：半透明会让 face color 被后层 box 干扰，opaque 时 color 更纯。这是 expressiveness vs signal clarity 的 trade-off。

---

## 个人评点

### 1. OSCR 是"用对的方式让 model 用对的能力"

整个 representation 没有黑科技——半透明、color-coded face、Blender 渲染，都是 graphics 101。但组合起来精准 hit 了 diffusion model 的 inductive bias：spatial correspondence via positional encoding，semantic via attention，occlusion via FLUX prior。

### 2. 与 ControlNet 路线的对比

ControlNet [77] 是加一个 condition encoder + zero conv，每个 resolution 都对齐——重。SeeThrough3D 走 token concat + LoRA scale 0，轻量且 prior-preserving。这是 DiT 时代 control 的新范式。

### 3. "Diffusion already knows occlusion" 这点的深意

重叠区不 mix 这个发现，本质是说 FLUX 的 latent space 里 dog 和 car 的 feature 是分开存的，attention 只是"激活开关"。OSCR 在重叠区同时 attend 两个 object，相当于同时打开两个开关，FLUX 的 3D prior 自动把两者摆到正确深度。

这点其实非常 deep——它意味着 **control method 不需要重新教 model 怎么渲染 occlusion，只需要告诉 model 哪里有什么**。和传统 graphics pipeline 思路完全不同。

### 4. 数据工程的 closed loop

用 FLUX.1-Depth-dev 给 SeeThrough3D 造数据，而 SeeThrough3D 又是 FLUX + LoRA。这是 self-bootstrapping 的漂亮范例。CLIP filter 0.25 threshold 看似不高，但对 partial occluded crop 是合理 trade-off。

### 5. 局限

- **继承 FLUX 失败模式**：FLUX 画不出 parrot behind cage，SeeThrough3D 也画不出（Fig 25）
- **Box 限制**：所有 object 都是 axis-aligned cuboid，长颈鹿脖子这种非 cuboid 形状无法表达
- **无物理**：cat on table 是漂浮的，没有 support/collision reasoning
- **Layout edit consistency**：改 layout 后 image 不一致，需要走 editing 路线
- **Multi-subject personalization VRAM 爆炸**

### 6. 可以延伸的方向

- Box → primitive set (cuboid + cylinder + sphere)
- Static camera → camera trajectory (video generation)
- 加 env map / lighting control
- 显式 pose control（不只是 canonical pose，靠 FLUX 泛化）

---

## 核心哲学

这篇 paper 的本质贡献**不是"让 diffusion 学会 occlusion"，而是"reveal diffusion already knows occlusion"**。

OSCR 是个翻译器，把 user 的 3D intent 翻译成 FLUX 能读懂的 visual signal。然后 FLUX 内部的 3D prior 自动完成 occlusion rendering。

这点呼应了近期一批工作：foundation model 内部的 latent space 是 3D-aware 的，只是缺少合适的"接口"暴露出来。SeeThrough3D 提供了这个 interface。这个哲学比 paper 本身的工程细节更值得记住。

---

## 关键 reference

- **SeeThrough3D project**: https://seethrough3d.github.io
- **FLUX**: https://arxiv.org/abs/2506.15742
- **LoRA**: https://arxiv.org/abs/2106.09685
- **OminiControl**: https://arxiv.org/abs/2411.15098
- **EasyControl**: https://arxiv.org/abs/2503.07027
- **LooseControl**: https://arxiv.org/abs/2405.12714
- **Build-A-Scene**: https://arxiv.org/abs/2408.14819
- **LaRender**: https://arxiv.org/abs/2504.12408
- **VODiff**: https://arxiv.org/abs/2410.08122
- **Compass Control**: https://arxiv.org/abs/2502.18480
- **ORIGEN**: https://arxiv.org/abs/2503.22194
- **OrientAnything**: https://arxiv.org/abs/2412.18605
- **Depth Anything**: https://arxiv.org/abs/2401.10891
- **SAM**: https://arxiv.org/abs/2304.02643
- **CLIP**: https://arxiv.org/abs/2103.00020
- **DiT**: https://arxiv.org/abs/2212.09748
- **ControlNet**: https://arxiv.org/abs/2302.05543
- **Probing 3D awareness**: https://arxiv.org/abs/2404.06002
- **Generative models know things**: https://arxiv.org/abs/2311.17137
- **Objaverse**: https://arxiv.org/abs/2212.08051

---

# SeeThrough3D 深度技术解析

## 1. 问题定位：为什么 occlusion reasoning 被忽略

这篇 paper 抓住了一个被 text-to-image 社区长期回避的痛点——**3D layout conditioned generation 中的 occlusion 建模**。我先把现有方法的根本缺陷拆开看：

**Depth-map conditioning 路线** (LooseControl [4], Build-A-Scene [19], CinemaMaster [65]):
- 把 3D bounding box layout 投影成 depth map $D \in \mathbb{R}^{H \times W}$
- $D(u,v) = z\text{-coordinate of nearest box surface at pixel } (u,v)$
- 致命缺陷：被遮挡物体的 depth 信息**完全丢失**——因为 depth map 只保留 front-most surface。这导致模型只能"看见"前面那个物体，遮挡物体在生成时凭空消失或位置错乱（Fig 3a 的红虚框）。

**2D layered decomposition 路线** (LaRender [76], VODiff [37]):
- 把场景拆成 stack of 2D object layers $\{L_1, L_2, ..., L_n\}$，按 visibility 排序
- 用 2D ordering 控制遮挡顺序
- 缺陷：把 3D 结构塌缩成平面，camera viewpoint 无法表达，perspective 错乱（Fig 3b）

**作者 insight**: occlusion 的本质是"被遮挡区域需要被模型'看见'但生成时被遮挡"。这是一种 amodal completion 的问题——人类视觉系统天然能补全被遮挡部分。所以 OSCR 的核心 trick 是用**半透明 box** 让被遮挡区域在 condition image 里仍然可见，给模型一个"看穿"的接口。

Project page: https://seethrough3d.github.io
FLUX 论文: https://arxiv.org/abs/2506.15742

---

## 2. OSCR (Occlusion-Aware 3D Scene Representation)

### 2.1 表示设计

输入：一组 3D bounding boxes $\{b_i\}_{i=1}^{N}$，每个 $b_i$ 由 中心位置 $c_i \in \mathbb{R}^3$、尺寸 $s_i \in \mathbb{R}^3$、朝向 $R_i \in SO(3)$ 定义。Camera 参数 $\mathcal{C} = (K, T)$（内参 + 外参）。

OSCR 渲染管线包含两个关键设计：

**1. Face color-coding (orientation encoding)**
- 每个 box 有 6 个 face，定义 canonical mapping：front=红, back=绿, left=蓝, right=黄, top=白, bottom=黑（具体颜色见 Fig 2b）
- 给定 box 朝向 $R_i$，每个 face 的法向量 $n_f$ 经旋转后为 $R_i \cdot n_f^{canonical}$
- 渲染时根据可见 face 的颜色，模型可以读出 orientation
- 这是一种把 3D rotation 编码到 RGB image space 的方法，类似 Cubemap 但更轻量

**2. Translucency (occlusion encoding)**
- box 表面 alpha 设为 $\alpha < 1$（论文用半透明，具体值未给，但直觉上 $\alpha \approx 0.5$）
- 渲染方程（简化形式）：
$$I(u,v) = \sum_{k} T_k \cdot \alpha_k \cdot C_k(u,v) \cdot \prod_{j<k}(1-\alpha_j)$$
- 其中 $k$ 索引沿 ray $(u,v)$ 的 box surface 交点，$T_k$ 是 transmission，$\alpha_k$ 是 surface opacity，$C_k$ 是 face color
- 关键性质：被遮挡的 box 面仍能贡献颜色（虽然被 front box 的透明度衰减），让模型"看到"被遮挡物体的存在与朝向

**3. Camera embedding**
- 通过从指定 $\mathcal{C}$ 渲染，camera intrinsics/extrinsics 隐式编码在 rendered image 的 perspective 中
- 不需要单独的 camera token，渲染本身就是 camera pose 的"投影"

这个设计的 elegant 之处：用一张 RGB image 就 encode 了 (a) N 个物体的 3D 位置 (b) 3D 朝向 (c) 相对深度遮挡关系 (d) camera viewpoint——四种信息压进单一 2D signal。

---

## 3. SeeThrough3D 架构

### 3.1 Token 序列构造

基于 FLUX.1-dev [35]，它是个 mmDiT (multimodal Diffusion Transformer) 架构。

输入 token 序列构造：
```
[z_OSCR ; p_text ; x_t_noisy]
```
- $z_\text{OSCR} = \text{VAE\_encode}(r) \in \mathbb{R}^{L_z \times d}$，其中 $L_z = (H/8) \times (W/8)$ 是 VAE 降采样后的 token 数，$d$ 是 hidden dim
- $p_\text{text}$ 来自 T5 text encoder
- $x_t$ 是 noisy latent

关键设计：**z_OSCR 沿用 x_t 的位置编码**。这意味着 OSCR tokens 与 image tokens 在 2D 网格上位置一一对应。这是 spatial correspondence 的基础——OSCR 在 $(i,j)$ 位置的 token 直接告诉 image 在 $(i,j)$ 位置该生成什么 box 信息。

### 3.2 Attention 拓扑

mmDiT block 内部，三类 tokens 通过 self-attention 交互。作者对 attention 做了**非对称约束**：

```
       attend from →  z_OSCR   p_text   x_t
z_OSCR      ✓(self)    ✓       ✗ (blocked)
p_text      ✓          ✓       ✓
x_t         ✓          ✓       ✓
```

- **Block z→x_t 的 attention**：防止 OSCR condition tokens 反过来"读" image latent，避免 condition leakage 污染生成过程。这是 EasyControl [79] 等工作的经验。
- **允许 x_t→z_OSCR**：image tokens 在生成时查询 OSCR，获取 layout 信息
- **允许 z_OSCR↔p_text**：让 OSCR tokens 可以吸收 text semantics（这就是 binding 的基础）

### 3.3 LoRA 微调策略

为保留 FLUX 的 text-to-image prior，作者**冻结所有原始 weights**，只对新加 token 相关的 projection 加 LoRA：

$$W_Q' = W_Q + \Delta W_Q, \quad \Delta W_Q = A \cdot B^T, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$$

- rank $r = 128$（较高，说明需要 expressiveness）
- 只在 query/key/value projection 上加 LoRA
- **LoRA scale 设为 0 对 text/image tokens**：这等价于"这些 token 走原始 W，不受 LoRA 干扰"。只有新加的 OSCR tokens 走 LoRA-modified projection
- 这个 trick 来自 OminiControl [61, 62] 系列

我直觉上理解：LoRA scale 0 on existing tokens 等价于把 LoRA 限制在 "新 token 走的新通路"，原始 text-image 通路完全保留，所以 FLUX 的强大 prior（text rendering, transparent objects, interactions）不退化——这一点在 qualitative results 里能看到（Fig 8 A,B,G,J 的透明物体，G 的文字 CVPR）。

LoRA 论文: https://arxiv.org/abs/2106.09685
OminiControl: https://arxiv.org/abs/2411.15098

---

## 4. Object Binding via Attention Masking

### 4.1 问题：semantic-geometric 解耦

OSCR 只 encode 几何，不 encode "box 1 是 car，box 2 是 dog"。如果纯 spatial conditioning，模型可能把 car 和 dog 放错 box。最 naive 的解法是在 box 里染色编码类别，但这会限制到固定 category、丧失 generalization。

作者的解法：用 text prompt 中的 object noun tokens 作为语义源，通过 attention mask 把 OSCR tokens 与对应 noun tokens 绑定。

### 4.2 Mask 构造

对每个 box $b_i$：
1. 在 Blender 里渲染其 **amodal segmentation mask** $s_i \in \{0,1\}^{H \times W}$（注意是 amodal——包括被遮挡部分的整体轮廓，而不是 visible-only mask）
2. 对应 text prompt 中第 $i$ 个 object 的 noun tokens 集合 $\mathbf{p}_i$

构造 attention mask $M \in \{0,1\}^{|z| \times |p|}$：
$$M_{j, k} = \begin{cases} 1 & \text{if OSCR token } z_j \text{ falls in } s_i \text{ and token } p_k \in \mathbf{p}_i \\ 0 & \text{otherwise} \end{cases}$$

在 mmDiT 的 attention 计算中：
$$\text{Attn}(z_j) = \sum_{k} \frac{M_{j,k} \cdot \exp(\text{score}_{j,k})}{\sum_{k'} M_{j,k'} \cdot \exp(\text{score}_{j,k'})} V_k$$

这样 OSCR tokens 在 box $i$ 的区域只读 $\mathbf{p}_i$ 的语义，把 "dog"、"car" 等语义信息注入到该区域的 OSCR tokens。

### 4.3 重叠区域的反直觉发现

当两个 box 的 rendered region 重叠时，intersection 区域的 OSCR tokens 同时 attend 两个 object 的 noun tokens（Fig 5b 绿色区）。直觉上这会导致 semantic blending——比如生成 "carg" 或 "dogn"。

但实验发现：**生成结果在 intersection 处有 sharp occlusion boundary，没有 attribute mixing**（Fig 6b）。

作者 visualize 了 image tokens $x_t$ 对 object tokens 的 attention map（Fig 6c, d）：
- 在 bicycle 的"空隙"区域（轮辐之间），attention to "van" 仍然可见
- 这说明 image latent 内部，object-specific features 是 **disentangled** 的，FLUX 的 prior 本身就能分离两个 object 的 semantic content
- OSCR 只是提供了"哪个位置属于哪个 object"的 spatial cue，模型用 prior 把对应 object 渲染过去，occlusion 自然产生

这个发现挺重要的，呼应了 El Banani et al. [18] 的 "Probing 3D awareness of foundation models"——diffusion model 内部已经 encode 了 3D priors，OSCR 只是把它们 unlock 出来。

更细的 attention 分析（Appendix D）：在 DiT 的 layer 11-23 出现强 spatial alignment，且大约第 5 个 denoising step（共 25 步）spatial structure 才 emerge。这说明 early layers + early timesteps 是"规划阶段"，late layers 是"rendering 阶段"。

Probing 3D awareness: https://arxiv.org/abs/2404.06002

---

## 5. Personalization 扩展

给定 reference object image $v$，目标：生成时让某个 box $b_i$ 渲染成 $v$ 中的物体。

机制：
1. $v$ 通过 VAE encoder → appearance tokens $\mathbf{v} \in \mathbb{R}^{L_v \times d}$
2. 拼接进 token 序列：$[z; p; v; x_t]$
3. 另起一个 **subject LoRA**（rank 128），加在 reference image tokens 的 projection 上
4. Attention mask：让 $b_i$ 内的 OSCR tokens 同时 attend $\mathbf{p}_i$ 和 $\mathbf{v}$
5. 训练 7.5K iterations

数据准备（Appendix I）：
- 从已有 rendered image 中随机选个 object
- 在 Blender 里给它套一个 FLUX 生成的 texture
- 单独渲染 textured object → reference image
- 把 textured object 放回原 scene → target image
- reference 的 orientation 略微 perturbed，强迫模型学 3D placement 而不是 pixel copy

直觉：这等价于 subject-driven generation（IP-Adapter 路线）+ 3D layout control 的并集。不是 test-time tuning，是离线训练一个通用 subject LoRA。

---

## 6. Dataset 构造细节

### 6.1 Procedural generation

- 39 个 assets from Objaverse [15] + SketchFab [59]
- 手动对齐 canonical front 到 +Y axis
- 用 Gemini 2.5 Pro [11] 获取真实物体相对尺寸（jeep < elephant 这种 prior knowledge）
- Object 放置区域：以 origin 为中心，半径 $R$ 的 hemisphere 上 + 内部
- Camera：在 hemisphere 表面，always look at origin

### 6.2 Filtering logic

**Visibility ratio**：$x_i = v_i / a_i$
- $v_i$：object $i$ visible pixel area（通过 rendered visible mask）
- $a_i$：object $i$ amodal pixel area
- Filter out if 所有 $x_i > 0.7$（occlusion 不够强）
- Filter out if 任何 $x_i < 0.3$（object 几乎看不见）

**Object 2D size**：
- 2D bbox 最大边长 ∈ $[0.125, 0.75]$ × image size
- 避免 object 太小（生成不出细节）或太大（占满画面无 occlusion）

### 6.3 Augmentation pipeline

纯渲染图背景单一，过拟合风险高。作者用一个 scalable augmentation：

1. Rendered image → Depth Anything [72] → depth map
2. Depth map + diverse background prompt → FLUX.1-Depth-dev → realistic image
3. CLIP filtering：对每个 object 的 crop region 计算 CLIP similarity with 对应 text description
4. 任何 object CLIP score < 0.25 → 整张图丢弃
5. 这样保证 augmented image 的 layout 与原 rendered image 一致

最终数据集：25K rendered + 25K augmented = 50K paired samples。

**数据集统计**（Fig 14）：
- (a) min visibility ratio 分布偏向 low（说明 filtering 有效，偏向 heavy occlusion）
- (b) orientation 均匀分布
- (c) 2D bbox size 偏小（小物体能容纳多 object，强 occlusion）
- (d) camera elevation 偏低（低视角 → 强 occlusion，高视角 bird's-eye occlusion 弱）

直觉：这个 filtering 策略与 ablation 中 "w/o hard data" 对应——去掉 hard data 后 depth ord 从 1.46 降到 1.24，证明 occlusion-heavy data 是模型学会遮挡推理的关键。

Depth Anything: https://arxiv.org/abs/2401.10891

---

## 7. Training 实现细节

| 参数 | 值 |
|---|---|
| Base model | FLUX.1-dev |
| LoRA rank | 128 |
| LoRA target | Q, K, V projections (every attention layer) |
| LoRA scale (text/image tokens) | 0 |
| Optimizer | AdamW |
| Learning rate | $10^{-4}$ |
| Total steps | 30K |
| Effective batch size | 2 |
| Resolution schedule | 25K @ 512², 5K @ 1024² |
| GPU | 2 × NVIDIA H100 |
| Training time | ~9 hours |
| Framework | PyTorch + HuggingFace Diffusers |

**Staged resolution training**：先 512 训练 25K 步，再切到 1024 训 5K 步。这个 trick 我理解是为了：
- 早期低分辨率：模型先学会 layout adherence 和 occlusion reasoning 这种"宏观结构"
- 后期高分辨率：fine-tune 出 image realism（细节、纹理）
- 类似 progressive training，避免高分辨率下 layout 学习困难

---

## 8. 3DOc-Bench 评测基准

500 paired samples，要求：① 多样 object 配置 ② 挑战性 occlusion ③ 多 camera viewpoint。

构造方法同训练集，但用 held-out configurations。

### 评测 metrics 设计（这是难点）

**1. 2D layout adherence (objectness score)**
- 用 SAM [30] 在 generated image 上获取 object masks
- 把 2D layout 与 SAM masks 结合，crop 出每个 object 区域
- 计算 CLIP similarity between crop and textual object description
- Aggregate 成 objectness score
- 直觉：mask 内的内容是否与 description 匹配

**2. Relative depth ordering (depth ord.)**
- 用 Depth Anything [72] 估计 per-pixel depth
- 对每个 object mask 区域平均 depth
- 对所有 object pair 比较 relative order，与 GT ordering 对比
- 正确 pair 数 / 总 pair 数 = score
- 用 objectness score > threshold 来 filter，避免缺失 object 干扰

**3. 3D orientation consistency (angular error)**
- 用 OrientAnything [67] 估计 generated object 的 orientation
- 与 GT 3D box orientation 算 angular distance
- 简化版：还提供 "180° flip free" 版本（因为 depth map 方法学不到 front/back 区分，flip 是固有缺陷）

**4. Text-image alignment**: 标准 CLIP score

**5. Image quality**: KID [5, 6]

---

## 9. 实验数据深度解读

### 9.1 主表（Table 1）

| Method | depth ord.↑ | obj.↑ | ang. err.↓ | text↑ | KID(×10⁻³)↓ |
|---|---|---|---|---|---|
| VODiff | 0.68 | 19.70 | 92.73 | 29.51 | 15.40 |
| LooseControl | 0.82 | 20.02 | 89.88 | 28.43 | 14.32 |
| Build-A-Scene | 0.89 | 21.0 | 91.62 | 28.05 | 20.12 |
| LaRender | 1.02 | 21.83 | 89.63 | 30.20 | 13.46 |
| **Ours** | **1.46** | **22.86** | **47.92** | **31.87** | **5.43** |

**关键 observations**:

1. **depth ord 翻倍**（1.46 vs 第二名 1.02）：这是 occlusion reasoning 的核心 metric，OSCR 的半透明设计直接 translate 到遮挡顺序准确度。

2. **angular err 减半**（47.92 vs 第二名 CompassControl 66.29 in Appendix Table 3）：color-coding of box faces 让 orientation 在 image space 直接可读，远超 adapter-based 方法。

3. **KID 暴跌**（5.43 vs 13.46）：这是 image quality 的大幅领先。我推测原因是：
   - LoRA scale 0 保留 FLUX prior
   - 不破坏 text-image alignment
   - 不需要 inversion (Build-A-Scene 的 inversion artifact 导致 KID 20.12)
   - 不需要 multi-step sequential generation

4. **VODiff 表现最差**：2D layered decomposition 在 3D 任务上根本不对路。

### 9.2 Baseline 失败模式分析

**LooseControl**：
- depth map 丢失被遮挡物体 → 生成不出 occluded object (Fig 9 A1, A3-5)
- 没有 binding → object 放错位置 (Fig 9 A1, A3)
- depth map 无法区分 front/back → 180° flip 频发（Appendix Table 3 显示 37.48° vs 我们 25.72° 的 flip-free error）

**Build-A-Scene**：
- 多次 generation-inversion 循环 → inversion artifacts (Fig 9 B2-3, B5)
- 顺序生成 → scene 不 coherent (B4)，因为早期生成不知道后期 object
- KID 20.12 最差

**LaRender / VODiff**：
- 2D layer → 缺乏 perspective
- "behind chair" 经常被生成成 "on top of chair" (Fig 9 C4, D4-5)
- 2D bbox 大重叠时直接生成不出 occluded object

---

## 10. Ablations 深度分析（Table 2）

| Ablation | depth ord.↑ | obj.↑ | ang.↓ | text↑ | KID↓ |
|---|---|---|---|---|---|
| w/o transparency | 1.20 | 21.67 | 46.15 | 31.39 | 5.90 |
| w/o color-coding | 1.36 | 22.23 | 88.77 | 31.57 | 5.93 |
| w/o binding | 0.98 | 20.45 | 57.44 | 31.61 | 6.35 |
| w/o hard data | 1.24 | 21.89 | 49.73 | 31.32 | 6.34 |
| Full | **1.46** | **22.86** | **47.92** | **31.87** | **5.43** |

**Intuition 提炼**：

1. **Translucency 是 occlusion 的核心**：去掉后 depth ord 从 1.46 → 1.20（-17.8%）。这证明"让被遮挡区域可见"的设计直接驱动了遮挡推理能力。

2. **Color-coding 是 orientation 的核心**：去掉后 angular err 从 47.92 → 88.77（+85%），几乎完全失效。这是 OSCR 设计中"face color = orientation encoding"的直接证据。

3. **Binding 是 layout adherence 的核心**：去掉后 depth ord 0.98（最低！），objectness 20.45（最低）。没有 binding，模型不知道哪个 box 放哪个 object，layout 散架。

4. **Hard data filtering 提升 occlusion**：去掉后 depth ord 1.24，证明 occlusion-heavy 的过滤策略有效。

**反直觉发现**：去掉 transparency 后 angular err 反而**略微变好**（46.15 vs 47.92）。作者解释："opaque boxes 提供更清晰的 color signal"——半透明会让 face color 被后层 box 干扰，opaque 时 color 更纯。这是 expressiveness vs signal clarity 的 trade-off。

---

## 11. Attention 可视化分析（Appendix D, Fig 17-18）

### 11.1 Layer/timestep spatial alignment 热图（Fig 18）

用 correlation coefficient (CC) 测量 image→object attention 与 GT segmentation 的空间对齐：
- **Spatially aware layers**: 大约 layer 8-25（早期 layer）
- **Spatial awareness emerge time**: 大约 denoising step 5/25

直觉：这与 Stable Flow [1] 的发现一致——DiT 不同层承担不同角色。早期层是 "layout planning"，后期层是 "appearance rendering"。Spatial structure 在前 1/5 denoising 步内就被确定，之后只是 refine 细节。

### 11.2 透明物体下的 attention（Fig 17 b, g）

对于 transparent object（如水杯、化学烧瓶），背后的 object（sparrow、另一个 flask）在 attention map 中**仍然可见**。这表明：
- Image latent 中，物体特征是 disentangled 的
- 透明物体的"看穿"在 latent space 中通过 attention 实现，与物理透明性同构
- 这进一步证明 FLUX prior 中已经 encode 了 occlusion reasoning 能力

---

## 12. User Study（Fig 10）

60 participants, A/B test vs random baseline：
- Image realism: ~76% prefer ours
- Layout adherence: ~82% prefer ours
- Text prompt alignment: ~70% prefer ours

全面碾压。

---

## 13. 局限性（Appendix M）

1. **继承 FLUX 的失败模式**：FLUX 无法生成的（如 parrot behind cage），SeeThrough3D 也不行（Fig 25）。这是 base model prior 的天花板。
2. **Personalization VRAM**：所有 reference image tokens 要驻留 context，multi-subject 时 VRAM 爆炸。
3. **Layout edit consistency**：作者承认修改 layout 后图像不能保持一致性（同 object 在不同 layout 下生成的 appearance 会变）。这是 editing 路线（而非 regen 路线）的工作。

---

## 14. 我的 intuition 评点

### 14.1 OSCR 设计的"美学"

这个 representation 的美在于 **用一个 RGB image encode 了 4D information** (3D layout + camera)。它避开了：
- NeRF/Gaussian Splat 那种 implicit 3D 表示（难以塞进 2D diffusion）
- 3D parameter tokenization（丢失 spatial inductive bias）
- 2D depth map（丢失 occlusion）

半透明 + color-coded face 这种"图形学 trick" + "diffusion inductive bias" 的组合，是典型的"用对的方式让 model 用对的能力"——而不是逼 model 学一个全新的能力。

### 14.2 与 ControlNet 路线的对比

传统 ControlNet 路线 [77] 是"加 condition encoder + zero conv"——这要求 condition 与 output 在每个 resolution 都对齐。SeeThrough3D 走的是 **token concatenation + LoRA** 路线，类似 OminiControl [61] / EasyControl [79]。优势：
- 不需要额外 encoder 网络
- 通过 LoRA scale 0 保留 prior
- 自然支持 multi-modal conditioning (text + image + 3D)
- 通过 attention masking 做 fine-grained binding

这是 DiT 时代 control 的新范式——比 ControlNet 优雅很多。

### 14.3 关于"为什么 overlapping 不 mix"的深层原因

这个发现让我想到 Task Arithmetic、model editing 这些工作里的发现：**diffusion model 的内部 feature space 是 linear/disentangled 的**。物体特征在 latent 中本来就是分开存储的，attention 只是激活对应 feature 的"开关"。OSCR 在重叠区域同时 attend 两个 object，相当于同时激活两个 feature cluster，渲染时 model 的 prior 自然把两者摆到正确深度（因为 FLUX 内部的 3D prior 知道 "car 在 dog 前面" 该怎么画）。

这点其实非常重要——它意味着 **control method 不需要重新教 model 怎么渲染 occlusion，只需要告诉 model 哪里有什么**。这与传统 graphics pipeline 思路完全不同。

### 14.4 数据工程的精彩

整个 pipeline：rendered → depth → FLUX depth2image → CLIP filter，是个**自举式**数据增强。它把 FLUX 自己的 capability 用回来生成训练数据给 SeeThrough3D——而 SeeThrough3D 又是 FLUX + LoRA。这是个漂亮的 closed loop。

CLIP filtering threshold 0.25 看起来不高，但考虑到 mask crop 内可能包含 partial occlusion，这个 threshold 是合理的 trade-off (precision vs recall)。

### 14.5 个人觉得可改进的方向

1. **Box-as-condition 的限制**：所有 object 都是 axis-aligned 3D bbox，没法表达非 cuboid 物体（如长颈鹿的脖子）。可以用 primitive set (cuboid + cylinder + sphere)。
2. **Static camera**：可以扩展到 camera trajectory，做 video generation。
3. **No physics**：object 之间无碰撞、无支撑关系的物理推理（如 "cat on table" 是漂浮的）。
4. **Texture/lighting control 缺失**：可以加 env map、light direction 的 control。
5. **Canonical pose bias**：训练数据 object 都是 canonical pose，靠 FLUX prior 泛化到 sitting、cycling 等姿态。如果显式控制姿态会更可控。

### 14.6 与同期工作 ComposeYourScene / SceneDesigner 的对比

SceneDesigner [55] 用 9-DOF pose + adapter，限制于单 domain。SeeThrough3D 用 OSCR 图像 condition + LoRA，自然 multi-domain。这是 "explicit 3D parameter" vs "implicit 3D-as-image" 的路线之争——后者更 scalable。

---

## 15. 总结：这篇 paper 的本质贡献

不是"让 diffusion 学会 occlusion"，而是"**reveal diffusion already knows occlusion**"。OSCR 是个"翻译器"，把 user 的 3D intent 翻译成 FLUX 能读懂的 visual signal，然后 FLUX 内部的 3D prior 自动完成 occlusion rendering。

这点非常 deep——它呼应了 El Banani et al. "Probing 3D awareness" 和 Bhattad et al. "Generative models know things" 的研究脉络：foundation model 内部的 latent space 是 3D-aware 的，只是缺少合适的"接口"暴露出来。SeeThrough3D 提供了这个 interface。

Probing 3D awareness: https://arxiv.org/abs/2404.06002
Generative models know things: https://arxiv.org/abs/2311.17137

---

## 关键 reference 链接

- **SeeThrough3D Project page**: https://seethrough3d.github.io
- **FLUX (base model)**: https://arxiv.org/abs/2506.15742
- **LoRA**: https://arxiv.org/abs/2106.09685
- **OminiControl**: https://arxiv.org/abs/2411.15098
- **OminiControl2**: https://arxiv.org/abs/2503.08280
- **EasyControl**: https://arxiv.org/abs/2503.07027
- **LooseControl**: https://arxiv.org/abs/2405.12714 (SIGGRAPH 2024)
- **Build-A-Scene**: https://arxiv.org/abs/2408.14819
- **LaRender**: https://arxiv.org/abs/2504.12408
- **VODiff**: https://arxiv.org/abs/2410.08122
- **Compass Control**: https://arxiv.org/abs/2502.18480
- **ORIGEN**: https://arxiv.org/abs/2503.22194
- **OrientAnything**: https://arxiv.org/abs/2412.18605
- **Depth Anything**: https://arxiv.org/abs/2401.10891
- **SAM**: https://arxiv.org/abs/2304.02643
- **CLIP**: https://arxiv.org/abs/2103.00020
- **DiT (Peebles & Xie)**: https://arxiv.org/abs/2212.09748
- **ControlNet**: https://arxiv.org/abs/2302.05543
- **Probing 3D awareness of foundation models**: https://arxiv.org/abs/2404.06002
- **Generative models: What do they know?**: https://arxiv.org/abs/2311.17137
- **Objaverse**: https://arxiv.org/abs/2212.08051

整体上，SeeThrough3D 是个"小巧但漂亮"的工作——不堆量、不 brute force，靠一个精巧的 representation (OSCR) + 几个 surgical 的 architecture choices (LoRA scale 0, attention masking) 就把 3D-aware occlusion control 这个难题解了。最值得记住的是 **"diffusion model already knows occlusion, we just need to ask correctly"** 这个哲学。
